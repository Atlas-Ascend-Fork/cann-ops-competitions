# 需求背景（required）

## 需求来源

CANN 社区任务《10月社区任务-aclblasHgemmBatched算子开发（950）》，补齐 ops-blas 仓
Level-3 GEMM 族的半精度批量矩阵乘能力，对标 cuBLAS `cublasHgemmBatched`。

## 背景介绍

### aclblasHgemmBatched 算子功能

半精度实数（FLOAT16）批量通用矩阵乘：对每个 batch i（i ∈ [0, batchCount)）计算

`C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`

- A[i]/B[i]/C[i] 均为 FLOAT16 矩阵，按列主序（Column-Major）存储；
- `Aarray/Barray/Carray` 为 Device 内存中的指针数组，第 i 个元素指向第 i 个 batch 矩阵首地址，前导维分别为 lda/ldb/ldc；
- alpha/beta 为 FLOAT16 标量（`const aclblasHalf*`，Host 内存）；
- `op()` ∈ {N, T, C}，FLOAT16 实数下 T 与 C 等价；
- op=N 时 A[i] 为 m×k、B[i] 为 k×n；op∈{T,C} 时物理存储分别为 k×m、n×k；C[i] 恒为 m×n。

### 现状分析

ops-blas 仓已有 `aclblasSgemmBatched`（FP32）与 `aclblasGemmBatchedEx`（多 dtype 扩展），
`include/cann_ops_blas.h` 尚无 `aclblasHgemmBatched` 声明，也无 `aclblasHalf` 类型定义。
gemm_batched 家族在 arch35 上已具备基于 tensor_api 的流水化 Cube 模板
（`blas/gemm_batched/arch35/gemm_batched_cube_impl.h`：FP16 dtype traits、L1 双 bank、
L0 ping-pong、FP32 L0C 累加），但 FLOAT16 输入时输出端仅有"FP32 临时缓冲 + 第二个
AIV kernel 合成"的通用路径，无 FP16 直写快速路径。

# 需求分析（required）

## 需求描述

在 Ascend 950PR（arch35 / DAV_3510）上以 Ascend C（kernel 直调）实现
`aclblasHgemmBatched`，接口签名与 `cublasHgemmBatched` 参数序列一致，声明放入
`include/cann_ops_blas.h` 与其他产品线共用；实现代码位于 `blas/gemm/arch35/`，
测试代码位于 `test/gemm/hgemmBatched/`。

## 需求拆解

1. 接口与语义：列主序、指针数组、FLOAT16 标量指针、N/T/C 全组合；边界语义
   （m/n/batchCount=0 no-op；k=0 仅 beta 缩放；beta=0 不读 C 旧值）与参数校验
   顺序严格对齐任务书 §2.4/§2.5（非法枚举返回 INVALID_ENUM，判空先于 no-op 短路，
   k=0 时允许 Aarray/Barray 为空、alpha=0 不放宽）。
2. 精度：按生态算子开源精度标准 FLOAT16 档（rtol=atol=2⁻¹⁰、matched_ratio≥0.99、
   max_abs_error≤max(1e-1, 32·ULP)），golden 为 cblas_sgemm（FP32）逐 batch 计算后
   IEEE 754 就近舍入。
3. 性能（Ascend 950PR，msprof op task duration 口径）：200 条性能用例逐条
   倍率（H100 基线/NPU 实测）≥ 0.4，且算术平均 ≥ 0.5。

# 详细设计（required）

## 算子分析

### 数学公式

C[i][r][c] = alpha × Σ_{p=0}^{k-1} op(A[i])[r][p] × op(B[i])[p][c] + beta × C[i][r][c]

### 支持数据类型

FLOAT16（输入/输出/标量）；Cube 内部以 FP32 累加，alpha/beta 合成在 FP32 域完成，
最终一次就近舍入写回 FLOAT16（与 golden 的"FP32 计算 + 一次 RNE 舍入"口径一致，
避免双重舍入）。

### 支持形状

m/n/k/batchCount ≥ 0 的任意组合（运行时入参，无编译期 shape 约束）；
前导维仅要求 ≥ max(1, 物理行数)，不支持超出前导维语义的非连续 stride，不支持广播。

## 算子实现

### 实现方案

总体结构：host 侧一次生成执行计划（纯函数 `HgbMakeKernelPlan`），kernel 侧为
单一模板化 tile 流水（`HgbRunTaskTiles`）。三条执行路径：

1. **直写快速路径（alpha=1 且 beta=0，覆盖全部性能用例）**：专用 Cube kernel，
   L0C（FP32）经 Fixpipe F322F16 一次舍入直写 FLOAT16 C，无中间缓冲、无第二个 kernel；
2. **通用 alpha/beta 路径**：复用 gemm_batched 家族 FP16 Cube kernel 输出 packed FP32
   临时区，FP16 AIV kernel 以 FP32 运算合成 `alpha*AB + beta*C` 后一次舍入写回；
   workspace 不足时按 batch 逐半分片流水执行（同 stream 天然串行，复用同一临时区）；
3. **退化路径（k=0 或 alpha=0）**：beta=1 直接返回；beta=0 走零填充 kernel
   （不读取 C 旧值，杜绝 0×NaN 传播）；其余走 early-exit AIV kernel 做 C=beta*C。

#### host 侧设计

**列主序归一化**：入口即把列主序问题等价映射为 kernel 行主序坐标
（C = op(A)op(B) 列主序 ⟺ Cᵀ = op(B)ᵀ op(A)ᵀ 行主序，即 kernel (m,n)=(用户 n,m)、
lda/ldb 与 transA/transB 互换，发射时 A/B 指针数组互换），其后 tile 选择、
空间切分、容量断言全部在唯一坐标系下进行，无中间可变状态。

**tile 选择（kernel 坐标）**，按 950PR 片上容量（每核 L1 512KB 双 bank、L0A/L0B
64KB 双缓冲、L0C 256KB）推导：

| 形状条件 | tileM×tileN | K 段(kc) | 依据 |
|---|---|---|---|
| 非TT 且 m,n≥512 且 k≥640 且宏块数≥2×核数 | 256×256（单 L0C 缓冲） | 256 | 输入 flop/byte 87→128（+47%），用于 L2 带宽受限的大 K 形状；(256+256)×256×2B=256KB 恰满 L1 ping |
| 双转置（TT）且 m≥256 | 256×128 | 320 | swap 后 A/B 均为 DN 布局，A 的 GM 连续段仅 tileM 元素；竖 tile 使 A 突发由 256B 增至 512B |
| n≥256 | 128×256 | 320 | B 片驻留翻倍、Fixpipe 次数减半；128×256×FP32=128KB 恰满 L0C bank |
| m≥256（窄 N） | 256×128 | 320 | 保持单 tile 32K FP32 累加区 |
| 其余 | 128×128 | 512 | (128+128)×512×2B=256KB 恰满 L1 ping |

kc 再按"等轮数均衡"收缩：加载轮数不变时取最小 64 对齐值（如 k=704:
320+320+64 → 256+256+192），使跨 tile/跨 task 预取获得更长的末段计算窗口。

**空间切分**：在最终 tile 粒度上枚举 (mBlocks, nBlocks)，最小化
`makespan ≈ 波数 × 单任务最大 tile 数`，平手时偏好更少任务、再偏好方形块；
允许切分数超过核数（多波整块切），单任务可小至 1 个 tile——任务边界气泡由
kernel 跨 task 预取消除。实测（8192³ 细切 363 TFLOPS）验证负载均匀性收益
大于 L2 面板复用损失。

**发射前守卫**：`HgbVerifyPlanInvariants` 对 L1 ping/L0A/L0B/L0C 容量做硬断言，
选参越界在发射前精确报错；计划出口打印单行决策摘要
（路径/tile/kc/切分/任务数/核数），用于排障时复现完整调度形态。

#### kernel 侧设计

单一模板 `HgbRunTaskTiles<DtypeId, SINGLE_L0C>` 承载两种 L0C 策略：

- `SINGLE_L0C=false`：L0C 双 bank 交替，Fixpipe 写回与下一 tile 计算重叠（常规 tile）；
- `SINGLE_L0C=true`：整个 L0C 256KB 作为一个 FP32 累加区（256×256 宏块）。

**软件流水**：将本 task 的 (tile, k-chunk) 展平为一条流做深度 1 预取——计算第 i 段
时 MTE2 搬入第 i+1 段，含下一 tile 乃至下一 task 的首段；`l1LoopCnt`/`l0PingPong`
跨 tile、跨 task 连续递增，L1 双 bank 与 L0 ping-pong 持续交替。事件协议复用家族
模板（MTE2_MTE1/MTE1_MTE2 双 bank + FIX_M/M_FIX L0C bank + M_MTE1/MTE1_M L0）。

**任务调度**（`HgbResolveTaskCoordinates`）：batch 主序 + 空间编号陪集轮转 +
大网格 2D 分组：

- 陪集轮转：g = gcd(空间块数 S, 核数 28)。g=1 时不轮转（taskId%S 天然以周期 S
  遍历全部块，盲目加 batch 轮转反而与 28 步进发生数论共振）；g>1 时以
  batch/(28/g) 为轮转量，严格保证每核每 S 波不重不漏遍历全部空间块，消除
  "同核永远领取同种块"的负载锁死（S=2 时最重核负载达均值 1.89×）。
- 2D 分组（mBlocks,nBlocks ≥ 8 时，GROUP_M=4）：一波 28 核聚成 4×7 子矩形，
  单波仅共享 4 条 A 面板 + 7 条 B 面板，跨波 L2 面板流量建模下降 36%~52%。

**边界**（`HgbResolveTaskBoundsTileGrain`）：按最终 tile 粒度比例均分，块边界
天然 tile 对齐，消除 16 粒度切分产生的碎 tile（8192/7 场景总 tile 数 +9.4%）。

**微矩阵大 batch 快路径**（mBlocks=nBlocks=1 时）：任务即 batch，每核改领连续
batch 区间，指针数组从 28×8B 跨步散读变为顺序读，坐标/边界的标量整除全部
外提循环（8³×b1024 设备耗时 16.3→14.4µs）。

**任务描述滚动复用**：上一轮为跨 task 预取解析出的下一任务描述（GM 张量视图、
边界、首 tile 尺寸）直接成为本轮当前任务，每任务仅做一次解析。

#### 精度设计

- Cube mmad FP32 累加（L0C），k 维全程不降精；
- 快速路径由 Fixpipe F322F16 完成唯一一次舍入；通用路径 FP32 临时 + AIV 在
  FP32 域合成后唯一一次 `Cast(CAST_ROUND)`；
- beta=0 一律不读取 C 旧值（快速路径天然不读；退化路径走零填充 kernel），
  杜绝 0×NaN/Inf 传播；alpha=0 不引用 A/B 数据，与 golden 退化口径一致。

### 关键函数

| 函数 | 位置 | 职责 |
|---|---|---|
| `aclblasHgemmBatched` | hgemmBatched_host.cpp | 参数校验、标量解码、三路径分发 |
| `HgbMakeKernelPlan` | 同上 | 列主序归一化 + tile/切分联合选择（纯函数） |
| `HgbVerifyPlanInvariants` | 同上 | 片上容量硬断言 |
| `HgbRunTaskTiles<_,SINGLE_L0C>` | hgemmBatched_kernel.cpp | 统一 tile 流水（双/单 L0C 编译期特化） |
| `HgbResolveTaskCoordinates` | 同上 | 陪集轮转 + 2D 分组任务映射 |
| `hgemm_batched_zero_fill_kernel` | 同上 | beta=0 零填充（AIV） |

复用自 gemm_batched 家族（经 `gemm_batched_kernel.h` 公共声明）：
`GbLoadL1Chunk`/`GbProcessL0Loop`（流水原语）、FP16 Cube kernel 与
alpha/beta、early-exit AIV kernel（通用路径）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

依赖 CANN 9.1.0（asc-devkit ≥ 9.1，tensor_api）；低版本工具链构建时经
`TENSOR_API_OPS` 列表自动跳过本算子。

## 算子约束限制

- 不支持广播；不支持超出前导维语义的非连续 stride；
- Carray 各 batch 原地覆写，各 C[i] 之间不得重叠；
- 不要求确定性计算（任务书 §2.5）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT16 混合容差档：rtol=atol=2⁻¹⁰，matched_ratio≥0.99，max_abs_error≤max(1e-1,32·ULP)；golden=cblas_sgemm(FP32) 后 IEEE754 RNE 舍入；Inf/NaN 按非有限值等效规则 | 生态算子开源精度标准 mixed_tolerance_standard §FLOAT16；任务书 §3.2 |
| 性能标准 | msprof op task duration：200 条性能用例单条倍率（H100/NPU）≥0.4 且算术平均 ≥0.5 | 任务书 §3.3 |

实测结果（Ascend 950PR 9579，CANN 9.1.0）：精度 1001/1001 全部通过；性能 200/200
通过，最低倍率 0.463（8192³ TT b1），平均倍率 1.503。

## 可维护性

- 发射前容量断言将选参越界转化为带现场参数的一行精确报错；
- 计划出口单行决策日志（路径/tile/kc/切分/任务数）支持按 shape 复现调度形态；
- kernel 双 L0C 策略以编译期布尔特化同一模板，流水逻辑单点维护；
- 跨目录 kernel 发射接口统一经 `gemm_batched/arch35/gemm_batched_kernel.h` 显式声明。

## 兼容性分析

新增 API（头文件新增 `aclblasHalf` typedef 与 `aclblasHgemmBatched` 声明），
与既有接口无签名冲突；接口定义与其他产品线共用，未引入 950PR 私有平行 API。
