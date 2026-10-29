# 需求背景（required）

## 需求来源

10 月社区任务：aclblasHgemm 算子开发（Ascend 950PR）。任务要求在 ops-blas 开源仓
（https://gitcode.com/cann/ops-blas）中补齐 Blas 仓 Level-3 GEMM 族的半精度（FLOAT16）能力，
以 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasHgemm`，对标 cuBLAS `cublasHgemm`，
验收通过后合入 `blas/gemm/arch35/`，测试代码合入 `test/gemm/hgemm/arch35/`。

## 背景介绍

### Gemm 算子族现状

ops-blas 仓 `blas/gemm/` 当前已实现：

| 接口 | 数据类型 | arch35 状态 |
| --- | --- | --- |
| aclblasSgemm | FP32 | 已支持 |
| aclblasCgemm | COMPLEX64 | 已支持 |
| aclblasHgemm | FLOAT16 | **缺失（本任务补齐）** |

`include/cann_ops_blas.h` 中无 `aclblasHgemm` 声明，需新增与其他产品线共用的 API
（禁止定义 950PR 私有平行接口）。

### 对标接口

cuBLAS `cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`：
列主序、alpha/beta 为半精度标量指针、维数为 int。本算子参数序列与之一一对应，
alpha/beta 采用 `const aclblasHalf*` 指针形式（`aclblasHalf` 为 IEEE 754 binary16 位模式，
与 ACL `aclFloat16` 约定一致）。

### 输入输出规格

| 参数 | 含义 | 数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- |
| alpha/beta | 标量乘数 | FLOAT16（Host 指针） | 不可为 nullptr | 标量 |
| A | 输入矩阵 | FLOAT16（Device） | m,n,k > 0 且 alpha≠0 时不可为 nullptr | m×k（N）/ k×m（T,C） |
| B | 输入矩阵 | FLOAT16（Device） | 同上 | k×n（N）/ n×k（T,C） |
| C | 原地输出 | FLOAT16（Device） | m,n > 0 且（k > 0 或 beta≠0）时不可为 nullptr | m×n |

计算公式：`C[i][j] = alpha × Σ_{p<k} op(A)[i][p] × op(B)[p][j] + beta × C[i][j]`

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR（arch35）上实现 `aclblasHgemm`：
FLOAT16 输入/输出、FLOAT32 累加、列主序、N/T/C 全转置组合、前导维语义、
边界（m=0/n=0 no-op、k=0 仅 beta 缩放、beta=0 置零且不读 C）与全量参数校验。

## 需求拆解

1. 接口：`include/cann_ops_blas.h` 新增共用声明；`cann_ops_blas_common.h` 新增 `aclblasHalf` 类型
2. 功能：`C = alpha*op(A)*op(B) + beta*C`，N/T/C 全组合，任意 m/n/k ≥ 0，ldc/lda/ldb padding
3. 边界：m=0/n=0 no-op；k=0 或 alpha=0 时 C=beta*C（beta=0 置零且不引用 A/B/C 值）；
   校验顺序遵循任务书 §2.5（alpha/beta 判空先于 m/n 短路，A/B/C 判空后于短路）
4. 精度：生态算子开源精度标准 FLOAT16 档（rtol=atol=2^-10、matched_ratio≥0.99、
   max_abs_error ≤ max(1e-1, 32*ULP)），golden 为 FP32 cblas_sgemm 后按 RNE 转回 FP16
5. 性能：单 case ≥ 0.4× H100、全部性能 case 平均倍率 ≥ 0.5×（任务书 §3.3）

# 详细设计（required）

## 算子分析

### 数学公式

C = alpha · op(A) · op(B) + beta · C，op ∈ {N, T, C}（实数下 C ≡ T）

### 支持数据类型

FLOAT16（A/B/C/alpha/beta），L0C 内以 FLOAT32 累加

### 支持形状

任意 m/n/k ≥ 0，前导维 lda/ldb/ldc ≥ max(1, 物理行数)；不支持非连续 Tensor 与 broadcast

## 算子实现

### 实现方案

实现位于 `blas/gemm/arch35/`，复用 Gemm 家族的 `GemmTilingData` 与 host 公共逻辑：

- `hgemm_host.cpp`：API 入口、参数校验、tiling、多核切分、退化路径与 kernel 下发
- `hgemm_kernel.cpp/.h`：Cube kernel + AIV epilogue/scale kernel 与下发函数

#### Host 侧设计

**列主序转换**：列主序 C(m×n) 等价于行主序 C^T(n×m)，C^T = op(B)^T · op(A)^T。
host 侧交换 m/n、lda/ldb、isTransA/isTransB 后以行主序视角下发 Cube kernel
（a 传 B、b 传 A），与仓内 Sgemm/Cgemm 保持一致的转换约定。

**参数校验**（顺序即优先级）：handle 空指针 → transa/transb 非法枚举（`ACLBLAS_STATUS_INVALID_ENUM`）
→ m/n/k 负值（`ACLBLAS_STATUS_INVALID_VALUE`）→ 前导维约束 → alpha/beta 判空
→ m/n 短路（返回成功，不访问矩阵）→ A/B/C 按条件判空。

**多核切分**：以 128×128 tile 为粒度做矩形切分，在 AIC 核数（950PR 为 28）内枚举
(mBlocks, nBlocks)，按**组级代价模型**选取最优组合：每核的 tile 块被切成
(GROUP_M × GROUP_N) tile 组、每组扫满 K 轴，单个 (mG×nG) 组的代价按
`max(MAC cycles, 带宽 cycles) × K + 流水重启` 估计（退化的 2×1/1×2 组带宽受限，
单位 MAC 代价约为完整 2×2 组的 2 倍），取代价最小者、并列取核数更多者。
该模型在 tile 网格不能整除时自动避免"makespan 最小但出现大量退化组"的劣化切分
（如 2936³：偏好 6 个完整 2×2 组/核而非 12 个 2×1 组/核）。

**执行路径**：

| 条件 | 路径 |
| --- | --- |
| k=0 或 alpha=0，beta=0 | 仅对 m×n 逻辑区域置零（ldc=m 时 memset，否则 AIV kernel 写零，保留前导维填充） |
| k=0 或 alpha=0，beta≠0 | AIV scale kernel：C = beta×C（FLOAT32 计算、RNE 回写 FP16；beta=1 时 no-op） |
| alpha=1 且 beta=0 | Cube kernel 直写 C（Fixpipe 完成 FLOAT32→FLOAT16） |
| 其他 | Cube kernel 写 FLOAT32 workspace（行距 CeilAlign(m,16)），AIV alpha_beta kernel
以 FLOAT32 施加 alpha/beta 后 RNE 写回 C |

alpha/beta 为 Host 半精度标量，host 侧经位转换读取为 FLOAT32 参与 tiling/判零。

#### Kernel 侧设计（Cube）

Cube kernel `hgemm_kernel_fp16`（`__cube__`，AIC）核心结构：

1. **L0C tile 组**：`(GROUP_M × GROUP_N) = (2×2)` 个 `128×128` tile 组共享 L0A/L0B 操作数，
   即一个 256×256 的输出块在 L0C（256KB，FLOAT32）中同时累加。组内每步 Mmad 的
   计算强度为 64 MAC/byte，与 4096 MAC/cycle 满算力下的 64B/cycle 搬运带宽平衡，
   这是性能的关键。
2. **L1 4 级环形缓冲**：每级 128KB 暂存整组的一个 K 分块（K_CHUNK=128：A 256×128、B 128×256），
   共 512KB（arch35 L1 实际容量）；多级 GM→L1 DMA 同时在飞，L2 miss 延迟（如 8192³ 的
   256MB 面板超出 128MB L2）被环形缓冲吸收。
3. **L0 装载**：以 64-K 子块（BK=64）为单位把组内 4 个子 tile 的 A/B 操作数装入 L0A/L0B
   2 级乒乓槽位，每个子块对应单条 `(128×128×64)` Mmad（指令数与 MTE1 请求减半）；
   每个拷贝目的布局与对应 Mmad 操作数布局完全一致（同形同基），保证 NZ/ZN 分形步长一致。
4. **Mmad**：`te::Mmad`（half×half→float）按 (miSub, niSub) 遍历组内 4 个 tile，
   首 K 块 cmatrixInitVal 清零累加，末 K 块 FINAL_ACCUMULATION。
5. **Fixpipe**：组级 K 循环结束后 `CopyL0C2GM` 将 4 个 tile 写出
   （直写路径完成 FLOAT32→FLOAT16 就近舍入；workspace 路径保持 FLOAT32）。
   Fixpipe 延迟到下一 tile 组首个 K 分块 DMA 之后下发，使其与下组的装载流水重叠。
6. **尾块**：m/n/k 任意尾块（非 128/64 对齐）通过运行时 curM/curN/curK 参与拷贝与 Mmad，
   分形对齐由 NZ/ZN 布局自动补齐。

#### Kernel 侧设计（AIV epilogue）

- `hgemm_alpha_beta_kernel`：读 FLOAT32 workspace 与原 C（FLOAT16），FLOAT32 计算
  `alpha×AB + beta×C` 后 `RoundMode::CAST_RINT` 就近舍入写回 FLOAT16；按列切分 AIV 核
- `hgemm_scale_kernel`：`C = beta×C` 原地路径，FLOAT32 计算保证与 golden 单次舍入一致；
  beta=0 分支写零不读 C（C 允许含 NaN）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

## 算子约束限制

- 不支持非连续 Tensor（仅前导维语义）、不支持 broadcast、C 原地覆写
- alpha/beta 为 Host 内存标量指针，不可为 nullptr
- A/B/C 为 Device 内存；判空条件见任务书 §2.5

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 生态算子开源精度标准 FLOAT16 档：rtol=atol=2^-10、matched_ratio≥0.99、max_abs_error≤max(1e-1,32×ULP)；Inf/NaN 按等效规则判定 | mixed_tolerance_standard.md（任务书 §3.2） |
| 性能标准 | 全部性能 case 单条倍率 ≥ 0.4× H100、平均倍率 ≥ 0.5×（msprof task duration 口径，warmup 5 + 采样 10 取均值） | 任务书 §3.3 |

## 测试设计

测试位于 `test/gemm/hgemm/`（参数解析、golden/判定、设备封装）与 `test/gemm/hgemm/arch35/`
（GTest 主体 + 1200 条 CSV），复用仓内 test/frame 框架：

- 精度：1000 条（L0 转置全组合 / MN 尺寸扫描 / AB 标量 / LD 前导维 / FL Inf/NaN 填充 /
  CV 中等覆盖 / ED 边界负向 / EX 扩展采样），golden = FP32 cblas_sgemm + RNE 转 FP16，
  按 §3.2 混合容差 + §3.2.5 特殊值规则逐元素全矩阵比对
- 性能：200 条 TC_PF，设备端常驻缓冲、warmup 5 + NPU event 有效采样 50 取均值，
  输出 `PERFORMANCE_SUMMARY_JSON`；官方口径以 msprof op 模式采集 task duration 复核

## 兼容性分析

新增 API 与新增标量类型 `aclblasHalf`（`cann_ops_blas_common.h`），不改变既有接口行为；
`test/frame/verify.h` 的 FLOAT16 档默认阈值按新精度标准从 2^-9 刷新为 2^-10（仅收严，不放宽）。
