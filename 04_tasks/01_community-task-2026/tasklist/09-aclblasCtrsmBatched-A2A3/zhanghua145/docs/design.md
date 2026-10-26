# aclblasCtrsmBatched 算子设计文档（Atlas A2/A3 / arch22）

> 算子：`aclblasCtrsmBatched`（单精度复数批量三角求解，Batched Complex Triangular Solve）
> 目标目录：`blas/trsmbatched/arch22/`（算子）+ `test/trsmbatched/ctrsmbatched/arch22/`（测试）
> 适配硬件：Atlas A2 系列（Atlas 800T A2 / 910B3，性能验收设备）与 Atlas A3 系列
> CANN 版本：9.1.0；编程模型：Ascend C（kernel 直调，handle + stream 句柄式 BLAS 接口）

# 需求背景（required）

## 需求来源

9 月社区任务《aclblasCtrsmBatched 算子开发（A2A3）》：在昇腾 NPU（Atlas A2/A3 系列）上基于 Ascend C 开发单精度复数批量三角求解算子，对齐 cuBLAS `cublasCtrsmBatched` 能力，验收通过后合入昇腾算子开源仓 `https://gitcode.com/cann/ops-blas`（目录 `blas/trsmbatched/arch22/`）。

## 背景介绍

### 数学语义

对每个 batch i（i = 0..batchCount-1）独立求解三角线性系统，解 X[i] **原地覆写** B[i]：

- `side=ACLBLAS_SIDE_LEFT`：`op(A[i]) · X[i] = alpha · B[i]`，A[i] 为 m×m 三角阵
- `side=ACLBLAS_SIDE_RIGHT`：`X[i] · op(A[i]) = alpha · B[i]`，A[i] 为 n×n 三角阵
- `op(A[i]) ∈ {A[i]（OP_N）、A[i]ᵀ（OP_T）、A[i]ᴴ（OP_C，共轭转置）}`

A、B 均为**列主序**存储，前导维分别为 lda / ldb；batch 为 uniform（各批次共享 m/n/lda/ldb 与 side/uplo/trans/diag 标志，仅矩阵地址不同）。`uplo` 指定引用 A 的上/下三角；`diag=UNIT` 时对角视为 1 且**不读取** A 的对角存储位置（Netlib 语义）。

### ops-blas 仓现状与工程先例

- 仓内已有 `blas/trsmbatched/arch35/`：Atlas 950PR 的 `aclblasStrsmBatched`（实数）与 `aclblasCtrsmBatched`（复数）参考实现，本任务目录组织（host/kernel/tiling 四文件制）与其对齐，但 arch35 采用 950 SIMT 编程模型（`asc_simt.h`），**不能**直接移植到 arch22（910B3 为传统 Ascend C 向量/Cube 模型）。
- arch22 复数 BLAS3 已合入先例：`cherk` / `cher2k` / `csymm` / `csyrk` / `ctrmm` 等，均基于「AoS↔SoA 平面分解 + 向量/`Matmul` API」模板；`blas/gemm/arch22`、`blas/gemm_strided_batched/arch22` 验证了 arch22 上 `Matmul` API（`lib/matmul_intf.h`）ND 格式 f32 Cube 路径可用。
- 接口声明需进入 `include/cann_ops_blas.h`（供其他产品线共用，禁止私有平行 API）。

### arch22 硬件特性（设计前提，Atlas 800T A2 / 910B3 实测口径）

| 资源 | 数值 | 对本算子的意义 |
| --- | --- | --- |
| AI Vector | 40 AIV | panel 回代 / 格式转换 / 小矩阵路径 |
| AI Cube | 20 AIC，f32 MMA 峰值 ~273 TFLOP/s（实平面） | trail 更新 GEMM；复数 4M 分解后有效 ~68 TFLOP/s |
| HBM | 实测可达聚合 ~1.45 TB/s（读） | workspace 往返流量预算 |
| UB | 256 KB/核 | panel 常驻、AoS↔SoA 暂存 |
| kernel launch（稳态） | ~5 µs/次 | staged 多 kernel 方案的轮次开销预算 |
| 多核同步 | kernel 边界同步开销远低于核间 flag 往返累积 | 避免 MIX 核内跨核事件链 |

### 功能缺口

ops-blas 仓当前无 arch22 版 `aclblasCtrsmBatched`；本任务补齐该缺口。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasCtrsmBatched` 句柄式接口（13 参，签名与任务书 §2.3 / cuBLAS 一致），COMPLEX64 输入输出，支持 side×uplo×trans×diag 全部 24 组枚举组合、m/n ∈ [0, 4096]、batchCount ∈ [1, 1024]、lda/ldb padding、alpha 特殊值（0 置零 / 复数标量）、B 含 Inf/NaN 的传播语义；精度满足生态混合容差标准（COMPLEX64 档），性能满足任务书 §3.3 五个典型 case 及 200 条 PF 用例（NPU 平均单次 kernel 耗时 ≤ gpu_ms / 0.8）。

## 需求拆解

1. **功能正确**：24 组枚举 × 尺寸 × batch × 前导维全组合精度通过（1000 条 CSV 精度用例 + 负向用例）；
2. **性能达标**：msprof 口径（5 warmup + 10 采样平均）下，200 条 TC_PF 用例 kernel 耗时 ≤ gpu_baseline 的 gpu_ms/0.8，含任务书 §3.3 五个锚点 case；
3. **工程规范**：四文件制目录、接口声明入 `cann_ops_blas.h`、README 产品支持表、代码简洁无冗余；
4. **异常语义**：handle nullptr → `HANDLE_IS_NULLPTR`；非法枚举/负维度/前导维不足/batchCount<1/alpha 或 B 数组空 → `INVALID_VALUE`；m=0 或 n=0 → no-op SUCCESS；alpha=(0,0) → B 逐批置零且不访问 A。

# 详细设计（required）

## 算子分析

### 数学公式

列主序下（LEFT 为例）：

```
X = alpha · op(A)⁻¹ · B          （LOWER + OP_N：前代；UPPER + OP_N：回代）
```

由于 A 三角且 op() 仅转置/共轭，可统一化归为「对 op(A) 的有效三角做逐块前代/回代」：
`op(A)` 的有效三角方向 = `uplo` 与 `trans` 共同决定（转置类 op 使三角方向翻转、共轭类 op 额外对引用元素取共轭），求解方向随之确定（有效下三角→前代，有效上三角→回代）。

### 列主序 → 行主序的转置恒等式（接口归一）

核心洞察（业界标准做法）：列主序 `op(A)·X=αB` 与行主序 `Xᵀ·op(A)ᵀ=αBᵀ` 是同一内存布局的两种解读。因此 host 侧做一次**语义映射**：

```
side ↔ 取反，uplo ↔ 取反，m ↔ n 互换，trans/diag/alpha/lda/ldb 不变
```

即可把所有 24 组枚举组合归一为**行主序 side=RIGHT 单一求解方向**（对 op(A)ᵀ 的行向量组逐行前代/回代），kernel 只实现一份索引逻辑。RIGHT+OP_T/C 组合再通过「A 逻辑转置读取」覆盖，消除分支乘法。

### 支持数据类型与形状

- dtype：COMPLEX64（实部/虚部 float32 交织，AoS 8B/元素）
- m, n ∈ [0, 4096]；batchCount ∈ [1, 1024]；lda ≥ max(1, kDim)、ldb ≥ max(1, m)（kDim = side=LEFT ? m : n）
- 非连续仅限 lda/ldb padding 语义（任务书明确不要求超出前导维的非连续支持）

## 算子实现

### 实现方案总览（路径分派）

```
aclblasCtrsmBatched(handle, ...)
  ├─ 参数校验（§异常语义）→ 非法即返回
  ├─ m==0 || n==0 → SUCCESS（no-op）
  ├─ alpha==(0,0) → 逐批 B 置零（异步 memset，不访问 A）
  ├─ 语义映射（转置恒等式）→ 行主序 side=RIGHT 视图
  ├─ kDim ≤ SMALL_K（=32）→ [P3] 单 kernel 向量直接回代（AIV 多核）
  └─ kDim > SMALL_K → [P1/P2] staged blocked TRSM：
        prep kernel（A/B 规范化：仅需要的转换）
        for p in 0..numPanels-1:
            panel kernel   （AIV：所有 batch 的第 p 个对角块三角求解）
            update kernel  （AIC：所有 batch 的 trail 批量 GEMM 更新）
        writeback kernel（workspace → B 原地覆写）
```

P1/P2/P3 三条路径共用同一 tiling 结构与 GEMM 构件，文件组织维持四文件制（host / kernel / kernel.h / tiling_data.h）。

### Host 侧设计

#### 参数校验与错误码（次序固定）

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. side/uplo/trans/diag 非法枚举 → `INVALID_VALUE`
3. `m<0 || n<0 || batchCount<1 || alpha==nullptr || B==nullptr` → `INVALID_VALUE`
4. `m==0 || n==0` → `SUCCESS`（quick return）
5. `alpha≠(0,0) && A==nullptr` → `INVALID_VALUE`
6. `lda < max(1,kDim) || ldb < max(1,m)` → `INVALID_VALUE`
7. alpha=(0,0)：B 逐批 `aclrtMemsetAsync` 置零（列跨度 `ldb*n` 字节）后返回 SUCCESS

#### Tiling 数据结构与分核

```
CtrsmBatchedTilingData {
  m, n, lda, ldb, batchCount,      // 原始接口参数
  kDim, rhs,                       // 语义映射后的求解阶数与 RHS 列数
  side/uplo/trans/diag（归一后）,
  alphaRe, alphaIm,
  nb,                              // panel 块宽（64/128，按 kDim 与 UB 预算选取）
  numPanels, forward,              // panel 数与求解方向（有效三角决定）
  workspace 布局参数（perBatch 偏移、A/B 区段、对角块逆暂存区）
}
```

- **分核策略**：
  - panel kernel：`(batch, rhsColBlock)` 二维任务网格，40 AIV 均分；每任务处理一个 batch 的一段 RHS 列块 × 一个 nb 对角面板。
  - update kernel：`(batch, trailRowBlock × rhsColBlock)` 任务网格，20 AIC 经 `Matmul` API 分块（M×N×K = trailRows×rhsCols×nb，按 K 单次、MN 分块覆盖 AIC）；batch 间天然独立。
  - 直接回代路径（P3）：`(batch, rhsColBlock)` 网格，kDim 小于阈值时单核内串行 kDim 步、多核吃 batch×列块维度。

#### workspace 布局（复用 handle 工作区）

```
[tiling(4K对齐)][perBatch 区: A规范化(按需) | B 工作副本 | 对角块暂存][Matmul 系统工作区]
```

- B 工作副本：staged 循环中所有中间结果留在 workspace，最终 writeback 一次写回 B（原地语义，避免每步读写原址）；workspace 优先复用 `aclblasHandle` 默认工作区（`EnsureDefaultWorkspace`），超限时申请大工作区并进程级缓存复用。
- A 规范化：仅 `trans≠N` 或非对齐尺寸时需要（转置/共轭打包为连续行段，供 panel 读取与 GEMM A 矩阵使用）；`useOrigA` 直通优化（N + 对齐 + LOWER 前代）跳过拷贝。
- 每次调用对使用的 workspace 段做异步清零（防止跨调用残留），tiling 走异步 H2D 拷贝（线程局部持久副本）。

### Kernel 侧设计

#### panel 求解（AIV，多 batch 并行）

行主序视图下，第 p 个对角面板是 nb×nb 三角块 `T`，对 RHS 列块做逐行前代/回代：

```
for r in 0..nb-1（面板内行，串行依赖链）:
    y[r] = (y[r] − Σ_{j<r 已解行} T[r,j]·y[j]) / d[r]     // NON_UNIT 除对角；UNIT 跳过
    对面板内 r 之后的所有行: y[i] −= T[i,r]·y[r]          // column-sweep，rank-1 向量更新
```

- 依赖链长度 = nb（64~128），链上每步为一次复数 FMA 向量段（面板行 × RHS 列块，连续内存），链外（batch × RHS 列块 × 面板后继行）完全并行；
- 复数运算以「实平面 4 元素交织分解」实现：`(a+bi)(c+di)` 用 2×Mul + 1×Add + 1×Sub 的双平面形式（沿用 arch22 cherk/cher2k 模板），避免逐元素标量乘除；
- 面板所需的 `T` 从 GM 一次性载入 UB（nb×nb×8B ≤ 128KB），RHS 列块常驻 UB 分块处理；
- NON_UNIT 对角先做**幅值归一化**再除法（`scale = max(|re|,|im|)`，防对角元素上溢/下溢），UNIT 不读对角。

#### trail 更新（AIC，batched 复数 GEMM，4M 分解）

第 p 步 trail 更新（right-looking）：

```
B[trail, :] −= A[trail, panel] · B[panel, :]        （行主序视图）
```

- 复数 GEMM 用 **4M 分解**：`(A+Ai)(B+Bi) = (AB−AiBi) + i(ABi+AiB)` → 4 次实平面 GEMM；实平面 GEMM 走 `Matmul` API（ND 格式 f32，AIC 多核，tiling 由 `MultiCoreMatmulTiling` 生成并缓存）；
- 4M 需要实部/虚部平面分离：prep/panel 阶段在写 workspace 时直接写**平面分离布局**（`A_Re | A_Im` 两个实矩阵），GEMM 步零转换开销；writeback 阶段一次交织还原；
- batch 维：A/B 实平面矩阵按 batch 顺序拼接为大矩阵（M = Σ trailRows, N = rhs, K = nb 的 batched 分块），一次 `Matmul` 调用吃全部 batch，AIC 利用率最大化；或按 AIC 数分组下发（避免超长 M 的 tiling 退化），host 侧按 kDim/bc 选择；
- alpha 缩放合并进最后一步 GEMM 的 C 累加系数或 writeback 阶段（一次乘法，不在循环内重复）。

#### 小矩阵直接回代路径（P3，kDim ≤ 32）

单 kernel：`(batch, rhsColBlock)` 分核，每核将 32×rhsCols 的 RHS 段常驻 UB，逐行回代（kDim ≤ 32 步串行），A 三角按列段载入。无 workspace、无 staged 开销，覆盖小尺寸×大 batch 的性能敏感区（cuBLAS batched 的典型使用场景）。

#### UB 内存布局（256KB 预算，panel kernel 单核）

```
[面板 T（nb×nb 复数 AoS，≤128KB）]
[RHS 列块 y（nb × COL_BLK 复数，COL_BLK 按剩余 UB 选取，对齐 64 元素）]
[行段暂存 / 复数乘临时平面（4×float 段）]
```

#### 精度策略与误差分析

- **直接回代/blocked 更新为主**（不显式求逆）：数值稳定性对齐 Netlib 参考实现（逐步消元，误差按条件数线性放大），不引入逆矩阵显式化的平方级放大——这对 diag=UNIT 的大尺寸用例（对角恒 1、条件数依赖非对角幅值）尤其关键；
- 任务包测试对 UNIT 用例取 [-0.1,0.1] 小幅值分布控制条件数（与仓内 strsmbatched 参考工程同款口径），NON_UNIT 用例对角加 `boost = max(5, kDim)` 符号保持偏移——两种分布下条件数有界，blocked 消元的相对误差在 fp32 ULP 级，满足 2⁻¹³ 混合容差 + 0.99 matched_ratio + max_abs_error ≤ max(1e-2, 32ULP)；
- 归约（面板内行点积）采用分块累加 + 尾段标量补偿，控制长求和的累积误差；
- Inf/NaN 按 Netlib 传播语义自然传播（golden 非有限 → actual 非有限即过，生态标准 §2.3 收窄规则）；
- fp32 全程计算（COMPLEX64 即 float32 复数），无降精度路径。

### 计算流程图

```
┌──────────────────── host：校验 / 语义映射 / tiling / workspace ────────────────────┐
│  handle∈{null→错误码}; 枚举/维度/前导维校验; m·n==0→SUCCESS; α==0→置零返回        │
│  列主序→行主序恒等映射（side/uplo 翻转, m↔n）; nb/panels/分核/workspace 布局        │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       ▼ kDim ≤ 32 ──► [P3] 单kernel向量回代 ──► 返回
                      ┌────────────────────────────────────────┐
                      │ [prep] A/B 规范化（按需：转置/共轭/平面分离） │
                      └───────────────────┬────────────────────┘
                                          ▼
              ┌─────────────── panel 循环 p = 0..numPanels-1 ───────────────┐
              │ [panel kernel]  AIV 多核: 全 batch 的第 p 对角面板三角求解    │
              │   (batch × rhsColBlock) 任务网格; 面板内 nb 步串行链          │
              │ [update kernel] AIC 多核: 全 batch trail 批量 GEMM 更新      │
              │   4M 实平面分解, Matmul API, batch 拼接大 GEMM               │
              │   trail==0 时跳过（首个/末个面板）                            │
              └───────────────────────────┬────────────────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ [writeback] 平面交织还原 + α 缩放 → B 原地 │
                      └────────────────────────────────────────┘
```

### 性能预算（五个锚点 case 的 roofline 反算）

FLOPs（复数 mul-add 计）≈ kDim²·rhs/2 ·batchCount·8flop；HBM 流量 ≈ batchCount·(kDim²/2·8B + kDim·rhs·8B·2)。

| case | bc | m×n | 枚举 | 目标 µs | FLOPs | 纯算理想 µs@68TF/s | 流量 µs@1.45TB/s | 预算评估 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 256² | L/Low/N/NU | 895.1 | 4.3G | 63 | 46 | panel 链 8×~25µs + GEMM@50% ~126 → ~400µs，余量 2.2× |
| 2 | 128 | 384×512 | L/Up/C/NU | 5551 | 39G | 570 | 380 | GEMM@50% ~1.1ms，余量 5× |
| 3 | 32 | 1024² | R/Low/T/U | 13884 | 137G | 2020 | 247 | GEMM@40% ~5ms，余量 2.7× |
| 4 | 8 | 2048² | L/Up/N/NU | 21816 | 137G | 2020 | 550 | GEMM@40% ~5ms，余量 4× |
| 5 | 2 | 4096² | R/Up/C/U | 40969 | 550G | 8080 | 730 | GEMM@50% ~16ms，余量 2.5× |

结论：staged blocked TRSM 在全部锚点上有 ≥2× 的预算余量，主要风险在小 K（=nb）GEMM 的 Cube 利用率与 panel 串行链延迟，均在 spike 阶段验证（见可维可测）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（含 Atlas 800T A2 / 910B3） | √ |
| Atlas A3 系列 | √ |

## 算子约束限制

- 仅支持 COMPLEX64；alpha 仅 Host 指针（较 cuBLAS 收紧，任务书 §2.4）
- 不支持超出 lda/ldb 语义的非连续 Tensor（任务书 §2.5）
- 各 B[i] 之间不得重叠（原地覆写语义）；diag=NON_UNIT 时对角非零由调用方保证（不做奇异性检测）
- 确定性计算不保证（浮点累加顺序与分块策略相关，任务书 §2.5 允许）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差：atol=rtol=2⁻¹³，逐元素 \|actual−golden\| ≤ atol+rtol·\|golden\|；matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32ULP)；golden 由 cblas（Netlib ctrsm）逐 batch 生成；Inf/NaN 按生态标准 §2.3 收窄/传播规则 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能标准 | msprof op 口径（5 warmup + 10 采样平均）单次 kernel 耗时 ≤ gpu_baseline.csv 的 gpu_ms/0.8（含 §3.3 五锚点）；测试设备 Atlas 800T A2 (910B3) | 任务书 §3.3 / 测试指导 README |

## 测试设计

- **用例**：任务包 `ctrsmbatched_test.csv` 1200 条（精度 1000 + 性能 200，固定种子 20260823），覆盖 24 组枚举全组合、尺寸扫描（0/1/奇数/2幂±1/非对齐→2048/4096）、batchCount 1~1024、前导维 padding、填充模式（含 Inf/NaN、UNIT 全零 A）、边界与 CSV 负向（null 指针/非法枚举/非法前导维/负维度/batchCount=0）；
- **测试工程**：`test/trsmbatched/ctrsmbatched/arch22/`，参照仓内 `strsmbatched`（同族 CSV 驱动 + 指针数组逐批准备 + ZeroOppositeTriangle/BoostDiagonal 对角保障）+ 复数 MERE/MARE 策略（chemm 先例）新建；GTest 二进制加载 CSV，golden 由 Netlib cblas `ctrsm` 逐 batch 生成；
- **性能测试**：GTest 执行 TC_PF 用例（设备驻留 buffer、warmup 5 + 55 iters 端到端参考），正式口径用 `msprof op --launch-count=10 --warm-up=5` 采集 `OpBasicInfo.csv` 的 Task Duration 取平均，与 `gpu_baseline.csv` 逐条关联判定；
- **Spike 验证（开发前置）**：对锚点 case1/case2 形态（竞品实测超标的两个 case）先做小规模 staged blocked 原型（panel kernel + Matmul API GEMM 两 kernel 交替），达标 60% 门槛（即 ≤ 目标/0.6）才进入全量开发；不达标则触发架构复盘（备选：对角块显式求逆 + 单次大 GEMM 的 inv-first 路线，仅用于 NON_UNIT 良态区间）。

## 兼容性分析

新算子（arch22 目录新增），不涉及既有接口兼容性；接口声明合入 `include/cann_ops_blas.h` 后供其他产品线共用，`aclblasStatus_t` 语义与仓内 `cann_ops_blas_common.h` 一致。
