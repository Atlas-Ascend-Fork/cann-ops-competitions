# aclblasLtMatmul 算子设计文档

> 昇腾社区 2026 年 10 月社区任务 · `aclblasLtMatmul`（Atlas A2/A3 补齐）
> 任务 ID：`96a87ecddfe140e7bc29fc6959205ac5`　奖金：¥19,620　截止：2026/11/20
> 目标仓：`https://gitcode.com/cann/ops-blas`（分支 `feature/aclblasLtMatmul-arch22`）
> 文档模板：`cann-competitions/04_tasks/01_community-task-2026/resources/design_template.md`

---

## 需求背景（required）

### 需求来源

昇腾社区「10 月社区任务」发布的 `aclblasLtMatmul` 算子开发任务书（Atlas A2/A3 系列）。任务要求在
Ascend C 编程语言下，于 ops-blas 开源仓中补齐 `aclblasLtMatmul` 这一 cuBLASLt 风格矩阵乘接口的
**Atlas A2 / A3** 实现，并完成设计、开发、测试全流程，最终以 PR 形式合入 `cann/ops-blas`。

同批发布的还有 `aclblasGemmEx`、`aclblasGemmGroupedBatchedEx`、`aclsparseCgtsv2` 三个任务，本任务
在四者中**接口面最小、语义最规整**（单接口 / 单精度 FLOAT32 / 固定行主序 / 单一形状 m×n×k），
且奖金最高、截止期最宽裕，因此选定本任务。

### 背景介绍

### cuBLASLt `cublasLtMatmul` 语义对齐

`cublasLtMatmul` 计算

```
D = alpha * op(A) * op(B) + beta * C
```

其中 `op(X) = X`（`OP_N`）、`op(X) = Xᵀ`（`OP_T`）、`op(X) = Xᴴ`（`OP_C`），`alpha`/`beta` 为标量，
A/B/C/D 为矩阵，D 可与 C 原地（`C == D`）或异地计算。本任务要求在语义、参数含义、边界行为上
**逐项对齐 cuBLASLt**，主路径对齐 `EPILOGUE_DEFAULT`（Bias/ReLU/GELU 等扩展 epilogue 不在范围）。

### ops-blas 仓现状

| 路径 | 现状 |
| --- | --- |
| `include/cann_ops_blasLt.h:932-936` | `aclblasLtMatmul` 已有声明，签名与任务书一致，**与 950PR 共用同一声明，不得重复定义** |
| `blasLt/api/aclblaslt_matmul.cpp` | 句柄式 API 校验层已完整实现（287 行），校验顺序符合任务书 §2.5 |
| `blasLt/matmul/engine/matmul_engine.cpp` | 算子分发引擎：按 dtype 路由 MMAD 内核 → epilogue |
| `blasLt/matmul/fp32/arch35/` | 950 的手写 Mmad 内核（host + kernel + tiling）**已存在** |
| `blasLt/matmul/epilogue/arch35/` | 950 的 **SIMT** epilogue（`__simt_vf__` / `asc_vf_call`）**已存在** |
| `blasLt/matmul/fp32/arch22/` | **缺失** —— 本任务的主要缺口 |
| `blasLt/matmul/epilogue/arch22/` | **缺失** —— 本任务的主要缺口 |

结论：**接口层不需要重写，需要补齐的是 A2/A3 侧的设备端实现**，并把仓内若干与任务书口径不一致的
地方（OP_C、非法枚举返回码、computeType 校验、k=0 / alpha=0 语义、行前导维）一并对齐。

### 为什么 arch22 不能照抄 arch35

1. **950 的 epilogue 是 SIMT 实现**（`__simt_vf__`、`asc_vf_call`、`simt_api/asc_simt.h`），
   这是 Ascend 950 的向量能力，A2/910B 上不存在，arch22 必须另写**经典向量**版本。
2. **950 的 FP32 内核是手写 Mmad**（`AscendC::Mmad` + ND2NZ + LoadData2DParamsV2），
   `LoadData2DParamsV2`、`DataCopyCO12DstParams` 等 950 专有 API 在 A2 上不可用；
   且手写路径要自行处理 A2 的 L1→L0 装入契约与 B 操作数的转置布局，易错。
3. A2 上已有**仓内验证过的高层 Cube Matmul API 用法**（`blas/gemm/arch22`、`blas/trmm/arch22`、
   `blas/symm/arch22`、`blas/trsmbatched/arch22`、`blas/her2k/arch22`、`blas/gemm_strided_batched/arch22`），
   且 `CubeFormat::ND` 原生支持「行主序 + 任意行前导维」，与本任务的存储约定完全一致。

因此 **arch22 采用高层 Cube Matmul API 路线**（`lib/matmul_intf.h` + `MatmulImpl` + `GetMatmulApiTiling`）。

---

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言，在 ops-blas 仓中补齐 `aclblasLtMatmul` 的 Atlas A2/A3（arch22）实现：

- 复用 `include/cann_ops_blasLt.h` 既有声明，**不新增产品私有平行接口**；
- 主路径数据类型：FLOAT32（`Atype/Btype/Ctype/Dtype = ACL_FLOAT`，`computeType/scaleType = 32F`）；
- 四个 layout 描述符固定 `order = ACLBLASLT_ORDER_ROW`，**前导维按行主序解释为「一行元素数（行宽）」**；
- 单 batch：`batchCount = 1`、`stridedBatchOffset = 0`；
- 对标基线：cuBLAS `cublasLtMatmul`，精度 golden 用 Netlib / OpenBLAS 实数实现生成。

### 需求拆解

1. **接口与校验层对齐**
   - `transA/transB` 放开 `ACLBLAS_OP_C`（实数路径下 `C ≡ T`，与仓内 `matrix_transform` 既有约定一致）；
   - 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（任务书 §2.5），`scaleType` / `computeType` 不支持时返回
     `ACLBLAS_STATUS_NOT_SUPPORTED`；
   - 补 `computeType` 白名单校验（此前完全未校验）。
2. **arch22 FP32 内核**
   - 高层 Cube Matmul API，4 个转置组合模板实例化；
   - 关闭 HF32，保证 FP32 尾数不被截断；
   - K 维分片累加，输出行前导维严格等于 `lddRaw`。
3. **arch22 epilogue**
   - 经典向量实现（非 SIMT），`D = alpha * D_raw + beta * C`；
   - `alpha == 0` 不读 raw、`beta == 0` 不读 C，保证 `0 * NaN` 不污染结果；
   - 支持 `beta == 0 && C == nullptr`、原地（`C == D`）、DEVICE 标量模式。
4. **退化路径**：任务书 §2.1.5
   - `k == 0`：只做 beta 缩放，**不引用 A/B**；
   - `alpha == 0`：`beta == 1` 时 C 不变、`beta == 0` 时 D 置零、其余逐元素缩放，**不引用 A/B**。
5. **算法（algo）联动**：`aclblasLtMatmulAlgoGetHeuristic` 返回的算法要真正影响 tiling，
   arch22 侧用**多份编译期特化**承接运行时 `baseN` 选择。
6. **工程集成**：CMake include 路径按 `SOC_ARCH_DIRS` 条件化，arch22 构建只编 arch22 源码。
7. **测试与交付**：1200 条随包用例（1000 精度 + 200 性能）驱动、CPU golden 比对、
   性能采样、README 支持表、自测报告。

---

## 详细设计（required）

本节的耗时示例与模型标定数字属于历史开发实验，原始分析保存在
`analysis/report_a53_before_strict_review.md`；当前冻结源码的全量验证结果见文末“当前验证状态”。

### 算子分析

### 数学公式

```
D = alpha · op(A) · op(B) + beta · C
```

- `op(A) ∈ ℝ^{m×k}`，`op(B) ∈ ℝ^{k×n}`，`C, D ∈ ℝ^{m×n}`；
- `alpha, beta ∈ ℝ`（FLOAT32 标量，HOST 指针或 DEVICE 指针两种模式）；
- 实数路径下 `op_C(X) = Xᴴ = Xᵀ = op_T(X)`。

### 存储与行主序前导维

全部矩阵按 **行主序** 存储，`ld` = 一行元素数（行宽）：

| trans | 物理形状 | 行宽约束 |
| --- | --- | --- |
| `transA = N` | A 为 m×k | `lda ≥ max(1, k)` |
| `transA = T/C` | A 为 k×m | `lda ≥ max(1, m)` |
| `transB = N` | B 为 k×n | `ldb ≥ max(1, n)` |
| `transB = T/C` | B 为 n×k | `ldb ≥ max(1, k)` |
| C / D | m×n | `ldc, ldd ≥ max(1, n)` |

`ld == 0` 按行宽归一（等价紧凑布局）；不提供行/列广播语义。

### 边界与退化语义（对齐 cuBLAS / Netlib quick return）

| 条件 | 语义 | 实现落点 |
| --- | --- | --- |
| `m == 0` 或 `n == 0` | 合法 no-op，返回 `SUCCESS`，不触碰任何矩阵 | API 层提前返回（先于 A/B/D 判空） |
| `k == 0` | 只做 beta 缩放，不引用 A 与 B | arch22 内核走**零填充**路径，raw 全 0 |
| `alpha == 0` | `beta=1` → C 不变；`beta=0` → D 全 0；其余 → 逐元素缩放；不引用 A 与 B | arch22 内核走**零填充**路径 + epilogue 跳过 raw |
| `beta == 0` | C 不必是有效输入（可传 `nullptr`） | epilogue `useC = 0`，不读 C |
| `C == D` | 原地计算 | raw 先 staging 到 workspace，epilogue 读 C 写 D |

### 支持数据类型

FLOAT32（主路径，验收范围）。`computeType ∈ {32F, 32F_PEDANTIC}`，`scaleType = ACL_FLOAT`。
混合精度（MXFP8/MXFP4）不在本任务范围。

### 支持形状

任意 `m, n, k ≥ 0`；单 batch。

---

### 算子实现

### 实现方案

分层链路（arch22 构建）：

```
aclblasLtMatmul (blasLt/api/aclblaslt_matmul.cpp)      ← 校验层：参数合法性、状态码、边界短路
   └─ BuildMatmulProblem (internal/src/aclblaslt_matmul_problem.cpp)   ← 纯 POD 问题描述
        └─ MatmulLaunch (blasLt/matmul/engine/matmul_engine.cpp)       ← 分发：dtype 路由 + epilogue
             ├─ matmul_fp32_get_tiling / matmul_fp32_do (blasLt/matmul/fp32/arch22/)   ← Cube 内核
             │      └─ arch22_design_notes.md 记录映射表与取舍
             └─ epilogue_alpha_beta_get_tiling / epilogue_alpha_beta_do
                    (blasLt/matmul/epilogue/arch22/)                    ← 经典向量 epilogue
```

**目录落点说明**：任务书 §2.2 写「实现代码放在 `blas/lt_matmul/`（arch22）」，但 ops-blas 仓的既有约定是
**blasLt 算子归 `blasLt/` 目录并链接 `libops_blasLt.so`**（`blas/` 是列主序 `libops_blas.so` 的算子集，
与 `blasLt` 是两个不同的库）。因此本实现落 `blasLt/matmul/fp32/arch22/` 与 `blasLt/matmul/epilogue/arch22/`，
与 `arch35` 目录结构一一对应。

### host 侧设计

#### 校验链（`blasLt/api/aclblaslt_matmul.cpp`）

| 顺序 | 校验项 | 失败返回 |
| --- | --- | --- |
| 1 | `lightHandle == nullptr` | `NOT_INITIALIZED` |
| 2 | `computeDesc/Adesc/Bdesc/Cdesc/Ddesc == nullptr` | `INVALID_VALUE` |
| 3 | `alpha == nullptr` 或 `beta == nullptr` | `INVALID_VALUE` |
| 4 | `pointerMode` 不支持 | `INVALID_VALUE` |
| 5 | `transA/transB ∉ {N, T, C}` | `INVALID_ENUM`（本次对齐） |
| 6 | `scaleType != ACL_FLOAT` | `NOT_SUPPORTED`（本次对齐） |
| 7 | `computeType ∉ {32F, 32F_PEDANTIC}` | `NOT_SUPPORTED`（本次补齐） |
| 8 | `m == 0 \|\| n == 0` | **提前返回 `SUCCESS`**（no-op 短路，先于布局与指针判空） |
| 9 | 维度 / 前导维 / batch / stride 非法 | `INVALID_VALUE` |
| 10 | dtype 组合不被任一内核支持 | `NOT_SUPPORTED` |
| 11 | `A/B/D == nullptr`（**`k == 0` 时 A/B 免除**：乘积项为空，两者是零元素矩阵） | `INVALID_VALUE` |
| 12 | 任一 layout `order != ROW` | `NOT_SUPPORTED` |
| 13 | `workspace` 指针非 16B 对齐 | `INVALID_VALUE` |
| 14 | `workspaceSizeInBytes` 非 0 且非 16B 对齐（任务书 §3.4「非空 workspace 须 16B 对齐」） | `INVALID_VALUE`（本次补齐） |
| 15 | `workspaceSizeInBytes < algo->max_workspace_bytes` | `INVALID_VALUE` |

`OP_C` 的处理：`HasSupportedTransposeModes` 放行 `OP_C`，内核侧把 `trans` 统一编码为布尔
（`transA = (desc->transA != OP_N)`），`OP_C` 与 `OP_T` 走同一条路径。**放行的前提是 dtype 白名单只含实数类型**
（FP32/BF16/MXFP8/MXFP4），复数类型在第 10 步被拒，因此不会出现「返回 Aᵀ 冒充 Aᴴ」。

#### tiling 设计

`MatmulFp32TilingData`（arch22 与 arch35 字段同名同型，保证 engine / heuristic 源码不变）：

| 字段 | 含义 | arch22 取值 |
| --- | --- | --- |
| `m, n, k` | 逻辑维度 | 来自 API 层 |
| `baseM, baseN, baseK` | Cube 基块 | `baseM/baseN` 由 host 侧 planner 在 {64,128,256}² 中按代价模型选定（见「代价模型与决策」）；`baseK = 32`（见下） |
| `kL1` | L1 K 深度 | 参与 algo 覆盖 |
| `usedCoreNum` | 启动核数 | `QueryCubeCoreNum` |
| `lda, ldb` | A/B 行前导维 | 来自 layout |
| `transA, transB` | 转置标志 | 0/1 |
| `lddRaw`（arch22 追加） | **raw 缓冲行前导维** | `ldd`（raw 落在 D）或 `n`（raw staging 在 workspace） |
| `alphaIsZero`（arch22 追加） | host 已知 `alpha == 0` | 0/1 |

`lddRaw` 由 engine 在 `matmul_fp32_get_tiling` 之后写入，**与传给 epilogue 的 `lddRaw` 同源**，
因此「内核写 raw 的行距」与「epilogue 读 raw 的行距」天然一致；raw 落在 D 时两者都是 `ldd`，
raw staging 在 workspace 时两者都是 `n`（workspace 只需 `m*n*4` 字节）。

| `baseM × baseN` 的取值依据见 `blas/gemm/arch22/gemm_kernel.cpp` 的仓内结论：FP32 累加器占 L0C，
`64×256` 基块为 64KB，`128×256` 需要 128KB 会导致 2048 以上方形场景的输出块被破坏。`baseK = 32`
与 `blas/gemm_strided_batched/arch22` 的 NN/NT/TT 一致（TN 因对齐问题用 64）。

**tile 数与核数的匹配（实测驱动的修正）**：kernel 把输出切成 `tileM × tileN` 的 tile，
再把 tile 轮流分给 `QueryCubeCoreNum` 起的 Cube 核。因此**并行度由 tile 数决定**，
而不是由矩阵规模决定。原先 `baseM = min(m, 256)`、`baseN = min(n, 256)` 的固定取值在
中尺寸上退化成极少的 tile：256×256×256 的问题只有 **1 个 tile**，24 个核里只有 1 个在工作，
实测 kernel 耗时 36.9 µs（对应 0.91 TFLOPS），而 512×512×512 反而只要 27.9 µs。

修正是让 tile 数不低于核数：当 tile 数小于核数时，把 `baseM` 降到 **MMA 的行粒度 64**
（`kArch22SingleM`），tile 数即变为 `ceil(m/64) × ceil(n/baseN)`。该调整**只改变输出的
分块方式，不改变 K 维的归约顺序**，因此结果逐位不变。实测（Atlas 910B1，CANN 8.5.0）：

| 用例 | 修正前 | 修正后 | 加速比 |
| --- | ---: | ---: | ---: |
| 128×128×128 | 11.28 µs | 7.34 µs | 1.54× |
| 256×256×256 | 36.87 µs | 15.61 µs | 2.36× |
| 512×512×512 | 27.90 µs | 11.27 µs | 2.48× |
| 270×270×270 | 42.43 µs | 15.68 µs | 2.71× |
| 1024×1024×1024 | 52.87 µs | 42.68 µs | 1.24× |
| 4096×4096×4096 | 1963.87 µs | 1961.63 µs | 1.00× |

大尺寸不受影响（tile 数本来就超过核数），中尺寸提升 2.4～2.7 倍。

**从固定规则到几何枚举**：上面的修正是「先固定 `baseN = 256`，再在 tile 数不足时把 `baseM`
降到 64」的单一规则。它保证了并行度，但**没有考虑 copy-in 流量随 tile 数放大**这件事：
`baseM` 越小 → M 方向的 tile 越多 → B 的每一行被读的次数越多 → 行距退化时流量越大。
因此最终实现把 `baseM / baseN` 一起交给代价模型枚举（见下文「代价模型与决策」），
在「并行度」与「copy-in 流量」之间取平衡。实测（74 条用例 × 8 组配置的扫描）：
planner 自动选择的总耗时为 6458 µs，逐条取最优的 oracle 为 6241 µs，差距 3.5%，
且 planner 的选择在 30/74 条用例上就是该用例的最优配置。

#### split-K 归约（K 维切分）

降低 `baseM` 只能把 tile 数补到核数级别，**但输出本身太小时 tile 数仍然远少于核数**：
113×233×1605 的问题在 `baseM = 64` 下只有 `ceil(113/64) × 1 = 2` 个 tile，24 个核里只有
2 个在算，其余 22 个空闲（利用率 8%）。这类「瘦长 / 大 K」形状正是性能超标最集中的形态
（窄维 n = 31/55/74/91、大 K 小 mn 的 113×233×1605 等）。

单纯调 `baseN` 无效：把 N tile 从 256 降到 128 会切换到 `singleN = 128` 特化，
单核吞吐约减半，tile 数翻倍带来的收益正好抵消（256×256×256 实测 15.61 µs vs 15.31 µs）。
**唯一能继续提并行度的方向是把 K 维的归约切开**——kernel 注释里也早就写明
「going below that needs a split of the K reduction」。

实现（arch22 专属，arch35 不受影响）：

| 环节 | 做法 |
| --- | --- |
| 决策 | host 侧 `ComputeSplitK`：`tile 数 < 核数` 时按 `核数 / tile 数` 取段数，受 `k / 64`（每段至少 64 个 K 元素）与 `≤ 8` 段约束，并受 workspace 预算约束（见下） |
| 段长 | `ceil(k / 段数)` 向上取整到 **16**（MMA 粒度）；由于向上取整可能产生**空段**（k = 520、8 段 → 段长 80，第 7 段起点 560 ≥ k），实际段数收敛为 `ceil(k / 段长)`，保证每段都有 K 元素 |
| 内核 | 工作项从「tile」扩展为「(tile, K 段)」二元组，按 `work / tile 数` 拆出段号；段 `s` 把部分和写到 raw 缓冲的第 `s` 个切片（切片步长 `m × lddRaw`） |
| 归约 | epilogue 逐段读回并相加，再统一乘 alpha；段缓冲与累加器各自使用独立队列，逐个 EnQue/DeQue 完成 MTE2→VEC 同步 |
| workspace | 每段一份 `m × n × 4` 的部分和缓冲，由 heuristic 计入 `algo->max_workspace_bytes`；**实际段数受调用方给的 workspace 容量约束**（`matmul_fp32_apply_workspace_budget`），容量不足则收缩段数直到 1 |

**为什么段数要受调用方 workspace 约束**：任务书 §3.4 规定 workspace 容量由
`aclblasLtMatmulAlgoGetHeuristic` 返回，容量不足应当返回 `ACLBLAS_STATUS_INVALID_VALUE`。
若实现无条件要求 `段数 × m × n × 4`，那么「`algo == nullptr` + 未预留 workspace」的合法调用形态
就会被误拒。实现的做法是：按实际可用容量收缩段数，最小退化为 1（单切片，不需要部分和缓冲），
此时行为与未启用 split-K 时完全一致。

**段数上限的由来**：部分和缓冲总量被限制在**输入输出数据流量的 1/4 以内**
（远低于任务书 §3.4 的 1.2× 上限），因此即使最坏形态也不会突破内存要求。

`splitK == 1` 时 kernel 走**无除法快路径**（不拆 (tile, 段)），保证原有路径零开销；
`splitK > 1` 时用 **32 位除法**拆分（工作项数远小于 2³²），避免 64 位除法在 Cube 核上的
显著开销。

#### 行距对齐重排（row-alignment repacking）

**问题**：Cube 的 copy-in 从 GM 读操作数时用「跨步重复 DMA」——一次搬多行，步长是行距。
该 DMA 只有在**行距是 64 个 FP32（256 B）的整数倍**时才能用；否则退化到「每 32 B 一次
DataCopy」，实测 A 侧慢 1.50×、B 侧慢 2.91×（自测报告 §4.4.3）。随包 200 条性能用例里
**184 条（92%）**的行距不满足该条件，超标用例绝大多数命中——这是与标杆差距的主因，
而**不是算力本身**。

**思路**：把 A/B 复制到 workspace 里行距对齐的暂存缓冲，让 Cube 读对齐后的副本。
逻辑尺寸与数值完全不变，**只有行距变了**，因此结果与不重排时逐位一致；
Cube kernel **一行都不用改**——它拿到的仍是同一套 `lda/ldb` 语义，只是这两个值换成了
暂存缓冲的行距。

**为什么改 `lda/ldb` 就够了**：`SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc)` 的映射是
（`blas/common/helper/complex_blas3_arch22.h` 与 SDK 头文件一致）：

| 模式 | transA/transB | orgM | orgN | orgKa | orgKb | orgKc |
| --- | --- | --- | --- | --- | --- | --- |
| NN | N / N | M | ldb | **lda** | K | lddRaw |
| TN | T / N | **lda** | ldb | K | K | lddRaw |
| NT | N / T | M | N | **lda** | **ldb** | lddRaw |
| TT | T / T | **lda** | N | K | **ldb** | lddRaw |

即 **A 的 GM 行距总是取 `orgKa`（A 非转置）或 `orgM`（A 转置），B 的总是取 `orgKb`（B 转置）
或 `orgN`（B 非转置）**。copy-in 用的行距就是这两个值，因此把 tiling 里的
`lda/ldb` 换成暂存缓冲的行距，对齐就自动生效，**不需要动 Cube kernel 的任何一行**。

**搬运实现**（`blasLt/matmul/fp32/arch22/matmul_fp32_pack_kernel.cpp`，
与 cube 共用 `matmul_fp32_pack_impl.h`）：

- 源行起点是任意字节偏移（行距不一定是 32 B 的倍数），普通 `DataCopy` 要求 32 B 对齐，
  因此用 **`DataCopyPad`**（Ascend C 的非对齐搬运原语，`DataCopyExtParams` 的长度与步长
  以**字节**为单位），经 UB 中转落到目标缓冲；
- 目标行距取 `round_up(宽, 64)` 个 FP32（256 B），首行 256 B 对齐后每一行都对齐；
- **一组行一次 DMA**：`DataCopyPad` 的 `blockCount` 取该组的行数，一次指令搬完一组；
  组内各行在 UB 里按 `round_up(宽, 8)` 个 FP32 的行距连续排列（DMA 把每行补齐到 32 B）；
- UB 暂存缓冲**整核只分配一次**（2 块 × 48 KB = 96 KB，两个操作数先后复用）；
- 两个操作数的行**不是分两次搬**，而是合成一条列表按「组」切分：一组行要么属于 A，
  要么属于 B，两个操作数的组在同一个 round robin 里分配给向量核；
- 两次握手用**定向事件**（`MTE2→MTE3` / `MTE3→MTE2`）而不是 `PIPE_ALL` 全局屏障：
  全局屏障每次几百周期，而一组一次握手，屏障开销会成为主导；
- 搬运 kernel 是 `AIV_ONLY`（纯向量核），与 Cube kernel 分属不同流水线，可与后续
  Cube 启动重叠。

**分组切法（按字节均衡）**：组的大小由「整张列表的字节 ÷ 目标组数」决定，
再被两个操作数各自的 staging 容量上限夹住（组必须能整组放进 UB 暂存缓冲），
最后按该组字节把每个操作数的行切成等大的组。

这样切的原因是旧切法（两个操作数各自均分）会让**窄操作数独占一个大组**：
`TC_PF_1176`（969×55×899）的 A 是 3.48 MB、B 只有 198 KB，旧切法给 B 一个 132 KB 的组，
最忙的向量核要搬 193 KB，实测重排 19.75 µs；按字节均衡后最忙的核搬 96.7 KB，
实测 16.28 µs。目标组数（20）与 staging 容量（48 KB）由 18 条超标用例的实测扫描标定，
详见历史实验记录 `analysis/report_a53_before_strict_review.md`。

**代价模型与决策**（`matmul_fp32_plan_workspace`，arch22 专属）：行距未对齐且预算允许时，
对需重排的操作数启用重排；代价模型选择 tile 几何和 split-K 段数，预算不足时保留不重排回退。

枚举空间：

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| `baseM` | 64 / 128 / 256 | M 方向 tile 高度；超过问题 M 的部分被裁掉（只留最小可用档） |
| `baseN` | 64 / 128 / 256 | N 方向 tile 宽度；`> 128` 走 `singleN = 256` 特化，`≤ 128` 走 `singleN = 128` 特化 |
| split-K 段数 | 1..8 | 受 `k / 64`（每段至少 64 个 K 元素）与 workspace 预算约束 |
| 重排 | 只排 A / 只排 B / 都排；预算不足可回退 | 由行距对齐及 workspace 预算决定，代价模型不将免重排路径作为常规性能竞争候选 |

> **为什么「是否重排」不交给模型**：copy-in 退化到「每 32 B 一次传输」时的代价
> **不由操作数的大小决定**。实测把「免重排」重新纳入候选集并扫描代价模型中标定的
> 非对齐 copy-in 带宽（4/8/16/30/60 GB/s），全量 200 条最好也只有 185 条达标：
> 模型在 `TC_PF_1175`（661×31×389）与 `TC_PF_1093`（38×1372×209）上选择了免重排，
> 分别比强制重排慢 24.3 µs 与 12.8 µs；而它选对的形状（`TC_PF_1061`、`TC_PF_1135`、
> `TC_PF_1165`）本来也都在达标线之外。历史实验记录 `analysis/report_a53_before_strict_review.md` 给出完整实测表。

代价模型：

```
总耗时 ≈ 轮次 × (单 tile 耗时 + 每次 launch 固定开销)
       + copy-in 耗时(行距对齐 ? HBM 速率 : 32 B 粒度退化速率)
       + 重排耗时(暂存字节 / 实测带宽 + 行数 × 每行 DMA 开销 + 额外一次 kernel 下发)
       + epilogue 耗时(split-K 段数 > 1 时才有)
```

其中：

- **单 tile 耗时** ∝ `baseM × baseN × k / singleN`（`kTileNsPerUs = 8.0e-4`，
  由 74 条用例 × 18 组几何配置的实测扫描标定）；
- **轮次** = `ceil(tile 数 × 段数 / 核数)`，`tile 数 = ceil(m/baseM) × ceil(n/baseN)`；
- **copy-in 耗时**：A 的每一行被每个 N tile 读一遍、B 的每一行被每个 M tile 读一遍，
  因此流量随 tile 数放大——这正是窄维形状退化最严重的原因；
- **重排耗时**：读一遍 + 写一遍操作数，再加一次 kernel 下发的固定开销；
- **workspace 预算是硬约束**：暂存缓冲 + split-K 部分和必须落在调用方给的容量里，
  放不下就退回到不重排（或更小的段数），**不会报错**（任务书 §3.4 允许调用方给 0 容量，
  实现必须给出正确结果）。

**几何枚举的一个实现陷阱（已修复）**：几何数组按「先按 M 裁剪、再按 N 裁剪」填充，
被裁掉的档位对应的数组元素保持默认值 `baseM = baseN = 0`。代价循环若直接遍历整个数组，
`baseM = 0` 的几何会算出 **0 轮次、0 tile 耗时**，其代价恒小于任何真实几何，
于是 planner 会「选出」一个 0×0 的 tile——内核拿到空的 tile 网格，
表现为该用例耗时暴涨 10～60 倍（实测 17 条用例命中）。因此代价循环必须只遍历
**已填充的前 `geomCount` 个几何**。该缺陷由 `tools/plan_probe.cpp`（直接调用 host 侧
planner 打印决策）定位，修复后 planner 的决策与实测最优配置的差距在 3.5% 以内。

**workspace 布局**：`[暂存 A][暂存 B][split-K 部分和]`，每个缓冲按 256 B 对齐放置，
末尾留 512 B 的对齐余量；总量由 heuristic 计入 `algo->max_workspace_bytes`，并受
任务书 §3.4 的 **1.2× 输入输出数据量**上限约束。

#### 算法（algo）联动

`aclblasLtMatmulAlgoGetHeuristic` 返回的 algo 经 `ApplyAlgoTilingOverrideFp32` 解码后写回
`tiling.baseN` / `tiling.kL1` / `tiling.baseK`（`baseN = min(baseN, n, l1n)`，`kL1 = min(k, l1k)`）。
arch22 侧据此在**编译期准备多份 baseN 特化**（256 / 128），设备端按 `tiling.baseN` 选择，
使 algo 选择在 A2 上**真实影响性能**，而不是变成空操作。

#### 退化路径（k == 0 / alpha == 0）

- `k == 0` 或 `alpha == 0`：host 侧**不启动 Cube 内核**，改启动一个零填充 kernel，
  把 raw 缓冲（行距 `lddRaw`）写满 0，全程不读 A/B；
- 随后 epilogue 计算 `D = alpha·0 + beta·C = beta·C`（`alpha == 0` 时 epilogue 跳过 raw 读取，
  `beta == 0` 时跳过 C 读取），结果与 cuBLAS 一致。

### kernel 侧设计

#### 操作数约定（与 950 的 gemm 不同）

`blas/gemm/arch22` 的公开 BLAS 是列主序，内部算的是 `Cᵀ = op(B)ᵀ·op(A)ᵀ`（left = B，right = A）。
本任务全行主序、算 `D = op(A)·op(B)`，因此：

- **left（`SetTensorA` / A_TYPE）= 本任务的 A**
- **right（`SetTensorB` / B_TYPE）= 本任务的 B**
- **out = 本任务的 D**（行主序，行距 `lddRaw`）

不引入 Cᵀ 变换。

#### `SetOrgShape` 映射

Cube 高层 API 的 `SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc)` 语义以仓内权威注释
`blas/common/helper/complex_blas3_arch22.h:182-197` 为准：

- A 的 GM 行距：`transA` 时取 `orgM`，否则取 `orgKa`；
- B 的 GM 行距：`transB` 时取 `orgKb`，否则取 `orgN`；
- 输出的 GM 行距：`orgKc`；
- 本次 launch 的 K 长度由 `SetSingleShape` 携带。

本任务全行主序下的填法：

| 组合 | `SetOrgShape` | `SetSingleShape` |
| --- | --- | --- |
| NN | `(M, ldb, lda, K, lddRaw)` | `(rows, cols, kChunk)` |
| TN | `(lda, ldb, K, K, lddRaw)` | 同上 |
| NT | `(M, N, lda, ldb, lddRaw)` | 同上 |
| TT | `(lda, N, K, ldb, lddRaw)` | 同上 |

其中 `lda/ldb/lddRaw` 为行前导维，`M/N/K` 为逻辑维度。

#### 关键实现点

1. **HF32 必须关闭**：`SetHF32(false)` / `SetHF32Mode(HF32Mode::DISABLE)`。
   仓内 `blas/gemm/arch22` 与 `blas/gemm_strided_batched/arch22` 均显式关闭，原因是 FP32 Cube 在部分
   CANN/SOC 组合下默认 HF32，会截断尾数、在长 K 归约上偏离 FP32 golden。
   两条 API 的签名随 CANN 版本不同（9.x 为 `SetHF32Mode(HF32Mode::DISABLE)`，8.x 为
   `SetHF32Mode(bool)`），因此按 `ASC_DEVKIT_MAJOR` 宏做版本分支，两条路径语义一致。
2. **K 维分片**：按 4096 一片循环，`IterateAll(out, accumulate)` 的累加标志取 `startK != 0`。
3. **分核与分块**：按输出 tile 划分，block 间以 `GetBlockIdx()/GetBlockNum()` 步长领取，
   尾块（`m % baseM`、`n % baseN`）单独处理，保证每个输出元素被恰好写一次。
4. **输出行距**：Cube 输出行距参数 = `tiling.lddRaw`。
5. **零填充 kernel**：独立启动（Cube kernel 是 AIC-only，无法做向量写），行距 `lddRaw`，64 位下标。
6. **编译期配置必须是命名空间作用域常量**：SDK 的 `GetMMConfig` / `GetMatmulApiTiling` 是
   `[aicore]` 函数，**不能从普通 host 函数里调用**（clang 报
   `candidate function not viable: call to [aicore] function from [host] function`）。
   同时 `MatmulImpl` 的第 5 个模板参数是 `const auto& MM_CFG`（引用型非类型模板参数），
   要求被引用的对象具有**静态存储期**且是常量表达式。
   因此 4 种转置 × 2 种 N 规格共 8 组编译期配置全部写成命名空间作用域的
   `constexpr MatmulShapeParams` / `MatmulConfig` / `MatmulApiStaticTiling` 变量，
   再由 `MatmulImpl<..., kTilingXxx>` 引用——与仓内 `blas/gemm/arch22`
   （`kShape` / `kConfig` / `kNnTiling`）同一写法。

#### epilogue（arch22，经典向量）

`D = alpha · D_raw + beta · C`，逐元素：

- 读 `D_raw`（行距 `lddRaw`）、读 `C`（行距 `ldc`）、写 `D`（行距 `ldd`）；
- `alpha == 0` → 不读 raw；`beta == 0` → 不读 C（`useC = 0`，也覆盖 `C == nullptr`）；
- 全部下标按 64 位计算（`m*n ≥ 2^32` 或大 `ld` 时 32 位会回绕）；
- DEVICE 标量模式下 `alpha/beta` 由设备侧从 GM 读回后再判定，语义一致；
- 原地（`C == D`）时同址读写，元素级一一对应，安全。

> **关键实现细节：DMA 补齐粒度必须是 8 个元素（32 字节），不是 16。**
> 每块按 `aligned = ceil(n / 8) * 8` 取整后，`DataCopyPad` 的 `rightPadding` 落在 `[0, 7]`。
> 若按 16 取整，`rightPadding` 会达到 15，而 MTE2 的 `DataCopyPad` 一次 DMA 只能补齐
> **一个 32 字节 burst（8 个 FP32）**，超出后 UB 中的补齐区不会被正确写入，
> 向量计算就会读到未初始化的 UB 内容。
> 实测症状：`n mod 16 ∈ [1, 7]` 的用例（如 n=1,5,17,20,33,65,100,129）结果全错，
> 而 `n mod 16 ∈ {0} ∪ [8,15]` 的用例全部正确——**与 n 本身无关，只与补齐量有关**。
> 该粒度与仓内已验证的 arch22 GEMM epilogue 一致
> （`blas/gemm/arch22/gemm_kernel.cpp`：`aligned = (cols + 7) / 8 * 8`）。

实现风格参照仓内 A2 经典向量算子（`blas/gemm/arch22` 的 `ScatterTileSegment` 使用
`Muls` / `Add` + `DataCopyPad` 的逐行拷贝与向量数学）。

> **split-K 归约的同步要求**：段缓冲与累加器各自使用**独立的单槽队列**，
> 每个缓冲在参与向量运算前都要 `EnQue` / `DeQue` 一次。
> `EnQue`/`DeQue` 承担的是 MTE2→VEC 的事件同步：省略它时，DMA 可能尚未写完，
> 向量单元就读到未完成的数据（实测表现为结果里混入数量级完全错误的值，
> 而各段部分和本身在 workspace 里是正确的）。
> 累加器必须跨越整个段循环存活，因此不能与段缓冲共用一个 FIFO 队列——
> 共用时 `DeQue` 会把累加器交还分配器，下一次 `AllocTensor` 就会覆盖掉正在累加的和。

#### 流程

```
aclblasLtMatmul
 ├─ 校验（状态码见上表）
 ├─ m==0||n==0 → SUCCESS
 ├─ BuildMatmulProblem → MatmulLaunch
 │    ├─ 决定 needEpilogue / rawStaged，算出 lddRaw
 │    ├─ FP32 路径：matmul_fp32_get_tiling → (algo 覆盖) → 注入 lddRaw / alphaIsZero
 │    │    ├─ 按调用方 workspace 容量收缩 split 因子
 │    │    ├─ k==0 或 alphaIsZero → 零填充 kernel → raw 全 0
 │    │    └─ 否则 → Cube kernel（4 转置特化，HF32 关闭，K 分片，行距 lddRaw）
 │    └─ needEpilogue → epilogue：alpha·raw + beta·C（splitK>1 时先归约各段部分和）
 └─ 返回状态码
```

### 工程集成（CMake）

`blasLt/CMakeLists.txt` 已按 `SOC_ARCH_DIRS` 过滤源码（只编当前 SOC 的 arch 目录），
但根 `CMakeLists.txt` 的 include 目录**硬编码**了 `blasLt/matmul/fp32/arch35` 与
`blasLt/matmul/epilogue/arch35`：arch22 构建会找到 arch35 的头文件，却编不到 arch35 的实现，
最终在 `matmul_fp32_get_tiling` / `epilogue_alpha_beta_do` 上出现未定义符号。
因此把这两个 include 目录改为**按 `SOC_ARCH_DIRS` 条件化**（`blasLt/matmul/fp32/<arch>`、
`blasLt/matmul/epilogue/<arch>`），arch35 构建行为保持不变。

除此之外还修复了 arch22 构建下暴露的**两个既有链接缺口**（这两个问题与 arch22 无关，
只要 `ENABLE_BLASLT_MXFP8=OFF` 或目标不是 arch35 就会命中）：

1. **engine 无条件引用 950 专有符号**：`blasLt/matmul/engine/matmul_engine.cpp`（架构无关）
   直接调用 `matmul_mxfp8_get_tiling/do`、`matmul_mxfp4_get_tiling/do`。这些实现只在
   `blasLt/matmul/mxfp8/arch35`、`blasLt/matmul/mxfp4/arch35` 下，前者被
   `ENABLE_BLASLT_MXFP8` 过滤、后者被 arch 过滤，于是 `libops_blasLt.so` 留下 4 个未定义符号，
   **任何链接该库的可执行文件都会链接失败**（实测 `libops_blasLt.so` 本身能建成，
   `blasLtMatmul_test` 在链接阶段报 5 个 undefined reference）。
   修复方式：新增编译期开关 `BLASLT_ENABLE_MXFP8`（= `ENABLE_BLASLT_MXFP8 && arch35 ∈ SOC_ARCH_DIRS`），
   engine 在开关为 0 时**不再引用**这些符号，MXFP8/MXFP4 请求显式返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
   同时把 mxfp8/mxfp4 的 include 目录也改为随开关条件化，arch22 构建不再解析到 arch35 的头文件。
2. **matrix transform 只有 arch35 实现**：`MatTransformLaunch` 由
   `blasLt/api/aclblaslt_transform.cpp`（架构无关）调用，实现只在
   `blasLt/matrix_transform/arch35`。新增架构无关桩实现
   `blasLt/matrix_transform/stub/matrix_transform_stub.cpp`，返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
   arch35 构建下该桩被排除（`blasLt/CMakeLists.txt`），不会与真实实现重复定义。

`blasLt/CMakeLists.txt` 的 `GLOB_RECURSE` 归档逻辑、`test/CMakeLists.txt`、
`test/blasLtMatmul/CMakeLists.txt` 均无需改动。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √（待与任务方确认：仓内 `get_soc_arch_dirs` 只把 `ascend910b*` / `ascend910_93*` 映射到 arch22，A3 是否有独立 arch 目录待确认） |
| Atlas 900I/T A3（950PR） | arch35 内核沿用；共享 API 校验和零乘积启动保护有修改，本轮仅有 CI 编译验证，未实机覆盖 |

---

### 算子约束限制

- 不支持：混合精度（MXFP8/MXFP4）、复数、`OP_C` 的复数共轭语义、多 batch / strided batch、
  非行主序 layout、Bias/ReLU/GELU 等扩展 epilogue、`ld == 0` 之外的广播语义。
- `workspace` 包含行距重排的 A/B 暂存、split-K 部分和及必要的 raw staging，不能概括为常规路径 0 字节。
  heuristic 按布局及最坏情况上报容量，planner 按调用方预算收缩；每项均计入任务书 §3.4 的 workspace/IO ≤1.2× 检查。
- `alpha`/`beta` 为 DEVICE 指针时，host 侧无法预知其数值，`alpha == 0` 的「跳过乘积」优化
  退化为设备侧判定（epilogue 读到 0 就不读 raw），结果仍然正确。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 任务书 §3.2 的 FLOAT32 档位：逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`（`atol = rtol = 2⁻¹³`）、`matched_ratio ≥ 0.99`、`max_abs_error ≤ max_abs_error_limit`（`1e-2` 或 `32×ULP`）；Inf/NaN 元素按 §3.2 的特殊值等价规则单独判定；`alpha == 0` 时 `D = beta·C` 须**位精确** | 任务书 §3.2（引 opbase `mixed_tolerance_standard.md`） |
| 性能标准 | 单 case ≤ 1.25× GPU 标杆（≥ 0.8× GPU）；预热 5 次、有效采样 > 10 次取平均 | 任务书 §4 交付件 3 |
| workspace | heuristic 上报及实际使用的 workspace ≤ 输入输出数据总量 1.2×；非空指针须 16B 对齐 | 任务书 §3.4 |
| 算子行为 | 与 `cublasLtMatmul` 逐项对齐（含边界、状态码、原地/异地） | 任务书 §2.1 |

> **判据说明（重要）**：随包 CSV 的 `mere_threshold` / `mare_multiplier` 两列只用于**统计**，
> 不作为判定口径。相对误差在乘积发生**相消**时是没有定义的：一个真值接近 0 的元素，
> 其绝对误差仍由参与求和的项的量级决定（K=2000、项量级 ~50 时绝对误差约 1e-4），
> 于是「最大相对误差」可以轻易达到 90%，而这与实现的精度无关。
>
> 判定实现放在**本算子专用**的 `test/blasLtMatmul/arch22/arch22_precision.h`，**没有改动**
> 仓内共享框架 `test/frame/verify.h`，原因有三：① 共享框架的 FLOAT32 档位是
> `rtol=2⁻¹⁰ / atol=2⁻¹⁶`，任务书 §3.2 的 A2/A3 验收口径是 `rtol=atol=2⁻¹³`；
> ② 共享框架把 `Inf/Inf`、`NaN/Inf` 一律判失败，与任务书 §3.2 的特殊值等价规则不一致；
> ③ 共享框架被仓内所有算子测试共用，不应为单个算子的验收口径改动。
>
> **Golden 的生成**：任务书 §3.1 指定 golden 由 cblas（Netlib 实数 `cblas_sgemm`）生成，
> 测试工程即按此实现——先把物理缓冲转成紧凑的逻辑视图 `op(A)` / `op(B)`，再用
> `cblas_sgemm(RowMajor, NoTrans, NoTrans)` 一次算完（行主序的 leading dimension 是行宽，
> 即 `lda = k`、`ldb = n`、`ldc = n`）。此外对小规模用例额外用 double 精度在 host 侧重算一份，
> 作为「内核距精确解有多远」的旁证（打印 `_vs_double` 统计，不参与判定）。
>
> **`max_abs_error_limit` 的 ULP 基数**：任务书把 `max_abs_error` / `max_abs_error_limit`
> 写成标量，而 ULP 需要基数。本实现取 `max(1e-2, 32×ULP(max|golden|))`（绝对误差上限随
> golden 的量级缩放）；同时把逐元素 `max(1e-2, 32×ULP(|golden_i|))` 的越界计数作为**诊断量**
> 打印在日志里（`per-element 32*ULP violations=`），两种口径的结果在自测报告中一并给出。

精度自验：随包 `blasLtMatmul_test.csv` 共 1200 行（1000 条精度用例 + 200 条性能用例），
覆盖维度扫描（0/1/质数/2 的幂 ±1/非对齐/大规模）、转置组合（含 `OP_C`）、原地/异地、
填充模式（均匀/正态/Inf/NaN）、退化与负向用例（空指针、非法枚举、非法前导维、workspace 不足等）。
任务书正文写「200 条」，与随包 CSV 行数不一致，**验收口径以随包 CSV 为准**并在自测报告中说明。

当前验证状态（代码 `a373e39c348a7e0a107d368f7274426038532c92`，h54）：

全量 GTest **1220 条，失败 1、跳过 0**；严格精度 **1008/1009**（主路径 999/1000、BF16 9/9）；性能 **189/200**；workspace **200/200**。

| 项目 | h54 实测结果 |
| --- | --- |
| 环境 | HiDevLab A2 / Ascend 910B2（1 物理 NPU）/ CANN 9.1.0；openEuler 24.03 LTS-SP3，gcc/gfortran 12.3.1 |
| 源码与二进制 | `a373e39c348a7e0a107d368f7274426038532c92`，52 个变更文件 SHA-256 一致；build_h53 二进制采集前后哈希一致 |
| 全量覆盖 | 1220 条，失败 1、跳过 0；CSV 1209、补充 8、其他 3 |
| 严格精度 | 1008/1009；TC_FL_177 PASS，TC_FL_178 FAIL（非有限等效 250/256） |
| 性能 | 189/200；逐例 msprof，预热 5 + 有效采样 10，各 kernel 均值之和 ≤GPU 1.25× |
| workspace | 200/200；含 BF16 共 209 条契约、0 违例，最大 1.193840 |
| 内存插桩 | 原有三组 49+9+17 条 PASS；追加 TC_PF_1103 PASS，确认 pack 与 Cube 均实际插桩 |
| 代码 CI | run#2949 COMPLETED；8 组 A2/A5 编译、CodeCheck、PreSmoke_A2 均完成 |

CANN 9.1.0 兼容性已实测；910B2 对任务书 910B3 的性能适用性待确认。当前不可提交验收。

### 兼容性分析

- **新增**：`blasLt/matmul/fp32/arch22/`、`blasLt/matmul/epilogue/arch22/`、`test/blasLtMatmul/arch22/`；
- **不改动**：950（arch35）内核代码、矩阵变换算子、`include/` 下的公开声明；
- **行为变更**（均在 API 校验层，为对齐任务书 §2.5 的口径）：
  - `transA/transB = OP_C` 由「拒绝」改为「接受（等价 OP_T，实数路径）」；
  - 非法 `transA/transB` 枚举返回码由 `NOT_SUPPORTED` 改为 `INVALID_ENUM`；
  - `scaleType != ACL_FLOAT` 返回码由 `INVALID_VALUE` 改为 `NOT_SUPPORTED`；
  - 新增 `computeType` 白名单校验（此前未校验，混合精度会被静默按 FP32 计算）；
  - 新增 `C` 指针契约校验：`C == nullptr` 只在「`beta == 0`（C 非必需输入）」时合法。
    HOST 指针模式按 beta 的实际值判定（`beta != 0` 且 `C == nullptr` → `INVALID_VALUE`）；
    DEVICE 指针模式的 beta 位于设备内存，host 侧读取需要设备到主机拷贝与流同步，
    会破坏接口的异步契约，因此对 `C == nullptr` **保守拒绝**（`INVALID_VALUE`），
    DEVICE 模式下需传入合法的 C 指针。该约束在 `test/blasLtMatmul/arch22/README.md`
    的状态码表中逐条写明。
- **workspace 上报口径**：`aclblasLtMatmulAlgoGetHeuristic` 的上报值包含全部四项需求
  （重打包缓冲、split-K 部分和、原始乘积 staging、对齐余量）。原始乘积 staging
  （`m*n*4`）只在 epilogue 运行且 raw 缓冲不是 D 本身时出现（`C == D` 原地，
  或 D 的元素宽度与 FP32 原始乘积不同）；查询里只有布局、没有指针，无法判断调用
  是否原地，因此按最坏情况计入，调用方**只需**按上报值预留 workspace。总量仍受
  任务书 §3.4 的 1.2× 输入输出数据量上限约束：staging 与重打包缓冲之和放不下时
  以重打包让位（引擎退化为不重排，结果仍正确）。
- **engine 层**：`MatmulFp32TilingData` 在公共头中统一定义，包含 `lddRaw` / `productTermIsZero`。
  共享启动层在乘积为零时跳过不需要的乘积 kernel，使 arch35 也不读取空 A/B；arch35 内核代码未改。
  arch35 本轮通过 CI 编译，未进行实机功能或性能验证。

---

# 交付件与状态

> 设计文档 PR：https://gitcode.com/cann/cann-ops-competitions/merge_requests/2507
> （路径 `04_tasks/01_community-task-2026/tasklist/96a87ecddfe140e7bc29fc6959205ac5-aclblasLtMatmul-A2A3/JerryJiao0628/docs/design.md`）

| 序号 | 交付件 | 状态 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档 |
| 2 | 自测用例及测试代码 | `test/blasLtMatmul/arch22/`（CSV + 生成器 + 精度/性能校验脚本 + gtest 用例 + README） |
| 3 | 自测报告 | `aclblasLtMatmul_自测报告.md`（精度/性能/workspace 数据已回填） |
| 4 | 待验收代码地址 | 个人仓 `https://gitcode.com/JerryJiao0628/ops-blas`（fork 自 cann/ops-blas），分支 `feature/aclblasLtMatmul-arch22`，算子目录 `blasLt/matmul/fp32/arch22/` + `blasLt/matmul/epilogue/arch22/`，测试目录 `test/blasLtMatmul/arch22/`；已邀请账号 `Ascend-CANN` 为开发者（push 权限）。代码 PR：https://gitcode.com/cann/ops-blas/merge_requests/734 |

## 风险与待确认

1. **A3 的 arch 归属（已核实）**：主仓 `CMakeLists.txt` 的 `get_soc_arch_dirs` 把
   `ascend910b*`（A2）与 `ascend910_93*`（A3）**同时**映射到 `arch22`，`ascend950*` 走 `arch35`。
   即 A3 与 A2 共用本实现，无需复制代码；本次仅在 A2（Ascend 910B1）上实测，A3 的实机复测
   由验收方在同一分支上进行。
2. **OP_C 口径**：本实现按任务书 §2.1.2 放行 `OP_C`（实数等价 `T`），与仓内原 issue #595
   「显式拒绝」的策略相反，需在 PR 中说明。
3. **用例口径**：任务书正文 200 条 vs 随包 CSV 1200 条，以随包 CSV 为准。
4. **目录位置**：任务书 §5 写的是 `blas/lt_matmul/arch22` 与 `test/lt_matmul/`，而当前 ops-blas
   主仓的 blasLt 实现实际位于 `blasLt/matmul/<op>/<arch>/` 与 `test/blasLtMatmul/<arch>/`
   （arch35 的既有实现即在此处）。本实现跟随主仓现状放在 `blasLt/` 下，PR 时按主仓实际目录提交。
5. **算子目录与 950 PR 共存**：接口声明（`include/cann_ops_blasLt.h`）未改动，与 950PR 共用，
   未定义任何产品私有平行接口。
6. **未覆盖的精度档**：本实现只做 FP32 主路径（任务书 §2.1.3 明确只验收 FLOAT32），
   MXFP8/MXFP4 在 arch22 上返回 `NOT_SUPPORTED`。
