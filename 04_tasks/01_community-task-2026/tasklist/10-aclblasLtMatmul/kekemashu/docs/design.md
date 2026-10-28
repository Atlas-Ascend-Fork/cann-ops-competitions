# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务书《aclblasLtMatmul 算子开发任务书》（`10-aclblasLtMatmul`）。目标代码仓为昇腾算子开源仓
`ops-blas`（https://gitcode.com/cann/ops-blas ），测试代码合入该仓 `test/blasLtMatmul/`。

本任务为补齐 Blas 仓能力：在 Atlas A2/A3 系列产品上使用 Ascend C 编程语言实现单精度实数 cuBLASLt 风格矩阵乘算子
`aclblasLtMatmul`。接口签名以 `include/cann_ops_blasLt.h:909-924` 既有声明为准，与现网 / 950PR 共用同一声明，
禁止定义产品私有平行接口。

> 目录落位说明：任务书 §2.2/§5 写的是 `blas/lt_matmul/arch22` 与 `test/lt_matmul/`，但当前 ops-blas 主干已完成
> blasLt 重构——`aclblasLtMatmul` 的 API / 引擎 / 实现全部位于 `blasLt/`（`blas/lt_matmul` 已不存在），
> `blasLt/CMakeLists.txt` 按 `<子系统>/<精度>/<archXX>/` 组织并按 SOC 裁剪编译。本实现按主干约定落位：
> 实现代码 `blasLt/matmul/fp32/arch22/`，测试工程 `test/blasLtMatmul/arch22/`；若合入时仓库再次调整目录，
> 按同一“按架构分流”原则平移即可。

## 背景介绍

### aclblasLtMatmul 算子实现补齐

`aclblasLtMatmul` 对齐 cuBLAS `cublasLtMatmul` 的核心功能与参数语义，计算：

```text
D = alpha * op(A) * op(B) + beta * C
```

其中 alpha、beta 为标量（类型对应 `computeType` / `scaleType`），A、B、C、D 为矩阵，D 可与 C 原地（C == D）
或异地（C != D）计算。本任务仅支持单 batch（各 layout `batchCount = 1`、`stridedBatchOffset = 0`）。

工程形态为 Ascend C kernel 直调：基于 ops-blas 工程框架实现句柄式 BLAS 接口，通过 handle 绑定 stream 直调
NPU kernel；workspace 由调用方分配并通过 `workspaceSizeInBytes` 传入。

### 现状分析

| 产品 | 现状 |
| --- | --- |
| Ascend 950PR/950DT（arch35） | 已实现 FP32 / MXFP8 / MXFP4 路径：MMAD kernel 产出原始乘积到 workspace 或 D，再由独立 epilogue kernel 完成 `alpha*raw + beta*C` |
| Atlas A2/A3（arch22） | **实现缺失**：`blasLt/matmul/fp32/` 只有 arch35 目录，A2/A3 目标下 `aclblasLtMatmul` 无可用内核 |

因此本任务不是“优化已有 TBE 算子”，而是为已有公共接口补齐 Atlas A2/A3 后端实现，并保证与 950 共用同一套
API 校验、描述符与引擎框架。

### 算子功能分析

| 参数 | 参数含义 | 数据类型 | 约束 |
| --- | --- | --- | --- |
| alpha / beta | 缩放标量 | FLOAT32（Host 或 Device 内存） | 不可为 nullptr |
| A | 输入矩阵，op(A) 为 m×k | FLOAT32（Device 只读） | 需计算时不可为 nullptr |
| B | 输入矩阵，op(B) 为 k×n | FLOAT32（Device 只读） | 需计算时不可为 nullptr |
| C | 累加矩阵，m×n | FLOAT32（Device 只读） | `beta != 0` 时不可为 nullptr；`beta == 0` 时可为 nullptr 且不读取 |
| D | 输出矩阵，m×n | FLOAT32（Device 覆写） | `m > 0 && n > 0` 时不可为 nullptr |
| computeDesc | 运算描述符（transA/transB、computeType、scaleType、pointerMode） | attr | 不可为 nullptr |
| Adesc/Bdesc/Cdesc/Ddesc | 布局描述符（dtype/rows/cols/ld/order/batch） | attr | 均不可为 nullptr；order 必须 ROW；单 batch |
| algo | 算法句柄 | attr | 可为 NULL（隐式 heuristics） |
| workspace / workspaceSizeInBytes | 调用方 workspace | scalar | 非空须 16B 对齐；`algo` 非空时容量须 ≥ `algo->max_workspace_bytes` |
| stream | 执行流 | scalar | 已创建 stream 或 nullptr |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上使用 Ascend C 实现 `aclblasLtMatmul` 的 FLOAT32 主路径，与 cuBLAS `cublasLtMatmul` 核心功能与
参数语义对齐：支持 N/T/C 转置组合、C == D 原地与 C != D 异地、padding 前导维、alpha/beta 任意取值、
`m=0/n=0/k=0/alpha=0` 边界行为，并满足任务书规定的精度、性能、内存与自验要求。

## 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 接口与 `include/cann_ops_blasLt.h:909-924` 一致，与 950PR 共用，不新增私有接口 | 代码审查；接口声明比对 |
| R2 | 实现 `D = alpha * op(A) * op(B) + beta * C`，单 batch，FLOAT32 主路径 | 功能用例；golden 由 cblas `cblas_sgemm` 生成 |
| R3 | transA/transB ∈ {N, T, C}，实数路径下 C 与 T 等价 | 转置组合用例 |
| R4 | C == D 原地与 C != D 异地；原地时 Cdesc 传与 Ddesc 相同布局 | 原地/异地用例 |
| R5 | 边界行为对齐 cuBLAS/Netlib quick return：m=0/n=0 no-op；k=0 只做 beta 缩放；alpha=0 不引用 A/B | 边界用例；`alpha=0` 且 `beta=1` 位精确 |
| R6 | 返回码语义与校验顺序符合任务书 §2.4/§2.5 | 负向用例状态码断言 |
| R7 | workspace 由调用方分配，非空 16B 对齐，容量满足启发式返回值 | workspace 负向用例；峰值统计 |
| R8 | 精度满足 FLOAT32 生态标准（含 Inf/NaN 判定规则与位精确特例） | 1000 条精度用例 |
| R9 | 性能达到 GPU 标杆 0.8 倍以上（平均单次耗时 ≤ 标杆 1.25 倍） | msprof op 采样；自测报告 |
| R10 | workspace 峰值 ≤ 输入输出数据总量的 1.2 倍 | 自测报告 workspace 数据 |
| R11 | 自验覆盖任务书全部用例（本包 1200 条：1000 精度 + 200 性能） | 测试工程 + 自测报告 |
| R12 | 交付设计文档 PR、自测用例与测试代码、自测报告、待验收代码地址；README 标注 A2/A3 支持 | 验收交付件审查 |

# 详细设计（required）

## 算子分析

### 数学公式

```text
D = alpha * op(A) * op(B) + beta * C
```

- `op(A)` 为 m×k、`op(B)` 为 k×n、C/D 为 m×n；四个 layout 固定 `order = ACLBLASLT_ORDER_ROW`，`ld` 为行宽；
- 转置语义：`N` → op(X)=X；`T` → op(X)=Xᵀ；`C` → op(X)=Xᴴ（实数路径与 T 等价）；
- 边界：`m=0||n=0` no-op 返回 SUCCESS；`k=0` 只做 `beta*C`；`alpha=0` 不引用 A/B（`beta=1` 时 D=C 位精确）。

### 支持数据类型

主路径 FLOAT32：`Atype/Btype/Ctype = ACL_FLOAT`，`computeType = ACLBLAS_COMPUTE_32F`（含 `_PEDANTIC`、
`_FAST_TF32`），`scaleType = ACL_FLOAT`。混合精度经 computeType/scaleType 描述，不在本任务范围：A2/A3 上
非 FP32 dtype 组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（950 的 MXFP8/MXFP4 路径保持不变）。

### 支持形状

m、n、k ≥ 0，`lda ≥ max(1,k)`（transA=N）或 `≥ max(1,m)`（否则）；`ldb ≥ max(1,n)`（transB=N）或 `≥ max(1,k)`；
`ldc/ldd ≥ max(1,n)`；`ld == 0` 按行宽归一。单 batch。不要求 dynamic shape（m/n/k 为运行时入参）。
A2/A3 FP32 路径单次 K 归约上限 8192（静态 tiling 决定），超出返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

### 接口与参数

接口签名以 `include/cann_ops_blasLt.h:909-924` 为准（16 参，句柄 `aclblasLtHandle_t`、`aclrtStream` 为平台类型映射）。
参数语义见上文“算子功能分析”。校验顺序（任务书 §2.5）：

```text
handle
→ 描述符判空（computeDesc/Adesc/Bdesc/Cdesc/Ddesc）
→ alpha/beta 判空
→ pointerMode
→ transA/transB 枚举（非法 → INVALID_ENUM）
→ 取 m/n/k
→ m == 0 || n == 0 → SUCCESS
→ computeType/scaleType 白名单（NOT_SUPPORTED）
→ dtype 白名单（NOT_SUPPORTED）
→ 布局合法性（维度/ld/batch/整数范围 → INVALID_VALUE）
→ order 必须 ROW（否则 NOT_SUPPORTED）
→ A/B/D 判空、beta != 0 且 C 为空（INVALID_VALUE）
→ workspace 对齐与容量（INVALID_VALUE）
```

## 算子实现

### 实现方案

整体分层与 950 共用，仅 FP32 计算流水按架构分流：

```text
include/cann_ops_blasLt.h                   既有声明（两产品共用）
        │
blasLt/api/aclblaslt_matmul.cpp             参数校验 + MatmulProblem 组装（产品无关）
        │
blasLt/matmul/engine/matmul_engine.cpp      dtype 路由：FP32 → MatmulLaunchFp32(problem)
        │
        ├── blasLt/matmul/fp32/arch22/      Atlas A2/A3：原始乘积 kernel + 向量 epilogue kernel（本次新增）
        └── blasLt/matmul/fp32/arch35/      Ascend 950：沿用既有两段式流水（逻辑原样搬到 arch35 自己的入口）
```

`CMakeLists.txt` 只把与当前 SOC 匹配的 `blasLt/matmul/fp32/<arch>/` 目录加入头文件搜索路径，其余子系统
（MXFP8/MXFP4、matrix_transform）布局不变；`blasLt/CMakeLists.txt` 的 glob 按 arch 目录自动裁剪源文件。

#### Host 侧设计

1. **参数校验与描述符解析**（`blasLt/api/aclblaslt_matmul.cpp`）：按上述顺序校验；从 computeDesc 读取
   transA/transB/computeType/scaleType/pointerMode，从四个 layout 读取 dtype/order/ld/batch；`ld == 0`
   归一为物理行宽；组装 `MatmulProblem`（同时携带 handle，供库内暂存区使用）。
2. **路径选择**（`blasLt/matmul/fp32/arch22/matmul_fp32_host.cpp`）：按 alpha/beta/k/C==D 选 4 条路径：

   | 条件 | 乘积 kernel | epilogue kernel | 原始乘积落点 |
   | --- | --- | --- | --- |
   | `k>0 && alpha!=0`，`C!=D` 或 `beta==0` | 执行 | `alpha==1 && beta==0` 时跳过 | D（按 `ldd` 写出，支持 padded D） |
   | `k>0 && alpha!=0`，`C==D && beta!=0` | 执行 | 执行 | workspace（紧凑 `n`）/ 库内暂存区 |
   | `k==0` 或 `alpha==0` | 跳过 | 执行（beta 缩放 / 置零 / 拷贝） | 不使用 |
   | `k==0/alpha==0 && beta==1 && C==D` | 跳过 | 跳过（D 已等于 C） | 不使用 |

3. **Tiling**：采用编译期静态 tiling（`GetMMConfig<CONFIG_NORM>` + `GetMatmulApiTiling`），参数沿用仓内已验证的
   arch22 sgemm 配置：单 block 输出 256×256，baseM/baseN/baseK = 64/256/32（TN 用 baseK=64），singleCoreK=8192，
   `baseM×baseN×4B = 64KB` L0C 留有余量。不引入 `MultiCoreMatmulTiling`/`platform_ascendc`，避免给
   `ops_blasLt` 增加 `tiling_api`/`platform` 链接依赖。
4. **分核**：输出 2D 分块轮转（`tile = blockIdx, blockIdx + blockNum, …`），block 数取
   `ACL_DEV_ATTR_CUBE_CORE_NUM`（handle 内 `QueryCubeCoreNum`）。4096² 时 256 个 tile / 20 核 ≈ 13 波。
5. **workspace**：`aclblasLtMatmulAlgoGetHeuristic` 统一预留 `m*n*4`（原始乘积暂存上界），调用方按启发式返回值
   申请即可覆盖全部 alpha/beta 组合；原地且调用方未传 workspace 时回退到句柄内缓存的暂存区。
6. **下发**：`<<<blocks, nullptr, stream>>>` 直调，异步执行；调用方读回前同步 stream。

#### Kernel 侧设计

**matmul kernel（AIC，`matmul_fp32_kernel.cpp`）**

- 四个 kernel 入口对应 `(transA, transB) ∈ {N, T}²`（实数下 OP_C 走转置分支），按 tiling 标志选择；
- 每个 block 认领 256×256 输出块，块内处理尾块（`rows/cols`）；
- 操作数绑定：非转置 A `[m][lda]` 原点偏移 `mBase*lda`，转置 A `[k][lda]` 原点偏移 `mBase`；B 恒偏移 `nBase`；
- `SetOrgShape(rows, rawLd, lda, ldb, rawLd)`：`orgKa/orgKb` 传物理行宽，`orgN/orgKc` 传输出行宽，
  使 padding 前导维与 padded D 直接生效；`SetSingleShape(rows, cols, k)` 约束实际 tile；
- `SetHF32Mode(HF32Mode::DISABLE)`：避免 FP32 Cube 退化为 HF32，保证长 K 归约精度。

**epilogue kernel（AIV，`matmul_fp32_epilogue_kernel.cpp`）**

- 每个 block 认领连续行区间，行内按 4096 元素分块，`DataCopyPad` 处理任意长度（无需 32B 对齐）；
- 逐行“读 raw（如需）→ 读 C（beta!=0）→ Muls/Add → 写 D”，同样行内先读后写，因此异地 raw 落在 D、
  以及 C == D 原地都安全，且允许 `ldc != ldd`；
- 模式：`D=0`、`D=C`（位精确）、`D=beta*C`、`D=alpha*raw`、`D=alpha*raw+beta*C`；
- `pointerMode = DEVICE` 时先读 4 字节设备标量再计算，与 950 行为对齐。

#### 原地/异地与 workspace

- C == D 原地：乘积不得覆盖 C，故先落到 workspace（调用方）或库内暂存区，再由 epilogue 读 C、写 D；
- C != D 异地：乘积直接写 D（按 `ldd`），epilogue 按行原地合并，不额外分配；
- workspace 峰值 = `m*n*4`（仅原地/窄输出场景），≤ 输入输出总量 1.2 倍。

#### 精度设计

| 场景 | 处理 |
| --- | --- |
| 一般乘加 | Cube FP32 累加（HF32 关闭）+ epilogue 单次乘/加 |
| `alpha==0` 或 `k==0` 且 `beta==1` | 跳过乘积，`D = beta*C` 位精确 |
| `beta==0` | 完全不读 C（C 填 NaN 也不污染结果） |
| `k==0` | 不启动 matmul，按 `beta*C` 或置零写出 |
| Inf/NaN | 原样传播，按任务书非有限值等价规则判定 |

#### 性能与内存设计

- 计算由 Cube 完成（`IterateAll`，API 内部完成 L1→L0 双缓冲与 fixpipe 搬出），Host 只做 kernel 参数组装；
- 额外开销：epilogue 多一趟 raw 读 + C 读 + D 写（约 3×m×n×4B）；4096³（≈7.2ms 基线）下约 0.2ms；
- 小尺寸下为一次 kernel 启动开销；后续可按官方 `matmul_fused` 模式（`GetTensorC` 输出到 UB）融合 epilogue 进一步优化；
- 内存：workspace 峰值 ≤ 1.2× 输入输出总量；不使用额外内部缓冲（除非调用方未提供 workspace 且原地计算）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800T A2 / 910B3） | √ |
| Atlas A3 系列产品（含 Atlas 800T A3 / 910_93） | √ |

自验证覆盖一种款型即可：精度建议 910B3 或 A3，性能建议 910B3。README 产品支持表标注 Atlas A2/A3 系列产品：支持。

## 算子约束限制

- 仅单 batch（`batchCount = 1`、`stridedBatchOffset = 0`）；
- 四个 layout 固定 `order = ACLBLASLT_ORDER_ROW`；`ld == 0` 按行宽归一；不支持超出 ld 语义的非连续访存；
- A2/A3 仅 FLOAT32 主路径；非 FP32 dtype、非 FP32 computeType/scaleType 返回 `NOT_SUPPORTED`；
- A2/A3 FP32 单次 K 归约上限 8192；
- 支持 C == D 原地（Cdesc 须传与 Ddesc 相同布局，不得省略），不返回视图；
- `beta != 0` 且 C 为 nullptr 返回 `INVALID_VALUE`；`beta == 0` 时 C 可为 nullptr 且不读取；
- workspace 非空须 16B 对齐，容量须 ≥ `algo->max_workspace_bytes`；
- 主路径对齐 `EPILOGUE_DEFAULT`（Bias/ReLU/GELU 不在本任务范围）；
- 禁止定义产品私有平行接口。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib BLAS 3.10.0 `cblas_sgemm`）单标杆生成，全 D 矩阵比对；逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`，`atol = rtol = 2^-13`；要求有限值 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max(1e-2, 32 ULP)` | 任务书 §3.2；生态算子开源精度标准 |
| Inf/NaN 规则 | 同为 NaN、同为同号 Inf、一方 NaN 一方 ±Inf 视为等效通过；一方有限另一方非有限不通过；非有限值不计入统计，需全部通过 | 任务书 §3.2 |
| 位精确特例 | `alpha == 0`（或 `k == 0`）且 `beta == 1` 时跳过矩阵乘，`D = C` 位精确 | 任务书 §3.2 |
| 性能标准 | Atlas 800T A2 (910B3)，FLOAT32 平均单次耗时 ≤ GPU 标杆 1.25 倍（≥ 0.8× 标杆）；msprof op 采集 task duration，5 次 warmup，10 次有效样本取平均 | 任务书 §3.3 |
| 内存标准 | workspace 峰值 ≤ 输入输出数据总量 1.2 倍；非空 workspace 须 16B 对齐 | 任务书 §3.4 |

自验方式：使用 ops-blas 仓 `test/blasLtMatmul/arch22/` 测试工程，CSV 描述用例（`blasLtMatmul_test.csv`，
1000 精度 + 200 性能），`verify_accuracy.py` / `verify_performance.py` 驱动；覆盖尺寸扫描（1~4096、质数、
2 的幂 ±1、非对齐）、9 种转置组合、原地/异地、padding 前导维、均匀/正态/极端/Inf/NaN 填充、49 条边界负向、
200 条性能用例；README 说明复现步骤。

## 兼容性分析

- 本算子为已有接口的 A2/A3 实现补齐，接口声明与 950PR 共用，不定义私有平行 API，不改变既有声明语义；
- 公共 API 校验、`MatmulProblem`、引擎路由为两产品共用；950 的 FP32 两段式流水原样保留（仅从引擎搬到
  `arch35/matmul_fp32_launch.cpp`），MXFP8/MXFP4 路径不受影响；
- `ACLBLAS_OP_C` 在实数 dtype 下与 `T` 等价，放开后对两产品 FP32 路径均正确（此前被拒绝）；
- `aclblasLtMatmulAlgoGetHeuristic` 的 `workspaceSize` 改为预留 `m*n*4`（原地/窄输出暂存上界）：调用方按返回值
  申请即可，属于“更保守的报告值”，不破坏既有调用方式；
- README 产品支持表更新为 A2/A3 支持，并补充约束（K ≤ 8192、C==nullptr 语义、workspace 建议）；
- 不涉及旧版本算子行为兼容；PR 需附带自验结果，遵循 `CONTRIBUTING.md`，不含冗余文件。

## 验收交付件

| 序号 | 交付件 | 状态 |
| --- | --- | --- |
| 1 | 设计文档（本文） | `cann-ops-competitions`（cann-competitions 链接重定向到此仓）`04_tasks/01_community-task-2026/tasklist/10-aclblasLtMatmul/kekemashu/docs/design.md` |
| 2 | 自测用例及测试代码 | `ops-blas`: `test/blasLtMatmul/arch22/`（测试工程 + CSV + README 复现步骤） |
| 3 | 自测报告 | 按模板填写：用例参数、精度结果与截图、性能数据（10 次有效采样均值）与截图、workspace 峰值 |
| 4 | 待验收代码地址 | 个人仓链接 + 分支 + 算子目录；邀请 `Ascend-CANN` 为开发者；README 标注 A2/A3 支持 |
| 5 | 合入 PR | 测试通过后向 `ops-blas` 提交 PR（合入 `blasLt/matmul/fp32/arch22/` 与 `test/blasLtMatmul/arch22/`） |
