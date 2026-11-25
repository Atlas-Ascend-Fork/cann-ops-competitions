# 需求背景

## 需求来源

CANN 社区任务 2026 年 11 月任务：在昇腾 950PR（arch35）上使用 Ascend C 开发单精度复数对称矩阵秩-k 更新
（扩展版，3M 算法）算子 `aclblasCsyrk3mEx`，补齐 ops-blas 仓 Level-3 复数 syrk 族的 3M 扩展接口，
验收通过后合入 `cann/ops-blas` 主仓。

对标接口为 cuBLAS `cublasCsyrk3mEx`，参数序列一一对应（handle 及 12 个参数逐个对应，无需额外映射）。

## 背景介绍

### 算子功能

列主序原地计算：

```
C = alpha * op(A) * op(A)^T + beta * C
```

- `trans = ACLBLAS_OP_N`：op(A) = A，A 为 n×k、lda ≥ max(1,n)；
- `trans = ACLBLAS_OP_T`：op(A) = A^T，A 以 k×n 存储、lda ≥ max(1,k)；
- `trans = ACLBLAS_OP_C`：本算子为**对称**语义而非 Hermitian 语义，按 OP_T 等价处理，不做共轭；
- C 为 n×n 复数对称矩阵，列主序，**仅 uplo 指定三角被读写**，另一三角由对称性 C = C^T 隐含；
- 对称语义下对角元素 C(i,i) 的虚部是有效数据，**计算后原样保留，不置 0**；
- alpha、beta 为单精度复数标量（`aclblasComplex`，实部/虚部各 float32），Host 或 Device 指针均支持；
- Atype ∈ {`ACL_COMPLEX64`, `ACL_INT8`}，Ctype = `ACL_COMPLEX64`；`ACL_INT8` 表示 complex-int8，
  每个复元素按 `(int8 real, int8 imag)` 交错 2 字节存储（对应 cuBLAS `CUDA_C_8I`），在 kernel 内精确提升为 float。

### ops-blas 仓现状分析

本算子建立在仓内两块成熟实现之上：

| 组件 | 位置 | 现状 |
| --- | --- | --- |
| 非 Ex 复数 syrk | `blas/syrk/arch35/csyrk_host.cpp`、`csyrk_kernel.cpp`、`csyrk_cube_kernel.cpp`、`csyrk_direct_kernel.cpp` 等 | 已合入。三段式流水：AIV 去交错（Ar/Ai）→ AIC **融合 4M** cube（Q0=Ar·Ar^T、Q1=Ai·Ai^T、Q2=Ar·Ai^T、Q3=Ai·Ar^T 四个积在同一 kernel 内完成）→ AIV 合成；另有 SIMT 小尺寸直算、HF32x3 补偿、特殊值 host 回退 |
| 复数 3M GEMM | `blas/gemm/arch35/cgemm3m_host.cpp` → `LaunchComplexGemm(..., true)`；`cgemm3m_rounding_kernel.cpp` | 已合入。仓内已具备"一次 kernel 下发内完成三次实数乘 + 舍入修正"的完整先例，证明 cube 单元与管线可承载 3M 结构 |
| 复数 syrk 公共设施 | `blas/common/helper/syrk_gemm_arch35.h`（四积融合 tile 驱动）、`syrk_host_utils.h`（核数选取）、`ssyrk_tiling_data.h`（tile 常量） | 已合入，可直接复用或按 3M 结构改造 |
| 头文件 | `include/cann_ops_blas.h` | 已有 `aclblasCsyrk`（4M 直接实现），**没有** 任何 Ex 后缀的 syrk 接口 |

syrk 族常量沿用：`SYRK_ARCH35_BASE_M/N = 16`、`BASE_K = 8`、默认 tile 128×128、K 分块 256、
L1 512KB（使用率 0.9、双 bank ping/pong）。

### 缺口分析

1. 公共头 `include/cann_ops_blas.h` 没有 `aclblasCsyrk3mEx` 声明；
2. 现有 csyrk 是 **4M** 分解（4 个实数积），本任务要求 **3M**（3 个实数积：P0=Ar·Ar^T、P1=Ai·Ai^T、
   P2=S·S^T，其中 S=Ar+Ai），需要物化 S 面板并新增 3 积融合驱动（现有四积驱动按 {Ar,Ai}×{Ar,Ai}
   选择左右面板，无法表达 S 面板）；
3. 现有去交错内核只处理 complex64 输入且只产出 Ar/Ai；需支持 complex-int8（2 字节交错）精确提升，
   并在同一遍内产出 S=Ar+Ai；
4. 现有合成内核实现 4M 的 `(Q0−Q1, Q2+Q3)` 组合；3M 需 `(P0−P1, P2−P0−P1)`，且虚部
   `P2−P0−P1 ≡ Q2+Q3` 的消去放大问题更大，对角元素需更早改走直接归约；
5. 缺少 `test/syrk/csyrk3m_ex/` 测试工程与 1200 条 CSV 用例。

# 需求分析

## 需求描述

新增句柄式 BLAS 接口（与任务书 §2.3 逐参数一致）：

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

通过 handle 绑定的 stream 直调 NPU kernel，声明与其他产品线共用，不定义私有平行 API。
本实现**不依赖** `aclblasCsyrkEx`（同为 1125 新增的另一任务接口），只参考非 Ex `aclblasCsyrk`
的参数校验/三角处理与 `cgemm3m` 的 3M 分解。

## 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | `include/cann_ops_blas.h` 新增 `aclblasCsyrk3mEx` 声明 | 头文件审查、外部工程 include 后可编译 |
| R2 | host 入口：参数校验、no-op 短路、tiling、kernel 下发；返回码逐条符合任务书 §2.4 | 负向用例逐条匹配 |
| R3 | 3M 分解：P0/P1/P2 三次实数积在**同一次 kernel launch** 内完成，不得用三次 GEMM 调用拼接 | 代码审查 + `TC_3M_*` 用例 |
| R4 | Atype 支持 (COMPLEX64, INT8)×COMPLEX64 两种组合，complex-int8 精确提升；其余组合 `NOT_SUPPORTED` | dtype 负向用例 |
| R5 | 对称语义：仅 uplo 三角读写、OP_C≡OP_T、对角虚部保留、未选三角/padding 不被改写 | 哨兵写保护 + 对角虚部专项用例 |
| R6 | 边界：`n=0` 合法 no-op；`k=0`/`alpha=(0,0)` 时 beta=(1,0) quick return、否则只做 `C=beta·C` 且不引用 A | TC_ED 用例 |
| R7 | 精度达生态算子开源标准 FLOAT32 档（实部/虚部分别判定） | §3.2 阈值 |
| R8 | 性能：kernel-to-kernel 口径单 case ≥ 0.4× GPU、平均 ≥ 0.5× | `msprof op` Task Duration + `gpu_baseline.csv` |
| R9 | 交付 1200 条 CSV 用例（1000 精度 + 200 性能）与可复现测试工程 | 交付件 + 自测报告 |

# 详细设计

## 算子分析

### 数学公式

记 X = op(A) = Xr + i·Xi（n×k）。对 uplo 三角内 (i,j)：

```
Z(i,j) = Σ_{t=0}^{k-1} X(i,t) · X(j,t)          （复数乘加，T 为普通转置而非共轭）
C(i,j) = alpha · Z(i,j) + beta · C_old(i,j)
```

复数展开（对称，非 Hermitian）：

```
X·X^T = (Xr·Xr^T − Xi·Xi^T) + i·(Xr·Xi^T + Xi·Xr^T)
```

这与 Hermitian 的 `X·X^H = (Xr·Xr^T + Xi·Xi^T) + i·(Xi·Xr^T − Xr·Xi^T)` 不同，不可套用。

**3M 分解**（Gauss 三次乘法）：

```
P0 = Xr·Xr^T
P1 = Xi·Xi^T
P2 = (Xr+Xi)·(Xr+Xi)^T

Re = P0 − P1
Im = P2 − P0 − P1        （恒等于 Xr·Xi^T + Xi·Xr^T）
Z  = Re + i·Im
```

alpha 作用（复数标量 alpha = ar + i·ai）：

```
C.real = ar·Re − ai·Im + beta·C_old
C.imag = ar·Im + ai·Re + beta·C_old
```

对角元素 Im(i,i) = 2·Σ_t Xr(i,t)·Xi(i,t)，为有效数据，不置 0。

### 支持数据类型

| 角色 | 类型 | 说明 |
| --- | --- | --- |
| A | `ACL_COMPLEX64` | complex64，8 字节/元素，实虚交错 |
| A | `ACL_INT8` | complex-int8，2 字节/元素，(int8 real, int8 imag) 交错；kernel 内精确提升为 float |
| C | `ACL_COMPLEX64` | 仅 Ctype = COMPLEX64 |
| alpha、beta | COMPLEX64（`aclblasComplex`） | 实部/虚部各 float32，Host 或 Device 指针 |
| 累加 | FLOAT32 | 内核与 golden 口径一致 |

### 支持形状与参数约束

| 参数 | 约束 | 违反时返回 |
| --- | --- | --- |
| handle | 非空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | ∈ {UPPER, LOWER} | `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | ∈ {N, T, C} | `ACLBLAS_STATUS_INVALID_ENUM` |
| n | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| k | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | ≥ max(1, n)（trans=N）或 max(1, k)（trans=T/C） | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | ≥ max(1, n) | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha / beta | 非空（先于一切短路） | `ACLBLAS_STATUS_INVALID_VALUE` |
| Atype | ∈ {COMPLEX64, INT8}，Ctype = COMPLEX64 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| A | n>0 且 k>0 时非空（见下方口径说明） | `ACLBLAS_STATUS_INVALID_VALUE` |
| C | n>0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |

补充语义：

- `n = 0`：合法 no-op，返回 SUCCESS，不引用 A/C、不启动 kernel；
- `k = 0` 或 `alpha = (0,0)`：不引用 A；`beta = (1,0)` 时 quick return（C 不变），
  否则只对 uplo 三角做 `C = beta·C`；
- `beta = (0,0)` 时不读 C 旧值（C 可以是任意无效数据）。

## 算子实现

### 总体方案

沿用 csyrk 的三段式流水，把"4 积"替换为"3 积"，并新增输入类型提升与 S 面板：

```
aclblasCsyrk3mEx (host, 新增)
  ├─ 参数校验（Atype/Ctype 组合、校验顺序按任务书 §2.5）
  ├─ 标量解析（alpha/beta 的 Host/Device 位置判定，成对同位置）
  ├─ 短路：n==0 → SUCCESS；k==0 / alpha==(0,0) → beta-only 或 quick return
  ├─ 特殊值回退（n*k <= 128*128 时 D2H 扫描，见 7 节）
  ├─ tiling 计算（与 csyrk 同常量体系）
  ├─ workspace 准备（右面板 5 平面 [+ alpha 左面板 3 平面] + 2 个 temp quad）
  └─ kernel 下发：
        Stage 0  csyrk3m_ex_deinterleave_kernel  (AIV)  A -> 右面板（int8 精确提升），并按需生成 alpha 左面板
        Stage 1  csyrk3m_ex_cube_kernel          (AIC)  单 launch 完成 P0/P1/P2 三个实数积，符号累加进 Re/Im 两个累加器
        Stage 2  csyrk3m_ex_combine_kernel       (AIV)  C = (Re, Im) + beta*C_old
```

### 文件与目录

| 文件 | 职责 | 类型 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 新增 `aclblasCsyrk3mEx` 声明（置于 `aclblasCsyrk` 附近） | 修改 |
| `blas/syrk/README.md` | 新增 `aclblasCsyrk3mEx` 章节（产品支持、参数、约束、调用示例） | 修改 |
| `blas/syrk/arch35/csyrk3m_ex_host.cpp` | 公开 API 入口、校验、标量解析、tiling、workspace、下发 | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.cpp` | 三个 stage 的 kernel（去交错/面板、双累加器 3M cube、合成） | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.h` | launcher 声明（host/kernel 共用） | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_tiling_data.h` | tiling 结构（host/kernel 共用） | 新增 |
| `test/syrk/csyrk3m_ex/CMakeLists.txt` | 测试目标注册 `csyrk3m_ex_test` | 新增 |
| `test/syrk/csyrk3m_ex/csyrk3m_ex_param.h` | CSV 参数结构（含 Atype/Ctype、四列复数标量） | 新增 |
| `test/syrk/csyrk3m_ex/csyrk3m_ex_golden.h` | CBLAS golden、int8 提升、实/虚判定与哨兵写保护 | 新增 |
| `test/syrk/csyrk3m_ex/README.md` | 复现步骤 | 新增 |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test.cpp` | GTest 用例驱动（含 `[PERF]` 输出） | 新增 |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_npu_wrapper.h` | 设备缓冲、int8/复合输入装配 | 新增 |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test.csv` | 1200 条用例（随任务包提供） | 新增 |
| `test/syrk/csyrk3m_ex/arch35/gpu_baseline.csv` | 200 条性能基线 | 新增 |

落位说明：

1. 实现目录与测试目录均按任务书 §2.2 指定（`blas/syrk/arch35/`、`test/syrk/csyrk3m_ex/arch35/`），
   也与仓内 csyrk 同目录惯例一致；
2. 构建无需修改 `blas/CMakeLists.txt`：单算子过滤规则按"目录名或文件名前缀匹配 `--ops`"保留源码，
   新文件统一以 `csyrk3m_ex_` 前缀命名即可被 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 收集；
3. 双累加器 3M cube 驱动放在 `csyrk3m_ex_kernel.cpp` 内（header-only 的 L1/L0 辅助模板），
   复用 `csyrk_gemm_fused.h` 的 `SyrkGemmFusedFlagsInit/Drain/TileSelect`，不改动 csyrk 使用的
   `syrk_gemm_arch35.h` / `csyrk_gemm_fused.h`。

### host 侧设计

#### 1. 类型与枚举处理

- `Atype/Ctype` 直接使用 ACL 的 `aclDataType` 枚举（与 `aclblasGemmEx`/`aclblasCgemmEx` 同风格）；
- 支持集判定：`(Atype == ACL_COMPLEX64 || Atype == ACL_INT8) && Ctype == ACL_COMPLEX64`，
  其余返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（如任务包中的 `ACL_FLOAT` 负向用例）；
- `ACL_INT8` 按 complex-int8 解释，元素大小 2 字节，`lda`/`ldc` 仍以元素计；
- trans=C 在 host 入口即规范化为 T（与 csyrk `PrepareCsyrkParams` 一致）。

#### 2. 参数校验

校验顺序严格按任务书 §2.5，与 `aclblasCsyrk` 对齐：

1. `handle` 判空 → `HANDLE_IS_NULLPTR`；
2. `uplo`、`trans` 枚举合法性 → `INVALID_ENUM`；
3. `n`、`k` 非负 → `INVALID_VALUE`；
4. `lda`/`ldc` 下界（按 trans 区分）→ `INVALID_VALUE`；
5. `alpha`、`beta` 非空 → `INVALID_VALUE`；
6. `Atype/Ctype` 组合 → `NOT_SUPPORTED`；
7. `A`、`C` 指针（按维度条件）→ `INVALID_VALUE`；
8. 全部通过后才执行 quick return / no-op。

**口径说明（任务书内部不一致，需评审确认）**：

- 任务书 §2.4 参数表规定"n > 0 且 k > 0 时 A 为 nullptr 返回 INVALID_VALUE"（不区分 alpha）；
- 任务书 §2.5 校验顺序说明又写"k = 0 或 alpha = (0,0) 时 A 可为 NULL"。

两者在"alpha=(0,0) 且 k>0"的交叉场景冲突。本设计**采用 §2.4 表的严格口径**（与仓内
`aclblasCsyrk` 现行为一致：`A` 在 n>0 且 k>0 时必须非空，不论 alpha；**该口径已确认采用**）。
依据：任务包 CSV 用"alpha=(0,0) 且 A 为 NaN"而不是 NULL 来表达"不引用 A"，说明验收用例同样假定
A 指针有效；k=0 时 A 仍允许为空（与 csyrk 一致）。若评审要求放宽到 §2.5 字面口径，改动仅是校验
条件一处，审计记录见 README「与任务书口径的差异说明」。

#### 3. 执行路径与短路

| 条件 | 行为 | 返回 |
| --- | --- | --- |
| `n == 0` | 不访问 A/C，不下发 kernel | `SUCCESS` |
| `k == 0` 或 `alpha == (0,0)`，且 `beta == (1,0)` | quick return，C 不变 | `SUCCESS` |
| `k == 0` 或 `alpha == (0,0)`，且 `beta != (1,0)` | beta-only：仅对 uplo 三角做 `C = beta·C`（combine kernel 的 betaOnly 分支），不需要 workspace | `SUCCESS` |
| 其余 | Stage 0/1/2 主路径 | `SUCCESS` |

#### 4. tiling 设计

tiling 结构分三组：

| 组 | 字段 | 取值 |
| --- | --- | --- |
| 去交错/提升 | `nRows/nCols/lda/outLdc/colsPerCore/aType/alphaScaled/alphaReal/alphaImag` | 按 trans 决定 A 物理形状；`outLdc = ceilAlign(rows, 8)`；INT8 时 `aType=1` 走 2 字节读；`alphaScaled = alpha != (1,0)` 时额外产出 alpha 左面板 |
| cube | `n/k/arLdc/tempLdc/usedCoreNum/isTransN/triangleMode` | `tempLdc = n`（紧凑；fixpipe 按 Slice 尺寸直写，与 csyrk tiling 约定一致）；L1 分块 `CHUNK_K = 64`、L0 基本块 `BASE_K = 32` 为 kernel 内常量；`triangleMode = 1/2`（上/下三角，跳过完全在三角外的 tile） |
| 合成 | `n/ldc/tempLdc/betaReal/betaImag/uploMode/isBetaZero/isBetaOne/betaOnly` | alpha 已折入中间积，合成端仅处理 beta；`isBetaOne` 时直接加旧值（Netlib 语义，避免 `0*Inf`） |

#### 5. 分核策略

- cube 阶段按 64×64 输出 tile 分核（单 owner），`GetUsedAicCoreNum(n, 16)` 计算核数，
  triangle 模式跳过完全位于 uplo 三角之外的 tile（约省一半 cube 工作）；
- 去交错阶段按列切分（`colsPerCore`）；
- 合成阶段按行带切分（SIMT，每核 ≥16 行，超过 640 后全部参与）。

#### 6. 工作区

复用 handle 的默认 workspace（`EnsureDefaultWorkspace`），容量按 64 位溢出检查后计算：

```
arBytes    = ceilAlign(arRows, 8) × arCols × 4
tempBytes  = n × n × 4 × 2                           # Re | Im（紧凑，tempLdc = n）
rightPlanes = 5 × arBytes                            # Ar | Ai | Sum | ArNeg | AiNeg
leftPlanes  = (alpha != (1,0)) ? 3 × arBytes : 0     # ArS | AiS | SumS（否则复用右面板）

total = rightPlanes + leftPlanes + tempBytes，向上取 32B 对齐，且 ≤ ACLBLAS_MAX_WORKSPACE_SIZE
```

- alpha == (1,0) 时左面板直接复用右面板，不额外占空间；
- beta-only 路径不申请 workspace（combine kernel 的 betaOnly 分支只读 C）。

#### 7. 特殊值回退

与 csyrk 同策略：3M 分解与直接复数乘在 Inf/NaN/FLT_MAX 量级输入下的 inf/nan 传播位置不同，
无法对齐 golden 语义。因此当 `n × k ≤ 128×128` 时，host 侧 D2H 拷贝 A（INT8 先提升），
扫描 Inf/NaN、`|v| > 1e19`（平方溢出）与 subnormal；命中则用 host 侧 Netlib 语义的复数 csyrk
循环（复用 `csyrk_netlib_reference.h` 的循环语义，按复杂标量与本任务 Atype 扩展）计算结果并回写 C。
规模门是热路径零 D2H 同步的保证；大矩阵不进入回退，按任务书 §3.5 的 Inf/NaN 用例仅在
`TC_FL` 小规模中出现。

### kernel 侧设计

#### Stage 0：去交错 / 精确提升 / 面板生成（AIV，SIMT）

- 输入 A 按列分核读取：COMPLEX64 每次读 8B 交错实虚，INT8 每次读 2B 交错（int8→float 精确转换）；
- 输出右侧面板 5 个 float 平面（列主序，`outLdc`）：`Ar`、`Ai`、`Sum = Ar + Ai`、
  `ArNeg = −Ar`、`AiNeg = −Ai`——两个负面板供 mmad 做"带符号累加"（cube 无取负指令）；
- `alpha != (1,0)` 时同一遍产出左侧面板 3 个平面：`ArS = αr·Ar − αi·Ai`、
  `AiS = αr·Ai + αi·Ar`、`SumS = ArS + AiS`，与参考实现的"先乘 alpha 再累加"逐步舍入一致；
  `alpha == (1,0)` 时左面板直接复用右面板。

#### Stage 1：3M cube（AIC，单次 launch，两个 L0C 累加器）

- **一次 kernel launch 内完成三个实数积**，并把 3M 组合折进两个 L0C 累加器（各 16KB，FP32）：

```
AccRe = ArS·Ar^T + AiS·(−Ai)^T                  = P0 − P1
AccIm = SumS·Sum^T + ArS·(−Ar)^T + AiS·(−Ai)^T  = P2 − P0 − P1
```

- 每个 K 基本块按符号发 5 次 mmad：三个积 P0/P1/P2 各被以带符号的左右面板使用，
  其中 −P1 在两个累加器上各出现一次（硬件无"同一次 mmad 写两个累加器"的能力）；
  乘积本身仍是 3 个实数矩阵乘，融合在一次下发内完成；
- 左右面板：trans=N 时左 op(A)（n×k，DN 布局）、右 op(A)^T（k×n，ND 布局）；trans=T/C 时左右互换；
- L1：每个 K 分块 8 个面板（左 3 + 右 5）× 64×64 fp32 = 128KB，4 个 buffer 轮转；
  L0：`BASE_K = 32`，每步 8 个面板各 8KB 单槽；
- **逐项带符号累加使每个元素的项顺序与 Netlib 参考一致**（见「精度设计」），这是通过严格
  逐元素判据的关键；三角裁剪（`triangleMode`）跳过完全在 uplo 三角外的 tile。

#### Stage 2：合成 / 缩放（AIV，SIMT）

- 按行带切分，对 uplo 三角每个 (i,j)：`C = Re + i·Im + beta·C_old`（alpha 已折入中间积）；
- `beta == (0,0)` 时不读 C 旧值（`beta=0 且 C=NaN` 用例不被污染）；
- `beta == (1,0)` 时直接加旧值而不乘（Netlib 语义；否则 `0*Inf` 会得到 NaN，与含 Inf 的旧 C 不符）；
- 未选三角与 ldc padding 一律不写。

#### beta-only 分支

- `k==0` 或 `alpha==(0,0)` 且 beta≠(1,0)：combine kernel 的 betaOnly 分支只读 C，
  按 uplo 三角逐元素 `C = beta·C`；不申请 workspace、不启动 Stage 0/1。

### 精度设计

- 全程 FP32 累加；alpha 折入左面板后，每个元素的乘法与累加顺序与 Netlib 参考实现一致；
- **关键点：不能"先累三个大和、最后相减"**。`P2 = (Xr+Xi)(Xr+Xi)^T` 的项恒正、部分和随 k
  线性增长；若先得到 P0/P1/P2 再算 `P2−P0−P1`，在近零（强消去）元素上与参考实现的散差会
  超过 `max(1e-2, 32×ULP)`（实测 k≥512 的用例即触发）。本实现把符号折进 mmad，逐项累加
  **已经相消的组合**（`xr·xi + xi·xr` 等），消去元素不再放大累加误差；
- 判据：`matched_ratio ≥ 0.99` 且每个元素 `|actual − golden| ≤ max(1e-2, 32×ULP)`（FLOAT32 分量，
  实部/虚部分别判定）；inf/NaN 成对一致（同 NaN、同号 Inf）；
- INT8 输入：|v| ≤ 128 在 FP32 中精确，提升无舍入，后续链路与 COMPLEX64 完全一致；
- golden：`cblas_csyrk`（Netlib BLAS 3.10.0），INT8 输入逐元素精确提升后调用；
  `TC_3M_*` 专项用例检查交替/极值填充下的实虚部抵消。

### 性能设计

- **3M 乘积 + 双累加**：三个实数积（P0/P1/P2）在一次下发内完成；由于 cube 没有"取负/同一
  mmad 写两个累加器"的能力，带符号累加需要 5 次 mmad/基本块（4M 为 4 次）。这换来的是
  逐元素舍入路径与参考实现一致（严格精度判据的前提）；面板与 temp 开销上，temp 从 3 个
  quad 降到 2 个；
- **三角裁剪**：triangle 模式只计算 uplo 三角覆盖的 tile，cube 侧工作量再减约一半（与 csyrk 相同）；
- **tile 结构**：64×64 单 owner tile、L0C 2×16KB、L1 8 面板×64KB 分块×4 buffer、L0 8 面板
  单槽（`BASE_K = 32`）；
- **后续优化空间**：小 k（3M 消去误差低于判据门槛）可退回"3 累加器 + 合成端相减"的更快路径，
  按性能实测决定是否启用并按 k 门控；
- 目标：单 case ≤ `gpu_ms` × 2.5（≥0.4×），全部性能 case 平均倍率 ≥ 0.5×
  （`msprof op` Task Duration kernel-to-kernel 口径，warmup 5 + 有效采样 10）；
- INT8 路径不做 INT8 cube 加速（无 int8×complex64 的 3M 硬件路径），按任务书以 COMPLEX64
  基线对比、性能数据单独补充说明。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | √ |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 本期不涉及 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 本期不涉及 |

前置条件：CANN ≥ 9.1.0，asc-devkit ≥ 9.1（tensor_api / RegBase，与 csyrk 相同）。

## 算子约束限制

1. 仅列主序 + lda/ldc 前导维语义，不支持任意非连续布局、不使用 AutoContiguous/ViewCopy；
2. 不支持广播；n/k 为运行时入参，不做编译期特化；
3. C 为原地覆写，调用方保证 C 可写区域与 A 可读区域不重叠；仅 uplo 三角被读写；
4. 确定性计算不作要求：3M 路径（HF32 开关、K 归约顺序）由形状决定，不按测试编号或随机种子分派，
   同一输入两次执行不保证逐位一致；
5. alpha/beta 须成对同为 Host 或同为 Device 指针，混用返回 `INVALID_VALUE`；
6. 结果与 `aclblasCsyrk`（4M）/cuBLAS `cublasCsyrk`（直接复数乘）不保证逐位一致。

# 可维可测分析

## 测试设计

### 用例构成

`test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test.csv` 共 1200 条（随任务包提供，1000 精度 + 200 性能）：

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 12 | 6 组 uplo×trans × 小尺寸 (4,8) |
| L1 尺寸 | TC_SQ | 204 | 34 种尺寸（1→4096，含 2 的幂 ±1）× 6 组参数组合 |
| L2 标量 | TC_AB | 48 | 8 组复数 alpha/beta × 6 组参数组合 |
| L3 矩形 | TC_WS/TH | 24 | k≫n（trans=N）+ n≫k（trans=T） |
| L4 前导维 | TC_LD | 27 | 最小 lda/ldc + padding（+1/+8/+64） |
| L5 填充 | TC_FL | 12 | 均匀/零/交替/极端/Inf/NaN（A、C 分别，仅 COMPLEX64） |
| L5b 覆盖 | TC_CV | 48 | 中等尺寸（10~400）× 6 组参数组合 |
| L6 边界 | TC_ED | 38 | 零维、k=0（含 beta≠(1,0) 缩放）、alpha=(0,0)、alpha=0 且 A=NaN、beta=0 且 C=NaN、对角虚部保留、空指针、非法枚举/前导维/负维度、不支持的 Atype/Ctype、int8 边界/零值 |
| 3M 专项 | TC_3M | 4 | 交替/极值填充下实虚部抵消、int8 极值、对角虚部保留 |
| EX 扩展 | TC_EX | 583 | 尺寸池 × 枚举 × 标量 × padding 确定性采样 |
| PF 性能 | TC_PF | 200 | 5 条典型 case + 小尺寸 + 方阵扫描 + 宽/窄矩阵 + 2 条 int8 |

CSV 列：`case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,a_fill,Atype,lda,beta_real,beta_imag,c_fill,Ctype,ldc,expect_result,mere_threshold,mare_multiplier,random_seed`。
`NULLPTR`/`null` 表示空数组/空标量指针；`handle=nullptr` 用例由测试工程在装配层固定覆盖。

### golden 与判定口径

- golden：`cblas_csyrk`（`CblasColMajor`，Netlib BLAS 3.10.0），INT8 输入逐元素精确提升后调用；
- 比对范围：仅 uplo 三角（含对角），实部、虚部分别判定；调用前对未选三角与 ldc padding
  写哨兵，调用后校验未被改写；
- 对角虚部按普通复数元素比对，**不**按 Hermitian 口径断言为 0（`TC_ED`/`TC_3M` 专项覆盖）；
- 判定：`|actual − golden| ≤ atol + rtol·|golden|`（rtol = atol = 2⁻¹³），
  `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 或 32×ULP`；
  Inf/NaN 按任务书规则单独一致（同 NaN、同号 Inf 视为一致），一方有限一方非有限判失败；
- 测试工程需实现的填充 token：`RANDOM_NORM_5_5`、`RANDOM_GAUSS_0_1`、`RANDOM_INT8`、
  `VALUE_INT8_EXTREME`、`RANDOM_DIAG_IMAG`。

### 性能测试

- 正式判据：`msprof op` 采集单算子 kernel `Task Duration(us)`（warmup 5 + 有效采样 10 取平均），
  除以 1000 与 `gpu_baseline.csv` 的 `gpu_ms`（GPU kernel-only）比较：
  单 case ≤ `gpu_ms` / 0.4 且全部 case 平均倍率 ≥ 0.5；
- GTest 按用例打印 `[PERF] case=<name> ... avg_us=<数值>`（warmup 5 + `aclrtEvent` 夹 10 次），
  `verify_performance.py` 仅做交叉验证，不作为正式判据；
- 采集命令：`msprof op --application="./build/test/syrk/csyrk3m_ex/csyrk3m_ex_test --gtest_filter=*TC_PF*" --output=./prof_csyrk3m_ex`。

### 复现入口

```bash
bash build.sh --ops=csyrk3m_ex --soc=ascend950 --run
```

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 分量档：rtol = atol = 2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP；实/虚部分别判定 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能标准 | kernel-to-kernel：单 case ≥ 0.4× GPU，全部 case 平均 ≥ 0.5× | 任务书 §3.3（msprof Task Duration） |

## 兼容性分析

1. 新增接口与实现文件，不修改任何既有函数签名与语义，`aclblasCsyrk`（4M）行为不变；
2. 新增文件统一以 `csyrk3m_ex_` 前缀命名，单算子构建过滤、测试注册均复用现有机制，
   不改 `blas/CMakeLists.txt` 的依赖登记表；
3. 3M 驱动为独立 header（`syrk3m_gemm_arch35.h`），只被新 kernel 引用，不触碰 csyrk 的
   `syrk_gemm_arch35.h`；
4. `aclDataType` 的用法与 `aclblasGemmEx`/`aclblasCgemmEx` 一致，不引入私有枚举；
5. 与任务书 §2.5 的一处口径差异（A 为空指针的判定）已在 host 设计第 2 节说明，实现时在
   README「与任务书口径的差异说明」中同步记录。
