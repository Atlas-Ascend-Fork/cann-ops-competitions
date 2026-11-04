# aclblasCsyrkx 算子设计文档

> 目标芯片：Ascend 950PR（arch35，SOC_VERSION=ascend950）｜CANN 9.1.0｜数据类型 complex64
> 合入位置：`ops-blas/blas/syrkx/arch35/`；测试：`ops-blas/test/syrkx/csyrkx/arch35/`
> 任务书：`aclblasCsyrkx_Atlas950PR_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 10 月社区任务 —— aclblasCsyrkx 算子开发（Atlas 950PR）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/syrkx/arch35`；测试代码 → `test/syrkx/arch35`（任务书 §5）。
- 验收流程：社区任务 IT 系统提交交付件（设计文档 / 自测用例与代码 / 自测报告 / 待验收代码地址，任务书 §4）；设计文档以 PR 形式提交至 `cann-competitions` 仓 tasklist 通过评审。

## 背景介绍

### 算子功能定位

`aclblasCsyrkx` 实现复数单精度（complex64）对称秩-k 扩展更新（syrkx）：对 n×n 对称矩阵 C 的 uplo 指定三角，累加两个不同输入矩阵的乘积。

```
C = alpha * op(A) * op(B)^T + beta * C
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCsyrkx | cuBLAS `cublasCsyrkx`（legacy）｜Netlib BLAS 无该扩展；golden 用 `cblas_cgemm` 拼装 | C 的 uplo 三角 ← alpha·op(A)·op(B)^T + beta·C |

与同族 `aclblasCsyrk`（A==B）的区别：A 与 B 相互独立，故乘积 op(A)·op(B)^T **不具备** Hermitian 对称性，C 是**对称（非 Hermitian）**矩阵，**对角元素虚部不强制清零**。这与同族 `aclblasCherkx`（共轭版，Hermitian，对角虚部强制清零）形成语义分叉（§2.1 算法说明 1、§2.4 C 行）。

### 业界对标

| 项 | cuBLAS cublasCsyrkx | 本算子 |
|---|---|---|
| 标量 | alpha、beta **均复数** | 同（`const aclblasComplex*`） |
| trans 取值 | N / T / C，**C 与 T 等价、不取共轭** | 同（§2.1 算法说明 2；任务书称已在 P102+CUDA 12.5 实测 T 与 C 逐位相同） |
| 存储 | 列主序，lda/ldb/ldc | 同 |
| 三角 | 仅更新 uplo 指定三角 | 同 |
| 差异 | cuBLAS 对 C 无 nullptr 约定、beta==0 仅表示 C 入口值不参与 | 任务书 §2.4 明确 n>0 时 C 为空返回 INVALID_VALUE（严于 cuBLAS，与仓内 ssyrkx 口径一致） |

### 工程形态

- 工程模式：**Kernel 直调**（句柄式 BLAS 接口：Host C API 解析参数 + 校验 + tiling，`<<<>>>` 直调 NPU kernel，通过 handle 绑定 stream，§2.2）。
- 代码落位：`blas/syrkx/arch35/csyrkx_{host,kernel}.cpp + csyrkx_{kernel.h,tiling_data.h}`（与同族 ssyrkx 五件套同构，目录已存在，**不新建族目录**）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/syrk/arch35/csyrk_*` | ★ 主模板 | host 校验/cluster 分发/workspace/特值兜底；AIV deinterleave、AIC fused-4M cube、AIV combine 三段流水；复数 alpha/beta 缩放；trans C≡T 映射；对称三角语义（不消对角） | 数据源由单矩阵改为双矩阵（4 象限交叉项）；补 ldb/B 参数；no-op 判定顺序改「校验先行」 |
| `blas/herkx/arch35/cherkx_*` | ◆ 备选 | 双矩阵 deinterleave 两次的 host 编排；4 次独立 GEMM 的降级写法 | 不采用其共轭符号、实 beta、ZeroDiagonal |
| `blas/syrkx/arch35/ssyrkx_*` | ◇ 参考 | README 体例、测试工程组织 | 算法复数化 |
| `blas/common/helper/syrk_gemm_arch35.h` | 共享基建 | 通用 per-quad GEMM（含 HF32x3 残差通道） | 直接调用 |

### 接口规格

本算子需在公共头 `include/cann_ops_blas.h` **新增**一条声明（§2.3：仓内统一句柄式 API；现状核验：该头已有 `aclblasSsyrkx`(604 行) 与 `aclblasCherkx`(616 行)，**无** `aclblasCsyrkx`）。仅新增，不改动任何既有声明，不定义 950PR 私有平行接口。

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,     // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,   // ACLBLAS_OP_N / OP_T / OP_C
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | aclblas 句柄，携带 stream | scalar | — | 非空；空 → `HANDLE_IS_NULLPTR` | 标量 |
| uplo | C 的三角引用方式 | attr | int | UPPER / LOWER；其它 → `INVALID_VALUE` | 标量 |
| trans | A、B 的运算方式 | attr | int | OP_N / OP_T / OP_C（C≡T，不取共轭）；其它 → `INVALID_VALUE` | 标量 |
| n | C 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE` | 标量 |
| k | 缩并维 | attr | int | ≥0；负 → `INVALID_VALUE` | 标量 |
| alpha | 复数标量乘子 | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE` | 标量 |
| A | 输入矩阵（列主序） | tensor | COMPLEX64 | n>0∧k>0 时非空；`lda ≥ max(1, trans=N?n:k)` | trans=N: n×k；trans=T/C: k×n |
| lda | A 前导维 | attr | int | 同上；不满足 → `INVALID_VALUE` | 标量 |
| B | 输入矩阵（列主序） | tensor | COMPLEX64 | 同 A（`ldb`） | 同 A |
| ldb | B 前导维 | attr | int | 同 lda；不满足 → `INVALID_VALUE` | 标量 |
| beta | 复数标量乘子 | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE` | 标量 |
| C | 输入/输出对称矩阵 | tensor | COMPLEX64 | n>0 时非空；`ldc ≥ max(1,n)`；仅写 uplo 三角 | n×n |
| ldc | C 前导维 | attr | int | 同上；不满足 → `INVALID_VALUE` | 标量 |

> 错误码口径说明：§2.4 表把非法 uplo/trans 记为 `ACLBLAS_STATUS_INVALID_VALUE`；同族 cherkx 实现返回 `INVALID_ENUM`。本实现**以任务书 §2.4 原文为准**（非法枚举一律 `INVALID_VALUE`），并在测试 CSV 的 `expect_result` 列同步该口径；仓内 README 中标注该族内差异。

### 数学公式

设 alpha = αr + i·αi，beta = βr + i·βi，A = Ar + i·Ai，B = Br + i·Bi，乘积 P = op(A)·op(B)^T = Pr + i·Pi：

```
Pr = Ar·Br^T − Ai·Bi^T          （trans=N；trans=T/C 为转置形态 Ar^T·Br − Ai^T·Bi）
Pi = Ar·Bi^T + Ai·Br^T
Cr = αr·Pr − αi·Pi + βr·Cr_old − βi·Ci_old
Ci = αr·Pi + αi·Pr + βr·Ci_old + βi·Cr_old
```

仅写 uplo 三角；非 uplo 三角不引用、不写回，与调用前逐位一致。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas 950PR（arch35）上实现 complex64 句柄式 BLAS 接口 `aclblasCsyrkx`：支持 n×n 对称 C 的单三角更新、trans N/T/C（C≡T 不共轭）、复数 alpha/beta、任意合法前导维，精度达生态算子开源混合容差标准（rtol=atol=2^-13，matched_ratio 0.99），性能达 GPU 标杆的 0.4×（单 case）且 200 条性能用例倍率均值 ≥0.5×。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | `C = α·op(A)·op(B)^T + β·C`；4M 分解（复数乘 → 4 个实矩阵乘）把复数 GEMM 全部转成已合入的实数 cube GEMM 通路；csyrk 的合并公式 `Pr=Q0−Q1, Pi=Q2+Q3` 与本算子**逐字相同**，故 combine 阶段可平移 |
| 2 | 接口原型 | §2.3 签名（13 参数，见上），返回 `aclblasStatus_t` |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | A/B/C/alpha/beta 全 COMPLEX64（实部/虚部各 FP32） |
| 5 | 目标芯片与 arch 映射 | 950PR → DAV_3510 → arch35；构建 `--soc=ascend950`；A2/A3（arch22）不支持（README 产品支持情况节声明） |
| 6 | 精度要求 | §3.2：rtol=atol=2^-13，matched_ratio ≥0.99，max_abs_error ≤ max(1e-2, 32·ULP)；COMPLEX64 按实部/虚部分量统计；Inf/NaN 按 §3.2 第 4 条等效规则 |
| 7 | 性能预期 | §3.3：单 case NPU ≤ 2.5×GPU（≥0.4×），且 200 条倍率算术均值 ≥0.5×；口径 = msprof op 单算子 task duration，warmup 5 次 + 10 次有效样本均值 |
| 8 | 代码架构选型 + 可行性 | 三段流水（AIV deint → AIC fused-4M cube → AIV combine）+ 小 n 单 kernel SIMT direct；**已实测可行性**（csyrk 同构基线：n=1 2.61us / n=512 86us / n=1024 296us，均在 §3.3 预算内；n=64 direct 11.77us 略超 10.34us 预算 → 列为 S5 冲刺项）。**已排除死路**：① 纯 SIMT 全尺寸直算（cube 算力用不上，大 n 必爆带宽）；② 直接复用 cherkx combine（对角清零会破坏对称语义、实 beta 少一半缩放） |
| 9 | 评测来源 | 任务书附 `test_cases/`（1200 条 CSV + gen_csv.py + verify_accuracy.py + verify_performance.py + gpu_baseline.csv 200 条 H100 实测） |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。4M 分解形式（实现直接使用）：

```
Q0 = op(Ar)·op(Br)^T      Q1 = op(Ai)·op(Bi)^T
Q2 = op(Ar)·op(Bi)^T      Q3 = op(Ai)·op(Br)^T
Pr = Q0 − Q1              Pi = Q2 + Q3
Cr = αr·Pr − αi·Pi + βr·Cr − βi·Ci
Ci = αr·Pi + αi·Pr + βr·Ci + βi·Cr
```

`op(X)` 在 trans=N 时为 X（各平面按列主序 n×k），在 trans=T/C 时为 X^T（平面按 k×n 存、GEMM 取另一转置方向）。

### 支持数据类型

| 数据类型 | A/B/C/alpha/beta | 说明 |
|---|---|---|
| COMPLEX64（complex\<float\>） | ✓ | 实部/虚部各 FP32，元素 8 字节，列主序 (row,col) → col·ld+row |
| FP32 实数 | ✗ | 由 `aclblasSsyrkx` 承担 |

### 支持形状

| 维度 | 约束 | 退化行为 |
|---|---|---|
| n | ≥0 | n=0：**参数校验通过后** quick return SUCCESS，不写 C |
| k | ≥0 | k=0：仅 uplo 三角 `C = beta·C`，不引用 A/B，另一三角不改写 |
| lda/ldb | ≥ max(1, trans=N?n:k) | 不满足 → INVALID_VALUE |
| ldc | ≥ max(1, n) | 不满足 → INVALID_VALUE |
| alpha | 复数任意 | alpha=(0,0)：等价 k=0 路径（跳过乘积） |
| beta | 复数任意 | beta=(1,0) 且 alpha=0/k=0：C 恒等（快返回，不 launch） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（变体/触发条件/职责）：

| 变体 | 触发条件 | 职责 | kernel 数 |
|---|---|---|---|
| Direct（SIMT 单 kernel） | n ≤ 64（含任意 k）或 lda/ldc 带 padding 的极小形状 | 逐元素直算 uplo 三角，Netlib 求值序（Kahan 补偿），alpha/beta 由 tiling 内联 | 1 |
| Direct-dirty | 同上 + host 预扫描发现 FLT_MAX 级/Inf/NaN/次正规输入 | 特值兜底：host 用双矩阵 Netlib 参考实现算好后 H2D 回写 | 1（+ host 计算） |
| 三段流水（主路径） | n > 64 | deint(A)+deint(B) → fused-4M cube → combine | 4（+1 HF32 残差） |
| beta-only | alpha=0 或 k=0 且 beta≠(1,0) | 仅 uplo 三角 C = beta·C（无 workspace） | 1 |
| 恒等快返回 | alpha=0 或 k=0 且 beta=(1,0) | host 直接 SUCCESS，不 launch | 0 |

数据流（主路径）：

```
 GM: A(complex) ──AIV deint──> Ar, Ai (real, 32B 对齐紧凑)
 GM: B(complex) ──AIV deint──> Br, Bi
        │
        ├──AIC fused-4M cube（tensor_api，64×64 单属主 tile，双 L0C 乒乓）──> GM temp[ Q0 | Q1 | Q2 | Q3 ]
        │       left 操作数取自 A 的平面，right 操作数取自 B 的平面
        │
        └──AIV combine──> 读 temp + C_old（uplo 三角）──> 写 C（uplo 三角）
```

workspace 布局表（主路径）：

| 区段 | 大小 | 说明 |
|---|---|---|
| dAr | arLdc·arCols·4 B | A 实部平面（arLdc = CeilAlign(行数,8)，32B 对齐） |
| dAi | 同上 | A 虚部平面 |
| dBr | 同上 | B 实部平面 |
| dBi | 同上 | B 虚部平面 |
| dArLow/dAiLow/dBrLow/dBiLow | 各同上 | 仅 HF32x3 补偿路径使用（x − HF32(x)） |
| dTemp | n · tempLdc · 4 · 4 B | 4 个象限，各 n×tempLdc 的 FP32 |
| 上限 | `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB） | 超出 → `ACLBLAS_STATUS_ALLOC_FAILED`（乘加前先做 uint64 溢出检查） |

#### host 侧设计

**分核策略**：满核原则 + 均分余块。
- cube：`GetUsedAicCoreNum(n, SYRK_ARCH35_BASE_M)`，按 n×n 的 64×64 块数分配；25 个核以上时 fused-4M 单属主 tile 平分块，不足 4 核时退化为「单核串行 4 象限通用驱动」。
- combine（AIV）：n ≤ 640 时每核 ≥16 行的行带划分，否则按 tile 跨步划分；小 n fast 配置走 SIMT 一线程一元素。
- direct：`ceil(n²/16)` 上限截到 AIV 核数。

**数据分块与内存优化策略**：
- L1 预算驱动的 `tileKChunk`：`l1Budget / (L1_BUF_NUM · sizeof(float) · (alignM+alignN))`，向下取 baseK 整数倍。
- L1/L0 双缓冲：`SYRK_ARCH35_L1_BUF_NUM` 槽 + L0C 双 acc-set 乒乓，K 方向流水。
- temp 只保留「全矩阵 4 象限」，combine 只消费 uplo 三角；三角跳 tile 留作后续性能项（首版正确性优先）。
- UB：combine 按 8 行 bulk 载入 4 象限 + C，一次向量化复数缩放写回。

**tilingkey 规划策略**（以 tiling 字段表达，不使用宏分支）：

| 字段 | 取值 | 分支含义 |
|---|---|---|
| `tilingKind`（host 侧选择） | 0=direct / 1=三段 / 2=beta-only | 顶层分流 |
| `isTransN` | 0=转置 / 1=不转置 | deint 几何 + cube 左右布局 + combine 读法 |
| `uploMode` | 121=UPPER / 122=LOWER | 三角掩码 |
| `isAlphaZero` / `isKZero` / `isBetaZero` | 0/1 | 跳过乘积 / 跳过 beta 累加 |
| `hf32` | 0=fp32 native / 1=HF32x3 补偿 | cube 精度通道 |
| `useDirectDiag` | 0/1 | 对角是否按元素重算（大 k 抵消风险） |
| `useSimt` | 0/1 | combine 是否走 SIMT 小 n 快路 |

**数据检测（校验顺序，逐条对齐 §2.4/§2.5）**：

1. handle 非空 → 否则 `HANDLE_IS_NULLPTR`
2. uplo ∈ {UPPER, LOWER} → 否则 `INVALID_VALUE`
3. trans ∈ {N, T, C} → 否则 `INVALID_VALUE`
4. n ≥ 0、k ≥ 0 → 否则 `INVALID_VALUE`
5. lda/ldb ≥ max(1, trans=N?n:k)、ldc ≥ max(1,n) → 否则 `INVALID_VALUE`
6. alpha/beta 非空 → 否则 `INVALID_VALUE`
7. alpha/beta 指针位置探测：混合 Host/Device → `INVALID_VALUE`；Device 模式 D2H 读回真值
8. 条件空指针：n>0∧k>0 且 alpha≠0 时 A/B 非空；n>0 时 C 非空 → 否则 `INVALID_VALUE`
9. **以上全部通过后**才判 n==0 quick return SUCCESS（§2.5「参数校验先于 no-op 判定」）

#### kernel 侧设计

**直接计算 kernel（Direct，SIMT）**：

```
Init: 解析 tiling（n,k,lda,ldb,ldc,transposed,upper,strided,alpha/beta 内联值）
Process:
  全局线程 tid → 覆盖 uplo 三角的下标 (i,j)（upper: j≥i；lower: j≤i）
  逐元素：s = Σ_{l=0..k-1} A(i,l)·conj?(B(j,l))   ← 复数 Kahan 补偿累加，trans=N 读 A[i+l·lda]/B[j+l·ldb]，trans=T 读 A[l+i·lda]/B[l+j·ldb]
  C(i,j) = α·s + β·C_old(i,j)     ← 仅写 uplo 位置
```

**三段流水（n > 64）**：

```
[AIV] deinterleave: 每列紧凑 8B 块搬入 UB → 向量 DeInterleave 拆实/虚 → 写 Ar/Ai（同样的第二次给 Br/Bi）
[AIC] fused-4M cube: 每个 64×64 单属主 tile 内，Q0..Q3 背靠背 mmad（left 取 A 平面、right 取 B 平面），
     双 L0C acc-set 乒乓；HF32x3 路径每个象限 3 次 mmad（xh·yh + xl·yh + xh·yl）
[AIV] combine: 8 行 bulk 读 4 象限 + C_old；若 useDirectDiag，则对角元素改按元素重算（A/B 双矩阵点积，Kahan）；
     复数 alpha/beta 交叉缩放后写回 uplo 三角（对角不做虚部清零）
```

**为什么对角要重算**：`Pr = Q0 − Q1` 在对角上发生灾难性抵消（Ar·Br 与 Ai·Bi 同号相减到接近 0），两个大 GEMM 相减会把相对误差放大到超出 2^-13；按元素先减后累加（Kahan）可保住精度。触发条件：k > 512，或 n ≤ 64 且 k > 128（小 n 对角元少、代价可忽略）。

#### 性能优化要点（首版与规划）

| 优化项 | 依据 | 状态 |
|---|---|---|
| 小 n 走单 kernel direct（避开 3–4 次 launch） | 实测 csyrk direct n=1 = 2.61us vs 三段流水 launch ≥7.5us | 首版实现 |
| alpha/beta 内联进 tiling（direct 不依赖 Device 标量） | csyrk 只在 scalarsOnDevice 时才走 direct，Host 标量调用丢失 direct 档 | 首版实现（本算子修正） |
| HF32x3 补偿 cube（大 n）×  | csyrk 实测 n=1024 用 HF32x3 仍 296us（<420 预算） | 首版沿用 csyrk 策略，S4 复测后按真实预算调整闸门 |
| 三角跳 tile（cube 只算 uplo 半区） | 理论省 ~50% cube 时间 | S5 候选 |
| 大 k 分块补偿求和 | 2^-13 容差 + max_abs_error 1e-2 | 首版（Kahan/补偿已有） |
| 未采用：纯 SIMT 全尺寸 | cube 算力闲置，大 n 带宽时间 ~300us 无法用 SIMT 喂满 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas 950PR / 950DT（arch35 / DAV_3510） | ✓ 本任务目标（`--soc=ascend950`） |
| Atlas A2 / A3 训练与推理系列（arch22） | ✗ 不支持（README 产品支持情况节声明） |

## 算子约束限制

逐条对齐 §2.5：

- 非连续 Tensor：不支持（A/B/C 按列主序前导维连续存放；ld 只需满足下界，允许 padding）。
- broadcast：不支持。
- dynamic shape：不要求（n/k 编译期已知）。
- 原地与视图语义：C 为 in/out；beta≠0 时读 uplo 三角旧值；不返回视图；另一三角不引用不修改。
- 确定性计算：要求，同输入两次执行逐位一致。实现上：cube 结果与 combine 均为固定归约序（K 方向顺序 mmad + 顺序累加），无跨线程原子归约、无动态并行 → 天然确定性。
- 空 Tensor 与 0 维：见"支持形状"表；参数校验先于 no-op 判定。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | COMPLEX64 按 FP32 分量：rtol=2^-13(1.2207e-4)、atol=2^-13、matched_ratio ≥0.99、max_abs_error ≤ max(1e-2, 32·ULP)；逐元素判据 `\|a−g\| ≤ atol + rtol·\|g\|` | 任务书 §3.2 + 生态算子开源精度标准 `mixed_tolerance_standard.md`（pin 28a1ce52） |
| 精度·特殊值 | golden 与 actual 同为 NaN / 同号 Inf / 一方 NaN 一方式 ±Inf → 等效通过；有限 vs 非有限 → 不通过；非有限值不计入 ratio/max_abs，单独统计 | 任务书 §3.2 第 4 条 |
| 精度·三角面 | 仅比对 uplo 三角（含对角）实部/虚部；非 uplo 三角与调用前逐位一致 | 任务书 §3.2 第 1 条 |
| 性能 | 单 case：GPU 标杆 / NPU 实测 ≥0.4（NPU ≤ 2.5×GPU）；附加：200 条倍率算术均值 ≥0.5 | 任务书 §3.3.2（L3 算子双条件） |
| 性能口径 | msprof op 单算子 task duration（us）；5 次 warmup 不计入 + 10 次有效样本平均 | 任务书 §3.3.1 |

## 测试概览

| 类别 | 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础 | TC_L0 | 12 | uplo × trans × 尺寸 (4,8) |
| L1 方阵尺寸 | TC_SQ | 162 | 27 种尺寸 × trans 3 × uplo 2 |
| L1 矩形 | TC_MN | 60 | 10 组 n≠k（含 1000×999、2048×1024）× trans 3 × uplo 2 |
| L2 标量 | TC_AB | 48 | 8 组 alpha/beta × trans 3 × uplo 2（含 (0,0)、(±1,0)、纯虚、大值） |
| L4 前导维 | TC_LD | 24 | lda/ldb/ldc padding +4/+8 × 4 尺寸 |
| L5 填充 | TC_FL | 18 | 均匀/全零/交替/极端/Inf/NaN（A、B、C 分别） |
| L5b 覆盖 | TC_CV | 32 | 中等尺寸 10~400 |
| L6 边界负向 | TC_ED | 31 | n=0 / k=0(双 uplo) / 条件空指针 / 非法枚举 / 非法 ld / 负维度 |
| EX 扩展 | TC_EX | 613 | 尺寸池 × uplo × trans × 标量 × padding 采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case + 小尺寸 + 规模对数扫描 + 随机组合 |

- 精度：1000 条，golden = `cblas_cgemm` 拼装（trans=N: opA=NoTrans/opB=Trans；trans=T、C: opA=Trans/opB=NoTrans，C 不共轭），非 uplo 三角恢复原值。
- 性能：200 条，`gpu_baseline.csv`（H100 实测 `gpu_ms`）；判据见上表。
- 复现入口：`test/syrkx/csyrkx/README.md` + 交付件 03 步骤说明（`bash build.sh --ops=csyrkx --soc=ascend950` → 跑 gtest → msprof 汇总）。

## 兼容性分析

- 新算子，对既有接口零影响；唯一公共文件变更为 `include/cann_ops_blas.h` **新增一条声明**（§2.3 授权；红线 3 例外清单已登记），不改动既有声明与 ABI。
- arch22（A2/A3）不提供实现，头文件声明对全产品线可见属仓内既有惯例（同 Ssyrkx/Cherkx 一致）。
- 不触碰 `blas/common/` 与 `cmake/` 之外的其他公共逻辑；cube 侧只调用共享 helper `blas/common/helper/syrk_gemm_arch35.h`。

