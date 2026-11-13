# aclblasChbmv 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasChbmv算子开发（A2A3）；算子 aclblasChbmv（对标 cuBLAS cublasChbmv，CUBLAS_Library.pdf；Netlib `chbmv.f`）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/sbmv/arch22/`（实现）+ `test/sbmv/chbmv/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能测试设备 Atlas 800T A2 (910B3)；接口新增声明进 `include/cann_ops_blas.h`（任务书 §2.3，与实数族 `aclblasSsbmv` 口径一致）。

## 背景介绍
### 算子功能定位
单精度复数 Hermitian 带状矩阵-向量乘：`y = α·A·x + β·y`，A 为 n×n Hermitian 带状矩阵（k 条次对角线 + k 条超对角线），按列主序带状格式存于 (k+1)×n 数组（前导维 lda）；x、y 为 n 元素复数向量，支持正负步长。补齐 Blas 仓 Atlas A2/A3 系列 BLAS-2 带状复数能力缺口。

### 业界对标
- cublasChbmv：参数序列、返回码与快速返回口径唯一依据；`uplo` ∈ {UPPER, LOWER}，带状存储仅引用 uplo 侧三角，对角元素虚部假定为 0（只取实部参与计算）。
- Netlib `chbmv.f`：golden 语义（测试真值由 `cblas_chbmv` 生成，输入转 float64 真值计算、按输出 FLOAT32 收窄落盘，issue #434 口径）。
- 语义差异点：本实现 α=(0,0) 不短路（仍按公式计算，不保证跳过 A/x 读取），golden 短路为 y=βy——两者仅当 x 含 Inf/NaN 时不同，故该组合不设精度用例（任务书 §2.1-4）。

### 工程形态
- Host C API + Kernel 直调（与同目录实数同族 `aclblasSsbmv`、本目录已合入复数近亲同一工程模式：host 校验/tiling + Ascend C SIMD 内核经 handle 绑定 stream 直调）。
- 代码落位：`blas/sbmv/arch22/`（并入既有目录，不新建族目录）。
- 仓内近亲：`blas/sbmv/arch22/ssbmv_*`（实数带状版，工程骨架与校验锚点）、`blas/sbmv/arch22/chbmv_*`（若 master 已合入则作逐行参考；本设计与之同构）、`blas/gbmv/arch22/cgbmv_*`（复数带状访存同构）。
- 红线：接口声明进公共头 `include/cann_ops_blas.h`，禁止私有平行 API（任务书 §7-3）。

### 接口规格
```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```
（新增声明；签名按 cuBLAS `cublasChbmv` 参数序列，维数参数 int，与仓内 `aclblasSsbmv` 口径一致）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR（最优先校验） | - |
| uplo | 带形三角存储模式 | aclblasFillMode_t | attr | ∈{UPPER, LOWER}；非法 → ACLBLAS_STATUS_INVALID_VALUE（本任务书口径，仓内 csymv 的 INVALID_ENUM 差异由研发后续统一，任务书 §7-5） | - |
| n | 方阵 A 的阶数 | int | attr | n≥0；n<0 → INVALID_VALUE；n=0 为合法 quick return（不先于判空） | - |
| k | 带宽（次/超对角线条数） | int | attr | k≥0；k>n-1 按有效带宽 min(k,n-1) 参与计算；k<0 → INVALID_VALUE | - |
| alpha | 复数标量 α 指针（Host） | const aclblasComplex* | COMPLEX64 | n>0 时非空；α=(0,0) 仍按公式计算 | - |
| A | Hermitian 带状矩阵（Device，只读） | const aclblasComplex* | COMPLEX64 | 列主序带状存于 (k+1)×n（数组维度 lda×n）；对角虚部假定为 0；n>0 时非空 | [lda,n]（有效 (k+1)×n） |
| lda | 带状数组前导维 | int | attr | lda≥k+1（k+1 计算与比较在 int64 域，防 k=INT32_MAX 回绕）；违反 → INVALID_VALUE | - |
| x | 输入向量（Device，只读） | const aclblasComplex* | COMPLEX64 | 非零即非空；跨度溢出守卫同 y | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | ≠0 且 ≠INT32_MIN，正负均支持；(n−1)×\|incx\|>INT32_MAX → INVALID_VALUE（host 侧 int64 计算，issue #780/#866） | - |
| beta | 复数标量 β 指针（Host） | const aclblasComplex* | COMPLEX64 | n>0 时非空；β=(0,0) 时 y 旧值不必有效（不豁免非空） | - |
| y | 输入兼输出向量（原地，Device） | aclblasComplex* | COMPLEX64 | 非零即非空；跨度溢出同 incx 口径 | [1+(n−1)·\|incy\|] |
| incy | y 步长 | int | attr | 同 incx | - |

### 数学公式
y[j] = α × Σ(i=max(1,j−k)..min(n,j+k)) A(i,j) × x[i] + β × y[j]，j = 1..n；
带状存储映射：UPPER 时 A(i,j) 位于 A(1+k+i−j, j)（i∈[max(1,j−k),j]），LOWER 时位于 A(1+i−j, j)（i∈[j,min(n,j+k)]）；对角元素只取实部（Hermitian 性质）；β=(0,0) 不读 y 旧值。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调（SIMD 向量内核）在 Atlas A2/A3（arch22/ascend910b）实现 aclblasChbmv 全语义（UPPER/LOWER 带状映射、对角取实、正负步长、lda 空洞、int64 跨度守卫），精度满足 mixed_tolerance FLOAT32 口径全绿，性能单 case 达 0.8×A100 cublasChbmv（≤1.25× 标杆耗时）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | y=α·A·x+βy；Hermitian 带状 = 仅引用 uplo 三角 + 对角取实 + 未引用侧共轭隐含；带状列主序访存与 arch22 cgbmv/csymv 同构（复数搬运机制复用，多出带内行偏移 (1+k+i−j 或 1+i−j)×lda 的索引换算） |
| 2 | 接口原型 | 见上节；新增声明进公共头（任务书 §2.3） |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐任务书 §2.5：handle 判空最优先 → n≥0 → n=0 quick return → uplo/k/lda/incx/incy/alpha/beta/A/x/y；quick return 不得先于判空（#765/#861 口径） |
| 4 | 支持数据类型 | COMPLEX64（单接口单 dtype） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / SOC=ascend910b*；自验覆盖 910B3（§3.1） |
| 6 | 精度要求 | mixed_tolerance：rtol=atol=2⁻¹³(1.2e-4)，ratio≥0.99，max_abs≤1e-2 或 32ULP（golden 值动态计算，2026-09-24 修订口径）；实部/虚部分别比对；Inf/NaN 等效规则（NaN≡NaN、同号 Inf≡、NaN↔±Inf 等效）（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup 5 + ≥10 均值）≤1.25×gpu_baseline.csv（§3.3） |
| 8 | 代码架构选型+可行性 | host（校验+tiling+直调）+ SIMD 内核按输出行块分片；已排除死路：SIMT 直调（simt_api 仅 arch35）、按全矩阵载入（带状存储下流量浪费，带内逐列访存为本征形态） |
| 9 | 评测来源 | 精度 golden=cblas_chbmv（Netlib，fp64 真值收窄 FLOAT32）；性能标杆=任务包 gpu_baseline.csv（A100 实测） |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景；kernel 内按输出维（n 行）分块：每核处理若干输出行 j，对其带内区间 [max(1,j−k), min(n,j+k)] 的列做乘累加；UPPER/LOWER 两种带内行偏移由 uplo 分支换算（同一内核以 tiling 传入的 uplo 分支，或两个特化模板）。

### 支持数据类型
COMPLEX64（A/x/y/α/β 全单精度复数，实部/虚部各 float32，交错存储）。

### 支持形状
n≥0、k≥0 任意；n=0 quick return；k=0（纯对角）、k=n−1（满带退化，覆盖全矩阵）、非对齐 n/k、lda≥k+1 的 padding 空洞、|incx|>1 间隔与负步长均支持；k>n−1 按有效带宽 min(k,n−1) 收敛。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| chbmv_upper | uplo=UPPER | 输出行块 j∈[j0,j1)；带内列区间 [max(1,j−k), j]；带内元素位于 A(1+k+i−j, j)，同一输出行内随 i 增大行索引递增 → 列内元素跨 lda 不连续，按列（j 固定）切片搬 UB 后做跨步乘累加 |
| chbmv_lower | uplo=LOWER | 输出行块同上；带内列区间 [j, min(n,j+k)]；元素位于 A(1+i−j, j)，行索引随 i 增大递增，访存形态与 UPPER 镜像对称 |

数据流：GM(A,x,y) → MTE2 → UB（A 带内切片 + x 复数段 + y 原地段双缓冲，事件 ID0/ID1 交替流水）→ V 计算复数乘累加（实部/虚部平面分离或 Interleaved 口径与仓内复数近亲一致）→ MTE3 → GM(y 原地写回，负步长反向寻址)。UB 规划对齐 arch22 cgbmv 的 AsdopsBuffer 分区（A 切片 2×ping-pong、Ybase/Xbase 区、tmp 缓冲）。

关键换算：UPPER 下第 j 列带内元素为行区间 [1+k+j−min(n,j+k)+ … ] 的连续段（列内行号连续），即 A 切片可按"列 j 的行段 [max(1,j−k), j] → 数组行 [1+k+j−j .. 1+k+j−max(1,j−k)]"整段连续搬运（列主序下同列内行连续），段首偏移 = (1+k+max(1,j−k)−j−1) + j×lda。LOWER 镜像。故列内搬运天然连续，无需逐元素 gather。

#### host 侧设计
- 分核策略：numBlocks = min(GetAivCoreCount(), 上限)；输出行 n 按行块 ceil 切分后按核交错（idxLoop % aiv_num == aiv_id），满核优先、余块自然落在尾部核。
- 数据分块与内存优化：行块大小与列切片长度由 tiling 携带（初版沿 cgbmv arch22 常量，S5 按带宽实测精调）；k≫n 时单列带内段长可达 n，切片按 UB 容量再分（段内循环）；软件流水 copy_in(k+1) 前置。
- tilingkey 规划：无独立 tilingkey（uplo 分支由 TilingData 字段分发）；TilingData 字段：uplo/n/k/lda/incx/incy/alpha/beta（int64 对齐 + 复数标量按偏移读）。
- 数据检测（校验顺序，逐条对齐 §2.4/§2.5）：handle 判空（HANDLE_IS_NULLPTR）→ n≥0 → n=0 quick return（SUCCESS）→ uplo 枚举 → k≥0 → lda≥k+1（int64 域）→ incx/incy（0/INT32_MIN）→ 跨度溢出（int64：(n−1)×\|inc\|>INT32_MAX）→ alpha/beta 非空（n>0）→ A/x/y 非空（n>0）；返回 INVALID_VALUE；quick return 在校验全过后。异步下发（不同步流）。

#### kernel 侧设计
```
chbmv_upper / chbmv_lower:
  for idxLoop in [0, 行块数) 按核交错:
    [MTE2] y_in 段搬 UB → [V] y_in×β 累入 Ybase（β=(0,0) 跳过读）
           逐带内列 i: A 列切片 + x[i] 搬 UB（列内行号连续，整段 MTE2）
    [V] 复数乘：y_re += a_re×x_re − a_im×x_im; y_im += a_re×x_im + a_im×x_re
        对角元素（i==j）只取 a_re（虚部视为 0）参与
        尾部位掩码处理非整向量段
    [MTE3] Ybase 写回 y（负步长反向寻址）
  事件 ID0/ID1 交替构成 copy_in(k+1) 与 compute(k) 流水
```
无 workspace 依赖（tiling 走 GM 小缓冲或直调参数，S2 按 csrot/cgbmv 先例定型）；无原子操作。

#### 性能优化要点（终态，S5 回填实测）
- 事件化 ping-pong 双缓冲；列内连续段 MTE2 快路（对齐搬运优先，半对齐/逐段兜底分支）。
- 带宽维度自适应：k 小（稀带）时输出行块加大、减少 y 搬运轮次；k 大（满带）时按列切片分摊 UB。
- 候选：行块/切片参数联调、numBlocks 上限放开（Crot P30 先例：40 核 cap 可弃）、A 段与 x 标量广播的 Axpy 融合。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（另行任务） |

## 算子约束限制
- 不支持超出 lda/incx/incy 语义的非连续内存访问（任务书 §2.5）。
- 无广播；y 原地覆写；不要求确定性归约顺序；α=(0,0) 不短路（含 Inf/NaN 的 x 不得与 α=(0,0) 组合出精度用例）。
- 对角元素虚部假定为 0（不校验，直接丢弃虚部）。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | rtol=atol=2⁻¹³，ratio≥0.99，max_abs≤1e-2 或 32ULP（按 golden 动态）；实部/虚部分别比对；Inf/NaN 等效规则 | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分类（照 CSV 前缀）：精度（基础小尺寸 uplo 全组合 / 尺寸扫描 / 带宽扫描 k=0,1,小带宽,满带 / 标量特殊值 / lda padding / 步长组合 ±1,±2,±3,负步长 / Inf/NaN）、负向（零维、空指针、非法 uplo、非法 lda、负维度、零步长、INT32_MIN、跨度溢出）、性能 200 条。
- 期望码：非法 uplo/lda/负维度/零步长/INT32_MIN/n>0 空指针 → INVALID_VALUE；handle 空指针 → HANDLE_IS_NULLPTR；n=0 → SUCCESS。
- 性能口径：msprof op，warmup 5 + 有效采样 ≥10 取均值；复现入口见交付件 03。
- 正态分布用例（GAUSS_NORM_5_2 token）由开发者扩展 gen_csv.py 补齐并纳入精度自验（§3.5 补充要求）。

## 兼容性分析
- 新增接口声明至 include/cann_ops_blas.h：仅追加声明，不改既有签名，无破坏性；与 950 线共用同一 API 口径。
- 代码并入既有 blas/sbmv/arch22 目录，无目录迁移；测试目录 test/sbmv/chbmv/arch22 新建，CMakeLists 在 arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} chbmv_test)` 注册（当前同族仅 arch35 注册，A2/A3 构建会退回非 GTest runner，随本任务补齐）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能优化要点为规划值（终态数据待 S5 实测回填），不预填未经测试的数字。
