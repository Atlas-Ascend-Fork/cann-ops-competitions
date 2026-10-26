# 单精度复数QR分解与最小二乘（Atlas 950）算子设计文档

> 口径声明：本文为**设计阶段**文档，给出实现方案、资源预算与验证方法，不包含本任务的实现、编译或跑测结论。
> 文中标注「同源实测」的数据来自**同一硬件（Ascend 950PR）、同一仓、同一工具链下的实数 QR 任务（10-3）**已完成的工程验证，仅作为本设计的**可行性输入与阈值初值**，不代表本任务任何 case 已达标。
> 版本 v1.0（2026-10-20）。

# 需求背景（required）

## 需求来源

本需求来自 2026 年 10 月 CANN 社区任务「单精度复数 QR 分解与最小二乘（950）」。任务要求在 `gitcode.com/cann/ops-solver` 仓中，面向 **Ascend 950PR**，以 ops-solver Host C API + AscendC/CATLASS Kernel 直调模式实现稠密复数 QR 分解、酉变换与最小二乘四接口，精度达到 CANN 生态算子开源标准，验收通过后合入 `gitcode.com/cann/ops-solver`。

设计依据与冻结基线：

| 项目 | 基线 |
| --- | --- |
| 任务输入 | `Atlas950_Cgeqrf_Cungqr_Cunmqr_CCgels_task_doc.md` 与 `cgeqrf/ cungqr/ cunmqr/ cgels/` 四套用例、GPU 金标、`package/` 测试工程 |
| 社区文档模板 | `cann-competitions/04_tasks/01_community-task-2026/resources/design_template.md` |
| 目标实现仓 | `cann/ops-solver`；**当前无复数 QR / gels 原型，须全新实现** |
| 目标硬件 / 架构 | Ascend 950PR（Atlas 950）/ DAV_3510（`arch35`） |
| CANN 版本 | ≥ 9.0.0，取仓 README 已验证配套版本（本设计以 9.1.0 为事实源） |
| 数据类型 / 布局 | COMPLEX64（`aclFloatComplex`）/ Column Major |
| 精度 golden | NumPy / SciPy LAPACK（complex128 计算，COMPLEX64 同精度链路提供 `ratio_cpu`） |
| 性能金标 | `cgeqrf/ cungqr/ cunmqr/ cgels/bench_result.json`（cuSolver 同 shape/dtype/列主序） |

四个接口作为**同一任务一并交付，不允许拆分验收**；本文按此为前提设计整体架构。

## 背景介绍

### 四个接口构成一条算法栈

本任务不是四个独立算子，而是一条 QR 算法栈：

```text
aclsolverCCgels（最小二乘，组合层）
    = aclsolverCgeqrf + aclsolverCunmqr(LEFT, C) + 内部复三角回代
aclsolverCgeqrf（核心，分块 Householder 复数 QR）
    = panel 分解 + 复数 larft(T) + 尾随 E(U)·S(V) 嵌入 GEMM
aclsolverCungqr / aclsolverCunmqr（WY 块反射器应用，共用 larfb 机制）
```

关键约束在于**数据依赖方向**：`Cungqr` / `Cunmqr` 的输入必须是 `Cgeqrf` 输出的 Householder 表示（A 的严格下三角存 v、上三角存 R、`tau` 存系数），而不是未分解的原始 A。因此三者共享同一套反射器约定；`CCgels` 复用同一条分解流水，不引入第二套数学。

本任务**不交付**公开 `aclsolverXlarft` 接口（实数 QR 任务有、本任务无）。形成块反射器所需的三角因子 T 作为**内部组件**实现，位于 `src/utils/qr_common/`，由三个接口共同 include，不复制多份。

### ops-solver 仓现状

仓内现有算子为 LU 系：`cgetrf` / `cgetri` / `sgetrf` / `sgetri` / `cmatinv_batched` / `cgetri_batched`，无任何 QR / gels 实现。可复用的只有**框架层**：

| 可复用 | 不可复用（须新建） |
| --- | --- |
| `aclsolverCreate/Destroy/SetStream/GetStream` handle 与 stream 管理 | 既有复数 LU 的 `std::complex<float>` 类型与部分返回 `aclError` 的原型——**不是本任务 ABI** |
| `aclsolverStatus_t` 状态枚举与映射惯例 | 既有单矩阵算子的 Host 指针 + 内部 H2D 拷贝通路——**本任务全部为 Device 指针** |
| `src/sgetrf/` 的 Host 校验顺序、TilingData POD 惯例、workspace 分区写法 | 任何 QR 数学实现 |

新接口按任务书定义 `aclFloatComplex` 与 `aclsolverStatus_t`，复用 handle 与 stream 语义，不把旧算子的 Host 指针处理方式带入本设计。

### 950PR 上的复数计算事实

以下为**同源实测**已确认、直接决定本设计架构的平台事实：

1. **arch35 Cube 具备 FP32 MMAD**（基块 `baseK=8`，fixpipe `C0=8`），是唯一满足 COMPLEX64 精度要求的 Cube 路径；禁用 HF32 / FP16 冒充。
2. **复数 AoS 布局不能直接进 Cube**，须先 pack。采用**实数块嵌入**：一次 FP32 Mmad 完成复数乘，不拆四次独立实 GEMM（依据见 §复数到实数块嵌入）。
3. **AIV 支持 SIMT 编程模型**（`__simt_vf__` + `LAUNCH_BOUND`），适合 panel 内列内归约这类细粒度并行。
4. **标量 UB 读写约 18ns/次**：panel 内的标量循环是隐形大户，「小 kernel 慢」必须先排查是否为单线程标量循环（同源实测中一处 `panel_trans` 从 39.95µs 降到 4.36µs，-89%，原因就是每 block 单标量线程拷 4096 元素）。
5. **跨 kernel 的 `volatile` 无必要且有害**：同源实测中全量 volatile 回代使 `sgels` 慢 25%；正确做法是 writer 侧 `asc_threadfence()` + 既有 barrier + reader 普通 cached load。
6. **V 域同管道程序序可靠**（V→V 免栅栏）；V→S 需门控栅栏。

# 需求分析（required）

## 需求描述

| 能力 | 输入 | 输出及行为 |
| --- | --- | --- |
| `Cgeqrf` | m×n 复矩阵 A（`lda×n`） | A 原地保存 R（上梯形）与反射器尾部（严格下三角），`TAU[min(m,n)]` 存系数 |
| `Cungqr` | geqrf 输出的 A 与 `tau[0..k-1]`，`m≥n≥k≥0` | A 原地输出 m×n 的 Q 前 n 列 |
| `Cunmqr` | geqrf 输出的反射器、`tau`、m×n 矩阵 C | C 原地更新为 `op(Q)·C` 或 `C·op(Q)`；`op ∈ {Q, Qᴴ}`，**不支持 T** |
| `CCgels` | m×n 的 A、m×nrhs 的 B，`m≥n` | 独立缓冲输出 n×nrhs 的 X；A 可被破坏，B 保持输入语义 |

四接口均提供 `_bufferSize`。`geqrf`/`ungqr`/`unmqr` 的 workspace 以 **`aclFloatComplex` 元素数**计，`CCgels` 以**字节**计（对齐 cuSolver）。全部矩阵为列主序、Device 存储，支持合法 leading dimension padding，计算遵循调用方 stream，核心计算全部在 NPU AI Core 完成。

## 接口原型（与任务书逐字一致，参数不得增删调换）

公共类型放入 `include/cann_ops_solver_common.h`：

```c
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_OP_N = 0,   /* 不转置 */
    ACLSOLVER_OP_T = 1,   /* 转置；Cunmqr 不支持，传入须报错 */
    ACLSOLVER_OP_C = 2    /* 共轭转置 */
} aclsolverOperation_t;

typedef enum {
    ACLSOLVER_SIDE_LEFT  = 0,
    ACLSOLVER_SIDE_RIGHT = 1
} aclsolverSideMode_t;
```

函数声明放入 `include/cann_ops_solver.h`：

```c
aclsolverStatus_t aclsolverCgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgeqrf(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *TAU, aclFloatComplex *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverCungqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k,
    const aclFloatComplex *A, int lda, const aclFloatComplex *tau, int *lwork);

aclsolverStatus_t aclsolverCungqr(
    aclsolverHandle_t handle, int m, int n, int k,
    aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    aclFloatComplex *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverCunmqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    const aclFloatComplex *C, int ldc, int *lwork);

aclsolverStatus_t aclsolverCunmqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    aclFloatComplex *C, int ldc, aclFloatComplex *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverCCgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclFloatComplex *dA, int ldda, aclFloatComplex *dB, int lddb,
    aclFloatComplex *dX, int lddx, void *dwork, size_t *lwork_bytes);

aclsolverStatus_t aclsolverCCgels(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclFloatComplex *dA, int ldda, aclFloatComplex *dB, int lddb,
    aclFloatComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes,
    int *niter, int *dinfo);
```

维数类型统一 `int`（32-bit，对齐 cuSolver legacy API）。公开头文件须为 C 可调用：使用 `<stddef.h>`、`<stdint.h>` 等 C 兼容头，新增 C API 置于 `extern "C"` 条件段内；历史依赖 `std::complex` 的旧声明继续留在 `__cplusplus` 条件段，互不干扰。交付前须分别验证 C 与 C++ translation unit 的可包含性以及 `sizeof` / `offsetof`（`aclFloatComplex` 必须 8 字节、`real` 偏移 0、`imag` 偏移 4）。

## 参数与语义要点

| 参数 | 归属 | 合法值 / 约定 | 异常行为 |
| --- | --- | --- | --- |
| `handle` | Host | 由 `aclsolverCreate` 创建 | 空 → `ACLSOLVER_STATUS_NOT_INITIALIZED` |
| `m, n, k, nrhs` | Host | 见任务书 §2.4 值域 | 越界 → `INVALID_VALUE`，info 可写时写 `-i` |
| `A / C / dA / dB / dX` | Device | 列主序；`lda≥max(1,m)`、`ldc≥max(1,m)`、`ldda/lddb≥max(1,m)`、`lddx≥max(1,n)` | 规模非空时空指针或 ld 不足 → `INVALID_VALUE` |
| `TAU / tau` | Device | 长度 `min(m,n)` / `k` | `k>0` 或 `min(m,n)>0` 时为空 → `INVALID_VALUE` |
| `Work / work / Workspace` | Device | 元素数 ≥ 查询值 | 不足 → `INVALID_VALUE` |
| `dWorkspace / lwork_bytes` | Device / Host | 字节数 ≥ 查询值 | 不足 → `INVALID_VALUE` |
| `devInfo / dinfo` | Device | INT32[1]，写 0 或 `-i` | 空指针 → `INVALID_VALUE` |
| `niter` | Host | 直接法置 0 | 空指针 → `INVALID_VALUE` |
| `side / trans` | Host | LEFT/RIGHT；N/C | `trans=T` 与非法枚举 → `INVALID_VALUE` |

列主序地址：`A(i,j) = A_base + 8×(i + j×lda)`（单位字节）。只访问逻辑有效区域，`[m, lda)` padding 既不读也不写。所有元素数、字节数与 GM 偏移先扩宽到 `int64_t`/`size_t` 再乘加，并做溢出检查。

## 需求拆解与范围

1. 提供任务书指定的 8 个 C API，参数名称、类型、顺序、Device/Host 归属逐字一致。
2. 三组 QR 接口的 `_bufferSize` 以复数元素数返回，`CCgels` 以字节返回；**查询与执行共用同一规划器**，查询结果必须覆盖执行实际使用的全部临时区。
3. 建立**统一的复数 Householder 约定**（实 beta、相位确定），正确处理共轭、左右乘、反射器遍历方向。
4. 为小矩阵、分块矩阵、非对齐尾块、空问题、非默认 stream 定义确定的分支。
5. `devInfo`/`dinfo` 真实写入（成功与空问题也须经 Device 写 0），非法参数写 `-i`（不计 handle）。
6. 给出可复现的精度、性能、确定性验证计划，性能目标为每个有效金标 case 达到 0.35 倍 GPU。

**不在范围内**：batched / broadcast、行主序、任意二维 stride、其他数据类型（含 COMPLEX32 / DOUBLE COMPLEX）、`trans=T`、`m<n` 的 gels（LQ 路径）、秩亏最小范数解与条件数估计、图融合。

## 代码架构选型

整体为**同一 stream 上的多 Kernel 分块流水**，以 Kernel 边界表达 panel 依赖；AIV 与 AIC 各自独占输出阶段，避免跨核 flag 协议与多 writer 风险：

| 子问题 | 选型 | 原因 |
| --- | --- | --- |
| panel 分解（列内 Householder + 列间 rank-1 更新） | **AIV SIMT 逐列**，warp/block 两级归约 | 列间严格串行依赖，三角结构无法直接进 Cube；`ops-blas geqrf_batched/arch35` 已验证该骨架 |
| 尾随更新 `A22 −= V·Tᴴ·(Vᴴ·A22)` | **AIC Cube FP32 实数块嵌入 Mmad** | 已化为矩阵乘；`E(U)·S(V)` 一次 Mmad 完成复数乘，全路径不降精度 |
| T 因子（larft）与 T 应用 | AIV 小 GEMM（NB×NB） | 规模小、依赖局部，不值得单独 Cube 流水 |
| `Cungqr` / `Cunmqr` 应用层 | 与 geqrf 尾随更新**同一套 larfb Kernel**（作用于不同 C） | 四组合 LEFT/RIGHT × N/C 共用「W=·VᴴC / CV → 乘 T → C−=V·W / W·Vᴴ」流程 |
| `CCgels` | geqrf + unmqr(LEFT, C) Kernel 序列 + AIV 复三角回代 | 组合层无新算法；门禁最宽松（2.87 TF） |
| 打包 / 合并 / 三角回写 / info 门控 | AIV 辅助 Kernel | 规则搬运与 AoS↔平面布局转换；读写区域两两不相交 |

**panel 分解方案的取舍**（本设计的核心决策，详见 §性能优化设计）：

| 方案 | 做法 | 取舍 |
| --- | --- | --- |
| A. Householder 逐列（基线，选定） | panel 内逐列生成 CLARFG 反射器，对 panel 未处理列做 rank-1 更新 | 精度与 LAPACK 逐位同构，最稳；panel 段为串行链 |
| B. 复数 Cholesky QR2（加速候选） | panel = Cholesky QR（`AᴴA` 复数 Cholesky）+ 一次重正交化 + R 修正 | 把串行链换成两次块 GEMM；需复数 Hermitian Cholesky，精度依赖 QR2 的二次迭代 |
| C. 跨核归约 panel | 行分核 + workspace 两段式归约 | 引入核间同步，违反确定性风险高，否决 |

基线取 A：它保证与 LAPACK `cgeqrf` 的反射器逐一对应，`tau` 相位可复现，是**精度与确定性的第一保障**。B 作为设计内已评估的加速路径，按 `min(m,n)` 阈值条件开启，阈值以 950PR 实测校准（依据见 §性能优化设计），不改变公开语义与精度判据。

# 详细设计（required）

## 算子分析

### 数学语义

令 `r = min(m,n)`，第 j 个反射器为 `H_j = I − tau_j·v_j·v_jᴴ`，其中 `v_j` 在第 j 行之前为零、第 j 行隐式为 1，其后元素存于 A 的严格下三角。定义：

```text
Q = H₁ H₂ … H_r
Qᴴ · A原始 = R完整                A原始 = Q · R完整
Q输出 = (H₁ … H_k) · E            E = I_m[:, 1:n]
LEFT  : C输出 = op(Q) · C输入
RIGHT : C输出 = C输入 · op(Q)     op(Q) ∈ {Q, Qᴴ}
CCgels: Y = Qᴴ·B,  R₁₁·X = Y[1:n, :]     （R₁₁ 为 n×n 上三角）
```

`R完整` 为 m×n 上梯形（未占据的下部为零）。`CCgels` 正常路径以满列秩 A 为前提；对严格零对角提供数值诊断，不用除零产生的非有限值充当成功输出。

### 复数 Householder：实 beta 约定（LAPACK CLARFG）

这是与实数 QR 的**第一个关键差异**，也是复数实现最容易出错的地方。对当前列 `α = A(j,j)` 与尾向量 `x`：

```text
当 x 全零且 Im(α)=0 :  tau = 0，H = I，保留 α
否则：
  sgn = (Re(α) < 0) ? −1 : +1          （含 Re(α)=±0 一律取 +1）
  beta = −sgn · sqrt( Re(α)² + Im(α)² + ‖x‖₂² )     ← beta 恒为实数
  tau  = (beta − α) / beta                            ← tau 为复数
  v_tail = x / (α − beta)，v_head = 1
  H = I − tau·v·vᴴ，  Hᴴ·[α; x] = [beta; 0]
```

三点必须写死：

1. **`beta` 取实数**，用 `−sgn(Re(α))` 定号。实数 QR 的 `alpha = (x₁≥0) ? −‖x‖ : ‖x‖` 不能直接照搬到复数，否则 `tau` 的相位随 `Im(α)` 漂移，破坏确定性。
2. **消元必须用共轭 tau**：`A_tail ← A_tail − v·conj(tau)·(vᴴ·A_tail)`。把实数 QR 的 `tau` 更新式直接搬来会**破坏复数分解**（H 一般不满足 `H = Hᴴ`）。
3. **范数必须缩放计算**：先取全部分量绝对值的最大值 `s`，累加 `(分量/s)²` 后乘回 `s`，`s = 0` 单独处理。低于安全阈值时按确定次数的二进制缩放重算，`beta` 写回前恢复尺度。直接平方和在 COMPLEX64 下会溢出。

归约采用**固定段长与固定合并树**，不因数据内容动态选择加法树——这是确定性的前提。

### 复数到实数的块嵌入

对复矩阵 `U = Ur + i·Ui`（p×k）、`V = Vr + i·Vi`（k×q）：

```text
E(U) = [ Ur  −Ui ]  (2p×2k)      S(V) = [ Vr ]  (2k×q)
       [ Ui   Ur ]                      [ Vi ]

E(U)·S(V) = [ Re(U·V) ]  (2p×q)      E(X)·E(X)ᵀ：左上 = Re(XXᴴ)，左下 = Im(XXᴴ)
            [ Im(U·V) ]
```

选型对比：

| 方案 | AIC 次数 | 临时结果块 | 取舍 |
| --- | --- | --- | --- |
| 四次实 GEMM + AIV 合并 | 4 次 | 4 个 p×q | 公式直观，但 launch 与临时块多，四次之间还要保证顺序 |
| **实数块嵌入（选定）** | 1 次 | 1 个 2p×q | 复制实数布局换取更少的 launch 与结果块；只用现有 FP32 MMAD，**不改变误差路径** |

**重要性质**：嵌入乘的实数 MAC 数 `2p×2k×q = 4pqk` 恰等于复数乘加的 flops 计数值（8pqk / 2），因此**嵌入法对 Cube 的实数吞吐要求与复数理论 flops 一一对应，没有冗余放大**。这条性质直接决定了 §性能标准 的门禁可行性判断。

`Hermitian` 积（`Cunmqr` 中的 `V·Vᴴ` 型、`AᴴA` 型）用 `E(X)·E(X)ᵀ`，无需转置打包。共轭转置读（`Xᴴ`、`Vᴴ`）由 AIV 打包 Kernel 在生成 E/S 操作数时完成（读三角 + 取负虚部 + 转置寻址），**不产生独立的转置 Kernel**。

### Cgeqrf：分块右视复数 QR

以面板宽 `nb` 分块，对 panel `j`（列区间 `[j, j+bs)`，`bs = min(nb, n−j)`）：

```text
① panel 分解（AIV，串行列）
   for i in [j, j+bs):
       (β, τᵢ, v) = CLARFG( A[i:m, i] )        # 见 §复数 Householder
       A[i,i] = β;  A[i+1:m, i] = v_tail        # 下三角区存 v（对角隐式 1）
       for c in [i+1, j+bs):
           A[i:m, c] −= v · conj(τᵢ) · (vᴴ · A[i:m, c])     # 列 rank-1 更新
② larft：由 V = A[j:m, j:j+bs] 与 tau[j:j+bs] 形成 T（bs×bs 上三角）
③ 尾随更新（AIC Cube，V22 = A[j:j+bs, j:j+bs]，A22 = A[j:m, j+bs:n)）
   W1 = Vᴴ · A22          # (bs) × (n−j−bs)，K = m−j      —— E(Vᴴ)·S(A22)
   W2 = Tᴴ · W1           # (bs) × (n−j−bs)，三角小乘     —— AIV
   A22 −= V · W2          # (m−j) × (n−j−bs)，K = bs     —— E(V)·S(W2)
```

尾随更新占 geqrf 总 flops 的 ~97.6%（4096² 时约 `4/3·n²(3m−n)·4 ≈ 7.3e11` 复数 flops），panel 段 flops 仅 ~2.4% 但**向量 bound 且串行**。同源实测中 panel 段（含反射器提取与跨 kernel 往返）占 sgeqrf-0104 的 **84%**，这是本设计性能工作的主战场（见 §性能优化设计）。

`m<n` 时仍只生成 `r` 个反射器，最后一个 panel 必须更新其右侧全部剩余列。

### Cungqr：生成显式 Q

```text
1. 复制前 k 个反射器到独立 V 工作区（快照，保护 A 原位覆盖）
2. 初始化 Q = I_m[:, 1:n]（覆盖完整有效输出，含输入上三角与第 k 列之后区域）
3. 按块起点从大到小执行 Q ← Q_p · Q      —— larfb(LEFT, N)，用 T 而非 Tᴴ
4. 按 lda 覆盖回 A 的有效区
```

逆序是关键：`Q = H₁H₂…H_k`，逐块左乘必须从最后一块往前，才能让块内 T 保持**正序乘积**语义。用 `Tᴴ` 会把方向搞反——同源实测中该类符号错误表现为「正交性完美但残差 O(1)」，极易蒙混（P 与 Pᴴ 都正交）。

`k<n` 时仍得到 n 列正交向量；`k=0` 走 Host 恒等分支直接初始化。

### Cunmqr：左右酉变换

反射器遍历顺序按 LAPACK `CUNMQR` 固定（编号 1-based）；分块实现按相同方向遍历块，块内 V/T 始终表示正序乘积：

| side | trans | 单反射器顺序 | 当前块公式 | V 行数 |
| --- | --- | --- | --- | --- |
| LEFT | N | k→1，用 `tau` | `W = VᴴC; C ← C − V(T·W)` | nq = m |
| LEFT | C | 1→k，用 `conj(tau)` | `W = VᴴC; C ← C − V(Tᴴ·W)` | nq = m |
| RIGHT | N | 1→k，用 `tau` | `W = CV; C ← C − (W·T)Vᴴ` | nq = n |
| RIGHT | C | k→1，用 `conj(tau)` | `W = CV; C ← C − (W·Tᴴ)Vᴴ` | nq = n |

两个陷阱：

1. **`nq` 的定义随 side 变化**：LEFT 用 `m`，RIGHT 用 `n`。RIGHT 时反射器矩阵是 `n×k`，若沿用 LEFT 的维数或工作矩阵转置逻辑，会读写越界。精度判据 CQRT03 的分母也随之为 `nq·‖C‖₁·ε`。
2. **输入 A/tau 必须只读**：隐式对角 1 在打包时构造进内部 V，不临时覆盖调用方的 A。

`trans = T` 在 **Host 层直接拦截**（`INVALID_VALUE`，info 写 `-2`），不允许进入 Kernel 后按转置近似实现——任务书 §2.1 与 §7.4 双重强调。

**官方用例覆盖的事实**：`cunmqr/package/canonical_cases.json` 的 189 条用例中，**LEFT×N 92 条、RIGHT×C 97 条，不含 LEFT×C 与 RIGHT×N**。因此性能门禁只需打这两种组合，而功能自验必须自行补齐另外两种（任务书 §3.5 明确要求四组合）。设计上四组合共用同一 larfb Kernel（参数化 side/trans），不因用例缺口而少实现路径。

### CCgels：复数最小二乘

```text
1. 打包 A；将 B 复制到 Y（B 全程不写）
2. 内部复用 Cgeqrf 数学流水（分解 dA）
3. Y ← Qᴴ·B，复用 Cunmqr(LEFT, C) 的块应用（作用于 Y，k = min(m,n)）
4. 检查 R 对角，取最小零对角编号；对角线可用则从最后对角块开始回代
5. 已解 X 块通过 R 的非对角块更新前方 Y（nrhs 小走 Vector GEMV，大走 FP32 块乘）
6. 按 lddx 将 X 写入独立输出 dX
```

- 对角块宽与面板宽一致，块内按**行倒序**做复数回代；不同 RHS 列可并行，行间依赖不可打乱。
- `Y` 的前 n 行最终为 X，后 m−n 行为变换后的残差，**不写入 dX**。
- `niter` 由 Host 置 0（主精度直接分解求解），不把 CUDA 金标的迭代次数当作必须匹配的输出。
- `dinfo`：正常为 0；严格零的 `R(i,i)` 令 `dinfo = i`（1-based 正值）作为数值诊断，抑制 X 写出。该正值不属于非法参数 `-i`，本文也不宣称 cuSolver 规定了相同的正值编码。Host 下发状态与 Device 数值诊断分别读取。
- 不生成完整 Q，不计算 R 的显式逆。

### 确定性分析

任务书要求「相同输入、相同 stream 串行多次执行，输出（含 `tau` / `info`）bit-wise 一致」。本设计的确定性来源：

| 环节 | 确定性保障 |
| --- | --- |
| Tiling | 仅由 `(m, n, k, nrhs, side, trans)` 决定，与数据内容无关 |
| 相位 | CLARFG 实 beta + `−sgn(Re(α))` 定号，`tau` 相位唯一 |
| 归约 | 固定段长 + 固定合并树，不因数据动态换树 |
| 分核 | 面板串行；尾随 GEMM 的 K 维**不跨核**，每输出 tile 唯一 writer |
| 浮点序 | 禁止浮点 atomic、禁止动态 SplitK、禁止数据依赖的分支选择不同求和顺序 |
| workspace | Kernel 只读自己写入过的区域；未初始化填充不参与归约 |
| 输出 | 有效区全部确定写入；padding 保持原内容不被改写 |

## Host/Tiling 设计

### Host 侧校验顺序（冻结）

```text
1. handle == nullptr                          → ACLSOLVER_STATUS_NOT_INITIALIZED
2. side / trans 枚举非法                        → INVALID_VALUE（trans=T 亦在此拦截）
3. 维数值域（m,n,k,nrhs ≥ 0；0≤k≤nq；m≥n for gels） → INVALID_VALUE
4. _bufferSize 输出指针为空                      → INVALID_VALUE（绝对约束）
5. 空问题（m==0 || n==0 || k==0 || nrhs==0）     → SUCCESS，不检查 ld、不访问矩阵、不下发 Kernel、查询返回 0
6. leading dimension 不足                       → INVALID_VALUE
7. 数据 / workspace / info 指针为空（对应规模非 0）→ INVALID_VALUE
8. 生成执行计划，检查 workspace 容量              → 不足则 INVALID_VALUE
9. 向 handle stream 顺序下发 kernel              → SUCCESS（异步语义，已提交）
```

第 2、3、6、7、8 步命中且 info 指针可写时，向 stream 追加一个 AIV 填充 Kernel 异步写 `-i`（`i` 为第 i 个非法参数，**不计 handle**，同时出现多项非法时报告最小编号）。整个过程不做 Host 同步，保持异步语义。

参数序号表（`-i` 的编号基准）：

| 计算接口 | 参数序号 |
| --- | --- |
| Cgeqrf | m=1，n=2，A=3，lda=4，TAU=5，Workspace=6，Lwork=7，devInfo=8 |
| Cungqr | m=1，n=2，k=3，A=4，lda=5，tau=6，work=7，lwork=8，devInfo=9 |
| Cunmqr | side=1，trans=2，m=3，n=4，k=5，A=6，lda=7，tau=8，C=9，ldc=10，work=11，lwork=12，devInfo=13 |
| CCgels | m=1，n=2，nrhs=3，dA=4，ldda=5，dB=6，lddb=7，dX=8，lddx=9，dWorkspace=10，lwork_bytes=11，niter=12，dinfo=13 |

**原始 C 指针的能力边界**：Host 可检查维数、布局约定、空指针与地址属性，但**无法从裸指针识别任意错误 dtype 或完整分配长度**。本文不宣称能做到这一点；输入数值的有限性（NaN/Inf）由 NPU 侧校验并回传 `INVALID_VALUE`，同样以参数序号报告。

### 空问题与特殊分支

| 情况 | 行为 | 查询返回 |
| --- | --- | --- |
| geqrf 的 m=0 或 n=0 | 不访问 A/TAU，写 info=0 | 0 |
| ungqr 的 m=0 或 n=0（其余维数合法） | 不访问 A/tau，写 info=0 | 0 |
| ungqr 的 k=0 且 m,n>0 | 将 A 有效区写成 `I_m[:,1:n]`，tau 可空，写 info=0 | 0（初始化 Kernel 直写 A） |
| unmqr 的 m=0 或 n=0 或 k=0 | C 不变，不访问反射器，写 info=0 | 0 |
| gels 的 m=0 / n=0 / nrhs=0（须 m≥n） | X 输出为空或无右端项；niter=0、dinfo=0 | 0 |

零 workspace 分支允许 work 指针为空，但容量不能为负。未被分支使用的数据指针可为空；`ungqr` 的 `k=0` 且输出非空时仍要求 A 可写。计算接口的 `info` 与 gels 的 `niter` **始终必需**。

### 路径选择阈值（初版，待 950PR 实测校准）

| 接口 | 条件 | 路径 |
| --- | --- | --- |
| Cgeqrf | `min(m,n) ≤ 64` 且最长维 ≤ 64 | direct：单 AIV Kernel 完成无阻塞分解 |
| Cgeqrf | 其他 | blocked，`nb = 32`（尾块取 min） |
| Cungqr | `n ≤ 64` | direct |
| Cungqr | 其他 | blocked，逆序块应用 |
| Cunmqr | `max(m,n)·k` 低于阈值 | direct（单 Kernel 逐列应用） |
| Cunmqr | 其他 | blocked |
| CCgels | 全部 | geqrf 路径 + unmqr(LEFT,C) + 回代；`nrhs ≤ 4` 回代走 Vector GEMV |

`nb = 32` 与各阈值均为**首版确定性初值，不是性能结论**。同源实测显示 panel 宽与线程数是性能主导参数（实数 QR 中默认 `bs` 必须按尺寸分支：`m≥1024` 用 32、`m<1024` 用 64），本任务在 M2 阶段用 950PR 同机 profile 校准后更新本文档。

### TilingData（Host/Device 共用 POD，固定宽度，trivially-copyable）

| 结构 | 核心字段 |
| --- | --- |
| `CgeqrfTilingData` | `m, n, lda, nb, numPanels, directMode, coreNum, numThreads, cqr2Threshold` |
| `CgeqrfPanelTilingData` | `panelStart, bs, mj, nRemaining, lda, reduceMode` |
| `CungqrTilingData` | `m, n, k, lda, nb, numBlocks, directMode, coreNum` |
| `CunmqrTilingData` | `side, trans, m, n, k, nq, lda, ldc, nb, directMode` |
| `CCgelsTilingData` | `m, n, nrhs, ldda, lddb, lddx, nb, rhsVectorMode` |
| `QrGemmTilingData` | `p, q, kdim, tileM, tileN, tileK, srcStride, dstStride, embedMode(E/S/EEᵀ)` |
| `QrPlanMeta` | `workOffset[], workBytes[], c`（workspace 分区与核数） |

布尔状态用明确 `uint32_t` 枚举，禁止 Host 与 Device 各自维护不同字段解释。

### 分核策略

- **核数一律运行时查询**（Host 侧平台接口 / Device 侧 `GetBlockNum`、`GetBlockIdx`），禁止写死 SKU 核数。
- **panel / direct**：矩阵或块粒度铺满 AIV，余量分给前几个核，不做再平衡。
- **Cube GEMM**：M/N 二维切分，在 `块数 ≤ AIC 数` 约束下取最大划分；**K 维不跨核**，避免跨核归约破坏确定性。
- **列切分（反射器提取类 Kernel）**：同源实测表明「每条反射器 GM 往返次数 × 延迟」是瓶颈而非带宽 → 采用 owner + poll 的列切分，**零跨块归约**；线程数按 `m` 自适应（`m ≥ 2048` 用 384，否则 256）。
- **多 kernel 依赖**由同 stream 顺序表达；kernel 内部仅块内同步，不放置需要全部核同时驻留的自旋屏障。

### workspace / bufferSize 口径

查询与执行**共用同一规划器**。定义 `P(x) = 128·⌈x/128⌉`、`A₂₅₆(x)` 为字节数向上对齐 256；内部复数工作矩阵 `L(u,v) = A₂₅₆(8·P(u)·P(v))`（含两个 FP32 平面，嵌入操作数亦由它承载）。

```text
R(c)     = A₂₅₆(4 × 4 × 128 × 128 × 2 × c)        # 每核两槽、每槽四个 FP32 实数积
F(c)     = A₂₅₆(16 × 3 × c + 1024)                # 校验结果 / 每核控制记录 / 计划元数据
S(u,v,c) = L(u,nb) + 2·L(nb,v) + A₂₅₆(8·nb²) + R(c) + F(c)

# 以下 W 全部为字节上界；q = min(m,n)，nq = (side == LEFT) ? m : n，nw = (side == LEFT) ? n : m
W_geqrf = 255 + L(m,n) + S(m,n,c)
W_ungqr = 255 + L(m,k) + L(m,n) + S(m,n,c)
W_unmqr = 255 + L(nq,k) + L(m,n) + S(nq,nw,c)
W_gels  = 255 + L(m,n) + L(m,nrhs) + A₂₅₆(8n) + S(m,max(n,nrhs),c)
```

其中 `S(u,v,c)` 覆盖面板 V 条带、两份 W、T 因子与每核控制记录；嵌入操作数 `E(U)` / `S(V)` 由 `L(·,·)` 复用，不单列。查询中的 `c` 取本调用全部阶段核数的最大值。

- `255` 字节用于对 workspace 起始地址做内部对齐，**不要求调用方指针已满足 256 字节对齐**。
- 三组 QR 接口返回 `⌈W/8⌉` 个 `aclFloatComplex` 元素（并检查不超过 `INT_MAX`），`CCgels` 返回 `W` 字节。
- 小矩阵允许在 UB 内紧凑计算，但查询仍按上述上界返回（保证足量）。

临时区生命周期：

| 区域 | 用途与生命周期 |
| --- | --- |
| geqrf / gels 的 A 平面 | 全程保存分解；末段回写公开 Householder 表示 |
| ungqr / unmqr 的 V 快照 | 从输入反射器复制并补隐式 1/0，保持到全部块应用结束；保护原地覆盖 |
| 面板 V、三角 T | 当前块反射器与因子；下一块可复用 |
| 两份 W | `VᴴC`/`CV` 与乘 T 后的结果；全部读完后才复用 |
| gels 的 Y 与内部 tau | Y 为 B 的独立副本，前 n 行最终保存 X；tau 长度 n |
| R 与 F | 每调用初始化；不沿用上次调用的标志 |

### 工程落点

```text
include/cann_ops_solver_common.h            # + aclFloatComplex / aclsolverOperation_t / aclsolverSideMode_t
include/cann_ops_solver.h                   # + 8 个接口声明（extern "C"）

src/cgeqrf/{cgeqrf_host.cpp, cgeqrf_kernel.cpp, cgeqrf_blocked_kernel.cpp, cgeqrf_tiling_data.h}
src/cungqr/{cungqr_host.cpp, cungqr_kernel.cpp, cungqr_blocked_kernel.cpp}
src/cunmqr/{cunmqr_host.cpp, cunmqr_kernel.cpp, cunmqr_blocked_kernel.cpp}
src/ccgels/{ccgels_host.cpp, ccgels_kernel.cpp, ccgels_blocked_kernel.cpp}
src/utils/qr_common/{clarfg.h, clarft.h, larfb.h, complex_pack.h, qr_gemm.h, param_fill.h}

test/cgeqrf|cungqr|cunmqr|ccgels/
    {CMakeLists.txt, <op>_test.cpp, solver_adapter.h, data/{gen_data.py, verify_accuracy.py, verify_perf.py}, README.md}
docs/zh/{cgeqrf.md, cungqr.md, cunmqr.md, ccgels.md}  + docs/api_list.md + README.md 接口表
```

三个接口共享 `qr_common` 的 larfg / larft / larfb / 打包实现，**不复制多份**；`complex_pack.h` 仅一个共享件，头文件内联。文件数量须符合仓内评审要求（不出现明显冗余的测试/编译文件）。

## Kernel 设计

### Kernel 清单

| Kernel | 核型 | 职责 |
| --- | --- | --- |
| `geqrf_direct_kernel` | AIV | 小矩阵单 Kernel 无阻塞复数 QR（含 tau 与 info 写入） |
| `geqrf_panel_kernel` | AIV | 当前面板逐列 CLARFG + panel 内 rank-1 更新 |
| `geqrf_pack_kernel` | AIV | A 解交错为实/虚平面；生成 `E(V)`/`S(A22)`/`E(Vᴴ)` 操作数 |
| `geqrf_gemm_kernel` | AIC | `W1 = Vᴴ·A22` 与 `A22 −= V·W2` 的 FP32 嵌入 Mmad |
| `larft_kernel` | AIV | panel 的 T 上三角递推 |
| `larfb_apply_kernel` | AIC+AIV | 块反射器应用（ungqr 逆序 / unmqr 四组合 / gels 的 QᴴB） |
| `cqr2_panel_kernel` | AIV+AIC | 可选加速：panel 的 Cholesky QR2（条件开启） |
| `trsm_kernel` | AIV | gels 复三角回代 + 前方 Y 更新 |
| `init_identity_kernel` | AIV | ungqr 的 `k=0` 恒等分支与 Q 初始化 |
| `param_fill_kernel` | AIV | Host 检出的非法参数向 Device info 异步填 `-i` |
| `info_write_kernel` | AIV | 成功 / 空问题的 Device info 置 0 |

### panel 分解伪代码（基线路径）

```text
panelStart = j; bs = min(nb, n − j); mj = m − j
for i in [j, j+bs):
    # 列内归约：warp-per-column 批量点积，每个反射器仅 1 次屏障
    (norm2, α) = parallel_dot( A[i:m, i] )              # 缩放平方和 + 固定合并树
    if norm2 == 0 && Im(α) == 0:  tau[i]=0; continue
    beta = −sgn(Re(α)) · sqrt(norm2)
    tau[i] = (beta − α) / beta
    v_head = α − beta
    A[i,i] = beta
    A[i+1:m, i] = A[i+1:m, i] / v_head                  # 复数缩放除法（按分量缩放）
    # panel 内列更新：SIMT 批量，去屏障（反射器之间仅 1 个）
    for c in [i+1, j+bs):
        t = vᴴ · A[i:m, c]                              # 批量 dot
        A[i:m, c] −= v · conj(tau[i]) · t
TAU[j:j+bs] = tau[j:j+bs]
```

**同源实测的三条硬约束**（直接迁移，不重复踩坑）：

1. **禁止「每个点积做一次整块 reduction」**：单点 barrier 是 µs 级，乘以反射器数平方后变成几十毫秒。必须批量点积 + 每反射器 1 次屏障。
2. **线程数下限 256**：否则小 `m` 时 warp 数不足，每 warp 串行处理十几列。
3. **跨 kernel 的 volatile 无必要**：writer 侧 `asc_threadfence()` + barrier + reader 普通 cached load 即可；全量 volatile 在同源实测中使 sgels 慢 25%。

### larft 与 larfb

```text
larft（FORWARD，bs×bs 上三角）：
for i in [0, bs):
    if tau[i] == 0:  T[:, i] = 0; continue
    T[i,i] = tau[i]
    if i > 0:
        T[0:i, i] = −tau[i] · ( T[0:i, 0:i] · ( V[:, 0:i]ᴴ · v_i ) )

larfb（块应用，四个 side/trans 组合共用）：
LEFT : W = Vᴴ·C ; W ← op(T)·W ; C −= V·W          （op(T)：trans=N→T，trans=C→Tᴴ）
RIGHT: W = C·V  ; W ← W·op(T); C −= W·Vᴴ
```

- `larft` 的递推式用 `Tᴴ` 还是 `T` 是**符号陷阱**：`P = I − V·Tᴴ·Vᴴ` 与 `Pᵀ = I − V·T·Vᴴ` 都正交，用错时正交性检查全绿、残差 O(1)。同源实测因此产生过 `residual 0.46~0.50` 的批量回归，判据是「残差量级」而非「正交性」。
- `T` 与 `tau` 的全局偏移必须按 panel 起址累加，否则每个 panel 覆写 `TAU[0..nb)`，多面板正交性崩坏（同源实测踩过）。

### 同步与生命周期

| 边界 | 同步设计 |
| --- | --- |
| Host 阶段间 | 同 stream 顺序下发：pack → panel → larft → GEMM → … |
| panel 列间 | block 内 barrier（每反射器 1 个）；面板间用 **Kernel 边界**，不用跨核自旋 |
| AIC L1/L0 | L1 双缓冲、L0A/L0B ping-pong 的 HardEvent 配对，循环结束完整 drain |
| AIV↔AIC | 事件编号按类型计数不超过 8；控制记录随调用初始化，释放前保证写回可见 |
| workspace 复用 | 后一 Kernel 仅在同 stream 前一 Kernel 完成后消费/覆写对应区 |
| 多 stream | 状态与 workspace 归属 handle；不使用进程级可变全局 scratch |
| 结果读取 | 返回 SUCCESS 仅表示已提交；调用方使用输出前须等待 handle stream |

## 性能优化设计

### 门禁画像（自行按 `bench_result.json` 的 `gflops_avg` 口径核算）

| 算子 | 最紧 case | GPU `avg_ms` | NPU 预算 | 需有效吞吐 |
| --- | --- | ---: | ---: | ---: |
| **cunmqr** | 4096×4096, k=4096 | 14.2046 | 40.58 ms | **13.55 TF（全局最紧）** |
| cungqr | 4051×2941, k=735 | 3.1883 | 9.11 ms | 12.06 TF |
| cgeqrf | 4096×4096 | 34.8149 | 99.47 ms | 3.68 TF |
| cgels | 3740×3706 | 33.5620 | 95.89 ms | 2.87 TF |

固定开销敏感档（预算即绝对时间上限，与 flops 无关）：

| 算子 | case | 预算 | 说明 |
| --- | --- | ---: | --- |
| cunmqr | m=n=k=2 | **29.7 µs** | 必须单 kernel 直算，多 kernel 流水必然超时 |
| cunmqr | m=n=k=128, k=1 | 35.1 µs | 同上 |
| cgeqrf | m=32, n=4 | 42.3 µs | direct 路径 |
| cgeqrf | m=2, n=2 | 45.1 µs | direct 路径 |
| cgels | m=n=2 | 737 µs | 组合链有余量，但仍需收敛 launch |

**结论一（门禁可行性）**：全局最紧需 13.55 TF。同源实测 arch35 FP32 GEMM 参考实现：4096³ 峰值 **20.33 TF**、`K=128` 段 **18.11 TF**、`M=128` 大 K 段 **19.41 TF**。结合 §复数到实数块嵌入 的性质（嵌入法实数 MAC 数 = 复数 flops 计数值，无冗余），`cunmqr` 最紧 case 的余量约 **34%**，比实数 QR 任务的 12% 更宽松。**但这是未经本任务实测的推算，不作为达标声明。**

**结论二（真正的风险在小形状与 panel）**：门禁最难的不是大 GEMM，而是 42 条预算低于 1ms 的用例——它们由固定 launch 开销与 panel 串行链支配，吞吐再高也救不回来。

### panel 串行链是主战场

同源实测（实数 QR 10-3，950PR 同机）给出的量化事实：

| 观察 | 数据 | 设计后果 |
| --- | --- | --- |
| panel 段（含反射器提取的 GM 往返）占 sgeqrf-0104 的 84%，该 case 门禁余量仅 1.3% | 每反射器约 9µs × n | panel 是本任务**第一优先优化对象** |
| 反射器提取是「每条反射器的 GM 往返次数 × 延迟」受限，不是带宽或 barrier 受限 | 列切分 owner+poll 后 341 → 255µs | 采用零跨块归约的列切分 |
| `panel_trans` 原为每 block 单标量线程拷贝 4096 元素 | SIMT 化后 39.95 → 4.36µs（−89%） | **「小 kernel 慢」先查单线程标量循环** |
| 小 kernel 空壳探针仅 0.81µs，全做 60.6µs | 57µs 全在 UB load-use 依赖链（167ns/项） | 4 路独立累加器 ILP：60.07 → 41.68µs（−31%）；8 路反而溢出寄存器变差 |

对复数 QR 的影响：复数点积的工作量是实数的 4 倍，且每个反射器多一次复数除法与相位判定，panel 段占比会**高于**实数的 84%。因此本设计把两条路径都保留：

### 加速候选：复数 Cholesky QR2（CQR2 panel）

用一次 Cholesky QR 加一次重正交化替换 panel 内的逐列 Householder 链：

```text
panel A₁（m×bs）:
  ① W = A₁ᴴ·A₁                     —— E(A₁)ᵀ·E(A₁) 一次 Mmad（Hermitian 积）
  ② R₁ = chol(W)                    —— 复数 Hermitian 正定 Cholesky（AIV，bs×bs）
  ③ Q₁ = A₁·R₁⁻¹                    —— 三角右除（AIV 或 TRSM 型）
  ④ 重正交化：W₂ = Q₁ᴴ·Q₁, R₂ = chol(W₂), Q = Q₁·R₂⁻¹, R = R₁·R₂
  ⑤ panel 的 tau / 反射器表示由 Q 反解（保证对外仍是 LAPACK Householder 语义）
```

- **精度不劣化**：QR2 的两次迭代把正交性残差从 `O(κε)` 压到 `O(ε)`，在 CQRT02（`‖I−QᴴQ‖₁/(mε) ≤ 1.5`）与 CQRT01 上**优于**单次 Householder；同源实测在实数链路全 PASS（resid ~4e-08，余量约 2000 倍；官方随机数据 cond≈1.2，距分解失稳点有 4 个数量级余量）。
- **性能收益（同源实测，实数）**：4096² 88.67 → 63.34ms，门禁余量由不足转为 12.9%；1024² 9.95ms（过 8.2%）、2048² 23.45ms（过 6.5%）。
- **代价与风险**：需要复数 Hermitian Cholesky（9 月复数 Cholesky 任务已有工程资产）+ 反射器反解步骤。因此**默认关闭，按 `min(m,n) ≥ 阈值` 条件开启**，阈值由 950PR 实测校准；关闭时不影响任何公开语义。
- **兼容性**：`tau` 的相位仍按 CLARFG 约定输出，`H = I − tau·v·vᴴ` 与 LAPACK 逐一对应，`Cungqr`/`Cunmqr` 的输入契约不变。

同源实测还给出「**不能无脑全开**」的边界：小形状上 Householder 反而快 35~40%（135/158 等），因此该路径必须有明确的 `m` 下限。

### 小形状与固定开销治理

| 手段 | 说明 |
| --- | --- |
| direct 单 Kernel 路径 | `min(m,n) ≤ 64`（初值）时一个 AIV Kernel 完成全部工作，杜绝多 launch |
| UB 内紧凑布局 | 小矩阵不打包到 GM workspace，避免一次额外往返 |
| Kernel 合并 | 打包与 panel 合并、合并与回写合并；但**不允许跨 Cast 位宽边界合并**（见 §算子约束限制） |
| launch 预算表 | M2 阶段对 42 条 <1ms 用例逐条建列 launch 预算，超预算即改 direct 路径 |

### 计时口径声明（须在验收前对齐）

- 门禁口径：`T_NPU ≤ T_GPU/0.35`，取 msprof `OpBasicInfo.csv` 的 `Task Duration(us)`。
- **多 Kernel 口径**：geqrf / ungqr / unmqr / gels 的一次 API 调用会下发 pack、panel、larft、GEMM、回代、回写等多个 kernel。本设计采用「**同一次调用内全部必要 kernel 的 Task Duration 求和**，再对重复调用求平均」，不以某个最快 kernel 替代完整流水。该声明将写入自测报告，并在验收前与验收方确认。
- **event 计时不可信**：同源实测 event 计时对小用例自差可达 305%，凡判 OVER 必须用 msprof 复核，不以 event 结论直接定案。

# 硬件与约束

## 支持硬件

| 芯片版本 | 支持状态 |
| --- | --- |
| Ascend 950PR（Atlas 950 / DAV_3510 / `arch35`） | 支持，本任务唯一验收目标 |
| 其他产品 | 本任务不声明支持；公共 API 保持产品线共享 |

## 平台能力与资源预算

- 核数、UB/L1/L0 容量一律**运行时查询**，文档中的数值仅为示例预算，不写成跨产品常量。
- 初始实数 tile 预算（`tileM=tileN=128, tileK=32`）：

```text
L1  = 2×(128×32 + 32×128)×4 = 64 KiB
L0A = 2×128×32×4 = 32 KiB        L0B = 2×32×128×4 = 32 KiB
L0C = 128×128×4 = 64 KiB
```

嵌入后操作数规模翻倍（`2p×2k`），tile 选择须按 `embedMode` 重新校验：`E(U)` 的 L1 占用为 `2·tileM × 2·tileK × 4B`。

- 面板 AIV 最长向量为 4096 个复数：`v`、双缓冲目标列、复数运算临时区、范数临时区与 `T` 的预算为 `72q + 8b² + 8192` 字节（`q=4096, b=32` 时约 320 KiB，须按实际 UB 上限裁剪分块）；**不将整个 4096×32 面板常驻 UB**。
- 若启用 CQR2 panel，额外需要 `bs×bs` 复数 Hermitian 矩阵（`E` 嵌入后 `4bs²` 复数）与两次 R 的逆，预算并入上面的 `S(u,v,c)`。

## 算子约束限制

| 约束 | 说明 |
| --- | --- |
| 数据类型 | COMPLEX64（`aclFloatComplex`）；主精度、矩阵乘输入与累加、中间复数运算全程 FLOAT32，不使用半精度或 HF32 |
| 存储 | Column Major；仅支持由 ld 表达的列间 padding |
| 内存位置 | 全部矩阵、tau、info、workspace 均在 Device；Host 不承担 H2D/D2H |
| 原地/异地 | geqrf/ungqr/unmqr 原地覆写；gels 的 dA 可被破坏、dB 只读、dX 异地独立输出 |
| 重叠 | A 与 C、dB 与 dX、矩阵与 workspace 之间的写入重叠不在支持范围 |
| 反射器输入 | ungqr/unmqr 的 A 必须是 geqrf 输出的 Householder 表示，不接受未分解的原始 A |
| trans | 仅 N / C；**T 在 Host 层报错拦截**，不得按转置实现 |
| gels 形状 | 仅 `m ≥ n`；`m < n` 报错 |
| 零维 | 合法 no-op：不访问矩阵、不分配 workspace、不下发 Kernel、查询返回 0 |
| 规模 | `m,n ∈ [1,4096]`（验收下限，允许实现支持更大并声明）；`nrhs ∈ [1,128]` |
| broadcast / 图融合 | 不支持；每次调用按本次维数生成 tiling |
| 自构造数据 | 输入占用内存须在 4 GB 以内 |
| 确定性 | 同输入、同 tiling、同软件版本 bit-wise 一致；不承诺跨版本 bit-exact |
| Cast 位宽 | **禁止跨 Cast 位宽的指令合并**（同源实测：i32→f16 单条大 count 直转曾产生 99.7% 坏点，须两段小 count 链） |
| SIMT 入口 | `__simt_vf__` + `LAUNCH_BOUND` 必须成对 |

# 可维可测分析

## 精度标准

精度判定按任务书 §3.2.1 的 LAPACK 残差判据执行，ε = 2⁻²⁴ ≈ 5.96e-8，矩阵 1-范数 `‖M‖₁ = max_j Σ_i |M_ij|`：

| 接口 | 判据溯源 | 残差定义 | 通过条件 |
| --- | --- | --- | --- |
| Cgeqrf | CQRT01 | `ratio = ‖R − Qᴴ·A‖₁ / (m·‖A‖₁·ε)`，Q 用 NPU 输出的 (v,tau) **升精度至 complex128** 后由 CPU 等价 `zungqr` 生成 m×m 完整酉矩阵 | `ratio ≤ max(5·ratio_cpu, 0.1)` |
| Cungqr | CQRT02 | `ratio = ‖Iₙ − Qᴴ·Q‖₁ / (m·ε)`，分母固定用行数 m | `ratio ≤ max(5·ratio_cpu, 1.5)` |
| Cunmqr | CQRT03 | `ratio = ‖op2 − op1‖₁ / (nq·‖C‖₁·ε)`，op2 由**同一份 NPU (v,tau)** 经 NPU ungqr 生成显式 m×m Q 后按同 side/trans 显式相乘 | `ratio ≤ max(5·ratio_cpu, 0.1)`；**ungqr 须先通过验收** |
| CCgels（方阵 m=n） | CQRT16 | 逐列 `resid_j = ‖B_j − A·X_j‖₁ / (m·‖A‖₁·‖X_j‖₁·ε)`，取 max | `ratio ≤ max(5·ratio_cpu, 0.015)` |
| CCgels（超定 m>n） | CQRT17 | `R_res = B − A·X`，`ratio_opt = ‖R_resᴴ·A‖₁ / (‖A‖₁·‖B‖₁·max(m,n,nrhs)·ε)` | `ratio_opt ≤ 10·ratio_opt_cpu` |

实现要求：

1. **升精度构造参考 Q**：`zungqr` 一律在 complex128 下执行，使残差只反映 NPU geqrf 本身的质量（CQRT01 口径）。
2. **`ratio_cpu` 必须独立**：CPU 参考链路使用同一个 case 的输入副本、独立实现（NumPy/SciPy LAPACK），**绝不与内核共用公式**。同源实测的血案是分级自检照抄内核公式，错也报 `bad=0`。
3. **超定 must 用 CQRT17 终裁**：随机 B 的超定问题最优残差非零，CQRT16 的分子含问题本身的不相容残差，恒超阈值、无判别力。
4. **退化处理**：分母为零且分子为零记 `ratio = 0`；分子非零记 `∞`，不用 NaN 通过判定。
5. **零阈值情形**：若 `ratio_opt_cpu = 0` 而 NPU 出现非零误差，按任务书零阈值处理并**记录分析**，不自行加容差。
6. **不伪造通过**：`Cungqr` 除正交性外**加测其输出与同一反射器乘积的关系**，避免「错误地生成单位阵」也通过单独的正交性判据。

## 性能标准与测量设计

1. **门禁**：对每个有效金标 case，`T_NPU ≤ T_GPU/0.35`，`T_GPU_us = 1000×perf.avg_ms`。
2. **采集**：`msprof op --application="{BUILD_DIR}/test/${op}/${op}_test --output=./prof_${op}`，取 `OPPROF_*/OpBasicInfo.csv` 的 `Task Duration(us)`；建议收敛到单条用例，避免多条用例 kernel 混叠。
3. **计时区间**：包含本接口所需的打包、info 写入、数值校验与全部计算 kernel；**不包含** workspace 查询、测试数据生成与验证 D2H。`ungqr`/`unmqr` 的输入准备 geqrf 在区间外；`gels` 内部 QR 计入本接口。
4. **原地接口**每轮恢复同一输入；预热与重复次数与 CUDA 金标口径一致；有效采样 > 50 次取平均。
5. **基线完整性**：本任务四目录 `bench_result.json` 的 `failed` 均为 0，`cgeqrf` 174、`cungqr` 165、`cunmqr` 189、`cgels` 166 条**全部带 `perf`**，不存在实数 QR 任务中 108 条缺基线的问题。关联时须按**完整参数元组**匹配，`cunmqr` 的 side/trans 存在字符串与整数两种表示，关联前须归一化，不能只按 m/n 或下标匹配。
6. **本文不填任何 NPU 实测耗时，也不声明任何 case 已过门禁。**

## 覆盖规划

| 类别 | 覆盖点 |
| --- | --- |
| 基础链路 | 四接口单测；geqrf→ungqr 还原 Q；geqrf→unmqr；gels→A·X 与最优性残差 |
| 形状 | 0/1、2 的幂、2 的幂−1、4096 上限；geqrf 的 m>n/m=n/m<n；ungqr 的 k=n/k<n/k=0 |
| 分块边界 | 31/32/33、63/64/65、127/128/129；多面板、尾块、矩形尾块 |
| side / trans | **四组合全测**（LEFT/RIGHT × N/C）；特别覆盖非方阵 C；`trans=T` 与非法枚举报错 |
| 反射器 | `tau=0`、纯虚对角、`Re(α)=±0` 的相位边界、每 panel 的 tau 全局偏移 |
| padding | lda/ldc/ldda/lddb/lddx 取最小合法值、+8、+32；padding 填哨兵并检查保持 |
| CCgels | m=n / m>n；nrhs=1/8/32/128/0；m<n 报错；dB 保持只读；dX 独立 |
| workspace | 查询值足量；少一个元素/字节报错；负 lwork；非空问题空 work；查询与执行一致 |
| info | 各参数逐项非法核验 `-i`；info 为空；成功与空问题经 Device 真实写 0 |
| 数据值 | NaN/Inf 输入报错；padding 的非有限值被忽略 |
| 流与确定性 | 非默认 stream 正确；同输入串行重复有效输出/tau/info **bit-wise 一致**；原地输入还原后重跑 |
| 公共 ABI | C/C++ 头文件包含；`sizeof`/`offsetof`；8 个原型与任务书逐字一致 |

自构造输入按任务书 §3.5 分布：实部、虚部各 50% 均匀 [-5,5]；50% 正态（μ∈[-5,5]、σ∈[0.1,2]）。反射器输入一律经 geqrf 生成；gels 的精度 case 使用满列秩矩阵，数值失败 case 单独验证诊断。

**实现纪律**：内部模块按「复数基础运算 / Householder 面板 / 块应用 / 三角求解 / workspace 与参数校验」划分，四接口共享，不为每个接口复制同一数学实现；CPU NumPy/SciPy 只作测试参考，**不得成为算子执行的 fallback**。

## 兼容性分析

八个接口均为新增公共 API，不修改既有 `aclsolverCgetrf` 等签名与行为。新增实现位于 `src/` 各算子目录，通过既有 handle / stream / workspace 机制执行；内部路径枚举与 TilingData 不暴露 ABI。旧复数 LU 的 `std::complex<float>` 与 `aclError` 返回**不作为新接口模板**；若确需短暂兼容，须在头文件与文档给出对照表，并以 `aclsolverStatus_t` 公开接口为最终验收口径。

## 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| panel 串行链占比高于实数 QR（同源实测已达 84%） | cgeqrf 大 case 与 42 条 <1ms 小 case 双线吃紧 | ①批量点积 + 每反射器 1 屏障；②列切分 owner+poll 零跨块归约；③CQR2 panel 按阈值开启 |
| CQR2 需要复数 Hermitian Cholesky | 新增实现与精度风险 | 复用 9 月复数 Cholesky 工程资产；默认关闭、阈值开启；反射器仍按 CLARFG 相位输出 |
| 复数 Householder 用错 `tau`/`conj(tau)` | 正交性全绿但残差 O(1)（`P` 与 `Pᴴ` 都正交） | 判据以残差量级为准；四组合 + 多面板专项用例；`larft` 的 `T`/`Tᴴ` 符号单测 |
| RIGHT 分支把 `nq` 误用为 `m` | 越界或精度判据分母错误 | `nq = LEFT ? m : n` 写死于 tiling；RIGHT 专项用例 |
| 42 条 <1ms 用例被 launch 开销吃掉 | 门禁逐 case 判定，个别失败即不达标 | direct 单 Kernel 路径 + M2 逐条 launch 预算表 |
| `Cunmqr` 官方用例缺 LEFT×C / RIGHT×N | 若只按用例实现会漏路径 | 设计与实现强制四组合；自构造用例补齐 |
| 多 Kernel 计时口径分歧（求和 vs 单 kernel） | 结论可能相反 | 本文显式声明求和口径，验收前书面确认 |
| 950 环境跳板授权窗口（约 5 分钟） | 调试效率受限 | 单次连接批量执行、长任务 nohup、结果另授权回收 |
| 同一账号在多个任务复用 | 平台「一号一任务」限制可能冲突 | 与用户确认账号归属后再报名与提交验收 |

# 参考资料

1. 任务书：`Atlas950_Cgeqrf_Cungqr_Cunmqr_CCgels_task_doc.md`；用例与金标：`cgeqrf|cungqr|cunmqr|cgels/{cases.json, bench_result.json, package/}`。
2. 社区设计模板与流程：`cann-competitions/04_tasks/01_community-task-2026/{resources/design_template.md, README.md}`；[社区任务流程与注意事项](https://gitcode.com/org/cann/discussions/39)。
3. 目标仓：[cann/ops-solver](https://gitcode.com/cann/ops-solver)，参考 `src/sgetrf/`、`src/cgetrf/`、`include/cann_ops_solver*.h`、`test/`。
4. 同源事实源：[cann/ops-blas](https://gitcode.com/cann/ops-blas) 的 `blas/geqrf_batched/arch35/`、`blas/gels_batched/arch35/`、`blas/gemm*/arch35/`。
5. [CATLASS](https://gitcode.com/cann/catlass)：950 FP32 样例 `examples/43_ascend950_basic_matmul/`、`gemm/dispatch_policy.hpp`、`arch/arch.hpp`。
6. Netlib LAPACK：[cgeqrf](https://www.netlib.org/lapack/complex/cgeqrf.f)、[clarfg](https://www.netlib.org/lapack/complex/clarfg.f)、[clarft](https://www.netlib.org/lapack/complex/clarft.f)、[clarfb](https://www.netlib.org/lapack/complex/clarfb.f)、[cungqr](https://www.netlib.org/lapack/complex/cungqr.f)、[cunmqr](https://www.netlib.org/lapack/complex/cunmqr.f)、[cgels](https://www.netlib.org/lapack/complex/cgels.f)；测试例程 [cqrt01](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cqrt01.f/)、[cqrt02](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cqrt02.f/)、[cqrt03](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cqrt03.f/)、[cqrt16](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cqrt16.f/)、[cqrt17](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cqrt17.f/)。
7. NVIDIA cuSolver DN legacy API：<https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api>。
8. 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
9. CANNBot Skills（架构、Tiling、Buffer 与验证边界核验）：<https://gitcode.com/cann/cannbot-skills>。
