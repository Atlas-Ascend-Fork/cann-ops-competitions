# 需求背景（required）

## 需求来源

- 来源：昇腾社区任务「10月社区任务-单精度复数 LU 分解、求解和批量求逆（950）」，
  任务书 `Atlas950_Cgetrf_Cgetrs_CgetriBatched_CmatinvBatched_task_doc.md`。
- 交付目标：将 ops-solver 仓（https://gitcode.com/cann/ops-solver ）中 4 个复数求逆族原型接口
  升级为公开接口并对标 cuSolver / cuBLAS，验收通过后以 PR 形式合入 ops-solver 仓。
- 本任务 4 个接口作为同一社区任务一并交付，不允许拆分验收：
  `aclsolverCgetrf`、`aclsolverCgetri`、`aclsolverCgetriBatched`、`aclsolverCmatinvBatched`
  （配套 `aclsolverCgetrf_bufferSize`、`aclsolverCgetri_bufferSize`）。

## 背景介绍

### 算子族功能定位

| 接口 | 定位 | 对标基线 | 输入语义 |
|---|---|---|---|
| `aclsolverCgetrf` | 单矩阵复数 LU 分解（带部分主元） | `cusolverDnCgetrf` | 原始矩阵 A，原地分解为打包 L/U + 主元 |
| `aclsolverCgetri` | 单矩阵求逆（基于已分解 LU） | LAPACK `cgetri`（调用约定对齐 cuSolver DN） | 已分解 LU 因子 + 对应主元 |
| `aclsolverCgetriBatched` | 批量求逆（基于已分解 LU） | `cublasCgetriBatched` | batch 个已分解 LU 因子 + 主元 |
| `aclsolverCmatinvBatched` | 小矩阵直接批量求逆 | `cublasCmatinvBatched` | batch 个原始矩阵，直接求逆（getrf+getri 短路） |

### 原型与公开接口的差距（本次设计要消除）

ops-solver 仓已有 4 个接口原型，但与对标接口不对齐：

| 差距项 | 原型行为 | 公开接口目标语义 |
|---|---|---|
| 指针归属 | Host 矩阵指针、Host 标量 info | 全部 Device 指针；info 真实写回 Device |
| 数据布局 | 连续 `[batch, n, n]` 张量、`lda` 强制等于 `n` | 列主序 + Device 指针数组；允许 `lda`/`ldc`/`lda_inv` padding |
| workspace | 无 workspace 查询 | `getrf`/`getri` 提供 `_bufferSize`，由调用方在 Device 侧切分 |
| `CmatinvBatched` n≥32 | 静默转调 `CgetriBatched` | `n > 32` 必须返回 `ACLSOLVER_STATUS_INVALID_VALUE`，绝不转调 |
| 返回值 | 部分为 `aclError` | 统一 `aclsolverStatus_t`，状态码语义对齐 cuSolver |

### 现有实现工程模式

Host C API + AscendC/CATLASS Kernel 直调（非 aclnn 两段式、非 PyTorch 接口）。公开头文件
`include/cann_ops_solver.h`、`include/cann_ops_solver_common.h`；实现分别位于
`src/cgetrf/`、`src/cgetri/`、`src/cgetri_batched/`、`src/cmatinv_batched/`。

# 需求分析（required）

## 需求描述

在 Ascend 950PR（Atlas 950，arch35）上，用 AscendC/CATLASS 实现上述 4 个复数求逆族公开接口，
满足：

1. 列主序（column-major）+ Device 指针语义；`lda/ldc/lda_inv ≥ max(1, 阶数)` 且允许 padding，
   禁止再要求 `lda == n`。
2. `aclsolverCgetrf`/`aclsolverCgetri` 提供 `_bufferSize`；workspace 为调用方分配的 Device 缓冲，
   算子内部不做隐蔽分配。
3. `devInfo`/`info`/`infoArray` 必须真实写入：`0` 成功；`-i` 第 i 个参数非法（不计 handle）；
   `i > 0` 表示 `U(i,i)=0`（奇异）；批量接口对每个 batch 独立写入 `info[i]`。
4. 确定性：同输入、同 stream 串行多次执行，输出（含 `ipiv`/`info`）bit-wise 一致。
5. 核心计算在 NPU AI Core 上完成，禁止 CPU fallback。

## 需求拆解

| 子问题 | 交付内容 | 关键设计点 |
|---|---|---|
| S1 单矩阵 LU 分解 | `aclsolverCgetrf` + `_bufferSize` | 列主序右视 LU、部分选主元、原地打包 L/U、1-based ipiv、奇异检测 |
| S2 单矩阵求逆 | `aclsolverCgetri` + `_bufferSize` | 由 LU 因子求逆（`inv(U)·inv(L)·P^T`）、workspace X、逆序列交换 |
| S3 批量求逆（已分解） | `aclsolverCgetriBatched` | Device 指针数组、batch 条带化到 AIV 核、`Carray` 复用为 scratch |
| S4 小矩阵直接批量求逆 | `aclsolverCmatinvBatched` | 原地 Gauss-Jordan + 部分选主元、`n ≤ 32` 硬约束、UB 成对复数平面向量化 |
| S5 公共设施 | `cann_ops_solver_common.h` 类型/状态码、handle/stream | `aclFloatComplex` 与 `cuComplex` 布局一致；复用 `aclsolverCreate` 等 |

# 详细设计（required）

## 算子分析

### 公开接口原型

| 函数 | 角色 | workspace 形参 | 备注 |
|---|---|---|---|
| `aclsolverCgetrf_bufferSize(handle, m, n, A, lda, *Lwork)` | 查询 | 无 | 返回 `Lwork = 0` |
| `aclsolverCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)` | 计算 | `Workspace`（未用） | 对齐 `cusolverDnCgetrf`，不带 `Lwork` |
| `aclsolverCgetri_bufferSize(handle, n, A, lda, *Lwork)` | 查询 | 无 | 返回 `Lwork = n*n` |
| `aclsolverCgetri(handle, n, A, lda, devIpiv, Workspace, Lwork, devInfo)` | 计算 | `Workspace` + `Lwork` | 对齐 LAPACK `cgetri`，带 `Lwork` |
| `aclsolverCgetriBatched(handle, n, Aarray[], lda, PivotArray, Carray[], ldc, infoArray, batchSize)` | 计算 | 无 | `Carray[i]` 兼作 scratch，对齐 `cublasCgetriBatched` |
| `aclsolverCmatinvBatched(handle, n, A[], lda, Ainv[], lda_inv, info, batchSize)` | 计算 | 无 | `Ainv[i]` 兼作 scratch，对齐 `cublasCmatinvBatched`；`n ≤ 32` |

### 数学公式

**S1 aclsolverCgetrf（LU 分解，带部分主元）**

$$ P \cdot A = L \cdot U $$

其中 $A$ 为 $m \times n$ 单精度复矩阵（COMPLEX64），$P$ 为置换矩阵，$L$ 为单位下三角（梯形）矩阵，
$U$ 为上三角（梯形）矩阵。分解结果原地写入 A：对角线及上方为 $U$，严格下方为 $L$ 非对角元。
`devIpiv` 为 1-based 主元序列（第 $i$ 行与第 `devIpiv(i)` 行交换）；`devIpiv == nullptr` 时不做选主元，
分解为 $A = L \cdot U$（语义对齐 `cusolverDnCgetrf`）。

**S2 aclsolverCgetri / S3 aclsolverCgetriBatched（由 LU 因子求逆）**

$$ A^{-1} A = I, \qquad A^{-1} = U^{-1} \, L^{-1} \, P^{T} $$

输入 A 为 `aclsolverCgetrf` 得到的打包 LU 因子，`devIpiv` 为对应 1-based 主元；
单矩阵接口原地覆盖 A 为 $A^{-1}$，批量接口对 $i=0,\dots,\text{batchSize}-1$ 计算
$C[i] = \mathrm{inv}(A[i])$，且 $C[i]$ 与 $A[i]$ 内存不重叠（out-of-place）。

**S4 aclsolverCmatinvBatched（直接批量求逆）**

$$ C[i] = \mathrm{inv}(A[i]), \quad i = 0,\dots,\text{batchSize}-1, \quad n \le 32 $$

对原始矩阵直接求逆（getrf+getri 短路），不走 LU 分解链；$A[i]$ 只读、永不改写。

### 支持数据类型

| 张量 | 数据类型 | 说明 |
|---|---|---|
| 矩阵 A / Ainv / C（复数矩阵及逆矩阵） | COMPLEX64（`aclFloatComplex`，两个连续 FLOAT32：real、imag） | 布局与 `cuComplex` 一致 |
| `ipiv` / `PivotArray` / `devInfo` / `info` / `infoArray` | INT32 | 1-based 主元 / 状态码 |
| `lda/ldc/lda_inv/n/m/batchSize/Lwork` | int（32-bit） | 维数与对标 CUDA legacy API 一致 |

`aclFloatComplex` 在 `cann_ops_solver_common.h` 定义为与 `cuComplex` 布局一致的 C 可调用类型，
C++ 侧可 `reinterpret_cast` 为 `std::complex<float>`。

### 支持形状（规模）

| 接口 | 形状约束（验收下限，实现可支持更大范围并在文档声明） |
|---|---|
| `aclsolverCgetrf` | $m,n \in [1, 4096]$，须覆盖方阵与矩形（$m \ne n$）；`lda ≥ max(1,m)` 允许 padding |
| `aclsolverCgetri` | $n \in [1, 4096]$；`lda ≥ max(1,n)` 允许 padding |
| `aclsolverCgetriBatched` | $n \in [1, 4096]$，`batchSize ∈ [1, 1000000]`；覆盖 `lda/ldc` padding |
| `aclsolverCmatinvBatched` | $n \in [1, 32]$，`batchSize ∈ [1, 1000000]`；**n = 33 必须报错** |

不支持 broadcast，不要求图融合；dynamic shape 由 Host 按本次调用维数生成 tiling。

## 算子实现

### host 侧设计

#### 1. 参数校验（对齐 cuSolver 参数序号约定：不计 handle，`m/n` 为参数 1）

三个 calculate 接口的校验顺序与 `devInfo` 写入值一致：

| 接口 | 校验项 → `devInfo` 值 | 返回值 |
|---|---|---|
| `aclsolverCgetrf` | `m<0 → -1`；`n<0 → -2`；`A==nullptr && m,n>0 → -3`；`lda<max(1,m) → -4`；`devInfo==nullptr` | `ACLSOLVER_STATUS_INVALID_VALUE` |
| `aclsolverCgetri` | `n<0 → -1`；`A==nullptr && n>0 → -2`；`lda<max(1,n) → -3`；`devInfo==nullptr`；`Lwork<need → -6`；`Workspace==nullptr && need>0 → -5` | `ACLSOLVER_STATUS_INVALID_VALUE` |
| `aclsolverCgetriBatched` | `n<0`(arg1)；`lda<max(1,n)`(arg3)；`ldc<max(1,n)`(arg6)；`batchSize<0`(arg8)；`infoArray==nullptr && batchSize>0`；`Aarray/Carray==nullptr` | `ACLSOLVER_STATUS_INVALID_VALUE` |
| `aclsolverCmatinvBatched` | `n<0`(arg1)；`n>32`(arg1)；`lda<max(1,n)`(arg3)；`lda_inv<max(1,n)`(arg5)；`batchSize<0`(arg7)；`info==nullptr`；`A/Ainv==nullptr` | `ACLSOLVER_STATUS_INVALID_VALUE` |

- `handle == nullptr` 一律返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。
- 空问题（`m==0 || n==0`、`n==0`、`batchSize==0`）视为成功：写 `devInfo/info = 0`（若可写），
  不下发 kernel。
- `_bufferSize` 接口用 `SOLVER_ECHECK` 做 Host 侧校验（失败返回状态码，不写 Device）；
  `Lwork` 指针为空返回 `ACLSOLVER_STATUS_INVALID_VALUE`。
- 参数错误路径的 `devInfo` 通过 `aclrtMemcpy(H2D, 4 字节)` 写回 Device（`WriteDevInfo`），
  保证 Device 侧状态可见。

#### 2. workspace 设计

| 接口 | `Lwork`（单位：`aclFloatComplex` 元素数） | 说明 |
|---|---|---|
| `aclsolverCgetrf_bufferSize` | `0` | 950 原生 LU 原地完成，无需 Device workspace |
| `aclsolverCgetri_bufferSize` | `n == 0 ? 0 : n * n` | 求逆需要一份 $n \times n$ 复数暂存 X（`ld = n`） |
| `aclsolverCgetriBatched` / `aclsolverCmatinvBatched` | 无 workspace 形参 | 对齐 cublas：`Carray[i]` / `Ainv[i]` 自身兼作 scratch，`Aarray[i]`/`A[i]` 永不被覆盖 |

workspace 完全由调用方在 Device 侧分配并传入，算子内部不申请 Device 内存（无隐蔽分配）；
`aclsolverCgetri` 在 `Lwork < Lwork_need` 时返回 `INVALID_VALUE` 并写 `devInfo = -6`。

#### 3. tiling 与分核策略

- `aclsolverCgetrf`、`aclsolverCgetri`：单矩阵问题，当前实现走 **单核** 右视/回代链
  （launcher 传入 `numBlocks = 1`，kernel 内 `GetBlockIdx() != 0` 直接 return）；
  多核 trailing update 作为后续优化项，不影响本次接口语义。
- `aclsolverCgetriBatched`、`aclsolverCmatinvBatched`：**按独立 batch 条带化到全部 AIV 核**，
  `blocks = min(GetCoreNumAiv(), 256, batchSize)`；kernel 内
  `for (b = GetBlockIdx(); b < batchSize; b += GetBlockNum())` 处理，各 batch 相互独立，
  无需跨核同步。950PR AIV 核数为 56（`platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv()`）。
- Host 侧提供环境变量 `SOLVER_FORCE_BLOCKS`（覆盖启动核数，实验用）、`SOLVER_PRINT_BLOCKS`
  （打印 `aiv/blocks/batch/n` 诊断信息），仅用于验证多核并行是否生效，不改变接口语义。

#### 4. kernel 下发

Host 通过内核调用符 `<<<numBlocks, nullptr, stream>>>` 下发，计算走调用方绑定到 handle 的 stream
（`aclsolverGetStream`），不做无必要的 Host 同步。Host 向 kernel 传递的实参如下：

| 接口 | 传入 kernel 的标量 | 传入 kernel 的 Device 指针 |
|---|---|---|
| `aclsolverCgetrf` | `m, n, lda` | `A, devIpiv, devInfo` |
| `aclsolverCgetri` | `n, lda` | `A, X(=Workspace), devIpiv, devInfo` |
| `aclsolverCgetriBatched` | `n, lda, ldc, batchSize` | `Aarray, Carray, PivotArray, infoArray` |
| `aclsolverCmatinvBatched` | `n, lda, lda_inv, batchSize` | `A, Ainv, info` |

### kernel 侧设计

四个 kernel 均声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，全部用
`AscendC::GlobalTensor` 的 `GetValue/SetValue` 标量原语（在该平台验证可靠）。
列主序复数元 $(i,j)$ 落在 float 偏移 `2*(i + j*ld)`（实部）与 `+1`（虚部）。

| 接口 | kernel 入口 | 启动核数 | 并行粒度 |
|---|---|---|---|
| `aclsolverCgetrf` | `cgetrf_lu_kernel`（`cgetrf_lu_kernel_do`） | 1 | 单矩阵，单核右视 |
| `aclsolverCgetri` | `cgetri_inv_kernel`（`cgetri_inv_kernel_do`） | 1 | 单矩阵，单核回代+递推 |
| `aclsolverCgetriBatched` | `cgetri_batched_inv_kernel`（`cgetri_batched_inv_kernel_do`） | `min(AIV 核数, 256, batchSize)` | batch 条带 |
| `aclsolverCmatinvBatched` | `cmatinv_batched_inv_kernel`（`cmatinv_batched_inv_kernel_do`） | `min(AIV 核数, 256, batchSize)` | batch 条带；内部 `n≥4` 走向量路径 |

其中 `cmatinv_batched_inv_kernel` 内部分派：`n >= 4` → `CmatinvVectorOne`（成对复数平面向量化），
`n <= 3` → `CmatinvScalarOne`（标量回退 + 参考）。

#### 1. cgetrf：`cgetrf_lu_kernel`（单核右视 LU）

- 定位：自包含右视 LU，针对公开 API 承诺的列主序布局重写；旧的 MIX_AIC_1_2 blocked-LU 核心
  面向 A2/A3 arch22，在 950（arch35）上跨核角色划分不执行、主元表不被触碰，故弃用
  （`src/cgetrf/kernel/getf2.hpp` 未被本 kernel 引用）。
- 主循环 `k = 0 .. t-1`（`t = min(m,n)`）：
  1. 部分选主元：在列 $k$ 的 $i \in [k,m)$ 内取 $\max |A[i,k]|^2$，**并列取最小行号**
     （确定性规则），写 `piv[k] = p+1`（1-based）；`devIpiv == nullptr` 时完全不交换
     （分解退化为 $A = L \cdot U$）。
  2. `p != k` 时对全部 $n$ 列交换第 $k$、$p$ 行。
  3. 奇异检测：$U(k,k) == 0$（精确判零）时记录首个奇异列号 `firstSingular = k+1`。
  4. 乘子 $L(i,k) = A(i,k) / A(k,k)$，$i \in (k,m)$，用 Smith 形式复除（先按较大分量缩放，
     防溢出）：设 $b=b_r+i b_i$，$s=\max(|b_r|,|b_i|)$，$\hat b_r=b_r/s$，$\hat b_i=b_i/s$，
     则 $a/b = \big[(a_r\hat b_r + a_i\hat b_i) + i\,(a_i\hat b_r - a_r\hat b_i)\big] /
     \big[(\hat b_r^2 + \hat b_i^2)\, s\big]$。
  5. 尾更新 $A(i,j) \mathrel{-}= L(i,k) \cdot U(k,j)$，$i \in (k,m)$、$j \in (k,n)$。
- 结果：A 原地持有打包 L/U，`info` 写回 `firstSingular`（0 表成功）。
- 数据搬入/搬出：本 kernel 直接以标量原语读写 GM，无 UB staging；正确性优先，
  尾更新向量化/多核化列入后续优化。

#### 2. cgetri：`cgetri_inv_kernel`（单核由 LU 因子求逆）

语义严格对齐 LAPACK `cgetri`：**从 LU 因子 + 主元求逆，绝不重新分解**，分四阶段：

1. **求 $U^{-1}$ 写入 workspace X 的上三角**（回代）：对 $j=0..n-1$，先 $\mathrm{inv}(U_{jj})$
   （Smith 复倒数），再解 $(X U)[i,j]=0$ 得
   $X[i,j] = -\mathrm{inv}(U_{jj}) \sum_{k=i}^{j-1} X[i,k] U[k,j]$，$i=j-1..0$；
   列 $j$ 对角线以下清零。
2. **$X \mathrel{:}= X \cdot L^{-1}$**（前向列递推，列从右到左）：$L$ 从 A 的严格下三角读取
   （阶段 1 未触碰），故单份 $n \times n$ X 即可完成。
3. **逆序列交换**（CLASWP 方向 -1，$k = n-1..0$）：$Z = Y \cdot P^{T}$，与 LAPACK
   `CSWAP(N, A(1,IPIV(I)), 1, A(1,I), 1)`（$I=N..1$）一致。
4. **X 原地写回 A**（in-place 契约）。

- `firstSingular`：首个使 $U(j,j) == 0$ 的 $j+1$；命中即停止计算，`info` 写回
  `firstSingular`（0 成功）。
- `ipiv == nullptr` 表示因子未选主元，跳过阶段 3。
- 计算落在 kernel 列暂存区设计上限内；A 与 X 均以列主序 `ld` 计偏移，支持 `lda` padding。

#### 3. cgetri_batched：`cgetri_batched_inv_kernel`（batch 条带化）

- **Device 指针数组语义**：`AarrayGm`/`CarrayGm` 是 Device 上长度为 `batchSize` 的 `uint64`
  指针数组（不是连续 `[batch,n,n]` 张量）；`PivotArray` 为线性 `n*batchSize` 表，batch $b$
  拥有 `[b*n,(b+1)*n)`。
- **out-of-place**：`Carray[i]` 同时是输出与 $n \times n$ scratch（接口无 workspace 形参，
  对齐 `cublasCgetriBatched`），`Aarray[i]` 永不被写。
- 每个 batch 复用 S2 的 `InvertOne`（$C[i] = \mathrm{inv}(U) \cdot \mathrm{inv}(L) \cdot P^{T}$：
  回代求 $U^{-1}$ → 前向列递推 $C \cdot L^{-1}$ → 逆序列交换），
  `infoArray[i] = firstSingular`（0 成功）。
- 分核：`for (b = blockId; b < batchSize; b += blockNum)`，batch 间独立、无跨核同步。

#### 4. cmatinv_batched：`cmatinv_batched_inv_kernel`（直接批量求逆，n ≤ 32）

**算法**：对每个 batch $i$ 直接求 $\mathrm{inv}(A[i])$（不走 getrf/getri 链），$n \le 32$。
采用**单缓冲原地 Gauss-Jordan + 部分选主元**，运行在 `Ainv[i]` 上（`A[i]` 只读、永不改写）；
最后用「逆序的主元列交换」补偿行交换；`info[i]` = 首个奇异列号（1-based），0 表成功。

**快速路径**：`n == 1` 直接取复倒数；`n == 2` 走标量闭式（$\det = ad-bc$，先按 $|a|<|c|$
做部分选主元，$1/\det$ 用 Smith 复倒数，再按同规则做列交换补偿）；`n >= 4` 走向量列形态
Gauss-Jordan（`CmatinvVectorOne`）；`n == 3` 走标量回退（`CmatinvScalarOne`，
用于非对齐载荷 + 作为向量路径的参考实现）。

**分核**：`for (b = blockId; b < batchSize; b += blockNum)` 把独立 batch 条带化到全部 AIV 核；
950PR AIV 核数 = 56，Host 用 `PlatformAscendC::GetCoreNumAiv()` 取。

**UB 布局 —— 成对复数平面**：列 $c$ 占据 `[c*WP, (c+1)*WP)`，`WP = 2*Hp`，
`Hp = roundup(2n, 8)`：
- 前 `Hp` 个 float 存实部行 `M[c*WP + r]`，
- 后 `Hp` 个 float 存虚部行 `M[c*WP + Hp + r]`。

之所以用 `Hp`（8 的倍数）而非 `2n` 作 pitch：任何 `n % 4 != 0` 时列基址 `4*(col*2n)`
不是 32B 对齐，会触发设备侧异常、运行时恢复成本约 320ms/launch（已在 652 用例基线上实测：
`n%4!=0` 全族平底 320ms）。行 `[2n, Hp)` 是零填充的 pitch 尾巴，只被整块向量指令读到，无害。

**初始化（v4）**：
1. 先用一条 `Duplicate` 预清零 64-float 列距的 mix 条带（`mix`，64-float pitch）；
   **注意：这条 `Duplicate` 是 PIPE_V，必须与随后 MTE2 的 `DataCopyPad` 入站之间显式加
   `PipeBarrier<PIPE_ALL>()`；否则慢落地的清零会把整列 A 抹成 0 → 假奇异 `info=1`
   （这是一个真实修过的低频竞态）**。
2. 逐列单块 `DataCopyPad`：`2n` 个 float 载荷 + 只补到 32B 边界的 rightPadding $R \le 7$
   （`R = roundup(2n,8) - 2n`，禁止大 pad 多块形态，会触发 507035）。
3. 逐列对整块成对平面 `Duplicate(M[col*WP], 0.0f, WP)`（两半一次清零，基址 32B 对齐、
   长度 8-float 倍数）。
4. 逐列两次 `GatherMask`（mask = 1 / 2）把交错数据解成 re/im 两半。
5. 逐列 `O(n)` 标量写单位阵对角线（**对角线 1.0 落在实部半区**
   `M[col*WP + n + col] = 1.0f`，虚部对角保持 0）。

**选主元**：标量扫描列 $k$ 的 A 块行 $k..n-1$ 取 $|\cdot|^2$ 最大（确定性取首个最大），
记 `pivRow[k]` 并在 `p != k` 时交换两半的第 $k$/$p$ 行。曾尝试向量化选主元
（掩码 `ReduceMax` + bit-cast 索引），但索引语义在 `forceSwap` 形态下会指错行，
且每 $k$ 的 PIPE_V 屏障吃掉了收益（11.70ms vs 11.45ms 标量），已回退并留注释。

**归一化列 $k$**：取 `invPivot = 1/M[k,k]`（Smith 复倒数），5 条向量指令完成
`col_k *= invPivot`（`im = ki*invRe + kr*invIm`、`re = kr*invRe - ki*invIm`）。

**列消除（本设计的核心优化）**：对 $c \neq k$，

```text
Axpy(M_c, M_k,   -fRe, WP)    // re_c -= re_k*fRe ; im_c -= im_k*fRe
Axpy(M_c, xneg,  +fIm, WP)    // re_c += im_k*fIm ; im_c -= re_k*fIm
```

两条 **WP 宽**指令即可完成复数乘减（其中 `xneg = [im_k; -re_k]` 每 $k$ 用两条 `Muls` 构建）。
等价的标量 4-Axpy 形态（每条宽 `Hp`）被替换掉。依据：950 向量指令成本
$\approx 22.8\text{ns}$ 固定 $+ 0.0116\text{ns/float}$（实测 8/64/256/512/1024 五点拟合），
瓶颈是**指令条数**而非数据量，故合并宽度收益大（主导段指令数约 1.9× 减少）。

**f 值读取**：每个 $k$ 用一次 `Gather`（`srcBaseOffset = 4*k`，**字节单位**）把「行 $k$」
抽成交错对 `[fr0,fi0,...]`（偏移表在 kernel 启动时按 $n$ 建一次），再用一次 `uint64` 标量读
（`GetValue` 对 float 缓冲做 2-float 别名，bit 精确）拿到 $(re,im)$ 对，
把 $2n^2$ 次标量读减半。

**回写**：逆块（行 $n..2n-1$）用预建 byte 偏移表做逐列 `Gather` zip 成交错列，
再逐列 `DataCopyPad` 写到 GM（`ldc` 列距）。预建偏移表（回写 zip 偏移 + f 行偏移）与矩阵无关，
kernel 启动时按 $n$ 建一次，全 batch 共享。

**UB 预算**：`kVBuf = 2*64*32`（成对平面，$n=32$ 时 $n \times WP$ 最大 $32 \times 128$ float）
+ wtmp(64) + mix(2048，64-float pitch 入站) + 回写 staging(2048) + xneg(128，`[im_k;-re_k]`
成对消除 scratch)，另加偏移表 `offBuf`。

## 支持硬件（required）

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR（Atlas 950，arch35） | √ |
| Ascend 950DT | √（同代际，950PR 为本次验收平台） |
| Atlas A2/A3 训练/推理系列 | ×（本任务适配 950，旧 blocked-LU 核心不适用 arch35） |

## 算子约束限制（required）

1. **阶数上限（`aclsolverCmatinvBatched`）**：仅支持 `0 <= n <= 32`；`n > 32` 直接返回
   `ACLSOLVER_STATUS_INVALID_VALUE`，**绝不转调 `aclsolverCgetriBatched`**
   （必须删除原型的 n≥32 转调行为）。
2. **Device 指针语义**：矩阵、主元、info、workspace、批量接口的 `Aarray/Carray/A/Ainv`
   均为 **Device 指针**（批量接口为 Device 上的指针数组，`uint64` 条目，非连续张量）；
   Host 侧仅传标量维数、枚举与 handle。`ipiv/PivotArray == nullptr` 表示不选主元。
3. **workspace 由用户切分**：`getrf` 的 `Lwork = 0`，`getri` 的 `Lwork = n*n`（`aclFloatComplex`
   元素数）；workspace 由调用方按 `_bufferSize` 返回值在 Device 侧分配后传入，
   算子内部禁止任何隐蔽 Device 分配。`getri` 要求 `Lwork >= 查询值`，否则 `devInfo = -6`。
4. **info 语义**：`0` 成功；`-i` 第 i 个参数非法（不计 handle，与对标 API 参数序号一致）；
   `i > 0` 表示 `U(i,i) = 0`（奇异，值为首个奇异列号，1-based）。批量接口逐 batch 独立写
   `info[i]`/`infoArray[i]`。空问题（`m/n/batchSize == 0`）成功返回 `info = 0`。
5. **奇异检测**：`getrf`/`getri` 精确判零 LU 因子对角线；`CmatinvBatched` 靠扫描 LU 因子
   对角线精确判零，**不依赖哨兵字段**；`info` 不做「未定义即 0」假设。
6. **lda/ldc/lda_inv 约束**：`lda ≥ max(1,m)`、`ldc ≥ max(1,n)`、`lda_inv ≥ max(1,n)`，
   允许 padding，禁止要求等于 `n`。
7. **cgetri/cgetriBatched 输入为已分解 LU 因子 + 主元**，不是原始矩阵；端到端批量求逆须先
   `getrf` 再 `getriBatched`；`CmatinvBatched` 才直接对原始矩阵求逆。
8. **批量接口 out-of-place**：`Carray[i]` 不得与 `Aarray[i]` 重叠、`Ainv[i]` 不得与 `A[i]` 重叠；
   重叠/非法主元应报 `INVALID_VALUE`。
9. **确定性**：部分主元选取为列内模最大且行号最小；无原子竞争；同输入重复执行 bit-wise 一致。
10. **无 CPU fallback**：核心计算在 AI Core（AIV）完成。

# 可维可测分析（required）

## 精度标准 / 性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
|---|---|---|
| 精度标准（`Cgetrf`） | LAPACK 残差判据 `ratio = ‖L·U − A‖₁ / (n·‖A‖₁·ε) ≤ max(5·ratio_cpu, 0.5)`；由打包 LU 重放 L·U 并按 ipiv 逆序（DLASWP 方向 -1）还原行序后计算 1-范数残差 | LAPACK 测试例程 CGET01 |
| 精度标准（`Cgetri`/`CgetriBatched`/`CmatinvBatched`） | 逐矩阵 `ratio_i = ‖I − C_i·A_i‖₁ / (n·‖A_i‖₁·‖C_i‖₁·ε) ≤ max(5·ratio_cpu_i, 0.05)`（`CmatinvBatched` 阈值取 `0.5`）；任一矩阵超阈值该用例不通过 | LAPACK 测试例程 CGET03 |
| 单位舍入 | `ε = 2^-24 ≈ 5.96e-8`（COMPLEX64） | — |
| 性能标准 | 对每个 case，NPU kernel 耗时 `T_NPU ≤ T_GPU数据 / 0.35`（性能倍率 ≥ 0.35×GPU 数据）；核耗时按 msprof `OpBasicInfo.csv` 的 `Task Duration(us)` 统计 | 任务书 3.3 |

性能对标 case（GPU 口径：CUDA 同 shape/dtype/列主序，含一次 bufferSize 不计入 kernel 耗时）：
`Cgetrf`（m=n=1024 / m=n=4096 / m=2048,n=1024）、`Cgetri`（n=1024 / n=4096）、
`CgetriBatched`（n=32,batch=1024 / n=128,batch=128）、
`CmatinvBatched`（n=16,batch=1024 / n=32,batch=256）。

本设计仅在文档层面声明判据；具体测试结果与性能数字由自测报告（交付件 3）给出，本文档不重复。
设计侧的性能保障要点：向量指令按「指令条数」而非数据量优化（成对复数平面 + WP 宽
Axpy 合并消除、Gather f 行 + uint64 别名读减半标量读、整块 `Duplicate`/`GatherMask`、
预建偏移表全 batch 共享），避免非对齐触发的 320ms/launch 恢复开销。

### 性能边界分析（950 AIV 实测，供评审参考）

`CmatinvBatched`（n ≤ 32）的判据预算极小（n=4 每矩阵 1.58µs；大 batch 下每矩阵 286ns 量级），
而 950 AIV 是**每核固定 issue rate** 的向量单元。为判断「继续加宽/并行能否达标」，本设计用
独立微基准（`ub_micro_kernel.cpp` 的 mode37～mode47）逐条做了正面与反面的实测取证：

| 探针 | 命题 | 实测结论 |
|---|---|---|
| mode37/38/41 | 向量指令成本模型 | `T ≈ 22.8ns 固定 + 0.0116ns/float`（宽 8→1024 float：23.1→34.6ns）⇒ 瓶颈是**指令条数**，不是数据量 |
| mode45/46 | 指令级并行（K 条独立 Axpy 链交织） | K=1/2/4/8 的 per-op 恒为 23.6~25.3ns，**加速比 1.00×** ⇒ 无依赖停顿可回收，纯 issue-rate 定界 |
| mode47 | 跨引擎重叠（MTE2 搬运 vs AIV 计算） | `copy→PIPE_ALL→Axpy`=107.2ns，`copy→Axpy`=**34.0ns**（=仅 Axpy）⇒ MTE2 延迟**可被完全隐藏**；同时量得一道 `PipeBarrier<PIPE_ALL>` ≈ **43ns** |
| mode39/40/42/43/44 | 廉价「加宽原语」（UB→UB 多块 `DataCopy`、多 repeat `Gather`、`Brcb`、`Copy{srcStride=0}` 广播） | **全部不可用**：四种原语在本机要么只写首块/首元素，要么显式参数语义非「逐元素展开」 |

```text
n=4 每矩阵预算 1.58µs 的定量下限（可证明的不可达）
  最优形态下 V 指令 ≈ 57 条 × 24.5ns        = 1.40µs
  标量下限      ≥ 38 op  × 20ns（微基准理想）= 0.76µs
  ------------------------------------------------------
  合计 ≥ 2.16µs > 1.58µs（1.37× 超预算，且此处假设零屏障、零停顿）
```

真 kernel 成本归因（n=4，实测每矩阵 4.8µs）：**标量 ≈2.7µs（56%）+ 向量 ≈1.9µs（40%）+
屏障/GM ≈0.4µs**。标量项远高于孤立微基准 20ns/op，差值是标量/向量管道的**互锁税**
（真实 kernel 内标量读写 ≈55~60ns/op）。

**结构性命中面**：n ≤ 32 的运算量极小 ⇒ 耗时由「指令条数 × ~24ns/条」定界；而把指令加宽所需的
四种廉价原语已逐一实测失效，ILP 零收益，Cube(AIC) 路径在本仓由 `#ifdef __DAV_C220_*` 门控
（C220=910B 世代，`--soc=ascend950` 下不编译）且公开接口无 workspace 形参（与「禁止内部隐蔽
内存分配」冲突）。本机最好的 950 复数 GEMM 参考也**主动废弃**了自研 Cube kernel 改委托 upstream，
记录 Cube 层瓶颈为「复数分解需 3 次独立 launch + HBM 流量放大」。

**仍可做但收益量级不足的优化**：`PipeBarrier<PIPE_ALL>` 次数（n=4 每矩阵 8 道 × ≈43ns）
与暴露的 GM 往返可压缩，mode47 已证 MTE2 可完全隐藏，预估合计 **1.3~2×**；该量级足以让
`n=25..32, batch=128` 这类 near-miss 组（ratio 1.02~1.08）过线，但无法弥合大 batch 的 6~18×。

**后续优化路线（本批未实施，按收益量级排序）**：

1. **`Cgetrf` / `Cgetri` 的分块 LU + 多核 + UB 向量化**（收益量级最大）。这两个 kernel 当前是
   **标量单核**实现（`GlobalTensor::GetValue/SetValue`，无 UB staging、无向量指令、只用 1 个 AIV 核），
   因此 1024³ 及以上 case 的单次耗时落在秒级、并被判据预算（毫秒级）远远甩开。改为
   分块右视 LU（panel 分解 + `TRSM` + `GEMM` 尾更新）并把尾更新向量化、按块条带化到全部 56 核后，
   FLOP/耗时比可回收 2~3 个数量级 —— 这是四接口中唯一存在**量级级**回收空间的地方。
2. `CmatinvBatched` 的屏障合并 + 软件预取（上一条，1.3~2×）。
3. **不推荐** Cube(AIC)：本仓复数 Cube 管线由 `#ifdef __DAV_C220_*` 门控（C220 世代），
   `--soc=ascend950` 下不编译；且公开接口无 workspace 形参，与「禁止内部隐蔽内存分配」冲突。

## 兼容性分析

1. **接口兼容**：4 个公开接口的参数名、顺序、含义、Device/Host 归属、维数类型（`int`，
   与对标 CUDA legacy API 一致）均对齐 cuSolver/cuBLAS；差异仅在命名前缀
   （`aclsolver` vs `cusolverDn`/`cublas`）、`aclsolverOperation_t`（枚举 0/1/2 对齐
   `cublasOperation_t`）、`aclFloatComplex`（布局对齐 `cuComplex`）。
2. **状态码兼容**：计算接口统一返回 `aclsolverStatus_t`，语义对齐 cuSolver。
   若需短暂兼容现存 `aclError` 原型，须在公开头文件与文档给出对照表，最终验收以
   `aclsolverStatus_t` 为准。
3. **语义兼容**：U 在 A 的打包布局、1-based `ipiv`、`ipiv == nullptr` 不选主元、
   `CmatinvBatched` n>32 报错等语义均与对标接口一致；`Cgetri`/`CgetriBatched` 要求输入
   已分解 LU 因子，与 cuSolver/cuBLAS 一致（cuSolver DN 无独立 getri，以
   `cusolverDnCgetrf` + 求解 `AX=I` 作为 GPU 金标）。
4. **硬件代际**：旧 `custom_lu`（MIX_AIC_1_2 blocked-LU）面向 A2/A3 arch22，其跨核角色
   划分在 arch35 上不执行，与本仓 950 实现并存但不作为本接口路径。
5. **文档既有说明差距**：`docs/zh/cgetrf.md`、`cgetri.md`、`cgetri_batched.md`、
   `cmatinv_batched.md` 中的「`lda` 必须等于 `n`」「info 不写回」等描述属于**旧原型语义**，
   与本公开接口设计（允许 padding、真实写回 info）不一致，需随接口升级同步更新；
   `cmatinv_batched` 文档中「n≥32 自动转调 CgetriBatched」亦须删除。
