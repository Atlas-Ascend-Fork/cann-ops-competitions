# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · `10月社区任务-单精度实数QR分解与最小二乘(950)`（任务书随任务包下发，名称见 `Atlas950_Sgeqrf_Sorgqr_Sormqr_Xlarft_SSgels_task_doc.md`）。本目录为 `kevin_lee1231` 团队的提交目录。

## 背景介绍

### 算子实现现状

本任务在 **Ascend 950PR（Atlas 950）** 上以 **Ascend C / catlass** 从零实现单精度实数稠密 QR 族，共 5 个接口**一并交付、不允许拆分验收**，对标 cuSOLVER DN 的 `Sgeqrf / Sorgqr / Sormqr / Xlarft / SSgels` 参考耗时（任务书 §3.3 性能门槛 P-01~P-12）：

| 接口 | 功能 | 输出位置 |
|---|---|---|
| `aclsolverSgeqrf` | Householder QR 分解 `A = Q·R` | 原地覆盖 A（下三角存反射器向量），另出 `TAU` |
| `aclsolverSorgqr` | 由 `(v, tau)` 生成显式 Q 的前 k 列 | 原地覆盖 A |
| `aclsolverSormqr` | `C := op(Q)·C` / `C := C·op(Q)` | 原地覆盖 C |
| `aclsolverXlarft` | 块反射器三角因子 `H = I − V·T·Vᵀ` | 输出 T（k×k 三角） |
| `aclsolverSSgels` | 最小二乘 `min ‖A·X − B‖_F` | **out-of-place** 写 dX |

`cann/ops-solver` 仓库当前仅有 `cgetrf / cgetri / cgetri_batched / cmatinv_batched / sgetrf / sgetri / cheevj` 七个算子，**QR 族与 gels 无任何原型**，五个接口须全新实现。

### 与仓内已有算子的关系

本任务不孤立：`Sorgqr`、`Sormqr`、`SSgels` 三者共用 `Xlarft` 产出的三角因子 T 来做块反射器（`H = I − V·T·Vᵀ`），`SSgels` 又以 `Sgeqrf` 为前导。因此 `Xlarft` 是整个族的公共基石，接口虽独立交付，**实现上必须按"T 因子唯一生产者"来组织**，否则三处各写一份 block reflector 会带来数值分歧。

仓内已有资产可直接复用（不修改公共仓）：

| 复用件 | 路径 | 用途 |
|---|---|---|
| c32 GEMM 封装 | `src/utils/kernel/c32/gemm.hpp` | 面板尾随更新、块反射器乘法 |
| c32 TRSM 封装 | `src/utils/kernel/c32/trsm.hpp`、`trsm_custom.hpp`、`trsm_upper_custom.hpp` | `SSgels` 的上三角回代、`Xlarft` 的 `T11⁻¹` |
| 单位阵生成 | `src/utils/eye_matrix.h`（`GenerateEyeMatrix`） | `Sorgqr` 初始化 `[0:k)×[0:n)` 单位阵块 |
| 对齐/补齐 | `src/utils/kernel/c32/pad.hpp` | 非 32B 对齐的 `lda/ldv/ldt/ldc` 处理 |
| 参数校验与 info 写入 | `src/utils/aclsolver_auxiliary.cpp` | 五个接口统一的入参校验与 `devInfo/dinfo` 写入 |
| 列主序带 stride 的 Cube 写出 | `src/cheevj/kernel/`（只覆盖 `ToUnderlyingArguments` 的内核包装 + TileCopy 包装） | 把 Cube 结果落到调用方列主序缓冲（C stride = `(ldc, 1)`） |

### 决定架构选型的关键约束

| 参数 | 含义 | 约束 |
|---|---|---|
| 布局 | **列主序** | `lda ≥ max(1,m)` 等；**禁止**要求 `lda == m` |
| `TAU` | 反射器系数 | 长度 `min(m,n)` |
| `k`（orgqr/ormqr/larft） | 反射器个数 | `0 ≤ k ≤ n`（ormqr：`k ≤ nq`，`nq = side==LEFT ? m : n`） |
| `k`（larft） | 块阶 | `k ∈ [1, min(n,256)]` |
| `dinfo` | 第 i 个参数非法 | `0 / −i`，**不含**奇异检测（QR 不做奇异判定） |
| 规模 | — | `m, n ∈ [1,4096]`；`SSgels` 须 `m ≥ n` |

1. **列主序对 NPU 是顺风**：列主序下"一列"就是一段连续地址，`DataCopy` 按列搬运天然是连续大块，对齐只受 `lda` 影响；反之若内部转行主序，每行都跨 stride，得不偿失。因此**全程保持列主序，不做转置**。
2. **Cube 的 GEMM 需要落到调用方的列主序缓冲**：catlass 的 `BatchedMatmulTla` 把每矩阵步长写死为 `strideA/strideB/strideC`，而本任务 C 的步长是运行期的 `(ldc, 1)` ⇒ 沿用仓内 cheevj 已验证的"**只覆盖 `ToUnderlyingArguments` 的内核包装**"提供运行期步长覆盖，不修改公共仓。
3. **面板内列间串行**：第 j 列的 `τ_j` 依赖前 j−1 列对它的更新，故面板无法列并行；能并行的是尾随块的 Cube GEMM。这决定了"面板核（AIV，串行）+ 尾随 GEMM（AIC，并行）"的两段结构。
4. **`Xlarft` 的有效规模是 k×k 而非 n×k**：`V(i,j)` 仅在 `i ≥ j` 非零，`T = triu/tril((VᵀV)⁻¹)` 只依赖 `V` 的前 k 行 ⇒ GEMM 取 `k×k×k`，不是 `n×k×k`。这把 `Xlarft` 的算量从 O(nk²) 降到 O(k³)，是本接口能跑在 4096 阶规模下的关键。

# 需求分析（required）

## 需求描述

以 Ascend C 实现上述 5 个接口：`float32`、列主序、合法 `lda/ldc/ldv/ldt` padding、`m,n ∈ [1,4096]`；精度满足任务书 §3.2.1 的 LAPACK 残差判据（SQRT01/02/03/16/17 口径）；性能满足 P-01~P-12 的 `T_NPU ≤ T_GPU / 0.35`；同输入重复执行输出 bit-wise 一致。

## 需求拆解

1. **公共基座**：统一入参校验（`m/n/k/nrhs/lda/ldc/ldv/ldt/side/trans/direct/storev/dataType`）、`devInfo/dinfo` 写入、空问题（`m/n/k/nrhs = 0`）成功返回、`aclsolverStatus_t` 返回值与仓内现存 `aclError` 原型的对照表；
2. **`Sgeqrf`**：面板 Householder 分解（AIV）+ 尾随块 GEMM 更新（AIC）两段流水；
3. **`Xlarft`**：块三角因子 T 的递推求解（`T22 ← T22 − T21ᵀ·T11⁻¹·T21`），FORWARD/BACKWARD 两向；
4. **`Sorgqr`**：单位阵块初始化 + 逆序块反射器作用（复用 T）+ 面板内逐反射器作用；
5. **`Sormqr`**：side×trans 四种组合的统一实现，其中 `trans=C` 走独立枚举分支（实数下与 `T` 等价但接口须分别接受）；
6. **`SSgels`**：`Sgeqrf` 前导 + 反射器作用于 B 的前 n 行 + 上三角回代，out-of-place 写 dX；
7. **精度**：五个接口分别按 SQRT01/02/03/16/17 残差判据通过；`Sormqr` 须在 `Sorgqr` 通过后再验；
8. **性能**：P-01~P-12 全部达标，其中 P-02/P-05（4096 阶）与 P-12（4096 方阵 gels）为算力主战场；
9. **确定性**：Householder 符号约定固定（LAPACK `DGEQR2` 的 `beta = −sign(alpha1)·‖x‖₂`），同输入重复执行 bit-wise 一致。

# 详细设计（required）

## 算子分析

### 数学公式

**1. Householder 反射器（`Sgeqrf` 的原子操作）**

对第 j 列尾部向量 `x = A[j:m, j]`（长度 `p = m − j`）：

```text
β  = −sign(x₀)·‖x‖₂                      # LAPACK 约定，保证 R 对角元符号确定
v  = x / ‖v‖₂ ,  v₀ = 1                    # 归一化后 v₀ 强制为 1
τ  = 2 − x₀/β = 1 + |x₀|/‖x‖₂
A[j,   j] = β          # R 的对角
A[j+1:m, j] = v[1:p]    # 反射器向量存于严格下三角
A[j:m, j+1:n] ← A[j:m, j+1:n] − τ·v·(vᵀ·A[j:m, j+1:n])
```

`‖x‖₂` 用三段缩放（ssq）算法计算，避免上溢/下溢；`τ` 倒数用 `Rsqrt` 反演。尾部全零时 `τ = 0`，该列不再更新。

**2. 分块 QR（`Sgeqrf` 主流程，面板宽 `nb`）**

```text
for j = 0, nb, 2nb, … :
    ① 面板  A[j:m, j:j+nb]  →  AIV 侧列式 Householder，产出 R、V、τ
    ② 尾随  A[j+nb:m, j+nb:n] ← A[j+nb:m, j:j+nb] · (τ ⊙ V[j, j:j+nb]ᵀ)      # Cube GEMM (M=m-j-nb, N=n-j-nb, K=nb)
             A[j+nb:m, j:j+nb] ← A[j+nb:m, j:j+nb] − V[j+nb:m, j:j+nb] · (τ ⊙ V[j, j:j+nb]ᵀ)   # Cube GEMM，原地 rank-1
```

**3. 块反射器三角因子（`Xlarft`）**

块反射器 `H = I − V·T·Vᵀ`，其中 `V` 为 n×k、只有前 k 行有效。`T` 满足 `T = triu((VᵀV)⁻¹)`（FORWARD）/ `T = tril((VᵀV)⁻¹)`（BACKWARD），用消元递推实现：

```text
FORWARD（从右下角起算）:
    T11 ← V[0:b,0:b]ᵀ·V[0:b,0:b] ,  T21 ← V[b:k,0:b]ᵀ·V[0:b,0:b] ,  T22 ← V[b:k,b:k]ᵀ·V[b:k,b:k]
    T22 ← T22 − T21ᵀ · T11⁻¹ · T21          # 递归处理右下子块
    T21 ← −T21 · T11⁻¹
    T11 ← T11⁻¹
BACKWARD: 从左上角起算的对称递推，结果取下三角
```

**4. 生成显式 Q（`Sorgqr`）**

`Q = H₀H₁…H_{k−1}`，需生成 m×n 的前 k 列：

```text
C[0:k, 0:n] = I ,  其余置零
for j = k−1 … 0 :  C[j:m, :] ← C[j:m, :] − τ_j·v_j·(v_jᵀ·C[j:m, :])
```

分块版把 `k` 个反射器按 `nb` 分组，尾部组用块反射器一次 GEMM 完成，面板内 `nb` 个逐个作用。

**5. `Sormqr` 四种组合**

| side | trans | 语义 | 反射器作用顺序 |
|---|---|---|---|
| LEFT | N | `C := Q·C` | 逆序 `H_{k−1}…H₀`（逐列） |
| LEFT | T / C | `C := Qᵀ·C` | 正序 `H₀…H_{k−1}`（逐列） |
| RIGHT | N | `C := C·Q` | 逆序（逐行） |
| RIGHT | T / C | `C := C·Qᵀ` | 正序（逐行） |

实数下 `trans=C` 与 `trans=T` 数学等价，但**必须走各自的枚举分支**：验收会分别传入 C，错误路径（如把 C 判为非法值）会导致直接判失败。

**6. 最小二乘（`SSgels`，`m ≥ n`）**

```text
A = Q·R = Q₁·R ,  Q = [Q₁ Q₂] , Q₁ 为 m×n
① 分解：A ← geqrf(A)，得 R（上三角 n×n）与反射器
② 投影：Y₁ = Q₁ᵀ·B = H_{n−1}…H₀·B[0:m, :]     # 只作用于 B 的前 n 行
③ 回代：R·X = Y₁                             # 上三角求解，n×n
④ 输出：dX = X（out-of-place，lddx ≥ max(1,n)）
```

第 ② 步对 `nrhs` 列整体做窄 GEMM：`B[0:n, :] −= V(0:n,0:n)·(τ ⊙ (Vᵀ·B[0:n,:]))`。

### 支持数据类型

`float32`（唯一）。`aclsolverXlarft` 的 `dataTypeV / dataTypeTau / dataTypeT / computeType` 四元组全为 `ACLSOLVER_R_32F`；其余组合返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`。

### 支持形状

| 接口 | 维数 | 必覆盖 |
|---|---|---|
| `Sgeqrf` | `m, n ∈ [1, 4096]` | 方阵与矩形，含 `m > n`、`m = n`、`m < n`（`m<n` 时 `τ` 长度为 `m`，R 为梯形）；`lda` padding |
| `Sorgqr` | `m ∈ [1,4096]`，`n, k ≤ m` | `k = n` 与 `k < n`；`lda` padding |
| `Sormqr` | `m, n ∈ [1,4096]` | `side ∈ {LEFT, RIGHT}`、`trans ∈ {N, T, C}`；`lda/ldc` padding |
| `Xlarft` | `n ∈ [1,4096]`，`k ∈ [1, min(n,256)]` | FORWARD / BACKWARD；仅 COLUMNWISE；ROWSWISE 与非 FLOAT32 返回 `NOT_SUPPORTED` |
| `SSgels` | `m ≥ n`，`m, n ∈ [1,4096]`，`nrhs ∈ [1,128]` | `m = n` 与 `m > n`；`m < n` 报错 |

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

**1. 分核策略**

- **优先满核**。尾随 GEMM 形状为 `(m−j−nb) × (n−j−nb) × nb`，由 catlass 按行/列块切到全部 AIC 核。
- **面板核单核**：面板内列间有数据依赖（`τ_j` 依赖前 j−1 列的更新），列并行需跨核同步，收益低于开销。设计上由**单个 AIV 核**承担面板，靠 Cube 拉平总负载。
- **小矩阵旁路**：`min(m, n) ≤ 64` 时不做分块，全部落到纯向量核（面板 + 直接更新），此时 Cube 无收益、调度开销反而占主导。该门控同时保证 P-10/P-11/P-12 等小规格不被分块路径的额外搬运拖累。
- **极端长条**（`m ≫ n`，如 `m=4096, n=136`）：面板核成为瓶颈，此时启用**多列并行的 SIMT 形态**（按 `n` 方向切给多核），参照仓内 `cgetri_batched` 的 SIMT 2048 线程先例。

**2. 路径门控**

分块路径（面板 + Cube GEMM）仅在 `min(m,n) > 64` 时启用；其余走向量路径。门控条件由 host 侧写入 TilingData 的 `tilingKey`，kernel 侧据此二分，避免在 device 上重复判断。

**3. TilingData 关键字段**

```text
m, n, k, lda, ldc, ldv, ldt          # 形状与 leading dimension
nb                                   # 面板宽，默认 64，可由环境变量覆盖
blockCount, blockSize                # 尾随 GEMM 的分块
useBlockedPath (tilingKey bit0)      # 路径门控
side, trans, direct, storev          # ormqr / larft 的枚举
tauOffset                            # larft 中 T 的有效子块偏移
```

面板宽 `nb` 默认取 64：面板代价 `O(m·n·nb)` 与尾随代价 `O(m·n²)` 之比约 `2nb/n`，`n=4096` 时不足 4%，且 `nb=64` 时面板 UB 占用可控。`nb ∈ {32, 64, 128}` 通过环境变量现场调优后固化。

**4. workspace / Lwork 计算**

`geqrf/orgqr/ormqr` 的 `Lwork` 以**元素个数**计，`Xlarft/SSgels` 以**字节**计（对齐 cuSolver）：

| 接口 | 设计取值 | 说明 |
|---|---|---|
| `Sgeqrf` | `Lwork = nb·n` | 面板核临时缓冲（打包 V / 缩放中间量） |
| `Sorgqr` | `Lwork = nb·max(m,n) + k` | 块反射器工作区 + 单位阵块 + τ 暂存 |
| `Sormqr` | `Lwork = nb·max(m,n) + k·k` | 同上，另加 T 的存储 |
| `Xlarft` | `bytes = k·k·4 + nb·k·4` | T 本体 + 递推中间块 |
| `SSgels` | `bytes = (nb·n + max(m,n)·nrhs + n·nrhs)·4` | 分解工作区 + B 的投影暂存 + 回代 |

取值为本设计的下限公式，host 侧同时保证 `≥ cuSolver 对应接口文档要求的最小值`；查询与计算接口使用同一函数生成，避免两处公式漂移。

**5. 参数校验与 info**

统一在 host 完成：负维数、`lda/ldc/ldv/ldt < max(1, 对应维数)`、`n > m`（orgqr）、`k > nq`（ormqr）、`k > n`（larft）、`m < n`（gels）、非法枚举、`dataType` 非 FLOAT32、`storev ≠ COLUMNWISE` → 返回 `ACLSOLVER_STATUS_INVALID_VALUE` 并在 `devInfo/dinfo` 写入 `−i`（不计 handle）。空问题（`m/n/k/nrhs = 0`）直接成功返回、不发射 kernel。`niter` 为 Host 标量，初值 0。

#### 3.2.2 kernel 侧设计

**1. `Sgeqrf`：两段流水**

同一 stream 顺序发射两个 kernel，天然形成栅栏，无需跨核 MIX 旗语：

- **① 面板核（AIV）**：对 `j ∈ [0, nb)` 逐列：`ssq` 求 ‖x‖₂ → 得 `β, τ` → 整列缩放写成归一化 `v`（`v₀=1`）→ 对本面板内后续列 `A[j:m, j+1:j+nb]` 做秩一更新 → 写回 `τ_j` 与对角 `β_j`。
- **② 尾随核（AIC，catlass GEMM）**：两次 GEMM（更新 `A[j+nb:m, j+nb:n]`、更新 `A[j+nb:m, j:j+nb]`）。

**2. Cube 结果落回调用方列主序缓冲（关键设计）**

沿用 cheevj 已验证的做法：新增只覆盖 `ToUnderlyingArguments` 的内核包装，把 C 的步长改为运行期传入的列主序 `MakeStride(ldc, Int<1>)`，并以 `SetAtomicAdd<float>()`（TileCopy 层完成 epilogue）把 GEMM 结果**直接原子累加**回 A22。收益是省掉"workspace 落盘 + 全量读改写"的一次 `m·n` 级往返。

> 实测教训固化进注释（来自 cheevj 的经验）："地址步进缺陷"这类结论要先排除批量核把 `strideC` 写死这一因素，不能直接归因于 fixpipe 写通路。

**3. `Xlarft`：递推 + trsm**

`k ≤ 256`，T 的工作集 `k·k·4 ≤ 256 KB`，可整体驻留。`T11⁻¹` 按规模二分：`k ≤ 64` 走分块显式求逆（Cube 上的小 trsm）；`k > 64` 走 catlass `Trsm` 逐块。`VᵀV` 只算下三角（对称），GEMM 取 `k×k×k` 而非 `n×k×k`。

**4. `Sorgqr`：单位阵块 + 逆序块反射器**

- 初始化：`GenerateEyeMatrix` 生成 `[0:k)×[0:n)` 单位阵块，其余清零（该函数按"单 float 平面 + 行宽 `strideN`"布局产出，与 trsm/restore 的消费口径一致）。
- 主体：从 `j = ⌊k/nb⌋·nb` 起**逆序**处理，每组 `nb` 个反射器先用 `Xlarft` 的 T 做一次 GEMM（`C[j:m, j+nb:n] −= V·(T·(Vᵀ·C))`），组内 `nb` 个反射器由面板核逐个作用。

**5. `Sormqr`：四种组合共用一套内核**

把 (side, trans) 映射到"作用对象（列 / 行）× 作用顺序（正序 / 逆序）× 块起点方向"，四条枚举分支在 kernel 侧汇入同一条"块 GEMM + 面板逐反射器"的流水，仅迭代方向与转置访问方式不同。`trans = C` 与 `trans = T` 共用实数转置路径但保留独立分支标识，确保枚举校验与错误信息正确。

**6. `SSgels`：三段串接**

`Sgeqrf` 分解 → 窄 GEMM 对 B 前 n 行做反射器投影 → `trsm_upper_custom.hpp` 上三角回代。三段在同一 stream 内顺序发射，段间天然栅栏；`dX` 为 out-of-place，回代结果直接写入调用方的列主序 dX（`lddx` padding 由 host 校验）。`m > n` 时 B 的 `[n:m, :]` 行不参与投影，原样保留。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（Atlas 950，dav-3510，arch35） | √ |
| Atlas A3 / A2 / 200I / 500 等 | × |

### 算子约束限制

1. 仅支持 `float32`；`Xlarft` 的 `dataTypeV/Tau/T/computeType` 须全为 `ACLSOLVER_R_32F`，其余组合 `NOT_SUPPORTED`；
2. 列主序，`lda ≥ max(1,m)`、`ldc ≥ max(1,m)`、`ldv ≥ max(1,n)`、`ldt ≥ max(1,k)`；**不要求**等于最小值（即必须支持合法 padding）；
3. `Xlarft` 仅支持 `storev = COLUMNWISE`，ROWSWISE 返回错误；`n ≥ k`；
4. `Sormqr` 须 `0 ≤ k ≤ nq`，`nq = side==LEFT ? m : n`；
5. `SSgels` 须 `m ≥ n`，`m < n` 报错；`niter` 初值 0（直接主精度分解求解）；
6. QR 不做奇异检测，`devInfo/dinfo` 只报参数非法（`0 / −i`）；
7. 空问题（`m/n/k/nrhs = 0`）成功返回，不发射 kernel；
8. 分块路径仅在 `min(m,n) > 64` 启用，其余走向量路径；`Sgeqrf` 在 `m < n` 时 `τ` 长度为 `m`、R 为梯形；
9. 确定性：Householder 符号固定为 LAPACK `beta = −sign(alpha1)·‖x‖₂`，同输入重复执行 bit-wise 一致。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 五个接口分别按 LAPACK 残差判据：`Sgeqrf` SQRT01（`ratio ≤ max(5·ratio_cpu, 0.1)`）、`Sorgqr` SQRT02（`≤ max(5·ratio_cpu, 1.5)`）、`Sormqr` SQRT03（`≤ max(5·ratio_cpu, 0.1)`）、`SSgels` 方阵 SQRT16（`≤ max(5·ratio_cpu, 0.015)`）与超定 SQRT17（`≤ 10·ratio_opt_cpu`）；ε = 2⁻²⁴ | 任务书 §3.2.1 |
| 性能标准 | `T_NPU`（平均单次 kernel 耗时，msprof 采集）`≤ T_GPU / 0.35`，P-01~P-12 全部达标 | 任务书 §3.3 |
| 内存标准 | 不做要求 | 任务书 §3.4 |

**精度验证的依赖顺序**：`Sormqr` 的 `op2` 参考量由 `Sorgqr` 的输出构造，故 **`Sorgqr` 须先通过验收，`Sormqr` 方可开始验收**。实现阶段把三者的回归绑在同一次 CI 序列里，避免 `Sormqr` 在 `Sorgqr` 未稳定时出现不可解释的失败。

**验证计划**

1. 随附 694 条用例（`sgeqrf` 174 / `sormqr` 189 / `sorgqr` 165 / `sgels` 166）逐条复跑；
2. 按任务书 §3.5「自构造用例入参生成规则」补齐随附用例未覆盖的场景：`lda/ldc/ldv/ldt` 的 `+8 / +32` padding；`m/n/k/nrhs = 0` 空问题；`Sormqr` 的 `trans = C`；`Sgeqrf` 的 `m < n`；`SSgels` 的 `nrhs ∈ {1, 8, 32, 128, 0}` 与 `m < n` 报错；`Xlarft` 的 FORWARD/BACKWARD、ROWSWISE 报错、非 FLOAT32 `NOT_SUPPORTED`；
3. 确定性：同一输入重复执行，输出（含 `τ`、`devInfo`）bit-wise 一致；
4. 非默认 stream 下结果正确；
5. 性能：msprof 逐条采集，与各算子目录 `bench_result.json` 的 `avg_ms` 关联比对。

**待社区确认事项**（已在 [ops-solver discussions #6](https://gitcode.com/cann/ops-solver/discussions/6) 提出，影响验证排期）：

| # | 事项 | 影响 |
| --- | --- | --- |
| 1 | `Xlarft` 无随附用例包，而 §3.5 将其列为必测场景、§3.3 的 P-08/P-09 亦指向它 | `Xlarft` 的验证用例需自建，需确认验收方是否另有提供 |
| 2 | P-03（`sgeqrf m=2048, n=1024`）、P-07（`sormqr trans=T`）、P-11/P-12（`sgels nrhs=8/32`）在随附用例中无对应规格 | 这四条性能项无 CUDA 参照可比，需确认补发或改判据 |
| 3 | `sgels/bench_result.json` 166 行中 108 行 `perf` 为空（65%） | 该部分无法做 0.35× 判定，需确认是否补齐 |
| 4 | §3.3 性能表 P-01~P-12 的"GPU 性能"列全为"待测"，而 `bench_result.json` 已有实测值 | 是否据此回填 |
| 5 | 随附四组用例是必须全覆盖的验收用例，还是仅作示例、其余按 §3.5 由开发者自建 | 决定第 2 项验证计划的工作量归属 |

在上述事项确认前，按"随附用例为样例、其余自建"的保守假设推进，不阻塞开发。

## 兼容性分析

新增算子，不涉及兼容性分析。

**不改公共仓**：`catlass` 与 `src/utils/**` 均不修改。所有定制收敛在本任务自己的 `src/sgeqrf/*`、`src/sorgqr/*`、`src/sormqr/*`、`src/xlarft/*`、`src/ssgels/*` 内（Cube 内核包装、面板核、递推核）。路径门控（`min(m,n) ≤ 64` 走向量路径）与 `nb` 均可通过环境变量回退到保守配置，便于现场定位。