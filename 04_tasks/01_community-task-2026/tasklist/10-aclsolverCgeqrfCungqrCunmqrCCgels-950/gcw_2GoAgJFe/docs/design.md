# 需求背景（required）

## 需求来源

昇腾社区单精度复数 QR 分解与最小二乘任务，面向 Ascend 950PR，在 ops-solver 中提供以下 Host C API 及各自的 workspace 查询接口：

| 功能 | 计算接口 | 语义对标 |
| --- | --- | --- |
| Householder QR 分解 | `aclsolverCgeqrf` | `cusolverDnCgeqrf` |
| 由反射器生成显式 Q | `aclsolverCungqr` | `cusolverDnCungqr` |
| 用 Q 或 Q 的共轭转置作用于 C | `aclsolverCunmqr` | `cusolverDnCunmqr` |
| 方阵及超定最小二乘 | `aclsolverCCgels` | `cusolverDnCCgels` |

四个接口作为同一功能组设计；`aclsolverXlarft` 不属于本任务的公开接口。

## 背景介绍

### 复数 QR 与最小二乘功能

复数 QR 将稠密矩阵分解为酉变换与上三角或上梯形矩阵，可复用同一组反射器完成显式 Q 构造、左右酉变换以及最小二乘求解。反射器形式避免始终存储完整方阵 Q；分块形式将主要尾部更新转换为矩阵乘法。

### 工程接入方式

采用 ops-solver Host C API 与 Ascend C kernel 直调方式。Host 负责参数校验、workspace 查询、路径选择和 tiling，AIV 负责反射器、归约、搬运、三角计算，AIC 负责原生 FP32 矩阵乘法。计算复用 `aclsolverCreate/Destroy/SetStream/GetStream` 管理的 handle 与 stream，不依赖 CUDA，也不以 CPU 计算代替 Device 计算。

### 功能与数据契约

对外矩阵为列主序 COMPLEX64，元素由连续的 `float real`、`float imag` 构成，大小为 8 字节。矩阵位置 `(i,j)` 的元素偏移为 `j*ld+i`。leading dimension 可以大于有效行数，列尾 padding 不是有效输入或输出区域。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现单精度复数 QR 功能组，支持动态 shape、合法 leading dimension、Device 信息输出、调用方 stream 和确定性执行。计算使用 COMPLEX64，内部实部和虚部均使用 FP32。

| 接口 | 主要行为 | 输入保留及输出 |
| --- | --- | --- |
| Cgeqrf | 支持方阵、高矩阵和宽矩阵 | A 原地保存 R 与反射器，TAU 单独输出 |
| Cungqr | 由 k 个反射器生成 Q 的前 n 列 | Q 原地覆盖 A，tau 保持不变 |
| Cunmqr | 支持 LEFT/RIGHT 与 N/C 四种组合 | 更新 C，A 和 tau 保持不变；拒绝 T |
| CCgels | 求解 `min ‖AX-B‖F`，要求 n≤m | A 可覆盖，B 保留，X 为独立输出 |

## 需求拆解

1. 公共类型与 C ABI：定义 COMPLEX64 和 side/trans 枚举，提供八个公开函数。
2. Host 校验与空间规划：校验 shape、leading dimension、指针和 workspace，使用 64 位地址及容量计算。
3. 基础数学：实现缩放 Householder、复数内积、反射器应用和上三角回代。
4. 分块更新：以 64 列 compact-WY 和持久平面布局复用尾部数据。
5. 数据搬运：分离有效区与内部 padding，处理非紧致、奇数尺寸和大 stride。
6. 流与确定性：保持同 stream 的依赖顺序，固定归约和递推次序，避免跨核浮点原子归约。
7. 可测性：分别定义数学残差、接口契约、保留范围和完整 API 的 kernel 时间统计方法。

# 详细设计（required）

## 算子分析

### 数学公式

令 `k=min(m,n)`。QR 使用反射器 `Hj=I-τj vj vjᴴ`，其中 vj 的首个活动元素为 1、此前元素为 0：

```text
Q = H0 H1 ... H(k-1)
R = Qᴴ A，A = Q R
```

构造反射器时，活动列写为 `x=[α;z]`。先以实、虚分量绝对值的最大值 s 缩放，再计算 `‖x/s‖2`，避免直接平方放大上溢、下溢风险。若 z=0 且 α 为实数，则取 τ=0、β=α；否则采用确定的实数相位：

```text
β = -sign0(Re α) ‖x‖2，sign0(a)=+1(a≥0)、-1(a<0)
v = [1; z/(α-β)]，τ=(β-α)/β
Hᴴ x = [β;0]
```

Cungqr 计算 `(H0...H(k-1))[:,0:n]`，即由 k 个反射器生成 m×n 的列正交矩阵，支持 k<n。Cunmqr 计算 `C←op(Q)C` 或 `C←C op(Q)`，`op(Q)` 为 Q 或 Qᴴ。

同一块的反射器可表示为 `Hb=I-V T Vᴴ`。令 G 为 `VᴴV` 的严格上三角部分，Dτ 为 tau 的对角矩阵：

```text
T = (I + Dτ G)^(-1) Dτ
```

T 的每列通过单位上三角递推独立构造，不需要对 tau 求逆，τ=0 也适用。块更新转化为两次复数乘法与一次复数秩更新。

CCgels 对 A 作 QR，先计算 `Y=QᴴB`，再解 `R X=Y[0:n,:]`。超定且采用分块 QR 时，保留原始 A，以 FP32 执行一次校正：

```text
F = B - A_original X
R Δ = (QᴴF)[0:n,:]
X ← X + Δ
```

此路径将 Host niter 置 1，其余路径置 0；不采用 Host 收敛判断驱动 Device 迭代。

### 支持数据类型

| 数据 | 类型 | 存储位置 |
| --- | --- | --- |
| A、B、C、X、tau、前三个接口的 workspace | `aclFloatComplex`，COMPLEX64 | Device |
| 内部实/虚/负虚平面与矩阵乘法累加 | FP32 | Device GM、UB 或 Cube 片上缓冲 |
| devInfo / dinfo | `int`，INT32 | Device |
| m、n、k、nrhs、leading dimension、前三个 lwork | `int` | Host 标量 |
| `_bufferSize` 的输出与 CCgels niter | `int*` 或 `size_t*` | Host |
| CCgels workspace | `void*`，大小以字节计 | Device |

公开头文件通过 C 类型和 `extern "C"` 导出。算法不降低为 FP16/BF16；Cube 显式关闭 HF32。

### 支持形状

| 接口 | 活动问题范围 | 布局及容量约束 |
| --- | --- | --- |
| Cgeqrf | 1≤m,n≤4096 | A 至少覆盖 lda×n，lda≥max(1,m)；TAU 长度 min(m,n) |
| Cungqr | 0≤k≤n≤m≤4096 | A 为 lda×n，lda≥max(1,m)；tau 长度 k |
| Cunmqr | 1≤m,n≤4096，0≤k≤nq | nq=LEFT?m:n；A 为 lda×k，lda≥max(1,nq)；C 为 ldc×n，ldc≥max(1,m) |
| CCgels | 1≤n≤m≤4096 | 支持任务范围 1≤nrhs≤128，接口活动上限为 4096；ldda,lddb≥max(1,m)，lddx≥max(1,n) |

零维问题在枚举、维度、leading dimension、按 shape 所需指针及 workspace 合法后作快速返回。Cungqr 的 k=0 生成矩形单位阵；Cunmqr 的 k=0 保持 C。空问题不要求分配查询值为零的 workspace。

不支持 broadcast，不引入批量维度。Host 按每次调用的实际维数规划；输入数据为有限复数，最小二乘要求 A 满列秩。

## 算子实现

### 实现方案

公开接口如下，参数顺序保持 legacy cuSolver 对应接口的约定：

```c
aclsolverStatus_t aclsolverCgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n,
    aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCgeqrf(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *TAU, aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCungqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k, const aclFloatComplex *A,
    int lda, const aclFloatComplex *tau, int *lwork);
aclsolverStatus_t aclsolverCungqr(
    aclsolverHandle_t handle, int m, int n, int k, aclFloatComplex *A, int lda,
    const aclFloatComplex *tau, aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCunmqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda,
    const aclFloatComplex *tau, const aclFloatComplex *C, int ldc, int *lwork);
aclsolverStatus_t aclsolverCunmqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda,
    const aclFloatComplex *tau, aclFloatComplex *C, int ldc,
    aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCCgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs, aclFloatComplex *dA,
    int ldda, aclFloatComplex *dB, int lddb, aclFloatComplex *dX, int lddx,
    void *dwork, size_t *lwork_bytes);
aclsolverStatus_t aclsolverCCgels(
    aclsolverHandle_t handle, int m, int n, int nrhs, aclFloatComplex *dA,
    int ldda, aclFloatComplex *dB, int lddb, aclFloatComplex *dX, int lddx,
    void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo);
```

`aclsolverSideMode_t` 的 LEFT/RIGHT 为 0/1；`aclsolverOperation_t` 的 N/T/C 为 0/1/2，Cunmqr 只接受 N/C。CCgels 查询函数的 dwork 为兼容参数，可为空；查询不读取矩阵内容。

#### 3.2.1 host侧设计：

Host 从 handle 取得调用方 stream，按顺序完成参数检查、workspace 计算和内核下发。Host 不解引用 Device 数据、不分配内部 Device 缓冲区、不等待 stream 完成；调用方负责查询和分配 workspace，并维持各缓冲区生命周期。

参数非法返回 `ACLSOLVER_STATUS_INVALID_VALUE`，info 可写时在同一 stream 写 `-i`，i 为计算接口中不计 handle 的参数序号。空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`；成功与空问题沿 stream 写 info=0。查询接口没有 Device info 输出，仅返回状态。维数及跨度计算使用 64 位，检查 int/size_t 容量上溢。

##### 1. 分核策略：

核预算查询调用方当前 Device 的 AIC/AIV 数量，支持可见设备映射；预算上限 64，查询失败时使用 AIC=16、AIV=32 的保守值，不缓存无关默认设备的核数。

| 内核类别 | 分工 |
| --- | --- |
| panel Factor | grid≤min(AIC预算,有效列数)，block 0 生成反射器，各核按列分配后续更新 |
| 直接 Apply / Solve | 每核拥有目标列或 RHS，完整递推在核内完成，辅助路径至多 32 核 |
| Gram | 将 count(count-1)/2 个严格上三角内积按 AIV 核预算划分连续区间 |
| Triangle | 每个 block 独立构造一列 T，不同列间没有同步依赖 |
| 复数 GEMM | 每个 AIC 拥有独立输出 slab；不跨核分割同一输出的 K 归约 |
| RankGemm | 总核预算≤min(AIC数,28)，实部/虚部两半独立分配输出 tile |
| 大块 pack、负虚刷新、单位平面初始化 | 按 tile 分配，grid≤min(AIV数,56) |

##### 2. 数据分块和内存优化策略：

小问题用直接向量路径，分块 QR 条件为 `m≥128 && n≥128`。令 nq 为 Q 阶数，rows 为 LEFT 的 n 或 RIGHT 的 m；分块应用条件为：

```text
(nq≥128 && k≥32) || (nq≥64 && k≥16 && rows≥256)
                  || (nq≥32 && k≥16 && rows≥512)
```

分块反射器宽度 P=64。内部实/虚/负虚部为 row-major FP32 平面，逻辑尾部补零到 64 倍数。公共矩阵仍按调用方 leading dimension 寻址；内部补齐不改变对外 shape。

令 `r=pad64(rows)`、`q=pad64(nq)`、C=P=64，分块 scratch 的 FP32 元素数为：

```text
F(rows,nq) = 3rq + 3qC + 3C² + 8rC + max(r,q)C + 2qP + 4P²
E(rows,nq) = 4F(rows,nq) / sizeof(aclFloatComplex)
align4(x) = ceil(x/4)×4，S=544 个 COMPLEX64
```

依次覆盖 D 三平面、V 三平面、T 三平面、W/Y 与 packedA、负虚临时区、复数 V/Gram/T；各段互不重叠。持久 D 只在入口 pack 和出口 unpack，反射器块之间复用其物理行 stride。

| 接口 | 非空 workspace 查询，单位为 COMPLEX64 元素（另注明者除外） |
| --- | --- |
| Cgeqrf | blocked：align4(E(n,m))+S；vector：S |
| Cungqr | blocked：align4(E(n,m))+S；vector：align4(mk)+S |
| Cunmqr | blocked：align4(E(rows,nq))+S；vector LEFT：S；vector RIGHT：align4(mn)+S |
| CCgels | sizeof(COMPLEX64)×[align4(n)+align4(m·nrhs)+max(QRplan,Applyplan)+Refineplan+S] 字节 |

QRplan 为选中分块 QR 时的 E(n,m)，Applyplan 为选中分块应用时的 E(nrhs,m)，否则为 0。Refineplan 只在超定分块 QR 时为 `align4(mn)+align4(n·nrhs)`，保存原始 A 与校正解；其余为 0。QR、应用及再次应用顺序复用 scratch，取最大值而非求和。查询容量是 scratch 需求，不代表总 Device 内存峰值。

上表的非空计算公式受以下快速返回规则覆盖：Cungqr/Cunmqr 的 k=0 时 workspace 查询为 0；m,n>0 的 Cungqr 仍下发矩形单位阵初始化，Cunmqr 保持 C 不变。各接口 m=0 或 n=0 时查询为 0；CCgels 的 nrhs=0 时也为 0。参数合法性仍按前述空问题规则检查。

复数 GEMM 的逻辑 m/n/k 和物理 lda/ldb/ldc 为 64 倍数，内部 slab 可按 16 对齐。Host 向 Cube tiling 传递逻辑 shape、物理 stride、slab 和核数；`SetOrgShape` 保留持久 D 的完整 leading dimension。

Stage1 在 transposeB、N=64、M≥256、K≥256 时仅拆 M：`budget=min(ceil(M/16),AIC预算)`，`tileM=align16(ceil(M/budget))`，实际 blocks=ceil(M/tileM)，保留完整 64 输出列以复用 A。其他长 K 窄输出在原分配未用满预算时枚举 M/N 拆分，先最小化每核 16×16 输出 tile 数，再优先更多核和更少 N 拆分，排除空 slab。

基础 Cube M/N 块取不超过 slab 的 16/32/64/128 二次幂，K 分块交给 FP32 tiling；RankGemm 的 K 固定为 128，按两轴划分输出，任一 slab 不跨实/虚输出分界。

##### 3. tilingkey规划策略：

该工程不是 aclnn 注册算子，不新增注册式整数 TilingKey。Host 以函数分发和 tiling 结构表达等价路径选择，以下为逻辑分类：

| 逻辑路径 | 选择依据 | Kernel 行为 |
| --- | --- | --- |
| Vector / Blocked | 上述 shape 条件 | 逐反射器应用 / compact-WY 更新 |
| LEFT / RIGHT，N / C | side、trans | 目标是否转置、V 是否共轭、T 形式和反射器顺序 |
| Identity | Cungqr 专用 | 直接初始化 D 单位平面，省略初始 A pack |
| Refine / Direct | m>n 且选中分块 QR / 其他 | 一次 FP32 校正 / 主精度直接求解 |
| Continuous / Strided / WideStride | 搬运方向及 DMA stride 是否容纳于 uint32 | 连续、多 block 或逐 block 的 64 位地址搬运 |

#### 3.2.2 kernel侧设计：

内核按 Init、CopyIn、Compute、CopyOut 组织，MTE/Vector 依赖使用事件和 PIPE 屏障。同一 API 的各 kernel 顺序进入同一 stream，跨 kernel 依赖由 stream 保证。

**向量与 panel。** 复数使用交错 FP32 lane，Gather 交换相邻实/虚分量；乘加与共轭通过固定分量操作实现。长度为 `L=align64(2×活动行数)`，最大 8192 lane。Sum/Max 按固定树归约，较长输入分为三阶段，Max 只返回值，不依赖索引排序。panel Factor 使用 MIX_AIV_1_0 和调度模式 1；反射器发布后以 `SyncAll<true>()` 汇合。block 0 更新下一列后通过 MTE3→MTE2 事件确认写回，再生成下一反射器；其他列更新可与其重叠。核间同步使用目标硬件支持的调度机制，不依赖 Host 同步。

**块反射器。** 融合提取 kernel 从 A 读取每列，补隐含对角 1 和其上方 0，输出原始列主序 V 以及转置的实/虚/负虚平面。Gram 用寄存器内复数点积形成严格上三角，分段输出各自独占元素；同列缓存右向量。Triangle 预载 Gram/tau，每列按固定逆序三角递推生成 T。

**持久目标与方向。** LEFT 令 D=Cᵀ，RIGHT 令 D=C。每块执行 `W=D V1`、`Y=W M`、`D←D-Y V3`：

| 操作 | V1 | M | V3 | 块处理顺序 |
| --- | --- | --- | --- | --- |
| LEFT/N | conj(V) | Tᵀ | Vᵀ | 逆序 |
| LEFT/C | conj(V) | conj(T) | Vᵀ | 正序 |
| RIGHT/N | V | T | Vᴴ | 正序 |
| RIGHT/C | V | Tᴴ | Vᴴ | 逆序 |

前两次复数 GEMM 各由四个实数乘积组成：`Cr=ArBr-AiBi`、`Ci=ArBi+AiBr`。优先读取现成的负虚部；第二阶段可用 `Ai(-Bi)` 实现实部修正，避免另一个取负 kernel。每个输出只有一个写入核，K 累加在本核内完成。

第三阶段把 Y 与 V3 打包为实数乘法：

```text
packedA = [ -Yr, +Yi ; -Yi, -Yr ]，shape=(2rows,128)
packedB = [ V3r ; V3i ]，shape=(128,cols)
[Dr;Di] ← [Dr;Di] + packedA packedB
```

RankGemm 直接累加 D 的实/虚平面，不生成整份 R 更新矩阵；随后按连续多行 tile 刷新 `Dneg=-Di`。融合 V pack 的实/虚平面相邻，供 packedB 直接读取。

**分解与显式 Q。** 分块 QR 先 pack Aᵀ。每个 panel 在列主序 A 中分解，从持久 D 只解包下一 panel 的活动区。最后仅写回 `i/P<j/P || j≥min(m,n)` 的 R 元素；同 panel 的对角、上三角和下三角反射器保留 panel Factor 内容，覆盖宽矩阵的额外 R 列。最后不足 P 的块补零，D 更新起点按 P 对齐，最终写回掩码排除反射器区域。

Cungqr 的分块路径直接初始化 D 的单位实部、零虚部/负虚部，原 A 反射器保留到最后 unpack，因此不需要 mk 副本。逆序构造时，当前块 first 之前的 Q 列无支撑，在 D 中跳过相应行；此省略只用于单位阵起点。向量路径先复制反射器，再逐列生成 Q。

**搬运与尾块。** 非转置 pack/unpack 使用 32×64 tile：从列主序 GM 批量读取至多 32 个复数×64 列，经预生成 Gather 索引转换 row-major 平面，逆过程写回有效区。转置目标按连续列搬运。所有 DMA 明确补零右侧不足 32 字节的 lane；输出只写有效行/列，不修改公共 leading-dimension padding。地址与 stride 先按 64 位计算，超出 uint32 stride 字段时退化为逐 block DMA，保留完整 GM 地址。

**最小二乘与信息输出。** CCgels 将 B 复制到工作区后应用 Qᴴ，每个 AIV 独立完成一个或多个 RHS 的逆序回代；复数除法先缩放分母。精确零对角元写正的 dinfo，表示从 1 开始的对角位置。精化以 AIV 计算残差、复用反射器与 R 求校正解，再相加。info 的空问题和参数错误写入也通过调用方 stream 的 Device kernel 完成。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（Atlas 950） | √ |

依赖与目标平台匹配的 CANN 9.0.0 及以上 ACL/Ascend C 工具链。构建时只在 `SOC_VERSION` 匹配 `ascend950*` 的分支加入本功能；其他 SOC 不编译 950 专用向量寄存器与 Cube 内核。

片上预算按 UB 221184 字节、L1 512 KiB、L0A/L0B 各 64 KiB、L0C 128 KiB 规划。以下为静态单 kernel UB 需求，分别复用，不相加：

| 主要缓冲 | 理论 UB 字节数 |
| --- | ---: |
| 六个向量平面与归约区，最大 L=8192 | `(6×8192+256)×4=197632` |
| 非转置 32×64 pack / unpack | 57344 / 65536 |
| 64 行 rank-A 打包 | 98304 |
| 负虚部刷新，至多 16384 FP32 | 65536 |
| 单位平面初始化，两块 16384 FP32 | 131072 |

## 算子约束限制

1. 公共矩阵必须为有效 Device 分配及列主序 COMPLEX64，公共输入须为有限复数；Host 无法从裸指针推断实际容量或 dtype，调用方保证分配与类型契约。
2. 输入/输出及 workspace 按接口保留语义使用不重叠区域；CCgels 的 X 独立于 A/B，禁止以覆盖 B 代替 out-of-place 结果。
3. 仅 Cunmqr 的 N/C 语义有效，T 及非法枚举报错；不支持 broadcast 或额外批量维度。
4. CCgels 不支持 m<n；满列秩是正常求解前提，不作列主元秩估计，不承诺秩亏或任意病态矩阵的前向误差。
5. workspace 必须按本次 shape 查询，前三个接口以 COMPLEX64 元素计，CCgels 以字节计；S 不是分块路径的全部空间。
6. Device info 与输出在 stream 完成后读取；所有 Device 缓冲区在此之前保持有效，不以 Host API 返回代表 Device 已完成。

# 可维可测分析

## 精度标准/性能标准

采用同精度 CPU LAPACK 作为对比链路，残差计算升至 complex128，单位舍入误差 `u=2^-24`；1-范数为复数绝对值的最大列和。ratio_cpu 用同一输入、同一公式计算，阈值按每个 case 确定。

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| Cgeqrf 精度 | `‖R-QᴴA‖1/(m‖A‖1u) ≤ max(5 ratio_cpu,0.1)` | 社区任务书 QR 残差标准 |
| Cungqr 精度 | `‖I-QᴴQ‖1/(mu) ≤ max(5 ratio_cpu,1.5)`，补充与相同反射器乘积的语义比较 | 社区任务书酉性标准及接口语义 |
| Cunmqr 精度 | `‖op2-op1‖1/(nq‖C‖1u) ≤ max(5 ratio_cpu,0.1)`，op2 为相同反射器显式 Q 的运算结果 | 社区任务书应用残差标准 |
| CCgels 方阵精度 | 各 RHS 的 `‖bj-Axj‖1/(m‖A‖1‖xj‖1u)` 最大值 ≤ `max(5 ratio_cpu,0.015)` | 社区任务书求解残差标准 |
| CCgels 超定精度 | `‖(B-AX)ᴴA‖1/(‖A‖1‖B‖1 max(m,n,nrhs)u) ≤ 10 ratio_cpu` | 社区任务书最优性残差标准 |
| 确定性与保留范围 | 同输入、同 stream 串行重复时输出、tau、info 逐位一致；按接口检查只读输入与 padding | 社区任务书接口约束 |
| 性能标准 | 同 shape/dtype/列主序，`T_NPU ≤ T_GPU/0.35` | 社区任务书性能门禁 |

测试设计覆盖方阵/高矩阵/宽矩阵、尺寸边界与非 64 倍数、k=0/k<n、LEFT/RIGHT×N/C、多个 RHS、非默认 stream、奇数及非紧致 leading dimension。另覆盖非法属性、负维度、非法 LD、空指针、空间不足和空问题；检查状态码及 Device info 的参数编号。正常输入采用实/虚分量的均匀/正态混合分布，并设置不参与计算的 padding 哨兵。

性能统计按完整 API 的必要 kernel 总耗时设计：每个 case 执行 2 次预热和 25 次测量，先累计每次 API 内全部 kernel，再取均值。Cungqr/Cunmqr 的输入准备 QR/Q 构造不计入对应 API；CCgels 内部的 QR、pack、Gram/T、GEMM、回代和适用的精化阶段均计入。Host 墙钟、marker 与等待时间不计入 kernel 总时长。可用 Device marker 检查每条 kernel 的唯一归属和调用序列完整性。比较必须具有相同参数的 GPU 基线，缺失基线不能推断倍率。

可维护性以公共向量原语、分块调度、pack、Gram、Triangle、复数 GEMM、实数 RankGemm、精化等模块划分；公共参数校验和 workspace 公式集中定义。精度、语义、内存边界、确定性与性能采集分离，使算法变更可针对相应层次设计回归，而不改变公开接口。

## 兼容性分析

本功能向 ops-solver 增加四个接口族及 COMPLEX64/枚举定义，沿用已有 handle 与状态码。公开参数、顺序和 Host/Device 归属与对应 legacy cuSolver 语义一致；CCgels 明确保留 B、独立输出 X。新增实现不要求 CUDA 库、aclnn 两段式接口或框架张量元数据。

同一公共头文件供 C/C++ 调用，内部 FP32 plane 布局不暴露为公共属性。针对不同 shape 的向量/分块选择、内部 tiling 和空间需求可调整，调用方只依赖 bufferSize 查询及 stream 生命周期。950 专用源码受 SOC 构建条件隔离，其他已有算子的源码收集与接口行为保持原有规则。
