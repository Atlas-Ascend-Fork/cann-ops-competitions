# aclblasChpmv（Atlas 950PR）算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasChpmv` |
| 目标硬件 | Ascend 950PR（arch35） |
| 软件版本 | CANN 9.1.0 |
| 数据类型 | COMPLEX64（实部、虚部均为 FLOAT32） |
| 开发方式 | Ascend C Kernel 直调，handle 绑定 Stream |
| 实现目录 | `blas/spmv/arch35/` |
| 文档状态 | 设计评审稿 |

截至 2026-10-31，归档自测日志记录实现已在 Ascend 950PR 上完成构建与功能/精度回归——原附件 1000 条正确性用例与 6 个专项用例共 1006 项全部通过，分布补充集与专项共 234 项全部通过，性能集 200 条独立正确性检查全部通过。**性能尚未达标**：任务书要求的 `msprof op` 正式计时在本环境采集失败，改用同环境 ACL event 只能作为候选对照，200 条中仅 2 条达到 `GPU/NPU ≥ 0.4`，故本文档把性能分析写成可检验的优化计划，而非已完成的调优结论。

---

# 1 需求背景（required）

## 1.1 需求来源

2026 年 CANN 社区任务，任务书 `aclblasChpmv_Atlas950PR_task_doc.md`。要求在 Ascend 950PR 上用 Ascend C 实现单精度复数（complex64）Hermitian 压缩存储矩阵-向量乘 `aclblasChpmv`，补齐 ops-blas 仓库在该 BLAS 能力上的空缺，验收通过后合入 `cann/ops-blas`。

## 1.2 背景介绍

BLAS 的 `hpmv` 计算

```text
y = alpha * A * x + beta * y
```

其中 `A` 为 Hermitian 矩阵，以 packed 格式只存储 `uplo` 指定的一半三角。它是 Level-2 BLAS 中同时涉及复数算术、Hermitian 共轭对称与压缩寻址的算子：数学上是标准复数乘加，工程上的难点集中在一套随 `uplo` 完全不同的下标映射，以及由此决定的访存形态。

开发基线（开始本次开发时的仓库状态）如下：

- `include/cann_ops_blas.h` 中没有 `aclblasChpmv` 声明，本次新增；
- `blas/spmv/` 下只有同族的实数算子 `aclblasSspmv`（对称 packed），arch22 与 arch35 两套实现；
- 仓内其他复数算子（`cherk`、`chemm`、`chpr`）各有独立目录与不同的存储约定，不能按家族直接复用。

当前工作树已按本设计新增了 Host、kernel、tiling 与测试工程（见 3.5），上段描述的是开发起点，不是当前状态。

## 1.3 现状与问题

| 项目 | 现状 |
|---|---|
| 接口 | 开发基线尚无声明，当前工作树已新增；须与其他产品线共用同一 API，禁止 950PR 私有平行接口 |
| 参考实现 | cuBLAS `cublasChpmv` 与 Netlib `chpmv` 语义一致，均只存储一半三角、对角只取实部 |
| 同族可复用资产 | `blas/spmv/arch35/sspmv_{host,kernel}.cpp` 提供 SIMT 启动、packed 列基址推进与负步长地址映射的写法 |
| 主要技术风险 | 压缩寻址的对角/共轭下标易错；性能受"一行一线程"算法形态限制，尚未取得正式计时证据 |

---

# 2 需求分析（required）

## 2.1 需求描述

实现 `aclblasChpmv`：complex64 的 Hermitian packed 矩阵-向量乘，句柄式接口，通过 handle 绑定的 stream 直调 NPU kernel；代码位于 `blas/spmv/arch35/`，与同族 `aclblasSspmv` 共用 `spmv` 家族目录。

## 2.2 接口定义

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

参数序列与 cuBLAS `cublasChpmv` 一一对应。`alpha`、`beta` 为 **Host 内存**的复数标量指针，`AP`、`x`、`y` 为 **Device 内存**指针；packed 存储没有前导维度，因此签名中没有 `lda`。`aclblasComplex` 定义于 `include/cann_ops_blas_common.h`（实部/虚部各 float32）。

## 2.3 参数说明

| 参数 | 输入/输出 | 类型 | 约束 | 异常行为 |
|---|---|---|---|---|
| handle | 输入 | aclblasHandle_t | 有效句柄，携带 stream | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | aclblasFillMode_t | {ACLBLAS_UPPER, ACLBLAS_LOWER} | 其他值 → `INVALID_VALUE` |
| n | 输入 | int | n ≥ 0；n = 0 为合法 no-op | n < 0 → `INVALID_VALUE` |
| alpha | 输入 | const aclblasComplex* | Host 内存 | nullptr → `INVALID_VALUE` |
| AP | 输入 | const aclblasComplex* | Device，长度 n(n+1)/2 | nullptr（n > 0）→ `INVALID_VALUE` |
| x | 输入 | const aclblasComplex* | Device，物理长度 1+(n-1)\|incx\| | nullptr（n > 0）→ `INVALID_VALUE` |
| incx | 输入 | int | ≠ 0 | 0 → `INVALID_VALUE` |
| beta | 输入 | const aclblasComplex* | Host 内存；beta = (0,0) 时 y 不必是有效输入 | nullptr → `INVALID_VALUE` |
| y | 输入/输出 | aclblasComplex* | Device，原地覆写，物理长度 1+(n-1)\|incy\| | nullptr（n > 0）→ `INVALID_VALUE` |
| incy | 输入 | int | ≠ 0 | 0 → `INVALID_VALUE` |

表中异常行为按 2.5 的校验顺序解释：有效 handle 且 n=0 时先成功返回，不校验其后的标量与数据指针。

## 2.4 需求拆解

1. **接口与语义**：新增公共声明，与 cuBLAS `cublasChpmv` 参数序列一一对应，复数标量留在 Host。
2. **压缩存储**：AP 按列主序逐列无间隙打包，无前导维度；UPPER 存 i ≤ j，LOWER 存 i ≥ j。
3. **Hermitian 性质**：未存三角由共轭隐含；对角元素虚部不被引用，只参与实部。
4. **边界语义**：n = 0 直接成功且不引用 AP/x/y；beta = (0,0) 不读旧 y；alpha = (0,0) 按 `y = beta*y` 计算且不计算 A·x。
5. **步长**：incx/incy 支持任意非零 int（含负值），物理长度按 1+(n-1)|inc| 保证。
6. **精度**：complex64 的实部、虚部分别按 FLOAT32 档判定（见 4.1），判据取到"最大绝对误差点"。
7. **性能**：逐条满足 `GPU_us / NPU_us ≥ 0.4`，计时口径为任务书规定的 `msprof op` Task Duration。
8. **工程**：新增声明 + 家族目录实现 + 独立测试工程，不扩大为多架构改造。

## 2.5 边界语义

| 场景 | 规定行为 | 测试体现 |
|---|---|---|
| n < 0 | `INVALID_VALUE`，优先于 handle 检查 | TC_ED_192/193、`ValidationOrder` |
| handle 为空 | `HANDLE_IS_NULLPTR`；优先于 n = 0 的快返 | `NullHandle`、`ValidationOrder` |
| n = 0 | 返回成功，不引用 AP/x/y，不下发 kernel | TC_ED_173…179（含叠加非法 uplo/零步长的用例） |
| beta = (0,0) | 不读旧 y；旧 y 中的 NaN 不得成为 alpha*A*x 的污染来源 | TC_ED_183、`BetaZeroIgnoresOldY` |
| alpha = (0,0) | 按 `y = beta*y` 计算，且不计算 A·x（有限输入在点积内部溢出也不得污染结果） | TC_ED_182/184/185、`AlphaZeroNeverEvaluatesAx`、`AlphaZeroBetaZeroWritesZero` |
| 对角虚部非零/NaN | 忽略，只取实部 | `DiagonalImagIgnored`（CSV 用例的 AP 按 Hermitian 约束生成，对角虚部恒为 0，故脏对角只在该专项中构造） |
| 负步长 | 逻辑元素 i 位于 `(n-1)·\|inc\| + i·inc` | TC_INC 全组合（原精度集中 409 条为非单位步长，其中 316 条含负步长） |

---

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
y_i = alpha * Σ_j A(i,j) * x_j + beta * y_i ,   i = 0..n-1
A = A^H ,  AP 中只存 uplo 指定的一半三角
```

复数展开后逐元素为两组实数乘加：

```text
accR += aR*xR - aI*xI
accI += aR*xI + aI*xR
outR = alphaR*accR - alphaI*accI + betaR*oldYR - betaI*oldYI
outI = alphaR*accI + alphaI*accR + betaR*oldYI + betaI*oldYR
```

### 3.1.2 压缩存储索引

记列基址 `base(j)`：

```text
UPPER:  A(i,j), i<=j  ->  AP[base(j) + i]         base(j) = j(j+1)/2
LOWER:  A(i,j), i>=j  ->  AP[base(j) + (i-j)]     base(j) = j(2n-j+1)/2
```

LOWER 的 `base(j)+(i-j)` 展开即任务书给出的 `AP[i + ((2*n-j-1)*j)/2]`，两式等价。

**对角位置随 uplo 而不同**，这是本算子最容易写错的细节：

```text
UPPER 对角 A(k,k) -> AP[k(k+1)/2 + k]
LOWER 对角 A(k,k) -> AP[k(2n-k+1)/2]
```

UPPER 的列从列首向对角方向堆积，LOWER 的列从对角向列尾堆积，两者不是同一个下标公式；golden 侧归一化对角虚部时同样必须按 uplo 分别取址（实现见 `ChpmvDiagIndex`）。

### 3.1.3 Hermitian 与共轭

未存储的另一半按下式读取：

```text
UPPER: j < i 时  A(i,j) = conj(AP[base(i) + j])
LOWER: j > i 时  A(i,j) = conj(AP[base(i) + (j-i)])
```

注意共轭引用的是**列 i 的基址**（`base(row)`），不是列 j 的。kernel 把 `base(row)` 提到行循环内、两段列循环之外计算一次，供整行的共轭访问复用。

对角元素 `i == j` 单独处理：kernel 只读取 `AP` 对角元素的实部分量（`apGm[2*diagIdx]`），虚部分量既不出现在算术表达式里，也不出现在读取语句里。对角不再是逐元素的 `row == j` 运行时判断，而是列循环分段后剩下的一个独立项；两种 uplo 的对角下标由模板参数在编译期选定。至于机器码层面是否完全不读取对角虚部分量，需要反汇编核对，本轮未做该核对，本文档不作断言。

### 3.1.4 与 Netlib 语义的一致性

Netlib BLAS 3.10.0 的 `chpmv.f` 在 UPPER 与 LOWER 两个分支中都显式取对角元素的实部（如 `Y(J) = Y(J) + TEMP1*REAL(AP(KK+J-1)) + ALPHA*TEMP2`，LOWER 分支对应 `REAL(AP(KK))`），其接口注释也写明"对角元素的虚部不必设置，按零处理"。也就是说，Netlib 本来就不引用对角虚部，这与任务书"实现与 golden 均只取对角实部"的要求一致，**不存在冲突，也不存在需要修复的 Netlib 缺陷**。

当前 golden 在调用 `cblas_chpmv` 前把 AP 对角虚部清零，属于测试侧的一项输入归一化：让喂给参考库的数据严格满足 Hermitian 语义，使 golden 结果不依赖具体 CBLAS 实现是否引用该虚部。该步骤只在 AP 中确实存在非零/NaN 对角虚部时才复制一份 AP，原附件用例按 Hermitian 约束生成（对角虚部为 0），常规路径不发生复制。

需要区分两类证据：随工程链接的 CBLAS 若为本地 stub，只能验证调用方行为，不能证明真实 Netlib 库的行为；本轮未在目标设备上对"脏对角 + 真实 Netlib 动态库"做独立实验，故该结论只基于上游源码，不作为设备侧实测结论。

### 3.1.5 负数步长的地址映射

BLAS 约定 `inc < 0` 时逻辑元素 0 位于分配块的高端：

```text
base(inc) = inc < 0 ? (n-1)*|inc| : 0
physical(i) = base + i*inc
```

所有索引在进入乘法前先扩展到 64 位：`n(n+1)/2` 在 n > 65535 时超出 int32，`(n-1)*|inc|` 同理；`inc == INT_MIN` 时先转 int64 再取绝对值，避免 `abs(INT_MIN)` 的未定义行为。

## 3.2 总体方案

单 kernel、按输出行独占写入：

- 每个 SIMT 线程负责一个输出行 `y_i`，行内串行遍历 j = 0..n-1；
- 每个 y 元素只有一个写者，不需要跨核归约、不需要 workspace、不使用原子操作；
- 行内 packed 下标用增量维护（UPPER 已存段每列 `+j`，LOWER 已存段每列 `+(n-j)-1`），避免每步做乘除；
- `uplo` 与 `betaIsZero` 作为编译期模板参数，编译出 4 个特化；`alpha == (0,0)` 走独立的运行时分枝（alpha 全网格一致，不产生 warp 分歧）。

规模与并行度：

```text
blocks  = min(ceil(n / 128), coreNum)
threads = min(align_up(ceil(n / blocks), 128), 2048)
```

即"行数够多时按核均分，行数少时退化为 1 个 block、至少 128 线程"，与同族 `sspmv` 的策略一致。n ≤ 128 时为 1 个 block，此时耗时由 kernel 启动与下发开销主导。

### 3.2.1 为什么选 SIMT 而不是 SIMD

仓内模板规范建议优先评估 SIMD，本算子评估后仍选 SIMT：

1. **首版的数据组织**：当前按输出行独占计算，每个 packed 非对角元素被两个输出行引用，x 也被多行复用。SIMT 可直接表达这种寻址，首版未引入 UB 搬运与协作归约。是否通过 tile 改善复用、合并访问或降低指令成本，需要单独设计和测量，不能仅凭当前实现排除 SIMD/UB 方案。
2. **访问模式随行变化**：同一行内，已存半边按列基址推进、共轭半边按下标 +1 递进，两种模式随 uplo 互换；用 SIMT 的 per-thread 寻址表达最直接，用 Vector 指令需要先构造 gather 索引。
3. **同族一致性**：arch35 上 `aclblasSspmv` 已用 SIMT 实现同一访问模式，两算子共用并行策略，便于维护与对比。

该选择保留了一行内较长的累加依赖链。线程间访存还取决于各线程的列循环是否对齐：固定同一列时，已存半边具有连续布局；当前 UPPER 已存段却从各自的 row+1 开始，不能直接推定连续合并访问（见 5.1）。是否引入列分片与 UB 归约，需由测量回答。

## 3.3 Host 侧设计

### 3.3.1 参数校验

`ValidateChpmvParams` 只负责 uplo / stride / 指针；API 入口额外承担 n 与 handle 的顺序约束，整体顺序由任务书 §2.5 固定：

```text
n < 0                              -> INVALID_VALUE
handle == nullptr                  -> HANDLE_IS_NULLPTR
n == 0                             -> SUCCESS（不触碰任何指针，不下发 kernel）
uplo -> incx -> incy -> alpha -> beta -> AP -> x -> y   -> 依次校验，失败 INVALID_VALUE
```

特别注意与仓内 `chpr` 的差异：`chpr` 把 handle 检查放在 n 之前，本算子不能照抄该顺序。

### 3.3.2 Tiling 数据

```cpp
struct ChpmvTilingData {
    uint32_t nthreads;    // 每 block 线程数
    uint32_t n;           // 阶数
    uint32_t uplo;        // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint32_t betaIsZero;  // 1 表示 beta == (0,0)，选择不读 y 的模板分支
    float alphaR, alphaI, betaR, betaI;
    int64_t incx, incy;
};
```

`betaIsZero` 在 Host 侧判定后写入 tiling（`real == 0 && imag == 0`，`-0.0` 因相等比较也归为零），kernel 据此选择特化，避免在设备端读 y 反推。

### 3.3.3 分核与线程配置

```text
coreNum   = GetAivCoreCount()                    // 取不到（返回 0）则 EXECUTION_FAILED
numBlocks = min(ceil(n / 128), coreNum)
nthreads  = min(align_up(ceil(n / numBlocks), 128), 2048)
```

kernel 侧用 grid-stride 循环覆盖剩余行，因此 `blocks × threads` 小于 n 时也能正确完成，不依赖 Host 侧算出精确的整除关系。

### 3.3.4 Stream 与错误处理

- 使用 `handle->stream` 异步下发，接口内部不强制同步，同步责任在调用方；
- 下发成功后立即返回 `SUCCESS`，执行期错误在调用方同步时暴露（与仓内同族算子一致）；
- 失败路径统一 `OP_LOGE`；launch 前 `OP_LOGI` 打印 blocks/threads/核数，tiling 明细走 `OP_LOGD`。

## 3.4 Kernel 侧设计

### 3.4.1 主循环结构

每个线程按行分段处理，把"逐元素判断属于哪半边"改成三个静态区间，段内不再有 `stored` / `row == j` 选择：

| uplo | 区间 `[0, row)` | 对角 `{row}` | 区间 `(row, n)` |
|---|---|---|---|
| UPPER | 共轭半边，`AP[base(row)+j]`，地址连续 | `AP[base(row)+row]`，只取实部 | 已存半边，`AP[base(j)+row]`，列基址每步 `+j` |
| LOWER | 已存半边，`AP[base(j)+row-j]`，下标每步 `+(n-j)-1` | `AP[base(row)]`，只取实部 | 共轭半边，`AP[base(row)+j-row]`，地址连续 |

x 的物理下标由逻辑下标乘 `incx` 推进，`xBase` 在负步长时指向高端。

### 3.4.2 累加顺序

浮点加法的结合顺序会改变结果，且该算子与单线程参考实现比对，因此各条路径的累加结构必须明确写清并分别验证：

- **UPPER**：共轭段（`mirR/mirI`）与对角项先合并，再逐列累加已存段。即 `acc = diag + mirror`，之后 `acc += stored_j`。
- **LOWER，alpha == (1,0)**：已存段与共轭段各自串行累加，最后 `acc = str + mir + diag`。
- **LOWER，其他非零 alpha**：已存段按 `j & 3`、共轭段按 `(j-1) & 3` 分配到各自 4 条独立累加链；段末按 k=0…3 顺序把 `str[k]+mir[k]` 加入总和，最后叠加对角项。这是固定顺序合并，不是平衡二叉树归约。

为什么 LOWER 要区分 alpha = (1,0)：无条件改用四条链的变体在补充集与性能集正确性上表现不一致（补充集全过、性能集正确性 15/200 失败），保留 alpha = (1,0) 的串行路径后三组回归才同时通过。这是当前版本经实测选定的数值路径，不是语义差异——两条路径使用相同的数学公式和精度判据，当前已测集合满足要求，但这些结果不构成对所有未测试输入的误差保证。该分支属于本版本的技术债：它按标量取值选择累加顺序，可维护性弱于统一结构，后续若找到同时满足三组回归的统一累加方案应当合并。

### 3.4.3 alpha = (0,0) 分支

```text
alpha == (0,0) -> y_i = beta * y_i
beta == (0,0)  -> 写 0，且全程不出现对 y 旧值的读取
```

必须独立成路径而不能"算出 A·x 再乘 0"：AP、x 全为有限值时点积内部仍可能先溢出为 Inf，`0 × Inf = NaN` 会把本应为 `beta*y` 的结果污染。任务书禁止的是 alpha = 0 与非有限 AP/x 的组合，不覆盖这一反例，故实现按"不计算 A·x"处理，并有专项用例 `AlphaZeroNeverEvaluatesAx` 用 `AP = 3e38、x = 2` 的构造做回归。

### 3.4.4 beta = (0,0) 分支

```cpp
if constexpr (BETA_IS_ZERO) {
    out = alpha * acc;                 // 不出现对 yGm 的任何读
} else {
    out = alpha * acc + beta * yOld;
}
```

用 `if constexpr` 而非运行时 `if (beta == 0)`，保证 beta = 0 时编译器不会保留旧 y 的读——旧 y 可能是 NaN，读一次再乘 0 仍是 NaN。这与 3.4.3 是同一类问题：语义上"不读取"必须落实为代码里不存在该次读。

### 3.4.5 数值与溢出

- 累加使用 float，与 golden 同为 FLOAT32 精度档；
- AP 索引、x/y 物理索引用 int64；`base(j)` 增量不会超过 `n(n+1)/2`，在 n ≤ INT_MAX 时远小于 2^63；
- 不申请 workspace、不使用原子累加，结果与线程调度无关（确定性属实现性质，任务书未要求）。

### 3.4.6 内存预算

| 规模 | AP 字节数（n(n+1)/2 × 8） | 说明 |
|---|---|---|
| n = 2048 | 16.8 MB | 精度/补充集最大规模 |
| n = 8192 | 268.5 MB（256 MiB） | 性能集最大规模 |

算子本身不额外申请 Device 内存，也不申请 UB：当前 kernel 完全靠寄存器与 GM 直读，没有 shared memory 缓冲，因此不存在 UB 分块与 double buffer 预算。测试侧仅在 golden 需要归一化脏对角虚部时复制 AP，原附件用例不发生复制，该数值只是 AP 输入体积，不是测试进程的 Host 峰值：当前正确性测试还保留 apOrig 副本，n=8192 时仅这两份 AP 就约 536.9 MB（512.06 MiB），另有 x/y、golden 和运行时开销。32 MiB 以内的用例还可能申请 Device 回读镜像；计时路径与正确性路径的内存分别核算。

## 3.5 文件结构设计

```text
include/cann_ops_blas.h                     # 新增 aclblasChpmv 声明（与其他产品线共用）
blas/spmv/README.md                         # 新增 Chpmv 章节与产品支持标记
blas/spmv/arch35/chpmv_host.cpp             # 校验 + tiling + launch + API 入口
blas/spmv/arch35/chpmv_kernel.cpp           # 4 个模板特化 + alpha=0 分支 + kernel_do
blas/spmv/arch35/chpmv_kernel.h             # kernel_do 声明（host/kernel 共用）
blas/spmv/arch35/chpmv_tiling_data.h        # Tiling 结构
test/spmv/chpmv/CMakeLists.txt
test/spmv/chpmv/README.md                   # 复现步骤与判据说明
test/spmv/chpmv/chpmv_param.h               # CSV 行解析
test/spmv/chpmv/chpmv_golden.h              # CPU golden + 精度判定
test/spmv/chpmv/arch35/chpmv_npu_wrapper.h  # Host↔Device 搬运与 Device 回读
test/spmv/chpmv/arch35/chpmv_test.cpp       # GTest 入口（CSV 驱动 + 专项）
test/spmv/chpmv/arch35/chpmv_test.csv       # 1200 条附件用例（与任务附件逐字节一致）
```

构建侧不改公共 CMake：`blas/CMakeLists.txt` 递归收集 `*/*/archXX/*.cpp`，`cmake/test.cmake` 通过 `blas/*/arch35/<op>_*.cpp` 判定算子存在，新增文件自动纳入。

## 3.6 支持硬件

| 芯片版本 | 是否支持 |
|---|---|
| Ascend 950PR | √（本任务实测环境：CANN 9.1.0 / arch35） |
| Ascend 950DT | 未验证（同属 arch35，但本任务没有 950DT 实机数据，不作承诺） |
| Atlas A3 训练/推理系列 | × |
| Atlas A2 训练/推理系列 | × |

## 3.7 算子约束限制

- 不支持除 incx/incy 语义之外的非连续内存访问；
- 不承诺 AP 与 x 或 y 之间可以别名（含 x 与 y 重叠），任务书未要求；
- 不提供 aclnn 版本，不定义 950PR 私有接口；
- `beta = (0,0)` 时 y 的内容不必有效（实现不读旧值），但指针本身仍须非空，与任务书参数表一致。

---

# 4 可维可测分析

## 4.1 精度标准

来源：任务书 §3.2 指定的 opbase 固定 SHA 版本 `docs/zh/ops_precision_standard/mixed_tolerance_standard.md`（§2.1.2/§2.2），FLOAT32 档（`rtol = atol = 2^-13`）。实部、虚部分别统计，逐元素匹配条件是

```text
abs(actual_i - golden_i) <= 2^-13 + 2^-13 * abs(golden_i)
```

用例级判定不是"每个元素都满足各自的硬上限"，而是只对**一个**最大绝对误差点检查：

```text
有限点集合 F = { i | actual_i 与 golden_i 均有限 }
e_i = abs(double(actual_i) - double(golden_i))
k   = argmax_{i in F} e_i                 // 并列时取最小下标，并单独记录并列点数
E   = e_k
L   = max(1e-2, 32 * ULP(float32(golden_k)))   // 锚点是 k 点 golden 的 float32 ULP
用例通过 = 有限点 matched_ratio >= 0.99 且 E <= L 且非有限点全部按 §3.2 规则通过
```

三个容易写错的点：

1. 锚点取最大误差点自身的 golden，**不是全向量 `max|golden|` 的 ULP**，也不是逐点各自对自己的 ULP 上限；
2. 并列最大误差必须有固定可复现规则，实现取最小下标，不允许事后挑更宽松的点；
3. `MERE`/`MARE` 只在日志中作诊断，不参与通过判定（CSV 的 `mere_threshold`/`mare_multiplier` 两列仅作记录）。

非有限值按任务书 §3.2 专门规则处理：同 NaN、同为同号 Inf、一方 NaN 另一方 ±Inf 视为等效；一方有限另一方非有限判失败；非有限元素不计入 matched_ratio 与最大绝对误差统计；有限元素个数为 0 时匹配率记 N/A，不以除零构造"通过"。

golden 为 Netlib CBLAS 3.10.0 的 `cblas_chpmv`（`CblasColMajor`），调用前按 3.1.4 做对角虚部归一化。

## 4.2 测试计划与实际执行范围

用例来源与关系必须先讲清，避免把同一条用例在不同套件里重复计数：

| 集合 | 文件 | 条目数 | 一次 GTest 实际执行数 |
|---|---|---|---|
| 任务附件正确性集 | `chpmv_test.csv`（TC_L0/CV/SQ/AB/INC/FL/ED/EX） | 1000 | 1000 + 6 专项 = 1006 |
| 任务附件性能集 | 同一 CSV 中的 TC_PF | 200 | 200（正确性运行）或 200（计时运行） |
| 分布补充集 | `chpmv_test_supplement.csv` | 228 | 228 + 6 专项 = 234 |

即：CSV 共 1200 行；"1000 条精度"与"200 条性能"是同一文件的两个子集，不可相加成 1200 条独立精度用例；补充集 228 条是独立文件，统计时单列，不与附件集合求并集。

原附件 CSV 保持逐字节不变（1201 行含表头），补充集的引入原因是：附件常规随机用例仅使用均匀填充（另有零、交替、极值、Inf/NaN 等特殊填充），未覆盖任务书 §3.5 要求的"均匀/正态各 50%"与随机复数标量分布。补充集分组如下：

| 分组 | 条数 | 覆盖点 |
|---|---|---|
| TC_DIST | 160 | (ap, x, y) 的 8 种均匀/正态组合 × UPPER/LOWER × n 扫描，每个 tensor 恰好 50% 均匀 / 50% 正态 |
| TC_SCA | 36 | alpha ∈ {(0,0)、(1,0)、纯虚、大值}，beta 轮转 |
| TC_SPEC | 16 | AP 或 x 含 Inf/NaN（不与 alpha=(0,0) 组合） |
| TC_NEG | 16 | n = 0、n < 0、零步长、空指针、非法 uplo |

附件正确性集的分布（1000 条）：

| 类别 | 条数 | 覆盖点 |
|---|---|---|
| TC_L0 / TC_CV | 4 / 16 | uplo × 小/中尺寸基础路径 |
| TC_SQ | 46 | n = 1…2048，质数、非对齐、2 的幂 ±1 |
| TC_AB | 16 | alpha/beta 取 0、1、±1、纯虚、一般复数 |
| TC_INC | 72 | incx × incy ∈ ±1/±2/±3；全集中 409 条非单位步长、316 条含负步长 |
| TC_FL | 18 | 零、交替、极值、Inf、NaN（AP/x/y 分别） |
| TC_ED | 25 | n = 0、非法 uplo、负 n、空指针、alpha = 0 |
| TC_EX | 803 | 尺寸池 × uplo × 标量 × 步长扩展 |

另有 **6 个**不读 CSV 的 GTest 用例：

- `NullHandle`：空 handle → `HANDLE_IS_NULLPTR`；
- `ValidationOrder`：n < 0 / 空 handle / n = 0 / 非法 uplo / 零步长 / 空 alpha 的优先级；
- `DiagonalImagIgnored`：对角写入非零与 NaN 虚部，结果仍与只取实部的 golden 一致；
- `AlphaZeroNeverEvaluatesAx`：alpha = (0,0)、AP/x 全有限但点积内部溢出，结果仍为 beta·y；
- `BetaZeroIgnoresOldY`：beta = (0,0)、y 预置 NaN/Inf，结果必须有限；
- `AlphaZeroBetaZeroWritesZero`：alpha = beta = 0 时 y 写 0 且不读旧值。

CSV 用例的附加检查：

- **AP/x 只读核验必须比对 Device 回读**。测试在调用后把 Device 上的 dAP/dX 拷回，与调用前的 Host 副本逐字节比对；只比较一直留在 Host 的缓冲区是无效检查。该核验仅对 `apBytes + xBytes ≤ 32 MiB` 的用例启用，日志打印 `device-input-probe=yes|skipped`。当前附件正确性集最大 n = 2048（AP 约 16.8 MB），全部在预算内，没有 skipped 记录；无 probe 的用例只有提前返回的负向/n = 0 用例。
- **stride 间隔检查 ≠ 越界检测**。`|incy| > 1` 时检查逻辑元素之间的间隔位置是否仍为哨兵，只能说明这些位置未被写入，不能证明分配范围之外没有越界。
- **性能集的独立正确性运行与计时运行分开**：正确性运行照常做 golden 比对；计时运行（`BLAS_MEASURE`）只计时，不比对，避免把大 n 的比对开销并入计时区间。两者都不屏蔽已知失败。

复现入口（与 `test/spmv/chpmv/README.md` 一致）：

```bash
bash build.sh --soc=ascend950 --ops=chpmv          # 只构建本算子
./build/test/spmv/chpmv/chpmv_test                 # 全量 CSV 用例
./build/test/spmv/chpmv/chpmv_test --gtest_filter='*TC_INC*'   # 按类过滤
```

正式性能采集按任务书使用 `msprof op`；该命令在本环境的日志中报 `Profiling channel start failed`（进程退出码仍为 0），未取得 task duration，因此本文档的性能数据均不含 msprof 口径。

## 4.3 验证结果

下表只统计**当前默认实现**（本文档 3.4 描述的结构，即 LOWER 在 alpha=(1,0) 时保留串行累加、其余非零 alpha 用四条链）的结果。更早的版本（单累加链、LOWER 无条件四链）结果不作为本版本结论，历史记录单独归档。

| 验证集 | 执行数 | 结果 | 说明 |
|---|---|---|---|
| 附件正确性 + 6 专项 | 1006 | 1006 通过 | 测试脚本按完整用例 ID 校验计划与实际集合，无缺失/重复 |
| 分布补充集 + 6 专项 | 234 | 234 通过 | 含 160 条分布用例与 16 条特殊值用例 |
| 性能集独立正确性 | 200 | 200 通过 | 与计时运行分离，未屏蔽比对 |
| 性能集 event 计时 | 200 | 2 条 ≥ 0.4 | 见 4.4，非正式验收口径 |

当前累加路径曾在多组回归间出现不同的误差表现，因此按 3.4.2 保留明确的数值路径选择。各轮实验记录单独归档；本表仅描述上述实现的已测结果，不把不同候选的通过项拼接成最终结论。

证据限制：现有归档保存了 kernel 源码指纹、构建日志与测试程序指纹，但未留存每轮 `libops_blas.so` 指纹。测试程序动态链接该库，程序指纹相同不能单独证明实际加载的 kernel 相同。因此这些结果属于按源码与重编记录关联的阶段自测；最终验收还须记录被加载动态库的指纹并绑定全量验证。

## 4.4 性能标准与当前状态

**正式口径**（任务书 §3.3）：`msprof op` 采集单算子 Task Duration（us），5 次 warmup 不计入样本，取 10 个有效样本的算术平均；逐条满足 `GPU_us / NPU_us ≥ 0.4`。

任务书内部存在"有效采样 10 次"（§3.3）与"有效采样 >10 次"（§7）的表述差异。本设计以 §3.3 主表的 10 个有效样本为准，并在交付材料中按交付清单补充更多采样；当前实际只采了 10 次，没有冒充已采 20 次或更多。

**当前状态：正式性能验收待完成。** `msprof op` 在本环境报 `Profiling channel start failed`，对相邻算子 `Scasum` 做同样的最小对照也失败（kernel 采集 0 成功、1 失败），而 msprof 进程退出码仍为 0——因此不能仅凭退出码判定采集成功。该现象支持"环境/工具侧共同问题"，尚不能确定具体原因，需要环境方确认权限或容器能力。

**可用替代：ACL event 区间计时**（同环境候选对照，不作为验收证据）。计时点为 `RecordEvent(start)` → 执行一次 `aclblasChpmv` → `RecordEvent(end)`，因此区间内可能夹入 Host 下发间隙、tiling 计算与 launch 开销，不能等同于纯 kernel task duration。当前实现的 200 条结果：

| 指标 | 数值 |
|---|---|
| 达到 `GPU/NPU ≥ 0.4` 的用例 | 2 / 200（n = 1：5.80 us vs GPU 3.49 us，0.60；n = 2：6.89 us vs 2.91 us，0.42） |
| 倍率范围 | 0.167 – 0.602，中位数 0.248 |
| 典型点 | n = 4：7.57 / 2.78 = 0.37；n = 512：166.0 / 31.2 = 0.19；n = 8192：5004.7 / 1161.7 = 0.23 |

倍率随 n 非单调：n ≤ 2 时能到 0.4 以上，n = 4…128 逐渐降到 0.20 附近，n ≈ 3000–3400 回升到 0.34–0.38，n ≥ 7000 又降到 0.22–0.25。分核随 `min(ceil(n/128), coreNum)` 阶跃变化是可疑因素之一，但在没有 task duration、核数与 stall 计数的情况下不作根因断言。

表中的 200 条来自同一个归档采样批次，保留了 2000 个有效样本，并与该批次源码及重编记录关联；动态库指纹限制见 4.3。此前几轮 event 日志分属不同构建，口径相同但版本不同，不能合并成一张结果表，本文档不引用它们的数字。

## 4.5 兼容性分析

新增算子，不改动既有接口与实现，不涉及兼容性迁移。`include/cann_ops_blas.h` 中的声明与仓内 `aclblasSspmv` 同层，其他产品线共用同一符号。

---

# 5 性能分析与优化计划

## 5.1 访存形态

先区分压缩布局的地址关系与当前 SIMT 循环实际发出的地址。

若相邻线程负责相邻行 r，且当前访问同一个列 j，UPPER 满足：

```text
r <= j：idx = base(j) + r      // 已存半边：随 r 连续
r >  j：idx = base(r) + j      // 共轭半边：随 r 跨列变化
```

LOWER 在相同列 j 下同样是已存半边随 r 连续、共轭半边跨列变化。但“固定 j 时连续”不等于实际执行必然合并：当前 UPPER 已存段的起点是各线程自己的 `j=row+1`，同一轮循环中相邻线程的 j 不同。记迭代偏移为 t，其访问为：

```text
j_r = r + 1 + t
idx_r = base(j_r) + r
idx_(r+1) - idx_r = j_r + 2  // complex64 元素数，通常不为 1
```

因此该 UPPER 分段循环不能直接套用“已存半边线程间连续”的结论。LOWER 已存段从 j=0 开始，在活跃线程访问相同 j 时保留连续地址关系；共轭段仍跨列散读。循环分歧、实际指令调度及缓存行为还需结合生成代码和 profiler 分析。

优化时需明确 lane 对应的行/列、列循环起点及活跃掩码。可比较按线程组统一列起点与当前按行起点的差别，确认是否能改善访问；不能只因把循环命名为“列块”就认定已解决散读。

## 5.2 逻辑引用量与实际事务量

行独占算法对每条输出行都要遍历整行，因此逻辑上引用 n² 个复数矩阵项（含对角）。packed 只存 n(n+1)/2 个元素，每个非对角元素在两个输出行的计算里各被引用一次，所以：

```text
n = 8192：逻辑引用量 8192² × 8 B ≈ 536.9 MB
         独立存储量 8192×8193/2 × 8 B ≈ 268.5 MB
```

这两者都不是 HBM 事务数：重复引用是否命中缓存、散读是否被合并、实际发出多少事务，都需要 profiler 的访存计数才能回答。按 event 均值 5.00 ms 折算，逻辑字节率约 107 GB/s、独立存储字节率约 54 GB/s，仅作量级参考，**不能用来说明硬件带宽利用率，也不能据此推断 GPU 侧的内部实现**。

## 5.3 优化计划

原则：每条候选只改一个可解释的因素，先过正确性再比性能；不通过修改判据、屏蔽大 n 比对或裁剪用例来"达标"。正式计时恢复前，用 event 做同环境相对比较，并明确它不是验收口径。

| 编号 | 内容 | 预期解决的问题 | 状态 |
|---|---|---|---|
| V0 | 恢复 `msprof` 通道，记录工具/驱动/固件版本与完整错误；在 Host 下发日志中记录实际 blocks/threads/核数 | 建立可信基线；当前日志未留存 AIV 核数与实际并行配置 | 待环境方确认 |
| V1 | 分核与线程配置复测（128/256/512 线程 × 合理 block 数） | 小 n 与阶跃点附近的利用率 | 已做 9 条代表性用例的 256/512 对照（跨批次 event 均值），未见稳定收益，未晋级；不据此排除其他配置 |
| V2 | 统一循环分段与累加结构（把 3.4.2 的 alpha=(1,0) 分支合并为单一策略） | 可维护性 + 依赖链长度 | 待做，需同时满足三组回归 |
| V3 | 列分片 + 共享 UB 归约（C-lite）：一个 block 负责 32 行，lane 映射行、slice 映射列区间，每 slice 独立部分和，`__ubuf__` 存部分和，`asc_syncthreads` 后按行归并，唯一写 y | 缩短归约链、提高并行度；统一各 slice 的列区间后核对同列活跃 lane 的连续访问 | 待做 |
| V4 | 为连续半边与共轭半边分别设计 lane 映射（必要时 tile/混合映射） | 散读半边 | 待做，且必须先说明输出所有权与最终归约 |

V3 的资源与正确性约束（写清是因为它是本轮最可能晋级的方案）：

- 部分和缓冲：32 行 × S 个 slice × 2 个分量 × 4 B；S = 8 时约 2 KiB，此项仅覆盖部分和数组，落地前还需核算所有同时存活的 UB/临时区、对齐与编译器资源约束；
- 归约必须在整个 block 一致到达的同步点上完成，尾行线程贡献零而不是提前 `return`，否则 `asc_syncthreads` 会死锁；
- 若使用 `asc_shfl_down`，源 lane 必须参与调用，尾部填零并保持组内一致；
- `alpha`、`beta` 只在最终写 y 时应用一次；`beta = (0,0)` 的路径不得读旧 y；
- 引入部分和后，归约顺序改变会影响消减/大幅值用例，每改一次都要重跑 TC_FL、TC_AB 与补充集特殊值用例。

已确认可用、但尚未在本任务编译验证的原语：`__ubuf__` 共享内存、`asc_syncthreads`、`asc_shfl_down`（warp 32 线程）。仓内 `blas/gbmv/arch35/cgbmv_kernel.cpp` 与 `blas/asum/arch35/scasum_kernel.cpp` 有共享 UB 与 warp/block 归约的现成写法。在线接口文档的产品支持表列出了 950PR，但这不等于本任务候选已在 CANN 9.1.0 上编译通过，V3 落地时以实测编译为准。

不在本轮计划内：跨核 y 原子累加（任务书未要求确定性，但争用与初始化成本需另行评估）、按列加载同时贡献两个输出的 tile 方案（需要完整说明输出所有权与归约，工作量大且未证明收益）。

---

# 6 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 性能未达 0.4 门槛 | 当前 event 口径仅 2/200 达标，是验收的主要缺口 | 按 5.3 排序推进；正式口径恢复前不以 event 数字替代 |
| `msprof` 不可用 | 无 task duration，无法给出验收级性能数据 | 保留 event 对照与逐次样本；用相邻算子对照缩小排查范围，并由环境方核查工具、驱动配套与采集能力，不把共同失败等同于唯一原因已确定 |
| 累加顺序敏感 | 改变归约结构会改变与单线程 golden 的误差分布，曾出现"补充集通过、性能集失败" | 每次改动同测三组回归；不接受用放宽判据换取通过 |
| 小 n 启动开销 | n = 4 附近倍率 0.37，紧贴门槛 | 在正式口径下确认是否为固定开销；必要时评估更轻的 launch 配置 |
| 散读半边 | 共轭半边跨线程散读，是长尾耗时的主要怀疑对象，但无事务计数佐证 | 先取得 profiler 访存指标再决定是否做 V4，不预设 4× 放大 |
| Device 只读核验的覆盖边界 | 32 MiB 预算外的用例不做回读 | 在日志中标注 skipped；代表性大规模用例另做只读核验，避免把未执行检查写成通过 |
| 950DT 未验证 | README/文档若声称支持会超出证据 | 只承诺 950PR，950DT 明确标注未验证 |

---

# 7 结论

本设计把 Hermitian packed 的寻址（两种打包、隐含共轭、对角只取实部、负步长）、边界语义（n = 0、alpha = 0、beta = 0）、精度判据（最大误差点锚点）与测试口径（CSV 子集关系、Device 回读、计时口径）固定下来，实现已在 Ascend 950PR 上完成功能与精度回归：附件 1006 项、补充集 234 项、性能集独立正确性 200 项全部通过。

**未完成项明确列出**：正式 `msprof` 性能数据尚未取得，event 口径下仅 2/200 条达到 0.4 倍门槛，性能优化仍在进行；`alpha=(1,0)` 的独立累加路径是可维护性技术债；950DT 未验证。本文档不宣称任务已验收通过。

---

# 8 参考资料

1. [Netlib BLAS 3.10.0 `chpmv.f`](https://raw.githubusercontent.com/Reference-LAPACK/lapack/v3.10.0/BLAS/SRC/chpmv.f)
2. cuBLAS `cublasChpmv`：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpmv
3. 生态算子开源精度标准（固定 SHA）：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. ops-blas 仓：https://gitcode.com/cann/ops-blas
5. [活动说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)与[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
6. Ascend C [SIMT 编程模型](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/overview.html)与[asc_shfl_down](https://asc.gitcode.com/api/SIMT-API/Warp_functions/Warp_shfl_functions/asc_shfl_down.html)（当前版本，目标工具链仍需单独编译验证）
