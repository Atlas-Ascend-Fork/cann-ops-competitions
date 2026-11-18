> **状态**：设计评审稿 v1.0（2026-11-17）。报名已通过；本稿申请设计评审，尚未取得 NPU 编译、精度或性能验证结果。
>
> **提交账号**：DR_Tony　　**目标任务**：11月社区任务 aclblasCtpttr 算子开发（Atlas 950PR）

---

# 需求背景

## 需求来源

昇腾 CANN 社区任务（2026 年 11 月）——在 Atlas 950PR（arch35）上使用 Ascend C 开发单精度复数（complex64）packed 三角解包算子 `aclblasCtpttr`，把 packed 存储的三角矩阵解包到列主序 lda×n 矩阵，完成算子设计、开发、测试全流程，验收通过后合入 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）。master 分支 `blas/tpttr/arch35` 只有实数 `aclblasStpttr`，复数版本缺失，本任务用于补齐 Blas 仓能力。

## 背景介绍

### 算子功能

`aclblasCtpttr` 把长度 n(n+1)/2 的 packed 三角数组 AP 解包为列主序 lda×n 复数矩阵 A，只写 uplo 指定的三角（含对角），两种填充模式的 packed 偏移公式不同：

```
LOWER：A(i,j) = AP[j·n − j·(j−1)/2 + (i−j)]，  j ≤ i   （列 j 从对角存到底）
UPPER：A(i,j) = AP[j·(j+1)/2 + i]，            i ≤ j   （列 j 从顶存到对角）
```

其中 0 ≤ i, j < n。该公式为**列主序**口径，与 Netlib `ctpttr.f` 及 cuBLAS 13.5 `cublasCtpttr` 一致。AP 与 A 均为 complex64（`aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 定义为准，实部/虚部各 float32）。本算子是**纯数据搬运**——不改变任何元素数值，A 的另一三角与 lda padding 行保持调用前内容。

### 与实数 stpttr 的本质区别

这个算子的核心难点不在"计算"（没有任何浮点运算），而在**变长映射下的高效搬运**。同目录族已有实数版 `aclblasStpttr`（元素 4B），复数版元素 8B：同样的列解包框架下，32B 对齐粒度从 8 个元素收紧到 4 个元素，packed 列起点（如 UPPER 的 j(j+1)/2）落在 32B 边界上的概率更低，非对齐头尾的处理占比翻倍；同时带宽压力翻倍——GPU 标杆 H100 上 n=8192 仅 0.37 ms，本算子在 NPU 上是与"内存带宽 + 启动开销"赛跑的搬运型算子。另注意方向不可写反：`ctrttp` 是打包（A→AP），`ctpttr` 是解包（AP→A），AP 为入参、A 为独立出参。

# 需求分析

## 需求描述

使用 Ascend C 以 kernel 直调方式开发（host 校验、tiling、kernel 与测试用例统一放在 `blas/tpttr/arch35/`，通过 handle 绑定 stream 直调 NPU kernel），实现 aclblasCtpttr，接口语义对标 cublasCtpttr（列主序），支持 UPPER/LOWER 两种填充模式，偏移量 64 位计算，处理全部 quick return 与负向路径（n=0 no-op、n=0 且 lda=0 报错、非法枚举、负维度、空指针），性能达到任务书 H100 标杆判据。

## 需求拆解

1. 实现 packed AP → 列主序三角矩阵 A 的解包核心搬运，LOWER/UPPER 两套列起点公式不可复用同一套偏移
2. 只写 uplo 指定三角，A 的另一三角与 lda padding 行保持调用前内容（自测以哨兵值校验）
3. 设计高效 Tiling 策略：变长列（长度 1..n）的均衡分核、列段分块、双缓冲、短列合并
4. packed 偏移与 A 偏移全程 64 位整数计算
5. 按 §2.5 顺序完成参数校验：handle 判空 → n < 0 → uplo 枚举 → lda ≥ max(1,n) → n==0 早返回 → AP/A 判空；n=0 且 lda=0 返回 INVALID_VALUE（不得沿用 S 版先返回 SUCCESS 的行为）
6. 性能达到任务书标杆：单 case ≤ 2.5× GPU 标杆耗时、全部 case 平均倍率 ≥ 0.5×（H100 cublasCtpttr）

# 详细设计

## 算子分析

### 数学公式

```
A = unpack(AP)，仅写 uplo 指定三角（含对角）

LOWER：A(i,j) = AP[j·n − j·(j−1)/2 + (i−j)]，j ≤ i
UPPER：A(i,j) = AP[j·(j+1)/2 + i]，           i ≤ j
```

列主序下 A(i,j) 存于 A[j·lda + i]。packed 数组长度 n(n+1)/2，n = 8192 时为 33,558,528 个元素（约 256 MiB），A 为 8192×8192×8B ≈ 512 MiB——所有偏移必须用 64 位整数计算。

### 核心计算分析

**关键洞察：按列看，这是一组"连续→连续"的 memcpy。** 列主序下，uplo 三角的每一列在 AP 中是一段连续内存，在 A 中也是一段连续内存：

| uplo | 列 j 长度 len(j) | AP 列起点 apStart(j) | A 列起点 aStart(j) |
|------|------------------|---------------------|-------------------|
| LOWER | n − j | j·n − j·(j−1)/2 | j·lda + j |
| UPPER | j + 1 | j·(j+1)/2 | j·lda |

即每列就是把 AP[apStart(j) .. apStart(j)+len(j)) 整段搬到 A[aStart(j) .. aStart(j)+len(j))，共 n 段，长度从 1 到 n 不等，AP 侧 n 段恰好首尾相接铺满整个 packed 数组。没有任何数值计算，性能完全取决于：

- **每段搬运的 DMA 效率**：段起点按元素偏移对齐（complex64 8B），32B 对齐 burst 是满带宽的前提，非对齐头尾需单独处理；
- **短列的调度开销**：尾部大量长度 1~2 的短列（LOWER 列 n−1、n−2…；UPPER 列 0、1…），逐列各做一次"读 + 写"的 DataCopy，指令下发与对齐处理的开销会超过数据本身，需要合并处理；
- **多核均衡**：按列等分失衡严重——LOWER 列 0 长 n、列 n−1 长 1，首核负载约为均值的 2 倍、与末核差距随核数放大；带宽瓶颈型算子以最慢核为准，必须按元素量均衡分核。

另外，按列搬运天然保证**写保护**：每列的写区间就是 uplo 三角的一列，永远不会触碰另一三角与 padding 行，无需任何额外掩码逻辑。

### 访存模式选型——列搬运法

参考 Netlib ctpttr.f 的算法思路：**逐列解包**。不按元素逐点搬运（packed 下标到 A 下标的映射非线性，逐点 gather/scatter 效率极低），而是按列整段搬运，每列两侧都是连续内存，DMA 友好。挑战在：每列长度不同（1..n）怎么分核、单列超过 UB 容量怎么切、大量短列怎么合并、段起点非 32B 对齐怎么办。

## 算子实现

### 实现方案

#### host 侧设计

##### 1. 分核策略：按元素量均衡划分

多核并行的维度选择：**按 packed 元素量划分**，每个 core 分到近似相等的一段"列区间"（列边界可落在段中间）。

原因：按列等分时 LOWER 首核负载约为均值的 2 倍、末核趋近于零；本算子是带宽瓶颈型，整体耗时取决于最慢的核。按元素量划分后各核负载均衡，且每核的 AP 读取区间、A 写入区间都是自己列区间内的连续段集合，核间零冲突、零同步。

```
totalElems N = n(n+1)/2                    // int64
MIN_ELEMS_PER_CORE = 4096                  // 分核下限，避免微核/空核
coreNum = clamp(ceil(N / MIN_ELEMS_PER_CORE), 1, 硬件核数)
baseElems    = N / coreNum
bigCoreCount = N % coreNum                 // 前 bigCoreCount 个核各多搬 1 个元素
// core c 的元素区间：[c·baseElems + min(c, bigCoreCount), +len)
```

n 很小（N ≤ MIN_ELEMS_PER_CORE，约 n ≤ 90）时退化为单核，避免启动开销超过搬运收益。

##### 2. tiling 策略：核心在于"列段分块 + 短列合并"

每个 core 负责一段元素区间，可能跨越多列，也可能从某列中间开始/结束。对每列（或列内残段）：

- 段长 ≤ colTileSize：一次 GM→UB→GM tile 搬运（double buffer）
- 段长 > colTileSize：按 colTileSize 切片循环搬运
- **短列合并**：当连续多列的总长不超过一个 tile 时，把它们的 AP 读取合并为一个 tile 读入 UB，再在 UB 内按列边界切分、逐列写出（1 次 DataCopy in + k 次 DataCopy out，替代 k 次 in + k 次 out）——对尾部大量长度 1~2 的短列收益显著

**短列合并的原因**：LOWER 模式的后半段（UPPER 模式的前半段）列长只有 1..几十个元素，逐列"读+写"两次 DataCopy 的指令条数远超数据体量；合并后 AP 侧保持满带宽顺序读，只有写出侧按列切分（写出侧本来就必须逐列——A 中各列不连续）。

##### 3. 数据分块和内存优化策略

本算子 UB 占用极小——纯搬运只需 packed tile 缓冲：

```
bufPacked×2（double buffer）：colTileSize 个元素 × 8B × 2
```

总 UB ≈ 16 × colTileSize 字节：

```
UB_AVAIL = GetUBSize() - 保留区域          // arch35 以运行时查询为准
ELEM_BYTES = 8                             // complex64
BUF_COUNT = 2                              // double buffer
colTileSize = UB_AVAIL / (BUF_COUNT × ELEM_BYTES)
colTileSize = AlignDown(colTileSize, 4)    // 32B 对齐（4 个 complex64 = 32B）
```

以 192 KB 可用 UB 为例：colTileSize ≈ 12288 元素（96 KB × 2）。colTileSize 同时充当 MIN_ELEMS_PER_CORE 的量级基准，单核工作集恒在一两个 tile 内。

##### 4. 路径分派策略（kernel 直调，无 TilingKey）

本算子采用 kernel 直调模式，host 侧不经过 TilingKey/框架二进制匹配流程，而由 **tiling 标量字段承载全部分派信息**：host 侧根据 `(uplo, n, lda)` 计算 tiling 结构（uplo 方向、实际启用核数、元素区间参数、列块大小等），序列化后随启动参数下发；kernel 侧读取标量字段完成分支分派。所有区间参数都是标量（N、baseElems、bigCoreCount、colTileSize），TilingData 不含变长数组——每核起始列由 kernel 用闭式公式自行反解（见 kernel 侧 §1）。

**uplo 统一**：上下三角通过"列长 / AP 列起点 / A 列起点"三个表达式的方向翻转统一：

```
LOWER: len(j) = n − j,   apStart(j) = j·n − j(j−1)/2,   aStart(j) = j·lda + j
UPPER: len(j) = j + 1,   apStart(j) = j(j+1)/2,         aStart(j) = j·lda
```

只需一个方向标志 `uplo` 控制上述三个表达式的切换，单一 kernel 模板即可覆盖两种填充模式，避免维护两份代码（两种 uplo 的列起点不同，不可复用同一套偏移——统一的是表达式选择器，不是偏移公式本身）。

**quick return 短路**：校验顺序严格按任务书 §2.5（前导维校验先于 n==0 早返回，不得沿用 S 版行为）：

```
1. handle == nullptr                     → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. n < 0                                 → ACLBLAS_STATUS_INVALID_VALUE（先于 uplo）
3. uplo ∉ {ACLBLAS_LOWER, ACLBLAS_UPPER} → ACLBLAS_STATUS_INVALID_VALUE（非 INVALID_ENUM）
4. lda < max(1, n)                       → ACLBLAS_STATUS_INVALID_VALUE（n=0 时即 lda<1，覆盖 n0_lda0）
5. n == 0                                → 返回 SUCCESS，不启动 kernel，不写 A（quick return）
6. n > 0 且 AP/A == nullptr              → ACLBLAS_STATUS_INVALID_VALUE
```

**与 TilingKey 方案的对比**：

| 维度 | TilingKey 分派 | 标量字段分派（本方案） |
|------|---------------|----------------------|
| 分支选择 | host 设 key → kernel 查表跳转 | kernel 读 tiling 字段内联判断 |
| 新增分支 | 需注册新 key + 对应 kernel 入口 | 加一个 tiling 字段即可 |
| 二进制数 | 每个 key 可能对应独立编译体 | 单一 kernel 二进制 |
| uplo 处理 | 通常 2 个 key 或 2 套循环 | 一套循环 + 方向标志 |

本算子分支少（uplo 两值）但尺寸连续变化大（n ∈ [0, 8192]、lda 契约上无上限），标量字段天然支持连续值域，与同族实数 aclblasStpttr 的直调风格保持一致。

##### 5. TilingData 结构体

```cpp
struct CtpttrTilingData {
    int32_t n;
    int32_t lda;
    int32_t uplo;             // 0=UPPER, 1=LOWER（方向标志，kernel 内统一表达式）
    int32_t coreNum;          // 实际启用核数（≤ 硬件核数，≥ 1）
    int64_t totalElems;       // N = n(n+1)/2，64 位
    int64_t baseElems;        // 每核基础元素数
    int32_t bigCoreCount;     // 多分 1 元素的核数
    int32_t colTileSize;      // 列搬运 tile 大小（元素数，32B 对齐粒度的倍数）
    int32_t minElemsPerCore;  // 分核下限（host 已据此定 coreNum，kernel 断言用）
};
```

#### kernel 侧设计

##### 1. Init 阶段——元素区间到列区间的反解

```cpp
void Init() {
    blockIdx = GetBlockIdx();
    // 本核元素区间（64 位）
    elemStart = blockIdx * baseElems + Min(blockIdx, bigCoreCount);
    elemEnd   = elemStart + baseElems + (blockIdx < bigCoreCount ? 1 : 0);

    // 反解起始列与列内偏移：列前缀和是 j 的二次函数，闭式求根 + 线性校正
    //   LOWER: cum(j) = (j+1)·n − j(j+1)/2  →  j ≈ n − isqrt(n² − 2·e)
    //   UPPER: cum(j) = (j+1)(j+2)/2        →  j ≈ (isqrt(8e + 1) − 3) / 2
    // 整数开方误差 ±2 列以内，按 cum(j) 单调性做 ≤4 次 O(1) 校正得到精确 (col, offInCol)
    LocateStartCol(elemStart, col, offInCol);

    AllocBuffers();   // bufPacked × 2，double buffer
}
```

反解成本 O(1) 量级；若退化为逐列累加定位则是 O(n) 标量循环，n=8192 时约数十 μs，不可接受，故必须用闭式公式。

##### 2. Process 阶段——列扫描核心循环

```cpp
void Process() {
    uint64 e = elemStart;          // 已搬元素游标（64 位）
    int32  j = col, off = offInCol;
    while (e < elemEnd) {
        int32  colLen = ColLen(j);                    // 方向标志统一表达式
        int32  run    = Min(colLen - off, colTileSize, elemEnd - e);
        uint64 apOff  = ApStart(j) + off;             // 64 位合成
        uint64 aOff   = AStart(j) + off;              // 64 位合成
        CopyTile(apOff, aOff, run);                   // GM→UB→GM，double buffer
        e += run; off += run;
        if (off == colLen) { j++; off = 0; }
    }
}
```

CopyTile 内部按对齐三段式处理（见 §4），并承担短列合并：当 run 因列边界提前截断、且后续多列总长仍在一个 tile 内时，扩大本次 AP 读取、在 UB 内切分逐列写出。

**写保护由循环结构天然保证**：每列写区间严格为 A[aStart(j) .. aStart(j)+len(j)) ⊂ uplo 三角，另一三角与 padding 行自始至终没有写操作，不做任何整矩阵初始化/清零。

##### 3. 64 位偏移计算

n = 8192 时 packed 长度 33,558,528 元素（256 MiB），A 的最大元素偏移 67,108,864（512 MiB；测试矩阵 lda 放宽到 2n 时字节偏移超 1 GB）。虽然该规模下字节偏移尚未溢出 int32，但任务书 §2.1 强制要求 64 位偏移，且 lda 契约上无上限：所有 packed/A 下标与字节地址合成统一使用 uint64_t，DataCopy 入参在 64 位域内完成乘加后再取地址，禁止任何 int32 中间量。

##### 4. 对齐与搬运宽度

complex64 元素 8B，32B 对齐粒度为 4 个元素。列段起点（AP 侧 apStart(j)+off、A 侧 aStart(j)+off）都是元素级偏移，mod 4 任意，每段搬运拆三段：

- **非对齐头**：k = (4 − offset mod 4) mod 4 个元素（k < 4），小粒度/标量搬运处理；
- **对齐主体**：剩余长度向下取整到 4 元素倍数，32B 对齐 burst + double buffer，满带宽；
- **非对齐尾**：剩余 <4 元素，同头部处理。

头尾合计 ≤6 个元素，开销可忽略；主体保证 32B 对齐。对齐逻辑与同族实数版 aclblasStpttr（4B 元素、8 元素对齐）同构，实现上直接参照该算子的处理方式。若 arch35 DataCopy 对 GM 侧非对齐地址的约束更宽松（如支持非 32B 对齐 burst），可简化头尾处理，以实际编译验证为准。

##### 5. 确定性与内存

纯搬运、无浮点运算、无归约，同输入同 tiling 下输出逐元素确定（任务书 §2.5 确定性要求天然满足）。内存峰值 = AP（n=8192 时约 256 MiB）+ A（约 512 MiB）+ UB 内数 MB tile 缓冲，**无整矩阵额外拷贝**（任务书 n=8192 内存核对用例的检查点，自测报告须附内存峰值）。异步执行依赖 handle 绑定的 stream（aclblasSetStream），读回 Device 输出前须同步 stream。

## 支持硬件

| 芯片版本 | 支持 |
|----------|------|
| Ascend 950PR（arch35） | √ |

CANN 版本 9.1.0；接口为全产品线共用的 `include/cann_ops_blas.h` 声明，本任务仅适配 950PR，其他产品线由各自目录演进。

## 算子约束限制

- 数据类型仅 COMPLEX64（`aclblasComplex`，实部/虚部各 float32，定义以 `include/cann_ops_blas_common.h` 为准）
- n ≥ 0，lda ≥ max(1, n)（n = 0 时要求 lda ≥ 1）；n = 0 且 lda ≥ 1 为合法 no-op（quick return，不写 A），n = 0 且 lda = 0 返回 INVALID_VALUE
- 只写 uplo 指定三角，A 的另一三角与 lda padding 行保持调用前内容；A 与 AP 不得重叠（重叠行为未定义，由调用方保证）
- LOWER 与 UPPER 的 packed 列起点公式不同，不可复用同一套偏移；偏移量 64 位整数计算
- 纯数据搬运，bit-exact，不改变元素数值（含 Inf/NaN 按位搬运，保留 NaN 载荷与 Inf 符号）；输出确定性
- 不涉及 broadcast，不要求 dynamic shape；AP 按 packed 连续排列、A 按 lda 前导维访问，不解析任意 stride

# 可维可测分析

## 精度标准/性能标准

| 标准 | 描述 |
|------|------|
| 精度 | 纯搬运取最严格口径：COMPLEX64 实部/虚部均逐元素按位相等（rtol=0, atol=0, matched_ratio=1.0, max_abs_error=0；即在生态标准 FLOAT32 通用档位之上取搬运类口径）；golden 用 LAPACK 3.12.0 `ctpttr_` 生成（CBLAS 无 `cblas_ctpttr`，不得以"未提供 golden"为由跳过比对）或按列主序公式实现 CPU 参考，随测试工程提交；以哨兵值校验 A 另一三角与 lda padding 未被改写；Inf/NaN 输入按位搬运 |
| 性能 | 200 条 TC_PF 用例（性能键 n/uplo/lda）对标 H100 80GB（sm_90，CUDA 13.2，cuBLAS 13.3）基线（gpu_ms 0.0088~0.3735 ms）：单 case 平均耗时 ≤ 2.5× GPU 标杆（≥0.4× 性能），全部 case 平均倍率 ≥ 0.5×，两项同时满足；msprof op 采集单算子 task duration，5 次 warmup 后有效采样 ≥10 次取算术平均 |

测试执行：精度用例（1000 条 CSV，ops-blas C++ GTest 加载，与 200 条 TC_PF 共 1200 条）由 verify_accuracy.py 驱动 build.sh 编译并逐条比对（自动排除 TC_PF_ 前缀）；性能用例由 verify_performance.py 按 `(n, uplo, lda)` 性能键关联 gpu_baseline.csv 比对。输入分布默认 50% 均匀 U(−5,5) + 50% GAUSS_NORM_5_2（μ∈[−5,5]、σ∈[0.1,2] 截断到 [−5,5]，与生态标准 N(0,1) 钳制口径的差异须在自测报告中说明）；性能用例固定均匀分布。CSV 无法表达的场景（AP/A 指针重叠、handle 为 nullptr、多流并发等）在 GTest 中以 TEST_F 手工构造并断言状态码。

## 兼容性分析

新增接口 `aclblasCtpttr` 声明至 `include/cann_ops_blas.h`（仓内当前仅有实数 `aclblasStpttr`），签名与 cuBLAS `cublasCtpttr` 逐参数对应、与同族实数接口同构，供其他产品线共用，不定义 950PR 私有平行 API。实现代码放 `blas/tpttr/arch35/`，测试代码（含 CSV）放 `test/tpttr/ctpttr/arch35/`，README 产品支持表标注 Ascend 950PR：支持；个人仓邀请 Ascend-CANN 为开发者，交付件按模板同步提交至私仓 `task_submission` 目录。
