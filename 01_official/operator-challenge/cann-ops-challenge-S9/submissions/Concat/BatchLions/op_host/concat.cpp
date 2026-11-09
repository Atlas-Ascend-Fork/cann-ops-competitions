/* Concat 的 Host 侧：InferShape / InferDataType / Tiling / 算子定义。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tiling 的决策依据都在 ops/README.md 里有量测记录，这里只摘结论：
 *   - 默认**按行切**（burst 最长）；只有按行切用不满核时才按列切，
 *     且列切能用几个核由 burst 下限反推（nMin·esz / MIN_BURST）。
 *   - 每核约 4 轮反推 ubBudget，夹在 [8 KB, 92 KB]。
 *   - TilingKey ＝ 元素字节宽度（1/2/4/8）：Concat 只搬不算，
 *     fp16/bf16/int16 走同一份实现。
 */
#include "concat_tiling.h"
#include <cstdlib>
#include <cstdio>
#include "register/op_def_registry.h"
// **切核与 UB 预算的算术在这个头里**，它不依赖任何 CANN 类型，因而能单测
//（op/src/test/test_tiling.cpp，并与 ops/test/gen_*.py 的已验证实现交叉核对过）。
// 这里只负责从 gert 取 shape/attr/dtype，再把结果填进 TilingData。
#include "s9_tiling_calc.h"
#include "s9_platform.h"
#include "s9_env.h"          // 环境开关的统一解析（原来这里是三十多段手抄的三行模板）
#include <vector>

namespace optiling {

/* **两个 CONCAT_MAX_IN 必须一致，否则静默算错。**
 *
 * `s9::CONCAT_MAX_IN`（s9_tiling_calc.h）是纯算术侧的上限，能单测；
 * `optiling::CONCAT_MAX_IN`（concat_tiling.h）定的是 TilingData 里数组的长度。
 * 分叉的后果：host 按大的那个填 ct.nIn[]，`tiling.set_nIn()` 只存得下小的那个，
 * kernel 用 `offsetof(ConcatTilingData, nIn) + i*4` 读第 N 项以后就越过数组边界，
 * 读到 offOut 甚至更后面的字段 —— **编译不报错，结果全错**。
 *
 * 这次就是这么踩的：只改了 s9 那个（64→1024），差点又交一版挂的。 */
static_assert(optiling::CONCAT_MAX_IN == s9::CONCAT_MAX_IN,
              "concat_tiling.h 与 s9_tiling_calc.h 的 CONCAT_MAX_IN 必须一致");
// 每带计划数组的长度（TilingData 里的 64）也必须与纯算术侧的 BAND_MAX 一致：
// host 按大的写、结构按小的存 ⇒ kernel 读到别的字段，且**编译不报错**。
static_assert(64 == s9::BAND_MAX,
              "concat_tiling.h 的 bandLo/bandHi/bandOff 长度必须等于 s9::BAND_MAX");


static uint32_t DtypeBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_INT8: case ge::DT_UINT8: case ge::DT_BOOL: return 1;
        case ge::DT_FLOAT16: case ge::DT_BF16: case ge::DT_INT16: case ge::DT_UINT16: return 2;
        case ge::DT_FLOAT: case ge::DT_INT32: case ge::DT_UINT32: return 4;
        case ge::DT_DOUBLE: case ge::DT_INT64: case ge::DT_UINT64: case ge::DT_COMPLEX64: return 8;
        default: return 0;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // **核数向平台要，不写死。** 见 s9_platform.h 顶部：
    // AIV_MAX 只是查不到时的兜底；910B3/B4 是 40 个 AIV，910B1/B2 是 48。
    // 用 SyncAll 的算子上，核数超过实际值会**死锁**（文档《SyncAll》约束）。
    uint32_t aivCores = s9::QueryAivCores(context);
    /* `S9_CORES`：**只为判别实验**——把核数往下压，看吞吐是不是跟着线性掉。
     * 掉 ⇒ 现在的 913 GB/s 是**发射侧**（核数）限制，加 AIC 有戏；
     * 不掉 ⇒ HBM 就是天花板，只能去搬更少的字节。
     * 向上压没有意义（超过实际 AIV 数在用 SyncAll 的算子上会死锁，见上）。 */
    {
        const uint32_t forced = s9::EnvU32("S9_CORES", 0, 0, 64);
        if (forced && forced < aivCores) { aivCores = forced; }
    }
    ConcatTilingData tiling;
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    const int64_t* dimPtr = attrs->GetInt(0);
    if (dimPtr == nullptr) { return ge::GRAPH_FAILED; }

    // **动态输入要用动态接口**：《如何开发动态输入算子》原文
            //「获取动态输入信息的接口，需使用对应的动态接口……
            // GetDynamicInputShape 接口用于获取动态输入的 shape 信息」。
            // 用普通的 GetInputShape(k) 会取到错位的条目 —— 真机上实测输出整体
            // **右移 2 个输入**（4 个输入时 1.0 落在第 8 列而非第 0 列）。
    const auto* desc0 = context->GetDynamicInputDesc(0, 0);
    if (desc0 == nullptr) { return ge::GRAPH_FAILED; }
    uint32_t esz = DtypeBytes(desc0->GetDataType());
    if (esz == 0) { return ge::GRAPH_FAILED; }

    const gert::StorageShape* s0 = context->GetDynamicInputShape(0, 0);
    if (s0 == nullptr) { return ge::GRAPH_FAILED; }
    int64_t rank = static_cast<int64_t>(s0->GetStorageShape().GetDimNum());
    int64_t dim = *dimPtr;
    if (dim < 0) { dim += rank; }                       // torch 允许负 dim
    // **这道校验不能少。** 原来它在 `CalcConcat` 里（`dim < 0 || dim >= rank`），
    // 改走 `CalcConcatLens` 之后那边看不到 dim 了 —— 校验跟着一起丢了。
    // 丢了的后果：下面 `for (i = 0; i < dim; ++i) sh.GetDim(i)` 会读出 shape 边界。
    if (dim < 0 || dim >= rank) { return ge::GRAPH_FAILED; }

    // **路数要从节点上取，不能靠"探到 nullptr 为止"。**
    //
    // 原来这里写的是 `for (k = 0; k < CONCAT_MAX_IN; ++k) { ... if null break; ++numIn; }`,
    // 而 CONCAT_MAX_IN 当时是 64。赛题的输入由
    // `generate_random_split_sizes(total_len, max_step)` 随机切出，
    // 路数约 `2·total_len/max_step` —— total_len=4096、max_step=64 就是 **123 路**。
    // 超过 64 时这个循环**只读前 64 路、numIn 停在 64、而且完全不知道还有更多**，
    // 于是 nOut 只算了前 64 路的列数，输出后半段留着 `at::empty` 的垃圾。
    // **提交时 5 个 case 全挂就是这么来的。**
    //
    // `x` 是本算子唯一的输入（`dim` 是属性），所以节点的输入个数就是路数。
    uint32_t numIn = static_cast<uint32_t>(context->GetComputeNodeInputNum());
    if (numIn == 0 || numIn > s9::CONCAT_MAX_IN) { return ge::GRAPH_FAILED; }

    // 拼接轴之前各维之积（各路必须一致，取第 0 路）。
    // **名字不能叫 rows** —— 下面 `uint32_t rows = ct.rows` 已经占了这个名字。
    uint32_t rowsPre = 1;
    {
        const auto& sh0 = s0->GetStorageShape();
        for (int64_t i = 0; i < dim; ++i) {
            if (!s9::MulDim(rowsPre, sh0.GetDim(static_cast<size_t>(i)))) { return ge::GRAPH_FAILED; }
        }
    }
    // 各路在拼接轴之后的元素数。**不再摊 shapes/ranks 大数组** ——
    // 1024 路时那是 128 KB 的栈，直接爆。
    std::vector<uint32_t> lens(numIn, 0);
    for (uint32_t k = 0; k < numIn; ++k) {
        const gert::StorageShape* sk = context->GetDynamicInputShape(0, k);
        if (sk == nullptr) { return ge::GRAPH_FAILED; }
        const auto& sh = sk->GetStorageShape();
        uint32_t n = 1;
        for (size_t i = static_cast<size_t>(dim); i < sh.GetDimNum(); ++i) {
            if (!s9::MulDim(n, sh.GetDim(i))) { return ge::GRAPH_FAILED; }
        }
        lens[k] = n;
    }

    s9::ConcatTiling ct{};
    // 每核至少搬多少字节 —— 用 S9_MINB 覆盖，便于扫核数找瓶颈（默认 1024）
    uint32_t minBurst = s9::EnvU32("S9_MINB", s9::MIN_BURST, 32, INT_MAX);
    // 按输入切核的门限 —— **没有实测依据**，用 S9_INSPLIT 扫（同 S9_MINB 的做法）。
    // 设 0 可以完全关掉这条路，回到已验证的按行切。
    uint32_t inSplit = s9::EnvU32("S9_INSPLIT", s9::IN_SPLIT_BYTES, 0, INT_MAX);
    // **向量拼装默认开**。原先默认关（注释写「实测更慢」），那是在旧实现下测的：
    // 那时暂存区用统一行距，读要每行一块，读放大 1.91× 没治好，所以白搭上 Gather 的钱。
    // 改成打包读之后，910B4 上 9 档实测：c 档 92.73 → 32.61（2.8×）、
    // Case5 那类形状 359.62 → 177.14（2.0×），其余档位持平，没有一档变差。
    // S9_GATHER=0 可以关掉做 A/B。
    bool gatherOn = s9::EnvFlag("S9_GATHER", true);
    // 列切核数的基准：默认按平均一路（见 s9_tiling_calc.h）。S9_COLBASIS=0 切回按最短一路。
    /* **不过滤区间**：越界值本身有语义（>=1 都表示"按平均一路"那一档），
     * 用 EnvU32 过滤会改行为。 */
    uint32_t colBasis = s9::COL_BASIS_AVG;
    const char* cb = getenv("S9_COLBASIS");
    if (cb != nullptr) { colBasis = (uint32_t)atoi(cb); }
    // 按列切的每核载荷（**字节**）。S9_COLEL 现在也是字节，而且会经 TilingData
    // 传到 kernel —— 以前它只改 host 那一半，扫出来的是不自洽的配置。
    uint32_t colBytes = s9::EnvU32("S9_COLEL", s9::COL_BURST_BYTES, 1, INT_MAX);
    // 列带的最小带宽（字节）。设很大 ⇒ bands 必为 1，回到分带前的行为，便于 A/B。
    uint32_t bandMin = s9::EnvU32("S9_BANDMIN", s9::BAND_MIN_BYTES, 32, INT_MAX);
    // 列带宽度的取整粒度（字节）。512 那个粗粒度自己制造了负载不均（见 s9_tiling_calc.h），
    // 现在按元素经 TilingData 下发，kernel 不再自带常量，所以这个旋钮是端到端有效的。
    uint32_t bandGrain = s9::EnvU32("S9_BANDGRAIN", s9::BAND_GRAIN_BYTES, 1, INT_MAX);
    // 按输入切核的每路写块下限（字节）。低于它且拼装路接得住，就让给拼装路。
    // 开拼装区双缓冲的路数上限（见 ABUF2_MAX_IN）。S9_ABUF2MAX 可扫。
    uint32_t abuf2Max = s9::EnvU32("S9_ABUF2MAX", s9::ABUF2_MAX_IN, 0, INT_MAX);
    uint32_t minWrite = s9::EnvU32("S9_MINW", s9::IN_SPLIT_MIN_WRITE, 0, INT_MAX);
    // 拼装路的 nOut 上限。分带之后偏移表只有 O(bandCols) 而不是 O(nOut)，
    // 8192 这个上限的原始理由（表是 O(nOut) 的一次性开销）已经不成立，用 S9_GMAXOUT 扫。
    uint32_t gMaxOut = s9::EnvU32("S9_GMAXOUT", s9::GATHER_MAX_OUT, 0, INT_MAX);
    // int8 借道 half 走拼装路。**默认开**（见 s9_tiling_calc.h 的实测），S9_I8G=0 关掉做 A/B。
    bool i8Gather = s9::EnvFlag("S9_I8G", true);
    // 列带边界：**默认 0（旧的「统一带宽＋末带吃剩下」）—— 均衡切分被配对 A/B 证伪。**
    //   G16 4096×16384 f16 246 路，同一份包交替 4 轮：B/A = 0.985 0.989 0.987 0.986
    //   （B 组是 S9_BAL=0 即旧行为）⇒ 均衡切分**慢 1.35%**。
    //
    // **为什么"修好负载不均"反而没用**：一条带的核时 ∝ 最宽的那条带，而
    //   最宽带 = ⌈G/bands⌉·grain
    // 在两种切法下**是同一个数**（旧法前 bands−1 条都取它，均衡法也有几条取它）。
    // 均衡只是把末带那点零头摊平，**关键路径一点没动**。
    // 更根本的是：这个算子是**带宽受限**的，空转的核并不浪费带宽 ——
    // 忙核会把它让出来的带宽用掉。所以 msprof 上 aiv_time 393.69 / dur 465.72
    // 那 15% 的"不忙"不是浪费的机器，是在等内存。
    //
    // 真正能动关键路径的是**选一个能整除 G 的带数**（imbalance = ⌈G/B⌉·B/G）：
    // G16 的 G=64、blockDim=40 ⇒ B=8 或 4 才是 1.00，B=10 是 1.094。
    // 但实测 B=8 只比 B=10 快 0.6% —— 与"带宽受限、不均很便宜"一致。
    // 旋钮留着：换到 blockDim=48 的评测机上带数选法不同，值得再验。
    uint32_t bandBal = s9::EnvU32("S9_BAL", 0, 0, 1);
    /* ═══ 边界路裁剪（`S9_TRIM`）═══
     *
     * 「一条带里与它相交的路一律整行入暂存」会让**被带切开的那 ≤2 条路**把带外的
     * 列也搬进来。裁剪让这两路只搬相交的那一段（见 ConcatBandStagedBytes）。
     *
     *   1 ＝ 只裁 kernel 的**读** —— UB 账仍按旧口径算 ⇒ **tileG 一个字不变**，
     *        量到的差分只来自"少读字节"；
     *   2 ＝ 读与 UB 账一起裁 —— 紧凑槽省下的 UB 会把 tileG 顶大，**必须单独量**
     *        （tileG 有实测最优点，混在一起就分不清是谁的功劳）。
     *
     * **默认 1（只裁读）** —— 2026-09-12 上板量过三个形状（910B4 / CANN 8.5.0）：
     *     512,2048 int32 248 路   B/A = 1.080（慢 8.0%）
     *     4096,4096  f16  123 路   B/A = 1.079（慢 7.9%）
     *     4096,16384 f16  246 路   B/A = 1.010（慢 1.0%）
     * **三个形状都是负的**，机理是"少读的那点字节抵不上跨步 DMA 的开销"。
     * 仍然开成默认，是因为：① 评测机是 910B1/48 核、形状未知，**唯一的 48 核信号
     * 只有提交**；② 裁剪在窄路族上**根本不会生效**（`ConcatRouteTrimmed` 的护栏，
     * Case1 那种 9 路、以及 int8 248 路都是 no-op），所以下行风险是有界的；
     * ③ 模式 1 不动 UB 账与 tileG，与上一版只差这一个变量，便于归因。
     * **想退回旧行为：`S9_TRIM=0`。** */
    /* **v24 提交（TRIM=1）实测回归，撤回默认值。**
     * 官方评测分 case：v23(TRIM=0) → v24(TRIM=1) 是
     *   C1 5.46→5.50 · C2 30.87→31.481 · C3 18.30→18.50 · C4 104.43→104.392 ·
     *   **C5 243.32→249.155（+5.84 µs / +2.4%）** · 合计 402.38→409.028（+1.65%）。
     * 回归几乎全在最大的那个 case，和本机 A/B 的机理一致：裁剪少了读字节、
     * 但把"一条大连续读"切成了 cnt 条小块 + 32B 补齐（官方《尽量一次搬运较大的
     * 数据块》优先级高：单次搬运 ≥16 KB 才接近带宽峰值）。
     * 代码与验证都留着，`S9_TRIM=1/2` 仍可打开。 */
    uint32_t trimOn = s9::EnvU32("S9_TRIM", 0, 0, 2);
    // 偏移表只留 2 行：**进 UB 账**，所以必须经 knobs 进 CalcConcatLens，
    // 不能像 directOn 那样只塞给 kernel。
    uint32_t tblKOn = s9::EnvU32("S9_TBLK", 0, 0, 64);
    const uint32_t directOn = s9::EnvU32("S9_DIRECT", 0, 0, 2);
    /* 二维分核的每列块字节下限（0 ＝ 关）。**进 CalcConcatLens**，因为它改 blockDim。 */
    uint32_t gridBurst = s9::EnvU32("S9_GRID", 0, 0, INT_MAX);
    uint32_t hoistOn = s9::EnvU32("S9_HOIST", 0, 0, 1);
    /* 输出线性切分（strategy 5，`rows == 1` 专用）。**默认开** ——
     * 910B4 实测 2.59×~3.21×，且只覆盖 rows==1，其余形状一个字节不变。
     * `S9_LIN=0` 关掉做配对 A/B。 */
    bool linearOn = s9::EnvFlag("S9_LIN", true);
    s9::ConcatKnobs kn;
    kn.minBurst = minBurst;                                 kn.gatherOn = gatherOn;
    kn.maxCores = aivCores;   kn.inSplitBytes = inSplit;    kn.colBasis = colBasis;
    kn.colBurstBytes = colBytes; kn.bandMinBytes = bandMin; kn.i8Gather = i8Gather;
    kn.gatherMaxOut = gMaxOut;   kn.bandBal = bandBal;      kn.bandGrainBytes = bandGrain;
    kn.inSplitMinWrite = minWrite; kn.abuf2MaxIn = abuf2Max; kn.trimOn = trimOn;
    kn.tblK = tblKOn;
    // DIRECT/TBLK 沿用原布局；保证可行性、布局表与 kernel 采用同一口径。
    kn.denseOn = directOn == 0 && tblKOn == 0 ? s9::EnvU32("S9_DENSE", 1, 0, 2) : 0;
    kn.gridBurstBytes = gridBurst;
    kn.s3MinTileG = s9::EnvU32("S9_S3MINTG", 0, 0, 4095);
    kn.s3MinTileGMaxBands = s9::EnvU32("S9_S3MINTGB", 4, 1, 64);
    kn.asm2On = s9::EnvFlag("S9_ASM2", true);
    kn.s2YieldBands = s9::EnvU32("S9_S2YIELD", 2, 0, 64);
    kn.s4YieldBands = s9::EnvU32("S9_S4YIELD", 3, 0, 64);
    kn.linearOn = linearOn;
    /* ═══ 只有一路非空、且行很窄 ⇒ 整块拷贝（`S9_COPY1`）═══
     *
     * harness 的切分允许 0 长度段，`shape[dim]` 小时常常只剩一路非空（规格抽样 3000 例里
     * 去掉 dim=0 还有 65 例，56 例落在 strategy 2）。此时输出**逐字节等于**那一路输入，
     * 而 host 仍按「rows 行 × 每行 nOut 列」去拼：`2,297,9612,1 int32 dim=3` 每行 4 B、
     * 570 万行，写被切成 4 B 的碎块 —— 21.8 MB 要 **4988 µs（9 GB/s）**，mte3 占 98%。
     *
     * 把行并进列（`rows→1、lens[k]→rows·lens[k]`）在只有一路非空时是**精确**的：
     * 那一路的 offOut 恒为 0，输入与输出都是 rows·n 个连续元素。rows==1 之后
     * 就交给 strategy 5（输出线性切分）。 */
    /* **只对窄行做**（`S9_COPY1` ＝ 一行字节数的上限，默认 128；0 ＝ 关）。910B1 行宽扫描
     *（64 MB、三种 dtype 一致）：行 ≤ 128 B 时整块拷贝快 1.54×~674×，≥ 256 B 时慢 11%~19%
     *（strategy 5 读写零重叠，宽行上不如按行拼）。 */
    const uint32_t copy1MaxRowBytes = s9::EnvU32("S9_COPY1", 128, 0, 0x7FFFFFFF);
    if (copy1MaxRowBytes != 0 && rowsPre > 1) {
        uint32_t nz = 0, kz = 0;
        for (uint32_t k = 0; k < numIn; ++k) { if (lens[k] != 0) { ++nz; kz = k; } }
        if (nz == 1 && (uint64_t)lens[kz] * esz <= copy1MaxRowBytes &&
            (uint64_t)rowsPre * lens[kz] <= 0x7FFFFFFFull) {
            lens[kz] *= rowsPre;
            rowsPre = 1;
        }
    }
    if (!s9::CalcConcatLens(rowsPre, lens.data(), numIn, esz, ct, kn)) {
        return ge::GRAPH_FAILED;
    }

    // 切核策略可用 S9_STRAT 强制（0=按行 1=按列 2=DMA拼装 3=向量拼装 5=输出线性切分），
    // 用来实测而不是靠推理
    const char* st = getenv("S9_STRAT");
    if (st != nullptr) {
        uint32_t want = (uint32_t)atoi(st);
        // **拼装路（2/3）只能"请求"，不能硬闯** —— 它们各有前提
        //（2 要 32B 对齐落点；3 要 16/32 位 dtype、nOut 上限、UB 容量）。
        // 绕过前提会直接踩出 vector core exception（507035），所以不满足时保持原值，
        // 只有降级到 0/1 才无条件生效。
        if (want <= 1 || want == ct.strategy) { ct.strategy = want; }
    }
    /* ═══ 已删除：strategy 5「按（路 × 行段）切工作单元 + 大块 DMA」═══
     *
     * 2026-09-12 加、2026-09-13 **代码整体删除**（负结果留档，见
     * `docs/submission-env.md` §④：`4096,16384|f16|-1|128` 246 路的配对 A/B，
     * A 无旋钮 463.58 µs ／ B `S9_S5=1` 2006.43 µs ⇒ **B/A = 4.328，慢 4.33×**）。
     * 它把每条读从 ≈5 KB 抬到大块，但 32B 补齐把 UB 侧搬运放大，
     * 且丢掉软件流水，净效果为负。
     *
     * 所以 `S9_S5` / `S9_CHUNK` 两个开关、TilingData 的 `s5Chunk` 字段，
     * 以及 kernel 的 `S9RunBigTile` / `S9BigEvIn/Out` 一并删掉 ——
     * 留着的唯一价值是"换机换形状再验一次"，但那要重写，不如从记录出发。
     * **策略号回到连续取值：0/1/2/3/4。** */
    uint32_t rows = ct.rows, nOut = ct.nOut;
    // **strategy 不在这里拷贝。** 它还要被 S9_STRAT / S9_BLK 与最终校验改，
    // 早拷一次就等于把旧值下发（设计文档点出的第 2 类问题）。
    uint32_t blockDim = ct.blockDim, ubBudget = ct.ubBudget;
    // 核数与 UB 预算的实测开关（S9_BLK / S9_UBX），只用于扫描定位瓶颈
    const char* bk = getenv("S9_BLK");
    if (bk != nullptr) {
        int v = atoi(bk);
        if (v >= 1 && v <= 48) {
            blockDim = (uint32_t)v;
            // **改了核数就必须重挑带数。** 带数是按原来的 blockDim 选的、而且
            // kernel 只接受整除核数的带数（不整除时它自己退回 bands=1），
            // 于是 UB 账按"分带"算、缓冲按"不分带"开 —— 直接 UB 越界。
            // 实测：G16 那档 S9_BLK=32 就这么挂的（要 246144 字节，上限 188416），
            // 而扫描表上只会显示成 0.00 µs，看起来像"慢"。
            // **收敛动作统一交给下面的 CheckConcatFinal**（它同时管带数收敛、
            // 策略降级与 UB 实账复核），这里只改核数。
        }
    }
    ubBudget = s9::EnvU32("S9_UBX", ubBudget, 1024, INT_MAX);
    tiling.set_nIn(ct.nIn);   // offOut 不传，kernel 自己累加（见 concat_tiling.h）

    tiling.set_rows(rows);
    tiling.set_nOut(nOut);
    tiling.set_numIn(ct.numIn);
    tiling.set_strategy(ct.strategy);      // 最终值在下面的 CheckConcatFinal 之后重设
    tiling.set_blockDim(blockDim);
    // 输入的 L2 模式（0=DISABLE 1=NORMAL 4=PERSISTENT），可用 S9_L2 覆盖。
    //
    // **默认 NORMAL —— 这里推翻了我此前的决定。**
    // 原先一律 DISABLE，依据是《设置合理的L2 CacheMode》「对于只访问一次的
    // Global Memory 数据设置其访问状态为不进入 L2 Cache」。**我把那条用错了地方**：
    // 它针对的是「本算子只读一次、但 L2 里还有别人要复用的数据」——
    // 目的是别把有用的数据冲掉。而 Concat 的输入**就是**要消费的数据，
    // 关掉 L2 只是把每次读都赶去 HBM，白白丢掉 L2 的读带宽。
    //
    // 真机实测（910B3，与官方 aclnnCat 同场）：
    //     小 shape（赛题用例 128×256 fp16）  DISABLE 11.10 us → NORMAL  8.44 us
    //     大 shape（2024×3000 fp16，36 MB） DISABLE 37.74 us → NORMAL 26.62 us
    // PERSISTENT 在小 shape 上 8.90 us，不如 NORMAL。
    // **L2 模式扫过，当前这组（输入/输出都 NORMAL）就是最优**，别再改：
    // 910B4 上 p4 档 in/out 都 NORMAL 117.31 µs · 输出改 DISABLE **142.68** ·
    // 输入改 PERSISTENT 119.46 · 两边都 DISABLE **757.07**（6.5×）。
    // dim=0 那档同样（84.25 vs 104.81）。输出的流式写**本身就吃 L2 的好处**
    // （写合并），把它赶出 L2 是净亏。
    // 平台参数：910B1 的 l2_size=192 MB、910B4=96 MB，两者 l2_rate=110 而 ddr_rate=32
    // —— L2 每核带宽是 DDR 的 3.4 倍，所以"别把 L2 让出来"这条在评测机上只会更成立。
    /* **v25 的单变量实验**：输入走 L2 旁路（`CACHE_MODE_DISABLE`），输出仍 NORMAL。
     *
     * 想验证的假设：评测机是 910B1（L2 **192 MB**，本地 B4 只有 96 MB），
     * 如果输出能留在 L2 里（写回推迟到 kernel 结束之后），HBM 就只需要承担"读"，
     * 流量近乎砍半 —— 这是本机复现不出来、却可能是"榜首快 1.93×"的唯一硬件级解释。
     *
     * **本机已知的负面证据（要如实记着）**：B4 上输入 DISABLE 在
     *   4096×4096（输出 32 MiB，本就装得进 96 MB L2）是 **112.61 → 146.27 µs（−29%）**，
     *   在 4096×16384（输出 128 MiB，装不下）是大档中性（464.53 → 461.08，
     *   但管道反而变好：mte3 96→42、mte2 367→328）。
     * 也就是说"输出装得下 L2"这一个条件**在 B4 上并没有让它变快** ——
     * 所以这个假设的可信度不高，这一次提交就是去问评测机要一个答案。
     * 若评测机也是负的，立即改回 1 并重交。 */
    /* ═══ 2026-09-13：**v28 的单变量实验在放回配对 A/B 之后被否掉，默认留在 0** ═══
     *
     * 经过：v27 的 A/A 对照证明**评测数字本身有 ±20% 的批次噪声**
     *（同一份源码重交：353.3775 vs 418.992，五个 case 同向等比例放大 1.16~1.29）。
     * 于是"v25 用 l2Mode=0 得到 415.86、v23 用 l2Mode=1 得到 402.38"这类
     * 按提交次数做的比较**全部失去检出力**，不能再用它们定这个默认值。
     *
     * 改回**同机交替的配对 A/B**（`ab_knob.sh`，同一份包、R=4、看比值中位数），
     * 在 8.3.RC1 / 910B4 上重测（括号里是四轮的原始值，离散度 <1%）：
     *
     *   | 旋钮 | 形状 | 默认(0) | S9_L2=1 | B/A |
     *   |---|---|---|---|---|
     *   | 输入 | 4096,4096 f16  | 101.70 (101.9 101.5 101.9 101.5) | 110.56 (110.9 112.2 110.2 110.2) | **1.087** |
     *   | 输入 | 4096,16384 f16 | 411.17 (411.9 412.5 408.3 410.4) | 457.60 (456.4 458.0 458.7 457.2) | **1.113** |
     *   | 输入 | 1024,1024 f32  |  19.77 |  19.79 | 1.001 |
     *   | 输出 | 4096,4096 f16  | 101.89 | 132.63 | 1.302 |
     *   | 输出 | 4096,16384 f16 | 410.23 | 503.12 | 1.226 |
     *
     * ⇒ **输入 DISABLE（默认 0）在两个大档上快 8.7%~11.3%**，输出 NORMAL（默认 1）
     * 快 23%~30%。**当前这两个默认都是对的，定住不动。**
     *
     * 这条同时**推翻了本文件下面 §⑤ 记的"S9_L2=0 在 4096×4096 是 −29%"** ——
     * 那是一次**没有配对**的单发测量（112.24 vs 144.61），落在该机的批次噪声里。
     * 机理上也说得通：输入是**流式读一次**，让它们进 L2 只会在里面过一遍就冲掉，
     * 白白占住缓冲；输出要写一次、马上可能被读，留在 L2 才有意义。 */
    /* ═══ 2026-09-14：默认从 0（DISABLE）改成 1（NORMAL）—— **前提不成立了** ═══
     *
     * 上面那张配对 A/B 表把 `l2Mode=0` 定成默认，依据是「输入 DISABLE 在两个大档上
     * 快 8.7~11.3%」。**那张表的隐含前提是"工作集装不下 L2"** ——
     * 它在 910B4 上量的，而 910B4 的 L2 只有 **96 MB**，测的两档
     * （4096×4096 f16 ＝ 32 MB 张量但读写 64 MB、4096×16384 f16 ＝ 134 MB）
     * 其中大的那个根本装不下。装不下时让输入进 L2 确实只是"过一遍就被冲掉"，
     * 所以 DISABLE 赢 —— 机理没错，**但那是那台机的机理**。
     *
     * **评测机是 910B1，L2 ＝ 192 MB**（`A2A3.ini` 的 `SoC_version=Ascend910B1`、
     * `l2_size=201326592`，2026-09-13 查实）。**是本机的两倍。**
     * 而今天在 910B4 上量到：同体积 `torch.copy_` 在 **L2 驻留时 993 GB/s、
     * 装不下时只有 575 GB/s —— 1.73×**。
     *
     * ⇒ 若评测的 C5（217 µs，占总分 62%）工作集在 192 MB 里装得下，
     * "输入旁路 L2"就恰好是**反的**。而**这件事本机永远验不出来**（L2 只有一半）。
     *
     * 这正是 `docs` 里记的那条纪律：**负面结论要连前提一起记**。
     * 那条 A/B 的前提在评测机上可能不成立，所以它不能外推。
     *
     * **这一版就是去问评测机要这个答案。** 若 C5 那一档没反应，说明它是 HBM 受限的，
     * 改回 0（`S9_L2=0` 或把下面这个默认值改回去）。 */
    /* ═══ 2026-09-14 **评测机答了：否。改回 0。** ═══
     *
     * v38 交了四次：416.916 / 396.6785 / 396.23 / 396.107。后三次的离散只有 **0.15%**，
     * 远小于慢簇内部的 4.9% —— 它们不是"慢簇抽样"，是一个稳定的新峰。
     *
     * 判读靠的是**逐档**而不是总分（"一次提交只带一个变量"那条纪律的配套）：
     *
     *   |          | C1 | C2 | C3 | C4 | C5 | 合计 |
     *   |---|---|---|---|---|---|---|
     *   | v37 快簇 | 5.690 | 25.770 | 16.201 | 85.942 | 217.855 | 351.46 |
     *   | v38 #3   | **5.46** | 31.12 | **16.25** | 102.57 | 240.83 | 396.23 |
     *   | v38 #4   | **5.40** | 32.47 | **16.83** |  99.86 | 241.55 | 396.11 |
     *
     * **C1 比 v37 还快 4~5%、C3 持平 ⇒ 那两次运行机器处在快状态。** 同一次运行里
     * C2 +21~26%、C4 +16~19%、C5 +10.5% —— 排除了"运气差"，是这个旋钮本身的代价。
     *
     * 所以上面那张 910B4 配对 A/B 表的结论**在 192 MB L2 的 910B1 上同样成立**，
     * 它的前提（工作集装不下 L2）比我想的更稳：**输入旁路 L2 是对的**。
     * 机理上最讲得通的一条：输出侧是 NORMAL（实测值 23~30%，见下），
     * 输入若也进 L2 就会**把输出的行挤掉** —— L2 是被争抢的资源，而输出赢。 */
    uint32_t l2Mode = 0;
    const char* l2 = getenv("S9_L2");
    if (l2 != nullptr) { l2Mode = (uint32_t)atoi(l2); }
    tiling.set_l2Mode(l2Mode);
    // **输出侧 NORMAL —— 这里我又一次把那条 L2 文档用错了，实测推翻。**
    // 《设置合理的L2 CacheMode》说「只访问一次的 Global Memory 数据设置其访问状态为
    // 不进入 L2 Cache」。输出确实只写一次，看起来正该 DISABLE，我也这么试了。真机不同意：
    //     big       DISABLE 22.84  NORMAL **19.82**  （DISABLE 慢 15%）
    //     shortbig  DISABLE 66.78  NORMAL **62.38**  PERSISTENT 62.52
    //     huge      DISABLE 83.51  NORMAL   82.25    PERSISTENT **78.76**
    // 原因大概率是**写不满整行**：本算子的写是 blockLen=nIn·esz 的跨步块
    //（大 shape 上 2000 字节，跨 6000 字节写一次），绕开 L2 后不满 cacheline 的部分
    // 要在 HBM 上做读-改-写；留在 L2 里反而能合并。
    // **这条文档我已经用错三次**（先给输入设 DISABLE、再给输出设 DISABLE），
    // 记在这里：它针对的是「本算子只读一次、而 L2 里另有别人要复用的数据」，
    // 不是「本算子只碰一次的数据」。
    uint32_t l2ModeOut = 1;
    const char* l2o = getenv("S9_L2O");
    if (l2o != nullptr) { l2ModeOut = (uint32_t)atoi(l2o); }
    tiling.set_l2ModeOut(l2ModeOut);
    // 软件流水的前瞻开关：0=按策略自动（默认，只在 strategy 4 上开）、
    // 1=强制开、2=强制关。实测分野很清楚（见 kernel 里 Emit 的注释），
    // 留 S9_PIPE 是为了换了形状/换了卡之后能重新量，而不是靠推理。
    uint32_t pipeMode = s9::EnvU32("S9_PIPE", 0, 0, 2);
    tiling.set_pipeMode(pipeMode);
    uint32_t gBuf = s9::EnvU32("S9_GBUF", 2, 1, 8);
    tiling.set_gBuf(gBuf);
    // 列带数可用 S9_BANDS 强制（只在拼装路生效，且必须整除 blockDim，
    // 否则网格填不满、有核空转）。用于上板 A/B，不改默认选法。
    uint32_t bands = ct.bands ? ct.bands : 1;
    const char* bs = getenv("S9_BANDS");
    if (bs != nullptr) {
        int v = atoi(bs);
        if (v >= 1 && (uint32_t)v <= blockDim && blockDim % (uint32_t)v == 0) {
            bands = (uint32_t)v;
        }
    }
    if (ct.strategy != 3) { bands = 1; }
    tiling.set_bands(bands);
    tiling.set_maxIn(ct.maxIn);
    tiling.set_colBurstEl(ct.colBurstEl ? ct.colBurstEl : 1);
    tiling.set_bandBal(ct.bandBal);
    // 默认 1（照旧调）。源码上 A2 的 SetL2CacheHint(NORMAL) 是空操作，但文档里记着
    // "删掉是 8~16 倍回退"，两者冲突，只能上板判 —— S9_HINT=0 跳过。
    uint32_t hintMode = s9::EnvU32("S9_HINT", 1, 0, 1);
    tiling.set_hintMode(hintMode);
    tiling.set_bandGrain(ct.bandGrain ? ct.bandGrain : 1);
    // 暂存余量的路数。**默认按全部 numIn 留 —— 精确值反而更慢，这是量出来的。**
    //
    // s9::MaxInputsPerBand 能精确数出「任一条带里最多几路」（248 路时约 33），
    // 按它留能省下 15872−2112 B 的 UB。但那块 UB 释放出来只会让 tileG 变大，
    // 而 tileG **不是越大越好**。同一份包、只切这个开关，重复两轮：
    //     G16 4096×16384 f16 246路   精确 516.24 / 522.32   全路数 **502.87 / 503.10**
    //     I2  2048×2048  int8 248路  精确 **39.83 / 39.71**  全路数 40.99 / 40.87
    //     d   2048×2048  f16  58路   两者持平（29.4~29.7）
    // G16 是 Case5 那一族（占总分 61%），所以默认取对它好的那一边。
    //
    // ⇒ **真正该调的是 tileG 本身**（S9_TILEG 是按「每 tile 拼装区字节数」给的上限），
    //   余量只是间接把它顶大了。tileG 的极值扫过一轮但跨版本不复现，先留着。
    uint32_t maxInB = ct.numIn;
    const char* mib = getenv("S9_MAXINB");
    if (mib != nullptr) {
        int v = atoi(mib);
        if (v == 0) { maxInB = ct.maxInBand ? ct.maxInBand : ct.numIn; }   // 0 ＝ 用精确值
        else if (v >= 1) { maxInB = (uint32_t)v; }
    }
    if (maxInB > ct.numIn) { maxInB = ct.numIn; }
    tiling.set_maxInBand(maxInB);
    // 拼装路每 tile 的**拼装区字节上限**（0 ＝ 不限）。S9_TILEG 扫它的最优点。
    uint32_t tileGMax = s9::EnvU32("S9_TILEG", 0, 0, INT_MAX);
    tiling.set_tileGMax(tileGMax);
    // **拼装区的缓冲块数由 tiling 算术侧定**（判据是路数，见 ABUF2_MAX_IN 的实测表），
    // 因为它进 UB 账，必须和 GatherFitsBands 用同一个值。S9_ABUF 仍可覆盖以便扫描 ——
    // 但覆盖成比算术侧**更大**的值会让 UB 账偏小，只适合在本机做 A/B，别当默认。
    uint32_t aBuf = s9::EnvU32("S9_ABUF", ct.aBuf ? ct.aBuf : 1, 1, 3);
    tiling.set_aBuf(aBuf);
    tiling.set_ubBudget(ubBudget);

    // ═══ 最终配置校验：**所有 S9_* 覆盖之后、下发之前** ═══
    //
    // 上面那一串开关各改各的（核数、带数、buffer 数、UB 预算、余量路数），必须在
    // 最后统一过一遍：**要么明确失败（带诊断），要么安全降级**。检查项与负例都在
    // `s9::CheckConcatFinal` 里（纯算术、可单测，见 test/test_concat_layout.cpp）：
    //   · 核数超过平台查询值 —— 对用 SyncAll 的算子是死锁，直接拒；
    //   · 暂存/拼装块数超过事件号上限（kernel 的 EvB 只到 3）；
    //   · 带数不整除核数 —— kernel 会悄悄退回 bands=1，而 UB 账按分带算；
    //   · UB 预算 × 块数 ＋ 序幕/指针暂存 超过单核 UB —— InitBuffer 直接踩别人；
    //   · 拼装路的 UB 实账放不下（含逐块对齐与序幕）—— 降级按行切，并留痕。
    uint32_t strategy = ct.strategy;
    {
        const s9::ConcatFinal fin = s9::CheckConcatFinal(ct, esz, blockDim, bands, gBuf, aBuf,
                                                        maxInB, ubBudget, tileGMax, aivCores);
        if (!fin.ok) {
            fprintf(stderr, "[S9] 非法配置（%s）：strategy=%u blockDim=%u bands=%u gBuf=%u "
                            "aBuf=%u maxInBand=%u ub=%u tileGMax=%u numIn=%u esz=%u\n",
                    fin.why, ct.strategy, blockDim, bands, gBuf, aBuf, maxInB, ubBudget,
                    tileGMax, ct.numIn, (uint32_t)esz);
            return ge::GRAPH_FAILED;
        }
        if (fin.changed) {
            // **降级/收敛必须留痕**：否则"这一版为什么慢了"永远查不出来
            //（本文件里 S9_MAXINB 那段记录的就是这类事后追查）。
            fprintf(stderr, "[S9] 配置收敛：%s（strategy %u→%u，bands %u→%u，gBuf %u→%u，"
                            "aBuf %u→%u）\n",
                    fin.why, ct.strategy, fin.strategy, bands, fin.bands, gBuf, fin.gBuf,
                    aBuf, fin.aBuf);
        }
        strategy = fin.strategy;
        bands    = fin.bands;
        gBuf     = fin.gBuf;
        aBuf     = fin.aBuf;
        tiling.set_strategy(strategy);
        tiling.set_bands(strategy == 3 ? bands : 1);   // 非拼装路的 kernel 会用 bands=1
        tiling.set_gBuf(gBuf);
        tiling.set_aBuf(aBuf);
    }

    /* ═══ 每带执行计划（strategy 3 专用；**默认开**，S9_BANDPLAN=0 关）═══
     *
     * 每个核原先要先扫一遍全部 numIn 路，才知道"哪些路与我这条带相交"
     *（kernel 的 iLo/iHi 与 stBase 都是这么来的）。这些量只依赖 tiling，
     * 在 host 上算一次下发即可 —— kernel 只读本带那一段。
     *
     * **2026-09-11 真机 A/B 之后改成默认开。** msprof 显示瓶颈就是**标量流水**
     *（128×256 那档 scalar 65%、2048×2048 int8 62%、512×2048 int32 59%），
     * 而这场"每核扫一遍全部路"正是标量时间的大头。同一份包交替 3 轮的配对结果：
     *     2048,2048 int8 248 路   43.76 → 33.60 µs   B/A = 0.768
     *     512,2048  int32 248 路  34.00 → 23.35 µs   B/A = 0.687
     *     4096,4096 f16  123 路  118.63 → 110.23 µs  B/A = 0.929
     *     4096,16384 f16 246 路  466.13 → 458.48 µs  B/A = 0.984
     * 组内抖动 ±0.5%，所以这不是噪声。代价是 TilingData 多 768 字节、
     * kernel 侧多读 3 个 GM 标量 —— 与省下的标量时间完全不成比例。
     * 前提全在计划函数里，越界或不合法就明确失败（退回全扫描并打日志）。 */
    uint32_t planOn = 0;
    const bool wantPlan = s9::EnvFlag("S9_BANDPLAN", true);   // 默认开
    if (wantPlan && strategy == 3) {
        s9::ConcatBandPlan bp{};
        // 用**最终发布的带数**（校验可能把带数收敛成核数的因数），不是 ct 里那个。
        s9::ConcatTiling ctPlan = ct;
        ctPlan.bands = bands;
        if (s9::CalcConcatBandPlan(ctPlan, bp) && bp.bands == bands) {
            planOn = 1;
            tiling.set_planOn(1);
            tiling.set_bandLo(bp.lo);
            tiling.set_bandHi(bp.hi);
            tiling.set_bandOff(bp.off);
        } else {
            // 计划算不出来（或与最终带数不符）就退回旧路径，并且**说出来** ——
            // 静默退回会让人以为"开了"，量出来的却是旧路径的数。
            fprintf(stderr, "[S9] 每带计划算不出（bands=%u/%u, strategy=%u）"
                            " —— 退回全扫描路径\n", bands, ct.bands, strategy);
        }
    }
    if (planOn == 0) { tiling.set_planOn(0); }

    /* ═══ 逐核布局表：**布局下沉**（`S9_LAYOUT=1`，默认关）═══
     *
     * 每个核原先要自己算「我是哪条带、管哪些列/行」再调 `PlanConcatStage` 反推
     * tileG 与整份 UB 布局。这些量只依赖 tiling，host 上算一次就够。
     *
     * **首版必须与 kernel 现算的逐字段一致**（否则量的就不是"读表更便宜"，
     * 而是"布局变了"）：映射走 `MapConcatCore`、布局走 `PlanConcatStage`，
     * 全是 host 与 kernel 共用的同一份算术；一致性由
     * `test_concat_layout.cpp` 的 `CheckCoreLayoutTable` 对 3000 个核逐字段钉住。
     *
     * 算不出来（任一个核放不下）就**不用这张表**并说出来 —— 静默降级会让
     * "这一版为什么慢"永远查不出来。 */
    uint32_t layOn = 0;
    const bool wantLayout = s9::EnvFlag("S9_LAYOUT", false);
    if (wantLayout && strategy == 3 && blockDim > 0 && blockDim <= s9::S9L_CORE_MAX) {
        s9::ConcatCoreLayoutRow rows[s9::S9L_CORE_MAX];
        const uint32_t nSlack = (maxInB > 0 && maxInB < ct.numIn) ? maxInB : ct.numIn;
        uint32_t wrote = 0;
        const uint32_t bad = s9::CalcConcatCoreLayouts(
            ct, esz, blockDim, gBuf, aBuf, maxInB, nSlack, tileGMax, ct.numIn, ct.maxIn,
            bands, tiling.get_bandGrain(), rows, s9::S9L_CORE_MAX, &wrote);
        if (bad == 0 && wrote == blockDim) {
            uint32_t flat[s9::S9L_CORE_MAX * s9::S9L_CORE_FIELDS];
            for (uint32_t k = 0; k < wrote; ++k) {
                s9::ConcatCoreLayoutToFlat(rows[k], &flat[k * s9::S9L_CORE_FIELDS]);
            }
            tiling.set_layOn(1);
            tiling.set_layRows(flat);
            layOn = 1;
        } else {
            fprintf(stderr, "[S9] 布局表算不出（%u 个核放不下，写了 %u 行，核数 %u）"
                            " —— 这一版退回逐核自算\n", bad, wrote, blockDim);
        }
    }
    if (layOn == 0) { tiling.set_layOn(0); }
    tiling.set_trimOn(ct.trimOn);
    /* 直写路径开关。**不进 ConcatKnobs、不影响任何 tiling 数值** ——
     * 这样 A 组（默认 Gather）与 B 组（`S9_DIRECT=1`）的分核、分带、tileG、
     * UB 账逐字节相同，配对 A/B 只差 kernel 走哪条分支。 */
    /* 0 = 向量拼装（Gather）；1 = 直写但沿用 Gather 算出的 tileG；
     * 2 = 直写并**把整个暂存区用满**（tileG 放大到 nbWant 个槽刚好装下）。
     * 1 与 2 分开是为了把两件事各量一次：
     *   1 → 只去掉拼装（向量开销 ＋ 4→2 次 UB 访问），读块大小不变；
     *   2 → 再把每路读块放大（Gather 要同时装下整条带，直写一次只装一路）。
     * 一次提交只带一个变量那条纪律，在旋钮上也该成立。 */
    tiling.set_directOn(directOn);
    tiling.set_colParts(ct.colParts ? ct.colParts : 1);
    tiling.set_hoistOn(hoistOn);
    tiling.set_oneWriteOn(s9::EnvU32("S9_ONEWRITE", 0, 0, 1));
    tiling.set_denseOn(ct.denseOn);
    tiling.set_tblKOn(tblKOn);
    tiling.set_preRdOn(s9::EnvU32("S9_PRERD", 0, 0, 1));

    // ═══ AIC＋AIV 混合整块搬运（实验开关 S9_AIC，单位百分数）═══
    //
    // 只在 `rows == 1` 时可用 —— dim=0 归一之后每一路在输出里是一整个连续块，
    // 而 AIC 只会搬「长度 32B 对齐的连续块」（GM 两头都可以是任意字节，
    // 见 op_kernel 里 RunConcatMix 的文档依据）。
    //
    // **`S9_AIC` 一旦设置就走混合分支，哪怕值是 0** —— 0 是同一份启动配置下的
    // 对照组（AIC 空转、全部由 AIV 做），用来把「混合核的启动开销」和
    // 「AIC 到底有没有带宽」分开量。不设这个环境变量时行为一个字节都没变。
    uint32_t aicPct = 0;
    bool useMix = false;
    const char* ap = getenv("S9_AIC");
    if (ap != nullptr && ct.rows == 1 && ct.numIn > 0) {
        int v = atoi(ap);
        if (v >= 0 && v <= 100) { aicPct = (uint32_t)v; useMix = true; }
    }
    tiling.set_aicPct(aicPct);
    // 序幕取 TilingData 的方式。**默认 0（逐字段 GM 标量读）—— 整块取被配对 A/B 证伪。**
    //   128×256 fp16 9 路，5 轮交替：B/A = 0.920 0.913 0.923 0.936 0.906，中位 **0.920**
    //   （B 组是 S9_PRE=0 即旧行为）⇒ 整块取**慢 8%**。
    // 机理：标量头的十几个字段是连续的，本来就落在同一条 cache line 上，
    // 十几次标量读远比「一条 DataCopyPad ＋ 一对 MTE2→S 同步」便宜。
    // 「每次 GM 标量读约 0.15 µs」那条拟合是从 nIn[]/GetDataPtr 上量的（地址分散、
    // 与别的工作交错），**不能外推到连续的标量头**。旋钮留着以便换机器再验。
    uint32_t preMode = s9::EnvU32("S9_PRE", 0, 0, 1);
    tiling.set_preMode(preMode);
    if (useMix) {
        // 混合核下 SetBlockDim(N) ＝ N 个 Cube ＋ 2N 个 Vector，所以按 AIC 数给。
        const uint32_t aicCores = s9::QueryAicCores(context);
        uint32_t ng = aicCores ? aicCores : 1;
        const char* nb = getenv("S9_AICBLK");
        if (nb != nullptr) { int v = atoi(nb); if (v >= 1 && (uint32_t)v <= aicCores) { ng = (uint32_t)v; } }
        blockDim = ng;
        // **改了就要重新下发。** 上面的 `tiling.set_blockDim()` 早就跑过了，
        // 不补这一句，TilingData 里留的是按 AIV 数算的旧值。
        tiling.set_blockDim(blockDim);
    }

    context->SetBlockDim(blockDim);
    // 1 / 2 / 4 / 8 ＝ 元素字节宽度；+16 ＝ AIC＋AIV 混合路。
    //
    // **这里曾经有过一档"编译期策略特化"（key ≥ 100，S9_SPECIALIZE=1）。**
    // 2026-09-11 上板：int8 能跑但 B/A = 1.009（没有收益），fp16 形状直接把算子打崩
    //（128×256 与 512×4096 都报 "Inner error … current working operator name is
    // aclnnConcat"）⇒ 整条实验撤掉，键回到元素字节宽度，kernel 侧也恢复运行时分支。
    const uint32_t tilingKey = useMix ? (esz + 16) : esz;
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* ws = context->GetWorkspaceSizes(1);
    // **要留系统 workspace**。设成 0 时真机上观察到输出缓冲开头被 TilingData 覆盖
    //（rows/nOut/numIn/strategy/blockDim/ubBudget 原样出现在 out 的前 24 字节），
    // 而 kernel 收到的 y 指针本身是对的 —— 说明踩踏发生在框架侧。
    // 16 MB 是官方算子常用的系统 workspace 大小。
    // 大小可用 S9_WS 覆盖（字节），用来量「框架是否每次 launch 都清 workspace」——
    // 若清，16 MB 就是每次几十微秒的固定开销，小档全被它吃掉。
    size_t wsBytes = 16 * 1024 * 1024;
    const char* wsEnv = getenv("S9_WS");
    if (wsEnv != nullptr) { long v = atol(wsEnv); if (v >= 0) { wsBytes = (size_t)v; } }
    if (ws != nullptr) { ws[0] = wsBytes; }
    if (getenv("S9_DBG") != nullptr) {
        /* ═══ 一行到底的**最终配置**（`[S9CFG]`）═══
         *
         * A/B 脚本要靠它回答"这一对我是不是把同一个配置测了两遍"。原先只打
         * `[S9] rows=… nOut=… strategy=… blockDim=… bands=…`，**planOn / gBuf /
         * aBuf / maxInBand / tileGMax / bandBal 都不在里面** —— 两行相同并不等于
         * 配置相同：改了 gBuf（进 UB 账、直接决定 tileG）或关了每带计划，
         * `[S9]` 那一行照样一模一样。
         *
         * 字段顺序**固定**（脚本按整行文本比对），只增不改 —— 要加字段就加在末尾，
         * 否则历史记录里的两行会开始"看起来不同"。
         *
         * 2026-09-13 删除**末尾**两个字段 `s5=… chunk=…`（strategy 5 整体撤掉）。
         * 它们本来就在最后，所以删除后**前面每个字段的位置与含义都没变**，
         * 历史记录与新记录的差别只剩尾缀 —— 这正是"新字段一律加在末尾"换来的。
         *
         * 注意它打在**最终校验与每带计划之后**，所以是真正下发给 kernel 的那一份
         * （`strategy`/`bands`/`gBuf`/`aBuf` 都取的是 CheckConcatFinal 收敛后的值）。 */
        fprintf(stderr,
                "[S9CFG] rows=%u nOut=%u numIn=%u esz=%u key=%u mix=%u aicPct=%u "
                "strategy=%u blockDim=%u bands=%u gBuf=%u aBuf=%u maxInBand=%u "
                "tileGMax=%u ub=%u planOn=%u bandBal=%u bandGrain=%u l2=%u l2o=%u "
                "pipe=%u hint=%u pre=%u layOn=%u trim=%u dense=%u\n",
                ct.rows, ct.nOut, ct.numIn, (uint32_t)esz, tilingKey, (uint32_t)useMix,
                aicPct, strategy, blockDim, bands, gBuf, aBuf, maxInB, tileGMax, ubBudget,
                planOn, ct.bandBal, tiling.get_bandGrain(), l2Mode, l2ModeOut, pipeMode,
                hintMode, preMode, layOn, ct.trimOn, ct.denseOn);
        fprintf(stderr, "[S9] rows=%u nOut=%u numIn=%u strategy=%u blockDim=%u bands=%u ub=%u l2=%u l2o=%u pipe=%u esz=%u mix=%u aicPct=%u key=%u\n",
                ct.rows, ct.nOut, ct.numIn, strategy, blockDim, bands, ubBudget, l2Mode, l2ModeOut, pipeMode, (uint32_t)esz,
                (uint32_t)useMix, aicPct, tilingKey);
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* s0 = context->GetDynamicInputShape(0, 0);
    gert::Shape* y = context->GetOutputShape(0);
    if (s0 == nullptr || y == nullptr) { return GRAPH_FAILED; }
    *y = *s0;
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return GRAPH_FAILED; }
    const int64_t* dimPtr = attrs->GetInt(0);
    if (dimPtr == nullptr) { return GRAPH_FAILED; }
    int64_t rank = static_cast<int64_t>(s0->GetDimNum());
    int64_t dim = *dimPtr;
    if (dim < 0) { dim += rank; }
    if (dim < 0 || dim >= rank) { return GRAPH_FAILED; }
    // 输出只有 dim 这一维变长，其余与输入相同。
    //
    // **路数必须从节点上取。** 原来这里也是"探到 nullptr 为止、最多 CONCAT_MAX_IN 次"，
    // 而当时 CONCAT_MAX_IN=64 —— 超过 64 路时推出来的 total 只算了前 64 路，
    // 输出 shape 比真实的小一截。InferShape 的结果是 aclnn 校验输出张量的依据，
    // **对不上就整个调用失败**（比"结果不对"更早、更硬）。
    // 赛题的切分能切出 123 路甚至 511 路，所以这条是提交失败的直接原因之一。
    const size_t numIn = context->GetComputeNodeInputNum();
    if (numIn == 0) { return GRAPH_FAILED; }
    int64_t total = 0;
    for (size_t k = 0; k < numIn; ++k) {
        const gert::Shape* sk = context->GetDynamicInputShape(0, k);
        if (sk == nullptr) { return GRAPH_FAILED; }
        total += sk->GetDim(dim);
    }
    y->SetDim(dim, total);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
