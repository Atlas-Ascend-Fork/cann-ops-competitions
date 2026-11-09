/* Concat 的主机孪生 —— 三条路径各自复刻一遍，逐位互相对拍。
 *
 * 这个算子有四条路：0 按行切、1 按列切、2 DMA 拼装、3 向量拼装（默认关）。
 * 路 0 在真机上跑过很多轮，路 1 与路 2 只在少数形状上跑过；
 * **而它们必须给出完全相同的结果** —— 这条不变式在主机上就能验，
 * 不需要 NPU，也不需要仿真。
 *
 * 三条检查：
 *   ① 每个输出位置**恰好写一次**，且写进去的是正确的来源（输入号 + 元素号）；
 *   ② 三条路径逐位相同；
 *   ③ host 选了拼装路（strategy 2）时，它的**对齐前提必须成立** ——
 *      `offOut[i]·esz` 是 32 的倍数、UB 行距推进恰好等于 pitch。
 *      前提不成立却选了这条路，真机上就是往没对齐的 UB 地址搬（未定义行为）。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include "../op_host/s9_tiling_calc.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...)                                                            \
    do {                                                                            \
        if (cond) { ++g_pass; }                                                     \
        else { ++g_fail; printf("  \033[31m✗\033[0m %s:%d  ", __func__, __LINE__);  \
               printf(__VA_ARGS__); printf("\n"); }                                 \
    } while (0)

static constexpr uint32_t BLOCK_BYTES = 32;
static uint32_t CeilDiv(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b : 0; }
static uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
static uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

/* 输出每个位置记 (输入号 << 32 | 元素下标)，比只比数值更能定位错在哪。 */
static inline int64_t Tag(uint32_t in, uint64_t el) { return ((int64_t)in << 32) | (int64_t)el; }

struct Out {
    std::vector<int64_t> v;
    std::vector<int>     owner;
    Out(size_t n) : v(n, -1), owner(n, -1) {}
};


/* 常规路径（0/1）的**搬运参数**回放：行距推进与 UB 上界。
 *
 * 这些是"算错了不报错、只是静默踩踏"的量，孪生里必须显式钉住：
 *   · GM 侧每块推进 = blockLen + srcStride(字节)，必须**恰好等于一行**；
 *   · UB 侧每块推进 = alignUp(blockLen,32) + dstStride·32，必须等于 rowUb·esz；
 *     且 (rowUb·esz − alignUp(blockLen,32)) 必须是 32 的倍数，否则 dstStride
 *     那个除法在 kernel 里是**整数截断**，推进量会悄悄少一截；
 *   · 一批 cnt 行占 cnt·rowUb·esz 字节，**不能超过 ubBudget** ——
 *     超了就是写出 UB 区域，真机上踩别人的缓冲；
 *   · blockCount ≤ 4095（DataCopyPad 的硬上限）。
 *
 * 注：列切路的 `per` 按 32B 对齐是**性能选择不是正确性** ——
 * UB 侧起点恒为 0、天然对齐，GM 侧 DataCopyPad 无对齐要求。
 * 所以这里不查它，去掉对齐也不该报错。 */
static void CheckCopyParams(const char *name, const s9::ConcatTiling &t, uint32_t esz,
                            uint32_t nIn, uint32_t cols, uint32_t cnt)
{
    uint32_t blockLen = cols * esz;
    uint32_t rowUb    = AlignUp(blockLen, BLOCK_BYTES) / esz;
    uint32_t ubBytes  = rowUb * esz;

    CHECK(ubBytes >= AlignUp(blockLen, BLOCK_BYTES) &&
          (ubBytes - AlignUp(blockLen, BLOCK_BYTES)) % BLOCK_BYTES == 0,
          "%s：UB 行距 %u 与 alignUp(blockLen,32)=%u 之差不是 32 的倍数",
          name, ubBytes, AlignUp(blockLen, BLOCK_BYTES));

    uint32_t srcStrideBytes = (nIn - cols) * esz;
    CHECK(blockLen + srcStrideBytes == nIn * esz,
          "%s：CopyIn 的 GM 推进 %u ≠ 一行 %u", name,
          blockLen + srcStrideBytes, nIn * esz);

    uint32_t dstStrideBytes = (t.nOut - cols) * esz;
    CHECK(blockLen + dstStrideBytes == t.nOut * esz,
          "%s：CopyOut 的 GM 推进 %u ≠ 输出一行 %u", name,
          blockLen + dstStrideBytes, t.nOut * esz);

    CHECK((uint64_t)cnt * ubBytes <= t.ubBudget,
          "%s：一批 %u 行要 %llu 字节 UB，超过预算 %u（cols=%u）",
          name, cnt, (unsigned long long)cnt * ubBytes, t.ubBudget, cols);
    CHECK(cnt <= 4095, "%s：blockCount=%u 超过 4095", name, cnt);
}

/* 路 0：按行切；每个输入各走一趟，列方向再按 colChunk 分块。 */
static void Strat0(const s9::ConcatTiling &t, uint32_t esz, Out &o)
{
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
    uint32_t colChunk = (t.ubBudget / esz / alignEl) * alignEl;
    if (colChunk == 0) { colChunk = alignEl; }
    /* **二维分核**（`S9_GRID`）：核编号拆成行组 × 列组。`colParts == 1` 时
     * `rowParts == bn`、`ri == bi`、列区间 [0, nIn)，与旧回放**逐字相同**。
     * 这一段必须独立复算：行组与列组任一处算错，核之间就互相写对方的地盘，
     * 而 `DataCopyPad` 对 GM 落点既不查对齐也不查越界 —— **错且不报错**。 */
    const uint32_t cp = t.colParts ? t.colParts : 1;
    const uint32_t rowParts = (cp > 1) ? (bn / cp ? bn / cp : 1) : bn;
    for (uint32_t bi = 0; bi < bn; ++bi) {
        const uint32_t ri = (cp > 1) ? (bi / cp) : bi;
        const uint32_t ci = (cp > 1) ? (bi % cp) : 0;
        uint32_t q = t.rows / rowParts, r = t.rows % rowParts;
        uint32_t beg = q * ri + (ri < r ? ri : r);
        uint32_t end = beg + q + (ri < r ? 1 : 0);
        if (beg >= end) { continue; }
        for (uint32_t k = 0; k < t.numIn; ++k) {
            uint32_t nIn = t.nIn[k], off = t.offOut[k];
            if (nIn == 0) { continue; }
            uint32_t cLo = 0, cHi = nIn;
            if (cp > 1) {
                const uint32_t G  = CeilDiv(nIn, alignEl);
                cLo = (uint32_t)((uint64_t)ci * G / cp) * alignEl;
                cHi = Min((uint32_t)((uint64_t)(ci + 1) * G / cp) * alignEl, nIn);
                if (cHi <= cLo) { continue; }
            }
            for (uint32_t cc = cLo; cc < cHi; cc += colChunk) {
                uint32_t cols = Min(cHi - cc, colChunk);
                for (uint32_t row = beg; row < end; ++row) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        size_t d = (size_t)row * t.nOut + off + cc + c;
                        if (o.owner[d] >= 0) { o.owner[d] = -2; }   // 重复写
                        else { o.owner[d] = (int)bi; }
                        o.v[d] = Tag(k, (uint64_t)row * nIn + cc + c);
                    }
                }
            }
        }
    }
}

/* 路 1：按列切；区间在**每个输入内部**算，所有行都归本核。 */
static void Strat1(const s9::ConcatTiling &t, uint32_t esz, Out &o)
{
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
    uint32_t colChunk = (t.ubBudget / esz / alignEl) * alignEl;
    if (colChunk == 0) { colChunk = alignEl; }
    for (uint32_t bi = 0; bi < bn; ++bi) {
        uint32_t colBase = 0;                    // 与 kernel 的 colBase_ 同步推进
        for (uint32_t k = 0; k < t.numIn; ++k) {
            uint32_t nIn = t.nIn[k], off = t.offOut[k];
            if (nIn == 0) { continue; }
            // **每一路自己定核数**（与 kernel 的 `One()` 一字一句对齐；
            // 那边的常量叫 S9_COL_BURST_EL，两边必须同值）。
            uint32_t coresK = nIn / (t.colBurstEl ? t.colBurstEl : 1);
            if (coresK == 0)  { coresK = 1; }
            if (coresK > bn)  { coresK = bn; }
            uint32_t base = colBase;                 // 起点＝已分出去的核数累计
            colBase = bn ? ((colBase + coresK) % bn) : 0;
            uint32_t rel  = bn ? ((bi + bn - base) % bn) : 0;
            if (rel >= coresK) { continue; }
            // **与 kernel 的 One() 一字一句对齐**：均衡切分，不是「统一份额＋末核吃剩下」。
            // 旧写法 per = AlignUp(⌈nIn/coresK⌉, alignEl) 会让尾巴上的核 c0 ≥ nIn 空转
            //（nIn=1000 fp32、alignEl=8、coresK=40 ⇒ per=32、32×40=1280 > 1000）。
            uint32_t c0 = 0, c1 = 0;
            if (coresK == 1) {
                c1 = nIn;
            } else if (t.bandBal == 0) {
                uint32_t per = AlignUp(CeilDiv(nIn, coresK), alignEl);
                c0 = per * rel;
                c1 = Min(c0 + per, nIn);
            } else {
                const uint32_t G  = CeilDiv(nIn, alignEl);
                const uint32_t g0 = (uint32_t)((uint64_t)rel * G / coresK);
                const uint32_t g1 = (uint32_t)((uint64_t)(rel + 1) * G / coresK);
                c0 = g0 * alignEl;
                c1 = Min(g1 * alignEl, nIn);
            }
            if (c0 >= nIn || c1 <= c0) { continue; }
            for (uint32_t cc = c0; cc < c1; cc += colChunk) {
                uint32_t cols = Min(c1 - cc, colChunk);
                for (uint32_t row = 0; row < t.rows; ++row) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        size_t d = (size_t)row * t.nOut + off + cc + c;
                        if (o.owner[d] >= 0) { o.owner[d] = -2; }
                        else { o.owner[d] = (int)bi; }
                        o.v[d] = Tag(k, (uint64_t)row * nIn + cc + c);
                    }
                }
            }
        }
    }
}

/* 路 2：DMA 拼装。**按 DataCopyPad 的语义回放 UB 行距的推进**，
 * 顺带把对齐前提当断言查掉。 */
/* 路 5：**输出线性切分**（`rows == 1` 专用）。
 *
 * 每核拿一段连续的输出列 `[c0, c1)`（按 32B 粒度取整），夹出与它相交的那几路，
 * 各自按 chunk 搬。复刻 kernel 的 `S9RunLinear` ＋ `S9MixRunHalf<kAiv,kPair=false>`。
 *
 * **这条路要验的两件事与别的路不同**：
 *   ① 区间按 32B 取整之后**仍然无缝覆盖 [0, nOut)** —— 取整是往上取的，
 *      相邻两核的 c1/c0 必须严丝合缝，差一个粒度就是漏写或重复写（都不报错）；
 *   ② 夹取 `[a, b)` 与 kernel 的 `S9ClipRoute` 同一个口径 —— 这里独立写一遍，
 *      两边算出不同的段就会被 Verify 的 dup/miss 抓住。 */
static void Strat5(const s9::ConcatTiling &t, uint32_t esz, Out &o, uint32_t &gapBad)
{
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    const uint32_t alignEl = (32 / esz) ? (32 / esz) : 1;
    // kernel 一次搬多大（S9_MIX_UB_CHUNK 默认 16 KB）；分块不该改变结果，
    // 这里照搬是为了让"分块边界"也进到回放里。
    const uint32_t chunkEl = (16u * 1024u) / esz;
    gapBad = 0;
    uint32_t prevC1 = 0;
    for (uint32_t bi = 0; bi < bn; ++bi) {
        uint32_t c0 = AlignUp((uint32_t)((uint64_t)t.nOut * bi / bn), alignEl);
        uint32_t c1 = AlignUp((uint32_t)((uint64_t)t.nOut * (bi + 1) / bn), alignEl);
        if (c0 > t.nOut) { c0 = t.nOut; }
        if (c1 > t.nOut) { c1 = t.nOut; }
        // ① 无缝：本核的起点必须正好接上上一核的终点
        if (c0 != prevC1) { ++gapBad; }
        prevC1 = c1 > c0 ? c1 : c0;
        if (c1 <= c0) { continue; }
        uint32_t off = 0;
        for (uint32_t k = 0; k < t.numIn; ++k) {
            const uint32_t n = t.nIn[k];
            const uint32_t sBeg = off;
            off += n;
            if (n == 0 || sBeg >= c1 || off <= c0) { continue; }   // 同 S9ClipRoute
            const uint32_t a = sBeg > c0 ? sBeg : c0;
            const uint32_t b = off < c1 ? off : c1;
            if (b <= a) { continue; }
            for (uint32_t pp = 0; pp < b - a; pp += chunkEl) {
                const uint32_t cnt = Min((b - a) - pp, chunkEl);
                for (uint32_t e = 0; e < cnt; ++e) {
                    const size_t d = (size_t)(a + pp + e);        // rows==1 ⇒ 行下标恒为 0
                    if (o.owner[d] >= 0) { o.owner[d] = -2; }
                    else { o.owner[d] = (int)bi; }
                    o.v[d] = Tag(k, (uint64_t)(a + pp + e) - sBeg);
                }
            }
        }
    }
    // 末核必须正好收到 nOut，否则尾巴没人写
    if (prevC1 != t.nOut) { ++gapBad; }
}

static void Strat2(const s9::ConcatTiling &t, uint32_t esz, Out &o,
                   uint32_t &alignBad, uint32_t &pitchBad)
{
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    const uint32_t pitch = AlignUp(t.nOut * esz, BLOCK_BYTES);
    uint32_t tile = pitch ? t.ubBudget / pitch : 0;
    if (tile == 0) { return; }
    if (tile > 4095) { tile = 4095; }
    alignBad = 0; pitchBad = 0;
    for (uint32_t k = 0; k < t.numIn; ++k) {
        if (t.nIn[k] == 0) { continue; }
        if ((t.offOut[k] * esz) % BLOCK_BYTES != 0) { ++alignBad; }
        uint32_t blockLen = t.nIn[k] * esz;
        uint32_t gap = pitch - AlignUp(blockLen, BLOCK_BYTES);
        if (gap % BLOCK_BYTES != 0) { ++pitchBad; }
        // UB 侧一块推进 alignUp(blockLen,32) + dstStride·32，必须恰好是 pitch
        if (AlignUp(blockLen, BLOCK_BYTES) + gap != pitch) { ++pitchBad; }
    }
    for (uint32_t bi = 0; bi < bn; ++bi) {
        uint32_t q = t.rows / bn, r = t.rows % bn;
        uint32_t beg = q * bi + (bi < r ? bi : r);
        uint32_t end = beg + q + (bi < r ? 1 : 0);
        for (uint32_t row0 = beg; row0 < end; row0 += tile) {
            uint32_t cnt = Min(end - row0, tile);
            // UB 里拼一段 cnt 行，再整段写出
            for (uint32_t b = 0; b < cnt; ++b) {
                uint32_t row = row0 + b;
                for (uint32_t k = 0; k < t.numIn; ++k) {
                    uint32_t nIn = t.nIn[k], off = t.offOut[k];
                    if (nIn == 0) { continue; }
                    for (uint32_t c = 0; c < nIn; ++c) {
                        size_t d = (size_t)row * t.nOut + off + c;
                        if (o.owner[d] >= 0) { o.owner[d] = -2; }
                        else { o.owner[d] = (int)bi; }
                        o.v[d] = Tag(k, (uint64_t)row * nIn + c);
                    }
                }
            }
        }
    }
}

/* 路 3：向量拼装（Gather）＋**列带分核**＋**一次摊开的多行偏移表**。
 *
 * 这条路真机上没有别的护栏 —— 表算错一格不会报错，只会把邻居的数据搬到输出里。
 * 所以孪生把 kernel 的每一步算术照抄一遍：
 *   ① 核排成 bands × (blockDim/bands) 的网格，本核只碰列带 [cb0,cb1) 与自己的行组；
 *   ② 与带相交的路**一律整行入暂存**（blockCount=1 的连续 DMA），行步长恒为 nIn·esz；
 *      暂存起点 stBase 是这些路的前缀和，每段起点按 32B 对齐（DataCopyPad 的 UB 侧硬要求）；
 *   ③ 偏移表第 0 行 off[cl] = cl·esz + (stBase[k] + (cb0−off_k)·esz)、dlt[cl] = nIn[k]·esz；
 *      补白列（cl ≥ bw）off 留 cl·esz、dlt 取 0；第 j 行 = 第 0 行 + j·dlt；
 *   ④ 一个 tile 只发一次 Gather，count = tblRow·cnt；
 *   ⑤ 写出是 blockCount=cnt、blockLen=bw·esz、dstStride=(nOut−bw)·esz 的跨步块。
 *
 * 顺带钉死三件"错了不报错"的事：暂存区不能越界、每路起点必须 32B 对齐、
 * GM 侧每块推进必须恰好是一整行输出。
 *
 * **硬件行为与代码公式要分开写。** 暂存里数据摆在哪由 DataCopyPad 决定（hwStrd），
 * 表里填什么由 kernel 的公式决定（tblStrd）；共用一个变量的话公式写错会两边一起错、
 * 孪生反而全对 —— 故障注入验过。 */
static void Strat3(const s9::ConcatTiling &t, uint32_t esz, Out &o,
                   uint32_t &ubBad, uint32_t &alignBad, uint32_t &gmBad,
                   uint32_t &planBad, uint32_t &trimBad,
                   uint64_t &readFull, uint64_t &readTrim)
{
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    /* **边界路裁剪**（`S9_TRIM`）：被带切开的那 ≤2 条路只搬相交的那一段。
     * 孪生里必须**独立复算**这件事（不能复用 kernel 的结论），而且要比到
     * **地址级**：搬进来的字节集合恰等于「原来那批 ∩ 本带」。 */
    const bool trim = (t.trimOn != 0);
    // Gather 的载体宽度：int8 借道 half（见 op_kernel 的 S9GatherAs）。
    // **暂存里数据摆在哪按 esz 算（DMA 的事），表里填什么按 gsz 算（Gather 的事）**，
    // 两个口径共用一个变量就会自洽假通过。
    const uint32_t gsz = (esz == 1) ? 2 : esz;
    uint32_t bands = t.bands ? t.bands : 1;
    // 与 kernel 的守卫逐字对齐：不整除就退回 1（host 只会产出因数，这里是兜底）
    if (bands > bn || bn % bands != 0) { bands = 1; }
    uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
    // **粒度取 tiling 里下发的那份**，不再自己算一遍 —— kernel 也是读这个字段的。
    // 这里若退回读常量，S9_BANDGRAIN 扫出来的配置孪生就验不到。
    const uint32_t grainEl  = t.bandGrain ? t.bandGrain : (s9::BAND_GRAIN_BYTES / esz);
    // **列带边界走 s9::BandRange，不在这里再写一遍公式。**
    // 孪生的意义就是"独立复算一遍"，但**边界必须与 kernel 同源** ——
    // 两边各写一份的后果是核之间互相写对方的地盘，而孪生反而验不出来。
    // 这里独立复算的是「每条带里到底挤进几路」和「余量够不够」。
    // **列带边界按 tiling 里发布的那个切法**（t.bandBal，默认 0 ＝ 旧的「统一带宽＋
    // 末带吃剩下」）。这里原先漏传 bal ⇒ 默认走均衡切分，孪生验的是**没发布的那套**。
    const uint32_t bandCols = s9::BandColsMax(t.nOut, bands,
                                              grainEl ? grainEl : alignEl, t.bandBal);
    // **暂存块数也取发布值**（t.gBuf），不再写死 2 —— 它进 UB 账，写死就验不到
    // 「S9_GBUF 改到 4、而事件号只有 3 个」这类错。
    const uint32_t gBuf = t.gBuf ? t.gBuf : 2;
    (void)bandCols;

    for (uint32_t bi = 0; bi < bn; ++bi) {
        // 核到 (带, 组) 的映射。带数目前一定整除核数（gExtra 恒为 0），
        // 但保留通用写法 —— 一旦哪天再放开约束，这里和"没有核空转"的断言都还成立。
        const uint32_t gBase = bn / bands, gExtra = bn % bands;
        uint32_t bandIdx = 0, grpIdx = 0, grp = gBase, start = 0;
        for (uint32_t b = 0; b < bands; ++b) {
            const uint32_t gcnt = gBase + (b < gExtra ? 1u : 0u);
            if (bi < start + gcnt) { bandIdx = b; grpIdx = bi - start; grp = gcnt; break; }
            start += gcnt;
        }
        uint32_t cb0 = 0, cb1 = 0;
        s9::BandRange(t.nOut, bands, grainEl ? grainEl : alignEl, bandIdx, cb0, cb1, t.bandBal);
        if (cb0 >= cb1) { continue; }
        // **带首必须落在 32B 上**（`cb0·esz` 是 32 的倍数）：一条带写出去是
        // blockLen=bw·esz、按 nOut·esz 跨步的块，起点不对齐时每块两端都要多压缓存行。
        // 判据在 host（bandGrain 是 bandGrainBytes/esz，而 512 % 32 == 0），
        // 这里独立复算一遍 —— 改了粒度又忘了这条，孪生要能报出来。
        if ((cb0 * esz) % BLOCK_BYTES != 0) { ++alignBad; }
        const uint32_t bw = cb1 - cb0;
        uint32_t q = t.rows / grp, rr = t.rows % grp;
        const uint32_t beg = q * grpIdx + (grpIdx < rr ? grpIdx : rr);
        const uint32_t end = beg + q + (grpIdx < rr ? 1 : 0);
        if (beg >= end) { continue; }

        // ═══ 逐块布局：**与 kernel / host 同一份函数** ═══
        //
        // 这里原先是照抄 kernel 的 perTile/fixed 再自己拼一遍 UB 总量。照抄的
        // 问题不是"抄错"，而是**两边会一起错**：公式漏项时孪生算出同一个 tileG，
        // 于是自洽地全绿（int8 的 slack half 副本就是这么漏的，真机 507035）。
        // 现在三方共用 `s9::PlanConcatStage`，孪生独立复算的是**轮转后的绝对地址**。
        //
        // 与发布的差异只剩一个：host 的 `S9_TILEG`（tileG 的字节上限）是环境变量，
        // 孪生按默认 0（不限）算 —— 那个旋钮只用于上板扫最优点，不是发布配置。
        const uint32_t myRows = (end > beg) ? (end - beg) : 1;
        // 模式 2 连 UB 账一起裁（与 host 的 CalcConcatCoreLayouts / GatherFitsBands 同源）；
        // 模式 1 只裁 kernel 的读 ⇒ **账与 tileG 一个字不变**，这样才能把
        // "少读字节"的收益单独量出来。
        const uint64_t trimRowBytes =
            (t.trimOn >= 2)
                ? s9::ConcatBandStagedBytes(esz, t.nIn, t.numIn, cb0, cb1, /*trim=*/true)
                : 0;
        const s9::ConcatStageLayout lay = s9::PlanConcatStage(
            bw, esz, t.numIn, t.maxIn, t.slackIn, gBuf, t.aBuf ? t.aBuf : 1,
            /*tileGMax=*/0, myRows, /*tblK=*/t.tblK, bands == 1 ? t.denseOn : 0u, trimRowBytes, bands == 1);
        const uint32_t tileG       = lay.tileG;
        const uint32_t outPitch    = lay.outPitch;
        const uint32_t tblRow      = lay.tblRow;
        const uint32_t superG      = lay.superG;
        const uint32_t rowUse      = lay.rowUse;
        const uint32_t slack       = lay.slack;
        const uint32_t stageSpan   = lay.stageSpan;      // **逐块 32B 对齐后的**
        const uint32_t stageSpanEl = stageSpan / (esz ? esz : 1);
        const uint32_t nSlackT     = (t.slackIn && t.slackIn < t.numIn) ? t.slackIn : t.numIn;

        // **总 UB 断言：这份账里含序幕与指针暂存**（host 的判据也是它）。
        if (!lay.fits || lay.totalBytes > lay.limitBytes) { ++ubBad; }
        // 跨度的定义本身也钉一下：**对齐过的**跨度 = AlignUp(tileG·rowUse + slack, 32)。
        // 旧写法少了外面那层 AlignUp —— 这条断言就是那个缺陷的哨兵。
        if (stageSpan != AlignUp(tileG * rowUse + slack, BLOCK_BYTES)) { ++ubBad; }
        // 每一块暂存的起点都要落在 32B 上（手动轮转的跨度自己管）
        if (lay.stageSpan % BLOCK_BYTES != 0) { ++alignBad; }
        for (uint32_t b = 0; b < gBuf; ++b) {
            if (lay.StgBase(b) % BLOCK_BYTES != 0 || !lay.StgBlockOk(b)) { ++alignBad; }
        }
        // 表的落点要 32B 对齐：kernel 里 off[j·tblRow] 是向量写的起点
        if ((tblRow * 4) % BLOCK_BYTES != 0) { ++alignBad; }
        // GM 写出参数：每块推进必须恰好是输出的一整行
        if (bw * esz + (t.nOut - bw) * esz != t.nOut * esz) { ++gmBad; }
        // UB 读出参数：srcStride 那个除法在 kernel 里是整数截断，差值必须是 32 的倍数
        if (superG == 1) {
            if ((outPitch - AlignUp(bw * esz, BLOCK_BYTES)) % BLOCK_BYTES != 0) { ++gmBad; }
        } else {
            if (bands != 1 || outPitch != bw * esz || lay.asmSpan % BLOCK_BYTES != 0) { ++gmBad; }
        }

        std::vector<uint32_t> stBase(t.numIn, 0), hwStrd(t.numIn, 0), tblStrd(t.numIn, 0);
        std::vector<uint32_t> lo0(t.numIn, 0), wid(t.numIn, 0);
        // **本带实际入暂存的路数**：host 数出来的 maxInBand 不能小于它 ——
        // 小了就是余量留少了，真机 UB 越界（vector core exception 507035），
        // 而各路宽度是随机的，任何"按均分估计"的上界都会低估。
        uint32_t staged = 0;
        uint32_t acc = 0;
        for (uint32_t i = 0; i < t.numIn; ++i) {
            stBase[i] = acc;
            if (t.nIn[i] == 0 || t.offOut[i] >= cb1 || t.offOut[i] + t.nIn[i] <= cb0) {
                continue;
            }
            uint32_t lo = t.offOut[i] > cb0 ? t.offOut[i] : cb0;
            uint32_t hi = Min(t.offOut[i] + t.nIn[i], cb1);
            lo0[i] = lo - cb0;
            wid[i] = hi - lo;
            /* 行距：不裁剪时就是整路宽度；裁剪时被切开的路只留 `W` 列、按 32B 补齐
             *（`DataCopyPad(blockCount=cnt, blockLen=W·esz, dstStride=0)` 的落点
             * 要求 —— 见 GMToUB 文档：dstStride 单位是 32B dataBlock、UB 目的地址
             * 必须 32B 对齐）。**两条路走同一个函数**（`s9::ConcatRoutePitchEl`），
             * 与 host 的 UB 账、kernel 的前缀和都同源。 */
            hwStrd[i]  = s9::ConcatRoutePitchEl(esz, t.nIn[i], t.offOut[i], cb0, cb1, trim);
            tblStrd[i] = hwStrd[i] * gsz;       // kernel 填进 dlt 的值（载体字节）
            // 段起点要同时满足两边的 32B 对齐 —— 按**窄**的元素宽度算
            const uint32_t alignEl = BLOCK_BYTES / ((esz < gsz) ? esz : gsz);
            if ((stBase[i] * esz) % BLOCK_BYTES != 0 ||
                (stBase[i] * gsz) % BLOCK_BYTES != 0) { ++alignBad; }
            acc += AlignUp(tileG * hwStrd[i], alignEl);
            ++staged;
        }
        if (acc > stageSpanEl) { ++ubBad; }
        CHECK(staged <= nSlackT,
              "带 %u 实际入暂存 %u 路，而余量只按 %u 路留（maxInBand=%u，numIn=%u）",
              bi, staged, nSlackT, t.maxInBand, t.numIn);

        /* **每带执行计划必须与"真扫一遍"逐位相同。**
         * kernel 开着 S9_BANDPLAN 时按计划跳过扫描、并且只取本带那几路的指针；
         * 计划一旦与扫描不一致，就会漏掉整条路 —— 输出留着垃圾，而且不报错。
         * 这里独立复算一遍（wid[] 是上面按带边界算出来的相交宽度）。 */
        {
            uint32_t lo = t.numIn, hi = 0;
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (wid[i] == 0) { continue; }
                if (i < lo) { lo = i; }
                hi = i + 1;
            }
            if (lo >= hi) { lo = 0; hi = 0; }
            s9::ConcatBandPlan bp{};
            if (s9::CalcConcatBandPlan(t, bp) && bandIdx < bp.bands) {
                if (bp.lo[bandIdx] != lo || bp.hi[bandIdx] != hi) { ++planBad; }
                if (hi > 0 && bp.off[bandIdx] != t.offOut[lo]) { ++planBad; }
            } else {
                ++planBad;               // 计划算不出来本身就是问题
            }
        }

        // 第 0 行的表；补白列 off=cl·esz、dlt=0
        // **kernel 现在依赖「与带相交的路是连续的一段下标」** —— offs[] 是 nIn 的
        // 前缀和、单调不减，所以这条应当恒成立。它一旦不成立，kernel 的读循环
        // 会漏掉中间某些路，而输出对应的列就变成暂存区里的垃圾（不报错）。
        {
            uint32_t lo = t.numIn, hi = 0, holes = 0;
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (wid[i] == 0) { continue; }
                if (i < lo) { lo = i; }
                hi = i + 1;
            }
            for (uint32_t i = lo; i < hi; ++i) {
                // 段内允许出现空片（nIn==0）和被 wid 判掉的路，但不能出现
                // "有宽度却不相交"的路 —— 那就说明相交集合不连续。
                if (t.nIn[i] != 0 && wid[i] == 0) { ++holes; }
            }
            if (holes != 0) { ++gmBad; }
        }

        std::vector<int64_t> off0(tblRow), dlt(tblRow, 0);
        for (uint32_t c = 0; c < tblRow; ++c) { off0[c] = (int64_t)c * gsz; }
        // **同一张表建两遍：一遍按定义，一遍照 kernel 的「64 通道窗口 ＋ 逐 bit mask」。**
        // kernel 里那段是纯向量的（没有标量兜底），掩码算错就是静默写错列 ——
        // 这里逐位对拍是唯一能在主机上抓到它的地方。
        std::vector<int64_t> off0m(off0), dltm(tblRow, 0);
        std::vector<uint8_t> hit(tblRow, 0);
        constexpr uint32_t kWin = 64;                    // int32 一次迭代的元素数
        for (uint32_t i = 0; i < t.numIn; ++i) {
            if (wid[i] == 0) { continue; }
            /* 表基址的"输入起点"：裁剪后**左边界路**的紧凑行只从 `lo` 开始存，
             * 所以基址要传 `lo`（同 kernel 的 `offBase`）；其余路传 `offOut`。
             * 传错了就会指到槽前一路的地盘、甚至缓冲之外 —— 而且**不报错**。 */
            const bool bnd = trim && (hwStrd[i] != t.nIn[i]);
            const int64_t offBase = bnd ? (int64_t)(cb0 + lo0[i]) : (int64_t)t.offOut[i];
            const int64_t k = ((int64_t)stBase[i] + (int64_t)cb0 - offBase) * (int64_t)gsz;
            for (uint32_t kk = 0; kk < superG; ++kk) {
              for (uint32_t cl = lo0[i]; cl < lo0[i] + wid[i]; ++cl) {
                const uint32_t c = kk * bw + cl;
                // 按真实来源定义，与下面的mask基址公式分别计算。
                off0[c] = ((int64_t)stBase[i] + (int64_t)kk * hwStrd[i] + cl + cb0 - offBase) * gsz;
                dlt[c]  = (int64_t)superG * tblStrd[i];
              }
            }
            // ── kernel 的写法 ──
            for (uint32_t kk = 0; kk < superG; ++kk) {
              const uint32_t l0 = lo0[i] + kk * bw, l1 = l0 + wid[i];
              const int64_t base = k + (int64_t)kk * ((int64_t)hwStrd[i] - bw) * gsz;
              for (uint32_t w = l0 & ~(kWin - 1); w < l1; w += kWin) {
                const uint32_t s0 = (l0 > w) ? (l0 - w) : 0;
                const uint32_t e0 = ((l1 - w) < kWin) ? (l1 - w) : kWin;
                if (e0 <= s0) { continue; }
                CHECK((w * 4) % BLOCK_BYTES == 0,
                      "掩码窗口起点 %u 的字节地址不是 32B 对齐", w);
                uint64_t m0 = ((e0 - s0) >= kWin) ? ~(uint64_t)0
                                                  : ((((uint64_t)1 << (e0 - s0)) - 1) << s0);
                for (uint32_t b = 0; b < kWin; ++b) {
                    if (((m0 >> b) & 1u) == 0) { continue; }
                    const uint32_t c = w + b;
                    if (c >= tblRow) { continue; }
                    off0m[c] = (int64_t)c * gsz + base; // Adds 只加被 mask 选中的通道
                    dltm[c]  = (int64_t)superG * tblStrd[i];
                    ++hit[c];
                }
              }
            }
        }
        {
            uint32_t diff = 0, dup = 0;
            for (uint32_t c = 0; c < tblRow; ++c) {
                if (off0m[c] != off0[c] || dltm[c] != dlt[c]) { ++diff; }
                if (hit[c] > 1) { ++dup; }
            }
            CHECK(diff == 0, "掩码建表与定义不一致，%u 列对不上（bi=%u）", diff, bi);
            CHECK(dup == 0, "掩码窗口重叠，%u 列被写了不止一次（bi=%u）", dup, bi);
        }

        /* 暂存区按**全部 gBuf 块**开（元素下标），每一趟按轮转基址落位 ——
         * 原先是每趟 assign 一个单块，于是"第 2 块的起点算错"这类错**根本进不了
         * 这个数组**（它总是从 0 开始），孪生也就验不出来。 */
        std::vector<int64_t> st((size_t)(lay.stgBytes / (esz ? esz : 1)), -99);
        const uint64_t stgEl = st.size();       // 全部 gBuf 块合起来的元素数
        uint32_t it = 0;
        for (uint32_t r = beg; r < end; r += tileG, ++it) {
            const uint32_t cnt = Min(end - r, tileG);
            // 轮转：第 it 趟用第 (it % gBuf) 块。**基址是绝对地址**，与 kernel 的
            // `stgAll[ub * stgSpanEl]` 一字不差。
            const uint32_t ub     = (gBuf <= 1) ? 0 : (it % gBuf);
            const uint64_t baseEl = (uint64_t)ub * stageSpanEl;
            st.assign(st.size(), -99);
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (wid[i] == 0) { continue; }
                /* 紧凑槽：行距 `hwStrd[i]`，其中前 `wid[i]` 个元素是真数据
                 *（对应整行里 [lo, lo+wid) 那些列），其余是 DMA 补的 dummy
                 *（`-98`：**必须允许存在，但绝不能被 Gather 读到** —— 读到就说明
                 *  表或行距算错了，最终输出对不上）。 */
                /* `bnd` ＝ 这一路**被裁了**（只有裁过的边界路才有紧凑槽与 dummy）。
                 * 没裁的路（含不裁剪的整条路）仍然是整行真数据 —— 这两件事必须分开，
                 * 混在一起会把"整行"的列号算错（第一次改就踩了：40 项失败）。 */
                /* **"该不该裁"与"怎么裁"必须与 kernel 同源**：都用
                 * `s9::ConcatRouteTrimmed`（窄路补到 32B 反而更大 ⇒ 不裁）。
                 * 这里若自己写一份判断，孪生就会验一套、kernel 跑另一套。 */
                const bool bnd = trim && s9::ConcatRouteTrimmed(
                    esz, t.nIn[i], t.offOut[i], cb0, cb1);
                const uint32_t realN   = bnd ? wid[i] : t.nIn[i];
                // Tag 的第二维是**路内相对列**（见 Ref），不是绝对列
                const uint32_t colBase = bnd ? (cb0 + lo0[i] - t.offOut[i]) : 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    for (uint32_t e = 0; e < hwStrd[i]; ++e) {
                        uint64_t b = baseEl + (uint64_t)stBase[i]
                                   + (uint64_t)k * hwStrd[i] + e;
                        if (b >= stgEl) { ++ubBad; continue; }   // 越过整条暂存队列
                        st[(size_t)b] = (e < realN)
                            ? Tag(i, (uint64_t)(r + k) * t.nIn[i] + colBase + e)
                            : (int64_t)-98;
                    }
                    readFull += (uint64_t)t.nIn[i] * esz;        // 整行读会搬多少
                    readTrim += (uint64_t)realN * esz;           // 裁剪后搬多少
                }
            }

            /* ═══ **地址级断言**：搬进来的字节集合恰等于「原来那批 ∩ 本带」 ═══
             *   · 该在的都在：`c ∈ [lo,hi)` 的每个 (路, 行, 列) 都能在紧凑槽里找到；
             *   · 不该在的一个都没：槽里除真数据外只允许 DMA 补的 dummy，
             *     **不允许出现带外列的真数据**（那正是"整行读"的读放大）。
             * 只比最终输出是抓不到这一条的：多读的部分常常被丢掉、输出照样对。 */
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (wid[i] == 0) { continue; }
                const bool bnd = trim && s9::ConcatRouteTrimmed(
                    esz, t.nIn[i], t.offOut[i], cb0, cb1);
                const uint32_t realN   = bnd ? wid[i] : t.nIn[i];
                const uint32_t colBase = bnd ? (cb0 + lo0[i] - t.offOut[i]) : 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    for (uint32_t e = 0; e < hwStrd[i]; ++e) {
                        uint64_t b = baseEl + (uint64_t)stBase[i]
                                   + (uint64_t)k * hwStrd[i] + e;
                        if (b >= stgEl) { continue; }
                        const int64_t v = st[(size_t)b];
                        if (e < realN) {
                            if (v != Tag(i, (uint64_t)(r + k) * t.nIn[i] + colBase + e)) {
                                ++trimBad;                       // 该在的不在 / 是别人的
                            }
                        } else if (v != -99 && v != -98) {
                            ++trimBad;                           // 带外的真数据被搬进来了
                        }
                    }
                }
            }
            for (uint32_t k = 0; k < cnt; ++k) {
                for (uint32_t cl = 0; cl < bw; ++cl) {
                    // 表里是**载体字节**偏移（相对本块起点），换回元素下标才能查暂存
                    const uint32_t tc = superG > 1 ? (k % superG) * bw + cl : cl;
                    int64_t b = off0[tc] + (int64_t)(k / superG) * dlt[tc];
                    size_t  d = (size_t)(r + k) * t.nOut + cb0 + cl;
                    if (o.owner[d] >= 0) { o.owner[d] = -2; } else { o.owner[d] = (int)bi; }
                    int64_t el = (b >= 0 && b % gsz == 0) ? (b / gsz) : -1;
                    o.v[d] = (el >= 0 && (uint64_t)el < stageSpanEl)
                                 ? st[(size_t)(baseEl + (uint64_t)el)] : (int64_t)-99;
                }
                // 补白列也要落在暂存区内（它们逐行不动，读到垃圾但不能越界）
                for (uint32_t cl = superG > 1 ? tblRow : bw; cl < tblRow; ++cl) {
                    int64_t b = off0[cl] + (int64_t)k * dlt[cl];
                    if (b < 0 || b % gsz != 0 || (uint64_t)(b / gsz) >= stageSpanEl) { ++ubBad; }
                }
            }
        }
    }
}

/* ═══ 直写路（`S9_DIRECT=1`）的回放 ═══
 *
 * 它与向量拼装路**共用全部 tiling**（分核、分带、tileG、UB 账逐字节相同），
 * 差别只有一处：不在 UB 里拼整行，而是**行块在外、路在内**地每路直接写出。
 * 所以这里只回放那一处，band/core 映射与 `Strat3` 走同一份 `s9::BandRange`
 * 与 `s9::PlanConcatStage` —— 边界公式各写一份的后果是核之间互相写对方的地盘，
 * 而孪生反而验不出来。
 *
 * 要钉住的是三件"算错了不报错、只是静默踩踏"的事：
 *   ① **覆盖**：每笔写恰好落在 [lo, lo+cw) × cnt 行 —— 漏写/重复写由 `Verify` 抓；
 *   ② **源范围**：读的最后一个元素不越过这一路张量的末尾；
 *   ③ **UB 槽**：`nb` 个槽 × `tileG·align32(maxW·esz)` 不超过 `gBuf·stageSpan`，
 *      且单块 `cnt·align32(w·esz)` 不超过一个槽（超了就是写出 UB 区域，
 *      真机上是 507035 且不一定当场炸）。
 * 另外把两条 DMA 推进量钉成恒等式：读的 GM 推进 = 输入一行、写的 = 输出一行。 */
static void StratDirect(const s9::ConcatTiling &t, uint32_t esz, Out &o,
                        uint32_t &ubBad, uint32_t &gmBad, uint32_t &nbMin,
                        uint32_t mode, uint32_t &fellBack, uint32_t &tgMax)
{
    fellBack = 0; tgMax = 0;
    const uint32_t bn = t.blockDim ? t.blockDim : 1;
    uint32_t bands = t.bands ? t.bands : 1;
    if (bands > bn || bn % bands != 0) { bands = 1; }
    uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
    const uint32_t grainEl = t.bandGrain ? t.bandGrain : (s9::BAND_GRAIN_BYTES / esz);
    const uint32_t gBuf = t.gBuf ? t.gBuf : 2;
    nbMin = 0xffffffffu;

    for (uint32_t bi = 0; bi < bn; ++bi) {
        const uint32_t gBase = bn / bands, gExtra = bn % bands;
        uint32_t bandIdx = 0, grpIdx = 0, grp = gBase, start = 0;
        for (uint32_t b = 0; b < bands; ++b) {
            const uint32_t gcnt = gBase + (b < gExtra ? 1u : 0u);
            if (bi < start + gcnt) { bandIdx = b; grpIdx = bi - start; grp = gcnt; break; }
            start += gcnt;
        }
        uint32_t cb0 = 0, cb1 = 0;
        s9::BandRange(t.nOut, bands, grainEl ? grainEl : alignEl, bandIdx, cb0, cb1, t.bandBal);
        if (cb0 >= cb1) { continue; }
        const uint32_t bw = cb1 - cb0;
        const uint32_t q = t.rows / grp, rr = t.rows % grp;
        const uint32_t beg = q * grpIdx + (grpIdx < rr ? grpIdx : rr);
        const uint32_t end = beg + q + (grpIdx < rr ? 1 : 0);
        if (beg >= end) { continue; }
        const uint32_t myRows = end - beg;
        const uint64_t trimRowBytes =
            (t.trimOn >= 2)
                ? s9::ConcatBandStagedBytes(esz, t.nIn, t.numIn, cb0, cb1, /*trim=*/true)
                : 0;
        const s9::ConcatStageLayout lay = s9::PlanConcatStage(
            bw, esz, t.numIn, t.maxIn, t.slackIn, gBuf, t.aBuf ? t.aBuf : 1,
            /*tileGMax=*/0, myRows, /*tblK=*/t.tblK, /*dense=*/false, trimRowBytes, bands == 1);
        const uint32_t tileG = lay.tileG;
        if (tileG == 0) { ++ubBad; continue; }

        // 与 kernel 的压紧循环同一趟：夹取判据、最宽路宽
        struct Act { uint32_t idx, n, w, sh, out; };
        std::vector<Act> act;
        uint32_t maxRow = 0;
        for (uint32_t i = 0; i < t.numIn; ++i) {
            const uint32_t n = t.nIn[i], off = t.offOut[i];
            if (n == 0 || off >= cb1 || off + n <= cb0) { continue; }
            const uint32_t lo = off > cb0 ? off : cb0;
            const uint32_t hi = (off + n) < cb1 ? (off + n) : cb1;
            if (hi <= lo) { continue; }
            const uint32_t cw = hi - lo;
            const uint32_t rb = AlignUp(cw * esz, BLOCK_BYTES);
            if (rb > maxRow) { maxRow = rb; }
            act.push_back({i, n, cw, lo - off, lo});
        }
        if (act.empty() || maxRow == 0) { continue; }

        // ── 行块：mode 2 把整个暂存区用满（**只放大不缩小**）──
        const uint32_t total = gBuf * lay.stageSpan;
        uint32_t tg = tileG;
        if (mode >= 2) {
            uint32_t t2 = (total / 3u) / maxRow;
            if (t2 < 1) { t2 = 1; }
            if (t2 > myRows) { t2 = myRows; }
            if (t2 > 4095)   { t2 = 4095; }
            if (t2 > tg)     { tg = t2; }
        }
        // 可行性护栏：逐次减半，到 1 行还放不下就交回 Gather（kernel 同）
        while (tg > 1 && (uint64_t)tg * maxRow > (uint64_t)total) { tg >>= 1; }
        if ((uint64_t)tg * maxRow > (uint64_t)total) { ++fellBack; continue; }
        if (tg > tgMax) { tgMax = tg; }
        const uint64_t slotB = (uint64_t)tg * maxRow;
        uint32_t nb = (uint32_t)((uint64_t)total / slotB);
        if (nb == 0) { nb = 1; }
        if (nb > 3) { nb = 3; }
        if (nb < nbMin) { nbMin = nb; }
        // ③ nb 个槽真的放得下吗
        if ((uint64_t)nb * slotB > (uint64_t)total) { ++ubBad; }

        for (uint32_t r = beg; r < end; r += tg) {
            const uint32_t cnt = Min(end - r, tg);
            if (cnt > 4095) { ++gmBad; }
            for (const Act &a : act) {
                // ② 源范围：最后一块的末元素不能越过这一路的张量
                const uint64_t lastEl = (uint64_t)(r + cnt - 1) * a.n + a.sh + a.w;
                if (lastEl > (uint64_t)t.rows * a.n) { ++gmBad; }
                // 两条推进量的恒等式（错了就是跨步算反，静默错列）
                if (a.w * esz + (a.n - a.w) * esz != a.n * esz) { ++gmBad; }
                if (a.w * esz + (t.nOut - a.w) * esz != t.nOut * esz) { ++gmBad; }
                // ③ 单块占用不超过一个槽
                if ((uint64_t)cnt * AlignUp(a.w * esz, BLOCK_BYTES) > slotB) { ++ubBad; }
                // ① 覆盖
                for (uint32_t e = 0; e < cnt; ++e) {
                    const size_t rowOut = (size_t)(r + e) * t.nOut + a.out;
                    const uint64_t rowIn = (uint64_t)(r + e) * a.n + a.sh;
                    for (uint32_t c = 0; c < a.w; ++c) {
                        const size_t d = rowOut + c;
                        if (o.owner[d] >= 0) { o.owner[d] = -2; }
                        else { o.owner[d] = (int)bi; }
                        o.v[d] = Tag(a.idx, rowIn + c);
                    }
                }
            }
        }
    }
    if (nbMin == 0xffffffffu) { nbMin = 0; }
}

static void Ref(const s9::ConcatTiling &t, std::vector<int64_t> &ref)
{
    for (uint32_t row = 0; row < t.rows; ++row) {
        for (uint32_t k = 0; k < t.numIn; ++k) {
            for (uint32_t c = 0; c < t.nIn[k]; ++c) {
                ref[(size_t)row * t.nOut + t.offOut[k] + c] =
                    Tag(k, (uint64_t)row * t.nIn[k] + c);
            }
        }
    }
}

static void Verify(const char *name, const char *path, const s9::ConcatTiling &t,
                   const Out &o, const std::vector<int64_t> &ref)
{
    size_t n = ref.size();
    uint64_t wrong = 0, dup = 0, miss = 0;
    for (size_t i = 0; i < n; ++i) {
        if (o.owner[i] == -2) { ++dup; }
        else if (o.owner[i] < 0) { ++miss; }
        if (o.v[i] != ref[i]) { ++wrong; }
    }
    CHECK(dup == 0 && miss == 0, "%s/%s：重复写 %llu、漏写 %llu（blockDim=%u rows=%u nOut=%u）",
          name, path, (unsigned long long)dup, (unsigned long long)miss,
          t.blockDim, t.rows, t.nOut);
    CHECK(wrong == 0, "%s/%s：%llu 个位置来源不对", name, path,
          (unsigned long long)wrong);
}

static void RunCaseCores(const char *name, uint32_t rows, const std::vector<uint32_t> &nIns,
                         uint32_t esz, uint32_t maxCores)
{
    // 造 shapes：dim=1 的二维，(rows, nIn_k)
    std::vector<std::vector<int64_t>> sh;
    for (uint32_t n : nIns) { sh.push_back({(int64_t)rows, (int64_t)n}); }
    std::vector<const int64_t *> sp;
    std::vector<int64_t> rk;
    for (auto &v : sh) { sp.push_back(v.data()); rk.push_back(2); }

    s9::ConcatTiling t{};
    // **int8 借道 half 的拼装路在孪生里一律打开**（线上默认还关着，等 A/B）。
    // 孪生的价值就在于覆盖还没上板的那条路 —— 默认关着的话它一行也跑不到，
    // 和之前 gatherOn 默认 false 把整条向量拼装路藏了三天是同一个坑。
    s9::ConcatKnobs kn;
    kn.maxCores = maxCores;
    if (!s9::CalcConcat(sp.data(), rk.data(), (uint32_t)nIns.size(), 1, esz, t, kn)) {
        CHECK(false, "%s：CalcConcat 失败", name); return;
    }
    size_t total = (size_t)t.rows * t.nOut;
    if (total == 0) { return; }
    std::vector<int64_t> ref(total, -1);
    Ref(t, ref);

    Out o0(total); Strat0(t, esz, o0); Verify(name, "行切", t, o0, ref);

    // 搬运参数：对每个输入取一次实际会出现的 (cols, cnt)
    {
        uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
        uint32_t colChunk = (t.ubBudget / esz / alignEl) * alignEl;
        if (colChunk == 0) { colChunk = alignEl; }
        for (uint32_t k = 0; k < t.numIn; ++k) {
            if (t.nIn[k] == 0) { continue; }
            uint32_t cols = Min(t.nIn[k], colChunk);
            uint32_t rowUb = AlignUp(cols * esz, BLOCK_BYTES) / esz;
            uint32_t tile = t.ubBudget / (rowUb * esz);
            if (tile == 0) { tile = 1; }
            if (tile > 4095) { tile = 4095; }
            uint32_t rowsPerCore = CeilDiv(t.rows, t.blockDim ? t.blockDim : 1);
            CheckCopyParams(name, t, esz, t.nIn[k], cols, Min(tile, rowsPerCore ? rowsPerCore : 1));
        }
    }
    Out o1(total); Strat1(t, esz, o1); Verify(name, "列切", t, o1, ref);

    // ── 按列切的负载：核不该白站着 ──
    //
    // 每一路只用它自己撑得起的 coresK 个核。**不轮转起点的话**，每一路都从 0 号核开始，
    // 于是高编号的核一点活都没有 —— 覆盖和不重叠仍然成立，Verify 查不出来。
    // 所以单独查一条：各路的 coresK 之和够铺满所有核时，**每个核都得分到东西**。
    {
        const uint32_t bn = t.blockDim ? t.blockDim : 1;
        uint32_t alignEl = BLOCK_BYTES / esz; if (alignEl == 0) { alignEl = 1; }
        uint64_t sumCores = 0;
        for (uint32_t k = 0; k < t.numIn; ++k) {
            if (t.nIn[k] == 0) { continue; }
            uint32_t ck = t.nIn[k] / (t.colBurstEl ? t.colBurstEl : 1);
            if (ck == 0) { ck = 1; }
            if (ck > bn) { ck = bn; }
            sumCores += ck;
        }
        if (sumCores >= bn && bn > 1) {
            std::vector<uint64_t> load(bn, 0);
            for (size_t i = 0; i < total; ++i) {
                if (o1.owner[i] >= 0) { load[(size_t)o1.owner[i]] += 1; }
            }
            uint32_t idle = 0;
            for (uint32_t b = 0; b < bn; ++b) { if (load[b] == 0) { ++idle; } }
            CHECK(idle == 0, "%s/列切：各路 coresK 之和 %llu ≥ 核数 %u，却有 %u 个核没分到活 "
                             "—— 起点没轮转？", name, (unsigned long long)sumCores, bn, idle);
        }
    }

    uint32_t alignBad = 0, pitchBad = 0;
    Out o2(total); Strat2(t, esz, o2, alignBad, pitchBad);
    if (o2.owner[0] >= 0 || o2.owner[0] == -2) { Verify(name, "拼装", t, o2, ref); }

    // ④ 向量拼装路只在 host 真的选了它时回放 —— 它的 UB 账、暂存布局都依赖
    //    host 定的 bands，硬套在别的策略上没有意义。
    if (t.strategy == 3) {
        uint32_t ubBad = 0, stAlignBad = 0, gmBad = 0, planBad = 0, trimBad = 0;
        uint64_t readFull = 0, readTrim = 0;
        Out o3(total);
        Strat3(t, esz, o3, ubBad, stAlignBad, gmBad, planBad, trimBad, readFull, readTrim);
        // 默认（不裁剪）时这条断言也必须成立 —— 它就是"整行读"的定义
        //（每个相交路的整行都在槽里，没有任何"带外真数据"这个说法）。
        CHECK(trimBad == 0, "%s/向量拼装：搬进来的字节集合不对（%u 处）", name, trimBad);
        CHECK(!t.trimOn || readTrim <= readFull,
              "%s/向量拼装：裁剪后读得更多了（%llu > %llu）", name,
              (unsigned long long)readTrim, (unsigned long long)readFull);
        Verify(name, "向量拼装", t, o3, ref);

        /* 直写路（`S9_DIRECT=1`）走同一份 tiling，所以同一个用例顺带验它。
         * **它必须产出逐字节相同的输出** —— 两条路只差"拼不拼装"。 */
        for (uint32_t mode = 1; mode <= 2; ++mode) {
            uint32_t dUb = 0, dGm = 0, nbMin = 0, fell = 0, tgMax = 0;
            Out od(total);
            StratDirect(t, esz, od, dUb, dGm, nbMin, mode, fell, tgMax);
            CHECK(dUb == 0, "%s/直写%u：UB 槽放不下（%u 处）—— 真机上是写出 UB 区域",
                  name, mode, dUb);
            CHECK(dGm == 0, "%s/直写%u：DMA 跨步/源范围不成立（%u 处）", name, mode, dGm);
            /* **退回是按核发生的**：只要有一个核退回，这一趟的输出就不完整，
             * 此时该由 Gather 路补齐（kernel 的调用点就是这么写的），
             * 所以只在"一个核都没退回"时才比对输出。 */
            if (fell == 0) { Verify(name, "直写", t, od, ref); }
            CHECK(nbMin >= 1, "%s/直写%u：算出 nb=0，槽数下限被破坏", name, mode);
        }
        CHECK(ubBad == 0, "%s/向量拼装：暂存区越界 %u 次（bands=%u）", name, ubBad, t.bands);
        CHECK(stAlignBad == 0, "%s/向量拼装：%u 条路的暂存起点不是 32 的倍数 —— "
                               "DataCopyPad 的 UB 侧目的地址硬要求 32B 对齐",
              name, stAlignBad);
        CHECK(gmBad == 0, "%s/向量拼装：搬运参数推进量不对（%u 处）", name, gmBad);
        CHECK(planBad == 0, "%s/向量拼装：每带计划与真扫一遍不一致（%u 条带）—— "
                            "开着 S9_BANDPLAN 会漏掉整条路", name, planBad);
        // **没有核空转。** 带数不再要求整除核数之后，这条不再是"构造上显然"的了：
        // 余数分给前几条带各多一组，映射写错就会有核拿不到 (带,组) 对，或者
        // 某一组分到 0 行。原先那条 `blockDim % bands == 0` 的断言保护的就是这个性质，
        // 放开约束就得把性质本身钉住。
        {
            std::vector<uint64_t> load((size_t)t.blockDim, 0);
            for (size_t i = 0; i < o3.owner.size(); ++i) {
                if (o3.owner[i] >= 0 && (uint32_t)o3.owner[i] < t.blockDim) {
                    load[(size_t)o3.owner[i]] += 1;
                }
            }
            uint32_t idle = 0;
            for (uint32_t b = 0; b < t.blockDim; ++b) { if (load[b] == 0) { ++idle; } }
            CHECK(idle == 0, "%s/向量拼装：bands=%u blockDim=%u 下有 %u 个核没分到活",
                  name, t.bands, t.blockDim, idle);
        }
        // 带数**不必**整除核数（那条约束把带数卡得太死，见 s9_tiling_calc.h）；
        // 该保证的是「每条带至少分到一组」，也就是 bands ≤ blockDim。
        CHECK(t.bands >= 1 && t.bands <= t.blockDim,
              "%s/向量拼装：bands=%u 超过 blockDim=%u，会有带分不到核",
              name, t.bands, t.blockDim);
    }

    // ⑤ 输出线性切分（strategy 5）只在 host 真的选了它时回放。
    //    它只接 rows==1，所以行下标恒为 0；要验的是"区间无缝"与"夹取同源"。
    if (t.strategy == 5) {
        CHECK(t.rows == 1, "%s：strategy 5 只该接 rows==1，实得 rows=%u", name, t.rows);
        uint32_t gapBad = 0;
        Out o5(total);
        Strat5(t, esz, o5, gapBad);
        CHECK(gapBad == 0, "%s/线性切分：核间列区间不无缝（%u 处）—— 会漏写或重复写且不报错",
              name, gapBad);
        Verify(name, "线性切分", t, o5, ref);
    }

    // ③ host 选了拼装路时，对齐前提必须成立
    if (t.strategy == 2) {
        CHECK(alignBad == 0, "%s：host 选了拼装路，但有 %u 个 offOut·esz 不是 32 的倍数 —— "
                             "真机上就是往没对齐的 UB 地址搬", name, alignBad);
        CHECK(pitchBad == 0, "%s：拼装路的 UB 行距推进不等于 pitch（%u 处）", name, pitchBad);
    }
}

/* 每个用例按**两种核数**各跑一遍。
 *
 * 评测机 910B1 有 48 个 AIV、本机 910B4 只有 40 —— 核数一变，
 * 分带（bands 必须整除 blockDim）、按行/按列的判据、按输入切核的门槛
 * 会**同时**挪动好几条，而这些配置在本机上一次也跑不到。
 * 孪生是唯一能在动手前覆盖 48 核那一半的地方。 */
static void RunCase(const char *name, uint32_t rows, std::vector<uint32_t> nIns,
                    uint32_t esz)
{
    RunCaseCores(name, rows, nIns, esz, s9::AIV_MAX);        // 本机档：40 核
    RunCaseCores(name, rows, nIns, esz, s9::CONCAT_MAX_CORES);  // 评测机档
}

/* ═══ 列带边界的两条硬不变式：**恰好覆盖、互不重叠** ═══
 *
 * 分带的边界公式在 host（BandRange）、kernel（Init）、孪生三处都要用到。
 * 只要有一处算出不同的区间，核之间就会互相写对方的地盘 —— 结果错，
 * 而且**不报错**（DataCopyPad 对 GM 落点没有对齐或越界检查）。
 * 所以这里穷举一遍：把每条带的区间排序拼起来，必须**严丝合缝**等于 [0, nOut)。
 *
 * 顺带量一下**均衡度**（最宽带 / 最窄非空带）。改成均衡切分之前，
 * nOut=16384、bands=10、grain=256 是 1792 : 256 ＝ **7.0**；
 * 均衡之后是 1792 : 1536 ＝ 1.17。这个比值直接就是被浪费掉的核时。
 */
/* ═══ 边界路裁剪（`S9_TRIM`）的**地址级**验证 ═══
 *
 * 复核把「只裁剪边界路」列为第三阶段，并明确要求：**"读到的字节集合恰等于原集合里
 * 相交的那部分"**。这一条只能按地址验 —— 只比最终输出抓不到它：多读的字节被丢掉、
 * 输出照样是对的。
 *
 * 对同一个形状跑三遍：
 *   ① `trimOn=0`：现状（边界路整行入暂存）；
 *   ② `trimOn=1`：只裁 kernel 的读，**UB 账与 tileG 一个字不变** —— 于是
 *      差分只可能来自"少读字节"（这是第一次上板 A/B 的那一档）；
 *   ③ `trimOn=2`：连 UB 账一起裁（tileG 会变大，必须**单独**一次 A/B）。
 *
 * 三遍的输出都必须与参考一致，且 ② 的暂存内容必须逐元素等于「① 的集合 ∩ 本带」。 */
uint32_t g_trimCases = 0, g_trimApplied = 0;
uint64_t g_trimFull = 0, g_trimTrimmed = 0;

static void TrimCompare(const char *name, uint32_t rows, const std::vector<uint32_t> &nIns,
                        uint32_t esz, uint32_t maxCores)
{
    std::vector<std::vector<int64_t>> sh;
    for (uint32_t n : nIns) { sh.push_back({(int64_t)rows, (int64_t)n}); }
    std::vector<const int64_t *> sp;
    std::vector<int64_t> rk;
    for (auto &v : sh) { sp.push_back(v.data()); rk.push_back(2); }

    s9::ConcatTiling t0{};
    s9::ConcatKnobs kn0;
    kn0.maxCores = maxCores;
    kn0.trimOn   = 0;
    if (!s9::CalcConcat(sp.data(), rk.data(), (uint32_t)nIns.size(), 1, esz, t0, kn0)) {
        return;
    }
    if (t0.strategy != 3 || t0.rows == 0 || t0.nOut == 0) { return; }
    const size_t total = (size_t)t0.rows * t0.nOut;
    std::vector<int64_t> ref(total, -1);
    Ref(t0, ref);

    // ① 现状
    Out a(total);
    uint32_t ub = 0, al = 0, gm = 0, pl = 0, tb = 0;
    uint64_t rFull = 0, rTrim0 = 0;
    Strat3(t0, esz, a, ub, al, gm, pl, tb, rFull, rTrim0);
    Verify(name, "裁剪前", t0, a, ref);
    CHECK(tb == 0, "%s：裁剪前（整行读）的暂存集合就不对（%u 处）", name, tb);

    // ② 只裁读：tiling 其余字段逐字节相同 ⇒ tileG / stageSpan 不变
    s9::ConcatTiling t1 = t0;
    t1.trimOn = 1;
    Out b(total);
    uint32_t ub1 = 0, al1 = 0, gm1 = 0, pl1 = 0, tb1 = 0;
    uint64_t rFull1 = 0, rTrim1 = 0;
    Strat3(t1, esz, b, ub1, al1, gm1, pl1, tb1, rFull1, rTrim1);
    Verify(name, "裁剪后（UB 账不变）", t1, b, ref);
    CHECK(tb1 == 0, "%s：裁剪后搬进来的字节集合 ≠「原来那批 ∩ 本带」（%u 处）", name, tb1);
    /* "整行读会搬多少"两遍必须**逐字节相同**（那是同一个物理量）；
     * "裁剪后搬多少"在关闭裁剪那一遍就等于整行（`realN = nIn`），所以不能拿来比。 */
    CHECK(rFull1 == rFull, "%s：两遍的「整行读」字节数对不上（%llu vs %llu）", name,
          (unsigned long long)rFull, (unsigned long long)rFull1);
    CHECK(rTrim1 <= rFull1, "%s：裁剪后读得更多了（%llu > %llu）", name,
          (unsigned long long)rTrim1, (unsigned long long)rFull1);
    CHECK(ub1 == 0 && al1 == 0 && gm1 == 0, "%s：裁剪后出现越界/对齐问题", name);
    // **这条检查不能是空的**：统计有多少个用例真的发生了裁剪、一共少搬多少字节。
    // 全是"整段在带内"的话（比如 bands=1），这个函数什么也没验到，必须看得出来。
    ++g_trimCases;
    g_trimFull += rFull1;
    g_trimTrimmed += rTrim1;
    if (rTrim1 < rFull1) { ++g_trimApplied; }
    /* **逐形状报一行。** 只给一个总数是不够的：评测有 5 个 case，而"边界裁剪"
     * 能不能起作用**完全取决于形状**（路有多宽、带有多窄、裁完是不是真的更省）。
     * 这一行就是"这个形状上裁剪到底可不可用、能省多少字节"的答案。 */
    printf("    %-30s %-4s 读完 %10llu → %10llu  (%+6.2f%%)\n", name,
           (rTrim1 < rFull1) ? "裁" : "不裁",
           (unsigned long long)rFull1, (unsigned long long)rTrim1,
           rFull1 ? -100.0 * (double)(rFull1 - rTrim1) / (double)rFull1 : 0.0);

    // ③ 连 UB 账一起裁
    s9::ConcatTiling t2{};
    s9::ConcatKnobs kn2;
    kn2.maxCores = maxCores;
    kn2.trimOn   = 2;
    if (s9::CalcConcat(sp.data(), rk.data(), (uint32_t)nIns.size(), 1, esz, t2, kn2)) {
        if (t2.strategy == 3 && t2.rows == t0.rows && t2.nOut == t0.nOut) {
            Out c(total);
            uint32_t ub2 = 0, al2 = 0, gm2 = 0, pl2 = 0, tb2 = 0;
            uint64_t rF2 = 0, rT2 = 0;
            Strat3(t2, esz, c, ub2, al2, gm2, pl2, tb2, rF2, rT2);
            Verify(name, "裁剪后（UB 账也裁）", t2, c, ref);
            CHECK(tb2 == 0, "%s：账也裁那一版搬进来的集合不对（%u 处）", name, tb2);
            CHECK(ub2 == 0, "%s：账也裁那一版 UB 越界 %u 次", name, ub2);
        }
    }
}

/* ═══ 二维分核（`S9_GRID`）：覆盖、不重叠、没有空核 ═══
 *
 * 这条路只在 `dim` 在中间轴、`rows` 远小于核数时触发，而**发布默认是关的**
 *（`gridBurstBytes = 0`）—— 于是上面所有用例里 `colParts` 恒为 1，等于没验。
 * 这里显式打开它再回放一遍 `Strat0`。
 *
 * 要钉住的是「核之间互相写对方地盘」这一类 —— `DataCopyPad` 对 GM 落点
 * 既不查对齐也不查越界，**错了不报错**。 */
static void CheckColGrid()
{
    struct C { const char *name; uint32_t rows; uint32_t numIn; uint32_t wide; uint32_t esz; };
    const C cs[] = {
        {"中间轴 16 行 33 路", 16, 33, 15515, 2},
        {"中间轴  8 行 33 路",  8, 33, 62060, 2},
        {"中间轴 24 行 17 路", 24, 17, 30000, 4},
        {"中间轴 47 行 19 路", 47, 19, 28000, 2},
        {"中间轴  3 行  5 路",  3,  5, 99000, 1},
        {"路宽参差",           12, 21,  8000, 2},
    };
    for (const C &c : cs) {
        std::vector<uint32_t> lens(c.numIn);
        // 路宽参差：在 [wide/2, wide) 之间摆，模拟 [0, max_step] 均匀分布的效果
        for (uint32_t i = 0; i < c.numIn; ++i) {
            lens[i] = c.wide / 2 + (c.wide / 2) * (i * 37 % 97) / 97;
            lens[i] = (lens[i] / 8) * 8;                 // 让偏移不总是对齐
            if (lens[i] == 0) { lens[i] = 8; }
        }
        for (uint32_t cores : {40u, 48u}) {
            s9::ConcatTiling t{};
            s9::ConcatKnobs kn{}; kn.maxCores = cores; kn.gridBurstBytes = 4096;
            if (!s9::CalcConcatLens(c.rows, lens.data(), c.numIn, c.esz, t, kn)) { continue; }
            if (t.strategy != 0) { continue; }           // 只有按行切才走二维分核
            CHECK(t.colParts >= 1, "%s：colParts=%u", c.name, t.colParts);
            /* **只有真的开了列组才要求这个恒等式。** `rows ≥ coreCap` 时
             * blockDim 被夹在 coreCap 上、colParts 恒为 1 —— 那是旧行为，
             * 不该被这条断言管住（`rows=47` 在 40 核上就是这种）。 */
            if (t.colParts > 1) {
                CHECK(t.blockDim == c.rows * t.colParts,
                      "%s（%u 核）：blockDim=%u ≠ rows×colParts=%u×%u",
                      c.name, cores, t.blockDim, c.rows, t.colParts);
            }
            CHECK(t.blockDim <= cores, "%s（%u 核）：blockDim=%u 超过核数",
                  c.name, cores, t.blockDim);
            std::vector<int64_t> ref((size_t)t.rows * t.nOut, -1);
            Ref(t, ref);
            Out o(ref.size());
            Strat0(t, c.esz, o);
            Verify(c.name, "二维分核", t, o, ref);
            // **没有空核**：每个 block 都要写到至少一个元素，否则 colParts 白开
            std::vector<uint64_t> load(t.blockDim, 0);
            for (size_t i = 0; i < o.owner.size(); ++i) {
                if (o.owner[i] >= 0 && (uint32_t)o.owner[i] < t.blockDim) { ++load[(size_t)o.owner[i]]; }
            }
            uint32_t idle = 0;
            for (uint32_t b = 0; b < t.blockDim; ++b) { if (load[b] == 0) { ++idle; } }
            CHECK(idle == 0, "%s（%u 核）：colParts=%u 下有 %u 个核没分到活",
                  c.name, cores, t.colParts, idle);
        }
    }
}

static void CheckBandPartition()
{
    uint32_t worstNum = 0, worstDen = 1, wN = 0, wB = 0, wG = 0;
    uint32_t combos = 0;
    for (uint32_t grainEl : {8u, 16u, 32u, 256u, 512u}) {
        for (uint32_t nOut : {4u, 100u, 256u, 1000u, 2048u, 3000u, 4096u, 16384u, 65536u}) {
            for (uint32_t bands = 1; bands <= 48; ++bands) {
                uint32_t pos = 0, wMax = 0, wMin = 0xFFFFFFFFu;
                bool bad = false;
                for (uint32_t b = 0; b < bands; ++b) {
                    uint32_t c0 = 0, c1 = 0;
                    s9::BandRange(nOut, bands, grainEl, b, c0, c1);
                    if (c0 != pos) { bad = true; break; }   // 有缝或有重叠
                    if (c1 < c0) { bad = true; break; }
                    if (c1 > c0) {
                        const uint32_t w = c1 - c0;
                        if (w > wMax) { wMax = w; }
                        if (w < wMin) { wMin = w; }
                    }
                    pos = c1;
                }
                CHECK(!bad && pos == nOut,
                      "分带不严丝合缝：nOut=%u bands=%u grain=%u 拼到 %u", nOut, bands, grainEl, pos);
                ++combos;
                if (wMax && wMin != 0xFFFFFFFFu && (uint64_t)wMax * worstDen > (uint64_t)worstNum * wMin) {
                    worstNum = wMax; worstDen = wMin; wN = nOut; wB = bands; wG = grainEl;
                }
            }
        }
    }
    printf("  列带划分：%u 个组合全部严丝合缝；最不均的一组 %u:%u = %.2f 倍"
           "（nOut=%u bands=%u grain=%u）\n",
           combos, worstNum, worstDen, (double)worstNum / (double)worstDen, wN, wB, wG);
}

// 密排必须改变布局并保持每个输出元素来源和写入所有者正确；含空路、尾超行和多轮缓冲。
static void CheckDense()
{
    uint32_t cases = 0;
    for (uint32_t esz : {1u, 2u, 4u}) {
        for (uint32_t width : {2u, 3u, 7u, 15u, 18u, 25u, 31u, 32u, 51u, 52u, 63u}) {
            for (uint32_t rows : {1u, 7u, 41u, 4101u}) {
                for (uint32_t cores : {1u, 3u}) {
                    const uint32_t lens[] = {0, 1, 0, width - 1, 0};
                    s9::ConcatKnobs kn;
                    kn.maxCores = cores;
                    kn.bandMinBytes = 1u << 20; // 保持单带，隔离输出布局。
                    kn.linearOn = false;
                    kn.denseOn = 2;
                    s9::ConcatTiling t{};
                    CHECK(s9::CalcConcatLens(rows, lens, 5, esz, t, kn), "dense tiling failed");
                    CHECK(t.denseOn && t.bands == 1, "dense flag/bands lost");
                    const auto lay = s9::PlanConcatStage(width, esz, 5, t.maxIn, t.slackIn,
                        t.gBuf, t.aBuf, 0, rows, 0, 2, 0, t.bands == 1);
                    CHECK(lay.fits && lay.outPitch == width * esz, "dense output pitch is padded");
                    CHECK(lay.asmSpan >= (uint64_t)lay.tileG * width * esz && lay.asmSpan % 32 == 0,
                          "dense output capacity/alignment failed");
                    std::vector<int64_t> ref((size_t)rows * width);
                    Ref(t, ref);
                    Out out((size_t)rows * width);
                    uint32_t ub = 0, al = 0, gm = 0, pl = 0, tr = 0;
                    uint64_t rf = 0, rt = 0;
                    Strat3(t, esz, out, ub, al, gm, pl, tr, rf, rt);
                    Verify("dense", "super-row", t, out, ref);
                    CHECK(ub + al + gm + pl + tr == 0,
                          "dense bounds: ub=%u al=%u gm=%u pl=%u tr=%u", ub, al, gm, pl, tr);
                    ++cases;
                }
            }
        }
    }
    printf("dense cases: %u\n", cases);
}

int main()
{
    CheckDense();
    CheckBandPartition();
    CheckColGrid();
    printf("── Concat 主机孪生 ─────────────\n");
    /* ══ 边界路裁剪（S9_TRIM）：地址级验证 ══
     *
     * **形状面必须铺够。** 只量一个形状说明不了什么：评测有 5 个 case，
     * 而"裁剪能不能起作用"完全取决于形状 —— 路有多宽（窄路补到 32B 反而更大）、
     * 一条带切开几条路、裁完到底省多少字节。所以这里把**贴近赛题 5 个 case 的
     * 形状族**都过一遍，而且**每个形状跑 40 核与 48 核两遍**：
     * 本机是 910B4/40 核，评测机是 910B1/48 核，而**带数 = blockDim 的因数**
     * 会随核数变（40→8 带、48→16 带），于是"每条带切开几条路"完全不同。
     * 耗时只能靠提交去量，**但形状派生的这一半在孪生里就算得出来**。 */
    {
        struct TrimCase {
            const char *name; uint32_t rows; std::vector<uint32_t> lens; uint32_t esz;
        };
        std::vector<TrimCase> tcs;
        auto addc = [&](const char *nm, uint32_t rows, std::vector<uint32_t> lens, uint32_t esz) {
            TrimCase c; c.name = nm; c.rows = rows; c.lens = std::move(lens); c.esz = esz;
            tcs.push_back(std::move(c));
        };
        addc("Case1 128x256 f16 9路",      128,  std::vector<uint32_t>(9, 32),   2);
        addc("128x256 f16 64路(细分)",      128,  std::vector<uint32_t>(64, 4),   2);
        addc("2024x3000 f16 88路",         2024, std::vector<uint32_t>(88, 34),  2);
        addc("2024x3000 f32 88路",         2024, std::vector<uint32_t>(88, 34),  4);
        addc("Case4族 4096x4096 f16 123路", 4096, std::vector<uint32_t>(123, 33), 2);
        addc("Case5族 4096x16384 f16 246路",4096, std::vector<uint32_t>(246, 66), 2);
        addc("2048x2048 f16 58路",         2048, std::vector<uint32_t>(58, 35),  2);
        addc("int8 2048x2048 248路",       2048, std::vector<uint32_t>(248, 8),  1);
        addc("int32 512x2048 248路",        512, std::vector<uint32_t>(248, 8),  4);
        addc("参差 97路含空片",             256,  [] {
            std::vector<uint32_t> v; for (int i = 0; i < 97; ++i) {
                v.push_back((uint32_t)((i * 37) % 64)); } return v; }(), 2);
        for (const TrimCase &c : tcs) {
            for (uint32_t cores : {40u, 48u}) {
                char nm[96];
                snprintf(nm, sizeof(nm), "%s @%u核", c.name, cores);
                TrimCompare(nm, c.rows, c.lens, c.esz, cores);
            }
        }
    }
    printf("  边界路裁剪：%u 个用例里 %u 个真的裁到了边界路；"
           "读完的字节 %llu → %llu（−%.1f%%）\n",
           g_trimCases, g_trimApplied, (unsigned long long)g_trimFull,
           (unsigned long long)g_trimTrimmed,
           g_trimFull ? 100.0 * (double)(g_trimFull - g_trimTrimmed) / (double)g_trimFull : 0.0);
    CHECK(g_trimApplied > 0, "没有一个用例真的裁到边界路 —— 这个检查是空的");
    for (uint32_t esz : {4u, 2u, 1u}) {
        RunCase("2 路等宽", 1024, {512, 512}, esz);
        RunCase("3 路不等宽", 512, {1000, 2000, 3000}, esz);
        RunCase("含空片", 256, {0, 128, 0, 256}, esz);
        RunCase("单路", 64, {777}, esz);
        RunCase("行少列多（逼列切）", 2, {100000, 100000}, esz);
        RunCase("行多列少", 100000, {3, 5}, esz);
        RunCase("对齐偏移（拼装可用）", 4096, {2000, 2000, 2000}, esz);
        RunCase("非对齐偏移", 4096, {1000, 1000, 1000}, esz);
        RunCase("rows=1", 1, {4096, 4096}, esz);
        RunCase("16 路", 128, {64, 64, 64, 64, 64, 64, 64, 64,
                               64, 64, 64, 64, 64, 64, 64, 64}, esz);
        // **路长悬殊** —— 按列切时每一路各自定核数（coresK 分别是 1/23/1/11/1/18），
        // 覆盖与不重叠都得成立。没有这一档的话，上面那些用例的 coresK 要么全是 1、
        // 要么全相等，"按路分核"这条新逻辑等于没被测到。
        RunCase("路长悬殊（逼按路分核）", 1, {3000, 192000, 3000, 96000, 6000, 150000}, esz);
        // ── 列带分核专用 ──
        // 多路 ＋ 中等宽度 ＝ host 会选向量拼装并把列切成多条带，
        // 于是**每条带都会切开若干条路**（边界路要跨步读、行步长与内部路不同）。
        // 没有这两档的话 bands 恒为 1，分带那套算术等于没被测到。
        {
            std::vector<uint32_t> many;
            for (int i = 0; i < 123; ++i) { many.push_back(33); }
            RunCase("123 路等宽（逼分带）", 512, many, esz);
            // **行多、路中等** —— 这是 int8 真正会走到拼装路的几何
            //（按输入切核的门槛是 rpc·avgN·esz < 512，行多时它不成立）。
            // slack 的 half 副本漏账就在这里溢出：实算 194400 字节 > UB 的 188416。
            // 少了这一档，UB 总量断言在其余用例上都富余，等于没测
            //（真机上 I5 那档已经在悄悄超 UB，只是碰巧没崩、结果还对）。
            std::vector<uint32_t> narrow(100, 20u);
            RunCase("100 路 · 行多（逼 UB 总量）", 2048, narrow, esz);
            std::vector<uint32_t> ragged;
            // 长度参差 ＋ 含空片：带边界会落在各种奇怪的位置
            for (int i = 0; i < 97; ++i) { ragged.push_back((uint32_t)((i * 37) % 64)); }
            RunCase("97 路参差含空片（逼分带）", 256, ragged, esz);
            // **窄路扎堆**：前 200 路宽度 1，后面几路很宽。
            // 这种形状会让某一条带里的路数远超「按均分估计」的上界，
            // 专门用来验 maxInBand 是精确数出来的而不是估的
            //（估小了就是暂存余量不够，真机 UB 越界）。
            {
                std::vector<uint32_t> clust;
                for (uint32_t i = 0; i < 200; ++i) { clust.push_back(1); }
                for (uint32_t i = 0; i < 8; ++i) { clust.push_back(456); }
                RunCase("200 路宽度 1 扎堆 ＋ 8 路很宽", 256, clust, esz);
            }
        }
    }
    // ── 按输入切核（strategy 4）的分区 ──
    //
    // 每一路归哪个核，由它的**起始列**落在哪一段决定：owner = off·bd/nOut。
    // 必须验三条：
    //   ① **每一路恰好归一个核** —— 归两个核就是重复写（结果未定），
    //      一个都不归就是那段输出留着 at::empty 的垃圾；
    //   ② 各核拿到的是**连续的一段路**（kernel 是顺序扫过去的，不连续会漏）；
    //   ③ 负载按列数分得不算太差 —— 按路数分会很不均（各路长短差几十倍）。
    {
        std::mt19937_64 rng(20260906);
        uint32_t worstRatio = 0, checked = 0;
        for (uint32_t total : {1024u, 4096u, 8192u})
        for (uint32_t step : {16u, 32u, 64u}) {
            std::vector<uint32_t> lens;
            uint32_t remain = total;
            while (remain) {
                uint32_t up = remain < step ? remain : step;
                uint32_t pick = (uint32_t)(rng() % (up + 1));
                lens.push_back(pick); remain -= pick;
            }
            if (lens.size() > s9::CONCAT_MAX_IN) { continue; }
            s9::ConcatTiling t{};
            // **按 strategy 4 的新定位来测：它是「拼装路接不住」时的退路。**
            // 加上写块下限（IN_SPLIT_MIN_WRITE）之后，拼装路可用时窄写块的形状
            // 一律让给它，所以这里要在 gatherOn=false 下才够得着 strategy 4 ——
            // 那正是它在线上唯一会被选中的处境。
            s9::ConcatKnobs kn;
            kn.gatherOn = false;
            if (!s9::CalcConcatLens(128, lens.data(), (uint32_t)lens.size(), 2, t, kn)) {
                continue;
            }
            if (t.strategy != 4) { continue; }
            ++checked;
            uint32_t bd = t.blockDim;
            std::vector<int> owner(t.numIn, -1);
            std::vector<uint64_t> load(bd, 0);
            uint32_t off = 0;
            for (uint32_t i = 0; i < t.numIn; ++i) {
                uint32_t o = (uint32_t)((uint64_t)off * bd / t.nOut);
                if (o >= bd) { o = bd - 1; }
                owner[i] = (int)o;
                load[o] += t.nIn[i];
                off += t.nIn[i];
            }
            // ① 每路恰好一个核（owner 数组按构造就是唯一的，这里查值域）
            uint32_t badOwner = 0;
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (owner[i] < 0 || owner[i] >= (int)bd) { ++badOwner; }
            }
            CHECK(badOwner == 0, "按输入切：%u 路的 owner 越界", badOwner);
            // ② 连续性：owner 序列必须单调不减
            uint32_t nonMono = 0;
            for (uint32_t i = 1; i < t.numIn; ++i) {
                if (owner[i] < owner[i - 1]) { ++nonMono; }
            }
            CHECK(nonMono == 0, "按输入切：owner 序列有 %u 处回退，各核拿到的路不连续",
                  nonMono);
            // ③ 覆盖：各核负载之和等于 nOut
            uint64_t sum = 0;
            for (uint32_t c = 0; c < bd; ++c) { sum += load[c]; }
            CHECK(sum == t.nOut, "按输入切：各核负载之和 %llu ≠ nOut %u",
                  (unsigned long long)sum, t.nOut);
            uint64_t mx = 0;
            for (uint32_t c = 0; c < bd; ++c) { if (load[c] > mx) { mx = load[c]; } }
            uint32_t ratio = (uint32_t)(mx * bd * 100 / (t.nOut ? t.nOut : 1));
            if (ratio > worstRatio) { worstRatio = ratio; }
        }
        CHECK(checked >= 3, "只有 %u 个组合走到按输入切核 —— 检查是空的", checked);
        // 最重的核不该超过平均的 3 倍（按列数分的话，各路长短的差异会被摊掉）
        CHECK(worstRatio <= 300, "按输入切：最重的核是平均的 %.1f 倍，太不均",
              worstRatio / 100.0);
        printf("  按输入切核：%u 个组合，最重的核是平均的 %.2f 倍\n",
               checked, worstRatio / 100.0);
    }

    printf("\n%d 项通过", g_pass);
    if (g_fail) { printf("，\033[31m%d 项失败\033[0m\n", g_fail); return 1; }
    printf("，全部通过\n");
    return 0;
}
