/* 向量拼装路（strategy 3）的 UB 逐块布局自检。
 *
 * 这个文件存在的理由是一个**没能在本机复现、但算术上确定的**缺陷：
 *   `stageSpan_ = tileG_*rowUse + slack` 没有逐块 32B 对齐，
 *   于是第二块（第三块…）暂存的起点落在非 32B 边界上，
 *   而 DataCopyPad 的 UB 侧目的地址要求 32B 对齐。
 *
 * 所以这里干三件事：
 *   ① 把那个反例（fp16、`(4096,63)×4 ＋ (4096,4)`、40 核）钉死成绝对地址断言；
 *   ② 穷举一批配置，验每一块的起点、每一条路的段起点、每一块分配**都**对齐，
 *      并且总量真的算进了序幕与指针暂存；
 *   ③ 验"放不下就回缩、缩到 1 行还放不下就判不可行" —— 不能靠强制 tileG=1 掩盖。
 *
 * 算术本体在 common/s9_concat_layout.h，host 与 kernel 共用同一份。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include <cstdint>
#include <string_view>
#include <vector>
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

/* 旧的跨度公式 —— **留在这里当反面教材**：它算出来的第二块起点不是 32B 倍数。 */
static uint32_t OldStageSpan(uint32_t tileG, uint32_t rowUse, uint32_t slack)
{
    return tileG * rowUse + slack;
}

/* ═══ ① 文档里那个反例 ═══
 *
 * fp16、dim=1、`(4096,63)×4 ＋ (4096,4)`、40 核、默认参数。
 * host 选 strategy=3、bands=1、aBuf=2；tileG=51、rowUse=764、slack=160
 * ⇒ 旧跨度 39124（%32 = 20），第二块暂存起点 39124 —— 落在 20 mod 32 上。
 * 新跨度 39136，两块起点 0 / 39136 都是 32 的倍数。 */
static void CheckCounterexample()
{
    std::vector<uint32_t> lens{63, 63, 63, 63, 4};
    s9::ConcatTiling t{};
    s9::ConcatKnobs kn;
    kn.maxCores = 40;
    const bool ok = s9::CalcConcatLens(4096, lens.data(), (uint32_t)lens.size(), 2, t, kn);
    CHECK(ok, "反例：CalcConcatLens 应当成功");
    if (!ok) { return; }
    CHECK(t.strategy == 3, "反例：host 应当选向量拼装路（实为 strategy=%u）", t.strategy);
    CHECK(t.aBuf == 2, "反例：aBuf 应为 2（实为 %u）", t.aBuf);
    CHECK(t.bands == 1, "反例：带数应为 1（实为 %u）", t.bands);
    if (t.strategy != 3) { return; }

    const uint32_t bandCols = s9::BandColsMax(t.nOut, t.bands, t.bandGrain, t.bandBal);
    const uint32_t grp      = t.blockDim / (t.bands ? t.bands : 1);
    const uint32_t myRows   = grp ? CeilDiv(t.rows, grp) : t.rows;
    s9::ConcatStageLayout L = s9::PlanConcatStage(bandCols, 2, t.numIn, t.maxIn, t.slackIn,
                                                 t.gBuf, t.aBuf, /*tileGMax=*/0, myRows,
                                                 /*tblK=*/t.tblK, /*dense=*/false);
    CHECK(L.fits, "反例：这个配置应当放得下（why=%s）", s9::WhyText(L.why));
    CHECK(L.tileG == 51, "反例：tileG 应为 51（实为 %u）", L.tileG);
    CHECK(L.rowUse == 764, "反例：rowUse 应为 764（实为 %u）", L.rowUse);
    CHECK(L.slack == 160, "反例：slack 应为 160（实为 %u）", L.slack);

    const uint32_t oldSpan = OldStageSpan(L.tileG, L.rowUse, L.slack);
    CHECK(oldSpan == 39124, "反例：旧跨度应为 39124（实为 %u）", oldSpan);
    CHECK(oldSpan % BLOCK_BYTES == 20 % BLOCK_BYTES && oldSpan % BLOCK_BYTES != 0,
          "反例：旧跨度应当**不**是 32 的倍数（实为 %%32 = %u）—— 这一条就是在钉那个缺陷",
          oldSpan % BLOCK_BYTES);
    CHECK(L.stageSpan == 39136, "反例：新跨度应为 39136（实为 %u）", L.stageSpan);
    CHECK(L.stageSpan % BLOCK_BYTES == 0, "反例：新跨度应当是 32 的倍数");
    for (uint32_t b = 0; b < L.gBuf; ++b) {
        CHECK(L.StgBase(b) % BLOCK_BYTES == 0,
              "反例：第 %u 块暂存起点 %llu 不是 32 的倍数", b,
              (unsigned long long)L.StgBase(b));
        CHECK(L.StgBlockOk(b), "反例：第 %u 块暂存越过了队列", b);
    }
    /* 这里的差正好是补齐到 32 的 12 字节 —— 换句话说，多花 12 字节 UB 换掉
     * 一个"数据搬错地方但什么都不报"的可能。 */
    CHECK(L.stageSpan - oldSpan == 12, "反例：补齐量应为 12 字节（实为 %u）",
          L.stageSpan - oldSpan);
    printf("  反例（fp16 (4096,63)×4＋(4096,4) 40 核）：旧跨度 %u（%%32=%u）→ 新跨度 %u"
           "（tileG=%u rowUse=%u slack=%u）\n",
           oldSpan, oldSpan % BLOCK_BYTES, L.stageSpan, L.tileG, L.rowUse, L.slack);
}

/* ═══ ② 对齐不变式（穷举一批配置）═══
 *
 * 每一块分配的**起点**都要落在 32B 上，每一条入暂存的路的段起点也要 ——
 * 前者由 `InitBuffer` 保证不了（我们手动轮转），后者由 DataCopyPad 硬要求。 */
static void CheckAlignmentInvariants()
{
    uint32_t combos = 0, oldBad = 0, e8Fits = 0, e8Rej = 0;
    for (uint32_t esz : {1u, 2u, 4u, 8u})
    for (uint32_t tileG : {1u, 2u, 3u, 7u, 8u, 51u, 128u, 1000u})
    for (uint32_t gBuf : {1u, 2u, 3u})
    for (uint32_t aBuf : {1u, 2u})
    for (uint32_t bw : {3u, 4u, 20u, 127u, 256u, 1024u, 4096u}) {
        const uint32_t maxNIn = 63;
        const uint32_t nSlack = 5;
        s9::ConcatStageLayout L = s9::LayoutConcatStage(bw, esz, /*numIn=*/5, maxNIn, nSlack,
                                                       gBuf, aBuf, tileG, /*tblK=*/0, /*dense=*/false);
        /* **8 字节 dtype 显式拒绝**：偏移表按行推进，行起点是 j·tblRow·4 字节，
         * 而 esz=8 时 tblRow·4 = outPitch/2 不保证 32 的倍数。本算子只注册了
         * fp16/fp32/int32/int8，所以这是死代码 —— 但"到不了"不等于"可以静默错"，
         * 这里就按显式拒绝测。 */
        if (esz == 8) {
            // 放行就必须满足"表的行起点 32B 对齐"；放不下的必须是显式原因，不能静默。
            if (L.fits) {
                CHECK((L.tblRow * 4) % BLOCK_BYTES == 0,
                      "esz=8：放行了但表的行起点 %u 不是 32 的倍数 —— 正是要拦的那种配置",
                      L.tblRow * 4);
                ++e8Fits;
            } else {
                CHECK(L.why == s9::C9_TABLE_UNALIGNED || L.why == s9::C9_UB_ACCOUNT,
                      "esz=8：不可行的原因应当是表的落点或 UB 实账（实为 %s）",
                      s9::WhyText(L.why));
                ++e8Rej;
            }
            continue;
        }
        ++combos;
        CHECK(L.stageSpan % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：跨度 %u 未对齐",
              bw, esz, tileG, L.stageSpan);
        CHECK(L.asmSpan % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：拼装区块跨度未对齐",
              bw, esz, tileG);
        CHECK(L.offTblBytes % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：偏移表未对齐",
              bw, esz, tileG);
        CHECK(L.dltBytes % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：dlt 表未对齐",
              bw, esz, tileG);
        CHECK(L.castSpan % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：half 副本未对齐",
              bw, esz, tileG);
        CHECK(L.asmGBytes % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：拼装副本未对齐",
              bw, esz, tileG);
        CHECK(L.tdBytes % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：指针暂存未对齐",
              bw, esz, tileG);
        for (uint32_t b = 0; b < gBuf; ++b) {
            CHECK(L.StgBase(b) % BLOCK_BYTES == 0, "bw=%u esz=%u tileG=%u：第 %u 块起点未对齐",
                  bw, esz, tileG, b);
        }
        for (uint32_t b = 0; b < aBuf; ++b) {
            CHECK(L.AsmBase(b) % BLOCK_BYTES == 0,
                  "bw=%u esz=%u tileG=%u：第 %u 块拼装区起点未对齐", bw, esz, tileG, b);
        }
        /* 每一条入暂存的路，段起点也要对齐（kernel 用 alignEl = 32/min(esz,gsz) 累加）。
         * 这里按最坏情况推：一条带最多切开两条路 ⇒ 入暂存的路宽度之和不超 bw + 2·maxNIn。 */
        {
            const uint32_t gsz = (esz == 1) ? 2 : esz;
            const uint32_t alignEl = BLOCK_BYTES / (esz < gsz ? esz : gsz);
            const uint32_t nIns[3] = {bw, maxNIn, maxNIn};
            uint32_t acc = 0, bad = 0;
            for (uint32_t i = 0; i < 3; ++i) {
                if ((acc * esz) % BLOCK_BYTES != 0 || (acc * gsz) % BLOCK_BYTES != 0) { ++bad; }
                acc += AlignUp(tileG * nIns[i], alignEl);
            }
            CHECK(bad == 0, "bw=%u esz=%u tileG=%u：%u 条路的段起点未对齐", bw, esz, tileG, bad);
            CHECK((uint64_t)acc * esz <= L.stageSpan,
                  "bw=%u esz=%u tileG=%u：最坏情况 %llu 字节装不进跨度 %u", bw, esz, tileG,
                  (unsigned long long)acc * esz, L.stageSpan);
        }
        /* 旧跨度在这一组配置里有多经常不对齐 —— 这个计数是"缺陷覆盖度"的度量。 */
        if (OldStageSpan(tileG, L.rowUse, L.slack) % BLOCK_BYTES != 0) { ++oldBad; }
    }
    printf("  对齐不变式：%u 组配置全部通过；其中 %u 组的**旧**跨度不是 32 的倍数"
           "（%.1f%%）；另有 8 字节 dtype %u 组放行、%u 组被显式拒绝\n",
           combos, oldBad, 100.0 * oldBad / (combos ? combos : 1), e8Fits, e8Rej);
    CHECK(combos > 1000, "对齐扫描的组合数太少（%u）", combos);
    CHECK(oldBad > 0, "没有一组配置暴露旧缺陷 —— 这个自检等于没测");
    CHECK(e8Rej > 0 && e8Fits > 0, "8 字节 dtype 的两条分支要都走到（放行 %u / 拒绝 %u）",
          e8Fits, e8Rej);
}

/* ═══ ③ 真实 tiling 下的账目 ═══ */
static void CheckRealTilings()
{
    struct Case { uint32_t rows; std::vector<uint32_t> lens; uint32_t esz; };
    std::vector<Case> cases = {
        {128,  {32, 32, 32, 32, 32, 32, 32, 32, 32}, 2},
        {512,  {1000, 2000, 3000}, 4},
        {512,  {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33}, 2},
        {512,  {1000, 1000, 1000}, 2},
        {4096, {63, 63, 63, 63, 4}, 2},
        {2048, {20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 1},
        {256,  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 2},
        {1,    {4096, 4000, 3200, 800}, 2},
        {128,  {256, 128, 64, 32, 16, 8, 4, 2, 1}, 4},
    };
    // 形状较大的几档：Case4/Case5 那一族
    {
        std::vector<uint32_t> big(123, 33);
        cases.push_back({4096, big, 2});
        std::vector<uint32_t> huge(246, 66);
        cases.push_back({4096, huge, 2});
        std::vector<uint32_t> i8(248, 8);
        cases.push_back({2048, i8, 1});
    }
    uint32_t gathers = 0;
    for (const Case &c : cases) {
        for (uint32_t cores : {s9::AIV_MAX, s9::CONCAT_MAX_CORES}) {
            s9::ConcatTiling t{};
            s9::ConcatKnobs kn;
            kn.maxCores = cores;
            const bool ok = s9::CalcConcatLens(c.rows, c.lens.data(),
                                               (uint32_t)c.lens.size(), c.esz, t, kn);
            CHECK(ok, "rows=%u 路数=%zu esz=%u 核数=%u：CalcConcatLens 应当成功",
                  c.rows, c.lens.size(), c.esz, cores);
            if (!ok) { continue; }
            if (t.strategy != 3) { continue; }        // 只有拼装路才吃这份账
            ++gathers;
            const uint32_t bandCols = s9::BandColsMax(t.nOut, t.bands, t.bandGrain, t.bandBal);
            const uint32_t grp    = t.blockDim / (t.bands ? t.bands : 1);
            const uint32_t myRows = grp ? CeilDiv(t.rows, grp) : t.rows;
            s9::ConcatStageLayout L = s9::PlanConcatStage(
                bandCols, c.esz, t.numIn, t.maxIn, t.slackIn, t.gBuf, t.aBuf, 0, myRows, t.tblK, /*dense=*/false, 0, t.bands == 1);
            CHECK(L.fits, "rows=%u 路数=%zu esz=%u 核数=%u：发布的拼装配置放不下（%s）",
                  c.rows, c.lens.size(), c.esz, cores, s9::WhyText(L.why));
            CHECK(L.tileG >= 1 && L.tileG <= 4095, "rows=%u：tileG=%u 越界", c.rows, L.tileG);
            if (myRows > 0) {
                CHECK(L.tileG <= myRows, "rows=%u：tileG=%u 超过本组行数 %u",
                      c.rows, L.tileG, myRows);
            }
            // 实账必须**含**序幕与指针暂存 —— 这两项以前只有 kernel 那边算了一半
            CHECK(L.totalBytes >= (uint64_t)L.hdBytes + L.tdBytes,
                  "rows=%u：总量没算进序幕/指针暂存", c.rows);
            CHECK(L.totalBytes <= L.limitBytes, "rows=%u：总量超过 UB 上限", c.rows);

            /* ═══ 缩表（`S9_TBL1`）的三条不变式 ═══
             *
             * 发布配置里 `tbl1=0`，所以上面那一整段**验不到新账**。这里对同一个
             * 配置再算一遍 `tbl1=1`，把三件"错了不报错"的事钉住：
             *   ① 表确实只剩 2 行（`tblRow·4·2`）——写错成 1 行，kernel 的工作行
             *      就落在缓冲外，真机 UB 越界且不一定当场炸；
             *   ② 账仍然成立（`fits` 且 `totalBytes ≤ limitBytes`）；
             *   ③ **tileG 不减** —— 缩表的全部意义就是把省下的 UB 变成 tileG/gBuf，
             *      若它反而变小，说明 `perTile`/`fixed` 的搬项搬错了方向。
             * ③ 用 `>=` 而不是 `>`：`myRows` 或 `S9_TILEG` 卡住上限时两者相等是对的。 */
            // K 取几都要成立 —— 1 是最省表的一档，4/8 是折中档
            for (uint32_t kTK : {1u, 4u, 8u}) {
            const s9::ConcatStageLayout L1 = s9::PlanConcatStage(
                bandCols, c.esz, t.numIn, t.maxIn, t.slackIn, t.gBuf, t.aBuf, 0, myRows,
                /*tblK=*/kTK, /*dense=*/false, 0, t.bands == 1);
            const uint32_t rowsWant = (kTK + s9::S9L_TBL_EXTRA_ROWS < L1.tileG)
                                    ? (kTK + s9::S9L_TBL_EXTRA_ROWS) : L1.tileG;
            const uint32_t want = L1.tblRow * 4 * rowsWant;
            CHECK(L1.offTblBytes == want,
                  "rows=%u esz=%u K=%u：偏移表应为 %u 行 %u 字节（实为 %u）",
                  c.rows, c.esz, kTK, rowsWant, want, L1.offTblBytes);
            CHECK(L1.fits && L1.totalBytes <= L1.limitBytes,
                  "rows=%u esz=%u 核数=%u K=%u：缩表下账不成立（%s，%llu > %llu）",
                  c.rows, c.esz, cores, kTK, s9::WhyText(L1.why),
                  (unsigned long long)L1.totalBytes, (unsigned long long)L1.limitBytes);
            CHECK(L1.tileG >= L.tileG,
                  "rows=%u esz=%u 核数=%u K=%u：缩表之后 tileG 反而变小（%u → %u）",
                  c.rows, c.esz, cores, kTK, L.tileG, L1.tileG);
            }
            // 上限没卡住时，tileG 必须是"再大一行就放不下"的那个值
            const bool capFree = (L.tileG < 4095) && (myRows == 0 || L.tileG < myRows);
            if (capFree) {
                s9::ConcatStageLayout Lp = s9::LayoutConcatStage(
                    bandCols, c.esz, t.numIn, t.maxIn, t.slackIn, t.gBuf, t.aBuf, L.tileG + 1, t.tblK, /*dense=*/false,
                    t.bands == 1 ? (uint64_t)bandCols * c.esz : 0);
                CHECK(!Lp.fits, "rows=%u 核数=%u：tileG=%u 明明还能再装一行却没装",
                      c.rows, cores, L.tileG);
            }
        }
    }
    CHECK(gathers >= 5, "只有 %u 个组合走到向量拼装路 —— 这份账没被测到", gathers);
    printf("  真实 tiling：%zu 个形状 × 2 种核数，其中 %u 个走拼装路，全部放得下且 tileG 取到极值\n",
           cases.size(), gathers);
}

/* ═══ ③b 新账比旧口径严多少 ═══
 *
 * 旧口径（kernel 里那句 `UB_TOTAL`）只扣了序幕的 1152 B，指针暂存那最多 2080 B
 * 是藏在 8 KB 余量里的，也没管逐块对齐；新账把这三项都算上。代价是 tileG
 * 偶尔要回缩一行 —— 这是**实打实的性能代价**，所以要把它量出来、写下来，
 * 而不是"反正放得下"。 */
static void CheckTileGRepair()
{
    struct Case { uint32_t rows; std::vector<uint32_t> lens; uint32_t esz; };
    std::vector<Case> cases = {
        {128,  {32, 32, 32, 32, 32, 32, 32, 32, 32}, 2},
        {1024, {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, 4},
        {512,  {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}, 4},
        {2048, {83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83}, 1},
        {4096, {63, 63, 63, 63, 4}, 2},
        {4096, {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33}, 2},
        {4096, {66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66}, 2},
    };
    {
        std::vector<uint32_t> p4(123, 33);
        cases.push_back({4096, p4, 2});
        std::vector<uint32_t> g16(246, 66);
        cases.push_back({4096, g16, 2});
        std::vector<uint32_t> i2(248, 8);
        cases.push_back({2048, i2, 1});
    }
    uint32_t n = 0, shrunk = 0, maxShrink = 0;
    for (const Case &c : cases) {
        for (uint32_t cores : {s9::AIV_MAX, s9::CONCAT_MAX_CORES}) {
            s9::ConcatTiling t{};
            s9::ConcatKnobs kn;
            kn.maxCores = cores;
            if (!s9::CalcConcatLens(c.rows, c.lens.data(), (uint32_t)c.lens.size(),
                                    c.esz, t, kn)) { continue; }
            if (t.strategy != 3) { continue; }
            const uint32_t bandCols = s9::BandColsMax(t.nOut, t.bands, t.bandGrain, t.bandBal);
            const uint32_t grp = t.blockDim / (t.bands ? t.bands : 1);
            const uint32_t myRows = grp ? CeilDiv(c.rows, grp) : c.rows;
            const s9::ConcatStageLayout L = s9::PlanConcatStage(
                bandCols, c.esz, t.numIn, t.maxIn, t.slackIn, t.gBuf, t.aBuf, 0, myRows, t.tblK, /*dense=*/false, 0, t.bands == 1);
            ++n;
            if (L.tileGBudget > L.tileG) {
                ++shrunk;
                if (L.tileGBudget - L.tileG > maxShrink) { maxShrink = L.tileGBudget - L.tileG; }
            }
        }
    }
    printf("  回缩统计：%u 个发布配置里 %u 个因新账回缩（最多 %u 行）；"
           "回缩的代价＝每核多跑一轮流水\n", n, shrunk, maxShrink);
    CHECK(n >= 10, "回缩统计的样本太少（%u）", n);
}

/* ═══ ③c 发布前的最终配置校验 ═══
 *
 * `TilingFunc` 里那串 S9_* 覆盖各改各的 —— 核数、带数、buffer 数、UB 预算、
 * 余量路数。这里逐条把"坏事"喂进去，要求**要么明确失败、要么安全降级**，
 * 不许"两个数各自为政"（那正是 vector core exception 507035 的来源）。 */
static void CheckFinalConfigValidation()
{
    std::vector<uint32_t> lens{63, 63, 63, 63, 4};
    s9::ConcatTiling t{};
    s9::ConcatKnobs kn;
    kn.maxCores = 40;
    const bool ok = s9::CalcConcatLens(4096, lens.data(), (uint32_t)lens.size(), 2, t, kn);
    CHECK(ok && t.strategy == 3, "校验用例的基座应当是 fp16 拼装路（实为 strategy=%u）",
          t.strategy);
    if (!ok || t.strategy != 3) { return; }

    // ── 发布默认配置：直接过 ──
    {
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, t.blockDim, t.bands, t.gBuf,
                                                      t.aBuf, t.slackIn, t.ubBudget, 0, 40);
        CHECK(f.ok && !f.changed, "发布默认配置应当直接通过（why=%s）", f.why);
        CHECK(f.strategy == 3 && f.gBuf == 2, "发布默认配置不该被改动");
    }
    // ① 核数超过平台查询值（S9_BLK=48 而平台只有 40）—— 对 SyncAll 是死锁，必须拒
    {
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, 48, t.bands, t.gBuf, t.aBuf,
                                                      t.slackIn, t.ubBudget, 0, 40);
        CHECK(!f.ok, "核数 48 > 平台 40 应当被拒");
        CHECK(f.why != nullptr && std::string_view(f.why).find("平台") != std::string_view::npos,
              "拒绝理由应当点明平台核数（实为 %s）", f.why ? f.why : "(无)");
    }
    // ② S9_GBUF=4 —— kernel 的事件号只到 3，必须收敛并留痕
    {
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, t.blockDim, t.bands, 4, t.aBuf,
                                                      t.slackIn, t.ubBudget, 0, 40);
        CHECK(f.ok && f.changed && f.gBuf == 3, "gBuf=4 应当收敛到 3（实为 %u）", f.gBuf);
        CHECK(std::string_view(f.why).find("事件") != std::string_view::npos,
              "收敛理由应当点明事件号上限（实为 %s）", f.why);
    }
    // ③ S9_MAXINB 低估 —— 一条带实际挤进的路数比余量还多，必须拒
    {
        const uint32_t need = s9::MaxInputsPerBand(t.nOut, t.nIn, t.numIn, t.bands,
                                                   t.bandGrain, 32 / 2, t.bandBal);
        CHECK(need >= 1, "本用例应当至少有一路在带里");
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, t.blockDim, t.bands, t.gBuf,
                                                      t.aBuf, need - 1, t.ubBudget, 0, 40);
        CHECK(!f.ok, "余量路数 %u 低于实际需要的 %u，应当被拒", need - 1, need);
        CHECK(std::string_view(f.why).find("余量") != std::string_view::npos,
              "拒绝理由应当点明余量路数（实为 %s）", f.why);
    }
    // ④ UB 预算覆盖到超过容量（S9_UBX 给太大）—— BUF_N 块 × ub 会踩别人
    {
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, t.blockDim, t.bands, t.gBuf,
                                                      t.aBuf, t.slackIn, 64 * 1024, 0, 40);
        CHECK(!f.ok, "UB 预算 64 KB × 3 块超过单核 UB，应当被拒");
        CHECK(std::string_view(f.why).find("UB") != std::string_view::npos,
              "拒绝理由应当点明 UB 预算（实为 %s）", f.why);
    }
    // ⑤ 带数不整除核数（S9_BANDS 或 S9_BLK 之后）—— kernel 会退回 1，必须收敛成因数
    {
        const uint32_t bad = t.blockDim - 1;          // 40 → 39，不整除
        const s9::ConcatFinal f = s9::CheckConcatFinal(t, 2, t.blockDim, bad, t.gBuf, t.aBuf,
                                                      t.slackIn, t.ubBudget, 0, 40);
        CHECK(f.ok && f.changed && t.blockDim % f.bands == 0,
              "不整除核数的带数应当收敛成因数（bands %u→%u）", bad, f.bands);
        CHECK(f.strategy == 3, "收敛带数不该把策略也改掉");
    }
    // ⑥ 带粒度不是 32B 的整数倍（S9_BANDGRAIN=10 配 fp16 ＝ 20 字节）—— 带首不再对齐
    {
        s9::ConcatTiling t2 = t;
        t2.bandGrain = 5;                             // 5 元素 × 2 字节 ＝ 10 字节
        const s9::ConcatFinal f = s9::CheckConcatFinal(t2, 2, t2.blockDim, t2.bands, t2.gBuf,
                                                      t2.aBuf, t2.slackIn, t2.ubBudget, 0, 40);
        CHECK(!f.ok, "带粒度 10 字节（非 32B 整数倍）应当被拒");
    }
    // ⑦ 拼装路的 UB 实账放不下 —— 安全降级到按行切，而不是硬闯
    {
        s9::ConcatTiling big{};
        big.strategy = 3; big.rows = 1; big.numIn = 2; big.nOut = 200000; big.maxIn = 100000;
        big.nIn[0] = 100000; big.nIn[1] = 100000;
        big.offOut[0] = 0; big.offOut[1] = 100000;
        big.bands = 1; big.aBuf = 1; big.gBuf = 2; big.bandGrain = 32; big.bandBal = 0;
        big.blockDim = 40; big.ubBudget = 16384; big.maxInBand = 2; big.slackIn = 2;
        const s9::ConcatFinal f = s9::CheckConcatFinal(big, 1, 40, 1, 2, 1, 2, 16384, 0, 40);
        CHECK(f.ok && f.changed && f.strategy == 0,
              "放不下的拼装路应当降级成按行切（实为 strategy=%u, ok=%d）", f.strategy,
              (int)f.ok);
        CHECK(std::string_view(f.why).find("降级") != std::string_view::npos,
              "降级理由应当写明（实为 %s）", f.why);
    }
    // ⑧ 路数越界 / nIn 与 nOut 对不上 —— 结构性错误，明确失败
    {
        s9::ConcatTiling t3 = t;
        t3.nOut += 1;                        // 与 nIn[] 之和对不上
        const s9::ConcatFinal f = s9::CheckConcatFinal(t3, 2, t3.blockDim, t3.bands, t3.gBuf,
                                                      t3.aBuf, t3.slackIn, t3.ubBudget, 0, 40);
        CHECK(!f.ok, "nIn[] 之和与 nOut 不符应当被拒");
    }
    // ── 48 核（评测机）那一半也要覆盖：核数一变，带数与各条判据会同时挪动 ──
    {
        s9::ConcatTiling t48{};
        s9::ConcatKnobs kn48;
        kn48.maxCores = s9::CONCAT_MAX_CORES;
        const bool ok48 = s9::CalcConcatLens(4096, lens.data(), (uint32_t)lens.size(),
                                             2, t48, kn48);
        CHECK(ok48, "48 核档应当算得出配置");
        if (ok48) {
            CHECK(t48.blockDim <= s9::CONCAT_MAX_CORES, "48 核档的核数不该超过上限");
            const s9::ConcatFinal f = s9::CheckConcatFinal(
                t48, 2, t48.blockDim, t48.bands, t48.gBuf, t48.aBuf, t48.slackIn,
                t48.ubBudget, 0, s9::CONCAT_MAX_CORES);
            CHECK(f.ok && !f.changed, "48 核的发布配置应当直接通过（why=%s）", f.why);
        }
    }
    printf("  最终配置校验：40/48 核的默认配置都通过；核数越界/余量低估/UB 超容量/"
           "带粒度未对齐/nIn 不符 五类负例被拒；gBuf 越界与带数不整除核数 两类被收敛并留痕；"
           "放不下的拼装路降级按行切\n");
}

/* ═══ ④ 溢出与不可行：不能悄悄截断 ═══ */
/* ═══ ③e 每带执行计划（T5 的契约）═══
 *
 * 计划让每个核**跳过"扫一遍全部路"**，所以它必须与"真扫一遍"的结果逐位相同。
 * 这里就按定义再扫一遍当靶子，并且专门覆盖：空片、极端不均匀、尾带、
 * 以及 40/48 核两种核数下的分带。
 */
static void CheckBandPlan()
{
    struct Case { uint32_t rows; std::vector<uint32_t> lens; uint32_t esz; };
    std::vector<Case> cases = {
        {4096, {63, 63, 63, 63, 4}, 2},
        {512,  {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33}, 2},
        {256,  {0, 128, 0, 256}, 2},
        {2048, {20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 1},
        {1,    {4096, 4000, 3200, 800}, 2},
        {1024, {1000, 2000, 3000}, 4},
        {128,  {0, 0, 0, 1, 0, 0, 4096, 0, 0}, 2},      // 空片夹着一条长的
    };
    {
        std::vector<uint32_t> many(123, 33);
        cases.push_back({4096, many, 2});
        std::vector<uint32_t> g16(246, 66);
        cases.push_back({4096, g16, 2});
        std::vector<uint32_t> ragged;
        for (uint32_t i = 0; i < 97; ++i) { ragged.push_back((i * 37) % 64); }
        cases.push_back({256, ragged, 2});
        std::vector<uint32_t> clust(200, 1);
        clust.push_back(456);
        cases.push_back({256, clust, 2});
    }

    uint32_t plans = 0, bandsChecked = 0, emptyBands = 0;
    for (const Case &c : cases) {
        for (uint32_t cores : {s9::AIV_MAX, s9::CONCAT_MAX_CORES}) {
            s9::ConcatTiling t{};
            s9::ConcatKnobs kn;
            kn.maxCores = cores;
            if (!s9::CalcConcatLens(c.rows, c.lens.data(), (uint32_t)c.lens.size(),
                                    c.esz, t, kn)) { continue; }
            if (t.strategy != 3) { continue; }
            s9::ConcatBandPlan p{};
            CHECK(s9::CalcConcatBandPlan(t, p), "rows=%u 核数=%u：计划应当算得出", c.rows, cores);
            CHECK(p.bands == t.bands, "计划带数 %u 与 tiling 的 %u 不一致", p.bands, t.bands);
            ++plans;
            uint32_t prevLo = 0, prevHi = 0;
            for (uint32_t b = 0; b < p.bands; ++b) {
                ++bandsChecked;
                // ① 区间合法
                CHECK(p.lo[b] <= p.hi[b] && p.hi[b] <= t.numIn,
                      "带 %u 的区间 [%u,%u) 越界（numIn=%u）", b, p.lo[b], p.hi[b], t.numIn);
                if (p.lo[b] == p.hi[b]) { ++emptyBands; }
                // ② 与"真扫一遍"逐位相同
                uint32_t c0 = 0, c1 = 0;
                s9::BandRange(t.nOut, t.bands, t.bandGrain, b, c0, c1, t.bandBal);
                uint32_t lo = t.numIn, hi = 0;
                if (c0 < c1) {           // 零宽带谁都不相交（与计划、kernel 同一条规则）
                    for (uint32_t i = 0; i < t.numIn; ++i) {
                        if (t.nIn[i] == 0) { continue; }
                        if (t.offOut[i] < c1 && t.offOut[i] + t.nIn[i] > c0) {
                            if (i < lo) { lo = i; }
                            hi = i + 1;
                        }
                    }
                }
                if (lo >= hi) { lo = 0; hi = 0; }
                CHECK(p.lo[b] == lo && p.hi[b] == hi,
                      "带 %u：计划 [%u,%u) 与真扫 [%u,%u) 不一致", b, p.lo[b], p.hi[b], lo, hi);
                // ③ 计划里的区间必须真的覆盖"带内的每一段"
                if (hi > 0) {
                    for (uint32_t i = lo; i < hi; ++i) {
                        if (t.nIn[i] == 0) { continue; }
                        CHECK(t.offOut[i] < c1 && t.offOut[i] + t.nIn[i] > c0,
                              "带 %u：区间里的第 %u 路其实不相交", b, i);
                    }
                    CHECK(p.off[b] == t.offOut[lo], "带 %u 的输出前缀 %u ≠ offOut[%u]=%u",
                          b, p.off[b], lo, t.offOut[lo]);
                }
                // ④ 单调：带是有序切的，区间不该来回跳
                CHECK(p.lo[b] >= prevLo && p.hi[b] >= prevHi,
                      "带 %u 的区间相对上一带回退（[%u,%u) 之后是 [%u,%u)）",
                      b, prevLo, prevHi, p.lo[b], p.hi[b]);
                prevLo = p.lo[b];
                prevHi = p.hi[b];
            }
            // ⑤ 覆盖面：所有非空输入至少被一条带的区间圈住
            std::vector<uint8_t> covered(t.numIn, 0);
            for (uint32_t b = 0; b < p.bands; ++b) {
                for (uint32_t i = p.lo[b]; i < p.hi[b]; ++i) { covered[i] = 1; }
            }
            uint32_t missed = 0;
            for (uint32_t i = 0; i < t.numIn; ++i) {
                if (t.nIn[i] != 0 && !covered[i]) { ++missed; }
            }
            CHECK(missed == 0, "有 %u 路非空输入没被任何一条带的区间覆盖", missed);
        }
    }
    CHECK(plans >= 8, "计划用例太少（%u）", plans);
    CHECK(bandsChecked > 40, "检查过的带太少（%u）", bandsChecked);
    /* **空带**：host 选带时要求"粒度数 ≥ 带数"，所以它在线上不该出现 ——
     * 但计划函数必须有明确行为（而不是越界读）。这里手工造一个 G < bands 的配置：
     * nOut=8、grain=4 ⇒ G=2，4 条带里有两条分不到粒度。 */
    {
        s9::ConcatTiling t{};
        t.rows = 1; t.nOut = 8; t.numIn = 1; t.nIn[0] = 8; t.offOut[0] = 0;
        t.bands = 4; t.bandGrain = 4; t.bandBal = 1; t.strategy = 3;
        s9::ConcatBandPlan p{};
        CHECK(s9::CalcConcatBandPlan(t, p), "空带配置也应当算得出计划");
        uint32_t empty = 0;
        for (uint32_t b = 0; b < p.bands; ++b) {
            CHECK(p.lo[b] <= p.hi[b] && p.hi[b] <= t.numIn, "空带 %u 的区间越界", b);
            if (p.lo[b] == p.hi[b]) { ++empty; }
        }
        CHECK(empty >= 2, "G=2 < bands=4 时应当至少有两条空带（实为 %u）", empty);
        ++emptyBands;
        printf("  每带计划（空带）：nOut=8 grain=4 bands=4 ⇒ %u 条空带，区间仍合法\n", empty);
    }
    printf("  每带计划：%u 个配置 / %u 条带，区间与真扫一遍逐位相同、单调、且覆盖全部非空输入"
           "（其中 %u 条空带）\n", plans, bandsChecked, emptyBands);
}

/* ═══ ④ 溢出与不可行：不能悄悄截断 ═══ */
static void CheckOverflowAndInfeasible()
{
    // 一行的字节数顶到 uint32 边界 ⇒ 必须显式判不可行，而不是截断成一个"看起来能装下"的数
    {
        const uint64_t bw = 0x40000000ull;          // ×4 字节 = 2^32
        s9::ConcatStageLayout L = s9::PlanConcatStage((uint32_t)bw, 4, /*numIn=*/4,
                                                     /*maxNIn=*/1000, /*nSlack=*/4,
                                                     2, 1, 0, 0, /*tblK=*/0, /*dense=*/false);
        CHECK(!L.fits, "2^32 字节的一行应当判不可行");
        CHECK(L.why == s9::C9_OVERFLOW,
              "2^32 字节的一行应当以**溢出**为原因（实为 %s）", s9::WhyText(L.why));
        printf("  溢出：bw=2^30 元素 × fp32 ⇒ %s\n", s9::WhyText(L.why));
    }
    // tileG 超过 DataCopyPad 的 blockCount 上限
    {
        s9::ConcatStageLayout L = s9::LayoutConcatStage(64, 2, 4, 64, 4, 2, 1, 5000, /*tblK=*/0, /*dense=*/false);
        CHECK(!L.fits, "tileG=5000 应当被 blockCount 上限拒绝");
        CHECK(L.why == s9::C9_TILEG_TOO_BIG,
              "tileG 越界的原因应当是 blockCount 上限（实为 %s）", s9::WhyText(L.why));
    }
    // 一行本身就装不下 ⇒ 缩到 1 行也不行，必须判不可行
    {
        s9::ConcatStageLayout L = s9::PlanConcatStage(/*bw=*/100000, 1, /*numIn=*/256,
                                                     /*maxNIn=*/100000, /*nSlack=*/256,
                                                     2, 2, 0, 0, /*tblK=*/0, /*dense=*/false);
        CHECK(!L.fits, "一行都装不下时应当判不可行");
        CHECK(L.tileG <= 1, "不可行时不该报出一个大于 1 的 tileG（实为 %u）", L.tileG);
        printf("  不可行：bw=100000 maxNIn=100000 ⇒ %s\n", s9::WhyText(L.why));
    }
    // 空配置
    {
        CHECK(!s9::PlanConcatStage(0, 2, 4, 64, 4, 2, 1, 0, 0, 0, false).fits, "bw=0 应当判不可行");
        CHECK(!s9::PlanConcatStage(64, 0, 4, 64, 4, 2, 1, 0, 0, 0, false).fits, "esz=0 应当判不可行");
        CHECK(!s9::PlanConcatStage(64, 2, 0, 64, 4, 2, 1, 0, 0, 0, false).fits, "numIn=0 应当判不可行");
    }
    // host 的可行性入口要与之一致
    {
        CHECK(!s9::GatherFitsBands(1u << 30, 4, 1000, 4, 1), "GatherFitsBands 应当拒绝溢出配置");
        CHECK(!s9::GatherFitsBands(0, 4, 1000, 4, 1), "nOut=0 应当被拒");
        CHECK(!s9::GatherFitsBands(4096, 4, 1000, 4, 0), "bands=0 应当被拒");
    }
}

/* ═══ 逐核布局表 == kernel 现算的那一份（布局下沉的前提）═══
 *
 * 布局下沉（`S9_LAYOUT=1`）把「每个核自己算布局」换成「host 算好、kernel 读」。
 * 首版要求**逐字段一致** —— 否则就不是"只改计算位置"，而是换了一份配置，
 * 量出来的差分不清是"读表更便宜"还是"布局变了"。
 *
 * 这个检查**把 kernel 侧那段算术原样抄一遍**（带的钳位、带内连续编号、
 * 按 bandBal 的两种列带切法、行按组切 + 余数分给前几组、空带的兜底），
 * 再与 `CalcConcatCoreLayouts` 逐核逐字段比。抄一遍是有意的：
 * 比的是"两份独立实现算得一样"，不是"自己和自己比"。 */
namespace {

struct KernelPathRow {
    uint32_t rowBeg, rowEnd, bw, cb0, cb1, outPitch, tblRow, tileG, stageSpan, asmSpan,
             offTblBytes, dltBytes, castSpan, asmGBytes;
};

/* `op_kernel/concat.cpp` 的 Init() 里 strategy 3 那一段（去掉 InitBuffer），
 * 逐行照抄 —— 包括它自己的 S9Min/S9AlignUp/S9CeilDiv。 */
KernelPathRow KernelPath(uint32_t rows, uint32_t nOut, uint32_t bn, uint32_t bandsIn,
                         uint32_t grainElIn, uint32_t bandBal, uint32_t bi,
                         uint32_t alignEl, uint32_t esz, uint32_t numIn, uint32_t maxNIn,
                         uint32_t nSlack, uint32_t gBuf, uint32_t aBuf, uint32_t tileGMax)
{
    const uint32_t bands = (bandsIn >= 1 && bandsIn <= bn && bn % bandsIn == 0) ? bandsIn : 1;
    const uint32_t gBase = bn / bands, gExtra = bn % bands;
    uint32_t bandIdx = 0, grpIdx = 0, grp = gBase, start = 0;
    for (uint32_t b = 0; b < bands; ++b) {
        const uint32_t gcnt = gBase + (b < gExtra ? 1u : 0u);
        if (bi < start + gcnt) { bandIdx = b; grpIdx = bi - start; grp = gcnt; break; }
        start += gcnt;
    }
    const uint32_t grainEl = grainElIn ? grainElIn : alignEl;
    uint32_t bandCols = 0, cb0 = 0, cb1 = 0;
    if (bandBal == 0) {
        bandCols = (bands == 1) ? nOut : AlignUp(CeilDiv(nOut, bands), grainEl);
        cb0 = bandIdx * bandCols;
        cb1 = (cb0 + bandCols < nOut) ? (cb0 + bandCols) : nOut;
        if (cb0 > nOut) { cb0 = nOut; }
    } else if (bands == 1) {
        bandCols = nOut; cb0 = 0; cb1 = nOut;
    } else {
        const uint32_t G = CeilDiv(nOut, grainEl);
        bandCols = CeilDiv(G, bands) * grainEl;
        const uint32_t g0 = (uint32_t)((uint64_t)bandIdx * G / bands);
        const uint32_t g1 = (uint32_t)((uint64_t)(bandIdx + 1) * G / bands);
        cb0 = g0 * grainEl; cb1 = g1 * grainEl;
        if (cb0 > nOut) { cb0 = nOut; }
        if (cb1 > nOut) { cb1 = nOut; }
    }
    if (cb0 >= cb1) { cb0 = 0; cb1 = 0; }
    uint32_t bw = cb1 - cb0;
    uint32_t rowBeg = 0, rowEnd = 0;
    {
        const uint32_t q = rows / grp, rr = rows % grp;
        rowBeg = q * grpIdx + (grpIdx < rr ? grpIdx : rr);
        rowEnd = rowBeg + q + (grpIdx < rr ? 1u : 0u);
    }
    if (bw == 0) { rowBeg = 0; rowEnd = 0; bw = alignEl; }
    const uint32_t myRows = (rowEnd > rowBeg) ? (rowEnd - rowBeg) : 1;
    const s9::ConcatStageLayout lay = s9::PlanConcatStage(bw, esz, numIn, maxNIn, nSlack,
                                                          gBuf, aBuf, tileGMax, myRows,
                                                          /*tblK=*/0, /*dense=*/false, 0, bands == 1);
    return KernelPathRow{rowBeg, rowEnd, bw, cb0, cb1, lay.outPitch, lay.tblRow, lay.tileG,
                         lay.stageSpan, lay.asmSpan, lay.offTblBytes, lay.dltBytes,
                         lay.castSpan, lay.asmGBytes};
}

bool SameRow(const KernelPathRow &k, const s9::ConcatCoreLayoutRow &h)
{
    return k.rowBeg == h.rowBeg && k.rowEnd == h.rowEnd && k.bw == h.bw && k.cb0 == h.cb0 &&
           k.cb1 == h.cb1 && k.outPitch == h.outPitch &&
           k.tblRow == h.tblRow && k.tileG == h.tileG && k.stageSpan == h.stageSpan &&
           k.asmSpan == h.asmSpan && k.offTblBytes == h.offTblBytes &&
           k.dltBytes == h.dltBytes && k.castSpan == h.castSpan && k.asmGBytes == h.asmGBytes;
}

}  // namespace

static void CheckCoreLayoutTable()
{
    struct Case { uint32_t rows, esz; std::vector<uint32_t> lens; };
    std::vector<Case> cases = {
        {128,  2, {32, 32, 32, 32, 32, 32, 32, 32, 32}},          // 样例 case1 形状
        {512,  4, {8, 8, 8, 8, 8, 8, 8, 8}},                      // 2048 列、8 路
        {512,  2, {0, 128, 0, 64, 64}},
        {4096, 2, {63, 63, 63, 63, 4}},                           // 文档里的反例
        {2048, 1, {16, 16, 16, 16, 16, 16, 16, 16}},             // int8 借道 half
        {1,    2, {3000, 3000, 3000, 3000, 3000, 3000, 3000}},   // dim=0 那族
        {8,    2, {4096, 4096, 4096, 4096}},                      // nOut 大、行少
    };
    {   // Case4/Case5 那一族：路数多、宽度随机（这里用确定性的参差宽度）
        std::vector<uint32_t> many, g16;
        for (uint32_t i = 0; i < 123; ++i) { many.push_back(31 + (i * 7) % 64); }
        for (uint32_t i = 0; i < 246; ++i) { g16.push_back(61 + (i * 11) % 128); }
        cases.push_back({4096, 2, many});
        cases.push_back({4096, 2, g16});
    }
    uint32_t checked = 0, skipped = 0;
    for (const Case &c : cases) {
        const std::vector<uint32_t> &lens = c.lens;
        for (uint32_t cores : {8u, 16u, 40u, 48u}) {
            s9::ConcatTiling t{};
            s9::ConcatKnobs kn;
            kn.maxCores = cores;
            kn.bandBal  = 0;
            if (!s9::CalcConcatLens(c.rows, lens.data(), (uint32_t)lens.size(),
                                    c.esz, t, kn)) {
                continue;
            }
            if (t.strategy != 3) { ++skipped; continue; }
            const uint32_t bn = t.blockDim;
            const uint32_t alignEl = 32 / c.esz;
            const uint32_t nSlack = t.slackIn ? t.slackIn : t.numIn;
            // 带数轴：不仅 host 选的那个，也扫核数的其它因数（工具里就是这么枚举的）
            for (uint32_t bands = 1; bands <= bn; ++bands) {
                if (bn % bands != 0) { continue; }
                s9::ConcatCoreLayoutRow tab[s9::S9L_CORE_MAX];
                uint32_t wrote = 0;
                const uint32_t bad = s9::CalcConcatCoreLayouts(
                    t, c.esz, bn, t.gBuf, t.aBuf, t.maxIn, nSlack, /*tileGMax=*/0, t.numIn,
                    t.maxIn, bands, t.bandGrain, tab, s9::S9L_CORE_MAX, &wrote);
                CHECK(wrote == bn, "布局表应写满 %u 行（写了 %u）", bn, wrote);
                for (uint32_t k = 0; k < wrote; ++k) {
                    const KernelPathRow kr = KernelPath(
                        c.rows, t.nOut, bn, bands, t.bandGrain, t.bandBal, k, alignEl, c.esz,
                        t.numIn, t.maxIn, nSlack, t.gBuf, t.aBuf, /*tileGMax=*/0);
                    ++checked;
                    CHECK(SameRow(kr, tab[k]),
                          "核 %u（rows=%u nOut=%u esz=%u 核数=%u 带数=%u 放不下核数=%u）："
                          "kernel(bw=%u cb0=%u cb1=%u tileG=%u span=%u asmSpan=%u off=%u "
                          "dlt=%u) ≠ host(bw=%u cb0=%u cb1=%u tileG=%u span=%u asmSpan=%u "
                          "off=%u dlt=%u)",
                          k, c.rows, t.nOut, c.esz, bn, bands, bad,
                          kr.bw, kr.cb0, kr.cb1, kr.tileG, kr.stageSpan, kr.asmSpan,
                          kr.offTblBytes, kr.dltBytes,
                          tab[k].bw, tab[k].cb0, tab[k].cb1, tab[k].tileG, tab[k].stageSpan,
                          tab[k].asmSpan, tab[k].offTblBytes, tab[k].dltBytes);
                }
            }
        }
    }
    printf("  逐核布局表：%u 个核的布局与 kernel 现算的**逐字段一致**"
           "（另有 %u 个形状不走拼装路，跳过）\n", checked, skipped);
}

static void CheckDenseLayout()
{
    CHECK(!s9::S9LUseDense(211, 1, 97, 72, 1, 0), "short multi-input group must keep padded layout");
    CHECK(s9::S9LUseDense(211, 1, 97, 72, 2, 0), "force dense for diagnosis");
    CHECK(s9::S9LUseDense(3, 1, 5, 124957, 1, 0), "large narrow group must use dense");
    CHECK(!s9::S9LUseDense(3, 1, 5, 124957, 1, 4), "TBLK excludes dense");
    const auto wideOld = s9::PlanConcatStage(1554, 2, 5, 666, 5, 2, 2, 0, 81, 0, 0);
    const auto wideAuto = s9::PlanConcatStage(1554, 2, 5, 666, 5, 2, 2, 0, 81, 0, 1);
    const auto wideForce = s9::PlanConcatStage(1554, 2, 5, 666, 5, 2, 2, 0, 81, 0, 2);
    CHECK(wideOld.tileG == 7 && wideForce.tileG == 4 && wideForce.superG == 8,
          "wide-row capacity regression fixture");
    CHECK(wideAuto.tileG == wideOld.tileG && wideAuto.superG == 1,
          "automatic dense must not shrink tileG");
    for (uint32_t esz : {1u, 2u, 4u}) {
        for (uint32_t bw = 1; bw <= 129; ++bw) {
            for (uint32_t rows : {1u, 3u, 31u, 32u, 33u, 4095u}) {
                const auto l = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, rows, 0, 2);
                CHECK(l.fits && l.superG * bw * esz % 32 == 0, "dense super-row alignment");
                CHECK(l.asmSpan >= (uint64_t)l.tileG * bw * esz && l.asmSpan % 32 == 0,
                      "dense assembly bounds");
                CHECK(l.offTblBytes >= (uint64_t)l.tileG * bw * 4 && l.dltBytes % 32 == 0,
                      "dense table bounds");
                CHECK(l.outPitch == bw * esz && l.tblRow == l.superG * bw,
                      "dense pitch/table semantics");
                const auto k = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, rows, 4, 2);
                CHECK(k.superG == 1, "TBLK must disable dense");
            }
            const uint32_t group = s9::S9LDenseG(bw, esz, true, 0);
            const auto below = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, group * 3 - 1, 0, 1);
            const auto above = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, group * 3, 0, 1);
            const auto old = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, group * 3, 0, 0);
            const auto forced = s9::PlanConcatStage(bw, esz, 3, bw, 3, 2, 2, 0, group * 3, 0, 2);
            CHECK(below.superG == 1 && above.superG == (forced.tileG >= old.tileG ? group : 1),
                  "automatic dense threshold and capacity guard");
            s9::ConcatKnobs kn;
            kn.maxCores = 3; kn.denseOn = 2; kn.bandMinBytes = 1u << 20;
            uint32_t lens[] = {0, bw / 2, bw - bw / 2};
            s9::ConcatTiling t{};
            CHECK(s9::CalcConcatLens(103, lens, 3, esz, t, kn), "dense host tiling");
            s9::ConcatCoreLayoutRow out[3];
            uint32_t written = 0;
            CHECK(s9::CalcConcatCoreLayouts(t, esz, 3, t.gBuf, t.aBuf, t.maxIn,
                t.slackIn, 0, t.numIn, t.maxIn, 1, t.bandGrain, out, 3, &written) == 0,
                "dense layout table fits");
            for (uint32_t core = 0; core < written; ++core) {
                const auto m = s9::MapConcatCore(t.rows, t.nOut, 3, 1, t.bandGrain, t.bandBal, core, 32 / esz);
                const auto l = s9::PlanConcatStage(m.bw, esz, t.numIn, t.maxIn, t.slackIn,
                    t.gBuf, t.aBuf, 0, m.myRows, 0, 2, 0, m.bands == 1);
                CHECK(out[core].outPitch == l.outPitch && out[core].tblRow == l.tblRow &&
                      out[core].tileG == l.tileG && out[core].offTblBytes == l.offTblBytes,
                      "host/kernel dense table mismatch");
            }
        }
    }
}

int main()
{
    CheckDenseLayout();
    printf("── Concat 拼装路 UB 布局 ─────────────\n");
    CheckCounterexample();
    CheckAlignmentInvariants();
    CheckRealTilings();
    CheckTileGRepair();
    CheckFinalConfigValidation();
    CheckBandPlan();
    CheckCoreLayoutTable();
    CheckOverflowAndInfeasible();
    printf("\n%d 项通过", g_pass);
    if (g_fail) { printf("，\033[31m%d 项失败\033[0m\n", g_fail); return 1; }
    printf("，全部通过\n");
    return 0;
}
