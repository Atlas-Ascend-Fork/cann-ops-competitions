#include "kernel_operator.h"
// BEGIN V403_SELF_CONTAINED_FP32_IMPLEMENTATION
// v403 UNVALIDATED_EXPERIMENT: distributed cold source-tile construction.
// This experiment has no measured result and is not an accepted baseline.
#ifndef INDEX_ADD_FLOAT_TILED_GATHERB_DRAFT_H
#define INDEX_ADD_FLOAT_TILED_GATHERB_DRAFT_H

namespace V397Draft {
using namespace AscendC;

// BEGIN V423_RESIDENT_PLAN_CLASS
// UNVALIDATED_EXPERIMENT: complete tile plans with two phase-owned arenas.
class FloatResidentPlan {
public:
    __aicore__ inline bool TryProcess(GM_ADDR self, GM_ADDR index, GM_ADDR source,
        GM_ADDR output, GM_ADDR workspace, uint32_t total, uint32_t outer,
        uint32_t dim, uint32_t indices, uint32_t inner)
    {
        // Unsupported metadata leaves the old workspace completely untouched.
        if (inner != 1U || outer < 512U || dim <= 512U || dim >= 1024U ||
            indices < 4096U || indices > 8192U || workspace == nullptr ||
            static_cast<uint64_t>(outer) * dim != total ||
            static_cast<uint64_t>(outer) * indices > 0xFFFFFFFFULL) { return false; }
        O = outer; D = dim; M = indices; P = GetBlockNum();
        if (P == 0U) { return false; }
        DP = Align(D, 16U); AM = Align(M, 8U);
        scratchWords = Align((M + P - 1U) / P, 8U);
        if (scratchWords < P * 8U) { scratchWords = P * 8U; }
        planBase = 32U + AM + P * 8U;
        if (!SelectLayout()) { return false; }
        selfGm.SetGlobalBuffer((__gm__ uint32_t*)self, total);
        indexGm.SetGlobalBuffer((__gm__ int32_t*)index, M);
        sourceGm.SetGlobalBuffer((__gm__ uint32_t*)source, O * M);
        sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetGlobalBuffer((__gm__ uint32_t*)output, total);
        planGm.SetGlobalBuffer((__gm__ int32_t*)GetUserWorkspace(workspace), WorkspaceWords());
        // The cold views are disjoint. After successful validation/publication,
        // their entire lifetime ends and this same allocation holds all plans.
        pipe.InitBuffer(planArena, arenaWords * 4U);
        if (!ValidateCache() && !BuildAndPublish()) {
            InvalidateBeforeFallback();
            return false;
        }
        pipe.InitBuffer(rawBuf, 8U * (K + 8U) * 4U);
        const uint32_t transformColumns = K + 8U > DP ? K + 8U : DP;
        // Transposed source and permuted accumulator never have live readers
        // at the same time. One actual allocation backs both typed views.
        pipe.InitBuffer(transformBuf, 8U * transformColumns * 4U);
        pipe.InitBuffer(matrixBuf, 8U * (K + 8U) * 4U);
        pipe.InitBuffer(accBuf, 8U * (alternateAccumArenas ? transformColumns : DP) * 4U);
        // No header, flag or normalized-index UB view survives this handoff.
        // Both warm validation and cold commit have already joined all peers.
        PipeBarrier<PIPE_ALL>();
        DataCopy(planArena.Get<int32_t>(), planGm[planBase], tileCount * planWords);
        PipeBarrier<PIPE_ALL>();
        selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        Compute();
        selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        return true;
    }

private:
    static constexpr uint32_t MagicA = 0x49333937U;
    static constexpr uint32_t MagicB = 0x54494C45U;
    // Resident tile permutations address the preceding tile's carried order.
    static constexpr uint32_t Version = 3U;
    __aicore__ inline uint32_t Align(uint32_t n, uint32_t a) const { return (n + a - 1U) / a * a; }
    __aicore__ inline int32_t Normalize(int32_t t) const {
        if (t < 0) { t += static_cast<int32_t>(D); }
        return t >= 0 && t < static_cast<int32_t>(D) ? t : -1;
    }
    __aicore__ inline uint32_t WorkspaceWords() const {
        const uint32_t local = (M + P - 1U) / P;
        const uint32_t occupied = local < D ? local : D;
        return 32U + 2U * P * DP + DP + P * Align(occupied * 8U, 16U) + AM + 64U * DP;
    }
    __aicore__ inline bool SelectLayout() {
        for (K = 1024U; K >= 128U; K /= 2U) {
            tileCount = (M + K - 1U) / K;
            sourcePlanBase = 32U;
            permPlanBase = sourcePlanBase + Align(K + 7U, 8U);
            inversePlanBase = permPlanBase + DP;
            prefixPlanBase = inversePlanBase + DP;
            planWords = prefixPlanBase + 256U;
            const uint64_t coldWords = static_cast<uint64_t>(DP) +
                AM + scratchWords + planWords;
            const uint64_t hotWords = static_cast<uint64_t>(tileCount) * planWords;
            const uint64_t candidateArenaWords = coldWords > hotWords ? coldWords : hotWords;
            const uint32_t transformColumns = K + 8U > DP ? K + 8U : DP;
            const uint64_t bytes = 8ULL * (K + 8U) * 4U +
                8ULL * transformColumns * 4U +
                8ULL * (K + 8U) * 4U + 8ULL * DP * 4U +
                candidateArenaWords * 4ULL;
            if (bytes + 8192ULL <= 196608ULL &&
                hotWords * 4ULL <= 196608ULL &&
                static_cast<uint64_t>(planBase) + hotWords <= WorkspaceWords()) {
                coldNextBase = DP;
                coldPlanBase = DP + AM + scratchWords;
                arenaWords = static_cast<uint32_t>(candidateArenaWords);
                // Keep the old K/layout choice and retain the original copy
                // path unless both alternating roles fit with full headroom.
                const uint64_t extraAccumBytes = 8ULL * (transformColumns - DP) * 4U;
                alternateAccumArenas = bytes + extraAccumBytes + 8192ULL <= 196608ULL;
                return true;
            }
        }
        // Metadata/resource rejection occurs before binding or touching GM.
        return false;
    }
    __aicore__ inline void ReadHeader() {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        DataCopy(planArena.Get<int32_t>(), planGm, 32U);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool HeaderMatches() {
        LocalTensor<int32_t> h = planArena.Get<int32_t>();
        return static_cast<uint32_t>(h.GetValue(0)) == MagicA &&
            static_cast<uint32_t>(h.GetValue(1)) == MagicB &&
            static_cast<uint32_t>(h.GetValue(2)) == Version &&
            static_cast<uint32_t>(h.GetValue(3)) == P &&
            static_cast<uint32_t>(h.GetValue(4)) == O &&
            static_cast<uint32_t>(h.GetValue(5)) == D &&
            static_cast<uint32_t>(h.GetValue(6)) == M &&
            static_cast<uint32_t>(h.GetValue(7)) == K &&
            static_cast<uint32_t>(h.GetValue(8)) == tileCount &&
            static_cast<uint32_t>(h.GetValue(9)) == planWords;
    }
    __aicore__ inline bool ValidateCache() {
        ReadHeader();
        const bool headerValid = HeaderMatches();
        const uint32_t core = GetBlockIdx(), n = M / P, extra = M % P;
        const uint32_t start = core * n + (core < extra ? core : extra);
        const uint32_t count = n + (core < extra ? 1U : 0U);
        LocalTensor<int32_t> current = planArena.Get<int32_t>()[coldNextBase];
        LocalTensor<int32_t> cached = current[AM];
        if (headerValid && count != 0U) {
            DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(current, indexGm[start], copy, pad);
            DataCopyPad(cached, planGm[32U + start], copy, pad);
        }
        PipeBarrier<PIPE_ALL>();
        uint32_t mismatch = headerValid ? 0U : 1U;
        // BEGIN V470_RESIDENT_VECTOR_VALIDATION
        // The published cache contains only -1 or normalized targets [0,D).
        // Raw equality alone is unsafe: cached -1 can equal a new raw -1
        // whose normalization is D-1. Require equality AND no cached -1.
        // Anything else uses the original Normalize comparison, not a forced miss.
        uint32_t scalarBegin = 0U;
        if (headerValid && count != 0U) {
            uint32_t vectorElements = (count / 64U) * 64U;
            const uint32_t paddedElements = Align(count, 64U);
            if (paddedElements <= AM && paddedElements <= scratchWords) {
                for (uint32_t i = count; i < paddedElements; ++i) {
                    current.SetValue(i, 0);
                    cached.SetValue(i, 0);
                }
                if (paddedElements != count) {
                    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
                }
                vectorElements = paddedElements;
            }
            // Both int32 Compare APIs execute only complete 64-element repeats.
            // The cold q view is dead until BuildTile; no allocation is added.
            const uint32_t bitmapBytes = Align(vectorElements / 8U, 32U);
            if (vectorElements != 0U &&
                2ULL * bitmapBytes <= static_cast<uint64_t>(planWords) * 4ULL) {
                LocalTensor<uint8_t> equalBits =
                    planArena.Get<int32_t>()[coldPlanBase].ReinterpretCast<uint8_t>();
                LocalTensor<uint8_t> sentinelBits = equalBits[bitmapBytes];
                Compare(equalBits, current, cached, CMPMODE::EQ, vectorElements);
                CompareScalar(sentinelBits, cached, static_cast<int32_t>(-1),
                    CMPMODE::EQ, vectorElements);
                SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
                LocalTensor<uint32_t> equalWords = equalBits.ReinterpretCast<uint32_t>();
                LocalTensor<uint32_t> sentinelWords = sentinelBits.ReinterpretCast<uint32_t>();
                bool vectorHit = true;
                for (uint32_t word = 0U; word < vectorElements / 32U; ++word) {
                    if (equalWords.GetValue(word) != 0xFFFFFFFFU ||
                        sentinelWords.GetValue(word) != 0U) {
                        vectorHit = false;
                        break;
                    }
                }
                if (vectorHit) {
                    scalarBegin = vectorElements < count ? vectorElements : count;
                }
            }
        }
        // END V470_RESIDENT_VECTOR_VALIDATION
        for (uint32_t i = scalarBegin; headerValid && i < count; ++i) {
            if (Normalize(current.GetValue(i)) != cached.GetValue(i)) { mismatch = 1U; break; }
        }
        Duplicate(cached, static_cast<int32_t>(0), 8U);
        PipeBarrier<PIPE_ALL>();
        cached.SetValue(0U, static_cast<int32_t>(mismatch));
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[32U + AM + core * 8U], cached, 8U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        // All header reads/validation finish before any cold writer can proceed.
        SyncAll<true>();
        DataCopy(cached, planGm[32U + AM], P * 8U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < P; ++c) { if (cached.GetValue(c * 8U) != 0) { return false; } }
        return true;
    }
    __aicore__ inline void ClearHeader() {
        LocalTensor<int32_t> h = planArena.Get<int32_t>();
        Duplicate(h, static_cast<int32_t>(0), 32U);
        PipeBarrier<PIPE_ALL>();
        DataCopy(planGm, h, 32U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
    }
    __aicore__ inline void InvalidateBeforeFallback() {
        // Equivalent cohort protection to the repaired v392/v394 helper.
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        SyncAll<true>();
        if (GetBlockIdx() == 0U) { ClearHeader(); }
        SyncAll<true>();
    }
    __aicore__ inline bool BuildTile(uint32_t tile) {
        LocalTensor<int32_t> counts = planArena.Get<int32_t>();
        LocalTensor<int32_t> current = planArena.Get<int32_t>()[coldNextBase];
        LocalTensor<int32_t> q = planArena.Get<int32_t>()[coldPlanBase];
        const uint32_t begin = tile * K, count = M - begin < K ? M - begin : K;
        Duplicate(counts, static_cast<int32_t>(0), DP);
        Duplicate(q, static_cast<int32_t>(0), planWords);
        PipeBarrier<PIPE_ALL>();
        uint32_t maximum = 0U, valid = 0U;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t target = current.GetValue(begin + i);
            if (target < 0) { continue; }
            const uint32_t next = static_cast<uint32_t>(counts.GetValue(target)) + 1U;
            if (next > 255U) { return false; } // No output has been written.
            counts.SetValue(target, static_cast<int32_t>(next));
            if (next > maximum) { maximum = next; }
            ++valid;
        }
        // Counting sort, descending count, stable target-number tiebreak.
        for (uint32_t t = 0; t < D; ++t) {
            const uint32_t c = static_cast<uint32_t>(counts.GetValue(t));
            q.SetValue(prefixPlanBase + c, q.GetValue(prefixPlanBase + c) + 1);
        }
        uint32_t cursor = 0U;
        for (int32_t c = 255; c >= 0; --c) {
            const uint32_t frequency = static_cast<uint32_t>(q.GetValue(prefixPlanBase + c));
            q.SetValue(prefixPlanBase + c, static_cast<int32_t>(cursor)); cursor += frequency;
        }
        for (uint32_t t = 0; t < D; ++t) {
            const uint32_t c = static_cast<uint32_t>(counts.GetValue(t));
            const uint32_t column = static_cast<uint32_t>(q.GetValue(prefixPlanBase + c));
            q.SetValue(prefixPlanBase + c, static_cast<int32_t>(column + 1U));
            q.SetValue(permPlanBase + column, static_cast<int32_t>(t));
            q.SetValue(inversePlanBase + t, static_cast<int32_t>(column));
        }
        uint32_t active = D, matrixCursor = 0U;
        q.SetValue(prefixPlanBase, 0);
        for (uint32_t rank = 0; rank < maximum; ++rank) {
            while (active > 0U && static_cast<uint32_t>(counts.GetValue(
                q.GetValue(permPlanBase + active - 1U))) <= rank) { --active; }
            matrixCursor += active;
            q.SetValue(prefixPlanBase + rank + 1U, static_cast<int32_t>(matrixCursor));
        }
        if (matrixCursor != valid || valid > count) { return false; }
        // Padding points at the permanent zero column after K transposed columns.
        for (uint32_t j = 0; j < Align(K + 7U, 8U); ++j) { q.SetValue(sourcePlanBase + j, static_cast<int32_t>(K * 32U)); }
        for (uint32_t t = 0; t < D; ++t) { counts.SetValue(t, 0); }
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t t = current.GetValue(begin + i);
            if (t < 0) { continue; }
            const uint32_t rank = static_cast<uint32_t>(counts.GetValue(t));
            counts.SetValue(t, static_cast<int32_t>(rank + 1U));
            const uint32_t column = static_cast<uint32_t>(q.GetValue(inversePlanBase + t));
            const uint32_t offset = static_cast<uint32_t>(q.GetValue(prefixPlanBase + rank)) + column;
            if (offset >= valid) { return false; }
            q.SetValue(sourcePlanBase + offset, static_cast<int32_t>(i * 32U));
        }
        for (uint32_t t = 0; t < D; ++t) {
            q.SetValue(permPlanBase + t, q.GetValue(permPlanBase + t) * 32);
            q.SetValue(inversePlanBase + t, q.GetValue(inversePlanBase + t) * 32);
        }
        q.SetValue(0U, static_cast<int32_t>(begin)); q.SetValue(1U, static_cast<int32_t>(count));
        q.SetValue(2U, static_cast<int32_t>(maximum)); q.SetValue(3U, static_cast<int32_t>(valid));
        q.SetValue(4U, static_cast<int32_t>(DP)); q.SetValue(5U, static_cast<int32_t>(K));
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[planBase + tile * planWords], q, planWords);
        // The next BuildTile clears q with V Duplicate. Drain the MTE3
        // payload read before allowing any cross-pipe overwrite of planBuf.
        PipeBarrier<PIPE_ALL>();
        return true;
    }
    __aicore__ inline void ComposeCarryPermutations() {
        LocalTensor<int32_t> previousInverse = planArena.Get<int32_t>();
        LocalTensor<int32_t> q = planArena.Get<int32_t>()[coldPlanBase];
        for (uint32_t tile = GetBlockIdx(); tile < tileCount; tile += P) {
            if (tile == 0U) { continue; }
            DataCopy(previousInverse,
                planGm[planBase + (tile - 1U) * planWords + inversePlanBase], DP);
            DataCopy(q[permPlanBase],
                planGm[planBase + tile * planWords + permPlanBase], DP);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<uint32_t> gatherOffsets = q[permPlanBase].ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> composed = q[inversePlanBase];
            // Original permutation entries are target*32. Gather consumes byte
            // offsets into the int32 inverse array, hence logical >>3.
            ShiftRight(gatherOffsets, gatherOffsets, static_cast<uint32_t>(3U), D);
            Duplicate(composed, static_cast<int32_t>(0), DP);
            PipeBarrier<PIPE_V>();
            Gather(composed.ReinterpretCast<uint32_t>(),
                previousInverse.ReinterpretCast<uint32_t>(), gatherOffsets, 0, D);
            // Zero padding remains defined; GM inverse plans are not modified.
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(planGm[planBase + tile * planWords + permPlanBase],
                composed, DP);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline bool BuildAndPublish() {
        const uint32_t core = GetBlockIdx();
        LocalTensor<int32_t> current = planArena.Get<int32_t>()[coldNextBase];
        LocalTensor<int32_t> flags = current[AM];
        if (core == 0U) { ClearHeader(); }
        // No core builds or publishes payload before the invalid header is visible.
        SyncAll<true>();
        bool localOk = true;
        for (uint32_t tile = core; tile < tileCount && localOk; tile += P) {
            const uint32_t begin = tile * K;
            const uint32_t count = M - begin < K ? M - begin : K;
            const uint32_t paddedCount = Align(count, 8U);
            // K and begin are 32-byte aligned. Only the final tile has padding.
            Duplicate(current[begin], static_cast<int32_t>(0), paddedCount);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(current[begin], indexGm[begin], copy, pad);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < count; ++i) {
                current.SetValue(begin + i, Normalize(current.GetValue(begin + i)));
            }
            // Define cache padding explicitly rather than relying on DMA pad contents.
            for (uint32_t i = count; i < paddedCount; ++i) { current.SetValue(begin + i, 0); }
            SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(planGm[32U + begin], current[begin], paddedCount);
            PipeBarrier<PIPE_ALL>();
            localOk = BuildTile(tile);
        }
        // All blocks, including those owning no tiles or failing a tile, publish
        // one complete status slot. Drain payload/UB readers before slot reuse.
        PipeBarrier<PIPE_ALL>();
        Duplicate(flags, static_cast<int32_t>(0), 8U);
        PipeBarrier<PIPE_ALL>();
        flags.SetValue(0U, localOk ? 0 : 1);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[32U + AM + core * 8U], flags, 8U);
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        // Every normalized slice, tile payload and status is complete before commit.
        SyncAll<true>();
        DataCopy(flags, planGm[32U + AM], P * 8U);
        PipeBarrier<PIPE_ALL>();
        bool allReady = true;
        for (uint32_t c = 0U; c < P; ++c) {
            if (flags.GetValue(c * 8U) != 0) { allReady = false; }
        }
        if (allReady) { ComposeCarryPermutations(); }
        // Version 3 is never published until all transition maps are complete.
        SyncAll<true>();
        if (core == 0U) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
            DataCopy(flags, planGm[32U + AM], P * 8U);
            PipeBarrier<PIPE_ALL>();
            bool ok = true;
            for (uint32_t c = 0U; c < P; ++c) {
                if (flags.GetValue(c * 8U) != 0) { ok = false; }
            }
            if (ok) {
                LocalTensor<int32_t> h = planArena.Get<int32_t>();
                Duplicate(h, static_cast<int32_t>(0), 32U); PipeBarrier<PIPE_ALL>();
                h.SetValue(0, static_cast<int32_t>(MagicA)); h.SetValue(1, static_cast<int32_t>(MagicB));
                h.SetValue(2, Version); h.SetValue(3, P); h.SetValue(4, O); h.SetValue(5, D);
                h.SetValue(6, M); h.SetValue(7, K); h.SetValue(8, tileCount); h.SetValue(9, planWords);
                SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(planGm, h, 32U); PipeBarrier<PIPE_MTE3>();
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
            }
        }
        // No header reader can race core 0's conditional, final commit.
        SyncAll<true>();
        ReadHeader();
        return HeaderMatches();
    }
    __aicore__ inline void ForwardTranspose(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, uint32_t columns, uint32_t rowStride) {
        uint64_t sl[16], dl[16];
        for (uint32_t r = 0; r < 8U; ++r) {
            sl[r] = reinterpret_cast<uint64_t>(src[r * rowStride].GetPhyAddr());
            sl[r + 8U] = reinterpret_cast<uint64_t>(src[r * rowStride + 8U].GetPhyAddr());
            dl[2U * r] = reinterpret_cast<uint64_t>(dst[r * 8U].GetPhyAddr());
            dl[2U * r + 1U] = reinterpret_cast<uint64_t>(dst[(r + 8U) * 8U].GetPhyAddr());
        }
        const uint8_t repeat = static_cast<uint8_t>(columns / 16U);
        TransDataTo5HDParams p{false, false, repeat,
            static_cast<uint16_t>(repeat == 1U ? 0U : 16U),
            static_cast<uint16_t>(repeat == 1U ? 0U : 2U)};
        TransDataTo5HD<uint32_t>(dl, sl, p); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InverseTranspose(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, uint32_t columns) {
        uint64_t sl[16], dl[16];
        for (uint32_t j = 0; j < 16U; ++j) { sl[j] = reinterpret_cast<uint64_t>(src[j * 8U].GetPhyAddr()); }
        for (uint32_t r = 0; r < 8U; ++r) {
            dl[2U * r] = reinterpret_cast<uint64_t>(dst[r * columns].GetPhyAddr());
            dl[2U * r + 1U] = reinterpret_cast<uint64_t>(dst[r * columns + 8U].GetPhyAddr());
        }
        const uint8_t repeat = static_cast<uint8_t>(columns / 16U);
        TransDataTo5HDParams p{false, false, repeat,
            static_cast<uint16_t>(repeat == 1U ? 0U : 2U),
            static_cast<uint16_t>(repeat == 1U ? 0U : 16U)};
        TransDataTo5HD<uint32_t>(dl, sl, p); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void GatherBlocks(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, LocalTensor<uint32_t> offsets, uint32_t blocks) {
        const uint32_t padded = Align(blocks, 8U);
        const GatherRepeatParams repeat{1, 8};
        for (uint32_t pos = 0; pos < padded; ) {
            uint32_t calls = (padded - pos) / 8U;
            if (calls > 255U) { calls = 255U; }
            Gatherb(dst[pos * 8U], src, offsets[pos], static_cast<uint8_t>(calls), repeat);
            pos += calls * 8U;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadSourceTile(uint32_t outer, uint32_t rows, uint32_t tile,
        LocalTensor<uint32_t> raw) {
        // The preceding transpose has finished consuming raw on V.
        const uint32_t begin = tile * K, count = M - begin < K ? M - begin : K;
        // A complete tile overwrites every word read by the K-column
        // transpose, even when this arena previously held a gathered matrix.
        // Partial tiles/outer batches retain the full original initialization.
        if (rows != 8U || count != K) {
            Duplicate(raw, static_cast<uint32_t>(0), 8U * (K + 8U));
        }
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyExtParams sourceCopy{static_cast<uint16_t>(rows), count * 4U,
            (M - count) * 4U, (K + 8U - Align(count, 8U)) / 8U, 0U};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(raw, sourceGm[outer * M + begin], sourceCopy, pad);
    }
    __aicore__ inline void Compute() {
        const uint32_t core = GetBlockIdx(), base = O / P, extra = O % P;
        const uint32_t start = core * base + (core < extra ? core : extra);
        const uint32_t end = start + base + (core < extra ? 1U : 0U);
        LocalTensor<uint32_t> raw0 = rawBuf.Get<uint32_t>();
        LocalTensor<uint32_t> raw1 = matrixBuf.Get<uint32_t>();
        LocalTensor<uint32_t> residentPlans = planArena.Get<uint32_t>();
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        for (uint32_t outer = start; outer < end; outer += 8U) {
            LocalTensor<uint32_t> trans = transformBuf.Get<uint32_t>();
            LocalTensor<uint32_t> acc = accBuf.Get<uint32_t>();
            LocalTensor<uint32_t> perm = trans;
            const uint32_t rows = end - outer < 8U ? end - outer : 8U;
            Duplicate(perm, static_cast<uint32_t>(0), 8U * DP); PipeBarrier<PIPE_ALL>();
            DataCopyExtParams selfCopy{static_cast<uint16_t>(rows), D * 4U,
                0U, (DP - Align(D, 8U)) / 8U, 0U};
            DataCopyPad(perm, selfGm[outer * D], selfCopy, pad); PipeBarrier<PIPE_ALL>();
            ForwardTranspose(acc, perm, DP, DP);
            LoadSourceTile(outer, rows, 0U, raw0);
            for (uint32_t tile = 0; tile < tileCount; ++tile) {
                // The preceding inverse Gather has drained its perm reader.
                // Join the source prefetch before this allocation becomes trans.
                PipeBarrier<PIPE_ALL>();
                LocalTensor<uint32_t> q = residentPlans[tile * planWords];
                const uint32_t maximum = q.GetValue(2U), valid = q.GetValue(3U);
                LocalTensor<uint32_t> raw = (tile & 1U) == 0U ? raw0 : raw1;
                LocalTensor<uint32_t> nextRaw = (tile & 1U) == 0U ? raw1 : raw0;
                if (tile + 1U < tileCount) {
                    // The other arena has no current raw or rank-update reader.
                    // Its original clear and DMA may overlap this transpose.
                    LoadSourceTile(outer, rows, tile + 1U, nextRaw);
                }
                Duplicate(trans[8U * K], static_cast<uint32_t>(0), 8U);
                ForwardTranspose(trans, raw, K, K + 8U);
                // ForwardTranspose's PIPE_V releases the current raw arena.
                // Reuse it as matrix while the other arena receives next raw.
                LocalTensor<uint32_t> matrix = raw;
                GatherBlocks(matrix, trans, q[sourcePlanBase], valid);
                // GatherBlocks ended with PIPE_V: no trans reader survives.
                GatherBlocks(perm, acc, q[permPlanBase], DP);
                LocalTensor<float> values = perm.ReinterpretCast<float>();
                LocalTensor<float> updates = matrix.ReinterpretCast<float>();
                for (uint32_t rank = 0; rank < maximum; ++rank) {
                    const uint32_t first = q.GetValue(prefixPlanBase + rank);
                    const uint32_t next = q.GetValue(prefixPlanBase + rank + 1U);
                    Add(values, values, updates[first * 8U], (next - first) * 8U);
                    PipeBarrier<PIPE_V>();
                }
                if (tile + 1U < tileCount) {
                    if (alternateAccumArenas) {
                        // Gather and every Add have drained their V readers.
                        // The just-updated permutation becomes the next carried
                        // accumulator; the old accumulator becomes next trans.
                        LocalTensor<uint32_t> previousAcc = acc;
                        acc = perm;
                        perm = previousAcc;
                        trans = perm;
                    } else {
                        Copy(acc, perm, static_cast<uint64_t>(64U),
                            static_cast<uint8_t>(DP / 8U), CopyRepeatParams{1, 1, 8, 8});
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    GatherBlocks(acc, perm, q[inversePlanBase], DP);
                }
            }
            InverseTranspose(perm, acc, DP);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams outputCopy{static_cast<uint16_t>(rows), D * 4U,
                (DP - Align(D, 8U)) / 8U, 0U, 0U};
            DataCopyPad(outputGm[outer * D], perm, outputCopy);
            SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> planArena;
    TBuf<TPosition::VECCALC> rawBuf, transformBuf, matrixBuf, accBuf;
    GlobalTensor<uint32_t> selfGm, sourceGm, outputGm;
    GlobalTensor<int32_t> indexGm, planGm;
    uint32_t O = 0, D = 0, M = 0, P = 0, DP = 0, AM = 0, K = 0;
    uint32_t scratchWords = 0, planBase = 0, tileCount = 0, planWords = 0;
    uint32_t coldNextBase = 0, coldPlanBase = 0, arenaWords = 0;
    uint32_t sourcePlanBase = 0, permPlanBase = 0, inversePlanBase = 0, prefixPlanBase = 0;
    bool alternateAccumArenas = false;
};
// END V423_RESIDENT_PLAN_CLASS

class FloatTiledGather {
public:
    __aicore__ inline bool TryProcess(GM_ADDR self, GM_ADDR index, GM_ADDR source,
        GM_ADDR output, GM_ADDR workspace, uint32_t total, uint32_t outer,
        uint32_t dim, uint32_t indices, uint32_t inner)
    {
        // Unsupported metadata leaves the old workspace completely untouched.
        if (inner != 1U || outer < 512U || dim <= 512U || dim >= 1024U ||
            indices < 4096U || indices > 8192U || workspace == nullptr ||
            static_cast<uint64_t>(outer) * dim != total ||
            static_cast<uint64_t>(outer) * indices > 0xFFFFFFFFULL) { return false; }
        O = outer; D = dim; M = indices; P = GetBlockNum();
        if (P == 0U) { return false; }
        DP = Align(D, 16U); AM = Align(M, 8U);
        scratchWords = Align((M + P - 1U) / P, 8U);
        if (scratchWords < P * 8U) { scratchWords = P * 8U; }
        planBase = 32U + AM + P * 8U;
        if (!SelectLayout()) { return false; }
        selfGm.SetGlobalBuffer((__gm__ uint32_t*)self, total);
        indexGm.SetGlobalBuffer((__gm__ int32_t*)index, M);
        sourceGm.SetGlobalBuffer((__gm__ uint32_t*)source, O * M);
        outputGm.SetGlobalBuffer((__gm__ uint32_t*)output, total);
        planGm.SetGlobalBuffer((__gm__ int32_t*)GetUserWorkspace(workspace), WorkspaceWords());
        pipe.InitBuffer(headBuf, DP * 4U);
        pipe.InitBuffer(nextBuf, (AM + scratchWords) * 4U);
        pipe.InitBuffer(planBuf, planWords * 4U);
        if (!ValidateCache() && !BuildAndPublish()) {
            InvalidateBeforeFallback();
            return false;
        }
        pipe.InitBuffer(alternatePlanBuf, planWords * 4U);
        pipe.InitBuffer(rawBuf, 8U * (K + 8U) * 4U);
        pipe.InitBuffer(transBuf, 8U * (K + 8U) * 4U);
        pipe.InitBuffer(matrixBuf, Align(8U * (K + 7U) * 4U, 32U));
        pipe.InitBuffer(accBuf, 8U * DP * 4U);
        pipe.InitBuffer(permBuf, 8U * DP * 4U);
        Compute();
        return true;
    }

private:
    static constexpr uint32_t MagicA = 0x49333937U;
    static constexpr uint32_t MagicB = 0x54494C45U;
    static constexpr uint32_t Version = 2U;
    __aicore__ inline uint32_t Align(uint32_t n, uint32_t a) const { return (n + a - 1U) / a * a; }
    __aicore__ inline int32_t Normalize(int32_t t) const {
        if (t < 0) { t += static_cast<int32_t>(D); }
        return t >= 0 && t < static_cast<int32_t>(D) ? t : -1;
    }
    __aicore__ inline uint32_t WorkspaceWords() const {
        const uint32_t local = (M + P - 1U) / P;
        const uint32_t occupied = local < D ? local : D;
        return 32U + 2U * P * DP + DP + P * Align(occupied * 8U, 16U) + AM + 64U * DP;
    }
    __aicore__ inline bool SelectLayout() {
        for (K = 1024U; K >= 128U; K /= 2U) {
            tileCount = (M + K - 1U) / K;
            sourcePlanBase = 32U;
            permPlanBase = sourcePlanBase + Align(K + 7U, 8U);
            inversePlanBase = permPlanBase + DP;
            prefixPlanBase = inversePlanBase + DP;
            planWords = prefixPlanBase + 256U;
            const uint64_t bytes = 2ULL * 8U * (K + 8U) * 4U +
                Align(8U * (K + 7U) * 4U, 32U) + 2ULL * 8U * DP * 4U +
                2ULL * planWords * 4ULL + DP * 4ULL + (AM + scratchWords) * 4ULL;
            if (bytes + 8192ULL <= 196608ULL &&
                static_cast<uint64_t>(tileCount) * planWords * 4U <= 196608ULL &&
                static_cast<uint64_t>(planBase) + tileCount * planWords <= WorkspaceWords()) { return true; }
        }
        return false;
    }
    __aicore__ inline void ReadHeader() {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        DataCopy(headBuf.Get<int32_t>(), planGm, 32U);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool HeaderMatches() {
        LocalTensor<int32_t> h = headBuf.Get<int32_t>();
        return static_cast<uint32_t>(h.GetValue(0)) == MagicA &&
            static_cast<uint32_t>(h.GetValue(1)) == MagicB &&
            static_cast<uint32_t>(h.GetValue(2)) == Version &&
            static_cast<uint32_t>(h.GetValue(3)) == P &&
            static_cast<uint32_t>(h.GetValue(4)) == O &&
            static_cast<uint32_t>(h.GetValue(5)) == D &&
            static_cast<uint32_t>(h.GetValue(6)) == M &&
            static_cast<uint32_t>(h.GetValue(7)) == K &&
            static_cast<uint32_t>(h.GetValue(8)) == tileCount &&
            static_cast<uint32_t>(h.GetValue(9)) == planWords;
    }
    __aicore__ inline bool ValidateCache() {
        ReadHeader();
        const bool headerValid = HeaderMatches();
        const uint32_t core = GetBlockIdx(), n = M / P, extra = M % P;
        const uint32_t start = core * n + (core < extra ? core : extra);
        const uint32_t count = n + (core < extra ? 1U : 0U);
        LocalTensor<int32_t> current = nextBuf.Get<int32_t>();
        LocalTensor<int32_t> cached = current[AM];
        if (headerValid && count != 0U) {
            DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(current, indexGm[start], copy, pad);
            DataCopyPad(cached, planGm[32U + start], copy, pad);
        }
        PipeBarrier<PIPE_ALL>();
        uint32_t mismatch = headerValid ? 0U : 1U;
        for (uint32_t i = 0; headerValid && i < count; ++i) {
            if (Normalize(current.GetValue(i)) != cached.GetValue(i)) { mismatch = 1U; break; }
        }
        Duplicate(cached, static_cast<int32_t>(0), 8U);
        PipeBarrier<PIPE_ALL>();
        cached.SetValue(0U, static_cast<int32_t>(mismatch));
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[32U + AM + core * 8U], cached, 8U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        // All header reads/validation finish before any cold writer can proceed.
        SyncAll<true>();
        DataCopy(cached, planGm[32U + AM], P * 8U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < P; ++c) { if (cached.GetValue(c * 8U) != 0) { return false; } }
        return true;
    }
    __aicore__ inline void ClearHeader() {
        LocalTensor<int32_t> h = headBuf.Get<int32_t>();
        Duplicate(h, static_cast<int32_t>(0), 32U);
        PipeBarrier<PIPE_ALL>();
        DataCopy(planGm, h, 32U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
    }
    __aicore__ inline void InvalidateBeforeFallback() {
        // Equivalent cohort protection to the repaired v392/v394 helper.
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        SyncAll<true>();
        if (GetBlockIdx() == 0U) { ClearHeader(); }
        SyncAll<true>();
    }
    __aicore__ inline bool BuildTile(uint32_t tile) {
        LocalTensor<int32_t> counts = headBuf.Get<int32_t>();
        LocalTensor<int32_t> current = nextBuf.Get<int32_t>();
        LocalTensor<int32_t> q = planBuf.Get<int32_t>();
        const uint32_t begin = tile * K, count = M - begin < K ? M - begin : K;
        Duplicate(counts, static_cast<int32_t>(0), DP);
        Duplicate(q, static_cast<int32_t>(0), planWords);
        PipeBarrier<PIPE_ALL>();
        uint32_t maximum = 0U, valid = 0U;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t target = current.GetValue(begin + i);
            if (target < 0) { continue; }
            const uint32_t next = static_cast<uint32_t>(counts.GetValue(target)) + 1U;
            if (next > 255U) { return false; } // No output has been written.
            counts.SetValue(target, static_cast<int32_t>(next));
            if (next > maximum) { maximum = next; }
            ++valid;
        }
        // Counting sort, descending count, stable target-number tiebreak.
        for (uint32_t t = 0; t < D; ++t) {
            const uint32_t c = static_cast<uint32_t>(counts.GetValue(t));
            q.SetValue(prefixPlanBase + c, q.GetValue(prefixPlanBase + c) + 1);
        }
        uint32_t cursor = 0U;
        for (int32_t c = 255; c >= 0; --c) {
            const uint32_t frequency = static_cast<uint32_t>(q.GetValue(prefixPlanBase + c));
            q.SetValue(prefixPlanBase + c, static_cast<int32_t>(cursor)); cursor += frequency;
        }
        for (uint32_t t = 0; t < D; ++t) {
            const uint32_t c = static_cast<uint32_t>(counts.GetValue(t));
            const uint32_t column = static_cast<uint32_t>(q.GetValue(prefixPlanBase + c));
            q.SetValue(prefixPlanBase + c, static_cast<int32_t>(column + 1U));
            q.SetValue(permPlanBase + column, static_cast<int32_t>(t));
            q.SetValue(inversePlanBase + t, static_cast<int32_t>(column));
        }
        uint32_t active = D, matrixCursor = 0U;
        q.SetValue(prefixPlanBase, 0);
        for (uint32_t rank = 0; rank < maximum; ++rank) {
            while (active > 0U && static_cast<uint32_t>(counts.GetValue(
                q.GetValue(permPlanBase + active - 1U))) <= rank) { --active; }
            matrixCursor += active;
            q.SetValue(prefixPlanBase + rank + 1U, static_cast<int32_t>(matrixCursor));
        }
        if (matrixCursor != valid || valid > count) { return false; }
        // Padding points at the permanent zero column after K transposed columns.
        for (uint32_t j = 0; j < Align(K + 7U, 8U); ++j) { q.SetValue(sourcePlanBase + j, static_cast<int32_t>(K * 32U)); }
        for (uint32_t t = 0; t < D; ++t) { counts.SetValue(t, 0); }
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t t = current.GetValue(begin + i);
            if (t < 0) { continue; }
            const uint32_t rank = static_cast<uint32_t>(counts.GetValue(t));
            counts.SetValue(t, static_cast<int32_t>(rank + 1U));
            const uint32_t column = static_cast<uint32_t>(q.GetValue(inversePlanBase + t));
            const uint32_t offset = static_cast<uint32_t>(q.GetValue(prefixPlanBase + rank)) + column;
            if (offset >= valid) { return false; }
            q.SetValue(sourcePlanBase + offset, static_cast<int32_t>(i * 32U));
        }
        for (uint32_t t = 0; t < D; ++t) {
            q.SetValue(permPlanBase + t, q.GetValue(permPlanBase + t) * 32);
            q.SetValue(inversePlanBase + t, q.GetValue(inversePlanBase + t) * 32);
        }
        q.SetValue(0U, static_cast<int32_t>(begin)); q.SetValue(1U, static_cast<int32_t>(count));
        q.SetValue(2U, static_cast<int32_t>(maximum)); q.SetValue(3U, static_cast<int32_t>(valid));
        q.SetValue(4U, static_cast<int32_t>(DP)); q.SetValue(5U, static_cast<int32_t>(K));
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[planBase + tile * planWords], q, planWords);
        // The next BuildTile clears q with V Duplicate. Drain the MTE3
        // payload read before allowing any cross-pipe overwrite of planBuf.
        PipeBarrier<PIPE_ALL>();
        return true;
    }
    __aicore__ inline bool BuildAndPublish() {
        const uint32_t core = GetBlockIdx();
        LocalTensor<int32_t> current = nextBuf.Get<int32_t>();
        LocalTensor<int32_t> flags = current[AM];
        if (core == 0U) { ClearHeader(); }
        // No core builds or publishes payload before the invalid header is visible.
        SyncAll<true>();
        bool localOk = true;
        for (uint32_t tile = core; tile < tileCount && localOk; tile += P) {
            const uint32_t begin = tile * K;
            const uint32_t count = M - begin < K ? M - begin : K;
            const uint32_t paddedCount = Align(count, 8U);
            // K and begin are 32-byte aligned. Only the final tile has padding.
            Duplicate(current[begin], static_cast<int32_t>(0), paddedCount);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(current[begin], indexGm[begin], copy, pad);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < count; ++i) {
                current.SetValue(begin + i, Normalize(current.GetValue(begin + i)));
            }
            // Define cache padding explicitly rather than relying on DMA pad contents.
            for (uint32_t i = count; i < paddedCount; ++i) { current.SetValue(begin + i, 0); }
            SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(planGm[32U + begin], current[begin], paddedCount);
            PipeBarrier<PIPE_ALL>();
            localOk = BuildTile(tile);
        }
        // All blocks, including those owning no tiles or failing a tile, publish
        // one complete status slot. Drain payload/UB readers before slot reuse.
        PipeBarrier<PIPE_ALL>();
        Duplicate(flags, static_cast<int32_t>(0), 8U);
        PipeBarrier<PIPE_ALL>();
        flags.SetValue(0U, localOk ? 0 : 1);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(planGm[32U + AM + core * 8U], flags, 8U);
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
        // Every normalized slice, tile payload and status is complete before commit.
        SyncAll<true>();
        if (core == 0U) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
            DataCopy(flags, planGm[32U + AM], P * 8U);
            PipeBarrier<PIPE_ALL>();
            bool ok = true;
            for (uint32_t c = 0U; c < P; ++c) {
                if (flags.GetValue(c * 8U) != 0) { ok = false; }
            }
            if (ok) {
                LocalTensor<int32_t> h = headBuf.Get<int32_t>();
                Duplicate(h, static_cast<int32_t>(0), 32U); PipeBarrier<PIPE_ALL>();
                h.SetValue(0, static_cast<int32_t>(MagicA)); h.SetValue(1, static_cast<int32_t>(MagicB));
                h.SetValue(2, Version); h.SetValue(3, P); h.SetValue(4, O); h.SetValue(5, D);
                h.SetValue(6, M); h.SetValue(7, K); h.SetValue(8, tileCount); h.SetValue(9, planWords);
                SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(planGm, h, 32U); PipeBarrier<PIPE_MTE3>();
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(planGm);
            }
        }
        // No header reader can race core 0's conditional, final commit.
        SyncAll<true>();
        ReadHeader();
        return HeaderMatches();
    }
    __aicore__ inline void ForwardTranspose(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, uint32_t columns, uint32_t rowStride) {
        uint64_t sl[16], dl[16];
        for (uint32_t r = 0; r < 8U; ++r) {
            sl[r] = reinterpret_cast<uint64_t>(src[r * rowStride].GetPhyAddr());
            sl[r + 8U] = reinterpret_cast<uint64_t>(src[r * rowStride + 8U].GetPhyAddr());
            dl[2U * r] = reinterpret_cast<uint64_t>(dst[r * 8U].GetPhyAddr());
            dl[2U * r + 1U] = reinterpret_cast<uint64_t>(dst[(r + 8U) * 8U].GetPhyAddr());
        }
        const uint8_t repeat = static_cast<uint8_t>(columns / 16U);
        TransDataTo5HDParams p{false, false, repeat,
            static_cast<uint16_t>(repeat == 1U ? 0U : 16U),
            static_cast<uint16_t>(repeat == 1U ? 0U : 2U)};
        TransDataTo5HD<uint32_t>(dl, sl, p); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InverseTranspose(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, uint32_t columns) {
        uint64_t sl[16], dl[16];
        for (uint32_t j = 0; j < 16U; ++j) { sl[j] = reinterpret_cast<uint64_t>(src[j * 8U].GetPhyAddr()); }
        for (uint32_t r = 0; r < 8U; ++r) {
            dl[2U * r] = reinterpret_cast<uint64_t>(dst[r * columns].GetPhyAddr());
            dl[2U * r + 1U] = reinterpret_cast<uint64_t>(dst[r * columns + 8U].GetPhyAddr());
        }
        const uint8_t repeat = static_cast<uint8_t>(columns / 16U);
        TransDataTo5HDParams p{false, false, repeat,
            static_cast<uint16_t>(repeat == 1U ? 0U : 2U),
            static_cast<uint16_t>(repeat == 1U ? 0U : 16U)};
        TransDataTo5HD<uint32_t>(dl, sl, p); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void GatherBlocks(LocalTensor<uint32_t> dst,
        LocalTensor<uint32_t> src, LocalTensor<uint32_t> offsets, uint32_t blocks) {
        const uint32_t padded = Align(blocks, 8U);
        const GatherRepeatParams repeat{1, 8};
        for (uint32_t pos = 0; pos < padded; ) {
            uint32_t calls = (padded - pos) / 8U;
            if (calls > 255U) { calls = 255U; }
            Gatherb(dst[pos * 8U], src, offsets[pos], static_cast<uint8_t>(calls), repeat);
            pos += calls * 8U;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadTile(uint32_t outer, uint32_t rows, uint32_t tile,
        LocalTensor<uint32_t> destinationPlan, LocalTensor<uint32_t> raw) {
        // Caller has finished the previous use of this plan and raw input.
        // The preceding transpose and this zero-fill are both on V. Hand raw
        // to MTE2 only after both complete; do not wait for the new DMA here.
        Duplicate(raw, static_cast<uint32_t>(0), 8U * (K + 8U));
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        const uint32_t begin = tile * K, count = M - begin < K ? M - begin : K;
        DataCopyExtParams sourceCopy{static_cast<uint16_t>(rows), count * 4U,
            (M - count) * 4U, (K - Align(count, 8U)) / 8U, 0U};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(raw, sourceGm[outer * M + begin], sourceCopy, pad);
        DataCopy(destinationPlan.ReinterpretCast<int32_t>(),
            planGm[planBase + tile * planWords], planWords);
    }
    __aicore__ inline void Compute() {
        const uint32_t core = GetBlockIdx(), base = O / P, extra = O % P;
        const uint32_t start = core * base + (core < extra ? core : extra);
        const uint32_t end = start + base + (core < extra ? 1U : 0U);
        LocalTensor<uint32_t> raw = rawBuf.Get<uint32_t>(), trans = transBuf.Get<uint32_t>();
        LocalTensor<uint32_t> matrix = matrixBuf.Get<uint32_t>(), acc = accBuf.Get<uint32_t>();
        LocalTensor<uint32_t> perm = permBuf.Get<uint32_t>();
        LocalTensor<uint32_t> plan0 = planBuf.Get<uint32_t>();
        LocalTensor<uint32_t> plan1 = alternatePlanBuf.Get<uint32_t>();
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        for (uint32_t outer = start; outer < end; outer += 8U) {
            const uint32_t rows = end - outer < 8U ? end - outer : 8U;
            Duplicate(perm, static_cast<uint32_t>(0), 8U * DP); PipeBarrier<PIPE_ALL>();
            DataCopyExtParams selfCopy{static_cast<uint16_t>(rows), D * 4U,
                0U, (DP - Align(D, 8U)) / 8U, 0U};
            DataCopyPad(perm, selfGm[outer * D], selfCopy, pad); PipeBarrier<PIPE_ALL>();
            ForwardTranspose(acc, perm, DP, DP);
            LoadTile(outer, rows, 0U, plan0, raw);
            for (uint32_t tile = 0; tile < tileCount; ++tile) {
                // Wait for this tile's prefetched input/plan and the preceding
                // tile's V work. Both plans' older readers are now complete.
                PipeBarrier<PIPE_ALL>();
                LocalTensor<uint32_t> q = (tile & 1U) == 0U ? plan0 : plan1;
                const uint32_t maximum = q.GetValue(2U), valid = q.GetValue(3U);
                Duplicate(trans[8U * K], static_cast<uint32_t>(0), 8U);
                ForwardTranspose(trans, raw, K, K);
                if (tile + 1U < tileCount) {
                    LocalTensor<uint32_t> nextPlan = (tile & 1U) == 0U ? plan1 : plan0;
                    // Compute below consumes trans and q, never raw/nextPlan.
                    LoadTile(outer, rows, tile + 1U, nextPlan, raw);
                }
                GatherBlocks(matrix, trans, q[sourcePlanBase], valid);
                GatherBlocks(perm, acc, q[permPlanBase], DP);
                LocalTensor<float> values = perm.ReinterpretCast<float>();
                LocalTensor<float> updates = matrix.ReinterpretCast<float>();
                for (uint32_t rank = 0; rank < maximum; ++rank) {
                    const uint32_t first = q.GetValue(prefixPlanBase + rank);
                    const uint32_t next = q.GetValue(prefixPlanBase + rank + 1U);
                    Add(values, values, updates[first * 8U], (next - first) * 8U);
                    PipeBarrier<PIPE_V>();
                }
                GatherBlocks(acc, perm, q[inversePlanBase], DP);
            }
            InverseTranspose(perm, acc, DP);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams outputCopy{static_cast<uint16_t>(rows), D * 4U,
                (DP - Align(D, 8U)) / 8U, 0U, 0U};
            DataCopyPad(outputGm[outer * D], perm, outputCopy);
            SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> headBuf, nextBuf, planBuf, alternatePlanBuf;
    TBuf<TPosition::VECCALC> rawBuf, transBuf, matrixBuf, accBuf, permBuf;
    GlobalTensor<uint32_t> selfGm, sourceGm, outputGm;
    GlobalTensor<int32_t> indexGm, planGm;
    uint32_t O = 0, D = 0, M = 0, P = 0, DP = 0, AM = 0, K = 0;
    uint32_t scratchWords = 0, planBase = 0, tileCount = 0, planWords = 0;
    uint32_t sourcePlanBase = 0, permPlanBase = 0, inversePlanBase = 0, prefixPlanBase = 0;
};
} // namespace V397Draft
#endif
// END V403_SELF_CONTAINED_FP32_IMPLEMENTATION

using namespace AscendC;

namespace {
constexpr uint32_t kMaxAccumBufferBytes = 32768;
constexpr uint32_t kGenericChainBufferBytes = 65536;
constexpr uint32_t kGenericGroupedTotalBytes = 131072;
constexpr uint32_t kGenericWaveTotalBytes = 196608;
constexpr uint32_t kGenericWaveValueElements = 8192;
// Keep only the count matrix plus the owned wave offsets in accumBuf.  Bucket
// starts are read from the shared plan while it is assembled, so 2304 words
// cover the Case5 uniform plan and leave UB for a two-row reduction group.
constexpr uint32_t kParallelSharedWaveValueElements = 2304;
constexpr uint32_t kGenericOuterBatch = 512;
constexpr uint32_t kHalfOuterBatch = 256;
constexpr uint32_t kVectorTileElements = 4096;
constexpr uint32_t kGenericVectorAccumBytes = 131072;
constexpr uint32_t kC4Outer2AccumBytes = 182400;
constexpr uint32_t kC4TargetTiledUbSafetyBytes = 4096;
constexpr uint32_t kC4TargetTiledAccumSafetyBytes = 8192;
constexpr uint32_t kGenericVectorTotalBytes = 196608;
constexpr uint32_t kGenericVectorMaxSourceRows = 32;
constexpr uint32_t kGenericVectorMaxInner = 2048;
constexpr uint32_t kGenericVectorMinTotal = 4096;
constexpr uint32_t kWaveOuterBatchMax = 8;
constexpr uint32_t kHalfWaveOuterBatchMax = 64;
constexpr uint32_t kHalfGatherOuterBatch = 3;
constexpr uint32_t kHalfCountBucketCount = 8;
constexpr uint32_t kHalfCountBucketGatherOuter = 16;
constexpr uint32_t kHalfBucketBlockOuterLanes = 16;
constexpr uint32_t kHalfCountBucketUbSafetyBytes = 8192;
constexpr uint32_t kHalfBucketCacheHeaderWords = 32;
constexpr uint32_t kHalfBucketCacheFlagStride = 8;
constexpr uint32_t kHalfBucketCacheMagicA = 0x49414842U;
constexpr uint32_t kHalfBucketCacheMagicB = 0x504C414EU;
constexpr uint32_t kHalfBucketCacheVersion = 11U;
constexpr uint32_t kBlockReduceTailMinWaves = 8;
constexpr uint32_t kSharedPlanHeaderWords = 32;
constexpr uint32_t kSharedPlanWaveSlack = 64;
constexpr uint32_t kSharedPlanCacheMagicA = 0x49414335U;
constexpr uint32_t kSharedPlanCacheMagicB = 0x504C414EU;
constexpr uint32_t kSharedPlanCacheVersion = 4U;
constexpr uint32_t kSharedPlanBankGroupCount = 8U;
constexpr uint32_t kSharedPlanBankGroupShift = 4U;
constexpr uint32_t kBatchedPlanCacheMagicA = 0x49414332U;
constexpr uint32_t kBatchedPlanCacheMagicB = 0x4C494E4BU;
constexpr uint32_t kBatchedPlanCacheVersion = 1U;
constexpr uint32_t kBatchedPlanCacheFlagStride = 8U;
constexpr uint32_t kBatchedSharedWaveCapacity = 64U;
constexpr uint32_t kBatchedSharedBucketWaveOffset = 16U;
constexpr uint32_t kBatchedSharedCacheMagicA = 0x49415732U;
constexpr uint32_t kBatchedSharedCacheMagicB = 0x504C414EU;
constexpr uint32_t kBatchedSharedCacheVersion = 22U;
constexpr uint32_t kBatchedBucketBatch = 4U;
constexpr uint32_t kBatchedExecutionBatch = 8U;
constexpr uint32_t kBatchedNextBufferBytes = 41984U;
constexpr uint32_t kCacheMagicAWord = 0U;
constexpr uint32_t kCacheMagicBWord = 1U;
constexpr uint32_t kCacheVersionWord = 2U;
constexpr uint32_t kCacheBlockNumWord = 3U;
constexpr uint32_t kCacheOuterSizeWord = 4U;
constexpr uint32_t kCacheDimSizeWord = 5U;
constexpr uint32_t kCacheIndexSizeWord = 6U;
constexpr uint32_t kCacheFallbackWord = 7U;
constexpr uint32_t kCacheWaveCountWord = 8U;
constexpr uint32_t kCacheWaveStrideWord = 9U;
constexpr uint32_t kCacheZeroElementWord = 10U;
constexpr uint32_t kCacheValueStrideWord = 11U;
constexpr uint32_t kCacheSelfElementWord = 12U;
constexpr uint32_t kCacheOuterGroupWord = 13U;
constexpr uint32_t kCacheIncludeSelfWord = 14U;
constexpr uint32_t kCacheSourceStrideWord = 15U;
constexpr uint32_t kCacheBlockGroupsWord = 16U;
constexpr uint32_t kCacheRemainderWord = 17U;
constexpr uint32_t kCacheGroupStrideWord = 18U;
constexpr uint32_t kCacheGatherElementsWord = 19U;
constexpr uint32_t kCacheOuterBatchWord = 20U;
constexpr uint32_t kCacheFirstWaveBytesWord = 21U;
constexpr uint32_t kCacheOverflowWaveBytesWord = 22U;
constexpr uint32_t kCacheOverflowWaveElementsWord = 23U;
template <typename T>
__aicore__ inline T AddValue(T lhs, T rhs)
{
    return lhs + rhs;
}

__aicore__ inline int8_t AddValue(int8_t lhs, int8_t rhs)
{
    return static_cast<int8_t>(static_cast<int16_t>(lhs) + static_cast<int16_t>(rhs));
}

__aicore__ inline uint32_t AlignUp32(uint32_t value)
{
    return (value + 31U) & ~31U;
}

template <typename T>
__aicore__ inline bool UseGenericBlockReduce(T*)
{
    return false;
}

__aicore__ inline bool UseGenericBlockReduce(float*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseParallelSharedPlan(T*)
{
    return false;
}

__aicore__ inline bool UseParallelSharedPlan(float*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseInt32GroupedRepeat(T*)
{
    return false;
}

__aicore__ inline bool UseInt32GroupedRepeat(int32_t*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseAtomicVector(T*)
{
    return false;
}

template <typename T>
__aicore__ inline bool UseDirectPaddedAtomic(T*)
{
    return false;
}

__aicore__ inline bool UseDirectPaddedAtomic(float*)
{
    return true;
}

__aicore__ inline bool UseDirectPaddedAtomic(int32_t*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseUnalignedDirectPaddedAtomic(T*)
{
    return false;
}

__aicore__ inline bool UseUnalignedDirectPaddedAtomic(int32_t*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseDirectScalarAtomic(T*)
{
    return false;
}

__aicore__ inline bool UseDirectScalarAtomic(float*)
{
    return true;
}

__aicore__ inline bool UseDirectScalarAtomic(int32_t*)
{
    return true;
}

template <typename T>
__aicore__ inline bool UseBatchedTargetGroup(T*)
{
    return false;
}

__aicore__ inline bool UseBatchedTargetGroup(int32_t*)
{
    return true;
}

__aicore__ inline bool UseAtomicVector(float*)
{
    return true;
}

__aicore__ inline bool UseAtomicVector(int32_t*)
{
    return true;
}

template <typename T>
__aicore__ inline void AccumulatePreparedGenericWave(
    const LocalTensor<T>& valueLocal,
    const LocalTensor<T>& waveValueLocal,
    uint32_t dimSize, uint32_t waveCount, uint32_t waveStride)
{
    if (waveCount == 0U) {
        return;
    }

    const uint32_t waveRowBytes = waveStride * sizeof(T);
    const uint32_t waveRowBlocks = waveRowBytes / 32U;
    constexpr uint32_t elementsPerRepeat = 256U / sizeof(T);
    const uint32_t repeatCallCount =
        (dimSize + elementsPerRepeat - 1U) / elementsPerRepeat;
    if (waveCount <= 255U && waveRowBytes % 32U == 0U &&
        waveRowBlocks <= 255U && repeatCallCount < waveCount) {
        const BinaryRepeatParams repeatParams{1, 1, 1, 0,
            static_cast<uint8_t>(waveRowBlocks), 0};
        for (uint32_t dimStart = 0U; dimStart < dimSize;
             dimStart += elementsPerRepeat) {
            uint32_t count = dimSize - dimStart;
            if (count > elementsPerRepeat) {
                count = elementsPerRepeat;
            }
            Add(valueLocal[dimStart], waveValueLocal[dimStart],
                valueLocal[dimStart], static_cast<uint64_t>(count),
                static_cast<uint8_t>(waveCount), repeatParams);
        }
        return;
    }

    for (uint32_t wave = 0U; wave < waveCount; ++wave) {
        Add(valueLocal, valueLocal,
            waveValueLocal[wave * waveStride], dimSize);
    }
}

__aicore__ inline void AccumulatePreparedGenericWave(
    const LocalTensor<int32_t>& valueLocal,
    const LocalTensor<int32_t>& waveValueLocal,
    uint32_t dimSize, uint32_t waveCount, uint32_t waveStride)
{
    if (waveCount == 0U) {
        return;
    }

    uint32_t activeRows = waveCount;
    while (activeRows > 1U) {
        const uint32_t pairedRows = activeRows / 2U;
        Add(waveValueLocal, waveValueLocal,
            waveValueLocal[pairedRows * waveStride],
            pairedRows * waveStride);
        if ((activeRows & 1U) != 0U) {
            Add(waveValueLocal, waveValueLocal,
                waveValueLocal[(activeRows - 1U) * waveStride],
                waveStride);
        }
        activeRows = pairedRows;
    }
    Add(valueLocal, valueLocal, waveValueLocal, dimSize);
}

__aicore__ inline void ReducePreparedInt32Rows(
    const LocalTensor<int32_t>& rowsLocal,
    uint32_t rowCount, uint32_t rowStride)
{
    uint32_t activeRows = rowCount;
    while (activeRows > 1U) {
        const uint32_t pairedRows = activeRows / 2U;
        Add(rowsLocal, rowsLocal,
            rowsLocal[pairedRows * rowStride],
            pairedRows * rowStride);
        if ((activeRows & 1U) != 0U) {
            Add(rowsLocal, rowsLocal,
                rowsLocal[(activeRows - 1U) * rowStride],
                rowStride);
        }
        activeRows = pairedRows;
    }
}

template <typename T>
__aicore__ inline void AccumulateGenericWave(const LocalTensor<T>& valueLocal,
    const LocalTensor<T>& sourceLocal, const LocalTensor<T>& waveValueLocal,
    const LocalTensor<uint32_t>& waveOffsetLocal, const LocalTensor<T>&,
    uint32_t dimSize, uint32_t waveCount, uint32_t waveStride,
    uint32_t, uint32_t, uint32_t, bool, T*)
{
    if (waveCount == 0) {
        return;
    }
    Gather(waveValueLocal, sourceLocal, waveOffsetLocal, 0, waveCount * waveStride);

    const uint32_t waveRowBytes = waveStride * sizeof(T);
    const uint32_t waveRowBlocks = waveRowBytes / 32;
    constexpr uint32_t elementsPerRepeat = 256 / sizeof(T);
    const uint32_t repeatCallCount =
        (dimSize + elementsPerRepeat - 1) / elementsPerRepeat;
    if (waveCount <= 255 && waveRowBytes % 32 == 0 && waveRowBlocks <= 255 &&
        repeatCallCount < waveCount) {
        const BinaryRepeatParams repeatParams{1, 1, 1, 0,
            static_cast<uint8_t>(waveRowBlocks), 0};
        for (uint32_t dimStart = 0; dimStart < dimSize; dimStart += elementsPerRepeat) {
            uint32_t count = dimSize - dimStart;
            if (count > elementsPerRepeat) {
                count = elementsPerRepeat;
            }
            Add(valueLocal[dimStart], waveValueLocal[dimStart], valueLocal[dimStart],
                static_cast<uint64_t>(count), static_cast<uint8_t>(waveCount), repeatParams);
        }
        return;
    }

    for (uint32_t wave = 0; wave < waveCount; ++wave) {
        Add(valueLocal, valueLocal, waveValueLocal[wave * waveStride], dimSize);
    }
}

__aicore__ inline void AccumulateGenericWave(const LocalTensor<float>& valueLocal,
    const LocalTensor<float>& sourceLocal, const LocalTensor<float>& waveValueLocal,
    const LocalTensor<uint32_t>& waveOffsetLocal, const LocalTensor<float>& reduceLocal,
    uint32_t dimSize, uint32_t waveCount, uint32_t waveStride,
    uint32_t blockGroupCount, uint32_t remainder, uint32_t groupStride,
    bool includeSelf, float*)
{
    if (waveCount == 0) {
        return;
    }
    const uint32_t reduceRepeat = waveStride / 8;
    if (includeSelf) {
        Gather(waveValueLocal, sourceLocal, waveOffsetLocal, 0, groupStride);
        BlockReduceSum<float>(valueLocal, waveValueLocal, reduceRepeat, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        return;
    }
    if (blockGroupCount == 0) {
        Gather(waveValueLocal, sourceLocal, waveOffsetLocal, 0, waveCount * waveStride);
        for (uint32_t wave = 0; wave < waveCount; ++wave) {
            Add(valueLocal, valueLocal, waveValueLocal[wave * waveStride], dimSize);
        }
        return;
    }

    for (uint32_t group = 0; group < blockGroupCount; ++group) {
        Gather(waveValueLocal, sourceLocal, waveOffsetLocal[group * groupStride], 0,
            groupStride);
        BlockReduceSum<float>(reduceLocal, waveValueLocal, reduceRepeat, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, dimSize);
    }

    if (remainder == 0) {
        return;
    }
    const uint32_t remainderOffset = blockGroupCount * groupStride;
    Gather(waveValueLocal, sourceLocal, waveOffsetLocal[remainderOffset], 0,
        remainder * waveStride);
    for (uint32_t wave = 0; wave < remainder; ++wave) {
        Add(valueLocal, valueLocal, waveValueLocal[wave * waveStride], dimSize);
    }
}

template <typename T>
__aicore__ inline void AccumulateGenericSplitWave(const LocalTensor<T>&,
    const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<uint32_t>&,
    const LocalTensor<uint32_t>&, const LocalTensor<T>&, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, bool, T*)
{
}

__aicore__ inline void AccumulateGenericSplitWave(const LocalTensor<float>& valueLocal,
    const LocalTensor<float>& sourceLocal, const LocalTensor<float>& waveValueLocal,
    const LocalTensor<uint32_t>& firstWaveOffsetLocal,
    const LocalTensor<uint32_t>& overflowWaveOffsetLocal,
    const LocalTensor<float>& reduceLocal, uint32_t dimSize, uint32_t waveStride,
    uint32_t blockGroupCount, uint32_t remainder, uint32_t groupStride,
    bool includeSelf, float*)
{
    const uint32_t reduceRepeat = waveStride / 8;
    Gather(waveValueLocal, sourceLocal, firstWaveOffsetLocal, 0, groupStride);
    if (includeSelf) {
        BlockReduceSum<float>(valueLocal, waveValueLocal, reduceRepeat, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    } else {
        BlockReduceSum<float>(reduceLocal, waveValueLocal, reduceRepeat, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, dimSize);
    }

    for (uint32_t group = 0; group < blockGroupCount; ++group) {
        Gather(waveValueLocal, sourceLocal,
            overflowWaveOffsetLocal[group * groupStride], 0, groupStride);
        BlockReduceSum<float>(reduceLocal, waveValueLocal, reduceRepeat, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, dimSize);
    }

    if (remainder == 0) {
        return;
    }
    const uint32_t remainderOffset = blockGroupCount * groupStride;
    Gather(waveValueLocal, sourceLocal, overflowWaveOffsetLocal[remainderOffset], 0,
        remainder * waveStride);
    for (uint32_t wave = 0; wave < remainder; ++wave) {
        Add(valueLocal, valueLocal, waveValueLocal[wave * waveStride], dimSize);
    }
}

template <typename T>
__aicore__ inline void AccumulateGenericSplitWavePair(const LocalTensor<T>&,
    const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&,
    const LocalTensor<uint32_t>&, const LocalTensor<uint32_t>&,
    const LocalTensor<T>&, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, bool, T*)
{
}

__aicore__ inline void AccumulateGenericSplitWavePair(
    const LocalTensor<float>& valueLocal,
    const LocalTensor<float>& firstSourceLocal,
    const LocalTensor<float>& secondSourceLocal,
    const LocalTensor<float>& waveValueLocal,
    const LocalTensor<uint32_t>& firstWaveOffsetLocal,
    const LocalTensor<uint32_t>& overflowWaveOffsetLocal,
    const LocalTensor<float>& reduceLocal, uint32_t dimSize,
    uint32_t waveStride, uint32_t blockGroupCount, uint32_t remainder,
    uint32_t groupStride, bool includeSelf, float*)
{
    const uint32_t reduceRepeat = waveStride / 8;
    const uint32_t pairRepeat = 2 * reduceRepeat;
    const uint32_t pairElements = 2 * waveStride;

    Gather(waveValueLocal, firstSourceLocal, firstWaveOffsetLocal, 0,
        groupStride);
    Gather(waveValueLocal[groupStride], secondSourceLocal,
        firstWaveOffsetLocal, 0, groupStride);
    if (includeSelf) {
        BlockReduceSum<float>(valueLocal, waveValueLocal, pairRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    } else {
        BlockReduceSum<float>(reduceLocal, waveValueLocal, pairRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, pairElements);
    }

    for (uint32_t group = 0; group < blockGroupCount; ++group) {
        LocalTensor<uint32_t> groupOffset =
            overflowWaveOffsetLocal[group * groupStride];
        Gather(waveValueLocal, firstSourceLocal, groupOffset, 0,
            groupStride);
        Gather(waveValueLocal[groupStride], secondSourceLocal,
            groupOffset, 0, groupStride);
        BlockReduceSum<float>(reduceLocal, waveValueLocal, pairRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, pairElements);
    }

    if (remainder == 0) {
        return;
    }
    const uint32_t remainderOffset = blockGroupCount * groupStride;
    const uint32_t remainderElements = remainder * waveStride;
    Gather(waveValueLocal, firstSourceLocal,
        overflowWaveOffsetLocal[remainderOffset], 0, remainderElements);
    Gather(waveValueLocal[remainderElements], secondSourceLocal,
        overflowWaveOffsetLocal[remainderOffset], 0, remainderElements);
    for (uint32_t wave = 0; wave < remainder; ++wave) {
        Add(valueLocal, valueLocal,
            waveValueLocal[wave * waveStride], dimSize);
        Add(valueLocal[waveStride], valueLocal[waveStride],
            waveValueLocal[(remainder + wave) * waveStride], dimSize);
    }
}

template <typename T>
__aicore__ inline void AccumulateGenericSplitWaveTriple(const LocalTensor<T>&,
    const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&,
    const LocalTensor<T>&, const LocalTensor<uint32_t>&,
    const LocalTensor<uint32_t>&, const LocalTensor<T>&, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, bool, T*)
{
}

__aicore__ inline void AccumulateGenericSplitWaveTriple(
    const LocalTensor<float>& valueLocal,
    const LocalTensor<float>& firstSourceLocal,
    const LocalTensor<float>& secondSourceLocal,
    const LocalTensor<float>& thirdSourceLocal,
    const LocalTensor<float>& waveValueLocal,
    const LocalTensor<uint32_t>& firstWaveOffsetLocal,
    const LocalTensor<uint32_t>& overflowWaveOffsetLocal,
    const LocalTensor<float>& reduceLocal, uint32_t dimSize,
    uint32_t waveStride, uint32_t blockGroupCount, uint32_t remainder,
    uint32_t groupStride, bool includeSelf, float*)
{
    const uint32_t reduceRepeat = waveStride / 8;
    const uint32_t tripleRepeat = 3 * reduceRepeat;
    const uint32_t tripleElements = 3 * waveStride;

    Gather(waveValueLocal, firstSourceLocal, firstWaveOffsetLocal, 0,
        groupStride);
    Gather(waveValueLocal[groupStride], secondSourceLocal,
        firstWaveOffsetLocal, 0, groupStride);
    Gather(waveValueLocal[2 * groupStride], thirdSourceLocal,
        firstWaveOffsetLocal, 0, groupStride);
    if (includeSelf) {
        BlockReduceSum<float>(valueLocal, waveValueLocal, tripleRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    } else {
        BlockReduceSum<float>(reduceLocal, waveValueLocal, tripleRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, tripleElements);
    }

    for (uint32_t group = 0; group < blockGroupCount; ++group) {
        LocalTensor<uint32_t> groupOffset =
            overflowWaveOffsetLocal[group * groupStride];
        Gather(waveValueLocal, firstSourceLocal, groupOffset, 0,
            groupStride);
        Gather(waveValueLocal[groupStride], secondSourceLocal,
            groupOffset, 0, groupStride);
        Gather(waveValueLocal[2 * groupStride], thirdSourceLocal,
            groupOffset, 0, groupStride);
        BlockReduceSum<float>(reduceLocal, waveValueLocal, tripleRepeat,
            64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(valueLocal, valueLocal, reduceLocal, tripleElements);
    }

    if (remainder == 0) {
        return;
    }
    const uint32_t remainderOffset = blockGroupCount * groupStride;
    const uint32_t remainderElements = remainder * waveStride;
    Gather(waveValueLocal, firstSourceLocal,
        overflowWaveOffsetLocal[remainderOffset], 0, remainderElements);
    Gather(waveValueLocal[remainderElements], secondSourceLocal,
        overflowWaveOffsetLocal[remainderOffset], 0, remainderElements);
    Gather(waveValueLocal[2 * remainderElements], thirdSourceLocal,
        overflowWaveOffsetLocal[remainderOffset], 0, remainderElements);
    for (uint32_t wave = 0; wave < remainder; ++wave) {
        Add(valueLocal, valueLocal,
            waveValueLocal[wave * waveStride], dimSize);
        Add(valueLocal[waveStride], valueLocal[waveStride],
            waveValueLocal[(remainder + wave) * waveStride], dimSize);
        Add(valueLocal[2 * waveStride], valueLocal[2 * waveStride],
            waveValueLocal[(2 * remainder + wave) * waveStride], dimSize);
    }
}

__aicore__ inline void BuildIndexChains(const LocalTensor<int32_t>& headLocal,
    const LocalTensor<int32_t>& nextLocal, GlobalTensor<int32_t>& indexGlobal,
    uint32_t indexSize, uint32_t dimSize)
{
    for (uint32_t dim = 0; dim < dimSize; ++dim) {
        headLocal.SetValue(dim, -1);
    }
    for (uint32_t cursor = indexSize; cursor > 0; --cursor) {
        const uint32_t i = cursor - 1;
        int32_t target = indexGlobal.GetValue(i);
        if (target < 0) {
            target += static_cast<int32_t>(dimSize);
        }
        if (target < 0 || target >= static_cast<int32_t>(dimSize)) {
            continue;
        }

        const uint32_t targetOffset = static_cast<uint32_t>(target);
        nextLocal.SetValue(i, headLocal.GetValue(targetOffset));
        headLocal.SetValue(targetOffset, static_cast<int32_t>(i));
    }
}

__aicore__ inline void BuildIndexChainsRangeCached(const LocalTensor<int32_t>& headLocal,
    const LocalTensor<int32_t>& nextLocal, GlobalTensor<int32_t>& indexGlobal,
    uint32_t indexSize, uint32_t dimSize, uint32_t dimStart, uint32_t dimEnd)
{
    for (uint32_t dim = dimStart; dim < dimEnd; ++dim) {
        headLocal.SetValue(dim, -1);
    }

    DataCopyExtParams indexParams{1, indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(nextLocal, indexGlobal, indexParams, padParams);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t cursor = indexSize; cursor > 0; --cursor) {
        const uint32_t i = cursor - 1;
        int32_t target = nextLocal.GetValue(i);
        if (target < 0) {
            target += static_cast<int32_t>(dimSize);
        }
        if (target < static_cast<int32_t>(dimStart) || target >= static_cast<int32_t>(dimEnd)) {
            continue;
        }

        const uint32_t targetOffset = static_cast<uint32_t>(target);
        nextLocal.SetValue(i, headLocal.GetValue(targetOffset));
        headLocal.SetValue(targetOffset, static_cast<int32_t>(i));
    }
}

template <typename T>
class KernelIndexAdd {
public:
    __aicore__ inline KernelIndexAdd() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
        uint32_t dimSize, uint32_t indexSize, uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ T*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ T*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ T*)output, totalSize);
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t paddedInner =
            ((innerSize + alignElements - 1) / alignElements) * alignElements;
        GM_ADDR userWorkspace = workspace;
        const bool useGroupedPaddedAtomic =
            innerSize >= alignElements && innerSize % alignElements != 0 &&
            outerSize < 32 && dimSize <= 65535 && indexSize >= 512 &&
            indexSize < 2048 && totalSize >= 4096;
        if (UseAtomicVector((T*)nullptr) &&
            !UseDirectPaddedAtomic((T*)nullptr) && useGroupedPaddedAtomic) {
            userWorkspace = GetUserWorkspace(workspace);
        }
        this->workspaceGm.SetGlobalBuffer((__gm__ T*)userWorkspace,
            outerSize * dimSize * paddedInner);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
        this->useBatchedPlanCache = UseBatchedTargetGroup((T*)nullptr) &&
            innerSize == 1 && outerSize > 256 && dimSize >= 64 &&
            dimSize < 512 && dimSize % 8U == 0 && indexSize >= 512 &&
            indexSize < 2048 && totalSize >= 4096;
        if (this->useBatchedPlanCache) {
            this->batchedAlignedIndex =
                AlignUp32(indexSize * sizeof(int32_t)) / sizeof(int32_t);
            this->batchedAlignedDim =
                AlignUp32(dimSize * sizeof(int32_t)) / sizeof(int32_t);
            this->batchedAlignedGroupCount = AlignUp32(
                (dimSize / 8U) * sizeof(int32_t)) / sizeof(int32_t);
            this->batchedCachedIndexBase = kSharedPlanHeaderWords;
            this->batchedPackedNextBase =
                this->batchedCachedIndexBase + this->batchedAlignedIndex;
            this->batchedGroupHeadBase =
                this->batchedPackedNextBase + this->batchedAlignedIndex;
            this->batchedCacheFlagBase =
                this->batchedGroupHeadBase +
                this->batchedAlignedGroupCount;
            const uint32_t blockNum = GetBlockNum();
            this->batchedSharedLocalCountBase =
                this->batchedCacheFlagBase +
                blockNum * kBatchedPlanCacheFlagStride;
            this->batchedSharedCoreStartBase =
                this->batchedSharedLocalCountBase +
                blockNum * this->batchedAlignedDim;
            const uint32_t maxLocalIndex =
                (indexSize + blockNum - 1U) / blockNum;
            const uint32_t maxOccupiedDims =
                maxLocalIndex < dimSize ? maxLocalIndex : dimSize;
            this->batchedSharedLocalPackedCapacity =
                (maxOccupiedDims * 8U + 15U) & ~15U;
            this->batchedSharedPackedBase =
                this->batchedSharedCoreStartBase +
                blockNum * this->batchedAlignedDim;
            this->batchedSharedWaveBase =
                this->batchedSharedPackedBase +
                blockNum * this->batchedSharedLocalPackedCapacity;
            const uint32_t batchedWorkspaceWords =
                this->batchedSharedWaveBase +
                kBatchedSharedWaveCapacity * this->batchedAlignedDim;
            GM_ADDR batchedWorkspace = GetUserWorkspace(workspace);
            this->batchedPlanGm.SetGlobalBuffer(
                (__gm__ int32_t*)batchedWorkspace,
                batchedWorkspaceWords);
        }
        this->useParallelSharedPlan = UseParallelSharedPlan((T*)nullptr) &&
            innerSize == 1 && outerSize >= 512 && dimSize > 512 &&
            dimSize < 1024 && indexSize >= 4096 && indexSize <= 16384 &&
            totalSize >= 4096;
        if (this->useParallelSharedPlan) {
            this->sharedAlignedDim = (dimSize + 15U) & ~15U;
            this->sharedAlignedIndex =
                AlignUp32(indexSize * sizeof(int32_t)) / sizeof(int32_t);
            const uint32_t blockNum = GetBlockNum();
            this->sharedLocalCountBase = kSharedPlanHeaderWords;
            this->sharedGlobalCountBase = this->sharedLocalCountBase +
                blockNum * this->sharedAlignedDim;
            this->sharedCoreStartBase = this->sharedGlobalCountBase +
                this->sharedAlignedDim;
            this->sharedPackedBase = this->sharedCoreStartBase +
                blockNum * this->sharedAlignedDim;
            const uint32_t maxLocalIndex =
                (indexSize + blockNum - 1U) / blockNum;
            this->sharedCachedIndexStride =
                AlignUp32(maxLocalIndex * sizeof(int32_t)) /
                sizeof(int32_t);
            constexpr uint32_t cacheFlagStride = 8U;
            const uint32_t cacheScratchWords =
                this->sharedCachedIndexStride + kSharedPlanHeaderWords +
                blockNum * cacheFlagStride;
            this->sharedPlanCacheEnabled =
                this->sharedAlignedDim >= cacheScratchWords;
            this->sharedCachedIndexWords =
                this->sharedPlanCacheEnabled
                    ? blockNum *
                        (this->sharedCachedIndexStride + cacheFlagStride)
                    : 0U;
            const uint32_t maxOccupiedDims =
                maxLocalIndex < dimSize ? maxLocalIndex : dimSize;
            this->sharedLocalPackedCapacity =
                (maxOccupiedDims * 8U + 15U) & ~15U;
            this->sharedPackedCapacity =
                blockNum * this->sharedLocalPackedCapacity;
            this->sharedWaveBase = this->sharedPackedBase +
                this->sharedPackedCapacity;
            this->sharedWaveCapacity = this->sharedAlignedIndex +
                kSharedPlanWaveSlack * this->sharedAlignedDim;
            this->sharedCachedIndexBase = this->sharedWaveBase +
                this->sharedWaveCapacity - this->sharedCachedIndexWords;
            this->sharedCachedFlagBase = this->sharedCachedIndexBase +
                blockNum * this->sharedCachedIndexStride;
            const uint32_t workspaceWords =
                this->sharedWaveBase + this->sharedWaveCapacity;
            GM_ADDR sharedWorkspace = GetUserWorkspace(workspace);
            this->sharedPlanGm.SetGlobalBuffer(
                (__gm__ int32_t*)sharedWorkspace, workspaceWords);
        }
    }

    __aicore__ inline void Process()
    {
        if (this->totalSize == 0 || this->blockSize == 0 || this->outerSize == 0 ||
            this->dimSize == 0 || this->innerSize == 0) {
            return;
        }

        if (UseBatchedTargetGroup((T*)nullptr) &&
            this->CanUseBatchedTargetGroup()) {
            constexpr uint32_t groupBufferBytes = 61440;
            constexpr uint32_t groupSourceBufferBytes = 86784;
            constexpr uint32_t targetGroupElements = 8;
            this->pipe.InitBuffer(this->accumBuf, groupBufferBytes);
            this->pipe.InitBuffer(this->sourceBuf,
                groupSourceBufferBytes);
            const uint32_t indexBytes =
                AlignUp32(this->indexSize * sizeof(int32_t));
            const uint32_t groupHeadBytes = AlignUp32(
                (this->dimSize / targetGroupElements) *
                sizeof(int32_t));
            const uint32_t planMetaBytes = groupHeadBytes >
                    kSharedPlanHeaderWords * sizeof(int32_t)
                ? groupHeadBytes
                : kSharedPlanHeaderWords * sizeof(int32_t);
            const uint32_t defaultNextBytes =
                2U * indexBytes + planMetaBytes;
            this->pipe.InitBuffer(this->nextBuf,
                defaultNextBytes > kBatchedNextBufferBytes
                    ? defaultNextBytes : kBatchedNextBufferBytes);
            this->ProcessBatchedWaveRows();
            return;
        }

        if (UseDirectScalarAtomic((T*)nullptr) &&
            this->CanUseDirectScalarAtomic()) {
            this->pipe.InitBuffer(this->accumBuf,
                kGenericVectorAccumBytes);
            const uint32_t blockNum = GetBlockNum();
            const uint32_t maxIndexesPerBlock =
                (this->indexSize + blockNum - 1) / blockNum;
            this->pipe.InitBuffer(this->nextBuf,
                AlignUp32(maxIndexesPerBlock * sizeof(int32_t)));
            this->ProcessDirectScalarAtomic();
            return;
        }

        if (UseAtomicVector((T*)nullptr) && this->CanUseAtomicVector()) {
            this->pipe.InitBuffer(this->accumBuf, kGenericVectorAccumBytes);
            this->ProcessAtomicVector();
            return;
        }
        if (UseDirectPaddedAtomic((T*)nullptr) &&
            this->CanUseDirectPaddedAtomicVector()) {
            this->directPaddedAccumBytes = kGenericVectorAccumBytes;
            const uint32_t directBlockNum = GetBlockNum();
            const uint32_t directOutputRowElements =
                this->dimSize * this->innerSize;
            const bool useC4Outer2WideBuffer =
                UseInt32GroupedRepeat((T*)nullptr) &&
                directBlockNum >= 2 && (directBlockNum & 1U) == 0 &&
                directOutputRowElements % (32U / sizeof(T)) == 0 &&
                this->totalSize % (32U / sizeof(T)) == 0 &&
                this->outerSize >= 8 && this->outerSize < 32 &&
                this->dimSize >= 2 && this->dimSize <= 32 &&
                this->innerSize >= 4096 && this->indexSize >= 512 &&
                this->indexSize < 2048;
            if (useC4Outer2WideBuffer) {
                this->directPaddedAccumBytes = kC4Outer2AccumBytes;
            }
            this->pipe.InitBuffer(this->accumBuf,
                this->directPaddedAccumBytes);
            const uint32_t directIndexBytes =
                AlignUp32(this->indexSize * sizeof(int32_t));
            this->pipe.InitBuffer(this->nextBuf, directIndexBytes);
            this->ProcessDirectPaddedAtomicVector();
            return;
        }
        if (UseUnalignedDirectPaddedAtomic((T*)nullptr) &&
            this->CanUseUnalignedDirectPaddedAtomicVector()) {
            this->pipe.InitBuffer(this->accumBuf, kGenericVectorAccumBytes);
            const uint32_t blockNum = GetBlockNum();
            const uint32_t maxIndexesPerBlock =
                (this->indexSize + blockNum - 1) / blockNum;
            this->pipe.InitBuffer(this->nextBuf,
                AlignUp32(maxIndexesPerBlock * sizeof(int32_t)));
            this->ProcessUnalignedDirectPaddedAtomicVector();
            return;
        }
        if (UseAtomicVector((T*)nullptr) && this->CanUsePaddedAtomicVector()) {
            this->pipe.InitBuffer(this->accumBuf, kGenericVectorAccumBytes);
            const uint32_t blockNum = GetBlockNum();
            const uint32_t maxIndexesPerBlock =
                (this->indexSize + blockNum - 1) / blockNum;
            this->pipe.InitBuffer(this->nextBuf,
                AlignUp32(maxIndexesPerBlock * sizeof(int32_t)));
            this->ProcessPaddedAtomicVector();
            return;
        }
        const uint32_t vectorInner = this->CalcVectorInner();
        if (vectorInner > 0) {
            this->pipe.InitBuffer(this->accumBuf,
                this->dimSize * vectorInner * sizeof(T));
            const uint32_t groupedVectorRows = this->CalcGroupedVectorRows(vectorInner);
            if (groupedVectorRows > 0) {
                this->pipe.InitBuffer(this->sourceBuf,
                    2 * groupedVectorRows * vectorInner * sizeof(T));
                this->pipe.InitBuffer(this->nextBuf,
                    AlignUp32(this->indexSize * sizeof(int32_t)));
                this->PrepareGroupedVectorIndex();
                this->ProcessGroupedVector(vectorInner, groupedVectorRows);
                return;
            }
            this->pipe.InitBuffer(this->sourceBuf, vectorInner * sizeof(T));
            this->ProcessVector(vectorInner);
            return;
        }

        const uint32_t accumBytes = AlignUp32(this->dimSize * sizeof(T));
        if (accumBytes > kMaxAccumBufferBytes) {
            this->ProcessByOutput();
            return;
        }

        this->useGroupedContiguous = this->CanUseGroupedContiguous();
        const bool useGenericOuterBatch =
            this->useGroupedContiguous && this->CanUseGenericOuterBatch();
        const uint32_t genericAccumBytes = useGenericOuterBatch
            ? (this->useParallelSharedPlan
                ? kParallelSharedWaveValueElements
                : kGenericWaveValueElements) * sizeof(T)
            : accumBytes;
        this->pipe.InitBuffer(this->accumBuf, genericAccumBytes);
        if (this->useGroupedContiguous) {
            uint32_t genericHeadStride =
                AlignUp32(this->dimSize * sizeof(int32_t));
            if (this->useParallelSharedPlan) {
                genericHeadStride =
                    this->sharedAlignedDim * sizeof(int32_t);
            }
            this->genericHeadStrideBytes = genericHeadStride;
            const uint32_t genericHeadElements =
                genericHeadStride / sizeof(int32_t);
            uint32_t genericHeadRows = 1;
            if (useGenericOuterBatch && UseGenericBlockReduce((T*)nullptr) &&
                (this->indexSize <= 4096 || this->useParallelSharedPlan)) {
                const uint32_t reduceRepeat = genericHeadElements / 8;
                if (3 * reduceRepeat <= 255) {
                    genericHeadRows = 3;
                } else if (2 * reduceRepeat <= 255) {
                    genericHeadRows = 2;
                }
            }
            if (this->useParallelSharedPlan && genericHeadRows > 2U) {
                genericHeadRows = 2U;
            }
            this->genericAllocatedOuterGroup = genericHeadRows;
            uint32_t genericHeadBufferRows = genericHeadRows;
            if (this->useParallelSharedPlan && genericHeadBufferRows < 2U) {
                genericHeadBufferRows = 2U;
            }
            this->genericAllocatedHeadRows = genericHeadBufferRows;
            this->pipe.InitBuffer(this->headBuf,
                genericHeadBufferRows * genericHeadStride);
            uint32_t genericNextBytes =
                AlignUp32(this->indexSize * sizeof(int32_t));
            if (this->useParallelSharedPlan) {
                const uint32_t blockNum = GetBlockNum();
                const uint32_t localIndexWords =
                    (this->indexSize + blockNum - 1U) / blockNum;
                genericNextBytes = AlignUp32(
                    localIndexWords * sizeof(int32_t));
            }
            this->sharedNextBytes = genericNextBytes;
            this->pipe.InitBuffer(this->nextBuf, genericNextBytes);
            if (useGenericOuterBatch) {
                if (this->PrepareGenericWave()) {
                    if (this->sharedUsePackedFallback) {
                        this->pipe.InitBuffer(this->sourceBuf,
                            AlignUp32(this->indexSize * sizeof(T)));
                        this->ProcessSharedPackedFallback();
                        return;
                    }
                    this->pipe.InitBuffer(this->sourceBuf,
                        this->genericWaveOuterBatch *
                            this->genericWaveSourceStride * sizeof(T));
                    const uint32_t waveValueElements =
                        this->CanPipelineGenericWaveRows()
                            ? this->genericWaveGroupStride
                            : this->genericWaveGatherElements;
                    this->pipe.InitBuffer(this->waveValueBuf,
                        waveValueElements * sizeof(T));
                    this->ProcessGenericWave();
                    return;
                }
                const uint32_t blockNum = GetBlockNum();
                const uint32_t blockIdx = GetBlockIdx();
                const uint32_t baseDims = this->dimSize / blockNum;
                const uint32_t extraDims = this->dimSize - baseDims * blockNum;
                const uint32_t dimStart =
                    blockIdx * baseDims + (blockIdx < extraDims ? blockIdx : extraDims);
                const uint32_t dimEnd =
                    dimStart + baseDims + (blockIdx < extraDims ? 1U : 0U);
                BuildIndexChainsRangeCached(this->headBuf.template Get<int32_t>(),
                    this->nextBuf.template Get<int32_t>(), this->indexGm, this->indexSize,
                    this->dimSize, dimStart, dimEnd);
                this->pipe.InitBuffer(this->sourceBuf, kVectorTileElements * sizeof(T));
                this->ProcessGenericOuterBatch(dimStart, dimEnd);
                return;
            }
            BuildIndexChains(this->headBuf.template Get<int32_t>(), this->nextBuf.template Get<int32_t>(),
                this->indexGm, this->indexSize, this->dimSize);
            this->useGroupedSourceCache = this->CanCacheGroupedSource();
            if (this->useGroupedSourceCache) {
                this->pipe.InitBuffer(this->sourceBuf, AlignUp32(this->indexSize * sizeof(T)));
            }
        }

        const uint32_t lineSize = this->outerSize * this->innerSize;
        const uint32_t start = GetBlockIdx() * this->blockSize;
        uint32_t end = start + this->blockSize;
        if (end > lineSize) {
            end = lineSize;
        }
        if (start >= end) {
            return;
        }

        for (uint32_t line = start; line < end; ++line) {
            this->ComputeLine(line);
        }
    }

private:
    __aicore__ inline bool CanUseBatchedTargetGroup() const
    {
        constexpr uint32_t groupElements = 8;
        return this->innerSize == 1 && this->outerSize > 256 &&
            this->dimSize >= 64 && this->dimSize < 512 &&
            this->dimSize % groupElements == 0 &&
            this->indexSize >= 512 && this->indexSize < 2048 &&
            this->totalSize >= 4096;
    }

    __aicore__ inline void LoadBatchedPlanCache(
        const LocalTensor<int32_t>& packedNextLocal,
        const LocalTensor<int32_t>& groupHeadLocal)
    {
        DataCopy(packedNextLocal,
            this->batchedPlanGm[this->batchedPackedNextBase],
            this->batchedAlignedIndex);
        DataCopy(groupHeadLocal,
            this->batchedPlanGm[this->batchedGroupHeadBase],
            this->batchedAlignedGroupCount);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool TryLoadBatchedPlanCache(
        const LocalTensor<int32_t>& currentIndexLocal,
        const LocalTensor<int32_t>& packedNextLocal,
        const LocalTensor<int32_t>& groupHeadLocal,
        const LocalTensor<int32_t>& cacheFlagLocal)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        DataCopy(groupHeadLocal, this->batchedPlanGm[0],
            kSharedPlanHeaderWords);
        PipeBarrier<PIPE_ALL>();
        const bool headerMatches =
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheMagicAWord)) == kBatchedPlanCacheMagicA &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheMagicBWord)) == kBatchedPlanCacheMagicB &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheVersionWord)) == kBatchedPlanCacheVersion &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheBlockNumWord)) == blockNum &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheOuterSizeWord)) == this->outerSize &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheDimSizeWord)) == this->dimSize &&
            static_cast<uint32_t>(groupHeadLocal.GetValue(
                kCacheIndexSizeWord)) == this->indexSize;
        if (!headerMatches) {
            return false;
        }

        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t localIndexCount =
            baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        DataCopyExtParams localIndexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{
            false, 0, 0, 0};
        DataCopyPad(currentIndexLocal,
            this->indexGm[indexStart],
            localIndexParams, indexPadParams);
        DataCopyPad(packedNextLocal,
            this->batchedPlanGm[
                this->batchedCachedIndexBase + indexStart],
            localIndexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        uint32_t mismatch = 0;
        for (uint32_t localIndex = 0;
             localIndex < localIndexCount; ++localIndex) {
            if (currentIndexLocal.GetValue(localIndex) !=
                packedNextLocal.GetValue(localIndex)) {
                mismatch = 1U;
                break;
            }
        }
        Duplicate(groupHeadLocal, static_cast<int32_t>(mismatch),
            kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->batchedPlanGm[
                this->batchedCacheFlagBase +
                blockIdx * kBatchedPlanCacheFlagStride],
            groupHeadLocal, kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        DataCopy(cacheFlagLocal,
            this->batchedPlanGm[this->batchedCacheFlagBase],
            blockNum * kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t core = 0; core < blockNum; ++core) {
            if (cacheFlagLocal.GetValue(
                    core * kBatchedPlanCacheFlagStride) != 0) {
                return false;
            }
        }
        this->LoadBatchedPlanCache(
            packedNextLocal, groupHeadLocal);
        return true;
    }

    __aicore__ inline void BuildAndCommitBatchedPlanCache(
        const LocalTensor<int32_t>& currentIndexLocal,
        const LocalTensor<int32_t>& packedNextLocal,
        const LocalTensor<int32_t>& groupHeadLocal)
    {
        constexpr uint32_t groupElements = 8;
        const uint32_t groupCount = this->dimSize / groupElements;
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{
            false, 0, 0, 0};
        DataCopyPad(currentIndexLocal, this->indexGm,
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t group = 0; group < groupCount; ++group) {
            groupHeadLocal.SetValue(group, -1);
        }
        for (uint32_t sourceIndex = 0;
             sourceIndex < this->indexSize; ++sourceIndex) {
            int32_t target = currentIndexLocal.GetValue(sourceIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 ||
                target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }
            const uint32_t group =
                static_cast<uint32_t>(target) / groupElements;
            const uint32_t lane =
                static_cast<uint32_t>(target) &
                (groupElements - 1U);
            const int32_t previous =
                groupHeadLocal.GetValue(group);
            const uint32_t packed =
                (static_cast<uint32_t>(previous + 1) << 3) | lane;
            packedNextLocal.SetValue(sourceIndex,
                static_cast<int32_t>(packed));
            groupHeadLocal.SetValue(group,
                static_cast<int32_t>(sourceIndex));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(
            this->batchedPlanGm[this->batchedCachedIndexBase],
            currentIndexLocal, indexParams);
        DataCopy(this->batchedPlanGm[this->batchedPackedNextBase],
            packedNextLocal, this->batchedAlignedIndex);
        DataCopy(this->batchedPlanGm[this->batchedGroupHeadBase],
            groupHeadLocal, this->batchedAlignedGroupCount);
        PipeBarrier<PIPE_ALL>();

        this->batchedPlanGm.SetValue(kCacheVersionWord,
            static_cast<int32_t>(kBatchedPlanCacheVersion));
        this->batchedPlanGm.SetValue(kCacheBlockNumWord,
            static_cast<int32_t>(GetBlockNum()));
        this->batchedPlanGm.SetValue(kCacheOuterSizeWord,
            static_cast<int32_t>(this->outerSize));
        this->batchedPlanGm.SetValue(kCacheDimSizeWord,
            static_cast<int32_t>(this->dimSize));
        this->batchedPlanGm.SetValue(kCacheIndexSizeWord,
            static_cast<int32_t>(this->indexSize));
        this->batchedPlanGm.SetValue(kCacheMagicBWord,
            static_cast<int32_t>(kBatchedPlanCacheMagicB));
        this->batchedPlanGm.SetValue(kCacheMagicAWord,
            static_cast<int32_t>(kBatchedPlanCacheMagicA));
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
    }

    __aicore__ inline bool TryProcessBatchedCompactWaveRows()
    {
        constexpr uint32_t bufferBytes = 65536U;
        constexpr uint32_t maxCompactTargets = 64U;
        constexpr uint32_t alignElements = 32U / sizeof(T);
        const uint32_t bufferElements = bufferBytes / sizeof(T);
        const uint32_t alignedDim =
            AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        const uint32_t alignedIndex =
            AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        const uint32_t sourceStride = alignedIndex + alignElements;
        const uint32_t alignedIndexWords =
            AlignUp32(this->indexSize * sizeof(int32_t)) /
            sizeof(int32_t);
        const uint32_t groupHeadWords =
            AlignUp32((this->dimSize / 8U) * sizeof(int32_t)) /
            sizeof(int32_t);
        const uint32_t planMetaWords = groupHeadWords >
                kSharedPlanHeaderWords
            ? groupHeadWords : kSharedPlanHeaderWords;
        const uint32_t metadataCapacity = alignedIndexWords + planMetaWords;
        if (alignedDim == 0U || sourceStride >= bufferElements ||
            3U * alignedDim > metadataCapacity) {
            return false;
        }

        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        LocalTensor<int32_t> targetSlotLocal =
            indexLocal[alignedIndexWords];
        LocalTensor<int32_t> activeTargetLocal =
            targetSlotLocal[alignedDim];
        LocalTensor<int32_t> countSlotLocal =
            activeTargetLocal[alignedDim];
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(targetSlotLocal, static_cast<int32_t>(-1), alignedDim);
        PipeBarrier<PIPE_ALL>();

        uint32_t activeCount = 0U;
        uint32_t maxWaveCount = 0U;
        for (uint32_t sourceIndex = 0U;
             sourceIndex < this->indexSize; ++sourceIndex) {
            int32_t target = indexLocal.GetValue(sourceIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 ||
                target >= static_cast<int32_t>(this->dimSize)) {
                indexLocal.SetValue(sourceIndex, -1);
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            int32_t slot = targetSlotLocal.GetValue(targetDim);
            if (slot < 0) {
                if (activeCount >= maxCompactTargets) {
                    return false;
                }
                slot = static_cast<int32_t>(activeCount);
                targetSlotLocal.SetValue(targetDim, slot);
                activeTargetLocal.SetValue(activeCount, target);
                countSlotLocal.SetValue(activeCount, 0);
                ++activeCount;
            }
            const uint32_t slotIndex = static_cast<uint32_t>(slot);
            const uint32_t currentCount = static_cast<uint32_t>(
                countSlotLocal.GetValue(slotIndex));
            indexLocal.SetValue(sourceIndex, static_cast<int32_t>(
                (currentCount << 6U) | slotIndex));
            const uint32_t nextCount = currentCount + 1U;
            countSlotLocal.SetValue(slotIndex,
                static_cast<int32_t>(nextCount));
            if (nextCount > maxWaveCount) {
                maxWaveCount = nextCount;
            }
        }
        if (activeCount == 0U) {
            return false;
        }

        const uint32_t compactStride =
            AlignUp32(activeCount * sizeof(T)) / sizeof(T);
        const uint32_t matrixElements = maxWaveCount * compactStride;
        if (matrixElements > bufferElements - sourceStride ||
            matrixElements + alignedDim + compactStride > bufferElements) {
            return false;
        }

        LocalTensor<uint32_t> offsetLocal =
            this->accumBuf.template Get<uint32_t>();
        LocalTensor<T> fullValueLocal =
            this->accumBuf.template Get<T>()[matrixElements];
        LocalTensor<T> compactValueLocal = fullValueLocal[alignedDim];
        LocalTensor<float> fullValueFloatLocal =
            this->accumBuf.template Get<float>()[matrixElements];
        LocalTensor<float> compactValueFloatLocal =
            fullValueFloatLocal[alignedDim];
        LocalTensor<T> sourceLocal =
            this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal = sourceLocal[sourceStride];
        LocalTensor<float> sourceFloatLocal =
            this->sourceBuf.template Get<float>();
        LocalTensor<float> waveValueFloatLocal =
            sourceFloatLocal[sourceStride];
        LocalTensor<uint32_t> targetOffsetLocal =
            this->nextBuf.template Get<uint32_t>()[
                alignedIndexWords + 2U * alignedDim];

        const uint32_t zeroByteOffset = alignedIndex * sizeof(T);
        Duplicate(offsetLocal, zeroByteOffset, matrixElements);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t sourceIndex = 0U;
             sourceIndex < this->indexSize; ++sourceIndex) {
            const int32_t packedValue = indexLocal.GetValue(sourceIndex);
            if (packedValue < 0) {
                continue;
            }
            const uint32_t packed = static_cast<uint32_t>(packedValue);
            const uint32_t slot = packed & (maxCompactTargets - 1U);
            const uint32_t wave = packed >> 6U;
            offsetLocal.SetValue(wave * compactStride + slot,
                sourceIndex * sizeof(T));
        }
        for (uint32_t slot = 0U; slot < activeCount; ++slot) {
            targetOffsetLocal.SetValue(slot,
                static_cast<uint32_t>(activeTargetLocal.GetValue(slot)) *
                    sizeof(T));
        }
        for (uint32_t slot = activeCount; slot < compactStride; ++slot) {
            targetOffsetLocal.SetValue(slot, 0U);
        }
        Duplicate(sourceLocal[alignedIndex], static_cast<T>(0), alignElements);
        PipeBarrier<PIPE_ALL>();

        const uint32_t sourcePadding = alignedIndex - this->indexSize;
        const uint32_t valuePadding = alignedDim - this->dimSize;
        DataCopyExtParams sourceParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> sourcePadParams{
            sourcePadding != 0U, 0,
            static_cast<uint8_t>(sourcePadding), static_cast<T>(0)};
        DataCopyExtParams valueParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> valuePadParams{
            valuePadding != 0U, 0,
            static_cast<uint8_t>(valuePadding), static_cast<T>(0)};
        DataCopyExtParams outputParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        for (uint32_t outer = blockIdx;
             outer < this->outerSize; outer += blockNum) {
            DataCopyPad(fullValueLocal,
                this->selfGm[outer * this->dimSize],
                valueParams, valuePadParams);
            DataCopyPad(sourceLocal,
                this->sourceGm[outer * this->indexSize],
                sourceParams, sourcePadParams);
            PipeBarrier<PIPE_ALL>();
            Gather(compactValueFloatLocal, fullValueFloatLocal,
                targetOffsetLocal, 0, compactStride);
            Gather(waveValueFloatLocal, sourceFloatLocal,
                offsetLocal, 0, matrixElements);
            const uint32_t waveRowBlocks =
                compactStride * sizeof(T) / 32U;
            const BinaryRepeatParams repeatParams{1, 1, 1, 0,
                static_cast<uint8_t>(waveRowBlocks), 0};
            for (uint32_t waveBase = 0U; waveBase < maxWaveCount;
                 waveBase += 255U) {
                uint32_t repeatCount = maxWaveCount - waveBase;
                if (repeatCount > 255U) {
                    repeatCount = 255U;
                }
                Add(compactValueLocal,
                    waveValueLocal[waveBase * compactStride],
                    compactValueLocal, compactStride,
                    static_cast<uint8_t>(repeatCount), repeatParams);
                if (waveBase + repeatCount < maxWaveCount) {
                    PipeBarrier<PIPE_V>();
                }
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t slot = 0U; slot < activeCount; ++slot) {
                fullValueLocal.SetValue(
                    static_cast<uint32_t>(activeTargetLocal.GetValue(slot)),
                    compactValueLocal.GetValue(slot));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(this->outputGm[outer * this->dimSize],
                fullValueLocal, outputParams);
            PipeBarrier<PIPE_ALL>();
        }
        return true;
    }

    __aicore__ inline void ExpandFlatBucketRowsScalar(
        const LocalTensor<int32_t>& wavePlanLocal,
        uint32_t bucketWidth, uint32_t zeroByteOffset,
        uint32_t sinkByteOffset)
    {
        if (bucketWidth == 0U) {
            return;
        }
        for (uint32_t row = 1U; row < kBatchedBucketBatch; ++row) {
            Adds(wavePlanLocal[row * bucketWidth], wavePlanLocal,
                static_cast<int32_t>(
                    row * this->indexSize * sizeof(T)),
                static_cast<int32_t>(bucketWidth));
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t slot = 0U; slot < bucketWidth; ++slot) {
            if (static_cast<uint32_t>(wavePlanLocal.GetValue(slot)) !=
                zeroByteOffset) {
                continue;
            }
            for (uint32_t row = 0U;
                 row < kBatchedBucketBatch; ++row) {
                wavePlanLocal.SetValue(row * bucketWidth + slot,
                    static_cast<int32_t>(sinkByteOffset));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t PrepareParallelBatchedWavePlan(
        const LocalTensor<int32_t>& offsetPlanLocal,
        const LocalTensor<int32_t>& countLocal,
        const LocalTensor<int32_t>& indexLocal,
        uint32_t waveCapacity, uint32_t alignedDim,
        uint32_t zeroByteOffset, uint32_t sourceStride)
    {
        constexpr uint32_t invalidPlan = 0xFFFFFFFFU;
        constexpr uint32_t targetGroupElements = 8U;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t dimGroups = alignedDim / targetGroupElements;
        const uint32_t maxOwnedGroups =
            (dimGroups + blockNum - 1U) / blockNum;
        const uint32_t maxOwnedDimCount =
            maxOwnedGroups * targetGroupElements;
        const uint64_t availablePlanWords =
            static_cast<uint64_t>(waveCapacity) * alignedDim;
        const uint64_t maxLocalPlanWords =
            static_cast<uint64_t>(2U * blockNum +
                kBatchedSharedWaveCapacity) * maxOwnedDimCount;
        if (!this->useBatchedPlanCache || alignedDim == 0U ||
            alignedDim != this->batchedAlignedDim ||
            blockNum == 0U || blockNum > 255U ||
            blockNum * kBatchedPlanCacheFlagStride >
                this->batchedAlignedIndex ||
            this->batchedSharedLocalPackedCapacity + alignedDim >
                availablePlanWords ||
            maxLocalPlanWords > availablePlanWords) {
            return invalidPlan;
        }

        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t localIndexCount =
            baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        DataCopyExtParams localIndexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        LocalTensor<int32_t> cachedIndexLocal =
            indexLocal[this->batchedAlignedIndex];
        // The three-row fused bucket plan extends past the legacy metadata
        // slot at 2 * alignedIndex. Keep the header in the reserved tail.
        LocalTensor<int32_t> headerLocal =
            indexLocal[kBatchedNextBufferBytes / sizeof(int32_t) -
                kSharedPlanHeaderWords];
        DataCopy(headerLocal, this->batchedPlanGm[0],
            kSharedPlanHeaderWords);
        PipeBarrier<PIPE_ALL>();
        const uint32_t cachedMaxWaveCount = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheWaveCountWord));
        const uint32_t cachedUseCountBuckets = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheFallbackWord));
        const uint32_t cachedBucketWidth0 = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheWaveStrideWord));
        const uint32_t cachedBucketWidth1 = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheZeroElementWord));
        const uint32_t cachedBucketWidth2 = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheValueStrideWord));
        const uint32_t cachedBucketWidth3 = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheSelfElementWord));
        const uint32_t cachedBucketPlanWords = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheOuterGroupWord));
        const uint32_t expectedBucketPlanWords =
            6U * cachedBucketWidth0 +
            10U * cachedBucketWidth1 +
            16U * cachedBucketWidth2;
        const uint32_t planMetaWords =
            this->batchedAlignedGroupCount > kSharedPlanHeaderWords
            ? this->batchedAlignedGroupCount
            : kSharedPlanHeaderWords;
        const uint32_t nextCapacity = kBatchedNextBufferBytes /
            static_cast<uint32_t>(sizeof(int32_t));
        const bool cachedBucketMetadataValid =
            cachedUseCountBuckets == 0U ||
            (cachedUseCountBuckets == 1U &&
             cachedMaxWaveCount <= 16U &&
             cachedBucketWidth0 % 8U == 0U &&
             cachedBucketWidth1 % 8U == 0U &&
             cachedBucketWidth2 % 8U == 0U &&
             cachedBucketWidth3 % 8U == 0U &&
             cachedBucketWidth3 == 0U &&
             cachedBucketPlanWords == expectedBucketPlanWords &&
             kBatchedBucketBatch * cachedBucketPlanWords +
                kBatchedBucketBatch * alignedDim <= nextCapacity &&
             kBatchedBucketBatch * cachedBucketPlanWords +
                kBatchedBucketBatch * alignedDim <=
                    (kBatchedSharedWaveCapacity -
                        kBatchedSharedBucketWaveOffset) * alignedDim);
        const bool cacheHeaderMatches =
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheMagicAWord)) == kBatchedSharedCacheMagicA &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheMagicBWord)) == kBatchedSharedCacheMagicB &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheVersionWord)) == kBatchedSharedCacheVersion &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheBlockNumWord)) == blockNum &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheOuterSizeWord)) == this->outerSize &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheDimSizeWord)) == this->dimSize &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheIndexSizeWord)) == this->indexSize &&
            cachedMaxWaveCount <= waveCapacity &&
            cachedMaxWaveCount <= kBatchedSharedWaveCapacity &&
            cachedBucketMetadataValid;
        bool currentIndexLoaded = false;
        if (cacheHeaderMatches) {
            DataCopyPad(indexLocal, this->indexGm[indexStart],
                localIndexParams, indexPadParams);
            DataCopyPad(cachedIndexLocal,
                this->batchedPlanGm[
                    this->batchedCachedIndexBase + indexStart],
                localIndexParams, indexPadParams);
            PipeBarrier<PIPE_ALL>();
            currentIndexLoaded = true;
            uint32_t mismatch = 0U;
            for (uint32_t localIndex = 0U;
                 localIndex < localIndexCount; ++localIndex) {
                if (indexLocal.GetValue(localIndex) !=
                    cachedIndexLocal.GetValue(localIndex)) {
                    mismatch = 1U;
                    break;
                }
            }
            Duplicate(headerLocal, static_cast<int32_t>(0),
                kBatchedPlanCacheFlagStride);
            headerLocal.SetValue(0,
                static_cast<int32_t>(mismatch));
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->batchedPlanGm[
                    this->batchedCacheFlagBase +
                    blockIdx * kBatchedPlanCacheFlagStride],
                headerLocal, kBatchedPlanCacheFlagStride);
            PipeBarrier<PIPE_MTE3>();
            DataCacheCleanAndInvalid<int32_t,
                CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
            SyncAll<true>();

            DataCopy(cachedIndexLocal,
                this->batchedPlanGm[this->batchedCacheFlagBase],
                blockNum * kBatchedPlanCacheFlagStride);
            PipeBarrier<PIPE_ALL>();
            bool cacheMatches = true;
            for (uint32_t core = 0U; core < blockNum; ++core) {
                if (cachedIndexLocal.GetValue(
                        core * kBatchedPlanCacheFlagStride) != 0) {
                    cacheMatches = false;
                    break;
                }
            }
            if (cacheMatches) {
                this->batchedUseCountBucketPlan =
                    cachedUseCountBuckets == 1U;
                this->batchedBucketWidth0 = cachedBucketWidth0;
                this->batchedBucketWidth1 = cachedBucketWidth1;
                this->batchedBucketWidth2 = cachedBucketWidth2;
                this->batchedBucketWidth3 = cachedBucketWidth3;
                this->batchedBucketPlanWords = cachedBucketPlanWords;
                if (this->batchedUseCountBucketPlan) {
                    DataCopy(indexLocal,
                        this->batchedPlanGm[
                            this->batchedSharedWaveBase +
                            kBatchedSharedBucketWaveOffset * alignedDim],
                        kBatchedBucketBatch *
                            cachedBucketPlanWords +
                        kBatchedBucketBatch * alignedDim);
                    PipeBarrier<PIPE_ALL>();
                } else if (cachedMaxWaveCount > 0U) {
                    DataCopy(offsetPlanLocal,
                        this->batchedPlanGm[
                            this->batchedSharedWaveBase],
                        cachedMaxWaveCount * alignedDim);
                    PipeBarrier<PIPE_ALL>();
                }
                return cachedMaxWaveCount;
            }
        }

        // A cache miss starts a rebuild.  Invalidate the old header before
        // replacing its cached index so any capacity/fallback exit below
        // cannot expose a stale plan on the next invocation.
        if (blockIdx == 0U) {
            this->batchedPlanGm.SetValue(kCacheMagicAWord, 0);
            this->batchedPlanGm.SetValue(kCacheMagicBWord, 0);
            PipeBarrier<PIPE_ALL>();
            DataCacheCleanAndInvalid<int32_t,
                CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
        }

        if (!currentIndexLoaded) {
            DataCopyPad(indexLocal, this->indexGm[indexStart],
                localIndexParams, indexPadParams);
        }
        DataCopyPad(this->batchedPlanGm[
                this->batchedCachedIndexBase + indexStart],
            indexLocal, localIndexParams);
        Duplicate(countLocal, static_cast<int32_t>(0), alignedDim);
        PipeBarrier<PIPE_ALL>();

        bool localInvalid = false;
        for (uint32_t localIndex = 0U;
             localIndex < localIndexCount; ++localIndex) {
            int32_t target = indexLocal.GetValue(localIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 ||
                target >= static_cast<int32_t>(this->dimSize)) {
                indexLocal.SetValue(localIndex, -1);
                localInvalid = true;
                continue;
            }
            indexLocal.SetValue(localIndex, target);
            const uint32_t targetDim = static_cast<uint32_t>(target);
            countLocal.SetValue(targetDim,
                countLocal.GetValue(targetDim) + 1);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->batchedPlanGm[
                this->batchedSharedLocalCountBase +
                blockIdx * alignedDim],
            countLocal, alignedDim);

        LocalTensor<int32_t> packedLocal = offsetPlanLocal;
        LocalTensor<int32_t> cursorLocal =
            packedLocal[this->batchedSharedLocalPackedCapacity];
        uint32_t localPackedWords = 0U;
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t localCount = static_cast<uint32_t>(
                countLocal.GetValue(dim));
            countLocal.SetValue(dim,
                static_cast<int32_t>(localPackedWords));
            cursorLocal.SetValue(dim,
                static_cast<int32_t>(localPackedWords));
            localPackedWords += (localCount + 7U) & ~7U;
        }
        if (localPackedWords >
            this->batchedSharedLocalPackedCapacity) {
            localInvalid = true;
        }
        if (!localInvalid && localPackedWords > 0U) {
            Duplicate(packedLocal, static_cast<int32_t>(0),
                localPackedWords);
        }
        PipeBarrier<PIPE_ALL>();
        if (!localInvalid) {
            for (uint32_t localIndex = 0U;
                 localIndex < localIndexCount; ++localIndex) {
                const int32_t target = indexLocal.GetValue(localIndex);
                if (target < 0) {
                    continue;
                }
                const uint32_t targetDim =
                    static_cast<uint32_t>(target);
                const uint32_t packedOffset =
                    static_cast<uint32_t>(
                        cursorLocal.GetValue(targetDim));
                packedLocal.SetValue(packedOffset,
                    static_cast<int32_t>(indexStart + localIndex));
                cursorLocal.SetValue(targetDim,
                    static_cast<int32_t>(packedOffset + 1U));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->batchedPlanGm[
                this->batchedSharedCoreStartBase +
                blockIdx * alignedDim],
            countLocal, alignedDim);
        if (!localInvalid && localPackedWords > 0U) {
            DataCopy(this->batchedPlanGm[
                    this->batchedSharedPackedBase +
                    blockIdx *
                        this->batchedSharedLocalPackedCapacity],
                packedLocal, localPackedWords);
        }

        LocalTensor<int32_t> flagLocal =
            indexLocal[this->batchedAlignedIndex];
        Duplicate(flagLocal, static_cast<int32_t>(0),
            kBatchedPlanCacheFlagStride);
        flagLocal.SetValue(0, localInvalid ? 1 : 0);
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->batchedPlanGm[
                this->batchedCacheFlagBase +
                blockIdx * kBatchedPlanCacheFlagStride],
            flagLocal, kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
        SyncAll<true>();

        DataCopy(flagLocal,
            this->batchedPlanGm[this->batchedCacheFlagBase],
            blockNum * kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t core = 0U; core < blockNum; ++core) {
            if (flagLocal.GetValue(
                    core * kBatchedPlanCacheFlagStride) != 0) {
                return invalidPlan;
            }
        }

        const uint32_t baseGroups = dimGroups / blockNum;
        const uint32_t extraGroups =
            dimGroups - baseGroups * blockNum;
        const uint32_t groupStart =
            blockIdx * baseGroups +
            (blockIdx < extraGroups ? blockIdx : extraGroups);
        const uint32_t groupEnd =
            groupStart + baseGroups +
            (blockIdx < extraGroups ? 1U : 0U);
        const uint32_t dimStart = groupStart * targetGroupElements;
        const uint32_t ownedDimCount =
            (groupEnd - groupStart) * targetGroupElements;
        const uint32_t matrixStride = ownedDimCount;
        LocalTensor<int32_t> countMatrix = offsetPlanLocal;
        LocalTensor<int32_t> startMatrix =
            countMatrix[blockNum * matrixStride];
        LocalTensor<int32_t> ownedWaveLocal =
            startMatrix[blockNum * matrixStride];
        if (ownedDimCount > 0U) {
            DataCopyExtParams matrixParams{
                static_cast<uint16_t>(blockNum),
                ownedDimCount * static_cast<uint32_t>(sizeof(int32_t)),
                (alignedDim - ownedDimCount) *
                    static_cast<uint32_t>(sizeof(int32_t)),
                0, 0};
            DataCopyPadExtParams<int32_t> matrixPadParams{
                false, 0, 0, 0};
            DataCopyPad(countMatrix,
                this->batchedPlanGm[
                    this->batchedSharedLocalCountBase + dimStart],
                matrixParams, matrixPadParams);
            DataCopyPad(startMatrix,
                this->batchedPlanGm[
                    this->batchedSharedCoreStartBase + dimStart],
                matrixParams, matrixPadParams);
            Duplicate(ownedWaveLocal,
                static_cast<int32_t>(zeroByteOffset),
                kBatchedSharedWaveCapacity * ownedDimCount);
            PipeBarrier<PIPE_ALL>();
        }

        uint32_t localMaxWaveCount = 0U;
        for (uint32_t localDim = 0U;
             localDim < ownedDimCount; ++localDim) {
            uint32_t rank = 0U;
            for (uint32_t core = 0U; core < blockNum; ++core) {
                const uint32_t localCount = static_cast<uint32_t>(
                    countMatrix.GetValue(
                        core * matrixStride + localDim));
                const uint32_t packedStart = static_cast<uint32_t>(
                    startMatrix.GetValue(
                        core * matrixStride + localDim));
                for (uint32_t localRank = 0U;
                     localRank < localCount;
                     ++localRank, ++rank) {
                    if (rank >= kBatchedSharedWaveCapacity) {
                        continue;
                    }
                    const uint32_t sourceIndex =
                        static_cast<uint32_t>(
                            this->batchedPlanGm.GetValue(
                                this->batchedSharedPackedBase +
                                core *
                                    this->batchedSharedLocalPackedCapacity +
                                packedStart + localRank));
                    ownedWaveLocal.SetValue(
                        rank * ownedDimCount + localDim,
                        static_cast<int32_t>(
                            sourceIndex * sizeof(T)));
                }
            }
            if (rank > localMaxWaveCount) {
                localMaxWaveCount = rank;
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (ownedDimCount > 0U) {
            DataCopyExtParams planParams{
                static_cast<uint16_t>(kBatchedSharedWaveCapacity),
                ownedDimCount * static_cast<uint32_t>(sizeof(int32_t)),
                0,
                (alignedDim - ownedDimCount) *
                    static_cast<uint32_t>(sizeof(int32_t)),
                0};
            DataCopyPad(this->batchedPlanGm[
                    this->batchedSharedWaveBase + dimStart],
                ownedWaveLocal, planParams);
        }
        Duplicate(flagLocal, static_cast<int32_t>(0),
            kBatchedPlanCacheFlagStride);
        flagLocal.SetValue(1,
            static_cast<int32_t>(localMaxWaveCount));
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->batchedPlanGm[
                this->batchedCacheFlagBase +
                blockIdx * kBatchedPlanCacheFlagStride],
            flagLocal, kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
        SyncAll<true>();

        DataCopy(flagLocal,
            this->batchedPlanGm[this->batchedCacheFlagBase],
            blockNum * kBatchedPlanCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        uint32_t maxWaveCount = 0U;
        for (uint32_t core = 0U; core < blockNum; ++core) {
            const uint32_t coreMax = static_cast<uint32_t>(
                flagLocal.GetValue(
                    core * kBatchedPlanCacheFlagStride + 1U));
            if (coreMax > maxWaveCount) {
                maxWaveCount = coreMax;
            }
        }
        if (maxWaveCount > waveCapacity ||
            maxWaveCount > kBatchedSharedWaveCapacity) {
            return invalidPlan;
        }
        bool useCountBuckets = false;
        uint32_t bucketWidth0 = 0U;
        uint32_t bucketWidth1 = 0U;
        uint32_t bucketWidth2 = 0U;
        uint32_t bucketWidth3 = 0U;
        uint32_t bucketPlanWords = 0U;
        if (maxWaveCount > 0U && maxWaveCount <= 16U &&
            blockNum <= waveCapacity) {
            const uint32_t countMatrixWords = blockNum * alignedDim;
            DataCopy(offsetPlanLocal,
                this->batchedPlanGm[this->batchedSharedLocalCountBase],
                countMatrixWords);
            PipeBarrier<PIPE_ALL>();
            ReducePreparedInt32Rows(offsetPlanLocal,
                blockNum, alignedDim);
            PipeBarrier<PIPE_ALL>();
            Adds(countLocal, offsetPlanLocal,
                static_cast<int32_t>(0), alignedDim);
            if (maxWaveCount < 16U) {
                Duplicate(offsetPlanLocal[maxWaveCount * alignedDim],
                    static_cast<int32_t>(zeroByteOffset),
                    (16U - maxWaveCount) * alignedDim);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(offsetPlanLocal,
                this->batchedPlanGm[this->batchedSharedWaveBase],
                maxWaveCount * alignedDim);
            PipeBarrier<PIPE_ALL>();
            uint32_t bucketCount0 = 0U;
            uint32_t bucketCount1 = 0U;
            uint32_t bucketCount2 = 0U;
            for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
                const uint32_t count = static_cast<uint32_t>(
                    countLocal.GetValue(dim));
                if (count <= 6U) {
                    ++bucketCount0;
                } else if (count <= 10U) {
                    ++bucketCount1;
                } else {
                    ++bucketCount2;
                }
            }
            bucketWidth0 = (bucketCount0 + 7U) & ~7U;
            bucketWidth1 = (bucketCount1 + 7U) & ~7U;
            bucketWidth2 = (bucketCount2 + 7U) & ~7U;
            bucketPlanWords =
                6U * bucketWidth0 + 10U * bucketWidth1 +
                16U * bucketWidth2;
            const uint32_t expandedBucketPlanWords =
                kBatchedBucketBatch * bucketPlanWords;
            const uint32_t expandedBucketPlanStart1 =
                6U * kBatchedBucketBatch * bucketWidth0;
            const uint32_t expandedBucketPlanStart2 =
                expandedBucketPlanStart1 +
                10U * kBatchedBucketBatch * bucketWidth1;
            const uint32_t bucketWorkspaceCapacity =
                (kBatchedSharedWaveCapacity -
                    kBatchedSharedBucketWaveOffset) * alignedDim;
            if (bucketPlanWords > 0U &&
                expandedBucketPlanWords +
                    kBatchedBucketBatch * alignedDim +
                    bucketWidth0 + bucketWidth1 + bucketWidth2 <=
                        nextCapacity &&
                expandedBucketPlanWords +
                    kBatchedBucketBatch * alignedDim <=
                    bucketWorkspaceCapacity) {
                LocalTensor<int32_t> bucketPlanLocal = indexLocal;
                LocalTensor<int32_t> bucketMapLocal =
                    indexLocal[expandedBucketPlanWords];
                LocalTensor<int32_t> bucketSourceMap0Local =
                    bucketMapLocal[
                        kBatchedBucketBatch * alignedDim];
                LocalTensor<int32_t> bucketSourceMap1Local =
                    bucketSourceMap0Local[bucketWidth0];
                LocalTensor<int32_t> bucketSourceMap2Local =
                    bucketSourceMap1Local[bucketWidth1];
                Duplicate(bucketSourceMap0Local,
                    static_cast<int32_t>(0),
                    bucketWidth0 + bucketWidth1 + bucketWidth2);
                PipeBarrier<PIPE_ALL>();
                uint32_t cursor0 = 0U;
                uint32_t cursor1 = 0U;
                uint32_t cursor2 = 0U;
                for (uint32_t dim = 0U;
                     dim < this->dimSize; ++dim) {
                    const uint32_t count = static_cast<uint32_t>(
                        countLocal.GetValue(dim));
                    uint32_t localTarget = 0U;
                    uint32_t planStart = 0U;
                    uint32_t bucketWidth = bucketWidth0;
                    if (count <= 6U) {
                        localTarget = cursor0++;
                        bucketSourceMap0Local.SetValue(localTarget,
                            static_cast<int32_t>(
                                dim * sizeof(int32_t)));
                    } else if (count <= 10U) {
                        planStart = expandedBucketPlanStart1;
                        bucketWidth = bucketWidth1;
                        localTarget = cursor1++;
                        bucketSourceMap1Local.SetValue(localTarget,
                            static_cast<int32_t>(
                                dim * sizeof(int32_t)));
                    } else {
                        planStart = expandedBucketPlanStart2;
                        bucketWidth = bucketWidth2;
                        localTarget = cursor2++;
                        bucketSourceMap2Local.SetValue(localTarget,
                            static_cast<int32_t>(
                                dim * sizeof(int32_t)));
                    }
                    for (uint32_t row = 0U;
                         row < kBatchedBucketBatch; ++row) {
                        bucketMapLocal.SetValue(
                            row * alignedDim + dim,
                            static_cast<int32_t>((planStart +
                                row * bucketWidth + localTarget) *
                                sizeof(T)));
                    }
                }
                PipeBarrier<PIPE_ALL>();
                for (uint32_t wave = 0U; wave < 6U; ++wave) {
                    LocalTensor<int32_t> wavePlanLocal =
                        bucketPlanLocal[wave *
                            kBatchedBucketBatch * bucketWidth0];
                    Gather(wavePlanLocal,
                        offsetPlanLocal[wave * alignedDim],
                        bucketSourceMap0Local.
                            template ReinterpretCast<uint32_t>(),
                        0, bucketWidth0);
                    this->ExpandFlatBucketRowsScalar(wavePlanLocal,
                        bucketWidth0, zeroByteOffset,
                        AlignUp32(kBatchedBucketBatch *
                            this->indexSize * sizeof(T)));
                }
                for (uint32_t wave = 0U; wave < 10U; ++wave) {
                    LocalTensor<int32_t> wavePlanLocal =
                        bucketPlanLocal[expandedBucketPlanStart1 +
                            wave * kBatchedBucketBatch * bucketWidth1];
                    Gather(wavePlanLocal,
                        offsetPlanLocal[wave * alignedDim],
                        bucketSourceMap1Local.
                            template ReinterpretCast<uint32_t>(),
                        0, bucketWidth1);
                    this->ExpandFlatBucketRowsScalar(wavePlanLocal,
                        bucketWidth1, zeroByteOffset,
                        AlignUp32(kBatchedBucketBatch *
                            this->indexSize * sizeof(T)));
                }
                for (uint32_t wave = 0U; wave < 16U; ++wave) {
                    LocalTensor<int32_t> wavePlanLocal =
                        bucketPlanLocal[expandedBucketPlanStart2 +
                            wave * kBatchedBucketBatch * bucketWidth2];
                    Gather(wavePlanLocal,
                        offsetPlanLocal[wave * alignedDim],
                        bucketSourceMap2Local.
                            template ReinterpretCast<uint32_t>(),
                        0, bucketWidth2);
                    this->ExpandFlatBucketRowsScalar(wavePlanLocal,
                        bucketWidth2, zeroByteOffset,
                        AlignUp32(kBatchedBucketBatch *
                            this->indexSize * sizeof(T)));
                }
                PipeBarrier<PIPE_ALL>();
                if (blockIdx == 0U) {
                    DataCopy(this->batchedPlanGm[
                            this->batchedSharedWaveBase +
                            kBatchedSharedBucketWaveOffset * alignedDim],
                        bucketPlanLocal,
                        expandedBucketPlanWords +
                            kBatchedBucketBatch * alignedDim);
                }
                useCountBuckets = true;
            }
        } else if (maxWaveCount > 0U) {
            DataCopy(offsetPlanLocal,
                this->batchedPlanGm[this->batchedSharedWaveBase],
                maxWaveCount * alignedDim);
            PipeBarrier<PIPE_ALL>();
        }

        if (blockIdx == 0U) {
            if (useCountBuckets) {
                PipeBarrier<PIPE_MTE3>();
            }
            Duplicate(headerLocal, static_cast<int32_t>(0),
                kSharedPlanHeaderWords);
            headerLocal.SetValue(kCacheVersionWord,
                static_cast<int32_t>(kBatchedSharedCacheVersion));
            headerLocal.SetValue(kCacheBlockNumWord,
                static_cast<int32_t>(blockNum));
            headerLocal.SetValue(kCacheOuterSizeWord,
                static_cast<int32_t>(this->outerSize));
            headerLocal.SetValue(kCacheDimSizeWord,
                static_cast<int32_t>(this->dimSize));
            headerLocal.SetValue(kCacheIndexSizeWord,
                static_cast<int32_t>(this->indexSize));
            headerLocal.SetValue(kCacheFallbackWord,
                useCountBuckets ? 1 : 0);
            headerLocal.SetValue(kCacheWaveCountWord,
                static_cast<int32_t>(maxWaveCount));
            headerLocal.SetValue(kCacheWaveStrideWord,
                static_cast<int32_t>(bucketWidth0));
            headerLocal.SetValue(kCacheZeroElementWord,
                static_cast<int32_t>(bucketWidth1));
            headerLocal.SetValue(kCacheValueStrideWord,
                static_cast<int32_t>(bucketWidth2));
            headerLocal.SetValue(kCacheSelfElementWord,
                static_cast<int32_t>(bucketWidth3));
            headerLocal.SetValue(kCacheOuterGroupWord,
                static_cast<int32_t>(bucketPlanWords));
            headerLocal.SetValue(kCacheMagicBWord,
                static_cast<int32_t>(kBatchedSharedCacheMagicB));
            headerLocal.SetValue(kCacheMagicAWord,
                static_cast<int32_t>(kBatchedSharedCacheMagicA));
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->batchedPlanGm[0], headerLocal,
                kSharedPlanHeaderWords);
            PipeBarrier<PIPE_MTE3>();
            DataCacheCleanAndInvalid<int32_t,
                CacheLine::ENTIRE_DATA_CACHE>(this->batchedPlanGm);
        }
        this->batchedUseCountBucketPlan = useCountBuckets;
        this->batchedBucketWidth0 = bucketWidth0;
        this->batchedBucketWidth1 = bucketWidth1;
        this->batchedBucketWidth2 = bucketWidth2;
        this->batchedBucketWidth3 = bucketWidth3;
        this->batchedBucketPlanWords = bucketPlanWords;
        return maxWaveCount;
    }

    __aicore__ inline void LoadFlatSourceRows(
        const LocalTensor<T>& destination, uint32_t outerStart,
        uint32_t rowCount, uint32_t alignElements)
    {
        if (rowCount == 0U) {
            return;
        }
        const uint32_t copyElements = rowCount * this->indexSize;
        const uint32_t rightPadding =
            (alignElements - copyElements % alignElements) % alignElements;
        DataCopyExtParams copyParams{1,
            copyElements * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0U, 0,
            static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
        DataCopyPad(destination,
            this->sourceGm[outerStart * this->indexSize],
            copyParams, padParams);
    }

    __aicore__ inline void ProcessBatchedWaveRows()
    {
        constexpr uint32_t bufferBytes = 61440U;
        constexpr uint32_t sourceBufferBytes = 86784U;
        constexpr uint32_t alignElements = 32U / sizeof(T);
        const uint32_t bufferElements = bufferBytes / sizeof(T);
        const uint32_t sourceBufferElements =
            sourceBufferBytes / sizeof(T);
        const uint32_t alignedDim =
            AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        const uint32_t alignedIndex =
            AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        const bool sourceHasPadding = alignedIndex > this->indexSize;
        const uint32_t sourceStride = alignedIndex +
            (sourceHasPadding ? 0U : alignElements);
        if (alignedDim == 0U || sourceStride >= bufferElements ||
            alignedDim >= bufferElements) {
            this->ProcessBatchedTargetGroup();
            return;
        }

        uint32_t waveCapacity =
            (bufferElements - sourceStride) / alignedDim;
        const uint32_t accumWaveCapacity =
            (bufferElements - alignedDim) / alignedDim;
        if (waveCapacity > accumWaveCapacity) {
            waveCapacity = accumWaveCapacity;
        }
        if (waveCapacity == 0U) {
            this->ProcessBatchedTargetGroup();
            return;
        }

        LocalTensor<uint32_t> offsetLocal =
            this->accumBuf.template Get<uint32_t>();
        LocalTensor<int32_t> countLocal =
            this->accumBuf.template Get<int32_t>()[
                waveCapacity * alignedDim];
        LocalTensor<T> valueLocal =
            this->accumBuf.template Get<T>()[waveCapacity * alignedDim];
        LocalTensor<T> sourceLocal =
            this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal = sourceLocal[sourceStride];
        LocalTensor<float> sourceGatherLocal =
            this->sourceBuf.template Get<float>();
        LocalTensor<float> waveGatherLocal =
            sourceGatherLocal[sourceStride];
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();

        const uint32_t zeroElementOffset = sourceHasPadding
            ? this->indexSize : alignedIndex;
        const uint32_t zeroByteOffset = zeroElementOffset * sizeof(T);
        LocalTensor<int32_t> offsetPlanLocal =
            this->accumBuf.template Get<int32_t>();
        const uint32_t maxWaveCount =
            this->PrepareParallelBatchedWavePlan(offsetPlanLocal,
                countLocal, indexLocal, waveCapacity, alignedDim,
                zeroByteOffset, sourceStride);
        if (maxWaveCount == 0xFFFFFFFFU) {
            this->ProcessBatchedTargetGroup();
            return;
        }

        const bool useCountBucketPlan =
            this->batchedUseCountBucketPlan;
        const uint32_t bucketWidth0 = this->batchedBucketWidth0;
        const uint32_t bucketWidth1 = this->batchedBucketWidth1;
        const uint32_t bucketWidth2 = this->batchedBucketWidth2;
        const uint32_t bucketPlanStart0 = 0U;
        const uint32_t bucketPlanStart1 =
            6U * kBatchedBucketBatch * bucketWidth0;
        const uint32_t bucketPlanStart2 = bucketPlanStart1 +
            10U * kBatchedBucketBatch * bucketWidth1;
        const uint32_t bucketPlanWords =
            this->batchedBucketPlanWords;
        const uint32_t expandedBucketPlanWords =
            kBatchedBucketBatch * bucketPlanWords;
        const uint32_t bucketMapBase =
            expandedBucketPlanWords;

        if (!sourceHasPadding) {
            Duplicate(sourceLocal[alignedIndex], static_cast<T>(0),
                alignElements);
        }
        const uint32_t sourcePadding = alignedIndex - this->indexSize;
        const uint32_t valuePadding = alignedDim - this->dimSize;
        DataCopyExtParams sourceParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};
        DataCopyPadExtParams<T> sourcePadParams{
            sourcePadding != 0U, 0,
            static_cast<uint8_t>(sourcePadding), static_cast<T>(0)};
        DataCopyExtParams valueParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};
        DataCopyPadExtParams<T> valuePadParams{
            valuePadding != 0U, 0,
            static_cast<uint8_t>(valuePadding), static_cast<T>(0)};
        DataCopyExtParams outputParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter =
            this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter +
            (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd =
            outerStart + baseOuter +
            (blockIdx < extraOuter ? 1U : 0U);

        if (useCountBucketPlan) {
            constexpr uint32_t planBatch = kBatchedBucketBatch;
            constexpr uint32_t executionBatch =
                kBatchedExecutionBatch;
            constexpr uint32_t sourceBufferCount = 3U;
            constexpr uint32_t valueBufferCount = 2U;
            const uint32_t packedSourceElements =
                planBatch * this->indexSize;
            const uint32_t sinkElementOffset = AlignUp32(
                packedSourceElements * sizeof(T)) / sizeof(T);
            const uint32_t sourceGroupStride = AlignUp32(
                (sinkElementOffset + alignElements) * sizeof(T)) /
                    sizeof(T);
            const uint32_t waveAccumulatorBase =
                (valueBufferCount * executionBatch + planBatch) *
                    alignedDim;
            if (alignedDim == this->dimSize &&
                sourceBufferCount * sourceGroupStride <=
                    sourceBufferElements &&
                waveAccumulatorBase + expandedBucketPlanWords <=
                    bufferElements) {
                LocalTensor<int32_t> bucketPlanLocal = indexLocal;
                LocalTensor<uint32_t> bucketMapLocal =
                    indexLocal[bucketMapBase].
                        template ReinterpretCast<uint32_t>();
                LocalTensor<int32_t> batchedOffsetLocal =
                    bucketPlanLocal;
                LocalTensor<T> bucketValueLocal =
                    this->accumBuf.template Get<T>();
                LocalTensor<T> restoredValueLocal =
                    bucketValueLocal[
                        valueBufferCount * executionBatch * alignedDim];
                LocalTensor<T> batchedWaveValueLocal =
                    this->accumBuf.template Get<T>()[
                        waveAccumulatorBase];
                for (uint32_t buffer = 0U;
                     buffer < sourceBufferCount; ++buffer) {
                    Duplicate(sourceLocal[
                            buffer * sourceGroupStride +
                            sinkElementOffset],
                        static_cast<T>(0), alignElements);
                }
                PipeBarrier<PIPE_ALL>();

                const uint32_t ownedOuterCount = outerEnd - outerStart;
                const uint32_t groupCount =
                    (ownedOuterCount + planBatch - 1U) / planBatch;
                DataCopyPadExtParams<T> flatValuePadParams{
                    false, 0, 0, static_cast<T>(0)};
                uint32_t firstBatchRows = ownedOuterCount;
                if (firstBatchRows > executionBatch) {
                    firstBatchRows = executionBatch;
                }
                DataCopyExtParams firstValueParams{1,
                    firstBatchRows * this->dimSize *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0, 0};
                DataCopyPad(bucketValueLocal,
                    this->selfGm[outerStart * this->dimSize],
                    firstValueParams, flatValuePadParams);
                uint32_t firstGroupRows = ownedOuterCount;
                if (firstGroupRows > planBatch) {
                    firstGroupRows = planBatch;
                }
                this->LoadFlatSourceRows(sourceLocal, outerStart,
                    firstGroupRows, alignElements);
                SetFlag<HardEvent::MTE2_V>(0);
                if (groupCount > 1U) {
                    const uint32_t secondGroupOuter =
                        outerStart + planBatch;
                    uint32_t secondGroupRows =
                        outerEnd - secondGroupOuter;
                    if (secondGroupRows > planBatch) {
                        secondGroupRows = planBatch;
                    }
                    this->LoadFlatSourceRows(
                        sourceLocal[sourceGroupStride],
                        secondGroupOuter, secondGroupRows,
                        alignElements);
                    SetFlag<HardEvent::MTE2_V>(1);
                }

                bool outputPending0 = false;
                bool outputPending1 = false;
                for (uint32_t group = 0U;
                     group < groupCount; ++group) {
                    const uint32_t groupOuter =
                        outerStart + group * planBatch;
                    uint32_t groupRows = outerEnd - groupOuter;
                    if (groupRows > planBatch) {
                        groupRows = planBatch;
                    }
                    const uint32_t sourceBuffer =
                        group % sourceBufferCount;
                    WaitFlag<HardEvent::MTE2_V>(sourceBuffer);

                    const uint32_t futureGroup = group + 2U;
                    if (futureGroup < groupCount) {
                        const uint32_t futureOuter =
                            outerStart + futureGroup * planBatch;
                        uint32_t futureRows = outerEnd - futureOuter;
                        if (futureRows > planBatch) {
                            futureRows = planBatch;
                        }
                        const uint32_t futureSourceBuffer =
                            futureGroup % sourceBufferCount;
                        this->LoadFlatSourceRows(
                            sourceLocal[
                                futureSourceBuffer *
                                    sourceGroupStride],
                            futureOuter, futureRows, alignElements);

                        if ((futureGroup & 1U) == 0U) {
                            const uint32_t futureBatch =
                                futureGroup / 2U;
                            const uint32_t futureValueBuffer =
                                futureBatch % valueBufferCount;
                            if (futureValueBuffer == 0U &&
                                outputPending0) {
                                WaitFlag<HardEvent::MTE3_MTE2>(0);
                                outputPending0 = false;
                            } else if (futureValueBuffer == 1U &&
                                outputPending1) {
                                WaitFlag<HardEvent::MTE3_MTE2>(1);
                                outputPending1 = false;
                            }
                            uint32_t futureBatchRows =
                                outerEnd - futureOuter;
                            if (futureBatchRows > executionBatch) {
                                futureBatchRows = executionBatch;
                            }
                            LocalTensor<T> futureValueLocal =
                                bucketValueLocal[
                                    futureValueBuffer *
                                        executionBatch * alignedDim];
                            DataCopyExtParams futureValueParams{1,
                                futureBatchRows * this->dimSize *
                                    static_cast<uint32_t>(sizeof(T)),
                                0, 0, 0};
                            DataCopyPad(futureValueLocal,
                                this->selfGm[
                                    futureOuter * this->dimSize],
                                futureValueParams,
                                flatValuePadParams);
                        }
                        SetFlag<HardEvent::MTE2_V>(
                            futureSourceBuffer);
                    }

                    Gather(batchedWaveValueLocal,
                        sourceLocal[
                            sourceBuffer * sourceGroupStride],
                        batchedOffsetLocal.
                            template ReinterpretCast<uint32_t>(),
                        0, expandedBucketPlanWords);
                    LocalTensor<int32_t> waveIntLocal =
                        batchedWaveValueLocal.
                            template ReinterpretCast<int32_t>();
                    if (bucketWidth0 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart0],
                            6U, planBatch * bucketWidth0);
                    }
                    if (bucketWidth1 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart1],
                            10U, planBatch * bucketWidth1);
                    }
                    if (bucketWidth2 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart2],
                            16U, planBatch * bucketWidth2);
                    }

                    const uint32_t batch = group / 2U;
                    const uint32_t groupInBatch = group & 1U;
                    const uint32_t valueBuffer =
                        batch % valueBufferCount;
                    LocalTensor<T> currentValueLocal =
                        bucketValueLocal[
                            valueBuffer * executionBatch * alignedDim];
                    for (uint32_t row = 0U;
                         row < groupRows; ++row) {
                        LocalTensor<T> restoredLocal =
                            restoredValueLocal[row * alignedDim];
                        Gather(restoredLocal,
                            batchedWaveValueLocal,
                            bucketMapLocal[row * alignedDim],
                            0, this->dimSize);
                        Add(currentValueLocal[
                                (groupInBatch * planBatch + row) *
                                    alignedDim],
                            currentValueLocal[
                                (groupInBatch * planBatch + row) *
                                    alignedDim],
                            restoredLocal, this->dimSize);
                    }

                    if (groupInBatch == 1U ||
                        group + 1U == groupCount) {
                        const uint32_t batchOuter =
                            outerStart + batch * executionBatch;
                        uint32_t batchRows = outerEnd - batchOuter;
                        if (batchRows > executionBatch) {
                            batchRows = executionBatch;
                        }
                        SetFlag<HardEvent::V_MTE3>(valueBuffer);
                        WaitFlag<HardEvent::V_MTE3>(valueBuffer);
                        DataCopyExtParams outputBatchParams{1,
                            batchRows * this->dimSize *
                                static_cast<uint32_t>(sizeof(T)),
                            0, 0, 0};
                        DataCopyPad(
                            this->outputGm[
                                batchOuter * this->dimSize],
                            currentValueLocal, outputBatchParams);
                        SetFlag<HardEvent::MTE3_MTE2>(valueBuffer);
                        if (valueBuffer == 0U) {
                            outputPending0 = true;
                        } else {
                            outputPending1 = true;
                        }
                    }
                }
                if (outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                }
                if (outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                }
                return;
            }
        }

        if (useCountBucketPlan) {
            constexpr uint32_t planBatch = kBatchedBucketBatch;
            constexpr uint32_t executionBatch =
                kBatchedExecutionBatch;
            const uint32_t packedSourceElements =
                planBatch * this->indexSize;
            const uint32_t sinkElementOffset = AlignUp32(
                packedSourceElements * sizeof(T)) / sizeof(T);
            const uint32_t sourceGroupStride = AlignUp32(
                (sinkElementOffset + alignElements) * sizeof(T)) /
                    sizeof(T);
            const uint32_t waveAccumulatorBase =
                (2U * executionBatch + planBatch) * alignedDim;
            if (alignedDim == this->dimSize &&
                2U * sourceGroupStride <=
                    sourceBufferElements &&
                waveAccumulatorBase + expandedBucketPlanWords <=
                    bufferElements) {
                LocalTensor<int32_t> bucketPlanLocal = indexLocal;
                LocalTensor<uint32_t> bucketMapLocal =
                    indexLocal[bucketMapBase].
                        template ReinterpretCast<uint32_t>();
                LocalTensor<int32_t> batchedOffsetLocal =
                    bucketPlanLocal;
                LocalTensor<T> batchedWaveValueLocal =
                    this->accumBuf.template Get<T>()[
                        waveAccumulatorBase];
                LocalTensor<T> bucketValueLocal =
                    this->accumBuf.template Get<T>();
                LocalTensor<T> restoredValueLocal =
                    bucketValueLocal[
                        2U * executionBatch * alignedDim];
                Duplicate(sourceLocal[sinkElementOffset],
                    static_cast<T>(0), alignElements);
                Duplicate(sourceLocal[
                        sourceGroupStride + sinkElementOffset],
                    static_cast<T>(0), alignElements);
                PipeBarrier<PIPE_ALL>();

                uint32_t currentOuter = outerStart;
                uint32_t currentBuffer = 0U;
                uint32_t currentCount = outerEnd - currentOuter;
                if (currentCount > executionBatch) {
                    currentCount = executionBatch;
                }
                DataCopyPadExtParams<T> flatValuePadParams{
                    false, 0, 0, static_cast<T>(0)};
                DataCopyExtParams firstValueParams{1,
                    currentCount * this->dimSize *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0, 0};
                DataCopyPad(bucketValueLocal,
                    this->selfGm[currentOuter * this->dimSize],
                    firstValueParams, flatValuePadParams);
                uint32_t firstGroupCount = currentCount;
                if (firstGroupCount > planBatch) {
                    firstGroupCount = planBatch;
                }
                this->LoadFlatSourceRows(sourceLocal, currentOuter,
                    firstGroupCount, alignElements);
                const uint32_t secondGroupCount =
                    currentCount - firstGroupCount;
                if (secondGroupCount > 0U) {
                    this->LoadFlatSourceRows(
                        sourceLocal[sourceGroupStride],
                        currentOuter + firstGroupCount,
                        secondGroupCount, alignElements);
                }
                SetFlag<HardEvent::MTE2_V>(0);

                bool outputPending0 = false;
                bool outputPending1 = false;
                while (currentOuter < outerEnd) {
                    if (currentBuffer == 0U) {
                        WaitFlag<HardEvent::MTE2_V>(0);
                    } else {
                        WaitFlag<HardEvent::MTE2_V>(1);
                    }
                    Gather(batchedWaveValueLocal, sourceLocal,
                        batchedOffsetLocal.template ReinterpretCast<uint32_t>(),
                        0, expandedBucketPlanWords);

                    const uint32_t nextOuter =
                        currentOuter + currentCount;
                    uint32_t nextCount = outerEnd - nextOuter;
                    if (nextCount > executionBatch) {
                        nextCount = executionBatch;
                    }
                    const uint32_t nextBuffer = currentBuffer ^ 1U;
                    if (nextCount > 0U) {
                        if (currentBuffer == 0U) {
                            SetFlag<HardEvent::V_MTE2>(0);
                        } else {
                            SetFlag<HardEvent::V_MTE2>(1);
                        }
                        if (nextBuffer == 0U && outputPending0) {
                            WaitFlag<HardEvent::MTE3_MTE2>(0);
                            outputPending0 = false;
                        } else if (nextBuffer == 1U && outputPending1) {
                            WaitFlag<HardEvent::MTE3_MTE2>(1);
                            outputPending1 = false;
                        }
                        if (currentBuffer == 0U) {
                            WaitFlag<HardEvent::V_MTE2>(0);
                        } else {
                            WaitFlag<HardEvent::V_MTE2>(1);
                        }
                        LocalTensor<T> nextValueLocal = bucketValueLocal[
                            nextBuffer * executionBatch * alignedDim];
                        DataCopyExtParams nextValueParams{1,
                            nextCount * this->dimSize *
                                static_cast<uint32_t>(sizeof(T)),
                            0, 0, 0};
                        DataCopyPad(nextValueLocal,
                            this->selfGm[nextOuter * this->dimSize],
                            nextValueParams, flatValuePadParams);
                        uint32_t nextFirstGroupCount = nextCount;
                        if (nextFirstGroupCount > planBatch) {
                            nextFirstGroupCount = planBatch;
                        }
                        this->LoadFlatSourceRows(sourceLocal, nextOuter,
                            nextFirstGroupCount, alignElements);
                    }

                    LocalTensor<T> currentValueLocal = bucketValueLocal[
                        currentBuffer * executionBatch * alignedDim];
                    LocalTensor<int32_t> waveIntLocal =
                        batchedWaveValueLocal.
                            template ReinterpretCast<int32_t>();
                    if (bucketWidth0 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart0],
                            6U, planBatch * bucketWidth0);
                    }
                    if (bucketWidth1 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart1],
                            10U, planBatch * bucketWidth1);
                    }
                    if (bucketWidth2 > 0U) {
                        ReducePreparedInt32Rows(
                            waveIntLocal[bucketPlanStart2],
                            16U, planBatch * bucketWidth2);
                    }
                    uint32_t currentFirstGroupCount = currentCount;
                    if (currentFirstGroupCount > planBatch) {
                        currentFirstGroupCount = planBatch;
                    }
                    for (uint32_t row = 0U;
                         row < currentFirstGroupCount; ++row) {
                        LocalTensor<T> restoredLocal =
                            restoredValueLocal[row * alignedDim];
                        Gather(restoredLocal,
                            batchedWaveValueLocal,
                            bucketMapLocal[row * alignedDim],
                            0, this->dimSize);
                        Add(currentValueLocal[row * alignedDim],
                            currentValueLocal[row * alignedDim],
                            restoredLocal, this->dimSize);
                    }

                    const uint32_t currentSecondGroupCount =
                        currentCount - currentFirstGroupCount;
                    if (currentSecondGroupCount > 0U) {
                        Gather(batchedWaveValueLocal,
                            sourceLocal[sourceGroupStride],
                            batchedOffsetLocal.
                                template ReinterpretCast<uint32_t>(),
                            0, expandedBucketPlanWords);
                        if (nextCount > planBatch) {
                            if (currentBuffer == 0U) {
                                SetFlag<HardEvent::V_MTE2>(0);
                                WaitFlag<HardEvent::V_MTE2>(0);
                            } else {
                                SetFlag<HardEvent::V_MTE2>(1);
                                WaitFlag<HardEvent::V_MTE2>(1);
                            }
                            this->LoadFlatSourceRows(
                                sourceLocal[sourceGroupStride],
                                nextOuter + planBatch,
                                nextCount - planBatch, alignElements);
                        }
                        if (nextCount > 0U) {
                            if (nextBuffer == 0U) {
                                SetFlag<HardEvent::MTE2_V>(0);
                            } else {
                                SetFlag<HardEvent::MTE2_V>(1);
                            }
                        }
                        if (bucketWidth0 > 0U) {
                            ReducePreparedInt32Rows(
                                waveIntLocal[bucketPlanStart0],
                                6U, planBatch * bucketWidth0);
                        }
                        if (bucketWidth1 > 0U) {
                            ReducePreparedInt32Rows(
                                waveIntLocal[bucketPlanStart1],
                                10U, planBatch * bucketWidth1);
                        }
                        if (bucketWidth2 > 0U) {
                            ReducePreparedInt32Rows(
                                waveIntLocal[bucketPlanStart2],
                                16U, planBatch * bucketWidth2);
                        }
                        for (uint32_t row = 0U;
                             row < currentSecondGroupCount; ++row) {
                            LocalTensor<T> restoredLocal =
                                restoredValueLocal[row * alignedDim];
                            Gather(restoredLocal,
                                batchedWaveValueLocal,
                                bucketMapLocal[row * alignedDim],
                                0, this->dimSize);
                            Add(currentValueLocal[
                                    (planBatch + row) * alignedDim],
                                currentValueLocal[
                                    (planBatch + row) * alignedDim],
                                restoredLocal, this->dimSize);
                        }
                    } else if (nextCount > 0U) {
                        if (nextBuffer == 0U) {
                            SetFlag<HardEvent::MTE2_V>(0);
                        } else {
                            SetFlag<HardEvent::MTE2_V>(1);
                        }
                    }
                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);
                    } else {
                        SetFlag<HardEvent::V_MTE3>(1);
                        WaitFlag<HardEvent::V_MTE3>(1);
                    }
                    DataCopyExtParams outputBatchParams{1,
                        currentCount * this->dimSize *
                            static_cast<uint32_t>(sizeof(T)),
                        0, 0, 0};
                    DataCopyPad(this->outputGm[
                            currentOuter * this->dimSize],
                        currentValueLocal, outputBatchParams);
                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::MTE3_MTE2>(0);
                        outputPending0 = true;
                    } else {
                        SetFlag<HardEvent::MTE3_MTE2>(1);
                        outputPending1 = true;
                    }
                    currentOuter = nextOuter;
                    if (currentOuter < outerEnd) {
                        currentBuffer = nextBuffer;
                        currentCount = nextCount;
                    }
                }
                if (outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                }
                if (outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                }
                return;
            }
        }

        const uint32_t waveLimit = maxWaveCount == 0U ? 1U : maxWaveCount;
        for (uint32_t waveBase = 0U; waveBase < waveLimit;
             waveBase += waveCapacity) {
            if (waveBase != 0U) {
                Duplicate(offsetLocal, zeroByteOffset,
                    waveCapacity * alignedDim);
                Duplicate(countLocal, static_cast<int32_t>(0), alignedDim);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t sourceIndex = 0U;
                     sourceIndex < this->indexSize; ++sourceIndex) {
                    int32_t target = indexLocal.GetValue(sourceIndex);
                    if (target < 0) {
                        target += static_cast<int32_t>(this->dimSize);
                    }
                    if (target < 0 ||
                        target >= static_cast<int32_t>(this->dimSize)) {
                        continue;
                    }
                    const uint32_t targetDim = static_cast<uint32_t>(target);
                    const uint32_t wave = static_cast<uint32_t>(
                        countLocal.GetValue(targetDim));
                    if (wave >= waveBase &&
                        wave < waveBase + waveCapacity) {
                        offsetLocal.SetValue(
                            (wave - waveBase) * alignedDim + targetDim,
                            sourceIndex * sizeof(T));
                    }
                    countLocal.SetValue(targetDim,
                        static_cast<int32_t>(wave + 1U));
                }
                PipeBarrier<PIPE_ALL>();
            }

            uint32_t currentWaveCount = maxWaveCount - waveBase;
            if (maxWaveCount == 0U) {
                currentWaveCount = 0U;
            } else if (currentWaveCount > waveCapacity) {
                currentWaveCount = waveCapacity;
            }
            const uint32_t rowBufferStride = sourceStride +
                currentWaveCount * alignedDim;
            const bool canPipelineRows = currentWaveCount > 0U &&
                2U * rowBufferStride <= bufferElements &&
                waveCapacity * alignedDim + 2U * alignedDim <=
                    bufferElements;
            const uint32_t currentWaveElements =
                currentWaveCount * alignedDim;
            const bool canPipelineQuads = currentWaveCount > 0U &&
                4U * currentWaveCount <= waveCapacity &&
                4U * sourceStride + 4U * currentWaveElements <=
                    bufferElements &&
                waveCapacity * alignedDim + 8U * alignedDim <=
                    bufferElements &&
                outerEnd - outerStart >= 4U;
            const bool canPipelineTriples = currentWaveCount > 0U &&
                3U * currentWaveCount <= waveCapacity &&
                3U * sourceStride + 3U * currentWaveElements <=
                    bufferElements &&
                waveCapacity * alignedDim + 6U * alignedDim <=
                    bufferElements &&
                outerEnd - outerStart >= 3U;
            const uint32_t pipelineBatch = canPipelineQuads ? 4U :
                (canPipelineTriples ? 3U : 0U);
            if (pipelineBatch != 0U) {
                LocalTensor<int32_t> signedOffsetLocal =
                    offsetLocal.ReinterpretCast<int32_t>();
                for (uint32_t row = 1U; row < pipelineBatch; ++row) {
                    Adds(signedOffsetLocal[row * currentWaveElements],
                        signedOffsetLocal,
                        static_cast<int32_t>(
                            row * sourceStride * sizeof(T)),
                        static_cast<int32_t>(currentWaveElements));
                }

                LocalTensor<T> batchedWaveValueLocal =
                    sourceLocal[pipelineBatch * sourceStride];
                for (uint32_t row = 0U; row < pipelineBatch; ++row) {
                    Duplicate(sourceLocal[
                            row * sourceStride + alignedIndex],
                        static_cast<T>(0), alignElements);
                }
                PipeBarrier<PIPE_ALL>();

                uint32_t currentOuter = outerStart;
                uint32_t currentBuffer = 0U;
                uint32_t currentCount = outerEnd - currentOuter;
                if (currentCount > pipelineBatch) {
                    currentCount = pipelineBatch;
                }
                LocalTensor<T> firstValueLocal = valueLocal;
                for (uint32_t row = 0U; row < currentCount; ++row) {
                    if (waveBase == 0U) {
                        DataCopyPad(firstValueLocal[row * alignedDim],
                            this->selfGm[
                                (currentOuter + row) * this->dimSize],
                            valueParams, valuePadParams);
                    } else {
                        DataCopyPad(firstValueLocal[row * alignedDim],
                            this->outputGm[
                                (currentOuter + row) * this->dimSize],
                            valueParams, valuePadParams);
                    }
                    DataCopyPad(sourceLocal[row * sourceStride],
                        this->sourceGm[
                            (currentOuter + row) * this->indexSize],
                        sourceParams, sourcePadParams);
                }
                SetFlag<HardEvent::MTE2_V>(0);

                bool outputPending0 = false;
                bool outputPending1 = false;
                while (currentOuter < outerEnd) {
                    if (currentBuffer == 0U) {
                        WaitFlag<HardEvent::MTE2_V>(0);
                    } else {
                        WaitFlag<HardEvent::MTE2_V>(1);
                    }
                    Gather(batchedWaveValueLocal, sourceLocal,
                        offsetLocal, 0,
                        currentCount * currentWaveElements);

                    const uint32_t nextOuter =
                        currentOuter + currentCount;
                    uint32_t nextCount = outerEnd - nextOuter;
                    if (nextCount > pipelineBatch) {
                        nextCount = pipelineBatch;
                    }
                    const uint32_t nextBuffer = currentBuffer ^ 1U;
                    if (nextCount > 0U) {
                        if (currentBuffer == 0U) {
                            SetFlag<HardEvent::V_MTE2>(0);
                        } else {
                            SetFlag<HardEvent::V_MTE2>(1);
                        }
                        if (nextBuffer == 0U && outputPending0) {
                            WaitFlag<HardEvent::MTE3_MTE2>(0);
                            outputPending0 = false;
                        } else if (nextBuffer == 1U && outputPending1) {
                            WaitFlag<HardEvent::MTE3_MTE2>(1);
                            outputPending1 = false;
                        }
                        if (currentBuffer == 0U) {
                            WaitFlag<HardEvent::V_MTE2>(0);
                        } else {
                            WaitFlag<HardEvent::V_MTE2>(1);
                        }
                        LocalTensor<T> nextValueLocal = valueLocal[
                            nextBuffer * pipelineBatch * alignedDim];
                        for (uint32_t row = 0U; row < nextCount; ++row) {
                            if (waveBase == 0U) {
                                DataCopyPad(nextValueLocal[
                                        row * alignedDim],
                                    this->selfGm[
                                        (nextOuter + row) * this->dimSize],
                                    valueParams, valuePadParams);
                            } else {
                                DataCopyPad(nextValueLocal[
                                        row * alignedDim],
                                    this->outputGm[
                                        (nextOuter + row) * this->dimSize],
                                    valueParams, valuePadParams);
                            }
                            DataCopyPad(sourceLocal[row * sourceStride],
                                this->sourceGm[
                                    (nextOuter + row) * this->indexSize],
                                sourceParams, sourcePadParams);
                        }
                        if (nextBuffer == 0U) {
                            SetFlag<HardEvent::MTE2_V>(0);
                        } else {
                            SetFlag<HardEvent::MTE2_V>(1);
                        }
                    }

                    LocalTensor<T> currentValueLocal = valueLocal[
                        currentBuffer * pipelineBatch * alignedDim];
                    for (uint32_t row = 0U; row < currentCount; ++row) {
                        AccumulatePreparedGenericWave(
                            currentValueLocal[row * alignedDim],
                            batchedWaveValueLocal[
                                row * currentWaveElements],
                            this->dimSize, currentWaveCount, alignedDim);
                    }

                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);
                    } else {
                        SetFlag<HardEvent::V_MTE3>(1);
                        WaitFlag<HardEvent::V_MTE3>(1);
                    }
                    for (uint32_t row = 0U; row < currentCount; ++row) {
                        DataCopyPad(this->outputGm[
                                (currentOuter + row) * this->dimSize],
                            currentValueLocal[row * alignedDim],
                            outputParams);
                    }
                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::MTE3_MTE2>(0);
                        outputPending0 = true;
                    } else {
                        SetFlag<HardEvent::MTE3_MTE2>(1);
                        outputPending1 = true;
                    }

                    currentOuter = nextOuter;
                    if (currentOuter < outerEnd) {
                        currentBuffer = nextBuffer;
                        currentCount = nextCount;
                    }
                }
                if (outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                }
                if (outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                }
                continue;
            }
            const bool canPipelinePairs = currentWaveCount > 0U &&
                2U * currentWaveCount <= waveCapacity &&
                4U * sourceStride + 2U * currentWaveElements <=
                    bufferElements &&
                waveCapacity * alignedDim + 4U * alignedDim <=
                    bufferElements &&
                outerEnd - outerStart >= 2U;
            if (canPipelinePairs) {
                LocalTensor<int32_t> signedOffsetLocal =
                    offsetLocal.ReinterpretCast<int32_t>();
                Adds(signedOffsetLocal[currentWaveElements],
                    signedOffsetLocal,
                    static_cast<int32_t>(sourceStride * sizeof(T)),
                    static_cast<int32_t>(currentWaveElements));

                LocalTensor<T> pairedWaveValueLocal =
                    sourceLocal[4U * sourceStride];
                for (uint32_t row = 0U; row < 4U; ++row) {
                    Duplicate(sourceLocal[
                            row * sourceStride + alignedIndex],
                        static_cast<T>(0), alignElements);
                }
                PipeBarrier<PIPE_ALL>();

                uint32_t currentOuter = outerStart;
                uint32_t currentBuffer = 0U;
                uint32_t currentCount = outerEnd - currentOuter;
                if (currentCount > 2U) {
                    currentCount = 2U;
                }
                LocalTensor<T> firstValueLocal = valueLocal;
                LocalTensor<T> firstSourceLocal = sourceLocal;
                for (uint32_t row = 0U; row < currentCount; ++row) {
                    if (waveBase == 0U) {
                        DataCopyPad(firstValueLocal[row * alignedDim],
                            this->selfGm[
                                (currentOuter + row) * this->dimSize],
                            valueParams, valuePadParams);
                    } else {
                        DataCopyPad(firstValueLocal[row * alignedDim],
                            this->outputGm[
                                (currentOuter + row) * this->dimSize],
                            valueParams, valuePadParams);
                    }
                    DataCopyPad(firstSourceLocal[row * sourceStride],
                        this->sourceGm[
                            (currentOuter + row) * this->indexSize],
                        sourceParams, sourcePadParams);
                }
                SetFlag<HardEvent::MTE2_V>(0);

                bool outputPending0 = false;
                bool outputPending1 = false;
                while (currentOuter < outerEnd) {
                    const uint32_t nextOuter =
                        currentOuter + currentCount;
                    uint32_t nextCount = outerEnd - nextOuter;
                    if (nextCount > 2U) {
                        nextCount = 2U;
                    }
                    const uint32_t nextBuffer = currentBuffer ^ 1U;
                    if (nextCount > 0U) {
                        if (nextBuffer == 0U && outputPending0) {
                            WaitFlag<HardEvent::MTE3_MTE2>(0);
                            outputPending0 = false;
                        } else if (nextBuffer == 1U && outputPending1) {
                            WaitFlag<HardEvent::MTE3_MTE2>(1);
                            outputPending1 = false;
                        }
                        LocalTensor<T> nextValueLocal = valueLocal[
                            nextBuffer * 2U * alignedDim];
                        LocalTensor<T> nextSourceLocal = sourceLocal[
                            nextBuffer * 2U * sourceStride];
                        for (uint32_t row = 0U; row < nextCount; ++row) {
                            if (waveBase == 0U) {
                                DataCopyPad(nextValueLocal[
                                        row * alignedDim],
                                    this->selfGm[
                                        (nextOuter + row) * this->dimSize],
                                    valueParams, valuePadParams);
                            } else {
                                DataCopyPad(nextValueLocal[
                                        row * alignedDim],
                                    this->outputGm[
                                        (nextOuter + row) * this->dimSize],
                                    valueParams, valuePadParams);
                            }
                            DataCopyPad(nextSourceLocal[
                                    row * sourceStride],
                                this->sourceGm[
                                    (nextOuter + row) * this->indexSize],
                                sourceParams, sourcePadParams);
                        }
                        if (nextBuffer == 0U) {
                            SetFlag<HardEvent::MTE2_V>(0);
                        } else {
                            SetFlag<HardEvent::MTE2_V>(1);
                        }
                    }

                    if (currentBuffer == 0U) {
                        WaitFlag<HardEvent::MTE2_V>(0);
                    } else {
                        WaitFlag<HardEvent::MTE2_V>(1);
                    }
                    LocalTensor<T> currentValueLocal = valueLocal[
                        currentBuffer * 2U * alignedDim];
                    LocalTensor<T> currentSourceLocal = sourceLocal[
                        currentBuffer * 2U * sourceStride];
                    Gather(pairedWaveValueLocal, currentSourceLocal,
                        offsetLocal, 0,
                        currentCount * currentWaveElements);
                    AccumulatePreparedGenericWave(currentValueLocal,
                        pairedWaveValueLocal, this->dimSize,
                        currentWaveCount, alignedDim);
                    if (currentCount == 2U) {
                        AccumulatePreparedGenericWave(
                            currentValueLocal[alignedDim],
                            pairedWaveValueLocal[currentWaveElements],
                            this->dimSize, currentWaveCount, alignedDim);
                    }

                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);
                    } else {
                        SetFlag<HardEvent::V_MTE3>(1);
                        WaitFlag<HardEvent::V_MTE3>(1);
                    }
                    for (uint32_t row = 0U; row < currentCount; ++row) {
                        DataCopyPad(this->outputGm[
                                (currentOuter + row) * this->dimSize],
                            currentValueLocal[row * alignedDim],
                            outputParams);
                    }
                    if (currentBuffer == 0U) {
                        SetFlag<HardEvent::MTE3_MTE2>(0);
                        outputPending0 = true;
                    } else {
                        SetFlag<HardEvent::MTE3_MTE2>(1);
                        outputPending1 = true;
                    }

                    currentOuter = nextOuter;
                    if (currentOuter < outerEnd) {
                        currentBuffer = nextBuffer;
                        currentCount = nextCount;
                    }
                }
                if (outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                }
                if (outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                }
                continue;
            }
            if (canPipelineRows) {
                LocalTensor<T> secondSourceLocal =
                    sourceLocal[rowBufferStride];
                Duplicate(secondSourceLocal[alignedIndex],
                    static_cast<T>(0), alignElements);
                PipeBarrier<PIPE_ALL>();

                uint32_t currentOuter = outerStart;
                if (currentOuter < outerEnd) {
                    uint32_t currentBuffer = 0U;
                    if (waveBase == 0U) {
                        DataCopyPad(valueLocal,
                            this->selfGm[currentOuter * this->dimSize],
                            valueParams, valuePadParams);
                    } else {
                        DataCopyPad(valueLocal,
                            this->outputGm[currentOuter * this->dimSize],
                            valueParams, valuePadParams);
                    }
                    DataCopyPad(sourceLocal,
                        this->sourceGm[currentOuter * this->indexSize],
                        sourceParams, sourcePadParams);
                    SetFlag<HardEvent::MTE2_V>(0);

                    bool outputPending0 = false;
                    bool outputPending1 = false;
                    while (currentOuter < outerEnd) {
                        const uint32_t nextOuter =
                            currentOuter + 1U;
                        const uint32_t nextBuffer = currentBuffer ^ 1U;
                        if (nextOuter < outerEnd) {
                            if (nextBuffer == 0U && outputPending0) {
                                WaitFlag<HardEvent::MTE3_MTE2>(0);
                                outputPending0 = false;
                            } else if (nextBuffer == 1U && outputPending1) {
                                WaitFlag<HardEvent::MTE3_MTE2>(1);
                                outputPending1 = false;
                            }
                            LocalTensor<T> nextValueLocal =
                                valueLocal[nextBuffer * alignedDim];
                            LocalTensor<T> nextSourceLocal =
                                sourceLocal[nextBuffer * rowBufferStride];
                            if (waveBase == 0U) {
                                DataCopyPad(nextValueLocal,
                                    this->selfGm[
                                        nextOuter * this->dimSize],
                                    valueParams, valuePadParams);
                            } else {
                                DataCopyPad(nextValueLocal,
                                    this->outputGm[
                                        nextOuter * this->dimSize],
                                    valueParams, valuePadParams);
                            }
                            DataCopyPad(nextSourceLocal,
                                this->sourceGm[
                                    nextOuter * this->indexSize],
                                sourceParams, sourcePadParams);
                            if (nextBuffer == 0U) {
                                SetFlag<HardEvent::MTE2_V>(0);
                            } else {
                                SetFlag<HardEvent::MTE2_V>(1);
                            }
                        }

                        if (currentBuffer == 0U) {
                            WaitFlag<HardEvent::MTE2_V>(0);
                        } else {
                            WaitFlag<HardEvent::MTE2_V>(1);
                        }
                        LocalTensor<T> currentValueLocal =
                            valueLocal[currentBuffer * alignedDim];
                        LocalTensor<T> currentSourceLocal =
                            sourceLocal[
                                currentBuffer * rowBufferStride];
                        LocalTensor<T> currentWaveValueLocal =
                            currentSourceLocal[sourceStride];
                        AccumulateGenericWave(currentValueLocal,
                            currentSourceLocal, currentWaveValueLocal,
                            offsetLocal, currentValueLocal,
                            this->dimSize, currentWaveCount, alignedDim,
                            0U, currentWaveCount,
                            currentWaveCount * alignedDim,
                            false, (T*)nullptr);

                        if (currentBuffer == 0U) {
                            SetFlag<HardEvent::V_MTE3>(0);
                            WaitFlag<HardEvent::V_MTE3>(0);
                        } else {
                            SetFlag<HardEvent::V_MTE3>(1);
                            WaitFlag<HardEvent::V_MTE3>(1);
                        }
                        DataCopyPad(
                            this->outputGm[
                                currentOuter * this->dimSize],
                            currentValueLocal, outputParams);
                        if (currentBuffer == 0U) {
                            SetFlag<HardEvent::MTE3_MTE2>(0);
                            outputPending0 = true;
                        } else {
                            SetFlag<HardEvent::MTE3_MTE2>(1);
                            outputPending1 = true;
                        }
                        currentOuter = nextOuter;
                        if (currentOuter < outerEnd) {
                            currentBuffer = nextBuffer;
                        }
                    }
                    if (outputPending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(0);
                    }
                    if (outputPending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(1);
                    }
                }
                continue;
            }
            for (uint32_t outer = outerStart;
                 outer < outerEnd; ++outer) {
                if (waveBase == 0U) {
                    DataCopyPad(valueLocal,
                        this->selfGm[outer * this->dimSize],
                        valueParams, valuePadParams);
                } else {
                    DataCopyPad(valueLocal,
                        this->outputGm[outer * this->dimSize],
                        valueParams, valuePadParams);
                }
                DataCopyPad(sourceLocal,
                    this->sourceGm[outer * this->indexSize],
                    sourceParams, sourcePadParams);
                PipeBarrier<PIPE_ALL>();
                if (currentWaveCount > 0U) {
                    Gather(waveGatherLocal, sourceGatherLocal, offsetLocal, 0,
                        currentWaveCount * alignedDim);
                    const uint32_t waveRowBytes =
                        alignedDim * static_cast<uint32_t>(sizeof(T));
                    const uint32_t waveRowBlocks = waveRowBytes / 32U;
                    constexpr uint32_t elementsPerRepeat =
                        256U / sizeof(T);
                    if (currentWaveCount <= 255U &&
                        waveRowBytes % 32U == 0U &&
                        waveRowBlocks <= 255U) {
                        const BinaryRepeatParams repeatParams{1, 1, 1, 0,
                            static_cast<uint8_t>(waveRowBlocks), 0};
                        for (uint32_t dimStart = 0U;
                             dimStart < this->dimSize;
                             dimStart += elementsPerRepeat) {
                            uint32_t count = this->dimSize - dimStart;
                            if (count > elementsPerRepeat) {
                                count = elementsPerRepeat;
                            }
                            Add(valueLocal[dimStart],
                                waveValueLocal[dimStart],
                                valueLocal[dimStart],
                                static_cast<uint64_t>(count),
                                static_cast<uint8_t>(currentWaveCount),
                                repeatParams);
                        }
                    } else {
                        for (uint32_t wave = 0U;
                             wave < currentWaveCount; ++wave) {
                            Add(valueLocal, valueLocal,
                                waveValueLocal[wave * alignedDim],
                                this->dimSize);
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                }
                DataCopyPad(
                    this->outputGm[outer * this->dimSize],
                    valueLocal, outputParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessBatchedTargetGroup()
    {
        constexpr uint32_t groupElements = 8;
        constexpr uint32_t groupBufferBytes = 61440;
        constexpr uint32_t outerPartitions = 5;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t groupCount = this->dimSize / groupElements;
        const uint32_t taskCount = groupCount * outerPartitions;
        if (blockIdx >= taskCount) {
            return;
        }

        LocalTensor<T> valueLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal =
            this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        const uint32_t alignedIndexWords =
            AlignUp32(this->indexSize * sizeof(int32_t)) /
            sizeof(int32_t);
        LocalTensor<int32_t> matchedLocal =
            indexLocal[alignedIndexWords];
        LocalTensor<int32_t> groupHeadLocal =
            matchedLocal[alignedIndexWords];
        LocalTensor<int32_t> cacheFlagLocal =
            this->accumBuf.template Get<int32_t>();
        if (!this->TryLoadBatchedPlanCache(indexLocal,
                matchedLocal, groupHeadLocal, cacheFlagLocal)) {
            if (blockIdx == 0U) {
                this->BuildAndCommitBatchedPlanCache(indexLocal,
                    matchedLocal, groupHeadLocal);
            }
            SyncAll<true>();
            this->LoadBatchedPlanCache(
                matchedLocal, groupHeadLocal);
        }

        const uint32_t maxOuterBatch =
            groupBufferBytes /
            (groupElements * static_cast<uint32_t>(sizeof(T)));
        const uint32_t baseOuterCount =
            this->outerSize / outerPartitions;
        const uint32_t extraOuterCount =
            this->outerSize - baseOuterCount * outerPartitions;
        for (uint32_t task = blockIdx; task < taskCount;
             task += blockNum) {
            const uint32_t group = task / outerPartitions;
            const uint32_t outerPartition =
                task - group * outerPartitions;
            const uint32_t targetStart = group * groupElements;
            const uint32_t outerPartitionStart =
                outerPartition * baseOuterCount +
                (outerPartition < extraOuterCount
                    ? outerPartition : extraOuterCount);
            const uint32_t outerPartitionCount =
                baseOuterCount +
                (outerPartition < extraOuterCount ? 1U : 0U);
            const uint32_t outerPartitionEnd =
                outerPartitionStart + outerPartitionCount;

            for (uint32_t outerStart = outerPartitionStart;
                 outerStart < outerPartitionEnd;
                 outerStart += maxOuterBatch) {
                uint32_t outerCount = outerPartitionEnd - outerStart;
                if (outerCount > maxOuterBatch) {
                    outerCount = maxOuterBatch;
                }
                DataCopyExtParams valueParams{
                    static_cast<uint16_t>(outerCount),
                    groupElements * static_cast<uint32_t>(sizeof(T)),
                    (this->dimSize - groupElements) *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0};
                DataCopyPadExtParams<T> valuePadParams{
                    false, 0, 0, static_cast<T>(0)};
                DataCopyPad(valueLocal,
                    this->selfGm[
                        outerStart * this->dimSize + targetStart],
                    valueParams, valuePadParams);
                PipeBarrier<PIPE_ALL>();

                int32_t matched = groupHeadLocal.GetValue(group);
                while (matched >= 0) {
                    const uint32_t sourceIndex =
                        static_cast<uint32_t>(matched);
                    const uint32_t packed = static_cast<uint32_t>(
                        matchedLocal.GetValue(sourceIndex));
                    const uint32_t lane =
                        packed & (groupElements - 1U);
                    DataCopyExtParams sourceParams{
                        static_cast<uint16_t>(outerCount),
                        static_cast<uint32_t>(sizeof(T)),
                        (this->indexSize - 1U) *
                            static_cast<uint32_t>(sizeof(T)),
                        0, 0};
                    DataCopyPadExtParams<T> sourcePadParams{true,
                        static_cast<uint8_t>(lane),
                        static_cast<uint8_t>(groupElements - 1U - lane),
                        static_cast<T>(0)};
                    DataCopyPad(sourceLocal,
                        this->sourceGm[
                            outerStart * this->indexSize + sourceIndex],
                        sourceParams, sourcePadParams);
                    PipeBarrier<PIPE_ALL>();
                    Add(valueLocal, valueLocal, sourceLocal,
                        outerCount * groupElements);
                    PipeBarrier<PIPE_ALL>();
                    matched = static_cast<int32_t>(packed >> 3) - 1;
                }

                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(outerCount),
                    groupElements * static_cast<uint32_t>(sizeof(T)),
                    0,
                    (this->dimSize - groupElements) *
                        static_cast<uint32_t>(sizeof(T)),
                    0};
                DataCopyPad(
                    this->outputGm[
                        outerStart * this->dimSize + targetStart],
                    valueLocal, outputParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline bool CanUseDirectScalarAtomic() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const bool legacyShape =
            this->outerSize >= 32 && this->outerSize <= 256 &&
            this->dimSize > 512 && this->dimSize < 1024 &&
            this->indexSize >= 2048;
        const bool batchedShape =
            this->outerSize > 256 && this->dimSize >= 64 &&
            this->dimSize < 512 && this->indexSize >= 512;
        return this->innerSize == 1 && this->dimSize % alignElements == 0 &&
            this->totalSize >= 4096 && (legacyShape || batchedShape);
    }

    __aicore__ inline void ProcessDirectScalarAtomic()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        this->CopySelfForAtomic();
        SyncAll<true>();

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t indexEnd =
            indexStart + baseIndexes +
            (blockIdx < extraIndexes ? 1U : 0U);
        const uint32_t localIndexCount = indexEnd - indexStart;
        if (localIndexCount == 0) {
            return;
        }

        LocalTensor<T> sourceLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm[indexStart],
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        uint32_t maxOuterBatch =
            kGenericVectorAccumBytes /
            (alignElements * static_cast<uint32_t>(sizeof(T)));
        if (maxOuterBatch > 2048U) {
            maxOuterBatch = 2048U;
        }

        SetAtomicAdd<T>();
        for (uint32_t localIndex = 0; localIndex < localIndexCount;
             ++localIndex) {
            int32_t target = indexLocal.GetValue(localIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t lane = targetDim & (alignElements - 1U);
            const uint32_t alignedTarget = targetDim - lane;
            DataCopyPadExtParams<T> sourcePadParams{true,
                static_cast<uint8_t>(lane),
                static_cast<uint8_t>(alignElements - 1U - lane),
                static_cast<T>(0)};
            for (uint32_t outerStart = 0; outerStart < this->outerSize;
                 outerStart += maxOuterBatch) {
                uint32_t outerCount = this->outerSize - outerStart;
                if (outerCount > maxOuterBatch) {
                    outerCount = maxOuterBatch;
                }
                DataCopyExtParams sourceParams{
                    static_cast<uint16_t>(outerCount),
                    static_cast<uint32_t>(sizeof(T)),
                    (this->indexSize - 1) *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0};
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(outerCount),
                    alignElements * static_cast<uint32_t>(sizeof(T)), 0,
                    (this->dimSize - alignElements) *
                        static_cast<uint32_t>(sizeof(T)),
                    0};
                const uint32_t sourceOffset =
                    outerStart * this->indexSize + indexStart + localIndex;
                const uint32_t outputOffset =
                    outerStart * this->dimSize + alignedTarget;
                DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                    sourceParams, sourcePadParams);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(this->outputGm[outputOffset], sourceLocal,
                    outputParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
        SetAtomicNone();
    }

    __aicore__ inline bool CanUseAtomicVector() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        return this->innerSize >= alignElements &&
            this->innerSize % alignElements == 0 && this->outerSize < 32 &&
            this->dimSize <= 65535 && this->indexSize >= 512 &&
            this->indexSize < 2048 && this->totalSize >= 4096;
    }

    __aicore__ inline bool CanUsePaddedAtomicVector() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        return this->innerSize >= alignElements &&
            this->innerSize % alignElements != 0 && this->outerSize < 32 &&
            this->dimSize <= 65535 && this->indexSize >= 512 &&
            this->indexSize < 2048 && this->totalSize >= 4096;
    }

    __aicore__ inline bool CanUseDirectPaddedAtomicVector() const
    {
        return this->CanUsePaddedAtomicVector();
    }

    __aicore__ inline bool CanUseUnalignedDirectPaddedAtomicVector() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t paddedInner =
            ((this->innerSize + alignElements - 1) / alignElements) *
                alignElements;
        const uint64_t sourceLocalBytes =
            static_cast<uint64_t>(this->outerSize) * paddedInner * sizeof(T);
        const uint64_t outputStrideBytes =
            static_cast<uint64_t>(this->dimSize - 1U) *
                this->innerSize * sizeof(T);
        return this->CanUsePaddedAtomicVector() &&
            sourceLocalBytes <= kGenericVectorAccumBytes &&
            outputStrideBytes <= 0xFFFFFFFFULL;
    }

    __aicore__ inline void ProcessUnalignedDirectPaddedAtomicVector()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t paddedInner =
            ((this->innerSize + alignElements - 1) / alignElements) *
                alignElements;
        this->CopySelfForAtomic();
        SyncAll<true>();

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t indexEnd =
            indexStart + baseIndexes +
            (blockIdx < extraIndexes ? 1U : 0U);
        const uint32_t localIndexCount = indexEnd - indexStart;
        if (localIndexCount == 0) {
            return;
        }

        LocalTensor<T> sourceLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm[indexStart],
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        const uint32_t elementBytes = static_cast<uint32_t>(sizeof(T));
        DataCopyExtParams sourceParams{
            static_cast<uint16_t>(this->outerSize),
            this->innerSize * elementBytes,
            (this->indexSize - 1U) * this->innerSize * elementBytes,
            0, 0};
        DataCopyPadExtParams<T> sourcePadParams{true, 0,
            static_cast<uint8_t>(paddedInner - this->innerSize),
            static_cast<T>(0)};
        DataCopyExtParams outputParams{
            static_cast<uint16_t>(this->outerSize),
            this->innerSize * elementBytes,
            (paddedInner - this->innerSize) * elementBytes,
            (this->dimSize - 1U) * this->innerSize * elementBytes,
            0};

        SetAtomicAdd<T>();
        for (uint32_t localIndex = 0; localIndex < localIndexCount;
             ++localIndex) {
            int32_t target = indexLocal.GetValue(localIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            const uint32_t sourceOffset =
                (indexStart + localIndex) * this->innerSize;
            const uint32_t outputOffset =
                static_cast<uint32_t>(target) * this->innerSize;
            DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                sourceParams, sourcePadParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(this->outputGm[outputOffset], sourceLocal,
                outputParams);
            PipeBarrier<PIPE_ALL>();
        }
        SetAtomicNone();
    }

    __aicore__ inline bool CanUseDirectPaddedTargetTiled() const
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t outerPartitions = 8U;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        const uint32_t maxOwnedOuter =
            (this->outerSize + outerPartitions - 1U) / outerPartitions;
        const uint32_t indexBytes =
            AlignUp32(this->indexSize * sizeof(int32_t));
        const uint64_t allocatedBytes =
            static_cast<uint64_t>(this->directPaddedAccumBytes) +
            indexBytes;
        const uint64_t minimumAccumBytes =
            static_cast<uint64_t>(kC4TargetTiledAccumSafetyBytes) +
            3ULL * maxOwnedOuter * 2U * alignElements * sizeof(T);
        return UseInt32GroupedRepeat((T*)nullptr) &&
            blockNum >= outerPartitions &&
            blockNum % outerPartitions == 0U &&
            outputRowElements % alignElements == 0U &&
            this->totalSize % alignElements == 0U &&
            this->outerSize >= 8U && this->outerSize < 32U &&
            this->dimSize >= 2U && this->dimSize <= 32U &&
            this->innerSize >= 4096U && this->indexSize >= 512U &&
            this->indexSize < 2048U &&
            this->directPaddedAccumBytes >= minimumAccumBytes &&
            allocatedBytes + kC4TargetTiledUbSafetyBytes <=
                kGenericVectorTotalBytes;
    }

    __aicore__ inline void ProcessDirectPaddedTargetTiled()
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t outerPartitions = 8U;
        this->CopySelfForAtomic();
        SyncAll<true>();

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t workerCount = blockNum / outerPartitions;
        const uint32_t outerPartition = blockIdx / workerCount;
        const uint32_t workerIndex =
            blockIdx - outerPartition * workerCount;
        const uint32_t baseOuter = this->outerSize / outerPartitions;
        const uint32_t extraOuter =
            this->outerSize - baseOuter * outerPartitions;
        const uint32_t ownedOuterStart =
            outerPartition * baseOuter +
            (outerPartition < extraOuter ? outerPartition : extraOuter);
        const uint32_t ownedOuterSize =
            baseOuter + (outerPartition < extraOuter ? 1U : 0U);
        if (ownedOuterSize == 0U) {
            return;
        }

        const uint32_t maxOwnedOuter =
            (this->outerSize + outerPartitions - 1U) / outerPartitions;
        const uint32_t availableAccumBytes =
            this->directPaddedAccumBytes -
                kC4TargetTiledAccumSafetyBytes;
        uint32_t maxPaddedElements = availableAccumBytes /
            (3U * maxOwnedOuter * static_cast<uint32_t>(sizeof(T)));
        maxPaddedElements =
            (maxPaddedElements / alignElements) * alignElements;
        const uint32_t tileElements =
            maxPaddedElements - alignElements;
        const uint32_t bufferStride =
            maxOwnedOuter * maxPaddedElements;

        LocalTensor<T> accumLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<T> sourceBase = accumLocal[bufferStride];
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm,
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        const uint32_t elementBytes =
            static_cast<uint32_t>(sizeof(T));
        const uint32_t sourceRowElements =
            this->indexSize * this->innerSize;
        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        SetAtomicAdd<T>();
        for (uint32_t targetDim = 0U;
             targetDim < this->dimSize; ++targetDim) {
            uint32_t firstIndex = workerIndex;
            while (firstIndex < this->indexSize) {
                int32_t target = indexLocal.GetValue(firstIndex);
                if (target < 0) {
                    target += static_cast<int32_t>(this->dimSize);
                }
                if (target == static_cast<int32_t>(targetDim)) {
                    break;
                }
                firstIndex += workerCount;
            }
            if (firstIndex >= this->indexSize) {
                continue;
            }
            uint32_t innerStart = 0U;
            while (innerStart < this->innerSize) {
                uint32_t count = this->innerSize - innerStart;
                if (count > tileElements) {
                    count = tileElements;
                }
                const uint32_t globalTarget =
                    ownedOuterStart * outputRowElements +
                    targetDim * this->innerSize + innerStart;
                const uint32_t leftPadding =
                    globalTarget & (alignElements - 1U);
                const uint32_t paddedCount =
                    ((leftPadding + count + alignElements - 1U) /
                        alignElements) * alignElements;
                DataCopyExtParams sourceParams{
                    static_cast<uint16_t>(ownedOuterSize),
                    count * elementBytes,
                    (sourceRowElements - count) * elementBytes,
                    0, 0};
                DataCopyPadExtParams<T> sourcePadParams{true,
                    static_cast<uint8_t>(leftPadding),
                    static_cast<uint8_t>(paddedCount -
                        leftPadding - count),
                    static_cast<T>(0)};
                DataCopyPad(accumLocal,
                    this->sourceGm[
                        (ownedOuterStart * this->indexSize +
                            firstIndex) *
                                this->innerSize + innerStart],
                    sourceParams, sourcePadParams);
                PipeBarrier<PIPE_ALL>();

                uint32_t currentIndex = firstIndex + workerCount;
                while (currentIndex < this->indexSize) {
                    int32_t target = indexLocal.GetValue(currentIndex);
                    if (target < 0) {
                        target += static_cast<int32_t>(this->dimSize);
                    }
                    if (target == static_cast<int32_t>(targetDim)) {
                        break;
                    }
                    currentIndex += workerCount;
                }
                if (currentIndex < this->indexSize) {
                    uint32_t currentBuffer = 0U;
                    bool pending0 = false;
                    bool pending1 = false;
                    DataCopyPad(sourceBase,
                        this->sourceGm[
                            (ownedOuterStart * this->indexSize +
                                currentIndex) *
                                    this->innerSize + innerStart],
                        sourceParams, sourcePadParams);
                    SetFlag<HardEvent::MTE2_V>(0);

                    while (currentIndex < this->indexSize) {
                        uint32_t nextIndex = currentIndex + workerCount;
                        while (nextIndex < this->indexSize) {
                            int32_t target = indexLocal.GetValue(nextIndex);
                            if (target < 0) {
                                target += static_cast<int32_t>(this->dimSize);
                            }
                            if (target == static_cast<int32_t>(targetDim)) {
                                break;
                            }
                            nextIndex += workerCount;
                        }
                        const uint32_t nextBuffer = currentBuffer ^ 1U;
                        if (nextIndex < this->indexSize) {
                            if (nextBuffer == 0U && pending0) {
                                WaitFlag<HardEvent::V_MTE2>(0);
                                pending0 = false;
                            } else if (nextBuffer == 1U && pending1) {
                                WaitFlag<HardEvent::V_MTE2>(1);
                                pending1 = false;
                            }
                            LocalTensor<T> nextSourceLocal =
                                sourceBase[nextBuffer * bufferStride];
                            DataCopyPad(nextSourceLocal,
                                this->sourceGm[
                                    (ownedOuterStart * this->indexSize +
                                        nextIndex) *
                                            this->innerSize + innerStart],
                                sourceParams, sourcePadParams);
                            if (nextBuffer == 0U) {
                                SetFlag<HardEvent::MTE2_V>(0);
                            } else {
                                SetFlag<HardEvent::MTE2_V>(1);
                            }
                        }

                        LocalTensor<T> currentSourceLocal =
                            sourceBase[currentBuffer * bufferStride];
                        if (currentBuffer == 0U) {
                            WaitFlag<HardEvent::MTE2_V>(0);
                            Add(accumLocal, accumLocal, currentSourceLocal,
                                ownedOuterSize * paddedCount);
                            SetFlag<HardEvent::V_MTE2>(0);
                            pending0 = true;
                        } else {
                            WaitFlag<HardEvent::MTE2_V>(1);
                            Add(accumLocal, accumLocal, currentSourceLocal,
                                ownedOuterSize * paddedCount);
                            SetFlag<HardEvent::V_MTE2>(1);
                            pending1 = true;
                        }
                        currentIndex = nextIndex;
                        currentBuffer = nextBuffer;
                    }
                    if (pending0) {
                        WaitFlag<HardEvent::V_MTE2>(0);
                    }
                    if (pending1) {
                        WaitFlag<HardEvent::V_MTE2>(1);
                    }
                }

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                const uint32_t alignedTarget =
                    globalTarget - leftPadding;
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(ownedOuterSize),
                    paddedCount * elementBytes, 0,
                    (outputRowElements - paddedCount) * elementBytes,
                    0};
                DataCopyPad(this->outputGm[alignedTarget],
                    accumLocal, outputParams);
                PipeBarrier<PIPE_ALL>();
                innerStart += count;
            }
        }
        SetAtomicNone();
    }

    __aicore__ inline uint32_t CalcDirectPaddedPingPongRowElements() const
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        const uint32_t phaseStep = this->innerSize & (alignElements - 1U);
        const uint32_t phaseCount =
            this->dimSize < alignElements ? this->dimSize : alignElements;
        uint32_t phase = 0U;
        uint32_t maxLeftPadding = 0U;
        for (uint32_t target = 0U; target < phaseCount; ++target) {
            if (phase > maxLeftPadding) {
                maxLeftPadding = phase;
            }
            phase = (phase + phaseStep) & (alignElements - 1U);
        }
        return ((maxLeftPadding + this->innerSize + alignElements - 1U) /
            alignElements) * alignElements;
    }

    __aicore__ inline bool CanUseDirectPaddedAtomicPingPong() const
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t outerPartitions = 4U;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t rowBufferElements =
            this->CalcDirectPaddedPingPongRowElements();
        const uint32_t maxOwnedOuter =
            (this->outerSize + outerPartitions - 1U) / outerPartitions;
        const uint64_t pingPongBytes =
            2ULL * maxOwnedOuter * rowBufferElements *
                static_cast<uint32_t>(sizeof(T));
        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        return UseInt32GroupedRepeat((T*)nullptr) &&
            blockNum >= outerPartitions &&
            blockNum % outerPartitions == 0U &&
            outputRowElements % alignElements == 0U &&
            this->totalSize % alignElements == 0U &&
            this->outerSize >= 8U && this->outerSize < 32U &&
            this->dimSize >= 2U && this->dimSize <= 32U &&
            this->innerSize >= 2048U && this->indexSize >= 512U &&
            this->indexSize < 2048U &&
            pingPongBytes <= this->directPaddedAccumBytes;
    }

    __aicore__ inline void ProcessDirectPaddedAtomicPingPong()
    {
        this->sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t outerPartitions = 4U;
        this->CopySelfForAtomic();

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseWorkers = blockNum / outerPartitions;
        const uint32_t basePartition =
            blockIdx - (blockIdx / outerPartitions) * outerPartitions;
        const uint32_t baseWorker = blockIdx / outerPartitions;
        const uint32_t heavyPartitions = this->outerSize % outerPartitions;

        const uint32_t rowBufferElements =
            this->CalcDirectPaddedPingPongRowElements();
        const uint32_t maxOwnedOuter =
            (this->outerSize + outerPartitions - 1U) / outerPartitions;
        // Imbalanced outer families rotate complete logical partitions across
        // four worker-aligned index intervals. No additional global rendezvous.
        const uint32_t jobs = heavyPartitions == 0U ? 1U : outerPartitions;
        const uint32_t bufferStride = maxOwnedOuter * rowBufferElements;
        LocalTensor<T> sourceLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm,
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        const uint32_t elementBytes =
            static_cast<uint32_t>(sizeof(T));
        const uint32_t sourceRowElements =
            this->indexSize * this->innerSize;
        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        for (uint32_t job = 0U; job < jobs; ++job) {
        const uint32_t outerPartition = (basePartition + job) % outerPartitions;
        const uint32_t workerCount = baseWorkers;
        const uint32_t indexBegin = static_cast<uint32_t>(
            static_cast<uint64_t>(this->indexSize) * job /
                (static_cast<uint64_t>(jobs) * baseWorkers) * baseWorkers) + baseWorker;
        const uint32_t indexEnd = job + 1U == jobs ? this->indexSize :
            static_cast<uint32_t>(static_cast<uint64_t>(this->indexSize) * (job + 1U) /
                (static_cast<uint64_t>(jobs) * baseWorkers) * baseWorkers);
        const uint32_t ownedOuterStart = outerPartition;
        const uint32_t ownedOuterSize =
            (this->outerSize + outerPartitions - 1U - outerPartition) / outerPartitions;
        DataCopyExtParams sourceParams{
            static_cast<uint16_t>(ownedOuterSize),
            this->innerSize * elementBytes,
            (outerPartitions * sourceRowElements - this->innerSize) *
                elementBytes,
            0, 0};

        // V429: only matched half-block phases borrow four in-bounds words.
        const bool alignedReadFamily = UseInt32GroupedRepeat((T*)nullptr) &&
            (this->innerSize & 7U) == 4U &&
            (this->indexSize & 1U) == 0U &&
            (this->dimSize & 1U) == 0U &&
            ownedOuterSize > 0U && ownedOuterSize <= 8U &&
            rowBufferElements == this->innerSize + 4U &&
            rowBufferElements / 8U <= 65535U &&
            (reinterpret_cast<uint64_t>(this->sourceGm.GetPhyAddr()) & 31ULL) == 0ULL &&
            (reinterpret_cast<uint64_t>(this->outputGm.GetPhyAddr()) & 31ULL) == 0ULL;
        DataCopyExtParams alignedSourceParams{
            static_cast<uint16_t>(ownedOuterSize),
            rowBufferElements * elementBytes,
            (outerPartitions * sourceRowElements - rowBufferElements) *
                elementBytes,
            0, 0};
        DataCopyPadExtParams<T> noPadParams{false, 0, 0, static_cast<T>(0)};
        uint64_t leftEdgeMask = 0ULL;
        uint64_t rightEdgeMask = 0ULL;
        if (alignedReadFamily) {
            for (uint32_t row = 0U; row < ownedOuterSize; ++row) {
                leftEdgeMask |= 0x0FULL << (row * 8U);
                rightEdgeMask |= 0xF0ULL << (row * 8U);
            }
        }

        uint32_t currentIndex = indexBegin;
        int32_t currentTarget = -1;
        while (currentIndex < indexEnd) {
            currentTarget = indexLocal.GetValue(currentIndex);
            if (currentTarget < 0) {
                currentTarget += static_cast<int32_t>(this->dimSize);
            }
            if (currentTarget >= 0 &&
                currentTarget < static_cast<int32_t>(this->dimSize)) {
                break;
            }
            currentIndex += workerCount;
        }
        uint32_t currentBuffer = 0U;
        bool pending0 = false;
        bool pending1 = false;
        uint32_t currentGlobalTarget = 0U;
        uint32_t currentLeftPadding = 0U;
        uint32_t currentPaddedCount = 0U;
        bool currentAlignedRead = false;
        const bool hasSource = ownedOuterSize != 0U &&
            currentIndex < indexEnd;
        if (hasSource) {
            currentGlobalTarget =
                ownedOuterStart * outputRowElements +
                static_cast<uint32_t>(currentTarget) * this->innerSize;
            currentLeftPadding =
                currentGlobalTarget & (alignElements - 1U);
            currentPaddedCount =
                ((currentLeftPadding + this->innerSize + alignElements - 1U) /
                    alignElements) * alignElements;
            DataCopyPadExtParams<T> firstPadParams{true,
                static_cast<uint8_t>(currentLeftPadding),
                static_cast<uint8_t>(currentPaddedCount -
                    currentLeftPadding - this->innerSize),
                static_cast<T>(0)};
            currentAlignedRead = alignedReadFamily &&
                ((currentIndex * this->innerSize) & 7U) == currentLeftPadding;
            if (currentAlignedRead) {
                DataCopyPad(sourceLocal,
                    this->sourceGm[
                        (ownedOuterStart * this->indexSize + currentIndex) *
                            this->innerSize - currentLeftPadding],
                    alignedSourceParams, noPadParams);
            } else {
                DataCopyPad(sourceLocal,
                    this->sourceGm[
                        (ownedOuterStart * this->indexSize + currentIndex) *
                            this->innerSize],
                    sourceParams, firstPadParams);
            }
        }
        // Local self-copy writes are complete before sourceLocal is reused.
        // Preload on otherwise idle cores can overlap peers' initialization.
        // Every core joins here; no atomic output may precede this rendezvous.
        if (job == 0U) {
            SyncAll<true>();
        }
        if (!hasSource) {
            continue;
        }
        // Keep the ping-pong readiness token outside SyncAll's PIPE_ALL drain.
        SetFlag<HardEvent::MTE2_MTE3>(0);
        if (currentAlignedRead) {
            SetFlag<HardEvent::MTE2_V>(0);
        }

        SetAtomicAdd<T>();
        while (currentIndex < indexEnd) {
            uint32_t nextIndex = currentIndex + workerCount;
            int32_t nextTarget = -1;
            bool nextAlignedRead = false;
            while (nextIndex < indexEnd) {
                nextTarget = indexLocal.GetValue(nextIndex);
                if (nextTarget < 0) {
                    nextTarget += static_cast<int32_t>(this->dimSize);
                }
                if (nextTarget >= 0 &&
                    nextTarget < static_cast<int32_t>(this->dimSize)) {
                    break;
                }
                nextIndex += workerCount;
            }

            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextIndex < indexEnd) {
                if (nextBuffer == 0U && pending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    pending0 = false;
                } else if (nextBuffer == 1U && pending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                    pending1 = false;
                }
                const uint32_t nextGlobalTarget =
                    ownedOuterStart * outputRowElements +
                    static_cast<uint32_t>(nextTarget) * this->innerSize;
                const uint32_t nextLeftPadding =
                    nextGlobalTarget & (alignElements - 1U);
                const uint32_t nextPaddedCount =
                    ((nextLeftPadding + this->innerSize +
                        alignElements - 1U) / alignElements) *
                            alignElements;
                DataCopyPadExtParams<T> nextPadParams{true,
                    static_cast<uint8_t>(nextLeftPadding),
                    static_cast<uint8_t>(nextPaddedCount -
                        nextLeftPadding - this->innerSize),
                    static_cast<T>(0)};
                LocalTensor<T> nextLocal =
                    sourceLocal[nextBuffer * bufferStride];
                nextAlignedRead = alignedReadFamily &&
                    ((nextIndex * this->innerSize) & 7U) == nextLeftPadding;
                if (nextAlignedRead) {
                    DataCopyPad(nextLocal,
                        this->sourceGm[
                            (ownedOuterStart * this->indexSize + nextIndex) *
                                this->innerSize - nextLeftPadding],
                        alignedSourceParams, noPadParams);
                } else {
                    DataCopyPad(nextLocal,
                        this->sourceGm[
                            (ownedOuterStart * this->indexSize + nextIndex) *
                                this->innerSize],
                        sourceParams, nextPadParams);
                }
                if (nextBuffer == 0U) {
                    SetFlag<HardEvent::MTE2_MTE3>(0);
                } else {
                    SetFlag<HardEvent::MTE2_MTE3>(1);
                }
                if (nextAlignedRead) {
                    if (nextBuffer == 0U) {
                        SetFlag<HardEvent::MTE2_V>(0);
                    } else {
                        SetFlag<HardEvent::MTE2_V>(1);
                    }
                }
            }

            LocalTensor<T> currentLocal =
                sourceLocal[currentBuffer * bufferStride];
            const uint32_t alignedTarget =
                currentGlobalTarget - currentLeftPadding;
            DataCopyExtParams outputParams{
                static_cast<uint16_t>(ownedOuterSize),
                currentPaddedCount * elementBytes, 0,
                (outerPartitions * outputRowElements -
                    currentPaddedCount) * elementBytes,
                0};
            // V429: MTE2 completes before V clears borrowed edge words.
            // V then completes before the unchanged atomic MTE3 consumes them.
            if (currentAlignedRead) {
                if (currentBuffer == 0U) {
                    WaitFlag<HardEvent::MTE2_V>(0);
                } else {
                    WaitFlag<HardEvent::MTE2_V>(1);
                }
                LocalTensor<int32_t> edgeLocal =
                    currentLocal.template ReinterpretCast<int32_t>();
                uint64_t edgeMask[2] = {
                    currentLeftPadding == 0U ? rightEdgeMask : leftEdgeMask,
                    0ULL};
                const uint32_t edgeStart =
                    currentLeftPadding == 0U ? rowBufferElements - 8U : 0U;
                Duplicate<int32_t>(edgeLocal[edgeStart], static_cast<int32_t>(0),
                    edgeMask, static_cast<uint8_t>(1),
                    static_cast<uint16_t>(rowBufferElements / 8U),
                    static_cast<uint8_t>(0));
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
            }
            if (currentBuffer == 0U) {
                WaitFlag<HardEvent::MTE2_MTE3>(0);
                DataCopyPad(this->outputGm[alignedTarget],
                    currentLocal, outputParams);
                SetFlag<HardEvent::MTE3_MTE2>(0);
                pending0 = true;
            } else {
                WaitFlag<HardEvent::MTE2_MTE3>(1);
                DataCopyPad(this->outputGm[alignedTarget],
                    currentLocal, outputParams);
                SetFlag<HardEvent::MTE3_MTE2>(1);
                pending1 = true;
            }

            currentIndex = nextIndex;
            currentTarget = nextTarget;
            currentBuffer = nextBuffer;
            currentAlignedRead = nextAlignedRead;
            if (currentIndex < indexEnd) {
                currentGlobalTarget =
                    ownedOuterStart * outputRowElements +
                    static_cast<uint32_t>(currentTarget) * this->innerSize;
                currentLeftPadding =
                    currentGlobalTarget & (alignElements - 1U);
                currentPaddedCount =
                    ((currentLeftPadding + this->innerSize +
                        alignElements - 1U) / alignElements) *
                            alignElements;
            }
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
        SetAtomicNone();
        }
    }

    __aicore__ inline void ProcessDirectPaddedAtomicVector()
    {
        if (this->CanUseDirectPaddedTargetTiled()) {
            this->ProcessDirectPaddedTargetTiled();
            return;
        }
        if (this->CanUseDirectPaddedAtomicPingPong()) {
            this->ProcessDirectPaddedAtomicPingPong();
            return;
        }
        constexpr uint32_t alignElements = 32 / sizeof(T);
        this->CopySelfForAtomic();
        SyncAll<true>();

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= this->indexSize) {
            return;
        }

        LocalTensor<T> sourceLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm,
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        const uint32_t sourceRowElements =
            this->indexSize * this->innerSize;
        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        const uint32_t outputRowPhase =
            outputRowElements & (alignElements - 1U);
        uint32_t workerCount = blockNum;
        uint32_t workerIndex = blockIdx;
        uint32_t ownedOuterStart = 0;
        uint32_t ownedOuterSize = this->outerSize;
        const bool useOuterSplit =
            UseInt32GroupedRepeat((T*)nullptr) &&
            (blockNum & 1U) == 0 && blockNum >= 2 &&
            outputRowPhase == 0 &&
            this->totalSize % alignElements == 0 &&
            this->outerSize >= 8 && this->outerSize < 32 &&
            this->dimSize >= 2 && this->dimSize <= 32 &&
            this->innerSize >= 4096 && this->indexSize >= 512 &&
            this->indexSize < 2048;
        if (useOuterSplit) {
            constexpr uint32_t outerPartitions = 2;
            workerCount = blockNum / outerPartitions;
            const uint32_t outerPartition = blockIdx / workerCount;
            workerIndex = blockIdx - outerPartition * workerCount;
            const uint32_t baseOuter =
                this->outerSize / outerPartitions;
            const uint32_t extraOuter =
                this->outerSize - baseOuter * outerPartitions;
            ownedOuterStart =
                outerPartition * baseOuter +
                (outerPartition < extraOuter
                    ? outerPartition : extraOuter);
            ownedOuterSize =
                baseOuter + (outerPartition < extraOuter ? 1U : 0U);
        }
        uint32_t outerPhasePeriod = 1;
        if ((outputRowPhase & 1U) != 0) {
            outerPhasePeriod = 8;
        } else if ((outputRowPhase & 2U) != 0) {
            outerPhasePeriod = 4;
        } else if ((outputRowPhase & 4U) != 0) {
            outerPhasePeriod = 2;
        }
        uint32_t outerPhaseCount = outerPhasePeriod;
        if (outerPhaseCount > ownedOuterSize) {
            outerPhaseCount = ownedOuterSize;
        }
        const bool processTailSeparately =
            this->totalSize % alignElements != 0;
        const uint32_t tailPhaseOuter =
            (this->outerSize - 1U) % outerPhasePeriod;

        SetAtomicAdd<T>();
        for (uint32_t sourceIndex = workerIndex;
             sourceIndex < this->indexSize; sourceIndex += workerCount) {
            int32_t target = indexLocal.GetValue(sourceIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            const uint32_t targetRowElement =
                static_cast<uint32_t>(target) * this->innerSize;
            for (uint32_t phaseOuter = 0; phaseOuter < outerPhaseCount;
                 ++phaseOuter) {
                uint32_t groupOuterCount =
                    (ownedOuterSize - 1U - phaseOuter) /
                        outerPhasePeriod + 1U;
                if (processTailSeparately &&
                    target == static_cast<int32_t>(this->dimSize - 1U) &&
                    phaseOuter == tailPhaseOuter) {
                    --groupOuterCount;
                    if (groupOuterCount == 0) {
                        continue;
                    }
                }
                uint32_t maxPaddedElements =
                    this->directPaddedAccumBytes /
                        sizeof(T) / groupOuterCount;
                maxPaddedElements =
                    (maxPaddedElements / alignElements) * alignElements;
                uint32_t innerStart = 0;
                while (innerStart < this->innerSize) {
                    const uint32_t globalTarget =
                        (ownedOuterStart + phaseOuter) * outputRowElements +
                        targetRowElement + innerStart;
                    const uint32_t leftPadding =
                        globalTarget & (alignElements - 1U);
                    uint32_t count = this->innerSize - innerStart;
                    const uint32_t maxCount =
                        maxPaddedElements - leftPadding;
                    if (count > maxCount) {
                        count = maxCount;
                    }
                    const uint32_t alignedTarget =
                        globalTarget - leftPadding;
                    const uint32_t paddedCount =
                        ((leftPadding + count + alignElements - 1U) /
                            alignElements) * alignElements;
                    const uint32_t rightPadding =
                        paddedCount - leftPadding - count;
                    DataCopyPadExtParams<T> sourcePadParams{true,
                        static_cast<uint8_t>(leftPadding),
                        static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
                    DataCopyExtParams sourceParams{
                        static_cast<uint16_t>(groupOuterCount),
                        count * static_cast<uint32_t>(sizeof(T)),
                        (outerPhasePeriod * sourceRowElements - count) *
                            static_cast<uint32_t>(sizeof(T)),
                        0, 0};
                    DataCopyExtParams outputParams{
                        static_cast<uint16_t>(groupOuterCount),
                        paddedCount * static_cast<uint32_t>(sizeof(T)), 0,
                        (outerPhasePeriod * outputRowElements - paddedCount) *
                            static_cast<uint32_t>(sizeof(T)),
                        0};
                    const uint32_t sourceOffset =
                        ((ownedOuterStart + phaseOuter) * this->indexSize +
                            sourceIndex) * this->innerSize +
                        innerStart;
                    DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                        sourceParams, sourcePadParams);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(this->outputGm[alignedTarget], sourceLocal,
                        outputParams);
                    PipeBarrier<PIPE_ALL>();
                    innerStart += count;
                }
            }
        }
        SetAtomicNone();
        if (processTailSeparately) {
            SyncAll<true>();
            if (blockIdx == 0) {
                this->ProcessDirectPaddedAtomicTail();
            }
        }
    }

    __aicore__ inline void ProcessDirectPaddedAtomicTail()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        constexpr uint32_t bufferElements =
            kGenericVectorAccumBytes / sizeof(T);
        constexpr uint32_t chunkElements =
            (bufferElements / 2U / alignElements) * alignElements;
        LocalTensor<T> valueLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = valueLocal[chunkElements];
        LocalTensor<int32_t> indexLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm,
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        uint32_t matchingCount = 0;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = indexLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target == static_cast<int32_t>(this->dimSize - 1U)) {
                indexLocal.SetValue(matchingCount,
                    static_cast<int32_t>(i));
                ++matchingCount;
            }
        }
        if (matchingCount == 0) {
            return;
        }

        const uint32_t outputRowElements =
            this->dimSize * this->innerSize;
        const uint32_t tailOutputOffset =
            (this->outerSize - 1U) * outputRowElements +
            (this->dimSize - 1U) * this->innerSize;
        const uint32_t sourceOuterOffset =
            (this->outerSize - 1U) * this->indexSize * this->innerSize;
        DataCopyPadExtParams<T> noPadParams{false, 0, 0, 0};
        for (uint32_t innerStart = 0; innerStart < this->innerSize;
             innerStart += chunkElements) {
            uint32_t count = this->innerSize - innerStart;
            if (count > chunkElements) {
                count = chunkElements;
            }
            DataCopyExtParams copyParams{1,
                count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(valueLocal,
                this->outputGm[tailOutputOffset + innerStart],
                copyParams, noPadParams);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t match = 0; match < matchingCount; ++match) {
                const uint32_t sourceIndex = static_cast<uint32_t>(
                    indexLocal.GetValue(match));
                const uint32_t sourceOffset =
                    sourceOuterOffset + sourceIndex * this->innerSize +
                    innerStart;
                DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                    copyParams, noPadParams);
                PipeBarrier<PIPE_ALL>();
                Add(valueLocal, valueLocal, sourceLocal, count);
                PipeBarrier<PIPE_V>();
            }
            DataCopyPad(this->outputGm[tailOutputOffset + innerStart],
                valueLocal, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline uint32_t CalcPaddedAtomicInner() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        return ((this->innerSize + alignElements - 1) / alignElements) *
            alignElements;
    }

    __aicore__ inline void CopyPaddedAtomicWorkspace(bool packSelf)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        constexpr uint32_t copyElements =
            kGenericVectorAccumBytes / sizeof(T);
        const uint32_t paddedInner = this->CalcPaddedAtomicInner();
        const uint32_t innerTile =
            (copyElements / alignElements) * alignElements;
        const uint32_t rowCount = this->outerSize * this->dimSize;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        LocalTensor<T> copyLocal = this->accumBuf.template Get<T>();
        // For the usual grouped-vector shapes, a complete padded row fits in UB.
        // Copy several consecutive rows per DMA command instead of issuing two
        // commands and two barriers for every row.
        if (paddedInner <= copyElements) {
            const uint32_t baseRows = rowCount / blockNum;
            const uint32_t extraRows = rowCount - baseRows * blockNum;
            const uint32_t rowStart =
                blockIdx * baseRows + (blockIdx < extraRows ? blockIdx : extraRows);
            const uint32_t rowEnd =
                rowStart + baseRows + (blockIdx < extraRows ? 1U : 0U);
            uint32_t rowsPerBatch = copyElements / paddedInner;
            if (rowsPerBatch > 4095U) {
                rowsPerBatch = 4095U;
            }
            const uint32_t realBytes = this->innerSize * sizeof(T);
            const uint32_t paddedBytes = paddedInner * sizeof(T);

            for (uint32_t row = rowStart; row < rowEnd; row += rowsPerBatch) {
                uint32_t rows = rowEnd - row;
                if (rows > rowsPerBatch) {
                    rows = rowsPerBatch;
                }
                const uint32_t realOffset = row * this->innerSize;
                const uint32_t paddedOffset = row * paddedInner;
                if (packSelf) {
                    DataCopyExtParams loadParams{
                        static_cast<uint16_t>(rows), realBytes, 0, 0, 0};
                    DataCopyPadExtParams<T> loadPadParams{
                        paddedInner != this->innerSize, 0,
                        static_cast<uint8_t>(paddedInner - this->innerSize),
                        static_cast<T>(0)};
                    DataCopyPad(copyLocal, this->selfGm[realOffset],
                        loadParams, loadPadParams);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyParams storeParams{
                        static_cast<uint16_t>(rows),
                        static_cast<uint16_t>(paddedBytes / 32), 0, 0};
                    DataCopy(this->workspaceGm[paddedOffset], copyLocal,
                        storeParams);
                    PipeBarrier<PIPE_ALL>();
                } else {
                    DataCopyParams loadParams{
                        static_cast<uint16_t>(rows),
                        static_cast<uint16_t>(paddedBytes / 32), 0, 0};
                    DataCopy(copyLocal, this->workspaceGm[paddedOffset],
                        loadParams);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyExtParams storeParams{
                        static_cast<uint16_t>(rows), realBytes, 0, 0, 0};
                    DataCopyPad(this->outputGm[realOffset], copyLocal,
                        storeParams);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            return;
        }

        const uint32_t chunkCount =
            (this->innerSize + innerTile - 1) / innerTile;
        const uint32_t taskCount = rowCount * chunkCount;
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / chunkCount;
            const uint32_t chunk = task - row * chunkCount;
            const uint32_t innerStart = chunk * innerTile;
            uint32_t count = this->innerSize - innerStart;
            if (count > innerTile) {
                count = innerTile;
            }
            const uint32_t paddedCount =
                ((count + alignElements - 1) / alignElements) * alignElements;
            const uint32_t countBytes = count * sizeof(T);
            const uint32_t realOffset = row * this->innerSize + innerStart;
            const uint32_t paddedOffset = row * paddedInner + innerStart;

            if (packSelf) {
                DataCopyExtParams loadParams{1, countBytes, 0, 0, 0};
                DataCopyPadExtParams<T> loadPadParams{
                    paddedCount != count, 0,
                    static_cast<uint8_t>(paddedCount - count), static_cast<T>(0)};
                DataCopyPad(copyLocal, this->selfGm[realOffset],
                    loadParams, loadPadParams);
                PipeBarrier<PIPE_ALL>();
                DataCopy(this->workspaceGm[paddedOffset], copyLocal,
                    paddedCount);
                PipeBarrier<PIPE_ALL>();
            } else {
                DataCopy(copyLocal, this->workspaceGm[paddedOffset],
                    paddedCount);
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams storeParams{1, countBytes, 0, 0, 0};
                DataCopyPad(this->outputGm[realOffset], copyLocal,
                    storeParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline bool CanDoubleBufferPaddedAtomic() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        constexpr uint32_t maxElements =
            kGenericVectorAccumBytes / sizeof(T);
        const uint32_t paddedInner =
            ((this->innerSize + alignElements - 1) / alignElements) *
                alignElements;
        const uint64_t bufferElements =
            static_cast<uint64_t>(this->outerSize) * paddedInner;
        return 2ULL * bufferElements <= maxElements;
    }

    __aicore__ inline void AccumulatePaddedAtomicWorkspaceDoubleBuffered(
        const LocalTensor<int32_t>& indexLocal, uint32_t indexStart,
        uint32_t localIndexCount)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t paddedInner = this->CalcPaddedAtomicInner();
        const uint32_t bufferStride = this->outerSize * paddedInner;
        LocalTensor<T> sourceLocal = this->accumBuf.template Get<T>();
        DataCopyExtParams sourceParams{
            static_cast<uint16_t>(this->outerSize),
            this->innerSize * static_cast<uint32_t>(sizeof(T)),
            (this->indexSize - 1U) * this->innerSize *
                static_cast<uint32_t>(sizeof(T)),
            0, 0};
        DataCopyPadExtParams<T> sourcePadParams{
            paddedInner != this->innerSize, 0,
            static_cast<uint8_t>(paddedInner - this->innerSize),
            static_cast<T>(0)};
        const uint32_t paddedBytes = paddedInner * sizeof(T);
        const uint64_t workspaceStrideBlocks =
            (static_cast<uint64_t>(this->dimSize) * paddedInner -
                paddedInner) * sizeof(T) / 32;
        const bool useAlignedWorkspace = workspaceStrideBlocks <= 65535ULL;
        DataCopyParams alignedWorkspaceParams{
            static_cast<uint16_t>(this->outerSize),
            static_cast<uint16_t>(paddedBytes / 32), 0,
            static_cast<uint16_t>(workspaceStrideBlocks)};
        DataCopyExtParams workspaceParams{
            static_cast<uint16_t>(this->outerSize), paddedBytes, 0,
            (this->dimSize * paddedInner - paddedInner) *
                static_cast<uint32_t>(sizeof(T)),
            0};

        uint32_t currentIndex = 0;
        int32_t currentTarget = -1;
        while (currentIndex < localIndexCount) {
            currentTarget = indexLocal.GetValue(currentIndex);
            if (currentTarget < 0) {
                currentTarget += static_cast<int32_t>(this->dimSize);
            }
            if (currentTarget >= 0 &&
                currentTarget < static_cast<int32_t>(this->dimSize)) {
                break;
            }
            ++currentIndex;
        }
        if (currentIndex >= localIndexCount) {
            return;
        }

        uint32_t currentBuffer = 0;
        bool pending0 = false;
        bool pending1 = false;
        DataCopyPad(sourceLocal,
            this->sourceGm[(indexStart + currentIndex) * this->innerSize],
            sourceParams, sourcePadParams);
        SetFlag<HardEvent::MTE2_MTE3>(0);

        SetAtomicAdd<T>();
        while (currentIndex < localIndexCount) {
            uint32_t nextIndex = currentIndex + 1U;
            int32_t nextTarget = -1;
            while (nextIndex < localIndexCount) {
                nextTarget = indexLocal.GetValue(nextIndex);
                if (nextTarget < 0) {
                    nextTarget += static_cast<int32_t>(this->dimSize);
                }
                if (nextTarget >= 0 &&
                    nextTarget < static_cast<int32_t>(this->dimSize)) {
                    break;
                }
                ++nextIndex;
            }

            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextIndex < localIndexCount) {
                if (nextBuffer == 0 && pending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    pending0 = false;
                } else if (nextBuffer == 1 && pending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                    pending1 = false;
                }
                LocalTensor<T> nextLocal =
                    sourceLocal[nextBuffer * bufferStride];
                DataCopyPad(nextLocal,
                    this->sourceGm[(indexStart + nextIndex) * this->innerSize],
                    sourceParams, sourcePadParams);
                if (nextBuffer == 0) {
                    SetFlag<HardEvent::MTE2_MTE3>(0);
                } else {
                    SetFlag<HardEvent::MTE2_MTE3>(1);
                }
            }

            LocalTensor<T> currentLocal =
                sourceLocal[currentBuffer * bufferStride];
            const uint32_t workspaceOffset =
                static_cast<uint32_t>(currentTarget) * paddedInner;
            if (currentBuffer == 0) {
                WaitFlag<HardEvent::MTE2_MTE3>(0);
                if (useAlignedWorkspace) {
                    DataCopy(this->workspaceGm[workspaceOffset], currentLocal,
                        alignedWorkspaceParams);
                } else {
                    DataCopyPad(this->workspaceGm[workspaceOffset], currentLocal,
                        workspaceParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(0);
                pending0 = true;
            } else {
                WaitFlag<HardEvent::MTE2_MTE3>(1);
                if (useAlignedWorkspace) {
                    DataCopy(this->workspaceGm[workspaceOffset], currentLocal,
                        alignedWorkspaceParams);
                } else {
                    DataCopyPad(this->workspaceGm[workspaceOffset], currentLocal,
                        workspaceParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(1);
                pending1 = true;
            }

            currentIndex = nextIndex;
            currentTarget = nextTarget;
            currentBuffer = nextBuffer;
        }

        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
        SetAtomicNone();
    }

    __aicore__ inline void AccumulatePaddedAtomicWorkspace()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t paddedInner = this->CalcPaddedAtomicInner();
        const uint32_t innerTile = this->CalcAtomicInner();
        if (innerTile == 0) {
            return;
        }

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t indexEnd =
            indexStart + baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        LocalTensor<T> sourceLocal = this->accumBuf.template Get<T>();
        LocalTensor<int32_t> indexLocal = this->nextBuf.template Get<int32_t>();
        const uint32_t localIndexCount = indexEnd - indexStart;
        DataCopyExtParams indexParams{
            1, localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, this->indexGm[indexStart],
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        if (this->CanDoubleBufferPaddedAtomic()) {
            this->AccumulatePaddedAtomicWorkspaceDoubleBuffered(
                indexLocal, indexStart, localIndexCount);
            return;
        }

        SetAtomicAdd<T>();
        for (uint32_t i = indexStart; i < indexEnd; ++i) {
            int32_t target = indexLocal.GetValue(i - indexStart);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            for (uint32_t innerStart = 0; innerStart < this->innerSize;
                 innerStart += innerTile) {
                uint32_t count = this->innerSize - innerStart;
                if (count > innerTile) {
                    count = innerTile;
                }
                const uint32_t paddedCount =
                    ((count + alignElements - 1) / alignElements) * alignElements;
                const uint32_t countBytes = count * sizeof(T);
                const uint32_t paddedBytes = paddedCount * sizeof(T);
                const uint32_t sourceOffset =
                    i * this->innerSize + innerStart;
                const uint32_t workspaceOffset =
                    static_cast<uint32_t>(target) * paddedInner + innerStart;
                DataCopyExtParams sourceParams{
                    static_cast<uint16_t>(this->outerSize), countBytes,
                    (this->indexSize * this->innerSize - count) *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0};
                DataCopyPadExtParams<T> sourcePadParams{
                    paddedCount != count, 0,
                    static_cast<uint8_t>(paddedCount - count), static_cast<T>(0)};
                DataCopyExtParams workspaceParams{
                    static_cast<uint16_t>(this->outerSize), paddedBytes, 0,
                    (this->dimSize * paddedInner - paddedCount) *
                        static_cast<uint32_t>(sizeof(T)),
                    0};
                DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                    sourceParams, sourcePadParams);
                PipeBarrier<PIPE_ALL>();
                const uint64_t workspaceStrideBlocks =
                    (static_cast<uint64_t>(this->dimSize) * paddedInner -
                        paddedCount) * sizeof(T) / 32;
                if (workspaceStrideBlocks <= 65535ULL) {
                    DataCopyParams alignedWorkspaceParams{
                        static_cast<uint16_t>(this->outerSize),
                        static_cast<uint16_t>(paddedBytes / 32), 0,
                        static_cast<uint16_t>(workspaceStrideBlocks)};
                    DataCopy(this->workspaceGm[workspaceOffset], sourceLocal,
                        alignedWorkspaceParams);
                } else {
                    DataCopyPad(this->workspaceGm[workspaceOffset],
                        sourceLocal, workspaceParams);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
        SetAtomicNone();
    }

    __aicore__ inline void ProcessPaddedAtomicVector()
    {
        this->CopyPaddedAtomicWorkspace(true);
        SyncAll<true>();
        this->AccumulatePaddedAtomicWorkspace();
        SyncAll<true>();
        this->CopyPaddedAtomicWorkspace(false);
    }

    __aicore__ inline void CopySelfForAtomic()
    {
        constexpr uint32_t copyElements = kGenericVectorAccumBytes / sizeof(T);
        LocalTensor<T> copyLocal = this->accumBuf.template Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        const uint32_t taskCount =
            (this->totalSize + copyElements - 1) / copyElements;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t start = task * copyElements;
            uint32_t count = this->totalSize - start;
            if (count > copyElements) {
                count = copyElements;
            }
            const uint32_t countBytes = count * sizeof(T);
            DataCopyExtParams copyParams{1, countBytes, 0, 0, 0};
            DataCopyPad(copyLocal, this->selfGm[start], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(this->outputGm[start], copyLocal, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline uint32_t CalcAtomicInner() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t maxElements = kGenericVectorAccumBytes / sizeof(T);
        uint32_t innerTile = maxElements / this->outerSize;
        innerTile = (innerTile / alignElements) * alignElements;
        if (innerTile < alignElements) {
            return 0;
        }
        if (this->innerSize < innerTile) {
            innerTile =
                ((this->innerSize + alignElements - 1) / alignElements) *
                    alignElements;
        }
        return innerTile;
    }

    __aicore__ inline void ProcessAtomicVector()
    {
        this->CopySelfForAtomic();
        SyncAll<true>();

        const uint32_t innerTile = this->CalcAtomicInner();
        if (innerTile == 0) {
            return;
        }

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t indexEnd =
            indexStart + baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        LocalTensor<T> sourceLocal = this->accumBuf.template Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        SetAtomicAdd<T>();
        for (uint32_t i = indexStart; i < indexEnd; ++i) {
            int32_t target = this->indexGm.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            for (uint32_t innerStart = 0; innerStart < this->innerSize;
                 innerStart += innerTile) {
                uint32_t count = this->innerSize - innerStart;
                if (count > innerTile) {
                    count = innerTile;
                }
                const uint32_t countBytes = count * sizeof(T);
                const uint32_t sourceOffset = i * this->innerSize + innerStart;
                const uint32_t outputOffset =
                    static_cast<uint32_t>(target) * this->innerSize + innerStart;
                DataCopyExtParams sourceParams{
                    static_cast<uint16_t>(this->outerSize), countBytes,
                    (this->indexSize * this->innerSize - count) *
                        static_cast<uint32_t>(sizeof(T)),
                    0, 0};
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(this->outerSize), countBytes, 0,
                    (this->dimSize * this->innerSize - count) *
                        static_cast<uint32_t>(sizeof(T)),
                    0};
                DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                    sourceParams, padParams);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(this->outputGm[outputOffset], sourceLocal,
                    outputParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
        SetAtomicNone();
    }

    __aicore__ inline bool PrepareGenericWave()
    {
        LocalTensor<int32_t> countLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.template Get<int32_t>();

        if (UseGenericBlockReduce((T*)nullptr)) {
            return this->PrepareGenericSplitWave(countLocal, targetLocal);
        }

        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();

        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();
        this->genericWaveCount = 0;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                targetLocal.SetValue(i, -1);
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t count = static_cast<uint32_t>(countLocal.GetValue(targetDim)) + 1;
            countLocal.SetValue(targetDim, static_cast<int32_t>(count));
            targetLocal.SetValue(i, target);
            if (count > this->genericWaveCount) {
                this->genericWaveCount = count;
            }
        }

        constexpr uint32_t alignElements = 32 / sizeof(T);
        this->genericWaveStride =
            AlignUp32(this->dimSize * sizeof(uint32_t)) / sizeof(uint32_t);
        this->genericZeroElement = AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        this->genericWaveValueStride = AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        this->genericSelfElement = this->genericZeroElement + alignElements;
        this->genericUseBlockReduce = UseGenericBlockReduce((T*)nullptr);
        this->genericWaveGroupStride = 8 * this->genericWaveStride;
        const uint32_t fullGroupCount = this->genericWaveCount / 8;
        const uint32_t tailWaveCount = this->genericWaveCount - fullGroupCount * 8;
        const bool reduceTail = this->genericUseBlockReduce &&
            tailWaveCount >= kBlockReduceTailMinWaves;
        this->genericWaveIncludeSelf = reduceTail && fullGroupCount == 0;
        this->genericWaveSourceStride = this->genericSelfElement +
            (this->genericWaveIncludeSelf ? this->genericWaveValueStride : 0U);
        this->genericWaveBlockGroups = this->genericUseBlockReduce
            ? fullGroupCount + (reduceTail ? 1U : 0U)
            : 0;
        this->genericWaveRemainder = this->genericUseBlockReduce
            ? (reduceTail ? 0U : tailWaveCount)
            : this->genericWaveCount;
        const uint64_t waveElements =
            static_cast<uint64_t>(this->genericWaveCount) * this->genericWaveStride;
        const uint64_t waveOffsetElements = this->genericUseBlockReduce
            ? static_cast<uint64_t>(this->genericWaveBlockGroups) * this->genericWaveGroupStride +
                static_cast<uint64_t>(this->genericWaveRemainder) * this->genericWaveStride
            : waveElements;
        this->genericWaveGatherElements = waveOffsetElements == 0
            ? this->genericWaveValueStride
            : (this->genericWaveBlockGroups > 0
                ? this->genericWaveGroupStride
                : static_cast<uint32_t>(waveOffsetElements));
        const uint64_t waveDataBytes = waveOffsetElements * sizeof(uint32_t);
        if (waveDataBytes > static_cast<uint64_t>(0xFFFFFFFFU) - 31U) {
            return false;
        }
        const uint32_t waveBytes = waveOffsetElements == 0
            ? 32U
            : AlignUp32(static_cast<uint32_t>(waveDataBytes));
        const uint64_t gatheredWaveBytes =
            static_cast<uint64_t>(this->genericWaveGatherElements) * sizeof(T);
        const uint64_t fixedBytes =
            static_cast<uint64_t>(AlignUp32(this->dimSize * sizeof(int32_t))) +
            AlignUp32(this->indexSize * sizeof(int32_t)) + waveBytes +
            kGenericWaveValueElements * sizeof(T) +
            gatheredWaveBytes;
        const uint64_t sourceRowBytes =
            static_cast<uint64_t>(this->genericWaveSourceStride) * sizeof(T);
        if (fixedBytes >= kGenericWaveTotalBytes || sourceRowBytes == 0) {
            return false;
        }
        uint64_t maxBatch = (kGenericWaveTotalBytes - fixedBytes) / sourceRowBytes;
        const uint32_t maxValueRows = kGenericWaveValueElements / this->genericWaveValueStride;
        if (maxBatch > maxValueRows) {
            maxBatch = maxValueRows;
        }
        if (maxBatch > kWaveOuterBatchMax) {
            maxBatch = kWaveOuterBatchMax;
        }
        if (maxBatch > this->outerSize) {
            maxBatch = this->outerSize;
        }
        if (maxBatch == 0) {
            return false;
        }
        this->genericWaveOuterBatch = static_cast<uint32_t>(maxBatch);

        this->pipe.InitBuffer(this->waveOffsetBuf, waveBytes);
        LocalTensor<uint32_t> waveOffsetLocal = this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t zeroByteOffset = this->genericZeroElement * sizeof(T);
        if (waveOffsetElements > 0) {
            Duplicate(waveOffsetLocal, zeroByteOffset,
                static_cast<uint32_t>(waveOffsetElements));
            PipeBarrier<PIPE_ALL>();
        }
        if (this->genericWaveIncludeSelf) {
            for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
                waveOffsetLocal.SetValue(dim * 8,
                    (this->genericSelfElement + dim) * sizeof(T));
            }
        }
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();
        if (this->genericUseBlockReduce) {
            const uint32_t blockWaveLimit = this->genericWaveBlockGroups * 8;
            for (uint32_t i = 0; i < this->indexSize; ++i) {
                const int32_t target = targetLocal.GetValue(i);
                if (target < 0) {
                    continue;
                }
                const uint32_t targetDim = static_cast<uint32_t>(target);
                const uint32_t wave = static_cast<uint32_t>(countLocal.GetValue(targetDim));
                uint32_t waveOffset;
                if (wave < blockWaveLimit) {
                    const uint32_t group = wave >> 3;
                    const uint32_t lane = (wave & 7U) +
                        (this->genericWaveIncludeSelf ? 1U : 0U);
                    waveOffset = group * this->genericWaveGroupStride + targetDim * 8 + lane;
                } else {
                    const uint32_t remainderWave = wave - blockWaveLimit;
                    waveOffset = this->genericWaveBlockGroups * this->genericWaveGroupStride +
                        remainderWave * this->genericWaveStride + targetDim;
                }
                waveOffsetLocal.SetValue(waveOffset, i * sizeof(T));
                countLocal.SetValue(targetDim, static_cast<int32_t>(wave + 1));
            }
        } else {
            for (uint32_t i = 0; i < this->indexSize; ++i) {
                const int32_t target = targetLocal.GetValue(i);
                if (target < 0) {
                    continue;
                }
                const uint32_t targetDim = static_cast<uint32_t>(target);
                const uint32_t wave = static_cast<uint32_t>(countLocal.GetValue(targetDim));
                waveOffsetLocal.SetValue(
                    wave * this->genericWaveStride + targetDim, i * sizeof(T));
                countLocal.SetValue(targetDim, static_cast<int32_t>(wave + 1));
            }
        }
        return true;
    }

    __aicore__ inline bool PrepareGenericSplitWave(
        const LocalTensor<int32_t>& countLocal, const LocalTensor<int32_t>& targetLocal)
    {
        if (this->useParallelSharedPlan) {
            return this->PrepareParallelSharedGenericSplitWave(
                countLocal, targetLocal);
        }
        return this->PrepareGenericSplitWaveLocal(countLocal, targetLocal);
    }

    __aicore__ inline bool TryLoadParallelSharedPlanCache(
        const LocalTensor<int32_t>& targetLocal,
        uint32_t indexStart, uint32_t localIndexCount)
    {
        if (!this->sharedPlanCacheEnabled) {
            return false;
        }
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        (void)indexStart;
        LocalTensor<int32_t> cachedIndexLocal =
            this->headBuf.template Get<int32_t>()[
                this->sharedAlignedDim];
        LocalTensor<int32_t> headerLocal =
            cachedIndexLocal[this->sharedCachedIndexStride];
        DataCopy(headerLocal, this->sharedPlanGm[0],
            kSharedPlanHeaderWords);
        PipeBarrier<PIPE_ALL>();
        const bool headerMatches =
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheMagicAWord)) == kSharedPlanCacheMagicA &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheMagicBWord)) == kSharedPlanCacheMagicB &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheVersionWord)) == kSharedPlanCacheVersion &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheBlockNumWord)) == blockNum &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheOuterSizeWord)) == this->outerSize &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheDimSizeWord)) == this->dimSize &&
            static_cast<uint32_t>(headerLocal.GetValue(
                kCacheIndexSizeWord)) == this->indexSize;
        if (!headerMatches) {
            return false;
        }

        DataCopyExtParams cachedIndexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> cachedIndexPadParams{
            false, 0, 0, 0};
        DataCopyPad(cachedIndexLocal,
            this->sharedPlanGm[this->sharedCachedIndexBase +
                blockIdx * this->sharedCachedIndexStride],
            cachedIndexParams, cachedIndexPadParams);
        PipeBarrier<PIPE_ALL>();
        uint32_t mismatch = 0;
        for (uint32_t localIndex = 0;
             localIndex < localIndexCount; ++localIndex) {
            if (cachedIndexLocal.GetValue(localIndex) !=
                targetLocal.GetValue(localIndex)) {
                mismatch = 1U;
                break;
            }
        }
        constexpr uint32_t cacheFlagStride = 8U;
        LocalTensor<int32_t> cacheFlagsLocal =
            headerLocal[kSharedPlanHeaderWords];
        Duplicate(cacheFlagsLocal, static_cast<int32_t>(mismatch),
            cacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->sharedPlanGm[
                this->sharedCachedFlagBase +
                blockIdx * cacheFlagStride],
            cacheFlagsLocal, cacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        DataCopy(cacheFlagsLocal,
            this->sharedPlanGm[this->sharedCachedFlagBase],
            blockNum * cacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t core = 0; core < blockNum; ++core) {
            if (cacheFlagsLocal.GetValue(core * cacheFlagStride) != 0) {
                return false;
            }
        }

        this->sharedUsePackedFallback =
            headerLocal.GetValue(kCacheFallbackWord) != 0;
        if (this->sharedUsePackedFallback) {
            return true;
        }
        this->genericSplitWave = true;
        this->genericUseBlockReduce = true;
        this->genericWaveCount = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheWaveCountWord));
        this->genericWaveStride = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheWaveStrideWord));
        this->genericZeroElement = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheZeroElementWord));
        this->genericWaveValueStride = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheValueStrideWord));
        this->genericSelfElement = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheSelfElementWord));
        this->genericOuterGroup = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheOuterGroupWord));
        this->genericWaveIncludeSelf =
            headerLocal.GetValue(kCacheIncludeSelfWord) != 0;
        this->genericWaveSourceStride = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheSourceStrideWord));
        this->genericWaveBlockGroups = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheBlockGroupsWord));
        this->genericWaveRemainder = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheRemainderWord));
        this->genericWaveGroupStride = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheGroupStrideWord));
        this->genericWaveGatherElements = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheGatherElementsWord));
        this->genericWaveOuterBatch = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheOuterBatchWord));
        const uint32_t firstWaveBytes = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheFirstWaveBytesWord));
        const uint32_t overflowWaveBytes = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheOverflowWaveBytesWord));
        const uint32_t overflowWaveElements = static_cast<uint32_t>(
            headerLocal.GetValue(kCacheOverflowWaveElementsWord));
        this->pipe.InitBuffer(this->waveOffsetBuf, firstWaveBytes);
        this->pipe.InitBuffer(this->waveOverflowOffsetBuf,
            overflowWaveBytes);
        LocalTensor<int32_t> firstWaveLocal =
            this->waveOffsetBuf.template Get<int32_t>();
        LocalTensor<int32_t> overflowWaveLocal =
            this->waveOverflowOffsetBuf.template Get<int32_t>();
        DataCopy(firstWaveLocal,
            this->sharedPlanGm[this->sharedWaveBase],
            firstWaveBytes / sizeof(uint32_t));
        if (overflowWaveElements > 0) {
            const uint32_t firstWaveElements =
                this->genericWaveGroupStride;
            DataCopy(overflowWaveLocal,
                this->sharedPlanGm[
                    this->sharedWaveBase + firstWaveElements],
                overflowWaveBytes / sizeof(uint32_t));
        }
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline void CommitParallelSharedPlanCache(
        const LocalTensor<int32_t>& targetLocal,
        uint32_t indexStart, uint32_t localIndexCount,
        bool fallback, uint32_t firstWaveBytes,
        uint32_t overflowWaveBytes, uint32_t overflowWaveElements)
    {
        (void)targetLocal;
        (void)indexStart;
        (void)localIndexCount;
        if (!this->sharedPlanCacheEnabled) {
            return;
        }
        if (GetBlockIdx() == 0) {
            this->sharedPlanGm.SetValue(kCacheVersionWord,
                static_cast<int32_t>(kSharedPlanCacheVersion));
            this->sharedPlanGm.SetValue(kCacheBlockNumWord,
                static_cast<int32_t>(GetBlockNum()));
            this->sharedPlanGm.SetValue(kCacheOuterSizeWord,
                static_cast<int32_t>(this->outerSize));
            this->sharedPlanGm.SetValue(kCacheDimSizeWord,
                static_cast<int32_t>(this->dimSize));
            this->sharedPlanGm.SetValue(kCacheIndexSizeWord,
                static_cast<int32_t>(this->indexSize));
            this->sharedPlanGm.SetValue(kCacheFallbackWord,
                fallback ? 1 : 0);
            this->sharedPlanGm.SetValue(kCacheWaveCountWord,
                static_cast<int32_t>(this->genericWaveCount));
            this->sharedPlanGm.SetValue(kCacheWaveStrideWord,
                static_cast<int32_t>(this->genericWaveStride));
            this->sharedPlanGm.SetValue(kCacheZeroElementWord,
                static_cast<int32_t>(this->genericZeroElement));
            this->sharedPlanGm.SetValue(kCacheValueStrideWord,
                static_cast<int32_t>(this->genericWaveValueStride));
            this->sharedPlanGm.SetValue(kCacheSelfElementWord,
                static_cast<int32_t>(this->genericSelfElement));
            this->sharedPlanGm.SetValue(kCacheOuterGroupWord,
                static_cast<int32_t>(this->genericOuterGroup));
            this->sharedPlanGm.SetValue(kCacheIncludeSelfWord,
                this->genericWaveIncludeSelf ? 1 : 0);
            this->sharedPlanGm.SetValue(kCacheSourceStrideWord,
                static_cast<int32_t>(this->genericWaveSourceStride));
            this->sharedPlanGm.SetValue(kCacheBlockGroupsWord,
                static_cast<int32_t>(this->genericWaveBlockGroups));
            this->sharedPlanGm.SetValue(kCacheRemainderWord,
                static_cast<int32_t>(this->genericWaveRemainder));
            this->sharedPlanGm.SetValue(kCacheGroupStrideWord,
                static_cast<int32_t>(this->genericWaveGroupStride));
            this->sharedPlanGm.SetValue(kCacheGatherElementsWord,
                static_cast<int32_t>(this->genericWaveGatherElements));
            this->sharedPlanGm.SetValue(kCacheOuterBatchWord,
                static_cast<int32_t>(this->genericWaveOuterBatch));
            this->sharedPlanGm.SetValue(kCacheFirstWaveBytesWord,
                static_cast<int32_t>(firstWaveBytes));
            this->sharedPlanGm.SetValue(kCacheOverflowWaveBytesWord,
                static_cast<int32_t>(overflowWaveBytes));
            this->sharedPlanGm.SetValue(kCacheOverflowWaveElementsWord,
                static_cast<int32_t>(overflowWaveElements));
            this->sharedPlanGm.SetValue(kCacheMagicBWord,
                static_cast<int32_t>(kSharedPlanCacheMagicB));
            this->sharedPlanGm.SetValue(kCacheMagicAWord,
                static_cast<int32_t>(kSharedPlanCacheMagicA));
        }
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->sharedPlanGm);
        SyncAll<true>();
    }

    __aicore__ inline bool PrepareParallelSharedGenericSplitWave(
        const LocalTensor<int32_t>& countLocal,
        const LocalTensor<int32_t>& targetLocal)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        constexpr uint32_t selfLane = 7;
        constexpr uint32_t groupSourceLanes = 8;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t localIndexCount =
            baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        DataCopyExtParams indexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm[indexStart],
            indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();
        if (this->TryLoadParallelSharedPlanCache(
                targetLocal, indexStart, localIndexCount)) {
            return true;
        }
        if (this->sharedPlanCacheEnabled) {
            DataCopyPad(
                this->sharedPlanGm[this->sharedCachedIndexBase +
                    blockIdx * this->sharedCachedIndexStride],
                targetLocal, indexParams);
            PipeBarrier<PIPE_ALL>();
        }
        Duplicate(countLocal, static_cast<int32_t>(0),
            this->sharedAlignedDim);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t localIndex = 0;
             localIndex < localIndexCount; ++localIndex) {
            int32_t target = targetLocal.GetValue(localIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 ||
                target >= static_cast<int32_t>(this->dimSize)) {
                targetLocal.SetValue(localIndex, -1);
                continue;
            }
            targetLocal.SetValue(localIndex, target);
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const int32_t count = countLocal.GetValue(targetDim);
            countLocal.SetValue(targetDim, count + 1);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->sharedPlanGm[
            this->sharedLocalCountBase +
            blockIdx * this->sharedAlignedDim],
            countLocal, this->sharedAlignedDim);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<int32_t> cursorLocal =
            this->headBuf.template Get<int32_t>()[
                this->sharedAlignedDim];
        LocalTensor<int32_t> packedLocal =
            this->accumBuf.template Get<int32_t>();
        uint32_t localPackedWords = 0;
        for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
            const uint32_t localCount = static_cast<uint32_t>(
                countLocal.GetValue(dim));
            countLocal.SetValue(
                dim, static_cast<int32_t>(localPackedWords));
            cursorLocal.SetValue(
                dim, static_cast<int32_t>(localPackedWords));
            localPackedWords += (localCount + 7U) & ~7U;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->sharedPlanGm[
            this->sharedCoreStartBase +
            blockIdx * this->sharedAlignedDim],
            countLocal, this->sharedAlignedDim);
        if (localPackedWords > 0) {
            Duplicate(packedLocal, static_cast<int32_t>(0),
                localPackedWords);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t localIndex = 0;
             localIndex < localIndexCount; ++localIndex) {
            const int32_t target = targetLocal.GetValue(localIndex);
            if (target < 0) {
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t localOffset =
                static_cast<uint32_t>(
                    cursorLocal.GetValue(targetDim));
            packedLocal.SetValue(
                localOffset,
                static_cast<int32_t>(indexStart + localIndex));
            cursorLocal.SetValue(
                targetDim, static_cast<int32_t>(localOffset + 1U));
        }
        PipeBarrier<PIPE_ALL>();
        if (localPackedWords > 0) {
            DataCopy(this->sharedPlanGm[
                this->sharedPackedBase +
                blockIdx * this->sharedLocalPackedCapacity],
                packedLocal, localPackedWords);
        }
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->sharedPlanGm);
        SyncAll<true>();

        constexpr uint32_t cacheLineWords = 16;
        const uint32_t dimGroups =
            this->sharedAlignedDim / cacheLineWords;
        const uint32_t activePlanCores =
            dimGroups < blockNum ? dimGroups : blockNum;
        const uint32_t baseGroups = dimGroups / blockNum;
        const uint32_t extraGroups =
            dimGroups - baseGroups * blockNum;
        const uint32_t groupStart =
            blockIdx * baseGroups +
            (blockIdx < extraGroups ? blockIdx : extraGroups);
        const uint32_t groupEnd =
            groupStart + baseGroups +
            (blockIdx < extraGroups ? 1U : 0U);
        const uint32_t dimStart =
            groupStart * cacheLineWords;
        const uint32_t planDimEnd = groupEnd * cacheLineWords;
        uint32_t dimEnd = planDimEnd;
        if (dimEnd > this->dimSize) {
            dimEnd = this->dimSize;
        }
        // Empty plan owners still participate in every collective, but must
        // not turn a padded starting coordinate into an unsigned row count.
        if (dimEnd < dimStart) {
            dimEnd = dimStart;
        }
        const uint32_t reduceOwnedDimCount = dimEnd - dimStart;
        const uint32_t reducePlanDimCount = planDimEnd - dimStart;
        const uint32_t reduceMatrixStride =
            (reduceOwnedDimCount + 7U) & ~7U;
        if (reduceOwnedDimCount > 0) {
            DataCopyExtParams reduceMatrixParams{
                static_cast<uint16_t>(blockNum),
                reduceOwnedDimCount *
                    static_cast<uint32_t>(sizeof(int32_t)),
                (this->sharedAlignedDim - reduceOwnedDimCount) *
                    static_cast<uint32_t>(sizeof(int32_t)),
                0, 0};
            DataCopyPadExtParams<int32_t> reduceMatrixPadParams{true, 0,
                static_cast<uint8_t>(
                    reduceMatrixStride - reduceOwnedDimCount), 0};
            DataCopyPad(packedLocal,
                this->sharedPlanGm[
                    this->sharedLocalCountBase + dimStart],
                reduceMatrixParams, reduceMatrixPadParams);
            Duplicate(cursorLocal, static_cast<int32_t>(0),
                reducePlanDimCount);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t core = 0; core < blockNum; ++core) {
                Add(cursorLocal, cursorLocal,
                    packedLocal[core * reduceMatrixStride],
                    reduceOwnedDimCount);
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t ownedMaxCount = 0;
            for (uint32_t localDim = 0;
                 localDim < reduceOwnedDimCount; ++localDim) {
                const uint32_t count = static_cast<uint32_t>(
                    cursorLocal.GetValue(localDim));
                if (count > ownedMaxCount) {
                    ownedMaxCount = count;
                }
            }
            this->sharedPlanGm.SetValue(
                this->sharedGlobalCountBase + blockIdx * cacheLineWords,
                static_cast<int32_t>(ownedMaxCount));
        }
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->sharedPlanGm);
        SyncAll<true>();

        uint32_t maxCount = 0;
        // Only these slots were published, and all fit the aligned-dim row.
        for (uint32_t core = 0; core < activePlanCores; ++core) {
            const uint32_t count = static_cast<uint32_t>(
                this->sharedPlanGm.GetValue(
                    this->sharedGlobalCountBase +
                    core * cacheLineWords));
            if (count > maxCount) {
                maxCount = count;
            }
        }

        this->genericSplitWave = true;
        this->genericWaveCount = maxCount;
        this->genericWaveStride = this->sharedAlignedDim;
        this->genericZeroElement =
            AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        this->genericWaveValueStride =
            AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        this->genericSelfElement =
            this->genericZeroElement + alignElements;
        this->genericUseBlockReduce = true;
        this->genericWaveGroupStride =
            8U * this->genericWaveStride;
        this->genericOuterGroup = this->genericAllocatedOuterGroup;

        this->genericWaveIncludeSelf =
            this->genericWaveCount < groupSourceLanes;
        this->genericWaveSourceStride = this->genericSelfElement +
            (this->genericWaveIncludeSelf
                ? this->genericWaveValueStride : 0U);
        const uint32_t overflowWaveCount =
            this->genericWaveCount > groupSourceLanes
                ? this->genericWaveCount - groupSourceLanes : 0U;
        const uint32_t fullGroupCount = overflowWaveCount / 8U;
        const uint32_t tailWaveCount =
            overflowWaveCount - fullGroupCount * 8U;
        const bool reduceTail =
            tailWaveCount >= kBlockReduceTailMinWaves;
        this->genericWaveBlockGroups =
            fullGroupCount + (reduceTail ? 1U : 0U);
        this->genericWaveRemainder =
            reduceTail ? 0U : tailWaveCount;
        const uint32_t firstWaveElements =
            this->genericWaveGroupStride;
        const uint32_t overflowWaveElements =
            this->genericWaveBlockGroups *
                this->genericWaveGroupStride +
            this->genericWaveRemainder *
                this->genericWaveStride;
        const uint32_t totalWaveElements =
            firstWaveElements + overflowWaveElements;
        const uint32_t firstWaveBytes =
            AlignUp32(firstWaveElements * sizeof(uint32_t));
        const uint32_t overflowWaveBytes =
            overflowWaveElements == 0 ? 32U :
            AlignUp32(overflowWaveElements * sizeof(uint32_t));

        const uint32_t allocatedHeadRows =
            this->genericAllocatedHeadRows;
        const uint64_t fixedBaseBytes =
            static_cast<uint64_t>(allocatedHeadRows) *
                this->genericHeadStrideBytes +
            this->sharedNextBytes + firstWaveBytes +
            overflowWaveBytes +
            kParallelSharedWaveValueElements * sizeof(T);
        const uint64_t sourceRowBytes =
            static_cast<uint64_t>(
                this->genericWaveSourceStride) * sizeof(T);
        uint64_t maxBatch = 0;
        while (this->genericOuterGroup > 0) {
            this->genericWaveGatherElements =
                this->genericOuterGroup *
                this->genericWaveGroupStride;
            const uint64_t fixedBytes = fixedBaseBytes +
                static_cast<uint64_t>(
                    this->genericWaveGatherElements) * sizeof(T);
            maxBatch = fixedBytes < kGenericWaveTotalBytes
                ? (kGenericWaveTotalBytes - fixedBytes) /
                    sourceRowBytes
                : 0U;
            const uint32_t maxValueRows =
                kParallelSharedWaveValueElements /
                this->genericWaveValueStride;
            if (maxBatch > maxValueRows) {
                maxBatch = maxValueRows;
            }
            if (maxBatch > kWaveOuterBatchMax) {
                maxBatch = kWaveOuterBatchMax;
            }
            if (maxBatch > this->outerSize) {
                maxBatch = this->outerSize;
            }
            if (maxBatch >= this->genericOuterGroup ||
                this->genericOuterGroup == 1U) {
                break;
            }
            --this->genericOuterGroup;
        }
        const uint32_t maxOwnedPlanDimCount =
            ((dimGroups + blockNum - 1U) / blockNum) * cacheLineWords;
        const uint32_t maxMatrixStride =
            (maxOwnedPlanDimCount + 7U) & ~7U;
        const uint32_t scheduleGroupCount =
            (maxCount + kSharedPlanBankGroupCount - 1U) /
                kSharedPlanBankGroupCount;
        const uint64_t maxLocalPlanWords =
            static_cast<uint64_t>(blockNum) * maxMatrixStride +
            static_cast<uint64_t>(maxOwnedPlanDimCount) *
                (8U * (1U + this->genericWaveBlockGroups) +
                 this->genericWaveRemainder) +
            2ULL * scheduleGroupCount;
        const uint32_t cacheableWaveCapacity =
            this->sharedWaveCapacity - this->sharedCachedIndexWords;
        if (totalWaveElements > cacheableWaveCapacity ||
            maxLocalPlanWords > kParallelSharedWaveValueElements ||
            this->genericOuterGroup == 0 || maxBatch == 0) {
            this->sharedUsePackedFallback = true;
            this->CommitParallelSharedPlanCache(targetLocal,
                indexStart, localIndexCount, true,
                firstWaveBytes, overflowWaveBytes,
                static_cast<uint32_t>(overflowWaveElements));
            return true;
        }
        this->genericWaveOuterBatch =
            static_cast<uint32_t>(maxBatch);

        this->pipe.InitBuffer(this->waveOffsetBuf,
            firstWaveBytes);
        this->pipe.InitBuffer(this->waveOverflowOffsetBuf,
            overflowWaveBytes);
        const uint32_t zeroByteOffset =
            this->genericZeroElement * sizeof(T);

        const uint32_t ownedDimCount = dimEnd - dimStart;
        const uint32_t matrixStride =
            (ownedDimCount + 7U) & ~7U;
        LocalTensor<int32_t> countMatrix = packedLocal;
        if (ownedDimCount > 0) {
            DataCopyExtParams matrixParams{
                static_cast<uint16_t>(blockNum),
                ownedDimCount * static_cast<uint32_t>(sizeof(int32_t)),
                (this->sharedAlignedDim - ownedDimCount) *
                    static_cast<uint32_t>(sizeof(int32_t)),
                0, 0};
            DataCopyPadExtParams<int32_t> matrixPadParams{true, 0,
                static_cast<uint8_t>(matrixStride - ownedDimCount), 0};
            DataCopyPad(countMatrix,
                this->sharedPlanGm[
                    this->sharedLocalCountBase + dimStart],
                matrixParams, matrixPadParams);
            PipeBarrier<PIPE_ALL>();
        }

        const uint32_t planDimCount = planDimEnd - dimStart;
        const uint32_t ownedGroupWords = planDimCount * 8U;
        const uint32_t ownedWaveWords =
            (1U + this->genericWaveBlockGroups) * ownedGroupWords +
            this->genericWaveRemainder * planDimCount;
        LocalTensor<int32_t> ownedWaveLocal =
            packedLocal[blockNum * matrixStride];
        LocalTensor<int32_t> scheduleLocal =
            ownedWaveLocal[ownedWaveWords];
        if (ownedWaveWords > 0) {
            Duplicate(ownedWaveLocal,
                static_cast<int32_t>(zeroByteOffset), ownedWaveWords);
            PipeBarrier<PIPE_ALL>();
        }

        const uint32_t blockWaveLimit =
            this->genericWaveBlockGroups * 8U;
        for (uint32_t dim = dimStart; dim < dimEnd; ++dim) {
            for (uint32_t group = 0U;
                 group < scheduleGroupCount; ++group) {
                scheduleLocal.SetValue(group, 0);
                scheduleLocal.SetValue(scheduleGroupCount + group, 0);
            }
            for (uint32_t core = 0;
                 core < blockNum; ++core) {
                const uint32_t localCount =
                    static_cast<uint32_t>(
                        countMatrix.GetValue(
                            core * matrixStride + dim - dimStart));
                const uint32_t packedStart =
                    static_cast<uint32_t>(
                        this->sharedPlanGm.GetValue(
                            this->sharedCoreStartBase +
                            core * this->sharedAlignedDim + dim));
                for (uint32_t localRank = 0;
                     localRank < localCount;
                     ++localRank) {
                    const uint32_t sourceIndex =
                        static_cast<uint32_t>(
                            this->sharedPlanGm.GetValue(
                                this->sharedPackedBase +
                                core * this->sharedLocalPackedCapacity +
                                packedStart + localRank));
                const uint32_t bankBit = 1U <<
                    ((sourceIndex >> kSharedPlanBankGroupShift) &
                        (kSharedPlanBankGroupCount - 1U));
                uint32_t scheduledGroup = scheduleGroupCount;
                for (uint32_t group = 0U;
                     group < scheduleGroupCount; ++group) {
                    const uint32_t capacity =
                        group + 1U < scheduleGroupCount
                            ? kSharedPlanBankGroupCount
                            : maxCount - group *
                                kSharedPlanBankGroupCount;
                    const uint32_t fill = static_cast<uint32_t>(
                        scheduleLocal.GetValue(group));
                    const uint32_t bankMask = static_cast<uint32_t>(
                        scheduleLocal.GetValue(
                            scheduleGroupCount + group));
                    if (fill < capacity && (bankMask & bankBit) == 0U) {
                        scheduledGroup = group;
                        break;
                    }
                }
                if (scheduledGroup == scheduleGroupCount) {
                    for (uint32_t group = 0U;
                         group < scheduleGroupCount; ++group) {
                        const uint32_t capacity =
                            group + 1U < scheduleGroupCount
                                ? kSharedPlanBankGroupCount
                                : maxCount - group *
                                    kSharedPlanBankGroupCount;
                        if (static_cast<uint32_t>(
                                scheduleLocal.GetValue(group)) < capacity) {
                            scheduledGroup = group;
                            break;
                        }
                    }
                }
                const uint32_t scheduledLane = static_cast<uint32_t>(
                    scheduleLocal.GetValue(scheduledGroup));
                const uint32_t scheduledRank =
                    scheduledGroup * kSharedPlanBankGroupCount +
                        scheduledLane;
                scheduleLocal.SetValue(
                    scheduledGroup,
                    static_cast<int32_t>(scheduledLane + 1U));
                scheduleLocal.SetValue(
                    scheduleGroupCount + scheduledGroup,
                    scheduleLocal.GetValue(
                        scheduleGroupCount + scheduledGroup) |
                            static_cast<int32_t>(bankBit));
                uint32_t localWaveOffset;
                if (scheduledRank < groupSourceLanes) {
                    localWaveOffset =
                        (dim - dimStart) * 8U + scheduledRank;
                } else {
                    const uint32_t wave =
                        scheduledRank - groupSourceLanes;
                    if (wave < blockWaveLimit) {
                        const uint32_t group = wave >> 3;
                        const uint32_t lane = wave & 7U;
                        localWaveOffset =
                            (1U + group) * ownedGroupWords +
                            (dim - dimStart) * 8U + lane;
                    } else {
                        const uint32_t remainderWave =
                            wave - blockWaveLimit;
                        localWaveOffset =
                            (1U + this->genericWaveBlockGroups) *
                                ownedGroupWords +
                            remainderWave * planDimCount +
                            dim - dimStart;
                    }
                }
                ownedWaveLocal.SetValue(
                    localWaveOffset,
                    static_cast<int32_t>(
                        sourceIndex * sizeof(T)));
                }
            }
            if (this->genericWaveIncludeSelf) {
                ownedWaveLocal.SetValue(
                    (dim - dimStart) * 8U + selfLane,
                    static_cast<int32_t>(
                        (this->genericSelfElement + dim) *
                        sizeof(T)));
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (planDimCount > 0) {
            for (uint32_t section = 0;
                 section <= this->genericWaveBlockGroups; ++section) {
                const uint32_t gmSectionBase = section == 0
                    ? 0U
                    : firstWaveElements +
                        (section - 1U) * this->genericWaveGroupStride;
                DataCopy(this->sharedPlanGm[
                    this->sharedWaveBase + gmSectionBase + dimStart * 8U],
                    ownedWaveLocal[section * ownedGroupWords],
                    ownedGroupWords);
            }
            const uint32_t gmRemainderBase =
                firstWaveElements +
                this->genericWaveBlockGroups *
                    this->genericWaveGroupStride;
            const uint32_t localRemainderBase =
                (1U + this->genericWaveBlockGroups) * ownedGroupWords;
            for (uint32_t wave = 0;
                 wave < this->genericWaveRemainder; ++wave) {
                DataCopy(this->sharedPlanGm[
                    this->sharedWaveBase + gmRemainderBase +
                    wave * this->genericWaveStride + dimStart],
                    ownedWaveLocal[
                        localRemainderBase + wave * planDimCount],
                    planDimCount);
            }
        }
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->sharedPlanGm);
        SyncAll<true>();

        LocalTensor<int32_t> firstWaveLocal =
            this->waveOffsetBuf.template Get<int32_t>();
        LocalTensor<int32_t> overflowWaveLocal =
            this->waveOverflowOffsetBuf.template Get<int32_t>();
        DataCopy(firstWaveLocal,
            this->sharedPlanGm[this->sharedWaveBase],
            firstWaveBytes / sizeof(uint32_t));
        if (overflowWaveElements > 0) {
            DataCopy(overflowWaveLocal,
                this->sharedPlanGm[
                    this->sharedWaveBase + firstWaveElements],
                overflowWaveBytes / sizeof(uint32_t));
        }
        PipeBarrier<PIPE_ALL>();
        this->CommitParallelSharedPlanCache(targetLocal,
            indexStart, localIndexCount, false,
            firstWaveBytes, overflowWaveBytes,
            static_cast<uint32_t>(overflowWaveElements));
        return true;
    }

    __aicore__ inline bool PrepareGenericSplitWaveLocal(
        const LocalTensor<int32_t>& countLocal,
        const LocalTensor<int32_t>& targetLocal)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        constexpr uint32_t selfLane = 7;
        constexpr uint32_t groupSourceLanes = 8;
        this->genericSplitWave = true;
        this->genericWaveCount = 0;
        this->genericWaveStride =
            AlignUp32(this->dimSize * sizeof(uint32_t)) / sizeof(uint32_t);
        this->genericZeroElement = AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        this->genericWaveValueStride = AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        this->genericSelfElement = this->genericZeroElement + alignElements;
        this->genericUseBlockReduce = true;
        this->genericWaveGroupStride = 8 * this->genericWaveStride;
        this->genericOuterGroup = this->genericAllocatedOuterGroup;
        this->genericWaveGatherElements =
            this->genericOuterGroup * this->genericWaveGroupStride;

        const uint32_t firstWaveElements = this->genericWaveGroupStride;
        const uint64_t firstWaveDataBytes =
            static_cast<uint64_t>(firstWaveElements) * sizeof(uint32_t);
        if (firstWaveDataBytes > static_cast<uint64_t>(0xFFFFFFFFU) - 31U) {
            return false;
        }
        const uint32_t firstWaveBytes =
            AlignUp32(static_cast<uint32_t>(firstWaveDataBytes));
        this->pipe.InitBuffer(this->waveOffsetBuf, firstWaveBytes);
        LocalTensor<uint32_t> firstWaveOffsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t zeroByteOffset = this->genericZeroElement * sizeof(T);
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(firstWaveOffsetLocal, zeroByteOffset, firstWaveElements);
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();
        uint32_t overflowCount = 0;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            // IndexAdd requires every index to be in [0, dimSize).  Avoid two
            // scalar branches per element on this hot preparation path.
            const uint32_t targetDim =
                static_cast<uint32_t>(targetLocal.GetValue(i));
            const uint32_t count =
                static_cast<uint32_t>(countLocal.GetValue(targetDim));
            if (count < groupSourceLanes) {
                firstWaveOffsetLocal.SetValue(targetDim * 8 + count, i * sizeof(T));
            } else {
                const uint32_t overflowWave = count - groupSourceLanes;
                const uint32_t packed = ((overflowWave & 0xFFU) << 24) |
                    (i << 10) | targetDim;
                targetLocal.SetValue(overflowCount, static_cast<int32_t>(packed));
                ++overflowCount;
            }
            const uint32_t nextCount = count + 1;
            countLocal.SetValue(targetDim, static_cast<int32_t>(nextCount));
        }

        // The maximum count only changes once per destination bucket.  Scanning
        // the much smaller dim table avoids a compare and conditional update for
        // every source index on large, narrow IndexAdd workloads.
        uint32_t maxWaveCount = 0;
        for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
            const uint32_t count =
                static_cast<uint32_t>(countLocal.GetValue(dim));
            if (count > maxWaveCount) {
                maxWaveCount = count;
            }
        }
        this->genericWaveCount = maxWaveCount;

        this->genericWaveIncludeSelf = this->genericWaveCount < groupSourceLanes;
        this->genericWaveSourceStride = this->genericSelfElement +
            (this->genericWaveIncludeSelf ? this->genericWaveValueStride : 0U);
        if (this->genericWaveIncludeSelf) {
            for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
                firstWaveOffsetLocal.SetValue(dim * 8 + selfLane,
                    (this->genericSelfElement + dim) * sizeof(T));
            }
        }

        const uint32_t overflowWaveCount = this->genericWaveCount > groupSourceLanes
            ? this->genericWaveCount - groupSourceLanes
            : 0U;
        const uint32_t fullGroupCount = overflowWaveCount / 8;
        const uint32_t tailWaveCount = overflowWaveCount - fullGroupCount * 8;
        const bool reduceTail = tailWaveCount >= kBlockReduceTailMinWaves;
        this->genericWaveBlockGroups = fullGroupCount + (reduceTail ? 1U : 0U);
        this->genericWaveRemainder = reduceTail ? 0U : tailWaveCount;
        const uint64_t overflowWaveElements =
            static_cast<uint64_t>(this->genericWaveBlockGroups) *
                this->genericWaveGroupStride +
            static_cast<uint64_t>(this->genericWaveRemainder) *
                this->genericWaveStride;
        const uint64_t overflowWaveDataBytes =
            overflowWaveElements * sizeof(uint32_t);
        if (overflowWaveDataBytes > static_cast<uint64_t>(0xFFFFFFFFU) - 31U) {
            return false;
        }
        const uint32_t overflowWaveBytes = overflowWaveElements == 0
            ? 32U
            : AlignUp32(static_cast<uint32_t>(overflowWaveDataBytes));
        const uint32_t allocatedHeadRows =
            this->genericAllocatedHeadRows;
        const uint64_t fixedBaseBytes =
            static_cast<uint64_t>(allocatedHeadRows) *
                this->genericHeadStrideBytes +
            AlignUp32(this->indexSize * sizeof(int32_t)) + firstWaveBytes +
            overflowWaveBytes + kGenericWaveValueElements * sizeof(T);
        const uint64_t sourceRowBytes =
            static_cast<uint64_t>(this->genericWaveSourceStride) * sizeof(T);
        if (sourceRowBytes == 0) {
            return false;
        }
        const uint32_t maxValueRows =
            kGenericWaveValueElements / this->genericWaveValueStride;
        uint64_t maxBatch = 0;
        while (this->genericOuterGroup > 0) {
            maxBatch = 0;
            this->genericWaveGatherElements =
                this->genericOuterGroup * this->genericWaveGroupStride;
            const uint64_t gatheredWaveBytes =
                static_cast<uint64_t>(this->genericWaveGatherElements) *
                    sizeof(T);
            const uint64_t fixedBytes = fixedBaseBytes + gatheredWaveBytes;
            if (fixedBytes < kGenericWaveTotalBytes) {
                maxBatch =
                    (kGenericWaveTotalBytes - fixedBytes) / sourceRowBytes;
                if (maxBatch > maxValueRows) {
                    maxBatch = maxValueRows;
                }
                if (maxBatch > kWaveOuterBatchMax) {
                    maxBatch = kWaveOuterBatchMax;
                }
                if (maxBatch > this->outerSize) {
                    maxBatch = this->outerSize;
                }
                if (maxBatch >= this->genericOuterGroup ||
                    this->genericOuterGroup == 1) {
                    break;
                }
            }
            --this->genericOuterGroup;
        }
        if (this->genericOuterGroup == 0 || maxBatch == 0) {
            return false;
        }
        this->genericWaveOuterBatch = static_cast<uint32_t>(maxBatch);

        this->pipe.InitBuffer(this->waveOverflowOffsetBuf, overflowWaveBytes);
        if (overflowWaveElements == 0) {
            return true;
        }
        LocalTensor<uint32_t> overflowWaveOffsetLocal =
            this->waveOverflowOffsetBuf.template Get<uint32_t>();
        Duplicate(overflowWaveOffsetLocal, zeroByteOffset,
            static_cast<uint32_t>(overflowWaveElements));
        const uint32_t blockWaveLimit = this->genericWaveBlockGroups * 8;
        PipeBarrier<PIPE_ALL>();
        for (uint32_t overflow = 0; overflow < overflowCount; ++overflow) {
            const uint32_t packed =
                static_cast<uint32_t>(targetLocal.GetValue(overflow));
            const uint32_t targetDim = packed & 0x3FFU;
            const uint32_t sourceIndex = (packed >> 10) & 0x3FFFU;
            const uint32_t wave = packed >> 24;
            uint32_t waveOffset;
            if (wave < blockWaveLimit) {
                const uint32_t group = wave >> 3;
                const uint32_t lane = wave & 7U;
                waveOffset = group * this->genericWaveGroupStride + targetDim * 8 + lane;
            } else {
                const uint32_t remainderWave = wave - blockWaveLimit;
                waveOffset = this->genericWaveBlockGroups * this->genericWaveGroupStride +
                    remainderWave * this->genericWaveStride + targetDim;
            }
            overflowWaveOffsetLocal.SetValue(waveOffset, sourceIndex * sizeof(T));
        }
        return true;
    }

    __aicore__ inline void ProcessSharedPackedFallback()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter =
            this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter +
            (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd =
            outerStart + baseOuter +
            (blockIdx < extraOuter ? 1U : 0U);
        LocalTensor<T> valueLocal =
            this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal =
            this->sourceBuf.template Get<T>();
        DataCopyPadExtParams<T> valuePadParams{
            false, 0, 0, static_cast<T>(0)};
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements),
            static_cast<T>(0)};
        DataCopyExtParams valueParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};
        DataCopyExtParams sourceParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(T)),
            0, 0, 0};

        for (uint32_t outer = outerStart;
             outer < outerEnd; ++outer) {
            DataCopyPad(valueLocal,
                this->selfGm[outer * this->dimSize],
                valueParams, valuePadParams);
            DataCopyPad(sourceLocal,
                this->sourceGm[outer * this->indexSize],
                sourceParams, sourcePadParams);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
                T value = valueLocal.GetValue(dim);
                for (uint32_t core = 0;
                     core < blockNum; ++core) {
                    const uint32_t localCount =
                        static_cast<uint32_t>(
                            this->sharedPlanGm.GetValue(
                                this->sharedLocalCountBase +
                                core * this->sharedAlignedDim + dim));
                    const uint32_t packedStart =
                        static_cast<uint32_t>(
                            this->sharedPlanGm.GetValue(
                                this->sharedCoreStartBase +
                                core * this->sharedAlignedDim + dim));
                    for (uint32_t packed = 0;
                         packed < localCount; ++packed) {
                        const uint32_t sourceIndex =
                            static_cast<uint32_t>(
                                this->sharedPlanGm.GetValue(
                                    this->sharedPackedBase +
                                    core * this->sharedLocalPackedCapacity +
                                    packedStart + packed));
                        value = AddValue(
                            value,
                            sourceLocal.GetValue(sourceIndex));
                    }
                }
                valueLocal.SetValue(dim, value);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(
                this->outputGm[outer * this->dimSize],
                valueLocal, valueParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline bool CanPipelineGenericWaveRows() const
    {
        return this->useParallelSharedPlan && this->genericSplitWave &&
            this->genericWaveOuterBatch >= 2U &&
            2U * this->genericWaveValueStride <=
                kParallelSharedWaveValueElements;
    }

    __aicore__ inline void ProcessPipelinedGenericWaveRows()
    {
        if (UseParallelSharedPlan((T*)nullptr)) {
            this->sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t outerStart = blockNum - 1U - blockIdx;
        const uint32_t outerEnd = this->outerSize;
        if (outerStart >= outerEnd) {
            return;
        }

        LocalTensor<T> valueLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal =
            this->waveValueBuf.template Get<T>();
        LocalTensor<T> reduceLocal = this->headBuf.template Get<T>();
        LocalTensor<uint32_t> firstWaveOffsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        LocalTensor<uint32_t> overflowWaveOffsetLocal =
            this->waveOverflowOffsetBuf.template Get<uint32_t>();
        DataCopyPadExtParams<T> padParams{
            false, 0, 0, static_cast<T>(0)};
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements),
            static_cast<T>(0)};
        DataCopyExtParams selfParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams sourceParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams outParams{1,
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

        const uint32_t valueBufferStride = this->genericWaveValueStride;
        const uint32_t sourceBufferStride = this->genericWaveSourceStride;
        uint32_t currentOuter = outerStart;
        uint32_t currentBuffer = 0;
        LocalTensor<T> initialValueLocal = valueLocal;
        LocalTensor<T> initialSourceLocal = sourceLocal;
        if (this->genericWaveIncludeSelf) {
            DataCopyPad(initialSourceLocal[this->genericSelfElement],
                this->selfGm[currentOuter * this->dimSize],
                selfParams, padParams);
        } else {
            DataCopyPad(initialValueLocal,
                this->selfGm[currentOuter * this->dimSize],
                selfParams, padParams);
        }
        DataCopyPad(initialSourceLocal,
            this->sourceGm[currentOuter * this->indexSize],
            sourceParams, sourcePadParams);
        SetFlag<HardEvent::MTE2_V>(0);

        bool outputPending0 = false;
        bool outputPending1 = false;
        while (currentOuter < outerEnd) {
            const uint32_t nextOuter = currentOuter + blockNum;
            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextOuter < outerEnd) {
                if (nextBuffer == 0U && outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    outputPending0 = false;
                } else if (nextBuffer == 1U && outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                    outputPending1 = false;
                }
                LocalTensor<T> nextValueLocal =
                    valueLocal[nextBuffer * valueBufferStride];
                LocalTensor<T> nextSourceLocal =
                    sourceLocal[nextBuffer * sourceBufferStride];
                if (this->genericWaveIncludeSelf) {
                    DataCopyPad(
                        nextSourceLocal[this->genericSelfElement],
                        this->selfGm[nextOuter * this->dimSize],
                        selfParams, padParams);
                } else {
                    DataCopyPad(nextValueLocal,
                        this->selfGm[nextOuter * this->dimSize],
                        selfParams, padParams);
                }
                DataCopyPad(nextSourceLocal,
                    this->sourceGm[nextOuter * this->indexSize],
                    sourceParams, sourcePadParams);
                if (nextBuffer == 0U) {
                    SetFlag<HardEvent::MTE2_V>(0);
                } else {
                    SetFlag<HardEvent::MTE2_V>(1);
                }
            }

            if (currentBuffer == 0U) {
                WaitFlag<HardEvent::MTE2_V>(0);
            } else {
                WaitFlag<HardEvent::MTE2_V>(1);
            }
            LocalTensor<T> currentValueLocal =
                valueLocal[currentBuffer * valueBufferStride];
            LocalTensor<T> currentSourceLocal =
                sourceLocal[currentBuffer * sourceBufferStride];
            AccumulateGenericSplitWave(currentValueLocal,
                currentSourceLocal, waveValueLocal,
                firstWaveOffsetLocal, overflowWaveOffsetLocal,
                reduceLocal, this->dimSize, this->genericWaveStride,
                this->genericWaveBlockGroups,
                this->genericWaveRemainder,
                this->genericWaveGroupStride,
                this->genericWaveIncludeSelf, (T*)nullptr);

            if (currentBuffer == 0U) {
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
            } else {
                SetFlag<HardEvent::V_MTE3>(1);
                WaitFlag<HardEvent::V_MTE3>(1);
            }
            DataCopyPad(
                this->outputGm[currentOuter * this->dimSize],
                currentValueLocal, outParams);
            if (currentBuffer == 0U) {
                SetFlag<HardEvent::MTE3_MTE2>(0);
                outputPending0 = true;
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(1);
                outputPending1 = true;
            }
            currentOuter = nextOuter;
            if (currentOuter < outerEnd) {
                currentBuffer = nextBuffer;
            }
        }
        if (outputPending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        if (outputPending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
    }

    __aicore__ inline void ProcessGenericWave()
    {
        if (this->CanPipelineGenericWaveRows()) {
            this->ProcessPipelinedGenericWaveRows();
            return;
        }
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter = this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter + (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd =
            outerStart + baseOuter + (blockIdx < extraOuter ? 1U : 0U);

        LocalTensor<T> valueLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal = this->waveValueBuf.template Get<T>();
        LocalTensor<T> reduceLocal = this->headBuf.template Get<T>();
        LocalTensor<uint32_t> waveOffsetLocal = this->waveOffsetBuf.template Get<uint32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements), static_cast<T>(0)};

        for (uint32_t outerBase = outerStart; outerBase < outerEnd;
             outerBase += this->genericWaveOuterBatch) {
            uint32_t outerCount = outerEnd - outerBase;
            if (outerCount > this->genericWaveOuterBatch) {
                outerCount = this->genericWaveOuterBatch;
            }
            const uint32_t sourceDstStride = this->genericWaveIncludeSelf
                ? this->genericWaveValueStride / alignElements
                : 0U;
            const uint32_t selfDstStride = this->genericWaveIncludeSelf
                ? (this->genericWaveSourceStride - this->genericWaveValueStride) /
                    alignElements
                : 0U;
            DataCopyExtParams selfParams{static_cast<uint16_t>(outerCount),
                this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, selfDstStride, 0};
            DataCopyExtParams sourceParams{static_cast<uint16_t>(outerCount),
                this->indexSize * static_cast<uint32_t>(sizeof(T)), 0,
                sourceDstStride, 0};
            DataCopyExtParams outParams{static_cast<uint16_t>(outerCount),
                this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            if (this->genericWaveIncludeSelf) {
                DataCopyPad(sourceLocal[this->genericSelfElement],
                    this->selfGm[outerBase * this->dimSize], selfParams, padParams);
            } else {
                DataCopyPad(valueLocal, this->selfGm[outerBase * this->dimSize],
                    selfParams, padParams);
            }
            DataCopyPad(sourceLocal, this->sourceGm[outerBase * this->indexSize],
                sourceParams, sourcePadParams);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            uint32_t localOuter = 0;
            while (localOuter < outerCount) {
                LocalTensor<T> valueSlice =
                    valueLocal[localOuter * this->genericWaveValueStride];
                LocalTensor<T> sourceSlice =
                    sourceLocal[localOuter * this->genericWaveSourceStride];
                if (this->genericSplitWave && this->genericOuterGroup >= 3 &&
                    localOuter + 2 < outerCount) {
                    LocalTensor<T> secondSourceSlice = sourceLocal[
                        (localOuter + 1) * this->genericWaveSourceStride];
                    LocalTensor<T> thirdSourceSlice = sourceLocal[
                        (localOuter + 2) * this->genericWaveSourceStride];
                    LocalTensor<uint32_t> overflowWaveOffsetLocal =
                        this->waveOverflowOffsetBuf.template Get<uint32_t>();
                    AccumulateGenericSplitWaveTriple(valueSlice, sourceSlice,
                        secondSourceSlice, thirdSourceSlice, waveValueLocal,
                        waveOffsetLocal, overflowWaveOffsetLocal, reduceLocal,
                        this->dimSize, this->genericWaveStride,
                        this->genericWaveBlockGroups,
                        this->genericWaveRemainder,
                        this->genericWaveGroupStride,
                        this->genericWaveIncludeSelf, (T*)nullptr);
                    localOuter += 3;
                    continue;
                }
                if (this->genericSplitWave && this->genericOuterGroup >= 2 &&
                    localOuter + 1 < outerCount) {
                    LocalTensor<T> secondSourceSlice = sourceLocal[
                        (localOuter + 1) * this->genericWaveSourceStride];
                    LocalTensor<uint32_t> overflowWaveOffsetLocal =
                        this->waveOverflowOffsetBuf.template Get<uint32_t>();
                    AccumulateGenericSplitWavePair(valueSlice, sourceSlice,
                        secondSourceSlice, waveValueLocal, waveOffsetLocal,
                        overflowWaveOffsetLocal, reduceLocal, this->dimSize,
                        this->genericWaveStride,
                        this->genericWaveBlockGroups,
                        this->genericWaveRemainder,
                        this->genericWaveGroupStride,
                        this->genericWaveIncludeSelf, (T*)nullptr);
                    localOuter += 2;
                    continue;
                }
                if (this->genericSplitWave) {
                    LocalTensor<uint32_t> overflowWaveOffsetLocal =
                        this->waveOverflowOffsetBuf.template Get<uint32_t>();
                    AccumulateGenericSplitWave(valueSlice, sourceSlice, waveValueLocal,
                        waveOffsetLocal, overflowWaveOffsetLocal, reduceLocal,
                        this->dimSize, this->genericWaveStride,
                        this->genericWaveBlockGroups, this->genericWaveRemainder,
                        this->genericWaveGroupStride, this->genericWaveIncludeSelf,
                        (T*)nullptr);
                } else {
                    AccumulateGenericWave(valueSlice, sourceSlice, waveValueLocal,
                        waveOffsetLocal, reduceLocal, this->dimSize,
                        this->genericWaveCount, this->genericWaveStride,
                        this->genericWaveBlockGroups, this->genericWaveRemainder,
                        this->genericWaveGroupStride, this->genericWaveIncludeSelf,
                        (T*)nullptr);
                }
                ++localOuter;
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(this->outputGm[outerBase * this->dimSize], valueLocal, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline bool CanUseGroupedContiguous() const
    {
        if (this->innerSize != 1 || this->outerSize < 32 || this->indexSize < 2048 ||
            this->totalSize < 4096) {
            return false;
        }
        constexpr uint32_t maxChainElements = kGenericChainBufferBytes / sizeof(int32_t);
        return this->dimSize <= maxChainElements &&
            this->indexSize <= maxChainElements - this->dimSize;
    }

    __aicore__ inline bool CanCacheGroupedSource() const
    {
        const uint32_t accumBytes = AlignUp32(this->dimSize * sizeof(T));
        const uint32_t headBytes = AlignUp32(this->dimSize * sizeof(int32_t));
        const uint32_t nextBytes = AlignUp32(this->indexSize * sizeof(int32_t));
        const uint32_t sourceBytes = AlignUp32(this->indexSize * sizeof(T));
        return accumBytes + headBytes + nextBytes + sourceBytes <= kGenericGroupedTotalBytes;
    }

    __aicore__ inline bool CanUseGenericOuterBatch() const
    {
        return this->innerSize == 1 && this->outerSize >= 32 &&
            this->dimSize > 512 && this->dimSize < 1024 && this->indexSize >= 2048;
    }

    __aicore__ inline void ProcessGenericOuterBatch(uint32_t dimStart, uint32_t dimEnd)
    {
        for (uint32_t targetDim = dimStart; targetDim < dimEnd; ++targetDim) {
            uint32_t outerStart = 0;
            while (outerStart < this->outerSize) {
                uint32_t outerCount = this->outerSize - outerStart;
                if (outerCount > kGenericOuterBatch) {
                    outerCount = kGenericOuterBatch;
                }
                this->ComputeGenericOuterBatch(outerStart, outerCount, targetDim);
                outerStart += outerCount;
            }
        }
    }

    __aicore__ inline void ComputeGenericOuterBatch(uint32_t outerStart, uint32_t outerCount,
        uint32_t targetDim)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t localElements = outerCount * alignElements;
        LocalTensor<T> valueLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();

        const uint32_t outputOffset = outerStart * this->dimSize + targetDim;
        DataCopyExtParams selfParams{static_cast<uint16_t>(outerCount),
            static_cast<uint32_t>(sizeof(T)), (this->dimSize - 1) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(valueLocal, this->selfGm[outputOffset], selfParams, padParams);

        PipeBarrier<PIPE_ALL>();

        int32_t sourceIndex = headLocal.GetValue(targetDim);
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = outerStart * this->indexSize + i;
            DataCopyExtParams sourceParams{static_cast<uint16_t>(outerCount),
                static_cast<uint32_t>(sizeof(T)),
                (this->indexSize - 1) * static_cast<uint32_t>(sizeof(T)), 0, 0};
            DataCopyPad(sourceLocal, this->sourceGm[sourceOffset], sourceParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Add(valueLocal, valueLocal, sourceLocal, localElements);
            PipeBarrier<PIPE_ALL>();
            sourceIndex = nextLocal.GetValue(i);
        }

        DataCopyExtParams outParams{static_cast<uint16_t>(outerCount),
            static_cast<uint32_t>(sizeof(T)), 0,
            (this->dimSize - 1) * static_cast<uint32_t>(sizeof(T)), 0};
        DataCopyPad(this->outputGm[outputOffset], valueLocal, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t CalcGroupedVectorRows(uint32_t vectorInner) const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        if (this->innerSize < alignElements || this->outerSize >= 32 ||
            this->dimSize > 65535 || this->indexSize < 512 ||
            this->indexSize >= 2048 || this->totalSize < 4096) {
            return 0;
        }
        const uint64_t fixedBytes =
            static_cast<uint64_t>(this->dimSize) * vectorInner * sizeof(T) +
            AlignUp32(this->indexSize * sizeof(int32_t));
        const uint64_t sourceRowBytes = static_cast<uint64_t>(vectorInner) * sizeof(T);
        if (fixedBytes >= kGenericVectorTotalBytes || sourceRowBytes == 0) {
            return 0;
        }
        uint32_t rows = static_cast<uint32_t>(
            (kGenericVectorTotalBytes - fixedBytes) / (2 * sourceRowBytes));
        if (rows > kGenericVectorMaxSourceRows) {
            rows = kGenericVectorMaxSourceRows;
        }
        if (rows > this->indexSize) {
            rows = this->indexSize;
        }
        return rows >= 2 ? rows : 0;
    }

    __aicore__ inline uint32_t CalcTargetOuterBatchInner() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        if (this->innerSize < alignElements || this->outerSize == 0 ||
            this->outerSize >= 32 || this->indexSize < 512 ||
            this->indexSize >= 2048 || this->totalSize < 4096) {
            return 0;
        }

        const uint64_t fixedBytes =
            static_cast<uint64_t>(AlignUp32(this->dimSize * sizeof(int32_t))) +
            AlignUp32(this->indexSize * sizeof(int32_t)) +
            2ULL * kVectorTileElements * sizeof(T);
        if (fixedBytes >= kGenericVectorTotalBytes) {
            return 0;
        }

        uint32_t rowStrideLimit = kVectorTileElements / this->outerSize;
        rowStrideLimit = (rowStrideLimit / alignElements) * alignElements;
        if (rowStrideLimit < alignElements) {
            return 0;
        }
        if (rowStrideLimit > kGenericVectorMaxInner) {
            rowStrideLimit = kGenericVectorMaxInner;
        }
        return rowStrideLimit;
    }

    __aicore__ inline void ProcessVectorTargetOuterBatch(uint32_t innerTile)
    {
        const uint32_t innerTaskCount = (this->innerSize + innerTile - 1) / innerTile;
        const uint32_t taskCount = this->dimSize * innerTaskCount;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t targetDim = task / innerTaskCount;
            const uint32_t innerStart = (task - targetDim * innerTaskCount) * innerTile;
            uint32_t count = this->innerSize - innerStart;
            if (count > innerTile) {
                count = innerTile;
            }
            this->ComputeVectorTargetOuterBatch(targetDim, innerStart, count);
        }
    }

    __aicore__ inline void ComputeVectorTargetOuterBatch(uint32_t targetDim, uint32_t innerStart,
        uint32_t count)
    {
        const uint32_t countBytes = count * sizeof(T);
        const uint32_t rowStride = AlignUp32(countBytes) / sizeof(T);
        const uint32_t localElements = this->outerSize * rowStride;
        LocalTensor<T> valueLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        const uint32_t selfBase = targetDim * this->innerSize + innerStart;
        DataCopyExtParams selfParams{static_cast<uint16_t>(this->outerSize), countBytes,
            (this->dimSize * this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPad(valueLocal, this->selfGm[selfBase], selfParams, padParams);
        PipeBarrier<PIPE_ALL>();

        int32_t sourceIndex = headLocal.GetValue(targetDim);
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceBase = i * this->innerSize + innerStart;
            DataCopyExtParams sourceParams{static_cast<uint16_t>(this->outerSize), countBytes,
                (this->indexSize * this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
            DataCopyPad(sourceLocal, this->sourceGm[sourceBase], sourceParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Add(valueLocal, valueLocal, sourceLocal, localElements);
            PipeBarrier<PIPE_V>();
            sourceIndex = nextLocal.GetValue(i);
        }

        DataCopyExtParams outParams{static_cast<uint16_t>(this->outerSize), countBytes, 0,
            (this->dimSize * this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0};
        DataCopyPad(this->outputGm[selfBase], valueLocal, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareGroupedVectorIndex()
    {
        LocalTensor<int32_t> targetLocal = this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, padParams);
    }

    __aicore__ inline void ProcessGroupedVector(uint32_t vectorInner, uint32_t groupRows)
    {
        const uint32_t tasksPerOuter = (this->innerSize + vectorInner - 1) / vectorInner;
        const uint32_t taskCount = this->outerSize * tasksPerOuter;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);
        if (taskStart >= taskEnd) {
            PipeBarrier<PIPE_ALL>();
            return;
        }

        bool firstTask = true;
        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t outer = task / tasksPerOuter;
            const uint32_t innerStart = (task - outer * tasksPerOuter) * vectorInner;
            uint32_t count = this->innerSize - innerStart;
            if (count > vectorInner) {
                count = vectorInner;
            }
            this->ComputeGroupedVectorTile(outer, innerStart, count, groupRows, firstTask);
            firstTask = false;
        }
    }

    __aicore__ inline void ComputeGroupedVectorTile(uint32_t outer, uint32_t innerStart,
        uint32_t count, uint32_t groupRows, bool waitForIndex)
    {
        const uint32_t countBytes = count * sizeof(T);
        const uint32_t rowStride = AlignUp32(countBytes) / sizeof(T);
        LocalTensor<T> valueLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.template Get<int32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        const BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, 8};
        constexpr uint32_t elementsPerRepeat = 256 / sizeof(T);
        const uint32_t rowStrideBlocks = rowStride * sizeof(T) / 32;
        const bool useRepeatPairs = UseInt32GroupedRepeat((T*)nullptr) &&
            count == rowStride && count <= elementsPerRepeat &&
            rowStrideBlocks > 0 && rowStrideBlocks <= 255;

        const uint32_t selfBase = outer * this->dimSize * this->innerSize + innerStart;
        DataCopyExtParams selfParams{static_cast<uint16_t>(this->dimSize), countBytes,
            (this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPad(valueLocal, this->selfGm[selfBase], selfParams, padParams);

        const uint32_t sourceOuterBase = outer * this->indexSize * this->innerSize;
        const uint32_t sourceBufferStride = groupRows * rowStride;
        uint32_t indexStart = 0;
        uint32_t groupCount = this->indexSize;
        if (groupCount > groupRows) {
            groupCount = groupRows;
        }
        DataCopyExtParams firstSourceParams{static_cast<uint16_t>(groupCount), countBytes,
            (this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPad(sourceLocal, this->sourceGm[sourceOuterBase + innerStart],
            firstSourceParams, padParams);
        if (waitForIndex) {
            PipeBarrier<PIPE_ALL>();
        } else {
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }
        if (useRepeatPairs) {
            SetMaskNorm();
            const uint64_t pairMask = count == elementsPerRepeat
                ? 0xFFFFFFFFFFFFFFFFULL
                : (1ULL << count) - 1ULL;
            SetVectorMask<T, MaskMode::NORMAL>(0, pairMask);
        } else {
            SetMaskCount();
            SetVectorMask<T, MaskMode::COUNTER>(count);
        }

        uint32_t currentBuffer = 0;
        while (indexStart < this->indexSize) {
            const uint32_t nextStart = indexStart + groupCount;
            const bool hasNext = nextStart < this->indexSize;
            uint32_t nextCount = 0;
            uint32_t nextBuffer = currentBuffer ^ 1U;
            if (hasNext) {
                nextCount = this->indexSize - nextStart;
                if (nextCount > groupRows) {
                    nextCount = groupRows;
                }
                const uint32_t nextSourceOffset =
                    sourceOuterBase + nextStart * this->innerSize + innerStart;
                DataCopyExtParams nextSourceParams{static_cast<uint16_t>(nextCount), countBytes,
                    (this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
                DataCopyPad(sourceLocal[nextBuffer * sourceBufferStride],
                    this->sourceGm[nextSourceOffset], nextSourceParams, padParams);
                SetFlag<HardEvent::MTE2_V>(0);
            }
            LocalTensor<T> currentSource =
                sourceLocal[currentBuffer * sourceBufferStride];
            if (useRepeatPairs) {
                constexpr uint32_t pairLookahead = 8;
                uint32_t pairedMask = 0;
                for (uint32_t slot = 0; slot < groupCount; ++slot) {
                    const uint32_t slotBit = 1U << slot;
                    if ((pairedMask & slotBit) != 0) {
                        continue;
                    }
                    const uint32_t target = static_cast<uint32_t>(
                        targetLocal.GetValue(indexStart + slot));
                    uint32_t partner = groupCount;
                    uint32_t partnerTarget = 0;
                    uint32_t dstRepeatStride = 0;
                    uint32_t sourceRepeatStride = 0;
                    uint32_t candidateEnd = slot + 1 + pairLookahead;
                    if (candidateEnd > groupCount) {
                        candidateEnd = groupCount;
                    }
                    for (uint32_t candidate = slot + 1;
                         candidate < candidateEnd; ++candidate) {
                        if ((pairedMask & (1U << candidate)) != 0) {
                            continue;
                        }
                        const uint32_t candidateSourceStride =
                            (candidate - slot) * rowStrideBlocks;
                        if (candidateSourceStride > 255) {
                            break;
                        }
                        const uint32_t candidateTarget =
                            static_cast<uint32_t>(
                                targetLocal.GetValue(indexStart + candidate));
                        if (candidateTarget < target) {
                            continue;
                        }
                        const uint32_t candidateDstStride =
                            (candidateTarget - target) * rowStrideBlocks;
                        if (candidateDstStride > 255) {
                            continue;
                        }
                        partner = candidate;
                        partnerTarget = candidateTarget;
                        dstRepeatStride = candidateDstStride;
                        sourceRepeatStride = candidateSourceStride;
                        break;
                    }

                    if (partner < groupCount) {
                        if (partnerTarget == target) {
                            const BinaryRepeatParams pairParams{1, 1, 1, 0,
                                static_cast<uint8_t>(sourceRepeatStride), 0};
                            Add<T, false>(valueLocal[target * rowStride],
                                currentSource[slot * rowStride],
                                valueLocal[target * rowStride], MASK_PLACEHOLDER,
                                2, pairParams);
                        } else {
                            const BinaryRepeatParams pairParams{1, 1, 1,
                                static_cast<uint8_t>(dstRepeatStride),
                                static_cast<uint8_t>(dstRepeatStride),
                                static_cast<uint8_t>(sourceRepeatStride)};
                            Add<T, false>(valueLocal[target * rowStride],
                                valueLocal[target * rowStride],
                                currentSource[slot * rowStride], MASK_PLACEHOLDER,
                                2, pairParams);
                        }
                        pairedMask |= 1U << partner;
                        continue;
                    }

                    Add<T, false>(valueLocal[target * rowStride],
                        valueLocal[target * rowStride],
                        currentSource[slot * rowStride], MASK_PLACEHOLDER, 1,
                        repeatParams);
                }
            } else {
                for (uint32_t slot = 0; slot < groupCount; ++slot) {
                    const uint32_t target =
                        static_cast<uint32_t>(targetLocal.GetValue(indexStart + slot));
                    Add<T, false>(valueLocal[target * rowStride],
                        valueLocal[target * rowStride],
                        currentSource[slot * rowStride], MASK_PLACEHOLDER, 1,
                        repeatParams);
                }
            }
            if (hasNext) {
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
            }
            indexStart = nextStart;
            groupCount = nextCount;
            currentBuffer = nextBuffer;
        }
        SetMaskNorm();
        ResetMask();

        DataCopyExtParams outParams{static_cast<uint16_t>(this->dimSize), countBytes, 0,
            (this->innerSize - count) * static_cast<uint32_t>(sizeof(T)), 0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(this->outputGm[selfBase], valueLocal, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline uint32_t CalcVectorInner() const
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        if (this->totalSize < kGenericVectorMinTotal || this->innerSize < alignElements) {
            return 0;
        }

        const uint32_t maxElements = kGenericVectorAccumBytes / sizeof(T);
        uint32_t vectorInner = maxElements / this->dimSize;
        if (vectorInner > kGenericVectorMaxInner) {
            vectorInner = kGenericVectorMaxInner;
        }
        vectorInner = (vectorInner / alignElements) * alignElements;
        if (vectorInner < alignElements) {
            return 0;
        }
        if (this->innerSize < vectorInner) {
            vectorInner = ((this->innerSize + alignElements - 1) / alignElements) * alignElements;
        }
        const uint32_t blockNum = GetBlockNum();
        if (this->outerSize < blockNum && this->indexSize >= 512 && this->indexSize < 2048 &&
            this->totalSize >= 4096) {
            uint32_t targetInner =
                static_cast<uint32_t>((static_cast<uint64_t>(this->innerSize) * this->outerSize +
                    blockNum - 1) / blockNum);
            targetInner = ((targetInner + alignElements - 1) / alignElements) * alignElements;
            if (targetInner < alignElements) {
                targetInner = alignElements;
            }
            if (targetInner < vectorInner) {
                vectorInner = targetInner;
            }
        }
        return vectorInner;
    }

    __aicore__ inline void ProcessVector(uint32_t vectorInner)
    {
        const uint32_t tasksPerOuter = (this->innerSize + vectorInner - 1) / vectorInner;
        const uint32_t taskCount = this->outerSize * tasksPerOuter;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t start = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t end = start + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outerIndex = task / tasksPerOuter;
            const uint32_t innerStart = (task - outerIndex * tasksPerOuter) * vectorInner;
            uint32_t count = this->innerSize - innerStart;
            if (count > vectorInner) {
                count = vectorInner;
            }
            this->ComputeVectorTile(outerIndex, innerStart, count, vectorInner);
        }
    }

    __aicore__ inline void ComputeVectorTile(uint32_t outerIndex, uint32_t innerStart,
        uint32_t count, uint32_t vectorInner)
    {
        LocalTensor<T> accumLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        DataCopyExtParams inParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        const uint32_t selfOuterBase = outerIndex * this->dimSize * this->innerSize;
        for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
            const uint32_t selfOffset = selfOuterBase + dim * this->innerSize + innerStart;
            DataCopyPad(accumLocal[dim * vectorInner], this->selfGm[selfOffset], inParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        const uint32_t sourceOuterBase = outerIndex * this->indexSize * this->innerSize;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t targetDim = this->indexGm.GetValue(i);
            if (targetDim < 0) {
                targetDim += static_cast<int32_t>(this->dimSize);
            }
            if (targetDim < 0 || targetDim >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }

            const uint32_t sourceOffset = sourceOuterBase + i * this->innerSize + innerStart;
            DataCopyPad(sourceLocal, this->sourceGm[sourceOffset], inParams, padParams);
            PipeBarrier<PIPE_ALL>();
            const uint32_t localOffset = static_cast<uint32_t>(targetDim) * vectorInner;
            Add(accumLocal[localOffset], accumLocal[localOffset], sourceLocal, count);
            PipeBarrier<PIPE_V>();
        }

        DataCopyExtParams outParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        for (uint32_t dim = 0; dim < this->dimSize; ++dim) {
            const uint32_t outputOffset = selfOuterBase + dim * this->innerSize + innerStart;
            DataCopyPad(this->outputGm[outputOffset], accumLocal[dim * vectorInner], outParams);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessByOutput()
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockLength = (this->totalSize + blockNum - 1) / blockNum;
        const uint32_t start = GetBlockIdx() * blockLength;
        uint32_t end = start + blockLength;
        if (end > this->totalSize) {
            end = this->totalSize;
        }
        if (start >= end) {
            return;
        }

        for (uint32_t outOffset = start; outOffset < end; ++outOffset) {
            this->ComputeOne(outOffset);
        }
    }

    __aicore__ inline void ComputeOne(uint32_t outOffset)
    {
        const uint32_t dimArea = this->dimSize * this->innerSize;
        const uint32_t outerIndex = outOffset / dimArea;
        const uint32_t dimRemainder = outOffset - outerIndex * dimArea;
        const uint32_t targetDim = dimRemainder / this->innerSize;
        const uint32_t innerIndex = dimRemainder - targetDim * this->innerSize;

        T value = this->selfGm.GetValue(outOffset);
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t sourceDim = this->indexGm.GetValue(i);
            if (sourceDim < 0) {
                sourceDim += static_cast<int32_t>(this->dimSize);
            }
            if (sourceDim == static_cast<int32_t>(targetDim)) {
                const uint32_t sourceOffset = (outerIndex * this->indexSize + i) * this->innerSize + innerIndex;
                value = AddValue(value, this->sourceGm.GetValue(sourceOffset));
            }
        }
        this->outputGm.SetValue(outOffset, value);
    }

    __aicore__ inline void ComputeLine(uint32_t line)
    {
        if (this->innerSize == 1) {
            if (this->useGroupedContiguous) {
                if (this->useGroupedSourceCache) {
                    this->ComputeContiguousLineGroupedCached(line);
                } else {
                    this->ComputeContiguousLineGrouped(line);
                }
            } else {
                this->ComputeContiguousLine(line);
            }
            return;
        }

        const uint32_t outerIndex = line / this->innerSize;
        const uint32_t innerIndex = line - outerIndex * this->innerSize;
        const uint32_t selfBase = outerIndex * this->dimSize * this->innerSize + innerIndex;
        const uint32_t sourceBase = outerIndex * this->indexSize * this->innerSize + innerIndex;
        LocalTensor<T> accumLocal = this->accumBuf.template Get<T>();

        for (uint32_t dimIndex = 0; dimIndex < this->dimSize; ++dimIndex) {
            const uint32_t outOffset = selfBase + dimIndex * this->innerSize;
            accumLocal.SetValue(dimIndex, this->selfGm.GetValue(outOffset));
        }

        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t targetDim = this->indexGm.GetValue(i);
            if (targetDim < 0) {
                targetDim += static_cast<int32_t>(this->dimSize);
            }
            if (targetDim < 0 || targetDim >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }
            const uint32_t sourceOffset = sourceBase + i * this->innerSize;
            const uint32_t dimIndex = static_cast<uint32_t>(targetDim);
            const T value = AddValue(accumLocal.GetValue(dimIndex), this->sourceGm.GetValue(sourceOffset));
            accumLocal.SetValue(dimIndex, value);
        }

        for (uint32_t dimIndex = 0; dimIndex < this->dimSize; ++dimIndex) {
            const uint32_t outOffset = selfBase + dimIndex * this->innerSize;
            this->outputGm.SetValue(outOffset, accumLocal.GetValue(dimIndex));
        }
    }

    __aicore__ inline void CopyContiguousToLocal(const LocalTensor<T>& local, GlobalTensor<T>& gm,
        uint32_t offset, uint32_t count)
    {
        DataCopyExtParams inParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], inParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyContiguousToGm(GlobalTensor<T>& gm, uint32_t offset,
        const LocalTensor<T>& local, uint32_t count)
    {
        DataCopyExtParams outParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeContiguousLine(uint32_t outerIndex)
    {
        const uint32_t selfBase = outerIndex * this->dimSize;
        const uint32_t sourceBase = outerIndex * this->indexSize;
        LocalTensor<T> accumLocal = this->accumBuf.template Get<T>();

        this->CopyContiguousToLocal(accumLocal, this->selfGm, selfBase, this->dimSize);

        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t targetDim = this->indexGm.GetValue(i);
            if (targetDim < 0) {
                targetDim += static_cast<int32_t>(this->dimSize);
            }
            if (targetDim < 0 || targetDim >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }
            const uint32_t dimIndex = static_cast<uint32_t>(targetDim);
            const T value = AddValue(accumLocal.GetValue(dimIndex), this->sourceGm.GetValue(sourceBase + i));
            accumLocal.SetValue(dimIndex, value);
        }

        this->CopyContiguousToGm(this->outputGm, selfBase, accumLocal, this->dimSize);
    }

    __aicore__ inline void ComputeContiguousLineGrouped(uint32_t outerIndex)
    {
        const uint32_t selfBase = outerIndex * this->dimSize;
        const uint32_t sourceBase = outerIndex * this->indexSize;
        LocalTensor<T> accumLocal = this->accumBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();

        this->CopyContiguousToLocal(accumLocal, this->selfGm, selfBase, this->dimSize);

        for (uint32_t dimIndex = 0; dimIndex < this->dimSize; ++dimIndex) {
            int32_t sourceIndex = headLocal.GetValue(dimIndex);
            if (sourceIndex < 0) {
                continue;
            }

            T value = accumLocal.GetValue(dimIndex);
            while (sourceIndex >= 0) {
                const uint32_t i = static_cast<uint32_t>(sourceIndex);
                value = AddValue(value, this->sourceGm.GetValue(sourceBase + i));
                sourceIndex = nextLocal.GetValue(i);
            }
            accumLocal.SetValue(dimIndex, value);
        }

        this->CopyContiguousToGm(this->outputGm, selfBase, accumLocal, this->dimSize);
    }

    __aicore__ inline void ComputeContiguousLineGroupedCached(uint32_t outerIndex)
    {
        const uint32_t selfBase = outerIndex * this->dimSize;
        const uint32_t sourceBase = outerIndex * this->indexSize;
        LocalTensor<T> accumLocal = this->accumBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();

        this->CopyContiguousToLocal(accumLocal, this->selfGm, selfBase, this->dimSize);
        this->CopyContiguousToLocal(sourceLocal, this->sourceGm, sourceBase, this->indexSize);

        for (uint32_t dimIndex = 0; dimIndex < this->dimSize; ++dimIndex) {
            int32_t sourceIndex = headLocal.GetValue(dimIndex);
            if (sourceIndex < 0) {
                continue;
            }

            T value = accumLocal.GetValue(dimIndex);
            while (sourceIndex >= 0) {
                const uint32_t i = static_cast<uint32_t>(sourceIndex);
                value = AddValue(value, sourceLocal.GetValue(i));
                sourceIndex = nextLocal.GetValue(i);
            }
            accumLocal.SetValue(dimIndex, value);
        }

        this->CopyContiguousToGm(this->outputGm, selfBase, accumLocal, this->dimSize);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> accumBuf;
    TBuf<TPosition::VECCALC> sourceBuf;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    TBuf<TPosition::VECCALC> waveOffsetBuf;
    TBuf<TPosition::VECCALC> waveOverflowOffsetBuf;
    TBuf<TPosition::VECCALC> waveValueBuf;
    GlobalTensor<T> selfGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> sourceGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> workspaceGm;
    GlobalTensor<int32_t> sharedPlanGm;
    GlobalTensor<int32_t> batchedPlanGm;
    uint32_t totalSize = 0;
    uint32_t blockSize = 0;
    uint32_t outerSize = 0;
    uint32_t dimSize = 0;
    uint32_t indexSize = 0;
    uint32_t innerSize = 1;
    uint32_t genericWaveCount = 0;
    uint32_t genericWaveStride = 0;
    uint32_t genericWaveBlockGroups = 0;
    uint32_t genericWaveRemainder = 0;
    uint32_t genericWaveGroupStride = 0;
    uint32_t genericWaveGatherElements = 0;
    uint32_t genericZeroElement = 0;
    uint32_t genericSelfElement = 0;
    uint32_t genericWaveOuterBatch = 1;
    uint32_t genericWaveValueStride = 0;
    uint32_t genericWaveSourceStride = 0;
    uint32_t genericHeadStrideBytes = 0;
    uint32_t genericAllocatedOuterGroup = 1;
    uint32_t genericAllocatedHeadRows = 1;
    uint32_t genericOuterGroup = 1;
    uint32_t sharedAlignedDim = 0;
    uint32_t sharedAlignedIndex = 0;
    uint32_t sharedLocalCountBase = 0;
    uint32_t sharedGlobalCountBase = 0;
    uint32_t sharedCoreStartBase = 0;
    uint32_t sharedPackedBase = 0;
    uint32_t sharedLocalPackedCapacity = 0;
    uint32_t sharedPackedCapacity = 0;
    uint32_t sharedWaveBase = 0;
    uint32_t sharedWaveCapacity = 0;
    uint32_t sharedCachedIndexBase = 0;
    uint32_t sharedCachedFlagBase = 0;
    uint32_t sharedCachedIndexStride = 0;
    uint32_t sharedCachedIndexWords = 0;
    uint32_t sharedNextBytes = 0;
    uint32_t batchedAlignedIndex = 0;
    uint32_t batchedAlignedDim = 0;
    uint32_t batchedAlignedGroupCount = 0;
    uint32_t batchedCachedIndexBase = 0;
    uint32_t batchedPackedNextBase = 0;
    uint32_t batchedGroupHeadBase = 0;
    uint32_t batchedCacheFlagBase = 0;
    uint32_t batchedSharedLocalCountBase = 0;
    uint32_t batchedSharedCoreStartBase = 0;
    uint32_t batchedSharedPackedBase = 0;
    uint32_t batchedSharedWaveBase = 0;
    uint32_t batchedSharedLocalPackedCapacity = 0;
    uint32_t batchedBucketWidth0 = 0;
    uint32_t batchedBucketWidth1 = 0;
    uint32_t batchedBucketWidth2 = 0;
    uint32_t batchedBucketWidth3 = 0;
    uint32_t batchedBucketPlanWords = 0;
    bool batchedUseCountBucketPlan = false;
    bool genericUseBlockReduce = false;
    bool genericWaveIncludeSelf = false;
    bool genericSplitWave = false;
    bool sharedPlanCacheEnabled = false;
    bool useGroupedContiguous = false;
    bool useGroupedSourceCache = false;
    bool useParallelSharedPlan = false;
    bool sharedUsePackedFallback = false;
    bool useBatchedPlanCache = false;
    uint32_t directPaddedAccumBytes = kGenericVectorAccumBytes;
};

class KernelIndexAddInt8 {
public:
    __aicore__ inline KernelIndexAddInt8() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        uint32_t totalSize, uint32_t blockSize, uint32_t outerSize, uint32_t dimSize, uint32_t indexSize,
        uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ int8_t*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ int8_t*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ int8_t*)output, totalSize);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
    }

    __aicore__ inline void Process()
    {
        if (this->totalSize == 0 || this->blockSize == 0 || this->outerSize == 0 ||
            this->dimSize == 0 || this->innerSize == 0) {
            return;
        }

        if (this->CanUseAlignedInt8AtomicRows()) {
            this->pipe.InitBuffer(this->nextBuf,
                AlignUp32(this->indexSize * sizeof(int32_t)));
            this->pipe.InitBuffer(this->rowBuf,
                2U * AlignUp32(this->innerSize * sizeof(int8_t)));
            this->ProcessAlignedInt8AtomicRows();
            return;
        }

        constexpr uint32_t kFallbackTileBytes = 8192;
        this->pipe.InitBuffer(this->rowBuf, kFallbackTileBytes);
        this->pipe.InitBuffer(this->sourceRowBuf, kFallbackTileBytes);
        this->ProcessGenericRows(kFallbackTileBytes);
    }

private:
    __aicore__ inline bool CanUseAlignedInt8AtomicRows() const
    {
        constexpr uint32_t routeUbBudgetBytes =
            kGenericWaveTotalBytes - 8192U;
        const uint32_t indexBytes =
            AlignUp32(this->indexSize * sizeof(int32_t));
        const uint32_t rowBytes = AlignUp32(this->innerSize);
        return this->outerSize == 1U && this->dimSize > 1U &&
            this->indexSize > 0U && this->innerSize >= 32U &&
            this->innerSize % 32U == 0U &&
            indexBytes + 2U * rowBytes <= routeUbBudgetBytes;
    }

    __aicore__ inline void ProcessAlignedInt8AtomicRows()
    {
        LocalTensor<int32_t> targetLocal = this->nextBuf.Get<int32_t>();
        LocalTensor<int8_t> rowLocal = this->rowBuf.Get<int8_t>();
        const uint32_t rowStride = AlignUp32(this->innerSize);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        for (uint32_t targetDim = blockIdx; targetDim < this->dimSize;
             targetDim += blockNum) {
            const uint32_t rowOffset = targetDim * this->innerSize;
            DataCopy(rowLocal, this->selfGm[rowOffset], this->innerSize);
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->outputGm[rowOffset], rowLocal, this->innerSize);
            PipeBarrier<PIPE_ALL>();
        }
        // V473_EARLY_PRIME_BEGIN
        // Local self-copy is complete before rowLocal is reused. Index and
        // source reads do not consume the output initialized by peer cores.
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart =
            blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t localIndexCount =
            baseIndexes + (blockIdx < extraIndexes ? 1U : 0U);
        if (localIndexCount != 0U) {
            DataCopyExtParams indexParams{1,
                localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
                0, 0, 0};
            DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
            DataCopyPad(targetLocal, this->indexGm[indexStart],
                indexParams, indexPadParams);
            PipeBarrier<PIPE_ALL>();
        }

        uint32_t currentIndex = 0U;
        int32_t currentTarget = -1;
        while (currentIndex < localIndexCount) {
            currentTarget = targetLocal.GetValue(currentIndex);
            if (currentTarget < 0) {
                currentTarget += static_cast<int32_t>(this->dimSize);
            }
            if (currentTarget >= 0 &&
                currentTarget < static_cast<int32_t>(this->dimSize)) {
                break;
            }
            ++currentIndex;
        }
        uint32_t currentBuffer = 0U;
        bool pending0 = false;
        bool pending1 = false;
        if (currentIndex < localIndexCount) {
            DataCopy(rowLocal,
                this->sourceGm[(indexStart + currentIndex) * this->innerSize],
                this->innerSize);
        }
        // Every core joins, including empty and all-invalid index slices.
        // Keep the initial readiness token after the hardware rendezvous.
        SyncAll<true>();
        if (currentIndex >= localIndexCount) {
            return;
        }
        // V473_EARLY_PRIME_END
        SetFlag<HardEvent::MTE2_MTE3>(0);

        SetAtomicAdd<int8_t>();
        while (currentIndex < localIndexCount) {
            uint32_t nextIndex = currentIndex + 1U;
            int32_t nextTarget = -1;
            while (nextIndex < localIndexCount) {
                nextTarget = targetLocal.GetValue(nextIndex);
                if (nextTarget < 0) {
                    nextTarget += static_cast<int32_t>(this->dimSize);
                }
                if (nextTarget >= 0 &&
                    nextTarget < static_cast<int32_t>(this->dimSize)) {
                    break;
                }
                ++nextIndex;
            }

            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextIndex < localIndexCount) {
                if (nextBuffer == 0U && pending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    pending0 = false;
                } else if (nextBuffer == 1U && pending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                    pending1 = false;
                }
                DataCopy(rowLocal[nextBuffer * rowStride],
                    this->sourceGm[(indexStart + nextIndex) * this->innerSize],
                    this->innerSize);
                SetFlag<HardEvent::MTE2_MTE3>(nextBuffer);
            }

            WaitFlag<HardEvent::MTE2_MTE3>(currentBuffer);
            const uint32_t rowOffset =
                static_cast<uint32_t>(currentTarget) * this->innerSize;
            DataCopy(this->outputGm[rowOffset],
                rowLocal[currentBuffer * rowStride], this->innerSize);
            SetFlag<HardEvent::MTE3_MTE2>(currentBuffer);
            if (currentBuffer == 0U) {
                pending0 = true;
            } else {
                pending1 = true;
            }

            currentIndex = nextIndex;
            currentTarget = nextTarget;
            currentBuffer = nextBuffer;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
        SetAtomicNone();
    }

    __aicore__ inline void ProcessGenericRows(uint32_t tileBytes)
    {
        LocalTensor<int8_t> rowLocal = this->rowBuf.Get<int8_t>();
        LocalTensor<int8_t> sourceLocal = this->sourceRowBuf.Get<int8_t>();
        const uint32_t rowCount = this->outerSize * this->dimSize;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        for (uint32_t row = blockIdx; row < rowCount; row += blockNum) {
            const uint32_t outerIndex = row / this->dimSize;
            const uint32_t targetDim = row - outerIndex * this->dimSize;
            const uint32_t rowBase = row * this->innerSize;
            for (uint32_t innerStart = 0; innerStart < this->innerSize;
                 innerStart += tileBytes) {
                uint32_t count = this->innerSize - innerStart;
                if (count > tileBytes) {
                    count = tileBytes;
                }
                const uint32_t paddedCount = AlignUp32(count);
                DataCopyExtParams loadParams{1, count, 0, 0, 0};
                DataCopyPadExtParams<int8_t> loadPadParams{
                    paddedCount != count, 0,
                    static_cast<uint8_t>(paddedCount - count),
                    static_cast<int8_t>(0)};
                DataCopyPad(rowLocal, this->selfGm[rowBase + innerStart],
                    loadParams, loadPadParams);
                PipeBarrier<PIPE_ALL>();

                for (uint32_t i = 0; i < this->indexSize; ++i) {
                    int32_t target = this->indexGm.GetValue(i);
                    if (target < 0) {
                        target += static_cast<int32_t>(this->dimSize);
                    }
                    if (target != static_cast<int32_t>(targetDim)) {
                        continue;
                    }
                    const uint32_t sourceOffset =
                        (outerIndex * this->indexSize + i) * this->innerSize +
                        innerStart;
                    DataCopyPad(sourceLocal, this->sourceGm[sourceOffset],
                        loadParams, loadPadParams);
                    PipeBarrier<PIPE_ALL>();
                    for (uint32_t j = 0; j < count; ++j) {
                        rowLocal.SetValue(j,
                            AddValue(rowLocal.GetValue(j), sourceLocal.GetValue(j)));
                    }
                }

                DataCopyExtParams storeParams{1, count, 0, 0, 0};
                DataCopyPad(this->outputGm[rowBase + innerStart], rowLocal,
                    storeParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ComputeOne(uint32_t outOffset)
    {
        const uint32_t dimArea = this->dimSize * this->innerSize;
        const uint32_t outerIndex = outOffset / dimArea;
        const uint32_t dimRemainder = outOffset - outerIndex * dimArea;
        const uint32_t targetDim = dimRemainder / this->innerSize;
        const uint32_t innerIndex = dimRemainder - targetDim * this->innerSize;
        int8_t value = this->selfGm.GetValue(outOffset);
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = this->indexGm.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target != static_cast<int32_t>(targetDim)) {
                continue;
            }
            const uint32_t sourceOffset =
                (outerIndex * this->indexSize + i) * this->innerSize + innerIndex;
            value = AddValue(value, this->sourceGm.GetValue(sourceOffset));
        }
        this->outputGm.SetValue(outOffset, value);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    TBuf<TPosition::VECCALC> rowBuf;
    TBuf<TPosition::VECCALC> sourceRowBuf;
    GlobalTensor<int8_t> selfGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<int8_t> sourceGm;
    GlobalTensor<int8_t> outputGm;
    uint32_t totalSize = 0;
    uint32_t blockSize = 0;
    uint32_t outerSize = 0;
    uint32_t dimSize = 0;
    uint32_t indexSize = 0;
    uint32_t innerSize = 1;
};

template <typename T>
class KernelIndexAddVector {
public:
    __aicore__ inline KernelIndexAddVector() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
        uint32_t dimSize, uint32_t indexSize, uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ T*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ T*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ T*)output, totalSize);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
        this->useBucketPlanCache = innerSize == 1U && outerSize >= 2048U &&
            dimSize >= 64U && dimSize < 512U && indexSize >= 512U &&
            indexSize < 2048U;
        if (this->useBucketPlanCache) {
            this->bucketAlignedIndexWords =
                AlignUp32(indexSize * sizeof(int32_t)) / sizeof(int32_t);
            this->bucketCachedIndexBase = kHalfBucketCacheHeaderWords;
            this->bucketCacheFlagBase = this->bucketCachedIndexBase +
                this->bucketAlignedIndexWords;
            this->bucketCachedPlanBase = this->bucketCacheFlagBase +
                GetBlockNum() * kHalfBucketCacheFlagStride;
            GM_ADDR userWorkspace = GetUserWorkspace(workspace);
            this->bucketPlanGm.SetGlobalBuffer(
                (__gm__ int32_t*)userWorkspace,
                this->bucketCachedPlanBase +
                    kGenericWaveTotalBytes / sizeof(uint32_t));
            // V440_PRIVATE_INIT_BEGIN: each logical core owns its complete slot.
            this->usePrivateBucketCache = sizeof(T) == 2U && dimSize >= 128U &&
                workspace != nullptr && GetBlockNum() > 0U &&
                GetBlockNum() <= 65535U;
            if (this->usePrivateBucketCache) {
                this->privateIndexWords = (indexSize + 63U) & ~63U;
                this->bucketCachedPlanBase = kHalfBucketCacheHeaderWords +
                    this->privateIndexWords;
                const uint64_t slotWords =
                    static_cast<uint64_t>(GetBlockIdx()) * kPrivateBucketSlotWords;
                this->bucketPlanGm.SetGlobalBuffer(
                    (__gm__ int32_t*)userWorkspace + slotWords,
                    kPrivateBucketSlotWords);
            }
            // V440_PRIVATE_INIT_END
        }
    }

    __aicore__ inline void Process()
    {
        if (this->totalSize == 0 || this->blockSize == 0 || this->outerSize == 0 ||
            this->dimSize == 0 || this->innerSize == 0) {
            return;
        }

        this->pipe.InitBuffer(this->headBuf, AlignUp32(this->dimSize * sizeof(int32_t)));
        const uint32_t nextBytes = this->ActualHalfNextBytes(); // V440_PRIVATE_NEXT_ALLOC
        this->pipe.InitBuffer(this->nextBuf, nextBytes);
        if (this->innerSize == 1 && this->CanUseContiguousOuterBatch()) {
            if (this->PrepareContiguousCountBuckets()) {
                this->pipe.InitBuffer(this->sourceBuf,
                    (this->bucketGatherOuterBatch *
                        this->bucketSourceRowStride +
                        this->bucketTotalMatrixElements) * sizeof(T));
                this->pipe.InitBuffer(this->pairSourceBuf,
                    17U * this->bucketSourceRowStride * sizeof(T));
                this->pipe.InitBuffer(this->waveValueBuf,
                    this->bucketGatherOuterBatch * this->bucketValueStride *
                        sizeof(T));
                this->ProcessContiguousCountBuckets();
                return;
            }
            if (this->PrepareContiguousWave()) {
                this->pipe.InitBuffer(this->valueBuf,
                    2U * this->waveOuterBatch * this->waveValueStride * sizeof(T));
                this->pipe.InitBuffer(this->sourceBuf,
                    2U * this->waveOuterBatch * this->waveSourceStride * sizeof(T));
                this->pipe.InitBuffer(this->waveValueBuf,
                    (this->waveCount == 0
                        ? this->waveValueStride
                        : this->waveGatherOuterBatch * this->waveCount *
                            this->waveStride) * sizeof(T));
                this->ProcessContiguousWave();
                return;
            }
            const uint32_t blockNum = GetBlockNum();
            const uint32_t blockIdx = GetBlockIdx();
            const uint32_t baseDims = this->dimSize / blockNum;
            const uint32_t extraDims = this->dimSize - baseDims * blockNum;
            const uint32_t dimStart =
                blockIdx * baseDims + (blockIdx < extraDims ? blockIdx : extraDims);
            const uint32_t dimEnd =
                dimStart + baseDims + (blockIdx < extraDims ? 1U : 0U);
            BuildIndexChainsRangeCached(this->headBuf.template Get<int32_t>(),
                this->nextBuf.template Get<int32_t>(), this->indexGm, this->indexSize,
                this->dimSize, dimStart, dimEnd);
            this->pipe.InitBuffer(this->valueBuf, kVectorTileElements * sizeof(T));
            this->pipe.InitBuffer(this->sourceBuf, kVectorTileElements * sizeof(T));
            this->ProcessContiguousOuterBatch(dimStart, dimEnd);
            return;
        }

        BuildIndexChains(this->headBuf.template Get<int32_t>(), this->nextBuf.template Get<int32_t>(),
            this->indexGm, this->indexSize, this->dimSize);
        this->pipe.InitBuffer(this->valueBuf, kVectorTileElements * sizeof(T));

        if (this->innerSize == 1) {
            this->ProcessContiguousDimMasked();
            return;
        }

        this->pipe.InitBuffer(this->sourceBuf, kVectorTileElements * sizeof(T));
        const uint32_t blockNum = GetBlockNum();
        const uint32_t taskCount = (this->totalSize + kVectorTileElements - 1) / kVectorTileElements;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd = taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);
        if (taskStart >= taskEnd) {
            return;
        }

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t start = task * kVectorTileElements;
            uint32_t length = this->totalSize - start;
            if (length > kVectorTileElements) {
                length = kVectorTileElements;
            }
            this->ComputeTile(start, length);
        }
    }

private:
    // V440_PRIVATE_METHODS_BEGIN
    static constexpr uint32_t kPrivateBucketSlotWords = 65536U / sizeof(int32_t);
    static constexpr uint32_t kPrivateBucketMagicA = 0x49415048U;
    static constexpr uint32_t kPrivateBucketMagicB = 0x43505256U;
    static constexpr uint32_t kPrivateBucketVersion = 2U;

    __aicore__ inline uint32_t ActualHalfNextBytes() const
    {
        if (this->usePrivateBucketCache) {
            return 2U * this->privateIndexWords * sizeof(int32_t);
        }
        return this->useBucketPlanCache
            ? (this->bucketAlignedIndexWords + GetBlockNum() * kHalfBucketCacheFlagStride) * sizeof(int32_t)
            : AlignUp32(this->indexSize * sizeof(int32_t));
    }

    __aicore__ inline bool PrivateBucketLayoutFits() const
    {
        const uint64_t planWords = (static_cast<uint64_t>(this->bucketRestorePlanBase) +
            static_cast<uint64_t>(this->bucketGatherOuterBatch) * this->dimSize + 7ULL) & ~7ULL;
        const uint64_t allocatedPlanWords =
            (static_cast<uint64_t>(this->bucketRestorePlanBase) +
                static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                    this->bucketValueStride + 7ULL) & ~7ULL;
        const uint64_t sourceBytes =
            (static_cast<uint64_t>(this->bucketGatherOuterBatch) * this->bucketSourceRowStride +
                this->bucketTotalMatrixElements + 17ULL * this->bucketSourceRowStride) * sizeof(T);
        const uint64_t outputBytes = static_cast<uint64_t>(this->bucketGatherOuterBatch) *
            this->bucketValueStride * sizeof(T);
        const uint64_t totalBytes = AlignUp32(this->dimSize * sizeof(int32_t)) +
            static_cast<uint64_t>(this->ActualHalfNextBytes()) +
            allocatedPlanWords * sizeof(uint32_t) + sourceBytes + outputBytes;
        return this->usePrivateBucketCache &&
            static_cast<uint64_t>(this->bucketCachedPlanBase) + planWords <= kPrivateBucketSlotWords &&
            this->bucketSourceRowStride / 16U <= 255U &&
            totalBytes + kHalfCountBucketUbSafetyBytes <= kGenericWaveTotalBytes;
    }

    __aicore__ inline void InvalidatePrivateBucketCache()
    {
        LocalTensor<int32_t> headerLocal = this->headBuf.template Get<int32_t>();
        PipeBarrier<PIPE_ALL>();
        Duplicate(headerLocal, static_cast<int32_t>(0), 8U);
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm, headerLocal, 8U);
        PipeBarrier<PIPE_MTE3>();
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
    }

    __aicore__ inline bool RejectPrivateBucketCache()
    {
        this->InvalidatePrivateBucketCache();
        return false;
    }

    __aicore__ inline bool TryLoadPrivateBucketCache()
    {
        LocalTensor<int32_t> headerLocal = this->headBuf.template Get<int32_t>();
        DataCopy(headerLocal, this->bucketPlanGm, kHalfBucketCacheHeaderWords);
        PipeBarrier<PIPE_ALL>();
        const bool headerMatches =
            static_cast<uint32_t>(headerLocal.GetValue(0U)) == kPrivateBucketMagicA &&
            static_cast<uint32_t>(headerLocal.GetValue(1U)) == kPrivateBucketMagicB &&
            static_cast<uint32_t>(headerLocal.GetValue(2U)) == kPrivateBucketVersion &&
            static_cast<uint32_t>(headerLocal.GetValue(3U)) == GetBlockNum() &&
            static_cast<uint32_t>(headerLocal.GetValue(4U)) == this->outerSize &&
            static_cast<uint32_t>(headerLocal.GetValue(5U)) == this->dimSize &&
            static_cast<uint32_t>(headerLocal.GetValue(6U)) == this->indexSize &&
            static_cast<uint32_t>(headerLocal.GetValue(25U)) == GetBlockIdx() &&
            static_cast<uint32_t>(headerLocal.GetValue(26U)) == this->privateIndexWords &&
            static_cast<uint32_t>(headerLocal.GetValue(27U)) == kPrivateBucketSlotWords &&
            static_cast<uint32_t>(headerLocal.GetValue(28U)) == sizeof(T) &&
            static_cast<uint32_t>(headerLocal.GetValue(29U)) == this->ActualHalfNextBytes() &&
            static_cast<uint32_t>(headerLocal.GetValue(30U)) == 1U;
        if (!headerMatches) {
            return this->RejectPrivateBucketCache();
        }
        uint64_t widthSum = 0U;
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount; ++bucket) {
            this->bucketWidth[bucket] = static_cast<uint32_t>(headerLocal.GetValue(8U + bucket));
            this->bucketCap[bucket] = static_cast<uint32_t>(headerLocal.GetValue(16U + bucket));
            if (this->bucketWidth[bucket] > this->dimSize) {
                return this->RejectPrivateBucketCache();
            }
            widthSum += this->bucketWidth[bucket];
        }
        this->bucketGatherOuterBatch = static_cast<uint32_t>(headerLocal.GetValue(24U));
        const bool metadataValid = widthSum == this->dimSize &&
            this->bucketCap[0] == 1U && this->bucketCap[1] == 2U &&
            this->bucketCap[2] == 3U && this->bucketCap[3] == 4U &&
            this->bucketCap[4] == 5U && this->bucketCap[5] == 6U &&
            this->bucketCap[6] == 8U && this->bucketCap[7] > 0U &&
            this->bucketCap[7] <= 255U && this->bucketGatherOuterBatch >= 2U &&
            this->bucketGatherOuterBatch <= kHalfCountBucketGatherOuter &&
            (this->bucketGatherOuterBatch & 1U) == 0U;
        if (!metadataValid) {
            return this->RejectPrivateBucketCache();
        }
        const uint32_t planElements = this->RebuildBucketDerivedMetadata();
        if (static_cast<uint32_t>(headerLocal.GetValue(7U)) != planElements ||
            !this->PrivateBucketLayoutFits()) {
            return this->RejectPrivateBucketCache();
        }

        LocalTensor<int32_t> currentIndexLocal = this->nextBuf.template Get<int32_t>();
        LocalTensor<int32_t> cachedIndexLocal = currentIndexLocal[this->privateIndexWords];
        DataCopyExtParams currentParams{1U, this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> noPad{false, 0U, 0U, 0};
        DataCopyPad(currentIndexLocal, this->indexGm, currentParams, noPad);
        DataCopy(cachedIndexLocal, this->bucketPlanGm[this->bucketCachedIndexBase], this->privateIndexWords);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = this->indexSize; i < this->privateIndexWords; ++i) {
            currentIndexLocal.SetValue(i, 0);
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        // Header metadata has been consumed. This small bitmap reuses headBuf.
        LocalTensor<uint8_t> equalBits = headerLocal.template ReinterpretCast<uint8_t>();
        // V527_BITMAP_REDUCTION_BEGIN: Q<=2048 and D>=128 on this private
        // family. Define every byte outside the true comparison as all-equal.
        Duplicate(headerLocal, static_cast<int32_t>(-1), 64U);
        PipeBarrier<PIPE_V>();
        Compare(equalBits, cachedIndexLocal, currentIndexLocal, CMPMODE::EQ, this->privateIndexWords);
        PipeBarrier<PIPE_V>();
        // The complete raw comparison has consumed nextBuf. Reuse its first
        // 512B for an exact uint8-to-half expansion; all values0..255 are exact.
        LocalTensor<half> expanded = currentIndexLocal.template ReinterpretCast<half>();
        Cast(expanded, equalBits, RoundMode::CAST_NONE, 256U);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> minima = headerLocal.template ReinterpretCast<half>();
        WholeReduceMin(minima, expanded, 128, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMin(minima[16U], expanded[128U], 128, 1, 1, 1, 8,
            ReduceOrder::ORDER_ONLY_VALUE);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        // V528: CANN scalar code cannot compare half. 255 is exactly0x5BF8;
        // all reduction inputs are finite nonnegative integral bytes.
        LocalTensor<uint16_t> minimumBits = minima.template ReinterpretCast<uint16_t>();
        if (minimumBits.GetValue(0U) != 0x5BF8U ||
            minimumBits.GetValue(16U) != 0x5BF8U) {
            return this->RejectPrivateBucketCache();
        }
        // V527_BITMAP_REDUCTION_END
        const uint32_t allocatedPlanWords = (this->bucketRestorePlanBase +
            this->bucketGatherOuterBatch * this->bucketValueStride + 7U) & ~7U;
        this->pipe.InitBuffer(this->waveOffsetBuf, allocatedPlanWords * sizeof(uint32_t));
        LocalTensor<uint32_t> offsetLocal = this->waveOffsetBuf.template Get<uint32_t>();
        DataCopy(offsetLocal.template ReinterpretCast<int32_t>(),
            this->bucketPlanGm[this->bucketCachedPlanBase], (planElements + 7U) & ~7U);
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline void CommitPrivateBucketCache(
        const LocalTensor<uint32_t>& offsetLocal, uint32_t planElements)
    {
        if (!this->PrivateBucketLayoutFits()) {
            this->InvalidatePrivateBucketCache();
            return;
        }
        // The original cold builder normalized nextBuf's first half. Reload raw
        // indices into its independent second half: alias changes must miss.
        LocalTensor<int32_t> rawIndexLocal =
            this->nextBuf.template Get<int32_t>()[this->privateIndexWords];
        DataCopyExtParams rawParams{1U, this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> noPad{false, 0U, 0U, 0};
        DataCopyPad(rawIndexLocal, this->indexGm, rawParams, noPad);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = this->indexSize; i < this->privateIndexWords; ++i) {
            rawIndexLocal.SetValue(i, 0);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm[this->bucketCachedIndexBase], rawIndexLocal, this->privateIndexWords);
        DataCopy(this->bucketPlanGm[this->bucketCachedPlanBase],
            offsetLocal.template ReinterpretCast<int32_t>(), (planElements + 7U) & ~7U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        LocalTensor<int32_t> headerLocal = this->headBuf.template Get<int32_t>();
        Duplicate(headerLocal, static_cast<int32_t>(0), kHalfBucketCacheHeaderWords);
        PipeBarrier<PIPE_ALL>();
        headerLocal.SetValue(0U, static_cast<int32_t>(kPrivateBucketMagicA));
        headerLocal.SetValue(1U, static_cast<int32_t>(kPrivateBucketMagicB));
        headerLocal.SetValue(2U, static_cast<int32_t>(kPrivateBucketVersion));
        headerLocal.SetValue(3U, static_cast<int32_t>(GetBlockNum()));
        headerLocal.SetValue(4U, static_cast<int32_t>(this->outerSize));
        headerLocal.SetValue(5U, static_cast<int32_t>(this->dimSize));
        headerLocal.SetValue(6U, static_cast<int32_t>(this->indexSize));
        headerLocal.SetValue(7U, static_cast<int32_t>(planElements));
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount; ++bucket) {
            headerLocal.SetValue(8U + bucket, static_cast<int32_t>(this->bucketWidth[bucket]));
            headerLocal.SetValue(16U + bucket, static_cast<int32_t>(this->bucketCap[bucket]));
        }
        headerLocal.SetValue(24U, static_cast<int32_t>(this->bucketGatherOuterBatch));
        headerLocal.SetValue(25U, static_cast<int32_t>(GetBlockIdx()));
        headerLocal.SetValue(26U, static_cast<int32_t>(this->privateIndexWords));
        headerLocal.SetValue(27U, static_cast<int32_t>(kPrivateBucketSlotWords));
        headerLocal.SetValue(28U, static_cast<int32_t>(sizeof(T)));
        headerLocal.SetValue(29U, static_cast<int32_t>(this->ActualHalfNextBytes()));
        headerLocal.SetValue(30U, 1);
        PipeBarrier<PIPE_ALL>();
        // Only this logical core writes this slot. Data is complete before validity.
        DataCopy(this->bucketPlanGm, headerLocal, kHalfBucketCacheHeaderWords);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
    }
    // V440_PRIVATE_METHODS_END

    __aicore__ inline uint32_t RebuildBucketDerivedMetadata()
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        this->bucketZeroElement = AlignUp32(
            this->indexSize * sizeof(T)) / sizeof(T);
        this->bucketSelfElement = this->bucketZeroElement + alignElements;
        this->bucketValueStride = AlignUp32(
            this->dimSize * sizeof(T)) / sizeof(T);
        this->bucketSourceRowStride =
            this->bucketSelfElement + this->bucketValueStride;
        const uint32_t waveStorageBase =
            this->bucketGatherOuterBatch * this->bucketSourceRowStride;
        uint32_t blockEntries = 0U;
        this->bucketMaxMatrixElements = 0U;
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount; ++bucket) {
            this->bucketStride[bucket] = this->bucketWidth[bucket];
            this->bucketPlanBase[bucket] = blockEntries;
            this->bucketWaveBase[bucket] = waveStorageBase +
                blockEntries * kHalfBucketBlockOuterLanes;
            this->bucketMatrixElements[bucket] =
                (this->bucketCap[bucket] + 1U) *
                this->bucketWidth[bucket] * kHalfBucketBlockOuterLanes;
            blockEntries += (this->bucketCap[bucket] + 1U) *
                this->bucketWidth[bucket];
            if (this->bucketMatrixElements[bucket] >
                this->bucketMaxMatrixElements) {
                this->bucketMaxMatrixElements =
                    this->bucketMatrixElements[bucket];
            }
        }
        // Gatherb consumes eight 32-byte blocks per repeat. Tail blocks always
        // point at the transposed zero column, never past the source allocation.
        this->bucketRestorePlanBase = (blockEntries + 7U) & ~7U;
        this->bucketTotalMatrixElements =
            this->bucketRestorePlanBase * kHalfBucketBlockOuterLanes;
        return this->bucketRestorePlanBase +
            this->bucketGatherOuterBatch * this->dimSize;
    }

    __aicore__ inline bool TryLoadContiguousCountBucketCache()
    {
        // V440_PRIVATE_LOAD_BEGIN
        if (this->usePrivateBucketCache) {
            return this->TryLoadPrivateBucketCache();
        }
        // V440_PRIVATE_LOAD_END
        if (!this->useBucketPlanCache) {
            return false;
        }
        LocalTensor<int32_t> headerLocal =
            this->headBuf.template Get<int32_t>();
        DataCopy(headerLocal, this->bucketPlanGm,
            kHalfBucketCacheHeaderWords);
        PipeBarrier<PIPE_ALL>();
        const bool headerMatches =
            static_cast<uint32_t>(headerLocal.GetValue(0U)) ==
                kHalfBucketCacheMagicA &&
            static_cast<uint32_t>(headerLocal.GetValue(1U)) ==
                kHalfBucketCacheMagicB &&
            static_cast<uint32_t>(headerLocal.GetValue(2U)) ==
                kHalfBucketCacheVersion &&
            static_cast<uint32_t>(headerLocal.GetValue(3U)) ==
                GetBlockNum() &&
            static_cast<uint32_t>(headerLocal.GetValue(4U)) ==
                this->outerSize &&
            static_cast<uint32_t>(headerLocal.GetValue(5U)) ==
                this->dimSize &&
            static_cast<uint32_t>(headerLocal.GetValue(6U)) ==
                this->indexSize;
        if (!headerMatches) {
            return false;
        }

        uint32_t widthSum = 0U;
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount;
             ++bucket) {
            this->bucketWidth[bucket] = static_cast<uint32_t>(
                headerLocal.GetValue(8U + bucket));
            this->bucketCap[bucket] = static_cast<uint32_t>(
                headerLocal.GetValue(16U + bucket));
            widthSum += this->bucketWidth[bucket];
        }
        this->bucketGatherOuterBatch = static_cast<uint32_t>(
            headerLocal.GetValue(24U));
        const uint32_t planElements =
            this->RebuildBucketDerivedMetadata();
        const uint32_t cachedPlanElements = static_cast<uint32_t>(
            headerLocal.GetValue(7U));
        const uint32_t alignedPlanWords =
            (planElements + 7U) & ~7U;
        const bool metadataValid = widthSum <= this->dimSize &&
            this->bucketCap[0] == 1U && this->bucketCap[1] == 2U &&
            this->bucketCap[2] == 3U && this->bucketCap[3] == 4U &&
            this->bucketCap[4] == 5U && this->bucketCap[5] == 6U &&
            this->bucketCap[6] == 8U && this->bucketCap[7] > 0U &&
            this->bucketCap[7] <= 255U &&
            this->bucketGatherOuterBatch >= 2U &&
            (this->bucketGatherOuterBatch & 1U) == 0U &&
            this->bucketGatherOuterBatch <= kHalfCountBucketGatherOuter &&
            cachedPlanElements == planElements &&
            alignedPlanWords <=
                kGenericWaveTotalBytes / sizeof(uint32_t);
        if (!metadataValid) {
            return false;
        }

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseIndexes = this->indexSize / blockNum;
        const uint32_t extraIndexes =
            this->indexSize - baseIndexes * blockNum;
        const uint32_t indexStart = blockIdx * baseIndexes +
            (blockIdx < extraIndexes ? blockIdx : extraIndexes);
        const uint32_t localIndexCount = baseIndexes +
            (blockIdx < extraIndexes ? 1U : 0U);
        LocalTensor<int32_t> currentIndexLocal =
            this->nextBuf.template Get<int32_t>();
        LocalTensor<int32_t> cachedIndexLocal =
            currentIndexLocal[this->bucketAlignedIndexWords];
        DataCopyExtParams localIndexParams{1,
            localIndexCount * static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(currentIndexLocal, this->indexGm[indexStart],
            localIndexParams, indexPadParams);
        DataCopyPad(cachedIndexLocal,
            this->bucketPlanGm[this->bucketCachedIndexBase + indexStart],
            localIndexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();
        uint32_t mismatch = 0U;
        for (uint32_t localIndex = 0U; localIndex < localIndexCount;
             ++localIndex) {
            int32_t target = currentIndexLocal.GetValue(localIndex);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 ||
                target >= static_cast<int32_t>(this->dimSize)) {
                target = -1;
            }
            if (target != cachedIndexLocal.GetValue(localIndex)) {
                mismatch = 1U;
                break;
            }
        }
        Duplicate(cachedIndexLocal, static_cast<int32_t>(0),
            kHalfBucketCacheFlagStride);
        cachedIndexLocal.SetValue(0U, static_cast<int32_t>(mismatch));
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm[this->bucketCacheFlagBase +
                blockIdx * kHalfBucketCacheFlagStride],
            cachedIndexLocal, kHalfBucketCacheFlagStride);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();

        DataCopy(cachedIndexLocal,
            this->bucketPlanGm[this->bucketCacheFlagBase],
            blockNum * kHalfBucketCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t core = 0U; core < blockNum; ++core) {
            if (cachedIndexLocal.GetValue(
                    core * kHalfBucketCacheFlagStride) != 0) {
                return false;
            }
        }

        // Keep the conservative padded allocation on cache hits as well, so
        // source/matrix UB addresses do not shift between cold and warm calls.
        const uint32_t allocatedPlanWords =
            (this->bucketRestorePlanBase + this->bucketGatherOuterBatch *
                this->bucketValueStride + 7U) & ~7U;
        this->pipe.InitBuffer(this->waveOffsetBuf,
            allocatedPlanWords * sizeof(uint32_t));
        LocalTensor<uint32_t> offsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        DataCopy(offsetLocal.template ReinterpretCast<int32_t>(),
            this->bucketPlanGm[this->bucketCachedPlanBase],
            alignedPlanWords);
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline void CommitContiguousCountBucketCache(
        const LocalTensor<int32_t>& normalizedIndexLocal,
        const LocalTensor<uint32_t>& offsetLocal,
        uint32_t planElements)
    {
        // V440_PRIVATE_COMMIT_BEGIN
        if (this->usePrivateBucketCache) {
            this->CommitPrivateBucketCache(offsetLocal, planElements);
            return;
        }
        // V440_PRIVATE_COMMIT_END
        if (!this->useBucketPlanCache || GetBlockIdx() != 0U) {
            return;
        }
        const uint32_t alignedPlanWords =
            (planElements + 7U) & ~7U;
        DataCopy(this->bucketPlanGm[this->bucketCachedIndexBase],
            normalizedIndexLocal, this->bucketAlignedIndexWords);
        DataCopy(this->bucketPlanGm[this->bucketCachedPlanBase],
            offsetLocal.template ReinterpretCast<int32_t>(),
            alignedPlanWords);

        LocalTensor<int32_t> headerLocal =
            this->headBuf.template Get<int32_t>();
        Duplicate(headerLocal, static_cast<int32_t>(0),
            kHalfBucketCacheHeaderWords);
        headerLocal.SetValue(0U,
            static_cast<int32_t>(kHalfBucketCacheMagicA));
        headerLocal.SetValue(1U,
            static_cast<int32_t>(kHalfBucketCacheMagicB));
        headerLocal.SetValue(2U,
            static_cast<int32_t>(kHalfBucketCacheVersion));
        headerLocal.SetValue(3U,
            static_cast<int32_t>(GetBlockNum()));
        headerLocal.SetValue(4U,
            static_cast<int32_t>(this->outerSize));
        headerLocal.SetValue(5U,
            static_cast<int32_t>(this->dimSize));
        headerLocal.SetValue(6U,
            static_cast<int32_t>(this->indexSize));
        headerLocal.SetValue(7U,
            static_cast<int32_t>(planElements));
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount;
             ++bucket) {
            headerLocal.SetValue(8U + bucket,
                static_cast<int32_t>(this->bucketWidth[bucket]));
            headerLocal.SetValue(16U + bucket,
                static_cast<int32_t>(this->bucketCap[bucket]));
        }
        headerLocal.SetValue(24U,
            static_cast<int32_t>(this->bucketGatherOuterBatch));
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm, headerLocal,
            kHalfBucketCacheHeaderWords);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
    }

    __aicore__ inline uint32_t SelectCountBucket(uint32_t count) const
    {
        if (count <= 1U) {
            return 0U;
        }
        if (count <= 2U) {
            return 1U;
        }
        if (count <= 3U) {
            return 2U;
        }
        if (count <= 4U) {
            return 3U;
        }
        if (count <= 5U) {
            return 4U;
        }
        if (count <= 6U) {
            return 5U;
        }
        if (count <= 8U) {
            return 6U;
        }
        return 7U;
    }

    __aicore__ inline void CompactBucketRestorePlan(
        const LocalTensor<uint32_t>& offsetLocal)
    {
        const uint32_t rowPadding = this->bucketValueStride - this->dimSize;
        if (rowPadding == 0U) {
            return;
        }
        if (this->dimSize >= 192U) {
            // Three 64-word vectors fit the existing head only for D>=192.
            // Counts are dead; normalized indices in nextBuf remain untouched.
            LocalTensor<int32_t> laneBase = this->headBuf.template Get<int32_t>();
            LocalTensor<int32_t> selection = laneBase[64U];
            LocalTensor<int32_t> compact = laneBase[128U];
            LocalTensor<uint32_t> restore = offsetLocal[this->bucketRestorePlanBase];
            const uint32_t total = this->bucketGatherOuterBatch * this->dimSize;
            const UnaryRepeatParams maskParams{1U, 1U, 8U, 8U};
            CreateVecIndex(laneBase, static_cast<int32_t>(0), 64U);
            PipeBarrier<PIPE_V>();
            for (uint32_t start = 0U; start < total; start += 64U) {
                uint32_t count = total - start;
                if (count > 64U) {
                    count = 64U;
                }
                const uint32_t firstRow = start / this->dimSize;
                const uint32_t nextBoundary = this->dimSize - start % this->dimSize;
                Adds(selection, laneBase,
                    static_cast<int32_t>(start + firstRow * rowPadding), count);
                PipeBarrier<PIPE_V>();
                if (nextBoundary < count) {
                    const uint64_t activeMask = count == 64U
                        ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << count) - 1ULL);
                    uint64_t suffixMask[2] = {
                        activeMask & (0xFFFFFFFFFFFFFFFFULL << nextBoundary), 0ULL};
                    Adds(selection, selection, static_cast<int32_t>(rowPadding),
                        suffixMask, 1U, maskParams);
                    PipeBarrier<PIPE_V>();
                }
                Muls(selection, selection, static_cast<int32_t>(sizeof(uint32_t)), count);
                PipeBarrier<PIPE_V>();
                Gather(compact.template ReinterpretCast<uint32_t>(), restore,
                    selection.template ReinterpretCast<uint32_t>(), 0, count);
                PipeBarrier<PIPE_V>();
                Adds(restore[start].template ReinterpretCast<int32_t>(), compact,
                    static_cast<int32_t>(0), count);
                // No scalar UB write occurs inside this loop. Preserve vector
                // read-before-overwrite order without a per-chunk full drain.
                PipeBarrier<PIPE_V>();
            }
            // Cache publication reads the final vector-written plan through MTE3.
            PipeBarrier<PIPE_ALL>();
            return;
        }
        // This route requires D>=128. Counts have finished producing the
        // restore map, so the 128-word head is dead until cache-header commit.
        // Keep nextBuf/normalized indices intact: no earlier GM publication.
        LocalTensor<int32_t> selection = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> compact = selection[64U];
        LocalTensor<uint32_t> restore = offsetLocal[this->bucketRestorePlanBase];
        const uint32_t total = this->bucketGatherOuterBatch * this->dimSize;
        const UnaryRepeatParams maskParams{1U, 1U, 8U, 8U};
        for (uint32_t start = 0U; start < total; start += 64U) {
            uint32_t count = total - start;
            if (count > 64U) {
                count = 64U;
            }
            const uint32_t firstRow = start / this->dimSize;
            const uint32_t nextBoundary = this->dimSize - start % this->dimSize;
            CreateVecIndex(selection,
                static_cast<int32_t>(start + firstRow * rowPadding), count);
            PipeBarrier<PIPE_V>();
            // A 64-entry chunk crosses at most one D>=128 row boundary.
            if (nextBoundary < count) {
                const uint64_t activeMask = count == 64U
                    ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << count) - 1ULL);
                uint64_t suffixMask[2] = {
                    activeMask & (0xFFFFFFFFFFFFFFFFULL << nextBoundary), 0ULL};
                Adds(selection, selection, static_cast<int32_t>(rowPadding),
                    suffixMask, 1U, maskParams);
                PipeBarrier<PIPE_V>();
            }
            Muls(selection, selection, static_cast<int32_t>(sizeof(uint32_t)), count);
            PipeBarrier<PIPE_V>();
            Gather(compact.template ReinterpretCast<uint32_t>(), restore,
                selection.template ReinterpretCast<uint32_t>(), 0, count);
            PipeBarrier<PIPE_V>();
            // Read the entire chunk before writing in place. All future source
            // positions are >= the next output chunk; no unread entry is lost.
            Adds(restore[start].template ReinterpretCast<int32_t>(), compact,
                static_cast<int32_t>(0), count);
            // The next CreateVecIndex writes selection through scalar SetValue.
            // Also drain the final restore write before cache publication to GM.
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline bool PrepareContiguousCountBuckets()
    {
        // A transposed column contains sixteen independent FP16 outer rows.
        // Gatherb fetches that complete 32-byte column using one plan entry.
        // Every target still accumulates self then source indices in order.
        if (this->dimSize >= 512U || sizeof(T) != 2U) {
            return false;
        }
        if (this->TryLoadContiguousCountBucketCache()) {
            return true;
        }

        constexpr uint32_t alignElements = 32U / sizeof(T);
        LocalTensor<int32_t> countLocal =
            this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> targetLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();

        uint32_t maxCount = 0U;
        for (uint32_t i = 0U; i < this->indexSize; ++i) {
            int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                targetLocal.SetValue(i, -1);
                continue;
            }
            targetLocal.SetValue(i, target);
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t count = static_cast<uint32_t>(
                countLocal.GetValue(targetDim)) + 1U;
            countLocal.SetValue(targetDim, static_cast<int32_t>(count));
            if (count > maxCount) {
                maxCount = count;
            }
        }
        if (maxCount == 0U || maxCount > 255U) {
            return false;
        }

        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount; ++bucket) {
            this->bucketWidth[bucket] = 0U;
            this->bucketStride[bucket] = 0U;
            this->bucketMatrixElements[bucket] = 0U;
            this->bucketPlanBase[bucket] = 0U;
        }
        this->bucketCap[0] = 1U;
        this->bucketCap[1] = 2U;
        this->bucketCap[2] = 3U;
        this->bucketCap[3] = 4U;
        this->bucketCap[4] = 5U;
        this->bucketCap[5] = 6U;
        this->bucketCap[6] = 8U;
        this->bucketCap[7] = maxCount;
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(countLocal.GetValue(dim));
            ++this->bucketWidth[this->SelectCountBucket(count)];
        }

        this->bucketMaxMatrixElements = 0U;
        for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount; ++bucket) {
            if (this->bucketWidth[bucket] == 0U) {
                continue;
            }
            this->bucketStride[bucket] = this->bucketWidth[bucket];
        }

        this->bucketZeroElement = AlignUp32(this->indexSize * sizeof(T)) /
            sizeof(T);
        this->bucketSelfElement = this->bucketZeroElement + alignElements;
        this->bucketValueStride = AlignUp32(this->dimSize * sizeof(T)) /
            sizeof(T);
        this->bucketSourceRowStride =
            this->bucketSelfElement + this->bucketValueStride;

        const uint64_t headBytes = AlignUp32(this->dimSize * sizeof(int32_t));
        const uint64_t nextBytes = this->ActualHalfNextBytes(); // V440_PRIVATE_NEXT_BUCKET
        if (this->bucketSourceRowStride / 16U > 255U) {
            return false;
        }
        this->bucketGatherOuterBatch = kHalfCountBucketGatherOuter;
        while (this->bucketGatherOuterBatch >= 2U) {
            this->RebuildBucketDerivedMetadata();
            // Cold construction still expands aligned rows before compaction.
            // Budget and allocate the larger padded map, not the cached size.
            const uint32_t planElements = this->bucketRestorePlanBase +
                this->bucketGatherOuterBatch * this->bucketValueStride;
            const uint64_t planBytes = AlignUp32(
                planElements * sizeof(uint32_t));
            const uint64_t sourceBytes =
                (static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                    this->bucketSourceRowStride +
                    this->bucketTotalMatrixElements +
                    17U * this->bucketSourceRowStride) * sizeof(T);
            const uint64_t outputBytes =
                static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                this->bucketValueStride * sizeof(T);
            if (headBytes + nextBytes + planBytes + sourceBytes + outputBytes +
                    kHalfCountBucketUbSafetyBytes <= kGenericWaveTotalBytes) {
                break;
            }
            this->bucketGatherOuterBatch -= 2U;
        }
        if (this->bucketGatherOuterBatch < 2U) {
            return false;
        }

        const uint32_t planElements = this->bucketRestorePlanBase +
            this->bucketGatherOuterBatch * this->bucketValueStride;
        this->pipe.InitBuffer(this->waveOffsetBuf,
            AlignUp32(planElements * sizeof(uint32_t)));
        LocalTensor<uint32_t> offsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t zeroByteOffset = this->bucketZeroElement *
            kHalfBucketBlockOuterLanes * sizeof(T);
        Duplicate(offsetLocal, zeroByteOffset, planElements);
        PipeBarrier<PIPE_ALL>();

        uint32_t bucketCursor[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(countLocal.GetValue(dim));
            const uint32_t bucket = this->SelectCountBucket(count);
            const uint32_t column = bucketCursor[bucket]++;
            const uint32_t encoded = (bucket << 24U) | (column << 8U);
            countLocal.SetValue(dim, static_cast<int32_t>(encoded));
            offsetLocal.SetValue(this->bucketPlanBase[bucket] + column,
                (this->bucketSelfElement + dim) *
                    kHalfBucketBlockOuterLanes * sizeof(T));
        }
        for (uint32_t i = 0U; i < this->indexSize; ++i) {
            const int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t encoded = static_cast<uint32_t>(
                countLocal.GetValue(targetDim));
            const uint32_t bucket = (encoded >> 24U) & 0xFFU;
            const uint32_t column = (encoded >> 8U) & 0xFFFFU;
            const uint32_t rank = encoded & 0xFFU;
            offsetLocal.SetValue(this->bucketPlanBase[bucket] +
                    (rank + 1U) * this->bucketWidth[bucket] + column,
                i * kHalfBucketBlockOuterLanes * sizeof(T));
            countLocal.SetValue(targetDim,
                static_cast<int32_t>((encoded & ~0xFFU) | (rank + 1U)));
        }

        // In the block layout every outer lane differs by exactly sizeof(T).
        // Build one restoration row scalarly, then expand complete aligned
        // rows with vector Adds rather than outer*dim scalar stores.
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t encoded =
                static_cast<uint32_t>(countLocal.GetValue(dim));
            const uint32_t bucket = (encoded >> 24U) & 0xFFU;
            const uint32_t column = (encoded >> 8U) & 0xFFFFU;
            const uint32_t restoreElement = this->bucketWaveBase[bucket] +
                column * kHalfBucketBlockOuterLanes;
            offsetLocal.SetValue(this->bucketRestorePlanBase + dim,
                restoreElement * sizeof(T));
        }
        const uint32_t matrixBaseBytes = this->bucketGatherOuterBatch *
            this->bucketSourceRowStride * sizeof(T);
        for (uint32_t dim = this->dimSize;
             dim < this->bucketValueStride; ++dim) {
            // Padded restoration columns are ignored by the output DMA. They
            // must still read matrix storage, not the concurrently reused raw.
            offsetLocal.SetValue(this->bucketRestorePlanBase + dim,
                matrixBaseBytes);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t outer = 1U;
             outer < this->bucketGatherOuterBatch; ++outer) {
            Adds(offsetLocal[this->bucketRestorePlanBase +
                    outer * this->bucketValueStride].template ReinterpretCast<int32_t>(),
                offsetLocal[this->bucketRestorePlanBase].template ReinterpretCast<int32_t>(),
                static_cast<int32_t>(outer * sizeof(T)),
                this->bucketValueStride);
        }
        PipeBarrier<PIPE_ALL>();
        this->CompactBucketRestorePlan(offsetLocal);
        const uint32_t compactPlanElements = this->bucketRestorePlanBase +
            this->bucketGatherOuterBatch * this->dimSize;
        this->CommitContiguousCountBucketCache(
            targetLocal, offsetLocal, compactPlanElements);
        return true;
    }

    __aicore__ inline void ProcessContiguousCountBuckets()
    {
        // V443: restore the default address before eligibility and all exits.
        this->sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t elementsPerRepeat = 256U / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter = this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart = blockIdx * baseOuter +
            (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd = outerStart + baseOuter +
            (blockIdx < extraOuter ? 1U : 0U);
        if (outerStart >= outerEnd) {
            return;
        }

        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal =
            this->waveValueBuf.template Get<T>();
        LocalTensor<uint32_t> offsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements), static_cast<T>(0)};
        DataCopyPadExtParams<T> valuePadParams{
            false, 0, 0, static_cast<T>(0)};
        const uint32_t sourceRowStride = this->bucketSourceRowStride;
        const uint32_t sourceDstStride =
            (sourceRowStride - this->bucketZeroElement - alignElements) /
                alignElements;
        const uint32_t valueDstStride =
            (sourceRowStride - this->bucketValueStride) /
                alignElements;

        const uint32_t bufferElements = this->bucketGatherOuterBatch * sourceRowStride;
        LocalTensor<T> pairSourceLocal = this->pairSourceBuf.template Get<T>();
        LocalTensor<T> zeroRow = pairSourceLocal[16U * sourceRowStride];
        LocalTensor<T> matrixLocal = sourceLocal[bufferElements];
        // V436: keep the interleaved layout, but use block DMA for aligned M.
        // Every raw zero sentinel is initialized per call, never cache-reused.
        const uint64_t sourceElements64 =
            static_cast<uint64_t>(this->outerSize) * this->indexSize;
        const uint64_t sourceBytes64 = sourceElements64 * sizeof(T);
        const uint64_t rawElements64 =
            static_cast<uint64_t>(this->bucketGatherOuterBatch) * sourceRowStride;
        const uint64_t headBytes64 =
            (static_cast<uint64_t>(this->dimSize) * sizeof(int32_t) + 31ULL) & ~31ULL;
        // V523: include actual private scratch in block-DMA eligibility.
        const uint64_t nextBytes64 = this->ActualHalfNextBytes();
        const uint64_t planBytes64 =
            ((static_cast<uint64_t>(this->bucketRestorePlanBase) +
                static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                    this->bucketValueStride) * sizeof(uint32_t) + 31ULL) & ~31ULL;
        const uint64_t allBuffers64 = headBytes64 + nextBytes64 + planBytes64 +
            (rawElements64 + this->bucketTotalMatrixElements +
                17ULL * sourceRowStride +
                static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                    this->bucketValueStride) * sizeof(T);
        const uint64_t sourceAddress64 =
            reinterpret_cast<uint64_t>(this->sourceGm.GetPhyAddr());
        const bool useBlockSourceDma = sizeof(T) == 2U &&
            this->innerSize == 1U && this->indexSize > 0U &&
            this->indexSize % alignElements == 0U &&
            sourceRowStride % alignElements == 0U &&
            this->bucketGatherOuterBatch > 0U &&
            this->bucketGatherOuterBatch <= 16U &&
            this->bucketZeroElement == this->indexSize &&
            static_cast<uint64_t>(this->bucketSelfElement) ==
                static_cast<uint64_t>(this->indexSize) + alignElements &&
            static_cast<uint64_t>(this->bucketSelfElement) +
                this->bucketValueStride == sourceRowStride &&
            this->dimSize <= this->bucketValueStride &&
            sourceRowStride >= static_cast<uint64_t>(this->indexSize) + alignElements &&
            this->indexSize / alignElements <= 65535U &&
            sourceRowStride / alignElements <= 65535U &&
            (sourceRowStride - this->indexSize) / alignElements <= 65535U &&
            sourceElements64 <= 0xFFFFFFFFULL && outerEnd <= this->outerSize &&
            static_cast<uint64_t>(outerEnd) * this->indexSize <= sourceElements64 &&
            sourceAddress64 <= ~0ULL - sourceBytes64 &&
            (sourceAddress64 & 31ULL) == 0ULL &&
            (reinterpret_cast<uint64_t>(sourceLocal.GetPhyAddr()) & 31ULL) == 0ULL &&
            allBuffers64 + kHalfCountBucketUbSafetyBytes <= kGenericWaveTotalBytes;
        if (useBlockSourceDma) {
            // Only the guarded ordinary source DMA uses the streaming hint.
            this->sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            for (uint32_t firstRow = 0U;
                 firstRow < this->bucketGatherOuterBatch; firstRow += 8U) {
                uint32_t rows = this->bucketGatherOuterBatch - firstRow;
                if (rows > 8U) {
                    rows = 8U;
                }
                Duplicate(sourceLocal[firstRow * sourceRowStride + this->bucketZeroElement],
                    static_cast<T>(0), static_cast<uint64_t>(rows * alignElements),
                    static_cast<uint8_t>(1),
                    static_cast<uint16_t>(sourceRowStride / alignElements),
                    static_cast<uint8_t>(0));
            }
        }
        Duplicate(zeroRow, static_cast<T>(0), sourceRowStride);
        PipeBarrier<PIPE_ALL>();
        uint32_t currentOuter = outerStart;
        uint32_t currentCount = outerEnd - currentOuter;
        if (currentCount > this->bucketGatherOuterBatch) {
            currentCount = this->bucketGatherOuterBatch;
        }
        DataCopyExtParams initialSourceParams{
            static_cast<uint16_t>(currentCount),
            this->indexSize * static_cast<uint32_t>(sizeof(T)),
            0, sourceDstStride, 0};
        DataCopyExtParams initialSelfParams{
            static_cast<uint16_t>(currentCount),
            this->dimSize * static_cast<uint32_t>(sizeof(T)),
            0, valueDstStride, 0};
        if (useBlockSourceDma) {
            DataCopyParams sourceBlockParams{
                static_cast<uint16_t>(currentCount),
                static_cast<uint16_t>(this->indexSize / alignElements), 0U,
                static_cast<uint16_t>((sourceRowStride - this->indexSize) / alignElements)};
            DataCopy(sourceLocal, this->sourceGm[currentOuter * this->indexSize],
                sourceBlockParams);
        } else {
            DataCopyPad(sourceLocal,
                this->sourceGm[currentOuter * this->indexSize],
                initialSourceParams, sourcePadParams);
        }
        DataCopyPad(sourceLocal[this->bucketSelfElement],
            this->selfGm[currentOuter * this->dimSize],
            initialSelfParams, valuePadParams);
        SetFlag<HardEvent::MTE2_V>(0);

        bool outputPending = false;
        while (currentOuter < outerEnd) {
            WaitFlag<HardEvent::MTE2_V>(0);
            const uint32_t nextOuter = currentOuter + currentCount;
            uint32_t nextCount = outerEnd - nextOuter;
            if (nextCount > this->bucketGatherOuterBatch) {
                nextCount = this->bucketGatherOuterBatch;
            }
            LocalTensor<T> currentSourceLocal =
                sourceLocal;
            uint64_t destinationList[16];
            uint64_t sourceList[16];
            for (uint32_t lane = 0U; lane < 16U; ++lane) {
                destinationList[lane] = (uint64_t)pairSourceLocal[lane * 16U].GetPhyAddr();
                sourceList[lane] = lane < currentCount
                    ? (uint64_t)currentSourceLocal[lane * sourceRowStride].GetPhyAddr()
                    : (uint64_t)zeroRow.GetPhyAddr();
            }
            TransDataTo5HDParams transposeParams{false, false,
                static_cast<uint8_t>(sourceRowStride / 16U), 16, 1};
            TransDataTo5HD<uint16_t>(destinationList, sourceList, transposeParams);
            PipeBarrier<PIPE_V>();
            if (nextCount > 0U) {
                // Only the transpose consumes the raw rows. Release that input
                // buffer now and overlap the next DMA with matrix reduction.
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                DataCopyExtParams nextSourceParams{
                    static_cast<uint16_t>(nextCount),
                    this->indexSize * static_cast<uint32_t>(sizeof(T)),
                    0, sourceDstStride, 0};
                DataCopyExtParams nextSelfParams{
                    static_cast<uint16_t>(nextCount),
                    this->dimSize * static_cast<uint32_t>(sizeof(T)),
                    0, valueDstStride, 0};
                if (useBlockSourceDma) {
                    DataCopyParams sourceBlockParams{
                        static_cast<uint16_t>(nextCount),
                        static_cast<uint16_t>(this->indexSize / alignElements), 0U,
                        static_cast<uint16_t>((sourceRowStride - this->indexSize) / alignElements)};
                    DataCopy(sourceLocal, this->sourceGm[nextOuter * this->indexSize],
                        sourceBlockParams);
                } else {
                    DataCopyPad(sourceLocal,
                        this->sourceGm[nextOuter * this->indexSize],
                        nextSourceParams, sourcePadParams);
                }
                DataCopyPad(sourceLocal[this->bucketSelfElement],
                    this->selfGm[nextOuter * this->dimSize],
                    nextSelfParams, valuePadParams);
                SetFlag<HardEvent::MTE2_V>(0);
            }
            const GatherRepeatParams blockGatherParams{1, 8};
            for (uint32_t entry = 0U;
                 entry < this->bucketRestorePlanBase;) {
                uint32_t blocks = this->bucketRestorePlanBase - entry;
                if (blocks > 255U * 8U) {
                    blocks = 255U * 8U;
                }
                Gatherb(matrixLocal[entry * kHalfBucketBlockOuterLanes]
                        .template ReinterpretCast<uint16_t>(),
                    pairSourceLocal.template ReinterpretCast<uint16_t>(),
                    offsetLocal[entry], static_cast<uint8_t>(blocks / 8U),
                    blockGatherParams);
                entry += blocks;
            }
            PipeBarrier<PIPE_V>();

            for (uint32_t bucket = 0U; bucket < kHalfCountBucketCount;
                 ++bucket) {
                const uint32_t width = this->bucketWidth[bucket];
                const uint32_t matrixElements =
                    this->bucketMatrixElements[bucket];
                if (width == 0U || matrixElements == 0U) {
                    continue;
                }
                const uint32_t stride = this->bucketStride[bucket];
                const uint32_t batchStride =
                    kHalfBucketBlockOuterLanes * stride;
                // Every column is a complete sixteen-lane block, including
                // zeroed inactive outer lanes in a partial final batch.
                const uint32_t activeElements = batchStride;
                const uint32_t cap = this->bucketCap[bucket];
                const uint32_t rowBlocks =
                    batchStride * sizeof(T) / 32U;
                LocalTensor<T> bucketValue =
                    sourceLocal[this->bucketWaveBase[bucket]];
                if (rowBlocks <= 255U) {
                    const BinaryRepeatParams repeatParams{1, 1, 1, 0,
                        static_cast<uint8_t>(rowBlocks), 0};
                    for (uint32_t column = 0U; column < activeElements;
                         column += elementsPerRepeat) {
                        uint32_t count = activeElements - column;
                        if (count > elementsPerRepeat) {
                            count = elementsPerRepeat;
                        }
                        Add(bucketValue[column],
                            bucketValue[batchStride + column],
                            bucketValue[column],
                            static_cast<uint64_t>(count),
                            static_cast<uint8_t>(cap), repeatParams);
                    }
                } else {
                    // Wider neighboring families must not truncate the
                    // eight-bit repeat stride. Preserve the same update order.
                    for (uint32_t rank = 1U; rank <= cap; ++rank) {
                        Add(bucketValue, bucketValue[rank * batchStride],
                            bucketValue, activeElements);
                        PipeBarrier<PIPE_V>();
                    }
                }
                PipeBarrier<PIPE_V>();
            }

            if (outputPending) {
                // The shared output buffer must not be overwritten by V until
                // the preceding GM write has consumed it.
                WaitFlag<HardEvent::MTE3_V>(0);
                outputPending = false;
            }
            Gather(waveValueLocal, sourceLocal,
                offsetLocal[this->bucketRestorePlanBase], 0,
                currentCount * this->dimSize);

            // A compact restore map permits one contiguous write for this
            // entire outer batch, without per-target-row partial writes.
            DataCopyExtParams outputParams{
                1U,
                currentCount * this->dimSize * static_cast<uint32_t>(sizeof(T)),
                0, 0, 0};
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(this->outputGm[currentOuter * this->dimSize],
                waveValueLocal, outputParams);
            SetFlag<HardEvent::MTE3_V>(0);
            outputPending = true;
            currentOuter = nextOuter;
            currentCount = nextCount;
        }
        if (outputPending) {
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        this->sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    }

    __aicore__ inline bool PrepareContiguousWave()
    {
        LocalTensor<int32_t> countLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.template Get<int32_t>();

        // Both the uint32 offset rows and gathered T rows must be aligned.
        this->waveStride = AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        constexpr uint32_t alignElements = 32 / sizeof(T);
        this->zeroElement = AlignUp32(this->indexSize * sizeof(T)) / sizeof(T);
        this->waveValueStride = AlignUp32(this->dimSize * sizeof(T)) / sizeof(T);
        this->waveSourceStride = this->zeroElement + alignElements;

        const uint64_t headBytes = AlignUp32(this->dimSize * sizeof(int32_t));
        const uint64_t nextBytes = this->ActualHalfNextBytes(); // V440_PRIVATE_NEXT_WAVE
        constexpr uint64_t waveUbBudget = kGenericWaveTotalBytes - 8192U;
        const uint64_t perOuterBytes =
            static_cast<uint64_t>(this->waveValueStride + this->waveSourceStride) * sizeof(T);
        const uint64_t bufferedPerOuterBytes = 2U * perOuterBytes;
        if (headBytes + nextBytes + bufferedPerOuterBytes >= waveUbBudget) {
            return false;
        }

        const uint32_t averageWaves =
            (this->indexSize + this->dimSize - 1) / this->dimSize;
        uint32_t waveCapacity = averageWaves * 2 + 10;
        if (waveCapacity > 64) {
            waveCapacity = 64;
        }
        const uint64_t basePerWaveBytes =
            static_cast<uint64_t>(this->waveStride) *
            (sizeof(uint32_t) + sizeof(T));
        const uint64_t availableWaveBytes =
            waveUbBudget - headBytes - nextBytes - bufferedPerOuterBytes;
        this->waveGatherOuterBatch = kHalfGatherOuterBatch;
        while (this->waveGatherOuterBatch > 1 &&
               waveCapacity > availableWaveBytes /
                   (basePerWaveBytes * this->waveGatherOuterBatch)) {
            --this->waveGatherOuterBatch;
        }
        const uint64_t maxWaveCapacity =
            availableWaveBytes /
            (basePerWaveBytes * this->waveGatherOuterBatch);
        if (waveCapacity > maxWaveCapacity) {
            waveCapacity = static_cast<uint32_t>(maxWaveCapacity);
        }
        if (waveCapacity == 0) {
            return false;
        }

        const uint64_t waveCapacityElements =
            static_cast<uint64_t>(this->waveGatherOuterBatch) *
            waveCapacity * this->waveStride;
        const uint64_t waveCapacityDataBytes =
            waveCapacityElements * sizeof(uint32_t);
        if (waveCapacityDataBytes > static_cast<uint64_t>(0xFFFFFFFFU) - 31U) {
            return false;
        }
        const uint32_t waveBytes =
            AlignUp32(static_cast<uint32_t>(waveCapacityDataBytes));
        // A rejected wave plan retains its offset allocation. Leave space for
        // both fixed-size buffers used by the contiguous chain fallback.
        const uint64_t fallbackBytes =
            2ULL * kVectorTileElements * sizeof(T);
        if (headBytes + nextBytes + waveBytes + fallbackBytes > waveUbBudget) {
            return false;
        }
        this->pipe.InitBuffer(this->waveOffsetBuf, waveBytes);
        LocalTensor<uint32_t> waveOffsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t zeroByteOffset = this->zeroElement * sizeof(T);
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(waveOffsetLocal, zeroByteOffset,
            static_cast<uint32_t>(waveCapacityElements));
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();

        this->waveCount = 0;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t wave =
                static_cast<uint32_t>(countLocal.GetValue(targetDim));
            if (wave >= waveCapacity) {
                return false;
            }
            waveOffsetLocal.SetValue(wave * this->waveStride + targetDim,
                i * sizeof(T));
            const uint32_t nextWave = wave + 1;
            countLocal.SetValue(targetDim, static_cast<int32_t>(nextWave));
            if (nextWave > this->waveCount) {
                this->waveCount = nextWave;
            }
        }

        const uint64_t waveElements =
            static_cast<uint64_t>(this->waveCount) * this->waveStride;
        if (this->waveCount == 0) {
            this->waveGatherOuterBatch = 1;
        } else if (this->waveGatherOuterBatch > 1) {
            LocalTensor<int32_t> signedOffsetLocal =
                waveOffsetLocal.ReinterpretCast<int32_t>();
            const uint32_t sourceRowBytes = this->waveSourceStride * sizeof(T);
            for (uint32_t gatherOuter = 1;
                 gatherOuter < this->waveGatherOuterBatch; ++gatherOuter) {
                Adds(signedOffsetLocal[
                        gatherOuter * static_cast<uint32_t>(waveElements)],
                    signedOffsetLocal,
                    static_cast<int32_t>(gatherOuter * sourceRowBytes),
                    static_cast<int32_t>(waveElements));
            }
            PipeBarrier<PIPE_V>();
        }
        const uint64_t gatheredWaveBytes = waveElements == 0
            ? static_cast<uint64_t>(this->waveValueStride) * sizeof(T)
            : static_cast<uint64_t>(this->waveGatherOuterBatch) *
                waveElements * sizeof(T);
        const uint64_t fixedBytes =
            headBytes + nextBytes + waveBytes + gatheredWaveBytes;
        if (fixedBytes >= waveUbBudget || bufferedPerOuterBytes == 0) {
            return false;
        }
        uint64_t maxBatch =
            (waveUbBudget - fixedBytes) / bufferedPerOuterBytes;
        if (maxBatch == 0) {
            return false;
        }
        if (maxBatch > kHalfWaveOuterBatchMax) {
            maxBatch = kHalfWaveOuterBatchMax;
        }
        if (maxBatch > this->outerSize) {
            maxBatch = this->outerSize;
        }
        this->waveOuterBatch = static_cast<uint32_t>(maxBatch);
        return true;
    }

    __aicore__ inline void ProcessContiguousWave()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter = this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter + (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd =
            outerStart + baseOuter + (blockIdx < extraOuter ? 1U : 0U);

        LocalTensor<T> valueLocal = this->valueBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> waveValueLocal = this->waveValueBuf.template Get<T>();
        LocalTensor<uint32_t> waveOffsetLocal = this->waveOffsetBuf.template Get<uint32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements), static_cast<T>(0)};
        const uint32_t waveElements = this->waveCount * this->waveStride;
        const uint32_t waveRowBytes = this->waveStride * sizeof(T);
        const uint32_t waveRowBlocks = waveRowBytes / 32;
        constexpr uint32_t elementsPerRepeat = 256 / sizeof(T);
        const uint32_t repeatCallCount =
            (this->dimSize + elementsPerRepeat - 1) / elementsPerRepeat;
        const bool useWaveRepeat = this->waveCount > 0 &&
            this->waveCount <= 255 && waveRowBytes % 32 == 0 &&
            waveRowBlocks <= 255 && repeatCallCount < this->waveCount;
        const BinaryRepeatParams repeatParams{1, 1, 1, 0,
            static_cast<uint8_t>(waveRowBlocks), 0};

        if (outerStart >= outerEnd) {
            return;
        }
        const uint32_t valueBufferStride =
            this->waveOuterBatch * this->waveValueStride;
        const uint32_t sourceBufferStride =
            this->waveOuterBatch * this->waveSourceStride;
        uint32_t currentOuterBase = outerStart;
        uint32_t currentOuterCount = outerEnd - currentOuterBase;
        if (currentOuterCount > this->waveOuterBatch) {
            currentOuterCount = this->waveOuterBatch;
        }
        DataCopyExtParams initialSelfParams{static_cast<uint16_t>(currentOuterCount),
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams initialSourceParams{static_cast<uint16_t>(currentOuterCount),
            this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(valueLocal, this->selfGm[currentOuterBase * this->dimSize],
            initialSelfParams, padParams);
        DataCopyPad(sourceLocal, this->sourceGm[currentOuterBase * this->indexSize],
            initialSourceParams, sourcePadParams);
        SetFlag<HardEvent::MTE2_V>(0);

        uint32_t currentBuffer = 0;
        bool outputPending0 = false;
        bool outputPending1 = false;
        while (currentOuterBase < outerEnd) {
            const uint32_t nextOuterBase =
                currentOuterBase + this->waveOuterBatch;
            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextOuterBase < outerEnd) {
                if (nextBuffer == 0 && outputPending0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    outputPending0 = false;
                } else if (nextBuffer == 1 && outputPending1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(1);
                    outputPending1 = false;
                }
                uint32_t nextOuterCount = outerEnd - nextOuterBase;
                if (nextOuterCount > this->waveOuterBatch) {
                    nextOuterCount = this->waveOuterBatch;
                }
                DataCopyExtParams nextSelfParams{static_cast<uint16_t>(nextOuterCount),
                    this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyExtParams nextSourceParams{static_cast<uint16_t>(nextOuterCount),
                    this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                LocalTensor<T> nextValueLocal =
                    valueLocal[nextBuffer * valueBufferStride];
                LocalTensor<T> nextSourceLocal =
                    sourceLocal[nextBuffer * sourceBufferStride];
                DataCopyPad(nextValueLocal,
                    this->selfGm[nextOuterBase * this->dimSize],
                    nextSelfParams, padParams);
                DataCopyPad(nextSourceLocal,
                    this->sourceGm[nextOuterBase * this->indexSize],
                    nextSourceParams, sourcePadParams);
                if (nextBuffer == 0) {
                    SetFlag<HardEvent::MTE2_V>(0);
                } else {
                    SetFlag<HardEvent::MTE2_V>(1);
                }
            }

            if (currentBuffer == 0) {
                WaitFlag<HardEvent::MTE2_V>(0);
            } else {
                WaitFlag<HardEvent::MTE2_V>(1);
            }
            LocalTensor<T> currentValueLocal =
                valueLocal[currentBuffer * valueBufferStride];
            LocalTensor<T> currentSourceLocal =
                sourceLocal[currentBuffer * sourceBufferStride];
            for (uint32_t localOuter = 0; localOuter < currentOuterCount;
                 localOuter += this->waveGatherOuterBatch) {
                uint32_t gatherOuterCount = currentOuterCount - localOuter;
                if (gatherOuterCount > this->waveGatherOuterBatch) {
                    gatherOuterCount = this->waveGatherOuterBatch;
                }
                if (this->waveCount > 0) {
                    Gather(waveValueLocal,
                        currentSourceLocal[localOuter * this->waveSourceStride],
                        waveOffsetLocal, 0, gatherOuterCount * waveElements);
                }
                for (uint32_t gatherOuter = 0;
                     gatherOuter < gatherOuterCount; ++gatherOuter) {
                    LocalTensor<T> valueSlice = currentValueLocal[
                        (localOuter + gatherOuter) * this->waveValueStride];
                    LocalTensor<T> gatheredSlice =
                        waveValueLocal[gatherOuter * waveElements];
                    if (useWaveRepeat) {
                        for (uint32_t dimStart = 0; dimStart < this->dimSize;
                             dimStart += elementsPerRepeat) {
                            uint32_t count = this->dimSize - dimStart;
                            if (count > elementsPerRepeat) {
                                count = elementsPerRepeat;
                            }
                            Add(valueSlice[dimStart], gatheredSlice[dimStart],
                                valueSlice[dimStart], static_cast<uint64_t>(count),
                                static_cast<uint8_t>(this->waveCount), repeatParams);
                        }
                    } else {
                        for (uint32_t wave = 0; wave < this->waveCount; ++wave) {
                            Add(valueSlice, valueSlice,
                                gatheredSlice[wave * this->waveStride],
                                this->dimSize);
                        }
                    }
                }
            }
            if (currentBuffer == 0) {
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
            } else {
                SetFlag<HardEvent::V_MTE3>(1);
                WaitFlag<HardEvent::V_MTE3>(1);
            }
            DataCopyExtParams outParams{static_cast<uint16_t>(currentOuterCount),
                this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(this->outputGm[currentOuterBase * this->dimSize],
                currentValueLocal, outParams);
            if (currentBuffer == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(0);
                outputPending0 = true;
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(1);
                outputPending1 = true;
            }
            currentOuterBase = nextOuterBase;
            if (currentOuterBase < outerEnd) {
                currentOuterCount = outerEnd - currentOuterBase;
                if (currentOuterCount > this->waveOuterBatch) {
                    currentOuterCount = this->waveOuterBatch;
                }
                currentBuffer = nextBuffer;
            }
        }
        if (outputPending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        if (outputPending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
    }

    __aicore__ inline bool CanUseContiguousOuterBatch() const
    {
        return this->outerSize >= 2048 && this->dimSize >= 128 &&
            this->indexSize >= 512 && this->indexSize < 2048;
    }

    __aicore__ inline void ProcessContiguousOuterBatch(uint32_t dimStart, uint32_t dimEnd)
    {
        for (uint32_t targetDim = dimStart; targetDim < dimEnd; ++targetDim) {
            uint32_t outerStart = 0;
            while (outerStart < this->outerSize) {
                uint32_t outerCount = this->outerSize - outerStart;
                if (outerCount > kHalfOuterBatch) {
                    outerCount = kHalfOuterBatch;
                }
                this->ComputeContiguousOuterBatch(outerStart, outerCount, targetDim);
                outerStart += outerCount;
            }
        }
    }

    __aicore__ inline void ComputeContiguousOuterBatch(uint32_t outerStart, uint32_t outerCount,
        uint32_t targetDim)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t localElements = outerCount * alignElements;
        LocalTensor<T> valueLocal = this->valueBuf.template Get<T>();
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();

        const uint32_t outputOffset = outerStart * this->dimSize + targetDim;
        DataCopyExtParams selfParams{static_cast<uint16_t>(outerCount),
            static_cast<uint32_t>(sizeof(T)), (this->dimSize - 1) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(valueLocal, this->selfGm[outputOffset], selfParams, padParams);
        PipeBarrier<PIPE_ALL>();

        int32_t sourceIndex = headLocal.GetValue(targetDim);
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = outerStart * this->indexSize + i;
            DataCopyExtParams sourceParams{static_cast<uint16_t>(outerCount),
                static_cast<uint32_t>(sizeof(T)),
                (this->indexSize - 1) * static_cast<uint32_t>(sizeof(T)), 0, 0};
            DataCopyPad(sourceLocal, this->sourceGm[sourceOffset], sourceParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Add(valueLocal, valueLocal, sourceLocal, localElements);
            PipeBarrier<PIPE_ALL>();
            sourceIndex = nextLocal.GetValue(i);
        }

        DataCopyExtParams outParams{static_cast<uint16_t>(outerCount),
            static_cast<uint32_t>(sizeof(T)), 0,
            (this->dimSize - 1) * static_cast<uint32_t>(sizeof(T)), 0};
        DataCopyPad(this->outputGm[outputOffset], valueLocal, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessContiguousDimScatter()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        if (this->outerSize >= blockNum) {
            this->ProcessContiguousDimScatterByOuter(blockNum);
            return;
        }

        uint32_t tileElements = (this->totalSize + blockNum - 1) / blockNum;
        tileElements = ((tileElements + alignElements - 1) / alignElements) * alignElements;
        if (tileElements > kVectorTileElements) {
            tileElements = kVectorTileElements;
        }

        const uint32_t taskCount = (this->totalSize + tileElements - 1) / tileElements;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd = taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t start = task * tileElements;
            uint32_t length = this->totalSize - start;
            if (length > tileElements) {
                length = tileElements;
            }
            this->ComputeContiguousDimScatterTile(start, length);
        }
    }

    __aicore__ inline void ProcessContiguousDimScatterByOuter(uint32_t blockNum)
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter = this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter + (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd = outerStart + baseOuter + (blockIdx < extraOuter ? 1U : 0U);

        for (uint32_t outer = outerStart; outer < outerEnd; ++outer) {
            uint32_t dimStart = 0;
            while (dimStart < this->dimSize) {
                uint32_t count = this->dimSize - dimStart;
                if (count > kVectorTileElements) {
                    count = kVectorTileElements;
                }
                this->ComputeContiguousDimScatterTile(outer * this->dimSize + dimStart, count);
                dimStart += count;
            }
        }
    }

    __aicore__ inline void ComputeContiguousDimScatterTile(uint32_t start, uint32_t length)
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        LocalTensor<T> valueLocal = this->valueBuf.template Get<T>();
        UnaryRepeatParams repeatParams{1, 1, 8, 8};
        SetMaskNorm();

        uint32_t processed = 0;
        while (processed < length) {
            const uint32_t outOffset = start + processed;
            const uint32_t outerIndex = outOffset / this->dimSize;
            const uint32_t dimOffset = outOffset - outerIndex * this->dimSize;
            uint32_t count = length - processed;
            const uint32_t dimRemaining = this->dimSize - dimOffset;
            if (count > dimRemaining) {
                count = dimRemaining;
            }

            this->CopyToLocal(valueLocal, this->selfGm, outOffset, count);
            const uint32_t sourceBase = outerIndex * this->indexSize;
            for (uint32_t i = 0; i < this->indexSize; ++i) {
                int32_t targetDim = this->indexGm.GetValue(i);
                if (targetDim < 0) {
                    targetDim += static_cast<int32_t>(this->dimSize);
                }
                if (targetDim < static_cast<int32_t>(dimOffset) ||
                    targetDim >= static_cast<int32_t>(dimOffset + count)) {
                    continue;
                }
                const uint32_t localIndex = static_cast<uint32_t>(targetDim) - dimOffset;
                const uint32_t blockStart = (localIndex / alignElements) * alignElements;
                const uint32_t lane = localIndex - blockStart;
                const T sourceValue = this->sourceGm.GetValue(sourceBase + i);
                SetVectorMask<T, MaskMode::NORMAL>(0, 1ULL << lane);
                Adds<T, false>(valueLocal[blockStart], valueLocal[blockStart], sourceValue,
                    MASK_PLACEHOLDER, 1, repeatParams);
                PipeBarrier<PIPE_V>();
            }
            this->CopyToGm(this->outputGm, outOffset, valueLocal, count);
            processed += count;
        }
        ResetMask();
    }

    __aicore__ inline void ProcessContiguousDimMasked()
    {
        constexpr uint32_t alignElements = 32 / sizeof(T);
        const uint32_t blockNum = GetBlockNum();
        if (this->outerSize >= blockNum) {
            this->ProcessContiguousDimByOuter(blockNum, alignElements);
            return;
        }

        uint32_t tileElements = (this->totalSize + blockNum - 1) / blockNum;
        tileElements = ((tileElements + alignElements - 1) / alignElements) * alignElements;
        if (tileElements > kVectorTileElements) {
            tileElements = kVectorTileElements;
        }

        const uint32_t taskCount = (this->totalSize + tileElements - 1) / tileElements;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd = taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t start = task * tileElements;
            uint32_t length = this->totalSize - start;
            if (length > tileElements) {
                length = tileElements;
            }
            this->ComputeContiguousDimTile(start, length, alignElements);
        }
    }

    __aicore__ inline void ProcessContiguousDimByOuter(uint32_t blockNum, uint32_t alignElements)
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseOuter = this->outerSize / blockNum;
        const uint32_t extraOuter = this->outerSize - baseOuter * blockNum;
        const uint32_t outerStart =
            blockIdx * baseOuter + (blockIdx < extraOuter ? blockIdx : extraOuter);
        const uint32_t outerEnd = outerStart + baseOuter + (blockIdx < extraOuter ? 1U : 0U);

        for (uint32_t outer = outerStart; outer < outerEnd; ++outer) {
            uint32_t dimStart = 0;
            while (dimStart < this->dimSize) {
                uint32_t count = this->dimSize - dimStart;
                if (count > kVectorTileElements) {
                    count = kVectorTileElements;
                }
                this->ComputeContiguousDimTile(
                    outer * this->dimSize + dimStart, count, alignElements);
                dimStart += count;
            }
        }
    }

    __aicore__ inline void ComputeContiguousDimTile(uint32_t start, uint32_t length,
        uint32_t alignElements)
    {
        LocalTensor<T> valueLocal = this->valueBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();
        UnaryRepeatParams repeatParams{1, 1, 8, 8};
        SetMaskNorm();

        uint32_t processed = 0;
        while (processed < length) {
            const uint32_t outOffset = start + processed;
            const uint32_t outerIndex = outOffset / this->dimSize;
            const uint32_t dimOffset = outOffset - outerIndex * this->dimSize;
            uint32_t count = length - processed;
            const uint32_t dimRemaining = this->dimSize - dimOffset;
            if (count > dimRemaining) {
                count = dimRemaining;
            }

            this->CopyToLocal(valueLocal, this->selfGm, outOffset, count);
            for (uint32_t localIndex = 0; localIndex < count; ++localIndex) {
                const uint32_t targetDim = dimOffset + localIndex;
                int32_t sourceIndex = headLocal.GetValue(targetDim);
                if (sourceIndex < 0) {
                    continue;
                }

                const uint32_t blockStart = (localIndex / alignElements) * alignElements;
                const uint32_t lane = localIndex - blockStart;
                SetVectorMask<T, MaskMode::NORMAL>(0, 1ULL << lane);
                while (sourceIndex >= 0) {
                    const uint32_t i = static_cast<uint32_t>(sourceIndex);
                    const uint32_t sourceOffset = outerIndex * this->indexSize + i;
                    const T sourceValue = this->sourceGm.GetValue(sourceOffset);
                    Adds<T, false>(valueLocal[blockStart], valueLocal[blockStart], sourceValue,
                        MASK_PLACEHOLDER, 1, repeatParams);
                    PipeBarrier<PIPE_V>();
                    sourceIndex = nextLocal.GetValue(i);
                }
            }
            this->CopyToGm(this->outputGm, outOffset, valueLocal, count);
            processed += count;
        }
        ResetMask();
    }

    __aicore__ inline void CopyToLocal(const LocalTensor<T>& local, GlobalTensor<T>& gm, uint32_t offset,
        uint32_t count)
    {
        DataCopyExtParams inParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], inParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyToGm(GlobalTensor<T>& gm, uint32_t offset, const LocalTensor<T>& local,
        uint32_t count)
    {
        DataCopyExtParams outParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeTile(uint32_t start, uint32_t length)
    {
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> valueLocal = this->valueBuf.template Get<T>();
        LocalTensor<int32_t> headLocal = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.template Get<int32_t>();

        uint32_t processed = 0;
        while (processed < length) {
            const uint32_t outOffset = start + processed;
            const uint32_t row = outOffset / this->innerSize;
            const uint32_t innerOffset = outOffset - row * this->innerSize;
            const uint32_t outerIndex = row / this->dimSize;
            const uint32_t targetDim = row - outerIndex * this->dimSize;
            uint32_t count = length - processed;
            const uint32_t rowRemaining = this->innerSize - innerOffset;
            if (count > rowRemaining) {
                count = rowRemaining;
            }

            this->CopyToLocal(valueLocal, this->selfGm, outOffset, count);
            int32_t sourceIndex = headLocal.GetValue(targetDim);
            while (sourceIndex >= 0) {
                const uint32_t i = static_cast<uint32_t>(sourceIndex);
                const uint32_t sourceOffset =
                    (outerIndex * this->indexSize + i) * this->innerSize + innerOffset;
                this->CopyToLocal(sourceLocal, this->sourceGm, sourceOffset, count);
                Add(valueLocal, valueLocal, sourceLocal, count);
                PipeBarrier<PIPE_ALL>();
                sourceIndex = nextLocal.GetValue(i);
            }
            this->CopyToGm(this->outputGm, outOffset, valueLocal, count);
            processed += count;
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> valueBuf;
    TBuf<TPosition::VECCALC> sourceBuf;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    TBuf<TPosition::VECCALC> waveOffsetBuf;
    TBuf<TPosition::VECCALC> waveValueBuf;
    GlobalTensor<T> selfGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> sourceGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int32_t> bucketPlanGm;
    TBuf<TPosition::VECCALC> pairSourceBuf;
    uint32_t totalSize = 0;
    uint32_t blockSize = 0;
    uint32_t outerSize = 0;
    uint32_t dimSize = 0;
    uint32_t indexSize = 0;
    uint32_t innerSize = 1;
    uint32_t waveCount = 0;
    uint32_t waveStride = 0;
    uint32_t zeroElement = 0;
    uint32_t waveOuterBatch = 1;
    uint32_t waveGatherOuterBatch = 1;
    uint32_t waveValueStride = 0;
    uint32_t waveSourceStride = 0;
    uint32_t bucketWidth[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketStride[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketCap[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketMatrixElements[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketPlanBase[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketWaveBase[kHalfCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketGatherOuterBatch = 0U;
    uint32_t bucketMaxMatrixElements = 0U;
    uint32_t bucketTotalMatrixElements = 0U;
    uint32_t bucketRestorePlanBase = 0U;
    uint32_t bucketZeroElement = 0U;
    uint32_t bucketSelfElement = 0U;
    uint32_t bucketValueStride = 0U;
    uint32_t bucketSourceRowStride = 0U;
    uint32_t bucketAlignedIndexWords = 0U;
    uint32_t bucketCachedIndexBase = 0U;
    uint32_t bucketCacheFlagBase = 0U;
    uint32_t bucketCachedPlanBase = 0U;
    bool useBucketPlanCache = false;
    // V440_PRIVATE_FIELDS_BEGIN
    bool usePrivateBucketCache = false;
    uint32_t privateIndexWords = 0U;
    // V440_PRIVATE_FIELDS_END
};

// Independent FP32 count-bucket cache; never shares layout with the FP16 route.
constexpr uint32_t kFloatCountBucketCount = 8U;
constexpr uint32_t kFloatCountBucketGatherOuter = 4U;
constexpr uint32_t kFloatCountBucketUbSafetyBytes = 8192U;
constexpr uint32_t kFloatBucketCacheHeaderWords = 32U;
constexpr uint32_t kFloatBucketCacheFlagStride = 8U;
constexpr uint32_t kFloatBucketCacheMagicA = 0x49414642U;
constexpr uint32_t kFloatBucketCacheMagicB = 0x504C414EU;
constexpr uint32_t kFloatBucketCacheVersion = 6U;

template <typename T>
class KernelIndexAddFloatCountBuckets {
public:
    __aicore__ inline KernelIndexAddFloatCountBuckets() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
        uint32_t dimSize, uint32_t indexSize, uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ T*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ T*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ T*)output, totalSize);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
        this->useBucketPlanCache = innerSize == 1U && outerSize >= 512U &&
            dimSize > 512U && dimSize < 1024U && indexSize >= 4096U &&
            indexSize <= 8192U;
        if (this->useBucketPlanCache) {
            this->bucketAlignedIndexWords =
                AlignUp32(indexSize * sizeof(int32_t)) / sizeof(int32_t);
            this->bucketCachedIndexBase = kFloatBucketCacheHeaderWords;
            this->bucketCacheFlagBase = this->bucketCachedIndexBase +
                this->bucketAlignedIndexWords;
            this->bucketCachedPlanBase = this->bucketCacheFlagBase +
                GetBlockNum() * kFloatBucketCacheFlagStride;
            GM_ADDR userWorkspace = GetUserWorkspace(workspace);
            this->bucketPlanGm.SetGlobalBuffer(
                (__gm__ int32_t*)userWorkspace,
                this->ParallelColdWorkspaceWords());
        }
    }

    __aicore__ inline bool TryProcess()
    {
        this->pipe.InitBuffer(this->headBuf, AlignUp32(this->dimSize * sizeof(int32_t)));
        const uint32_t nextWords = this->bucketAlignedIndexWords + this->CacheScratchWords();
        this->pipe.InitBuffer(this->nextBuf, nextWords * sizeof(int32_t));
        if (!this->PrepareContiguousCountBuckets()) {
            this->InvalidateSharedWorkspaceBeforeFallback();
            return false;
        }
        this->pipe.InitBuffer(this->sourceBuf,
            (this->bucketGatherOuterBatch * this->bucketSourceRowStride +
                this->bucketTotalMatrixElements) * sizeof(T));
        this->pipe.InitBuffer(this->waveValueBuf,
            this->bucketGatherOuterBatch * this->bucketValueStride * sizeof(T));
        this->ProcessContiguousCountBuckets();
        return true;
    }

private:
    __aicore__ inline void InvalidateSharedWorkspaceBeforeFallback()
    {
        // The bucket cache and the established generic cache share workspace.
        // Even a bucket header miss writes cohort flags, and rejected parallel
        // preparation can overwrite a formerly valid generic count/packed plan.
        // Drain every peer before invalidating; no late scratch write or cache
        // clean may survive into the generic cache's next lifetime.
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();
        if (GetBlockIdx() == 0U) {
            LocalTensor<int32_t> headerLocal = this->headBuf.template Get<int32_t>();
            Duplicate(headerLocal, static_cast<int32_t>(0),
                kFloatBucketCacheHeaderWords);
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->bucketPlanGm, headerLocal,
                kFloatBucketCacheHeaderWords);
            PipeBarrier<PIPE_MTE3>();
            DataCacheCleanAndInvalid<int32_t,
                CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        }
        // All blocks must observe the invalid header before constructing the
        // generic kernel. Successful bucket cache/compute paths never enter.
        SyncAll<true>();
    }

    __aicore__ inline uint32_t ParallelColdWorkspaceWords() const
    {
        // Same allocation formula as the unchanged FP32 host shared-plan family.
        const uint32_t blocks = GetBlockNum();
        const uint32_t alignedDim = (this->dimSize + 15U) & ~15U;
        const uint32_t alignedIndex = (this->indexSize + 7U) & ~7U;
        const uint32_t localIndex = (this->indexSize + blocks - 1U) / blocks;
        const uint32_t occupied = localIndex < this->dimSize ? localIndex : this->dimSize;
        const uint32_t packed = (occupied * 8U + 15U) & ~15U;
        return 32U + 2U * blocks * alignedDim + alignedDim + blocks * packed +
            alignedIndex + 64U * alignedDim;
    }

    __aicore__ inline uint32_t CacheScratchWords() const
    {
        const uint32_t localWords = ((this->indexSize + GetBlockNum() - 1U) / GetBlockNum() + 7U) & ~7U;
        const uint32_t flagWords = GetBlockNum() * kFloatBucketCacheFlagStride;
        return localWords > flagWords ? localWords : flagWords;
    }

    __aicore__ inline uint32_t RebuildBucketDerivedMetadata()
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        this->bucketTotalMatrixElements = 0U;
        this->bucketMaxMatrixElements = 0U;
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount;
             ++bucket) {
            this->bucketStride[bucket] = this->bucketWidth[bucket] == 0U
                ? 0U
                : this->bucketWidth[bucket];
            const uint32_t batchStride = this->bucketWidth[bucket] == 0U
                ? 0U
                : AlignUp32(this->bucketGatherOuterBatch *
                      this->bucketWidth[bucket] * sizeof(T)) /
                    sizeof(T);
            this->bucketMatrixElements[bucket] =
                (this->bucketCap[bucket] + 1U) *
                    batchStride;
            this->bucketPlanBase[bucket] =
                this->bucketTotalMatrixElements;
            this->bucketTotalMatrixElements +=
                this->bucketMatrixElements[bucket];
            if (this->bucketMatrixElements[bucket] >
                this->bucketMaxMatrixElements) {
                this->bucketMaxMatrixElements =
                    this->bucketMatrixElements[bucket];
            }
        }
        this->bucketZeroElement = AlignUp32(
            this->indexSize * sizeof(T)) / sizeof(T);
        this->bucketSelfElement = this->bucketZeroElement + alignElements;
        this->bucketValueStride = AlignUp32(
            this->dimSize * sizeof(T)) / sizeof(T);
        this->bucketSourceRowStride =
            this->bucketSelfElement + this->bucketValueStride;
        uint32_t planBase = 0U;
        const uint32_t waveStorageBase =
            this->bucketGatherOuterBatch * this->bucketSourceRowStride;
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount;
             ++bucket) {
            this->bucketPlanBase[bucket] = planBase;
            this->bucketWaveBase[bucket] =
                waveStorageBase + planBase;
            planBase += this->bucketMatrixElements[bucket];
        }
        this->bucketRestorePlanBase = planBase;
        return planBase + this->bucketGatherOuterBatch * this->dimSize;
    }

    __aicore__ inline bool CountBucketLayoutFits(uint32_t planElements) const
    {
        const uint64_t bytes = AlignUp32(this->dimSize * 4U) +
            (this->bucketAlignedIndexWords + this->CacheScratchWords()) * 4ULL +
            AlignUp32(planElements * 4U) +
            (static_cast<uint64_t>(this->bucketGatherOuterBatch) * this->bucketSourceRowStride +
                this->bucketTotalMatrixElements) * 4ULL +
            static_cast<uint64_t>(this->bucketGatherOuterBatch) * this->bucketValueStride * 4ULL;
        return planElements <= kGenericWaveTotalBytes / 4U &&
            bytes + kFloatCountBucketUbSafetyBytes <= kGenericWaveTotalBytes;
    }

    __aicore__ inline bool TryLoadContiguousCountBucketCache()
    {
        if (!this->useBucketPlanCache) { return false; }
        LocalTensor<int32_t> headerLocal = this->headBuf.template Get<int32_t>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        DataCopy(headerLocal, this->bucketPlanGm, kFloatBucketCacheHeaderWords);
        PipeBarrier<PIPE_ALL>();
        bool valid = static_cast<uint32_t>(headerLocal.GetValue(0U)) == kFloatBucketCacheMagicA &&
            static_cast<uint32_t>(headerLocal.GetValue(1U)) == kFloatBucketCacheMagicB &&
            static_cast<uint32_t>(headerLocal.GetValue(2U)) == kFloatBucketCacheVersion &&
            static_cast<uint32_t>(headerLocal.GetValue(3U)) == GetBlockNum() &&
            static_cast<uint32_t>(headerLocal.GetValue(4U)) == this->outerSize &&
            static_cast<uint32_t>(headerLocal.GetValue(5U)) == this->dimSize &&
            static_cast<uint32_t>(headerLocal.GetValue(6U)) == this->indexSize;
        uint32_t planElements = 0U;
        if (valid) {
            uint32_t widthSum = 0U;
            for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
                this->bucketWidth[bucket] = static_cast<uint32_t>(headerLocal.GetValue(8U + bucket));
                this->bucketCap[bucket] = static_cast<uint32_t>(headerLocal.GetValue(16U + bucket));
                valid = valid && this->bucketWidth[bucket] <= this->dimSize;
                widthSum += this->bucketWidth[bucket];
            }
            this->bucketGatherOuterBatch = static_cast<uint32_t>(headerLocal.GetValue(24U));
            valid = valid && widthSum == this->dimSize &&
                this->bucketCap[0] == 2U && this->bucketCap[1] == 4U &&
                this->bucketCap[2] == 6U && this->bucketCap[3] == 8U &&
                this->bucketCap[4] == 10U && this->bucketCap[5] == 12U &&
                this->bucketCap[6] == 16U && this->bucketCap[7] > 0U &&
                this->bucketCap[7] <= 255U && this->bucketGatherOuterBatch > 0U &&
                this->bucketGatherOuterBatch <= kFloatCountBucketGatherOuter;
            if (valid) {
                planElements = this->RebuildBucketDerivedMetadata();
                valid = static_cast<uint32_t>(headerLocal.GetValue(7U)) == planElements &&
                    this->CountBucketLayoutFits(planElements);
            }
        }
        const uint32_t blocks = GetBlockNum();
        const uint32_t core = GetBlockIdx();
        const uint32_t base = this->indexSize / blocks;
        const uint32_t extra = this->indexSize % blocks;
        const uint32_t start = core * base + (core < extra ? core : extra);
        const uint32_t count = base + (core < extra ? 1U : 0U);
        LocalTensor<int32_t> current = this->nextBuf.template Get<int32_t>();
        LocalTensor<int32_t> cached = current[this->bucketAlignedIndexWords];
        if (valid && count > 0U) {
            DataCopyExtParams params{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(current, this->indexGm[start], params, pad);
            DataCopyPad(cached, this->bucketPlanGm[this->bucketCachedIndexBase + start], params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        uint32_t mismatch = valid ? 0U : 1U;
        for (uint32_t i = 0U; valid && i < count; ++i) {
            int32_t target = current.GetValue(i);
            if (target < 0) { target += static_cast<int32_t>(this->dimSize); }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) { target = -1; }
            if (target != cached.GetValue(i)) { mismatch = 1U; break; }
        }
        Duplicate(cached, static_cast<int32_t>(0), kFloatBucketCacheFlagStride);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        cached.SetValue(0U, static_cast<int32_t>(mismatch));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(this->bucketPlanGm[this->bucketCacheFlagBase + core * kFloatBucketCacheFlagStride],
            cached, kFloatBucketCacheFlagStride);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        // Header misses join the same cohort as input mismatches. No core may
        // start cold publication while a late peer is still deciding warm/cold.
        SyncAll<true>();
        DataCopy(cached, this->bucketPlanGm[this->bucketCacheFlagBase],
            blocks * kFloatBucketCacheFlagStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t other = 0U; other < blocks; ++other) {
            if (cached.GetValue(other * kFloatBucketCacheFlagStride) != 0) { return false; }
        }
        const uint32_t alignedPlanWords = (planElements + 7U) & ~7U;
        this->pipe.InitBuffer(this->waveOffsetBuf, alignedPlanWords * 4U);
        DataCopy(this->waveOffsetBuf.template Get<int32_t>(),
            this->bucketPlanGm[this->bucketCachedPlanBase], alignedPlanWords);
        PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline void CommitContiguousCountBucketCache(
        const LocalTensor<int32_t>& normalizedIndexLocal,
        const LocalTensor<uint32_t>& offsetLocal,
        uint32_t planElements)
    {
        if (!this->useBucketPlanCache || GetBlockIdx() != 0U) {
            return;
        }
        const uint32_t alignedPlanWords =
            (planElements + 7U) & ~7U;
        DataCopy(this->bucketPlanGm[this->bucketCachedIndexBase],
            normalizedIndexLocal, this->bucketAlignedIndexWords);
        DataCopy(this->bucketPlanGm[this->bucketCachedPlanBase],
            offsetLocal.template ReinterpretCast<int32_t>(),
            alignedPlanWords);
        PipeBarrier<PIPE_ALL>();
        this->PublishCountBucketHeader(planElements);
    }

    __aicore__ inline void PublishCountBucketHeader(uint32_t planElements)
    {
        if (!this->useBucketPlanCache || GetBlockIdx() != 0U) { return; }
        LocalTensor<int32_t> headerLocal =
            this->headBuf.template Get<int32_t>();
        Duplicate(headerLocal, static_cast<int32_t>(0),
            kFloatBucketCacheHeaderWords);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        headerLocal.SetValue(0U,
            static_cast<int32_t>(kFloatBucketCacheMagicA));
        headerLocal.SetValue(1U,
            static_cast<int32_t>(kFloatBucketCacheMagicB));
        headerLocal.SetValue(2U,
            static_cast<int32_t>(kFloatBucketCacheVersion));
        headerLocal.SetValue(3U,
            static_cast<int32_t>(GetBlockNum()));
        headerLocal.SetValue(4U,
            static_cast<int32_t>(this->outerSize));
        headerLocal.SetValue(5U,
            static_cast<int32_t>(this->dimSize));
        headerLocal.SetValue(6U,
            static_cast<int32_t>(this->indexSize));
        headerLocal.SetValue(7U,
            static_cast<int32_t>(planElements));
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount;
             ++bucket) {
            headerLocal.SetValue(8U + bucket,
                static_cast<int32_t>(this->bucketWidth[bucket]));
            headerLocal.SetValue(16U + bucket,
                static_cast<int32_t>(this->bucketCap[bucket]));
        }
        headerLocal.SetValue(24U,
            static_cast<int32_t>(this->bucketGatherOuterBatch));
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm, headerLocal,
            kFloatBucketCacheHeaderWords);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t,
            CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
    }

    __aicore__ inline uint32_t SelectCountBucket(uint32_t count) const
    {
        if (count <= 2U) { return 0U; }
        if (count <= 4U) { return 1U; }
        if (count <= 6U) { return 2U; }
        if (count <= 8U) { return 3U; }
        if (count <= 10U) { return 4U; }
        if (count <= 12U) { return 5U; }
        if (count <= 16U) { return 6U; }
        return 7U;
    }

    __aicore__ inline bool TryPrepareParallelCountBuckets()
    {
        const uint32_t blocks = GetBlockNum();
        const uint32_t core = GetBlockIdx();
        const uint32_t alignedDim = AlignUp32(this->dimSize * 4U) / 4U;
        const uint32_t countBase = this->bucketCachedPlanBase + kGenericWaveTotalBytes / 4U;
        const uint32_t localCountBase = countBase + alignedDim;
        const uint32_t localStartBase = localCountBase + blocks * alignedDim;
        const uint32_t packedBase = localStartBase + blocks * alignedDim;
        const uint32_t indexGroups = this->bucketAlignedIndexWords / 8U;
        const uint32_t localStride = ((indexGroups + blocks - 1U) / blocks) * 8U;
        const uint32_t packedWords = blocks * localStride;
        const uint32_t canonicalBase = packedBase + packedWords;
        const uint32_t workspaceWords = this->ParallelColdWorkspaceWords();
        const uint32_t nextWords = this->bucketAlignedIndexWords + this->CacheScratchWords();
        const uint32_t dimGroups = alignedDim / 8U;
        const uint32_t baseGroups = dimGroups / blocks;
        const uint32_t extraGroups = dimGroups % blocks;
        const uint32_t groupStart = core * baseGroups + (core < extraGroups ? core : extraGroups);
        const uint32_t ownedGroups = baseGroups + (core < extraGroups ? 1U : 0U);
        const uint32_t dimStart = groupStart * 8U;
        const uint32_t ownedDims = ownedGroups * 8U;
        const uint32_t planDimEnd = dimStart + ownedDims;
        const uint32_t dimEnd = planDimEnd < this->dimSize ? planDimEnd : this->dimSize;
        const uint32_t maxOwnedDims = ((dimGroups + blocks - 1U) / blocks) * 8U;
        const uint32_t maxMatrixWords = blocks * maxOwnedDims;
        // All preliminary capacity tests use global maxima, never local ranges.
        if (canonicalBase > workspaceWords || packedWords > nextWords ||
            2U * localStride + alignedDim > nextWords || maxMatrixWords > nextWords) {
            return false;
        }
        const uint32_t baseIndexGroups = indexGroups / blocks;
        const uint32_t extraIndexGroups = indexGroups % blocks;
        const uint32_t indexStart =
            (core * baseIndexGroups + (core < extraIndexGroups ? core : extraIndexGroups)) * 8U;
        const uint32_t indexWords =
            (baseIndexGroups + (core < extraIndexGroups ? 1U : 0U)) * 8U;
        uint32_t inputCount = indexStart < this->indexSize ? this->indexSize - indexStart : 0U;
        if (inputCount > indexWords) { inputCount = indexWords; }
        LocalTensor<int32_t> counts = this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> indices = this->nextBuf.template Get<int32_t>();
        LocalTensor<int32_t> localStarts = indices[localStride];
        LocalTensor<int32_t> localPacked = localStarts[alignedDim];
        Duplicate(counts, static_cast<int32_t>(0), alignedDim);
        Duplicate(indices, static_cast<int32_t>(0), localStride);
        PipeBarrier<PIPE_ALL>();
        if (inputCount > 0U) {
            DataCopyExtParams params{1, inputCount * 4U, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(indices, this->indexGm[indexStart], params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        // Only this core's contiguous source-index slice is scanned.
        for (uint32_t i = 0U; i < inputCount; ++i) {
            int32_t target = indices.GetValue(i);
            if (target < 0) { target += static_cast<int32_t>(this->dimSize); }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                indices.SetValue(i, -1);
                continue;
            }
            indices.SetValue(i, target);
            counts.SetValue(target, counts.GetValue(target) + 1);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm[localCountBase + core * alignedDim], counts, alignedDim);
        if (indexWords > 0U) {
            DataCopy(this->bucketPlanGm[this->bucketCachedIndexBase + indexStart], indices, indexWords);
        }
        // counts becomes the per-target stable pack cursor only after its
        // histogram DMA has finished reading the original values.
        PipeBarrier<PIPE_ALL>();
        uint32_t packedCount = 0U;
        for (uint32_t dim = 0U; dim < alignedDim; ++dim) {
            const uint32_t count = static_cast<uint32_t>(counts.GetValue(dim));
            localStarts.SetValue(dim, static_cast<int32_t>(packedCount));
            counts.SetValue(dim, static_cast<int32_t>(packedCount));
            packedCount += count;
        }
        Duplicate(localPacked, static_cast<int32_t>(0), localStride);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0U; i < inputCount; ++i) {
            const int32_t target = indices.GetValue(i);
            if (target >= 0) {
                const uint32_t cursor = static_cast<uint32_t>(counts.GetValue(target));
                localPacked.SetValue(cursor, static_cast<int32_t>(indexStart + i));
                counts.SetValue(target, static_cast<int32_t>(cursor + 1U));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->bucketPlanGm[localStartBase + core * alignedDim], localStarts, alignedDim);
        DataCopy(this->bucketPlanGm[packedBase + core * localStride], localPacked, localStride);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();

        // Target owners reduce only their histogram columns. Reuse the dead
        // local source buffers; no whole-M branch scan remains on this path.
        const uint32_t matrixWords = blocks * ownedDims;
        LocalTensor<int32_t> countMatrix = indices;
        Duplicate(counts, static_cast<int32_t>(0), alignedDim);
        if (ownedDims > 0U) {
            DataCopyExtParams params{static_cast<uint16_t>(blocks), ownedDims * 4U,
                (alignedDim - ownedDims) * 4U, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(countMatrix, this->bucketPlanGm[localCountBase + dimStart], params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        if (ownedDims > 0U) {
            for (uint32_t sourceCore = 0U; sourceCore < blocks; ++sourceCore) {
                Add(counts[dimStart], counts[dimStart], countMatrix[sourceCore * ownedDims], ownedDims);
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->bucketPlanGm[countBase + dimStart], counts[dimStart], ownedDims);
        }
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();
        DataCopy(counts, this->bucketPlanGm[countBase], alignedDim);
        PipeBarrier<PIPE_ALL>();
        this->bucketCap[0] = 2U; this->bucketCap[1] = 4U;
        this->bucketCap[2] = 6U; this->bucketCap[3] = 8U;
        this->bucketCap[4] = 10U; this->bucketCap[5] = 12U;
        this->bucketCap[6] = 16U;
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
            this->bucketWidth[bucket] = 0U;
        }
        uint32_t maxCount = 0U;
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(counts.GetValue(dim));
            if (count > maxCount) { maxCount = count; }
            ++this->bucketWidth[this->SelectCountBucket(count)];
        }
        if (maxCount == 0U || maxCount > 255U) { return false; }
        this->bucketCap[7] = maxCount;
        uint32_t planElements = 0U;
        this->bucketGatherOuterBatch = kFloatCountBucketGatherOuter;
        while (this->bucketGatherOuterBatch > 0U) {
            planElements = this->RebuildBucketDerivedMetadata();
            if (this->CountBucketLayoutFits(planElements)) { break; }
            --this->bucketGatherOuterBatch;
        }
        if (this->bucketGatherOuterBatch == 0U) { return false; }
        const uint32_t alignedPlanWords = (planElements + 7U) & ~7U;
        const uint32_t canonicalStride = (maxCount + 7U) & ~7U;
        // Both matrix slices and owned canonical lists share the already
        // budgeted final-plan UB. All cores use the worst owner for eligibility.
        if (static_cast<uint64_t>(canonicalBase) +
                static_cast<uint64_t>(alignedDim) * canonicalStride > workspaceWords ||
            2ULL * maxMatrixWords +
                static_cast<uint64_t>(maxOwnedDims) * canonicalStride > alignedPlanWords) {
            return false;
        }
        this->pipe.InitBuffer(this->waveOffsetBuf, alignedPlanWords * 4U);
        LocalTensor<int32_t> planTemp = this->waveOffsetBuf.template Get<int32_t>();
        countMatrix = planTemp;
        LocalTensor<int32_t> startMatrix = planTemp[matrixWords];
        LocalTensor<int32_t> canonicalOwned = planTemp[2U * matrixWords];
        const uint32_t ownedWords = ownedDims * canonicalStride;
        if (ownedWords > 0U) { Duplicate(canonicalOwned, static_cast<int32_t>(-1), ownedWords); }
        DataCopy(indices, this->bucketPlanGm[packedBase], packedWords);
        if (ownedDims > 0U) {
            DataCopyExtParams params{static_cast<uint16_t>(blocks), ownedDims * 4U,
                (alignedDim - ownedDims) * 4U, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(countMatrix, this->bucketPlanGm[localCountBase + dimStart], params, pad);
            DataCopyPad(startMatrix, this->bucketPlanGm[localStartBase + dimStart], params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t dim = dimStart; dim < dimEnd; ++dim) {
            const uint32_t localDim = dim - dimStart;
            uint32_t rank = 0U;
            // Contiguous input partitions are concatenated in core order;
            // each partition was packed in input order. This is exactly the
            // original ascending source-index order, never a bank permutation.
            for (uint32_t sourceCore = 0U; sourceCore < blocks; ++sourceCore) {
                const uint32_t count = static_cast<uint32_t>(
                    countMatrix.GetValue(sourceCore * ownedDims + localDim));
                const uint32_t start = static_cast<uint32_t>(
                    startMatrix.GetValue(sourceCore * ownedDims + localDim));
                for (uint32_t i = 0U; i < count; ++i) {
                    canonicalOwned.SetValue(localDim * canonicalStride + rank,
                        indices.GetValue(sourceCore * localStride + start + i));
                    ++rank;
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (ownedWords > 0U) {
            DataCopy(this->bucketPlanGm[canonicalBase + dimStart * canonicalStride], canonicalOwned, ownedWords);
        }
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();
        LocalTensor<int32_t> canonical = this->waveOffsetBuf.template Get<int32_t>();

        // The gathered packed-index buffer is now dead. Reuse it for the inverse
        // (bucket,column)->target map; counts retain count/bucket/column tags.
        uint32_t bucketStart[kFloatCountBucketCount] = {0U};
        uint32_t bucketCursor[kFloatCountBucketCount] = {0U};
        for (uint32_t bucket = 1U; bucket < kFloatCountBucketCount; ++bucket) {
            bucketStart[bucket] = bucketStart[bucket - 1U] + this->bucketWidth[bucket - 1U];
        }
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(counts.GetValue(dim));
            const uint32_t bucket = this->SelectCountBucket(count);
            const uint32_t column = bucketCursor[bucket]++;
            indices.SetValue(bucketStart[bucket] + column, static_cast<int32_t>(dim));
            counts.SetValue(dim, static_cast<int32_t>((bucket << 24U) | (column << 8U) | count));
        }
        const uint32_t planGroups = alignedPlanWords / 8U;
        const uint32_t basePlanGroups = planGroups / blocks;
        const uint32_t extraPlanGroups = planGroups % blocks;
        const uint32_t planStart =
            (core * basePlanGroups + (core < extraPlanGroups ? core : extraPlanGroups)) * 8U;
        const uint32_t planWords = (basePlanGroups + (core < extraPlanGroups ? 1U : 0U)) * 8U;
        const uint32_t sourceRowBytes = this->bucketSourceRowStride * 4U;
        for (uint32_t local = 0U; local < planWords; ++local) {
            const uint32_t word = planStart + local;
            uint32_t offset = 0U;
            if (word < this->bucketRestorePlanBase) {
                uint32_t bucket = 0U;
                while (bucket + 1U < kFloatCountBucketCount &&
                    word >= this->bucketPlanBase[bucket] + this->bucketMatrixElements[bucket]) {
                    ++bucket;
                }
                const uint32_t width = this->bucketWidth[bucket];
                const uint32_t batchStride = AlignUp32(this->bucketGatherOuterBatch * width * 4U) / 4U;
                const uint32_t inBucket = word - this->bucketPlanBase[bucket];
                const uint32_t rank = inBucket / batchStride;
                const uint32_t batchColumn = inBucket % batchStride;
                offset = this->bucketZeroElement * 4U;
                if (batchColumn < this->bucketGatherOuterBatch * width) {
                    const uint32_t outer = batchColumn / width;
                    const uint32_t column = batchColumn % width;
                    const uint32_t dim = static_cast<uint32_t>(indices.GetValue(bucketStart[bucket] + column));
                    const uint32_t count = static_cast<uint32_t>(counts.GetValue(dim)) & 0xffU;
                    if (rank == 0U) {
                        offset = (this->bucketSelfElement + dim) * 4U;
                    } else if (rank <= count) {
                        offset = static_cast<uint32_t>(this->bucketPlanGm.GetValue(
                            canonicalBase + dim * canonicalStride + rank - 1U)) * 4U;
                    }
                    offset += outer * sourceRowBytes;
                }
            } else if (word < planElements) {
                const uint32_t restore = word - this->bucketRestorePlanBase;
                const uint32_t outer = restore / this->dimSize;
                const uint32_t dim = restore % this->dimSize;
                const uint32_t tag = static_cast<uint32_t>(counts.GetValue(dim));
                const uint32_t bucket = (tag >> 24U) & 0xffU;
                const uint32_t column = (tag >> 8U) & 0xffffU;
                offset = (this->bucketWaveBase[bucket] + outer * this->bucketStride[bucket] + column) * 4U;
            }
            canonical.SetValue(local, static_cast<int32_t>(offset));
        }
        PipeBarrier<PIPE_ALL>();
        if (planWords > 0U) {
            DataCopy(this->bucketPlanGm[this->bucketCachedPlanBase + planStart], canonical, planWords);
        }
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->bucketPlanGm);
        SyncAll<true>();
        DataCopy(canonical, this->bucketPlanGm[this->bucketCachedPlanBase], alignedPlanWords);
        PipeBarrier<PIPE_ALL>();
        // Only now are cached indices and every final plan block complete.
        this->PublishCountBucketHeader(planElements);
        return true;
    }

    __aicore__ inline bool PrepareContiguousCountBuckets()
    {
        // Preserve FP32 source-order additions while grouping by count.
        // The established float shared-plan kernel handles resource overflow.
        if (this->dimSize <= 512U || this->dimSize >= 1024U) {
            return false;
        }
        if (this->TryLoadContiguousCountBucketCache()) {
            return true;
        }

        if (this->TryPrepareParallelCountBuckets()) {
            return true;
        }

        constexpr uint32_t alignElements = 32U / sizeof(T);
        LocalTensor<int32_t> countLocal =
            this->headBuf.template Get<int32_t>();
        LocalTensor<int32_t> targetLocal =
            this->nextBuf.template Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();

        uint32_t maxCount = 0U;
        for (uint32_t i = 0U; i < this->indexSize; ++i) {
            int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                targetLocal.SetValue(i, -1);
                continue;
            }
            targetLocal.SetValue(i, target);
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t count = static_cast<uint32_t>(
                countLocal.GetValue(targetDim)) + 1U;
            countLocal.SetValue(targetDim, static_cast<int32_t>(count));
            if (count > maxCount) {
                maxCount = count;
            }
        }
        if (maxCount == 0U || maxCount > 255U) {
            return false;
        }

        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
            this->bucketWidth[bucket] = 0U;
            this->bucketStride[bucket] = 0U;
            this->bucketMatrixElements[bucket] = 0U;
            this->bucketPlanBase[bucket] = 0U;
        }
        this->bucketCap[0] = 2U;
        this->bucketCap[1] = 4U;
        this->bucketCap[2] = 6U;
        this->bucketCap[3] = 8U;
        this->bucketCap[4] = 10U;
        this->bucketCap[5] = 12U;
        this->bucketCap[6] = 16U;
        this->bucketCap[7] = maxCount;
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(countLocal.GetValue(dim));
            ++this->bucketWidth[this->SelectCountBucket(count)];
        }

        this->bucketMaxMatrixElements = 0U;
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
            if (this->bucketWidth[bucket] == 0U) {
                continue;
            }
            this->bucketStride[bucket] = this->bucketWidth[bucket];
        }

        this->bucketZeroElement = AlignUp32(this->indexSize * sizeof(T)) /
            sizeof(T);
        this->bucketSelfElement = this->bucketZeroElement + alignElements;
        this->bucketValueStride = AlignUp32(this->dimSize * sizeof(T)) /
            sizeof(T);
        this->bucketSourceRowStride =
            this->bucketSelfElement + this->bucketValueStride;

        const uint64_t headBytes = AlignUp32(this->dimSize * sizeof(int32_t));
        const uint64_t nextBytes = (this->bucketAlignedIndexWords + this->CacheScratchWords()) * sizeof(int32_t);
        this->bucketGatherOuterBatch = kFloatCountBucketGatherOuter;
        while (this->bucketGatherOuterBatch > 0U) {
            uint64_t batchPlanElements = 0U;
            for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount;
                 ++bucket) {
                if (this->bucketWidth[bucket] == 0U) {
                    continue;
                }
                const uint32_t batchStride = AlignUp32(
                    this->bucketGatherOuterBatch * this->bucketWidth[bucket] *
                        sizeof(T)) /
                    sizeof(T);
                batchPlanElements +=
                    (this->bucketCap[bucket] + 1U) * batchStride;
            }
            const uint64_t planBytes = AlignUp32(
                (batchPlanElements + this->bucketGatherOuterBatch * this->dimSize) * sizeof(uint32_t));
            const uint64_t sourceBytes =
                (static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                        this->bucketSourceRowStride +
                    batchPlanElements) * sizeof(T);
            const uint64_t waveBytes =
                static_cast<uint64_t>(this->bucketGatherOuterBatch) *
                this->bucketValueStride * sizeof(T);
            if (headBytes + nextBytes + planBytes + sourceBytes + waveBytes +
                    kFloatCountBucketUbSafetyBytes <= kGenericWaveTotalBytes) {
                break;
            }
            --this->bucketGatherOuterBatch;
        }
        if (this->bucketGatherOuterBatch == 0U) {
            return false;
        }

        uint32_t totalPlanElements = 0U;
        const uint32_t waveStorageBase =
            this->bucketGatherOuterBatch * this->bucketSourceRowStride;
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
            const uint32_t batchStride = this->bucketWidth[bucket] == 0U
                ? 0U
                : AlignUp32(this->bucketGatherOuterBatch *
                      this->bucketWidth[bucket] * sizeof(T)) /
                    sizeof(T);
            this->bucketMatrixElements[bucket] =
                (this->bucketCap[bucket] + 1U) * batchStride;
            this->bucketPlanBase[bucket] = totalPlanElements;
            this->bucketWaveBase[bucket] =
                waveStorageBase + totalPlanElements;
            totalPlanElements += this->bucketMatrixElements[bucket];
            if (this->bucketMatrixElements[bucket] >
                this->bucketMaxMatrixElements) {
                this->bucketMaxMatrixElements =
                    this->bucketMatrixElements[bucket];
            }
        }
        this->bucketTotalMatrixElements = totalPlanElements;
        this->bucketRestorePlanBase = totalPlanElements;
        const uint32_t allPlanElements = totalPlanElements +
            this->bucketGatherOuterBatch * this->dimSize;
        this->pipe.InitBuffer(this->waveOffsetBuf,
            AlignUp32(allPlanElements * sizeof(uint32_t)));
        LocalTensor<uint32_t> offsetLocal =
            this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t zeroByteOffset =
            this->bucketZeroElement * sizeof(T);
        Duplicate(offsetLocal, zeroByteOffset, allPlanElements);
        PipeBarrier<PIPE_ALL>();

        uint32_t bucketCursor[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
        for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
            const uint32_t count = static_cast<uint32_t>(countLocal.GetValue(dim));
            const uint32_t bucket = this->SelectCountBucket(count);
            const uint32_t column = bucketCursor[bucket]++;
            const uint32_t encoded = (bucket << 24U) | (column << 8U);
            countLocal.SetValue(dim, static_cast<int32_t>(encoded));
            offsetLocal.SetValue(this->bucketPlanBase[bucket] + column,
                (this->bucketSelfElement + dim) * sizeof(T));
        }
        for (uint32_t i = 0U; i < this->indexSize; ++i) {
            const int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t encoded = static_cast<uint32_t>(
                countLocal.GetValue(targetDim));
            const uint32_t bucket = (encoded >> 24U) & 0xFFU;
            const uint32_t column = (encoded >> 8U) & 0xFFFFU;
            const uint32_t rank = encoded & 0xFFU;
            const uint32_t batchStride = AlignUp32(
                this->bucketGatherOuterBatch * this->bucketWidth[bucket] *
                    sizeof(T)) /
                sizeof(T);
            offsetLocal.SetValue(this->bucketPlanBase[bucket] +
                    (rank + 1U) * batchStride + column,
                i * sizeof(T));
            countLocal.SetValue(targetDim,
                static_cast<int32_t>((encoded & ~0xFFU) | (rank + 1U)));
        }
        const uint32_t sourceRowBytes =
            this->bucketSourceRowStride * sizeof(T);
        for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount;
             ++bucket) {
            const uint32_t stride = this->bucketStride[bucket];
            if (stride == 0U) {
                continue;
            }
            const uint32_t batchStride = AlignUp32(
                this->bucketGatherOuterBatch * stride * sizeof(T)) /
                sizeof(T);
            for (uint32_t rank = 0U; rank <= this->bucketCap[bucket];
                 ++rank) {
                const uint32_t rankBase = this->bucketPlanBase[bucket] +
                    rank * batchStride;
                for (uint32_t outer = 1U;
                     outer < this->bucketGatherOuterBatch; ++outer) {
                    for (uint32_t column = 0U; column < stride; ++column) {
                        offsetLocal.SetValue(
                            rankBase + outer * stride + column,
                            offsetLocal.GetValue(rankBase + column) +
                                outer * sourceRowBytes);
                    }
                }
            }
        }
        for (uint32_t outer = 0U; outer < this->bucketGatherOuterBatch;
             ++outer) {
            const uint32_t rowBase = outer * this->bucketSourceRowStride;
            for (uint32_t dim = 0U; dim < this->dimSize; ++dim) {
                const int32_t encodedValue = countLocal.GetValue(dim);
                uint32_t restoreElement =
                    rowBase + this->bucketSelfElement + dim;
                if (encodedValue >= 0) {
                    const uint32_t encoded =
                        static_cast<uint32_t>(encodedValue);
                    const uint32_t bucket = (encoded >> 24U) & 0xFFU;
                    const uint32_t column = (encoded >> 8U) & 0xFFFFU;
                    restoreElement = this->bucketWaveBase[bucket] +
                        outer * this->bucketStride[bucket] + column;
                }
                offsetLocal.SetValue(this->bucketRestorePlanBase +
                        outer * this->dimSize + dim,
                    restoreElement * sizeof(T));
            }
        }
        const uint32_t planElements =
            this->bucketRestorePlanBase +
                this->bucketGatherOuterBatch * this->dimSize;
        const uint32_t alignedPlanWords =
            (planElements + 7U) & ~7U;
        for (uint32_t tail = planElements; tail < alignedPlanWords;
             ++tail) {
            offsetLocal.SetValue(tail, 0U);
        }
        PipeBarrier<PIPE_ALL>();
        this->CommitContiguousCountBucketCache(
            targetLocal, offsetLocal, planElements);
        return true;
    }

    __aicore__ inline void ProcessContiguousCountBuckets()
    {
        constexpr uint32_t alignElements = 32U / sizeof(T);
        constexpr uint32_t elementsPerRepeat = 256U / sizeof(T);
        const uint32_t baseOuter = this->outerSize / GetBlockNum();
        const uint32_t extraOuter = this->outerSize - baseOuter * GetBlockNum();
        const uint32_t core = GetBlockIdx();
        const uint32_t outerStart = core * baseOuter + (core < extraOuter ? core : extraOuter);
        const uint32_t outerEnd = outerStart + baseOuter + (core < extraOuter ? 1U : 0U);
        if (outerStart >= outerEnd) { return; }
        LocalTensor<T> sourceLocal = this->sourceBuf.template Get<T>();
        LocalTensor<T> outputLocal = this->waveValueBuf.template Get<T>();
        LocalTensor<uint32_t> offsetLocal = this->waveOffsetBuf.template Get<uint32_t>();
        const uint32_t rowStride = this->bucketSourceRowStride;
        const uint32_t rawElements = this->bucketGatherOuterBatch * rowStride;
        const uint32_t sourceDstStride =
            (rowStride - this->bucketZeroElement - alignElements) / alignElements;
        const uint32_t selfDstStride =
            (rowStride - this->bucketValueStride) / alignElements;
        DataCopyPadExtParams<T> sourcePad{true, 0, static_cast<uint8_t>(alignElements), static_cast<T>(0)};
        DataCopyPadExtParams<T> selfPad{false, 0, 0, static_cast<T>(0)};
        uint32_t outer = outerStart;
        uint32_t count = outerEnd - outer;
        if (count > this->bucketGatherOuterBatch) { count = this->bucketGatherOuterBatch; }
        DataCopyExtParams firstSource{static_cast<uint16_t>(count),
            this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, sourceDstStride, 0};
        DataCopyExtParams firstSelf{static_cast<uint16_t>(count),
            this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, selfDstStride, 0};
        DataCopyPad(sourceLocal, this->sourceGm[outer * this->indexSize], firstSource, sourcePad);
        DataCopyPad(sourceLocal[this->bucketSelfElement], this->selfGm[outer * this->dimSize], firstSelf, selfPad);
        SetFlag<HardEvent::MTE2_V>(0);
        bool outputPending = false;
        while (outer < outerEnd) {
            WaitFlag<HardEvent::MTE2_V>(0);
            const uint32_t nextOuter = outer + count;
            uint32_t nextCount = outerEnd - nextOuter;
            if (nextCount > this->bucketGatherOuterBatch) { nextCount = this->bucketGatherOuterBatch; }
            Gather(sourceLocal[rawElements], sourceLocal, offsetLocal, 0, this->bucketTotalMatrixElements);
            PipeBarrier<PIPE_V>();
            if (nextCount > 0U) {
                // Every target, including zero-update targets, now resides in
                // the matrix. Release the raw rows after this first Gather.
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                DataCopyExtParams nextSource{static_cast<uint16_t>(nextCount),
                    this->indexSize * static_cast<uint32_t>(sizeof(T)), 0, sourceDstStride, 0};
                DataCopyExtParams nextSelf{static_cast<uint16_t>(nextCount),
                    this->dimSize * static_cast<uint32_t>(sizeof(T)), 0, selfDstStride, 0};
                DataCopyPad(sourceLocal, this->sourceGm[nextOuter * this->indexSize], nextSource, sourcePad);
                DataCopyPad(sourceLocal[this->bucketSelfElement], this->selfGm[nextOuter * this->dimSize], nextSelf, selfPad);
                SetFlag<HardEvent::MTE2_V>(0);
            }
            for (uint32_t bucket = 0U; bucket < kFloatCountBucketCount; ++bucket) {
                const uint32_t width = this->bucketWidth[bucket];
                if (width == 0U) { continue; }
                const uint32_t stride = AlignUp32(this->bucketGatherOuterBatch * width * sizeof(T)) / sizeof(T);
                const uint32_t active = count * width;
                LocalTensor<T> value = sourceLocal[this->bucketWaveBase[bucket]];
                const BinaryRepeatParams repeats{1,1,1,0,static_cast<uint8_t>(stride * sizeof(T) / 32U),0};
                for (uint32_t column = 0U; column < active; column += elementsPerRepeat) {
                    uint32_t lanes = active - column;
                    if (lanes > elementsPerRepeat) { lanes = elementsPerRepeat; }
                    Add(value[column], value[stride + column], value[column],
                        static_cast<uint64_t>(lanes), static_cast<uint8_t>(this->bucketCap[bucket]), repeats);
                }
                PipeBarrier<PIPE_V>();
            }
            if (outputPending) { WaitFlag<HardEvent::MTE3_V>(0); }
            Gather(outputLocal, sourceLocal, offsetLocal[this->bucketRestorePlanBase], 0, count * this->dimSize);
            DataCopyExtParams out{1, count * this->dimSize * static_cast<uint32_t>(sizeof(T)), 0,0,0};
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(this->outputGm[outer * this->dimSize], outputLocal, out);
            SetFlag<HardEvent::MTE3_V>(0);
            outputPending = true;
            outer = nextOuter;
            count = nextCount;
        }
        if (outputPending) { WaitFlag<HardEvent::MTE3_V>(0); }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuf;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    TBuf<TPosition::VECCALC> waveOffsetBuf;
    TBuf<TPosition::VECCALC> waveValueBuf;
    GlobalTensor<T> selfGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> sourceGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int32_t> bucketPlanGm;
    uint32_t totalSize = 0;
    uint32_t blockSize = 0;
    uint32_t outerSize = 0;
    uint32_t dimSize = 0;
    uint32_t indexSize = 0;
    uint32_t innerSize = 1;
    uint32_t bucketWidth[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketStride[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketCap[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketMatrixElements[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketPlanBase[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketWaveBase[kFloatCountBucketCount] = {0U, 0U, 0U, 0U};
    uint32_t bucketGatherOuterBatch = 0U;
    uint32_t bucketMaxMatrixElements = 0U;
    uint32_t bucketTotalMatrixElements = 0U;
    uint32_t bucketRestorePlanBase = 0U;
    uint32_t bucketZeroElement = 0U;
    uint32_t bucketSelfElement = 0U;
    uint32_t bucketValueStride = 0U;
    uint32_t bucketSourceRowStride = 0U;
    uint32_t bucketAlignedIndexWords = 0U;
    uint32_t bucketCachedIndexBase = 0U;
    uint32_t bucketCacheFlagBase = 0U;
    uint32_t bucketCachedPlanBase = 0U;
    bool useBucketPlanCache = false;
};

class KernelIndexAddBfloatOnePassWave {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize, uint32_t dimSize, uint32_t indexSize,
        uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ bfloat16_t*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ bfloat16_t*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output, totalSize);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
        this->workspace = workspace;
    }

    __aicore__ inline bool TryProcess()
    {
        if (!this->PrepareOnePassWavefront()) {
            return false;
        }
        const uint32_t stride = this->groupedVectorStride;
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        // Gatherb reorders eight target rows per repeat, including safe tail rows.
        this->pipe.InitBuffer(this->valueBuf, paddedDim * stride * 2U);
        this->pipe.InitBuffer(this->valueFloatBuf, paddedDim * stride * 4U);
        this->pipe.InitBuffer(this->sourceBuf, (this->indexSize + 1U) * stride * 2U);
        this->pipe.InitBuffer(this->sourceFloatBuf, paddedDim * stride * 4U);
        this->pipe.InitBuffer(this->waveOffsetBuf, paddedDim * stride * 2U);
        this->ProcessBfloatWavefront();
        return true;
    }

private:
    __aicore__ inline uint32_t ReorderExtraRows(uint32_t stride) const
    {
        // First FP32/BF16 block maps already occupy the permutation/inverse.
        return stride / 8U + stride / 16U - 2U;
    }

    __aicore__ inline bool PrepareOnePassWavefront()
    {
        if (this->totalSize == 0U || this->blockSize == 0U ||
            this->outerSize <= 1U || this->innerSize <= 1U ||
            this->dimSize < 32U || this->dimSize > 255U ||
            this->indexSize == 0U || this->indexSize > 4096U ||
            this->outerSize > GetBlockNum()) {
            return false;
        }
        const uint32_t tiles = GetBlockNum() / this->outerSize;
        const uint32_t inner = (this->innerSize + tiles - 1U) / tiles;
        if (inner > 512U) {
            return false;
        }
        const uint32_t stride = AlignUp32(inner * 2U) / 2U;
        // Float reorder uses dstRepStride=stride in the byte-sized API constructor.
        if (stride > 128U) {
            return false;
        }
        const uint32_t actualTiles = (this->innerSize + inner - 1U) / inner;
        const uint32_t lastCount = this->innerSize - (actualTiles - 1U) * inner;
        if (AlignUp32(lastCount * 2U) / 2U != stride) {
            return false;
        }
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        const uint32_t alignedIndex = AlignUp32(this->indexSize * 4U) / 4U;
        const uint32_t flagWords = GetBlockNum() * 8U;
        this->cachedLocalBase = alignedIndex > flagWords ? alignedIndex : flagWords;
        const uint32_t maxSlice = (this->indexSize + GetBlockNum() - 1U) / GetBlockNum();
        this->cacheSliceWords = AlignUp32(maxSlice * 4U) / 4U;
        if (this->cacheSliceWords < 8U) {
            this->cacheSliceWords = 8U;
        }
        this->cacheFlagBase = 16U + alignedIndex;
        this->cachePlanBase = this->cacheFlagBase + flagWords;
        const uint32_t planRows = 18U + this->ReorderExtraRows(stride);
        if (this->cachePlanBase + planRows * paddedDim + 16U > 65536U / 4U) {
            return false;
        }
        const uint64_t bytes = AlignUp32(this->dimSize * 4U) +
            static_cast<uint64_t>(this->cachedLocalBase + this->cacheSliceWords) * 4U +
            static_cast<uint64_t>(paddedDim) * stride * 10U +
            static_cast<uint64_t>(this->indexSize + 1U) * stride * 2U +
            static_cast<uint64_t>(paddedDim) * stride * 2U +
            (planRows * paddedDim + 40U + stride / 2U) * 4U;
        if (bytes + 8192U > kGenericWaveTotalBytes) {
            return false;
        }
        this->pipe.InitBuffer(this->headBuf, paddedDim * 4U);
        this->pipe.InitBuffer(this->nextBuf, (this->cachedLocalBase + this->cacheSliceWords) * 4U);
        this->pipe.InitBuffer(this->waveValueBuf, (planRows * paddedDim + 40U + stride / 2U) * 4U);
        this->groupedVectorInner = inner;
        this->groupedVectorStride = stride;
        this->planGm.SetGlobalBuffer((__gm__ int32_t*)GetUserWorkspace(this->workspace), 65536U / 4U);
        if (this->TryRestoreVerifiedPlan()) {
            return true;
        }
        LocalTensor<int32_t> counts = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> indices = this->nextBuf.Get<int32_t>();
        LocalTensor<int32_t> plan = this->waveValueBuf.Get<int32_t>();
        Duplicate(counts, static_cast<int32_t>(0), paddedDim);
        Duplicate(plan, static_cast<int32_t>(this->indexSize * stride * 2U), 16U * paddedDim);
        DataCopyExtParams params{1, this->indexSize * 4U, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(indices, this->indexGm, params, pad);
        PipeBarrier<PIPE_ALL>();
        uint32_t maxCount = 0U;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = indices.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                continue;
            }
            const uint32_t rank = static_cast<uint32_t>(counts.GetValue(target));
            if (rank >= 16U) {
                return false;
            }
            counts.SetValue(target, static_cast<int32_t>(rank + 1U));
            plan.SetValue(rank * paddedDim + static_cast<uint32_t>(target),
                static_cast<int32_t>(i * stride * 2U));
            if (rank + 1U > maxCount) {
                maxCount = rank + 1U;
            }
        }
        if (maxCount * this->dimSize > 4U * this->indexSize + this->dimSize) {
            return false;
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        this->groupedVectorInner = inner;
        this->groupedVectorStride = stride;
        this->groupedVectorRows = maxCount;
        this->SortWaveTargets(counts, plan, paddedDim, stride);
        this->CommitVerifiedPlan();
        return true;
    }

    __aicore__ inline void SortWaveTargets(const LocalTensor<int32_t>& counts,
        const LocalTensor<int32_t>& plan, uint32_t paddedDim, uint32_t stride)
    {
        LocalTensor<int32_t> permutation = plan[16U * paddedDim];
        LocalTensor<int32_t> inverse = plan[17U * paddedDim];
        LocalTensor<int32_t> active = plan[18U * paddedDim];
        LocalTensor<int32_t> cursor = active[16U];
        Duplicate(permutation, static_cast<int32_t>(0), 2U * paddedDim + 40U);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t target = 0U; target < this->dimSize; ++target) {
            const uint32_t count = static_cast<uint32_t>(counts.GetValue(target));
            cursor.SetValue(count, cursor.GetValue(count) + 1);
        }
        uint32_t prefix = 0U;
        for (int32_t count = 16; count >= 0; --count) {
            const uint32_t size = static_cast<uint32_t>(cursor.GetValue(count));
            cursor.SetValue(count, static_cast<int32_t>(prefix));
            prefix += size;
            if (count > 0) {
                active.SetValue(static_cast<uint32_t>(count - 1),
                    static_cast<int32_t>(prefix));
            }
        }
        // Stable counting sort changes target order, never a target's source order.
        for (uint32_t target = 0U; target < this->dimSize; ++target) {
            const uint32_t count = static_cast<uint32_t>(counts.GetValue(target));
            const uint32_t sorted = static_cast<uint32_t>(cursor.GetValue(count));
            cursor.SetValue(count, static_cast<int32_t>(sorted + 1U));
            permutation.SetValue(sorted, static_cast<int32_t>(target * 4U));
            inverse.SetValue(target, static_cast<int32_t>(sorted * stride * 2U));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        for (uint32_t wave = 0U; wave < this->groupedVectorRows; ++wave) {
            Gather(counts.ReinterpretCast<uint32_t>(),
                plan[wave * paddedDim].ReinterpretCast<uint32_t>(),
                permutation.ReinterpretCast<uint32_t>(), 0, paddedDim);
            PipeBarrier<PIPE_V>();
            Adds(plan[wave * paddedDim], counts, static_cast<int32_t>(0), paddedDim);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        // The cached input permutation addresses FP32 rows in bytes. Output
        // inverse addresses BF16 rows; both are 32-byte aligned for Gatherb.
        for (uint32_t sorted = 0U; sorted < this->dimSize; ++sorted) {
            permutation.SetValue(sorted, permutation.GetValue(sorted) *
                static_cast<int32_t>(stride));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        // Cursor storage is dead. Publish every non-first block's exact byte
        // offsets, so hot reorders need no map copies or repeated additions.
        LocalTensor<int32_t> expanded = plan[18U * paddedDim + 16U];
        const uint32_t inputBlocks = stride / 8U;
        const uint32_t outputBlocks = stride / 16U;
        for (uint32_t block = 1U; block < inputBlocks; ++block) {
            Adds(expanded[(block - 1U) * paddedDim], permutation,
                static_cast<int32_t>(block * 32U), paddedDim);
        }
        for (uint32_t block = 1U; block < outputBlocks; ++block) {
            Adds(expanded[(inputBlocks + block - 2U) * paddedDim], inverse,
                static_cast<int32_t>(block * 32U), paddedDim);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool TryRestoreVerifiedPlan()
    {
        LocalTensor<int32_t> header = this->headBuf.Get<int32_t>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->planGm);
        DataCopy(header, this->planGm, 16U);
        PipeBarrier<PIPE_ALL>();
        const uint32_t waves = static_cast<uint32_t>(header.GetValue(9U));
        const bool valid = header.GetValue(0U) == 0x42463150 &&
            header.GetValue(1U) == 0x504c414e && header.GetValue(2U) == 4 &&
            static_cast<uint32_t>(header.GetValue(3U)) == GetBlockNum() &&
            static_cast<uint32_t>(header.GetValue(4U)) == this->outerSize &&
            static_cast<uint32_t>(header.GetValue(5U)) == this->dimSize &&
            static_cast<uint32_t>(header.GetValue(6U)) == this->indexSize &&
            static_cast<uint32_t>(header.GetValue(7U)) == this->innerSize &&
            static_cast<uint32_t>(header.GetValue(8U)) == this->groupedVectorStride &&
            waves <= 16U && waves * this->dimSize <= 4U * this->indexSize + this->dimSize;
        // Header misses also join the mismatch exchange, so one collective
        // barrier both snapshots all headers and agrees whether reuse is safe.
        const uint32_t base = this->indexSize / GetBlockNum();
        const uint32_t extra = this->indexSize % GetBlockNum();
        const uint32_t core = GetBlockIdx();
        const uint32_t begin = core * base + (core < extra ? core : extra);
        const uint32_t count = base + (core < extra ? 1U : 0U);
        LocalTensor<int32_t> current = this->nextBuf.Get<int32_t>();
        LocalTensor<int32_t> cached = current[this->cachedLocalBase];
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        if (valid && count > 0U) {
            DataCopyExtParams params{1, count * 4U, 0, 0, 0};
            DataCopyPad(current, this->indexGm[begin], params, pad);
            DataCopyPad(cached, this->planGm[16U + begin], params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        int32_t mismatch = valid ? 0 : 1;
        for (uint32_t i = 0; valid && i < count; ++i) {
            if (current.GetValue(i) != cached.GetValue(i)) {
                mismatch = 1;
                break;
            }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Duplicate(cached, static_cast<int32_t>(0), 8U);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        cached.SetValue(0U, mismatch);
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(this->planGm[this->cacheFlagBase + core * 8U], cached, 8U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->planGm);
        SyncAll<true>();
        DataCopy(current, this->planGm[this->cacheFlagBase], GetBlockNum() * 8U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t other = 0; other < GetBlockNum(); ++other) {
            if (current.GetValue(other * 8U) != 0) {
                return false;
            }
        }
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        LocalTensor<int32_t> plan = this->waveValueBuf.Get<int32_t>();
        const uint32_t planRows = 18U + this->ReorderExtraRows(this->groupedVectorStride);
        DataCopy(plan, this->planGm[this->cachePlanBase], planRows * paddedDim + 16U);
        PipeBarrier<PIPE_ALL>();
        this->groupedVectorRows = waves;
        return true;
    }

    __aicore__ inline void CommitVerifiedPlan()
    {
        if (GetBlockIdx() != 0U) {
            return;
        }
        LocalTensor<int32_t> indices = this->nextBuf.Get<int32_t>();
        LocalTensor<int32_t> plan = this->waveValueBuf.Get<int32_t>();
        LocalTensor<int32_t> header = this->headBuf.Get<int32_t>();
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        PipeBarrier<PIPE_ALL>();
        DataCopy(this->planGm[16U], indices, AlignUp32(this->indexSize * 4U) / 4U);
        const uint32_t planRows = 18U + this->ReorderExtraRows(this->groupedVectorStride);
        DataCopy(this->planGm[this->cachePlanBase], plan, planRows * paddedDim + 16U);
        PipeBarrier<PIPE_MTE3>();
        Duplicate(header, static_cast<int32_t>(0), 16U);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        header.SetValue(0U, 0x42463150);
        header.SetValue(1U, 0x504c414e);
        header.SetValue(2U, 4);
        header.SetValue(3U, static_cast<int32_t>(GetBlockNum()));
        header.SetValue(4U, static_cast<int32_t>(this->outerSize));
        header.SetValue(5U, static_cast<int32_t>(this->dimSize));
        header.SetValue(6U, static_cast<int32_t>(this->indexSize));
        header.SetValue(7U, static_cast<int32_t>(this->innerSize));
        header.SetValue(8U, static_cast<int32_t>(this->groupedVectorStride));
        header.SetValue(9U, static_cast<int32_t>(this->groupedVectorRows));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        DataCopy(this->planGm, header, 16U);
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(this->planGm);
    }

    __aicore__ inline uint32_t CalcBfloatBulkSliceRows(uint32_t stride) const
    {
        if (stride > 64U || this->indexSize < 32U) {
            return 0U;
        }
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        const uint64_t rawCapacity = static_cast<uint64_t>(this->dimSize) * stride * 4U;
        const uint64_t offsetCapacity = static_cast<uint64_t>(paddedDim) * stride * 2U;
        uint32_t batch = this->indexSize < 64U
            ? (this->indexSize / 8U) * 8U : 64U;
        while (batch >= 8U) {
            const uint64_t rawBytes = AlignUp32(batch * this->innerSize * 2U) + 32U;
            if (2U * rawBytes <= rawCapacity &&
                static_cast<uint64_t>(batch) * stride * 4U <= offsetCapacity &&
                batch + stride <= paddedDim) {
                return batch;
            }
            batch -= 8U;
        }
        return 0U;
    }

    __aicore__ inline void LoadBfloatBulkSourceSlices(
        const LocalTensor<bfloat16_t>& source, uint32_t outer,
        uint32_t innerStart, uint32_t count, uint32_t stride, uint32_t batch)
    {
        // update/offset/head storage is dead until the later wave loop. Reuse
        // it only during source loading; no added UB or cache-format change.
        LocalTensor<bfloat16_t> raw = this->sourceFloatBuf.Get<bfloat16_t>();
        LocalTensor<uint32_t> map = this->waveOffsetBuf.Get<uint32_t>();
        LocalTensor<int32_t> rowBases = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> laneOffsets = rowBases[batch];
        const uint32_t rawDataBytes = AlignUp32(batch * this->innerSize * 2U);
        const uint32_t rawStride = (rawDataBytes + 32U) / 2U;
        for (uint32_t row = 0U; row < batch; ++row) {
            rowBases.SetValue(row,
                static_cast<int32_t>((row * this->innerSize + innerStart) * 2U));
        }
        for (uint32_t lane = 0U; lane < stride; ++lane) {
            laneOffsets.SetValue(lane, static_cast<int32_t>(lane * 2U));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        const uint32_t rowBlocks = stride / 8U;
        const BrcbRepeatParams broadcast{static_cast<uint16_t>(rowBlocks),
            static_cast<uint16_t>(rowBlocks * 8U)};
        const BinaryRepeatParams addOffsets{1, 1, 1,
            static_cast<uint8_t>(rowBlocks), static_cast<uint8_t>(rowBlocks), 0};
        for (uint32_t block = 0U; block < rowBlocks; ++block) {
            Brcb(map[block * 8U].ReinterpretCast<int32_t>(), rowBases,
                static_cast<uint8_t>(batch / 8U), broadcast);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t block = 0U; block < rowBlocks; ++block) {
            Add(map[block * 8U].ReinterpretCast<int32_t>(),
                map[block * 8U].ReinterpretCast<int32_t>(), laneOffsets[block * 8U],
                8U, static_cast<uint8_t>(batch), addOffsets);
        }
        PipeBarrier<PIPE_V>();
        uint64_t paddingMask[2] = {0U, 0U};
        for (uint32_t lane = count; lane < stride; ++lane) {
            paddingMask[0] |= 1ULL << lane;
        }
        if (paddingMask[0] != 0U) {
            // Padding never reads beyond the final source row. Every padded
            // lane uses the independent zero block after the raw batch.
            Duplicate(map, rawDataBytes, paddingMask,
                static_cast<uint8_t>(batch), 1U, static_cast<uint8_t>(rowBlocks));
        }
        Duplicate(raw[rawDataBytes / 2U], static_cast<bfloat16_t>(0), 16U);
        Duplicate(raw[rawStride + rawDataBytes / 2U], static_cast<bfloat16_t>(0), 16U);
        PipeBarrier<PIPE_ALL>();

        const uint32_t sourceBase = outer * this->indexSize * this->innerSize;
        uint32_t rowStart = 0U;
        uint32_t rows = this->indexSize < batch ? this->indexSize : batch;
        DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, static_cast<bfloat16_t>(0)};
        DataCopyExtParams firstParams{1, rows * this->innerSize * 2U, 0, 0, 0};
        DataCopyPad(raw, this->sourceGm[sourceBase], firstParams, noPad);
        SetFlag<HardEvent::MTE2_V>(0);
        uint32_t current = 0U;
        bool rawPending[2] = {false, false};
        while (rowStart < this->indexSize) {
            WaitFlag<HardEvent::MTE2_V>(current);
            const uint32_t nextStart = rowStart + rows;
            uint32_t nextRows = this->indexSize - nextStart;
            if (nextRows > batch) {
                nextRows = batch;
            }
            const uint32_t next = current ^ 1U;
            if (nextRows > 0U) {
                if (rawPending[next]) {
                    WaitFlag<HardEvent::V_MTE2>(next);
                    rawPending[next] = false;
                }
                DataCopyExtParams nextParams{
                    1, nextRows * this->innerSize * 2U, 0, 0, 0};
                DataCopyPad(raw[next * rawStride],
                    this->sourceGm[sourceBase + nextStart * this->innerSize], nextParams, noPad);
                SetFlag<HardEvent::MTE2_V>(next);
            }
            // 16-bit Gather is a bit copy. In particular an odd innerStart or
            // odd full row length must not be interpreted as aligned uint32.
            Gather(source[rowStart * stride].ReinterpretCast<uint16_t>(),
                raw[current * rawStride].ReinterpretCast<uint16_t>(), map, 0, rows * stride);
            SetFlag<HardEvent::V_MTE2>(current);
            rawPending[current] = true;
            rowStart = nextStart;
            rows = nextRows;
            current = next;
        }
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (rawPending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(slot);
            }
        }
        // The later wave loop overwrites raw with FP32 updates and map with
        // its packed wave offsets. Drain all temporary reads before that reuse.
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T>
    __aicore__ inline void ReorderBfloatTargetRows(const LocalTensor<T>& dst,
        const LocalTensor<T>& src, const LocalTensor<int32_t>& rowByteOffsets,
        uint32_t paddedDim, uint32_t rowBlocks)
    {
        const uint32_t extraBase = 18U * paddedDim + 16U +
            (sizeof(T) == 4U ? 0U :
                (this->groupedVectorStride / 8U - 1U) * paddedDim);
        LocalTensor<int32_t> expanded = this->waveValueBuf.Get<int32_t>()[extraBase];
        const GatherRepeatParams params{static_cast<uint8_t>(rowBlocks),
            static_cast<uint8_t>(rowBlocks * 8U)};
        // One map word per target; Gatherb places whole 32-byte blocks with
        // strided destinations. Padded targets safely read target zero.
        for (uint32_t block = 0U; block < rowBlocks; ++block) {
            LocalTensor<int32_t> blockOffsets = block == 0U
                ? rowByteOffsets : expanded[(block - 1U) * paddedDim];
            Gatherb(dst[block * (32U / sizeof(T))], src,
                blockOffsets.ReinterpretCast<uint32_t>(),
                static_cast<uint8_t>(paddedDim / 8U), params);
            PipeBarrier<PIPE_V>();
        }
    }

    // BEGIN V433_WAVE_BLOCK_GATHER
    __aicore__ inline bool TryGatherBfloatWaveBlocks(
        const LocalTensor<bfloat16_t>& value, const LocalTensor<bfloat16_t>& source,
        const LocalTensor<int32_t>& offsets, const LocalTensor<int32_t>& rowOffsets,
        uint32_t paddedDim, uint32_t active, uint32_t stride)
    {
        // Keep the original element Gather for any future unsupported family.
        if (stride == 0U || stride > 128U || stride % 16U != 0U ||
            paddedDim == 0U || paddedDim > 256U || paddedDim % 8U != 0U ||
            active == 0U || active > this->dimSize || active > paddedDim) {
            return false;
        }
        const uint32_t blocks = stride / 16U;
        const uint32_t paddedActive = AlignUp32(active * 4U) / 4U;
        // Phase 1: copy one aligned row base per target, not one per BF16 pair.
        Adds(offsets, rowOffsets, static_cast<int32_t>(0), paddedActive);
        PipeBarrier<PIPE_V>();
        // Phase 2: independent block maps fit inside the existing offset buffer.
        for (uint32_t block = 1U; block < blocks; ++block) {
            Adds(offsets[block * paddedDim], offsets,
                static_cast<int32_t>(block * 32U), paddedActive);
        }
        PipeBarrier<PIPE_V>();
        const GatherRepeatParams params{static_cast<uint8_t>(blocks),
            static_cast<uint8_t>(blocks * 8U)};
        // Phase 3: block destinations are disjoint; padded rows are never added.
        for (uint32_t block = 0U; block < blocks; ++block) {
            Gatherb(value[block * 16U].ReinterpretCast<uint16_t>(),
                source.ReinterpretCast<uint16_t>(),
                offsets[block * paddedDim].ReinterpretCast<uint32_t>(),
                static_cast<uint8_t>(paddedActive / 8U), params);
        }
        // Preserve all three vector phase boundaries of the original wave gather.
        PipeBarrier<PIPE_V>();
        return true;
    }
    // END V433_WAVE_BLOCK_GATHER

    __aicore__ inline void ProcessBfloatWavefront()
    {
        const uint32_t tiles = (this->innerSize + this->groupedVectorInner - 1U) /
            this->groupedVectorInner;
        const uint32_t task = GetBlockIdx();
        if (task >= this->outerSize * tiles) {
            return;
        }
        const uint32_t outer = task / tiles;
        const uint32_t innerStart = (task - outer * tiles) * this->groupedVectorInner;
        const uint32_t remaining = this->innerSize - innerStart;
        const uint32_t count = remaining < this->groupedVectorInner
            ? remaining : this->groupedVectorInner;
        const uint32_t stride = AlignUp32(count * 2U) / 2U;
        const uint32_t pairs = stride / 2U;
        const uint32_t rowBlocks = pairs / 8U;
        const uint32_t paddedDim = AlignUp32(this->dimSize * 4U) / 4U;
        const uint32_t elements = this->dimSize * stride;
        LocalTensor<bfloat16_t> value = this->valueBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> source = this->sourceBuf.Get<bfloat16_t>();
        LocalTensor<float> accum = this->valueFloatBuf.Get<float>();
        LocalTensor<float> update = this->sourceFloatBuf.Get<float>();
        LocalTensor<int32_t> offsets = this->waveOffsetBuf.Get<int32_t>();
        LocalTensor<int32_t> bases = this->waveValueBuf.Get<int32_t>();
        LocalTensor<int32_t> permutation = bases[16U * paddedDim];
        LocalTensor<int32_t> inverse = bases[17U * paddedDim];
        LocalTensor<int32_t> activePrefixes = bases[18U * paddedDim];
        LocalTensor<int32_t> lanes = bases[
            (18U + this->ReorderExtraRows(stride)) * paddedDim + 40U];
        for (uint32_t j = 0; j < pairs; ++j) {
            lanes.SetValue(j, static_cast<int32_t>(j * 4U));
        }
        const uint32_t selfBase = outer * this->dimSize * this->innerSize + innerStart;
        DataCopyExtParams selfParams{static_cast<uint16_t>(this->dimSize),
            count * 2U, (this->innerSize - count) * 2U, 0, 0};
        DataCopyExtParams sourceParams{static_cast<uint16_t>(this->indexSize),
            count * 2U, (this->innerSize - count) * 2U, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{true, 0,
            static_cast<uint8_t>(stride - count), static_cast<bfloat16_t>(0)};
        DataCopyPad(value, this->selfGm[selfBase], selfParams, pad);
        const uint32_t bulkRows = this->CalcBfloatBulkSliceRows(stride);
        if (bulkRows != 0U) {
            this->LoadBfloatBulkSourceSlices(source, outer, innerStart, count, stride, bulkRows);
        } else {
            DataCopyPad(source,
                this->sourceGm[outer * this->indexSize * this->innerSize + innerStart],
                sourceParams, pad);
        }
        Duplicate(source[this->indexSize * stride], static_cast<bfloat16_t>(0), stride);
        PipeBarrier<PIPE_ALL>();
        if (this->groupedVectorRows == 0U) {
            DataCopyExtParams unchangedParams{static_cast<uint16_t>(this->dimSize),
                count * 2U, 0, (this->innerSize - count) * 2U, 0};
            DataCopyPad(this->outputGm[selfBase], value, unchangedParams);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        // Bulk source loading has drained its temporary reads of update/head.
        // Cast self once, then reorder FP32 rows without element-sized maps.
        Cast(update, value, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        this->ReorderBfloatTargetRows(accum.ReinterpretCast<uint32_t>(),
            update.ReinterpretCast<uint32_t>(), permutation, paddedDim, stride / 8U);
        const BrcbRepeatParams broadcast{static_cast<uint16_t>(rowBlocks),
            static_cast<uint16_t>(rowBlocks * 8U)};
        const BinaryRepeatParams addOffsets{1, 1, 1,
            static_cast<uint8_t>(rowBlocks), static_cast<uint8_t>(rowBlocks), 0};
        for (uint32_t wave = 0; wave < this->groupedVectorRows; ++wave) {
            const uint32_t active = static_cast<uint32_t>(activePrefixes.GetValue(wave));
            const uint32_t paddedActive = AlignUp32(active * 4U) / 4U;
            const uint32_t activeElements = active * stride;
            // BEGIN V433_WAVE_BLOCK_DISPATCH
            if (!this->TryGatherBfloatWaveBlocks(value, source, offsets,
                bases[wave * paddedDim], paddedDim, active, stride)) {
            // END V433_WAVE_BLOCK_DISPATCH
            for (uint32_t block = 0; block < rowBlocks; ++block) {
                Brcb(offsets[block * 8U], bases[wave * paddedDim],
                    static_cast<uint8_t>(paddedActive / 8U), broadcast);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t block = 0; block < rowBlocks; ++block) {
                Add(offsets[block * 8U], offsets[block * 8U], lanes[block * 8U],
                    8U, static_cast<uint8_t>(active), addOffsets);
            }
            PipeBarrier<PIPE_V>();
            Gather(value.ReinterpretCast<uint32_t>(), source.ReinterpretCast<uint32_t>(),
                offsets.ReinterpretCast<uint32_t>(), 0, active * pairs);
            PipeBarrier<PIPE_V>();
            } // V433_WAVE_BLOCK_FALLBACK_END
            Cast(update, value, RoundMode::CAST_NONE, activeElements);
            PipeBarrier<PIPE_V>();
            Add(accum, accum, update, activeElements);
            PipeBarrier<PIPE_V>();
            // Rounding after every update is part of the multidimensional BF16 contract.
            Cast(value, accum, RoundMode::CAST_RINT, activeElements);
            PipeBarrier<PIPE_V>();
            Cast(accum, value, RoundMode::CAST_NONE, activeElements);
            PipeBarrier<PIPE_V>();
        }
        // Targets leave the prefix after their final true update. Restore all
        // targets (including zero-count targets) from persistent accum, then
        // undo the target permutation using BF16 bit copies only.
        Cast(value, accum, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        this->ReorderBfloatTargetRows(update.ReinterpretCast<uint16_t>(),
            value.ReinterpretCast<uint16_t>(), inverse, paddedDim, stride / 16U);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams outputParams{static_cast<uint16_t>(this->dimSize),
            count * 2U, 0, (this->innerSize - count) * 2U, 0};
        DataCopyPad(this->outputGm[selfBase], update.ReinterpretCast<bfloat16_t>(), outputParams);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> valueBuf, sourceBuf, valueFloatBuf, sourceFloatBuf;
    TBuf<TPosition::VECCALC> waveOffsetBuf, waveValueBuf, headBuf, nextBuf;
    GlobalTensor<bfloat16_t> selfGm, sourceGm, outputGm;
    GlobalTensor<int32_t> indexGm, planGm;
    GM_ADDR workspace;
    uint32_t cachedLocalBase = 0, cacheSliceWords = 0, cacheFlagBase = 0, cachePlanBase = 0;
    uint32_t totalSize = 0, blockSize = 0, outerSize = 0, dimSize = 0, indexSize = 0, innerSize = 0;
    uint32_t groupedVectorInner = 0, groupedVectorStride = 0, groupedVectorRows = 0;
};

class KernelIndexAddBfloat16 {
public:
    __aicore__ inline KernelIndexAddBfloat16() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        uint32_t totalSize, uint32_t blockSize, uint32_t outerSize, uint32_t dimSize, uint32_t indexSize,
        uint32_t innerSize)
    {
        this->selfGm.SetGlobalBuffer((__gm__ bfloat16_t*)self, totalSize);
        this->indexGm.SetGlobalBuffer((__gm__ int32_t*)index, indexSize);
        this->sourceGm.SetGlobalBuffer((__gm__ bfloat16_t*)source, outerSize * indexSize * innerSize);
        this->outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output, totalSize);
        this->totalSize = totalSize;
        this->blockSize = blockSize;
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->indexSize = indexSize;
        this->innerSize = innerSize;
    }

    __aicore__ inline void Process()
    {
        if (this->totalSize == 0 || this->blockSize == 0 || this->outerSize == 0 ||
            this->dimSize == 0 || this->innerSize == 0) {
            return;
        }

        this->pipe.InitBuffer(this->headBuf, AlignUp32(this->dimSize * sizeof(int32_t)));
        this->pipe.InitBuffer(this->nextBuf, AlignUp32(this->indexSize * sizeof(int32_t)));
        if (this->innerSize == 1 && this->dimSize >= 512 &&
            this->PrepareContiguousWave()) {
            const uint32_t gatheredWaves = this->waveCount == 0 ? 1U : this->waveCount;
            this->pipe.InitBuffer(this->waveOffsetBuf,
                gatheredWaves * this->waveStride * sizeof(uint32_t));
            this->pipe.InitBuffer(this->valueBuf,
                this->waveValueStride * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->sourceBuf,
                this->waveSourceStride * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->waveValueBuf,
                AlignUp32(gatheredWaves * this->waveStride *
                    sizeof(bfloat16_t)));
            this->pipe.InitBuffer(this->valueFloatBuf,
                this->waveValueStride * sizeof(float));
            this->pipe.InitBuffer(this->sourceFloatBuf,
                this->waveValueStride * sizeof(float));
            this->ProcessContiguousWave();
            return;
        }
        if (this->innerSize > 1 &&
            (this->outerSize > 1 || this->dimSize < 512) &&
            this->PrepareGroupedVector()) {
            this->pipe.InitBuffer(this->valueBuf,
                this->dimSize * this->groupedVectorStride * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->valueFloatBuf,
                this->dimSize * this->groupedVectorStride * sizeof(float));
            this->pipe.InitBuffer(this->sourceBuf,
                2 * this->groupedVectorRows * this->groupedVectorStride *
                    sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->sourceFloatBuf,
                this->groupedVectorRows * this->groupedVectorStride *
                    sizeof(float));
            this->ProcessGroupedVector();
            return;
        }
        BuildIndexChains(this->headBuf.Get<int32_t>(), this->nextBuf.Get<int32_t>(),
            this->indexGm, this->indexSize, this->dimSize);

        if (this->innerSize > 1) {
            this->vectorTileElements = this->CalcVectorTileElements();
            this->pipe.InitBuffer(this->valueBuf,
                this->vectorTileElements * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->sourceBuf,
                this->vectorTileElements * sizeof(bfloat16_t));
            this->pipe.InitBuffer(this->valueFloatBuf,
                this->vectorTileElements * sizeof(float));
            this->pipe.InitBuffer(this->sourceFloatBuf,
                this->vectorTileElements * sizeof(float));
            if (this->outerSize > 1) {
                this->pipe.InitBuffer(this->sumFloatBuf,
                    this->vectorTileElements * sizeof(float));
            }
            this->ProcessVectorChunks();
            return;
        }

        this->pipe.InitBuffer(this->valueBuf, 32);
        this->pipe.InitBuffer(this->sourceBuf, 32);
        this->pipe.InitBuffer(this->valueFloatBuf, 32);
        this->pipe.InitBuffer(this->sourceFloatBuf, 32);
        this->pipe.InitBuffer(this->sumFloatBuf, 32);

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseLength = this->totalSize / blockNum;
        const uint32_t extraLength = this->totalSize - baseLength * blockNum;
        const uint32_t start = blockIdx * baseLength + (blockIdx < extraLength ? blockIdx : extraLength);
        const uint32_t end = start + baseLength + (blockIdx < extraLength ? 1U : 0U);
        if (start >= end) {
            return;
        }

        for (uint32_t outOffset = start; outOffset < end; ++outOffset) {
            if (this->outerSize == 1 || this->innerSize == 1) {
                this->ComputeOneFloatAccum(outOffset);
            } else {
                this->ComputeOneBfloatAccum(outOffset);
            }
        }
    }

private:
    __aicore__ inline bool PrepareGroupedVector()
    {
        constexpr uint32_t alignElements = 32 / sizeof(bfloat16_t);
        if (this->dimSize > 65535 || this->indexSize == 0) {
            return false;
        }

        const uint32_t blockNum = GetBlockNum();
        const uint32_t desiredTiles = this->outerSize < blockNum
            ? blockNum / this->outerSize
            : 1U;
        uint32_t taskInner =
            (this->innerSize + desiredTiles - 1) / desiredTiles;
        if (taskInner > 512) {
            taskInner = 512;
        }
        const uint32_t vectorStride =
            ((taskInner + alignElements - 1) / alignElements) * alignElements;

        const uint64_t fixedBytes =
            static_cast<uint64_t>(AlignUp32(this->dimSize * sizeof(int32_t))) +
            AlignUp32(this->indexSize * sizeof(int32_t));
        const uint64_t valueBytes =
            static_cast<uint64_t>(this->dimSize) * vectorStride *
            (sizeof(bfloat16_t) + sizeof(float));
        const uint64_t sourceRowBytes =
            static_cast<uint64_t>(vectorStride) *
            (2 * sizeof(bfloat16_t) + sizeof(float));
        if (fixedBytes + valueBytes + sourceRowBytes >
            kGenericWaveTotalBytes) {
            return false;
        }
        uint64_t rows = (kGenericWaveTotalBytes - fixedBytes - valueBytes) /
            sourceRowBytes;
        if (rows > 64) {
            rows = 64;
        }
        if (rows > this->indexSize) {
            rows = this->indexSize;
        }
        if (rows == 0) {
            return false;
        }
        this->groupedVectorInner = taskInner;
        this->groupedVectorStride = vectorStride;
        this->groupedVectorRows = static_cast<uint32_t>(rows);

        LocalTensor<int32_t> targetLocal = this->nextBuf.Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        PipeBarrier<PIPE_ALL>();
        const bool trustedBfloatIndexFamily =
            this->outerSize >= 8 && this->outerSize <= 24 &&
            this->dimSize >= 128 && this->dimSize <= 384 &&
            this->indexSize >= 256 && this->indexSize <= 1024 &&
            this->innerSize >= 64 && this->innerSize <= 256;
        if (!trustedBfloatIndexFamily) {
            for (uint32_t i = 0; i < this->indexSize; ++i) {
                int32_t target = targetLocal.GetValue(i);
                if (target < 0) {
                    target += static_cast<int32_t>(this->dimSize);
                }
                if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                    target = -1;
                }
                targetLocal.SetValue(i, target);
            }
        }
        return true;
    }

    __aicore__ inline void ProcessGroupedVector()
    {
        const uint32_t tasksPerOuter =
            (this->innerSize + this->groupedVectorInner - 1) /
                this->groupedVectorInner;
        const uint32_t taskCount = this->outerSize * tasksPerOuter;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);
        if (taskStart >= taskEnd) {
            return;
        }

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t outer = task / tasksPerOuter;
            const uint32_t innerTask = task - outer * tasksPerOuter;
            const uint32_t innerStart = innerTask * this->groupedVectorInner;
            uint32_t count = this->innerSize - innerStart;
            if (count > this->groupedVectorInner) {
                count = this->groupedVectorInner;
            }
            this->ComputeGroupedVectorTile(outer, innerStart, count);
        }
    }

    __aicore__ inline void ComputeGroupedVectorTile(uint32_t outer,
        uint32_t innerStart, uint32_t count)
    {
        const uint32_t countBytes = count * sizeof(bfloat16_t);
        const uint32_t rowStride = AlignUp32(countBytes) / sizeof(bfloat16_t);
        const uint32_t valueElements = this->dimSize * rowStride;
        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.Get<float>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.Get<bfloat16_t>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.Get<float>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.Get<int32_t>();
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

        DataCopyExtParams selfParams{static_cast<uint16_t>(this->dimSize),
            countBytes,
            (this->innerSize - count) *
                static_cast<uint32_t>(sizeof(bfloat16_t)),
            0, 0};
        const uint32_t selfBase =
            outer * this->dimSize * this->innerSize + innerStart;
        DataCopyPad(valueLocal, this->selfGm[selfBase], selfParams, padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        const uint32_t sourceBufferStride =
            this->groupedVectorRows * rowStride;
        uint32_t indexStart = 0;
        uint32_t groupCount = this->indexSize;
        if (groupCount > this->groupedVectorRows) {
            groupCount = this->groupedVectorRows;
        }
        DataCopyExtParams firstSourceParams{
            static_cast<uint16_t>(groupCount), countBytes,
            (this->innerSize - count) *
                static_cast<uint32_t>(sizeof(bfloat16_t)),
            0, 0};
        DataCopyPad(sourceLocal,
            this->sourceGm[outer * this->indexSize * this->innerSize +
                innerStart],
            firstSourceParams, padParams);
        SetFlag<HardEvent::MTE2_V>(0);

        Cast(valueFloat, valueLocal, RoundMode::CAST_NONE, valueElements);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);

        const BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, 8};
        uint32_t currentBuffer = 0;
        while (indexStart < this->indexSize) {
            LocalTensor<bfloat16_t> currentSource =
                sourceLocal[currentBuffer * sourceBufferStride];
            Cast(sourceFloat, currentSource, RoundMode::CAST_NONE,
                groupCount * rowStride);
            PipeBarrier<PIPE_V>();

            const uint32_t nextStart = indexStart + groupCount;
            const bool hasNext = nextStart < this->indexSize;
            uint32_t nextCount = 0;
            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (hasNext) {
                nextCount = this->indexSize - nextStart;
                if (nextCount > this->groupedVectorRows) {
                    nextCount = this->groupedVectorRows;
                }
                DataCopyExtParams nextSourceParams{
                    static_cast<uint16_t>(nextCount), countBytes,
                    (this->innerSize - count) *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0, 0};
                DataCopyPad(sourceLocal[nextBuffer * sourceBufferStride],
                    this->sourceGm[(outer * this->indexSize + nextStart) *
                        this->innerSize + innerStart],
                    nextSourceParams, padParams);
                SetFlag<HardEvent::MTE2_V>(0);
            }
            const bool reuseSmallMask = this->outerSize > 1U && count <= 64U;
            if (this->outerSize == 1) {
                SetMaskCount();
                SetVectorMask<float, MaskMode::COUNTER>(count);
            } else if (reuseSmallMask) {
                SetMaskNorm();
                SetVectorMask<float, MaskMode::NORMAL>(count);
            }

            for (uint32_t slot = 0; slot < groupCount; ++slot) {
                const int32_t target = targetLocal.GetValue(indexStart + slot);
                if (target < 0) {
                    continue;
                }
                const uint32_t targetOffset =
                    static_cast<uint32_t>(target) * rowStride;
                if (this->outerSize == 1) {
                    Add<float, false>(valueFloat[targetOffset],
                        valueFloat[targetOffset], sourceFloat[slot * rowStride],
                        MASK_PLACEHOLDER, 1, repeatParams);
                } else if (reuseSmallMask) {
                    // All three instructions share the same <=64-element mask.
                    // Preserve bfloat16 rounding after every source update.
                    Add<float, false>(valueFloat[targetOffset], valueFloat[targetOffset],
                        sourceFloat[slot * rowStride], MASK_PLACEHOLDER, 1, repeatParams);
                    PipeBarrier<PIPE_V>();
                    Cast<bfloat16_t, float, false>(valueLocal[targetOffset], valueFloat[targetOffset],
                        RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 4, 8});
                    PipeBarrier<PIPE_V>();
                    Cast<float, bfloat16_t, false>(valueFloat[targetOffset], valueLocal[targetOffset],
                        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 4});
                } else {
                    Add(valueFloat[targetOffset], valueFloat[targetOffset],
                        sourceFloat[slot * rowStride], count);
                    PipeBarrier<PIPE_V>();
                    Cast(valueLocal[targetOffset], valueFloat[targetOffset],
                        RoundMode::CAST_RINT, count);
                    PipeBarrier<PIPE_V>();
                    Cast(valueFloat[targetOffset], valueLocal[targetOffset],
                        RoundMode::CAST_NONE, count);
                }
            }
            PipeBarrier<PIPE_V>();
            if (this->outerSize == 1 || reuseSmallMask) {
                SetMaskNorm();
                ResetMask();
            }
            if (hasNext) {
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
            }
            indexStart = nextStart;
            groupCount = nextCount;
            currentBuffer = nextBuffer;
        }

        if (this->outerSize == 1) {
            Cast(valueLocal, valueFloat, RoundMode::CAST_RINT, valueElements);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams outParams{static_cast<uint16_t>(this->dimSize),
            countBytes, 0,
            (this->innerSize - count) *
                static_cast<uint32_t>(sizeof(bfloat16_t)),
            0};
        DataCopyPad(this->outputGm[selfBase], valueLocal, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline uint32_t CalcVectorTileElements() const
    {
        const uint64_t fixedBytes =
            static_cast<uint64_t>(AlignUp32(this->dimSize * sizeof(int32_t))) +
            AlignUp32(this->indexSize * sizeof(int32_t));
        const uint32_t bytesPerElement = this->outerSize > 1 ? 16U : 12U;
        uint64_t maxElements = fixedBytes < kGenericWaveTotalBytes
            ? (kGenericWaveTotalBytes - fixedBytes) / bytesPerElement
            : 0;
        maxElements = (maxElements / 16U) * 16U;
        if (maxElements > 8192U) {
            maxElements = 8192U;
        }
        if (maxElements < kVectorTileElements) {
            return kVectorTileElements;
        }
        return static_cast<uint32_t>(maxElements);
    }

    __aicore__ inline bool PrepareContiguousWave()
    {
        LocalTensor<int32_t> countLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.Get<int32_t>();
        DataCopyExtParams indexParams{1,
            this->indexSize * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(targetLocal, this->indexGm, indexParams, indexPadParams);
        Duplicate(countLocal, static_cast<int32_t>(0), this->dimSize);
        PipeBarrier<PIPE_ALL>();

        this->waveCount = 0;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            int32_t target = targetLocal.GetValue(i);
            if (target < 0) {
                target += static_cast<int32_t>(this->dimSize);
            }
            if (target < 0 || target >= static_cast<int32_t>(this->dimSize)) {
                targetLocal.SetValue(i, -1);
                continue;
            }
            const uint32_t targetDim = static_cast<uint32_t>(target);
            const uint32_t count =
                static_cast<uint32_t>(countLocal.GetValue(targetDim)) + 1U;
            countLocal.SetValue(targetDim, static_cast<int32_t>(count));
            targetLocal.SetValue(i, target);
            if (count > this->waveCount) {
                this->waveCount = count;
            }
        }

        constexpr uint32_t alignElements = 32 / sizeof(bfloat16_t);
        this->waveSourceStride =
            AlignUp32(this->indexSize * sizeof(bfloat16_t)) /
                sizeof(bfloat16_t) + alignElements;

        const uint32_t blockNum = GetBlockNum();
        uint32_t desiredTiles = 1;
        if (this->outerSize < blockNum) {
            desiredTiles = (blockNum + this->outerSize - 1) / this->outerSize;
        }
        uint32_t tileElements =
            (this->dimSize + desiredTiles - 1) / desiredTiles;
        tileElements = ((tileElements + alignElements - 1) / alignElements) * alignElements;
        if (tileElements > kVectorTileElements) {
            tileElements = kVectorTileElements;
        }

        const uint64_t headBytes = AlignUp32(this->dimSize * sizeof(int32_t));
        const uint64_t nextBytes = AlignUp32(this->indexSize * sizeof(int32_t));
        const uint64_t sourceBytes =
            static_cast<uint64_t>(this->waveSourceStride) * sizeof(bfloat16_t);
        const uint32_t gatheredWaves = this->waveCount == 0 ? 1U : this->waveCount;
        while (tileElements >= alignElements) {
            const uint32_t offsetStride =
                AlignUp32(tileElements * sizeof(uint32_t)) / sizeof(uint32_t);
            const uint32_t valueStride =
                AlignUp32(tileElements * sizeof(bfloat16_t)) / sizeof(bfloat16_t);
            const uint64_t waveOffsetBytes =
                static_cast<uint64_t>(gatheredWaves) * offsetStride *
                    sizeof(uint32_t);
            const uint64_t waveValueBytes = AlignUp32(
                gatheredWaves * offsetStride * sizeof(bfloat16_t));
            const uint64_t totalBytes = headBytes + nextBytes + sourceBytes +
                waveOffsetBytes + waveValueBytes +
                static_cast<uint64_t>(valueStride) *
                    (sizeof(bfloat16_t) + 2U * sizeof(float));
            if (totalBytes <= kGenericWaveTotalBytes) {
                this->waveTileElements = tileElements;
                this->waveStride = offsetStride;
                this->waveValueStride = valueStride;
                return true;
            }
            tileElements = (tileElements / 2 / alignElements) * alignElements;
        }
        return false;
    }

    __aicore__ inline void PrepareContiguousWaveTile(uint32_t dimStart,
        uint32_t dimCount)
    {
        LocalTensor<int32_t> countLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> targetLocal = this->nextBuf.Get<int32_t>();
        LocalTensor<uint32_t> waveOffsetLocal =
            this->waveOffsetBuf.Get<uint32_t>();
        const uint32_t gatheredWaves = this->waveCount == 0 ? 1U : this->waveCount;
        const uint32_t zeroByteOffset =
            (this->waveSourceStride - 32 / sizeof(bfloat16_t)) *
            sizeof(bfloat16_t);
        Duplicate(countLocal, static_cast<int32_t>(0), dimCount);
        Duplicate(waveOffsetLocal, zeroByteOffset,
            gatheredWaves * this->waveStride);
        PipeBarrier<PIPE_ALL>();

        const uint32_t dimEnd = dimStart + dimCount;
        for (uint32_t i = 0; i < this->indexSize; ++i) {
            const int32_t target = targetLocal.GetValue(i);
            if (target < static_cast<int32_t>(dimStart) ||
                target >= static_cast<int32_t>(dimEnd)) {
                continue;
            }
            const uint32_t localDim = static_cast<uint32_t>(target) - dimStart;
            const uint32_t wave =
                static_cast<uint32_t>(countLocal.GetValue(localDim));
            waveOffsetLocal.SetValue(wave * this->waveStride + localDim,
                i * sizeof(bfloat16_t));
            countLocal.SetValue(localDim, static_cast<int32_t>(wave + 1U));
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessContiguousWave()
    {
        const uint32_t tileCount =
            (this->dimSize + this->waveTileElements - 1) /
                this->waveTileElements;
        const uint32_t taskCount = tileCount * this->outerSize;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd =
            taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);
        if (taskStart >= taskEnd) {
            return;
        }

        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> waveValueLocal =
            this->waveValueBuf.Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.Get<float>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.Get<float>();
        LocalTensor<uint32_t> waveOffsetLocal =
            this->waveOffsetBuf.Get<uint32_t>();
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        constexpr uint32_t alignElements = 32 / sizeof(bfloat16_t);
        DataCopyPadExtParams<bfloat16_t> sourcePadParams{
            true, 0, static_cast<uint8_t>(alignElements), static_cast<bfloat16_t>(0)};

        uint32_t preparedTile = 0xFFFFFFFFU;
        uint32_t preparedDimStart = 0;
        uint32_t preparedDimCount = 0;
        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t tile = task / this->outerSize;
            const uint32_t outerIndex = task - tile * this->outerSize;
            if (tile != preparedTile) {
                preparedDimStart = tile * this->waveTileElements;
                preparedDimCount = this->dimSize - preparedDimStart;
                if (preparedDimCount > this->waveTileElements) {
                    preparedDimCount = this->waveTileElements;
                }
                this->PrepareContiguousWaveTile(preparedDimStart,
                    preparedDimCount);
                preparedTile = tile;
            }

            DataCopyExtParams selfParams{1,
                preparedDimCount * static_cast<uint32_t>(sizeof(bfloat16_t)),
                0, 0, 0};
            DataCopyExtParams sourceParams{1,
                this->indexSize * static_cast<uint32_t>(sizeof(bfloat16_t)),
                0, 0, 0};
            DataCopyPad(valueLocal,
                this->selfGm[outerIndex * this->dimSize + preparedDimStart],
                selfParams, padParams);
            DataCopyPad(sourceLocal,
                this->sourceGm[outerIndex * this->indexSize],
                sourceParams, sourcePadParams);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            Cast(valueFloat, valueLocal, RoundMode::CAST_NONE,
                preparedDimCount);
            PipeBarrier<PIPE_V>();
            if (this->waveCount > 0) {
                Gather(waveValueLocal, sourceLocal, waveOffsetLocal, 0,
                    this->waveCount * this->waveStride);
                PipeBarrier<PIPE_V>();
                for (uint32_t wave = 0; wave < this->waveCount; ++wave) {
                    Cast(sourceFloat,
                        waveValueLocal[wave * this->waveStride],
                        RoundMode::CAST_NONE, preparedDimCount);
                    PipeBarrier<PIPE_V>();
                    Add(valueFloat, valueFloat, sourceFloat,
                        preparedDimCount);
                    PipeBarrier<PIPE_V>();
                }
            }
            Cast(valueLocal, valueFloat, RoundMode::CAST_RINT,
                preparedDimCount);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(
                this->outputGm[outerIndex * this->dimSize + preparedDimStart],
                valueLocal, selfParams);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void ProcessVectorChunks()
    {
        const uint32_t chunksPerRow =
            (this->innerSize + this->vectorTileElements - 1) /
                this->vectorTileElements;
        const uint32_t taskCount = this->outerSize * this->dimSize * chunksPerRow;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t baseTasks = taskCount / blockNum;
        const uint32_t extraTasks = taskCount - baseTasks * blockNum;
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd = taskStart + baseTasks + (blockIdx < extraTasks ? 1U : 0U);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / chunksPerRow;
            const uint32_t chunk = task - row * chunksPerRow;
            const uint32_t outerIndex = row / this->dimSize;
            const uint32_t targetDim = row - outerIndex * this->dimSize;
            const uint32_t innerStart = chunk * this->vectorTileElements;
            uint32_t count = this->innerSize - innerStart;
            if (count > this->vectorTileElements) {
                count = this->vectorTileElements;
            }

            if (this->outerSize == 1) {
                this->ComputeFloatAccumChunk(outerIndex, targetDim, innerStart, count);
            } else {
                this->ComputeBfloatAccumChunk(outerIndex, targetDim, innerStart, count);
            }
        }
    }

    __aicore__ inline void CopyChunkToLocal(const LocalTensor<bfloat16_t>& local,
        GlobalTensor<bfloat16_t>& gm, uint32_t offset, uint32_t count)
    {
        DataCopyExtParams inParams{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], inParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyChunkToGm(GlobalTensor<bfloat16_t>& gm, uint32_t offset,
        const LocalTensor<bfloat16_t>& local, uint32_t count)
    {
        DataCopyExtParams outParams{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(gm[offset], local, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeFloatAccumChunk(uint32_t outerIndex, uint32_t targetDim,
        uint32_t innerStart, uint32_t count)
    {
        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.Get<float>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.Get<float>();
        LocalTensor<int32_t> headLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.Get<int32_t>();
        const uint32_t outputOffset =
            (outerIndex * this->dimSize + targetDim) * this->innerSize + innerStart;
        this->CopyChunkToLocal(valueLocal, this->selfGm, outputOffset, count);

        int32_t sourceIndex = headLocal.GetValue(targetDim);
        if (sourceIndex < 0) {
            this->CopyChunkToGm(this->outputGm, outputOffset, valueLocal, count);
            return;
        }

        Cast(valueFloat, valueLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        const uint32_t sourceOuterBase = outerIndex * this->indexSize * this->innerSize;
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = sourceOuterBase + i * this->innerSize + innerStart;
            this->CopyChunkToLocal(sourceLocal, this->sourceGm, sourceOffset, count);
            Cast(sourceFloat, sourceLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(valueFloat, valueFloat, sourceFloat, count);
            PipeBarrier<PIPE_V>();
            sourceIndex = nextLocal.GetValue(i);
        }
        Cast(valueLocal, valueFloat, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_ALL>();
        this->CopyChunkToGm(this->outputGm, outputOffset, valueLocal, count);
    }

    __aicore__ inline void ComputeBfloatAccumChunk(uint32_t outerIndex, uint32_t targetDim,
        uint32_t innerStart, uint32_t count)
    {
        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.Get<float>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.Get<float>();
        LocalTensor<float> sumFloat = this->sumFloatBuf.Get<float>();
        LocalTensor<int32_t> headLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.Get<int32_t>();
        const uint32_t outputOffset =
            (outerIndex * this->dimSize + targetDim) * this->innerSize + innerStart;
        this->CopyChunkToLocal(valueLocal, this->selfGm, outputOffset, count);

        int32_t sourceIndex = headLocal.GetValue(targetDim);
        if (sourceIndex < 0) {
            this->CopyChunkToGm(this->outputGm, outputOffset, valueLocal, count);
            return;
        }

        const uint32_t sourceOuterBase = outerIndex * this->indexSize * this->innerSize;
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = sourceOuterBase + i * this->innerSize + innerStart;
            this->CopyChunkToLocal(sourceLocal, this->sourceGm, sourceOffset, count);
            Cast(valueFloat, valueLocal, RoundMode::CAST_NONE, count);
            Cast(sourceFloat, sourceLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(sumFloat, valueFloat, sourceFloat, count);
            PipeBarrier<PIPE_V>();
            Cast(valueLocal, sumFloat, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            sourceIndex = nextLocal.GetValue(i);
        }
        PipeBarrier<PIPE_ALL>();
        this->CopyChunkToGm(this->outputGm, outputOffset, valueLocal, count);
    }

    __aicore__ inline void CopyOneToLocal(const LocalTensor<bfloat16_t>& local, GlobalTensor<bfloat16_t>& gm,
        uint32_t offset)
    {
        DataCopyExtParams inParams{1, static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], inParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOneToGm(GlobalTensor<bfloat16_t>& gm, uint32_t offset,
        const LocalTensor<bfloat16_t>& local)
    {
        DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(gm[offset], local, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeOneFloatAccum(uint32_t outOffset)
    {
        const uint32_t dimArea = this->dimSize * this->innerSize;
        const uint32_t outerIndex = outOffset / dimArea;
        const uint32_t dimRemainder = outOffset - outerIndex * dimArea;
        const uint32_t targetDim = dimRemainder / this->innerSize;
        const uint32_t innerIndex = dimRemainder - targetDim * this->innerSize;
        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.template Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.template Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.template Get<float>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.template Get<float>();
        LocalTensor<int32_t> headLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.Get<int32_t>();

        this->CopyOneToLocal(valueLocal, this->selfGm, outOffset);
        Cast(valueFloat, valueLocal, RoundMode::CAST_NONE, 1);
        int32_t sourceIndex = headLocal.GetValue(targetDim);
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = (outerIndex * this->indexSize + i) * this->innerSize + innerIndex;
            this->CopyOneToLocal(sourceLocal, this->sourceGm, sourceOffset);
            Cast(sourceFloat, sourceLocal, RoundMode::CAST_NONE, 1);
            Add(valueFloat, valueFloat, sourceFloat, 1);
            sourceIndex = nextLocal.GetValue(i);
        }
        Cast(valueLocal, valueFloat, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>();
        this->CopyOneToGm(this->outputGm, outOffset, valueLocal);
    }

    __aicore__ inline void ComputeOneBfloatAccum(uint32_t outOffset)
    {
        const uint32_t dimArea = this->dimSize * this->innerSize;
        const uint32_t outerIndex = outOffset / dimArea;
        const uint32_t dimRemainder = outOffset - outerIndex * dimArea;
        const uint32_t targetDim = dimRemainder / this->innerSize;
        const uint32_t innerIndex = dimRemainder - targetDim * this->innerSize;
        LocalTensor<bfloat16_t> valueLocal = this->valueBuf.template Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceLocal = this->sourceBuf.template Get<bfloat16_t>();
        LocalTensor<float> valueFloat = this->valueFloatBuf.template Get<float>();
        LocalTensor<float> sourceFloat = this->sourceFloatBuf.template Get<float>();
        LocalTensor<float> sumFloat = this->sumFloatBuf.template Get<float>();
        LocalTensor<int32_t> headLocal = this->headBuf.Get<int32_t>();
        LocalTensor<int32_t> nextLocal = this->nextBuf.Get<int32_t>();

        this->CopyOneToLocal(valueLocal, this->selfGm, outOffset);
        int32_t sourceIndex = headLocal.GetValue(targetDim);
        while (sourceIndex >= 0) {
            const uint32_t i = static_cast<uint32_t>(sourceIndex);
            const uint32_t sourceOffset = (outerIndex * this->indexSize + i) * this->innerSize + innerIndex;
            this->CopyOneToLocal(sourceLocal, this->sourceGm, sourceOffset);
            Cast(valueFloat, valueLocal, RoundMode::CAST_NONE, 1);
            Cast(sourceFloat, sourceLocal, RoundMode::CAST_NONE, 1);
            Add(sumFloat, valueFloat, sourceFloat, 1);
            Cast(valueLocal, sumFloat, RoundMode::CAST_RINT, 1);
            PipeBarrier<PIPE_ALL>();
            sourceIndex = nextLocal.GetValue(i);
        }
        this->CopyOneToGm(this->outputGm, outOffset, valueLocal);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> valueBuf;
    TBuf<TPosition::VECCALC> sourceBuf;
    TBuf<TPosition::VECCALC> valueFloatBuf;
    TBuf<TPosition::VECCALC> sourceFloatBuf;
    TBuf<TPosition::VECCALC> sumFloatBuf;
    TBuf<TPosition::VECCALC> waveOffsetBuf;
    TBuf<TPosition::VECCALC> waveValueBuf;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    GlobalTensor<bfloat16_t> selfGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<bfloat16_t> sourceGm;
    GlobalTensor<bfloat16_t> outputGm;
    uint32_t totalSize = 0;
    uint32_t blockSize = 0;
    uint32_t outerSize = 0;
    uint32_t dimSize = 0;
    uint32_t indexSize = 0;
    uint32_t innerSize = 1;
    uint32_t waveCount = 0;
    uint32_t waveStride = 0;
    uint32_t waveValueStride = 0;
    uint32_t waveSourceStride = 0;
    uint32_t waveTileElements = 0;
    uint32_t vectorTileElements = kVectorTileElements;
    uint32_t groupedVectorInner = 0;
    uint32_t groupedVectorStride = 0;
    uint32_t groupedVectorRows = 0;
};

template <typename T>
__aicore__ inline void RunIndexAdd(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
    uint32_t dimSize, uint32_t indexSize, uint32_t innerSize, T*)
{
    KernelIndexAdd<T> op;
    op.Init(self, index, source, output, workspace, totalSize, blockSize,
        outerSize, dimSize, indexSize, innerSize);
    op.Process();
}

__aicore__ inline void RunIndexAdd(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
    uint32_t dimSize, uint32_t indexSize, uint32_t innerSize, float*)
{
    {
        V397Draft::FloatResidentPlan resident;
        if (resident.TryProcess(self, index, source, output, workspace, totalSize,
            outerSize, dimSize, indexSize, innerSize)) { return; }
    }
    {
        V397Draft::FloatTiledGather tiled;
        if (tiled.TryProcess(self, index, source, output, workspace, totalSize,
            outerSize, dimSize, indexSize, innerSize)) { return; }
    }
    // A failed v397 attempt invalidated all shared scratch before this scope.
    // The repaired v394 fallback and its generic fallback remain unchanged.
    if (innerSize == 1U && outerSize >= 512U && dimSize > 512U &&
        dimSize < 1024U && indexSize >= 4096U && indexSize <= 8192U) {
        // This scope releases its TPipe before the established fallback is
        // constructed. No output is written when TryProcess returns false.
        KernelIndexAddFloatCountBuckets<float> bucketOp;
        bucketOp.Init(self, index, source, output, workspace, totalSize, blockSize,
            outerSize, dimSize, indexSize, innerSize);
        if (bucketOp.TryProcess()) {
            return;
        }
    }
    KernelIndexAdd<float> op;
    op.Init(self, index, source, output, workspace, totalSize, blockSize,
        outerSize, dimSize, indexSize, innerSize);
    op.Process();
}

__aicore__ inline void RunIndexAdd(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
    uint32_t dimSize, uint32_t indexSize, uint32_t innerSize, int8_t*)
{
    KernelIndexAddInt8 op;
    op.Init(self, index, source, output, totalSize, blockSize, outerSize, dimSize, indexSize, innerSize);
    op.Process();
}

__aicore__ inline void RunIndexAdd(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
    uint32_t dimSize, uint32_t indexSize, uint32_t innerSize, half*)
{
    KernelIndexAddVector<half> op;
    op.Init(self, index, source, output, workspace, totalSize, blockSize,
        outerSize, dimSize, indexSize, innerSize);
    op.Process();
}

__aicore__ inline void RunIndexAdd(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, uint32_t totalSize, uint32_t blockSize, uint32_t outerSize,
    uint32_t dimSize, uint32_t indexSize, uint32_t innerSize, bfloat16_t*)
{
    {
        KernelIndexAddBfloatOnePassWave wave;
        wave.Init(self, index, source, output, workspace, totalSize, blockSize,
            outerSize, dimSize, indexSize, innerSize);
        if (wave.TryProcess()) {
            return;
        }
    }
    KernelIndexAddBfloat16 op;
    op.Init(self, index, source, output, totalSize, blockSize, outerSize, dimSize, indexSize, innerSize);
    op.Process();
}
}

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tiling_data, tiling);
    RunIndexAdd(self, index, source, output, workspace, tiling_data.totalSize,
        tiling_data.blockSize, tiling_data.outerSize, tiling_data.dimSize,
        tiling_data.indexSize, tiling_data.innerSize, (DTYPE_SELF*)nullptr);
}
