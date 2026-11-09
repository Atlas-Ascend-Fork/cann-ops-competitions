#ifndef GREATER_TILING_CORE_H
#define GREATER_TILING_CORE_H
/*
 * Dependency-free tiling planner for Greater. It derives broadcast metadata,
 * UB bounds, output partitions and kernel dispatch keys from the input metadata.
 */
#include "greater_tiling_fields.h"
#include "s9_variant.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#ifndef GT_DEFAULT_TUNING
#define GT_DEFAULT_TUNING 2
#endif

namespace greater_core {

constexpr uint32_t GT_MAX_DIM = GT_MAX_DIM_VALUE;

constexpr uint32_t FAM_FLAT = 0;   // no broadcast anywhere
constexpr uint32_t FAM_BCAST = 1;  // odometer over collapsed dims
constexpr uint32_t FAM_ONE = 2;    // flat AND one core's whole share fits one tile
constexpr uint32_t FAM_FAST = 3;   // FAM_ONE's conditions AND an exact split

// The aligned route requires no broadcast, an exact split
// (totalNum == usedCores * perCore), a 32-element-aligned per-core share,
// and sufficient UB capacity within GT_FAST_TILE. These conditions permit
// aligned copies and a payload containing only perCore. Other inputs use
// the one-tile, flat or broadcast routes.
constexpr uint64_t FAST_TILE_ELEMS = 2048;   // Keep this bound aligned with GT_FAST_TILE.
constexpr uint64_t FAST_UB_PER_ELEM = 32;    // Conservative per-element UB bound shared with the one-tile route.
// The aligned-route payload stores perCore as one uint64_t at offset zero.
constexpr size_t GT_FAST_TILING_BYTES = 8;

// The one-tile kernel uses a fixed UB layout. Select it only when a core share
// fits GT_ONE_TILE and the available UB; keep both host and kernel bounds aligned.
constexpr uint64_t ONE_TILE_ELEMS = 2048;
// A deliberately loose upper bound on the kernel's per-element UB cost. The
// real figure is KernelGreaterOne::UB_BYTES / N_ELEM, at most 17.125 bytes
// (int32 and bfloat16); 32 leaves the guard true on any platform where the real
// layout fits with room to spare, and needs no dtype table on this side.
constexpr uint64_t ONE_UB_PER_ELEM = 32;
constexpr uint64_t ONE_UB_RESERVE = 8192;  // Keep this reservation aligned with the kernel UB_RESERVE.

// Output is one byte per element: keep every core's output range 32-byte
// aligned so two cores can never touch the same 32-byte block of GM. That block
// is therefore also the smallest piece of work the split can hand to a core,
// which is why the core count is sized in these rather than in bytes - see the
// core-split comment in MakePlan.
constexpr uint64_t OUT_BLOCK_ELEMS = 32;

// Minimum per-core work used to cap the core count on flat routes.
// Broadcast routes use their own work partitioning.
constexpr uint64_t MIN_FLAT_ELEMS_PER_CORE = 512;

// Runtime variant 1 suppresses promotion to the one-tile route.
constexpr uint32_t GT_VAR_NO_ONESHOT = 1;
// Runtime variant 2 disables the flat-route minimum-work core cap.
constexpr uint32_t GT_VAR_NO_CORECAP = 2;
// Runtime variant 3 serializes the full tiling layout. The one-tile kernel
// continues to read its two-field prefix.
constexpr uint32_t GT_VAR_FULL_TILING = 3;
// Runtime variant 4 suppresses the aligned route while retaining the one-tile route.
constexpr uint32_t GT_VAR_NO_FAST = 4;

// Runtime variants 10 through 16 select diagnostic prefixes for the aligned
// fp16 route. These prefixes intentionally omit parts of the computation and
// must not be used as ordinary Greater results.
constexpr uint32_t GT_VAR_LADDER0 = 10;
constexpr uint32_t GT_LADDER_MAX_LEVEL = 6;      // L0, L0w, L1..L5
constexpr uint64_t GT_LADDER_KEY0 = 900;         // key = GT_LADDER_KEY0 + level
constexpr uint32_t GT_LADDER_DTYPE = 1;          // The ladder keys are reserved for diagnostic prefixes of the aligned fp16 route.
constexpr uint32_t GT_LADDER_NONE = 0xffffffffu;

// The one-tile reader consumes perCore and totalNum, the first two uint64_t
// fields. General routes read the metadata required by their registered layout.
constexpr size_t GT_ONE_TILING_BYTES = 16;

// Registered byte layout: perCore 0, totalNum 8, innerLen 16, rowNum 24,
// ubSize 32, usedCores 36, dimNum 40, innerMode 44, cacheMode 48, pad0 52,
// shape 56..87, strides 88..215, tilingKey 216..223, diagCode 224..231.
// Keep the shared schema and every direct kernel reader consistent with it.
constexpr size_t GT_GEN_TILING_BYTES = 224;
constexpr size_t GT_DIAG_OFFSET_GEN = 224;


// Minimum reuse count before a repeated inner block is worth holding in UB.
// Each reload costs a full pipe barrier, so at a reuse of 2 the barrier can eat
// the saved DMA; 4 keeps the amortisation comfortably positive.
constexpr uint64_t MIN_ROW_BLOCK_REUSE = 4;

// Plan uses the shared field schema. Diagnostic encoding is applied after planning.
struct Plan {
#define GT_PLAN_SCALAR(type, name) type name;
#define GT_PLAN_ARRAY(type, count, name) type name[count];
    GT_TILING_FIELDS(GT_PLAN_SCALAR, GT_PLAN_ARRAY)
#undef GT_PLAN_SCALAR
#undef GT_PLAN_ARRAY
};


// dtypeIdx: 0 float32, 1 float16, 2 bfloat16, 3 int32, 4 int8
inline uint32_t ElemSize(uint32_t dtypeIdx)
{
    const uint32_t table[5] = {4, 2, 2, 4, 1};
    return (dtypeIdx < 5) ? table[dtypeIdx] : 0;
}

inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// MakePlanV receives explicit dimensions, dtype, UB size, core count and
// variant. The caller applies any downward core-count cap beforehand.
inline bool MakePlanV(const int64_t *s1, size_t r1, const int64_t *s2, size_t r2, uint32_t dtypeIdx,
                      uint64_t ubSize, uint32_t coreNum, uint32_t variant, Plan *out, uint32_t *family)
{
    const uint32_t elemSize = ElemSize(dtypeIdx);
    if (elemSize == 0) {
        return false;
    }
    const size_t rank = std::max<size_t>(std::max(r1, r2), 1);
    std::vector<int64_t> a(rank, 1), b(rank, 1), o(rank, 1);
    for (size_t i = 0; i < r1; ++i) {
        a[rank - r1 + i] = s1[i];
    }
    for (size_t i = 0; i < r2; ++i) {
        b[rank - r2 + i] = s2[i];
    }
    for (size_t i = 0; i < rank; ++i) {
        o[i] = std::max(a[i], b[i]);
        if (a[i] == 0 || b[i] == 0) {
            o[i] = 0;
        }
    }

    uint64_t totalNum = 1;
    for (size_t i = 0; i < rank; ++i) {
        totalNum *= static_cast<uint64_t>(o[i]);
    }

    // ---- collapse adjacent dims that share a broadcast pattern --------------
    // Pattern per axis is (f1, f2), f = 1 meaning "this operand is broadcast
    // here". Equal patterns keep both operands contiguous across the merge, so
    // the merge is layout-safe. Axes with extent 1 in the output contribute
    // nothing to either layout and are dropped first.
    std::vector<uint64_t> sh;
    std::vector<int> f1, f2;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        if (o[i] == 1) {
            continue;
        }
        const int p1 = (a[i] != o[i]) ? 1 : 0;
        const int p2 = (b[i] != o[i]) ? 1 : 0;
        if (!sh.empty() && f1.back() == p1 && f2.back() == p2) {
            sh.back() *= static_cast<uint64_t>(o[i]);
        } else {
            sh.push_back(static_cast<uint64_t>(o[i]));
            f1.push_back(p1);
            f2.push_back(p2);
        }
    }
    if (sh.empty()) {
        sh.push_back(totalNum == 0 ? 0 : 1);
        f1.push_back(0);
        f2.push_back(0);
    }
    std::reverse(sh.begin(), sh.end());
    std::reverse(f1.begin(), f1.end());
    std::reverse(f2.begin(), f2.end());

    const uint32_t dimNum = static_cast<uint32_t>(sh.size());
    if (dimNum > GT_MAX_DIM) {
        return false;
    }

    // ---- per-axis element strides for each operand -------------------------
    uint64_t st1[GT_MAX_DIM] = {0};
    uint64_t st2[GT_MAX_DIM] = {0};
    uint64_t run1 = 1, run2 = 1;
    for (int i = static_cast<int>(dimNum) - 1; i >= 0; --i) {
        st1[i] = f1[i] ? 0 : run1;
        if (!f1[i]) {
            run1 *= sh[i];
        }
        st2[i] = f2[i] ? 0 : run2;
        if (!f2[i]) {
            run2 *= sh[i];
        }
    }

    bool anyBroadcast = false;
    for (uint32_t i = 0; i < dimNum; ++i) {
        if (f1[i] || f2[i]) {
            anyBroadcast = true;
        }
    }
    *family = anyBroadcast ? FAM_BCAST : FAM_FLAT;

    const uint64_t innerLen = sh[dimNum - 1];
    const uint64_t rowNum = (innerLen == 0) ? 0 : totalNum / innerLen;
    const uint32_t innerMode = f1[dimNum - 1] ? 1u : (f2[dimNum - 1] ? 2u : 0u);
    uint32_t tuning=variant;
#if GT_DEFAULT_TUNING
    if (variant==0 && totalNum >= (1ULL<<20)) {
        if (*family==FAM_BCAST && innerLen>=256) {
            if (dtypeIdx==3) tuning=GT_DEFAULT_TUNING>=2?53:21;
            if (dtypeIdx==2) tuning=GT_DEFAULT_TUNING>=2?52:24;
        } else if (*family==FAM_FLAT && totalNum*2*elemSize >= (1ULL<<24)) {
            tuning=22;
            if(GT_DEFAULT_TUNING>=2) {
                if(dtypeIdx==0) tuning=56;
                if(dtypeIdx==4) tuning=54;
            }
        }
    }
#endif

    const bool repeatPattern=dimNum==3 && innerMode==0 &&
        ((st1[1]==0 && st2[1]==innerLen && st1[0]==innerLen && st2[0]==sh[1]*innerLen) ||
         (st2[1]==0 && st1[1]==innerLen && st2[0]==innerLen && st1[0]==sh[1]*innerLen));
    const bool fixedRepeat=(tuning==28 || tuning==29 || tuning==45 || tuning==53) && *family==FAM_BCAST && dtypeIdx==3 &&
        repeatPattern && innerLen>=256 && innerLen<=1024 &&
        sh[1]*innerLen>=8192 && (sh[1]*innerLen)%8==0 &&
        173056+CeilDiv(innerLen,256)*1024+CeilDiv(2*innerLen,256)*1024+8192<=ubSize;
    const bool phaseCached=(tuning==47 || tuning==50) && *family==FAM_BCAST && dtypeIdx==3 &&
        repeatPattern && sh[1]<=0xffffffffULL && innerLen>=256 && innerLen<=1024 && (8192%innerLen)%8==0 &&
        (sh[1]*innerLen)%8==0 && CeilDiv(sh[1]*innerLen,8192)*(8192%innerLen)<=1024 &&
        173056+CeilDiv(innerLen,256)*1024+CeilDiv(2*innerLen,256)*1024+8192<=ubSize;

    // Retain a repeated row in UB when the reuse count meets the threshold and
    // the row fits the route allocation bounds.
    uint32_t cacheMode = 0;
    if (dimNum >= 2 && sh[dimNum - 2] >= MIN_ROW_BLOCK_REUSE) {
        if (!f1[dimNum - 1] && st1[dimNum - 2] == 0) {
            cacheMode |= 1u;
        }
        if (!f2[dimNum - 1] && st2[dimNum - 2] == 0) {
            cacheMode |= 2u;
        }
    }

    // Partition output in 32-element units. Bound the core count by both the
    // number of output units and the available hardware cores.
    const uint64_t workUnits = CeilDiv(totalNum, OUT_BLOCK_ELEMS);
    uint64_t want = (workUnits == 0) ? 1 : workUnits;
    // Cap flat-route core count using the minimum per-core work threshold.
    if (*family == FAM_FLAT && variant != GT_VAR_NO_CORECAP) {
        const uint64_t byWork = CeilDiv(totalNum, MIN_FLAT_ELEMS_PER_CORE);
        if (byWork > 0 && byWork < want) {
            want = byWork;
        }
    }
    uint32_t usedCores = static_cast<uint32_t>(std::min<uint64_t>(want, coreNum));
    if (usedCores == 0) {
        usedCores = 1;
    }
    const uint64_t splitGran=(((tuning>=30 && tuning<=41 || tuning==48 || tuning==49 || tuning>=54 && tuning<=56) && *family==FAM_FLAT && totalNum>=(1ULL<<20)) || fixedRepeat || phaseCached) ? 512 : OUT_BLOCK_ELEMS;
    uint64_t perCore = CeilDiv(CeilDiv(totalNum, usedCores), splitGran) * splitGran;
    if (perCore == 0) {
        perCore = OUT_BLOCK_ELEMS;
    }
    usedCores = static_cast<uint32_t>(CeilDiv(totalNum, perCore));
    if (usedCores == 0) {
        usedCores = 1;
    }

    // Promote to the one-tile route only when its tile bound and UB reservation
    // are satisfied. Apply variant controls before constructing the dispatch key.
    if (*family == FAM_FLAT && perCore <= ONE_TILE_ELEMS &&
        ONE_TILE_ELEMS * ONE_UB_PER_ELEM + ONE_UB_RESERVE <= ubSize &&
        variant != GT_VAR_NO_ONESHOT) {
        *family = FAM_ONE;
    }

    // Promote to the aligned route only for exact core splits with aligned
    // per-core work and a sufficient fixed UB allocation.
    if (*family == FAM_ONE && perCore <= FAST_TILE_ELEMS &&
        FAST_TILE_ELEMS * FAST_UB_PER_ELEM + ONE_UB_RESERVE <= ubSize &&
        perCore % OUT_BLOCK_ELEMS == 0 &&
        static_cast<uint64_t>(usedCores) * perCore == totalNum &&
        variant != GT_VAR_NO_FAST) {
        *family = FAM_FAST;
    }

    // Apply the diagnostic ladder after route promotion. It changes only the key
    // for eligible fp16 aligned work and retains the same metadata payload.
    // The selected prefix intentionally does not compute an ordinary output.
    uint32_t ladderLevel = GT_LADDER_NONE;
    if (*family == FAM_FAST && dtypeIdx == GT_LADDER_DTYPE) {
        const uint32_t v = variant;
        if (v >= GT_VAR_LADDER0 && v <= GT_VAR_LADDER0 + GT_LADDER_MAX_LEVEL) {
            ladderLevel = v - GT_VAR_LADDER0;
        }
    }

    out->totalNum = totalNum;
    out->perCore = perCore;
    out->innerLen = innerLen;
    out->rowNum = rowNum;
    out->ubSize = static_cast<uint32_t>(ubSize);
    out->usedCores = usedCores;
    out->dimNum = dimNum;
    out->innerMode = innerMode;
    out->cacheMode = cacheMode;
    out->pad0 = 0;
    if(phaseCached) out->pad0=32768+(tuning==50?2048:0);
    if(fixedRepeat) out->pad0=514u+(tuning!=28?1024u:0u)+(tuning==45?2048u:0u)+(tuning==53?131072u:0u);
    if((tuning>=42 && tuning<=44 || tuning==46) && *family==FAM_BCAST && (dtypeIdx==2 || dtypeIdx==3) &&
       repeatPattern && innerLen>=256 && innerLen<=2016 &&
       182016+CeilDiv(innerLen,256)*256*elemSize+8192<=ubSize) {
        out->pad0=8192+(tuning==43?0:2)+(tuning==44?2048:0)+(tuning==46?16384:0);
    }
    // Resident repeated-row route.
    if ((tuning >= 20 && tuning <= 27 || tuning==52) && *family == FAM_BCAST && (dtypeIdx == 2 || dtypeIdx == 3) &&
        dimNum == 3 && innerMode == 0 && innerLen <= 2048 && innerLen > 0 &&
        ((st1[1] == 0 && st2[1] == innerLen && st1[0] == innerLen && st2[0] == sh[1]*innerLen) ||
         (st2[1] == 0 && st1[1] == innerLen && st2[0] == innerLen && st1[0] == sh[1]*innerLen))) {
        const uint64_t rk=CeilDiv(innerLen,256)*256;
        const uint64_t need=(dtypeIdx==3 ? 181248+rk*4+(tuning==27?256:0) : 169984+rk*6)+8192;
        if (need <= ubSize) {
            out->pad0 = tuning==52 ? 65539 : tuning==20 ? 1 : tuning==21 ? 3 : tuning==22 ? 8 : tuning==23 ? 16 :
                        tuning==24 ? 24 : tuning==25 ? 48 : tuning==26 ? 56 : 67;
        }
    }
    if (tuning==22 && *family==FAM_FLAT) out->pad0=8;
    if (tuning>=30 && tuning<=37 && *family==FAM_FLAT && totalNum>=(1ULL<<20)) {
        out->pad0=8;
        if(tuning>=31) out->pad0|=32;
        if(tuning>=31 && tuning!=33) out->pad0|=256;
        if(tuning==32 || tuning==35 || tuning==37) out->pad0|=128;
        if(tuning>=34) out->pad0|=1024;
        if(tuning>=36) out->pad0|=2048;
    }
    if(tuning>=38 && tuning<=41 && *family==FAM_FLAT && dtypeIdx==0 && totalNum>=(1ULL<<20) && ubSize>=65536) {
        out->pad0=4096;
        if(tuning==39 || tuning==41) out->pad0|=128;
        if(tuning>=40) out->pad0|=2048;
    }
    if((tuning==48 || tuning==49) && *family==FAM_FLAT && totalNum>=(1ULL<<20)) {
        out->pad0=8+32+256+16384+(tuning==49?1024:0);
    }
    if(tuning>=54 && tuning<=56 && *family==FAM_FLAT && totalNum>=(1ULL<<20)) {
        out->pad0=8+32+256+131072;
        if(tuning==55) out->pad0|=128;
        if(tuning==56) out->pad0|=16384;
    }
    for (uint32_t i = 0; i < GT_MAX_DIM; ++i) {
        out->shape[i] = (i < dimNum) ? static_cast<uint32_t>(sh[i]) : 0u;
        out->strides[2 * i] = (i < dimNum) ? st1[i] : 0u;
        out->strides[2 * i + 1] = (i < dimNum) ? st2[i] : 0u;
    }
    // Keep the dispatch key consistent with the selected route. Keys starting at
    // 900 are reserved for the optional diagnostic ladder.
    out->tilingKey = (ladderLevel == GT_LADDER_NONE)
                         ? (static_cast<uint64_t>(dtypeIdx) * 10ULL + static_cast<uint64_t>(*family))
                         : (GT_LADDER_KEY0 + static_cast<uint64_t>(ladderLevel));
    out->diagCode = 0;  // the TilingFunc overwrites this; see greater_diag.h
    return true;
}

// Default wrapper using the cached runtime variant.
inline bool MakePlan(const int64_t *s1, size_t r1, const int64_t *s2, size_t r2, uint32_t dtypeIdx,
                     uint64_t ubSize, uint32_t coreNum, Plan *out, uint32_t *family)
{
    return MakePlanV(s1, r1, s2, r2, dtypeIdx, ubSize, coreNum, s9::Variant(), out, family);
}

}  // namespace greater_core
#endif  // GREATER_TILING_CORE_H
