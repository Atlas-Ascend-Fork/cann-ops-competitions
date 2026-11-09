#ifndef S9_SQUARE_SUM_V1_DIAG_PROBE_H
#define S9_SQUARE_SUM_V1_DIAG_PROBE_H

// Optional diagnostic encoding for the diagSpin tiling field.
// The kernel consumes the encoded iteration count after computing output.
// A zero kDiagVersion disables the diagnostic delay.
#include <cstdint>

namespace squaresum_diag {

// Zero disables the diagnostic encoding; other versions select formulas below.
constexpr uint32_t kDiagVersion = 0;

// Iteration multiplier for diagnostic delay encoding.
constexpr uint64_t kItersPerStep = 58000ULL;

struct DiagShapeInfo {
    uint32_t dtypeIdx;      // f16=1 bf16=2 f32=3
    uint32_t rank;          // raw rank before collapsing
    uint32_t keepDims;      // 0/1
    uint32_t innerIsReduce; // from the plan
    uint32_t axisMask;      // bit i set = raw dim i is reduced (i < rank)
    uint64_t totalElems;
    uint64_t keptInnerLen;  // collapsed kept inner run length; 0 when reduced
    uint32_t tilingKey;     // v5: the plan's own routing verdict
    uint32_t usedCores;     // v5: and its launch width
    // v7 gate-conjunct word (append-only, host-only struct):
    uint32_t runGroupOn;    // plan.runGroup != 0 (CONTIG ladder formed)
    uint32_t runsContig;    // plan.rStride[0] == plan.innerLen
    uint32_t nOuterR;
    uint32_t nOuterK;
    uint64_t outTiles;
    uint32_t redChunksGt1;  // the split actually engaged
    uint32_t indepLtCores;  // indepUnits < the core budget it saw
    uint64_t innerLenRaw;   // v8: the collapsed inner run, reduced or kept
};

inline uint64_t DiagCode(const DiagShapeInfo& d)
{
    if (kDiagVersion == 1U) {
        return static_cast<uint64_t>(d.dtypeIdx) * 20ULL + d.rank * 4ULL +
               d.keepDims * 2ULL + d.innerIsReduce;
    }
    if (kDiagVersion == 2U) {
        uint64_t lg = 0;
        uint64_t t = d.totalElems;
        while (t > 1ULL) {
            t >>= 1ULL;
            ++lg;
        }
        return static_cast<uint64_t>(d.axisMask) * 40ULL + lg;
    }
    if (kDiagVersion == 3U) {
        return d.keptInnerLen;
    }
    if (kDiagVersion == 5U) {
        // Encode the selected tiling key and core count as tilingKey*50 + usedCores.
        return static_cast<uint64_t>(d.tilingKey) * 50ULL + d.usedCores;
    }
    if (kDiagVersion == 6U) {
        // Split-condition word:
        // b0 innerIsReduce; b1 runGroup!=0; b2 rStride[0]==innerLen;
        // b3..4 nOuterR capped at 3; b5..6 nOuterK capped at 3;
        // b7..9 outTiles capped at 7; b10 redChunks>1; b11 indepUnits<cores.
        return static_cast<uint64_t>(d.innerIsReduce) +
               2ULL * d.runGroupOn + 4ULL * d.runsContig +
               8ULL * ((d.nOuterR < 3U) ? d.nOuterR : 3U) +
               32ULL * ((d.nOuterK < 3U) ? d.nOuterK : 3U) +
               128ULL * ((d.outTiles < 7ULL) ? d.outTiles : 7ULL) +
               1024ULL * d.redChunksGt1 + 2048ULL * d.indepLtCores;
    }
    if (kDiagVersion == 8U) {
        // Encode the collapsed inner length, capped at 600, with the larger step.
        const uint64_t il = (d.innerLenRaw < 600ULL) ? d.innerLenRaw : 600ULL;
        return il;
    }
    if (kDiagVersion == 9U) {
        // Encode min(outTiles,31) + 32*floor(log2(totalElems)).
        uint64_t lg = 0;
        uint64_t t = d.totalElems;
        while (t > 1ULL) {
            t >>= 1ULL;
            ++lg;
        }
        const uint64_t ot = (d.outTiles < 31ULL) ? d.outTiles : 31ULL;
        return ot + 32ULL * lg;
    }
    if (kDiagVersion == 10U) {
        // Encode floor(log2(totalElems)) + 32*(dtypeSize==4) + 64*innerIsReduce.
        uint64_t lg = 0;
        uint64_t t = d.totalElems;
        while (t > 1ULL) {
            t >>= 1ULL;
            ++lg;
        }
        return lg + ((d.dtypeIdx == 3U) ? 32ULL : 0ULL) +
               (d.innerIsReduce ? 64ULL : 0ULL);
    }
    if (kDiagVersion == 4U) {
        // Encode element width and a six-bit fractional mantissa of totalElems:
        // 512*(dtypeSize==4) + mantissa, where totalElems is approximately
        // (64+mantissa) * 2^(floor(log2(totalElems))-6).
        uint64_t t = d.totalElems;
        uint64_t lg = 0ULL;
        while ((t >> (lg + 1ULL)) != 0ULL) {
            ++lg;
        }
        const uint64_t mant =
            (lg >= 6ULL) ? ((d.totalElems >> (lg - 6ULL)) - 64ULL) : 0ULL;
        const uint64_t wide = (d.dtypeIdx == 3U) ? 1ULL : 0ULL;
        return wide * 512ULL + mant;
    }
    return 0ULL;
}

inline uint64_t DiagSpin(const DiagShapeInfo& d)
{
    const uint64_t code = DiagCode(d);
    // Use version-dependent iteration multipliers for bounded codes and
    // element-count codes.
    const uint64_t mult = (kDiagVersion == 3U || kDiagVersion == 6U) ? (kItersPerStep / 5ULL) : (kItersPerStep * 10ULL);
    return (kDiagVersion == 0U || code == 0ULL) ? 0ULL : code * mult;
}

}  // namespace squaresum_diag

#endif  // S9_SQUARE_SUM_V1_DIAG_PROBE_H
