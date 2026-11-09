#ifndef ASCEND_S9_TRANSPOSE_DIAG_H
#define ASCEND_S9_TRANSPOSE_DIAG_H

// Optional diagnostic encoding for the trailing diagSpin tiling field.
// The kernel consumes the encoded iteration count after computing output.
// A zero kDiagVersion disables the diagnostic delay.
#include <cstdint>

namespace s9 {
namespace transpose_diag {

// Zero disables the diagnostic encoding; other versions select formulas below.
constexpr uint32_t kDiagVersion = 0;

// Iteration multiplier for diagnostic delay encoding.
constexpr uint64_t kItersPerUnit = 58000ULL;

inline uint64_t DiagCode(const int64_t* dims, uint32_t rank, const int64_t* perm,
                         uint32_t permRank, uint32_t dtypeSize)
{
    if (permRank != rank || rank == 0U) {
        return 0ULL;
    }
    // Normalized permutation (attrs may carry negatives).
    int64_t p[8];
    for (uint32_t i = 0; i < rank && i < 8U; ++i) {
        p[i] = (perm[i] < 0) ? (perm[i] + static_cast<int64_t>(rank)) : perm[i];
    }
    if (kDiagVersion == 1U) {
        // dtypeIdx: 2-byte=1, 4-byte comes back from the runtime as either
        // float or int32 -- both size 4, indistinguishable here, encoded 2;
        // int8=4.  (A dtype refinement can ride a later version if needed.)
        const uint64_t dtypeIdx = (dtypeSize == 2U) ? 1ULL : ((dtypeSize == 4U) ? 2ULL : 4ULL);
        const uint64_t lastKept = (p[rank - 1U] == static_cast<int64_t>(rank - 1U)) ? 1ULL : 0ULL;
        const uint64_t firstKept = (p[0] == 0) ? 1ULL : 0ULL;
        uint64_t isRev = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            if (p[i] != static_cast<int64_t>(rank - 1U - i)) {
                isRev = 0ULL;
                break;
            }
        }
        return dtypeIdx * 40ULL + rank * 8ULL + lastKept * 4ULL + firstKept * 2ULL + isRev;
    }
    if (kDiagVersion == 2U) {
        uint64_t numel = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            numel *= static_cast<uint64_t>(dims[i]);
        }
        uint64_t lgN = 0ULL;
        while (numel > 1ULL) {
            numel >>= 1ULL;
            ++lgN;
        }
        uint64_t innerOut = static_cast<uint64_t>(dims[p[rank - 1U]]);
        uint64_t lgI = 0ULL;
        while (innerOut > 1ULL) {
            innerOut >>= 1ULL;
            ++lgI;
        }
        return lgN * 16ULL + lgI;
    }
    if (kDiagVersion == 3U) {
        // The output's innermost run length, exactly.
        return static_cast<uint64_t>(dims[p[rank - 1U]]);
    }
    return 0ULL;
}

// Encode the selected tiling key and core count as tilingKey*50 + usedCores.
inline uint64_t DiagCodeRoute(uint64_t tilingKey, uint32_t usedCores)
{
    return tilingKey * 50ULL + static_cast<uint64_t>(usedCores);
}

// Encode the collapsed input-innermost extent, capped at 8191.
inline uint64_t DiagSpin(const int64_t* dims, uint32_t rank, const int64_t* perm,
                         uint32_t permRank, uint32_t dtypeSize,
                         uint64_t tilingKey = 0ULL, uint32_t usedCores = 0U,
                         uint64_t tileK = 0ULL, uint64_t packGateWord = 0ULL,
                         uint64_t rowLen = 0ULL, uint64_t extraA = 0ULL)
{
    if (kDiagVersion == 16U) {
        // Select the diagnostic formula by tiling key: fold-gate conditions,
        // batch count, or tilingKey*50 + usedCores + tilePrefetch*500.
        uint64_t code;
        if (tilingKey == 4ULL) {
            code = packGateWord;
        } else if (tilingKey == 3ULL || tilingKey == 7ULL) {
            uint64_t numel = 1ULL;
            for (uint32_t i = 0; i < rank; ++i) {
                numel *= static_cast<uint64_t>(dims[i]);
            }
            const uint64_t denom =
                (rowLen == 0ULL ? 1ULL : rowLen) * (tileK == 0ULL ? 1ULL : tileK);
            code = numel / denom;              // the batch count
            if (code > 4095ULL) {
                code = 4095ULL;
            }
        } else {
            const uint64_t tp = (tilingKey == 9ULL) ? extraA : 0ULL;
            code = tilingKey * 50ULL + static_cast<uint64_t>(usedCores) + tp * 500ULL;
        }
        return (code == 0ULL) ? 0ULL : code * (kItersPerUnit / 5ULL);
    }
    if (kDiagVersion == 15U) {
        // Select a diagnostic formula by floor(log2(numel)): output row length,
        // packed-route conditions, or the selected key/core/prefetch values.
        uint64_t numel = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            numel *= static_cast<uint64_t>(dims[i]);
        }
        uint64_t lg = 0ULL;
        for (uint64_t v = numel; v > 1ULL; v >>= 1ULL) {
            ++lg;
        }
        uint64_t code;
        if (lg == 22ULL) {
            code = (rowLen < 4095ULL) ? rowLen : 4095ULL;
        } else if (lg == 23ULL) {
            code = packGateWord;
        } else {
            const uint64_t tp = (tilingKey == 9ULL) ? extraA : 0ULL;
            code = tilingKey * 50ULL + static_cast<uint64_t>(usedCores) + tp * 500ULL;
        }
        return (code == 0ULL) ? 0ULL : code * (kItersPerUnit / 5ULL);
    }
    if (kDiagVersion == 14U) {
        // Encode tilingKey*50 + usedCores + tilePrefetch*500.
        // Only the wide-tile route contributes the prefetch term.
        const uint64_t tp = (tilingKey == 9ULL) ? extraA : 0ULL;
        const uint64_t code = tilingKey * 50ULL + static_cast<uint64_t>(usedCores) + tp * 500ULL;
        return (code == 0ULL) ? 0ULL : code * (kItersPerUnit / 5ULL);
    }
    if (kDiagVersion == 5U) {
        const uint64_t k = (tileK < 8191ULL) ? tileK : 8191ULL;
        return k * (kItersPerUnit / 5ULL);
    }
    // Versions 10..13 encode byte count, element count, output row length,
    // and the collapsed input-innermost extent, respectively.
    if (kDiagVersion == 11U) {   // ELEMENT COUNT
        uint64_t q = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) { q *= static_cast<uint64_t>(dims[i]); }
        if (q < 1ULL) { q = 1ULL; }
        uint64_t lg = 0;
        uint64_t t = q;
        while (t > 1ULL) { t >>= 1ULL; ++lg; }
        static const uint64_t kP[31] = {
            1072286, 1096532, 1121326, 1146681, 1172609, 1199123,
            1226237, 1253964, 1282318, 1311313, 1340964, 1371285,
            1402291, 1433999, 1466424, 1499582, 1533490, 1568164,
            1603623, 1639883, 1676963, 1714881, 1753657, 1793310,
            1833859, 1875326, 1917729, 1961092, 2005435, 2050781,
            2097152
        };
        uint64_t frac = 0;
        {
            const uint64_t bs = 1ULL << lg;
            const uint64_t q20 = (q << 20ULL) / bs;
            for (uint64_t k = 0; k < 31ULL; ++k) {
                if (q20 >= kP[k]) { frac = k + 1ULL; } else { break; }
            }
        }
        uint64_t code = lg * 31ULL + frac;
        if (code > 620ULL) { code = 620ULL; }
        return code * (kItersPerUnit * 10ULL);
    }
    if (kDiagVersion == 12U) {   // OUTPUT ROW LENGTH (plan.rowLen == bigM)
        uint64_t q = rowLen;
        if (q < 1ULL) { q = 1ULL; }
        uint64_t lg = 0;
        uint64_t t = q;
        while (t > 1ULL) { t >>= 1ULL; ++lg; }
        static const uint64_t kP[31] = {
            1072286, 1096532, 1121326, 1146681, 1172609, 1199123,
            1226237, 1253964, 1282318, 1311313, 1340964, 1371285,
            1402291, 1433999, 1466424, 1499582, 1533490, 1568164,
            1603623, 1639883, 1676963, 1714881, 1753657, 1793310,
            1833859, 1875326, 1917729, 1961092, 2005435, 2050781,
            2097152
        };
        uint64_t frac = 0;
        {
            const uint64_t bs = 1ULL << lg;
            const uint64_t q20 = (q << 20ULL) / bs;
            for (uint64_t k = 0; k < 31ULL; ++k) {
                if (q20 >= kP[k]) { frac = k + 1ULL; } else { break; }
            }
        }
        uint64_t code = lg * 31ULL + frac;
        if (code > 620ULL) { code = 620ULL; }
        return code * (kItersPerUnit * 10ULL);
    }
    if (kDiagVersion == 13U) {   // COLLAPSED INPUT-INNERMOST (plan.tileK)
        uint64_t q = tileK;
        if (q < 1ULL) { q = 1ULL; }
        uint64_t lg = 0;
        uint64_t t = q;
        while (t > 1ULL) { t >>= 1ULL; ++lg; }
        static const uint64_t kP[31] = {
            1072286, 1096532, 1121326, 1146681, 1172609, 1199123,
            1226237, 1253964, 1282318, 1311313, 1340964, 1371285,
            1402291, 1433999, 1466424, 1499582, 1533490, 1568164,
            1603623, 1639883, 1676963, 1714881, 1753657, 1793310,
            1833859, 1875326, 1917729, 1961092, 2005435, 2050781,
            2097152
        };
        uint64_t frac = 0;
        {
            const uint64_t bs = 1ULL << lg;
            const uint64_t q20 = (q << 20ULL) / bs;
            for (uint64_t k = 0; k < 31ULL; ++k) {
                if (q20 >= kP[k]) { frac = k + 1ULL; } else { break; }
            }
        }
        uint64_t code = lg * 31ULL + frac;
        if (code > 620ULL) { code = 620ULL; }
        return code * (kItersPerUnit * 10ULL);
    }
    if (kDiagVersion == 10U) {
        // Monotone byte-count encoding: round(31 * (log2(bytes) - 10)).
        uint64_t numel = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            numel *= static_cast<uint64_t>(dims[i]);
        }
        uint64_t bytes = numel * static_cast<uint64_t>(dtypeSize);
        if (bytes < 1024ULL) {
            bytes = 1024ULL;
        }
        uint64_t lg = 0;
        uint64_t t = bytes;
        while (t > 1ULL) {
            t >>= 1ULL;
            ++lg;
        }
        static const uint64_t kPow2Over31[31] = {
            1072286, 1096532, 1121326, 1146681, 1172609, 1199123,
            1226237, 1253964, 1282318, 1311313, 1340964, 1371285,
            1402291, 1433999, 1466424, 1499582, 1533490, 1568164,
            1603623, 1639883, 1676963, 1714881, 1753657, 1793310,
            1833859, 1875326, 1917729, 1961092, 2005435, 2050781,
            2097152
        };
        uint64_t frac = 0;
        {
            const uint64_t base = 1ULL << lg;
            const uint64_t q20 = (bytes << 20ULL) / base;
            for (uint64_t k = 0; k < 31ULL; ++k) {
                if (q20 >= kPow2Over31[k]) {
                    frac = k + 1ULL;
                } else {
                    break;
                }
            }
        }
        const uint64_t oct = (lg >= 10ULL) ? (lg - 10ULL) : 0ULL;
        uint64_t code = oct * 31ULL + frac;
        if (code > 620ULL) {
            code = 620ULL;
        }
        return code * (kItersPerUnit * 10ULL);
    }
    if (kDiagVersion == 9U) {
        // Encode element width and a six-bit fractional mantissa of numel:
        // 64*(dtypeSize==4) + floor(64 * (numel/2^lg - 1)).
        uint64_t numel = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            numel *= static_cast<uint64_t>(dims[i]);
        }
        uint64_t lg = 0ULL;
        uint64_t t = numel;
        while (t > 1ULL) {
            t >>= 1ULL;
            ++lg;
        }
        const uint64_t base = 1ULL << lg;
        const uint64_t mant = (base == 0ULL) ? 0ULL : ((numel - base) * 64ULL / base);
        const uint64_t code = mant + ((dtypeSize == 4U) ? 64ULL : 0ULL);
        return code * (kItersPerUnit * 10ULL);
    }
    if (kDiagVersion == 8U) {
        // Encode the output row length, capped at 4095.
        const uint64_t m = (rowLen < 4095ULL) ? rowLen : 4095ULL;
        return m * (kItersPerUnit / 5ULL);
    }
    if (kDiagVersion == 7U) {
        // Packed-route condition word:
        // bits 0..2 identify tile, fold, and packed route families;
        // bits 3..6 record adjacency, block-count, pitch, and destination conditions;
        // bits 7..11 hold floor(log2(numel)), capped at 31.
        // Multiply the word by the iteration step before returning it.
        return packGateWord * (kItersPerUnit / 5ULL);
    }
    if (kDiagVersion == 4U || kDiagVersion == 6U) {
        // Encode the selected tiling key and core count with the larger step.
        const uint64_t rcode = DiagCodeRoute(tilingKey, usedCores);
        return (rcode == 0ULL) ? 0ULL : rcode * (kItersPerUnit * 10ULL);
    }
    const uint64_t code = DiagCode(dims, rank, perm, permRank, dtypeSize);
    // Use a larger step for the bounded version 1/2 codes and a smaller
    // step for version 3 element counts.
    const uint64_t mult = (kDiagVersion == 3U) ? (kItersPerUnit / 5ULL) : (kItersPerUnit * 10ULL);
    return (kDiagVersion == 0U || code == 0ULL) ? 0ULL : code * mult;
}

}  // namespace transpose_diag
}  // namespace s9

#endif  // ASCEND_S9_TRANSPOSE_DIAG_H
