#ifndef GREATER_DIAG_H
#define GREATER_DIAG_H
/*
 * Optional host-side diagnostic encoding applied after tiling.
 * GT_DIAG_VERSION == 0 disables encoding. Keep the kernel diagnostic flag
 * and route-specific payload offsets consistent with this configuration.
 */

#include "greater_tiling_core.h"
#include "greater_clock_probe.h"

#include <cstdint>

namespace greater_diag {

// Diagnostic version zero disables the optional diagnostic payload and work.
#ifndef GT_DIAG_VERSION
#define GT_DIAG_VERSION 0
#endif

constexpr uint32_t GT_DIAG_V_OFF   = 0;  // Diagnostics disabled.
constexpr uint32_t GT_DIAG_V_CONST = 1;  // Constant diagnostic code.
constexpr uint32_t GT_DIAG_V_ROUTE = 2;  // Encode the selected tiling key.
constexpr uint32_t GT_DIAG_V_SIZE  = 3;  // Encode the output element-count range.
constexpr uint32_t GT_DIAG_V_INNER = 4;  // Encode the inner extent.
constexpr uint32_t GT_DIAG_V_GEOM  = 5;  // Encode dimension and broadcast mode metadata.
constexpr uint32_t GT_DIAG_V_CORES = 6;  // Encode the selected core count.

// Bound the diagnostic code before serializing it.
constexpr uint64_t GT_DIAG_MAX_CODE = 900;

// Integer logarithm quantization with fixed thresholds for deterministic encoding.
inline uint64_t Lg31(uint64_t v)
{
    if (v == 0) {
        return 0;
    }
    static const uint64_t TH[31] = {
        0x0000000102E0CA33ULL, 0x0000000108BB5011ULL, 0x000000010EB7B82CULL, 0x0000000114D6C6A9ULL,
        0x000000011B194419ULL, 0x00000001217FFD97ULL, 0x00000001280BC4E3ULL, 0x000000012EBD707AULL,
        0x000000013595DBB0ULL, 0x000000013C95E6D4ULL, 0x0000000143BE7742ULL, 0x000000014B10778AULL,
        0x00000001528CD787ULL, 0x000000015A348C80ULL, 0x000000016208914CULL, 0x000000016A09E668ULL,
        0x0000000172399222ULL, 0x000000017A98A0B6ULL, 0x000000018328246EULL, 0x000000018BE935CBULL,
        0x0000000194DCF3A4ULL, 0x000000019E04834BULL, 0x00000001A76110B8ULL, 0x00000001B0F3CEA6ULL,
        0x00000001BABDF6C5ULL, 0x00000001C4C0C9D9ULL, 0x00000001CEFD8FE7ULL, 0x00000001D9759861ULL,
        0x00000001E42A3A4EULL, 0x00000001EF1CD474ULL, 0x00000001FA4ECD8BULL,
    };
    uint32_t e = 63;
    while (((v >> e) & 1ULL) == 0ULL) {
        --e;
    }
    const uint64_t mq = (e <= 32) ? (v << (32 - e)) : (v >> (e - 32));  // Quantize the mantissa for the diagnostic code.
    uint32_t k = 0;
    while (k < 31 && mq >= TH[k]) {
        ++k;
    }
    return static_cast<uint64_t>(31) * e + k;
}

// Map the dispatch key to a diagnostic index.
inline uint64_t KeyIndex(uint32_t dtypeIdx, uint32_t family)
{
    return static_cast<uint64_t>(dtypeIdx) * 4 + family;
}

// Construct the diagnostic code from the completed tiling plan and selected version.
inline uint64_t CodeFor(uint32_t version, const greater_core::Plan &plan, uint32_t family, uint32_t dtypeIdx)
{
    uint64_t c = 0;
    switch (version) {
        case GT_DIAG_V_OFF:
            c = 0;
            break;

        // Constant code for this diagnostic version.
        case GT_DIAG_V_CONST:
            c = 700;
            break;

        // Encode the key index as 40 * (keyIndex + 1).
        case GT_DIAG_V_ROUTE:
            c = 40 * (KeyIndex(dtypeIdx, family) + 1);
            break;

        // Encode the output element-count range.
        case GT_DIAG_V_SIZE:
            c = Lg31(plan.totalNum);
            break;

        // Encode the inner extent.
        case GT_DIAG_V_INNER:
            c = Lg31(plan.innerLen);
            break;

        // Encode dimension and broadcast mode metadata.
        case GT_DIAG_V_GEOM:
            c = 100 * static_cast<uint64_t>(plan.dimNum) + 30 * static_cast<uint64_t>(plan.innerMode);
            break;

        // Encode the selected core count.
        case GT_DIAG_V_CORES:
            c = 20 * static_cast<uint64_t>(plan.usedCores);
            break;

        default:
            c = 0;
            break;
    }
    return (c > GT_DIAG_MAX_CODE) ? GT_DIAG_MAX_CODE : c;
}

inline uint64_t Code(const greater_core::Plan &plan, uint32_t family, uint32_t dtypeIdx)
{
#if GT_DIAG_VERSION == 100
    return greater_clock::Code(plan, family, dtypeIdx);
#else
    return CodeFor(GT_DIAG_VERSION, plan, family, dtypeIdx);
#endif
}

// Enabled diagnostics append a word and reserve a scratch sink.
constexpr bool Enabled() { return GT_DIAG_VERSION != GT_DIAG_V_OFF; }

// Select the diagnostic-word offset from the kernel reader layout, including
// when the host serializes a full payload for a route that reads a prefix.
struct BlobLayout {
    uint64_t diagOffset;    // where the host writes the code and the kernel reads it
    uint64_t shippedBytes;  // what SetDataSize is given
};

inline BlobLayout Layout(uint32_t family, bool fullTilingArm, bool diagOn)
{
    BlobLayout L{};
    uint64_t shipped = greater_core::GT_GEN_TILING_BYTES;
    if (!fullTilingArm) {
        if (family == greater_core::FAM_FAST) {
            shipped = greater_core::GT_FAST_TILING_BYTES;
        } else if (family == greater_core::FAM_ONE) {
            shipped = greater_core::GT_ONE_TILING_BYTES;
        }
    }
    L.diagOffset = (family == greater_core::FAM_FAST)  ? greater_core::GT_FAST_TILING_BYTES
                   : (family == greater_core::FAM_ONE) ? greater_core::GT_ONE_TILING_BYTES
                                                       : greater_core::GT_DIAG_OFFSET_GEN;
    if (diagOn && shipped < L.diagOffset + 8) {
        shipped = L.diagOffset + 8;
    }
    L.shippedBytes = shipped;
    return L;
}

// Accept regular and diagnostic payload sizes for the selected reader layout.
// Reject a short payload if the route could read beyond it.
inline bool ShippedSizeIsLegal(uint32_t family, uint64_t bytes)
{
    for (int arm = 0; arm <= 1; ++arm) {
        for (int on = 0; on <= 1; ++on) {
            if (Layout(family, arm != 0, on != 0).shippedBytes == bytes) {
                return true;
            }
        }
    }
    return false;
}

// The diagnostic scratch sink is one uint64_t at workspace offset eight.
// The first uint64_t is reserved for the launch metadata.
constexpr uint64_t GT_DIAG_WORKSPACE = 32;

}  // namespace greater_diag

#endif  // GREATER_DIAG_H
