#ifndef S9_VARIANT_H
#define S9_VARIANT_H
/*
 * Host-side runtime selection among compiled tiling alternatives.
 *
 * S9_VARIANT selects an operator-specific alternative; zero uses the default.
 * S9_CORES clamps the core budget supplied to the planner, without increasing
 * it beyond the hardware count. Configuration is supplied before derivation,
 * so dependent fields remain consistent.
 *
 * Values are parsed as bounded decimal integers and cached per process.
 * Invalid values use the corresponding fallback. Non-default settings emit
 * a diagnostic message on stderr.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace s9 {
namespace variant_detail {

// Strict decimal parse. Anything else - a sign, a space, a suffix, an empty
// string, an overflow, a value past `hi` - returns `fallback`. Deliberately
// not strtoul: strtoul accepts "1abc" as 1 and " -1" as ULONG_MAX, and either
// would let a typo silently select an arm.
inline uint32_t ReadU32(const char* name, uint32_t fallback, uint32_t hi)
{
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') {
        return fallback;
    }
    uint32_t v = 0;
    for (const char* p = s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return fallback;
        }
        const uint32_t d = static_cast<uint32_t>(*p - '0');
        if (v > (0xffffffffu - d) / 10u) {
            return fallback;
        }
        v = v * 10u + d;
    }
    return (v > hi) ? fallback : v;
}

// Read each setting once per process. Function-local static initialization
// is thread-safe in C++11 and later.
inline uint32_t Once(const char* name, uint32_t hi)
{
    const uint32_t v = ReadU32(name, 0u, hi);
    if (v != 0u) {
        std::fprintf(stderr,
                     "[s9] %s=%u SELECTED. This run is NOT the shipped default "
                     "path; its number must not be reported as a score.\n",
                     name, v);
        std::fflush(stderr);
    }
    return v;
}

}  // namespace variant_detail
//
// Operator-specific selection. Zero is the default; values are bounded by 255.
inline uint32_t Variant()
{
    static const uint32_t v = variant_detail::Once("S9_VARIANT", 255u);
    return v;
}

// Core budget passed to the planner. A nonzero S9_CORES can reduce the
// hardware count but cannot increase it. S9_CORES=0 leaves it unchanged.
// S9_COLFLOOR controls the byte floor of a column piece.
// S9_REDCAP optionally limits redChunks on split routes; zero adds no cap.
inline uint32_t RedCap()
{
    static const uint32_t c = variant_detail::Once("S9_REDCAP", 4096u);
    return c;
}

// S9_SPLITKB overrides the reduction-split byte floor in KiB.
// Zero leaves the family-specific MIN_SPLIT_BYTES_FULL / _KEPT defaults in use.
// A nonzero value applies to both families.
inline uint32_t SplitFloorKB()
{
    static const uint32_t k = variant_detail::Once("S9_SPLITKB", 1048576u);
    return k;
}

// S9_RUNMAX is the byte ceiling for the batched reduced-inner-run routes.
// Larger runs use the generic odometer route.
inline uint32_t RunMaxBytes()
{
    static const uint32_t b = variant_detail::Once("S9_RUNMAX", 262144u);
    // The byte ceiling works with the layered route element bound
    // innerLen*8 <= accLen; both conditions must hold.
    return (b == 0u) ? 4096u : b;
}

inline uint32_t ColFloorBytes()
{
    static const uint32_t b = variant_detail::Once("S9_COLFLOOR", 4096u);
    // The column floor is applied when dividing the kept inner run among cores.
    return (b == 0u) ? 64u : b;
}

inline uint32_t CoreLimit(uint32_t hw)
{
    static const uint32_t c = variant_detail::Once("S9_CORES", 4096u);
    return (c == 0u || c >= hw) ? hw : c;
}

}  // namespace s9

#endif  // S9_VARIANT_H
