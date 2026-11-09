#ifndef S9_VARIANT_H
#define S9_VARIANT_H
/*
 * Runtime options shared by the host tiling code. S9_VARIANT selects a variant
 * (default zero); S9_CORES can only reduce the available core count. Parse each
 * option once per process, treating invalid values as zero. Apply the core cap
 * before deriving metadata, tiling keys and block dimensions. Nondefault options
 * are reported on stderr.
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

// Cache parsed values once per process using thread-safe local initialization.
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

// Variant zero selects the default route. Values above 255 are rejected.
inline uint32_t Variant()
{
    static const uint32_t v = variant_detail::Once("S9_VARIANT", 255u);
    return v;
}

// Apply a downward core-count cap and report a nondefault setting.
inline uint32_t CoreLimit(uint32_t hw)
{
    static const uint32_t c = variant_detail::Once("S9_CORES", 4096u);
    return (c == 0u || c >= hw) ? hw : c;
}

}  // namespace s9

#endif  // S9_VARIANT_H
