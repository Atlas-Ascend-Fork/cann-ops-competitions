#ifndef GREATER_KERNEL_DIAG_H
#define GREATER_KERNEL_DIAG_H
/*
 * Optional kernel-side diagnostic delay. The host encodes an integer in the
 * route-specific metadata payload. Enabled diagnostics consume that code after
 * the computation route drains its output transfers. GT_DIAG == 0 excludes
 * the diagnostic body. Keep GT_DIAG aligned with host GT_DIAG_VERSION.
 */

// Zero disables diagnostics; one enables the diagnostic build.
#ifndef GT_DIAG
#define GT_DIAG 0
#endif
#ifndef GT_DIAG_TIMER
#define GT_DIAG_TIMER 0
#endif
#ifndef GT_CLOCK_STEP_US
#define GT_CLOCK_STEP_US 1000
#endif

namespace {

// Diagnostic-word offsets follow the serialized route payloads:
// aligned route 8, one-tile route 16, general route 224.
// Keep these offsets consistent with the host payload sizes and diagCode layout.
constexpr uint32_t GT_DIAG_OFF_FAST = 8;
constexpr uint32_t GT_DIAG_OFF_ONE = 16;
constexpr uint32_t GT_DIAG_OFF_GEN = 224;

// Iterations of the dependent chain per diagnostic code unit.
constexpr uint64_t GT_DIAG_ITERS_PER_UNIT = 58000ULL * 10ULL;

// A dependent 64-bit LCG chain with a workspace side effect prevents removal
// of the diagnostic loop. The sink is workspace[1]; workspace[0] is reserved
// for launch metadata. Call this after the route drains its output transfers.
// Timer mode instead uses a deadline measured from kernel entry. Its clock
// conversion and deadline floor must be valid for the target device.
__aicore__ inline uint64_t GtDiagStart()
{
#if GT_DIAG && GT_DIAG_TIMER
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
#else
    return 0;
#endif
}

__aicore__ inline void GtDiagSpin(GM_ADDR tiling, GM_ADDR workspace, uint32_t off, uint64_t start)
{
#if GT_DIAG
    if (GetBlockIdx() != 0) {
        return;   // Only block zero performs the diagnostic wait.
    }
    const uint64_t code = (reinterpret_cast<__gm__ uint64_t *>(tiling))[off / 8];
#if GT_DIAG_TIMER
    // One code unit adds one millisecond to a ten-millisecond deadline floor.
    // Only block zero waits. The floor must exceed the remaining cores
    // computation time for the diagnostic measurement to be interpretable.
    const uint64_t cycles = (10000ULL + GT_CLOCK_STEP_US * code) * 50ULL;
    while (static_cast<uint64_t>(AscendC::GetSystemCycle()) - start < cycles) {}
    (void)workspace;
#else
    if (code == 0) {
        return;
    }
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ code;
    const uint64_t n = code * GT_DIAG_ITERS_PER_UNIT;
    for (uint64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    (reinterpret_cast<__gm__ uint64_t *>(workspace))[1] = s;
#endif
#else
    (void)tiling;
    (void)workspace;
    (void)off;
#endif
    (void)start;
}

}  // namespace

#endif  // GREATER_KERNEL_DIAG_H
