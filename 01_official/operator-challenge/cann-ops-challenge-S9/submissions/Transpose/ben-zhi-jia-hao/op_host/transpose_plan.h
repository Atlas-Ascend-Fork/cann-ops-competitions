// Route selection and tile sizing for Transpose (torch.permute).
// The planner uses integer arithmetic on transpose_geometry.h and has no
// CANN dependency. Inputs include dtype width, collapsed geometry, UB size,
// and the available core count.

#ifndef ASCEND_S9_TRANSPOSE_PLAN_H
#define ASCEND_S9_TRANSPOSE_PLAN_H

#include <cstdint>

#include "transpose_fields.h"
#include "transpose_geometry.h"

#ifndef TP_DEFAULT_REPAIR
#define TP_DEFAULT_REPAIR 1
#endif

namespace s9 {
namespace transpose {

// ---------------------------------------------------------------------------
// Tiling keys. 1 and 2 together cover every legal input, so the fallback is
// always present in the binary.
// ---------------------------------------------------------------------------
constexpr uint32_t kKeyBulk = 1;  // rows of L contiguous elements, chunked
constexpr uint32_t kKeyRows = 2;  // rows assembled from nblk blocks, batched
constexpr uint32_t kKeyTile = 3;  // 2-D tile transposed in UB with Gather
constexpr uint32_t kKeyFold = 4;  // a run of constant-stride rows as ONE descriptor
constexpr uint32_t kKeyTileOne = 5;  // the tile route when a core owns exactly ONE tile
constexpr uint32_t kKeyTileI8 = 6;  // The tile route for int8 bridges through half.
// The following switch controls eligibility of packed tile and fold routes.
constexpr bool kPackedRoutesOn = true;

// Optional pack-aware fold search, disabled by default.
// The search tracks candidates whose output rows are adjacent and whose packed
// run fits the Gather element bound. The comparison includes DMA descriptors
// and blocks; additional Gather work remains a separate route cost.
#ifndef S9_TRANSPOSE_FOLD_PACK_AWARE
#define S9_TRANSPOSE_FOLD_PACK_AWARE 0
#endif
constexpr bool kFoldPackAware = S9_TRANSPOSE_FOLD_PACK_AWARE != 0;
constexpr uint32_t kKeyTilePack = 7;  // the tile route whose store is ONE block
constexpr uint32_t kKeyFoldPack = 8;  // the fold route whose store is ONE block
constexpr uint32_t kKeyTileWide = 9;  // full-K panel via TransDataTo5HD (b32)
constexpr uint32_t kKeyOutput = 13;  // Complete output-row panels (b32).
// The wide-panel route is enabled by default. Its automatic gate requires
// a panel width of at least 32 elements; explicit selection retains other widths.
constexpr bool kWideRouteOn = true;
constexpr uint32_t kVarTileWide = 21;     // force ON  (redundant while default)
constexpr uint32_t kVarTileWideOff = 22;  // Force the wide-panel route off.

constexpr uint32_t kKeyTileVia = 10;  // Two-stage wide transpose through workspace.
// This route is disabled by default and remains selectable through kVarTileVia.
// The two stages execute in order so scratch writes complete before reads.
constexpr bool kViaRouteOn = false;
constexpr uint32_t kVarTileVia = 23;     // force ON  (redundant while default)
constexpr uint32_t kVarTileViaOff = 24;  // Force the workspace route off.

constexpr uint32_t kKeyTileCube = 11;  // Cube-assisted two-stage transpose (MIX 1:2).
// S9_VARIANT=25 enables this route: the AIC nd2nz transfer rearranges the grid,
// and its two AIVs process the 8x8 fractals and pack output rows.
constexpr uint32_t kVarTileCube = 25;
// Key 11's chunk rows are SOLVED per shape, dual-L1-slot preferred: two
// slots let the AIC's nd2nz overlap the previous chunk's scratch store (legs
// cost max, not sum), which beats a fatter single slot's wider output rows.
// The L1 budget is EMPIRICAL: 932 KB of dual slots threw an aicore exception
// while 399 KB single ran clean; the knob ladder pins the true ceiling.
constexpr uint32_t kKeyTileL2 = 12;  // Two-pass transpose through scratch. Pass A writes dense panels from the
// slab/vnchwconv path. Pass B collects a 16-row group from the panels, assembles
// the output layout with Gather, and stores the group contiguously.
// The scratch allocation is bounded to allow reuse through L2.
constexpr uint32_t kVarTileL2 = 26;     // force ON (measurement arm)
constexpr uint32_t kVarTileL2Off = 27;  // reserved OFF control
constexpr uint64_t kL2PanelW = 48;      // pass-A panel width (elems); the
// pipelined kernel co-resides pass A's slabs+panel with pass B's chunk in
// UB, so the panel stays at 48 and the chunk takes the remainder.
constexpr uint64_t kL2ScratchCap = 96ull << 20;  // stay in half the 192MB L2

constexpr uint32_t kCubeNcMax = 240;   // nd2nz nValue is an 8-bit field; 240 is the largest multiple of 16 below 255.
// The L1 budget also constrains the number and size of staging slots.
constexpr uint64_t kCubeL1Budget = 499200ull;
// Bytes the AscendC runtime reserves at the front of the workspace; the
// kernel's GetUserWorkspace(workspace) skips exactly this much. Same constant
// SquareSumV1 ships (its plan header, SYS_WORKSPACE).
constexpr uint64_t kSysWorkspaceBytes = 16ull * 1024 * 1024;

constexpr int kPlanMaxOdo = 8;
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kMaxBlockCount = 4095;   // DataCopy blockCount practical cap
constexpr uint64_t kUint32Max = 0xFFFFFFFFull;

// Optional total-byte clamp for the bulk route. smallClamp=1 keeps the
// 64 KiB / 8-core limit; the default disables this clamp.
constexpr uint64_t kSmallTensorBytes = 64u * 1024u;
constexpr uint32_t kSmallTensorCores = 8;

// Optional minimum work per core, disabled by default for this operator.
// The floor is an input to tile and block-dimension derivation, so changing
// it must recompute dependent geometry.
constexpr uint64_t kFloorElemsPerCore = 4096;

// Minimum core budget passed to the tile ladder when the work floor is
// enabled. A narrower budget can make the one-shot tile exceed its UB limit.
constexpr uint32_t kMinFloorCores = 8;

// S9_VARIANT=9 enables the per-core work floor at kFloorElemsPerCore.
// S9_CORES separately bounds the hardware budget supplied to the planner.
constexpr uint32_t kVarCoreFloor = 9;

// A row whose contiguous run is at least this wide is already a healthy DMA
// burst on its own, so the per-row bulk route beats batching many rows.
constexpr uint32_t kBulkRowBytes = 512;

// One Gather instruction handles at most 255 repeats; 8192 elements stays well
// inside that for every supported dtype and keeps the tile's UB footprint small
// enough to leave room for double buffering later.
constexpr uint64_t kMaxGatherElems = 8192;

// DMA cost model in picosecond units:
// pipe cost = bytes*Byte + blocks*Block. Read and write pipes use separate
// constants, and the common blockDim factor is omitted when comparing tiles.
// These constants approximate small-block transfers; they do not model
// large contiguous transfers or the additional Gather work of packed routes.
constexpr uint64_t kPipeReadByte = 122;     // ps per byte read    (328 GB/s)
constexpr uint64_t kPipeReadBlock = 13200;  // ps per MTE2 block
constexpr uint64_t kPipeWriteByte = 126;    // ps per byte written (317 GB/s)
constexpr uint64_t kPipeWriteBlock = 21100; // Picoseconds per MTE3 block.
//
// A packed tile store requires a complete output row (mt == bigM). Gather
// needs input and output resident together, so UB use includes at least
// 2 * bigM * kt * dtypeSize plus the table. Eligibility must check this fit.
//
// A zero destination stride makes DMA blocks adjacent in GM but does not merge
// them. UB rows are padded to 32 bytes, so a merged store requires a genuinely
// packed UB source. kKeyTilePack supplies that packed source.
//
// The one-shot route has a compile-time UB map bounded by kOneTileElems.
// The host must not select it for a larger tile. The limit matches the Gather
// element cap; float32 input, output, and table occupy 8192*(4+4+4) bytes.
constexpr uint64_t kOneTileElems = kMaxGatherElems;

// Minimum tile edge, in 32-byte data blocks, allowed during the
// core-filling shrink. This preserves a minimum DMA burst width.
constexpr uint32_t kTileEdgeFloorBlocks = 1;

// Minimum work a core must be given on the row engine to be worth launching.
constexpr uint64_t kMinBlocksPerCore = 128;
constexpr uint64_t kMinBytesPerCore = 32u * 1024u;

struct TransposePlan {
    bool valid = false;
    uint32_t tilingKey = kKeyBulk;
    uint32_t blockDim = 1;

    uint32_t dtypeSize = 1;
    uint64_t numel = 0;

    // Row engine: rowCount rows of rowLen elements; destination row r starts
    // at r * rowLen.
    uint64_t rowCount = 1;
    uint64_t rowLen = 0;

    // A row is nblk DMA blocks of blen elements; block b of row r starts at
    // srcBase(r) + b * srcBlkPitch. Either nblk == 1 (contiguous suffix) or
    // blen == 1 (true transpose), never both > 1.
    uint32_t blkPerRow = 1;
    uint32_t blkLenElems = 0;
    uint64_t srcBlkPitch = 1;

    // Tile route (kKeyTile): the exchanged axes are rowLen (output innermost,
    // input stride srcBlkPitch) and tileK (input innermost, output stride
    // dstRowStride). rowCount is the batch count over the remaining axes.
    uint64_t tileK = 0;
    uint64_t dstRowStride = 0;
    uint32_t tileMT = 0;
    uint32_t tileKT = 0;
    uint64_t outerOutStride[kPlanMaxOdo] = {};

    uint32_t rowsPerBatch = 1;   // rows gathered before one store (kKeyRows/kKeyFold)
    uint32_t chunkBlks = 1;      // blocks of a row staged at once (kKeyRows)
    uint32_t slotBytes = 32;     // UB bytes one block occupies
    uint64_t chunkElems = 1;     // elements per iteration (kKeyBulk)
    uint32_t ubBufBytes = 0;     // bytes per queue buffer

    // Fold route (kKeyFold): `foldLen` consecutive rows along one odometer axis
    // are described by a single strided DataCopyPad. The axis has extent
    // `foldExt`, source stride `foldSrcStride` elements, and moves the output
    // row index by `foldDstRowStride` rows per step. The odometer below covers
    // every OTHER axis, so a work unit is (odometer position, fold chunk).
    uint32_t foldLen = 0;           // rows per descriptor; 0 = route unused
    uint64_t foldExt = 0;
    uint64_t foldSrcStride = 0;
    uint64_t foldDstRowStride = 1;

    // Tile prefetch: 1 issues tile n+1 input before waiting on tile n Gather;
    // 0 keeps the load and its wait in the same iteration.
    uint32_t tilePrefetch = 1;

    // DIAGNOSTIC ONLY -- written by BuildPlan, read by nothing but the delay
    // probe.  Bit layout in transpose_diag.h, version 7.  Never gate on it.
    uint64_t packGateWord = 0;

    // Odometer over the outer output axes, index 0 slowest.
    uint32_t outerRank = 0;
    uint64_t outerShape[kPlanMaxOdo] = {};
    uint64_t outerInStride[kPlanMaxOdo] = {};
};

namespace detail {

inline uint64_t AlignUp(uint64_t value, uint64_t grain)
{
    return (value + grain - 1) / grain * grain;
}

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

}  // namespace detail
//
// Spread work items over at most coreLimit cores without assigning empty work.
// clampSmall optionally applies the total-byte clamp used by the bulk route.
inline uint32_t PickBlockDim(uint64_t work, uint64_t totalBytes, uint32_t coreLimit,
                             bool clampSmall)
{
    uint32_t cores = coreLimit == 0 ? 1 : coreLimit;
    if (clampSmall && totalBytes <= kSmallTensorBytes && cores > kSmallTensorCores) {
        cores = kSmallTensorCores;
    }
    if (work == 0) {
        return 1;
    }
    if (static_cast<uint64_t>(cores) > work) {
        cores = static_cast<uint32_t>(work);
    }
    return cores == 0 ? 1 : cores;
}

// Choose the widest UB-fitting bulk chunk that still leaves work per core.
// smallClamp applies the optional 64 KiB / 8-core limit to both chunk sizing
// and block-dimension derivation; changing only blockDim would be inconsistent.
inline void SizeBulkChunks(TransposePlan& plan, uint64_t bufBytes, uint64_t totalBytes,
                           uint32_t coreNum, bool smallClamp)
{
    const uint64_t elemsPerBlock = kDataBlockBytes / plan.dtypeSize;
    uint32_t cores = coreNum == 0 ? 1 : coreNum;
    if (smallClamp && totalBytes <= kSmallTensorBytes && cores > kSmallTensorCores) {
        cores = kSmallTensorCores;
    }
    uint64_t maxChunk = bufBytes / plan.dtypeSize;
    if (maxChunk == 0) {
        maxChunk = 1;
    }
    uint64_t wantPerRow = detail::CeilDiv(cores, plan.rowCount);
    if (wantPerRow == 0) {
        wantPerRow = 1;
    }
    uint64_t chunk = detail::AlignUp(detail::CeilDiv(plan.rowLen, wantPerRow), elemsPerBlock);
    if (chunk == 0) {
        chunk = elemsPerBlock;
    }
    if (chunk > maxChunk) {
        chunk = maxChunk;
    }
    if (chunk > plan.rowLen) {
        chunk = plan.rowLen;
    }
    if (chunk == 0) {
        chunk = 1;
    }
    plan.chunkElems = chunk;
    const uint64_t chunksPerRow = detail::CeilDiv(plan.rowLen, chunk);
    plan.blockDim = PickBlockDim(plan.rowCount * chunksPerRow, totalBytes, coreNum, smallClamp);
}

// ubSize and coreNum are runtime platform limits.
// allowTile controls whether the tile family may be selected.
// tileFloorBlocks bounds tile-edge shrinking in 32-byte blocks.
// foldMode: 0 disables folding, 1 uses the innermost odometer axis, and
// 2 selects the axis with the fewest descriptors.
// tilePrefetch chooses the tile dependency structure.
// foldCoreCap optionally applies the row-engine work cap to fold routes.
// tileBalance: 0 uses the core-filling ladder; 1 uses the balanced ladder only
// when it reaches the one-shot route; 2 always uses the balanced ladder.
// oneShot permits the straight-line tile route when its preconditions hold.
// smallClamp enables the optional total-byte clamp for bulk work.
// minElemsPerCore is a per-core element floor; zero disables it. Runtime
// environment settings are supplied by the caller, keeping this header
// independent of the environment parser.
inline bool BuildPlan(
    const int64_t* shape, int rank, const int64_t* rawPerm, int permRank,
    uint32_t dtypeSize, uint64_t ubSize, uint32_t coreNum, uint32_t bufferNum,
    TransposePlan& plan, bool allowTile = true,
    uint32_t tileFloorBlocks = kTileEdgeFloorBlocks, uint32_t foldMode = 2,
    uint32_t tilePrefetch = 1, uint32_t foldCoreCap = 0, uint32_t smallClamp = 0,
    uint32_t tileBalance = 1, uint32_t oneShot = 1,
    uint64_t minElemsPerCore = 0,
    // Optional geometry overrides; 65535 means unset. Pass them from the caller
    // to keep runtime environment handling outside this planner header.
    uint32_t mtOverride = 65535u, uint32_t ktOverride = 65535u,
    // Wide-route selection: 0 follows kWideRouteOn, 1 requests the eligible
    // wide route, and 2 disables it. The caller resolves runtime settings.
    uint32_t wideArm = 0,
    // kKeyTileVia arm (kVarTileVia): 1 sends the eligible wide-b32 family to
    // the two-stage workspace route. Default 0 = off; measurement only.
    uint32_t viaArm = 0,
    // kKeyTileCube arm (kVarTileCube): 1 sends the eligible family to the
    // MIX cube-assisted route. Default 0 = off; measurement only.
    uint32_t cubeArm = 0,
    // L2-route selection: 1 requests the eligible b32 two-pass route;
    // zero leaves it disabled.
    uint32_t l2Arm = 0)
{
    plan = TransposePlan{};
    plan.dtypeSize = dtypeSize;
    plan.tilePrefetch = tilePrefetch;

    Geometry g;
    if (!BuildGeometry(shape, rank, rawPerm, permRank, g)) {
        return false;
    }
    if (dtypeSize == 0 || bufferNum == 0) {
        return false;
    }
    plan.numel = static_cast<uint64_t>(g.numel);

    const uint64_t reserve = 8192;
    uint64_t usable = ubSize > reserve ? ubSize - reserve : ubSize / 2;
    uint64_t bufBytes = usable / bufferNum;
    bufBytes = bufBytes / kDataBlockBytes * kDataBlockBytes;
    if (bufBytes < kDataBlockBytes) {
        bufBytes = kDataBlockBytes;
    }
    plan.ubBufBytes = static_cast<uint32_t>(bufBytes);

    const uint64_t totalBytes = plan.numel * dtypeSize;

    // Apply the per-core work floor before deriving chunk sizes and tile geometry.
    // The one-shot route requires units == blockDim, so changing blockDim after
    // the tile ladder runs could invalidate its launch precondition.
    if (minElemsPerCore != 0) {
        uint64_t byWork = detail::CeilDiv(plan.numel, minElemsPerCore);
        if (byWork == 0) {
            byWork = 1;
        }
        if (byWork < static_cast<uint64_t>(kMinFloorCores)) {
            byWork = kMinFloorCores;      // see kMinFloorCores: keeps kKeyTileOne
        }
        if (byWork < static_cast<uint64_t>(coreNum)) {
            coreNum = static_cast<uint32_t>(byWork);
        }
    }

    // Empty tensor: nothing to move, but the kernel still has to be launchable.
    if (plan.numel == 0) {
        plan.valid = true;
        plan.tilingKey = kKeyBulk;
        plan.blockDim = 1;
        plan.rowCount = 0;
        plan.rowLen = 0;
        plan.chunkElems = 1;
        plan.blkLenElems = 0;
        return true;
    }

    // ---------------------------------------------------------------------
    // Family A: the permutation does not move any element. Flat copy.
    // ---------------------------------------------------------------------
    if (g.reshapeOnly || g.collapsedRank <= 1) {
        plan.tilingKey = kKeyBulk;
        plan.outerRank = 0;
        plan.rowCount = 1;
        plan.rowLen = plan.numel;
        plan.blkPerRow = 1;
        plan.blkLenElems = 0;  // unused on the bulk route
        SizeBulkChunks(plan, bufBytes, totalBytes, coreNum, smallClamp != 0);
        plan.valid = true;
        return true;
    }

    // ---------------------------------------------------------------------
    // Canonical reduced problem: collapsed shape and permutation.
    // ---------------------------------------------------------------------
    const int R = g.collapsedRank;
    int64_t cshape[kMaxRank];
    int32_t cperm[kMaxRank];
    for (int i = 0; i < R; ++i) {
        cshape[i] = g.collapsedShape[i];
        cperm[i] = g.collapsedPerm[i];
    }
    int64_t cinStride[kMaxRank];
    int64_t coutShape[kMaxRank];
    int64_t coutStride[kMaxRank];
    {
        int64_t s = 1;
        for (int a = R - 1; a >= 0; --a) {
            cinStride[a] = s;
            s *= cshape[a];
        }
        for (int i = 0; i < R; ++i) {
            coutShape[i] = cshape[cperm[i]];
        }
        s = 1;
        for (int i = R - 1; i >= 0; --i) {
            coutStride[i] = s;
            s *= coutShape[i];
        }
    }

    // Outer odometer: output axes 0 .. R-2, index 0 slowest. The destination
    // base of row r is the row-major sum over those axes of the OUTPUT strides,
    // and because they are the leading axes of a row-major output that sum is
    // exactly r * coutShape[R-1] = r * rowLen. That identity holds in both
    // families, so the kernel never needs the output strides themselves.
    plan.outerRank = static_cast<uint32_t>(R - 1);
    for (int a = 0; a < R - 1; ++a) {
        plan.outerShape[a] = static_cast<uint64_t>(coutShape[a]);
        plan.outerInStride[a] = static_cast<uint64_t>(cinStride[cperm[a]]);
    }

    const int p = cperm[R - 1];
    if (p == R - 1) {
        // -----------------------------------------------------------------
        // Family B: the innermost input axis is still innermost in the
        // output, so each row is one contiguous run in both tensors.
        // -----------------------------------------------------------------
        plan.rowLen = static_cast<uint64_t>(cshape[R - 1]);
        plan.rowCount = plan.numel / plan.rowLen;
        plan.blkPerRow = 1;
        plan.blkLenElems = 0;
        plan.srcBlkPitch = plan.rowLen;
    } else {
        // -----------------------------------------------------------------
        // Family C: a genuine transpose. Row = one output-innermost run of M
        // elements, gathered from the source at stride cinStride[p].
        // -----------------------------------------------------------------
        plan.rowLen = static_cast<uint64_t>(cshape[p]);
        plan.rowCount = plan.numel / plan.rowLen;
        plan.blkPerRow = 0;  // filled below
        plan.blkLenElems = 1;
        plan.srcBlkPitch = static_cast<uint64_t>(cinStride[p]);
    }

    const uint64_t rowBytes = plan.rowLen * dtypeSize;

    if (p == R - 1 && rowBytes >= kBulkRowBytes) {
        // Wide contiguous rows: one DMA pair per chunk already saturates the
        // burst, and this is the only route that survives rows larger than UB.
        plan.tilingKey = kKeyBulk;
        SizeBulkChunks(plan, bufBytes, totalBytes, coreNum, smallClamp != 0);
        plan.valid = true;
        return true;
    }

    // Transpose a 2-D tile in UB with Gather. Both exchanged axes must be at
    // least one 32-byte data block wide. b16/b32 routes gather their native type.
    // The int8 route casts through half because Gather has no one-byte form.
    // Every int8 value is exactly representable in half, so this data permutation
    // round trip is exact. Its UB budget includes two half work buffers.
    // The int8 route does not use the one-shot kernel.
    const bool isI8 = (dtypeSize == 1);
    if (allowTile && p != R - 1 && (dtypeSize == 2 || dtypeSize == 4 || isI8)) {
        const uint64_t bigM = static_cast<uint64_t>(cshape[p]);
        const uint64_t bigK = static_cast<uint64_t>(cshape[R - 1]);
        int q = -1;
        for (int i = 0; i < R - 1; ++i) {
            if (cperm[i] == R - 1) {
                q = i;
                break;
            }
        }
        const uint64_t dq = (q >= 0) ? static_cast<uint64_t>(coutStride[q]) : 0;
        const uint64_t sp = static_cast<uint64_t>(cinStride[p]);
        const uint64_t elemsPerBlock = kDataBlockBytes / dtypeSize;

        if (q >= 0 && bigM * dtypeSize >= kDataBlockBytes &&
            bigK * dtypeSize >= kDataBlockBytes && sp * dtypeSize <= kUint32Max &&
            dq * dtypeSize <= kUint32Max) {
            // Both tile edges are whole data blocks, so every DMA burst is at
            // least 32 bytes and the UB row pitches are expressible as the
            // 32-byte strides DataCopyPad takes.
            uint64_t mt = elemsPerBlock;
            uint64_t kt = elemsPerBlock;
            // Grow the tile while it still fits: ubIn + ubOut + offset table.
            // The int8 bridge additionally needs BOTH half work buffers resident
            // (2 bytes/elem each: the Cast target the Gather reads, and the
            // Gather's half output the Cast-back consumes). A dtypeSize-only
            // budget would undercount them by 4*m*k bytes and could size a tile
            // that overruns UB, so the int8 arm counts them explicitly.
            auto tileBytes = [&](uint64_t m, uint64_t k) -> uint64_t {
                if (isI8) {
                    return static_cast<uint64_t>(bufferNum) * m * k * dtypeSize  // int8 in queue
                         + static_cast<uint64_t>(bufferNum) * k * m * dtypeSize  // int8 out queue
                         + m * k * sizeof(uint16_t)                             // half work in
                         + k * m * sizeof(uint16_t)                             // half work out
                         + k * m * sizeof(uint32_t);                           // gather offsets
                }
                return static_cast<uint64_t>(bufferNum) * m * k * dtypeSize  // ubIn queue
                     + static_cast<uint64_t>(bufferNum) * k * m * dtypeSize  // ubOut queue
                     + k * m * sizeof(uint32_t);                            // gather offsets
            };
            const uint64_t budget = usable > 4096 ? usable - 4096 : usable;
            // One Gather covers the whole tile, so the element count also has
            // to stay inside the vector unit's repeat limit.
            bool grew = true;
            while (grew) {
                grew = false;
                if (mt * 2 <= bigM && tileBytes(mt * 2, kt) <= budget &&
                    mt * 2 <= kMaxBlockCount && mt * 2 * kt <= kMaxGatherElems) {
                    mt *= 2;
                    grew = true;
                }
                if (kt * 2 <= bigK && tileBytes(mt, kt * 2) <= budget &&
                    kt * 2 <= kMaxBlockCount && mt * kt * 2 <= kMaxGatherElems) {
                    kt *= 2;
                    grew = true;
                }
            }
            const uint64_t batchCount = plan.numel / (bigM * bigK);
            // A tile is one work unit. Having grown it as large as UB allows,
            // shrink the longer edge back until there are at least as many
            // units as cores, so the machine is not left idle. The floor is
            // one 32-byte data block per edge, which is the point where a DMA
            // burst stops being a full block.
            const uint64_t floorElems =
                elemsPerBlock * (tileFloorBlocks == 0 ? 1 : tileFloorBlocks);
            const uint64_t coreCap = coreNum == 0 ? 1 : static_cast<uint64_t>(coreNum);
            auto unitsAt = [&](uint64_t m, uint64_t k) {
                return batchCount * detail::CeilDiv(bigM, m) * detail::CeilDiv(bigK, k);
            };
            // Two tile-shrink ladders start from the same grown tile.
            // Ladder A stops at the first rung with at least coreNum units.
            // Ladder B stops at the last rung within coreNum units, assigning one tile
            // to each launched core. The default uses B only when its plan qualifies
            // for the one-shot route; other plans keep ladder A.
            auto shrinkA = [&](uint64_t& m, uint64_t& k) {
                while (unitsAt(m, k) < coreCap) {
                    if (k >= m && k > floorElems) {
                        k /= 2;
                    } else if (m > floorElems) {
                        m /= 2;
                    } else {
                        break;
                    }
                }
            };
            auto shrinkB = [&](uint64_t& m, uint64_t& k) {
                for (;;) {
                    uint64_t nm = m;
                    uint64_t nk = k;
                    if (k >= m && k > floorElems) {
                        nk = k / 2;
                    } else if (m > floorElems) {
                        nm = m / 2;
                    } else {
                        break;
                    }
                    if (unitsAt(nm, nk) > coreCap) {
                        break;
                    }
                    m = nm;
                    k = nk;
                }
            };
            uint64_t mtA = mt;
            uint64_t ktA = kt;
            shrinkA(mtA, ktA);
            uint64_t mtB = mt;
            uint64_t ktB = kt;
            shrinkB(mtB, ktB);

            // outerRank is computed below as R - 2, so R == 2 is exactly the
            // "no batch odometer" case the one-shot route requires. int8 is
            // never one-shot: there is no int8 one-shot kernel (kKeyTileOne's
            // TransposeOneKernel Gathers the dtype directly, which is exactly
            // what int8 cannot do), so it always takes ladder A and the general
            // key 6 body -- the same choice b16/b32 make when B earns nothing.
            const bool oneShotFits = !isI8 && (R == 2) && unitsAt(mtB, ktB) <= coreCap &&
                                     mtB * ktB <= kOneTileElems;
            bool useOne = false;
            bool usePack = false;   // kKeyTilePack, decided by the refinement
            bool useWide = false;   // kKeyTileWide, decided by the arm below
            bool useVia = false;    // kKeyTileVia, family rule at kViaRouteOn
            bool useCube = false;   // kKeyTileCube, arm-gated MIX measurement
            bool useL2 = false;     // kKeyTileL2, arm-gated L2 two-pass
            uint32_t wideBuf = 2;   // key 9 slab buffering, carried in tilePrefetch
            uint32_t widePadRows = 0; // packed-load repair temporary group size
            uint32_t viaIb = 0;     // key 10 stage-2 row block, carried in rowsPerBatch
            uint64_t wideMtW = 0;   // the mt key 9 reached; the via family feature
            if (isI8) {
                mt = mtA;                      // int8: general tile route only (key 6)
                kt = ktA;
            } else if (tileBalance == 2) {
                mt = mtB;                      // Use the balanced ladder unconditionally.
                kt = ktB;
                useOne = oneShot != 0 && oneShotFits;
            } else if (tileBalance == 1 && oneShot != 0 && oneShotFits) {
                mt = mtB;                      // Use the balanced ladder only when it reaches the one-shot route.
                kt = ktB;
                useOne = true;
            } else {
                mt = mtA;                      // tileBalance 0, or B earns nothing
                kt = ktA;
            }
            // Refine DMA widths for the general tile route. A tile reads mi blocks of
            // kk*ds bytes and writes kk blocks of mi*ds bytes. The objective is:
            // readBlocks = batch * bigM * ceil(bigK/k);
            // writeBlocks = batch * bigK * ceil(bigM/m);
            // cost = max(bytes*ReadByte + readBlocks*ReadBlock,
            //            bytes*WriteByte + writeBlocks*WriteBlock).
            //
            // Tile edges are multiples of 32 bytes for UB pitch alignment. They may
            // overhang an extent because mi and kk are clipped to the remaining data.
            // Since dq >= bigM and sp >= bigK, neither descriptor gap underflows.
            // This refinement does not modify the one-shot or int8 route geometry.
            if (!isI8 && !useOne) {
                // Per-element UB cost includes two double-buffered dtype queues and the
                // uint32 offset table, matching the non-int8 tileBytes calculation.
                // Retain the UB limit even when the Gather element bound is tighter.
                const uint64_t perElem =
                    2ull * static_cast<uint64_t>(bufferNum) * dtypeSize + sizeof(uint32_t);
                uint64_t areaCap = budget / perElem;
                if (areaCap > kMaxGatherElems) {
                    areaCap = kMaxGatherElems;
                }
                const uint64_t unitsCur = unitsAt(mt, kt);
                const uint64_t occupancy = (coreCap < unitsCur) ? coreCap : unitsCur;

                auto upEb = [&](uint64_t v) {
                    return detail::CeilDiv(v, elemsPerBlock) * elemsPerBlock;
                };
                const uint64_t mSnap = upEb(bigM);   // smallest edge covering M
                const uint64_t kSnap = upEb(bigK);   // smallest edge covering K

                uint64_t bestM = mt;
                uint64_t bestK = kt;
                uint64_t bestScore = 0;
                auto score = [&](uint64_t m, uint64_t k) -> uint64_t {
                    const uint64_t tm = detail::CeilDiv(bigM, m);
                    const uint64_t tk = detail::CeilDiv(bigK, k);
                    const uint64_t rb = batchCount * bigM * tk;   // sum of mi
                    const uint64_t wb = batchCount * bigK * tm;   // sum of kk
                    const uint64_t rc = totalBytes * kPipeReadByte + rb * kPipeReadBlock;
                    const uint64_t wc = totalBytes * kPipeWriteByte + wb * kPipeWriteBlock;
                    return rc > wc ? rc : wc;
                };
                bestScore = score(mt, kt);
                auto tryPair = [&](uint64_t m, uint64_t k) {
                    if (m < elemsPerBlock || k < elemsPerBlock || m > kMaxBlockCount ||
                        k > kMaxBlockCount || m * k > areaCap) {
                        return;
                    }
                    if (batchCount * detail::CeilDiv(bigM, m) * detail::CeilDiv(bigK, k) <
                        occupancy) {
                        return;   // never hand back cores the ladder had filled
                    }
                    const uint64_t s = score(m, k);
                    if (s < bestScore) {
                        bestScore = s;
                        bestM = m;
                        bestK = k;
                    }
                };
                // Propose one edge and grow the other within the area cap and exact cover.
                // Then regrow the first edge into unused capacity. Longer edges cannot
                // increase block counts and avoid unnecessarily increasing the number of tiles.
                auto fitEdge = [&](uint64_t other, uint64_t snap) -> uint64_t {
                    uint64_t e = (other == 0) ? 0 : (areaCap / other);
                    if (e > snap) { e = snap; }
                    if (e > kMaxBlockCount) { e = kMaxBlockCount; }
                    return e / elemsPerBlock * elemsPerBlock;
                };
                auto probeM = [&](uint64_t m) {
                    if (m == 0) { return; }
                    m = upEb(m > mSnap ? mSnap : m);
                    if (m == 0 || m > kMaxBlockCount) { return; }
                    const uint64_t k = fitEdge(m, kSnap);
                    if (k == 0) { return; }
                    const uint64_t mGrown = fitEdge(k, mSnap);
                    tryPair(mGrown > m ? mGrown : m, k);
                };
                auto probeK = [&](uint64_t k) {
                    if (k == 0) { return; }
                    k = upEb(k > kSnap ? kSnap : k);
                    if (k == 0 || k > kMaxBlockCount) { return; }
                    const uint64_t m = fitEdge(k, mSnap);
                    if (m == 0) { return; }
                    const uint64_t kGrown = fitEdge(m, kSnap);
                    tryPair(m, kGrown > k ? kGrown : k);
                };
                // The ladder's own vocabulary (a data block doubled), the exact
                // cover, and the "t tiles across this extent" family, which is
                // what supplies the aspect ratios between the powers of two.
                for (uint64_t e = elemsPerBlock; e <= areaCap; e *= 2) {
                    probeM(e);
                    probeK(e);
                }
                probeM(mSnap);
                probeK(kSnap);
                for (uint64_t t = 1; t <= 16; ++t) {
                    probeM(detail::CeilDiv(bigM, t));
                    probeK(detail::CeilDiv(bigK, t));
                }
                mt = bestM;
                kt = bestK;
                if (mtOverride != 65535u && mtOverride > 0u) {
                    mt = mtOverride;
                }
                if (ktOverride != 65535u && ktOverride > 0u) {
                    kt = ktOverride;
                }

                // ---- kKeyTilePack: the store as ONE transaction -------------
                //
                // Everything above still issues one MTE3 block per output row,
                // because a DataCopyPad's UB source pitch is ceil32(blockLen).
                // Key 7 makes the Gather write the packed output run directly,
                // so the store is a single block of kk*bigM*ds bytes. It needs:
                //
                //   * tilesM == 1 -- one tile spans the whole output row, so
                //     mi == bigM for every tile and i0 is always 0. The kernel
                //     does not compute an M index at all;
                //   * bigM == dq -- the tile's kk output rows are ADJACENT in
                //     GM, which is what makes them one run. dq >= bigM always,
                //     so this is the equality case and nothing can underflow;
                //   * kt*ds a multiple of 32 -- the LOAD's UB pitch, unchanged
                //     from key 3;
                //   * kt*bigM <= kMaxGatherElems -- the Gather bound, and also
                //     the bound that gives the table's float divide a 512x
                //     margin (see the kernel's block comment);
                //   * the largest int32 intermediate in the table build fits.
                //
                // bigM is NOT rounded up to a data block here, unlike key 3:
                // there is no output UB pitch to align any more, so the tile is
                // bigM rows exactly and the UB it saves goes into kt.
                plan.packGateWord |= 1ull;                                  // b0 tile family
                plan.packGateWord |= (bigM == dq) ? 8ull : 0ull;             // b3
                plan.packGateWord |= (bigM <= kMaxBlockCount) ? 16ull : 0ull; // b4
                plan.packGateWord |= ((kt * dtypeSize) % kDataBlockBytes == 0) ? 32ull : 0ull;  // b5
                if (bigM == dq && bigM <= kMaxBlockCount) {
                    // in queue + packed out queue + table + int32 scratch.
                    const uint64_t packPerElem =
                        2ull * static_cast<uint64_t>(bufferNum) * dtypeSize +
                        2ull * sizeof(uint32_t);
                    uint64_t packArea = budget / packPerElem;
                    if (packArea > kMaxGatherElems) {
                        packArea = kMaxGatherElems;
                    }
                    uint64_t pk = packArea / bigM;                  // rows of K
                    pk = pk / elemsPerBlock * elemsPerBlock;         // UB pitch
                    if (pk > kSnap) {
                        pk = kSnap;
                    }
                    if (pk > kMaxBlockCount) {
                        pk = kMaxBlockCount / elemsPerBlock * elemsPerBlock;
                    }
                    const uint64_t nOff = pk * bigM;
                    const uint64_t pitchIn = pk * dtypeSize;
                    // The table build's largest intermediate is Arange(0, P, N)
                    // reaching (N-1)*P; it is int32, so refuse the route rather
                    // than overflow it. Also refuse a store wider than a uint32
                    // blockLen.
                    const uint64_t maxInter =
                        (nOff == 0) ? 0 : (nOff - 1) * pitchIn;
                    const uint64_t packUnits = batchCount * detail::CeilDiv(bigK, pk);
                    if (pk >= elemsPerBlock && nOff <= kMaxGatherElems &&
                        maxInter < 0x7FFFFFFFull &&
                        pk * bigM * dtypeSize <= kUint32Max &&
                        packUnits >= occupancy) {
                        // Same objective, but this route legitimately earns the
                        // one-block-per-tile credit the key-3 score must not.
                        const uint64_t rbP = batchCount * bigM * detail::CeilDiv(bigK, pk);
                        const uint64_t rcP =
                            totalBytes * kPipeReadByte + rbP * kPipeReadBlock;
                        const uint64_t wcP =
                            totalBytes * kPipeWriteByte + packUnits * kPipeWriteBlock;
                        const uint64_t sP = rcP > wcP ? rcP : wcP;
                        // Guard the entire packed-geometry assignment. When packing is disabled,
                        // mt/kt must remain legal for key 3, whose UB pitch requires 32-byte alignment.
                        if (sP < bestScore && kPackedRoutesOn) {
                            usePack = true;
                            mt = bigM;
                            kt = pk;
                            // Choose the largest adjacent-tile slab count that fits while retaining
                            // a usable per-tile area.
                            {
                                const uint64_t tryT[3] = {6ull, 3ull, 2ull};
                                for (int ti = 0; ti < 3; ++ti) {
                                    const uint64_t Ts = tryT[ti];
                                    const uint64_t per =
                                        Ts * dtypeSize + 2ull * dtypeSize + 8ull;
                                    uint64_t pkT = budget / (bigM * per);
                                    if (pkT * bigM > kMaxGatherElems) {
                                        pkT = kMaxGatherElems / bigM;
                                    }
                                    pkT = pkT / elemsPerBlock * elemsPerBlock;
                                    if (pkT > kSnap) {
                                        pkT = kSnap;
                                    }
                                    const uint64_t inter =
                                        (pkT * bigM == 0)
                                            ? 0
                                            : (pkT * bigM - 1) *
                                                  (pkT * Ts * dtypeSize);
                                    if (pkT >= 96ull && pkT >= elemsPerBlock &&
                                        inter < 0x7FFFFFFFull &&
                                        pkT * bigM * dtypeSize <= kUint32Max) {
                                        kt = pkT;
                                        plan.tilePrefetch =
                                            static_cast<uint32_t>(Ts);
                                        break;
                                    }
                                }
                            }
                            // S9_TILEKT=2..8 selects a slab of adjacent K tiles, widening each read.
                            // The input slab is single-buffered; kt must fit it plus two output buffers,
                            // the table, and scratch. S9_TILEKT=9 selects the vnchwconv pack path,
                            // which requires bigM <= 32 and uses a six-tile slab with double-buffered panels.
                            if (ktOverride == 9u && bigM <= 32ull) {
                                const uint64_t per =
                                    bigM * 6ull * dtypeSize + 2ull * 32ull * dtypeSize;
                                uint64_t pkV = budget / per;
                                pkV = pkV / elemsPerBlock * elemsPerBlock;
                                if (pkV > kSnap) {
                                    pkV = kSnap;
                                }
                                // prefer an odd 32B-block pitch (bank rule)
                                while (pkV > 8ull &&
                                       ((pkV * dtypeSize / 32ull) % 2ull) == 0ull) {
                                    pkV -= 8ull;
                                }
                                if (pkV >= elemsPerBlock &&
                                    pkV * bigM * dtypeSize <= kUint32Max) {
                                    kt = pkV;
                                    plan.tilePrefetch = 9u;
                                }
                            }
                            if (ktOverride >= 2u && ktOverride <= 8u) {
                                const uint64_t Ts = ktOverride;
                                const uint64_t per =
                                    Ts * dtypeSize + 2ull * dtypeSize + 8ull;
                                uint64_t pkT = budget / (bigM * per);
                                if (pkT * bigM > kMaxGatherElems) {
                                    pkT = kMaxGatherElems / bigM;
                                }
                                pkT = pkT / elemsPerBlock * elemsPerBlock;
                                if (pkT > kSnap) {
                                    pkT = kSnap;
                                }
                                const uint64_t inter =
                                    (pkT * bigM == 0)
                                        ? 0
                                        : (pkT * bigM - 1) * (pkT * Ts * dtypeSize);
                                if (pkT >= elemsPerBlock &&
                                    inter < 0x7FFFFFFFull &&
                                    pkT * bigM * dtypeSize <= kUint32Max) {
                                    kt = pkT;
                                    plan.tilePrefetch = static_cast<uint32_t>(Ts);
                                }
                            }
                        }
                    }
                }

                // Full-K panel route using TransDataTo5HD. Span the input row with kSnap,
                // then use the remaining UB capacity for output-panel width. This route
                // avoids the per-element Gather offset table. Check all UB and pitch bounds
                // before selecting it.
                const bool wideOn = (wideArm == 1u) || (wideArm == 0u && kWideRouteOn);
                if (wideOn && wideArm != 2u && !usePack && dtypeSize == 4) {
                    const uint64_t slabW = 2ull * 16ull * kSnap * dtypeSize;
                    uint64_t mtW = 0;
                    if ((kSnap % 8ull) == 0 && kSnap / 8ull <= 255ull &&
                        budget > slabW) {
                        mtW = (budget - slabW) / (kSnap * dtypeSize) / 16ull * 16ull;
                    }
                    if (mtOverride != 65535u && mtOverride > 0u) {
                        mtW = static_cast<uint64_t>(mtOverride) / 16ull * 16ull;
                    }
                    // measurement: S9_TILEKT=4 on the wide arm upgrades the
                    // slab mode chosen below to pair8+wave (tilePrefetch 4).
                    const uint64_t mtCap = bigM / 16ull * 16ull;
                    if (mtW > mtCap) {
                        mtW = mtCap;
                    }
                    if (mtW > 0xFFF0ull) {
                        mtW = 0xFFF0ull;   // dstRepStride is uint16
                    }
                    // Select slab buffering from the UB fit. The default uses the
                    // double-buffered width; a wider override may use one slab.
                    // If neither layout fits, retain the general tile route.
                    if (mtW >= 16) {
                        const uint64_t slab1 = 16ull * kSnap * dtypeSize;
                        const uint64_t slabH = 8ull * kSnap * dtypeSize;
                        const uint64_t slab3 = 3ull * slabH;
                        const uint64_t panelW = kSnap * mtW * dtypeSize;
                        if (slab1 * 2ull + panelW <= budget) {
                            // Default mode 5 uses full-depth double-buffered slabs with wave scheduling.
                            wideBuf = (ktOverride == 7u) ? 7 : (ktOverride == 8u) ? 8
                                      : (ktOverride == 10u) ? 10 : 5;
                            const bool repairWanted=ktOverride==11u || ktOverride==12u ||
                                (TP_DEFAULT_REPAIR && (ktOverride==0u || ktOverride==65535u));
                            const uint64_t fixed=slab1*2ull+panelW;
                            if(repairWanted && sp==bigK && fixed+4ull*kSnap*dtypeSize<=budget) {
                                wideBuf=11;
                                // Default keeps the 12KiB reserve. Arm12 tests a
                                // four-row group only when an 8KiB reserve fits.
                                widePadRows=(fixed+8ull*kSnap*dtypeSize<=budget ||
                                             (ktOverride==12u && fixed+8ull*kSnap*dtypeSize<=usable))?4:2;
                            }
                        } else if (ktOverride == 6u && slab3 + panelW <= budget) {
                            wideBuf = 6;
                        } else if (slabH * 2ull + panelW <= budget) {
                            // PAIRED-8: two db'd half slabs feed one
                            // vnchwconv -- loads stay overlapped at panel
                            // widths the 16-row pair cannot afford (mt 64).
                            wideBuf = (ktOverride == 4u) ? 4
                                      : (ktOverride == 6u) ? 6
                                                           : 3;
                        } else if (slab1 + panelW <= budget) {
                            wideBuf = 1;
                        } else {
                            mtW = 0;
                        }
                    }
                    // The automatic panel-width floor is 32; explicit route selection can use 16.
                    const uint64_t mtFloor = (wideArm == 1u) ? 16ull : 32ull;
                    if (mtW >= mtFloor && bigK <= kMaxBlockCount && dq >= mtW &&
                        (sp - bigK) * dtypeSize <= kUint32Max &&
                        (dq - mtW) * dtypeSize <= kUint32Max &&
                        batchCount * detail::CeilDiv(bigM, mtW) >= occupancy) {
                        useWide = true;
                        mt = mtW;
                        kt = kSnap;
                        wideMtW = mtW;
                    }
                }

                // Two-stage route through workspace. Stage 1 uses four 16 x kSnap buffers;
                // stage 2 adds two Ib x J row buffers. J defaults to 192 output columns,
                // subject to extent and UB limits. S9_TILEMT overrides J; S9_TILEKT overrides Ib.
                // Selection is controlled by the route switch and the preceding wide-panel
                // width. If its preconditions fail, retain the wide-panel route.
                const bool viaOn =
                    (viaArm == 1u) ||
                    (viaArm == 0u && wideArm == 0u && kViaRouteOn && useWide &&
                     wideMtW <= 64);
                if (viaOn && viaArm != 2u && dtypeSize == 4 && (kSnap % 8ull) == 0 &&
                    kSnap / 8ull <= 255ull && bigK >= 24 && bigM >= 32) {
                    // The default Ib is 48. The single-buffered panel leaves UB capacity
                    // for this assembled-row depth.
                    uint64_t ib = 48;
                    if (ktOverride != 65535u && ktOverride > 0u) {
                        ib = static_cast<uint64_t>(ktOverride) / 8ull * 8ull;
                    }
                    if (ib < 8) {
                        ib = 8;
                    }
                    if (ib > 64) {
                        ib = 64;
                    }
                    if (ib > bigK) {
                        ib = bigK / 8ull * 8ull;
                    }
                    // Two double-buffered slabs plus ONE panel (see the
                    // kernel's InitBuffer comment).
                    const uint64_t stage1Ub = 3ull * 16ull * kSnap * dtypeSize;
                    uint64_t jv = 0;
                    if (ib >= 8 && budget > stage1Ub) {
                        jv = (budget - stage1Ub) / (2ull * ib * dtypeSize);
                        jv = jv / 16ull * 16ull;
                        if (jv > 192) {
                            jv = 192;
                        }
                    }
                    if (mtOverride != 65535u && mtOverride > 0u) {
                        jv = static_cast<uint64_t>(mtOverride) / 16ull * 16ull;
                        if (stage1Ub + 2ull * ib * jv * dtypeSize > budget) {
                            jv = 0;   // an override that does not fit falls closed
                        }
                    }
                    const uint64_t jvCap = bigM / 16ull * 16ull;
                    if (jv > jvCap) {
                        jv = jvCap;
                    }
                    if (jv >= 32 && ib >= 8 && dq >= jv &&
                        (sp - bigK) * dtypeSize <= kUint32Max &&
                        batchCount * detail::CeilDiv(bigM, jv) >= occupancy) {
                        useVia = true;
                        useWide = false;
                        usePack = false;
                        mt = jv;
                        kt = kSnap;
                        viaIb = static_cast<uint32_t>(ib);
                    }
                }

                // ---- kKeyTileCube: the MIX route, arm-gated ----------------
                // The kernel addresses batches FLAT (b * bigM * sp in, b *
                // bigK * dq out), so the batch odometer must be exactly that
                // product -- rank 0/1 with the trailing-extent strides, which
                // the (0,2,1) collapsed family satisfies. Every uint16 the
                // nd2nz descriptor and the L1-chunk store consume is bounded
                // here; the L1 chunk must fit the cube's 1 MB with margin.
                if (cubeArm == 1u && dtypeSize == 4 && sp <= 0xFFFFull &&
                    bigK <= 0xFFFFull && dq == bigM && sp == bigK) {
                    const uint64_t lanes = 8;
                    const uint64_t colsPad = detail::CeilDiv(bigK, lanes) * lanes;
                    const uint64_t rowB = colsPad * dtypeSize;
                    // Choose the largest 16-row multiple fitting the requested slot layout.
                    // S9_TILEMT overrides Nc and S9_TILEKT overrides slot count. The default
                    // uses one slot; an explicit value of 2 requests dual slots.
                    uint64_t slots = 1;
                    uint64_t nc = kCubeL1Budget / rowB / 16ull * 16ull;
                    if (nc > kCubeNcMax) {
                        nc = kCubeNcMax;
                    }
                    if (mtOverride != 65535u && mtOverride > 0u) {
                        nc = static_cast<uint64_t>(mtOverride) / 16ull * 16ull;
                    }
                    if (ktOverride != 65535u && ktOverride > 0u && ktOverride <= 2u) {
                        slots = ktOverride;
                    }
                    const uint64_t chunkBytes = nc * rowB;
                    const uint64_t groups = (coreNum == 0 ? 1 : coreNum) / 2;
                    if (nc >= 16 && bigM >= nc &&
                        slots * chunkBytes <= kCubeL1Budget &&
                        chunkBytes / kDataBlockBytes <= 0xFFFFull && groups >= 1 &&
                        batchCount >= 1) {
                        useVia = false;
                        useWide = false;
                        usePack = false;
                        useOne = false;
                        mt = nc;
                        kt = colsPad;
                        viaIb = static_cast<uint32_t>(slots);
                        useCube = true;
                    }
                }
            }

                // ---- kKeyTileL2: the L2-resident two-pass, arm-gated ------
                // Preconditions, each tied to a kernel assumption:
                //   dq == bigM         pass B writes CONTIGUOUS 16-row output
                //                      groups (row pitch must equal bigM);
                //   sp == bigK         pass A slab loads are gap-0 full rows;
                //   kSnap/8 <= 255     one vnchwconv per sub-slab (repeat cap);
                //   bigM >= 96         at least two panels (and the tail
                //                      clamp j0 = bigM-48 stays >= 0);
                //   bigK >= 16         one full pass-B group exists (tail
                //                      groups left-shift, idempotent);
                //   scratch <= cap     cores * tilesP * kSnap * 48 * 4 rides
                //                      inside half the L2, or pass B's reads
                //                      fall back to HBM and the route loses
                //                      its whole reason to exist.
                const uint64_t l2Snap = detail::AlignUp(bigK, 8ull);
                if (l2Arm == 1u && dtypeSize == 4 && dq == bigM && sp == bigK &&
                    l2Snap / 8ull <= 255ull && bigM >= 2ull * kL2PanelW &&
                    bigK >= 8ull) {
                    const uint64_t imageB = bigK * bigM * dtypeSize;
                    // Balanced batch ownership: the largest core count that
                    // divides batchCount evenly, floored at half the cores;
                    // otherwise round-robin with a one-batch ragged tail.
                    uint64_t cores = coreNum == 0 ? 1 : coreNum;
                    if (cores > batchCount) {
                        cores = batchCount;
                    }
                    // Two images per core must ride inside the L2 cap; shed
                    // cores first (fabric saturates by ~16 anyway), then
                    // look for a balanced count under the clamp.
                    const uint64_t maxByL2 =
                        kL2ScratchCap / (2ull * bigK * bigM * dtypeSize + 1ull);
                    if (maxByL2 >= 1 && cores > maxByL2) {
                        cores = maxByL2;
                    }
                    uint64_t bal = 0;
                    for (uint64_t c = cores; c * 2 >= cores && c >= 1; --c) {
                        if (batchCount % c == 0) {
                            bal = c;
                            break;
                        }
                    }
                    if (bal != 0) {
                        cores = bal;
                    }
                    // UB arena: pass A (two paired-8 slabs + the mt=48
                    // panel) CO-RESIDES with pass B's single chunk buffer of
                    // at least four output rows -- the pipelined kernel
                    // needs both phases live at once.
                    const uint64_t slabB = 8ull * l2Snap * dtypeSize;
                    const uint64_t aB = 2ull * slabB + l2Snap * kL2PanelW * dtypeSize;
                    const uint64_t l2Budget = usable > 4096 ? usable - 4096 : usable;
                    if (cores >= 1 && aB + 4ull * bigM * dtypeSize <= l2Budget &&
                        cores * 2ull * imageB <= kL2ScratchCap &&
                        bigM * bigK * dtypeSize <= kUint32Max) {
                        useL2 = true;
                        useVia = false;
                        useWide = false;
                        usePack = false;
                        useOne = false;
                        useCube = false;
                        mt = kL2PanelW;
                        kt = l2Snap;
                        viaIb = static_cast<uint32_t>(cores);  // key 12: cores
                    }
                }

            const uint64_t tilesM = detail::CeilDiv(bigM, mt);
            const uint64_t tilesK = detail::CeilDiv(bigK, kt);

            plan.tilingKey = isI8 ? kKeyTileI8
                             : (useL2 ? kKeyTileL2
                             : (useCube ? kKeyTileCube
                                : (useVia ? kKeyTileVia
                                          : (useWide ? kKeyTileWide
                                                     : (usePack ? kKeyTilePack
                                                                : kKeyTile)))));
            if (useVia || useCube || useL2) {
                plan.rowsPerBatch = viaIb;   // key 10: Ib; 11: slots; 12: cores
            }
            plan.rowLen = bigM;
            plan.tileK = bigK;
            plan.tileMT = static_cast<uint32_t>(mt);
            plan.tileKT = static_cast<uint32_t>(kt);
            plan.srcBlkPitch = sp;
            plan.dstRowStride = dq;
            plan.rowCount = batchCount;
            // Key 7's UB map is a different shape -- the out queue is the PACKED
            // run, and the table build needs an int32 scratch of the same length
            // -- so it is sized here rather than by tileBytes().
            plan.ubBufBytes = static_cast<uint32_t>(
                useL2 ? (2ull * 8ull * kt + kt * mt + 4ull * bigM) * dtypeSize
                : useCube ? 4ull * 4ull * mt * 8ull * dtypeSize
                : useVia  ? (3ull * 16ull * kt + 2ull * viaIb * mt) * dtypeSize
                : useWide ? (static_cast<uint64_t>(wideBuf) * 16ull * kt + kt * mt) * dtypeSize
                : usePack ? (mt * kt) *
                        ((plan.tilePrefetch > 1u
                              ? static_cast<uint64_t>(plan.tilePrefetch)
                              : 2ull * bufferNum) *
                             dtypeSize +
                         2ull * dtypeSize + 2ull * sizeof(uint32_t))
                          : tileBytes(mt, kt));
            if (useWide) {
                // Field reuse, documented at the kernel: key 9 does not
                // prefetch (its pipeline is the two queues), so tilePrefetch
                // carries the slab buffer count instead -- 2 = double
                // buffered loads, 1 = the single-slab arm that trades load
                // overlap for panel width.
                plan.tilePrefetch = wideBuf;
                if(wideBuf==11) {
                    plan.rowsPerBatch=widePadRows;
                    plan.ubBufBytes=static_cast<uint32_t>((32ull*kt+kt*mt+2ull*widePadRows*kt)*dtypeSize);
                }
            }

            // Batch odometer: output axes 0..R-2 except q.
            uint32_t j = 0;
            for (int i = 0; i < R - 1; ++i) {
                if (i == q) {
                    continue;
                }
                plan.outerShape[j] = static_cast<uint64_t>(coutShape[i]);
                plan.outerInStride[j] = static_cast<uint64_t>(cinStride[cperm[i]]);
                plan.outerOutStride[j] = static_cast<uint64_t>(coutStride[i]);
                ++j;
            }
            plan.outerRank = j;

            const uint64_t units = batchCount * tilesM * tilesK;
            plan.blockDim = PickBlockDim(units, totalBytes, coreNum, false);
            if (useL2) {
                // A core owns whole batches; use the balanced core count derived above.
                plan.blockDim = static_cast<uint32_t>(viaIb);
            }
            if (useCube) {
                // MIX 1:2: blockDim counts AIC GROUPS; the runtime launches
                // one cube core plus two vector cores per group.
                uint32_t groups = (coreNum == 0 ? 1u : coreNum) / 2u;
                if (groups == 0) {
                    groups = 1;
                }
                plan.blockDim = groups;
            }

            // Recheck the one-shot kernel preconditions before launch:
            // units == blockDim gives each core exactly one tile;
            // outerRank == 0 removes the batch odometer;
            // mt * kt <= kOneTileElems fits the compile-time UB layout.
            // Otherwise keep the general tile route.
            if (useOne && plan.outerRank == 0 &&
                units == static_cast<uint64_t>(plan.blockDim) && mt * kt <= kOneTileElems) {
                plan.tilingKey = kKeyTileOne;
            }
            // Recheck the packed-tile kernel preconditions. tilesM == 1 gives
            // mi == bigM and i0 == 0; rowLen == dstRowStride makes the output rows
            // adjacent. The element limit bounds Gather and table size; the pitch rule
            // applies to input.
            // Reject an invalid packed plan instead of falling back to key 3: packed
            // geometry need not have a 32-byte-aligned M edge, which key 3 requires.
            if (plan.tilingKey == kKeyTilePack &&
                !(tilesM == 1 && plan.rowLen == plan.dstRowStride &&
                  mt * kt <= kMaxGatherElems && (kt * dtypeSize) % kDataBlockBytes == 0 &&
                  mt <= kMaxBlockCount && kt <= kMaxBlockCount)) {
                return false;
            }
            // The same discipline for the wide-panel route -- these are
            // TransposeTileWideKernel's actual preconditions (b32 only; a
            // 16-row-slab / 8-rows-per-repeat vnchwconv with repeat <= 255;
            // uniform strips via the left-shifted tail, so bigM >= mt with mt
            // a multiple of 16; a store of bigK blocks; tileKT is the padded
            // slab pitch covering bigK). All guaranteed by the arm block
            // above; if they ever come apart, refusing to launch is the only
            // safe answer -- the panel pitch key 3 would inherit is not a
            // geometry its kernel can run.
            if (plan.tilingKey == kKeyTileWide &&
                !(dtypeSize == 4 && mt >= 16 && mt % 16 == 0 && plan.rowLen >= mt &&
                  kt % 8 == 0 && kt / 8 <= 255 && kt >= plan.tileK &&
                  plan.tileK <= kMaxBlockCount && mt <= 0xFFF0ull &&
                  plan.tilePrefetch >= 1 && plan.tilePrefetch <= 11)) {
                return false;
            }
            // TransposeTileViaKernel's actual preconditions, re-asserted at the
            // point of use like the two routes above; refusing to launch is
            // the only safe answer if they ever come apart from the arm block.
            if (plan.tilingKey == kKeyTileVia &&
                !(dtypeSize == 4 && mt >= 32 && mt % 16 == 0 && kt % 8 == 0 &&
                  kt / 8 <= 255 && kt >= plan.tileK && plan.rowsPerBatch >= 8 &&
                  plan.rowsPerBatch % 8 == 0 && plan.rowsPerBatch <= plan.tileK &&
                  plan.rowLen >= 16)) {
                return false;
            }
            // TransposeTileCubeKernel's actual preconditions: flat batch
            // addressing (dq == bigM, sp == bigK), every uint16 the nd2nz
            // and L1-store descriptors take, and the uniform chunk fitting L1.
            if (plan.tilingKey == kKeyTileCube &&
                !(dtypeSize == 4 && mt >= 16 && mt % 16 == 0 && plan.rowLen >= mt &&
                  plan.srcBlkPitch == plan.tileK && plan.dstRowStride == plan.rowLen &&
                  plan.srcBlkPitch <= 0xFFFFull && plan.tileK <= 0xFFFFull &&
                  kt % 8 == 0 && mt / 16 <= 255 &&
                  plan.rowsPerBatch >= 1 && plan.rowsPerBatch <= 2 &&
                  plan.rowsPerBatch * mt * kt * dtypeSize <= kCubeL1Budget)) {
                return false;
            }
            plan.valid = true;
            return true;
        }
    }

    // ---------------------------------------------------------------------
    // Row engine.
    // ---------------------------------------------------------------------
    plan.tilingKey = kKeyRows;
    uint32_t blen;
    uint32_t nblk;
    if (p == R - 1) {
        blen = static_cast<uint32_t>(plan.rowLen);
        nblk = 1;
    } else {
        blen = 1;
        if (plan.rowLen > kUint32Max) {
            return false;
        }
        nblk = static_cast<uint32_t>(plan.rowLen);
    }
    plan.blkLenElems = blen;
    plan.blkPerRow = nblk;

    const uint64_t blockBytes = static_cast<uint64_t>(blen) * dtypeSize;
    const uint64_t slot = detail::AlignUp(blockBytes, kDataBlockBytes);
    if (slot == 0 || slot > kUint32Max) {
        return false;
    }
    plan.slotBytes = static_cast<uint32_t>(slot);

    // A core on this route earns its launch either by moving many DMA blocks
    // or by moving many bytes; whichever justifies more cores wins. The byte
    // arm keeps wide contiguous segments parallel, the block arm keeps narrow
    // element-wise transposes from being spread over cores that would each
    // get a handful of blocks.
    const uint64_t totalBlocks = plan.rowCount * nblk;
    const uint64_t byBlocks = detail::CeilDiv(totalBlocks, kMinBlocksPerCore);
    const uint64_t byBytes = detail::CeilDiv(totalBytes, kMinBytesPerCore);
    uint64_t allowed = byBlocks > byBytes ? byBlocks : byBytes;
    if (allowed == 0) {
        allowed = 1;
    }
    uint32_t cap = coreNum;
    if (allowed < static_cast<uint64_t>(cap)) {
        cap = static_cast<uint32_t>(allowed);
    }

    // ---------------------------------------------------------------------
    // Fold route (kKeyFold). Only the short-contiguous-suffix sub-shape
    // (nblk == 1) qualifies: there a row is one contiguous run and the route
    // above spends one DataCopyPad, and one DataCopyExtParams construction,
    // per row. Consecutive rows along any single odometer axis differ by a
    // constant source stride, which is exactly what a strided DataCopyPad
    // describes, so `foldLen` rows collapse into ONE descriptor. Under the
    // per-core cost model (fixed prologue + k * iteration count) that is a
    // direct division of the variable term.
    //
    // Which axis to fold along is a real choice, not a formality:
    //   * the innermost odometer axis leaves the destination contiguous, so
    //     many folded runs still share a single store, but the run is only as
    //     long as that axis;
    //   * an outer axis gives far longer runs at the price of a strided store,
    //     i.e. one store per run instead of one per batch.
    // Score both by the number of descriptors a core ends up issuing and take
    // the cheaper. Everything below is a function of extents, strides, UB size
    // and core count only.
    // ---------------------------------------------------------------------
    if (foldMode != 0 && nblk == 1 && plan.outerRank > 0 && plan.rowLen > 0) {
        uint64_t rowStride[kPlanMaxOdo] = {};  // output row-index stride per axis
        {
            uint64_t acc = 1;
            for (int a = static_cast<int>(plan.outerRank) - 1; a >= 0; --a) {
                rowStride[a] = acc;
                acc *= plan.outerShape[a];
            }
        }
        uint64_t rowsInUb = bufBytes / slot;
        if (rowsInUb > kMaxBlockCount) {
            rowsInUb = kMaxBlockCount;
        }
        // A fold unit covers many rows with one or two descriptors. foldCoreCap
        // controls whether it inherits the row-engine minimum-work limit.
        uint32_t foldCores = (foldCoreCap != 0) ? cap : coreNum;
        if (foldCores == 0) {
            foldCores = 1;
        }
        const uint64_t coresWanted = foldCores;
        // Cost of the unfolded route: one load descriptor per row.
        const uint64_t baseScore = detail::CeilDiv(plan.rowCount, coresWanted);

        bool found = false;
        uint32_t bestAxis = 0;
        uint64_t bestLen = 0;
        uint64_t bestUnits = 0;
        uint64_t bestScore = 0;
        // The best PACK-ELIGIBLE candidate, tracked alongside the key-4 winner.
        // The substitution block after the loop says why it cannot simply be
        // folded into `score`.
        bool packFound = false;
        uint32_t packAxis = 0;
        uint64_t packLen = 0;
        uint64_t packUnits = 0;
        uint64_t packScore = 0;
        uint64_t packRowsBest = 0;
        // Longest fold run whose whole batch still fits ONE Gather and the
        // packed UB map -- a function of rowLen, the UB size and bufferNum, so
        // a feature of the case rather than a shape constant. Mirrors the
        // `perRow` / `byGather` pair the kKeyFoldPack block below derives.
        uint64_t packLenCap = 0;
        {
            const uint64_t packPerRow =
                static_cast<uint64_t>(bufferNum) * (slot + rowBytes) +
                2ull * sizeof(uint32_t) * plan.rowLen;
            const uint64_t byGather =
                (plan.rowLen == 0) ? 0 : (kMaxGatherElems / plan.rowLen);
            const uint64_t byUb = (packPerRow == 0) ? 0 : (usable / packPerRow);
            packLenCap = (byGather < byUb) ? byGather : byUb;
        }
        const uint32_t axisLo = (foldMode == 1) ? plan.outerRank - 1 : 0;
        for (uint32_t a = axisLo; a < plan.outerRank; ++a) {
            const uint64_t ext = plan.outerShape[a];
            const uint64_t sstr = plan.outerInStride[a];
            if (ext < 2 || sstr < plan.rowLen) {
                continue;
            }
            if ((sstr - plan.rowLen) * dtypeSize > kUint32Max) {
                continue;  // source gap not expressible in the uint32 field
            }
            const uint64_t dstRows = rowStride[a];
            if ((dstRows - 1) * rowBytes > kUint32Max) {
                continue;  // destination gap not expressible
            }
            const uint64_t groups = plan.rowCount / ext;
            if (groups == 0) {
                continue;
            }
            // Minimise waves first: pick the fewest chunks of this axis that
            // still hands every core a unit, then make each chunk as long as
            // that allows.
            uint64_t chunks = (groups >= coresWanted) ? 1 : (coresWanted / groups);
            if (chunks > ext) {
                chunks = ext;
            }
            if (chunks == 0) {
                chunks = 1;
            }
            uint64_t len = detail::CeilDiv(ext, chunks);
            if (len > rowsInUb) {
                len = rowsInUb;
            }
            if (len < 2) {
                continue;  // a fold of one row is the unfolded route
            }
            const uint64_t units = groups * detail::CeilDiv(ext, len);
            const uint64_t perCore = detail::CeilDiv(units, coresWanted);
            // A contiguous destination (dstRows == 1) lets one store serve
            // several units; a strided one needs its own store per unit.
            const uint64_t score = (dstRows == 1) ? perCore : 2 * perCore;
            if (!found || score < bestScore) {
                found = true;
                bestScore = score;
                bestAxis = a;
                bestLen = len;
                bestUnits = units;
            }
            // ---- the same axis, sized so kKeyFoldPack could take it --------
            // Only the contiguous-destination axis can pack, and the run has
            // to be short enough that a whole batch is one Gather. Shortening
            // the run adds descriptors and CANNOT add a single DMA block --
            // the fold's read is `rowCount` blocks of `rowBytes` whatever axis
            // and length are chosen -- so this candidate is never worse on the
            // pipes and is often reachable where the key-4 winner is not.
            if (dstRows == 1 && packLenCap >= 2) {
                uint64_t pLen = (len > packLenCap) ? packLenCap : len;
                if (pLen >= 2) {
                    const uint64_t pUnits = groups * detail::CeilDiv(ext, pLen);
                    const uint64_t pScore = detail::CeilDiv(pUnits, coresWanted);
                    // Mirrors the batchRows / cap arithmetic in the
                    // kKeyFoldPack block, so `packRowsBest` is the number that
                    // block will actually derive for this candidate.
                    uint64_t pBatch = rowsInUb / pLen;
                    if (pBatch > pScore) {
                        pBatch = pScore;
                    }
                    if (pBatch == 0) {
                        pBatch = 1;
                    }
                    uint64_t pCap = pBatch * pLen;
                    if (pCap > packLenCap) {
                        pCap = packLenCap;
                    }
                    const uint64_t pRows = pCap / pLen * pLen;
                    if (pRows >= pLen && (!packFound || pScore < packScore)) {
                        packFound = true;
                        packScore = pScore;
                        packAxis = a;
                        packLen = pLen;
                        packUnits = pUnits;
                        packRowsBest = pRows;
                    }
                }
            }
        }

        // Compare the best ordinary fold against a pack-eligible fold.
        // Axis choice does not change ordinary fold block count, but it changes
        // output adjacency and the packed-run size. Include descriptors and DMA
        // blocks in the comparison. Require an existing fold candidate and retain
        // the fold eligibility threshold before substituting the packed candidate.
        if (kFoldPackAware && found && packFound && packRowsBest > 0 &&
            packScore < baseScore) {
            const uint64_t packBatches = detail::CeilDiv(plan.rowCount, packRowsBest);
            const uint64_t rd =
                totalBytes * kPipeReadByte + plan.rowCount * kPipeReadBlock;
            const uint64_t w4 =
                totalBytes * kPipeWriteByte + plan.rowCount * kPipeWriteBlock;
            const uint64_t w8 =
                totalBytes * kPipeWriteByte + packBatches * kPipeWriteBlock;
            const uint64_t rd4 = rd + bestUnits * kPipeReadBlock;
            const uint64_t rd8 = rd + packUnits * kPipeReadBlock;
            const uint64_t c4 = rd4 > w4 ? rd4 : w4;
            const uint64_t c8 = rd8 > w8 ? rd8 : w8;
            if (c8 < c4) {
                bestAxis = packAxis;
                bestLen = packLen;
                bestUnits = packUnits;
                bestScore = packScore;
            }
        }

        if (found && bestScore < baseScore) {
            const uint64_t dstRows = rowStride[bestAxis];
            plan.tilingKey = kKeyFold;
            plan.foldLen = static_cast<uint32_t>(bestLen);
            plan.foldExt = plan.outerShape[bestAxis];
            plan.foldSrcStride = plan.outerInStride[bestAxis];
            plan.foldDstRowStride = dstRows;
            plan.chunkBlks = 1;

            uint64_t batchRows = bestLen;
            if (dstRows == 1) {
                // Units are consecutive in the output, so stage as many as UB
                // and the blockCount field allow behind one store.
                uint64_t perBatch = rowsInUb / bestLen;
                const uint64_t perCore = detail::CeilDiv(bestUnits, coresWanted);
                if (perBatch > perCore) {
                    perBatch = perCore;
                }
                if (perBatch == 0) {
                    perBatch = 1;
                }
                batchRows = perBatch * bestLen;
            }
            plan.rowsPerBatch = static_cast<uint32_t>(batchRows);

            // The ordinary fold store uses one block per padded UB row. Gather the
            // rows into a packed buffer so each batch can leave in one transaction.
            // Only foldDstRowStride == 1 qualifies because destination rows must be
            // adjacent. The batch is bounded by Gather size and by UB capacity for
            // staging, packed output, and the offset table.
            plan.packGateWord |= 2ull;                                  // b1 fold family
            plan.packGateWord |= (dstRows == 1) ? 64ull : 0ull;          // b6
            if (dstRows == 1 && plan.rowLen > 0 && bestLen > 0) {
                const uint64_t rowBytesU = plan.rowLen * dtypeSize;
                // bufferNum copies each of the padded staging and the packed
                // buffer, plus the uint32 table and the int32 scratch its float
                // build needs (rowLen entries of each per row).
                const uint64_t perRow =
                    static_cast<uint64_t>(bufferNum) * (slot + rowBytesU) +
                    2ull * sizeof(uint32_t) * plan.rowLen;
                uint64_t cap = (perRow == 0) ? 0 : (usable / perRow);
                const uint64_t byGather = kMaxGatherElems / plan.rowLen;
                if (cap > byGather) {
                    cap = byGather;
                }
                if (cap > batchRows) {
                    cap = batchRows;
                }
                // Whole fold runs only. The kernel stages the FIRST run
                // unconditionally to guarantee forward progress, so a batch
                // smaller than one run would overrun both buffers; refusing the
                // route is the fail-closed answer.
                const uint64_t packRows = cap / bestLen * bestLen;
                if (packRows >= bestLen) {
                    const uint64_t batches = detail::CeilDiv(plan.rowCount, packRows);
                    const uint64_t rd =
                        totalBytes * kPipeReadByte + plan.rowCount * kPipeReadBlock;
                    const uint64_t w4 =
                        totalBytes * kPipeWriteByte + plan.rowCount * kPipeWriteBlock;
                    const uint64_t w8 =
                        totalBytes * kPipeWriteByte + batches * kPipeWriteBlock;
                    const uint64_t c4 = rd > w4 ? rd : w4;
                    const uint64_t c8 = rd > w8 ? rd : w8;
                    // Same discipline as the tile block: rowsPerBatch and
                    // ubBufBytes are the PACK's sizing; leaking them into a
                    // key-4 plan undersizes the staging key 4 derives its own
                    // way.
                    if (c8 < c4 && kPackedRoutesOn) {
                        plan.tilingKey = kKeyFoldPack;
                        plan.rowsPerBatch = static_cast<uint32_t>(packRows);
                        // Size of the staging buffer; packed output, table, and scratch have
                        // separate bounds in the route layout.
                        plan.ubBufBytes = static_cast<uint32_t>(packRows * slot);
                    }
                }
            }

            // The odometer now covers every axis except the folded one, and
            // has to carry output row strides so the kernel can place a run
            // whose destination is no longer simply `row * rowLen`.
            uint64_t keepShape[kPlanMaxOdo] = {};
            uint64_t keepIn[kPlanMaxOdo] = {};
            uint64_t keepOut[kPlanMaxOdo] = {};
            uint32_t kept = 0;
            for (uint32_t a = 0; a < plan.outerRank; ++a) {
                if (a == bestAxis) {
                    continue;
                }
                keepShape[kept] = plan.outerShape[a];
                keepIn[kept] = plan.outerInStride[a];
                keepOut[kept] = rowStride[a];
                ++kept;
            }
            for (uint32_t a = 0; a < kPlanMaxOdo; ++a) {
                plan.outerShape[a] = keepShape[a];
                plan.outerInStride[a] = keepIn[a];
                plan.outerOutStride[a] = keepOut[a];
            }
            plan.outerRank = kept;
            plan.blockDim = PickBlockDim(bestUnits, totalBytes, foldCores, false);
            // Recheck the packed-fold kernel preconditions at the point of use.
            // Reject an invalid plan rather than reusing it with the parent route:
            // the packed route has a distinct UB map and must hold a complete fold run.
            if (plan.tilingKey == kKeyFoldPack &&
                !(plan.foldDstRowStride == 1 && plan.foldLen > 0 &&
                  plan.rowsPerBatch >= plan.foldLen &&
                  plan.rowsPerBatch % plan.foldLen == 0 &&
                  static_cast<uint64_t>(plan.rowsPerBatch) * plan.rowLen <=
                      kMaxGatherElems)) {
                return false;
            }
            plan.valid = true;
            return true;
        }
    }

    // How many blocks of a row UB can hold at once.
    uint64_t maxBlks = bufBytes / slot;
    if (maxBlks == 0) {
        maxBlks = 1;
    }
    if (maxBlks > kMaxBlockCount) {
        maxBlks = kMaxBlockCount;
    }

    // When there are fewer rows than cores, the only way to occupy the machine
    // is to cut a row into chunks and let cores claim (row, chunk) pairs. This
    // is the tall-and-thin transpose: few output rows, each very long.
    uint64_t wantChunksPerRow = detail::CeilDiv(coreNum, plan.rowCount);
    if (wantChunksPerRow == 0) {
        wantChunksPerRow = 1;
    }
    uint64_t splitBlks = detail::CeilDiv(nblk, wantChunksPerRow);
    if (splitBlks == 0) {
        splitBlks = 1;
    }

    uint64_t blks = maxBlks < static_cast<uint64_t>(nblk) ? maxBlks : nblk;
    if (splitBlks < blks) {
        blks = splitBlks;
    }
    plan.chunkBlks = static_cast<uint32_t>(blks);

    if (plan.chunkBlks >= nblk) {
        // A whole row fits and is not being split: batch rows into one store.
        uint64_t rows = bufBytes / (static_cast<uint64_t>(nblk) * slot);
        if (rows == 0) {
            rows = 1;
        }
        if (rows > plan.rowCount) {
            rows = plan.rowCount;
        }
        if (rows * nblk > kMaxBlockCount) {
            rows = kMaxBlockCount / nblk;
        }
        if (rows == 0) {
            rows = 1;
        }
        plan.rowsPerBatch = static_cast<uint32_t>(rows);
    } else {
        plan.rowsPerBatch = 1;
    }

    // DataCopyExtParams.srcStride is a byte gap in a uint32 field. If the
    // source stride is too wide to express, fall back to one block per DMA,
    // where the field is unused.
    const uint64_t gapBytes = (plan.srcBlkPitch > blen)
        ? (plan.srcBlkPitch - blen) * dtypeSize : 0;
    if (nblk > 1 && gapBytes > kUint32Max) {
        plan.rowsPerBatch = 1;
        plan.chunkBlks = 1;
    }

    const uint64_t chunksPerRow = detail::CeilDiv(nblk, plan.chunkBlks);
    const uint64_t work = (chunksPerRow <= 1) ? plan.rowCount
                                             : plan.rowCount * chunksPerRow;

    plan.blockDim = PickBlockDim(work, totalBytes, cap, false);
    plan.valid = true;
    return true;
}


// Independent output-panel route. A core owns whole contiguous batches.
// The panel holds every M row and a UB-sized strip of K columns.
inline bool BuildOutputPlan(const int64_t* shape,int rank,const int64_t* perm,int permRank,
                            uint32_t ds,uint64_t ub,uint32_t coreNum,uint32_t mode,
                            uint32_t requestedKt,TransposePlan& p) {
    Geometry g;
    if(!BuildGeometry(shape,rank,perm,permRank,g) || ds!=4 || g.numel<=0 || mode==0)return false;
    const int r=g.collapsedRank;
    if(r<2 || r>3 || g.collapsedPerm[r-2]!=r-1 || g.collapsedPerm[r-1]!=r-2)return false;
    for(int a=0;a<r-2;++a)if(g.collapsedPerm[a]!=a)return false;
    const uint64_t m=g.collapsedShape[r-2], k=g.collapsedShape[r-1];
    const uint64_t b=r==3 ? g.collapsedShape[0] : 1;
    const uint64_t mp=(m+15)/16*16;
    const uint32_t inputs=mode>=3?2:1;
    const uint64_t budget=ub>12288?ub-12288:0;
    if(m<8 || m>4095 || k<8 || k>0x3fffffff || mp>65535 || !coreNum)return false;
    uint64_t kt=(budget/((inputs+1)*mp*ds))/8*8;
    if(kt>k/8*8)kt=k/8*8;
    if(kt>2040)kt=2040;
    if(requestedKt>0 && requestedKt!=65535) {
        if(requestedKt%8 || requestedKt>kt)return false;
        kt=requestedKt;
    }
    if(kt<8)return false;
    p=TransposePlan{};
    p.valid=true;p.tilingKey=kKeyOutput;p.dtypeSize=ds;p.numel=g.numel;
    p.rowCount=b;p.rowLen=m;p.tileK=k;p.tileMT=mp;p.tileKT=kt;
    p.srcBlkPitch=k;p.dstRowStride=m;p.tilePrefetch=mode;
    p.blockDim=static_cast<uint32_t>(b<coreNum?b:coreNum);
    p.ubBufBytes=static_cast<uint32_t>((inputs+1)*mp*kt*ds);
    return true;
}

}  // namespace transpose
}  // namespace s9

#endif  // ASCEND_S9_TRANSPOSE_PLAN_H
