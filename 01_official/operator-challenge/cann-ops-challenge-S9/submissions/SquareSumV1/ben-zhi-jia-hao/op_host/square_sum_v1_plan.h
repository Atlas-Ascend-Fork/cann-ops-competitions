/*
 * SquareSumV1 planner in plain C++, without GE/gert dependencies.
 *
 * Canonical layout: remove unit dimensions, then merge adjacent dimensions
 * that are both kept or both reduced. The remaining groups alternate between
 * kept and reduced, and the innermost group is a contiguous memory run.
 */
#ifndef S9_SQUARE_SUM_V1_PLAN_H
#define S9_SQUARE_SUM_V1_PLAN_H

#include "s9_variant.h"

#include <cstdint>

namespace squaresum {

constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t MAX_GROUP = 4;

// One key per ROUTE, not one key per family.  The toolchain emits one compiled
// kernel variant per TILING_KEY_IS(N) literal, so a key is the only thing that
// separates code: a runtime branch inside one key merges both bodies into one
// variant and every core fetches both.  The generic family used to be one key
// with a three-way runtime branch; the path to the first descriptor then walked
// 35 instruction-cache lines against the row route's 12.
constexpr uint64_t KEY_GENERIC = 0;      // trailing reduce, odometer walk
constexpr uint64_t KEY_ROWS = 1;         // many short contiguous rows
constexpr uint64_t KEY_COLS_CONTIG = 2;  // strided reduce, contiguous run span
constexpr uint64_t KEY_COLS_PAD = 3;     // strided reduce, padded multi-block batch
constexpr uint64_t KEY_ROWS_ONE = 4;     // KEY_ROWS when a core's whole share is one tile
// KEY_ROWS_ONE with its tiling reads collapsed to ONE 64-bit word (PackOneHdr).
constexpr uint64_t KEY_ROWS_HDR = 5;
// KEY_ROWS_HDR with TPipe/TBuf/InitBuffer removed as well (Greater's strip).
constexpr uint64_t KEY_ROWS_BARE = 6;
// Generic stage-1 with the partial folded into y by the MTE3 atomic adder;
// no workspace, no SyncAll, no stage 2.  The TilingFunc raises SetNeedAtomic
// for exactly this key.  See the long comment at the split site.
constexpr uint64_t KEY_RED_ATOMIC = 7;
// Split a contiguous reduced-run range across chunks, publish float-vector
// partials, and combine the slots. This extends parallelism when the kept
// inner run provides too few independent column pieces.
constexpr uint64_t KEY_COLS_POLLED = 8;
// Batch short reduced inner runs in an R K R layout with one multi-block
// DataCopyPad, then fold the accumulator to one scalar per output.
constexpr uint64_t KEY_STRIDED_SCALAR = 9;
// When the outer reduced group is outermost, each layer is contiguous.
// Stream layers into an accumulator slice and fold each innerLen segment
// at the end. The Gather-transposed slice must fit within 8192 elements.
constexpr uint64_t KEY_LAYERED = 10;
// Padded-run counterpart of the contiguous polled-combine route.
// It batches noncontiguous runs and publishes the same partial-slot format.
// The retag site requires one segment per independent unit.
constexpr uint64_t KEY_PAD_POLLED = 11;
// For an outermost reduced group, stream complete layers and fold the
// kept-output sub-blocks into partial slots. The polled combine sums chunks.
// Require outTiles*innerLen <= 4096 so slots and lanes fit the accumulator.
constexpr uint64_t KEY_KRK_POLLED = 12;
// Tiny-total direct route: totalBytes <= MICRO_BYTES, any shape, one core.
constexpr uint64_t KEY_MICRO = 13;

constexpr uint32_t SCALAR_ACC_LEN = 2048;
constexpr uint32_t STAGE_SLOTS = 240;
constexpr uint32_t STAGE_PITCH = 8;
constexpr uint32_t SMALL_TOTAL_BYTES = 64 * 1024;
// Core limit used by the optional sub-64-KiB byte cap.
// The default work floor uses input elements instead.
constexpr uint32_t SMALL_TOTAL_CORES = 6;

// Minimum input elements per core. Apply this work floor before deriving
// chunks, task counts, UB chunk size, and the tiling key. It can only reduce
// the hardware core budget.
constexpr uint64_t MIN_ELEMS_PER_CORE = 768;
// Minimum byte counts for a cross-core reduction split. Full reductions
// combine scalar partials, while kept-inner reductions combine vectors,
// so their fixed split and combine costs use separate thresholds.
constexpr uint64_t MIN_SPLIT_BYTES_FULL = 128ULL << 10;
constexpr uint64_t MIN_SPLIT_BYTES_KEPT = 2ULL << 20;
// Maximum input size for the single-core micro route. It loads one UB tile,
// folds squared values, and writes output without the queued task machinery.
constexpr uint64_t MICRO_BYTES = 8192;

// S9_VARIANT selections for SquareSumV1; zero uses the default.
// VAR_BYTE_CAP applies cores = min(cores, max(1, totalBytes / 2048))
// before deriving the task split.
constexpr uint32_t VAR_BYTE_CAP = 1;
constexpr uint64_t VAR_BYTE_CAP_MIN_BYTES_PER_CORE = 2048;
// VAR_NO_SMALL_CAP disables MIN_ELEMS_PER_CORE. The resulting core budget
// remains bounded by hardware and available work units.
constexpr uint32_t VAR_NO_SMALL_CAP = 2;
// VAR_NO_ROWS_ONE disables the KEY_ROWS_ONE promotion and keeps the
// compatible tiling data on the general row route.
constexpr uint32_t VAR_NO_ROWS_ONE = 3;
// VAR_SMALL_BYTE_CAP applies SMALL_TOTAL_CORES when totalBytes <= 64 KiB.
constexpr uint32_t VAR_SMALL_BYTE_CAP = 4;
// VAR_ROWS_ONE stops the row-route promotion ladder at KEY_ROWS_ONE.
// The packed header is appended, preserving the earlier tiling fields.
constexpr uint32_t VAR_ROWS_ONE = 5;
// VAR_ROWS_HDR stops the promotion ladder at KEY_ROWS_HDR, using the
// packed tiling word while retaining TPipe, TBuf, and InitBuffer.
constexpr uint32_t VAR_ROWS_HDR = 6;
constexpr uint32_t UB_RESERVE = 4 * 1024;
// Upper bound on a batched row.  The binding limit is really the UB budget
// (the rowBatch >= 2 guard below), so this is only a sanity cap; it used to
// be 8 KB, which kept the 8-23 KB band off the row route for no reason.
constexpr uint32_t MAX_ROW_BYTES = 64 * 1024;
constexpr uint64_t MAX_WS_FLOATS = 2ULL * 1024 * 1024;
constexpr uint64_t MIN_CHUNK_BYTES = 8 * 1024;
constexpr uint64_t SYS_WORKSPACE = 16ULL * 1024 * 1024;

// KEY_ROWS_ONE uses a straight-line kernel with a compile-time UB layout.
// The busiest core must fit its entire padded tile within ONE_TILE_ELEMS;
// this counts rows-per-core * rowPitch, including row padding.
constexpr uint32_t ONE_TILE_ELEMS = 2048;
// A padded row must fit ONE vector repeat (256 B of fp32 = 64 lanes), because
// that is what lets the fold be a single WholeReduceSum with no `stage`, no
// `work` and no inner loop.
constexpr uint32_t ONE_MAX_PITCH = 64;

struct Plan {
    bool ok = false;
    uint64_t tilingKey = KEY_GENERIC;
    uint64_t oneHdr = 0;   // PackOneHdr, only on KEY_ROWS_HDR / KEY_ROWS_BARE

    uint32_t innerIsReduce = 1;
    uint32_t nOuterK = 0;
    uint32_t nOuterR = 0;
    uint32_t usedCores = 1;
    uint32_t redChunks = 1;
    uint32_t splitRuns = 1;
    uint32_t rowBatch = 1;
    uint32_t rowPitch = 1;
    uint32_t accLen = 1;
    uint32_t ubChunk = 1;
    uint32_t workLen = 1;
    uint32_t outStage = 1;
    uint32_t runGroup = 0;
    uint32_t runLanes = 0;
    uint32_t runFold = 1;
    uint32_t tmpLen = 0;   // tmpBuf floats; 0 -> same as accLen

    uint64_t innerLen = 1;
    uint64_t colPiece = 1;
    uint64_t colChunks = 1;
    uint64_t outTiles = 1;
    uint64_t redPerTile = 1;
    uint64_t totalOut = 1;
    uint64_t indepUnits = 1;
    uint64_t outAligned = 1;

    uint64_t kSize[MAX_GROUP] = {1, 1, 1, 1};
    uint64_t kStride[MAX_GROUP] = {0, 0, 0, 0};
    uint64_t rSize[MAX_GROUP] = {1, 1, 1, 1};
    uint64_t rStride[MAX_GROUP] = {0, 0, 0, 0};

    uint64_t totalIn = 1;
    uint64_t userWsBytes = 0;
};

inline uint64_t PlanCeilDiv(uint64_t a, uint64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }
inline uint64_t PlanAlignUp(uint64_t a, uint64_t b) { return (b == 0) ? a : PlanCeilDiv(a, b) * b; }
inline uint64_t PlanMin(uint64_t a, uint64_t b) { return a < b ? a : b; }
inline uint64_t PlanMax(uint64_t a, uint64_t b) { return a > b ? a : b; }
inline uint64_t PlanGcd(uint64_t a, uint64_t b)
{
    while (b != 0) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Rows are at least 32 B of padded pitch each, so a tile of ONE_TILE_ELEMS
// elements can hold at most that many; and WholeReduceSum's repeat count is
// the row count, so it is also capped the way rowBatch is.
inline uint64_t OneMaxRows(uint32_t dtypeSize)
{
    return PlanMin(STAGE_SLOTS, static_cast<uint64_t>(ONE_TILE_ELEMS) * dtypeSize / 32);
}

// The UB map of SquareSumV1RowsOne, byte for byte, kept here so the host can
// refuse the route on a part whose UB cannot hold it.  in | acc(fp32) |
// dst(fp32 row sums) | out, and `out` does not exist for float32 because the
// row sums are already the output dtype and the store reads `dst` directly.
inline uint64_t OneUbBytes(uint32_t dtypeSize)
{
    const uint64_t rows = OneMaxRows(dtypeSize);
    uint64_t b = static_cast<uint64_t>(ONE_TILE_ELEMS) * dtypeSize +
                 static_cast<uint64_t>(ONE_TILE_ELEMS) * 4 + rows * 4;
    if (dtypeSize != 4) {
        b += PlanAlignUp(rows * dtypeSize, 32);
    }
    return b;
}

// Packed header for the row split, least-significant bits first:
// [0..15] innerLen; [16..31] rowPitch; [32..47] rows/usedCores;
// [48..55] rows%usedCores; [56..63] usedCores.
//
// The kernel computes nb = per + (idx < rem) and
// lo = idx*per + min(idx, rem), avoiding per-core division.
// OneHdrFits checks all field widths and route bounds; a value that cannot
// be represented retains KEY_ROWS_ONE.
inline bool OneHdrFits(uint64_t innerLen, uint64_t pitch, uint64_t per, uint64_t rem,
                       uint64_t cores)
{
    return innerLen <= 0xffffULL && pitch <= 0xffffULL && per <= 0xffffULL &&
           rem <= 0xffULL && cores >= 1 && cores <= 0xffULL;
}

inline uint64_t PackOneHdr(uint64_t innerLen, uint64_t pitch, uint64_t per, uint64_t rem,
                           uint64_t cores)
{
    return (innerLen & 0xffffULL) | ((pitch & 0xffffULL) << 16) | ((per & 0xffffULL) << 32) |
           ((rem & 0xffULL) << 48) | ((cores & 0xffULL) << 56);
}

// dims[0..rank) and reducedMask[0..rank) describe the input.
// dtypeSize and ubSize are in bytes. forceGeneric and noRunGroup select
// alternative planner paths. The outermost reduced-group extent and its
// inner run count are derived for the layered routes.
inline uint64_t rSizeOuter0(const Plan& p)
{
    return (p.nOuterR >= 1) ? p.rSize[0] : 1ULL;
}

inline Plan MakePlan(const int64_t* dims, uint32_t rank, const bool* reducedMask,
                     uint32_t dtypeSize, uint64_t ubSize, uint32_t hwCores,
                     bool forceGeneric = false, bool noRunGroup = false)
{
    Plan p;
    if (rank > MAX_DIM || dims == nullptr || reducedMask == nullptr || dtypeSize == 0) {
        return p;
    }

    // ---- strides and extents
    uint64_t dim[MAX_DIM] = {0};
    uint64_t stride[MAX_DIM] = {0};
    uint64_t running = 1;
    uint32_t reducedCount = 0;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        if (dims[i] < 0) {
            return p;
        }
        dim[i] = static_cast<uint64_t>(dims[i]);
        stride[i] = running;
        running *= dim[i];
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (reducedMask[i]) {
            ++reducedCount;
        }
    }
    p.totalIn = running;

    // ---- collapse into the alternating group chain
    uint64_t gSize[MAX_DIM] = {0};
    uint64_t gStride[MAX_DIM] = {0};
    bool gRed[MAX_DIM] = {false};
    uint32_t nG = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        if (dim[i] == 1) {
            continue;
        }
        const bool isRed = reducedMask[i];
        if (nG > 0 && gRed[nG - 1] == isRed) {
            gSize[nG - 1] *= dim[i];
            gStride[nG - 1] = stride[i];
        } else {
            gSize[nG] = dim[i];
            gStride[nG] = stride[i];
            gRed[nG] = isRed;
            ++nG;
        }
    }
    if (nG == 0) {
        gSize[0] = p.totalIn;
        gStride[0] = 1;
        gRed[0] = (reducedCount > 0);
        nG = 1;
    }

    p.innerIsReduce = gRed[nG - 1] ? 1U : 0U;
    p.innerLen = gSize[nG - 1];
    for (uint32_t i = 0; i + 1 < nG; ++i) {
        if (gRed[i]) {
            if (p.nOuterR >= MAX_GROUP) {
                return p;
            }
            p.rSize[p.nOuterR] = gSize[i];
            p.rStride[p.nOuterR] = gStride[i];
            ++p.nOuterR;
            p.redPerTile *= gSize[i];
        } else {
            if (p.nOuterK >= MAX_GROUP) {
                return p;
            }
            p.kSize[p.nOuterK] = gSize[i];
            p.kStride[p.nOuterK] = gStride[i];
            ++p.nOuterK;
            p.outTiles *= gSize[i];
        }
    }
    p.totalOut = p.outTiles * ((p.innerIsReduce != 0) ? 1ULL : p.innerLen);

    // ---- how many cores are worth using (data volume feature)
    if (hwCores == 0) {
        hwCores = 1;
    }
    const uint64_t totalBytes = p.totalIn * dtypeSize;
    uint64_t cores = hwCores;
    // Apply the element-based core floor before deriving redChunks, tasks,
    // usedCores, ubChunk, and the tiling key. Dependent fields must be computed
    // from the same core budget.
    if (s9::Variant() != VAR_NO_SMALL_CAP && s9::Variant() != VAR_SMALL_BYTE_CAP) {
        const uint64_t byWork = PlanCeilDiv(p.totalIn, MIN_ELEMS_PER_CORE);
        cores = PlanMin(cores, PlanMax(1, byWork));
    }
    // Apply the optional byte-based cap to the core budget before derivation.
    if (s9::Variant() == VAR_SMALL_BYTE_CAP && totalBytes <= SMALL_TOTAL_BYTES) {
        cores = PlanMin(cores, SMALL_TOTAL_CORES);
    }
    // The optional byte-based core cap is applied before route and task
    // derivation. Row outputs use byte-granular DataCopyPad stores, so disjoint
    // output row ranges may share a 32-byte GM block without sharing output bytes.
    if (s9::Variant() == VAR_BYTE_CAP) {
        cores = PlanMin(cores, PlanMax(1, totalBytes / VAR_BYTE_CAP_MIN_BYTES_PER_CORE));
    }
    cores = PlanMax(1, cores);

    // ---- UB budget first: it bounds how a kept run may be cut
    const uint32_t ubUsable =
        (ubSize > UB_RESERVE) ? static_cast<uint32_t>(ubSize - UB_RESERVE) : 32768U;
    const uint32_t stageBytes = STAGE_SLOTS * STAGE_PITCH * 4;

    p.tilingKey = KEY_GENERIC;
    p.accLen = SCALAR_ACC_LEN;
    p.workLen = 512;
    p.outStage = STAGE_SLOTS;
    {
        uint64_t chunk;
        if (p.innerIsReduce == 1) {
            const uint32_t fixed = stageBytes + SCALAR_ACC_LEN * 4 * 2 + 512 * 4 + STAGE_SLOTS * 4;
            chunk = (ubUsable > fixed) ? (ubUsable - fixed) / (2ULL * dtypeSize) : 256;
        } else {
            chunk = (ubUsable > 8192) ? (ubUsable - 8192) / (12ULL + 2ULL * dtypeSize) : 256;
        }
        chunk = PlanMax(256, (chunk / 256) * 256);
        p.ubChunk = static_cast<uint32_t>(PlanMin(chunk, 262144));
    }

    // ---- independent (workspace free) units
    //
    // A kept innermost run is cut ONLY because the accumulator has to fit in
    // UB.  Cutting it for parallelism instead would turn every read into a
    // short strided burst; parallelism is taken from the reduction dimension,
    // where the source runs of one core stay adjacent in memory.
    p.colChunks = 1;
    p.colPiece = p.innerLen;
    if (p.innerIsReduce == 0) {
        p.colChunks = PlanMax(1, PlanCeilDiv(p.innerLen, p.ubChunk));
        // If independent output units cannot fill the available cores, divide the
        // kept inner run into more column pieces. Keep each piece at least one
        // 32-byte vector block and honor the configured byte floor. Output pieces
        // are disjoint, so this split needs no partial-reduction workspace.
        if (p.outTiles * p.colChunks < cores) {
            const uint64_t blkElems = PlanMax(1, 32 / dtypeSize);
            // Use the configured byte floor when sizing each independent column piece.
            const uint64_t minPiece = PlanMax(blkElems, s9::ColFloorBytes() / dtypeSize);
            const uint64_t maxCC = PlanMax(1, p.innerLen / minPiece);
            const uint64_t wantCC = PlanCeilDiv(cores, PlanMax(1, p.outTiles));
            const uint64_t cc = PlanMin(wantCC, maxCC);
            if (cc > p.colChunks) {
                p.colChunks = cc;
            }
        }
        p.colPiece = PlanCeilDiv(p.innerLen, p.colChunks);
        // Piece boundaries land on 32 B blocks (except the tail), so every
        // chunk's DMA blocks and vector operands stay aligned.
        {
            const uint64_t blkElems = PlanMax(1, 32 / dtypeSize);
            p.colPiece = PlanMax(blkElems, PlanAlignUp(p.colPiece, blkElems));
        }
        p.colChunks = PlanCeilDiv(p.innerLen, PlanMax(1, p.colPiece));
    }
    p.indepUnits = PlanMax(1, p.outTiles * p.colChunks);

    // ---- split the reduction only when the output cannot fill the cores
    p.redChunks = 1;
    p.splitRuns = 1;
    // Cross-core reduction uses workspace partials, a synchronization point,
    // and a float32 combine followed by a single output cast. SetSysWorkspace
    // must be called before no-argument SyncAll, which reads the system workspace
    // pointer. Each partial is written before the combine consumes it.
    //
    // The separate atomic route additionally requires initialized output; the
    // API must not be assumed to zero output automatically. Split byte floors
    // limit use of the synchronization and combine machinery on small workloads.
    const uint64_t splitKnob = static_cast<uint64_t>(s9::SplitFloorKB()) << 10;
    const uint64_t floorFull = (splitKnob != 0) ? splitKnob : MIN_SPLIT_BYTES_FULL;
    const uint64_t floorKept = (splitKnob != 0) ? splitKnob : MIN_SPLIT_BYTES_KEPT;
    if (p.innerIsReduce == 1 && p.indepUnits < cores && totalBytes >= floorFull) {
        uint64_t splittable = p.redPerTile;
        if (p.redPerTile < PlanCeilDiv(cores, p.indepUnits)) {
            p.splitRuns = 0;
            splittable = p.innerLen;
        }
        const uint64_t want = PlanCeilDiv(cores, p.indepUnits);
        // never cut a task below MIN_CHUNK_BYTES of input
        const uint64_t bytesPerUnit = PlanMax(1, totalBytes / p.indepUnits);
        const uint64_t byWork = PlanMax(1, bytesPerUnit / MIN_CHUNK_BYTES);
        const uint64_t rc = PlanMax(1, PlanMin(want, PlanMin(byWork, PlanMax(1, splittable))));
        p.redChunks = static_cast<uint32_t>(rc);
    }
    // For a kept inner run with too few independent column units, reduction
    // chunks can provide additional parallelism. The eligibility checks below
    // select a compatible contiguous or padded publish path. The vector slot
    // combine merges partials, and the previous column split is undone.
    if (p.innerIsReduce == 0 && p.outTiles <= 16 && p.nOuterR >= 1 &&
        p.innerLen <= 8192 && p.indepUnits < cores && totalBytes >= floorKept) {
        // The padded body can batch noncontiguous reduced runs and publish the
        // same slot format; it does not require the contiguous-run stride condition.
        const uint64_t bytesPerUnit = PlanMax(1, totalBytes / PlanMax(1, p.outTiles));
        const uint64_t byWork = PlanMax(1, bytesPerUnit / MIN_CHUNK_BYTES);
        const uint64_t want = PlanCeilDiv(cores, PlanMax(1, p.outTiles));
        const uint64_t rc = PlanMax(1, PlanMin(want, PlanMin(byWork, p.redPerTile)));
        if (rc > 1) {
            p.colChunks = 1;
            p.colPiece = p.innerLen;
            p.indepUnits = p.outTiles;
            p.redChunks = static_cast<uint32_t>(rc);
            p.splitRuns = 1;
        }
    }
    // More than one wave of independent outputs can still leave a large
    // last-wave imbalance. Split long contiguous reductions into smaller tasks.
    if (p.innerIsReduce == 0 && p.nOuterR == 1 && p.rStride[0] == p.innerLen &&
        p.colChunks == 1 && p.indepUnits >= cores && p.indepUnits < cores * 8ULL) {
        const uint64_t bytesPerUnit = totalBytes / p.indepUnits;
        const uint64_t want = PlanCeilDiv(cores * 8ULL, p.indepUnits);
        const uint64_t byWork = PlanMax(1, bytesPerUnit / (128ULL * 1024ULL));
        p.redChunks = static_cast<uint32_t>(PlanMin(want, PlanMin(byWork, p.redPerTile)));
        p.splitRuns = 1;
    }
    if (p.redChunks > 1) {
        const uint32_t cap = s9::RedCap();
        if (cap != 0u && p.redChunks > cap) {
            p.redChunks = cap;
        }
    }
    const uint64_t tasks = p.indepUnits * static_cast<uint64_t>(p.redChunks);
    p.usedCores = static_cast<uint32_t>(PlanMax(1, PlanMin(cores, tasks)));

    if (p.innerIsReduce == 1) {
        const uint64_t rowBytes = p.innerLen * dtypeSize;
        if (!forceGeneric && p.nOuterR == 0 && p.redPerTile == 1 && p.outTiles >= 2 &&
            p.redChunks == 1 && rowBytes <= MAX_ROW_BYTES && p.innerLen >= 1) {
            const uint64_t pitch = PlanAlignUp(rowBytes, 32) / dtypeSize;
            // UB per batched row: the double-buffered input tile (2*dtypeSize)
            // and the fp32 squares (4).  tmpBuf holds only the `nb` reduce
            // results on this route, not nb*pitch of them, so it is one float
            // per row rather than one per element - the same for outQue.
            // + 40: tmpBuf keeps 8 floats per row (one 32 B block, so the packed
            //   WholeReduceSum destination is safe even if a repeat ever lands
            //   block-spaced) and outQue one.
            const uint64_t per = pitch * (2ULL * dtypeSize + 4) + 40;
            const uint64_t fixed = stageBytes + 1024;
            uint64_t nb = (ubUsable > fixed) ? (ubUsable - fixed) / per : 0;
            nb = PlanMin(nb, STAGE_SLOTS);
            nb = PlanMin(nb, p.outTiles);
            // Batching quantises a core's work into units of nb rows.  When
            // rows-per-core straddles a multiple of nb that turns a balanced
            // split into an unbalanced one - 3 rows against 2 at nb = 2 is one
            // batch against two, and the wall clock is the slowest core.  Take
            // the route only when every core ends up with the same number of
            // batches, or with enough batches that one extra is noise.
            const uint64_t nc = PlanMax(1, p.usedCores);
            const uint64_t rpcMax = PlanCeilDiv(p.outTiles, nc);
            const uint64_t rpcMin = p.outTiles / nc;
            const uint64_t bMax = (nb > 0) ? PlanCeilDiv(rpcMax, nb) : 0;
            const uint64_t bMin = (nb > 0) ? PlanCeilDiv(rpcMin, nb) : 0;
            const bool balanced = (bMax == bMin) || (bMin >= 3);
            if (nb >= 2 && balanced) {
                p.tilingKey = KEY_ROWS;
                p.rowBatch = static_cast<uint32_t>(nb);
                p.rowPitch = static_cast<uint32_t>(pitch);
                p.ubChunk = static_cast<uint32_t>(nb * pitch);
                p.accLen = p.ubChunk;
                p.tmpLen = p.rowBatch * STAGE_PITCH;
                p.workLen = 256;
                p.outStage = p.rowBatch;

                // Promote to KEY_ROWS_ONE only when the busiest core fits its whole share
                // in one padded tile with one vector reduction repeat per row. These
                // conditions refine the ordinary row-route conditions. Existing tiling
                // fields remain the same; only the selected kernel key changes.
                const uint64_t rpc = PlanCeilDiv(p.outTiles, nc);
                if (pitch <= ONE_MAX_PITCH && rpc <= nb &&
                    rpc * pitch <= ONE_TILE_ELEMS && rpc <= OneMaxRows(dtypeSize) &&
                    OneUbBytes(dtypeSize) <= ubSize && s9::Variant() != VAR_NO_ROWS_ONE) {
                    p.tilingKey = KEY_ROWS_ONE;

                    // KEY_ROWS_HDR / KEY_ROWS_BARE add bounds to KEY_ROWS_ONE eligibility.
                    // They read the appended packed header; the bare variant manages its UB
                    // views directly. Appending oneHdr preserves every earlier field offset.
                    const uint64_t per = p.outTiles / nc;
                    const uint64_t rem = p.outTiles % nc;
                    if (s9::Variant() != VAR_ROWS_ONE &&
                        OneHdrFits(p.innerLen, pitch, per, rem, nc)) {
                        p.oneHdr = PackOneHdr(p.innerLen, pitch, per, rem, nc);
                        p.tilingKey = (s9::Variant() == VAR_ROWS_HDR) ? KEY_ROWS_HDR
                                                                      : KEY_ROWS_BARE;
                    }
                }
            }
        }
    } else {
        p.accLen = p.ubChunk;
        p.workLen = 64;
        p.outStage = p.ubChunk;

        // Wide run-span route for strided reductions. A batch shares one widened
        // accumulator and vector accumulation. When the only outer reduced group
        // steps by one run, load a contiguous run span with one descriptor.
        //
        // Vector operands require 32-byte alignment. If a run is not aligned, use
        // the smallest aligned multiple runFold*innerLen, then fold the remaining
        // runFold partials with scalar operations.
        p.runGroup = 0;
        p.runLanes = 0;
        p.runFold = 1;
        if (p.nOuterR == 1 && p.colChunks == 1 && p.rStride[0] == p.innerLen &&
            p.redPerTile >= 2 && !noRunGroup) {
            const uint64_t lanes32 = 32 / dtypeSize;
            const uint64_t g0 = lanes32 / PlanGcd(p.innerLen, lanes32);
            const uint64_t base = g0 * p.innerLen;
            const uint64_t cap = PlanMin(p.ubChunk, p.accLen);
            if (base <= cap) {
                uint64_t g = g0;
                uint64_t L = base;
                while (L * 2 <= cap && g * 2 <= p.redPerTile) {
                    L *= 2;
                    g *= 2;
                }
                p.runGroup = static_cast<uint32_t>(g);
                p.runLanes = static_cast<uint32_t>(L);
                p.runFold = static_cast<uint32_t>(g0);
            }
        }
        // The kernel used to pick between these two at run time on runGroup.
        // Naming them as separate keys is what actually keeps the unused body
        // out of the variant that runs.
        p.tilingKey = (p.runGroup != 0) ? KEY_COLS_CONTIG : KEY_COLS_PAD;
    }

    // Retag split reductions after all family-key assignments. Select the
    // variant compatible with the partial layout. If no compatible split route
    // is available, remove the split and restore the unsplit task count.
    if (p.redChunks > 1) {
        if (p.tilingKey == KEY_GENERIC) {
            p.tilingKey = KEY_RED_ATOMIC;
        } else if (p.tilingKey == KEY_COLS_CONTIG) {
            p.tilingKey = KEY_COLS_POLLED;
        } else if (p.tilingKey == KEY_COLS_PAD && p.colChunks == 1 &&
                   p.innerLen <= p.ubChunk) {
            // When the outermost group is reduced, the complete layer below it is
            // contiguous and can use the layered route. Otherwise retain padded publishing.
            const uint64_t l0 = rSizeOuter0(p);
            const uint64_t layerW = p.outTiles * p.innerLen;          // 真实层宽
            const uint64_t slotW = p.outTiles * ((p.innerLen + 7ULL) & ~7ULL);  // 槽容量(带行pitch)
            const uint64_t innerRuns = (l0 != 0) ? (p.redPerTile / l0) : 0;
            // Feature gate, not a shape: a chunk streams its units one DMA
            // each, so a unit must carry real bytes.  innerRuns==1 with a
            // short kept run degenerates to one tiny descriptor per LAYER
            // ([47000,289]-class: 578-byte DMAs by the ten thousand per
            // core); that family belongs to the PAD body, whose batched
            // multi-block descriptor rides dozens of runs per issue.
            const uint64_t subBytes = innerRuns * p.innerLen * dtypeSize;
            // Check the initial ladder value before doubling. A starting value above
            // 4096 cannot fit the lanes region and would make the computed slab zero.
            // An unfittable ladder retains KEY_PAD_POLLED.
            const uint64_t lanes32g = 32 / dtypeSize;
            const uint64_t ladderBase =
                (lanes32g / PlanGcd(p.innerLen, lanes32g)) * p.innerLen;
            if (p.nOuterR >= 1 && l0 >= 2 && slotW <= 4096ULL && innerRuns >= 1 &&
                ladderBase <= 4096ULL &&
                (innerRuns >= 2 || subBytes >= 4096ULL) &&
                p.rStride[0] == layerW * innerRuns) {
                p.tilingKey = KEY_KRK_POLLED;
                p.redChunks = static_cast<uint32_t>(
                    PlanMax(2, PlanMin(static_cast<uint64_t>(cores), l0)));
                p.splitRuns = 1;
                p.indepUnits = 1;
                p.usedCores = p.redChunks;
                // The layered body folds each contiguous sub-block through
                // the run-group ladder; PAD plans carry runLanes==0, so set
                // the ladder here exactly the way the CONTIG arm does, with
                // the lanes region capped at half the 8192 accumulator (the
                // slot region owns the other half).
                {
                    const uint64_t lanes32 = 32 / dtypeSize;
                    const uint64_t g0 = lanes32 / PlanGcd(p.innerLen, lanes32);
                    uint64_t g = g0;
                    uint64_t L = g0 * p.innerLen;
                    while (L * 2 <= 4096ULL && g * 2 <= innerRuns) {
                        L *= 2;
                        g *= 2;
                    }
                    p.runGroup = static_cast<uint32_t>(g);
                    p.runLanes = static_cast<uint32_t>(L);
                    p.runFold = static_cast<uint32_t>(g0);
                }
            } else {
                p.tilingKey = KEY_PAD_POLLED;
            }
        } else if (p.tilingKey == KEY_COLS_PAD && p.colChunks > 1 &&
                   p.colPiece <= p.ubChunk) {
            // PAD with a column split: each (tile, piece) unit publishes ONE
            // partial per chunk at the uniform colPiece slot pitch, and the
            // combiner stores each unit at its own column offset (ragged
            // tail narrower, same pitch).  This branch used to fall through
            // to the serial fallback below -- a shape family whose kept run
            // is too misaligned for the CONTIG ladder and whose column split
            // cannot fill the cores alone then ran the whole reduction on
            // indepUnits cores.  [470069, 289]-class members measure in the
            // hundreds of microseconds where the split family runs tens.
            p.tilingKey = KEY_PAD_POLLED;
        } else {
            // A kept run that cannot fit the run-group accumulator and supplies too
            // few column pieces remains on the padded route.
            p.redChunks = 1;
            p.splitRuns = 1;
            p.usedCores = static_cast<uint32_t>(
                PlanMax(1, PlanMin(static_cast<uint64_t>(cores), p.indepUnits)));
        }
    }

    // The batched strided-scalar promotion, AFTER the split retag: it owns
    // only unsplit generic plans.  Gates are features: one outer reduced
    // group (runs equidistant, batchable), at least one kept group (a scalar
    // per output, many outputs), a reduced inner run short enough that the
    // generic route's per-run DMA is the cost (<= 512 bytes), and enough
    // runs for a batch to mean anything.
    if (p.tilingKey == KEY_GENERIC && p.redChunks == 1 && p.innerIsReduce == 1 &&
        p.nOuterR == 1 && p.nOuterK >= 1 && p.redPerTile >= 8 &&
        p.innerLen * dtypeSize <= s9::RunMaxBytes() &&
        // The layered tail transposes outPerSlice*innerLen floats through the
        // accumulator. outPerSlice is rounded up to a multiple of eight for
        // 32-byte row alignment, so require innerLen*8 <= accLen.
        p.innerLen * 8ULL <= 8192ULL) {
        // Layer-contiguity: the outer reduced dim outermost means
        // rStride[0] == outTiles * innerLen exactly.  Then the layered walk
        // (key 10) supersedes the strided batch: sequential reads only.
        // The Gather-transpose tail needs slice = nOut*innerLen <= 8192 with
        // nOut >= 1, i.e. innerLen <= 8192 -- already true under the 512-byte
        // gate.  workLen becomes the offset table (8192 u32), tmpLen the
        // transpose destination; both live in the one arena.
        if (p.rStride[0] == p.outTiles * p.innerLen) {
            p.tilingKey = KEY_LAYERED;
        } else {
            p.tilingKey = KEY_STRIDED_SCALAR;
        }
        // Batch size is bounded by accLen, ubChunk, and 4095*pitch. The larger
        // accumulator permits more runs per batch, subject to this route UB budget.
        p.accLen = 8192;
        p.workLen = (p.tilingKey == KEY_LAYERED) ? 8192U : 256U;
        if (p.tilingKey == KEY_LAYERED) {
            p.tmpLen = 8192;
        }
        // Recompute ubChunk from this route arena sizes. Reusing the generic
        // chunk bound after increasing the accumulator could overlap the input queue
        // and the calculation arena.
        {
            const uint32_t fixed9 =
                stageBytes + p.accLen * 4U * 2U + p.workLen * 4U + STAGE_SLOTS * 4U;
            uint64_t c9 = (ubUsable > fixed9)
                              ? (ubUsable - fixed9) / (2ULL * dtypeSize)
                              : 256ULL;
            c9 = PlanMax(256, (c9 / 256) * 256);
            p.ubChunk = static_cast<uint32_t>(PlanMin(c9, 262144));
        }
    }

    if (p.tilingKey == KEY_ROWS && p.rowPitch > 64 && p.rowPitch <= 8192) {
        const uint64_t target = PlanAlignUp(p.innerLen, 64);
        if (target <= 8192 && (target == p.rowPitch || target * dtypeSize / 32 <= 255)) {
            p.rowPitch = static_cast<uint32_t>(target);
        }
        const uint64_t initialBlocks = p.rowPitch / 8;
        uint64_t blocks = initialBlocks;
        while ((blocks & 1) == 0) blocks >>= 1;
        uint64_t nb = p.rowBatch;
        while (nb > 1) {
            const uint64_t rows = PlanAlignUp(nb, 8);
            const uint64_t table = rows * blocks;
            const uint64_t bytes = stageBytes + nb * p.rowPitch * (2ULL * dtypeSize + 4ULL)
                + 4ULL * (rows * initialBlocks + table + rows);
            if (table <= 8192 && bytes <= ubUsable) break;
            --nb;
        }
        p.rowBatch = static_cast<uint32_t>(nb);
        p.ubChunk = p.accLen = p.rowBatch * p.rowPitch;
        p.tmpLen = static_cast<uint32_t>(PlanAlignUp(nb, 8) * initialBlocks);
        p.workLen = static_cast<uint32_t>(PlanAlignUp(nb, 8) * blocks);
        p.outStage = static_cast<uint32_t>(PlanAlignUp(nb, 8));
    }

    if (p.tilingKey == KEY_LAYERED) {
        // A streaming accumulator plus one input tile in each queue slot.
        // Gather operates on smaller sub-slices; no scalar staging area is used.
        if (p.innerLen <= 256) {
            p.tmpLen = 1024; p.workLen = 512; p.outStage = 512;
        } else {
            p.tmpLen = p.workLen = 8192;
        }
        const uint64_t fixed = 4ULL * (p.tmpLen + p.workLen + p.outStage);
        p.accLen = static_cast<uint32_t>(((ubUsable - fixed) / (4ULL + 2ULL * dtypeSize) / 256ULL) * 256ULL);
        p.ubChunk = p.accLen;
    }

    if (p.tilingKey == KEY_COLS_CONTIG || p.tilingKey == KEY_COLS_POLLED) {
        // Keep only live accumulator/output data; devote the remaining UB to reads.
        p.accLen = p.runLanes;
        p.tmpLen = (dtypeSize == 4) ? 128U : p.runLanes;
        p.workLen = static_cast<uint32_t>(PlanMax(128, p.redChunks * 8ULL));
        p.outStage = static_cast<uint32_t>(PlanAlignUp(PlanMax(p.innerLen, 4), 8));
        const uint64_t fixed = stageBytes + 4ULL *
            (p.accLen + p.tmpLen + p.workLen + p.outStage);
        const uint64_t cap = (ubUsable - fixed) / (2ULL * dtypeSize);
        p.ubChunk = static_cast<uint32_t>((cap / p.runLanes) * p.runLanes);
        const uint64_t table = p.innerLen * p.runFold;
        if (dtypeSize == 2 && p.runFold >= 8 && table <= 8192) {
            const uint64_t work = PlanMax(p.workLen, table);
            const uint64_t largerFixed = fixed + (work - p.workLen) * 4ULL;
            const uint64_t inputBytes = 2ULL * dtypeSize * p.ubChunk;
            if (largerFixed + inputBytes <= ubUsable) p.workLen = static_cast<uint32_t>(work);
        }
    }

    if (p.tmpLen == 0) {
        p.tmpLen = p.accLen;
    }

    // Apply the micro-route retag after family selection. The gate includes
    // input size and route eligibility; row and contiguous scalar-output paths
    // keep their existing reductions. The micro kernel reads group arrays,
    // innerLen, and totalOut rather than fields derived from the previous core
    // split, so key and core-count changes are paired here.
    const bool microGeneric = (p.tilingKey == KEY_GENERIC &&
                               p.innerIsReduce == 1 && p.totalOut >= 32);
    if (totalBytes <= MICRO_BYTES && p.redChunks == 1 &&
        (microGeneric || p.tilingKey == KEY_COLS_CONTIG ||
         p.tilingKey == KEY_COLS_PAD || p.tilingKey == KEY_STRIDED_SCALAR)) {
        p.tilingKey = KEY_MICRO;
        p.usedCores = 1;
        p.indepUnits = 1;
    }

    p.outAligned = PlanAlignUp(p.totalOut, 8);
    // KEY_RED_ATOMIC publishes one 16-byte validated slot per (unit, chunk);
    // the retired SyncAll route kept one float per (chunk, aligned output).
    // Size for whichever the key uses.
    if (p.redChunks > 1) {
        if (p.tilingKey == KEY_RED_ATOMIC) {
            p.userWsBytes = p.indepUnits * static_cast<uint64_t>(p.redChunks) * 16ULL;
        } else if (p.tilingKey == KEY_COLS_POLLED || p.tilingKey == KEY_PAD_POLLED ||
                   p.tilingKey == KEY_KRK_POLLED) {
            // KRK slots carry the WHOLE output (outTiles*innerLen floats),
            // not one kept-inner run -- sizing them by innerLen undercounts
            // 10x and the publishes write past the workspace.
            const uint64_t slotFloats =
                (p.tilingKey == KEY_KRK_POLLED)
                    ? (p.outTiles * ((p.innerLen + 7ULL) & ~7ULL))
                    : ((p.tilingKey == KEY_PAD_POLLED && p.colChunks > 1)
                           ? p.colPiece
                           : p.innerLen);
            const uint64_t pitch = ((16ULL + slotFloats * 4ULL) + 31ULL) & ~31ULL;
            p.userWsBytes = p.indepUnits * static_cast<uint64_t>(p.redChunks) * pitch;
        } else {
            p.userWsBytes = static_cast<uint64_t>(p.redChunks) * p.outAligned * sizeof(float);
        }
    } else {
        p.userWsBytes = 0ULL;
    }
    p.ok = true;
    return p;
}

}  // namespace squaresum

#endif  // S9_SQUARE_SUM_V1_PLAN_H
