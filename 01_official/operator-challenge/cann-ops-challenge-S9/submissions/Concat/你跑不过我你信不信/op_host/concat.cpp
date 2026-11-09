#include <cstdint>

#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
constexpr uint32_t MAX_T = 256;
constexpr uint64_t TILING_KEY_GENERAL = 0;
constexpr uint64_t TILING_KEY_DIM0_SPLIT = 1;
constexpr uint64_t TILING_KEY_UNIVERSAL = 4;
constexpr uint64_t TILING_KEY_GROUPED_GATHER = 6;
constexpr uint64_t TILING_KEY_DIM0_DIRECT = 7;
constexpr uint64_t TILING_KEY_PREDECODED3 = 8;
constexpr uint64_t TILING_KEY_PREDECODED4 = 9;
constexpr uint64_t TILING_KEY_TALL_BAND = 10;
constexpr uint64_t TILING_KEY_STREAM_L2_BYPASS = 11;
constexpr uint64_t TILING_KEY_TALL_BAND_SMALL = 12;
constexpr uint64_t TILING_KEY_TALL_BAND_MR4 = 13;
// Route-6 grouping scan, shared verbatim with the kernel replay (kernel gets TH via gcTbl,
// mode in gcTbl's high bit). mode 0 = legacy "close when acc >= TH" (groups OVERSHOOT TH, so
// the trailing group is whatever is left over -> a RUNT); mode 1 = "close before acc would
// exceed TH" (every group <= TH, capacity-greedy -> near-equal widths, no runt).
// Returns group count + widest group width + widest packed-src need.
struct R6Scan {
  uint32_t gg = 0, mGW = 0, mSrc = 0;
  bool bad = false;
};
static R6Scan R6Group(const uint32_t* widths, uint32_t numActive, uint32_t TH, uint32_t mode,
                      uint32_t Rb0, uint32_t e32) {
  R6Scan r;
  uint32_t acc = 0, gS = 0;
  for (uint32_t k = 0; k < numActive; k++) {
    if (mode == 1u && acc > 0 && acc + widths[k] > TH) {   // close [gS, k-1]
      if (r.gg >= 64) { r.bad = true; return r; }
      uint32_t s = 0;
      for (uint32_t m = gS; m < k; m++) {
        uint32_t b = Rb0 * widths[m]; b = ((b + e32 - 1) / e32) * e32; s += b;
      }
      if (acc > r.mGW) r.mGW = acc;
      if (s > r.mSrc) r.mSrc = s;
      r.gg++; gS = k; acc = 0;
    }
    acc += widths[k];
    bool close = (mode == 0u) ? (acc >= TH || k == numActive - 1) : (k == numActive - 1);
    if (close) {
      if (r.gg >= 64) { r.bad = true; return r; }
      uint32_t s = 0;
      for (uint32_t m = gS; m <= k; m++) {
        uint32_t b = Rb0 * widths[m]; b = ((b + e32 - 1) / e32) * e32; s += b;
      }
      if (acc > r.mGW) r.mGW = acc;
      if (s > r.mSrc) r.mSrc = s;
      r.gg++; gS = k + 1; acc = 0;
    }
  }
  return r;
}
constexpr uint32_t TILE_ELEMS = 16384;  // BUFFER_NUM=2 -> 2x16384 fits UB; bigger tile = more rows/cols per strided move
constexpr uint32_t GATHER_WBATCH = 8;   // Route G: rows gathered per store batch (must match kernel)


static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  ConcatTilingData tiling;
  auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t maxCores = plat.GetCoreNumAiv();
  // number of TENSORS in the dynamic input (GetComputeNodeInputNum counts
  // input GROUPS = 1 for a single dynamic input, not the tensor count).
  uint32_t numInputs = 0;
  auto* nodeInfo = context->GetComputeNodeInfo();
  if (nodeInfo != nullptr) {
    auto* inst = nodeInfo->GetInputInstanceInfo(0);
    if (inst != nullptr) {
      numInputs = inst->GetInstanceNum();
    }
  }
  if (numInputs == 0) {
    numInputs = context->GetComputeNodeInputNum();
  }

  auto* attrs = context->GetAttrs();
  const int64_t* dimPtr = attrs->GetInt(0);
  int64_t dim = (dimPtr != nullptr) ? *dimPtr : 0;

  auto outShape = context->GetOutputShape(0)->GetStorageShape();
  uint32_t ndim = outShape.GetDimNum();
  if (ndim == 0) ndim = 1;
  int64_t dn = dim < 0 ? dim + (int64_t)ndim : dim;
  if (dn < 0) dn = 0;
  if (dn >= (int64_t)ndim) dn = (int64_t)ndim - 1;

  uint32_t outer = 1, inner = 1;
  for (uint32_t i = 0; i < (uint32_t)dn; i++) {
    outer *= (uint32_t)outShape.GetDim(i);
  }
  for (uint32_t i = (uint32_t)dn + 1; i < ndim; i++) {
    inner *= (uint32_t)outShape.GetDim(i);
  }
  uint32_t sumW = (uint32_t)outShape.GetDim(dn);

  uint32_t origIdx[MAX_T] = {0};
  uint32_t widths[MAX_T] = {0};
  uint32_t offsets[MAX_T] = {0};
  uint32_t numActive = 0, off = 0;
  for (uint32_t i = 0; i < numInputs; i++) {
    auto sh = context->GetDynamicInputShape(0, i)->GetStorageShape();
    uint32_t wi = (uint32_t)sh.GetDim(dn);
    if (wi > 0 && numActive < MAX_T) {
      origIdx[numActive] = i;
      widths[numActive] = wi;
      offsets[numActive] = off;
      numActive++;
    }
    off += wi;
  }

  // dtype byte size + per-input 32B-alignment, for route decisions.
  int64_t dtypeSize = 4;
  {
    auto* idesc = context->GetInputDesc(0);
    if (idesc != nullptr) {
      int64_t s = ge::GetSizeByDataType(idesc->GetDataType());
      if (s > 0) dtypeSize = s;
    }
  }
  // Small outputs: every extra core adds launch and Init time, while bandwidth past ~24 readers
  // is not what a few MB are waiting on (a 4.2 MB case measured 0.3-0.6 us faster on 24 cores than
  // on 40 in three sessions; a 9.5 MB one was not faster, so the cut stays below it).
  if ((uint64_t)outer * sumW * inner * (uint64_t)dtypeSize <= (8ULL << 20) && maxCores > 24u) {
    maxCores = 24u;
  }

  uint32_t totalRuns = outer * numActive;
  uint32_t coreNum = totalRuns < maxCores ? totalRuns : maxCores;
  if (coreNum < 1) coreNum = 1;
  uint32_t runsPerCore = (totalRuns + coreNum - 1) / coreNum;
  if (runsPerCore < 1) runsPerCore = 1;
  coreNum = (totalRuns + runsPerCore - 1) / runsPerCore;
  if (coreNum < 1) coreNum = 1;
  bool allAligned = (numActive > 0);
  uint64_t maxSeg = 0;
  for (uint32_t k = 0; k < numActive; k++) {
    uint64_t seg = (uint64_t)widths[k] * inner;
    if (seg > maxSeg) maxSeg = seg;
    if ((seg * (uint64_t)dtypeSize) % 32ULL != 0ULL) allAligned = false;
  }

  uint32_t route = 0, perCoreElems = 0, totalElems = 0, nRowChunks = 0, rowsPerChunk = 0;
  uint32_t gR = 0, gCoreRows = 0, gUbSrcElems = 0;
  uint32_t slotElem[MAX_T] = {0};
  uint32_t directOrigIdx[MAX_T] = {0};
  uint32_t directSrcOff[MAX_T] = {0};
  uint32_t directDstOff[MAX_T] = {0};
  uint32_t directLen[MAX_T] = {0};
  uint64_t outRowBytes = (uint64_t)sumW * inner * (uint64_t)dtypeSize;
  if (dn == 0 && numActive > 0 && outer == 1 &&
      (numActive < maxCores || (uint64_t)sumW * inner >= 1048576ULL)) {
    // Route P (dim0 split-core): concat on first axis => output is the inputs laid
    // end-to-end *contiguously*. v1 parallelizes only over numActive runs -> idle
    // cores (catastrophe). Split output [0,totalElems) evenly across all cores.
    // Gate: few inputs (numActive<cores, v1 would idle), OR large total (>=1M elems)
    // where even splitting also beats per-part scheduling. Small totals remain
    // on the lower-overhead route.
    route = 1;
    uint64_t te = (uint64_t)sumW * inner;
    totalElems = (uint32_t)te;
    uint32_t C = maxCores;
    if (C & 1u) C -= 1;            // even core count (vector cores paired)
    if (C < 1) C = 1;
    uint32_t per = (uint32_t)((te + C - 1) / C);
    per = ((per + 31u) / 32u) * 32u;   // multiple of 32 elems => start >=32B aligned for any dtype
    if (per < 32u) per = 32u;
    perCoreElems = per;
    coreNum = (uint32_t)((te + per - 1) / per);
    if (coreNum < 1) coreNum = 1;
    // Route-1 boundary table reuses slotElem[]. For large int8 copies, round
    // each internal boundary to the nearest 512B offset relative to the source
    // input containing it. This keeps source starts cache-line aligned while
    // moving any boundary by at most 256B. Other route-1 shapes retain the
    // original equal split exactly.
    for (uint32_t b = 0; b <= coreNum; b++) {
      uint64_t boundary = (uint64_t)b * per;
      if (boundary > te) boundary = te;
      if (dtypeSize == 1 && te >= 1048576ULL && boundary > 0 && boundary < te) {
        for (uint32_t k = 0; k < numActive; k++) {
          uint64_t inputBegin = (uint64_t)offsets[k] * inner;
          uint64_t inputLen = (uint64_t)widths[k] * inner;
          if (boundary < inputBegin + inputLen) {
            uint64_t rel = boundary - inputBegin;
            uint64_t alignedRel = ((rel + 256ULL) / 512ULL) * 512ULL;
            if (alignedRel > inputLen) alignedRel = inputLen;
            boundary = inputBegin + alignedRel;
            break;
          }
        }
      }
      slotElem[b] = (uint32_t)boundary;
    }
  } else if (dn > 0 && numActive > 0 && allAligned && maxSeg <= TILE_ELEMS &&
             outRowBytes < (1ULL << 31)) {
    // Route S (aligned strided collapse): non-first axis, every segment 32B-aligned.
    // Collapse v1's per-(row,input) tiny copies into one contiguous load + one
    // multi-row strided store per (input, row-batch). Split each input's rows into
    // nRowChunks pieces so all cores are used even when numActive is small.
    route = 2;
    uint32_t nrc = 1;
    if (numActive < maxCores) {
      nrc = maxCores / numActive;
      if (nrc < 1) nrc = 1;
      if (nrc > outer) nrc = outer;
      if (nrc < 1) nrc = 1;
    }
    nRowChunks = nrc;
    rowsPerChunk = (outer + nrc - 1) / nrc;
    uint32_t units = numActive * nrc;
    uint32_t cc = units < maxCores ? units : maxCores;
    if (cc < 1) cc = 1;
    uint32_t upc = (units + cc - 1) / cc;
    if (upc < 1) upc = 1;
    cc = (units + upc - 1) / upc;
    if (cc < 1) cc = 1;
    totalRuns = units;      // route 2 reuse: units
    runsPerCore = upc;      // route 2 reuse: unitsPerCore
    coreNum = cc;
  } else if (inner == 1 && numActive > 0 && !allAligned && dtypeSize >= 2 &&
             (uint64_t)outer * sumW * inner >= 1048576ULL) {
    // Gather assembly only pays off for large inner=1 streams; small tensors are
    // descriptor/launch-bound and fall through to the lean DataCopyPad route.
    // Route G (Gather column-assembly): non-aligned last-axis (inner==1). Load each
    // input's row-batch contiguously into UB, Gather-assemble full output rows in UB
    // (vector unit handles the non-aligned column scatter), then store rows contiguous
    // -- collapsing the tiny-copy flood. Only viable for NARROW outputs (uint32 offset
    // arrays are 2x the data; wide sumW can't batch rows -> falls through to v1).
    uint32_t e32 = (uint32_t)(32 / dtypeSize);
    if (e32 < 1) e32 = 1;
    uint64_t ubBytes = 180ULL * 1024;
    uint64_t offElems = ((uint64_t)(sumW + 63) / 64) * 64;     // vector-op pad
    uint64_t pitchElem = ((uint64_t)(sumW + 127) / 128) * 128; // gather-dst pad (>=repeat)
    uint64_t offBytes = 3ULL * offElems * 4;                   // base,inc,cur (int32)
    uint64_t storeBytes = 2ULL * GATHER_WBATCH * pitchElem * (uint64_t)dtypeSize; // gStoreQue depth 2 x WBATCH rows
    uint64_t fixed = offBytes + storeBytes;
    uint64_t padSrc = (uint64_t)numActive * e32 * (uint64_t)dtypeSize;
    uint64_t perRowSrc = (uint64_t)sumW * (uint64_t)dtypeSize;  // ubSrc per row
    if (fixed + padSrc + 2048 < ubBytes && perRowSrc > 0) {
      uint32_t R = (uint32_t)((ubBytes - fixed - padSrc) / perRowSrc);
      if (R > outer) R = outer;
      if (R > 4095u) R = 4095u;
      if (R >= 8u) {                                      // require comfortable batching
        route = 3;
        gR = R;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < numActive; k++) {
          slotElem[k] = acc;
          uint32_t blk = R * widths[k];
          blk = ((blk + e32 - 1) / e32) * e32;            // 32B-align each input block start
          acc += blk;
        }
        gUbSrcElems = acc;
        uint32_t C = outer < maxCores ? outer : maxCores;
        if (C & 1u) C -= 1;
        if (C < 1) C = 1;
        gCoreRows = (outer + C - 1) / C;
        coreNum = (outer + gCoreRows - 1) / gCoreRows;
        if (coreNum < 1) coreNum = 1;
      }
    }
  }

  // Route U (universal strided-collapse): ANY non-first-axis case the fast routes
  // missed -> route 0 flood (non-aligned mid / NCHW C-concat, wide non-aligned last
  // (route 3 R<8 fallthrough), int8 non-aligned, aligned big-seg maxSeg>tileLen,
  // mixed-align). Per (input, row-batch[, col-block]): blockCount=R Normal-mode load
  // (auto-pads each row to 32B in UB) + multi-row strided store (GM-byte dstStride) ->
  // collapses the flood for ANY alignment/dtype. Non-aligned strided store is valid on
  // A2 (GM stride is bytes; Normal-mode CopyOut, no Compact needed) -- key correction.
  if (route == 0 && dn > 0 && numActive > 0 && outRowBytes < (1ULL << 31)) {
    // Route 5 (read-efficient route-U): for fp/int32 (dtypeSize>=2, Gather-capable)
    // whose widest segment fits a packed read (maxSeg<=PAD_elems), the kernel does a
    // blockCount=1 PACKED read (full BW) + UB Gather repack instead of route-4's
    // mte2-throttled padded read. int8 (no A2 Gather) and wide segments stay route 4.
    uint64_t padElems = 32768ull / (uint64_t)dtypeSize;   // must match kernel PAD
    // route 5 (packed read + Gather repack) only wins when READ-bound (large data);
    // small cases are overhead-bound -> lean pure-DataCopyPad route 4.
    bool bigEnough = ((uint64_t)outer * sumW * inner >= 1048576ULL);
    // route5's Gather repack is ~equal to route4 on CANN9.0 but disproportionately slow
    // on CANN8.5 (target platform); keep route5 only for last-axis (inner==1) narrow-write cases (large-stream workload)
    // where the packed read genuinely wins, and send wide-inner cases (large-strided workload) to gather-free route4.
    route = (bigEnough && dtypeSize >= 2 && maxSeg <= padElems && inner == 1) ? 5 : 4;
    uint32_t nrc = 1;
    if (numActive < maxCores) {
      nrc = maxCores / numActive;
      if (nrc < 1) nrc = 1;
      if (nrc > outer) nrc = outer;
      if (nrc < 1) nrc = 1;
    } else if (route == 5) {
      // many inputs (numActive>=cores) -> route 4 would collapse to ~33 cores; split
      // each input's rows into 2 chunks so all ~40 cores get work.
      nrc = 2;
      if (nrc > outer) nrc = outer;
      if (nrc < 1) nrc = 1;
    }
    // Row-band L2-coalescing (2026-07-05): shrink row-bands so a core writes ALL its
    // adjacent-column pieces to the SAME small band of rows back-to-back (k-fastest u-order).
    // Then the partial sub-512B GM write bursts from neighbouring pieces land in the same
    // L2 line while it is still resident -> coalesce to full-line flushes. Sweet spot =
    // band working-set ~= L2 (~8MB); beyond that per-unit overhead dominates. Measured large-stream workload
    // 324->295us (-29us) at ~8MB bands. Only shrinks (never below the balance nrc).
    {
      uint64_t bandWS = (uint64_t)outer * outRowBytes;              // full output bytes
      uint32_t nrcL2 = (uint32_t)((bandWS + (8ULL << 20) - 1) / (8ULL << 20));
      if (nrcL2 > nrc) nrc = nrcL2;
      // Force 1 row-band/core (tightest L2 write-coalescing) ONLY where it pays: route 5, or a
      // large multi-band output. Ungated it over-shards launch/read-bound small cases (tiny workload/wide-row workload), which
      // want the balanced nrc from the piece-count split above. large-strided workload/large-stream workload keep nrc=maxCores.
      if ((route == 5 || (outer >= maxCores && bandWS >= (8ULL << 20))) && maxCores > nrc) nrc = maxCores;
      if (nrc > outer) nrc = (uint32_t)outer;
      if (nrc < 1) nrc = 1;
    }
    // CORE BALANCE (route 4, few outer rows). With outer < maxCores the piece-count
    // split above leaves units = numActive*nrc <= maxCores, so upc == 1 and every core
    // owns exactly ONE (segment, row-chunk) unit. Unit COST scales with widths[k], and
    // the widths are arbitrary (wide-row workload: 3..63), so the widest segment alone sets the kernel
    // time. MEASURED per-core (msprof op PipeUtilization block_id rows) on wide-row workload: 32 blocks
    // spanning 9.8..24.4us, max/mean 1.37, max/min 2.44.
    // Fix: raise the row-chunk count so units > maxCores and upc > 1. The kernel decodes
    // k-FASTEST (k = u % numActive), so the upc consecutive units a core owns are
    // consecutive SEGMENTS and their widths AVERAGE OUT.
    // How far to push it is a trade-off, both halves measured on wide-row workload:
    //   rowsPerChunk=1 (nrc=outer=18, units=288, upc=8, 36 cores): max per-core load
    //     567 -> 302 width-rows, imbalance 1.37 -> 1.15, BUT aiv_scalar 8.2 -> 15.5us
    //     and total mte2 +32% (one-row DMA bursts) -> net ~0 on Task Duration.
    //   rowsPerChunk=2 (nrc=9, units=144, upc=4, 36 cores): max load 328 width-rows,
    //     imbalance 1.17, scalar only 8.2 -> 9.8us, bursts stay 2 rows wide
    //     -> wide-row workload 37.15 -> 35.98us median (A-B-A, 11 vs 6 isolated reps, ranges disjoint).
    // So: walk rowsPerChunk up from 1 until upc <= UPC_CAP, i.e. take the finest chunking
    // whose per-unit fixed cost (GetDataPtr + descriptor setup, ~0.6us/unit measured)
    // still amortises. UPC_CAP=4 is the measured optimum for this trade.
    // Gate: route==4 && outer<maxCores. tiny workload (outer=128) and large-strided workload (outer=1077) are route 4
    // but fail the outer gate; fused-axis workload is route 1 and large-stream workload is route 6 -> all four keep
    // BYTE-IDENTICAL tiling (verified by dumping every tiling field before/after).
    if (route == 4 && outer < maxCores && numActive > 0) {
      const uint32_t UPC_CAP = 4;
      for (uint32_t rpc = 1; rpc <= outer; rpc++) {
        uint32_t cand = (outer + rpc - 1) / rpc;          // nrc for this rowsPerChunk
        uint32_t un = numActive * cand;
        uint32_t c0 = un < maxCores ? un : maxCores;
        if (c0 < 1) c0 = 1;
        uint32_t up = (un + c0 - 1) / c0;
        if (up <= UPC_CAP || rpc == outer) { if (cand > nrc) nrc = cand; break; }
      }
    }
    nRowChunks = nrc;
    rowsPerChunk = (outer + nrc - 1) / nrc;
    uint32_t units = numActive * nrc;
    uint32_t cc = units < maxCores ? units : maxCores;
    if (cc < 1) cc = 1;
    uint32_t upc = (units + cc - 1) / cc;
    if (upc < 1) upc = 1;
    cc = (units + upc - 1) / upc;
    if (cc < 1) cc = 1;
    totalRuns = units;        // route 4 reuse: units
    runsPerCore = upc;        // route 4 reuse: unitsPerCore
    coreNum = cc;
  }

  uint32_t gcX = 0, gcWg = 0, gcStg = 0, gcChk = 0, gcTbl = 0;

  // Tall-band rewrite: make the input transaction tall before making the
  // output group wide.  The general form uses R=16 and ten balanced
  // ~0.9K-column groups on 40 cores.  Exact C5 can instead exchange source
  // double buffering for a 32-row source transaction and two 8-row output
  // buffers.  Both sides then issue half as many DMA commands while the total
  // UB footprint remains below 192 KiB.
  const uint64_t outTotalBytesTB = (uint64_t)outer * sumW * inner * (uint64_t)dtypeSize;
  bool smallPretable = false;
  bool mr4TallBand = false;
  uint32_t pretableBase[MAX_T] = {0};
  uint32_t pretableInc[MAX_T] = {0};
  // Small fragmented last-axis workloads (many narrow inputs, a few hundred KB at most) spend
  // their time issuing one DMA pair per input and row band.  The tall-band kernel instead reads
  // each input once per band, assembles whole rows with Gather and writes whole rows, on only as
  // many cores as the bytes need.
  const bool smallFragmented = (route == 4 || route == 5) && dtypeSize >= 2 && inner == 1 &&
                               numActive >= 8 && outer >= 32 &&
                               outTotalBytesTB <= (256ULL << 10);
  bool tallBand = ((route == 5 && dtypeSize == 4 && outer >= 1024 && sumW >= 4096 &&
                    numActive >= 64) ||
                   smallFragmented) &&
                  inner == 1 && numActive <= MAX_T && ((outer + 15u) / 16u) < 8192u &&
                  maxCores >= 12;
  if (tallBand) {
    // Rb0 (rows per source band) is decided below from the UB budget, not from
    // the shape.  The group layout does not depend on it -- group boundaries
    // come from groupCap on accumulated width -- and srcElems scales exactly 2x
    // from 16 to 32 rows because AlignUp(16*w, 8) == 16*w.  So lay the groups
    // out once at 16 rows and choose afterwards.
    uint32_t Rb0 = 16u;
    constexpr uint32_t groupCap = 929;
    constexpr uint64_t ubBudget = 192ULL * 1024;
    const uint32_t e32 = 32u / (uint32_t)dtypeSize;  // elements per 32B
    const uint64_t tablePadBytes = 512ULL;
    uint32_t groupStart[64] = {0};
    uint32_t groupEnd[64] = {0};
    uint32_t groupWidth[64] = {0};
    uint32_t groupCoreCount[64] = {0};
    uint32_t numGroups = 0;
    uint32_t gStart = 0;
    uint32_t acc = 0;
    uint32_t maxGroupW = 0;
    uint32_t maxSrc = 0;
    uint32_t maxMembers = 0;
    bool ok = true;
    for (uint32_t k = 0; k < numActive; k++) {
      if (acc > 0 && acc + widths[k] > groupCap) {
        if (numGroups >= 64) { ok = false; break; }
        groupStart[numGroups] = gStart;
        groupEnd[numGroups] = k;
        groupWidth[numGroups] = acc;
        uint32_t srcElems = 0;
        for (uint32_t m = gStart; m < k; m++) {
          uint32_t block = Rb0 * widths[m];
          srcElems += ((block + e32 - 1) / e32) * e32;
        }
        if (acc > maxGroupW) maxGroupW = acc;
        if (srcElems > maxSrc) maxSrc = srcElems;
        uint32_t members = k - gStart;
        if (members > maxMembers) maxMembers = members;
        numGroups++;
        gStart = k;
        acc = 0;
      }
      acc += widths[k];
    }
    if (ok && gStart < numActive) {
      if (numGroups >= 64) {
        ok = false;
      } else {
        groupStart[numGroups] = gStart;
        groupEnd[numGroups] = numActive;
        groupWidth[numGroups] = acc;
        uint32_t srcElems = 0;
        for (uint32_t m = gStart; m < numActive; m++) {
          uint32_t block = Rb0 * widths[m];
          srcElems += ((block + e32 - 1) / e32) * e32;
        }
        if (acc > maxGroupW) maxGroupW = acc;
        if (srcElems > maxSrc) maxSrc = srcElems;
        uint32_t members = numActive - gStart;
        if (members > maxMembers) maxMembers = members;
        numGroups++;
      }
    }
    uint32_t pitch = ((maxGroupW + e32 - 1) / e32) * e32;
    uint32_t tableCap = ((maxGroupW + 63) / 64) * 64;
    // The wide pipeline trades source double buffering for a 32-row source
    // transaction plus two 8-row output tiles, so both sides issue half as many
    // DMA commands.  Its source footprint is identical (one buffer of 2*maxSrc
    // against two of maxSrc) and it costs exactly 16*pitch bytes more of output
    // staging, so the only question is whether that still fits UB.  Take it
    // when it does -- no shape literal.
    uint64_t ubCommon = 3ULL * tableCap * 4ULL + tablePadBytes;
    uint64_t ubSrc = 2ULL * (uint64_t)maxSrc * (uint64_t)dtypeSize;  // identical either way
    uint64_t ubWide = ubSrc + 2ULL * 8ULL * pitch * (uint64_t)dtypeSize + ubCommon;
    uint64_t ubNarrow = ubSrc + 3ULL * 4ULL * pitch * (uint64_t)dtypeSize + ubCommon;
    // Small plans stay on 16-row bands: more, lighter cores measured faster than 32-row bands.
    const bool wideReadPipeline = ubWide <= ubBudget && !smallFragmented;
    uint64_t ubNeed = wideReadPipeline ? ubWide : ubNarrow;
    // Key 13: two 4-row tiles instead of two 8-row tiles free 8 rows of pitch, which hold base,
    // inc and the 4-row index and increment tables beside the same 32-row source band.
    {
      const uint64_t rowB = (uint64_t)pitch * 4ULL;
      // base at 256 B, then cur4 / inc / inc4, each placed within 512 B of the previous end.
      const uint64_t mr4Tables = ((rowB + 256ULL + 511ULL) / 512ULL) * 512ULL + 4ULL * rowB + 512ULL +
                                 rowB + 512ULL + 4ULL * rowB;
      const uint64_t ubMR4 = ubSrc + 2ULL * 4ULL * pitch * (uint64_t)dtypeSize + mr4Tables;
      mr4TallBand = wideReadPipeline && ubMR4 <= ubBudget;
    }
    if (wideReadPipeline) {
      Rb0 = 32u;
      // maxSrc was measured at 16 rows and is handed to the kernel as the source
      // buffer capacity (gcStg), so it has to follow Rb0.  Every group's srcElems
      // is exactly 2x at 32 rows because AlignUp(32*w, 8) == 32*w, so the max is
      // too.  Missing this under-allocates the kernel's source queue by half.
      maxSrc *= 2u;
    }
    uint32_t targetCores = maxCores < 40 ? maxCores : 40;
    uint32_t planBase = numActive + numGroups;
    if (!ok || numGroups == 0 || targetCores < numGroups || maxMembers > 32 ||
        planBase + targetCores > MAX_T || ubNeed > ubBudget) {
      tallBand = false;
    } else {
      // Allocate cores by repeatedly splitting the group with the largest
      // current band-load.  Ten near-equal groups deterministically receive
      // four cores each on 910B4.
      uint32_t totalBands = (outer + Rb0 - 1) / Rb0;
      if (smallFragmented) {
        // 8 KB of output per core, never more cores than bands.
        uint64_t byteCores = (outTotalBytesTB + (8ULL << 10) - 1) / (8ULL << 10);
        uint32_t cap = byteCores < totalBands ? (uint32_t)byteCores : totalBands;
        if (cap < numGroups) cap = numGroups;
        if (cap < targetCores) targetCores = cap;
      }
      for (uint32_t g = 0; g < numGroups; g++) groupCoreCount[g] = 1;
      uint32_t usedCores = numGroups;
      while (usedCores < targetCores) {
        uint32_t bestGroup = 0;
        uint64_t bestLoad = 0;
        for (uint32_t g = 0; g < numGroups; g++) {
          uint32_t bands = (totalBands + groupCoreCount[g] - 1) / groupCoreCount[g];
          uint64_t load = (uint64_t)groupWidth[g] * bands;
          if (load > bestLoad) { bestLoad = load; bestGroup = g; }
        }
        groupCoreCount[bestGroup]++;
        usedCores++;
      }
      uint32_t packedPrefix = 0;
      for (uint32_t k = 0; k < numActive; k++) {
        slotElem[k] = packedPrefix;
        uint32_t block = Rb0 * widths[k];
        packedPrefix += ((block + e32 - 1) / e32) * e32;
      }
      for (uint32_t g = 0; g < numGroups; g++) {
        slotElem[numActive + g] = groupStart[g] | (groupEnd[g] << 9);
      }
      // A single small group whose Gather tables fit the tiling arrays gets them precomputed:
      // row-0 source byte offset per output column, and the per-row increment.
      smallPretable = smallFragmented && numGroups == 1u && tableCap <= MAX_T;
      if (smallPretable) {
        const uint32_t elemBytes = (uint32_t)dtypeSize;
        for (uint32_t k = 0; k < numActive; k++) {
          for (uint32_t c = offsets[k]; c < offsets[k] + widths[k]; c++) {
            pretableBase[c] = (slotElem[k] + (c - offsets[k])) * elemBytes;
            pretableInc[c] = widths[k] * elemBytes;
          }
        }
      }
      uint32_t block = 0;
      for (uint32_t g = 0; g < numGroups; g++) {
        uint32_t count = groupCoreCount[g];
        for (uint32_t ordinal = 0; ordinal < count; ordinal++) {
          uint32_t bandStart = totalBands * ordinal / count;
          uint32_t bandEnd = totalBands * (ordinal + 1) / count;
          uint32_t bandCount = bandEnd - bandStart;
          slotElem[planBase + block] = g | (bandStart << 6) | (bandCount << 19);
          block++;
        }
      }
      route = 10;
      coreNum = targetCores;
      totalRuns = targetCores;
      runsPerCore = 1;
      rowsPerChunk = Rb0;
      nRowChunks = totalBands;
      gcX = Rb0;
      gcWg = numGroups;
      gcStg = maxSrc;
      gcChk = maxGroupW;
      gcTbl = groupCap | (1u << 30);  // exact group metadata is present
    }
  }

  // Route 6 (grouped-Gather 4KB-burst store): STRICTLY upgrades route-5 cases (the large-stream workload
  // class: inner==1, non-aligned, dtypeSize>=2, big). Route 5's store is 256B-fragmented
  // (~52% write eff); route 6 groups adjacent segments into ~4KB column-groups, Gather-
  // assembles a [Rb,groupW] tile in UB, and writes each row as one contiguous PadOut.
  // Gated on route==5 so it never touches tiny workload/wide-row workload/fused-axis workload/large-strided workload (they never reach route 5) and
  // int8 (route 4). No per-group tiling arrays: the kernel replays the WG-accumulation
  // scan (numGroups<=64, WG, maxGroupW, maxSrcElems, gR6 are the only tiling params).
  if (route == 5) {
    uint32_t e32 = (uint32_t)(32 / dtypeSize);
    if (e32 < 1) e32 = 1;
    uint32_t Rb0 = 8;                             // rows per gather band (gR6)
    // Three Gather tables are phase-shifted by 256B in UB. Account for the two
    // gaps here so route selection and kernel allocation stay byte-identical.
    const uint64_t tablePadBytes = 512ULL;
    // Pick the group-width threshold WG that maximizes core utilization (units =
    // numGroups*floor(maxCores/numGroups)) among UB-FITTING candidates; since route 6 is
    // READ-bound, spreading over all cores matters more than burst width, so tie-break
    // toward the WIDEST WG (best write efficiency, but must still fit UB). Search from
    // ~8KB bursts down; the widest WG hitting max-util-and-fits wins.
    const uint64_t UB_BUDGET = 176ULL * 1024;
    uint32_t WG = (uint32_t)(4096 / dtypeSize);
    if (WG < 1) WG = 1;
    {
      uint32_t WGmax = (uint32_t)(8192 / dtypeSize);
      if (WGmax > sumW) WGmax = sumW;
      if (WGmax < 1) WGmax = 1;
      uint32_t bestUnits = 0;
      bool found = false;
      for (uint32_t cand = WGmax; cand >= e32; cand -= e32) {
        uint32_t gg = 0, a = 0, mGW = 0, mSrc = 0, gS = 0;
        bool bad = false;
        for (uint32_t k = 0; k < numActive; k++) {
          a += widths[k];
          if (a >= cand || k == numActive - 1) {
            if (gg >= 64) { bad = true; break; }
            uint32_t s = 0;
            for (uint32_t m = gS; m <= k; m++) {
              uint32_t blk = Rb0 * widths[m];
              blk = ((blk + e32 - 1) / e32) * e32;
              s += blk;
            }
            if (a > mGW) mGW = a;
            if (s > mSrc) mSrc = s;
            gg++; gS = k + 1; a = 0;
          }
        }
        if (bad || gg < 1 || gg > 64) continue;
        uint32_t pitch = ((mGW + e32 - 1) / e32) * e32;
        uint32_t tblCap = ((mGW + 63) / 64) * 64;
        uint64_t ubNeed = 2ULL * mSrc * (uint64_t)dtypeSize
                        + 2ULL * Rb0 * pitch * (uint64_t)dtypeSize
                        + 3ULL * tblCap * 4ULL + tablePadBytes;
        if (ubNeed > UB_BUDGET) continue;         // reject; narrower WG will fit
        uint32_t nrc = maxCores / gg; if (nrc < 1) nrc = 1;
        if (nrc > outer) nrc = outer; if (nrc < 1) nrc = 1;
        uint32_t units = gg * nrc;
        if (units > maxCores) units = maxCores;
        if (units > bestUnits) { bestUnits = units; WG = cand; found = true; }  // widest WG at max-util
      }
      (void)found;
    }
    // CORE BALANCE (route 6). The legacy scan closes a group as soon as acc >= WG, so every
    // group OVERSHOOTS to ~WG+w and the TRAILING group is only the leftover: measured large-stream workload
    // widths [1212,1245,1207,1211,1279,1237,1241,*179*]. With units = numGroups*nrc == maxCores
    // the tiling has upc == 1, so the 5 cores that draw the 179-wide runt finish at ~58us while
    // the widest group's cores run ~187us (msprof op PipeUtilization block_id rows: max/mean
    // 1.327, max/min 3.29, 40 blocks, 0 idle -> ~25% of wall lost).
    // Fix WITHOUT raising the unit count (raising it multiplies route 6's per-unit cost: a
    // 135-entry group scan + 2*Wg SetValue offset-table fill + nk GetDataPtr per unit): keep the
    // group COUNT that the WG search already chose (so the ~4-5KB write burst that route 6 was
    // built for is untouched) and re-partition those same G groups with a CAPACITY-greedy scan,
    // i.e. the minimal cap TH whose greedy still fits in G groups. That is the exact min-max
    // contiguous partition -> large-stream workload widths become [1043..1137], max group 1279 -> 1137 (-11.1%),
    // and ubNeed 174.9KB -> 156.1KB. Per-unit cost is UNCHANGED (same scan shape, same units).
    uint32_t balMode = 0;
    {
      R6Scan leg = R6Group(widths, numActive, WG, 0u, Rb0, e32);
      if (!leg.bad && leg.gg >= 2 && leg.gg <= 64) {
        uint32_t lo = 0;                     // TH must be >= widest single segment
        for (uint32_t k = 0; k < numActive; k++) if (widths[k] > lo) lo = widths[k];
        uint32_t hi = sumW;
        while (lo < hi) {                    // minimal TH whose capacity-greedy fits in leg.gg
          uint32_t mid = lo + (hi - lo) / 2;
          R6Scan t = R6Group(widths, numActive, mid, 1u, Rb0, e32);
          if (!t.bad && t.gg <= leg.gg) hi = mid; else lo = mid + 1;
        }
        R6Scan bal = R6Group(widths, numActive, lo, 1u, Rb0, e32);
        uint32_t bpitch = ((bal.mGW + e32 - 1) / e32) * e32;
        uint32_t btbl = ((bal.mGW + 63) / 64) * 64;
        uint64_t bub = 2ULL * bal.mSrc * (uint64_t)dtypeSize
                     + 2ULL * Rb0 * bpitch * (uint64_t)dtypeSize
                     + 3ULL * btbl * 4ULL + tablePadBytes;
        if (!bal.bad && bal.gg == leg.gg && bal.mGW <= leg.mGW && bub <= 176ULL * 1024) {
          balMode = 1; WG = lo;              // same group count, tighter max width, smaller UB
        }
      }
    }
    uint32_t g = 0, acc = 0, gStart = 0, maxGW = 0, maxSrc = 0;
    bool ok = true;
    for (uint32_t k = 0; k < numActive; k++) {
      if (balMode == 1u && acc > 0 && acc + widths[k] > WG) {
        if (g >= 64) { ok = false; break; }
        if (numActive + g < MAX_T) {
          slotElem[numActive + g] = gStart | (k << 9);
        }
        uint32_t s = 0;
        for (uint32_t m = gStart; m < k; m++) {
          uint32_t blk = Rb0 * widths[m];
          blk = ((blk + e32 - 1) / e32) * e32;
          s += blk;
        }
        if (acc > maxGW) maxGW = acc;
        if (s > maxSrc) maxSrc = s;
        g++; gStart = k; acc = 0;
      }
      acc += widths[k];
      bool close = (balMode == 0u) ? (acc >= WG || k == numActive - 1) : (k == numActive - 1);
      if (close) {
        if (g >= 64) { ok = false; break; }       // too many groups -> keep route 5
        if (numActive + g < MAX_T) {
          slotElem[numActive + g] = gStart | ((k + 1u) << 9);
        }
        uint32_t s = 0;
        for (uint32_t m = gStart; m <= k; m++) {
          uint32_t blk = Rb0 * widths[m];
          blk = ((blk + e32 - 1) / e32) * e32;    // 32B-align each member's packed src slot
          s += blk;
        }
        if (acc > maxGW) maxGW = acc;
        if (s > maxSrc) maxSrc = s;
        g++; gStart = k + 1; acc = 0;
      }
    }
    if (ok && g >= 1 && g <= 64) {
      uint32_t numGroups = g;
      uint32_t pitch = ((maxGW + e32 - 1) / e32) * e32;   // assembled-tile row stride
      uint32_t tblCap = ((maxGW + 63) / 64) * 64;         // vector table cap
      // UB budget guard: srcQ(DB) + tileQ(DB) + 3 tables. Keep <=176KB (route 5 uses ~160KB).
      uint64_t ubNeed = 2ULL * maxSrc * (uint64_t)dtypeSize
                      + 2ULL * Rb0 * pitch * (uint64_t)dtypeSize
                      + 3ULL * tblCap * 4ULL + tablePadBytes;
      if (ubNeed <= 176ULL * 1024) {
        route = 6;
        // Balance group width against per-unit scalar setup. Aim for two units
        // per core, then prefer a row-chunk containing a whole number of gather
        // bands so a ragged tail does not pay a full descriptor setup.
        uint32_t nrc = numGroups > 0 ? maxCores / numGroups : 1;
        if (nrc < 1) nrc = 1;
        if (nrc > outer) nrc = outer;
        if (nrc < 1) nrc = 1;
        uint32_t baseNrc = nrc;
        {
          uint32_t lo = nrc + 1, hi = nrc * 2;
          if (hi > outer) hi = outer;
          if (hi >= lo) {
            uint32_t pick = hi;                       // fallback: top of the upc==2 window
            for (uint32_t c = hi; c >= lo; c--) {
              if (((outer + c - 1) / c) % Rb0 == 0) { pick = c; break; }
              if (c == lo) break;                     // guard (unsigned)
            }
            nrc = pick;
          }
        }
        // A fragment-heavy stream may profit from three persistent row chunks
        // per core because group pointers and Gather tables remain reusable.
        if (numActive >= 64 && outer >= 1024) {
          uint32_t lo = baseNrc * 2 + 1, hi = baseNrc * 3;
          if (hi > outer) hi = outer;
          for (uint32_t c = lo; c <= hi; c++) {
            if (((outer + c - 1) / c) % Rb0 == 0) { nrc = c; break; }
          }
        }
        nRowChunks = nrc;
        rowsPerChunk = (outer + nrc - 1) / nrc;
        uint32_t units = numGroups * nrc;
        uint32_t cc = units < maxCores ? units : maxCores;
        if (cc < 1) cc = 1;
        uint32_t upc = (units + cc - 1) / cc;
        if (upc < 1) upc = 1;
        cc = (units + upc - 1) / upc;
        if (cc < 1) cc = 1;
        totalRuns = units;
        runsPerCore = upc;
        coreNum = cc;
        gcX = Rb0;         // gR6
        gcWg = numGroups;
        gcStg = maxSrc;    // packed-src cap (elems)
        gcChk = maxGW;     // widest group width (elems) -> kernel derives pitch/tblCap
        uint32_t metaMode = (numActive + numGroups <= MAX_T) ? 1u : 0u;
        gcTbl = WG | (metaMode << 30) | (balMode << 31);
        // When room remains in slotElem[], [numActive, numActive+numGroups)
        // stores packed [groupStart, groupEnd) input indices (9 bits each).
        // The route-6 kernel then avoids replaying the full widths[] scan per unit.
        // Global packed-source prefix. Route 6 subtracts the current group's
        // prefix so members can be loaded in a row-chunk-dependent cyclic order
        // while still landing in the original UB slots expected by Gather.
        uint32_t packedPrefix = 0;
        for (uint32_t k = 0; k < numActive; k++) {
          slotElem[k] = packedPrefix;
          uint32_t blk = Rb0 * widths[k];
          blk = ((blk + e32 - 1) / e32) * e32;
          packedPrefix += blk;
        }
      }
    }
  }

  // Tiny workloads over-shard: cap to roughly 4096 elements/core and keep an
  // even core count. Large streams naturally retain all available cores.
  {
    uint64_t outTotal = (uint64_t)outer * (uint64_t)sumW * (uint64_t)inner;
    uint32_t coreCap = (uint32_t)((outTotal + 4095ULL) / 4096ULL);
    // Fragmented small last-axis work needs slightly more cores than a pure
    // byte-volume estimate because each core also amortizes descriptor setup.
    if (route == 4 && dtypeSize == 2 && inner == 1 && numActive >= 8 &&
        outTotal >= 16384 && outTotal <= 65536) {
      coreCap = (uint32_t)((outTotal + 3071ULL) / 3072ULL);
    }
    if (coreCap & 1u) coreCap += 1;
    if (coreCap < 2) coreCap = 2;
    if (route != 10 && coreCap > 0 && coreCap < coreNum) {
      coreNum = coreCap;
      if (totalRuns > 0) runsPerCore = (totalRuns + coreNum - 1) / coreNum;
      uint32_t pe = (uint32_t)((outTotal + coreNum - 1) / coreNum);
      pe = ((pe + 31u) / 32u) * 32u; if (pe < 32u) pe = 32u;
      perCoreElems = pe;
      if (route == 1) {
        // Route 1 built its boundary table in slotElem[] from the pre-cap
        // split, and dim0Direct decodes each core's pieces out of that table.
        // Leaving it behind hands the kernel the first coreNum boundaries of a
        // finer split, so everything past coreNum*oldPer is simply never
        // written -- silently, and only on workloads small enough to be capped.
        coreNum = (uint32_t)((outTotal + pe - 1) / pe);
        if (coreNum < 1) coreNum = 1;
        if (totalRuns > 0) runsPerCore = (totalRuns + coreNum - 1) / coreNum;
        for (uint32_t b = 0; b <= coreNum; b++) {
          uint64_t boundary = (uint64_t)b * pe;
          if (boundary > outTotal) boundary = outTotal;
          slotElem[b] = (uint32_t)boundary;
        }
      }
    }
  }
  // Dim-0 concat is already a sequence of contiguous input streams.
  // The generic route-1 kernel gives every core an output interval, but then
  // scans all inputs on every core to discover the one-to-three intersecting
  // pieces. Materialize those pieces on the host instead: three packed slots
  // per core are enough for the eligible geometry, so the device path performs no
  // input-table scan and no interval arithmetic before issuing the large DMA.
  bool dim0Direct = route == 1 && dtypeSize == 1 && outer == 1 &&
                    numActive <= 64 && coreNum * 3u <= MAX_T;
  if (dim0Direct) {
    // Snap a split to a nearby input boundary when the load displacement stays
    // below 1/16 of the ideal core span.  This reduces cross-input pieces while
    // preserving balance, and applies to any dim-0 segment distribution.
    uint32_t oldBoundary[64] = {0};
    uint64_t ideal = ((uint64_t)sumW * inner + coreNum - 1u) / coreNum;
    for (uint32_t b = 0; b <= coreNum; ++b) oldBoundary[b] = slotElem[b];
    for (uint32_t b = 1; b < coreNum; ++b) {
      uint64_t cur = slotElem[b];
      uint64_t best = cur;
      uint64_t bestDist = ideal;
      for (uint32_t k = 1; k < numActive; ++k) {
        uint64_t edge = (uint64_t)offsets[k] * inner;
        uint64_t dist = edge > cur ? edge - cur : cur - edge;
        if (dist < bestDist && dist * 16u <= ideal) {
          best = edge;
          bestDist = dist;
        }
      }
      slotElem[b] = (uint32_t)best;
    }
    uint32_t maxPieces = 0;
    for (uint32_t b = 0; b < coreNum; ++b) {
      uint32_t pieces = 0;
      for (uint32_t k = 0; k < numActive; ++k) {
        uint64_t ib = (uint64_t)offsets[k] * inner;
        uint64_t ie = ib + (uint64_t)widths[k] * inner;
        if ((uint64_t)slotElem[b] < ie && (uint64_t)slotElem[b + 1] > ib) ++pieces;
      }
      if (pieces > maxPieces) maxPieces = pieces;
    }
    if (maxPieces > 3u) {
      dim0Direct = false;
      for (uint32_t b = 0; b <= coreNum; ++b) slotElem[b] = oldBoundary[b];
    }
  }
  if (dim0Direct) {
    for (uint32_t b = 0; b < coreNum; b++) {
      uint64_t lo = slotElem[b];
      uint64_t hi = slotElem[b + 1];
      uint32_t piece = 0;
      for (uint32_t k = 0; k < numActive && piece < 3; k++) {
        uint64_t inputBegin = (uint64_t)offsets[k] * inner;
        uint64_t inputLen = (uint64_t)widths[k] * inner;
        uint64_t pieceBegin = lo > inputBegin ? lo : inputBegin;
        uint64_t inputEnd = inputBegin + inputLen;
        uint64_t pieceEnd = hi < inputEnd ? hi : inputEnd;
        if (pieceBegin >= pieceEnd) continue;
        uint32_t j = b * 3u + piece;
        directOrigIdx[j] = origIdx[k];
        directSrcOff[j] = (uint32_t)(pieceBegin - inputBegin);
        directDstOff[j] = (uint32_t)pieceBegin;
        directLen[j] = (uint32_t)(pieceEnd - pieceBegin);
        piece++;
      }
    }
  }

  // Pre-decode three complete (input,row-band) units per core so
  // the tiny device path has no division/modulo and copies no nine-entry
  // metadata tables before issuing its three strided DMA pairs.
  uint32_t maxSegElems = 0;
  for (uint32_t k = 0; k < numActive; ++k) {
    uint32_t seg = widths[k] * inner;
    if (seg > maxSegElems) maxSegElems = seg;
  }
  uint32_t directAlign = 32u / dtypeSize;
  uint32_t maxSegPitch = ((maxSegElems + directAlign - 1u) / directAlign) * directAlign;
  bool predecoded3 = route == 4 && runsPerCore == 3 &&
                     totalRuns == coreNum * 3u && outer % rowsPerChunk == 0 &&
                     (uint64_t)maxSegPitch * rowsPerChunk <= 4096u &&
                     coreNum * 3u <= MAX_T;
  if (predecoded3) {
    for (uint32_t b = 0; b < coreNum; b++) {
      for (uint32_t j = 0; j < 3; j++) {
        uint32_t u = b * 3u + j;
        uint32_t k = u % numActive;
        uint32_t rowChunk = u / numActive;
        uint32_t r0 = rowChunk * rowsPerChunk;
        uint32_t q = b * 3u + j;
        uint32_t segLen = widths[k] * inner;
        directOrigIdx[q] = origIdx[k];
        directSrcOff[q] = r0 * segLen;
        directDstOff[q] = r0 * sumW * inner + offsets[k] * inner;
        directLen[q] = segLen;
      }
    }
  }

  // Pre-decode four two-row units per core and whether both rows fit in one
  // single UB transaction. The device path then has no 16-entry metadata copy,
  // modulo/division, row bounds, pitch division, or dynamic run loop.
  bool predecoded4 = route == 4 && runsPerCore == 4 && rowsPerChunk == 2 &&
                     outer % 2u == 0u && totalRuns == coreNum * 4u &&
                     maxSegElems < (1u << 24) && coreNum * 4u <= MAX_T;
  if (predecoded4) {
    const uint32_t e32 = 8;  // fp32 elements per 32B
    const uint32_t rowW = sumW * inner;
    for (uint32_t b = 0; b < coreNum; b++) {
      for (uint32_t j = 0; j < 4; j++) {
        uint32_t u = b * 4u + j;
        uint32_t k = u % numActive;
        uint32_t rowChunk = u / numActive;
        uint32_t r0 = rowChunk * rowsPerChunk;
        uint32_t q = b * 4u + j;
        uint32_t segLen = widths[k] * inner;
        uint32_t pitch = ((segLen + e32 - 1) / e32) * e32;
        uint32_t rowBatch = pitch * 2u <= TILE_ELEMS ? 2u : 1u;
        directOrigIdx[q] = origIdx[k];
        directSrcOff[q] = r0 * segLen;
        directDstOff[q] = r0 * rowW + offsets[k] * inner;
        directLen[q] = segLen | (rowBatch << 24);
      }
    }
  }

  // A large one-pass source stream should not allocate in L2, while the output
  // keeps normal L2 behavior
  // so partial row writes from different source runs can still be combined.
  uint64_t streamBytes = (uint64_t)outer * sumW * inner * dtypeSize;
  bool streamL2Bypass = route == 4 && streamBytes >= (8ULL << 20) &&
                        coreNum == maxCores;

  uint64_t tilingKey = TILING_KEY_GENERAL;
  if (predecoded3) tilingKey = TILING_KEY_PREDECODED3;
  else if (predecoded4) tilingKey = TILING_KEY_PREDECODED4;
  else if (dim0Direct) tilingKey = TILING_KEY_DIM0_DIRECT;
  else if (route == 1) tilingKey = TILING_KEY_DIM0_SPLIT;
  else if (streamL2Bypass) tilingKey = TILING_KEY_STREAM_L2_BYPASS;
  else if (route == 4) tilingKey = TILING_KEY_UNIVERSAL;
  else if (route == 6) tilingKey = TILING_KEY_GROUPED_GATHER;
  else if (route == 10)
    tilingKey = smallPretable ? TILING_KEY_TALL_BAND_SMALL
                              : (mr4TallBand ? TILING_KEY_TALL_BAND_MR4 : TILING_KEY_TALL_BAND);
  context->SetTilingKey(tilingKey);
  context->SetBlockDim(coreNum);
  tiling.set_numActive(numActive);
  tiling.set_outer(outer);
  tiling.set_inner(inner);
  tiling.set_sumW(sumW);
  // dtype-adaptive tile: small dtypes (int8/fp16) get a bigger element tile so a
  // contiguous copy uses fewer, larger chunks -> fewer EnQue/DeQue syncs (route 1
  // int8 axis0 was scalar-stall-bound on per-chunk sync). UB stays ~128KB (tileLen
  // *sizeof(T)*2): int8=65536, fp16=32768, fp32/int32=16384 (unchanged).
  // large-strided workload-only large route-4 gate: use the otherwise idle UB headroom to batch
  // more strided rows per queue transaction. Two 92 KiB queue buffers consume
  // 184 KiB, matching the established per-core guard while leaving tiny workload/wide-row workload and
  // all non-route-4 tiling byte-identical to the route4-split control.
  uint32_t tileBytes = 65536u;
  if (route == 4 && outer >= maxCores &&
      (uint64_t)outer * outRowBytes >= (8ULL << 20)) {
    // large-strided workload is MTE2-bound after input L2 bypass.  Use the full 192 KiB UB as two
    // 96 KiB ping-pong tiles; 92--95 KiB leave bandwidth on the table.
    tileBytes = 96u * 1024u;
  }
  uint32_t tileLenAdaptive = (uint32_t)(tileBytes / dtypeSize);
  if (tileLenAdaptive < TILE_ELEMS) tileLenAdaptive = TILE_ELEMS;
  tiling.set_tileLen(tileLenAdaptive);
  tiling.set_totalRuns(totalRuns);
  tiling.set_runsPerCore(runsPerCore);
  tiling.set_route(route);
  tiling.set_perCoreElems(perCoreElems);
  tiling.set_totalElems(totalElems);
  tiling.set_nRowChunks(nRowChunks);
  tiling.set_rowsPerChunk(rowsPerChunk);
  tiling.set_gR(gR);
  tiling.set_gCoreRows(gCoreRows);
  tiling.set_gUbSrcElems(gUbSrcElems);
  tiling.set_gcX(gcX);
  tiling.set_gcWg(gcWg);
  tiling.set_gcStg(gcStg);
  tiling.set_gcChk(gcChk);
  tiling.set_gcTbl(gcTbl);
  if (dim0Direct || predecoded3 || predecoded4) {
    tiling.set_origIdx(directOrigIdx);
    tiling.set_widths(directSrcOff);
    tiling.set_offsets(directDstOff);
    tiling.set_slotElem(directLen);
  } else if (tilingKey == TILING_KEY_TALL_BAND_SMALL) {
    // Key 12: packed members as for key 10; offsets[]/widths[] carry the Gather tables.
    uint32_t packedIdx[MAX_T] = {0};
    for (uint32_t k = 0; k < numActive; k++) packedIdx[k] = origIdx[k] | (widths[k] << 9);
    tiling.set_origIdx(packedIdx);
    tiling.set_widths(pretableInc);
    tiling.set_offsets(pretableBase);
    tiling.set_slotElem(slotElem);
  } else if (tilingKey == TILING_KEY_TALL_BAND || tilingKey == TILING_KEY_TALL_BAND_MR4) {
    // Key 10 reads one GM word per group member instead of copying the tiling
    // struct: origIdx in the low 9 bits, the width above them.  The tall-band UB
    // gate bounds every width far below 2^23 (two 16-row source blocks of one
    // member must fit in 192 KiB), and origIdx < MAX_T = 256.
    uint32_t packedIdx[MAX_T] = {0};
    for (uint32_t k = 0; k < numActive; k++) packedIdx[k] = origIdx[k] | (widths[k] << 9);
    tiling.set_origIdx(packedIdx);
    tiling.set_widths(widths);
    tiling.set_offsets(offsets);
    tiling.set_slotElem(slotElem);
  } else {
    tiling.set_origIdx(origIdx);
    tiling.set_widths(widths);
    tiling.set_offsets(offsets);
    tiling.set_slotElem(slotElem);
  }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Concat : public OpDef {
 public:
  explicit Concat(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("concat_dim").Int();

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Concat);
}  // namespace ops
