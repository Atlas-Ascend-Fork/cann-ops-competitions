#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numActive);
  TILING_DATA_FIELD_DEF(uint32_t, outer);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
  TILING_DATA_FIELD_DEF(uint32_t, sumW);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
  TILING_DATA_FIELD_DEF(uint32_t, totalRuns);
  TILING_DATA_FIELD_DEF(uint32_t, runsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, route);         // 0=general(v1), 1=dim0 split-core(P), 2=aligned strided collapse(S)
  TILING_DATA_FIELD_DEF(uint32_t, perCoreElems);  // Route P: output elems per core
  TILING_DATA_FIELD_DEF(uint32_t, totalElems);    // Route P: total output elems (dim0: sumW*inner)
  TILING_DATA_FIELD_DEF(uint32_t, nRowChunks);    // Route S: how many row-chunks each input is split into
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerChunk);  // Route S: rows per chunk
  TILING_DATA_FIELD_DEF(uint32_t, gR);            // Route G: rows per gather-batch
  TILING_DATA_FIELD_DEF(uint32_t, gCoreRows);    // Route G: output rows per core
  TILING_DATA_FIELD_DEF(uint32_t, gUbSrcElems);  // Route G: ubSrc capacity (elems)
  TILING_DATA_FIELD_DEF(uint32_t, gcX);          // Route 6: gR6 = rows per assembled gather band
  TILING_DATA_FIELD_DEF(uint32_t, gcWg);         // Route 6: numGroups (column-group count)
  TILING_DATA_FIELD_DEF(uint32_t, gcStg);        // Route 6: maxSrcElems = packed-src buffer cap (elems)
  TILING_DATA_FIELD_DEF(uint32_t, gcChk);        // Route 6: maxGroupW = widest group width (elems)
  TILING_DATA_FIELD_DEF(uint32_t, gcTbl);        // Route 6: WG = grouping width threshold (kernel replays scan)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, origIdx);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, slotElem);  // Route G: input k block offset in ubSrc (elems)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, widths);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, offsets);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling
#endif  // CONCAT_TILING_H
