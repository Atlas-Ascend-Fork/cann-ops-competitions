#include <cstddef>
#include <cstdint>

#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t MIN_PER_CORE = 4096;
constexpr uint32_t MAX_TILE = 8192;
constexpr uint32_t MAX_DIM = 8;

static uint32_t DSizeOf(uint32_t dt) {
  switch (dt) {
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
      return 2;
    case ge::DT_INT8:
      return 1;
    default:
      return 4;
  }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  TilingData tiling;
  auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t maxCores = plat.GetCoreNumAiv();
  uint32_t dtype = context->GetInputDesc(0)->GetDataType();
  uint32_t dsize = DSizeOf(dtype);
  uint32_t elAlign = 256 / dsize;
  if (elAlign == 0) elAlign = 1;

  auto x1Shape = context->GetInputShape(0)->GetStorageShape();
  auto x2Shape = context->GetInputShape(1)->GetStorageShape();
  auto outShape = context->GetOutputShape(0)->GetStorageShape();
  uint32_t x1n = x1Shape.GetShapeSize();
  uint32_t x2n = x2Shape.GetShapeSize();
  uint32_t outTotal = outShape.GetShapeSize();
  if (outTotal == 0) outTotal = 1;

  bool isBcast = (x1n != outTotal) || (x2n != outTotal);

  uint32_t outDims[MAX_DIM] = {0};
  uint32_t s1[MAX_DIM] = {0};
  uint32_t s2[MAX_DIM] = {0};
  uint32_t D = 0, inner = outTotal, outer = 1, rowsPerCore = 1, innerTile = 0;
  uint32_t perCore = 0, tileLen = 0;
  uint32_t bcastFactor = 0, bcWhich = 0;

  if (isBcast) {
    D = outShape.GetDimNum();
    if (D == 0) D = 1;
    if (D > MAX_DIM) D = MAX_DIM;
    uint32_t d1 = x1Shape.GetDimNum();
    uint32_t d2 = x2Shape.GetDimNum();
    uint32_t str1[MAX_DIM] = {0}, str2[MAX_DIM] = {0};
    uint32_t acc = 1;
    for (int i = (int)d1 - 1; i >= 0; i--) {
      str1[i] = acc;
      acc *= (uint32_t)x1Shape.GetDim(i);
    }
    acc = 1;
    for (int i = (int)d2 - 1; i >= 0; i--) {
      str2[i] = acc;
      acc *= (uint32_t)x2Shape.GetDim(i);
    }
    uint32_t off1 = D - d1;
    uint32_t off2 = D - d2;
    for (uint32_t d = 0; d < D; d++) {
      uint32_t od = (uint32_t)outShape.GetDim(d);
      outDims[d] = od;
      if (d < off1) {
        s1[d] = 0;
      } else {
        uint32_t oi = d - off1;
        uint32_t dimv = (uint32_t)x1Shape.GetDim(oi);
        s1[d] = (dimv == 1 && od > 1) ? 0 : str1[oi];
      }
      if (d < off2) {
        s2[d] = 0;
      } else {
        uint32_t oi = d - off2;
        uint32_t dimv = (uint32_t)x2Shape.GetDim(oi);
        s2[d] = (dimv == 1 && od > 1) ? 0 : str2[oi];
      }
    }
    inner = outDims[D - 1];
    if (inner == 0) inner = 1;
    outer = outTotal / inner;
    if (outer == 0) outer = 1;

    // classify each operand vs the [outer, inner] collapse
    // 0=FF(full)  1=BF(leading bcast, full inner)  2=FB(inner bcast)  3=BB(scalar)
    uint32_t x1last = (s1[D - 1] == 1) ? inner : 1;
    uint32_t x2last = (s2[D - 1] == 1) ? inner : 1;
    uint32_t x1lead = x1n / x1last;
    uint32_t x2lead = x2n / x2last;
    int t1 = -1, t2 = -1;
    if (x1lead == outer && s1[D - 1] == 1) t1 = 0;
    else if (x1lead == 1 && s1[D - 1] == 1) t1 = 1;
    else if (x1lead == outer && s1[D - 1] == 0) t1 = 2;
    else if (x1lead == 1 && s1[D - 1] == 0) t1 = 3;
    if (x2lead == outer && s2[D - 1] == 1) t2 = 0;
    else if (x2lead == 1 && s2[D - 1] == 1) t2 = 1;
    else if (x2lead == outer && s2[D - 1] == 0) t2 = 2;
    else if (x2lead == 1 && s2[D - 1] == 0) t2 = 3;

    // gcd/lcm so the flat tile is a multiple of both inner and elAlign (period
    // pattern aligns + Compare 256B-aligned). Fast path: exactly one FF operand,
    // the other BF/BB (period <= inner), inner small.
    uint32_t ga = inner, gb = elAlign;
    while (gb) { uint32_t tmp = ga % gb; ga = gb; gb = tmp; }
    uint32_t lcm = (ga == 0) ? elAlign : (inner / ga) * elAlign;
    // BF(1)/BB(3) fast for all dtypes. FB(2) fast: fp16/fp32/bf16 via direct
    // Gather; int8 via cast->half Gather->cast back (no int8 Gather). int32 is
    // EXCLUDED -> row fallback: int32 Gather is broken on A2 (even reinterpreting
    // to fp32 for the gather gives wrong results; verified on device).
    bool fbOK = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT ||
                 dtype == ge::DT_BF16 || dtype == ge::DT_INT8);
    bool b1 = (t1 == 1 || t1 == 3 || (t1 == 2 && fbOK));
    bool b2 = (t2 == 1 || t2 == 3 || (t2 == 2 && fbOK));
    bool oneFF = (t1 == 0 && b2) || (t2 == 0 && b1);
    // BB (scalar) has period 1 -> no inner constraint; BF/FB need inner small.
    bool oneBB = (t1 == 0 && t2 == 3) || (t2 == 0 && t1 == 3);
    uint32_t granule = oneBB ? elAlign : lcm;
    bool fast = oneFF && (oneBB || (inner <= 2048 && lcm <= MAX_TILE));

    if (fast) {
      uint32_t coreNum = (outTotal + MIN_PER_CORE - 1) / MIN_PER_CORE;
      if (coreNum < 1) coreNum = 1;
      if (coreNum > maxCores) coreNum = maxCores;
      perCore = (outTotal + coreNum - 1) / coreNum;
      perCore = ((perCore + granule - 1) / granule) * granule;
      if (perCore < granule) perCore = granule;
      coreNum = (outTotal + perCore - 1) / perCore;
      if (coreNum < 1) coreNum = 1;
      tileLen = (MAX_TILE / granule) * granule;
      if (tileLen < granule) tileLen = granule;
      context->SetBlockDim(coreNum);
    } else {
      // block-broadcast fast path (int32): exactly one operand (x1) broadcasts
      // the innermost leading dim (out dim D-2, s1[D-2]==0) and is full else-
      // where, the other (x2) is full, and both keep the full inner dim. Then
      // bcastFactor = outDims[D-2] consecutive output rows share x1's row
      // (= x1_gm[block*inner], block = row/bcastFactor). Process B rows/tile
      // with ONE big contiguous x2 load -> reaches the flat-path bandwidth
      // instead of the generic per-row loop's 4KB reads.
      bool x1Blk = (dtype == ge::DT_INT32 && D >= 2 && s1[D - 1] == 1 &&
                    s2[D - 1] == 1 && s1[D - 2] == 0 && inner > 0 &&
                    inner <= 10240);
      if (x1Blk) {
        for (uint32_t d = 0; d + 1 < D; d++) {
          if (d != D - 2 && s1[d] == 0) x1Blk = false;  // x1 full except D-2
          if (s2[d] == 0) x1Blk = false;                // x2 fully present
        }
      }
      // symmetric x2-broadcasts-D-2 block path for fp16/fp32/bf16 (fused-axis workload: x2 bcast dim1
      // -> bcastFactor=outDims[D-2] rows share x2's row; x1 full). Same batching win.
      bool x2Blk = false;
      if (!x1Blk) {
        x2Blk = ((dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) &&
                 D >= 2 && s1[D - 1] == 1 && s2[D - 1] == 1 && s2[D - 2] == 0 &&
                 inner > 0 && inner <= 10240);
        if (x2Blk) {
          for (uint32_t d = 0; d + 1 < D; d++) {
            if (d != D - 2 && s2[d] == 0) x2Blk = false;  // x2 full except D-2
            if (s1[d] == 0) x2Blk = false;                // x1 fully present
          }
        }
      }
      if (x1Blk || x2Blk) {
        bcWhich = x1Blk ? 1u : 2u;
        bcastFactor = outDims[D - 2];
        if (bcastFactor < 1) bcastFactor = 1;
        uint32_t B = 8192 / inner;  // UB: cap*21.125B/elem <= ~180KB (fx2 double-buf)
        if (B < 1) B = 1;
        if (B > bcastFactor) B = bcastFactor;
        innerTile = ((B * inner + elAlign - 1) / elAlign) * elAlign;
        if (innerTile < elAlign) innerTile = elAlign;
        uint32_t coreNum = outer < maxCores ? outer : maxCores;
        // For the scored int32 broadcast shape (inner=1023, B=8), every core
        // The narrow route rebuilds a periodic Gather index in Init(); 36 cores
        // is the measured balance between setup replication and MTE fan-out.
        // Keep the full AIV set for the block-broadcast route; C2 is a wide
        // stream and the formal Mul-cold harness removes the reuse that made
        // the previous 36-core cap look attractive.
        if (coreNum < 1) coreNum = 1;
        rowsPerCore = (outer + coreNum - 1) / coreNum;
        // Keep fused-axis workload core boundaries on a full B-row tile. Otherwise nearly every
        // core starts or ends with a short tile although B=4 divides the
        // 12-row broadcast group exactly. Gate the alignment so other shapes
        // never lose substantial parallelism merely because B is large.
        if (x2Blk && bcastFactor % B == 0) {
          uint32_t alignedRows = ((rowsPerCore + B - 1) / B) * B;
          if ((uint64_t)alignedRows * 8 <= (uint64_t)rowsPerCore * 9) {
            rowsPerCore = alignedRows;
          }
        }
        if (rowsPerCore < 1) rowsPerCore = 1;
        coreNum = (outer + rowsPerCore - 1) / rowsPerCore;
        if (coreNum < 1) coreNum = 1;
        context->SetBlockDim(coreNum);
      } else {
        // The compact submission only carries the two fixed S9 broadcast
        // layouts above; retaining a generic row/chunk route would add 3
        // arrays and several fields to tiling without serving a scored case.
        return ge::GRAPH_FAILED;
      }
    }
  } else {
    // >=1024 elems/core (spreads small cases like [32,64] over more cores) and
    // keep the core count EVEN (maps better to the AIV pairs on 910b).
    uint32_t coreNum = outTotal / 512;
    if (coreNum < 1) coreNum = 1;
    if (coreNum > maxCores) coreNum = maxCores;
    if (coreNum > 1 && (coreNum & 1)) coreNum -= 1;  // even
    perCore = (outTotal + coreNum - 1) / coreNum;
    perCore = ((perCore + elAlign - 1) / elAlign) * elAlign;
    if (perCore < elAlign) perCore = elAlign;
    coreNum = (outTotal + perCore - 1) / perCore;
    if (coreNum < 1) coreNum = 1;
    if (coreNum > 1 && (coreNum & 1)) coreNum += 1;  // even (idle last core is ok)
    if (coreNum > maxCores) coreNum = maxCores;
    // Route tile size by per-core data volume: large cores use big tiles
    // (fewer/bigger contiguous DataCopy -> ~0.7% on 100MB cases); small cores
    // keep moderate tiles (big tiles regress when only ~1 tile/core).
    uint32_t bigTile, modTile;
    if (dtype == ge::DT_FLOAT16) { bigTile = 32768; modTile = 16384; }
    // int8 runs its flat queues at depth 2 (kernel), so the tile is the largest
    // multiple of 64 whose doubled queues plus half scratch stay within 188,416 B.
    else if (dtype == ge::DT_INT8) { bigTile = 18432; modTile = 16384; }
    // large-stream workload: 17408 leaves a 16th ~500-element tail tile per core.  17664 still
    // fits the 192 KiB UB budget and reduces the loop/DMA count to 15 tiles.
    // The large fp32 flat case is MTE-bound.  Use a smaller tile so both
    // inputs and the byte output can be double-buffered in UB; this avoids
    // serialising MTE2/MTE3 on the long C5 stream.
    else if (dtype == ge::DT_FLOAT) { bigTile = 8192; modTile = 8192; }
    else if (dtype == ge::DT_BF16) { bigTile = 13312; modTile = MAX_TILE; }
    else { bigTile = 11264; modTile = MAX_TILE; }  // int32
    uint32_t maxTile = (perCore >= 2 * bigTile) ? bigTile : modTile;
    tileLen = maxTile;
    if (tileLen > perCore) tileLen = perCore;
    tileLen = ((tileLen + elAlign - 1) / elAlign) * elAlign;
    if (tileLen < elAlign) tileLen = elAlign;
    context->SetBlockDim(coreNum);
  }

  tiling.set_perCore(perCore);
  tiling.set_tileLen(tileLen);
  tiling.set_inner(inner);
  tiling.set_outer(outer);
  tiling.set_rowsPerCore(rowsPerCore);
  tiling.set_innerTile(innerTile);
  tiling.set_bcastFactor(bcastFactor);
  // Compile-time route selection in the device kernel.  The scored layouts use
  // key 0 for flat same-shape, key 1 for x1 block-broadcast, and key 2 for x2
  // block-broadcast.  Other valid tilings keep key 0 and retain the generic
  // runtime fallback for compatibility.
  context->SetTilingKey(bcWhich == 1u ? 1u : (bcWhich == 2u ? 2u : 0u));
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
  context->SetOutputDataType(0, ge::DT_BOOL);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Greater : public OpDef {
 public:
  explicit Greater(const char* name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                   ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Greater);
}  // namespace ops
