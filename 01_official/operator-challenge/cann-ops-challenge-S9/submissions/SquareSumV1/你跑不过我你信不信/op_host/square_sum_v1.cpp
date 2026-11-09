#include <cstddef>
#include <cstdint>
#include <cstring>

#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t TILE_ELEMS = 8192;
constexpr uint32_t MAX_DIM = 8;
constexpr uint64_t MIN_RPAR = 32768;

// The tiling key selects the kernel binary: each mode is compiled on its own (key = mode + 1),
// so one mode's kernel carries none of the other modes' code.
enum class SquareSumKey : uint32_t {
  KEY_ROW = 1,          // mode 0
  KEY_GENERAL = 2,      // mode 1
  KEY_RPAR = 3,         // mode 2
  KEY_COLUMN = 4,       // mode 3
  KEY_KEPT_MIDDLE = 5,  // mode 4
  KEY_FLAT = 10,        // mode 9
};

static inline uint32_t AlignUpU(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }
static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static inline uint32_t GcdU(uint32_t a, uint32_t b) {
  while (b != 0u) {
    uint32_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  TilingData tiling;
  auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t maxCores = plat.GetCoreNumAiv();
  if (maxCores < 1) maxCores = 1;
  uint32_t sysWs = plat.GetLibApiWorkSpaceSize();

  auto inShape = context->GetInputShape(0)->GetStorageShape();
  uint32_t D = (uint32_t)inShape.GetDimNum();
  if (D == 0) D = 1;
  if (D > MAX_DIM) D = MAX_DIM;
  uint64_t inDims[MAX_DIM], inStr[MAX_DIM];
  uint64_t total = 1;
  for (uint32_t d = 0; d < D; d++) {
    inDims[d] = (uint64_t)inShape.GetDim(d);
    if (inDims[d] == 0) inDims[d] = 1;
    total *= inDims[d];
  }
  inStr[D - 1] = 1;
  for (int d = (int)D - 2; d >= 0; d--) inStr[d] = inStr[d + 1] * inDims[d + 1];

  uint32_t elemSz = 2;
  { auto dt = context->GetInputDesc(0)->GetDataType(); if (dt == ge::DT_FLOAT) elemSz = 4; }

  auto attrs = context->GetAttrs();
  const auto* axisAttr = attrs->GetListInt(0);
  size_t axisNum = axisAttr->GetSize();
  const int64_t* axisData = axisAttr->GetData();
  bool red[MAX_DIM] = {false};
  for (size_t k = 0; k < axisNum; k++) {
    int64_t a = axisData[k];
    if (a < 0) a += (int64_t)D;
    if (a >= 0 && a < (int64_t)D) red[a] = true;
  }
  int lo = -1, hi = -1;
  for (uint32_t d = 0; d < D; d++) if (red[d]) { if (lo < 0) lo = (int)d; hi = (int)d; }
  bool contig = true;
  if (lo >= 0) for (int d = lo; d <= hi; d++) if (!red[d]) contig = false;

  uint64_t userWsBytes = 0;

  if (lo < 0 || contig) {
    uint64_t outer = 1, reduceLen = 1, inner = 1;
    if (lo >= 0) {
      for (int d = 0; d < lo; d++) outer *= inDims[d];
      for (int d = lo; d <= hi; d++) reduceLen *= inDims[d];
      for (int d = hi + 1; d < (int)D; d++) inner *= inDims[d];
    } else { outer = total; }
    if (outer == 0) outer = 1;
    tiling.set_outer((uint32_t)outer);
    tiling.set_reduceLen((uint32_t)reduceLen);
    tiling.set_inner((uint32_t)inner);

    if (inner > 1) {
      // ---- COLUMN ----
      uint32_t L = (uint32_t)outer, M = (uint32_t)reduceLen, N = (uint32_t)inner;
      // kept inner is tiled into <=ntCap slices when N is large; N-tiles are an extra
      // source of core parallelism.
      uint32_t ntCapH = (elemSz == 4) ? TILE_ELEMS : (TILE_ELEMS / 2);
      uint32_t numNtiles = (N <= ntCapH) ? 1u : CeilDiv(N, ntCapH);  // N-tiles as extra parallelism
      // one contiguous multi-MB run per core: past a point, more readers only add
      // GM stream interference (measured: 20 readers beat 40 by a sixth on this stream)
      uint32_t streamCores = maxCores;
      if (elemSz == 2 && (uint64_t)M * inner * elemSz >= (96ULL << 20)) {
        streamCores = 32u;
      }
      if (elemSz == 4 && (uint64_t)M * inner * elemSz >= (128ULL << 20)) {
        streamCores = 20u;
        if (streamCores > maxCores) streamCores = maxCores;
      }
      uint32_t chunksPerL;
      if ((uint64_t)L * numNtiles >= streamCores) {
        // (l, n-tile) parallelism alone fills the cores -> don't m-split: each work-item
        // folds ALL M rows for its (l) -> complete sum -> direct cast&write y_gm (no ws/atomic).
        chunksPerL = 1;
      } else {
        chunksPerL = (L >= streamCores) ? 1u : CeilDiv(streamCores, L);
        if (chunksPerL > M) chunksPerL = M;
        if (chunksPerL < 1) chunksPerL = 1;
      }
      uint32_t mPerCore = CeilDiv(M, chunksPerL);
      if (elemSz == 4 && (uint64_t)M * inner * elemSz >= (128ULL << 20)) {
        uint32_t rowBytes = N * elemSz;
        uint32_t phaseRows = 512u / GcdU(rowBytes, 512u);
        mPerCore = AlignUpU(mPerCore, phaseRows);
      }
      chunksPerL = CeilDiv(M, mPerCore);
      uint64_t totalUnits = (uint64_t)L * chunksPerL * numNtiles;
      uint32_t colCores = (uint32_t)(totalUnits < streamCores ? totalUnits : streamCores);
      if (colCores < 1) colCores = 1;
      // Tried mode 5 (flat 512B-aligned read + phase-gather realign) for large-stream workload: profile showed it
      // flips the kernel from READ-bound (mode3, 910us) to VEC-bound (gather-of-whole-input =
      // 2094us vec, mte2 only 16.6%). The whole-input gather can't beat mode3's MTE2 align. Keep 3.
      {
        // cores past the last whole share of units would launch with nothing to do
        uint64_t uPer = (totalUnits + colCores - 1) / colCores;
        colCores = (uint32_t)((totalUnits + uPer - 1) / uPer);
      }
      context->SetBlockDim(colCores);
      tiling.set_mode(3);
      context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_COLUMN));
      tiling.set_colCores(colCores);
      tiling.set_mPerCore(mPerCore);
      tiling.set_innerP(AlignUpU(N, 8));
      // only the m-split/atomic path (chunksPerL>1) needs the fp32 scratch; the direct
      // path (chunksPerL==1) casts complete sums straight to y_gm.
      if (elemSz != 4 && chunksPerL > 1) userWsBytes = (uint64_t)L * N * sizeof(float);
    } else {
      // inner == 1: ROW or RPAR
      bool rpar = (outer < maxCores) && (total >= MIN_RPAR) && (reduceLen >= 2);
      if (rpar) {
        uint32_t coreNum = (uint32_t)(reduceLen < maxCores ? reduceLen : maxCores);
        if (coreNum < 1) coreNum = 1;
        uint32_t rPerCore = CeilDiv((uint32_t)reduceLen, coreNum);
        uint32_t rCores = CeilDiv((uint32_t)reduceLen, rPerCore);
        uint32_t Oslot = AlignUpU((uint32_t)outer, 8);
        context->SetBlockDim(rCores);
        tiling.set_mode(2);
        context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_RPAR));
        tiling.set_O((uint32_t)outer);
        tiling.set_Oslot(Oslot);
        tiling.set_rCores(rCores);
        tiling.set_rPerCore(rPerCore);
        userWsBytes = (uint64_t)rCores * Oslot * sizeof(float);
      } else {
        uint32_t coreNum = (uint32_t)(outer < maxCores ? outer : maxCores);
        // tiny launch-bound ROW case (e.g. tiny workload [123,31]=3813 elems): 40-core dispatch overhead
        // dominates. Cap cores so each holds >= ~512 elems (even count, >=2; 8 cores measured
        // 2.8us against 3.3us on 4 and 7.3us on 1). Large ROW cases (fused-axis workload 15.6M)
        // keep full cores.
        if (total < 32768) {
          uint32_t want = (uint32_t)((total + 511) / 512);
          if (want < 2) want = 2;
          want = (want + 1) & ~1u;  // round up to even
          if (want < coreNum) coreNum = want;
        }
        if (coreNum < 1) coreNum = 1;
        uint32_t perCore = CeilDiv((uint32_t)outer, coreNum);
        // 2-byte rows of 8q+1 elements reduce in 8-row phase cycles; a share that starts on a
        // multiple of eight rows starts every cycle within eight elements after a 32B boundary
        if (elemSz == 2u && (reduceLen % 8u) == 1u && perCore >= 8u) {
          perCore = AlignUpU(perCore, 8u);
          coreNum = CeilDiv((uint32_t)outer, perCore);
        }
        context->SetBlockDim(coreNum);
        tiling.set_mode(0);
        context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_ROW));
        tiling.set_outerPerCore(perCore);
      }
    }
  } else {
    // ---- non-contiguous reduce ----
    // Fast-path mode 4: reduced = prefix + suffix, kept = contiguous middle block ->
    //   fold to [OUT_R (reduced planes), KEPT (kept contig), IN_R (reduced inner)];
    //   owned-KEPT-rows read [keptTile, IN_R] contiguous per plane -> reduce inner -> accumulate.
    int kLo = -1, kHi = -1;
    for (uint32_t d = 0; d < D; d++) if (!red[d]) { if (kLo < 0) kLo = (int)d; kHi = (int)d; }
    bool keptContig = (kLo >= 0);
    if (kLo >= 0) for (int d = kLo; d <= kHi; d++) if (red[d]) keptContig = false;
    bool psRed = true;
    if (kLo >= 0) {
      for (int d = 0; d < kLo; d++) if (!red[d]) psRed = false;
      for (int d = kHi + 1; d < (int)D; d++) if (!red[d]) psRed = false;
    }
    uint64_t OUT_R = 1, KEPT = 1, IN_R = 1;
    if (kLo >= 0) {
      for (int d = 0; d < kLo; d++) OUT_R *= inDims[d];
      for (int d = kLo; d <= kHi; d++) KEPT *= inDims[d];
      for (int d = kHi + 1; d < (int)D; d++) IN_R *= inDims[d];
    }
    // FLAT (mode 9), fp16 only: every core reads an equal plane-order share of positions, aligned
    // to 16 positions so each read of whole positions starts and ends on a 32B boundary.  A row
    // plus its start phase (at most 15 elements) must fit one 128-element reduce window; rows
    // below 64 elements leave reads too small for the per-read reduce to pay off.
    bool flat = kLo >= 0 && keptContig && psRed && OUT_R > 1 && elemSz == 2 && IN_R >= 64u &&
                IN_R <= 112u && context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16 &&
                OUT_R * KEPT * IN_R * elemSz >= (64ULL << 20);
    uint64_t flatPer = 0u;
    if (flat) {
      uint64_t positions = OUT_R * KEPT;
      flatPer = ((positions + maxCores - 1u) / maxCores + 15u) / 16u * 16u;
      if (flatPer > 0xFFFFF000ULL) flat = false;
    }
    if (flat) {
      uint64_t positions = OUT_R * KEPT;
      uint32_t cn = (uint32_t)((positions + flatPer - 1u) / flatPer);
      context->SetBlockDim(cn);
      tiling.set_mode(9);
      context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_FLAT));
      tiling.set_outer((uint32_t)OUT_R);        // reduced planes
      tiling.set_reduceLen((uint32_t)IN_R);     // reduced inner
      tiling.set_inner((uint32_t)KEPT);         // kept total
      tiling.set_mPerCore((uint32_t)flatPer);   // positions per core
      tiling.set_colCores(cn);
      // plane sums accumulate in fp32: an fp16 output that is much larger than each plane's
      // increment rounds every atomic add to its own ulp, and those rounding errors add up
      userWsBytes = KEPT * sizeof(float);
    } else if (kLo >= 0 && keptContig && psRed && OUT_R > 1 && IN_R > 1) {
      uint32_t keptPerCore = CeilDiv((uint32_t)KEPT, maxCores);
      if (keptPerCore < 1) keptPerCore = 1;
      uint32_t cn = CeilDiv((uint32_t)KEPT, keptPerCore);
      if (cn < 1) cn = 1;
      context->SetBlockDim(cn);
      tiling.set_mode(4);
      context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_KEPT_MIDDLE));
      tiling.set_outer((uint32_t)OUT_R);       // reduced plane count
      tiling.set_reduceLen((uint32_t)IN_R);    // reduced inner
      tiling.set_inner((uint32_t)KEPT);        // kept total
      tiling.set_outerPerCore(keptPerCore);    // kept rows per core
    } else {
    // ---- GENERAL (non-contiguous) ----
    uint32_t unitReduced = red[D - 1] ? 1 : 0;
    int t = (int)D - 1;
    while (t - 1 >= 0 && (red[t - 1] ? 1u : 0u) == unitReduced) t--;
    uint64_t unit = 1;
    for (int d = t; d < (int)D; d++) unit *= inDims[d];
    uint32_t klDim[MAX_DIM] = {0}, klStr[MAX_DIM] = {0}, rlDim[MAX_DIM] = {0}, rlStr[MAX_DIM] = {0};
    uint32_t nKL = 0, nRL = 0;
    uint64_t klProd = 1, rlProd = 1;
    for (int d = 0; d < t; d++) {
      if (red[d]) { rlDim[nRL] = (uint32_t)inDims[d]; rlStr[nRL] = (uint32_t)inStr[d]; nRL++; rlProd *= inDims[d]; }
      else { klDim[nKL] = (uint32_t)inDims[d]; klStr[nKL] = (uint32_t)inStr[d]; nKL++; klProd *= inDims[d]; }
    }
    if (klProd == 0) klProd = 1;
    uint32_t coreNum = (uint32_t)(klProd < maxCores ? klProd : maxCores);
    if (coreNum < 1) coreNum = 1;
    context->SetBlockDim(coreNum);
    tiling.set_mode(1);
    context->SetTilingKey(static_cast<uint32_t>(SquareSumKey::KEY_GENERAL));
    tiling.set_unitReduced(unitReduced);
    tiling.set_unit((uint32_t)unit);
    tiling.set_klProd((uint32_t)klProd);
    tiling.set_rlProd((uint32_t)rlProd);
    tiling.set_koPerCore(CeilDiv((uint32_t)klProd, coreNum));
    tiling.set_nKL(nKL);
    tiling.set_nRL(nRL);
    tiling.set_klDim(klDim); tiling.set_klStr(klStr);
    tiling.set_rlDim(rlDim); tiling.set_rlStr(rlStr);
    }
  }

  tiling.set_tileElems(TILE_ELEMS);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* ws = context->GetWorkspaceSizes(1);
  ws[0] = (size_t)sysWs + (size_t)userWsBytes;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SquareSumV1 : public OpDef {
 public:
  explicit SquareSumV1(const char* name) : OpDef(name) {
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("axis").AttrType(REQUIRED).ListInt();
    this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(SquareSumV1);
}  // namespace ops
