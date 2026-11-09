#include <algorithm>
#include <cstdint>
#include <limits>
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
bool Product(uint64_t &v, uint64_t n) {
  if (n && v > std::numeric_limits<uint64_t>::max() / n) return false;
  v *= n;
  return true;
}
uint32_t Type(ge::DataType t) {
  switch (t) {
    case ge::DT_FLOAT: return 1;
    case ge::DT_FLOAT16: return 2;
    case ge::DT_BF16: return 3;
    case ge::DT_INT32: return 4;
    case ge::DT_INT8: return 5;
    default: return 0;
  }
}
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
  const auto *x = ctx->GetInputShape(0);
  const auto *ix = ctx->GetInputShape(1);
  const auto *s = ctx->GetInputShape(2);
  const auto *xd = ctx->GetInputDesc(0);
  const auto *id = ctx->GetInputDesc(1);
  const auto *sd = ctx->GetInputDesc(2);
  const auto *attrs = ctx->GetAttrs();
  if (!x || !ix || !s || !xd || !id || !sd || !attrs || !attrs->GetInt(0))
    return ge::GRAPH_FAILED;
  const auto &xs = x->GetStorageShape();
  const auto &is = ix->GetStorageShape();
  const auto &ss = s->GetStorageShape();
  const int64_t rank = xs.GetDimNum();
  const int64_t originalDim = *attrs->GetInt(0);
  const int64_t dim = originalDim < 0 ? originalDim + rank : originalDim;
  const uint32_t dtype = Type(xd->GetDataType());
  if (!dtype || id->GetDataType() != ge::DT_INT32 || sd->GetDataType() != xd->GetDataType() ||
      rank < 1 || dim < 0 || dim >= rank || is.GetDimNum() != 1 ||
      ss.GetDimNum() != static_cast<size_t>(rank) || is.GetDim(0) < 0)
    return ge::GRAPH_FAILED;
  uint64_t outer = 1, inner = 1;
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t n = xs.GetDim(d);
    if (n < 0 || ss.GetDim(d) != (d == dim ? is.GetDim(0) : n)) return ge::GRAPH_FAILED;
    if (d < dim && !Product(outer, n)) return ge::GRAPH_FAILED;
    if (d > dim && !Product(inner, n)) return ge::GRAPH_FAILED;
  }
  const uint64_t axis = xs.GetDim(dim), count = is.GetDim(0);
  uint64_t total = outer, sourceTotal = outer;
  const uint32_t bytes = dtype == 5 ? 1 : (dtype == 2 || dtype == 3 ? 2 : 4);
  if (!Product(total, axis) || !Product(total, inner) ||
      !Product(sourceTotal, count) || !Product(sourceTotal, inner) ||
      total > std::numeric_limits<uint64_t>::max() / bytes ||
      sourceTotal > std::numeric_limits<uint64_t>::max() / bytes)
    return ge::GRAPH_FAILED;
  IndexAddTilingData data;
  data.set_outer(outer); data.set_axisSize(axis); data.set_inner(inner);
  data.set_indexCount(count); data.set_total(total); data.set_dtype(dtype);
  // The linked list preserves the original index order. Large metadata uses
  // the general scan fallback instead of truncating the actual index sequence.
  const bool lists = axis <= 20000 && count <= 20000 && axis + count <= 24576;
  data.set_listBytes(lists ? static_cast<uint32_t>(((axis + count) * 4 + 31) / 32 * 32) : 0);
  uint32_t bf16Batch=0;uint64_t bf16Workspace=0;
  if(dtype==3 && inner>=16 && inner<=512 && axis>0 && axis<=2000 && count>=64 && count<=8000 && outer>0) {
    const uint64_t stride=(inner+15)/16*16,accBytes=axis*stride*2;
    const uint64_t fixed=accBytes+stride*8+(count+7)/8*32;
    const uint64_t slabPad=(axis*inner+15)/16*16;
    if(fixed+stride*4<=180000 && outer<=(512ULL*1024*1024)/(slabPad*2)) {
      bf16Batch=static_cast<uint32_t>(std::min<uint64_t>(128,(180000-fixed)/(stride*4)));
      bf16Workspace=outer*slabPad*2;
    }
  }
  data.set_bf16Batch(bf16Batch);
  auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
  const bool parallelPlan=(dtype==1 || dtype==2) && inner==1 && outer>=64 && axis>0 && axis<=1024 && count>=64 && count<=8000;
  data.set_parallelPlan(parallelPlan?1:0);
  const uint64_t axisPad=(axis+(32/bytes)-1)/(32/bytes)*(32/bytes);
  const uint64_t planWorkspace=parallelPlan?4*(axis*((count+15)/8*8)+std::max(1U,platform.GetCoreNumAiv())*8+count*axisPad):0;
  const uint64_t work = (dtype==4 || dtype==5) ? std::max(total,sourceTotal) : total;
  uint64_t chunks = work / 2048 + (work % 2048 != 0);
  const uint64_t outputBlocks=total/(32/bytes)+(total%(32/bytes)!=0);
  if(chunks>outputBlocks)chunks=outputBlocks;
  ctx->SetBlockDim(static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(
      std::max(1U, platform.GetCoreNumAiv()), chunks))));
  ctx->SetTilingKey(dtype);
  auto *raw = ctx->GetRawTilingData();
  if (data.GetDataSize() > raw->GetCapacity()) return ge::GRAPH_FAILED;
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  const uint64_t userWorkspace=std::max(bf16Workspace,planWorkspace);
  ctx->GetWorkspaceSizes(1)[0] = userWorkspace ? userWorkspace + platform.GetLibApiWorkSpaceSize() : 0;
  return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
  if (!ctx->GetInputShape(0) || !ctx->GetOutputShape(0)) return GRAPH_FAILED;
  *ctx->GetOutputShape(0) = *ctx->GetInputShape(0);
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
  ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
  return GRAPH_SUCCESS;
}
}
namespace ops {
class IndexAdd : public OpDef {
 public:
  explicit IndexAdd(const char *name) : OpDef(name) {
    this->Input("self").ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("index").ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("source").ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output").ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("dim").AttrType(OPTIONAL).Int(0);
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};
OP_ADD(IndexAdd);
}
