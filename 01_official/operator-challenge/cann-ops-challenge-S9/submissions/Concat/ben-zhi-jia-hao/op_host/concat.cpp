#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kCachedInputs = 256;

uint32_t ElementBytes(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_INT8: return 1;
    case ge::DT_FLOAT16: return 2;
    case ge::DT_FLOAT:
    case ge::DT_INT32: return 4;
    default: return 0;
  }
}

bool Multiply(uint64_t &value, uint64_t factor) {
  if (factor && value > std::numeric_limits<uint64_t>::max() / factor) return false;
  value *= factor;
  return true;
}

const gert::StorageShape *InputShape(gert::TilingContext *ctx, size_t i) {
  auto *shape = ctx->GetDynamicInputShape(0, i);
  return shape ? shape : ctx->GetInputShape(i);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
  const size_t count = ctx->GetComputeNodeInputNum();
  if (!count || count > std::numeric_limits<uint32_t>::max()) return ge::GRAPH_FAILED;
  const auto *first = InputShape(ctx, 0);
  const auto *desc = ctx->GetInputDesc(0);
  const auto *attrs = ctx->GetAttrs();
  if (!first || !desc || !attrs || !attrs->GetInt(0)) return ge::GRAPH_FAILED;
  const auto &shape = first->GetStorageShape();
  const int64_t rank = shape.GetDimNum();
  int64_t axis = *attrs->GetInt(0);
  if (axis < 0) axis += rank;
  const uint32_t elementBytes = ElementBytes(desc->GetDataType());
  if (rank < 1 || axis < 0 || axis >= rank || !elementBytes) return ge::GRAPH_FAILED;

  uint64_t outer = 1, innerBytes = elementBytes;
  for (int64_t d = 0; d < rank; ++d) {
    if (shape.GetDim(d) < 0) return ge::GRAPH_FAILED;
    if (d < axis && !Multiply(outer, shape.GetDim(d))) return ge::GRAPH_FAILED;
    if (d > axis && !Multiply(innerBytes, shape.GetDim(d))) return ge::GRAPH_FAILED;
  }
  uint64_t rowBytes = 0, maxSegmentBytes = 0, rows[kCachedInputs] = {};
  for (size_t i = 0; i < count; ++i) {
    const auto *storage = InputShape(ctx, i);
    if (!storage) return ge::GRAPH_FAILED;
    const auto &input = storage->GetStorageShape();
    if (input.GetDimNum() != static_cast<size_t>(rank)) return ge::GRAPH_FAILED;
    for (int64_t d = 0; d < rank; ++d) {
      if (input.GetDim(d) < 0 || (d != axis && input.GetDim(d) != shape.GetDim(d)))
        return ge::GRAPH_FAILED;
    }
    uint64_t bytes = innerBytes;
    if (!Multiply(bytes, input.GetDim(axis)) ||
        rowBytes > std::numeric_limits<uint64_t>::max() - bytes) return ge::GRAPH_FAILED;
    rowBytes += bytes;
    maxSegmentBytes = std::max(maxSegmentBytes, bytes);
    if (i < kCachedInputs) rows[i] = bytes;
  }
  uint64_t totalBytes = rowBytes;
  if (!Multiply(totalBytes, outer)) return ge::GRAPH_FAILED;

  ConcatTilingData data;
  data.set_inputCount(static_cast<uint32_t>(count));
  data.set_axis(static_cast<uint32_t>(axis));
  data.set_rank(static_cast<uint32_t>(rank));
  data.set_reserved(0);
  data.set_outer(outer);
  data.set_innerBytes(innerBytes);
  data.set_rowBytes(rowBytes);
  data.set_totalBytes(totalBytes);
  data.set_inputRowBytes(rows);
  auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
  const uint32_t cores = std::max(1U, platform.GetCoreNumAiv());
  const uint64_t wanted = totalBytes / 32768 + (totalBytes % 32768 != 0);
  uint64_t active = std::max<uint64_t>(1, std::min<uint64_t>(cores, wanted));
  uint64_t key = 4;
  if (outer == 1) {
    key = 1;
    active = std::max<uint64_t>(1, std::min<uint64_t>(cores, (totalBytes + 262143) / 262144));
  } else if (outer > 1 && rowBytes && maxSegmentBytes <= 65536 &&
             rowBytes <= std::numeric_limits<uint32_t>::max() / 32) {
    if (totalBytes <= 262144) {
      uint32_t alignment = 32;
      while (rowBytes % alignment) alignment /= 2;
      const uint64_t groups = (outer + 32 / alignment - 1) / (32 / alignment);
      const uint64_t smallWanted = (totalBytes + 8191) / 8192;
      active = std::max<uint64_t>(1, std::min<uint64_t>(cores, std::min(groups, smallWanted)));
      key = 2;
    } else {
      key = 3;
    }
  }

  if (outer > 1 && innerBytes % 4 == 0 && totalBytes >= 1048576 &&
      maxSegmentBytes <= 16384 && rowBytes && rowBytes <= UINT32_MAX / 8) {
    key = 5;
    uint64_t tasks = 0;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t bytes = i < kCachedInputs ? rows[i] :
          innerBytes * InputShape(ctx, i)->GetStorageShape().GetDim(axis);
      if (!bytes) continue;
      const uint64_t perCopy = std::min<uint64_t>(4095, 32768 / ((bytes + 31) / 32 * 32));
      tasks = std::min<uint64_t>(cores, tasks + std::min<uint64_t>(cores, (outer + perCopy - 1) / perCopy));
    }
    active = std::max<uint64_t>(1, tasks);
  }

  uint64_t workspaceBytes = 0;
  uint32_t inputOffsets[256] = {}, columnFirst[256] = {};
  if (count <= 256 && totalBytes <= UINT32_MAX && rowBytes <= 524288 &&
      outer > 1 && innerBytes % 4 == 0 && maxSegmentBytes <= 512 &&
      totalBytes >= 1048576 && rowBytes >= 2048) {
    key = 6;
    active = cores;
    uint32_t prefix = 0;
    for (uint32_t j = 0; j < count; ++j) {
      inputOffsets[j] = prefix;
      prefix += rows[j];
    }
    uint32_t groups = 1, width = 0;
    columnFirst[0] = 0;
    for (uint32_t j = 0; j < count; ++j) {
      if (width && width + rows[j] > 4096) { columnFirst[groups++] = j; width = 0; }
      width += rows[j];
    }
    data.set_reserved(groups);
    if(groups<64 && groups<=cores) {
      const uint64_t issueBytes=192;
      std::vector<uint64_t> prefix(count+1,0),weight(count+1,0);
      for(uint32_t j=0;j<count;++j) {
        prefix[j+1]=prefix[j]+rows[j];
        weight[j+1]=weight[j]+rows[j]+(rows[j]?issueBytes:0);
      }
      const uint64_t inf=UINT64_MAX/4;
      std::vector<uint64_t> previous(count+1,inf),current(count+1,inf);
      std::vector<uint32_t> back((groups+1)*(count+1),UINT32_MAX);
      previous[0]=0;
      for(uint32_t g=1;g<=groups;++g) {
        std::fill(current.begin(),current.end(),inf);
        for(uint32_t end=1;end<=count;++end) {
          for(uint32_t begin=end;begin>0;) {
            --begin;
            const uint64_t width=prefix[end]-prefix[begin];
            if(width>4096)break;
            if(!width || previous[begin]==inf)continue;
            const int64_t diff=static_cast<int64_t>((weight[end]-weight[begin])*groups)-static_cast<int64_t>(weight[count]);
            const uint64_t cost=previous[begin]+static_cast<uint64_t>(diff*diff);
            if(cost<current[end]) {current[end]=cost;back[g*(count+1)+end]=begin;}
          }
        }
        previous.swap(current);
      }
      if(previous[count]!=inf) {
        uint32_t end=count;
        for(uint32_t g=groups;g>0;--g) {
          const uint32_t begin=back[g*(count+1)+end];
          columnFirst[g-1]=begin;end=begin;
        }
      }
      uint64_t costs[64]={};uint32_t workers[64]={};
      for(uint32_t g=0;g<groups;++g) {
        const uint32_t begin=columnFirst[g],end=g+1<groups?columnFirst[g+1]:count;
        costs[g]=weight[end]-weight[begin];workers[g]=1;
      }
      for(uint32_t assigned=groups;assigned<cores;++assigned) {
        uint32_t selected=0;
        for(uint32_t g=1;g<groups;++g)
          if(costs[g]*workers[selected]>costs[selected]*workers[g])selected=g;
        ++workers[selected];
      }
      uint32_t first=0;
      for(uint32_t g=0;g<groups;++g) {columnFirst[128+g]=first;first+=workers[g];}
      columnFirst[128+groups]=first;
    }


    uint64_t l2Bytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
    const uint64_t suggestedRows = std::max<uint64_t>(8, (l2Bytes/8/rowBytes)/8*8);
    const uint64_t bandRows = std::min<uint64_t>((outer+7)/8*8, suggestedRows);
    for (uint32_t g = 0; g < groups; ++g) {
      const uint32_t begin = columnFirst[g], end = g+1 < groups ? columnFirst[g+1] : count;
      const uint64_t width = (end < count ? inputOffsets[end] : rowBytes) - inputOffsets[begin];
      if (!width) continue;
      const uint64_t stride = (width+31)/32*32;
      const uint64_t tileRows = std::min<uint64_t>(32768/stride/8*8, (outer+7)/8*8);

      if (groups >=64 || groups > cores) continue;
      const uint64_t firstCore = columnFirst[128+g];
      const uint64_t lastCore = columnFirst[128+g+1];
      const uint64_t workers = lastCore-firstCore;
      const uint64_t localRows = (bandRows+workers*tileRows-1)/(workers*tileRows)*tileRows;
      workspaceBytes += workers*localRows*width;
    }
    // At least one core per positive-width group; remaining cores follow width.
    // The last planning slot holds bandRows and is not a group-start entry.
    if (groups >= 64 || groups > cores || workspaceBytes > l2Bytes/4) {
      key = 5; workspaceBytes = 0;
    } else {
      columnFirst[255] = static_cast<uint32_t>(bandRows);
    }


  }
  uint32_t panelPlan[8192]={};
  if(key==2 && count<=16 && innerBytes%2==0 && rowBytes<=1024 && totalBytes) {
    key=7;active=std::min<uint64_t>(cores,(outer+15)/16);
    const uint32_t stride=(rowBytes+31)/32*32;
    uint32_t prefix=0;
    for(uint32_t j=0;j<count;++j) {
      for(uint32_t r=0;r<16;++r)
        for(uint32_t b=0;b<rows[j];b+=2)
          panelPlan[(r*stride+prefix+b)/2]=16*prefix+r*rows[j]+b;
      prefix+=rows[j];
    }
  }
  if(key==1 && count<=256 && active<=64 && totalBytes && totalBytes<=UINT32_MAX) {
    uint32_t cursor=0;bool fits=true;
    const uint64_t blocks=(totalBytes+31)/32;
    for(uint32_t core=0;core<active && fits;++core) {
      panelPlan[core]=cursor;
      const uint64_t begin=blocks*core/active*32;
      const uint64_t end=std::min<uint64_t>(totalBytes,blocks*(core+1)/active*32);
      uint64_t prefix=0;
      for(uint32_t j=0;j<count && fits;++j) {
        uint64_t lo=std::max(begin,prefix),hi=std::min(end,prefix+rows[j]);
        while(lo<hi) {
          if(65+4*(cursor+1)>8192){fits=false;break;}
          const uint32_t length=std::min<uint64_t>(65536,hi-lo);
          panelPlan[65+4*cursor]=j;panelPlan[66+4*cursor]=lo-prefix;
          panelPlan[67+4*cursor]=lo;panelPlan[68+4*cursor]=length;
          ++cursor;lo+=length;
        }
        prefix+=rows[j];
      }
    }
    panelPlan[active]=cursor;
    if(fits)key=8;
  }
  if(key==3 && count<=96 && innerBytes%4==0 && maxSegmentBytes<=65536 && outer<=UINT32_MAX && totalBytes<=UINT32_MAX) {
    struct RowJob {uint32_t input,src,dest,bytes;};
    std::vector<RowJob> jobs;
    bool fits=outer*count<=(8192-65)/4;
    uint32_t prefix=0;
    for(uint32_t row=0;row<outer && fits;++row) {
      prefix=0;
      for(uint32_t j=0;j<count;++j) {
        if(rows[j])jobs.push_back({j,static_cast<uint32_t>(row*rows[j]),static_cast<uint32_t>(row*rowBytes+prefix),static_cast<uint32_t>(rows[j])});
        prefix+=rows[j];
      }
    }
    if(65+4*jobs.size()>8192)fits=false;
    if(fits && !jobs.empty()) {
      active=std::min<uint64_t>(cores,jobs.size());
      std::stable_sort(jobs.begin(),jobs.end(),[](const RowJob&a,const RowJob&b){return a.bytes>b.bytes;});
      std::vector<uint64_t> load(active,0);std::vector<std::vector<RowJob>> assigned(active);
      for(const auto &job:jobs){const uint32_t core=std::min_element(load.begin(),load.end())-load.begin();assigned[core].push_back(job);load[core]+=job.bytes;}
      uint32_t cursor=0;
      for(uint32_t core=0;core<active;++core) {
        std::stable_sort(assigned[core].begin(),assigned[core].end(),[](const RowJob&a,const RowJob&b){return a.dest<b.dest;});
        panelPlan[core]=cursor;
        for(const auto &job:assigned[core]){panelPlan[65+4*cursor]=job.input;panelPlan[66+4*cursor]=job.src;panelPlan[67+4*cursor]=job.dest;panelPlan[68+4*cursor]=job.bytes;++cursor;}
      }
      panelPlan[active]=cursor;key=9;
    }
  }
  data.set_panelPlan(panelPlan);
  data.set_inputOffsets(inputOffsets);
  data.set_columnFirst(columnFirst);
  ctx->SetBlockDim(static_cast<uint32_t>(active));
  ctx->SetTilingKey(key);
  if(std::getenv("CONCAT_PLAN_LOG"))std::fprintf(stderr,"COMBINED_PLAN key=%llu cores=%llu outer=%llu row=%llu tilingBytes=%u\n",(unsigned long long)key,(unsigned long long)active,(unsigned long long)outer,(unsigned long long)rowBytes,data.GetDataSize());

  if (data.GetDataSize() != 36912) return ge::GRAPH_FAILED;
  auto *raw = ctx->GetRawTilingData();
  if (data.GetDataSize() > raw->GetCapacity()) return ge::GRAPH_FAILED;
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  ctx->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
  const size_t count = ctx->GetComputeNodeInputNum();
  auto *out = ctx->GetOutputShape(0);
  auto *attrs = ctx->GetAttrs();
  auto *first = ctx->GetInputShape(0);
  if (!count || !out || !attrs || !attrs->GetInt(0) || !first) return GRAPH_FAILED;
  const int64_t rank = first->GetDimNum();
  int64_t axis = *attrs->GetInt(0);
  if (axis < 0) axis += rank;
  if (rank < 1 || axis < 0 || axis >= rank) return GRAPH_FAILED;
  int64_t extent = 0;
  for (size_t i = 0; i < count; ++i) {
    auto *s = ctx->GetDynamicInputShape(0, i);
    if (!s) s = ctx->GetInputShape(i);
    if (!s || s->GetDimNum() != static_cast<size_t>(rank)) return GRAPH_FAILED;
    const int64_t n = s->GetDim(axis);
    if (n < 0) { extent = -1; break; }
    if (extent > std::numeric_limits<int64_t>::max() - n) return GRAPH_FAILED;
    extent += n;
  }
  *out = *first;
  out->SetDim(axis, extent);
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
  ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
  return GRAPH_SUCCESS;
}
}

namespace ops {
class Concat : public OpDef {
 public:
  explicit Concat(const char *name) : OpDef(name) {
    this->Input("x").ParamType(DYNAMIC)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y").ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("dim").AttrType(OPTIONAL).Int(0);
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};
OP_ADD(Concat);
}
