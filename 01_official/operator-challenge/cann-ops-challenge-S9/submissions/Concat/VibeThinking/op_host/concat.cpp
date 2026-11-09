#include "../op_kernel/concat_layout.h"
#include "concat_tiling.h"
#include "narrow_planner.h"
#include "swing_planner.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "algorithm"
#include "cstring"
#include "initializer_list"
#include "vector"

namespace {
using namespace compact;

// 统一负轴并校验非拼接维度，拼接轴长度累加得到输出形状。
template <class ShapeAt>
bool ConcatenatedShape(size_t count, int64_t const *requested, ShapeAt shapeAt,
                       gert::Shape &output, int64_t &dim) {
  auto const *first = count ? shapeAt(0) : nullptr;
  if (!first || !requested)
    return false;
  int64_t const rank = first->GetDimNum();
  if (*requested < -rank || *requested >= rank)
    return false;
  dim = (*requested + rank) % rank;
  output = *first;
  for (int64_t d = 0; (d < rank); ++d)
    if (first->GetDim(d) < 0)
      return false;
  int64_t extent = 0;
  for (size_t i = 0; (i != count); ++i) {
    auto const *part = shapeAt(i);
    if (!part || part->GetDimNum() != rank)
      return false;
    for (int64_t d = 0; (d < rank); ++d) {
      int64_t const size = part->GetDim(d);
      if (d == dim) {
        if (size < 0 || size > INT64_MAX - extent)
          return false;
        extent += size;
      } else if (size != first->GetDim(d)) {
        return false;
      }
    }
  }
  output.SetDim(dim, extent);
  return true;
}

template <class Port> void TensorContract(Port &port) {
  const std::initializer_list<ge::DataType> types = {
      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
  const std::initializer_list<ge::Format> formats = {
      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  port.DataType(types).Format(formats).UnknownShapeFormat(formats);
}

bool Specialized(Plan const &p, bool byRows, gert::TilingContext *c,
                 uint64_t ub) {
  if (!p.total || p.total > UINT32_MAX || p.cores > 64)
    return false;
  uint32_t active = 0;
  for (uint32_t i = 0; (i < p.count); ++i)
    active += p.width[i] != 0;
  uint32_t const key =
      p.rows == 1
          ? 7
          : (!byRows && p.rows < p.cores
                 ? 6
                 : (byRows && active <= 8 && p.total >= 1048576 ? 8 : 0));
  if (!key)
    return false;
  if (key == 6 && ub < kStreamUbBytes)
    return false;
  uint32_t const cores = key == 7 ? p.cores : std::min(32U, p.cores);
  Schedule s{};
  s.inputs = p.count;
  s.tile = key == 6 ? kStreamTile : (key == 8 ? 57344 : 180224);
  std::vector<Transfer> tasks;
  uint64_t const length = key == 8 ? p.rows : p.total;
  uint64_t const quantum = key == 8 ? p.quantum : 32;
  uint64_t const units = (length + quantum - 1) / quantum;
  for (uint32_t core = 0; (core < cores); ++core) {
    s.begin[core] = tasks.size();
    uint64_t const lo = units * core / cores * quantum;
    uint64_t const hi = std::min(length, units * (core + 1) / cores * quantum);
    if (key == 8) {
      uint64_t prefix = 0;
      for (uint32_t input = 0; (input < p.count); ++input) {
        uint32_t const width = p.width[input];
        if (!width)
          continue;
        uint32_t const batch =
            std::min(255U, s.tile / ((width + 31) / 32 * 32));
        if (!batch)
          return false;
        for (uint64_t row = lo; (row < hi); row += batch) {
          tasks.push_back(
              {static_cast<uint32_t>(row * width),
               static_cast<uint32_t>(row * p.rowBytes + prefix), width,
               static_cast<uint32_t>(p.rowBytes - width),
               static_cast<uint16_t>(std::min<uint64_t>(batch, hi - row)),
               static_cast<uint16_t>(input), 0});
          if (tasks.size() - s.begin[core] > 128)
            return false;
        }
        prefix += width;
      }
    } else {
      uint32_t arena = 0;
      for (uint64_t row = lo / p.rowBytes; (row * p.rowBytes < hi); ++row) {
        uint64_t prefix = row * p.rowBytes;
        for (uint32_t input = 0; (input < p.count); ++input) {
          uint64_t const end = std::min(hi, prefix + p.width[input]);
          for (uint64_t dst = std::max(lo, prefix); (dst < end);) {
            uint32_t const bytes = std::min<uint64_t>(s.tile, end - dst);
            tasks.push_back(
                {static_cast<uint32_t>(row * p.width[input] + dst - prefix),
                 static_cast<uint32_t>(dst), bytes, 0, 1,
                 static_cast<uint16_t>(input), key == 7 ? arena : 0});
            if (key == 7) {
              arena += (bytes + 31) / 32 * 32;
              if (arena > s.tile)
                return false;
            }
            if (tasks.size() - s.begin[core] > 128)
              return false;
            dst += bytes;
          }
          prefix += p.width[input];
        }
      }
    }
    if (tasks.size() - s.begin[core] > 128 ||
        sizeof(s) + tasks.size() * sizeof(Transfer) > 32768)
      return false;
  }
  s.begin[cores] = tasks.size();
  auto *raw = c->GetRawTilingData();
  std::vector<StreamSpan> spans;
  if (key == 6)
    for (auto const &t : tasks)
      spans.push_back({t.src, t.dst, t.bytes, t.input});
  size_t const payload =
      tasks.size() * (key == 6 ? sizeof(StreamSpan) : sizeof(Transfer));
  size_t const size = sizeof(s) + payload;
  if (raw->GetCapacity() < size)
    return false;
  std::memcpy(raw->GetData(), &s, sizeof(s));
  void const *records =
      key == 6 ? static_cast<void const *>(spans.data()) : tasks.data();
  std::memcpy(static_cast<char *>(raw->GetData()) + sizeof(s), records,
              payload);
  raw->SetDataSize(size);
  c->SetBlockDim(cores);
  c->SetTilingKey(key);
  return true;
}

bool PhasePlan(Plan const &p, gert::TilingContext *c, uint64_t ub) {
  if (p.count > 8 || p.rows < 64 || p.total < 1048576 ||
      p.rowBytes > UINT32_MAX / 8 || ub < kPhaseUbBytes)
    return false;
  uint32_t active = 0;
  bool unaligned = false;
  for (uint32_t i = 0; (i < p.count); ++i) {
    if (!p.width[i])
      continue;
    if (p.width[i] % 4 || p.width[i] < 512 || p.width[i] > 32768)
      return false;
    ++active;
    unaligned |= p.width[i] % 32 != 0;
  }
  if (active < 2 || !unaligned)
    return false;
  for (uint32_t i = 0; (i < p.count); ++i) {
    uint32_t const width = p.width[i];
    if (!width)
      continue;
    uint32_t period = 1;
    while (period * width % 32)
      period *= 2;
    uint32_t const batch = std::min(64U, kPhaseArena / width) / period * period;
    if (!batch)
      return false;
  }
  auto *raw = c->GetRawTilingData();
  size_t const size = sizeof(Plan);
  if (size > raw->GetCapacity())
    return false;
  Plan q = p;
  for (uint32_t i = 0; (i < p.count); ++i)
    while ((p.width[i] * q.quantum) % 32)
      q.quantum *= 2;
  uint64_t const units = (p.rows + q.quantum - 1) / q.quantum;
  uint64_t const perCore = (units + p.cores - 1) / p.cores;
  q.cores = (units + perCore - 1) / perCore;
  std::memcpy(raw->GetData(), &q, sizeof(q));
  raw->SetDataSize(size);
  c->SetBlockDim(q.cores);
  c->SetTilingKey(10);
  return true;
}

bool AffineRowSchedule(Plan const &p, gert::TilingContext *c, uint64_t ub) {
  if (!p.total || p.total > 262144 || p.rows < 16 || p.rowBytes > 512 ||
      p.count > 16)
    return false;
  uint32_t alignmentRows = 1;
  for (uint32_t i = 0; i < p.count; ++i) {
    if (p.width[i] % 2)
      return false;
    while (alignmentRows * p.width[i] % 32)
      alignmentRows *= 2;
  }
  AffineRowPlan a{};
  a.rows = p.rows;
  a.rowBytes = p.rowBytes;
  a.pitch = (a.rowBytes + 31) / 32 * 32;
  a.count = p.count;
  uint64_t const fixed = 8192 + 32 * a.count;
  if (ub <= fixed)
    return false;
  auto *raw = c->GetRawTilingData();
  if (raw->GetCapacity() <= sizeof(a))
    return false;
  uint32_t const ubRows = (ub - fixed) / (a.rowBytes + 3 * a.pitch);
  uint32_t const mapRows = (raw->GetCapacity() - sizeof(a)) / (2 * a.pitch);
  uint32_t const limit = std::min(ubRows, mapRows) / alignmentRows * alignmentRows;
  if (!limit)
    return false;
  uint32_t const desired = ((a.rows + p.cores - 1) / p.cores +
                            alignmentRows - 1) / alignmentRows * alignmentRows;
  a.tileRows = std::min(desired, limit);
  a.cores = std::min(p.cores, (a.rows + a.tileRows - 1) / a.tileRows);
  uint32_t const lanes = a.pitch / 2;
  std::vector<uint32_t> indices(a.tileRows * lanes, 0);
  uint32_t column = 0;
  for (uint32_t i = 0; i < a.count; ++i) {
    a.parts[i] = {static_cast<uint32_t>(p.width[i]), a.stageBytes};
    for (uint32_t row = 0; row < a.tileRows; ++row)
      for (uint32_t b = 0; b < p.width[i]; b += 2)
        indices[row * lanes + (column + b) / 2] = a.stageBytes + row * p.width[i] + b;
    a.stageBytes += (a.tileRows * p.width[i] + 31) / 32 * 32;
    column += p.width[i];
  }
  if (a.stageBytes + 3ULL * a.tileRows * a.pitch + 8192 > ub)
    return false;
  size_t const bytes = sizeof(a) + indices.size() * sizeof(uint32_t);
  if (bytes > raw->GetCapacity())
    return false;
  std::memcpy(raw->GetData(), &a, sizeof(a));
  std::memcpy(static_cast<char *>(raw->GetData()) + sizeof(a), indices.data(),
              indices.size() * sizeof(uint32_t));
  raw->SetDataSize(bytes);
  c->SetBlockDim(a.cores);
  c->SetTilingKey(9);
  return true;
}

bool ScheduleMoves(Plan const &p, bool byRows, gert::TilingContext *c) {
  if (p.cores > 64)
    return false;
  Schedule s{};
  s.tile = byRows ? 32768 : 65536;
  s.inputs = p.count;
  s.reserved = byRows ? 5 : 2;
  uint64_t rowStorage = 0;
  uint32_t active = 0;
  for (uint32_t i = 0; (i < p.count); ++i) {
    rowStorage += (p.width[i] + 31) / 32 * 32;
    active += p.width[i] != 0;
  }
  bool const cohort =
      byRows && active > 0 && active <= 8 && rowStorage <= 180224;
  uint32_t const cohortRows =
      cohort ? std::min<uint64_t>(255, 180224 / rowStorage) : 0;
  if (cohort)
    s.reserved = active;
  std::vector<Move> moves;
  uint64_t const quantum = byRows ? p.quantum : 32;
  uint64_t const length = byRows ? p.rows : p.total;
  uint64_t const units = (length + quantum - 1) / quantum;
  if (!length)
    return false;
  for (uint32_t core = 0; (core < p.cores); ++core) {
    s.begin[core] = moves.size();
    uint64_t const lo = units * core / p.cores * quantum;
    uint64_t const hi =
        std::min(length, units * (core + 1) / p.cores * quantum);
    if (cohort) {
      for (uint64_t row = lo; (row < hi); row += cohortRows) {
        uint32_t offset = 0;
        uint64_t prefix = 0;
        for (uint32_t i = 0; (i < p.count); ++i) {
          uint32_t const width = p.width[i];
          if (!width)
            continue;
          moves.push_back(
              {row * width, row * p.rowBytes + prefix, width,
               static_cast<uint32_t>(p.rowBytes - width),
               static_cast<uint16_t>(std::min<uint64_t>(cohortRows, hi - row)),
               static_cast<uint16_t>(i), offset});
          offset += (width + 31) / 32 * 32 * cohortRows;
          prefix += width;
        }
        if (moves.size() - s.begin[core] > 128)
          return false;
      }
    } else if (byRows) {
      uint64_t prefix = 0;
      for (uint32_t i = 0; (i < p.count); ++i) {
        uint64_t const width = p.width[i];
        if (!width)
          continue;
        if (width > s.tile)
          return false;
        uint32_t const batch =
            std::min<uint64_t>(255, s.tile / ((width + 31) / 32 * 32));
        for (uint64_t r = lo; (r < hi); r += batch) {
          moves.push_back(
              {r * width, r * p.rowBytes + prefix, static_cast<uint32_t>(width),
               static_cast<uint32_t>(p.rowBytes - width),
               static_cast<uint16_t>(std::min<uint64_t>(batch, hi - r)),
               static_cast<uint16_t>(i), 0});
          if (moves.size() - s.begin[core] > 128)
            return false;
        }
        prefix += width;
      }
    } else {
      for (uint64_t r = lo / p.rowBytes; (r * p.rowBytes < hi); ++r) {
        uint64_t prefix = r * p.rowBytes;
        for (uint32_t i = 0; (i < p.count); ++i) {
          uint64_t const stop = std::min(hi, prefix + p.width[i]);
          for (uint64_t at = std::max(lo, prefix); (at < stop);) {
            uint32_t const bytes = std::min<uint64_t>(s.tile, stop - at);
            moves.push_back({r * p.width[i] + at - prefix, at, bytes, 0, 1,
                             static_cast<uint16_t>(i), 0});
            at += bytes;
            if (moves.size() - s.begin[core] > 128)
              return false;
          }
          prefix += p.width[i];
        }
      }
    }
    if (!cohort)
      for (size_t i = s.begin[core]; (i < moves.size()); ++i)
        moves[i].reserved = ((i - s.begin[core]) % s.reserved) * s.tile;
    if (moves.size() * sizeof(Move) + sizeof(s) > 32768)
      return false;
  }
  s.begin[p.cores] = moves.size();
  auto *raw = c->GetRawTilingData();
  size_t const bytes = sizeof(s) + moves.size() * sizeof(Move);
  if (raw->GetCapacity() < bytes)
    return false;
  std::memcpy(raw->GetData(), &s, sizeof(s));
  std::memcpy(reinterpret_cast<char *>(raw->GetData()) + sizeof(s),
              moves.data(), moves.size() * sizeof(Move));
  raw->SetDataSize(bytes);
  c->SetTilingKey(5);
  return true;
}
bool Product(uint64_t &value, uint64_t factor) {
  if (factor && value > UINT64_MAX / factor)
    return false;
  value *= factor;
  return true;
}
ge::graphStatus Tile(gert::TilingContext *c) {
  auto *const info = c->GetIrInputInstanceInfo(0);
  auto *const attrs = c->GetAttrs();
  auto *const desc = c->GetDynamicInputDesc(0, 0);
  if (!info || !attrs || !desc)
    return ge::GRAPH_FAILED;
  Plan p{};
  auto const count = info->GetInstanceNum();
  if (!count || count > 256)
    return ge::GRAPH_FAILED;
  p.count = count;
  gert::Shape shape;
  int64_t dim;
  auto shapeAt = [c](uint32_t i) -> const gert::Shape * {
    auto const *input = c->GetDynamicInputShape(0, i);
    return input ? &input->GetStorageShape() : nullptr;
  };
  if (!ConcatenatedShape(p.count, attrs->GetAttrPointer<int64_t>(0), shapeAt,
                         shape, dim))
    return ge::GRAPH_FAILED;
  int64_t const rank = shape.GetDimNum();
  uint64_t inner;
  switch (desc->GetDataType()) {
  case ge::DT_FLOAT:
  case ge::DT_INT32:
    inner = 4;
    break;
  case ge::DT_FLOAT16:
    inner = 2;
    break;
  case ge::DT_INT8:
    inner = 1;
    break;
  default:
    return ge::GRAPH_FAILED;
  }
  // 将拼接转为按字节处理的二维布局：外层行数、各输入行宽和输出行宽。
  p.rows = 1;
  for (int64_t d = 0; (d < rank); ++d) {
    if ((d < dim) && (!Product(p.rows, shape.GetDim(d))))
      return ge::GRAPH_FAILED;
    if ((d > dim) && (!Product(inner, shape.GetDim(d))))
      return ge::GRAPH_FAILED;
  }
  uint64_t widest = 0;
  for (uint32_t i = 0; (i < p.count); ++i) {
    auto *input = c->GetDynamicInputShape(0, i);
    auto *type = c->GetDynamicInputDesc(0, i);
    if (!input || !type || type->GetDataType() != desc->GetDataType())
      return ge::GRAPH_FAILED;
    auto const &s = input->GetStorageShape();
    p.width[i] = inner;
    if (!Product(p.width[i], s.GetDim(dim)) ||
        p.width[i] > UINT64_MAX - p.rowBytes)
      return ge::GRAPH_FAILED;
    p.rowBytes += p.width[i];
    widest = std::max(widest, p.width[i]);
  }
  p.total = p.rowBytes;
  if (!Product(p.total, p.rows))
    return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  if (ub < 8192)
    return ge::GRAPH_FAILED;
  p.tile = std::min<uint64_t>(65536, (ub - 8192) / 2 / 32 * 32);

  // 按 32 字节边界分配连续行，避免不同核写入同一输出块。
  p.quantum = 1;
  while ((p.rowBytes % 32) * p.quantum % 32)
    p.quantum *= 2;
  bool const rows =
      p.rows >= std::max<uint64_t>(p.quantum, platform.GetCoreNumAiv()) &&
      widest <= p.tile && p.rowBytes <= UINT32_MAX;
  uint64_t key = rows ? 2 : 1;
  if (inner == 4 &&
      (desc->GetDataType() == ge::DT_FLOAT ||
       desc->GetDataType() == ge::DT_INT32) &&
      p.rows >= 256 && p.count >= 32 && widest <= 1024 && p.total >= 4194304 &&
      DeriveNarrowResources(p, ub, platform.GetCoreNumAiv())) {
    key = 3;
  }
  uint64_t const units =
      rows ? (p.rows + p.quantum - 1) / p.quantum : (p.total + 31) / 32;
  p.cores = std::max<uint64_t>(
      1, std::min<uint64_t>(platform.GetCoreNumAiv(), units));
  if (key == 3) {
    uint64_t const batches = (p.rows + p.quantum - 1) / p.quantum;
    uint64_t const perCore = (batches + p.cores - 1) / p.cores;
    p.cores = (batches + perCore - 1) / perCore;
  }
  auto *data = c->GetRawTilingData();
  auto *workspace = c->GetWorkspaceSizes(1);
  if (!data || !workspace || data->GetCapacity() < sizeof(p))
    return ge::GRAPH_FAILED;
  workspace[0] = 0;
  if (key == 3 && data->GetCapacity() >= sizeof(SwingPlan)) {
    SwingPlan swing{};
    if (MakeSwingPlan(p, ub, platform.GetCoreNumAiv(), swing)) {
      ShapeTablePlan shaped{};
      if (data->GetCapacity() >= sizeof(shaped) && EncodeShapeTable(swing, shaped)) {
        std::memcpy(data->GetData(), &shaped, sizeof(shaped));
        data->SetDataSize(sizeof(shaped));
        c->SetBlockDim(swing.p.cores); c->SetTilingKey(12);
        return ge::GRAPH_SUCCESS;
      }
      std::memcpy(data->GetData(), &swing, sizeof(swing));
      data->SetDataSize(sizeof(swing));
      c->SetBlockDim(swing.p.cores); c->SetTilingKey(11);
      return ge::GRAPH_SUCCESS;
    }
  }

  // 根据行宽、输入数和 UB 容量选择批量搬运方案，未命中时使用通用路径。
  if (AffineRowSchedule(p, c, ub))
    return ge::GRAPH_SUCCESS;
  if (key == 2 && PhasePlan(p, c, ub))
    return ge::GRAPH_SUCCESS;
  std::vector<SmallMove> small;
  if (p.total && p.total <= 65536 && widest <= 4096) {
    uint32_t prefix = 0;
    for (uint32_t i = 0; (i < p.count); ++i) {
      uint32_t const width = p.width[i];
      if (!width)
        continue;
      uint32_t const batch = std::min(255U, 4096 / ((width + 31) / 32 * 32));
      for (uint32_t row = 0; (row < p.rows); row += batch)
        small.push_back(
            {row * width, static_cast<uint32_t>(row * p.rowBytes + prefix),
             static_cast<uint16_t>(width),
             static_cast<uint8_t>(std::min<uint64_t>(batch, p.rows - row)),
             static_cast<uint8_t>(i),
             static_cast<uint32_t>(p.rowBytes - width)});
      prefix += width;
    }
    if (small.size() <= platform.GetCoreNumAiv()) {
      key = 4;
      p.cores = small.size();
    }
  }
  if (key == 4) {
    std::memcpy(data->GetData(), small.data(),
                small.size() * sizeof(SmallMove));
    data->SetDataSize(small.size() * sizeof(SmallMove));
  } else {
    std::memcpy(data->GetData(), &p, sizeof(p));
    data->SetDataSize(sizeof(p));
  }
  c->SetBlockDim(p.cores);
  c->SetTilingKey(key);
  if ((key == 1 || key == 2) && ub >= 188416 &&
      !Specialized(p, key == 2, c, ub))
    ScheduleMoves(p, key == 2, c);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus Infer(gert::InferShapeContext *c) {
  auto const *info = c->GetComputeNodeInfo()->GetInputInstanceInfo(0);
  auto *out = c->GetOutputShape(0);
  auto *first = c->GetDynamicInputShape(0, 0);
  if (!info || !out || !first || !c->GetAttrs())
    return ge::GRAPH_FAILED;
  int64_t dim;
  bool const valid = ConcatenatedShape(
      info->GetInstanceNum(), c->GetAttrs()->GetAttrPointer<int64_t>(0),
      [c](uint32_t i) { return c->GetDynamicInputShape(0, i); }, *out, dim);
  return valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}
namespace ops {
class Concat : public OpDef {
  static ge::graphStatus InferElementType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return ge::GRAPH_SUCCESS;
  }

public:
  explicit Concat(const char *name) : OpDef(name) {
    TensorContract(Input("inputs").ParamType(DYNAMIC).AutoContiguous());
    TensorContract(Output("output").ParamType(REQUIRED));
    this->Attr("dim").AttrType(OPTIONAL).Int(0);
    SetInferShape(Infer).SetInferDataType(InferElementType);
    this->AICore().SetTiling(Tile).AddConfig("ascend910b");
  }
};
OP_ADD(Concat);
}
