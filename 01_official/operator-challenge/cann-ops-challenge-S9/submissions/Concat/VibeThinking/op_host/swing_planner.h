#pragma once
#include "../op_kernel/concat_layout.h"
#include <algorithm>
#include <vector>

inline bool MakeSwingPlan(const compact::Plan &input, uint64_t ub, uint32_t cores,
                          compact::SwingPlan &out) {
  using namespace compact;
  if (input.rows < 1024 || input.rows > UINT32_MAX || input.count < 64 ||
      input.count > 256 || cores < 8 || cores > 64 || input.rowBytes > UINT32_MAX)
    return false;
  struct Item { uint32_t index, bytes; };
  std::vector<Item> active;
  uint32_t largest = 0;
  for (uint32_t i = 0; i < input.count; ++i) {
    if (input.width[i] > UINT32_MAX || input.width[i] % 4) return false;
    if (input.width[i]) {
      active.push_back({i,static_cast<uint32_t>(input.width[i])});
      largest = std::max(largest,static_cast<uint32_t>(input.width[i]));
    }
  }
  const uint32_t groups = cores / 8;
  if (active.size() < groups) return false;
  uint32_t lo = largest, hi = input.rowBytes;
  while (lo < hi) {
    const uint32_t cap = lo + (hi - lo) / 2;
    uint32_t used = 1, bytes = 0;
    for (const auto &x : active) {
      if (bytes + x.bytes > cap) { ++used; bytes = 0; }
      bytes += x.bytes;
    }
    if (used <= groups) hi = cap; else lo = cap + 1;
  }
  struct Group { uint32_t first,last,bytes,prefix; };
  std::vector<Group> parts;
  uint32_t at = 0, prefix = 0, peak = 0;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint32_t start = at, stop = active.size() - (groups - g - 1);
    uint32_t bytes = 0;
    while (at < stop && bytes + active[at].bytes <= lo) bytes += active[at++].bytes;
    if (at == start) return false;
    parts.push_back({active[start].index,active[at-1].index+1,bytes,prefix});
    peak = std::max(peak,bytes); prefix += bytes;
  }
  if (at != active.size() || prefix != input.rowBytes) return false;
  constexpr uint32_t W = kWriteRows;
  const uint32_t pitch = (peak + 255) / 256 * 256;
  const uint64_t fixed = 512 + 2112 + (input.count * 8 + 31) / 32 * 32 + 256;
  if (ub <= fixed + 4ULL * W * pitch || W * pitch > 255 * 256) return false;
  const uint32_t R = kReadRows;
  if (fixed + uint64_t(R) * peak + 4ULL * W * pitch > ub) return false;
  out.p = input; out.p.cores = groups * 8; out.p.quantum = R;
  out.p.tile = R * peak; out.p.reserved = W * pitch;
  const uint32_t bands = (input.rows + R - 1) / R;
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t owner = 0; owner < 8; ++owner) {
      auto &job = out.jobs[8*g+owner]; const auto &part = parts[g];
      job.first = part.first; job.last = part.last; job.prefix = part.prefix;
      job.begin = uint64_t(bands) * owner / 8 * R;
      job.end = std::min<uint64_t>(input.rows,uint64_t(bands) * (owner+1) / 8 * R);
    }
  }
  return true;
}

inline bool EncodeShapeTable(const compact::SwingPlan &s, compact::ShapeTablePlan &out) {
  using namespace compact;
  if (s.p.rowBytes % 4 || s.p.rowBytes / 4 > kShapeColumns - 64 ||
      s.p.tile / 4 > 32767 || s.p.quantum != kReadRows) return false;
  for (uint32_t i=0;i<s.p.count;++i)
    if (s.p.width[i] % 4 || s.p.width[i] > 512) return false;
  out = {}; out.swing = s;
  for (uint32_t c=0;c<s.p.cores;++c) {
    const auto &job=s.jobs[c]; uint32_t column=0;
    for (uint32_t i=job.first;i<job.last;++i) {
      const uint32_t width=s.p.width[i]/4;
      for (uint32_t x=0;x<width;++x) {
        const uint32_t pos=job.prefix/4+column+x;
        out.firstRow[pos]=column*kReadRows+x;
        out.stepMinusOne[pos]=static_cast<int8_t>(width-1);
      }
      column+=width;
    }
  }
  return true;
}
