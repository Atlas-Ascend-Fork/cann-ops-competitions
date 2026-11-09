#pragma once
#include "../op_kernel/concat_layout.h"

inline bool DeriveNarrowResources(compact::Plan &p, uint64_t ub,
                                  uint32_t cores) {
  if (!p.count || p.count > 256 || !cores)
    return false;
  uint32_t period = 1;
  uint64_t widest = 0;
  for (uint32_t i = 0; i < p.count; ++i) {
    if (p.width[i] % 4 || p.width[i] > UINT32_MAX)
      return false;
    while (period * p.width[i] % 32)
      period *= 2;
    if (widest < p.width[i])
      widest = p.width[i];
  }
  if (!widest)
    return false;
  uint32_t const batch = period;
  uint64_t const fixed = 512 + (sizeof(compact::Plan) + 31) / 32 * 32 +
                         (p.count * 8 + 31) / 32 * 32 + 256;
  if (ub <= fixed)
    return false;
  uint64_t pitchLimit = (ub - fixed) / (5ULL * batch + 1) / 256 * 256;
  uint64_t const repeatLimit = (255U * 256U / batch) / 256 * 256;
  if (pitchLimit > repeatLimit)
    pitchLimit = repeatLimit;
  if (!pitchLimit || widest > pitchLimit)
    return false;
  uint32_t maxBytes = 0, maxPitch = 0;
  for (uint32_t first = 0; first < p.count;) {
    uint32_t last = first, bytes = 0;
    while (last < p.count && bytes + p.width[last] <= pitchLimit)
      bytes += static_cast<uint32_t>(p.width[last++]);
    if (last == first)
      return false;
    uint32_t const pitch = (bytes + 255) / 256 * 256;
    if (maxBytes < bytes)
      maxBytes = bytes;
    if (maxPitch < pitch)
      maxPitch = pitch;
    first = last;
  }
  uint64_t const inputSlot = static_cast<uint64_t>(batch) * maxBytes;
  uint64_t const outputSlot = static_cast<uint64_t>(batch) * maxPitch;
  if (2 * inputSlot + 3 * outputSlot + maxPitch + fixed > ub ||
      inputSlot > UINT32_MAX || outputSlot > 255U * 256U)
    return false;
  p.quantum = batch;
  p.tile = static_cast<uint32_t>(inputSlot);
  p.reserved = static_cast<uint32_t>(outputSlot);
  return true;
}
