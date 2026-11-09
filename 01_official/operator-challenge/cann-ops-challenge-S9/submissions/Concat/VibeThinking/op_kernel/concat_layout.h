#pragma once
#include <stdint.h>

namespace compact {
constexpr uint32_t kReadRows = 16, kWriteRows = 2;
struct Plan {
  uint64_t rows, rowBytes, total, quantum;
  uint32_t count, tile, cores, reserved;
  uint64_t width[256];
};
struct SwingJob { uint32_t begin, end, prefix; uint16_t first, last; };
struct SwingPlan { Plan p; SwingJob jobs[64]; };
constexpr uint32_t kShapeColumns = 9216;
struct ShapeTablePlan {
  SwingPlan swing;
  int16_t firstRow[kShapeColumns];
  int8_t stepMinusOne[kShapeColumns];
};
static_assert(sizeof(ShapeTablePlan) <= 32768, "");
struct AffineRowPlan {
  uint32_t rows, rowBytes, pitch, tileRows;
  uint32_t count, cores, stageBytes, reserved;
  struct Part { uint32_t width, offset; } parts[16];
};
static_assert(sizeof(AffineRowPlan) == 160, "");
struct Move {
  uint64_t src, dst;
  uint32_t bytes, gap;
  uint16_t rows, input;
  uint32_t reserved;
};
struct Schedule {
  uint32_t tile, inputs, begin[65], reserved;
};
struct Transfer {
  uint32_t src, dst, bytes, gap;
  uint16_t rows, input;
  uint32_t ubOffset;
};
struct StreamSpan {
  uint32_t src, dst, bytes, input;
};
constexpr uint32_t kStreamTile = 65440;
constexpr uint32_t kStreamArena = 94208;
constexpr uint32_t kStreamUbBytes = 2 * kStreamArena + 4096;
constexpr uint32_t kPhaseArena = 192000;
constexpr uint32_t kPhaseUbBytes = kPhaseArena + 4096;
static_assert(sizeof(StreamSpan) == 16, "");
static_assert(sizeof(Transfer) == 24, "");
struct alignas(8) SmallMove {
  uint32_t src, dst;
  uint16_t bytes;
  uint8_t rows, input;
  uint32_t gap;
};
static_assert(sizeof(Plan) == 2096, "");
static_assert(sizeof(Move) == 32, "");
static_assert(sizeof(Schedule) == 272, "");
static_assert(sizeof(SmallMove) == 16, "");
}
