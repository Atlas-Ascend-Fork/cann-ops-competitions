// Pure geometry shared by host/kernel reasoning. Column/lane/count fields
// use B32 elements; Fragment::sourceByteBase alone is a byte offset in UB.
// Span records the intersection between a core-owned interval and an output band.
#pragma once
#include <cstdint>
#ifndef CONCAT_STREAM_INLINE
#define CONCAT_STREAM_INLINE inline
#endif
namespace concat_input_stream_geometry {
struct Fragment { uint32_t alignedColumn, lane, count, sourceByteBase; };
CONCAT_STREAM_INLINE Fragment MakeFragment(uint32_t inputColumn, uint32_t remaining,
                                            uint32_t bandColumn, uint32_t bandWidth) {
    const uint32_t lane = bandColumn % 8U;
    uint32_t count = remaining < bandWidth - bandColumn ? remaining : bandWidth - bandColumn;
    count = count < 64U - lane ? count : 64U - lane;
    return {bandColumn - lane, lane, count, (8U + inputColumn - lane) * 4U};
}
struct Span { uint32_t first, rows, ubFirst, head, tail; };
CONCAT_STREAM_INLINE Span MakeSpan(uint32_t firstRow, uint32_t lastRow,
                                   uint32_t firstCol, uint32_t lastCol,
                                   uint32_t left, uint32_t width) {
    const uint32_t right = left + width;
    const uint32_t first = firstRow + (right <= firstCol ? 1U : 0U);
    const uint32_t last = lastRow + 1U - (left >= lastCol ? 1U : 0U);
    const uint32_t head = first == firstRow && firstCol > left ? firstCol - left : 0U;
    const uint32_t tail = last - 1U == lastRow && lastCol < right ? lastCol - left : width;
    return {first, last - first, first - firstRow, head, tail};
}
} // namespace concat_input_stream_geometry
