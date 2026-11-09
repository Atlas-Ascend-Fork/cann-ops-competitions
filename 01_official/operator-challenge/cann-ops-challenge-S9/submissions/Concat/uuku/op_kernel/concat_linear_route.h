#pragma once
#include <cstdint>

// Host/kernel contract for linear DMA admission. The descriptor is independent
// of the earlier row/group routes; it is consumed only if they decline.
namespace concat_linear_route {
constexpr uint32_t kBulkDmaFlag = 0x80000000U;
constexpr uint32_t kSingleInputFlag = 0x40000000U;
constexpr uint32_t kInputIndexMask = 0x0000FFFFU;
constexpr uint32_t kMinBulkBytes = 1024U;
constexpr uint32_t kWorkBlockBytes = 1024U;
} // namespace concat_linear_route
