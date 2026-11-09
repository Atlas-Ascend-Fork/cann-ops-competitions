
#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
constexpr uint32_t kMaxRank = 8;
enum SquareSumTilingKey : uint32_t {
    kKeySquareOnly = 1,
    kKeyAllReduce = 2,
    kKeyTailReduce = 3,
    kKeyHeadReduce = 4,
    kKeyMiddleReduceAligned = 5,
    kKeyMiddleReduceFallback = 6,
    kKeyMiddleReduceOuterBlock = 7,
    kKeySeparatedFast = 8,
    kKeySeparatedFallback = 9,
    kKeySeparatedInnerVector = 10,
    kKeySeparatedTailVector = 11,
    kKeyBf16TailGroupedPipeline = 12,
    kKeyBf16TailHalfAccum = 13,
    kKeyBf16TailDenseGather = 14,
    kKeyBf16MiddleOuterBanked = 15,
    kKeyBf16MiddlePhaseRepeat = 16,
    kKeyHeadFp32OffsetPrefix = 17,
    kKeyHeadFp32Width518Phase = 18,
    kKeyHeadFp32Width519Reverse = 19,
    kKeyHeadFp32Width520Aligned = 20,
    kKeyHeadFp32Width521Forward = 21,
    kKeySeparatedFp16BalancedFold = 22,
    kKeySeparatedFp16WaveFold = 23,
    kKeySeparatedFp16AlignedWindow = 24,
};
constexpr uint32_t kUbReserveBytes = 16 * 1024;
constexpr uint32_t kMaxTileElems = 16384;
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kPartialStride = 8;
constexpr uint32_t kSmallTailRowsPerCore = 16;
constexpr uint32_t kSmallTailSingleCoreTotal = 2048;
constexpr uint32_t kMaxTailRowGroup = 255;
constexpr uint32_t kSeparatedFp16LayoutNone = 0;
constexpr uint32_t kSeparatedFp16LayoutFloat = 1;
constexpr uint32_t kSeparatedFp16LayoutHalf = 2;
constexpr uint32_t kSeparatedFp16LayoutHalfShort = 3;
constexpr uint32_t kSeparatedFp16LayoutGm512 = 1U << 8;
constexpr uint32_t kSeparatedFp16CycleRows = 16;
constexpr uint32_t kSeparatedFp16HalfTileRows = 8;
constexpr uint32_t kSeparatedFp16MaxRows = 255;
constexpr uint32_t kSeparatedFp16D0Batch = 2;
constexpr uint32_t kSeparatedFp16InputGuardBytes = 256;
constexpr uint32_t kSeparatedFp16Gm512InputGuardBytes = 1024;
constexpr uint32_t kSeparatedFp16FloatWorkBytes =
    kSeparatedFp16CycleRows * 64U * sizeof(float);
constexpr uint64_t kSeparatedFp16UbReserveBytes = 8U * 1024U;
constexpr uint32_t kSeparatedFp16HostBytesPerTileElem = 20;
constexpr uint32_t kBf16MiddlePhasePrimaryRows = 128;
constexpr uint32_t kBf16MiddlePhaseSecondaryRows = 128;
constexpr uint32_t kBf16MiddlePhaseConvertRows = 32;
constexpr uint32_t kBf16MiddlePhaseAccBanks = 16;
constexpr uint32_t kBf16MiddlePhaseRawPrimaryRows = 160;
constexpr uint32_t kBf16MiddlePhaseRawSecondaryRows = 96;
constexpr uint32_t kBf16MiddlePhaseRawConvertRows = 32;
constexpr uint32_t kBf16MiddlePhaseRawAccBanks = 16;
constexpr uint32_t kBf16MiddlePhaseRawOutputRows = 4;
constexpr uint32_t kBf16MiddlePhaseOffsetBanks = 4;
constexpr uint32_t kBf16MiddlePhaseInputGuardBytes = 544;
constexpr uint32_t kBf16MiddlePhaseGuardBytes = 32;
constexpr uint64_t kBf16MiddlePhaseUbReserveBytes = 8U * 1024U;
constexpr uint32_t kHeadFp32Width519Packet0Elems = 23040;
constexpr uint32_t kHeadFp32Width519Packet1StreamElems = 20224;
constexpr uint32_t kHeadFp32Width519Packet1EpochElems = 19200;
constexpr uint32_t kHeadFp32Width519EpochPairs = 96;
constexpr uint32_t kHeadFp32Width518Packet0Elems = 23040;
constexpr uint32_t kHeadFp32Width518Packet1StreamElems = 20224;
constexpr uint32_t kHeadFp32Width518Packet1EpochElems = 19200;
constexpr uint32_t kHeadFp32Width518EpochPairs = 96;
constexpr uint32_t kHeadFp32Width521Packet0StreamElems = 23040;
constexpr uint32_t kHeadFp32Width521Packet0EpochElems = 23040;
constexpr uint32_t kHeadFp32Width521Packet1StreamElems = 20736;
constexpr uint32_t kHeadFp32Width521Packet1EpochElems = 19200;
constexpr uint32_t kHeadFp32Width521EpochPairs = 96;
constexpr uint32_t kHeadFp32Width521StreamPartialStride = 640;
constexpr uint32_t kHeadFp32AccPhaseBytes = 64;
constexpr uint32_t kHeadFp32AccBanks = 8;
constexpr uint32_t kHeadFp32InputGuardBytes = 32;
constexpr uint32_t kHeadFp32AccGuardBytes = 32;
constexpr uint32_t kHeadFp32HostBytesPerTileElem = 28;
constexpr uint32_t kHeadFp32PacketUbReclaimBytes = 12 * 1024;
constexpr uint32_t kHeadFp32AlignedUbReclaimBytes = 16 * 1024;
constexpr uint32_t kHeadFp32AlignedInnerSize = 520;
constexpr uint32_t kHeadFp32AlignedEpochBuffers = 1;

uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    return ((value + align - 1) / align) * align;
}

uint64_t CalcHeadFp32WidthPacketUbBytes(
    uint32_t inner_size, uint32_t packet0_elems,
    uint32_t packet1_elems, uint32_t epoch_packet1_elems,
    uint32_t stream_aux_buffers = 1U)
{
    const uint64_t compact_bytes = AlignUpU64(
        static_cast<uint64_t>(inner_size) * sizeof(float), 32U);
    const uint64_t aux_buffers =
        packet1_elems == epoch_packet1_elems
            ? 3U
            : stream_aux_buffers;
    return kHeadFp32AccPhaseBytes +
           AlignUpU64(
               static_cast<uint64_t>(packet0_elems) * sizeof(float) +
                   kHeadFp32InputGuardBytes,
               32U) +
           AlignUpU64(
               static_cast<uint64_t>(packet1_elems) * sizeof(float) +
                   kHeadFp32InputGuardBytes,
               32U) +
           AlignUpU64(
               static_cast<uint64_t>(kHeadFp32AccBanks) * inner_size *
                       sizeof(float) +
                   kHeadFp32AccGuardBytes,
               32U) +
           aux_buffers * compact_bytes;
}

uint64_t CalcHeadFp32Width520UbBytes(uint32_t host_tile_elems)
{
    const uint64_t usable_bytes =
        static_cast<uint64_t>(host_tile_elems) *
            kHeadFp32HostBytesPerTileElem +
        kHeadFp32AlignedUbReclaimBytes;
    const uint64_t fixed_bytes =
        static_cast<uint64_t>(1U + kHeadFp32AccBanks) *
            kHeadFp32AlignedInnerSize * sizeof(float);
    const uint64_t row_bytes =
        static_cast<uint64_t>(kBufferNum) *
        kHeadFp32AlignedInnerSize * sizeof(float);
    uint64_t rows = usable_bytes > fixed_bytes
                        ? (usable_bytes - fixed_bytes) / row_bytes
                        : kHeadFp32AccBanks;
    rows = std::min<uint64_t>(rows, kMaxTailRowGroup);
    rows = (rows / kHeadFp32AccBanks) * kHeadFp32AccBanks;
    rows = std::max<uint64_t>(rows, kHeadFp32AccBanks);
    return static_cast<uint64_t>(kBufferNum) * rows *
               kHeadFp32AlignedInnerSize * sizeof(float) +
           static_cast<uint64_t>(
               kHeadFp32AccBanks + kHeadFp32AlignedEpochBuffers) *
               kHeadFp32AlignedInnerSize * sizeof(float);
}

uint32_t SelectHeadFp32Width519Packet1Elems(
    uint32_t total_size, uint32_t partial_count)
{
    if (total_size == 0U || partial_count == 0U) {
        return kHeadFp32Width519Packet1StreamElems;
    }
    const uint32_t stream_pair_elems =
        kHeadFp32Width519Packet0Elems +
        kHeadFp32Width519Packet1StreamElems;
    const uint32_t pair_count =
        (total_size - 1U) / stream_pair_elems + 1U;
    const uint32_t max_owned_pairs =
        (pair_count - 1U) / partial_count + 1U;
    return max_owned_pairs > kHeadFp32Width519EpochPairs
               ? kHeadFp32Width519Packet1EpochElems
               : kHeadFp32Width519Packet1StreamElems;
}

uint32_t SelectHeadFp32Width518Packet1Elems(
    uint32_t total_size, uint32_t partial_count)
{
    if (total_size == 0U || partial_count == 0U) {
        return kHeadFp32Width518Packet1StreamElems;
    }
    const uint32_t stream_pair_elems =
        kHeadFp32Width518Packet0Elems +
        kHeadFp32Width518Packet1StreamElems;
    const uint32_t pair_count =
        (total_size - 1U) / stream_pair_elems + 1U;
    const uint32_t max_owned_pairs =
        (pair_count - 1U) / partial_count + 1U;
    return max_owned_pairs > kHeadFp32Width518EpochPairs
               ? kHeadFp32Width518Packet1EpochElems
               : kHeadFp32Width518Packet1StreamElems;
}

uint32_t SelectHeadFp32Width521Packet1Elems(
    uint32_t total_size, uint32_t partial_count)
{
    if (total_size == 0U || partial_count == 0U) {
        return kHeadFp32Width521Packet1StreamElems;
    }
    const uint32_t stream_pair_elems =
        kHeadFp32Width521Packet0StreamElems +
        kHeadFp32Width521Packet1StreamElems;
    const uint32_t pair_count =
        (total_size - 1U) / stream_pair_elems + 1U;
    const uint32_t max_owned_pairs =
        (pair_count - 1U) / partial_count + 1U;
    return max_owned_pairs > kHeadFp32Width521EpochPairs
               ? kHeadFp32Width521Packet1EpochElems
               : kHeadFp32Width521Packet1StreamElems;
}

uint32_t SelectHeadFp32Width521Packet0Elems(uint32_t packet1_elems)
{
    return packet1_elems == kHeadFp32Width521Packet1StreamElems
               ? kHeadFp32Width521Packet0StreamElems
               : kHeadFp32Width521Packet0EpochElems;
}

uint64_t CalcBf16MiddlePhaseUbBytes(uint32_t inner_size, bool raw_square)
{
    const uint32_t primary_rows = raw_square
        ? kBf16MiddlePhaseRawPrimaryRows : kBf16MiddlePhasePrimaryRows;
    const uint32_t secondary_rows = raw_square
        ? kBf16MiddlePhaseRawSecondaryRows : kBf16MiddlePhaseSecondaryRows;
    const uint32_t convert_rows = raw_square
        ? kBf16MiddlePhaseRawConvertRows : kBf16MiddlePhaseConvertRows;
    const uint32_t acc_banks = raw_square
        ? kBf16MiddlePhaseRawAccBanks : kBf16MiddlePhaseAccBanks;
    const uint64_t compact_stride =
        AlignUpU64(
            static_cast<uint64_t>(inner_size) * sizeof(float), 32U) /
        sizeof(float);
    const uint64_t primary_input_bytes = AlignUpU64(
        static_cast<uint64_t>(primary_rows) *
                inner_size * sizeof(uint16_t) +
            kBf16MiddlePhaseInputGuardBytes,
        32U);
    const uint64_t secondary_input_bytes = AlignUpU64(
        static_cast<uint64_t>(secondary_rows) *
                inner_size * sizeof(uint16_t) +
            kBf16MiddlePhaseInputGuardBytes,
        32U);
    return primary_input_bytes + secondary_input_bytes +
           (raw_square ? kBf16MiddlePhaseRawOutputRows : 2ULL) * AlignUpU64(
                      static_cast<uint64_t>(inner_size) * sizeof(uint16_t),
                      32U) +
           AlignUpU64(
               static_cast<uint64_t>(acc_banks) *
                       inner_size * sizeof(float) +
                   kBf16MiddlePhaseGuardBytes,
               32U) +
           AlignUpU64(
               static_cast<uint64_t>(convert_rows) *
                       inner_size * sizeof(float) +
                   kBf16MiddlePhaseGuardBytes,
               32U) +
           AlignUpU64(
               static_cast<uint64_t>(kBf16MiddlePhaseOffsetBanks) *
                   compact_stride * sizeof(uint32_t),
               32U);
}

uint32_t ProductDims(const uint32_t *dims, uint32_t begin, uint32_t end)
{
    uint32_t prod = 1;
    for (uint32_t i = begin; i < end; ++i) {
        prod *= dims[i];
    }
    return prod;
}

bool IsAxisReduced(uint32_t axis_mask, uint32_t axis)
{
    return ((axis_mask >> axis) & 1U) != 0;
}

uint32_t ChooseCoreNum(uint32_t work_items, uint32_t aiv_num)
{
    if (aiv_num == 0 || work_items <= 1024) {
        return 1;
    }
    uint32_t core_num = work_items < aiv_num ? work_items : aiv_num;
    return core_num == 0 ? 1 : core_num;
}

uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint32_t ChooseTailReduceCoreNum(uint32_t total_size, uint32_t output_size, uint32_t row_group, uint32_t aiv_num)
{
    if (aiv_num == 0 || output_size <= 1 || total_size <= kSmallTailSingleCoreTotal || row_group <= 1) {
        return 1;
    }
    uint32_t core_num = CeilDivU32(output_size, kSmallTailRowsPerCore);
    core_num = std::min(core_num, aiv_num);
    core_num = std::min(core_num, output_size);
    return core_num == 0 ? 1 : core_num;
}

uint32_t ChooseContinuousReduceCoreNum(uint32_t total_size, uint32_t output_size, uint32_t aiv_num)
{
    if (aiv_num == 0 || output_size <= 1 || total_size <= 2048) {
        return 1;
    }
    uint32_t core_num = CeilDivU32(output_size, 64);
    core_num = std::min(core_num, aiv_num);
    core_num = std::min(core_num, output_size);
    return core_num == 0 ? 1 : core_num;
}

uint32_t ChooseHeadReduceCoreNum(uint32_t total_size, uint32_t reduce_size, uint32_t inner_size,
                                 uint32_t tile_elems, uint32_t aiv_num)
{
    if (aiv_num == 0 || inner_size <= 1 || total_size <= 2048) {
        return 1;
    }
    if (reduce_size >= 4096 && inner_size <= tile_elems) {
        uint32_t core_num = std::min(16U, aiv_num);
        core_num = std::min(core_num, inner_size);
        return core_num == 0 ? 1 : core_num;
    }
    return ChooseContinuousReduceCoreNum(total_size, inner_size, aiv_num);
}

uint32_t ChooseOuterBlockCoreNum(uint32_t outer_size, uint32_t aiv_num)
{
    if (aiv_num == 0 || outer_size <= 1) {
        return 1;
    }
    uint32_t core_num = std::min(outer_size, aiv_num);
    return core_num == 0 ? 1 : core_num;
}

uint32_t ChooseSeparatedCoreNum(uint32_t total_size, uint32_t output_size, uint32_t aiv_num)
{
    if (aiv_num == 0 || output_size <= 1 || total_size <= 2048) {
        return 1;
    }
    uint32_t core_num = std::min(output_size, aiv_num);
    return core_num == 0 ? 1 : core_num;
}

std::vector<int32_t> NormalizeAxes(const gert::RuntimeAttrs *attrs, uint32_t rank)
{
    std::vector<int32_t> axes;
    if (attrs == nullptr) {
        return axes;
    }
    auto axis_attr = attrs->GetListInt(0);
    if (axis_attr == nullptr) {
        return axes;
    }
    for (size_t i = 0; i < axis_attr->GetSize(); ++i) {
        int64_t axis = static_cast<int64_t>(axis_attr->GetData()[i]);
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            axes.clear();
            axes.push_back(-1);
            return axes;
        }
        axes.push_back(static_cast<int32_t>(axis));
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

bool GetKeepDims(const gert::RuntimeAttrs *attrs)
{
    if (attrs == nullptr || attrs->GetBool(1) == nullptr) {
        return false;
    }
    return *attrs->GetBool(1);
}

uint32_t CalcTileElems(uint64_t ub_size, uint32_t dtype_bytes)
{
    if (ub_size == 0) {
        ub_size = 192 * 1024;
    }
    uint64_t usable_ub = ub_size > kUbReserveBytes ? ub_size - kUbReserveBytes : ub_size;
    uint64_t bytes_per_elem = static_cast<uint64_t>(2 * kBufferNum * dtype_bytes + 3 * sizeof(float));
    uint32_t align_elems = 32 / dtype_bytes;
    uint64_t raw_tile = bytes_per_elem == 0 ? align_elems : usable_ub / bytes_per_elem;
    if (raw_tile > kMaxTileElems) {
        raw_tile = kMaxTileElems;
    }
    uint32_t tile = static_cast<uint32_t>(raw_tile);
    tile = (tile / align_elems) * align_elems;
    return tile < align_elems ? align_elems : tile;
}

uint32_t CalcTailRowGroup(uint32_t reduce_size, uint32_t tile_elems, uint32_t dtype_bytes)
{
    if (reduce_size == 0) {
        return 1;
    }
    uint32_t input_align_elems = 32 / dtype_bytes;
    uint32_t fp32_align_elems = 32 / sizeof(float);
    uint32_t input_stride = AlignUpU32(reduce_size, input_align_elems);
    uint32_t fp32_stride = AlignUpU32(reduce_size, fp32_align_elems);
    if (input_stride == 0 || fp32_stride == 0) {
        return 1;
    }
    if ((input_stride * dtype_bytes) / 32 > 255 || (fp32_stride * sizeof(float)) / 32 > 255) {
        return 1;
    }
    uint32_t group = std::min(tile_elems / input_stride, tile_elems / fp32_stride);
    group = std::min(group, kMaxTailRowGroup);
    return group > 1 ? group : 1;
}

uint32_t CalcBf16TailGroupedTileElems(uint64_t ub_size,
                                   uint32_t dtype_bytes)
{
    if (ub_size == 0) {
        ub_size = 192 * 1024;
    }
    constexpr uint64_t reserve_bytes = 24 * 1024;
    uint64_t usable_bytes =
        ub_size > reserve_bytes ? ub_size - reserve_bytes : ub_size;
    uint64_t bytes_per_elem =
        2ULL * dtype_bytes + 2ULL * sizeof(float);
    uint64_t raw =
        bytes_per_elem == 0 ? 0 : usable_bytes / bytes_per_elem;
    raw = std::min<uint64_t>(raw, kMaxTileElems);
    uint32_t align_elems = 32 / dtype_bytes;
    uint32_t elems = static_cast<uint32_t>(raw);
    elems = (elems / align_elems) * align_elems;
    return elems < align_elems ? align_elems : elems;
}

uint32_t CalcBf16TailDenseTileElems(
    uint64_t ub_size, uint32_t reduce_size,
    uint32_t dtype_bytes)
{
    constexpr uint64_t runtime_ub_limit = 196352U;
    if (ub_size == 0) {
        ub_size = runtime_ub_limit;
    }
    ub_size = std::min<uint64_t>(ub_size, runtime_ub_limit);
    if (dtype_bytes != sizeof(uint16_t) || reduce_size == 0) {
        return 0;
    }
    constexpr uint32_t tile_rows = 24;
    constexpr uint32_t primary_rows = 32;
    constexpr uint32_t secondary_rows = 16;
    constexpr uint32_t output_pair_waves = 7;
    constexpr uint32_t output_rows =
        (output_pair_waves + 1U) * (primary_rows + secondary_rows);
    constexpr uint32_t scratch_rows = 16;
    constexpr uint32_t tail_stride = 16;
    constexpr uint32_t input_guard_bytes = 544;
    constexpr uint64_t runtime_reserve_bytes = 20 * 1024;
    const uint32_t padded_stride =
        AlignUpU32(reduce_size, 32U / dtype_bytes);
    // Sixteen-row group starts have BF16 phase zero within each DMA tile.
    const uint32_t scratch_elems = AlignUpU32(
        scratch_rows * reduce_size + scratch_rows, 16U);
    const uint64_t live_bytes =
        AlignUpU32(
            primary_rows * reduce_size * dtype_bytes + input_guard_bytes,
            32U) +
        AlignUpU32(
            secondary_rows * reduce_size * dtype_bytes + input_guard_bytes,
            32U) +
        2ULL * AlignUpU32(output_rows * dtype_bytes, 32U) +
        AlignUpU32(scratch_elems * sizeof(float), 32U) +
        AlignUpU32(
            2U * scratch_rows * tail_stride * sizeof(float),
            32U) +
        AlignUpU32(
            2U * scratch_rows * tail_stride * sizeof(float),
            32U) +
        AlignUpU32(primary_rows * sizeof(float), 32U) +
        AlignUpU32(scratch_rows * sizeof(float), 32U);
    return live_bytes + runtime_reserve_bytes <= ub_size
               ? tile_rows * padded_stride
               : 0U;
}

struct SeparatedFp16Layout {
    uint32_t mode = kSeparatedFp16LayoutNone;
    uint32_t rows = kSeparatedFp16CycleRows;
};

uint64_t CalcSeparatedFp16LayoutBytes(
    uint32_t tail_size, uint32_t rows, uint32_t mode)
{
    const bool use_gm512 =
        (mode & kSeparatedFp16LayoutGm512) != 0U;
    mode &= ~kSeparatedFp16LayoutGm512;
    const bool use_half =
        mode == kSeparatedFp16LayoutHalf ||
        mode == kSeparatedFp16LayoutHalfShort;
    const uint32_t align_elems = use_half ? 16U : 8U;
    const uint32_t low_phase_limit = use_half ? 7U : 3U;
    const uint32_t tail_mod = tail_size % align_elems;

    uint32_t head_guard = 0;
    uint32_t tail_guard = 0;
    uint32_t extra_count = 0;
    if (tail_mod <= low_phase_limit) {
        const uint32_t phase_delta = tail_mod;
        tail_guard = phase_delta * kSeparatedFp16CycleRows;
        extra_count = phase_delta * (kSeparatedFp16CycleRows - 1U);
    } else {
        const uint32_t phase_delta = align_elems - tail_mod;
        head_guard = phase_delta * kSeparatedFp16CycleRows;
        tail_guard = (phase_delta + 1U) * kSeparatedFp16CycleRows;
        extra_count = head_guard + phase_delta;
    }
    const uint32_t accumulator_bytes =
        use_half ? sizeof(uint16_t) : sizeof(float);
    const uint32_t extra_stride =
        AlignUpU32(extra_count * accumulator_bytes, 32U) /
        accumulator_bytes;
    const uint64_t dense_capacity =
        static_cast<uint64_t>(rows) * tail_size;
    const uint64_t aligned_dense_bytes =
        ((dense_capacity * sizeof(uint16_t) + 31ULL) / 32ULL) *
        32ULL;
    const uint64_t input_slot_bytes = use_gm512
        ? aligned_dense_bytes * kSeparatedFp16D0Batch +
              kSeparatedFp16Gm512InputGuardBytes
        : ((kSeparatedFp16D0Batch * aligned_dense_bytes +
            kSeparatedFp16InputGuardBytes + 31ULL) /
           32ULL) *
              32ULL;
    const uint64_t output_slot_bytes =
        AlignUpU32(rows * sizeof(uint16_t), 32U);
    const uint64_t acc_count =
        static_cast<uint64_t>(head_guard) + dense_capacity +
        tail_guard;
    const uint64_t acc_bytes =
        ((acc_count * accumulator_bytes + 31ULL) / 32ULL) *
        32ULL;
    const uint64_t sum_slot_bytes =
        AlignUpU32(rows * sizeof(float), 32U);
    if (use_half) {
        uint64_t work_bytes = AlignUpU32(rows * sizeof(uint16_t), 32U);
        if (tail_size > 255U) {
            const uint32_t chunks = (tail_size + 15U + 127U) / 128U;
            const uint32_t partial_stride = AlignUpU32(chunks, 8U);
            const uint32_t partial_count = rows * partial_stride;
            work_bytes = AlignUpU32(partial_count * sizeof(uint16_t), 32U) +
                         AlignUpU32(partial_count * sizeof(float), 32U);
        }
        return kBufferNum * input_slot_bytes +
               kBufferNum * output_slot_bytes + acc_bytes +
               work_bytes + 2ULL * sum_slot_bytes;
    }
    const uint64_t offsets_bytes =
        AlignUpU32(
            kSeparatedFp16CycleRows * extra_stride * sizeof(uint32_t),
            32U);
    return kBufferNum * input_slot_bytes +
           kBufferNum * output_slot_bytes + acc_bytes +
           offsets_bytes + kSeparatedFp16FloatWorkBytes +
           2ULL * sum_slot_bytes;
}

uint32_t CalcSeparatedFp16LayoutRows(
    uint32_t tail_size, uint32_t output_size,
    uint64_t usable_bytes, uint32_t mode,
    uint32_t row_align)
{
    uint32_t rows = std::min(output_size, kSeparatedFp16MaxRows);
    rows = (rows / row_align) * row_align;
    if (rows < row_align) {
        rows = row_align;
    }
    while (rows > row_align &&
           CalcSeparatedFp16LayoutBytes(tail_size, rows, mode) >
               usable_bytes) {
        rows -= row_align;
    }
    while (rows > 1U &&
           CalcSeparatedFp16LayoutBytes(tail_size, rows, mode) >
               usable_bytes) {
        rows /= 2U;
    }
    if (CalcSeparatedFp16LayoutBytes(tail_size, rows, mode) > usable_bytes) {
        return 0U;
    }
    return rows;
}

SeparatedFp16Layout ExpandSeparatedFp16AlignedHalfLayout(
    SeparatedFp16Layout layout, uint32_t tail_size,
    uint32_t output_size, uint64_t expanded_bytes)
{
    if (layout.mode != kSeparatedFp16LayoutHalf &&
        layout.mode != kSeparatedFp16LayoutHalfShort) {
        return layout;
    }
    const uint32_t aligned_rows = CalcSeparatedFp16LayoutRows(
        tail_size, output_size, expanded_bytes,
        layout.mode, kSeparatedFp16CycleRows);
    if (aligned_rows > layout.rows) {
        layout.rows = aligned_rows;
    }
    return layout;
}

SeparatedFp16Layout SelectSeparatedFp16Layout(
    uint32_t tail_size, uint32_t output_size,
    uint64_t conservative_bytes,
    uint64_t expanded_bytes)
{
    const uint32_t half_tail_mod = tail_size % 16U;
    const uint32_t half_phase_delta =
        half_tail_mod <= 7U ? half_tail_mod
                            : 16U - half_tail_mod;
    if (half_phase_delta <= 5U) {
        const uint32_t mode =
            half_phase_delta == 5U
                ? kSeparatedFp16LayoutHalfShort
                : kSeparatedFp16LayoutHalf;
        SeparatedFp16Layout layout;
        layout.mode = mode;
        layout.rows = CalcSeparatedFp16LayoutRows(
            tail_size, output_size, conservative_bytes,
            mode, kSeparatedFp16HalfTileRows);
        return ExpandSeparatedFp16AlignedHalfLayout(
            layout, tail_size, output_size,
            expanded_bytes);
    }

    const uint32_t float_rows = CalcSeparatedFp16LayoutRows(
        tail_size, output_size, conservative_bytes,
        kSeparatedFp16LayoutFloat, kSeparatedFp16CycleRows);
    if (half_phase_delta <= 7U) {
        const uint32_t half_rows = CalcSeparatedFp16LayoutRows(
            tail_size, output_size, conservative_bytes,
            kSeparatedFp16LayoutHalf, kSeparatedFp16HalfTileRows);
        if (half_rows * 3U >= float_rows * 4U) {
            SeparatedFp16Layout layout;
            layout.mode = kSeparatedFp16LayoutHalf;
            layout.rows = half_rows;
            return ExpandSeparatedFp16AlignedHalfLayout(
                layout, tail_size, output_size,
                expanded_bytes);
        }
    }
    SeparatedFp16Layout layout;
    layout.mode = kSeparatedFp16LayoutFloat;
    layout.rows = float_rows;
    return layout;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SquareSumV1TilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto shape = x1_shape->GetStorageShape();
    uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    if (rank > kMaxRank) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t strides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t total_size = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        dims[i] = static_cast<uint32_t>(shape.GetDim(i));
        total_size *= dims[i];
    }
    if (rank > 0) {
        strides[rank - 1] = 1;
        for (int32_t i = static_cast<int32_t>(rank) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }

    std::vector<int32_t> axes = NormalizeAxes(context->GetAttrs(), rank);
    if (!axes.empty() && axes[0] < 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t axis_mask = 0;
    for (int32_t axis : axes) {
        axis_mask |= (1U << static_cast<uint32_t>(axis));
    }
    uint32_t axis_count = static_cast<uint32_t>(axes.size());
    uint32_t output_size = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (!IsAxisReduced(axis_mask, i)) {
            output_size *= dims[i];
        }
    }
    if (axis_count == 0) {
        output_size = total_size;
    }

    auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();

    ge::DataType input_dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtype_bytes = input_dtype == ge::DT_FLOAT ? 4 : 2;
    uint32_t tile_elems = CalcTileElems(ub_size, dtype_bytes);
    const uint32_t head_fp32_host_tile_elems = tile_elems;

    uint32_t selected_key = kKeySeparatedFallback;
    uint32_t outer_size = 1;
    uint32_t reduce_size = 1;
    uint32_t inner_size = 1;
    uint32_t row_group = 1;
    uint32_t reduce_axis0 = axis_count > 0 ? static_cast<uint32_t>(axes[0]) : 0;
    uint32_t reduce_axis1 = axis_count > 1 ? static_cast<uint32_t>(axes[1]) : reduce_axis0;

    if (axis_count == 0) {
        selected_key = kKeySquareOnly;
        outer_size = total_size;
        reduce_size = 1;
        inner_size = 1;
    } else if (axis_count == rank) {
        selected_key = kKeyAllReduce;
        outer_size = 1;
        reduce_size = total_size;
        inner_size = 1;
    } else {
        bool contiguous = true;
        for (uint32_t i = 1; i < axis_count; ++i) {
            if (axes[i] != axes[i - 1] + 1) {
                contiguous = false;
                break;
            }
        }
        if (contiguous) {
            uint32_t first_axis = static_cast<uint32_t>(axes.front());
            uint32_t last_axis = static_cast<uint32_t>(axes.back());
            outer_size = ProductDims(dims, 0, first_axis);
            reduce_size = ProductDims(dims, first_axis, last_axis + 1);
            inner_size = ProductDims(dims, last_axis + 1, rank);
            if (inner_size == 1) {
                selected_key = kKeyTailReduce;
                row_group = CalcTailRowGroup(reduce_size, tile_elems, dtype_bytes);
            } else if (first_axis == 0) {
                selected_key = kKeyHeadReduce;
            } else if ((inner_size * dtype_bytes) % 32 == 0) {
                selected_key = kKeyMiddleReduceAligned;
                uint64_t block_elems = static_cast<uint64_t>(reduce_size) * inner_size;
                if (block_elems <= tile_elems && (inner_size * sizeof(float)) % 32 == 0) {
                    selected_key = kKeyMiddleReduceOuterBlock;
                }
            } else {
                selected_key = kKeyMiddleReduceFallback;
            }
        } else {
            reduce_size = 1;
            for (int32_t axis : axes) {
                reduce_size *= dims[static_cast<uint32_t>(axis)];
            }
            outer_size = output_size;
            inner_size = 1;
            bool last_axis_reduced = rank > 0 && IsAxisReduced(axis_mask, rank - 1);
            bool fast_4d_tail_reduce =
                rank == 4 && axis_count == 2 && IsAxisReduced(axis_mask, 3);
            if (!last_axis_reduced) {
                uint32_t last_reduce_axis = static_cast<uint32_t>(axes.back());
                inner_size = ProductDims(dims, last_reduce_axis + 1, rank);
                selected_key = kKeySeparatedInnerVector;
            } else if (fast_4d_tail_reduce) {
                inner_size = dims[rank - 1];
                selected_key = kKeySeparatedFast;
            } else {
                inner_size = dims[rank - 1];
                selected_key = kKeySeparatedTailVector;
            }
        }
    }

    const bool fp32_wide_head_reduce_split =
        selected_key == kKeyHeadReduce && input_dtype == ge::DT_FLOAT &&
        inner_size >= 513 && inner_size <= 1024 && reduce_size > 8192 &&
        (inner_size % 128) != 0;
    if (fp32_wide_head_reduce_split) {
        if (inner_size == 518U) {
            selected_key = kKeyHeadFp32Width518Phase;
        } else if (inner_size == 519U) {
            selected_key = kKeyHeadFp32Width519Reverse;
        } else if (inner_size == 520U) {
            selected_key = kKeyHeadFp32Width520Aligned;
        } else if (inner_size == 521U) {
            selected_key = kKeyHeadFp32Width521Forward;
        } else {
            selected_key = kKeyHeadFp32OffsetPrefix;
        }
    }
    const bool use_direct_small_fp16_tail =
        selected_key == kKeyTailReduce &&
        input_dtype == ge::DT_FLOAT16 && reduce_size <= 32U &&
        total_size <= 4096U;
    const bool use_dense_bf16_tail_slab =
        selected_key == kKeyTailReduce &&
        input_dtype == ge::DT_BF16 && rank > 0U &&
        axis_count == 1U &&
        axis_mask == (1U << (rank - 1U)) &&
        reduce_size == 1025U &&
        output_size >= std::max(1U, aiv_num) * 96U;
    if (selected_key == kKeyTailReduce && !use_direct_small_fp16_tail) {
        tile_elems = use_dense_bf16_tail_slab
                         ? CalcBf16TailDenseTileElems(
                               ub_size, reduce_size, dtype_bytes)
                         : CalcBf16TailGroupedTileElems(
                               ub_size, dtype_bytes);
        if (use_dense_bf16_tail_slab && tile_elems == 0U) {
            return ge::GRAPH_FAILED;
        }
        row_group =
            CalcTailRowGroup(reduce_size, tile_elems, dtype_bytes);
        if (use_dense_bf16_tail_slab) {
            selected_key = kKeyBf16TailDenseGather;
        } else if (row_group > 1) {
            selected_key =
                input_dtype == ge::DT_BF16 && reduce_size <= 128U
                    ? kKeyBf16TailHalfAccum
                    : kKeyBf16TailGroupedPipeline;
        }
    }

    const uint32_t bf16_middle_acc_stride =
        AlignUpU32(inner_size * sizeof(float), 32) / sizeof(float);
    const bool bf16_middle_outer_banked =
        selected_key == kKeyMiddleReduceFallback && outer_size >= 8 &&
        static_cast<uint64_t>(9) * bf16_middle_acc_stride <=
            static_cast<uint64_t>(2) * tile_elems;
    const uint64_t bf16_middle_dense_usable_bytes =
        static_cast<uint64_t>(tile_elems) * 20U;
    const uint64_t bf16_middle_dense_fixed_bytes =
        2ULL * AlignUpU32(inner_size * sizeof(uint16_t), 32) +
        AlignUpU32(32U * bf16_middle_acc_stride * sizeof(float), 32) +
        3U * 32U;
    const uint64_t bf16_middle_dense_bytes_per_row =
        static_cast<uint64_t>(inner_size) *
            (kBufferNum * sizeof(uint16_t) + sizeof(float)) +
        2ULL * bf16_middle_acc_stride * sizeof(float);
    uint32_t bf16_middle_dense_rows = 0;
    if (bf16_middle_dense_usable_bytes > bf16_middle_dense_fixed_bytes &&
        bf16_middle_dense_bytes_per_row > 0) {
        bf16_middle_dense_rows = static_cast<uint32_t>(std::min<uint64_t>(
            reduce_size,
            (bf16_middle_dense_usable_bytes - bf16_middle_dense_fixed_bytes) /
                bf16_middle_dense_bytes_per_row));
        bf16_middle_dense_rows = std::min(bf16_middle_dense_rows, 255U);
        if (bf16_middle_dense_rows >= 32U) {
            bf16_middle_dense_rows = (bf16_middle_dense_rows / 32U) * 32U;
        }
    }
    uint32_t bf16_middle_dense_elems = bf16_middle_dense_rows * inner_size;
    uint32_t bf16_middle_dense_padded_elems =
        bf16_middle_dense_rows * bf16_middle_acc_stride;
    uint64_t bf16_middle_dense_live_bytes =
        2ULL * AlignUpU32(
                   bf16_middle_dense_elems * sizeof(uint16_t) + 32U, 32) +
        2ULL * AlignUpU32(inner_size * sizeof(uint16_t), 32) +
        AlignUpU32(32U * bf16_middle_acc_stride * sizeof(float), 32) +
        AlignUpU32(bf16_middle_dense_elems * sizeof(float) + 32U, 32) +
        2ULL * AlignUpU32(
                   bf16_middle_dense_padded_elems * sizeof(float), 32);
    while (bf16_middle_dense_rows > 0 &&
           bf16_middle_dense_live_bytes > bf16_middle_dense_usable_bytes) {
        bf16_middle_dense_rows =
            bf16_middle_dense_rows > 32U ? bf16_middle_dense_rows - 32U : 0U;
        bf16_middle_dense_elems = bf16_middle_dense_rows * inner_size;
        bf16_middle_dense_padded_elems =
            bf16_middle_dense_rows * bf16_middle_acc_stride;
        bf16_middle_dense_live_bytes =
            2ULL * AlignUpU32(
                       bf16_middle_dense_elems * sizeof(uint16_t) + 32U, 32) +
            2ULL * AlignUpU32(inner_size * sizeof(uint16_t), 32) +
            AlignUpU32(32U * bf16_middle_acc_stride * sizeof(float), 32) +
            AlignUpU32(bf16_middle_dense_elems * sizeof(float) + 32U, 32) +
            2ULL * AlignUpU32(
                       bf16_middle_dense_padded_elems * sizeof(float), 32);
    }
    const bool bf16_middle_dense_gather =
        bf16_middle_outer_banked && input_dtype == ge::DT_BF16 &&
        bf16_middle_dense_rows >= 32U &&
        (bf16_middle_acc_stride * sizeof(float)) / 32 <= 255 &&
        bf16_middle_dense_live_bytes <= bf16_middle_dense_usable_bytes;
    const uint64_t bf16_middle_physical_ub_bytes =
        ub_size == 0U ? 192U * 1024U : ub_size;
    const uint64_t bf16_middle_phase_usable_bytes =
        bf16_middle_physical_ub_bytes > kBf16MiddlePhaseUbReserveBytes
            ? bf16_middle_physical_ub_bytes - kBf16MiddlePhaseUbReserveBytes
            : 0U;
    const bool bf16_middle_packed_square =
        input_dtype == ge::DT_BF16 && inner_size == 255U &&
        reduce_size >= 1024U && outer_size >= 4U;
    const bool bf16_middle_phase_repeat =
        bf16_middle_dense_gather && inner_size == 255U &&
        CalcBf16MiddlePhaseUbBytes(inner_size, bf16_middle_packed_square) <=
            bf16_middle_phase_usable_bytes;
    if (selected_key == kKeyMiddleReduceFallback &&
        bf16_middle_outer_banked) {
        selected_key =
            bf16_middle_phase_repeat
                ? kKeyBf16MiddlePhaseRepeat
                : kKeyBf16MiddleOuterBanked;
    }

    const uint32_t separated_fp16_padded_tail =
        AlignUpU32(dims[3], 16U);
    const bool fp16_separated_dense_unaligned =
        selected_key == kKeySeparatedFast &&
        input_dtype == ge::DT_FLOAT16 && rank == 4 &&
        axis_mask == ((1U << 0) | (1U << 3)) &&
        output_size >= 2048 && dims[0] >= 8 &&
        dims[3] > 0 && dims[3] <= tile_elems &&
        (dims[3] % 16U) != 0 &&
        separated_fp16_padded_tail <= 2032;
    if (fp16_separated_dense_unaligned) {
        selected_key =
            dims[3] >= 113U && (dims[3] % 2U) == 1U
                ? kKeySeparatedFp16AlignedWindow
                : (dims[3] >= 201U
                       ? kKeySeparatedFp16WaveFold
                       : kKeySeparatedFp16BalancedFold);
    }
    const bool use_compact_fp16_wave =
        selected_key == kKeySeparatedFp16BalancedFold &&
        dims[0] >= 2U && dims[0] <= 32U &&
        (dims[0] % 2U) == 0U &&
        dims[3] > 0U && dims[3] <= 112U &&
        output_size >= 2048U;
    if (use_compact_fp16_wave) {
        constexpr uint64_t runtime_ub_limit = 196352U;
        constexpr uint64_t reserve_bytes = 3U * 1024U;
        const uint64_t physical_ub_bytes = std::min<uint64_t>(
            ub_size == 0U ? runtime_ub_limit : ub_size,
            runtime_ub_limit);
        const uint64_t usable_bytes = physical_ub_bytes > reserve_bytes
            ? physical_ub_bytes - reserve_bytes : 0U;
        constexpr uint32_t align_elems = 32U / sizeof(uint16_t);
        tile_elems = static_cast<uint32_t>(
            usable_bytes / kSeparatedFp16HostBytesPerTileElem);
        tile_elems = tile_elems / align_elems * align_elems;
    }
    SeparatedFp16Layout separated_fp16_layout;
    if (selected_key == kKeySeparatedFp16AlignedWindow) {
        const uint64_t physical_ub_bytes =
            ub_size == 0U ? 192U * 1024U : ub_size;
        const uint64_t separated_fp16_usable_bytes =
            physical_ub_bytes > kSeparatedFp16UbReserveBytes
                ? physical_ub_bytes - kSeparatedFp16UbReserveBytes
                : 0U;
        const uint64_t conservative_bytes =
            static_cast<uint64_t>(tile_elems) *
            kSeparatedFp16HostBytesPerTileElem;
        separated_fp16_layout = SelectSeparatedFp16Layout(
            dims[3], output_size, conservative_bytes,
            separated_fp16_usable_bytes);
        if (separated_fp16_layout.rows == 0U) {
            return ge::GRAPH_FAILED;
        }
        const bool gm512_logical_aligned =
            dims[0] >= kSeparatedFp16D0Batch &&
            separated_fp16_layout.rows % kSeparatedFp16CycleRows == 0U &&
            (static_cast<uint64_t>(strides[0]) *
             dtype_bytes) % 32ULL == 0ULL;
        const uint32_t gm512_mode =
            separated_fp16_layout.mode | kSeparatedFp16LayoutGm512;
        if (gm512_logical_aligned &&
            CalcSeparatedFp16LayoutBytes(
                dims[3], separated_fp16_layout.rows, gm512_mode) <=
                separated_fp16_usable_bytes) {
            separated_fp16_layout.mode = gm512_mode;
        }
    }

    uint32_t core_num = 1;
    if (fp32_wide_head_reduce_split) {
        const uint32_t head_fp32_core_cap = 48U;
        core_num = std::min(head_fp32_core_cap, aiv_num);
        core_num = std::min(core_num, reduce_size);
        if (core_num == 0) {
            core_num = 1;
        }
    } else if (selected_key == kKeyBf16MiddlePhaseRepeat) {
        constexpr uint32_t outputs_per_core = 4U;
        core_num = std::min(
            aiv_num,
            (outer_size + outputs_per_core - 1U) /
                outputs_per_core);
        if (core_num == 0) {
            core_num = 1;
        }
    } else if (selected_key == kKeyBf16MiddleOuterBanked) {
        core_num = std::min(outer_size, aiv_num);
        if (core_num == 0) {
            core_num = 1;
        }
    } else if (selected_key == kKeyAllReduce) {
        uint32_t tile_count = (total_size + tile_elems - 1) / tile_elems;
        core_num = total_size <= tile_elems * 2 ? 1 : ChooseCoreNum(tile_count, aiv_num);
    } else if (selected_key == kKeyTailReduce ||
               selected_key == kKeyBf16TailGroupedPipeline ||
               selected_key == kKeyBf16TailDenseGather ||
               selected_key == kKeyBf16TailHalfAccum) {
        core_num = ChooseTailReduceCoreNum(total_size, output_size, row_group, aiv_num);
    } else if (selected_key == kKeyMiddleReduceOuterBlock) {
        core_num = ChooseOuterBlockCoreNum(outer_size, aiv_num);
    } else if (selected_key == kKeyHeadReduce) {
        core_num = ChooseHeadReduceCoreNum(total_size, reduce_size, inner_size, tile_elems, aiv_num);
    } else if (selected_key == kKeyMiddleReduceAligned || selected_key == kKeyMiddleReduceFallback) {
        if (reduce_size == 64 && inner_size == 256) {
            core_num = 8;
        } else {
            core_num = ChooseContinuousReduceCoreNum(total_size, output_size, aiv_num);
        }
    } else if (selected_key == kKeySeparatedFast ||
               selected_key == kKeySeparatedFp16BalancedFold ||
               selected_key == kKeySeparatedFp16WaveFold ||
               selected_key == kKeySeparatedFp16AlignedWindow ||
               selected_key == kKeySeparatedInnerVector ||
               selected_key == kKeySeparatedTailVector) {
        if (selected_key == kKeySeparatedFast && rank == 4U &&
            dims[2] > 0U && dims[3] == 128U &&
            axis_mask == ((1U << 1) | (1U << 3))) {
            core_num = std::min(48U, aiv_num);
        } else {
            core_num = ChooseSeparatedCoreNum(total_size, output_size, aiv_num);
        }
    } else {
        uint32_t work_items = selected_key == kKeySquareOnly ? total_size : output_size;
        core_num = ChooseCoreNum(work_items, aiv_num);
    }

    if (selected_key == kKeyHeadFp32Width519Reverse) {
        tile_elems = SelectHeadFp32Width519Packet1Elems(
            total_size, core_num);
    } else if (selected_key == kKeyHeadFp32Width518Phase) {
        tile_elems = SelectHeadFp32Width518Packet1Elems(
            total_size, core_num);
    } else if (selected_key == kKeyHeadFp32Width521Forward) {
        tile_elems = SelectHeadFp32Width521Packet1Elems(
            total_size, core_num);
    }

    uint64_t head_fp32_required_ub_bytes = 0U;
    uint64_t head_fp32_host_ub_bytes = 0U;
    if (selected_key == kKeyHeadFp32Width518Phase) {
        head_fp32_required_ub_bytes = CalcHeadFp32WidthPacketUbBytes(
            518U, kHeadFp32Width518Packet0Elems, tile_elems,
            kHeadFp32Width518Packet1EpochElems);
    } else if (selected_key == kKeyHeadFp32Width519Reverse) {
        head_fp32_required_ub_bytes = CalcHeadFp32WidthPacketUbBytes(
            519U, kHeadFp32Width519Packet0Elems, tile_elems,
            kHeadFp32Width519Packet1EpochElems);
    } else if (selected_key == kKeyHeadFp32Width520Aligned) {
        head_fp32_required_ub_bytes =
            CalcHeadFp32Width520UbBytes(head_fp32_host_tile_elems);
    } else if (selected_key == kKeyHeadFp32Width521Forward) {
        head_fp32_required_ub_bytes = CalcHeadFp32WidthPacketUbBytes(
            521U, SelectHeadFp32Width521Packet0Elems(tile_elems), tile_elems,
            kHeadFp32Width521Packet1EpochElems, 0U);
    }
    if (head_fp32_required_ub_bytes != 0U) {
        const uint64_t reclaim_bytes =
            selected_key == kKeyHeadFp32Width520Aligned
                ? kHeadFp32AlignedUbReclaimBytes
                : kHeadFp32PacketUbReclaimBytes;
        head_fp32_host_ub_bytes =
            static_cast<uint64_t>(head_fp32_host_tile_elems) *
                kHeadFp32HostBytesPerTileElem +
            reclaim_bytes;
        const uint64_t physical_ub_bytes =
            ub_size == 0U ? 192U * 1024U : ub_size;
        if (head_fp32_required_ub_bytes >
            std::min(head_fp32_host_ub_bytes, physical_ub_bytes)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t workspace_offset = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = kPartialStride;
    size_t *current_workspace = context->GetWorkspaceSizes(1);
    if (current_workspace != nullptr) {
        current_workspace[0] = 0;
    }
    if (selected_key == kKeyAllReduce && core_num > 1) {
        workspace_offset = AlignUpU32(sys_workspace_size, 32);
        partial_count = core_num;
        if (current_workspace != nullptr) {
            current_workspace[0] = static_cast<size_t>(workspace_offset + partial_count * kPartialStride * sizeof(float));
        }
    } else if (selected_key == kKeyBf16MiddlePhaseRepeat &&
               core_num > 1 && outer_size % core_num != 0) {
        workspace_offset = AlignUpU32(sys_workspace_size, 32);
        partial_count = core_num;
        partial_stride = AlignUpU32(inner_size, 8);
        const uint64_t sync_bytes =
            static_cast<uint64_t>(partial_count) * 32U;
        if (current_workspace != nullptr) {
            current_workspace[0] = static_cast<size_t>(
                workspace_offset + sync_bytes +
                static_cast<uint64_t>(partial_count) *
                    partial_stride * sizeof(float));
        }
    } else if ((selected_key == kKeyHeadFp32OffsetPrefix ||
                selected_key == kKeyHeadFp32Width519Reverse ||
                selected_key == kKeyHeadFp32Width521Forward ||
                selected_key == kKeyHeadFp32Width520Aligned ||
                selected_key == kKeyHeadFp32Width518Phase) &&
               core_num > 1) {
        workspace_offset = AlignUpU32(sys_workspace_size, 32);
        partial_count = core_num;
        partial_stride =
            selected_key == kKeyHeadFp32Width521Forward &&
                    tile_elems == kHeadFp32Width521Packet1StreamElems
                ? kHeadFp32Width521StreamPartialStride
                : AlignUpU32(inner_size, 8);
        const uint64_t sync_bytes =
            static_cast<uint64_t>(partial_count) * 32U;
        if (current_workspace != nullptr) {
            current_workspace[0] = static_cast<size_t>(
                workspace_offset + sync_bytes +
                static_cast<uint64_t>(partial_count) *
                    partial_stride * sizeof(float));
        }
    }

    tiling.set_rank(rank);
    tiling.set_axis_count(axis_count);
    tiling.set_axis_mask(axis_mask);
    tiling.set_total_size(total_size);
    tiling.set_output_size(output_size);
    tiling.set_outer_size(outer_size);
    tiling.set_reduce_size(reduce_size);
    tiling.set_inner_size(inner_size);
    tiling.set_tile_elems(tile_elems);
    tiling.set_row_group(row_group);
    tiling.set_workspace_offset(workspace_offset);
    tiling.set_partial_count(partial_count);
    tiling.set_partial_stride(partial_stride);
    tiling.set_dtype_bytes(dtype_bytes);
    tiling.set_reduce_axis0(reduce_axis0);
    tiling.set_reduce_axis1(reduce_axis1);
    tiling.set_dims(dims);
    tiling.set_strides(strides);
    tiling.set_separated_fp16_layout_mode(separated_fp16_layout.mode);
    tiling.set_separated_fp16_layout_rows(separated_fp16_layout.rows);

    context->SetTilingKey(selected_key);
    context->SetBlockDim(core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    uint32_t rank = static_cast<uint32_t>(x1_shape->GetDimNum());
    if (rank > kMaxRank) {
        return GRAPH_FAILED;
    }
    std::vector<int32_t> axes = NormalizeAxes(context->GetAttrs(), rank);
    if (!axes.empty() && axes[0] < 0) {
        return GRAPH_FAILED;
    }
    uint32_t axis_mask = 0;
    for (int32_t axis : axes) {
        axis_mask |= (1U << static_cast<uint32_t>(axis));
    }
    bool keep_dims = GetKeepDims(context->GetAttrs());

    y_shape->SetDimNum(0);
    if (axes.empty()) {
        for (uint32_t i = 0; i < rank; ++i) {
            y_shape->AppendDim(x1_shape->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (IsAxisReduced(axis_mask, i)) {
            if (keep_dims) {
                y_shape->AppendDim(1);
            }
        } else {
            y_shape->AppendDim(x1_shape->GetDim(i));
        }
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        OpAICoreConfig aiv_config("ascend910b");
        aiv_config.ExtendCfgInfo("coreType.value", "VectorCore");
        this->AICore().AddConfig("ascend910b", aiv_config);

    }
};

OP_ADD(SquareSumV1);
}
