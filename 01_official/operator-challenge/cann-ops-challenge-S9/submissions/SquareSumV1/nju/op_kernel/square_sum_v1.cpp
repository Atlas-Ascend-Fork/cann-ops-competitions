#ifndef L2_CACHE_HINT
#define L2_CACHE_HINT
#endif
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN_BYTES = 32;
constexpr uint32_t PARTIAL_STRIDE = 8;
constexpr uint32_t MAX_REPEAT_ROWS = 255;
constexpr uint32_t MAX_FP32_REPEAT_MASK = 64;
constexpr uint32_t FP32_ALIGN_ELEMS = UB_ALIGN_BYTES / sizeof(float);
constexpr uint32_t HEAD_FP32_ACC_PHASE_BYTES = 64;

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return ((bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
}

template <typename T>
__aicore__ inline uint32_t PaddedStrideElems(uint32_t elems)
{
    return AlignUp32(elems * sizeof(T)) / sizeof(T);
}

template <typename T>
__aicore__ inline LocalTensor<T> GetHeadFp32AccView(
    TBuf<TPosition::VECOUT> &buf, uint32_t size,
    uint32_t logical_byte_offset = 0U)
{
    return buf.GetWithOffset<T>(
        size, HEAD_FP32_ACC_PHASE_BYTES + logical_byte_offset);
}

template <typename T>
__aicore__ inline uint32_t PaddedStrideBlocks(uint32_t elems)
{
    return PaddedStrideElems<T>(elems) * sizeof(T) / UB_ALIGN_BYTES;
}

__aicore__ inline void GetBlockRange(uint32_t total, uint32_t &start, uint32_t &end)
{
    uint32_t block_idx = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t base = total / block_num;
    uint32_t tail = total - base * block_num;
    start = block_idx * base + MinU32(block_idx, tail);
    end = start + base + (block_idx < tail ? 1 : 0);
}

__aicore__ inline void GetAlignedGroupRange(uint32_t total, uint32_t group_rows,
                                             uint32_t part_idx, uint32_t part_num,
                                             uint32_t &start, uint32_t &end)
{
    const uint32_t groups = total / group_rows;
    const uint32_t base = groups / part_num;
    const uint32_t tail = groups - base * part_num;
    const uint32_t group_start = part_idx * base + MinU32(part_idx, tail);
    const uint32_t group_end = group_start + base + (part_idx < tail ? 1U : 0U);
    start = group_start * group_rows;
    end = group_end * group_rows;
    if (part_idx + 1U == part_num) {
        end = total;
    }
}

__aicore__ inline uint32_t GetSplitSubblockTileOwner(
    uint32_t partial_count)
{
    const uint32_t block_idx =
        static_cast<uint32_t>(GetBlockIdx());
    if ((partial_count & 1U) != 0U ||
        static_cast<uint32_t>(GetBlockNum()) != partial_count) {
        return block_idx;
    }
    const uint32_t physical_count = partial_count / 2U;
    return block_idx / 2U +
           (block_idx & 1U) * physical_count;
}

__aicore__ inline bool IsReducedAxis(uint32_t axis_mask, uint32_t axis)
{
    return ((axis_mask >> axis) & 1U) != 0;
}

template <typename T>
__aicore__ inline float ToFloatScalar(T value)
{
    return static_cast<float>(value);
}

template <>
__aicore__ inline float ToFloatScalar<bfloat16_t>(bfloat16_t value)
{
    return ToFloat(value);
}

template <typename T>
__aicore__ inline T FromFloatScalar(float value)
{
    return static_cast<T>(value);
}

template <>
__aicore__ inline bfloat16_t FromFloatScalar<bfloat16_t>(float value)
{
    return ToBfloat16(value);
}

template <typename T>
__aicore__ inline bool CanDirectCopy(uint32_t offset, uint32_t count)
{
    return ((offset * sizeof(T)) % UB_ALIGN_BYTES == 0) && ((count * sizeof(T)) % UB_ALIGN_BYTES == 0);
}

template <typename T>
__aicore__ inline void CopyInContiguous(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset, uint32_t count)
{
    if (CanDirectCopy<T>(offset, count)) {
        DataCopy(dst, src[offset], count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad_params{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad_params);
    }
}

template <>
__aicore__ inline void CopyInContiguous<bfloat16_t>(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src,
                                                    uint32_t offset, uint32_t count)
{
    if (CanDirectCopy<bfloat16_t>(offset, count)) {
        DataCopy(dst, src[offset], count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, ToBfloat16(0.0f)};
        DataCopyPad(dst, src[offset], params, pad_params);
    }
}

template <typename T>
__aicore__ inline void CopyInPaddedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset, uint32_t row_elems,
                                        uint32_t rows)
{
    uint32_t total_elems = row_elems * rows;
    if ((row_elems * sizeof(T)) % UB_ALIGN_BYTES == 0 && CanDirectCopy<T>(offset, total_elems)) {
        DataCopy(dst, src[offset], total_elems);
        return;
    }
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(row_elems * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad_params{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(dst, src[offset], params, pad_params);
}

template <typename T>
__aicore__ inline uint32_t CopyInPaddedRowsAlignedGM(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset, uint32_t row_elems,
                                                     uint32_t rows)
{
    uint32_t total_elems = row_elems * rows;
    uint32_t align_unit = UB_ALIGN_BYTES / sizeof(T);
    uint32_t aligned_offset = (offset / align_unit) * align_unit;
    uint32_t head_skip = offset - aligned_offset;

    if ((row_elems % align_unit) == 0) {
        uint32_t fetch_count = total_elems + head_skip;
        uint32_t aligned_fetch_elems = AlignUp32(fetch_count * sizeof(T)) / sizeof(T);
        DataCopy(dst, src[aligned_offset], aligned_fetch_elems);
        return head_skip;
    }
    CopyInPaddedRows(dst, src, offset, row_elems, rows);
    return 0;
}

template <>
__aicore__ inline void CopyInPaddedRows<bfloat16_t>(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src,
                                                    uint32_t offset, uint32_t row_elems, uint32_t rows)
{
    uint32_t total_elems = row_elems * rows;
    if ((row_elems * sizeof(bfloat16_t)) % UB_ALIGN_BYTES == 0 && CanDirectCopy<bfloat16_t>(offset, total_elems)) {
        DataCopy(dst, src[offset], total_elems);
        return;
    }
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(row_elems * sizeof(bfloat16_t)), 0, 0,
                             0};
    DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, ToBfloat16(0.0f)};
    DataCopyPad(dst, src[offset], params, pad_params);
}

template <typename T>
__aicore__ inline void CopyOutContiguous(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t offset, uint32_t count)
{
    if (CanDirectCopy<T>(offset, count)) {
        DataCopy(dst[offset], src, count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst[offset], src, params);
    }
}

template <uint32_t InnerSize>
__aicore__ inline void MergeHeadFp32ColumnPartials(
    TQue<TPosition::VECIN, BUFFER_NUM> &input_queue,
    GlobalTensor<float> partial_gm, GlobalTensor<float> output_gm,
    uint32_t partial_count, uint32_t partial_stride)
{
    constexpr uint32_t kElemsPerBlock =
        UB_ALIGN_BYTES / sizeof(float);
    constexpr uint32_t kBlocksPerMergeCore = 8U;
    const uint32_t total_blocks =
        (InnerSize + kElemsPerBlock - 1U) / kElemsPerBlock;
    const uint32_t merge_core_count =
        (total_blocks + kBlocksPerMergeCore - 1U) /
        kBlocksPerMergeCore;
    const uint32_t core_idx =
        static_cast<uint32_t>(GetBlockIdx());
    const uint32_t base_blocks = total_blocks / merge_core_count;
    const uint32_t extra_cores =
        total_blocks - base_blocks * merge_core_count;
    for (uint32_t merge_idx = core_idx;
         merge_idx < merge_core_count;
         merge_idx += partial_count) {
        const uint32_t block_start =
            merge_idx * base_blocks + MinU32(merge_idx, extra_cores);
        const uint32_t block_count =
            base_blocks + (merge_idx < extra_cores ? 1U : 0U);
        const uint32_t row_elems = block_count * kElemsPerBlock;
        const uint32_t row_blocks = block_count;

        LocalTensor<float> partials = input_queue.AllocTensor<float>();
        DataCopyParams copy_params{
            static_cast<uint16_t>(partial_count),
            static_cast<uint16_t>(block_count),
            static_cast<uint16_t>(
                partial_stride / kElemsPerBlock - block_count),
            0};
        DataCopy(
            partials, partial_gm[block_start * kElemsPerBlock],
            copy_params);
        input_queue.EnQue(partials);
        partials = input_queue.DeQue<float>();

        SetMaskNorm();
        SetVectorMask<float>(row_elems);
        for (uint32_t step = 1U;
             step < partial_count; step <<= 1U) {
            const uint32_t group = step << 1U;
            const uint32_t repeat_count =
                (partial_count + step - 1U) / group;
            const uint32_t repeat_stride =
                repeat_count > 1U ? group * row_blocks : 0U;
            BinaryRepeatParams repeat_params{
                1, 1, 1,
                static_cast<uint8_t>(repeat_stride),
                static_cast<uint8_t>(repeat_stride),
                static_cast<uint8_t>(repeat_stride)};
            Add<float, false>(
                partials, partials,
                partials[step * row_elems],
                MASK_PLACEHOLDER,
                static_cast<uint8_t>(repeat_count),
                repeat_params);
            PipeBarrier<PIPE_V>();
        }
        ResetMask();

        const uint32_t output_start =
            block_start * kElemsPerBlock;
        const uint32_t output_elems =
            MinU32(row_elems, InnerSize - output_start);
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        CopyOutContiguous(
            output_gm, partials,
            output_start, output_elems);
        input_queue.FreeTensor(partials);
    }
}

template <typename T>
__aicore__ inline void SquareAddToAccum(LocalTensor<float> acc, LocalTensor<T> src, LocalTensor<float> tmp,
                                        uint32_t count)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(tmp, src, RoundMode::CAST_NONE, count);
        MulAddDst(acc, tmp, tmp, count);
    } else {
        MulAddDst(acc, src, src, count);
    }
}

template <typename T>
__aicore__ inline void SquareToFloat(LocalTensor<float> dst, LocalTensor<T> src, LocalTensor<float> tmp,
                                     uint32_t count)
{
    Duplicate(dst, 0.0f, count);
    SquareAddToAccum(dst, src, tmp, count);
}

template <typename T>
__aicore__ inline void SquarePaddedRowsToFloat(LocalTensor<float> dst, LocalTensor<T> src, LocalTensor<float> tmp,
                                               uint32_t row_elems, uint32_t rows, uint32_t input_stride,
                                               uint32_t float_stride)
{
    if (input_stride == row_elems && float_stride == row_elems) {
        SquareToFloat(dst, src, tmp, rows * row_elems);
        return;
    }
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        for (uint32_t r = 0; r < rows; ++r) {
            SquareToFloat(dst[r * float_stride], src[r * input_stride], tmp, row_elems);
        }
    } else {
        if (row_elems > MAX_FP32_REPEAT_MASK || PaddedStrideBlocks<T>(row_elems) > 255 ||
            PaddedStrideBlocks<float>(row_elems) > 255) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareToFloat(dst[r * float_stride], src[r * input_stride], tmp, row_elems);
            }
            return;
        }
        Duplicate(dst, 0.0f, rows * float_stride);
        BinaryRepeatParams repeat_params(1, 1, 1,
                                         static_cast<uint8_t>((float_stride * sizeof(float)) / UB_ALIGN_BYTES),
                                         static_cast<uint8_t>((input_stride * sizeof(T)) / UB_ALIGN_BYTES),
                                         static_cast<uint8_t>((input_stride * sizeof(T)) / UB_ALIGN_BYTES));
        MulAddDst(dst, src, src, static_cast<uint64_t>(row_elems), static_cast<uint8_t>(rows), repeat_params);
    }
}

template <typename T>
__aicore__ inline void SquareAccumulatePaddedRowsToFloat(LocalTensor<float> dst, LocalTensor<T> src, LocalTensor<float> tmp,
                                                         uint32_t row_elems, uint32_t rows, uint32_t input_stride,
                                                         uint32_t float_stride)
{
    if (input_stride == row_elems && float_stride == row_elems) {
        SquareAddToAccum(dst, src, tmp, rows * row_elems);
        return;
    }
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        for (uint32_t r = 0; r < rows; ++r) {
            SquareAddToAccum(dst[r * float_stride], src[r * input_stride], tmp, row_elems);
        }
    } else {
        if (row_elems > MAX_FP32_REPEAT_MASK || PaddedStrideBlocks<T>(row_elems) > 255 ||
            PaddedStrideBlocks<float>(row_elems) > 255) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(dst[r * float_stride], src[r * input_stride], tmp, row_elems);
            }
            return;
        }
        BinaryRepeatParams repeat_params(1, 1, 1,
                                         static_cast<uint8_t>((float_stride * sizeof(float)) / UB_ALIGN_BYTES),
                                         static_cast<uint8_t>((input_stride * sizeof(T)) / UB_ALIGN_BYTES),
                                         static_cast<uint8_t>((input_stride * sizeof(T)) / UB_ALIGN_BYTES));
        MulAddDst(dst, src, src, static_cast<uint64_t>(row_elems), static_cast<uint8_t>(rows), repeat_params);
    }
}

__aicore__ inline float ReduceFloatTile(LocalTensor<float> src, LocalTensor<float> work, LocalTensor<float> sum,
                                        uint32_t count)
{
    if (count == 1) {
        return src.GetValue(0);
    }
    if (count <= MAX_FP32_REPEAT_MASK) {
        WholeReduceSum<float>(sum, src, static_cast<int32_t>(count), 1, 1, 1, 8);
        return sum.GetValue(0);
    }
    ReduceSum<float>(sum, src, work, static_cast<int32_t>(count));
    return sum.GetValue(0);
}

template <typename T>
class KernelSquareOnly {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        total_size = tiling.total_size;
        tile_elems = tiling.tile_elems;
        input_gm.SetGlobalBuffer((__gm__ T *)input, total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, total_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(total_size, start, end);
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        for (uint32_t offset = start; offset < end;) {
            uint32_t cur = MinU32(tile_elems, end - offset);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, offset, cur);
            input_queue.EnQue(input_local);
            input_local = input_queue.DeQue<T>();

            LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            SquareToFloat(square, input_local, tmp, cur);

            LocalTensor<T> output_local = output_queue.AllocTensor<T>();
            if constexpr (std::is_same<T, float>::value) {
                Adds(output_local, square, 0.0f, cur);
            } else {
                Cast(output_local, square, RoundMode::CAST_RINT, cur);
            }
            input_queue.FreeTensor(input_local);
            output_queue.EnQue(output_local);
            output_local = output_queue.DeQue<T>();
            CopyOutContiguous(output_gm, output_local, offset, cur);
            output_queue.FreeTensor(output_local);
            offset += cur;
        }
    }

private:
    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t total_size = 0;
    uint32_t tile_elems = 0;
};

template <typename T>
class KernelAllReduce {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const SquareSumV1TilingData &tiling)
    {
        total_size = tiling.total_size;
        tile_elems = tiling.tile_elems;
        workspace_offset = tiling.workspace_offset;
        partial_count = tiling.partial_count;
        input_gm.SetGlobalBuffer((__gm__ T *)input, total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, 1);
        if (partial_count > 1) {
            partial_gm.SetGlobalBuffer((__gm__ float *)(workspace + workspace_offset),
                                       partial_count * PARTIAL_STRIDE);
        }
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, AlignUp32(PARTIAL_STRIDE * sizeof(T)));
        pipe.InitBuffer(partial_queue, BUFFER_NUM, AlignUp32(PARTIAL_STRIDE * sizeof(float)));
        pipe.InitBuffer(calc_buf, 3 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(total_size, start, end);
        float acc = ReduceInputRange(start, end);
        if (partial_count > 1) {
            WritePartial(acc);
            SyncAll();
            if (GetBlockIdx() == 0) {
                float total = 0.0f;
                for (uint32_t i = 0; i < partial_count; ++i) {
                    total += partial_gm.GetValue(i * PARTIAL_STRIDE);
                }
                WriteScalarOutput(0, total);
            }
        } else {
            WriteScalarOutput(0, acc);
        }
    }

private:
    __aicore__ inline float ReduceInputRange(uint32_t start, uint32_t end)
    {
        float acc = 0.0f;
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        for (uint32_t offset = start; offset < end;) {
            uint32_t cur = MinU32(tile_elems, end - offset);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, offset, cur);
            input_queue.EnQue(input_local);
            input_local = input_queue.DeQue<T>();
            SquareToFloat(square, input_local, tmp, cur);
            acc += ReduceFloatTile(square, work, sum, cur);
            input_queue.FreeTensor(input_local);
            offset += cur;
        }
        return acc;
    }

    __aicore__ inline void WritePartial(float value)
    {
        LocalTensor<float> local = partial_queue.AllocTensor<float>();
        local.SetValue(0, value);
        partial_queue.EnQue(local);
        local = partial_queue.DeQue<float>();
        DataCopy(partial_gm[GetBlockIdx() * PARTIAL_STRIDE], local, PARTIAL_STRIDE);
        partial_queue.FreeTensor(local);
    }

    __aicore__ inline void WriteScalarOutput(uint32_t offset, float value)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        out.SetValue(0, FromFloatScalar<T>(value));
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(output_gm[offset], out, params);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> partial_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t total_size = 0;
    uint32_t tile_elems = 0;
    uint32_t workspace_offset = 0;
    uint32_t partial_count = 0;
};

template <typename T>
class KernelTailReduce {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        outer_size = tiling.outer_size;
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        tile_elems = tiling.tile_elems;
        row_group = tiling.row_group == 0 ? 1 : tiling.row_group;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        const uint32_t core_count = static_cast<uint32_t>(GetBlockNum());
        const uint32_t aligned_rows = UB_ALIGN_BYTES / sizeof(T);
        if (CanProcessPaddedRows() &&
            output_size <= core_count * aligned_rows &&
            output_size > (core_count - 1U) * aligned_rows) {
            start = static_cast<uint32_t>(GetBlockIdx()) * aligned_rows;
            end = MinU32(start + aligned_rows, output_size);
        } else {
            GetBlockRange(output_size, start, end);
        }
        for (uint32_t row = start; row < end;) {
            if (CanProcessPaddedRows()) {
                uint32_t rows = MinU32(row_group, end - row);
                uint32_t input_stride = PaddedStrideElems<T>(reduce_size);
                uint32_t float_stride = PaddedStrideElems<float>(reduce_size);
                rows = MinU32(rows, MAX_REPEAT_ROWS);
                while (rows > 1 &&
                       (((rows - 1) * input_stride + MAX_FP32_REPEAT_MASK > tile_elems) ||
                        ((rows - 1) * float_stride + MAX_FP32_REPEAT_MASK > tile_elems))) {
                    --rows;
                }
                ProcessGroupedRows(row, rows, input_stride, float_stride);
                row += rows;
            } else {
                float acc = ReduceContiguousSegment(row * reduce_size, reduce_size);
                WriteScalarOutput(row, acc);
                ++row;
            }
        }
    }

private:
    __aicore__ inline float ReduceContiguousSegment(uint32_t base, uint32_t count)
    {
        float acc = 0.0f;
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        // Conversion scratch is dead when the reduction starts.
        LocalTensor<float> work = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 2 * float_bytes);
        for (uint32_t offset = 0; offset < count;) {
            uint32_t cur = MinU32(tile_elems, count - offset);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, base + offset, cur);
            input_queue.EnQue(input_local);
            input_local = input_queue.DeQue<T>();
            SquareToFloat(square, input_local, tmp, cur);
            PipeBarrier<PIPE_V>();
            acc += ReduceFloatTile(square, work, sum, cur);
            input_queue.FreeTensor(input_local);
            offset += cur;
        }
        return acc;
    }

    __aicore__ inline bool CanProcessPaddedRows()
    {
        if (row_group <= 1 || reduce_size == 0 || reduce_size > MAX_FP32_REPEAT_MASK) {
            return false;
        }
        uint32_t input_stride = PaddedStrideElems<T>(reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(reduce_size);
        if (MAX_FP32_REPEAT_MASK > tile_elems || input_stride > tile_elems || float_stride > tile_elems) {
            return false;
        }
        return PaddedStrideBlocks<T>(reduce_size) <= 255 && PaddedStrideBlocks<float>(reduce_size) <= 255;
    }

    __aicore__ inline void ProcessGroupedRows(uint32_t row_start, uint32_t rows, uint32_t input_stride,
                                              uint32_t float_stride)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<T> input_local = input_queue.AllocTensor<T>();
        CopyInPaddedRows(input_local, input_gm, row_start * reduce_size, reduce_size, rows);
        input_queue.EnQue(input_local);
        input_local = input_queue.DeQue<T>();

        LocalTensor<T> output_local = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, half>::value) {
            const uint8_t stride_blocks = static_cast<uint8_t>(
                input_stride * sizeof(half) / UB_ALIGN_BYTES);
            Mul(input_local, input_local, input_local,
                static_cast<uint64_t>(reduce_size), static_cast<uint8_t>(rows),
                {1, 1, 1, stride_blocks, stride_blocks, stride_blocks});
            PipeBarrier<PIPE_V>();
            WholeReduceSum<half>(
                output_local, input_local, reduce_size, rows, 1, 1, stride_blocks);
        } else {
            LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            SquarePaddedRowsToFloat(square, input_local, tmp, reduce_size, rows, input_stride, float_stride);
            PipeBarrier<PIPE_V>();
            const uint32_t stride_blocks = float_stride * sizeof(float) / UB_ALIGN_BYTES;
            if constexpr (std::is_same<T, float>::value) {
                WholeReduceSum<float>(output_local, square, reduce_size, rows, 1, 1, stride_blocks);
            } else {
                WholeReduceSum<float>(tmp, square, reduce_size, rows, 1, 1, stride_blocks);
                PipeBarrier<PIPE_V>();
                Cast(output_local, tmp, RoundMode::CAST_RINT, rows);
            }
        }
        input_queue.FreeTensor(input_local);
        output_queue.EnQue(output_local);
        output_local = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, output_local, row_start, rows);
        output_queue.FreeTensor(output_local);
    }

    __aicore__ inline void WriteScalarOutput(uint32_t offset, float value)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        out.SetValue(0, FromFloatScalar<T>(value));
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(output_gm[offset], out, params);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t outer_size = 0;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t tile_elems = 0;
    uint32_t row_group = 1;
};

template <typename T>
class KernelGroupedTailReduce {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output,
                                const SquareSumV1TilingData &tiling)
    {
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        tile_elems = tiling.tile_elems;
        row_group = tiling.row_group;
        input_stride = PaddedStrideElems<T>(reduce_size);
        float_stride = PaddedStrideElems<float>(reduce_size);
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        input_gm.template SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);

        pipe.InitBuffer(input_queue, 2,
                        tile_elems * sizeof(T));
        pipe.InitBuffer(output_queue, 2,
                        AlignUp32(kMaxRows * sizeof(T)));
        pipe.InitBuffer(square_buf,
                        tile_elems * sizeof(float));
        pipe.InitBuffer(work_buf,
                        tile_elems * sizeof(float));
        pipe.InitBuffer(calc_buf,
                        AlignUp32(kCalcElems * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        if (start >= end) {
            return;
        }

        uint32_t current_row = start;
        uint32_t previous_rows =
            RowsForTile(current_row, end);
        LocalTensor<T> input =
            input_queue.AllocTensor<T>();
        CopyInPaddedRows(
            input, input_gm,
            current_row * reduce_size,
            reduce_size, previous_rows);
        input_queue.EnQue(input);
        uint32_t next_row = current_row + previous_rows;

        while (next_row < end) {
            const uint32_t current_rows =
                RowsForTile(next_row, end);
            LocalTensor<T> next =
                input_queue.AllocTensor<T>();
            CopyInPaddedRows(
                next, input_gm,
                next_row * reduce_size,
                reduce_size, current_rows);
            input_queue.EnQue(next);

            LocalTensor<T> current =
                input_queue.DeQue<T>();
            ComputeRows(
                current_row, previous_rows, current);
            input_queue.FreeTensor(current);

            current_row = next_row;
            next_row += current_rows;
            previous_rows = current_rows;
        }

        input = input_queue.DeQue<T>();
        ComputeRows(current_row, previous_rows, input);
        input_queue.FreeTensor(input);
    }

private:
    __aicore__ inline uint32_t RowsForTile(
        uint32_t row, uint32_t end)
    {
        uint32_t rows =
            MinU32(
                MinU32(row_group, kMaxRows),
                end - row);
        rows =
            MinU32(rows, tile_elems / input_stride);
        rows =
            MinU32(rows, tile_elems / float_stride);
        return rows == 0 ? 1U : rows;
    }

    __aicore__ inline void SquareRows(LocalTensor<float> square,
                                      LocalTensor<T> input,
                                      uint32_t rows)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t row = 0; row < rows; ++row) {
                LocalTensor<float> dst =
                    square[row * float_stride];
                Cast(dst, input[row * input_stride],
                     RoundMode::CAST_NONE, reduce_size);
                PipeBarrier<PIPE_V>();
                Mul(dst, dst, dst, reduce_size);
            }
        } else {
            LocalTensor<float> unused = work_buf.Get<float>();
            SquarePaddedRowsToFloat(
                square, input, unused, reduce_size, rows,
                input_stride, float_stride);
        }
    }

    __aicore__ inline void ComputeRows(
        uint32_t row_start, uint32_t rows,
        LocalTensor<T> input)
    {
        LocalTensor<float> square = square_buf.Get<float>();
        LocalTensor<float> work = work_buf.Get<float>();
        LocalTensor<float> acc = calc_buf.Get<float>();
        LocalTensor<float> row_sums =
            calc_buf.GetWithOffset<float>(
                kRowSumElems,
                kAccElems * sizeof(float));
        SquareRows(square, input, rows);
        const uint32_t stride_blocks =
            float_stride * sizeof(float) / UB_ALIGN_BYTES;
        if (rows <= kMaxRows && stride_blocks <= 255) {
            uint32_t tail_offset = 0;
            uint32_t current =
                MinU32(MAX_FP32_REPEAT_MASK, reduce_size);
            WholeReduceSum<float>(
                acc, square, static_cast<int32_t>(current),
                static_cast<int32_t>(rows), 1, 1,
                static_cast<int32_t>(stride_blocks));
            PipeBarrier<PIPE_V>();
            tail_offset += current;
            while (tail_offset < reduce_size) {
                current =
                    MinU32(MAX_FP32_REPEAT_MASK,
                           reduce_size - tail_offset);
                WholeReduceSum<float>(
                    row_sums, square[tail_offset],
                    static_cast<int32_t>(current),
                    static_cast<int32_t>(rows), 1, 1,
                    static_cast<int32_t>(stride_blocks));
                PipeBarrier<PIPE_V>();
                Add(acc, acc, row_sums, rows);
                PipeBarrier<PIPE_V>();
                tail_offset += current;
            }
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                const float value = ReduceFloatTile(
                    square[row * float_stride], work, row_sums,
                    reduce_size);
                acc.SetValue(row, value);
            }
        }
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, rows);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, rows);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, row_start, rows);
        output_queue.FreeTensor(out);
    }

    static constexpr uint32_t kMaxRows = 255;
    static constexpr uint32_t kAccElems = 256;
    static constexpr uint32_t kRowSumElems = 256;
    static constexpr uint32_t kCalcElems =
        kAccElems + kRowSumElems;

    TQue<TPosition::VECIN, 2> input_queue;
    TQue<TPosition::VECOUT, 2> output_queue;
    TBuf<TPosition::VECCALC> square_buf;
    TBuf<TPosition::VECCALC> work_buf;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t tile_elems = 0;
    uint32_t row_group = 1;
    uint32_t input_stride = 1;
    uint32_t float_stride = 1;
};

class KernelBf16TailDenseGatherSlab {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output,
        const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        input_gm.SetGlobalBuffer(
            (__gm__ bfloat16_t *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ bfloat16_t *)output, output_size);

        pipe.InitBuffer(
            primary_input_queue, 1,
            AlignUp32(
                kPrimaryRows * kReduceSize * sizeof(bfloat16_t) +
                kInputGuardBytes));
        pipe.InitBuffer(
            secondary_input_queue, 1,
            AlignUp32(
                kSecondaryRows * kReduceSize * sizeof(bfloat16_t) +
                kInputGuardBytes));
        pipe.InitBuffer(
            output_queue, BUFFER_NUM,
            AlignUp32(kOutputRows * sizeof(bfloat16_t)));
        pipe.InitBuffer(
            converted_buf,
            AlignUp32(kConvertedElems * sizeof(float)));
        pipe.InitBuffer(
            mask_buf,
            2U * kMaskElems * sizeof(uint16_t));
        pipe.InitBuffer(
            tail_buf,
            2U * kTailElems * sizeof(float));
        pipe.InitBuffer(
            result_buf,
            kResultSlotBytes + kScratchRows * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block_idx =
            static_cast<uint32_t>(GetBlockIdx());
        const uint32_t block_num =
            static_cast<uint32_t>(GetBlockNum());
        if (block_num == 0U) {
            return;
        }

        const uint32_t pair_wave_rows =
            block_num * kPairRows;
        const uint32_t full_pair_waves =
            output_size / pair_wave_rows;
        const uint32_t pair_rows =
            full_pair_waves * pair_wave_rows;
        const uint32_t remaining_rows = output_size - pair_rows;
        const uint32_t uniform_rows =
            remaining_rows >= block_num * kPrimaryRows
                ? kPrimaryRows
                : (remaining_rows >= block_num * kSecondaryRows
                       ? kSecondaryRows
                       : 0U);
        const uint32_t residual_base =
            pair_rows + block_num * uniform_rows;
        const uint32_t residual_rows =
            output_size - residual_base;
        const uint32_t tail_base = residual_rows / block_num;
        const uint32_t tail_extra =
            residual_rows - tail_base * block_num;
        const uint32_t tail_rows =
            tail_base + (block_idx < tail_extra ? 1U : 0U);
        const uint32_t task_count =
            2U * full_pair_waves +
            (uniform_rows > 0U ? 1U : 0U) +
            (tail_rows > 0U ? 1U : 0U);
        if (task_count == 0U) {
            return;
        }

        LocalTensor<uint16_t> masks =
            mask_buf.Get<uint16_t>(2U * kMaskElems);
        const uint32_t group_count = full_pair_waves == 0U
            ? 1U
            : (full_pair_waves + kOutputPairWaves - 1U) / kOutputPairWaves;
        for (uint32_t group_idx = 0U; group_idx < group_count; ++group_idx) {
            const uint32_t first_wave = group_idx * kOutputPairWaves;
            const uint32_t group_waves =
                MinU32(kOutputPairWaves, full_pair_waves - first_wave);
            const bool final_group =
                first_wave + group_waves == full_pair_waves;
            const uint32_t first_task = 2U * first_wave;
            const uint32_t end_task = final_group
                ? task_count
                : 2U * (first_wave + group_waves);
            uint32_t current_row = 0U;
            uint32_t current_rows = 0U;
            bool current_primary = true;
            GetTask(first_task, block_idx, block_num, full_pair_waves,
                    uniform_rows, residual_base, tail_base, tail_extra,
                    current_row, current_rows, current_primary);
            LoadInput(current_primary, current_row, current_rows);
            if (group_idx == 0U) {
                BuildMasks(masks);
            }

            // Pack results without changing the original input task sequence.
            LocalTensor<bfloat16_t> out =
                output_queue.AllocTensor<bfloat16_t>();
            uint32_t output_offset = 0U;
            for (uint32_t task_idx = first_task + 1U;
                 task_idx < end_task; ++task_idx) {
                uint32_t next_row = 0U;
                uint32_t next_rows = 0U;
                bool next_primary = true;
                GetTask(task_idx, block_idx, block_num, full_pair_waves,
                        uniform_rows, residual_base, tail_base, tail_extra,
                        next_row, next_rows, next_primary);
                LoadInput(next_primary, next_row, next_rows);
                ConsumeInput(
                    current_primary, current_rows, masks, out[output_offset]);
                output_offset += current_rows;
                current_rows = next_rows;
                current_primary = next_primary;
            }
            ConsumeInput(
                current_primary, current_rows, masks, out[output_offset]);
            output_queue.EnQue(out);
            out = output_queue.DeQue<bfloat16_t>();

            if (group_waves > 0U) {
                const uint32_t pair_bytes =
                    kPairRows * sizeof(bfloat16_t);
                const uint32_t pair_start =
                    (first_wave * block_num + block_idx) * kPairRows;
                DataCopyExtParams params{
                    static_cast<uint16_t>(group_waves), pair_bytes,
                    0U, (block_num - 1U) * pair_bytes, 0U};
                DataCopyPad(output_gm[pair_start], out, params);
            }
            if (final_group) {
                const uint32_t uniform_offset = group_waves * kPairRows;
                if (uniform_rows > 0U) {
                    CopyOutContiguous(
                        output_gm, out[uniform_offset],
                        pair_rows + block_idx * uniform_rows, uniform_rows);
                }
                if (tail_rows > 0U) {
                    CopyOutContiguous(
                        output_gm, out[uniform_offset + uniform_rows],
                        residual_base + block_idx * tail_base +
                            MinU32(block_idx, tail_extra),
                        tail_rows);
                }
            }
            output_queue.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void GetTask(
        uint32_t task_idx, uint32_t block_idx, uint32_t block_num,
        uint32_t full_pair_waves, uint32_t uniform_rows,
        uint32_t residual_base, uint32_t tail_base,
        uint32_t tail_extra, uint32_t &row_start, uint32_t &rows,
        bool &use_primary)
    {
        const uint32_t pair_task_count = 2U * full_pair_waves;
        if (task_idx < pair_task_count) {
            const uint32_t wave = task_idx / 2U;
            use_primary = (task_idx & 1U) == 0U;
            row_start =
                (wave * block_num + block_idx) * kPairRows +
                (use_primary ? 0U : kPrimaryRows);
            rows = use_primary ? kPrimaryRows : kSecondaryRows;
            return;
        }

        if (uniform_rows > 0U) {
            if (task_idx == pair_task_count) {
                use_primary = true;
                row_start =
                    full_pair_waves * block_num * kPairRows +
                    block_idx * uniform_rows;
                rows = uniform_rows;
                return;
            }
        }

        use_primary = uniform_rows == 0U;
        row_start = residual_base + block_idx * tail_base +
                    MinU32(block_idx, tail_extra);
        rows = tail_base + (block_idx < tail_extra ? 1U : 0U);
    }

    __aicore__ inline void LoadInput(
        bool use_primary, uint32_t row_start, uint32_t rows)
    {
        if (use_primary) {
            LocalTensor<bfloat16_t> input =
                primary_input_queue.AllocTensor<bfloat16_t>();
            primary_input_head = CopyDenseRows(input, row_start, rows);
            primary_input_queue.EnQue(input);
            return;
        }
        LocalTensor<bfloat16_t> input =
            secondary_input_queue.AllocTensor<bfloat16_t>();
        secondary_input_head = CopyDenseRows(input, row_start, rows);
        secondary_input_queue.EnQue(input);
    }

    __aicore__ inline void ConsumeInput(
        bool use_primary, uint32_t rows,
        LocalTensor<uint16_t> masks, LocalTensor<bfloat16_t> out)
    {
        if (use_primary) {
            LocalTensor<bfloat16_t> input =
                primary_input_queue.DeQue<bfloat16_t>();
            ComputeRows(
                rows, input[primary_input_head], masks, out);
            primary_input_queue.FreeTensor(input);
            return;
        }
        LocalTensor<bfloat16_t> input =
            secondary_input_queue.DeQue<bfloat16_t>();
        ComputeRows(
            rows, input[secondary_input_head], masks, out);
        secondary_input_queue.FreeTensor(input);
    }

    __aicore__ inline uint32_t CopyDenseRows(
        LocalTensor<bfloat16_t> input,
        uint32_t row_start, uint32_t rows)
    {
        const uint32_t offset = row_start * kReduceSize;
        const uint32_t prefix = offset % kGmAlignElems;
        const uint32_t source_offset = offset - prefix;
        const uint32_t left_padding =
            (kBf16AlignElems - prefix % kBf16AlignElems) %
            kBf16AlignElems;
        const uint32_t logical_head = prefix + left_padding;
        const uint32_t logical_elems = rows * kReduceSize;
        const uint32_t fetch_elems = prefix + logical_elems;
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(
                fetch_elems * sizeof(bfloat16_t)),
            0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{
            true, static_cast<uint8_t>(left_padding), 0,
            ToBfloat16(0.0f)};
        DataCopyPad(input, input_gm[source_offset], params, pad);
        return logical_head;
    }

    __aicore__ inline void BuildMasks(
        LocalTensor<uint16_t> masks)
    {
        Duplicate(masks, static_cast<uint16_t>(0xffffU),
                  kScratchRows * kMaskStride);
        PipeBarrier<PIPE_V>();

        // Both boundary masks stay live across every row group.
        for (uint32_t row = 0U; row < kScratchRows; ++row) {
            Duplicate(masks[row * kMaskStride],
                      static_cast<uint16_t>(0U),
                      static_cast<uint64_t>(2U * (row + 1U)), 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        Not(masks[kMaskElems], masks, kMaskElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceRows(
        LocalTensor<float> windows, LocalTensor<float> folded,
        LocalTensor<float> float_sums,
        LocalTensor<uint16_t> masks, uint32_t rows)
    {
        LocalTensor<float> paired_rows = tail_buf.Get<float>(2U * kTailElems);
        LocalTensor<float> tails = paired_rows[kTailElems];

        LocalTensor<uint16_t> window_bits = windows.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> tail_bits = tails.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> head_masks = masks;
        LocalTensor<uint16_t> tail_masks = masks[kMaskElems];
        const BinaryRepeatParams head_params{1, 1, 1, 128, 128, 2};
        const BinaryRepeatParams tail_params{1, 1, 1, 2, 128, 2};
        // Gather and mask the saved tails before clearing their source heads.
        And(tail_bits, window_bits[kAlignedRowStride * 2U], tail_masks,
            static_cast<uint64_t>(kMaskStride),
            static_cast<uint8_t>(rows), tail_params);
        PipeBarrier<PIPE_V>();
        // Bitwise zeroing preserves owned FP32 bits and removes even foreign
        // NaN/Inf, with exact dst==src0 alias and disjoint repeat write sets.
        if (rows > 1U) {
            And(window_bits[kAlignedRowStride * 2U],
                window_bits[kAlignedRowStride * 2U], head_masks,
                static_cast<uint64_t>(kMaskStride),
                static_cast<uint8_t>(rows - 1U), head_params);
        }
        PipeBarrier<PIPE_V>();

        const uint32_t half_fold = kFoldStride / 2U;
        // Both batches write consumed raw input, disjoint from converted source.
        for (uint32_t row = 0U; row < rows; row += 8U) {
            const uint32_t batch_rows = MinU32(8U, rows - row);
            BlockReduceSum<float>(
                folded[row * kFoldStride], windows[row * kAlignedRowStride],
                batch_rows * (kAlignedRowStride / half_fold), half_fold,
                1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        // Place the head rows immediately before the saved boundary tails.
        BlockReduceSum<float>(
            paired_rows, folded, rows * 2U, half_fold,
            1, 1, 8);
        if (rows < kScratchRows) {
            Duplicate(paired_rows[rows * kTailStride], 0.0f,
                      (kScratchRows - rows) * kTailStride);
        }
        PipeBarrier<PIPE_V>();
        if (rows == kScratchRows) {
            // Compact each 16-value row to two block sums, then join the pair.
            BlockReduceSum<float>(
                paired_rows, paired_rows, 8, half_fold, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            PairReduceSum<float>(
                float_sums, paired_rows, 1, half_fold, 1, 1, 8);
        } else {
            WholeReduceSum<float>(
                float_sums, paired_rows, kTailStride, kScratchRows + rows,
                1, 1, kTailStride * sizeof(float) / UB_ALIGN_BYTES);
        }
        PipeBarrier<PIPE_V>();
        Add(float_sums, float_sums, float_sums[kScratchRows], rows);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRows(
        uint32_t rows,
        LocalTensor<bfloat16_t> input,
        LocalTensor<uint16_t> masks, LocalTensor<bfloat16_t> out)
    {
        LocalTensor<float> converted =
            converted_buf.Get<float>(kConvertedElems);
        LocalTensor<float> float_sums =
            result_buf.Get<float>(kResultElems);
        for (uint32_t row = 0; row < rows; row += kScratchRows) {
            const uint32_t group_rows = MinU32(kScratchRows, rows - row);
            const uint32_t dense_offset = row * kReduceSize;
            const uint32_t aligned_offset = dense_offset / 16U * 16U;
            const uint32_t phase = dense_offset - aligned_offset;
            const uint32_t convert_elems =
                AlignUp32((phase + group_rows * kReduceSize) *
                          sizeof(bfloat16_t)) / sizeof(bfloat16_t);
            // Rounded reads include only a future group's head or zero pad.
            Cast(converted, input[aligned_offset], RoundMode::CAST_NONE,
                 convert_elems);
            PipeBarrier<PIPE_V>();
            Mul(converted, converted, converted, convert_elems);
            PipeBarrier<PIPE_V>();
            // A full fold needs 8192 bytes; the next group starts 32800 bytes later.
            LocalTensor<float> folded =
                input[aligned_offset].ReinterpretCast<float>();
            ReduceRows(
                converted[phase], folded, float_sums[row], masks, group_rows);
        }

        Cast(out, float_sums, RoundMode::CAST_RINT, rows);
    }

    static constexpr uint32_t kPrimaryRows = 32;
    static constexpr uint32_t kSecondaryRows = 16;
    static constexpr uint32_t kReduceSize = 1025;
    static constexpr uint32_t kAlignedRowStride = kReduceSize - 1U;
    static constexpr uint32_t kPairRows =
        kPrimaryRows + kSecondaryRows;
    static constexpr uint32_t kOutputPairWaves = 7;
    static constexpr uint32_t kOutputRows =
        kOutputPairWaves * kPairRows + kPrimaryRows + kSecondaryRows;
    static constexpr uint32_t kScratchRows = 16;
    static constexpr uint32_t kTailStride = 16;
    static constexpr uint32_t kMaskStride =
        kTailStride * sizeof(float) / sizeof(uint16_t);
    static constexpr uint32_t kInputGuardBytes = 544;
    static constexpr uint32_t kGmAlignElems =
        512U / sizeof(bfloat16_t);
    static constexpr uint32_t kBf16AlignElems =
        UB_ALIGN_BYTES / sizeof(bfloat16_t);
    static constexpr uint32_t kFoldStride = 128;
    static constexpr uint32_t kConvertedElems =
        kScratchRows * (kReduceSize + 1U);
    static constexpr uint32_t kMaskElems =
        kScratchRows * kMaskStride;
    static constexpr uint32_t kTailElems =
        kScratchRows * kTailStride;
    static constexpr uint32_t kResultSlotBytes =
        kPrimaryRows * sizeof(float);
    static constexpr uint32_t kResultElems =
        kResultSlotBytes / sizeof(float) + kScratchRows;

    TQue<TPosition::VECIN, 1> primary_input_queue;
    TQue<TPosition::VECIN, 1> secondary_input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> converted_buf;
    TBuf<TPosition::VECCALC> mask_buf;
    TBuf<TPosition::VECCALC> tail_buf;
    TBuf<TPosition::VECCALC> result_buf;
    GlobalTensor<bfloat16_t> input_gm;
    GlobalTensor<bfloat16_t> output_gm;
    uint32_t output_size = 0;
    uint32_t primary_input_head = 0U;
    uint32_t secondary_input_head = 0U;
};

class KernelBf16SmallTailReduce {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output,
        const SquareSumV1TilingData &tiling)
    {
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        tile_elems = tiling.tile_elems;
        row_group = tiling.row_group;
        row_stride = PaddedStrideElems<bfloat16_t>(reduce_size);
        float_stride = PaddedStrideElems<float>(reduce_size);
        input_gm.SetGlobalBuffer(
            (__gm__ bfloat16_t *)input, tiling.total_size);
        output_gm.SetGlobalBuffer(
            (__gm__ bfloat16_t *)output, output_size);

        pipe.InitBuffer(
            input_queue, BUFFER_NUM,
            tile_elems * sizeof(bfloat16_t));
        pipe.InitBuffer(
            output_queue, BUFFER_NUM,
            AlignUp32(kMaxRows * sizeof(bfloat16_t)));
        pipe.InitBuffer(
            converted_buf, tile_elems * sizeof(float));
        pipe.InitBuffer(
            square_buf, tile_elems * sizeof(half));
        half_sums_bytes = AlignUp32(kMaxRows * sizeof(half));
        pipe.InitBuffer(
            result_buf,
            half_sums_bytes +
                AlignUp32(kMaxRows * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        if (start >= end) {
            return;
        }

        uint32_t current_row = start;
        uint32_t previous_rows =
            RowsForTile(current_row, end);
        LocalTensor<bfloat16_t> input =
            input_queue.AllocTensor<bfloat16_t>();
        CopyInPaddedRows(
            input, input_gm,
            current_row * reduce_size,
            reduce_size, previous_rows);
        input_queue.EnQue(input);
        uint32_t next_row = current_row + previous_rows;

        while (next_row < end) {
            const uint32_t current_rows =
                RowsForTile(next_row, end);
            LocalTensor<bfloat16_t> next =
                input_queue.AllocTensor<bfloat16_t>();
            CopyInPaddedRows(
                next, input_gm,
                next_row * reduce_size,
                reduce_size, current_rows);
            input_queue.EnQue(next);

            LocalTensor<bfloat16_t> current =
                input_queue.DeQue<bfloat16_t>();
            ComputeRows(
                current_row, previous_rows, current);
            input_queue.FreeTensor(current);

            current_row = next_row;
            next_row += current_rows;
            previous_rows = current_rows;
        }

        input = input_queue.DeQue<bfloat16_t>();
        ComputeRows(current_row, previous_rows, input);
        input_queue.FreeTensor(input);
    }

private:
    __aicore__ inline uint32_t RowsForTile(
        uint32_t row, uint32_t end)
    {
        uint32_t rows = MinU32(
            MinU32(row_group, kMaxRows), end - row);
        rows = MinU32(rows, tile_elems / row_stride);
        rows = MinU32(rows, tile_elems / float_stride);
        return rows == 0 ? 1U : rows;
    }

    __aicore__ inline void ComputeRows(
        uint32_t row_start, uint32_t rows,
        LocalTensor<bfloat16_t> input)
    {
        LocalTensor<float> converted =
            converted_buf.Get<float>();
        LocalTensor<half> square = square_buf.Get<half>();
        const uint8_t input_rep_stride =
            static_cast<uint8_t>(
                row_stride * sizeof(bfloat16_t) /
                UB_ALIGN_BYTES);
        const uint8_t float_rep_stride =
            static_cast<uint8_t>(
                float_stride * sizeof(float) /
                UB_ALIGN_BYTES);
        UnaryRepeatParams to_float_params(
            1, 1, float_rep_stride, input_rep_stride);
        UnaryRepeatParams to_half_params(
            1, 1, input_rep_stride, float_rep_stride);
        for (uint32_t col = 0; col < reduce_size;
             col += MAX_FP32_REPEAT_MASK) {
            const uint32_t current = MinU32(
                MAX_FP32_REPEAT_MASK, reduce_size - col);
            Cast(
                converted[col], input[col],
                RoundMode::CAST_NONE,
                static_cast<uint64_t>(current),
                static_cast<uint8_t>(rows),
                to_float_params);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < reduce_size;
             col += MAX_FP32_REPEAT_MASK) {
            const uint32_t current = MinU32(
                MAX_FP32_REPEAT_MASK, reduce_size - col);
            Cast(
                square[col], converted[col],
                RoundMode::CAST_RINT,
                static_cast<uint64_t>(current),
                static_cast<uint8_t>(rows),
                to_half_params);
        }
        PipeBarrier<PIPE_V>();

        const uint8_t stride_blocks =
            static_cast<uint8_t>(
                row_stride * sizeof(half) /
                UB_ALIGN_BYTES);
        BinaryRepeatParams repeat_params{
            1, 1, 1,
            stride_blocks, stride_blocks, stride_blocks};
        Mul(
            square, square, square,
            static_cast<uint64_t>(reduce_size),
            static_cast<uint8_t>(rows), repeat_params);
        PipeBarrier<PIPE_V>();

        LocalTensor<half> half_sums =
            result_buf.Get<half>();
        WholeReduceSum<half>(
            half_sums, square,
            static_cast<int32_t>(reduce_size),
            static_cast<int32_t>(rows),
            1, 1, static_cast<int32_t>(stride_blocks));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> float_sums =
            result_buf.GetWithOffset<float>(
                kMaxRows, half_sums_bytes);
        Cast(
            float_sums, half_sums,
            RoundMode::CAST_NONE, rows);
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> out =
            output_queue.AllocTensor<bfloat16_t>();
        Cast(
            out, float_sums,
            RoundMode::CAST_RINT, rows);
        output_queue.EnQue(out);
        out = output_queue.DeQue<bfloat16_t>();
        CopyOutContiguous(
            output_gm, out, row_start, rows);
        output_queue.FreeTensor(out);
    }

    static constexpr uint32_t kMaxRows = 255;

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> converted_buf;
    TBuf<TPosition::VECCALC> square_buf;
    TBuf<TPosition::VECCALC> result_buf;
    GlobalTensor<bfloat16_t> input_gm;
    GlobalTensor<bfloat16_t> output_gm;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t tile_elems = 0;
    uint32_t row_group = 1;
    uint32_t row_stride = 1;
    uint32_t float_stride = 1;
    uint32_t half_sums_bytes = 0;
};

template <typename T>
class KernelHeadReduce {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const SquareSumV1TilingData &tiling)
    {
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        workspace_offset = tiling.workspace_offset;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride == 0 ? PARTIAL_STRIDE : tiling.partial_stride;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, tiling.output_size);
        if (partial_count > 1) {
            partial_gm.SetGlobalBuffer((__gm__ float *)(workspace + workspace_offset),
                                       partial_count * partial_stride);
        }
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        if (partial_count > 1) {
            pipe.InitBuffer(partial_queue, 1, AlignUp32(inner_size * sizeof(float)));
        }
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        if (CanSplitReduceAxis()) {
            ProcessReduceAxisSplit();
            return;
        }
        if (CanProcessContiguousHeadReduce()) {
            ProcessContiguousHeadReduce();
            return;
        }
        if (CanProcessFullInnerBlocks()) {
            ProcessFullInnerBlocks();
            return;
        }
        ProcessStridedRowGroups();
    }

private:
    __aicore__ inline void ProcessStridedRowGroups()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(inner_size, start, end);
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        for (uint32_t offset = start; offset < end;) {
            uint32_t cur = MinU32(tile_elems, end - offset);
            LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            Duplicate(acc, 0.0f, cur);
            uint32_t input_stride = PaddedStrideElems<T>(cur);
            uint32_t rows_per_tile = tile_elems / input_stride;
            rows_per_tile = MinU32(rows_per_tile, 4095);
            if (rows_per_tile == 0) {
                rows_per_tile = 1;
            }

            uint32_t r_base = 0;
            uint32_t rows_prev = 0;

            // Prime the pipeline
            if (r_base < reduce_size) {
                rows_prev = MinU32(rows_per_tile, reduce_size - r_base);
                LocalTensor<T> input_local = input_queue.AllocTensor<T>();
                CopyInStridedRows(input_local, input_gm, r_base * inner_size + offset, cur, rows_prev);
                input_queue.EnQue(input_local);
                r_base += rows_prev;
            }

            while (r_base < reduce_size) {
                uint32_t rows_next = MinU32(rows_per_tile, reduce_size - r_base);
                LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
                CopyInStridedRows(input_local_next, input_gm, r_base * inner_size + offset, cur, rows_next);
                input_queue.EnQue(input_local_next);

                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                AccumulateStridedRows(acc, input_local_curr, tmp, cur, rows_prev, input_stride);
                input_queue.FreeTensor(input_local_curr);

                r_base += rows_next;
                rows_prev = rows_next;
            }

            if (rows_prev > 0) {
                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                AccumulateStridedRows(acc, input_local_curr, tmp, cur, rows_prev, input_stride);
                input_queue.FreeTensor(input_local_curr);
            }

            StoreOutputTile(offset, acc, cur);
            offset += cur;
        }
    }

    __aicore__ inline void CopyInStridedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset,
                                             uint32_t row_elems, uint32_t rows)
    {
        uint32_t row_bytes = row_elems * sizeof(T);
        uint32_t stride_bytes = (inner_size - row_elems) * sizeof(T);
        if (row_bytes % UB_ALIGN_BYTES == 0 && stride_bytes % UB_ALIGN_BYTES == 0 &&
            (offset * sizeof(T)) % UB_ALIGN_BYTES == 0) {
            DataCopyParams params{static_cast<uint16_t>(rows),
                                  static_cast<uint16_t>(row_bytes / UB_ALIGN_BYTES),
                                  static_cast<uint16_t>(stride_bytes / UB_ALIGN_BYTES), 0};
            DataCopy(dst, src[offset], params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(rows), row_bytes, stride_bytes, 0, 0};
            DataCopyPadExtParams<T> pad_params{false, 0, 0, FromFloatScalar<T>(0.0f)};
            DataCopyPad(dst, src[offset], params, pad_params);
        }
    }

    __aicore__ inline void AccumulateStridedRows(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                 LocalTensor<float> tmp, uint32_t row_elems, uint32_t rows,
                                                 uint32_t input_stride)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * input_stride], tmp, row_elems);
            }
        } else {
            uint32_t src_stride_blocks = (input_stride * sizeof(T)) / UB_ALIGN_BYTES;
            if (src_stride_blocks <= 255) {
                uint32_t col_done = 0;
                while (col_done < row_elems) {
                    uint32_t col_cur = MinU32(64, row_elems - col_done);
                    uint32_t r_done = 0;
                    while (r_done < rows) {
                        uint32_t r_cur = MinU32(255, rows - r_done);
                        BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                         static_cast<uint8_t>(src_stride_blocks));
                        MulAddDst(acc[col_done], input_local[r_done * input_stride + col_done],
                                  input_local[r_done * input_stride + col_done],
                                  static_cast<uint64_t>(col_cur), static_cast<uint8_t>(r_cur), repeat_params);
                        r_done += r_cur;
                    }
                    col_done += col_cur;
                }
                return;
            }
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * input_stride], tmp, row_elems);
            }
        }
    }

    __aicore__ inline void AccumulateContiguousRows(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                    LocalTensor<float> tmp, uint32_t rows)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * inner_size], tmp, inner_size);
            }
        } else {
            uint32_t src_stride_blocks = (inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (inner_size <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * inner_size], input_local[r_done * inner_size],
                              static_cast<uint64_t>(inner_size), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * inner_size], tmp, inner_size);
                }
            }
        }
    }

    __aicore__ inline bool CanProcessContiguousHeadReduce()
    {
        if (inner_size == 0 || inner_size > tile_elems) {
            return false;
        }
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(inner_size, start, end);
        uint32_t cur = end - start;
        return ((start * sizeof(T)) % UB_ALIGN_BYTES == 0) && ((cur * sizeof(T)) % UB_ALIGN_BYTES == 0);
    }

    __aicore__ inline void AccumulateRowsRepeat(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                LocalTensor<float> tmp, uint32_t start_col, uint32_t cur_cols, uint32_t rows)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
            }
        } else {
            uint32_t src_stride_blocks = (inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (cur_cols <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * inner_size + start_col], input_local[r_done * inner_size + start_col],
                              static_cast<uint64_t>(cur_cols), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
                }
            }
        }
    }

    __aicore__ inline void ProcessContiguousHeadReduce()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(inner_size, start, end);
        uint32_t cur = end - start;

        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, cur);

        uint32_t rows_per_tile = tile_elems / inner_size;
        if (rows_per_tile == 0) {
            rows_per_tile = 1;
        }

        uint32_t r_base = 0;
        uint32_t rows_prev = 0;

        // Prime the pipeline
        if (r_base < reduce_size) {
            rows_prev = MinU32(rows_per_tile, reduce_size - r_base);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, r_base * inner_size, rows_prev * inner_size);
            input_queue.EnQue(input_local);
            r_base += rows_prev;
        }

        while (r_base < reduce_size) {
            uint32_t rows_next = MinU32(rows_per_tile, reduce_size - r_base);
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local_next, input_gm, r_base * inner_size, rows_next * inner_size);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateStridedRows(acc, input_local_curr[start], tmp, cur, rows_prev, inner_size);
            input_queue.FreeTensor(input_local_curr);

            r_base += rows_next;
            rows_prev = rows_next;
        }

        if (rows_prev > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateStridedRows(acc, input_local_curr[start], tmp, cur, rows_prev, inner_size);
            input_queue.FreeTensor(input_local_curr);
        }

        StoreOutputTile(start, acc, cur);
    }

    __aicore__ inline bool CanProcessFullInnerBlocks()
    {
        return false;
    }

    __aicore__ inline void ProcessFullInnerBlocks()
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, inner_size);

        uint32_t rows_per_tile = tile_elems / inner_size;
        if (rows_per_tile == 0) {
            return;
        }

        uint32_t r_base = 0;
        uint32_t rows_prev = 0;

        // Prime the pipeline
        if (r_base < reduce_size) {
            rows_prev = MinU32(rows_per_tile, reduce_size - r_base);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, r_base * inner_size, rows_prev * inner_size);
            input_queue.EnQue(input_local);
            r_base += rows_prev;
        }

        while (r_base < reduce_size) {
            uint32_t rows_next = MinU32(rows_per_tile, reduce_size - r_base);
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local_next, input_gm, r_base * inner_size, rows_next * inner_size);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateContiguousRows(acc, input_local_curr, tmp, rows_prev);
            input_queue.FreeTensor(input_local_curr);

            r_base += rows_next;
            rows_prev = rows_next;
        }

        if (rows_prev > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateContiguousRows(acc, input_local_curr, tmp, rows_prev);
            input_queue.FreeTensor(input_local_curr);
        }

        StoreOutputTile(0, acc, inner_size);
    }

    __aicore__ inline bool CanSplitReduceAxis()
    {
        return partial_count > 1 && inner_size > 0 && inner_size <= tile_elems;
    }

    __aicore__ inline void ProcessReduceAxisSplit()
    {
        if (GetBlockIdx() >= partial_count) {
            return;
        }
        uint32_t reduce_start = 0;
        uint32_t reduce_end = 0;
        GetRangeByParts(reduce_size, GetBlockIdx(), partial_count, reduce_start, reduce_end);
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, inner_size);

        uint32_t rows_per_tile = tile_elems / inner_size;
        if (rows_per_tile == 0) {
            rows_per_tile = 1;
        }

        uint32_t block_reduce_size = reduce_end - reduce_start;
        uint32_t r_base = 0;
        uint32_t rows_prev = 0;

        // Prime the pipeline
        if (r_base < block_reduce_size) {
            rows_prev = MinU32(rows_per_tile, block_reduce_size - r_base);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, (reduce_start + r_base) * inner_size, rows_prev * inner_size);
            input_queue.EnQue(input_local);
            r_base += rows_prev;
        }

        while (r_base < block_reduce_size) {
            uint32_t rows_next = MinU32(rows_per_tile, block_reduce_size - r_base);
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local_next, input_gm, (reduce_start + r_base) * inner_size, rows_next * inner_size);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateContiguousRows(acc, input_local_curr, tmp, rows_prev);
            input_queue.FreeTensor(input_local_curr);

            r_base += rows_next;
            rows_prev = rows_next;
        }

        if (rows_prev > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateContiguousRows(acc, input_local_curr, tmp, rows_prev);
            input_queue.FreeTensor(input_local_curr);
        }

        WritePartial(acc);
        SyncAll();
        if (GetBlockIdx() == 0) {
            MergePartialsAndStore();
        }
        return;
    }

    __aicore__ inline void GetRangeByParts(uint32_t total, uint32_t part_idx, uint32_t part_num,
                                           uint32_t &start, uint32_t &end)
    {
        uint32_t base = total / part_num;
        uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1 : 0);
    }

    __aicore__ inline void WritePartial(LocalTensor<float> acc)
    {
        LocalTensor<float> local = partial_queue.AllocTensor<float>();
        Adds(local, acc, 0.0f, inner_size);
        partial_queue.EnQue(local);
        local = partial_queue.DeQue<float>();
        CopyOutContiguous(partial_gm, local, GetBlockIdx() * partial_stride, inner_size);
        partial_queue.FreeTensor(local);
    }

    __aicore__ inline void MergePartialsAndStore()
    {
        uint32_t float_bytes = AlignUp32(inner_size * sizeof(float));
        LocalTensor<float> acc_sum = calc_buf.Get<float>(inner_size);
        LocalTensor<float> acc_load = calc_buf.GetWithOffset<float>(inner_size, float_bytes);

        Duplicate(acc_sum, 0.0f, inner_size);

        for (uint32_t c = 0; c < partial_count; ++c) {
            CopyInContiguous(acc_load, partial_gm, c * partial_stride, inner_size);
            Add(acc_sum, acc_sum, acc_load, inner_size);
        }

        StoreOutputTile(0, acc_sum, inner_size);
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TQue<TPosition::VECOUT, 1> partial_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
    uint32_t workspace_offset = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = PARTIAL_STRIDE;
};

class KernelHeadFp32OffsetPrefix {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
        const SquareSumV1TilingData &tiling)
    {
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        input_stride = PaddedStrideElems<float>(inner_size);
        acc_stride = PaddedStrideElems<float>(inner_size);
        const uint32_t usable_bytes =
            tile_elems * kHostBytesPerTileElem;
        const uint32_t fixed_bytes =
            BUFFER_NUM * AlignUp32(
                inner_size * sizeof(float)) +
            kAccBanks * acc_stride * sizeof(float) +
            kSyncBytes +
            BUFFER_NUM * kInputGuardBytes;
        const uint32_t row_bytes =
            BUFFER_NUM * inner_size * sizeof(float) +
            input_stride *
                (sizeof(float) + sizeof(uint32_t));
        rows_per_tile =
            usable_bytes > fixed_bytes
                ? (usable_bytes - fixed_bytes) / row_bytes
                : 1U;
        if (rows_per_tile == 0) {
            rows_per_tile = 1;
        }
        rows_per_tile = MinU32(
            rows_per_tile, MAX_REPEAT_ROWS);
        dense_tile_elems = rows_per_tile * inner_size;
        padded_tile_elems = rows_per_tile * input_stride;

        input_gm.SetGlobalBuffer(
            (__gm__ float *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ float *)output, tiling.output_size);
        SetSysWorkspace(workspace);
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        sync_gm.SetGlobalBuffer(
            (__gm__ int32_t *)user_workspace,
            partial_count * kSyncElems);
        partial_gm.SetGlobalBuffer(
            (__gm__ float *)(user_workspace +
                partial_count * kSyncBytes),
            partial_count * partial_stride);

        pipe.InitBuffer(
            input_queue, BUFFER_NUM,
            AlignUp32(
                dense_tile_elems * sizeof(float) +
                kInputGuardBytes));
        pipe.InitBuffer(
            output_queue, BUFFER_NUM,
            AlignUp32(inner_size * sizeof(float)));
        acc_bytes = AlignUp32(
            kAccBanks * acc_stride * sizeof(float));
        padded_bytes = AlignUp32(
            padded_tile_elems * sizeof(float));
        offsets_bytes = AlignUp32(
            padded_tile_elems * sizeof(uint32_t));
        pipe.InitBuffer(
            acc_buf,
            acc_bytes + padded_bytes + offsets_bytes);
        pipe.InitBuffer(sync_buf, kSyncBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        if (partial_count <= 1 || core_idx >= partial_count) {
            return;
        }
        InitSyncSlot(core_idx);

        uint32_t reduce_start = 0;
        uint32_t reduce_end = 0;
        GetRangeByParts(
            reduce_size, core_idx, partial_count,
            reduce_start, reduce_end);

        LocalTensor<float> acc = acc_buf.Get<float>();
        Duplicate(acc, 0.0f, kAccBanks * acc_stride);
        ProcessReduceRange(acc, reduce_start, reduce_end);
        ReduceBanks(acc);
        LocalTensor<int32_t> sync_local =
            sync_buf.Get<int32_t>();
        for (uint32_t step = 1; step < partial_count; step <<= 1) {
            const uint32_t group = step << 1;
            const uint32_t lane = core_idx % group;
            if (lane == step) {
                WritePartial(acc, core_idx);
                IBSet(sync_gm, sync_local, core_idx, 0);
                return;
            }
            if (lane == 0) {
                const uint32_t peer = core_idx + step;
                if (peer < partial_count) {
                    IBWait(sync_gm, sync_local, peer, 0);
                    MergePartial(acc, peer);
                }
            }
        }
        StoreOutput(acc);
    }

private:
    __aicore__ inline void InitSyncSlot(uint32_t core_idx)
    {
        GlobalTensor<int32_t> slot =
            sync_gm[core_idx * kSyncElems];
        InitGlobalMemory(
            slot, kSyncElems, static_cast<int32_t>(0));
        event_t event_id = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void GetRangeByParts(
        uint32_t total, uint32_t part_idx, uint32_t part_num,
        uint32_t &start, uint32_t &end)
    {
        const uint32_t base = total / part_num;
        const uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1U : 0U);
    }

    __aicore__ inline void CopyRows(
        LocalTensor<float> dst, uint32_t row_start,
        uint32_t rows)
    {
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(
                rows * inner_size * sizeof(float)),
            0, 0, 0};
        DataCopyPadExtParams<float> pad{
            false, 0, 0, 0.0f};
        DataCopyPad(
            dst, input_gm[row_start * inner_size],
            params, pad);
    }

    __aicore__ inline void BuildOffsets(uint32_t rows)
    {
        LocalTensor<int32_t> offsets =
            acc_buf.GetWithOffset<int32_t>(
                padded_tile_elems,
                acc_bytes + padded_bytes);
        const uint32_t padded_count =
            rows * input_stride;
        CreateVecIndex(
            offsets, static_cast<int32_t>(0),
            padded_count);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)),
            padded_count);
        const uint32_t row_gap_bytes =
            (input_stride - inner_size) *
            sizeof(float);
        if (row_gap_bytes != 0) {
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 1; row < rows; ++row) {
                Adds(
                    offsets[row * input_stride],
                    offsets[row * input_stride],
                    -static_cast<int32_t>(
                        row * row_gap_bytes),
                    input_stride);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateRows(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t rows)
    {
        LocalTensor<float> padded =
            acc_buf.GetWithOffset<float>(
                padded_tile_elems, acc_bytes);
        LocalTensor<int32_t> offsets =
            acc_buf.GetWithOffset<int32_t>(
                padded_tile_elems,
                acc_bytes + padded_bytes);
        const uint32_t padded_count =
            rows * input_stride;
        Gather(
            padded, input,
            offsets.ReinterpretCast<uint32_t>(),
            0, padded_count);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t bank = row % kAccBanks;
            MulAddDst(
                acc[bank * acc_stride],
                padded[row * input_stride],
                padded[row * input_stride], inner_size);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessReduceRange(
        LocalTensor<float> acc, uint32_t reduce_start,
        uint32_t reduce_end)
    {
        uint32_t row_start = reduce_start;
        uint32_t previous_rows = 0;
        if (row_start < reduce_end) {
            previous_rows =
                MinU32(rows_per_tile, reduce_end - row_start);
            LocalTensor<float> input =
                input_queue.AllocTensor<float>();
            CopyRows(input, row_start, previous_rows);
            input_queue.EnQue(input);
            row_start += previous_rows;
            BuildOffsets(previous_rows);
        }

        while (row_start < reduce_end) {
            const uint32_t current_rows =
                MinU32(rows_per_tile, reduce_end - row_start);
            LocalTensor<float> next =
                input_queue.AllocTensor<float>();
            CopyRows(next, row_start, current_rows);
            input_queue.EnQue(next);

            LocalTensor<float> current =
                input_queue.DeQue<float>();
            AccumulateRows(acc, current, previous_rows);
            input_queue.FreeTensor(current);
            row_start += current_rows;
            previous_rows = current_rows;
        }

        if (previous_rows > 0) {
            LocalTensor<float> current =
                input_queue.DeQue<float>();
            AccumulateRows(acc, current, previous_rows);
            input_queue.FreeTensor(current);
        }
    }

    __aicore__ inline void ReduceBanks(LocalTensor<float> acc)
    {
        for (uint32_t step = 1;
             step < kAccBanks; step <<= 1) {
            for (uint32_t bank = 0; bank < kAccBanks;
                 bank += (step << 1)) {
                Add(
                    acc[bank * acc_stride],
                    acc[bank * acc_stride],
                    acc[(bank + step) * acc_stride],
                    inner_size);
            }
        }
    }

    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        LocalTensor<float> local =
            output_queue.AllocTensor<float>();
        Adds(local, acc, 0.0f, inner_size);
        output_queue.EnQue(local);
        local = output_queue.DeQue<float>();
        CopyOutContiguous(
            partial_gm, local,
            core_idx * partial_stride, inner_size);
        output_queue.FreeTensor(local);
    }

    __aicore__ inline void MergePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        LocalTensor<float> local =
            input_queue.AllocTensor<float>();
        CopyInContiguous(
            local, partial_gm,
            core_idx * partial_stride, inner_size);
        input_queue.EnQue(local);
        local = input_queue.DeQue<float>();
        Add(acc, acc, local, inner_size);
        input_queue.FreeTensor(local);
    }

    __aicore__ inline void StoreOutput(LocalTensor<float> acc)
    {
        LocalTensor<float> out =
            output_queue.AllocTensor<float>();
        Adds(out, acc, 0.0f, inner_size);
        output_queue.EnQue(out);
        out = output_queue.DeQue<float>();
        CopyOutContiguous(output_gm, out, 0, inner_size);
        output_queue.FreeTensor(out);
    }

    static constexpr uint32_t kAccBanks = 8;
    static constexpr uint32_t kHostBytesPerTileElem = 28;
    static constexpr uint32_t kInputGuardBytes = 32;
    static constexpr uint32_t kSyncBytes = 32;
    static constexpr uint32_t kSyncElems =
        kSyncBytes / sizeof(int32_t);

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> acc_buf;
    TBuf<TPosition::VECCALC> sync_buf;
    GlobalTensor<float> input_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<float> partial_gm;
    GlobalTensor<int32_t> sync_gm;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
    uint32_t rows_per_tile = 1;
    uint32_t dense_tile_elems = 0;
    uint32_t padded_tile_elems = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    uint32_t input_stride = 0;
    uint32_t acc_stride = 0;
    uint32_t acc_bytes = 0;
    uint32_t padded_bytes = 0;
    uint32_t offsets_bytes = 0;
};

class KernelHeadFp32Width519Reverse {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
        const SquareSumV1TilingData &tiling)
    {
        total_size = tiling.total_size;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        packet1_capacity = tiling.tile_elems;
        use_input_scratch =
            packet1_capacity == kPacket1StreamElems;
        input1_holds_scratch = false;
        compact_stride = PaddedStrideElems<float>(kInnerSize);
        acc_bytes = AlignUp32(
            kAccBanks * kInnerSize * sizeof(float) +
            kAccGuardBytes);
        aux_bytes = AlignUp32(
            (use_input_scratch
                 ? kStreamAuxBuffers
                 : kEpochAuxBuffers) * compact_stride *
            sizeof(uint32_t));
        input0_bytes = AlignUp32(
            kPacket0Elems * sizeof(float) +
            kInputGuardBytes);
        input1_bytes = AlignUp32(
            packet1_capacity * sizeof(float) +
            kInputGuardBytes);

        input_gm.SetGlobalBuffer(
            (__gm__ float *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ float *)output, tiling.output_size);
        SetSysWorkspace(workspace);
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        partial_gm.SetGlobalBuffer(
            (__gm__ float *)user_workspace,
            partial_count * partial_stride);

        pipe.InitBuffer(
            input_queue0, 1, input0_bytes);
        pipe.InitBuffer(
            input_queue1, 1, input1_bytes);
        pipe.InitBuffer(
            acc_buf,
            HEAD_FP32_ACC_PHASE_BYTES + acc_bytes + aux_bytes);
    }

    __aicore__ inline void Process()
    {
        if (partial_count <= 1 ||
            static_cast<uint32_t>(GetBlockNum()) != partial_count) {
            return;
        }
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        if (core_idx >= partial_count) {
            return;
        }
        LocalTensor<float> acc = GetHeadFp32AccView<float>(
            acc_buf, acc_bytes / sizeof(float));
        uint32_t pair_idx = 0;
        uint32_t packet0_start = 0;
        uint32_t packet0_elems = 0;
        uint32_t packet1_start = 0;
        uint32_t packet1_elems = 0;
        const uint32_t pair_count = PairCount();
        const uint32_t owner =
            GetSplitSubblockTileOwner(partial_count);
        PrimeReducePair(
            owner, pair_count, pair_idx,
            packet0_start, packet0_elems,
            packet1_start, packet1_elems);
        BuildPackedOffsets();
        if (!use_input_scratch &&
            OwnedPairCount(owner, pair_count) > kEpochPairs) {
            InitializeEpochSum();
            ProcessReducePairs<true>(
                acc, pair_count, pair_idx,
                packet0_start, packet0_elems,
                packet1_start, packet1_elems);
            CompactAndReduceBanks(acc, AuxiliaryPacked());
            MergeFinalEpoch(acc);
        } else {
            ProcessReducePairs<false>(
                acc, pair_count, pair_idx,
                packet0_start, packet0_elems,
                packet1_start, packet1_elems);
            CompactFinalBanks(acc);
        }
        WritePartial(acc, core_idx);
        SyncAll();
        MergePartials();
    }

private:
    __aicore__ inline void GetRangeByParts(
        uint32_t total, uint32_t part_idx, uint32_t part_num,
        uint32_t &start, uint32_t &end)
    {
        const uint32_t base = total / part_num;
        const uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1U : 0U);
    }

    __aicore__ inline void CopyElements(
        LocalTensor<float> dst, uint32_t element_start,
        uint32_t elements, uint32_t packet_capacity)
    {
        if (elements == packet_capacity) {
            DataCopyParams params{
                1,
                static_cast<uint16_t>(
                    elements / kFloatAlignElems),
                0, 0};
            DataCopy(dst, input_gm[element_start], params);
            return;
        }
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(elements * sizeof(float)),
            0, 0, 0};
        DataCopyPadExtParams<float> pad{
            false, 0, 0, 0.0f};
        DataCopyPad(dst, input_gm[element_start], params, pad);
    }

    __aicore__ inline void BuildPackedOffsets()
    {
        LocalTensor<int32_t> offsets =
            GetHeadFp32AccView<int32_t>(
                acc_buf,
                compact_stride, acc_bytes);
        CreateVecIndex(
            offsets, static_cast<int32_t>(0), compact_stride);
        PipeBarrier<PIPE_V>();
        // Gather's discarded lane 519 repeats column 518 inside each bank.
        uint64_t padding_mask[2] = {
            1ULL << (kInnerSize % MAX_FP32_REPEAT_MASK), 0ULL};
        const uint32_t padding_base =
            (kInnerSize / MAX_FP32_REPEAT_MASK) * MAX_FP32_REPEAT_MASK;
        const UnaryRepeatParams padding_params{1, 1, 8, 8};
        Adds(offsets[padding_base], offsets[padding_base],
             static_cast<int32_t>(-1), padding_mask, 1, padding_params);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)), compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        uint32_t input_offset = 0;
        uint32_t bank_offset = element_start % kBankSlabElems;
        while (input_offset < elements) {
            const uint32_t span = MinU32(
                elements - input_offset,
                kBankSlabElems - bank_offset);
            MulAddDst(
                acc[bank_offset], input[input_offset],
                input[input_offset], span);
            input_offset += span;
            bank_offset = 0U;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
        AccumulateElements(acc, input, element_start, elements);
    }

    __aicore__ inline uint32_t PairCount()
    {
        return total_size == 0U
            ? 0U
            : (total_size - 1U) / PairElems() + 1U;
    }

    __aicore__ inline uint32_t PairElems()
    {
        return kPacket0Elems + packet1_capacity;
    }

    __aicore__ inline uint32_t PairElementStart(uint32_t pair_idx)
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(pair_idx) * PairElems());
    }

    __aicore__ inline uint32_t OwnedPairCount(
        uint32_t owner, uint32_t pair_count)
    {
        return owner < pair_count
            ? (pair_count - 1U - owner) / partial_count + 1U
            : 0U;
    }

    __aicore__ inline uint32_t PacketElements(
        uint32_t element_start, uint32_t packet_capacity)
    {
        return element_start < total_size
            ? MinU32(packet_capacity, total_size - element_start)
            : 0U;
    }

    __aicore__ inline void EnqueuePacket0(
        uint32_t element_start, uint32_t elements)
    {
        input_queue0.AllocTensor<float>(input_slot0);
        CopyElements(
            input_slot0, element_start, elements, kPacket0Elems);
        input_queue0.EnQue(input_slot0);
    }

    __aicore__ inline void EnqueuePacket1(
        uint32_t element_start, uint32_t elements)
    {
        input_queue1.AllocTensor<float>(input_slot1);
        CopyElements(
            input_slot1, element_start, elements, packet1_capacity);
        input_queue1.EnQue(input_slot1);
    }

    __aicore__ inline void PrimeReducePair(
        uint32_t owner, uint32_t pair_count, uint32_t &pair_idx,
        uint32_t &packet0_start, uint32_t &packet0_elems,
        uint32_t &packet1_start, uint32_t &packet1_elems)
    {
        pair_idx = owner;
        if (pair_idx >= pair_count) {
            packet0_elems = 0U;
            packet1_elems = 0U;
            return;
        }
        packet0_start = PairElementStart(pair_idx);
        packet0_elems = PacketElements(
            packet0_start, kPacket0Elems);
        if (packet0_elems > 0U) {
            EnqueuePacket0(packet0_start, packet0_elems);
        }
        packet1_elems = 0U;
        if (packet0_elems == kPacket0Elems) {
            packet1_start = packet0_start + kPacket0Elems;
            packet1_elems = PacketElements(
                packet1_start, packet1_capacity);
            if (packet1_elems > 0U) {
                EnqueuePacket1(packet1_start, packet1_elems);
            }
        }
    }

    __aicore__ inline void ConsumePacket0(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool initialize)
    {
        input_queue0.DeQue<float>(input_slot0);
        if (initialize) {
            InitializeElements(
                acc, input_slot0, element_start, elements);
        } else {
            AccumulateElements(
                acc, input_slot0, element_start, elements);
        }
        input_queue0.FreeTensor(input_slot0);
    }

    __aicore__ inline void ConsumePacket1(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool release)
    {
        input_queue1.DeQue<float>(input_slot1);
        AccumulateElements(
            acc, input_slot1, element_start, elements);
        if (release) {
            input_queue1.FreeTensor(input_slot1);
        }
    }

    template <bool EnableEpoch>
    __aicore__ inline void ProcessReducePairs(
        LocalTensor<float> acc, uint32_t pair_count,
        uint32_t pair_idx, uint32_t packet0_start,
        uint32_t packet0_elems, uint32_t packet1_start,
        uint32_t packet1_elems)
    {
        if (packet0_elems == 0U) {
            return;
        }
        uint32_t pairs_in_epoch = 0;
        ConsumePacket0(
            acc, packet0_start, packet0_elems, true);

        while (packet1_elems > 0U) {
            const uint64_t next_pair_idx64 =
                static_cast<uint64_t>(pair_idx) + partial_count;
            uint32_t next_pair_idx = 0U;
            uint32_t next_packet0_start = 0U;
            uint32_t next_packet0_elems = 0U;
            if (next_pair_idx64 < pair_count) {
                next_pair_idx = static_cast<uint32_t>(next_pair_idx64);
                next_packet0_start = PairElementStart(next_pair_idx);
                next_packet0_elems = PacketElements(
                    next_packet0_start, kPacket0Elems);
                EnqueuePacket0(
                    next_packet0_start, next_packet0_elems);
            }

            const bool retain_final_input =
                use_input_scratch && next_packet0_elems == 0U;
            ConsumePacket1(
                acc, packet1_start, packet1_elems,
                !retain_final_input);
            input1_holds_scratch = retain_final_input;
            if (next_packet0_elems == 0U) {
                return;
            }
            if constexpr (EnableEpoch) {
                ++pairs_in_epoch;
            }

            pair_idx = next_pair_idx;
            packet0_start = next_packet0_start;
            packet0_elems = next_packet0_elems;
            packet1_elems = 0U;
            if (packet0_elems == kPacket0Elems) {
                packet1_start = packet0_start + kPacket0Elems;
                packet1_elems = PacketElements(
                    packet1_start, packet1_capacity);
                if (packet1_elems > 0U) {
                    EnqueuePacket1(packet1_start, packet1_elems);
                }
            }
            if constexpr (EnableEpoch) {
                if (pairs_in_epoch == kEpochPairs) {
                    FlushLongEpoch(acc);
                    pairs_in_epoch = 0;
                }
            }
            ConsumePacket0(
                acc, packet0_start, packet0_elems, false);
        }
    }

    __aicore__ inline void InitializeEpochSum()
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Duplicate(epoch_sum, 0.0f, compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushLongEpoch(LocalTensor<float> acc)
    {
        CompactAndReduceBanks(acc, AuxiliaryPacked());
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(epoch_sum, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFinalEpoch(LocalTensor<float> acc)
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(acc, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> AuxiliaryPacked()
    {
        return GetHeadFp32AccView<float>(
            acc_buf,
            compact_stride,
            acc_bytes + compact_stride * sizeof(float));
    }

    __aicore__ inline void CompactFinalBanks(LocalTensor<float> acc)
    {
        if (use_input_scratch) {
            if (!input1_holds_scratch) {
                input_queue1.AllocTensor<float>(input_slot1);
            }
            CompactAndReduceBanks(acc, input_slot1);
            input_queue1.FreeTensor(input_slot1);
            input1_holds_scratch = false;
            return;
        }
        CompactAndReduceBanks(acc, AuxiliaryPacked());
    }

    __aicore__ inline void CompactAndReduceBanks(
        LocalTensor<float> acc, LocalTensor<float> packed)
    {
        LocalTensor<float> dense = GetHeadFp32AccView<float>(
            acc_buf, kBankSlabElems);
        LocalTensor<uint32_t> offsets =
            GetHeadFp32AccView<uint32_t>(
                acc_buf,
                compact_stride, acc_bytes);
        // Keep bank zero in place; the auxiliary Gather destination is disjoint.
        for (uint32_t bank = 1U; bank < kAccBanks; ++bank) {
            Gather(
                packed, dense, offsets,
                bank * kInnerSize * sizeof(float), compact_stride);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, packed, kInnerSize);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        uint64_t padding_mask[2] = {1ULL << (kInnerSize % kMergeCols), 0ULL};
        Duplicate(acc[kFullMergeTiles * kMergeCols], 0.0f, padding_mask, 1, 1, 8);
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        const DataCopyParams columns{
            kFullMergeTiles, kMergeCols / kFloatAlignElems, 0,
            static_cast<uint16_t>(
                (partial_count - 1U) * kMergeCols / kFloatAlignElems)};
        DataCopy(partial_gm[core_idx * kMergeCols], acc, columns);
        DataCopy(
            partial_gm[kFullMergeTiles * partial_count * kMergeCols +
                       core_idx * kTailStorageCols],
            acc[kFullMergeTiles * kMergeCols], kTailStorageCols);
    }

    __aicore__ inline void MergePartials()
    {
        const uint32_t core_idx = static_cast<uint32_t>(GetBlockIdx());
        for (uint32_t merge_idx = core_idx;
             merge_idx <= kFullMergeTiles;
             merge_idx += partial_count) {
            const uint32_t row_elems = merge_idx < kFullMergeTiles
                ? kMergeCols : kTailStorageCols;
            const uint32_t row_blocks = row_elems / kFloatAlignElems;
            const uint32_t base = merge_idx * partial_count * kMergeCols;
            LocalTensor<float> partials = GetHeadFp32AccView<float>(
                acc_buf, acc_bytes / sizeof(float));
            DataCopy(partials, partial_gm[base], partial_count * row_elems);
            const event_t loaded = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loaded);
            WaitFlag<HardEvent::MTE2_V>(loaded);

            SetMaskNorm();
            SetVectorMask<float>(row_elems);
            for (uint32_t step = 1U; step < partial_count; step <<= 1U) {
                const uint32_t group = step << 1U;
                const uint32_t repeats = (partial_count + step - 1U) / group;
                const uint32_t stride = repeats > 1U ? group * row_blocks : 0U;
                const BinaryRepeatParams params{
                    1, 1, 1, static_cast<uint8_t>(stride),
                    static_cast<uint8_t>(stride), static_cast<uint8_t>(stride)};
                Add<float, false>(
                    partials, partials, partials[step * row_elems],
                    MASK_PLACEHOLDER, static_cast<uint8_t>(repeats), params);
                PipeBarrier<PIPE_V>();
            }
            ResetMask();

            const uint32_t output_start = merge_idx * kMergeCols;
            const uint32_t count = MinU32(row_elems, kInnerSize - output_start);
            const event_t ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            CopyOutContiguous(output_gm, partials, output_start, count);
            if (merge_idx + partial_count <= kFullMergeTiles) {
                const event_t reuse = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(reuse);
                WaitFlag<HardEvent::MTE3_MTE2>(reuse);
            }
        }
    }

    static constexpr uint32_t kInnerSize = 519;
    static constexpr uint32_t kMergeCols = 64;
    static constexpr uint32_t kFullMergeTiles = kInnerSize / kMergeCols;
    static constexpr uint32_t kTailStorageCols = 8;
    static constexpr uint32_t kAccBanks = 8;
    static constexpr uint32_t kBankSlabElems =
        kAccBanks * kInnerSize;
    static constexpr uint32_t kPacket0Elems = 23040;
    static constexpr uint32_t kPacket1StreamElems = 20224;
    static constexpr uint32_t kPacket1EpochElems = 19200;
    static constexpr uint32_t kStreamAuxBuffers = 1;
    static constexpr uint32_t kEpochAuxBuffers = 3;
    static constexpr uint32_t kEpochPairs = 96;
    static constexpr uint32_t kAccGuardBytes = 32;
    static constexpr uint32_t kInputGuardBytes = 32;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);

    // Independent depth-zero queues retain unequal persistent physical slots.
    TQue<TPosition::VECIN, 0> input_queue0;
    TQue<TPosition::VECIN, 0> input_queue1;
    LocalTensor<float> input_slot0;
    LocalTensor<float> input_slot1;
    TBuf<TPosition::VECOUT> acc_buf;
    GlobalTensor<float> input_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t total_size = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    uint32_t compact_stride = 0;
    uint32_t packet1_capacity = 0;
    uint32_t input0_bytes = 0;
    uint32_t input1_bytes = 0;
    uint32_t acc_bytes = 0;
    uint32_t aux_bytes = 0;
    bool use_input_scratch = false;
    bool input1_holds_scratch = false;
};

class KernelHeadFp32Width521Forward {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
        const SquareSumV1TilingData &tiling)
    {
        total_size = tiling.total_size;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        packet1_capacity = tiling.tile_elems;
        use_input_scratch =
            packet1_capacity == kPacket1StreamElems;
        packet0_capacity = use_input_scratch
                               ? kPacket0StreamElems
                               : kPacket0EpochElems;
        input1_holds_scratch = false;
        compact_stride = PaddedStrideElems<float>(kInnerSize);
        acc_bytes = AlignUp32(
            kAccBanks * kInnerSize * sizeof(float) +
            kAccGuardBytes);
        aux_bytes = AlignUp32(
            (use_input_scratch
                 ? 0U
                 : kEpochAuxBuffers) * compact_stride *
            sizeof(uint32_t));
        input0_bytes = AlignUp32(
            packet0_capacity * sizeof(float) +
            kInputGuardBytes);
        input1_bytes = AlignUp32(
            packet1_capacity * sizeof(float) +
            kInputGuardBytes);

        input_gm.SetGlobalBuffer(
            (__gm__ float *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ float *)output, tiling.output_size);
        SetSysWorkspace(workspace);
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        partial_gm.SetGlobalBuffer(
            (__gm__ float *)user_workspace,
            partial_count * partial_stride);

        pipe.InitBuffer(
            input_queue0, 1, input0_bytes);
        pipe.InitBuffer(
            input_queue1, 1, input1_bytes);
        pipe.InitBuffer(
            acc_buf,
            HEAD_FP32_ACC_PHASE_BYTES + acc_bytes + aux_bytes);
    }

    __aicore__ inline void Process()
    {
        if (partial_count <= 1 ||
            static_cast<uint32_t>(GetBlockNum()) != partial_count) {
            return;
        }
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        if (core_idx >= partial_count) {
            return;
        }
        LocalTensor<float> acc = GetHeadFp32AccView<float>(
            acc_buf, acc_bytes / sizeof(float));
        const uint32_t owner =
            GetSplitSubblockTileOwner(partial_count);
        if (use_input_scratch) {
            ProcessMode<true>(acc, owner);
            WritePartial<true>(acc, core_idx);
        } else {
            ProcessMode<false>(acc, owner);
            WritePartial<false>(acc, core_idx);
        }
        SyncAll();
        if (use_input_scratch) {
            MergeStreamPartials(acc);
        } else {
            MergePartials();
        }
    }

private:
    template <bool Stream>
    __aicore__ inline void ProcessMode(LocalTensor<float> acc, uint32_t owner)
    {
        uint32_t pair_idx = 0;
        uint32_t packet0_start = 0;
        uint32_t packet0_elems = 0;
        uint32_t packet1_start = 0;
        uint32_t packet1_elems = 0;
        const uint32_t pair_count = PairCount<Stream>();
        PrimeReducePair<Stream>(
            owner, pair_count, pair_idx,
            packet0_start, packet0_elems,
            packet1_start, packet1_elems);
        if constexpr (Stream) {
            ProcessReducePairs<true, false>(
                acc, pair_count, pair_idx,
                packet0_start, packet0_elems,
                packet1_start, packet1_elems);
            CompactFinalBanks<true>(acc);
        } else {
            BuildBaseOffsets(AuxiliaryOffsets());
            if (OwnedPairCount(owner, pair_count) > kEpochPairs) {
                InitializeEpochSum();
                ProcessReducePairs<false, true>(
                    acc, pair_count, pair_idx,
                    packet0_start, packet0_elems,
                    packet1_start, packet1_elems);
                CompactAndReduceBanks(
                    acc, AuxiliaryPacked(), AuxiliaryOffsets());
                MergeFinalEpoch(acc);
            } else {
                ProcessReducePairs<false, false>(
                    acc, pair_count, pair_idx,
                    packet0_start, packet0_elems,
                    packet1_start, packet1_elems);
                CompactFinalBanks<false>(acc);
            }
        }
    }

    __aicore__ inline void GetRangeByParts(
        uint32_t total, uint32_t part_idx, uint32_t part_num,
        uint32_t &start, uint32_t &end)
    {
        const uint32_t base = total / part_num;
        const uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1U : 0U);
    }

    template <uint32_t PacketCapacity>
    __aicore__ inline void CopyElements(
        LocalTensor<float> dst, uint32_t element_start,
        uint32_t elements)
    {
        if (elements == PacketCapacity) {
            DataCopyParams params{
                1,
                static_cast<uint16_t>(
                    PacketCapacity / kFloatAlignElems),
                0, 0};
            DataCopy(dst, input_gm[element_start], params);
            return;
        }
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(elements * sizeof(float)),
            0, 0, 0};
        DataCopyPadExtParams<float> pad{
            false, 0, 0, 0.0f};
        DataCopyPad(dst, input_gm[element_start], params, pad);
    }

    __aicore__ inline void BuildBaseOffsets(
        LocalTensor<int32_t> offsets)
    {
        CreateVecIndex(
            offsets, static_cast<int32_t>(0), compact_stride);
        PipeBarrier<PIPE_V>();
        const uint32_t padding_base =
            (kInnerSize / MAX_FP32_REPEAT_MASK) * MAX_FP32_REPEAT_MASK;
        const uint32_t logical_tail = kInnerSize - padding_base;
        const uint32_t storage_tail = compact_stride - padding_base;
        uint64_t padding_mask[2] = {
            ((1ULL << storage_tail) - 1ULL) ^
                ((1ULL << logical_tail) - 1ULL),
            0ULL};
        Duplicate(
            offsets[padding_base],
            static_cast<int32_t>(kInnerSize - 1U),
            padding_mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)), compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpandPackedOffsets(
        LocalTensor<int32_t> offsets)
    {
        const int32_t bank_step = static_cast<int32_t>(
            kInnerSize * sizeof(float));
        Adds(
            offsets[compact_stride], offsets,
            bank_step, compact_stride);
        PipeBarrier<PIPE_V>();
        Adds(
            offsets[2U * compact_stride], offsets,
            2 * bank_step, 2U * compact_stride);
        PipeBarrier<PIPE_V>();
        Adds(
            offsets[4U * compact_stride], offsets,
            4 * bank_step, 4U * compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        uint32_t input_offset = 0;
        uint32_t bank_offset = element_start % kBankSlabElems;
        SetMaskCount();
        while (input_offset < elements) {
            const uint32_t span = MinU32(
                elements - input_offset,
                kBankSlabElems - bank_offset);
            SetVectorMask<float, MaskMode::COUNTER>(span);
            MulAddDst<float, float, false>(
                acc[bank_offset], input[input_offset],
                input[input_offset], MASK_PLACEHOLDER, 1,
                {1, 1, 1, 8, 8, 8});
            input_offset += span;
            bank_offset = 0U;
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
        AccumulateElements(acc, input, element_start, elements);
    }

    template <bool Stream>
    __aicore__ inline uint32_t PairCount()
    {
        return total_size == 0U
            ? 0U
            : (total_size - 1U) / PairElems<Stream>() + 1U;
    }

    template <bool Stream>
    __aicore__ inline uint32_t PairElems()
    {
        return Stream
            ? kPacket0StreamElems + kPacket1StreamElems
            : kPacket0EpochElems + kPacket1EpochElems;
    }

    template <bool Stream>
    __aicore__ inline uint32_t PairElementStart(uint32_t pair_idx)
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(pair_idx) * PairElems<Stream>());
    }

    __aicore__ inline uint32_t OwnedPairCount(
        uint32_t owner, uint32_t pair_count)
    {
        return owner < pair_count
            ? (pair_count - 1U - owner) / partial_count + 1U
            : 0U;
    }

    template <uint32_t PacketCapacity>
    __aicore__ inline uint32_t PacketElements(uint32_t element_start)
    {
        return element_start < total_size
            ? MinU32(PacketCapacity, total_size - element_start)
            : 0U;
    }

    template <bool Stream>
    __aicore__ inline void EnqueuePacket0(
        uint32_t element_start, uint32_t elements)
    {
        constexpr uint32_t capacity = Stream
            ? kPacket0StreamElems : kPacket0EpochElems;
        input_queue0.AllocTensor<float>(input_slot0);
        CopyElements<capacity>(input_slot0, element_start, elements);
        input_queue0.EnQue(input_slot0);
    }

    template <bool Stream>
    __aicore__ inline void EnqueuePacket1(
        uint32_t element_start, uint32_t elements)
    {
        constexpr uint32_t capacity = Stream
            ? kPacket1StreamElems : kPacket1EpochElems;
        input_queue1.AllocTensor<float>(input_slot1);
        CopyElements<capacity>(input_slot1, element_start, elements);
        input_queue1.EnQue(input_slot1);
    }

    template <bool Stream>
    __aicore__ inline void PrimeReducePair(
        uint32_t owner, uint32_t pair_count, uint32_t &pair_idx,
        uint32_t &packet0_start, uint32_t &packet0_elems,
        uint32_t &packet1_start, uint32_t &packet1_elems)
    {
        constexpr uint32_t packet0_elems_max = Stream
            ? kPacket0StreamElems : kPacket0EpochElems;
        constexpr uint32_t packet1_elems_max = Stream
            ? kPacket1StreamElems : kPacket1EpochElems;
        pair_idx = owner;
        if (pair_idx >= pair_count) {
            packet0_elems = 0U;
            packet1_elems = 0U;
            return;
        }
        packet0_start = PairElementStart<Stream>(pair_idx);
        packet0_elems = PacketElements<packet0_elems_max>(packet0_start);
        if (packet0_elems > 0U) {
            EnqueuePacket0<Stream>(packet0_start, packet0_elems);
        }
        packet1_elems = 0U;
        if (packet0_elems == packet0_elems_max) {
            packet1_start = packet0_start + packet0_elems_max;
            packet1_elems = PacketElements<packet1_elems_max>(packet1_start);
            if (packet1_elems > 0U) {
                EnqueuePacket1<Stream>(packet1_start, packet1_elems);
            }
        }
    }

    __aicore__ inline void ConsumePacket0(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool initialize)
    {
        input_queue0.DeQue<float>(input_slot0);
        if (initialize) {
            InitializeElements(
                acc, input_slot0, element_start, elements);
        } else {
            AccumulateElements(
                acc, input_slot0, element_start, elements);
        }
        input_queue0.FreeTensor(input_slot0);
    }

    __aicore__ inline void ConsumePacket1(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool release)
    {
        input_queue1.DeQue<float>(input_slot1);
        AccumulateElements(
            acc, input_slot1, element_start, elements);
        if (release) {
            input_queue1.FreeTensor(input_slot1);
        }
    }

    template <bool Stream, bool EnableEpoch>
    __aicore__ inline void ProcessReducePairs(
        LocalTensor<float> acc, uint32_t pair_count,
        uint32_t pair_idx, uint32_t packet0_start,
        uint32_t packet0_elems, uint32_t packet1_start,
        uint32_t packet1_elems)
    {
        constexpr uint32_t packet0_elems_max = Stream
            ? kPacket0StreamElems : kPacket0EpochElems;
        constexpr uint32_t packet1_elems_max = Stream
            ? kPacket1StreamElems : kPacket1EpochElems;
        if (packet0_elems == 0U) {
            return;
        }
        uint32_t pairs_in_epoch = 0;
        ConsumePacket0(
            acc, packet0_start, packet0_elems, true);

        while (packet1_elems > 0U) {
            const uint64_t next_pair_idx64 =
                static_cast<uint64_t>(pair_idx) + partial_count;
            uint32_t next_pair_idx = 0U;
            uint32_t next_packet0_start = 0U;
            uint32_t next_packet0_elems = 0U;
            if (next_pair_idx64 < pair_count) {
                next_pair_idx = static_cast<uint32_t>(next_pair_idx64);
                next_packet0_start = PairElementStart<Stream>(next_pair_idx);
                next_packet0_elems =
                    PacketElements<packet0_elems_max>(next_packet0_start);
                EnqueuePacket0<Stream>(
                    next_packet0_start, next_packet0_elems);
            }

            const bool retain_final_input =
                Stream && next_packet0_elems == 0U;
            ConsumePacket1(
                acc, packet1_start, packet1_elems,
                !retain_final_input);
            input1_holds_scratch = retain_final_input;
            if (next_packet0_elems == 0U) {
                return;
            }
            if constexpr (EnableEpoch) {
                ++pairs_in_epoch;
            }

            pair_idx = next_pair_idx;
            packet0_start = next_packet0_start;
            packet0_elems = next_packet0_elems;
            packet1_elems = 0U;
            if (packet0_elems == packet0_elems_max) {
                packet1_start = packet0_start + packet0_elems_max;
                packet1_elems =
                    PacketElements<packet1_elems_max>(packet1_start);
                if (packet1_elems > 0U) {
                    EnqueuePacket1<Stream>(packet1_start, packet1_elems);
                }
            }
            if constexpr (EnableEpoch) {
                if (pairs_in_epoch == kEpochPairs) {
                    FlushLongEpoch(acc);
                    pairs_in_epoch = 0;
                }
            }
            ConsumePacket0(
                acc, packet0_start, packet0_elems, false);
        }
    }

    __aicore__ inline void InitializeEpochSum()
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Duplicate(epoch_sum, 0.0f, compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushLongEpoch(LocalTensor<float> acc)
    {
        CompactAndReduceBanks(
            acc, AuxiliaryPacked(), AuxiliaryOffsets());
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(epoch_sum, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFinalEpoch(LocalTensor<float> acc)
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(acc, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> AuxiliaryPacked()
    {
        return GetHeadFp32AccView<float>(
            acc_buf,
            compact_stride,
            acc_bytes + compact_stride * sizeof(float));
    }

    __aicore__ inline LocalTensor<int32_t> AuxiliaryOffsets()
    {
        return GetHeadFp32AccView<int32_t>(
            acc_buf, compact_stride, acc_bytes);
    }

    template <bool Stream>
    __aicore__ inline void CompactFinalBanks(LocalTensor<float> acc)
    {
        if constexpr (Stream) {
            input_queue0.AllocTensor<float>(input_slot0);
            LocalTensor<int32_t> offsets =
                input_slot0.ReinterpretCast<int32_t>();
            BuildBaseOffsets(offsets);
            ExpandPackedOffsets(offsets);
            if (!input1_holds_scratch) {
                input_queue1.AllocTensor<float>(input_slot1);
            }
            CompactPackedBanks(acc, input_slot1, offsets);
            input_queue1.FreeTensor(input_slot1);
            input_queue0.FreeTensor(input_slot0);
            input1_holds_scratch = false;
        } else {
            CompactAndReduceBanks(
                acc, AuxiliaryPacked(), AuxiliaryOffsets());
        }
    }

    __aicore__ inline void CompactAndReduceBanks(
        LocalTensor<float> acc, LocalTensor<float> packed,
        LocalTensor<int32_t> offsets)
    {
        LocalTensor<float> dense = GetHeadFp32AccView<float>(
            acc_buf, kBankSlabElems);
        // Keep bank zero in place; the auxiliary Gather destination is disjoint.
        for (uint32_t bank = 1U; bank < kAccBanks; ++bank) {
            Gather(
                packed, dense, offsets.ReinterpretCast<uint32_t>(),
                bank * kInnerSize * sizeof(float), compact_stride);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, packed, kInnerSize);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CompactPackedBanks(
        LocalTensor<float> acc, LocalTensor<float> packed,
        LocalTensor<int32_t> offsets)
    {
        LocalTensor<float> dense = GetHeadFp32AccView<float>(
            acc_buf, kBankSlabElems);
        Gather(
            packed, dense, offsets.ReinterpretCast<uint32_t>(),
            0U, kAccBanks * compact_stride);
        PipeBarrier<PIPE_V>();
        Add(packed, packed, packed[4U * compact_stride], 4U * compact_stride);
        PipeBarrier<PIPE_V>();
        Add(packed, packed, packed[2U * compact_stride], 2U * compact_stride);
        PipeBarrier<PIPE_V>();
        Add(acc, packed, packed[compact_stride], kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    template <bool Stream>
    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        const uint32_t tail_cols = kInnerSize % kMergeCols;
        uint64_t padding_mask[2] = {
            ((1ULL << kTailStorageCols) - 1ULL) ^
                ((1ULL << tail_cols) - 1ULL),
            0ULL};
        Duplicate(
            acc[kFullMergeTiles * kMergeCols],
            0.0f, padding_mask, 1, 1, 8);
        if constexpr (Stream) {
            Duplicate(
                acc[compact_stride], 0.0f,
                kStreamPartialStride - compact_stride);
        }
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        if constexpr (Stream) {
            DataCopy(
                partial_gm[core_idx * partial_stride], acc,
                kStreamPartialStride);
            return;
        }
        const DataCopyParams columns{
            kFullMergeTiles, kMergeCols / kFloatAlignElems, 0,
            static_cast<uint16_t>(
                (partial_count - 1U) * kMergeCols / kFloatAlignElems)};
        DataCopy(partial_gm[core_idx * kMergeCols], acc, columns);
        DataCopy(
            partial_gm[kFullMergeTiles * partial_count * kMergeCols +
                       core_idx * kTailStorageCols],
            acc[kFullMergeTiles * kMergeCols], kTailStorageCols);
    }

    __aicore__ inline void MergeStreamPartials(LocalTensor<float> acc)
    {
        const uint32_t core_idx = static_cast<uint32_t>(GetBlockIdx());
        for (uint32_t merge_idx = core_idx;
             merge_idx < kStreamMergeTiles;
             merge_idx += partial_count) {
            // Reuse the dead input slot for 64 contiguous rows (32768 bytes).
            input_queue0.AllocTensor<float>(input_slot0);
            const DataCopyParams columns{
                static_cast<uint16_t>(partial_count),
                kStreamMergeCols / kFloatAlignElems,
                static_cast<uint16_t>(
                    (partial_stride - kStreamMergeCols) / kFloatAlignElems),
                0};
            DataCopy(
                input_slot0, partial_gm[merge_idx * kStreamMergeCols], columns);
            if (partial_count < kStreamMergeRows) {
                Duplicate(
                    input_slot0[partial_count * kStreamMergeCols], 0.0f,
                    (kStreamMergeRows - partial_count) * kStreamMergeCols);
            }
            input_queue0.EnQue(input_slot0);
            input_queue0.DeQue<float>(input_slot0);
            PipeBarrier<PIPE_V>();

            for (uint32_t half = (kStreamMergeRows / 2U) * kStreamMergeCols;
                 half > kStreamMergeCols; half >>= 1U) {
                Add(input_slot0, input_slot0, input_slot0[half], half);
                PipeBarrier<PIPE_V>();
            }
            Add(acc, input_slot0, input_slot0[kStreamMergeCols], kStreamMergeCols);
            PipeBarrier<PIPE_V>();
            input_queue0.FreeTensor(input_slot0);

            const uint32_t output_start = merge_idx * kStreamMergeCols;
            const uint32_t count = MinU32(kStreamMergeCols, kInnerSize - output_start);
            const event_t ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            CopyOutContiguous(output_gm, acc, output_start, count);
            if (merge_idx + partial_count < kStreamMergeTiles) {
                const event_t reuse = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(reuse);
                WaitFlag<HardEvent::MTE3_V>(reuse);
            }
        }
    }

    __aicore__ inline void MergePartials()
    {
        const uint32_t core_idx = static_cast<uint32_t>(GetBlockIdx());
        for (uint32_t merge_idx = core_idx;
             merge_idx <= kFullMergeTiles;
             merge_idx += partial_count) {
            const uint32_t row_elems = merge_idx < kFullMergeTiles
                ? kMergeCols : kTailStorageCols;
            const uint32_t row_blocks = row_elems / kFloatAlignElems;
            const uint32_t base = merge_idx * partial_count * kMergeCols;
            LocalTensor<float> partials = GetHeadFp32AccView<float>(
                acc_buf, acc_bytes / sizeof(float));
            DataCopy(partials, partial_gm[base], partial_count * row_elems);
            const event_t loaded = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loaded);
            WaitFlag<HardEvent::MTE2_V>(loaded);

            SetMaskNorm();
            SetVectorMask<float>(row_elems);
            for (uint32_t step = 1U; step < partial_count; step <<= 1U) {
                const uint32_t group = step << 1U;
                const uint32_t repeats = (partial_count + step - 1U) / group;
                const uint32_t stride = repeats > 1U ? group * row_blocks : 0U;
                const BinaryRepeatParams params{
                    1, 1, 1, static_cast<uint8_t>(stride),
                    static_cast<uint8_t>(stride), static_cast<uint8_t>(stride)};
                Add<float, false>(
                    partials, partials, partials[step * row_elems],
                    MASK_PLACEHOLDER, static_cast<uint8_t>(repeats), params);
                PipeBarrier<PIPE_V>();
            }
            ResetMask();

            const uint32_t output_start = merge_idx * kMergeCols;
            const uint32_t count = MinU32(row_elems, kInnerSize - output_start);
            const event_t ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            CopyOutContiguous(output_gm, partials, output_start, count);
            if (merge_idx + partial_count <= kFullMergeTiles) {
                const event_t reuse = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(reuse);
                WaitFlag<HardEvent::MTE3_MTE2>(reuse);
            }
        }
    }

    static constexpr uint32_t kInnerSize = 521;
    static constexpr uint32_t kMergeCols = 32;
    static constexpr uint32_t kFullMergeTiles = kInnerSize / kMergeCols;
    static constexpr uint32_t kTailStorageCols = 16;
    static constexpr uint32_t kStreamPartialStride = 640;
    static constexpr uint32_t kStreamMergeCols = 128;
    static constexpr uint32_t kStreamMergeRows = 64;
    static constexpr uint32_t kStreamMergeTiles =
        (kInnerSize + kStreamMergeCols - 1U) / kStreamMergeCols;
    static constexpr uint32_t kAccBanks = 8;
    static constexpr uint32_t kBankSlabElems =
        kAccBanks * kInnerSize;
    static constexpr uint32_t kPacket0StreamElems = 23040;
    static constexpr uint32_t kPacket0EpochElems = 23040;
    static constexpr uint32_t kPacket1StreamElems = 20736;
    static constexpr uint32_t kPacket1EpochElems = 19200;
    static constexpr uint32_t kEpochAuxBuffers = 3;
    static constexpr uint32_t kEpochPairs = 96;
    static constexpr uint32_t kAccGuardBytes = 32;
    static constexpr uint32_t kInputGuardBytes = 32;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);

    // Independent depth-zero queues retain unequal persistent physical slots.
    TQue<TPosition::VECIN, 0> input_queue0;
    TQue<TPosition::VECIN, 0> input_queue1;
    LocalTensor<float> input_slot0;
    LocalTensor<float> input_slot1;
    TBuf<TPosition::VECOUT> acc_buf;
    GlobalTensor<float> input_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t total_size = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    uint32_t compact_stride = 0;
    uint32_t packet0_capacity = 0;
    uint32_t packet1_capacity = 0;
    uint32_t input0_bytes = 0;
    uint32_t input1_bytes = 0;
    uint32_t acc_bytes = 0;
    uint32_t aux_bytes = 0;
    bool use_input_scratch = false;
    bool input1_holds_scratch = false;
};

class KernelHeadFp32Width520Aligned {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
        const SquareSumV1TilingData &tiling)
    {
        reduce_size = tiling.reduce_size;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        acc_stride = kInnerSize;

        const uint32_t usable_bytes =
            tiling.tile_elems * kHostBytesPerTileElem +
            kClassUbReclaimBytes;
        const uint32_t fixed_bytes =
            kOutputBufferNum * kInnerSize * sizeof(float) +
            kAccBanks * acc_stride * sizeof(float);
        const uint32_t row_bytes =
            BUFFER_NUM * kInnerSize * sizeof(float);
        rows_per_tile =
            usable_bytes > fixed_bytes
                ? (usable_bytes - fixed_bytes) / row_bytes
                : kAccBanks;
        rows_per_tile = MinU32(
            rows_per_tile, MAX_REPEAT_ROWS);
        rows_per_tile =
            (rows_per_tile / kAccBanks) * kAccBanks;
        if (rows_per_tile < kAccBanks) {
            rows_per_tile = kAccBanks;
        }
        dense_tile_elems = rows_per_tile * kInnerSize;

        input_gm.SetGlobalBuffer(
            (__gm__ float *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ float *)output, tiling.output_size);
        SetSysWorkspace(workspace);
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        partial_gm.SetGlobalBuffer(
            (__gm__ float *)user_workspace,
            partial_count * partial_stride);

        pipe.InitBuffer(
            input_queue, BUFFER_NUM,
            dense_tile_elems * sizeof(float));
        pipe.InitBuffer(
            acc_buf,
            (kAccBanks + kEpochBuffers) * acc_stride * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        if (partial_count <= 1 ||
            static_cast<uint32_t>(GetBlockNum()) != partial_count ||
            core_idx >= partial_count) {
            return;
        }
        LocalTensor<float> acc = acc_buf.Get<float>();
        uint32_t tile_idx = 0;
        uint32_t previous_rows = 0;
        const uint32_t owner =
            GetSplitSubblockTileOwner(partial_count);
        const uint32_t tile_count =
            (reduce_size + rows_per_tile - 1U) / rows_per_tile;
        const uint32_t owned_tiles =
            OwnedTileCount(owner, tile_count);
        const uint32_t epoch_tiles =
            SelectEpochTiles(owned_tiles);
        PrimeReduceTile(
            owner,
            tile_idx, previous_rows);
        if (epoch_tiles < owned_tiles) {
            InitializeEpochSum();
            ProcessReduceTiles<true>(
                acc, tile_idx, previous_rows, epoch_tiles);
        } else {
            ProcessReduceTiles<false>(
                acc, tile_idx, previous_rows, epoch_tiles);
        }
        ReduceBanks(acc);
        if (epoch_tiles < owned_tiles) {
            MergeFinalEpoch(acc);
        }
        WritePartial(acc, core_idx);
        SyncAll();
        MergeHeadFp32ColumnPartials<kInnerSize>(
            input_queue, partial_gm, output_gm,
            partial_count, partial_stride);
    }

private:
    __aicore__ inline void GetRangeByParts(
        uint32_t total, uint32_t part_idx, uint32_t part_num,
        uint32_t &start, uint32_t &end)
    {
        const uint32_t base = total / part_num;
        const uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1U : 0U);
    }

    __aicore__ inline void CopyRows(
        LocalTensor<float> dst, uint32_t row_start,
        uint32_t rows)
    {
        const uint32_t logical_elems = rows * kInnerSize;
        DataCopyParams params{
            1,
            static_cast<uint16_t>(
                logical_elems / kFloatAlignElems),
            0, 0};
        DataCopy(
            dst, input_gm[row_start * kInnerSize], params);
    }

    __aicore__ inline void AccumulateRows(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t rows)
    {
        const BinaryRepeatParams contiguous_params{
            1, 1, 1, 8, 8, 8};
        const uint32_t full_rows =
            (rows / kAccBanks) * kAccBanks;
        SetMaskCount();
        if (full_rows > 0U) {
            SetVectorMask<float, MaskMode::COUNTER>(kCohortElems);
            for (uint32_t row = 0; row < full_rows;
                 row += kAccBanks) {
                MulAddDst<float, float, false>(
                    acc,
                    input[row * kInnerSize],
                    input[row * kInnerSize],
                    MASK_PLACEHOLDER, 1, contiguous_params);
                // The next cohort reads and updates these same eight banks.
                PipeBarrier<PIPE_V>();
            }
        }
        if (full_rows < rows) {
            const uint32_t tail_elems =
                (rows - full_rows) * kInnerSize;
            SetVectorMask<float, MaskMode::COUNTER>(tail_elems);
            MulAddDst<float, float, false>(
                acc, input[full_rows * kInnerSize],
                input[full_rows * kInnerSize],
                MASK_PLACEHOLDER, 1, contiguous_params);
            PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void InitializeRows(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t rows)
    {
        const uint32_t initial_rows =
            MinU32(kAccBanks, rows);
        if (initial_rows < kAccBanks) {
            Duplicate(acc, 0.0f, kAccBanks * acc_stride);
            PipeBarrier<PIPE_V>();
        }
        const BinaryRepeatParams contiguous_params{
            1, 1, 1, 8, 8, 8};
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(
            initial_rows * kInnerSize);
        Mul<float, false>(
            acc, input, input,
            MASK_PLACEHOLDER, 1, contiguous_params);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        if (rows > initial_rows) {
            AccumulateRows(
                acc, input[initial_rows * kInnerSize],
                rows - initial_rows);
        }
    }

    __aicore__ inline void PrimeReduceTile(
        uint32_t core_idx, uint32_t &tile_idx,
        uint32_t &previous_rows)
    {
        const uint32_t tile_count =
            (reduce_size + rows_per_tile - 1U) / rows_per_tile;
        tile_idx = core_idx;
        previous_rows = 0;
        if (tile_idx < tile_count) {
            const uint32_t row_start = tile_idx * rows_per_tile;
            previous_rows = MinU32(
                rows_per_tile, reduce_size - row_start);
            LocalTensor<float> input =
                input_queue.AllocTensor<float>();
            CopyRows(input, row_start, previous_rows);
            input_queue.EnQue(input);
            tile_idx += partial_count;
        }
    }

    template <bool EnableEpoch>
    __aicore__ inline void ProcessReduceTiles(
        LocalTensor<float> acc, uint32_t tile_idx,
        uint32_t previous_rows, uint32_t epoch_tiles)
    {
        const uint32_t tile_count =
            (reduce_size + rows_per_tile - 1U) / rows_per_tile;
        if (previous_rows == 0U) {
            return;
        }
        if (tile_idx >= tile_count) {
            LocalTensor<float> current =
                input_queue.DeQue<float>();
            InitializeRows(acc, current, previous_rows);
            input_queue.FreeTensor(current);
            return;
        }

        uint32_t row_start = tile_idx * rows_per_tile;
        uint32_t current_rows = MinU32(
            rows_per_tile, reduce_size - row_start);
        LocalTensor<float> next =
            input_queue.AllocTensor<float>();
        CopyRows(next, row_start, current_rows);
        input_queue.EnQue(next);
        tile_idx += partial_count;

        LocalTensor<float> current =
            input_queue.DeQue<float>();
        InitializeRows(acc, current, previous_rows);
        input_queue.FreeTensor(current);
        uint32_t tiles_in_epoch = 1U;
        previous_rows = current_rows;

        while (tile_idx < tile_count) {
            row_start = tile_idx * rows_per_tile;
            current_rows = MinU32(
                rows_per_tile, reduce_size - row_start);
            next =
                input_queue.AllocTensor<float>();
            CopyRows(next, row_start, current_rows);
            input_queue.EnQue(next);

            current =
                input_queue.DeQue<float>();
            if constexpr (EnableEpoch) {
                if (tiles_in_epoch == epoch_tiles) {
                    FlushEpoch(acc);
                    InitializeRows(acc, current, previous_rows);
                    tiles_in_epoch = 1U;
                } else {
                    AccumulateRows(acc, current, previous_rows);
                    ++tiles_in_epoch;
                }
            } else {
                AccumulateRows(acc, current, previous_rows);
            }
            input_queue.FreeTensor(current);
            previous_rows = current_rows;
            tile_idx += partial_count;
        }

        current = input_queue.DeQue<float>();
        if constexpr (EnableEpoch) {
            if (tiles_in_epoch == epoch_tiles) {
                FlushEpoch(acc);
                InitializeRows(acc, current, previous_rows);
            } else {
                AccumulateRows(acc, current, previous_rows);
            }
        } else {
            AccumulateRows(acc, current, previous_rows);
        }
        input_queue.FreeTensor(current);
    }

    __aicore__ inline uint32_t OwnedTileCount(
        uint32_t owner, uint32_t tile_count)
    {
        return owner < tile_count
            ? (tile_count - 1U - owner) / partial_count + 1U
            : 0U;
    }

    __aicore__ inline uint32_t WorkspaceLevels()
    {
        uint32_t levels = 0U;
        uint32_t width = 1U;
        while (width < partial_count) {
            width <<= 1U;
            ++levels;
        }
        return levels;
    }

    __aicore__ inline uint32_t SelectEpochTiles(
        uint32_t owned_tiles)
    {
        if (owned_tiles <= 1U) {
            return owned_tiles;
        }
        const uint32_t terms_per_tile = rows_per_tile / kAccBanks;
        const uint32_t fixed_levels =
            kBankFoldLevels + WorkspaceLevels();
        const uint32_t no_epoch_levels =
            owned_tiles * terms_per_tile - 1U + fixed_levels;
        if (no_epoch_levels <= kMaxRoundingLevels) {
            return owned_tiles;
        }

        uint32_t epoch_tiles = MinU32(
            owned_tiles,
            (kMaxRoundingLevels - fixed_levels + 1U) /
                terms_per_tile);
        while (epoch_tiles > 1U) {
            const uint32_t epoch_count =
                (owned_tiles + epoch_tiles - 1U) / epoch_tiles;
            const uint32_t levels =
                epoch_tiles * terms_per_tile - 1U +
                kBankFoldLevels + epoch_count - 1U +
                WorkspaceLevels();
            if (levels <= kMaxRoundingLevels) {
                return epoch_tiles;
            }
            --epoch_tiles;
        }
        return 1U;
    }

    __aicore__ inline LocalTensor<float> EpochSum()
    {
        return acc_buf.GetWithOffset<float>(
            acc_stride,
            kAccBanks * acc_stride * sizeof(float));
    }

    __aicore__ inline void InitializeEpochSum()
    {
        Duplicate(EpochSum(), 0.0f, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushEpoch(LocalTensor<float> acc)
    {
        ReduceBanks(acc);
        Add(EpochSum(), EpochSum(), acc, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFinalEpoch(LocalTensor<float> acc)
    {
        Add(acc, EpochSum(), acc, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceBanks(LocalTensor<float> acc)
    {
        for (uint32_t step = 1;
             step < kAccBanks; step <<= 1) {
            for (uint32_t bank = 0; bank < kAccBanks;
                 bank += (step << 1)) {
                Add(
                    acc[bank * acc_stride],
                    acc[bank * acc_stride],
                    acc[(bank + step) * acc_stride],
                    kInnerSize);
            }
            // The next tree level consumes this level's destinations.
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        CopyOutContiguous(
            partial_gm, acc,
            core_idx * partial_stride, kInnerSize);
    }

    static constexpr uint32_t kInnerSize = 520;
    static constexpr uint32_t kAccBanks = 8;
    static constexpr uint32_t kCohortElems =
        kAccBanks * kInnerSize;
    static constexpr uint32_t kOutputBufferNum = 1;
    static constexpr uint32_t kEpochBuffers = 1;
    static constexpr uint32_t kBankFoldLevels = 3;
    // Reserve 76 FP32 error levels for squaring and the reference side.
    static constexpr uint32_t kMaxRoundingLevels = 1600;
    static constexpr uint32_t kHostBytesPerTileElem = 28;
    static constexpr uint32_t kClassUbReclaimBytes = 16 * 1024;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TBuf<TPosition::VECOUT> acc_buf;
    GlobalTensor<float> input_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t reduce_size = 0;
    uint32_t rows_per_tile = kAccBanks;
    uint32_t dense_tile_elems = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    uint32_t acc_stride = 0;
};

class KernelHeadFp32Width518Phase {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
        const SquareSumV1TilingData &tiling)
    {
        total_size = tiling.total_size;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        packet1_capacity = tiling.tile_elems;
        use_input_scratch =
            packet1_capacity == kPacket1StreamElems;
        input1_holds_scratch = false;
        compact_stride = PaddedStrideElems<float>(kInnerSize);
        acc_bytes = AlignUp32(
            kAccBanks * kInnerSize * sizeof(float) +
            kAccGuardBytes);
        aux_bytes = AlignUp32(
            (use_input_scratch
                 ? kStreamAuxBuffers
                 : kEpochAuxBuffers) * compact_stride *
            sizeof(uint32_t));
        input0_bytes = AlignUp32(
            kPacket0Elems * sizeof(float) +
            kInputGuardBytes);
        input1_bytes = AlignUp32(
            packet1_capacity * sizeof(float) +
            kInputGuardBytes);

        input_gm.SetGlobalBuffer(
            (__gm__ float *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ float *)output, tiling.output_size);
        SetSysWorkspace(workspace);
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        partial_gm.SetGlobalBuffer(
            (__gm__ float *)user_workspace,
            partial_count * partial_stride);

        pipe.InitBuffer(
            input_queue0, 1, input0_bytes);
        pipe.InitBuffer(
            input_queue1, 1, input1_bytes);
        pipe.InitBuffer(
            acc_buf,
            HEAD_FP32_ACC_PHASE_BYTES + acc_bytes + aux_bytes);
    }

    __aicore__ inline void Process()
    {
        if (partial_count <= 1 ||
            static_cast<uint32_t>(GetBlockNum()) != partial_count) {
            return;
        }
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        if (core_idx >= partial_count) {
            return;
        }
        LocalTensor<float> acc = GetHeadFp32AccView<float>(
            acc_buf, acc_bytes / sizeof(float));
        uint32_t pair_idx = 0;
        uint32_t packet0_start = 0;
        uint32_t packet0_elems = 0;
        uint32_t packet1_start = 0;
        uint32_t packet1_elems = 0;
        const uint32_t pair_count = PairCount();
        const uint32_t owner =
            GetSplitSubblockTileOwner(partial_count);
        PrimeReducePair(
            owner, pair_count, pair_idx,
            packet0_start, packet0_elems,
            packet1_start, packet1_elems);
        BuildPackedOffsets();
        if (!use_input_scratch &&
            OwnedPairCount(owner, pair_count) > kEpochPairs) {
            InitializeEpochSum();
            ProcessReducePairs<true>(
                acc, pair_count, pair_idx,
                packet0_start, packet0_elems,
                packet1_start, packet1_elems);
            CompactAndReduceBanks(acc, AuxiliaryPacked());
            MergeFinalEpoch(acc);
        } else {
            ProcessReducePairs<false>(
                acc, pair_count, pair_idx,
                packet0_start, packet0_elems,
                packet1_start, packet1_elems);
            CompactFinalBanks(acc);
        }
        WritePartial(acc, core_idx);
        SyncAll();
        MergePartials();
    }

private:
    __aicore__ inline void GetRangeByParts(
        uint32_t total, uint32_t part_idx, uint32_t part_num,
        uint32_t &start, uint32_t &end)
    {
        const uint32_t base = total / part_num;
        const uint32_t tail = total - base * part_num;
        start = part_idx * base + MinU32(part_idx, tail);
        end = start + base + (part_idx < tail ? 1U : 0U);
    }

    __aicore__ inline void CopyElements(
        LocalTensor<float> dst, uint32_t element_start,
        uint32_t elements, uint32_t packet_capacity)
    {
        if (elements == packet_capacity) {
            DataCopyParams params{
                1,
                static_cast<uint16_t>(
                    elements / kFloatAlignElems),
                0, 0};
            DataCopy(dst, input_gm[element_start], params);
            return;
        }
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(elements * sizeof(float)),
            0, 0, 0};
        DataCopyPadExtParams<float> pad{
            false, 0, 0, 0.0f};
        DataCopyPad(dst, input_gm[element_start], params, pad);
    }

    __aicore__ inline void BuildPackedOffsets()
    {
        LocalTensor<int32_t> offsets =
            GetHeadFp32AccView<int32_t>(
                acc_buf,
                compact_stride, acc_bytes);
        CreateVecIndex(
            offsets, static_cast<int32_t>(0), compact_stride);
        PipeBarrier<PIPE_V>();
        const uint32_t padding_base =
            (kInnerSize / MAX_FP32_REPEAT_MASK) * MAX_FP32_REPEAT_MASK;
        const uint32_t logical_tail = kInnerSize - padding_base;
        const uint32_t storage_tail = compact_stride - padding_base;
        uint64_t padding_mask[2] = {
            ((1ULL << storage_tail) - 1ULL) ^
                ((1ULL << logical_tail) - 1ULL),
            0ULL};
        Duplicate(
            offsets[padding_base],
            static_cast<int32_t>(kInnerSize - 1U),
            padding_mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)), compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        uint32_t input_offset = 0;
        uint32_t bank_offset = element_start % kBankSlabElems;
        while (input_offset < elements) {
            const uint32_t span = MinU32(
                elements - input_offset,
                kBankSlabElems - bank_offset);
            MulAddDst(
                acc[bank_offset], input[input_offset],
                input[input_offset], span);
            input_offset += span;
            bank_offset = 0U;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeElements(
        LocalTensor<float> acc, LocalTensor<float> input,
        uint32_t element_start, uint32_t elements)
    {
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
        AccumulateElements(acc, input, element_start, elements);
    }

    __aicore__ inline uint32_t PairCount()
    {
        return total_size == 0U
            ? 0U
            : (total_size - 1U) / PairElems() + 1U;
    }

    __aicore__ inline uint32_t PairElems()
    {
        return kPacket0Elems + packet1_capacity;
    }

    __aicore__ inline uint32_t PairElementStart(uint32_t pair_idx)
    {
        return static_cast<uint32_t>(
            static_cast<uint64_t>(pair_idx) * PairElems());
    }

    __aicore__ inline uint32_t OwnedPairCount(
        uint32_t owner, uint32_t pair_count)
    {
        return owner < pair_count
            ? (pair_count - 1U - owner) / partial_count + 1U
            : 0U;
    }

    __aicore__ inline uint32_t PacketElements(
        uint32_t element_start, uint32_t packet_capacity)
    {
        return element_start < total_size
            ? MinU32(packet_capacity, total_size - element_start)
            : 0U;
    }

    __aicore__ inline void EnqueuePacket0(
        uint32_t element_start, uint32_t elements)
    {
        input_queue0.AllocTensor<float>(input_slot0);
        CopyElements(
            input_slot0, element_start, elements, kPacket0Elems);
        input_queue0.EnQue(input_slot0);
    }

    __aicore__ inline void EnqueuePacket1(
        uint32_t element_start, uint32_t elements)
    {
        input_queue1.AllocTensor<float>(input_slot1);
        CopyElements(
            input_slot1, element_start, elements, packet1_capacity);
        input_queue1.EnQue(input_slot1);
    }

    __aicore__ inline void PrimeReducePair(
        uint32_t owner, uint32_t pair_count, uint32_t &pair_idx,
        uint32_t &packet0_start, uint32_t &packet0_elems,
        uint32_t &packet1_start, uint32_t &packet1_elems)
    {
        pair_idx = owner;
        if (pair_idx >= pair_count) {
            packet0_elems = 0U;
            packet1_elems = 0U;
            return;
        }
        packet0_start = PairElementStart(pair_idx);
        packet0_elems = PacketElements(
            packet0_start, kPacket0Elems);
        if (packet0_elems > 0U) {
            EnqueuePacket0(packet0_start, packet0_elems);
        }
        packet1_elems = 0U;
        if (packet0_elems == kPacket0Elems) {
            packet1_start = packet0_start + kPacket0Elems;
            packet1_elems = PacketElements(
                packet1_start, packet1_capacity);
            if (packet1_elems > 0U) {
                EnqueuePacket1(packet1_start, packet1_elems);
            }
        }
    }

    __aicore__ inline void ConsumePacket0(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool initialize)
    {
        input_queue0.DeQue<float>(input_slot0);
        if (initialize) {
            InitializeElements(
                acc, input_slot0, element_start, elements);
        } else {
            AccumulateElements(
                acc, input_slot0, element_start, elements);
        }
        input_queue0.FreeTensor(input_slot0);
    }

    __aicore__ inline void ConsumePacket1(
        LocalTensor<float> acc, uint32_t element_start,
        uint32_t elements, bool release)
    {
        input_queue1.DeQue<float>(input_slot1);
        AccumulateElements(
            acc, input_slot1, element_start, elements);
        if (release) {
            input_queue1.FreeTensor(input_slot1);
        }
    }

    template <bool EnableEpoch>
    __aicore__ inline void ProcessReducePairs(
        LocalTensor<float> acc, uint32_t pair_count,
        uint32_t pair_idx, uint32_t packet0_start,
        uint32_t packet0_elems, uint32_t packet1_start,
        uint32_t packet1_elems)
    {
        if (packet0_elems == 0U) {
            return;
        }
        uint32_t pairs_in_epoch = 0;
        ConsumePacket0(
            acc, packet0_start, packet0_elems, true);

        while (packet1_elems > 0U) {
            const uint64_t next_pair_idx64 =
                static_cast<uint64_t>(pair_idx) + partial_count;
            uint32_t next_pair_idx = 0U;
            uint32_t next_packet0_start = 0U;
            uint32_t next_packet0_elems = 0U;
            if (next_pair_idx64 < pair_count) {
                next_pair_idx = static_cast<uint32_t>(next_pair_idx64);
                next_packet0_start = PairElementStart(next_pair_idx);
                next_packet0_elems = PacketElements(
                    next_packet0_start, kPacket0Elems);
                EnqueuePacket0(
                    next_packet0_start, next_packet0_elems);
            }

            const bool retain_final_input =
                use_input_scratch && next_packet0_elems == 0U;
            ConsumePacket1(
                acc, packet1_start, packet1_elems,
                !retain_final_input);
            input1_holds_scratch = retain_final_input;
            if (next_packet0_elems == 0U) {
                return;
            }
            if constexpr (EnableEpoch) {
                ++pairs_in_epoch;
            }

            pair_idx = next_pair_idx;
            packet0_start = next_packet0_start;
            packet0_elems = next_packet0_elems;
            packet1_elems = 0U;
            if (packet0_elems == kPacket0Elems) {
                packet1_start = packet0_start + kPacket0Elems;
                packet1_elems = PacketElements(
                    packet1_start, packet1_capacity);
                if (packet1_elems > 0U) {
                    EnqueuePacket1(packet1_start, packet1_elems);
                }
            }
            if constexpr (EnableEpoch) {
                if (pairs_in_epoch == kEpochPairs) {
                    FlushLongEpoch(acc);
                    pairs_in_epoch = 0;
                }
            }
            ConsumePacket0(
                acc, packet0_start, packet0_elems, false);
        }
    }

    __aicore__ inline void InitializeEpochSum()
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Duplicate(epoch_sum, 0.0f, compact_stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushLongEpoch(LocalTensor<float> acc)
    {
        CompactAndReduceBanks(acc, AuxiliaryPacked());
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(epoch_sum, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
        Duplicate(acc, 0.0f, acc_bytes / sizeof(float));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFinalEpoch(LocalTensor<float> acc)
    {
        LocalTensor<float> epoch_sum =
            GetHeadFp32AccView<float>(
                acc_buf,
                compact_stride,
                acc_bytes + 2U * compact_stride * sizeof(float));
        Add(acc, epoch_sum, acc, kInnerSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> AuxiliaryPacked()
    {
        return GetHeadFp32AccView<float>(
            acc_buf,
            compact_stride,
            acc_bytes + compact_stride * sizeof(float));
    }

    __aicore__ inline void CompactFinalBanks(LocalTensor<float> acc)
    {
        if (use_input_scratch) {
            if (!input1_holds_scratch) {
                input_queue1.AllocTensor<float>(input_slot1);
            }
            CompactAndReduceBanks(acc, input_slot1);
            input_queue1.FreeTensor(input_slot1);
            input1_holds_scratch = false;
            return;
        }
        CompactAndReduceBanks(acc, AuxiliaryPacked());
    }

    __aicore__ inline void CompactAndReduceBanks(
        LocalTensor<float> acc, LocalTensor<float> packed)
    {
        LocalTensor<float> dense = GetHeadFp32AccView<float>(
            acc_buf, kBankSlabElems);
        LocalTensor<uint32_t> offsets =
            GetHeadFp32AccView<uint32_t>(
                acc_buf,
                compact_stride, acc_bytes);
        // Keep bank zero in place; the auxiliary Gather destination is disjoint.
        for (uint32_t bank = 1U; bank < kAccBanks; ++bank) {
            Gather(
                packed, dense, offsets,
                bank * kInnerSize * sizeof(float), compact_stride);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, packed, kInnerSize);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        const uint32_t tail_cols = kInnerSize % kMergeCols;
        uint64_t padding_mask[2] = {
            ((1ULL << kTailStorageCols) - 1ULL) ^
                ((1ULL << tail_cols) - 1ULL),
            0ULL};
        Duplicate(
            acc[kFullMergeTiles * kMergeCols],
            0.0f, padding_mask, 1, 1, 8);
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        const DataCopyParams columns{
            kFullMergeTiles, kMergeCols / kFloatAlignElems, 0,
            static_cast<uint16_t>(
                (partial_count - 1U) * kMergeCols / kFloatAlignElems)};
        DataCopy(partial_gm[core_idx * kMergeCols], acc, columns);
        DataCopy(
            partial_gm[kFullMergeTiles * partial_count * kMergeCols +
                       core_idx * kTailStorageCols],
            acc[kFullMergeTiles * kMergeCols], kTailStorageCols);
    }

    __aicore__ inline void MergePartials()
    {
        const uint32_t core_idx = static_cast<uint32_t>(GetBlockIdx());
        for (uint32_t merge_idx = core_idx;
             merge_idx <= kFullMergeTiles;
             merge_idx += partial_count) {
            const uint32_t row_elems = merge_idx < kFullMergeTiles
                ? kMergeCols : kTailStorageCols;
            const uint32_t row_blocks = row_elems / kFloatAlignElems;
            const uint32_t base = merge_idx * partial_count * kMergeCols;
            LocalTensor<float> partials = GetHeadFp32AccView<float>(
                acc_buf, acc_bytes / sizeof(float));
            DataCopy(partials, partial_gm[base], partial_count * row_elems);
            const event_t loaded = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loaded);
            WaitFlag<HardEvent::MTE2_V>(loaded);

            SetMaskNorm();
            SetVectorMask<float>(row_elems);
            for (uint32_t step = 1U; step < partial_count; step <<= 1U) {
                const uint32_t group = step << 1U;
                const uint32_t repeats = (partial_count + step - 1U) / group;
                const uint32_t stride = repeats > 1U ? group * row_blocks : 0U;
                const BinaryRepeatParams params{
                    1, 1, 1, static_cast<uint8_t>(stride),
                    static_cast<uint8_t>(stride), static_cast<uint8_t>(stride)};
                Add<float, false>(
                    partials, partials, partials[step * row_elems],
                    MASK_PLACEHOLDER, static_cast<uint8_t>(repeats), params);
                PipeBarrier<PIPE_V>();
            }
            ResetMask();

            const uint32_t output_start = merge_idx * kMergeCols;
            const uint32_t count = MinU32(row_elems, kInnerSize - output_start);
            const event_t ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            CopyOutContiguous(output_gm, partials, output_start, count);
            if (merge_idx + partial_count <= kFullMergeTiles) {
                const event_t reuse = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(reuse);
                WaitFlag<HardEvent::MTE3_MTE2>(reuse);
            }
        }
    }

    static constexpr uint32_t kInnerSize = 518;
    static constexpr uint32_t kMergeCols = 64;
    static constexpr uint32_t kFullMergeTiles = kInnerSize / kMergeCols;
    static constexpr uint32_t kTailStorageCols = 8;
    static constexpr uint32_t kAccBanks = 8;
    static constexpr uint32_t kBankSlabElems =
        kAccBanks * kInnerSize;
    static constexpr uint32_t kPacket0Elems = 23040;
    static constexpr uint32_t kPacket1StreamElems = 20224;
    static constexpr uint32_t kPacket1EpochElems = 19200;
    static constexpr uint32_t kStreamAuxBuffers = 1;
    static constexpr uint32_t kEpochAuxBuffers = 3;
    static constexpr uint32_t kEpochPairs = 96;
    static constexpr uint32_t kAccGuardBytes = 32;
    static constexpr uint32_t kInputGuardBytes = 32;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);

    // Independent depth-zero queues retain unequal persistent physical slots.
    TQue<TPosition::VECIN, 0> input_queue0;
    TQue<TPosition::VECIN, 0> input_queue1;
    LocalTensor<float> input_slot0;
    LocalTensor<float> input_slot1;
    TBuf<TPosition::VECOUT> acc_buf;
    GlobalTensor<float> input_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t total_size = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    uint32_t compact_stride = 0;
    uint32_t packet1_capacity = 0;
    uint32_t input0_bytes = 0;
    uint32_t input1_bytes = 0;
    uint32_t acc_bytes = 0;
    uint32_t aux_bytes = 0;
    bool use_input_scratch = false;
    bool input1_holds_scratch = false;
};

template <typename T>
class KernelMiddleReduceAligned {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        if (CanProcessFastMiddleReduce()) {
            ProcessFastMiddleReduce();
            return;
        }
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        ProcessOutputRange(start, end);
    }

private:
    __aicore__ inline void ProcessOutputRange(uint32_t start, uint32_t end)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        for (uint32_t out_offset = start; out_offset < end;) {
            uint32_t outer_idx = out_offset / inner_size;
            uint32_t inner_offset = out_offset - outer_idx * inner_size;
            uint32_t cur = MinU32(tile_elems, MinU32(inner_size - inner_offset, end - out_offset));
            if (CanProcessContiguousMiddleReduce(cur)) {
                ProcessContiguousMiddleReduce(outer_idx, out_offset, cur);
                out_offset += cur;
                continue;
            }
            LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            Duplicate(acc, 0.0f, cur);
            uint32_t r_base = 0;
            if (r_base < reduce_size) {
                uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size + inner_offset;
                LocalTensor<T> input_local = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local, input_gm, input_offset, cur);
                input_queue.EnQue(input_local);
                r_base++;
            }

            while (r_base < reduce_size) {
                uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size + inner_offset;
                LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local_next, input_gm, input_offset, cur);
                input_queue.EnQue(input_local_next);

                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquareAddToAccum(acc, input_local_curr, tmp, cur);
                input_queue.FreeTensor(input_local_curr);

                r_base++;
            }

            if (r_base > 0) {
                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquareAddToAccum(acc, input_local_curr, tmp, cur);
                input_queue.FreeTensor(input_local_curr);
            }
            StoreOutputTile(out_offset, acc, cur);
            out_offset += cur;
        }
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline bool CanProcessContiguousMiddleReduce(uint32_t cur)
    {
        return cur == inner_size;
    }

    __aicore__ inline void AccumulateRowsRepeat(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                LocalTensor<float> tmp, uint32_t start_col, uint32_t cur_cols, uint32_t rows)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
            }
        } else {
            uint32_t src_stride_blocks = (inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (cur_cols <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * inner_size + start_col], input_local[r_done * inner_size + start_col],
                              static_cast<uint64_t>(cur_cols), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
                }
            }
        }
    }

    __aicore__ inline void CopyInPaddedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset,
                                            uint32_t row_elems, uint32_t rows)
    {
        uint32_t row_bytes = row_elems * sizeof(T);
        uint32_t aligned_row_elems = AlignUp32(row_elems * sizeof(T)) / sizeof(T);
        uint32_t stride_bytes = (aligned_row_elems - row_elems) * sizeof(T);
        if (row_bytes % UB_ALIGN_BYTES == 0 && stride_bytes % UB_ALIGN_BYTES == 0 &&
            (offset * sizeof(T)) % UB_ALIGN_BYTES == 0) {
            DataCopyParams params{static_cast<uint16_t>(rows),
                                  static_cast<uint16_t>(row_bytes / UB_ALIGN_BYTES), 0,
                                  static_cast<uint16_t>(stride_bytes / UB_ALIGN_BYTES)};
            DataCopy(dst, src[offset], params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(rows), row_bytes, 0, 0, 0};
            DataCopyPadExtParams<T> pad_params{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(dst, src[offset], params, pad_params);
        }
    }

    __aicore__ inline void AccumulateRowsRepeatAlignedUB(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                         LocalTensor<float> tmp, uint32_t cur_cols, uint32_t rows, uint32_t aligned_inner_size)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * aligned_inner_size], tmp, cur_cols);
            }
        } else {
            uint32_t src_stride_blocks = (aligned_inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (cur_cols <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * aligned_inner_size], input_local[r_done * aligned_inner_size],
                              static_cast<uint64_t>(cur_cols), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * aligned_inner_size], tmp, cur_cols);
                }
            }
        }
    }

    __aicore__ inline void ProcessContiguousMiddleReduce(uint32_t outer_idx, uint32_t out_offset, uint32_t cur)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, cur);

        uint32_t aligned_inner_size = AlignUp32(inner_size * sizeof(T)) / sizeof(T);
        uint32_t rows_per_tile = tile_elems / aligned_inner_size;
        if (rows_per_tile == 0) {
            rows_per_tile = 1;
        }

        uint32_t r_base = 0;
        uint32_t rows_prev = 0;

        // Prime the pipeline
        if (r_base < reduce_size) {
            rows_prev = MinU32(rows_per_tile, reduce_size - r_base);
            uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size;
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local, input_gm, input_offset, inner_size, rows_prev);
            input_queue.EnQue(input_local);
            r_base += rows_prev;
        }

        while (r_base < reduce_size) {
            uint32_t rows_next = MinU32(rows_per_tile, reduce_size - r_base);
            uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size;
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local_next, input_gm, input_offset, inner_size, rows_next);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateRowsRepeatAlignedUB(acc, input_local_curr, tmp, cur, rows_prev, aligned_inner_size);
            input_queue.FreeTensor(input_local_curr);

            r_base += rows_next;
            rows_prev = rows_next;
        }

        if (rows_prev > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateRowsRepeatAlignedUB(acc, input_local_curr, tmp, cur, rows_prev, aligned_inner_size);
            input_queue.FreeTensor(input_local_curr);
        }

        StoreOutputTile(out_offset, acc, cur);
    }

    __aicore__ inline bool CanProcessFastMiddleReduce()
    {
        uint32_t block_num = GetBlockNum();
        if (inner_size != 256 || reduce_size != 64 || block_num < 2 || inner_size % block_num != 0) {
            return false;
        }
        uint32_t col_len = inner_size / block_num;
        return reduce_size * col_len <= tile_elems &&
               (col_len * sizeof(T)) % UB_ALIGN_BYTES == 0;
    }

    __aicore__ inline void CopyInStridedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset,
                                             uint32_t row_elems, uint32_t rows)
    {
        uint32_t row_bytes = row_elems * sizeof(T);
        uint32_t stride_bytes = (inner_size - row_elems) * sizeof(T);
        if (row_bytes % UB_ALIGN_BYTES == 0 && stride_bytes % UB_ALIGN_BYTES == 0 &&
            (offset * sizeof(T)) % UB_ALIGN_BYTES == 0) {
            DataCopyParams params{static_cast<uint16_t>(rows),
                                  static_cast<uint16_t>(row_bytes / UB_ALIGN_BYTES),
                                  static_cast<uint16_t>(stride_bytes / UB_ALIGN_BYTES), 0};
            DataCopy(dst, src[offset], params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(rows), row_bytes, stride_bytes, 0, 0};
            DataCopyPadExtParams<T> pad_params{false, 0, 0, FromFloatScalar<T>(0.0f)};
            DataCopyPad(dst, src[offset], params, pad_params);
        }
    }

    __aicore__ inline void AccumulateStridedRows(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                 LocalTensor<float> tmp, uint32_t row_elems, uint32_t rows,
                                                 uint32_t input_stride)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * input_stride], tmp, row_elems);
            }
        } else {
            uint32_t src_stride_blocks = (input_stride * sizeof(T)) / UB_ALIGN_BYTES;
            if (row_elems <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * input_stride], input_local[r_done * input_stride],
                              static_cast<uint64_t>(row_elems), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
                return;
            }
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * input_stride], tmp, row_elems);
            }
        }
    }

    __aicore__ inline void ProcessFastMiddleReduce()
    {
        uint32_t block_num = GetBlockNum();
        uint32_t col_len = inner_size / block_num;
        uint32_t col_start = GetBlockIdx() * col_len;

        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, col_len);

        uint32_t num_outer = output_size / inner_size;

        for (uint32_t out_idx = 0; out_idx < num_outer; ++out_idx) {
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            uint32_t base_offset = out_idx * reduce_size * inner_size + col_start;
            CopyInStridedRows(input_local, input_gm, base_offset, col_len, reduce_size);
            AccumulateStridedRows(acc, input_local, tmp, col_len, reduce_size, col_len);
            input_queue.FreeTensor(input_local);
            StoreOutputTile(out_idx * inner_size + col_start, acc, col_len);
            Duplicate(acc, 0.0f, col_len);
        }
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t output_size = 0;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
};

template <typename T>
class KernelBf16MiddleOuterBanked {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output,
                                const SquareSumV1TilingData &tiling)
    {
        outer_size = tiling.outer_size;
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        input_stride = PaddedStrideElems<T>(inner_size);
        acc_stride = PaddedStrideElems<float>(inner_size);

        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, tiling.output_size);
        pipe.InitBuffer(input_queue, BUFFER_NUM,
                        AlignUp32(tile_elems * sizeof(T)));
        pipe.InitBuffer(output_queue, BUFFER_NUM,
                        AlignUp32(inner_size * sizeof(T)));
        pipe.InitBuffer(calc_buf,
                        AlignUp32((kAccBanks + 1) * acc_stride *
                                  sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t outer = GetBlockIdx(); outer < outer_size;
             outer += GetBlockNum()) {
            ProcessOuter(outer);
        }
    }

private:
    __aicore__ inline void CopyRows(LocalTensor<T> dst,
                                    uint32_t outer,
                                    uint32_t row_start,
                                    uint32_t rows)
    {
        const uint32_t input_offset =
            (outer * reduce_size + row_start) * inner_size;
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(inner_size * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, input_gm[input_offset], params, pad);
    }

    __aicore__ inline void AccumulateRows(LocalTensor<float> acc,
                                          LocalTensor<T> input,
                                          LocalTensor<float> tmp,
                                          uint32_t rows)
    {
        #pragma unroll
        for (uint32_t bank = 0; bank < kAccBanks; ++bank) {
            for (uint32_t row = bank; row < rows; row += kAccBanks) {
                SquareAddToAccum(acc[bank * acc_stride],
                                 input[row * input_stride], tmp,
                                 inner_size);
            }
        }
    }

    __aicore__ inline void Store(uint32_t outer,
                                 LocalTensor<float> acc)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, inner_size);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, inner_size);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, outer * inner_size,
                          inner_size);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void ProcessOuter(uint32_t outer)
    {
        LocalTensor<float> acc = calc_buf.Get<float>();
        LocalTensor<float> tmp =
            calc_buf.GetWithOffset<float>(
                acc_stride, kAccBanks * acc_stride * sizeof(float));
        Duplicate(acc, 0.0f, kAccBanks * acc_stride);

        const uint32_t rows_per_tile = tile_elems / input_stride;
        uint32_t row_start = 0;
        uint32_t previous_rows = 0;

        if (row_start < reduce_size) {
            previous_rows =
                MinU32(rows_per_tile, reduce_size - row_start);
            LocalTensor<T> input = input_queue.AllocTensor<T>();
            CopyRows(input, outer, row_start, previous_rows);
            input_queue.EnQue(input);
            row_start += previous_rows;
        }

        while (row_start < reduce_size) {
            const uint32_t current_rows =
                MinU32(rows_per_tile, reduce_size - row_start);
            LocalTensor<T> next = input_queue.AllocTensor<T>();
            CopyRows(next, outer, row_start, current_rows);
            input_queue.EnQue(next);

            LocalTensor<T> current = input_queue.DeQue<T>();
            AccumulateRows(acc, current, tmp, previous_rows);
            input_queue.FreeTensor(current);

            row_start += current_rows;
            previous_rows = current_rows;
        }

        if (previous_rows > 0) {
            LocalTensor<T> current = input_queue.DeQue<T>();
            AccumulateRows(acc, current, tmp, previous_rows);
            input_queue.FreeTensor(current);
        }

        #pragma unroll
        for (uint32_t step = 1; step < kAccBanks; step <<= 1) {
            #pragma unroll
            for (uint32_t bank = 0; bank < kAccBanks;
                 bank += (step << 1)) {
                Add(acc[bank * acc_stride],
                    acc[bank * acc_stride],
                    acc[(bank + step) * acc_stride], inner_size);
            }
        }
        Store(outer, acc);
    }

    static constexpr uint32_t kAccBanks = 8;
    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t outer_size = 0;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
    uint32_t input_stride = 0;
    uint32_t acc_stride = 0;
};

class KernelBf16MiddlePhaseRepeat {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output,
                                GM_ADDR workspace,
                                const SquareSumV1TilingData &tiling)
    {
        outer_size = tiling.outer_size;
        reduce_size = tiling.reduce_size;
        total_size = tiling.total_size;
        partial_count = tiling.partial_count;
        partial_stride = tiling.partial_stride;
        use_raw_square =
            reduce_size >= 1024U && outer_size >= 4U;
        primary_capacity_rows = use_raw_square
            ? kRawPrimaryRows
            : kPrimaryRows;
        secondary_capacity_rows = use_raw_square
            ? kRawSecondaryRows
            : kSecondaryRows;
        convert_capacity_rows = use_raw_square
            ? kRawConvertRows
            : kConvertRows;
        compact_stride = PaddedStrideElems<float>(inner_size);
        acc_bytes = AlignUp32(
            (use_raw_square ? kRawAccBanks : kAccBanks) *
                inner_size * sizeof(float) + kGuardBytes);

        ConfigureTileLayout();

        input_gm.SetGlobalBuffer((__gm__ bfloat16_t *)input,
                                 tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer((__gm__ bfloat16_t *)output,
                                  tiling.output_size);
        if (partial_count > 0U) {
            SetSysWorkspace(workspace);
            GM_ADDR user_workspace = GetUserWorkspace(workspace);
            partial_gm.SetGlobalBuffer(
                (__gm__ float *)(user_workspace +
                    partial_count * kWorkspacePrefixBytes),
                partial_count * partial_stride);
        }
        pipe.InitBuffer(
            primary_input_queue, 1,
            AlignUp32(primary_capacity_rows * inner_size * sizeof(bfloat16_t) +
                      kInputGuardBytes));
        pipe.InitBuffer(
            secondary_input_queue, 1,
            AlignUp32(secondary_capacity_rows * inner_size * sizeof(bfloat16_t) +
                      kInputGuardBytes));
        pipe.InitBuffer(
            output_queue, use_raw_square ? 1U : BUFFER_NUM,
            use_raw_square
                ? kRawOutputRows * kRawOutputStride * sizeof(bfloat16_t)
                : AlignUp32(inner_size * sizeof(bfloat16_t)));
        pipe.InitBuffer(calc_buf,
                        acc_bytes + converted_bytes + offsets_bytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_idx =
            static_cast<uint32_t>(GetBlockIdx());
        const uint32_t core_count =
            static_cast<uint32_t>(GetBlockNum());
        const uint32_t tail_outputs =
            outer_size % core_count;
        bool build_offsets = true;
        if (tail_outputs == 0U) {
            if (use_raw_square) {
                ProcessPackedOutputs(core_idx, core_count);
                return;
            }
            for (uint32_t outer = core_idx; outer < outer_size;
                 outer += core_count) {
                ProcessOuter(outer, build_offsets);
                build_offsets = false;
            }
            return;
        }
        if (partial_count != core_count) {
            return;
        }

        const uint32_t main_outer =
            outer_size - tail_outputs;
        uint32_t tail_idx = 0U;
        uint32_t group_start = 0U;
        uint32_t group_size = 0U;
        uint32_t group_lane = 0U;
        ResolveTailGroup(
            core_idx, core_count, tail_outputs,
            tail_idx, group_start, group_size, group_lane);

        // Publish helper partials before the full waves so the leader's
        // endpoint merge does not wait on sibling AIV work.
        if (group_lane != 0U) {
            ProcessTailRemainder(
                core_idx, main_outer,
                tail_idx, group_start, group_size, group_lane,
                build_offsets);
            build_offsets = false;
        }
        for (uint32_t outer = core_idx; outer < main_outer;
             outer += core_count) {
            ProcessOuter(outer, build_offsets);
            build_offsets = false;
        }
        if (group_lane == 0U) {
            ProcessTailRemainder(
                core_idx, main_outer,
                tail_idx, group_start, group_size, group_lane,
                build_offsets);
        } else {
            CrossCoreWaitFlag(8);
        }
    }

private:
    __aicore__ inline void ConfigureTileLayout()
    {
        const uint32_t convert_rows = MinU32(
            convert_capacity_rows, reduce_size);
        convert_tile_elems = convert_rows * inner_size;
        converted_bytes = AlignUp32(
            convert_tile_elems * sizeof(float) + kGuardBytes);
        offsets_bytes = AlignUp32(
            kOffsetBanks * compact_stride * sizeof(uint32_t));
    }

    __aicore__ inline void LoadPrimary(uint32_t outer,
                                       uint32_t row_start,
                                       uint32_t rows)
    {
        primary_input_queue.AllocTensor<bfloat16_t>(primary_input_slot);
        primary_input_head =
            CopyRows(primary_input_slot, outer, row_start, rows);
        primary_input_queue.EnQue(primary_input_slot);
    }

    __aicore__ inline void LoadSecondary(uint32_t outer,
                                         uint32_t row_start,
                                         uint32_t rows)
    {
        secondary_input_queue.AllocTensor<bfloat16_t>(secondary_input_slot);
        secondary_input_head =
            CopyRows(secondary_input_slot, outer, row_start, rows);
        secondary_input_queue.EnQue(secondary_input_slot);
    }

    template <bool RawSquare>
    __aicore__ inline void ConsumePrimary(LocalTensor<float> acc,
                                          uint32_t rows, uint32_t outer,
                                          uint32_t refill_start,
                                          uint32_t refill_rows,
                                          bool &initialize)
    {
        primary_input_queue.DeQue<bfloat16_t>(primary_input_slot);
        AccumulateRows<RawSquare>(
            acc, primary_input_slot[primary_input_head], rows,
            false, outer, refill_start, refill_rows,
            initialize);
    }

    template <bool RawSquare>
    __aicore__ inline void ConsumeSecondary(LocalTensor<float> acc,
                                            uint32_t rows, uint32_t outer,
                                            uint32_t refill_start,
                                            uint32_t refill_rows,
                                            bool &initialize)
    {
        secondary_input_queue.DeQue<bfloat16_t>(secondary_input_slot);
        AccumulateRows<RawSquare>(
            acc, secondary_input_slot[secondary_input_head], rows,
            true, outer, refill_start, refill_rows,
            initialize);
    }

    __aicore__ inline void BuildPackedOffsets()
    {
        LocalTensor<int32_t> offsets =
            calc_buf.GetWithOffset<int32_t>(
                kOffsetBanks * compact_stride,
                acc_bytes + converted_bytes);
        CreateVecIndex(
            offsets, static_cast<int32_t>(0), compact_stride);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)),
            compact_stride);
        PipeBarrier<PIPE_V>();

        for (uint32_t bank = 1U; bank < kOffsetBanks; ++bank) {
            Adds(
                offsets[bank * compact_stride], offsets,
                static_cast<int32_t>(
                    bank * inner_size * sizeof(float)),
                compact_stride);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t CopyRows(LocalTensor<bfloat16_t> dst,
                                        uint32_t outer,
                                        uint32_t row_start,
                                        uint32_t rows)
    {
        const uint32_t input_offset =
            (outer * reduce_size + row_start) * inner_size;
        const uint32_t logical_elems = rows * inner_size;
        if (input_offset % kBf16AlignElems == 0U &&
            logical_elems % kBf16AlignElems == 0U) {
            DataCopy(dst, input_gm[input_offset], logical_elems);
            return 0U;
        }
        const uint32_t source_prefix =
            input_offset % kGmAlignElems;
        const uint32_t source_offset =
            input_offset - source_prefix;
        const uint32_t left_padding =
            (kBf16AlignElems -
             source_prefix % kBf16AlignElems) %
            kBf16AlignElems;
        const uint32_t logical_head =
            source_prefix + left_padding;
        const uint32_t fetch_elems =
            source_prefix + logical_elems;
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(
                fetch_elems * sizeof(bfloat16_t)),
            0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{
            true, static_cast<uint8_t>(left_padding), 0,
            ToBfloat16(0.0f)};
        DataCopyPad(dst, input_gm[source_offset], params, pad);
        return logical_head;
    }

    __aicore__ inline void RoundBf16Squares(
        LocalTensor<float> values,
        LocalTensor<bfloat16_t> rounded,
        uint32_t count)
    {
        Cast(rounded, values, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(values, rounded, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    template <bool RawSquare>
    __aicore__ inline void AccumulateRows(LocalTensor<float> acc,
                                          LocalTensor<bfloat16_t> input,
                                          uint32_t rows,
                                          bool input_is_secondary,
                                          uint32_t outer,
                                          uint32_t refill_start,
                                          uint32_t refill_rows,
                                          bool &initialize)
    {
        LocalTensor<float> converted =
            calc_buf.GetWithOffset<float>(
                convert_tile_elems, acc_bytes);
        constexpr uint32_t convert_rows = RawSquare
            ? kRawConvertRows
            : kConvertRows;

        for (uint32_t row = 0; row < rows;) {
            const uint32_t current_rows = MinU32(
                convert_rows, rows - row);
            const uint32_t input_offset = row * inner_size;
            const uint32_t current_elems =
                current_rows * inner_size;
            Cast(
                converted, input[input_offset],
                RoundMode::CAST_NONE, current_elems);
            PipeBarrier<PIPE_V>();
            if constexpr (!RawSquare) {
                // Preserve the BF16 square result before any FP32 accumulation.
                Mul(converted, converted, converted, current_elems);
                PipeBarrier<PIPE_V>();
                RoundBf16Squares(
                    converted, input[input_offset], current_elems);
            }

            if (row + current_rows == rows) {
                ReleaseAndRefillInput(
                    input_is_secondary, outer,
                    refill_start, refill_rows);
            }

            if constexpr (RawSquare) {
                // A converted group may span more rows than the accumulator.
                for (uint32_t bank = 0U; bank < current_rows;
                     bank += kRawAccBanks) {
                    const uint32_t active_banks = MinU32(
                        kRawAccBanks, current_rows - bank);
                    LocalTensor<float> cohort = converted[bank * inner_size];
                    const uint32_t cohort_elems = active_banks * inner_size;
                    if (initialize) {
                        Mul(acc, cohort, cohort, cohort_elems);
                        initialize = false;
                    } else {
                        MulAddDst(acc, cohort, cohort, cohort_elems);
                    }
                    PipeBarrier<PIPE_V>();
                }
            } else {
                for (uint32_t bank = 0; bank < current_rows;
                     bank += kAccBanks) {
                    const uint32_t active_banks = MinU32(
                        kAccBanks, current_rows - bank);
                    // Every full chunk contains one or two aligned bank cohorts.
                    Add(
                        acc, acc, converted[bank * inner_size],
                        active_banks * inner_size);
                    PipeBarrier<PIPE_V>();
                }
            }
            row += current_rows;
        }
    }

    __aicore__ inline void ReleaseAndRefillInput(
        bool input_is_secondary, uint32_t outer,
        uint32_t refill_start, uint32_t refill_rows)
    {
        if (input_is_secondary) {
            secondary_input_queue.FreeTensor(secondary_input_slot);
            if (refill_rows > 0U) {
                LoadSecondary(outer, refill_start, refill_rows);
            }
            return;
        }
        primary_input_queue.FreeTensor(primary_input_slot);
        if (refill_rows > 0U) {
            LoadPrimary(outer, refill_start, refill_rows);
        }
    }

    __aicore__ inline void ProcessReduceRows(LocalTensor<float> acc,
                                             uint32_t outer,
                                             bool build_offsets)
    {
        if (use_raw_square) {
            ProcessReduceRange<true>(acc, outer, 0U, reduce_size,
                                     build_offsets, true);
        } else {
            ProcessReduceRange<false>(acc, outer, 0U, reduce_size,
                                      build_offsets, true);
        }
    }

    template <bool RawSquare>
    __aicore__ inline uint32_t FirstPrimaryRows(
        uint32_t outer, uint32_t range_start, uint32_t range_end)
    {
        uint32_t first_capacity = RawSquare ? kRawPrimaryRows : kPrimaryRows;
        if constexpr (RawSquare) {
            // A 510-byte row reaches the next 32-byte GM phase in <=15 rows.
            const uint32_t phase_rows =
                (outer + kBf16AlignElems -
                 range_start % kBf16AlignElems) %
                kBf16AlignElems;
            if (phase_rows != 0U) {
                first_capacity -= kBf16AlignElems - phase_rows;
            }
        }
        return MinU32(first_capacity, range_end - range_start);
    }

    template <bool RawSquare>
    __aicore__ __attribute__((noinline)) void ProcessReduceRange(
        LocalTensor<float> acc, uint32_t outer,
        uint32_t range_start, uint32_t range_end,
        bool build_offsets, bool prime_before_init = false,
        bool inputs_primed = false, uint32_t next_outer = ~0U)
    {
        constexpr uint32_t primary_capacity = RawSquare
            ? kRawPrimaryRows
            : kPrimaryRows;
        constexpr uint32_t secondary_capacity = RawSquare
            ? kRawSecondaryRows
            : kSecondaryRows;
        const bool has_first = range_start < range_end;
        uint32_t primary_rows = 0U;
        if (has_first) {
            primary_rows = FirstPrimaryRows<RawSquare>(outer, range_start, range_end);
            if (prime_before_init && !inputs_primed) {
                LoadPrimary(outer, range_start, primary_rows);
            }
        }

        if constexpr (!RawSquare) {
            Duplicate(
                acc, 0.0f, acc_bytes / sizeof(float));
        }
        if (build_offsets) {
            BuildPackedOffsets();
        }
        if (!has_first) {
            return;
        }
        if (!prime_before_init && !inputs_primed) {
            LoadPrimary(outer, range_start, primary_rows);
        }
        uint32_t next_row_start = range_start + primary_rows;
        uint32_t secondary_rows = MinU32(
            secondary_capacity, range_end - next_row_start);
        if (secondary_rows > 0U) {
            if (!inputs_primed) {
                LoadSecondary(outer, next_row_start, secondary_rows);
            }
            next_row_start += secondary_rows;
        }

        const bool prefetch_next = RawSquare && next_outer < outer_size;
        const uint32_t next_primary_rows = prefetch_next
            ? FirstPrimaryRows<RawSquare>(next_outer, 0U, reduce_size)
            : 0U;
        bool current_is_secondary = false;
        bool initialize = true;
        while (primary_rows > 0U || secondary_rows > 0U) {
            const uint32_t refill_rows = MinU32(
                current_is_secondary
                    ? secondary_capacity
                    : primary_capacity,
                range_end - next_row_start);
            uint32_t refill_outer = outer;
            uint32_t refill_start = next_row_start;
            uint32_t load_rows = refill_rows;
            if (prefetch_next && refill_rows == 0U) {
                // Reuse each exhausted slot after its final cast. Keep the
                // queued next-outer rows out of this reduction's row counts.
                refill_outer = next_outer;
                refill_start = current_is_secondary ? next_primary_rows : 0U;
                load_rows = current_is_secondary
                    ? MinU32(secondary_capacity, reduce_size - next_primary_rows)
                    : next_primary_rows;
            }
            if (current_is_secondary) {
                ConsumeSecondary<RawSquare>(
                    acc, secondary_rows, refill_outer, refill_start,
                    load_rows, initialize);
                secondary_rows = refill_rows;
            } else {
                ConsumePrimary<RawSquare>(
                    acc, primary_rows, refill_outer, refill_start,
                    load_rows, initialize);
                primary_rows = refill_rows;
            }
            next_row_start += refill_rows;
            current_is_secondary = !current_is_secondary;
        }
    }

    __aicore__ inline void ResolveTailGroup(
        uint32_t core_idx, uint32_t core_count,
        uint32_t tail_outputs, uint32_t &tail_idx,
        uint32_t &group_start, uint32_t &group_size,
        uint32_t &group_lane)
    {
        const uint32_t small_group =
            core_count / tail_outputs;
        const uint32_t large_groups =
            core_count - small_group * tail_outputs;
        const uint32_t large_size = small_group + 1U;
        const uint32_t large_core_count =
            large_groups * large_size;
        if (core_idx < large_core_count) {
            tail_idx = core_idx / large_size;
            group_start = tail_idx * large_size;
            group_size = large_size;
            group_lane = core_idx - group_start;
            return;
        }

        const uint32_t remaining_core =
            core_idx - large_core_count;
        const uint32_t small_idx =
            remaining_core / small_group;
        tail_idx = large_groups + small_idx;
        group_start =
            large_core_count + small_idx * small_group;
        group_size = small_group;
        group_lane = core_idx - group_start;
    }

    __aicore__ inline void WritePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        event_t ready_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready_event);
        WaitFlag<HardEvent::V_MTE3>(ready_event);
        DataCopy(
            partial_gm[core_idx * partial_stride],
            acc, partial_stride);
    }

    __aicore__ inline void MergePartial(
        LocalTensor<float> acc, uint32_t core_idx)
    {
        LocalTensor<float> local;
        primary_input_queue.AllocTensor<float>(local);
        DataCopy(
            local,
            partial_gm[core_idx * partial_stride],
            partial_stride);
        primary_input_queue.EnQue(local);
        primary_input_queue.DeQue<float>(local);
        Add(acc, acc, local, inner_size);
        primary_input_queue.FreeTensor(local);
    }

    __aicore__ inline void ProcessTailRemainder(
        uint32_t core_idx, uint32_t main_outer,
        uint32_t tail_idx,
        uint32_t group_start, uint32_t group_size,
        uint32_t group_lane,
        bool build_offsets)
    {
        LocalTensor<float> acc = calc_buf.Get<float>(
            (acc_bytes + converted_bytes + offsets_bytes) / sizeof(float));
        // Partition aligned cohorts independently of the two DMA slot sizes.
        const uint32_t cohorts =
            (reduce_size + kHelperRows - 1U) / kHelperRows;
        const uint32_t cohorts_per_lane = cohorts / group_size;
        const uint32_t extra_lanes = cohorts % group_size;
        const uint32_t first_cohort =
            group_lane * cohorts_per_lane +
            MinU32(group_lane, extra_lanes);
        const uint32_t lane_cohorts =
            cohorts_per_lane + (group_lane < extra_lanes ? 1U : 0U);
        const uint32_t range_start =
            MinU32(first_cohort * kHelperRows, reduce_size);
        const uint32_t range_end =
            MinU32((first_cohort + lane_cohorts) * kHelperRows, reduce_size);
        if (use_raw_square) {
            ProcessReduceRange<true>(
                acc, main_outer + tail_idx,
                range_start, range_end, build_offsets);
        } else {
            ProcessReduceRange<false>(
                acc, main_outer + tail_idx,
                range_start, range_end, build_offsets);
        }
        LocalTensor<float> reduced = ReduceBanks(acc);

        if (group_lane != 0U) {
            WritePartial(reduced, core_idx);
            const event_t partial_reuse_event = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(
                    HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(partial_reuse_event);
            WaitFlag<HardEvent::MTE3_V>(partial_reuse_event);
            CrossCoreSetFlag<0x0, PIPE_MTE3>(8);
            return;
        }

        CrossCoreSetFlag<0x0, PIPE_MTE3>(8);
        CrossCoreWaitFlag(8);
        for (uint32_t peer = group_start + 1U;
             peer < group_start + group_size; ++peer) {
            MergePartial(reduced, peer);
        }
        Store(main_outer + tail_idx, reduced);
    }

    __aicore__ inline LocalTensor<float> ReduceBanks(
        LocalTensor<float> acc)
    {
        LocalTensor<float> packed =
            calc_buf.GetWithOffset<float>(
                convert_tile_elems, acc_bytes);
        LocalTensor<uint32_t> offsets =
            calc_buf.GetWithOffset<uint32_t>(
                kOffsetBanks * compact_stride,
                acc_bytes + converted_bytes);

        Add(
            acc, acc, acc[8U * inner_size],
            8U * inner_size);
        PipeBarrier<PIPE_V>();
        // Pad the eight dense survivors before the first unaligned bank split.
        // Gather lane 255 stays isolated in padding and is excluded at output.
        for (uint32_t bank = 0U; bank < 8U;
             bank += kOffsetBanks) {
            Gather(
                packed[bank * compact_stride], acc, offsets,
                bank * inner_size * sizeof(float),
                kOffsetBanks * compact_stride);
        }
        PipeBarrier<PIPE_V>();

        Add(packed, packed, packed[4U * compact_stride], 4U * compact_stride);
        PipeBarrier<PIPE_V>();
        Add(packed, packed, packed[2U * compact_stride], 2U * compact_stride);
        PipeBarrier<PIPE_V>();
        Add(acc, packed, packed[compact_stride], inner_size);
        PipeBarrier<PIPE_V>();

        return acc;
    }

    __aicore__ inline void Store(uint32_t outer,
                                 LocalTensor<float> acc)
    {
        LocalTensor<bfloat16_t> out =
            output_queue.AllocTensor<bfloat16_t>();
        Cast(out, acc, RoundMode::CAST_RINT, inner_size);
        output_queue.EnQue(out);
        out = output_queue.DeQue<bfloat16_t>();
        CopyOutContiguous(output_gm, out, outer * inner_size,
                          inner_size);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void ProcessOuter(uint32_t outer,
                                        bool build_offsets)
    {
        LocalTensor<float> acc = calc_buf.Get<float>(
            (acc_bytes + converted_bytes + offsets_bytes) / sizeof(float));
        ProcessReduceRows(acc, outer, build_offsets);
        LocalTensor<float> reduced = ReduceBanks(acc);
        Store(outer, reduced);
    }

    __aicore__ inline void ProcessPackedOutputs(
        uint32_t core_idx, uint32_t core_count)
    {
        LocalTensor<float> acc = calc_buf.Get<float>(
            (acc_bytes + converted_bytes + offsets_bytes) / sizeof(float));
        bool build_offsets = true;
        bool inputs_primed = false;
        for (uint32_t first_outer = core_idx; first_outer < outer_size;
             first_outer += kRawOutputRows * core_count) {
            const uint32_t rows = MinU32(
                kRawOutputRows, (outer_size - 1U - first_outer) / core_count + 1U);
            LocalTensor<bfloat16_t> out =
                output_queue.AllocTensor<bfloat16_t>();
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t outer = first_outer + row * core_count;
                const uint32_t next_outer = outer + core_count;
                ProcessReduceRange<true>(
                    acc, outer, 0U, reduce_size, build_offsets, true,
                    inputs_primed, next_outer);
                inputs_primed = next_outer < outer_size;
                LocalTensor<float> reduced = ReduceBanks(acc);
                Cast(
                    out[row * kRawOutputStride], reduced,
                    RoundMode::CAST_RINT, inner_size);
                PipeBarrier<PIPE_V>();
                build_offsets = false;
            }
            output_queue.EnQue(out);
            out = output_queue.DeQue<bfloat16_t>();
            const DataCopyExtParams params{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(inner_size * sizeof(bfloat16_t)),
                0U,
                static_cast<uint32_t>(
                    (core_count - 1U) * inner_size * sizeof(bfloat16_t)),
                0U};
            DataCopyPad(output_gm[first_outer * inner_size], out, params);
            // The next allocation waits for this slot's MTE3-to-V reuse event.
            output_queue.FreeTensor(out);
        }
    }

    static constexpr uint32_t kAccBanks = 16;
    static constexpr uint32_t kHelperRows = 16;
    static constexpr uint32_t kPrimaryRows = 96;
    static constexpr uint32_t kSecondaryRows = 64;
    static constexpr uint32_t kConvertRows = 64;
    static constexpr uint32_t kRawAccBanks = 16;
    static constexpr uint32_t kRawPrimaryRows = 160;
    static constexpr uint32_t kRawSecondaryRows = 96;
    static constexpr uint32_t kRawConvertRows = 32;
    static constexpr uint32_t kRawOutputRows = 4;
    static constexpr uint32_t kRawOutputStride = 256;
    static constexpr uint32_t kOffsetBanks = 4;
    static constexpr uint32_t kInputGuardBytes = 544;
    static constexpr uint32_t kGuardBytes = 32;
    static constexpr uint32_t kWorkspacePrefixBytes = 32;
    static constexpr uint32_t kGmAlignElems =
        512U / sizeof(bfloat16_t);
    static constexpr uint32_t kBf16AlignElems =
        UB_ALIGN_BYTES / sizeof(bfloat16_t);

    TQue<TPosition::VECIN, 0> primary_input_queue;
    TQue<TPosition::VECIN, 0> secondary_input_queue;
    LocalTensor<bfloat16_t> primary_input_slot;
    LocalTensor<bfloat16_t> secondary_input_slot;
    uint32_t primary_input_head = 0U;
    uint32_t secondary_input_head = 0U;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<bfloat16_t> input_gm;
    GlobalTensor<bfloat16_t> output_gm;
    GlobalTensor<float> partial_gm;
    uint32_t outer_size = 0;
    uint32_t reduce_size = 0;
    static constexpr uint32_t inner_size = 255;
    uint32_t total_size = 0;
    uint32_t primary_capacity_rows = 0;
    uint32_t secondary_capacity_rows = 0;
    uint32_t convert_capacity_rows = 0;
    uint32_t convert_tile_elems = 0;
    uint32_t compact_stride = 0;
    uint32_t acc_bytes = 0;
    uint32_t converted_bytes = 0;
    uint32_t offsets_bytes = 0;
    uint32_t partial_count = 0;
    uint32_t partial_stride = 0;
    bool use_raw_square = false;
};

template <typename T>
class KernelMiddleReduceFallback {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        for (uint32_t out_offset = start; out_offset < end;) {
            uint32_t outer_idx = out_offset / inner_size;
            uint32_t inner_offset = out_offset - outer_idx * inner_size;
            uint32_t cur = MinU32(tile_elems, MinU32(inner_size - inner_offset, end - out_offset));
            if (CanProcessContiguousMiddleReduce(cur)) {
                ProcessContiguousMiddleReduce(outer_idx, out_offset, cur);
                out_offset += cur;
                continue;
            }
            LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            Duplicate(acc, 0.0f, cur);
            uint32_t r_base = 0;
            if (r_base < reduce_size) {
                uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size + inner_offset;
                LocalTensor<T> input_local = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local, input_gm, input_offset, cur);
                input_queue.EnQue(input_local);
                r_base++;
            }

            while (r_base < reduce_size) {
                uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size + inner_offset;
                LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local_next, input_gm, input_offset, cur);
                input_queue.EnQue(input_local_next);

                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquareAddToAccum(acc, input_local_curr, tmp, cur);
                input_queue.FreeTensor(input_local_curr);

                r_base++;
            }

            if (r_base > 0) {
                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquareAddToAccum(acc, input_local_curr, tmp, cur);
                input_queue.FreeTensor(input_local_curr);
            }
            StoreOutputTile(out_offset, acc, cur);
            out_offset += cur;
        }
    }

private:
    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline bool CanProcessContiguousMiddleReduce(uint32_t cur)
    {
        return cur == inner_size;
    }

    __aicore__ inline void AccumulateRowsRepeat(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                LocalTensor<float> tmp, uint32_t start_col, uint32_t cur_cols, uint32_t rows)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
            }
        } else {
            uint32_t src_stride_blocks = (inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (cur_cols <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * inner_size + start_col], input_local[r_done * inner_size + start_col],
                              static_cast<uint64_t>(cur_cols), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * inner_size + start_col], tmp, cur_cols);
                }
            }
        }
    }

    __aicore__ inline void CopyInPaddedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset,
                                            uint32_t row_elems, uint32_t rows)
    {
        uint32_t row_bytes = row_elems * sizeof(T);
        uint32_t aligned_row_elems = AlignUp32(row_elems * sizeof(T)) / sizeof(T);
        uint32_t stride_bytes = (aligned_row_elems - row_elems) * sizeof(T);
        if (row_bytes % UB_ALIGN_BYTES == 0 && stride_bytes % UB_ALIGN_BYTES == 0 &&
            (offset * sizeof(T)) % UB_ALIGN_BYTES == 0) {
            DataCopyParams params{static_cast<uint16_t>(rows),
                                  static_cast<uint16_t>(row_bytes / UB_ALIGN_BYTES), 0,
                                  static_cast<uint16_t>(stride_bytes / UB_ALIGN_BYTES)};
            DataCopy(dst, src[offset], params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(rows), row_bytes, 0, 0, 0};
            DataCopyPadExtParams<T> pad_params{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(dst, src[offset], params, pad_params);
        }
    }

    __aicore__ inline void AccumulateRowsRepeatAlignedUB(LocalTensor<float> acc, LocalTensor<T> input_local,
                                                         LocalTensor<float> tmp, uint32_t cur_cols, uint32_t rows, uint32_t aligned_inner_size)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                SquareAddToAccum(acc, input_local[r * aligned_inner_size], tmp, cur_cols);
            }
        } else {
            uint32_t src_stride_blocks = (aligned_inner_size * sizeof(T)) / UB_ALIGN_BYTES;
            if (cur_cols <= MAX_FP32_REPEAT_MASK && src_stride_blocks <= 255) {
                uint32_t r_done = 0;
                while (r_done < rows) {
                    uint32_t r_cur = MinU32(255, rows - r_done);
                    BinaryRepeatParams repeat_params(0, 1, 1, 0, static_cast<uint8_t>(src_stride_blocks),
                                                     static_cast<uint8_t>(src_stride_blocks));
                    MulAddDst(acc, input_local[r_done * aligned_inner_size], input_local[r_done * aligned_inner_size],
                              static_cast<uint64_t>(cur_cols), static_cast<uint8_t>(r_cur), repeat_params);
                    r_done += r_cur;
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    SquareAddToAccum(acc, input_local[r * aligned_inner_size], tmp, cur_cols);
                }
            }
        }
    }

    __aicore__ inline void ProcessContiguousMiddleReduce(uint32_t outer_idx, uint32_t out_offset, uint32_t cur)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, cur);

        uint32_t aligned_inner_size = AlignUp32(inner_size * sizeof(T)) / sizeof(T);
        uint32_t rows_per_tile = tile_elems / aligned_inner_size;
        if (rows_per_tile == 0) {
            rows_per_tile = 1;
        }

        uint32_t r_base = 0;
        uint32_t rows_prev = 0;

        // Prime the pipeline
        if (r_base < reduce_size) {
            rows_prev = MinU32(rows_per_tile, reduce_size - r_base);
            uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size;
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local, input_gm, input_offset, inner_size, rows_prev);
            input_queue.EnQue(input_local);
            r_base += rows_prev;
        }

        while (r_base < reduce_size) {
            uint32_t rows_next = MinU32(rows_per_tile, reduce_size - r_base);
            uint32_t input_offset = (outer_idx * reduce_size + r_base) * inner_size;
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local_next, input_gm, input_offset, inner_size, rows_next);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateRowsRepeatAlignedUB(acc, input_local_curr, tmp, cur, rows_prev, aligned_inner_size);
            input_queue.FreeTensor(input_local_curr);

            r_base += rows_next;
            rows_prev = rows_next;
        }

        if (rows_prev > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            AccumulateRowsRepeatAlignedUB(acc, input_local_curr, tmp, cur, rows_prev, aligned_inner_size);
            input_queue.FreeTensor(input_local_curr);
        }

        StoreOutputTile(out_offset, acc, cur);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t output_size = 0;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
};

template <typename T>
class KernelMiddleReduceOuterBlock {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        outer_size = tiling.outer_size;
        reduce_size = tiling.reduce_size;
        inner_size = tiling.inner_size;
        tile_elems = tiling.tile_elems;
        block_elems = reduce_size * inner_size;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, tiling.output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(outer_size, start, end);
        for (uint32_t outer_idx = start; outer_idx < end; ++outer_idx) {
            ProcessOuterBlock(outer_idx);
        }
    }

private:
    __aicore__ inline void ProcessOuterBlock(uint32_t outer_idx)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<T> input_local = input_queue.AllocTensor<T>();
        CopyInContiguous(input_local, input_gm, outer_idx * block_elems, block_elems);
        input_queue.EnQue(input_local);
        input_local = input_queue.DeQue<T>();

        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        Duplicate(acc, 0.0f, inner_size);
        for (uint32_t r = 0; r < reduce_size; ++r) {
            SquareAddToAccum(acc, input_local[r * inner_size], tmp, inner_size);
        }
        input_queue.FreeTensor(input_local);
        StoreOutputTile(outer_idx * inner_size, acc, inner_size);
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t outer_size = 0;
    uint32_t reduce_size = 0;
    uint32_t inner_size = 0;
    uint32_t tile_elems = 0;
    uint32_t block_elems = 0;
};

template <typename T>
class KernelSeparatedInnerVector {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        rank = tiling.rank;
        axis_mask = tiling.axis_mask;
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        inner_size = tiling.inner_size == 0 ? 1 : tiling.inner_size;
        tile_elems = tiling.tile_elems;
        for (uint32_t i = 0; i < 8; ++i) {
            dims[i] = tiling.dims[i];
            strides[i] = tiling.strides[i];
        }
        vector_start_axis = rank;
        for (uint32_t i = 0; i < rank; ++i) {
            if (IsReducedAxis(axis_mask, i)) {
                vector_start_axis = i + 1;
            }
        }
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 2 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        for (uint32_t out_offset = start; out_offset < end;) {
            uint32_t group_idx = out_offset / inner_size;
            uint32_t inner_offset = out_offset - group_idx * inner_size;
            uint32_t cur = MinU32(tile_elems, MinU32(inner_size - inner_offset, end - out_offset));
            LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
            LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
            Duplicate(acc, 0.0f, cur);
            uint32_t base = InputBaseFromOutputGroup(group_idx) + inner_offset;
            for (uint32_t r = 0; r < reduce_size; ++r) {
                LocalTensor<T> input_local = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local, input_gm, base + ReduceOffset(r), cur);
                input_queue.EnQue(input_local);
                input_local = input_queue.DeQue<T>();
                SquareAddToAccum(acc, input_local, tmp, cur);
                input_queue.FreeTensor(input_local);
            }
            StoreOutputTile(out_offset, acc, cur);
            out_offset += cur;
        }
    }

private:
    __aicore__ inline uint32_t InputBaseFromOutputGroup(uint32_t group_idx)
    {
        uint32_t base = 0;
        uint32_t rem = group_idx;
        for (int32_t d = static_cast<int32_t>(vector_start_axis) - 1; d >= 0; --d) {
            if (!IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                base += coord * strides[d];
            }
        }
        return base;
    }

    __aicore__ inline uint32_t ReduceOffset(uint32_t reduce_idx)
    {
        uint32_t offset = 0;
        uint32_t rem = reduce_idx;
        for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
            if (IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                offset += coord * strides[d];
            }
        }
        return offset;
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t rank = 0;
    uint32_t axis_mask = 0;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t inner_size = 1;
    uint32_t tile_elems = 0;
    uint32_t vector_start_axis = 0;
    uint32_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t strides[8] = {1, 1, 1, 1, 1, 1, 1, 1};
};

template <typename T>
class KernelSeparatedTailVector {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        rank = tiling.rank;
        axis_mask = tiling.axis_mask;
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        tile_elems = tiling.tile_elems;
        for (uint32_t i = 0; i < 8; ++i) {
            dims[i] = tiling.dims[i];
            strides[i] = tiling.strides[i];
        }
        tail_start_axis = rank == 0 ? 0 : rank - 1;
        while (tail_start_axis > 0 && IsReducedAxis(axis_mask, tail_start_axis - 1)) {
            --tail_start_axis;
        }
        tail_reduce_size = 1;
        for (uint32_t i = tail_start_axis; i < rank; ++i) {
            tail_reduce_size *= dims[i];
        }
        outer_reduce_size = tail_reduce_size == 0 ? 1 : reduce_size / tail_reduce_size;
        inner_keep_axis = tail_start_axis == 0 ? 0 : tail_start_axis - 1;
        inner_keep_size =
            (tail_start_axis > 0 && !IsReducedAxis(axis_mask, inner_keep_axis)) ? dims[inner_keep_axis] : 1;
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 3 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        if (CanProcessOutputBlock()) {
            ProcessBlockRange(start, end);
            return;
        }
        ProcessScalarRange(start, end);
    }

private:
    __aicore__ inline bool CanProcessOutputBlock()
    {
        return tail_start_axis > 0 && !IsReducedAxis(axis_mask, inner_keep_axis) && tail_reduce_size > 0 &&
               tail_reduce_size <= tile_elems && inner_keep_size > 0 && MaxPaddedRows() > 0;
    }

    __aicore__ inline uint32_t MaxPaddedRows()
    {
        if (PaddedStrideBlocks<T>(tail_reduce_size) > 255 || PaddedStrideBlocks<float>(tail_reduce_size) > 255) {
            return 0;
        }
        uint32_t input_stride = PaddedStrideElems<T>(tail_reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);
        if (input_stride == 0 || float_stride == 0) {
            return 0;
        }
        uint32_t rows = MinU32(tile_elems / input_stride, tile_elems / float_stride);
        return MinU32(rows, MAX_REPEAT_ROWS);
    }

    __aicore__ inline void ProcessBlockRange(uint32_t start, uint32_t end)
    {
        uint32_t max_rows = MaxPaddedRows();
        for (uint32_t out_offset = start; out_offset < end;) {
            uint32_t group_idx = out_offset / inner_keep_size;
            uint32_t inner_offset = out_offset - group_idx * inner_keep_size;
            uint32_t rows = MinU32(
                max_rows,
                MinU32(inner_keep_size - inner_offset,
                       end - out_offset));
            ProcessOutputBlock(group_idx, inner_offset, rows, out_offset);
            out_offset += rows;
        }
    }

    __aicore__ inline void ProcessScalarRange(uint32_t start, uint32_t end)
    {
        for (uint32_t out_idx = start; out_idx < end; ++out_idx) {
            uint32_t base = InputBaseFromOutput(out_idx);
            float acc = 0.0f;
            for (uint32_t r = 0; r < outer_reduce_size; ++r) {
                acc += ReduceContiguousSegment(base + OuterReduceOffset(r), tail_reduce_size);
            }
            WriteScalarOutput(out_idx, acc);
        }
    }

    __aicore__ inline void ProcessOutputBlock(uint32_t group_idx, uint32_t inner_offset, uint32_t rows,
                                              uint32_t output_offset)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> square = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> tmp_work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        Duplicate(acc, 0.0f, rows);

        uint32_t base = InputBaseFromOutputGroup(group_idx) + inner_offset * strides[inner_keep_axis];
        uint32_t input_stride = PaddedStrideElems<T>(tail_reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);

        uint32_t r = 0;
        if (r < outer_reduce_size) {
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local, input_gm, base + OuterReduceOffset(r), tail_reduce_size, rows);
            input_queue.EnQue(input_local);
            r++;
        }

        while (r < outer_reduce_size) {
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            CopyInPaddedRows(input_local_next, input_gm, base + OuterReduceOffset(r), tail_reduce_size, rows);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            SquarePaddedRowsToFloat(square, input_local_curr, tmp_work, tail_reduce_size, rows, input_stride,
                                    float_stride);
            AccumulateTailRows(acc, square, tmp_work, sum, tail_reduce_size, rows, float_stride);
            input_queue.FreeTensor(input_local_curr);
            r++;
        }

        if (outer_reduce_size > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            SquarePaddedRowsToFloat(square, input_local_curr, tmp_work, tail_reduce_size, rows, input_stride,
                                    float_stride);
            AccumulateTailRows(acc, square, tmp_work, sum, tail_reduce_size, rows, float_stride);
            input_queue.FreeTensor(input_local_curr);
        }

        StoreOutputTile(output_offset, acc, rows);
    }

    __aicore__ inline void AccumulateTailRows(LocalTensor<float> acc, LocalTensor<float> square,
                                              LocalTensor<float> row_sums, LocalTensor<float> sum,
                                              uint32_t tail_elems, uint32_t rows, uint32_t float_stride)
    {
        uint32_t float_stride_blocks = (float_stride * sizeof(float)) / UB_ALIGN_BYTES;
        if (tail_elems <= MAX_FP32_REPEAT_MASK && rows <= MAX_REPEAT_ROWS && float_stride_blocks <= 255) {
            WholeReduceSum<float>(row_sums, square, static_cast<int32_t>(tail_elems),
                                  static_cast<int32_t>(rows), 1, 1,
                                  static_cast<int32_t>(float_stride_blocks));
            Add(acc, acc, row_sums, rows);
            return;
        }
        uint32_t row_sums2_offset = AlignUp32(rows * sizeof(float)) / sizeof(float);
        if (tail_elems <= 4 * MAX_FP32_REPEAT_MASK && rows <= MAX_REPEAT_ROWS &&
            row_sums2_offset + rows <= tile_elems && float_stride_blocks <= 255) {
            uint32_t first_tail = MinU32(MAX_FP32_REPEAT_MASK, tail_elems);
            LocalTensor<float> row_sums2 = row_sums[row_sums2_offset];
            WholeReduceSum<float>(row_sums, square, static_cast<int32_t>(first_tail),
                                  static_cast<int32_t>(rows), 1, 1,
                                  static_cast<int32_t>(float_stride_blocks));
            for (uint32_t tail_offset = MAX_FP32_REPEAT_MASK; tail_offset < tail_elems;
                 tail_offset += MAX_FP32_REPEAT_MASK) {
                uint32_t cur_tail = MinU32(MAX_FP32_REPEAT_MASK, tail_elems - tail_offset);
                WholeReduceSum<float>(row_sums2, square[tail_offset], static_cast<int32_t>(cur_tail),
                                      static_cast<int32_t>(rows), 1, 1,
                                      static_cast<int32_t>(float_stride_blocks));
                Add(row_sums, row_sums, row_sums2, rows);
            }
            Add(acc, acc, row_sums, rows);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            LocalTensor<float> square_row = square[row * float_stride];
            float part = ReduceFloatTile(square_row, row_sums, sum, tail_elems);
            acc.SetValue(row, acc.GetValue(row) + part);
        }
    }

    __aicore__ inline float ReduceContiguousSegment(uint32_t base, uint32_t count)
    {
        float acc = 0.0f;
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        for (uint32_t offset = 0; offset < count;) {
            uint32_t cur = MinU32(tile_elems, count - offset);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, base + offset, cur);
            input_queue.EnQue(input_local);
            input_local = input_queue.DeQue<T>();
            SquareToFloat(square, input_local, tmp, cur);
            acc += ReduceFloatTile(square, work, sum, cur);
            input_queue.FreeTensor(input_local);
            offset += cur;
        }
        return acc;
    }

    __aicore__ inline uint32_t InputBaseFromOutputGroup(uint32_t group_idx)
    {
        uint32_t base = 0;
        uint32_t rem = group_idx;
        for (int32_t d = static_cast<int32_t>(inner_keep_axis) - 1; d >= 0; --d) {
            if (!IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                base += coord * strides[d];
            }
        }
        return base;
    }

    __aicore__ inline uint32_t InputBaseFromOutput(uint32_t out_idx)
    {
        uint32_t base = 0;
        uint32_t rem = out_idx;
        for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
            if (!IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                base += coord * strides[d];
            }
        }
        return base;
    }

    __aicore__ inline uint32_t OuterReduceOffset(uint32_t reduce_idx)
    {
        uint32_t offset = 0;
        uint32_t rem = reduce_idx;
        for (int32_t d = static_cast<int32_t>(tail_start_axis) - 1; d >= 0; --d) {
            if (IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                offset += coord * strides[d];
            }
        }
        return offset;
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void WriteScalarOutput(uint32_t offset, float value)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        out.SetValue(0, FromFloatScalar<T>(value));
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(output_gm[offset], out, params);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t rank = 0;
    uint32_t axis_mask = 0;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t tile_elems = 0;
    uint32_t tail_reduce_size = 1;
    uint32_t outer_reduce_size = 1;
    uint32_t tail_start_axis = 0;
    uint32_t inner_keep_axis = 0;
    uint32_t inner_keep_size = 1;
    uint32_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t strides[8] = {1, 1, 1, 1, 1, 1, 1, 1};
};

class KernelSeparatedFp16BalancedFoldedTail {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output,
        const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        total_size = tiling.total_size;
        tile_elems = tiling.tile_elems;
        d0_size = tiling.dims[0];
        tail_size = tiling.dims[3];
        d0_stride = tiling.strides[0];
        use_compact_half_wave =
            d0_size >= 2U && d0_size <= 32U &&
            (d0_size % 2U) == 0U &&
            tail_size > 0U && tail_size <= 112U &&
            output_size >= 2048U;
        ConfigureWindow();

        const uint32_t usable_bytes =
            tile_elems * kHostBytesPerTileElem;
        if (use_compact_half_wave) {
            max_rows = kPreferredHalfRows;
        } else {
            const uint32_t bytes_per_row =
                BUFFER_NUM * kD0Batch * tail_size *
                    sizeof(half) +
                tail_size * sizeof(float) +
                2U * sizeof(float);
            max_rows =
                bytes_per_row == 0
                    ? 1U
                    : usable_bytes / bytes_per_row;
            max_rows = MinU32(
                max_rows == 0 ? 1U : max_rows,
                kMaxCohortRows);
        }
        ConfigureLayout();
        while (max_rows > 1 &&
               TotalUbBytes() > usable_bytes) {
            --max_rows;
            ConfigureLayout();
        }

        input_gm.SetGlobalBuffer(
            (__gm__ half *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ half *)output, output_size);
        if (use_compact_half_wave) {
            pipe.InitBuffer(
                half_input_buf, BUFFER_NUM * input_slot_bytes);
        } else {
            pipe.InitBuffer(
                input_queue, BUFFER_NUM, input_slot_bytes);
        }
        pipe.InitBuffer(
            calc_buf,
            acc_bytes +
                (use_compact_half_wave ? kHalfOutputSlots * sum_slot_bytes
                                : 2U * sum_slot_bytes));
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        if (start >= end) {
            return;
        }
        // Keep D0=0 on the half-square initializer, then spread the
        // remaining plane reads across AIVs and successive row tiles.
        uint32_t d0_phase = d0_size > 1U
            ? 1U + static_cast<uint32_t>(GetBlockIdx()) %
                       (d0_size - 1U)
            : 0U;
        if (use_compact_half_wave) {
            StartHalfInputPipe();
            const uint32_t core_idx =
                static_cast<uint32_t>(GetBlockIdx());
            const uint32_t core_count =
                static_cast<uint32_t>(GetBlockNum());
            const uint32_t wave_rows = core_count * max_rows;
            const uint32_t full_waves = output_size / wave_rows;
            uint32_t collected_waves = 0U;
            uint32_t group_output_offset = core_idx * max_rows;
            for (uint32_t wave = 0U; wave < full_waves; ++wave) {
                const uint32_t output_offset =
                    (wave * core_count + core_idx) * max_rows;
                ProcessHalfRows(
                    output_offset, max_rows, d0_phase, collected_waves);
                d0_phase = AdvanceSkewedD0(d0_phase);
                ++collected_waves;
                if (collected_waves == kHalfOutputSlots) {
                    FlushHalfRows(
                        group_output_offset, collected_waves,
                        core_count, 0U, 0U);
                    collected_waves = 0U;
                    group_output_offset = output_offset + wave_rows;
                }
            }

            // Distribute the last incomplete wave across every core.
            const uint32_t full_rows = full_waves * wave_rows;
            const uint32_t residual = output_size - full_rows;
            const uint32_t residual_base = residual / core_count;
            const uint32_t residual_extra = residual % core_count;
            const uint32_t rows = residual_base +
                (core_idx < residual_extra ? 1U : 0U);
            const uint32_t output_offset = full_rows +
                core_idx * residual_base +
                MinU32(core_idx, residual_extra);
            if (rows > 0U) {
                ProcessHalfRows(
                    output_offset, rows, d0_phase, collected_waves);
            }
            if (collected_waves > 0U || rows > 0U) {
                FlushHalfRows(
                    group_output_offset, collected_waves,
                    core_count, output_offset, rows);
            }
            WaitPendingOutputCopy();
            FinishHalfInputPipe();
            return;
        }
        for (uint32_t output_offset = start;
             output_offset < end;) {
            const uint32_t rows =
                MinU32(max_rows, end - output_offset);
            ProcessRows(output_offset, rows, d0_phase);
            output_offset += rows;
            if (d0_size > 1U) {
                d0_phase = AdvanceSkewedD0(d0_phase);
            }
        }
        WaitPendingOutputCopy();
    }

private:
    __aicore__ inline uint32_t AdvanceSkewedD0(uint32_t d0_idx)
    {
        ++d0_idx;
        return d0_idx == d0_size ? 1U : d0_idx;
    }

    __aicore__ inline void ConfigureWindow()
    {
        if (use_compact_half_wave) {
            const uint32_t tail_mod =
                tail_size % kHalfAlignElems;
            head_guard = 0U;
            tail_guard = tail_mod * kHalfCycleRows;
            return;
        }
        const uint32_t tail_mod =
            tail_size % kFloatAlignElems;
        if (tail_mod <= 3U) {
            head_guard = 0U;
            tail_guard = tail_mod * kCycleRows;
        } else {
            const uint32_t phase_delta =
                kFloatAlignElems - tail_mod;
            head_guard = phase_delta * kCycleRows;
            tail_guard =
                (phase_delta + 1U) * kCycleRows;
        }
    }

    __aicore__ inline void ConfigureLayout()
    {
        dense_capacity = max_rows * tail_size;
        input_slot_bytes = AlignUp32(
            kD0Batch * dense_capacity * sizeof(half) +
            kInputGuardBytes);
        acc_bytes = AlignUp32(
            (head_guard + dense_capacity + tail_guard) *
                (use_compact_half_wave ? sizeof(half)
                                : sizeof(float)) +
            kAccumGuardBytes);
        sum_slot_bytes = AlignUp32(
            max_rows *
            (use_compact_half_wave ? sizeof(half)
                            : sizeof(float)));
    }

    __aicore__ inline uint32_t TotalUbBytes()
    {
        return BUFFER_NUM *
                   input_slot_bytes +
               acc_bytes +
               (use_compact_half_wave ? kHalfOutputSlots * sum_slot_bytes
                               : 2U * sum_slot_bytes);
    }

    __aicore__ inline void CopyD0Batch(
        LocalTensor<half> dst, uint32_t output_offset,
        uint32_t d0_start, uint32_t block_elems)
    {
        const uint32_t block_bytes =
            block_elems * sizeof(half);
        const uint32_t gm_offset =
            output_offset * tail_size +
            d0_start * d0_stride;
        const bool gm_aligned =
            (gm_offset * sizeof(half)) %
                UB_ALIGN_BYTES == 0U;

        const bool can_direct =
            gm_aligned &&
            (block_bytes % UB_ALIGN_BYTES) == 0 &&
            block_bytes / UB_ALIGN_BYTES <= 65535U;
        if (can_direct) {
            DataCopyParams params{
                1,
                static_cast<uint16_t>(
                    block_bytes / UB_ALIGN_BYTES),
                0, 0};
            DataCopy(dst, input_gm[gm_offset], params);
            return;
        }

        const uint32_t fetch_elems =
            AlignUp32(
                block_elems * sizeof(half)) /
            sizeof(half);
        const uint32_t final_read_end =
            gm_offset + fetch_elems;
        const bool can_aligned_superset =
            gm_aligned &&
            final_read_end <= total_size &&
            fetch_elems / kHalfAlignElems <= 65535U;
        if (can_aligned_superset) {
            DataCopyParams params{
                1,
                static_cast<uint16_t>(
                    fetch_elems / kHalfAlignElems),
                0, 0};
            DataCopy(
                dst, input_gm[gm_offset], params);
            return;
        }

        DataCopyExtParams params{
            1, block_bytes, 0, 0, 0};
        DataCopyPadExtParams<half> pad{
            false, 0, 0, static_cast<half>(0)};
        DataCopyPad(
            dst, input_gm[gm_offset],
            params, pad);
    }

    __aicore__ inline void AccumulateBatch(
        LocalTensor<float> acc,
        LocalTensor<half> input,
        uint32_t block_elems)
    {
        MulAddDst(
            acc, input, input, block_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateHalfBatch(
        LocalTensor<half> acc,
        LocalTensor<half> input,
        uint32_t block_elems)
    {
        MulAddDst(
            acc, input, input, block_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeBatch(
        LocalTensor<float> acc,
        LocalTensor<half> input,
        uint32_t block_elems)
    {
        Mul(input, input, input, block_elems);
        PipeBarrier<PIPE_V>();
        Cast(
            acc, input, RoundMode::CAST_NONE,
            block_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeHalfBatch(
        LocalTensor<half> acc,
        LocalTensor<half> input,
        uint32_t block_elems)
    {
        Mul(acc, input, input, block_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t SegmentMask(
        uint32_t begin, uint32_t end, uint32_t base)
    {
        const uint32_t local_begin =
            begin > base ? begin - base : 0U;
        const uint32_t local_end =
            end > base ? MinU32(end - base, 64U) : 0U;
        if (local_begin >= local_end ||
            local_begin >= 64U) {
            return 0ULL;
        }
        const uint64_t end_mask =
            local_end == 64U
                ? 0xffffffffffffffffULL
                : ((1ULL << local_end) - 1ULL);
        const uint64_t begin_mask =
            local_begin == 0U
                ? 0ULL
                : ((1ULL << local_begin) - 1ULL);
        return end_mask & ~begin_mask;
    }

    __aicore__ inline void ReduceAndStoreRows(
        LocalTensor<float> acc_all,
        uint32_t output_offset,
        uint32_t rows)
    {
        WaitPendingOutputCopy();
        const uint32_t work_offset = acc_bytes;
        LocalTensor<float> sums =
            calc_buf.GetWithOffset<float>(
                max_rows, work_offset);
        LocalTensor<float> partial =
            calc_buf.GetWithOffset<float>(
                max_rows, work_offset + sum_slot_bytes);

        const uint32_t cohorts =
            MinU32(rows, kCycleRows);
        const uint32_t chunks =
            (tail_size + kFloatAlignElems - 1U +
             kReduceLanes - 1U) /
            kReduceLanes;
        for (uint32_t chunk = 0U;
             chunk < chunks; ++chunk) {
            const uint32_t chunk_start =
                chunk * kReduceLanes;
            if (chunk > 0U && chunk_start >= tail_size) {
                // The last chunk may have no active lanes in some cohorts.
                Duplicate(partial, 0.0f, rows);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t cohort = 0U;
                 cohort < cohorts; ++cohort) {
                const uint32_t logical =
                    head_guard + cohort * tail_size;
                const uint32_t aligned =
                    logical / kFloatAlignElems *
                    kFloatAlignElems;
                const uint32_t phase = logical - aligned;
                const uint32_t row_end =
                    phase + tail_size;
                if (chunk_start >= row_end) {
                    continue;
                }
                const uint32_t begin =
                    phase > chunk_start
                        ? phase - chunk_start
                        : 0U;
                const uint32_t end = MinU32(
                    kReduceLanes,
                    row_end - chunk_start);
                uint64_t mask[2] = {
                    SegmentMask(begin, end, 0U), 0ULL};
                const uint32_t repeats =
                    (rows - 1U - cohort) /
                        kCycleRows +
                    1U;
                if (chunk == 0U) {
                    WholeReduceSum<float>(
                        sums[cohort],
                        acc_all[aligned + chunk_start],
                        mask,
                        static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(kCycleRows),
                        1,
                        static_cast<uint8_t>(tail_size));
                } else {
                    WholeReduceSum<float>(
                        partial[cohort],
                        acc_all[aligned + chunk_start],
                        mask,
                        static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(kCycleRows),
                        1,
                        static_cast<uint8_t>(tail_size));
                }
            }
            PipeBarrier<PIPE_V>();
            if (chunk > 0U) {
                Add(sums, sums, partial, rows);
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<half> out =
            partial.ReinterpretCast<half>();
        Cast(
            out, sums,
            RoundMode::CAST_RINT, rows);
        const event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        CopyOutContiguous(
            output_gm, out,
            output_offset, rows);
        output_reuse_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(
            output_reuse_event);
        output_copy_pending = true;
    }

    __aicore__ inline void ReduceHalfRows(
        LocalTensor<half> acc_all,
        uint32_t rows,
        uint32_t output_slot)
    {
        // Earlier groups may still be draining while this tile accumulates.
        if (output_slot == 0U) {
            WaitPendingOutputCopy();
        }
        LocalTensor<half> partial =
            calc_buf.GetWithOffset<half>(
                max_rows, acc_bytes + output_slot * sum_slot_bytes);
        #pragma unroll
        for (uint32_t cohort = 0U;
             cohort < kHalfCycleRows; ++cohort) {
            if (cohort >= rows) {
                continue;
            }
            const uint32_t logical = cohort * tail_size;
            const uint32_t aligned =
                logical / kHalfAlignElems *
                kHalfAlignElems;
            const uint32_t phase = logical - aligned;
            const uint32_t row_end = phase + tail_size;
            uint64_t mask[2] = {
                SegmentMask(phase, row_end, 0U),
                SegmentMask(phase, row_end, 64U)};
            const uint32_t repeats =
                (rows - 1U - cohort) /
                    kHalfCycleRows +
                1U;
            WholeReduceSum<half>(
                partial[cohort], acc_all[aligned],
                mask,
                static_cast<uint8_t>(repeats),
                static_cast<uint16_t>(kHalfCycleRows),
                1, static_cast<uint8_t>(tail_size));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushHalfRows(
        uint32_t output_offset,
        uint32_t full_tiles,
        uint32_t core_count,
        uint32_t residual_offset,
        uint32_t residual_rows)
    {
        LocalTensor<half> out = calc_buf.GetWithOffset<half>(
            kHalfOutputSlots * sum_slot_bytes / sizeof(half), acc_bytes);
        const event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        if (full_tiles > 0U) {
            const uint32_t block_bytes = max_rows * sizeof(half);
            const uint32_t gm_gap_bytes = (core_count - 1U) * block_bytes;
            if (block_bytes % UB_ALIGN_BYTES == 0U &&
                gm_gap_bytes / UB_ALIGN_BYTES <= 65535U) {
                DataCopyParams params{
                    static_cast<uint16_t>(full_tiles),
                    static_cast<uint16_t>(block_bytes / UB_ALIGN_BYTES),
                    0,
                    static_cast<uint16_t>(gm_gap_bytes / UB_ALIGN_BYTES)};
                DataCopy(output_gm[output_offset], out, params);
            } else {
                DataCopyExtParams params{
                    static_cast<uint16_t>(full_tiles), block_bytes,
                    0, gm_gap_bytes, 0};
                DataCopyPad(output_gm[output_offset], out, params);
            }
        }
        if (residual_rows > 0U) {
            CopyOutContiguous(
                output_gm,
                out[full_tiles * sum_slot_bytes / sizeof(half)],
                residual_offset, residual_rows);
        }
        output_reuse_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(
            output_reuse_event);
        output_copy_pending = true;
    }

    __aicore__ inline void WaitPendingOutputCopy()
    {
        if (!output_copy_pending) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(
            output_reuse_event);
        output_copy_pending = false;
    }

    __aicore__ inline void ProcessRows(
        uint32_t output_offset,
        uint32_t rows,
        uint32_t d0_phase)
    {
        const uint32_t block_elems =
            rows * tail_size;
        LocalTensor<float> acc_all =
            calc_buf.Get<float>(
                acc_bytes / sizeof(float));
        if (head_guard > 0U) {
            Duplicate(
                acc_all, 0.0f,
                head_guard);
        }
        const uint32_t tail_start =
            head_guard + block_elems;
        const uint32_t aligned_tail_start =
            tail_start / kFloatAlignElems *
            kFloatAlignElems;
        const uint32_t tail_clear =
            AlignUp32(
                (tail_start - aligned_tail_start +
                 tail_guard) * sizeof(float)) /
            sizeof(float);
        Duplicate(
            acc_all[aligned_tail_start],
            0.0f, tail_clear);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> acc = acc_all[head_guard];

        LocalTensor<half> input =
            input_queue.AllocTensor<half>();
        CopyD0Batch(
            input, output_offset, 0U,
            block_elems);
        input_queue.EnQue(input);
        if (d0_size == 1U) {
            input = input_queue.DeQue<half>();
            InitializeBatch(acc, input, block_elems);
            input_queue.FreeTensor(input);
            ReduceAndStoreRows(
                acc_all, output_offset, rows);
            return;
        }

        LocalTensor<half> next =
            input_queue.AllocTensor<half>();
        uint32_t d0_idx = d0_phase;
        CopyD0Batch(
            next, output_offset, d0_idx,
            block_elems);
        d0_idx = AdvanceSkewedD0(d0_idx);
        input_queue.EnQue(next);
        input = input_queue.DeQue<half>();
        InitializeBatch(acc, input, block_elems);
        input_queue.FreeTensor(input);

        uint32_t visited_d0 = 2U;
        while (visited_d0 < d0_size) {
            next =
                input_queue.AllocTensor<half>();
            CopyD0Batch(
                next, output_offset, d0_idx,
                block_elems);
            d0_idx = AdvanceSkewedD0(d0_idx);
            input_queue.EnQue(next);

            LocalTensor<half> current =
                input_queue.DeQue<half>();
            AccumulateBatch(
                acc, current, block_elems);
            input_queue.FreeTensor(current);
            ++visited_d0;
        }

        input = input_queue.DeQue<half>();
        AccumulateBatch(
            acc, input, block_elems);
        input_queue.FreeTensor(input);
        ReduceAndStoreRows(
            acc_all, output_offset, rows);
    }

    __aicore__ inline void StartHalfInputPipe()
    {
        #pragma unroll
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            half_input_ready[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            half_input_reuse[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            SetFlag<HardEvent::V_MTE2>(half_input_reuse[slot]);
        }
    }

    __aicore__ inline void FinishHalfInputPipe()
    {
        #pragma unroll
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(half_input_reuse[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(
                half_input_ready[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(
                half_input_reuse[slot]);
        }
    }

    template <uint32_t Slot>
    __aicore__ inline void CopyHalfPlane(
        LocalTensor<half> input, uint32_t gm_offset,
        const DataCopyExtParams &params,
        const DataCopyPadExtParams<half> &pad)
    {
        WaitFlag<HardEvent::V_MTE2>(half_input_reuse[Slot]);
        DataCopyPad(input, input_gm[gm_offset], params, pad);
        SetFlag<HardEvent::MTE2_V>(half_input_ready[Slot]);
    }

    template <uint32_t Slot, bool Initialize = false>
    __aicore__ inline void ConsumeHalfPlane(
        LocalTensor<half> acc, LocalTensor<half> input,
        uint32_t block_elems)
    {
        WaitFlag<HardEvent::MTE2_V>(half_input_ready[Slot]);
        if constexpr (Initialize) {
            InitializeHalfBatch(acc, input, block_elems);
        } else {
            AccumulateHalfBatch(acc, input, block_elems);
        }
        SetFlag<HardEvent::V_MTE2>(half_input_reuse[Slot]);
    }

    __aicore__ inline uint32_t AdvanceHalfPlaneOffset(uint32_t offset)
    {
        offset += d0_stride;
        return offset == d0_size * d0_stride ? d0_stride : offset;
    }

    __aicore__ __attribute__((noinline)) void ProcessHalfRows(
        uint32_t output_offset,
        uint32_t rows,
        uint32_t d0_phase,
        uint32_t output_slot)
    {
        const uint32_t block_elems =
            rows * tail_size;
        LocalTensor<half> acc_all =
            calc_buf.Get<half>(
                acc_bytes / sizeof(half));
        LocalTensor<half> acc = acc_all[head_guard];

        LocalTensor<half> input0 = half_input_buf.GetWithOffset<half>(
            input_slot_bytes / sizeof(half), 0U);
        LocalTensor<half> input1 = half_input_buf.GetWithOffset<half>(
            input_slot_bytes / sizeof(half), input_slot_bytes);
        const uint32_t block_bytes = block_elems * sizeof(half);
        const DataCopyExtParams params{1, block_bytes, 0, 0, 0};
        const DataCopyPadExtParams<half> pad{
            false, 0, 0, static_cast<half>(0)};
        const uint32_t row_offset = output_offset * tail_size;
        uint32_t plane_offset = d0_phase * d0_stride;

        CopyHalfPlane<0U>(input0, row_offset, params, pad);
        CopyHalfPlane<1U>(input1, row_offset + plane_offset, params, pad);
        plane_offset = AdvanceHalfPlaneOffset(plane_offset);
        ConsumeHalfPlane<0U, true>(acc, input0, block_elems);

        // Refill the consumed slot before waiting on the other ready plane.
        for (uint32_t visited_d0 = 2U; visited_d0 < d0_size;
             visited_d0 += 2U) {
            CopyHalfPlane<0U>(input0, row_offset + plane_offset, params, pad);
            plane_offset = AdvanceHalfPlaneOffset(plane_offset);
            ConsumeHalfPlane<1U>(acc, input1, block_elems);

            CopyHalfPlane<1U>(input1, row_offset + plane_offset, params, pad);
            plane_offset = AdvanceHalfPlaneOffset(plane_offset);
            ConsumeHalfPlane<0U>(acc, input0, block_elems);
        }
        ConsumeHalfPlane<1U>(acc, input1, block_elems);
        ReduceHalfRows(acc_all, rows, output_slot);
    }

    static constexpr uint32_t kD0Batch = 1;
    static constexpr uint32_t kHostBytesPerTileElem = 20;
    static constexpr uint32_t kInputGuardBytes = 256;
    static constexpr uint32_t kAccumGuardBytes = 32;
    static constexpr uint32_t kCycleRows = 8;
    static constexpr uint32_t kHalfCycleRows = 16;
    static constexpr uint32_t kPreferredHalfRows = 448;
    static constexpr uint32_t kHalfOutputSlots = 7;
    static constexpr uint32_t kMaxCohortRows =
        kCycleRows * MAX_REPEAT_ROWS;
    static constexpr uint32_t kReduceLanes = 64;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);
    static constexpr uint32_t kHalfAlignElems =
        UB_ALIGN_BYTES / sizeof(half);

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<half> input_gm;
    GlobalTensor<half> output_gm;
    uint32_t output_size = 0;
    uint32_t total_size = 0;
    uint32_t tile_elems = 0;
    uint32_t d0_size = 0;
    uint32_t tail_size = 0;
    uint32_t d0_stride = 0;
    uint32_t max_rows = 1;
    uint32_t dense_capacity = 1;
    uint32_t input_slot_bytes = 0;
    uint32_t acc_bytes = 0;
    uint32_t sum_slot_bytes = 0;
    uint32_t head_guard = 0;
    uint32_t tail_guard = 0;
    bool use_compact_half_wave = false;
    TBuf<TPosition::VECIN> half_input_buf;
    event_t half_input_ready[BUFFER_NUM];
    event_t half_input_reuse[BUFFER_NUM];
    event_t output_reuse_event;
    bool output_copy_pending = false;
};

class KernelSeparatedFp16WaveBalancedFoldedTail {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output,
        const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        total_size = tiling.total_size;
        tile_elems = tiling.tile_elems;
        d0_size = tiling.dims[0];
        tail_size = tiling.dims[3];
        d0_stride = tiling.strides[0];
        padded_stride =
            PaddedStrideElems<float>(tail_size);

        const uint32_t usable_bytes =
            tile_elems * kHostBytesPerTileElem;
        const uint32_t bytes_per_row =
            BUFFER_NUM * padded_stride *
                sizeof(float) +
            tail_size * sizeof(float) +
            padded_stride * sizeof(uint32_t) +
            BUFFER_NUM * sizeof(half);
        max_rows =
            bytes_per_row == 0
                ? 1U
                : usable_bytes / bytes_per_row;
        max_rows = MinU32(
            max_rows == 0 ? 1U : max_rows,
            MAX_REPEAT_ROWS);
        ConfigureLayout();
        while (max_rows > 1 &&
               TotalUbBytes() > usable_bytes) {
            --max_rows;
            ConfigureLayout();
        }

        input_gm.SetGlobalBuffer(
            (__gm__ half *)input, tiling.total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ half *)output, output_size);
        pipe.InitBuffer(
            input_queue, BUFFER_NUM,
            AlignUp32(
                padded_count * sizeof(float) +
                kInputGuardBytes));
        pipe.InitBuffer(
            output_queue, BUFFER_NUM,
            AlignUp32(max_rows * sizeof(half)));
        pipe.InitBuffer(
            calc_buf,
            acc_bytes + offsets_bytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block_idx =
            static_cast<uint32_t>(GetBlockIdx());
        const uint32_t block_num =
            static_cast<uint32_t>(GetBlockNum());
        if (block_num == 0U) {
            return;
        }
        const uint32_t supertile_rows =
            block_num * max_rows;
        const uint32_t full_waves =
            output_size / supertile_rows;
        const uint32_t tail_total =
            output_size - full_waves * supertile_rows;
        const uint32_t tail_base =
            tail_total / block_num;
        const uint32_t tail_extra =
            tail_total - tail_base * block_num;

        LocalTensor<int32_t> offsets =
            calc_buf.GetWithOffset<int32_t>(
                padded_count, acc_bytes);
        const uint32_t offset_rows =
            MinU32(max_rows, output_size);
        BuildOffsets(offsets, offset_rows);

        for (uint32_t wave = 0; wave < full_waves;
             ++wave) {
            const uint32_t output_offset =
                (wave * block_num + block_idx) *
                max_rows;
            ProcessRows(
                output_offset, max_rows, offsets);
        }

        const uint32_t tail_rows =
            tail_base +
            (block_idx < tail_extra ? 1U : 0U);
        if (tail_rows > 0U) {
            const uint32_t output_offset =
                full_waves * supertile_rows +
                block_idx * tail_base +
                MinU32(block_idx, tail_extra);
            ProcessRows(
                output_offset, tail_rows, offsets);
        }
    }

private:
    __aicore__ inline void ConfigureLayout()
    {
        dense_capacity = max_rows * tail_size;
        padded_count = max_rows * padded_stride;
        acc_bytes = AlignUp32(
            dense_capacity * sizeof(float) +
            kAccumGuardBytes);
        offsets_bytes = AlignUp32(
            padded_count * sizeof(uint32_t));
    }

    __aicore__ inline uint32_t TotalUbBytes()
    {
        return BUFFER_NUM *
                   AlignUp32(
                       padded_count * sizeof(float) +
                       kInputGuardBytes) +
               BUFFER_NUM *
                   AlignUp32(max_rows * sizeof(half)) +
               acc_bytes + offsets_bytes;
    }

    __aicore__ inline void CopyD0Batch(
        LocalTensor<half> dst, uint32_t output_offset,
        uint32_t d0_start, uint32_t batch,
        uint32_t block_elems,
        uint32_t &block_stride)
    {
        const uint32_t block_bytes =
            block_elems * sizeof(half);
        const uint32_t gap_bytes =
            (d0_stride - block_elems) * sizeof(half);
        const uint32_t gm_offset =
            output_offset * tail_size +
            d0_start * d0_stride;

        const bool can_direct =
            (block_bytes % UB_ALIGN_BYTES) == 0 &&
            (gap_bytes % UB_ALIGN_BYTES) == 0 &&
            block_bytes / UB_ALIGN_BYTES <= 65535U &&
            gap_bytes / UB_ALIGN_BYTES <= 65535U;
        if (can_direct) {
            DataCopyParams params{
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(
                    block_bytes / UB_ALIGN_BYTES),
                static_cast<uint16_t>(
                    gap_bytes / UB_ALIGN_BYTES),
                0};
            DataCopy(dst, input_gm[gm_offset], params);
            block_stride =
                block_bytes / sizeof(half);
            return;
        }

        const uint32_t fetch_elems =
            AlignUp32(
                block_elems * sizeof(half)) /
            sizeof(half);
        const uint32_t final_read_end =
            gm_offset +
            (batch - 1U) * d0_stride + fetch_elems;
        const bool fetch_within_stride =
            fetch_elems <= d0_stride;
        const bool stride_encodable =
            batch == 1U ||
            (fetch_within_stride &&
             (d0_stride % kHalfAlignElems) == 0 &&
             (d0_stride - fetch_elems) /
                     kHalfAlignElems <=
                 65535U);
        const bool can_aligned_superset =
            stride_encodable &&
            fetch_within_stride &&
            final_read_end <= total_size &&
            fetch_elems / kHalfAlignElems <= 65535U;
        if (can_aligned_superset) {
            DataCopyParams params{
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(
                    fetch_elems / kHalfAlignElems),
                static_cast<uint16_t>(
                    batch == 1U
                        ? 0U
                        : (d0_stride - fetch_elems) /
                              kHalfAlignElems),
                0};
            DataCopy(
                dst, input_gm[gm_offset], params);
            block_stride = fetch_elems;
            return;
        }

        DataCopyExtParams params{
            static_cast<uint16_t>(batch),
            block_bytes, gap_bytes, 0, 0};
        DataCopyPadExtParams<half> pad{
            false, 0, 0, static_cast<half>(0)};
        DataCopyPad(
            dst, input_gm[gm_offset],
            params, pad);
        block_stride =
            AlignUp32(block_bytes) / sizeof(half);
    }

    __aicore__ inline void BuildOffsets(
        LocalTensor<int32_t> offsets,
        uint32_t rows)
    {
        CreateVecIndex(
            offsets, static_cast<int32_t>(0),
            padded_stride);
        PipeBarrier<PIPE_V>();
        Muls(
            offsets, offsets,
            static_cast<int32_t>(sizeof(float)),
            padded_stride);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1;
             row < rows; ++row) {
            Adds(
                offsets[row * padded_stride],
                offsets,
                static_cast<int32_t>(
                    row * tail_size * sizeof(float)),
                padded_stride);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateBatch(
        LocalTensor<float> acc,
        LocalTensor<half> input,
        uint32_t block_stride,
        uint32_t block_elems,
        uint32_t batch)
    {
        for (uint32_t b = 0; b < batch; ++b) {
            MulAddDst(
                acc,
                input[b * block_stride],
                input[b * block_stride],
                block_elems);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceAndStoreRows(
        LocalTensor<float> acc,
        LocalTensor<float> padded,
        uint32_t output_offset,
        uint32_t rows)
    {
        LocalTensor<float> row_sums = acc;
        const uint32_t stride_blocks =
            padded_stride * sizeof(float) /
            UB_ALIGN_BYTES;
        const uint32_t first =
            MinU32(MAX_FP32_REPEAT_MASK, tail_size);
        BinaryRepeatParams repeat_params{
            1, 1, 1,
            static_cast<uint8_t>(stride_blocks),
            static_cast<uint8_t>(stride_blocks),
            static_cast<uint8_t>(stride_blocks)};
        if (tail_size > 3U * MAX_FP32_REPEAT_MASK) {
            Add(
                padded, padded,
                padded[MAX_FP32_REPEAT_MASK],
                static_cast<uint64_t>(MAX_FP32_REPEAT_MASK),
                static_cast<uint8_t>(rows),
                repeat_params);
            const uint32_t fourth_chunk =
                tail_size - 3U * MAX_FP32_REPEAT_MASK;
            Add(
                padded[2U * MAX_FP32_REPEAT_MASK],
                padded[2U * MAX_FP32_REPEAT_MASK],
                padded[3U * MAX_FP32_REPEAT_MASK],
                static_cast<uint64_t>(fourth_chunk),
                static_cast<uint8_t>(rows),
                repeat_params);
            PipeBarrier<PIPE_V>();
            Add(
                padded, padded,
                padded[2U * MAX_FP32_REPEAT_MASK],
                static_cast<uint64_t>(MAX_FP32_REPEAT_MASK),
                static_cast<uint8_t>(rows),
                repeat_params);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t tail_offset = MAX_FP32_REPEAT_MASK;
                 tail_offset < tail_size;
                 tail_offset += MAX_FP32_REPEAT_MASK) {
                const uint32_t current = MinU32(
                    MAX_FP32_REPEAT_MASK,
                    tail_size - tail_offset);
                Add(
                    padded, padded, padded[tail_offset],
                    static_cast<uint64_t>(current),
                    static_cast<uint8_t>(rows),
                    repeat_params);
                PipeBarrier<PIPE_V>();
            }
        }
        WholeReduceSum<float>(
            row_sums, padded,
            static_cast<int32_t>(first),
            static_cast<int32_t>(rows),
            1, 1,
            static_cast<int32_t>(stride_blocks));

        LocalTensor<half> out =
            output_queue.AllocTensor<half>();
        Cast(
            out, row_sums,
            RoundMode::CAST_RINT, rows);
        output_queue.EnQue(out);
        out = output_queue.DeQue<half>();
        CopyOutContiguous(
            output_gm, out,
            output_offset, rows);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void ProcessRows(
        uint32_t output_offset,
        uint32_t rows,
        LocalTensor<int32_t> offsets)
    {
        const uint32_t block_elems =
            rows * tail_size;
        LocalTensor<float> acc =
            calc_buf.Get<float>(
                acc_bytes / sizeof(float));
        Duplicate(acc, 0.0f, block_elems);
        PipeBarrier<PIPE_V>();

        uint32_t d0_start = 0;
        uint32_t previous_batch =
            MinU32(kD0Batch, d0_size);
        LocalTensor<half> input =
            input_queue.AllocTensor<half>();
        uint32_t previous_stride =
            0;
        CopyD0Batch(
            input, output_offset, d0_start,
            previous_batch, block_elems,
            previous_stride);
        input_queue.EnQue(input);
        d0_start += previous_batch;
        while (d0_start < d0_size) {
            const uint32_t current_batch =
                MinU32(
                    kD0Batch, d0_size - d0_start);
            LocalTensor<half> next =
                input_queue.AllocTensor<half>();
            uint32_t current_stride = 0;
            CopyD0Batch(
                next, output_offset, d0_start,
                current_batch, block_elems,
                current_stride);
            input_queue.EnQue(next);

            LocalTensor<half> current =
                input_queue.DeQue<half>();
            AccumulateBatch(
                acc, current, previous_stride,
                block_elems, previous_batch);
            input_queue.FreeTensor(current);
            d0_start += current_batch;
            previous_batch = current_batch;
            previous_stride = current_stride;
        }

        input = input_queue.DeQue<half>();
        AccumulateBatch(
            acc, input, previous_stride,
            block_elems, previous_batch);
        input_queue.FreeTensor(input);

        LocalTensor<half> padded_storage =
            input_queue.AllocTensor<half>();
        LocalTensor<float> padded =
            padded_storage.ReinterpretCast<float>();
        Gather(
            padded, acc,
            offsets.ReinterpretCast<uint32_t>(),
            0, rows * padded_stride);
        PipeBarrier<PIPE_V>();
        ReduceAndStoreRows(
            acc, padded, output_offset, rows);
        event_t reuse_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(
                HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(reuse_event);
        WaitFlag<HardEvent::V_MTE2>(reuse_event);
        input_queue.FreeTensor(padded_storage);
    }

    static constexpr uint32_t kD0Batch = 2;
    static constexpr uint32_t kHostBytesPerTileElem = 20;
    static constexpr uint32_t kInputGuardBytes = 256;
    static constexpr uint32_t kAccumGuardBytes = 32;
    static constexpr uint32_t kHalfAlignElems =
        UB_ALIGN_BYTES / sizeof(half);

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<half> input_gm;
    GlobalTensor<half> output_gm;
    uint32_t output_size = 0;
    uint32_t total_size = 0;
    uint32_t tile_elems = 0;
    uint32_t d0_size = 0;
    uint32_t tail_size = 0;
    uint32_t d0_stride = 0;
    uint32_t max_rows = 1;
    uint32_t padded_stride = 1;
    uint32_t dense_capacity = 1;
    uint32_t padded_count = 1;
    uint32_t acc_bytes = 0;
    uint32_t offsets_bytes = 0;
};

class KernelSeparatedFp16WaveAlignedWindow {
public:
    __aicore__ inline void Init(
        TPipe &pipe, GM_ADDR input, GM_ADDR output,
        const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        total_size = tiling.total_size;
        d0_size = tiling.dims[0];
        tail_size = tiling.dims[3];
        d0_stride = tiling.strides[0];

        uint32_t layout_mode =
            tiling.separated_fp16_layout_mode;
        use_gm512 =
            (layout_mode & kLayoutGm512) != 0U;
        layout_mode &= ~kLayoutGm512;
        max_rows = tiling.separated_fp16_layout_rows;
        const uint32_t half_tail_mod =
            tail_size % kHalfAlignElems;
        use_half_acc = layout_mode == kLayoutHalf ||
                       layout_mode == kLayoutHalfShort;
        if (use_half_acc) {
            ConfigureWindow(
                half_tail_mod, kHalfAlignElems, 7U,
                sizeof(half));
        } else {
            ConfigureWindow(
                tail_size % kFloatAlignElems,
                kFloatAlignElems, 3U, sizeof(float));
        }
        ConfigureLayout();

        input_gm.SetGlobalBuffer(
            (__gm__ half *)input, total_size);
        input_gm.SetL2CacheHint<CacheRwMode::READ>(
            CacheMode::CACHE_MODE_DISABLE);
        output_gm.SetGlobalBuffer(
            (__gm__ half *)output, output_size);
        pipe.InitBuffer(
            input_queue, BUFFER_NUM, input_slot_bytes);
        pipe.InitBuffer(
            output_queue, BUFFER_NUM,
            AlignUp32(max_rows * sizeof(half)));
        pipe.InitBuffer(
            calc_buf,
            acc_bytes + offsets_bytes + calc_work_bytes +
                2U * sum_slot_bytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block_idx =
            static_cast<uint32_t>(GetBlockIdx());
        const uint32_t block_num =
            static_cast<uint32_t>(GetBlockNum());
        if (block_num == 0U) {
            return;
        }
        const uint32_t supertile_rows =
            block_num * max_rows;
        const uint32_t full_waves =
            output_size / supertile_rows;
        const uint32_t tail_total =
            output_size - full_waves * supertile_rows;
        const uint32_t tail_base =
            tail_total / block_num;
        const uint32_t tail_extra =
            tail_total - tail_base * block_num;

        const uint32_t tail_rows =
            tail_base +
            (block_idx < tail_extra ? 1U : 0U);
        const uint32_t task_count =
            full_waves + (tail_rows > 0U ? 1U : 0U);
        if (task_count == 0U) {
            return;
        }

        uint32_t output_offset = 0U;
        uint32_t rows = 0U;
        ResolveTask(
            0U, block_idx, block_num, full_waves,
            supertile_rows, tail_base, tail_extra,
            output_offset, rows);
        PrimeRows(output_offset, rows);

        LocalTensor<uint32_t> offsets;
        if (!use_half_acc) {
            offsets = calc_buf.GetWithOffset<uint32_t>(offset_count, acc_bytes);
            BuildFloatOffsets(offsets);
        }

        for (uint32_t task_idx = 0U;
             task_idx < task_count; ++task_idx) {
            const bool has_next = task_idx + 1U < task_count;
            uint32_t next_output_offset = 0U;
            uint32_t next_rows = 0U;
            if (has_next) {
                ResolveTask(
                    task_idx + 1U, block_idx, block_num,
                    full_waves, supertile_rows, tail_base,
                    tail_extra, next_output_offset, next_rows);
            }

            if (use_half_acc) {
                AccumulateHalfPrimedRows(
                    output_offset, rows, has_next,
                    next_output_offset, next_rows);
            } else {
                AccumulateFloatPrimedRows(
                    output_offset, rows, has_next,
                    next_output_offset, next_rows);
            }

            if (has_next && d0_size > kD0Batch) {
                IssueInputPacket(
                    next_output_offset, next_rows,
                    kD0Batch);
            }

            if (use_half_acc) {
                ReduceAndStoreHalfRows(
                    output_offset, rows);
            } else {
                ReduceAndStoreFloatRows(
                    output_offset, rows, offsets);
            }
            output_offset = next_output_offset;
            rows = next_rows;
        }
    }

private:
    __aicore__ inline void ResolveTask(
        uint32_t task_idx, uint32_t block_idx,
        uint32_t block_num, uint32_t full_waves,
        uint32_t supertile_rows, uint32_t tail_base,
        uint32_t tail_extra, uint32_t &output_offset,
        uint32_t &rows)
    {
        if (task_idx < full_waves) {
            output_offset =
                (task_idx * block_num + block_idx) *
                max_rows;
            rows = max_rows;
            return;
        }
        output_offset =
            full_waves * supertile_rows +
            block_idx * tail_base +
            MinU32(block_idx, tail_extra);
        rows = tail_base +
               (block_idx < tail_extra ? 1U : 0U);
    }

    __aicore__ inline void ConfigureWindow(
        uint32_t tail_mod, uint32_t align_elems,
        uint32_t low_phase_limit,
        uint32_t accumulator_bytes)
    {
        if (tail_mod <= low_phase_limit) {
            const uint32_t phase_delta = tail_mod;
            window_stride = tail_size - phase_delta;
            head_guard = 0U;
            tail_guard = phase_delta * kCycleRows;
            window_elems =
                tail_size + phase_delta *
                                (kCycleRows - 1U);
        } else {
            const uint32_t phase_delta =
                align_elems - tail_mod;
            window_stride = tail_size + phase_delta;
            head_guard = phase_delta * kCycleRows;
            tail_guard =
                (phase_delta + 1U) * kCycleRows;
            window_elems =
                tail_size + head_guard + phase_delta;
        }
        extra_count = window_elems - tail_size;
        extra_stride =
            AlignUp32(extra_count * accumulator_bytes) /
            accumulator_bytes;
    }

    __aicore__ inline void ConfigureLayout()
    {
        dense_capacity = max_rows * tail_size;
        const uint32_t aligned_dense =
            AlignUp32(dense_capacity * sizeof(half)) /
            sizeof(half);
        input_capacity = kD0Batch * aligned_dense;
        input_slot_bytes = use_gm512
            ? input_capacity * sizeof(half) +
                  kGm512InputGuardBytes
            : AlignUp32(
                  input_capacity * sizeof(half) +
                  kInputGuardBytes);
        acc_count =
            head_guard + dense_capacity + tail_guard;
        acc_bytes = AlignUp32(
            acc_count *
            (use_half_acc ? sizeof(half) : sizeof(float)));
        offset_count = use_half_acc ? 0U : kCycleRows * extra_stride;
        offsets_bytes = AlignUp32(
            offset_count * sizeof(uint32_t));
        calc_work_bytes = kFoldBytes;
        if (use_half_acc) {
            calc_work_bytes = AlignUp32(max_rows * sizeof(half));
            if (tail_size > 255U) {
                const uint32_t chunks = (tail_size + 15U + 127U) / 128U;
                const uint32_t partial_stride = (chunks + 7U) / 8U * 8U;
                const uint32_t partial_count = max_rows * partial_stride;
                calc_work_bytes = AlignUp32(partial_count * sizeof(half)) +
                                  AlignUp32(partial_count * sizeof(float));
            }
        }
        sum_slot_bytes = AlignUp32(
            max_rows * sizeof(float));
    }

    __aicore__ inline void CopyD0Batch(
        LocalTensor<half> dst, uint32_t output_offset,
        uint32_t d0_start, uint32_t batch,
        uint32_t block_elems,
        uint32_t &first_offset,
        uint32_t &second_offset)
    {
        const uint32_t block_bytes =
            block_elems * sizeof(half);
        const uint32_t gap_bytes =
            (d0_stride - block_elems) * sizeof(half);
        const uint32_t gm_offset =
            output_offset * tail_size +
            d0_start * d0_stride;
        const bool gm_aligned =
            (gm_offset * sizeof(half)) %
                UB_ALIGN_BYTES == 0U;

        if (use_gm512 && batch == kD0Batch && gm_aligned &&
            block_bytes % UB_ALIGN_BYTES == 0U &&
            (d0_stride * sizeof(half)) % UB_ALIGN_BYTES == 0U) {
            const uint64_t first_byte =
                static_cast<uint64_t>(gm_offset) * sizeof(half);
            const uint64_t second_byte =
                (static_cast<uint64_t>(gm_offset) + d0_stride) *
                sizeof(half);
            const uint32_t first_prefix =
                static_cast<uint32_t>(
                    first_byte % kGmAlignBytes);
            const uint32_t second_prefix =
                static_cast<uint32_t>(
                    second_byte % kGmAlignBytes);
            const uint32_t common_prefix =
                first_prefix > second_prefix
                    ? first_prefix
                    : second_prefix;
            const uint32_t expanded_bytes =
                block_bytes + common_prefix;
            const uint64_t first_aligned_byte =
                first_byte - first_prefix;
            const uint64_t second_aligned_byte =
                second_byte - second_prefix;
            const uint64_t final_read_end =
                second_aligned_byte + expanded_bytes;
            const uint64_t source_stride =
                second_aligned_byte - first_aligned_byte -
                expanded_bytes;
            if (final_read_end <=
                    static_cast<uint64_t>(total_size) *
                        sizeof(half) &&
                source_stride <= 0xffffffffULL) {
                DataCopyExtParams params{
                    static_cast<uint16_t>(kD0Batch),
                    expanded_bytes,
                    static_cast<uint32_t>(source_stride),
                    0, 0};
                DataCopyPadExtParams<half> pad{
                    false, 0, 0, static_cast<half>(0)};
                DataCopyPad(
                    dst,
                    input_gm[static_cast<uint32_t>(
                        first_aligned_byte / sizeof(half))],
                    params, pad);
                first_offset = first_prefix / sizeof(half);
                second_offset =
                    expanded_bytes / sizeof(half) +
                    second_prefix / sizeof(half);
                return;
            }
        }

        const bool can_direct =
            gm_aligned &&
            (block_bytes % UB_ALIGN_BYTES) == 0U &&
            (gap_bytes % UB_ALIGN_BYTES) == 0U &&
            block_bytes / UB_ALIGN_BYTES <= 65535U &&
            gap_bytes / UB_ALIGN_BYTES <= 65535U;
        if (can_direct) {
            DataCopyParams params{
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(
                    block_bytes / UB_ALIGN_BYTES),
                static_cast<uint16_t>(
                    gap_bytes / UB_ALIGN_BYTES),
                0};
            DataCopy(dst, input_gm[gm_offset], params);
            first_offset = 0U;
            second_offset = block_elems;
            return;
        }

        const uint32_t fetch_elems =
            AlignUp32(block_bytes) / sizeof(half);
        const uint32_t final_read_end =
            gm_offset + (batch - 1U) * d0_stride +
            fetch_elems;
        const bool fetch_within_stride =
            fetch_elems <= d0_stride;
        const bool stride_encodable =
            batch == 1U ||
            (fetch_within_stride &&
             (d0_stride % kHalfAlignElems) == 0U &&
             (d0_stride - fetch_elems) /
                     kHalfAlignElems <=
                 65535U);
        const bool can_aligned_superset =
            gm_aligned && stride_encodable &&
            fetch_within_stride &&
            final_read_end <= total_size &&
            fetch_elems / kHalfAlignElems <= 65535U;
        if (can_aligned_superset) {
            DataCopyParams params{
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(
                    fetch_elems / kHalfAlignElems),
                static_cast<uint16_t>(
                    batch == 1U
                        ? 0U
                        : (d0_stride - fetch_elems) /
                              kHalfAlignElems),
                0};
            DataCopy(dst, input_gm[gm_offset], params);
            first_offset = 0U;
            second_offset = fetch_elems;
            return;
        }

        DataCopyExtParams params{
            static_cast<uint16_t>(batch),
            block_bytes, gap_bytes, 0, 0};
        DataCopyPadExtParams<half> pad{
            false, 0, 0, static_cast<half>(0)};
        DataCopyPad(
            dst, input_gm[gm_offset], params, pad);
        first_offset = 0U;
        second_offset =
            AlignUp32(block_bytes) / sizeof(half);
    }


    __aicore__ inline void BuildFloatOffsets(
        LocalTensor<uint32_t> offsets)
    {
        LocalTensor<int32_t> signed_offsets =
            offsets.ReinterpretCast<int32_t>();
        const uint32_t tail_mod =
            tail_size % kFloatAlignElems;
        const bool low_phase = tail_mod <= 3U;
        const uint32_t phase_delta =
            low_phase
                ? tail_mod
                : kFloatAlignElems - tail_mod;

        CreateVecIndex(
            signed_offsets, static_cast<int32_t>(0),
            extra_stride);
        PipeBarrier<PIPE_V>();
        Muls(
            signed_offsets, signed_offsets,
            static_cast<int32_t>(sizeof(float)),
            extra_stride);
        PipeBarrier<PIPE_V>();

        for (uint32_t row_block = 1U;
             row_block < kCycleRows; row_block <<= 1U) {
            Adds(
                signed_offsets[row_block * extra_stride],
                signed_offsets,
                static_cast<int32_t>(
                    row_block * window_stride *
                    sizeof(float)),
                row_block * extra_stride);
            PipeBarrier<PIPE_V>();
        }

        const UnaryRepeatParams masked_params{
            1, 1, 0, 0};
        const uint64_t valid_bits =
            (1ULL << extra_count) - 1ULL;
        const uint64_t high_phase_bits =
            ((1ULL << phase_delta) - 1ULL) <<
            head_guard;
        uint64_t lane_bits =
            low_phase ? valid_bits : high_phase_bits;
        for (uint32_t row = 0; row < kCycleRows;
             ++row) {
            if (lane_bits != 0ULL) {
                uint64_t mask[2] = {lane_bits, 0ULL};
                Adds(
                    signed_offsets[row * extra_stride],
                    signed_offsets[row * extra_stride],
                    static_cast<int32_t>(
                        tail_size * sizeof(float)),
                    mask, 1, masked_params);
            }
            lane_bits =
                low_phase
                    ? (lane_bits << phase_delta) &
                          valid_bits
                    : (lane_bits >> phase_delta) |
                          high_phase_bits;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t SegmentMask(
        uint32_t begin, uint32_t end, uint32_t base)
    {
        const uint32_t local_begin =
            begin > base ? begin - base : 0U;
        const uint32_t local_end =
            end > base ? MinU32(end - base, 64U) : 0U;
        if (local_begin >= local_end ||
            local_begin >= 64U) {
            return 0ULL;
        }
        const uint64_t end_mask =
            local_end == 64U
                ? 0xffffffffffffffffULL
                : ((1ULL << local_end) - 1ULL);
        const uint64_t begin_mask =
            local_begin == 0U
                ? 0ULL
                : ((1ULL << local_begin) - 1ULL);
        return end_mask & ~begin_mask;
    }

    __aicore__ inline void AccumulateHalfBatch(
        LocalTensor<half> acc,
        LocalTensor<half> input,
        uint32_t first_offset,
        uint32_t second_offset,
        uint32_t block_elems,
        uint32_t batch)
    {
        MulAddDst(
            acc, input[first_offset],
            input[first_offset], block_elems);
        if (batch > 1U) {
            MulAddDst(
                acc, input[second_offset],
                input[second_offset], block_elems);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateFloatBatch(
        LocalTensor<float> acc,
        LocalTensor<half> input,
        uint32_t first_offset,
        uint32_t second_offset,
        uint32_t block_elems,
        uint32_t batch)
    {
        MulAddDst(
            acc, input[first_offset],
            input[first_offset], block_elems);
        if (batch > 1U) {
            MulAddDst(
                acc, input[second_offset],
                input[second_offset], block_elems);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeHalfGuards(
        LocalTensor<half> acc_all, uint32_t block_elems)
    {
        if (head_guard > 0U) {
            Duplicate(
                acc_all, static_cast<half>(0.0f),
                head_guard);
        }
        const uint32_t tail_start =
            head_guard + block_elems;
        const uint32_t aligned_tail_start =
            (tail_start / kHalfAlignElems) *
            kHalfAlignElems;
        const uint32_t tail_clear_elems =
            AlignUp32(
                (tail_start - aligned_tail_start +
                 tail_guard) * sizeof(half)) /
            sizeof(half);
        Duplicate(
            acc_all[aligned_tail_start],
            static_cast<half>(0.0f), tail_clear_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeHalfBatch(
        LocalTensor<half> acc,
        LocalTensor<half> input,
        uint32_t first_offset,
        uint32_t second_offset,
        uint32_t block_elems,
        uint32_t batch)
    {
        Mul(
            acc, input[first_offset], input[first_offset],
            block_elems);
        PipeBarrier<PIPE_V>();
        if (batch > 1U) {
            MulAddDst(
                acc, input[second_offset],
                input[second_offset], block_elems);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void InitializeFloatGuards(
        LocalTensor<float> acc_all,
        uint32_t block_elems)
    {
        if (head_guard > 0U) {
            Duplicate(acc_all, 0.0f, head_guard);
        }
        const uint32_t tail_start =
            head_guard + block_elems;
        const uint32_t aligned_tail_start =
            (tail_start / kFloatAlignElems) *
            kFloatAlignElems;
        const uint32_t tail_clear_elems =
            AlignUp32(
                (tail_start - aligned_tail_start +
                 tail_guard) * sizeof(float)) /
            sizeof(float);
        Duplicate(
            acc_all[aligned_tail_start],
            0.0f, tail_clear_elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeFloatBatch(
        LocalTensor<float> acc,
        LocalTensor<half> input,
        uint32_t first_offset,
        uint32_t second_offset,
        uint32_t block_elems,
        uint32_t batch)
    {
        Mul(
            input[first_offset], input[first_offset],
            input[first_offset], block_elems);
        PipeBarrier<PIPE_V>();
        Cast(
            acc, input[first_offset], RoundMode::CAST_NONE,
            block_elems);
        PipeBarrier<PIPE_V>();
        if (batch > 1U) {
            MulAddDst(
                acc, input[second_offset],
                input[second_offset], block_elems);
            PipeBarrier<PIPE_V>();
        }
    }


    __aicore__ inline void ReduceFloatAlignedRows(
        LocalTensor<float> acc,
        LocalTensor<uint32_t> offsets,
        uint32_t rows)
    {
        const uint32_t fold_offset =
            acc_bytes + offsets_bytes;
        const uint32_t sums_offset =
            fold_offset + kFoldBytes;
        LocalTensor<float> fold =
            calc_buf.GetWithOffset<float>(
                kCycleRows * MAX_FP32_REPEAT_MASK,
                fold_offset);
        LocalTensor<float> sums =
            calc_buf.GetWithOffset<float>(
                max_rows, sums_offset);
        LocalTensor<float> partial =
            calc_buf.GetWithOffset<float>(
                max_rows, sums_offset + sum_slot_bytes);
        const uint32_t stride_blocks =
            window_stride * sizeof(float) /
            UB_ALIGN_BYTES;
        const uint32_t fold_stride_blocks =
            MAX_FP32_REPEAT_MASK * sizeof(float) /
            UB_ALIGN_BYTES;
        const uint32_t extra_stride_blocks =
            extra_stride * sizeof(float) /
            UB_ALIGN_BYTES;

        for (uint32_t group_start = 0;
             group_start < rows;
             group_start += kCycleRows) {
            const uint32_t current_rows = MinU32(
                kCycleRows, rows - group_start);
            const uint32_t group_base =
                group_start * tail_size;
            Gather(
                fold, acc, offsets,
                group_base * sizeof(float),
                current_rows * extra_stride);
            PipeBarrier<PIPE_V>();
            WholeReduceSum<float>(
                partial[group_start], fold,
                static_cast<int32_t>(extra_count),
                static_cast<int32_t>(current_rows),
                1, 1,
                static_cast<int32_t>(
                    extra_stride_blocks));
            PipeBarrier<PIPE_V>();

            const UnaryRepeatParams copy_params{
                1, 1,
                static_cast<uint8_t>(fold_stride_blocks),
                static_cast<uint8_t>(stride_blocks)};
            Adds(
                fold, acc[group_base], 0.0f,
                static_cast<uint64_t>(
                    MAX_FP32_REPEAT_MASK),
                static_cast<uint8_t>(current_rows),
                copy_params);
            PipeBarrier<PIPE_V>();

            const BinaryRepeatParams fold_params{
                1, 1, 1,
                static_cast<uint8_t>(fold_stride_blocks),
                static_cast<uint8_t>(fold_stride_blocks),
                static_cast<uint8_t>(stride_blocks)};
            for (uint32_t window_offset =
                     MAX_FP32_REPEAT_MASK;
                 window_offset < window_elems;
                 window_offset += MAX_FP32_REPEAT_MASK) {
                const uint32_t current = MinU32(
                    MAX_FP32_REPEAT_MASK,
                    window_elems - window_offset);
                Add(
                    fold, fold,
                    acc[group_base + window_offset],
                    static_cast<uint64_t>(current),
                    static_cast<uint8_t>(current_rows),
                    fold_params);
                PipeBarrier<PIPE_V>();
            }

            WholeReduceSum<float>(
                sums[group_start], fold,
                static_cast<int32_t>(
                    MAX_FP32_REPEAT_MASK),
                static_cast<int32_t>(current_rows),
                1, 1,
                static_cast<int32_t>(
                    fold_stride_blocks));
            PipeBarrier<PIPE_V>();
            Sub(
                sums[group_start], sums[group_start],
                partial[group_start], current_rows);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline LocalTensor<half> &AllocateInput()
    {
        LocalTensor<half> &input = input_slots[input_alloc_slot];
        input_queue.AllocTensor<half>(input);
        input_alloc_slot = (input_alloc_slot + 1U) % BUFFER_NUM;
        return input;
    }

    __aicore__ inline LocalTensor<half> &DequeueInput()
    {
        LocalTensor<half> &input = input_slots[input_dequeue_slot];
        input_queue.DeQue<half>(input);
        input_dequeue_slot = (input_dequeue_slot + 1U) % BUFFER_NUM;
        return input;
    }

    __aicore__ inline void StoreInputOffsets(
        uint32_t slot, uint32_t first_offset,
        uint32_t second_offset)
    {
        if (slot == 0U) {
            input_first_offset_slot0 = first_offset;
            input_second_offset_slot0 = second_offset;
        } else {
            input_first_offset_slot1 = first_offset;
            input_second_offset_slot1 = second_offset;
        }
    }

    __aicore__ inline void LoadInputOffsets(
        uint32_t slot, uint32_t &first_offset,
        uint32_t &second_offset)
    {
        if (slot == 0U) {
            first_offset = input_first_offset_slot0;
            second_offset = input_second_offset_slot0;
        } else {
            first_offset = input_first_offset_slot1;
            second_offset = input_second_offset_slot1;
        }
    }

    __aicore__ inline void IssueInputPacket(
        uint32_t output_offset, uint32_t rows,
        uint32_t d0_start)
    {
        const uint32_t slot = input_alloc_slot;
        const uint32_t batch =
            MinU32(kD0Batch, d0_size - d0_start);
        LocalTensor<half> &input = AllocateInput();
        uint32_t first_offset = 0U;
        uint32_t second_offset = 0U;
        CopyD0Batch(
            input, output_offset, d0_start, batch,
            rows * tail_size, first_offset,
            second_offset);
        StoreInputOffsets(
            slot, first_offset, second_offset);
        input_queue.EnQue(input);
    }

    __aicore__ inline void PrimeRows(
        uint32_t output_offset, uint32_t rows)
    {
        const uint32_t packet_count =
            (d0_size + kD0Batch - 1U) / kD0Batch;
        const uint32_t prefetch_count =
            MinU32(BUFFER_NUM, packet_count);
        for (uint32_t packet = 0U;
             packet < prefetch_count; ++packet) {
            IssueInputPacket(
                output_offset, rows, packet * kD0Batch);
        }
    }

    __aicore__ inline void AccumulateHalfPrimedRows(
        uint32_t output_offset, uint32_t rows,
        bool has_next, uint32_t next_output_offset,
        uint32_t next_rows)
    {
        const uint32_t block_elems = rows * tail_size;
        LocalTensor<half> acc_all =
            calc_buf.Get<half>(acc_bytes / sizeof(half));
        InitializeHalfGuards(acc_all, block_elems);
        LocalTensor<half> acc = acc_all[head_guard];

        const uint32_t packet_count =
            (d0_size + kD0Batch - 1U) / kD0Batch;
        const uint32_t prefetch_count =
            MinU32(BUFFER_NUM, packet_count);
        uint32_t next_d0_start =
            prefetch_count * kD0Batch;
        bool initialize = true;
        for (uint32_t packet = 0U;
             packet < packet_count; ++packet) {
            if (has_next && packet + 1U == packet_count) {
                IssueInputPacket(
                    next_output_offset, next_rows, 0U);
            }
            const uint32_t batch = MinU32(
                kD0Batch, d0_size - packet * kD0Batch);
            const uint32_t slot = input_dequeue_slot;
            LocalTensor<half> &current = DequeueInput();
            uint32_t first_offset = 0U;
            uint32_t second_offset = 0U;
            LoadInputOffsets(
                slot, first_offset, second_offset);
            if (initialize) {
                InitializeHalfBatch(
                    acc, current, first_offset,
                    second_offset,
                    block_elems, batch);
                initialize = false;
            } else {
                AccumulateHalfBatch(
                    acc, current, first_offset,
                    second_offset,
                    block_elems, batch);
            }
            input_queue.FreeTensor(current);
            if (next_d0_start < d0_size) {
                IssueInputPacket(
                    output_offset, rows, next_d0_start);
                next_d0_start += kD0Batch;
            }
        }
    }

    __aicore__ inline void ReduceWideHalfRows(
        LocalTensor<half> acc, LocalTensor<float> sums,
        uint32_t rows, uint32_t work_offset)
    {
        const uint32_t chunks = (tail_size + 15U + 127U) / 128U;
        const uint32_t partial_stride = (chunks + 7U) / 8U * 8U;
        const uint32_t partial_capacity = max_rows * partial_stride;
        const uint32_t half_bytes = AlignUp32(partial_capacity * sizeof(half));
        LocalTensor<half> partial = calc_buf.GetWithOffset<half>(
            partial_capacity, work_offset);
        LocalTensor<float> converted = calc_buf.GetWithOffset<float>(
            partial_capacity, work_offset + half_bytes);
        Duplicate(partial, static_cast<half>(0.0f), rows * partial_stride);
        PipeBarrier<PIPE_V>();

        for (uint32_t row = 0U; row < rows; ++row) {
            const uint32_t logical = head_guard + row * tail_size;
            const uint32_t aligned = logical / kHalfAlignElems * kHalfAlignElems;
            const uint32_t phase = logical - aligned;
            const uint32_t row_end = phase + tail_size;
            const uint32_t full_chunks = row_end / 128U;
            const uint32_t tail_count = row_end % 128U;
            const uint32_t dst_base = row * partial_stride;
            uint64_t head_mask[2] = {
                SegmentMask(phase, 128U, 0U),
                SegmentMask(phase, 128U, 64U)};
            WholeReduceSum<half>(
                partial[dst_base], acc[aligned], head_mask, 1, 1, 1, 0);
            // Batch the aligned interior along one row: eight 32B blocks
            // per repeat, independent of the unaligned distance between rows.
            if (full_chunks > 1U) {
                WholeReduceSum<half>(
                    partial[dst_base + 1U], acc[aligned + 128U],
                    128, static_cast<int32_t>(full_chunks - 1U), 1, 1, 8);
            }
            if (tail_count > 0U) {
                WholeReduceSum<half>(
                    partial[dst_base + full_chunks],
                    acc[aligned + full_chunks * 128U],
                    static_cast<int32_t>(tail_count), 1, 1, 1, 0);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(converted, partial, RoundMode::CAST_NONE, rows * partial_stride);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(
            sums, converted, static_cast<int32_t>(chunks),
            static_cast<int32_t>(rows), 1, 1,
            static_cast<int32_t>(partial_stride / kFloatAlignElems));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceAndStoreHalfRows(
        uint32_t output_offset, uint32_t rows)
    {
        LocalTensor<half> acc_all =
            calc_buf.Get<half>(acc_bytes / sizeof(half));
        const uint32_t work_offset =
            acc_bytes + offsets_bytes;
        LocalTensor<half> partial =
            calc_buf.GetWithOffset<half>(max_rows, work_offset);
        LocalTensor<float> sums =
            calc_buf.GetWithOffset<float>(
                max_rows, work_offset + calc_work_bytes);
        LocalTensor<float> float_partial =
            calc_buf.GetWithOffset<float>(
                max_rows, work_offset + calc_work_bytes + sum_slot_bytes);
        if (tail_size > 255U) {
            ReduceWideHalfRows(acc_all, sums, rows, work_offset);
        } else {
            Duplicate(sums, 0.0f, rows);
            PipeBarrier<PIPE_V>();
            // Sixteen dense rows share the same 32-byte phase.
            const uint32_t row_step = 16U;
            const uint32_t cohorts = MinU32(rows, 16U);
            const uint32_t chunks = (tail_size + 15U + 127U) / 128U;
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                Duplicate(partial, static_cast<half>(0.0f), rows);
                PipeBarrier<PIPE_V>();
                const uint32_t chunk_start = chunk * 128U;
                for (uint32_t cohort = 0; cohort < cohorts; ++cohort) {
                    const uint32_t logical = head_guard + cohort * tail_size;
                    const uint32_t aligned = logical / 16U * 16U;
                    const uint32_t phase = logical - aligned;
                    const uint32_t row_end = phase + tail_size;
                    if (chunk_start >= row_end) {
                        continue;
                    }
                    const uint32_t lo = phase > chunk_start ? phase - chunk_start : 0U;
                    const uint32_t hi = MinU32(128U, row_end - chunk_start);
                    uint64_t mask[2] = {0, 0};
                    for (uint32_t word = 0; word < 2U; ++word) {
                        const uint32_t base = word * 64U;
                        const uint32_t begin = lo > base ? lo - base : 0U;
                        const uint32_t end = MinU32(64U, hi > base ? hi - base : 0U);
                        if (begin < end) {
                            const uint64_t upper = end == 64U ? ~0ULL : (1ULL << end) - 1ULL;
                            mask[word] = upper ^ ((1ULL << begin) - 1ULL);
                        }
                    }
                    const uint32_t repeats =
                        (rows - 1U - cohort) / row_step + 1U;
                    WholeReduceSum<half>(
                        partial[cohort], acc_all[aligned + chunk_start], mask,
                        static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(row_step), 1,
                        static_cast<uint8_t>(tail_size));
                }
                PipeBarrier<PIPE_V>();
                Cast(float_partial, partial, RoundMode::CAST_NONE, rows);
                PipeBarrier<PIPE_V>();
                Add(sums, sums, float_partial, rows);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<half> out =
            output_queue.AllocTensor<half>();
        Cast(out, sums, RoundMode::CAST_RINT, rows);
        output_queue.EnQue(out);
        out = output_queue.DeQue<half>();
        CopyOutContiguous(
            output_gm, out, output_offset, rows);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void AccumulateFloatPrimedRows(
        uint32_t output_offset, uint32_t rows,
        bool has_next, uint32_t next_output_offset,
        uint32_t next_rows)
    {
        const uint32_t block_elems = rows * tail_size;
        LocalTensor<float> acc_all =
            calc_buf.Get<float>(acc_bytes / sizeof(float));
        InitializeFloatGuards(acc_all, block_elems);
        LocalTensor<float> acc = acc_all[head_guard];

        const uint32_t packet_count =
            (d0_size + kD0Batch - 1U) / kD0Batch;
        const uint32_t prefetch_count =
            MinU32(BUFFER_NUM, packet_count);
        uint32_t next_d0_start =
            prefetch_count * kD0Batch;
        bool initialize = true;
        for (uint32_t packet = 0U;
             packet < packet_count; ++packet) {
            if (has_next && packet + 1U == packet_count) {
                IssueInputPacket(
                    next_output_offset, next_rows, 0U);
            }
            const uint32_t batch = MinU32(
                kD0Batch, d0_size - packet * kD0Batch);
            const uint32_t slot = input_dequeue_slot;
            LocalTensor<half> &current = DequeueInput();
            uint32_t first_offset = 0U;
            uint32_t second_offset = 0U;
            LoadInputOffsets(
                slot, first_offset, second_offset);
            if (initialize) {
                InitializeFloatBatch(
                    acc, current, first_offset,
                    second_offset,
                    block_elems, batch);
                initialize = false;
            } else {
                AccumulateFloatBatch(
                    acc, current, first_offset,
                    second_offset,
                    block_elems, batch);
            }
            input_queue.FreeTensor(current);
            if (next_d0_start < d0_size) {
                IssueInputPacket(
                    output_offset, rows, next_d0_start);
                next_d0_start += kD0Batch;
            }
        }
    }

    __aicore__ inline void ReduceAndStoreFloatRows(
        uint32_t output_offset, uint32_t rows,
        LocalTensor<uint32_t> offsets)
    {
        LocalTensor<float> acc_all =
            calc_buf.Get<float>(acc_bytes / sizeof(float));
        ReduceFloatAlignedRows(acc_all, offsets, rows);
        LocalTensor<float> sums =
            calc_buf.GetWithOffset<float>(
                max_rows,
                acc_bytes + offsets_bytes + kFoldBytes);
        LocalTensor<half> out =
            output_queue.AllocTensor<half>();
        Cast(out, sums, RoundMode::CAST_RINT, rows);
        output_queue.EnQue(out);
        out = output_queue.DeQue<half>();
        CopyOutContiguous(
            output_gm, out, output_offset, rows);
        output_queue.FreeTensor(out);
    }

    static constexpr uint32_t kD0Batch = 2;
    static constexpr uint32_t kInputGuardBytes = 256;
    static constexpr uint32_t kCycleRows = 16;
    static constexpr uint32_t kFoldBytes =
        kCycleRows * MAX_FP32_REPEAT_MASK * sizeof(float);
    static constexpr uint32_t kTileRowAlign = 16;
    static constexpr uint32_t kFloatAlignElems =
        UB_ALIGN_BYTES / sizeof(float);
    static constexpr uint32_t kHalfAlignElems =
        UB_ALIGN_BYTES / sizeof(half);
    static constexpr uint32_t kLayoutHalf = 2;
    static constexpr uint32_t kLayoutHalfShort = 3;
    static constexpr uint32_t kLayoutGm512 = 1U << 8;
    static constexpr uint32_t kGmAlignBytes = 512;
    static constexpr uint32_t kGm512InputGuardBytes = 1024;

    // Depth zero keeps events attached to the two InitBuffer slots. The free
    // slot beside a tile's final packet starts the next tile before reduction.
    TQue<TPosition::VECIN, 0> input_queue;
    LocalTensor<half> input_slots[BUFFER_NUM];
    uint32_t input_first_offset_slot0 = 0U;
    uint32_t input_second_offset_slot0 = 0U;
    uint32_t input_first_offset_slot1 = 0U;
    uint32_t input_second_offset_slot1 = 0U;
    uint32_t input_alloc_slot = 0U;
    uint32_t input_dequeue_slot = 0U;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<half> input_gm;
    GlobalTensor<half> output_gm;
    uint32_t output_size = 0;
    uint32_t total_size = 0;
    uint32_t d0_size = 0;
    uint32_t tail_size = 0;
    uint32_t d0_stride = 0;
    bool use_half_acc = false;
    bool use_gm512 = false;
    uint32_t max_rows = kTileRowAlign;
    uint32_t window_stride = 1;
    uint32_t window_elems = 1;
    uint32_t extra_count = 1;
    uint32_t extra_stride = 1;
    uint32_t head_guard = 0;
    uint32_t tail_guard = 0;
    uint32_t dense_capacity = 1;
    uint32_t input_capacity = 1;
    uint32_t input_slot_bytes = 0;
    uint32_t acc_count = 1;
    uint32_t acc_bytes = 0;
    uint32_t offset_count = 1;
    uint32_t offsets_bytes = 0;
    uint32_t calc_work_bytes = 0;
    uint32_t sum_slot_bytes = 0;
};

template <typename T>
class KernelSeparatedFast {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        output_size = tiling.output_size;
        tile_elems = tiling.tile_elems;
        axis_mask = tiling.axis_mask;
        reduce_axis0 = tiling.reduce_axis0;
        reduce_axis1 = tiling.reduce_axis1;
        for (uint32_t i = 0; i < 8; ++i) {
            dims[i] = tiling.dims[i];
            strides[i] = tiling.strides[i];
        }
        tail_reduce_size = dims[3];
        inner_keep_size = dims[2] == 0 ? 1 : dims[2];
        if (IsReducedAxis(axis_mask, 0)) {
            reduce_dim = 0;
        } else if (IsReducedAxis(axis_mask, 1)) {
            reduce_dim = 1;
        } else {
            reduce_dim = 2;
        }
        reduced_count = dims[reduce_dim];
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        uint32_t in_bytes = AlignUp32(tile_elems * sizeof(T));
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));

        pipe.InitBuffer(input_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(output_queue, BUFFER_NUM, in_bytes);
        pipe.InitBuffer(calc_buf, 3 * float_bytes + 256);
    }

    __aicore__ inline void Process()
    {
        if (CanProcessFastSeparated()) {
            ProcessFastSeparated();
            return;
        }
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        if (!CanProcessOutputBlock()) {
            for (uint32_t out_idx = start; out_idx < end; ++out_idx) {
                float acc = ReduceOneOutputScalar(out_idx);
                WriteScalarOutput(out_idx, acc);
            }
            return;
        }

        uint32_t max_rows = MaxPaddedRows();
        for (uint32_t out_offset = start; out_offset < end;) {
            uint32_t group_idx = out_offset / inner_keep_size;
            uint32_t inner_offset = out_offset - group_idx * inner_keep_size;
            uint32_t rows = MinU32(
                max_rows,
                MinU32(inner_keep_size - inner_offset,
                       end - out_offset));
            ProcessOutputBlock(group_idx, inner_offset, rows, out_offset);
            out_offset += rows;
        }
    }

private:
    __aicore__ inline bool CanProcessOutputBlock()
    {
        return reduce_dim < 2 && tail_reduce_size > 0 && tail_reduce_size <= tile_elems && inner_keep_size > 0 &&
               MaxPaddedRows() > 0;
    }

    __aicore__ inline uint32_t MaxPaddedRows()
    {
        if (PaddedStrideBlocks<T>(tail_reduce_size) > 255 || PaddedStrideBlocks<float>(tail_reduce_size) > 255) {
            return 0;
        }
        uint32_t input_stride = PaddedStrideElems<T>(tail_reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);
        if (input_stride == 0 || float_stride == 0) {
            return 0;
        }
        uint32_t rows = MinU32(tile_elems / input_stride, tile_elems / float_stride);
        return MinU32(rows, MAX_REPEAT_ROWS);
    }

    __aicore__ inline uint32_t BlockBase(uint32_t group_idx, uint32_t inner_offset, uint32_t reduce_coord)
    {
        if (reduce_dim == 0) {
            return reduce_coord * strides[0] + group_idx * strides[1] + inner_offset * strides[2];
        }
        return group_idx * strides[0] + reduce_coord * strides[1] + inner_offset * strides[2];
    }

    __aicore__ inline bool CanProcessStridedBlocks(uint32_t rows)
    {
        return reduce_dim == 0 && tail_reduce_size > 0 && (tail_reduce_size % 16U) == 0 &&
               rows > 0 && rows * tail_reduce_size <= tile_elems;
    }

    __aicore__ inline void ProcessOutputBlockStrided(uint32_t group_idx, uint32_t inner_offset, uint32_t rows,
                                                     uint32_t output_offset)
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> square = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> tmp_work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        Duplicate(acc, 0.0f, rows);

        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);
        uint32_t total_acc_elems = rows * float_stride;

        Duplicate(square, 0.0f, total_acc_elems);

        uint32_t base = output_offset * tail_reduce_size;
        uint32_t block_elems = rows * tail_reduce_size;
        uint32_t block_bytes = block_elems * sizeof(T);
        uint32_t gap_bytes = (strides[0] - block_elems) * sizeof(T);

        // Batch multiple d0 steps per 2D DMA call
        uint32_t batch = tile_elems / block_elems;
        if (batch == 0) batch = 1;
        if (batch > reduced_count) batch = reduced_count;
        bool can_2d = (block_bytes % 32 == 0) && (gap_bytes % 32 == 0) &&
                      ((base * sizeof(T)) % 32 == 0) && (block_bytes / 32 <= 65535) &&
                      (gap_bytes / 32 <= 65535);

        // Double-buffered batched pipeline
        uint32_t r_base = 0;
        uint32_t prev_batch = 0;
        if (r_base < reduced_count) {
            prev_batch = MinU32(batch, reduced_count - r_base);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            if (can_2d && prev_batch > 1) {
                DataCopyParams params{static_cast<uint16_t>(prev_batch),
                                      static_cast<uint16_t>(block_bytes / 32),
                                      static_cast<uint16_t>(gap_bytes / 32), 0};
                DataCopy(input_local, input_gm[base + r_base * strides[0]], params);
            } else {
                #pragma unroll
                for (uint32_t b = 0; b < prev_batch; ++b) {
                    CopyInContiguous(input_local[b * block_elems], input_gm,
                                     base + (r_base + b) * strides[0], block_elems);
                }
            }
            input_queue.EnQue(input_local);
            r_base += prev_batch;
        }

        while (r_base < reduced_count) {
            uint32_t cur_batch = MinU32(batch, reduced_count - r_base);
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            if (can_2d && cur_batch > 1) {
                DataCopyParams params{static_cast<uint16_t>(cur_batch),
                                      static_cast<uint16_t>(block_bytes / 32),
                                      static_cast<uint16_t>(gap_bytes / 32), 0};
                DataCopy(input_local_next, input_gm[base + r_base * strides[0]], params);
            } else {
                #pragma unroll
                for (uint32_t b = 0; b < cur_batch; ++b) {
                    CopyInContiguous(input_local_next[b * block_elems], input_gm,
                                     base + (r_base + b) * strides[0], block_elems);
                }
            }
            input_queue.EnQue(input_local_next);

            // Process previous batch (overlapped with next DMA)
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            #pragma unroll
            for (uint32_t b = 0; b < prev_batch; ++b) {
                SquareAccumulatePaddedRowsToFloat(square, input_local_curr[b * block_elems], tmp_work,
                                                  tail_reduce_size, rows, tail_reduce_size, float_stride);
            }
            input_queue.FreeTensor(input_local_curr);
            r_base += cur_batch;
            prev_batch = cur_batch;
        }

        if (prev_batch > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            #pragma unroll
            for (uint32_t b = 0; b < prev_batch; ++b) {
                SquareAccumulatePaddedRowsToFloat(square, input_local_curr[b * block_elems], tmp_work,
                                                  tail_reduce_size, rows, tail_reduce_size, float_stride);
            }
            input_queue.FreeTensor(input_local_curr);
        }

        AccumulateTailRows(acc, square, tmp_work, sum, tail_reduce_size, rows, float_stride);
        StoreOutputTile(output_offset, acc, rows);
    }

    __aicore__ inline void ProcessOutputBlock(uint32_t group_idx, uint32_t inner_offset, uint32_t rows,
                                              uint32_t output_offset)
    {
        if (CanProcessStridedBlocks(rows)) {
            ProcessOutputBlockStrided(group_idx, inner_offset, rows, output_offset);
            return;
        }
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> square = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> tmp_work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        Duplicate(acc, 0.0f, rows);

        uint32_t input_stride = PaddedStrideElems<T>(tail_reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);
        uint32_t total_acc_elems = rows * float_stride;

        // Accumulate x² across ALL reduce steps, then do ONE final reduction.
        // This replaces reduced_count × WholeReduceSum with 1 × WholeReduceSum.
        Duplicate(square, 0.0f, total_acc_elems);

        uint32_t head_skip_curr = 0;
        uint32_t head_skip_next = 0;
        uint32_t r = 0;
        if (r < reduced_count) {
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            head_skip_curr = CopyInPaddedRowsAlignedGM(input_local, input_gm, BlockBase(group_idx, inner_offset, r), tail_reduce_size, rows);
            input_queue.EnQue(input_local);
            r++;
        }

        while (r < reduced_count) {
            LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
            head_skip_next = CopyInPaddedRowsAlignedGM(input_local_next, input_gm, BlockBase(group_idx, inner_offset, r), tail_reduce_size, rows);
            input_queue.EnQue(input_local_next);

            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            SquareAccumulatePaddedRowsToFloat(square, input_local_curr[head_skip_curr], tmp_work, tail_reduce_size, rows, input_stride,
                                              float_stride);
            input_queue.FreeTensor(input_local_curr);
            head_skip_curr = head_skip_next;
            r++;
        }

        if (reduced_count > 0) {
            LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
            SquareAccumulatePaddedRowsToFloat(square, input_local_curr[head_skip_curr], tmp_work, tail_reduce_size, rows, input_stride,
                                              float_stride);
            input_queue.FreeTensor(input_local_curr);
        }

        // Single final reduction: square (rows × float_stride) → acc (rows)
        Duplicate(acc, 0.0f, rows);
        AccumulateTailRows(acc, square, tmp_work, sum, tail_reduce_size, rows, float_stride);
        StoreOutputTile(output_offset, acc, rows);
    }

    __aicore__ inline void AccumulateTailRows(LocalTensor<float> acc, LocalTensor<float> square,
                                              LocalTensor<float> row_sums, LocalTensor<float> sum,
                                              uint32_t tail_elems, uint32_t rows, uint32_t float_stride)
    {
        uint32_t float_stride_blocks = (float_stride * sizeof(float)) / UB_ALIGN_BYTES;
        if (tail_elems <= MAX_FP32_REPEAT_MASK && rows <= MAX_REPEAT_ROWS && float_stride_blocks <= 255) {
            WholeReduceSum<float>(row_sums, square, static_cast<int32_t>(tail_elems),
                                  static_cast<int32_t>(rows), 1, 1,
                                  static_cast<int32_t>(float_stride_blocks));
            Add(acc, acc, row_sums, rows);
            return;
        }
        uint32_t row_sums2_offset = AlignUp32(rows * sizeof(float)) / sizeof(float);
        if (tail_elems <= 4 * MAX_FP32_REPEAT_MASK && rows <= MAX_REPEAT_ROWS &&
            row_sums2_offset + rows <= tile_elems && float_stride_blocks <= 255) {
            uint32_t first_tail = MinU32(MAX_FP32_REPEAT_MASK, tail_elems);
            LocalTensor<float> row_sums2 = row_sums[row_sums2_offset];
            WholeReduceSum<float>(row_sums, square, static_cast<int32_t>(first_tail),
                                  static_cast<int32_t>(rows), 1, 1,
                                  static_cast<int32_t>(float_stride_blocks));
            for (uint32_t tail_offset = MAX_FP32_REPEAT_MASK; tail_offset < tail_elems;
                 tail_offset += MAX_FP32_REPEAT_MASK) {
                uint32_t cur_tail = MinU32(MAX_FP32_REPEAT_MASK, tail_elems - tail_offset);
                WholeReduceSum<float>(row_sums2, square[tail_offset], static_cast<int32_t>(cur_tail),
                                      static_cast<int32_t>(rows), 1, 1,
                                      static_cast<int32_t>(float_stride_blocks));
                Add(row_sums, row_sums, row_sums2, rows);
            }
            Add(acc, acc, row_sums, rows);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            LocalTensor<float> square_row = square[row * float_stride];
            float part = ReduceFloatTile(square_row, row_sums, sum, tail_elems);
            acc.SetValue(row, acc.GetValue(row) + part);
        }
    }

    __aicore__ inline float ReduceContiguousSegment(uint32_t base, uint32_t count)
    {
        float acc = 0.0f;
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> square = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp = calc_buf.GetWithOffset<float>(tile_elems, float_bytes);
        LocalTensor<float> work = calc_buf.GetWithOffset<float>(tile_elems, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(PARTIAL_STRIDE, 3 * float_bytes);
        for (uint32_t offset = 0; offset < count;) {
            uint32_t cur = MinU32(tile_elems, count - offset);
            LocalTensor<T> input_local = input_queue.AllocTensor<T>();
            CopyInContiguous(input_local, input_gm, base + offset, cur);
            input_queue.EnQue(input_local);
            input_local = input_queue.DeQue<T>();
            SquareToFloat(square, input_local, tmp, cur);
            acc += ReduceFloatTile(square, work, sum, cur);
            input_queue.FreeTensor(input_local);
            offset += cur;
        }
        return acc;
    }

    __aicore__ inline float ReduceOneOutputScalar(uint32_t out_idx)
    {
        uint32_t coords[4] = {0, 0, 0, 0};
        uint32_t rem = out_idx;
        for (int32_t d = 3; d >= 0; --d) {
            if (!IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                coords[d] = rem % dims[d];
                rem /= dims[d];
            }
        }
        uint32_t other_axis = reduce_axis0 == 3 ? reduce_axis1 : reduce_axis0;
        float acc = 0.0f;
        for (uint32_t r = 0; r < dims[other_axis]; ++r) {
            coords[other_axis] = r;
            coords[3] = 0;
            uint32_t base = coords[0] * strides[0] + coords[1] * strides[1] + coords[2] * strides[2];
            acc += ReduceContiguousSegment(base, dims[3]);
        }
        return acc;
    }

    __aicore__ inline void StoreOutputTile(uint32_t offset, LocalTensor<float> acc, uint32_t count)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(out, acc, 0.0f, count);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, count);
        }
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        CopyOutContiguous(output_gm, out, offset, count);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline void WriteScalarOutput(uint32_t offset, float value)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        out.SetValue(0, FromFloatScalar<T>(value));
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(output_gm[offset], out, params);
        output_queue.FreeTensor(out);
    }

    __aicore__ inline bool CanProcessFastSeparated()
    {
        return tail_reduce_size == 128U && reduced_count > 0U &&
               inner_keep_size > 0U && dims[0] > 0U && reduce_dim == 1U;
    }

    __aicore__ inline void ProcessFastSeparated()
    {
        uint32_t float_bytes = AlignUp32(tile_elems * sizeof(float));
        LocalTensor<float> acc = calc_buf.Get<float>(tile_elems);
        LocalTensor<float> tmp_work = calc_buf.GetWithOffset<float>(512, 1024);
        LocalTensor<float> acc_square = calc_buf.GetWithOffset<float>(512, float_bytes);
        LocalTensor<float> square = calc_buf.GetWithOffset<float>(512, 2 * float_bytes);
        LocalTensor<float> sum = calc_buf.GetWithOffset<float>(256, 2 * float_bytes + 8192);

        uint32_t input_stride = PaddedStrideElems<T>(tail_reduce_size);
        uint32_t float_stride = PaddedStrideElems<float>(tail_reduce_size);
        constexpr uint32_t col_group = 3U;
        const uint32_t col_group_count =
            (inner_keep_size + col_group - 1U) / col_group;
        const uint32_t task_count = dims[0] * col_group_count;
        for (uint32_t task_idx = GetBlockIdx(); task_idx < task_count; task_idx += GetBlockNum()) {
            uint32_t outer_idx = task_idx / col_group_count;
            uint32_t col_group_idx = task_idx - outer_idx * col_group_count;
            uint32_t col_start = col_group_idx * col_group;
            uint32_t col_len = MinU32(col_group, inner_keep_size - col_start);
            uint32_t total_acc_elems = col_len * float_stride;
            Duplicate(acc, 0.0f, col_len);
            Duplicate(acc_square, 0.0f, total_acc_elems);

            uint32_t base_offset = outer_idx * strides[0] + col_start * 128;
            uint32_t block_elems = col_len * 128;

            uint32_t r_base = 0;
            if (r_base < reduced_count) {
                LocalTensor<T> input_local = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local, input_gm, base_offset + r_base * strides[1], block_elems);
                input_queue.EnQue(input_local);
                r_base++;
            }

            while (r_base < reduced_count) {
                LocalTensor<T> input_local_next = input_queue.AllocTensor<T>();
                CopyInContiguous(input_local_next, input_gm, base_offset + r_base * strides[1], block_elems);
                input_queue.EnQue(input_local_next);

                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquarePaddedRowsToFloat(square, input_local_curr, tmp_work, tail_reduce_size, col_len, input_stride, float_stride);
                Add(acc_square, acc_square, square, total_acc_elems);
                input_queue.FreeTensor(input_local_curr);

                r_base++;
            }

            if (r_base > 0) {
                LocalTensor<T> input_local_curr = input_queue.DeQue<T>();
                SquarePaddedRowsToFloat(square, input_local_curr, tmp_work, tail_reduce_size, col_len, input_stride, float_stride);
                Add(acc_square, acc_square, square, total_acc_elems);
                input_queue.FreeTensor(input_local_curr);
            }

            AccumulateTailRows(acc, acc_square, tmp_work, sum, tail_reduce_size, col_len, float_stride);
            StoreOutputTile(outer_idx * inner_keep_size + col_start, acc, col_len);
        }
    }

    TQue<TPosition::VECIN, BUFFER_NUM> input_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t output_size = 0;
    uint32_t tile_elems = 0;
    uint32_t axis_mask = 0;
    uint32_t reduce_axis0 = 0;
    uint32_t reduce_axis1 = 0;
    uint32_t reduce_dim = 0;
    uint32_t reduced_count = 1;
    uint32_t tail_reduce_size = 1;
    uint32_t inner_keep_size = 1;
    uint32_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t strides[8] = {1, 1, 1, 1, 1, 1, 1, 1};
};

template <typename T>
class KernelSeparatedFallback {
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, const SquareSumV1TilingData &tiling)
    {
        rank = tiling.rank;
        axis_mask = tiling.axis_mask;
        reduce_size = tiling.reduce_size;
        output_size = tiling.output_size;
        for (uint32_t i = 0; i < 8; ++i) {
            dims[i] = tiling.dims[i];
            strides[i] = tiling.strides[i];
        }
        input_gm.SetGlobalBuffer((__gm__ T *)input, tiling.total_size);
        output_gm.SetGlobalBuffer((__gm__ T *)output, output_size);
        pipe.InitBuffer(output_queue, BUFFER_NUM, AlignUp32(PARTIAL_STRIDE * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        uint32_t start = 0;
        uint32_t end = 0;
        GetBlockRange(output_size, start, end);
        for (uint32_t out_idx = start; out_idx < end; ++out_idx) {
            uint32_t base = InputBaseFromOutput(out_idx);
            float acc = 0.0f;
            for (uint32_t r = 0; r < reduce_size; ++r) {
                uint32_t input_offset = base + ReduceOffset(r);
                float value = ToFloatScalar(input_gm.GetValue(input_offset));
                acc += value * value;
            }
            WriteScalarOutput(out_idx, acc);
        }
    }

private:
    __aicore__ inline uint32_t InputBaseFromOutput(uint32_t out_idx)
    {
        uint32_t base = 0;
        uint32_t rem = out_idx;
        for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
            if (!IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                base += coord * strides[d];
            }
        }
        return base;
    }

    __aicore__ inline uint32_t ReduceOffset(uint32_t reduce_idx)
    {
        uint32_t offset = 0;
        uint32_t rem = reduce_idx;
        for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
            if (IsReducedAxis(axis_mask, static_cast<uint32_t>(d))) {
                uint32_t coord = rem % dims[d];
                rem /= dims[d];
                offset += coord * strides[d];
            }
        }
        return offset;
    }

    __aicore__ inline void WriteScalarOutput(uint32_t offset, float value)
    {
        LocalTensor<T> out = output_queue.AllocTensor<T>();
        out.SetValue(0, FromFloatScalar<T>(value));
        output_queue.EnQue(out);
        out = output_queue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(output_gm[offset], out, params);
        output_queue.FreeTensor(out);
    }

    TQue<TPosition::VECOUT, BUFFER_NUM> output_queue;
    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;
    uint32_t rank = 0;
    uint32_t axis_mask = 0;
    uint32_t reduce_size = 0;
    uint32_t output_size = 0;
    uint32_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t strides[8] = {1, 1, 1, 1, 1, 1, 1, 1};
};

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(18, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(19, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(20, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(21, KERNEL_TYPE_MIX_AIV_1_0);

    TPipe pipe;
    if (TILING_KEY_IS(1)) {  // SquareOnly
        KernelSquareOnly<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {  // AllReduce
        KernelAllReduce<DTYPE_INPUT> op;
        op.Init(pipe, input, output, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(3)) {  // TailReduce
        KernelTailReduce<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(4)) {  // HeadReduce
        KernelHeadReduce<DTYPE_INPUT> op;
        op.Init(pipe, input, output, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(5)) {  // MiddleReduceAligned
        KernelMiddleReduceAligned<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(6)) {  // MiddleReduceFallback
        KernelMiddleReduceFallback<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(7)) {  // MiddleReduceOuterBlock
        KernelMiddleReduceOuterBlock<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(8)) {  // SeparatedFast
        KernelSeparatedFast<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(9)) {  // SeparatedFallback
        KernelSeparatedFallback<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(10)) {  // SeparatedInnerVector
        KernelSeparatedInnerVector<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(11)) {  // SeparatedTailVector
        KernelSeparatedTailVector<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(12)) {  // GroupedTailReduce
        KernelGroupedTailReduce<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(13)) {  // Bf16SmallTailReduce
        if constexpr (std::is_same<DTYPE_INPUT, bfloat16_t>::value) {
            KernelBf16SmallTailReduce op;
            op.Init(pipe, input, output, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(14)) {  // Bf16TailDenseGatherSlab
        if constexpr (std::is_same<DTYPE_INPUT, bfloat16_t>::value) {
            KernelBf16TailDenseGatherSlab op;
            op.Init(pipe, input, output, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(15)) {  // Bf16MiddleOuterBanked
        KernelBf16MiddleOuterBanked<DTYPE_INPUT> op;
        op.Init(pipe, input, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(16)) {  // Bf16MiddlePhaseRepeat
        if constexpr (std::is_same<DTYPE_INPUT, bfloat16_t>::value) {
            KernelBf16MiddlePhaseRepeat op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(17)) {  // HeadFp32OffsetPrefix
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelHeadFp32OffsetPrefix op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(18)) {  // HeadFp32Width518Phase
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelHeadFp32Width518Phase op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(19)) {  // HeadFp32Width519Reverse
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelHeadFp32Width519Reverse op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(20)) {  // HeadFp32Width520Aligned
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelHeadFp32Width520Aligned op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(21)) {  // HeadFp32Width521Forward
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelHeadFp32Width521Forward op;
            op.Init(pipe, input, output, workspace, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(22)) {  // SeparatedFp16BalancedFoldedTail
        if constexpr (std::is_same<DTYPE_INPUT, half>::value) {
            KernelSeparatedFp16BalancedFoldedTail op;
            op.Init(pipe, input, output, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(23)) {  // SeparatedFp16WaveBalancedFoldedTail
        if constexpr (std::is_same<DTYPE_INPUT, half>::value) {
            KernelSeparatedFp16WaveBalancedFoldedTail op;
            op.Init(pipe, input, output, tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(24)) {  // SeparatedFp16WaveAlignedWindow
        if constexpr (std::is_same<DTYPE_INPUT, half>::value) {
            KernelSeparatedFp16WaveAlignedWindow op;
            op.Init(pipe, input, output, tiling_data);
            op.Process();
        }
    }
}
