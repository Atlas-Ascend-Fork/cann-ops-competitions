#include "kernel_operator.h"
#include "../op_host/square_sum_v1_tiling.h"
#include <type_traits>

namespace s9squaresum {

template<typename T> __aicore__ inline void S9PackedCopy(const AscendC::LocalTensor<T>& dst,const AscendC::GlobalTensor<T>& src,uint64_t offset,uint32_t count,bool preferRegular){
 const uint64_t addr=reinterpret_cast<uint64_t>(src.GetPhyAddr(offset));const uint32_t bytes=count*sizeof(T);
 if(preferRegular && ((addr|bytes)&31ULL)==0){AscendC::DataCopy(dst,src[offset],count);}
 else{AscendC::DataCopyExtParams cp{1,bytes,0,0,0};AscendC::DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};AscendC::DataCopyPad(dst,src[offset],cp,pad);}
}

constexpr uint64_t kLongArPartialWidth = 64;
constexpr uint64_t kLongArRowsPerBatch = 64;
constexpr uint64_t kLongArBatchElements = kLongArPartialWidth * kLongArRowsPerBatch;
// Long AR rows in the published range are often much wider than the short
// one-instruction reduction but still fit several complete rows in UB.  Keep
// at most 240 first-stage reductions in one call (below the 255-repeat ISA
// limit), and align every logical row to eight 64-element partials so the
// second reduction can stride between rows exactly in 32-byte units.
constexpr uint64_t kLongArGroupedRowAlignment = 8 * kLongArPartialWidth;
constexpr uint64_t kLongArGroupedBufferElements = 240 * kLongArPartialWidth;
constexpr uint64_t kLongArGroupedMinReduceElements = kLongArGroupedRowAlignment;
constexpr uint64_t kLongArGroupedMaxReduceElements = kLongArBatchElements;
constexpr uint64_t kLongArMaxGroupedRows =
    kLongArGroupedBufferElements / kLongArGroupedRowAlignment;

__aicore__ inline uint64_t AlignUpElements(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
class KernelSquareSumArVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(rawBuffer_, data_.rowTile * data_.reduceAligned * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, data_.rowTile * data_.reduceAligned * sizeof(float));
        }
        pipe_.InitBuffer(reductionBuffer_, AlignUpElements(data_.rowTile, 8) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.outputElements, start, count);
        for (uint64_t begin = start; begin < start + count; begin += data_.rowTile) {
            const uint64_t rows = start + count - begin < data_.rowTile
                                      ? start + count - begin
                                      : data_.rowTile;
            ProcessRows(begin, static_cast<uint32_t>(rows));
        }
    }

private:
    __aicore__ inline void Partition(uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        if (data_.reserved0 == 2 && data_.tileElements != 0) {
            const uint64_t units =
                (total + data_.tileElements - 1) / data_.tileElements;
            const uint64_t smallUnits = units / blocks;
            const uint64_t remainderUnits = units % blocks;
            const uint64_t unitCount =
                smallUnits + (block < remainderUnits ? 1 : 0);
            const uint64_t unitStart =
                block * smallUnits + (block < remainderUnits ? block : remainderUnits);
            start = unitStart * data_.tileElements;
            const uint64_t end = start + unitCount * data_.tileElements;
            count = (end < total ? end : total) - start;
            return;
        }
        const uint64_t small = total / blocks;
        const uint64_t remainder = total % blocks;
        count = small + (block < remainder ? 1 : 0);
        start = block * small + (block < remainder ? block : remainder);
    }

    __aicore__ inline void ProcessRows(uint64_t firstRow, uint32_t rows)
    {
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(data_.reduce * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(data_.reduceAligned - data_.reduce), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, xGm_[firstRow * data_.reduce], inputParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> values;
        if constexpr (std::is_same<T, float>::value) {
            values = raw.template ReinterpretCast<float>();
        } else {
            values = floatBuffer_.Get<float>();
            AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, rows * data_.reduceAligned);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(values, values, values, rows * data_.reduceAligned);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> reduced = reductionBuffer_.Get<float>();
        AscendC::WholeReduceSum<float>(reduced, values, static_cast<int32_t>(data_.reduceAligned),
            static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(data_.reduceAligned / 8));
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(yGm_[firstRow], reduced, outputParams);
        } else {
            AscendC::Cast(raw, reduced, AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(yGm_[firstRow], raw, outputParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reductionBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

template <typename T, bool kMedium = false>
class KernelSquareSumLongArVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(rawBuffer_, kLongArGroupedBufferElements * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, kLongArGroupedBufferElements * sizeof(float));
        }
        pipe_.InitBuffer(partialBuffer_,
            (kLongArGroupedBufferElements / kLongArPartialWidth) * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, AlignUpElements(kLongArMaxGroupedRows, 8) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.outputElements, start, count);
        if constexpr (kMedium) { ProcessGroupedRows(start, count); return; }
        if (static_cast<Mode>(data_.mode) == Mode::kAR &&
            data_.reduce >= kLongArGroupedMinReduceElements &&
            data_.reduce <= kLongArGroupedMaxReduceElements) {
            ProcessGroupedRows(start, count);
            return;
        }
        for (uint64_t output = start; output < start + count; ++output) {
            ProcessRow(output);
        }
    }

private:
    __aicore__ inline void Partition(uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        if constexpr (kMedium) {
            const uint64_t unit = 32 / sizeof(T);
            const uint64_t units = (total + unit - 1) / unit;
            start = units * block / blocks * unit;
            const uint64_t rawEnd = units * (block + 1) / blocks * unit;
            count = (rawEnd < total ? rawEnd : total) - start;
            return;
        }
        const uint64_t small = total / blocks;
        const uint64_t remainder = total % blocks;
        count = small + (block < remainder ? 1 : 0);
        start = block * small + (block < remainder ? block : remainder);
    }

    __aicore__ inline void ZeroRaw(const AscendC::LocalTensor<T>& raw, uint64_t count)
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::LocalTensor<uint16_t> bits = raw.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(bits, static_cast<uint16_t>(0), static_cast<int32_t>(count));
        } else {
            AscendC::Duplicate(raw, static_cast<T>(0), static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void ProcessGroupedRows(uint64_t start, uint64_t count)
    {
        const uint64_t rowAligned =
            AlignUpElements(data_.reduce, kLongArGroupedRowAlignment);
        const uint64_t partialsPerRow = rowAligned / kLongArPartialWidth;
        const uint64_t rowsPerGroup = kLongArGroupedBufferElements / rowAligned;
        for (uint64_t firstRow = start; firstRow < start + count;
             firstRow += rowsPerGroup) {
            const uint64_t rows = start + count - firstRow < rowsPerGroup
                                      ? start + count - firstRow
                                      : rowsPerGroup;
            const uint64_t groupedElements = rows * rowAligned;
            AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
            ZeroRaw(raw, groupedElements);
            AscendC::PipeBarrier<PIPE_V>();

            const uint64_t dmaAlignment = 32 / sizeof(T);
            const uint64_t dmaAligned =
                AlignUpElements(data_.reduce, dmaAlignment);
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(data_.reduce * sizeof(T)),
                0,
                static_cast<uint32_t>((rowAligned - dmaAligned) * sizeof(T) / 32),
                0};
            AscendC::DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(dmaAligned - data_.reduce),
                static_cast<T>(0)};
            if constexpr (kMedium) SyncMedium<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPad(
                raw, xGm_[firstRow * data_.reduce], inputParams, padParams);
            if constexpr (kMedium) SyncMedium<AscendC::HardEvent::MTE2_V>();
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
            } else {
                values = floatBuffer_.Get<float>();
                if constexpr (kMedium && std::is_same<T, half>::value) {
                    AscendC::Mul(raw, raw, raw, static_cast<int32_t>(groupedElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(groupedElements));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (!(kMedium && std::is_same<T, half>::value)) {
                AscendC::Mul(values, values, values, static_cast<int32_t>(groupedElements));
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
            AscendC::LocalTensor<float> reduced = sumBuffer_.Get<float>();
            AscendC::WholeReduceSum<float>(
                partial, values, static_cast<int32_t>(kLongArPartialWidth),
                static_cast<int32_t>(rows * partialsPerRow), 1, 1,
                static_cast<int32_t>(kLongArPartialWidth / 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                reduced, partial, static_cast<int32_t>(partialsPerRow),
                static_cast<int32_t>(rows), 1, 1,
                static_cast<int32_t>(partialsPerRow / 8));
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                if constexpr (kMedium) SyncMedium<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[firstRow], reduced, outputParams);
            } else {
                AscendC::Cast(raw, reduced, AscendC::RoundMode::CAST_RINT,
                              static_cast<int32_t>(rows));
                AscendC::PipeBarrier<PIPE_ALL>();
                if constexpr (kMedium) SyncMedium<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[firstRow], raw, outputParams);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (kMedium) {
                SyncMedium<AscendC::HardEvent::MTE3_V>();
                SyncMedium<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
    }

    __aicore__ inline void ProcessRow(uint64_t output)
    {
        const uint64_t inputBase = output * data_.reduce;
        float total = 0.0f;
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> batchSum = sumBuffer_.Get<float>();

        uint64_t begin = 0;
        for (; begin + kLongArBatchElements <= data_.reduce;
             begin += kLongArBatchElements) {
            ZeroRaw(raw, kLongArBatchElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(kLongArBatchElements * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(
                raw, xGm_[inputBase + begin], inputParams, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
            } else {
                values = floatBuffer_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              kLongArBatchElements);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(values, values, values, kLongArBatchElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(partial, values,
                static_cast<int32_t>(kLongArPartialWidth),
                static_cast<int32_t>(kLongArRowsPerBatch),
                1, 1, static_cast<int32_t>(kLongArPartialWidth / 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(batchSum, partial,
                static_cast<int32_t>(kLongArRowsPerBatch), 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            total += batchSum.GetValue(0);
        }

        if (begin < data_.reduce) {
            const uint64_t valid = data_.reduce - begin;
            const uint64_t rows =
                (valid + kLongArPartialWidth - 1) / kLongArPartialWidth;
            const uint64_t aligned = rows * kLongArPartialWidth;
            ZeroRaw(raw, aligned);
            AscendC::PipeBarrier<PIPE_V>();

            const uint64_t copyAlignment = 32 / sizeof(T);
            const uint64_t copyAligned =
                (valid + copyAlignment - 1) / copyAlignment * copyAlignment;
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                valid % copyAlignment != 0, 0,
                static_cast<uint8_t>(copyAligned - valid), static_cast<T>(0)};
            AscendC::DataCopyPad(
                raw, xGm_[inputBase + begin], inputParams, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
            } else {
                values = floatBuffer_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(aligned));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(values, values, values, static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(partial, values,
                static_cast<int32_t>(kLongArPartialWidth), static_cast<int32_t>(rows),
                1, 1, static_cast<int32_t>(kLongArPartialWidth / 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(batchSum, partial,
                static_cast<int32_t>(rows), 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            total += batchSum.GetValue(0);
        }

        batchSum.SetValue(0, total);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams outputParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(yGm_[output], batchSum, outputParams);
        } else {
            AscendC::Cast(raw, batchSum, AscendC::RoundMode::CAST_RINT, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(yGm_[output], raw, outputParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    template <AscendC::HardEvent event> __aicore__ inline void SyncMedium() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

// A reduce-all output has only one logical row, so the ordinary AR kernel can
// never use more than one AIV. Large tensors are instead split into balanced
// contiguous ranges. Every AIV writes a cache-line-disjoint FP32 partial to
// user workspace, all AIVs rendezvous once, and core 0 performs the tiny final
// fold. The 32-byte stride is intentional: concurrent sub-block GM writes can
// otherwise turn into a read-modify-write race on the same MTE block.
template <typename T>
class KernelSquareSumReduceAllMultiCore {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
        const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        AscendC::SetSysWorkspace(workspace);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), 1);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        partialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(userWorkspace),
            AscendC::GetBlockNum() * kWorkspaceStrideFloats);

        pipe_.InitBuffer(rawBuffer_, kLongArBatchElements * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, kLongArBatchElements * sizeof(float));
        }
        pipe_.InitBuffer(partialBuffer_, kLongArRowsPerBatch * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(batchSumBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(corePartialsBuffer_,
            kMaxAivBlocks * kWorkspaceStrideFloats * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t begin = 0;
        uint64_t count = 0;
        Partition(data_.totalElements, begin, count);
        AscendC::LocalTensor<float> blockValue = sumBuffer_.Get<float>();
        ReduceRange(begin, count, blockValue);
        SyncEvent<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            partialGm_[AscendC::GetBlockIdx() * kWorkspaceStrideFloats],
            blockValue, kWorkspaceStrideFloats);
        SyncEvent<AscendC::HardEvent::MTE3_S>();

        // Host tiling launches exactly all available AIVs for this key, which
        // satisfies the hard-barrier contract on Atlas A2 / 910B.
        AscendC::SyncAll<true>();
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        const uint32_t blocks = AscendC::GetBlockNum();
        AscendC::LocalTensor<float> allPartials = corePartialsBuffer_.Get<float>();
        AscendC::DataCopy(
            allPartials, partialGm_, blocks * kWorkspaceStrideFloats);
        SyncEvent<AscendC::HardEvent::MTE2_V>();

        // Every 32-byte workspace record is zero except for lane 0. Reducing
        // the complete records is therefore equivalent to gathering lane 0,
        // and keeps the final fold entirely on the vector pipeline (scalar
        // D-cache reads from UB fault on the reference 910B-family AIV path).
        const uint32_t partialElements = blocks * kWorkspaceStrideFloats;
        const uint32_t partialRows =
            (partialElements + kLongArPartialWidth - 1) /
            kLongArPartialWidth;
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::WholeReduceSum<float>(
            partial, allPartials, static_cast<int32_t>(kLongArPartialWidth),
            static_cast<int32_t>(partialRows), 1, 1,
            static_cast<int32_t>(kLongArPartialWidth / 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(
            blockValue, partial, static_cast<int32_t>(partialRows),
            1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreResult(blockValue);
    }

private:
    template <AscendC::HardEvent event>
    __aicore__ inline void SyncEvent() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }
    static constexpr uint32_t kWorkspaceStrideFloats = 8;
    static constexpr uint32_t kMaxAivBlocks = 64;

    __aicore__ inline void Partition(
        uint64_t total, uint64_t& begin, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t base = total / blocks;
        const uint64_t extra = total % blocks;
        count = base + (block < extra ? 1 : 0);
        begin = block * base + (block < extra ? block : extra);
    }

    __aicore__ inline void ZeroRaw(
        const AscendC::LocalTensor<T>& raw, uint64_t count)
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::LocalTensor<uint16_t> bits =
                raw.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(bits, static_cast<uint16_t>(0),
                               static_cast<int32_t>(count));
        } else {
            AscendC::Duplicate(raw, static_cast<T>(0),
                               static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void ReduceRange(
        uint64_t rangeBegin, uint64_t count,
        const AscendC::LocalTensor<float>& total)
    {
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> batchSum = batchSumBuffer_.Get<float>();
        AscendC::Duplicate(total, 0.0f, kWorkspaceStrideFloats);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint64_t offset = 0; offset < count;
             offset += kLongArBatchElements) {
            const uint64_t valid = count - offset < kLongArBatchElements
                                       ? count - offset
                                       : kLongArBatchElements;
            const uint64_t rows =
                (valid + kLongArPartialWidth - 1) / kLongArPartialWidth;
            const uint64_t aligned = rows * kLongArPartialWidth;
            if (valid != kLongArBatchElements) {
                ZeroRaw(raw, aligned);
                AscendC::PipeBarrier<PIPE_V>();
            }

            SyncEvent<AscendC::HardEvent::V_MTE2>();
            const uint64_t copyAlignment = 32 / sizeof(T);
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            if (valid % copyAlignment == 0) {
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(
                    raw, xGm_[rangeBegin + offset], inputParams, padParams);
            } else {
                const uint64_t copyAligned =
                    (valid + copyAlignment - 1) / copyAlignment * copyAlignment;
                AscendC::DataCopyPadExtParams<T> padParams{
                    true, 0, static_cast<uint8_t>(copyAligned - valid), static_cast<T>(0)};
                AscendC::DataCopyPad(
                    raw, xGm_[rangeBegin + offset], inputParams, padParams);
            }
            SyncEvent<AscendC::HardEvent::MTE2_V>();

            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
            } else {
                values = floatBuffer_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(aligned));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(values, values, values, static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                partial, values, static_cast<int32_t>(kLongArPartialWidth),
                static_cast<int32_t>(rows), 1, 1,
                static_cast<int32_t>(kLongArPartialWidth / 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(batchSum, 0.0f, kWorkspaceStrideFloats);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                batchSum, partial, static_cast<int32_t>(rows), 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(total, total, batchSum,
                         static_cast<int32_t>(kWorkspaceStrideFloats));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreResult(
        const AscendC::LocalTensor<float>& result)
    {
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(yGm_[0], result, outputParams);
        } else {
            AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
            AscendC::Cast(raw, result, AscendC::RoundMode::CAST_RINT, 1);
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(yGm_[0], raw, outputParams);
        }
        SyncEvent<AscendC::HardEvent::MTE3_S>();
    }

private:
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> batchSumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> corePartialsBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<float> partialGm_;
};

template <typename T>
class KernelSquareSumSplitReduceVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        outerAxisCount_ = data_.axisCount - 1;
        while (outerAxisCount_ > 0 && data_.axes[outerAxisCount_ - 1] + 1 == data_.axes[outerAxisCount_]) {
            --outerAxisCount_;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        const uint64_t rawElements = UseGroupedOutputs() ? kGroupedRows * kLongArPartialWidth : data_.rowTile;
        pipe_.InitBuffer(rawBuffer_, rawElements * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, rawElements * sizeof(float));
        }
        pipe_.InitBuffer(partialBuffer_, (UseGroupedOutputs() ? kGroupedRows : data_.rowTile / kLongArPartialWidth) * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(outputFloatBuffer_, kOutputBatch * sizeof(float));
        pipe_.InitBuffer(outputRawBuffer_, kOutputBatch * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        rawReusable_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        inputReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        if (data_.reserved0 == 3) {
            const uint64_t outputBlockElements = 32 / sizeof(T);
            const uint64_t outputBlocks =
                (data_.outputElements + outputBlockElements - 1) /
                outputBlockElements;
            const uint64_t baseBlocks = outputBlocks / blocks;
            const uint64_t extraBlocks = outputBlocks - baseBlocks * blocks;
            const uint64_t firstBlock = block * baseBlocks +
                (block < extraBlocks ? block : extraBlocks);
            const uint64_t ownedBlocks = baseBlocks +
                (block < extraBlocks ? 1 : 0);
            const uint64_t outputBegin = firstBlock * outputBlockElements;
            const uint64_t rawOutputEnd =
                (firstBlock + ownedBlocks) * outputBlockElements;
            const uint64_t outputEnd = rawOutputEnd < data_.outputElements
                ? rawOutputEnd
                : data_.outputElements;
            if (UseGroupedOutputs()) {
                ProcessGroupedOutputs(outputBegin, outputEnd);
                return;
            }
            for (uint64_t output = outputBegin; output < outputEnd; ++output) {
                ProcessOutput(output);
            }
            FlushOutputBatch();
            return;
        }
        for (uint64_t output = block; output < data_.outputElements; output += blocks) {
            ProcessOutput(output);
        }
        FlushOutputBatch();
    }

private:
    __aicore__ inline void ZeroRaw(
        const AscendC::LocalTensor<T>& raw, uint64_t elements)
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::LocalTensor<uint16_t> bits = raw.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(bits, static_cast<uint16_t>(0), static_cast<int32_t>(elements));
        } else {
            AscendC::Duplicate(raw, static_cast<T>(0), static_cast<int32_t>(elements));
        }
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void SyncEvent()
    {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }

    __aicore__ inline void StoreOutput(uint64_t output, float value,
        const AscendC::LocalTensor<T>& raw, const AscendC::LocalTensor<float>& sum)
    {
        (void)raw; (void)sum;
        // Keep reduction order and per-output rounding unchanged. Only the
        // final adjacent stores are combined into a contiguous DMA batch.
        if (pendingOutputs_ != 0 && output != pendingOutputStart_ + pendingOutputs_) {
            FlushOutputBatch();
        }
        if (pendingOutputs_ == 0) { pendingOutputStart_ = output; }
        auto values = outputFloatBuffer_.Get<float>();
        values.SetValue(pendingOutputs_++, value);
        if (pendingOutputs_ == kOutputBatch) { FlushOutputBatch(); }
    }

    __aicore__ inline void FlushOutputBatch()
    {
        if (pendingOutputs_ == 0) { return; }
        auto values = outputFloatBuffer_.Get<float>();
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(pendingOutputs_ * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            SyncEvent<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(yGm_[pendingOutputStart_], values, params);
        } else {
            auto output = outputRawBuffer_.Get<T>();
            SyncEvent<AscendC::HardEvent::S_V>();
            AscendC::Cast(output, values, AscendC::RoundMode::CAST_RINT, pendingOutputs_);
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(yGm_[pendingOutputStart_], output, params);
        }
        SyncEvent<AscendC::HardEvent::MTE3_V>();
        SyncEvent<AscendC::HardEvent::MTE3_S>();
        pendingOutputs_ = 0;
    }

    // Batch adjacent outputs of [reduceOuter, output, reduceInner]. Each
    // vector repeat reduces one row padded to a DMA block, and FP32 vector adds
    // fold outer rows in their original order. Geometry-only dispatch.
    static constexpr uint32_t kGroupedRows = 64;

    __aicore__ inline bool UseGroupedOutputs() const
    {
        return data_.reserved0 == 3 && data_.reserved1 == 0 &&
               data_.inner != 0 && data_.inner <= kLongArPartialWidth &&
               data_.outer != 0 && data_.outer <= 16 && data_.outputElements >= 256;
    }

    __aicore__ inline void ProcessGroupedOutputs(uint64_t begin, uint64_t end)
    {
        auto raw = rawBuffer_.Get<T>();
        auto sums = partialBuffer_.Get<float>();
        auto accumulated = outputFloatBuffer_.Get<float>();
        const uint64_t dmaAligned = AlignUpElements(data_.inner, 32 / sizeof(T));
        for (uint64_t first = begin; first < end; first += kGroupedRows) {
            const uint32_t rows = static_cast<uint32_t>(end - first < kGroupedRows
                ? end - first : kGroupedRows);
            AscendC::Duplicate(accumulated, 0.0f, kGroupedRows);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t outer = 0; outer < data_.outer; ++outer) {
                ZeroRaw(raw, rows * dmaAligned);
                SyncEvent<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams inputParams{
                    static_cast<uint16_t>(rows), static_cast<uint32_t>(data_.inner * sizeof(T)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padding{
                    true, 0, static_cast<uint8_t>(dmaAligned - data_.inner), static_cast<T>(0)};
                AscendC::DataCopyPad(raw, xGm_[(outer * data_.outputElements + first) * data_.inner],
                                    inputParams, padding);
                SyncEvent<AscendC::HardEvent::MTE2_V>();
                AscendC::LocalTensor<float> values;
                if constexpr (std::is_same<T, float>::value) {
                    values = raw.template ReinterpretCast<float>();
                } else {
                    values = floatBuffer_.Get<float>();
                    if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(raw, raw, raw, rows * dmaAligned);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, rows * dmaAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values, rows * dmaAligned); }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(sums, values, static_cast<int32_t>(dmaAligned),
                    static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(dmaAligned / 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumulated, accumulated, sums, rows);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::DataCopyExtParams outputParams{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                SyncEvent<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[first], accumulated, outputParams);
            } else {
                auto output = outputRawBuffer_.Get<T>();
                AscendC::Cast(output, accumulated, AscendC::RoundMode::CAST_RINT, rows);
                SyncEvent<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[first], output, outputParams);
            }
            SyncEvent<AscendC::HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline uint64_t GeneralOutputBase(uint64_t output) const
    {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(data_.outputRank) - 1; i >= 0; --i) {
            const uint32_t axis = data_.nonReduceAxes[i];
            offset += (output % data_.shape[axis]) * data_.inputStride[axis];
            output /= data_.shape[axis];
        }
        return offset;
    }
    __aicore__ inline uint64_t GeneralOuterOffset(uint64_t outer) const
    {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(outerAxisCount_) - 1; i >= 0; --i) {
            const uint32_t axis = data_.axes[i];
            offset += (outer % data_.shape[axis]) * data_.inputStride[axis];
            outer /= data_.shape[axis];
        }
        return offset;
    }

    __aicore__ inline void ProcessOutput(uint64_t output)
    {
        if (data_.reserved1 == 1 || data_.reserved1 == 3) {
            ProcessLargeInnerOutput(output);
            return;
        }
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        const uint64_t rowBytes = data_.inner * sizeof(T);
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        const uint64_t rowsPerBatch = data_.rowTile / data_.reduceAligned;
        float total = 0.0f;
        const uint64_t generalBase = data_.reserved1 == 2 ? GeneralOutputBase(output) : 0;
        for (uint64_t outerBegin = 0; outerBegin < data_.outer;) {
            uint64_t batchRows = data_.outer - outerBegin < rowsPerBatch
                ? data_.outer - outerBegin : rowsPerBatch;
            if (data_.reserved1 == 2) {
                const uint64_t extent = data_.shape[data_.axes[outerAxisCount_ - 1]];
                const uint64_t available = extent - outerBegin % extent;
                if (batchRows > available) { batchRows = available; }
            }
            const uint64_t validElements = batchRows * data_.reduceAligned;
            const uint64_t vectorElements =
                AlignUpElements(validElements, kLongArPartialWidth);
            ZeroRaw(raw, vectorElements);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(batchRows), static_cast<uint32_t>(rowBytes),
                static_cast<uint32_t>(data_.reserved1 == 2
                    ? data_.inputStride[data_.axes[outerAxisCount_ - 1]] * sizeof(T) - rowBytes
                    : (data_.outputElements - 1) * rowBytes), 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(data_.reduceAligned - data_.inner),
                static_cast<T>(0)};
            const uint64_t inputOffset =
                data_.reserved1 == 2 ? generalBase + GeneralOuterOffset(outerBegin)
                    : (outerBegin * data_.outputElements + output) * data_.inner;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
            AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);

            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
            } else {
                values = floatBuffer_.Get<float>();
                if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(raw, raw, raw,
                              static_cast<int32_t>(vectorElements));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(vectorElements));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values,
                         static_cast<int32_t>(vectorElements)); }
            AscendC::PipeBarrier<PIPE_V>();

            const uint64_t partialCount = vectorElements / kLongArPartialWidth;
            AscendC::WholeReduceSum<float>(
                partial, values, static_cast<int32_t>(kLongArPartialWidth),
                static_cast<int32_t>(partialCount), 1, 1,
                static_cast<int32_t>(kLongArPartialWidth / 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                sum, partial, static_cast<int32_t>(partialCount), 1, 1, 1, 8);
            SyncEvent<AscendC::HardEvent::V_S>();
            total += sum.GetValue(0);
            outerBegin += batchRows;
        }

        StoreOutput(output, total, raw, sum);
    }

    __aicore__ inline void ProcessLargeInnerOutput(uint64_t output)
    {
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        float total = 0.0f;
        const uint64_t generalBase = data_.reserved1 == 3 ? GeneralOutputBase(output) : 0;

        for (uint64_t outer = 0; outer < data_.outer; ++outer) {
            for (uint64_t innerBegin = 0; innerBegin < data_.inner;
                 innerBegin += data_.rowTile) {
                const uint64_t valid = data_.inner - innerBegin < data_.rowTile
                    ? data_.inner - innerBegin
                    : data_.rowTile;
                const uint64_t vectorElements =
                    AlignUpElements(valid, kLongArPartialWidth);

                // DMA padding is limited to one 32-byte block. Clear only
                // a partial 64-lane reduction tail before copying its values.
                if (vectorElements != valid) {
                    ZeroRaw(raw, vectorElements);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // The previous vector batch may still read raw while MTE2
                // wants to reuse it.  A PIPE_ALL barrier is not a cross-pipe
                // dependency on 910B; use the exact V->MTE2 event instead.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
                AscendC::DataCopyExtParams inputParams{
                    1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{
                    true, 0, static_cast<uint8_t>(AlignUpElements(valid, 32 / sizeof(T)) - valid),
                    static_cast<T>(0)};
                const uint64_t inputOffset =
                    data_.reserved1 == 3 ? generalBase + GeneralOuterOffset(outer) + innerBegin
                        : (outer * data_.outputElements + output) * data_.inner + innerBegin;
                AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);

                AscendC::LocalTensor<float> values;
                if constexpr (std::is_same<T, float>::value) {
                    values = raw.template ReinterpretCast<float>();
                } else {
                    values = floatBuffer_.Get<float>();
                    if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(raw, raw, raw,
                                  static_cast<int32_t>(vectorElements));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                                  static_cast<int32_t>(vectorElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values,
                             static_cast<int32_t>(vectorElements)); }
                AscendC::PipeBarrier<PIPE_V>();

                const uint64_t partialCount =
                    vectorElements / kLongArPartialWidth;
                AscendC::WholeReduceSum<float>(
                    partial, values, static_cast<int32_t>(kLongArPartialWidth),
                    static_cast<int32_t>(partialCount), 1, 1,
                    static_cast<int32_t>(kLongArPartialWidth / 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(
                    sum, partial, static_cast<int32_t>(partialCount),
                    1, 1, 1, 8);
                SyncEvent<AscendC::HardEvent::V_S>();
                total += sum.GetValue(0);
            }
        }

        StoreOutput(output, total, raw, sum);
    }

private:
    static constexpr uint32_t kOutputBatch = 256;
    uint32_t pendingOutputs_ = 0;
    uint64_t pendingOutputStart_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputFloatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputRawBuffer_;
    uint32_t outerAxisCount_ = 0;
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    event_t rawReusable_{};
    event_t inputReady_{};
};

// Long prefix reduction with a tiny preserved suffix.  The logical tensor is
// [reduce, inner] with inner <= 16.  Loading one contiguous slab amortizes MTE2
// setup across thousands of rows; a reusable byte-offset table then gathers
// one column into a dense vector before the ordinary two-stage FP32 reduction.
// Every AIV owns complete 32-byte output blocks, so short MTE3 writes never
// race and no cross-core synchronization is required.
template <typename T>
class KernelSquareSumPrefixSmallInner {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(rawBuffer_, data_.rowTile * data_.inner * sizeof(T));
        pipe_.InitBuffer(gatherBuffer_, data_.rowTile * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, data_.rowTile * sizeof(float));
        }
        pipe_.InitBuffer(indexBuffer_, data_.rowTile * sizeof(uint32_t));
        pipe_.InitBuffer(partialBuffer_,
            (data_.rowTile / kLongArPartialWidth) * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint64_t outputBlockElements = 32 / sizeof(T);
        const uint64_t outputBlocks =
            (data_.outputElements + outputBlockElements - 1) /
            outputBlockElements;
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t baseBlocks = outputBlocks / blocks;
        const uint64_t extraBlocks = outputBlocks - baseBlocks * blocks;
        const uint64_t firstBlock = block * baseBlocks +
            (block < extraBlocks ? block : extraBlocks);
        const uint64_t ownedBlocks = baseBlocks +
            (block < extraBlocks ? 1 : 0);
        const uint64_t outputBegin = firstBlock * outputBlockElements;
        const uint64_t rawOutputEnd =
            (firstBlock + ownedBlocks) * outputBlockElements;
        const uint64_t outputEnd = rawOutputEnd < data_.outputElements
            ? rawOutputEnd
            : data_.outputElements;
        if (outputBegin >= outputEnd) {
            return;
        }

        AscendC::LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        auto signedOffsets = offsets.template ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signedOffsets, 0, static_cast<uint32_t>(data_.rowTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(signedOffsets, signedOffsets, static_cast<int32_t>(data_.inner * sizeof(T)), static_cast<uint32_t>(data_.rowTile));
        AscendC::PipeBarrier<PIPE_V>();

        float totals[kPrefixSmallInnerMax] = {};
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<T> gathered = gatherBuffer_.Get<T>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        for (uint64_t rowBegin = 0; rowBegin < data_.reduce;
             rowBegin += data_.rowTile) {
            const uint64_t rows = data_.reduce - rowBegin < data_.rowTile
                ? data_.reduce - rowBegin
                : data_.rowTile;
            const uint64_t rawElements = rows * data_.inner;
            const uint64_t dmaElements = 32 / sizeof(T);
            const uint64_t rawDmaCount =
                AlignUpElements(rawElements, dmaElements);
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(rawElements * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> inputPad{
                true, 0, static_cast<uint8_t>(rawDmaCount - rawElements),
                static_cast<T>(0)};
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPad(
                raw, xGm_[rowBegin * data_.inner], inputParams, inputPad);
            Sync<AscendC::HardEvent::MTE2_V>();

            const uint64_t vectorCount =
                AlignUpElements(rows, kLongArPartialWidth);
            for (uint64_t output = outputBegin; output < outputEnd; ++output) {
                ZeroTensor(gathered, vectorCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(
                    gathered, raw[output], offsets, 0,
                    static_cast<uint32_t>(rows));
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::LocalTensor<float> values;
                if constexpr (std::is_same<T, float>::value) {
                    values = gathered.template ReinterpretCast<float>();
                } else {
                    values = floatBuffer_.Get<float>();
                    if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(gathered, gathered, gathered,
                        static_cast<int32_t>(vectorCount));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, gathered,
                        AscendC::RoundMode::CAST_NONE,
                        static_cast<int32_t>(vectorCount));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values,
                    static_cast<int32_t>(vectorCount)); }
                AscendC::PipeBarrier<PIPE_V>();

                const uint64_t partialCount =
                    vectorCount / kLongArPartialWidth;
                AscendC::WholeReduceSum<float>(
                    partial, values,
                    static_cast<int32_t>(kLongArPartialWidth),
                    static_cast<int32_t>(partialCount), 1, 1,
                    static_cast<int32_t>(kLongArPartialWidth / 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(
                    sum, partial, static_cast<int32_t>(partialCount),
                    1, 1, 1, 8);
                Sync<AscendC::HardEvent::V_S>();
                totals[output - outputBegin] += sum.GetValue(0);
            }
        }

        AscendC::LocalTensor<T> outputLocal = raw;
        for (uint64_t output = outputBegin; output < outputEnd; ++output) {
            outputLocal.SetValue(
                output - outputBegin,
                StoreValue(totals[output - outputBegin]));
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>((outputEnd - outputBegin) * sizeof(T)),
            0, 0, 0};
        AscendC::DataCopyPad(yGm_[outputBegin], outputLocal, outputParams);
        Sync<AscendC::HardEvent::MTE3_S>();
    }

private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void ZeroTensor(
        const AscendC::LocalTensor<T>& tensor, uint64_t elements) const
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::LocalTensor<uint16_t> bits =
                tensor.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(
                bits, static_cast<uint16_t>(0), static_cast<int32_t>(elements));
        } else {
            AscendC::Duplicate(
                tensor, static_cast<T>(0), static_cast<int32_t>(elements));
        }
    }

    __aicore__ inline T StoreValue(float value) const
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            return AscendC::ToBfloat16(value);
        } else {
            return static_cast<T>(value);
        }
    }

private:
    static constexpr uint64_t kPrefixSmallInnerMax = 16;
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

template <typename T>
class KernelSquareSumPrefixParallel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(rawBuffer_, data_.rowTile * data_.inner * sizeof(T));
        pipe_.InitBuffer(gatherBuffer_, data_.rowTile * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, data_.rowTile * sizeof(float));
        }
        pipe_.InitBuffer(indexBuffer_, data_.rowTile * sizeof(uint32_t));
        pipe_.InitBuffer(partialBuffer_,
            (data_.rowTile / kLongArPartialWidth) * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(foldBuffer_, 64 * 16 * sizeof(float));
        AscendC::SetSysWorkspace(workspace);
        partialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
    }

    __aicore__ inline void Process()
    {
        const uint64_t outputBegin = 0;
        const uint64_t outputEnd = data_.inner;
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t chunk = (data_.reduce + blocks - 1) / blocks;
        const uint64_t firstRow = block * chunk;
        const uint64_t lastRow = firstRow + chunk < data_.reduce ? firstRow + chunk : data_.reduce;
        auto offsets = indexBuffer_.Get<uint32_t>();
        auto signedOffsets = offsets.template ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signedOffsets, 0, static_cast<uint32_t>(data_.rowTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(signedOffsets, signedOffsets, static_cast<int32_t>(data_.inner * sizeof(T)), static_cast<uint32_t>(data_.rowTile));
        AscendC::PipeBarrier<PIPE_V>();

        float totals[kPrefixSmallInnerMax] = {};
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<T> gathered = gatherBuffer_.Get<T>();
        AscendC::LocalTensor<float> partial = partialBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        for (uint64_t rowBegin = firstRow; rowBegin < lastRow;
             rowBegin += data_.rowTile) {
            const uint64_t rows = lastRow - rowBegin < data_.rowTile
                ? lastRow - rowBegin
                : data_.rowTile;
            const uint64_t rawElements = rows * data_.inner;
            const uint64_t dmaElements = 32 / sizeof(T);
            const uint64_t rawDmaCount =
                AlignUpElements(rawElements, dmaElements);
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(rawElements * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> inputPad{
                true, 0, static_cast<uint8_t>(rawDmaCount - rawElements),
                static_cast<T>(0)};
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPad(
                raw, xGm_[rowBegin * data_.inner], inputParams, inputPad);
            Sync<AscendC::HardEvent::MTE2_V>();

            const uint64_t vectorCount =
                AlignUpElements(rows, kLongArPartialWidth);
            for (uint64_t output = outputBegin; output < outputEnd; ++output) {
                ZeroTensor(gathered, vectorCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(
                    gathered, raw[output], offsets, 0,
                    static_cast<uint32_t>(rows));
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::LocalTensor<float> values;
                if constexpr (std::is_same<T, float>::value) {
                    values = gathered.template ReinterpretCast<float>();
                } else {
                    values = floatBuffer_.Get<float>();
                    if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(gathered, gathered, gathered,
                        static_cast<int32_t>(vectorCount));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, gathered,
                        AscendC::RoundMode::CAST_NONE,
                        static_cast<int32_t>(vectorCount));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values,
                    static_cast<int32_t>(vectorCount)); }
                AscendC::PipeBarrier<PIPE_V>();

                const uint64_t partialCount =
                    vectorCount / kLongArPartialWidth;
                AscendC::WholeReduceSum<float>(
                    partial, values,
                    static_cast<int32_t>(kLongArPartialWidth),
                    static_cast<int32_t>(partialCount), 1, 1,
                    static_cast<int32_t>(kLongArPartialWidth / 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(
                    sum, partial, static_cast<int32_t>(partialCount),
                    1, 1, 1, 8);
                Sync<AscendC::HardEvent::V_S>();
                totals[output - outputBegin] += sum.GetValue(0);
            }
        }

        auto fold = foldBuffer_.Get<float>();
        for (uint32_t j = 0; j < 16; ++j) { fold.SetValue(j, j < data_.inner ? totals[j] : 0.0f); }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(partialGm_[block * 16], fold, 16);
        Sync<AscendC::HardEvent::MTE3_S>();
        AscendC::SyncAll<true>();
        if (block == 0) {
            uint32_t power = 1; while (power < blocks) power *= 2;
            AscendC::Duplicate(fold, 0.0f, power * 16);
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(fold, partialGm_, static_cast<uint32_t>(blocks * 16));
            Sync<AscendC::HardEvent::MTE2_V>();
            for (uint32_t half = power / 2; half != 0; half /= 2) {
                AscendC::Add(fold, fold, fold[half * 16], half * 16);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(data_.inner * sizeof(T)), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_, fold, store);
            } else {
                AscendC::Cast(raw, fold, AscendC::RoundMode::CAST_RINT, 16);
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_, raw, store);
            }
            Sync<AscendC::HardEvent::MTE3_V>();
        }
    }

private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void ZeroTensor(
        const AscendC::LocalTensor<T>& tensor, uint64_t elements) const
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::LocalTensor<uint16_t> bits =
                tensor.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(
                bits, static_cast<uint16_t>(0), static_cast<int32_t>(elements));
        } else {
            AscendC::Duplicate(
                tensor, static_cast<T>(0), static_cast<int32_t>(elements));
        }
    }

    __aicore__ inline T StoreValue(float value) const
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            return AscendC::ToBfloat16(value);
        } else {
            return static_cast<T>(value);
        }
    }

private:
    static constexpr uint64_t kPrefixSmallInnerMax = 16;
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> foldBuffer_;
    AscendC::GlobalTensor<float> partialGm_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

// Vector path for a consecutive reduction block in the middle of an ND
// tensor. The logical layout is [outer, reduce, inner]. One task owns an
// inner tile for one outer row; a 2-D MTE2 copy gathers all reduction rows at
// once, and MulAddDst accumulates the squared rows in FP32.
template <typename T>
class KernelSquareSumMiddleReduceVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(rawBuffer_, data_.rowTile * data_.reduceAligned * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, data_.rowTile * data_.reduceAligned * sizeof(float));
        }
        pipe_.InitBuffer(accumulatorBuffer_, data_.rowTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint64_t innerTiles = (data_.inner + data_.rowTile - 1) / data_.rowTile;
        const uint64_t tasks = data_.outer * innerTiles;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks;
             task += AscendC::GetBlockNum()) {
            const uint64_t outerIndex = task / innerTiles;
            const uint64_t innerTile = task - outerIndex * innerTiles;
            const uint64_t innerStart = innerTile * data_.rowTile;
            const uint64_t valid = innerStart + data_.rowTile <= data_.inner
                                       ? data_.rowTile
                                       : data_.inner - innerStart;
            ProcessTile(outerIndex, innerStart, valid);
        }
    }

private:
    __aicore__ inline void ProcessTile(
        uint64_t outerIndex, uint64_t innerStart, uint64_t valid)
    {
        const uint64_t aligned = data_.rowTile;
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::Duplicate(accumulator, 0.0f, static_cast<int32_t>(aligned));
        AscendC::PipeBarrier<PIPE_V>();

        // Consecutive reduction rows are separated by the preserved inner
        // extent. DataCopyPad's srcStride is the gap after each block.
        const uint64_t inputOffset = outerIndex * data_.reduce * data_.inner + innerStart;
        const uint64_t gapBytes = (data_.inner - valid) * sizeof(T);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(data_.reduce),
            static_cast<uint32_t>(valid * sizeof(T)),
            static_cast<uint32_t>(gapBytes), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(aligned - valid), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> values;
        if constexpr (std::is_same<T, float>::value) {
            values = raw.template ReinterpretCast<float>();
        } else {
            values = floatBuffer_.Get<float>();
            AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                          data_.reduce * aligned);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
            AscendC::LocalTensor<float> row = values[reduction * aligned];
            AscendC::MulAddDst(accumulator, row, row, static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t outputOffset = outerIndex * data_.inner + innerStart;
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(yGm_[outputOffset], accumulator, outputParams);
        } else {
            AscendC::Cast(raw, accumulator, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(yGm_[outputOffset], raw, outputParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

template <typename T>
class KernelSquareSumStridedVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        // The middle-reduction variant uses one 2-D GM gather and keeps all
        // reduction rows in UB.  It deliberately shares the strided-vector
        // tiling key (see host tiling) so CANN 8.5 only has one stable binary
        // entry for this dtype.
        const uint64_t rawElements = data_.reserved0 != 0
            ? data_.rowTile * data_.reduceAligned
            : data_.rowTile;
        pipe_.InitBuffer(rawBuffer_, rawElements * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(floatBuffer_, rawElements * sizeof(float));
        }
        pipe_.InitBuffer(accumulatorBuffer_, data_.rowTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        rawReusable_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        inputReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        outputReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        outputToVector_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        outputToMte2_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        if (data_.reserved0 != 0) {
            ProcessMiddle();
            return;
        }
        if (data_.reserved1 != 0) {
            ProcessGenericSuffix();
            return;
        }
        const uint64_t innerTiles = (data_.inner + data_.rowTile - 1) / data_.rowTile;
        const uint64_t tasks = data_.outer * innerTiles;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum()) {
            const uint64_t outerIndex = task / innerTiles;
            const uint64_t innerTile = task - outerIndex * innerTiles;
            const uint64_t innerStart = innerTile * data_.rowTile;
            const uint64_t valid = innerStart + data_.rowTile <= data_.inner
                                       ? data_.rowTile
                                       : data_.inner - innerStart;
            ProcessTile(outerIndex, innerStart, valid);
        }
    }

private:
    __aicore__ inline void WaitRawReusable() const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawReusable_);
    }

    __aicore__ inline void WaitInputReady() const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
    }

    __aicore__ inline void WaitOutputReady() const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
    }

    __aicore__ inline void WaitOutputConsumed() const
    {
        if constexpr (std::is_same<T, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputToVector_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputToVector_);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(outputToMte2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(outputToMte2_);
        }
    }

    __aicore__ inline uint64_t GenericOutputBase(uint64_t outputLinear) const
    {
        uint64_t remaining = outputLinear;
        uint64_t offset = 0;
        for (int32_t pos = static_cast<int32_t>(data_.outputRank) - 1;
             pos >= 0; --pos) {
            const int32_t axis = data_.nonReduceAxes[pos];
            const uint64_t coordinate = remaining % data_.shape[axis];
            remaining /= data_.shape[axis];
            offset += coordinate * data_.inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline uint64_t GenericReductionOffset(
        uint64_t reductionLinear) const
    {
        uint64_t remaining = reductionLinear;
        uint64_t offset = 0;
        for (int32_t pos = static_cast<int32_t>(data_.axisCount) - 1;
             pos >= 0; --pos) {
            const int32_t axis = data_.axes[pos];
            const uint64_t coordinate = remaining % data_.shape[axis];
            remaining /= data_.shape[axis];
            offset += coordinate * data_.inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline void ProcessGenericSuffix()
    {
        if (data_.reserved1 == 3) {
            // Own contiguous 32-byte output units.  Boundaries between AIVs
            // are therefore race-free even when a unit cuts across a short
            // logical suffix.  Split only within one AIV before invoking the
            // established suffix tile implementation.
            const uint64_t unitElements = data_.tileElements;
            const uint64_t units =
                (data_.outputElements + unitElements - 1) / unitElements;
            const uint64_t blockIndex = AscendC::GetBlockIdx();
            const uint64_t blockCount = AscendC::GetBlockNum();
            const uint64_t baseUnits = units / blockCount;
            const uint64_t extraUnits = units - baseUnits * blockCount;
            const uint64_t startUnit = blockIndex * baseUnits +
                (blockIndex < extraUnits ? blockIndex : extraUnits);
            const uint64_t ownedUnits = baseUnits +
                (blockIndex < extraUnits ? 1 : 0);
            uint64_t outputLinear = startUnit * unitElements;
            const uint64_t rawEnd =
                (startUnit + ownedUnits) * unitElements;
            const uint64_t outputEnd = rawEnd < data_.outputElements
                ? rawEnd
                : data_.outputElements;
            while (outputLinear < outputEnd) {
                const uint64_t outputOuter = outputLinear / data_.inner;
                const uint64_t innerStart =
                    outputLinear - outputOuter * data_.inner;
                const uint64_t groupRemaining = data_.inner - innerStart;
                const uint64_t ownedRemaining = outputEnd - outputLinear;
                const uint64_t remaining = groupRemaining < ownedRemaining
                    ? groupRemaining : ownedRemaining;
                const uint64_t valid = remaining < data_.rowTile
                    ? remaining : data_.rowTile;
                ProcessGenericSuffixTile(outputOuter, innerStart, valid);
                outputLinear += valid;
            }
            return;
        }
        if (data_.reserved1 == 2) {
            // Partition whole 32-byte-aligned output-group units contiguously.
            // This preserves exact coverage while keeping short, unaligned
            // output DMAs from two AIVs out of the same GM cache line.
            const uint64_t groupsPerUnit = data_.tileElements;
            const uint64_t units =
                (data_.outer + groupsPerUnit - 1) / groupsPerUnit;
            const uint64_t blockIndex = AscendC::GetBlockIdx();
            const uint64_t blockCount = AscendC::GetBlockNum();
            const uint64_t baseUnits = units / blockCount;
            const uint64_t extraUnits = units - baseUnits * blockCount;
            const uint64_t startUnit = blockIndex * baseUnits +
                (blockIndex < extraUnits ? blockIndex : extraUnits);
            const uint64_t ownedUnits = baseUnits +
                (blockIndex < extraUnits ? 1 : 0);
            const uint64_t outerBegin = startUnit * groupsPerUnit;
            const uint64_t rawOuterEnd =
                (startUnit + ownedUnits) * groupsPerUnit;
            const uint64_t outerEnd = rawOuterEnd < data_.outer
                ? rawOuterEnd
                : data_.outer;
            for (uint64_t outputOuter = outerBegin;
                 outputOuter < outerEnd; ++outputOuter) {
                ProcessGenericSuffixTile(outputOuter, 0, data_.inner);
            }
            return;
        }
        const uint64_t innerTiles =
            (data_.inner + data_.rowTile - 1) / data_.rowTile;
        const uint64_t tasks = data_.outer * innerTiles;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks;
             task += AscendC::GetBlockNum()) {
            const uint64_t outputOuter = task / innerTiles;
            const uint64_t innerTile = task - outputOuter * innerTiles;
            const uint64_t innerStart = innerTile * data_.rowTile;
            const uint64_t valid = innerStart + data_.rowTile <= data_.inner
                                       ? data_.rowTile
                                       : data_.inner - innerStart;
            ProcessGenericSuffixTile(outputOuter, innerStart, valid);
        }
    }

    __aicore__ inline void ProcessGenericSuffixTile(
        uint64_t outputOuter, uint64_t innerStart, uint64_t valid)
    {
        const uint64_t rawAlignment = 32 / sizeof(T);
        const uint64_t alignment = rawAlignment > 8 ? rawAlignment : 8;
        const uint64_t aligned = AlignUpElements(valid, alignment);
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> accumulator =
            accumulatorBuffer_.Get<float>();
        AscendC::Duplicate(accumulator, 0.0f,
                           static_cast<int32_t>(aligned));
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t outputBase = outputOuter * data_.inner;
        const uint64_t inputBase = GenericOutputBase(outputBase);
        for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
            // Each tile reuses rawBuffer_.  The first reduction of a new tile
            // must also wait for the previous tile's vector read to finish;
            // otherwise its MTE2 can overwrite raw while PIPE_V is still
            // consuming the old tile (especially visible when reduce == 1).
            WaitRawReusable();
            const uint64_t inputOffset = inputBase +
                GenericReductionOffset(reduction) + innerStart;
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(aligned - valid),
                static_cast<T>(0)};
            AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
            WaitInputReady();

            if constexpr (std::is_same<T, float>::value) {
                AscendC::LocalTensor<float> values =
                    raw.template ReinterpretCast<float>();
                AscendC::MulAddDst(accumulator, values, values,
                                   static_cast<int32_t>(aligned));
            } else {
                AscendC::LocalTensor<float> values = floatBuffer_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(aligned));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(accumulator, values, values,
                                   static_cast<int32_t>(aligned));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t outputOffset = outputBase + innerStart;
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            WaitOutputReady();
            AscendC::DataCopyPad(yGm_[outputOffset], accumulator, outputParams);
        } else {
            AscendC::Cast(raw, accumulator, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_ALL>();
            WaitOutputReady();
            AscendC::DataCopyPad(yGm_[outputOffset], raw, outputParams);
        }
        WaitOutputConsumed();
    }

    __aicore__ inline void ProcessMiddle()
    {
        if (data_.reserved1 == 4) {
            const uint64_t unitElements = data_.tileElements;
            const uint64_t units =
                (data_.outputElements + unitElements - 1) / unitElements;
            const uint64_t blockIndex = AscendC::GetBlockIdx();
            const uint64_t blockCount = AscendC::GetBlockNum();
            const uint64_t baseUnits = units / blockCount;
            const uint64_t extraUnits = units % blockCount;
            const uint64_t startUnit = blockIndex * baseUnits +
                (blockIndex < extraUnits ? blockIndex : extraUnits);
            const uint64_t ownedUnits = baseUnits +
                (blockIndex < extraUnits ? 1 : 0);
            uint64_t outputLinear = startUnit * unitElements;
            const uint64_t rawEnd =
                (startUnit + ownedUnits) * unitElements;
            const uint64_t outputEnd = rawEnd < data_.outputElements
                ? rawEnd
                : data_.outputElements;
            while (outputLinear < outputEnd) {
                const uint64_t outerIndex = outputLinear / data_.inner;
                const uint64_t innerStart =
                    outputLinear - outerIndex * data_.inner;
                const uint64_t innerRemaining = data_.inner - innerStart;
                const uint64_t ownedRemaining = outputEnd - outputLinear;
                uint64_t valid = innerRemaining < ownedRemaining
                    ? innerRemaining
                    : ownedRemaining;
                valid = valid < data_.rowTile ? valid : data_.rowTile;
                ProcessTile(outerIndex, innerStart, valid);
                outputLinear += valid;
            }
            return;
        }
        const uint64_t innerTiles =
            (data_.inner + data_.rowTile - 1) / data_.rowTile;
        const uint64_t tasks = data_.outer * innerTiles;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks;
             task += AscendC::GetBlockNum()) {
            const uint64_t outerIndex = task / innerTiles;
            const uint64_t innerTile = task - outerIndex * innerTiles;
            const uint64_t innerStart = innerTile * data_.rowTile;
            const uint64_t valid = innerStart + data_.rowTile <= data_.inner
                                       ? data_.rowTile
                                       : data_.inner - innerStart;
            // A multi-block DataCopyPad has device-specific destination-row
            // packing rules for a non-32-byte block length.  Feeding that
            // packed buffer to VEC with a software-computed row stride caused
            // both silent row aliasing and MTE/VEC exceptions on 910B-family
            // devices.  The ordinary strided tile path issues one block per
            // reduction row and therefore has an unambiguous aligned UB
            // layout while retaining vector FP32 accumulation.
            ProcessTile(outerIndex, innerStart, valid);
        }
    }

    __aicore__ inline void ProcessMiddleTile(
        uint64_t outerIndex, uint64_t innerStart, uint64_t valid)
    {
        const uint64_t aligned = data_.rowTile;
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::Duplicate(accumulator, 0.0f, static_cast<int32_t>(aligned));
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t inputOffset =
            outerIndex * data_.reduce * data_.inner + innerStart;
        const uint64_t gapBytes = (data_.inner - valid) * sizeof(T);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(data_.reduce),
            static_cast<uint32_t>(valid * sizeof(T)),
            static_cast<uint32_t>(gapBytes), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(aligned - valid), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> values;
        if constexpr (std::is_same<T, float>::value) {
            values = raw.template ReinterpretCast<float>();
        } else {
            values = floatBuffer_.Get<float>();
            AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(data_.reduce * aligned));
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
            AscendC::LocalTensor<float> row = values[reduction * aligned];
            AscendC::MulAddDst(accumulator, row, row,
                               static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t outputOffset = outerIndex * data_.inner + innerStart;
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyPad(yGm_[outputOffset], accumulator, outputParams);
        } else {
            AscendC::Cast(raw, accumulator, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(yGm_[outputOffset], raw, outputParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessTile(uint64_t outerIndex, uint64_t innerStart, uint64_t valid)
    {
        const uint64_t rawAlignment = 32 / sizeof(T);
        const uint64_t alignment = rawAlignment > 8 ? rawAlignment : 8;
        const uint64_t aligned = AlignUpElements(valid, alignment);
        AscendC::LocalTensor<T> raw = rawBuffer_.Get<T>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        AscendC::Duplicate(accumulator, 0.0f, aligned);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
            // Keep V -> MTE2 ordering across tile boundaries as well as
            // between reduction rows.  A reduce-one tile has no later
            // iteration that could otherwise establish this dependency.
            WaitRawReusable();
            const uint64_t inputOffset =
                (outerIndex * data_.reduce + reduction) * data_.inner + innerStart;
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(aligned - valid), static_cast<T>(0)};
            AscendC::DataCopyPad(raw, xGm_[inputOffset], inputParams, padParams);
            WaitInputReady();

            if constexpr (std::is_same<T, float>::value) {
                AscendC::LocalTensor<float> values = raw.template ReinterpretCast<float>();
                AscendC::MulAddDst(accumulator, values, values, static_cast<int32_t>(aligned));
            } else {
                AscendC::LocalTensor<float> values = floatBuffer_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, aligned);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(accumulator, values, values, static_cast<int32_t>(aligned));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t outputOffset = outerIndex * data_.inner + innerStart;
        AscendC::DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            WaitOutputReady();
            AscendC::DataCopyPad(yGm_[outputOffset], accumulator, outputParams);
        } else {
            AscendC::Cast(raw, accumulator, AscendC::RoundMode::CAST_RINT, aligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            WaitOutputReady();
            AscendC::DataCopyPad(yGm_[outputOffset], raw, outputParams);
        }
        WaitOutputConsumed();
    }

private:
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    event_t rawReusable_{};
    event_t inputReady_{};
    event_t outputReady_{};
    event_t outputToVector_{};
    event_t outputToMte2_{};
};

template <typename T, bool kSplit = false, bool kGeneric = false>
class KernelSquareSumBatchedColumns {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& data, GM_ADDR workspace = nullptr) {
        data_ = data;
        columns_ = static_cast<uint32_t>(data.rowTile);
        batchRows_ = 64;
        while (batchRows_ < 512 && batchRows_ * 2 * columns_ <= 16384) batchRows_ *= 2;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(raw_, batchRows_ * columns_ * sizeof(T));
        if constexpr (!std::is_same<T, float>::value)
            pipe_.InitBuffer(converted_, batchRows_ * columns_ * sizeof(float));
        pipe_.InitBuffer(accumulated_, columns_ * sizeof(float));
        if constexpr (kSplit) {
            parts_ = data.reserved0;
            part_ = AscendC::GetBlockIdx() % parts_;
            partialStride_ = (data.outputElements + 15) / 16 * 16;
            AscendC::SetSysWorkspace(workspace);
            partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
            pipe_.InitBuffer(fold_, 64 * columns_ * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        constexpr uint64_t unit = 32 / sizeof(T);
        const uint64_t units = (data_.outputElements + unit - 1) / unit;
        const uint64_t owners = AscendC::GetBlockNum() / parts_;
        const uint64_t owner = AscendC::GetBlockIdx() / parts_;
        const uint64_t perCore = (units + owners - 1) / owners;
        const uint64_t begin = owner * perCore * unit;
        const uint64_t rawEnd = begin + perCore * unit;
        const uint64_t end = rawEnd < data_.outputElements ? rawEnd : data_.outputElements;
        for (uint64_t output = begin; output < end;) {
            const uint64_t outer = output / data_.inner;
            const uint64_t innerStart = output % data_.inner;
            uint32_t valid = static_cast<uint32_t>(data_.inner - innerStart < end - output
                ? data_.inner - innerStart : end - output);
            if (valid > columns_) valid = columns_;
            Tile(output, outer, innerStart, valid);
            output += valid;
        }
        if constexpr (kSplit) {
            AscendC::SyncAll<true>();
            if (part_ == 0) { Finish(begin, end); }
        }
    }
private:
    template <AscendC::HardEvent event>
    __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline uint64_t OutputOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = data_.outputRank - 1; i >= 0; --i) {
            uint32_t axis = data_.nonReduceAxes[i];
            offset += index % data_.shape[axis] * data_.inputStride[axis];
            index /= data_.shape[axis];
        }
        return offset;
    }
    __aicore__ inline uint64_t ReduceOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = data_.axisCount - 1; i >= 0; --i) {
            uint32_t axis = data_.axes[i];
            offset += index % data_.shape[axis] * data_.inputStride[axis];
            index /= data_.shape[axis];
        }
        return offset;
    }
    __aicore__ inline void Tile(uint64_t output, uint64_t outer, uint64_t innerStart, uint32_t valid) {
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint32_t pitch = (valid + alignment - 1) / alignment * alignment;
        auto raw = raw_.Get<T>(); auto accumulated = accumulated_.Get<float>();
        AscendC::Duplicate(accumulated, 0.0f, pitch);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t chunk = (data_.reduce + parts_ - 1) / parts_;
        const uint64_t reduceBegin = part_ * chunk;
        const uint64_t reduceEnd = reduceBegin + chunk < data_.reduce ? reduceBegin + chunk : data_.reduce;
        for (uint64_t first = reduceBegin; first < reduceEnd;) {
            uint32_t rows = static_cast<uint32_t>(reduceEnd - first < batchRows_ ? reduceEnd - first : batchRows_);
            if constexpr (kGeneric) {
                const uint64_t available = data_.reduceAligned - first % data_.reduceAligned;
                if (rows > available) rows = static_cast<uint32_t>(available);
            }
            uint32_t power = 1; while (power < rows) power *= 2;
            if (power != rows) {
                if constexpr (sizeof(T) == 2) {
                    auto bits = raw.template ReinterpretCast<uint16_t>();
                    AscendC::Duplicate(bits, static_cast<uint16_t>(0), power * pitch);
                } else AscendC::Duplicate(raw, static_cast<T>(0), power * pitch);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),
                static_cast<uint32_t>(valid * sizeof(T)),
                static_cast<uint32_t>((data_.inner - valid) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - valid), static_cast<T>(0)};
            uint64_t offset;
            if constexpr (kGeneric) offset = OutputOffset(output) + ReduceOffset(first);
            else offset = (outer * data_.reduce + first) * data_.inner + innerStart;
            AscendC::DataCopyPad(raw, x_[offset], copy, pad);
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) values = raw.template ReinterpretCast<float>();
            else {
                values = converted_.Get<float>();
                if constexpr (std::is_same<T, half>::value) {
                        AscendC::Mul(raw, raw, raw, power * pitch);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, power * pitch);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (!std::is_same<T, half>::value) { AscendC::Mul(values, values, values, power * pitch); }
            AscendC::PipeBarrier<PIPE_V>();
            // Fold matching columns in contiguous halves: log2(rows) vector
            // operations replace one DMA and one vector launch per row.
            for (uint32_t half = power / 2; half != 0; half /= 2) {
                AscendC::Add(values, values, values[half * pitch], half * pitch);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(accumulated, accumulated, values, pitch);
            AscendC::PipeBarrier<PIPE_V>();
            first += rows;
        }
        if constexpr (kSplit) {
            AscendC::DataCopyExtParams storePartial{1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(partial_[part_ * partialStride_ + output], accumulated, storePartial);
            Sync<AscendC::HardEvent::MTE3_V>();
            return;
        }
        AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], accumulated, store);
        } else {
            AscendC::Cast(raw, accumulated, AscendC::RoundMode::CAST_RINT, pitch);
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], raw, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
        Sync<AscendC::HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Finish(uint64_t begin, uint64_t end) {
        for (uint64_t output = begin; output < end; output += columns_) {
            const uint32_t valid = static_cast<uint32_t>(end - output < columns_ ? end - output : columns_);
            const uint32_t pitch = (valid + 7) / 8 * 8;
            uint32_t power = 1; while (power < parts_) power *= 2;
            auto values = fold_.Get<float>();
            AscendC::Duplicate(values, 0.0f, power * pitch);
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(parts_), valid * 4,
                static_cast<uint32_t>((partialStride_ - valid) * 4), 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(pitch - valid), 0.0f};
            AscendC::DataCopyPad(values, partial_[output], copy, pad);
            Sync<AscendC::HardEvent::MTE2_V>();
            for (uint32_t half = power / 2; half != 0; half /= 2) {
                AscendC::Add(values, values, values[half * pitch], half * pitch);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(y_[output], values, store);
            } else {
                auto raw = raw_.Get<T>();
                AscendC::Cast(raw, values, AscendC::RoundMode::CAST_RINT, pitch);
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(y_[output], raw, store);
            }
            Sync<AscendC::HardEvent::MTE3_V>();
        }
    }
    uint32_t parts_ = 1, part_ = 0;
    uint64_t partialStride_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fold_;
    AscendC::GlobalTensor<float> partial_;
    SquareSumV1TilingData data_{};
    uint32_t columns_, batchRows_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> raw_, converted_, accumulated_;
    AscendC::GlobalTensor<T> x_, y_;
};


// Contiguous sum of squares: strip-mined vector accumulators and double-buffered MTE2.
// Each lane accumulates at most 128 tiles before a tree fold bounds error growth.
template <typename T, bool kCooperative>
class KernelSquareSumStreaming {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                               const SquareSumV1TilingData& data) {
        data_ = data;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(input_, 2, kTile * sizeof(T));
        pipe_.InitBuffer(acc_, kTile * sizeof(float));
        if constexpr (!std::is_same<T, float>::value && !std::is_same<T, half>::value)
            pipe_.InitBuffer(converted_, kTile * sizeof(float));
        pipe_.InitBuffer(scratch_, 256 * sizeof(float));
        pipe_.InitBuffer(total_, 16 * sizeof(float));
        pipe_.InitBuffer(result_, 64 * 8 * sizeof(float));
        if constexpr (kCooperative) {
            AscendC::SetSysWorkspace(workspace);
            partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
            pipe_.InitBuffer(fold_, 64 * 8 * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t blocks = AscendC::GetBlockNum();
        if constexpr (kCooperative) {
            const uint64_t parts = data_.reserved0;
            const uint64_t tasks = data_.outputElements * parts;
            for (uint64_t task = block; task < tasks; task += blocks) {
            const uint64_t output = task / parts;
            const uint64_t part = task % parts;
            const uint64_t tiles = (data_.reduce + kTile - 1) / kTile;
            const uint64_t beginTile = tiles * part / parts;
            const uint64_t endTile = tiles * (part + 1) / parts;
            const uint64_t begin = beginTile * kTile;
            const uint64_t rawEnd = endTile * kTile;
            const uint64_t end = rawEnd < data_.reduce ? rawEnd : data_.reduce;
            Range(output * data_.reduce + begin, end - begin);
            auto sum = total_.Get<float>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(partial_[task * 8], sum, 8);
            Sync<AscendC::HardEvent::MTE3_S>();
            }
            AscendC::SyncAll<true>();
            Finish();
        } else {
            constexpr uint64_t unit = 32 / sizeof(T);
            const uint64_t units = (data_.outputElements + unit - 1) / unit;
            const uint64_t first = units * block / blocks * unit;
            const uint64_t limit = units * (block + 1) / blocks * unit;
            const uint64_t end = limit < data_.outputElements ? limit : data_.outputElements;
            for (uint64_t group = first; group < end; group += unit) {
                const uint32_t rows = static_cast<uint32_t>(end - group < unit ? end - group : unit);
                auto results = result_.Get<float>();
                for (uint32_t i = 0; i < rows; ++i) {
                    Range((group + i) * data_.reduce, data_.reduce);
                    auto sum = total_.Get<float>();
                    // Copy the scalar with a vector instruction; no V->S readback.
                    AscendC::Adds(results[i * 8], sum, 0.0f, 8);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                auto compact = scratch_.Get<float>();
                AscendC::WholeReduceSum<float>(compact, results, 8, rows, 1, 1, 1);
                AscendC::PipeBarrier<PIPE_V>();
                Store(group, rows, compact);
            }
        }
    }
private:
    static constexpr uint32_t kTile = 8192;
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t valid) {
        auto raw = input_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        constexpr uint32_t align = 32 / sizeof(T);
        const uint32_t padded = (valid + align - 1) / align * align;
        AscendC::DataCopyPadExtParams<T> pad{true, 0,
            static_cast<uint8_t>(padded - valid), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, x_[offset], copy, pad);
        input_.EnQue(raw);
    }
    __aicore__ inline void Flush() {
        auto lanes = acc_.Get<float>(); auto scratch = scratch_.Get<float>();
        auto total = total_.Get<float>(); auto epoch = total[8];
        AscendC::WholeReduceSum<float>(scratch, lanes, 64, kTile / 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch, scratch, scratch[64], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(epoch, scratch, 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(total, total, epoch, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Range(uint64_t begin, uint64_t count) {
        auto lanes = acc_.Get<float>(); auto total = total_.Get<float>();
        AscendC::Duplicate(lanes, 0.0f, kTile);
        AscendC::Duplicate(total, 0.0f, 16);
        AscendC::PipeBarrier<PIPE_V>();
        if (count == 0) return;
        const uint64_t tiles = (count + kTile - 1) / kTile;
        CopyIn(begin, static_cast<uint32_t>(count < kTile ? count : kTile));
        for (uint64_t t = 0; t < tiles; ++t) {
            if (t + 1 < tiles) {
                const uint64_t remaining = count - (t + 1) * kTile;
                CopyIn(begin + (t + 1) * kTile,
                       static_cast<uint32_t>(remaining < kTile ? remaining : kTile));
            }
            auto raw = input_.template DeQue<T>();
            const uint64_t remaining = count - t * kTile;
            const uint32_t valid = static_cast<uint32_t>(remaining < kTile ? remaining : kTile);
            if constexpr (std::is_same<T, half>::value) {
                AscendC::Mul(raw, raw, raw, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Axpy(lanes, raw, static_cast<half>(1.0f), valid);
            } else if constexpr (std::is_same<T, float>::value) {
                AscendC::MulAddDst(lanes, raw, raw, valid);
            } else {
                auto values = converted_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, valid);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(lanes, values, values, valid);
            }
            AscendC::PipeBarrier<PIPE_V>();
            input_.FreeTensor(raw);
            if ((t + 1) % 128 == 0 && t + 1 < tiles) {
                Flush();
                AscendC::Duplicate(lanes, 0.0f, kTile);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        Flush();
    }
    __aicore__ inline void Store(uint64_t output, uint32_t count,
                                AscendC::LocalTensor<float> results) {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], results, copy);
        } else {
            auto cast = acc_.Get<T>();
            AscendC::Cast(cast, results, AscendC::RoundMode::CAST_RINT, count);
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], cast, copy);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    __aicore__ inline void Finish() {
        constexpr uint32_t unit = 32 / sizeof(T);
        const uint32_t parts = data_.reserved0;
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t blocks = AscendC::GetBlockNum();
        auto values = fold_.Get<float>();
        auto results = result_.Get<float>();
        auto scratch = scratch_.Get<float>();
        for (uint64_t first = block * unit; first < data_.outputElements; first += blocks * unit) {
            const uint32_t rows = static_cast<uint32_t>(data_.outputElements - first < unit
                ? data_.outputElements - first : unit);
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(values, partial_[first * parts * 8], rows * parts * 8);
            Sync<AscendC::HardEvent::MTE2_V>();
            if (parts <= 8) {
                // Each partial record has one sum and seven zeros. Up to eight
                // records fit one 64-lane repeat, so all output rows fold in
                // one instruction instead of two reductions per output.
                AscendC::WholeReduceSum<float>(scratch, values, parts * 8, rows, 1, 1, parts);
                AscendC::PipeBarrier<PIPE_V>();
                Store(first, rows, scratch);
                continue;
            }
            AscendC::Duplicate(results, 0.0f, rows * 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t o = 0; o < rows; ++o) {
                AscendC::Duplicate(scratch, 0.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(scratch, values[o * parts * 8], 8, parts, 1, 1, 1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(results[o * 8], scratch, parts, 1, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::WholeReduceSum<float>(scratch, results, 8, rows, 1, 1, 1);
            AscendC::PipeBarrier<PIPE_V>();
            Store(first, rows, scratch);
        }
    }
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_, converted_, scratch_, total_, result_, fold_;
    AscendC::GlobalTensor<T> x_, y_;
    AscendC::GlobalTensor<float> partial_;
};


// Batch adjacent outputs of arbitrary alternating layouts ending in a short
// reduced suffix. Only geometry is decoded; input values are never inspected.
template <typename T>
class KernelSquareSumGroupedSuffix {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& data) {
        data_ = data;
        outerAxes_ = data_.axisCount - 1;
        while (outerAxes_ > 0 && data_.axes[outerAxes_ - 1] + 1 == data_.axes[outerAxes_]) --outerAxes_;
        pitch_ = (data_.inner + 32 / sizeof(T) - 1) / (32 / sizeof(T)) * (32 / sizeof(T));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(input_, 2, 4096 * sizeof(T));
        pipe_.InitBuffer(lanes_, 4096 * sizeof(float));
        if constexpr (!std::is_same<T, float>::value && !std::is_same<T, half>::value)
            pipe_.InitBuffer(converted_, 4096 * sizeof(float));
        pipe_.InitBuffer(results_, 128 * sizeof(float));
        pipe_.InitBuffer(output_, 64 * sizeof(T));
    }
    __aicore__ inline void Process() {
        constexpr uint64_t unit = 32 / sizeof(T);
        const uint64_t units = (data_.outputElements + unit - 1) / unit;
        const uint64_t blocks = AscendC::GetBlockNum(), block = AscendC::GetBlockIdx();
        const uint64_t first = units * block / blocks * unit;
        const uint64_t rawEnd = units * (block + 1) / blocks * unit;
        const uint64_t end = rawEnd < data_.outputElements ? rawEnd : data_.outputElements;
        for (uint64_t start = first; start < end;) {
            uint64_t count = end - start;
            const uint64_t available = data_.reduceAligned - start % data_.reduceAligned;
            if (count > available) count = available;
            if (count > 64) count = 64;
            Group(start, static_cast<uint32_t>(count));
            start += count;
        }
    }
private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline uint64_t OutputOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = data_.outputRank - 1; i >= 0; --i) {
            uint32_t axis = data_.nonReduceAxes[i];
            offset += index % data_.shape[axis] * data_.inputStride[axis];
            index /= data_.shape[axis];
        }
        return offset;
    }
    __aicore__ inline uint64_t OuterOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(outerAxes_) - 1; i >= 0; --i) {
            uint32_t axis = data_.axes[i];
            offset += index % data_.shape[axis] * data_.inputStride[axis];
            index /= data_.shape[axis];
        }
        return offset;
    }
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t rows) {
        auto raw = input_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),
            static_cast<uint32_t>(data_.inner * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0,
            static_cast<uint8_t>(pitch_ - data_.inner), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, x_[offset], copy, pad); input_.EnQue(raw);
    }
    __aicore__ inline void Flush(uint32_t rows) {
        auto results = results_.Get<float>(); auto epoch = results[64];
        auto lanes = lanes_.Get<float>();
        AscendC::WholeReduceSum<float>(epoch, lanes, pitch_, rows, 1, 1, pitch_ / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(results, results, epoch, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Group(uint64_t start, uint32_t rows) {
        auto lanes = lanes_.Get<float>(); auto results = results_.Get<float>();
        AscendC::Duplicate(lanes, 0.0f, rows * pitch_);
        AscendC::Duplicate(results, 0.0f, 128);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t base = OutputOffset(start);
        const uint64_t outer = data_.reduce / data_.inner;
        CopyIn(base, rows);
        for (uint64_t r = 0; r < outer; ++r) {
            if (r + 1 < outer) CopyIn(base + OuterOffset(r + 1), rows);
            auto raw = input_.template DeQue<T>();
            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
                AscendC::MulAddDst(lanes, raw, raw, rows * pitch_);
            } else {
                auto values = converted_.Get<float>();
                AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, rows * pitch_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(lanes, values, values, rows * pitch_);
            }
            AscendC::PipeBarrier<PIPE_V>(); input_.FreeTensor(raw);
            if ((r + 1) % 128 == 0 && r + 1 < outer) {
                Flush(rows); AscendC::Duplicate(lanes, 0.0f, rows * pitch_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        Flush(rows);
        AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[start], results, store);
        } else {
            auto output = output_.Get<T>();
            AscendC::Cast(output, results, AscendC::RoundMode::CAST_RINT, rows);
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[start], output, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    SquareSumV1TilingData data_{};
    uint32_t outerAxes_, pitch_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lanes_, converted_, results_, output_;
    AscendC::GlobalTensor<T> x_, y_;
};


// Joint tiling of reduction rows and neighbouring outputs: load a rectangle,
// fold outer rows in UB, then gather short suffixes for batched horizontal sums.
template <typename T, bool kSplit = false>
class KernelSquareSumRectangularSuffix {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& data, GM_ADDR workspace = nullptr) {
        data_ = data;
        if constexpr (kSplit) {
            parts_ = data_.reserved0; part_ = AscendC::GetBlockIdx() % parts_;
            partialStride_ = (data_.outputElements + 15) / 16 * 16;
            AscendC::SetSysWorkspace(workspace);
            partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
        }
        outerAxes_ = data_.axisCount - 1;
        while (outerAxes_ > 0 && data_.axes[outerAxes_ - 1] + 1 == data_.axes[outerAxes_]) --outerAxes_;
        outerRun_ = data_.shape[data_.axes[outerAxes_ - 1]];
        for (int32_t i = static_cast<int32_t>(outerAxes_) - 2; i >= 0; --i) {
            if (data_.axes[i] + 1 != data_.axes[i + 1]) break;
            outerRun_ *= data_.shape[data_.axes[i]];
        }
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(raw_, 8192 * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) pipe_.InitBuffer(converted_, 8192 * sizeof(float));
        pipe_.InitBuffer(acc_, 4096 * sizeof(float)); pipe_.InitBuffer(gather_, 4096 * sizeof(float));
        pipe_.InitBuffer(indices_, 4096 * sizeof(int32_t)); pipe_.InitBuffer(indexTmp_, 4096 * sizeof(int32_t));
        pipe_.InitBuffer(result_, 128 * sizeof(float)); pipe_.InitBuffer(output_, 64 * sizeof(T));
        suffixPitch_ = 1; logPitch_ = 0;
        while (suffixPitch_ < data_.inner || suffixPitch_ < 8) { suffixPitch_ *= 2; ++logPitch_; }
        auto index = indices_.Get<int32_t>(); auto temp = indexTmp_.Get<int32_t>();
        AscendC::CreateVecIndex(index, 0, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(temp, index, static_cast<int32_t>(logPitch_), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temp, temp, static_cast<int32_t>(data_.inner) - static_cast<int32_t>(suffixPitch_), 4096);
        AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(index, index, temp, 4096);
        AscendC::PipeBarrier<PIPE_V>(); AscendC::Muls(index, index, static_cast<int32_t>(4), 4096);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        constexpr uint64_t unit = 32 / sizeof(T);
        const uint64_t units = (data_.outputElements + unit - 1) / unit;
        const uint64_t blocks = AscendC::GetBlockNum() / parts_, block = AscendC::GetBlockIdx() / parts_;
        const uint64_t begin = units * block / blocks * unit;
        const uint64_t rawEnd = units * (block + 1) / blocks * unit;
        const uint64_t end = rawEnd < data_.outputElements ? rawEnd : data_.outputElements;
        for (uint64_t first = begin; first < end;) {
            uint64_t rows = end - first;
            const uint64_t available = data_.reduceAligned - first % data_.reduceAligned;
            if (rows > available) rows = available;
            if (rows > 64) rows = 64;
            if (rows > 4096 / suffixPitch_) rows = 4096 / suffixPitch_;
            Group(first, static_cast<uint32_t>(rows)); first += rows;
        }
        if constexpr (kSplit) {
            AscendC::SyncAll<true>();
            if (part_ == 0) Finish(begin, end);
        }
    }
private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline uint64_t OutputOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = data_.outputRank - 1; i >= 0; --i) {
            uint32_t a = data_.nonReduceAxes[i]; offset += index % data_.shape[a] * data_.inputStride[a]; index /= data_.shape[a];
        } return offset;
    }
    __aicore__ inline uint64_t OuterOffset(uint64_t index) const {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(outerAxes_) - 1; i >= 0; --i) {
            uint32_t a = data_.axes[i]; offset += index % data_.shape[a] * data_.inputStride[a]; index /= data_.shape[a];
        } return offset;
    }
    __aicore__ inline void Fold(uint32_t rows) {
        auto acc = acc_.Get<float>(); auto gathered = gather_.Get<float>();
        auto index = indices_.Get<int32_t>().template ReinterpretCast<uint32_t>();
        auto result = result_.Get<float>(); auto partial = result[64];
        AscendC::Gather(gathered, acc, index, 0, rows * suffixPitch_);
        AscendC::PipeBarrier<PIPE_V>();
        if (data_.inner <= 64) {
            AscendC::WholeReduceSum<float>(partial, gathered, data_.inner, rows, 1, 1, suffixPitch_ / 8);
        } else {
            // Gather padding is deliberately not included in the first-stage masks.
            auto stage = indexTmp_.Get<float>();
            const uint32_t chunks = suffixPitch_ / 64;
            AscendC::Duplicate(stage, 0.0f, rows * chunks * 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < chunks; ++c) {
                const uint32_t start = c * 64;
                if (start >= data_.inner) break;
                const uint32_t width = data_.inner - start < 64 ? data_.inner - start : 64;
                AscendC::WholeReduceSum<float>(stage[c * 8], gathered[start], width, rows, chunks * 8, 1, suffixPitch_ / 8);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(partial, stage, chunks * 8, rows, 1, 1, chunks);
        }
        AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(result, result, partial, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Group(uint64_t first, uint32_t rows) {
        const uint32_t columns = rows * data_.inner;
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint32_t pitch = (columns + alignment - 1) / alignment * alignment;
        uint32_t batch = 1; while (batch * 2 <= 64 && batch * 2 * pitch <= 8192) batch *= 2;
        auto raw = raw_.Get<T>(); auto acc = acc_.Get<float>(); auto result = result_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, 4096); AscendC::Duplicate(result, 0.0f, 128);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t base = OutputOffset(first), outer = data_.reduce / data_.inner;
        const uint64_t chunk = (outer + parts_ - 1) / parts_;
        const uint64_t reduceBegin = part_ * chunk;
        const uint64_t reduceEnd = reduceBegin + chunk < outer ? reduceBegin + chunk : outer;
        const uint64_t stride = data_.inputStride[data_.axes[outerAxes_ - 1]];
        uint32_t steps = 0;
        for (uint64_t r = reduceBegin; r < reduceEnd;) {
            uint64_t take = reduceEnd - r;
            if (take > batch) take = batch;
            if (take > outerRun_ - r % outerRun_) take = outerRun_ - r % outerRun_;
            uint32_t power = 1; while (power < take) power *= 2;
            if (power != take) {
                if constexpr (sizeof(T) == 2) AscendC::Duplicate(raw.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), power * pitch);
                else AscendC::Duplicate(raw, 0.0f, power * pitch);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(take), static_cast<uint32_t>(columns * sizeof(T)),
                static_cast<uint32_t>((stride - columns) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - columns), static_cast<T>(0)};
            AscendC::DataCopyPad(raw, x_[base + OuterOffset(r)], copy, pad);
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::LocalTensor<float> values;
            if constexpr (std::is_same<T, float>::value) {
                values = raw.template ReinterpretCast<float>();
                AscendC::Mul(values, values, values, power * pitch);
            } else if constexpr (std::is_same<T, half>::value) {
                AscendC::Mul(raw, raw, raw, power * pitch); AscendC::PipeBarrier<PIPE_V>();
                values = converted_.Get<float>(); AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, power * pitch);
            } else {
                values = converted_.Get<float>(); AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, power * pitch);
                AscendC::PipeBarrier<PIPE_V>(); AscendC::Mul(values, values, values, power * pitch);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t halfRows = power / 2; halfRows; halfRows /= 2) {
                AscendC::Add(values, values, values[halfRows * pitch], halfRows * pitch); AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(acc, acc, values, pitch); AscendC::PipeBarrier<PIPE_V>();
            r += take;
            if (++steps == 128 && r < reduceEnd) { Fold(rows); AscendC::Duplicate(acc, 0.0f, 4096); AscendC::PipeBarrier<PIPE_V>(); steps=0; }
        }
        Fold(rows);
        if constexpr (kSplit) {
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams copy{1, rows * 4, 0, 0, 0};
            AscendC::DataCopyPad(partial_[part_ * partialStride_ + first], result, copy);
            Sync<AscendC::HardEvent::MTE3_V>(); return;
        }
        AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], result, store);
        } else {
            auto output = output_.Get<T>(); AscendC::Cast(output, result, AscendC::RoundMode::CAST_RINT, rows);
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], output, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    __aicore__ inline void Finish(uint64_t begin, uint64_t end) {
        const uint32_t rows = static_cast<uint32_t>(end - begin);
        const uint32_t pitch = (rows + 7) / 8 * 8;
        uint32_t power = 1; while (power < parts_) power *= 2;
        auto values = acc_.Get<float>();
        AscendC::Duplicate(values, 0.0f, power * pitch);
        Sync<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(parts_), rows * 4,
            static_cast<uint32_t>((partialStride_ - rows) * 4), 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true,0,static_cast<uint8_t>(pitch-rows),0.0f};
        AscendC::DataCopyPad(values, partial_[begin], copy, pad);
        Sync<AscendC::HardEvent::MTE2_V>();
        for (uint32_t halfRows=power/2;halfRows;halfRows/=2) {
            AscendC::Add(values,values,values[halfRows*pitch],halfRows*pitch);AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopyExtParams store{1,static_cast<uint32_t>(rows*sizeof(T)),0,0,0};
        if constexpr (std::is_same<T,float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyPad(y_[begin],values,store);
        } else {
            auto out=output_.Get<T>();AscendC::Cast(out,values,AscendC::RoundMode::CAST_RINT,rows);
            Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyPad(y_[begin],out,store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    uint32_t parts_=1,part_=0; uint64_t partialStride_=0;
    AscendC::GlobalTensor<float> partial_;
    SquareSumV1TilingData data_{}; uint32_t outerAxes_,suffixPitch_,logPitch_; uint64_t outerRun_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> raw_,converted_,acc_,gather_,indices_,indexTmp_,result_,output_;
    AscendC::GlobalTensor<T> x_,y_;
};

template <typename T>
class KernelSquareSumV1 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.outputElements);
        pipe_.InitBuffer(outputBuffer_, data_.tileElements * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const Mode mode = static_cast<Mode>(data_.mode);
        if (mode == Mode::kAR || mode == Mode::kReduceAll) {
            ProcessContiguous();
        } else if (mode == Mode::kRA) {
            ProcessStrided();
        } else {
            // kARA may carry a non-zero inner value from host-side candidate
            // analysis (for example splitReduce with one preserved middle
            // run), even when the specialized vector key is rejected by its
            // UB-size gate.  Such an axis set is not the consecutive
            // [outer, reduce, inner] layout assumed by ProcessStrided.
            ProcessGeneric();
        }
    }

private:
    __aicore__ inline void Partition(uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        if (data_.reserved0 == 4 && data_.rowTile != 0) {
            const uint64_t units =
                (total + data_.rowTile - 1) / data_.rowTile;
            const uint64_t baseUnits = units / blocks;
            const uint64_t extraUnits = units % blocks;
            const uint64_t ownedUnits =
                baseUnits + (block < extraUnits ? 1 : 0);
            const uint64_t firstUnit = block * baseUnits +
                (block < extraUnits ? block : extraUnits);
            start = firstUnit * data_.rowTile;
            const uint64_t rawEnd = start + ownedUnits * data_.rowTile;
            count = (rawEnd < total ? rawEnd : total) - start;
            return;
        }
        const uint64_t small = total / blocks;
        const uint64_t remainder = total % blocks;
        count = small + (block < remainder ? 1 : 0);
        start = block * small + (block < remainder ? block : remainder);
    }

    __aicore__ inline T StoreValue(float value) const
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            return AscendC::ToBfloat16(value);
        } else {
            return static_cast<T>(value);
        }
    }

    __aicore__ inline float SquareAt(uint64_t inputOffset) const
    {
        const T input = xGm_.GetValue(inputOffset);
        float value = 0.0f;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            value = AscendC::ToFloat(input);
        } else {
            value = static_cast<float>(input);
        }
        return value * value;
    }

    __aicore__ inline void CopyOutputTile(
        uint64_t outputOffset, const AscendC::LocalTensor<T>& outputLocal, uint64_t count)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessContiguous()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.outputElements, start, count);
        for (uint64_t begin = start; begin < start + count; begin += data_.tileElements) {
            const uint64_t once = start + count - begin < data_.tileElements
                                      ? start + count - begin
                                      : data_.tileElements;
            AscendC::LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
            for (uint64_t local = 0; local < once; ++local) {
                const uint64_t output = begin + local;
                const uint64_t inputBase =
                    static_cast<Mode>(data_.mode) == Mode::kReduceAll ? 0 : output * data_.reduce;
                float accumulator = 0.0f;
                for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
                    accumulator += SquareAt(inputBase + reduction);
                }
                outputLocal.SetValue(local, StoreValue(accumulator));
            }
            CopyOutputTile(begin, outputLocal, once);
        }
    }

    __aicore__ inline void ProcessStrided()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.outputElements, start, count);
        for (uint64_t begin = start; begin < start + count; begin += data_.tileElements) {
            const uint64_t once = start + count - begin < data_.tileElements
                                      ? start + count - begin
                                      : data_.tileElements;
            AscendC::LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
            for (uint64_t local = 0; local < once; ++local) {
                const uint64_t output = begin + local;
                const uint64_t outerIndex = output / data_.inner;
                const uint64_t innerIndex = output % data_.inner;
                const uint64_t inputBase = outerIndex * data_.reduce * data_.inner + innerIndex;
                float accumulator = 0.0f;
                for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
                    accumulator += SquareAt(inputBase + reduction * data_.inner);
                }
                outputLocal.SetValue(local, StoreValue(accumulator));
            }
            CopyOutputTile(begin, outputLocal, once);
        }
    }

    __aicore__ inline uint64_t OutputBase(uint64_t outputLinear) const
    {
        uint64_t remaining = outputLinear;
        uint64_t offset = 0;
        for (int32_t pos = static_cast<int32_t>(data_.outputRank) - 1; pos >= 0; --pos) {
            const int32_t axis = data_.nonReduceAxes[pos];
            const uint64_t coordinate = remaining % data_.shape[axis];
            remaining /= data_.shape[axis];
            offset += coordinate * data_.inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline uint64_t ReductionOffset(uint64_t reductionLinear) const
    {
        uint64_t remaining = reductionLinear;
        uint64_t offset = 0;
        for (int32_t pos = static_cast<int32_t>(data_.axisCount) - 1; pos >= 0; --pos) {
            const int32_t axis = data_.axes[pos];
            const uint64_t coordinate = remaining % data_.shape[axis];
            remaining /= data_.shape[axis];
            offset += coordinate * data_.inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline void ProcessGeneric()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.outputElements, start, count);
        for (uint64_t begin = start; begin < start + count; begin += data_.tileElements) {
            const uint64_t once = start + count - begin < data_.tileElements
                                      ? start + count - begin
                                      : data_.tileElements;
            AscendC::LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
            for (uint64_t local = 0; local < once; ++local) {
                const uint64_t output = begin + local;
                const uint64_t base = OutputBase(output);
                float accumulator = 0.0f;
                for (uint64_t reduction = 0; reduction < data_.reduce; ++reduction) {
                    accumulator += SquareAt(base + ReductionOffset(reduction));
                }
                outputLocal.SetValue(local, StoreValue(accumulator));
            }
            CopyOutputTile(begin, outputLocal, once);
        }
    }

private:
    SquareSumV1TilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

// Stream complete [reduce, inner] rectangles. Each accumulator lane sums
// one residue class of rows; fold those classes once after the last DMA.
template <typename T, bool kPacked = false>
class KernelSquareSumSlab {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const SquareSumV1TilingData& data) {
        data_ = data;
        columns_ = static_cast<uint32_t>(AlignUpElements(data.inner, 32 / sizeof(T)));
        rows_ = 1;
        while (rows_ * 2 * columns_ <= data.rowTile && rows_ < 512) rows_ *= 2;
        count_ = rows_ * columns_;
        parts_ = data.reserved0;
        part_ = AscendC::GetBlockIdx() % parts_;
        owner_ = AscendC::GetBlockIdx() / parts_;
        owners_ = AscendC::GetBlockNum() / parts_;
        partialStride_ = AlignUpElements(data.outputElements, 16);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(input_, 2, count_ * sizeof(T));
        pipe_.InitBuffer(acc_, count_ * sizeof(float));
        pipe_.InitBuffer(scratch_, count_ * sizeof(float));
        if (parts_ > 1 || kPacked) {
            AscendC::SetSysWorkspace(workspace);
            partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
        }
    }
    __aicore__ inline void Process() {
        const uint64_t group = kPacked ? 1 : data_.reduceAligned;
        for (uint64_t begin = owner_ * group; begin < data_.outer; begin += owners_ * group) {
            const uint64_t end = begin + group < data_.outer ? begin + group : data_.outer;
            for (uint64_t outer = begin; outer < end; ++outer) Tile(outer);
        }
        if constexpr (kPacked) {
            AscendC::SyncAll<true>();
            PackOutput();
        } else if (parts_ > 1) {
            AscendC::SyncAll<true>();
            if (part_ == 0) {
                for (uint64_t begin = owner_ * group; begin < data_.outer; begin += owners_ * group) {
                    const uint64_t end = begin + group < data_.outer ? begin + group : data_.outer;
                    for (uint64_t outer = begin; outer < end; ++outer) Finish(outer);
                }
            }
        }
    }
private:
    template <AscendC::HardEvent event>
    __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void Copy(uint64_t outer, uint64_t first, uint32_t rows) {
        auto raw = input_.AllocTensor<T>();
        const uint64_t offset = (outer * data_.reduce + first) * data_.inner;
        if (data_.inner == columns_) {
            AscendC::DataCopy(raw, x_[offset], rows * columns_);
        } else {
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),
                static_cast<uint32_t>(data_.inner * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{true, 0,
                static_cast<uint8_t>(columns_ - data_.inner), static_cast<T>(0)};
            AscendC::DataCopyPad(raw, x_[offset], copy, pad);
        }
        input_.EnQue(raw);
    }
    __aicore__ inline void Compute(uint32_t rows) {
        auto raw = input_.DeQue<T>();
        const uint32_t count = rows * columns_;
        auto sum = acc_.Get<float>();
        AscendC::LocalTensor<float> values;
        if constexpr (std::is_same<T, float>::value) values = raw.template ReinterpretCast<float>();
        else {
            values = scratch_.Get<float>();
            if constexpr (std::is_same<T, half>::value) {
                AscendC::Mul(raw, raw, raw, count);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (!std::is_same<T, half>::value) {
            AscendC::Mul(values, values, values, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(sum, sum, values, count);
        AscendC::PipeBarrier<PIPE_V>();
        input_.FreeTensor(raw);
    }
    __aicore__ inline void Tile(uint64_t outer) {
        auto sum = acc_.Get<float>();
        AscendC::Duplicate(sum, 0.0f, count_);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t chunk = (data_.reduce + parts_ - 1) / parts_;
        const uint64_t begin = part_ * chunk;
        const uint64_t end = begin + chunk < data_.reduce ? begin + chunk : data_.reduce;
        if (begin < end) {
            uint32_t rows = static_cast<uint32_t>(end - begin < rows_ ? end - begin : rows_);
            Copy(outer, begin, rows);
            for (uint64_t first = begin; first < end;) {
                const uint64_t next = first + rows;
                const uint32_t nextRows = static_cast<uint32_t>(end - next < rows_ ? end - next : rows_);
                if (next < end) Copy(outer, next, nextRows);
                Compute(rows);
                first = next; rows = nextRows;
            }
        }
        for (uint32_t halfRows = rows_ / 2; halfRows != 0; halfRows /= 2) {
            AscendC::Add(sum, sum, sum[halfRows * columns_], halfRows * columns_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const uint64_t output = outer * data_.inner;
        if constexpr (kPacked) {
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(partial_[outer * columns_], sum, columns_);
            Sync<AscendC::HardEvent::MTE3_V>();
        } else if (parts_ > 1) {
            AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(data_.inner * 4), 0, 0, 0};
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(partial_[part_ * partialStride_ + output], sum, store);
            Sync<AscendC::HardEvent::MTE3_V>();
        } else Store(output);
    }
    __aicore__ inline void Store(uint64_t output) {
        auto sum = acc_.Get<float>();
        AscendC::DataCopyExtParams store{1, static_cast<uint32_t>(data_.inner * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], sum, store);
        } else {
            auto dst = scratch_.Get<T>();
            AscendC::Cast(dst, sum, AscendC::RoundMode::CAST_RINT, columns_);
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(y_[output], dst, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    __aicore__ inline void Finish(uint64_t outer) {
        auto sum = acc_.Get<float>();
        auto values = scratch_.Get<float>();
        AscendC::Duplicate(sum, 0.0f, columns_);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t output = outer * data_.inner;
        for (uint32_t part = 0; part < parts_; ++part) {
            Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(data_.inner * 4), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{true, 0,
                static_cast<uint8_t>(AlignUpElements(data_.inner, 8) - data_.inner), 0.0f};
            AscendC::DataCopyPad(values, partial_[part * partialStride_ + output], copy, pad);
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::Add(sum, sum, values, static_cast<uint32_t>(AlignUpElements(data_.inner, 8)));
            AscendC::PipeBarrier<PIPE_V>();
        }
        Store(output);
    }
    __aicore__ inline void PackOutput() {
        const uint64_t group = data_.reduceAligned;
        auto values = acc_.Get<float>();
        for (uint64_t begin = owner_ * group; begin < data_.outer; begin += owners_ * group) {
            const uint64_t end = begin + group < data_.outer ? begin + group : data_.outer;
            for (uint64_t first = begin; first < end;) {
                const uint32_t rows = static_cast<uint32_t>(end - first < rows_ ? end - first : rows_);
                Sync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(values, partial_[first * columns_], rows * columns_);
                Sync<AscendC::HardEvent::MTE2_V>();
                AscendC::DataCopyExtParams store{static_cast<uint16_t>(rows),
                    static_cast<uint32_t>(data_.inner * sizeof(T)), 0, 0, 0};
                if constexpr (std::is_same<T, float>::value) {
                    Sync<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopyPad(y_[first * data_.inner], values, store);
                } else {
                    auto out = scratch_.Get<T>();
                    AscendC::Cast(out, values, AscendC::RoundMode::CAST_RINT, rows * columns_);
                    Sync<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopyPad(y_[first * data_.inner], out, store);
                }
                Sync<AscendC::HardEvent::MTE3_V>();
                first += rows;
            }
        }
    }
    SquareSumV1TilingData data_{};
    uint32_t columns_, rows_, count_, parts_, part_, owner_, owners_;
    uint64_t partialStride_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_, scratch_;
    AscendC::GlobalTensor<T> x_, y_;
    AscendC::GlobalTensor<float> partial_;
};

// Consecutive [reduceOuter, output, reduceInner] layout. Read contiguous slabs,
// retain the full output slab in FP32, and reduce its suffix only once.
template <typename T>
class KernelSquareSumRARSlab {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& d) {
        d_ = d;
        needsGather_ = (d.inner % 8) != 0;
        columns_ = static_cast<uint32_t>(d.inner);
        uint32_t log = 0;
        if (needsGather_) { columns_ = 1; while (columns_ < d.inner) { columns_ *= 2; ++log; } }
        const uint32_t unit = 32 / sizeof(T);
        rows_ = static_cast<uint32_t>(d.rowTile / columns_);
        if (rows_ > 240) rows_ = 240;
        rows_ = rows_ / unit * unit;
        count_ = rows_ * columns_;
        chunks_ = static_cast<uint32_t>((d.inner + 63) / 64);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(input_, 2, count_ * sizeof(T));
        pipe_.InitBuffer(acc_, count_ * sizeof(float));
        if (!std::is_same<T, float>::value || needsGather_)
            pipe_.InitBuffer(convert_, count_ * sizeof(float));
        if (needsGather_) {
            pipe_.InitBuffer(indices_, count_ * sizeof(int32_t));
            auto index = indices_.Get<int32_t>(); auto temp = acc_.Get<int32_t>();
            AscendC::CreateVecIndex(index, 0, count_); AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(temp, index, static_cast<int32_t>(log), count_); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(temp, temp, static_cast<int32_t>(d.inner) - static_cast<int32_t>(columns_), count_); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(index, index, temp, count_); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(index, index, static_cast<int32_t>(4), count_); AscendC::PipeBarrier<PIPE_V>();
        }
        pipe_.InitBuffer(stage_, rows_ * chunks_ * 8 * sizeof(float));
        pipe_.InitBuffer(result_, 256 * sizeof(float));
        if constexpr (!std::is_same<T, float>::value)
            pipe_.InitBuffer(output_, 256 * sizeof(T));
    }
    __aicore__ inline void Process() {
        const uint64_t unit = 32 / sizeof(T);
        const uint64_t units = (d_.outputElements + unit - 1) / unit;
        const uint64_t begin = units * AscendC::GetBlockIdx() / AscendC::GetBlockNum() * unit;
        uint64_t end = units * (AscendC::GetBlockIdx() + 1) / AscendC::GetBlockNum() * unit;
        if (end > d_.outputElements) end = d_.outputElements;
        for (uint64_t first = begin; first < end; first += rows_) {
            const uint32_t rows = static_cast<uint32_t>(end - first < rows_ ? end - first : rows_);
            Group(first, rows);
        }
    }
private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void Copy(uint64_t first, uint32_t rows, uint64_t outer) {
        auto raw = input_.AllocTensor<T>();
        const uint64_t offset = (outer * d_.outputElements + first) * d_.inner;
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(rows * d_.inner * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        if (d_.inner % (32 / sizeof(T)) == 0)
            AscendC::DataCopy(raw, x_[offset], static_cast<uint32_t>(rows * d_.inner));
        else AscendC::DataCopyPad(raw, x_[offset], cp, pad);
        input_.EnQue(raw);
    }
    __aicore__ inline void Compute(uint32_t count) {
        auto raw = input_.DeQue<T>();
        auto acc = acc_.Get<float>();
        AscendC::LocalTensor<float> v;
        if constexpr (std::is_same<T, float>::value) v = raw.template ReinterpretCast<float>();
        else {
            v = convert_.Get<float>();
            if constexpr (std::is_same<T, half>::value) {
                AscendC::Mul(raw, raw, raw, count); AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(v, raw, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (!std::is_same<T, half>::value) {
            AscendC::Mul(v, v, v, count); AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(acc, acc, v, count); AscendC::PipeBarrier<PIPE_V>();
        input_.FreeTensor(raw);
    }
    __aicore__ inline void Group(uint64_t first, uint32_t rows) {
        auto acc = acc_.Get<float>();
        auto result = result_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, rows * columns_); AscendC::PipeBarrier<PIPE_V>();
        Copy(first, rows, 0);
        for (uint64_t r = 0; r < d_.outer; ++r) {
            if (r + 1 < d_.outer) Copy(first, rows, r + 1);
            Compute(static_cast<uint32_t>(rows * d_.inner));
        }
        auto reduced = acc;
        if (needsGather_) {
            reduced = convert_.Get<float>();
            auto index = indices_.Get<int32_t>().template ReinterpretCast<uint32_t>();
            AscendC::Gather(reduced, acc, index, 0, rows * columns_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (d_.inner <= 64) {
            AscendC::WholeReduceSum<float>(result, reduced, d_.inner, rows, 1, 1, columns_ / 8);
        } else {
            auto stage = stage_.Get<float>();
            AscendC::Duplicate(stage, 0.0f, rows * chunks_ * 8); AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < chunks_; ++c) {
                const uint32_t start = c * 64;
                const uint32_t width = d_.inner - start < 64 ? d_.inner - start : 64;
                AscendC::WholeReduceSum<float>(stage[c * 8], reduced[start], width, rows, chunks_ * 8, 1, columns_ / 8);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(result, stage, chunks_ * 8, rows, 1, 1, chunks_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams store{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], result, store);
        } else {
            auto out = output_.Get<T>();
            AscendC::Cast(out, result, AscendC::RoundMode::CAST_RINT, rows);
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], out, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    SquareSumV1TilingData d_{};
    uint32_t columns_, rows_, count_, chunks_;
    bool needsGather_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_, convert_, stage_, result_, output_, indices_;
    AscendC::GlobalTensor<T> x_, y_;
};

// Packed storage and 64-column Gather panels preserve the PB reduction order.
template <typename T>
class KernelSquareSumRARPacked {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData& d) {
        d_ = d;
        columns_ = 64;
        const uint32_t unit = 32 / sizeof(T);
        rows_ = static_cast<uint32_t>(d.rowTile / d.inner);
        if (rows_ > 240) rows_ = 240;
        rows_ = rows_ / unit * unit;
        count_ = rows_ * static_cast<uint32_t>(d.inner);
        chunks_ = static_cast<uint32_t>((d.inner + 63) / 64);
        accCount_ = (count_ + chunks_ * 64 - static_cast<uint32_t>(d.inner) + 7) / 8 * 8;
        indexCount_ = rows_ * 64;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(input_, 2, count_ * sizeof(T));
        pipe_.InitBuffer(acc_, accCount_ * sizeof(float));
        pipe_.InitBuffer(convert_, count_ * sizeof(float));
        pipe_.InitBuffer(indices_, indexCount_ * sizeof(int32_t));
        auto index = indices_.Get<int32_t>();
        auto temp = convert_.Get<int32_t>();
        AscendC::CreateVecIndex(index, 0, indexCount_); AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(temp, index, static_cast<int32_t>(6), indexCount_); AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temp, temp, static_cast<int32_t>(d.inner) - 64, indexCount_); AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(index, index, temp, indexCount_); AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(index, index, static_cast<int32_t>(4), indexCount_); AscendC::PipeBarrier<PIPE_V>();
        pipe_.InitBuffer(stage_, rows_ * chunks_ * 8 * sizeof(float));
        pipe_.InitBuffer(result_, 256 * sizeof(float));
        if constexpr (!std::is_same<T, float>::value)
            pipe_.InitBuffer(output_, 256 * sizeof(T));
    }
    __aicore__ inline void Process() {
        const uint64_t unit = 32 / sizeof(T);
        const uint64_t units = (d_.outputElements + unit - 1) / unit;
        const uint64_t begin = units * AscendC::GetBlockIdx() / AscendC::GetBlockNum() * unit;
        uint64_t end = units * (AscendC::GetBlockIdx() + 1) / AscendC::GetBlockNum() * unit;
        if (end > d_.outputElements) end = d_.outputElements;
        for (uint64_t first = begin; first < end; first += rows_) {
            const uint32_t rows = static_cast<uint32_t>(end - first < rows_ ? end - first : rows_);
            Group(first, rows);
        }
    }
private:
    template <AscendC::HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void Copy(uint64_t first, uint32_t rows, uint64_t outer) {
        auto raw = input_.AllocTensor<T>();
        const uint64_t offset = (outer * d_.outputElements + first) * d_.inner;
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(rows * d_.inner * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        if (d_.inner % (32 / sizeof(T)) == 0)
            AscendC::DataCopy(raw, x_[offset], static_cast<uint32_t>(rows * d_.inner));
        else AscendC::DataCopyPad(raw, x_[offset], cp, pad);
        input_.EnQue(raw);
    }
    __aicore__ inline void Compute(uint32_t count) {
        auto raw = input_.DeQue<T>();
        auto acc = acc_.Get<float>();
        AscendC::LocalTensor<float> v;
        if constexpr (std::is_same<T, float>::value) v = raw.template ReinterpretCast<float>();
        else {
            v = convert_.Get<float>();
            if constexpr (std::is_same<T, half>::value) {
                AscendC::Mul(raw, raw, raw, count); AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(v, raw, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (!std::is_same<T, half>::value) {
            AscendC::Mul(v, v, v, count); AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(acc, acc, v, count); AscendC::PipeBarrier<PIPE_V>();
        input_.FreeTensor(raw);
    }
    __aicore__ inline void Group(uint64_t first, uint32_t rows) {
        auto acc = acc_.Get<float>();
        auto result = result_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, accCount_); AscendC::PipeBarrier<PIPE_V>();
        Copy(first, rows, 0);
        for (uint64_t r = 0; r < d_.outer; ++r) {
            if (r + 1 < d_.outer) Copy(first, rows, r + 1);
            Compute(static_cast<uint32_t>(rows * d_.inner));
        }
        auto reduced = convert_.Get<float>();
        auto stage = stage_.Get<float>();
        auto index = indices_.Get<int32_t>().template ReinterpretCast<uint32_t>();
        AscendC::Duplicate(stage, 0.0f, rows * chunks_ * 8); AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < chunks_; ++c) {
            const uint32_t start = c * 64;
            const uint32_t width = d_.inner - start < 64 ? d_.inner - start : 64;
            AscendC::Gather(reduced, acc[start], index, 0, rows * 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(stage[c * 8], reduced, width, rows, chunks_ * 8, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum<float>(result, stage, chunks_ * 8, rows, 1, 1, chunks_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams store{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], result, store);
        } else {
            auto out = output_.Get<T>();
            AscendC::Cast(out, result, AscendC::RoundMode::CAST_RINT, rows);
            Sync<AscendC::HardEvent::V_MTE3>(); AscendC::DataCopyPad(y_[first], out, store);
        }
        Sync<AscendC::HardEvent::MTE3_V>();
    }
    SquareSumV1TilingData d_{};
    uint32_t columns_, rows_, count_, chunks_, accCount_, indexCount_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_, convert_, stage_, result_, output_, indices_;
    AscendC::GlobalTensor<T> x_, y_;
};


template<uint32_t Width> class KernelShortRow {
 static constexpr int HalfOnly=0;
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const SquareSumV1TilingData& d){outputs_=d.outputElements;reduce_=d.reduce;x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));pipe_.InitBuffer(raw_,64*Width*2);if constexpr(!HalfOnly)pipe_.InitBuffer(values_,64*Width*4);pipe_.InitBuffer(result_,64*4);}
 __aicore__ inline void Process(){const uint64_t units=(outputs_+15)/16;uint64_t first=units*AscendC::GetBlockIdx()/AscendC::GetBlockNum()*16;uint64_t end=units*(AscendC::GetBlockIdx()+1)/AscendC::GetBlockNum()*16;if(end>outputs_)end=outputs_;
  for(;first<end;first+=64){const uint32_t rows=end-first<64?end-first:64;auto raw=raw_.Get<half>();AscendC::DataCopyExtParams load{static_cast<uint16_t>(rows),static_cast<uint32_t>(reduce_*2),0,0,0};AscendC::DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(Width-reduce_),static_cast<half>(0)};AscendC::DataCopyPad(raw,x_[first*reduce_],load,pad);Sync<AscendC::HardEvent::MTE2_V>();AscendC::Mul(raw,raw,raw,rows*Width);AscendC::PipeBarrier<PIPE_V>();
   if constexpr(HalfOnly){auto out=result_.Get<half>();AscendC::WholeReduceSum<half>(out,raw,Width,rows,1,1,Width/16);Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],out,store);}
   else{auto v=values_.Get<float>();auto result=result_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,rows*Width);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum<float>(result,v,Width,rows,1,1,Width/8);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(raw,result,AscendC::RoundMode::CAST_RINT,rows);Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],raw,store);}
   Sync<AscendC::HardEvent::MTE3_V>();
  }
 }
private:
 template<AscendC::HardEvent E>__aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 uint64_t outputs_,reduce_;AscendC::TPipe pipe_;AscendC::TBuf<AscendC::TPosition::VECCALC>raw_,values_,result_;AscendC::GlobalTensor<half>x_,y_;
};
template<typename T,uint32_t Base> class KernelPowerBoundaryAr {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const SquareSumV1TilingData& d){
  d_=d;rows_=Base>=256?16384/Base:64;cols_=rows_;
  maxN_=rows_*(Base+1);matrix_=(rows_+1)*cols_;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));if((d_.reserved1&256U))x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(q_,1,AlignUpElements(maxN_,16)*2);pipe_.InitBuffer(outQ_,2,AlignUpElements(rows_,16)*2);
  pipe_.InitBuffer(values_,AlignUpElements(maxN_+cols_+8,8)*4);pipe_.InitBuffer(boundary_,matrix_*4);pipe_.InitBuffer(tailMask_,matrix_*4);pipe_.InitBuffer(keepMask_,matrix_*4);
  pipe_.InitBuffer(partial_,rows_*(Base>=512?Base/8:64)*4);pipe_.InitBuffer(base_,AlignUpElements(rows_,8)*4);pipe_.InitBuffer(prefix_,AlignUpElements(rows_+1,8)*4);pipe_.InitBuffer(shift_,AlignUpElements(rows_,8)*4);
  auto tail=tailMask_.Get<int32_t>();auto keep=keepMask_.Get<int32_t>();AscendC::Duplicate(tail,static_cast<int32_t>(0),matrix_);AscendC::Duplicate(keep,static_cast<int32_t>(-1),matrix_);AscendC::PipeBarrier<PIPE_V>();for(uint32_t j=1;j<=rows_;++j){AscendC::Duplicate(tail[j*cols_],static_cast<int32_t>(-1),j);AscendC::Duplicate(keep[j*cols_],static_cast<int32_t>(0),j);}AscendC::PipeBarrier<PIPE_V>();
  auto shift=shift_.Get<int32_t>();AscendC::CreateVecIndex(shift,1,rows_);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(shift,shift,static_cast<int32_t>(4),rows_);AscendC::PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){const uint64_t units=(d_.outputElements+15)/16;const uint64_t begin=units*AscendC::GetBlockIdx()/AscendC::GetBlockNum()*16;uint64_t end=units*(AscendC::GetBlockIdx()+1)/AscendC::GetBlockNum()*16;if(end>d_.outputElements)end=d_.outputElements;if(begin>=end)return;
  uint32_t rows=end-begin<rows_?end-begin:rows_;Copy(begin,rows);for(uint64_t first=begin;first<end;){const uint64_t next=first+rows;uint32_t nr=end-next<rows_?end-next:rows_;Compute(first,rows,next,next<end?nr:0);first=next;rows=nr;}
 }
private:
 __aicore__ inline void Copy(uint64_t first,uint32_t rows){auto raw=q_.AllocTensor<T>();AscendC::DataCopyExtParams cp{1,rows*(Base+1)*2,0,0,0};AscendC::DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};if((first*(Base+1))%16==0 && (rows*(Base+1))%16==0)AscendC::DataCopy(raw,x_[first*(Base+1)],rows*(Base+1));else AscendC::DataCopyPad(raw,x_[first*(Base+1)],cp,pad);q_.EnQue(raw);}
 __aicore__ inline void Compute(uint64_t first,uint32_t rows,uint64_t next,uint32_t nextRows){auto raw=q_.DeQue<T>();auto v=values_.Get<float>();const uint32_t n=rows*(Base+1);AscendC::Duplicate(v[n/8*8],0.0f,cols_+8);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();q_.FreeTensor(raw);if(nextRows)Copy(next,nextRows);
  // Square once before the main and boundary reduction paths diverge.
  if constexpr(Base>=512){AscendC::Mul(v,v,v,n);AscendC::PipeBarrier<PIPE_V>();}
  // Save the shifted tail before clearing values that belong to the preceding row.
  auto b=boundary_.Get<float>();AscendC::Adds(b,v,0.0f,static_cast<uint64_t>(cols_),static_cast<uint8_t>(rows+1),AscendC::UnaryRepeatParams{1,1,static_cast<uint8_t>(cols_/8),Base/8});AscendC::PipeBarrier<PIPE_V>();
  auto bits=b.template ReinterpretCast<uint16_t>();auto valuesBits=v.template ReinterpretCast<uint16_t>();
  AscendC::And(bits,bits,tailMask_.Get<uint16_t>(),(rows+1)*cols_*2);
  AscendC::And(valuesBits,valuesBits,keepMask_.Get<uint16_t>(),static_cast<uint64_t>(cols_*2),static_cast<uint8_t>(rows),AscendC::BinaryRepeatParams{1,1,1,Base/8,Base/8,static_cast<uint8_t>(cols_/8)});
  AscendC::PipeBarrier<PIPE_V>();if constexpr(Base<512){AscendC::Mul(b,b,b,(rows+1)*cols_);AscendC::PipeBarrier<PIPE_V>();}auto pref=prefix_.Get<float>();AscendC::WholeReduceSum<float>(pref,b,cols_,rows+1,1,1,cols_/8);AscendC::PipeBarrier<PIPE_V>();
  auto part=partial_.Get<float>();auto base=base_.Get<float>();
  if constexpr(Base>=512){
   const uint32_t repeats=rows*(Base/64),head=repeats<255?repeats:255;
   AscendC::BlockReduceSum<float>(part,v,head,64,1,1,8);
   if(repeats>255)AscendC::BlockReduceSum<float>(part[255*8],v[255*64],repeats-255,64,1,1,8);
   AscendC::PipeBarrier<PIPE_V>();
   AscendC::BlockReduceSum<float>(v,part,rows*(Base/512),64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
   AscendC::WholeReduceSum<float>(base,v,Base/64,rows,1,1,Base/512);AscendC::PipeBarrier<PIPE_V>();
  }else{
AscendC::Duplicate(part,0.0f,rows*64);AscendC::PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t block=0;block<Base/64;++block){AscendC::MulAddDst(part,v[block*64],v[block*64],static_cast<uint64_t>(64),static_cast<uint8_t>(rows),AscendC::BinaryRepeatParams{1,1,1,8,Base/8,Base/8});AscendC::PipeBarrier<PIPE_V>();}
AscendC::WholeReduceSum<float>(base,part,64,rows,1,1,8);AscendC::PipeBarrier<PIPE_V>();
  }AscendC::Gather(b,pref,shift_.Get<uint32_t>(),0,rows);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(base,base,b,rows);AscendC::PipeBarrier<PIPE_V>();
  auto out=outQ_.AllocTensor<T>();AscendC::Cast(out,base,AscendC::RoundMode::CAST_RINT,rows);outQ_.EnQue(out);out=outQ_.DeQue<T>();AscendC::DataCopyExtParams cp{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],out,cp);outQ_.FreeTensor(out);
 }
 SquareSumV1TilingData d_{};uint32_t rows_,cols_,maxN_,matrix_;AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,1>q_;AscendC::TQue<AscendC::QuePosition::VECOUT,2>outQ_;
 AscendC::TBuf<AscendC::TPosition::VECCALC>values_,boundary_,tailMask_,keepMask_,partial_,base_,prefix_,shift_;AscendC::GlobalTensor<T>x_,y_;
};
template<typename T, int Method, int Buffers>
class KernelPackedReduction {
public:
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& d) {
  d_=d; ar_=d.rowTile; rr_=d.tileElements; n_=ar_*d.inner; readN_=rr_*d.inner;
  columns_=AlignUpElements(d.inner,32/sizeof(T));parts_=d.reserved0;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
  if((d_.reserved1&256U)) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  AscendC::SetSysWorkspace(workspace);w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
  pipe_.InitBuffer(q_,Buffers,AlignUpElements(readN_*sizeof(T),32));
  accElems_=AlignUpElements(n_+16,8);
  if constexpr(!std::is_same<T,float>::value){if(accElems_<16*columns_)accElems_=16*columns_;}
  pipe_.InitBuffer(acc_,accElems_*4);
  convertElems_=AlignUpElements(readN_,8);if(convertElems_<16*columns_)convertElems_=16*columns_;
  if constexpr(!std::is_same<T,float>::value)pipe_.InitBuffer(convert_,convertElems_*4);
  pipe_.InitBuffer(partOut_,2,columns_*4);
  pipe_.InitBuffer(gather_,8*columns_*4);pipe_.InitBuffer(index_,8*columns_*4);
  auto index=index_.Get<int32_t>();
  for(uint32_t r=0;r<8;++r){AscendC::CreateVecIndex(index[r*columns_],static_cast<int32_t>(r*d.inner),columns_);}
  AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(index,index,static_cast<int32_t>(4),8*columns_);AscendC::PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  for(uint64_t task=AscendC::GetBlockIdx();task<d_.outer*parts_;task+=AscendC::GetBlockNum()){
   const uint64_t o=task/parts_;uint32_t part=task%parts_;
   if(d_.reserved1&2U)part=(part+o+o/7)%parts_;Tile(o,part);
  }
  Sync<AscendC::HardEvent::MTE3_S>();
  AscendC::SyncAll<true>();
  const uint64_t group=d_.reduceAligned;
  for(uint64_t first=AscendC::GetBlockIdx()*group;first<d_.outer;first+=AscendC::GetBlockNum()*group){
   uint64_t end=first+group<d_.outer?first+group:d_.outer;
   if constexpr(std::is_same<T,float>::value){for(uint64_t outer=first;outer<end;++outer)FinishBulk(outer);}
   else{for(uint64_t o=first;o<end;){uint32_t count=end-o;uint32_t cap=accElems_/columns_;if(cap>16)cap=16;if(cap>convertElems_/columns_)cap=readN_/columns_;if(count>cap)count=cap;FinishGroup(o,count);o+=count;}}
  }
 }
private:
 template<AscendC::HardEvent E> __aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 __aicore__ inline void Copy(uint64_t o,uint64_t first,uint32_t rows){auto t=q_.template AllocTensor<T>();
  AscendC::DataCopyExtParams cp{1,static_cast<uint32_t>(rows*d_.inner*sizeof(T)),0,0,0};
  AscendC::DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  S9PackedCopy(t,x_,(o*d_.reduce+first)*d_.inner,rows*d_.inner,(d_.reserved1&512U)!=0);q_.EnQue(t);
 }
 __aicore__ inline void Compute(uint32_t rows,uint32_t accOffset){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();AscendC::LocalTensor<float> v;
  const uint32_t count=rows*d_.inner;
  if constexpr(std::is_same<T,float>::value)v=raw.template ReinterpretCast<float>();
  else{v=convert_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();}
  if constexpr(Method==0){AscendC::Mul(v,v,v,count);AscendC::PipeBarrier<PIPE_V>();}
  for(uint32_t start=0;start<count;){const uint32_t len=count-start<n_-accOffset?count-start:n_-accOffset;
   if constexpr(Method==1)AscendC::MulAddDst(acc[accOffset],v[start],v[start],len);
   else AscendC::Add(acc[accOffset],acc[accOffset],v[start],len);
   AscendC::PipeBarrier<PIPE_V>(); start+=len;accOffset=0;
  }
  q_.FreeTensor(raw);
 }
 __aicore__ inline void Tile(uint64_t o,uint32_t part){auto acc=acc_.Get<float>();AscendC::Duplicate(acc,0.0f,n_+16);AscendC::PipeBarrier<PIPE_V>();
  const bool cyclic=d_.reserved1&1U;
  const uint64_t totalTiles=(d_.reduce+rr_-1)/rr_;
  const uint64_t firstTile=totalTiles*part/parts_;
  const uint64_t endTile=totalTiles*(part+1)/parts_;
  const uint64_t begin=cyclic?part*rr_:firstTile*rr_;
  const uint64_t end=cyclic?d_.reduce:(endTile*rr_<d_.reduce?endTile*rr_:d_.reduce);
  const uint64_t step=cyclic?parts_*rr_:rr_;
  if constexpr(Buffers==1){
   uint64_t localRows=0;for(uint64_t first=begin;first<end;first+=step){const uint32_t rows=end-first<rr_?end-first:rr_;Copy(o,first,rows);Compute(rows,(d_.reserved1&4U)?0:(localRows%ar_)*d_.inner);localRows+=rows;}
  }else{
  if(begin<end){uint32_t rows=end-begin<rr_?end-begin:rr_;Copy(o,begin,rows);uint64_t localRows=0;
   for(uint64_t first=begin;first<end;){uint64_t next=first+step;uint32_t nr=next<end?(end-next<rr_?end-next:rr_):0;
    if(next<end)Copy(o,next,nr);Compute(rows,(d_.reserved1&4U)?0:(localRows%ar_)*d_.inner);localRows+=rows;first=next;rows=nr;
   }
  }
  }
  for(uint32_t h=ar_/2;h>=8;h/=2){AscendC::Add(acc,acc,acc[h*d_.inner],h*d_.inner);AscendC::PipeBarrier<PIPE_V>();}
  auto g=gather_.Get<float>();auto index=index_.Get<uint32_t>();AscendC::Gather(g,acc,index,0,8*columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t h=4;h;h/=2){AscendC::Add(g,g,g[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
  auto partOut=partOut_.AllocTensor<float>();AscendC::DataCopy(partOut,g,columns_);partOut_.EnQue(partOut);partOut=partOut_.DeQue<float>();AscendC::DataCopy(w_[(o*parts_+part)*columns_],partOut,columns_);partOut_.FreeTensor(partOut);
 }
 __aicore__ inline void FinishBulk(uint64_t o){
  auto scratch=q_.template AllocTensor<T>();q_.EnQue(scratch);scratch=q_.template DeQue<T>();
  auto values=acc_.Get<float>();uint32_t capacity=accElems_;if(readN_>capacity){values=scratch.template ReinterpretCast<float>();capacity=readN_;}
  auto total=gather_.Get<float>();AscendC::Duplicate(total,0.0f,columns_);AscendC::PipeBarrier<PIPE_V>();
  uint32_t cap=capacity/columns_;if(cap>64)cap=64;
  for(uint32_t first=0;first<parts_;first+=cap){const uint32_t count=parts_-first<cap?parts_-first:cap;
   Sync<AscendC::HardEvent::V_MTE2>();AscendC::DataCopy(values,w_[(o*parts_+first)*columns_],count*columns_);Sync<AscendC::HardEvent::MTE2_V>();
   uint32_t base=1;while(base*2<=count)base*=2;const uint32_t tail=count-base;
   if(tail){AscendC::Add(values,values,values[base*columns_],tail*columns_);AscendC::PipeBarrier<PIPE_V>();}
   for(uint32_t h=base/2;h;h/=2){AscendC::Add(values,values,values[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
   AscendC::Add(total,total,values,columns_);AscendC::PipeBarrier<PIPE_V>();
  }
  AscendC::DataCopyExtParams store{1,static_cast<uint32_t>(d_.inner*4),0,0,0};Sync<AscendC::HardEvent::V_MTE3>();
  if constexpr(std::is_same<T,float>::value)AscendC::DataCopyPad(y_[o*d_.inner],total,store);
  Sync<AscendC::HardEvent::MTE3_V>();q_.FreeTensor(scratch);
 }
 __aicore__ inline void FinishGroup(uint64_t first,uint32_t rows){
  auto sum=acc_.Get<float>();auto values=convert_.Get<float>();AscendC::Duplicate(sum,0.0f,rows*columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t part=0;part<parts_;++part){
   AscendC::DataCopyExtParams load{static_cast<uint16_t>(rows),columns_*4,(parts_-1)*columns_*4,0,0};AscendC::DataCopyPadExtParams<float> pad{false,0,0,0.0f};
   Sync<AscendC::HardEvent::V_MTE2>();AscendC::DataCopyPad(values,w_[(first*parts_+part)*columns_],load,pad);Sync<AscendC::HardEvent::MTE2_V>();
   AscendC::Add(sum,sum,values,rows*columns_);AscendC::PipeBarrier<PIPE_V>();
  }
  if constexpr(!std::is_same<T,float>::value){
   auto out=gather_.Get<T>();AscendC::Cast(out,sum,AscendC::RoundMode::CAST_RINT,rows*columns_);Sync<AscendC::HardEvent::V_MTE3>();
   AscendC::DataCopyExtParams store{static_cast<uint16_t>(rows),static_cast<uint32_t>(d_.inner*sizeof(T)),0,0,0};AscendC::DataCopyPad(y_[first*d_.inner],out,store);Sync<AscendC::HardEvent::MTE3_V>();
  }
 }
 SquareSumV1TilingData d_{};uint32_t ar_,rr_,n_,readN_,columns_,parts_,accElems_,convertElems_;
 AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,Buffers>q_;
 AscendC::TQue<AscendC::QuePosition::VECOUT,2>partOut_;
 AscendC::TBuf<AscendC::TPosition::VECCALC>acc_,convert_,gather_,index_;
 AscendC::GlobalTensor<T>x_,y_;AscendC::GlobalTensor<float>w_;
};
template<uint32_t AccumulationSpan> class KernelMaskedRar {
 static constexpr int Banks=1;
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const SquareSumV1TilingData& d){
  d_=d;rows_=d.rowTile/d.inner;count_=rows_*d.inner;accCount_=AlignUpElements(count_+128,16);pitch_=AlignUpElements((rows_+15)/16,16);
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));if((d_.reserved1&256U))x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(q_,2,AlignUpElements(count_,16)*2);pipe_.InitBuffer(acc_,accCount_*2);if constexpr(Banks==2)pipe_.InitBuffer(acc1_,accCount_*2);
  pipe_.InitBuffer(index_,AlignUpElements(rows_,8)*4);pipe_.InitBuffer(partial_,16*pitch_*2);pipe_.InitBuffer(result_,AlignUpElements(rows_,16)*2);pipe_.InitBuffer(floatSum_,2*AlignUpElements(rows_,8)*4);
  auto idx=index_.Get<int32_t>();auto tmp=acc_.Get<int32_t>();AscendC::CreateVecIndex(idx,0,rows_);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftRight(tmp,idx,static_cast<int32_t>(4),rows_);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(idx,idx,static_cast<int32_t>(pitch_),rows_);AscendC::Muls(tmp,tmp,static_cast<int32_t>(1-16*pitch_),rows_);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(idx,idx,tmp,rows_);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(idx,idx,static_cast<int32_t>(2),rows_);AscendC::PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  const uint64_t blocks=AscendC::GetBlockNum(),id=AscendC::GetBlockIdx();const uint64_t roundSize=blocks*rows_;const uint64_t rounds=d_.outputElements/roundSize;
  for(uint64_t r=0;r<rounds;++r)Group((r*blocks+id)*rows_,rows_);
  const uint64_t done=rounds*roundSize;const uint64_t units=(d_.outputElements-done+15)/16;
  const uint64_t begin=done+units*id/blocks*16;uint64_t end=done+units*(id+1)/blocks*16;if(end>d_.outputElements)end=d_.outputElements;
  if(begin<end)Group(begin,static_cast<uint32_t>(end-begin));
 }

private:
 __aicore__ inline uint64_t LowBits(uint32_t n){return n>=64?~0ULL:n?((1ULL<<n)-1ULL):0ULL;}
 template<AscendC::HardEvent E>__aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 __aicore__ inline void Copy(uint64_t first,uint32_t rows,uint32_t a){auto raw=q_.AllocTensor<half>();AscendC::DataCopyExtParams cp{1,static_cast<uint32_t>(rows*d_.inner*2),0,0,0};AscendC::DataCopyPadExtParams<half> pad{false,0,0,static_cast<half>(0)};S9PackedCopy(raw,x_,(a*d_.outputElements+first)*d_.inner,rows*d_.inner,d_.shape[6]!=0);q_.EnQue(raw);}
 __aicore__ inline void Compute(uint32_t rows,uint32_t a){auto raw=q_.DeQue<half>();auto sum=acc_.Get<half>();if constexpr(Banks==2){if(a>=(d_.outer+1)/2)sum=acc1_.Get<half>();}AscendC::MulAddDst(sum,raw,raw,rows*d_.inner);AscendC::PipeBarrier<PIPE_V>();q_.FreeTensor(raw);}
 __aicore__ inline void ReduceChunk(uint32_t rows){
  auto sum=acc_.Get<half>();auto temp=partial_.Get<half>();
  for(uint32_t j=0;j<16 && j<rows;++j){const uint32_t offset=j*d_.inner,low=offset%16,repeat=(rows-j+15)/16;const uint64_t mask[2]={~0ULL<<low,LowBits(d_.inner+low-64)};AscendC::WholeReduceSum<half>(temp[j*pitch_],sum[offset-low],mask,repeat,1,1,d_.inner);}
  AscendC::PipeBarrier<PIPE_V>();auto out=result_.Get<half>();AscendC::Gather(out,temp,index_.Get<uint32_t>(),0,rows);AscendC::PipeBarrier<PIPE_V>();
  auto total=floatSum_.Get<float>();auto part=total[AlignUpElements(rows_,8)];AscendC::Cast(part,out,AscendC::RoundMode::CAST_NONE,rows);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(total,total,part,rows);AscendC::PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Group(uint64_t first,uint32_t rows){
  auto sum=acc_.Get<half>();auto total=floatSum_.Get<float>();AscendC::Duplicate(total,0.0f,rows);AscendC::Duplicate(sum,static_cast<half>(0),accCount_);AscendC::PipeBarrier<PIPE_V>();
  Copy(first,rows,0);
  for(uint32_t a=0;a<d_.outer;++a){if(a+1<d_.outer)Copy(first,rows,a+1);Compute(rows,a);if((a+1)%AccumulationSpan==0 || a+1==d_.outer){ReduceChunk(rows);if(a+1<d_.outer){AscendC::Duplicate(sum,static_cast<half>(0),accCount_);AscendC::PipeBarrier<PIPE_V>();}}}
  auto out=result_.Get<half>();AscendC::Cast(out,total,AscendC::RoundMode::CAST_RINT,rows);Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],out,store);Sync<AscendC::HardEvent::MTE3_V>();
 }

 SquareSumV1TilingData d_{};uint32_t rows_,count_,accCount_,pitch_;AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,2>q_;AscendC::TBuf<AscendC::TPosition::VECCALC>acc_,acc1_,index_,partial_,result_,floatSum_;AscendC::GlobalTensor<half>x_,y_;
};
template<typename T, int Method, int Buffers>
class KernelStreamingPrefix {
public:
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& d) {
  d_=d; ar_=d.rowTile; rr_=d.tileElements; n_=ar_*d.inner; readN_=rr_;
  columns_=AlignUpElements(d.inner,32/sizeof(T));parts_=d.reserved0;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
  if(d_.reserved1&256U) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  AscendC::SetSysWorkspace(workspace);w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));
  pipe_.InitBuffer(q_,Buffers,AlignUpElements(readN_*sizeof(T),32));
  accElems_=AlignUpElements(n_+16,8);
  if constexpr(!std::is_same<T,float>::value){if(accElems_<16*columns_)accElems_=16*columns_;}
  pipe_.InitBuffer(acc_,accElems_*4);
  convertElems_=AlignUpElements(readN_,8);if(convertElems_<16*columns_)convertElems_=16*columns_;
  if constexpr(!std::is_same<T,float>::value)pipe_.InitBuffer(convert_,convertElems_*4);
  pipe_.InitBuffer(partOut_,1,columns_*4);
 }
 __aicore__ inline void Process(){
  for(uint64_t task=AscendC::GetBlockIdx();task<d_.outer*parts_;task+=AscendC::GetBlockNum()){
   const uint64_t o=task/parts_;uint32_t part=task%parts_;
   if(d_.reserved1&2U)part=(part+o+o/7)%parts_;Tile(o,part);
  }
  Sync<AscendC::HardEvent::MTE3_S>();
  AscendC::SyncAll<true>();
  FinishCooperative();return;
  const uint64_t group=d_.reduceAligned;
  for(uint64_t first=AscendC::GetBlockIdx()*group;first<d_.outer;first+=AscendC::GetBlockNum()*group){
   uint64_t end=first+group<d_.outer?first+group:d_.outer;
   if constexpr(std::is_same<T,float>::value){for(uint64_t outer=first;outer<end;++outer)FinishBulk(outer);}
   else{for(uint64_t o=first;o<end;){uint32_t count=end-o;uint32_t cap=accElems_/columns_;if(cap>16)cap=16;if(cap>convertElems_/columns_)cap=readN_/columns_;if(count>cap)count=cap;FinishGroup(o,count);o+=count;}}
  }
 }
private:
 template<AscendC::HardEvent E> __aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 __aicore__ inline void CopyFlat(uint64_t first,uint32_t count){auto t=q_.template AllocTensor<T>();S9PackedCopy(t,x_,first,count,true);q_.EnQue(t);}
 __aicore__ inline void Compute(uint32_t count,uint32_t accOffset){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();AscendC::LocalTensor<float> v;
  if constexpr(std::is_same<T,float>::value)v=raw.template ReinterpretCast<float>();
  else{v=convert_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();}
  if constexpr(Method==0){AscendC::Mul(v,v,v,count);AscendC::PipeBarrier<PIPE_V>();}
  for(uint32_t start=0;start<count;){const uint32_t len=count-start<n_-accOffset?count-start:n_-accOffset;
   if constexpr(Method==1)AscendC::MulAddDst(acc[accOffset],v[start],v[start],len);
   else AscendC::Add(acc[accOffset],acc[accOffset],v[start],len);
   AscendC::PipeBarrier<PIPE_V>(); start+=len;accOffset=0;
  }
  q_.FreeTensor(raw);
 }
 __aicore__ inline void Descriptor(uint32_t part,uint64_t ordinal,uint64_t begin,uint64_t end,uint64_t totalTiles,uint64_t& first,uint32_t& count){const uint32_t mode=d_.reserved1&3U;
  if(mode&1U){const uint64_t tasks=part<totalTiles?(totalTiles-part+parts_-1)/parts_:0;uint64_t tile=part+ordinal*parts_;if(mode&2U)tile=totalTiles-1-tile;first=tile*readN_;count=d_.totalElements-first<readN_?d_.totalElements-first:readN_;}
  else if(mode&2U){const uint32_t tail=end%8;if(tail && ordinal==0){first=end-tail;count=tail;}else{const uint64_t k=ordinal-(tail?1:0);const uint64_t stop=end-tail-k*readN_;count=stop-begin<readN_?stop-begin:readN_;first=stop-count;}}
  else{first=begin+ordinal*readN_;count=end-first<readN_?end-first:readN_;}
 }
 __aicore__ inline void Tile(uint64_t o,uint32_t part){auto acc=acc_.Get<float>();AscendC::Duplicate(acc,0.0f,n_+16);AscendC::PipeBarrier<PIPE_V>();const uint32_t mode=d_.reserved1&3U;const uint64_t units=(d_.totalElements+7)/8;const uint64_t begin=units*part/parts_*8;uint64_t end=units*(part+1)/parts_*8;if(end>d_.totalElements)end=d_.totalElements;const uint64_t totalTiles=(d_.totalElements+readN_-1)/readN_;uint64_t tasks=0;
  if(mode&1U)tasks=part<totalTiles?(totalTiles-part+parts_-1)/parts_:0;
  else if(end>begin){if(mode&2U){const uint64_t span=end-end%8-begin;tasks=(span+readN_-1)/readN_+(end%8?1:0);}else tasks=(end-begin+readN_-1)/readN_;}
  for(uint64_t i=0;i<tasks && i<Buffers;++i){uint64_t first;uint32_t count;Descriptor(part,i,begin,end,totalTiles,first,count);CopyFlat(first,count);}
  for(uint64_t i=0;i<tasks;++i){uint64_t first;uint32_t count;Descriptor(part,i,begin,end,totalTiles,first,count);Compute(count,static_cast<uint32_t>(first)%n_);if(i+Buffers<tasks){Descriptor(part,i+Buffers,begin,end,totalTiles,first,count);CopyFlat(first,count);}}
  for(uint32_t h=ar_/2;h>=8;h/=2){AscendC::Add(acc,acc,acc[h*d_.inner],h*d_.inner);AscendC::PipeBarrier<PIPE_V>();}
  auto scratch=q_.template AllocTensor<T>();q_.EnQue(scratch);scratch=q_.template DeQue<T>();auto g=scratch.template ReinterpretCast<float>();auto ix=scratch.template ReinterpretCast<int32_t>()[8*columns_];
  // Reuse one row of byte offsets; Gather base shifts preserve the original mapping.
  AscendC::CreateVecIndex(ix,static_cast<int32_t>(0),columns_);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(ix,ix,static_cast<int32_t>(4),columns_);AscendC::PipeBarrier<PIPE_V>();auto index=ix.template ReinterpretCast<uint32_t>();
  for(uint32_t r=0;r<8;++r)AscendC::Gather(g[r*columns_],acc,index,r*d_.inner*4,columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t h=4;h;h/=2){AscendC::Add(g,g,g[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
  auto partOut=partOut_.AllocTensor<float>();AscendC::DataCopy(partOut,g,columns_);partOut_.EnQue(partOut);partOut=partOut_.DeQue<float>();AscendC::DataCopy(w_[(o*parts_+part)*columns_],partOut,columns_);partOut_.FreeTensor(partOut);q_.FreeTensor(scratch);
 }
 __aicore__ inline void FinishCooperative(){
  const uint32_t width=64;
  for(uint64_t first=AscendC::GetBlockIdx()*width;first<d_.inner;first+=AscendC::GetBlockNum()*width){
   const uint32_t valid=d_.inner-first<width?d_.inner-first:width;
   auto raw=q_.template AllocTensor<T>();q_.EnQue(raw);raw=q_.template DeQue<T>();auto values=raw.template ReinterpretCast<float>();
   AscendC::Duplicate(values,0.0f,parts_*width);Sync<AscendC::HardEvent::V_MTE2>();
   const uint32_t aligned=AlignUpElements(valid,8);
   AscendC::DataCopyExtParams load{static_cast<uint16_t>(parts_),valid*4,static_cast<uint32_t>((columns_-valid)*4),static_cast<uint32_t>((width-aligned)/8),0};
   AscendC::DataCopyPadExtParams<float> pad{true,0,static_cast<uint8_t>(aligned-valid),0.0f};
   AscendC::DataCopyPad(values,w_[first],load,pad);Sync<AscendC::HardEvent::MTE2_V>();
   uint32_t base=1;while(base*2<=parts_)base*=2;const uint32_t tail=parts_-base;
   if(tail){AscendC::Add(values,values,values[base*width],tail*width);AscendC::PipeBarrier<PIPE_V>();}
   for(uint32_t h=base/2;h;h/=2){AscendC::Add(values,values,values[h*width],h*width);AscendC::PipeBarrier<PIPE_V>();}
   Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,valid*4,0,0,0};AscendC::DataCopyPad(y_[first],values,store);Sync<AscendC::HardEvent::MTE3_V>();q_.FreeTensor(raw);
  }
 }
 __aicore__ inline void FinishBulk(uint64_t o){
  auto scratch=q_.template AllocTensor<T>();q_.EnQue(scratch);scratch=q_.template DeQue<T>();
  auto values=scratch.template ReinterpretCast<float>();uint32_t capacity=readN_;
  auto total=acc_.Get<float>();AscendC::Duplicate(total,0.0f,columns_);AscendC::PipeBarrier<PIPE_V>();
  uint32_t cap=capacity/columns_;if(cap>64)cap=64;
  for(uint32_t first=0;first<parts_;first+=cap){const uint32_t count=parts_-first<cap?parts_-first:cap;
   Sync<AscendC::HardEvent::V_MTE2>();AscendC::DataCopy(values,w_[(o*parts_+first)*columns_],count*columns_);Sync<AscendC::HardEvent::MTE2_V>();
   uint32_t base=1;while(base*2<=count)base*=2;const uint32_t tail=count-base;
   if(tail){AscendC::Add(values,values,values[base*columns_],tail*columns_);AscendC::PipeBarrier<PIPE_V>();}
   for(uint32_t h=base/2;h;h/=2){AscendC::Add(values,values,values[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
   AscendC::Add(total,total,values,columns_);AscendC::PipeBarrier<PIPE_V>();
  }
  AscendC::DataCopyExtParams store{1,static_cast<uint32_t>(d_.inner*4),0,0,0};Sync<AscendC::HardEvent::V_MTE3>();
  if constexpr(std::is_same<T,float>::value)AscendC::DataCopyPad(y_[o*d_.inner],total,store);
  Sync<AscendC::HardEvent::MTE3_V>();q_.FreeTensor(scratch);
 }
 __aicore__ inline void FinishGroup(uint64_t first,uint32_t rows){
  auto sum=acc_.Get<float>();auto values=convert_.Get<float>();AscendC::Duplicate(sum,0.0f,rows*columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t part=0;part<parts_;++part){
   AscendC::DataCopyExtParams load{static_cast<uint16_t>(rows),columns_*4,(parts_-1)*columns_*4,0,0};AscendC::DataCopyPadExtParams<float> pad{false,0,0,0.0f};
   Sync<AscendC::HardEvent::V_MTE2>();AscendC::DataCopyPad(values,w_[(first*parts_+part)*columns_],load,pad);Sync<AscendC::HardEvent::MTE2_V>();
   AscendC::Add(sum,sum,values,rows*columns_);AscendC::PipeBarrier<PIPE_V>();
  }
  if constexpr(!std::is_same<T,float>::value){
   auto out=gather_.Get<T>();AscendC::Cast(out,sum,AscendC::RoundMode::CAST_RINT,rows*columns_);Sync<AscendC::HardEvent::V_MTE3>();
   AscendC::DataCopyExtParams store{static_cast<uint16_t>(rows),static_cast<uint32_t>(d_.inner*sizeof(T)),0,0,0};AscendC::DataCopyPad(y_[first*d_.inner],out,store);Sync<AscendC::HardEvent::MTE3_V>();
  }
 }
 SquareSumV1TilingData d_{};uint32_t ar_,rr_,n_,readN_,columns_,parts_,accElems_,convertElems_;
 AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,Buffers>q_;
 AscendC::TQue<AscendC::QuePosition::VECOUT,1>partOut_;
 AscendC::TBuf<AscendC::TPosition::VECCALC>acc_,convert_,gather_,index_;
 AscendC::GlobalTensor<T>x_,y_;AscendC::GlobalTensor<float>w_;
};
template<uint32_t Width> class KernelPackedShortRow {
 static constexpr int HalfOnly=0;
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const SquareSumV1TilingData& d){outputs_=d.outputElements;reduce_=d.reduce;x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));pipe_.InitBuffer(raw_,64*Width*2);if constexpr(!HalfOnly)pipe_.InitBuffer(values_,64*Width*4);pipe_.InitBuffer(result_,64*4);}
 __aicore__ inline void Process(){const uint64_t units=(outputs_+15)/16;uint64_t first=units*AscendC::GetBlockIdx()/AscendC::GetBlockNum()*16;uint64_t end=units*(AscendC::GetBlockIdx()+1)/AscendC::GetBlockNum()*16;if(end>outputs_)end=outputs_;
  for(;first<end;first+=64){const uint32_t rows=end-first<64?end-first:64;if(reduce_<=32){Packed(first,rows);continue;}auto raw=raw_.Get<half>();AscendC::DataCopyExtParams load{static_cast<uint16_t>(rows),static_cast<uint32_t>(reduce_*2),0,0,0};AscendC::DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(Width-reduce_),static_cast<half>(0)};AscendC::DataCopyPad(raw,x_[first*reduce_],load,pad);Sync<AscendC::HardEvent::MTE2_V>();AscendC::Mul(raw,raw,raw,rows*Width);AscendC::PipeBarrier<PIPE_V>();
   if constexpr(HalfOnly){auto out=result_.Get<half>();AscendC::WholeReduceSum<half>(out,raw,Width,rows,1,1,Width/16);Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],out,store);}
   else{auto v=values_.Get<float>();auto result=result_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,rows*Width);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum<float>(result,v,Width,rows,1,1,Width/8);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(raw,result,AscendC::RoundMode::CAST_RINT,rows);Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],raw,store);}
   Sync<AscendC::HardEvent::MTE3_V>();
  }
 }
private:
 __aicore__ inline void Packed(uint64_t first,uint32_t rows){
  auto raw=raw_.Get<half>();const uint32_t count=rows*reduce_;
  AscendC::DataCopyExtParams load{1,count*2,0,0,0};AscendC::DataCopyPadExtParams<half> pad{false,0,0,static_cast<half>(0)};
  if((first*reduce_)%16==0 && count%16==0)AscendC::DataCopy(raw,x_[first*reduce_],count);else AscendC::DataCopyPad(raw,x_[first*reduce_],load,pad);
  Sync<AscendC::HardEvent::MTE2_V>();AscendC::Mul(raw,raw,raw,count);AscendC::PipeBarrier<PIPE_V>();
  auto v=values_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();
  uint32_t period=8;while(period>1 && (period/2*reduce_)%8==0)period/=2;
  // WholeReduceSum destination stride is in elements: emit each phase directly
  // into its final row positions, avoiding an index tensor and Gather pass.
  auto sums=result_.Get<float>();
  for(uint32_t j=0;j<period && j<rows;++j){const uint32_t offset=j*reduce_,low=offset%8;uint64_t mask[2]={((1ULL<<reduce_)-1)<<low,0};AscendC::WholeReduceSum<float>(sums[j],v[offset-low],mask,(rows-j+period-1)/period,period,1,period*reduce_/8);}
  AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(raw,sums,AscendC::RoundMode::CAST_RINT,rows);
  Sync<AscendC::HardEvent::V_MTE3>();AscendC::DataCopyExtParams store{1,rows*2,0,0,0};AscendC::DataCopyPad(y_[first],raw,store);Sync<AscendC::HardEvent::MTE3_V>();
 }
 template<AscendC::HardEvent E>__aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 uint64_t outputs_,reduce_;AscendC::TPipe pipe_;AscendC::TBuf<AscendC::TPosition::VECCALC>raw_,values_,result_;AscendC::GlobalTensor<half>x_,y_;
};
template<typename T, int Method, int Buffers>
class KernelOwnedReduction {
public:
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& d) {
  d_=d; ar_=d.rowTile; rr_=d.tileElements; n_=ar_*d.inner; readN_=rr_*d.inner;
  columns_=AlignUpElements(d.inner,32/sizeof(T));parts_=d.reserved0;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
  if((d_.reserved1&256U)) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

  pipe_.InitBuffer(q_,Buffers,AlignUpElements(readN_*sizeof(T),32));
  accElems_=AlignUpElements(n_+16,8);
  if constexpr(!std::is_same<T,float>::value){if(accElems_<16*columns_)accElems_=16*columns_;}
  pipe_.InitBuffer(acc_,accElems_*4);
  convertElems_=AlignUpElements(readN_,8);if(convertElems_<16*columns_)convertElems_=16*columns_;
  if constexpr(!std::is_same<T,float>::value)pipe_.InitBuffer(convert_,convertElems_*4);
  pipe_.InitBuffer(partOut_,2,columns_*4);
  pipe_.InitBuffer(gather_,8*columns_*4);pipe_.InitBuffer(index_,8*columns_*4);
  auto index=index_.Get<int32_t>();
  for(uint32_t r=0;r<8;++r){AscendC::CreateVecIndex(index[r*columns_],static_cast<int32_t>(r*d.inner),columns_);}
  AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(index,index,static_cast<int32_t>(4),8*columns_);AscendC::PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  // A complete output row belongs to one core; all partial sums remain in UB.
  for(uint64_t o=AscendC::GetBlockIdx();o<d_.outer;o+=AscendC::GetBlockNum())Tile(o,0);
 }

private:
 template<AscendC::HardEvent E> __aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 __aicore__ inline void Copy(uint64_t o,uint64_t first,uint32_t rows){auto t=q_.template AllocTensor<T>();
  AscendC::DataCopyExtParams cp{1,static_cast<uint32_t>(rows*d_.inner*sizeof(T)),0,0,0};
  AscendC::DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  S9PackedCopy(t,x_,(o*d_.reduce+first)*d_.inner,rows*d_.inner,(d_.reserved1&512U)!=0);q_.EnQue(t);
 }
 __aicore__ inline void Compute(uint32_t rows,uint32_t accOffset){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();AscendC::LocalTensor<float> v;
  const uint32_t count=rows*d_.inner;
  if constexpr(std::is_same<T,float>::value)v=raw.template ReinterpretCast<float>();
  else{v=convert_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();}
  if constexpr(Method==0){AscendC::Mul(v,v,v,count);AscendC::PipeBarrier<PIPE_V>();}
  for(uint32_t start=0;start<count;){const uint32_t len=count-start<n_-accOffset?count-start:n_-accOffset;
   if constexpr(Method==1)AscendC::MulAddDst(acc[accOffset],v[start],v[start],len);
   else AscendC::Add(acc[accOffset],acc[accOffset],v[start],len);
   AscendC::PipeBarrier<PIPE_V>(); start+=len;accOffset=0;
  }
  q_.FreeTensor(raw);
 }
 __aicore__ inline void First(uint32_t rows){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();AscendC::LocalTensor<float> v;if constexpr(std::is_same<T,float>::value)v=raw.template ReinterpretCast<float>();else{v=convert_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,rows*d_.inner);AscendC::PipeBarrier<PIPE_V>();}AscendC::Mul(acc,v,v,n_);AscendC::PipeBarrier<PIPE_V>();for(uint32_t start=n_;start<rows*d_.inner;start+=n_){uint32_t len=rows*d_.inner-start<n_?rows*d_.inner-start:n_;AscendC::MulAddDst(acc,v[start],v[start],len);AscendC::PipeBarrier<PIPE_V>();}q_.FreeTensor(raw);}
 __aicore__ inline void Tile(uint64_t o,uint32_t part){auto acc=acc_.Get<float>();const bool direct=Method==1 && d_.reduce>=ar_;if(direct)AscendC::Duplicate(acc[n_],0.0f,16);else AscendC::Duplicate(acc,0.0f,n_+16);AscendC::PipeBarrier<PIPE_V>();
  // Ownership guarantees part=0 and parts=1. Peel the first input tile.
  if(d_.reduce){uint32_t rows=d_.reduce<rr_?d_.reduce:rr_;Copy(o,0,rows);uint64_t first=rr_;uint32_t nr=first<d_.reduce?(d_.reduce-first<rr_?d_.reduce-first:rr_):0;if constexpr(Buffers>1){if(nr)Copy(o,first,nr);}if(direct)First(rows);else Compute(rows,0);uint64_t localRows=rows;rows=nr;
   for(;first<d_.reduce;first+=rr_){uint64_t next=first+rr_;uint32_t nextRows=next<d_.reduce?(d_.reduce-next<rr_?d_.reduce-next:rr_):0;if constexpr(Buffers==1)Copy(o,first,rows);else if(nextRows)Copy(o,next,nextRows);Compute(rows,(d_.reserved1&4U)?0:(localRows%ar_)*d_.inner);localRows+=rows;rows=nextRows;}
  }
  for(uint32_t h=ar_/2;h>=8;h/=2){AscendC::Add(acc,acc,acc[h*d_.inner],h*d_.inner);AscendC::PipeBarrier<PIPE_V>();}
  auto g=gather_.Get<float>();auto index=index_.Get<uint32_t>();AscendC::Gather(g,acc,index,0,8*columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t h=4;h;h/=2){AscendC::Add(g,g,g[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
  auto out=partOut_.AllocTensor<T>();AscendC::Cast(out,g,AscendC::RoundMode::CAST_RINT,columns_);partOut_.EnQue(out);out=partOut_.DeQue<T>();
  AscendC::DataCopyExtParams store{1,static_cast<uint32_t>(d_.inner*sizeof(T)),0,0,0};AscendC::DataCopyPad(y_[o*d_.inner],out,store);partOut_.FreeTensor(out);
 }
 SquareSumV1TilingData d_{};uint32_t ar_,rr_,n_,readN_,columns_,parts_,accElems_,convertElems_;
 AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,Buffers>q_;
 AscendC::TQue<AscendC::QuePosition::VECOUT,2>partOut_;
 AscendC::TBuf<AscendC::TPosition::VECCALC>acc_,convert_,gather_,index_;
 AscendC::GlobalTensor<T>x_,y_;AscendC::GlobalTensor<float>w_;
};
template<class T,int Method,int Buffers>
class KernelOwnedChunked {
public:
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& d) {
  d_=d; ar_=d.rowTile; rr_=d.tileElements; n_=ar_*d.inner; readN_=rr_*d.inner;
  columns_=AlignUpElements(d.inner,32/sizeof(T));parts_=d.reserved0;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
  if((d_.reserved1&256U)) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

  pipe_.InitBuffer(q_,Buffers,AlignUpElements(readN_*sizeof(T),32));
  accElems_=AlignUpElements(n_>7*d.inner+columns_?n_:7*d.inner+columns_,8);
  pipe_.InitBuffer(acc_,accElems_*4);
  convertElems_=AlignUpElements(n_,8);if(convertElems_<16*columns_)convertElems_=16*columns_;
  pipe_.InitBuffer(convert_,convertElems_*4);
 }
 __aicore__ inline void Process(){
  // A complete output row belongs to one core; all partial sums remain in UB.
  for(uint64_t o=AscendC::GetBlockIdx();o<d_.outer;o+=AscendC::GetBlockNum())Tile(o,0);
 }

private:
 template<AscendC::HardEvent E> __aicore__ inline void Sync(){const event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);}
 __aicore__ inline void Copy(uint64_t o,uint64_t first,uint32_t rows){auto t=q_.template AllocTensor<T>();
  AscendC::DataCopyExtParams cp{1,static_cast<uint32_t>(rows*d_.inner*sizeof(T)),0,0,0};
  AscendC::DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  S9PackedCopy(t,x_,(o*d_.reduce+first)*d_.inner,rows*d_.inner,(d_.reserved1&512U)!=0);q_.EnQue(t);
 }
 __aicore__ inline void Compute(uint32_t rows,uint32_t off){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();auto v=convert_.Get<float>();const uint32_t count=rows*d_.inner;
  for(uint32_t start=0;start<count;){uint32_t len=count-start<n_-off?count-start:n_-off;AscendC::Cast(v,raw[start],AscendC::RoundMode::CAST_NONE,len);AscendC::PipeBarrier<PIPE_V>();AscendC::MulAddDst(acc[off],v,v,len);AscendC::PipeBarrier<PIPE_V>();start+=len;off=0;}
  q_.FreeTensor(raw);
 }
 __aicore__ inline void First(uint32_t rows){auto raw=q_.template DeQue<T>();auto acc=acc_.Get<float>();auto v=convert_.Get<float>();AscendC::Cast(v,raw,AscendC::RoundMode::CAST_NONE,n_);AscendC::PipeBarrier<PIPE_V>();AscendC::Mul(acc,v,v,n_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t start=n_;start<rows*d_.inner;start+=n_){uint32_t len=rows*d_.inner-start<n_?rows*d_.inner-start:n_;AscendC::Cast(v,raw[start],AscendC::RoundMode::CAST_NONE,len);AscendC::PipeBarrier<PIPE_V>();AscendC::MulAddDst(acc,v,v,len);AscendC::PipeBarrier<PIPE_V>();}q_.FreeTensor(raw);
 }
 __aicore__ inline void Tile(uint64_t o,uint32_t part){auto acc=acc_.Get<float>();const bool direct=Method==1 && d_.reduce>=ar_;if(!direct)AscendC::Duplicate(acc,0.0f,accElems_);AscendC::PipeBarrier<PIPE_V>();
  // Ownership guarantees part=0 and parts=1. Peel the first input tile.
  if(d_.reduce){uint32_t rows=d_.reduce<rr_?d_.reduce:rr_;Copy(o,0,rows);uint64_t first=rr_;uint32_t nr=first<d_.reduce?(d_.reduce-first<rr_?d_.reduce-first:rr_):0;if constexpr(Buffers>1){if(nr)Copy(o,first,nr);}if(direct)First(rows);else Compute(rows,0);uint64_t localRows=rows;rows=nr;
   for(;first<d_.reduce;first+=rr_){uint64_t next=first+rr_;uint32_t nextRows=next<d_.reduce?(d_.reduce-next<rr_?d_.reduce-next:rr_):0;if constexpr(Buffers==1)Copy(o,first,rows);else if(nextRows)Copy(o,next,nextRows);Compute(rows,(d_.reserved1&4U)?0:(localRows%ar_)*d_.inner);localRows+=rows;rows=nextRows;}
  }
  for(uint32_t h=ar_/2;h>=8;h/=2){AscendC::Add(acc,acc,acc[h*d_.inner],h*d_.inner);AscendC::PipeBarrier<PIPE_V>();}
  auto g=convert_.Get<float>();auto index=convert_.Get<uint32_t>()[8*columns_];
  auto signedIndex=index.ReinterpretCast<int32_t>();
  for(uint32_t r=0;r<8;++r)AscendC::CreateVecIndex(signedIndex[r*columns_],static_cast<int32_t>(r*d_.inner),columns_);
  AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(signedIndex,signedIndex,static_cast<int32_t>(4),8*columns_);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Gather(g,acc,index,0,8*columns_);AscendC::PipeBarrier<PIPE_V>();
  for(uint32_t h=4;h;h/=2){AscendC::Add(g,g,g[h*columns_],h*columns_);AscendC::PipeBarrier<PIPE_V>();}
  // The accumulator is dead after Gather; its bytes now hold the output.
  auto out=acc.template ReinterpretCast<T>();AscendC::Cast(out,g,AscendC::RoundMode::CAST_RINT,columns_);Sync<AscendC::HardEvent::V_MTE3>();
  AscendC::DataCopyExtParams store{1,static_cast<uint32_t>(d_.inner*sizeof(T)),0,0,0};AscendC::DataCopyPad(y_[o*d_.inner],out,store);Sync<AscendC::HardEvent::MTE3_V>();
 }
 SquareSumV1TilingData d_{};uint32_t ar_,rr_,n_,readN_,columns_,parts_,accElems_,convertElems_;
 AscendC::TPipe pipe_;AscendC::TQue<AscendC::QuePosition::VECIN,Buffers>q_;

 AscendC::TBuf<AscendC::TPosition::VECCALC>acc_,convert_;
 AscendC::GlobalTensor<T>x_,y_;AscendC::GlobalTensor<float>w_;
};
} // namespace s9squaresum
extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR x,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 KERNEL_TASK_TYPE(544,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(502,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(264,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(262,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(254,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(252,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(224,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(222,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(204,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(202,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(184,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(182,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(144,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(142,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(504,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(542,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(820,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(844,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(842,KERNEL_TYPE_MIX_AIV_1_0);
 KERNEL_TASK_TYPE(841,KERNEL_TYPE_MIX_AIV_1_0);
 REGISTER_TILING_DEFAULT(s9squaresum::SquareSumV1TilingData);GET_TILING_DATA(tilingData,tiling);
 if(TILING_KEY_IS(800)){s9squaresum::KernelShortRow<16> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(801)){s9squaresum::KernelShortRow<32> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(802)){s9squaresum::KernelShortRow<48> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(803)){s9squaresum::KernelShortRow<64> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(810)){s9squaresum::KernelPowerBoundaryAr<bfloat16_t,64> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(811)){s9squaresum::KernelPowerBoundaryAr<bfloat16_t,128> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(812)){s9squaresum::KernelPowerBoundaryAr<bfloat16_t,256> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(813)){s9squaresum::KernelPowerBoundaryAr<bfloat16_t,512> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(814)){s9squaresum::KernelPowerBoundaryAr<bfloat16_t,1024> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(820)){s9squaresum::KernelPackedReduction<DTYPE_X,1,2> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
 if(TILING_KEY_IS(823)){s9squaresum::KernelOwnedChunked<bfloat16_t,1,2> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
 if(TILING_KEY_IS(821)){s9squaresum::KernelOwnedReduction<DTYPE_X,1,2> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
 if(TILING_KEY_IS(830)){s9squaresum::KernelMaskedRar<11> op;op.Init(x,y,tilingData);op.Process();return;}
 if(TILING_KEY_IS(844)){s9squaresum::KernelStreamingPrefix<float,1,4> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
 if(TILING_KEY_IS(842)){s9squaresum::KernelStreamingPrefix<float,1,2> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
 if(TILING_KEY_IS(841)){s9squaresum::KernelStreamingPrefix<float,1,1> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
if(TILING_KEY_IS(102)){s9squaresum::KernelSquareSumArVector<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();return;}
if(TILING_KEY_IS(122)){s9squaresum::KernelSquareSumLongArVector<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();return;}
if(TILING_KEY_IS(504)){s9squaresum::KernelSquareSumSlab<DTYPE_X> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
if(TILING_KEY_IS(542)){s9squaresum::KernelSquareSumSlab<DTYPE_X,true> op;op.Init(x,y,workspace,tilingData);op.Process();return;}
if(TILING_KEY_IS(562)){s9squaresum::KernelSquareSumRARPacked<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();return;}
if (TILING_KEY_IS(104)) {
        s9squaresum::KernelSquareSumArVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(124)) {
        s9squaresum::KernelSquareSumLongArVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(142)) {
        s9squaresum::KernelSquareSumReduceAllMultiCore<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(144)) {
        s9squaresum::KernelSquareSumReduceAllMultiCore<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(152)) {
        s9squaresum::KernelSquareSumPrefixSmallInner<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(154)) {
        s9squaresum::KernelSquareSumPrefixSmallInner<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(132)) {
        s9squaresum::KernelSquareSumSplitReduceVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(134)) {
        s9squaresum::KernelSquareSumSplitReduceVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(112)) {
        s9squaresum::KernelSquareSumStridedVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(114)) {
        s9squaresum::KernelSquareSumStridedVector<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(2)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(4)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(12)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(14)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(22)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(24)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(32)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(34)) {
        s9squaresum::KernelSquareSumV1<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
if (TILING_KEY_IS(182)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(184)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(162)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(164)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(202)) {
        s9squaresum::KernelSquareSumPrefixParallel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(204)) {
        s9squaresum::KernelSquareSumPrefixParallel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(212)) {
        s9squaresum::KernelSquareSumStreaming<DTYPE_X, false> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(214)) {
        s9squaresum::KernelSquareSumStreaming<DTYPE_X, false> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(222)) {
        s9squaresum::KernelSquareSumStreaming<DTYPE_X, true> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(224)) {
        s9squaresum::KernelSquareSumStreaming<DTYPE_X, true> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(232)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, false, true> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(234)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, false, true> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(242)) {
        s9squaresum::KernelSquareSumRectangularSuffix<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(244)) {
        s9squaresum::KernelSquareSumRectangularSuffix<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(252)) {
        s9squaresum::KernelSquareSumRectangularSuffix<DTYPE_X, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(254)) {
        s9squaresum::KernelSquareSumRectangularSuffix<DTYPE_X, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(262)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, true, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(264)) {
        s9squaresum::KernelSquareSumBatchedColumns<DTYPE_X, true, true> op;
        op.Init(x, y, tilingData, workspace); op.Process(); return;
    }
if (TILING_KEY_IS(282)) {
        s9squaresum::KernelSquareSumLongArVector<DTYPE_X, true> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(284)) {
        s9squaresum::KernelSquareSumLongArVector<DTYPE_X, true> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(502)) {
        s9squaresum::KernelSquareSumSlab<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(522)) {
        s9squaresum::KernelSquareSumRARSlab<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(524)) {
        s9squaresum::KernelSquareSumRARSlab<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process(); return;
    }
if (TILING_KEY_IS(544)) {
        s9squaresum::KernelSquareSumSlab<DTYPE_X, true> op;
        op.Init(x, y, workspace, tilingData); op.Process(); return;
    }
if(TILING_KEY_IS(980)){s9squaresum::KernelPackedShortRow<16> op;op.Init(x,y,tilingData);op.Process();return;}
if(TILING_KEY_IS(981)){s9squaresum::KernelPackedShortRow<32> op;op.Init(x,y,tilingData);op.Process();return;}
}
