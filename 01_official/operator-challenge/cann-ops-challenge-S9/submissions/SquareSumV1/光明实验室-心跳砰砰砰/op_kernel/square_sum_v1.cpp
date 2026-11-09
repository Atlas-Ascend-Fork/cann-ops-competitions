#include "square_sum_v1_tiling.h"
#include "kernel_operator.h"

#include <cstdint>
#include <type_traits>

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t kOutputAlignNum = 16;
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kUbBytes = 192 * 1024;
constexpr uint32_t kElemAlign = 32;
constexpr uint32_t kMaxReduceRows = 224;
constexpr uint32_t kWholeReduceMaxLen = 128;
constexpr uint32_t kGenericKeptElems = 9728;
constexpr uint32_t kGenericPlainHalfElems = 30720;
constexpr uint32_t kGenericPlainFloatElems = 23040;
constexpr uint32_t kSplitFp32Elems = 23680;
constexpr uint32_t kCycleBankOffsetBytes = 256;
constexpr uint32_t kFp32CycleBankOffsetBytes = 320;
constexpr uint32_t kCycleGroupRows = 16;
constexpr uint32_t kFp32CycleMaxCols = 544;
constexpr uint32_t kBf16CycleRows = 16;
constexpr uint32_t kGenericBatchCols = 1024;
constexpr uint32_t kGenericReduceCols = 64;
constexpr uint32_t kGenericPackedCols = 19;
constexpr uint32_t kGenericPackedRowAlign = 512;
constexpr uint32_t kGenericPackedMaxRows = 4096;
constexpr uint32_t kGenericBatchMinReduce = 64;
constexpr uint32_t kGenericFlattenMinCols = 256;
constexpr uint32_t kGenericReduceRows = 224;
constexpr uint32_t kGenericTransposeMaxCols = 256;
constexpr uint32_t kGenericLongSuffixCols = 1024;
constexpr uint32_t kSuffixWideRows = 14;
constexpr uint32_t kSuffixWideCols = 1024;
constexpr uint32_t kSuffixWideSegments = kSuffixWideCols / kGenericReduceCols;
constexpr uint32_t kSuffixWideAlignedRows = 16;
constexpr uint32_t kSuffixWideCompactRows = 23;
constexpr uint32_t kNarrowAxis2TileRows = 4096;
constexpr uint32_t kNarrowAxis2ReduceRows = 64;
constexpr uint32_t kNarrowAxis2ResultElems = 8;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

__aicore__ inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}

__aicore__ inline uint32_t GetSuffixTileRows(uint32_t alignedReduceLength)
{
    const uint32_t bytesPerRow = alignedReduceLength * sizeof(half) * 2U + sizeof(half);
    uint32_t rows = AlignDown(kUbBytes / bytesPerRow, kElemAlign);
    if (rows == 0) {
        rows = kElemAlign;
    }
    if (rows > kMaxReduceRows) {
        rows = kMaxReduceRows;
    }
    return rows;
}

__aicore__ inline void GetCoreRange(uint32_t smallCoreItems, uint32_t bigCoreCount,
                                    uint32_t& begin, uint32_t& length)
{
    const uint32_t blockIdx = GetBlockIdx();
    const bool isBigCore = blockIdx < bigCoreCount;
    begin = blockIdx * smallCoreItems + (isBigCore ? blockIdx : bigCoreCount);
    length = smallCoreItems + (isBigCore ? 1U : 0U);
}

template <typename T>
__aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t len)
{
    Cast(dst, src, RoundMode::CAST_NONE, len);
}

template <>
__aicore__ inline void CastInputToFloat<float>(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len)
{
    Adds(dst, src, 0.0F, static_cast<int32_t>(len));
}

template <typename T>
__aicore__ inline void SquareInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t len)
{
    Cast(dst, src, RoundMode::CAST_NONE, len);
    PipeBarrier<PIPE_V>();
    Mul(dst, dst, dst, static_cast<int32_t>(len));
}

template <>
__aicore__ inline void SquareInputToFloat<float>(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len)
{
    Mul(dst, src, src, static_cast<int32_t>(len));
}

template <typename T>
__aicore__ inline void CastFloatToOutput(LocalTensor<T>& dst, LocalTensor<float>& src, uint32_t len)
{
    Cast(dst, src, RoundMode::CAST_RINT, len);
}

template <>
__aicore__ inline void CastFloatToOutput<float>(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len)
{
    Adds(dst, src, 0.0F, static_cast<int32_t>(len));
}

class KernelSquareSumV1 {
public:
    __aicore__ inline KernelSquareSumV1() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset,
                                uint32_t currentBlockLength, uint32_t reduceLength,
                                uint32_t alignedReduceLength)
    {
        blockLength_ = currentBlockLength;
        reduceLength_ = reduceLength;
        alignedReduceLength_ = alignedReduceLength;
        tileRows_ = GetSuffixTileRows(alignedReduceLength);
        inputGm_.SetGlobalBuffer((__gm__ half*)input + blockOffset * reduceLength,
                                 currentBlockLength * reduceLength);
        outputGm_.SetGlobalBuffer((__gm__ half*)output + blockOffset, currentBlockLength);

        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileRows_ * alignedReduceLength_ * sizeof(half));
        pipe_.InitBuffer(squareQueue_, BUFFER_NUM, tileRows_ * alignedReduceLength_ * sizeof(half));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, AlignUp(tileRows_, kOutputAlignNum) * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowBase = 0; rowBase < blockLength_; rowBase += tileRows_) {
            uint32_t rows = blockLength_ - rowBase;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            CopyIn(rowBase, rows);
            Compute(rows);
            CopyOut(rowBase, rows);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return ((value + align - 1U) / align) * align;
    }

    __aicore__ inline void CopyIn(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<half> inputLocal = inputQueue_.AllocTensor<half>();
        if (alignedReduceLength_ == reduceLength_) {
            DataCopy(inputLocal, inputGm_[rowBase * reduceLength_], rows * reduceLength_);
            inputQueue_.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(reduceLength_ * sizeof(half)),
            0,
            0,
            0};
        DataCopyPadExtParams<half> padParams = {
            true,
            0,
            static_cast<uint8_t>(alignedReduceLength_ - reduceLength_),
            0};
        DataCopyPad(inputLocal, inputGm_[rowBase * reduceLength_], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t rows)
    {
        LocalTensor<half> inputLocal = inputQueue_.DeQue<half>();
        LocalTensor<half> squareLocal = squareQueue_.AllocTensor<half>();
        LocalTensor<half> outputLocal = outputQueue_.AllocTensor<half>();

        Mul(squareLocal, inputLocal, inputLocal, rows * alignedReduceLength_);
        PipeBarrier<PIPE_V>();

        WholeReduceSum(outputLocal, squareLocal, static_cast<int32_t>(reduceLength_),
                       static_cast<int32_t>(rows), 1, 1,
                       static_cast<int32_t>(alignedReduceLength_ * sizeof(half) / kBlockBytes));
        PipeBarrier<PIPE_V>();

        outputQueue_.EnQue(outputLocal);
        squareQueue_.FreeTensor(squareLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<half> outputLocal = outputQueue_.DeQue<half>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0};
        DataCopyPad(outputGm_[rowBase], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECCALC, BUFFER_NUM> squareQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t reduceLength_ = 0;
    uint32_t alignedReduceLength_ = 0;
    uint32_t tileRows_ = 0;
};

__aicore__ inline void RunSquareSumV1(GM_ADDR input, GM_ADDR output, uint32_t smallCoreItems,
                                      uint32_t bigCoreCount, uint32_t reduceLength,
                                      uint32_t alignedReduceLength)
{
    uint32_t blockOffset = 0;
    uint32_t currentBlockLength = 0;
    GetCoreRange(smallCoreItems, bigCoreCount, blockOffset, currentBlockLength);
    KernelSquareSumV1 kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, reduceLength, alignedReduceLength);
    kernel.Process();
}

class KernelSquareSumV1Split {
public:
    __aicore__ inline KernelSquareSumV1Split() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset,
                                uint32_t currentBlockLength, uint32_t reduceLength,
                                uint32_t alignedReduceLength)
    {
        blockLength_ = currentBlockLength;
        reduceLength_ = reduceLength;
        alignedReduceLength_ = alignedReduceLength;
        tileRows_ = GetSuffixTileRows(alignedReduceLength);
        inputGm_.SetGlobalBuffer((__gm__ half*)input + blockOffset * reduceLength,
                                 currentBlockLength * reduceLength);
        outputGm_.SetGlobalBuffer((__gm__ half*)output + blockOffset, currentBlockLength);

        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileRows_ * alignedReduceLength_ * sizeof(half));
        pipe_.InitBuffer(squareQueue_, BUFFER_NUM, tileRows_ * alignedReduceLength_ * sizeof(half));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, AlignUp(tileRows_, kOutputAlignNum) * sizeof(half));
        pipe_.InitBuffer(partialBuf_, AlignUp(tileRows_, kOutputAlignNum) * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowBase = 0; rowBase < blockLength_; rowBase += tileRows_) {
            uint32_t rows = blockLength_ - rowBase;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            CopyIn(rowBase, rows);
            Compute(rows);
            CopyOut(rowBase, rows);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return ((value + align - 1U) / align) * align;
    }

    __aicore__ inline void CopyIn(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<half> inputLocal = inputQueue_.AllocTensor<half>();
        if (alignedReduceLength_ == reduceLength_) {
            DataCopy(inputLocal, inputGm_[rowBase * reduceLength_], rows * reduceLength_);
            inputQueue_.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(reduceLength_ * sizeof(half)),
            0,
            0,
            0};
        DataCopyPadExtParams<half> padParams = {
            true,
            0,
            static_cast<uint8_t>(alignedReduceLength_ - reduceLength_),
            0};
        DataCopyPad(inputLocal, inputGm_[rowBase * reduceLength_], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t rows)
    {
        LocalTensor<half> inputLocal = inputQueue_.DeQue<half>();
        LocalTensor<half> squareLocal = squareQueue_.AllocTensor<half>();
        LocalTensor<half> outputLocal = outputQueue_.AllocTensor<half>();
        LocalTensor<half> partialLocal = partialBuf_.Get<half>();

        Mul(squareLocal, inputLocal, inputLocal, rows * alignedReduceLength_);
        PipeBarrier<PIPE_V>();

        const uint32_t tailLength = reduceLength_ - kWholeReduceMaxLen;
        const int32_t rowStrideBlocks = static_cast<int32_t>(alignedReduceLength_ * sizeof(half) / kBlockBytes);
        WholeReduceSum(outputLocal, squareLocal, static_cast<int32_t>(kWholeReduceMaxLen),
                       static_cast<int32_t>(rows), 1, 1, rowStrideBlocks);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(partialLocal, squareLocal[kWholeReduceMaxLen], static_cast<int32_t>(tailLength),
                       static_cast<int32_t>(rows), 1, 1, rowStrideBlocks);
        PipeBarrier<PIPE_V>();
        Add(outputLocal, outputLocal, partialLocal, static_cast<int32_t>(AlignUp(rows, kOutputAlignNum)));
        PipeBarrier<PIPE_V>();

        outputQueue_.EnQue(outputLocal);
        squareQueue_.FreeTensor(squareLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<half> outputLocal = outputQueue_.DeQue<half>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0};
        DataCopyPad(outputGm_[rowBase], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECCALC, BUFFER_NUM> squareQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<QuePosition::VECCALC> partialBuf_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t reduceLength_ = 0;
    uint32_t alignedReduceLength_ = 0;
    uint32_t tileRows_ = 0;
};

__aicore__ inline void RunSquareSumV1Split(GM_ADDR input, GM_ADDR output, uint32_t smallCoreItems,
                                           uint32_t bigCoreCount, uint32_t reduceLength,
                                           uint32_t alignedReduceLength)
{
    uint32_t blockOffset = 0;
    uint32_t currentBlockLength = 0;
    GetCoreRange(smallCoreItems, bigCoreCount, blockOffset, currentBlockLength);
    KernelSquareSumV1Split kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, reduceLength, alignedReduceLength);
    kernel.Process();
}

template <typename T, bool DoubleBuffered, bool Compact = false>
class KernelSquareSumV1SuffixWide {
    static constexpr int32_t kInputBuffers = DoubleBuffered ? 2 : 1;

public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset,
                                uint32_t blockLength, uint32_t reduceLength)
    {
        blockLength_ = blockLength;
        reduceLength_ = reduceLength;
        if constexpr (Compact && std::is_same_v<T, bfloat16_t>) {
            if (reduceLength_ > kSuffixWideCols &&
                reduceLength_ <= kSuffixWideCols + kBlockBytes / sizeof(T)) {
                tileCols_ = kSuffixWideCols;
                tileRows_ = 32U / (reduceLength_ - tileCols_);
                if (tileRows_ > 21U) tileRows_ = 21U;
                accRows_ = AlignUp(tileRows_, kSuffixWideAlignedRows);
                inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset * reduceLength,
                                         blockLength * reduceLength);
                inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, blockLength);
                const uint32_t dataElems = tileRows_ * tileCols_ + 32U;
                const uint32_t boundaryElems = AlignUp((tileRows_ + 1U) * 32U, 64U);
                pipe_.InitBuffer(inputQueue_, 2, dataElems * sizeof(T));
                pipe_.InitBuffer(inputFloatBuf_, dataElems * sizeof(float));
                pipe_.InitBuffer(partialFloatBuf_, accRows_ * sizeof(float));
                pipe_.InitBuffer(squareFloatBuf_, (2U * boundaryElems + 128U) * sizeof(float));
                pipe_.InitBuffer(outputBuf_, 32U * sizeof(uint32_t));
                pipe_.InitBuffer(accFloatBuf_, 256U);
                pipe_.InitBuffer(outputQueue_, 2, accRows_ * sizeof(T));
                useNarrowTail_ = true;
                return;
            }
        }
        if constexpr (DoubleBuffered) {
            if constexpr (Compact) {
                tileCols_ = reduceLength > kSuffixWideCols
                                ? kSuffixWideCols
                                : AlignUp(reduceLength, kGenericReduceCols);
            } else {
                tileCols_ = reduceLength <= kSuffixWideCols
                                ? AlignUp(reduceLength, kGenericReduceCols) : kSuffixWideCols;
            }
            const uint32_t rowBytes = tileCols_ *
                                          (kInputBuffers * sizeof(T) +
                                           (Compact ? 1U : 2U) * sizeof(float)) +
                                      (tileCols_ / kGenericReduceCols) * sizeof(float);
            tileRows_ = (kUbBytes - 512U) / rowBytes;
            const uint32_t maxRows = Compact ? kSuffixWideCompactRows : 31U;
            if (tileRows_ > maxRows) {
                tileRows_ = maxRows;
            }
        } else {
            tileCols_ = reduceLength <= 512U ? 512U : kSuffixWideCols;
            tileRows_ = tileCols_ == 512U ? 31U : kSuffixWideRows;
        }
        segments_ = tileCols_ / kGenericReduceCols;
        accRows_ = AlignUp(tileRows_, kSuffixWideAlignedRows);
        inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset * reduceLength,
                                 blockLength * reduceLength);
        outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, blockLength);
        const uint32_t dataElems = AlignUp(tileRows_ * tileCols_, kBlockBytes / sizeof(T));
        if constexpr (DoubleBuffered) {
            pipe_.InitBuffer(inputQueue_, kInputBuffers, dataElems * sizeof(T));
        } else {
            pipe_.InitBuffer(inputBuf_, dataElems * sizeof(T));
        }
        pipe_.InitBuffer(inputFloatBuf_, dataElems * sizeof(float));
        if constexpr (!Compact) {
            pipe_.InitBuffer(squareFloatBuf_, dataElems * sizeof(float));
        }
        if constexpr (!Compact) {
            pipe_.InitBuffer(partialFloatBuf_, tileRows_ * segments_ * sizeof(float));
        } else if (reduceLength_ > tileCols_) {
            pipe_.InitBuffer(partialFloatBuf_, accRows_ * sizeof(float));
        }
        pipe_.InitBuffer(accFloatBuf_, accRows_ * sizeof(float));
        if constexpr (Compact) {
            pipe_.InitBuffer(outputQueue_, 2, accRows_ * sizeof(T));
        } else {
            pipe_.InitBuffer(outputBuf_, accRows_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (useNarrowTail_) {
            ProcessRowsPipelinedNarrowTail();
            return;
        }
        if constexpr (!DoubleBuffered) {
            ProcessSerial();
            return;
        }
        if (reduceLength_ <= tileCols_) {
            ProcessRowsPipelined();
            return;
        }
        if constexpr (Compact) {
            ProcessRowsPipelinedCompact2();
            return;
        }
        for (uint32_t rowBase = 0; rowBase < blockLength_; rowBase += tileRows_) {
            uint32_t rows = blockLength_ - rowBase;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            LocalTensor<float> acc = accFloatBuf_.Get<float>();
            Duplicate(acc, 0.0F, accRows_);
            CopyIn(rowBase, 0, rows);
            for (uint32_t colBase = 0; colBase < reduceLength_; colBase += tileCols_) {
                const uint32_t nextCol = colBase + tileCols_;
                if constexpr (kInputBuffers == 2) {
                    if (nextCol + tileCols_ <= reduceLength_) {
                        CopyIn(rowBase, nextCol, rows);
                    }
                }
                ComputeChunk(rows);
                if (nextCol < reduceLength_) {
                    if constexpr (kInputBuffers == 1) {
                        CopyIn(rowBase, nextCol, rows);
                    } else if (nextCol + tileCols_ > reduceLength_) {
                        CopyIn(rowBase, nextCol, rows);
                    }
                }
            }
            CopyOut(rowBase, rows);
        }
    }

private:
    __aicore__ inline void ProcessRowsPipelinedNarrowTail()
    {
        uint32_t rowBase = 0U;
        uint32_t rows = blockLength_ < tileRows_ ? blockLength_ : tileRows_;
        CopyInNarrowTail(rowBase, rows);
        const uint32_t boundaryElems = AlignUp((tileRows_ + 1U) * 32U, 64U);
        auto offsets = inputFloatBuf_.Get<int32_t>();
        auto row = offsets[boundaryElems];
        auto col = squareFloatBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsets, 0, 1, boundaryElems);
        PipeBarrier<PIPE_V>();
        ShiftRight(row, offsets, 5, boundaryElems);
        Duplicate(col, 31, boundaryElems);
        PipeBarrier<PIPE_V>();
        And(col.ReinterpretCast<uint16_t>(), offsets.ReinterpretCast<uint16_t>(),
            col.ReinterpretCast<uint16_t>(), 2U * boundaryElems);
        PipeBarrier<PIPE_V>();
        Muls(row, row, static_cast<int32_t>(reduceLength_ - tileCols_), boundaryElems);
        PipeBarrier<PIPE_V>();
        Cast(row.ReinterpretCast<float>(), row, RoundMode::CAST_NONE, boundaryElems);
        Cast(col.ReinterpretCast<float>(), col, RoundMode::CAST_NONE, boundaryElems);
        PipeBarrier<PIPE_V>();
        Compare(accFloatBuf_.Get<uint8_t>(), col.ReinterpretCast<float>(),
                row.ReinterpretCast<float>(), CMPMODE::LT, boundaryElems);
        Compare(accFloatBuf_.Get<uint8_t>()[128U], col.ReinterpretCast<float>(),
                row.ReinterpretCast<float>(), CMPMODE::GE, boundaryElems);
        PipeBarrier<PIPE_V>();
        Duplicate(row, 0, 32U);
        ArithProgression<int32_t>(outputBuf_.Get<int32_t>(), 4, 4, 32U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(row.ReinterpretCast<uint16_t>());
        while (rowBase < blockLength_) {
            const uint32_t nextBase = rowBase + rows;
            const uint32_t nextRows = blockLength_ - nextBase < tileRows_
                                          ? blockLength_ - nextBase : tileRows_;
            if (nextBase < blockLength_) {
                CopyInNarrowTail(nextBase, nextRows);
            }
            ComputeNarrowTail(rowBase, rows);
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void CopyInNarrowTail(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        const uint32_t elems = rows * reduceLength_;
        DataCopyExtParams copyParams = {
            1U, static_cast<uint32_t>(elems * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {
            true, 0, static_cast<uint8_t>(AlignUp(elems, 16U) - elems), static_cast<T>(0)};
        DataCopyPad(input, inputGm_[rowBase * reduceLength_], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    template <uint32_t Cols>
    __aicore__ inline void ReduceNarrowPairs(LocalTensor<float>& data,
                                            LocalTensor<float>& result, uint32_t rows)
    {
        SetVectorMask<float, MaskMode::COUNTER>(0U, rows * Cols);
        if constexpr (Cols >= 8U) {
            BlockReduceSum<float, false>(data, data, 1U, MASK_PLACEHOLDER, 1U, 1U, 8U);
            PipeBarrier<PIPE_V>();
            ReduceNarrowPairs<Cols / 8U>(data, result, rows);
        } else {
            PairReduceSum<float, false>(result, data, 1U, MASK_PLACEHOLDER, 1U, 1U, 8U);
        }
    }

    __aicore__ inline void ComputeNarrowTail(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<float> converted = inputFloatBuf_.Get<float>();
        LocalTensor<float> reduced = partialFloatBuf_.Get<float>();
        const uint32_t boundaryElems = AlignUp((tileRows_ + 1U) * 32U, 64U);
        auto boundary = squareFloatBuf_.Get<float>();
        auto selected = boundary[boundaryElems];
        auto prefix = boundary[2U * boundaryElems];
        auto suffix = prefix[32U];
        auto shifted = suffix[32U];
        const uint32_t elems = rows * tileCols_ + 32U;
        const uint32_t copiedElems = AlignUp(rows * reduceLength_, 16U);
        CastInputToFloat<T>(converted, input, copiedElems);
        if (copiedElems < elems) {
            Duplicate(converted[copiedElems], 0.0F, elems - copiedElems);
        }
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Mul(converted, converted, converted, elems);
        PipeBarrier<PIPE_V>();
        DataCopyParams boundaryCopy = {static_cast<uint16_t>(rows + 1U), 4U, 124U, 0U};
        DataCopy(boundary, converted, boundaryCopy);
        PipeBarrier<PIPE_V>();

        Duplicate(converted, 0.0F, 32U, rows, 1U, 128U);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0U, (rows + 1U) * 32U);
        Select(selected, accFloatBuf_.Get<uint8_t>(), boundary, 1U, BinaryRepeatParams{});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        WholeReduceSum(prefix, selected, 32U, rows + 1U, 1U, 1U, 4U);
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0U, (rows + 1U) * 32U);
        Select(selected, accFloatBuf_.Get<uint8_t>()[128U], boundary, 1U, BinaryRepeatParams{});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        WholeReduceSum(suffix, selected, 32U, rows, 1U, 1U, 4U);
        SetMaskCount();
        ReduceNarrowPairs<kSuffixWideCols>(converted, reduced, rows);
        SetMaskNorm();
        ResetMask();
        Gather(shifted, prefix, outputBuf_.Get<uint32_t>(), 0U, rows);
        PipeBarrier<PIPE_V>();
        Add(reduced, reduced, suffix, rows);
        PipeBarrier<PIPE_V>();
        Add(reduced, reduced, shifted, rows);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> output = outputQueue_.template AllocTensor<T>();
        CastFloatToOutput<T>(output, reduced, rows);
        outputQueue_.EnQue(output);
        output = outputQueue_.template DeQue<T>();
        DataCopyExtParams outputParams = {
            1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[rowBase], output, outputParams);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessRowsPipelined()
    {
        if constexpr (Compact) {
            ProcessRowsPipelinedCompact();
            return;
        }
        uint32_t rowBase = 0U;
        uint32_t rows = blockLength_ < tileRows_ ? blockLength_ : tileRows_;
        CopyIn(rowBase, 0U, rows);
        while (rowBase < blockLength_) {
            const uint32_t nextBase = rowBase + rows;
            const uint32_t nextRows = blockLength_ - nextBase < tileRows_
                                          ? blockLength_ - nextBase : tileRows_;
            if (nextBase < blockLength_) {
                CopyIn(nextBase, 0U, nextRows);
            }
            Duplicate(accFloatBuf_.Get<float>(), 0.0F, accRows_);
            ComputeChunk(rows);
            CopyOut(rowBase, rows);
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void ProcessRowsPipelinedCompact()
    {
        if (blockLength_ < kSuffixWideCompactRows) {
            CopyIn(0U, 0U, blockLength_);
            ComputeChunkCompact(blockLength_);
            CopyOut(0U, blockLength_);
            return;
        }
        const uint32_t fullRows = blockLength_ / kSuffixWideCompactRows * kSuffixWideCompactRows;
        uint32_t rowBase = 0U;
        CopyIn(0U, 0U, kSuffixWideCompactRows);
        for (; rowBase + 2U * kSuffixWideCompactRows < fullRows;
             rowBase += 2U * kSuffixWideCompactRows) {
            CopyIn(rowBase + kSuffixWideCompactRows, 0U, kSuffixWideCompactRows);
            ComputeChunkCompact<kSuffixWideCompactRows>();
            CopyOut(rowBase, kSuffixWideCompactRows);
            CopyIn(rowBase + 2U * kSuffixWideCompactRows, 0U, kSuffixWideCompactRows);
            ComputeChunkCompact<kSuffixWideCompactRows>();
            CopyOut(rowBase + kSuffixWideCompactRows, kSuffixWideCompactRows);
        }
        for (; rowBase + kSuffixWideCompactRows < fullRows;
             rowBase += kSuffixWideCompactRows) {
            CopyIn(rowBase + kSuffixWideCompactRows, 0U, kSuffixWideCompactRows);
            ComputeChunkCompact<kSuffixWideCompactRows>();
            CopyOut(rowBase, kSuffixWideCompactRows);
        }
        const uint32_t nextBase = rowBase + kSuffixWideCompactRows;
        if (nextBase < blockLength_) {
            CopyIn(nextBase, 0U, blockLength_ - nextBase);
        }
        ComputeChunkCompact<kSuffixWideCompactRows>();
        CopyOut(rowBase, kSuffixWideCompactRows);
        rowBase = nextBase;
        if (rowBase < blockLength_) {
            ComputeChunkCompact(blockLength_ - rowBase);
            CopyOut(rowBase, blockLength_ - rowBase);
        }
    }

    __aicore__ inline void ProcessRowsPipelinedCompact2()
    {
        const uint32_t tailCols = AlignUp(reduceLength_ - tileCols_, kBlockBytes / sizeof(T));
        uint32_t rowBase = 0U;
        uint32_t rows = blockLength_ < tileRows_ ? blockLength_ : tileRows_;
        CopyIn(0U, 0U, rows);
        CopyIn(0U, tileCols_, rows, tailCols);
        while (rowBase < blockLength_) {
            ComputeChunkCompactAccumulate<true>(rows, tileCols_);
            const uint32_t nextBase = rowBase + rows;
            const uint32_t nextRows = blockLength_ - nextBase < tileRows_
                                          ? blockLength_ - nextBase : tileRows_;
            if (nextBase < blockLength_) {
                CopyIn(nextBase, 0U, nextRows);
            }
            ComputeChunkCompactAccumulate(rows, tailCols);
            if (nextBase < blockLength_) {
                CopyIn(nextBase, tileCols_, nextRows, tailCols);
            }
            CopyOut(rowBase, rows);
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    template <uint32_t FixedRows = 0U>
    __aicore__ inline void ComputeChunkCompact(uint32_t runtimeRows = 0U)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> acc = accFloatBuf_.Get<float>();
        const uint32_t rows = FixedRows == 0U ? runtimeRows : FixedRows;
        const uint32_t cols = FixedRows == 0U ? tileCols_ : kSuffixWideCols;
        const uint32_t elems = rows * cols;
        CastInputToFloat<T>(inputFloat, input, elems);
        PipeBarrier<PIPE_V>();
        Mul(inputFloat, inputFloat, inputFloat, elems);
        PipeBarrier<PIPE_V>();
        uint32_t shape[2] = {rows, cols};
        ReduceSum<float, Pattern::Reduce::AR, false>(
            acc, inputFloat, input.template ReinterpretCast<uint8_t>(), shape, false);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
    }

    template <bool Assign = false>
    __aicore__ inline void ComputeChunkCompactAccumulate(uint32_t rows, uint32_t cols)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> reduced = partialFloatBuf_.Get<float>();
        LocalTensor<float> acc = accFloatBuf_.Get<float>();
        const uint32_t elems = rows * cols;
        CastInputToFloat<T>(inputFloat, input, elems);
        PipeBarrier<PIPE_V>();
        Mul(inputFloat, inputFloat, inputFloat, elems);
        PipeBarrier<PIPE_V>();
        uint32_t shape[2] = {rows, cols};
        ReduceSum<float, Pattern::Reduce::AR, false>(
            reduced, inputFloat, input.template ReinterpretCast<uint8_t>(), shape, false);
        PipeBarrier<PIPE_V>();
        if constexpr (Assign) {
            Adds(acc, reduced, 0.0F, accRows_);
        } else {
            Add(acc, acc, reduced, accRows_);
        }
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void ProcessSerial()
    {
        for (uint32_t rowBase = 0; rowBase < blockLength_; rowBase += tileRows_) {
            const uint32_t rows = blockLength_ - rowBase < tileRows_
                                      ? blockLength_ - rowBase : tileRows_;
            LocalTensor<float> acc = accFloatBuf_.Get<float>();
            Duplicate(acc, 0.0F, accRows_);
            for (uint32_t colBase = 0; colBase < reduceLength_; colBase += tileCols_) {
                const uint32_t cols = reduceLength_ - colBase < tileCols_
                                          ? reduceLength_ - colBase : tileCols_;
                LoadSquare(rowBase, colBase, rows, cols);
                ReduceChunk(rows);
            }
            CopyOut(rowBase, rows);
        }
    }

    __aicore__ inline void LoadSquare(uint32_t rowBase, uint32_t colBase,
                                     uint32_t rows, uint32_t cols)
    {
        LocalTensor<T> input = inputBuf_.Get<T>();
        LoadInput(input, rowBase, colBase, rows, cols);
        PipeBarrier<PIPE_ALL>();
        SquareInput(input, rows);
    }

    __aicore__ inline void CopyIn(uint32_t rowBase, uint32_t colBase, uint32_t rows,
                                 uint32_t packedCols = 0U)
    {
        LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        const uint32_t cols = reduceLength_ - colBase < tileCols_
                                  ? reduceLength_ - colBase : tileCols_;
        LoadInput(input, rowBase, colBase, rows, cols,
                  packedCols == 0U ? tileCols_ : packedCols);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void LoadInput(LocalTensor<T>& input, uint32_t rowBase, uint32_t colBase,
                                    uint32_t rows, uint32_t cols, uint32_t rowCols = 0U)
    {
        rowCols = rowCols == 0U ? tileCols_ : rowCols;
        const uint32_t elemAlign = kBlockBytes / sizeof(T);
        const uint32_t alignedCols = AlignUp(cols, elemAlign);
        if (alignedCols < rowCols) {
            if constexpr (Compact) {
                Duplicate(input[alignedCols], static_cast<T>(0),
                          static_cast<uint64_t>(rowCols - alignedCols),
                          static_cast<uint8_t>(rows), 1,
                          static_cast<uint8_t>(rowCols * sizeof(T) / kBlockBytes));
            } else {
                Duplicate(input, static_cast<T>(0), rows * rowCols);
            }
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((reduceLength_ - cols) * sizeof(T)),
            static_cast<uint32_t>((rowCols - alignedCols) * sizeof(T) / kBlockBytes), 0};
        DataCopyPadExtParams<T> padParams = {
            true, 0, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        DataCopyPad(input, inputGm_[rowBase * reduceLength_ + colBase], copyParams, padParams);
    }

    __aicore__ inline void ComputeChunk(uint32_t rows)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> square = squareFloatBuf_.Get<float>();
        const uint32_t elems = rows * tileCols_;
        CastInputToFloat<T>(inputFloat, input, elems);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Mul(square, inputFloat, inputFloat, elems);
        PipeBarrier<PIPE_V>();
        ReduceChunk(rows);
    }

    __aicore__ inline void SquareInput(LocalTensor<T>& input, uint32_t rows)
    {
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> square = squareFloatBuf_.Get<float>();
        const uint32_t elems = rows * tileCols_;
        CastInputToFloat<T>(inputFloat, input, elems);
        PipeBarrier<PIPE_V>();
        Mul(square, inputFloat, inputFloat, elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceChunk(uint32_t rows)
    {
        LocalTensor<float> square = squareFloatBuf_.Get<float>();
        LocalTensor<float> partial = partialFloatBuf_.Get<float>();
        LocalTensor<float> reduced = inputFloatBuf_.Get<float>();
        LocalTensor<float> acc = accFloatBuf_.Get<float>();
        if (tileCols_ < kSuffixWideCols) {
            uint32_t shape[2] = {rows, tileCols_};
            ReduceSum<float, Pattern::Reduce::AR, true>(
                reduced, square, partial.ReinterpretCast<uint8_t>(), shape, false);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, reduced, accRows_);
            PipeBarrier<PIPE_V>();
            return;
        }
        WholeReduceSum(partial, square, kGenericReduceCols,
                       rows * segments_, 1, 1,
                       kGenericReduceCols * sizeof(float) / kBlockBytes);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduced, partial, segments_,
                       rows, 1, 1, segments_ * sizeof(float) / kBlockBytes);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, reduced, accRows_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(uint32_t rowBase, uint32_t rows)
    {
        LocalTensor<float> acc = accFloatBuf_.Get<float>();
        if constexpr (Compact) {
            LocalTensor<T> output = outputQueue_.template AllocTensor<T>();
            CastFloatToOutput<T>(output, acc, accRows_);
            outputQueue_.EnQue(output);
            output = outputQueue_.template DeQue<T>();
            DataCopyExtParams params = {1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[rowBase], output, params);
            outputQueue_.FreeTensor(output);
            return;
        }
        LocalTensor<T> output = outputBuf_.Get<T>();
        CastFloatToOutput<T>(output, acc, accRows_);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams params = {1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[rowBase], output, params);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TQue<QuePosition::VECIN, kInputBuffers> inputQueue_;
    TBuf<QuePosition::VECCALC> inputFloatBuf_;
    TBuf<QuePosition::VECCALC> squareFloatBuf_;
    TBuf<QuePosition::VECCALC> partialFloatBuf_;
    TBuf<QuePosition::VECCALC> accFloatBuf_;
    TBuf<QuePosition::VECCALC> outputBuf_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t reduceLength_ = 0;
    uint32_t tileCols_ = kSuffixWideCols;
    uint32_t tileRows_ = kSuffixWideRows;
    uint32_t segments_ = kSuffixWideSegments;
    uint32_t accRows_ = kSuffixWideAlignedRows;
    bool useNarrowTail_ = false;
};

template <typename T, bool DoubleBuffered = false, bool Compact = false>
__aicore__ inline void RunSquareSumV1SuffixWide(GM_ADDR input, GM_ADDR output,
                                                uint32_t smallCoreItems, uint32_t bigCoreCount,
                                                uint32_t reduceLength)
{
    uint32_t blockOffset = 0;
    uint32_t blockLength = 0;
    GetCoreRange(smallCoreItems, bigCoreCount, blockOffset, blockLength);
    KernelSquareSumV1SuffixWide<T, DoubleBuffered, Compact> kernel;
    kernel.Init(input, output, blockOffset, blockLength, reduceLength);
    kernel.Process();
}

template <typename T, bool SplitFp32Pipeline = false,
          uint32_t Bf16CycleRows = kBf16CycleRows>
class KernelSquareSumV1Generic {
public:
    __aicore__ inline KernelSquareSumV1Generic() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const SquareSumV1TilingData& tiling, bool bf16Axis12 = false)
    {
        inputGm_.SetGlobalBuffer((__gm__ T*)input, tiling.inputSize);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, tiling.outputSize);
        workspaceGm_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace),
                                     GetBlockNum() * AlignUp(tiling.genericTileCols,
                                                              kBlockBytes / sizeof(float)));
        rank_ = tiling.rank;
        axisMask_ = tiling.axisMask;
        smallCoreItems_ = tiling.smallCoreItems;
        bigCoreCount_ = tiling.bigCoreCount;
        genericTileCols_ = tiling.genericTileCols;

        const uint32_t* dims = tiling.dims;
        inputSize_ = 1;
        outputRank_ = 0;
        reduceSize_ = 1;
        for (uint32_t idx = 0; idx < kMaxRank; ++idx) {
            dims_[idx] = (idx < rank_) ? dims[idx] : 1;
        }
        for (int32_t idx = static_cast<int32_t>(rank_) - 1; idx >= 0; --idx) {
            inputStrides_[idx] = inputSize_;
            inputSize_ *= dims_[idx];
        }
        for (uint32_t idx = 0; idx < rank_; ++idx) {
            if ((axisMask_ & (1U << idx)) == 0) {
                outputDims_[outputRank_] = idx;
                ++outputRank_;
            } else {
                reduceSize_ *= dims_[idx];
            }
        }

        uint32_t reduceRank = 0;
        for (uint32_t idx = 0; idx < rank_; ++idx) {
            if ((axisMask_ & (1U << idx)) != 0) {
                reduceDims_[reduceRank] = idx;
                reduceStrides_[reduceRank] = inputStrides_[idx];
                ++reduceRank;
            }
        }
        reduceRank_ = reduceRank;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const bool useNarrowAxis12 = rank_ == 4U && axisMask_ == 0x6U &&
                                         dims_[3] <= 255U &&
                                         dims_[3] % (kBlockBytes / sizeof(T)) != 0U;
            if (useNarrowAxis12) {
                inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                if constexpr (Bf16CycleRows == 64U) {
                    workspacePartialGm_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace),
                        2U * (GetBlockNum() - 1U) *
                            AlignUp(dims_[3], kBlockBytes / sizeof(half)));
                }
            }
        }

        const bool lastDimReduced = (axisMask_ & (1U << (rank_ - 1U))) != 0;
        if (lastDimReduced) {
            const uint32_t dataElems = kGenericReduceRows * kGenericReduceCols;
            pipe_.InitBuffer(inputBuf_, dataElems * sizeof(T));
            pipe_.InitBuffer(inputFloatBuf_, dataElems * sizeof(float));
            pipe_.InitBuffer(squareFloatBuf_, dataElems * sizeof(float));
            pipe_.InitBuffer(accFloatBuf_, kGenericReduceRows * sizeof(float));
            pipe_.InitBuffer(partialFloatBuf_, kGenericReduceRows * sizeof(float));
            pipe_.InitBuffer(outputBuf_, kGenericReduceRows * sizeof(T));
            pipe_.InitBuffer(offsetBuf_,
                             kGenericTransposeMaxCols * (kBlockBytes / sizeof(float)) * sizeof(uint32_t));
            return;
        }
        if (reduceSize_ < kGenericBatchMinReduce) {
            keptDataElems_ = kGenericKeptElems;
            pipe_.InitBuffer(inputBuf_, keptDataElems_ * sizeof(T));
            pipe_.InitBuffer(inputFloatBuf_, keptDataElems_ * sizeof(float));
            pipe_.InitBuffer(squareFloatBuf_, keptDataElems_ * sizeof(float));
            pipe_.InitBuffer(accFloatBuf_, keptDataElems_ * sizeof(float));
            pipe_.InitBuffer(outputBuf_, keptDataElems_ * sizeof(T));
            return;
        }
        if constexpr (SplitFp32Pipeline) {
            const uint32_t alignedTileCols = AlignUp(genericTileCols_, kBlockBytes / sizeof(float));
            const uint32_t innerDim = dims_[rank_ - 1U];
            useCycleFp32_ = genericTileCols_ == innerDim && innerDim >= 513U &&
                            innerDim <= kFp32CycleMaxCols &&
                            reduceSize_ > 4096U && innerDim % (kBlockBytes / sizeof(float)) != 0U;
            if (useCycleFp32_) {
                inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                cycleAlignedCols_ = AlignUp(innerDim, kBlockBytes / sizeof(float));
                cycleGroupElems_ = kCycleGroupRows * innerDim;
                pipe_.InitBuffer(splitInputQueue_, 2, 2U * cycleGroupElems_ * sizeof(float));
                pipe_.InitBuffer(inputFloatBuf_,
                                 cycleGroupElems_ * sizeof(float) + kFp32CycleBankOffsetBytes);
                pipe_.InitBuffer(squareFloatBuf_,
                                 (kCycleGroupRows / 2U) * cycleAlignedCols_ * sizeof(float));
                pipe_.InitBuffer(offsetBuf_, innerDim * sizeof(uint32_t));
            } else {
                keptDataElems_ = kSplitFp32Elems;
                pipe_.InitBuffer(splitInputQueue_, 2, keptDataElems_ * sizeof(float));
                pipe_.InitBuffer(accFloatBuf_, alignedTileCols * sizeof(float));
            }
            pipe_.InitBuffer(splitOutputQueue_, 1, alignedTileCols * sizeof(float));
            return;
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (bf16Axis12 && rank_ == 4U && axisMask_ == 0x6U &&
                dims_[3] <= 511U && dims_[3] % 16U != 0U) {
                constexpr bool useBalancedAcc = Bf16CycleRows == 64U;
                bf16AlignedInner_ = AlignUp(dims_[3], kBlockBytes / sizeof(T));
                bf16GroupElems_ = Bf16CycleRows * dims_[3];
                pipe_.InitBuffer(splitInputQueue_, useBalancedAcc ? 2 : 3,
                                 (useBalancedAcc ? 2U : 1U) * bf16GroupElems_ * sizeof(T));
                const uint32_t castElems = useBalancedAcc ? bf16GroupElems_ / 2U : bf16GroupElems_;
                const uint32_t scratchElems = useBalancedAcc && castElems < 8U * bf16AlignedInner_
                                                  ? 8U * bf16AlignedInner_ : castElems;
                pipe_.InitBuffer(squareFloatBuf_, scratchElems * sizeof(float));
                pipe_.InitBuffer(inputFloatBuf_,
                                 (useBalancedAcc ? bf16GroupElems_ / 2U : bf16GroupElems_) * sizeof(float) +
                                     kCycleBankOffsetBytes);
                if constexpr (!useBalancedAcc) {
                    pipe_.InitBuffer(partialFloatBuf_,
                                     Bf16CycleRows * sizeof(float) * bf16AlignedInner_);
                    pipe_.InitBuffer(offsetBuf_, dims_[3] * sizeof(uint32_t));
                    pipe_.InitBuffer(splitOutputQueue_, 2, bf16AlignedInner_ * sizeof(T));
                }
                return;
            }
        }
        const uint32_t innerReduceDim = reduceDims_[reduceRank_ - 1U];
        const uint32_t trailingKeptLength = inputStrides_[innerReduceDim];
        const uint32_t innerDim = trailingKeptLength >= kGenericFlattenMinCols
                                      ? trailingKeptLength
                                      : dims_[rank_ - 1U];
        usePacked_ = dims_[innerReduceDim] >= kGenericReduceCols &&
                     reduceStrides_[reduceRank_ - 1U] == innerDim &&
                     innerDim <= kGenericPackedCols;
        keptDataElems_ = usePacked_ ? kGenericKeptElems
                                    : (sizeof(T) == sizeof(float)
                                           ? kGenericPlainFloatElems
                                           : kGenericPlainHalfElems);
        pipe_.InitBuffer(inputBuf_, keptDataElems_ * sizeof(T));
        pipe_.InitBuffer(squareFloatBuf_, keptDataElems_ * sizeof(float));
        if (usePacked_) {
            pipe_.InitBuffer(inputFloatBuf_, keptDataElems_ * sizeof(float));
        }
        pipe_.InitBuffer(accFloatBuf_, kGenericBatchCols * sizeof(float));
        pipe_.InitBuffer(outputBuf_, kGenericBatchCols * sizeof(T));
        if (usePacked_) {
            pipe_.InitBuffer(offsetBuf_, keptDataElems_ * sizeof(uint32_t));
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        const uint32_t end = begin + length;
        if ((axisMask_ & (1U << (rank_ - 1U))) == 0) {
            if (reduceSize_ < kGenericBatchMinReduce) {
                ProcessLastDimKeptLinear(begin, end);
                return;
            }
            ProcessLastDimKept(begin, end);
            return;
        }
        ProcessLastDimReduced(begin, end);
    }

    __aicore__ inline void ProcessRank4Axis12Bf16()
    {
        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        const uint32_t innerDim = dims_[3];
        const uint32_t tileRows = keptDataElems_ / innerDim;
        LocalTensor<T> input = inputBuf_.Get<T>();
        LocalTensor<float> square = squareFloatBuf_.Get<float>();
        LocalTensor<float> acc = accFloatBuf_.Get<float>();

        for (uint32_t outputRow = begin; outputRow < begin + length; ++outputRow) {
            for (uint32_t rowBase = 0; rowBase < reduceSize_; rowBase += tileRows) {
                const uint32_t rows = reduceSize_ - rowBase < tileRows
                                          ? reduceSize_ - rowBase : tileRows;
                const uint32_t tileElems = rows * innerDim;
                DataCopyExtParams copyParams = {
                    1, static_cast<uint32_t>(tileElems * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
                DataCopyPad(input,
                            inputGm_[(outputRow * reduceSize_ + rowBase) * innerDim],
                            copyParams, padParams);
                PipeBarrier<PIPE_ALL>();
                SquareInputToFloat<T>(square, input, tileElems);
                PipeBarrier<PIPE_V>();
                ReduceRows(square, rows, innerDim);
                if (rowBase == 0U) {
                    Adds(acc, square, 0.0F, static_cast<int32_t>(innerDim));
                } else {
                    Add(acc, acc, square, static_cast<int32_t>(innerDim));
                }
                PipeBarrier<PIPE_V>();
            }
            CopyOutputContiguous(outputRow * innerDim, innerDim,
                                 AlignUp(innerDim, kBlockBytes / sizeof(T)));
        }
    }

    __aicore__ inline void ProcessRank4Axis12Bf16Padded()
    {
        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        const uint32_t innerDim = dims_[3];
        LocalTensor<float> castLocal = squareFloatBuf_.Get<float>();
        LocalTensor<float> groupAcc = inputFloatBuf_.GetWithOffset<float>(
            bf16GroupElems_, kCycleBankOffsetBytes);
        LocalTensor<float> padded = partialFloatBuf_.Get<float>();
        LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsets, 0, sizeof(float), innerDim);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> gatherOffsets = offsets.ReinterpretCast<uint32_t>();

        uint32_t prefetchedRows = length == 0U ? 0U : PrefetchBf16Output(begin, innerDim);
        for (uint32_t outputRow = begin; outputRow < begin + length; ++outputRow) {
            Duplicate(groupAcc, 0.0F, static_cast<int32_t>(bf16GroupElems_));
            PipeBarrier<PIPE_V>();
            uint32_t rowBase = 0U;
            while (rowBase < reduceSize_) {
                const uint32_t rows = reduceSize_ - rowBase < Bf16CycleRows
                                          ? reduceSize_ - rowBase : Bf16CycleRows;
                if (prefetchedRows < reduceSize_) {
                    const uint32_t nextRows = reduceSize_ - prefetchedRows < Bf16CycleRows
                                                  ? reduceSize_ - prefetchedRows : Bf16CycleRows;
                    CopyInBf16Compact((outputRow * reduceSize_ + prefetchedRows) * innerDim,
                                      nextRows * innerDim);
                    prefetchedRows += nextRows;
                }
                LocalTensor<T> input = splitInputQueue_.DeQue<T>();
                CastInputToFloat<T>(castLocal, input, rows * innerDim);
                PipeBarrier<PIPE_V>();
                MulAddDst(groupAcc, castLocal, castLocal,
                          static_cast<int32_t>(rows * innerDim));
                splitInputQueue_.FreeTensor(input);
                rowBase += rows;
            }
            prefetchedRows = outputRow + 1U < begin + length
                                 ? PrefetchBf16Output(outputRow + 1U, innerDim) : 0U;
            PipeBarrier<PIPE_V>();
            constexpr uint32_t gatherRows = Bf16CycleRows == 32U ? 8U : Bf16CycleRows;
            if constexpr (Bf16CycleRows == 32U) {
                FoldCycleRowsUntil<32U, gatherRows>(groupAcc, innerDim);
            }
            Duplicate(padded, 0.0F,
                      static_cast<int32_t>(gatherRows * bf16AlignedInner_));
            GatherCycleRows<0U, gatherRows>(padded, groupAcc, gatherOffsets,
                                            innerDim, bf16AlignedInner_);
            PipeBarrier<PIPE_V>();
            FoldCycleRows<gatherRows>(padded, bf16AlignedInner_);
            LocalTensor<T> output = splitOutputQueue_.AllocTensor<T>();
            CastFloatToOutput<T>(output, padded, bf16AlignedInner_);
            splitOutputQueue_.EnQue(output);
            output = splitOutputQueue_.DeQue<T>();
            DataCopyExtParams outputParams = {
                1, static_cast<uint32_t>(innerDim * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outputRow * innerDim], output, outputParams);
            splitOutputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline uint32_t ReduceBf16FloatSegment(uint32_t inputRowBegin,
                                                      uint32_t rowCount,
                                                      uint32_t nextRowCount,
                                                      uint32_t prefetchedRows,
                                                      LocalTensor<float>& castLocal,
                                                      LocalTensor<float>& groupAcc)
    {
        const uint32_t innerDim = dims_[3];
        uint32_t nextPrefetchedRows = 0U;
        constexpr uint32_t readRows = 2U * Bf16CycleRows;
        const uint32_t accElems = bf16GroupElems_ / 2U;
        Duplicate(groupAcc, 0.0F, static_cast<int32_t>(accElems));
        PipeBarrier<PIPE_V>();
        for (uint32_t rowBase = 0U; rowBase < rowCount;) {
            const uint32_t rows = rowCount - rowBase < readRows
                                      ? rowCount - rowBase : readRows;
            if (prefetchedRows < rowCount) {
                const uint32_t nextRows = rowCount - prefetchedRows < readRows
                                              ? rowCount - prefetchedRows : readRows;
                CopyInBf16Compact((inputRowBegin + prefetchedRows) * innerDim,
                                  nextRows * innerDim);
                prefetchedRows += nextRows;
            } else if (nextPrefetchedRows < nextRowCount &&
                       nextPrefetchedRows < 2U * readRows) {
                const uint32_t nextRows = nextRowCount - nextPrefetchedRows < readRows
                                              ? nextRowCount - nextPrefetchedRows
                                              : readRows;
                CopyInBf16Compact((inputRowBegin + rowCount + nextPrefetchedRows) * innerDim,
                                  nextRows * innerDim);
                nextPrefetchedRows += nextRows;
            }
            LocalTensor<T> input = splitInputQueue_.DeQue<T>();
            const uint32_t elems = rows * innerDim;
            #pragma unroll
            for (uint32_t group = 0U; group < 4U; ++group) {
                const uint32_t offset = group * accElems;
                if (offset < elems) {
                    const uint32_t count = elems - offset < accElems ? elems - offset : accElems;
                    auto source = input[offset];
                    CastInputToFloat<T>(castLocal, source, count);
                    PipeBarrier<PIPE_V>();
                    MulAddDst(groupAcc, castLocal, castLocal, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                }
            }
            splitInputQueue_.FreeTensor(input);
            rowBase += rows;
        }
        PipeBarrier<PIPE_V>();
        FoldCycleRowsUntil<32U, 8U>(groupAcc, innerDim);
        LocalTensor<int32_t> offsets =
            groupAcc[8U * innerDim].template ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(offsets, 0, sizeof(float), innerDim);
        LocalTensor<float> padded = squareFloatBuf_.Get<float>();
        Duplicate(padded, 0.0F,
                  static_cast<int32_t>(8U * bf16AlignedInner_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> gatherOffsets = offsets.ReinterpretCast<uint32_t>();
        GatherCycleRows<0U, 8U>(padded, groupAcc, gatherOffsets,
                                 innerDim, bf16AlignedInner_);
        PipeBarrier<PIPE_V>();
        FoldCycleRows<8U>(padded, bf16AlignedInner_);
        return nextPrefetchedRows;
    }

    __aicore__ inline void CopyBf16Result(uint32_t outputRow,
                                               LocalTensor<float>& result,
                                               LocalTensor<float>& scratch)
    {
        const uint32_t innerDim = dims_[3];
        LocalTensor<float> outputFloat = scratch.template ReinterpretCast<float>();
        Adds(outputFloat, result, 0.0F, bf16AlignedInner_);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> output = result.template ReinterpretCast<T>();
        CastFloatToOutput<T>(output, outputFloat, bf16AlignedInner_);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outputParams = {
            1, static_cast<uint32_t>(innerDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outputRow * innerDim], output, outputParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRank4Axis12Bf16Balanced()
    {
        constexpr uint32_t readRows = 2U * Bf16CycleRows;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t totalRows = dims_[0] * reduceSize_;
        const uint32_t smallRows = totalRows / blockNum;
        const uint32_t bigCount = totalRows % blockNum;
        uint32_t globalRow = blockIdx * smallRows + (blockIdx < bigCount ? blockIdx : bigCount);
        const uint32_t blockRows = smallRows + (blockIdx < bigCount ? 1U : 0U);
        const uint32_t globalEnd = globalRow + blockRows;
        LocalTensor<float> castLocal = squareFloatBuf_.Get<float>();
        LocalTensor<float> groupAcc = inputFloatBuf_.GetWithOffset<float>(
            bf16GroupElems_ / 2U, kCycleBankOffsetBytes);
        LocalTensor<float> padded = squareFloatBuf_.Get<float>();
        uint32_t prefetchedRows = 0U;

        while (globalRow < globalEnd) {
            const uint32_t outputRow = globalRow / reduceSize_;
            const uint32_t reduceBegin = globalRow - outputRow * reduceSize_;
            const uint32_t rows = globalEnd - globalRow < reduceSize_ - reduceBegin
                                      ? globalEnd - globalRow : reduceSize_ - reduceBegin;
            const uint32_t nextGlobalRow = globalRow + rows;
            const uint32_t nextRows = nextGlobalRow < globalEnd
                                          ? (globalEnd - nextGlobalRow < reduceSize_
                                                 ? globalEnd - nextGlobalRow
                                                 : reduceSize_)
                                          : 0U;
            uint32_t queuedTiles =
                (prefetchedRows + readRows - 1U) / readRows;
            while (queuedTiles < 1U && prefetchedRows < rows) {
                const uint32_t loadRows = rows - prefetchedRows < readRows
                                              ? rows - prefetchedRows : readRows;
                CopyInBf16Compact((globalRow + prefetchedRows) * dims_[3],
                                  loadRows * dims_[3]);
                prefetchedRows += loadRows;
                ++queuedTiles;
            }
            prefetchedRows = ReduceBf16FloatSegment(globalRow, rows, nextRows,
                                                    prefetchedRows, castLocal, groupAcc);
            if (reduceBegin == 0U && rows == reduceSize_) {
                CopyBf16Result(outputRow, padded, groupAcc);
            } else {
                const uint32_t slot = reduceBegin == 0U ? 2U * blockIdx
                                                        : 2U * (blockIdx - 1U) + 1U;
                PipeBarrier<PIPE_ALL>();
                DataCopy(workspacePartialGm_[slot * bf16AlignedInner_], padded, bf16AlignedInner_);
                PipeBarrier<PIPE_ALL>();
            }
            globalRow += rows;
        }

        SyncAll<true>();
        if (blockIdx + 1U >= blockNum) {
            return;
        }
        const uint32_t boundaryCore = blockIdx + 1U;
        const uint32_t boundaryRow = boundaryCore * smallRows +
                                     (boundaryCore < bigCount ? boundaryCore : bigCount);
        if (boundaryRow % reduceSize_ == 0U) {
            return;
        }
        DataCopy(groupAcc, workspacePartialGm_[2U * blockIdx * bf16AlignedInner_],
                 2U * bf16AlignedInner_);
        PipeBarrier<PIPE_ALL>();
        Add(padded, groupAcc, groupAcc[bf16AlignedInner_],
            static_cast<int32_t>(bf16AlignedInner_));
        PipeBarrier<PIPE_V>();
        CopyBf16Result(boundaryRow / reduceSize_, padded, groupAcc);
    }

    __aicore__ inline void ProcessRank4Axis03Batches()
    {
        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        const uint32_t outputSize = inputSize_ / reduceSize_;
        const uint32_t batchRows = dims_[0] <= 16U && dims_[3] % 8U == 0U ? 32U
                                   : (dims_[0] <= 11U ? 16U : 8U);
        const uint32_t outputBegin = begin * batchRows;
        uint32_t outputEnd = (begin + length) * batchRows;
        if (outputEnd > outputSize) {
            outputEnd = outputSize;
        }
        ProcessLastDimReduced(outputBegin, outputEnd);
    }

    __aicore__ inline void ProcessSplitKept()
    {
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        const uint32_t innerDim = inputStrides_[reduceDims_[reduceRank_ - 1U]];
        const uint32_t outputRows = inputSize_ / (reduceSize_ * innerDim);
        const uint32_t colTiles = (innerDim + genericTileCols_ - 1U) / genericTileCols_;
        const uint32_t outputTasks = outputRows * colTiles;
        const uint32_t splitCount = GetBlockNum() / outputTasks;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t outputTask = blockIdx / splitCount;
        const uint32_t splitIdx = blockIdx % splitCount;
        const uint32_t smallRows = reduceSize_ / splitCount;
        const uint32_t bigCount = reduceSize_ % splitCount;
        const uint32_t reduceBegin = splitIdx * smallRows +
                                     (splitIdx < bigCount ? splitIdx : bigCount);
        const uint32_t reduceRows = smallRows + (splitIdx < bigCount ? 1U : 0U);
        const uint32_t outputRow = outputTask / colTiles;
        const uint32_t colBase = (outputTask % colTiles) * genericTileCols_;
        const uint32_t cols = innerDim - colBase < genericTileCols_
                                  ? innerDim - colBase : genericTileCols_;
        const uint32_t alignedCols = AlignUp(cols, kBlockBytes / sizeof(T));
        const uint32_t tileRows = keptDataElems_ / alignedCols;

        Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedCols));
        if (usePacked_) {
            uint32_t packedRows = AlignDown(keptDataElems_ / innerDim, kGenericPackedRowAlign);
            if (packedRows > kGenericPackedMaxRows) {
                packedRows = kGenericPackedMaxRows;
            }
            BuildPackedOffsets(innerDim, packedRows);
            for (uint32_t rowBase = 0; rowBase < reduceRows; rowBase += packedRows) {
                const uint32_t rows = reduceRows - rowBase < packedRows
                                          ? reduceRows - rowBase : packedRows;
                const uint32_t inputOffset = ((outputRow * reduceSize_ + reduceBegin + rowBase) * innerDim);
                LoadSquarePacked(inputOffset, rows * innerDim, packedRows * innerDim);
                AccumulatePackedTile(innerDim, packedRows, alignedCols);
            }
        } else {
            for (uint32_t rowBase = 0; rowBase < reduceRows; rowBase += tileRows) {
                const uint32_t rows = reduceRows - rowBase < tileRows
                                          ? reduceRows - rowBase : tileRows;
                const uint32_t inputOffset = ((outputRow * reduceSize_ + reduceBegin + rowBase) * innerDim) + colBase;
                LoadSquareRowsPlain(inputOffset, rows, cols, alignedCols, innerDim);
                ReduceRows(squareFloat, rows, alignedCols);
                Add(accFloat, accFloat, squareFloat, static_cast<int32_t>(alignedCols));
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGm_[blockIdx * genericTileCols_], accFloat, alignedCols);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();

        if (blockIdx >= outputTasks) {
            return;
        }
        const uint32_t ownerRow = blockIdx / colTiles;
        const uint32_t ownerColBase = (blockIdx % colTiles) * genericTileCols_;
        const uint32_t ownerCols = innerDim - ownerColBase < genericTileCols_
                                       ? innerDim - ownerColBase : genericTileCols_;
        const uint32_t ownerAlignedCols = AlignUp(ownerCols, kBlockBytes / sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (ownerAlignedCols == genericTileCols_ &&
                splitCount * ownerAlignedCols <= keptDataElems_) {
                DataCopy(squareFloat,
                         workspaceGm_[blockIdx * splitCount * genericTileCols_],
                         splitCount * ownerAlignedCols);
                PipeBarrier<PIPE_ALL>();
                ReduceRows(squareFloat, splitCount, ownerAlignedCols);
                Adds(accFloat, squareFloat, 0.0F, static_cast<int32_t>(ownerAlignedCols));
                CopyOutputContiguous(ownerRow * innerDim + ownerColBase,
                                     ownerCols, ownerAlignedCols);
                return;
            }
        }
        Duplicate(accFloat, 0.0F, static_cast<int32_t>(ownerAlignedCols));
        for (uint32_t split = 0; split < splitCount; ++split) {
            DataCopy(squareFloat,
                     workspaceGm_[(blockIdx * splitCount + split) * genericTileCols_],
                     ownerAlignedCols);
            PipeBarrier<PIPE_ALL>();
            Add(accFloat, accFloat, squareFloat, static_cast<int32_t>(ownerAlignedCols));
            PipeBarrier<PIPE_V>();
        }
        CopyOutputContiguous(ownerRow * innerDim + ownerColBase, ownerCols, ownerAlignedCols);
    }

    __aicore__ inline void ProcessSplitKeptFp32Pipeline()
    {
        if (useCycleFp32_) {
            ProcessSplitKeptFp32Cycle<true>();
            return;
        }
        LocalTensor<float> acc = accFloatBuf_.Get<float>();
        const uint32_t innerDim = inputStrides_[reduceDims_[reduceRank_ - 1U]];
        const uint32_t outputRows = inputSize_ / (reduceSize_ * innerDim);
        const uint32_t colTiles = (innerDim + genericTileCols_ - 1U) / genericTileCols_;
        const uint32_t outputTasks = outputRows * colTiles;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t splitCount = GetBlockNum() / outputTasks;
        const uint32_t outputTask = blockIdx / splitCount;
        const uint32_t splitIdx = blockIdx % splitCount;
        const uint32_t smallRows = reduceSize_ / splitCount;
        const uint32_t bigCount = reduceSize_ % splitCount;
        const uint32_t reduceBegin = splitIdx * smallRows +
                                     (splitIdx < bigCount ? splitIdx : bigCount);
        const uint32_t reduceRows = smallRows + (splitIdx < bigCount ? 1U : 0U);
        const uint32_t outputRow = outputTask / colTiles;
        const uint32_t colBase = (outputTask % colTiles) * genericTileCols_;
        const uint32_t cols = innerDim - colBase < genericTileCols_
                                  ? innerDim - colBase : genericTileCols_;
        const uint32_t alignedCols = AlignUp(cols, kBlockBytes / sizeof(float));
        const uint32_t outputOffset = outputRow * innerDim + colBase;
        const uint32_t tileRows = keptDataElems_ / alignedCols;

        if (splitIdx == 0U) {
            LocalTensor<float> output = splitOutputQueue_.AllocTensor<float>();
            Duplicate(output, 0.0F, static_cast<int32_t>(alignedCols));
            splitOutputQueue_.EnQue(output);
            output = splitOutputQueue_.DeQue<float>();
            CopyOutSplitFp32(output, outputOffset, cols, alignedCols);
            event_t eventId = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventId);
            WaitFlag<HardEvent::MTE3_S>(eventId);
            pipe_.ReleaseEventID<HardEvent::MTE3_S>(eventId);
            splitOutputQueue_.FreeTensor(output);
        }
        SyncAll<true>();
        Duplicate(acc, 0.0F, static_cast<int32_t>(alignedCols));
        CopyInSplitFp32(((outputRow * reduceSize_ + reduceBegin) * innerDim) + colBase,
                        reduceRows < tileRows ? reduceRows : tileRows,
                        cols, alignedCols, innerDim);
        for (uint32_t rowBase = 0; rowBase < reduceRows; rowBase += tileRows) {
            const uint32_t rows = reduceRows - rowBase < tileRows
                                      ? reduceRows - rowBase : tileRows;
            const uint32_t nextBase = rowBase + tileRows;
            if (nextBase < reduceRows) {
                const uint32_t nextRows = reduceRows - nextBase < tileRows
                                              ? reduceRows - nextBase : tileRows;
                CopyInSplitFp32(((outputRow * reduceSize_ + reduceBegin + nextBase) * innerDim) + colBase,
                                nextRows, cols, alignedCols, innerDim);
            }
            LocalTensor<float> tile = splitInputQueue_.DeQue<float>();
            Mul(tile, tile, tile, static_cast<int32_t>(rows * alignedCols));
            PipeBarrier<PIPE_V>();
            ReduceRows(tile, rows, alignedCols);
            Add(acc, acc, tile, static_cast<int32_t>(alignedCols));
            PipeBarrier<PIPE_V>();
            splitInputQueue_.FreeTensor(tile);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();
        CopyOutSplitFp32(acc, outputOffset, cols, alignedCols);
        SetAtomicNone();
    }

private:
    __aicore__ inline void CopyInCycleFp32Rows(uint32_t outputTask, uint32_t rowBase,
                                               uint32_t rows)
    {
        LocalTensor<float> input = splitInputQueue_.AllocTensor<float>();
        constexpr uint32_t pairRows = 2U * kCycleGroupRows;
        const uint32_t inputOffset =
            (outputTask * reduceSize_ + rowBase) * dims_[rank_ - 1U];
        if (rows == pairRows) {
            DataCopy(input, inputGm_[inputOffset], 2U * cycleGroupElems_);
            splitInputQueue_.EnQue(input);
            return;
        }
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(rows * dims_[rank_ - 1U] * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {false, 0, 0, 0.0F};
        DataCopyPad(input, inputGm_[inputOffset], copyParams, padParams);
        splitInputQueue_.EnQue(input);
    }

    __aicore__ inline void CopyInCycleFp32Pair(uint32_t outputTask, uint32_t pairIndex)
    {
        constexpr uint32_t pairRows = 2U * kCycleGroupRows;
        const uint32_t rowBase = pairIndex * pairRows;
        const uint32_t rows = reduceSize_ - rowBase < pairRows ? reduceSize_ - rowBase : pairRows;
        CopyInCycleFp32Rows(outputTask, rowBase, rows);
    }

    template <uint32_t Row, uint32_t Rows, typename U>
    __aicore__ inline void GatherCycleRows(LocalTensor<U>& padded,
                                            LocalTensor<U>& compact,
                                            LocalTensor<uint32_t>& offsets,
                                            uint32_t cols, uint32_t alignedCols)
    {
        Gather(padded[Row * alignedCols], compact, offsets,
               Row * cols * sizeof(U), cols);
        if constexpr (Row + 1U < Rows) {
            GatherCycleRows<Row + 1U, Rows>(padded, compact, offsets, cols, alignedCols);
        }
    }

    template <uint32_t Rows, typename U>
    __aicore__ inline void FoldCycleRows(LocalTensor<U>& data, uint32_t rowWidth)
    {
        constexpr uint32_t pairs = Rows / 2U;
        Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
        PipeBarrier<PIPE_V>();
        if constexpr (pairs > 1U) {
            FoldCycleRows<pairs>(data, rowWidth);
        }
    }

    template <uint32_t Rows, uint32_t StopRows, typename U>
    __aicore__ inline void FoldCycleRowsUntil(LocalTensor<U>& data, uint32_t rowWidth)
    {
        constexpr uint32_t pairs = Rows / 2U;
        Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
        PipeBarrier<PIPE_V>();
        if constexpr (pairs > StopRows) {
            FoldCycleRowsUntil<pairs, StopRows>(data, rowWidth);
        }
    }

    template <bool RotateLanes>
    __aicore__ inline void ProcessSplitKeptFp32Cycle()
    {
        const uint32_t innerDim = dims_[rank_ - 1U];
        const uint32_t outputTasks = inputSize_ / (reduceSize_ * innerDim);
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t splitCount = GetBlockNum() / outputTasks;
        const uint32_t outputTask = blockIdx / splitCount;
        const uint32_t splitIdx = blockIdx % splitCount;
        constexpr uint32_t pairRows = 2U * kCycleGroupRows;
        const uint32_t totalPairs = (reduceSize_ + pairRows - 1U) / pairRows;
        const uint32_t smallPairs = totalPairs / splitCount;
        const uint32_t bigPairCount = totalPairs % splitCount;
        const uint32_t remainingRows = reduceSize_ - smallPairs * splitCount * pairRows;
        const uint32_t extraGroupCount =
            (remainingRows + kCycleGroupRows - 1U) / kCycleGroupRows;
        const bool balanceTailGroups = bigPairCount != 0U && extraGroupCount <= splitCount;
        const uint32_t pairCount = balanceTailGroups
                                       ? smallPairs
                                       : smallPairs + (splitIdx < bigPairCount ? 1U : 0U);
        const uint32_t extraRowBase = smallPairs * splitCount * pairRows +
                                      splitIdx * kCycleGroupRows;
        const uint32_t extraRows = balanceTailGroups && splitIdx < extraGroupCount
                                       ? (reduceSize_ - extraRowBase < kCycleGroupRows
                                              ? reduceSize_ - extraRowBase : kCycleGroupRows)
                                       : 0U;
        const uint32_t outputOffset = outputTask * innerDim;
        constexpr uint32_t laneRotation = 25U;
        uint32_t lane = splitIdx;
        uint32_t pairIndex = lane;
        if (pairCount > 0U) {
            CopyInCycleFp32Pair(outputTask, pairIndex);
        } else if (extraRows > 0U) {
            CopyInCycleFp32Rows(outputTask, extraRowBase, extraRows);
        }

        if (splitIdx == 0U) {
            LocalTensor<float> output = splitOutputQueue_.AllocTensor<float>();
            Duplicate(output, 0.0F, static_cast<int32_t>(cycleAlignedCols_));
            splitOutputQueue_.EnQue(output);
            output = splitOutputQueue_.DeQue<float>();
            CopyOutSplitFp32(output, outputOffset, innerDim, cycleAlignedCols_);
            event_t eventId = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventId);
            WaitFlag<HardEvent::MTE3_S>(eventId);
            pipe_.ReleaseEventID<HardEvent::MTE3_S>(eventId);
            splitOutputQueue_.FreeTensor(output);
        }
        SyncAll<true>();

        LocalTensor<float> compact = inputFloatBuf_.GetWithOffset<float>(
            cycleGroupElems_, kFp32CycleBankOffsetBytes);
        LocalTensor<float> padded = squareFloatBuf_.Get<float>();
        LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
        Duplicate(compact, 0.0F, static_cast<int32_t>(cycleGroupElems_));
        Duplicate(padded, 0.0F,
                  static_cast<int32_t>((kCycleGroupRows / 2U) * cycleAlignedCols_));
        ArithProgression<int32_t>(offsets, 0, sizeof(float), innerDim);
        PipeBarrier<PIPE_V>();

        for (uint32_t localPair = 0; localPair < pairCount; ++localPair) {
            LocalTensor<float> input = splitInputQueue_.DeQue<float>();
            uint32_t nextPairIndex = pairIndex + splitCount;
            if constexpr (RotateLanes) {
                if (localPair + 1U < smallPairs) {
                    lane += laneRotation;
                    if (lane >= splitCount) {
                        lane -= splitCount;
                    }
                    nextPairIndex = (localPair + 1U) * splitCount + lane;
                } else {
                    nextPairIndex = (localPair + 1U) * splitCount + splitIdx;
                }
            }
            if (localPair + 1U < pairCount) {
                CopyInCycleFp32Pair(outputTask, nextPairIndex);
            } else if (extraRows > 0U) {
                CopyInCycleFp32Rows(outputTask, extraRowBase, extraRows);
            }
            const uint32_t rowBase = pairIndex * pairRows;
            const uint32_t pairLength = reduceSize_ - rowBase < pairRows
                                            ? reduceSize_ - rowBase : pairRows;
            const uint32_t rows = pairLength < kCycleGroupRows
                                      ? pairLength : kCycleGroupRows;
            MulAddDst(compact, input, input, static_cast<int32_t>(rows * innerDim));
            if (pairLength > kCycleGroupRows) {
                const uint32_t secondRows = pairLength - kCycleGroupRows;
                MulAddDst(compact, input[cycleGroupElems_], input[cycleGroupElems_],
                          static_cast<int32_t>(secondRows * innerDim));
            }
            splitInputQueue_.FreeTensor(input);
            pairIndex = nextPairIndex;
        }
        if (extraRows > 0U) {
            LocalTensor<float> input = splitInputQueue_.DeQue<float>();
            MulAddDst(compact, input, input, static_cast<int32_t>(extraRows * innerDim));
            splitInputQueue_.FreeTensor(input);
        }
        PipeBarrier<PIPE_V>();

        constexpr uint32_t gatherRows = kCycleGroupRows / 2U;
        FoldCycleRowsUntil<kCycleGroupRows, gatherRows>(compact, innerDim);
        LocalTensor<uint32_t> gatherOffsets = offsets.ReinterpretCast<uint32_t>();
        GatherCycleRows<0U, gatherRows>(padded, compact, gatherOffsets,
                                        innerDim, cycleAlignedCols_);
        PipeBarrier<PIPE_V>();
        FoldCycleRows<gatherRows>(padded, cycleAlignedCols_);
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();
        CopyOutSplitFp32(padded, outputOffset, innerDim, cycleAlignedCols_);
        SetAtomicNone();
    }

    __aicore__ inline void CopyInBf16Compact(uint32_t inputOffset, uint32_t elemCount)
    {
        LocalTensor<T> input = splitInputQueue_.AllocTensor<T>();
        if (inputOffset % (kBlockBytes / sizeof(T)) == 0U &&
            elemCount % (kBlockBytes / sizeof(T)) == 0U) {
            DataCopy(input, inputGm_[inputOffset], elemCount);
        } else {
            DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elemCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(input, inputGm_[inputOffset], copyParams, padParams);
        }
        splitInputQueue_.EnQue(input);
    }

    __aicore__ inline uint32_t PrefetchBf16Output(uint32_t outputRow, uint32_t innerDim)
    {
        uint32_t rows = reduceSize_ < Bf16CycleRows ? reduceSize_ : Bf16CycleRows;
        CopyInBf16Compact(outputRow * reduceSize_ * innerDim, rows * innerDim);
        if (rows < reduceSize_) {
            const uint32_t nextRows = reduceSize_ - rows < Bf16CycleRows
                                          ? reduceSize_ - rows : Bf16CycleRows;
            CopyInBf16Compact((outputRow * reduceSize_ + rows) * innerDim,
                              nextRows * innerDim);
            rows += nextRows;
        }
        return rows;
    }

    __aicore__ inline void CopyOutSplitFp32(LocalTensor<float>& output, uint32_t outputOffset,
                                           uint32_t cols, uint32_t alignedCols)
    {
        if (cols == alignedCols) {
            DataCopy(outputGm_[outputOffset], output, alignedCols);
            return;
        }
        DataCopyExtParams outputParams = {
            1, static_cast<uint32_t>(cols * sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], output, outputParams);
    }

    __aicore__ inline void CopyInSplitFp32(uint32_t inputOffset, uint32_t rows,
                                          uint32_t cols, uint32_t alignedCols,
                                          uint32_t rowStride)
    {
        LocalTensor<float> input = splitInputQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams = {
            true, 0, static_cast<uint8_t>(alignedCols - cols), 0.0F};
        DataCopyPad(input, inputGm_[inputOffset], copyParams, padParams);
        splitInputQueue_.EnQue(input);
    }

    __aicore__ inline void LoadSquareTile(uint32_t inputOffset, uint32_t len, uint32_t alignedLen)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(len * sizeof(T)),
            0,
            0,
            0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        CastInputToFloat<T>(inputFloat, inputLocal, alignedLen);
        PipeBarrier<PIPE_V>();
        Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutputContiguous(uint32_t outputOffset, uint32_t tileLen, uint32_t alignedLen)
    {
        LocalTensor<T> outputLocal = outputBuf_.Get<T>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        CastFloatToOutput<T>(outputLocal, accFloat, alignedLen);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outputParams = {1, static_cast<uint32_t>(tileLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, outputParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadSquareRows(uint32_t inputOffset, uint32_t rows, uint32_t cols,
                                          uint32_t alignedCols, uint32_t rowStride)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(T)),
            0,
            0};
        DataCopyPadExtParams<T> padParams = {
            true,
            0,
            static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        const uint32_t tileElems = rows * alignedCols;
        CastInputToFloat<T>(inputFloat, inputLocal, tileElems);
        PipeBarrier<PIPE_V>();
        Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(tileElems));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadSquareRowsPlain(uint32_t inputOffset, uint32_t rows, uint32_t cols,
                                               uint32_t alignedCols, uint32_t rowStride)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams = {
            true, 0, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        const uint32_t tileElems = rows * alignedCols;
        SquareInputToFloat<T>(squareFloat, inputLocal, tileElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceRows(LocalTensor<float>& data, uint32_t rows, uint32_t rowWidth)
    {
        while (rows > 1U) {
            const uint32_t pairs = rows / 2U;
            Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
            if ((rows & 1U) != 0U) {
                Adds(data[pairs * rowWidth], data[(rows - 1U) * rowWidth],
                     0.0F, static_cast<int32_t>(rowWidth));
            }
            PipeBarrier<PIPE_V>();
            rows = pairs + (rows & 1U);
        }
    }

    __aicore__ inline void LoadSquarePacked(uint32_t inputOffset, uint32_t tileElems, uint32_t fullElems)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        if (tileElems < fullElems) {
            Duplicate(squareFloat, 0.0F, static_cast<int32_t>(fullElems));
            PipeBarrier<PIPE_V>();
        }
        DataCopyExtParams copyParams = {
            1,
            static_cast<uint32_t>(tileElems * sizeof(T)),
            0,
            0,
            0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        CastInputToFloat<T>(inputFloat, inputLocal, tileElems);
        PipeBarrier<PIPE_V>();
        Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(tileElems));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPackedOffsets(uint32_t cols, uint32_t tileRows)
    {
        LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
        for (uint32_t col = 0; col < cols; ++col) {
            ArithProgression<int32_t>(offsets[col * tileRows], static_cast<int32_t>(col * sizeof(float)),
                                      static_cast<int32_t>(cols * sizeof(float)), tileRows);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulatePackedTile(uint32_t cols, uint32_t tileRows, uint32_t alignedCols)
    {
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        LocalTensor<float> gathered = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        LocalTensor<float> partialFloat = inputBuf_.Get<float>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        const uint32_t chunks = tileRows / kGenericReduceCols;
        Gather(gathered, squareFloat, offsets, 0, cols * tileRows);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(partialFloat, gathered, static_cast<int32_t>(kGenericReduceCols),
                       static_cast<int32_t>(cols * chunks), 1, 1,
                       static_cast<int32_t>(kGenericReduceCols * sizeof(float) / kBlockBytes));
        PipeBarrier<PIPE_V>();
        WholeReduceSum(gathered, partialFloat, static_cast<int32_t>(chunks),
                       static_cast<int32_t>(cols), 1, 1,
                       static_cast<int32_t>(chunks * sizeof(float) / kBlockBytes));
        PipeBarrier<PIPE_V>();
        Add(accFloat, accFloat, gathered, static_cast<int32_t>(alignedCols));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessLastDimKeptPacked(uint32_t begin, uint32_t end, uint32_t innerDim,
                                                    uint32_t innerReduceRows, uint32_t outerReduceSize)
    {
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        const uint32_t alignedCols = AlignUp(innerDim, kElemAlign);
        uint32_t packedRows = AlignDown(keptDataElems_ / innerDim,
                                        kGenericPackedRowAlign);
        if (packedRows > kGenericPackedMaxRows) {
            packedRows = kGenericPackedMaxRows;
        }
        BuildPackedOffsets(innerDim, packedRows);
        for (uint32_t outputRow = begin; outputRow < end; ++outputRow) {
            const uint32_t outputOffset = outputRow * innerDim;
            const uint32_t baseOffset = OutputToInputBase(outputOffset);
            Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedCols));
            for (uint32_t outerIdx = 0; outerIdx < outerReduceSize; ++outerIdx) {
                const uint32_t outerOffset = ReducePrefixOffset(outerIdx);
                for (uint32_t packedBase = 0; packedBase < innerReduceRows; packedBase += packedRows) {
                    uint32_t currentPackedRows = innerReduceRows - packedBase;
                    if (currentPackedRows > packedRows) {
                        currentPackedRows = packedRows;
                    }
                    const uint32_t tileElems = currentPackedRows * innerDim;
                    LoadSquarePacked(baseOffset + outerOffset + packedBase * innerDim,
                                     tileElems, packedRows * innerDim);
                    AccumulatePackedTile(innerDim, packedRows, alignedCols);
                }
            }
            CopyOutputContiguous(outputOffset, innerDim, alignedCols);
        }
    }

    __aicore__ inline void ProcessLastDimKept(uint32_t begin, uint32_t end)
    {
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        const uint32_t innerReduceDim = reduceDims_[reduceRank_ - 1U];
        const uint32_t trailingKeptLength = inputStrides_[innerReduceDim];
        const uint32_t innerDim = trailingKeptLength >= kGenericFlattenMinCols
                                      ? trailingKeptLength
                                      : dims_[rank_ - 1U];
        const uint32_t innerReduceRows = dims_[innerReduceDim];
        const uint32_t innerReduceStride = reduceStrides_[reduceRank_ - 1U];
        const uint32_t outerReduceSize = reduceSize_ / innerReduceRows;
        if (usePacked_) {
            ProcessLastDimKeptPacked(begin, end, innerDim, innerReduceRows, outerReduceSize);
            return;
        }
        const uint32_t colTiles = (innerDim + genericTileCols_ - 1U) / genericTileCols_;
        for (uint32_t taskIdx = begin; taskIdx < end; ++taskIdx) {
            const uint32_t outputRow = taskIdx / colTiles;
            const uint32_t colBase = (taskIdx % colTiles) * genericTileCols_;
            const uint32_t outputRowOffset = outputRow * innerDim;
            uint32_t cols = innerDim - colBase;
            if (cols > genericTileCols_) {
                cols = genericTileCols_;
            }
            const uint32_t alignedCols = AlignUp(cols, kBlockBytes / sizeof(T));
            const uint32_t tileRows = keptDataElems_ / alignedCols;
            Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedCols));
            const uint32_t baseOffset = OutputToInputBase(outputRowOffset + colBase);
            for (uint32_t outerIdx = 0; outerIdx < outerReduceSize; ++outerIdx) {
                const uint32_t outerOffset = ReducePrefixOffset(outerIdx);
                for (uint32_t rowBase = 0; rowBase < innerReduceRows; rowBase += tileRows) {
                    uint32_t rows = innerReduceRows - rowBase;
                    if (rows > tileRows) {
                        rows = tileRows;
                    }
                    LoadSquareRowsPlain(baseOffset + outerOffset + rowBase * innerReduceStride,
                                        rows, cols, alignedCols, innerReduceStride);
                    ReduceRows(squareFloat, rows, alignedCols);
                    Add(accFloat, accFloat, squareFloat, static_cast<int32_t>(alignedCols));
                    PipeBarrier<PIPE_V>();
                }
            }
            CopyOutputContiguous(outputRowOffset + colBase, cols, alignedCols);
        }
    }

    __aicore__ inline void ProcessLastDimKeptLinear(uint32_t begin, uint32_t end)
    {
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        const uint32_t innerDim = dims_[rank_ - 1U];
        for (uint32_t tileBase = begin; tileBase < end;) {
            uint32_t tileLen = end - tileBase;
            if (tileLen > keptDataElems_) {
                tileLen = keptDataElems_;
            }
            const uint32_t innerRemain = innerDim - (tileBase % innerDim);
            if (tileLen > innerRemain) {
                tileLen = innerRemain;
            }
            const uint32_t alignedLen = AlignUp(tileLen, kElemAlign);
            Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedLen));
            const uint32_t baseOffset = OutputToInputBase(tileBase);
            for (uint32_t reduceIdx = 0; reduceIdx < reduceSize_; ++reduceIdx) {
                LoadSquareTile(baseOffset + ReduceOffset(reduceIdx), tileLen, alignedLen);
                Add(accFloat, accFloat, squareFloat, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }
            CopyOutputContiguous(tileBase, tileLen, alignedLen);
            tileBase += tileLen;
        }
    }

    __aicore__ inline void ProcessSinglePrefixReduced(uint32_t begin, uint32_t end,
                                                      uint32_t suffixReduceLength,
                                                      uint32_t innerKeptDim, uint32_t prefixDim)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> partialFloat = partialFloatBuf_.Get<float>();
        const uint32_t prefixSize = dims_[prefixDim];
        const uint32_t prefixStride = inputStrides_[prefixDim];
        const uint32_t outerPrefixSize = reduceSize_ / (prefixSize * suffixReduceLength);
        const uint32_t inputAlign = kBlockBytes / sizeof(T);
        constexpr uint32_t transposeRows = kBlockBytes / sizeof(float);
        const bool transposeTail = suffixReduceLength > kGenericReduceCols ||
                                   suffixReduceLength % transposeRows != 0U;
        const bool rank4Axis03 = rank_ == 4U && axisMask_ == 0x9U;
        constexpr uint32_t dataElems = kGenericReduceRows * kGenericReduceCols;
        LocalTensor<uint32_t> gatherOffsets;
        if (transposeTail) {
            LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
            LocalTensor<int32_t> temp = inputBuf_.Get<int32_t>();
            const uint32_t count = suffixReduceLength * transposeRows;
            ArithProgression<int32_t>(offsets, 0, 1, count);
            PipeBarrier<PIPE_V>();
            ShiftRight(temp, offsets, static_cast<int32_t>(3), static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets, static_cast<int32_t>(suffixReduceLength * sizeof(float)),
                 static_cast<int32_t>(count));
            Muls(temp, temp,
                 static_cast<int32_t>((1 - static_cast<int32_t>(transposeRows * suffixReduceLength)) *
                                      static_cast<int32_t>(sizeof(float))),
                 static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Add(offsets, offsets, temp, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            gatherOffsets = offsets.ReinterpretCast<uint32_t>();
        }
        for (uint32_t outBase = begin; outBase < end;) {
            uint32_t rows = end - outBase;
            const uint32_t maxRows = transposeTail && rank4Axis03 && dims_[0] <= 11U ? 2U * transposeRows
                                     : (transposeTail ? transposeRows : kGenericReduceRows);
            if (rows > maxRows) {
                rows = maxRows;
            }
            const uint32_t innerRemain = innerKeptDim - (outBase % innerKeptDim);
            if (rows > innerRemain) {
                rows = innerRemain;
            }
            const uint32_t alignedRows = AlignUp(rows, kElemAlign);
            const uint32_t blockElems = rows * suffixReduceLength;
            const uint32_t copiedBlockElems = AlignUp(blockElems, inputAlign);
            const uint32_t alignedBlockElems = transposeTail
                                                   ? AlignUp(maxRows * suffixReduceLength, inputAlign)
                                                   : copiedBlockElems;
            const uint32_t prefixTile = dataElems / alignedBlockElems;
            Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedRows));
            const uint32_t baseOffset = rank4Axis03
                                            ? outBase * suffixReduceLength
                                            : OutputToInputBase(outBase);
            for (uint32_t outerPrefix = 0; outerPrefix < outerPrefixSize; ++outerPrefix) {
                const uint32_t outerOffset = rank4Axis03
                                                 ? 0U
                                                 : ReduceBeforePrefixOffset(outerPrefix, prefixDim);
                for (uint32_t prefixBase = 0; prefixBase < prefixSize; prefixBase += prefixTile) {
                    uint32_t prefixRows = prefixSize - prefixBase;
                    if (prefixRows > prefixTile) {
                        prefixRows = prefixTile;
                    }
                    if (transposeTail) {
                        Duplicate(inputLocal, static_cast<T>(0), prefixRows * alignedBlockElems);
                        PipeBarrier<PIPE_ALL>();
                    }
                    DataCopyExtParams copyParams = {
                        static_cast<uint16_t>(prefixRows),
                        static_cast<uint32_t>(blockElems * sizeof(T)),
                        static_cast<uint32_t>((prefixStride - blockElems) * sizeof(T)),
                        static_cast<uint32_t>((alignedBlockElems - copiedBlockElems) * sizeof(T) / kBlockBytes), 0};
                    DataCopyPadExtParams<T> padParams = {
                        true, 0, static_cast<uint8_t>(copiedBlockElems - blockElems), static_cast<T>(0)};
                    DataCopyPad(inputLocal,
                                inputGm_[baseOffset + outerOffset + prefixBase * prefixStride],
                                copyParams, padParams);
                    PipeBarrier<PIPE_ALL>();
                    const uint32_t tileElems = prefixRows * alignedBlockElems;
                    CastInputToFloat<T>(inputFloat, inputLocal, tileElems);
                    PipeBarrier<PIPE_V>();
                    Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(tileElems));
                    PipeBarrier<PIPE_V>();
                    ReduceRows(squareFloat, prefixRows, alignedBlockElems);
                    if (transposeTail) {
                        Gather(inputFloat, squareFloat, gatherOffsets, 0,
                               suffixReduceLength * transposeRows);
                        PipeBarrier<PIPE_V>();
                        ReduceRows(inputFloat, suffixReduceLength, transposeRows);
                        Add(accFloat, accFloat, inputFloat, static_cast<int32_t>(transposeRows));
                        if (rank4Axis03 && rows > transposeRows) {
                            Gather(inputFloat, squareFloat, gatherOffsets,
                                   transposeRows * suffixReduceLength * sizeof(float),
                                   suffixReduceLength * transposeRows);
                            PipeBarrier<PIPE_V>();
                            ReduceRows(inputFloat, suffixReduceLength, transposeRows);
                            Add(accFloat[transposeRows], accFloat[transposeRows], inputFloat,
                                static_cast<int32_t>(transposeRows));
                        }
                    } else {
                        Duplicate(partialFloat, 0.0F, static_cast<int32_t>(alignedRows));
                        WholeReduceSum(partialFloat, squareFloat, static_cast<int32_t>(suffixReduceLength),
                                       static_cast<int32_t>(rows), 1, 1,
                                       static_cast<int32_t>(suffixReduceLength * sizeof(float) / kBlockBytes));
                        PipeBarrier<PIPE_V>();
                        Add(accFloat, accFloat, partialFloat, static_cast<int32_t>(alignedRows));
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            CopyOutputContiguous(outBase, rows, alignedRows);
            outBase += rows;
        }
    }

    __aicore__ inline void ProcessLongPrefixReduced(uint32_t begin, uint32_t end,
                                                    uint32_t suffixReduceLength,
                                                    uint32_t prefixDim)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> partialFloat = partialFloatBuf_.Get<float>();
        const uint32_t prefixSize = dims_[prefixDim];
        const uint32_t prefixStride = inputStrides_[prefixDim];
        const uint32_t outerPrefixSize = reduceSize_ / (prefixSize * suffixReduceLength);
        constexpr uint32_t dataElems = kGenericReduceRows * kGenericReduceCols;
        constexpr uint32_t floatAlign = kBlockBytes / sizeof(float);
        for (uint32_t outBase = begin; outBase < end; ++outBase) {
            Duplicate(accFloat, 0.0F, kElemAlign);
            const uint32_t baseOffset = OutputToInputBase(outBase);
            for (uint32_t outerPrefix = 0; outerPrefix < outerPrefixSize; ++outerPrefix) {
                const uint32_t outerOffset = ReduceBeforePrefixOffset(outerPrefix, prefixDim);
                for (uint32_t colBase = 0; colBase < suffixReduceLength;
                     colBase += kGenericLongSuffixCols) {
                    uint32_t cols = suffixReduceLength - colBase;
                    if (cols > kGenericLongSuffixCols) {
                        cols = kGenericLongSuffixCols;
                    }
                    const uint32_t alignedCols = AlignUp(cols, kGenericReduceCols);
                    const uint32_t segments = alignedCols / kGenericReduceCols;
                    const uint32_t prefixTile = dataElems / alignedCols;
                    for (uint32_t prefixBase = 0; prefixBase < prefixSize; prefixBase += prefixTile) {
                        uint32_t prefixRows = prefixSize - prefixBase;
                        if (prefixRows > prefixTile) {
                            prefixRows = prefixTile;
                        }
                        Duplicate(inputLocal, static_cast<T>(0), prefixRows * alignedCols);
                        PipeBarrier<PIPE_ALL>();
                        const uint32_t copyAlignedCols = AlignUp(cols, kBlockBytes / sizeof(T));
                        DataCopyExtParams copyParams = {
                            static_cast<uint16_t>(prefixRows), static_cast<uint32_t>(cols * sizeof(T)),
                            static_cast<uint32_t>((prefixStride - cols) * sizeof(T)),
                            static_cast<uint32_t>((alignedCols - copyAlignedCols) * sizeof(T) / kBlockBytes), 0};
                        DataCopyPadExtParams<T> padParams = {
                            true, 0, static_cast<uint8_t>(copyAlignedCols - cols), static_cast<T>(0)};
                        DataCopyPad(inputLocal,
                                    inputGm_[baseOffset + outerOffset +
                                             prefixBase * prefixStride + colBase],
                                    copyParams, padParams);
                        PipeBarrier<PIPE_ALL>();
                        const uint32_t tileElems = prefixRows * alignedCols;
                        CastInputToFloat<T>(inputFloat, inputLocal, tileElems);
                        PipeBarrier<PIPE_V>();
                        Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(tileElems));
                        PipeBarrier<PIPE_V>();
                        ReduceRows(squareFloat, prefixRows, alignedCols);
                        WholeReduceSum(partialFloat, squareFloat, kGenericReduceCols,
                                       static_cast<int32_t>(segments), 1, 1,
                                       kGenericReduceCols * sizeof(float) / kBlockBytes);
                        PipeBarrier<PIPE_V>();
                        WholeReduceSum(inputFloat, partialFloat, static_cast<int32_t>(segments),
                                       1, 1, 1, 1);
                        PipeBarrier<PIPE_V>();
                        Add(accFloat, accFloat, inputFloat, floatAlign);
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            CopyOutputContiguous(outBase, 1, kElemAlign);
        }
    }

    __aicore__ inline void ProcessLongPrefixReducedRows(uint32_t begin, uint32_t end,
                                                        uint32_t suffixReduceLength,
                                                        uint32_t innerKeptDim,
                                                        uint32_t prefixDim)
    {
        LocalTensor<T> inputLocal = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> partialFloat = partialFloatBuf_.Get<float>();
        const uint32_t prefixSize = dims_[prefixDim];
        const uint32_t prefixStride = inputStrides_[prefixDim];
        const uint32_t outerPrefixSize = reduceSize_ / (prefixSize * suffixReduceLength);
        const uint32_t alignedCols = AlignUp(suffixReduceLength, kGenericReduceCols);
        const uint32_t copyAlignedCols = AlignUp(suffixReduceLength, kBlockBytes / sizeof(T));
        const uint32_t segments = alignedCols / kGenericReduceCols;
        constexpr uint32_t dataElems = kGenericReduceRows * kGenericReduceCols;
        uint32_t maxRows = dataElems / alignedCols;
        if (maxRows > 2U) {
            maxRows = 2U;
        }
        DataCopyExtParams copyParams = {
            0, static_cast<uint32_t>(suffixReduceLength * sizeof(T)), 0,
            static_cast<uint32_t>((alignedCols - copyAlignedCols) * sizeof(T) / kBlockBytes), 0};
        DataCopyPadExtParams<T> padParams = {
            true, 0, static_cast<uint8_t>(copyAlignedCols - suffixReduceLength), static_cast<T>(0)};
        for (uint32_t outBase = begin; outBase < end;) {
            uint32_t rows = end - outBase;
            if (rows > maxRows) {
                rows = maxRows;
            }
            const uint32_t innerRemain = innerKeptDim - (outBase % innerKeptDim);
            if (rows > innerRemain) {
                rows = innerRemain;
            }
            const uint32_t alignedRows = AlignUp(rows, kElemAlign);
            Duplicate(accFloat, 0.0F, alignedRows);
            const uint32_t baseOffset = OutputToInputBase(outBase);
            copyParams.blockCount = static_cast<uint16_t>(rows);
            const uint32_t rowWidth = rows * alignedCols;
            const uint32_t prefixTile = dataElems / rowWidth;
            for (uint32_t outerPrefix = 0; outerPrefix < outerPrefixSize; ++outerPrefix) {
                const uint32_t outerOffset = ReduceBeforePrefixOffset(outerPrefix, prefixDim);
                for (uint32_t prefixBase = 0; prefixBase < prefixSize; prefixBase += prefixTile) {
                    uint32_t prefixRows = prefixSize - prefixBase;
                    if (prefixRows > prefixTile) {
                        prefixRows = prefixTile;
                    }
                    const uint32_t tileElems = prefixRows * rowWidth;
                    Duplicate(inputLocal, static_cast<T>(0), tileElems);
                    PipeBarrier<PIPE_ALL>();
                    for (uint32_t prefixRow = 0; prefixRow < prefixRows; ++prefixRow) {
                        DataCopyPad(inputLocal[prefixRow * rowWidth],
                                    inputGm_[baseOffset + outerOffset +
                                             (prefixBase + prefixRow) * prefixStride],
                                    copyParams, padParams);
                    }
                    PipeBarrier<PIPE_ALL>();
                    CastInputToFloat<T>(inputFloat, inputLocal, tileElems);
                    PipeBarrier<PIPE_V>();
                    Mul(squareFloat, inputFloat, inputFloat, static_cast<int32_t>(tileElems));
                    PipeBarrier<PIPE_V>();
                    ReduceRows(squareFloat, prefixRows, rowWidth);
                    WholeReduceSum(partialFloat, squareFloat, kGenericReduceCols,
                                   static_cast<int32_t>(rows * segments), 1, 1,
                                   kGenericReduceCols * sizeof(float) / kBlockBytes);
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(inputFloat, partialFloat, static_cast<int32_t>(segments),
                                   static_cast<int32_t>(rows), 1, 1,
                                   static_cast<int32_t>(segments * sizeof(float) / kBlockBytes));
                    PipeBarrier<PIPE_V>();
                    Add(accFloat, accFloat, inputFloat, alignedRows);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            CopyOutputContiguous(outBase, rows, alignedRows);
            outBase += rows;
        }
    }

    __aicore__ inline void ProcessLastDimReduced(uint32_t begin, uint32_t end)
    {
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> squareFloat = squareFloatBuf_.Get<float>();
        LocalTensor<float> partialFloat = partialFloatBuf_.Get<float>();
        uint32_t suffixStart = rank_;
        uint32_t suffixReduceLength = 1;
        while (suffixStart > 0U && (axisMask_ & (1U << (suffixStart - 1U))) != 0U) {
            --suffixStart;
            suffixReduceLength *= dims_[suffixStart];
        }
        const uint32_t innerKeptDim = suffixStart == 0U ? 1U : dims_[suffixStart - 1U];
        const uint32_t prefixReduceSize = reduceSize_ / suffixReduceLength;
        uint32_t prefixRank = 0;
        uint32_t prefixDim = 0;
        for (uint32_t idx = 0; idx < reduceRank_; ++idx) {
            if (reduceDims_[idx] < suffixStart) {
                prefixDim = reduceDims_[idx];
                ++prefixRank;
            }
        }
        if (prefixRank > 0U) {
            if (suffixReduceLength <= kGenericTransposeMaxCols) {
                ProcessSinglePrefixReduced(begin, end, suffixReduceLength, innerKeptDim, prefixDim);
            } else if (suffixReduceLength <= kGenericLongSuffixCols &&
                       AlignUp(suffixReduceLength, kGenericReduceCols) %
                               (8U * kGenericReduceCols) == 0U) {
                ProcessLongPrefixReducedRows(begin, end, suffixReduceLength,
                                             innerKeptDim, prefixDim);
            } else {
                ProcessLongPrefixReduced(begin, end, suffixReduceLength, prefixDim);
            }
            return;
        }
        for (uint32_t outBase = begin; outBase < end;) {
            uint32_t rows = end - outBase;
            if (rows > kGenericReduceRows) {
                rows = kGenericReduceRows;
            }
            const uint32_t innerRemain = innerKeptDim - (outBase % innerKeptDim);
            if (rows > innerRemain) {
                rows = innerRemain;
            }
            const uint32_t alignedRows = AlignUp(rows, kElemAlign);
            Duplicate(accFloat, 0.0F, static_cast<int32_t>(alignedRows));
            const uint32_t baseOffset = OutputToInputBase(outBase);
            for (uint32_t prefixIdx = 0; prefixIdx < prefixReduceSize; ++prefixIdx) {
                const uint32_t prefixOffset = ReduceBeforeSuffixOffset(prefixIdx, suffixStart);
                for (uint32_t inner = 0; inner < suffixReduceLength;) {
                    uint32_t cols = suffixReduceLength - inner;
                    if (cols > kGenericReduceCols) {
                        cols = kGenericReduceCols;
                    }
                    const uint32_t alignedCols = AlignUp(cols, kBlockBytes / sizeof(T));
                    LoadSquareRows(baseOffset + prefixOffset + inner, rows, cols,
                                   alignedCols, suffixReduceLength);
                    Duplicate(partialFloat, 0.0F, static_cast<int32_t>(alignedRows));
                    WholeReduceSum(partialFloat, squareFloat, static_cast<int32_t>(cols),
                                   static_cast<int32_t>(rows), 1, 1,
                                   static_cast<int32_t>(alignedCols * sizeof(float) / kBlockBytes));
                    PipeBarrier<PIPE_V>();
                    Add(accFloat, accFloat, partialFloat, static_cast<int32_t>(alignedRows));
                    PipeBarrier<PIPE_V>();
                    inner += cols;
                }
            }
            CopyOutputContiguous(outBase, rows, alignedRows);
            outBase += rows;
        }
    }

    __aicore__ inline uint32_t OutputToInputBase(uint32_t linear) const
    {
        uint32_t offset = 0;
        for (int32_t idx = static_cast<int32_t>(outputRank_) - 1; idx >= 0; --idx) {
            const uint32_t dimIdx = outputDims_[idx];
            const uint32_t coord = linear % dims_[dimIdx];
            linear /= dims_[dimIdx];
            offset += coord * inputStrides_[dimIdx];
        }
        return offset;
    }

    __aicore__ inline uint32_t ReducePrefixOffset(uint32_t linear) const
    {
        uint32_t offset = 0;
        for (int32_t idx = static_cast<int32_t>(reduceRank_) - 2; idx >= 0; --idx) {
            const uint32_t dimIdx = reduceDims_[idx];
            const uint32_t coord = linear % dims_[dimIdx];
            linear /= dims_[dimIdx];
            offset += coord * reduceStrides_[idx];
        }
        return offset;
    }

    __aicore__ inline uint32_t ReduceOffset(uint32_t linear) const
    {
        uint32_t offset = 0;
        for (int32_t idx = static_cast<int32_t>(reduceRank_) - 1; idx >= 0; --idx) {
            const uint32_t dimIdx = reduceDims_[idx];
            const uint32_t coord = linear % dims_[dimIdx];
            linear /= dims_[dimIdx];
            offset += coord * reduceStrides_[idx];
        }
        return offset;
    }

    __aicore__ inline uint32_t ReduceBeforeSuffixOffset(uint32_t linear, uint32_t suffixStart) const
    {
        uint32_t offset = 0;
        for (int32_t idx = static_cast<int32_t>(reduceRank_) - 1; idx >= 0; --idx) {
            const uint32_t dimIdx = reduceDims_[idx];
            if (dimIdx >= suffixStart) {
                continue;
            }
            const uint32_t coord = linear % dims_[dimIdx];
            linear /= dims_[dimIdx];
            offset += coord * reduceStrides_[idx];
        }
        return offset;
    }

    __aicore__ inline uint32_t ReduceBeforePrefixOffset(uint32_t linear, uint32_t prefixDim) const
    {
        uint32_t offset = 0;
        for (int32_t idx = static_cast<int32_t>(reduceRank_) - 1; idx >= 0; --idx) {
            const uint32_t dimIdx = reduceDims_[idx];
            if (dimIdx >= prefixDim) {
                continue;
            }
            const uint32_t coord = linear % dims_[dimIdx];
            linear /= dims_[dimIdx];
            offset += coord * reduceStrides_[idx];
        }
        return offset;
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 3> splitInputQueue_;
    TQue<QuePosition::VECOUT, 2> splitOutputQueue_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> inputFloatBuf_;
    TBuf<QuePosition::VECCALC> squareFloatBuf_;
    TBuf<QuePosition::VECCALC> accFloatBuf_;
    TBuf<QuePosition::VECCALC> partialFloatBuf_;
    TBuf<QuePosition::VECCALC> outputBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<float> workspaceGm_;
    GlobalTensor<float> workspacePartialGm_;
    uint32_t dims_[kMaxRank] = {};
    uint32_t inputStrides_[kMaxRank] = {};
    uint32_t outputDims_[kMaxRank] = {};
    uint32_t reduceDims_[kMaxRank] = {};
    uint32_t reduceStrides_[kMaxRank] = {};
    uint32_t inputSize_ = 0;
    uint32_t reduceSize_ = 1;
    uint32_t rank_ = 0;
    uint32_t outputRank_ = 0;
    uint32_t reduceRank_ = 0;
    uint32_t axisMask_ = 0;
    uint32_t smallCoreItems_ = 1;
    uint32_t bigCoreCount_ = 0;
    uint32_t genericTileCols_ = kGenericBatchCols;
    uint32_t keptDataElems_ = kGenericKeptElems;
    uint32_t cycleAlignedCols_ = 1;
    uint32_t cycleGroupElems_ = 1;
    uint32_t bf16AlignedInner_ = 1;
    uint32_t bf16GroupElems_ = 1;
    bool usePacked_ = false;
    bool useCycleFp32_ = false;
};

template <typename T, bool SplitFp32Pipeline = false,
          uint32_t Bf16CycleRows = kBf16CycleRows>
__aicore__ inline void RunSquareSumV1Generic(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                             const SquareSumV1TilingData& tilingData, bool splitKept = false)
{
    KernelSquareSumV1Generic<T, SplitFp32Pipeline, Bf16CycleRows> kernel;
    kernel.Init(input, output, workspace, tilingData);
    if constexpr (SplitFp32Pipeline) {
        kernel.ProcessSplitKeptFp32Pipeline();
    } else if (splitKept) {
        kernel.ProcessSplitKept();
    } else {
        kernel.Process();
    }
}

template <uint32_t BatchRows, uint32_t SuffixCols = 69U>
class KernelSquareSumV1Rank4Axis03Pipeline {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SquareSumV1TilingData& tiling)
    {
        prefixRows_ = tiling.dims[0];
        suffixCols_ = tiling.dims[3];
        outputSize_ = tiling.outputSize;
        smallCoreItems_ = tiling.smallCoreItems;
        bigCoreCount_ = tiling.bigCoreCount;
        alignedBlockElems_ = AlignUp(kBatchRows * suffixCols_, kBlockBytes / sizeof(half));
        tileElems_ = prefixRows_ * alignedBlockElems_;
        useHalfChunk69_ = BatchRows >= 48U && prefixRows_ >= 2U &&
                          prefixRows_ <= 2U * kPhasePairCount &&
                          prefixRows_ % 2U == 0U && suffixCols_ == SuffixCols;
        prefixPairCount_ = prefixRows_ / 2U;
        chunkElems_ = kPrefixChunkRows * (alignedBlockElems_ + kAlignedSourceExtraElems);
        inputGm_.SetGlobalBuffer((__gm__ half*)input, tiling.inputSize);
        if (useHalfChunk69_) {
            inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        outputGm_.SetGlobalBuffer((__gm__ half*)output, outputSize_);
        if constexpr (BatchRows >= 32U) {
            if (useHalfChunk69_) {
                pipe_.InitBuffer(inputQueue_, 2, chunkElems_ * sizeof(half));
                pipe_.InitBuffer(inputBuf_, alignedBlockElems_ * sizeof(half));
            } else {
                pipe_.InitBuffer(inputBuf_, tileElems_ * sizeof(half));
            }
        } else {
            pipe_.InitBuffer(inputQueue_, 2, tileElems_ * sizeof(half));
        }
        if (useHalfChunk69_) {
            pipe_.InitBuffer(squareBuf_, kBatchRows * sizeof(half));
            pipe_.InitBuffer(offsetBuf_, kBlockBytes);
        } else {
            pipe_.InitBuffer(squareBuf_, alignedBlockElems_ * sizeof(float));
            pipe_.InitBuffer(gatherBuf_, kGatherRows * suffixCols_ * sizeof(float));
            pipe_.InitBuffer(offsetBuf_, kGatherRows * suffixCols_ * sizeof(uint32_t));
        }
        pipe_.InitBuffer(outputQueue_, 2, kBatchRows * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const uint32_t gatherElems = kGatherRows * suffixCols_;
        const uint32_t gatherBytes = useHalfChunk69_ ? sizeof(half) : sizeof(float);
        LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
        if (!useHalfChunk69_) {
            LocalTensor<int32_t> temp = gatherBuf_.Get<int32_t>();
            ArithProgression<int32_t>(offsets, 0, 1, gatherElems);
            PipeBarrier<PIPE_V>();
            ShiftRight(temp, offsets, 4, gatherElems);
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets,
                 static_cast<int32_t>(suffixCols_ * gatherBytes),
                 gatherElems);
            Muls(temp, temp,
                 static_cast<int32_t>((1 - static_cast<int32_t>(
                     kGatherRows * suffixCols_)) * gatherBytes),
                 gatherElems);
            PipeBarrier<PIPE_V>();
            Add(offsets, offsets, temp, gatherElems);
            PipeBarrier<PIPE_V>();
        }

        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        if constexpr (BatchRows >= 32U) {
            if (useHalfChunk69_) {
                ProcessHalfChunk69(length);
                return;
            }
            LocalTensor<half> input = inputBuf_.template Get<half>();
            for (uint32_t task = begin; task < begin + length; ++task) {
                LoadInput(task, input);
                PipeBarrier<PIPE_ALL>();
                Compute(task, input, offsets.ReinterpretCast<uint32_t>());
            }
            return;
        }
        CopyIn(begin);
        for (uint32_t task = begin; task < begin + length; ++task) {
            if (task + 1U < begin + length) {
                CopyIn(task + 1U);
            }
            LocalTensor<half> input = inputQueue_.template DeQue<half>();
            Compute(task, input, offsets.ReinterpretCast<uint32_t>());
            inputQueue_.FreeTensor(input);
        }
    }

private:
    template <bool ClearPadding = true>
    __aicore__ inline void CopyInHalfChunk69Range(uint32_t outputBegin, uint32_t rows,
                                                  uint32_t prefixBegin,
                                                  uint32_t prefixStride = 1U)
    {
        LocalTensor<half> input = inputQueue_.template AllocTensor<half>();
        const uint32_t blockElems = rows * suffixCols_;
        const uint32_t copiedBlockElems = AlignUp(blockElems, kBlockBytes / sizeof(half));
        const uint32_t inputOffset = (prefixBegin * outputSize_ + outputBegin) * suffixCols_;
        if constexpr (ClearPadding) {
            if (rows < kBatchRows) {
                Duplicate(input, static_cast<half>(0), chunkElems_);
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (prefixStride == 16U) {
            constexpr uint32_t blockElems = kBlockBytes / sizeof(half);
            const uint32_t leadElems = inputOffset % blockElems;
            const uint32_t logicalOffset = leadElems == 0U ? 0U : blockElems;
            const uint32_t sourceCopyElems = rows * suffixCols_ + leadElems;
            DataCopyExtParams alignedParams = {
                2U, static_cast<uint32_t>(sourceCopyElems * sizeof(half)),
                static_cast<uint32_t>((prefixStride * outputSize_ * suffixCols_ - sourceCopyElems) *
                                      sizeof(half)),
                static_cast<uint32_t>((alignedBlockElems_ + kAlignedSourceExtraElems -
                                       logicalOffset - copiedBlockElems) *
                                      sizeof(half) / kBlockBytes), 0};
            DataCopyPadExtParams<half> alignedPad = {
                leadElems != 0U, static_cast<uint8_t>(logicalOffset - leadElems), 0,
                static_cast<half>(0)};
            DataCopyPad(input, inputGm_[inputOffset - leadElems], alignedParams, alignedPad);
            inputQueue_.EnQue(input);
            return;
        }
        const uint32_t pairGapElems = prefixStride == 1U ? 0U : kAlignedSourceExtraElems;
        DataCopyExtParams copyParams = {
            2U, static_cast<uint32_t>(blockElems * sizeof(half)),
            static_cast<uint32_t>((prefixStride * outputSize_ * suffixCols_ - blockElems) *
                                  sizeof(half)),
            static_cast<uint32_t>((alignedBlockElems_ + pairGapElems - copiedBlockElems) *
                                  sizeof(half) / kBlockBytes), 0};
        DataCopyPadExtParams<half> padParams = {
            true, 0, static_cast<uint8_t>(copiedBlockElems - blockElems), static_cast<half>(0)};
        DataCopyPad(input, inputGm_[inputOffset], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    template <bool PairByPhase, bool BalanceTail = false>
    __aicore__ inline void ProcessHalfChunk69Pairs(uint32_t length)
    {
        LocalTensor<half> acc = inputBuf_.template Get<half>();
        LocalTensor<half> gathered = squareBuf_.template Get<half>();
        const uint32_t taskStride = GetBlockNum();
        const uint32_t physicalIdx = GetBlockIdx();

        const uint32_t blockIdx = (physicalIdx % 2U) * ((taskStride + 1U) / 2U) + physicalIdx / 2U;
        uint32_t fullWaveCount = 0U;
        uint32_t tailOutputBegin = 0U;
        uint32_t tailRows = 0U;
        if constexpr (BalanceTail) {
            fullWaveCount = outputSize_ / (taskStride * kBatchRows);
            const uint32_t fullWaveRows = fullWaveCount * taskStride * kBatchRows;
            const uint32_t remainingRows = outputSize_ - fullWaveRows;
            const uint32_t tailGroups = remainingRows / kGatherRows;
            const uint32_t smallGroups = tailGroups / taskStride;
            const uint32_t bigGroupCount = tailGroups % taskStride;
            const uint32_t blockGroups = smallGroups + (blockIdx < bigGroupCount ? 1U : 0U);
            tailOutputBegin = fullWaveRows +
                              (blockIdx * smallGroups +
                               (blockIdx < bigGroupCount ? blockIdx : bigGroupCount)) * kGatherRows;
            tailRows = blockGroups * kGatherRows;
            if (blockIdx + 1U == taskStride) {
                tailRows += remainingRows % kGatherRows;
            }
        }
        const uint32_t localLength = BalanceTail ? fullWaveCount + (tailRows != 0U ? 1U : 0U)
                                                 : length;
        if (localLength == 0U) {
            return;
        }
        uint32_t outputBegin = BalanceTail && fullWaveCount == 0U
                                   ? tailOutputBegin : blockIdx * kBatchRows;
        uint32_t rows = BalanceTail && fullWaveCount == 0U ? tailRows : kBatchRows;
        CopyInHalfChunk69Range<!BalanceTail>(
            outputBegin, rows, 0U, PairByPhase ? 16U : 1U);
        for (uint32_t i = 0U; i < localLength; ++i) {
            const uint32_t activeElems = BalanceTail && rows < kBatchRows
                ? AlignUp(rows * suffixCols_, kBlockBytes / sizeof(half))
                : alignedBlockElems_;
            for (uint32_t chunk = 0U; chunk < prefixPairCount_; ++chunk) {
                if (chunk + 1U < prefixPairCount_) {
                    const uint32_t nextChunk = chunk + 1U;
                    const uint32_t nextPrefix = !PairByPhase ? 2U * nextChunk
                                                : nextChunk < 9U ? nextChunk
                                                                 : 2U * nextChunk - 8U;
                    const uint32_t nextStride = !PairByPhase ? 1U
                                                : nextChunk < 6U ? 16U
                                                : nextChunk == 7U ? 7U : 3U;
                    CopyInHalfChunk69Range<!BalanceTail>(
                        outputBegin, rows, nextPrefix, nextStride);
                } else if (i + 1U < localLength) {
                    uint32_t nextOutputBegin = outputBegin + taskStride * kBatchRows;
                    uint32_t nextRows = kBatchRows;
                    if constexpr (BalanceTail) {
                        if (i + 1U == fullWaveCount) {
                            nextOutputBegin = tailOutputBegin;
                            nextRows = tailRows;
                        }
                    }
                    CopyInHalfChunk69Range<!BalanceTail>(
                        nextOutputBegin, nextRows, 0U, PairByPhase ? 16U : 1U);
                }
                LocalTensor<half> input = inputQueue_.template DeQue<half>();
                LocalTensor<half> reduced = input;
                if constexpr (PairByPhase) {
                    const uint32_t leadElems = chunk < 6U
                        ? (chunk * outputSize_ * suffixCols_) % (kBlockBytes / sizeof(half)) : 0U;
                    const uint32_t logicalOffset =
                        leadElems == 0U ? 0U : kBlockBytes / sizeof(half);
                    const uint32_t secondOffset =
                        alignedBlockElems_ + kAlignedSourceExtraElems + logicalOffset;
                    Mul(input[logicalOffset], input[logicalOffset], input[logicalOffset],
                        static_cast<int32_t>(activeElems));
                    Mul(input[secondOffset], input[secondOffset], input[secondOffset],
                        static_cast<int32_t>(activeElems));
                    PipeBarrier<PIPE_V>();
                    Add(input[logicalOffset], input[logicalOffset], input[secondOffset],
                        static_cast<int32_t>(activeElems));
                    PipeBarrier<PIPE_V>();
                    reduced = input[logicalOffset];
                } else {
                    ReduceHalfChunk69(input);
                }
                if (chunk == 0U) {
                    Adds(acc, reduced, static_cast<half>(0),
                         static_cast<int32_t>(activeElems));
                } else {
                    Add(acc, acc, reduced, static_cast<int32_t>(activeElems));
                }
                inputQueue_.FreeTensor(input);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (BalanceTail) {
                if (rows == kBatchRows) {
                    ReduceSuffix69Whole<0U, kBatchRows / kGatherRows>(gathered, acc);
                } else {
                    ReduceSuffix69Tail<0U>(gathered, acc, rows);
                }
            } else {
                ReduceSuffix69Whole<0U, kBatchRows / kGatherRows>(gathered, acc);
            }
            CopyOutHalfRange(outputBegin, rows, gathered);
            if constexpr (BalanceTail) {
                if (i + 1U == fullWaveCount) {
                    outputBegin = tailOutputBegin;
                    rows = tailRows;
                } else {
                    outputBegin += taskStride * kBatchRows;
                }
            } else {
                outputBegin += taskStride * kBatchRows;
            }
        }
    }

    __aicore__ inline void ReduceHalfChunk69(LocalTensor<half>& input)
    {
        Mul(input, input, input, static_cast<int32_t>(2U * alignedBlockElems_));
        PipeBarrier<PIPE_V>();
        Add(input, input, input[alignedBlockElems_],
            static_cast<int32_t>(alignedBlockElems_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessHalfChunk69(uint32_t length)
    {
        if (length == 0U) {
            return;
        }
        if (prefixPairCount_ != kPhasePairCount ||
            (outputSize_ * suffixCols_ * sizeof(half)) % kBlockBytes == 0U) {
            ProcessHalfChunk69Pairs<false, true>(length);
        } else {
            ProcessHalfChunk69Pairs<true, true>(length);
        }
    }

    template <uint32_t Bits>
    __aicore__ static constexpr uint64_t LowMask()
    {
        if constexpr (Bits >= 64U) {
            return ~0ULL;
        } else {
            return (1ULL << Bits) - 1ULL;
        }
    }

    template <uint32_t Row, uint32_t Repeats>
    __aicore__ inline void ReduceSuffix69Row(LocalTensor<half>& output,
                                             LocalTensor<half>& input)
    {
        constexpr uint32_t rowOffset = Row * SuffixCols;
        constexpr uint32_t alignedOffset = rowOffset / 16U * 16U;
        constexpr uint32_t bitOffset = rowOffset - alignedOffset;
        constexpr uint32_t lowBits = 64U - bitOffset < SuffixCols ? 64U - bitOffset : SuffixCols;
        constexpr uint32_t highBits = SuffixCols - lowBits;
        uint64_t mask[2] = {LowMask<lowBits>() << bitOffset, LowMask<highBits>()};
        WholeReduceSum(output[Row], input[alignedOffset], mask,
                       static_cast<int32_t>(Repeats), 16, 1, SuffixCols);
    }

    template <uint32_t Row, uint32_t Repeats>
    __aicore__ inline void ReduceSuffix69Whole(LocalTensor<half>& output,
                                                LocalTensor<half>& input)
    {
        ReduceSuffix69Row<Row, Repeats>(output, input);
        if constexpr (Row + 1U < kGatherRows) {
            ReduceSuffix69Whole<Row + 1U, Repeats>(output, input);
        }
    }

    template <uint32_t Row>
    __aicore__ inline void ReduceSuffix69Tail(LocalTensor<half>& output,
                                              LocalTensor<half>& input,
                                              uint32_t rows)
    {
        const uint32_t repeats = rows > Row ? (rows - Row + kGatherRows - 1U) / kGatherRows : 0U;
        if (repeats != 0U) {
            constexpr uint32_t rowOffset = Row * SuffixCols;
            constexpr uint32_t alignedOffset = rowOffset / 16U * 16U;
            constexpr uint32_t bitOffset = rowOffset - alignedOffset;
            constexpr uint32_t lowBits = 64U - bitOffset < SuffixCols ? 64U - bitOffset : SuffixCols;
            constexpr uint32_t highBits = SuffixCols - lowBits;
            uint64_t mask[2] = {LowMask<lowBits>() << bitOffset, LowMask<highBits>()};
            WholeReduceSum(output[Row], input[alignedOffset], mask,
                           static_cast<int32_t>(repeats), 16, 1, SuffixCols);
        }
        if constexpr (Row + 1U < kGatherRows) {
            ReduceSuffix69Tail<Row + 1U>(output, input, rows);
        }
    }

    __aicore__ inline void CopyIn(uint32_t task)
    {
        LocalTensor<half> input = inputQueue_.template AllocTensor<half>();
        LoadInput(task, input);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void LoadInput(uint32_t task, LocalTensor<half>& input)
    {
        const uint32_t outputBegin = task * kBatchRows;
        const uint32_t rows = outputSize_ - outputBegin < kBatchRows
                                  ? outputSize_ - outputBegin : kBatchRows;
        const uint32_t blockElems = rows * suffixCols_;
        const uint32_t copiedBlockElems = AlignUp(blockElems, kBlockBytes / sizeof(half));
        if (rows < kBatchRows) {
            Duplicate(input, static_cast<half>(0), tileElems_);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(prefixRows_),
            static_cast<uint32_t>(blockElems * sizeof(half)),
            static_cast<uint32_t>((outputSize_ * suffixCols_ - blockElems) * sizeof(half)),
            static_cast<uint32_t>((alignedBlockElems_ - copiedBlockElems) * sizeof(half) / kBlockBytes), 0};
        DataCopyPadExtParams<half> padParams = {
            true, 0, static_cast<uint8_t>(copiedBlockElems - blockElems), static_cast<half>(0)};
        DataCopyPad(input, inputGm_[outputBegin * suffixCols_], copyParams, padParams);
    }

    __aicore__ inline void Compute(uint32_t task, LocalTensor<half>& input,
                                   LocalTensor<uint32_t> gatherOffsets)
    {
        LocalTensor<float> square = squareBuf_.Get<float>();
        LocalTensor<float> gathered = gatherBuf_.Get<float>();
        Mul(input, input, input, static_cast<int32_t>(tileElems_));
        PipeBarrier<PIPE_V>();
        ReduceRows(input, prefixRows_, alignedBlockElems_);
        CastInputToFloat<half>(square, input, alignedBlockElems_);
        PipeBarrier<PIPE_V>();
        Gather(gathered, square, gatherOffsets, 0, kGatherRows * suffixCols_);
        PipeBarrier<PIPE_V>();
        ReduceRows(gathered, suffixCols_, kGatherRows);
        CopyOut(task, gathered);
    }

    __aicore__ inline void CopyOut(uint32_t task, LocalTensor<float>& result)
    {
        const uint32_t outputBegin = task * kBatchRows;
        const uint32_t rows = outputSize_ - outputBegin < kBatchRows
                                  ? outputSize_ - outputBegin : kBatchRows;
        LocalTensor<half> output = outputQueue_.template AllocTensor<half>();
        CastFloatToOutput<half>(output, result, kBatchRows);
        outputQueue_.EnQue(output);
        output = outputQueue_.template DeQue<half>();
        DataCopyExtParams outputParams = {
            1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0};
        DataCopyPad(outputGm_[outputBegin], output, outputParams);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void CopyOutHalf(uint32_t task, LocalTensor<half>& result)
    {
        const uint32_t outputBegin = task * kBatchRows;
        const uint32_t rows = outputSize_ - outputBegin < kBatchRows
                                  ? outputSize_ - outputBegin : kBatchRows;
        CopyOutHalfRange(outputBegin, rows, result);
    }

    __aicore__ inline void CopyOutHalfRange(uint32_t outputBegin, uint32_t rows,
                                            LocalTensor<half>& result)
    {
        LocalTensor<half> output = outputQueue_.template AllocTensor<half>();
        Adds(output, result, static_cast<half>(0), kBatchRows);
        outputQueue_.EnQue(output);
        output = outputQueue_.template DeQue<half>();
        DataCopyExtParams outputParams = {
            1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0};
        DataCopyPad(outputGm_[outputBegin], output, outputParams);
        outputQueue_.FreeTensor(output);
    }

    template <uint32_t Rows, typename U>
    __aicore__ inline void ReduceRowsFixed(LocalTensor<U>& data, uint32_t rowWidth)
    {
        constexpr uint32_t pairs = Rows / 2U;
        constexpr uint32_t nextRows = pairs + (Rows & 1U);
        Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
        if constexpr ((Rows & 1U) != 0U) {
            Adds(data[pairs * rowWidth], data[(Rows - 1U) * rowWidth],
                 static_cast<U>(0), static_cast<int32_t>(rowWidth));
        }
        PipeBarrier<PIPE_V>();
        if constexpr (nextRows > 1U) {
            ReduceRowsFixed<nextRows>(data, rowWidth);
        }
    }

    template <typename T>
    __aicore__ inline void ReduceRows(LocalTensor<T>& data, uint32_t rows, uint32_t rowWidth)
    {
        while (rows > 1U) {
            const uint32_t pairs = rows / 2U;
            Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
            if ((rows & 1U) != 0U) {
                Adds(data[pairs * rowWidth], data[(rows - 1U) * rowWidth],
                     static_cast<T>(0), static_cast<int32_t>(rowWidth));
            }
            PipeBarrier<PIPE_V>();
            rows = pairs + (rows & 1U);
        }
    }

    static constexpr uint32_t kBatchRows = BatchRows;
    static constexpr uint32_t kGatherRows = 16;
    static constexpr uint32_t kPhasePairCount = 11;
    static constexpr uint16_t kPrefixChunkRows = 2U;
    static constexpr uint32_t kAlignedSourceExtraElems = 32U;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> squareBuf_;
    TBuf<QuePosition::VECCALC> gatherBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint32_t prefixRows_ = 0;
    uint32_t suffixCols_ = 0;
    uint32_t outputSize_ = 0;
    uint32_t smallCoreItems_ = 0;
    uint32_t bigCoreCount_ = 0;
    uint32_t alignedBlockElems_ = 0;
    uint32_t tileElems_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t prefixPairCount_ = 0;
    bool useHalfChunk69_ = false;
};

class KernelSquareSumV1Rank4Axis03Batch32 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SquareSumV1TilingData& tiling)
    {
        prefixRows_ = tiling.dims[0];
        suffixCols_ = tiling.dims[3];
        outputSize_ = tiling.outputSize;
        smallCoreItems_ = tiling.smallCoreItems;
        bigCoreCount_ = tiling.bigCoreCount;
        const uint32_t blockElems = kBatchRows * suffixCols_;
        const uint32_t paddedBlockElems = kBatchRows * AlignUp(suffixCols_, 16U);
        const uint32_t tileElems = kMaxPrefixRows * blockElems;
        inputGm_.SetGlobalBuffer((__gm__ half*)input, tiling.inputSize);
        outputGm_.SetGlobalBuffer((__gm__ half*)output, outputSize_);
        pipe_.InitBuffer(inputBuf_, tileElems * sizeof(half));
        pipe_.InitBuffer(squareBuf_, kPrefixChunkRows * blockElems * sizeof(float));
        pipe_.InitBuffer(accBuf_, paddedBlockElems * sizeof(float));
        pipe_.InitBuffer(gatherBuf_, paddedBlockElems * sizeof(float));
        pipe_.InitBuffer(outputBuf_, kBatchRows * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> input = inputBuf_.Get<half>();
        LocalTensor<float> square = squareBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> gathered = gatherBuf_.Get<float>();
        LocalTensor<half> output = outputBuf_.Get<half>();
        const uint32_t blockElems = kBatchRows * suffixCols_;

        TransDataTo5HDParams transposeParams;
        transposeParams.dstHighHalf = false;
        transposeParams.srcHighHalf = false;
        transposeParams.repeatTimes = static_cast<uint8_t>((suffixCols_ + 7U) / 8U);
        transposeParams.srcRepStride = 1U;
        transposeParams.dstRepStride = kBatchRows;
        uint64_t srcList0[16];
        uint64_t srcList1[16];
        uint64_t dstList0[16];
        uint64_t dstList1[16];
        const uint64_t accBase = reinterpret_cast<uint64_t>(acc.GetPhyAddr());
        const uint64_t gatheredBase = reinterpret_cast<uint64_t>(gathered.GetPhyAddr());
        for (uint32_t lane = 0; lane < 16U; ++lane) {
            srcList0[lane] = accBase + lane * suffixCols_ * sizeof(float);
            srcList1[lane] = accBase + (lane + 16U) * suffixCols_ * sizeof(float);
            const uint32_t dstOffset = (lane / 2U) * kBatchRows + (lane % 2U) * 8U;
            dstList0[lane] = gatheredBase + dstOffset * sizeof(float);
            dstList1[lane] = gatheredBase + (dstOffset + 16U) * sizeof(float);
        }

        uint32_t begin = 0;
        uint32_t length = 0;
        GetCoreRange(smallCoreItems_, bigCoreCount_, begin, length);
        for (uint32_t task = begin; task < begin + length; ++task) {
            const uint32_t outputBegin = task * kBatchRows;
            const uint32_t rows = outputSize_ - outputBegin < kBatchRows
                                      ? outputSize_ - outputBegin : kBatchRows;
            const uint32_t copiedElems = rows * suffixCols_;
            const uint32_t copyAlignedElems = AlignUp(copiedElems, kBlockBytes / sizeof(half));
            if (rows < kBatchRows) {
                Duplicate(input, static_cast<half>(0), prefixRows_ * blockElems);
                PipeBarrier<PIPE_V>();
            }
            DataCopyExtParams copyParams = {
                static_cast<uint16_t>(prefixRows_),
                static_cast<uint32_t>(copiedElems * sizeof(half)),
                static_cast<uint32_t>((outputSize_ * suffixCols_ - copiedElems) * sizeof(half)),
                static_cast<uint32_t>((blockElems - copyAlignedElems) * sizeof(half) / kBlockBytes), 0};
            DataCopyPadExtParams<half> padParams = {
                true, 0, static_cast<uint8_t>(copyAlignedElems - copiedElems), static_cast<half>(0)};
            DataCopyPad(input, inputGm_[outputBegin * suffixCols_], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t prefixBase = 0; prefixBase < prefixRows_; prefixBase += kPrefixChunkRows) {
                const uint32_t prefixRows = prefixRows_ - prefixBase < kPrefixChunkRows
                                                ? prefixRows_ - prefixBase : kPrefixChunkRows;
                const uint32_t tileElems = prefixRows * blockElems;
                LocalTensor<half> inputTile = input[prefixBase * blockElems];
                CastInputToFloat<half>(square, inputTile, tileElems);
                PipeBarrier<PIPE_V>();
                Mul(square, square, square, static_cast<int32_t>(tileElems));
                PipeBarrier<PIPE_V>();
                ReduceRows(square, prefixRows, blockElems);
                if (prefixBase == 0U) {
                    Adds(acc, square, 0.0F, static_cast<int32_t>(blockElems));
                } else {
                    Add(acc, acc, square, static_cast<int32_t>(blockElems));
                }
                PipeBarrier<PIPE_V>();
            }
            TransDataTo5HD<float>(dstList0, srcList0, transposeParams);
            TransDataTo5HD<float>(dstList1, srcList1, transposeParams);
            PipeBarrier<PIPE_V>();
            ReduceRows(gathered, suffixCols_, kBatchRows);
            CastFloatToOutput<half>(output, gathered, kBatchRows);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams outputParams = {
                1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0};
            DataCopyPad(outputGm_[outputBegin], output, outputParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void ReduceRows(LocalTensor<float>& data, uint32_t rows, uint32_t rowWidth)
    {
        while (rows > 1U) {
            const uint32_t pairs = rows / 2U;
            Add(data, data, data[pairs * rowWidth], static_cast<int32_t>(pairs * rowWidth));
            if ((rows & 1U) != 0U) {
                Adds(data[pairs * rowWidth], data[(rows - 1U) * rowWidth],
                     0.0F, static_cast<int32_t>(rowWidth));
            }
            PipeBarrier<PIPE_V>();
            rows = pairs + (rows & 1U);
        }
    }

    static constexpr uint32_t kBatchRows = 32;
    static constexpr uint32_t kMaxPrefixRows = 16;
    static constexpr uint32_t kPrefixChunkRows = 8;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> squareBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> gatherBuf_;
    TBuf<QuePosition::VECCALC> outputBuf_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint32_t prefixRows_ = 0;
    uint32_t suffixCols_ = 0;
    uint32_t outputSize_ = 0;
    uint32_t smallCoreItems_ = 0;
    uint32_t bigCoreCount_ = 0;
};

template <uint32_t Cols = 65U>
__aicore__ inline void RunSquareSumV1Axis03Widths(GM_ADDR input, GM_ADDR output,
                                                 const SquareSumV1TilingData& tiling)
{
    if (tiling.dims[3] == Cols) {
        KernelSquareSumV1Rank4Axis03Pipeline<272U, Cols> kernel;
        kernel.Init(input, output, tiling);
        kernel.Process();
    } else if constexpr (Cols < 69U) {
        RunSquareSumV1Axis03Widths<Cols + 1U>(input, output, tiling);
    }
}

template <uint32_t Cols>
class KernelSquareSumV1NarrowAxis2 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SquareSumV1TilingData& tiling)
    {
        reduceRows_ = tiling.reduceLength;
        smallCoreItems_ = tiling.smallCoreItems;
        bigCoreCount_ = tiling.bigCoreCount;
        inputGm_.SetGlobalBuffer((__gm__ half*)input, tiling.inputSize);
        outputGm_.SetGlobalBuffer((__gm__ half*)output, tiling.outputSize);
        pipe_.InitBuffer(inputQueue_, 2, kNarrowAxis2TileRows * Cols * sizeof(half));
        pipe_.InitBuffer(squareBuf_, kNarrowAxis2TileRows * Cols * sizeof(float));
        pipe_.InitBuffer(gatherBuf_, kNarrowAxis2ReduceRows * Cols * sizeof(float));
        pipe_.InitBuffer(offsetBuf_, kNarrowAxis2ReduceRows * Cols * sizeof(uint32_t));
        pipe_.InitBuffer(accBuf_, kNarrowAxis2ResultElems * sizeof(float));
        pipe_.InitBuffer(outputBuf_, kNarrowAxis2ResultElems * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t gatherElems = kNarrowAxis2ReduceRows * Cols;
        LocalTensor<int32_t> offsets = offsetBuf_.Get<int32_t>();
        LocalTensor<int32_t> temp = gatherBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsets, 0, 1, gatherElems);
        PipeBarrier<PIPE_V>();
        ShiftRight(temp, offsets, 6, gatherElems);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(Cols * sizeof(float)), gatherElems);
        Muls(temp, temp,
             static_cast<int32_t>((1 - static_cast<int32_t>(kNarrowAxis2ReduceRows * Cols)) *
                                  static_cast<int32_t>(sizeof(float))), gatherElems);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, temp, gatherElems);
        PipeBarrier<PIPE_V>();

        uint32_t groupBegin = 0U;
        uint32_t groupCount = 0U;
        GetCoreRange(smallCoreItems_, bigCoreCount_, groupBegin, groupCount);
        for (uint32_t group = groupBegin; group < groupBegin + groupCount; ++group) {
            Duplicate(accBuf_.Get<float>(), 0.0F, kNarrowAxis2ResultElems);
            ProcessGroup(group);
            CopyOut(group);
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t group)
    {
        constexpr uint32_t tailRows = kNarrowAxis2TileRows / 2U;
        const uint32_t validTailRows = reduceRows_ - 2U * kNarrowAxis2TileRows;
        CopyIn<kNarrowAxis2TileRows>(group, 0U, kNarrowAxis2TileRows);
        CopyIn<kNarrowAxis2TileRows>(group, kNarrowAxis2TileRows, kNarrowAxis2TileRows);
        ComputeFixed<kNarrowAxis2TileRows>(kNarrowAxis2TileRows);
        if (validTailRows != 0U) {
            CopyIn<tailRows>(group, 2U * kNarrowAxis2TileRows, validTailRows);
        }
        ComputeFixed<kNarrowAxis2TileRows>(kNarrowAxis2TileRows);
        if (validTailRows != 0U) {
            ComputeFixed<tailRows>(validTailRows);
        }
    }

    template <uint32_t Rows>
    __aicore__ inline void CopyIn(uint32_t group, uint32_t rowBase, uint32_t validRows)
    {
        LocalTensor<half> input = inputQueue_.AllocTensor<half>();
        const uint32_t elems = validRows * Cols;
        if (validRows == Rows || (elems & (kBlockBytes / sizeof(half) - 1U)) == 0U) {
            DataCopy(input, inputGm_[(group * reduceRows_ + rowBase) * Cols], elems);
        } else {
            const uint32_t alignedElems = AlignUp(elems, kBlockBytes / sizeof(half));
            DataCopyExtParams params = {1, static_cast<uint32_t>(elems * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> pad = {
                true, 0, static_cast<uint8_t>(alignedElems - elems), static_cast<half>(0)};
            DataCopyPad(input, inputGm_[(group * reduceRows_ + rowBase) * Cols], params, pad);
        }
        inputQueue_.EnQue(input);
    }

    template <uint32_t Rows>
    __aicore__ inline void ComputeFixed(uint32_t validRows)
    {
        const uint32_t elems = validRows * Cols;
        LocalTensor<half> input = inputQueue_.DeQue<half>();
        LocalTensor<float> square = squareBuf_.Get<float>();
        LocalTensor<float> gathered = gatherBuf_.Get<float>();
        if constexpr (Rows < kNarrowAxis2ReduceRows) {
            Duplicate(square, 0.0F, kNarrowAxis2ReduceRows * Cols);
            PipeBarrier<PIPE_V>();
        } else if (validRows < Rows) {
            Duplicate(square, 0.0F, Rows * Cols);
            PipeBarrier<PIPE_V>();
        }
        CastInputToFloat<half>(square, input, elems);
        PipeBarrier<PIPE_V>();
        Mul(square, square, square, elems);
        PipeBarrier<PIPE_V>();
        ReduceRows<Rows>(square);
        Gather(gathered, square, offsetBuf_.Get<uint32_t>(), 0,
               kNarrowAxis2ReduceRows * Cols);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(square, gathered, kNarrowAxis2ReduceRows, Cols, 1, 1,
                       kNarrowAxis2ReduceRows * sizeof(float) / kBlockBytes);
        PipeBarrier<PIPE_V>();
        Add(accBuf_.Get<float>(), accBuf_.Get<float>(), square, kNarrowAxis2ResultElems);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
    }

    template <uint32_t Rows>
    __aicore__ inline void ReduceRows(LocalTensor<float>& data)
    {
        if constexpr (Rows > kNarrowAxis2ReduceRows) {
            constexpr uint32_t pairs = Rows / 2U;
            Add(data, data, data[pairs * Cols], pairs * Cols);
            PipeBarrier<PIPE_V>();
            ReduceRows<pairs>(data);
        }
    }

    __aicore__ inline void CopyOut(uint32_t group)
    {
        LocalTensor<half> output = outputBuf_.Get<half>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        CastFloatToOutput<half>(output, acc, kNarrowAxis2ResultElems);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams params = {1, static_cast<uint32_t>(Cols * sizeof(half)), 0, 0, 0};
        DataCopyPad(outputGm_[group * Cols], output, params);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<QuePosition::VECCALC> squareBuf_;
    TBuf<QuePosition::VECCALC> gatherBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECOUT> outputBuf_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint32_t reduceRows_ = 0U;
    uint32_t smallCoreItems_ = 0U;
    uint32_t bigCoreCount_ = 0U;
};

template <uint32_t Cols>
__aicore__ inline void RunSquareSumV1NarrowAxis2(GM_ADDR input, GM_ADDR output,
                                                 const SquareSumV1TilingData& tiling)
{
    KernelSquareSumV1NarrowAxis2<Cols> kernel;
    kernel.Init(input, output, tiling);
    kernel.Process();
}

template <typename T>
class KernelSquareSumV1Wide {
    static constexpr uint32_t kTile = 8192U;
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const SquareSumV1TilingData& tiling)
    {
        rank_ = tiling.rank;
        mask_ = tiling.axisMask;
        reduceLength_ = tiling.wideReduceLength;
        lastReduced_ = (mask_ & (1U << (rank_ - 1U))) != 0U;
        uint64_t stride = 1U;
        bool trailing = true;
        for (int32_t d = rank_ - 1; d >= 0; --d) {
            dims_[d] = tiling.dims[d];
            strides_[d] = stride;
            stride *= dims_[d];
            trailing = trailing && (((mask_ & (1U << d)) != 0U) == lastReduced_);
            if (trailing) inner_ *= dims_[d];
        }
        innerTiles_ = (inner_ + kTile - 1U) / kTile;
        outputTasks_ = lastReduced_ ? tiling.wideOutputSize
            : (tiling.wideOutputSize / inner_) * innerTiles_;
        reduceTasks_ = lastReduced_ ? (reduceLength_ / inner_) * innerTiles_ : reduceLength_;
        splits_ = outputTasks_ < GetBlockNum() ? GetBlockNum() / outputTasks_ : 1U;
        if (splits_ > reduceTasks_) splits_ = reduceTasks_;
        input_.SetGlobalBuffer((__gm__ T*)input, tiling.wideInputSize);
        input_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        output_.SetGlobalBuffer((__gm__ T*)output, tiling.wideOutputSize);
        workspace_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace), GetBlockNum() * kTile);
        pipe_.InitBuffer(raw_, 2, kTile * sizeof(T));
        pipe_.InitBuffer(square_, kTile * sizeof(float));
        pipe_.InitBuffer(acc_, kTile * sizeof(float));
        pipe_.InitBuffer(out_, kTile * sizeof(T));
        pipe_.InitBuffer(partial_, 128U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint64_t tasks = outputTasks_ * splits_;
        const uint64_t small = tasks / GetBlockNum();
        const uint32_t big = tasks % GetBlockNum();
        const uint64_t begin = GetBlockIdx() * small +
            (GetBlockIdx() < big ? GetBlockIdx() : big);
        const uint64_t end = begin + small + (GetBlockIdx() < big ? 1U : 0U);
        auto acc = acc_.template Get<float>();
        auto square = square_.template Get<float>();
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t outputTask = task / splits_;
            const uint32_t split = task % splits_;
            const uint64_t outputIndex = OutputIndex(outputTask);
            const uint64_t rows = reduceTasks_ / splits_;
            const uint32_t extra = reduceTasks_ % splits_;
            const uint64_t first = split * rows + (split < extra ? split : extra);
            const uint64_t stop = first + rows + (split < extra ? 1U : 0U);
            Duplicate(acc, 0.0F, kTile);
            PipeBarrier<PIPE_V>();
            Load(outputIndex, first);
            for (uint64_t r = first; r < stop; ++r) {
                if (r + 1U < stop) Load(outputIndex, r + 1U);
                auto raw = raw_.template DeQue<T>();
                const uint32_t count = AlignUp(InputCount(outputIndex, r), 32U);
                SquareInputToFloat<T>(square, raw, count);
                if constexpr (!std::is_same_v<T, float>) {
                    PipeBarrier<PIPE_V>();
                    Cast(raw, square, RoundMode::CAST_RINT, count);
                    PipeBarrier<PIPE_V>();
                    Cast(square, raw, RoundMode::CAST_NONE, count);
                }
                PipeBarrier<PIPE_V>();
                Add(acc, acc, square, count);
                PipeBarrier<PIPE_V>();
                raw_.FreeTensor(raw);
            }
            if (lastReduced_) {
                auto partial = partial_.template Get<float>();
                WholeReduceSum(partial, acc, 64U, 128U, 1U, 1U, 8U);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(square, partial, 64U, 2U, 1U, 1U, 8U);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(acc, square, 2U, 1U, 1U, 1U, 1U);
                PipeBarrier<PIPE_V>();
            }
            const uint32_t cols = OutputCount(outputIndex);
            if (splits_ == 1U) {
                Store(outputIndex, cols);
            } else {
                PipeBarrier<PIPE_ALL>();
                DataCopy(workspace_[task * kTile], acc, AlignUp(cols, 32U));
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (splits_ == 1U) return;
        SyncAll<true>();
        if (GetBlockIdx() >= outputTasks_) return;
        const uint64_t outputTask = GetBlockIdx();
        const uint64_t outputIndex = OutputIndex(outputTask);
        const uint32_t cols = OutputCount(outputIndex);
        Duplicate(acc, 0.0F, AlignUp(cols, 32U));
        PipeBarrier<PIPE_ALL>();
        for (uint32_t split = 0U; split < splits_; ++split) {
            DataCopy(square, workspace_[(outputTask * splits_ + split) * kTile], AlignUp(cols, 32U));
            PipeBarrier<PIPE_ALL>();
            Add(acc, acc, square, AlignUp(cols, 32U));
            PipeBarrier<PIPE_ALL>();
        }
        Store(outputIndex, cols);
    }

private:
    __aicore__ inline uint64_t Offset(uint64_t index, bool reduced) const
    {
        uint64_t offset = 0U;
        for (int32_t d = rank_ - 1; d >= 0; --d) {
            if (((mask_ & (1U << d)) != 0U) == reduced) {
                offset += (index % dims_[d]) * strides_[d];
                index /= dims_[d];
            }
        }
        return offset;
    }
    __aicore__ inline uint64_t OutputIndex(uint64_t task) const
    {
        return lastReduced_ ? task : (task / innerTiles_) * inner_ + (task % innerTiles_) * kTile;
    }
    __aicore__ inline uint32_t OutputCount(uint64_t output) const
    {
        const uint64_t remaining = lastReduced_ ? 1U : inner_ - output % inner_;
        return remaining < kTile ? remaining : kTile;
    }
    __aicore__ inline uint32_t InputCount(uint64_t output, uint64_t task) const
    {
        if (!lastReduced_) return OutputCount(output);
        const uint64_t remaining = inner_ - (task % innerTiles_) * kTile;
        return remaining < kTile ? remaining : kTile;
    }
    __aicore__ inline void Load(uint64_t output, uint64_t task)
    {
        const uint64_t reduced = lastReduced_
            ? (task / innerTiles_) * inner_ + (task % innerTiles_) * kTile : task;
        auto raw = raw_.template AllocTensor<T>();
        const uint32_t count = InputCount(output, task);
        Duplicate(raw, static_cast<T>(0), AlignUp(count, 32U));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams params = {1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad = {true, 0U,
            static_cast<uint8_t>(AlignUp(count, 32U / sizeof(T)) - count), static_cast<T>(0)};
        DataCopyPad(raw, input_[Offset(output, false) + Offset(reduced, true)], params, pad);
        raw_.EnQue(raw);
    }
    __aicore__ inline void Store(uint64_t index, uint32_t count)
    {
        auto out = out_.template Get<T>();
        auto acc = acc_.template Get<float>();
        CastFloatToOutput<T>(out, acc, AlignUp(count, 32U));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams params = {1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(output_[index], out, params);
        PipeBarrier<PIPE_ALL>();
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> raw_;
    TBuf<QuePosition::VECCALC> square_, acc_, partial_;
    TBuf<QuePosition::VECOUT> out_;
    GlobalTensor<T> input_, output_;
    GlobalTensor<float> workspace_;
    uint64_t dims_[8], strides_[8];
    uint64_t inner_ = 1U, innerTiles_, outputTasks_, reduceTasks_, reduceLength_;
    uint32_t rank_, mask_, splits_;
    bool lastReduced_;
};

}

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(14)) {
        KernelSquareSumV1Wide<DTYPE_INPUT> kernel;
        kernel.Init(input, output, workspace, tilingData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(4)) {
        RunSquareSumV1NarrowAxis2<5U>(input, output, tilingData);
        return;
    }
    if (TILING_KEY_IS(0)) {
        RunSquareSumV1(input, output, tilingData.smallCoreItems, tilingData.bigCoreCount,
                       tilingData.reduceLength, tilingData.alignedReduceLength);
    } else if (TILING_KEY_IS(2)) {
        RunSquareSumV1Generic<DTYPE_INPUT>(input, output, workspace, tilingData);
    } else if (TILING_KEY_IS(3)) {
        RunSquareSumV1Split(input, output, tilingData.smallCoreItems, tilingData.bigCoreCount,
                            tilingData.reduceLength, tilingData.alignedReduceLength);
    } else if (TILING_KEY_IS(5)) {
        RunSquareSumV1SuffixWide<DTYPE_INPUT>(input, output, tilingData.smallCoreItems,
                                             tilingData.bigCoreCount, tilingData.reduceLength);
    } else if (TILING_KEY_IS(6)) {
        RunSquareSumV1Generic<DTYPE_INPUT>(input, output, workspace, tilingData, true);
    } else if (TILING_KEY_IS(7)) {
        RunSquareSumV1SuffixWide<DTYPE_INPUT, true>(input, output, tilingData.smallCoreItems,
                                                   tilingData.bigCoreCount, tilingData.reduceLength);
    } else if (TILING_KEY_IS(8)) {
        RunSquareSumV1Generic<float, true>(input, output, workspace, tilingData);
    } else if (TILING_KEY_IS(9)) {
        KernelSquareSumV1Generic<bfloat16_t> kernel;
        kernel.Init(input, output, workspace, tilingData);
        kernel.ProcessRank4Axis12Bf16();
    } else if (TILING_KEY_IS(10)) {
        if (tilingData.dims[0] <= 16U && tilingData.dims[3] % 8U == 0U) {
            KernelSquareSumV1Rank4Axis03Batch32 kernel;
            kernel.Init(input, output, tilingData);
            kernel.Process();
        } else if (tilingData.dims[0] >= 2U && tilingData.dims[0] <= 22U &&
                   tilingData.dims[0] % 2U == 0U &&
                   tilingData.dims[3] >= 65U && tilingData.dims[3] <= 69U) {
            RunSquareSumV1Axis03Widths(input, output, tilingData);
        } else if (tilingData.dims[0] > 11U) {
            KernelSquareSumV1Rank4Axis03Pipeline<16U> kernel;
            kernel.Init(input, output, tilingData);
            kernel.Process();
        } else {
            KernelSquareSumV1Generic<half> kernel;
            kernel.Init(input, output, workspace, tilingData);
            kernel.ProcessRank4Axis03Batches();
        }
    } else if (TILING_KEY_IS(11)) {
        KernelSquareSumV1Generic<bfloat16_t> kernel;
        kernel.Init(input, output, workspace, tilingData, true);
        kernel.ProcessRank4Axis12Bf16Padded();
    } else if (TILING_KEY_IS(12)) {
        RunSquareSumV1SuffixWide<bfloat16_t, true, true>(
            input, output, tilingData.smallCoreItems, tilingData.bigCoreCount,
            tilingData.reduceLength);
    } else if (TILING_KEY_IS(13)) {
        KernelSquareSumV1Generic<bfloat16_t, false, 64U> kernel;
        kernel.Init(input, output, workspace, tilingData, true);
        kernel.ProcessRank4Axis12Bf16Balanced();
    }
}
