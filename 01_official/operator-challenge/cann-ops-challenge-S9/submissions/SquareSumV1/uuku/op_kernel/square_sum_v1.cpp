#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

namespace {
constexpr uint32_t kDimNum = 5U;
constexpr uint32_t kBufferNum = 1U;
constexpr uint32_t kModeSuffixReduce = 0U;
constexpr uint32_t kModeInnerAccum = 1U;
constexpr uint32_t kModePlaneReduce = 2U;
constexpr uint32_t kModeGenericFallback = 3U;
constexpr uint32_t kModeContiguousBlockReduce = 4U;
constexpr uint32_t kModeRank4Reduce02 = 5U;
constexpr uint32_t kSuffixOutTileElems = 256U;
constexpr uint32_t kSuffixBatchRows = 128U;
constexpr uint32_t kSuffixBatchMaxReduce = 64U;
constexpr uint32_t kInnerReduceBatchRows = 64U;
constexpr uint32_t kRank4Reduce02BatchB = 2U;
// Non-BF16 inner reductions use MulAddDst directly into the accumulator.
// Keep one full vector repeat per MTE2 batch while the double input queue
// overlaps the next transfer with the current vector operation.
constexpr uint32_t kInnerPipelinedBatchRows = 64U;
constexpr uint32_t kPlaneReduceChunkElems = 64U;
// For a large output plane with a moderately narrow suffix reduction, reading
// complete plane slices is substantially cheaper than issuing one padded 2-D
// DMA per reduction stripe.  Keep this as a layout/width class: all geometry
// below is derived from the runtime shape and the available UB.
constexpr uint32_t kPlaneGatherMaxSuffixElems = 128U;
constexpr uint32_t kPlaneGatherGroupRows = 32U;
constexpr uint32_t kPlaneGatherMinRowsPerCore = 256U;
constexpr uint32_t kPlaneGatherMinPrefixReduce = 4U;
constexpr uint32_t kPlaneGatherMinBatchRows = 64U;
constexpr uint32_t kPlaneGatherMaxBatchRows = 255U;
constexpr uint32_t kAivUbBytes = 192U * 1024U;
// A contiguous middle reduction followed by a moderately wide suffix is a
// dense [reduceRows, innerCols] matrix reduction.  The class bounds are based
// on vector/DMA geometry; batch rows and the alignment-cycle DMA schedule are
// derived from the runtime layout and UB budget.
constexpr uint32_t kMiddleReduceMinRows = 512U;
constexpr uint32_t kMiddleReduceMinInnerCols = 65U;
constexpr uint32_t kMiddleReduceMaxInnerCols = 512U;
constexpr uint32_t kMiddleReduceComputeRows = 32U;
constexpr uint32_t kMiddleReduceMaxBatchRows = 255U;
constexpr uint32_t kMiddleReduceMinFullPlanesPerCore = 2U;
constexpr uint32_t kPhaseMinReduceRows = 64U * 1024U;
constexpr uint32_t kPhaseMinInnerCols = 257U;
constexpr uint32_t kPhaseMaxInnerCols = 1024U;
constexpr uint32_t kPhaseMaxSuperRows = 32U;
constexpr uint64_t kPhaseUbGuardBytes = 8ULL * 1024ULL;
// Large contiguous suffix reductions are naturally a 2D [output-row, reduce]
// tile.  Keeping one 64-element reduction stripe per row lets WholeReduceSum
// stay on its efficient vector path while one MTE2 command fetches every row
// in the batch.
constexpr uint32_t kLargeSuffixChunkElems = 64U;
constexpr uint32_t kLargeSuffixBatchRows = 128U;
constexpr uint32_t kWideBf16SuffixMinReduce = 1025U;
constexpr uint32_t kWideBf16SuffixMaxReduce = 2048U;
constexpr uint32_t kWideBf16SuffixChunkElems = 208U;
constexpr uint32_t kWideBf16SuffixBatchRows = 128U;
constexpr uint32_t kWideBf16SuffixCalcRows = 104U;
constexpr uint32_t kWideBf16SuffixMinRowsPerCore = 128U;
constexpr uint32_t kFullRowBf16MaxBatchRows = 32U;
constexpr uint32_t kFullRowBf16MinBatchRows = 16U;
constexpr uint64_t kFullRowBf16UbGuardBytes = 8ULL * 1024ULL;
// Eight FP32 block-partials occupy one 32B vector block. For suffix lengths
// congruent to one modulo 8*64, compact rows have aligned interior partials.
constexpr uint32_t kCompactSuffixGroupRows = 8U;
constexpr uint32_t kCompactSuffixPeriodElems = 8U * 64U;

__aicore__ static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ static inline uint32_t GcdU32(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0U) {
        const uint32_t rem = lhs % rhs;
        lhs = rhs;
        rhs = rem;
    }
    return lhs;
}

__aicore__ static inline float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t raw = static_cast<uint32_t>(bits) << 16U;
    return *reinterpret_cast<float*>(&raw);
}

__aicore__ static inline bfloat16_t FloatToBf16(float value)
{
    uint32_t raw = *reinterpret_cast<uint32_t*>(&value);
    uint16_t bits;
    if ((raw & 0x7F800000U) == 0x7F800000U &&
        (raw & 0x007FFFFFU) != 0U) {
        // Preserve every FP32 NaN as a BF16 NaN.  Round-to-nearest-even can
        // otherwise carry a tiny NaN payload into the infinity encoding.
        bits = static_cast<uint16_t>((raw >> 16U) | 0x0040U);
    } else {
        const uint32_t lsb = (raw >> 16U) & 1U;
        raw += 0x7FFFU + lsb;
        bits = static_cast<uint16_t>(raw >> 16U);
    }
    bfloat16_t out;
    *reinterpret_cast<uint16_t*>(&out) = bits;
    return out;
}

template <typename T>
__aicore__ static inline uint32_t AlignElems32(uint32_t elems)
{
    constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
    return ((elems + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
}

template <typename T>
class SquareSumKernel {
public:
    __aicore__ inline SquareSumKernel() = default;

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                GM_ADDR tiling, TPipe* pipeIn)
    {
        pipe = pipeIn;
        GET_TILING_DATA(td, tiling);
        mode = td.mode;
        inputSize = td.input_size;
        outputSize = td.output_size;
        blockElems = td.block_elems;
        totalBlocks = td.total_blocks;
        tileElems = td.tile_elems;
        accTileElems = td.acc_tile_elems;
        reduceSize = td.reduce_size;
        innerSize = td.inner_size;
        taskNum = td.task_num;
        for (uint32_t i = 0U; i < kDimNum; ++i) {
            inDims[i] = td.in_dims[i];
            outStrides[i] = td.out_strides[i];
            inStrides[i] = td.in_strides[i];
            reduceMask[i] = td.reduce_mask[i];
        }

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), inputSize);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), outputSize);
        if constexpr (std::is_same_v<T, float>) {
            if (CanProcessPhaseLargeMiddleReduce()) {
                workspaceGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(workspace),
                    static_cast<uint32_t>(PhaseWorkspaceElems()));
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (CanProcessRowShardedBf16Middle()) {
                workspaceGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(workspace),
                    static_cast<uint32_t>(RowShardWorkspaceElems()));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (outputSize == 0U || tileElems == 0U) {
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            if (CanProcessPhaseLargeMiddleReduce()) {
                InitPhaseLargeMiddleReduceBuffers();
                ProcessPhaseLargeMiddleReduce();
                return;
            }
        }
        if (mode == kModeSuffixReduce) {
            if (CanProcessLargeSuffixChunked()) {
                InitLargeSuffixChunkedBuffers();
                ProcessLargeSuffixChunked();
                return;
            }
            InitSuffixBuffers();
            ProcessSuffixReduce();
        } else if (mode == kModeRank4Reduce02) {
            InitRank4Reduce02Buffers();
            ProcessRank4Reduce02();
        } else if (mode == kModeInnerAccum) {
            InitInnerBuffers();
            ProcessInnerAccum();
        } else if (mode == kModePlaneReduce) {
            if (CanProcessNarrowPlaneGather()) {
                InitDirectPlaneReduceBuffers();
                ProcessDirectPlaneReduce();
                return;
            }
            InitPlaneBuffers();
            ProcessPlaneReduce();
        } else if (mode == kModeContiguousBlockReduce) {
            InitContiguousBlockBuffers();
            ProcessContiguousBlockReduce();
        } else {
            InitSuffixBuffers();
            ProcessGenericFallback();
        }
    }

    __aicore__ inline void ProcessRowShardedBf16Entry()
    {
        // Host TilingKey 4 already proves the general row-shard admission,
        // including layout, output ownership and nonzero UB capacity. Other
        // BF16 layouts keep key 2 and the complete original Process fallback.
        if (outputSize == 0U) {
            return;
        }
        if (CanProcessWideDmaRowShard()) {
            if (CanReuseWideDmaCalc()) {
                InitWideDmaRowShardBuffers<true>();
                ProcessWideDmaRowShardedMiddle<true>();
            } else {
                InitWideDmaRowShardBuffers<false>();
                ProcessWideDmaRowShardedMiddle<false>();
            }
            return;
        }
        InitRowShardedBf16MiddleBuffers();
        ProcessRowShardedBf16Middle();
    }

private:
    __aicore__ inline void CalcRange(uint32_t total, uint32_t& start, uint32_t& end)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t base = total / blockNum;
        const uint32_t remain = total % blockNum;
        start = blockIdx * base + (blockIdx < remain ? blockIdx : remain);
        end = start + base + (blockIdx < remain ? 1U : 0U);
    }

    __aicore__ inline void CalcOutputRange(uint32_t& start, uint32_t& end)
    {
        uint32_t blockStart = 0U;
        uint32_t blockEnd = 0U;
        CalcRange(totalBlocks, blockStart, blockEnd);
        start = blockStart * blockElems;
        end = MinU32(outputSize, blockEnd * blockElems);
    }

    __aicore__ inline void InitSuffixBuffers()
    {
        const uint32_t alignedTileT = AlignElems32<T>(tileElems);
        const uint32_t alignedTileF = AlignElems32<float>(tileElems);
        const uint32_t alignedOutT = AlignElems32<T>(kSuffixOutTileElems);
        const uint32_t alignedOutF = AlignElems32<float>(kSuffixOutTileElems);
        pipe->InitBuffer(inQueue, kBufferNum, alignedTileT * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedOutT * sizeof(T));
        pipe->InitBuffer(calcBuf, alignedTileF * sizeof(float));
        pipe->InitBuffer(workBuf, alignedTileF * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(kSuffixBatchRows) * sizeof(float));
        pipe->InitBuffer(valueFloatBuf, alignedOutF * sizeof(float));
    }

    __aicore__ inline void InitInnerBuffers()
    {
        const uint32_t alignedRowElems = AlignElems32<T>(accTileElems);
        pipe->InitBuffer(inQueue, 2U,
                         alignedRowElems * kInnerPipelinedBatchRows * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedRowElems * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(calcBuf,
                             alignedRowElems * kInnerPipelinedBatchRows * sizeof(float));
            pipe->InitBuffer(squareBuf,
                             alignedRowElems * kInnerPipelinedBatchRows * sizeof(float));
        }
        pipe->InitBuffer(accBuf, alignedRowElems * sizeof(float));
    }

    __aicore__ inline void InitRank4Reduce02Buffers()
    {
        const uint32_t alignedRowT = AlignElems32<T>(accTileElems);
        const uint32_t alignedRowF = AlignElems32<float>(accTileElems);
        if (CanProcessRank4Reduce02BatchB()) {
            const uint32_t inputElems = kRank4Reduce02BatchB * inDims[3] * alignedRowT;
            pipe->InitBuffer(rank4InQueue, 2U, inputElems * sizeof(T));
            pipe->InitBuffer(outQueue, kBufferNum,
                             kRank4Reduce02BatchB * alignedRowT * sizeof(T));
            pipe->InitBuffer(accBuf,
                             kRank4Reduce02BatchB * alignedRowF * sizeof(float));
            return;
        }
        const uint32_t batchElemsT = alignedRowT * kInnerReduceBatchRows;
        pipe->InitBuffer(rank4InQueue, 2U, batchElemsT * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedRowT * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t batchElemsF = alignedRowF * kInnerReduceBatchRows;
            pipe->InitBuffer(calcBuf, batchElemsF * sizeof(float));
            pipe->InitBuffer(squareBuf, batchElemsF * sizeof(float));
        }
        pipe->InitBuffer(accBuf, alignedRowF * sizeof(float));
    }

    __aicore__ inline bool CanProcessRank4Reduce02BatchB() const
    {
        const uint32_t dimB = inDims[2];
        const uint32_t dimD = inDims[4];
        return dimB != 0U && accTileElems == dimD &&
               taskNum == (dimB + kRank4Reduce02BatchB - 1U) /
                          kRank4Reduce02BatchB;
    }

    __aicore__ inline uint32_t PlaneSuffixReduceSize()
    {
        const uint32_t suffixReduceStart = SuffixReduceStart();
        uint32_t suffixReduceSize = 1U;
        for (uint32_t dim = suffixReduceStart; dim < kDimNum; ++dim) {
            suffixReduceSize *= inDims[dim];
        }
        return suffixReduceSize;
    }

    __aicore__ inline uint64_t NarrowPlaneGatherBufferBytes(
        uint32_t rows, uint32_t sourceCols, uint32_t paddedCols)
    {
        const uint32_t tightElems = rows * sourceCols;
        uint64_t bytes = 0ULL;
        bytes += 2ULL * AlignElems32<T>(tightElems) * sizeof(T);
        bytes += static_cast<uint64_t>(AlignElems32<T>(rows)) * sizeof(T);
        bytes += static_cast<uint64_t>(rows) * paddedCols * sizeof(float);
        bytes += static_cast<uint64_t>(kPlaneGatherGroupRows) * paddedCols *
                 sizeof(uint32_t);
        bytes += static_cast<uint64_t>(AlignElems32<float>(rows)) * sizeof(float);
        bytes += static_cast<uint64_t>(AlignElems32<float>(tightElems)) * sizeof(float);
        return bytes;
    }

    __aicore__ inline uint32_t NarrowPlaneGatherRows()
    {
        const uint32_t sourceCols = PlaneSuffixReduceSize();
        const uint32_t paddedCols = AlignElems32<float>(sourceCols);
        for (uint32_t rows = kPlaneGatherMaxBatchRows;
             rows >= kPlaneGatherMinBatchRows; --rows) {
            if (NarrowPlaneGatherBufferBytes(rows, sourceCols, paddedCols) <=
                kAivUbBytes) {
                return rows;
            }
        }
        return 0U;
    }

    __aicore__ inline bool CanProcessNarrowPlaneGather()
    {
        if constexpr (!std::is_same_v<T, half>) {
            return false;
        } else {
            const uint32_t suffixReduceStart = SuffixReduceStart();
            const uint32_t planeStart = PlaneStart(suffixReduceStart);
            const uint32_t suffixReduceSize = PlaneSuffixReduceSize();
            const uint32_t prefixReduceSize =
                suffixReduceSize == 0U ? 0U : reduceSize / suffixReduceSize;
            const uint64_t minOutputSize =
                static_cast<uint64_t>(GetBlockNum()) * kPlaneGatherMinRowsPerCore;
            return suffixReduceStart != 0U && planeStart < suffixReduceStart &&
                   innerSize != 0U &&
                   suffixReduceSize > kPlaneReduceChunkElems &&
                   suffixReduceSize <= kPlaneGatherMaxSuffixElems &&
                   prefixReduceSize >= kPlaneGatherMinPrefixReduce &&
                   outputSize >= minOutputSize &&
                   NarrowPlaneGatherRows() != 0U;
        }
    }

    __aicore__ inline void InitNarrowPlaneGatherBuffers()
    {
        const uint32_t rows = NarrowPlaneGatherRows();
        const uint32_t sourceCols = PlaneSuffixReduceSize();
        const uint32_t paddedCols = AlignElems32<float>(sourceCols);
        const uint32_t tightElems = rows * sourceCols;
        pipe->InitBuffer(inQueue, 2U,
                         AlignElems32<T>(tightElems) * sizeof(T));
        pipe->InitBuffer(outQueue, 1U,
                         AlignElems32<T>(rows) * sizeof(T));
        pipe->InitBuffer(workBuf, rows * paddedCols * sizeof(float));
        pipe->InitBuffer(valueFloatBuf,
                         kPlaneGatherGroupRows * paddedCols * sizeof(uint32_t));
        pipe->InitBuffer(sumBuf,
                         AlignElems32<float>(rows) * sizeof(float));
        pipe->InitBuffer(accBuf,
                         AlignElems32<float>(tightElems) * sizeof(float));
    }

    __aicore__ inline void BuildNarrowPlaneGatherIndex(
        uint32_t sourceCols, uint32_t paddedCols)
    {
        LocalTensor<uint32_t> indexLocal = valueFloatBuf.Get<uint32_t>();
        LocalTensor<int32_t> indexSigned =
            indexLocal.template ReinterpretCast<int32_t>();
        const uint32_t paddedElems = kPlaneGatherGroupRows * paddedCols;
        CreateVecIndex(indexSigned, 0, paddedElems);
        Muls(indexSigned, indexSigned, static_cast<int32_t>(sizeof(float)),
             paddedElems);

        // Turn flattened padded-row offsets into offsets into the compact
        // [row, sourceCols] accumulator.  The same 32-row index template is
        // reused for every output batch.
        const uint32_t paddingCols = paddedCols - sourceCols;
        for (uint32_t row = 1U; row < kPlaneGatherGroupRows; ++row) {
            LocalTensor<int32_t> rowIndex = indexSigned[row * paddedCols];
            Adds(rowIndex, rowIndex,
                 -static_cast<int32_t>(row * paddingCols * sizeof(float)),
                 sourceCols);
        }

        // Gather evaluates the padding destinations too.  Redirect those
        // indices to a valid accumulator element; the subsequent fold only
        // consumes the real tail columns.
        if (paddingCols != 0U) {
            const uint32_t tailStart = sourceCols - kPlaneReduceChunkElems;
            const uint64_t paddingBits =
                ((1ULL << paddingCols) - 1ULL) << tailStart;
            uint64_t paddingMask[2] = {paddingBits, 0ULL};
            Duplicate<uint32_t>(
                indexLocal[kPlaneReduceChunkElems], 0U, paddingMask,
                static_cast<uint8_t>(kPlaneGatherGroupRows), 1U,
                static_cast<uint8_t>(paddedCols * sizeof(uint32_t) / 32U));
        }
    }

    __aicore__ inline void CopyNarrowPlaneTight(
        uint32_t inputBase, uint32_t prefixOffset, uint32_t rowCount,
        uint32_t sourceCols)
    {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        const uint32_t tightElems = rowCount * sourceCols;
        const uint32_t copiedElems = AlignElems32<T>(tightElems);
        const uint64_t sourceOffset =
            static_cast<uint64_t>(inputBase) + prefixOffset;

        // A plane slice is contiguous.  Use a block copy whenever its aligned
        // tail remains inside the tensor, and retain byte-exact copying for the
        // final physical slice.
        if (sourceOffset + copiedElems <= inputSize) {
            DataCopyParams blockParams{
                1U,
                static_cast<uint16_t>(copiedElems * sizeof(T) / 32U),
                0U, 0U};
            DataCopy(inputLocal, inputGm[sourceOffset], blockParams);
            inQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(tightElems * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[sourceOffset], copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessNarrowPlaneGather()
    {
        const uint32_t suffixReduceStart = SuffixReduceStart();
        const uint32_t planeStart = PlaneStart(suffixReduceStart);
        const uint32_t sourceCols = PlaneSuffixReduceSize();
        const uint32_t paddedCols = AlignElems32<float>(sourceCols);
        const uint32_t prefixReduceSize = reduceSize / sourceCols;
        const uint32_t batchRows = NarrowPlaneGatherRows();
        const uint8_t rowStrideBlocks = static_cast<uint8_t>(
            paddedCols * sizeof(float) / 32U);
        BuildNarrowPlaneGatherIndex(sourceCols, paddedCols);
        LocalTensor<uint32_t> indexLocal = valueFloatBuf.Get<uint32_t>();

        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            // Do not let a tight transfer cross from one logical output plane
            // into another; BaseInputOffset then remains valid for every row.
            const uint32_t planeOffset = outputOffset % innerSize;
            const uint32_t rowsToPlaneEnd = innerSize - planeOffset;
            const uint32_t rowCount = MinU32(
                batchRows, MinU32(end - outputOffset, rowsToPlaneEnd));
            const uint32_t tightElems = rowCount * sourceCols;
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(tightElems));

            uint32_t prefixCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t prefixReduce = 0U;
            uint32_t prefixOffset = 0U;
            CopyNarrowPlaneTight(inputBase, prefixOffset, rowCount, sourceCols);
            while (true) {
                LocalTensor<T> ready = inQueue.DeQue<T>();
                const uint32_t nextPrefixReduce = prefixReduce + 1U;
                uint32_t nextPrefixOffset = prefixOffset;
                const bool hasNext = nextPrefixReduce < prefixReduceSize;
                if (hasNext) {
                    AdvancePrefixReduceOffset(planeStart, prefixCoords,
                                              nextPrefixOffset);
                    CopyNarrowPlaneTight(inputBase, nextPrefixOffset,
                                         rowCount, sourceCols);
                }

                MulAddDst(accLocal, ready, ready, tightElems);
                inQueue.FreeTensor(ready);
                if (!hasNext) {
                    break;
                }
                prefixReduce = nextPrefixReduce;
                prefixOffset = nextPrefixOffset;
            }

            LocalTensor<float> paddedLocal = workBuf.Get<float>();
            for (uint32_t groupBase = 0U; groupBase < rowCount;
                 groupBase += kPlaneGatherGroupRows) {
                const uint32_t groupRows = MinU32(
                    kPlaneGatherGroupRows, rowCount - groupBase);
                Gather(paddedLocal[groupBase * paddedCols],
                       accLocal[groupBase * sourceCols], indexLocal, 0U,
                       groupRows * paddedCols);
            }
            Add<float>(
                paddedLocal, paddedLocal,
                paddedLocal[kPlaneReduceChunkElems],
                sourceCols - kPlaneReduceChunkElems,
                static_cast<uint8_t>(rowCount),
                {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks,
                 rowStrideBlocks});
            LocalTensor<float> sumsLocal = sumBuf.Get<float>();
            WholeReduceSum<float>(
                sumsLocal, paddedLocal,
                static_cast<int32_t>(kPlaneReduceChunkElems),
                static_cast<int32_t>(rowCount), 1, 1,
                static_cast<int32_t>(rowStrideBlocks));

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, sumsLocal, rowCount);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{
                1U, static_cast<uint32_t>(rowCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
            outputOffset += rowCount;
        }
    }

    __aicore__ inline uint32_t DirectPlanePeriod(uint32_t cols)
    {
        uint32_t divisor = 8U;  // FP32 elements in one 32-byte UB block.
        while (cols % divisor != 0U) {
            divisor /= 2U;
        }
        return 8U / divisor;
    }

    __aicore__ inline uint64_t DirectPlaneBufferBytes(uint32_t rows)
    {
        const uint32_t tightElems = rows * PlaneSuffixReduceSize();
        return 2ULL * AlignElems32<T>(tightElems) * sizeof(T) +
               (static_cast<uint64_t>(AlignElems32<float>(tightElems)) +
                kPlaneReduceChunkElems) * sizeof(float) +
               3ULL * AlignElems32<float>(rows) * sizeof(float) +
               static_cast<uint64_t>(AlignElems32<T>(rows)) * sizeof(T) +
               8ULL * 1024ULL;
    }

    __aicore__ inline uint32_t DirectPlaneRows()
    {
        for (uint32_t rows = 512U; rows >= kPlaneGatherMinBatchRows;
             rows -= 32U / sizeof(T)) {
            if (DirectPlaneBufferBytes(rows) <= kAivUbBytes) {
                return rows;
            }
        }
        return kPlaneGatherMinBatchRows;
    }

    __aicore__ inline void InitDirectPlaneReduceBuffers()
    {
        const uint32_t rows = DirectPlaneRows();
        const uint32_t tightElems = rows * PlaneSuffixReduceSize();
        pipe->InitBuffer(inQueue, 2U, AlignElems32<T>(tightElems) * sizeof(T));
        pipe->InitBuffer(outQueue, 1U, AlignElems32<T>(rows) * sizeof(T));
        // One extra vector keeps even masked-off tail reads inside this buffer.
        pipe->InitBuffer(accBuf,
                         (AlignElems32<float>(tightElems) +
                          kPlaneReduceChunkElems) * sizeof(float));
        pipe->InitBuffer(workBuf, AlignElems32<float>(rows) * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(rows) * sizeof(float));
        pipe->InitBuffer(valueFloatBuf,
                         AlignElems32<float>(rows) * sizeof(float));
        const uint32_t cols = PlaneSuffixReduceSize();
        if (CanUseDirectPlaneBlockRows(rows, cols)) {
            BuildDirectPlaneBlockIndex(rows, cols);
        }
    }

    __aicore__ inline bool CanUseDirectPlaneBlockRows(uint32_t rows,
                                                      uint32_t cols)
    {
        if (cols <= 64U || cols > 128U) {
            return false;
        }
        const uint32_t period = DirectPlanePeriod(cols);
        if (cols + 8U - 8U / period > 128U) {
            return false;
        }
        const uint32_t phasePitch =
            AlignElems32<float>((rows + period - 1U) / period);
        const uint64_t scratchElems =
            2ULL * (phasePitch * 8U + kPlaneReduceChunkElems) +
            period * phasePitch + kPlaneReduceChunkElems;
        return scratchElems * sizeof(float) <=
               static_cast<uint64_t>(AlignElems32<T>(rows * cols)) * sizeof(T);
    }

    __aicore__ inline void BuildDirectPlaneBlockIndex(uint32_t rows,
                                                      uint32_t cols)
    {
        const uint32_t period = DirectPlanePeriod(cols);
        const uint32_t phasePitch =
            AlignElems32<float>((rows + period - 1U) / period);
        LocalTensor<int32_t> index = valueFloatBuf.Get<int32_t>();
        // Seed one aligned block; doubling preserves the phase because period
        // divides eight.  The byte-offset table lives only within this launch.
        for (uint32_t row = 0U; row < 8U; ++row) {
            index.SetValue(row, static_cast<int32_t>(
                ((row % period) * phasePitch + row / period) * sizeof(float)));
        }
        const auto event = pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        for (uint32_t built = 8U; built < rows; built *= 2U) {
            Adds(index[built], index,
                 static_cast<int32_t>((built / period) * sizeof(float)),
                 MinU32(built, rows - built));
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReduceDirectPlaneBlockRows(
        uint32_t rowCount, uint32_t cols, uint32_t batchRows,
        const LocalTensor<T>& lastInput)
    {
        const uint32_t period = DirectPlanePeriod(cols);
        const uint32_t phasePitch =
            AlignElems32<float>((batchRows + period - 1U) / period);
        const uint32_t regionElems = phasePitch * 8U + kPlaneReduceChunkElems;
        const uint32_t srcStride = period * cols / 8U;
        LocalTensor<float> scratch = lastInput.template ReinterpretCast<float>();
        LocalTensor<float> first = scratch;
        LocalTensor<float> second = scratch[regionElems];
        LocalTensor<float> packed = scratch[2U * regionElems];
        LocalTensor<float> acc = accBuf.Get<float>();
        // The last input remains DeQue-owned.  Its final MulAddDst read must
        // finish before the same physical bytes become Vector scratch.
        PipeBarrier<PIPE_V>();
        for (uint32_t phase = 0U; phase < MinU32(period, rowCount); ++phase) {
            const uint32_t misalign = (phase * cols) & 7U;
            const uint32_t alignedBase = phase * cols - misalign;
            const uint32_t tail = cols + misalign - kPlaneReduceChunkElems;
            const uint64_t fullMask = ~0ULL;
            uint64_t firstMask[2] = {fullMask << misalign, 0ULL};
            uint64_t secondMask[2] = {
                tail == kPlaneReduceChunkElems ? fullMask : (1ULL << tail) - 1ULL,
                0ULL};
            const uint32_t phaseRows = (rowCount - phase + period - 1U) / period;
            // An entirely masked DataBlock does not write its destination.
            // Reusing this window across phases therefore requires this clear.
            Duplicate(second, 0.0F, phaseRows * 8U);
            PipeBarrier<PIPE_V>();
            for (uint32_t base = 0U; base < phaseRows; base += 255U) {
                const uint32_t repeats = MinU32(255U, phaseRows - base);
                const uint32_t srcOffset = alignedBase + base * period * cols;
                // Legacy BlockReduceSum takes repeatTime before either mask.
                BlockReduceSum<float>(first[base * 8U], acc[srcOffset], repeats,
                                      firstMask, 1, 1, srcStride);
                BlockReduceSum<float>(second[base * 8U],
                                      acc[srcOffset + kPlaneReduceChunkElems],
                                      repeats, secondMask, 1, 1, srcStride);
            }
            PipeBarrier<PIPE_V>();
            Add(first, first, second, phaseRows * 8U);
            PipeBarrier<PIPE_V>();
            const uint32_t fullRepeats = phaseRows / 8U;
            if (fullRepeats != 0U) {
                BlockReduceSum<float>(packed[phase * phasePitch], first,
                                      fullRepeats, 64, 1, 1, 8);
            }
            const uint32_t tailRows = phaseRows % 8U;
            if (tailRows != 0U) {
                BlockReduceSum<float>(packed[phase * phasePitch + fullRepeats * 8U],
                    first[fullRepeats * 64U], 1, tailRows * 8U, 1, 1, 8);
            }
            // A later phase reuses first/second; packed remains disjoint.
            PipeBarrier<PIPE_V>();
        }
        Gather(sumBuf.Get<float>(), packed, valueFloatBuf.Get<uint32_t>(),
               0U, rowCount);
        // The caller releases the original lastInput handle only after this
        // last Vector consumer; TQue protects its next MTE2 producer.
    }

    __aicore__ inline void ReduceDirectPlaneRows(uint32_t rowCount,
                                                uint32_t cols)
    {
        const uint32_t period = DirectPlanePeriod(cols);
        const uint32_t srcStride = period * cols / 8U;
        LocalTensor<float> acc = accBuf.Get<float>();
        LocalTensor<float> first = workBuf.Get<float>();
        LocalTensor<float> second = sumBuf.Get<float>();
        LocalTensor<float> third = valueFloatBuf.Get<float>();
        Duplicate(third, 0.0F, AlignElems32<float>(rowCount));
        bool hasThird = false;
        for (uint32_t phase = 0U; phase < MinU32(period, rowCount); ++phase) {
            const uint32_t misalign = (phase * cols) & 7U;
            const uint32_t alignedBase = phase * cols - misalign;
            const uint32_t tail = cols + misalign - kPlaneReduceChunkElems;
            const uint64_t fullMask = ~0ULL;
            uint64_t firstMask[2] = {fullMask << misalign, 0ULL};
            uint64_t secondMask[2] = {
                tail >= kPlaneReduceChunkElems ? fullMask : (1ULL << tail) - 1ULL,
                0ULL};
            const uint32_t phaseRows = (rowCount - phase + period - 1U) / period;
            for (uint32_t base = 0U; base < phaseRows; base += 255U) {
                const uint32_t repeats = MinU32(255U, phaseRows - base);
                const uint32_t dstOffset = phase + base * period;
                const uint32_t srcOffset = alignedBase + base * period * cols;
                // vcadd FP32 destinations require only 4-byte alignment.
                // Interleaved destinations restore row order without Gather.
                WholeReduceSum<float>(first[dstOffset], acc[srcOffset], firstMask,
                                      repeats, period, 1, srcStride);
                WholeReduceSum<float>(second[dstOffset],
                                      acc[srcOffset + kPlaneReduceChunkElems],
                                      secondMask, repeats, period, 1, srcStride);
                if (tail > kPlaneReduceChunkElems) {
                    uint64_t thirdMask[2] = {
                        (1ULL << (tail - kPlaneReduceChunkElems)) - 1ULL, 0ULL};
                    WholeReduceSum<float>(third[dstOffset],
                                          acc[srcOffset + 2U * kPlaneReduceChunkElems],
                                          thirdMask, repeats, period, 1, srcStride);
                    hasThird = true;
                }
            }
        }
        Add(second, first, second, rowCount);
        if (hasThird) {
            Add(second, second, third, rowCount);
        }
    }

    __aicore__ inline void ProcessDirectPlaneReduce()
    {
        // This narrow-plane route streams each input element once per call.
        // Keep its input reads from allocating L2; other routes are unchanged.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (outputSize != innerSize) {
            uint32_t start = 0U;
            uint32_t end = 0U;
            CalcOutputRange(start, end);
            ProcessDirectPlaneRange(start, end);
            return;
        }
        // For one complete output plane, distribute whole aligned tiles.
        // Each output block still belongs to exactly one core, while adjacent
        // cores start from neighboring input tiles instead of distant shards.
        const uint32_t batchRows = DirectPlaneRows();
        const uint64_t tileStep = static_cast<uint64_t>(GetBlockNum()) * batchRows;
        for (uint64_t tileStart = static_cast<uint64_t>(GetBlockIdx()) * batchRows;
             tileStart < outputSize; tileStart += tileStep) {
            const uint32_t start = static_cast<uint32_t>(tileStart);
            const uint32_t count = MinU32(batchRows, outputSize - start);
            ProcessDirectPlaneRange(start, start + count);
        }
    }

    __aicore__ inline void ProcessDirectPlaneRange(uint32_t start, uint32_t end)
    {
        const uint32_t planeStart = PlaneStart(SuffixReduceStart());
        const uint32_t cols = PlaneSuffixReduceSize();
        const uint32_t prefixSize = reduceSize / cols;
        const uint32_t batchRows = DirectPlaneRows();
        const bool useBlockRows = CanUseDirectPlaneBlockRows(batchRows, cols);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t rowCount = MinU32(batchRows,
                MinU32(end - outputOffset, innerSize - outputOffset % innerSize));
            const uint32_t tightElems = rowCount * cols;
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            LocalTensor<float> acc = accBuf.Get<float>();
            Duplicate(acc, 0.0F,
                      AlignElems32<float>(tightElems) + kPlaneReduceChunkElems);
            uint32_t prefixCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t prefixOffset = 0U;
            CopyNarrowPlaneTight(inputBase, prefixOffset, rowCount, cols);
            for (uint32_t prefix = 0U; prefix < prefixSize; ++prefix) {
                LocalTensor<T> ready = inQueue.DeQue<T>();
                if (prefix + 1U < prefixSize) {
                    AdvancePrefixReduceOffset(planeStart, prefixCoords, prefixOffset);
                    CopyNarrowPlaneTight(inputBase, prefixOffset, rowCount, cols);
                }
                MulAddDst(acc, ready, ready, tightElems);
                if (useBlockRows && prefix + 1U == prefixSize) {
                    ReduceDirectPlaneBlockRows(rowCount, cols, batchRows, ready);
                }
                inQueue.FreeTensor(ready);
            }
            if (!useBlockRows) {
                ReduceDirectPlaneRows(rowCount, cols);
            }
            LocalTensor<T> output = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(output, sumBuf.Get<float>(), rowCount);
            outQueue.EnQue(output);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams params{
                1U, static_cast<uint32_t>(rowCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, params);
            outQueue.FreeTensor(readyOut);
            outputOffset += rowCount;
        }
    }

    __aicore__ inline uint64_t BalancedMiddleReduceBufferBytes(
        uint32_t batchRows, uint32_t rowPitch)
    {
        uint64_t bytes = 0ULL;
        bytes += 2ULL * batchRows * rowPitch * sizeof(T);
        bytes += static_cast<uint64_t>(rowPitch) * sizeof(T);
        bytes += static_cast<uint64_t>(kMiddleReduceComputeRows) * rowPitch *
                 sizeof(float);
        bytes += static_cast<uint64_t>(rowPitch) * sizeof(float);
        return bytes;
    }

    __aicore__ inline uint32_t BalancedMiddleReduceBatchRows()
    {
        const uint32_t rowPitch = AlignElems32<T>(innerSize);
        for (uint32_t rows = kMiddleReduceMaxBatchRows;
             rows >= kMiddleReduceComputeRows; --rows) {
            if (BalancedMiddleReduceBufferBytes(rows, rowPitch) <=
                kAivUbBytes) {
                return rows;
            }
        }
        return 0U;
    }

    __aicore__ inline bool HasContiguousMiddleReduceLayout()
    {
        uint32_t firstReduceDim = kDimNum;
        uint32_t lastReduceDim = kDimNum;
        for (uint32_t dim = 0U; dim < kDimNum; ++dim) {
            if (reduceMask[dim] != 0U) {
                if (firstReduceDim == kDimNum) {
                    firstReduceDim = dim;
                }
                lastReduceDim = dim;
            }
        }
        if (firstReduceDim == kDimNum || lastReduceDim + 1U >= kDimNum) {
            return false;
        }
        for (uint32_t dim = firstReduceDim; dim <= lastReduceDim; ++dim) {
            if (reduceMask[dim] == 0U) {
                return false;
            }
        }

        uint64_t contiguousReduceRows = 1ULL;
        for (uint32_t dim = firstReduceDim; dim <= lastReduceDim; ++dim) {
            contiguousReduceRows *= inDims[dim];
        }
        uint64_t suffixCols = 1ULL;
        for (uint32_t dim = lastReduceDim + 1U; dim < kDimNum; ++dim) {
            suffixCols *= inDims[dim];
        }
        return contiguousReduceRows == reduceSize && suffixCols == innerSize;
    }

    __aicore__ inline bool CanProcessBalancedMiddleReduce()
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            return false;
        } else {
            if (mode != kModeInnerAccum || innerSize < kMiddleReduceMinInnerCols ||
                innerSize > kMiddleReduceMaxInnerCols ||
                reduceSize < kMiddleReduceMinRows || outputSize == 0U ||
                (outputSize % innerSize) != 0U) {
                return false;
            }

            if (!HasContiguousMiddleReduceLayout()) {
                return false;
            }

            const uint32_t blockNum = GetBlockNum();
            const uint32_t outputPlanes = outputSize / innerSize;
            const uint64_t minOutputPlanes =
                static_cast<uint64_t>(blockNum) *
                kMiddleReduceMinFullPlanesPerCore;
            return blockNum != 0U && outputPlanes >= minOutputPlanes &&
                   BalancedMiddleReduceBatchRows() != 0U;
        }
    }

    __aicore__ inline void InitBalancedMiddleReduceBuffers()
    {
        const uint32_t rowPitch = AlignElems32<T>(innerSize);
        const uint32_t batchRows = BalancedMiddleReduceBatchRows();
        pipe->InitBuffer(inQueue, 2U,
                         batchRows * rowPitch * sizeof(T));
        pipe->InitBuffer(outQueue, 1U, rowPitch * sizeof(T));
        pipe->InitBuffer(calcBuf,
                         kMiddleReduceComputeRows * rowPitch * sizeof(float));
        pipe->InitBuffer(accBuf, rowPitch * sizeof(float));
    }

    __aicore__ inline void CopyBalancedMiddleReduceBatch(
        uint32_t inputBase, uint32_t outputCount, uint32_t reduceBase,
        uint32_t reduceCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(outputCount);
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        if (outputCount != innerSize) {
            const uint8_t rightPadding =
                static_cast<uint8_t>(rowPitch - outputCount);
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(reduceCount),
                static_cast<uint32_t>(outputCount * sizeof(T)),
                static_cast<uint32_t>((innerSize - outputCount) * sizeof(T)),
                0U, 0U};
            DataCopyPadExtParams<T> padParams{
                rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
            DataCopyPad(inputLocal,
                        inputGm[static_cast<uint64_t>(inputBase) +
                                static_cast<uint64_t>(reduceBase) * innerSize],
                        copyParams, padParams);
            inQueue.EnQue(inputLocal);
            return;
        }

        if (rowPitch == innerSize) {
            DataCopy(inputLocal,
                     inputGm[static_cast<uint64_t>(inputBase) +
                             static_cast<uint64_t>(reduceBase) * innerSize],
                     reduceCount * rowPitch);
            inQueue.EnQue(inputLocal);
            return;
        }

        // An unaligned source row becomes aligned again after a runtime-derived
        // number of rows.  Copy each row residue as one strided block stream;
        // its harmless over-read lands only in the UB padding column.  The last
        // physical tensor row is copied byte-exactly.
        const uint32_t rowBytes = innerSize * sizeof(T);
        const uint32_t alignmentPeriod = 32U / GcdU32(rowBytes, 32U);
        const uint16_t rowBlocks =
            static_cast<uint16_t>((rowBytes + 31U) / 32U);
        const uint32_t periodSourceBlocks =
            alignmentPeriod * rowBytes / 32U;
        const uint32_t periodDestinationBlocks =
            alignmentPeriod * rowPitch * sizeof(T) / 32U;
        const uint16_t sourceGapBlocks = static_cast<uint16_t>(
            periodSourceBlocks - rowBlocks);
        const uint16_t destinationGapBlocks = static_cast<uint16_t>(
            periodDestinationBlocks - rowBlocks);
        for (uint32_t residue = 0U;
             residue < alignmentPeriod && residue < reduceCount; ++residue) {
            const uint32_t streamRows =
                (reduceCount - 1U - residue) / alignmentPeriod + 1U;
            const uint32_t firstRow = reduceBase + residue;
            const uint32_t lastRow =
                firstRow + (streamRows - 1U) * alignmentPeriod;
            const uint64_t sourceEnd =
                static_cast<uint64_t>(inputBase) +
                static_cast<uint64_t>(lastRow + 1U) * innerSize;
            const bool reachesInputEnd = sourceEnd == inputSize;
            const uint32_t blockRows =
                streamRows - (reachesInputEnd ? 1U : 0U);
            if (blockRows != 0U) {
                DataCopyParams blockParams{
                    static_cast<uint16_t>(blockRows), rowBlocks,
                    sourceGapBlocks, destinationGapBlocks};
                DataCopy(inputLocal[residue * rowPitch],
                         inputGm[static_cast<uint64_t>(inputBase) +
                                 static_cast<uint64_t>(firstRow) * innerSize],
                         blockParams);
            }
            if (reachesInputEnd) {
                const uint32_t destinationRow =
                    residue + (streamRows - 1U) * alignmentPeriod;
                DataCopyExtParams tailParams{
                    1U, rowBytes, 0U, 0U, 0U};
                DataCopyPadExtParams<T> tailPad{
                    true, 0U,
                    static_cast<uint8_t>(rowPitch - innerSize),
                    static_cast<T>(0)};
                DataCopyPad(inputLocal[destinationRow * rowPitch],
                            inputGm[static_cast<uint64_t>(inputBase) +
                                    static_cast<uint64_t>(lastRow) * innerSize],
                            tailParams, tailPad);
            }
        }
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessBalancedMiddleReduceSegment(
        uint32_t outputOffset, uint32_t outputCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(outputCount);
        const uint32_t batchRows = BalancedMiddleReduceBatchRows();
        const uint32_t inputBase = BaseInputOffset(outputOffset);
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, rowPitch);

        uint32_t reduceBase = 0U;
        uint32_t reduceCount = MinU32(batchRows, reduceSize);
        CopyBalancedMiddleReduceBatch(
            inputBase, outputCount, reduceBase, reduceCount);
        while (true) {
            LocalTensor<T> ready = inQueue.DeQue<T>();
            const uint32_t nextReduceBase = reduceBase + reduceCount;
            const bool hasNext = nextReduceBase < reduceSize;
            uint32_t nextReduceCount = 0U;
            if (hasNext) {
                nextReduceCount =
                    MinU32(batchRows, reduceSize - nextReduceBase);
                CopyBalancedMiddleReduceBatch(
                    inputBase, outputCount, nextReduceBase, nextReduceCount);
            }

            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            for (uint32_t computeBase = 0U; computeBase < reduceCount;
                 computeBase += kMiddleReduceComputeRows) {
                const uint32_t computeRows = MinU32(
                    kMiddleReduceComputeRows, reduceCount - computeBase);
                const uint32_t computeElems = computeRows * rowPitch;
                Cast(calcLocal, ready[computeBase * rowPitch],
                     RoundMode::CAST_NONE, computeElems);
                Mul(calcLocal, calcLocal, calcLocal, computeElems);
                AccumulateRows(accLocal, calcLocal, rowPitch, computeRows);
            }
            inQueue.FreeTensor(ready);
            if (!hasNext) {
                break;
            }
            reduceBase = nextReduceBase;
            reduceCount = nextReduceCount;
        }

        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        ConvertPlaneOutput(outputLocal, accLocal, outputCount);
        outQueue.EnQue(outputLocal);
        LocalTensor<T> readyOut = outQueue.DeQue<T>();
        DataCopyExtParams outParams{
            1U, static_cast<uint32_t>(outputCount * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outputOffset], readyOut, outParams);
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline void ProcessBalancedMiddleReduce()
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t outputPlanes = outputSize / innerSize;
        const uint32_t fullPlanesPerCore = outputPlanes / blockNum;
        const uint32_t remainingPlanes = outputPlanes % blockNum;
        const uint32_t fullPlaneBase = blockIdx * fullPlanesPerCore;
        for (uint32_t plane = 0U; plane < fullPlanesPerCore; ++plane) {
            ProcessBalancedMiddleReduceSegment(
                (fullPlaneBase + plane) * innerSize, innerSize);
        }
        if (remainingPlanes == 0U) {
            return;
        }

        // Split the remaining full planes into exactly blockNum balanced
        // segments, so every core receives one similarly sized tail task.  If
        // integer parts would leave a material load skew, use a flat linear
        // split instead; it costs at most one extra segment per core and keeps
        // the predicted per-core element load within one element.
        const uint32_t baseParts = blockNum / remainingPlanes;
        const uint32_t extraParts = blockNum % remainingPlanes;
        const uint32_t maxTailElems =
            (innerSize + baseParts - 1U) / baseParts;
        const uint32_t minPartCount =
            baseParts + (extraParts != 0U ? 1U : 0U);
        const uint32_t minTailElems = innerSize / minPartCount;
        const uint32_t maxCoreElems =
            fullPlanesPerCore * innerSize + maxTailElems;
        const bool useLinearTail =
            16ULL * (maxTailElems - minTailElems) > maxCoreElems;
        if (useLinearTail) {
            const uint64_t tailElems =
                static_cast<uint64_t>(remainingPlanes) * innerSize;
            uint64_t cursor =
                static_cast<uint64_t>(blockIdx) * tailElems / blockNum;
            const uint64_t cursorEnd =
                static_cast<uint64_t>(blockIdx + 1U) * tailElems / blockNum;
            while (cursor < cursorEnd) {
                const uint32_t plane =
                    static_cast<uint32_t>(cursor / innerSize);
                const uint32_t innerOffset =
                    static_cast<uint32_t>(cursor % innerSize);
                const uint32_t segmentElems = MinU32(
                    static_cast<uint32_t>(cursorEnd - cursor),
                    innerSize - innerOffset);
                const uint32_t planeIndex =
                    fullPlanesPerCore * blockNum + plane;
                ProcessBalancedMiddleReduceSegment(
                    planeIndex * innerSize + innerOffset, segmentElems);
                cursor += segmentElems;
            }
            return;
        }

        uint32_t partCursor = blockIdx;
        for (uint32_t plane = 0U; plane < remainingPlanes; ++plane) {
            const uint32_t parts = baseParts + (plane < extraParts ? 1U : 0U);
            if (partCursor < parts) {
                const uint32_t segmentStart =
                    partCursor * innerSize / parts;
                const uint32_t segmentEnd =
                    (partCursor + 1U) * innerSize / parts;
                if (segmentEnd > segmentStart) {
                    const uint32_t planeIndex =
                        fullPlanesPerCore * blockNum + plane;
                    ProcessBalancedMiddleReduceSegment(
                        planeIndex * innerSize + segmentStart,
                        segmentEnd - segmentStart);
                }
                return;
            }
            partCursor -= parts;
        }
    }

    __aicore__ inline uint32_t RowShardMiddlePeriod()
    {
        // Compact FP32 arithmetic needs an 8-element, not BF16 16-element, cycle.
        return 8U / GcdU32(innerSize, 8U);
    }

    __aicore__ inline uint32_t RowShardMiddleSuperRows()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        for (uint32_t cycles = kPhaseMaxSuperRows; cycles > 0U; cycles /= 2U) {
            const uint64_t bytes = 8ULL * cycles * period * pitch +
                4ULL * period * innerSize + 8ULL * period * pitch +
                8ULL * pitch + kPhaseUbGuardBytes;
            if (bytes <= kAivUbBytes) {
                return cycles;
            }
        }
        return 0U;
    }

    __aicore__ inline bool CanProcessRowShardedBf16Middle()
    {
        return CanProcessBalancedMiddleReduce() && GetBlockNum() > 1U &&
               (innerSize & 15U) != 0U && RowShardMiddleSuperRows() != 0U;
    }

    __aicore__ inline uint64_t RowShardWorkspaceElems()
    {
        return 2ULL * (outputSize / innerSize) * AlignElems32<bfloat16_t>(innerSize);
    }

    __aicore__ inline void InitRowShardedBf16MiddleBuffers()
    {
        // Contiguous row shards consume each input once; partial workspace keeps L2.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t batch = period * RowShardMiddleSuperRows();
        pipe->InitBuffer(inQueue, 2U, batch * pitch * sizeof(bfloat16_t));
        pipe->InitBuffer(outQueue, 2U, pitch * sizeof(float));
        pipe->InitBuffer(calcBuf, batch * pitch * sizeof(float));
        pipe->InitBuffer(accBuf, period * innerSize * sizeof(float));
        pipe->InitBuffer(workBuf, period * pitch * sizeof(float));
        pipe->InitBuffer(valueFloatBuf, period * pitch * sizeof(uint32_t));
    }

    __aicore__ inline void BuildRowShardGatherIndex()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        LocalTensor<uint32_t> indices = valueFloatBuf.Get<uint32_t>();
        LocalTensor<int32_t> signedIndices = indices.template ReinterpretCast<int32_t>();
        CreateVecIndex(signedIndices, 0, period * pitch);
        Muls(signedIndices, signedIndices, static_cast<int32_t>(sizeof(float)), period * pitch);
        const uint32_t paddingCols = pitch - innerSize;
        for (uint32_t phase = 1U; phase < period; ++phase) {
            Adds(signedIndices[phase * pitch], signedIndices[phase * pitch],
                 -static_cast<int32_t>(phase * paddingCols * sizeof(float)), innerSize);
        }
        if (paddingCols != 0U) {
            const uint32_t tailBase = (innerSize / 64U) * 64U;
            uint64_t mask[2] = {
                ((1ULL << paddingCols) - 1ULL) << (innerSize - tailBase), 0ULL};
            Duplicate<uint32_t>(indices[tailBase], 0U, mask,
                static_cast<uint8_t>(period), 1U, static_cast<uint8_t>(pitch / 8U));
        }
    }

    __aicore__ inline void CopyRowShardInput(uint32_t flatRow, uint32_t rows)
    {
        LocalTensor<T> input = inQueue.AllocTensor<T>();
        const uint64_t offset = static_cast<uint64_t>(flatRow) * innerSize;
        const uint32_t count = rows * innerSize;
        if ((offset & 15ULL) == 0ULL && (count & 15U) == 0U) {
            DataCopy(input, inputGm[offset], count);
        } else {
            DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padding{false, 0U, 0U, static_cast<T>(0)};
            DataCopyPad(input, inputGm[offset], params, padding);
        }
        inQueue.EnQue(input);
    }

    __aicore__ inline void ProcessRowShardSegment(
        uint32_t flatRow, uint32_t rows, uint32_t plane, uint32_t slot)
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t compact = period * innerSize;
        const uint32_t batch = period * RowShardMiddleSuperRows();
        LocalTensor<float> accumulator = accBuf.Get<float>();
        LocalTensor<float> calc = calcBuf.Get<float>();
        Duplicate(accumulator, 0.0F, compact);
        CopyRowShardInput(flatRow, MinU32(batch, rows));
        for (uint32_t base = 0U; base < rows; base += batch) {
            const uint32_t count = MinU32(batch, rows - base);
            LocalTensor<T> ready = inQueue.DeQue<T>();
            if (base + count < rows) {
                CopyRowShardInput(flatRow + base + count,
                                  MinU32(batch, rows - base - count));
            }
            uint32_t active = 1U;
            while (active * period < count) {
                active *= 2U;
            }
            const uint32_t computeElems = active * compact;
            Cast(calc, ready, RoundMode::CAST_NONE, count * innerSize);
            ZeroPhaseTail(calc, count * innerSize, computeElems);
            Mul(calc, calc, calc, computeElems);
            for (uint32_t pairElems = computeElems / 2U;
                 pairElems >= compact; pairElems /= 2U) {
                Add(calc, calc, calc[pairElems], pairElems);
            }
            Add(accumulator, accumulator, calc, compact);
            inQueue.FreeTensor(ready);
        }
        LocalTensor<float> partial = workBuf.Get<float>();
        Gather(partial, accumulator, valueFloatBuf.Get<uint32_t>(), 0U, period * pitch);
        for (uint32_t count = period * pitch / 2U; count >= pitch; count /= 2U) {
            Add(partial, partial, partial[count], count);
        }
        // The output queue owns the producer-store lifetime. Scratch can then
        // be reused by the next input segment without racing an MTE3 reader.
        LocalTensor<float> producer = outQueue.AllocTensor<float>();
        DataCopy(producer, partial, pitch);
        outQueue.EnQue(producer);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        DataCopy(workspaceGm[(2ULL * plane + slot) * pitch], readyOut, pitch);
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline uint32_t WideDmaRowShardSuperRows()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        for (uint32_t cycles = kPhaseMaxSuperRows; cycles > 0U; cycles /= 2U) {
            // Two BF16 input slots; one two-cycle FP32 calculation stripe;
            // the original accumulator, gather scratch/index and output slots.
            const uint64_t bytes = 4ULL * cycles * period * pitch +
                8ULL * period * pitch + 4ULL * period * innerSize +
                8ULL * period * pitch + 8ULL * pitch + kPhaseUbGuardBytes;
            if (bytes <= kAivUbBytes) {
                return cycles;
            }
        }
        return 0U;
    }

    __aicore__ inline bool CanProcessWideDmaRowShard()
    {
        const uint32_t stripeRows = 2U * RowShardMiddlePeriod();
        // Bound the longer FP32 recurrence for this general short-reduction
        // class; longer reductions retain the original balanced tile tree.
        // A plane may start inside a global stripe. Include that first
        // partial stripe when bounding the accumulator recurrence to 512.
        return reduceSize <= 511U * stripeRows + 1U &&
               WideDmaRowShardSuperRows() > RowShardMiddleSuperRows();
    }

    __aicore__ inline bool CanReuseWideDmaCalc()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t cycles = WideDmaRowShardSuperRows();
        // Keep the original DMA batch and recurrence guard. Only reuse calc
        // when a four-cycle stripe fits without reducing input queue capacity.
        const uint64_t bytes = 4ULL * cycles * period * pitch +
            16ULL * period * pitch + 4ULL * period * innerSize +
            4ULL * period * pitch + 8ULL * pitch + kPhaseUbGuardBytes;
        return bytes <= kAivUbBytes;
    }

    template <bool ReuseCalc>
    __aicore__ inline void InitWideDmaRowShardBuffers()
    {
        // Contiguous row shards consume each input once; partial workspace keeps L2.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t batch = period * WideDmaRowShardSuperRows();
        pipe->InitBuffer(inQueue, 2U, batch * pitch * sizeof(bfloat16_t));
        pipe->InitBuffer(outQueue, 2U, pitch * sizeof(float));
        pipe->InitBuffer(calcBuf, (ReuseCalc ? 4U : 2U) * period * pitch * sizeof(float));
        pipe->InitBuffer(accBuf, period * innerSize * sizeof(float));
        if constexpr (!ReuseCalc) {
            pipe->InitBuffer(workBuf, period * pitch * sizeof(float));
        }
        pipe->InitBuffer(valueFloatBuf, period * pitch * sizeof(uint32_t));
    }

    template <bool ReuseCalc>
    __aicore__ inline void StoreWideDmaRowShardPartial(
        LocalTensor<float> accumulator, uint32_t period, uint32_t pitch,
        uint32_t plane, uint32_t slot)
    {
        // calc's stripe contribution is already consumed by accumulator.
        // UB-to-UB copy below transfers the final reader to outQueue before
        // the next Cast may overwrite this scratch; MTE3 never reads calc.
        LocalTensor<float> partial = ReuseCalc ? calcBuf.Get<float>() : workBuf.Get<float>();
        Gather(partial, accumulator, valueFloatBuf.Get<uint32_t>(), 0U, period * pitch);
        for (uint32_t count = period * pitch / 2U; count >= pitch; count /= 2U) {
            Add(partial, partial, partial[count], count);
        }
        // The output queue owns the producer-store lifetime. Scratch can then
        // be reused by the next input segment without racing an MTE3 reader.
        LocalTensor<float> producer = outQueue.AllocTensor<float>();
        DataCopy(producer, partial, pitch);
        outQueue.EnQue(producer);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        DataCopy(workspaceGm[(2ULL * plane + slot) * pitch], readyOut, pitch);
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline uint32_t WideDmaRowShardOwner(uint32_t flatRow, uint32_t groups)
    {
        const uint32_t group = flatRow / (2U * RowShardMiddlePeriod());
        return static_cast<uint32_t>(
            ((static_cast<uint64_t>(group) + 1ULL) * GetBlockNum() - 1ULL) / groups);
    }

    template <bool ReuseCalc>
    __aicore__ inline void ProcessWideDmaRowShardedMiddle()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t compact = period * innerSize;
        const uint32_t ownerRows = 2U * period;
        const uint32_t stripeRows = (ReuseCalc ? 4U : 2U) * period;
        const uint32_t batch = period * WideDmaRowShardSuperRows();
        const uint32_t totalRows = inputSize / innerSize;
        const uint32_t groups = (totalRows + ownerRows - 1U) / ownerRows;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        uint32_t cursor = static_cast<uint32_t>(
            static_cast<uint64_t>(blockIdx) * groups / blockNum) * ownerRows;
        const uint32_t end = MinU32(totalRows, static_cast<uint32_t>(
            static_cast<uint64_t>(blockIdx + 1U) * groups / blockNum) * ownerRows);
        BuildRowShardGatherIndex();
        if (cursor < end) {
            LocalTensor<float> accumulator = accBuf.Get<float>();
            LocalTensor<float> calc = calcBuf.Get<float>();
            uint32_t plane = cursor / reduceSize;
            uint32_t planeEnd = (plane + 1U) * reduceSize;
            uint32_t slot = cursor % reduceSize == 0U ? 0U : 1U;
            Duplicate(accumulator, 0.0F, compact);
            CopyRowShardInput(cursor, MinU32(batch, end - cursor));
            while (cursor < end) {
                const uint32_t count = MinU32(batch, end - cursor);
                LocalTensor<T> ready = inQueue.DeQue<T>();
                if (cursor + count < end) {
                    CopyRowShardInput(cursor + count,
                                      MinU32(batch, end - cursor - count));
                }
                for (uint32_t stripe = 0U; stripe < count; stripe += stripeRows) {
                    const uint32_t stripeCount = MinU32(stripeRows, count - stripe);
                    const uint32_t active = stripeCount > 2U * period ? 4U :
                        (stripeCount > period ? 2U : 1U);
                    const uint32_t computeElems = active * compact;
                    const uint32_t stripeStart = cursor + stripe;
                    uint32_t lower = 0U;
                    while (lower < stripeCount) {
                        const uint32_t upper = MinU32(stripeCount, planeEnd - stripeStart);
                        // Re-read the same owned UB stripe at a plane boundary.
                        // Clear other planes before squaring: NaN * 0 is unsafe.
                        Cast(calc, ready[stripe * innerSize], RoundMode::CAST_NONE,
                             stripeCount * innerSize);
                        if (lower != 0U) {
                            Duplicate(calc, 0.0F, lower * innerSize);
                        }
                        ZeroPhaseTail(calc, upper * innerSize, computeElems);
                        Mul(calc, calc, calc, computeElems);
                        if constexpr (ReuseCalc) {
                            if (active == 4U) {
                                Add(calc, calc, calc[2U * compact], 2U * compact);
                            }
                        }
                        if (active >= 2U) {
                            Add(calc, calc, calc[compact], compact);
                        }
                        Add(accumulator, accumulator, calc, compact);
                        if (stripeStart + upper == planeEnd || stripeStart + upper == end) {
                            StoreWideDmaRowShardPartial<ReuseCalc>(accumulator, period, pitch, plane, slot);
                            ++plane;
                            planeEnd += reduceSize;
                            slot = 0U;
                            if (stripeStart + upper < end) {
                                Duplicate(accumulator, 0.0F, compact);
                            }
                        }
                        lower = upper;
                    }
                }
                // Both contributions of a boundary stripe are consumed before
                // this physical input slot may be reused by the next DMA.
                inQueue.FreeTensor(ready);
                cursor += count;
            }
        }
        // At least two full planes of work per core implies at most two
        // producers per plane. No uninitialized second slot is ever read.
        CrossCoreSetFlag<0, PIPE_MTE3>(0U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(0U);
        uint32_t outputBlockStart = 0U;
        uint32_t outputBlockEnd = 0U;
        CalcRange((outputSize + 15U) / 16U, outputBlockStart, outputBlockEnd);
        cursor = outputBlockStart * 16U;
        const uint32_t outputEnd = MinU32(outputSize, outputBlockEnd * 16U);
        while (cursor < outputEnd) {
            const uint32_t plane = cursor / innerSize;
            const uint32_t column = cursor % innerSize;
            const uint32_t count = MinU32(innerSize - column, outputEnd - cursor);
            const uint32_t localPitch = AlignElems32<float>(count);
            const bool split = WideDmaRowShardOwner(plane * reduceSize, groups) !=
                WideDmaRowShardOwner((plane + 1U) * reduceSize - 1U, groups);
            LocalTensor<float> input = inQueue.AllocTensor<float>();
            DataCopyExtParams params{static_cast<uint16_t>(split ? 2U : 1U),
                static_cast<uint32_t>(count * sizeof(float)),
                static_cast<uint32_t>((pitch - count) * sizeof(float)), 0U, 0U};
            DataCopyPadExtParams<float> padding{localPitch != count, 0U,
                static_cast<uint8_t>(localPitch - count), 0.0F};
            DataCopyPad(input, workspaceGm[2ULL * plane * pitch + column], params, padding);
            inQueue.EnQue(input);
            LocalTensor<float> ready = inQueue.DeQue<float>();
            LocalTensor<float> result = ReuseCalc ? calcBuf.Get<float>() : workBuf.Get<float>();
            Duplicate(result, 0.0F, AlignElems32<bfloat16_t>(count));
            if (split) {
                Add(result, ready, ready[localPitch], localPitch);
            } else {
                DataCopy(result, ready, localPitch);
            }
            inQueue.FreeTensor(ready);
            LocalTensor<T> output = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(output, result, count);
            outQueue.EnQue(output);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outputParams{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[cursor], readyOut, outputParams);
            outQueue.FreeTensor(readyOut);
            cursor += count;
        }
    }

    __aicore__ inline uint32_t RowShardOwner(uint32_t flatRow, uint32_t groups)
    {
        const uint32_t group = flatRow / RowShardMiddlePeriod();
        return static_cast<uint32_t>(
            ((static_cast<uint64_t>(group) + 1ULL) * GetBlockNum() - 1ULL) / groups);
    }

    __aicore__ inline void ProcessRowShardedBf16Middle()
    {
        const uint32_t period = RowShardMiddlePeriod();
        const uint32_t pitch = AlignElems32<bfloat16_t>(innerSize);
        const uint32_t totalRows = inputSize / innerSize;
        const uint32_t groups = (totalRows + period - 1U) / period;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        uint32_t cursor = static_cast<uint32_t>(
            static_cast<uint64_t>(blockIdx) * groups / blockNum) * period;
        const uint32_t end = MinU32(totalRows, static_cast<uint32_t>(
            static_cast<uint64_t>(blockIdx + 1U) * groups / blockNum) * period);
        BuildRowShardGatherIndex();
        while (cursor < end) {
            const uint32_t plane = cursor / reduceSize;
            const uint32_t reduceOffset = cursor % reduceSize;
            const uint32_t count = MinU32(reduceSize - reduceOffset, end - cursor);
            ProcessRowShardSegment(cursor, count, plane, reduceOffset == 0U ? 0U : 1U);
            cursor += count;
        }
        // At least two full planes of work per core implies at most two
        // producers per plane. No uninitialized second slot is ever read.
        CrossCoreSetFlag<0, PIPE_MTE3>(0U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(0U);
        uint32_t outputBlockStart = 0U;
        uint32_t outputBlockEnd = 0U;
        CalcRange((outputSize + 15U) / 16U, outputBlockStart, outputBlockEnd);
        cursor = outputBlockStart * 16U;
        const uint32_t outputEnd = MinU32(outputSize, outputBlockEnd * 16U);
        while (cursor < outputEnd) {
            const uint32_t plane = cursor / innerSize;
            const uint32_t column = cursor % innerSize;
            const uint32_t count = MinU32(innerSize - column, outputEnd - cursor);
            const uint32_t localPitch = AlignElems32<float>(count);
            const bool split = RowShardOwner(plane * reduceSize, groups) !=
                RowShardOwner((plane + 1U) * reduceSize - 1U, groups);
            LocalTensor<float> input = inQueue.AllocTensor<float>();
            DataCopyExtParams params{static_cast<uint16_t>(split ? 2U : 1U),
                static_cast<uint32_t>(count * sizeof(float)),
                static_cast<uint32_t>((pitch - count) * sizeof(float)), 0U, 0U};
            DataCopyPadExtParams<float> padding{localPitch != count, 0U,
                static_cast<uint8_t>(localPitch - count), 0.0F};
            DataCopyPad(input, workspaceGm[2ULL * plane * pitch + column], params, padding);
            inQueue.EnQue(input);
            LocalTensor<float> ready = inQueue.DeQue<float>();
            LocalTensor<float> result = workBuf.Get<float>();
            Duplicate(result, 0.0F, AlignElems32<bfloat16_t>(count));
            if (split) {
                Add(result, ready, ready[localPitch], localPitch);
            } else {
                DataCopy(result, ready, localPitch);
            }
            inQueue.FreeTensor(ready);
            LocalTensor<T> output = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(output, result, count);
            outQueue.EnQue(output);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outputParams{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[cursor], readyOut, outputParams);
            outQueue.FreeTensor(readyOut);
            cursor += count;
        }
    }

    __aicore__ inline uint32_t PhaseRowPitch()
    {
        return AlignElems32<float>(innerSize);
    }

    __aicore__ inline uint32_t PhasePeriod()
    {
        return 8U / GcdU32(innerSize, 8U);
    }

    __aicore__ inline uint64_t PhaseWorkspaceElems()
    {
        return 1ULL * GetBlockNum() * PhaseRowPitch();
    }

    __aicore__ inline uint64_t PhaseBufferBytes(uint32_t superRows)
    {
        const uint32_t period = PhasePeriod();
        const uint32_t compactElems = period * innerSize;
        const uint32_t paddedElems = period * PhaseRowPitch();
        return 2ULL * superRows * compactElems * sizeof(float) +
               static_cast<uint64_t>(compactElems) * sizeof(float) +
               2ULL * paddedElems * sizeof(float) +
               static_cast<uint64_t>(PhaseRowPitch()) * sizeof(float) +
               kPhaseUbGuardBytes;
    }

    __aicore__ inline uint32_t PhaseSuperRows()
    {
        for (uint32_t rows = kPhaseMaxSuperRows; rows > 0U; rows /= 2U) {
            if (PhaseBufferBytes(rows) <= kAivUbBytes) {
                return rows;
            }
        }
        return 0U;
    }

    __aicore__ inline bool CanProcessPhaseLargeMiddleReduce()
    {
        if constexpr (!std::is_same_v<T, float>) {
            return false;
        } else {
            return mode == kModeInnerAccum && outputSize == innerSize &&
                   reduceSize >= kPhaseMinReduceRows &&
                   innerSize >= kPhaseMinInnerCols &&
                   innerSize <= kPhaseMaxInnerCols &&
                   GetBlockNum() > 1U &&
                   HasContiguousMiddleReduceLayout() &&
                   PhaseSuperRows() != 0U;
        }
    }

    __aicore__ inline void InitPhaseLargeMiddleReduceBuffers()
    {
        // Single-pass streaming input has no in-launch reuse; workspace keeps L2.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t period = PhasePeriod();
        const uint32_t compactElems = period * innerSize;
        const uint32_t paddedElems = period * PhaseRowPitch();
        pipe->InitBuffer(inQueue, 2U,
                         PhaseSuperRows() * compactElems * sizeof(float));
        pipe->InitBuffer(accBuf, compactElems * sizeof(float));
        pipe->InitBuffer(workBuf, paddedElems * sizeof(float));
        pipe->InitBuffer(valueFloatBuf, paddedElems * sizeof(uint32_t));
        pipe->InitBuffer(sumBuf, PhaseRowPitch() * sizeof(float));
    }

    __aicore__ inline void CopyPhaseRows(uint32_t rowBase, uint32_t rowCount)
    {
        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        const uint32_t count = rowCount * innerSize;
        const uint64_t offset = static_cast<uint64_t>(rowBase) * innerSize;
        // Every shard and full tile starts at an alignment-cycle boundary.
        // Compact input needs no row-padding transfers or duplicate GM reads.
        if ((count & 7U) == 0U) {
            DataCopy(inputLocal, inputGm[offset], count);
        } else {
            DataCopyExtParams params{
                1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> padding{false, 0U, 0U, 0.0F};
            DataCopyPad(inputLocal, inputGm[offset], params, padding);
        }
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ZeroPhaseTail(
        LocalTensor<float> values, uint32_t validElems, uint32_t totalElems)
    {
        if (validElems == totalElems) {
            return;
        }
        // Mask the first partial 32-byte block without changing valid values.
        // All vector base addresses remain aligned, including the final tile.
        const uint32_t remainder = validElems & 7U;
        if (remainder != 0U) {
            uint64_t mask[2] = {
                ((1ULL << (8U - remainder)) - 1ULL) << remainder, 0ULL};
            Duplicate<float>(values[validElems - remainder], 0.0F,
                             mask, 1U, 1U, 8U);
        }
        const uint32_t alignedValid = AlignElems32<float>(validElems);
        if (totalElems > alignedValid) {
            Duplicate(values[alignedValid], 0.0F, totalElems - alignedValid);
        }
    }

    __aicore__ inline void BuildPhaseGatherIndex()
    {
        const uint32_t period = PhasePeriod();
        const uint32_t pitch = PhaseRowPitch();
        const uint32_t paddedElems = period * pitch;
        LocalTensor<uint32_t> indexLocal = valueFloatBuf.Get<uint32_t>();
        LocalTensor<int32_t> indexSigned =
            indexLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(indexSigned, 0, paddedElems);
        Muls(indexSigned, indexSigned, static_cast<int32_t>(sizeof(float)),
             paddedElems);
        const uint32_t paddingCols = pitch - innerSize;
        for (uint32_t phase = 1U; phase < period; ++phase) {
            Adds(indexSigned[phase * pitch], indexSigned[phase * pitch],
                 -static_cast<int32_t>(phase * paddingCols * sizeof(float)),
                 innerSize);
        }
        if (paddingCols != 0U) {
            const uint32_t tailBase = (innerSize / 64U) * 64U;
            uint64_t mask[2] = {
                ((1ULL << paddingCols) - 1ULL) << (innerSize - tailBase),
                0ULL};
            Duplicate<uint32_t>(
                indexLocal[tailBase], 0U, mask,
                static_cast<uint8_t>(period), 1U,
                static_cast<uint8_t>(pitch / 8U));
        }
    }

    __aicore__ inline void ProcessPhaseLargeMiddleReduce()
    {
        const uint32_t period = PhasePeriod();
        const uint32_t compactElems = period * innerSize;
        const uint32_t superRows = PhaseSuperRows();
        const uint32_t batchRows = superRows * period;
        const uint32_t rowPitch = PhaseRowPitch();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        // Adjacent full tiles go to adjacent cores; each core then strides by
        // one grid. This preserves single-pass input traffic and tile alignment.
        const uint64_t rowStart = static_cast<uint64_t>(blockIdx) * batchRows;
        const uint64_t rowStep = static_cast<uint64_t>(blockNum) * batchRows;
        // Rotate request-start phases within complete core cohorts. The
        // final incomplete grid keeps the original balanced tile ownership.
        const uint32_t requestPeriod = 512U / GcdU32(batchRows * innerSize * sizeof(float), 512U);
        const uint32_t cohort = (blockNum % requestPeriod == 0U) ? requestPeriod : 1U;
        const uint64_t completeRows = (static_cast<uint64_t>(reduceSize) / rowStep) * rowStep;
        uint32_t phaseLane = blockIdx % cohort;

        LocalTensor<float> phaseAcc = accBuf.Get<float>();
        Duplicate(phaseAcc, 0.0F, compactElems);
        if (rowStart < reduceSize) {
            CopyPhaseRows(static_cast<uint32_t>(rowStart),
                          MinU32(batchRows, reduceSize - static_cast<uint32_t>(rowStart)));
        }
        for (uint64_t row = rowStart; row < reduceSize;) {
            const uint32_t rowCount = MinU32(batchRows, reduceSize - static_cast<uint32_t>(row));
            LocalTensor<float> ready = inQueue.DeQue<float>();
            uint64_t nextRow = row + rowStep;
            if (nextRow < completeRows && cohort > 1U) {
                nextRow += batchRows;
                if (++phaseLane == cohort) {
                    nextRow -= static_cast<uint64_t>(cohort) * batchRows;
                    phaseLane = 0U;
                }
            } else if (row < completeRows && nextRow >= completeRows) {
                nextRow = completeRows + static_cast<uint64_t>(blockIdx) * batchRows;
            }
            if (nextRow < reduceSize) {
                CopyPhaseRows(static_cast<uint32_t>(nextRow),
                              MinU32(batchRows, reduceSize - static_cast<uint32_t>(nextRow)));
            }
            uint32_t activeSuperRows = 1U;
            while (activeSuperRows * period < rowCount) {
                activeSuperRows *= 2U;
            }
            const uint32_t computeElems = activeSuperRows * compactElems;
            ZeroPhaseTail(ready, rowCount * innerSize, computeElems);
            // Explicit FP32 square, then a short tree across complete
            // alignment cycles. Each phase accumulator receives one sum per
            // tile, avoiding a long row-by-row floating-point recurrence.
            Mul(ready, ready, ready, computeElems);
            for (uint32_t count = computeElems / 2U;
                 count >= compactElems; count /= 2U) {
                Add(ready, ready, ready[count], count);
            }
            Add(phaseAcc, phaseAcc, ready, compactElems);
            inQueue.FreeTensor(ready);
            row = nextRow;
        }

        BuildPhaseGatherIndex();
        LocalTensor<float> partial = workBuf.Get<float>();
        LocalTensor<uint32_t> indexLocal = valueFloatBuf.Get<uint32_t>();
        Gather(partial, phaseAcc, indexLocal, 0U, period * rowPitch);
        for (uint32_t count = period * rowPitch / 2U;
             count >= rowPitch; count /= 2U) {
            Add(partial, partial, partial[count], count);
        }

        event_t eventID = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        DataCopy(workspaceGm[static_cast<uint64_t>(blockIdx) * rowPitch],
                 partial, rowPitch);
        CrossCoreSetFlag<0, PIPE_MTE3>(0U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(0U);

        // Assign whole physical 32-byte output blocks; no two cores write
        // the same block. All producers participate in the barrier first.
        uint32_t outputBlockStart = 0U;
        uint32_t outputBlockEnd = 0U;
        CalcRange((outputSize + 7U) / 8U, outputBlockStart, outputBlockEnd);
        const uint32_t outputStart = outputBlockStart * 8U;
        if (outputBlockEnd <= outputBlockStart || outputStart >= outputSize) {
            return;
        }
        const uint32_t outputCount =
            MinU32(outputSize, outputBlockEnd * 8U) - outputStart;
        const uint32_t outputPitch = AlignElems32<float>(outputCount);
        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        DataCopyExtParams params{
            static_cast<uint16_t>(blockNum),
            static_cast<uint32_t>(outputCount * sizeof(float)),
            static_cast<uint32_t>((rowPitch - outputCount) * sizeof(float)),
            0U, 0U};
        DataCopyPadExtParams<float> padding{
            outputPitch != outputCount, 0U,
            static_cast<uint8_t>(outputPitch - outputCount), 0.0F};
        DataCopyPad(inputLocal, workspaceGm[outputStart], params, padding);
        inQueue.EnQue(inputLocal);
        LocalTensor<float> ready = inQueue.DeQue<float>();
        LocalTensor<float> result = sumBuf.Get<float>();
        Duplicate(result, 0.0F, outputPitch);
        for (uint32_t shard = 0U; shard < blockNum; ++shard) {
            Add(result, result, ready[shard * outputPitch], outputPitch);
        }
        inQueue.FreeTensor(ready);
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        DataCopyExtParams outParams{
            1U, static_cast<uint32_t>(outputCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outputStart], result, outParams);
    }

    __aicore__ inline void InitPlaneBuffers()
    {
        const uint32_t inputElems = accTileElems * AlignElems32<T>(kPlaneReduceChunkElems);
        const uint32_t floatElems = accTileElems * AlignElems32<float>(kPlaneReduceChunkElems);
        pipe->InitBuffer(inQueue, 2U, inputElems * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, AlignElems32<T>(accTileElems) * sizeof(T));
        pipe->InitBuffer(calcBuf, floatElems * sizeof(float));
        pipe->InitBuffer(squareBuf, floatElems * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(accTileElems) * sizeof(float));
        pipe->InitBuffer(accBuf, AlignElems32<float>(accTileElems) * sizeof(float));
    }

    __aicore__ inline void InitContiguousBlockBuffers()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint32_t batchElems = accTileElems * rowPitch;
        pipe->InitBuffer(inQueue, 2U, batchElems * sizeof(T));
        pipe->InitBuffer(outQueue, 2U, AlignElems32<T>(accTileElems) * sizeof(T));
        pipe->InitBuffer(calcBuf, batchElems * sizeof(float));
        pipe->InitBuffer(squareBuf, batchElems * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(accTileElems) * sizeof(float));
        pipe->InitBuffer(accBuf, AlignElems32<float>(accTileElems) * sizeof(float));
    }

    __aicore__ inline bool CanProcessLargeSuffixChunked() const
    {
        if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                        std::is_same_v<T, bfloat16_t>)) {
            return false;
        } else {
            // The source-row gap in the 2D DMA is expressed in bytes.  Leave
            // unusually large logical rows on the old tiled fallback rather
            // than risking a narrowed stride field.
            return reduceSize > kSuffixBatchMaxReduce &&
                   reduceSize <= (0xFFFFFFFFU / sizeof(T));
        }
    }

    __aicore__ inline bool CanProcessWideBf16Suffix() const
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            return false;
        } else {
            const uint64_t minOutputRows =
                static_cast<uint64_t>(GetBlockNum()) *
                kWideBf16SuffixMinRowsPerCore;
            return mode == kModeSuffixReduce &&
                   reduceSize >= kWideBf16SuffixMinReduce &&
                   reduceSize <= kWideBf16SuffixMaxReduce &&
                   outputSize >= minOutputRows;
        }
    }

    __aicore__ inline uint64_t FullRowBf16BufferBytes(uint32_t rows)
    {
        const uint32_t fullPitch = AlignElems32<T>(reduceSize);
        const uint32_t stripePitch = AlignElems32<T>(kWideBf16SuffixChunkElems);
        return 2ULL * rows * fullPitch * sizeof(T) +
               1ULL * rows * stripePitch * (sizeof(T) + sizeof(float)) +
               2ULL * AlignElems32<float>(rows) * sizeof(float) +
               2ULL * AlignElems32<T>(rows) * sizeof(T) +
               kFullRowBf16UbGuardBytes;
    }

    __aicore__ inline uint32_t FullRowBf16BatchRows()
    {
        for (uint32_t rows = kFullRowBf16MaxBatchRows;
             rows >= kFullRowBf16MinBatchRows; rows /= 2U) {
            if (FullRowBf16BufferBytes(rows) <= kAivUbBytes) {
                return rows;
            }
        }
        return 0U;
    }

    __aicore__ inline bool CanProcessFullRowBf16Suffix()
    {
        return CanProcessWideBf16Suffix() && FullRowBf16BatchRows() != 0U;
    }

    __aicore__ inline uint32_t CompactSuffixCalcElems(uint32_t groups)
    {
        const uint32_t maxHead = 32U / sizeof(T) - groups;
        return AlignElems32<T>(groups * reduceSize + maxHead);
    }

    __aicore__ inline uint64_t CompactSuffixBufferBytes(uint32_t rows, uint32_t groups)
    {
        const uint32_t phases = (32U / sizeof(T)) / groups;
        const uint32_t partials = groups * (reduceSize / kPlaneReduceChunkElems);
        return 2ULL * AlignElems32<T>(rows * reduceSize) * sizeof(T) +
               2ULL * AlignElems32<T>(rows) * sizeof(T) +
               (CompactSuffixCalcElems(groups) + 8ULL) * sizeof(float) +
               1ULL * phases * groups * kPlaneReduceChunkElems * sizeof(uint32_t) +
               (AlignElems32<float>(partials) + 64ULL) * sizeof(float) +
               AlignElems32<T>(rows) * sizeof(float) +
               1ULL * groups * kPlaneReduceChunkElems * sizeof(float) +
               1ULL * phases * groups * sizeof(uint32_t) +
               2ULL * groups * sizeof(float) + kFullRowBf16UbGuardBytes;
    }

    __aicore__ inline bool CanProcessCompactSuffix(uint32_t rows)
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            return false;
        } else {
            const uint32_t blocks = reduceSize / kPlaneReduceChunkElems;
            return rows != 0U &&
                   (reduceSize - 1U) % kCompactSuffixPeriodElems == 0U &&
                   blocks != 0U && blocks < kPlaneReduceChunkElems &&
                   kCompactSuffixGroupRows * blocks <= 255U &&
                   CompactSuffixBufferBytes(rows, kCompactSuffixGroupRows) <= kAivUbBytes;
        }
    }

    __aicore__ inline bool CanUseBlockSuffixPartials(uint32_t rows, uint32_t groups)
    {
        // Preserve the existing input batch and alignment group. The extra
        // first-stage scratch is independent of all live edge/index buffers.
        const uint64_t scratchBytes =
            1ULL * groups * (reduceSize / kPlaneReduceChunkElems) * 8U * sizeof(float);
        return CompactSuffixBufferBytes(rows, groups) + scratchBytes <= kAivUbBytes;
    }

    __aicore__ inline void InitCompactSuffixBuffers(uint32_t rows, uint32_t groups)
    {
        const uint32_t phases = (32U / sizeof(T)) / groups;
        pipe->InitBuffer(suffixChunkInQueue, 2U,
                         AlignElems32<T>(rows * reduceSize) * sizeof(T));
        pipe->InitBuffer(suffixChunkOutQueue, 2U, AlignElems32<T>(rows) * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf, (CompactSuffixCalcElems(groups) + 8U) * sizeof(float));
        pipe->InitBuffer(suffixChunkSquareBuf,
                         phases * groups * kPlaneReduceChunkElems * sizeof(uint32_t));
        // Extra masked-off storage keeps even a full last partial vector in UB.
        pipe->InitBuffer(suffixChunkSumBuf,
                         (AlignElems32<float>(groups * (reduceSize / kPlaneReduceChunkElems)) +
                          kPlaneReduceChunkElems) * sizeof(float));
        pipe->InitBuffer(suffixChunkAccBuf, AlignElems32<T>(rows) * sizeof(float));
        pipe->InitBuffer(workBuf, groups * kPlaneReduceChunkElems * sizeof(float));
        pipe->InitBuffer(valueFloatBuf, phases * groups * sizeof(uint32_t));
        pipe->InitBuffer(accBuf, groups * sizeof(float));
        pipe->InitBuffer(sumBuf, groups * sizeof(float));
        if (CanUseBlockSuffixPartials(rows, groups)) {
            pipe->InitBuffer(squareBuf,
                groups * (reduceSize / kPlaneReduceChunkElems) * 8U * sizeof(float));
        }
    }

    __aicore__ inline void BuildCompactSuffixIndices(uint32_t batchRows, uint32_t groups)
    {
        const uint32_t phases = (32U / sizeof(T)) / groups;
        const uint32_t blocks = reduceSize / kPlaneReduceChunkElems;
        const uint32_t sentinel = CompactSuffixCalcElems(groups);
        LocalTensor<float> calc = suffixChunkCalcBuf.Get<float>();
        Duplicate(calc[sentinel], 0.0F, 8U);
        // All real output rows are overwritten, not accumulated. Only the
        // output cast's unused final lanes need this once-per-invocation zero.
        Duplicate(suffixChunkAccBuf.Get<float>(), 0.0F,
                  AlignElems32<T>(batchRows));
        LocalTensor<int32_t> edgeIndex = suffixChunkSquareBuf.Get<int32_t>();
        LocalTensor<int32_t> lastIndex = valueFloatBuf.Get<int32_t>();
        // Edge values are not live before the scan. Build one byte-offset
        // ramp in their buffer instead of repeating scalar index seeds for
        // every edge row and each final-element table.
        LocalTensor<int32_t> baseIndex = workBuf.Get<int32_t>();
        CreateVecIndex(baseIndex, 0, kPlaneReduceChunkElems);
        Muls(baseIndex, baseIndex, static_cast<int32_t>(sizeof(float)),
             kPlaneReduceChunkElems);
        for (uint32_t phase = 0U; phase < phases; ++phase) {
            const uint32_t head = phase * groups;
            for (uint32_t row = 0U; row < groups; ++row) {
                LocalTensor<int32_t> index =
                    edgeIndex[(phase * groups + row) * kPlaneReduceChunkElems];
                const uint32_t misalign = head + row;
                if (misalign == 0U) {
                    Duplicate(index, static_cast<int32_t>(sentinel * sizeof(float)),
                              kPlaneReduceChunkElems);
                } else {
                    Adds(index, baseIndex,
                         static_cast<int32_t>((head + row * reduceSize) * sizeof(float)),
                         kPlaneReduceChunkElems);
                    uint64_t mask[2] = {~0ULL << (kPlaneReduceChunkElems - misalign), 0ULL};
                    Adds<int32_t>(index, index,
                        static_cast<int32_t>((blocks - 1U) * kPlaneReduceChunkElems * sizeof(float)),
                        mask, 1U, {1U, 1U, 8U, 8U});
                }
            }
            LocalTensor<int32_t> last = lastIndex[phase * groups];
            Muls(last, baseIndex, static_cast<int32_t>(reduceSize), groups);
            Adds(last, last, static_cast<int32_t>((head + reduceSize - 1U) * sizeof(float)),
                 groups);
        }
        // The scan later reuses workBuf as Gather's destination. Retire all
        // vector reads of its temporary integer ramp before that write phase.
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyCompactSuffix(uint32_t outputBase, uint32_t rows)
    {
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        DataCopyExtParams params{1U, static_cast<uint32_t>(rows * reduceSize * sizeof(T)),
                                 0U, 0U, 0U};
        // One contiguous block, including exact final-tile length. The padded
        // interface also supports an element-aligned, non-32B input base.
        DataCopyPadExtParams<T> padding{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[static_cast<uint64_t>(outputBase) * reduceSize],
                    params, padding);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessCompactSuffix(uint32_t groups)
    {
        const uint32_t batchRows = groups == 32U / sizeof(T) ? groups : FullRowBf16BatchRows();
        const uint32_t blocks = reduceSize / kPlaneReduceChunkElems;
        const bool useBlockPartials = CanUseBlockSuffixPartials(batchRows, groups);
        LocalTensor<float> blockScratch;
        if (useBlockPartials) {
            blockScratch = squareBuf.Get<float>();
        }
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        if (start >= end) {
            return;
        }
        CopyCompactSuffix(start, MinU32(batchRows, end - start));
        BuildCompactSuffixIndices(batchRows, groups);
        LocalTensor<float> calc = suffixChunkCalcBuf.Get<float>();
        LocalTensor<float> partial = suffixChunkSumBuf.Get<float>();
        LocalTensor<float> outputSum = suffixChunkAccBuf.Get<float>();
        LocalTensor<float> edge = workBuf.Get<float>();
        LocalTensor<float> edgeSum = accBuf.Get<float>();
        LocalTensor<float> last = sumBuf.Get<float>();
        LocalTensor<uint32_t> edgeIndex = suffixChunkSquareBuf.Get<uint32_t>();
        LocalTensor<uint32_t> lastIndex = valueFloatBuf.Get<uint32_t>();
        uint64_t interiorMask[2] = {(1ULL << blocks) - 2ULL, 0ULL};
        uint32_t pendingOffset = 0U;
        uint32_t pendingRows = 0U;
        bool hasPendingOutput = false;
        for (uint32_t outputBase = start; outputBase < end; outputBase += batchRows) {
            const uint32_t rows = MinU32(batchRows, end - outputBase);
            LocalTensor<T> input = suffixChunkInQueue.DeQue<T>();
            if (outputBase + rows < end) {
                CopyCompactSuffix(outputBase + rows, MinU32(batchRows, end - outputBase - rows));
            }
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOffset, readyOut, pendingRows);
                suffixChunkOutQueue.FreeTensor(readyOut);
            }
            for (uint32_t groupRow = 0U; groupRow < rows; groupRow += groups) {
                const uint32_t validRows = MinU32(groups, rows - groupRow);
                const uint32_t inputOffset = groupRow * reduceSize;
                const uint32_t head = inputOffset % (32U / sizeof(T));
                const uint32_t castElems = AlignElems32<T>(head + validRows * reduceSize);
                // Align the cast source backwards inside the current tile.
                // Full-block and edge indices exclude all prefix/tail padding.
                Cast(calc, input[inputOffset - head], RoundMode::CAST_NONE, castElems);
                Mul(calc, calc, calc, castElems);
                // Keep each repeat count <=255 and every partial destination
                // 32B aligned: 248 is the largest aligned repeat chunk.
                const uint32_t blockRepeats = validRows * blocks;
                constexpr uint32_t alignedRepeatLimit = (255U / 8U) * 8U;
                if (useBlockPartials) {
                    // Two eight-element trees reproduce the original partial
                    // layout: one FP32 value per complete 64-element block.
                    for (uint32_t blockBase = 0U; blockBase < blockRepeats;
                         blockBase += alignedRepeatLimit) {
                        // Legacy BlockReduceSum takes repeat BEFORE mask,
                        // unlike WholeReduceSum and the newer ReduceDataBlock.
                        BlockReduceSum<float>(blockScratch[blockBase * 8U],
                            calc[blockBase * kPlaneReduceChunkElems],
                            MinU32(alignedRepeatLimit, blockRepeats - blockBase), 64, 1, 1, 8);
                    }
                    // Existing suffix admission makes blocks a multiple of
                    // eight, including every valid-row tail of the group.
                    BlockReduceSum<float>(partial, blockScratch,
                        blockRepeats / 8U, 64, 1, 1, 8);
                } else {
                    for (uint32_t blockBase = 0U; blockBase < blockRepeats;
                         blockBase += alignedRepeatLimit) {
                        WholeReduceSum<float>(partial[blockBase],
                            calc[blockBase * kPlaneReduceChunkElems], 64,
                            MinU32(alignedRepeatLimit, blockRepeats - blockBase), 1, 1, 8);
                    }
                }
                WholeReduceSum<float>(outputSum[groupRow], partial, interiorMask,
                                      validRows, 1, 1, blocks / 8U);
                if (head == 0U) {
                    // Only the first row begins at a whole 64-element block.
                    Add<float>(outputSum[groupRow], outputSum[groupRow], partial,
                               static_cast<uint64_t>(1U), 1U,
                               {1U, 1U, 1U, 8U, 8U, 8U});
                }
                const uint32_t phase = head / groups;
                Gather(edge, calc, edgeIndex[phase * groups * kPlaneReduceChunkElems],
                       0U, validRows * kPlaneReduceChunkElems);
                WholeReduceSum<float>(edgeSum, edge, 64, validRows, 1, 1, 8);
                Gather(last, calc, lastIndex[phase * groups], 0U, validRows);
                Add(outputSum[groupRow], outputSum[groupRow], edgeSum, validRows);
                Add(outputSum[groupRow], outputSum[groupRow], last, validRows);
            }
            suffixChunkInQueue.FreeTensor(input);
            LocalTensor<T> output = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(output, outputSum, rows);
            suffixChunkOutQueue.EnQue(output);
            pendingOffset = outputBase;
            pendingRows = rows;
            hasPendingOutput = true;
        }
        if (hasPendingOutput) {
            LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOffset, readyOut, pendingRows);
            suffixChunkOutQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline uint32_t InitFullRowBf16SuffixBuffers()
    {
        // Complete suffix rows are streamed once; stripe reuse remains inside UB.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t rows = FullRowBf16BatchRows();
        if (CanProcessCompactSuffix(rows)) {
            // A full source-alignment group needs only one boundary table.
            // Couple its input batch to the group and admit it by UB capacity.
            const uint32_t alignedGroup = 32U / sizeof(T);
            if (CompactSuffixBufferBytes(alignedGroup, alignedGroup) <= kAivUbBytes) {
                InitCompactSuffixBuffers(alignedGroup, alignedGroup);
                return alignedGroup;
            }
            InitCompactSuffixBuffers(rows, kCompactSuffixGroupRows);
            return kCompactSuffixGroupRows;
        }
        const uint32_t stripeElems = rows * AlignElems32<T>(kWideBf16SuffixChunkElems);
        pipe->InitBuffer(suffixChunkInQueue, 2U,
                         rows * AlignElems32<T>(reduceSize) * sizeof(T));
        pipe->InitBuffer(suffixChunkOutQueue, 2U, AlignElems32<T>(rows) * sizeof(T));
        // UB-to-UB stripe staging and FP32 arithmetic have separate storage.
        // Full input slots remain live until every stripe has been consumed.
        pipe->InitBuffer(suffixChunkSquareBuf, stripeElems * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf, stripeElems * sizeof(float));
        pipe->InitBuffer(suffixChunkSumBuf, AlignElems32<float>(rows) * sizeof(float));
        pipe->InitBuffer(suffixChunkAccBuf, AlignElems32<float>(rows) * sizeof(float));
        return 0U;
    }

    __aicore__ inline void CopyFullRowBf16Suffix(uint32_t outputBase, uint32_t rows)
    {
        const uint32_t pitch = AlignElems32<T>(reduceSize);
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        // One padded 2-D command reads complete rows, with no GM over-read.
        // Padding belongs to each UB row, never to the reduction's valid mask.
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(reduceSize * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padding{
            pitch != reduceSize, 0U, static_cast<uint8_t>(pitch - reduceSize),
            static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[static_cast<uint64_t>(outputBase) * reduceSize],
                    params, padding);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessFullRowBf16Suffix(uint32_t compactGroups)
    {
        if (compactGroups != 0U) {
            ProcessCompactSuffix(compactGroups);
            return;
        }
        const uint32_t batchRows = FullRowBf16BatchRows();
        const uint32_t fullPitch = AlignElems32<T>(reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        if (start >= end) {
            return;
        }
        CopyFullRowBf16Suffix(start, MinU32(batchRows, end - start));
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;
        for (uint32_t outputBase = start; outputBase < end; outputBase += batchRows) {
            const uint32_t rows = MinU32(batchRows, end - outputBase);
            LocalTensor<T> fullRows = suffixChunkInQueue.DeQue<T>();
            if (outputBase + rows < end) {
                CopyFullRowBf16Suffix(outputBase + rows,
                                     MinU32(batchRows, end - outputBase - rows));
            }
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
                suffixChunkOutQueue.FreeTensor(readyOut);
            }
            LocalTensor<T> stripe = suffixChunkSquareBuf.Get<T>();
            LocalTensor<float> calc = suffixChunkCalcBuf.Get<float>();
            LocalTensor<float> sums = suffixChunkSumBuf.Get<float>();
            LocalTensor<float> acc = suffixChunkAccBuf.Get<float>();
            Duplicate(acc, 0.0F, AlignElems32<float>(rows));
            for (uint32_t reduceBase = 0U; reduceBase < reduceSize;
                 reduceBase += kWideBf16SuffixChunkElems) {
                const uint32_t count = MinU32(kWideBf16SuffixChunkElems,
                                              reduceSize - reduceBase);
                const uint32_t pitch = AlignElems32<T>(count);
                // The stripe offset, both row pitches and both UB addresses
                // are 32B aligned. The final stripe reads only in-row padding.
                DataCopyParams localCopy{
                    static_cast<uint16_t>(rows),
                    static_cast<uint16_t>(pitch * sizeof(T) / 32U),
                    static_cast<uint16_t>((fullPitch - pitch) * sizeof(T) / 32U),
                    0U};
                DataCopy(stripe, fullRows[reduceBase], localCopy);
                Cast(calc, stripe, RoundMode::CAST_NONE, rows * pitch);
                Mul(calc, calc, calc, rows * pitch);
                const uint8_t rowStrideBlocks =
                    static_cast<uint8_t>(pitch * sizeof(float) / 32U);
                // Preserve the baseline's 64-lane stripe folds and ordered
                // stripe accumulation; only the GM/UB staging geometry changes.
                if (count > kPlaneReduceChunkElems) {
                    Add<float>(calc, calc, calc[kPlaneReduceChunkElems],
                        static_cast<uint64_t>(MinU32(kPlaneReduceChunkElems,
                                                     count - kPlaneReduceChunkElems)),
                        static_cast<uint8_t>(rows),
                        {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks, rowStrideBlocks});
                }
                if (count > 2U * kPlaneReduceChunkElems) {
                    Add<float>(calc, calc, calc[2U * kPlaneReduceChunkElems],
                        static_cast<uint64_t>(MinU32(kPlaneReduceChunkElems,
                                                     count - 2U * kPlaneReduceChunkElems)),
                        static_cast<uint8_t>(rows),
                        {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks, rowStrideBlocks});
                }
                if (count > 3U * kPlaneReduceChunkElems) {
                    Add<float>(calc, calc, calc[3U * kPlaneReduceChunkElems],
                        static_cast<uint64_t>(count - 3U * kPlaneReduceChunkElems),
                        static_cast<uint8_t>(rows),
                        {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks, rowStrideBlocks});
                }
                WholeReduceSum<float>(sums, calc,
                    static_cast<int32_t>(MinU32(kPlaneReduceChunkElems, count)),
                    static_cast<int32_t>(rows), 1, 1,
                    static_cast<int32_t>(rowStrideBlocks));
                Add(acc, sums, acc, rows);
            }
            suffixChunkInQueue.FreeTensor(fullRows);
            LocalTensor<T> outputLocal = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, acc, rows);
            suffixChunkOutQueue.EnQue(outputLocal);
            pendingOutputOffset = outputBase;
            pendingOutputRows = rows;
            hasPendingOutput = true;
        }
        if (hasPendingOutput) {
            LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
            suffixChunkOutQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void InitLargeSuffixChunkedBuffers()
    {
        const bool useWideBf16 = CanProcessWideBf16Suffix();
        const uint32_t chunkElems = useWideBf16
            ? kWideBf16SuffixChunkElems : kLargeSuffixChunkElems;
        const uint32_t batchRows = useWideBf16
            ? kWideBf16SuffixBatchRows : kLargeSuffixBatchRows;
        const uint32_t inputRowPitch = AlignElems32<T>(chunkElems);
        const uint32_t inputElems = batchRows * inputRowPitch;
        const uint32_t floatRows = useWideBf16
            ? kWideBf16SuffixCalcRows : batchRows;
        const uint32_t floatElems = floatRows * inputRowPitch;
        const uint32_t outputElems = AlignElems32<T>(batchRows);
        const uint32_t sumElems = AlignElems32<float>(batchRows);

        // The two VECIN slots retain the current stripe while MTE2 fetches
        // the next one.  VECOUT is also double-buffered so MTE3 can retire a
        // previous output-row batch while the next batch is reduced.
        pipe->InitBuffer(suffixChunkInQueue, 2U, inputElems * sizeof(T));
        pipe->InitBuffer(suffixChunkOutQueue, 2U, outputElems * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf, floatElems * sizeof(float));
        if (!useWideBf16) {
            pipe->InitBuffer(suffixChunkSquareBuf, floatElems * sizeof(float));
        }
        pipe->InitBuffer(suffixChunkSumBuf, sumElems * sizeof(float));
        pipe->InitBuffer(suffixChunkAccBuf, sumElems * sizeof(float));
    }

    __aicore__ inline void CopyLargeSuffixChunk(uint32_t outputBase, uint32_t rowCount,
                                                uint32_t reduceBase, uint32_t reduceCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceCount);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceCount);
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        if (CanProcessWideBf16Suffix()) {
            const uint32_t sourceRowBytes = reduceSize * sizeof(T);
            const uint32_t residueCount =
                32U / GcdU32(sourceRowBytes, 32U);
            const uint16_t rowBlocks = static_cast<uint16_t>(
                rowPitch * sizeof(T) / 32U);
            const uint32_t sourcePeriodBlocks =
                residueCount * sourceRowBytes / 32U;
            const uint16_t sourceGapBlocks = static_cast<uint16_t>(
                sourcePeriodBlocks - rowBlocks);
            const uint16_t destinationGapBlocks = static_cast<uint16_t>(
                residueCount * rowBlocks - rowBlocks);
            for (uint32_t residue = 0U;
                 residue < residueCount && residue < rowCount; ++residue) {
                const uint32_t streamRows =
                    (rowCount - 1U - residue) / residueCount + 1U;
                const uint32_t firstLocalRow = residue;
                const uint32_t lastLocalRow =
                    residue + (streamRows - 1U) * residueCount;
                const uint32_t firstGlobalRow = outputBase + firstLocalRow;
                const uint32_t lastGlobalRow = outputBase + lastLocalRow;
                const uint64_t lastCopyEnd =
                    static_cast<uint64_t>(lastGlobalRow) * reduceSize +
                    reduceBase + rowPitch;
                const bool needsSafeTail = lastCopyEnd > inputSize;
                const uint32_t blockRows =
                    streamRows - (needsSafeTail ? 1U : 0U);
                if (blockRows != 0U) {
                    DataCopyParams blockParams{
                        static_cast<uint16_t>(blockRows), rowBlocks,
                        sourceGapBlocks, destinationGapBlocks};
                    DataCopy(inputLocal[firstLocalRow * rowPitch],
                             inputGm[firstGlobalRow * reduceSize + reduceBase],
                             blockParams);
                }
                if (needsSafeTail) {
                    DataCopyExtParams tailParams{
                        1U, static_cast<uint32_t>(reduceCount * sizeof(T)),
                        0U, 0U, 0U};
                    DataCopyPadExtParams<T> tailPad{
                        rightPadding != 0U, 0U, rightPadding,
                        static_cast<T>(0)};
                    DataCopyPad(
                        inputLocal[lastLocalRow * rowPitch],
                        inputGm[lastGlobalRow * reduceSize + reduceBase],
                        tailParams, tailPad);
                }
            }
            suffixChunkInQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(reduceCount * sizeof(T)),
            static_cast<uint32_t>((reduceSize - reduceCount) * sizeof(T)), 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[outputBase * reduceSize + reduceBase],
                    copyParams, padParams);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessLargeSuffixChunked()
    {
        const bool useWideBf16 = CanProcessWideBf16Suffix();
        const uint32_t chunkElems = useWideBf16
            ? kWideBf16SuffixChunkElems : kLargeSuffixChunkElems;
        const uint32_t batchRows = useWideBf16
            ? kWideBf16SuffixBatchRows : kLargeSuffixBatchRows;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;

        for (uint32_t outputBase = start; outputBase < end;
             outputBase += batchRows) {
            const uint32_t rowCount = MinU32(batchRows, end - outputBase);
            uint32_t reduceBase = 0U;
            uint32_t reduceCount = MinU32(chunkElems, reduceSize);
            CopyLargeSuffixChunk(outputBase, rowCount, reduceBase, reduceCount);

            // MTE3 starts before this batch's first DeQue, so it can overlap
            // both the first MTE2 stripe and all following vector reductions.
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
                suffixChunkOutQueue.FreeTensor(readyOut);
            }

            LocalTensor<float> accLocal = suffixChunkAccBuf.Get<float>();
            LocalTensor<float> sumsLocal = suffixChunkSumBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(rowCount));

            while (true) {
                LocalTensor<T> ready = suffixChunkInQueue.DeQue<T>();
                const uint32_t nextReduceBase = reduceBase + reduceCount;
                const bool hasNext = nextReduceBase < reduceSize;
                uint32_t nextReduceCount = 0U;
                if (hasNext) {
                    nextReduceCount = MinU32(chunkElems,
                                              reduceSize - nextReduceBase);
                    CopyLargeSuffixChunk(outputBase, rowCount, nextReduceBase,
                                         nextReduceCount);
                }

                const uint32_t rowPitch = AlignElems32<T>(reduceCount);
                const uint32_t batchElems = rowCount * rowPitch;
                LocalTensor<float> calcLocal = suffixChunkCalcBuf.Get<float>();
                if (useWideBf16) {
                    const uint8_t rowStrideBlocks = static_cast<uint8_t>(
                        rowPitch * sizeof(float) / 32U);
                    for (uint32_t calcBase = 0U; calcBase < rowCount;
                         calcBase += kWideBf16SuffixCalcRows) {
                        const uint32_t calcRows = MinU32(
                            kWideBf16SuffixCalcRows, rowCount - calcBase);
                        const uint32_t calcElems = calcRows * rowPitch;
                        Cast(calcLocal, ready[calcBase * rowPitch],
                             RoundMode::CAST_NONE, calcElems);
                        Mul(calcLocal, calcLocal, calcLocal, calcElems);
                        if (reduceCount > kPlaneReduceChunkElems) {
                            Add<float>(
                                calcLocal, calcLocal,
                                calcLocal[kPlaneReduceChunkElems],
                                static_cast<uint64_t>(MinU32(
                                    kPlaneReduceChunkElems,
                                    reduceCount - kPlaneReduceChunkElems)),
                                static_cast<uint8_t>(calcRows),
                                {1U, 1U, 1U, rowStrideBlocks,
                                 rowStrideBlocks, rowStrideBlocks});
                        }
                        if (reduceCount > 2U * kPlaneReduceChunkElems) {
                            Add<float>(
                                calcLocal, calcLocal,
                                calcLocal[2U * kPlaneReduceChunkElems],
                                static_cast<uint64_t>(MinU32(
                                    kPlaneReduceChunkElems,
                                    reduceCount - 2U * kPlaneReduceChunkElems)),
                                static_cast<uint8_t>(calcRows),
                                {1U, 1U, 1U, rowStrideBlocks,
                                 rowStrideBlocks, rowStrideBlocks});
                        }
                        if (reduceCount > 3U * kPlaneReduceChunkElems) {
                            Add<float>(
                                calcLocal, calcLocal,
                                calcLocal[3U * kPlaneReduceChunkElems],
                                static_cast<uint64_t>(
                                    reduceCount - 3U * kPlaneReduceChunkElems),
                                static_cast<uint8_t>(calcRows),
                                {1U, 1U, 1U, rowStrideBlocks,
                                 rowStrideBlocks, rowStrideBlocks});
                        }
                        WholeReduceSum<float>(
                            sumsLocal[calcBase], calcLocal,
                            static_cast<int32_t>(MinU32(
                                kPlaneReduceChunkElems, reduceCount)),
                            static_cast<int32_t>(calcRows), 1, 1,
                            static_cast<int32_t>(rowStrideBlocks));
                    }
                } else {
                    LocalTensor<float> squareLocal =
                        suffixChunkSquareBuf.Get<float>();
                    SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
                    WholeReduceSum<float>(
                        sumsLocal, squareLocal,
                        static_cast<int32_t>(reduceCount),
                        static_cast<int32_t>(rowCount), 1, 1,
                        static_cast<int32_t>(
                            rowPitch * sizeof(float) / 32U));
                }
                Add(accLocal, sumsLocal, accLocal, rowCount);
                suffixChunkInQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                reduceBase = nextReduceBase;
                reduceCount = nextReduceCount;
            }

            LocalTensor<T> outputLocal = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            suffixChunkOutQueue.EnQue(outputLocal);
            pendingOutputOffset = outputBase;
            pendingOutputRows = rowCount;
            hasPendingOutput = true;
        }

        if (hasPendingOutput) {
            LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
            suffixChunkOutQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline uint32_t BaseInputOffset(uint32_t outputOffset)
    {
        uint32_t rem = outputOffset;
        uint32_t inputOffset = 0U;
        for (uint32_t i = 0U; i < kDimNum; ++i) {
            uint32_t coord = rem / outStrides[i];
            rem -= coord * outStrides[i];
            if (reduceMask[i] == 0U) {
                inputOffset += coord * inStrides[i];
            }
        }
        return inputOffset;
    }

    __aicore__ inline uint32_t LastReduceDim()
    {
        for (int32_t dim = static_cast<int32_t>(kDimNum) - 1; dim >= 0; --dim) {
            if (reduceMask[static_cast<uint32_t>(dim)] != 0U) {
                return static_cast<uint32_t>(dim);
            }
        }
        return kDimNum;
    }

    __aicore__ inline uint32_t SuffixReduceStart()
    {
        uint32_t start = kDimNum - 1U;
        while (start > 0U && reduceMask[start - 1U] != 0U) {
            --start;
        }
        return start;
    }

    __aicore__ inline uint32_t PlaneStart(uint32_t suffixReduceStart)
    {
        uint32_t start = suffixReduceStart;
        while (start > 0U && reduceMask[start - 1U] == 0U) {
            --start;
        }
        return start;
    }

    __aicore__ inline void AdvanceOuterReduceOffset(uint32_t lastReduceDim,
                                                    uint32_t (&coords)[kDimNum], uint32_t& inputOffset)
    {
        for (int32_t dim = static_cast<int32_t>(lastReduceDim) - 1; dim >= 0; --dim) {
            const uint32_t dimIdx = static_cast<uint32_t>(dim);
            if (reduceMask[dimIdx] == 0U) {
                continue;
            }
            const uint32_t stride = inStrides[dimIdx];
            ++coords[dimIdx];
            inputOffset += stride;
            if (coords[dimIdx] < inDims[dimIdx]) {
                return;
            }
            coords[dimIdx] = 0U;
            inputOffset -= inDims[dimIdx] * stride;
        }
    }

    __aicore__ inline void AdvancePrefixReduceOffset(uint32_t planeStart,
                                                     uint32_t (&coords)[kDimNum],
                                                     uint32_t& inputOffset)
    {
        for (int32_t dim = static_cast<int32_t>(planeStart) - 1; dim >= 0; --dim) {
            const uint32_t dimIdx = static_cast<uint32_t>(dim);
            if (reduceMask[dimIdx] == 0U) {
                continue;
            }
            const uint32_t stride = inStrides[dimIdx];
            ++coords[dimIdx];
            inputOffset += stride;
            if (coords[dimIdx] < inDims[dimIdx]) {
                return;
            }
            coords[dimIdx] = 0U;
            inputOffset -= inDims[dimIdx] * stride;
        }
    }

    __aicore__ inline void SquarePlaneBatch(LocalTensor<float> squareLocal,
                                            LocalTensor<T> inputLocal,
                                            LocalTensor<float> calcLocal,
                                            uint32_t count)
    {
        Cast(calcLocal, inputLocal, RoundMode::CAST_NONE, count);
        Mul(squareLocal, calcLocal, calcLocal, count);
    }

    __aicore__ inline void ConvertPlaneOutput(LocalTensor<T> outputLocal,
                                              LocalTensor<float> accLocal,
                                              uint32_t count)
    {
        Cast(outputLocal, accLocal, RoundMode::CAST_RINT, count);
    }

    __aicore__ inline void AccumulateRows(LocalTensor<float> accLocal, LocalTensor<float> rowsLocal,
                                          uint32_t rowPitch, uint32_t rowCount)
    {
        uint32_t activeRows = rowCount;
        while (activeRows > 1U) {
            const uint32_t pairedRows = activeRows / 2U;
            Add(rowsLocal, rowsLocal[pairedRows * rowPitch], rowsLocal, pairedRows * rowPitch);
            if ((activeRows & 1U) != 0U) {
                Add(rowsLocal, rowsLocal[(activeRows - 1U) * rowPitch], rowsLocal, rowPitch);
            }
            activeRows = pairedRows;
        }
        if (rowCount != 0U) {
            Add(accLocal, rowsLocal, accLocal, rowPitch);
        }
    }

    __aicore__ inline void AccumulateSquaredRows(LocalTensor<float> accLocal,
                                                  LocalTensor<T> rowsLocal,
                                                  uint32_t rowPitch, uint32_t rowCount)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t batchElems = rowCount * rowPitch;
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            LocalTensor<float> squareLocal = squareBuf.Get<float>();
            SquarePlaneBatch(squareLocal, rowsLocal, calcLocal, batchElems);
            AccumulateRows(accLocal, squareLocal, rowPitch, rowCount);
            return;
        }
        constexpr uint32_t kFusedCols = 64U;
        const uint32_t srcRowStrideBlocks = rowPitch * sizeof(T) / 32U;
        for (uint32_t col = 0U; col < rowPitch; col += kFusedCols) {
            const uint32_t colCount = MinU32(kFusedCols, rowPitch - col);
            // A zero destination repeat stride deliberately reuses this output
            // slice for every input row: acc += row * row.  Rank4 tiling keeps
            // both source rows and accumulator slices 32-byte aligned.
            MulAddDst(accLocal[col], rowsLocal[col], rowsLocal[col],
                      static_cast<uint64_t>(colCount), static_cast<uint8_t>(rowCount),
                      {1U, 1U, 1U, 0U,
                       static_cast<uint8_t>(srcRowStrideBlocks),
                       static_cast<uint8_t>(srcRowStrideBlocks)});
        }
    }

    __aicore__ inline void ProcessSmallSuffixRows()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixBatchRows) {
            const uint32_t outCount = MinU32(kSuffixBatchRows, end - outBase);
            const uint32_t batchElems = outCount * rowPitch;
            LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(outCount),
                                         static_cast<uint32_t>(reduceSize * sizeof(T)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm[outBase * reduceSize], copyParams, padParams);
            inQueue.EnQue(inputLocal);

            LocalTensor<T> ready = inQueue.DeQue<T>();
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            LocalTensor<float> squareLocal = workBuf.Get<float>();
            LocalTensor<float> sumsLocal = valueFloatBuf.Get<float>();
            Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
            Mul(squareLocal, calcLocal, calcLocal, batchElems);
            WholeReduceSum<float>(sumsLocal, squareLocal, static_cast<int32_t>(reduceSize),
                                  static_cast<int32_t>(outCount), 1, 1,
                                  static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
            inQueue.FreeTensor(ready);

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Cast(outputLocal, sumsLocal, RoundMode::CAST_NONE, AlignElems32<T>(outCount));
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessSuffixReduce()
    {
        if (reduceSize != 0U && reduceSize <= kSuffixBatchMaxReduce) {
            ProcessSmallSuffixRows();
            return;
        }
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
            const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
            LocalTensor<float> scalarFloat = valueFloatBuf.Get<float>();
            for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
                const uint32_t outIdx = outBase + localOut;
                float acc = 0.0F;
                const uint32_t base = outIdx * reduceSize;
                for (uint32_t r = 0U; r < reduceSize; r += tileElems) {
                    const uint32_t count = MinU32(tileElems, reduceSize - r);
                    LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
                    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
                    DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
                    DataCopyPad(inputLocal, inputGm[base + r], copyParams, padParams);
                    inQueue.EnQue(inputLocal);

                    LocalTensor<T> ready = inQueue.DeQue<T>();
                    LocalTensor<float> calcLocal = calcBuf.Get<float>();
                    LocalTensor<float> workLocal = workBuf.Get<float>();
                    LocalTensor<float> sumLocal = sumBuf.Get<float>();
                    Cast(calcLocal, ready, RoundMode::CAST_NONE, AlignElems32<T>(count));
                    Mul(workLocal, calcLocal, calcLocal, count);
                    ReduceSum(sumLocal, workLocal, calcLocal, count);
                    acc += sumLocal.GetValue(0U);
                    inQueue.FreeTensor(ready);
                }
                scalarFloat.SetValue(localOut, acc);
            }
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Cast(outputLocal, scalarFloat, RoundMode::CAST_NONE, AlignElems32<T>(outCount));
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessInnerAccum()
    {
        const uint32_t lastReduceDim = LastReduceDim();
        if (lastReduceDim >= kDimNum) {
            return;
        }
        const uint32_t innerReduceSize = inDims[lastReduceDim];
        const uint32_t outerReduceSize = innerReduceSize == 0U ? 0U : reduceSize / innerReduceSize;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outputOffset = task * accTileElems;
            if (outputOffset >= outputSize) {
                continue;
            }
            const uint32_t outputCount = MinU32(accTileElems, outputSize - outputOffset);
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            uint32_t localOutputOffset = 0U;
            while (localOutputOffset < outputCount) {
                const uint32_t globalOutputOffset = outputOffset + localOutputOffset;
                const uint32_t innerOffset = globalOutputOffset % innerSize;
                const uint32_t count = MinU32(outputCount - localOutputOffset, innerSize - innerOffset);
                const uint32_t rowPitch = AlignElems32<T>(count);
                const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
                LocalTensor<float> accLocal = accBuf.Get<float>();
                Duplicate(accLocal, 0.0F, rowPitch);
                const uint32_t inputBase = BaseInputOffset(globalOutputOffset);
                uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
                uint32_t outerReduceOffset = 0U;
                for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                    uint32_t reduceBase = 0U;
                    uint32_t reduceCount =
                        MinU32(kInnerPipelinedBatchRows, innerReduceSize);
                    LocalTensor<T> firstInput = inQueue.AllocTensor<T>();
                    DataCopyExtParams firstCopyParams{
                        static_cast<uint16_t>(reduceCount),
                        static_cast<uint32_t>(count * sizeof(T)),
                        static_cast<uint32_t>((innerSize - count) * sizeof(T)), 0U, 0U};
                    DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                                      static_cast<T>(0)};
                    DataCopyPad(firstInput, inputGm[inputBase + outerReduceOffset],
                                firstCopyParams, padParams);
                    inQueue.EnQue(firstInput);

                    while (true) {
                        LocalTensor<T> ready = inQueue.DeQue<T>();
                        const uint32_t nextReduceBase = reduceBase + reduceCount;
                        uint32_t nextReduceCount = 0U;
                        if (nextReduceBase < innerReduceSize) {
                            nextReduceCount = MinU32(kInnerPipelinedBatchRows,
                                                     innerReduceSize - nextReduceBase);
                            LocalTensor<T> nextInput = inQueue.AllocTensor<T>();
                            DataCopyExtParams nextCopyParams{
                                static_cast<uint16_t>(nextReduceCount),
                                static_cast<uint32_t>(count * sizeof(T)),
                                static_cast<uint32_t>((innerSize - count) * sizeof(T)),
                                0U, 0U};
                            DataCopyPad(nextInput,
                                        inputGm[inputBase + outerReduceOffset +
                                                nextReduceBase * innerSize],
                                        nextCopyParams, padParams);
                            inQueue.EnQue(nextInput);
                        }

                        AccumulateSquaredRows(accLocal, ready, rowPitch, reduceCount);
                        inQueue.FreeTensor(ready);

                        if (nextReduceBase >= innerReduceSize) {
                            break;
                        }
                        reduceBase = nextReduceBase;
                        reduceCount = nextReduceCount;
                    }
                    AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    outputLocal.SetValue(localOutputOffset + i, static_cast<T>(accLocal.GetValue(i)));
                }
                localOutputOffset += count;
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outputCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void CopyRank4Reduce02Input(uint32_t dimB, uint32_t dimC,
                                                   uint32_t dimD, uint32_t b,
                                                   uint32_t dBase, uint32_t count,
                                                   uint8_t rightPadding,
                                                   uint32_t a, uint32_t cBase,
                                                   uint32_t cCount)
    {
        const uint32_t gmOffset = ((a * dimB + b) * dimC + cBase) * dimD + dBase;
        LocalTensor<T> inputLocal = rank4InQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(cCount),
                                     static_cast<uint32_t>(count * sizeof(T)),
                                     static_cast<uint32_t>((dimD - count) * sizeof(T)),
                                     0U, 0U};
        DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                          static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        rank4InQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyRank4Reduce02BatchBInput(uint32_t dimB,
                                                        uint32_t dimC,
                                                        uint32_t dimD,
                                                        uint32_t a,
                                                        uint32_t bBase,
                                                        uint32_t bCount)
    {
        const uint32_t gmOffset = (a * dimB + bBase) * dimC * dimD;
        LocalTensor<T> inputLocal = rank4InQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(bCount * dimC),
            static_cast<uint32_t>(dimD * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        rank4InQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessRank4Reduce02BatchB()
    {
        const uint32_t dimA = inDims[1];
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        const uint32_t rowPitchT = AlignElems32<T>(dimD);
        const uint32_t rowPitchF = AlignElems32<float>(dimD);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t bBase = task * kRank4Reduce02BatchB;
            if (bBase >= dimB) {
                continue;
            }
            const uint32_t bCount = MinU32(kRank4Reduce02BatchB, dimB - bBase);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, bCount * rowPitchF);

            CopyRank4Reduce02BatchBInput(dimB, dimC, dimD, 0U, bBase, bCount);
            for (uint32_t a = 0U; a < dimA; ++a) {
                LocalTensor<T> ready = rank4InQueue.DeQue<T>();
                if (a + 1U < dimA) {
                    CopyRank4Reduce02BatchBInput(dimB, dimC, dimD,
                                                 a + 1U, bBase, bCount);
                }
                for (uint32_t localB = 0U; localB < bCount; ++localB) {
                    AccumulateSquaredRows(accLocal[localB * rowPitchF],
                                          ready[localB * dimC * rowPitchT],
                                          rowPitchT, dimC);
                }
                rank4InQueue.FreeTensor(ready);
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            for (uint32_t localB = 0U; localB < bCount; ++localB) {
                ConvertPlaneOutput(outputLocal[localB * rowPitchT],
                                   accLocal[localB * rowPitchF], dimD);
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopy(outputGm[bBase * dimD], readyOut, bCount * dimD);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessRank4Reduce02()
    {
        if (CanProcessRank4Reduce02BatchB()) {
            ProcessRank4Reduce02BatchB();
            return;
        }
        const uint32_t dimA = inDims[1];
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        if (dimA == 0U || dimB == 0U || dimC == 0U || dimD == 0U ||
            accTileElems == 0U) {
            return;
        }
        const uint32_t dTiles = (dimD + accTileElems - 1U) / accTileElems;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t b = task / dTiles;
            const uint32_t dTile = task - b * dTiles;
            if (b >= dimB) {
                continue;
            }
            const uint32_t dBase = dTile * accTileElems;
            const uint32_t count = MinU32(accTileElems, dimD - dBase);
            const uint32_t rowPitch = AlignElems32<T>(count);
            const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(count));

            uint32_t a = 0U;
            uint32_t cBase = 0U;
            uint32_t cCount = MinU32(kInnerReduceBatchRows, dimC);
            CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                                   rightPadding, a, cBase, cCount);
            while (true) {
                LocalTensor<T> ready = rank4InQueue.DeQue<T>();
                uint32_t nextA = a;
                uint32_t nextCBase = cBase + cCount;
                if (nextCBase >= dimC) {
                    nextCBase = 0U;
                    ++nextA;
                }
                const bool hasNext = nextA < dimA;
                uint32_t nextCCount = 0U;
                if (hasNext) {
                    nextCCount = MinU32(kInnerReduceBatchRows, dimC - nextCBase);
                    CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                                           rightPadding, nextA, nextCBase, nextCCount);
                }

                AccumulateSquaredRows(accLocal, ready, rowPitch, cCount);
                rank4InQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                a = nextA;
                cBase = nextCBase;
                cCount = nextCCount;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, count);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(count * sizeof(T)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[b * dimD + dBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void CopyPlaneInput(uint32_t inputBase, uint32_t prefixOffset,
                                          uint32_t suffixReduceSize, uint32_t reduceBase,
                                          uint32_t reduceCount, uint32_t rowCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceCount);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceCount);
        const uint32_t gmOffset = inputBase + prefixOffset + reduceBase;
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();

        // Fully aligned plane chunks can use the normal 2-D copy. Retain
        // DataCopyPad for every tail or unaligned row.
        const uint64_t blockBytes = static_cast<uint64_t>(reduceCount) * sizeof(T);
        const uint64_t srcGapBytes =
            static_cast<uint64_t>(suffixReduceSize - reduceCount) * sizeof(T);
        const uint64_t gmOffsetBytes = static_cast<uint64_t>(gmOffset) * sizeof(T);
        if (rightPadding == 0U && (blockBytes & 31U) == 0U &&
            (srcGapBytes & 31U) == 0U && (gmOffsetBytes & 31U) == 0U &&
            blockBytes / 32U <= 65535U && srcGapBytes / 32U <= 65535U) {
            DataCopyParams copyParams{static_cast<uint16_t>(rowCount),
                                      static_cast<uint16_t>(blockBytes / 32U),
                                      static_cast<uint16_t>(srcGapBytes / 32U), 0U};
            DataCopy(inputLocal, inputGm[gmOffset], copyParams);
            inQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(reduceCount * sizeof(T)),
            static_cast<uint32_t>((suffixReduceSize - reduceCount) * sizeof(T)),
            0U, 0U};
        DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                          static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessPlaneReduce()
    {
        const uint32_t suffixReduceStart = SuffixReduceStart();
        const uint32_t planeStart = PlaneStart(suffixReduceStart);
        if (suffixReduceStart == 0U || planeStart == suffixReduceStart || innerSize == 0U) {
            return;
        }

        uint32_t suffixReduceSize = 1U;
        for (uint32_t dim = suffixReduceStart; dim < kDimNum; ++dim) {
            suffixReduceSize *= inDims[dim];
        }
        if (suffixReduceSize == 0U) {
            return;
        }
        const uint32_t prefixReduceSize = reduceSize / suffixReduceSize;
        const uint32_t planeTiles = (innerSize + accTileElems - 1U) / accTileElems;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outerOutput = task / planeTiles;
            const uint32_t planeTile = task - outerOutput * planeTiles;
            const uint32_t planeBase = planeTile * accTileElems;
            const uint32_t rowCount = MinU32(accTileElems, innerSize - planeBase);
            const uint32_t outputOffset = outerOutput * innerSize + planeBase;
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(rowCount));

            uint32_t prefixCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t prefixReduce = 0U;
            uint32_t prefixOffset = 0U;
            uint32_t reduceBase = 0U;
            uint32_t reduceCount = MinU32(kPlaneReduceChunkElems, suffixReduceSize);
            CopyPlaneInput(inputBase, prefixOffset, suffixReduceSize, reduceBase,
                           reduceCount, rowCount);

            while (true) {
                LocalTensor<T> ready = inQueue.DeQue<T>();

                uint32_t nextPrefixReduce = prefixReduce;
                uint32_t nextPrefixOffset = prefixOffset;
                uint32_t nextReduceBase = reduceBase + reduceCount;
                if (nextReduceBase >= suffixReduceSize) {
                    nextReduceBase = 0U;
                    ++nextPrefixReduce;
                    if (nextPrefixReduce < prefixReduceSize) {
                        AdvancePrefixReduceOffset(planeStart, prefixCoords, nextPrefixOffset);
                    }
                }
                const bool hasNext = nextPrefixReduce < prefixReduceSize;
                uint32_t nextReduceCount = 0U;
                if (hasNext) {
                    nextReduceCount =
                        MinU32(kPlaneReduceChunkElems, suffixReduceSize - nextReduceBase);
                    CopyPlaneInput(inputBase, nextPrefixOffset, suffixReduceSize, nextReduceBase,
                                   nextReduceCount, rowCount);
                }

                const uint32_t rowPitch = AlignElems32<T>(reduceCount);
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                LocalTensor<float> squareLocal = squareBuf.Get<float>();
                LocalTensor<float> sumsLocal = sumBuf.Get<float>();
                const uint32_t batchElems = rowCount * rowPitch;
                SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
                WholeReduceSum<float>(sumsLocal, squareLocal,
                                      static_cast<int32_t>(reduceCount),
                                      static_cast<int32_t>(rowCount), 1, 1,
                                      static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                Add(accLocal, sumsLocal, accLocal, rowCount);
                inQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                prefixReduce = nextPrefixReduce;
                prefixOffset = nextPrefixOffset;
                reduceBase = nextReduceBase;
                reduceCount = nextReduceCount;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(rowCount * sizeof(T)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessContiguousBlockReduce()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        if (start >= end) {
            return;
        }

        uint32_t prefetchedTask = start;
        uint32_t prefetchedOffset = prefetchedTask * accTileElems;
        uint32_t prefetchedRows = MinU32(accTileElems, outputSize - prefetchedOffset);
        {
            LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
            CopyContiguousBlockInput(inputLocal, prefetchedOffset * reduceSize,
                                     prefetchedRows, rowPitch, rightPadding);
            inQueue.EnQue(inputLocal);
        }
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;

        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outputOffset = prefetchedOffset;
            const uint32_t rowCount = prefetchedRows;
            const uint32_t batchElems = rowCount * rowPitch;

            LocalTensor<T> ready = inQueue.DeQue<T>();
            if (task + 1U < end) {
                prefetchedTask = task + 1U;
                prefetchedOffset = prefetchedTask * accTileElems;
                prefetchedRows = MinU32(accTileElems, outputSize - prefetchedOffset);
                LocalTensor<T> nextInput = inQueue.AllocTensor<T>();
                CopyContiguousBlockInput(nextInput, prefetchedOffset * reduceSize,
                                         prefetchedRows, rowPitch, rightPadding);
                inQueue.EnQue(nextInput);
            }
            // The two VECOUT buffers allow the preceding result to retire on
            // MTE3 while this task squares and reduces its prefetched rows.
            // Queue events retain the ownership dependency for each buffer.
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = outQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
                outQueue.FreeTensor(readyOut);
            }
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            LocalTensor<float> squareLocal = squareBuf.Get<float>();
            LocalTensor<float> sumsLocal = sumBuf.Get<float>();
            SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
            const bool usedPackedChunks =
                TryReduceContiguousFloatChunks(sumsLocal, squareLocal, rowCount, rowPitch);
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            if (!usedPackedChunks) {
                LocalTensor<float> accLocal = accBuf.Get<float>();
                Duplicate(accLocal, 0.0F, AlignElems32<float>(rowCount));
                if (reduceSize <= kPlaneReduceChunkElems) {
                    WholeReduceSum<float>(sumsLocal, squareLocal,
                                          static_cast<int32_t>(reduceSize),
                                          static_cast<int32_t>(rowCount), 1, 1,
                                          static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                    Add(accLocal, sumsLocal, accLocal, rowCount);
                } else {
                    for (uint32_t reduceBase = 0U; reduceBase < reduceSize;
                         reduceBase += kPlaneReduceChunkElems) {
                        const uint32_t reduceCount =
                            MinU32(kPlaneReduceChunkElems, reduceSize - reduceBase);
                        WholeReduceSum<float>(sumsLocal, squareLocal[reduceBase],
                                              static_cast<int32_t>(reduceCount),
                                              static_cast<int32_t>(rowCount), 1, 1,
                                          static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                        Add(accLocal, sumsLocal, accLocal, rowCount);
                    }
                }
                ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            } else {
                ConvertPlaneOutput(outputLocal, sumsLocal, rowCount);
            }
            inQueue.FreeTensor(ready);

            outQueue.EnQue(outputLocal);
            pendingOutputOffset = outputOffset;
            pendingOutputRows = rowCount;
            hasPendingOutput = true;
        }
        if (hasPendingOutput) {
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline bool TryReduceContiguousFloatChunks(LocalTensor<float> sumsLocal,
                                                           LocalTensor<float> squareLocal,
                                                           uint32_t rowCount,
                                                           uint32_t rowPitch)
    {
        if constexpr (!std::is_same_v<T, float>) {
            (void)sumsLocal;
            (void)squareLocal;
            (void)rowCount;
            (void)rowPitch;
            return false;
        } else {
            constexpr uint32_t kChunkElems = kPlaneReduceChunkElems;
            constexpr uint32_t kMaxRepeat = 255U;
            if (rowPitch != reduceSize || reduceSize < kChunkElems * 8U ||
                (reduceSize % kChunkElems) != 0U) {
                return false;
            }
            const uint32_t chunkCount = reduceSize / kChunkElems;
            // The second stage reads one aligned partial row per output row.
            if (chunkCount > kChunkElems || (chunkCount & 7U) != 0U) {
                return false;
            }
            const uint32_t rowsPerCall = kMaxRepeat / chunkCount;
            if (rowsPerCall == 0U) {
                return false;
            }

            LocalTensor<float> partialLocal = calcBuf.Get<float>();
            for (uint32_t rowBase = 0U; rowBase < rowCount; rowBase += rowsPerCall) {
                const uint32_t rows = MinU32(rowsPerCall, rowCount - rowBase);
                RepeatReduceSum<float>(partialLocal, squareLocal[rowBase * rowPitch],
                                       static_cast<int32_t>(rows * chunkCount),
                                       static_cast<int32_t>(kChunkElems), 0, 1, 1,
                                       static_cast<int32_t>(kChunkElems / 8U));
                WholeReduceSum<float>(sumsLocal[rowBase], partialLocal,
                                      static_cast<int32_t>(chunkCount),
                                      static_cast<int32_t>(rows), 1, 1,
                                      static_cast<int32_t>(chunkCount / 8U));
            }
            return true;
        }
    }

    __aicore__ inline void CopyContiguousBlockInput(LocalTensor<T>& inputLocal,
                                                    uint32_t gmOffset, uint32_t rows,
                                                    uint32_t rowPitch, uint8_t rightPadding)
    {
        if (rightPadding == 0U) {
            DataCopy(inputLocal, inputGm[gmOffset], rows * rowPitch);
            return;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows),
                                     static_cast<uint32_t>(reduceSize * sizeof(T)),
                                     0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{true, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyContiguousBlockOutput(uint32_t outputOffset,
                                                     LocalTensor<T>& outputLocal,
                                                     uint32_t rows)
    {
        if (rows == AlignElems32<T>(rows)) {
            DataCopy(outputGm[outputOffset], outputLocal, rows);
            return;
        }
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(rows * sizeof(T)),
                                     0U, 0U, 0U};
        DataCopyPad(outputGm[outputOffset], outputLocal, outParams);
    }

    __aicore__ inline void ProcessGenericFallback()
    {
        const uint32_t lastReduceDim = LastReduceDim();
        if (lastReduceDim != kDimNum - 1U) {
            return;
        }
        const uint32_t contiguousReduceSize = inDims[lastReduceDim];
        const uint32_t outerReduceSize =
            contiguousReduceSize == 0U ? 0U : reduceSize / contiguousReduceSize;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
            const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
            LocalTensor<float> scalarFloat = valueFloatBuf.Get<float>();
            for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
                const uint32_t outIdx = outBase + localOut;
                const uint32_t inputBase = BaseInputOffset(outIdx);
                uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
                uint32_t outerReduceOffset = 0U;
                float acc = 0.0F;
                for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                    for (uint32_t r = 0U; r < contiguousReduceSize; r += tileElems) {
                        const uint32_t count = MinU32(tileElems, contiguousReduceSize - r);
                        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
                        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(T)),
                                                     0U, 0U, 0U};
                        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
                        DataCopyPad(inputLocal, inputGm[inputBase + outerReduceOffset + r],
                                    copyParams, padParams);
                        inQueue.EnQue(inputLocal);

                        LocalTensor<T> ready = inQueue.DeQue<T>();
                        LocalTensor<float> calcLocal = calcBuf.Get<float>();
                        LocalTensor<float> workLocal = workBuf.Get<float>();
                        LocalTensor<float> sumLocal = sumBuf.Get<float>();
                        Cast(calcLocal, ready, RoundMode::CAST_NONE, AlignElems32<T>(count));
                        Mul(workLocal, calcLocal, calcLocal, count);
                        ReduceSum(sumLocal, workLocal, calcLocal, count);
                        acc += sumLocal.GetValue(0U);
                        inQueue.FreeTensor(ready);
                    }
                    AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                }
                scalarFloat.SetValue(localOut, acc);
            }
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Cast(outputLocal, scalarFloat, RoundMode::CAST_NONE, AlignElems32<T>(outCount));
            outQueue.EnQue(outputLocal);
            LocalTensor<T> ready = outQueue.DeQue<T>();
            DataCopyExtParams params{1U, static_cast<uint32_t>(outCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], ready, params);
            outQueue.FreeTensor(ready);
        }
    }

    TPipe* pipe = nullptr;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<float> workspaceGm;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECIN, 2> rank4InQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 2> suffixChunkInQueue;
    TQue<QuePosition::VECOUT, 2> suffixChunkOutQueue;
    TBuf<TPosition::VECCALC> valueFloatBuf;
    TBuf<TPosition::VECCALC> calcBuf;
    TBuf<TPosition::VECCALC> squareBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> suffixChunkCalcBuf;
    TBuf<TPosition::VECCALC> suffixChunkSquareBuf;
    TBuf<TPosition::VECCALC> suffixChunkSumBuf;
    TBuf<TPosition::VECCALC> suffixChunkAccBuf;
    uint32_t mode = kModeGenericFallback;
    uint32_t inputSize = 0U;
    uint32_t outputSize = 0U;
    uint32_t blockElems = 0U;
    uint32_t totalBlocks = 0U;
    uint32_t tileElems = 0U;
    uint32_t accTileElems = 0U;
    uint32_t reduceSize = 0U;
    uint32_t innerSize = 0U;
    uint32_t taskNum = 0U;
    uint32_t inDims[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t inStrides[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t outStrides[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t reduceMask[kDimNum] = {0U, 0U, 0U, 0U, 0U};
};

template <>
__aicore__ inline void SquareSumKernel<float>::SquarePlaneBatch(
    LocalTensor<float> squareLocal, LocalTensor<float> inputLocal,
    LocalTensor<float> calcLocal, uint32_t count)
{
    (void)calcLocal;
    Mul(squareLocal, inputLocal, inputLocal, count);
}

template <>
__aicore__ inline void SquareSumKernel<bfloat16_t>::SquarePlaneBatch(
    LocalTensor<float> squareLocal, LocalTensor<bfloat16_t> inputLocal,
    LocalTensor<float> calcLocal, uint32_t count)
{
    Cast(calcLocal, inputLocal, RoundMode::CAST_NONE, count);
    Mul(squareLocal, calcLocal, calcLocal, count);
}

template <>
__aicore__ inline void SquareSumKernel<float>::ConvertPlaneOutput(
    LocalTensor<float> outputLocal, LocalTensor<float> accLocal, uint32_t count)
{
    DataCopy(outputLocal, accLocal, AlignElems32<float>(count));
}

template <>
__aicore__ inline void SquareSumKernel<bfloat16_t>::ConvertPlaneOutput(
    LocalTensor<bfloat16_t> outputLocal, LocalTensor<float> accLocal, uint32_t count)
{
    // Both tensors start at their buffer base, so this conversion preserves
    // the 32B alignment required by the vector cast for a partial tile.
    Cast(outputLocal, accLocal, RoundMode::CAST_RINT, AlignElems32<bfloat16_t>(count));
}

template <>
__aicore__ inline void SquareSumKernel<float>::InitSuffixBuffers()
{
    const uint32_t alignedTileF = AlignElems32<float>(tileElems);
    const uint32_t alignedOutF = AlignElems32<float>(kSuffixOutTileElems);
    pipe->InitBuffer(inQueue, kBufferNum, alignedTileF * sizeof(float));
    pipe->InitBuffer(outQueue, kBufferNum, alignedOutF * sizeof(float));
    pipe->InitBuffer(workBuf, alignedTileF * sizeof(float));
    pipe->InitBuffer(sumBuf, AlignElems32<float>(1U) * sizeof(float));
}

template <>
__aicore__ inline void SquareSumKernel<float>::InitInnerBuffers()
{
    const uint32_t alignedRowF = AlignElems32<float>(accTileElems);
    pipe->InitBuffer(inQueue, 2U,
                     alignedRowF * kInnerPipelinedBatchRows * sizeof(float));
    pipe->InitBuffer(outQueue, kBufferNum, alignedRowF * sizeof(float));
    pipe->InitBuffer(accBuf, alignedRowF * sizeof(float));
}

template <>
__aicore__ inline void SquareSumKernel<float>::InitRank4Reduce02Buffers()
{
    const uint32_t alignedRow = AlignElems32<float>(accTileElems);
    if (CanProcessRank4Reduce02BatchB()) {
        const uint32_t inputElems = kRank4Reduce02BatchB * inDims[3] * alignedRow;
        pipe->InitBuffer(rank4InQueue, 2U, inputElems * sizeof(float));
        pipe->InitBuffer(outQueue, kBufferNum,
                         kRank4Reduce02BatchB * alignedRow * sizeof(float));
        pipe->InitBuffer(accBuf,
                         kRank4Reduce02BatchB * alignedRow * sizeof(float));
        return;
    }
    const uint32_t batchElems = alignedRow * kInnerReduceBatchRows;
    pipe->InitBuffer(rank4InQueue, 2U, batchElems * sizeof(float));
    pipe->InitBuffer(outQueue, kBufferNum, alignedRow * sizeof(float));
    pipe->InitBuffer(accBuf, alignedRow * sizeof(float));
}

template <>
__aicore__ inline void SquareSumKernel<float>::ProcessSuffixReduce()
{
    if (reduceSize != 0U && reduceSize <= kSuffixBatchMaxReduce) {
        const uint32_t rowPitch = AlignElems32<float>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixBatchRows) {
            const uint32_t outCount = MinU32(kSuffixBatchRows, end - outBase);
            const uint32_t batchElems = outCount * rowPitch;
            LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(outCount),
                                         static_cast<uint32_t>(reduceSize * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> padParams{rightPadding != 0U, 0U, rightPadding, 0.0F};
            DataCopyPad(inputLocal, inputGm[outBase * reduceSize], copyParams, padParams);
            inQueue.EnQue(inputLocal);

            LocalTensor<float> ready = inQueue.DeQue<float>();
            LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
            LocalTensor<float> squareLocal = workBuf.Get<float>();
            Mul(squareLocal, ready, ready, batchElems);
            WholeReduceSum<float>(outputLocal, squareLocal, static_cast<int32_t>(reduceSize),
                                  static_cast<int32_t>(outCount), 1, 1,
                                  static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
            inQueue.FreeTensor(ready);
            outQueue.EnQue(outputLocal);
            LocalTensor<float> readyOut = outQueue.DeQue<float>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
        return;
    }
    uint32_t start = 0U;
    uint32_t end = 0U;
    CalcOutputRange(start, end);
    for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
        const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
        LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
            const uint32_t outIdx = outBase + localOut;
            float acc = 0.0F;
            const uint32_t base = outIdx * reduceSize;
            for (uint32_t r = 0U; r < reduceSize; r += tileElems) {
                const uint32_t count = MinU32(tileElems, reduceSize - r);
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
                DataCopyPad(inputLocal, inputGm[base + r], copyParams, padParams);
                inQueue.EnQue(inputLocal);

                LocalTensor<float> ready = inQueue.DeQue<float>();
                LocalTensor<float> workLocal = workBuf.Get<float>();
                LocalTensor<float> sumLocal = sumBuf.Get<float>();
                Mul(workLocal, ready, ready, count);
                ReduceSum(sumLocal, workLocal, ready, count);
                acc += sumLocal.GetValue(0U);
                inQueue.FreeTensor(ready);
            }
            outputLocal.SetValue(localOut, acc);
        }
        outQueue.EnQue(outputLocal);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outBase], readyOut, outParams);
        outQueue.FreeTensor(readyOut);
    }
}

template <>
__aicore__ inline void SquareSumKernel<float>::ProcessInnerAccum()
{
    const uint32_t lastReduceDim = LastReduceDim();
    if (lastReduceDim >= kDimNum) {
        return;
    }
    const uint32_t innerReduceSize = inDims[lastReduceDim];
    const uint32_t outerReduceSize = innerReduceSize == 0U ? 0U : reduceSize / innerReduceSize;
    uint32_t start = 0U;
    uint32_t end = 0U;
    CalcRange(taskNum, start, end);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t outputOffset = task * accTileElems;
        if (outputOffset >= outputSize) {
            continue;
        }
        const uint32_t outputCount = MinU32(accTileElems, outputSize - outputOffset);
        LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        uint32_t localOutputOffset = 0U;
        while (localOutputOffset < outputCount) {
            const uint32_t globalOutputOffset = outputOffset + localOutputOffset;
            const uint32_t innerOffset = globalOutputOffset % innerSize;
            const uint32_t count = MinU32(outputCount - localOutputOffset, innerSize - innerOffset);
            const uint32_t rowPitch = AlignElems32<float>(count);
            const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, rowPitch);
            const uint32_t inputBase = BaseInputOffset(globalOutputOffset);
            uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t outerReduceOffset = 0U;
            for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                uint32_t reduceBase = 0U;
                uint32_t reduceCount =
                    MinU32(kInnerPipelinedBatchRows, innerReduceSize);
                LocalTensor<float> firstInput = inQueue.AllocTensor<float>();
                DataCopyExtParams firstCopyParams{
                    static_cast<uint16_t>(reduceCount),
                    static_cast<uint32_t>(count * sizeof(float)),
                    static_cast<uint32_t>((innerSize - count) * sizeof(float)), 0U, 0U};
                DataCopyPadExtParams<float> padParams{rightPadding != 0U, 0U, rightPadding, 0.0F};
                DataCopyPad(firstInput, inputGm[inputBase + outerReduceOffset],
                            firstCopyParams, padParams);
                inQueue.EnQue(firstInput);

                while (true) {
                    LocalTensor<float> ready = inQueue.DeQue<float>();
                    const uint32_t nextReduceBase = reduceBase + reduceCount;
                    uint32_t nextReduceCount = 0U;
                    if (nextReduceBase < innerReduceSize) {
                        nextReduceCount = MinU32(kInnerPipelinedBatchRows,
                                                 innerReduceSize - nextReduceBase);
                        LocalTensor<float> nextInput = inQueue.AllocTensor<float>();
                        DataCopyExtParams nextCopyParams{
                            static_cast<uint16_t>(nextReduceCount),
                            static_cast<uint32_t>(count * sizeof(float)),
                            static_cast<uint32_t>((innerSize - count) * sizeof(float)),
                            0U, 0U};
                        DataCopyPad(nextInput,
                                    inputGm[inputBase + outerReduceOffset +
                                            nextReduceBase * innerSize],
                                    nextCopyParams, padParams);
                        inQueue.EnQue(nextInput);
                    }

                    AccumulateSquaredRows(accLocal, ready, rowPitch, reduceCount);
                    inQueue.FreeTensor(ready);

                    if (nextReduceBase >= innerReduceSize) {
                        break;
                    }
                    reduceBase = nextReduceBase;
                    reduceCount = nextReduceCount;
                }
                AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
            }
            for (uint32_t i = 0U; i < count; ++i) {
                outputLocal.SetValue(localOutputOffset + i, accLocal.GetValue(i));
            }
            localOutputOffset += count;
        }
        outQueue.EnQue(outputLocal);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(outputCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outputOffset], readyOut, outParams);
        outQueue.FreeTensor(readyOut);
    }
}

template <>
__aicore__ inline void SquareSumKernel<float>::ProcessRank4Reduce02()
{
    if (CanProcessRank4Reduce02BatchB()) {
        ProcessRank4Reduce02BatchB();
        return;
    }
    const uint32_t dimA = inDims[1];
    const uint32_t dimB = inDims[2];
    const uint32_t dimC = inDims[3];
    const uint32_t dimD = inDims[4];
    if (dimA == 0U || dimB == 0U || dimC == 0U || dimD == 0U ||
        accTileElems == 0U) {
        return;
    }
    const uint32_t dTiles = (dimD + accTileElems - 1U) / accTileElems;
    uint32_t start = 0U;
    uint32_t end = 0U;
    CalcRange(taskNum, start, end);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t b = task / dTiles;
        const uint32_t dTile = task - b * dTiles;
        if (b >= dimB) {
            continue;
        }
        const uint32_t dBase = dTile * accTileElems;
        const uint32_t count = MinU32(accTileElems, dimD - dBase);
        const uint32_t rowPitch = AlignElems32<float>(count);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, rowPitch);

        uint32_t a = 0U;
        uint32_t cBase = 0U;
        uint32_t cCount = MinU32(kInnerReduceBatchRows, dimC);
        CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                               rightPadding, a, cBase, cCount);
        while (true) {
            LocalTensor<float> ready = rank4InQueue.DeQue<float>();
            uint32_t nextA = a;
            uint32_t nextCBase = cBase + cCount;
            if (nextCBase >= dimC) {
                nextCBase = 0U;
                ++nextA;
            }
            const bool hasNext = nextA < dimA;
            uint32_t nextCCount = 0U;
            if (hasNext) {
                nextCCount = MinU32(kInnerReduceBatchRows, dimC - nextCBase);
                CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                                       rightPadding, nextA, nextCBase, nextCCount);
            }

            AccumulateSquaredRows(accLocal, ready, rowPitch, cCount);
            rank4InQueue.FreeTensor(ready);

            if (!hasNext) {
                break;
            }
            a = nextA;
            cBase = nextCBase;
            cCount = nextCCount;
        }

        LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        DataCopy(outputLocal, accLocal, rowPitch);
        outQueue.EnQue(outputLocal);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(count * sizeof(float)),
                                    0U, 0U, 0U};
        DataCopyPad(outputGm[b * dimD + dBase], readyOut, outParams);
        outQueue.FreeTensor(readyOut);
    }
}

template <>
__aicore__ inline void SquareSumKernel<float>::ProcessGenericFallback()
{
    const uint32_t lastReduceDim = LastReduceDim();
    if (lastReduceDim != kDimNum - 1U) {
        return;
    }
    const uint32_t contiguousReduceSize = inDims[lastReduceDim];
    const uint32_t outerReduceSize =
        contiguousReduceSize == 0U ? 0U : reduceSize / contiguousReduceSize;
    uint32_t start = 0U;
    uint32_t end = 0U;
    CalcOutputRange(start, end);
    for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
        const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
        LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
            const uint32_t outIdx = outBase + localOut;
            const uint32_t inputBase = BaseInputOffset(outIdx);
            uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t outerReduceOffset = 0U;
            float acc = 0.0F;
            for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                for (uint32_t r = 0U; r < contiguousReduceSize; r += tileElems) {
                    const uint32_t count = MinU32(tileElems, contiguousReduceSize - r);
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(float)),
                                                 0U, 0U, 0U};
                    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
                    DataCopyPad(inputLocal, inputGm[inputBase + outerReduceOffset + r],
                                copyParams, padParams);
                    inQueue.EnQue(inputLocal);

                    LocalTensor<float> ready = inQueue.DeQue<float>();
                    LocalTensor<float> workLocal = workBuf.Get<float>();
                    LocalTensor<float> sumLocal = sumBuf.Get<float>();
                    Mul(workLocal, ready, ready, count);
                    ReduceSum(sumLocal, workLocal, ready, count);
                    acc += sumLocal.GetValue(0U);
                    inQueue.FreeTensor(ready);
                }
                AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
            }
            outputLocal.SetValue(localOut, acc);
        }
        outQueue.EnQue(outputLocal);
        LocalTensor<float> ready = outQueue.DeQue<float>();
        DataCopyExtParams params{1U, static_cast<uint32_t>(outCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outBase], ready, params);
        outQueue.FreeTensor(ready);
    }
}

template <>
__aicore__ inline void SquareSumKernel<bfloat16_t>::Process()
{
    if (outputSize == 0U) {
        return;
    }
    if (CanProcessFullRowBf16Suffix()) {
        const uint32_t compactGroups = InitFullRowBf16SuffixBuffers();
        ProcessFullRowBf16Suffix(compactGroups);
        return;
    }
    if (mode == kModeSuffixReduce && CanProcessLargeSuffixChunked()) {
        InitLargeSuffixChunkedBuffers();
        ProcessLargeSuffixChunked();
        return;
    }
    if (mode == kModePlaneReduce) {
        InitPlaneBuffers();
        ProcessPlaneReduce();
        return;
    }
    if (mode == kModeContiguousBlockReduce) {
        InitContiguousBlockBuffers();
        ProcessContiguousBlockReduce();
        return;
    }
    if (mode == kModeInnerAccum && CanProcessRowShardedBf16Middle()) {
        InitRowShardedBf16MiddleBuffers();
        ProcessRowShardedBf16Middle();
        return;
    }
    if (mode == kModeInnerAccum && CanProcessBalancedMiddleReduce()) {
        InitBalancedMiddleReduceBuffers();
        ProcessBalancedMiddleReduce();
        return;
    }
    if (mode == kModeInnerAccum) {
        const uint32_t alignedRowElems = AlignElems32<bfloat16_t>(accTileElems);
        pipe->InitBuffer(inQueue, 2U,
                         alignedRowElems * kInnerReduceBatchRows * sizeof(bfloat16_t));
        pipe->InitBuffer(outQueue, kBufferNum, alignedRowElems * sizeof(bfloat16_t));
        pipe->InitBuffer(calcBuf, alignedRowElems * kInnerReduceBatchRows * sizeof(float));
        pipe->InitBuffer(accBuf, alignedRowElems * sizeof(float));
        const uint32_t lastReduceDim = LastReduceDim();
        if (lastReduceDim >= kDimNum) {
            return;
        }
        const uint32_t innerReduceSize = inDims[lastReduceDim];
        const uint32_t outerReduceSize = innerReduceSize == 0U ? 0U : reduceSize / innerReduceSize;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outputOffset = task * accTileElems;
            if (outputOffset >= outputSize) {
                continue;
            }
            const uint32_t outputCount = MinU32(accTileElems, outputSize - outputOffset);
            LocalTensor<bfloat16_t> outputLocal = outQueue.AllocTensor<bfloat16_t>();
            uint32_t localOutputOffset = 0U;
            while (localOutputOffset < outputCount) {
                const uint32_t globalOutputOffset = outputOffset + localOutputOffset;
                const uint32_t innerOffset = globalOutputOffset % innerSize;
                const uint32_t count = MinU32(outputCount - localOutputOffset, innerSize - innerOffset);
                const uint32_t rowPitch = AlignElems32<bfloat16_t>(count);
                const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
                LocalTensor<float> accLocal = accBuf.Get<float>();
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                Duplicate(accLocal, 0.0F, rowPitch);
                const uint32_t inputBase = BaseInputOffset(globalOutputOffset);
                uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
                uint32_t outerReduceOffset = 0U;
                for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                    uint32_t reduceBase = 0U;
                    uint32_t reduceCount =
                        MinU32(kInnerReduceBatchRows, innerReduceSize - reduceBase);
                    LocalTensor<bfloat16_t> firstInput = inQueue.AllocTensor<bfloat16_t>();
                    DataCopyExtParams firstCopyParams{
                        static_cast<uint16_t>(reduceCount),
                        static_cast<uint32_t>(count * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((innerSize - count) * sizeof(bfloat16_t)),
                        0U, 0U};
                    DataCopyPadExtParams<bfloat16_t> padParams{rightPadding != 0U, 0U, rightPadding,
                                                               static_cast<bfloat16_t>(0)};
                    DataCopyPad(firstInput, inputGm[inputBase + outerReduceOffset],
                                firstCopyParams, padParams);
                    inQueue.EnQue(firstInput);

                    while (true) {
                        LocalTensor<bfloat16_t> ready = inQueue.DeQue<bfloat16_t>();
                        const uint32_t nextReduceBase = reduceBase + reduceCount;
                        uint32_t nextReduceCount = 0U;
                        if (nextReduceBase < innerReduceSize) {
                            nextReduceCount = MinU32(kInnerReduceBatchRows,
                                                     innerReduceSize - nextReduceBase);
                            LocalTensor<bfloat16_t> nextInput = inQueue.AllocTensor<bfloat16_t>();
                            DataCopyExtParams nextCopyParams{
                                static_cast<uint16_t>(nextReduceCount),
                                static_cast<uint32_t>(count * sizeof(bfloat16_t)),
                                static_cast<uint32_t>((innerSize - count) * sizeof(bfloat16_t)),
                                0U, 0U};
                            DataCopyPad(nextInput,
                                        inputGm[inputBase + outerReduceOffset +
                                                nextReduceBase * innerSize],
                                        nextCopyParams, padParams);
                            inQueue.EnQue(nextInput);
                        }

                        const uint32_t batchElems = reduceCount * rowPitch;
                        // Keep the next MTE2 transfer in flight while the current batch
                        // expands BF16, squares it, and accumulates the row partials.
                        Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
                        Mul(calcLocal, calcLocal, calcLocal, batchElems);
                        AccumulateRows(accLocal, calcLocal, rowPitch, reduceCount);
                        inQueue.FreeTensor(ready);

                        if (nextReduceBase >= innerReduceSize) {
                            break;
                        }
                        reduceBase = nextReduceBase;
                        reduceCount = nextReduceCount;
                    }
                    AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    outputLocal.SetValue(localOutputOffset + i, FloatToBf16(accLocal.GetValue(i)));
                }
                localOutputOffset += count;
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<bfloat16_t> readyOut = outQueue.DeQue<bfloat16_t>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outputCount * sizeof(bfloat16_t)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
        return;
    }

    const uint32_t alignedTileT = AlignElems32<bfloat16_t>(tileElems);
    const uint32_t alignedTileF = AlignElems32<float>(tileElems);
    pipe->InitBuffer(inQueue, kBufferNum, alignedTileT * sizeof(bfloat16_t));
    pipe->InitBuffer(outQueue, kBufferNum,
                     AlignElems32<bfloat16_t>(kSuffixOutTileElems) * sizeof(bfloat16_t));
    pipe->InitBuffer(calcBuf, alignedTileF * sizeof(float));
    pipe->InitBuffer(workBuf, alignedTileF * sizeof(float));
    pipe->InitBuffer(sumBuf, AlignElems32<float>(kSuffixBatchRows) * sizeof(float));
    pipe->InitBuffer(valueFloatBuf,
                     AlignElems32<float>(kSuffixOutTileElems) * sizeof(float));

    uint32_t start = 0U;
    uint32_t end = 0U;
    CalcOutputRange(start, end);
    if (mode == kModeSuffixReduce && reduceSize != 0U && reduceSize <= kSuffixBatchMaxReduce) {
        const uint32_t rowPitch = AlignElems32<bfloat16_t>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixBatchRows) {
            const uint32_t outCount = MinU32(kSuffixBatchRows, end - outBase);
            LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(outCount),
                                         static_cast<uint32_t>(reduceSize * sizeof(bfloat16_t)),
                                         0U, 0U, 0U};
            DataCopyPadExtParams<bfloat16_t> padParams{rightPadding != 0U, 0U, rightPadding,
                                                       static_cast<bfloat16_t>(0)};
            DataCopyPad(inputLocal, inputGm[outBase * reduceSize], copyParams, padParams);
            inQueue.EnQue(inputLocal);

            LocalTensor<bfloat16_t> ready = inQueue.DeQue<bfloat16_t>();
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            const uint32_t batchElems = outCount * rowPitch;
            Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
            Mul(calcLocal, calcLocal, calcLocal, batchElems);
            LocalTensor<float> sumsLocal = sumBuf.Get<float>();
            WholeReduceSum<float>(sumsLocal, calcLocal, static_cast<int32_t>(reduceSize),
                                  static_cast<int32_t>(outCount), 1, 1,
                                  static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
            inQueue.FreeTensor(ready);

            LocalTensor<bfloat16_t> outputLocal = outQueue.AllocTensor<bfloat16_t>();
            Cast(outputLocal, sumsLocal, RoundMode::CAST_RINT,
                 AlignElems32<bfloat16_t>(outCount));
            outQueue.EnQue(outputLocal);
            LocalTensor<bfloat16_t> readyOut = outQueue.DeQue<bfloat16_t>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(bfloat16_t)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
        return;
    }

    const uint32_t lastReduceDim = mode == kModeSuffixReduce ? kDimNum : LastReduceDim();
    const uint32_t contiguousReduceSize = mode == kModeSuffixReduce ? reduceSize : inDims[kDimNum - 1U];
    const uint32_t outerReduceSize = contiguousReduceSize == 0U ? 0U : reduceSize / contiguousReduceSize;
    for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
        const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
        LocalTensor<bfloat16_t> outputLocal = outQueue.AllocTensor<bfloat16_t>();
        LocalTensor<float> outputFloat = valueFloatBuf.Get<float>();
        for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
            const uint32_t outIdx = outBase + localOut;
            const uint32_t inputBase = mode == kModeSuffixReduce ? outIdx * reduceSize : BaseInputOffset(outIdx);
            uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t outerReduceOffset = 0U;
            float acc = 0.0F;
            for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                for (uint32_t r = 0U; r < contiguousReduceSize; r += tileElems) {
                    const uint32_t count = MinU32(tileElems, contiguousReduceSize - r);
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(bfloat16_t)),
                                                 0U, 0U, 0U};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U,
                                                               static_cast<bfloat16_t>(0)};
                    DataCopyPad(inputLocal, inputGm[inputBase + outerReduceOffset + r],
                                copyParams, padParams);
                    inQueue.EnQue(inputLocal);

                    LocalTensor<bfloat16_t> ready = inQueue.DeQue<bfloat16_t>();
                    LocalTensor<float> calcLocal = calcBuf.Get<float>();
                    LocalTensor<float> sumLocal = sumBuf.Get<float>();
                    LocalTensor<float> workLocal = workBuf.Get<float>();
                    Cast(calcLocal, ready, RoundMode::CAST_NONE,
                         AlignElems32<bfloat16_t>(count));
                    Mul(workLocal, calcLocal, calcLocal, count);
                    ReduceSum(sumLocal, workLocal, calcLocal, count);
                    acc += sumLocal.GetValue(0U);
                    inQueue.FreeTensor(ready);
                }
                if (mode != kModeSuffixReduce) {
                    AdvanceOuterReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                }
            }
            outputFloat.SetValue(localOut, acc);
        }
        Cast(outputLocal, outputFloat, RoundMode::CAST_RINT,
             AlignElems32<bfloat16_t>(outCount));
        outQueue.EnQue(outputLocal);
        LocalTensor<bfloat16_t> readyOut = outQueue.DeQue<bfloat16_t>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(bfloat16_t)),
                                    0U, 0U, 0U};
        DataCopyPad(outputGm[outBase], readyOut, outParams);
        outQueue.FreeTensor(readyOut);
    }
}

template <typename T>
__aicore__ inline void RunSquareSum(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    SquareSumKernel<T> kernel;
    kernel.Init(input, output, workspace, tiling, &pipe);
    kernel.Process();
}

__aicore__ inline void RunSquareSumBf16RowShard(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    SquareSumKernel<bfloat16_t> kernel;
    kernel.Init(input, output, workspace, tiling, &pipe);
    kernel.ProcessRowShardedBf16Entry();
}
}  // namespace

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(0)) {
        RunSquareSum<float>(input, output, userWorkspace, tiling);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIV_1_0);
        RunSquareSum<float>(input, output, userWorkspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunSquareSum<half>(input, output, userWorkspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        RunSquareSum<bfloat16_t>(input, output, userWorkspace, tiling);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIV_1_0);
        RunSquareSumBf16RowShard(input, output, userWorkspace, tiling);
    }
}
