#include "kernel_operator.h"
#include "square_sum_v1_tiling_key.h"

namespace nce_squaresum {
using namespace AscendC;

constexpr uint32_t LAYOUT_AR = 0;
constexpr uint32_t LAYOUT_ARA = 1;
constexpr uint32_t LAYOUT_RAR = 2;
constexpr uint32_t LAYOUT_RA = 3;
constexpr uint32_t LAYOUT_MISC = 4;
constexpr uint32_t ALIGNMENT_BYTES = 32;
constexpr uint32_t BUFFER_PADDING_BYTES = 256;
constexpr uint32_t AR_BATCH_ROWS = 248;
constexpr uint32_t REPEAT_CAP = 255;
constexpr uint32_t GATHER_CAP = REPEAT_CAP * 64U;
constexpr uint64_t COPY_STRIDE_CAP = 0xFFFFFFFFULL;
constexpr uint32_t ARA_PHASE_LANES = 2;
constexpr uint32_t RA_TILED_MODE = 1;
constexpr uint32_t RA_PHASE = 2;
constexpr uint32_t RA_PHASE_LANES = 1;
constexpr uint32_t RA_PHASE_SEGMENT_LIMIT = 2;
constexpr uint32_t RA_PHASE_WAVE_MODE = 1;
constexpr uint32_t RAR_WAVE_MODE = 1;

__aicore__ inline uint64_t Smallest(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t PadTo8(uint32_t count)
{
    return (count + 7U) / 8U * 8U;
}

__aicore__ inline uint32_t GCD(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

template <typename T>
class SquareSumComputeEngine {
private:
    const SquareSumV1TilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    GlobalTensor<T> gmIn_;
    GlobalTensor<T> gmOut_;
    GlobalTensor<float> gmWorkspace_;
    TQue<TPosition::VECIN, 1> qIn_;
    TQue<TPosition::VECOUT, 1> qOut_;
    TBuf<TPosition::VECCALC> bufScratch_;
    TBuf<TPosition::VECCALC> bufCheckpoint_;
    TBuf<TPosition::VECCALC> bufReduce_;
    TBuf<TPosition::VECCALC> bufAcc_;
    TBuf<TPosition::VECCALC> bufShared_;
    TBuf<TPosition::VECCALC> bufBlocks_;
    TBuf<TPosition::VECCALC> bufPacked_;

    // Generic / Fallback methods
    __aicore__ inline void ExecuteFallback()
    {
        const uint32_t lastDim = tiling_->rank - 1;
        const uint32_t lastSize = static_cast<uint32_t>(tiling_->shape[lastDim]);
        const bool lastReduced = (tiling_->axisMask & (1U << lastDim)) != 0;
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        if (lastReduced) {
            const uint64_t outerReduce = tiling_->reduceSize / lastSize;
            for (uint64_t output = 0; output < tiling_->outputSize; ++output) {
                Duplicate<float>(accumulator, 0.0f, 8);
                PipeBarrier<PIPE_V>();
                const uint64_t outputBase = DecodeNormalOffset(output, false);
                for (uint64_t reduce = 0; reduce < outerReduce; ++reduce) {
                    const uint64_t gmOffset = outputBase + DecodeReducedOffset(reduce, true);
                    LocalTensor<T> inputLocal = LoadAndSquare(gmOffset, 1, lastSize, lastSize);
                    LocalTensor<float> partial = RowReduceAr(1, SquaredView(inputLocal));
                    qIn_.FreeTensor(inputLocal);
                    Add<float>(accumulator, accumulator, partial, 1);
                    PipeBarrier<PIPE_V>();
                }
                CommitOutput(output, accumulator, 1);
            }
            return;
        }

        const uint64_t outputGroups = tiling_->outputSize / lastSize;
        for (uint64_t group = 0; group < outputGroups; ++group) {
            Duplicate<float>(accumulator, 0.0f, tiling_->alignedInner);
            PipeBarrier<PIPE_V>();
            const uint64_t outputBase = DecodeNormalOffset(group, true);
            for (uint64_t reduce = 0; reduce < tiling_->reduceSize; ++reduce) {
                const uint64_t gmOffset = outputBase + DecodeReducedOffset(reduce, false);
                LocalTensor<T> inputLocal = LoadAndSquare(gmOffset, 1, lastSize, lastSize);
                Add<float>(accumulator, accumulator, SquaredView(inputLocal), tiling_->alignedInner);
                PipeBarrier<PIPE_V>();
                qIn_.FreeTensor(inputLocal);
            }
            CommitOutput(group * lastSize, accumulator, lastSize);
        }
    }

    // RA methods
    __aicore__ inline void ExecuteRa()
    {
        if constexpr (std::is_same<T, float>::value) {
            if (tiling_->compactRa == RA_PHASE) {
                ProcessRaCompactPhase();
                return;
            }
            if (tiling_->compactRa == RA_TILED_MODE) {
                ProcessRaCompactTiled();
                return;
            }
        }
        const uint64_t reduceSize = tiling_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim1);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(reduceSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        Duplicate<float>(accumulator, 0.0f, tiling_->alignedInner);
        PipeBarrier<PIPE_V>();
        uint64_t reduceBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, coreEnd - reduceBase));
        const uint32_t staggerGroups = tiling_->raStaggerGroups;
        if (staggerGroups > 1 && rows == tiling_->tilePrimary) {
            const uint64_t coreRows = coreEnd - coreStart;
            const uint64_t rounds = (coreRows + tiling_->tilePrimary - 1) /
                tiling_->tilePrimary;
            const uint32_t tailRows = static_cast<uint32_t>(
                coreRows - (rounds - 1) * tiling_->tilePrimary);
            const uint32_t shiftBudget = tiling_->tilePrimary - tailRows;
            const uint32_t group = GetBlockIdx() % staggerGroups;
            const uint32_t firstShift = shiftBudget * group / (staggerGroups - 1);
            rows = tiling_->tilePrimary - firstShift;
        }
        LoadStridedTile(reduceBase * innerSize, rows, innerSize, innerSize);
        while (reduceBase < coreEnd) {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            const uint64_t nextBase = reduceBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                LoadStridedTile(nextBase * innerSize, nextRows, innerSize, innerSize);
            }
            ComputeSquare(inputLocal, rows * tiling_->alignedInner);
            LocalTensor<float> partial = RowReduceRa(rows, SquaredView(inputLocal));
            qIn_.FreeTensor(inputLocal);
            Add<float>(accumulator, accumulator, partial, tiling_->alignedInner);
            PipeBarrier<PIPE_V>();
            if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                LoadStridedTile(nextBase * innerSize, nextRows, innerSize, innerSize);
            }
            reduceBase = nextBase;
            rows = nextRows;
        }

        const uint64_t workspaceStrideElements = tiling_->workspaceStride / sizeof(float);
        SyncVToMte3();
        DataCopy(gmWorkspace_[static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements],
            accumulator, tiling_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<T> mergeInput = qIn_.AllocTensor<T>();
            qIn_.EnQue(mergeInput);
            mergeInput = qIn_.DeQue<T>();
            ConsolidateCores(accumulator, mergeInput, workspaceStrideElements);
            qIn_.FreeTensor(mergeInput);
        } else {
            ConsolidateCores(accumulator, bufScratch_.Get<float>(), workspaceStrideElements);
        }
        CommitOutput(0, accumulator, innerSize);
    }

    __aicore__ inline void ProcessRaCompactTiled()
    {
        const uint64_t reduceSize = tiling_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim1);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(reduceSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = bufShared_.Get<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), innerSize);
        Duplicate<float>(accumulator, 0.0f, tiling_->alignedInner);
        PipeBarrier<PIPE_V>();

        uint64_t reduceBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, coreEnd - reduceBase));
        const uint32_t staggerGroups = tiling_->raStaggerGroups;
        if (staggerGroups > 1 && rows == tiling_->tilePrimary) {
            const uint64_t coreRows = coreEnd - coreStart;
            const uint64_t rounds = (coreRows + tiling_->tilePrimary - 1) /
                tiling_->tilePrimary;
            const uint32_t tailRows = static_cast<uint32_t>(
                coreRows - (rounds - 1) * tiling_->tilePrimary);
            const uint32_t shiftBudget = tiling_->tilePrimary - tailRows;
            const uint32_t group = GetBlockIdx() % staggerGroups;
            const uint32_t firstShift = shiftBudget * group / (staggerGroups - 1);
            rows = tiling_->tilePrimary - firstShift;
        }
        LoadCompactTile(reduceBase * innerSize, rows, innerSize);
        while (reduceBase < coreEnd) {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            const uint64_t nextBase = reduceBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
            ComputeSquare(inputLocal, rows * innerSize);
            AccumulateRa(rows, innerSize, SquaredView(inputLocal),
                accumulator, gatherOffsets);
            qIn_.FreeTensor(inputLocal);
            if (nextRows > 0) {
                LoadCompactTile(nextBase * innerSize, nextRows, innerSize);
            }
            reduceBase = nextBase;
            rows = nextRows;
        }

        const uint64_t workspaceStrideElements = tiling_->workspaceStride / sizeof(float);
        SyncVToMte3();
        DataCopy(gmWorkspace_[static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements],
            accumulator, tiling_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<T> mergeInput = qIn_.AllocTensor<T>();
        qIn_.EnQue(mergeInput);
        mergeInput = qIn_.DeQue<T>();
        ConsolidateCores(accumulator, mergeInput, workspaceStrideElements);
        qIn_.FreeTensor(mergeInput);
        SyncVToMte3();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(innerSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(gmOut_, accumulator, copyParams);
    }

    __aicore__ inline void ProcessRaCompactPhase()
    {
        const uint64_t reduceSize = tiling_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim1);
        const uint32_t phaseRows = tiling_->compactRaPhaseRows;
        const uint32_t phaseElements = phaseRows * innerSize;
        const bool segmented =
            tiling_->compactRaPhaseSegments == RA_PHASE_SEGMENT_LIMIT;
        const uint32_t checkpointGroups = tiling_->compactRaPhaseCheckpointGroups;
        const uint64_t workspaceStrideElements = tiling_->workspaceStride / sizeof(float);
        const uint64_t workspaceOffset =
            static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements;
        LocalTensor<float> phaseAccumulator = bufAcc_.Get<float>();
        Duplicate<float>(phaseAccumulator, 0.0f,
            RA_PHASE_LANES * phaseElements);
        PipeBarrier<PIPE_V>();

        uint32_t groupIndex = 0;
        bool checkpointStored = false;
        if (tiling_->compactRaPhaseScheduleMode == RA_PHASE_WAVE_MODE) {
            const uint64_t tileRows = tiling_->tilePrimary;
            const uint64_t fullTiles = reduceSize / tileRows;
            const uint64_t tailRows = reduceSize % tileRows;
            const uint64_t physicalBlockIdx = GetBlockIdx();
            const bool ownsTail = tailRows != 0 &&
                physicalBlockIdx == fullTiles % tiling_->blockDim;
            uint64_t tileIndex = physicalBlockIdx;
            uint64_t reduceBase = tileIndex * tileRows;
            uint32_t rows = static_cast<uint32_t>(tileRows);
            bool currentIsTail = false;
            bool tailIssued = false;
            LoadCompactTile(reduceBase * innerSize, rows, innerSize);
            do {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                uint64_t nextBase = 0;
                uint64_t nextTileIndex = tileIndex;
                uint32_t nextRows = 0;
                bool nextIsTail = false;
                if (!currentIsTail) {
                    nextTileIndex = tileIndex + tiling_->blockDim;
                    if (nextTileIndex < fullTiles) {
                        nextBase = nextTileIndex * tileRows;
                        nextRows = static_cast<uint32_t>(tileRows);
                    } else if (ownsTail && !tailIssued) {
                        nextBase = fullTiles * tileRows;
                        nextRows = static_cast<uint32_t>(tailRows);
                        nextIsTail = true;
                        tailIssued = true;
                    }
                }
                if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                    LoadCompactTile(nextBase * innerSize, nextRows, innerSize);
                }
                AccumulateRaPhases(inputLocal, rows, innerSize,
                    phaseRows, phaseElements, phaseAccumulator, groupIndex);
                qIn_.FreeTensor(inputLocal);
                if (segmented && !checkpointStored && groupIndex == checkpointGroups) {
                    LocalTensor<float> phaseWork = bufScratch_.Get<float>();
                    LocalTensor<float> checkpointResult = FinalizeRaPhases(
                        innerSize, phaseRows, phaseElements, phaseAccumulator, phaseWork);
                    LocalTensor<float> checkpoint = bufCheckpoint_.Get<float>();
                    Adds<float>(checkpoint, checkpointResult, 0.0f, tiling_->alignedInner);
                    Duplicate<float>(phaseAccumulator, 0.0f,
                        RA_PHASE_LANES * phaseElements);
                    PipeBarrier<PIPE_V>();
                    groupIndex = 0;
                    checkpointStored = true;
                }
                if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                    LoadCompactTile(nextBase * innerSize, nextRows, innerSize);
                }
                reduceBase = nextBase;
                tileIndex = nextTileIndex;
                rows = nextRows;
                currentIsTail = nextIsTail;
            } while (rows > 0);
        } else {
            uint64_t coreStart = 0;
            uint64_t coreEnd = 0;
            if (tiling_->compactRaPhaseAlignedSplit == 0 ||
                !SplitPhaseGroups(reduceSize, phaseRows, coreStart, coreEnd)) {
                SplitWork(reduceSize, coreStart, coreEnd);
            }
            uint64_t reduceBase = coreStart;
            uint32_t rows = static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - reduceBase));
            LoadCompactTile(reduceBase * innerSize, rows, innerSize);
            while (reduceBase < coreEnd) {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                    Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
                if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                    LoadCompactTile(nextBase * innerSize, nextRows, innerSize);
                }
                AccumulateRaPhases(inputLocal, rows, innerSize,
                    phaseRows, phaseElements, phaseAccumulator, groupIndex);
                qIn_.FreeTensor(inputLocal);
                if (segmented && !checkpointStored && groupIndex == checkpointGroups) {
                    LocalTensor<float> phaseWork = bufScratch_.Get<float>();
                    LocalTensor<float> checkpointResult = FinalizeRaPhases(
                        innerSize, phaseRows, phaseElements, phaseAccumulator, phaseWork);
                    LocalTensor<float> checkpoint = bufCheckpoint_.Get<float>();
                    Adds<float>(checkpoint, checkpointResult, 0.0f, tiling_->alignedInner);
                    Duplicate<float>(phaseAccumulator, 0.0f,
                        RA_PHASE_LANES * phaseElements);
                    PipeBarrier<PIPE_V>();
                    groupIndex = 0;
                    checkpointStored = true;
                }
                if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                    LoadCompactTile(nextBase * innerSize, nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
        }

        if (segmented) {
            LocalTensor<float> phaseWork = bufScratch_.Get<float>();
            LocalTensor<float> result = FinalizeRaPhases(innerSize, phaseRows,
                phaseElements, phaseAccumulator, phaseWork);
            Adds<float>(phaseAccumulator, result, 0.0f, tiling_->alignedInner);
            PipeBarrier<PIPE_V>();
            if (checkpointStored) {
                Add<float>(phaseAccumulator, phaseAccumulator, bufCheckpoint_.Get<float>(),
                    tiling_->alignedInner);
                PipeBarrier<PIPE_V>();
            }
        } else {
            LocalTensor<T> phaseWorkTensor = qIn_.AllocTensor<T>();
            qIn_.EnQue(phaseWorkTensor);
            phaseWorkTensor = qIn_.DeQue<T>();
            LocalTensor<float> result = FinalizeRaPhases(innerSize, phaseRows,
                phaseElements, phaseAccumulator, phaseWorkTensor);
            Adds<float>(phaseAccumulator, result, 0.0f, tiling_->alignedInner);
            PipeBarrier<PIPE_V>();
            qIn_.FreeTensor(phaseWorkTensor);
        }

        SyncVToMte3();
        DataCopy(gmWorkspace_[workspaceOffset], phaseAccumulator, tiling_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<T> mergeInput = qIn_.AllocTensor<T>();
        ConsolidateCores(phaseAccumulator, mergeInput, workspaceStrideElements);
        qIn_.FreeTensor(mergeInput);
        SyncVToMte3();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(innerSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(gmOut_, phaseAccumulator, copyParams);
    }

    __aicore__ inline void AccumulateRa(uint32_t rows, uint32_t inner,
        LocalTensor<float> source, LocalTensor<float> accumulator,
        LocalTensor<uint32_t> gatherOffsets)
    {
        const uint32_t phaseRows = 8U / GCD(inner, 8U);
        const uint32_t groupElements = phaseRows * inner;
        const uint32_t fullGroups = rows / phaseRows;
        const uint32_t tailRows = rows % phaseRows;
        uint32_t activeGroups = fullGroups;
        while (activeGroups > 1) {
            const uint32_t leftGroups = (activeGroups + 1U) / 2U;
            const uint32_t pairGroups = activeGroups / 2U;
            Add<float>(source, source, source[leftGroups * groupElements],
                pairGroups * groupElements);
            PipeBarrier<PIPE_V>();
            activeGroups = leftGroups;
        }
        if (fullGroups > 0 && tailRows > 0) {
            Add<float>(source, source, source[fullGroups * groupElements], tailRows * inner);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> gatherLocal = bufReduce_.Get<float>();
        const uint32_t remainingRows = fullGroups > 0 ? phaseRows : tailRows;
        for (uint32_t row = 0; row < remainingRows; ++row) {
            for (uint32_t column = 0; column < inner; column += GATHER_CAP) {
                const uint32_t count = static_cast<uint32_t>(
                    Smallest(GATHER_CAP, inner - column));
                Gather<float>(gatherLocal[column], source, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Add<float>(accumulator[column], accumulator[column], gatherLocal[column], count);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void AccumulateRaPhases(LocalTensor<float> inputLocal,
        uint32_t rows, uint32_t inner, uint32_t phaseRows, uint32_t phaseElements,
        LocalTensor<float> phaseAccumulator, uint32_t& groupIndex)
    {
        uint32_t issuedSinceBarrier = 0;
        for (uint32_t row = 0; row < rows; row += phaseRows) {
            const uint32_t currentRows = static_cast<uint32_t>(
                Smallest(phaseRows, rows - row));
            const uint32_t lane = groupIndex % RA_PHASE_LANES;
            MulAddDst<float, float>(phaseAccumulator[lane * phaseElements],
                inputLocal[row * inner], inputLocal[row * inner],
                static_cast<int32_t>(currentRows * inner));
            ++groupIndex;
            ++issuedSinceBarrier;
            if (issuedSinceBarrier == RA_PHASE_LANES) {
                PipeBarrier<PIPE_V>();
                issuedSinceBarrier = 0;
            }
        }
        if (issuedSinceBarrier != 0) {
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline LocalTensor<float> FinalizeRaPhases(uint32_t inner,
        uint32_t phaseRows, uint32_t phaseElements, LocalTensor<float> phaseAccumulator,
        LocalTensor<float> phaseWork)
    {
        const uint32_t alignedInner = PadTo8(inner);
        if constexpr (RA_PHASE_LANES > 1) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[phaseElements], phaseElements);
        }
        LocalTensor<uint32_t> gatherOffsets = phaseWork.ReinterpretCast<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), inner);
        LocalTensor<float> result = phaseWork[alignedInner];
        LocalTensor<float> gatherLocal = phaseWork[2U * alignedInner];
        Duplicate<float>(result, 0.0f, alignedInner);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < inner; column += GATHER_CAP) {
            const uint32_t count = static_cast<uint32_t>(
                Smallest(GATHER_CAP, inner - column));
            Gather<float>(result[column], phaseAccumulator, gatherOffsets[column], 0, count);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < phaseRows; ++row) {
            for (uint32_t column = 0; column < inner; column += GATHER_CAP) {
                const uint32_t count = static_cast<uint32_t>(
                    Smallest(GATHER_CAP, inner - column));
                Gather<float>(gatherLocal[column], phaseAccumulator, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Add<float>(result[column], result[column], gatherLocal[column], count);
                PipeBarrier<PIPE_V>();
            }
        }
        return result;
    }

    __aicore__ inline LocalTensor<float> RowReduceRa(uint32_t rows, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = bufReduce_.Get<float>();
        LocalTensor<uint8_t> sharedTmp = ReduceScratch();
        uint32_t shape[2] = {rows, tiling_->alignedInner};
        ReduceSum<float, Pattern::Reduce::RA, true>(
            sumLocal, source, sharedTmp, shape, true);
        PipeBarrier<PIPE_V>();
        return sumLocal;
    }

    __aicore__ inline LocalTensor<float> FoldRaInPlace(uint32_t rows, LocalTensor<float> source)
    {
        BinaryRepeatParams repeatParams;
        uint32_t splitRows = 1;
        while ((splitRows << 1U) <= rows) {
            splitRows <<= 1U;
        }
        const uint32_t remainRows = rows - splitRows;
        SetMaskCount();
        if (remainRows > 0) {
            SetVectorMask<float, MaskMode::COUNTER>(
                0, remainRows * tiling_->alignedInner);
            Add<float, false>(source, source,
                source[splitRows * tiling_->alignedInner], MASK_PLACEHOLDER, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        while (splitRows > 1) {
            splitRows >>= 1U;
            SetVectorMask<float, MaskMode::COUNTER>(
                0, splitRows * tiling_->alignedInner);
            Add<float, false>(source, source,
                source[splitRows * tiling_->alignedInner], MASK_PLACEHOLDER, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        ResetMask();
        return source;
    }

    __aicore__ inline void ConsolidateCores(LocalTensor<float> accumulator,
        LocalTensor<float> mergeLocal, uint64_t workspaceStrideElements)
    {
        const uint32_t rowBytes = tiling_->alignedInner * sizeof(float);
        const uint32_t srcStride = static_cast<uint32_t>(tiling_->workspaceStride - rowBytes);
        uint32_t mergeRows = tiling_->tilePrimary;
        if (tiling_->compactRa != 0) {
            const uint64_t compactElements = static_cast<uint64_t>(tiling_->tilePrimary) *
                tiling_->dim1;
            mergeRows = static_cast<uint32_t>(compactElements / tiling_->alignedInner);
            if (mergeRows == 0) {
                mergeRows = 1;
            }
        }
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        for (uint32_t coreBase = 1; coreBase < tiling_->blockDim;
             coreBase += mergeRows) {
            const uint32_t rows = static_cast<uint32_t>(Smallest(
                mergeRows, tiling_->blockDim - coreBase));
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), rowBytes, srcStride, 0, 0};
            DataCopyPad(mergeLocal,
                gmWorkspace_[static_cast<uint64_t>(coreBase) * workspaceStrideElements],
                copyParams, padParams);
            SyncMte2ToV();
            LocalTensor<float> partial = tiling_->compactRa != 0 ?
                FoldRaInPlace(rows, mergeLocal) : RowReduceRa(rows, mergeLocal);
            Add<float>(accumulator, accumulator, partial, tiling_->alignedInner);
            SyncVToMte2();
        }
    }

    // RAR methods
    __aicore__ inline void ExecuteRar()
    {
        if constexpr (std::is_same<T, half>::value) {
            if (tiling_->compactRar != 0) {
                ProcessRarCompact();
                return;
            }
        }
        const uint64_t outerReduce = tiling_->dim0;
        const uint64_t normalSize = tiling_->dim1;
        const uint32_t innerReduce = static_cast<uint32_t>(tiling_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(normalSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        for (uint64_t normalBase = coreStart; normalBase < coreEnd; normalBase += tiling_->tilePrimary) {
            const uint32_t rows = static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - normalBase));
            Duplicate<float>(accumulator, 0.0f, PadTo8(rows));
            PipeBarrier<PIPE_V>();
            uint64_t outer = 0;
            LoadStridedTile(normalBase * innerReduce, rows, innerReduce, innerReduce);
            while (outer < outerReduce) {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                const uint64_t nextOuter = outer + 1;
                if (nextOuter < outerReduce && tiling_->inputBufferNum == 2) {
                    LoadStridedTile((nextOuter * normalSize + normalBase) * innerReduce,
                        rows, innerReduce, innerReduce);
                }
                ComputeSquare(inputLocal, rows * tiling_->alignedInner);
                LocalTensor<float> partial = RowReduceAr(rows, SquaredView(inputLocal));
                qIn_.FreeTensor(inputLocal);
                Add<float>(accumulator, accumulator, partial, rows);
                PipeBarrier<PIPE_V>();
                if (nextOuter < outerReduce && tiling_->inputBufferNum == 1) {
                    LoadStridedTile((nextOuter * normalSize + normalBase) * innerReduce,
                        rows, innerReduce, innerReduce);
                }
                outer = nextOuter;
            }
            CommitOutput(normalBase, accumulator, rows);
        }
    }

    __aicore__ inline void ProcessRarCompact()
    {
        const uint64_t outerReduce = tiling_->dim0;
        const uint64_t normalSize = tiling_->dim1;
        const uint32_t innerReduce = static_cast<uint32_t>(tiling_->dim2);
        LocalTensor<float> matrixAccumulator = bufScratch_.Get<float>();
        if (tiling_->rarScheduleMode == RAR_WAVE_MODE) {
            const uint64_t blockIdx = GetBlockIdx();
            const uint64_t secondGroupBegin =
                (static_cast<uint64_t>(tiling_->blockDim) + 1U) / 2U;
            const uint64_t outerStart =
                blockIdx >= secondGroupBegin && outerReduce > 1U ? outerReduce / 2U : 0U;
            const uint64_t tileRows = tiling_->tilePrimary;
            const uint64_t waveRows = static_cast<uint64_t>(tiling_->blockDim) * tileRows;
            const uint64_t fullWaves = normalSize / waveRows;
            const uint64_t prefixRows = fullWaves * waveRows;
            uint64_t normalBase = blockIdx * tileRows;
            for (uint64_t wave = 0; wave < fullWaves; ++wave) {
                ProcessRarCompactTile(normalBase, tiling_->tilePrimary,
                    outerReduce, normalSize, innerReduce, outerStart, matrixAccumulator);
                normalBase += waveRows;
            }
            const uint64_t tailRows = normalSize - prefixRows;
            uint64_t tailStart = 0;
            uint64_t tailEnd = 0;
            SplitWork(tailRows, tailStart, tailEnd);
            if (tailStart < tailEnd) {
                ProcessRarCompactTile(prefixRows + tailStart,
                    static_cast<uint32_t>(tailEnd - tailStart), outerReduce,
                    normalSize, innerReduce, outerStart, matrixAccumulator);
            }
            return;
        }

        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(normalSize, coreStart, coreEnd);
        for (uint64_t normalBase = coreStart; normalBase < coreEnd; normalBase += tiling_->tilePrimary) {
            const uint32_t rows = static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - normalBase));
            ProcessRarCompactTile(normalBase, rows, outerReduce,
                normalSize, innerReduce, 0, matrixAccumulator);
        }
    }

    __aicore__ inline void ProcessRarCompactTile(uint64_t normalBase, uint32_t rows,
        uint64_t outerReduce, uint64_t normalSize, uint32_t innerReduce,
        uint64_t outerStart, LocalTensor<float> matrixAccumulator)
    {
        const uint32_t matrixElements = rows * innerReduce;
        Duplicate<float>(matrixAccumulator, 0.0f, matrixElements);
        PipeBarrier<PIPE_V>();
        uint64_t outer = outerStart;
        LoadCompactTile((outer * normalSize + normalBase) * innerReduce,
            rows, innerReduce);
        do {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            uint64_t nextOuter = outer + 1U;
            if (nextOuter == outerReduce) {
                nextOuter = 0;
            }
            const bool hasNext = nextOuter != outerStart;
            if (hasNext && tiling_->inputBufferNum == 2) {
                LoadCompactTile((nextOuter * normalSize + normalBase) * innerReduce,
                    rows, innerReduce);
            }
            MulAddDst<float, T>(
                matrixAccumulator, inputLocal, inputLocal, static_cast<int32_t>(matrixElements));
            qIn_.FreeTensor(inputLocal);
            PipeBarrier<PIPE_V>();
            if (hasNext && tiling_->inputBufferNum == 1) {
                LoadCompactTile((nextOuter * normalSize + normalBase) * innerReduce,
                    rows, innerReduce);
            }
            outer = nextOuter;
        } while (outer != outerStart);
        LocalTensor<float> partial = ReducePacked(rows, innerReduce, matrixAccumulator);
        CommitOutput(normalBase, partial, rows);
    }

    // ARA methods
    __aicore__ inline void ExecuteAra()
    {
        if (tiling_->compactAra == 2) {
            ProcessAraPhase();
            return;
        }
        if (tiling_->compactAra == 1) {
            ProcessAraCompact();
            return;
        }
        const uint64_t outerSize = tiling_->dim0;
        const uint64_t reduceSize = tiling_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(outerSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(accumulator, 0.0f, tiling_->alignedInner);
            PipeBarrier<PIPE_V>();
            uint64_t reduceBase = 0;
            uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, reduceSize));
            LoadStridedTile(outer * reduceSize * innerSize, rows, innerSize, innerSize);
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? static_cast<uint32_t>(
                    Smallest(tiling_->tilePrimary, reduceSize - nextBase)) : 0;
                if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                    LoadStridedTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize, innerSize);
                }
                ComputeSquare(inputLocal, rows * tiling_->alignedInner);
                LocalTensor<float> partial = RowReduceRa(rows, SquaredView(inputLocal));
                qIn_.FreeTensor(inputLocal);
                Add<float>(accumulator, accumulator, partial, tiling_->alignedInner);
                PipeBarrier<PIPE_V>();
                if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                    LoadStridedTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            CommitOutput(outer * innerSize, accumulator, innerSize);
        }
    }

    __aicore__ inline void ProcessAraCompact()
    {
        const uint64_t outerSize = tiling_->dim0;
        const uint64_t reduceSize = tiling_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(outerSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = bufAcc_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = bufShared_.Get<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), innerSize);
        PipeBarrier<PIPE_V>();
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(accumulator, 0.0f, PadTo8(innerSize));
            PipeBarrier<PIPE_V>();
            uint64_t reduceBase = 0;
            uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, reduceSize));
            LoadCompactTile(outer * reduceSize * innerSize, rows, innerSize);
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? static_cast<uint32_t>(
                    Smallest(tiling_->tilePrimary, reduceSize - nextBase)) : 0;
                if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                    LoadCompactTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                ComputeSquare(inputLocal, rows * tiling_->alignedInner);
                AccumulateRa(rows, innerSize, SquaredView(inputLocal),
                    accumulator, gatherOffsets);
                qIn_.FreeTensor(inputLocal);
                if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                    LoadCompactTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            CommitOutput(outer * innerSize, accumulator, innerSize);
        }
    }

    __aicore__ inline void ProcessAraPhase()
    {
        const uint64_t outerSize = tiling_->dim0;
        const uint64_t reduceSize = tiling_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tiling_->dim2);
        const uint32_t phaseRows = 16U / GCD(innerSize, 16U);
        const uint32_t phaseElements = phaseRows * innerSize;
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(outerSize, coreStart, coreEnd);
        LocalTensor<float> phaseAccumulator = bufAcc_.Get<float>();
        bool inputPrefetched = false;
        uint32_t prefetchedRows = 0;
        if (coreStart < coreEnd) {
            prefetchedRows = SelectARAPhaseRows(
                coreStart, 0, reduceSize, innerSize, phaseRows);
            LoadCompactTile(coreStart * reduceSize * innerSize, prefetchedRows, innerSize);
            inputPrefetched = true;
        }
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(phaseAccumulator, 0.0f,
                ARA_PHASE_LANES * phaseElements);
            PipeBarrier<PIPE_V>();
            uint32_t groupIndex = 0;
            uint64_t reduceBase = 0;
            uint32_t rows = 0;
            if (inputPrefetched) {
                rows = prefetchedRows;
                inputPrefetched = false;
            } else {
                rows = SelectARAPhaseRows(
                    outer, reduceBase, reduceSize, innerSize, phaseRows);
                LoadCompactTile(outer * reduceSize * innerSize, rows, innerSize);
            }
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = qIn_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? SelectARAPhaseRows(
                    outer, nextBase, reduceSize, innerSize, phaseRows) : 0;
                if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                    LoadCompactTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                } else if (nextRows == 0 && tiling_->inputBufferNum == 2 &&
                    outer + 1U < coreEnd) {
                    prefetchedRows = SelectARAPhaseRows(
                        outer + 1U, 0, reduceSize, innerSize, phaseRows);
                    LoadCompactTile((outer + 1U) * reduceSize * innerSize,
                        prefetchedRows, innerSize);
                    inputPrefetched = true;
                }
                AccumulateAraPhases(inputLocal, rows, innerSize, phaseRows,
                    phaseElements, phaseAccumulator, groupIndex);
                qIn_.FreeTensor(inputLocal);
                if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                    LoadCompactTile((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            LocalTensor<float> result = FinalizeAraPhases(
                innerSize, phaseRows, phaseElements, phaseAccumulator);
            CommitOutput(outer * innerSize, result, innerSize);
        }
    }

    __aicore__ inline void AccumulateAraPhases(LocalTensor<T> inputLocal, uint32_t rows,
        uint32_t inner, uint32_t phaseRows, uint32_t phaseElements,
        LocalTensor<float> phaseAccumulator, uint32_t& groupIndex)
    {
        LocalTensor<float> castLocal = bufScratch_.Get<float>();
        for (uint32_t row = 0; row < rows; row += phaseRows) {
            const uint32_t currentRows = static_cast<uint32_t>(
                Smallest(phaseRows, rows - row));
            const uint32_t count = currentRows * inner;
            Cast<float, T>(castLocal, inputLocal[row * inner], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            const uint32_t lane = groupIndex % ARA_PHASE_LANES;
            MulAddDst<float, float>(phaseAccumulator[lane * phaseElements],
                castLocal, castLocal, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            ++groupIndex;
        }
    }

    __aicore__ inline LocalTensor<float> FinalizeAraPhases(uint32_t inner,
        uint32_t phaseRows, uint32_t phaseElements, LocalTensor<float> phaseAccumulator)
    {
        if constexpr (ARA_PHASE_LANES > 1) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[phaseElements], phaseElements);
            PipeBarrier<PIPE_V>();
        }
        const uint32_t remainingRows = 8U / GCD(inner, 8U);
        if (phaseRows > remainingRows) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[remainingRows * inner], remainingRows * inner);
            PipeBarrier<PIPE_V>();
        }

        const uint32_t alignedInner = PadTo8(inner);
        LocalTensor<float> phaseWork = bufScratch_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = phaseWork.ReinterpretCast<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), inner);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> result = phaseWork[alignedInner];
        LocalTensor<float> gatherLocal = phaseWork[2U * alignedInner];
        for (uint32_t column = 0; column < inner; column += GATHER_CAP) {
            const uint32_t count = static_cast<uint32_t>(
                Smallest(GATHER_CAP, inner - column));
            Gather<float>(result[column], phaseAccumulator, gatherOffsets[column], 0, count);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < remainingRows; ++row) {
            for (uint32_t column = 0; column < inner; column += GATHER_CAP) {
                const uint32_t count = static_cast<uint32_t>(
                    Smallest(GATHER_CAP, inner - column));
                Gather<float>(gatherLocal[column], phaseAccumulator, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Add<float>(result[column], result[column], gatherLocal[column], count);
                PipeBarrier<PIPE_V>();
            }
        }
        return result;
    }

    __aicore__ inline uint32_t SelectARAPhaseRows(uint64_t outer, uint64_t reduceBase,
        uint64_t reduceSize, uint32_t inner, uint32_t phaseRows) const
    {
        const uint64_t remaining = reduceSize - reduceBase;
        if (reduceBase == 0) {
            const uint32_t blockElements = ALIGNMENT_BYTES / sizeof(T);
            const uint32_t outerPhase = static_cast<uint32_t>(
                (outer * reduceSize * inner) % blockElements);
            if (outerPhase != 0) {
                for (uint32_t leadingRows = 1; leadingRows < phaseRows; ++leadingRows) {
                    if ((outerPhase + leadingRows * inner) % blockElements == 0) {
                        return static_cast<uint32_t>(Smallest(leadingRows, remaining));
                    }
                }
            }
        }
        return static_cast<uint32_t>(Smallest(tiling_->tilePrimary, remaining));
    }

    // AR methods
    __aicore__ inline void ExecuteAr()
    {
        if (tiling_->compactAr != 0) {
            ProcessArCompact();
            return;
        }
        const uint64_t normalSize = tiling_->dim0;
        const uint64_t reduceSize = tiling_->dim1;
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(normalSize, coreStart, coreEnd);
        if (coreStart >= coreEnd) {
            return;
        }
        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, coreEnd - rowBase));
        LoadStridedTile(rowBase * reduceSize, rows, static_cast<uint32_t>(reduceSize), reduceSize);
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                LoadStridedTile(nextBase * reduceSize, nextRows, static_cast<uint32_t>(reduceSize), reduceSize);
            }
            ComputeSquare(inputLocal, rows * tiling_->alignedInner);
            LocalTensor<float> partial = RowReduceAr(rows, SquaredView(inputLocal));
            qIn_.FreeTensor(inputLocal);
            CommitOutput(rowBase, partial, rows);
            if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                LoadStridedTile(nextBase * reduceSize, nextRows, static_cast<uint32_t>(reduceSize), reduceSize);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void ProcessArCompact()
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            const uint32_t inner = static_cast<uint32_t>(tiling_->dim1);
            const uint32_t packStride = PadTo8(inner / 8U);
            if ((inner & 7U) != 0 && packStride >= 64U) {
                ProcessArCompactDirect();
                return;
            }
        }
        ProcessArCompactGather();
    }

    __aicore__ inline void ProcessArCompactDirect()
    {
        const uint64_t normalSize = tiling_->dim0;
        const uint32_t inner = static_cast<uint32_t>(tiling_->dim1);
        const uint32_t packStride = PadTo8(inner / 8U);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(normalSize, coreStart, coreEnd);
        LocalTensor<float> rowPack = bufPacked_.Get<float>();
        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, coreEnd - rowBase));
        LoadCompactTile(rowBase * inner, rows, inner);
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                LoadCompactTile(nextBase * inner, nextRows, inner);
            }
            ComputeSquare(inputLocal, rows * tiling_->alignedInner);
            LocalTensor<float> source = SquaredView(inputLocal);
            BuildPackedDirect(rowPack, source, rows, inner, packStride);
            const uint32_t alignedRows = PadTo8(rows);
            LocalTensor<float> head = bufReduce_.Get<float>();
            LocalTensor<float> tail = bufReduce_.Get<float>()[alignedRows];
            ReduceRowEdges(head, tail, source, rows, inner);
            qIn_.FreeTensor(inputLocal);
            LocalTensor<float> result = ReducePackedRows(
                rowPack, head, tail, rows, packStride);
            CommitOutput(rowBase, result, rows);
            if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                LoadCompactTile(nextBase * inner, nextRows, inner);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void ProcessArCompactGather()
    {
        const uint64_t normalSize = tiling_->dim0;
        const uint32_t inner = static_cast<uint32_t>(tiling_->dim1);
        const uint32_t packStride = PadTo8(inner / 8U);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitWork(normalSize, coreStart, coreEnd);
        LocalTensor<float> rowPack = bufPacked_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = bufShared_.Get<uint32_t>();
        LocalTensor<float> blockSums = bufBlocks_.Get<float>();
        const uint32_t maxTileElements = tiling_->tilePrimary * inner;
        const uint32_t zeroBlock = (maxTileElements + 63U) / 64U * 8U;
        InitPackOffsets(gatherOffsets, inner, packStride, zeroBlock);
        Duplicate<float>(blockSums[zeroBlock], 0.0f, 8);
        PipeBarrier<PIPE_V>();

        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(Smallest(tiling_->tilePrimary, coreEnd - rowBase));
        LoadCompactTile(rowBase * inner, rows, inner);
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = qIn_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                Smallest(tiling_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tiling_->inputBufferNum == 2) {
                LoadCompactTile(nextBase * inner, nextRows, inner);
            }
            ComputeSquare(inputLocal, rows * tiling_->alignedInner);
            LocalTensor<float> source = SquaredView(inputLocal);
            qIn_.FreeTensor(inputLocal);
            blockSums = CompressBlocks(source, rows, inner);
            BuildPacked(rowPack, blockSums, gatherOffsets, rows, packStride);

            const uint32_t alignedRows = PadTo8(rows);
            LocalTensor<float> head = bufReduce_.Get<float>();
            LocalTensor<float> tail = bufReduce_.Get<float>()[alignedRows];
            ReduceRowEdges(head, tail, source, rows, inner);
            LocalTensor<float> result = ReducePackedRows(
                rowPack, head, tail, rows, packStride);
            CommitOutput(rowBase, result, rows);
            if (nextRows > 0 && tiling_->inputBufferNum == 1) {
                LoadCompactTile(nextBase * inner, nextRows, inner);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline LocalTensor<float> RowReduceAr(uint32_t rows, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = bufReduce_.Get<float>();
        LocalTensor<uint8_t> sharedTmp = ReduceScratch();
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += AR_BATCH_ROWS) {
            const uint32_t currentRows = static_cast<uint32_t>(
                Smallest(AR_BATCH_ROWS, rows - rowBase));
            uint32_t shape[2] = {currentRows, tiling_->alignedInner};
            ReduceSum<float, Pattern::Reduce::AR, true>(sumLocal[rowBase],
                source[rowBase * tiling_->alignedInner], sharedTmp, shape, true);
        }
        PipeBarrier<PIPE_V>();
        return sumLocal;
    }

    __aicore__ inline LocalTensor<float> CompressBlocks(
        LocalTensor<float> source, uint32_t rows, uint32_t inner)
    {
        const uint32_t totalElements = rows * inner;
        const uint32_t fullRepeats = totalElements / 64U;
        const uint32_t tailElements = totalElements % 64U;
        LocalTensor<float> blockSums = bufBlocks_.Get<float>();
        blockSums.SetSize((fullRepeats + (tailElements > 0 ? 1U : 0U)) * 8U);
        for (uint32_t repeatBase = 0; repeatBase < fullRepeats;
             repeatBase += REPEAT_CAP) {
            const uint32_t repeats = static_cast<uint32_t>(
                Smallest(REPEAT_CAP, fullRepeats - repeatBase));
            BlockReduceSum<float>(blockSums[repeatBase * 8U], source[repeatBase * 64U],
                static_cast<int32_t>(repeats), 64, 1, 1, 8);
        }
        if (tailElements > 0) {
            BlockReduceSum<float>(blockSums[fullRepeats * 8U], source[fullRepeats * 64U],
                1, static_cast<int32_t>(tailElements), 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        return blockSums;
    }

    __aicore__ inline void InitPackOffsets(LocalTensor<uint32_t> gatherOffsets,
        uint32_t inner, uint32_t packStride, uint32_t zeroBlock)
    {
        const uint32_t offsetCount = tiling_->tilePrimary * packStride;
        Duplicate<uint32_t>(gatherOffsets, zeroBlock * sizeof(float), offsetCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < tiling_->tilePrimary; ++row) {
            const uint32_t rowBegin = row * inner;
            const uint32_t rowEnd = rowBegin + inner;
            const uint32_t firstBlock = (rowBegin + 7U) / 8U;
            const uint32_t lastBlock = rowEnd / 8U;
            const uint32_t blockCount = lastBlock - firstBlock;
            const uint32_t alignedBlockCount = blockCount / 8U * 8U;
            if (alignedBlockCount > 0) {
                Arange<int32_t>(gatherOffsets[row * packStride].ReinterpretCast<int32_t>(),
                    static_cast<int32_t>(firstBlock * sizeof(float)),
                    static_cast<int32_t>(sizeof(float)), alignedBlockCount);
            }
            if (alignedBlockCount < blockCount) {
                Arange<int32_t>(gatherOffsets[row * packStride + alignedBlockCount].ReinterpretCast<int32_t>(),
                    static_cast<int32_t>((firstBlock + alignedBlockCount) * sizeof(float)),
                    static_cast<int32_t>(sizeof(float)), blockCount - alignedBlockCount);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPacked(LocalTensor<float> rowPack,
        LocalTensor<float> blockSums, LocalTensor<uint32_t> gatherOffsets,
        uint32_t rows, uint32_t packStride)
    {
        const uint32_t count = rows * packStride;
        for (uint32_t base = 0; base < count; base += GATHER_CAP) {
            const uint32_t current = static_cast<uint32_t>(
                Smallest(GATHER_CAP, count - base));
            Gather<float>(rowPack[base], blockSums, gatherOffsets[base], 0, current);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPackedDirect(LocalTensor<float> rowPack,
        LocalTensor<float> source, uint32_t rows, uint32_t inner, uint32_t packStride)
    {
        Duplicate<float>(rowPack, 0.0f, rows * packStride);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rowBegin = row * inner;
            const uint32_t rowEnd = rowBegin + inner;
            const uint32_t firstBlock = (rowBegin + 7U) / 8U;
            const uint32_t lastBlock = rowEnd / 8U;
            const uint32_t blockCount = lastBlock - firstBlock;
            const uint32_t fullGroups = blockCount / 8U;
            const uint32_t tailBlocks = blockCount % 8U;
            if (fullGroups > 0) {
                BlockReduceSum<float>(rowPack[row * packStride],
                    source[firstBlock * 8U], static_cast<int32_t>(fullGroups), 64, 1, 1, 8);
            }
            if (tailBlocks > 0) {
                BlockReduceSum<float>(rowPack[row * packStride + fullGroups * 8U],
                    source[(firstBlock + fullGroups * 8U) * 8U], 1,
                    static_cast<int32_t>(tailBlocks * 8U), 1, 1, 8);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceRowEdges(LocalTensor<float> head,
        LocalTensor<float> tail, LocalTensor<float> source, uint32_t rows, uint32_t inner)
    {
        const uint32_t phaseRows = 8U / GCD(inner, 8U);
        const uint32_t srcRepStride = inner / GCD(inner, 8U);
        const uint32_t alignedRows = PadTo8(rows);
        Duplicate<float>(head, 0.0f, alignedRows);
        Duplicate<float>(tail, 0.0f, alignedRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t firstRow = 0; firstRow < phaseRows && firstRow < rows; ++firstRow) {
            const uint32_t phase = firstRow * inner % 8U;
            const uint32_t endPhase = (phase + inner) % 8U;
            const uint32_t classRows = (rows - firstRow + phaseRows - 1U) / phaseRows;
            for (uint32_t repeatBase = 0; repeatBase < classRows;
                 repeatBase += REPEAT_CAP) {
                const uint32_t repeats = static_cast<uint32_t>(
                    Smallest(REPEAT_CAP, classRows - repeatBase));
                const uint32_t row = firstRow + repeatBase * phaseRows;
                if (phase != 0) {
                    const uint64_t lowBits = (1ULL << (8U - phase)) - 1ULL;
                    uint64_t mask[2] = {lowBits << phase, 0};
                    WholeReduceSum<float>(head[row], source[row * inner - phase], mask,
                        static_cast<int32_t>(repeats), static_cast<int32_t>(phaseRows), 1,
                        static_cast<int32_t>(srcRepStride));
                }
                if (endPhase != 0) {
                    const uint64_t lowBits = (1ULL << endPhase) - 1ULL;
                    uint64_t mask[2] = {lowBits, 0};
                    WholeReduceSum<float>(tail[row], source[row * inner + inner - endPhase], mask,
                        static_cast<int32_t>(repeats), static_cast<int32_t>(phaseRows), 1,
                        static_cast<int32_t>(srcRepStride));
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> ReducePackedRows(LocalTensor<float> rowPack,
        LocalTensor<float> head, LocalTensor<float> tail, uint32_t rows, uint32_t packStride)
    {
        const uint32_t alignedRows = PadTo8(rows);
        LocalTensor<float> result = bufAcc_.Get<float>();
        LocalTensor<float> partial = bufAcc_.Get<float>()[alignedRows];
        for (uint32_t column = 0; column < packStride; column += 64U) {
            const uint32_t count = static_cast<uint32_t>(Smallest(64U, packStride - column));
            LocalTensor<float> destination = column == 0 ? result : partial;
            WholeReduceSum<float>(destination, rowPack[column], static_cast<int32_t>(count),
                static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(packStride / 8U));
            PipeBarrier<PIPE_V>();
            if (column != 0) {
                Add<float>(result, result, partial, rows);
                PipeBarrier<PIPE_V>();
            }
        }
        Add<float>(result, result, head, rows);
        PipeBarrier<PIPE_V>();
        Add<float>(result, result, tail, rows);
        PipeBarrier<PIPE_V>();
        return result;
    }

    // Primitives & Helpers
    __aicore__ inline void SyncVToMte3()
    {
        const event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        const event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        const event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SplitWork(uint64_t total, uint64_t& start, uint64_t& end) const
    {
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t base = total / tiling_->blockDim;
        const uint64_t extra = total % tiling_->blockDim;
        start = blockIdx * base + Smallest(blockIdx, extra);
        end = start + base + (blockIdx < extra ? 1 : 0);
    }

    __aicore__ inline bool SplitPhaseGroups(uint64_t total, uint32_t phaseRows,
        uint64_t& start, uint64_t& end) const
    {
        start = 0;
        end = 0;
        const uint64_t blockDim = tiling_->blockDim;
        const uint64_t physicalBlockIdx = GetBlockIdx();
        if (phaseRows == 0 || blockDim == 0 || physicalBlockIdx >= blockDim) {
            return false;
        }
        const uint64_t lowerHalf = (blockDim + 1U) / 2U;
        uint64_t blockIdx = 0;
        if ((blockDim & 1U) == 0) {
            const uint64_t pairCount = blockDim / 2U;
            const uint64_t pairRank = physicalBlockIdx / 2U;
            const uint64_t pairLowerHalf = (pairCount + 1U) / 2U;
            const uint64_t logicalPairRank = (pairRank & 1U) == 0 ?
                pairRank / 2U : pairLowerHalf + pairRank / 2U;
            blockIdx = (physicalBlockIdx & 1U) == 0 ?
                logicalPairRank : pairCount + logicalPairRank;
        } else {
            blockIdx = (physicalBlockIdx & 1U) == 0 ?
                physicalBlockIdx / 2U : lowerHalf + physicalBlockIdx / 2U;
        }
        const uint64_t fullGroups = total / phaseRows;
        const uint64_t residualRows = total % phaseRows;
        if (fullGroups < blockDim) {
            return false;
        }
        const uint64_t baseGroups = fullGroups / blockDim;
        const uint64_t extraGroups = fullGroups % blockDim;
        const uint64_t groupStart = blockIdx * baseGroups + Smallest(blockIdx, extraGroups);
        const uint64_t groupCount = baseGroups + (blockIdx < extraGroups ? 1U : 0U);
        if (groupStart > fullGroups || groupCount > fullGroups - groupStart) {
            return false;
        }
        start = groupStart * phaseRows;
        end = start + groupCount * phaseRows;
        if (blockIdx + 1U == blockDim) {
            if (end > total - residualRows) {
                return false;
            }
            end += residualRows;
        }
        return start <= end && end <= total;
    }

    __aicore__ inline void LoadStridedTile(uint64_t gmOffset, uint32_t rows,
        uint32_t validInner, uint64_t gmRowStride)
    {
        LocalTensor<T> inputLocal = qIn_.AllocTensor<T>();
        const uint64_t gapBytes = (gmRowStride - validInner) * sizeof(T);
        DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(tiling_->alignedInner - validInner), static_cast<T>(0)};
        if (gapBytes <= COPY_STRIDE_CAP) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(validInner * sizeof(T)),
                static_cast<uint32_t>(gapBytes), 0, 0};
            DataCopyPad(inputLocal, gmIn_[gmOffset], copyParams, padParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(validInner * sizeof(T)), 0, 0, 0};
            for (uint32_t row = 0; row < rows; ++row) {
                DataCopyPad(inputLocal[row * tiling_->alignedInner],
                    gmIn_[gmOffset + static_cast<uint64_t>(row) * gmRowStride], copyParams, padParams);
            }
        }
        qIn_.EnQue(inputLocal);
    }

    __aicore__ inline void LoadCompact(LocalTensor<T> inputLocal,
        uint64_t gmOffset, uint32_t rows, uint32_t validInner)
    {
        const uint32_t count = rows * validInner;
        const uint32_t blockElements = ALIGNMENT_BYTES / sizeof(T);
        const uint32_t alignedCount = (count + blockElements - 1) / blockElements * blockElements;
        const uint32_t copyBytes = count * sizeof(T);
        const bool gmAligned = (gmOffset * sizeof(T)) % ALIGNMENT_BYTES == 0;
        if (gmAligned && copyBytes % ALIGNMENT_BYTES == 0) {
            DataCopy(inputLocal, gmIn_[gmOffset], count);
            return;
        }
        DataCopyExtParams copyParams{
            1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(alignedCount - count), static_cast<T>(0)};
        DataCopyPad(inputLocal, gmIn_[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void LoadCompactTile(uint64_t gmOffset, uint32_t rows, uint32_t validInner)
    {
        LocalTensor<T> inputLocal = qIn_.AllocTensor<T>();
        LoadCompact(inputLocal, gmOffset, rows, validInner);
        qIn_.EnQue(inputLocal);
    }

    __aicore__ inline void ComputeSquare(LocalTensor<T> inputLocal, uint32_t count)
    {
        if constexpr (std::is_same<T, float>::value) {
            Mul<float>(inputLocal, inputLocal, inputLocal, count);
        } else {
            LocalTensor<float> workLocal = bufScratch_.Get<float>();
            Cast<float, T>(workLocal, inputLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul<float>(workLocal, workLocal, workLocal, count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<T> LoadAndSquare(uint64_t gmOffset, uint32_t rows,
        uint32_t validInner, uint64_t gmRowStride)
    {
        LoadStridedTile(gmOffset, rows, validInner, gmRowStride);
        LocalTensor<T> inputLocal = qIn_.DeQue<T>();
        ComputeSquare(inputLocal, rows * tiling_->alignedInner);
        return inputLocal;
    }

    __aicore__ inline LocalTensor<float> SquaredView(const LocalTensor<T>& inputLocal)
    {
        if constexpr (std::is_same<T, float>::value) {
            return inputLocal;
        }
        return bufScratch_.Get<float>();
    }

    __aicore__ inline LocalTensor<uint8_t> ReduceScratch()
    {
        if (tiling_->sharedTmpBytes > 0) {
            return bufShared_.Get<uint8_t>();
        }
        const uint32_t vectorElements = tiling_->tilePrimary > tiling_->alignedInner ?
            tiling_->tilePrimary : tiling_->alignedInner;
        const uint32_t scratchOffset = PadTo8(vectorElements) * sizeof(float);
        return bufReduce_.Get<uint8_t>()[scratchOffset];
    }

    __aicore__ inline LocalTensor<float> ReducePacked(
        uint32_t rows, uint32_t inner, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = bufReduce_.Get<float>();
        LocalTensor<float> partialLocal = bufAcc_.Get<float>();
        const uint32_t phaseStep = 8U / GCD(inner, 8U);
        const uint32_t srcRepStride = inner / GCD(inner, 8U);
        const uint32_t chunkCount = (inner + 7U + 63U) / 64U;
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            if (chunk > 0) {
                Duplicate<float>(partialLocal, 0.0f, PadTo8(rows));
                PipeBarrier<PIPE_V>();
            }
            const uint32_t chunkBegin = chunk * 64U;
            const uint32_t chunkEnd = chunkBegin + 64U;
            for (uint32_t firstRow = 0; firstRow < phaseStep && firstRow < rows; ++firstRow) {
                const uint32_t phase = firstRow * inner % 8U;
                const uint32_t validBegin = phase > chunkBegin ? phase : chunkBegin;
                const uint32_t rowEnd = phase + inner;
                const uint32_t validEnd = rowEnd < chunkEnd ? rowEnd : chunkEnd;
                if (validBegin >= validEnd) {
                    continue;
                }
                const uint32_t maskOffset = validBegin - chunkBegin;
                const uint32_t maskCount = validEnd - validBegin;
                const uint64_t lowBits = maskCount == 64U ? UINT64_MAX : ((1ULL << maskCount) - 1ULL);
                uint64_t mask[2] = {lowBits << maskOffset, 0};
                const uint32_t phaseRows = (rows - firstRow + phaseStep - 1U) / phaseStep;
                for (uint32_t repeatBase = 0; repeatBase < phaseRows;
                     repeatBase += REPEAT_CAP) {
                    const uint32_t repeats = static_cast<uint32_t>(
                        Smallest(REPEAT_CAP, phaseRows - repeatBase));
                    const uint32_t row = firstRow + repeatBase * phaseStep;
                    const uint32_t alignedBase = row * inner - phase;
                    if (chunk == 0) {
                        WholeReduceSum<float>(sumLocal[row], source[alignedBase + chunkBegin], mask,
                            static_cast<int32_t>(repeats), static_cast<int32_t>(phaseStep), 1,
                            static_cast<int32_t>(srcRepStride));
                    } else {
                        WholeReduceSum<float>(partialLocal[row], source[alignedBase + chunkBegin], mask,
                            static_cast<int32_t>(repeats), static_cast<int32_t>(phaseStep), 1,
                            static_cast<int32_t>(srcRepStride));
                    }
                }
            }
            PipeBarrier<PIPE_V>();
            if (chunk > 0) {
                Add<float>(sumLocal, sumLocal, partialLocal, rows);
                PipeBarrier<PIPE_V>();
            }
        }
        return sumLocal;
    }

    __aicore__ inline void CommitOutput(uint64_t gmOffset, LocalTensor<float> source, uint32_t count)
    {
        LocalTensor<T> outputLocal = qOut_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds<float>(outputLocal, source, 0.0f, count);
        } else {
            Cast<T, float>(outputLocal, source, RoundMode::CAST_RINT, count);
        }
        qOut_.EnQue(outputLocal);
        outputLocal = qOut_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(gmOut_[gmOffset], outputLocal, copyParams);
        qOut_.FreeTensor(outputLocal);
    }

    __aicore__ inline uint64_t DecodeNormalOffset(uint64_t linear, bool skipLastNormal) const
    {
        uint64_t base = 0;
        for (int32_t dim = static_cast<int32_t>(tiling_->rank) - 1; dim >= 0; --dim) {
            const bool isReduce = (tiling_->axisMask & (1U << dim)) != 0;
            if (isReduce || (skipLastNormal && dim == static_cast<int32_t>(tiling_->rank) - 1)) {
                continue;
            }
            const uint64_t coordinate = linear % tiling_->shape[dim];
            linear /= tiling_->shape[dim];
            base += coordinate * tiling_->stride[dim];
        }
        return base;
    }

    __aicore__ inline uint64_t DecodeReducedOffset(uint64_t linear, bool skipLastReduce) const
    {
        uint64_t offset = 0;
        for (int32_t dim = static_cast<int32_t>(tiling_->rank) - 1; dim >= 0; --dim) {
            const bool isReduce = (tiling_->axisMask & (1U << dim)) != 0;
            if (!isReduce || (skipLastReduce && dim == static_cast<int32_t>(tiling_->rank) - 1)) {
                continue;
            }
            const uint64_t coordinate = linear % tiling_->shape[dim];
            linear /= tiling_->shape[dim];
            offset += coordinate * tiling_->stride[dim];
        }
        return offset;
    }

public:
    __aicore__ inline void Initialize(GM_ADDR input, GM_ADDR output, GM_ADDR userWorkspace,
        const SquareSumV1TilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        pipe_ = pipe;
        gmIn_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        if (tiling_->bypassL2 != 0) {
            gmIn_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if (userWorkspace != nullptr) {
            gmWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace));
        }

        const bool compactAr = tiling_->compactAr != 0;
        const bool compactRa = tiling_->compactRa != 0;
        const bool phaseRa = tiling_->compactRa == RA_PHASE;
        const bool phaseAra = tiling_->compactAra == 2;
        const uint32_t inputInner = compactRa ?
            static_cast<uint32_t>(tiling_->dim1) : tiling_->alignedInner;
        const uint32_t tileElements = tiling_->tilePrimary * inputInner;
        const uint32_t vectorElements = tiling_->tilePrimary > tiling_->alignedInner ?
            tiling_->tilePrimary : tiling_->alignedInner;
        const uint32_t inputExtraBytes = (phaseAra || phaseRa) ? ALIGNMENT_BYTES : BUFFER_PADDING_BYTES;
        pipe_->InitBuffer(qIn_, tiling_->inputBufferNum,
            tileElements * sizeof(T) + inputExtraBytes);
        const uint32_t phaseInner = phaseRa ?
            static_cast<uint32_t>(tiling_->dim1) : tiling_->alignedInner;
        const uint32_t phaseRows = phaseAra ?
            16U / GCD(phaseInner, 16U) : (phaseRa ? tiling_->compactRaPhaseRows : 0U);
        const uint32_t phaseElements = phaseRows * phaseInner;
        if constexpr (!std::is_same<T, float>::value) {
            const uint32_t phaseWorkElements = 3U * PadTo8(tiling_->alignedInner);
            const uint32_t workElements = phaseAra ?
                (phaseElements > phaseWorkElements ? phaseElements : phaseWorkElements) : tileElements;
            pipe_->InitBuffer(bufScratch_, workElements * sizeof(float) +
                (phaseAra ? 0U : BUFFER_PADDING_BYTES));
        }
        const uint32_t compactArRows = 2 * PadTo8(tiling_->tilePrimary);
        const uint32_t sumElements = compactAr ? compactArRows : vectorElements;
        if (!phaseAra && !phaseRa) {
            pipe_->InitBuffer(bufReduce_, sumElements * sizeof(float) + BUFFER_PADDING_BYTES);
        }
        const uint32_t accElements = compactAr ? compactArRows : (phaseAra ?
            ARA_PHASE_LANES * phaseElements : (phaseRa ?
            RA_PHASE_LANES * phaseElements : vectorElements));
        pipe_->InitBuffer(bufAcc_, accElements * sizeof(float) +
            ((phaseAra || phaseRa) ? 0U : BUFFER_PADDING_BYTES));
        if constexpr (std::is_same<T, float>::value) {
            if (phaseRa &&
                tiling_->compactRaPhaseSegments == RA_PHASE_SEGMENT_LIMIT) {
                pipe_->InitBuffer(bufScratch_,
                    3U * PadTo8(tiling_->alignedInner) * sizeof(float));
                pipe_->InitBuffer(bufCheckpoint_,
                    PadTo8(tiling_->alignedInner) * sizeof(float));
            }
        }
        if (tiling_->sharedTmpBytes > 0) {
            pipe_->InitBuffer(bufShared_, tiling_->sharedTmpBytes);
        }
        if (compactAr) {
            const uint32_t blockSumElements = (tileElements + 63U) / 64U * 8U + 8U;
            pipe_->InitBuffer(bufBlocks_, blockSumElements * sizeof(float) + BUFFER_PADDING_BYTES);
            const uint32_t packStride = PadTo8(tiling_->alignedInner / 8U);
            pipe_->InitBuffer(bufPacked_,
                tiling_->tilePrimary * packStride * sizeof(float) + BUFFER_PADDING_BYTES);
        }
        if (!compactRa) {
            const uint32_t outputElements = compactAr ? tiling_->tilePrimary :
                (phaseAra ? tiling_->alignedInner : vectorElements);
            pipe_->InitBuffer(qOut_, 1,
                ((outputElements * sizeof(T) + ALIGNMENT_BYTES - 1) / ALIGNMENT_BYTES) * ALIGNMENT_BYTES);
        }
    }

    __aicore__ inline void Execute()
    {
        switch (tiling_->topology) {
            case LAYOUT_AR:
                ExecuteAr();
                break;
            case LAYOUT_ARA:
                ExecuteAra();
                break;
            case LAYOUT_RAR:
                ExecuteRar();
                break;
            case LAYOUT_MISC:
                ExecuteFallback();
                break;
            default:
                break;
        }
    }

    __aicore__ inline void ExecuteSynchronized()
    {
        ExecuteRa();
    }

    __aicore__ inline void ExecuteCompactDirect()
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            ProcessArCompactDirect();
        }
    }
};
}

template <uint32_t schMode>
__global__ __aicore__ void square_sum_v1(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(SQUARE_SUM_KEY_SYNC, KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (schMode == SQUARE_SUM_KEY_SYNC) {
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        AscendC::TPipe pipe;
        nce_squaresum::SquareSumComputeEngine<DTYPE_INPUT> op;
        op.Initialize(input, output, userWorkspace, &tilingData, &pipe);
        op.ExecuteSynchronized();
        return;
    }
#if defined(ORIG_DTYPE_INPUT) && defined(DT_BF16) && (ORIG_DTYPE_INPUT == DT_BF16)
    if constexpr (schMode == SQUARE_SUM_KEY_COMPACT_AR_DIRECT) {
        KERNEL_TASK_TYPE(SQUARE_SUM_KEY_COMPACT_AR_DIRECT, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        nce_squaresum::SquareSumComputeEngine<DTYPE_INPUT> op;
        op.Initialize(input, output, nullptr, &tilingData, &pipe);
        op.ExecuteCompactDirect();
        return;
    }
#endif
    if constexpr (schMode == SQUARE_SUM_KEY_NORMAL) {
        KERNEL_TASK_TYPE(SQUARE_SUM_KEY_NORMAL, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        nce_squaresum::SquareSumComputeEngine<DTYPE_INPUT> op;
        op.Initialize(input, output, nullptr, &tilingData, &pipe);
        op.Execute();
    }
}
