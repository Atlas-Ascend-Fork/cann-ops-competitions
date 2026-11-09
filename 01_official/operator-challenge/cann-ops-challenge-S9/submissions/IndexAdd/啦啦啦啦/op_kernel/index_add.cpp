#include "kernel_operator.h"

namespace {
using namespace AscendC;

constexpr uint32_t kDataBlockBytes = 32U;
constexpr uint32_t kVectorRepeatBytes = 256U;
constexpr uint32_t kTransposeTileElements = 16U;
constexpr uint32_t kFloatTransposeRows = 8U;
constexpr uint32_t kGatherPipelineDepth = 6U;
constexpr uint32_t kBfloatPipelineDepth = 4U;
constexpr uint32_t kBfloatRawBufferCount = 2U;
constexpr uint32_t kAtomicRawBufferCount = 2U;
constexpr uint32_t kBatchedGatherBufferCount = 1U;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

template <typename T>
class KernelIndexAdd {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                GM_ADDR workspace, const IndexAddTilingData& tiling)
    {
        outerSize_ = tiling.outerSize;
        innerSize_ = tiling.innerSize;
        axisSize_ = tiling.axisSize;
        indexSize_ = tiling.indexSize;
        tilesPerOuter_ = tiling.tilesPerOuter;
        tileElements_ = tiling.tileElements;
        sourceChunk_ = tiling.sourceChunk;
        outerBatch_ = tiling.outerBatch;
        batchAxisChunk_ = tiling.batchAxisChunk;
        sourceTranspose_ = tiling.sourceTranspose;
        sourceBufferCount_ = tiling.sourceBufferCount;
        atomicAdd_ = tiling.atomicAdd;
        selfGm_.SetGlobalBuffer((__gm__ T*)self);
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index);
        sourceGm_.SetGlobalBuffer((__gm__ T*)source);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
        if constexpr (IsSameType<T, int32_t>::value) {
            if (atomicAdd_ == 3U) {
                workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
            }
        }

        const uint32_t rawAlignElements = kDataBlockBytes / sizeof(T);
        const uint32_t alignedTileElements = AlignUp(tileElements_, rawAlignElements);
        const uint32_t indexBytes = AlignUp(indexSize_ * sizeof(int32_t), kDataBlockBytes);
        pipe_.InitBuffer(indexBuf_, indexBytes);

        if constexpr (IsSameType<T, int32_t>::value) {
            if (atomicAdd_ == 4U) {
                const uint32_t alignedInner = AlignUp(innerSize_, rawAlignElements);
                const uint32_t vectorRepeatElements =
                    kVectorRepeatBytes / sizeof(int32_t);
                const uint32_t gatherRepeatCount =
                    (innerSize_ + rawAlignElements / 2U +
                     vectorRepeatElements - 1U) /
                    vectorRepeatElements;
                const uint32_t gatherMaskElements =
                    gatherRepeatCount * rawAlignElements;
                pipe_.InitBuffer(outputBuf_,
                                 batchAxisChunk_ * alignedInner * sizeof(T));
                pipe_.InitBuffer(sourceRawBuf_,
                                 ((sourceChunk_ + sourceChunk_ / 2U) * innerSize_ +
                                  kBatchedGatherBufferCount * alignedInner +
                                  gatherMaskElements) * sizeof(T));
                return;
            }
            if (atomicAdd_ == 1U) {
                const uint32_t alignedInner = AlignUp(innerSize_, rawAlignElements);
                const uint32_t atomicRowElements =
                    (innerSize_ % rawAlignElements) * 2U == rawAlignElements
                    ? alignedInner + rawAlignElements / 2U + alignedInner / 2U
                    : alignedInner;
                pipe_.InitBuffer(sourceRawBuf_,
                                 kAtomicRawBufferCount * sourceChunk_ *
                                     atomicRowElements * sizeof(T));
                return;
            }
            if (atomicAdd_ == 2U) {
                const uint32_t alignedInner = AlignUp(innerSize_, rawAlignElements);
                const uint32_t pairRecordElements =
                    2U * innerSize_ + rawAlignElements;
                const uint32_t vectorRepeatElements =
                    kVectorRepeatBytes / sizeof(int32_t);
                const uint32_t gatherRepeatCount =
                    (innerSize_ + rawAlignElements / 2U +
                     vectorRepeatElements - 1U) /
                    vectorRepeatElements;
                const uint32_t gatherMaskElements =
                    gatherRepeatCount * rawAlignElements;
                pipe_.InitBuffer(outputBuf_,
                                 axisSize_ * alignedInner * sizeof(T));
                pipe_.InitBuffer(sourceRawBuf_,
                                 (kAtomicRawBufferCount * pairRecordElements +
                                  alignedInner + gatherMaskElements) * sizeof(T));
                return;
            }
            if (atomicAdd_ == 3U) {
                const uint32_t alignedInner = AlignUp(innerSize_, rawAlignElements);
                const uint32_t alignedSourceRow =
                    alignedInner + rawAlignElements / 2U + alignedInner / 2U;
                // Full-axis private partial plus four rows in the verified
                // pair/reread layout used by the direct int32 path.
                pipe_.InitBuffer(outputBuf_,
                                 axisSize_ * alignedInner * sizeof(T));
                pipe_.InitBuffer(sourceRawBuf_,
                                 sourceChunk_ * alignedSourceRow * sizeof(T));
                return;
            }
        }

        if (outerBatch_ > 1U) {
            const uint32_t sourceRowElements = AlignUp(sourceChunk_, rawAlignElements);
            const uint32_t axisRowElements = AlignUp(batchAxisChunk_, rawAlignElements);
            const uint32_t batchVectorElements = sourceTranspose_ != 0U
                ? AlignUp(outerBatch_, kTransposeTileElements)
                : AlignUp(outerBatch_, rawAlignElements);
            const uint32_t stagingRowElements =
                axisRowElements > sourceBufferCount_ * sourceRowElements
                ? axisRowElements : sourceBufferCount_ * sourceRowElements;
            const uint32_t outputRows = sourceTranspose_ != 0U &&
                    IsSameType<T, float>::value
                ? AlignUp(axisSize_, kTransposeTileElements) : axisSize_;
            pipe_.InitBuffer(outputBuf_, outputRows * batchVectorElements * sizeof(T));
            pipe_.InitBuffer(sourceRawBuf_, batchVectorElements * stagingRowElements * sizeof(T));
            if (sourceTranspose_ != 0U) {
                pipe_.InitBuffer(batchTransposedBuf_,
                                 sourceChunk_ * batchVectorElements * sizeof(T));
                if constexpr (IsSameType<T, float>::value) {
                    if (sourceBufferCount_ == 2U) {
                        const uint32_t transposeGroups =
                            batchVectorElements / kTransposeTileElements;
                        pipe_.InitBuffer(batchTransposeDstConfigBuf_,
                                         transposeGroups * 4U * sizeof(uint64_t));
                        pipe_.InitBuffer(batchTransposeSrcConfigBuf_,
                                         sourceBufferCount_ * transposeGroups * 4U *
                                             sizeof(uint64_t));
                    }
                }
            }
            const uint32_t gatherValueBuffers = sourceTranspose_ != 0U ||
                IsSameType<T, float>::value ? 1U : kGatherPipelineDepth;
            pipe_.InitBuffer(sourcePromoteBuf_,
                             gatherValueBuffers * batchVectorElements * sizeof(T));
            pipe_.InitBuffer(batchOutputOffsetBuf_,
                             AlignUp(outerBatch_ * sizeof(uint32_t), kDataBlockBytes));
            pipe_.InitBuffer(batchSourceOffsetBuf_,
                             AlignUp(outerBatch_ * sizeof(uint32_t), kDataBlockBytes));
            const uint32_t axisOffsetEntries = sourceTranspose_ != 0U &&
                axisSize_ > kTransposeTileElements ? kTransposeTileElements : axisSize_;
            pipe_.InitBuffer(batchAxisOffsetBuf_,
                             AlignUp(axisOffsetEntries * sizeof(uint32_t), kDataBlockBytes));
            return;
        }

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            pipe_.InitBuffer(outputBuf_, axisSize_ * alignedTileElements * sizeof(T));
            pipe_.InitBuffer(sourceRawBuf_,
                             kBfloatRawBufferCount * sourceChunk_ *
                                 alignedTileElements * sizeof(T));
            pipe_.InitBuffer(sourcePromoteBuf_,
                             sourceChunk_ * alignedTileElements * sizeof(float));
            pipe_.InitBuffer(bfloatOutputPromoteBuf_,
                             kBfloatPipelineDepth * alignedTileElements * sizeof(float));
        } else {
            pipe_.InitBuffer(outputBuf_, axisSize_ * alignedTileElements * sizeof(T));
            pipe_.InitBuffer(sourceRawBuf_, sourceChunk_ * alignedTileElements * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        DataCopyExtParams indexCopy = {
            1U,
            static_cast<uint32_t>(indexSize_ * sizeof(int32_t)),
            0U,
            0U,
            0U,
        };
        DataCopyPadExtParams<int32_t> indexPad = {false, 0U, 0U, 0};
        DataCopyPad(indexLocal, indexGm_, indexCopy, indexPad);
        if constexpr (IsSameType<T, int32_t>::value) {
            if (atomicAdd_ == 1U) {
                // ProcessAtomicInt32 reaches an all-pipeline, cross-core
                // barrier before its first scalar index read. Avoid an
                // otherwise redundant MTE2-to-scalar event here.
                ProcessAtomicInt32(indexLocal);
                return;
            }
        }
        event_t indexReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(indexReady);
        WaitFlag<HardEvent::MTE2_S>(indexReady);
        if constexpr (IsSameType<T, int32_t>::value) {
            if (atomicAdd_ == 3U) {
                ProcessWorkspaceReduceInt32(indexLocal);
                return;
            }
            if (atomicAdd_ == 4U) {
                ProcessBatchedContiguousReduceInt32(indexLocal);
                return;
            }
            if (atomicAdd_ == 2U) {
                ProcessContiguousReduceInt32(indexLocal);
                return;
            }
        }
        if constexpr (IsSameType<T, float>::value) {
            if (outerBatch_ > 1U && sourceTranspose_ != 0U) {
                const uint32_t batchVectorElements =
                    AlignUp(outerBatch_, kTransposeTileElements);
                WaitMte2Vector();
                Muls(indexLocal, indexLocal,
                     static_cast<int32_t>(batchVectorElements * sizeof(float)),
                     static_cast<int32_t>(indexSize_));
                WaitVectorScalar();
                if (sourceBufferCount_ == 2U &&
                    (indexSize_ % kFloatTransposeRows) == 0U) {
                    PreparePipelinedFloatTransposeConfigs(batchVectorElements);
                    WaitScalarVector();
                }
            }
        } else {
            if (outerBatch_ > 1U) {
                const uint32_t batchVectorElements = AlignUp(
                    outerBatch_, kDataBlockBytes / static_cast<uint32_t>(sizeof(T)));
                WaitMte2Vector();
                Muls(indexLocal, indexLocal, static_cast<int32_t>(batchVectorElements),
                     static_cast<int32_t>(indexSize_));
                WaitVectorScalar();
            }
        }
        const uint64_t totalTasks = outerBatch_ > 1U
            ? (outerSize_ + outerBatch_ - 1U) / outerBatch_
            : outerSize_ * tilesPerOuter_;
        bool outputCopyPending = false;
        for (uint64_t task = GetBlockIdx(); task < totalTasks; task += GetBlockNum()) {
            if (outputCopyPending) {
                event_t outputReleased =
                    static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(outputReleased);
                WaitFlag<HardEvent::MTE3_MTE2>(outputReleased);
            }
            if (outerBatch_ > 1U) {
                const uint64_t outerStart = task * outerBatch_;
                const uint32_t batchLength = static_cast<uint32_t>(
                    outerSize_ - outerStart > outerBatch_ ? outerBatch_ : outerSize_ - outerStart);
                if constexpr (IsSameType<T, float>::value) {
                    ProcessFloatOuterBatchTask(indexLocal, outerStart, batchLength);
                } else {
                    ProcessOuterBatchTask(indexLocal, outerStart, batchLength);
                }
                outputCopyPending = true;
                continue;
            }
            const uint64_t outer = task / tilesPerOuter_;
            const uint32_t tile = static_cast<uint32_t>(task % tilesPerOuter_);
            const uint64_t tileStart = static_cast<uint64_t>(tile) * tileElements_;
            if (tileStart >= innerSize_) {
                continue;
            }
            const uint32_t tileLength = static_cast<uint32_t>(
                innerSize_ - tileStart > tileElements_ ? tileElements_ : innerSize_ - tileStart);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ProcessBfloat16Task(indexLocal, outer, tileStart, tileLength);
            } else {
                ProcessNativeTask(indexLocal, outer, tileStart, tileLength);
            }
            outputCopyPending = true;
        }
    }

private:
    __aicore__ inline void CopyAtomicInt32Chunk(
        const LocalTensor<int32_t>& staging, uint32_t slotOffset,
        uint64_t firstRow, uint32_t rows, uint32_t alignedInner) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t tailElements = innerSize_ % alignElements;
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};
            if (tailElements * 2U != alignElements) {
                DataCopyExtParams sourceCopy = {
                    static_cast<uint16_t>(rows),
                    static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                    0U,
                    0U,
                    0U,
                };
                DataCopyPad(staging[slotOffset], sourceGm_[firstRow * innerSize_],
                            sourceCopy, noPad);
                return;
            }

            if ((firstRow & 1U) == 0U && rows >= 4U && (rows & 1U) == 0U) {
                const uint32_t rowBytes = innerSize_ * sizeof(int32_t);
                DataCopyExtParams pairCopy = {
                    static_cast<uint16_t>(rows / 2U), 2U * rowBytes,
                    0U, 1U, 0U,
                };
                DataCopyPad(staging[slotOffset],
                            sourceGm_[firstRow * innerSize_], pairCopy, noPad);
                const uint32_t pairRecordElements =
                    2U * innerSize_ + alignElements;
                const uint32_t oddBase =
                    slotOffset + (rows / 2U) * pairRecordElements;
                // The half-block pair allocation leaves one data block per
                // pair after its odd reread. Use that capacity to phase
                // adjacent reread records without increasing UB usage.
                DataCopyExtParams oddCopy = {
                    static_cast<uint16_t>(rows / 2U), rowBytes,
                    rowBytes, 1U, 0U,
                };
                DataCopyPad(staging[oddBase],
                            sourceGm_[(firstRow + 1U) * innerSize_],
                            oddCopy, noPad);
                return;
            }

            const bool firstIsAligned = (firstRow & 1U) == 0U;
            const uint32_t alignedCount = firstIsAligned ? (rows + 1U) / 2U : rows / 2U;
            const uint32_t shiftedCount = rows - alignedCount;
            const uint64_t alignedFirstRow = firstRow + (firstIsAligned ? 0U : 1U);
            const uint64_t shiftedFirstRow = firstRow + (firstIsAligned ? 1U : 0U);
            const uint32_t bodyElements = innerSize_ - tailElements;
            const uint32_t bodyBytes = bodyElements * sizeof(int32_t);
            const uint32_t tailBytes = tailElements * sizeof(int32_t);
            const uint32_t pairBytes = 2U * innerSize_ * sizeof(int32_t);
            const uint32_t shiftedBodyBase = slotOffset + alignedCount * alignedInner;
            const uint32_t shiftedTailBase =
                shiftedBodyBase + shiftedCount * bodyElements;

            if (alignedCount != 0U) {
                DataCopyExtParams bodyCopy = {
                    static_cast<uint16_t>(alignedCount), bodyBytes,
                    pairBytes - bodyBytes, 1U, 0U,
                };
                DataCopyPad(staging[slotOffset],
                            sourceGm_[alignedFirstRow * innerSize_], bodyCopy, noPad);
                DataCopyExtParams tailCopy = {
                    static_cast<uint16_t>(alignedCount), tailBytes,
                    pairBytes - tailBytes, bodyBytes / kDataBlockBytes, 0U,
                };
                DataCopyPad(staging[slotOffset + bodyElements],
                            sourceGm_[alignedFirstRow * innerSize_ + bodyElements],
                            tailCopy, noPad);
            }
            if (shiftedCount != 0U) {
                DataCopyExtParams bodyCopy = {
                    static_cast<uint16_t>(shiftedCount), bodyBytes,
                    pairBytes - bodyBytes, 0U, 0U,
                };
                DataCopyPad(staging[shiftedBodyBase],
                            sourceGm_[shiftedFirstRow * innerSize_ + tailElements],
                            bodyCopy, noPad);
                DataCopyExtParams tailCopy = {
                    static_cast<uint16_t>(shiftedCount), tailBytes,
                    pairBytes - tailBytes, 0U, 0U,
                };
                DataCopyPad(staging[shiftedTailBase],
                            sourceGm_[shiftedFirstRow * innerSize_], tailCopy, noPad);
            }
        }
    }

    __aicore__ inline void WriteAtomicInt32Chunk(
        const LocalTensor<int32_t>& indexLocal, const LocalTensor<int32_t>& staging,
        uint32_t slotOffset, uint64_t firstRow, uint32_t rows,
        uint32_t alignedInner) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t tailElements = innerSize_ % alignElements;
            if (tailElements * 2U != alignElements) {
                DataCopyExtParams outputCopy = {
                    1U,
                    static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                    0U,
                    0U,
                    0U,
                };
                for (uint32_t row = 0U; row < rows; ++row) {
                    const uint64_t sourceRow = firstRow + row;
                    const uint64_t outer = sourceRow / indexSize_;
                    const uint32_t sourceAxis = static_cast<uint32_t>(sourceRow % indexSize_);
                    const uint32_t destinationAxis =
                        static_cast<uint32_t>(indexLocal(sourceAxis));
                    const uint64_t destination =
                        (outer * axisSize_ + destinationAxis) * innerSize_;
                    DataCopyPad(outputGm_[destination],
                                staging[slotOffset + row * alignedInner], outputCopy);
                }
                return;
            }

            if ((firstRow & 1U) == 0U && rows >= 4U && (rows & 1U) == 0U) {
                const uint32_t pairRecordElements =
                    2U * innerSize_ + alignElements;
                const uint32_t oddBase =
                    slotOffset + (rows / 2U) * pairRecordElements;
                const uint32_t oddRecordElements = alignedInner + alignElements;
                DataCopyExtParams outputCopy = {
                    1U, static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                    0U, 0U, 0U,
                };

                const uint64_t outputOuterStride =
                    static_cast<uint64_t>(axisSize_) * innerSize_;
                const uint64_t firstEvenOuter = firstRow / indexSize_;
                uint32_t evenSourceAxis = static_cast<uint32_t>(
                    firstRow - firstEvenOuter * indexSize_);
                uint64_t evenOutputBase = firstEvenOuter * outputOuterStride;
                for (uint32_t row = 0U; row < rows; row += 2U) {
                    const uint32_t pairOffset =
                        slotOffset + (row / 2U) * pairRecordElements;
                    const uint32_t destinationAxis =
                        static_cast<uint32_t>(indexLocal(evenSourceAxis));
                    const uint64_t destination =
                        evenOutputBase +
                        static_cast<uint64_t>(destinationAxis) * innerSize_;
                    DataCopyPad(outputGm_[destination],
                                staging[pairOffset], outputCopy);
                    evenSourceAxis += 2U;
                    while (evenSourceAxis >= indexSize_) {
                        evenSourceAxis -= indexSize_;
                        evenOutputBase += outputOuterStride;
                    }
                }

                const uint64_t firstOddRow = firstRow + 1U;
                const uint64_t firstOddOuter = firstOddRow / indexSize_;
                uint32_t oddSourceAxis = static_cast<uint32_t>(
                    firstOddRow - firstOddOuter * indexSize_);
                uint64_t oddOutputBase = firstOddOuter * outputOuterStride;
                for (uint32_t row = 1U; row < rows; row += 2U) {
                    const uint32_t destinationAxis =
                        static_cast<uint32_t>(indexLocal(oddSourceAxis));
                    const uint64_t destination =
                        oddOutputBase +
                        static_cast<uint64_t>(destinationAxis) * innerSize_;
                    DataCopyPad(outputGm_[destination],
                                staging[oddBase + (row / 2U) * oddRecordElements],
                                outputCopy);
                    oddSourceAxis += 2U;
                    while (oddSourceAxis >= indexSize_) {
                        oddSourceAxis -= indexSize_;
                        oddOutputBase += outputOuterStride;
                    }
                }
                return;
            }

            const bool firstIsAligned = (firstRow & 1U) == 0U;
            const uint32_t alignedCount = firstIsAligned ? (rows + 1U) / 2U : rows / 2U;
            const uint32_t shiftedCount = rows - alignedCount;
            const uint32_t bodyElements = innerSize_ - tailElements;
            const uint32_t shiftedBodyBase = slotOffset + alignedCount * alignedInner;
            const uint32_t shiftedTailBase =
                shiftedBodyBase + shiftedCount * bodyElements;
            DataCopyExtParams outputCopy = {
                1U, static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                0U, 0U, 0U,
            };
            DataCopyExtParams bodyCopy = {
                1U, static_cast<uint32_t>(bodyElements * sizeof(int32_t)),
                0U, 0U, 0U,
            };
            DataCopyExtParams tailCopy = {
                1U, static_cast<uint32_t>(tailElements * sizeof(int32_t)),
                0U, 0U, 0U,
            };

            const uint64_t outputOuterStride =
                static_cast<uint64_t>(axisSize_) * innerSize_;
            const uint64_t firstOuter = firstRow / indexSize_;
            const uint32_t firstSourceAxis = static_cast<uint32_t>(
                firstRow - firstOuter * indexSize_);
            uint32_t alignedSourceAxis = firstSourceAxis +
                (firstIsAligned ? 0U : 1U);
            uint64_t alignedOutputBase = firstOuter * outputOuterStride;
            if (alignedSourceAxis >= indexSize_) {
                alignedSourceAxis -= indexSize_;
                alignedOutputBase += outputOuterStride;
            }
            for (uint32_t row = 0U; row < alignedCount; ++row) {
                const uint32_t destinationAxis =
                    static_cast<uint32_t>(indexLocal(alignedSourceAxis));
                const uint64_t destination =
                    alignedOutputBase + static_cast<uint64_t>(destinationAxis) * innerSize_;
                DataCopyPad(outputGm_[destination],
                            staging[slotOffset + row * alignedInner], outputCopy);
                alignedSourceAxis += 2U;
                if (alignedSourceAxis >= indexSize_) {
                    alignedSourceAxis -= indexSize_;
                    alignedOutputBase += outputOuterStride;
                }
            }
            uint32_t shiftedSourceAxis = firstSourceAxis +
                (firstIsAligned ? 1U : 0U);
            uint64_t shiftedOutputBase = firstOuter * outputOuterStride;
            if (shiftedSourceAxis >= indexSize_) {
                shiftedSourceAxis -= indexSize_;
                shiftedOutputBase += outputOuterStride;
            }
            for (uint32_t row = 0U; row < shiftedCount; ++row) {
                const uint32_t destinationAxis =
                    static_cast<uint32_t>(indexLocal(shiftedSourceAxis));
                const uint64_t destination =
                    shiftedOutputBase + static_cast<uint64_t>(destinationAxis) * innerSize_;
                DataCopyPad(outputGm_[destination + tailElements],
                            staging[shiftedBodyBase + row * bodyElements], bodyCopy);
                DataCopyPad(outputGm_[destination],
                            staging[shiftedTailBase + row * alignElements], tailCopy);
                shiftedSourceAxis += 2U;
                if (shiftedSourceAxis >= indexSize_) {
                    shiftedSourceAxis -= indexSize_;
                    shiftedOutputBase += outputOuterStride;
                }
            }
        }
    }

    __aicore__ inline void AddReduceInt32PairChunk(
        const LocalTensor<int32_t>& indexLocal,
        const LocalTensor<int32_t>& staging,
        const LocalTensor<int32_t>& partial,
        uint32_t firstSourceAxis, uint32_t rows, uint32_t alignedInner) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t pairRecordElements = 2U * innerSize_ + alignElements;
            const uint32_t oddBase = (rows / 2U) * pairRecordElements;
            const uint32_t oddRecordElements = alignedInner + alignElements;
            for (uint32_t row = 0U; row < rows; row += 2U) {
                const uint32_t evenDestination = static_cast<uint32_t>(
                    indexLocal(firstSourceAxis + row));
                const uint32_t oddDestination = static_cast<uint32_t>(
                    indexLocal(firstSourceAxis + row + 1U));
                Add(partial[evenDestination * alignedInner],
                    partial[evenDestination * alignedInner],
                    staging[(row / 2U) * pairRecordElements],
                    static_cast<int32_t>(innerSize_));
                Add(partial[oddDestination * alignedInner],
                    partial[oddDestination * alignedInner],
                    staging[oddBase + (row / 2U) * oddRecordElements],
                    static_cast<int32_t>(innerSize_));
            }
        }
    }

    __aicore__ inline void CopyAlignedInt32Pair(
        const LocalTensor<int32_t>& staging, uint32_t slotOffset,
        uint64_t firstRow) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            DataCopyExtParams pairCopy = {
                1U,
                static_cast<uint32_t>(2U * innerSize_ * sizeof(int32_t)),
                0U,
                0U,
                0U,
            };
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};
            DataCopyPad(staging[slotOffset], sourceGm_[firstRow * innerSize_],
                        pairCopy, noPad);
        }
    }

    __aicore__ inline void PrepareInt32GatherMask(
        const LocalTensor<uint32_t>& gatherMask,
        uint32_t gatherRepeatCount, uint32_t gatherMaskElements,
        uint32_t vectorRepeatElements, uint32_t tailElements) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            Duplicate(gatherMask, UINT32_MAX,
                      static_cast<int32_t>(gatherMaskElements));
            WaitVectorScalar();

            gatherMask.SetValue(0U, UINT32_MAX << tailElements);
            const uint32_t lastValid =
                (innerSize_ + tailElements) % vectorRepeatElements;
            if (lastValid != 0U) {
                const uint32_t lastBase =
                    (gatherRepeatCount - 1U) *
                    (kDataBlockBytes / sizeof(uint32_t));
                if (lastValid <= 32U) {
                    const uint32_t lowMask = lastValid == 32U
                        ? UINT32_MAX : (1U << lastValid) - 1U;
                    gatherMask.SetValue(lastBase, lowMask);
                    gatherMask.SetValue(lastBase + 1U, 0U);
                } else {
                    const uint32_t highBits = lastValid - 32U;
                    const uint32_t highMask = highBits == 32U
                        ? UINT32_MAX : (1U << highBits) - 1U;
                    gatherMask.SetValue(lastBase + 1U, highMask);
                }
            }
            event_t maskReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(maskReady);
            WaitFlag<HardEvent::S_V>(maskReady);
        }
    }

    __aicore__ inline void AddReduceInt32GatherPair(
        const LocalTensor<int32_t>& indexLocal,
        const LocalTensor<int32_t>& staging,
        const LocalTensor<int32_t>& partial,
        const LocalTensor<int32_t>& gathered,
        const LocalTensor<uint32_t>& gatherMask,
        uint32_t slotOffset, uint32_t firstSourceAxis,
        uint32_t alignedInner, uint32_t gatherRepeatCount,
        uint32_t vectorRepeatElements, uint32_t tailElements) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            const uint32_t evenDestination = static_cast<uint32_t>(
                indexLocal(firstSourceAxis));
            const uint32_t oddDestination = static_cast<uint32_t>(
                indexLocal(firstSourceAxis + 1U));
            Add(partial[evenDestination * alignedInner],
                partial[evenDestination * alignedInner],
                staging[slotOffset], static_cast<int32_t>(innerSize_));

            GatherMaskParams gatherParams(
                DEFAULT_BLK_STRIDE,
                static_cast<uint16_t>(gatherRepeatCount),
                DEFAULT_REPEAT_STRIDE,
                DEFAULT_BLK_STRIDE);
            uint64_t gatheredCount = 0U;
            GatherMask<int32_t, uint32_t>(
                gathered, staging[slotOffset + innerSize_ - tailElements],
                gatherMask, true, vectorRepeatElements,
                gatherParams, gatheredCount);
            PipeBarrier<PIPE_V>();
            Add(partial[oddDestination * alignedInner],
                partial[oddDestination * alignedInner], gathered,
                static_cast<int32_t>(innerSize_));
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessAtomicInt32(const LocalTensor<int32_t>& indexLocal)
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t alignedInner = AlignUp(innerSize_, alignElements);
            const uint32_t atomicRowElements =
                (innerSize_ % alignElements) * 2U == alignElements
                ? alignedInner + alignElements / 2U + alignedInner / 2U
                : alignedInner;
            LocalTensor<int32_t> staging = sourceRawBuf_.Get<int32_t>();

            const uint64_t totalSourceRows = outerSize_ * indexSize_;
            const uint64_t rowsPerCore =
                (totalSourceRows + GetBlockNum() - 1U) / GetBlockNum();
            const uint64_t firstRow = GetBlockIdx() * rowsPerCore;
            uint64_t finalRow = firstRow + rowsPerCore;
            if (finalRow > totalSourceRows) {
                finalRow = totalSourceRows;
            }
            uint64_t chunkStart = firstRow;
            uint32_t rows = static_cast<uint32_t>(
                finalRow - chunkStart > sourceChunk_ ? sourceChunk_ : finalRow - chunkStart);
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};
            bool sourcePrefetched = false;

            const uint64_t totalOutputElements = outerSize_ * axisSize_ * innerSize_;
            const uint64_t totalOutputBlocks =
                (totalOutputElements + alignElements - 1U) / alignElements;
            const uint64_t blocksPerCore =
                (totalOutputBlocks + GetBlockNum() - 1U) / GetBlockNum();
            const uint64_t firstBlock = GetBlockIdx() * blocksPerCore;
            if (firstBlock < totalOutputBlocks) {
                uint64_t blockCount = totalOutputBlocks - firstBlock;
                if (blockCount > blocksPerCore) {
                    blockCount = blocksPerCore;
                }
                const uint64_t firstElement = firstBlock * alignElements;
                uint64_t elementCount = blockCount * alignElements;
                if (elementCount > totalOutputElements - firstElement) {
                    elementCount = totalOutputElements - firstElement;
                }
                DataCopyExtParams initialCopy = {
                    1U,
                    static_cast<uint32_t>(elementCount * sizeof(int32_t)),
                    0U,
                    0U,
                    0U,
                };
                DataCopyPad(staging, selfGm_[firstElement], initialCopy, noPad);
                event_t selfReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(selfReady);
                WaitFlag<HardEvent::MTE2_MTE3>(selfReady);
                const uint32_t slotElements = sourceChunk_ * atomicRowElements;
                if (elementCount <= slotElements && chunkStart < finalRow) {
                    CopyAtomicInt32Chunk(
                        staging, slotElements, chunkStart, rows, alignedInner);
                    sourcePrefetched = true;
                }
                DataCopyPad(outputGm_[firstElement], staging, initialCopy);
            }

            SyncAll<true>();

            SetAtomicAdd<int32_t>();
            uint32_t slot = sourcePrefetched ? 1U : 0U;
            uint32_t chunksIssued = 0U;
            if (chunkStart < finalRow && !sourcePrefetched) {
                CopyAtomicInt32Chunk(staging, 0U, chunkStart, rows, alignedInner);
            }
            while (chunkStart < finalRow) {
                event_t sourceReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(sourceReady);
                WaitFlag<HardEvent::MTE2_MTE3>(sourceReady);

                const uint64_t nextChunkStart = chunkStart + rows;
                const uint32_t nextSlot = slot ^ 1U;
                uint32_t nextRows = 0U;
                if (nextChunkStart < finalRow) {
                    nextRows = static_cast<uint32_t>(
                        finalRow - nextChunkStart > sourceChunk_
                            ? sourceChunk_ : finalRow - nextChunkStart);
                    if (chunksIssued >= kAtomicRawBufferCount - 1U) {
                        event_t outputReleased = static_cast<event_t>(
                            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                        SetFlag<HardEvent::MTE3_MTE2>(outputReleased);
                        WaitFlag<HardEvent::MTE3_MTE2>(outputReleased);
                    }
                    CopyAtomicInt32Chunk(
                        staging, nextSlot * sourceChunk_ * atomicRowElements,
                        nextChunkStart, nextRows, alignedInner);
                }

                WriteAtomicInt32Chunk(
                    indexLocal, staging, slot * sourceChunk_ * atomicRowElements,
                    chunkStart, rows, alignedInner);
                ++chunksIssued;
                chunkStart = nextChunkStart;
                rows = nextRows;
                slot = nextSlot;
            }
            SetAtomicNone();
        }
    }

    __aicore__ inline void ProcessWorkspaceReduceInt32(
        const LocalTensor<int32_t>& indexLocal)
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            constexpr uint32_t workspaceSlotsPerCore = 2U;
            const uint32_t alignedInner = AlignUp(innerSize_, alignElements);
            const uint32_t partialElements = axisSize_ * alignedInner;
            const uint32_t alignedSourceRow =
                alignedInner + alignElements / 2U + alignedInner / 2U;
            const uint32_t stagingElements = sourceChunk_ * alignedSourceRow;
            const uint32_t workspaceRowsPerLoad = stagingElements / alignedInner;
            LocalTensor<int32_t> partial = outputBuf_.Get<int32_t>();
            LocalTensor<int32_t> staging = sourceRawBuf_.Get<int32_t>();
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};

            const uint64_t totalSourceRows = outerSize_ * indexSize_;
            const uint64_t unalignedRowsPerCore =
                (totalSourceRows + GetBlockNum() - 1U) / GetBlockNum();
            const uint64_t rowsPerCore =
                (unalignedRowsPerCore + sourceChunk_ - 1U) / sourceChunk_ * sourceChunk_;
            const uint64_t firstRow = GetBlockIdx() * rowsPerCore;
            uint64_t finalRow = firstRow + rowsPerCore;
            if (finalRow > totalSourceRows) {
                finalRow = totalSourceRows;
            }

            bool workspaceWritePending = false;
            uint32_t workspaceSlot = 0U;
            uint64_t segmentStart = firstRow;
            while (segmentStart < finalRow) {
                const uint64_t outer = segmentStart / indexSize_;
                uint64_t segmentFinal = (outer + 1U) * indexSize_;
                if (segmentFinal > finalRow) {
                    segmentFinal = finalRow;
                }
                if (workspaceWritePending) {
                    WaitMte3Vector();
                }
                Duplicate(partial, static_cast<int32_t>(0),
                          static_cast<int32_t>(partialElements));

                bool sourceInUse = false;
                for (uint64_t chunkStart = segmentStart; chunkStart < segmentFinal;
                     chunkStart += sourceChunk_) {
                    if (sourceInUse) {
                        WaitVectorMte2();
                    }
                    const uint32_t rows = static_cast<uint32_t>(
                        segmentFinal - chunkStart > sourceChunk_
                            ? sourceChunk_ : segmentFinal - chunkStart);
                    CopyAtomicInt32Chunk(
                        staging, 0U, chunkStart, rows, alignedInner);
                    WaitMte2Vector();
                    AddReduceInt32PairChunk(
                        indexLocal, staging, partial,
                        static_cast<uint32_t>(chunkStart - outer * indexSize_),
                        rows, alignedInner);
                    sourceInUse = true;
                }

                // The final two Adds in a segment may still have vector
                // repeats in flight when the scalar loop exits. Fully drain
                // V before MTE3 snapshots the private partial to workspace.
                PipeBarrier<PIPE_V>();
                WaitVectorScalar();
                WaitVectorMte3();
                const uint64_t workspaceOffset =
                    (static_cast<uint64_t>(GetBlockIdx()) * workspaceSlotsPerCore +
                     workspaceSlot) * partialElements;
                DataCopyExtParams workspaceCopy = {
                    1U,
                    static_cast<uint32_t>(partialElements * sizeof(int32_t)),
                    0U,
                    0U,
                    0U,
                };
                DataCopyPad(workspaceGm_[workspaceOffset], partial, workspaceCopy);
                workspaceWritePending = true;
                ++workspaceSlot;
                segmentStart = segmentFinal;
            }

            if (workspaceWritePending) {
                event_t workspaceWriteComplete = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(workspaceWriteComplete);
                WaitFlag<HardEvent::MTE3_S>(workspaceWriteComplete);
            }
            // The workspace is a non-atomic producer/consumer channel across
            // vector cores. Drain MTE3 and clean/invalidate the local data
            // cache before publishing arrival at the inter-core barrier.
            PipeBarrier<PIPE_ALL>();
            SyncAll<true>();
            // Drop any pre-barrier cache state before owner cores issue their
            // workspace MTE2 reads.
            PipeBarrier<PIPE_ALL>();

            const uint64_t outer = GetBlockIdx();
            if (outer >= outerSize_) {
                return;
            }

            Duplicate(partial, static_cast<int32_t>(0),
                      static_cast<int32_t>(partialElements));
            WaitVectorMte2();
            const uint64_t outputBase = outer * axisSize_ * innerSize_;
            DataCopyExtParams partialRowCopy = {
                1U,
                static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                0U,
                0U,
                0U,
            };
            for (uint32_t axis = 0U; axis < axisSize_; ++axis) {
                DataCopyPad(partial[axis * alignedInner],
                            selfGm_[outputBase + axis * innerSize_],
                            partialRowCopy, noPad);
            }
            WaitMte2Vector();

            bool stagingInUse = false;
            for (uint32_t producer = 0U; producer < GetBlockNum(); ++producer) {
                const uint64_t producerFirst =
                    static_cast<uint64_t>(producer) * rowsPerCore;
                if (producerFirst >= totalSourceRows) {
                    break;
                }
                uint64_t producerFinal = producerFirst + rowsPerCore;
                if (producerFinal > totalSourceRows) {
                    producerFinal = totalSourceRows;
                }
                const uint64_t producerFirstOuter = producerFirst / indexSize_;
                const uint64_t producerFinalOuter =
                    (producerFinal - 1U) / indexSize_;
                uint32_t producerSlot = 0U;
                if (outer == producerFirstOuter) {
                    producerSlot = 0U;
                } else if (producerFinalOuter != producerFirstOuter &&
                           outer == producerFinalOuter) {
                    producerSlot = 1U;
                } else {
                    continue;
                }

                const uint64_t workspaceBase =
                    (static_cast<uint64_t>(producer) * workspaceSlotsPerCore +
                     producerSlot) * partialElements;
                for (uint32_t axisStart = 0U; axisStart < axisSize_;
                     axisStart += workspaceRowsPerLoad) {
                    if (stagingInUse) {
                        WaitVectorMte2();
                    }
                    const uint32_t rows = axisSize_ - axisStart > workspaceRowsPerLoad
                        ? workspaceRowsPerLoad : axisSize_ - axisStart;
                    const uint32_t elements = rows * alignedInner;
                    DataCopyExtParams workspaceCopy = {
                        1U,
                        static_cast<uint32_t>(elements * sizeof(int32_t)),
                        0U,
                        0U,
                        0U,
                    };
                    DataCopyPad(staging, workspaceGm_[workspaceBase +
                                                      axisStart * alignedInner],
                                workspaceCopy, noPad);
                    WaitMte2Vector();
                    Add(partial[axisStart * alignedInner],
                        partial[axisStart * alignedInner], staging,
                        static_cast<int32_t>(elements));
                    stagingInUse = true;
                }
            }

            WaitVectorMte3();
            for (uint32_t axis = 0U; axis < axisSize_; ++axis) {
                DataCopyPad(outputGm_[outputBase + axis * innerSize_],
                            partial[axis * alignedInner], partialRowCopy);
            }
            event_t outputDrainComplete = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(outputDrainComplete);
            WaitFlag<HardEvent::MTE3_S>(outputDrainComplete);
        }
    }

    __aicore__ inline void CopyContiguousInt32Group(
        const LocalTensor<int32_t>& staging, uint32_t slotOffset,
        uint64_t firstRow, uint32_t rows) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            DataCopyExtParams groupCopy = {
                1U,
                static_cast<uint32_t>(rows * innerSize_ * sizeof(int32_t)),
                0U,
                0U,
                0U,
            };
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};
            DataCopyPad(staging[slotOffset], sourceGm_[firstRow * innerSize_],
                        groupCopy, noPad);
        }
    }

    __aicore__ inline void ProcessBatchedContiguousReduceInt32(
        const LocalTensor<int32_t>& indexLocal)
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t alignedInner = AlignUp(innerSize_, alignElements);
            const uint32_t tailElements = alignElements / 2U;
            const uint32_t groupSlotsElements =
                (sourceChunk_ + sourceChunk_ / 2U) * innerSize_;
            const uint32_t vectorRepeatElements =
                kVectorRepeatBytes / sizeof(int32_t);
            const uint32_t gatherRepeatCount =
                (innerSize_ + tailElements + vectorRepeatElements - 1U) /
                vectorRepeatElements;
            const uint32_t gatherMaskElements =
                gatherRepeatCount * alignElements;
            const uint32_t partialAxisCount = batchAxisChunk_;

            LocalTensor<int32_t> partial = outputBuf_.Get<int32_t>();
            LocalTensor<int32_t> staging = sourceRawBuf_.Get<int32_t>();
            LocalTensor<int32_t> gathered = staging[groupSlotsElements];
            LocalTensor<uint32_t> gatherMask =
                staging[groupSlotsElements +
                        kBatchedGatherBufferCount * alignedInner]
                    .ReinterpretCast<uint32_t>();
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};

            const uint64_t outer = GetBlockIdx();
            if (outer >= outerSize_) {
                return;
            }
            PrepareInt32GatherMask(
                gatherMask, gatherRepeatCount, gatherMaskElements,
                vectorRepeatElements, tailElements);

            const uint64_t outputBase = outer * axisSize_ * innerSize_;
            DataCopyExtParams rowCopy = {
                1U,
                static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                0U,
                0U,
                0U,
            };
            for (uint32_t axis = 0U; axis < partialAxisCount; ++axis) {
                DataCopyPad(partial[axis * alignedInner],
                            selfGm_[outputBase + axis * innerSize_],
                            rowCopy, noPad);
            }

            // The non-resident destination axes form one contiguous suffix.
            // Initialize that suffix in bounded multi-row chunks before
            // enabling atomic mode.  The same source-group slot is reused only
            // after MTE3 has consumed it.
            for (uint32_t axisStart = partialAxisCount; axisStart < axisSize_;) {
                const uint32_t rows = axisSize_ - axisStart > sourceChunk_
                    ? sourceChunk_ : axisSize_ - axisStart;
                DataCopyExtParams selfCopy = {
                    1U,
                    static_cast<uint32_t>(rows * innerSize_ * sizeof(int32_t)),
                    0U,
                    0U,
                    0U,
                };
                DataCopyPad(staging, selfGm_[outputBase + axisStart * innerSize_],
                            selfCopy, noPad);
                event_t selfReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(selfReady);
                WaitFlag<HardEvent::MTE2_MTE3>(selfReady);
                DataCopyPad(outputGm_[outputBase + axisStart * innerSize_],
                            staging, selfCopy);
                event_t selfReleased = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(selfReleased);
                WaitFlag<HardEvent::MTE3_MTE2>(selfReleased);
                axisStart += rows;
            }
            WaitMte2Vector();

            const uint64_t outerFirstRow = outer * indexSize_;
            event_t groupReadyV[kAtomicRawBufferCount];
            event_t groupReadyMte3[kAtomicRawBufferCount];
            event_t groupReleasedV[kAtomicRawBufferCount];
            event_t groupReleasedMte3[kAtomicRawBufferCount];
            event_t gatheredReadyMte3[kBatchedGatherBufferCount];
            event_t gatheredReleasedV[kBatchedGatherBufferCount];
            bool groupHasUse[kAtomicRawBufferCount] = {false, false};
            bool gatheredMte3Pending[kBatchedGatherBufferCount] = {false};

            for (uint32_t slot = 0U; slot < kAtomicRawBufferCount; ++slot) {
                groupReadyV[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
                groupReadyMte3[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
                groupReleasedV[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
                groupReleasedMte3[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
            }
            for (uint32_t slot = 0U; slot < kBatchedGatherBufferCount; ++slot) {
                gatheredReadyMte3[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
                gatheredReleasedV[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            }

            const uint32_t firstRows = indexSize_ > sourceChunk_
                ? sourceChunk_ : indexSize_;
            CopyContiguousInt32Group(staging, 0U, outerFirstRow, firstRows);
            SetFlag<HardEvent::MTE2_V>(groupReadyV[0]);
            SetFlag<HardEvent::MTE2_MTE3>(groupReadyMte3[0]);
            SetAtomicAdd<int32_t>();

            uint32_t groupSlot = 0U;
            for (uint32_t groupStart = 0U; groupStart < indexSize_;) {
                const uint32_t slotCapacity = groupSlot == 0U
                    ? sourceChunk_ : sourceChunk_ / 2U;
                const uint32_t rows = indexSize_ - groupStart > slotCapacity
                    ? slotCapacity : indexSize_ - groupStart;
                WaitFlag<HardEvent::MTE2_V>(groupReadyV[groupSlot]);
                WaitFlag<HardEvent::MTE2_MTE3>(groupReadyMte3[groupSlot]);

                const uint32_t nextStart = groupStart + rows;
                const uint32_t nextSlot = groupSlot ^ 1U;
                if (nextStart < indexSize_) {
                    if (groupHasUse[nextSlot]) {
                        WaitFlag<HardEvent::V_MTE2>(groupReleasedV[nextSlot]);
                        WaitFlag<HardEvent::MTE3_MTE2>(groupReleasedMte3[nextSlot]);
                    }
                    const uint32_t nextCapacity = nextSlot == 0U
                        ? sourceChunk_ : sourceChunk_ / 2U;
                    const uint32_t nextRows = indexSize_ - nextStart > nextCapacity
                        ? nextCapacity : indexSize_ - nextStart;
                    const uint32_t nextOffset = nextSlot == 0U
                        ? 0U : sourceChunk_ * innerSize_;
                    CopyContiguousInt32Group(
                        staging[nextOffset], 0U,
                        outerFirstRow + nextStart, nextRows);
                    SetFlag<HardEvent::MTE2_V>(groupReadyV[nextSlot]);
                    SetFlag<HardEvent::MTE2_MTE3>(groupReadyMte3[nextSlot]);
                }

                const uint32_t groupOffset = groupSlot == 0U
                    ? 0U : sourceChunk_ * innerSize_;
                for (uint32_t row = 0U; row < rows; row += 2U) {
                    constexpr uint32_t pairSlot = 0U;
                    if (gatheredMte3Pending[pairSlot]) {
                        WaitFlag<HardEvent::MTE3_V>(gatheredReleasedV[pairSlot]);
                        gatheredMte3Pending[pairSlot] = false;
                    }
                    const uint32_t sourceAxis = groupStart + row;
                    const uint32_t evenDestination = static_cast<uint32_t>(
                        indexLocal(sourceAxis));
                    const uint32_t oddDestination = static_cast<uint32_t>(
                        indexLocal(sourceAxis + 1U));
                    const uint32_t pairOffset =
                        groupOffset + row * innerSize_;

                    if (evenDestination < partialAxisCount) {
                        Add(partial[evenDestination * alignedInner],
                            partial[evenDestination * alignedInner],
                            staging[pairOffset],
                            static_cast<int32_t>(innerSize_));
                    } else {
                        DataCopyPad(
                            outputGm_[outputBase +
                                      static_cast<uint64_t>(evenDestination) * innerSize_],
                            staging[pairOffset], rowCopy);
                    }

                    GatherMaskParams gatherParams(
                        DEFAULT_BLK_STRIDE,
                        static_cast<uint16_t>(gatherRepeatCount),
                        DEFAULT_REPEAT_STRIDE,
                        DEFAULT_BLK_STRIDE);
                    uint64_t gatheredCount = 0U;
                    GatherMask<int32_t, uint32_t>(
                        gathered,
                        staging[pairOffset + innerSize_ - tailElements],
                        gatherMask, true, vectorRepeatElements,
                        gatherParams, gatheredCount);
                    PipeBarrier<PIPE_V>();
                    if (oddDestination < partialAxisCount) {
                        Add(partial[oddDestination * alignedInner],
                            partial[oddDestination * alignedInner],
                            gathered,
                            static_cast<int32_t>(innerSize_));
                    } else {
                        SetFlag<HardEvent::V_MTE3>(gatheredReadyMte3[pairSlot]);
                        WaitFlag<HardEvent::V_MTE3>(gatheredReadyMte3[pairSlot]);
                        DataCopyPad(
                            outputGm_[outputBase +
                                      static_cast<uint64_t>(oddDestination) * innerSize_],
                            gathered, rowCopy);
                        SetFlag<HardEvent::MTE3_V>(gatheredReleasedV[pairSlot]);
                        gatheredMte3Pending[pairSlot] = true;
                    }
                    PipeBarrier<PIPE_V>();
                }

                SetFlag<HardEvent::V_MTE2>(groupReleasedV[groupSlot]);
                SetFlag<HardEvent::MTE3_MTE2>(groupReleasedMte3[groupSlot]);
                groupHasUse[groupSlot] = true;
                groupSlot = nextSlot;
                groupStart = nextStart;
            }

            for (uint32_t slot = 0U; slot < kAtomicRawBufferCount; ++slot) {
                if (groupHasUse[slot]) {
                    WaitFlag<HardEvent::V_MTE2>(groupReleasedV[slot]);
                    WaitFlag<HardEvent::MTE3_MTE2>(groupReleasedMte3[slot]);
                }
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(groupReadyV[slot]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(
                    groupReadyMte3[slot]);
                GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(
                    groupReleasedV[slot]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(
                    groupReleasedMte3[slot]);
            }
            for (uint32_t slot = 0U; slot < kBatchedGatherBufferCount; ++slot) {
                if (gatheredMte3Pending[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(gatheredReleasedV[slot]);
                }
                GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(
                    gatheredReadyMte3[slot]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(
                    gatheredReleasedV[slot]);
            }
            SetAtomicNone();

            WaitVectorMte3();
            for (uint32_t axis = 0U; axis < partialAxisCount; ++axis) {
                DataCopyPad(outputGm_[outputBase + axis * innerSize_],
                            partial[axis * alignedInner], rowCopy);
            }
            event_t outputDrainComplete = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(outputDrainComplete);
            WaitFlag<HardEvent::MTE3_S>(outputDrainComplete);
        }
    }

    __aicore__ inline void ProcessContiguousReduceInt32(
        const LocalTensor<int32_t>& indexLocal)
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(int32_t);
            const uint32_t alignedInner = AlignUp(innerSize_, alignElements);
            const uint32_t tailElements = alignElements / 2U;
            const uint32_t pairRecordElements =
                2U * innerSize_ + alignElements;
            const uint32_t pairSlotsElements =
                kAtomicRawBufferCount * pairRecordElements;
            const uint32_t vectorRepeatElements =
                kVectorRepeatBytes / sizeof(int32_t);
            const uint32_t gatherRepeatCount =
                (innerSize_ + tailElements + vectorRepeatElements - 1U) /
                vectorRepeatElements;
            const uint32_t gatherMaskElements =
                gatherRepeatCount * alignElements;
            LocalTensor<int32_t> partial = outputBuf_.Get<int32_t>();
            LocalTensor<int32_t> staging = sourceRawBuf_.Get<int32_t>();
            LocalTensor<int32_t> gathered = staging[pairSlotsElements];
            LocalTensor<uint32_t> gatherMask =
                staging[pairSlotsElements + alignedInner].ReinterpretCast<uint32_t>();
            DataCopyPadExtParams<int32_t> noPad = {false, 0U, 0U, 0};

            const uint64_t outer = GetBlockIdx();
            if (outer >= outerSize_) {
                return;
            }
            PrepareInt32GatherMask(
                gatherMask, gatherRepeatCount, gatherMaskElements,
                vectorRepeatElements, tailElements);
            const uint64_t outputBase = outer * axisSize_ * innerSize_;
            DataCopyExtParams partialRowCopy = {
                1U,
                static_cast<uint32_t>(innerSize_ * sizeof(int32_t)),
                0U,
                0U,
                0U,
            };
            for (uint32_t axis = 0U; axis < axisSize_; ++axis) {
                DataCopyPad(partial[axis * alignedInner],
                            selfGm_[outputBase + axis * innerSize_],
                            partialRowCopy, noPad);
            }
            WaitMte2Vector();

            const uint64_t outerFirstRow = outer * indexSize_;
            const uint64_t outerFinalRow = outerFirstRow + indexSize_;
            event_t pairReady[kAtomicRawBufferCount];
            event_t pairReleased[kAtomicRawBufferCount];
            bool slotHasVectorUse[kAtomicRawBufferCount] = {false, false};

            // AllocEventID reserves the software event-pool bit; FetchEventID
            // only observes a free bit and therefore cannot reserve two
            // independent events when called twice.
            pairReady[0] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            pairReady[1] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            pairReleased[0] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            pairReleased[1] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            CopyAlignedInt32Pair(staging, 0U, outerFirstRow);
            SetFlag<HardEvent::MTE2_V>(pairReady[0]);
            uint32_t slot = 0U;

            for (uint64_t pairStart = outerFirstRow; pairStart < outerFinalRow;
                 pairStart += sourceChunk_) {
                WaitFlag<HardEvent::MTE2_V>(pairReady[slot]);

                const uint64_t nextPairStart = pairStart + sourceChunk_;
                const uint32_t nextSlot = slot ^ 1U;
                if (nextPairStart < outerFinalRow) {
                    if (slotHasVectorUse[nextSlot]) {
                        WaitFlag<HardEvent::V_MTE2>(pairReleased[nextSlot]);
                    }
                    CopyAlignedInt32Pair(
                        staging, nextSlot * pairRecordElements,
                        nextPairStart);
                    SetFlag<HardEvent::MTE2_V>(pairReady[nextSlot]);
                }

                AddReduceInt32GatherPair(
                    indexLocal, staging, partial, gathered, gatherMask,
                    slot * pairRecordElements,
                    static_cast<uint32_t>(pairStart - outerFirstRow),
                    alignedInner, gatherRepeatCount,
                    vectorRepeatElements, tailElements);
                SetFlag<HardEvent::V_MTE2>(pairReleased[slot]);
                slotHasVectorUse[slot] = true;
                slot = nextSlot;
            }

            for (uint32_t drainSlot = 0U;
                 drainSlot < kAtomicRawBufferCount; ++drainSlot) {
                if (slotHasVectorUse[drainSlot]) {
                    WaitFlag<HardEvent::V_MTE2>(pairReleased[drainSlot]);
                }
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(pairReady[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(pairReady[1]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(pairReleased[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(pairReleased[1]);

            WaitVectorMte3();
            for (uint32_t axis = 0U; axis < axisSize_; ++axis) {
                DataCopyPad(outputGm_[outputBase + axis * innerSize_],
                            partial[axis * alignedInner], partialRowCopy);
            }
            event_t outputDrainComplete = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(outputDrainComplete);
            WaitFlag<HardEvent::MTE3_S>(outputDrainComplete);
        }
    }

    __aicore__ inline void AddIndependentFloatVector(uint64_t destinationAddress,
                                                      const LocalTensor<float>& source) const
    {
        if ASCEND_IS_AIV {
            constexpr uint64_t config = 0x0100080808010101UL;
            __builtin_cce_vadd(
                reinterpret_cast<__ubuf__ float*>(destinationAddress),
                reinterpret_cast<__ubuf__ float*>(destinationAddress),
                reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr()), config);
        }
    }

    __aicore__ inline void SetIndependentFloatMask(uint32_t count) const
    {
        if ASCEND_IS_AIV {
            set_mask_norm();
            const uint64_t mask = count == 64U
                ? static_cast<uint64_t>(-1)
                : (1ULL << count) - 1ULL;
            set_vector_mask(0ULL, mask);
        }
    }

    __aicore__ inline void ProcessIndependentFloatGroup(
        __ubuf__ uint64_t* indexPacked, uint32_t packedOffset,
        uint64_t outputAddress, const LocalTensor<float>& transposedLocal,
        uint32_t row, uint32_t batchVectorElements) const
    {
        const uint64_t destination01 = indexPacked[packedOffset];
        const uint64_t destination23 = indexPacked[packedOffset + 1U];
        const uint64_t destination45 = indexPacked[packedOffset + 2U];
        const uint64_t destination67 = indexPacked[packedOffset + 3U];
        const uint32_t destination0 = static_cast<uint32_t>(destination01);
        const uint32_t destination1 = static_cast<uint32_t>(destination01 >> 32U);
        const uint32_t destination2 = static_cast<uint32_t>(destination23);
        const uint32_t destination3 = static_cast<uint32_t>(destination23 >> 32U);
        const uint32_t destination4 = static_cast<uint32_t>(destination45);
        const uint32_t destination5 = static_cast<uint32_t>(destination45 >> 32U);
        const uint32_t destination6 = static_cast<uint32_t>(destination67);
        const uint32_t destination7 = static_cast<uint32_t>(destination67 >> 32U);
        const bool distinctFirst =
            destination1 != destination0 &&
            destination2 != destination0 && destination2 != destination1 &&
            destination3 != destination0 && destination3 != destination1 &&
            destination3 != destination2;
        const bool distinctSecond =
            destination5 != destination4 &&
            destination6 != destination4 && destination6 != destination5 &&
            destination7 != destination4 && destination7 != destination5 &&
            destination7 != destination6;
        PipeBarrier<PIPE_V>();
        if (distinctFirst) {
            AddIndependentFloatVector(
                outputAddress + destination0,
                transposedLocal[row * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination1,
                transposedLocal[(row + 1U) * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination2,
                transposedLocal[(row + 2U) * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination3,
                transposedLocal[(row + 3U) * batchVectorElements]);
        } else {
            AddIndependentFloatVector(
                outputAddress + destination0,
                transposedLocal[row * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination1,
                transposedLocal[(row + 1U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination2,
                transposedLocal[(row + 2U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination3,
                transposedLocal[(row + 3U) * batchVectorElements]);
        }
        PipeBarrier<PIPE_V>();
        if (distinctSecond) {
            AddIndependentFloatVector(
                outputAddress + destination4,
                transposedLocal[(row + 4U) * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination5,
                transposedLocal[(row + 5U) * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination6,
                transposedLocal[(row + 6U) * batchVectorElements]);
            AddIndependentFloatVector(
                outputAddress + destination7,
                transposedLocal[(row + 7U) * batchVectorElements]);
        } else {
            AddIndependentFloatVector(
                outputAddress + destination4,
                transposedLocal[(row + 4U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination5,
                transposedLocal[(row + 5U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination6,
                transposedLocal[(row + 6U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
            AddIndependentFloatVector(
                outputAddress + destination7,
                transposedLocal[(row + 7U) * batchVectorElements]);
        }
    }

    __aicore__ inline void ProcessLevel2FloatGroup(
        __ubuf__ uint64_t* indexPacked, uint32_t packedOffset,
        const LocalTensor<float>& outputLocal,
        const LocalTensor<float>& transposedLocal, uint32_t row,
        uint32_t batchVectorElements, uint32_t batchLength) const
    {
        const uint64_t destination01 = indexPacked[packedOffset];
        const uint64_t destination23 = indexPacked[packedOffset + 1U];
        const uint64_t destination45 = indexPacked[packedOffset + 2U];
        const uint64_t destination67 = indexPacked[packedOffset + 3U];
        const uint32_t destination0 = static_cast<uint32_t>(destination01);
        const uint32_t destination1 = static_cast<uint32_t>(destination01 >> 32U);
        const uint32_t destination2 = static_cast<uint32_t>(destination23);
        const uint32_t destination3 = static_cast<uint32_t>(destination23 >> 32U);
        const uint32_t destination4 = static_cast<uint32_t>(destination45);
        const uint32_t destination5 = static_cast<uint32_t>(destination45 >> 32U);
        const uint32_t destination6 = static_cast<uint32_t>(destination67);
        const uint32_t destination7 = static_cast<uint32_t>(destination67 >> 32U);
        Add(outputLocal[destination0 / sizeof(float)],
            outputLocal[destination0 / sizeof(float)],
            transposedLocal[row * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination1 / sizeof(float)],
            outputLocal[destination1 / sizeof(float)],
            transposedLocal[(row + 1U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination2 / sizeof(float)],
            outputLocal[destination2 / sizeof(float)],
            transposedLocal[(row + 2U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination3 / sizeof(float)],
            outputLocal[destination3 / sizeof(float)],
            transposedLocal[(row + 3U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination4 / sizeof(float)],
            outputLocal[destination4 / sizeof(float)],
            transposedLocal[(row + 4U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination5 / sizeof(float)],
            outputLocal[destination5 / sizeof(float)],
            transposedLocal[(row + 5U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination6 / sizeof(float)],
            outputLocal[destination6 / sizeof(float)],
            transposedLocal[(row + 6U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
        Add(outputLocal[destination7 / sizeof(float)],
            outputLocal[destination7 / sizeof(float)],
            transposedLocal[(row + 7U) * batchVectorElements],
            static_cast<int32_t>(batchLength));
    }

    __aicore__ inline void WaitMte2Vector() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
    }

    __aicore__ inline void WaitVectorMte2() const
    {
        event_t released = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(released);
        WaitFlag<HardEvent::V_MTE2>(released);
    }

    __aicore__ inline void WaitVectorMte3() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
    }

    __aicore__ inline void WaitMte3Vector() const
    {
        event_t released = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(released);
        WaitFlag<HardEvent::MTE3_V>(released);
    }

    __aicore__ inline void WaitMte2Scalar() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ready);
        WaitFlag<HardEvent::MTE2_S>(ready);
    }

    __aicore__ inline void WaitScalarMte2() const
    {
        event_t released = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(released);
        WaitFlag<HardEvent::S_MTE2>(released);
    }

    __aicore__ inline void WaitScalarMte3() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ready);
        WaitFlag<HardEvent::S_MTE3>(ready);
    }

    __aicore__ inline void WaitScalarVector() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(ready);
        WaitFlag<HardEvent::S_V>(ready);
    }

    __aicore__ inline void WaitVectorScalar() const
    {
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
    }

    __aicore__ inline uint32_t GetFullTransposeRows(uint32_t rowCount) const
    {
        constexpr uint32_t rowTile = sizeof(T) == sizeof(float)
            ? kFloatTransposeRows : kTransposeTileElements;
        return rowCount / rowTile * rowTile;
    }

    __aicore__ inline void PrepareStridedOffsets(LocalTensor<int32_t> offsets,
                                                  int32_t firstValue,
                                                  int32_t stride,
                                                  uint32_t count) const
    {
        CreateVecIndex(offsets, firstValue, count);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, stride, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransposeOuterToRows(const LocalTensor<T>& destination,
                                                 const LocalTensor<T>& source,
                                                 uint32_t sourceRowElements,
                                                 uint32_t rowCount,
                                                 uint32_t batchVectorElements) const
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            const uint32_t fullRows =
                rowCount / kFloatTransposeRows * kFloatTransposeRows;
            if (fullRows == 0U) {
                return;
            }
            const TransDataTo5HDParams params = {
                false,
                false,
                static_cast<uint8_t>(fullRows / kFloatTransposeRows),
                static_cast<uint16_t>(kFloatTransposeRows * batchVectorElements * sizeof(T) /
                                      kDataBlockBytes),
                static_cast<uint16_t>(kFloatTransposeRows * sizeof(T) / kDataBlockBytes),
            };
            for (uint32_t outerStart = 0U; outerStart < batchVectorElements;
                 outerStart += kTransposeTileElements) {
                LocalTensor<T> destinationList[kTransposeTileElements];
                LocalTensor<T> sourceList[kTransposeTileElements];
                for (uint32_t lane = 0U; lane < kTransposeTileElements; ++lane) {
                    sourceList[lane] = source[(outerStart + lane) * sourceRowElements];
                }
                for (uint32_t lane = 0U; lane < kFloatTransposeRows; ++lane) {
                    destinationList[2U * lane] =
                        destination[lane * batchVectorElements + outerStart];
                    destinationList[2U * lane + 1U] =
                        destination[lane * batchVectorElements + outerStart + kFloatTransposeRows];
                }
                TransDataTo5HD<T>(destinationList, sourceList, params);
            }
            return;
        }
        const uint32_t fullRows =
            rowCount / kTransposeTileElements * kTransposeTileElements;
        if (fullRows == 0U) {
            return;
        }
        const TransDataTo5HDParams params = {
            false,
            false,
            static_cast<uint8_t>(fullRows / kTransposeTileElements),
            static_cast<uint16_t>(kTransposeTileElements * batchVectorElements * sizeof(T) /
                                  kDataBlockBytes),
            static_cast<uint16_t>(kTransposeTileElements * sizeof(T) / kDataBlockBytes),
        };
        for (uint32_t outerStart = 0U; outerStart < batchVectorElements;
             outerStart += kTransposeTileElements) {
            LocalTensor<T> destinationList[kTransposeTileElements];
            LocalTensor<T> sourceList[kTransposeTileElements];
            for (uint32_t lane = 0U; lane < kTransposeTileElements; ++lane) {
                destinationList[lane] =
                    destination[lane * batchVectorElements + outerStart];
                sourceList[lane] = source[(outerStart + lane) * sourceRowElements];
            }
            TransDataTo5HD<T>(destinationList, sourceList, params);
        }
    }

    __aicore__ inline uint64_t PackTransposeVaConfig(
        uint64_t address0, uint64_t address1,
        uint64_t address2, uint64_t address3) const
    {
        constexpr uint64_t addressMask = 0x1fffULL;
        return ((address0 >> 5U) & addressMask) |
               (((address1 >> 5U) & addressMask) << 16U) |
               (((address2 >> 5U) & addressMask) << 32U) |
               (((address3 >> 5U) & addressMask) << 48U);
    }

    __aicore__ inline void PreparePipelinedFloatTransposeConfigs(
        uint32_t batchVectorElements)
    {
        if constexpr (IsSameType<T, float>::value) {
            constexpr uint32_t slotCount = 2U;
            const uint32_t transposeGroups =
                batchVectorElements / kTransposeTileElements;
            const uint32_t sourceSlotRows = AlignUp(
                sourceChunk_, kDataBlockBytes / sizeof(float));
            const uint32_t sourceSlotElements =
                sourceSlotRows * batchVectorElements;
            LocalTensor<float> destination = batchTransposedBuf_.Get<float>();
            LocalTensor<float> source = sourceRawBuf_.Get<float>();
            LocalTensor<uint64_t> destinationConfig =
                batchTransposeDstConfigBuf_.Get<uint64_t>();
            LocalTensor<uint64_t> sourceConfig =
                batchTransposeSrcConfigBuf_.Get<uint64_t>();
            for (uint32_t group = 0U; group < transposeGroups; ++group) {
                const uint32_t outerStart = group * kTransposeTileElements;
                for (uint32_t config = 0U; config < 4U; ++config) {
                    uint64_t destinationAddress[4];
                    for (uint32_t item = 0U; item < 4U; ++item) {
                        const uint32_t lane = config * 4U + item;
                        const uint32_t destinationRow = lane / 2U;
                        const uint32_t destinationLane =
                            outerStart + (lane & 1U) * kFloatTransposeRows;
                        destinationAddress[item] = destination[
                            destinationRow * batchVectorElements + destinationLane]
                                                       .GetPhyAddr();
                    }
                    destinationConfig.SetValue(
                        group * 4U + config,
                        PackTransposeVaConfig(
                            destinationAddress[0], destinationAddress[1],
                            destinationAddress[2], destinationAddress[3]));
                    for (uint32_t slot = 0U; slot < slotCount; ++slot) {
                        uint64_t sourceAddress[4];
                        for (uint32_t item = 0U; item < 4U; ++item) {
                            const uint32_t lane = config * 4U + item;
                            sourceAddress[item] = source[
                                slot * sourceSlotElements +
                                (outerStart + lane) * sourceSlotRows]
                                                      .GetPhyAddr();
                        }
                        sourceConfig.SetValue(
                            (slot * transposeGroups + group) * 4U + config,
                            PackTransposeVaConfig(
                                sourceAddress[0], sourceAddress[1],
                                sourceAddress[2], sourceAddress[3]));
                    }
                }
            }
        }
    }

    __aicore__ inline void TransposePipelinedFloatSourceCached(
        uint32_t slot, uint32_t rowCount, uint32_t batchVectorElements)
    {
        if constexpr (IsSameType<T, float>::value) {
            const uint32_t transposeGroups =
                batchVectorElements / kTransposeTileElements;
            const TransDataTo5HDParams params = {
                false,
                false,
                static_cast<uint8_t>(rowCount / kFloatTransposeRows),
                static_cast<uint16_t>(
                    kFloatTransposeRows * batchVectorElements * sizeof(float) /
                    kDataBlockBytes),
                static_cast<uint16_t>(
                    kFloatTransposeRows * sizeof(float) / kDataBlockBytes),
            };
            LocalTensor<uint64_t> destinationConfig =
                batchTransposeDstConfigBuf_.Get<uint64_t>();
            LocalTensor<uint64_t> sourceConfig =
                batchTransposeSrcConfigBuf_.Get<uint64_t>();
            for (uint32_t group = 0U; group < transposeGroups; ++group) {
                TransDataTo5HDVldVaRegImpl<float>(
                    reinterpret_cast<__ubuf__ uint64_t*>(
                        destinationConfig[group * 4U].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint64_t*>(
                        sourceConfig[(slot * transposeGroups + group) * 4U]
                            .GetPhyAddr()),
                    params);
            }
        }
    }

    // Transpose a row-major [sourceRows][sourceRowElements] matrix into
    // [batchVectorElements][destinationRowElements].  The float transpose
    // instruction consumes 16 source rows and produces 16 destination rows
    // per repeated tile. Callers may provide a padded final tile when both
    // source and destination storage have the required aligned extent.
    __aicore__ inline void TransposeRowsToOuter(
        const LocalTensor<float>& destination, const LocalTensor<float>& source,
        uint32_t sourceRowElements, uint32_t sourceRows,
        uint32_t destinationRowElements, uint32_t batchVectorElements) const
    {
        const uint32_t fullRows = sourceRows / kTransposeTileElements *
            kTransposeTileElements;
        if (fullRows == 0U) {
            return;
        }
        const TransDataTo5HDParams params = {
            false,
            false,
            static_cast<uint8_t>(fullRows / kTransposeTileElements),
            2U,
            static_cast<uint16_t>(batchVectorElements * 2U),
        };
        for (uint32_t columnStart = 0U; columnStart < batchVectorElements;
             columnStart += kFloatTransposeRows) {
            uint64_t destinationList[kTransposeTileElements];
            uint64_t sourceList[kTransposeTileElements];
            for (uint32_t lane = 0U; lane < kTransposeTileElements; ++lane) {
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[lane * sourceRowElements + columnStart].GetPhyAddr());
            }
            for (uint32_t lane = 0U; lane < kFloatTransposeRows; ++lane) {
                destinationList[2U * lane] = reinterpret_cast<uint64_t>(
                    destination[(columnStart + lane) * destinationRowElements].GetPhyAddr());
                destinationList[2U * lane + 1U] = reinterpret_cast<uint64_t>(
                    destination[(columnStart + lane) * destinationRowElements +
                                kFloatTransposeRows].GetPhyAddr());
            }
            TransDataTo5HD<float>(destinationList, sourceList, params);
        }
    }

    __aicore__ inline void BeginIndependentHalfVectors(uint32_t count) const
    {
        if constexpr (IsSameType<T, half>::value) {
            if ASCEND_IS_AIV {
                set_mask_count();
                set_vector_mask(0UL, static_cast<uint64_t>(count));
            }
        }
    }

    __aicore__ inline void AddIndependentHalfVector(
        const LocalTensor<T>& destination, const LocalTensor<T>& source) const
    {
        if constexpr (IsSameType<T, half>::value) {
            if ASCEND_IS_AIV {
                vadd(reinterpret_cast<__ubuf__ half*>(destination.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ half*>(destination.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ half*>(source.GetPhyAddr()), 1U,
                     DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
                     DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                     DEFAULT_REPEAT_STRIDE);
            }
        }
    }

    __aicore__ inline void ProcessIndependentHalfGroup(
        const LocalTensor<T>& output, const LocalTensor<T>& source,
        __ubuf__ uint64_t* indexPacked, uint32_t packedOffset,
        uint32_t row, uint32_t batchVectorElements) const
    {
        if constexpr (IsSameType<T, half>::value) {
            const uint64_t destination01 = indexPacked[packedOffset];
            const uint64_t destination23 = indexPacked[packedOffset + 1U];
            const uint64_t destination45 = indexPacked[packedOffset + 2U];
            const uint32_t destination0 = static_cast<uint32_t>(destination01);
            const uint32_t destination1 = static_cast<uint32_t>(destination01 >> 32U);
            const uint32_t destination2 = static_cast<uint32_t>(destination23);
            const uint32_t destination3 = static_cast<uint32_t>(destination23 >> 32U);
            const uint32_t destination4 = static_cast<uint32_t>(destination45);
            const uint32_t destination5 = static_cast<uint32_t>(destination45 >> 32U);
            const bool distinctFirst = destination1 != destination0 &&
                destination2 != destination0 && destination2 != destination1;
            const bool distinctSecond = destination4 != destination3 &&
                destination5 != destination3 && destination5 != destination4;
            const bool distinctAcross =
                destination3 != destination0 &&
                destination3 != destination1 &&
                destination3 != destination2 &&
                destination4 != destination0 &&
                destination4 != destination1 &&
                destination4 != destination2 &&
                destination5 != destination0 &&
                destination5 != destination1 &&
                destination5 != destination2;
            AddIndependentHalfVector(output[destination0],
                                     source[row * batchVectorElements]);
            if (!distinctFirst) {
                PipeBarrier<PIPE_V>();
            }
            AddIndependentHalfVector(output[destination1],
                                     source[(row + 1U) * batchVectorElements]);
            if (!distinctFirst) {
                PipeBarrier<PIPE_V>();
            }
            AddIndependentHalfVector(output[destination2],
                                     source[(row + 2U) * batchVectorElements]);
            if (!distinctAcross) {
                PipeBarrier<PIPE_V>();
            }
            AddIndependentHalfVector(output[destination3],
                                     source[(row + 3U) * batchVectorElements]);
            if (!distinctSecond) {
                PipeBarrier<PIPE_V>();
            }
            AddIndependentHalfVector(output[destination4],
                                     source[(row + 4U) * batchVectorElements]);
            if (!distinctSecond) {
                PipeBarrier<PIPE_V>();
            }
            AddIndependentHalfVector(output[destination5],
                                     source[(row + 5U) * batchVectorElements]);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void EndIndependentHalfVectors() const
    {
        if constexpr (IsSameType<T, half>::value) {
            if ASCEND_IS_AIV {
                set_mask_norm();
                set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
    }

    __aicore__ inline void ReducePipelinedHalfChunk(
        const LocalTensor<int32_t>& indexLocal, __ubuf__ uint64_t* indexPacked,
        const LocalTensor<T>& outputLocal, const LocalTensor<T>& sourceLocal,
        const LocalTensor<T>& batchValueLocal,
        const LocalTensor<uint32_t>& sourceOffsetLocal, uint32_t chunkStart,
        uint32_t rows, uint32_t batchVectorElements, uint32_t batchLength) const
    {
        if constexpr (IsSameType<T, half>::value) {
            uint32_t row = 0U;
            for (; row + kGatherPipelineDepth - 1U < rows;
                 row += kGatherPipelineDepth) {
                const uint32_t packedOffset = (chunkStart + row) >> 1U;
                const uint64_t destination01 = indexPacked[packedOffset];
                const uint64_t destination23 = indexPacked[packedOffset + 1U];
                const uint64_t destination45 = indexPacked[packedOffset + 2U];
                const uint32_t destination0 = static_cast<uint32_t>(destination01);
                const uint32_t destination1 = static_cast<uint32_t>(destination01 >> 32U);
                const uint32_t destination2 = static_cast<uint32_t>(destination23);
                const uint32_t destination3 = static_cast<uint32_t>(destination23 >> 32U);
                const uint32_t destination4 = static_cast<uint32_t>(destination45);
                const uint32_t destination5 = static_cast<uint32_t>(destination45 >> 32U);
                for (uint32_t gatherSlot = 0U; gatherSlot < kGatherPipelineDepth;
                     ++gatherSlot) {
                    Gather(batchValueLocal[gatherSlot * batchVectorElements], sourceLocal,
                           sourceOffsetLocal,
                           (row + gatherSlot) * static_cast<uint32_t>(sizeof(T)),
                           batchLength);
                }

                const bool distinctFirst = destination1 != destination0 &&
                    destination2 != destination0 && destination2 != destination1;
                const bool distinctSecond = destination4 != destination3 &&
                    destination5 != destination3 && destination5 != destination4;
                const bool distinctAcross =
                    destination3 != destination0 &&
                    destination3 != destination1 &&
                    destination3 != destination2 &&
                    destination4 != destination0 &&
                    destination4 != destination1 &&
                    destination4 != destination2 &&
                    destination5 != destination0 &&
                    destination5 != destination1 &&
                    destination5 != destination2;
                PipeBarrier<PIPE_V>();
                BeginIndependentHalfVectors(batchLength);
                AddIndependentHalfVector(outputLocal[destination0], batchValueLocal);
                if (!distinctFirst) {
                    PipeBarrier<PIPE_V>();
                }
                AddIndependentHalfVector(
                    outputLocal[destination1], batchValueLocal[batchVectorElements]);
                if (!distinctFirst) {
                    PipeBarrier<PIPE_V>();
                }
                AddIndependentHalfVector(
                    outputLocal[destination2], batchValueLocal[2U * batchVectorElements]);
                if (!distinctAcross) {
                    PipeBarrier<PIPE_V>();
                }
                AddIndependentHalfVector(
                    outputLocal[destination3], batchValueLocal[3U * batchVectorElements]);
                if (!distinctSecond) {
                    PipeBarrier<PIPE_V>();
                }
                AddIndependentHalfVector(
                    outputLocal[destination4], batchValueLocal[4U * batchVectorElements]);
                if (!distinctSecond) {
                    PipeBarrier<PIPE_V>();
                }
                AddIndependentHalfVector(
                    outputLocal[destination5], batchValueLocal[5U * batchVectorElements]);
                EndIndependentHalfVectors();
                PipeBarrier<PIPE_V>();
            }
            for (; row < rows; ++row) {
                const uint32_t destinationOffset =
                    static_cast<uint32_t>(indexLocal(chunkStart + row));
                Gather(batchValueLocal, sourceLocal, sourceOffsetLocal,
                       row * static_cast<uint32_t>(sizeof(T)), batchLength);
                Add(outputLocal[destinationOffset], outputLocal[destinationOffset],
                    batchValueLocal, static_cast<int32_t>(batchLength));
            }
        }
    }

    __aicore__ inline void ProcessPipelinedHalfSources(
        const LocalTensor<int32_t>& indexLocal, __ubuf__ uint64_t* indexPacked,
        const LocalTensor<T>& outputLocal, const LocalTensor<T>& stagingLocal,
        const LocalTensor<T>& batchValueLocal,
        const LocalTensor<uint32_t>& sourceOffsetLocal,
        const LocalTensor<int32_t>& sourceOffsetVector, uint64_t outerStart,
        uint32_t batchLength, uint32_t batchVectorElements,
        uint32_t preparedSourceRowElements) const
    {
        if constexpr (IsSameType<T, half>::value) {
            constexpr uint32_t alignElements = kDataBlockBytes / sizeof(half);
            constexpr uint32_t slotCount = 2U;
            const uint32_t sourceSlotRows = AlignUp(sourceChunk_, alignElements);
            const uint32_t sourceSlotElements = sourceSlotRows * batchVectorElements;
            DataCopyPadExtParams<T> pad = {false, 0U, 0U, static_cast<T>(0)};
            event_t sourceReady[slotCount];
            event_t sourceReleased[slotCount];
            bool releasePending[slotCount] = {false, false};
            for (uint32_t slot = 0U; slot < slotCount; ++slot) {
                sourceReady[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
                sourceReleased[slot] = static_cast<event_t>(
                    GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            }

            WaitVectorMte2();
            const uint32_t firstRows = indexSize_ > sourceChunk_
                ? sourceChunk_ : indexSize_;
            DataCopyExtParams firstCopy = {
                static_cast<uint16_t>(batchLength),
                static_cast<uint32_t>(firstRows * sizeof(T)),
                static_cast<uint32_t>((indexSize_ - firstRows) * sizeof(T)),
                0U,
                0U,
            };
            DataCopyPad(stagingLocal, sourceGm_[outerStart * indexSize_], firstCopy, pad);
            SetFlag<HardEvent::MTE2_V>(sourceReady[0]);

            uint32_t chunkNumber = 0U;
            for (uint32_t chunkStart = 0U; chunkStart < indexSize_;
                 chunkStart += sourceChunk_, ++chunkNumber) {
                const uint32_t currentSlot = chunkNumber & 1U;
                const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                    ? sourceChunk_ : indexSize_ - chunkStart;
                const uint32_t sourceRowElements = AlignUp(rows, alignElements);
                WaitFlag<HardEvent::MTE2_V>(sourceReady[currentSlot]);

                const uint32_t nextStart = chunkStart + sourceChunk_;
                if (nextStart < indexSize_) {
                    const uint32_t nextSlot = currentSlot ^ 1U;
                    const uint32_t nextRows = indexSize_ - nextStart > sourceChunk_
                        ? sourceChunk_ : indexSize_ - nextStart;
                    if (releasePending[nextSlot]) {
                        WaitFlag<HardEvent::V_MTE2>(sourceReleased[nextSlot]);
                        releasePending[nextSlot] = false;
                    }
                    DataCopyExtParams nextCopy = {
                        static_cast<uint16_t>(batchLength),
                        static_cast<uint32_t>(nextRows * sizeof(T)),
                        static_cast<uint32_t>((indexSize_ - nextRows) * sizeof(T)),
                        0U,
                        0U,
                    };
                    DataCopyPad(stagingLocal[nextSlot * sourceSlotElements],
                                sourceGm_[outerStart * indexSize_ + nextStart],
                                nextCopy, pad);
                    SetFlag<HardEvent::MTE2_V>(sourceReady[nextSlot]);
                }

                if (sourceRowElements != preparedSourceRowElements) {
                    PrepareStridedOffsets(
                        sourceOffsetVector, 0,
                        static_cast<int32_t>(sourceRowElements * sizeof(T)),
                        batchLength);
                    preparedSourceRowElements = sourceRowElements;
                }
                ReducePipelinedHalfChunk(
                    indexLocal, indexPacked, outputLocal,
                    stagingLocal[currentSlot * sourceSlotElements], batchValueLocal,
                    sourceOffsetLocal, chunkStart, rows, batchVectorElements,
                    batchLength);
                SetFlag<HardEvent::V_MTE2>(sourceReleased[currentSlot]);
                releasePending[currentSlot] = true;
            }

            for (uint32_t slot = 0U; slot < slotCount; ++slot) {
                if (releasePending[slot]) {
                    WaitFlag<HardEvent::V_MTE2>(sourceReleased[slot]);
                }
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(sourceReady[slot]);
                GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(sourceReleased[slot]);
            }
        }
    }

    __aicore__ inline void ProcessOuterBatchTask(const LocalTensor<int32_t>& indexLocal,
                                                  uint64_t outerStart,
                                                  uint32_t batchLength)
    {
        if constexpr (IsSameType<T, int8_t>::value || IsSameType<T, bfloat16_t>::value) {
            return;
        } else {
            const uint32_t alignElements = kDataBlockBytes / sizeof(T);
            const uint32_t batchVectorElements = sourceTranspose_ != 0U
                ? AlignUp(outerBatch_, kTransposeTileElements)
                : AlignUp(outerBatch_, alignElements);
            LocalTensor<T> outputLocal = outputBuf_.Get<T>();
            LocalTensor<T> stagingLocal = sourceRawBuf_.Get<T>();
            LocalTensor<T> transposedLocal;
            if (sourceTranspose_ != 0U) {
                transposedLocal = batchTransposedBuf_.Get<T>();
            }
            LocalTensor<T> batchValueLocal = sourcePromoteBuf_.Get<T>();
            __ubuf__ uint64_t* indexPacked =
                reinterpret_cast<__ubuf__ uint64_t*>(indexLocal.GetPhyAddr());
            LocalTensor<uint32_t> outputOffsetLocal = batchOutputOffsetBuf_.Get<uint32_t>();
            LocalTensor<uint32_t> sourceOffsetLocal = batchSourceOffsetBuf_.Get<uint32_t>();
            LocalTensor<uint32_t> axisOffsetLocal = batchAxisOffsetBuf_.Get<uint32_t>();
            LocalTensor<int32_t> outputOffsetVector = batchOutputOffsetBuf_.Get<int32_t>();
            LocalTensor<int32_t> sourceOffsetVector = batchSourceOffsetBuf_.Get<int32_t>();
            LocalTensor<int32_t> axisOffsetVector = batchAxisOffsetBuf_.Get<int32_t>();
            DataCopyPadExtParams<T> pad = {false, 0U, 0U, static_cast<T>(0)};

            const uint64_t selfBase = outerStart * axisSize_;
            uint32_t preparedSourceRowElements = 0U;
            if (sourceTranspose_ == 0U) {
                const uint32_t firstSourceRows = indexSize_ > sourceChunk_
                    ? sourceChunk_ : indexSize_;
                preparedSourceRowElements = AlignUp(firstSourceRows, alignElements);
                PrepareStridedOffsets(
                    sourceOffsetVector, 0,
                    static_cast<int32_t>(preparedSourceRowElements * sizeof(T)), batchLength);
            }
            bool stagingReadPending = false;
            for (uint32_t axisStart = 0U; axisStart < axisSize_;
                 axisStart += batchAxisChunk_) {
                const uint32_t rows = axisSize_ - axisStart > batchAxisChunk_
                    ? batchAxisChunk_ : axisSize_ - axisStart;
                const uint32_t axisRowElements = AlignUp(rows, alignElements);
                if (stagingReadPending) {
                    WaitVectorMte2();
                }
                PrepareStridedOffsets(
                    outputOffsetVector, 0,
                    static_cast<int32_t>(axisRowElements * sizeof(T)), batchLength);
                DataCopyExtParams selfCopy = {
                    static_cast<uint16_t>(batchLength),
                    static_cast<uint32_t>(rows * sizeof(T)),
                    static_cast<uint32_t>((axisSize_ - rows) * sizeof(T)),
                    0U,
                    0U,
                };
                DataCopyPad(stagingLocal, selfGm_[selfBase + axisStart], selfCopy, pad);
                WaitMte2Vector();
                for (uint32_t axis = 0U; axis < rows; ++axis) {
                    Gather(outputLocal[(axisStart + axis) * batchVectorElements], stagingLocal,
                           outputOffsetLocal, axis * static_cast<uint32_t>(sizeof(T)),
                           batchLength);
                }
                stagingReadPending = true;
            }

            bool sourceReductionComplete = false;
            if constexpr (IsSameType<T, half>::value) {
                if (sourceBufferCount_ == 2U && sourceTranspose_ == 0U) {
                    ProcessPipelinedHalfSources(
                        indexLocal, indexPacked, outputLocal, stagingLocal,
                        batchValueLocal, sourceOffsetLocal, sourceOffsetVector,
                        outerStart, batchLength, batchVectorElements,
                        preparedSourceRowElements);
                    sourceReductionComplete = true;
                }
            }
            if (!sourceReductionComplete) {
              for (uint32_t chunkStart = 0U; chunkStart < indexSize_; chunkStart += sourceChunk_) {
                const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                    ? sourceChunk_ : indexSize_ - chunkStart;
                const uint32_t sourceRowElements = AlignUp(rows, alignElements);
                const uint32_t fullRows = sourceTranspose_ != 0U
                    ? GetFullTransposeRows(rows) : 0U;
                const uint64_t sourceBase = outerStart * indexSize_ + chunkStart;
                WaitVectorMte2();
                if (fullRows < rows && sourceRowElements != preparedSourceRowElements) {
                    PrepareStridedOffsets(
                        sourceOffsetVector, 0,
                        static_cast<int32_t>(sourceRowElements * sizeof(T)), batchLength);
                    preparedSourceRowElements = sourceRowElements;
                }
                DataCopyExtParams sourceCopy = {
                    static_cast<uint16_t>(batchLength),
                    static_cast<uint32_t>(rows * sizeof(T)),
                    static_cast<uint32_t>((indexSize_ - rows) * sizeof(T)),
                    0U,
                    0U,
                };
                DataCopyPad(stagingLocal, sourceGm_[sourceBase], sourceCopy, pad);
                WaitMte2Vector();
                if (sourceTranspose_ != 0U) {
                    TransposeOuterToRows(transposedLocal, stagingLocal, sourceRowElements, rows,
                                         batchVectorElements);
                }

                uint32_t row = 0U;
                if constexpr (IsSameType<T, half>::value) {
                    if (sourceTranspose_ != 0U) {
                        PipeBarrier<PIPE_V>();
                        BeginIndependentHalfVectors(batchLength);
                        for (; row + kGatherPipelineDepth - 1U < fullRows;
                             row += kGatherPipelineDepth) {
                            ProcessIndependentHalfGroup(
                                outputLocal, transposedLocal, indexPacked,
                                (chunkStart + row) >> 1U, row, batchVectorElements);
                        }
                        EndIndependentHalfVectors();
                    }
                }
                for (; row < fullRows; ++row) {
                    const uint32_t destinationOffset =
                        static_cast<uint32_t>(indexLocal(chunkStart + row));
                    Add(outputLocal[destinationOffset],
                        outputLocal[destinationOffset],
                        transposedLocal[row * batchVectorElements],
                        static_cast<int32_t>(batchLength));
                }
                row = fullRows;
                for (; row + kGatherPipelineDepth - 1U < rows;
                     row += kGatherPipelineDepth) {
                    const uint32_t packedOffset = (chunkStart + row) >> 1U;
                    const uint64_t destination01 = indexPacked[packedOffset];
                    const uint64_t destination23 = indexPacked[packedOffset + 1U];
                    const uint64_t destination45 = indexPacked[packedOffset + 2U];
                    const uint32_t destination0 = static_cast<uint32_t>(destination01);
                    const uint32_t destination1 = static_cast<uint32_t>(destination01 >> 32U);
                    const uint32_t destination2 = static_cast<uint32_t>(destination23);
                    const uint32_t destination3 = static_cast<uint32_t>(destination23 >> 32U);
                    const uint32_t destination4 = static_cast<uint32_t>(destination45);
                    const uint32_t destination5 = static_cast<uint32_t>(destination45 >> 32U);
                    uint32_t destinationOffset[kGatherPipelineDepth] = {
                        destination0, destination1, destination2,
                        destination3, destination4, destination5,
                    };
                    for (uint32_t slot = 0U; slot < kGatherPipelineDepth; ++slot) {
                        Gather(batchValueLocal[slot * batchVectorElements], stagingLocal,
                               sourceOffsetLocal,
                               (row + slot) * static_cast<uint32_t>(sizeof(T)), batchLength);
                    }
                    if constexpr (IsSameType<T, half>::value) {
                        if (sourceTranspose_ == 0U) {
                            const bool distinctFirst = destination1 != destination0 &&
                                destination2 != destination0 && destination2 != destination1;
                            const bool distinctSecond = destination4 != destination3 &&
                                destination5 != destination3 && destination5 != destination4;
                            const bool distinctAcross =
                                destination3 != destination0 &&
                                destination3 != destination1 &&
                                destination3 != destination2 &&
                                destination4 != destination0 &&
                                destination4 != destination1 &&
                                destination4 != destination2 &&
                                destination5 != destination0 &&
                                destination5 != destination1 &&
                                destination5 != destination2;
                            PipeBarrier<PIPE_V>();
                            BeginIndependentHalfVectors(batchLength);
                            AddIndependentHalfVector(outputLocal[destination0], batchValueLocal);
                            if (!distinctFirst) {
                                PipeBarrier<PIPE_V>();
                            }
                            AddIndependentHalfVector(
                                outputLocal[destination1],
                                batchValueLocal[batchVectorElements]);
                            if (!distinctFirst) {
                                PipeBarrier<PIPE_V>();
                            }
                            AddIndependentHalfVector(
                                outputLocal[destination2],
                                batchValueLocal[2U * batchVectorElements]);
                            if (!distinctAcross) {
                                PipeBarrier<PIPE_V>();
                            }
                            AddIndependentHalfVector(
                                outputLocal[destination3],
                                batchValueLocal[3U * batchVectorElements]);
                            if (!distinctSecond) {
                                PipeBarrier<PIPE_V>();
                            }
                            AddIndependentHalfVector(
                                outputLocal[destination4],
                                batchValueLocal[4U * batchVectorElements]);
                            if (!distinctSecond) {
                                PipeBarrier<PIPE_V>();
                            }
                            AddIndependentHalfVector(
                                outputLocal[destination5],
                                batchValueLocal[5U * batchVectorElements]);
                            EndIndependentHalfVectors();
                            PipeBarrier<PIPE_V>();
                        } else {
                            for (uint32_t slot = 0U; slot < kGatherPipelineDepth; ++slot) {
                                Add(outputLocal[destinationOffset[slot]],
                                    outputLocal[destinationOffset[slot]],
                                    batchValueLocal[slot * batchVectorElements],
                                    static_cast<int32_t>(batchLength));
                            }
                        }
                    } else {
                        for (uint32_t slot = 0U; slot < kGatherPipelineDepth; ++slot) {
                            Add(outputLocal[destinationOffset[slot]],
                                outputLocal[destinationOffset[slot]],
                                batchValueLocal[slot * batchVectorElements],
                                static_cast<int32_t>(batchLength));
                        }
                    }
                }
                for (; row < rows; ++row) {
                    const uint32_t destinationOffset =
                        static_cast<uint32_t>(indexLocal(chunkStart + row));
                    Gather(batchValueLocal, stagingLocal, sourceOffsetLocal,
                           row * static_cast<uint32_t>(sizeof(T)), batchLength);
                    Add(outputLocal[destinationOffset],
                        outputLocal[destinationOffset], batchValueLocal,
                        static_cast<int32_t>(batchLength));
                }
              }
            }

            bool outputCopyPending = false;
            for (uint32_t axisStart = 0U; axisStart < axisSize_;
                 axisStart += batchAxisChunk_) {
                const uint32_t rows = axisSize_ - axisStart > batchAxisChunk_
                    ? batchAxisChunk_ : axisSize_ - axisStart;
                const uint32_t axisRowElements = AlignUp(rows, alignElements);
                if (outputCopyPending) {
                    WaitMte3Vector();
                }
                PrepareStridedOffsets(
                    axisOffsetVector, static_cast<int32_t>(axisStart),
                    static_cast<int32_t>(batchVectorElements * sizeof(T)), rows);
                for (uint32_t outer = 0U; outer < batchLength; ++outer) {
                    Gather(stagingLocal[outer * axisRowElements], outputLocal,
                           axisOffsetLocal, outer * static_cast<uint32_t>(sizeof(T)), rows);
                }
                WaitVectorMte3();
                DataCopyExtParams outputCopy = {
                    static_cast<uint16_t>(batchLength),
                    static_cast<uint32_t>(rows * sizeof(T)),
                    0U,
                    static_cast<uint32_t>((axisSize_ - rows) * sizeof(T)),
                    0U,
                };
                DataCopyPad(outputGm_[selfBase + axisStart], stagingLocal, outputCopy);
                outputCopyPending = true;
            }
        }
    }

    __aicore__ inline void ProcessPipelinedFloatTransposeSources(
        __ubuf__ uint64_t* indexPacked, const LocalTensor<float>& outputLocal,
        const LocalTensor<float>& stagingLocal,
        const LocalTensor<float>& transposedLocal, uint64_t outerStart,
        uint32_t batchLength, uint32_t batchVectorElements)
    {
        constexpr uint32_t alignElements = kDataBlockBytes / sizeof(float);
        constexpr uint32_t slotCount = 2U;
        const uint32_t sourceSlotRows = AlignUp(sourceChunk_, alignElements);
        const uint32_t sourceSlotElements = sourceSlotRows * batchVectorElements;
        const uint64_t outputAddress = outputLocal.GetPhyAddr();
        DataCopyPadExtParams<float> pad = {false, 0U, 0U, 0.0F};
        event_t sourceReady[slotCount];
        event_t sourceReleased[slotCount];
        bool releasePending[slotCount] = {false, false};
        for (uint32_t slot = 0U; slot < slotCount; ++slot) {
            sourceReady[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            sourceReleased[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        }

        WaitVectorMte2();
        const uint32_t firstRows = indexSize_ > sourceChunk_
            ? sourceChunk_ : indexSize_;
        DataCopyExtParams firstCopy = {
            static_cast<uint16_t>(batchLength),
            static_cast<uint32_t>(firstRows * sizeof(float)),
            static_cast<uint32_t>((indexSize_ - firstRows) * sizeof(float)),
            0U,
            0U,
        };
        DataCopyPad(stagingLocal, sourceGm_[outerStart * indexSize_], firstCopy, pad);
        SetFlag<HardEvent::MTE2_V>(sourceReady[0]);

        const bool useIndependentMask = batchLength <= 64U;
        if (useIndependentMask) {
            SetIndependentFloatMask(batchLength);
        }
        uint32_t chunkNumber = 0U;
        for (uint32_t chunkStart = 0U; chunkStart < indexSize_;
             chunkStart += sourceChunk_, ++chunkNumber) {
            const uint32_t currentSlot = chunkNumber & 1U;
            const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                ? sourceChunk_ : indexSize_ - chunkStart;
            const uint32_t sourceRowElements = AlignUp(rows, alignElements);
            WaitFlag<HardEvent::MTE2_V>(sourceReady[currentSlot]);

            const uint32_t nextStart = chunkStart + sourceChunk_;
            if (nextStart < indexSize_) {
                const uint32_t nextSlot = currentSlot ^ 1U;
                const uint32_t nextRows = indexSize_ - nextStart > sourceChunk_
                    ? sourceChunk_ : indexSize_ - nextStart;
                if (releasePending[nextSlot]) {
                    WaitFlag<HardEvent::V_MTE2>(sourceReleased[nextSlot]);
                    releasePending[nextSlot] = false;
                }
                DataCopyExtParams nextCopy = {
                    static_cast<uint16_t>(batchLength),
                    static_cast<uint32_t>(nextRows * sizeof(float)),
                    static_cast<uint32_t>((indexSize_ - nextRows) * sizeof(float)),
                    0U,
                    0U,
                };
                DataCopyPad(stagingLocal[nextSlot * sourceSlotElements],
                            sourceGm_[outerStart * indexSize_ + nextStart], nextCopy, pad);
                SetFlag<HardEvent::MTE2_V>(sourceReady[nextSlot]);
            }

            if (rows == sourceChunk_) {
                TransposePipelinedFloatSourceCached(
                    currentSlot, rows, batchVectorElements);
            } else {
                TransposeOuterToRows(
                    transposedLocal, stagingLocal[currentSlot * sourceSlotElements],
                    sourceRowElements, rows, batchVectorElements);
            }
            SetFlag<HardEvent::V_MTE2>(sourceReleased[currentSlot]);
            releasePending[currentSlot] = true;

            uint32_t row = 0U;
            if (useIndependentMask) {
                for (; row + 7U < rows; row += 8U) {
                    ProcessIndependentFloatGroup(
                        indexPacked, (chunkStart + row) >> 1U, outputAddress,
                        transposedLocal, row, batchVectorElements);
                }
            } else {
                for (; row + 7U < rows; row += 8U) {
                    ProcessLevel2FloatGroup(
                        indexPacked, (chunkStart + row) >> 1U, outputLocal,
                        transposedLocal, row, batchVectorElements, batchLength);
                }
            }
        }

        for (uint32_t slot = 0U; slot < slotCount; ++slot) {
            if (releasePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(sourceReleased[slot]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(sourceReady[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(sourceReleased[slot]);
        }
    }

    __aicore__ inline void ProcessFloatOuterBatchTask(
        const LocalTensor<int32_t>& indexLocal, uint64_t outerStart, uint32_t batchLength)
    {
        const uint32_t alignElements = kDataBlockBytes / sizeof(float);
        const uint32_t batchVectorElements = sourceTranspose_ != 0U
            ? AlignUp(outerBatch_, kTransposeTileElements)
            : AlignUp(outerBatch_, alignElements);
        LocalTensor<float> outputLocal = outputBuf_.Get<float>();
        const uint64_t outputAddress = outputLocal.GetPhyAddr();
        __ubuf__ uint64_t* indexPacked =
            reinterpret_cast<__ubuf__ uint64_t*>(indexLocal.GetPhyAddr());
        LocalTensor<float> stagingLocal = sourceRawBuf_.Get<float>();
        LocalTensor<float> transposedLocal;
        if (sourceTranspose_ != 0U) {
            transposedLocal = batchTransposedBuf_.Get<float>();
        }
        LocalTensor<float> batchValueLocal = sourcePromoteBuf_.Get<float>();
        LocalTensor<uint32_t> outputOffsetLocal = batchOutputOffsetBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> sourceOffsetLocal = batchSourceOffsetBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> axisOffsetLocal = batchAxisOffsetBuf_.Get<uint32_t>();
        LocalTensor<int32_t> axisOffsetVector = batchAxisOffsetBuf_.Get<int32_t>();
        DataCopyPadExtParams<float> pad = {false, 0U, 0U, 0.0F};

        const uint64_t selfBase = outerStart * axisSize_;
        bool stagingReadPending = false;
        for (uint32_t axisStart = 0U; axisStart < axisSize_;
             axisStart += batchAxisChunk_) {
            const uint32_t rows = axisSize_ - axisStart > batchAxisChunk_
                ? batchAxisChunk_ : axisSize_ - axisStart;
            const uint32_t axisRowElements = AlignUp(rows, alignElements);
            const uint32_t fullRows = sourceTranspose_ != 0U
                ? AlignUp(rows, kFloatTransposeRows) : 0U;
            if (stagingReadPending) {
                WaitVectorMte2();
                WaitVectorScalar();
            }
            if (fullRows < rows) {
                for (uint32_t outer = 0U; outer < batchLength; ++outer) {
                    outputOffsetLocal.SetValue(
                        outer, outer * axisRowElements * static_cast<uint32_t>(sizeof(float)));
                }
            }
            DataCopyExtParams selfCopy = {
                static_cast<uint16_t>(batchLength),
                static_cast<uint32_t>(rows * sizeof(float)),
                static_cast<uint32_t>((axisSize_ - rows) * sizeof(float)),
                0U,
                0U,
            };
            DataCopyPad(stagingLocal, selfGm_[selfBase + axisStart], selfCopy, pad);
            WaitMte2Vector();
            if (sourceTranspose_ != 0U) {
                TransposeOuterToRows(outputLocal[axisStart * batchVectorElements],
                                     stagingLocal, axisRowElements, fullRows,
                                     batchVectorElements);
            }
            if (fullRows < rows) {
                WaitScalarVector();
                for (uint32_t axis = fullRows; axis < rows; ++axis) {
                    Gather(outputLocal[(axisStart + axis) * batchVectorElements], stagingLocal,
                           outputOffsetLocal, axis * static_cast<uint32_t>(sizeof(float)),
                           batchLength);
                }
            }
            stagingReadPending = true;
        }

        bool sourceReductionComplete = false;
        if (sourceTranspose_ != 0U && sourceBufferCount_ == 2U &&
            (indexSize_ % kFloatTransposeRows) == 0U) {
            ProcessPipelinedFloatTransposeSources(
                indexPacked, outputLocal, stagingLocal, transposedLocal, outerStart,
                batchLength, batchVectorElements);
            sourceReductionComplete = true;
        }
        if (!sourceReductionComplete) {
          event_t sourceReleased =
              static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
          bool sourceReleasePending = false;
          const bool useIndependentMask = sourceTranspose_ != 0U && batchLength <= 64U;
          if (useIndependentMask) {
              SetIndependentFloatMask(batchLength);
          }
          uint32_t preparedSourceRowElements = 0U;
          for (uint32_t chunkStart = 0U; chunkStart < indexSize_; chunkStart += sourceChunk_) {
            const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                ? sourceChunk_ : indexSize_ - chunkStart;
            const uint32_t sourceRowElements = AlignUp(rows, alignElements);
            const uint32_t fullRows = sourceTranspose_ != 0U
                ? GetFullTransposeRows(rows) : 0U;
            const uint64_t sourceBase = outerStart * indexSize_ + chunkStart;
            if (sourceReleasePending) {
                WaitFlag<HardEvent::V_MTE2>(sourceReleased);
                sourceReleasePending = false;
            } else {
                WaitVectorMte2();
            }
            if (fullRows < rows && sourceRowElements != preparedSourceRowElements) {
                WaitVectorScalar();
                for (uint32_t outer = 0U; outer < batchLength; ++outer) {
                    sourceOffsetLocal.SetValue(
                        outer, outer * sourceRowElements * static_cast<uint32_t>(sizeof(float)));
                }
                WaitScalarVector();
                preparedSourceRowElements = sourceRowElements;
            }
            DataCopyExtParams sourceCopy = {
                static_cast<uint16_t>(batchLength),
                static_cast<uint32_t>(rows * sizeof(float)),
                static_cast<uint32_t>((indexSize_ - rows) * sizeof(float)),
                0U,
                0U,
            };
            DataCopyPad(stagingLocal, sourceGm_[sourceBase], sourceCopy, pad);
            WaitMte2Vector();
            if (sourceTranspose_ != 0U) {
                TransposeOuterToRows(transposedLocal, stagingLocal, sourceRowElements, rows,
                                     batchVectorElements);
            }
            const bool releaseSourceEarly = sourceTranspose_ != 0U && fullRows == rows;
            if (releaseSourceEarly) {
                SetFlag<HardEvent::V_MTE2>(sourceReleased);
                sourceReleasePending = true;
            }

            uint32_t row = 0U;
            if (useIndependentMask) {
                for (; row + 7U < fullRows; row += 8U) {
                    ProcessIndependentFloatGroup(
                        indexPacked, (chunkStart + row) >> 1U, outputAddress,
                        transposedLocal, row, batchVectorElements);
                }
            } else {
                for (; row + 7U < fullRows; row += 8U) {
                    ProcessLevel2FloatGroup(
                        indexPacked, (chunkStart + row) >> 1U, outputLocal,
                        transposedLocal, row, batchVectorElements, batchLength);
                }
            }
            for (; row < fullRows; ++row) {
                const uint32_t destination =
                    static_cast<uint32_t>(indexLocal(chunkStart + row));
                Add(outputLocal[destination / sizeof(float)],
                    outputLocal[destination / sizeof(float)],
                    transposedLocal[row * batchVectorElements],
                    static_cast<int32_t>(batchLength));
            }
            const bool hasSourceTail = row < rows;
            for (; row < rows; ++row) {
                const uint32_t destination =
                    static_cast<uint32_t>(indexLocal(chunkStart + row));
                Gather(batchValueLocal, stagingLocal, sourceOffsetLocal,
                       row * static_cast<uint32_t>(sizeof(float)), batchLength);
                const uint32_t destinationOffset = sourceTranspose_ != 0U
                    ? destination / sizeof(float) : destination * batchVectorElements;
                Add(outputLocal[destinationOffset],
                    outputLocal[destinationOffset], batchValueLocal,
                    static_cast<int32_t>(batchLength));
            }
            if (hasSourceTail && useIndependentMask) {
                // Tail helpers restore a full mask.  Reinstall the task-wide
                // raw-VADD mask before a later transposable source chunk.
                SetIndependentFloatMask(batchLength);
            }
            if (!releaseSourceEarly) {
                SetFlag<HardEvent::V_MTE2>(sourceReleased);
                sourceReleasePending = true;
            }
          }
          if (sourceReleasePending) {
              WaitFlag<HardEvent::V_MTE2>(sourceReleased);
          }
        }

        const bool hasOutputStagingAlt = sourceTranspose_ != 0U &&
            AlignUp(batchAxisChunk_, alignElements) <= sourceChunk_;
        LocalTensor<float> outputStagingAlt;
        if (hasOutputStagingAlt) {
            outputStagingAlt = batchTransposedBuf_.Get<float>();
            // The source-transpose buffer is repurposed only after every
            // reduction has issued its final vector read.
            PipeBarrier<PIPE_V>();
        }
        bool outputBufferPending[2] = {false, false};
        uint32_t outputBufferIndex = 0U;
        for (uint32_t axisStart = 0U; axisStart < axisSize_;
             axisStart += batchAxisChunk_) {
            const uint32_t rows = axisSize_ - axisStart > batchAxisChunk_
                ? batchAxisChunk_ : axisSize_ - axisStart;
            const uint32_t axisRowElements = AlignUp(rows, alignElements);
            const uint32_t outputSlot = hasOutputStagingAlt
                ? outputBufferIndex & 1U : 0U;
            if (outputBufferPending[outputSlot]) {
                WaitMte3Vector();
            }
            LocalTensor<float> outputStaging = outputSlot == 0U
                ? stagingLocal : outputStagingAlt;
            const uint32_t paddedRows = AlignUp(rows, kTransposeTileElements);
            const bool canTransposePadding =
                AlignUp(rows, alignElements) == paddedRows;
            const uint32_t fullRows = sourceTranspose_ != 0U
                ? (canTransposePadding ? paddedRows
                                       : rows / kTransposeTileElements * kTransposeTileElements)
                : 0U;
            if (fullRows != 0U) {
                TransposeRowsToOuter(
                    outputStaging, outputLocal[axisStart * batchVectorElements],
                    batchVectorElements, fullRows, axisRowElements, batchVectorElements);
            }
            if (fullRows < rows) {
                PipeBarrier<PIPE_V>();
                PrepareStridedOffsets(
                    axisOffsetVector, static_cast<int32_t>(axisStart + fullRows),
                    static_cast<int32_t>(batchVectorElements * sizeof(float)), rows - fullRows);
                for (uint32_t outer = 0U; outer < batchLength; ++outer) {
                    Gather(outputStaging[outer * axisRowElements + fullRows], outputLocal,
                           axisOffsetLocal, outer * static_cast<uint32_t>(sizeof(float)),
                           rows - fullRows);
                }
            }
            WaitVectorMte3();
            DataCopyExtParams outputCopy = {
                static_cast<uint16_t>(batchLength),
                static_cast<uint32_t>(rows * sizeof(float)),
                0U,
                static_cast<uint32_t>((axisSize_ - rows) * sizeof(float)),
                0U,
            };
            DataCopyPad(outputGm_[selfBase + axisStart], outputStaging, outputCopy);
            outputBufferPending[outputSlot] = true;
            ++outputBufferIndex;
        }
    }

    __aicore__ inline void ProcessNativeTask(const LocalTensor<int32_t>& indexLocal, uint64_t outer,
                                             uint64_t tileStart, uint32_t tileLength)
    {
        const uint32_t alignedTile = AlignUp(tileLength, kDataBlockBytes / sizeof(T));
        LocalTensor<T> outputLocal = outputBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceRawBuf_.Get<T>();
        DataCopyPadExtParams<T> pad = {false, 0U, 0U, static_cast<T>(0)};
        DataCopyExtParams selfCopy = {
            static_cast<uint16_t>(axisSize_),
            static_cast<uint32_t>(tileLength * sizeof(T)),
            static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
            0U,
            0U,
        };
        DataCopyExtParams outputCopy = {
            static_cast<uint16_t>(axisSize_),
            static_cast<uint32_t>(tileLength * sizeof(T)),
            0U,
            static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
            0U,
        };

        const uint64_t selfBase = outer * axisSize_ * innerSize_ + tileStart;
        DataCopyPad(outputLocal, selfGm_[selfBase], selfCopy, pad);

        bool sourceInUse = false;
        for (uint32_t chunkStart = 0U; chunkStart < indexSize_; chunkStart += sourceChunk_) {
            if (sourceInUse) {
                if constexpr (IsSameType<T, int8_t>::value) {
                    WaitScalarMte2();
                } else {
                    WaitVectorMte2();
                }
            }
            const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                ? sourceChunk_ : indexSize_ - chunkStart;
            const uint64_t sourceBase =
                (outer * indexSize_ + chunkStart) * innerSize_ + tileStart;
            DataCopyExtParams sourceCopy = {
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(tileLength * sizeof(T)),
                static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
                0U,
                0U,
            };
            DataCopyPad(sourceLocal, sourceGm_[sourceBase], sourceCopy, pad);
            if constexpr (IsSameType<T, int8_t>::value) {
                WaitMte2Scalar();
            } else {
                WaitMte2Vector();
            }
            for (uint32_t row = 0U; row < rows; ++row) {
                const int32_t destination = indexLocal(chunkStart + row);
                const uint32_t destinationBase = static_cast<uint32_t>(destination) * alignedTile;
                const uint32_t sourceBaseLocal = row * alignedTile;
                if constexpr (IsSameType<T, int8_t>::value) {
                    __ubuf__ int8_t* outputPtr =
                        reinterpret_cast<__ubuf__ int8_t*>(outputLocal.GetPhyAddr());
                    __ubuf__ int8_t* sourcePtr =
                        reinterpret_cast<__ubuf__ int8_t*>(sourceLocal.GetPhyAddr());
                    __ubuf__ uint32_t* outputWordPtr =
                        reinterpret_cast<__ubuf__ uint32_t*>(outputLocal.GetPhyAddr());
                    __ubuf__ uint32_t* sourceWordPtr =
                        reinterpret_cast<__ubuf__ uint32_t*>(sourceLocal.GetPhyAddr());
                    constexpr uint32_t byteLowMask = 0x7f7f7f7fU;
                    constexpr uint32_t byteHighMask = 0x80808080U;
                    uint32_t element = 0U;
                    for (; element + sizeof(uint32_t) <= tileLength;
                         element += sizeof(uint32_t)) {
                        const uint32_t outputWord =
                            outputWordPtr[(destinationBase + element) / sizeof(uint32_t)];
                        const uint32_t sourceWord =
                            sourceWordPtr[(sourceBaseLocal + element) / sizeof(uint32_t)];
                        const uint32_t lowSum =
                            (outputWord & byteLowMask) + (sourceWord & byteLowMask);
                        const uint32_t result = (lowSum & byteLowMask) |
                            ((lowSum & byteHighMask) ^
                             ((outputWord ^ sourceWord) & byteHighMask));
                        outputWordPtr[(destinationBase + element) / sizeof(uint32_t)] = result;
                    }
                    for (; element < tileLength; ++element) {
                        int32_t sum = static_cast<int32_t>(outputPtr[destinationBase + element]);
                        sum += static_cast<int32_t>(sourcePtr[sourceBaseLocal + element]);
                        sum %= 256;
                        if (sum > 127) {
                            sum -= 256;
                        } else if (sum < -128) {
                            sum += 256;
                        }
                        outputPtr[destinationBase + element] = static_cast<int8_t>(sum);
                    }
                } else {
                    Add(outputLocal[destinationBase], outputLocal[destinationBase],
                        sourceLocal[sourceBaseLocal], static_cast<int32_t>(tileLength));
                }
            }
            sourceInUse = true;
        }

        if constexpr (IsSameType<T, int8_t>::value) {
            WaitScalarMte3();
        } else {
            WaitVectorMte3();
        }
        DataCopyPad(outputGm_[selfBase], outputLocal, outputCopy);
    }

    __aicore__ inline void ProcessBfloat16Task(const LocalTensor<int32_t>& indexLocal,
                                               uint64_t outer, uint64_t tileStart,
                                               uint32_t tileLength)
    {
        const uint32_t alignedTile = AlignUp(tileLength, kDataBlockBytes / sizeof(T));
        LocalTensor<T> outputLocal = outputBuf_.Get<T>();
        LocalTensor<T> sourceRawLocal = sourceRawBuf_.Get<T>();
        LocalTensor<float> sourceLocal = sourcePromoteBuf_.Get<float>();
        LocalTensor<float> outputPromoteLocal = bfloatOutputPromoteBuf_.Get<float>();
        DataCopyPadExtParams<T> pad = {false, 0U, 0U, static_cast<T>(0)};
        DataCopyExtParams selfCopy = {
            static_cast<uint16_t>(axisSize_),
            static_cast<uint32_t>(tileLength * sizeof(T)),
            static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
            0U,
            0U,
        };
        DataCopyExtParams outputCopy = {
            static_cast<uint16_t>(axisSize_),
            static_cast<uint32_t>(tileLength * sizeof(T)),
            0U,
            static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
            0U,
        };

        const uint64_t selfBase = outer * axisSize_ * innerSize_ + tileStart;
        DataCopyPad(outputLocal, selfGm_[selfBase], selfCopy, pad);
        WaitMte2Vector();

        uint32_t currentRawSlot = 0U;
        const uint32_t firstRows = indexSize_ > sourceChunk_ ? sourceChunk_ : indexSize_;
        DataCopyExtParams firstSourceCopy = {
            static_cast<uint16_t>(firstRows),
            static_cast<uint32_t>(tileLength * sizeof(T)),
            static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
            0U,
            0U,
        };
        WaitVectorMte2();
        DataCopyPad(sourceRawLocal, sourceGm_[outer * indexSize_ * innerSize_ + tileStart],
                    firstSourceCopy, pad);
        WaitMte2Vector();

        for (uint32_t chunkStart = 0U; chunkStart < indexSize_; chunkStart += sourceChunk_) {
            const uint32_t rows = indexSize_ - chunkStart > sourceChunk_
                ? sourceChunk_ : indexSize_ - chunkStart;
            const uint32_t nextChunkStart = chunkStart + rows;
            const uint32_t nextRawSlot = currentRawSlot ^ 1U;
            if (nextChunkStart < indexSize_) {
                if (chunkStart != 0U) {
                    WaitVectorMte2();
                }
                const uint32_t nextRows = indexSize_ - nextChunkStart > sourceChunk_
                    ? sourceChunk_ : indexSize_ - nextChunkStart;
                const uint64_t nextSourceBase =
                    (outer * indexSize_ + nextChunkStart) * innerSize_ + tileStart;
                DataCopyExtParams nextSourceCopy = {
                    static_cast<uint16_t>(nextRows),
                    static_cast<uint32_t>(tileLength * sizeof(T)),
                    static_cast<uint32_t>((innerSize_ - tileLength) * sizeof(T)),
                    0U,
                    0U,
                };
                DataCopyPad(
                    sourceRawLocal[nextRawSlot * sourceChunk_ * alignedTile],
                    sourceGm_[nextSourceBase], nextSourceCopy, pad);
            }
            Cast(sourceLocal,
                 sourceRawLocal[currentRawSlot * sourceChunk_ * alignedTile],
                 RoundMode::CAST_NONE,
                 static_cast<int32_t>(rows * alignedTile));
            uint32_t row = 0U;
            for (; row + kBfloatPipelineDepth - 1U < rows;
                 row += kBfloatPipelineDepth) {
                const uint32_t destination0 = static_cast<uint32_t>(
                    indexLocal(chunkStart + row));
                const uint32_t destination1 = static_cast<uint32_t>(
                    indexLocal(chunkStart + row + 1U));
                const uint32_t destination2 = static_cast<uint32_t>(
                    indexLocal(chunkStart + row + 2U));
                const uint32_t destination3 = static_cast<uint32_t>(
                    indexLocal(chunkStart + row + 3U));
                const bool distinct = destination1 != destination0 &&
                    destination2 != destination0 && destination2 != destination1 &&
                    destination3 != destination0 && destination3 != destination1 &&
                    destination3 != destination2;
                if (distinct) {
                    Cast(outputPromoteLocal,
                         outputLocal[destination0 * alignedTile],
                         RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                    Cast(outputPromoteLocal[alignedTile],
                         outputLocal[destination1 * alignedTile],
                         RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                    Cast(outputPromoteLocal[2U * alignedTile],
                         outputLocal[destination2 * alignedTile],
                         RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                    Cast(outputPromoteLocal[3U * alignedTile],
                         outputLocal[destination3 * alignedTile],
                         RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                    Add(outputPromoteLocal, outputPromoteLocal,
                        sourceLocal[row * alignedTile],
                        static_cast<int32_t>(kBfloatPipelineDepth * alignedTile));
                    Cast(outputLocal[destination0 * alignedTile], outputPromoteLocal,
                         RoundMode::CAST_RINT, static_cast<int32_t>(tileLength));
                    Cast(outputLocal[destination1 * alignedTile],
                         outputPromoteLocal[alignedTile], RoundMode::CAST_RINT,
                         static_cast<int32_t>(tileLength));
                    Cast(outputLocal[destination2 * alignedTile],
                         outputPromoteLocal[2U * alignedTile], RoundMode::CAST_RINT,
                         static_cast<int32_t>(tileLength));
                    Cast(outputLocal[destination3 * alignedTile],
                         outputPromoteLocal[3U * alignedTile], RoundMode::CAST_RINT,
                         static_cast<int32_t>(tileLength));
                    continue;
                }
                for (uint32_t slot = 0U; slot < kBfloatPipelineDepth; ++slot) {
                    const uint32_t destination = static_cast<uint32_t>(
                        indexLocal(chunkStart + row + slot));
                    Cast(outputPromoteLocal, outputLocal[destination * alignedTile],
                         RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                    Add(outputPromoteLocal, outputPromoteLocal,
                        sourceLocal[(row + slot) * alignedTile],
                        static_cast<int32_t>(tileLength));
                    Cast(outputLocal[destination * alignedTile], outputPromoteLocal,
                         RoundMode::CAST_RINT, static_cast<int32_t>(tileLength));
                }
            }
            for (; row < rows; ++row) {
                const uint32_t destination =
                    static_cast<uint32_t>(indexLocal(chunkStart + row));
                Cast(outputPromoteLocal, outputLocal[destination * alignedTile],
                     RoundMode::CAST_NONE, static_cast<int32_t>(tileLength));
                Add(outputPromoteLocal, outputPromoteLocal,
                    sourceLocal[row * alignedTile], static_cast<int32_t>(tileLength));
                Cast(outputLocal[destination * alignedTile], outputPromoteLocal,
                     RoundMode::CAST_RINT, static_cast<int32_t>(tileLength));
            }
            if (nextChunkStart < indexSize_) {
                WaitMte2Vector();
                currentRawSlot = nextRawSlot;
            }
        }

        WaitVectorMte3();
        DataCopyPad(outputGm_[selfBase], outputLocal, outputCopy);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> outputBuf_;
    TBuf<QuePosition::VECCALC> sourceRawBuf_;
    TBuf<QuePosition::VECCALC> batchTransposedBuf_;
    TBuf<QuePosition::VECCALC> batchTransposeDstConfigBuf_;
    TBuf<QuePosition::VECCALC> batchTransposeSrcConfigBuf_;
    TBuf<QuePosition::VECCALC> sourcePromoteBuf_;
    TBuf<QuePosition::VECCALC> bfloatOutputPromoteBuf_;
    TBuf<QuePosition::VECCALC> batchOutputOffsetBuf_;
    TBuf<QuePosition::VECCALC> batchSourceOffsetBuf_;
    TBuf<QuePosition::VECCALC> batchAxisOffsetBuf_;
    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    uint64_t outerSize_;
    uint64_t innerSize_;
    uint32_t axisSize_;
    uint32_t indexSize_;
    uint32_t tilesPerOuter_;
    uint32_t tileElements_;
    uint32_t sourceChunk_;
    uint32_t outerBatch_;
    uint32_t batchAxisChunk_;
    uint32_t sourceTranspose_;
    uint32_t sourceBufferCount_;
    uint32_t atomicAdd_;
};
}  // namespace

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                                  GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelIndexAdd<DTYPE_SELF> kernel;
    kernel.Init(self, index, source, output, workspace, tilingData);
    kernel.Process();
}
