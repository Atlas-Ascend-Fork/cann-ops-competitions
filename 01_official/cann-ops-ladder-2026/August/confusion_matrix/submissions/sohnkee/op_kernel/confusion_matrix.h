/*!
 * \file confusion_matrix.h
 * \brief ConfusionMatrix kernel implementation
 */

#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "confusion_matrix_tiling_data.h"
#include "confusion_matrix_tiling_key.h"

namespace NsConfusionMatrix {

using namespace AscendC;

constexpr uint32_t CONFUSION_MATRIX_ZERO_TILE_ELEMS = 16384;
constexpr uint64_t CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS = 1024;
constexpr uint64_t CONFUSION_MATRIX_TINY_FASTPATH_MAX_SAMPLES = 64;
constexpr uint64_t CONFUSION_MATRIX_TINY_FASTPATH_MAX_OUTPUT = 256;
constexpr uint64_t CONFUSION_MATRIX_SMALL_ROW_SPLIT_MIN_SAMPLES = 512;
constexpr uint64_t CONFUSION_MATRIX_SMALL_ROW_SPLIT_MAX_CLASSES = 16;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS = 3072;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES =
    CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS * sizeof(int32_t) * 2;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES = CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS / 8;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_MASK_BUFFER_BYTES = CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES * 3;
constexpr uint32_t CONFUSION_MATRIX_C3_VECTOR_WORK_BUFFER_BYTES =
    CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES + CONFUSION_MATRIX_C3_VECTOR_MASK_BUFFER_BYTES;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_CAPACITY = 10240;
constexpr uint32_t CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX = 512;
constexpr uint32_t CONFUSION_MATRIX_OWNER_OFFSET_LOCAL_OFFSET = 0;
constexpr uint32_t CONFUSION_MATRIX_OWNER_ZERO_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY;
constexpr uint32_t CONFUSION_MATRIX_OWNER_ONE_LOCAL_OFFSET = CONFUSION_MATRIX_OWNER_ZERO_LOCAL_OFFSET + 8;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS = CONFUSION_MATRIX_SPARSE_CAPACITY;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_LABEL_LOCAL_OFFSET =
    CONFUSION_MATRIX_OWNER_ONE_LOCAL_OFFSET + 8;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_PREDICTION_LOCAL_OFFSET =
    CONFUSION_MATRIX_OWNER_FILTER_LABEL_LOCAL_OFFSET + CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_FLOAT_LOCAL_OFFSET =
    CONFUSION_MATRIX_OWNER_FILTER_PREDICTION_LOCAL_OFFSET + CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_MASK_BYTES = CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS / 8;
constexpr uint32_t CONFUSION_MATRIX_OWNER_FILTER_MASK_LOCAL_OFFSET_BYTES =
    (CONFUSION_MATRIX_OWNER_FILTER_FLOAT_LOCAL_OFFSET + CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS) *
    sizeof(int32_t);
constexpr uint32_t CONFUSION_MATRIX_OWNER_BUFFER_BYTES =
    CONFUSION_MATRIX_OWNER_FILTER_MASK_LOCAL_OFFSET_BYTES +
    CONFUSION_MATRIX_OWNER_FILTER_MASK_BYTES * 3;
constexpr uint32_t CONFUSION_MATRIX_OWNER_BUFFER_WORDS =
    (CONFUSION_MATRIX_OWNER_BUFFER_BYTES + sizeof(int32_t) - 1) / sizeof(int32_t);
constexpr uint32_t CONFUSION_MATRIX_SPARSE_COUNT_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_OCCUPIED_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY * 2;
constexpr uint32_t CONFUSION_MATRIX_BULK_BUFFER_BYTES =
    CONFUSION_MATRIX_SPARSE_CAPACITY * sizeof(int32_t) * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS = 5120;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET = CONFUSION_MATRIX_SPARSE_CAPACITY * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET =
    CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET =
    CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES = CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS / 8;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES =
    (CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET + CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS) *
    sizeof(int32_t);
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_STORAGE_BYTES =
    CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES + CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES * 3;
constexpr uint32_t CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES = 184U * 1024U;
constexpr uint64_t CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES = 160UL * 1024UL * 1024UL;
constexpr uint64_t CONFUSION_MATRIX_UINT32_OFFSET_MAX = 0xFFFFFFFFULL;

static_assert((CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES & 31U) == 0U,
    "sparse vector-filter mask storage must be 32-byte aligned");
static_assert(CONFUSION_MATRIX_SPARSE_FILTER_STORAGE_BYTES <= CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES,
    "sparse vector-filter storage exceeds the Vector Core UB budget");
static_assert((CONFUSION_MATRIX_OWNER_ZERO_LOCAL_OFFSET * sizeof(int32_t) & 31U) == 0U,
    "owner zero source must be 32-byte aligned");
static_assert((CONFUSION_MATRIX_OWNER_ONE_LOCAL_OFFSET * sizeof(int32_t) & 31U) == 0U,
    "owner one source must be 32-byte aligned");
static_assert((CONFUSION_MATRIX_OWNER_FILTER_MASK_LOCAL_OFFSET_BYTES & 31U) == 0U,
    "owner vector-filter mask storage must be 32-byte aligned");
static_assert(CONFUSION_MATRIX_OWNER_BUFFER_BYTES <= 192U * 1024U,
    "owner single-tile storage exceeds the Vector Core UB budget");

#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6

__aicore__ inline bool IsOrderedRowPrefix(
    const LocalTensor<int32_t>& values,
    int32_t rowStart,
    uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (values.GetValue(i) != rowStart + static_cast<int32_t>(i)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool HasUniqueOwnerOffsets(
    const LocalTensor<int32_t>& offsets,
    const LocalTensor<int32_t>& workspace,
    uint32_t count)
{
    if (count < 2U) {
        return true;
    }
    // A single 32-element Sort repeat does not expose a stable packed mask for
    // the shifted adjacency check on C220.  Conservatively keep those small
    // owner sets on the exact clear-plus-atomic path.
    if (count < 64U) {
        return false;
    }
    if (count > CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX) {
        return false;
    }

    const uint32_t sortCount = (count + 31U) & ~31U;
    LocalTensor<float> values = workspace.ReinterpretCast<float>();
    LocalTensor<uint32_t> indices =
        workspace[CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX].ReinterpretCast<uint32_t>();
    LocalTensor<float> sorted =
        workspace[CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX * 2U].ReinterpretCast<float>();
    LocalTensor<float> temporary =
        workspace[CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX * 4U].ReinterpretCast<float>();
    LocalTensor<uint8_t> duplicateMask =
        workspace[CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX * 6U].ReinterpretCast<uint8_t>();
    values.SetSize(sortCount);
    indices.SetSize(sortCount);
    sorted.SetSize(sortCount * 2U);
    temporary.SetSize(sortCount * 2U);
    duplicateMask.SetSize(CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX / 8U);

    Duplicate(values, -1.0f, sortCount);
    Cast(values, offsets, RoundMode::CAST_NONE, count);
    CreateVecIndex(indices.ReinterpretCast<int32_t>(), 0, sortCount);
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sorted, values, indices, temporary, sortCount / 32U);
    PipeBarrier<PIPE_V>();
    Extract(values, indices, sorted, sortCount / 32U);
    PipeBarrier<PIPE_V>();

    CreateVecIndex(indices.ReinterpretCast<int32_t>(), 1, sortCount);
    PipeBarrier<PIPE_V>();
    Muls(
        indices.ReinterpretCast<int32_t>(), indices.ReinterpretCast<int32_t>(),
        static_cast<int32_t>(sizeof(float)), sortCount);
    Duplicate(
        duplicateMask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
        CONFUSION_MATRIX_OWNER_UNIQUE_SORT_MAX / 16U);
    PipeBarrier<PIPE_V>();
    Gather(temporary, values, indices, 0, count - 1U);
    PipeBarrier<PIPE_V>();
    Compare(duplicateMask, values, temporary, CMPMODE::EQ, count - 1U);
    PipeBarrier<PIPE_V>();
    const TEventID uniqueReadyEvent = static_cast<TEventID>(2);
    SetFlag<HardEvent::V_S>(uniqueReadyEvent);
    WaitFlag<HardEvent::V_S>(uniqueReadyEvent);

    const uint32_t maskWordCount = (count - 1U + 31U) / 32U;
    LocalTensor<uint32_t> maskWords = duplicateMask.ReinterpretCast<uint32_t>();
    for (uint32_t i = 0; i < maskWordCount; ++i) {
        if (maskWords.GetValue(i) != 0U) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void PublishSparseValues(
    __gm__ int32_t* output,
    __ubuf__ int32_t* value,
    const LocalTensor<int32_t>& offsets,
    uint32_t count)
{
    const __ubuf__ uint64_t* packedOffsets =
        reinterpret_cast<const __ubuf__ uint64_t*>(offsets.GetPhyAddr());
    uint32_t i = 0;
#define CM_PUBLISH_DECODED_OFFSET(outOffset)                                               \
    copy_ubuf_to_gm_align_b32(                                                             \
        output + (outOffset), value, 0, 1, sizeof(int32_t), 0, 0, 0, 0)
    for (; i + 16U <= count; i += 16U) {
        const uint32_t pairBase = i >> 1U;
        const uint64_t packed0 = packedOffsets[pairBase];
        const uint64_t packed1 = packedOffsets[pairBase + 1U];
        const uint64_t packed2 = packedOffsets[pairBase + 2U];
        const uint64_t packed3 = packedOffsets[pairBase + 3U];
        const uint64_t packed4 = packedOffsets[pairBase + 4U];
        const uint64_t packed5 = packedOffsets[pairBase + 5U];
        const uint64_t packed6 = packedOffsets[pairBase + 6U];
        const uint64_t packed7 = packedOffsets[pairBase + 7U];
        const uint32_t outOffset0 = static_cast<uint32_t>(packed0);
        const uint32_t outOffset1 = static_cast<uint32_t>(packed0 >> 32U);
        const uint32_t outOffset2 = static_cast<uint32_t>(packed1);
        const uint32_t outOffset3 = static_cast<uint32_t>(packed1 >> 32U);
        const uint32_t outOffset4 = static_cast<uint32_t>(packed2);
        const uint32_t outOffset5 = static_cast<uint32_t>(packed2 >> 32U);
        const uint32_t outOffset6 = static_cast<uint32_t>(packed3);
        const uint32_t outOffset7 = static_cast<uint32_t>(packed3 >> 32U);
        const uint32_t outOffset8 = static_cast<uint32_t>(packed4);
        const uint32_t outOffset9 = static_cast<uint32_t>(packed4 >> 32U);
        const uint32_t outOffset10 = static_cast<uint32_t>(packed5);
        const uint32_t outOffset11 = static_cast<uint32_t>(packed5 >> 32U);
        const uint32_t outOffset12 = static_cast<uint32_t>(packed6);
        const uint32_t outOffset13 = static_cast<uint32_t>(packed6 >> 32U);
        const uint32_t outOffset14 = static_cast<uint32_t>(packed7);
        const uint32_t outOffset15 = static_cast<uint32_t>(packed7 >> 32U);
        CM_PUBLISH_DECODED_OFFSET(outOffset0);
        CM_PUBLISH_DECODED_OFFSET(outOffset1);
        CM_PUBLISH_DECODED_OFFSET(outOffset2);
        CM_PUBLISH_DECODED_OFFSET(outOffset3);
        CM_PUBLISH_DECODED_OFFSET(outOffset4);
        CM_PUBLISH_DECODED_OFFSET(outOffset5);
        CM_PUBLISH_DECODED_OFFSET(outOffset6);
        CM_PUBLISH_DECODED_OFFSET(outOffset7);
        CM_PUBLISH_DECODED_OFFSET(outOffset8);
        CM_PUBLISH_DECODED_OFFSET(outOffset9);
        CM_PUBLISH_DECODED_OFFSET(outOffset10);
        CM_PUBLISH_DECODED_OFFSET(outOffset11);
        CM_PUBLISH_DECODED_OFFSET(outOffset12);
        CM_PUBLISH_DECODED_OFFSET(outOffset13);
        CM_PUBLISH_DECODED_OFFSET(outOffset14);
        CM_PUBLISH_DECODED_OFFSET(outOffset15);
    }
#undef CM_PUBLISH_DECODED_OFFSET
    for (; i + 2U <= count; i += 2U) {
        const uint64_t packed = packedOffsets[i >> 1U];
        const uint32_t outOffset0 = static_cast<uint32_t>(packed);
        const uint32_t outOffset1 = static_cast<uint32_t>(packed >> 32U);
        copy_ubuf_to_gm_align_b32(
            output + outOffset0, value, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
        copy_ubuf_to_gm_align_b32(
            output + outOffset1, value, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
    }
    for (; i < count; ++i) {
        const uint32_t outOffset = static_cast<uint32_t>(offsets.GetValue(i));
        copy_ubuf_to_gm_align_b32(
            output + outOffset, value, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
    }
}

// Key 6 is compiled as a self-contained sparse-atomic kernel.  Its Host gate
// fixes the dtype/weights contract, so this translation unit keeps only the
// state consumed by the large-output path.
class ConfusionMatrix {
public:
    __aicore__ inline ConfusionMatrix() {};

    __aicore__ inline void Init(
        GM_ADDR labels,
        GM_ADDR predictions,
        GM_ADDR weights,
        GM_ADDR y,
        const ConfusionMatrixTilingData* tilingData);
    __aicore__ inline void InitAtomicSparse(
        GM_ADDR labels,
        GM_ADDR predictions,
        GM_ADDR y,
        int64_t totalNum,
        int64_t numClasses);
    __aicore__ inline void ProcessAtomicSparse(uint64_t samplesPerCore);

private:
    __gm__ int32_t* labelsI32Gm_ = nullptr;
    __gm__ int32_t* predictionsI32Gm_ = nullptr;
    GlobalTensor<int32_t> labelsI32Tensor_;
    GlobalTensor<int32_t> predictionsI32Tensor_;
    GlobalTensor<int32_t> outputI32Gm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ownerBuf_;
    int64_t totalNum_ = 0;
    int64_t numClasses_ = 0;
};

__aicore__ inline void ConfusionMatrix::Init(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    labelsI32Gm_ = reinterpret_cast<__gm__ int32_t*>(labels);
    predictionsI32Gm_ = reinterpret_cast<__gm__ int32_t*>(predictions);
    labelsI32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsI32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    outputI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI32Gm_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    pipe_.InitBuffer(ownerBuf_, CONFUSION_MATRIX_OWNER_BUFFER_BYTES);
    totalNum_ = tilingData->totalNum;
    numClasses_ = tilingData->numClasses;
}

__aicore__ inline void ConfusionMatrix::InitAtomicSparse(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    int64_t totalNum,
    int64_t numClasses)
{
    labelsI32Gm_ = reinterpret_cast<__gm__ int32_t*>(labels);
    predictionsI32Gm_ = reinterpret_cast<__gm__ int32_t*>(predictions);
    labelsI32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsI32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    outputI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI32Gm_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    pipe_.InitBuffer(ownerBuf_, CONFUSION_MATRIX_OWNER_BUFFER_BYTES);
    totalNum_ = totalNum;
    numClasses_ = numClasses;
}

__aicore__ inline void ConfusionMatrix::ProcessAtomicSparse(uint64_t samplesPerCore)
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint32_t classCount = static_cast<uint32_t>(numClasses_);
    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    const uint64_t logicalBlockNum =
        static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetSubBlockNum());
    const uint64_t rowsPerWorker =
        (static_cast<uint64_t>(classCount) + logicalBlockNum - 1) / logicalBlockNum;
    const uint64_t activeWorkerCount =
        (static_cast<uint64_t>(classCount) + rowsPerWorker - 1) / rowsPerWorker;
    const uint64_t rowStart = blockIdx * rowsPerWorker;
    if (rowStart >= static_cast<uint64_t>(classCount)) {
        return;
    }
    uint64_t rowEnd = rowStart + rowsPerWorker;
    if (rowEnd > static_cast<uint64_t>(classCount)) {
        rowEnd = static_cast<uint64_t>(classCount);
    }
    const uint32_t rowStartU32 = static_cast<uint32_t>(rowStart);
    const uint32_t rowEndU32 = static_cast<uint32_t>(rowEnd);
    const int32_t classCountI32 = static_cast<int32_t>(classCount);
    const int32_t rowStartI32 = static_cast<int32_t>(rowStartU32);
    const int32_t rowEndI32 = static_cast<int32_t>(rowEndU32);

    LocalTensor<int32_t> ownerStorage = ownerBuf_.Get<int32_t>();
    ownerStorage.SetSize(CONFUSION_MATRIX_OWNER_BUFFER_WORDS);
    LocalTensor<int32_t> offsetLocal = ownerStorage[CONFUSION_MATRIX_OWNER_OFFSET_LOCAL_OFFSET];
    LocalTensor<int32_t> zeroLocal = ownerStorage[CONFUSION_MATRIX_OWNER_ZERO_LOCAL_OFFSET];
    LocalTensor<int32_t> oneLocal = ownerStorage[CONFUSION_MATRIX_OWNER_ONE_LOCAL_OFFSET];
    zeroLocal.SetValue(0, static_cast<int32_t>(0));
    oneLocal.SetValue(0, static_cast<int32_t>(1));
    const TEventID valuesReadyEvent = static_cast<TEventID>(0);
    SetFlag<HardEvent::S_MTE3>(valuesReadyEvent);
    LocalTensor<int32_t> filterLabelsLocal =
        ownerStorage[CONFUSION_MATRIX_OWNER_FILTER_LABEL_LOCAL_OFFSET];
    LocalTensor<int32_t> filterPredictionsLocal =
        ownerStorage[CONFUSION_MATRIX_OWNER_FILTER_PREDICTION_LOCAL_OFFSET];
    LocalTensor<int32_t> rangeClampLocal =
        ownerStorage[CONFUSION_MATRIX_OWNER_FILTER_FLOAT_LOCAL_OFFSET];
    LocalTensor<uint8_t> filterMaskStorage =
        ownerStorage.ReinterpretCast<uint8_t>()[CONFUSION_MATRIX_OWNER_FILTER_MASK_LOCAL_OFFSET_BYTES];
    LocalTensor<uint8_t> ownedMask = filterMaskStorage[CONFUSION_MATRIX_OWNER_FILTER_MASK_BYTES * 2];

    // When the complete runtime label tensor is a consecutive unique sequence,
    // every valid sample owns a distinct output address.  Each active worker
    // proves that property independently, so the input can then be sharded
    // without a cross-core rendezvous.  Any failed precheck or proof falls
    // through to the generalized row-owner implementation below.
    bool labelsAreConsecutive = totalNum != 0U &&
        totalNum <= CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS && classCount <= 46340U;
    int32_t firstLabel = 0;
    if (labelsAreConsecutive) {
        firstLabel = labelsI32Tensor_.GetValue(0);
        const int32_t lastLabel = labelsI32Tensor_.GetValue(totalNum - 1U);
        const int64_t expectedLast =
            static_cast<int64_t>(firstLabel) + static_cast<int64_t>(totalNum) - 1;
        labelsAreConsecutive = firstLabel >= 0 &&
            expectedLast < static_cast<int64_t>(classCount) &&
            static_cast<int64_t>(lastLabel) == expectedLast;
    }
    if (labelsAreConsecutive) {
        const uint32_t proofCount = static_cast<uint32_t>(totalNum);
        const uint32_t alignedProofCount = (proofCount + 7U) & ~7U;
        const uint32_t proofVectorCount = (proofCount + 63U) & ~63U;
        if ((proofCount & 7U) == 0U) {
            DataCopy(filterLabelsLocal, labelsI32Tensor_, proofCount);
        } else {
            DataCopyParams proofCopyParams{
                1, static_cast<uint16_t>(proofCount * sizeof(int32_t)), 0, 0};
            DataCopyPadParams proofPadParams{true, 0, 0, 0};
            DataCopyPad(filterLabelsLocal, labelsI32Tensor_, proofCopyParams, proofPadParams);
        }
        CreateVecIndex(filterPredictionsLocal, firstLabel, proofVectorCount);
        const TEventID proofCopyEvent = static_cast<TEventID>(3);
        SetFlag<HardEvent::MTE2_V>(proofCopyEvent);
        WaitFlag<HardEvent::MTE2_V>(proofCopyEvent);
        if (alignedProofCount > proofCount) {
            for (uint32_t i = proofCount; i < alignedProofCount; ++i) {
                filterLabelsLocal.SetValue(i, static_cast<int32_t>(-1));
            }
            const TEventID proofTailReadyEvent = static_cast<TEventID>(3);
            SetFlag<HardEvent::S_V>(proofTailReadyEvent);
            WaitFlag<HardEvent::S_V>(proofTailReadyEvent);
        }
        if (proofVectorCount > alignedProofCount) {
            Duplicate(
                filterLabelsLocal[alignedProofCount], static_cast<int32_t>(-1),
                proofVectorCount - alignedProofCount);
            PipeBarrier<PIPE_V>();
        }
        Compare(
            ownedMask, filterLabelsLocal, filterPredictionsLocal,
            CMPMODE::EQ, proofVectorCount);
        PipeBarrier<PIPE_V>();
        const uint32_t fullProofMaskBytes = proofCount >> 3U;
        LocalTensor<half> proofMaskHalf = rangeClampLocal.ReinterpretCast<half>();
        LocalTensor<half> proofReduceResult = filterPredictionsLocal.ReinterpretCast<half>();
        LocalTensor<half> proofReduceTmp = filterLabelsLocal.ReinterpretCast<half>();
        if (fullProofMaskBytes != 0U) {
            Cast(
                proofMaskHalf, ownedMask,
                RoundMode::CAST_NONE, fullProofMaskBytes);
            PipeBarrier<PIPE_V>();
            ReduceMin(
                proofReduceResult, proofMaskHalf, proofReduceTmp,
                static_cast<int32_t>(fullProofMaskBytes), false);
            PipeBarrier<PIPE_V>();
        }
        const TEventID proofReadyEvent = static_cast<TEventID>(3);
        SetFlag<HardEvent::V_S>(proofReadyEvent);
        WaitFlag<HardEvent::V_S>(proofReadyEvent);
        if (fullProofMaskBytes != 0U) {
            LocalTensor<uint16_t> proofReduceBits =
                proofReduceResult.ReinterpretCast<uint16_t>();
            labelsAreConsecutive =
                proofReduceBits.GetValue(0) == static_cast<uint16_t>(0x5BF8U);
        }
        const uint32_t proofTailBits = proofCount & 7U;
        if (labelsAreConsecutive && proofTailBits != 0U) {
            const uint8_t expectedTailMask =
                static_cast<uint8_t>((1U << proofTailBits) - 1U);
            labelsAreConsecutive =
                ownedMask.GetValue(fullProofMaskBytes) == expectedTailMask;
        }
    }

    if (labelsAreConsecutive) {
        const uint64_t samplesPerActiveWorker =
            (totalNum + activeWorkerCount - 1U) / activeWorkerCount;
        const uint64_t sampleStart = blockIdx * samplesPerActiveWorker;
        uint64_t sampleEnd = sampleStart + samplesPerActiveWorker;
        if (sampleEnd > totalNum) {
            sampleEnd = totalNum;
        }
        const uint32_t shardCount = sampleStart < totalNum ?
            static_cast<uint32_t>(sampleEnd - sampleStart) : 0U;
        uint32_t validCountU32 = 0U;
        if (shardCount != 0U) {
            const uint32_t alignedShardCount = (shardCount + 7U) & ~7U;
            const uint32_t shardVectorCount = (shardCount + 63U) & ~63U;
            if ((shardCount & 7U) == 0U) {
                DataCopy(
                    filterLabelsLocal, predictionsI32Tensor_[sampleStart], shardCount);
            } else {
                DataCopyParams shardCopyParams{
                    1, static_cast<uint16_t>(shardCount * sizeof(int32_t)), 0, 0};
                DataCopyPadParams shardPadParams{
                    true, 0, 0, static_cast<uint64_t>(classCount)};
                DataCopyPad(
                    filterLabelsLocal, predictionsI32Tensor_[sampleStart],
                    shardCopyParams, shardPadParams);
            }
            CreateVecIndex(
                offsetLocal,
                firstLabel + static_cast<int32_t>(sampleStart),
                shardVectorCount);
            const TEventID shardCopyEvent = static_cast<TEventID>(3);
            SetFlag<HardEvent::MTE2_V>(shardCopyEvent);
            WaitFlag<HardEvent::MTE2_V>(shardCopyEvent);
            if (alignedShardCount > shardCount) {
                for (uint32_t i = shardCount; i < alignedShardCount; ++i) {
                    filterLabelsLocal.SetValue(i, classCountI32);
                }
                const TEventID shardTailReadyEvent = static_cast<TEventID>(3);
                SetFlag<HardEvent::S_V>(shardTailReadyEvent);
                WaitFlag<HardEvent::S_V>(shardTailReadyEvent);
            }
            if (shardVectorCount > alignedShardCount) {
                Duplicate(
                    filterLabelsLocal[alignedShardCount], classCountI32,
                    shardVectorCount - alignedShardCount);
                PipeBarrier<PIPE_V>();
            }

            Maxs(rangeClampLocal, filterLabelsLocal, static_cast<int32_t>(0), shardVectorCount);
            PipeBarrier<PIPE_V>();
            Mins(rangeClampLocal, rangeClampLocal, classCountI32 - 1, shardVectorCount);
            PipeBarrier<PIPE_V>();
            Compare(
                ownedMask, rangeClampLocal, filterLabelsLocal,
                CMPMODE::EQ, shardVectorCount);
            PipeBarrier<PIPE_V>();

            uint64_t validLabelCount = 0U;
            GatherMask(
                filterPredictionsLocal, offsetLocal,
                ownedMask.ReinterpretCast<uint32_t>(), true,
                shardVectorCount, {1, 1, 8, 0}, validLabelCount);
            PipeBarrier<PIPE_V>();
            uint64_t validCount = 0U;
            GatherMask(
                offsetLocal, filterLabelsLocal,
                ownedMask.ReinterpretCast<uint32_t>(), true,
                shardVectorCount, {1, 1, 8, 0}, validCount);
            PipeBarrier<PIPE_V>();
            const TEventID shardReadyEvent = static_cast<TEventID>(3);
            SetFlag<HardEvent::V_S>(shardReadyEvent);
            WaitFlag<HardEvent::V_S>(shardReadyEvent);
            validCountU32 = static_cast<uint32_t>(validCount);
            if (validCountU32 != 0U) {
                Muls(
                    filterPredictionsLocal, filterPredictionsLocal,
                    classCountI32, static_cast<int32_t>(validCountU32));
                PipeBarrier<PIPE_V>();
                Add(
                    offsetLocal, filterPredictionsLocal, offsetLocal,
                    static_cast<int32_t>(validCountU32));
                PipeBarrier<PIPE_V>();
                const TEventID offsetsReadyEvent = static_cast<TEventID>(3);
                SetFlag<HardEvent::V_S>(offsetsReadyEvent);
                WaitFlag<HardEvent::V_S>(offsetsReadyEvent);
            }
        }

        WaitFlag<HardEvent::S_MTE3>(valuesReadyEvent);
        if (validCountU32 != 0U) {
            __gm__ int32_t* outputPtr = outputI32Gm_.GetPhyAddr(0);
            __ubuf__ int32_t* onePtr =
                reinterpret_cast<__ubuf__ int32_t*>(oneLocal.GetPhyAddr());
            PublishSparseValues(outputPtr, onePtr, offsetLocal, validCountU32);
            const TEventID writeCompleteEvent = static_cast<TEventID>(1);
            SetFlag<HardEvent::MTE3_S>(writeCompleteEvent);
            WaitFlag<HardEvent::MTE3_S>(writeCompleteEvent);
        }
        return;
    }

    uint32_t ownedCount = 0;
    bool ownedLabelsAreOrdered = true;

    const TEventID labelsCopyEvent = static_cast<TEventID>(0);
    const TEventID maskReadyEvent = static_cast<TEventID>(1);
    uint64_t sampleOffset = 0;
    while (sampleOffset < totalNum) {
        const uint64_t remaining = totalNum - sampleOffset;
        const uint32_t copyCount = remaining > CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS ?
            CONFUSION_MATRIX_OWNER_FILTER_TILE_ELEMS : static_cast<uint32_t>(remaining);
        const uint32_t alignedCopyCount = (copyCount + 7U) & ~7U;
        const uint32_t vectorCount = (copyCount + 63U) & ~63U;
        if ((copyCount & 7U) == 0U) {
            DataCopy(filterLabelsLocal, labelsI32Tensor_[sampleOffset], copyCount);
            DataCopy(filterPredictionsLocal, predictionsI32Tensor_[sampleOffset], copyCount);
        } else {
            DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
            DataCopyPadParams labelPadParams{true, 0, 0, 0xFFFFFFFFULL};
            DataCopyPadParams predictionPadParams{true, 0, 0, 0};
            DataCopyPad(filterLabelsLocal, labelsI32Tensor_[sampleOffset], inputParams, labelPadParams);
            DataCopyPad(
                filterPredictionsLocal, predictionsI32Tensor_[sampleOffset], inputParams,
                predictionPadParams);
        }
        SetFlag<HardEvent::MTE2_V>(labelsCopyEvent);
        WaitFlag<HardEvent::MTE2_V>(labelsCopyEvent);
        if (alignedCopyCount > copyCount) {
            for (uint32_t i = copyCount; i < alignedCopyCount; ++i) {
                filterLabelsLocal.SetValue(i, static_cast<int32_t>(-1));
                filterPredictionsLocal.SetValue(i, static_cast<int32_t>(0));
            }
            const TEventID tailReadyEvent = static_cast<TEventID>(2);
            SetFlag<HardEvent::S_V>(tailReadyEvent);
            WaitFlag<HardEvent::S_V>(tailReadyEvent);
        }
        if (vectorCount > alignedCopyCount) {
            Duplicate(
                filterLabelsLocal[alignedCopyCount], static_cast<int32_t>(-1),
                vectorCount - alignedCopyCount);
            Duplicate(
                filterPredictionsLocal[alignedCopyCount], static_cast<int32_t>(0),
                vectorCount - alignedCopyCount);
            PipeBarrier<PIPE_V>();
        }

        Maxs(rangeClampLocal, filterLabelsLocal, rowStartI32, vectorCount);
        PipeBarrier<PIPE_V>();
        Mins(rangeClampLocal, rangeClampLocal, rowEndI32 - 1, vectorCount);
        PipeBarrier<PIPE_V>();
        Compare(ownedMask, rangeClampLocal, filterLabelsLocal, CMPMODE::EQ, vectorCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(maskReadyEvent);
        WaitFlag<HardEvent::V_S>(maskReadyEvent);

        uint64_t selectedLabelCount = 0;
        GatherMask(
            offsetLocal, filterLabelsLocal, ownedMask.ReinterpretCast<uint32_t>(), true,
            vectorCount, {1, 1, 8, 0}, selectedLabelCount);
        PipeBarrier<PIPE_V>();
        uint64_t selectedCount = 0;
        GatherMask(
            filterLabelsLocal, filterPredictionsLocal, ownedMask.ReinterpretCast<uint32_t>(), true,
            vectorCount, {1, 1, 8, 0}, selectedCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(maskReadyEvent);
        WaitFlag<HardEvent::V_S>(maskReadyEvent);

        const uint32_t selectedCountU32 = static_cast<uint32_t>(selectedCount);
        if (ownedLabelsAreOrdered) {
            ownedLabelsAreOrdered = classCount <= 46340U &&
                IsOrderedRowPrefix(offsetLocal, rowStartI32, selectedCountU32);
        }
        if (selectedCountU32 != 0U && classCount <= 46340U) {
            const uint32_t alignedSelectedCount = (selectedCountU32 + 7U) & ~7U;
            const uint32_t selectedVectorCount = (selectedCountU32 + 63U) & ~63U;
            if (alignedSelectedCount > selectedCountU32) {
                for (uint32_t i = selectedCountU32; i < alignedSelectedCount; ++i) {
                    offsetLocal.SetValue(i, static_cast<int32_t>(0));
                    filterLabelsLocal.SetValue(i, static_cast<int32_t>(classCount));
                }
                const TEventID selectedTailReadyEvent = static_cast<TEventID>(2);
                SetFlag<HardEvent::S_V>(selectedTailReadyEvent);
                WaitFlag<HardEvent::S_V>(selectedTailReadyEvent);
            }
            if (selectedVectorCount > alignedSelectedCount) {
                Duplicate(
                    offsetLocal[alignedSelectedCount], static_cast<int32_t>(0),
                    selectedVectorCount - alignedSelectedCount);
                Duplicate(
                    filterLabelsLocal[alignedSelectedCount], static_cast<int32_t>(classCount),
                    selectedVectorCount - alignedSelectedCount);
                PipeBarrier<PIPE_V>();
            }

            Maxs(
                rangeClampLocal, filterLabelsLocal, static_cast<int32_t>(0),
                selectedVectorCount);
            PipeBarrier<PIPE_V>();
            Mins(
                rangeClampLocal, rangeClampLocal, classCountI32 - 1,
                selectedVectorCount);
            PipeBarrier<PIPE_V>();
            Compare(
                ownedMask, rangeClampLocal, filterLabelsLocal, CMPMODE::EQ,
                selectedVectorCount);
            PipeBarrier<PIPE_V>();

            uint64_t validLabelCount = 0;
            GatherMask(
                filterPredictionsLocal, offsetLocal, ownedMask.ReinterpretCast<uint32_t>(), true,
                selectedVectorCount, {1, 1, 8, 0}, validLabelCount);
            PipeBarrier<PIPE_V>();
            uint64_t validCount = 0;
            GatherMask(
                offsetLocal, filterLabelsLocal, ownedMask.ReinterpretCast<uint32_t>(), true,
                selectedVectorCount, {1, 1, 8, 0}, validCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(maskReadyEvent);
            WaitFlag<HardEvent::V_S>(maskReadyEvent);
            ownedCount = static_cast<uint32_t>(validCount);
            if (ownedCount != 0U) {
                Muls(
                    filterPredictionsLocal, filterPredictionsLocal, static_cast<int32_t>(classCount),
                    static_cast<int32_t>(ownedCount));
                PipeBarrier<PIPE_V>();
                Add(offsetLocal, filterPredictionsLocal, offsetLocal, static_cast<int32_t>(ownedCount));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(maskReadyEvent);
                WaitFlag<HardEvent::V_S>(maskReadyEvent);
            }
        } else {
            for (uint32_t i = 0; i < selectedCountU32; ++i) {
                const uint32_t labelValue = static_cast<uint32_t>(offsetLocal.GetValue(i));
                const uint32_t predictionValue =
                    static_cast<uint32_t>(filterLabelsLocal.GetValue(i));
                if (predictionValue < classCount) {
                    offsetLocal.SetValue(
                        ownedCount, static_cast<int32_t>(labelValue * classCount + predictionValue));
                    ++ownedCount;
                }
            }
        }
        sampleOffset += copyCount;
    }

    const bool ownedOffsetsAreUnique = ownedLabelsAreOrdered ||
        HasUniqueOwnerOffsets(offsetLocal, filterLabelsLocal, ownedCount);
    __gm__ int32_t* outputPtr = outputI32Gm_.GetPhyAddr(0);
    __ubuf__ int32_t* zeroPtr = reinterpret_cast<__ubuf__ int32_t*>(zeroLocal.GetPhyAddr());
    __ubuf__ int32_t* onePtr = reinterpret_cast<__ubuf__ int32_t*>(oneLocal.GetPhyAddr());
    WaitFlag<HardEvent::S_MTE3>(valuesReadyEvent);
    if (ownedOffsetsAreUnique) {
        PublishSparseValues(outputPtr, onePtr, offsetLocal, ownedCount);
        const TEventID writeCompleteEvent = static_cast<TEventID>(1);
        SetFlag<HardEvent::MTE3_S>(writeCompleteEvent);
        WaitFlag<HardEvent::MTE3_S>(writeCompleteEvent);
        return;
    }
    PublishSparseValues(outputPtr, zeroPtr, offsetLocal, ownedCount);
    const TEventID clearCompleteEvent = static_cast<TEventID>(1);
    SetFlag<HardEvent::MTE3_S>(clearCompleteEvent);
    WaitFlag<HardEvent::MTE3_S>(clearCompleteEvent);

    SetAtomicAdd<int32_t>();
    PublishSparseValues(outputPtr, onePtr, offsetLocal, ownedCount);
    const TEventID addCompleteEvent = static_cast<TEventID>(1);
    SetFlag<HardEvent::MTE3_S>(addCompleteEvent);
    WaitFlag<HardEvent::MTE3_S>(addCompleteEvent);
    SetAtomicNone();
}

#else

__aicore__ inline void ProcessTinyInt32NoWeight(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    uint32_t totalNum,
    uint32_t numClasses)
{
    const uint32_t outputNum = numClasses * numClasses;
    const __gm__ int32_t* __restrict labelsPtr = reinterpret_cast<const __gm__ int32_t*>(labels);
    const __gm__ int32_t* __restrict predictionsPtr = reinterpret_cast<const __gm__ int32_t*>(predictions);
    __gm__ int32_t* __restrict outputPtr = reinterpret_cast<__gm__ int32_t*>(y);
    __ubuf__ int32_t outputLocal[32];

    if (outputNum <= 9) {
        GlobalTensor<int32_t> labelsGm;
        GlobalTensor<int32_t> predictionsGm;
        GlobalTensor<int32_t> outputGm;
        labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
        predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
        if (outputNum == 9) {
            outputGm.SetValue(0, static_cast<int32_t>(0));
            outputGm.SetValue(1, static_cast<int32_t>(0));
            outputGm.SetValue(2, static_cast<int32_t>(0));
            outputGm.SetValue(3, static_cast<int32_t>(0));
            outputGm.SetValue(4, static_cast<int32_t>(0));
            outputGm.SetValue(5, static_cast<int32_t>(0));
            outputGm.SetValue(6, static_cast<int32_t>(0));
            outputGm.SetValue(7, static_cast<int32_t>(0));
            outputGm.SetValue(8, static_cast<int32_t>(0));
        } else {
            for (uint32_t i = 0; i < outputNum; ++i) {
                outputGm.SetValue(i, static_cast<int32_t>(0));
            }
        }
#define CM_TINY_DIRECT_GM_STEP_UNCHECKED(IDX) \
        do { \
            const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(IDX)); \
            const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(IDX)); \
            const uint32_t outOffset = label * numClasses + prediction; \
            outputGm.SetValue(outOffset, outputGm.GetValue(outOffset) + 1); \
        } while (false)
#define CM_TINY_DIRECT_GM_STEP_GUARDED(IDX) \
        if (totalNum > IDX) { \
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(IDX); \
        }
        if (totalNum == 4) {
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(0);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(1);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(2);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(3);
        } else if (totalNum == 5) {
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(0);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(1);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(2);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(3);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(4);
        } else if (totalNum == 6) {
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(0);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(1);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(2);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(3);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(4);
            CM_TINY_DIRECT_GM_STEP_UNCHECKED(5);
        } else {
            CM_TINY_DIRECT_GM_STEP_GUARDED(0);
            CM_TINY_DIRECT_GM_STEP_GUARDED(1);
            CM_TINY_DIRECT_GM_STEP_GUARDED(2);
            CM_TINY_DIRECT_GM_STEP_GUARDED(3);
            CM_TINY_DIRECT_GM_STEP_GUARDED(4);
            CM_TINY_DIRECT_GM_STEP_GUARDED(5);
        }
#undef CM_TINY_DIRECT_GM_STEP_GUARDED
#undef CM_TINY_DIRECT_GM_STEP_UNCHECKED
        return;
    }

    __ubuf__ uint64_t* outputPairs = reinterpret_cast<__ubuf__ uint64_t*>(outputLocal);
    const uint32_t outputPairCount = outputNum >> 1U;
    for (uint32_t i = 0; i < outputPairCount; ++i) {
        outputPairs[i] = 0;
    }
    if ((outputNum & 1U) != 0) {
        outputLocal[outputNum - 1U] = static_cast<int32_t>(0);
    }

#define CM_TINY_UB_HISTOGRAM_STEP(IDX) \
    if (totalNum > IDX) { \
        const uint32_t label = static_cast<uint32_t>(labelsPtr[IDX]); \
        const uint32_t prediction = static_cast<uint32_t>(predictionsPtr[IDX]); \
        const uint32_t outOffset = label * numClasses + prediction; \
        outputLocal[outOffset] += 1; \
    }
    CM_TINY_UB_HISTOGRAM_STEP(0);
    CM_TINY_UB_HISTOGRAM_STEP(1);
    CM_TINY_UB_HISTOGRAM_STEP(2);
    CM_TINY_UB_HISTOGRAM_STEP(3);
    CM_TINY_UB_HISTOGRAM_STEP(4);
    CM_TINY_UB_HISTOGRAM_STEP(5);
#undef CM_TINY_UB_HISTOGRAM_STEP
    const TEventID outputReadyEvent = static_cast<TEventID>(0);
    SetFlag<HardEvent::S_MTE3>(outputReadyEvent);
    WaitFlag<HardEvent::S_MTE3>(outputReadyEvent);
    copy_ubuf_to_gm_align_b32(
        outputPtr, outputLocal, 0, 1, outputNum * sizeof(int32_t), 0, 0, 0, 0);
}

__aicore__ inline void ProcessTinyInt32WeightInt32(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int32_t> labelsGm;
    GlobalTensor<int32_t> predictionsGm;
    GlobalTensor<int32_t> weightsGm;
    GlobalTensor<int32_t> outputGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(weights));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    const uint32_t totalNum = static_cast<uint32_t>(tilingData->totalNum);
    const uint32_t numClasses = static_cast<uint32_t>(tilingData->numClasses);
    const uint32_t outputNum = numClasses * numClasses;

    for (uint32_t i = 0; i < outputNum; ++i) {
        outputGm.SetValue(i, static_cast<int32_t>(0));
    }

    const __gm__ uint64_t* labelsPairs = reinterpret_cast<const __gm__ uint64_t*>(labels);
    const __gm__ uint64_t* predictionsPairs = reinterpret_cast<const __gm__ uint64_t*>(predictions);
    const __gm__ uint64_t* weightsPairs = reinterpret_cast<const __gm__ uint64_t*>(weights);
#define CM_TINY_I32_WEIGHT_PAIR(PAIR_IDX) \
    do { \
        const uint64_t labelPair = labelsPairs[PAIR_IDX]; \
        const uint64_t predictionPair = predictionsPairs[PAIR_IDX]; \
        const uint64_t weightPair = weightsPairs[PAIR_IDX]; \
        const uint32_t label0 = static_cast<uint32_t>(labelPair); \
        const uint32_t prediction0 = static_cast<uint32_t>(predictionPair); \
        const uint32_t offset0 = label0 * numClasses + prediction0; \
        outputGm.SetValue(offset0, outputGm.GetValue(offset0) + static_cast<int32_t>(weightPair)); \
        const uint32_t label1 = static_cast<uint32_t>(labelPair >> 32); \
        const uint32_t prediction1 = static_cast<uint32_t>(predictionPair >> 32); \
        const uint32_t offset1 = label1 * numClasses + prediction1; \
        outputGm.SetValue(offset1, outputGm.GetValue(offset1) + static_cast<int32_t>(weightPair >> 32)); \
    } while (false)
    if (totalNum >= 2) {
        CM_TINY_I32_WEIGHT_PAIR(0);
    }
    if (totalNum >= 4) {
        CM_TINY_I32_WEIGHT_PAIR(1);
    }
    if (totalNum >= 6) {
        CM_TINY_I32_WEIGHT_PAIR(2);
    }
#undef CM_TINY_I32_WEIGHT_PAIR
    if ((totalNum & 1U) != 0) {
        const uint32_t index = totalNum - 1U;
        const uint32_t label = static_cast<uint32_t>(labelsGm.GetValue(index));
        const uint32_t prediction = static_cast<uint32_t>(predictionsGm.GetValue(index));
        const uint32_t outOffset = label * numClasses + prediction;
        outputGm.SetValue(outOffset, outputGm.GetValue(outOffset) + weightsGm.GetValue(index));
    }
}

__aicore__ inline void ProcessTinyInt64WeightFloat(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<int64_t> predictionsGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<float> outputGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(predictions));
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);
    const uint64_t numClasses = static_cast<uint64_t>(tilingData->numClasses);
    const uint64_t outputNum = numClasses * numClasses;

    if (outputNum == 4) {
        __gm__ uint64_t* outputPairs = reinterpret_cast<__gm__ uint64_t*>(y);
        outputPairs[0] = 0;
        outputPairs[1] = 0;
    } else {
        for (uint64_t i = 0; i < outputNum; ++i) {
            outputGm.SetValue(i, 0.0f);
        }
    }

#define CM_TINY_I64_FLOAT_STEP(IDX) \
    if (totalNum > IDX) { \
        const uint64_t label = static_cast<uint64_t>(labelsGm.GetValue(IDX)); \
        const uint64_t prediction = static_cast<uint64_t>(predictionsGm.GetValue(IDX)); \
        const uint64_t outOffset = label * numClasses + prediction; \
        const float oldValue = outputGm.GetValue(outOffset); \
        outputGm.SetValue(outOffset, oldValue + weightsGm.GetValue(IDX)); \
    }
    CM_TINY_I64_FLOAT_STEP(0);
    CM_TINY_I64_FLOAT_STEP(1);
    CM_TINY_I64_FLOAT_STEP(2);
    CM_TINY_I64_FLOAT_STEP(3);
    CM_TINY_I64_FLOAT_STEP(4);
    CM_TINY_I64_FLOAT_STEP(5);
#undef CM_TINY_I64_FLOAT_STEP
}

template <bool useMte2TailPadding>
__aicore__ inline void ProcessC3VectorHistogram(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    uint64_t totalNum);

class ConfusionMatrix {
public:
    __aicore__ inline ConfusionMatrix() {};

    __aicore__ inline void Init(
        GM_ADDR labels,
        GM_ADDR predictions,
        GM_ADDR weights,
        GM_ADDR y,
        const ConfusionMatrixTilingData* tilingData);
    __aicore__ inline void Process();
#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
    __aicore__ inline void ProcessAtomicSparse();
#endif

private:
    __aicore__ inline int64_t ReadLabel(uint64_t offset);
    __aicore__ inline int64_t ReadPrediction(uint64_t offset);
    __aicore__ inline int32_t ReadWeightAsInt32(uint64_t offset);
    __aicore__ inline int64_t ReadWeightAsInt64(uint64_t offset);
    __aicore__ inline float ReadWeightAsFloat(uint64_t offset);
    __aicore__ inline void ClearOutput();
    __aicore__ inline void ClearOutputRange(uint64_t start, uint64_t end);
    __aicore__ inline void ClearOutputRangeInt32(uint64_t start, uint64_t end);
    __aicore__ inline void ClearOutputRangeFloat(uint64_t start, uint64_t end);
    __aicore__ inline void AddToOutput(uint64_t offset, uint64_t sampleIdx);
    __aicore__ inline void ProcessSingleCore();
    __aicore__ inline void ProcessSplitOutput();
    __aicore__ inline bool CanUseSmallOutputRowSplit(uint64_t totalNum) const;
    __aicore__ inline void ProcessSmallOutputRowSplit(uint64_t totalNum);
    __aicore__ inline bool CanUseSparseSplitInt32(uint64_t totalNum) const;
    __aicore__ inline void ProcessSplitOutputSparseInt32(
        uint64_t rowStart,
        uint64_t rowEnd,
        uint64_t classCount,
        uint64_t totalNum);

private:
    GlobalTensor<int32_t> labelsI32Gm_;
    GlobalTensor<int64_t> labelsI64Gm_;
    GlobalTensor<int32_t> predictionsI32Gm_;
    GlobalTensor<int64_t> predictionsI64Gm_;
    GlobalTensor<int32_t> weightsI32Gm_;
    GlobalTensor<int64_t> weightsI64Gm_;
    GlobalTensor<float> weightsF32Gm_;
    GlobalTensor<int32_t> outputI32Gm_;
    GlobalTensor<int32_t> outputI32BypassGm_;
    GlobalTensor<int64_t> outputI64Gm_;
    GlobalTensor<float> outputF32Gm_;
    TPipe pipe_;
    TBuf<TPosition::VECOUT> zeroBuf_;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 1;
    int64_t numClasses_ = 0;
    int64_t outputNum_ = 0;
    int32_t labelsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t predictionsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t weightsDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t outputDtype_ = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t hasWeights_ = 0;
    int32_t reserved_ = 0;
    int32_t useBulkClear_ = 0;
    uint64_t sparseNormalStart_ = 0;
};

__aicore__ inline void ConfusionMatrix::Init(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR weights,
    GM_ADDR y,
    const ConfusionMatrixTilingData* tilingData)
{
    labelsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    labelsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(labels));
    predictionsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));
    predictionsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(predictions));
    outputI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI32BypassGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    outputI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y));
    outputF32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    numClasses_ = tilingData->numClasses;
    outputNum_ = numClasses_ * numClasses_;
#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
    if (static_cast<uint64_t>(outputNum_) >
        CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES / sizeof(int32_t)) {
        outputI32Gm_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    labelsDtype_ = tilingData->labelsDtype;
    predictionsDtype_ = tilingData->predictionsDtype;
    weightsDtype_ = tilingData->weightsDtype;
    outputDtype_ = tilingData->outputDtype;
    hasWeights_ = tilingData->hasWeights;
    reserved_ = tilingData->reserved;
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        outputI32BypassGm_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    }
    if ((outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 || outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) &&
        outputNum_ >= static_cast<int64_t>(CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS)) {
        useBulkClear_ = 1;
        const uint32_t bufferBytes =
            (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 ?
            CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES : CONFUSION_MATRIX_BULK_BUFFER_BYTES;
        pipe_.InitBuffer(zeroBuf_, bufferBytes);
    }

    if (hasWeights_ != 0) {
        weightsI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(weights));
        weightsI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(weights));
        weightsF32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    }
}

__aicore__ inline int64_t ConfusionMatrix::ReadLabel(uint64_t offset)
{
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return labelsI64Gm_.GetValue(offset);
    }
    return static_cast<int64_t>(labelsI32Gm_.GetValue(offset));
}

__aicore__ inline int64_t ConfusionMatrix::ReadPrediction(uint64_t offset)
{
    if (predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return predictionsI64Gm_.GetValue(offset);
    }
    return static_cast<int64_t>(predictionsI32Gm_.GetValue(offset));
}

__aicore__ inline int32_t ConfusionMatrix::ReadWeightAsInt32(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return static_cast<int32_t>(weightsI64Gm_.GetValue(offset));
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return static_cast<int32_t>(weightsF32Gm_.GetValue(offset));
    }
    return weightsI32Gm_.GetValue(offset);
}

__aicore__ inline int64_t ConfusionMatrix::ReadWeightAsInt64(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return weightsI64Gm_.GetValue(offset);
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return static_cast<int64_t>(weightsF32Gm_.GetValue(offset));
    }
    return static_cast<int64_t>(weightsI32Gm_.GetValue(offset));
}

__aicore__ inline float ConfusionMatrix::ReadWeightAsFloat(uint64_t offset)
{
    if (hasWeights_ == 0) {
        return 1.0f;
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        return static_cast<float>(weightsI64Gm_.GetValue(offset));
    }
    if (weightsDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        return weightsF32Gm_.GetValue(offset);
    }
    return static_cast<float>(weightsI32Gm_.GetValue(offset));
}

__aicore__ inline void ConfusionMatrix::ClearOutput()
{
    ClearOutputRange(0, static_cast<uint64_t>(outputNum_));
}

__aicore__ inline void ConfusionMatrix::ClearOutputRange(uint64_t start, uint64_t end)
{
    const uint64_t outputNum = static_cast<uint64_t>(outputNum_);
    if (start >= outputNum) {
        return;
    }
    if (end > outputNum) {
        end = outputNum;
    }
    if (outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        if (useBulkClear_ != 0 && end - start >= CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS) {
            ClearOutputRangeFloat(start, end);
            return;
        }
        for (uint64_t i = start; i < end; ++i) {
            outputF32Gm_.SetValue(i, 0.0f);
        }
    } else if (outputDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        for (uint64_t i = start; i < end; ++i) {
            outputI64Gm_.SetValue(i, static_cast<int64_t>(0));
        }
    } else {
        if (useBulkClear_ != 0 && end - start >= CONFUSION_MATRIX_BULK_CLEAR_MIN_ELEMS) {
            ClearOutputRangeInt32(start, end);
            return;
        }
        for (uint64_t i = start; i < end; ++i) {
            outputI32Gm_.SetValue(i, static_cast<int32_t>(0));
        }
    }
}

__aicore__ inline void ConfusionMatrix::ClearOutputRangeInt32(uint64_t start, uint64_t end)
{
    const bool useMixedL2 =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 &&
        sparseNormalStart_ > start && sparseNormalStart_ < end;
    uint64_t offset = start;
    while (offset < end && (offset & 7UL) != 0) {
        if (useMixedL2 && offset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(offset, static_cast<int32_t>(0));
        } else {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
        }
        ++offset;
    }

    const uint64_t alignedEnd = end & (~7UL);
    if (offset >= alignedEnd) {
        while (offset < end) {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
            ++offset;
        }
        return;
    }

    LocalTensor<int32_t> zeroLocal = zeroBuf_.Get<int32_t>();
    const uint32_t clearTileElems =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0 ?
        CONFUSION_MATRIX_SPARSE_FILTER_BUFFER_BYTES / sizeof(int32_t) :
        CONFUSION_MATRIX_ZERO_TILE_ELEMS;
    Duplicate(zeroLocal, static_cast<int32_t>(0), clearTileElems);
    TEventID eventIdVToMte3 = static_cast<TEventID>(0);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // Keep the long case-8 clear copies on 512-byte GM boundaries.  The row
    // partition only guarantees 32-byte alignment, so issue one short prefix
    // before the 184 KiB tiles.  Other cases keep their original clear path.
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        const uint32_t offsetIn512Bytes = static_cast<uint32_t>(offset & 127UL);
        if (offsetIn512Bytes != 0 && offset < alignedEnd) {
            uint64_t prefixElems = 128U - offsetIn512Bytes;
            const uint64_t remainingElems = alignedEnd - offset;
            if (prefixElems > remainingElems) {
                prefixElems = remainingElems;
            }
            if (useMixedL2 && offset < sparseNormalStart_) {
                const uint64_t bypassRemaining = sparseNormalStart_ - offset;
                if (prefixElems > bypassRemaining) {
                    prefixElems = bypassRemaining;
                }
                DataCopy(outputI32BypassGm_[offset], zeroLocal, static_cast<uint32_t>(prefixElems));
            } else {
                DataCopy(outputI32Gm_[offset], zeroLocal, static_cast<uint32_t>(prefixElems));
            }
            offset += prefixElems;
        }
    }

    while (offset < alignedEnd) {
        uint64_t segmentEnd = alignedEnd;
        if (useMixedL2 && offset < sparseNormalStart_) {
            segmentEnd = sparseNormalStart_;
        }
        uint64_t elems = segmentEnd - offset;
        if (elems > clearTileElems) {
            elems = clearTileElems;
        }
        if (useMixedL2 && offset < sparseNormalStart_) {
            DataCopy(outputI32BypassGm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        } else {
            DataCopy(outputI32Gm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        }
        offset += elems;
    }
    PipeBarrier<PIPE_ALL>();

    while (offset < end) {
        if (useMixedL2 && offset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(offset, static_cast<int32_t>(0));
        } else {
            outputI32Gm_.SetValue(offset, static_cast<int32_t>(0));
        }
        ++offset;
    }
}

__aicore__ inline void ConfusionMatrix::ClearOutputRangeFloat(uint64_t start, uint64_t end)
{
    uint64_t offset = start;
    while (offset < end && (offset & 7UL) != 0) {
        outputF32Gm_.SetValue(offset, 0.0f);
        ++offset;
    }

    const uint64_t alignedEnd = end & (~7UL);
    if (offset >= alignedEnd) {
        while (offset < end) {
            outputF32Gm_.SetValue(offset, 0.0f);
            ++offset;
        }
        return;
    }

    LocalTensor<float> zeroLocal = zeroBuf_.Get<float>();
    Duplicate(zeroLocal, 0.0f, CONFUSION_MATRIX_ZERO_TILE_ELEMS);
    TEventID eventIdVToMte3 = static_cast<TEventID>(0);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    while (offset < alignedEnd) {
        uint64_t elems = alignedEnd - offset;
        if (elems > CONFUSION_MATRIX_ZERO_TILE_ELEMS) {
            elems = CONFUSION_MATRIX_ZERO_TILE_ELEMS;
        }
        DataCopy(outputF32Gm_[offset], zeroLocal, static_cast<uint32_t>(elems));
        offset += elems;
    }
    PipeBarrier<PIPE_ALL>();

    while (offset < end) {
        outputF32Gm_.SetValue(offset, 0.0f);
        ++offset;
    }
}

__aicore__ inline void ConfusionMatrix::AddToOutput(uint64_t offset, uint64_t sampleIdx)
{
    if (outputDtype_ == CONFUSION_MATRIX_DTYPE_FLOAT32) {
        const float oldValue = outputF32Gm_.GetValue(offset);
        outputF32Gm_.SetValue(offset, oldValue + ReadWeightAsFloat(sampleIdx));
    } else if (outputDtype_ == CONFUSION_MATRIX_DTYPE_INT64) {
        const int64_t oldValue = outputI64Gm_.GetValue(offset);
        outputI64Gm_.SetValue(offset, oldValue + ReadWeightAsInt64(sampleIdx));
    } else {
        const int32_t oldValue = outputI32Gm_.GetValue(offset);
        outputI32Gm_.SetValue(offset, oldValue + ReadWeightAsInt32(sampleIdx));
    }
}

__aicore__ inline void ConfusionMatrix::Process()
{
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_SPLIT_OUTPUT) != 0) {
        ProcessSplitOutput();
        return;
    }
    ProcessSingleCore();
}

#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
__aicore__ inline void ConfusionMatrix::ProcessAtomicSparse()
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);

    LocalTensor<int32_t> addLocal = zeroBuf_.Get<int32_t>();
    LocalTensor<int32_t> clearLocal = addLocal[64];
    for (uint32_t i = 0; i < 64; ++i) {
        addLocal.SetValue(i, static_cast<int32_t>(0));
    }
    for (uint32_t i = 0; i < 8; ++i) {
        clearLocal.SetValue(i, static_cast<int32_t>(0));
    }
    for (uint32_t lane = 0; lane < 8; ++lane) {
        addLocal.SetValue(lane * 8 + lane, static_cast<int32_t>(1));
    }
    const TEventID atomicDataReadyEvent = static_cast<TEventID>(0);
    SetFlag<HardEvent::S_MTE3>(atomicDataReadyEvent);
    WaitFlag<HardEvent::S_MTE3>(atomicDataReadyEvent);

    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    const uint64_t samplesPerCore = (totalNum + blockNum - 1) / blockNum;
    const uint64_t sampleStart = blockIdx * samplesPerCore;
    uint64_t sampleEnd = sampleStart + samplesPerCore;
    if (sampleEnd > totalNum) {
        sampleEnd = totalNum;
    }

    __gm__ int32_t* clearLabelsPtr = labelsI32Gm_.GetPhyAddr(sampleStart);
    __gm__ int32_t* clearPredictionsPtr = predictionsI32Gm_.GetPhyAddr(sampleStart);
    for (uint64_t i = sampleStart; i < sampleEnd; ++i) {
        const int32_t labelValue = *clearLabelsPtr;
        const int32_t predictionValue = *clearPredictionsPtr;
        ++clearLabelsPtr;
        ++clearPredictionsPtr;
        if (labelValue < 0 || predictionValue < 0 ||
            labelValue >= numClasses_ || predictionValue >= numClasses_) {
            continue;
        }
        const uint32_t outOffset = static_cast<uint32_t>(labelValue) * static_cast<uint32_t>(classCount) +
            static_cast<uint32_t>(predictionValue);
        DataCopy(outputI32Gm_[outOffset & ~7U], clearLocal, 8);
    }
    const TEventID clearCompleteEvent = static_cast<TEventID>(1);
    SetFlag<HardEvent::MTE3_S>(clearCompleteEvent);
    WaitFlag<HardEvent::MTE3_S>(clearCompleteEvent);
    SyncAll();

    __gm__ int32_t* labelsPtr = labelsI32Gm_.GetPhyAddr(sampleStart);
    __gm__ int32_t* predictionsPtr = predictionsI32Gm_.GetPhyAddr(sampleStart);
    SetAtomicAdd<int32_t>();
    for (uint64_t i = sampleStart; i < sampleEnd; ++i) {
        const int32_t labelValue = *labelsPtr;
        const int32_t predictionValue = *predictionsPtr;
        ++labelsPtr;
        ++predictionsPtr;
        if (labelValue < 0 || predictionValue < 0 ||
            labelValue >= numClasses_ || predictionValue >= numClasses_) {
            continue;
        }
        const uint32_t outOffset = static_cast<uint32_t>(labelValue) * static_cast<uint32_t>(classCount) +
            static_cast<uint32_t>(predictionValue);
        const uint32_t lane = outOffset & 7U;
        DataCopy(outputI32Gm_[outOffset - lane], addLocal[lane * 8], 8);
    }
    SetAtomicNone();
}
#endif

__aicore__ inline void ConfusionMatrix::ProcessSingleCore()
{
    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    if (CanUseSmallOutputRowSplit(totalNum)) {
        ProcessSmallOutputRowSplit(totalNum);
        return;
    }

    if (GetBlockIdx() != 0) {
        return;
    }
    ClearOutput();

    const int64_t numClasses = numClasses_;
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0) {
        if (numClasses == 3 && totalNum >= 512) {
            int32_t h0 = 0;
            int32_t h1 = 0;
            int32_t h2 = 0;
            int32_t h3 = 0;
            int32_t h4 = 0;
            int32_t h5 = 0;
            int32_t h6 = 0;
            int32_t h7 = 0;
            int32_t h8 = 0;
            uint64_t i = 0;
#define CM_C3_HIST9_STEP(IDX) \
            do { \
                const uint64_t label = static_cast<uint64_t>(labelsI32Gm_.GetValue(IDX)); \
                const uint64_t prediction = static_cast<uint64_t>(predictionsI32Gm_.GetValue(IDX)); \
                if (label >= 3 || prediction >= 3) { \
                    break; \
                } \
                switch (label * 3 + prediction) { \
                    case 0: ++h0; break; \
                    case 1: ++h1; break; \
                    case 2: ++h2; break; \
                    case 3: ++h3; break; \
                    case 4: ++h4; break; \
                    case 5: ++h5; break; \
                    case 6: ++h6; break; \
                    case 7: ++h7; break; \
                    case 8: ++h8; break; \
                    default: break; \
                } \
            } while (false)
            for (; i + 5 < totalNum; i += 6) {
                CM_C3_HIST9_STEP(i);
                CM_C3_HIST9_STEP(i + 1);
                CM_C3_HIST9_STEP(i + 2);
                CM_C3_HIST9_STEP(i + 3);
                CM_C3_HIST9_STEP(i + 4);
                CM_C3_HIST9_STEP(i + 5);
            }
            for (; i < totalNum; ++i) {
                CM_C3_HIST9_STEP(i);
            }
#undef CM_C3_HIST9_STEP
            outputI32Gm_.SetValue(0, h0);
            outputI32Gm_.SetValue(1, h1);
            outputI32Gm_.SetValue(2, h2);
            outputI32Gm_.SetValue(3, h3);
            outputI32Gm_.SetValue(4, h4);
            outputI32Gm_.SetValue(5, h5);
            outputI32Gm_.SetValue(6, h6);
            outputI32Gm_.SetValue(7, h7);
            outputI32Gm_.SetValue(8, h8);
            return;
        }
        for (uint64_t i = 0; i < totalNum; ++i) {
            const int32_t labelValue = labelsI32Gm_.GetValue(i);
            const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
            if (labelValue < 0 || predictionValue < 0 ||
                labelValue >= numClasses || predictionValue >= numClasses) {
                continue;
            }
            const uint64_t label = static_cast<uint64_t>(labelValue);
            const uint64_t prediction = static_cast<uint64_t>(predictionValue);
            const uint64_t outOffset = label * static_cast<uint64_t>(numClasses) + prediction;
            const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
            outputI32Gm_.SetValue(outOffset, oldValue + 1);
        }
        return;
    }

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int64_t label = ReadLabel(i);
        const int64_t prediction = ReadPrediction(i);
        if (label < 0 || prediction < 0 || label >= numClasses || prediction >= numClasses) {
            continue;
        }
        const uint64_t outOffset = static_cast<uint64_t>(label) * static_cast<uint64_t>(numClasses) +
            static_cast<uint64_t>(prediction);
        AddToOutput(outOffset, i);
    }
}

template <bool useMte2TailPadding>
__aicore__ inline void ProcessC3VectorHistogram(
    GM_ADDR labels,
    GM_ADDR predictions,
    GM_ADDR y,
    uint64_t totalNum)
{
    GlobalTensor<int32_t> labelsGm;
    GlobalTensor<int32_t> predictionsGm;
    labelsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(labels));
    predictionsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(predictions));

    LocalTensor<int32_t> dataLocal(
        TPosition::VECCALC, 0, CONFUSION_MATRIX_C3_VECTOR_WORK_BUFFER_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> labelsLocal = dataLocal;
    LocalTensor<int32_t> predictionsLocal = dataLocal[CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS];
    LocalTensor<uint8_t> maskStorage =
        dataLocal.ReinterpretCast<uint8_t>()[CONFUSION_MATRIX_C3_VECTOR_DATA_BUFFER_BYTES];
    LocalTensor<uint8_t> labelMask = maskStorage;
    LocalTensor<uint8_t> predictionMask0 = maskStorage[CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES];
    LocalTensor<uint8_t> predictionMask1 = maskStorage[CONFUSION_MATRIX_C3_VECTOR_MASK_BYTES * 2];

    int32_t h0 = 0;
    int32_t h1 = 0;
    int32_t h2 = 0;
    const int32_t row = static_cast<int32_t>(GetBlockIdx());
    uint64_t offset = 0;
    const TEventID copyEvent = static_cast<TEventID>(0);
    const TEventID countEvent = static_cast<TEventID>(1);

    do {
        uint32_t copyCount = 0;
        uint32_t vectorCount = 0;
        if constexpr (useMte2TailPadding) {
            copyCount = static_cast<uint32_t>(totalNum);
            vectorCount = (copyCount + 63U) & ~63U;
            const uint32_t paddingCount = vectorCount - copyCount;
            DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
            DataCopyPadParams inputPadParams{
                true, 0, static_cast<uint8_t>(paddingCount),
                static_cast<uint64_t>(static_cast<uint32_t>(-1))};
            DataCopyPad(labelsLocal, labelsGm[offset], inputParams, inputPadParams);
        } else {
            const uint64_t remaining = totalNum - offset;
            const bool usePaddedTail =
                remaining <= CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS && (remaining & 7UL) == 0;
            copyCount = remaining > CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS ?
                CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS : static_cast<uint32_t>(remaining);
            vectorCount = copyCount;
            if (usePaddedTail) {
                vectorCount = (copyCount + 63U) & ~63U;
                DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
                DataCopyPadParams inputPadParams{true, 0, 0, 0};
                DataCopyPad(labelsLocal, labelsGm[offset], inputParams, inputPadParams);
            } else {
                vectorCount &= ~63U;
                copyCount = vectorCount;
                DataCopy(labelsLocal, labelsGm[offset], vectorCount);
            }
        }
        SetFlag<HardEvent::MTE2_V>(copyEvent);
        WaitFlag<HardEvent::MTE2_V>(copyEvent);
        if constexpr (!useMte2TailPadding) {
            if (vectorCount > copyCount) {
                Duplicate(labelsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                PipeBarrier<PIPE_V>();
            }
        }

        if constexpr (useMte2TailPadding) {
            const uint32_t paddingCount = vectorCount - copyCount;
            DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
            DataCopyPadParams inputPadParams{
                true, 0, static_cast<uint8_t>(paddingCount),
                static_cast<uint64_t>(static_cast<uint32_t>(-1))};
            DataCopyPad(predictionsLocal, predictionsGm[offset], inputParams, inputPadParams);
        } else {
            const uint64_t remaining = totalNum - offset;
            const bool usePaddedTail =
                remaining <= CONFUSION_MATRIX_C3_VECTOR_TILE_ELEMS && (remaining & 7UL) == 0;
            if (usePaddedTail) {
                DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
                DataCopyPadParams inputPadParams{true, 0, 0, 0};
                DataCopyPad(predictionsLocal, predictionsGm[offset], inputParams, inputPadParams);
            } else {
                DataCopy(predictionsLocal, predictionsGm[offset], vectorCount);
            }
        }

#define CM_C3_MASK_COUNT(MASK, COUNT) \
        do { \
            uint64_t selectedCount = 0; \
            GatherMask( \
                predictionsLocal, labelsLocal, MASK.ReinterpretCast<uint32_t>(), true, vectorCount, \
                {1, 1, 8, 0}, selectedCount); \
            PipeBarrier<PIPE_V>(); \
            SetFlag<HardEvent::V_S>(countEvent); \
            WaitFlag<HardEvent::V_S>(countEvent); \
            COUNT = static_cast<int32_t>(selectedCount); \
        } while (false)
        CompareScalar(labelMask, labelsLocal, row, CMPMODE::EQ, vectorCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_V>(copyEvent);
        WaitFlag<HardEvent::MTE2_V>(copyEvent);
        if constexpr (!useMte2TailPadding) {
            if (vectorCount > copyCount) {
                Duplicate(predictionsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                PipeBarrier<PIPE_V>();
            }
        }
        uint64_t selectedRowCount = 0;
        GatherMask(
            labelsLocal, predictionsLocal, labelMask.ReinterpretCast<uint32_t>(), true, vectorCount,
            {1, 1, 8, 0}, selectedRowCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(countEvent);
        WaitFlag<HardEvent::V_S>(countEvent);
        const int32_t rowCount = static_cast<int32_t>(selectedRowCount);
        if (rowCount > 0) {
            const uint32_t rowCountU32 = static_cast<uint32_t>(rowCount);
            const uint32_t compactVectorCount = (rowCountU32 + 63U) & ~63U;
            const uint32_t alignedRowCount = (rowCountU32 + 7U) & ~7U;
            if (alignedRowCount > rowCountU32) {
                for (uint32_t i = rowCountU32; i < alignedRowCount; ++i) {
                    labelsLocal.SetValue(i, static_cast<int32_t>(-1));
                }
                SetFlag<HardEvent::S_V>(copyEvent);
                WaitFlag<HardEvent::S_V>(copyEvent);
            }
            if (compactVectorCount > alignedRowCount) {
                Duplicate(labelsLocal[alignedRowCount], static_cast<int32_t>(-1), compactVectorCount - alignedRowCount);
                PipeBarrier<PIPE_V>();
            }
            CompareScalar(predictionMask0, labelsLocal, static_cast<int32_t>(0), CMPMODE::EQ, compactVectorCount);
            CompareScalar(predictionMask1, labelsLocal, static_cast<int32_t>(1), CMPMODE::EQ, compactVectorCount);
            PipeBarrier<PIPE_V>();
            vectorCount = compactVectorCount;
            int32_t count0 = 0;
            int32_t count1 = 0;
            CM_C3_MASK_COUNT(predictionMask0, count0);
            CM_C3_MASK_COUNT(predictionMask1, count1);
            const int32_t count2 = rowCount - count0 - count1;
            h0 += count0;
            h1 += count1;
            h2 += count2;
        }
#undef CM_C3_MASK_COUNT
        offset += copyCount;
        if constexpr (useMte2TailPadding) {
            break;
        }
    } while (offset + 64 <= totalNum);

    for (; offset < totalNum; ++offset) {
        const int32_t label = labelsGm.GetValue(offset);
        const int32_t prediction = predictionsGm.GetValue(offset);
        if (label != row || prediction < 0 || prediction >= 3) {
            continue;
        }
        switch (prediction) {
            case 0: ++h0; break;
            case 1: ++h1; break;
            case 2: ++h2; break;
            default: break;
        }
    }

    const uint64_t outputOffset = static_cast<uint64_t>(row) * 3;
    __ubuf__ int32_t* outputLocal = reinterpret_cast<__ubuf__ int32_t*>(dataLocal.GetPhyAddr());
    reinterpret_cast<__ubuf__ uint64_t*>(outputLocal)[0] =
        static_cast<uint64_t>(static_cast<uint32_t>(h0)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(h1)) << 32U);
    outputLocal[2] = h2;
    const TEventID outputReadyEvent = static_cast<TEventID>(2);
    SetFlag<HardEvent::S_MTE3>(outputReadyEvent);
    WaitFlag<HardEvent::S_MTE3>(outputReadyEvent);
    __gm__ int32_t* outputPtr = reinterpret_cast<__gm__ int32_t*>(y) + outputOffset;
    copy_ubuf_to_gm_align_b32(
        outputPtr, outputLocal, 0, 1, 3 * sizeof(int32_t), 0, 0, 0, 0);
}

__aicore__ inline bool ConfusionMatrix::CanUseSmallOutputRowSplit(uint64_t totalNum) const
{
    if (numClasses_ <= 1 || numClasses_ > static_cast<int64_t>(CONFUSION_MATRIX_SMALL_ROW_SPLIT_MAX_CLASSES)) {
        return false;
    }
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowBytes = classCount * sizeof(int32_t);
    return labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0 &&
        totalNum >= CONFUSION_MATRIX_SMALL_ROW_SPLIT_MIN_SAMPLES &&
        rowBytes >= 32 &&
        (rowBytes & 31UL) == 0 &&
        GetBlockNum() > 1;
}

__aicore__ inline void ConfusionMatrix::ProcessSmallOutputRowSplit(uint64_t totalNum)
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowsPerCore = (classCount + blockNum - 1) / blockNum;
    const uint64_t rowStart = blockIdx * rowsPerCore;
    if (rowStart >= classCount) {
        return;
    }
    uint64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > classCount) {
        rowEnd = classCount;
    }

    ClearOutputRange(rowStart * classCount, rowEnd * classCount);

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int32_t labelValue = labelsI32Gm_.GetValue(i);
        if (labelValue < 0) {
            continue;
        }
        const uint64_t label = static_cast<uint64_t>(labelValue);
        if (label < rowStart || label >= rowEnd) {
            continue;
        }
        const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
        if (predictionValue < 0) {
            continue;
        }
        const uint64_t prediction = static_cast<uint64_t>(predictionValue);
        if (prediction >= classCount) {
            continue;
        }
        const uint64_t outOffset = label * classCount + prediction;
        const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
        outputI32Gm_.SetValue(outOffset, oldValue + 1);
    }
}

__aicore__ inline bool ConfusionMatrix::CanUseSparseSplitInt32(uint64_t totalNum) const
{
    if (useBulkClear_ == 0 || totalNum == 0 || totalNum > CONFUSION_MATRIX_SPARSE_CAPACITY ||
        outputNum_ > static_cast<int64_t>(CONFUSION_MATRIX_UINT32_OFFSET_MAX)) {
        return false;
    }
    const uint64_t outputNum = static_cast<uint64_t>(outputNum_);
    return outputNum / totalNum >= totalNum;
}

__aicore__ inline void ConfusionMatrix::ProcessSplitOutputSparseInt32(
    uint64_t rowStart,
    uint64_t rowEnd,
    uint64_t classCount,
    uint64_t totalNum)
{
    const bool useVectorFilter =
        (reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0;
    LocalTensor<int32_t> offsetLowLocal = zeroBuf_.Get<int32_t>();
    LocalTensor<int32_t> countLocal = zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_COUNT_LOCAL_OFFSET];
    LocalTensor<int32_t> occupiedSlotLocal =
        zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_OCCUPIED_LOCAL_OFFSET];
    if (!useVectorFilter) {
        Duplicate(countLocal, static_cast<int32_t>(0), CONFUSION_MATRIX_SPARSE_CAPACITY);
        TEventID eventIdVToS = static_cast<TEventID>(1);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    uint32_t occupiedCount = 0;
    const int32_t rowStartI32 = static_cast<int32_t>(rowStart);
    const int32_t rowEndI32 = static_cast<int32_t>(rowEnd);
    const int32_t classCountI32 = static_cast<int32_t>(classCount);
#define CM_ACCUMULATE_SPARSE_OFFSET(OUT_OFFSET_LOW) \
    do { \
        const uint32_t currentOffsetLow = (OUT_OFFSET_LOW); \
        const uint32_t hashValue = currentOffsetLow ^ (currentOffsetLow >> 7); \
        uint32_t slot = hashValue % CONFUSION_MATRIX_SPARSE_CAPACITY; \
        for (uint32_t probe = 0; probe < CONFUSION_MATRIX_SPARSE_CAPACITY; ++probe) { \
            const int32_t oldCount = countLocal.GetValue(slot); \
            if (oldCount == 0) { \
                offsetLowLocal.SetValue(slot, static_cast<int32_t>(currentOffsetLow)); \
                countLocal.SetValue(slot, static_cast<int32_t>(1)); \
                occupiedSlotLocal.SetValue(occupiedCount, static_cast<int32_t>(slot)); \
                ++occupiedCount; \
                break; \
            } \
            if (static_cast<uint32_t>(offsetLowLocal.GetValue(slot)) == currentOffsetLow) { \
                countLocal.SetValue(slot, oldCount + 1); \
                break; \
            } \
            ++slot; \
            if (slot == CONFUSION_MATRIX_SPARSE_CAPACITY) { \
                slot = 0; \
            } \
        } \
    } while (false)

    if (useVectorFilter) {
        LocalTensor<int32_t> labelsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_LABEL_LOCAL_OFFSET];
        LocalTensor<int32_t> predictionsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_PREDICTION_LOCAL_OFFSET];
        LocalTensor<int32_t> compactLabelsLocal =
            zeroBuf_.Get<int32_t>()[CONFUSION_MATRIX_SPARSE_FILTER_COMPACT_LABEL_LOCAL_OFFSET];
        LocalTensor<float> labelsFloatLocal = compactLabelsLocal.ReinterpretCast<float>();
        LocalTensor<uint8_t> maskStorage =
            zeroBuf_.Get<uint8_t>()[CONFUSION_MATRIX_SPARSE_FILTER_MASK_LOCAL_OFFSET_BYTES];
        LocalTensor<uint8_t> lowerMask = maskStorage;
        LocalTensor<uint8_t> upperMask = maskStorage[CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES];
        LocalTensor<uint8_t> ownedMask = maskStorage[CONFUSION_MATRIX_SPARSE_FILTER_MASK_BYTES * 2];

        const TEventID copyEvent = static_cast<TEventID>(0);
        const TEventID selectedCountEvent = static_cast<TEventID>(2);
        uint64_t offset = 0;
        while (offset + 64 <= totalNum) {
            const uint64_t remaining = totalNum - offset;
            const bool usePaddedTail = remaining <= CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS &&
                (remaining & 7UL) == 0;
            uint32_t copyCount = remaining > CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS ?
                CONFUSION_MATRIX_SPARSE_FILTER_TILE_ELEMS : static_cast<uint32_t>(remaining);
            uint32_t vectorCount = copyCount;
            if (usePaddedTail) {
                vectorCount = (copyCount + 63U) & ~63U;
                DataCopyParams inputParams{1, static_cast<uint16_t>(copyCount * sizeof(int32_t)), 0, 0};
                DataCopyPadParams inputPadParams{true, 0, 0, 0};
                DataCopyPad(labelsLocal, labelsI32Gm_[offset], inputParams, inputPadParams);
                DataCopyPad(predictionsLocal, predictionsI32Gm_[offset], inputParams, inputPadParams);
            } else {
                vectorCount &= ~63U;
                copyCount = vectorCount;
                DataCopy(labelsLocal, labelsI32Gm_[offset], vectorCount);
                DataCopy(predictionsLocal, predictionsI32Gm_[offset], vectorCount);
            }
            SetFlag<HardEvent::MTE2_V>(copyEvent);
            WaitFlag<HardEvent::MTE2_V>(copyEvent);
            if (vectorCount > copyCount) {
                Duplicate(labelsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                Duplicate(predictionsLocal[copyCount], static_cast<int32_t>(-1), vectorCount - copyCount);
                PipeBarrier<PIPE_V>();
            }

            Cast(labelsFloatLocal, labelsLocal, RoundMode::CAST_NONE, vectorCount);
            PipeBarrier<PIPE_V>();
            CompareScalar(lowerMask, labelsFloatLocal, static_cast<float>(rowStartI32), CMPMODE::GE, vectorCount);
            CompareScalar(upperMask, labelsFloatLocal, static_cast<float>(rowEndI32), CMPMODE::LT, vectorCount);
            PipeBarrier<PIPE_V>();
            And(
                ownedMask.ReinterpretCast<uint16_t>(), lowerMask.ReinterpretCast<uint16_t>(),
                upperMask.ReinterpretCast<uint16_t>(), vectorCount / 16);
            PipeBarrier<PIPE_V>();

            uint64_t selectedLabelCount = 0;
            GatherMask(
                compactLabelsLocal, labelsLocal, ownedMask.ReinterpretCast<uint32_t>(), true, vectorCount,
                {1, 1, 8, 0}, selectedLabelCount);
            PipeBarrier<PIPE_V>();
            uint64_t selectedCount = 0;
            GatherMask(
                labelsLocal, predictionsLocal, ownedMask.ReinterpretCast<uint32_t>(), true, vectorCount,
                {1, 1, 8, 0}, selectedCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(selectedCountEvent);
            WaitFlag<HardEvent::V_S>(selectedCountEvent);
            for (uint32_t j = 0; j < static_cast<uint32_t>(selectedCount); ++j) {
                const int32_t labelValue = compactLabelsLocal.GetValue(j);
                const int32_t predictionValue = labelsLocal.GetValue(j);
                if (predictionValue < 0 || predictionValue >= classCountI32) {
                    continue;
                }
                const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                    static_cast<uint64_t>(predictionValue);
                CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
            }
            offset += copyCount;
        }

        for (; offset < totalNum; ++offset) {
            const int32_t labelValue = labelsI32Gm_.GetValue(offset);
            if (labelValue < rowStartI32 || labelValue >= rowEndI32) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(offset);
            if (predictionValue < 0 || predictionValue >= classCountI32) {
                continue;
            }
            const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                static_cast<uint64_t>(predictionValue);
            CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
        }
    } else {
        for (uint64_t i = 0; i < totalNum; ++i) {
            const int32_t labelValue = labelsI32Gm_.GetValue(i);
            if (labelValue < rowStartI32 || labelValue >= rowEndI32) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
            if (predictionValue < 0 || predictionValue >= classCountI32) {
                continue;
            }
            const uint64_t outOffset = static_cast<uint64_t>(labelValue) * classCount +
                static_cast<uint64_t>(predictionValue);
            CM_ACCUMULATE_SPARSE_OFFSET(static_cast<uint32_t>(outOffset));
        }
    }
#undef CM_ACCUMULATE_SPARSE_OFFSET

    for (uint32_t i = 0; i < occupiedCount; ++i) {
        const uint32_t k = static_cast<uint32_t>(occupiedSlotLocal.GetValue(i));
        const int32_t count = countLocal.GetValue(k);
        const uint64_t outOffset = static_cast<uint64_t>(static_cast<uint32_t>(offsetLowLocal.GetValue(k)));
        if (useVectorFilter && outOffset < sparseNormalStart_) {
            outputI32BypassGm_.SetValue(outOffset, count);
        } else {
            outputI32Gm_.SetValue(outOffset, count);
        }
    }
}

__aicore__ inline void ConfusionMatrix::ProcessSplitOutput()
{
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t classCount = static_cast<uint64_t>(numClasses_);
    const uint64_t rowsPerCore = static_cast<uint64_t>(blockFactor_);
    const uint64_t rowStart = blockIdx * rowsPerCore;
    if (rowStart >= classCount) {
        return;
    }
    uint64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > classCount) {
        rowEnd = classCount;
    }

    const uint64_t start = rowStart * classCount;
    const uint64_t end = rowEnd * classCount;
    sparseNormalStart_ = start;
    if ((reserved_ & CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE) != 0) {
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t normalTailElems =
            CONFUSION_MATRIX_SPARSE_NORMAL_L2_BUDGET_BYTES / sizeof(int32_t) / blockNum;
        normalTailElems &= ~127UL;
        if (normalTailElems != 0 && end - start > normalTailElems) {
            sparseNormalStart_ = (end - normalTailElems + 127UL) & ~127UL;
            if (sparseNormalStart_ > end) {
                sparseNormalStart_ = end;
            }
        }
    }
    ClearOutputRange(start, end);

    const uint64_t totalNum = static_cast<uint64_t>(totalNum_);
    if (labelsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype_ == CONFUSION_MATRIX_DTYPE_INT32 &&
        hasWeights_ == 0) {
        if (CanUseSparseSplitInt32(totalNum)) {
            ProcessSplitOutputSparseInt32(rowStart, rowEnd, classCount, totalNum);
            return;
        }
        for (uint64_t i = 0; i < totalNum; ++i) {
            const int32_t labelValue = labelsI32Gm_.GetValue(i);
            if (labelValue < 0) {
                continue;
            }
            const uint64_t label = static_cast<uint64_t>(labelValue);
            if (label < rowStart || label >= rowEnd) {
                continue;
            }
            const int32_t predictionValue = predictionsI32Gm_.GetValue(i);
            if (predictionValue < 0) {
                continue;
            }
            const uint64_t prediction = static_cast<uint64_t>(predictionValue);
            if (prediction >= classCount) {
                continue;
            }
            const uint64_t outOffset = label * classCount + prediction;
            const int32_t oldValue = outputI32Gm_.GetValue(outOffset);
            outputI32Gm_.SetValue(outOffset, oldValue + 1);
        }
        return;
    }

    for (uint64_t i = 0; i < totalNum; ++i) {
        const int64_t label = ReadLabel(i);
        if (label < 0) {
            continue;
        }
        const uint64_t labelOffset = static_cast<uint64_t>(label);
        if (labelOffset < rowStart || labelOffset >= rowEnd) {
            continue;
        }
        const int64_t prediction = ReadPrediction(i);
        if (prediction < 0 || prediction >= numClasses_) {
            continue;
        }
        const uint64_t outOffset = labelOffset * classCount + static_cast<uint64_t>(prediction);
        AddToOutput(outOffset, i);
    }
}

#endif

} // namespace NsConfusionMatrix
#endif // CONFUSIONMATRIX_H
