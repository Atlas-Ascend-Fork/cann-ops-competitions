/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 kernel implementation.
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/quantization/ascend_dequant.h"
#include "adv_api/reduce/reduce.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr auto QBM_CFG_MDL_INTRINSICS = GetMDLConfig(true);

constexpr uint32_t QBM_INT8_C0_ELEMS = 32U;

// Matmul owns the A1 slot and its lifecycle.  Build the exact transposed-A NZ
// image in that slot with one GM->A1 ND2NZ descriptor instead of gathering one
// 32-column fractal at a time or relaying A through compact user GM.
__aicore__ inline void CopyStridedTransA1Nd2Nz(
    const LocalTensor<int8_t>& aMatrix,
    const __gm__ void* gm,
    int row,
    int col,
    int useM,
    int useK,
    const uint64_t tilingPtr,
    const uint64_t dataPtr)
{
    (void)tilingPtr;
    const uint32_t sourcePitch = static_cast<uint32_t>(dataPtr);
    const uint32_t baseM = static_cast<uint32_t>((dataPtr >> 32U) & 0xFFFFUL);
    const uint32_t baseK = static_cast<uint32_t>((dataPtr >> 48U) & 0xFFFFUL);
    const uint32_t tileM = static_cast<uint32_t>(useM);
    const uint32_t tileK = static_cast<uint32_t>(useK);
    const uint32_t alignedK = (tileK + QBM_INT8_C0_ELEMS - 1U) /
        QBM_INT8_C0_ELEMS * QBM_INT8_C0_ELEMS;
    const uint64_t sourceBase =
        static_cast<uint64_t>(static_cast<uint32_t>(col)) * baseK * sourcePitch +
        static_cast<uint64_t>(static_cast<uint32_t>(row)) * baseM;
    // The physical int8 row pitch can be 65536 and therefore does not fit the
    // 16-bit ND2NZ srcDValue.  Reinterpreting four adjacent int8 values as one
    // int32 divides the pitch by four.  B32 C0 is eight elements, so each C0 is
    // still the same 32 raw bytes and the resulting A1 byte image is unchanged.
    LocalTensor<int32_t> aMatrix32 = aMatrix.template ReinterpretCast<int32_t>();
    GlobalTensor<int32_t> source32;
    source32.SetGlobalBuffer((__gm__ int32_t*)gm);
    Nd2NzParams copyParams;
    copyParams.ndNum = 1;
    copyParams.nValue = tileK;
    copyParams.dValue = tileM / sizeof(int32_t);
    copyParams.srcDValue = sourcePitch / sizeof(int32_t);
    copyParams.srcNdMatrixStride = 0;
    copyParams.dstNzC0Stride = alignedK;
    copyParams.dstNzNStride = 1;
    copyParams.dstNzMatrixStride = 0;
    DataCopy(aMatrix32, source32[sourceBase / sizeof(int32_t)], copyParams);
}

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
constexpr MatmulConfig QBM_CFG_DIRECT_FUSION =
    GetMDLConfig(true, false, 0, false, false, false, true, true, false, false, false, true);
#else
constexpr MatmulConfig QBM_CFG_DIRECT_FUSION = GetMDLConfig();
#endif

__aicore__ inline void CopyB1ByBlockAlignedParity(
    const LocalTensor<int8_t>& bMatrix,
    const __gm__ void* gm,
    int row,
    int col,
    int useK,
    int useN,
    const uint64_t tilingPtr,
    const uint64_t dataPtr)
{
    constexpr uint64_t DIM_MASK = 0xFFFFULL;
    constexpr uint32_t BASE_K_SHIFT = 24U;
    constexpr uint32_t BASE_N_SHIFT = 40U;
    const uint32_t orgN = static_cast<uint32_t>(tilingPtr & 0xFFFFFFULL);
    const uint32_t baseK = static_cast<uint32_t>((tilingPtr >> BASE_K_SHIFT) & DIM_MASK);
    const uint32_t baseN = static_cast<uint32_t>((tilingPtr >> BASE_N_SHIFT) & DIM_MASK);
    const uint32_t rowOffset = static_cast<uint32_t>(row) * baseK;
    const uint32_t colOffset = static_cast<uint32_t>(col) * baseN;
    const uint32_t alignedK = (static_cast<uint32_t>(useK) + 31U) / 32U * 32U;
    const uint32_t nBlocks = (static_cast<uint32_t>(useN) + 31U) / 32U;
    GlobalTensor<int8_t> src;
    src.SetGlobalBuffer((__gm__ int8_t*)gm);
    for (uint32_t nBlock = 0; nBlock < nBlocks; ++nBlock) {
        const uint64_t srcOffset = static_cast<uint64_t>(rowOffset) * orgN + colOffset + nBlock * 32U;
        const uint64_t dstOffset = static_cast<uint64_t>(nBlock) * alignedK * 32U;
        if ((orgN & 31U) == 0U) {
            DataCopyParams params{
                static_cast<uint16_t>(useK), 1,
                static_cast<uint16_t>(orgN / 32U - 1U), 0};
            DataCopy(bMatrix[dstOffset], src[srcOffset], params);
        } else {
            const uint16_t sameParityGap = static_cast<uint16_t>(orgN / 16U - 1U);
            const uint16_t evenRows = static_cast<uint16_t>((static_cast<uint32_t>(useK) + 1U) / 2U);
            DataCopyParams evenParams{evenRows, 1, sameParityGap, 1};
            DataCopy(bMatrix[dstOffset], src[srcOffset], evenParams);
            const uint16_t oddRows = static_cast<uint16_t>(static_cast<uint32_t>(useK) / 2U);
            if (oddRows > 0U) {
                DataCopyParams oddParams{oddRows, 1, sameParityGap, 1};
                DataCopy(bMatrix[dstOffset + 32U], src[srcOffset + orgN], oddParams);
            }
        }
    }
}

template <bool TRANS_X1, bool TRANS_X2, bool VECTOR_LARGE_M = false, bool NO_INTRINSICS = false,
    bool A_IBSHARE = false, bool INTERNAL_B1_PARITY = false, bool MID_N_INT32_PUBLIC = false>
class QuantBatchMatMulV3Matmul {
public:
    using ANormalMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, TRANS_X1>;
    using AIBShareMatmulType =
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, TRANS_X1, LayoutMode::NONE, true>;
    using AMatmulType = typename Conditional<A_IBSHARE, AIBShareMatmulType, ANormalMatmulType>::type;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, TRANS_X2>;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

    static constexpr auto QBM_CFG =
        A_IBSHARE ? GetIBShareNormConfig(false) :
        (NO_INTRINSICS ? GetMDLConfig(false) : QBM_CFG_MDL_INTRINSICS);
    using DefaultCallback = MatmulCallBackFunc<nullptr, nullptr, nullptr>;
    using InternalB1Callback = MatmulCallBackFunc<nullptr, nullptr, CopyB1ByBlockAlignedParity>;
    using MatmulCallback = typename Conditional<INTERNAL_B1_PARITY, InternalB1Callback, DefaultCallback>::type;
    matmul::Matmul<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, QBM_CFG, MatmulCallback> mm;

    __aicore__ inline QuantBatchMatMulV3Matmul() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData,
        bool useDualAiv = false)
    {
        useDualAiv_ = useDualAiv;
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx()) * (useDualAiv_ ? 2U : 1U) +
            (useDualAiv_ ? static_cast<uint32_t>(GetSubBlockIdx()) : 0U);
        pipe_ = pipe;
        tilingData_ = tilingData;
        if (!useDualAiv_ && GetSubBlockIdx() > 0) {
            return;
        }
        m_ = static_cast<uint32_t>(tilingData->m);
        n_ = static_cast<uint32_t>(tilingData->n);
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        vectorLargeM_ = VECTOR_LARGE_M && TRANS_X1 && !TRANS_X2 && m_ > MATMUL_M_LIMIT &&
            n_ <= THIN_M_LIMIT && k_ <= LARGE_N_SMALL_K_LIMIT;
        singleCoreM_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreM);
        singleCoreN_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreN);
        singleCoreK_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreK);
        baseM_ = static_cast<uint32_t>(tilingData->matmulTiling.baseM);
        baseN_ = static_cast<uint32_t>(tilingData->matmulTiling.baseN);
        if constexpr (INTERNAL_B1_PARITY) {
            constexpr uint32_t BASE_K_SHIFT = 24U;
            constexpr uint32_t BASE_N_SHIFT = 40U;
            const uint64_t callbackInfo = static_cast<uint64_t>(n_) |
                (static_cast<uint64_t>(tilingData->matmulTiling.baseK) << BASE_K_SHIFT) |
                (static_cast<uint64_t>(baseN_) << BASE_N_SHIFT);
            mm.SetUserDefInfo(callbackInfo);
        }
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        isMOuter_ = tilingData->matmulTiling.iterateOrder == 0;
        workspacePerCore_ = tilingData->workspacePerCore;

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1),
            static_cast<uint64_t>(tilingData->x1Dim0) * static_cast<uint64_t>(tilingData->x1Dim1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2),
            static_cast<uint64_t>(tilingData->x2Dim0) * static_cast<uint64_t>(tilingData->x2Dim1));
        // Cache-policy leaves: retain a repeatedly consumed small operand; bypass L2 for a one-pass large weight.
        if (NeedTransBLargeKSmallNOriginalStride()) {
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        if (NeedRowMajorMidLargeKOriginalStride()) {
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        if (NeedRowMajorSmallNOriginalStride()) {
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        if (NeedSingleColumnVector()) {
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        if (NeedLargeMSmallNOriginalStride() || NeedLargeMMidNOriginalStride()) {
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        if (NeedLargeMTinyNOriginalStride()) {
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        if (NeedTransBLargeNOriginalStride()) {
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        scaleScalar_ = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(m_) * n_);

        if (singleCoreK_ == 0) {
            singleCoreK_ = k_;
        }
        if (blockIdx_ < usedCoreNum_) {
            GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
            const uint64_t normalWorkspace =
                AlignUp64(static_cast<uint64_t>(Max(baseM_, singleCoreM_)) * Max(baseN_, singleCoreN_) *
                    sizeof(int32_t), WORKSPACE_ALIGN);
            matmulWorkspacePerCore_ = normalWorkspace;
            cWorkspacePerCore_ = matmulWorkspacePerCore_;
            aCompactStride_ = AlignUp(baseM_, GM_BLOCK_BYTES);
            if constexpr (!VECTOR_LARGE_M) {
                if (!NeedSingleColumnVector()) {
                    mm.SetWorkspace(coreWorkspace, matmulWorkspacePerCore_);
                }
            }
            mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coreWorkspace + matmulWorkspacePerCore_),
                cWorkspacePerCore_ / sizeof(int32_t));
            if constexpr (NO_INTRINSICS) {
                uint64_t stagedABytes = 0;
                uint64_t stagedBBytes = 0;
                if constexpr (TRANS_X1) {
                    // LargeM leaves: stage A [K, singleCoreM] compact so that orgM stays below the 65536 limit.
                    stagedABytes = AlignUp64(static_cast<uint64_t>(AlignUp(singleCoreM_, GM_BLOCK_BYTES)) *
                        AlignUp(k_, GM_BLOCK_BYTES), WORKSPACE_ALIGN);
                } else if constexpr (TRANS_X2) {
                    stagedABytes =
                        AlignUp64(static_cast<uint64_t>(singleCoreM_) * TRANS_B_COMPACT_K_CHUNK, WORKSPACE_ALIGN);
                    stagedBBytes =
                        AlignUp64(static_cast<uint64_t>(singleCoreN_) * TRANS_B_COMPACT_K_CHUNK, WORKSPACE_ALIGN);
                } else {
                    // LargeN MediumM leaf: stage B [K, tileN] compact so that orgN stays below the limit.
                    // The staged pitch must cover the widest tile, which is max(singleCoreN_, baseN_).
                    stagedBBytes = AlignUp64(static_cast<uint64_t>(AlignUp(k_, GM_BLOCK_BYTES)) *
                        AlignUp(Max(singleCoreN_, baseN_), GM_BLOCK_BYTES), WORKSPACE_ALIGN) *
                        (UseLargeNStagedPrefetch() ? 2UL : 1UL);
                }
                aCompactGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int8_t*>(coreWorkspace + matmulWorkspacePerCore_ + cWorkspacePerCore_),
                    stagedABytes == 0 ? GM_BLOCK_BYTES : stagedABytes);
                bCompactGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int8_t*>(coreWorkspace + matmulWorkspacePerCore_ + cWorkspacePerCore_ +
                        stagedABytes),
                    stagedBBytes == 0 ? GM_BLOCK_BYTES : stagedBBytes);
            } else {
                const uint32_t compactKStep = NeedRowMajorSmallNOriginalStride() ? singleCoreK_ :
                    (NeedRowMajorMidNStreamK() ? AlignUp(CeilDiv(k_, usedCoreNum_), GM_BLOCK_BYTES) :
                        (NeedRowMajorTallMStreamK() ? ROW_MAJOR_TALL_M_A_COMPACT_K_STEP : A_COMPACT_K_STEP));
                const uint32_t compactRows = NeedRowMajorSmallNOriginalStride() ? singleCoreM_ : aCompactStride_;
                const uint64_t aCompactBytes = static_cast<uint64_t>(compactKStep) * compactRows *
                    (NeedLargeMMidNOriginalStride() ? 2UL : 1UL);
                aCompactGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int8_t*>(coreWorkspace + matmulWorkspacePerCore_ + cWorkspacePerCore_),
                    aCompactBytes);
                if (NeedBothTransFullKCompactB()) {
                    const uint32_t compactBKStride = AlignUp(CeilDiv(k_, usedCoreNum_), GM_BLOCK_BYTES);
                    bCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(
                        coreWorkspace + matmulWorkspacePerCore_ + cWorkspacePerCore_ + aCompactBytes),
                        static_cast<uint64_t>(n_) * compactBKStride);
                }
            }
        }
        if (NeedRowMajorStreamK()) {
            partialGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(userWorkspace + workspacePerCore_ * usedCoreNum_),
                static_cast<uint64_t>(usedCoreNum_) * m_ * n_);
            int32PartialGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(userWorkspace + workspacePerCore_ * usedCoreNum_),
                static_cast<uint64_t>(usedCoreNum_) * m_ * n_);
        }
        ubNAligned_ = AlignUp(NeedRowMajorStreamK() ? n_ : baseN_, BF16_BLOCK_ELEMS);
        if (pipe_ != nullptr && blockIdx_ < usedCoreNum_ && ubNAligned_ > 0) {
            const bool singleColumnVector = !TRANS_X1 && !TRANS_X2 && n_ <= 1U && k_ > MATMUL_K_LIMIT;
            const bool largeKTransBVector = !TRANS_X1 && TRANS_X2 && k_ > MATMUL_K_LIMIT &&
                m_ <= LARGE_K_TRANS_B_VECTOR_M_LIMIT && n_ <= LARGE_K_TRANS_B_VECTOR_N_LIMIT;
            const bool largeMVector = TRANS_X1 && m_ > MATMUL_M_LIMIT && k_ <= LARGE_M_VECTOR_K_LIMIT &&
                n_ <= LARGE_M_VECTOR_N_LIMIT;
            const bool directOriginalStride = NeedRowMajorMidLargeKOriginalStride() ||
                NeedLargeMTinyNOriginalStride() || NeedLargeNThinMOriginalStride() ||
                NeedLargeMSmallNOriginalStride() || NeedTransBLargeNOriginalStride() || NeedLargeMMidNOriginalStride() ||
                NeedLargeNMediumMOriginalStride() || NeedTransBMidLargeKSmallNDirect();
            const uint32_t vectorKTile = singleColumnVector ? SINGLE_COLUMN_VECTOR_K_TILE :
                (largeKTransBVector ? LARGE_K_TRANS_B_VECTOR_K_TILE : 0U);
            const uint32_t compactCopyRows = VECTOR_LARGE_M ? 1U :
                ((NeedLargeMSmallNOriginalStride() || NeedLargeMMidNOriginalStride()) ? 192U : A_COMPACT_COPY_ROWS);
            const uint32_t directElems = directOriginalStride ? SelectDirectAicRowStep() * ubNAligned_ : 0U;
            const uint32_t streamKReduceElems = NeedRowMajorStreamK() ?
                STREAMK_REDUCE_ROWS * ubNAligned_ :
                0U;
            const uint32_t streamKLoadElems = NeedRowMajorStreamK() ?
                STREAMK_PARTIAL_LOAD_ROWS * ubNAligned_ :
                0U;
            const uint32_t smallTilePartialElems =
                (NeedTransBLargeKSmallNOriginalStride() || NeedRowMajorSmallNOriginalStride() ||
                    NeedBothTransLargeKMidNOriginalStride()) ?
                singleCoreM_ * singleCoreN_ : 0U;
            const uint32_t srcElems = Max(Max(Max(ubNAligned_, AlignUp(baseM_, FP32_BLOCK_ELEMS)), vectorKTile),
                Max(Max(Max(directElems, streamKReduceElems), streamKLoadElems), smallTilePartialElems));
            // Two 32 KiB A rows fit in one 64 KiB staging tensor, halving the
            // MTE2->MTE3->MTE2 fence count without changing GM pitches or ownership.
            const uint32_t streamKCompactARow =
                (NeedRowMajorSmallNOriginalStride() || NeedRowMajorMidNStreamK() || NeedBothTransFullKCompactB()) ?
                STREAMK_COMPACT_A_STAGE_BYTES : 0U;
            const uint32_t int8Elems = Max(Max(Max(Max(compactCopyRows * AlignUp(baseM_, GM_BLOCK_BYTES),
                AlignUp(baseN_, GM_BLOCK_BYTES)), AlignUp(vectorKTile, GM_BLOCK_BYTES)),
                (NO_INTRINSICS && !TRANS_X1 && TRANS_X2) ? TRANS_B_COMPACT_K_CHUNK : 0U),
                streamKCompactARow);
            pipe_->InitBuffer(vecQueA_, NeedBothTransFullKCompactB() ? 1U : BUFFER_NUM,
                int8Elems * sizeof(int8_t));
            vecQueABytes_ = int8Elems * sizeof(int8_t);
            const bool largeMTransBCache = TRANS_X2 && largeMVector &&
                n_ * k_ <= LARGE_M_TRANS_B_CACHE_ELEMS;
            const uint32_t srcBytes = Max(srcElems * static_cast<uint32_t>(sizeof(int32_t)),
                largeMTransBCache ? LARGE_M_TRANS_B_CACHE_ELEMS * static_cast<uint32_t>(sizeof(int8_t)) : 0U);
            if (singleColumnVector) {
                pipe_->InitBuffer(vecQueSingleColumnSrc_, SINGLE_COLUMN_SRC_BUFFER_NUM,
                    int8Elems * SINGLE_COLUMN_REDUCE_ROWS * sizeof(int8_t));
            } else {
                const bool prefetchStreamKReduce = NeedRowMajorSmallMStreamK() || NeedRowMajorTallMStreamK() ||
                    NeedTransALargeKStreamK() || NeedBothTransLargeKMidNOriginalStride();
                pipe_->InitBuffer(vecQueSrc_, prefetchStreamKReduce ? STREAMK_SRC_BUFFER_NUM : BUFFER_NUM, srcBytes);
            }
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubNAligned_ * sizeof(float));
            uint32_t outElems = (VECTOR_LARGE_M || largeMVector) ?
                Max(ubNAligned_, AlignUp(baseM_, BF16_BLOCK_ELEMS)) : ubNAligned_;
            if (singleColumnVector) {
                outElems = Max(outElems, AlignUp(singleCoreM_, BF16_BLOCK_ELEMS));
            }
            if (largeMVector && n_ >= 64U) {
                outElems = Max(outElems, LARGE_M_BLOCK16_OUT_ELEMS);
            }
            if (directOriginalStride) {
                outElems = Max(outElems, directElems);
            }
            const uint32_t streamKPartialBytes = STREAMK_PARTIAL_STORE_ROWS * ubNAligned_ *
                static_cast<uint32_t>(sizeof(float));
            const uint32_t outBytes = NeedRowMajorStreamK() ?
                Max(outElems * static_cast<uint32_t>(sizeof(bfloat16_t)), streamKPartialBytes) :
                outElems * static_cast<uint32_t>(sizeof(bfloat16_t));
            pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, outBytes);
            const uint32_t fp32TmpElems = singleColumnVector ?
                Max(ubNAligned_, AlignUp(singleCoreM_, FP32_BLOCK_ELEMS)) : srcElems;
            pipe_->InitBuffer(vecFp32Tmp_, fp32TmpElems * sizeof(float));
            const uint32_t halfTmpElems = singleColumnVector ?
                vectorKTile * SINGLE_COLUMN_PIPELINE_ROWS :
                Max(Max(AlignUp(baseM_, BF16_BLOCK_ELEMS), ubNAligned_), vectorKTile);
            pipe_->InitBuffer(vecHalfTmp_, halfTmpElems * sizeof(half));
            if (!TRANS_X2 && n_ > MATMUL_N_LIMIT && k_ <= LARGE_N_VECTOR_K_LIMIT) {
                pipe_->InitBuffer(vecLargeNB_, ubNAligned_ * sizeof(float));
            }
            if constexpr (TRANS_X1) {
                pipe_->InitBuffer(vecTokenScale_, AlignUp(baseM_, FP32_BLOCK_ELEMS) * sizeof(float));
            } else {
                if (directOriginalStride || NeedRowMajorStreamK()) {
                    pipe_->InitBuffer(vecTokenScale_, AlignUp(baseM_, FP32_BLOCK_ELEMS) * sizeof(float));
                }
            }
            const uint32_t tokenBroadcastElems =
                directOriginalStride ?
                AlignUp(SelectDirectAicRowStep(), FP32_BLOCK_ELEMS) * FP32_BLOCK_ELEMS : FP32_BLOCK_ELEMS;
            uint64_t accumElems = directOriginalStride ?
                tokenBroadcastElems + ubNAligned_ :
                static_cast<uint64_t>(AlignUp(singleColumnVector ? singleCoreM_ : baseM_, FP32_BLOCK_ELEMS)) *
                    ubNAligned_;
            if (!directOriginalStride && vectorKTile > 0U) {
                const uint32_t reduceOutElems = singleColumnVector ?
                    AlignUp(singleCoreM_, FP32_BLOCK_ELEMS) * FP32_BLOCK_ELEMS : FP32_BLOCK_ELEMS;
                accumElems += static_cast<uint64_t>(vectorKTile) *
                    (SINGLE_COLUMN_REDUCE_ROWS + 1U) + reduceOutElems;
            }
            if (!directOriginalStride && largeMVector) {
                const uint32_t largeMExtraVecs = n_ >= 64U ? 3U : 1U;
                accumElems += static_cast<uint64_t>(AlignUp(baseM_, FP32_BLOCK_ELEMS)) * largeMExtraVecs;
            }
            if (!directOriginalStride && NeedRowMajorStreamK()) {
                accumElems += STREAMK_TALL_M_REDUCE_ROWS * FP32_BLOCK_ELEMS;
                // D20: the staged-chunk slab path accumulates a whole singleCoreM_-high tile in
                // UB, which can exceed the reduce-oriented AlignUp(baseM_) sizing.
                // Keep the larger slab allocation restricted to the Tall-M Stream-K leaf.
                if (NeedRowMajorTallMStreamK()) {
                    const uint64_t slabAccumElems =
                        static_cast<uint64_t>(Max(singleCoreM_, baseM_)) * ubNAligned_;
                    if (slabAccumElems > accumElems) {
                        accumElems = slabAccumElems;
                    }
                }
            }
            pipe_->InitBuffer(vecAccum_, accumElems * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || (!useDualAiv_ && GetSubBlockIdx() > 0) || m_ == 0 || n_ == 0 || k_ == 0 ||
            singleCoreM_ == 0 || singleCoreN_ == 0 || baseM_ == 0 || baseN_ == 0 || scaleLen_ == 0) {
            return;
        }
        if (NeedRowMajorStreamK()) {
            if (NeedTransALargeKStreamK()) {
                ProcessRowMajorStreamK<true, false, false>();
            } else if (NeedRowMajorSingleRowStreamK()) {
                ProcessRowMajorStreamK<false, true, false>();
            } else if (NeedRowMajorMidNStreamK()) {
                ProcessRowMajorStreamK<false, false, true>();
            } else {
                ProcessRowMajorStreamK<false, false, false>();
            }
            return;
        }
        if constexpr (VECTOR_LARGE_M) {
            if (!vectorLargeM_) {
                return;
            }
        }

        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        if (mBlocks == 0 || nBlocks == 0 || static_cast<uint64_t>(blockIdx_) >= totalBlocks) {
            return;
        }

        if (NeedContiguousTaskSchedule()) {
            ProcessContiguousTasks(totalBlocks, mBlocks);
            return;
        }
        for (uint64_t taskIdx = blockIdx_; taskIdx < totalBlocks; taskIdx += usedCoreNum_) {
            ProcessBlock(taskIdx, mBlocks);
        }
    }

    __aicore__ inline void ProcessBothTransLargeKMidN()
    {
        if (blockIdx_ >= usedCoreNum_ || (!useDualAiv_ && GetSubBlockIdx() > 0) || m_ == 0 || n_ == 0 || k_ == 0 ||
            singleCoreM_ == 0 || singleCoreN_ == 0 || baseM_ == 0 || baseN_ == 0 || scaleLen_ == 0) {
            return;
        }
        ProcessRowMajorStreamK<false, false, false, true>();
    }

    __aicore__ inline void ProcessRowMajorMidNInt32Public()
    {
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0 || m_ == 0U || n_ == 0U || k_ == 0U ||
            singleCoreM_ == 0U || singleCoreN_ == 0U || baseM_ == 0U || baseN_ == 0U || scaleLen_ == 0U ||
            !NeedRowMajorMidNStreamK() || (m_ % 16U) != 0U || (n_ % 16U) != 0U ||
            (k_ % GM_BLOCK_BYTES) != 0U || singleCoreM_ < m_ || singleCoreN_ < n_) {
            return;
        }
        ProcessRowMajorStreamK<false, false, true, false, true>();
    }

    __aicore__ inline void ProcessRowMajorSingleRowInt32Public()
    {
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0 || m_ == 0U || n_ == 0U || k_ == 0U ||
            singleCoreM_ == 0U || singleCoreN_ == 0U || baseM_ == 0U || baseN_ == 0U || scaleLen_ != n_ ||
            !NeedRowMajorSingleRowStreamK() || (n_ % STREAMK_MID_N_LIMIT) != 0U ||
            (k_ % GM_BLOCK_BYTES) != 0U) {
            return;
        }
        ProcessRowMajorStreamK<false, true, false, false, true>();
    }

    __aicore__ inline void ProcessSingleColumnVector()
    {
        if (blockIdx_ >= usedCoreNum_ || (!useDualAiv_ && GetSubBlockIdx() > 0) || m_ == 0 || n_ != 1U ||
            k_ == 0 || singleCoreM_ == 0 || baseM_ == 0 || baseN_ == 0 || scaleLen_ == 0) {
            return;
        }
        const uint32_t mOffset = blockIdx_ * singleCoreM_;
        if (mOffset >= m_) {
            return;
        }
        const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
        LocalTensor<float> accumLocal = vecAccum_.Get<float>();
        Duplicate(accumLocal, 0.0f, curM * ubNAligned_);
        PipeBarrier<PIPE_V>();
        ProcessSingleColumnVectorTile(accumLocal, mOffset, 0U, curM, 1U);
    }

private:
    __aicore__ inline void ProcessBlock(uint64_t taskIdx, uint32_t mBlocks)
    {
        const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
        const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
        const uint32_t mOffset = mBlockIdx * singleCoreM_;
        const uint32_t nOffset = nBlockIdx * singleCoreN_;
        const uint32_t singleM = Min(singleCoreM_, m_ - mOffset);
        const uint32_t singleN = Min(singleCoreN_, n_ - nOffset);

        const bool compactK = NeedCompactK();
        const bool caseOriginalStride = NeedRowMajorMidLargeKOriginalStride() || NeedRowMajorSmallNOriginalStride() ||
            NeedRowMajorSmallMStreamK() || NeedRowMajorTallMStreamK() || NeedTransBLargeKSmallNOriginalStride() ||
            NeedTransBLargeNOriginalStride() || NeedLargeNMediumMOriginalStride();
        uint32_t tileM = NeedSingleColumnVector() ? singleCoreM_ :
            (caseOriginalStride ? singleCoreM_ : (compactK && !TRANS_X1 ? 1U : baseM_));
        uint32_t tileN = (NeedTransBLargeKSmallNOriginalStride() || NeedBothTransLargeKMidNOriginalStride()) ? singleCoreN_ :
                (compactK && TRANS_X2 && !NeedLargeKTransBColumnTile() ? 1U : baseN_);
        if constexpr (NO_INTRINSICS) {
            // Staged A buffer is sized by singleCoreM_; never exceed it inside one tile.
            tileM = singleCoreM_;
        }
        const uint32_t mTileNum = CeilDiv(singleM, tileM);
        const uint32_t nTileNum = CeilDiv(singleN, tileN);
        if (UseLargeNStagedPrefetch()) {
            ProcessBlockLargeNStagedPipelined(mOffset, nOffset, singleM, singleN, tileN);
            return;
        }
        if (NeedLargeMMidNOriginalStride()) {
            // D20: pipeline staged-A across sub-tiles so chunk DMA overlaps the in-flight matmul.
            ProcessBlockLargeMStagedPipelined(mOffset, nOffset, singleM, singleN, tileM, tileN,
                mTileNum, nTileNum);
            return;
        }
        if (isMOuter_) {
            for (uint32_t mTile = 0; mTile < mTileNum; ++mTile) {
                const uint32_t curM = Min(tileM, singleM - mTile * tileM);
                for (uint32_t nTile = 0; nTile < nTileNum; ++nTile) {
                    const uint32_t curN = Min(tileN, singleN - nTile * tileN);
                    ProcessBaseTile(mOffset + mTile * tileM, nOffset + nTile * tileN, curM, curN);
                }
            }
        } else {
            for (uint32_t nTile = 0; nTile < nTileNum; ++nTile) {
                const uint32_t curN = Min(tileN, singleN - nTile * tileN);
                for (uint32_t mTile = 0; mTile < mTileNum; ++mTile) {
                    const uint32_t curM = Min(tileM, singleM - mTile * tileM);
                    ProcessBaseTile(mOffset + mTile * tileM, nOffset + nTile * tileN, curM, curN);
                }
            }
        }
        if constexpr (A_IBSHARE) {
            // Shared A spans every N tile owned by this task.
            mm.End();
        }
    }

    // Mode12 full-width tasks: overlap staging tile n+1 with cube work for tile n.
    __aicore__ inline void ProcessBlockLargeNStagedPipelined(
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t singleM,
        uint32_t singleN,
        uint32_t tileN)
    {
        const uint32_t nTileNum = CeilDiv(singleN, tileN);
        const uint32_t bStride = AlignUp(Max(singleCoreN_, baseN_), GM_BLOCK_BYTES);
        const uint64_t regionBytes = AlignUp64(
            static_cast<uint64_t>(AlignUp(k_, GM_BLOCK_BYTES)) * bStride, WORKSPACE_ALIGN);

        StageCompactOperand(bCompactGm_, x2Gm_, GetBOffset(nOffset, 0), k_,
            Min(tileN, singleN) * sizeof(int8_t), bStride, n_);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t nTile = 0; nTile < nTileNum; ++nTile) {
            const uint32_t gnOff = nOffset + nTile * tileN;
            const uint32_t curN = Min(tileN, singleN - nTile * tileN);
            GlobalTensor<int8_t> stagedB = bCompactGm_[(nTile % 2U) * regionBytes];

            mm.SetOrgShape(singleM, bStride, k_, k_, curN);
            mm.SetTail(singleM, curN, k_);
            mm.SetTensorA(x1Gm_[GetAOffset(mOffset, 0)], false);
            mm.SetTensorB(stagedB, false);
            mm.template Iterate<false>();

            if (nTile + 1U < nTileNum) {
                const uint32_t next = nTile + 1U;
                const uint32_t nextN = Min(tileN, singleN - next * tileN);
                GlobalTensor<int8_t> nextB = bCompactGm_[(next % 2U) * regionBytes];
                StageCompactOperand(nextB, x2Gm_, GetBOffset(nOffset + next * tileN, 0), k_,
                    nextN * sizeof(int8_t), bStride, n_);
            }

            auto mmOutGm = mm.GetTensorC();
            PipeBarrier<PIPE_ALL>();
            ScaleMatmulOutAndStore(mmOutGm, mOffset, gnOff, singleM, curN);
            if constexpr (!A_IBSHARE) {
                mm.End();
            }
        }
        if constexpr (A_IBSHARE) {
            mm.End();
        }
    }

    // D20: MidN Large-M only — stage sub-tile s+1 while matmul(s) runs, using two parity regions
    // of the (host-doubled) compact-A workspace. Sub-tiles are visited n-outer/m-inner; output
    // ownership and per-tile post-processing are identical to the generic path.
    __aicore__ inline void ProcessBlockLargeMStagedPipelined(
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t singleM,
        uint32_t singleN,
        uint32_t tileM,
        uint32_t tileN,
        uint32_t mTileNum,
        uint32_t nTileNum)
    {
        const uint32_t total = mTileNum * nTileNum;
        const uint64_t regionBytes = static_cast<uint64_t>(A_COMPACT_K_STEP) * aCompactStride_;
        for (uint32_t s = 0; s < total; ++s) {
            const uint32_t mTile = s % mTileNum;
            const uint32_t nTile = s / mTileNum;
            const uint32_t gmOff = mOffset + mTile * tileM;
            const uint32_t gnOff = nOffset + nTile * tileN;
            const uint32_t curM = Min(tileM, singleM - mTile * tileM);
            const uint32_t curN = Min(tileN, singleN - nTile * tileN);
            if (s == 0) {
                StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(gmOff, 0), k_, curM,
                    aCompactStride_, m_);
                PipeBarrier<PIPE_ALL>();
            }
            mm.SetOrgShape(aCompactStride_, n_, k_, k_, n_);
            mm.SetTail(curM, curN, k_);
            GlobalTensor<int8_t> stagedA = aCompactGm_[(s % 2U) * regionBytes];
            mm.SetTensorA(stagedA, TRANS_X1);
            mm.SetTensorB(x2Gm_[GetBOffset(gnOff, 0U)], TRANS_X2);
            mm.template Iterate<false>();
            if (s + 1U < total) {
                const uint32_t sNext = s + 1U;
                const uint32_t mTileNext = sNext % mTileNum;
                const uint32_t curMNext = Min(tileM, singleM - mTileNext * tileM);
                GlobalTensor<int8_t> dstNext = aCompactGm_[(sNext % 2U) * regionBytes];
                StageCompactOperand(dstNext, x1Gm_, GetAOffset(mOffset + mTileNext * tileM, 0), k_,
                    curMNext, aCompactStride_, m_);
            }
            auto mmOutGm = mm.GetTensorC();
            PipeBarrier<PIPE_ALL>();
            ScaleMatmulOutAndStore(mmOutGm, gmOff, gnOff, curM, curN);
            mm.End();
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) const
    {
        return (value + factor - 1U) / factor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline bool NeedCompactK() const
    {
        return k_ > MATMUL_K_LIMIT;
    }

    __aicore__ inline bool NeedCompactN() const
    {
        return n_ > MATMUL_N_LIMIT;
    }

    __aicore__ inline bool NeedCompactNUnitK() const
    {
        return !TRANS_X2 && n_ > MATMUL_N_LIMIT;
    }

    __aicore__ inline bool NeedCompactM() const
    {
        return TRANS_X1 && m_ > MATMUL_M_LIMIT;
    }

    __aicore__ inline bool NeedRowMajorMidLargeKOriginalStride() const
    {
        // O18: the whole Mid-N Large-K band now routes to Stream-K; keep the direct
        // original-stride leaf disabled so no call site can pick the slow full-K path.
        return false;
    }

    __aicore__ inline bool NeedRowMajorMidNStreamK() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && k_ <= LARGE_K_MID_LIMIT && m_ > STREAMK_SMALL_M_LIMIT &&
                m_ <= MEDIUM_M_LIMIT && n_ > SMALL_N_LIMIT && n_ <= MEDIUM_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedRowMajorSmallNOriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ > STREAMK_SMALL_M_LIMIT && m_ <= MEDIUM_M_LIMIT &&
                n_ <= SMALL_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedRowMajorSingleRowStreamK() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ <= 1U && n_ > MEDIUM_N_LIMIT && n_ <= STREAMK_MID_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedRowMajorSmallMStreamK() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && k_ <= LARGE_K_MID_LIMIT && m_ > 1U &&
                m_ <= STREAMK_SMALL_M_LIMIT && n_ > MEDIUM_N_LIMIT && n_ <= STREAMK_MID_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedRowMajorTallMStreamK() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ > MEDIUM_M_LIMIT && m_ <= TALL_M_LIMIT &&
                n_ > 1U && n_ <= SMALL_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedTransALargeKStreamK() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ <= MEDIUM_M_LIMIT && n_ <= SMALL_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedRowMajorLargeKStreamK() const
    {
        return NeedRowMajorSingleRowStreamK() || NeedRowMajorSmallMStreamK() || NeedRowMajorTallMStreamK() ||
            NeedRowMajorMidNStreamK();
    }

    __aicore__ inline bool NeedRowMajorStreamK() const
    {
        return NeedTransALargeKStreamK() || NeedRowMajorLargeKStreamK() ||
            NeedRowMajorSmallNOriginalStride() || NeedBothTransLargeKMidNOriginalStride();
    }

    __aicore__ inline bool NeedTransBLargeKSmallNOriginalStride() const
    {
        if constexpr (!TRANS_X1 && TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ <= MEDIUM_M_LIMIT && n_ <= SMALL_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedTransBMidLargeKSmallNDirect() const
    {
        if constexpr (!TRANS_X1 && TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && k_ <= LARGE_K_MID_LIMIT && m_ > 32U && m_ <= MEDIUM_M_LIMIT &&
                n_ > 32U && n_ <= SMALL_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedBothTransLargeKMidNOriginalStride() const
    {
        if constexpr (TRANS_X1 && TRANS_X2) {
            return k_ > MATMUL_K_LIMIT && m_ <= MEDIUM_M_LIMIT && n_ > SMALL_N_LIMIT &&
                n_ <= MEDIUM_N_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedBothTransFullKCompactB() const
    {
        if (!NeedBothTransLargeKMidNOriginalStride() || usedCoreNum_ == 0U) {
            return false;
        }
        const uint64_t compactKStride = AlignUp(CeilDiv(k_, usedCoreNum_), GM_BLOCK_BYTES);
        return static_cast<uint64_t>(n_) * compactKStride <= BOTH_TRANS_COMPACT_B_BYTES_LIMIT;
    }

    __aicore__ inline bool NeedLargeNMediumMOriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return n_ > MATMUL_N_LIMIT && m_ > THIN_M_LIMIT && m_ <= MEDIUM_M_LIMIT &&
                k_ <= LARGE_N_VECTOR_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool UseLargeNStagedPrefetch() const
    {
        return NeedLargeNMediumMOriginalStride() && singleCoreN_ != 0U && n_ % singleCoreN_ == 0U;
    }

    __aicore__ inline bool NeedLargeNThinMOriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return n_ > MATMUL_N_LIMIT && m_ <= THIN_M_LIMIT && k_ <= LARGE_N_SMALL_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedLargeMMidNOriginalStride() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return m_ > MATMUL_M_LIMIT && n_ > SMALL_N_LIMIT && n_ <= MEDIUM_N_LIMIT &&
                k_ <= LARGE_M_VECTOR_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedLargeMTinyNOriginalStride() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return m_ > MATMUL_M_LIMIT && n_ <= THIN_M_LIMIT && k_ <= LARGE_N_SMALL_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedLargeMSmallNOriginalStride() const
    {
        if constexpr (TRANS_X1 && TRANS_X2) {
            return m_ > MATMUL_M_LIMIT && n_ <= SMALL_N_LIMIT && k_ <= LARGE_M_VECTOR_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedTransBLargeNOriginalStride() const
    {
        if constexpr (!TRANS_X1 && TRANS_X2) {
            return n_ > MATMUL_N_LIMIT && m_ <= MEDIUM_M_LIMIT && k_ <= LARGE_N_VECTOR_K_LIMIT;
        }
        return false;
    }

    __aicore__ inline bool NeedContiguousTaskSchedule() const
    {
        // Scheduling leaf: spatially adjacent tiles stay on one core for selected original-stride paths.
        return NeedLargeMTinyNOriginalStride() || NeedLargeNThinMOriginalStride() ||
            NeedTransBLargeNOriginalStride() || NeedLargeNMediumMOriginalStride();
    }

    __aicore__ inline uint32_t SelectDirectAicRowStep() const
    {
        if constexpr (TRANS_X1) {
            if (NeedLargeMMidNOriginalStride()) {
                return DIRECT_AIC_LARGE_M_ROW_STEP;
            }
            if (NeedLargeMTinyNOriginalStride()) {
                return DIRECT_AIC_LARGE_M_TINY_N_ROW_STEP;
            }
            if (NeedLargeMSmallNOriginalStride()) {
                return DIRECT_AIC_LARGE_M_SMALL_N_ROW_STEP;
            }
        }
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            if (NeedLargeNThinMOriginalStride()) {
                return DIRECT_AIC_LARGE_N_THIN_M_ROW_STEP;
            }
            return NeedLargeNMediumMOriginalStride() ? DIRECT_AIC_LARGE_N_MEDIUM_M_ROW_STEP : DIRECT_AIC_ROW_STEP;
        }
        if constexpr (!TRANS_X1 && TRANS_X2) {
            if (NeedTransBLargeNOriginalStride()) {
                return DIRECT_AIC_TRANS_B_LARGE_N_ROW_STEP;
            }
        }
        return DIRECT_AIC_ROW_STEP;
    }

    __aicore__ inline uint32_t SelectKStep(bool compactNUnitK, bool compactA, bool compactK) const
    {
        if (compactNUnitK) {
            return 1U;
        }
        if (compactA) {
            return A_COMPACT_K_STEP;
        }
        if (compactK) {
            if constexpr (!TRANS_X1 && !TRANS_X2) {
                if (NeedRowMajorMidLargeKOriginalStride()) {
                    return ROW_MAJOR_MID_LARGE_K_CHUNK_LIMIT;
                }
                if (NeedRowMajorSmallNOriginalStride()) {
                    return ROW_MAJOR_SMALL_N_LARGE_K_CHUNK_LIMIT;
                }
            }
            if constexpr (TRANS_X1 && TRANS_X2) {
                if (NeedBothTransLargeKMidNOriginalStride()) {
                    return BOTH_TRANS_LARGE_K_CHUNK_LIMIT;
                }
            }
            return K_CHUNK_LIMIT;
        }
        return k_;
    }

    __aicore__ inline uint64_t GetAOffset(uint32_t mOffset, uint32_t kOffset) const
    {
        return TRANS_X1 ? static_cast<uint64_t>(kOffset) * m_ + mOffset : static_cast<uint64_t>(mOffset) * k_ + kOffset;
    }

    __aicore__ inline uint64_t GetBOffset(uint32_t nOffset, uint32_t kOffset) const
    {
        return TRANS_X2 ? static_cast<uint64_t>(nOffset) * k_ + kOffset : static_cast<uint64_t>(kOffset) * n_ + nOffset;
    }

    __aicore__ inline void ProcessBaseTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const bool compactK = NeedCompactK();
        const bool compactN = NeedCompactN();
        const bool largeNMediumMOriginalStride = NeedLargeNMediumMOriginalStride();
        const bool largeMTinyNOriginalStride = NeedLargeMTinyNOriginalStride();
        const bool largeMMidNOriginalStride = NeedLargeMMidNOriginalStride();
        const bool largeMSmallNOriginalStride = NeedLargeMSmallNOriginalStride();
        const bool transBLargeNOriginalStride = NeedTransBLargeNOriginalStride();
        const bool largeNThinMOriginalStride = NeedLargeNThinMOriginalStride();
        const bool transBMidLargeKSmallNDirect = NeedTransBMidLargeKSmallNDirect();
        if constexpr (NO_INTRINSICS) {
            // Host routes exactly one staged leaf to each NO_INTRINSICS key; run it before any legacy branch.
            ProcessStagedOriginalStrideTile(globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedRowMajorMidLargeKOriginalStride() || largeNMediumMOriginalStride || largeMTinyNOriginalStride ||
            largeMMidNOriginalStride ||
            largeMSmallNOriginalStride || transBLargeNOriginalStride || largeNThinMOriginalStride ||
            transBMidLargeKSmallNDirect) {
            ProcessDirectOriginalStrideAicTile(globalMOffset, globalNOffset, curM, curN);
            return;
        }
        LocalTensor<float> accumLocal = vecAccum_.Get<float>();
        Duplicate(accumLocal, 0.0f, curM * ubNAligned_);
        PipeBarrier<PIPE_V>();

        const bool compactNUnitK = NeedCompactNUnitK() && !largeNMediumMOriginalStride;
        const bool compactM = NeedCompactM() && !largeMMidNOriginalStride && !largeMSmallNOriginalStride;
        const bool compactA = compactM && !compactNUnitK;
        if (NeedLargeMVector() && !largeMMidNOriginalStride && !largeMSmallNOriginalStride) {
            ProcessLargeMVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeNVector()) {
            ProcessLargeNVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedSingleColumnVector()) {
            ProcessSingleColumnVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeKTransBVector() && !NeedTransBLargeKSmallNOriginalStride()) {
            ProcessLargeKTransBVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        const uint32_t kStep = SelectKStep(compactNUnitK, compactA, compactK);
        for (uint32_t kOffset = 0; kOffset < k_; kOffset += kStep) {
            const uint32_t curK = Min(kStep, k_ - kOffset);
            if (compactA) {
                CopyCompactA(globalMOffset, kOffset, curM, curK);
            }
            const uint32_t orgM = (compactA || compactM) ? (compactA ? aCompactStride_ : curM) : m_;
            const uint32_t orgN = largeNMediumMOriginalStride ? n_ : (compactN ? curN : n_);
            const uint32_t orgKa = (NeedRowMajorMidLargeKOriginalStride() || NeedRowMajorSmallNOriginalStride() ||
                NeedRowMajorSmallMStreamK() || NeedRowMajorTallMStreamK() || NeedTransBLargeKSmallNOriginalStride() ||
                largeNMediumMOriginalStride) ?
                k_ : (compactK || compactM ? curK : k_);
            const uint32_t orgKb = (NeedTransBLargeKSmallNOriginalStride() || NeedBothTransLargeKMidNOriginalStride() ||
                largeNMediumMOriginalStride) ?
                k_ : (compactK || compactNUnitK || (compactM && !TRANS_X2) ? curK : k_);
            const uint32_t orgKc = largeNMediumMOriginalStride ? n_ : (compactN ? curN : n_);
            mm.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
            mm.SetTail(curM, curN, curK);
            if (compactA) {
                mm.SetTensorA(aCompactGm_, TRANS_X1);
            } else {
                mm.SetTensorA(x1Gm_[GetAOffset(globalMOffset, kOffset)], TRANS_X1);
            }
            mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, kOffset)], TRANS_X2);
            if (compactA) {
                mm.SetSingleShape(curM, curN, curK);
                mm.Iterate();
                mm.GetTensorC(mmOutGm_[0], 0, true);
                mm.End();
                AccumulatePartial(mmOutGm_, accumLocal, curM, curN);
            } else {
                mm.template Iterate<false>();
                auto mmOutGm = mm.GetTensorC();
                AccumulatePartial(mmOutGm, accumLocal, curM, curN);
                mm.End();
            }
        }

        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessDirectOriginalStrideAicTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        // D20: LargeM transA direct leaves — stage A [K, curM] compact so the matmul org pitch
        // stays below the 65535 slow-path limit; PIPE_ALL orders the MTE3 staging before the cube read.
        const bool largeMDirect = NeedLargeMSmallNOriginalStride() || NeedLargeMMidNOriginalStride();
        if (largeMDirect) {
            const uint32_t aStride = aCompactStride_;
            StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(globalMOffset, 0), k_, curM, aStride, m_);
            PipeBarrier<PIPE_ALL>();
            mm.SetOrgShape(aStride, n_, k_, k_, n_);
            mm.SetTail(curM, curN, k_);
            mm.SetTensorA(aCompactGm_, TRANS_X1);
            mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, 0U)], TRANS_X2);
            mm.template Iterate<false>();
            auto mmOutGm = mm.GetTensorC();
            ScaleMatmulOutAndStore(mmOutGm, globalMOffset, globalNOffset, curM, curN);
            mm.End();
            return;
        }
        mm.SetOrgShape(m_, n_, k_, k_, n_);
        mm.SetTail(curM, curN, k_);
        mm.SetTensorA(x1Gm_[GetAOffset(globalMOffset, 0U)], TRANS_X1);
        mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, 0U)], TRANS_X2);
        mm.template Iterate<false>();
        auto mmOutGm = mm.GetTensorC();
        ScaleMatmulOutAndStore(mmOutGm, globalMOffset, globalNOffset, curM, curN);
        if constexpr (!A_IBSHARE) {
            mm.End();
        }
    }

    __aicore__ inline void ProcessContiguousTasks(uint64_t totalBlocks, uint32_t mBlocks)
    {
        const uint64_t coreCount = static_cast<uint64_t>(usedCoreNum_);
        const uint64_t baseTasks = totalBlocks / coreCount;
        const uint64_t extraTasks = totalBlocks - baseTasks * coreCount;
        const uint64_t blockIdx = static_cast<uint64_t>(blockIdx_);
        const uint64_t taskCount = baseTasks + (blockIdx < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx) {
            ProcessBlock(taskIdx, mBlocks);
        }
    }

    template <bool TRANS_A_SPECIALIZED, bool SINGLE_ROW_SPECIALIZED, bool MID_N_SPECIALIZED,
        bool BOTH_TRANS_FIRST_PARTIAL = false, bool USE_INT32_PUBLIC = false>
    __aicore__ inline void ProcessRowMajorStreamK()
    {
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const bool partitionByM = NeedRowMajorSmallNOriginalStride();
        const uint32_t kGroups = partitionByM ? Max(1U, usedCoreNum_ / mBlocks) : usedCoreNum_;
        const uint32_t kGroupIdx = partitionByM ? blockIdx_ / mBlocks : blockIdx_;
        const uint32_t ownedMBlock = partitionByM ? blockIdx_ % mBlocks : 0U;
        const uint32_t kSlice = AlignUp(CeilDiv(k_, kGroups), GM_BLOCK_BYTES);
        const uint32_t kOffset = kGroupIdx * kSlice;
        if (kOffset < k_) {
            const uint32_t curK = Min(kSlice, k_ - kOffset);
            const uint32_t compactBKStride = NeedBothTransFullKCompactB() ?
                AlignUp(curK, GM_BLOCK_BYTES) : 0U;
            if (NeedBothTransFullKCompactB()) {
                StageCompactOperand(bCompactGm_, x2Gm_, GetBOffset(0U, kOffset),
                    n_, curK * sizeof(int8_t), compactBKStride, k_);
            }
            const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
            const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
            for (uint64_t taskIdx = 0; taskIdx < totalBlocks; ++taskIdx) {
                const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
                if (partitionByM && mBlockIdx != ownedMBlock) {
                    continue;
                }
                const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
                const uint32_t mOffset = mBlockIdx * singleCoreM_;
                const uint32_t nOffset = nBlockIdx * singleCoreN_;
                const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
                const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
                if constexpr (SINGLE_ROW_SPECIALIZED) {
                    mm.SetOrgShape(m_, n_, k_, k_, n_);
                    mm.SetTail(curM, curN, curK);
                    mm.SetTensorA(x1Gm_[GetAOffset(mOffset, kOffset)], false);
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], false);
                    if constexpr (USE_INT32_PUBLIC) {
                        mm.template Iterate<false>();
                        const uint64_t partialOffset =
                            (static_cast<uint64_t>(blockIdx_) * m_ + mOffset) * n_ + nOffset;
                        mm.GetTensorC(int32PartialGm_[partialOffset], 0U, true);
                    } else {
                        mm.template Iterate<false>();
                        auto mmOutGm = mm.GetTensorC();
                        StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                    }
                    mm.End();
                    continue;
                }
                if constexpr (TRANS_A_SPECIALIZED) {
                    mm.SetOrgShape(m_, n_, k_, k_, n_);
                    mm.SetTail(curM, curN, curK);
                    mm.SetTensorA(x1Gm_[GetAOffset(mOffset, kOffset)], true);
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], false);
                    mm.template Iterate<false>();
                    auto mmOutGm = mm.GetTensorC();
                    StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                    mm.End();
                    continue;
                }
                if constexpr (MID_N_SPECIALIZED) {
                    const uint32_t compactKStride = kSlice;
                    if (nBlockIdx == 0U) {
                        StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(mOffset, kOffset),
                            curM, curK * sizeof(int8_t), compactKStride, k_);
                    }
                    PipeBarrier<PIPE_ALL>();
                    mm.SetOrgShape(curM, n_, compactKStride, k_, n_);
                    mm.SetTail(curM, curN, curK);
                    mm.SetTensorA(aCompactGm_, false);
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], false);
                    if constexpr (USE_INT32_PUBLIC) {
                        mm.SetSingleShape(curM, curN, curK);
                        mm.Iterate();
                        const uint64_t partialOffset =
                            (static_cast<uint64_t>(blockIdx_) * m_ + mOffset) * n_ + nOffset;
                        mm.GetTensorC(int32PartialGm_[partialOffset], 0U, true);
                    } else {
                        mm.template Iterate<false>();
                        auto mmOutGm = mm.GetTensorC();
                        StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                    }
                    mm.End();
                    continue;
                }
                if (NeedRowMajorMidNStreamK()) {
                    const uint32_t compactKStride = kSlice;
                    if (nBlockIdx == 0U) {
                        StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(mOffset, kOffset),
                            curM, curK * sizeof(int8_t), compactKStride, k_);
                    }
                    PipeBarrier<PIPE_ALL>();
                    mm.SetOrgShape(curM, n_, compactKStride, k_, n_);
                    mm.SetTail(curM, curN, curK);
                    mm.SetTensorA(aCompactGm_, false);
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], false);
                    mm.template Iterate<false>();
                    auto mmOutGm = mm.GetTensorC();
                    StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                    mm.End();
                    continue;
                }
                if (NeedRowMajorSmallNOriginalStride()) {
                    // The K group is already below the Matmul axis limit. Stage this core's
                    // A slice with a compact pitch, while B keeps its original row-major GM
                    // layout and row ownership/partial reduction remain unchanged.
                    const uint32_t compactKStride = singleCoreK_;
                    StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(mOffset, kOffset),
                        curM, curK * sizeof(int8_t), compactKStride, k_);
                    PipeBarrier<PIPE_ALL>();
                    mm.SetOrgShape(curM, n_, compactKStride, k_, n_);
                    mm.SetTail(curM, curN, curK);
                    mm.SetTensorA(aCompactGm_, false);
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], false);
                    mm.template Iterate<false>();
                    auto mmOutGm = mm.GetTensorC();
                    StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                    mm.End();
                    continue;
                }
                if (NeedRowMajorTallMStreamK()) {
                    // D20: the full-K slab matmul fed orgKa=orgKb=k_ (>65535) to the MDL slow loop;
                    // stage A in 1024-deep K chunks so every org dim stays legal, accumulate chunk
                    // partials in UB, and store the slab partial once.
                    // Tall-M-only gate; B stays global so orgKb must be the full k_ extent.
                    if (curK <= ROW_MAJOR_TALL_M_A_COMPACT_K_STEP) {
                        const uint32_t aPitch = AlignUp(curK, GM_BLOCK_BYTES);
                        StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(mOffset, kOffset),
                            curM, curK * sizeof(int8_t), aPitch, k_);
                        PipeBarrier<PIPE_ALL>();
                        mm.SetOrgShape(curM, n_, aPitch, k_, n_);
                        mm.SetTail(curM, curN, curK);
                        mm.SetTensorA(aCompactGm_, TRANS_X1);
                        mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], TRANS_X2);
                        mm.template Iterate<false>();
                        auto mmOutGm = mm.GetTensorC();
                        StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                        mm.End();
                        continue;
                    }
                    LocalTensor<float> accumLocal = vecAccum_.Get<float>();
                    Duplicate(accumLocal, 0.0f, curM * ubNAligned_);
                    PipeBarrier<PIPE_V>();
                    for (uint32_t kInSlab = 0; kInSlab < curK;
                         kInSlab += ROW_MAJOR_TALL_M_A_COMPACT_K_STEP) {
                        const uint32_t curKC = Min(ROW_MAJOR_TALL_M_A_COMPACT_K_STEP, curK - kInSlab);
                        const uint32_t aPitch = AlignUp(curKC, GM_BLOCK_BYTES);
                        StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(mOffset, kOffset + kInSlab),
                            curM, curKC * sizeof(int8_t), aPitch, k_);
                        PipeBarrier<PIPE_ALL>();
                        mm.SetOrgShape(curM, n_, aPitch, k_, n_);
                        mm.SetTail(curM, curN, curKC);
                        mm.SetTensorA(aCompactGm_, TRANS_X1);
                        mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset + kInSlab)], TRANS_X2);
                        mm.template Iterate<false>();
                        auto mmChunkGm = mm.GetTensorC();
                        AccumulatePartial(mmChunkGm, accumLocal, curM, curN);
                        mm.End();
                        // The next chunk reuses the same staged-A GM slab.  Keep that MTE3 write
                        // behind the completed Cube/L1 read of the current slab.
                        PipeBarrier<PIPE_ALL>();
                    }
                    StoreAccumStreamKPartial(accumLocal, mOffset, nOffset, curM, curN);
                    continue;
                }
                mm.SetOrgShape(m_, n_, k_, NeedBothTransFullKCompactB() ? compactBKStride : k_, n_);
                mm.SetTail(curM, curN, curK);
                mm.SetTensorA(x1Gm_[GetAOffset(mOffset, kOffset)], TRANS_X1);
                if (NeedBothTransFullKCompactB()) {
                    mm.SetTensorB(bCompactGm_[static_cast<uint64_t>(nOffset) * compactBKStride], TRANS_X2);
                } else {
                    mm.SetTensorB(x2Gm_[GetBOffset(nOffset, kOffset)], TRANS_X2);
                }
                mm.template Iterate<false>();
                auto mmOutGm = mm.GetTensorC();
                StoreMatmulOutStreamKPartial(mmOutGm, mOffset, nOffset, curM, curN);
                mm.End();
            }
        }

        CrossCoreSetFlag<STREAMK_SYNC_MODE, PIPE_MTE3>(STREAMK_AIC_SYNC_FLAG);
        CrossCoreWaitFlag(STREAMK_AIC_SYNC_FLAG);
        ReduceStreamKPartialsAndStore<TRANS_A_SPECIALIZED, SINGLE_ROW_SPECIALIZED, MID_N_SPECIALIZED,
            BOTH_TRANS_FIRST_PARTIAL, USE_INT32_PUBLIC>();
    }

    __aicore__ inline void StoreMatmulOutStreamKPartial(
        GlobalTensor<int32_t>& mmOutGm,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const uint64_t partialBase = static_cast<uint64_t>(blockIdx_) * m_ * n_;
        const uint32_t blockBytes = curN * static_cast<uint32_t>(sizeof(float));
        const uint32_t dstStrideBytes = (n_ - curN) * static_cast<uint32_t>(sizeof(float));
        if (blockBytes % GM_BLOCK_BYTES == 0U && dstStrideBytes % GM_BLOCK_BYTES == 0U) {
            for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += STREAMK_PARTIAL_STORE_ROWS) {
                const uint32_t rowsThis = Min(STREAMK_PARTIAL_STORE_ROWS, curM - rowBlock);
                LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
                DataCopyExtParams copyInParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(curN * sizeof(int32_t)),
                    0,
                    0,
                    0};
                DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(rowBlock) * curN], copyInParams, padParams);
                vecQueSrc_.EnQue<int32_t>(srcLocal);

                srcLocal = vecQueSrc_.DeQue<int32_t>();
                LocalTensor<float> partialLocal = vecQueOut_.AllocTensor<float>();
                Cast(partialLocal, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
                PipeBarrier<PIPE_V>();
                vecQueSrc_.FreeTensor<int32_t>(srcLocal);
                vecQueOut_.EnQue<float>(partialLocal);

                partialLocal = vecQueOut_.DeQue<float>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis), blockBytes, 0, dstStrideBytes, 0};
                DataCopyPad(
                    partialGm_[partialBase + static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset],
                    partialLocal, copyOutParams);
                vecQueOut_.FreeTensor<float>(partialLocal);
            }
            return;
        }
        for (uint32_t row = 0; row < curM; ++row) {
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curN * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(row) * curN], copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);

            srcLocal = vecQueSrc_.DeQue<int32_t>();
            LocalTensor<float> partialLocal = vecQueOut_.AllocTensor<float>();
            Cast(partialLocal, srcLocal, RoundMode::CAST_RINT, curN);
            PipeBarrier<PIPE_V>();
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
            vecQueOut_.EnQue<float>(partialLocal);

            partialLocal = vecQueOut_.DeQue<float>();
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPad(partialGm_[partialBase + static_cast<uint64_t>(globalMOffset + row) * n_ + globalNOffset],
                partialLocal, copyOutParams);
            vecQueOut_.FreeTensor<float>(partialLocal);
        }
    }

    // D20: store a UB-accumulated (float) slab partial with the same layout as
    // StoreMatmulOutStreamKPartial; used by the staged-chunk MidN/TallM slab path.
    __aicore__ inline void StoreAccumStreamKPartial(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const uint64_t partialBase = static_cast<uint64_t>(blockIdx_) * m_ * n_;
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += STREAMK_PARTIAL_STORE_ROWS) {
            const uint32_t rowsThis = Min(STREAMK_PARTIAL_STORE_ROWS, curM - rowBlock);
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(rowsThis),
                static_cast<uint32_t>(curN * sizeof(float)),
                static_cast<uint32_t>((ubNAligned_ - curN) * sizeof(float)),
                static_cast<uint32_t>((n_ - curN) * sizeof(float)),
                0};
            DataCopyPad(
                partialGm_[partialBase + static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset],
                accumLocal[static_cast<uint64_t>(rowBlock) * ubNAligned_], copyOutParams);
        }
    }

    __aicore__ inline void EnqueueStreamKPartial(
        uint32_t core,
        uint32_t rowBlock,
        uint32_t nOffset,
        uint32_t rowsThis,
        uint32_t curN)
    {
        LocalTensor<float> srcLocal = vecQueSrc_.AllocTensor<float>();
        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(rowsThis),
            static_cast<uint32_t>(curN * sizeof(float)),
            static_cast<uint32_t>((n_ - curN) * sizeof(float)),
            static_cast<uint32_t>((ubNAligned_ - curN) * sizeof(float) / GM_BLOCK_BYTES),
            0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        const uint64_t partialOffset =
            (static_cast<uint64_t>(core) * m_ + rowBlock) * n_ + nOffset;
        DataCopyPad(srcLocal, partialGm_[partialOffset], copyInParams, padParams);
        vecQueSrc_.EnQue<float>(srcLocal);
    }

    __aicore__ inline void LoadStreamKInt32Partial(
        LocalTensor<float>& fp32Local,
        uint32_t core,
        uint32_t rowBlock,
        uint32_t nOffset,
        uint32_t rowsThis,
        uint32_t curN)
    {
        LocalTensor<int32_t> int32Local = vecQueSrc_.AllocTensor<int32_t>();
        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(rowsThis),
            static_cast<uint32_t>(curN * sizeof(int32_t)),
            static_cast<uint32_t>((n_ - curN) * sizeof(int32_t)),
            static_cast<uint32_t>((ubNAligned_ - curN) * sizeof(int32_t) / GM_BLOCK_BYTES),
            0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        const uint64_t partialOffset =
            (static_cast<uint64_t>(core) * m_ + rowBlock) * n_ + nOffset;
        DataCopyPad(int32Local, int32PartialGm_[partialOffset], copyInParams, padParams);
        vecQueSrc_.EnQue<int32_t>(int32Local);

        int32Local = vecQueSrc_.DeQue<int32_t>();
        fp32Local = vecFp32Tmp_.Get<float>();
        Cast(fp32Local, int32Local, RoundMode::CAST_RINT, rowsThis * ubNAligned_);
        PipeBarrier<PIPE_V>();
        vecQueSrc_.FreeTensor<int32_t>(int32Local);
    }

    template <bool TRANS_A_SPECIALIZED, bool SINGLE_ROW_SPECIALIZED, bool MID_N_SPECIALIZED,
        bool BOTH_TRANS_FIRST_PARTIAL, bool USE_INT32_PUBLIC>
    __aicore__ inline void ReduceStreamKPartialsAndStore()
    {
        const float scaleScalar = scaleScalar_;
        LocalTensor<float> accumLocal = vecAccum_.Get<float>();
        LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
        const uint32_t reduceRows = (TRANS_A_SPECIALIZED || SINGLE_ROW_SPECIALIZED || MID_N_SPECIALIZED) ?
            STREAMK_REDUCE_ROWS :
            (NeedRowMajorSmallMStreamK() ? STREAMK_SMALL_M_REDUCE_ROWS :
                (NeedRowMajorTallMStreamK() ? STREAMK_TALL_M_REDUCE_ROWS : STREAMK_REDUCE_ROWS));
        LocalTensor<float> tokenScaleLocal = vecTokenScale_.Get<float>();
        const uint32_t accumBaseElems = AlignUp(baseM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<float> tokenScaleBrcbLocal = accumLocal[accumBaseElems];
        for (uint32_t nOffset = 0; nOffset < n_; nOffset += baseN_) {
            const uint32_t curN = Min(baseN_, n_ - nOffset);
            LocalTensor<float> scaleLocal;
            if (scaleLen_ != 1U) {
                scaleLocal = vecQueScale_.AllocTensor<float>();
                DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
                DataCopyPad(scaleLocal, scaleGm_[nOffset], scaleCopyParams, scalePadParams);
                vecQueScale_.EnQue<float>(scaleLocal);
                scaleLocal = vecQueScale_.DeQue<float>();
            }

            for (uint32_t rowBlock = blockIdx_ * reduceRows; rowBlock < m_;
                 rowBlock += usedCoreNum_ * reduceRows) {
                const uint32_t rowsThis = Min(reduceRows, m_ - rowBlock);
                const bool partitionByM = (TRANS_A_SPECIALIZED || SINGLE_ROW_SPECIALIZED || MID_N_SPECIALIZED) ?
                    false : NeedRowMajorSmallNOriginalStride();
                const uint32_t sourceMBlocks = partitionByM ? CeilDiv(m_, singleCoreM_) : 1U;
                const uint32_t sourceCoreCount = partitionByM ? usedCoreNum_ / sourceMBlocks : usedCoreNum_;
                const uint32_t sourceMBlock = partitionByM ? rowBlock / singleCoreM_ : 0U;
                const bool smallNFirstPartial = partitionByM && NeedRowMajorSmallNOriginalStride();
                if constexpr (!BOTH_TRANS_FIRST_PARTIAL) {
                    if (!smallNFirstPartial) {
                        for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                            Duplicate(accumLocal[rowInner * ubNAligned_], 0.0f, curN);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                if constexpr (USE_INT32_PUBLIC) {
                    LocalTensor<float> srcLocal;
                    for (uint32_t core = 0; core < usedCoreNum_; ++core) {
                        LoadStreamKInt32Partial(srcLocal, core, rowBlock, nOffset, rowsThis, curN);
                        for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                            Add(accumLocal[rowInner * ubNAligned_], accumLocal[rowInner * ubNAligned_],
                                srcLocal[rowInner * ubNAligned_], curN);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                } else if constexpr (SINGLE_ROW_SPECIALIZED) {
                    for (uint32_t core = 0; core < usedCoreNum_; ++core) {
                        EnqueueStreamKPartial(core, rowBlock, nOffset, rowsThis, curN);
                        LocalTensor<float> srcLocal = vecQueSrc_.DeQue<float>();
                        for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                            Add(accumLocal[rowInner * ubNAligned_], accumLocal[rowInner * ubNAligned_],
                                srcLocal[rowInner * ubNAligned_], curN);
                        }
                        PipeBarrier<PIPE_V>();
                        vecQueSrc_.FreeTensor<float>(srcLocal);
                    }
                } else if constexpr (MID_N_SPECIALIZED) {
                    for (uint32_t core = 0; core < usedCoreNum_; ++core) {
                        EnqueueStreamKPartial(core, rowBlock, nOffset, rowsThis, curN);
                        LocalTensor<float> srcLocal = vecQueSrc_.DeQue<float>();
                        for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                            Add(accumLocal[rowInner * ubNAligned_], accumLocal[rowInner * ubNAligned_],
                                srcLocal[rowInner * ubNAligned_], curN);
                        }
                        PipeBarrier<PIPE_V>();
                        vecQueSrc_.FreeTensor<float>(srcLocal);
                    }
                } else if (TRANS_A_SPECIALIZED || NeedRowMajorSmallMStreamK() || NeedRowMajorTallMStreamK() ||
                    NeedTransALargeKStreamK() || NeedBothTransLargeKMidNOriginalStride()) {
                    uint32_t firstReduceCore = 0U;
                    if constexpr (BOTH_TRANS_FIRST_PARTIAL) {
                        DataCopyExtParams firstCopy{
                            static_cast<uint16_t>(rowsThis),
                            static_cast<uint32_t>(curN * sizeof(float)),
                            static_cast<uint32_t>((n_ - curN) * sizeof(float)),
                            static_cast<uint32_t>((ubNAligned_ - curN) * sizeof(float) / GM_BLOCK_BYTES),
                            0};
                        DataCopyPadExtParams<float> firstPad{false, 0, 0, 0.0f};
                        const uint64_t firstOffset = static_cast<uint64_t>(rowBlock) * n_ + nOffset;
                        DataCopyPad(accumLocal, partialGm_[firstOffset], firstCopy, firstPad);
                        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                        firstReduceCore = 1U;
                    }
                    if (firstReduceCore < usedCoreNum_) {
                        EnqueueStreamKPartial(firstReduceCore, rowBlock, nOffset, rowsThis, curN);
                    }
                    for (uint32_t core = firstReduceCore; core < usedCoreNum_; ++core) {
                        LocalTensor<float> srcLocal = vecQueSrc_.DeQue<float>();
                        if (core + 1U < usedCoreNum_) {
                            EnqueueStreamKPartial(core + 1U, rowBlock, nOffset, rowsThis, curN);
                        }
                        if (NeedRowMajorTallMStreamK() && ubNAligned_ == curN) {
                            Add(accumLocal, accumLocal, srcLocal, rowsThis * curN);
                        } else {
                            for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                                Add(accumLocal[rowInner * ubNAligned_], accumLocal[rowInner * ubNAligned_],
                                    srcLocal[rowInner * ubNAligned_], curN);
                            }
                        }
                        PipeBarrier<PIPE_V>();
                        vecQueSrc_.FreeTensor<float>(srcLocal);
                    }
                } else {
                    uint32_t firstSourceCore = 0U;
                    if (smallNFirstPartial) {
                        DataCopyExtParams firstCopy{
                            static_cast<uint16_t>(rowsThis),
                            static_cast<uint32_t>(curN * sizeof(float)),
                            static_cast<uint32_t>((n_ - curN) * sizeof(float)),
                            static_cast<uint32_t>((ubNAligned_ - curN) * sizeof(float) / GM_BLOCK_BYTES),
                            0};
                        DataCopyPadExtParams<float> firstPad{false, 0, 0, 0.0f};
                        const uint32_t sourceCore = sourceMBlock;
                        const uint64_t firstOffset =
                            (static_cast<uint64_t>(sourceCore) * m_ + rowBlock) * n_ + nOffset;
                        DataCopyPad(accumLocal, partialGm_[firstOffset], firstCopy, firstPad);
                        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                        firstSourceCore = 1U;
                    }
                    for (uint32_t core = firstSourceCore; core < sourceCoreCount; ++core) {
                        const uint32_t sourceCore = partitionByM ? core * sourceMBlocks + sourceMBlock : core;
                        EnqueueStreamKPartial(sourceCore, rowBlock, nOffset, rowsThis, curN);
                        LocalTensor<float> srcLocal = vecQueSrc_.DeQue<float>();
                        if (NeedRowMajorSmallNOriginalStride() && ubNAligned_ == curN) {
                            Add(accumLocal, accumLocal, srcLocal, rowsThis * curN);
                        } else {
                            for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                                Add(accumLocal[rowInner * ubNAligned_], accumLocal[rowInner * ubNAligned_],
                                    srcLocal[rowInner * ubNAligned_], curN);
                            }
                        }
                        PipeBarrier<PIPE_V>();
                        vecQueSrc_.FreeTensor<float>(srcLocal);
                    }
                }

                const bool useBatchedReduce = (SINGLE_ROW_SPECIALIZED || MID_N_SPECIALIZED) ? false :
                    (TRANS_A_SPECIALIZED || NeedTransALargeKStreamK() || NeedRowMajorTallMStreamK()) &&
                        curN % FP32_BLOCK_ELEMS == 0U;
                if (useBatchedReduce) {
                    DataCopyExtParams tokenCopyParams{
                        1, static_cast<uint32_t>(rowsThis * sizeof(float)), 0, 0, 0};
                    DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
                    DataCopyPad(tokenScaleLocal, pertokenScaleGm_[rowBlock], tokenCopyParams, tokenPadParams);
                    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                    const uint32_t alignedRows = AlignUp(rowsThis, FP32_BLOCK_ELEMS);
                    Brcb(tokenScaleBrcbLocal, tokenScaleLocal,
                        static_cast<uint8_t>(alignedRows / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
                    if (scaleLen_ == 1U) {
                        PipeBarrier<PIPE_V>();
                        Muls(tokenScaleBrcbLocal, tokenScaleBrcbLocal, scaleScalar,
                            alignedRows * FP32_BLOCK_ELEMS);
                    }
                    PipeBarrier<PIPE_V>();

                    const uint8_t dstRowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                    const uint8_t srcRowStride = static_cast<uint8_t>(ubNAligned_ / FP32_BLOCK_ELEMS);
                    constexpr uint32_t scaleChunkElems = 64U;
                    if (scaleLen_ != 1U) {
                        BinaryRepeatParams channelScaleParams{
                            1, 1, 1, dstRowStride, srcRowStride, 0};
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += scaleChunkElems) {
                            const uint32_t colThis = Min(scaleChunkElems, curN - colOffset);
                            Mul(fp32Local[colOffset], accumLocal[colOffset], scaleLocal[colOffset],
                                static_cast<uint64_t>(colThis), static_cast<uint8_t>(rowsThis),
                                channelScaleParams);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    if (scaleLen_ == 1U) {
                        BinaryRepeatParams tokenScaleParams{
                            1, 1, 0, dstRowStride, srcRowStride, 1};
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += scaleChunkElems) {
                            const uint32_t colThis = Min(scaleChunkElems, curN - colOffset);
                            Mul(fp32Local[colOffset], accumLocal[colOffset], tokenScaleBrcbLocal,
                                static_cast<uint64_t>(colThis), static_cast<uint8_t>(rowsThis), tokenScaleParams);
                        }
                    } else {
                        BinaryRepeatParams tokenScaleParams{
                            1, 1, 0, dstRowStride, dstRowStride, 1};
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += scaleChunkElems) {
                            const uint32_t colThis = Min(scaleChunkElems, curN - colOffset);
                            Mul(fp32Local[colOffset], fp32Local[colOffset], tokenScaleBrcbLocal,
                                static_cast<uint64_t>(colThis), static_cast<uint8_t>(rowsThis), tokenScaleParams);
                        }
                    }
                } else {
                    for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                        const float tokenScale = pertokenScaleGm_.GetValue(rowBlock + rowInner);
                        if (scaleLen_ == 1U) {
                            Muls(fp32Local[rowInner * curN], accumLocal[rowInner * ubNAligned_],
                                scaleScalar * tokenScale, curN);
                        } else {
                            Mul(fp32Local[rowInner * curN], accumLocal[rowInner * ubNAligned_], scaleLocal, curN);
                            PipeBarrier<PIPE_V>();
                            Muls(fp32Local[rowInner * curN], fp32Local[rowInner * curN], tokenScale, curN);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();

                if (useBatchedReduce) {
                    LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                    Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
                    PipeBarrier<PIPE_V>();
                    vecQueOut_.EnQue<bfloat16_t>(outLocal);

                    outLocal = vecQueOut_.DeQue<bfloat16_t>();
                    DataCopyExtParams copyOutParams{
                        static_cast<uint16_t>(rowsThis),
                        static_cast<uint32_t>(curN * sizeof(bfloat16_t)),
                        0,
                        static_cast<uint32_t>((n_ - curN) * sizeof(bfloat16_t)),
                        0};
                    DataCopyPad(outGm_[static_cast<uint64_t>(rowBlock) * n_ + nOffset], outLocal, copyOutParams);
                    vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
                } else {
                    for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                        LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                        Cast(outLocal, fp32Local[rowInner * curN], RoundMode::CAST_RINT, curN);
                        PipeBarrier<PIPE_V>();
                        vecQueOut_.EnQue<bfloat16_t>(outLocal);

                        outLocal = vecQueOut_.DeQue<bfloat16_t>();
                        DataCopyExtParams copyOutParams{
                            1, static_cast<uint32_t>(curN * sizeof(bfloat16_t)), 0, 0, 0};
                        DataCopyPad(outGm_[static_cast<uint64_t>(rowBlock + rowInner) * n_ + nOffset], outLocal,
                            copyOutParams);
                        vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
                    }
                }
            }

            if (scaleLen_ != 1U) {
                vecQueScale_.FreeTensor<float>(scaleLocal);
            }
        }
    }

    // Stage one operand from its original GM layout into a compact per-core buffer so that every
    // stride fed to SetOrgShape stays below the 65536 hardware limit (intrinsics-free execution).
    __aicore__ inline void StageCompactOperand(
        GlobalTensor<int8_t>& dstGm,
        GlobalTensor<int8_t>& srcGm,
        uint64_t srcBaseOffset,
        uint32_t rows,
        uint32_t rowBytes,
        uint32_t dstRowStride,
        uint32_t srcRowStride)
    {
        // Conservative copies: the source row advance stays in 64-bit address arithmetic for the chunk
        // base, while rows inside a chunk ride one 2D DataCopyPad whose byte strides tolerate arbitrarily
        // large original strides (same pattern as the proven epilogue copy-out with (n_ - curN) strides).
        // O12: batch rows per chunk instead of one DMA round-trip per row; per-chunk MTE2->MTE3->MTE2
        // flags keep the single UB buffer safe (O11 proved this flag pairing).
        const uint32_t rowPitchUb = AlignUp(rowBytes, GM_BLOCK_BYTES);
        uint32_t rowsPerChunk = vecQueABytes_ / rowPitchUb;
        if (rowsPerChunk == 0U) {
            rowsPerChunk = 1U;
        }
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += rowsPerChunk) {
            const uint32_t chunkRows = Min(rowsPerChunk, rows - rowBase);
            LocalTensor<int8_t> tmp = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(chunkRows),
                rowBytes,
                static_cast<uint32_t>(srcRowStride - rowBytes),
                static_cast<uint32_t>(rowPitchUb - rowBytes),
                0};
            DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
            DataCopyPad(
                tmp, srcGm[srcBaseOffset + static_cast<uint64_t>(rowBase) * srcRowStride], copyInParams, padParams);
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));

            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(chunkRows),
                rowBytes,
                static_cast<uint32_t>(rowPitchUb - rowBytes),
                static_cast<uint32_t>(dstRowStride - rowBytes),
                0};
            DataCopyPad(dstGm[static_cast<uint64_t>(rowBase) * dstRowStride], tmp, copyOutParams);
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            vecQueA_.FreeTensor<int8_t>(tmp);
        }
    }

    __aicore__ inline void ProcessStagedOriginalStrideTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        if constexpr (TRANS_X1) {
            // LargeM leaves (k_ <= 1024): stage A [K, curM] compact; orgM becomes the staged pitch.
            const uint32_t aStride = AlignUp(singleCoreM_, GM_BLOCK_BYTES);
            StageCompactOperand(aCompactGm_, x1Gm_, GetAOffset(globalMOffset, 0), k_,
                curM * sizeof(int8_t), aStride, m_);
            mm.SetOrgShape(aStride, n_, k_, k_, curN);
            mm.SetTail(curM, curN, k_);
            mm.SetTensorA(aCompactGm_, true);
            mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, 0)], TRANS_X2);
            mm.template Iterate<false>();
            auto mmOutGm = mm.GetTensorC();
            ScaleMatmulOutAndStore(mmOutGm, globalMOffset, globalNOffset, curM, curN);
            mm.End();
            return;
        }
        if constexpr (TRANS_X2) {
            // TransB LargeK SmallN leaf: chunk K so both staged strides stay below the 65536 limit.
            LocalTensor<float> accumLocal = vecAccum_.Get<float>();
            Duplicate(accumLocal, 0.0f, curM * ubNAligned_);
            PipeBarrier<PIPE_V>();
            for (uint32_t kOffset = 0; kOffset < k_; kOffset += TRANS_B_COMPACT_K_CHUNK) {
                const uint32_t curK = Min(TRANS_B_COMPACT_K_CHUNK, k_ - kOffset);
                const uint32_t compactStride = AlignUp(curK, GM_BLOCK_BYTES);
                StageCompactOperand(aCompactGm_, x1Gm_,
                    static_cast<uint64_t>(globalMOffset) * k_ + kOffset, curM, curK * sizeof(int8_t),
                    compactStride, k_);
                StageCompactOperand(bCompactGm_, x2Gm_,
                    static_cast<uint64_t>(globalNOffset) * k_ + kOffset, curN, curK * sizeof(int8_t),
                    compactStride, k_);
                mm.SetOrgShape(curM, curN, compactStride, compactStride, curN);
                mm.SetTail(curM, curN, curK);
                mm.SetTensorA(aCompactGm_, TRANS_X1);
                mm.SetTensorB(bCompactGm_, TRANS_X2);
                mm.template Iterate<false>();
                auto mmOutGm = mm.GetTensorC();
                AccumulatePartial(mmOutGm, accumLocal, curM, curN);
                mm.End();
            }
            ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        // LargeN MediumM leaf (k_ <= 1024): stage B [K, curN] compact; orgN becomes the staged pitch.
        // bStride must cover curN for every tile: tileN comes from baseN_, the buffer from singleCoreN_.
        const uint32_t bStride = AlignUp(Max(singleCoreN_, baseN_), GM_BLOCK_BYTES);
        StageCompactOperand(bCompactGm_, x2Gm_, GetBOffset(globalNOffset, 0), k_,
            curN * sizeof(int8_t), bStride, n_);
        mm.SetOrgShape(curM, bStride, k_, k_, curN);
        mm.SetTail(curM, curN, k_);
        mm.SetTensorA(x1Gm_[GetAOffset(globalMOffset, 0)], false);
        mm.SetTensorB(bCompactGm_, false);
        mm.template Iterate<false>();
        auto mmOutGm = mm.GetTensorC();
        ScaleMatmulOutAndStore(mmOutGm, globalMOffset, globalNOffset, curM, curN);
        if constexpr (!A_IBSHARE) {
            mm.End();
        }
    }

    __aicore__ inline void CopyCompactA(uint32_t globalMOffset, uint32_t kOffset, uint32_t curM, uint32_t curK)
    {
        for (uint32_t row = 0; row < curK; row += A_COMPACT_COPY_ROWS) {
            const uint32_t copyRows = Min(A_COMPACT_COPY_ROWS, curK - row);
            LocalTensor<int8_t> aLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(copyRows),
                static_cast<uint32_t>(curM * sizeof(int8_t)),
                static_cast<uint32_t>((m_ - curM) * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                0};
            DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
            DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(kOffset + row) * m_ + globalMOffset], copyInParams,
                padParams);
            vecQueA_.EnQue<int8_t>(aLocal);

            aLocal = vecQueA_.DeQue<int8_t>();
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(copyRows),
                static_cast<uint32_t>(curM * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                0};
            DataCopyPad(aCompactGm_[static_cast<uint64_t>(row) * aCompactStride_], aLocal, copyOutParams);
            SetFlag<HardEvent::MTE3_MTE1>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE1>(static_cast<event_t>(0));
            vecQueA_.FreeTensor<int8_t>(aLocal);
        }
    }

    __aicore__ inline void AccumulatePartial(
        GlobalTensor<int32_t>& mmOutGm,
        LocalTensor<float>& accumLocal,
        uint32_t curM,
        uint32_t curN)
    {
        if (NeedTransBLargeKSmallNOriginalStride() || NeedRowMajorSmallNOriginalStride() ||
            NeedBothTransLargeKMidNOriginalStride()) {
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(curM),
                static_cast<uint32_t>(curN * sizeof(int32_t)),
                0,
                0,
                0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm, copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);

            srcLocal = vecQueSrc_.DeQue<int32_t>();
            LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
            Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, curM * curN);
            PipeBarrier<PIPE_V>();
            if ((NeedBothTransLargeKMidNOriginalStride() || NeedRowMajorSmallNOriginalStride()) &&
                curN % FP32_BLOCK_ELEMS == 0U) {
                BinaryRepeatParams addParams{1, 1, 1,
                    static_cast<uint8_t>(ubNAligned_ / FP32_BLOCK_ELEMS),
                    static_cast<uint8_t>(ubNAligned_ / FP32_BLOCK_ELEMS),
                    static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS)};
                Add(accumLocal, accumLocal, fp32Local, static_cast<uint64_t>(curN),
                    static_cast<uint8_t>(curM), addParams);
            } else {
                for (uint32_t row = 0; row < curM; ++row) {
                    Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_],
                        fp32Local[row * curN], curN);
                }
            }
            PipeBarrier<PIPE_V>();
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
            return;
        }
        for (uint32_t row = 0; row < curM; ++row) {
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curN * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(row) * curN], copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);

            srcLocal = vecQueSrc_.DeQue<int32_t>();
            LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
            Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, curN);
            PipeBarrier<PIPE_V>();
            Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], fp32Local, curN);
            PipeBarrier<PIPE_V>();
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
        }
    }

    __aicore__ inline void ScaleMatmulOutAndStore(
        GlobalTensor<int32_t>& mmOutGm,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal = vecTokenScale_.Get<float>();
        const bool rowAligned = curN % FP32_BLOCK_ELEMS == 0U;
        if (cachedTokenMOffset_ != globalMOffset) {
            DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
            DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);
            if (rowAligned) {
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            } else {
                SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            }
            cachedTokenMOffset_ = globalMOffset;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            const uint32_t tokenBroadcastElems =
                AlignUp(SelectDirectAicRowStep(), FP32_BLOCK_ELEMS) * FP32_BLOCK_ELEMS;
            scaleLocal = vecAccum_.Get<float>()[tokenBroadcastElems];
            if (cachedScaleNOffset_ != globalNOffset) {
                DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
                DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                cachedScaleNOffset_ = globalNOffset;
            }
        }

        const float scaleScalar = scaleScalar_;
        LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
        const uint32_t rowStep = SelectDirectAicRowStep();
        const bool useScaleBroadcast = scaleLen_ != 1U && rowAligned;
        const bool useTokenScaleBroadcast = rowAligned;
        LocalTensor<float> tokenScaleBrcbLocal = vecAccum_.Get<float>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            // Copy-policy leaf: collapse zero-stride rows for full-N tiny/small tiles; keep mid-N batched copies.
            const bool contiguousRows = curN == n_ && !NeedLargeMMidNOriginalStride();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(contiguousRows ? 1U : rowsThis),
                static_cast<uint32_t>((contiguousRows ? rowsThis : 1U) * curN * sizeof(int32_t)),
                0,
                0,
                0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(rowBlock) * curN], copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);
            srcLocal = vecQueSrc_.DeQue<int32_t>();
            Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
            PipeBarrier<PIPE_V>();

            if (useScaleBroadcast) {
                const uint8_t rowRepeat = static_cast<uint8_t>(rowsThis);
                const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                BinaryRepeatParams scaleParams{1, 1, 1, rowStride, rowStride, 0};
                const uint32_t scaleChunkElems = 64U;
                for (uint32_t colOffset = 0; colOffset < curN; colOffset += scaleChunkElems) {
                    const uint32_t colThis = Min(scaleChunkElems, curN - colOffset);
                    Mul(fp32Local[colOffset], fp32Local[colOffset], scaleLocal[colOffset],
                        static_cast<uint64_t>(colThis), rowRepeat, scaleParams);
                }
                PipeBarrier<PIPE_V>();
            }

            if (useTokenScaleBroadcast) {
                const uint32_t alignedRows = AlignUp(rowsThis, FP32_BLOCK_ELEMS);
                Brcb(tokenScaleBrcbLocal, tokenScaleLocal[rowBlock],
                    static_cast<uint8_t>(alignedRows / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
                if (scaleLen_ == 1U) {
                    PipeBarrier<PIPE_V>();
                    Muls(tokenScaleBrcbLocal, tokenScaleBrcbLocal, scaleScalar,
                        alignedRows * FP32_BLOCK_ELEMS);
                }
                PipeBarrier<PIPE_V>();
                const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                BinaryRepeatParams tokenScaleParams{1, 1, 0, rowStride, rowStride, 1};
                constexpr uint32_t tokenScaleChunkElems = 64U;
                for (uint32_t colOffset = 0; colOffset < curN; colOffset += tokenScaleChunkElems) {
                    const uint32_t colThis = Min(tokenScaleChunkElems, curN - colOffset);
                    Mul(fp32Local[colOffset], fp32Local[colOffset], tokenScaleBrcbLocal,
                        static_cast<uint64_t>(colThis), static_cast<uint8_t>(rowsThis), tokenScaleParams);
                }
            } else {
                for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                    float tokenScale = 1.0f;
                    const uint32_t row = rowBlock + rowInner;
                    if constexpr (TRANS_X1) {
                        tokenScale = tokenScaleLocal.GetValue(row);
                    } else {
                        tokenScale = tokenScaleLocal.GetValue(row);
                    }
                    if (scaleLen_ == 1U) {
                        Muls(fp32Local[rowInner * curN], fp32Local[rowInner * curN],
                            scaleScalar * tokenScale, curN);
                    } else {
                        if (!useScaleBroadcast) {
                            Mul(fp32Local[rowInner * curN], fp32Local[rowInner * curN], scaleLocal, curN);
                            PipeBarrier<PIPE_V>();
                        }
                        Muls(fp32Local[rowInner * curN], fp32Local[rowInner * curN], tokenScale, curN);
                    }
                }
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);
            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(contiguousRows ? 1U : rowsThis),
                static_cast<uint32_t>((contiguousRows ? rowsThis : 1U) * curN * sizeof(bfloat16_t)),
                0,
                contiguousRows ? 0U : static_cast<uint32_t>((n_ - curN) * sizeof(bfloat16_t)),
                0};
            DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset],
                outLocal, copyOutParams);
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
        }

    }

    __aicore__ inline void ProcessLargeMVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal = vecTokenScale_.Get<float>();
        DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
        DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);

        const uint32_t colStride = curM;
        for (uint32_t col = 0; col < curN; ++col) {
            Duplicate(accumLocal[col * colStride], 0.0f, curM);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<half> aHalfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();
        const uint32_t tmpStride = AlignUp(baseM_, FP32_BLOCK_ELEMS);
        LocalTensor<float> tmpLocal = accumLocal[curN * colStride];
        LocalTensor<float> tmp2Local = tmpLocal;
        LocalTensor<float> tmp3Local = tmpLocal[tmpStride];
        LocalTensor<float> genericTmpLocal = tmpLocal[2U * tmpStride];
        LocalTensor<int8_t> transBLocal;
        bool cacheTransB = false;
        if constexpr (TRANS_X2) {
            cacheTransB = curN * k_ <= LARGE_M_TRANS_B_CACHE_ELEMS;
            if (cacheTransB) {
                transBLocal = vecQueSrc_.AllocTensor<int8_t>();
                DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * k_ * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                DataCopyPad(transBLocal, x2Gm_[static_cast<uint64_t>(globalNOffset) * k_], bCopyParams, bPadParams);
                vecQueSrc_.EnQue<int8_t>(transBLocal);
                transBLocal = vecQueSrc_.DeQue<int8_t>();
            }
        }
        for (uint32_t kk = 0; kk < k_; ++kk) {
            LocalTensor<int8_t> aLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curM * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
            DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(kk) * m_ + globalMOffset], aCopyParams, aPadParams);
            vecQueA_.EnQue<int8_t>(aLocal);

            aLocal = vecQueA_.DeQue<int8_t>();
            Cast(aHalfLocal, aLocal, RoundMode::CAST_NONE, curM);
            PipeBarrier<PIPE_V>();
            Cast(aFp32Local, aHalfLocal, RoundMode::CAST_NONE, curM);
            PipeBarrier<PIPE_V>();
            const bool reuseSmallB = curN >= 64U;
            if (reuseSmallB) {
                Add(tmp2Local, aFp32Local, aFp32Local, curM);
                PipeBarrier<PIPE_V>();
                Add(tmp3Local, tmp2Local, aFp32Local, curM);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<int8_t> bLocal;
            bool cacheBRow = false;
            if constexpr (!TRANS_X2) {
                cacheBRow = curN > 64U;
                if (cacheBRow) {
                    bLocal = vecQueSrc_.AllocTensor<int8_t>();
                    DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * sizeof(int8_t)), 0, 0, 0};
                    DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                    DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(kk) * n_ + globalNOffset], bCopyParams,
                        bPadParams);
                    vecQueSrc_.EnQue<int8_t>(bLocal);
                    bLocal = vecQueSrc_.DeQue<int8_t>();
                }
            }
            for (uint32_t col = 0; col < curN; ++col) {
                int8_t bValue = 0;
                if constexpr (TRANS_X2) {
                    bValue = cacheTransB ?
                        transBLocal.GetValue(col * k_ + kk) :
                        x2Gm_.GetValue(static_cast<uint64_t>(globalNOffset + col) * k_ + kk);
                } else {
                    bValue = cacheBRow ?
                        bLocal.GetValue(col) :
                        x2Gm_.GetValue(static_cast<uint64_t>(kk) * n_ + globalNOffset + col);
                }
                if (reuseSmallB) {
                    if (bValue == 0) {
                        continue;
                    }
                    if (bValue == 1) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], aFp32Local, curM);
                        continue;
                    }
                    if (bValue == -1) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], aFp32Local, curM);
                        continue;
                    }
                    if (bValue == 2) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                        continue;
                    }
                    if (bValue == -2) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                        continue;
                    }
                    if (bValue == 3) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                        continue;
                    }
                    if (bValue == -3) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                        continue;
                    }
                }
                LocalTensor<float> mulTmpLocal = reuseSmallB ? genericTmpLocal : tmpLocal;
                Muls(mulTmpLocal, aFp32Local, static_cast<float>(bValue), curM);
                PipeBarrier<PIPE_V>();
                Add(accumLocal[col * colStride], accumLocal[col * colStride], mulTmpLocal, curM);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (!TRANS_X2) {
                if (cacheBRow) {
                    vecQueSrc_.FreeTensor<int8_t>(bLocal);
                }
            }
            vecQueA_.FreeTensor<int8_t>(aLocal);
        }
        if constexpr (TRANS_X2) {
            if (cacheTransB) {
                vecQueSrc_.FreeTensor<int8_t>(transBLocal);
            }
        }

        if (curN >= 16U && curN <= 64U && (curN % 16U) == 0U && (curM % 16U) == 0U) {
            StoreLargeMAccumBlock16Small(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        } else if (curN <= 64U) {
            StoreLargeMAccumRowMajor(tokenScaleLocal, accumLocal, globalMOffset, globalNOffset, curM, curN, colStride);
        } else if constexpr (!TRANS_X2) {
            StoreLargeMAccumBlock16(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        } else {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        }
    }

    __aicore__ inline void StoreLargeMAccumRowMajor(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleScalar_;
        LocalTensor<float> rowFp32Local = vecFp32Tmp_.Get<float>();
        for (uint32_t row = 0; row < curM; ++row) {
            const float tokenScale = tokenScaleLocal.GetValue(row);
            for (uint32_t col = 0; col < curN; ++col) {
                const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(col);
                rowFp32Local.SetValue(col,
                    accumLocal.GetValue(static_cast<uint64_t>(col) * colStride + row) * tokenScale * scaleValue);
            }
            SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, rowFp32Local, RoundMode::CAST_RINT, curN);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);

            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curN * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + row) * n_ + globalNOffset],
                outLocal, copyOutParams);
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumColumnMajor(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        for (uint32_t col = 0; col < curN; ++col) {
            const float scaleValue = scaleLen_ == 1U ? scaleScalar_ :
                scaleGm_.GetValue(globalNOffset + col);
            LocalTensor<float> outFp32Local = tmpLocal;
            Mul(outFp32Local, accumLocal[col * colStride], tokenScaleLocal, curM);
            PipeBarrier<PIPE_V>();
            Muls(outFp32Local, outFp32Local, scaleValue, curM);
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, outFp32Local, RoundMode::CAST_RINT, curM);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);

            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            for (uint32_t row = 0; row < curM; ++row) {
                outGm_.SetValue(static_cast<uint64_t>(globalMOffset + row) * n_ + globalNOffset + col,
                    outLocal.GetValue(row));
            }
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumBlock16(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        constexpr uint32_t block = 16U;
        constexpr uint32_t rowStep = 32U;
        if (curN != LARGE_M_VECTOR_N_LIMIT || (curM % block) != 0U) {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
            return;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleScalar_;
        LocalTensor<float> outFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<bfloat16_t> bf16SrcLocal = vecHalfTmp_.Get<half>().template ReinterpretCast<bfloat16_t>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            for (uint32_t colBlock = 0; colBlock < curN; colBlock += block) {
                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> transSrc = bf16SrcLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t rowInner = 0; rowInner < rowsThis; rowInner += block) {
                    const uint32_t rowBase = rowBlock + rowInner;
                    for (uint32_t col = 0; col < block; ++col) {
                        const uint32_t globalCol = colBlock + col;
                        const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(globalCol);
                        Mul(outFp32Local[col * block], accumLocal[globalCol * colStride + rowBase],
                            tokenScaleLocal[rowBase], block);
                        PipeBarrier<PIPE_V>();
                        Muls(outFp32Local[col * block], outFp32Local[col * block], scaleValue, block);
                        PipeBarrier<PIPE_V>();
                        Cast(bf16SrcLocal[col * block], outFp32Local[col * block], RoundMode::CAST_RINT, block);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<uint16_t> transDst = outLocal[rowInner * block].template ReinterpretCast<uint16_t>();
                    Transpose(transDst, transSrc);
                    PipeBarrier<PIPE_V>();
                }
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(block * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((n_ - block) * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset + colBlock],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumBlock16Small(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        constexpr uint32_t block = 16U;
        constexpr uint32_t rowStep = 32U;
        if ((curN % block) != 0U || (curM % block) != 0U) {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
            return;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleScalar_;
        LocalTensor<float> outFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<bfloat16_t> bf16SrcLocal = vecHalfTmp_.Get<half>().template ReinterpretCast<bfloat16_t>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            for (uint32_t colBlock = 0; colBlock < curN; colBlock += block) {
                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> transSrc = bf16SrcLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t rowInner = 0; rowInner < rowsThis; rowInner += block) {
                    const uint32_t rowBase = rowBlock + rowInner;
                    for (uint32_t col = 0; col < block; ++col) {
                        const uint32_t globalCol = colBlock + col;
                        const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(globalCol);
                        Mul(outFp32Local[col * block], accumLocal[globalCol * colStride + rowBase],
                            tokenScaleLocal[rowBase], block);
                        PipeBarrier<PIPE_V>();
                        Muls(outFp32Local[col * block], outFp32Local[col * block], scaleValue, block);
                        PipeBarrier<PIPE_V>();
                        Cast(bf16SrcLocal[col * block], outFp32Local[col * block], RoundMode::CAST_RINT, block);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<uint16_t> transDst = outLocal[rowInner * block].template ReinterpretCast<uint16_t>();
                    Transpose(transDst, transSrc);
                    PipeBarrier<PIPE_V>();
                }
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(block * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((n_ - block) * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset + colBlock],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline bool NeedLargeNVector() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return n_ > MATMUL_N_LIMIT && k_ <= LARGE_N_VECTOR_K_LIMIT && !NeedLargeNMediumMOriginalStride();
        }
        return false;
    }

    __aicore__ inline bool NeedLargeMVector() const
    {
        return TRANS_X1 && m_ > MATMUL_M_LIMIT && k_ <= LARGE_M_VECTOR_K_LIMIT && n_ <= LARGE_M_VECTOR_N_LIMIT;
    }

    __aicore__ inline bool NeedSingleColumnVector() const
    {
        return !TRANS_X1 && !TRANS_X2 && n_ <= 1U && k_ > MATMUL_K_LIMIT;
    }

    __aicore__ inline bool NeedLargeKTransBVector() const
    {
        return !TRANS_X1 && TRANS_X2 && k_ > MATMUL_K_LIMIT &&
            m_ <= LARGE_K_TRANS_B_VECTOR_M_LIMIT && n_ <= LARGE_K_TRANS_B_VECTOR_N_LIMIT;
    }

    __aicore__ inline bool NeedLargeKTransBColumnTile() const
    {
        return NeedTransBLargeKSmallNOriginalStride();
    }

    __aicore__ inline void ProcessSingleColumnVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        if (globalNOffset != 0U || curN != 1U) {
            return;
        }

        const uint32_t accumElems = AlignUp(singleCoreM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<half> bHalfLocal = accumLocal[accumElems].template ReinterpretCast<half>();
        LocalTensor<float> mulLocal = accumLocal[accumElems + SINGLE_COLUMN_VECTOR_K_TILE];
        LocalTensor<float> reduceOutLocal = accumLocal[
            accumElems + SINGLE_COLUMN_VECTOR_K_TILE * (SINGLE_COLUMN_REDUCE_ROWS + 1U)];
        LocalTensor<half> halfLocal = vecHalfTmp_.Get<half>();

        for (uint32_t kOffset = 0; kOffset < k_; kOffset += SINGLE_COLUMN_VECTOR_K_TILE) {
            const uint32_t curK = Min(SINGLE_COLUMN_VECTOR_K_TILE, k_ - kOffset);
            const uint32_t reduceRowStride = AlignUp(curK, FP32_BLOCK_ELEMS);

            LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
            DataCopyPad(bLocal, x2Gm_[kOffset], bCopyParams, bPadParams);
            vecQueA_.EnQue<int8_t>(bLocal);

            bLocal = vecQueA_.DeQue<int8_t>();
            Cast(bHalfLocal, bLocal, RoundMode::CAST_NONE, curK);
            PipeBarrier<PIPE_V>();
            vecQueA_.FreeTensor<int8_t>(bLocal);

            for (uint32_t row = 0; row < curM; row += SINGLE_COLUMN_REDUCE_ROWS) {
                const uint32_t rowsThis = Min(SINGLE_COLUMN_REDUCE_ROWS, curM - row);
                const bool alignedBatchCopy = (k_ % GM_BLOCK_BYTES) == 0U &&
                    (curK % GM_BLOCK_BYTES) == 0U;
                if (alignedBatchCopy) {
                    LocalTensor<int8_t> aLocal = vecQueSingleColumnSrc_.AllocTensor<int8_t>();
                    DataCopyParams batchCopy{
                        static_cast<uint16_t>(rowsThis),
                        static_cast<uint16_t>(curK / GM_BLOCK_BYTES),
                        static_cast<uint16_t>((k_ - curK) / GM_BLOCK_BYTES),
                        0U};
                    DataCopy(aLocal, x1Gm_[static_cast<uint64_t>(globalMOffset + row) * k_ + kOffset],
                        batchCopy);
                    vecQueSingleColumnSrc_.EnQue<int8_t>(aLocal);

                    aLocal = vecQueSingleColumnSrc_.DeQue<int8_t>();
                    const uint32_t pipelineRows = Min(SINGLE_COLUMN_PIPELINE_ROWS, rowsThis);
                    for (uint32_t rowInGroup = 0; rowInGroup < pipelineRows; ++rowInGroup) {
                        Cast(halfLocal[static_cast<uint64_t>(rowInGroup) * curK],
                            aLocal[static_cast<uint64_t>(rowInGroup) * curK],
                            RoundMode::CAST_NONE, curK);
                    }
                    PipeBarrier<PIPE_V>();
                    for (uint32_t rowInGroup = 0; rowInGroup < pipelineRows; ++rowInGroup) {
                        LocalTensor<half> halfRowLocal =
                            halfLocal[static_cast<uint64_t>(rowInGroup) * curK];
                        Mul(halfRowLocal, halfRowLocal, bHalfLocal, curK);
                    }
                    PipeBarrier<PIPE_V>();
                    for (uint32_t rowInGroup = 0; rowInGroup < pipelineRows; ++rowInGroup) {
                        LocalTensor<float> mulRowLocal =
                            mulLocal[static_cast<uint64_t>(rowInGroup) * reduceRowStride];
                        Cast(mulRowLocal, halfLocal[static_cast<uint64_t>(rowInGroup) * curK],
                            RoundMode::CAST_NONE, curK);
                    }
                    PipeBarrier<PIPE_V>();
                    for (uint32_t rowInGroup = pipelineRows; rowInGroup < rowsThis; ++rowInGroup) {
                        Cast(halfLocal, aLocal[static_cast<uint64_t>(rowInGroup) * curK],
                            RoundMode::CAST_NONE, curK);
                        PipeBarrier<PIPE_V>();
                        LocalTensor<float> mulRowLocal =
                            mulLocal[static_cast<uint64_t>(rowInGroup) * reduceRowStride];
                        Mul(halfLocal, halfLocal, bHalfLocal, curK);
                        PipeBarrier<PIPE_V>();
                        Cast(mulRowLocal, halfLocal, RoundMode::CAST_NONE, curK);
                        PipeBarrier<PIPE_V>();
                    }
                    vecQueSingleColumnSrc_.FreeTensor<int8_t>(aLocal);
                } else {
                    DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                    DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
                    for (uint32_t rowInGroup = 0; rowInGroup < rowsThis; ++rowInGroup) {
                        LocalTensor<int8_t> aLocal = vecQueSingleColumnSrc_.AllocTensor<int8_t>();
                        DataCopyPad(aLocal,
                            x1Gm_[static_cast<uint64_t>(globalMOffset + row + rowInGroup) * k_ + kOffset],
                            aCopyParams, aPadParams);
                        vecQueSingleColumnSrc_.EnQue<int8_t>(aLocal);

                        aLocal = vecQueSingleColumnSrc_.DeQue<int8_t>();
                        Cast(halfLocal, aLocal, RoundMode::CAST_NONE, curK);
                        PipeBarrier<PIPE_V>();
                        LocalTensor<float> mulRowLocal =
                            mulLocal[static_cast<uint64_t>(rowInGroup) * reduceRowStride];
                        Mul(halfLocal, halfLocal, bHalfLocal, curK);
                        PipeBarrier<PIPE_V>();
                        vecQueSingleColumnSrc_.FreeTensor<int8_t>(aLocal);
                        Cast(mulRowLocal, halfLocal, RoundMode::CAST_NONE, curK);
                        PipeBarrier<PIPE_V>();
                    }
                }
                LocalTensor<float> reduceDst =
                    reduceOutLocal[(row / SINGLE_COLUMN_REDUCE_ROWS) * FP32_BLOCK_ELEMS];
                if (rowsThis > 1U) {
                    uint32_t reduceShape[2] = {rowsThis, curK};
                    ReduceSum<float, Pattern::Reduce::AR, true>(reduceDst, mulLocal, reduceShape, true);
                } else {
                    LocalTensor<float> reduceTmpLocal = mulLocal[reduceRowStride];
                    ReduceSum(reduceDst, mulLocal, reduceTmpLocal, static_cast<int32_t>(curK));
                }
            }
            const uint32_t packedReduceElems =
                CeilDiv(curM, SINGLE_COLUMN_REDUCE_ROWS) * FP32_BLOCK_ELEMS;
            PipeBarrier<PIPE_V>();
            Add(accumLocal, accumLocal, reduceOutLocal, packedReduceElems);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
        WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
        const float scaleScalar = scaleScalar_;
        LocalTensor<float> outputScaleLocal = accumLocal[accumElems];
        for (uint32_t row = 0; row < curM; ++row) {
            const uint32_t reduceOffset = (row / SINGLE_COLUMN_REDUCE_ROWS) * FP32_BLOCK_ELEMS +
                row % SINGLE_COLUMN_REDUCE_ROWS;
            accumLocal.SetValue(row, accumLocal.GetValue(reduceOffset));
            outputScaleLocal.SetValue(row, scaleScalar * pertokenScaleGm_.GetValue(globalMOffset + row));
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));
        Mul(accumLocal, accumLocal, outputScaleLocal, curM);
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
        Cast(outLocal, accumLocal, RoundMode::CAST_RINT, curM);
        PipeBarrier<PIPE_V>();
        vecQueOut_.EnQue<bfloat16_t>(outLocal);
        outLocal = vecQueOut_.DeQue<bfloat16_t>();
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curM * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(outGm_[globalMOffset], outLocal, copyOutParams);
        vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
    }

    __aicore__ inline void ProcessLargeKTransBVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const uint32_t accumElems = AlignUp(baseM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<float> bFp32Local = accumLocal[accumElems];
        LocalTensor<float> mulLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE];
        LocalTensor<float> reduceTmpLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE * 2U];
        LocalTensor<float> reduceOutLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE * 3U];
        LocalTensor<half> halfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();

        for (uint32_t col = 0; col < curN; ++col) {
            const uint32_t globalCol = globalNOffset + col;
            for (uint32_t kOffset = 0; kOffset < k_; kOffset += LARGE_K_TRANS_B_VECTOR_K_TILE) {
                const uint32_t curK = Min(LARGE_K_TRANS_B_VECTOR_K_TILE, k_ - kOffset);

                LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
                DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(globalCol) * k_ + kOffset], bCopyParams, bPadParams);
                vecQueA_.EnQue<int8_t>(bLocal);

                bLocal = vecQueA_.DeQue<int8_t>();
                Cast(halfLocal, bLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                Cast(bFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                vecQueA_.FreeTensor<int8_t>(bLocal);

                for (uint32_t row = 0; row < curM; ++row) {
                    LocalTensor<int8_t> aLocal = vecQueSrc_.AllocTensor<int8_t>();
                    DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                    DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
                    DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(globalMOffset + row) * k_ + kOffset],
                        aCopyParams, aPadParams);
                    vecQueSrc_.EnQue<int8_t>(aLocal);

                    aLocal = vecQueSrc_.DeQue<int8_t>();
                    Cast(halfLocal, aLocal, RoundMode::CAST_NONE, curK);
                    PipeBarrier<PIPE_V>();
                    Cast(aFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                    PipeBarrier<PIPE_V>();
                    vecQueSrc_.FreeTensor<int8_t>(aLocal);

                    Mul(mulLocal, aFp32Local, bFp32Local, curK);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(reduceOutLocal, mulLocal, reduceTmpLocal, static_cast<int32_t>(curK));
                    SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    const uint32_t accumOffset = row * ubNAligned_ + col;
                    accumLocal.SetValue(accumOffset, accumLocal.GetValue(accumOffset) + reduceOutLocal.GetValue(0));
                }
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessLargeNVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<half> bHalfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> bFp32Local = vecLargeNB_.Get<float>();
        LocalTensor<float> tmpLocal = vecFp32Tmp_.Get<float>();
        for (uint32_t kk = 0; kk < k_; ++kk) {
            LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
            DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(kk) * n_ + globalNOffset], bCopyParams, bPadParams);
            vecQueA_.EnQue<int8_t>(bLocal);

            bLocal = vecQueA_.DeQue<int8_t>();
            Cast(bHalfLocal, bLocal, RoundMode::CAST_NONE, curN);
            PipeBarrier<PIPE_V>();
            Cast(bFp32Local, bHalfLocal, RoundMode::CAST_NONE, curN);
            PipeBarrier<PIPE_V>();
            vecQueA_.FreeTensor<int8_t>(bLocal);

            bool hasMul2 = false;
            bool tmpBusy = false;
            const bool enableSmallA = k_ > 256U;
            for (uint32_t row = 0; row < curM; ++row) {
                const float aValue = static_cast<float>(TRANS_X1 ?
                    x1Gm_.GetValue(static_cast<uint64_t>(kk) * m_ + globalMOffset + row) :
                    x1Gm_.GetValue(static_cast<uint64_t>(globalMOffset + row) * k_ + kk));
                if (aValue == 0.0f) {
                    continue;
                }
                if (enableSmallA) {
                    if (aValue == 1.0f) {
                        Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], bFp32Local, curN);
                        continue;
                    }
                    if (aValue == -1.0f) {
                        Sub(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], bFp32Local, curN);
                        continue;
                    }
                    if (aValue == 2.0f || aValue == -2.0f) {
                        if (!hasMul2) {
                            if (tmpBusy) {
                                PipeBarrier<PIPE_V>();
                                tmpBusy = false;
                            }
                            Muls(tmpLocal, bFp32Local, 2.0f, curN);
                            PipeBarrier<PIPE_V>();
                            hasMul2 = true;
                        }
                        if (aValue == 2.0f) {
                            Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                        } else {
                            Sub(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                        }
                        tmpBusy = true;
                        continue;
                    }
                }
                if (tmpBusy) {
                    PipeBarrier<PIPE_V>();
                    tmpBusy = false;
                }
                hasMul2 = false;
                Muls(tmpLocal, bFp32Local, aValue, curN);
                PipeBarrier<PIPE_V>();
                Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                tmpBusy = true;
            }
            PipeBarrier<PIPE_V>();
        }
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ScaleAccumAndStore(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal;
        if constexpr (TRANS_X1 && !TRANS_X2) {
            tokenScaleLocal = vecTokenScale_.Get<float>();
            DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
            DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
        }

        const float scaleScalar = scaleScalar_;
        for (uint32_t colOffset = 0; colOffset < curN; colOffset += baseN_) {
            const uint32_t tileN = Min(baseN_, curN - colOffset);
            LocalTensor<float> scaleLocal;
            if (scaleLen_ != 1U) {
                scaleLocal = vecQueScale_.AllocTensor<float>();
                DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(tileN * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
                DataCopyPad(scaleLocal, scaleGm_[globalNOffset + colOffset], scaleCopyParams, scalePadParams);
                vecQueScale_.EnQue<float>(scaleLocal);
                scaleLocal = vecQueScale_.DeQue<float>();
            }

            for (uint32_t row = 0; row < curM; ++row) {
                const uint32_t globalRow = globalMOffset + row;
                float tokenScale = 1.0f;
                if constexpr (TRANS_X1 && !TRANS_X2) {
                    tokenScale = tokenScaleLocal.GetValue(row);
                } else {
                    tokenScale = pertokenScaleGm_.GetValue(globalRow);
                }
                LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
                if (scaleLen_ == 1U) {
                    Muls(fp32Local, accumLocal[row * ubNAligned_ + colOffset], scaleScalar * tokenScale, tileN);
                } else {
                    Mul(fp32Local, accumLocal[row * ubNAligned_ + colOffset], scaleLocal, tileN);
                    PipeBarrier<PIPE_V>();
                    Muls(fp32Local, fp32Local, tokenScale, tileN);
                }
                PipeBarrier<PIPE_V>();

                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                Cast(outLocal, fp32Local, RoundMode::CAST_RINT, tileN);
                PipeBarrier<PIPE_V>();
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(tileN * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalRow) * n_ + globalNOffset + colOffset],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }

            if (scaleLen_ != 1U) {
                vecQueScale_.FreeTensor<float>(scaleLocal);
            }
        }
    }

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t STREAMK_SRC_BUFFER_NUM = 2;
    static constexpr uint32_t SINGLE_COLUMN_SRC_BUFFER_NUM = 1;
    static constexpr uint32_t SINGLE_COLUMN_REDUCE_ROWS = 3;
    static constexpr uint32_t SINGLE_COLUMN_PIPELINE_ROWS = 2;
    static constexpr uint32_t GM_BLOCK_BYTES = 32;
    static constexpr uint64_t WORKSPACE_ALIGN = 32;
    static constexpr uint32_t FP32_BLOCK_ELEMS = 8;
    static constexpr uint32_t BF16_BLOCK_ELEMS = 16;
    static constexpr uint32_t MATMUL_M_LIMIT = 65535;
    static constexpr uint32_t MATMUL_K_LIMIT = 65535;
    static constexpr uint32_t MATMUL_N_LIMIT = 65535;
    static constexpr uint32_t LARGE_K_MID_LIMIT = 98304;
    static constexpr uint32_t LARGE_N_SMALL_K_LIMIT = 256;
    static constexpr uint32_t THIN_M_LIMIT = 16;
    static constexpr uint32_t STREAMK_SMALL_M_LIMIT = 64;
    static constexpr uint32_t MEDIUM_M_LIMIT = 128;
    static constexpr uint32_t TALL_M_LIMIT = 256;
    static constexpr uint32_t SMALL_N_LIMIT = 64;
    static constexpr uint32_t MEDIUM_N_LIMIT = 128;
    static constexpr uint32_t STREAMK_MID_N_LIMIT = 256;
    static constexpr uint32_t K_CHUNK_LIMIT = 64512;
    static constexpr uint32_t ROW_MAJOR_MID_LARGE_K_CHUNK_LIMIT = 32768;
    static constexpr uint32_t ROW_MAJOR_SMALL_N_LARGE_K_CHUNK_LIMIT = 49152;
    static constexpr uint32_t BOTH_TRANS_LARGE_K_CHUNK_LIMIT = 65536;
    static constexpr uint32_t LARGE_N_VECTOR_K_LIMIT = 1024;
    static constexpr uint32_t LARGE_M_VECTOR_K_LIMIT = 1024;
    static constexpr uint32_t LARGE_M_VECTOR_N_LIMIT = 128;
    static constexpr uint32_t LARGE_M_TRANS_B_CACHE_ELEMS = 32768;
    static constexpr uint32_t LARGE_M_BLOCK16_OUT_ELEMS = 512;
    static constexpr uint32_t DIRECT_AIC_ROW_STEP = 32;
    static constexpr uint32_t DIRECT_AIC_LARGE_N_THIN_M_ROW_STEP = 16;
    static constexpr uint32_t DIRECT_AIC_LARGE_N_MEDIUM_M_ROW_STEP = 64;
    static constexpr uint32_t DIRECT_AIC_LARGE_M_ROW_STEP = 96;
    static constexpr uint32_t DIRECT_AIC_LARGE_M_TINY_N_ROW_STEP = 128;
    static constexpr uint32_t DIRECT_AIC_LARGE_M_SMALL_N_ROW_STEP = 128;
    static constexpr uint32_t DIRECT_AIC_TRANS_B_LARGE_N_ROW_STEP = 64;
    static constexpr uint32_t SINGLE_COLUMN_VECTOR_K_TILE = 6080;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_K_TILE = 7168;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_M_LIMIT = 128;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_N_LIMIT = 128;
    static constexpr uint32_t A_COMPACT_K_STEP = 1024;
    static constexpr uint32_t ROW_MAJOR_TALL_M_A_COMPACT_K_STEP = 8192;
    static constexpr uint32_t A_COMPACT_COPY_ROWS = 64;
    static constexpr uint32_t TRANS_B_COMPACT_K_CHUNK = 49152;
    static constexpr uint32_t STREAMK_COMPACT_A_STAGE_BYTES = 65536;
    static constexpr uint8_t STREAMK_SYNC_MODE = 0;
    static constexpr uint16_t STREAMK_AIC_SYNC_FLAG = 7;
    static constexpr uint32_t STREAMK_PARTIAL_LOAD_ROWS = 48;
    static constexpr uint32_t STREAMK_PARTIAL_STORE_ROWS = 48;
    static constexpr uint32_t STREAMK_SMALL_M_REDUCE_ROWS = 4;
    static constexpr uint32_t STREAMK_TALL_M_REDUCE_ROWS = 12;
    static constexpr uint32_t STREAMK_REDUCE_ROWS = 8;
    static constexpr uint64_t BOTH_TRANS_COMPACT_B_BYTES_LIMIT = 1UL << 20;

    const QuantBatchMatMulV3TilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueA_;
    TQue<QuePosition::VECIN, STREAMK_SRC_BUFFER_NUM> vecQueSrc_;
    TQue<QuePosition::VECIN, SINGLE_COLUMN_SRC_BUFFER_NUM> vecQueSingleColumnSrc_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueScale_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> vecQueOut_;
    TBuf<TPosition::VECCALC> vecFp32Tmp_;
    TBuf<TPosition::VECCALC> vecLargeNB_;
    TBuf<TPosition::VECCALC> vecHalfTmp_;
    TBuf<TPosition::VECCALC> vecTokenScale_;
    TBuf<TPosition::VECCALC> vecAccum_;

    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int8_t> bCompactGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<float> partialGm_;
    GlobalTensor<int32_t> int32PartialGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;

    uint32_t blockIdx_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t scaleLen_ = 0;
    float scaleScalar_ = 1.0f;
    uint32_t singleCoreM_ = 0;
    uint32_t singleCoreN_ = 0;
    uint32_t singleCoreK_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t ubNAligned_ = 0;
    uint32_t vecQueABytes_ = 0;
    uint32_t aCompactStride_ = 0;
    uint64_t matmulWorkspacePerCore_ = 0;
    uint32_t cachedTokenMOffset_ = 0xFFFFFFFFU;
    uint32_t cachedScaleNOffset_ = 0xFFFFFFFFU;
    uint64_t cWorkspacePerCore_ = 0;
    uint64_t workspacePerCore_ = 0;
    bool isMOuter_ = true;
    bool useDualAiv_ = false;
    bool vectorLargeM_ = false;
};

template <bool BOTH_TRANS_DUAL = false, bool INTERNAL_B1_ND2NZ = false>
class QuantBatchMatMulV3TransBDirectFusion {
public:
    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, BOTH_TRANS_DUAL>;
    using BGmMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using BLocalMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, true>;
    using BMatmulType = typename Conditional<INTERNAL_B1_ND2NZ, BLocalMatmulType, BGmMatmulType>::type;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    static constexpr auto QBM_CFG =
        INTERNAL_B1_ND2NZ ? QBM_CFG_MDL_INTRINSICS : QBM_CFG_DIRECT_FUSION;

    __aicore__ inline QuantBatchMatMulV3TransBDirectFusion() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t taskRatio = static_cast<uint32_t>(GetTaskRation());
        taskRatio_ = taskRatio > 0U ? taskRatio : 1U;
        subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        if (taskRatio > 0U) {
            blockIdx_ /= taskRatio;
        }
        m_ = static_cast<uint32_t>(tilingData->m);
        n_ = static_cast<uint32_t>(tilingData->n);
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        singleCoreM_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreM);
        singleCoreN_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreN);
        singleCoreK_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreK);
        baseM_ = static_cast<uint32_t>(tilingData->matmulTiling.baseM);
        baseN_ = static_cast<uint32_t>(tilingData->matmulTiling.baseN);
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        workspacePerCore_ = tilingData->workspacePerCore;
        directMode_ = BOTH_TRANS_DUAL ||
            (n_ > MATMUL_N_LIMIT && m_ <= DIRECT_M_LIMIT && k_ <= DIRECT_K_LIMIT);

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(m_) * k_);
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), static_cast<uint64_t>(n_) * k_);
        if constexpr (INTERNAL_B1_ND2NZ) {
            x2Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2),
                static_cast<uint64_t>(n_) * k_ / sizeof(int32_t));
        }
        x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        if (directMode_) {
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(m_) * n_);

        if (blockIdx_ >= usedCoreNum_ || singleCoreM_ == 0U || singleCoreN_ == 0U) {
            return;
        }
        const uint64_t slotBytes = AlignUp64(
            static_cast<uint64_t>(Max(baseM_, singleCoreM_)) * Max(baseN_, singleCoreN_) * sizeof(int32_t),
            WORKSPACE_ALIGN);
        slotStrideElems_ = slotBytes / sizeof(int32_t);
        GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
        mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coreWorkspace), 2UL * slotStrideElems_);
        partialStrideElems_ = workspacePerCore_ / sizeof(int32_t);
        partialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(userWorkspace),
            static_cast<uint64_t>(usedCoreNum_) * partialStrideElems_);
        if constexpr (BOTH_TRANS_DUAL) {
            compactMStride_ = AlignUp(baseM_, GM_BLOCK_BYTES);
            compactASlotBytes_ = AlignUp64(
                static_cast<uint64_t>(compactMStride_) * k_ * sizeof(int8_t), WORKSPACE_ALIGN);
            dualWorkspaceValid_ = 2UL * slotBytes + 2UL * compactASlotBytes_ <= workspacePerCore_;
            GM_ADDR compactBase = coreWorkspace + 2UL * slotBytes;
            aCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(compactBase),
                2UL * compactASlotBytes_);
        } else if (!directMode_) {
            compactKStride_ = AlignUp(singleCoreK_, GM_BLOCK_BYTES);
            compactABytes_ = AlignUp64(static_cast<uint64_t>(singleCoreM_) * compactKStride_, WORKSPACE_ALIGN);
            GM_ADDR compactBase = coreWorkspace + 2UL * slotBytes;
            aCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(compactBase), compactABytes_);
            bCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(compactBase + compactABytes_),
                static_cast<uint64_t>(singleCoreN_) * compactKStride_);
        }

        if ASCEND_IS_AIC {
            if constexpr (INTERNAL_B1_ND2NZ) {
                pipe->InitBuffer(b1Queue_, 1U,
                    static_cast<uint64_t>(AlignUp(n_, QBM_INT8_C0_ELEMS)) * compactKStride_ * sizeof(int8_t));
            }
            mm_.SetSubBlockIdx(0);
            mm_.Init(&tilingData->matmulTiling, pipe);
        }
        if ASCEND_IS_AIV {
            if (directMode_) {
                srcRowStride_ = AlignUp(baseN_, FP32_BLOCK_ELEMS);
                outRowStride_ = AlignUp(baseN_, BF16_BLOCK_ELEMS);
                reduceRows_ = Min(BOTH_TRANS_DUAL ? baseM_ : singleCoreM_, DIRECT_VECTOR_ROWS);
            } else {
                srcRowStride_ = AlignUp(n_, FP32_BLOCK_ELEMS);
                outRowStride_ = AlignUp(n_, BF16_BLOCK_ELEMS);
                const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
                const uint32_t kGroups = usedCoreNum_ / mBlocks;
                reduceRows_ = CeilDiv(singleCoreM_, kGroups);
            }
            reduceElems_ = reduceRows_ * srcRowStride_;
            pipe->InitBuffer(srcQue_, directMode_ ? 1U : 2U, reduceElems_ * sizeof(int32_t));
            pipe->InitBuffer(outQue_, 1, reduceRows_ * outRowStride_ * sizeof(bfloat16_t));
            pipe->InitBuffer(fp32Buf_, (directMode_ ? 1U : 2U) * reduceElems_ * sizeof(float));
            pipe->InitBuffer(scaleBuf_, srcRowStride_ * sizeof(float));
            if (!directMode_ || BOTH_TRANS_DUAL) {
                pipe->InitBuffer(stageQue_, 1, STAGE_UB_BYTES);
            }
            if (directMode_) {
                pipe->InitBuffer(tokenBuf_, AlignUp(reduceRows_, FP32_BLOCK_ELEMS) *
                    (FP32_BLOCK_ELEMS + 1U) * sizeof(float));
            }
            if (directMode_ && scaleLen_ != 1U) {
                vToMte2Event_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
            }
            if (scaleLen_ == 1U) {
                scaleScalar_ = scaleGm_.GetValue(0);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U || scaleLen_ == 0U ||
            singleCoreM_ == 0U || singleCoreN_ == 0U || (BOTH_TRANS_DUAL && !dualWorkspaceValid_)) {
            return;
        }
        if constexpr (BOTH_TRANS_DUAL) {
            ProcessBothTransDual();
        } else if (directMode_) {
            ProcessDirect();
        } else {
            ProcessSplitK();
        }
    }

private:
    __aicore__ inline void LoadInternalB1(
        LocalTensor<int8_t>& b1Local,
        uint32_t kOffset,
        uint32_t curK)
    {
        LocalTensor<int32_t> b1Local32 = b1Local.template ReinterpretCast<int32_t>();
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = n_;
        copy.dValue = curK / sizeof(int32_t);
        copy.srcDValue = k_ / sizeof(int32_t);
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(n_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        DataCopy(b1Local32, x2Int32Gm_[kOffset / sizeof(int32_t)], copy);
    }

    __aicore__ inline void StageCompactRows(
        GlobalTensor<int8_t>& dst,
        GlobalTensor<int8_t>& src,
        uint64_t srcOffset,
        uint32_t rows,
        uint32_t rowBytes,
        uint32_t dstStride,
        uint32_t srcStride)
    {
        // The original implementation fenced MTE2->MTE3 and MTE3->MTE2 for every row.
        // Pack as many independently addressed rows as fit in the 32-byte-aligned UB slab,
        // then fence once per batch.  Source/destination pitches and the logical compact
        // layout are unchanged, including the large original GM stride.
        const uint32_t ubRowStride = AlignUp(rowBytes, GM_BLOCK_BYTES);
        const uint32_t rowsPerBatch = Max(1U, STAGE_UB_BYTES / ubRowStride);
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += rowsPerBatch) {
            const uint32_t rowsThis = Min(rowsPerBatch, rows - rowBase);
            LocalTensor<int8_t> local = stageQue_.AllocTensor<int8_t>();
            if constexpr (BOTH_TRANS_DUAL) {
                // The registered dual-AIV path stages full 32-byte-aligned M rows.
                // Collapse the row-wise GM<->UB descriptors into one 2D block copy
                // whenever both pitches are exactly representable by DataCopy.
                constexpr uint64_t MAX_COPY_STRIDE_BLOCKS = 0xFFFFUL;
                const uint32_t rowBlocks = rowBytes / GM_BLOCK_BYTES;
                const uint64_t srcStrideBlocks =
                    static_cast<uint64_t>(srcStride) / GM_BLOCK_BYTES;
                const uint64_t ubStrideBlocks =
                    static_cast<uint64_t>(ubRowStride) / GM_BLOCK_BYTES;
                const uint64_t dstStrideBlocks =
                    static_cast<uint64_t>(dstStride) / GM_BLOCK_BYTES;
                const bool blockUnit2dCopy = rowBlocks > 0U &&
                    (rowBytes & (GM_BLOCK_BYTES - 1U)) == 0U &&
                    (srcStride & (GM_BLOCK_BYTES - 1U)) == 0U &&
                    (dstStride & (GM_BLOCK_BYTES - 1U)) == 0U &&
                    srcStrideBlocks >= rowBlocks && ubStrideBlocks >= rowBlocks &&
                    dstStrideBlocks >= rowBlocks &&
                    srcStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                    ubStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                    dstStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS;
                if (blockUnit2dCopy) {
                    const uint16_t blockCount = static_cast<uint16_t>(rowsThis);
                    const uint16_t blockLen = static_cast<uint16_t>(rowBlocks);
                    DataCopyParams copyIn{blockCount, blockLen,
                        static_cast<uint16_t>(srcStrideBlocks - rowBlocks),
                        static_cast<uint16_t>(ubStrideBlocks - rowBlocks)};
                    DataCopy(local,
                        src[srcOffset + static_cast<uint64_t>(rowBase) * srcStride], copyIn);
                    SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                    DataCopyParams copyOut{blockCount, blockLen,
                        static_cast<uint16_t>(ubStrideBlocks - rowBlocks),
                        static_cast<uint16_t>(dstStrideBlocks - rowBlocks)};
                    DataCopy(dst[static_cast<uint64_t>(rowBase) * dstStride], local, copyOut);
                    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                    stageQue_.FreeTensor<int8_t>(local);
                    continue;
                }
            }
            DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyExtParams copyIn{1, rowBytes, 0, 0, 0};
                DataCopyPad(local[static_cast<uint64_t>(row) * ubRowStride],
                    src[srcOffset + static_cast<uint64_t>(rowBase + row) * srcStride], copyIn, pad);
            }
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyExtParams copyOut{1, rowBytes, 0, 0, 0};
                DataCopyPad(dst[static_cast<uint64_t>(rowBase + row) * dstStride],
                    local[static_cast<uint64_t>(row) * ubRowStride], copyOut);
            }
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            stageQue_.FreeTensor<int8_t>(local);
        }
    }

    __aicore__ inline void ProcessSplitK()
    {
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t kGroups = usedCoreNum_ / mBlocks;
        const uint32_t mBlockIdx = blockIdx_ % mBlocks;
        const uint32_t kGroupIdx = blockIdx_ / mBlocks;
        const uint32_t mOffset = mBlockIdx * singleCoreM_;
        const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
        const uint32_t kSlice = AlignUp(CeilDiv(k_, kGroups), GM_BLOCK_BYTES);
        const uint32_t kOffset = kGroupIdx * kSlice;
        const uint32_t curK = Min(kSlice, k_ - kOffset);
        if ASCEND_IS_AIV {
            StageCompactRows(aCompactGm_, x1Gm_, static_cast<uint64_t>(mOffset) * k_ + kOffset,
                curM, curK, compactKStride_, k_);
            if constexpr (!INTERNAL_B1_ND2NZ) {
                StageCompactRows(bCompactGm_, x2Gm_, kOffset, n_, curK, compactKStride_, k_);
            }
            NotifyEvent<PIPE_MTE3>(STAGE_READY_FLAG);
        }
        if ASCEND_IS_AIC {
            LocalTensor<int8_t> internalB1;
            if constexpr (INTERNAL_B1_ND2NZ) {
                internalB1 = b1Queue_.AllocTensor<int8_t>();
                LoadInternalB1(internalB1, kOffset, curK);
                b1Queue_.EnQue<int8_t>(internalB1);
                internalB1 = b1Queue_.DeQue<int8_t>();
            }
            WaitEvent(STAGE_READY_FLAG);
            mm_.SetOrgShape(curM, n_, compactKStride_,
                INTERNAL_B1_ND2NZ ? curK : compactKStride_, n_);
            mm_.SetSingleShape(curM, n_, curK);
            mm_.SetTail(curM, n_, curK);
            mm_.SetTensorA(aCompactGm_, false);
            if constexpr (INTERNAL_B1_ND2NZ) {
                mm_.SetTensorB(internalB1, true);
            } else {
                mm_.SetTensorB(bCompactGm_, true);
            }
            mm_.Iterate();
            mm_.GetTensorC(partialGm_[static_cast<uint64_t>(blockIdx_) * partialStrideElems_], 0, true);
            NotifyEvent<PIPE_FIX>(V2C_PING_FLAG);
            mm_.End();
            if constexpr (INTERNAL_B1_ND2NZ) {
                b1Queue_.FreeTensor<int8_t>(internalB1);
            }
        }
        if ASCEND_IS_AIV {
            WaitEvent(V2C_PING_FLAG);
            SyncAll();
            ReduceSplitKAndStore(mBlockIdx, kGroupIdx, mBlocks, kGroups, mOffset, curM);
        }
    }

    __aicore__ inline void ReduceSplitKAndStore(
        uint32_t mBlockIdx,
        uint32_t kGroupIdx,
        uint32_t mBlocks,
        uint32_t kGroups,
        uint32_t mOffset,
        uint32_t curM)
    {
        const uint32_t baseRows = curM / kGroups;
        const uint32_t extraRows = curM - baseRows * kGroups;
        const uint32_t rowsThis = baseRows + (kGroupIdx < extraRows ? 1U : 0U);
        if (rowsThis == 0U) {
            return;
        }
        const uint32_t rowStart = kGroupIdx * baseRows + Min(kGroupIdx, extraRows);
        LocalTensor<int32_t> accumIntLocal = fp32Buf_.Get<int32_t>();
        LocalTensor<float> accumLocal = fp32Buf_.Get<float>()[reduceElems_];
        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        if (scaleLen_ != 1U) {
            DataCopyExtParams scaleCopy{1, n_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePad{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_, scaleCopy, scalePad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        }

        for (uint32_t row = 0; row < rowsThis; ++row) {
            Duplicate(accumIntLocal[row * srcRowStride_], static_cast<int32_t>(0), n_);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> firstSrc = srcQue_.AllocTensor<int32_t>();
        const uint64_t firstOffset = static_cast<uint64_t>(mBlockIdx) * partialStrideElems_ +
            static_cast<uint64_t>(rowStart) * n_;
        DataCopyExtParams copyIn{
            static_cast<uint16_t>(rowsThis), n_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(firstSrc, partialGm_[firstOffset], copyIn, pad);
        srcQue_.EnQue<int32_t>(firstSrc);
        for (uint32_t group = 0; group < kGroups; ++group) {
            LocalTensor<int32_t> srcLocal = srcQue_.DeQue<int32_t>();
            if (group + 1U < kGroups) {
                LocalTensor<int32_t> nextSrc = srcQue_.AllocTensor<int32_t>();
                const uint32_t nextSourceCore = (group + 1U) * mBlocks + mBlockIdx;
                const uint64_t nextOffset = static_cast<uint64_t>(nextSourceCore) * partialStrideElems_ +
                    static_cast<uint64_t>(rowStart) * n_;
                DataCopyPad(nextSrc, partialGm_[nextOffset], copyIn, pad);
                srcQue_.EnQue<int32_t>(nextSrc);
            }
            for (uint32_t row = 0; row < rowsThis; ++row) {
                Add(accumIntLocal[row * srcRowStride_], accumIntLocal[row * srcRowStride_],
                    srcLocal[row * srcRowStride_], n_);
            }
            PipeBarrier<PIPE_V>();
            srcQue_.FreeTensor<int32_t>(srcLocal);
        }

        for (uint32_t row = 0; row < rowsThis; ++row) {
            Cast(accumLocal[row * srcRowStride_], accumIntLocal[row * srcRowStride_], RoundMode::CAST_RINT, n_);
        }
        PipeBarrier<PIPE_V>();

        if (scaleLen_ == 1U) {
            for (uint32_t row = 0; row < rowsThis; ++row) {
                const uint32_t globalRow = mOffset + rowStart + row;
                const float tokenScale = pertokenScaleGm_.GetValue(globalRow);
                Muls(accumLocal[row * srcRowStride_], accumLocal[row * srcRowStride_],
                    scaleScalar_ * tokenScale, n_);
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t row = 0; row < rowsThis; ++row) {
                Mul(accumLocal[row * srcRowStride_], accumLocal[row * srcRowStride_], scaleLocal, n_);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < rowsThis; ++row) {
                const uint32_t globalRow = mOffset + rowStart + row;
                const float tokenScale = pertokenScaleGm_.GetValue(globalRow);
                Muls(accumLocal[row * srcRowStride_], accumLocal[row * srcRowStride_], tokenScale, n_);
            }
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<bfloat16_t> outLocal = outQue_.AllocTensor<bfloat16_t>();
        for (uint32_t row = 0; row < rowsThis; ++row) {
            Cast(outLocal[row * outRowStride_], accumLocal[row * srcRowStride_], RoundMode::CAST_RINT, n_);
        }
        PipeBarrier<PIPE_V>();
        outQue_.EnQue<bfloat16_t>(outLocal);
        outLocal = outQue_.DeQue<bfloat16_t>();
        const uint32_t globalRow = mOffset + rowStart;
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(rowsThis), n_ * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(outGm_[static_cast<uint64_t>(globalRow) * n_], outLocal, copyOut);
        outQue_.FreeTensor<bfloat16_t>(outLocal);
    }

    __aicore__ inline void WaitDualStageReady(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_STAGE_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_STAGE_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualStageReady(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(DUAL_STAGE_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(DUAL_STAGE_PING_FLAG);
        }
    }

    __aicore__ inline void WaitDualCReady(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_C_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_C_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualCReady(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(DUAL_C_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(DUAL_C_PING_FLAG);
        }
    }

    __aicore__ inline void WaitDualPostDone(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_DONE_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_DONE_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualPostDone(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(DUAL_DONE_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(DUAL_DONE_PING_FLAG);
        }
    }

    __aicore__ inline void StageBothTransATile(uint32_t mOffset, uint32_t curM, bool pong)
    {
        const uint32_t kStart = static_cast<uint32_t>(
            static_cast<uint64_t>(subBlockIdx_) * k_ / taskRatio_);
        const uint32_t kEnd = static_cast<uint32_t>(
            static_cast<uint64_t>(subBlockIdx_ + 1U) * k_ / taskRatio_);
        const uint32_t kRows = kEnd - kStart;
        const uint64_t slotOffset = (pong ? compactASlotBytes_ : 0UL) +
            static_cast<uint64_t>(kStart) * compactMStride_;
        GlobalTensor<int8_t> compactDst = aCompactGm_[slotOffset];
        StageCompactRows(compactDst, x1Gm_, static_cast<uint64_t>(kStart) * m_ + mOffset,
            kRows, curM, compactMStride_, m_);
        SignalDualStageReady(pong);
    }

    __aicore__ inline void ProcessBothTransDual()
    {
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        const uint64_t baseTasks = totalBlocks / usedCoreNum_;
        const uint64_t extraTasks = totalBlocks - baseTasks * usedCoreNum_;
        const uint64_t taskCount = baseTasks + (blockIdx_ < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        uint64_t tileLoop = 0UL;
        if ASCEND_IS_AIC {
            mm_.SetOrgShape(compactMStride_, n_, k_, k_, n_);
        }
        for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx) {
            const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
            const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
            const uint32_t blockMOffset = mBlockIdx * singleCoreM_;
            const uint32_t nOffset = nBlockIdx * singleCoreN_;
            const uint32_t blockM = Min(singleCoreM_, m_ - blockMOffset);
            const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
            const uint32_t mTileCount = CeilDiv(blockM, baseM_);
            for (uint32_t mTile = 0; mTile < mTileCount; ++mTile) {
                const uint32_t mOffset = blockMOffset + mTile * baseM_;
                const uint32_t curM = Min(baseM_, blockM - mTile * baseM_);
                const bool pong = (tileLoop & 1UL) != 0UL;
                if ASCEND_IS_AIV {
                    if (tileLoop == 0UL) {
                        StageBothTransATile(mOffset, curM, pong);
                    }
                }
                if ASCEND_IS_AIC {
                    WaitDualStageReady(pong);
                    if (tileLoop >= 2UL) {
                        WaitDualPostDone(pong);
                    }
                    mm_.SetSingleShape(curM, curN, k_);
                    mm_.SetTail(curM, curN, k_);
                    GlobalTensor<int8_t> compactA =
                        aCompactGm_[pong ? compactASlotBytes_ : 0UL];
                    mm_.SetTensorA(compactA, true);
                    mm_.SetTensorB(x2Gm_[static_cast<uint64_t>(nOffset) * k_], true);
                    mm_.Iterate();
                    mm_.GetTensorC(mmOutGm_[pong ? slotStrideElems_ : 0UL], 0, true);
                    SignalDualCReady(pong);
                }
                if ASCEND_IS_AIV {
                    const bool hasNext = mTile + 1U < mTileCount || taskIdx + 1UL < taskEnd;
                    if (hasNext) {
                        uint32_t nextMOffset = 0U;
                        uint32_t nextCurM = 0U;
                        if (mTile + 1U < mTileCount) {
                            nextMOffset = blockMOffset + (mTile + 1U) * baseM_;
                            nextCurM = Min(baseM_, blockM - (mTile + 1U) * baseM_);
                        } else {
                            const uint64_t nextTaskIdx = taskIdx + 1UL;
                            const uint32_t nextMBlockIdx = static_cast<uint32_t>(nextTaskIdx % mBlocks);
                            nextMOffset = nextMBlockIdx * singleCoreM_;
                            nextCurM = Min(singleCoreM_, m_ - nextMOffset);
                            nextCurM = Min(baseM_, nextCurM);
                        }
                        StageBothTransATile(nextMOffset, nextCurM, !pong);
                    }
                    WaitDualCReady(pong);
                    ScaleAndStore(mmOutGm_[pong ? slotStrideElems_ : 0UL],
                        mOffset, nOffset, curM, curN);
                    SignalDualPostDone(pong);
                }
                ++tileLoop;
            }
        }
        if ASCEND_IS_AIC {
            const uint64_t pending = Min(static_cast<uint32_t>(tileLoop), 2U);
            for (uint64_t i = tileLoop - pending; i < tileLoop; ++i) {
                WaitDualPostDone((i & 1UL) != 0UL);
            }
            mm_.End();
        }
    }

    __aicore__ inline void ProcessDirect()
    {
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        const uint64_t baseTasks = totalBlocks / usedCoreNum_;
        const uint64_t extraTasks = totalBlocks - baseTasks * usedCoreNum_;
        const uint64_t taskCount = baseTasks + (blockIdx_ < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        uint64_t localLoop = 0;
        for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx, ++localLoop) {
            const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
            const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
            const uint32_t mOffset = mBlockIdx * singleCoreM_;
            const uint32_t nOffset = nBlockIdx * singleCoreN_;
            const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
            const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
            const bool pong = (localLoop & 1UL) != 0UL;
            const uint64_t workspaceOffset = pong ? slotStrideElems_ : 0UL;
            ComputeTile(
                mOffset,
                nOffset,
                curM,
                curN,
                workspaceOffset,
                static_cast<uint16_t>(C2V_PING_FLAG | pong),
                static_cast<uint16_t>(V2C_PING_FLAG | pong));
        }
        End();
    }
    __aicore__ inline void ComputeTile(
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t curM,
        uint32_t curN,
        uint64_t workspaceOffset,
        uint16_t aivToAicFlag,
        uint16_t aicToAivFlag)
    {
        if ASCEND_IS_AIC {
            if (++cubeLoop_ > 2UL) {
                WaitEvent(aivToAicFlag);
            }
            mm_.SetOrgShape(m_, n_, k_, k_, n_);
            mm_.SetSingleShape(curM, curN, k_);
            mm_.SetTail(curM, curN, k_);
            mm_.SetTensorA(x1Gm_[static_cast<uint64_t>(mOffset) * k_], false);
            mm_.SetTensorB(x2Gm_[static_cast<uint64_t>(nOffset) * k_], true);
            mm_.Iterate();
            mm_.GetTensorC(mmOutGm_[workspaceOffset], 0, true);
            NotifyEvent<PIPE_FIX>(aicToAivFlag);
        }
        if ASCEND_IS_AIV {
            WaitEvent(aicToAivFlag);
            ScaleAndStore(mmOutGm_[workspaceOffset], mOffset, nOffset, curM, curN);
            NotifyEvent<PIPE_MTE2>(aivToAicFlag);
        }
    }


    __aicore__ inline void ScaleAndStore(
        GlobalTensor<int32_t> mmOut,
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        if (scaleLen_ != 1U) {
            DataCopyExtParams scaleCopy{1, curN * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePad{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[nOffset], scaleCopy, scalePad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        }
        const uint32_t rowStart = subBlockIdx_ * curM / taskRatio_;
        const uint32_t rowsForSubBlock = curM / taskRatio_ + subBlockIdx_ * (curM % taskRatio_);
        const uint32_t rowEnd = rowStart + rowsForSubBlock;
        for (uint32_t rowBlock = rowStart; rowBlock < rowEnd; rowBlock += reduceRows_) {
            const uint32_t rowsThis = Min(reduceRows_, rowEnd - rowBlock);
            const bool batchRows = curN == srcRowStride_ && curN == outRowStride_ &&
                curN % FP32_BLOCK_ELEMS == 0U;
            LocalTensor<int32_t> srcLocal = srcQue_.AllocTensor<int32_t>();
            if (batchRows) {
                DataCopyExtParams copyIn{
                    1, rowsThis * curN * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(srcLocal, mmOut[static_cast<uint64_t>(rowBlock) * curN], copyIn, pad);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    DataCopyExtParams copyIn{1, curN * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                    DataCopyPad(srcLocal[row * srcRowStride_],
                        mmOut[static_cast<uint64_t>(rowBlock + row) * curN], copyIn, pad);
                }
            }
            srcQue_.EnQue<int32_t>(srcLocal);
            srcLocal = srcQue_.DeQue<int32_t>();

            LocalTensor<float> fp32Local = fp32Buf_.Get<float>();
            if (batchRows) {
                Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(fp32Local[row * srcRowStride_], srcLocal[row * srcRowStride_],
                        RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();
            if (batchRows) {
                LocalTensor<float> tokenBrcbLocal = tokenBuf_.Get<float>();
                const uint32_t tokenBrcbElems = AlignUp(reduceRows_, FP32_BLOCK_ELEMS) * FP32_BLOCK_ELEMS;
                LocalTensor<float> tokenLocal = tokenBrcbLocal[tokenBrcbElems];
                DataCopyExtParams tokenCopy{
                    1, rowsThis * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> tokenPad{false, 0, 0, 0.0f};
                DataCopyPad(tokenLocal, pertokenScaleGm_[mOffset + rowBlock], tokenCopy, tokenPad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                const uint32_t alignedRows = AlignUp(rowsThis, FP32_BLOCK_ELEMS);
                Brcb(tokenBrcbLocal, tokenLocal, static_cast<uint8_t>(alignedRows / FP32_BLOCK_ELEMS),
                    {1, FP32_BLOCK_ELEMS});
                PipeBarrier<PIPE_V>();
                if (scaleLen_ == 1U) {
                    Muls(tokenBrcbLocal, tokenBrcbLocal, scaleScalar_, alignedRows * FP32_BLOCK_ELEMS);
                    PipeBarrier<PIPE_V>();
                } else {
                    const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                    BinaryRepeatParams channelScaleParams{1, 1, 1, rowStride, rowStride, 0};
                    for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_CHUNK_ELEMS) {
                        const uint32_t colsThis = Min(VECTOR_CHUNK_ELEMS, curN - colOffset);
                        Mul(fp32Local[colOffset], fp32Local[colOffset], scaleLocal[colOffset],
                            static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), channelScaleParams);
                    }
                    PipeBarrier<PIPE_V>();
                }
                const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                BinaryRepeatParams tokenScaleParams{1, 1, 0, rowStride, rowStride, 1};
                for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_CHUNK_ELEMS) {
                    const uint32_t colsThis = Min(VECTOR_CHUNK_ELEMS, curN - colOffset);
                    Mul(fp32Local[colOffset], fp32Local[colOffset], tokenBrcbLocal,
                        static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), tokenScaleParams);
                }
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    const float tokenScale = pertokenScaleGm_.GetValue(mOffset + rowBlock + row);
                    if (scaleLen_ == 1U) {
                        Muls(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_],
                            scaleScalar_ * tokenScale, curN);
                    } else {
                        Mul(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_], scaleLocal, curN);
                        PipeBarrier<PIPE_V>();
                        Muls(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_], tokenScale, curN);
                    }
                }
            }
            PipeBarrier<PIPE_V>();
            if (scaleLen_ != 1U) {
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
            }

            LocalTensor<bfloat16_t> outLocal = outQue_.AllocTensor<bfloat16_t>();
            if (batchRows) {
                Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(outLocal[row * outRowStride_], fp32Local[row * srcRowStride_],
                        RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();
            outQue_.EnQue<bfloat16_t>(outLocal);
            outLocal = outQue_.DeQue<bfloat16_t>();
            if (batchRows) {
                DataCopyExtParams copyOut{
                    static_cast<uint16_t>(rowsThis),
                    curN * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    (n_ - curN) * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(mOffset + rowBlock) * n_ + nOffset],
                    outLocal, copyOut);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    DataCopyExtParams copyOut{1, curN * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outGm_[static_cast<uint64_t>(mOffset + rowBlock + row) * n_ + nOffset],
                        outLocal[row * outRowStride_], copyOut);
                }
            }
            outQue_.FreeTensor<bfloat16_t>(outLocal);
            srcQue_.FreeTensor<int32_t>(srcLocal);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (cubeLoop_ > 0UL) {
                WaitEvent(C2V_PING_FLAG);
            }
            if (cubeLoop_ > 1UL) {
                WaitEvent(C2V_PONG_FLAG);
            }
            mm_.End();
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) const
    {
        return (value + factor - 1U) / factor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1UL) / align * align;
    }

    static constexpr uint32_t FP32_BLOCK_ELEMS = 8;
    static constexpr uint32_t BF16_BLOCK_ELEMS = 16;
    static constexpr uint32_t GM_BLOCK_BYTES = 32;
    static constexpr uint32_t MATMUL_N_LIMIT = 65535;
    static constexpr uint32_t DIRECT_M_LIMIT = 64;
    static constexpr uint32_t DIRECT_K_LIMIT = 1024;
    static constexpr uint32_t DIRECT_VECTOR_ROWS = 32;
    static constexpr uint32_t VECTOR_CHUNK_ELEMS = 64;
    static constexpr uint32_t STAGE_UB_BYTES = 65536;
    static constexpr uint64_t WORKSPACE_ALIGN = 32;
    static constexpr uint16_t C2V_PING_FLAG = 0x4;
    static constexpr uint16_t C2V_PONG_FLAG = 0x5;
    static constexpr uint16_t V2C_PING_FLAG = 0x6;
    static constexpr uint16_t STAGE_READY_FLAG = 0x7;
    static constexpr uint16_t DUAL_STAGE_PING_FLAG = 0x4;
    static constexpr uint16_t DUAL_STAGE_PONG_FLAG = 0x5;
    static constexpr uint16_t DUAL_C_PING_FLAG = 0x6;
    static constexpr uint16_t DUAL_C_PONG_FLAG = 0x7;
    static constexpr uint16_t DUAL_DONE_PING_FLAG = 0x8;
    static constexpr uint16_t DUAL_DONE_PONG_FLAG = 0x9;

    matmul::MatmulImpl<
        AMatmulType,
        BMatmulType,
        CMatmulType,
        BiasMatmulType,
        QBM_CFG> mm_;
    TQue<QuePosition::B1, 1> b1Queue_;
    TQue<QuePosition::VECIN, 1> srcQue_;
    TQue<QuePosition::VECIN, 1> stageQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> tokenBuf_;
    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int32_t> x2Int32Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int8_t> bCompactGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<int32_t> partialGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;

    uint32_t blockIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    uint32_t taskRatio_ = 1;
    uint32_t usedCoreNum_ = 0;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t scaleLen_ = 0;
    uint32_t singleCoreM_ = 0;
    uint32_t singleCoreN_ = 0;
    uint32_t singleCoreK_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t srcRowStride_ = 0;
    uint32_t outRowStride_ = 0;
    uint32_t reduceRows_ = 0;
    uint32_t reduceElems_ = 0;
    uint32_t compactKStride_ = 0;
    uint64_t workspacePerCore_ = 0;
    uint64_t slotStrideElems_ = 0;
    uint64_t partialStrideElems_ = 0;
    uint64_t compactABytes_ = 0;
    uint64_t compactASlotBytes_ = 0;
    uint64_t cubeLoop_ = 0;
    uint32_t compactMStride_ = 0;
    event_t vToMte2Event_ = static_cast<event_t>(0);
    float scaleScalar_ = 1.0f;
    bool directMode_ = false;
    bool dualWorkspaceValid_ = false;
};

template <bool DUAL_AIV_PIPELINE = false, bool WIDE_N_MULTI_BASE = false, bool ALGEBRAIC_TRANSPOSE = false,
    bool INTERNAL_A1_ND2NZ = false, bool INTERNAL_B_TRANSPOSE = false,
    bool ALGEBRAIC_B_TRANSPOSE = ALGEBRAIC_TRANSPOSE, bool ALGEBRAIC_FLAT_OUTPUT = false>
class QuantBatchMatMulV3TransADirectFusion {
public:
    using AGmMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using ALocalMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, true>;
    using AMatmulType = typename Conditional<INTERNAL_A1_ND2NZ, ALocalMatmulType, AGmMatmulType>::type;
    using BGmMatmulType =
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, ALGEBRAIC_B_TRANSPOSE>;
    using BLocalMatmulType =
        matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, INTERNAL_B_TRANSPOSE>;
    using BMatmulType = typename Conditional<INTERNAL_A1_ND2NZ, BLocalMatmulType, BGmMatmulType>::type;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using MatmulCallback = MatmulCallBackFunc<>;
    static constexpr auto QBM_CFG =
        INTERNAL_B_TRANSPOSE ? QBM_CFG_MDL_INTRINSICS : QBM_CFG_DIRECT_FUSION;
    // After swapping the Case10 operands, computeX1 is the complete Kx16
    // matrix and can be consumed directly as transposed A.  Unlike the
    // existing Case11 algebraic route, computeX2 is already KxN and must not
    // be transposed.
    static constexpr bool ALGEBRAIC_DIRECT_A =
        ALGEBRAIC_TRANSPOSE && !ALGEBRAIC_B_TRANSPOSE;

    __aicore__ inline QuantBatchMatMulV3TransADirectFusion() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t taskRatio = static_cast<uint32_t>(GetTaskRation());
        taskRatio_ = taskRatio > 0U ? taskRatio : 1U;
        subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        if (taskRatio > 0U) {
            blockIdx_ /= taskRatio;
        }
        originalM_ = static_cast<uint32_t>(tilingData->m);
        originalN_ = static_cast<uint32_t>(tilingData->n);
        m_ = ALGEBRAIC_TRANSPOSE ? originalN_ : originalM_;
        n_ = ALGEBRAIC_TRANSPOSE ? originalM_ : originalN_;
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        singleCoreM_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreM);
        singleCoreN_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreN);
        baseM_ = static_cast<uint32_t>(tilingData->matmulTiling.baseM);
        baseN_ = static_cast<uint32_t>(tilingData->matmulTiling.baseN);
        baseK_ = static_cast<uint32_t>(tilingData->matmulTiling.baseK);
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        workspacePerCore_ = tilingData->workspacePerCore;

        GM_ADDR computeX1 = ALGEBRAIC_TRANSPOSE ? x2 : x1;
        GM_ADDR computeX2 = ALGEBRAIC_TRANSPOSE ? x1 : x2;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(computeX1), static_cast<uint64_t>(k_) * m_);
        if constexpr (INTERNAL_A1_ND2NZ || ALGEBRAIC_DIRECT_A) {
            x1Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(computeX1),
                static_cast<uint64_t>(k_) * m_ / sizeof(int32_t));
        }
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(computeX2), static_cast<uint64_t>(k_) * n_);
        if constexpr ((INTERNAL_A1_ND2NZ && INTERNAL_B_TRANSPOSE) || ALGEBRAIC_DIRECT_A) {
            x2Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(computeX2),
                static_cast<uint64_t>(k_) * n_ / sizeof(int32_t));
        }
        x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale),
            ALGEBRAIC_TRANSPOSE ? n_ : m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(m_) * n_);

        if (blockIdx_ >= usedCoreNum_ || singleCoreM_ == 0U || singleCoreN_ == 0U) {
            return;
        }
        const uint64_t slotBytes = AlignUp64(
            static_cast<uint64_t>(Max(baseM_, singleCoreM_)) * Max(baseN_, singleCoreN_) * sizeof(int32_t),
            WORKSPACE_ALIGN);
        slotStrideElems_ = slotBytes / sizeof(int32_t);
        GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
        const uint64_t totalBlocks = static_cast<uint64_t>(CeilDiv(m_, singleCoreM_)) *
            CeilDiv(n_, singleCoreN_);
        const uint64_t algebraicTaskCapacity = ALGEBRAIC_DIRECT_A ?
            CeilDiv(totalBlocks, static_cast<uint64_t>(usedCoreNum_)) : 0UL;
        const uint64_t algebraicOutputBytes = algebraicTaskCapacity * slotBytes;
        if constexpr (ALGEBRAIC_DIRECT_A) {
            compactBSlotBytes_ = AlignUp64(
                static_cast<uint64_t>(k_) * singleCoreN_ * sizeof(int8_t), WORKSPACE_ALIGN);
            mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coreWorkspace),
                algebraicOutputBytes / sizeof(int32_t));
            bCompactGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int8_t*>(coreWorkspace + algebraicOutputBytes),
                algebraicTaskCapacity * compactBSlotBytes_);
        } else {
            mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coreWorkspace),
                2UL * slotStrideElems_);
        }
        if constexpr (DUAL_AIV_PIPELINE) {
            if constexpr (INTERNAL_A1_ND2NZ) {
                dualWorkspaceValid_ = 2UL * slotBytes <= workspacePerCore_;
            } else if constexpr (ALGEBRAIC_DIRECT_A) {
                dualWorkspaceValid_ = algebraicOutputBytes +
                    algebraicTaskCapacity * compactBSlotBytes_ <= workspacePerCore_;
            } else {
                compactMStride_ = AlignUp(Max(baseM_, singleCoreM_), GM_BLOCK_BYTES);
                compactASlotBytes_ = AlignUp64(
                    static_cast<uint64_t>(compactMStride_) * k_ * sizeof(int8_t), WORKSPACE_ALIGN);
                dualWorkspaceValid_ = 2UL * slotBytes + 2UL * compactASlotBytes_ <= workspacePerCore_;
                GM_ADDR compactBase = coreWorkspace + 2UL * slotBytes;
                aCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(compactBase),
                    2UL * compactASlotBytes_);
            }
        }

        if ASCEND_IS_AIC {
            if constexpr (ALGEBRAIC_DIRECT_A) {
                // The swapped Case10 shape is 16x256 by 256x65536.  Keep the
                // complete 4 KiB A matrix in A1 and stage one 256-column B
                // slice (64 KiB) at a time so both operands fit in L1/L0.
                pipe->InitBuffer(a1Buf_,
                    static_cast<uint64_t>(AlignUp(m_, QBM_INT8_C0_ELEMS)) * k_ * sizeof(int8_t));
                pipe->InitBuffer(b1Buf_,
                    static_cast<uint64_t>(k_) * AlignUp(singleCoreN_, QBM_INT8_C0_ELEMS) * sizeof(int8_t));
                pipe->InitBuffer(manualA2Queue_, 1U,
                    static_cast<uint64_t>(AlignUp(m_, QBM_INT8_C0_ELEMS)) * k_ * sizeof(int8_t));
                pipe->InitBuffer(manualB2Queue_, 1U,
                    static_cast<uint64_t>(k_) * AlignUp(singleCoreN_, QBM_INT8_C0_ELEMS) * sizeof(int8_t));
                pipe->InitBuffer(manualC1Queue_, MANUAL_C1_SLOTS,
                    static_cast<uint64_t>(m_) * AlignUp(singleCoreN_, MANUAL_N_TILE) * sizeof(int32_t));
            } else if constexpr (INTERNAL_A1_ND2NZ) {
                if constexpr (INTERNAL_B_TRANSPOSE) {
                    pipe->InitBuffer(a1Queue_, 1U,
                        static_cast<uint64_t>(singleCoreM_) * k_ * sizeof(int8_t));
                    pipe->InitBuffer(b1Queue_, 1U,
                        static_cast<uint64_t>(k_) * AlignUp(n_, QBM_INT8_C0_ELEMS) * sizeof(int8_t));
                } else {
                    // Case10 reuses one A1 and one resident B1 allocation for
                    // every tile. Fixed buffers avoid repeated queue state
                    // transitions without changing the local-memory layout.
                    pipe->InitBuffer(a1Buf_,
                        static_cast<uint64_t>(singleCoreM_) * k_ * sizeof(int8_t));
                    pipe->InitBuffer(b1Buf_,
                        static_cast<uint64_t>(k_) * AlignUp(n_, QBM_INT8_C0_ELEMS) * sizeof(int8_t));
                    pipe->InitBuffer(manualA2Queue_, 1U,
                        MANUAL_M_TILE * k_ * sizeof(int8_t));
                    pipe->InitBuffer(manualB2Queue_, 1U,
                        static_cast<uint64_t>(k_) * AlignUp(n_, QBM_INT8_C0_ELEMS) * sizeof(int8_t));
                    pipe->InitBuffer(manualC1Queue_, MANUAL_C1_SLOTS,
                        MANUAL_M_TILE * AlignUp(n_, MANUAL_N_TILE) * sizeof(int32_t));
                }
            }
            if constexpr ((!INTERNAL_A1_ND2NZ || INTERNAL_B_TRANSPOSE) && !ALGEBRAIC_DIRECT_A) {
                mm_.SetSubBlockIdx(0);
                mm_.Init(&tilingData->matmulTiling, pipe);
            }
        }
        if ASCEND_IS_AIV {
            srcRowStride_ = AlignUp(ALGEBRAIC_DIRECT_A ? singleCoreN_ : n_, FP32_BLOCK_ELEMS);
            outRowStride_ = AlignUp(ALGEBRAIC_DIRECT_A ? singleCoreN_ : n_, BF16_BLOCK_ELEMS);
            const uint32_t directVectorRows = INTERNAL_A1_ND2NZ && !INTERNAL_B_TRANSPOSE ?
                INTERNAL_A1_VECTOR_ROWS :
                (ALGEBRAIC_TRANSPOSE && n_ <= ALGEBRAIC_THIN_M_N_LIMIT ?
                    ALGEBRAIC_THIN_M_VECTOR_ROWS :
                    (WIDE_N_MULTI_BASE ? DUAL_MID_N_VECTOR_ROWS : DIRECT_VECTOR_ROWS));
            vectorRows_ = Min(singleCoreM_, directVectorRows);
            vectorElems_ = vectorRows_ * srcRowStride_;
            tokenStride_ = AlignUp(singleCoreM_, FP32_BLOCK_ELEMS);
            pipe->InitBuffer(srcQue_, 1, vectorElems_ * sizeof(int32_t));
            pipe->InitBuffer(outQue_, 1, vectorRows_ * outRowStride_ * sizeof(bfloat16_t));
            pipe->InitBuffer(fp32Buf_, vectorElems_ * sizeof(float));
            pipe->InitBuffer(scaleBuf_, srcRowStride_ * sizeof(float));
            pipe->InitBuffer(tokenBuf_,
                (tokenStride_ + vectorRows_ * FP32_BLOCK_ELEMS) * sizeof(float));
            if constexpr (ALGEBRAIC_TRANSPOSE && !ALGEBRAIC_FLAT_OUTPUT) {
                const uint32_t transposeElems = vectorRows_ * outRowStride_ +
                    (ALGEBRAIC_DIRECT_A ? TRANSPOSE_BLOCK * TRANSPOSE_BLOCK : 0U);
                pipe->InitBuffer(transposeBuf_, transposeElems * sizeof(bfloat16_t));
            }
            if constexpr (DUAL_AIV_PIPELINE && !INTERNAL_A1_ND2NZ) {
                if constexpr (ALGEBRAIC_DIRECT_A) {
                    pipe->InitBuffer(stageQue_, 1, STAGE_UB_BYTES);
                } else {
                    if constexpr (!WIDE_N_MULTI_BASE && !ALGEBRAIC_TRANSPOSE) {
                        pipe->InitBuffer(stageBuf_, STAGE_UB_BYTES);
                    } else {
                        pipe->InitBuffer(stageQue_, 1, STAGE_UB_BYTES);
                    }
                }
            }
            vToMte2Event_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
            if (scaleLen_ == 1U) {
                scaleScalar_ = scaleGm_.GetValue(0);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U || scaleLen_ == 0U ||
            singleCoreM_ == 0U || singleCoreN_ == 0U || (DUAL_AIV_PIPELINE && !dualWorkspaceValid_)) {
            return;
        }
        if constexpr (DUAL_AIV_PIPELINE) {
            ProcessDualAivPipeline();
            return;
        }
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        const uint64_t baseTasks = totalBlocks / usedCoreNum_;
        const uint64_t extraTasks = totalBlocks - baseTasks * usedCoreNum_;
        const uint64_t taskCount = baseTasks + (blockIdx_ < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        uint64_t localLoop = 0;
        for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx, ++localLoop) {
            const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
            const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
            const uint32_t mOffset = mBlockIdx * singleCoreM_;
            const uint32_t nOffset = nBlockIdx * singleCoreN_;
            const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
            const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
            const bool pong = (localLoop & 1UL) != 0UL;
            ComputeTile(
                mOffset,
                nOffset,
                curM,
                curN,
                pong ? slotStrideElems_ : 0UL,
                static_cast<uint16_t>(C2V_PING_FLAG | pong),
                static_cast<uint16_t>(V2C_PING_FLAG | pong));
        }
        End();
    }

private:
    __aicore__ inline void LoadInternalA1(
        LocalTensor<int8_t>& a1Local,
        uint32_t mOffset,
        uint32_t curM)
    {
        LocalTensor<int32_t> a1Local32 = a1Local.template ReinterpretCast<int32_t>();
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = k_;
        copy.dValue = curM / sizeof(int32_t);
        copy.srcDValue = m_ / sizeof(int32_t);
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(k_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        DataCopy(a1Local32, x1Int32Gm_[mOffset / sizeof(int32_t)], copy);
    }

    __aicore__ inline void LoadInternalB1(LocalTensor<int8_t>& b1Local)
    {
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = INTERNAL_B_TRANSPOSE ? n_ : k_;
        copy.dValue = INTERNAL_B_TRANSPOSE ? k_ / sizeof(int32_t) : n_;
        copy.srcDValue = INTERNAL_B_TRANSPOSE ? k_ / sizeof(int32_t) : n_;
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = INTERNAL_B_TRANSPOSE ? AlignUp(n_, 16U) : AlignUp(k_, 16U);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        if constexpr (INTERNAL_B_TRANSPOSE) {
            LocalTensor<int32_t> b1Local32 = b1Local.template ReinterpretCast<int32_t>();
            DataCopy(b1Local32, x2Int32Gm_, copy);
        } else {
            DataCopy(b1Local, x2Gm_, copy);
        }
    }

    __aicore__ inline void LoadAlgebraicA1(LocalTensor<int8_t>& a1Local)
    {
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = k_;
        copy.dValue = m_;
        copy.srcDValue = m_;
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(k_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        DataCopy(a1Local, x1Gm_, copy);
    }

    __aicore__ inline void LoadAlgebraicB1(
        LocalTensor<int8_t>& b1Local,
        uint32_t nOffset,
        uint32_t curN)
    {
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = k_;
        copy.dValue = curN;
        copy.srcDValue = n_;
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(k_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        if (n_ >= UINT16_MAX) {
            // srcDValue is encoded with a 16-bit row pitch.  Case10 has a
            // 65536-byte pitch, so copying all K rows at once would wrap the
            // pitch to zero and repeat the first row.  Keep the same NZ
            // placement while issuing one legal row copy at a time.
            copy.nValue = 1U;
            copy.srcDValue = curN;
            for (uint32_t kIndex = 0U; kIndex < k_; ++kIndex) {
                DataCopy(b1Local[static_cast<uint64_t>(kIndex) * QBM_INT8_C0_ELEMS],
                    x2Gm_[static_cast<uint64_t>(kIndex) * n_ + nOffset], copy);
            }
        } else {
            DataCopy(b1Local, x2Gm_[nOffset], copy);
        }
    }

    __aicore__ inline void LoadAlgebraicCompactB1(
        LocalTensor<int8_t>& b1Local,
        uint64_t localLoop,
        uint32_t curN)
    {
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = k_;
        copy.dValue = curN;
        copy.srcDValue = curN;
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(k_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        DataCopy(b1Local, bCompactGm_[localLoop * compactBSlotBytes_], copy);
    }

    __aicore__ inline void LoadAlgebraicDirectNzB1(
        LocalTensor<int8_t>& b1Local,
        uint32_t nOffset,
        uint32_t curN)
    {
        if ((curN & (QBM_INT8_C0_ELEMS - 1U)) != 0U ||
            (n_ & (GM_BLOCK_BYTES - 1U)) != 0U) {
            LoadAlgebraicB1(b1Local, nOffset, curN);
            return;
        }
        const uint32_t nBlocks = curN / QBM_INT8_C0_ELEMS;
        const uint32_t srcStrideBlocks = n_ / GM_BLOCK_BYTES;
        DataCopyParams copy{static_cast<uint16_t>(k_), 1U,
            static_cast<uint16_t>(srcStrideBlocks - 1U), 0U};
        for (uint32_t nBlock = 0U; nBlock < nBlocks; ++nBlock) {
            const uint64_t dstOffset = static_cast<uint64_t>(nBlock) *
                k_ * QBM_INT8_C0_ELEMS;
            const uint64_t srcOffset = static_cast<uint64_t>(nBlock) *
                QBM_INT8_C0_ELEMS + nOffset;
            DataCopy(b1Local[dstOffset], x2Gm_[srcOffset], copy);
        }
    }

    __aicore__ inline void LoadAlgebraicInt32B1(
        LocalTensor<int8_t>& b1Local,
        uint32_t nOffset,
        uint32_t curN)
    {
        LocalTensor<int32_t> b1Local32 = b1Local.template ReinterpretCast<int32_t>();
        Nd2NzParams copy;
        copy.ndNum = 1U;
        copy.nValue = k_;
        copy.dValue = curN / sizeof(int32_t);
        copy.srcDValue = n_ / sizeof(int32_t);
        copy.srcNdMatrixStride = 0U;
        copy.dstNzC0Stride = AlignUp(k_, QBM_INT8_C0_ELEMS);
        copy.dstNzNStride = 1U;
        copy.dstNzMatrixStride = 0U;
        DataCopy(b1Local32, x2Int32Gm_[nOffset / sizeof(int32_t)], copy);
    }

    __aicore__ inline void LoadAlgebraicCompactNzB1(
        LocalTensor<int8_t>& b1Local,
        uint32_t curN,
        uint64_t slot)
    {
        const uint32_t copyBlocks = k_ * curN / GM_BLOCK_BYTES;
        DataCopyParams copy{1U, static_cast<uint16_t>(copyBlocks), 0U, 0U};
        DataCopy(b1Local, bCompactGm_[slot * compactBSlotBytes_], copy);
    }

    __aicore__ inline void StageAlgebraicBNz(
        uint32_t nOffset,
        uint32_t curN,
        uint64_t slot)
    {
        LocalTensor<int8_t> local = stageQue_.AllocTensor<int8_t>();
        const uint32_t nBlocks = curN / QBM_INT8_C0_ELEMS;
        const uint32_t srcStrideBlocks = n_ / GM_BLOCK_BYTES;
        DataCopyParams gather{static_cast<uint16_t>(k_), 1U,
            static_cast<uint16_t>(srcStrideBlocks - 1U), 0U};
        for (uint32_t nBlock = 0U; nBlock < nBlocks; ++nBlock) {
            const uint64_t dstOffset = static_cast<uint64_t>(nBlock) *
                k_ * QBM_INT8_C0_ELEMS;
            const uint64_t srcOffset = static_cast<uint64_t>(nBlock) *
                QBM_INT8_C0_ELEMS + nOffset;
            DataCopy(local[dstOffset], x2Gm_[srcOffset], gather);
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        const uint32_t copyBlocks = k_ * curN / GM_BLOCK_BYTES;
        DataCopyParams publish{1U, static_cast<uint16_t>(copyBlocks), 0U, 0U};
        DataCopy(bCompactGm_[slot * compactBSlotBytes_], local, publish);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        stageQue_.FreeTensor<int8_t>(local);
    }

    __aicore__ inline void ManualInternalMatmul(
        LocalTensor<int8_t>& a1Local,
        LocalTensor<int8_t>& b1Local,
        GlobalTensor<int32_t> dst,
        uint32_t curM,
        uint32_t curN)
    {
        const uint32_t kBlocks = CeilDiv(k_, QBM_INT8_C0_ELEMS);
        const uint32_t nBlocks = CeilDiv(curN, QBM_INT8_C0_ELEMS);
        const uint32_t nFractalRows = CeilDiv(curN, MANUAL_N_TILE);
        LocalTensor<int8_t> b2Local = manualB2Queue_.AllocTensor<int8_t>();
        LoadData2dTransposeParams loadB{
            0U,
            static_cast<uint8_t>(nBlocks),
            static_cast<uint16_t>(kBlocks),
            1U,
            0U,
            inc};
        for (uint32_t kBlock = 0U; kBlock < kBlocks; ++kBlock) {
            const uint64_t b1Offset = static_cast<uint64_t>(kBlock) *
                QBM_INT8_C0_ELEMS * QBM_INT8_C0_ELEMS;
            const uint64_t b2Offset = static_cast<uint64_t>(kBlock) *
                nFractalRows * MANUAL_N_TILE * QBM_INT8_C0_ELEMS;
            LoadDataWithTranspose(b2Local[b2Offset], b1Local[b1Offset], loadB);
        }
        manualB2Queue_.EnQue<int8_t>(b2Local);
        b2Local = manualB2Queue_.DeQue<int8_t>();
        LocalTensor<int32_t> c1Slots[MANUAL_C1_SLOTS];
        uint32_t c1Count = 0U;
        for (uint32_t mOffset = 0U; mOffset < curM; mOffset += MANUAL_M_TILE) {
            const uint32_t mThis = Min(MANUAL_M_TILE, curM - mOffset);
            LocalTensor<int8_t> a2Local = manualA2Queue_.AllocTensor<int8_t>();

            const uint32_t mBlocks = CeilDiv(mThis, QBM_INT8_C0_ELEMS);
            LoadData2dTransposeParams loadA{
                0U,
                static_cast<uint8_t>(kBlocks),
                1U,
                0U,
                static_cast<uint16_t>(kBlocks - 1U),
                inc};
            const uint64_t a1MBlockBase = static_cast<uint64_t>(mOffset) * k_;
            for (uint32_t mBlock = 0U; mBlock < mBlocks; ++mBlock) {
                const uint64_t a1Offset = a1MBlockBase +
                    static_cast<uint64_t>(mBlock) * k_ * QBM_INT8_C0_ELEMS;
                const uint64_t a2Offset =
                    static_cast<uint64_t>(mBlock) * kBlocks * QBM_INT8_C0_ELEMS * QBM_INT8_C0_ELEMS;
                LoadDataWithTranspose(a2Local[a2Offset], a1Local[a1Offset], loadA);
            }

            manualA2Queue_.EnQue<int8_t>(a2Local);

            a2Local = manualA2Queue_.DeQue<int8_t>();
            LocalTensor<int32_t> c1Local = manualC1Queue_.AllocTensor<int32_t>();
            MmadParams mmParams;
            mmParams.m = mThis;
            mmParams.n = curN;
            mmParams.k = k_;
            mmParams.cmatrixInitVal = true;
            mmParams.cmatrixSource = false;
            Mmad(c1Local, a2Local, b2Local, mmParams);
            manualC1Queue_.EnQue<int32_t>(c1Local);
            manualA2Queue_.FreeTensor<int8_t>(a2Local);

            c1Local = manualC1Queue_.DeQue<int32_t>();
            FixpipeParamsV220 fix;
            fix.nSize = curN;
            fix.mSize = mThis;
            fix.srcStride = AlignUp(mThis, MANUAL_N_TILE);
            fix.dstStride = curN;
            fix.ndNum = 1U;
            fix.quantPre = QuantMode_t::NoQuant;
            Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                dst[static_cast<uint64_t>(mOffset) * curN], c1Local, fix);
            c1Slots[c1Count++] = c1Local;
        }
        // Keep every C1 source alive until all queued Fixpipe writes have
        // reached GM, then publish the complete tile to the AIV consumer.
        SetFlag<HardEvent::FIX_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::FIX_MTE2>(static_cast<event_t>(0));
        for (uint32_t slot = 0U; slot < c1Count; ++slot) {
            manualC1Queue_.FreeTensor<int32_t>(c1Slots[slot]);
        }
        manualB2Queue_.FreeTensor<int8_t>(b2Local);
    }

    __aicore__ inline void StageCompactRows(
        GlobalTensor<int8_t>& dst,
        GlobalTensor<int8_t>& src,
        uint64_t srcOffset,
        uint32_t rows,
        uint32_t rowBytes,
        uint32_t dstStride,
        uint32_t srcStride)
    {
        const uint32_t ubRowStride = AlignUp(rowBytes, GM_BLOCK_BYTES);
        const uint32_t rowsPerBatch = Max(1U, STAGE_UB_BYTES / ubRowStride);
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += rowsPerBatch) {
            const uint32_t rowsThis = Min(rowsPerBatch, rows - rowBase);
            LocalTensor<int8_t> local;
            if constexpr (!WIDE_N_MULTI_BASE && !ALGEBRAIC_TRANSPOSE) {
                local = stageBuf_.Get<int8_t>();
            } else {
                local = stageQue_.AllocTensor<int8_t>();
            }
            constexpr uint64_t MAX_COPY_STRIDE_BLOCKS = 0xFFFFUL;
            const uint32_t rowBlocks = rowBytes / GM_BLOCK_BYTES;
            const uint64_t srcStrideBlocks =
                static_cast<uint64_t>(srcStride) / GM_BLOCK_BYTES;
            const uint64_t ubStrideBlocks =
                static_cast<uint64_t>(ubRowStride) / GM_BLOCK_BYTES;
            const uint64_t dstStrideBlocks =
                static_cast<uint64_t>(dstStride) / GM_BLOCK_BYTES;
            const bool alignedBlockCopy = (rowBytes & (GM_BLOCK_BYTES - 1U)) == 0U &&
                (srcStride & (GM_BLOCK_BYTES - 1U)) == 0U &&
                (dstStride & (GM_BLOCK_BYTES - 1U)) == 0U &&
                srcStrideBlocks >= rowBlocks && ubStrideBlocks >= rowBlocks &&
                dstStrideBlocks >= rowBlocks &&
                srcStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                ubStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                dstStrideBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS;
            if (alignedBlockCopy) {
                DataCopyParams copyIn{static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(rowBlocks),
                    static_cast<uint16_t>(srcStrideBlocks - rowBlocks),
                    static_cast<uint16_t>(ubStrideBlocks - rowBlocks)};
                DataCopy(local, src[srcOffset + static_cast<uint64_t>(rowBase) * srcStride], copyIn);
                SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                DataCopyParams copyOut{static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(rowBlocks),
                    static_cast<uint16_t>(ubStrideBlocks - rowBlocks),
                    static_cast<uint16_t>(dstStrideBlocks - rowBlocks)};
                DataCopy(dst[static_cast<uint64_t>(rowBase) * dstStride], local, copyOut);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                if constexpr (WIDE_N_MULTI_BASE || ALGEBRAIC_TRANSPOSE) {
                    stageQue_.FreeTensor<int8_t>(local);
                }
                continue;
            }
            if constexpr (ALGEBRAIC_TRANSPOSE) {
                // A half-block GM pitch cannot be represented for adjacent rows by
                // block-unit DataCopy.  Same-parity rows are one full-block-aligned
                // two-row pitch apart, so gather even and odd rows independently.
                // This preserves the compact row order while replacing per-row
                // DataCopyPad issue overhead on aligned full tiles.
                const uint64_t sameParitySrcBlocks =
                    2UL * static_cast<uint64_t>(srcStride) / GM_BLOCK_BYTES;
                const uint64_t sameParityDstBlocks =
                    2UL * static_cast<uint64_t>(ubRowStride) / GM_BLOCK_BYTES;
                const uint64_t outDstBlocks =
                    static_cast<uint64_t>(dstStride) / GM_BLOCK_BYTES;
                const bool parityBlockCopy = (rowBytes & (GM_BLOCK_BYTES - 1U)) == 0U &&
                    (srcStride & (GM_BLOCK_BYTES - 1U)) == GM_BLOCK_BYTES / 2U &&
                    (dstStride & (GM_BLOCK_BYTES - 1U)) == 0U &&
                    sameParitySrcBlocks >= rowBlocks && sameParityDstBlocks >= rowBlocks &&
                    outDstBlocks >= rowBlocks &&
                    sameParitySrcBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                    sameParityDstBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS &&
                    outDstBlocks - rowBlocks <= MAX_COPY_STRIDE_BLOCKS;
                if (parityBlockCopy) {
                    const uint16_t evenRows = static_cast<uint16_t>((rowsThis + 1U) / 2U);
                    const uint16_t oddRows = static_cast<uint16_t>(rowsThis / 2U);
                    const uint16_t blockLen = static_cast<uint16_t>(rowBlocks);
                    const uint16_t srcGap =
                        static_cast<uint16_t>(sameParitySrcBlocks - rowBlocks);
                    const uint16_t localGap =
                        static_cast<uint16_t>(sameParityDstBlocks - rowBlocks);
                    const uint64_t batchSrcOffset = srcOffset +
                        static_cast<uint64_t>(rowBase) * srcStride;
                    DataCopyParams evenCopy{evenRows, blockLen, srcGap, localGap};
                    DataCopy(local, src[batchSrcOffset], evenCopy);
                    if (oddRows > 0U) {
                        DataCopyParams oddCopy{oddRows, blockLen, srcGap, localGap};
                        DataCopy(local[ubRowStride], src[batchSrcOffset + srcStride], oddCopy);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                    DataCopyParams copyOut{static_cast<uint16_t>(rowsThis), blockLen, 0,
                        static_cast<uint16_t>(outDstBlocks - rowBlocks)};
                    DataCopy(dst[static_cast<uint64_t>(rowBase) * dstStride], local, copyOut);
                    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                    if constexpr (WIDE_N_MULTI_BASE || ALGEBRAIC_TRANSPOSE) {
                        stageQue_.FreeTensor<int8_t>(local);
                    }
                    continue;
                }
            }
            DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyExtParams copyIn{1, rowBytes, 0, 0, 0};
                DataCopyPad(local[static_cast<uint64_t>(row) * ubRowStride],
                    src[srcOffset + static_cast<uint64_t>(rowBase + row) * srcStride], copyIn, pad);
            }
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
            for (uint32_t row = 0; row < rowsThis; ++row) {
                DataCopyExtParams copyOut{1, rowBytes, 0, 0, 0};
                DataCopyPad(dst[static_cast<uint64_t>(rowBase + row) * dstStride],
                    local[static_cast<uint64_t>(row) * ubRowStride], copyOut);
            }
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            if constexpr (WIDE_N_MULTI_BASE || ALGEBRAIC_TRANSPOSE) {
                stageQue_.FreeTensor<int8_t>(local);
            }
        }
    }

    __aicore__ inline void WaitDualStageReady(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_STAGE_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_STAGE_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualStageReady(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(DUAL_STAGE_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(DUAL_STAGE_PING_FLAG);
        }
    }

    __aicore__ inline void WaitDualCReady(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_C_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_C_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualCReady(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(DUAL_C_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(DUAL_C_PING_FLAG);
        }
    }

    __aicore__ inline void WaitDualPostDone(bool pong) const
    {
        if (pong) {
            CrossCoreWaitFlag(DUAL_DONE_PONG_FLAG);
        } else {
            CrossCoreWaitFlag(DUAL_DONE_PING_FLAG);
        }
    }

    __aicore__ inline void SignalDualPostDone(bool pong) const
    {
        if (pong) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(DUAL_DONE_PONG_FLAG);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(DUAL_DONE_PING_FLAG);
        }
    }

    __aicore__ inline void StageDualATile(uint32_t mOffset, uint32_t curM, bool pong)
    {
        const uint32_t kStart = static_cast<uint32_t>(
            static_cast<uint64_t>(subBlockIdx_) * k_ / taskRatio_);
        const uint32_t kEnd = static_cast<uint32_t>(
            static_cast<uint64_t>(subBlockIdx_ + 1U) * k_ / taskRatio_);
        const uint32_t kRows = kEnd - kStart;
        const uint64_t slotOffset = (pong ? compactASlotBytes_ : 0UL) +
            static_cast<uint64_t>(kStart) * compactMStride_;
        GlobalTensor<int8_t> compactDst = aCompactGm_[slotOffset];
        StageCompactRows(compactDst, x1Gm_, static_cast<uint64_t>(kStart) * m_ + mOffset,
            kRows, curM, compactMStride_, m_);
        SignalDualStageReady(pong);
    }

    __aicore__ inline void ProcessDualAivPipeline()
    {
        if constexpr (ALGEBRAIC_DIRECT_A) {
            ProcessAlgebraicDirectBatch();
            return;
        }
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        const uint64_t baseTasks = totalBlocks / usedCoreNum_;
        const uint64_t extraTasks = totalBlocks - baseTasks * usedCoreNum_;
        const uint64_t taskCount = baseTasks + (blockIdx_ < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        uint64_t localLoop = 0UL;
        LocalTensor<int8_t> internalB1;
        if ASCEND_IS_AIC {
            if constexpr (INTERNAL_A1_ND2NZ) {
                if constexpr (INTERNAL_B_TRANSPOSE) {
                    internalB1 = b1Queue_.AllocTensor<int8_t>();
                    LoadInternalB1(internalB1);
                    b1Queue_.EnQue<int8_t>(internalB1);
                    internalB1 = b1Queue_.DeQue<int8_t>();
                } else {
                    internalB1 = b1Buf_.Get<int8_t>();
                    LoadInternalB1(internalB1);
                    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                }
            }
            if constexpr (!INTERNAL_A1_ND2NZ && !ALGEBRAIC_DIRECT_A) {
                mm_.SetOrgShape(compactMStride_, n_, k_, k_, n_);
            } else if constexpr (ALGEBRAIC_DIRECT_A) {
                mm_.SetOrgShape(m_, n_, k_, k_, singleCoreN_);
            }
        }
        for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx, ++localLoop) {
            const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
            const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
            const uint32_t mOffset = mBlockIdx * singleCoreM_;
            const uint32_t nOffset = nBlockIdx * singleCoreN_;
            const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
            const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
            const bool pong = (localLoop & 1UL) != 0UL;
            if constexpr (!INTERNAL_A1_ND2NZ && !ALGEBRAIC_DIRECT_A) {
                if ASCEND_IS_AIV {
                    if (localLoop == 0UL) {
                        StageDualATile(mOffset, curM, pong);
                    }
                }
            }
            if ASCEND_IS_AIC {
                if constexpr (!INTERNAL_A1_ND2NZ && !ALGEBRAIC_DIRECT_A) {
                    WaitDualStageReady(pong);
                }
                if (localLoop >= 2UL) {
                    WaitDualPostDone(pong);
                }
                if constexpr (INTERNAL_A1_ND2NZ && INTERNAL_B_TRANSPOSE) {
                    mm_.SetOrgShape(curM, n_, k_, k_, n_);
                }
                if constexpr (!INTERNAL_A1_ND2NZ || INTERNAL_B_TRANSPOSE) {
                    mm_.SetSingleShape(curM, curN, k_);
                    mm_.SetTail(curM, curN, k_);
                }
                if constexpr (INTERNAL_A1_ND2NZ) {
                    if constexpr (INTERNAL_B_TRANSPOSE) {
                        LocalTensor<int8_t> a1Local = a1Queue_.AllocTensor<int8_t>();
                        LoadInternalA1(a1Local, mOffset, curM);
                        a1Queue_.EnQue<int8_t>(a1Local);
                        a1Local = a1Queue_.DeQue<int8_t>();
                        mm_.SetTensorA(a1Local, true);
                        mm_.SetTensorB(internalB1, true);
                        mm_.IterateAll(mmOutGm_[pong ? slotStrideElems_ : 0UL], false);
                        mm_.End();
                        a1Queue_.FreeTensor<int8_t>(a1Local);
                    } else {
                        LocalTensor<int8_t> a1Local = a1Buf_.Get<int8_t>();
                        LoadInternalA1(a1Local, mOffset, curM);
                        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                        ManualInternalMatmul(a1Local, internalB1,
                            mmOutGm_[pong ? slotStrideElems_ : 0UL], curM, curN);
                    }
                } else {
                    if constexpr (ALGEBRAIC_DIRECT_A) {
                        mm_.SetTensorA(x1Gm_, true);
                    } else {
                        GlobalTensor<int8_t> compactA = aCompactGm_[pong ? compactASlotBytes_ : 0UL];
                        mm_.SetTensorA(compactA, true);
                    }
                    const uint64_t bOffset = ALGEBRAIC_B_TRANSPOSE ?
                        static_cast<uint64_t>(nOffset) * k_ : nOffset;
                    mm_.SetTensorB(x2Gm_[bOffset], ALGEBRAIC_B_TRANSPOSE);
                    if constexpr (WIDE_N_MULTI_BASE) {
                        mm_.IterateAll(mmOutGm_[pong ? slotStrideElems_ : 0UL], false);
                        mm_.End();
                    } else {
                        mm_.Iterate();
                        mm_.GetTensorC(mmOutGm_[pong ? slotStrideElems_ : 0UL], 0,
                            !ALGEBRAIC_DIRECT_A);
                    }
                }
                SignalDualCReady(pong);
            }
            if ASCEND_IS_AIV {
                if constexpr (!INTERNAL_A1_ND2NZ && !ALGEBRAIC_DIRECT_A) {
                    if (taskIdx + 1UL < taskEnd) {
                        const uint64_t nextTaskIdx = taskIdx + 1UL;
                        const uint32_t nextMBlockIdx = static_cast<uint32_t>(nextTaskIdx % mBlocks);
                        const uint32_t nextMOffset = nextMBlockIdx * singleCoreM_;
                        const uint32_t nextCurM = Min(singleCoreM_, m_ - nextMOffset);
                        StageDualATile(nextMOffset, nextCurM, !pong);
                    }
                }
                WaitDualCReady(pong);
                if constexpr (ALGEBRAIC_TRANSPOSE) {
                    ScaleTransposeAndStore(mmOutGm_[pong ? slotStrideElems_ : 0UL],
                        mOffset, nOffset, curM, curN);
                } else {
                    ScaleAndStore(mmOutGm_[pong ? slotStrideElems_ : 0UL],
                        mOffset, nOffset, curM, curN);
                }
                SignalDualPostDone(pong);
            }
        }
        if ASCEND_IS_AIC {
            const uint64_t pending = Min(static_cast<uint32_t>(localLoop), 2U);
            for (uint64_t i = localLoop - pending; i < localLoop; ++i) {
                WaitDualPostDone((i & 1UL) != 0UL);
            }
            if constexpr (!INTERNAL_A1_ND2NZ && !WIDE_N_MULTI_BASE && !INTERNAL_B_TRANSPOSE) {
                mm_.End();
            }
            if constexpr (INTERNAL_A1_ND2NZ && INTERNAL_B_TRANSPOSE) {
                b1Queue_.FreeTensor<int8_t>(internalB1);
            }
        }
    }

    __aicore__ inline void ProcessAlgebraicDirectBatch()
    {
        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        const uint64_t baseTasks = totalBlocks / usedCoreNum_;
        const uint64_t extraTasks = totalBlocks - baseTasks * usedCoreNum_;
        const uint64_t taskCount = baseTasks + (blockIdx_ < extraTasks ? 1UL : 0UL);
        const uint64_t taskStart = static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t taskEnd = taskStart + taskCount;
        if (taskCount == 0UL) {
            return;
        }

        if ASCEND_IS_AIC {
            LocalTensor<int8_t> a1Local = a1Buf_.Get<int8_t>();
            LoadAlgebraicA1(a1Local);
            SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
            uint64_t localLoop = 0UL;
            for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx, ++localLoop) {
                const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
                const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
                const uint32_t mOffset = mBlockIdx * singleCoreM_;
                const uint32_t nOffset = nBlockIdx * singleCoreN_;
                const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
                const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
                LocalTensor<int8_t> b1Local = b1Buf_.Get<int8_t>();
                LoadAlgebraicInt32B1(b1Local, nOffset, curN);
                SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
                ManualInternalMatmul(a1Local, b1Local,
                    mmOutGm_[localLoop * slotStrideElems_], curM, curN);
            }
            SignalDualCReady(false);
        }
        if ASCEND_IS_AIV {
            WaitDualCReady(false);
            uint64_t localLoop = 0UL;
            for (uint64_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx, ++localLoop) {
                const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
                const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
                const uint32_t mOffset = mBlockIdx * singleCoreM_;
                const uint32_t nOffset = nBlockIdx * singleCoreN_;
                const uint32_t curM = Min(singleCoreM_, m_ - mOffset);
                const uint32_t curN = Min(singleCoreN_, n_ - nOffset);
                ScaleTransposeAndStore(mmOutGm_[localLoop * slotStrideElems_],
                    mOffset, nOffset, curM, curN);
            }
        }
    }

    __aicore__ inline void ComputeTile(
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t curM,
        uint32_t curN,
        uint64_t workspaceOffset,
        uint16_t aivToAicFlag,
        uint16_t aicToAivFlag)
    {
        if ASCEND_IS_AIC {
            if (++cubeLoop_ > 2UL) {
                WaitEvent(aivToAicFlag);
            }
            mm_.SetOrgShape(m_, n_, k_, k_, n_);
            mm_.SetSingleShape(curM, curN, k_);
            mm_.SetTail(curM, curN, k_);
            mm_.SetTensorA(x1Gm_[mOffset], true);
            const uint64_t bOffset = ALGEBRAIC_B_TRANSPOSE ?
                static_cast<uint64_t>(nOffset) * k_ : nOffset;
            mm_.SetTensorB(x2Gm_[bOffset], ALGEBRAIC_B_TRANSPOSE);
            mm_.Iterate();
            mm_.GetTensorC(mmOutGm_[workspaceOffset], 0, true);
            NotifyEvent<PIPE_FIX>(aicToAivFlag);
        }
        if ASCEND_IS_AIV {
            WaitEvent(aicToAivFlag);
            if constexpr (ALGEBRAIC_TRANSPOSE) {
                ScaleTransposeAndStore(mmOutGm_[workspaceOffset], mOffset, nOffset, curM, curN);
            } else {
                ScaleAndStore(mmOutGm_[workspaceOffset], mOffset, nOffset, curM, curN);
            }
            NotifyEvent<PIPE_MTE2>(aivToAicFlag);
        }
    }

    __aicore__ inline void TransposeBf16Tile(
        LocalTensor<bfloat16_t>& dstLocal,
        LocalTensor<bfloat16_t>& srcLocal,
        uint32_t rows,
        uint32_t cols)
    {
        uint64_t srcAddrList[TRANSPOSE_BLOCK];
        uint64_t dstAddrList[TRANSPOSE_BLOCK];
        TransDataTo5HDParams params;
        if constexpr (ALGEBRAIC_DIRECT_A) {
            // Use the LocalTensor-list overload so the API builds VA pointer
            // registers directly instead of loading a uint64_t stack table.
            // Keep each call on one naturally aligned 16x16 tile while this
            // path uses the wide 16x512 Case10 layout.
            LocalTensor<half> srcHalf = srcLocal.template ReinterpretCast<half>();
            LocalTensor<half> dstHalf = dstLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcTensorList[TRANSPOSE_BLOCK];
            LocalTensor<half> dstTensorList[TRANSPOSE_BLOCK];
            params.repeatTimes = 1U;
            params.srcRepStride = 0U;
            params.dstRepStride = 0U;
            for (uint32_t rowBlock = 0; rowBlock < rows; rowBlock += TRANSPOSE_BLOCK) {
                for (uint32_t colBlock = 0; colBlock < cols; colBlock += TRANSPOSE_BLOCK) {
                    for (uint32_t row = 0; row < TRANSPOSE_BLOCK; ++row) {
                        srcTensorList[row] =
                            srcHalf[static_cast<uint64_t>(rowBlock + row) * cols + colBlock];
                        dstTensorList[row] =
                            dstHalf[static_cast<uint64_t>(colBlock + row) * rows + rowBlock];
                    }
                    TransDataTo5HD<half>(dstTensorList, srcTensorList, params);
                }
            }
        } else {
            params.repeatTimes = cols / TRANSPOSE_BLOCK;
            params.srcRepStride = params.repeatTimes == 1U ? 0U : 1U;
            params.dstRepStride = params.repeatTimes == 1U ? 0U : rows;
            for (uint32_t rowBlock = 0; rowBlock < rows; rowBlock += TRANSPOSE_BLOCK) {
                for (uint32_t row = 0; row < TRANSPOSE_BLOCK; ++row) {
                    srcAddrList[row] = reinterpret_cast<uint64_t>(
                        srcLocal[static_cast<uint64_t>(rowBlock + row) * cols].GetPhyAddr());
                    dstAddrList[row] = reinterpret_cast<uint64_t>(
                        dstLocal[static_cast<uint64_t>(row) * rows + rowBlock].GetPhyAddr());
                }
                TransDataTo5HD<half>(dstAddrList, srcAddrList, params);
            }
        }
    }

    __aicore__ inline void ScaleTransposeAndStore(
        GlobalTensor<int32_t> mmOut,
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal = scaleBuf_.Get<float>();
        DataCopyExtParams tokenCopy{1, curN * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> fp32Pad{false, 0, 0, 0.0f};
        DataCopyPad(tokenScaleLocal, pertokenScaleGm_[nOffset], tokenCopy, fp32Pad);

        LocalTensor<float> channelScaleLocal = tokenBuf_.Get<float>();
        LocalTensor<float> channelBrcbLocal = channelScaleLocal[tokenStride_];
        const bool rowAligned = (curN % FP32_BLOCK_ELEMS) == 0U;
        if (rowAligned) {
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        } else {
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
        }

        const uint32_t rowStart = subBlockIdx_ * curM / taskRatio_;
        const uint32_t rowsForSubBlock =
            curM / taskRatio_ + subBlockIdx_ * (curM % taskRatio_);
        const uint32_t rowEnd = rowStart + rowsForSubBlock;
        for (uint32_t rowBlock = rowStart; rowBlock < rowEnd; rowBlock += vectorRows_) {
            const uint32_t rowsThis = Min(vectorRows_, rowEnd - rowBlock);
            const uint32_t alignedRows = AlignUp(rowsThis, TRANSPOSE_BLOCK);
            LocalTensor<int32_t> srcLocal = srcQue_.AllocTensor<int32_t>();
            DataCopyExtParams copyIn;
            if (curN <= ALGEBRAIC_THIN_M_N_LIMIT && srcRowStride_ == curN) {
                copyIn = {1, rowsThis * curN * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            } else {
                copyIn = {
                    static_cast<uint16_t>(rowsThis),
                    curN * static_cast<uint32_t>(sizeof(int32_t)),
                    0,
                    (srcRowStride_ - curN) * static_cast<uint32_t>(sizeof(int32_t)),
                    0};
            }
            DataCopyPadExtParams<int32_t> int32Pad{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOut[static_cast<uint64_t>(rowBlock) * curN], copyIn, int32Pad);

            DataCopyExtParams channelCopy{
                1, rowsThis * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(channelScaleLocal, scaleGm_[mOffset + rowBlock], channelCopy, fp32Pad);
            srcQue_.EnQue<int32_t>(srcLocal);
            if (rowAligned) {
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
            } else {
                SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(1));
                WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(1));
            }

            srcLocal = srcQue_.DeQue<int32_t>();
            LocalTensor<float> fp32Local = fp32Buf_.Get<float>();
            if (srcRowStride_ == curN) {
                Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(fp32Local[row * srcRowStride_], srcLocal[row * srcRowStride_],
                        RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();

            if (rowAligned) {
                Brcb(channelBrcbLocal, channelScaleLocal,
                    static_cast<uint8_t>(AlignUp(rowsThis, FP32_BLOCK_ELEMS) / FP32_BLOCK_ELEMS),
                    {1, FP32_BLOCK_ELEMS});
                PipeBarrier<PIPE_V>();
                const uint8_t rowStride = static_cast<uint8_t>(srcRowStride_ / FP32_BLOCK_ELEMS);
                BinaryRepeatParams channelParams{1, 1, 0, rowStride, rowStride, 1};
                BinaryRepeatParams tokenParams{1, 1, 1, rowStride, rowStride, 0};
                if (rowsThis <= MAX_VECTOR_REPEAT) {
                    for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                        const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                        Mul(fp32Local[colOffset], fp32Local[colOffset], channelBrcbLocal,
                            static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), channelParams);
                    }
                } else {
                    for (uint32_t repeatBase = 0; repeatBase < rowsThis; repeatBase += MAX_VECTOR_REPEAT) {
                        const uint32_t repeatsThis = Min(MAX_VECTOR_REPEAT, rowsThis - repeatBase);
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                            const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                            const uint32_t valueOffset = repeatBase * srcRowStride_ + colOffset;
                            Mul(fp32Local[valueOffset], fp32Local[valueOffset],
                                channelBrcbLocal[repeatBase * FP32_BLOCK_ELEMS],
                                static_cast<uint64_t>(colsThis), static_cast<uint8_t>(repeatsThis), channelParams);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                if (rowsThis <= MAX_VECTOR_REPEAT) {
                    for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                        const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                        Mul(fp32Local[colOffset], fp32Local[colOffset], tokenScaleLocal[colOffset],
                            static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), tokenParams);
                    }
                } else {
                    for (uint32_t repeatBase = 0; repeatBase < rowsThis; repeatBase += MAX_VECTOR_REPEAT) {
                        const uint32_t repeatsThis = Min(MAX_VECTOR_REPEAT, rowsThis - repeatBase);
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                            const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                            const uint32_t valueOffset = repeatBase * srcRowStride_ + colOffset;
                            Mul(fp32Local[valueOffset], fp32Local[valueOffset], tokenScaleLocal[colOffset],
                                static_cast<uint64_t>(colsThis), static_cast<uint8_t>(repeatsThis), tokenParams);
                        }
                    }
                }
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Muls(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_],
                        channelScaleLocal.GetValue(row), curN);
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Mul(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_],
                        tokenScaleLocal, curN);
                }
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = outQue_.AllocTensor<bfloat16_t>();
            if (srcRowStride_ == curN && outRowStride_ == curN) {
                Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(outLocal[row * outRowStride_], fp32Local[row * srcRowStride_],
                        RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();
            outQue_.EnQue<bfloat16_t>(outLocal);
            outLocal = outQue_.DeQue<bfloat16_t>();
            if constexpr (ALGEBRAIC_FLAT_OUTPUT) {
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
                DataCopyExtParams copyOut{
                    1U, rowsThis * curN * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
                DataCopyPad(outGm_[static_cast<uint64_t>(nOffset) * originalN_ +
                    mOffset + rowBlock], outLocal, copyOut);
                outQue_.FreeTensor<bfloat16_t>(outLocal);
                srcQue_.FreeTensor<int32_t>(srcLocal);
                continue;
            }
            LocalTensor<bfloat16_t> transposeLocal = transposeBuf_.Get<bfloat16_t>();
            if constexpr (ALGEBRAIC_DIRECT_A) {
                LocalTensor<half> compactSrc = transposeLocal.template ReinterpretCast<half>();
                LocalTensor<half> transposeDst = compactSrc[TRANSPOSE_BLOCK * TRANSPOSE_BLOCK];
                LocalTensor<bfloat16_t> transposeDstBf16 =
                    transposeDst.template ReinterpretCast<bfloat16_t>();
                LocalTensor<half> outHalf = outLocal.template ReinterpretCast<half>();
                const uint16_t srcGap = static_cast<uint16_t>(
                    outRowStride_ / BF16_BLOCK_ELEMS - 1U);
                DataCopyParams compactCopy{
                    static_cast<uint16_t>(TRANSPOSE_BLOCK), 1U, srcGap, 0U};
                for (uint32_t colBlock = 0U; colBlock < curN; colBlock += TRANSPOSE_BLOCK) {
                    DataCopy(compactSrc, outHalf[colBlock], compactCopy);
                    PipeBarrier<PIPE_V>();
                    Transpose(transposeDst[colBlock * TRANSPOSE_BLOCK], compactSrc);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
                DataCopyExtParams compactOut{
                    1U, rowsThis * curN * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
                DataCopyPad(outGm_[static_cast<uint64_t>(nOffset) * originalN_ + mOffset + rowBlock],
                    transposeDstBf16, compactOut);
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                outQue_.FreeTensor<bfloat16_t>(outLocal);
                srcQue_.FreeTensor<int32_t>(srcLocal);
                continue;
            }
            TransposeBf16Tile(transposeLocal, outLocal, alignedRows, outRowStride_);
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
            if (alignedRows == rowsThis) {
                DataCopyExtParams copyOut{
                    static_cast<uint16_t>(curN),
                    rowsThis * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    (originalN_ - rowsThis) * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(nOffset) * originalN_ + mOffset + rowBlock],
                    transposeLocal, copyOut);
            } else {
                DataCopyExtParams copyOut{
                    1, rowsThis * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                for (uint32_t col = 0; col < curN; ++col) {
                    DataCopyPad(outGm_[static_cast<uint64_t>(nOffset + col) * originalN_ +
                        mOffset + rowBlock], transposeLocal[static_cast<uint64_t>(col) * alignedRows], copyOut);
                }
            }
            outQue_.FreeTensor<bfloat16_t>(outLocal);
            srcQue_.FreeTensor<int32_t>(srcLocal);
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
    }

    __aicore__ inline void ScaleAndStore(
        GlobalTensor<int32_t> mmOut,
        uint32_t mOffset,
        uint32_t nOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const uint32_t rowStart = DUAL_AIV_PIPELINE ? subBlockIdx_ * curM / taskRatio_ : 0U;
        const uint32_t rowsForSubBlock = DUAL_AIV_PIPELINE ?
            curM / taskRatio_ + subBlockIdx_ * (curM % taskRatio_) : curM;
        const uint32_t rowEnd = rowStart + rowsForSubBlock;
        constexpr bool SLICE_TOKEN_SCALE = DUAL_AIV_PIPELINE && INTERNAL_A1_ND2NZ;
        const uint32_t tokenRowStart = SLICE_TOKEN_SCALE ? rowStart : 0U;
        const uint32_t tokenRows = SLICE_TOKEN_SCALE ? rowsForSubBlock : curM;
        LocalTensor<float> tokenLocal = tokenBuf_.Get<float>();
        DataCopyExtParams tokenCopy{
            1, tokenRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> fp32Pad{false, 0, 0, 0.0f};
        DataCopyPad(tokenLocal, pertokenScaleGm_[mOffset + tokenRowStart], tokenCopy, fp32Pad);

        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        if (scaleLen_ != 1U) {
            DataCopyExtParams scaleCopy{1, curN * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(scaleLocal, scaleGm_[nOffset], scaleCopy, fp32Pad);
        }
        const bool rowAligned = (curN % FP32_BLOCK_ELEMS) == 0U;
        if (rowAligned) {
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        } else {
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
        }

        LocalTensor<float> tokenBrcbLocal = tokenLocal[tokenStride_];
        for (uint32_t rowBlock = rowStart; rowBlock < rowEnd; rowBlock += vectorRows_) {
            const uint32_t rowsThis = Min(vectorRows_, rowEnd - rowBlock);
            LocalTensor<int32_t> srcLocal = srcQue_.AllocTensor<int32_t>();
            if (curN == n_ && srcRowStride_ == curN) {
                DataCopyExtParams copyIn{
                    1, rowsThis * curN * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(srcLocal, mmOut[static_cast<uint64_t>(rowBlock) * curN], copyIn, pad);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    DataCopyExtParams copyIn{1, curN * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                    DataCopyPad(srcLocal[row * srcRowStride_],
                        mmOut[static_cast<uint64_t>(rowBlock + row) * curN], copyIn, pad);
                }
            }
            srcQue_.EnQue<int32_t>(srcLocal);
            srcLocal = srcQue_.DeQue<int32_t>();
            LocalTensor<float> fp32Local = fp32Buf_.Get<float>();
            if (srcRowStride_ == curN) {
                Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(fp32Local[row * srcRowStride_], srcLocal[row * srcRowStride_], RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();

            if (rowAligned) {
                if (scaleLen_ != 1U) {
                    const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                    BinaryRepeatParams scaleParams{1, 1, 1, rowStride, rowStride, 0};
                    if constexpr (WIDE_N_MULTI_BASE) {
                        for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                            const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                            Mul(fp32Local[colOffset], fp32Local[colOffset], scaleLocal[colOffset],
                                static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), scaleParams);
                        }
                    } else {
                        // The repeat field is uint8_t.  Case10 deliberately
                        // keeps a full 512-row AIV half-tile in UB so the GM
                        // input/output transfers are issued once; split only
                        // the vector repeats instead of truncating 512 to 0.
                        for (uint32_t rowBase = 0; rowBase < rowsThis;
                             rowBase += MAX_VECTOR_REPEAT) {
                            const uint32_t rowsThisRepeat =
                                Min(MAX_VECTOR_REPEAT, rowsThis - rowBase);
                            Mul(fp32Local[static_cast<uint64_t>(rowBase) * srcRowStride_],
                                fp32Local[static_cast<uint64_t>(rowBase) * srcRowStride_], scaleLocal,
                                static_cast<uint64_t>(curN),
                                static_cast<uint8_t>(rowsThisRepeat), scaleParams);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                }
                const uint32_t alignedRows = AlignUp(rowsThis, FP32_BLOCK_ELEMS);
                Brcb(tokenBrcbLocal, tokenLocal[rowBlock - tokenRowStart],
                    static_cast<uint8_t>(alignedRows / FP32_BLOCK_ELEMS), {1, FP32_BLOCK_ELEMS});
                if (scaleLen_ == 1U) {
                    PipeBarrier<PIPE_V>();
                    Muls(tokenBrcbLocal, tokenBrcbLocal, scaleScalar_, alignedRows * FP32_BLOCK_ELEMS);
                }
                PipeBarrier<PIPE_V>();
                const uint8_t rowStride = static_cast<uint8_t>(curN / FP32_BLOCK_ELEMS);
                BinaryRepeatParams tokenParams{1, 1, 0, rowStride, rowStride, 1};
                if constexpr (WIDE_N_MULTI_BASE) {
                    for (uint32_t colOffset = 0; colOffset < curN; colOffset += VECTOR_MASK_ELEMS) {
                        const uint32_t colsThis = Min(VECTOR_MASK_ELEMS, curN - colOffset);
                        Mul(fp32Local[colOffset], fp32Local[colOffset], tokenBrcbLocal,
                            static_cast<uint64_t>(colsThis), static_cast<uint8_t>(rowsThis), tokenParams);
                    }
                } else {
                    for (uint32_t rowBase = 0; rowBase < rowsThis;
                         rowBase += MAX_VECTOR_REPEAT) {
                        const uint32_t rowsThisRepeat =
                            Min(MAX_VECTOR_REPEAT, rowsThis - rowBase);
                        Mul(fp32Local[static_cast<uint64_t>(rowBase) * srcRowStride_],
                            fp32Local[static_cast<uint64_t>(rowBase) * srcRowStride_],
                            tokenBrcbLocal[static_cast<uint64_t>(rowBase) * FP32_BLOCK_ELEMS],
                            static_cast<uint64_t>(curN),
                            static_cast<uint8_t>(rowsThisRepeat), tokenParams);
                    }
                }
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    const float tokenScale = tokenLocal.GetValue(rowBlock - tokenRowStart + row);
                    if (scaleLen_ == 1U) {
                        Muls(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_],
                            scaleScalar_ * tokenScale, curN);
                    } else {
                        Mul(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_], scaleLocal, curN);
                        PipeBarrier<PIPE_V>();
                        Muls(fp32Local[row * srcRowStride_], fp32Local[row * srcRowStride_], tokenScale, curN);
                    }
                }
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = outQue_.AllocTensor<bfloat16_t>();
            if (srcRowStride_ == curN && outRowStride_ == curN) {
                Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    Cast(outLocal[row * outRowStride_], fp32Local[row * srcRowStride_], RoundMode::CAST_RINT, curN);
                }
            }
            PipeBarrier<PIPE_V>();
            outQue_.EnQue<bfloat16_t>(outLocal);
            outLocal = outQue_.DeQue<bfloat16_t>();
            if (curN == n_ && outRowStride_ == curN) {
                DataCopyExtParams copyOut{
                    1, rowsThis * curN * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outGm_[static_cast<uint64_t>(mOffset + rowBlock) * n_], outLocal, copyOut);
            } else {
                for (uint32_t row = 0; row < rowsThis; ++row) {
                    DataCopyExtParams copyOut{1, curN * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outGm_[static_cast<uint64_t>(mOffset + rowBlock + row) * n_ + nOffset],
                        outLocal[row * outRowStride_], copyOut);
                }
            }
            outQue_.FreeTensor<bfloat16_t>(outLocal);
            srcQue_.FreeTensor<int32_t>(srcLocal);
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (cubeLoop_ > 0UL) {
                WaitEvent(C2V_PING_FLAG);
            }
            if (cubeLoop_ > 1UL) {
                WaitEvent(C2V_PONG_FLAG);
            }
            mm_.End();
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) const
    {
        return (value + factor - 1U) / factor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1UL) / align * align;
    }

    static constexpr uint32_t FP32_BLOCK_ELEMS = 8;
    static constexpr uint32_t BF16_BLOCK_ELEMS = 16;
    static constexpr uint32_t GM_BLOCK_BYTES = 32;
    static constexpr uint32_t DIRECT_VECTOR_ROWS = 192;
    static constexpr uint32_t INTERNAL_A1_VECTOR_ROWS = 512;
    static constexpr uint32_t DUAL_MID_N_VECTOR_ROWS = 64;
    static constexpr uint32_t ALGEBRAIC_THIN_M_N_LIMIT = 16;
    static constexpr uint32_t ALGEBRAIC_THIN_M_VECTOR_ROWS = 512;
    static constexpr uint32_t MAX_VECTOR_REPEAT = 255;
    static constexpr uint32_t VECTOR_MASK_ELEMS = 64;
    static constexpr uint32_t TRANSPOSE_BLOCK = 16;
    static constexpr uint32_t STAGE_UB_BYTES = 65536;
    static constexpr uint32_t MANUAL_M_TILE = 256;
    static constexpr uint32_t MANUAL_N_TILE = 16;
    static constexpr uint32_t MANUAL_C1_SLOTS = 4;
    static constexpr uint64_t WORKSPACE_ALIGN = 32;
    static constexpr uint16_t C2V_PING_FLAG = 0x4;
    static constexpr uint16_t C2V_PONG_FLAG = 0x5;
    static constexpr uint16_t V2C_PING_FLAG = 0x6;
    static constexpr uint16_t DUAL_STAGE_PING_FLAG = 0x4;
    static constexpr uint16_t DUAL_STAGE_PONG_FLAG = 0x5;
    static constexpr uint16_t DUAL_C_PING_FLAG = 0x6;
    static constexpr uint16_t DUAL_C_PONG_FLAG = 0x7;
    static constexpr uint16_t DUAL_DONE_PING_FLAG = 0x8;
    static constexpr uint16_t DUAL_DONE_PONG_FLAG = 0x9;

    matmul::MatmulImpl<
        AMatmulType,
        BMatmulType,
        CMatmulType,
        BiasMatmulType,
        QBM_CFG,
        MatmulCallback> mm_;
    TQue<QuePosition::A1, 1> a1Queue_;
    TQue<QuePosition::B1, 1> b1Queue_;
    TBuf<TPosition::A1> a1Buf_;
    TBuf<TPosition::B1> b1Buf_;
    TQue<TPosition::A2, 1> manualA2Queue_;
    TQue<TPosition::B2, 1> manualB2Queue_;
    TQue<TPosition::CO1, MANUAL_C1_SLOTS> manualC1Queue_;
    TQue<QuePosition::VECIN, 1> srcQue_;
    TQue<QuePosition::VECIN, 1> stageQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> stageBuf_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> scaleBuf_;
    TBuf<TPosition::VECCALC> tokenBuf_;
    TBuf<TPosition::VECCALC> transposeBuf_;
    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int32_t> x1Int32Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int32_t> x2Int32Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int8_t> bCompactGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;

    uint32_t blockIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    uint32_t taskRatio_ = 1;
    uint32_t usedCoreNum_ = 0;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t originalM_ = 0;
    uint32_t originalN_ = 0;
    uint32_t k_ = 0;
    uint32_t scaleLen_ = 0;
    uint32_t singleCoreM_ = 0;
    uint32_t singleCoreN_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t baseK_ = 0;
    uint32_t srcRowStride_ = 0;
    uint32_t outRowStride_ = 0;
    uint32_t vectorRows_ = 0;
    uint32_t vectorElems_ = 0;
    uint32_t tokenStride_ = 0;
    uint32_t compactMStride_ = 0;
    uint64_t workspacePerCore_ = 0;
    uint64_t slotStrideElems_ = 0;
    uint64_t compactASlotBytes_ = 0;
    uint64_t compactBSlotBytes_ = 0;
    uint64_t cubeLoop_ = 0;
    event_t vToMte2Event_ = static_cast<event_t>(0);
    float scaleScalar_ = 1.0f;
    bool dualWorkspaceValid_ = false;
};

// Case05 structural path: relay each original-stride A slice through compact
// GM on AIV, convert compact ND to local NZ A1 on AIC, compute one int32
// partial per K owner, then reuse the proven source-order FP32 reduction.
template <bool IS_MID_N = false, bool IS_BOTH_TRANS_LOCAL_B1 = false, bool IS_SINGLE_ROW_LOCAL_A1 = false,
    bool IS_TRANS_A_LOCAL_B1 = false, bool IS_ALGEBRAIC_TRANSPOSE = false,
    bool IS_SINGLE_COLUMN_CUBE = false>
class QuantBatchMatMulV3Case05AicLocalA1 {
public:
    using ALocalMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, false>;
    using ATransGlobalMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using AMatmulType = typename Conditional<IS_BOTH_TRANS_LOCAL_B1 || IS_TRANS_A_LOCAL_B1,
        ATransGlobalMatmulType, ALocalMatmulType>::type;
    using BGlobalMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BBothTransMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, true>;
    using BLocalMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, false>;
    using BNonTransMatmulType = typename Conditional<IS_TRANS_A_LOCAL_B1,
        BLocalMatmulType, BGlobalMatmulType>::type;
    using BMatmulType = typename Conditional<IS_BOTH_TRANS_LOCAL_B1,
        BBothTransMatmulType, BNonTransMatmulType>::type;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

    __aicore__ inline QuantBatchMatMulV3Case05AicLocalA1() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        taskRatio_ = static_cast<uint32_t>(GetTaskRation());
        taskRatio_ = taskRatio_ > 0U ? taskRatio_ : 1U;
        subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        if (GetTaskRation() > 0) {
            blockIdx_ /= taskRatio_;
        }
        originalM_ = static_cast<uint32_t>(tilingData->m);
        originalN_ = static_cast<uint32_t>(tilingData->n);
        m_ = IS_ALGEBRAIC_TRANSPOSE ? originalN_ : originalM_;
        n_ = IS_ALGEBRAIC_TRANSPOSE ? originalM_ : originalN_;
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        baseN_ = static_cast<uint32_t>(tilingData->matmulTiling.baseN);
        singleCoreK_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreK);
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        workspacePerCore_ = tilingData->workspacePerCore;

        if constexpr (IS_ALGEBRAIC_TRANSPOSE && IS_BOTH_TRANS_LOCAL_B1) {
            x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2),
                static_cast<uint64_t>(k_) * m_);
            x2Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1),
                static_cast<uint64_t>(n_) * k_ / sizeof(int32_t));
        } else if constexpr (IS_ALGEBRAIC_TRANSPOSE) {
            x1Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2),
                static_cast<uint64_t>(m_) * k_ / sizeof(int32_t));
        } else if constexpr (IS_BOTH_TRANS_LOCAL_B1) {
            x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(k_) * m_);
            x2Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2),
                static_cast<uint64_t>(n_) * k_ / sizeof(int32_t));
        } else if constexpr (IS_MID_N || IS_SINGLE_ROW_LOCAL_A1 || IS_SINGLE_COLUMN_CUBE) {
            x1Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1),
                static_cast<uint64_t>(m_) * k_ / sizeof(int32_t));
        } else {
            x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(m_) * k_);
        }
        if constexpr (IS_ALGEBRAIC_TRANSPOSE) {
            x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(k_) * n_);
        } else if constexpr (!IS_BOTH_TRANS_LOCAL_B1) {
            x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), static_cast<uint64_t>(k_) * n_);
        }
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), originalM_);
        outGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(originalM_) * originalN_);
        partialStrideElems_ = workspacePerCore_ / sizeof(int32_t);
        partialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace),
            static_cast<uint64_t>(usedCoreNum_) * partialStrideElems_);

        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U || singleCoreK_ == 0U) {
            return;
        }
        compactKStride_ = (IS_MID_N || IS_BOTH_TRANS_LOCAL_B1 || IS_SINGLE_ROW_LOCAL_A1 ||
            IS_TRANS_A_LOCAL_B1 || IS_ALGEBRAIC_TRANSPOSE || IS_SINGLE_COLUMN_CUBE) ?
            AlignUp(CeilDiv(k_, usedCoreNum_), GM_BLOCK_BYTES) : singleCoreK_;
        if constexpr (!IS_MID_N && !IS_BOTH_TRANS_LOCAL_B1 && !IS_SINGLE_ROW_LOCAL_A1 &&
            !IS_TRANS_A_LOCAL_B1 && !IS_ALGEBRAIC_TRANSPOSE && !IS_SINGLE_COLUMN_CUBE) {
            const uint64_t partialBytes = AlignUp64(
                static_cast<uint64_t>(m_) * n_ * sizeof(int32_t), GM_BLOCK_BYTES);
            GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
            aCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(coreWorkspace + 2UL * partialBytes),
                static_cast<uint64_t>(m_) * compactKStride_);
        }
        if ASCEND_IS_AIC {
            if constexpr (IS_BOTH_TRANS_LOCAL_B1) {
                pipe->InitBuffer(b1Queue_, 1U,
                    BOTH_TRANS_LOCAL_N_TILE * BOTH_TRANS_LOCAL_B1_K_CHUNK * sizeof(int8_t));
            } else if constexpr (IS_TRANS_A_LOCAL_B1) {
                pipe->InitBuffer(b1Queue_, 1U,
                    TRANS_A_LOCAL_B1_K_SLICE * TRANS_A_LOCAL_B1_MAX_N * sizeof(int8_t));
            } else {
                const uint32_t alignedM = AlignUp(m_, 16U);
                pipe->InitBuffer(a1Queue_, 1U, alignedM * LOCAL_A1_K_CHUNK * sizeof(int8_t));
            }
            mm_.SetSubBlockIdx(0);
            mm_.Init(&tilingData->matmulTiling, pipe);
        }
        if ASCEND_IS_AIV {
            reduceElems_ = REDUCE_ROWS * n_;
            if constexpr (!IS_MID_N && !IS_BOTH_TRANS_LOCAL_B1 && !IS_SINGLE_ROW_LOCAL_A1 &&
                !IS_TRANS_A_LOCAL_B1 && !IS_ALGEBRAIC_TRANSPOSE && !IS_SINGLE_COLUMN_CUBE) {
                pipe->InitBuffer(stageQueue_, 1U, STAGE_UB_BYTES);
            }
            pipe->InitBuffer(srcQueue_, 2U, reduceElems_ * sizeof(int32_t));
            pipe->InitBuffer(fp32Buffer_, 2U * reduceElems_ * sizeof(float));
            pipe->InitBuffer(scaleBuffer_,
                (IS_SINGLE_COLUMN_CUBE ? REDUCE_ROWS : n_) * sizeof(float));
            const uint32_t outElems = IS_ALGEBRAIC_TRANSPOSE && IS_BOTH_TRANS_LOCAL_B1 ? n_ :
                (IS_ALGEBRAIC_TRANSPOSE ? TRANSPOSE_BLOCK * n_ : reduceElems_);
            pipe->InitBuffer(outQueue_, 1U, outElems * sizeof(bfloat16_t));
            if constexpr (IS_ALGEBRAIC_TRANSPOSE && !IS_BOTH_TRANS_LOCAL_B1) {
                pipe->InitBuffer(transposeBuffer_, TRANSPOSE_BLOCK * n_ * sizeof(bfloat16_t));
            }
            if (scaleLen_ == 1U) {
                scaleScalar_ = scaleGm_.GetValue(0U);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U || scaleLen_ == 0U ||
            singleCoreK_ == 0U) {
            return;
        }
        const uint32_t kOffset = blockIdx_ * compactKStride_;
        const uint32_t curK = kOffset < k_ ? Min(compactKStride_, k_ - kOffset) : 0U;
        if ASCEND_IS_AIV {
            if constexpr (!IS_MID_N && !IS_BOTH_TRANS_LOCAL_B1 && !IS_SINGLE_ROW_LOCAL_A1 &&
                !IS_TRANS_A_LOCAL_B1 && !IS_ALGEBRAIC_TRANSPOSE && !IS_SINGLE_COLUMN_CUBE) {
                StageCompactA(kOffset, curK);
            }
        }
        if ASCEND_IS_AIC {
            if constexpr (IS_BOTH_TRANS_LOCAL_B1) {
                ProduceBothTransPartial(kOffset, curK);
            } else if constexpr (IS_TRANS_A_LOCAL_B1) {
                ProduceTransALocalB1Partial(kOffset, curK);
            } else {
                ProducePartial(curK);
            }
        }
        if ASCEND_IS_AIV {
            WaitEvent(PARTIAL_READY_FLAG);
            SyncAll();
            ReducePartialsAndStore();
        }
    }

private:
    __aicore__ inline void StageCompactA(uint32_t kOffset, uint32_t curK)
    {
        uint32_t chunkIndex = 0U;
        for (uint32_t localK = 0U; localK < curK; localK += LOCAL_A1_K_CHUNK, ++chunkIndex) {
            const uint32_t chunkK = Min(LOCAL_A1_K_CHUNK, curK - localK);
            const uint32_t ubRowStride = AlignUp(chunkK, GM_BLOCK_BYTES);
            const uint32_t rowsPerChunk = STAGE_UB_BYTES / ubRowStride;
            for (uint32_t rowBase = 0U; rowBase < m_; rowBase += rowsPerChunk) {
                const uint32_t rowsThis = Min(rowsPerChunk, m_ - rowBase);
                LocalTensor<int8_t> local = stageQueue_.AllocTensor<int8_t>();
                DataCopyParams copyIn{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(chunkK / GM_BLOCK_BYTES),
                    static_cast<uint16_t>((k_ - chunkK) / GM_BLOCK_BYTES),
                    static_cast<uint16_t>((ubRowStride - chunkK) / GM_BLOCK_BYTES)};
                DataCopy(local,
                    x1Gm_[static_cast<uint64_t>(rowBase) * k_ + kOffset + localK], copyIn);
                SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                DataCopyParams copyOut{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(chunkK / GM_BLOCK_BYTES),
                    static_cast<uint16_t>((ubRowStride - chunkK) / GM_BLOCK_BYTES),
                    static_cast<uint16_t>((compactKStride_ - chunkK) / GM_BLOCK_BYTES)};
                DataCopy(aCompactGm_[static_cast<uint64_t>(rowBase) * compactKStride_ + localK],
                    local, copyOut);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                stageQueue_.FreeTensor<int8_t>(local);
            }
            NotifyEvent<PIPE_MTE3>(static_cast<uint16_t>(STAGE_READY_FLAG_BASE + chunkIndex));
        }
    }

    __aicore__ inline void ProducePartial(uint32_t curK)
    {
        GlobalTensor<int32_t> dstPartial = partialGm_[static_cast<uint64_t>(blockIdx_) * partialStrideElems_];
        bool firstChunk = true;
        uint32_t chunkIndex = 0U;
        for (uint32_t localK = 0U; localK < curK; localK += LOCAL_A1_K_CHUNK, ++chunkIndex) {
            const uint32_t chunkK = Min(LOCAL_A1_K_CHUNK, curK - localK);
            if constexpr (!IS_MID_N && !IS_SINGLE_ROW_LOCAL_A1 && !IS_ALGEBRAIC_TRANSPOSE &&
                !IS_SINGLE_COLUMN_CUBE) {
                WaitEvent(static_cast<uint16_t>(STAGE_READY_FLAG_BASE + chunkIndex));
            }
            LocalTensor<int8_t> a1Local = a1Queue_.AllocTensor<int8_t>();
            const uint32_t alignedM = AlignUp(m_, 16U);
            Nd2NzParams a1Copy;
            a1Copy.ndNum = 1U;
            a1Copy.nValue = m_;
            a1Copy.dValue = (IS_MID_N || IS_SINGLE_ROW_LOCAL_A1 || IS_ALGEBRAIC_TRANSPOSE ||
                IS_SINGLE_COLUMN_CUBE) ?
                chunkK / sizeof(int32_t) : chunkK;
            a1Copy.srcDValue = (IS_MID_N || IS_SINGLE_ROW_LOCAL_A1 || IS_ALGEBRAIC_TRANSPOSE ||
                IS_SINGLE_COLUMN_CUBE) ?
                k_ / sizeof(int32_t) : compactKStride_;
            a1Copy.srcNdMatrixStride = 0U;
            a1Copy.dstNzC0Stride = alignedM;
            a1Copy.dstNzNStride = 1U;
            a1Copy.dstNzMatrixStride = 0U;
            if constexpr (IS_MID_N || IS_SINGLE_ROW_LOCAL_A1 || IS_ALGEBRAIC_TRANSPOSE ||
                IS_SINGLE_COLUMN_CUBE) {
                LocalTensor<int32_t> a1Local32 = a1Local.template ReinterpretCast<int32_t>();
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(blockIdx_) * compactKStride_ + localK) / sizeof(int32_t);
                DataCopy(a1Local32, x1Int32Gm_[sourceOffset], a1Copy);
            } else {
                DataCopy(a1Local, aCompactGm_[localK], a1Copy);
            }
            a1Queue_.EnQue<int8_t>(a1Local);
            a1Local = a1Queue_.DeQue<int8_t>();

            mm_.SetOrgShape(m_, n_, chunkK, k_, n_);
            mm_.SetSingleShape(m_, n_, chunkK);
            mm_.SetTail(m_, n_, chunkK);
            mm_.SetTensorA(a1Local, false);
            mm_.SetTensorB(x2Gm_[static_cast<uint64_t>(blockIdx_ * compactKStride_ + localK) * n_], false);
            mm_.Iterate();
            mm_.GetTensorC(dstPartial, firstChunk ? 0U : 1U, true);
            firstChunk = false;
            a1Queue_.FreeTensor<int8_t>(a1Local);
        }
        mm_.End();
        NotifyEvent<PIPE_FIX>(PARTIAL_READY_FLAG);
    }

    __aicore__ inline void ProduceBothTransPartial(uint32_t kOffset, uint32_t curK)
    {
        GlobalTensor<int32_t> dstPartial = partialGm_[static_cast<uint64_t>(blockIdx_) * partialStrideElems_];
        const uint32_t nTile = Min(BOTH_TRANS_LOCAL_N_TILE, baseN_);
        for (uint32_t nOffset = 0U; nOffset < n_; nOffset += nTile) {
            const uint32_t curN = Min(nTile, n_ - nOffset);
            const uint64_t dstTileOffset = static_cast<uint64_t>(nOffset) * m_;
            if (baseN_ <= 64U) {
                LocalTensor<int8_t> b1Local = b1Queue_.AllocTensor<int8_t>();
                LocalTensor<int32_t> b1Local32 = b1Local.template ReinterpretCast<int32_t>();
                Nd2NzParams b1Copy;
                b1Copy.ndNum = 1U;
                b1Copy.nValue = curN;
                b1Copy.dValue = curK / sizeof(int32_t);
                b1Copy.srcDValue = k_ / sizeof(int32_t);
                b1Copy.srcNdMatrixStride = 0U;
                b1Copy.dstNzC0Stride = AlignUp(curN, 16U);
                b1Copy.dstNzNStride = 1U;
                b1Copy.dstNzMatrixStride = 0U;
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(nOffset) * k_ + kOffset) / sizeof(int32_t);
                DataCopy(b1Local32, x2Int32Gm_[sourceOffset], b1Copy);
                b1Queue_.EnQue<int8_t>(b1Local);
                b1Local = b1Queue_.DeQue<int8_t>();

                mm_.SetOrgShape(m_, curN, k_, curK, curN);
                mm_.SetSingleShape(m_, curN, curK);
                mm_.SetTail(m_, curN, curK);
                mm_.SetTensorA(x1Gm_[static_cast<uint64_t>(kOffset) * m_], true);
                mm_.SetTensorB(b1Local, true);
                mm_.Iterate();
                mm_.GetTensorC(dstPartial[dstTileOffset], 0U, true);
                mm_.End();
                b1Queue_.FreeTensor<int8_t>(b1Local);
                continue;
            }
            bool firstChunk = true;
            for (uint32_t localK = 0U; localK < curK; localK += BOTH_TRANS_LOCAL_B1_K_CHUNK) {
                const uint32_t chunkK = Min(BOTH_TRANS_LOCAL_B1_K_CHUNK, curK - localK);
                LocalTensor<int8_t> b1Local = b1Queue_.AllocTensor<int8_t>();
                LocalTensor<int32_t> b1Local32 = b1Local.template ReinterpretCast<int32_t>();
                Nd2NzParams b1Copy;
                b1Copy.ndNum = 1U;
                b1Copy.nValue = curN;
                b1Copy.dValue = chunkK / sizeof(int32_t);
                b1Copy.srcDValue = k_ / sizeof(int32_t);
                b1Copy.srcNdMatrixStride = 0U;
                b1Copy.dstNzC0Stride = AlignUp(curN, 16U);
                b1Copy.dstNzNStride = 1U;
                b1Copy.dstNzMatrixStride = 0U;
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(nOffset) * k_ + kOffset + localK) / sizeof(int32_t);
                DataCopy(b1Local32, x2Int32Gm_[sourceOffset], b1Copy);
                b1Queue_.EnQue<int8_t>(b1Local);
                b1Local = b1Queue_.DeQue<int8_t>();

                mm_.SetOrgShape(m_, curN, k_, chunkK, curN);
                mm_.SetSingleShape(m_, curN, chunkK);
                mm_.SetTail(m_, curN, chunkK);
                mm_.SetTensorA(x1Gm_[static_cast<uint64_t>(kOffset + localK) * m_], true);
                mm_.SetTensorB(b1Local, true);
                mm_.Iterate(!firstChunk);
                firstChunk = false;
                b1Queue_.FreeTensor<int8_t>(b1Local);
            }
            mm_.GetTensorC(dstPartial[dstTileOffset], 0U, true);
            mm_.End();
        }
        NotifyEvent<PIPE_FIX>(PARTIAL_READY_FLAG);
    }

    __aicore__ inline void ProduceTransALocalB1Partial(uint32_t kOffset, uint32_t curK)
    {
        GlobalTensor<int32_t> dstPartial = partialGm_[static_cast<uint64_t>(blockIdx_) * partialStrideElems_];
        LocalTensor<int8_t> b1Local = b1Queue_.AllocTensor<int8_t>();
        Nd2NzParams b1Copy;
        b1Copy.ndNum = 1U;
        b1Copy.nValue = curK;
        b1Copy.dValue = n_;
        b1Copy.srcDValue = n_;
        b1Copy.srcNdMatrixStride = 0U;
        b1Copy.dstNzC0Stride = AlignUp(curK, 16U);
        b1Copy.dstNzNStride = 1U;
        b1Copy.dstNzMatrixStride = 0U;
        DataCopy(b1Local, x2Gm_[static_cast<uint64_t>(kOffset) * n_], b1Copy);
        b1Queue_.EnQue<int8_t>(b1Local);
        b1Local = b1Queue_.DeQue<int8_t>();

        mm_.SetOrgShape(m_, n_, k_, curK, n_);
        mm_.SetSingleShape(m_, n_, curK);
        mm_.SetTail(m_, n_, curK);
        mm_.SetTensorA(x1Gm_[static_cast<uint64_t>(kOffset) * m_], true);
        mm_.SetTensorB(b1Local, false);
        mm_.Iterate();
        mm_.GetTensorC(dstPartial, 0U, true);
        mm_.End();
        b1Queue_.FreeTensor<int8_t>(b1Local);
        NotifyEvent<PIPE_FIX>(PARTIAL_READY_FLAG);
    }

    __aicore__ inline void LoadPartial(
        LocalTensor<int32_t>& local,
        uint32_t sourceCore,
        uint32_t rowStart,
        uint32_t rowsThis)
    {
        const uint64_t offset = static_cast<uint64_t>(sourceCore) * partialStrideElems_ +
            static_cast<uint64_t>(rowStart) * n_;
        if constexpr (IS_BOTH_TRANS_LOCAL_B1) {
            const uint64_t sourceBase = static_cast<uint64_t>(sourceCore) * partialStrideElems_;
            const uint32_t nTile = Min(BOTH_TRANS_LOCAL_N_TILE, baseN_);
            for (uint32_t nOffset = 0U; nOffset < n_; nOffset += nTile) {
                const uint32_t curN = Min(nTile, n_ - nOffset);
                const uint64_t tileOffset = static_cast<uint64_t>(nOffset) * m_ +
                    static_cast<uint64_t>(rowStart) * curN;
                DataCopyExtParams tileCopy{
                    static_cast<uint16_t>(rowsThis),
                    curN * static_cast<uint32_t>(sizeof(int32_t)),
                    0U,
                    (n_ - curN) * static_cast<uint32_t>(sizeof(int32_t)) / GM_BLOCK_BYTES,
                    0U};
                DataCopyPadExtParams<int32_t> tilePad{false, 0U, 0U, 0};
                DataCopyPad(local[nOffset], partialGm_[sourceBase + tileOffset], tileCopy, tilePad);
            }
            return;
        }
        DataCopyExtParams copy{
            1U, rowsThis * n_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
        DataCopyPad(local, partialGm_[offset], copy, pad);
    }

    __aicore__ inline void TransposeBf16Tile(
        LocalTensor<bfloat16_t>& dst,
        LocalTensor<bfloat16_t>& src)
    {
        uint64_t srcAddrList[TRANSPOSE_BLOCK];
        uint64_t dstAddrList[TRANSPOSE_BLOCK];
        TransDataTo5HDParams params;
        params.repeatTimes = n_ / TRANSPOSE_BLOCK;
        params.srcRepStride = params.repeatTimes == 1U ? 0U : 1U;
        params.dstRepStride = params.repeatTimes == 1U ? 0U : TRANSPOSE_BLOCK;
        for (uint32_t row = 0U; row < TRANSPOSE_BLOCK; ++row) {
            srcAddrList[row] = reinterpret_cast<uint64_t>(
                src[static_cast<uint64_t>(row) * n_].GetPhyAddr());
            dstAddrList[row] = reinterpret_cast<uint64_t>(
                dst[static_cast<uint64_t>(row) * TRANSPOSE_BLOCK].GetPhyAddr());
        }
        TransDataTo5HD<half>(dstAddrList, srcAddrList, params);
    }

    __aicore__ inline void ReducePartialsAndStore()
    {
        const uint32_t rowStart = blockIdx_ * REDUCE_ROWS;
        if (rowStart >= m_) {
            return;
        }
        const uint32_t rowsThis = Min(REDUCE_ROWS, m_ - rowStart);
        const uint32_t elemsThis = rowsThis * n_;
        LocalTensor<int32_t> accumIntLocal = fp32Buffer_.Get<int32_t>();
        LocalTensor<float> accumLocal = fp32Buffer_.Get<float>()[reduceElems_];
        Duplicate(accumIntLocal, static_cast<int32_t>(0), elemsThis);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> first = srcQueue_.AllocTensor<int32_t>();
        LoadPartial(first, 0U, rowStart, rowsThis);
        srcQueue_.EnQue<int32_t>(first);
        for (uint32_t sourceCore = 0U; sourceCore < usedCoreNum_; ++sourceCore) {
            LocalTensor<int32_t> src = srcQueue_.DeQue<int32_t>();
            if (sourceCore + 1U < usedCoreNum_) {
                LocalTensor<int32_t> next = srcQueue_.AllocTensor<int32_t>();
                LoadPartial(next, sourceCore + 1U, rowStart, rowsThis);
                srcQueue_.EnQue<int32_t>(next);
            }
            Add(accumIntLocal, accumIntLocal, src, elemsThis);
            PipeBarrier<PIPE_V>();
            srcQueue_.FreeTensor<int32_t>(src);
        }
        Cast(accumLocal, accumIntLocal, RoundMode::CAST_RINT, elemsThis);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        if constexpr (IS_ALGEBRAIC_TRANSPOSE) {
            DataCopyExtParams tokenCopy{1U, n_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> scalePad{false, 0U, 0U, 0.0f};
            DataCopyPad(scaleLocal, pertokenScaleGm_, tokenCopy, scalePad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            for (uint32_t row = 0U; row < rowsThis; ++row) {
                const float channelScale = scaleGm_.GetValue(rowStart + row);
                Muls(accumLocal[row * n_], accumLocal[row * n_], channelScale, n_);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0U; row < rowsThis; ++row) {
                Mul(accumLocal[row * n_], accumLocal[row * n_], scaleLocal, n_);
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = outQueue_.AllocTensor<bfloat16_t>();
            if constexpr (IS_BOTH_TRANS_LOCAL_B1) {
                Cast(outLocal, accumLocal, RoundMode::CAST_RINT, elemsThis);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
                DataCopyExtParams outCopy{
                    1U, elemsThis * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
                DataCopyPad(outGm_[rowStart], outLocal, outCopy);
            } else {
                Duplicate(outLocal, static_cast<bfloat16_t>(0.0f), TRANSPOSE_BLOCK * n_);
                PipeBarrier<PIPE_V>();
                Cast(outLocal, accumLocal, RoundMode::CAST_RINT, elemsThis);
                PipeBarrier<PIPE_V>();
                LocalTensor<bfloat16_t> transpose = transposeBuffer_.Get<bfloat16_t>();
                TransposeBf16Tile(transpose, outLocal);
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
                DataCopyExtParams outCopy{
                    static_cast<uint16_t>(n_),
                    rowsThis * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0U,
                    (originalN_ - rowsThis) * static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0U};
                DataCopyPad(outGm_[rowStart], transpose, outCopy);
            }
            outQueue_.FreeTensor<bfloat16_t>(outLocal);
        } else {
            if constexpr (IS_SINGLE_COLUMN_CUBE) {
                DataCopyExtParams tokenCopy{
                    1U, rowsThis * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> tokenPad{false, 0U, 0U, 0.0f};
                DataCopyPad(scaleLocal, pertokenScaleGm_[rowStart], tokenCopy, tokenPad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                Muls(scaleLocal, scaleLocal, scaleScalar_, rowsThis);
                PipeBarrier<PIPE_V>();
                Mul(accumLocal, accumLocal, scaleLocal, rowsThis);
                PipeBarrier<PIPE_V>();
            } else {
            if (scaleLen_ != 1U) {
                DataCopyExtParams scaleCopy{1U, n_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> scalePad{false, 0U, 0U, 0.0f};
                DataCopyPad(scaleLocal, scaleGm_, scaleCopy, scalePad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                for (uint32_t row = 0U; row < rowsThis; ++row) {
                    Mul(accumLocal[row * n_], accumLocal[row * n_], scaleLocal, n_);
                }
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t row = 0U; row < rowsThis; ++row) {
                const float tokenScale = pertokenScaleGm_.GetValue(rowStart + row);
                const float combinedScale = scaleLen_ == 1U ? scaleScalar_ * tokenScale : tokenScale;
                Muls(accumLocal[row * n_], accumLocal[row * n_], combinedScale, n_);
            }
            PipeBarrier<PIPE_V>();
            }

            LocalTensor<bfloat16_t> outLocal = outQueue_.AllocTensor<bfloat16_t>();
            Cast(outLocal, accumLocal, RoundMode::CAST_RINT, elemsThis);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue<bfloat16_t>(outLocal);
            outLocal = outQueue_.DeQue<bfloat16_t>();
            DataCopyExtParams outCopy{
                1U, elemsThis * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
            DataCopyPad(outGm_[static_cast<uint64_t>(rowStart) * n_], outLocal, outCopy);
            outQueue_.FreeTensor<bfloat16_t>(outLocal);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) const
    {
        return (value + factor - 1U) / factor;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1UL) / align * align;
    }

    static constexpr uint32_t GM_BLOCK_BYTES = 32U;
    static constexpr uint32_t LOCAL_A1_K_CHUNK = IS_SINGLE_COLUMN_CUBE ? 1536U :
        (IS_SINGLE_ROW_LOCAL_A1 ? 5504U :
            (IS_ALGEBRAIC_TRANSPOSE ? 2816U : (IS_MID_N ? 3296U : 2816U)));
    static constexpr uint32_t TRANSPOSE_BLOCK = 16U;
    static constexpr uint32_t BOTH_TRANS_LOCAL_N_TILE = 128U;
    static constexpr uint32_t BOTH_TRANS_LOCAL_B1_K_CHUNK = 3280U;
    static constexpr uint32_t TRANS_A_LOCAL_B1_K_SLICE = 6144U;
    static constexpr uint32_t TRANS_A_LOCAL_B1_MAX_N = 64U;
    static constexpr uint32_t STAGE_UB_BYTES = 64U * 1024U;
    static constexpr uint32_t REDUCE_ROWS =
        IS_ALGEBRAIC_TRANSPOSE ? TRANSPOSE_BLOCK : (IS_SINGLE_COLUMN_CUBE ? 16U : 8U);
    static constexpr uint16_t STAGE_READY_FLAG_BASE = 0x4U;
    static constexpr uint16_t PARTIAL_READY_FLAG = 0x7U;

    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType,
        QBM_CFG_MDL_INTRINSICS> mm_;
    TQue<QuePosition::A1, 1> a1Queue_;
    TQue<QuePosition::B1, 1> b1Queue_;
    TQue<QuePosition::VECIN, 1> stageQueue_;
    TQue<QuePosition::VECIN, 2> srcQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> fp32Buffer_;
    TBuf<TPosition::VECCALC> scaleBuffer_;
    TBuf<TPosition::VECCALC> transposeBuffer_;
    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int32_t> x1Int32Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int32_t> x2Int32Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int32_t> partialGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;
    uint32_t blockIdx_ = 0U;
    uint32_t taskRatio_ = 1U;
    uint32_t subBlockIdx_ = 0U;
    uint32_t usedCoreNum_ = 0U;
    uint32_t m_ = 0U;
    uint32_t n_ = 0U;
    uint32_t originalM_ = 0U;
    uint32_t originalN_ = 0U;
    uint32_t k_ = 0U;
    uint32_t scaleLen_ = 0U;
    uint32_t baseN_ = 0U;
    uint32_t singleCoreK_ = 0U;
    uint32_t compactKStride_ = 0U;
    uint32_t reduceElems_ = 0U;
    uint64_t workspacePerCore_ = 0UL;
    uint64_t partialStrideElems_ = 0UL;
    float scaleScalar_ = 1.0f;
};

// Case09 structural path: load each original-stride A slice directly into
// local A1 while preserving the retained Tall-M int32 partial surface and
// twelve-row source-order reducer.
class QuantBatchMatMulV3Case09AicLocalA1 {
public:
    using AMatmulType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, int8_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

    __aicore__ inline QuantBatchMatMulV3Case09AicLocalA1() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        taskRatio_ = static_cast<uint32_t>(GetTaskRation());
        taskRatio_ = taskRatio_ > 0U ? taskRatio_ : 1U;
        if (GetTaskRation() > 0) {
            blockIdx_ /= taskRatio_;
        }
        m_ = static_cast<uint32_t>(tilingData->m);
        n_ = static_cast<uint32_t>(tilingData->n);
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        singleCoreK_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreK);
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        workspacePerCore_ = tilingData->workspacePerCore;

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), static_cast<uint64_t>(m_) * k_);
        x1Int32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1),
            static_cast<uint64_t>(m_) * k_ / sizeof(int32_t));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), static_cast<uint64_t>(k_) * n_);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(m_) * n_);

        int32PartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(userWorkspace + workspacePerCore_ * usedCoreNum_),
            static_cast<uint64_t>(usedCoreNum_) * m_ * n_);

        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U ||
            singleCoreK_ == 0U) {
            return;
        }
        const uint64_t partialBytes = AlignUp64(
            static_cast<uint64_t>(m_) * n_ * sizeof(int32_t), GM_BLOCK_BYTES);
        compactKStride_ = singleCoreK_;
        GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
        aCompactGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(coreWorkspace + 2UL * partialBytes),
            static_cast<uint64_t>(m_) * compactKStride_);

        if ASCEND_IS_AIC {
            pipe->InitBuffer(a1Queue_, 1U, AlignUp(m_, 16U) * LOCAL_A1_K_CHUNK * sizeof(int8_t));
            mm_.SetSubBlockIdx(0);
            mm_.Init(&tilingData->matmulTiling, pipe);
        }
        if ASCEND_IS_AIV {
            reduceElems_ = REDUCE_ROWS * n_;
            const uint32_t tokenBroadcastElems =
                AlignUp(REDUCE_ROWS, FP32_BLOCK_ELEMS) * FP32_BLOCK_ELEMS;
            pipe->InitBuffer(reduceInt32Queue_, 2U, reduceElems_ * sizeof(int32_t));
            pipe->InitBuffer(reduceSrcQueue_, 2U, reduceElems_ * sizeof(float));
            pipe->InitBuffer(scaleQueue_, 1U, n_ * sizeof(float));
            pipe->InitBuffer(outQueue_, 1U, reduceElems_ * sizeof(bfloat16_t));
            pipe->InitBuffer(accumBuffer_, (reduceElems_ + tokenBroadcastElems) * sizeof(float));
            pipe->InitBuffer(fp32Buffer_, reduceElems_ * sizeof(float));
            pipe->InitBuffer(tokenBuffer_, AlignUp(REDUCE_ROWS, FP32_BLOCK_ELEMS) * sizeof(float));
            if (scaleLen_ == 1U) {
                scaleScalar_ = scaleGm_.GetValue(0U);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || m_ == 0U || n_ == 0U || k_ == 0U || scaleLen_ == 0U ||
            singleCoreK_ == 0U || singleCoreK_ > MAX_SINGLE_CORE_K || m_ > MAX_M ||
            k_ > MAX_SOURCE_K || (n_ % BF16_BLOCK_ELEMS) != 0U || (k_ % GM_BLOCK_BYTES) != 0U) {
            return;
        }
        const uint32_t kOffset = blockIdx_ * singleCoreK_;
        const uint32_t curK = kOffset < k_ ? Min(singleCoreK_, k_ - kOffset) : 0U;
        if ASCEND_IS_AIC {
            ProduceInt32Partial(kOffset, curK);
        }
        if ASCEND_IS_AIV {
            WaitEvent(PARTIAL_READY_FLAG);
            SyncAll();
            ReducePartialsAndStore();
        }
    }

private:
    __aicore__ inline void StageCompactA(uint32_t kOffset, uint32_t curK)
    {
        uint32_t chunkIndex = 0U;
        for (uint32_t localK = 0U; localK < curK; localK += LOCAL_A1_K_CHUNK, ++chunkIndex) {
            const uint32_t chunkK = Min(LOCAL_A1_K_CHUNK, curK - localK);
            const uint32_t rowsPerBatch = STAGE_UB_BYTES / chunkK;
            for (uint32_t rowBase = 0U; rowBase < m_; rowBase += rowsPerBatch) {
                const uint32_t rowsThis = Min(rowsPerBatch, m_ - rowBase);
                LocalTensor<int8_t> local = stageQueue_.AllocTensor<int8_t>();
                DataCopyParams copyIn{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(chunkK / GM_BLOCK_BYTES),
                    static_cast<uint16_t>((k_ - chunkK) / GM_BLOCK_BYTES),
                    0U};
                DataCopy(local, x1Gm_[static_cast<uint64_t>(rowBase) * k_ + kOffset + localK], copyIn);
                SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
                DataCopyParams copyOut{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint16_t>(chunkK / GM_BLOCK_BYTES),
                    0U,
                    static_cast<uint16_t>((compactKStride_ - chunkK) / GM_BLOCK_BYTES)};
                DataCopy(aCompactGm_[static_cast<uint64_t>(rowBase) * compactKStride_ + localK], local, copyOut);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                stageQueue_.FreeTensor<int8_t>(local);
            }
            NotifyEvent<PIPE_MTE3>(static_cast<uint16_t>(STAGE_READY_FLAG_BASE + chunkIndex));
        }
    }

    __aicore__ inline void ProduceInt32Partial(uint32_t kOffset, uint32_t curK)
    {
        GlobalTensor<int32_t> dst =
            int32PartialGm_[static_cast<uint64_t>(blockIdx_) * m_ * n_];
        bool firstChunk = true;
        uint32_t chunkIndex = 0U;
        for (uint32_t localK = 0U; localK < curK; localK += LOCAL_A1_K_CHUNK, ++chunkIndex) {
            const uint32_t chunkK = Min(LOCAL_A1_K_CHUNK, curK - localK);
            LocalTensor<int8_t> a1Local = a1Queue_.AllocTensor<int8_t>();
            LocalTensor<int32_t> a1Local32 = a1Local.template ReinterpretCast<int32_t>();
            Nd2NzParams copy;
            copy.ndNum = 1U;
            copy.nValue = m_;
            copy.dValue = chunkK / sizeof(int32_t);
            copy.srcDValue = k_ / sizeof(int32_t);
            copy.srcNdMatrixStride = 0U;
            copy.dstNzC0Stride = AlignUp(m_, 16U);
            copy.dstNzNStride = 1U;
            copy.dstNzMatrixStride = 0U;
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(kOffset) + localK) / sizeof(int32_t);
            DataCopy(a1Local32, x1Int32Gm_[sourceOffset], copy);
            a1Queue_.EnQue<int8_t>(a1Local);
            a1Local = a1Queue_.DeQue<int8_t>();

            mm_.SetOrgShape(m_, n_, chunkK, k_, n_);
            mm_.SetSingleShape(m_, n_, chunkK);
            mm_.SetTail(m_, n_, chunkK);
            mm_.SetTensorA(a1Local, false);
            mm_.SetTensorB(x2Gm_[static_cast<uint64_t>(kOffset + localK) * n_], false);
            mm_.Iterate();
            mm_.GetTensorC(dst, firstChunk ? 0U : 1U, true);
            firstChunk = false;
            a1Queue_.FreeTensor<int8_t>(a1Local);
        }
        mm_.End();
        NotifyEvent<PIPE_FIX>(PARTIAL_READY_FLAG);
    }

    __aicore__ inline void EnqueuePartial(uint32_t sourceCore, uint32_t rowStart, uint32_t rowsThis)
    {
        const uint32_t elemsThis = rowsThis * n_;
        LocalTensor<int32_t> int32Local = reduceInt32Queue_.AllocTensor<int32_t>();
        const uint64_t offset =
            (static_cast<uint64_t>(sourceCore) * m_ + rowStart) * n_;
        DataCopyExtParams copy{
            static_cast<uint16_t>(rowsThis), n_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
        DataCopyPad(int32Local, int32PartialGm_[offset], copy, pad);
        reduceInt32Queue_.EnQue<int32_t>(int32Local);
        int32Local = reduceInt32Queue_.DeQue<int32_t>();
        LocalTensor<float> fp32Local = reduceSrcQueue_.AllocTensor<float>();
        Cast(fp32Local, int32Local, RoundMode::CAST_RINT, elemsThis);
        PipeBarrier<PIPE_V>();
        reduceInt32Queue_.FreeTensor<int32_t>(int32Local);
        reduceSrcQueue_.EnQue<float>(fp32Local);
    }

    __aicore__ inline void ReducePartialsAndStore()
    {
        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = scaleQueue_.AllocTensor<float>();
            DataCopyExtParams scaleCopy{1U, n_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> scalePad{false, 0U, 0U, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_, scaleCopy, scalePad);
            scaleQueue_.EnQue<float>(scaleLocal);
            scaleLocal = scaleQueue_.DeQue<float>();
        }

        for (uint32_t rowStart = blockIdx_ * REDUCE_ROWS; rowStart < m_;
             rowStart += usedCoreNum_ * REDUCE_ROWS) {
            const uint32_t rowsThis = Min(REDUCE_ROWS, m_ - rowStart);
            const uint32_t elemsThis = rowsThis * n_;
            LocalTensor<float> accum = accumBuffer_.Get<float>();
            LocalTensor<float> fp32 = fp32Buffer_.Get<float>();
            for (uint32_t row = 0U; row < rowsThis; ++row) {
                Duplicate(accum[row * n_], 0.0f, n_);
            }
            PipeBarrier<PIPE_V>();

            EnqueuePartial(0U, rowStart, rowsThis);
            for (uint32_t sourceCore = 0U; sourceCore < usedCoreNum_; ++sourceCore) {
                LocalTensor<float> src = reduceSrcQueue_.DeQue<float>();
                if (sourceCore + 1U < usedCoreNum_) {
                    EnqueuePartial(sourceCore + 1U, rowStart, rowsThis);
                }
                Add(accum, accum, src, elemsThis);
                PipeBarrier<PIPE_V>();
                reduceSrcQueue_.FreeTensor<float>(src);
            }

            LocalTensor<float> token = tokenBuffer_.Get<float>();
            DataCopyExtParams tokenCopy{1U, rowsThis * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> tokenPad{false, 0U, 0U, 0.0f};
            DataCopyPad(token, pertokenScaleGm_[rowStart], tokenCopy, tokenPad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

            const uint32_t alignedRows = AlignUp(rowsThis, FP32_BLOCK_ELEMS);
            LocalTensor<float> tokenBrcb = accumBuffer_.Get<float>()[reduceElems_];
            Brcb(tokenBrcb, token, static_cast<uint8_t>(alignedRows / FP32_BLOCK_ELEMS),
                {1, FP32_BLOCK_ELEMS});
            if (scaleLen_ == 1U) {
                PipeBarrier<PIPE_V>();
                Muls(tokenBrcb, tokenBrcb, scaleScalar_, alignedRows * FP32_BLOCK_ELEMS);
            }
            PipeBarrier<PIPE_V>();

            const uint8_t rowStride = static_cast<uint8_t>(n_ / FP32_BLOCK_ELEMS);
            constexpr uint32_t SCALE_CHUNK_ELEMS = 64U;
            if (scaleLen_ != 1U) {
                BinaryRepeatParams channelParams{1, 1, 1, rowStride, rowStride, 0};
                for (uint32_t col = 0U; col < n_; col += SCALE_CHUNK_ELEMS) {
                    const uint32_t colsThis = Min(SCALE_CHUNK_ELEMS, n_ - col);
                    Mul(fp32[col], accum[col], scaleLocal[col], static_cast<uint64_t>(colsThis),
                        static_cast<uint8_t>(rowsThis), channelParams);
                }
                PipeBarrier<PIPE_V>();
            }
            BinaryRepeatParams tokenParams{1, 1, 0, rowStride, rowStride, 1};
            if (scaleLen_ == 1U) {
                for (uint32_t col = 0U; col < n_; col += SCALE_CHUNK_ELEMS) {
                    const uint32_t colsThis = Min(SCALE_CHUNK_ELEMS, n_ - col);
                    Mul(fp32[col], accum[col], tokenBrcb, static_cast<uint64_t>(colsThis),
                        static_cast<uint8_t>(rowsThis), tokenParams);
                }
            } else {
                for (uint32_t col = 0U; col < n_; col += SCALE_CHUNK_ELEMS) {
                    const uint32_t colsThis = Min(SCALE_CHUNK_ELEMS, n_ - col);
                    Mul(fp32[col], fp32[col], tokenBrcb, static_cast<uint64_t>(colsThis),
                        static_cast<uint8_t>(rowsThis), tokenParams);
                }
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = outQueue_.AllocTensor<bfloat16_t>();
            Cast(outLocal, fp32, RoundMode::CAST_RINT, elemsThis);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue<bfloat16_t>(outLocal);
            outLocal = outQueue_.DeQue<bfloat16_t>();
            DataCopyExtParams outCopy{
                static_cast<uint16_t>(rowsThis), n_ * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
            DataCopyPad(outGm_[static_cast<uint64_t>(rowStart) * n_], outLocal, outCopy);
            outQueue_.FreeTensor<bfloat16_t>(outLocal);
        }
        if (scaleLen_ != 1U) {
            scaleQueue_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1UL) / align * align;
    }

    static constexpr uint32_t GM_BLOCK_BYTES = 32U;
    static constexpr uint32_t FP32_BLOCK_ELEMS = 8U;
    static constexpr uint32_t BF16_BLOCK_ELEMS = 16U;
    static constexpr uint32_t MAX_M = 200U;
    static constexpr uint32_t MAX_SINGLE_CORE_K = 6720U;
    static constexpr uint32_t MAX_SOURCE_K = 2097120U;
    static constexpr uint32_t LOCAL_A1_K_CHUNK = 2240U;
    static constexpr uint32_t STAGE_UB_BYTES = 64U * 1024U;
    static constexpr uint32_t REDUCE_ROWS = 12U;
    static constexpr uint16_t STAGE_READY_FLAG_BASE = 0x4U;
    static constexpr uint16_t PARTIAL_READY_FLAG = 0x7U;

    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType,
        QBM_CFG_MDL_INTRINSICS> mm_;
    TQue<QuePosition::A1, 1> a1Queue_;
    TQue<QuePosition::VECIN, 1> stageQueue_;
    TQue<QuePosition::VECIN, 2> reduceInt32Queue_;
    TQue<QuePosition::VECIN, 2> reduceSrcQueue_;
    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> accumBuffer_;
    TBuf<TPosition::VECCALC> fp32Buffer_;
    TBuf<TPosition::VECCALC> tokenBuffer_;
    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int32_t> x1Int32Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int32_t> int32PartialGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;
    uint32_t blockIdx_ = 0U;
    uint32_t taskRatio_ = 1U;
    uint32_t usedCoreNum_ = 0U;
    uint32_t m_ = 0U;
    uint32_t n_ = 0U;
    uint32_t k_ = 0U;
    uint32_t scaleLen_ = 0U;
    uint32_t singleCoreK_ = 0U;
    uint32_t compactKStride_ = 0U;
    uint32_t reduceElems_ = 0U;
    uint64_t workspacePerCore_ = 0UL;
    float scaleScalar_ = 1.0f;
};

} // namespace NsQuantBatchMatMulV3
#endif // QUANTBATCHMATMULV3_H
