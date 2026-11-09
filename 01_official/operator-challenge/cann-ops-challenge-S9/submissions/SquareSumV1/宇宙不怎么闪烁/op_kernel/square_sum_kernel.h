#pragma once
#include "kernel_operator.h"
#include <type_traits>
#define SQUARE_SUM_POLICY_INLINE __aicore__ inline
#include "square_sum_policy.h"
#undef SQUARE_SUM_POLICY_INLINE

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
constexpr uint32_t kModeRank4Reduce02CoreRange = 6U;
constexpr uint32_t kModeWorkspaceBf16Inner = 7U;
constexpr uint32_t kSuffixOutTileElems = 256U;
constexpr uint32_t kSuffixBatchRows = 128U;
constexpr uint32_t kSuffixBatchMaxReduce = 64U;
constexpr uint32_t kInnerReduceBatchRows = 64U;
constexpr uint32_t kRank4Reduce02BatchB = 2U;
// Keep the existing per-row fused accumulation order for FP32 and half.
constexpr uint32_t kInnerPipelinedBatchRows = 64U;
constexpr uint32_t kPlaneReduceChunkElems = 64U;
constexpr uint32_t kWidePlaneReduceChunkElems = 128U;
constexpr uint32_t kLargeSuffixChunkElems = 64U;
constexpr uint32_t kLargeSuffixBatchRows = 128U;

__aicore__ static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ static inline bfloat16_t FloatToBf16(float value)
{
    uint32_t raw = *reinterpret_cast<uint32_t*>(&value);
    const uint32_t lsb = (raw >> 16U) & 1U;
    raw += 0x7FFFU + lsb;
    const uint16_t bits = static_cast<uint16_t>(raw >> 16U);
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

        if (mode == kModeInnerAccum) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                balancedBatchRows = square_sum_policy::BalancedRows(innerSize);
            } else if constexpr (std::is_same_v<T, float>) {
                atomicMainCols = square_sum_policy::AtomicMainCols(innerSize);
                atomicTailCols = innerSize - atomicMainCols;
                atomicBatchRows = square_sum_policy::AtomicRows(innerSize);
            }
        }
        if (mode == kModePlaneReduce) {
            planeSourceCols = 1U;
            for (uint32_t dim = SuffixReduceStart(); dim < kDimNum; ++dim) {
                planeSourceCols *= inDims[dim];
            }
            directPlaneRows = square_sum_policy::DirectPlaneRows(planeSourceCols);
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            fusedSuffixRows = square_sum_policy::FusedSuffixRows();
            if (mode == kModeSuffixReduce) {
                fullRowSuffixRows = square_sum_policy::FullRowSuffixRows(reduceSize);
                packedSuffixRows = square_sum_policy::PackedSuffixRows(reduceSize);
                if (fullRowSuffixRows != 0U) {
                    fullRowSuffixPitch = static_cast<uint32_t>(
                        square_sum_policy::FullRowSuffixPitch(reduceSize));
                }
            }
        }

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), inputSize);
        if (td.bypass_l2 != 0U) {
            inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (fullRowSuffixRows != 0U) {
                // This path loads each complete input row exactly once.
                inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), outputSize);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (mode == kModeWorkspaceBf16Inner) {
                workspacePitch = AlignElems32<T>(innerSize);
                workspaceSlots = static_cast<uint32_t>(square_sum_policy::WorkspaceBf16Slots(
                    outputSize / innerSize, reduceSize, GetBlockNum()));
                workspaceElems = GetBlockNum() * workspaceSlots * workspacePitch;
                // Use the runtime accessor instead of assuming that the raw
                // entry argument omits system/profiling workspace metadata.
                GM_ADDR userWorkspace = GetUserWorkspace(workspace);
                workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace),
                                            workspaceElems);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (mode == kModeWorkspaceBf16Inner) {
                InitWorkspaceBf16Buffers();
                ProcessWorkspaceBf16Inner();
                return;
            }
        }
        if (outputSize == 0U || tileElems == 0U) return;
        if (mode == kModeSuffixReduce) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                if (packedSuffixRows != 0U) {
                    InitPackedSuffixBuffers();
                    ProcessPackedSuffix();
                    return;
                }
                if (fullRowSuffixRows != 0U) {
                    InitFullRowSuffixBuffers();
                    ProcessFullRowSuffix();
                    return;
                }
            }
            if (CanProcessLargeSuffixChunked()) {
                InitLargeSuffixChunkedBuffers();
                ProcessLargeSuffixChunked();
            } else {
                InitSuffixBuffers();
                ProcessSuffixReduce();
            }
        } else if (mode == kModeRank4Reduce02CoreRange) {
            InitRank4Reduce02CoreRangeBuffers();
            ProcessRank4Reduce02CoreRange();
        } else if (mode == kModeRank4Reduce02) {
            InitRank4Reduce02Buffers();
            ProcessRank4Reduce02();
        } else if (mode == kModeInnerAccum) {
            if constexpr (std::is_same_v<T, float>) {
                if (CanProcessAtomicInner()) {
                    if (square_sum_policy::TightAtomicFits(innerSize)) {
                        InitTightAtomicInnerBuffers();
                        ProcessTightAtomicInner();
                    } else {
                        InitAtomicInnerBuffers();
                        ProcessAtomicInner();
                    }
                    return;
                }
            }
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                if (CanProcessBalancedInner()) {
                    if (square_sum_policy::TightBf16Fits(innerSize)) {
                        InitTightBf16InnerBuffers();
                        ProcessTightBf16Inner();
                    } else {
                        InitBalancedInnerBuffers();
                        ProcessBalancedInner();
                    }
                    return;
                }
            }
            InitInnerBuffers();
            ProcessInnerAccum();
        } else if (mode == kModePlaneReduce) {
            if constexpr (std::is_same_v<T, half>) {
                if (CanProcessDirectPlane()) {
                    InitDirectPlaneBuffers();
                    ProcessDirectPlane();
                    return;
                }
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

private:
#include "square_sum_index.h"
#include "square_sum_numeric.h"
#include "square_sum_suffix.h"
#include "square_sum_fullrow.h"
#include "square_sum_packed.h"
#include "square_sum_inner.h"
#include "square_sum_tight.h"
#include "square_sum_bf16.h"
#include "square_sum_workspace.h"
#include "square_sum_plane.h"
#include "square_sum_fallback.h"

    TPipe* pipe = nullptr;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<float> workspaceGm;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECIN, 2> rank4InQueue;
    TQue<QuePosition::VECIN, 2> tightInQueue;
    TQue<QuePosition::VECIN, 2> workspaceInQueue;
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
    TBuf<TPosition::VECCALC> planeSecondSumBuf;
    TBuf<TPosition::VECCALC> planeThirdSumBuf;
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
    uint32_t workspacePitch = 0U;
    uint32_t workspaceSlots = 0U;
    uint32_t workspaceElems = 0U;
    uint32_t workspacePrefetchRow = 0U;
    uint32_t workspaceRowEnd = 0U;
    uint32_t balancedBatchRows = 0U;
    uint32_t atomicBatchRows = 0U;
    uint32_t atomicMainCols = 0U;
    uint32_t atomicTailCols = 0U;
    uint32_t directPlaneRows = 0U;
    uint32_t planeSourceCols = 0U;
    uint32_t fusedSuffixRows = 0U;
    uint32_t packedSuffixRows = 0U;
    uint32_t fullRowSuffixRows = 0U;
    uint32_t fullRowSuffixPitch = 0U;
    uint32_t inDims[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t inStrides[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t outStrides[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t reduceMask[kDimNum] = {0U, 0U, 0U, 0U, 0U};
};

template <typename T>
__aicore__ inline void RunSquareSum(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    SquareSumKernel<T> kernel;
    kernel.Init(input, output, workspace, tiling, &pipe);
    kernel.Process();
}
}  // namespace
