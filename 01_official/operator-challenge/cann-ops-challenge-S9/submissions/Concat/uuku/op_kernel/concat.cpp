// Concat entry, dispatch and shared state. Algorithms are class-body includes.
// Logical output: [outerSize64, outputAxisSize, innerSize64]. AxisLen(i) is
// input i's axis extent; input/output payloads are copied as raw B8/B16/B32 bits.
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "concat_linear_route.h"
#define CONCAT_STREAM_INLINE __aicore__ inline
#include "concat_input_stream_geometry.h"
#undef CONCAT_STREAM_INLINE

using namespace AscendC;

namespace {
constexpr uint32_t kMaxConcatInputs = 1792U;
constexpr uint32_t kBufferNum = 2U;
constexpr uint32_t kGroupInputThreshold = 64U;
constexpr uint32_t kGroupMaxElems = 256U;
constexpr uint32_t kGroupMinElems = 32U;
constexpr uint32_t kTilingAxisLensOffset = 14U;
constexpr uint32_t kTilingAxisOffsetsOffset = kTilingAxisLensOffset + kMaxConcatInputs;
constexpr uint32_t kTilingScheduleOrderOffset =
    kTilingAxisOffsetsOffset + kMaxConcatInputs;
constexpr uint32_t kMetaCacheMinInputs = 64U;
constexpr uint32_t kMetaCacheMaxInputs = 256U;
constexpr uint32_t kPeriodicGatherAxisElems = 32U;
constexpr uint32_t kContinuousMaxWidth = 128U;
constexpr uint32_t kPeriodicGatherFlag = 0x80000000U;
constexpr uint32_t kContinuousB32Flag = 0x40000000U;
constexpr uint32_t kCompositeGatherB32Flag = 0x20000000U;
constexpr uint32_t kBlockPackB32Flag = 0x10000000U;
constexpr uint32_t kSlabRowsB32Flag = 0x08000000U;
constexpr uint32_t kAsymPackB32Flag = 0x04000000U;
constexpr uint32_t kWideGeometryFlag = 0x02000000U;
constexpr uint32_t kTilingWideGeometryOffset = kTilingScheduleOrderOffset + kMaxConcatInputs;
constexpr uint32_t kTilingLinearPlanOffset = kTilingWideGeometryOffset + 3U;
constexpr uint32_t kPhaseOwnedB32Flag = kCompositeGatherB32Flag;
constexpr uint32_t kRouteModeMask =
    kPeriodicGatherFlag | kContinuousB32Flag | kCompositeGatherB32Flag |
    kBlockPackB32Flag | kSlabRowsB32Flag | kAsymPackB32Flag | kWideGeometryFlag;
constexpr uint32_t kRouteContinuousB32 = 1U;
constexpr uint32_t kRouteCompositeGatherB32 = 2U;
constexpr uint32_t kRoutePhaseOwnedB32 = kRouteCompositeGatherB32;
constexpr uint32_t kPhaseCount = 8U;
constexpr uint32_t kPhaseOwnerBits = 6U;
constexpr uint32_t kAivCoreNum910B = 40U;
constexpr uint32_t kPhaseColumnMask = 0x1FFFFFFFU;
constexpr uint32_t kPhaseBoundarySourcePitch = 64U;

__aicore__ static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ static inline uint32_t CeilDivU32(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}

template <typename T>
__aicore__ static inline uint32_t AlignElems32(uint32_t elems)
{
    constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
    return ((elems + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
}

__aicore__ static inline uint32_t AlignDownU32(uint32_t x, uint32_t alignment)
{
    return (x / alignment) * alignment;
}

__aicore__ static inline uint32_t AlignUpU32(uint32_t x, uint32_t alignment)
{
    return ((x + alignment - 1U) / alignment) * alignment;
}

__aicore__ static inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

class ConcatKernel {
public:
    __aicore__ inline ConcatKernel() = default;

    // Read the serialized route contract; low 32-bit geometry is used by legacy
    // fast paths only when wideGeometry is false. No UB is allocated here.
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace,
                                GM_ADDR tiling, TPipe* pipeIn)
    {
        (void)workspace;
        inputList = inputs;
        outputAddr = output;
        tilingAddr = tiling;
        pipe = pipeIn;
        __gm__ uint64_t* inputWords = reinterpret_cast<__gm__ uint64_t*>(inputList);
        inputPtrBase = inputWords[0] / sizeof(uint64_t);

        __gm__ uint32_t* td = reinterpret_cast<__gm__ uint32_t*>(tilingAddr);
        inputCount = td[0];
        outerSize = td[1];
        innerSize = td[2];
        outputAxisSize = td[3];
        totalElems = td[4];
        tileElems = td[7];
#if !defined(DTYPE_INPUTS)
        // Preserve the original runtime-width entry for builds without the
        // generated operator dtype options (including standalone diagnostics).
        elemBytes = td[8];
#endif
        ubBytes = td[9];
        const uint32_t groupInfo = td[10];
        wideGeometry = (groupInfo & kWideGeometryFlag) != 0U;
        outerSize64 = outerSize;
        innerSize64 = innerSize;
        totalElems64 = totalElems;
        if (wideGeometry) {
            // Appended high words preserve all existing low-word/array offsets.
            outerSize64 |= static_cast<uint64_t>(td[kTilingWideGeometryOffset]) << 32U;
            innerSize64 |= static_cast<uint64_t>(td[kTilingWideGeometryOffset + 1U]) << 32U;
            totalElems64 |= static_cast<uint64_t>(td[kTilingWideGeometryOffset + 2U]) << 32U;
        }
        periodicPatternInputs = (groupInfo & kPeriodicGatherFlag) != 0U
                                    ? groupInfo & ~kPeriodicGatherFlag
                                    : 0U;
        // Serialized route tags: 0 generic, 1 continuous rows, 2 composite,
        // 3 block pack, 4 row-phase prefix, 5 alternating/sliced/input-stream.
        // Tag 5 uses schedule[7] to select mode 0, 1 or 3. Keep ABI values fixed.
        routeKey = (groupInfo & kAsymPackB32Flag) != 0U ? 5U
                   : (groupInfo & kSlabRowsB32Flag) != 0U ? 4U
                   : (groupInfo & kBlockPackB32Flag) != 0U ? 3U
                   : (groupInfo & kCompositeGatherB32Flag) != 0U
                       ? kRouteCompositeGatherB32
                       : ((groupInfo & kContinuousB32Flag) != 0U
                              ? kRouteContinuousB32
                              : 0U);
        routeScheduleCount = routeKey == kRouteContinuousB32
                                 ? groupInfo & ~kRouteModeMask
                                 : 0U;
        alignedGroupCount = (groupInfo & kRouteModeMask) == 0U ? groupInfo : 0U;
        phaseBoundaryCount = td[11];
        phaseUnitCount = td[12];
        phaseOwnerWordCount = td[13];

        // Generic ownership is computed with checked U64 byte boundaries.
    }

    __aicore__ inline void Process()
    {
        if (inputCount == 0U || innerSize64 == 0U || outputAxisSize == 0U ||
            totalElems64 == 0U || tileElems == 0U || elemBytes == 0U) {
            return;
        }

        switch (elemBytes) {
            case 1U:
                ProcessTyped<uint8_t>();
                break;
            case 2U:
                ProcessTyped<uint16_t>();
                break;
            case 4U:
                ProcessTyped<uint32_t>();
                break;
            default:
                break;
        }
    }

private:
#include "concat_block_pack.h"
#include "concat_asym_pack.h"
#include "concat_slice_pack.h"
#include "concat_input_stream.h"

    struct SegmentInfo {
        uint32_t inputIdx;
        uint32_t axisLen;
        uint64_t inputOffset;
        uint64_t count;
    };

    struct AlignedGroupPipelineState {
        uint32_t groupStart;
        uint32_t groupEnd;
        uint32_t groupAxisOffset;
        uint32_t groupElems;
        bool valid;
    };

    struct CompositeGatherGroup {
        uint32_t groupStart;
        uint32_t groupEnd;
        uint32_t groupAxisOffset;
        uint32_t groupElems;
        uint32_t sourceElems;
        uint32_t nonZeroMembers;
        bool valid;
    };

    // Metadata helpers read either the optional UB cache or the original GM
    // descriptors. They never read input payload values for route selection.
    __aicore__ inline uint32_t AxisLen(uint32_t inputIdx) const
    {
        if (useLocalMeta) {
            return metaLocal.GetValue(inputIdx);
        }
        __gm__ uint32_t* td = reinterpret_cast<__gm__ uint32_t*>(tilingAddr);
        return td[kTilingAxisLensOffset + inputIdx];
    }

    __aicore__ inline uint32_t AxisOffset(uint32_t inputIdx) const
    {
        if (useLocalMeta) {
            return metaLocal.GetValue(metaStride + inputIdx);
        }
        __gm__ uint32_t* td = reinterpret_cast<__gm__ uint32_t*>(tilingAddr);
        return td[kTilingAxisOffsetsOffset + inputIdx];
    }

    __aicore__ inline uint32_t ScheduleInput(uint32_t scheduleIdx) const
    {
        return ScheduleWord(scheduleIdx);
    }

    __aicore__ inline uint32_t ScheduleWord(uint32_t scheduleIdx) const
    {
        if (useLocalMeta) {
            return metaLocal.GetValue(4U * metaStride + scheduleIdx);
        }
        __gm__ uint32_t* td = reinterpret_cast<__gm__ uint32_t*>(tilingAddr);
        return td[kTilingScheduleOrderOffset + scheduleIdx];
    }

    __aicore__ inline __gm__ uint8_t* InputPtr(uint32_t inputIdx) const
    {
        if (useLocalMeta) {
            const uint32_t wordOffset = 2U * metaStride + 2U * inputIdx;
            const uint64_t low = metaLocal.GetValue(wordOffset);
            const uint64_t high = metaLocal.GetValue(wordOffset + 1U);
            return reinterpret_cast<__gm__ uint8_t*>(low | (high << 32U));
        }
        __gm__ uint64_t* inputWords = reinterpret_cast<__gm__ uint64_t*>(inputList);
        return reinterpret_cast<__gm__ uint8_t*>(inputWords[inputPtrBase + inputIdx]);
    }

    __aicore__ inline void InitMetaCache()
    {
        useLocalMeta = inputCount >= kMetaCacheMinInputs && inputCount <= kMetaCacheMaxInputs;
        if (!useLocalMeta) {
            return;
        }
        metaStride = AlignElems32<uint32_t>(inputCount);
        // Cache axis lengths, axis offsets and the dynamic-input pointer table.
        // Many-input routes revisit every input on every core, so leaving the pointers in
        // GM still creates one scalar GM load for every copied segment.
        // Composite-Gather reconstructs its ordered groups directly from the
        // cached width vector.  Only the continuous route consumes schedule
        // words; the three phase-named ABI scalars are route contracts, not a
        // schedule payload, for Composite-Gather.
        const uint32_t planWords =
            routeKey == 5U ? inputCount + 8U :
            routeKey == kRouteContinuousB32 ? inputCount : 0U;
        const uint32_t metaBytes = AlignUpU32(
            (4U * metaStride + planWords) * sizeof(uint32_t), 32U);
        pipe->InitBuffer(metaBuf, metaBytes);
        metaLocal = metaBuf.Get<uint32_t>();

        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tilingAddr),
                                 kTilingScheduleOrderOffset + kMaxConcatInputs);
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(inputCount * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(metaLocal, tilingGm[kTilingAxisLensOffset], copyParams, padParams);
        DataCopyPad(metaLocal[metaStride], tilingGm[kTilingAxisOffsetsOffset], copyParams, padParams);
        if (planWords != 0U) {
            DataCopyExtParams planCopyParams{
                1U, static_cast<uint32_t>(planWords * sizeof(uint32_t)),
                0U, 0U, 0U};
            DataCopyPad(metaLocal[4U * metaStride],
                        tilingGm[kTilingScheduleOrderOffset],
                        planCopyParams, padParams);
        }

        GlobalTensor<uint32_t> inputPtrGm;
        inputPtrGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(inputList) +
                                   inputPtrBase * 2U,
                                   static_cast<uint64_t>(inputCount) * 2U);
        DataCopyExtParams pointerCopyParams{
            1U, static_cast<uint32_t>(inputCount * sizeof(uint64_t)), 0U, 0U, 0U};
        DataCopyPad(metaLocal[2U * metaStride], inputPtrGm, pointerCopyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool IsManyInputLastAxisB32() const
    {
        return inputCount > 64U && inputCount <= 256U && elemBytes == 4U &&
               innerSize == 1U && outputAxisSize >= 512U && outerSize >= 64U;
    }

    // Priority is significant: complete host-selected routes return before the
    // generic preflight can allocate its own buffers. Do not reorder for style.
    template <typename T>
    __aicore__ inline void ProcessTyped()
    {
        GlobalTensor<T> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputAddr),
                            totalElems64);

        if (wideGeometry) {
            ProcessGenericFallback<T>(yGm);
            return;
        }

        if (TryProcessAsymPack()) {
            return;
        }
        if (TryProcessSlabRows()) {
            return;
        }
        if (TryProcessBlockPack()) {
            return;
        }
        if (TryProcessPeriodicGather<T>(yGm)) {
            return;
        }
        if (TryProcessCompositeGatherRows<T>(yGm)) {
            return;
        }
        if (TryProcessContinuousSmallWidthRows<T>(yGm)) {
            return;
        }

        if (TryProcessSmallB16RowGroup<T>(yGm)) {
            return;
        }

        ProcessGenericFallback<T>(yGm);
    }

    // Algorithm modules remain at the original textual positions. Including
    // them here adds no wrapper, runtime dispatch, or separate translation unit.
    #include "concat_periodic_gather.h"
    #include "concat_composite_gather.h"
    #include "concat_continuous_rows.h"
    #include "concat_grouped_rows.h"
    #include "concat_row_dma.h"

    #include "concat_small_b16_row_group.h"
    #include "concat_linear_copy.h"
    #include "concat_linear32.h"
    #include "concat_generic_dispatch.h"
    #include "concat_vector_fallback.h"

    // Invocation-local state. Member order is retained, including route-exclusive
    // queues: declaration does not allocate UB; each admitted route owns InitBuffer.
    TPipe* pipe = nullptr;
    GM_ADDR inputList = 0;
    GM_ADDR outputAddr = 0;
    GM_ADDR tilingAddr = 0;
    uint64_t inputPtrBase = 0U;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> periodicIndexWorkBuf;
    TBuf<TPosition::VECCALC> continuousBaseIndexBuf;
    TBuf<TPosition::VECCALC> metaBuf;
    LocalTensor<uint32_t> metaLocal;
    uint32_t inputCount = 0U;
    uint32_t outerSize = 0U;
    uint32_t innerSize = 0U;
    uint32_t outputAxisSize = 0U;
    uint32_t totalElems = 0U;
    uint32_t tileElems = 0U;
#if defined(DTYPE_INPUTS)
    // The generated binary/dynamic/concat.py provides DTYPE_INPUTS for all four
    // registered dtypes. Keep raw-bit B8/B16/B32 copies, not numeric casts.
    // A class constant also reaches non-template B32-only route predicates.
    static constexpr uint32_t elemBytes = sizeof(DTYPE_INPUTS);
    static_assert(elemBytes == 1U || elemBytes == 2U || elemBytes == 4U,
                  "Concat requires a supported byte width");
#if defined(DTYPE_OUTPUT)
    static_assert(sizeof(DTYPE_OUTPUT) == elemBytes,
                  "Concat input and output widths must match");
#endif
#else
    uint32_t elemBytes = 0U;
#endif
    uint32_t ubBytes = 0U;
    uint32_t alignedGroupCount = 0U;
    uint32_t periodicPatternInputs = 0U;
    uint32_t routeKey = 0U;
    uint32_t routeScheduleCount = 0U;
    // Historical tiling field names carry route-specific capacities. Their ABI
    // positions are shared by host/kernel and are intentionally not renamed here.
    uint32_t phaseBoundaryCount = 0U;
    uint32_t phaseUnitCount = 0U;
    uint32_t phaseOwnerWordCount = 0U;
    uint64_t outerSize64 = 0U, innerSize64 = 0U, totalElems64 = 0U;
    bool wideGeometry = false;
    uint32_t metaStride = 0U;
    bool useLocalMeta = false;
};
}  // namespace

extern "C" __global__ __aicore__ void concat(GM_ADDR inputs, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    ConcatKernel kernel;
    kernel.Init(inputs, output, workspace, tiling, &pipe);
    kernel.Process();
}
