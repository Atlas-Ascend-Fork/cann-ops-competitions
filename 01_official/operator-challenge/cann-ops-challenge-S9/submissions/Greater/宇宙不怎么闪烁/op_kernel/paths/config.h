constexpr uint32_t kDimNum = 5U;
constexpr uint32_t kBufferNum = 1U;
constexpr uint32_t kCScalarCacheBytes = 1024U;
constexpr uint32_t kCScalarPipelineFloatElems = 10240U;
constexpr uint32_t kCScalarPipelineHalfElems = 16384U;
constexpr uint32_t kCScalarRowGroupFloatElems = 8192U;
constexpr uint32_t kCScalarRowGroupHalfElems = 16384U;
constexpr uint32_t kCRowBroadcastMaxRows = 128U;
constexpr uint32_t kInt32RowSelfPitchElems = 1024U;
constexpr uint32_t kInt32RowSelfMaxRows = 8U;
constexpr uint32_t kInt32RowComputeRows = 8U;
constexpr uint32_t kInt32RowSlotBankPadBytes = 256U;
constexpr uint32_t kInt32RowSelfToExpandedPadBytes = 256U;
constexpr uint32_t kInt32RowExpandedToOtherPadBytes = 320U;
constexpr uint32_t kInt32RowOtherToOutPadBytes = 64U;
// The case2 row-broadcast fast path keeps the original 12-row 2-D DMA tile,
// but streams the dense rhs and byte output through two slots.  The consumed
// rhs slot is reused in place for Min and half 0/1 materialization, so all 12
// rows use one vector chain and no separate Min scratch is needed.  Compare
// temporarily stores its packed predicate in the output bytes that Cast then
// overwrites.  Every size below is already 32-byte aligned.
constexpr uint32_t kInt32RowSelfBytes =
    kInt32RowSelfPitchElems * sizeof(int32_t);
constexpr uint32_t kInt32RowBatchElems =
    kInt32RowSelfMaxRows * kInt32RowSelfPitchElems;
constexpr uint32_t kInt32RowPackedWidthElems = 1023U;
constexpr uint32_t kInt32RowHalfRepeatElems = 128U;
constexpr uint32_t kInt32RowGatherRepeats =
    kInt32RowBatchElems / kInt32RowHalfRepeatElems;
constexpr uint32_t kInt32RowGatherPatternStrideBytes = 32U;
constexpr uint32_t kInt32RowGatherPatternBytes =
    kInt32RowGatherRepeats * kInt32RowGatherPatternStrideBytes;
constexpr uint32_t kInt32RowCompactGatherRepeats =
    kInt32RowBatchElems / (256U / sizeof(int32_t));
constexpr uint32_t kInt32RowCompactGatherPatternBytes =
    kInt32RowCompactGatherRepeats * kInt32RowGatherPatternStrideBytes;
constexpr uint32_t kInt32RowPackedHalfOffsetBytes =
    kInt32RowBatchElems * sizeof(half) + kInt32RowSlotBankPadBytes;
constexpr uint32_t kInt32RowPackedHalfBytes =
    kInt32RowSelfMaxRows * kInt32RowPackedWidthElems * sizeof(half);
constexpr uint32_t kInt32RowOtherSlotBytes =
    kInt32RowBatchElems * sizeof(int32_t) + kInt32RowSlotBankPadBytes;
constexpr uint32_t kInt32RowOutSlotBytes =
    kInt32RowBatchElems * sizeof(uint8_t) + kInt32RowSlotBankPadBytes;
constexpr uint32_t kInt32RowUbBytes =
    kInt32RowSelfBytes + kInt32RowSelfToExpandedPadBytes +
    kInt32RowBatchElems * sizeof(int32_t) +
    kInt32RowExpandedToOtherPadBytes + 2U * kInt32RowOtherSlotBytes +
    kInt32RowOtherToOutPadBytes + 2U * kInt32RowOutSlotBytes +
    kInt32RowCompactGatherPatternBytes;
static_assert(kInt32RowGatherRepeats == 64U,
              "unexpected int32 row gather repeat count");
static_assert(kInt32RowCompactGatherRepeats == 128U,
              "unexpected compact int32 row gather repeat count");
static_assert(kInt32RowPackedHalfOffsetBytes + kInt32RowPackedHalfBytes <=
                  kInt32RowOtherSlotBytes,
              "packed half result exceeds the consumed rhs slot");
static_assert(kInt32RowUbBytes == 124544U,
              "unexpected int32 row-broadcast UB layout");
static_assert(kInt32RowUbBytes + 8U * 1024U <= 192U * 1024U,
              "int32 row-broadcast UB layout overlaps Select scratch");
// BF16 row broadcast batches keep every row on a 256-byte boundary.  Twelve
// rows cover the complete fixed-case H extent.  Only four rows are widened to
// fp32 at a time, leaving enough UB to double buffer all three pipeline queues.
constexpr uint32_t kBf16CRowMaxRows = 12U;
constexpr uint32_t kBf16CRowFloatRows = 4U;
constexpr uint32_t kBf16CRowBankPadBytes = 256U;
constexpr uint32_t kBf16CRowBankPadCount = 5U;
// Double-buffered BF16 broadcast tiles use 185600 bytes: two input queues,
// one output queue, two fp32 cast buffers, and the packed compare mask.
// This leaves 11008 bytes in the 192 KiB UB for Select mode-1 temporaries.
constexpr uint32_t kBf16BroadcastPipelineElems = 10240U;
// The dense fp32 path reuses its two consumed input slots for half precision
// bool materialization, allowing a larger tile within the 192 KiB UB.  Keep
// VECOUT double buffered so asynchronous MTE3 overlaps the next vector tile.
constexpr uint32_t kDensePipelineFloatElems = 10240U;
constexpr uint32_t kDensePipelineHalfElems = 15360U;
// Dense int8 uses two input slots and two output slots so MTE2/MTE3 can
// overlap the cast/compare/materialization vector chain.  Pad the two VECIN
// queue slots by 256 bytes so their alternating slots start eight UB banks
// apart.  Inter-buffer pads are included in the complete UB budget;
// the packed predicate reuses the output slot after allocation.
constexpr uint32_t kDensePipelineInt8Elems = 18432U;
constexpr uint32_t kInt8InputSlotPadBytes = 256U;
constexpr uint32_t kInt8BankPadSelfToOtherBytes = 512U;
constexpr uint32_t kInt8BankPadOtherToOutBytes = 512U;
constexpr uint32_t kInt8BankPadOutToSelfHalfBytes = 1024U;
constexpr uint32_t kInt8BankPadSelfHalfToOtherHalfBytes = 1024U;

// CANN 8.5 Select constraint: A2 mode1 AND mode2 need 8 KiB UB.
constexpr uint32_t kGreaterUbBytes = 192U * 1024U;
constexpr uint32_t kSelectTemporaryBytes = 8U * 1024U;
constexpr uint32_t kDenseFloatMaskBytes =
    (((kDensePipelineFloatElems + 7U) / 8U + 255U) / 256U) * 256U;
constexpr uint32_t kDenseFloatUbBytes =
    18U * kDensePipelineFloatElems + kDenseFloatMaskBytes;
constexpr uint32_t kDenseInt8UbBytes =
    10U * kDensePipelineInt8Elems + 4U * kInt8InputSlotPadBytes +
    kInt8BankPadSelfToOtherBytes + kInt8BankPadOtherToOutBytes +
    kInt8BankPadOutToSelfHalfBytes + kInt8BankPadSelfHalfToOtherHalfBytes;
static_assert(kDenseFloatUbBytes + kSelectTemporaryBytes <= kGreaterUbBytes,
              "FP32 queues/mask overlap Select temporary allowance");
static_assert(kDenseInt8UbBytes + kSelectTemporaryBytes <= kGreaterUbBytes,
              "INT8 queues/casts overlap Select temporary allowance");

static_assert(kInt32RowSelfToExpandedPadBytes >= 128U * sizeof(half),
              "bank-separation buffer must hold the invariant Select zero vector");

__aicore__ static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ static inline uint32_t GcdU32(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0U) {
        const uint32_t next = lhs % rhs;
        lhs = rhs;
        rhs = next;
    }
    return lhs == 0U ? 1U : lhs;
}

__aicore__ static inline uint32_t AlignUp32Bytes(uint32_t bytes)
{
    return ((bytes + 31U) / 32U) * 32U;
}

template <typename T>
__aicore__ static inline uint32_t AlignElems256(uint32_t elems)
{
    constexpr uint32_t elemsPerBlock = 256U / sizeof(T);
    return ((elems + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
}

template <>
__aicore__ inline uint32_t AlignElems256<uint8_t>(uint32_t elems)
{
    return ((elems + 255U) / 256U) * 256U;
}

template <typename T>
__aicore__ static inline uint32_t ElemBlock32()
{
    return 32U / sizeof(T);
}

template <typename T>
__aicore__ static inline uint32_t AlignDownElems32(uint32_t elems)
{
    const uint32_t blockElems = ElemBlock32<T>();
    return (elems / blockElems) * blockElems;
}

template <typename T>
__aicore__ static inline bool IsLocalAligned32(uint32_t elemOffset)
{
    return ((elemOffset * sizeof(T)) & 31U) == 0U;
}

template <typename T>
__aicore__ static inline uint32_t PrefixToAlign32(uint32_t elemOffset, uint32_t count)
{
    const uint32_t byteRem = (elemOffset * sizeof(T)) & 31U;
    if (byteRem == 0U) {
        return 0U;
    }
    const uint32_t bytesNeeded = 32U - byteRem;
    const uint32_t elemsNeeded = (bytesNeeded + sizeof(T) - 1U) / sizeof(T);
    return MinU32(elemsNeeded, count);
}

template <typename T>
__aicore__ static inline bool IsGmCopyAligned(uint32_t elemOffset, uint32_t elemCount)
{
    return (((elemOffset * sizeof(T)) & 31U) == 0U) &&
           (((elemCount * sizeof(T)) & 31U) == 0U);
}
