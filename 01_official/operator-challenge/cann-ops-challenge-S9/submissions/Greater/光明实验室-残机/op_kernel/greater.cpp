#include "kernel_operator.h"

#include <cstdint>
#include <type_traits>

using namespace AscendC;

namespace {

constexpr int64_t kMaxRank = 8;
constexpr int64_t kCanonRank = 4;
constexpr int32_t kBufferNum = 1;
constexpr uint32_t kTileLength = 4096;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kVectorAlignBytes = 256;
constexpr uint32_t kDirectTileBytes = 16U << 10;
constexpr uint32_t kFp16DirectDoubleBufferTileLength = 17920;
constexpr uint32_t kInt8NoBroadcastTileLength = 15872;
constexpr uint32_t kInt32DirectTQueTileLength = 10240;
constexpr uint32_t kInt32DirectStaticTileLength = 10240;
constexpr uint32_t kInt32CompactRows = 8;
constexpr uint32_t kInt32CompactLargeRows = 16;
constexpr uint32_t kInt32CompactBatchRows = 24;
constexpr uint32_t kInt32CompactRowElems = 1023;
constexpr uint32_t kInt32CompactTileLength = 8192;
constexpr uint32_t kCompareBankGapBytes = 256;
constexpr uint32_t kFp32NoBroadcastSingleBufferTileLength = 15872;
constexpr uint64_t kFp32SingleBufferMinBytes = 64U << 20;
constexpr uint32_t kBf16ScalarCachedTileLength = 12800;
constexpr uint32_t kBatchGatherTileLength = 8192;
constexpr uint32_t kBatchPatternMaxLength = 4096;
constexpr uint32_t kScalarIndexBytes = 32;
constexpr uint32_t kScalarScratchBytes = 384;
constexpr uint64_t kL2BypassMinBytes = 1U << 20;
constexpr int64_t kDirectBalancedMaxSegments = 128;
constexpr int64_t kModeNoBroadcast = 0;
constexpr int64_t kModeWorkspaceBroadcast = 1;
constexpr int64_t kModeScalarBroadcast = 2;
constexpr int64_t kModeDirectBroadcast = 3;
constexpr int64_t kModeBatchBroadcast = 4;
constexpr uint32_t kBf16RepeatedCacheTileLength = 10752;
constexpr uint32_t kBf16RepeatedBatchRows = 4;
constexpr uint32_t kBf16RepeatedBatchMaxRowElems = 2048;
constexpr uint32_t kBf16RepeatedBatchMaxCalcElems =
    kBf16RepeatedBatchRows * kBf16RepeatedBatchMaxRowElems;

constexpr uint32_t AlignUpConst(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

constexpr uint32_t kFp32SingleX1Addr = 0;
constexpr uint32_t kFp32SingleX2Addr =
    AlignUpConst(kFp32SingleX1Addr + kFp32NoBroadcastSingleBufferTileLength * sizeof(float), 256U) + 256U;
constexpr uint32_t kFp32SingleMaskAddr =
    AlignUpConst(kFp32SingleX2Addr + kFp32NoBroadcastSingleBufferTileLength * sizeof(float), 256U) + 256U;
constexpr uint32_t kFp32SingleMaskBytes =
    AlignUpConst((kFp32NoBroadcastSingleBufferTileLength + 7U) / 8U, kAlignBytes);
constexpr uint32_t kFp32SingleValueAddr =
    AlignUpConst(kFp32SingleMaskAddr + kFp32SingleMaskBytes, 256U);
constexpr uint32_t kFp32SingleOutputAddr =
    AlignUpConst(kFp32SingleValueAddr + kFp32NoBroadcastSingleBufferTileLength * sizeof(half), 256U) + 256U;
constexpr uint32_t kFp32SingleBufferEnd =
    kFp32SingleOutputAddr + kFp32NoBroadcastSingleBufferTileLength * sizeof(uint8_t);
static_assert(kFp32SingleBufferEnd <= (192U << 10), "FP32 single-buffer layout exceeds the A2 UB capacity");

constexpr uint32_t kBf16RepeatedFull0Addr = 0;
constexpr uint32_t kBf16RepeatedValueAddr =
    kBf16RepeatedFull0Addr + kBf16RepeatedCacheTileLength * sizeof(bfloat16_t);
constexpr uint32_t kBf16RepeatedFull1Addr =
    2U * kBf16RepeatedCacheTileLength * sizeof(bfloat16_t) + kCompareBankGapBytes;
constexpr uint32_t kBf16RepeatedBcastAddr =
    kBf16RepeatedFull1Addr + kBf16RepeatedCacheTileLength * sizeof(bfloat16_t);
constexpr uint32_t kBf16RepeatedMaskAddr =
    kBf16RepeatedBcastAddr + kBf16RepeatedCacheTileLength * sizeof(bfloat16_t);
constexpr uint32_t kBf16RepeatedMaskBytes =
    AlignUpConst((kBf16RepeatedCacheTileLength + 7U) / 8U, kAlignBytes);
constexpr uint32_t kBf16RepeatedOutputAddr =
    AlignUpConst(kBf16RepeatedMaskAddr + kBf16RepeatedMaskBytes, 256U) +
    kCompareBankGapBytes;
constexpr uint32_t kBf16RepeatedFullFloatAddr =
    AlignUpConst(kBf16RepeatedOutputAddr +
                     kBf16RepeatedCacheTileLength * sizeof(uint8_t),
                 256U) +
    kCompareBankGapBytes;
constexpr uint32_t kBf16RepeatedBcastFloatAddr =
    AlignUpConst(kBf16RepeatedFullFloatAddr +
                     kBf16RepeatedCacheTileLength * sizeof(float),
                 256U) +
    kCompareBankGapBytes;
constexpr uint32_t kBf16RepeatedBufferEnd =
    kBf16RepeatedBcastFloatAddr + kBf16RepeatedCacheTileLength * sizeof(float);
static_assert(kBf16RepeatedBufferEnd <= (192U << 10),
              "BF16 repeated-cache static layout exceeds the A2 UB capacity");

struct Bf16RepeatedBatchLayout {
    static constexpr uint32_t kFullBytes =
        kBf16RepeatedBatchMaxCalcElems * sizeof(bfloat16_t);
    static constexpr uint32_t kFull0Addr = 0;
    static constexpr uint32_t kFull1Addr =
        AlignUpConst(kFull0Addr + kFullBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kBcastAddr =
        AlignUpConst(kFull1Addr + kFullBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kFullFloatAddr =
        AlignUpConst(kBcastAddr +
                         kBf16RepeatedBatchMaxRowElems * sizeof(bfloat16_t),
                     256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kBcastFloatAddr =
        AlignUpConst(kFullFloatAddr +
                         kBf16RepeatedBatchMaxCalcElems * sizeof(float),
                     256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kMaskAddr =
        AlignUpConst(kBcastFloatAddr +
                         kBf16RepeatedBatchMaxCalcElems * sizeof(float),
                     256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kMaskBytes =
        AlignUpConst((kBf16RepeatedBatchMaxCalcElems + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kValueAddr =
        AlignUpConst(kMaskAddr + kMaskBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kOutput0Addr =
        AlignUpConst(kValueAddr +
                         kBf16RepeatedBatchMaxCalcElems * sizeof(half),
                     256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kOutput1Addr =
        AlignUpConst(kOutput0Addr +
                         kBf16RepeatedBatchMaxCalcElems * sizeof(uint8_t),
                     256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kBufferEnd =
        kOutput1Addr + kBf16RepeatedBatchMaxCalcElems * sizeof(uint8_t);
};

static_assert(Bf16RepeatedBatchLayout::kBufferEnd <= (192U << 10),
              "BF16 repeated-row batch layout exceeds the A2 UB capacity");

struct Int32CompactStaticLayout {
    static constexpr uint32_t kChunkLength = kInt32CompactTileLength;
    static constexpr uint32_t kChunkValidLength =
        kInt32CompactRows * kInt32CompactRowElems;
    static constexpr uint32_t kChunkBytes = kChunkLength * sizeof(int32_t);
    static constexpr uint32_t kChunkPaddingBytes =
        (kChunkLength - kChunkValidLength) * sizeof(int32_t);
    static constexpr uint32_t kLargeLength = 2U * kChunkLength;
    static constexpr uint32_t kLargeBytes = kLargeLength * sizeof(int32_t);
    static constexpr uint32_t kLargeAddr = 0;
    static constexpr uint32_t kBcast1Addr =
        kLargeAddr + kLargeBytes + kCompareBankGapBytes - kChunkPaddingBytes;
    static constexpr uint32_t kBcast0Addr =
        kBcast1Addr + kChunkBytes + kAlignBytes;
    static constexpr uint32_t kSmallAddr =
        kBcast0Addr + kChunkBytes + kCompareBankGapBytes;
    static constexpr uint32_t kMaskAddr =
        kSmallAddr + kChunkBytes + kCompareBankGapBytes;
    static constexpr uint32_t kMaskBytes =
        AlignUpConst((kChunkLength + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kAllMaskBytes = 2U * kMaskBytes;
    static constexpr uint32_t kZeroAddr =
        AlignUpConst(kMaskAddr + kAllMaskBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kZeroBytes = kChunkLength * sizeof(half);
    static constexpr uint32_t kBcastScratchAddr =
        AlignUpConst(kZeroAddr + kZeroBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kBcastScratchLength =
        AlignUpConst(kInt32CompactRowElems, kAlignBytes / sizeof(int32_t));
    static constexpr uint32_t kBufferEnd =
        kBcastScratchAddr + kBcastScratchLength * sizeof(int32_t);
};

static_assert(Int32CompactStaticLayout::kBufferEnd <= (192U << 10),
              "INT32 compact static layout exceeds the A2 UB capacity");
static_assert((Int32CompactStaticLayout::kBcast0Addr -
               Int32CompactStaticLayout::kLargeAddr) % (32U << 10) ==
                  kCompareBankGapBytes,
              "large chunk0 and broadcast0 must retain the 256-byte bank phase");
static_assert((Int32CompactStaticLayout::kBcast1Addr -
               Int32CompactStaticLayout::kChunkValidLength * sizeof(int32_t)) %
                      (32U << 10) ==
                  kCompareBankGapBytes,
              "large chunk1 and broadcast1 must retain the 256-byte bank phase");
static_assert((Int32CompactStaticLayout::kSmallAddr -
               Int32CompactStaticLayout::kBcast0Addr) % (32U << 10) ==
                  kCompareBankGapBytes,
              "small slot and broadcast0 must retain the 256-byte bank phase");

template <typename T>
__aicore__ inline uint32_t AlignForVector(uint32_t len)
{
    uint32_t align = kVectorAlignBytes / sizeof(T);
    if (align == 0) {
        align = 1;
    }
    return ((len + align - 1) / align) * align;
}

template <typename T>
__aicore__ inline uint32_t AlignForHalfCompare(uint32_t len)
{
    const uint32_t sourceAligned = AlignForVector<T>(len);
    const uint32_t halfAligned = AlignForVector<half>(len);
    return sourceAligned > halfAligned ? sourceAligned : halfAligned;
}

template <typename T>
__aicore__ inline constexpr bool NeedHalfCastQueues()
{
    return std::is_same<T, int8_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool NeedFloatCastQueues()
{
    return std::is_same<T, bfloat16_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool NeedInt32TmpQueue()
{
    return std::is_same<T, int32_t>::value;
}

template <typename T>
__aicore__ inline constexpr uint32_t TileLengthForType()
{
    return std::is_same<T, half>::value ? 12288U : kTileLength;
}

template <typename T>
__aicore__ inline constexpr uint32_t DirectDoubleBufferTileLengthForType()
{
    if constexpr (std::is_same<T, half>::value) {
        return kFp16DirectDoubleBufferTileLength;
    } else if constexpr (std::is_same<T, float>::value) {
        return 10240U;
    } else if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, int32_t>::value) {
        return 7168U;
    } else {
        return 10240U;
    }
}

template <typename T>
__aicore__ inline constexpr uint32_t NoBroadcastDoubleBufferTileLengthForType()
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return 7424U;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return kInt8NoBroadcastTileLength;
    } else {
        return DirectDoubleBufferTileLengthForType<T>();
    }
}

template <typename T>
__aicore__ inline constexpr uint32_t NoBroadcastStaticTileLengthForType()
{
    if constexpr (std::is_same<T, half>::value) {
        return kFp16DirectDoubleBufferTileLength;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return 7424U;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return 7168U;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return kInt8NoBroadcastTileLength;
    } else {
        return kFp32NoBroadcastSingleBufferTileLength;
    }
}

template <typename T>
__aicore__ inline constexpr uint32_t DirectBroadcastStaticTileLengthForType()
{
    if constexpr (std::is_same<T, int32_t>::value) {
        return kInt32DirectStaticTileLength;
    } else {
        return NoBroadcastStaticTileLengthForType<T>();
    }
}

template <typename T, uint32_t TileLength>
struct GreaterStaticLayout {
    static constexpr uint32_t kTileLength = TileLength;
    static constexpr uint32_t kSourceBytes = kTileLength * sizeof(T);
    static constexpr uint32_t kAuxBytes =
        std::is_same<T, bfloat16_t>::value ? kTileLength * sizeof(float)
        : std::is_same<T, int8_t>::value   ? kTileLength * sizeof(half)
        : std::is_same<T, int32_t>::value  ? kTileLength * sizeof(int32_t)
                                           : 0U;
    static constexpr uint32_t kAuxCount =
        (std::is_same<T, bfloat16_t>::value || std::is_same<T, int8_t>::value)
            ? 2U
            : (std::is_same<T, int32_t>::value ? 1U : 0U);
    static constexpr uint32_t kX1Addr = 0;
    static constexpr uint32_t kX2Addr = AlignUpConst(kX1Addr + kSourceBytes, 256U) + 256U;
    static constexpr uint32_t kAux1Addr = AlignUpConst(kX2Addr + kSourceBytes, 256U);
    static constexpr uint32_t kAux2Addr = AlignUpConst(kAux1Addr + kAuxBytes, 256U) + 256U;
    static constexpr uint32_t kAfterAuxAddr =
        kAuxCount == 0U ? kX2Addr + kSourceBytes
        : kAuxCount == 1U ? kAux1Addr + kAuxBytes
                          : kAux2Addr + kAuxBytes;
    static constexpr uint32_t kMaskAddr = AlignUpConst(kAfterAuxAddr, 256U);
    static constexpr uint32_t kMaskBytes = AlignUpConst((kTileLength + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kValueAddr = AlignUpConst(kMaskAddr + kMaskBytes, 256U) + 256U;
    static constexpr uint32_t kOutputAddr =
        AlignUpConst(kValueAddr + kTileLength * sizeof(half), 256U);
    static constexpr uint32_t kBufferEnd = kOutputAddr + kTileLength * sizeof(uint8_t);
};

template <uint32_t TileLength>
struct GreaterInt32DirectStaticLayout {
    static constexpr uint32_t kTileLength = TileLength;
    static constexpr uint32_t kSourceBytes = kTileLength * sizeof(int32_t);
    static constexpr uint32_t kX1Addr = 0;
    static constexpr uint32_t kX2Addr = AlignUpConst(kX1Addr + kSourceBytes, 256U) + 256U;
    static constexpr uint32_t kAux1Addr = AlignUpConst(kX2Addr + kSourceBytes, 256U);
    static constexpr uint32_t kAux2Addr = kAux1Addr;
    static constexpr uint32_t kMaskAddr = AlignUpConst(kAux1Addr + kSourceBytes, 256U);
    static constexpr uint32_t kMaskBytes = AlignUpConst((kTileLength + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kValueAddr = kAux1Addr;
    static constexpr uint32_t kOutputAddr = AlignUpConst(kMaskAddr + kMaskBytes, 256U) + 256U;
    static constexpr uint32_t kBufferEnd = kOutputAddr + kTileLength * sizeof(uint8_t);
};

template <typename T>
using NoBroadcastStaticLayout = GreaterStaticLayout<T, NoBroadcastStaticTileLengthForType<T>()>;

template <typename T>
using DirectBroadcastStaticLayout = typename std::conditional<
    std::is_same<T, int32_t>::value,
    GreaterInt32DirectStaticLayout<DirectBroadcastStaticTileLengthForType<T>()>,
    GreaterStaticLayout<T, DirectBroadcastStaticTileLengthForType<T>()>>::type;

template <typename T>
__aicore__ inline constexpr uint32_t ScalarBroadcastStaticTileLengthForType()
{
    if constexpr (std::is_same<T, half>::value) {
        return 32768U;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return kBf16ScalarCachedTileLength;
    } else if constexpr (std::is_same<T, float>::value) {
        return kFp32NoBroadcastSingleBufferTileLength;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return 7168U;
    } else {
        return 10240U;
    }
}

template <typename T>
struct ScalarBroadcastStaticLayout {
    static constexpr uint32_t kTileLength = ScalarBroadcastStaticTileLengthForType<T>();
    static constexpr uint32_t kCompareBytes =
        std::is_same<T, bfloat16_t>::value ? kTileLength * sizeof(float)
        : std::is_same<T, int8_t>::value   ? kTileLength * sizeof(half)
        : std::is_same<T, int32_t>::value  ? kTileLength * sizeof(int32_t)
                                           : 0U;
    static constexpr bool kNeedsVectorScalar =
        std::is_same<T, bfloat16_t>::value || std::is_same<T, int8_t>::value ||
        std::is_same<T, int32_t>::value;
    static constexpr uint32_t kFullAddr = 0;
    static constexpr uint32_t kAuxAddr =
        AlignUpConst(kFullAddr + kTileLength * sizeof(T), 256U);
    static constexpr uint32_t kScalarVectorAddr =
        AlignUpConst(kAuxAddr + kCompareBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kAfterCompareAddr =
        kNeedsVectorScalar ? kScalarVectorAddr + kCompareBytes
                           : kFullAddr + kTileLength * sizeof(T);
    static constexpr uint32_t kMaskAddr = AlignUpConst(kAfterCompareAddr, 256U);
    static constexpr uint32_t kMaskBytes =
        AlignUpConst((kTileLength + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kValueAddr =
        AlignUpConst(kMaskAddr + kMaskBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kOutputAddr =
        AlignUpConst(kValueAddr + kTileLength * sizeof(half), 256U);
    static constexpr uint32_t kScalarSourceAddr =
        AlignUpConst(kOutputAddr + kTileLength * sizeof(uint8_t), 256U);
    static constexpr uint32_t kScalarIndexAddr = kScalarSourceAddr + kAlignBytes;
    static constexpr uint32_t kScalarScratchAddr = kScalarIndexAddr + kScalarIndexBytes;
    static constexpr uint32_t kBufferEnd = kScalarScratchAddr + kScalarScratchBytes;
};

struct BatchBroadcastStaticLayout {
    static constexpr uint32_t kTileLength = kBatchGatherTileLength;
    static constexpr uint32_t kFullAddr = 0;
    static constexpr uint32_t kPatternAddr =
        AlignUpConst(kFullAddr + kTileLength * sizeof(half), 256U);
    static constexpr uint32_t kBcastAddr =
        AlignUpConst(kPatternAddr + kBatchPatternMaxLength * sizeof(half), 256U) +
        kCompareBankGapBytes;
    static constexpr uint32_t kMaskAddr =
        AlignUpConst(kBcastAddr + kTileLength * sizeof(half), 256U);
    static constexpr uint32_t kMaskBytes =
        AlignUpConst((kTileLength + 7U) / 8U, kAlignBytes);
    static constexpr uint32_t kValueAddr =
        AlignUpConst(kMaskAddr + kMaskBytes, 256U) + kCompareBankGapBytes;
    static constexpr uint32_t kOutputAddr =
        AlignUpConst(kValueAddr + kTileLength * sizeof(half), 256U);
    static constexpr uint32_t kBufferEnd = kOutputAddr + kTileLength * sizeof(uint8_t);
};

template <typename T>
__aicore__ inline constexpr uint32_t ScalarCacheElemBytesForType()
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return sizeof(float);
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return sizeof(half);
    } else {
        return sizeof(T);
    }
}

template <typename T>
__aicore__ inline constexpr bool UseDedicatedScalarCache()
{
    return std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value ||
           std::is_same<T, int8_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool UseCompareScalar()
{
    return std::is_same<T, half>::value || std::is_same<T, float>::value;
}

template <typename T>
__aicore__ inline constexpr uint32_t ScalarBankGapBytesForType()
{
    if constexpr (std::is_same<T, half>::value) {
        return 2944U;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return 256U;
    } else {
        return 0U;
    }
}

__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline uint32_t AlignUpBytes(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> MakePadParams(uint32_t leftPadding, uint32_t rightPadding, T paddingValue)
{
    return {
        (leftPadding != 0U) || (rightPadding != 0U),
        static_cast<uint8_t>(leftPadding),
        static_cast<uint8_t>(rightPadding),
        paddingValue,
    };
}

template <typename T>
__aicore__ inline uint16_t PackDupByte(T value)
{
    const uint8_t byteValue = static_cast<uint8_t>(value);
    return static_cast<uint16_t>(byteValue) | (static_cast<uint16_t>(byteValue) << 8);
}

template <typename T>
class BroadcastMaterializer {
public:
    __aicore__ inline void Init(TPipe* pipe, uint32_t tileElems)
    {
        tileElems_ = tileElems == 0 ? 1 : tileElems;
        pipe->InitBuffer(queue_, 1, tileElems_ * sizeof(T));
    }

    __aicore__ inline void BroadcastIfNeeded(GM_ADDR srcAddr, int64_t srcElems, GM_ADDR dstAddr, GM_ADDR scratchAddr,
                                             const int64_t (&canonIn)[kCanonRank], const int64_t (&canonOut)[kCanonRank])
    {
        if (srcElems <= 0 || dstAddr == nullptr) {
            return;
        }

        GlobalTensor<T> srcGm;
        GlobalTensor<T> dstGm;
        GlobalTensor<T> scratchGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcAddr), srcElems);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstAddr));
        scratchGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scratchAddr));

        int64_t broadcastCount = 0;
        for (int64_t i = 0; i < kCanonRank; ++i) {
            if (canonIn[i] != canonOut[i]) {
                ++broadcastCount;
            }
        }

        GlobalTensor<T>* currentSrc = &srcGm;
        GlobalTensor<T>* currentDst = (broadcastCount % 2 == 1) ? &dstGm : &scratchGm;
        int64_t highSize = canonIn[0] * canonIn[1] * canonIn[2] * canonIn[3];
        int64_t lowSize = 1;
        int64_t pending = broadcastCount;
        for (int64_t dim = 0; dim < kCanonRank; ++dim) {
            highSize /= canonIn[dim];
            if (canonIn[dim] != canonOut[dim]) {
                ExpandOneDim(*currentSrc, *currentDst, highSize, canonOut[dim], lowSize);
                --pending;
                if (pending > 0) {
                    CrossCoreSetFlag<0, PIPE_MTE3>(0);
                    CrossCoreWaitFlag(0);
                    GlobalTensor<T>* nextSrc = currentDst;
                    currentDst = (currentDst == &dstGm) ? &scratchGm : &dstGm;
                    currentSrc = nextSrc;
                }
            }
            lowSize *= canonOut[dim];
        }
    }

private:
    __aicore__ inline void CopyInPadExact(LocalTensor<T>& local, GlobalTensor<T>& gm, int64_t offset, uint32_t len)
    {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = MakePadParams<T>(0, 0, static_cast<T>(0));
        DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyOutExact(GlobalTensor<T>& gm, int64_t offset, LocalTensor<T>& local, uint32_t len)
    {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, copyParams);
    }

    __aicore__ inline void ExpandOneDim(GlobalTensor<T>& srcGm, GlobalTensor<T>& dstGm, int64_t highSize,
                                        int64_t midSize, int64_t lowSize)
    {
        if (highSize <= 0 || midSize <= 0 || lowSize <= 0) {
            return;
        }
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t dataBlockElems = 512 / sizeof(T);
        const int64_t totalRows = highSize * midSize;
        if (lowSize <= static_cast<int64_t>(tileElems_) && highSize < blockNum && totalRows >= 512) {
            int64_t row = totalRows * blockIdx / blockNum;
            const int64_t rowEnd = totalRows * (blockIdx + 1) / blockNum;
            while (row < rowEnd) {
                const int64_t high = row / midSize;
                const int64_t repFrom = row % midSize;
                int64_t repTo = midSize;
                const int64_t highRowEnd = (high + 1) * midSize;
                if (rowEnd < highRowEnd) {
                    repTo = rowEnd - high * midSize;
                }

                LocalTensor<T> local = queue_.AllocTensor<T>();
                CopyInPadExact(local, srcGm, high * lowSize, static_cast<uint32_t>(lowSize));
                queue_.EnQue(local);
                local = queue_.DeQue<T>();
                const int64_t dstBase = high * midSize * lowSize;
                for (int64_t rep = repFrom; rep < repTo; ++rep) {
                    CopyOutExact(dstGm, dstBase + rep * lowSize, local, static_cast<uint32_t>(lowSize));
                }
                queue_.FreeTensor(local);
                row = high * midSize + repTo;
            }
            return;
        }

        int64_t highFrom = 0;
        int64_t highTo = highSize;
        int64_t lowFrom = 0;
        int64_t lowTo = lowSize;
        if (lowSize > static_cast<int64_t>(tileElems_)) {
            const int64_t lowBlocks = (lowSize + dataBlockElems - 1) / dataBlockElems;
            lowFrom = (lowBlocks * blockIdx / blockNum) * dataBlockElems;
            lowTo = (lowBlocks * (blockIdx + 1) / blockNum) * dataBlockElems;
            if (lowTo > lowSize) {
                lowTo = lowSize;
            }
        } else {
            highFrom = highSize * blockIdx / blockNum;
            highTo = highSize * (blockIdx + 1) / blockNum;
        }
        if (highFrom >= highTo || lowFrom >= lowTo) {
            return;
        }

        for (int64_t high = highFrom; high < highTo; ++high) {
            const int64_t srcBase = high * lowSize;
            for (int64_t low = lowFrom; low < lowTo; low += tileElems_) {
                uint32_t len = static_cast<uint32_t>(lowTo - low);
                if (len > tileElems_) {
                    len = tileElems_;
                }
                LocalTensor<T> local = queue_.AllocTensor<T>();
                CopyInPadExact(local, srcGm, srcBase + low, len);
                queue_.EnQue(local);
                local = queue_.DeQue<T>();
                const int64_t dstBase = (high * midSize) * lowSize + low;
                for (int64_t rep = 0; rep < midSize; ++rep) {
                    CopyOutExact(dstGm, dstBase + rep * lowSize, local, len);
                }
                queue_.FreeTensor(local);
            }
        }
    }

private:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queue_;
    uint32_t tileElems_ = 1;
};

class KernelGreaterBf16RepeatedCache {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GreaterTilingData& tiling)
    {
        totalElems_ = tiling.total_elems;
        directLowSize_ = tiling.direct_low_size;
        directTotalSegments_ = tiling.direct_total_segments;
        directOuterRank_ = tiling.direct_outer_rank;
        directBroadcastInputIndex_ = tiling.direct_broadcast_input_index;
        directOuterDims_[0] = tiling.direct_outer_dim0;
        directOuterDims_[1] = tiling.direct_outer_dim1;
        directOuterDims_[2] = tiling.direct_outer_dim2;
        directOuterDims_[3] = tiling.direct_outer_dim3;
        directOutStrides_[0] = tiling.direct_out_stride0;
        directOutStrides_[1] = tiling.direct_out_stride1;
        directOutStrides_[2] = tiling.direct_out_stride2;
        directOutStrides_[3] = tiling.direct_out_stride3;
        directSrcStrides_[0] = tiling.direct_src_stride0;
        directSrcStrides_[1] = tiling.direct_src_stride1;
        directSrcStrides_[2] = tiling.direct_src_stride2;
        directSrcStrides_[3] = tiling.direct_src_stride3;

        if (directBroadcastInputIndex_ == 1) {
            bcastGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x1);
            fullGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x2);
        } else {
            fullGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x1);
            bcastGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x2);
        }
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        if (static_cast<uint64_t>(totalElems_) * sizeof(bfloat16_t) >= kL2BypassMinBytes) {
            // Case3 重复广播路径：full 流是一次性冷读，只有读方向需要关 L2；
            // 写方向保持默认（原先单参数调用等价于 CacheRwMode::RW）。
            // bcastGm_ 被复用 12 次，刻意保持默认缓存。
            fullGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
    }

    __aicore__ inline void Process()
    {
        if (directLowSize_ <= 0 || directTotalSegments_ <= 0 || directOuterRank_ <= 0 ||
            directOuterRank_ > kCanonRank) {
            return;
        }

        int64_t segmentStart = 0;
        int64_t segmentEnd = 0;
        int64_t lowStart = 0;
        int64_t lowEnd = 0;
        if (!ResolveDirectCoreRange(segmentStart, segmentEnd, lowStart, lowEnd)) {
            return;
        }

        InitSocState();
        if (CanUseBatchedRows(lowStart, lowEnd)) {
            ProcessBatchedRows(segmentStart, segmentEnd);
            return;
        }

        LocalTensor<bfloat16_t> fullLocal[2] = {
            LocalTensor<bfloat16_t>(TPosition::VECCALC, kBf16RepeatedFull0Addr,
                                    kBf16RepeatedCacheTileLength),
            LocalTensor<bfloat16_t>(TPosition::VECCALC, kBf16RepeatedFull1Addr,
                                    kBf16RepeatedCacheTileLength),
        };
        LocalTensor<bfloat16_t> bcastLocal(TPosition::VECCALC, kBf16RepeatedBcastAddr,
                                           kBf16RepeatedCacheTileLength);
        LocalTensor<half> valueLocal(TPosition::VECCALC, kBf16RepeatedValueAddr,
                                     kBf16RepeatedCacheTileLength);
        LocalTensor<float> fullFloat(TPosition::VECCALC, kBf16RepeatedFullFloatAddr,
                                     kBf16RepeatedCacheTileLength);
        LocalTensor<float> bcastFloat(TPosition::VECCALC, kBf16RepeatedBcastFloatAddr,
                                      kBf16RepeatedCacheTileLength);
        LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, kBf16RepeatedMaskAddr,
                                       kBf16RepeatedMaskBytes);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, kBf16RepeatedOutputAddr,
                                         kBf16RepeatedCacheTileLength);

        bool fullReleasePending[2] = {false, false};
        bool bcastReleasePending = false;
        bool outputInFlight = false;
        const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
        int64_t segment = segmentStart;
        while (segment < segmentEnd) {
            int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
            if (groupEnd > segmentEnd) {
                groupEnd = segmentEnd;
            }
            const int64_t srcSegment = ComputeDirectSourceSegment(segment);
            const int64_t bcastBase = srcSegment * directLowSize_;

            for (int64_t chunkOffset = lowStart; chunkOffset < lowEnd;
                 chunkOffset += kBf16RepeatedCacheTileLength) {
                uint32_t len = static_cast<uint32_t>(lowEnd - chunkOffset);
                if (len > kBf16RepeatedCacheTileLength) {
                    len = kBf16RepeatedCacheTileLength;
                }
                const uint32_t alignedLen = AlignForHalfCompare<bfloat16_t>(len);

                if (bcastReleasePending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                    bcastReleasePending = false;
                }
                CopyIn(bcastLocal, bcastGm_, bcastBase + chunkOffset, len);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

                const int64_t segmentCount = groupEnd - segment;
                PrepareFull(fullLocal[0], 0, segment * directLowSize_ + chunkOffset, len,
                            fullReleasePending);
                if (segmentCount > 1) {
                    PrepareFull(fullLocal[1], 1, (segment + 1) * directLowSize_ + chunkOffset, len,
                                fullReleasePending);
                }

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                Cast(bcastFloat, bcastLocal, RoundMode::CAST_NONE, alignedLen);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
                bcastReleasePending = true;

                for (int64_t segmentIndex = 0; segmentIndex < segmentCount; ++segmentIndex) {
                    const uint32_t slot = static_cast<uint32_t>(segmentIndex & 1);
                    const int32_t fullEvent = FullEvent(slot);
                    WaitFlag<HardEvent::MTE2_V>(fullEvent);
                    Cast(fullFloat, fullLocal[slot], RoundMode::CAST_NONE, alignedLen);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(fullEvent);
                    fullReleasePending[slot] = true;

                    const int64_t prefetchIndex = segmentIndex + 2;
                    if (prefetchIndex < segmentCount) {
                        PrepareFull(fullLocal[slot], slot,
                                    (segment + prefetchIndex) * directLowSize_ + chunkOffset, len,
                                    fullReleasePending);
                    }

                    if (directBroadcastInputIndex_ == 1) {
                        Compare(maskLocal, bcastFloat, fullFloat, CMPMODE::GT, alignedLen);
                    } else {
                        Compare(maskLocal, fullFloat, bcastFloat, CMPMODE::GT, alignedLen);
                    }
                    PipeBarrier<PIPE_V>();
                    Duplicate(valueLocal, static_cast<half>(1.0f), alignedLen);
                    PipeBarrier<PIPE_V>();
                    Select(valueLocal, maskLocal, valueLocal, static_cast<half>(0.0f),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
                    PipeBarrier<PIPE_V>();

                    if (outputInFlight) {
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                    Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    const int64_t outputOffset = (segment + segmentIndex) * directLowSize_ + chunkOffset;
                    CopyOut(outputOffset, outputLocal, len);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    outputInFlight = true;
                }
            }
            segment = groupEnd;
        }

        if (bcastReleasePending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (fullReleasePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(FullEvent(slot));
            }
        }
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline bool CanUseBatchedRows(int64_t lowStart, int64_t lowEnd) const
    {
        return lowStart == 0 && lowEnd == directLowSize_ && directLowSize_ > 0 &&
               directLowSize_ <= kBf16RepeatedBatchMaxRowElems;
    }

    __aicore__ inline void CopyBatchedBcastRow(LocalTensor<bfloat16_t>& local,
                                                int64_t offset, uint32_t rowElems,
                                                uint32_t rowStride)
    {
        const uint32_t rowBytes = rowElems * sizeof(bfloat16_t);
        const uint32_t alignedRowElems =
            AlignUpBytes(rowBytes, kAlignBytes) / sizeof(bfloat16_t);
        const uint32_t requestedPadding = rowStride - rowElems;
        constexpr uint32_t maxPadding = kAlignBytes / sizeof(bfloat16_t);
        const uint8_t rightPadding = static_cast<uint8_t>(
            requestedPadding <= maxPadding ? requestedPadding
                                           : alignedRowElems - rowElems);
        const DataCopyExtParams copyParams = {1, rowBytes, 0, 0, 0};
        const DataCopyPadExtParams<bfloat16_t> padParams =
            MakePadParams<bfloat16_t>(0, rightPadding, static_cast<bfloat16_t>(0));
        DataCopyPad(local, bcastGm_[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyBatchedFullRows(LocalTensor<bfloat16_t>& local,
                                                int64_t segment, uint32_t rowCount,
                                                uint32_t rowElems, uint32_t rowStride)
    {
        const uint32_t rowBytes = rowElems * sizeof(bfloat16_t);
        const uint32_t requestedPadding = rowStride - rowElems;
        constexpr uint32_t maxPadding = kAlignBytes / sizeof(bfloat16_t);
        if (requestedPadding <= maxPadding) {
            const DataCopyExtParams copyParams = {
                static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
            const DataCopyPadExtParams<bfloat16_t> padParams =
                MakePadParams<bfloat16_t>(0, requestedPadding,
                                          static_cast<bfloat16_t>(0));
            DataCopyPad(local, fullGm_[segment * directLowSize_], copyParams,
                        padParams);
            return;
        }

        const uint32_t alignedRowElems =
            AlignUpBytes(rowBytes, kAlignBytes) / sizeof(bfloat16_t);
        const uint8_t rightPadding =
            static_cast<uint8_t>(alignedRowElems - rowElems);
        const DataCopyExtParams copyParams = {1, rowBytes, 0, 0, 0};
        const DataCopyPadExtParams<bfloat16_t> padParams =
            MakePadParams<bfloat16_t>(0, rightPadding, static_cast<bfloat16_t>(0));
        for (uint32_t row = 0; row < rowCount; ++row) {
            DataCopyPad(local[row * rowStride],
                        fullGm_[(segment + row) * directLowSize_],
                        copyParams, padParams);
        }
    }

    __aicore__ inline void CopyBatchedOutputRows(int64_t segment,
                                                 LocalTensor<uint8_t>& local,
                                                 uint32_t rowCount,
                                                 uint32_t rowElems,
                                                 uint32_t rowStride)
    {
        const uint32_t naturalRowStride = AlignUpBytes(rowElems, kAlignBytes);
        const uint32_t sourceStride =
            (rowStride - naturalRowStride) / kAlignBytes;
        const DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowCount), rowElems, sourceStride, 0, 0};
        DataCopyPad(yGm_[segment * directLowSize_], local, copyParams);
    }

    __aicore__ inline void ExpandBatchedBcastRows(LocalTensor<float>& bcastFloat,
                                                   uint32_t rowStride,
                                                   uint32_t rowCount)
    {
        constexpr uint32_t repeatElems = kVectorAlignBytes / sizeof(float);
        uint32_t builtRows = 1;
        while (builtRows < rowCount) {
            uint32_t copyRows = builtRows;
            if (copyRows > rowCount - builtRows) {
                copyRows = rowCount - builtRows;
            }
            const uint32_t copyElems = copyRows * rowStride;
            const uint8_t repeatTimes =
                static_cast<uint8_t>(copyElems / repeatElems);
            Copy(bcastFloat[builtRows * rowStride], bcastFloat, repeatElems,
                 repeatTimes, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            builtRows += copyRows;
        }
    }

    __aicore__ inline void PrepareBatchedFull(LocalTensor<bfloat16_t>& local,
                                               uint32_t slot, int64_t segment,
                                               uint32_t rowCount, uint32_t rowElems,
                                               uint32_t rowStride,
                                               bool (&releasePending)[2])
    {
        const int32_t event = FullEvent(slot);
        if (releasePending[slot]) {
            WaitFlag<HardEvent::V_MTE2>(event);
            releasePending[slot] = false;
        }
        CopyBatchedFullRows(local, segment, rowCount, rowElems, rowStride);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void ProcessBatchedRows(int64_t segmentStart, int64_t segmentEnd)
    {
        const uint32_t rowElems = static_cast<uint32_t>(directLowSize_);
        constexpr uint32_t vectorRowAlign = kVectorAlignBytes / sizeof(float);
        const uint32_t rowStride = AlignUpBytes(rowElems, vectorRowAlign);
        LocalTensor<bfloat16_t> fullLocal[2] = {
            LocalTensor<bfloat16_t>(TPosition::VECCALC,
                                    Bf16RepeatedBatchLayout::kFull0Addr,
                                    kBf16RepeatedBatchMaxCalcElems),
            LocalTensor<bfloat16_t>(TPosition::VECCALC,
                                    Bf16RepeatedBatchLayout::kFull1Addr,
                                    kBf16RepeatedBatchMaxCalcElems),
        };
        LocalTensor<bfloat16_t> bcastLocal(
            TPosition::VECCALC, Bf16RepeatedBatchLayout::kBcastAddr,
            kBf16RepeatedBatchMaxRowElems);
        LocalTensor<float> fullFloat(
            TPosition::VECCALC, Bf16RepeatedBatchLayout::kFullFloatAddr,
            kBf16RepeatedBatchMaxCalcElems);
        LocalTensor<float> bcastFloat(
            TPosition::VECCALC, Bf16RepeatedBatchLayout::kBcastFloatAddr,
            kBf16RepeatedBatchMaxCalcElems);
        LocalTensor<uint8_t> maskLocal(
            TPosition::VECCALC, Bf16RepeatedBatchLayout::kMaskAddr,
            Bf16RepeatedBatchLayout::kMaskBytes);
        LocalTensor<half> valueLocal(
            TPosition::VECCALC, Bf16RepeatedBatchLayout::kValueAddr,
            kBf16RepeatedBatchMaxCalcElems);
        LocalTensor<uint8_t> outputLocal[2] = {
            LocalTensor<uint8_t>(TPosition::VECCALC,
                                 Bf16RepeatedBatchLayout::kOutput0Addr,
                                 kBf16RepeatedBatchMaxCalcElems),
            LocalTensor<uint8_t>(TPosition::VECCALC,
                                 Bf16RepeatedBatchLayout::kOutput1Addr,
                                 kBf16RepeatedBatchMaxCalcElems),
        };

        bool fullReleasePending[2] = {false, false};
        bool outputInFlight[2] = {false, false};
        bool bcastReleasePending = false;
        const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
        int64_t segment = segmentStart;
        while (segment < segmentEnd) {
            int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
            if (groupEnd > segmentEnd) {
                groupEnd = segmentEnd;
            }
            const uint32_t expandedRows = static_cast<uint32_t>(
                groupEnd - segment > kBf16RepeatedBatchRows
                    ? kBf16RepeatedBatchRows
                    : groupEnd - segment);
            const int64_t srcSegment = ComputeDirectSourceSegment(segment);
            if (bcastReleasePending) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                bcastReleasePending = false;
            }
            CopyBatchedBcastRow(bcastLocal, srcSegment * directLowSize_, rowElems,
                                rowStride);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

            int64_t batchSegment = segment;
            uint32_t rowCount = static_cast<uint32_t>(groupEnd - batchSegment);
            if (rowCount > kBf16RepeatedBatchRows) {
                rowCount = kBf16RepeatedBatchRows;
            }
            PrepareBatchedFull(fullLocal[0], 0, batchSegment, rowCount, rowElems,
                               rowStride, fullReleasePending);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Cast(bcastFloat, bcastLocal, RoundMode::CAST_NONE, rowStride);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
            bcastReleasePending = true;
            ExpandBatchedBcastRows(bcastFloat, rowStride, expandedRows);

            uint32_t batchIndex = 0;
            while (batchSegment < groupEnd) {
                const uint32_t slot = batchIndex & 1U;
                const int32_t fullEvent = FullEvent(slot);
                const uint32_t calcLen = rowCount * rowStride;
                WaitFlag<HardEvent::MTE2_V>(fullEvent);
                Cast(fullFloat, fullLocal[slot], RoundMode::CAST_NONE, calcLen);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(fullEvent);
                fullReleasePending[slot] = true;

                const int64_t nextSegment = batchSegment + rowCount;
                if (nextSegment < groupEnd) {
                    uint32_t nextRows = static_cast<uint32_t>(groupEnd - nextSegment);
                    if (nextRows > kBf16RepeatedBatchRows) {
                        nextRows = kBf16RepeatedBatchRows;
                    }
                    const uint32_t nextSlot = (batchIndex + 1U) & 1U;
                    PrepareBatchedFull(fullLocal[nextSlot], nextSlot, nextSegment,
                                       nextRows, rowElems, rowStride,
                                       fullReleasePending);
                }

                if (directBroadcastInputIndex_ == 1) {
                    Compare(maskLocal, bcastFloat, fullFloat, CMPMODE::GT, calcLen);
                } else {
                    Compare(maskLocal, fullFloat, bcastFloat, CMPMODE::GT, calcLen);
                }
                PipeBarrier<PIPE_V>();
                Duplicate(valueLocal, static_cast<half>(1.0f), calcLen);
                PipeBarrier<PIPE_V>();
                Select(valueLocal, maskLocal, valueLocal, static_cast<half>(0.0f),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, calcLen);
                PipeBarrier<PIPE_V>();

                if (outputInFlight[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(fullEvent);
                    outputInFlight[slot] = false;
                }
                Cast(outputLocal[slot], valueLocal, RoundMode::CAST_TRUNC, calcLen);
                SetFlag<HardEvent::V_MTE3>(fullEvent);
                WaitFlag<HardEvent::V_MTE3>(fullEvent);
                CopyBatchedOutputRows(batchSegment, outputLocal[slot], rowCount,
                                      rowElems, rowStride);
                SetFlag<HardEvent::MTE3_V>(fullEvent);
                outputInFlight[slot] = true;

                batchSegment = nextSegment;
                if (batchSegment < groupEnd) {
                    rowCount = static_cast<uint32_t>(groupEnd - batchSegment);
                    if (rowCount > kBf16RepeatedBatchRows) {
                        rowCount = kBf16RepeatedBatchRows;
                    }
                }
                ++batchIndex;
            }
            segment = groupEnd;
        }

        if (bcastReleasePending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (fullReleasePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(FullEvent(slot));
            }
            if (outputInFlight[slot]) {
                WaitFlag<HardEvent::MTE3_V>(FullEvent(slot));
            }
        }
    }

    __aicore__ inline int32_t FullEvent(uint32_t slot) const
    {
        return slot == 0 ? EVENT_ID0 : EVENT_ID1;
    }

    __aicore__ inline void CopyIn(LocalTensor<bfloat16_t>& local, GlobalTensor<bfloat16_t>& gm,
                                  int64_t offset, uint32_t len)
    {
        const uint32_t copyBytes = len * sizeof(bfloat16_t);
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(bfloat16_t);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams =
            MakePadParams<bfloat16_t>(0, rightPadding, static_cast<bfloat16_t>(0));
        DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(int64_t offset, LocalTensor<uint8_t>& local, uint32_t len)
    {
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(len * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[offset], local, copyParams);
    }

    __aicore__ inline void PrepareFull(LocalTensor<bfloat16_t>& local, uint32_t slot, int64_t offset,
                                       uint32_t len, bool (&releasePending)[2])
    {
        const int32_t event = FullEvent(slot);
        if (releasePending[slot]) {
            WaitFlag<HardEvent::V_MTE2>(event);
            releasePending[slot] = false;
        }
        CopyIn(local, fullGm_, offset, len);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline int64_t ComputeDirectSourceSegment(int64_t outSegment) const
    {
        int64_t srcSegment = 0;
        int64_t remaining = outSegment;
        for (int64_t i = 0; i < directOuterRank_; ++i) {
            const int64_t stride = directOutStrides_[i];
            int64_t coord = 0;
            if (stride > 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            srcSegment += coord * directSrcStrides_[i];
        }
        return srcSegment;
    }

    __aicore__ inline int64_t DirectLowBoundary(int64_t segment, int64_t boundary, int64_t laneCount) const
    {
        if (boundary <= 0) {
            return 0;
        }
        if (boundary >= laneCount) {
            return directLowSize_;
        }
        constexpr int64_t splitBytes = 512;
        const int64_t segmentBaseBytes = segment * directLowSize_ * sizeof(bfloat16_t);
        const int64_t rawBoundaryBytes = directLowSize_ * sizeof(bfloat16_t) * boundary / laneCount;
        const int64_t absoluteBytes = segmentBaseBytes + rawBoundaryBytes;
        const int64_t alignedAbsoluteBytes = ((absoluteBytes + splitBytes - 1) / splitBytes) * splitBytes;
        int64_t alignedElems = (alignedAbsoluteBytes - segmentBaseBytes) / sizeof(bfloat16_t);
        if (alignedElems > directLowSize_) {
            alignedElems = directLowSize_;
        }
        return alignedElems;
    }

    __aicore__ inline bool ResolveDirectCoreRange(int64_t& segmentStart, int64_t& segmentEnd,
                                                   int64_t& lowStart, int64_t& lowEnd) const
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        lowStart = 0;
        lowEnd = directLowSize_;
        if (blockNum <= directTotalSegments_) {
            segmentStart = directTotalSegments_ * blockIdx / blockNum;
            segmentEnd = directTotalSegments_ * (blockIdx + 1) / blockNum;
            return segmentStart < segmentEnd;
        }

        const int64_t baseLanesPerSegment = blockNum / directTotalSegments_;
        const int64_t wideSegmentCount = blockNum % directTotalSegments_;
        const int64_t wideCoreCount = (baseLanesPerSegment + 1) * wideSegmentCount;
        int64_t segment = 0;
        int64_t lane = 0;
        int64_t laneCount = baseLanesPerSegment;
        if (blockIdx < wideCoreCount) {
            laneCount = baseLanesPerSegment + 1;
            segment = blockIdx / laneCount;
            lane = blockIdx % laneCount;
        } else {
            const int64_t narrowIndex = blockIdx - wideCoreCount;
            segment = wideSegmentCount + narrowIndex / laneCount;
            lane = narrowIndex % laneCount;
        }

        lowStart = DirectLowBoundary(segment, lane, laneCount);
        lowEnd = DirectLowBoundary(segment, lane + 1, laneCount);
        segmentStart = segment;
        segmentEnd = segment + 1;
        return segmentStart < directTotalSegments_ && lowStart < lowEnd;
    }

private:
    GlobalTensor<bfloat16_t> fullGm_;
    GlobalTensor<bfloat16_t> bcastGm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t directLowSize_ = 0;
    int64_t directTotalSegments_ = 0;
    int64_t directOuterRank_ = 0;
    int64_t directBroadcastInputIndex_ = 0;
    int64_t directOuterDims_[kCanonRank] = {1, 1, 1, 1};
    int64_t directOutStrides_[kCanonRank] = {1, 1, 1, 1};
    int64_t directSrcStrides_[kCanonRank] = {0, 0, 0, 0};
};

class KernelGreaterInt32CompactStatic {
    using Layout = Int32CompactStaticLayout;

public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling, TPipe* pipe)
    {
        (void)pipe;
        totalElems_ = tiling.total_elems;
        directLowSize_ = tiling.direct_low_size;
        directTotalSegments_ = tiling.direct_total_segments;
        directOuterRank_ = tiling.direct_outer_rank;
        directBroadcastInputIndex_ = tiling.direct_broadcast_input_index;
        directOuterDims_[0] = tiling.direct_outer_dim0;
        directOuterDims_[1] = tiling.direct_outer_dim1;
        directOuterDims_[2] = tiling.direct_outer_dim2;
        directOuterDims_[3] = tiling.direct_outer_dim3;
        directOutStrides_[0] = tiling.direct_out_stride0;
        directOutStrides_[1] = tiling.direct_out_stride1;
        directOutStrides_[2] = tiling.direct_out_stride2;
        directOutStrides_[3] = tiling.direct_out_stride3;
        directSrcStrides_[0] = tiling.direct_src_stride0;
        directSrcStrides_[1] = tiling.direct_src_stride1;
        directSrcStrides_[2] = tiling.direct_src_stride2;
        directSrcStrides_[3] = tiling.direct_src_stride3;

        if (directBroadcastInputIndex_ == 1) {
            bcastGm_.SetGlobalBuffer((__gm__ int32_t*)x1);
            fullGm_.SetGlobalBuffer((__gm__ int32_t*)x2);
        } else {
            fullGm_.SetGlobalBuffer((__gm__ int32_t*)x1);
            bcastGm_.SetGlobalBuffer((__gm__ int32_t*)x2);
        }
        fullGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    }

    __aicore__ inline void Process()
    {
        if (directLowSize_ != kInt32CompactRowElems || directOuterRank_ <= 0 ||
            directOuterRank_ > kCanonRank || directTotalSegments_ <= 0 ||
            directTotalSegments_ % kInt32CompactBatchRows != 0) {
            return;
        }

        const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        if (repeatSegments < kInt32CompactBatchRows ||
            repeatSegments % kInt32CompactBatchRows != 0 || blockNum <= 0 ||
            directTotalSegments_ % (blockNum * kInt32CompactBatchRows) != 0) {
            return;
        }

        const int64_t segmentsPerCore = directTotalSegments_ / blockNum;
        int64_t segment = segmentsPerCore * blockIdx;
        const int64_t segmentEnd = segment + segmentsPerCore;

        InitSocState();
        LocalTensor<int32_t> largeLocal(TPosition::VECCALC, Layout::kLargeAddr,
                                        Layout::kLargeLength);
        LocalTensor<int32_t> smallLocal(TPosition::VECCALC, Layout::kSmallAddr,
                                        Layout::kChunkLength);
        LocalTensor<int32_t> bcastLocal0(TPosition::VECCALC, Layout::kBcast0Addr,
                                         Layout::kChunkLength);
        LocalTensor<int32_t> bcastLocal1(TPosition::VECCALC, Layout::kBcast1Addr,
                                         Layout::kChunkLength);
        LocalTensor<uint8_t> maskLocal0(TPosition::VECCALC, Layout::kMaskAddr,
                                        Layout::kMaskBytes);
        LocalTensor<uint8_t> maskLocal1(
            TPosition::VECCALC, Layout::kMaskAddr + Layout::kMaskBytes,
            Layout::kMaskBytes);
        LocalTensor<half> zeroLocal(TPosition::VECCALC, Layout::kZeroAddr,
                                    Layout::kChunkLength);
        LocalTensor<int32_t> bcastScratch(
            TPosition::VECCALC, Layout::kBcastScratchAddr,
            Layout::kBcastScratchLength);
        Duplicate(zeroLocal, static_cast<half>(0.0f), Layout::kChunkLength);
        PipeBarrier<PIPE_V>();

        bool largeOutputInFlight = false;
        bool smallOutputInFlight = false;
        bool bcastScratchVectorInFlight = false;
        bool bcastLoadPrefetched = false;
        while (segment < segmentEnd) {
            int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
            if (groupEnd > segmentEnd) {
                groupEnd = segmentEnd;
            }

            if (!bcastLoadPrefetched) {
                const int64_t srcSegment = ComputeDirectSourceSegment(segment);
                StartBroadcastLoad(bcastScratch, srcSegment * directLowSize_,
                                   bcastScratchVectorInFlight);
            }
            FinishBroadcastExpand(bcastLocal0, bcastLocal1, bcastScratch,
                                  groupEnd < segmentEnd,
                                  bcastScratchVectorInFlight);
            bcastLoadPrefetched = false;

            PrepareFullFixed<EVENT_ID0, kInt32CompactLargeRows>(
                largeLocal, segment, largeOutputInFlight);

            while (segment < groupEnd) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                CompareChunk<Layout::kChunkValidLength>(
                    largeLocal, bcastLocal1, maskLocal1);
                CompareChunk<0>(largeLocal, bcastLocal0, maskLocal0);
                PackChunk<Layout::kLargeAddr,
                          Layout::kLargeAddr +
                              Layout::kChunkLength * sizeof(half)>(
                    maskLocal0, zeroLocal);
                PackChunk<Layout::kLargeAddr +
                              Layout::kChunkLength * sizeof(int32_t),
                          Layout::kLargeAddr + Layout::kChunkLength *
                              (sizeof(half) + sizeof(uint8_t))>(
                    maskLocal1, zeroLocal);

                const int64_t smallSegment = segment + kInt32CompactLargeRows;
                PrepareFullFixed<EVENT_ID1, kInt32CompactRows>(
                    smallLocal, smallSegment, smallOutputInFlight);
                CopyPackedOutputFixed<EVENT_ID0, 2>(
                    Layout::kLargeAddr, segment, largeOutputInFlight);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                const int64_t nextSegment = segment + kInt32CompactBatchRows;
                if (nextSegment == groupEnd && nextSegment < segmentEnd) {
                    // small8 输入已经到达后，MTE2 可预取下一组广播；独立 scratch
                    // 允许这笔 4KiB 搬运与当前组最后一段向量计算并行。
                    const int64_t nextSrcSegment =
                        ComputeDirectSourceSegment(nextSegment);
                    StartBroadcastLoad(
                        bcastScratch, nextSrcSegment * directLowSize_,
                        bcastScratchVectorInFlight);
                    bcastLoadPrefetched = true;
                }
                CompareChunk<0>(smallLocal, bcastLocal0, maskLocal0);
                PackChunk<Layout::kSmallAddr,
                          Layout::kSmallAddr +
                              Layout::kChunkLength * sizeof(half)>(
                    maskLocal0, zeroLocal);

                if (nextSegment < groupEnd) {
                    PrepareFullFixed<EVENT_ID0, kInt32CompactLargeRows>(
                        largeLocal, nextSegment, largeOutputInFlight);
                }
                CopyPackedOutputFixed<EVENT_ID1, 1>(
                    Layout::kSmallAddr, smallSegment, smallOutputInFlight);
                segment = nextSegment;
            }
        }

        if (largeOutputInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (smallOutputInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

private:
    __aicore__ inline void StartBroadcastLoad(
        LocalTensor<int32_t>& scratchLocal, int64_t offset,
        bool& scratchVectorInFlight)
    {
        if (scratchVectorInFlight) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            scratchVectorInFlight = false;
        }
        constexpr uint32_t rowBytes = kInt32CompactRowElems * sizeof(int32_t);
        const DataCopyExtParams copyParams = {1, rowBytes, 0, 0, 0};
        const DataCopyPadExtParams<int32_t> padParams = {
            true, 0, 1, static_cast<int32_t>(0)};
        DataCopyPad(scratchLocal, bcastGm_[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void FinishBroadcastExpand(
        LocalTensor<int32_t>& bcastLocal0, LocalTensor<int32_t>& bcastLocal1,
        LocalTensor<int32_t>& scratchLocal, bool scratchWillBeReused,
        bool& scratchVectorInFlight)
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);

        const GatherMaskParams gatherParams = {
            1, static_cast<uint16_t>(kInt32CompactRows), 0, 0};
        uint64_t selectedCount0 = 0;
        GatherMask(bcastLocal0, scratchLocal, static_cast<uint8_t>(7), true,
                   static_cast<uint32_t>(kInt32CompactRowElems), gatherParams,
                   selectedCount0);
        PipeBarrier<PIPE_V>();
        uint64_t selectedCount1 = 0;
        GatherMask(bcastLocal1, scratchLocal, static_cast<uint8_t>(7), true,
                   static_cast<uint32_t>(kInt32CompactRowElems), gatherParams,
                   selectedCount1);
        PipeBarrier<PIPE_V>();
        if (scratchWillBeReused) {
            // 下一组会在当前组末尾覆盖 scratch；用 V_MTE2 明确保护
            // GatherMask 的最后一次读取，最后一组则不制造未消费事件。
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
            scratchVectorInFlight = true;
        }
    }

    template <int32_t Event, uint32_t RowCount>
    __aicore__ inline void PrepareFullFixed(LocalTensor<int32_t>& fullLocal,
                                             int64_t segment,
                                             bool& outputInFlight)
    {
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(Event);
            outputInFlight = false;
        }
        constexpr uint32_t calcLen = RowCount * kInt32CompactRowElems;
        constexpr uint32_t copyBytes = calcLen * sizeof(int32_t);
        constexpr uint32_t blockElems = kAlignBytes / sizeof(int32_t);
        constexpr uint32_t rightPadding =
            (blockElems - calcLen % blockElems) % blockElems;
        const DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        const DataCopyPadExtParams<int32_t> padParams = {
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding),
            static_cast<int32_t>(0)};
        DataCopyPad(fullLocal, fullGm_[segment * directLowSize_], copyParams,
                    padParams);
        SetFlag<HardEvent::MTE2_V>(Event);
    }

    template <uint32_t FullOffset>
    __aicore__ inline void CompareChunk(
        LocalTensor<int32_t>& fullLocal, LocalTensor<int32_t>& bcastLocal,
        LocalTensor<uint8_t>& maskLocal)
    {
        LocalTensor<int32_t> fullChunk = fullLocal[FullOffset];
        if (directBroadcastInputIndex_ == 1) {
            Min(fullChunk, bcastLocal, fullChunk, Layout::kChunkLength);
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, bcastLocal, fullChunk, CMPMODE::EQ,
                    Layout::kChunkLength);
        } else {
            Max(fullChunk, fullChunk, bcastLocal, Layout::kChunkLength);
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, bcastLocal, fullChunk, CMPMODE::EQ,
                    Layout::kChunkLength);
        }
        PipeBarrier<PIPE_V>();
    }

    template <uint32_t ValueAddr, uint32_t OutputAddr>
    __aicore__ inline void PackChunk(LocalTensor<uint8_t>& maskLocal,
                                      LocalTensor<half>& zeroLocal)
    {
        LocalTensor<half> valueLocal(TPosition::VECCALC, ValueAddr,
                                     Layout::kChunkLength);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, OutputAddr,
                                          Layout::kChunkLength);
        Select(valueLocal, maskLocal, zeroLocal, static_cast<half>(1.0f),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, Layout::kChunkLength);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC,
             Layout::kChunkLength);
    }

    template <int32_t Event, uint16_t ChunkCount>
    __aicore__ inline void CopyPackedOutputFixed(uint32_t outputAddr,
                                                  int64_t segment,
                                                  bool& outputInFlight)
    {
        SetFlag<HardEvent::V_MTE3>(Event);
        WaitFlag<HardEvent::V_MTE3>(Event);
        LocalTensor<uint8_t> outputLocal(
            TPosition::VECCALC,
            outputAddr + Layout::kChunkLength * sizeof(half),
            Layout::kChunkLength * ChunkCount);
        const DataCopyExtParams copyParams = {
            ChunkCount, Layout::kChunkValidLength, 0, 0, 0};
        DataCopyPad(yGm_[segment * directLowSize_], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(Event);
        outputInFlight = true;
    }

    __aicore__ inline int64_t ComputeDirectSourceSegment(int64_t outSegment) const
    {
        int64_t srcSegment = 0;
        int64_t remaining = outSegment;
        for (int64_t i = 0; i < directOuterRank_; ++i) {
            const int64_t stride = directOutStrides_[i];
            int64_t coord = 0;
            if (stride > 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            srcSegment += coord * directSrcStrides_[i];
        }
        return srcSegment;
    }

private:
    GlobalTensor<int32_t> fullGm_;
    GlobalTensor<int32_t> bcastGm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t directLowSize_ = 0;
    int64_t directTotalSegments_ = 0;
    int64_t directOuterRank_ = 0;
    int64_t directBroadcastInputIndex_ = 0;
    int64_t directOuterDims_[kCanonRank] = {1, 1, 1, 1};
    int64_t directOutStrides_[kCanonRank] = {1, 1, 1, 1};
    int64_t directSrcStrides_[kCanonRank] = {0, 0, 0, 0};
};

template <typename T>
class KernelGreaterNoBroadcastStatic {
    using Layout = NoBroadcastStaticLayout<T>;
    using AuxT = typename std::conditional<
        std::is_same<T, bfloat16_t>::value, float,
        typename std::conditional<std::is_same<T, int8_t>::value, half, int32_t>::type>::type;

public:
    static_assert(Layout::kBufferEnd <= (192U << 10),
                  "NoBroadcast static layout exceeds the A2 UB capacity");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GreaterTilingData& tiling)
    {
        totalElems_ = tiling.total_elems;
        elemsPerCore_ = tiling.elems_per_core;
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
// >>> GREATER_C4_KEY2_L2_BEGIN
        // 无广播静态路径（key2，全部非 float dtype 都走这里）的输入同样是
        // 每 tile 读一次、跨 tile 不重用的一次性冷流，与 key0 的 Case5 同性质。
        // 原来这段被 float-only 的 if constexpr 挡住，int8/int32/bf16/half
        // 的实例化里整段不存在；门槛也用的是 FP32 大流的 64MiB。
        // 这里改为与通用路径一致的口径：只看冷流规模 kL2BypassMinBytes，
        // 且只对读方向关闭 L2。
        if (static_cast<uint64_t>(totalElems_) * sizeof(T) >= kL2BypassMinBytes) {
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
// <<< GREATER_C4_KEY2_L2_END
    }

    __aicore__ inline void Process()
    {
        int64_t coreStart = 0;
        int64_t coreEnd = 0;
        ResolveCoreRange(coreStart, coreEnd);
        if (coreStart >= coreEnd) {
            return;
        }

        InitSocState();
        LocalTensor<T> x1Local(TPosition::VECCALC, Layout::kX1Addr, Layout::kTileLength);
        LocalTensor<T> x2Local(TPosition::VECCALC, Layout::kX2Addr, Layout::kTileLength);
        LocalTensor<AuxT> aux1Local(TPosition::VECCALC, Layout::kAux1Addr, Layout::kTileLength);
        LocalTensor<AuxT> aux2Local(TPosition::VECCALC, Layout::kAux2Addr, Layout::kTileLength);
        LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, Layout::kMaskAddr, Layout::kMaskBytes);
        LocalTensor<half> valueLocal(TPosition::VECCALC, Layout::kValueAddr, Layout::kTileLength);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, Layout::kOutputAddr, Layout::kTileLength);

        int64_t tileStart = coreStart;
        uint32_t currentLen = TileLength(coreEnd - tileStart);
        uint32_t currentAlignedLen = AlignForHalfCompare<T>(currentLen);
        if constexpr (std::is_same<T, int8_t>::value) {
            CopyOne(x1Local, x1Gm_, tileStart, currentLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            CopyOne(x2Local, x2Gm_, tileStart, currentLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        } else {
            CopyInputs(x1Local, x2Local, tileStart, currentLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }

        bool outputInFlight = false;
        while (tileStart < coreEnd) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            const int64_t nextStart = tileStart + currentLen;
            const bool hasNext = nextStart < coreEnd;
            const uint32_t nextLen = hasNext ? TileLength(coreEnd - nextStart) : 0U;
            const uint32_t nextAlignedLen = hasNext ? AlignForHalfCompare<T>(nextLen) : 0U;

            if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
                Compare(maskLocal, x1Local, x2Local, CMPMODE::GT, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(x1Local, x2Local, nextStart, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(aux1Local, x1Local, RoundMode::CAST_NONE, currentAlignedLen);
                Cast(aux2Local, x2Local, RoundMode::CAST_NONE, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(x1Local, x2Local, nextStart, nextLen, hasNext);
                Compare(maskLocal, aux1Local, aux2Local, CMPMODE::GT, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else if constexpr (std::is_same<T, int32_t>::value) {
                Min(aux1Local, x1Local, x2Local, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, x1Local, aux1Local, CMPMODE::EQ, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(x1Local, x2Local, nextStart, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, false, outputInFlight);
            } else {
                Cast(aux1Local, x1Local, RoundMode::CAST_NONE, currentAlignedLen);
                PrefetchOne(x1Local, x1Gm_, nextStart, nextLen, hasNext, EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                Cast(aux2Local, x2Local, RoundMode::CAST_NONE, currentAlignedLen);
                PrefetchOne(x2Local, x2Gm_, nextStart, nextLen, hasNext, EVENT_ID1);
                PipeBarrier<PIPE_V>();
                Compare(maskLocal, aux1Local, aux2Local, CMPMODE::GT, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, true, outputInFlight);
            }

            tileStart = nextStart;
            currentLen = nextLen;
            currentAlignedLen = nextAlignedLen;
        }
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline uint32_t TileLength(int64_t remaining) const
    {
        return static_cast<uint32_t>(remaining > Layout::kTileLength ? Layout::kTileLength : remaining);
    }

    __aicore__ inline void ResolveCoreRange(int64_t& coreStart, int64_t& coreEnd) const
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if constexpr (std::is_same<T, int8_t>::value) {
            const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
            const int64_t totalTiles = CeilDiv(totalElems_, static_cast<int64_t>(Layout::kTileLength));
            const int64_t baseTiles = totalTiles / blockNum;
            const int64_t extraTiles = totalTiles % blockNum;
            const int64_t coreTiles = baseTiles + (blockIdx < extraTiles ? 1 : 0);
            const int64_t startTile = blockIdx * baseTiles + (blockIdx < extraTiles ? blockIdx : extraTiles);
            coreStart = startTile * static_cast<int64_t>(Layout::kTileLength);
            coreEnd = coreStart + coreTiles * static_cast<int64_t>(Layout::kTileLength);
        } else {
            coreStart = blockIdx * elemsPerCore_;
            coreEnd = coreStart + elemsPerCore_;
        }
        if (coreStart > totalElems_) {
            coreStart = totalElems_;
        }
        if (coreEnd > totalElems_) {
            coreEnd = totalElems_;
        }
    }

    __aicore__ inline void CopyOne(LocalTensor<T>& local, GlobalTensor<T>& gm, int64_t offset,
                                   uint32_t len)
    {
        const uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(T));
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams = MakePadParams<T>(0, rightPadding, static_cast<T>(0));
        DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyInputs(LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
                                      int64_t offset, uint32_t len)
    {
        CopyOne(x1Local, x1Gm_, offset, len);
        CopyOne(x2Local, x2Gm_, offset, len);
    }

    __aicore__ inline void PrefetchNext(LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
                                        int64_t nextStart, uint32_t nextLen, bool hasNext)
    {
        if (!hasNext) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInputs(x1Local, x2Local, nextStart, nextLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void PrefetchOne(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                       int64_t nextStart, uint32_t nextLen, bool hasNext,
                                       int32_t event)
    {
        if (!hasNext) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::V_MTE2>(event);
        CopyOne(local, gm, nextStart, nextLen);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<half>& valueLocal,
                                            LocalTensor<uint8_t>& outputLocal,
                                            int64_t outputOffset, uint32_t len,
                                            uint32_t alignedLen, bool trueWhenMask,
                                            bool& outputInFlight)
    {
        const half baseValue = trueWhenMask ? static_cast<half>(1.0f) : static_cast<half>(0.0f);
        const half selectedValue = trueWhenMask ? static_cast<half>(0.0f) : static_cast<half>(1.0f);
        Duplicate(valueLocal, baseValue, alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, selectedValue,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(len * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outputInFlight = true;
    }

private:
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t elemsPerCore_ = 0;
};

template <typename T>
class KernelGreaterDirectBroadcastStatic {
    using Layout = DirectBroadcastStaticLayout<T>;
    using AuxT = typename std::conditional<
        std::is_same<T, bfloat16_t>::value, float,
        typename std::conditional<std::is_same<T, int8_t>::value, half, int32_t>::type>::type;

public:
    static_assert(Layout::kBufferEnd <= (192U << 10),
                  "DirectBroadcast static layout exceeds the A2 UB capacity");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GreaterTilingData& tiling)
    {
        totalElems_ = tiling.total_elems;
        directLowSize_ = tiling.direct_low_size;
        directTotalSegments_ = tiling.direct_total_segments;
        directOuterRank_ = tiling.direct_outer_rank;
        directBroadcastInputIndex_ = tiling.direct_broadcast_input_index;
        directOuterDims_[0] = tiling.direct_outer_dim0;
        directOuterDims_[1] = tiling.direct_outer_dim1;
        directOuterDims_[2] = tiling.direct_outer_dim2;
        directOuterDims_[3] = tiling.direct_outer_dim3;
        directOutStrides_[0] = tiling.direct_out_stride0;
        directOutStrides_[1] = tiling.direct_out_stride1;
        directOutStrides_[2] = tiling.direct_out_stride2;
        directOutStrides_[3] = tiling.direct_out_stride3;
        directSrcStrides_[0] = tiling.direct_src_stride0;
        directSrcStrides_[1] = tiling.direct_src_stride1;
        directSrcStrides_[2] = tiling.direct_src_stride2;
        directSrcStrides_[3] = tiling.direct_src_stride3;

        if (directBroadcastInputIndex_ == 1) {
            fullGm_.SetGlobalBuffer((__gm__ T*)x2);
            bcastGm_.SetGlobalBuffer((__gm__ T*)x1);
        } else {
            fullGm_.SetGlobalBuffer((__gm__ T*)x1);
            bcastGm_.SetGlobalBuffer((__gm__ T*)x2);
        }
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        if (static_cast<uint64_t>(totalElems_) * sizeof(T) >= kL2BypassMinBytes) {
            fullGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }

    __aicore__ inline void Process()
    {
        if (!InitCursor()) {
            return;
        }

        int64_t fullOffset = 0;
        int64_t bcastOffset = 0;
        int64_t outputOffset = 0;
        uint32_t currentLen = 0;
        if (!NextDescriptor(fullOffset, bcastOffset, outputOffset, currentLen)) {
            return;
        }

        InitSocState();
        LocalTensor<T> fullLocal(TPosition::VECCALC, Layout::kX1Addr, Layout::kTileLength);
        LocalTensor<T> bcastLocal(TPosition::VECCALC, Layout::kX2Addr, Layout::kTileLength);
        LocalTensor<AuxT> aux1Local(TPosition::VECCALC, Layout::kAux1Addr, Layout::kTileLength);
        LocalTensor<AuxT> aux2Local(TPosition::VECCALC, Layout::kAux2Addr, Layout::kTileLength);
        LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, Layout::kMaskAddr, Layout::kMaskBytes);
        LocalTensor<half> valueLocal(TPosition::VECCALC, Layout::kValueAddr, Layout::kTileLength);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, Layout::kOutputAddr, Layout::kTileLength);

        uint32_t currentAlignedLen = AlignForHalfCompare<T>(currentLen);
        CopyInputs(fullLocal, bcastLocal, fullOffset, bcastOffset, currentLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        bool outputInFlight = false;
        while (currentLen > 0U) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            int64_t nextFullOffset = 0;
            int64_t nextBcastOffset = 0;
            int64_t nextOutputOffset = 0;
            uint32_t nextLen = 0;
            const bool hasNext =
                NextDescriptor(nextFullOffset, nextBcastOffset, nextOutputOffset, nextLen);
            const uint32_t nextAlignedLen = hasNext ? AlignForHalfCompare<T>(nextLen) : 0U;

            if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
                CompareLogical(maskLocal, fullLocal, bcastLocal, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, bcastLocal, nextFullOffset, nextBcastOffset, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, outputOffset, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(aux1Local, fullLocal, RoundMode::CAST_NONE, currentAlignedLen);
                Cast(aux2Local, bcastLocal, RoundMode::CAST_NONE, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, bcastLocal, nextFullOffset, nextBcastOffset, nextLen, hasNext);
                CompareLogical(maskLocal, aux1Local, aux2Local, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, outputOffset, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else if constexpr (std::is_same<T, int32_t>::value) {
                Min(aux1Local, fullLocal, bcastLocal, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                if (directBroadcastInputIndex_ == 1) {
                    Compare(maskLocal, bcastLocal, aux1Local, CMPMODE::EQ, currentAlignedLen);
                } else {
                    Compare(maskLocal, fullLocal, aux1Local, CMPMODE::EQ, currentAlignedLen);
                }
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, bcastLocal, nextFullOffset, nextBcastOffset, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, outputOffset, currentLen,
                                currentAlignedLen, false, outputInFlight);
            } else {
                Cast(aux1Local, fullLocal, RoundMode::CAST_NONE, currentAlignedLen);
                Cast(aux2Local, bcastLocal, RoundMode::CAST_NONE, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, bcastLocal, nextFullOffset, nextBcastOffset, nextLen, hasNext);
                CompareLogical(maskLocal, aux1Local, aux2Local, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, outputOffset, currentLen,
                                currentAlignedLen, true, outputInFlight);
            }

            fullOffset = nextFullOffset;
            bcastOffset = nextBcastOffset;
            outputOffset = nextOutputOffset;
            currentLen = nextLen;
            currentAlignedLen = nextAlignedLen;
        }
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline bool InitCursor()
    {
        if (directLowSize_ <= 0 || directTotalSegments_ <= 0) {
            return false;
        }
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        if constexpr (std::is_same<T, int32_t>::value) {
            const int64_t totalBytes = totalElems_ * static_cast<int64_t>(sizeof(T));
            if (blockNum <= directTotalSegments_ && directTotalSegments_ <= kDirectBalancedMaxSegments &&
                totalBytes >= blockNum * 512) {
                flatMode_ = true;
                flatCursor_ = DirectFlatBoundary(blockIdx, blockNum);
                flatEnd_ = DirectFlatBoundary(blockIdx + 1, blockNum);
                return flatCursor_ < flatEnd_;
            }
        }

        lowStart_ = 0;
        lowEnd_ = directLowSize_;
        if (blockNum <= directTotalSegments_) {
            segmentCursor_ = directTotalSegments_ * blockIdx / blockNum;
            segmentEnd_ = directTotalSegments_ * (blockIdx + 1) / blockNum;
        } else {
            const int64_t baseLanesPerSegment = blockNum / directTotalSegments_;
            const int64_t wideSegmentCount = blockNum % directTotalSegments_;
            const int64_t wideCoreCount = (baseLanesPerSegment + 1) * wideSegmentCount;
            int64_t lane = 0;
            int64_t laneCount = baseLanesPerSegment;
            if (blockIdx < wideCoreCount) {
                laneCount = baseLanesPerSegment + 1;
                segmentCursor_ = blockIdx / laneCount;
                lane = blockIdx % laneCount;
            } else {
                const int64_t narrowIndex = blockIdx - wideCoreCount;
                segmentCursor_ = wideSegmentCount + narrowIndex / laneCount;
                lane = narrowIndex % laneCount;
            }
            lowStart_ = DirectLowBoundary(segmentCursor_, lane, laneCount);
            lowEnd_ = DirectLowBoundary(segmentCursor_, lane + 1, laneCount);
            segmentEnd_ = segmentCursor_ + 1;
        }
        chunkCursor_ = lowStart_;
        return segmentCursor_ < directTotalSegments_ && segmentCursor_ < segmentEnd_ && lowStart_ < lowEnd_;
    }

    __aicore__ inline bool NextDescriptor(int64_t& fullOffset, int64_t& bcastOffset,
                                          int64_t& outputOffset, uint32_t& len)
    {
        if (flatMode_) {
            if (flatCursor_ >= flatEnd_) {
                return false;
            }
            const int64_t segment = flatCursor_ / directLowSize_;
            const int64_t chunk = flatCursor_ - segment * directLowSize_;
            int64_t chunkEnd = flatEnd_ - segment * directLowSize_;
            if (chunkEnd > directLowSize_) {
                chunkEnd = directLowSize_;
            }
            int64_t currentLen = chunkEnd - chunk;
            if (currentLen > static_cast<int64_t>(Layout::kTileLength)) {
                currentLen = Layout::kTileLength;
            }
            fullOffset = flatCursor_;
            outputOffset = flatCursor_;
            bcastOffset = ComputeDirectSourceSegment(segment) * directLowSize_ + chunk;
            len = static_cast<uint32_t>(currentLen);
            flatCursor_ += currentLen;
            return len > 0U;
        }

        while (segmentCursor_ < segmentEnd_ && chunkCursor_ >= lowEnd_) {
            ++segmentCursor_;
            chunkCursor_ = lowStart_;
        }
        if (segmentCursor_ >= segmentEnd_) {
            return false;
        }
        int64_t currentLen = lowEnd_ - chunkCursor_;
        if (currentLen > static_cast<int64_t>(Layout::kTileLength)) {
            currentLen = Layout::kTileLength;
        }
        fullOffset = segmentCursor_ * directLowSize_ + chunkCursor_;
        outputOffset = fullOffset;
        bcastOffset = ComputeDirectSourceSegment(segmentCursor_) * directLowSize_ + chunkCursor_;
        len = static_cast<uint32_t>(currentLen);
        chunkCursor_ += currentLen;
        return len > 0U;
    }

    __aicore__ inline int64_t ComputeDirectSourceSegment(int64_t outSegment) const
    {
        int64_t srcSegment = 0;
        int64_t remaining = outSegment;
        for (int64_t i = 0; i < directOuterRank_; ++i) {
            const int64_t stride = directOutStrides_[i];
            int64_t coord = 0;
            if (stride > 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            srcSegment += coord * directSrcStrides_[i];
        }
        return srcSegment;
    }

    __aicore__ inline int64_t DirectLowBoundary(int64_t segment, int64_t boundary,
                                                int64_t laneCount) const
    {
        if (boundary <= 0) {
            return 0;
        }
        if (boundary >= laneCount) {
            return directLowSize_;
        }
        const int64_t segmentBaseBytes = segment * directLowSize_ * static_cast<int64_t>(sizeof(T));
        const int64_t rawBoundaryBytes =
            directLowSize_ * static_cast<int64_t>(sizeof(T)) * boundary / laneCount;
        const int64_t absoluteBytes = segmentBaseBytes + rawBoundaryBytes;
        const int64_t alignedAbsoluteBytes = ((absoluteBytes + 511) / 512) * 512;
        int64_t alignedElems =
            (alignedAbsoluteBytes - segmentBaseBytes) / static_cast<int64_t>(sizeof(T));
        if (alignedElems > directLowSize_) {
            alignedElems = directLowSize_;
        }
        return alignedElems;
    }

    __aicore__ inline int64_t DirectFlatBoundary(int64_t boundary, int64_t blockNum) const
    {
        if (boundary <= 0) {
            return 0;
        }
        if (boundary >= blockNum) {
            return totalElems_;
        }
        const int64_t totalBytes = totalElems_ * static_cast<int64_t>(sizeof(T));
        const int64_t rawBytes = totalBytes * boundary / blockNum;
        int64_t alignedBytes = ((rawBytes + 511) / 512) * 512;
        if (alignedBytes > totalBytes) {
            alignedBytes = totalBytes;
        }
        return alignedBytes / static_cast<int64_t>(sizeof(T));
    }

    template <typename U>
    __aicore__ inline void CompareLogical(LocalTensor<uint8_t>& maskLocal,
                                          LocalTensor<U>& fullLocal,
                                          LocalTensor<U>& bcastLocal,
                                          uint32_t alignedLen)
    {
        if (directBroadcastInputIndex_ == 1) {
            Compare(maskLocal, bcastLocal, fullLocal, CMPMODE::GT, alignedLen);
        } else {
            Compare(maskLocal, fullLocal, bcastLocal, CMPMODE::GT, alignedLen);
        }
    }

    __aicore__ inline void CopyOne(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                   int64_t offset, uint32_t len)
    {
        const uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(T));
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams = MakePadParams<T>(0, rightPadding, static_cast<T>(0));
        DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyInputs(LocalTensor<T>& fullLocal,
                                      LocalTensor<T>& bcastLocal,
                                      int64_t fullOffset, int64_t bcastOffset,
                                      uint32_t len)
    {
        CopyOne(fullLocal, fullGm_, fullOffset, len);
        CopyOne(bcastLocal, bcastGm_, bcastOffset, len);
    }

    __aicore__ inline void PrefetchNext(LocalTensor<T>& fullLocal,
                                        LocalTensor<T>& bcastLocal,
                                        int64_t nextFullOffset, int64_t nextBcastOffset,
                                        uint32_t nextLen, bool hasNext)
    {
        if (!hasNext) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInputs(fullLocal, bcastLocal, nextFullOffset, nextBcastOffset, nextLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<half>& valueLocal,
                                            LocalTensor<uint8_t>& outputLocal,
                                            int64_t outputOffset, uint32_t len,
                                            uint32_t alignedLen, bool trueWhenMask,
                                            bool& outputInFlight)
    {
        const half baseValue = trueWhenMask ? static_cast<half>(1.0f) : static_cast<half>(0.0f);
        const half selectedValue = trueWhenMask ? static_cast<half>(0.0f) : static_cast<half>(1.0f);
        Duplicate(valueLocal, baseValue, alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, selectedValue,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(len * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outputInFlight = true;
    }

private:
    GlobalTensor<T> fullGm_;
    GlobalTensor<T> bcastGm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t directLowSize_ = 0;
    int64_t directTotalSegments_ = 0;
    int64_t directOuterRank_ = 0;
    int64_t directBroadcastInputIndex_ = 0;
    int64_t directOuterDims_[kCanonRank] = {1, 1, 1, 1};
    int64_t directOutStrides_[kCanonRank] = {1, 1, 1, 1};
    int64_t directSrcStrides_[kCanonRank] = {0, 0, 0, 0};
    int64_t segmentCursor_ = 0;
    int64_t segmentEnd_ = 0;
    int64_t chunkCursor_ = 0;
    int64_t lowStart_ = 0;
    int64_t lowEnd_ = 0;
    int64_t flatCursor_ = 0;
    int64_t flatEnd_ = 0;
    bool flatMode_ = false;
};

template <typename T>
class KernelGreaterScalarBroadcastStatic {
    using Layout = ScalarBroadcastStaticLayout<T>;
    using CompareT = typename std::conditional<
        std::is_same<T, bfloat16_t>::value, float,
        typename std::conditional<std::is_same<T, int8_t>::value, half, int32_t>::type>::type;

public:
    static_assert(Layout::kBufferEnd <= (192U << 10),
                  "ScalarBroadcast static layout exceeds the A2 UB capacity");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        totalElems_ = tiling.total_elems;
        elemsPerCore_ = tiling.elems_per_core;
        scalarInputIndex_ = tiling.scalar_input_index;
        if (scalarInputIndex_ == 1) {
            scalarGm_.SetGlobalBuffer((__gm__ T*)x1, 1);
            fullGm_.SetGlobalBuffer((__gm__ T*)x2, totalElems_);
        } else {
            scalarGm_.SetGlobalBuffer((__gm__ T*)x2, 1);
            fullGm_.SetGlobalBuffer((__gm__ T*)x1, totalElems_);
        }
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y, totalElems_);
        if (static_cast<uint64_t>(totalElems_) * sizeof(T) >= kL2BypassMinBytes) {
            fullGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
            scalarValue_ = scalarGm_.GetValue(0);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t coreStart = static_cast<int64_t>(GetBlockIdx()) * elemsPerCore_;
        if (coreStart >= totalElems_) {
            return;
        }
        int64_t coreEnd = coreStart + elemsPerCore_;
        if (coreEnd > totalElems_) {
            coreEnd = totalElems_;
        }

        InitSocState();
        LocalTensor<T> fullLocal(TPosition::VECCALC, Layout::kFullAddr, Layout::kTileLength);
        LocalTensor<CompareT> auxLocal(TPosition::VECCALC, Layout::kAuxAddr, Layout::kTileLength);
        LocalTensor<CompareT> scalarVector(TPosition::VECCALC, Layout::kScalarVectorAddr,
                                           Layout::kTileLength);
        LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, Layout::kMaskAddr, Layout::kMaskBytes);
        LocalTensor<half> valueLocal(TPosition::VECCALC, Layout::kValueAddr, Layout::kTileLength);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, Layout::kOutputAddr, Layout::kTileLength);
        LocalTensor<T> scalarSource(TPosition::VECCALC, Layout::kScalarSourceAddr,
                                    kAlignBytes / sizeof(T));
        LocalTensor<uint32_t> scalarIndices(TPosition::VECCALC, Layout::kScalarIndexAddr,
                                            kScalarIndexBytes / sizeof(uint32_t));
        LocalTensor<CompareT> scalarScratch(TPosition::VECCALC, Layout::kScalarScratchAddr,
                                            kScalarScratchBytes / sizeof(CompareT));

        if constexpr (Layout::kNeedsVectorScalar) {
            InitScalarVector(scalarSource, scalarIndices, scalarScratch, scalarVector);
        }

        int64_t tileStart = coreStart;
        uint32_t currentLen = TileLength(coreEnd - tileStart);
        uint32_t currentAlignedLen = AlignForHalfCompare<T>(currentLen);
        CopyFull(fullLocal, tileStart, currentLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        bool outputInFlight = false;
        while (tileStart < coreEnd) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            const int64_t nextStart = tileStart + currentLen;
            const bool hasNext = nextStart < coreEnd;
            const uint32_t nextLen = hasNext ? TileLength(coreEnd - nextStart) : 0U;
            const uint32_t nextAlignedLen = hasNext ? AlignForHalfCompare<T>(nextLen) : 0U;

            if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
                const CMPMODE mode = scalarInputIndex_ == 1 ? CMPMODE::LT : CMPMODE::GT;
                CompareScalar(maskLocal, fullLocal, scalarValue_, mode, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, nextStart, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else if constexpr (std::is_same<T, bfloat16_t>::value ||
                                 std::is_same<T, int8_t>::value) {
                Cast(auxLocal, fullLocal, RoundMode::CAST_NONE, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, nextStart, nextLen, hasNext);
                if (scalarInputIndex_ == 1) {
                    Compare(maskLocal, scalarVector, auxLocal, CMPMODE::GT, currentAlignedLen);
                } else {
                    Compare(maskLocal, auxLocal, scalarVector, CMPMODE::GT, currentAlignedLen);
                }
                PipeBarrier<PIPE_V>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, true, outputInFlight);
            } else {
                Min(auxLocal, fullLocal, scalarVector, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                if (scalarInputIndex_ == 1) {
                    Compare(maskLocal, scalarVector, auxLocal, CMPMODE::EQ, currentAlignedLen);
                } else {
                    Compare(maskLocal, fullLocal, auxLocal, CMPMODE::EQ, currentAlignedLen);
                }
                PipeBarrier<PIPE_V>();
                PrefetchNext(fullLocal, nextStart, nextLen, hasNext);
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, tileStart, currentLen,
                                currentAlignedLen, false, outputInFlight);
            }

            tileStart = nextStart;
            currentLen = nextLen;
            currentAlignedLen = nextAlignedLen;
        }
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline uint32_t TileLength(int64_t remaining) const
    {
        return static_cast<uint32_t>(remaining > Layout::kTileLength ? Layout::kTileLength : remaining);
    }

    __aicore__ inline void InitScalarVector(LocalTensor<T>& scalarSource,
                                            LocalTensor<uint32_t>& scalarIndices,
                                            LocalTensor<CompareT>& scalarScratch,
                                            LocalTensor<CompareT>& scalarVector)
    {
        const uint32_t rightPadding = kAlignBytes / sizeof(T) - 1U;
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams =
            MakePadParams<T>(0, rightPadding, static_cast<T>(0));
        DataCopyPad(scalarSource, scalarGm_, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        LocalTensor<CompareT> scalarConverted = scalarScratch;
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, int8_t>::value) {
            Cast(scalarConverted, scalarSource, RoundMode::CAST_NONE, kAlignBytes / sizeof(T));
            PipeBarrier<PIPE_V>();
        } else {
            scalarConverted = scalarSource.template ReinterpretCast<CompareT>();
        }

        Duplicate(scalarIndices, static_cast<uint32_t>(0), 8);
        PipeBarrier<PIPE_V>();
        LocalTensor<CompareT> gathered = scalarScratch[64U / sizeof(CompareT)];
        LocalTensor<CompareT> oneRepeat = scalarScratch[128U / sizeof(CompareT)];
        Gather(gathered, scalarConverted, scalarIndices, 0, 8);
        PipeBarrier<PIPE_V>();
        Brcb(oneRepeat, gathered, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        constexpr uint32_t repeatElems = kVectorAlignBytes / sizeof(CompareT);
        constexpr uint8_t repeatTimes =
            static_cast<uint8_t>(Layout::kTileLength / repeatElems);
        Copy(scalarVector, oneRepeat, repeatElems, repeatTimes, {1, 1, 8, 0});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyFull(LocalTensor<T>& fullLocal, int64_t offset, uint32_t len)
    {
        const uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(T));
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams =
            MakePadParams<T>(0, rightPadding, static_cast<T>(0));
        DataCopyPad(fullLocal, fullGm_[offset], copyParams, padParams);
    }

    __aicore__ inline void PrefetchNext(LocalTensor<T>& fullLocal, int64_t nextStart,
                                        uint32_t nextLen, bool hasNext)
    {
        if (!hasNext) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyFull(fullLocal, nextStart, nextLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<half>& valueLocal,
                                            LocalTensor<uint8_t>& outputLocal,
                                            int64_t outputOffset, uint32_t len,
                                            uint32_t alignedLen, bool trueWhenMask,
                                            bool& outputInFlight)
    {
        const half baseValue = trueWhenMask ? static_cast<half>(1.0f) : static_cast<half>(0.0f);
        const half selectedValue = trueWhenMask ? static_cast<half>(0.0f) : static_cast<half>(1.0f);
        Duplicate(valueLocal, baseValue, alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, selectedValue,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(len * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outputInFlight = true;
    }

private:
    GlobalTensor<T> fullGm_;
    GlobalTensor<T> scalarGm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t elemsPerCore_ = 0;
    int64_t scalarInputIndex_ = 0;
    T scalarValue_ = static_cast<T>(0);
};

class KernelGreaterBatchBroadcastStatic {
    using Layout = BatchBroadcastStaticLayout;

public:
    static_assert(Layout::kBufferEnd <= (192U << 10),
                  "BatchBroadcast static layout exceeds the A2 UB capacity");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        totalElems_ = tiling.total_elems;
        rank_ = tiling.rank;
        broadcastInputIndex_ = tiling.direct_broadcast_input_index;
        splitAxis_ = tiling.batch_split_axis;
        midFormer_ = tiling.batch_mid_former;
        totalTiles_ = tiling.batch_total_tiles;
        patternReciprocal_ = tiling.batch_pattern_reciprocal;
        dims_[0] = tiling.dim0; dims_[1] = tiling.dim1;
        dims_[2] = tiling.dim2; dims_[3] = tiling.dim3;
        dims_[4] = tiling.dim4; dims_[5] = tiling.dim5;
        dims_[6] = tiling.dim6; dims_[7] = tiling.dim7;
        outStrides_[0] = tiling.out_stride0; outStrides_[1] = tiling.out_stride1;
        outStrides_[2] = tiling.out_stride2; outStrides_[3] = tiling.out_stride3;
        outStrides_[4] = tiling.out_stride4; outStrides_[5] = tiling.out_stride5;
        outStrides_[6] = tiling.out_stride6; outStrides_[7] = tiling.out_stride7;
        x1Strides_[0] = tiling.x1_stride0; x1Strides_[1] = tiling.x1_stride1;
        x1Strides_[2] = tiling.x1_stride2; x1Strides_[3] = tiling.x1_stride3;
        x1Strides_[4] = tiling.x1_stride4; x1Strides_[5] = tiling.x1_stride5;
        x1Strides_[6] = tiling.x1_stride6; x1Strides_[7] = tiling.x1_stride7;
        x2Strides_[0] = tiling.x2_stride0; x2Strides_[1] = tiling.x2_stride1;
        x2Strides_[2] = tiling.x2_stride2; x2Strides_[3] = tiling.x2_stride3;
        x2Strides_[4] = tiling.x2_stride4; x2Strides_[5] = tiling.x2_stride5;
        x2Strides_[6] = tiling.x2_stride6; x2Strides_[7] = tiling.x2_stride7;

        if (broadcastInputIndex_ == 1) {
            bcastGm_.SetGlobalBuffer((__gm__ half*)x1);
            fullGm_.SetGlobalBuffer((__gm__ half*)x2, totalElems_);
        } else {
            fullGm_.SetGlobalBuffer((__gm__ half*)x1, totalElems_);
            bcastGm_.SetGlobalBuffer((__gm__ half*)x2);
        }
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y, totalElems_);
    }

    __aicore__ inline void Process()
    {
        if (splitAxis_ < 0 || splitAxis_ >= rank_ || midFormer_ <= 1 || totalTiles_ <= 0) {
            return;
        }

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t tilesPerCore = CeilDiv(totalTiles_, blockNum);
        tileCursor_ = blockIdx * tilesPerCore;
        tileEnd_ = tileCursor_ + tilesPerCore;
        if (tileEnd_ > totalTiles_) {
            tileEnd_ = totalTiles_;
        }
        if (tileCursor_ >= tileEnd_) {
            return;
        }

        patternLen_ = static_cast<uint32_t>(outStrides_[splitAxis_]);
        rowStride_ = AlignUpBytes(patternLen_ * sizeof(half), kAlignBytes) / sizeof(half);
        splitTiles_ = CeilDiv(dims_[splitAxis_], midFormer_);
        InitSocState();
        LocalTensor<half> fullLocal(TPosition::VECCALC, Layout::kFullAddr, Layout::kTileLength);
        LocalTensor<half> patternLocal(TPosition::VECCALC, Layout::kPatternAddr,
                                       kBatchPatternMaxLength);
        LocalTensor<half> bcastLocal(TPosition::VECCALC, Layout::kBcastAddr, Layout::kTileLength);
        LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, Layout::kMaskAddr, Layout::kMaskBytes);
        LocalTensor<half> valueLocal(TPosition::VECCALC, Layout::kValueAddr, Layout::kTileLength);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, Layout::kOutputAddr,
                                         Layout::kTileLength);

        int64_t outputOffset = 0;
        int64_t bcastOffset = 0;
        uint32_t currentLen = 0;
        if (!NextDescriptor(outputOffset, bcastOffset, currentLen)) {
            return;
        }
        uint32_t currentAlignedLen = PaddedLength(currentLen);
        bool refreshBcast = bcastOffset != cachedBcastOffset_;
        CopyInputs(fullLocal, patternLocal, outputOffset, bcastOffset, currentLen,
                   currentAlignedLen, refreshBcast);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        bool outputInFlight = false;
        while (currentLen > 0U) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (refreshBcast) {
                ExpandPattern(bcastLocal, patternLocal, currentLen / patternLen_);
                PipeBarrier<PIPE_V>();
                cachedBcastOffset_ = bcastOffset;
                cachedBcastLen_ = currentAlignedLen;
            }

            int64_t nextOutputOffset = 0;
            int64_t nextBcastOffset = 0;
            uint32_t nextLen = 0;
            const bool hasNext = NextDescriptor(nextOutputOffset, nextBcastOffset, nextLen);
            const uint32_t nextAlignedLen = hasNext ? PaddedLength(nextLen) : 0U;
            const bool nextRefreshBcast =
                hasNext && (nextBcastOffset != cachedBcastOffset_ ||
                            nextAlignedLen > cachedBcastLen_);

            if (broadcastInputIndex_ == 1) {
                Compare(maskLocal, bcastLocal, fullLocal, CMPMODE::GT, currentAlignedLen);
            } else {
                Compare(maskLocal, fullLocal, bcastLocal, CMPMODE::GT, currentAlignedLen);
            }
            PipeBarrier<PIPE_V>();
            PrefetchNext(fullLocal, patternLocal, nextOutputOffset, nextBcastOffset,
                         nextLen, nextAlignedLen, nextRefreshBcast, hasNext);
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, outputOffset, currentLen,
                            currentAlignedLen, outputInFlight);

            outputOffset = nextOutputOffset;
            bcastOffset = nextBcastOffset;
            currentLen = nextLen;
            currentAlignedLen = nextAlignedLen;
            refreshBcast = nextRefreshBcast;
        }
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline bool NextDescriptor(int64_t& outputOffset, int64_t& bcastOffset,
                                          uint32_t& calcLen)
    {
        if (tileCursor_ >= tileEnd_) {
            return false;
        }
        int64_t outerIndex = tileCursor_ / splitTiles_;
        const int64_t splitTile = tileCursor_ % splitTiles_;
        const int64_t splitStart = splitTile * midFormer_;
        int64_t splitSize = dims_[splitAxis_] - splitStart;
        if (splitSize > midFormer_) {
            splitSize = midFormer_;
        }

        outputOffset = splitStart * outStrides_[splitAxis_];
        bcastOffset = 0;
        for (int64_t axis = splitAxis_ - 1; axis >= 0; --axis) {
            const int64_t coord = outerIndex % dims_[axis];
            outerIndex /= dims_[axis];
            outputOffset += coord * outStrides_[axis];
            bcastOffset += coord * (broadcastInputIndex_ == 1 ? x1Strides_[axis]
                                                              : x2Strides_[axis]);
        }
        calcLen = static_cast<uint32_t>(splitSize) * patternLen_;
        ++tileCursor_;
        return calcLen > 0U && calcLen <= Layout::kTileLength;
    }

    __aicore__ inline uint32_t PaddedLength(uint32_t len) const
    {
        return (len / patternLen_) * rowStride_;
    }

    __aicore__ inline void ExpandPattern(LocalTensor<half>& bcastLocal,
                                          LocalTensor<half>& patternLocal,
                                          uint32_t rowCount)
    {
        if (rowStride_ <= kVectorAlignBytes / sizeof(half)) {
            uint32_t row = 0;
            const uint8_t rowBlocks = static_cast<uint8_t>(rowStride_ * sizeof(half) / kAlignBytes);
            while (row < rowCount) {
                const uint8_t repeats = static_cast<uint8_t>(
                    rowCount - row > 255U ? 255U : rowCount - row);
                Copy(bcastLocal[row * rowStride_], patternLocal, rowStride_, repeats,
                     {1, 1, rowBlocks, 0});
                row += repeats;
            }
            return;
        }
        for (uint32_t row = 0; row < rowCount; ++row) {
            Adds(bcastLocal[row * rowStride_], patternLocal, static_cast<half>(0.0f),
                 rowStride_);
        }
    }

    __aicore__ inline void CopyFull(LocalTensor<half>& fullLocal, int64_t outputOffset,
                                    uint32_t len, uint32_t alignedLen)
    {
        (void)alignedLen;
        const uint32_t rowCount = len / patternLen_;
        const uint32_t copyBytes = patternLen_ * sizeof(half);
        const uint32_t alignedBytes = rowStride_ * sizeof(half);
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowCount), copyBytes, 0, 0, 0};
        DataCopyPadExtParams<half> padParams =
            MakePadParams<half>(0, (alignedBytes - copyBytes) / sizeof(half),
                                static_cast<half>(0.0f));
        DataCopyPad(fullLocal, fullGm_[outputOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyPattern(LocalTensor<half>& patternLocal, int64_t bcastOffset)
    {
        const uint32_t copyBytes = patternLen_ * sizeof(half);
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(half);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<half> padParams =
            MakePadParams<half>(0, rightPadding, static_cast<half>(0.0f));
        DataCopyPad(patternLocal, bcastGm_[bcastOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyInputs(LocalTensor<half>& fullLocal,
                                      LocalTensor<half>& patternLocal,
                                      int64_t outputOffset, int64_t bcastOffset,
                                      uint32_t len, uint32_t alignedLen,
                                      bool refreshBcast)
    {
        CopyFull(fullLocal, outputOffset, len, alignedLen);
        if (refreshBcast) {
            CopyPattern(patternLocal, bcastOffset);
        }
    }

    __aicore__ inline void PrefetchNext(LocalTensor<half>& fullLocal,
                                        LocalTensor<half>& patternLocal,
                                        int64_t nextOutputOffset, int64_t nextBcastOffset,
                                        uint32_t nextLen, uint32_t nextAlignedLen,
                                        bool nextRefreshBcast, bool hasNext)
    {
        if (!hasNext) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInputs(fullLocal, patternLocal, nextOutputOffset, nextBcastOffset,
                   nextLen, nextAlignedLen, nextRefreshBcast);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<half>& valueLocal,
                                            LocalTensor<uint8_t>& outputLocal,
                                            int64_t outputOffset, uint32_t len,
                                            uint32_t alignedLen, bool& outputInFlight)
    {
        Duplicate(valueLocal, static_cast<half>(1.0f), alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, static_cast<half>(0.0f),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        if (outputInFlight) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams = {
            1, len, 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], outputLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outputInFlight = true;
    }

private:
    GlobalTensor<half> fullGm_;
    GlobalTensor<half> bcastGm_;
    GlobalTensor<uint8_t> yGm_;
    int64_t totalElems_ = 0;
    int64_t rank_ = 0;
    int64_t broadcastInputIndex_ = 0;
    int64_t splitAxis_ = 0;
    int64_t midFormer_ = 1;
    int64_t totalTiles_ = 0;
    int64_t splitTiles_ = 0;
    int64_t tileCursor_ = 0;
    int64_t tileEnd_ = 0;
    int64_t cachedBcastOffset_ = -1;
    uint32_t cachedBcastLen_ = 0;
    int64_t dims_[kMaxRank] = {};
    int64_t outStrides_[kMaxRank] = {};
    int64_t x1Strides_[kMaxRank] = {};
    int64_t x2Strides_[kMaxRank] = {};
    uint32_t patternLen_ = 1;
    uint32_t rowStride_ = 1;
    float patternReciprocal_ = 1.0f;
};

template <typename T>
class KernelGreater {
public:
    __aicore__ inline KernelGreater() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const GreaterTilingData& tiling,
                                TPipe* pipe)
    {
        pipe_ = pipe;
        x1InputAddr_ = x1;
        x2InputAddr_ = x2;
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        totalElems_ = tiling.total_elems;
        x1TotalElems_ = tiling.x1_total_elems;
        x2TotalElems_ = tiling.x2_total_elems;
        elemsPerCore_ = tiling.elems_per_core;
        rank_ = tiling.rank;
        mode_ = tiling.mode;
        scalarInputIndex_ = tiling.scalar_input_index;
        directBroadcastInputIndex_ = tiling.direct_broadcast_input_index;
        elemSize_ = static_cast<uint32_t>(tiling.elem_size);
        x1WorkspaceOffset_ = tiling.x1_workspace_offset;
        x2WorkspaceOffset_ = tiling.x2_workspace_offset;
        scratchWorkspaceOffset_ = tiling.scratch_workspace_offset;
        directLowSize_ = tiling.direct_low_size;
        directTotalSegments_ = tiling.direct_total_segments;
        directOuterRank_ = tiling.direct_outer_rank;
        batchSplitAxis_ = tiling.batch_split_axis;
        batchMidFormer_ = tiling.batch_mid_former;
        batchTotalTiles_ = tiling.batch_total_tiles;
        batchPatternReciprocal_ = tiling.batch_pattern_reciprocal;
        directOuterDims_[0] = tiling.direct_outer_dim0;
        directOuterDims_[1] = tiling.direct_outer_dim1;
        directOuterDims_[2] = tiling.direct_outer_dim2;
        directOuterDims_[3] = tiling.direct_outer_dim3;
        directOutStrides_[0] = tiling.direct_out_stride0;
        directOutStrides_[1] = tiling.direct_out_stride1;
        directOutStrides_[2] = tiling.direct_out_stride2;
        directOutStrides_[3] = tiling.direct_out_stride3;
        directSrcStrides_[0] = tiling.direct_src_stride0;
        directSrcStrides_[1] = tiling.direct_src_stride1;
        directSrcStrides_[2] = tiling.direct_src_stride2;
        directSrcStrides_[3] = tiling.direct_src_stride3;
        dims_[0] = tiling.dim0; dims_[1] = tiling.dim1; dims_[2] = tiling.dim2; dims_[3] = tiling.dim3;
        dims_[4] = tiling.dim4; dims_[5] = tiling.dim5; dims_[6] = tiling.dim6; dims_[7] = tiling.dim7;
        outStrides_[0] = tiling.out_stride0; outStrides_[1] = tiling.out_stride1;
        outStrides_[2] = tiling.out_stride2; outStrides_[3] = tiling.out_stride3;
        outStrides_[4] = tiling.out_stride4; outStrides_[5] = tiling.out_stride5;
        outStrides_[6] = tiling.out_stride6; outStrides_[7] = tiling.out_stride7;
        x1Strides_[0] = tiling.x1_stride0; x1Strides_[1] = tiling.x1_stride1;
        x1Strides_[2] = tiling.x1_stride2; x1Strides_[3] = tiling.x1_stride3;
        x1Strides_[4] = tiling.x1_stride4; x1Strides_[5] = tiling.x1_stride5;
        x1Strides_[6] = tiling.x1_stride6; x1Strides_[7] = tiling.x1_stride7;
        x2Strides_[0] = tiling.x2_stride0; x2Strides_[1] = tiling.x2_stride1;
        x2Strides_[2] = tiling.x2_stride2; x2Strides_[3] = tiling.x2_stride3;
        x2Strides_[4] = tiling.x2_stride4; x2Strides_[5] = tiling.x2_stride5;
        x2Strides_[6] = tiling.x2_stride6; x2Strides_[7] = tiling.x2_stride7;
        canonX1_[0] = tiling.canon_x10; canonX1_[1] = tiling.canon_x11; canonX1_[2] = tiling.canon_x12; canonX1_[3] = tiling.canon_x13;
        canonX2_[0] = tiling.canon_x20; canonX2_[1] = tiling.canon_x21; canonX2_[2] = tiling.canon_x22; canonX2_[3] = tiling.canon_x23;
        canonOut_[0] = tiling.canon_out0; canonOut_[1] = tiling.canon_out1; canonOut_[2] = tiling.canon_out2; canonOut_[3] = tiling.canon_out3;

        const bool fp16Batch = mode_ == kModeBatchBroadcast && std::is_same<T, half>::value;
        const bool regularDoubleBuffer = mode_ == kModeDirectBroadcast || mode_ == kModeNoBroadcast ||
                                         mode_ == kModeScalarBroadcast || mode_ == kModeWorkspaceBroadcast;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            bf16ScalarCached_ = mode_ == kModeScalarBroadcast;
        }
        if constexpr (std::is_same<T, float>::value) {
            useFp32NoBroadcastSingleBuffer_ =
                mode_ == kModeNoBroadcast &&
                static_cast<uint64_t>(totalElems_) * sizeof(T) >= kFp32SingleBufferMinBytes;
        }
        uint32_t tileLength = bf16ScalarCached_
                                  ? kBf16ScalarCachedTileLength
                                  : (fp16Batch
                                         ? kBatchGatherTileLength
                                         : (mode_ == kModeNoBroadcast
                                                ? NoBroadcastDoubleBufferTileLengthForType<T>()
                                                : (regularDoubleBuffer
                                                       ? DirectDoubleBufferTileLengthForType<T>()
                                                       : TileLengthForType<T>())));
        if constexpr (std::is_same<T, int32_t>::value) {
            if (mode_ == kModeDirectBroadcast) {
                tileLength = kInt32DirectTQueTileLength;
            }
        }
        if (useFp32NoBroadcastSingleBuffer_) {
            tileLength = kFp32NoBroadcastSingleBufferTileLength;
        }
        const uint32_t bufferNum = (regularDoubleBuffer || fp16Batch) ? 2U : kBufferNum;
        processTileLength_ = tileLength;
        if (mode_ != kModeWorkspaceBroadcast && !useFp32NoBroadcastSingleBuffer_) {
            InitCompareBuffers(pipe, bufferNum, tileLength, fp16Batch);
        }

        if (mode_ == kModeDirectBroadcast) {
            directTileElems_ = tileLength;
            if (directLowSize_ > 0 && static_cast<int64_t>(directTileElems_) > directLowSize_) {
                directTileElems_ = static_cast<uint32_t>(directLowSize_);
            }
        }

        if constexpr (std::is_same<T, half>::value) {
            if (mode_ == kModeBatchBroadcast) {
                batchPatternLen_ = static_cast<int32_t>(outStrides_[batchSplitAxis_]);
                pipe->InitBuffer(batchIndexBuf_, kBatchGatherTileLength * sizeof(uint32_t));
                pipe->InitBuffer(batchScratchBuf_, kBatchGatherTileLength * sizeof(uint32_t));
                pipe->InitBuffer(batchBcastCacheBuf_, kBatchGatherTileLength * sizeof(half));
                pipe->InitBuffer(batchPatternQueue_, 1, kBatchPatternMaxLength * sizeof(half));
                InitBatchGatherIndices();
            }
        }

        if (mode_ == kModeWorkspaceBroadcast) {
            __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspace);
            x1CompareAddr_ = (x1WorkspaceOffset_ >= 0) ? reinterpret_cast<GM_ADDR>(userWorkspace + x1WorkspaceOffset_) : x1;
            x2CompareAddr_ = (x2WorkspaceOffset_ >= 0) ? reinterpret_cast<GM_ADDR>(userWorkspace + x2WorkspaceOffset_) : x2;
            x1CompareGm_.SetGlobalBuffer((__gm__ T*)x1CompareAddr_, totalElems_);
            x2CompareGm_.SetGlobalBuffer((__gm__ T*)x2CompareAddr_, totalElems_);
            scratchAddr_ = reinterpret_cast<GM_ADDR>(userWorkspace + scratchWorkspaceOffset_);
            broadcaster_.Init(pipe, static_cast<uint32_t>((95U << 10) / sizeof(T)));
        } else {
            x1CompareAddr_ = x1;
            x2CompareAddr_ = x2;
            x1CompareGm_.SetGlobalBuffer((__gm__ T*)x1, (scalarInputIndex_ == 1) ? 1 : totalElems_);
            x2CompareGm_.SetGlobalBuffer((__gm__ T*)x2, (scalarInputIndex_ == 2) ? 1 : totalElems_);
        }

        ConfigureL2CacheHints();

        if (mode_ == kModeScalarBroadcast) {
            if constexpr (UseCompareScalar<T>()) {
                scalarValue_ = (scalarInputIndex_ == 1) ? x1Gm_.GetValue(0) : x2Gm_.GetValue(0);
            } else {
                if constexpr (UseDedicatedScalarCache<T>()) {
                    constexpr uint32_t bankGapBytes = ScalarBankGapBytesForType<T>();
                    if constexpr (bankGapBytes > 0) {
                        pipe->InitBuffer(scalarBankGapBuf_, bankGapBytes);
                    }
                    pipe->InitBuffer(scalarVectorBuf_, processTileLength_ * ScalarCacheElemBytesForType<T>());
                }
                pipe->InitBuffer(scalarSourceQueue_, 1, kAlignBytes);
                pipe->InitBuffer(scalarIndexBuf_, kScalarIndexBytes);
                pipe->InitBuffer(scalarScratchBuf_, kScalarScratchBytes);
                InitScalarCache();
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same<T, float>::value) {
            if (useFp32NoBroadcastSingleBuffer_) {
                ProcessFloatNoBroadcastSingleBuffer();
                return;
            }
        }
        if (mode_ == kModeBatchBroadcast) {
            ProcessBatchBroadcast();
            return;
        }
        if (mode_ == kModeDirectBroadcast) {
            ProcessDirectBroadcast();
            return;
        }
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            if (bf16ScalarCached_) {
                ProcessBfloat16ScalarCached();
                return;
            }
        }

        if (mode_ == kModeWorkspaceBroadcast) {
            if (x1WorkspaceOffset_ >= 0) {
                broadcaster_.BroadcastIfNeeded(x1InputAddr_, x1TotalElems_, x1CompareAddr_, scratchAddr_, canonX1_, canonOut_);
                if (x2WorkspaceOffset_ >= 0) {
                    CrossCoreSetFlag<0, PIPE_MTE3>(0);
                    CrossCoreWaitFlag(0);
                }
            }
            if (x2WorkspaceOffset_ >= 0) {
                broadcaster_.BroadcastIfNeeded(x2InputAddr_, x2TotalElems_, x2CompareAddr_, scratchAddr_, canonX2_, canonOut_);
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(0);
            CrossCoreWaitFlag(0);
            pipe_->Reset();
            InitCompareBuffers(pipe_, 2U, processTileLength_, false);
        }

        const uint32_t tileLength = processTileLength_;
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t coreStart = blockIdx * elemsPerCore_;
        int64_t coreEnd = coreStart + elemsPerCore_;
        if (mode_ == kModeNoBroadcast && std::is_same<T, int8_t>::value) {
            const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
            const int64_t totalTiles = CeilDiv(totalElems_, static_cast<int64_t>(tileLength));
            const int64_t baseTiles = totalTiles / blockNum;
            const int64_t extraTiles = totalTiles % blockNum;
            const int64_t coreTiles = baseTiles + ((blockIdx < extraTiles) ? 1 : 0);
            const int64_t startTile = blockIdx * baseTiles + ((blockIdx < extraTiles) ? blockIdx : extraTiles);
            coreStart = startTile * static_cast<int64_t>(tileLength);
            coreEnd = coreStart + coreTiles * static_cast<int64_t>(tileLength);
        }
        if (coreStart >= totalElems_) {
            return;
        }
        if (coreEnd > totalElems_) {
            coreEnd = totalElems_;
        }

        int64_t tileStart = coreStart;
        while (tileStart < coreEnd) {
            calcLength_ = static_cast<uint32_t>(coreEnd - tileStart);
            if (calcLength_ > tileLength) {
                calcLength_ = tileLength;
            }
            const uint32_t alignedLen = AlignForHalfCompare<T>(calcLength_);
            if (mode_ == kModeScalarBroadcast) {
                CopyInScalar(tileStart, alignedLen);
                ComputeScalar(alignedLen);
            } else {
                CopyIn(tileStart, alignedLen);
                Compute(alignedLen);
            }
            CopyOut(tileStart);
            tileStart += calcLength_;
        }
        if (mode_ == kModeScalarBroadcast) {
            ReleaseScalarCache();
        }
    }

private:
    __aicore__ inline void DisableX1L2Cache()
    {
        x1Gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        x1CompareGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void DisableX2L2Cache()
    {
        x2Gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        x2CompareGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void ConfigureL2CacheHints()
    {
        const uint64_t outputBytes = static_cast<uint64_t>(totalElems_) * sizeof(T);
        if (outputBytes < kL2BypassMinBytes) {
            return;
        }

        if (mode_ == kModeNoBroadcast) {
            // 无广播路径每一路输入在一个 tile 内只被读一次、跨 tile 不重用，
            // 且 Case5 的 83.7MB 读工作集已接近 96MiB L2 容量：把它放进 L2 只会
            // 造成分配/替换压力，不会产生命中。只对读方向显式关闭 L2
            // （提示作用于读方向，真正改变策略的是 CACHE_MODE_DISABLE），
            // 写方向保持默认。覆盖全部 dtype（bf16/int8 会先转成 float/half 再比较，
            // 但原始输入张量仍是那条一次性读流）；FP32 大流 >=64MiB 的静态路径
            // 另有自己的 bypass，两者语义一致。
            x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            x1CompareGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            x2CompareGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            return;
        }

        if constexpr (std::is_same<T, float>::value) {
            if (outputBytes >= (64U << 20)) {
                DisableX1L2Cache();
                DisableX2L2Cache();
                return;
            }
        }

        if (mode_ == kModeDirectBroadcast) {
            if (directBroadcastInputIndex_ == 1) {
                DisableX2L2Cache();
            } else {
                DisableX1L2Cache();
            }
        }
    }

    template <typename U>
    __aicore__ inline void BuildScalarVector(LocalTensor<U>& dst, LocalTensor<U>& src)
    {
        LocalTensor<uint32_t> indices = scalarIndexBuf_.Get<uint32_t>();
        LocalTensor<U> scratch = scalarScratchBuf_.Get<U>();
        LocalTensor<U> gathered = scratch[64U / sizeof(U)];
        LocalTensor<U> oneRepeat = scratch[128U / sizeof(U)];
        Gather(gathered, src, indices, 0, 8);
        PipeBarrier<PIPE_V>();
        Brcb(oneRepeat, gathered, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        constexpr uint32_t repeatElems = kVectorAlignBytes / sizeof(U);
        const uint8_t repeatTimes = static_cast<uint8_t>(processTileLength_ / repeatElems);
        Copy(dst, oneRepeat, repeatElems, repeatTimes, {1, 1, 8, 0});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitScalarCache()
    {
        LocalTensor<uint32_t> indices = scalarIndexBuf_.Get<uint32_t>();
        Duplicate(indices, static_cast<uint32_t>(0), 8);
        PipeBarrier<PIPE_V>();

        GlobalTensor<T>& scalarGm = (scalarInputIndex_ == 1) ? x1Gm_ : x2Gm_;
        LocalTensor<T> scalarSource = scalarSourceQueue_.AllocTensor<T>();
        CopyInPad(scalarSource, 0, scalarGm, 0, 1);
        scalarSourceQueue_.EnQue(scalarSource);
        scalarSource = scalarSourceQueue_.DeQue<T>();

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> converted = scalarScratchBuf_.Get<float>();
            Cast(converted, scalarSource, RoundMode::CAST_NONE, kAlignBytes / sizeof(T));
            PipeBarrier<PIPE_V>();
            LocalTensor<float> scalarFloat = scalarVectorBuf_.Get<float>();
            BuildScalarVector(scalarFloat, converted);
        } else if constexpr (std::is_same<T, int8_t>::value) {
            LocalTensor<half> converted = scalarScratchBuf_.Get<half>();
            Cast(converted, scalarSource, RoundMode::CAST_NONE, kAlignBytes / sizeof(T));
            PipeBarrier<PIPE_V>();
            scalarHalfVector_ = scalarVectorBuf_.Get<half>();
            BuildScalarVector(scalarHalfVector_, converted);
        } else {
            if constexpr (UseDedicatedScalarCache<T>()) {
                scalarVector_ = scalarVectorBuf_.Get<T>();
            } else {
                scalarVector_ = (scalarInputIndex_ == 1) ? x1Queue_.AllocTensor<T>() : x2Queue_.AllocTensor<T>();
            }
            BuildScalarVector(scalarVector_, scalarSource);
        }
        scalarSourceQueue_.FreeTensor(scalarSource);
    }

    __aicore__ inline void ReleaseScalarCache()
    {
        if constexpr (UseCompareScalar<T>()) {
            return;
        } else if constexpr (!UseDedicatedScalarCache<T>()) {
            if (scalarInputIndex_ == 1) {
                x1Queue_.FreeTensor(scalarVector_);
            } else {
                x2Queue_.FreeTensor(scalarVector_);
            }
        }
    }

    __aicore__ inline void InitCompareBuffers(TPipe* pipe, uint32_t bufferNum, uint32_t tileLength, bool fp16Batch)
    {
        const bool scalarMode = mode_ == kModeScalarBroadcast;
        const uint32_t tempBufferNum =
            (mode_ == kModeNoBroadcast && std::is_same<T, int8_t>::value) ? 1U : bufferNum;
        if (scalarMode && UseCompareScalar<T>()) {
            if (scalarInputIndex_ == 1) {
                pipe->InitBuffer(x2Queue_, bufferNum, tileLength * sizeof(T));
            } else {
                pipe->InitBuffer(x1Queue_, bufferNum, tileLength * sizeof(T));
            }
        } else if (scalarMode && !bf16ScalarCached_ && UseDedicatedScalarCache<T>()) {
            if (scalarInputIndex_ == 1) {
                pipe->InitBuffer(x2Queue_, bufferNum, tileLength * sizeof(T));
            } else {
                pipe->InitBuffer(x1Queue_, bufferNum, tileLength * sizeof(T));
            }
        } else {
            pipe->InitBuffer(x1Queue_, bufferNum, tileLength * sizeof(T));
            if (!fp16Batch && !bf16ScalarCached_) {
                pipe->InitBuffer(compareBankGapBuf_, kCompareBankGapBytes);
                pipe->InitBuffer(x2Queue_, bufferNum, tileLength * sizeof(T));
            }
        }
        const uint32_t maskBytes = AlignUpBytes((tileLength + 7U) / 8U, kAlignBytes);
        pipe->InitBuffer(maskQueue_, tempBufferNum, maskBytes);
        if constexpr (std::is_same<T, int8_t>::value) {
            pipe->InitBuffer(valueQueue_, tempBufferNum, tileLength * sizeof(half));
        }
        pipe->InitBuffer(outQueue_, bufferNum, tileLength * sizeof(uint8_t));
        if constexpr (NeedHalfCastQueues<T>()) {
            if (scalarMode) {
                if (scalarInputIndex_ == 1) {
                    pipe->InitBuffer(x2CastQueue_, bufferNum, tileLength * sizeof(half));
                } else {
                    pipe->InitBuffer(x1CastQueue_, bufferNum, tileLength * sizeof(half));
                }
            } else {
                pipe->InitBuffer(x1CastQueue_, tempBufferNum, tileLength * sizeof(half));
                pipe->InitBuffer(castBankGapBuf_, kCompareBankGapBytes);
                pipe->InitBuffer(x2CastQueue_, tempBufferNum, tileLength * sizeof(half));
            }
        }
        if constexpr (NeedFloatCastQueues<T>()) {
            pipe->InitBuffer(x1FloatQueue_, bf16ScalarCached_ ? 1U : bufferNum, tileLength * sizeof(float));
            if (!bf16ScalarCached_) {
                pipe->InitBuffer(castBankGapBuf_, kCompareBankGapBytes);
                pipe->InitBuffer(x2FloatQueue_, bufferNum, tileLength * sizeof(float));
            }
        }
        if constexpr (NeedInt32TmpQueue<T>()) {
            if (mode_ != kModeDirectBroadcast) {
                pipe->InitBuffer(int32Queue1_, bufferNum, tileLength * sizeof(int32_t));
            }
        }
    }

    __aicore__ inline void ProcessBfloat16ScalarCached()
    {
        if constexpr (!std::is_same<T, bfloat16_t>::value) {
            return;
        } else {
            const int64_t coreStart = static_cast<int64_t>(GetBlockIdx()) * elemsPerCore_;
            if (coreStart >= totalElems_) {
                return;
            }
            int64_t coreEnd = coreStart + elemsPerCore_;
            if (coreEnd > totalElems_) {
                coreEnd = totalElems_;
            }
            GlobalTensor<T>& fullGm = (scalarInputIndex_ == 1) ? x2Gm_ : x1Gm_;
            LocalTensor<float> scalarFloat = scalarVectorBuf_.Get<float>();
            for (int64_t start = coreStart; start < coreEnd; start += processTileLength_) {
                uint32_t len = static_cast<uint32_t>(coreEnd - start);
                if (len > processTileLength_) {
                    len = processTileLength_;
                }
                const uint32_t alignedLen = AlignForHalfCompare<T>(len);
                LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                CopyInContiguous(fullLocal, fullGm, start, len, alignedLen);
                x1Queue_.EnQue(fullLocal);

                LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
                LocalTensor<float> fullFloat = x1FloatQueue_.AllocTensor<float>();
                fullLocal = x1Queue_.DeQue<T>();
                Cast(fullFloat, fullLocal, RoundMode::CAST_NONE, alignedLen);
                PipeBarrier<PIPE_V>();
                if (scalarInputIndex_ == 1) {
                    Compare(maskLocal, scalarFloat, fullFloat, CMPMODE::GT, alignedLen);
                } else {
                    Compare(maskLocal, fullFloat, scalarFloat, CMPMODE::GT, alignedLen);
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<half> valueLocal = fullLocal.template ReinterpretCast<half>();
                WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, true);
                x1FloatQueue_.FreeTensor(fullFloat);
                x1Queue_.FreeTensor(fullLocal);
                maskQueue_.FreeTensor(maskLocal);
                outQueue_.EnQue(outputLocal);
                outputLocal = outQueue_.DeQue<uint8_t>();
                CopyOutDirect(start, outputLocal, len);
                outQueue_.FreeTensor(outputLocal);
            }
        }
    }

    __aicore__ inline void CopyInPad(LocalTensor<T>& local, uint32_t localOffset, GlobalTensor<T>& gm, int64_t gmOffset,
                                     uint32_t len)
    {
        const uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(T));
        const uint32_t alignedBytes = AlignUpBytes(copyBytes, kAlignBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams =
            MakePadParams<T>(0, rightPadding, static_cast<T>(0));
        DataCopyPad(local[localOffset], gm[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyInContiguous(LocalTensor<T>& local, GlobalTensor<T>& gm, int64_t offset, uint32_t len,
                                            uint32_t paddedLen)
    {
        (void)paddedLen;
        CopyInPad(local, 0, gm, offset, len);
    }

    __aicore__ inline int64_t ComputeDirectSourceSegment(int64_t outSegment) const
    {
        int64_t srcSegment = 0;
        int64_t remaining = outSegment;
        for (int64_t i = 0; i < directOuterRank_; ++i) {
            const int64_t stride = directOutStrides_[i];
            int64_t coord = 0;
            if (stride > 0) {
                coord = remaining / stride;
                remaining %= stride;
            }
            srcSegment += coord * directSrcStrides_[i];
        }
        return srcSegment;
    }

    __aicore__ inline int64_t DirectLowBoundary(int64_t segment, int64_t boundary, int64_t laneCount) const
    {
        if (boundary <= 0) {
            return 0;
        }
        if (boundary >= laneCount) {
            return directLowSize_;
        }

        constexpr int64_t splitBytes = 512;
        const int64_t segmentBaseBytes = segment * directLowSize_ * static_cast<int64_t>(sizeof(T));
        const int64_t rawBoundaryBytes =
            (directLowSize_ * static_cast<int64_t>(sizeof(T)) * boundary) / laneCount;
        const int64_t absoluteBytes = segmentBaseBytes + rawBoundaryBytes;
        const int64_t alignedAbsoluteBytes =
            ((absoluteBytes + splitBytes - 1) / splitBytes) * splitBytes;
        int64_t alignedElems = (alignedAbsoluteBytes - segmentBaseBytes) / static_cast<int64_t>(sizeof(T));
        if (alignedElems > directLowSize_) {
            alignedElems = directLowSize_;
        }
        return alignedElems;
    }

    __aicore__ inline bool ResolveDirectCoreRange(int64_t& segmentStart, int64_t& segmentEnd,
                                                   int64_t& lowStart, int64_t& lowEnd) const
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        lowStart = 0;
        lowEnd = directLowSize_;
        if (blockNum <= directTotalSegments_) {
            segmentStart = directTotalSegments_ * blockIdx / blockNum;
            segmentEnd = directTotalSegments_ * (blockIdx + 1) / blockNum;
            return segmentStart < segmentEnd;
        }

        const int64_t baseLanesPerSegment = blockNum / directTotalSegments_;
        const int64_t wideSegmentCount = blockNum % directTotalSegments_;
        const int64_t wideCoreCount = (baseLanesPerSegment + 1) * wideSegmentCount;
        int64_t segment = 0;
        int64_t lane = 0;
        int64_t laneCount = baseLanesPerSegment;
        if (blockIdx < wideCoreCount) {
            laneCount = baseLanesPerSegment + 1;
            segment = blockIdx / laneCount;
            lane = blockIdx % laneCount;
        } else {
            const int64_t narrowIndex = blockIdx - wideCoreCount;
            segment = wideSegmentCount + narrowIndex / laneCount;
            lane = narrowIndex % laneCount;
        }

        lowStart = DirectLowBoundary(segment, lane, laneCount);
        lowEnd = DirectLowBoundary(segment, lane + 1, laneCount);
        segmentStart = segment;
        segmentEnd = segment + 1;
        return segmentStart < directTotalSegments_ && lowStart < lowEnd;
    }

    __aicore__ inline int64_t DirectFlatBoundary(int64_t boundary, int64_t blockNum) const
    {
        if (boundary <= 0) {
            return 0;
        }
        if (boundary >= blockNum) {
            return totalElems_;
        }

        constexpr int64_t splitBytes = 512;
        const int64_t totalBytes = totalElems_ * static_cast<int64_t>(sizeof(T));
        const int64_t rawBytes = totalBytes * boundary / blockNum;
        int64_t alignedBytes = ((rawBytes + splitBytes - 1) / splitBytes) * splitBytes;
        if (alignedBytes > totalBytes) {
            alignedBytes = totalBytes;
        }
        return alignedBytes / static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline bool ResolveDirectAlignedFlatRange(int64_t& flatStart, int64_t& flatEnd) const
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        flatStart = DirectFlatBoundary(blockIdx, blockNum);
        flatEnd = DirectFlatBoundary(blockIdx + 1, blockNum);
        return flatStart < flatEnd;
    }

    __aicore__ inline void CopyOutDirect(int64_t start, LocalTensor<uint8_t>& outputLocal, uint32_t len)
    {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[start], outputLocal, copyParams);
    }

    __aicore__ inline void ProcessBatchBroadcast()
    {
        if constexpr (!std::is_same<T, half>::value) {
            return;
        } else {
            if (batchSplitAxis_ < 0 || batchSplitAxis_ >= rank_ || batchMidFormer_ <= 1 || batchTotalTiles_ <= 0) {
                return;
            }

            const int64_t blockIdx = GetBlockIdx();
            const int64_t blockNum = GetBlockNum();
            const int64_t tilesPerCore = CeilDiv(batchTotalTiles_, blockNum);
            const int64_t tileStart = blockIdx * tilesPerCore;
            int64_t tileEnd = tileStart + tilesPerCore;
            if (tileEnd > batchTotalTiles_) {
                tileEnd = batchTotalTiles_;
            }
            if (tileStart >= tileEnd) {
                return;
            }

            GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
            GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
            const int64_t splitTiles = CeilDiv(dims_[batchSplitAxis_], batchMidFormer_);
            const uint32_t patternLen = batchPatternLen_;
            LocalTensor<uint32_t> gatherIndex = batchIndexBuf_.Get<uint32_t>();
            LocalTensor<T> bcastLocal = batchBcastCacheBuf_.Get<T>();
            int64_t cachedBcastOffset = -1;

            for (int64_t tile = tileStart; tile < tileEnd; ++tile) {
                int64_t outerIndex = tile / splitTiles;
                const int64_t splitTile = tile % splitTiles;
                const int64_t splitStart = splitTile * batchMidFormer_;
                int64_t splitSize = dims_[batchSplitAxis_] - splitStart;
                if (splitSize > batchMidFormer_) {
                    splitSize = batchMidFormer_;
                }

                int64_t outputOffset = splitStart * outStrides_[batchSplitAxis_];
                int64_t bcastOffset = 0;
                for (int64_t axis = batchSplitAxis_ - 1; axis >= 0; --axis) {
                    const int64_t coord = outerIndex % dims_[axis];
                    outerIndex /= dims_[axis];
                    outputOffset += coord * outStrides_[axis];
                    bcastOffset += coord * ((directBroadcastInputIndex_ == 1) ? x1Strides_[axis] : x2Strides_[axis]);
                }

                const uint32_t rowCount = static_cast<uint32_t>(splitSize);
                const uint32_t calcLen = rowCount * patternLen;
                const uint32_t dataBlockAlignedLen = AlignUpBytes(calcLen * sizeof(T), kAlignBytes) / sizeof(T);
                const uint32_t alignedLen = AlignForHalfCompare<T>(calcLen);
                LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                CopyInPad(fullLocal, 0, fullGm, outputOffset, calcLen);
                x1Queue_.EnQue(fullLocal);
                const bool refreshBcast = bcastOffset != cachedBcastOffset;
                LocalTensor<T> patternLocal;
                if (refreshBcast) {
                    patternLocal = batchPatternQueue_.AllocTensor<T>();
                    CopyInPad(patternLocal, 0, bcastGm, bcastOffset, patternLen);
                    batchPatternQueue_.EnQue(patternLocal);
                }

                LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
                fullLocal = x1Queue_.DeQue<T>();
                LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
                if (refreshBcast) {
                    patternLocal = batchPatternQueue_.DeQue<T>();
                    Gather(bcastLocal, patternLocal, gatherIndex, 0, kBatchGatherTileLength);
                    PipeBarrier<PIPE_V>();
                    batchPatternQueue_.FreeTensor(patternLocal);
                    cachedBcastOffset = bcastOffset;
                }
                if (alignedLen > dataBlockAlignedLen) {
                    FillLocal(fullLocal[dataBlockAlignedLen], static_cast<T>(0), alignedLen - dataBlockAlignedLen);
                    PipeBarrier<PIPE_V>();
                }
                if (directBroadcastInputIndex_ == 1) {
                    CompareTensors(outputLocal, maskLocal, valueLocal, bcastLocal, fullLocal, alignedLen);
                } else {
                    CompareTensors(outputLocal, maskLocal, valueLocal, fullLocal, bcastLocal, alignedLen);
                }
                x1Queue_.FreeTensor(fullLocal);
                ReleaseValueTensor(valueLocal);
                maskQueue_.FreeTensor(maskLocal);
                outQueue_.EnQue(outputLocal);
                outputLocal = outQueue_.DeQue<uint8_t>();
                CopyOutDirect(outputOffset, outputLocal, calcLen);
                outQueue_.FreeTensor(outputLocal);
            }
        }
    }

    __aicore__ inline void InitBatchGatherIndices()
    {
        if constexpr (!std::is_same<T, half>::value) {
            return;
        } else {
            LocalTensor<int32_t> offsets = batchIndexBuf_.Get<int32_t>();
            LocalTensor<float> quotient = batchScratchBuf_.Get<float>();
            ArithProgression(offsets, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(T)),
                             kBatchGatherTileLength);
            ArithProgression(quotient, 0.0f, 1.0f, kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
            Muls(quotient, quotient, batchPatternReciprocal_, kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
            Adds(quotient, quotient, 1.0e-6f, kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> quotientInt = batchScratchBuf_.Get<int32_t>();
            Cast(quotientInt, quotient, RoundMode::CAST_FLOOR, kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
            Muls(quotientInt, quotientInt, batchPatternLen_ * static_cast<int32_t>(sizeof(T)),
                 kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
            Sub(offsets, offsets, quotientInt, kBatchGatherTileLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessDirectBroadcastGenericFlat(int64_t flatStart, int64_t flatEnd)
    {
        GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
        GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
        int64_t segment = flatStart / directLowSize_;
        int64_t chunkStart = flatStart - segment * directLowSize_;
        while (segment < directTotalSegments_ && flatStart < flatEnd) {
            const int64_t srcSegment = ComputeDirectSourceSegment(segment);
            const int64_t fullBase = segment * directLowSize_;
            const int64_t bcastBase = srcSegment * directLowSize_;
            int64_t chunkEnd = flatEnd - fullBase;
            if (chunkEnd > directLowSize_) {
                chunkEnd = directLowSize_;
            }
            for (int64_t chunkOffset = chunkStart; chunkOffset < chunkEnd; chunkOffset += directTileElems_) {
                uint32_t len = static_cast<uint32_t>(chunkEnd - chunkOffset);
                if (len > directTileElems_) {
                    len = directTileElems_;
                }
                const uint32_t alignedLen = AlignForHalfCompare<T>(len);

                LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                LocalTensor<T> bcastLocal = x2Queue_.AllocTensor<T>();
                CopyInContiguous(fullLocal, fullGm, fullBase + chunkOffset, len, alignedLen);
                CopyInContiguous(bcastLocal, bcastGm, bcastBase + chunkOffset, len, alignedLen);
                x1Queue_.EnQue(fullLocal);
                x2Queue_.EnQue(bcastLocal);

                LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
                fullLocal = x1Queue_.DeQue<T>();
                bcastLocal = x2Queue_.DeQue<T>();
                LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
                if (directBroadcastInputIndex_ == 1) {
                    CompareTensors(outputLocal, maskLocal, valueLocal, bcastLocal, fullLocal, alignedLen);
                } else {
                    CompareTensors(outputLocal, maskLocal, valueLocal, fullLocal, bcastLocal, alignedLen);
                }
                x1Queue_.FreeTensor(fullLocal);
                x2Queue_.FreeTensor(bcastLocal);
                ReleaseValueTensor(valueLocal);
                maskQueue_.FreeTensor(maskLocal);
                outQueue_.EnQue(outputLocal);
                outputLocal = outQueue_.DeQue<uint8_t>();
                CopyOutDirect(fullBase + chunkOffset, outputLocal, len);
                outQueue_.FreeTensor(outputLocal);
            }
            ++segment;
            flatStart = segment * directLowSize_;
            chunkStart = 0;
        }
    }

    __aicore__ inline void ComputeDirectInt32WithCached(LocalTensor<T>& fullLocal,
                                                         LocalTensor<T>& bcastLocal,
                                                         int64_t fullOffset,
                                                         uint32_t len,
                                                         uint32_t alignedLen)
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            return;
        } else {
            LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
            if (directBroadcastInputIndex_ == 1) {
                Min(fullLocal, bcastLocal, fullLocal, alignedLen);
            } else {
                Max(fullLocal, fullLocal, bcastLocal, alignedLen);
            }
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, bcastLocal, fullLocal, CMPMODE::EQ, alignedLen);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, false);
            ReleaseValueTensor(valueLocal);
            maskQueue_.FreeTensor(maskLocal);
            outQueue_.EnQue(outputLocal);
            outputLocal = outQueue_.DeQue<uint8_t>();
            CopyOutDirect(fullOffset, outputLocal, len);
            outQueue_.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void ProcessDirectBroadcastCachedInt32(int64_t segmentStart, int64_t segmentEnd,
                                                              int64_t lowStart, int64_t lowEnd)
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            return;
        } else {
            GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
            GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
            const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
            int64_t segment = segmentStart;
            while (segment < segmentEnd) {
                int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
                if (groupEnd > segmentEnd) {
                    groupEnd = segmentEnd;
                }
                const int64_t srcSegment = ComputeDirectSourceSegment(segment);
                const int64_t bcastBase = srcSegment * directLowSize_;

                for (int64_t chunkOffset = lowStart; chunkOffset < lowEnd; chunkOffset += directTileElems_) {
                    uint32_t len = static_cast<uint32_t>(lowEnd - chunkOffset);
                    if (len > directTileElems_) {
                        len = directTileElems_;
                    }
                    const uint32_t alignedLen = AlignForHalfCompare<T>(len);

                    LocalTensor<T> bcastLocal = x2Queue_.AllocTensor<T>();
                    CopyInContiguous(bcastLocal, bcastGm, bcastBase + chunkOffset, len, alignedLen);
                    x2Queue_.EnQue(bcastLocal);
                    bcastLocal = x2Queue_.DeQue<T>();

                    for (int64_t currentSegment = segment; currentSegment < groupEnd; ++currentSegment) {
                        const int64_t fullOffset = currentSegment * directLowSize_ + chunkOffset;
                        LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                        CopyInContiguous(fullLocal, fullGm, fullOffset, len, alignedLen);
                        x1Queue_.EnQue(fullLocal);
                        fullLocal = x1Queue_.DeQue<T>();
                        ComputeDirectInt32WithCached(fullLocal, bcastLocal, fullOffset, len, alignedLen);
                        x1Queue_.FreeTensor(fullLocal);
                    }
                    x2Queue_.FreeTensor(bcastLocal);
                }
                segment = groupEnd;
            }
        }
    }

    __aicore__ inline void ExpandDirectInt32BroadcastRows(LocalTensor<T>& bcastLocal,
                                                           uint32_t alignedRowElems,
                                                           uint32_t rowCount)
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            return;
        } else {
            constexpr uint32_t repeatElems = kVectorAlignBytes / sizeof(T);
            uint32_t builtRows = 1;
            while (builtRows < rowCount) {
                uint32_t copyRows = builtRows;
                if (copyRows > rowCount - builtRows) {
                    copyRows = rowCount - builtRows;
                }
                const uint32_t copyElems = copyRows * alignedRowElems;
                const uint8_t repeatTimes = static_cast<uint8_t>(copyElems / repeatElems);
                Copy(bcastLocal[builtRows * alignedRowElems], bcastLocal, repeatElems,
                     repeatTimes, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                builtRows += copyRows;
            }
        }
    }

    __aicore__ inline void ProcessDirectBroadcastCachedInt32PaddedRows(
        int64_t segmentStart, int64_t segmentEnd)
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            return;
        } else {
            const uint32_t rowElems = static_cast<uint32_t>(directLowSize_);
            const uint32_t alignedRowElems =
                AlignUpBytes(rowElems * sizeof(T), kAlignBytes) / sizeof(T);
            const uint32_t rowsPerTile = processTileLength_ / alignedRowElems;
            if (rowsPerTile == 0) {
                return;
            }

            GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
            GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
            const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
            const uint32_t rowBytes = rowElems * sizeof(T);
            const uint8_t rightPadding = static_cast<uint8_t>(alignedRowElems - rowElems);
            const DataCopyPadExtParams<T> padParams{true, 0, rightPadding, static_cast<T>(0)};

            int64_t segment = segmentStart;
            while (segment < segmentEnd) {
                int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
                if (groupEnd > segmentEnd) {
                    groupEnd = segmentEnd;
                }
                const uint32_t expandedRows = static_cast<uint32_t>(
                    groupEnd - segment > rowsPerTile ? rowsPerTile : groupEnd - segment);
                const int64_t srcSegment = ComputeDirectSourceSegment(segment);
                LocalTensor<T> bcastLocal = x2Queue_.AllocTensor<T>();
                const DataCopyExtParams bcastCopyParams{1, rowBytes, 0, 0, 0};
                DataCopyPad(bcastLocal, bcastGm[srcSegment * directLowSize_],
                            bcastCopyParams, padParams);
                x2Queue_.EnQue(bcastLocal);
                bcastLocal = x2Queue_.DeQue<T>();
                ExpandDirectInt32BroadcastRows(bcastLocal, alignedRowElems, expandedRows);

                while (segment < groupEnd) {
                    uint32_t rowCount = static_cast<uint32_t>(groupEnd - segment);
                    if (rowCount > rowsPerTile) {
                        rowCount = rowsPerTile;
                    }
                    const uint32_t calcLen = rowCount * alignedRowElems;
                    const int64_t fullOffset = segment * directLowSize_;

                    LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                    const DataCopyExtParams fullCopyParams{
                        static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
                    DataCopyPad(fullLocal, fullGm[fullOffset], fullCopyParams, padParams);
                    x1Queue_.EnQue(fullLocal);
                    fullLocal = x1Queue_.DeQue<T>();

                    LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
                    LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
                    if (directBroadcastInputIndex_ == 1) {
                        Min(fullLocal, bcastLocal, fullLocal, calcLen);
                    } else {
                        Max(fullLocal, fullLocal, bcastLocal, calcLen);
                    }
                    PipeBarrier<PIPE_V>();
                    Compare(maskLocal, bcastLocal, fullLocal, CMPMODE::EQ, calcLen);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
                    WriteMaskAsBool(maskLocal, valueLocal, outputLocal, calcLen, false);
                    ReleaseValueTensor(valueLocal);
                    maskQueue_.FreeTensor(maskLocal);
                    x1Queue_.FreeTensor(fullLocal);
                    outQueue_.EnQue(outputLocal);
                    outputLocal = outQueue_.DeQue<uint8_t>();
                    const DataCopyExtParams outputCopyParams{
                        static_cast<uint16_t>(rowCount), rowElems, 0, 0, 0};
                    DataCopyPad(yGm_[fullOffset], outputLocal, outputCopyParams);
                    outQueue_.FreeTensor(outputLocal);
                    segment += rowCount;
                }
                x2Queue_.FreeTensor(bcastLocal);
            }
        }
    }

    __aicore__ inline void ProcessDirectBroadcast()
    {
        if (directLowSize_ <= 0 || directTotalSegments_ <= 0) {
            return;
        }

        if constexpr (std::is_same<T, int32_t>::value) {
            const int64_t blockNum = GetBlockNum();
            const int64_t totalBytes = totalElems_ * static_cast<int64_t>(sizeof(T));
            const int64_t innerOuterAxis = directOuterRank_ - 1;
            const bool repeatedSource = innerOuterAxis >= 0 && directOuterDims_[innerOuterAxis] > 1 &&
                                        directSrcStrides_[innerOuterAxis] == 0;
            const bool singleSourceGroup =
                repeatedSource && directOuterDims_[innerOuterAxis] == directTotalSegments_;
            if (repeatedSource && blockNum <= directTotalSegments_ &&
                directTotalSegments_ / blockNum >= 8) {
                int64_t segmentStart = 0;
                int64_t segmentEnd = 0;
                int64_t lowStart = 0;
                int64_t lowEnd = 0;
                if (ResolveDirectCoreRange(segmentStart, segmentEnd, lowStart, lowEnd)) {
                    if (directLowSize_ == 1023 && lowStart == 0 && lowEnd == directLowSize_) {
                        ProcessDirectBroadcastCachedInt32PaddedRows(segmentStart, segmentEnd);
                    } else {
                        ProcessDirectBroadcastCachedInt32(segmentStart, segmentEnd, lowStart, lowEnd);
                    }
                }
                return;
            }
            if (blockNum <= directTotalSegments_ &&
                (singleSourceGroup || directTotalSegments_ <= kDirectBalancedMaxSegments) &&
                totalBytes >= blockNum * 512) {
                int64_t flatStart = 0;
                int64_t flatEnd = 0;
                if (ResolveDirectAlignedFlatRange(flatStart, flatEnd)) {
                    ProcessDirectBroadcastGenericFlat(flatStart, flatEnd);
                }
                return;
            }
        }

        int64_t segmentStart = 0;
        int64_t segmentEnd = 0;
        int64_t lowStart = 0;
        int64_t lowEnd = 0;
        if (!ResolveDirectCoreRange(segmentStart, segmentEnd, lowStart, lowEnd)) {
            return;
        }

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            const int64_t innerOuterAxis = directOuterRank_ - 1;
            if (innerOuterAxis >= 0 && directOuterDims_[innerOuterAxis] > 1 &&
                directSrcStrides_[innerOuterAxis] == 0) {
                ProcessDirectBroadcastCachedBfloatDualQueue(segmentStart, segmentEnd, lowStart, lowEnd);
                return;
            }
        }

        GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
        GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
        for (int64_t segment = segmentStart; segment < segmentEnd; ++segment) {
            const int64_t srcSegment = ComputeDirectSourceSegment(segment);
            const int64_t fullBase = segment * directLowSize_;
            const int64_t bcastBase = srcSegment * directLowSize_;
            for (int64_t chunkOffset = lowStart; chunkOffset < lowEnd; chunkOffset += directTileElems_) {
                uint32_t len = static_cast<uint32_t>(lowEnd - chunkOffset);
                if (len > directTileElems_) {
                    len = directTileElems_;
                }
                const uint32_t alignedLen = AlignForHalfCompare<T>(len);

                LocalTensor<T> fullLocal = x1Queue_.AllocTensor<T>();
                LocalTensor<T> bcastLocal = x2Queue_.AllocTensor<T>();
                CopyInContiguous(fullLocal, fullGm, fullBase + chunkOffset, len, alignedLen);
                CopyInContiguous(bcastLocal, bcastGm, bcastBase + chunkOffset, len, alignedLen);
                x1Queue_.EnQue(fullLocal);
                x2Queue_.EnQue(bcastLocal);

                LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
                fullLocal = x1Queue_.DeQue<T>();
                bcastLocal = x2Queue_.DeQue<T>();
                LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
                if (directBroadcastInputIndex_ == 1) {
                    CompareTensors(outputLocal, maskLocal, valueLocal, bcastLocal, fullLocal, alignedLen);
                } else {
                    CompareTensors(outputLocal, maskLocal, valueLocal, fullLocal, bcastLocal, alignedLen);
                }
                x1Queue_.FreeTensor(fullLocal);
                x2Queue_.FreeTensor(bcastLocal);
                ReleaseValueTensor(valueLocal);
                maskQueue_.FreeTensor(maskLocal);
                outQueue_.EnQue(outputLocal);
                outputLocal = outQueue_.DeQue<uint8_t>();
                CopyOutDirect(fullBase + chunkOffset, outputLocal, len);
                outQueue_.FreeTensor(outputLocal);
            }
        }
    }

    __aicore__ inline void ComputeDirectBfloatWithCached(
        LocalTensor<T>& fullLocal, LocalTensor<float>& bcastFloat, int64_t fullOffset,
        uint32_t len, uint32_t alignedLen)
    {
        LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
        LocalTensor<float> fullFloat = x1FloatQueue_.AllocTensor<float>();
        Cast(fullFloat, fullLocal, RoundMode::CAST_NONE, alignedLen);
        PipeBarrier<PIPE_V>();
        if (directBroadcastInputIndex_ == 1) {
            Compare(maskLocal, bcastFloat, fullFloat, CMPMODE::GT, alignedLen);
        } else {
            Compare(maskLocal, fullFloat, bcastFloat, CMPMODE::GT, alignedLen);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
        WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, true);
        x1FloatQueue_.FreeTensor(fullFloat);
        ReleaseValueTensor(valueLocal);
        maskQueue_.FreeTensor(maskLocal);
        outQueue_.EnQue(outputLocal);
        outputLocal = outQueue_.DeQue<uint8_t>();
        CopyOutDirect(fullOffset, outputLocal, len);
        outQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ProcessDirectBroadcastCachedBfloatDualQueue(int64_t segmentStart, int64_t segmentEnd,
                                                                        int64_t lowStart, int64_t lowEnd)
    {
        if constexpr (!std::is_same<T, bfloat16_t>::value) {
            return;
        } else {
            GlobalTensor<T>& fullGm = (directBroadcastInputIndex_ == 1) ? x2Gm_ : x1Gm_;
            GlobalTensor<T>& bcastGm = (directBroadcastInputIndex_ == 1) ? x1Gm_ : x2Gm_;
            const int64_t repeatSegments = directOuterDims_[directOuterRank_ - 1];
            int64_t segment = segmentStart;
            while (segment < segmentEnd) {
                int64_t groupEnd = ((segment / repeatSegments) + 1) * repeatSegments;
                if (groupEnd > segmentEnd) {
                    groupEnd = segmentEnd;
                }
                const int64_t srcSegment = ComputeDirectSourceSegment(segment);
                const int64_t bcastBase = srcSegment * directLowSize_;

                for (int64_t chunkOffset = lowStart; chunkOffset < lowEnd; chunkOffset += directTileElems_) {
                    uint32_t len = static_cast<uint32_t>(lowEnd - chunkOffset);
                    if (len > directTileElems_) {
                        len = directTileElems_;
                    }
                    const uint32_t alignedLen = AlignForHalfCompare<T>(len);

                    LocalTensor<T> bcastLocal = x2Queue_.AllocTensor<T>();
                    CopyInContiguous(bcastLocal, bcastGm, bcastBase + chunkOffset, len, alignedLen);
                    x2Queue_.EnQue(bcastLocal);
                    bcastLocal = x2Queue_.DeQue<T>();
                    LocalTensor<float> bcastFloat = x2FloatQueue_.AllocTensor<float>();
                    Cast(bcastFloat, bcastLocal, RoundMode::CAST_NONE, alignedLen);
                    PipeBarrier<PIPE_V>();
                    x2Queue_.FreeTensor(bcastLocal);

                    int64_t currentSegment = segment;
                    bool currentInX1 = true;
                    LocalTensor<T> firstFullLocal = x1Queue_.AllocTensor<T>();
                    CopyInContiguous(firstFullLocal, fullGm,
                                     currentSegment * directLowSize_ + chunkOffset, len, alignedLen);
                    x1Queue_.EnQue(firstFullLocal);

                    while (currentSegment < groupEnd) {
                        const int64_t nextSegment = currentSegment + 1;
                        if (nextSegment < groupEnd) {
                            if (currentInX1) {
                                LocalTensor<T> nextFullLocal = x2Queue_.AllocTensor<T>();
                                CopyInContiguous(nextFullLocal, fullGm,
                                                 nextSegment * directLowSize_ + chunkOffset, len, alignedLen);
                                x2Queue_.EnQue(nextFullLocal);
                            } else {
                                LocalTensor<T> nextFullLocal = x1Queue_.AllocTensor<T>();
                                CopyInContiguous(nextFullLocal, fullGm,
                                                 nextSegment * directLowSize_ + chunkOffset, len, alignedLen);
                                x1Queue_.EnQue(nextFullLocal);
                            }
                        }

                        const int64_t fullOffset = currentSegment * directLowSize_ + chunkOffset;
                        if (currentInX1) {
                            LocalTensor<T> fullLocal = x1Queue_.DeQue<T>();
                            ComputeDirectBfloatWithCached(fullLocal, bcastFloat, fullOffset, len, alignedLen);
                            x1Queue_.FreeTensor(fullLocal);
                        } else {
                            LocalTensor<T> fullLocal = x2Queue_.DeQue<T>();
                            ComputeDirectBfloatWithCached(fullLocal, bcastFloat, fullOffset, len, alignedLen);
                            x2Queue_.FreeTensor(fullLocal);
                        }
                        currentInX1 = !currentInX1;
                        currentSegment = nextSegment;
                    }
                    x2FloatQueue_.FreeTensor(bcastFloat);
                }
                segment = groupEnd;
            }
        }
    }

    __aicore__ inline void CopyIn(int64_t start, uint32_t alignedLen)
    {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        CopyInContiguous(x1Local, x1CompareGm_, start, calcLength_, alignedLen);
        CopyInContiguous(x2Local, x2CompareGm_, start, calcLength_, alignedLen);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void ProcessFloatNoBroadcastSingleBuffer()
    {
        if constexpr (!std::is_same<T, float>::value) {
            return;
        } else {
            InitSocState();
            const int64_t coreStart = static_cast<int64_t>(GetBlockIdx()) * elemsPerCore_;
            if (coreStart >= totalElems_) {
                return;
            }
            int64_t coreEnd = coreStart + elemsPerCore_;
            if (coreEnd > totalElems_) {
                coreEnd = totalElems_;
            }

            LocalTensor<float> x1Local(TPosition::VECCALC, kFp32SingleX1Addr,
                                       kFp32NoBroadcastSingleBufferTileLength);
            LocalTensor<float> x2Local(TPosition::VECCALC, kFp32SingleX2Addr,
                                       kFp32NoBroadcastSingleBufferTileLength);
            LocalTensor<uint8_t> maskLocal(TPosition::VECCALC, kFp32SingleMaskAddr,
                                           kFp32SingleMaskBytes);
            LocalTensor<half> valueLocal(TPosition::VECCALC, kFp32SingleValueAddr,
                                         kFp32NoBroadcastSingleBufferTileLength);
            LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, kFp32SingleOutputAddr,
                                             kFp32NoBroadcastSingleBufferTileLength);

            int64_t tileStart = coreStart;
            uint32_t currentLen = static_cast<uint32_t>(coreEnd - tileStart);
            if (currentLen > kFp32NoBroadcastSingleBufferTileLength) {
                currentLen = kFp32NoBroadcastSingleBufferTileLength;
            }
            uint32_t currentAlignedLen = AlignForHalfCompare<float>(currentLen);
            CopyInPad(x1Local, 0, x1CompareGm_, tileStart, currentLen);
            CopyInPad(x2Local, 0, x2CompareGm_, tileStart, currentLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            bool outputInFlight = false;
            while (tileStart < coreEnd) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Compare(maskLocal, x1Local, x2Local, CMPMODE::GT, currentAlignedLen);
                PipeBarrier<PIPE_V>();

                const int64_t nextStart = tileStart + currentLen;
                const bool hasNext = nextStart < coreEnd;
                uint32_t nextLen = 0;
                uint32_t nextAlignedLen = 0;
                if (hasNext) {
                    nextLen = static_cast<uint32_t>(coreEnd - nextStart);
                    if (nextLen > kFp32NoBroadcastSingleBufferTileLength) {
                        nextLen = kFp32NoBroadcastSingleBufferTileLength;
                    }
                    nextAlignedLen = AlignForHalfCompare<float>(nextLen);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyInPad(x1Local, 0, x1CompareGm_, nextStart, nextLen);
                    CopyInPad(x2Local, 0, x2CompareGm_, nextStart, nextLen);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }

                Duplicate(valueLocal, static_cast<half>(1.0f), currentAlignedLen);
                PipeBarrier<PIPE_V>();
                Select(valueLocal, maskLocal, valueLocal, static_cast<half>(0.0f),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, currentAlignedLen);
                PipeBarrier<PIPE_V>();
                if (outputInFlight) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
                Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, currentAlignedLen);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams copyParams = {
                    1, static_cast<uint32_t>(currentLen * sizeof(uint8_t)), 0, 0, 0};
                DataCopyPad(yGm_[tileStart], outputLocal, copyParams);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                outputInFlight = true;

                tileStart = nextStart;
                currentLen = nextLen;
                currentAlignedLen = nextAlignedLen;
            }
            if (outputInFlight) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void CopyInScalar(int64_t start, uint32_t alignedLen)
    {
        if (scalarInputIndex_ == 1) {
            LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
            CopyInContiguous(x2Local, x2CompareGm_, start, calcLength_, alignedLen);
            x2Queue_.EnQue(x2Local);
        } else {
            LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
            CopyInContiguous(x1Local, x1CompareGm_, start, calcLength_, alignedLen);
            x1Queue_.EnQue(x1Local);
        }
    }

    __aicore__ inline void FillLocal(LocalTensor<T> local, T value, uint32_t len)
    {
        if (len == 0) {
            return;
        }
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            Duplicate(local.template ReinterpretCast<uint16_t>(), PackDupByte(value), (len + 1U) / 2U);
        } else {
            Duplicate(local, value, len);
        }
    }

    __aicore__ inline LocalTensor<half> AcquireValueTensor(LocalTensor<T>& reusableLocal)
    {
        if constexpr (std::is_same<T, int8_t>::value) {
            return valueQueue_.AllocTensor<half>();
        } else {
            return reusableLocal.template ReinterpretCast<half>();
        }
    }

    __aicore__ inline void ReleaseValueTensor(LocalTensor<half>& valueLocal)
    {
        if constexpr (std::is_same<T, int8_t>::value) {
            valueQueue_.FreeTensor(valueLocal);
        }
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal, LocalTensor<half>& valueLocal,
                                           LocalTensor<uint8_t>& outputLocal, uint32_t alignedLen,
                                           bool trueWhenMask = true)
    {
        const half baseValue = trueWhenMask ? static_cast<half>(1.0f) : static_cast<half>(0.0f);
        const half selectedValue = trueWhenMask ? static_cast<half>(0.0f) : static_cast<half>(1.0f);
        Duplicate(valueLocal, baseValue, alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, selectedValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
    }

    __aicore__ inline void CompareTensors(LocalTensor<uint8_t>& outputLocal, LocalTensor<uint8_t>& maskLocal,
                                          LocalTensor<half>& valueLocal, LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
                                          uint32_t alignedLen)
    {
        if constexpr (std::is_same<T, half>::value) {
            Compare(maskLocal, x1Local, x2Local, CMPMODE::GT, alignedLen);
        } else if constexpr (std::is_same<T, float>::value) {
            Compare(maskLocal, x1Local, x2Local, CMPMODE::GT, alignedLen);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x1Float = x1FloatQueue_.AllocTensor<float>();
            LocalTensor<float> x2Float = x2FloatQueue_.AllocTensor<float>();
            Cast(x1Float, x1Local, RoundMode::CAST_NONE, alignedLen);
            Cast(x2Float, x2Local, RoundMode::CAST_NONE, alignedLen);
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, x1Float, x2Float, CMPMODE::GT, alignedLen);
            x1FloatQueue_.FreeTensor(x1Float);
            x2FloatQueue_.FreeTensor(x2Float);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            LocalTensor<int32_t> tmp;
            if (mode_ == kModeDirectBroadcast) {
                Min(x2Local, x1Local, x2Local, alignedLen);
            } else {
                tmp = int32Queue1_.AllocTensor<int32_t>();
                Min(tmp, x1Local, x2Local, alignedLen);
            }
            PipeBarrier<PIPE_V>();
            if (mode_ == kModeDirectBroadcast) {
                Compare(maskLocal, x1Local, x2Local, CMPMODE::EQ, alignedLen);
            } else {
                Compare(maskLocal, x1Local, tmp, CMPMODE::EQ, alignedLen);
            }
            PipeBarrier<PIPE_V>();
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, false);
            if (mode_ != kModeDirectBroadcast) {
                int32Queue1_.FreeTensor(tmp);
            }
            return;
        } else {
            LocalTensor<half> x1Half = x1CastQueue_.AllocTensor<half>();
            LocalTensor<half> x2Half = x2CastQueue_.AllocTensor<half>();
            Cast(x1Half, x1Local, RoundMode::CAST_NONE, alignedLen);
            Cast(x2Half, x2Local, RoundMode::CAST_NONE, alignedLen);
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, x1Half, x2Half, CMPMODE::GT, alignedLen);
            x1CastQueue_.FreeTensor(x1Half);
            x2CastQueue_.FreeTensor(x2Half);
        }

        PipeBarrier<PIPE_V>();
        WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, true);
    }

    __aicore__ inline void Compute(uint32_t alignedLen)
    {
        LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<half> valueLocal = AcquireValueTensor(x1Local);
        CompareTensors(outputLocal, maskLocal, valueLocal, x1Local, x2Local, alignedLen);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
        ReleaseValueTensor(valueLocal);
        maskQueue_.FreeTensor(maskLocal);
        outQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CompareTensorWithScalar(LocalTensor<uint8_t>& outputLocal,
                                                   LocalTensor<uint8_t>& maskLocal,
                                                   LocalTensor<half>& valueLocal,
                                                   LocalTensor<T>& vectorLocal,
                                                   uint32_t alignedLen)
    {
        const CMPMODE cmpMode = (scalarInputIndex_ == 1) ? CMPMODE::LT : CMPMODE::GT;
        CompareScalar(maskLocal, vectorLocal, scalarValue_, cmpMode, alignedLen);
        PipeBarrier<PIPE_V>();
        WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, true);
    }

    __aicore__ inline void ComputeScalar(uint32_t alignedLen)
    {
        LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
        LocalTensor<T> fullLocal = (scalarInputIndex_ == 1) ? x2Queue_.DeQue<T>() : x1Queue_.DeQue<T>();
        LocalTensor<half> valueLocal = AcquireValueTensor(fullLocal);
        if constexpr (UseCompareScalar<T>()) {
            CompareTensorWithScalar(outputLocal, maskLocal, valueLocal, fullLocal, alignedLen);
        } else if constexpr (std::is_same<T, int8_t>::value) {
            LocalTensor<half> fullHalf = (scalarInputIndex_ == 1) ? x2CastQueue_.AllocTensor<half>()
                                                                 : x1CastQueue_.AllocTensor<half>();
            Cast(fullHalf, fullLocal, RoundMode::CAST_NONE, alignedLen);
            PipeBarrier<PIPE_V>();
            if (scalarInputIndex_ == 1) {
                Compare(maskLocal, scalarHalfVector_, fullHalf, CMPMODE::GT, alignedLen);
            } else {
                Compare(maskLocal, fullHalf, scalarHalfVector_, CMPMODE::GT, alignedLen);
            }
            PipeBarrier<PIPE_V>();
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen, true);
            if (scalarInputIndex_ == 1) {
                x2CastQueue_.FreeTensor(fullHalf);
            } else {
                x1CastQueue_.FreeTensor(fullHalf);
            }
        } else {
            LocalTensor<T>& x1Local = (scalarInputIndex_ == 1) ? scalarVector_ : fullLocal;
            LocalTensor<T>& x2Local = (scalarInputIndex_ == 1) ? fullLocal : scalarVector_;
            CompareTensors(outputLocal, maskLocal, valueLocal, x1Local, x2Local, alignedLen);
        }
        if (scalarInputIndex_ == 1) {
            x2Queue_.FreeTensor(fullLocal);
        } else {
            x1Queue_.FreeTensor(fullLocal);
        }
        ReleaseValueTensor(valueLocal);
        maskQueue_.FreeTensor(maskLocal);
        outQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t start)
    {
        LocalTensor<uint8_t> outputLocal = outQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(calcLength_ * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm_[start], outputLocal, copyParams);
        outQueue_.FreeTensor(outputLocal);
    }

private:
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECCALC, 1> x1CastQueue_;
    TQue<QuePosition::VECCALC, 1> x2CastQueue_;
    TQue<QuePosition::VECCALC, 1> x1FloatQueue_;
    TQue<QuePosition::VECCALC, 1> x2FloatQueue_;
    TQue<QuePosition::VECCALC, 1> int32Queue1_;
    TQue<QuePosition::VECCALC, 1> maskQueue_;
    TQue<QuePosition::VECCALC, 1> valueQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECIN, 1> batchPatternQueue_;
    TQue<QuePosition::VECIN, 1> scalarSourceQueue_;
    TBuf<TPosition::VECCALC> batchIndexBuf_;
    TBuf<TPosition::VECCALC> batchScratchBuf_;
    TBuf<TPosition::VECCALC> batchBcastCacheBuf_;
    TBuf<TPosition::VECCALC> compareBankGapBuf_;
    TBuf<TPosition::VECCALC> castBankGapBuf_;
    TBuf<TPosition::VECCALC> scalarIndexBuf_;
    TBuf<TPosition::VECCALC> scalarScratchBuf_;
    TBuf<TPosition::VECCALC> scalarBankGapBuf_;
    TBuf<TPosition::VECCALC> scalarVectorBuf_;
    BroadcastMaterializer<T> broadcaster_;
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> x1CompareGm_;
    GlobalTensor<T> x2CompareGm_;
    GlobalTensor<uint8_t> yGm_;
    GM_ADDR x1InputAddr_ = nullptr;
    GM_ADDR x2InputAddr_ = nullptr;
    GM_ADDR x1CompareAddr_ = nullptr;
    GM_ADDR x2CompareAddr_ = nullptr;
    GM_ADDR scratchAddr_ = nullptr;
    int64_t totalElems_ = 0;
    int64_t x1TotalElems_ = 0;
    int64_t x2TotalElems_ = 0;
    int64_t elemsPerCore_ = 0;
    int64_t rank_ = 0;
    int64_t mode_ = kModeNoBroadcast;
    int64_t scalarInputIndex_ = 0;
    int64_t directBroadcastInputIndex_ = 0;
    uint32_t elemSize_ = 0;
    int64_t x1WorkspaceOffset_ = -1;
    int64_t x2WorkspaceOffset_ = -1;
    int64_t scratchWorkspaceOffset_ = -1;
    int64_t directLowSize_ = 0;
    int64_t directTotalSegments_ = 0;
    int64_t directOuterRank_ = 0;
    int64_t batchSplitAxis_ = 0;
    int64_t batchMidFormer_ = 1;
    int64_t batchTotalTiles_ = 0;
    int64_t directOuterDims_[kCanonRank] = {1, 1, 1, 1};
    int64_t directOutStrides_[kCanonRank] = {1, 1, 1, 1};
    int64_t directSrcStrides_[kCanonRank] = {0, 0, 0, 0};
    int64_t dims_[kMaxRank] = {};
    int64_t outStrides_[kMaxRank] = {};
    int64_t x1Strides_[kMaxRank] = {};
    int64_t x2Strides_[kMaxRank] = {};
    int64_t canonX1_[kCanonRank] = {1, 1, 1, 1};
    int64_t canonX2_[kCanonRank] = {1, 1, 1, 1};
    int64_t canonOut_[kCanonRank] = {1, 1, 1, 1};
    uint32_t calcLength_ = 0;
    uint32_t processTileLength_ = 1;
    uint32_t directTileElems_ = 0;
    int32_t batchPatternLen_ = 1;
    bool bf16ScalarCached_ = false;
    bool useFp32NoBroadcastSingleBuffer_ = false;
    float batchPatternReciprocal_ = 1.0f;
    T scalarValue_ = static_cast<T>(0);
    LocalTensor<T> scalarVector_;
    LocalTensor<half> scalarHalfVector_;
};

template <typename T>
__aicore__ inline void RunGreater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const GreaterTilingData& tiling)
{
    TPipe pipe;
    KernelGreater<T> kernel;
    kernel.Init(x1, x2, y, workspace, tiling, &pipe);
    kernel.Process();
}

__aicore__ inline void RunGreaterBf16RepeatedCache(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                    const GreaterTilingData& tiling)
{
    KernelGreaterBf16RepeatedCache kernel;
    kernel.Init(x1, x2, y, tiling);
    kernel.Process();
}

__aicore__ inline void RunGreaterInt32CompactStatic(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                     const GreaterTilingData& tiling)
{
    TPipe pipe;
    KernelGreaterInt32CompactStatic kernel;
    kernel.Init(x1, x2, y, tiling, &pipe);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunGreaterNoBroadcastStatic(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                    const GreaterTilingData& tiling)
{
    KernelGreaterNoBroadcastStatic<T> kernel;
    kernel.Init(x1, x2, y, tiling);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunGreaterDirectBroadcastStatic(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                        const GreaterTilingData& tiling)
{
    KernelGreaterDirectBroadcastStatic<T> kernel;
    kernel.Init(x1, x2, y, tiling);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunGreaterScalarBroadcastStatic(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                        const GreaterTilingData& tiling)
{
    KernelGreaterScalarBroadcastStatic<T> kernel;
    kernel.Init(x1, x2, y, tiling);
    kernel.Process();
}

__aicore__ inline void RunGreaterBatchBroadcastStatic(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                       const GreaterTilingData& tiling)
{
    KernelGreaterBatchBroadcastStatic kernel;
    kernel.Init(x1, x2, y, tiling);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(7) && GetBlockIdx() != 0) {
        return;
    }
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        RunGreaterBf16RepeatedCache(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(2)) {
        RunGreaterNoBroadcastStatic<DTYPE_X1>(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(3)) {
        RunGreaterDirectBroadcastStatic<DTYPE_X1>(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(4)) {
        RunGreaterScalarBroadcastStatic<DTYPE_X1>(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(5)) {
        RunGreaterBatchBroadcastStatic(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(6)) {
        RunGreater<DTYPE_X1>(x1, x2, y, workspace, tiling_data);
        return;
    }
    if (TILING_KEY_IS(7)) {
        RunGreaterNoBroadcastStatic<DTYPE_X1>(x1, x2, y, tiling_data);
        return;
    }
    if (TILING_KEY_IS(8)) {
        RunGreaterInt32CompactStatic(x1, x2, y, tiling_data);
        return;
    }
    if (tiling_data.mode == kModeWorkspaceBroadcast) {
        if (workspace == nullptr) {
            return;
        }
        SetSysWorkspace(workspace);
        if (GetUserWorkspace(workspace) == nullptr) {
            return;
        }
    }
    if (TILING_KEY_IS(0)) {
        RunGreater<DTYPE_X1>(x1, x2, y, workspace, tiling_data);
    }
}
