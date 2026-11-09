#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kWorkspaceTileBytes = 95 * 1024;
constexpr uint64_t kLargeStreamBytes = 1024U * 1024U;
constexpr uint32_t kSingleWorkerElements = 4096U;
constexpr uint32_t kBf16RowsPerTile = 4U;
constexpr uint64_t kBf16UbBudget = 184U * 1024U;

template <typename T>
struct CompareType { using type = T; };

template <>
struct CompareType<bfloat16_t> { using type = float; };

template <>
struct CompareType<int8_t> { using type = half; };

template <typename T>
using CompareT = typename CompareType<T>::type;

template <typename T>
constexpr bool kNeedsCast = !std::is_same_v<T, CompareT<T>>;

template <typename T>
__aicore__ inline uint32_t Min(T a, T b)
{ return static_cast<uint32_t>(a < b ? a : b); }

template <typename T>
__aicore__ inline T CeilDiv(T value, T divisor)
{ return (value + divisor - 1) / divisor; }

template <typename T>
__aicore__ inline uint32_t TileElements()
{
    if constexpr (std::is_same_v<T, half>) return 16384U;
    if constexpr (std::is_same_v<T, float>) return 8448U;
    return 8192U;
}

template <typename T, bool SingleOutput = false>
__aicore__ inline uint32_t LinearTileElements()
{
    if constexpr (std::is_same_v<T, half>) return 18432U;
    if constexpr (std::is_same_v<T, int8_t>) return 19200U;
    if constexpr (std::is_same_v<T, float>) return SingleOutput ? 11392U : 10240U;
    return 10240U;
}

template <typename T>
__aicore__ inline uint32_t VectorLength(uint32_t length)
{
    const uint32_t align = 256U / sizeof(T);
    return CeilDiv(length, align) * align;
}

__aicore__ inline uint32_t MaskBytes(uint32_t length)
{ return CeilDiv(length, 256U) * kAlignBytes; }

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<T>& local, GlobalTensor<T>& global,
                               uint32_t offset, uint32_t length)
{
    const uint32_t bytes = length * sizeof(T);
    if ((offset * sizeof(T)) % kAlignBytes == 0U && bytes % kAlignBytes == 0U) {
        DataCopy(local, global[offset], length);
    } else {
        DataCopyExtParams copy = {1, bytes, 0, 0, 0};
        DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, global[offset], copy, pad);
    }
}

template <typename T>
__aicore__ inline void CopyInBatch(LocalTensor<T>& local, GlobalTensor<T>& global,
                                   uint32_t offset, uint32_t blockLength,
                                   uint32_t localStride, uint16_t blockCount)
{
    const uint32_t blockBytes = blockLength * sizeof(T);
    const uint32_t alignedBytes = CeilDiv(blockBytes, kAlignBytes) * kAlignBytes;
    DataCopyExtParams copy = {
        blockCount, blockBytes, 0,
        static_cast<uint32_t>((localStride * sizeof(T) - alignedBytes) / kAlignBytes), 0};
    DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, global[offset], copy, pad);
}

__aicore__ inline void CopyOut(GlobalTensor<uint8_t>& global, LocalTensor<uint8_t>& local,
                                uint32_t offset, uint32_t length)
{
    if (offset % kAlignBytes == 0U && length % kAlignBytes == 0U) {
        DataCopy(global[offset], local, length);
    } else {
        DataCopyExtParams copy = {1, length, 0, 0, 0};
        DataCopyPad(global[offset], local, copy);
    }
}

__aicore__ inline void CopyOutBatch(GlobalTensor<uint8_t>& global,
                                    LocalTensor<uint8_t>& local,
                                    uint32_t offset, uint32_t blockLength,
                                    uint32_t localStride, uint16_t blockCount)
{
    const uint32_t alignedLength = CeilDiv(blockLength, kAlignBytes) * kAlignBytes;
    DataCopyExtParams copy = {
        blockCount, blockLength,
        static_cast<uint32_t>((localStride - alignedLength) / kAlignBytes), 0, 0};
    DataCopyPad(global[offset], local, copy);
}

__aicore__ inline void MaskToBool(LocalTensor<half>& values, LocalTensor<uint8_t>& mask,
                                   LocalTensor<uint8_t>& output, uint32_t length)
{
    Duplicate(values, static_cast<half>(1.0f), length);
    PipeBarrier<PIPE_V>();
    Select(values, mask, values, static_cast<half>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    PipeBarrier<PIPE_V>();
    Cast(output, values, RoundMode::CAST_TRUNC, length);
}

__aicore__ inline void MaskToBool(LocalTensor<half>& values, LocalTensor<uint8_t>& mask,
                                   LocalTensor<uint8_t>& output, LocalTensor<half>& ones,
                                   uint32_t length)
{
    Select(values, mask, ones, static_cast<half>(0.0f),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    PipeBarrier<PIPE_V>();
    Cast(output, values, RoundMode::CAST_TRUNC, length);
}

__aicore__ inline void MaskToBoolRepeatedConstants(LocalTensor<half>& values,
                                                    LocalTensor<uint8_t>& mask,
                                                    LocalTensor<uint8_t>& output,
                                                    LocalTensor<half>& ones,
                                                    LocalTensor<half>& zeros,
                                                    uint32_t length)
{
    BinaryRepeatParams params(1, 1, 1, 8, 0, 0);
    const uint8_t repeats = static_cast<uint8_t>(CeilDiv(length, 128U));
    Select(values, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           static_cast<uint64_t>(128U), repeats, params);
    PipeBarrier<PIPE_V>();
    Cast(output, values, RoundMode::CAST_TRUNC, repeats * 128U);
}

__aicore__ inline void MaskToBoolFixedMask(LocalTensor<half>& values,
                                           LocalTensor<uint8_t>& output,
                                           LocalTensor<half>& ones,
                                           LocalTensor<half>& zeros,
                                           uint32_t length)
{
    BinaryRepeatParams selectParams(1, 1, 1, 8, 0, 0);
    const uint8_t repeats = static_cast<uint8_t>(CeilDiv(length, 128U));
    Select<half, SELMODE::VSEL_TENSOR_TENSOR_MODE>(values, ones, zeros,
                                                   repeats, selectParams);
    PipeBarrier<PIPE_V>();
    Cast(output, values, RoundMode::CAST_TRUNC, repeats * 128U);
}

template <typename T>
__aicore__ inline void CompareToBool(LocalTensor<T>& x1, LocalTensor<T>& x2,
                                      LocalTensor<uint8_t>& mask, LocalTensor<half>& values,
                                      LocalTensor<uint8_t>& output, uint32_t length)
{
    Compare(mask, x1, x2, CMPMODE::GT, length);
    MaskToBool(values, mask, output, length);
}

template <bool Less, typename T>
__aicore__ inline void CastCompare(LocalTensor<uint8_t>& mask, LocalTensor<T>& x1,
                                   LocalTensor<T>& x2, LocalTensor<CompareT<T>>& cast1,
                                   LocalTensor<CompareT<T>>& cast2, uint32_t length)
{
    const uint32_t castLength = VectorLength<T>(length);
    Cast(cast1, x1, RoundMode::CAST_NONE, castLength);
    Cast(cast2, x2, RoundMode::CAST_NONE, castLength);
    if constexpr (Less) {
        Compare(mask, cast1, cast2, CMPMODE::LT, VectorLength<CompareT<T>>(length));
    } else {
        Compare(mask, cast1, cast2, CMPMODE::GT, VectorLength<CompareT<T>>(length));
    }
}

template <bool Less, typename T>
__aicore__ inline void NativeCompare(LocalTensor<uint8_t>& mask, LocalTensor<T>& x1,
                                     LocalTensor<T>& x2, LocalTensor<uint8_t>& scratch,
                                     uint32_t length)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (Less) {
            Max(x2, x1, x2, length);
            Compare(mask, x2, x1, CMPMODE::EQ, length);
        } else {
            Max(x1, x1, x2, length);
            Compare(mask, x1, x2, CMPMODE::EQ, length);
        }
        LocalTensor<uint16_t> result = mask.ReinterpretCast<uint16_t>();
        Not(result, result, length / 16U);
    } else if constexpr (Less) {
        Compare(mask, x1, x2, CMPMODE::LT, length);
    } else {
        Compare(mask, x1, x2, CMPMODE::GT, length);
    }
}

template <bool Less>
__aicore__ inline void Int32Compare(LocalTensor<uint8_t>& mask, LocalTensor<int32_t>& x1,
                                    LocalTensor<int32_t>& x2, LocalTensor<int32_t>& maximum,
                                    LocalTensor<uint8_t>& scratch, uint32_t length)
{
    (void)scratch;
    Max(maximum, x1, x2, length);
    if constexpr (Less) Compare(mask, maximum, x1, CMPMODE::EQ, length);
    else Compare(mask, maximum, x2, CMPMODE::EQ, length);
    LocalTensor<uint16_t> result = mask.ReinterpretCast<uint16_t>();
    Not(result, result, length / 16U);
}

template <bool BroadcastIsX1>
__aicore__ inline void Int32CompareInPlace(LocalTensor<uint8_t>& mask,
                                           LocalTensor<int32_t>& full,
                                           LocalTensor<int32_t>& broadcast,
                                           uint32_t length)
{
    if constexpr (BroadcastIsX1) Min(full, full, broadcast, length);
    else Max(full, full, broadcast, length);
    Compare(mask, full, broadcast, CMPMODE::EQ, length);
    LocalTensor<uint16_t> result = mask.ReinterpretCast<uint16_t>();
    Not(result, result, length / 16U);
}

template <typename T, bool SingleOutput = false>
__aicore__ inline void RunLinear(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, uint32_t totalLength)
{
    if (totalLength == 0) {
        return;
    }
    const bool singleWorkerLaunch = totalLength <= kSingleWorkerElements && GetBlockNum() > 1U;
    const uint32_t physicalBlockIdx = GetBlockIdx();
    if (singleWorkerLaunch && physicalBlockIdx != 0U) {
        return;
    }
    const uint32_t blockUnit = 256U / sizeof(T);
    const uint32_t blocks = CeilDiv(totalLength, blockUnit);
    const uint32_t blockIdx = singleWorkerLaunch ? 0U : physicalBlockIdx;
    const uint32_t blockNum = singleWorkerLaunch ? 1U : GetBlockNum();
    const uint32_t start = blocks * blockIdx / blockNum * blockUnit;
    const uint32_t end = Min(totalLength, blocks * (blockIdx + 1U) / blockNum * blockUnit);
    if (start >= end) {
        return;
    }

    const uint32_t tile = LinearTileElements<T, SingleOutput>();
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, SingleOutput ? 1 : 2> outQueue;
    TBuf<QuePosition::VECCALC> inputGapBuf, maskBuf, cast1Buf, cast2Buf, constantsBuf;
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<uint8_t> outGm;
    x1Gm.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2);
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    if (totalLength > 1024U * 1024U / sizeof(T)) {
        x1Gm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        x2Gm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }
    pipe.InitBuffer(x1Queue, 2, tile * sizeof(T));
    if constexpr (std::is_same_v<T, float>) pipe.InitBuffer(inputGapBuf, 256U);
    pipe.InitBuffer(x2Queue, 2, tile * sizeof(T));
    pipe.InitBuffer(outQueue, SingleOutput ? 1 : 2, tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    if constexpr (kNeedsCast<T>) {
        pipe.InitBuffer(cast1Buf, tile * sizeof(CompareT<T>));
        pipe.InitBuffer(cast2Buf, tile * sizeof(CompareT<T>));
    }
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones, zeros;
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, float>) {
        pipe.InitBuffer(constantsBuf, 512U);
        ones = constantsBuf.Get<half>();
        zeros = ones[128];
        Duplicate(ones, static_cast<half>(1.0f), 128U);
        Duplicate(zeros, static_cast<half>(0.0f), 128U);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, int8_t>) {
            VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                        SELMODE::VSEL_TENSOR_TENSOR_MODE);
        }
    }

    const uint32_t tileCount = CeilDiv(end - start, tile);
    for (uint32_t step = 0; step < tileCount + 2U; ++step) {
        if constexpr (SingleOutput) {
            if (step > 1U) {
                const uint32_t offset = start + (step - 2U) * tile;
                const uint32_t length = Min(end - offset, tile);
                LocalTensor<uint8_t> out = outQueue.template DeQue<uint8_t>();
                CopyOut(outGm, out, offset, length);
                outQueue.FreeTensor(out);
            }
        }
        if (step < tileCount) {
            const uint32_t offset = start + step * tile;
            const uint32_t length = Min(end - offset, tile);
            LocalTensor<T> a = x1Queue.AllocTensor<T>();
            LocalTensor<T> b = x2Queue.AllocTensor<T>();
            CopyIn(a, x1Gm, offset, length);
            CopyIn(b, x2Gm, offset, length);
            x1Queue.EnQue(a);
            x2Queue.EnQue(b);
        }
        if (step > 0U && step <= tileCount) {
            const uint32_t offset = start + (step - 1U) * tile;
            const uint32_t length = Min(end - offset, tile);
            LocalTensor<T> a = x1Queue.DeQue<T>();
            LocalTensor<T> b = x2Queue.DeQue<T>();
            LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
            if constexpr (kNeedsCast<T>) {
                LocalTensor<CompareT<T>> cast1 = cast1Buf.Get<CompareT<T>>();
                LocalTensor<CompareT<T>> cast2 = cast2Buf.Get<CompareT<T>>();
                CastCompare<false>(mask, a, b, cast1, cast2, length);
                LocalTensor<half> values = cast1.template ReinterpretCast<half>();
                if constexpr (std::is_same_v<T, int8_t>) {
                    MaskToBoolFixedMask(values, out, ones, zeros,
                                        VectorLength<CompareT<T>>(length));
                } else {
                    MaskToBool(values, mask, out, VectorLength<CompareT<T>>(length));
                }
            } else {
                LocalTensor<half> values = a.template ReinterpretCast<half>();
                NativeCompare<false>(mask, a, b, out, VectorLength<T>(length));
                if constexpr (std::is_same_v<T, float>) {
                    MaskToBoolRepeatedConstants(values, mask, out, ones, zeros,
                                                VectorLength<T>(length));
                } else {
                    MaskToBool(values, mask, out, VectorLength<T>(length));
                }
            }
            outQueue.EnQue(out);
            x2Queue.FreeTensor(b);
            x1Queue.FreeTensor(a);
        }
        if constexpr (!SingleOutput) {
            if (step > 1U) {
                const uint32_t offset = start + (step - 2U) * tile;
                const uint32_t length = Min(end - offset, tile);
                LocalTensor<uint8_t> out = outQueue.template DeQue<uint8_t>();
                CopyOut(outGm, out, offset, length);
                outQueue.FreeTensor(out);
            }
        }
    }
}

template <typename T>
__aicore__ inline void RunLinearSimple(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, uint32_t totalLength)
{
    const uint32_t blockUnit = 256U / sizeof(T);
    const uint32_t blocks = CeilDiv(totalLength, blockUnit);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t start = blocks * blockIdx / blockNum * blockUnit;
    const uint32_t end = Min(totalLength, blocks * (blockIdx + 1U) / blockNum * blockUnit);
    if (start >= end) return;

    const uint32_t tile = TileElements<T>();
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> x1Queue, x2Queue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> maskBuf, valueBuf, cast1Buf, cast2Buf;
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<uint8_t> outGm;
    x1Gm.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2);
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    pipe.InitBuffer(x1Queue, 1, tile * sizeof(T));
    pipe.InitBuffer(x2Queue, 1, tile * sizeof(T));
    pipe.InitBuffer(outQueue, 1, tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(valueBuf, tile * sizeof(half));
    if constexpr (kNeedsCast<T>) {
        pipe.InitBuffer(cast1Buf, tile * sizeof(CompareT<T>));
        pipe.InitBuffer(cast2Buf, tile * sizeof(CompareT<T>));
    }
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> values = valueBuf.Get<half>();

    for (uint32_t offset = start; offset < end; offset += tile) {
        const uint32_t length = Min(end - offset, tile);
        LocalTensor<T> a = x1Queue.AllocTensor<T>();
        LocalTensor<T> b = x2Queue.AllocTensor<T>();
        CopyIn(a, x1Gm, offset, length);
        CopyIn(b, x2Gm, offset, length);
        x1Queue.EnQue(a);
        x2Queue.EnQue(b);
        a = x1Queue.DeQue<T>();
        b = x2Queue.DeQue<T>();
        LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
        if constexpr (kNeedsCast<T>) {
            LocalTensor<CompareT<T>> cast1 = cast1Buf.Get<CompareT<T>>();
            LocalTensor<CompareT<T>> cast2 = cast2Buf.Get<CompareT<T>>();
            CastCompare<false>(mask, a, b, cast1, cast2, length);
            MaskToBool(values, mask, out, VectorLength<CompareT<T>>(length));
        } else {
            NativeCompare<false>(mask, a, b, out, VectorLength<T>(length));
            MaskToBool(values, mask, out, VectorLength<T>(length));
        }
        outQueue.EnQue(out);
        x2Queue.FreeTensor(b);
        x1Queue.FreeTensor(a);
        out = outQueue.DeQue<uint8_t>();
        CopyOut(outGm, out, offset, length);
        outQueue.FreeTensor(out);
    }
}

__aicore__ inline void RunLargeInt8Linear(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                          uint32_t totalLength)
{
    constexpr uint32_t tile = 24064U;
    const uint32_t blocks = CeilDiv(totalLength, 256U);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t start = blocks * blockIdx / blockNum * 256U;
    const uint32_t end = Min(totalLength, blocks * (blockIdx + 1U) / blockNum * 256U);
    if (start >= end) return;

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> x1OutQueue;
    TQue<QuePosition::VECIN, 2> x2Queue;
    TBuf<QuePosition::VECCALC> maskBuf, cast1Buf, cast2Buf, constantsBuf;
    GlobalTensor<int8_t> x1Gm, x2Gm;
    GlobalTensor<uint8_t> outGm;
    x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1);
    x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2);
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    x1Gm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    x2Gm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    pipe.InitBuffer(x1OutQueue, 2, tile);
    pipe.InitBuffer(x2Queue, 2, tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(cast1Buf, tile * sizeof(half));
    pipe.InitBuffer(cast2Buf, tile * sizeof(half));
    pipe.InitBuffer(constantsBuf, 512U);

    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> cast1 = cast1Buf.Get<half>();
    LocalTensor<half> cast2 = cast2Buf.Get<half>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    PipeBarrier<PIPE_V>();
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                SELMODE::VSEL_TENSOR_TENSOR_MODE);

    const uint32_t tileCount = CeilDiv(end - start, tile);
    LocalTensor<int8_t> first = x1OutQueue.AllocTensor<int8_t>();
    LocalTensor<int8_t> second = x2Queue.AllocTensor<int8_t>();
    const uint32_t firstLength = Min(end - start, tile);
    CopyIn(first, x1Gm, start, firstLength);
    CopyIn(second, x2Gm, start, firstLength);
    x1OutQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, int8_t>(first);
    x2Queue.EnQue(second);

    for (uint32_t task = 0; task < tileCount; ++task) {
        const uint32_t offset = start + task * tile;
        const uint32_t length = Min(end - offset, tile);
        const uint32_t vectorLength = VectorLength<half>(length);
        LocalTensor<int8_t> a =
            x1OutQueue.template DeQue<QuePosition::GM, QuePosition::VECIN, int8_t>();
        LocalTensor<int8_t> b = x2Queue.DeQue<int8_t>();
        LocalTensor<uint8_t> out = a.ReinterpretCast<uint8_t>();

        if (task + 1U < tileCount) {
            const uint32_t nextOffset = offset + tile;
            const uint32_t nextLength = Min(end - nextOffset, tile);
            LocalTensor<int8_t> nextB = x2Queue.AllocTensor<int8_t>();
            CopyIn(nextB, x2Gm, nextOffset, nextLength);
            x2Queue.EnQue(nextB);
        }

        CastCompare<false>(mask, a, b, cast1, cast2, length);
        MaskToBoolFixedMask(cast1, out, ones, zeros, vectorLength);
        x1OutQueue.template EnQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>(out);
        x2Queue.FreeTensor(b);

        if (task + 1U < tileCount) {
            const uint32_t nextOffset = offset + tile;
            const uint32_t nextLength = Min(end - nextOffset, tile);
            LocalTensor<int8_t> nextA = x1OutQueue.AllocTensor<int8_t>();
            CopyIn(nextA, x1Gm, nextOffset, nextLength);
            x1OutQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, int8_t>(nextA);
        }

        out = x1OutQueue.template DeQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>();
        CopyOut(outGm, out, offset, length);
        x1OutQueue.FreeTensor(out);
    }
}

__aicore__ inline void RunLargeFp32Linear(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                          uint32_t totalLength)
{
    constexpr uint32_t tile = 11392U;
    constexpr uint32_t blockUnit = 64U;
    const uint32_t blocks = CeilDiv(totalLength, blockUnit);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const bool cyclic = totalLength >= blockNum * tile;
    const uint32_t start = cyclic ? blockIdx * tile : blocks * blockIdx / blockNum * blockUnit;
    const uint32_t end = cyclic ? totalLength :
        Min(totalLength, blocks * (blockIdx + 1U) / blockNum * blockUnit);
    const uint32_t step = (cyclic ? blockNum : 1U) * tile;
    if (start >= end) return;

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> x1OutQueues[2];
    TQue<QuePosition::VECIN, 2> x2Queue;
    TBuf<QuePosition::VECCALC> inputGapBuf, maskBuf, constantsBuf;
    GlobalTensor<float> x1Gm, x2Gm;
    GlobalTensor<uint8_t> outGm;
    x1Gm.SetGlobalBuffer((__gm__ float*)x1);
    x2Gm.SetGlobalBuffer((__gm__ float*)x2);
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    x1Gm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    x2Gm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    pipe.InitBuffer(x1OutQueues[0], 1, tile * sizeof(float));
    pipe.InitBuffer(x1OutQueues[1], 1, tile * sizeof(float));
    pipe.InitBuffer(inputGapBuf, 256U);
    pipe.InitBuffer(x2Queue, 2, tile * sizeof(float));
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(constantsBuf, 512U);

    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    PipeBarrier<PIPE_V>();

    const uint32_t tileCount = CeilDiv(end - start, step);
    LocalTensor<float> first = x1OutQueues[0].AllocTensor<float>();
    LocalTensor<float> second = x2Queue.AllocTensor<float>();
    const uint32_t firstLength = Min(end - start, tile);
    CopyIn(first, x1Gm, start, firstLength);
    CopyIn(second, x2Gm, start, firstLength);
    x1OutQueues[0].template EnQue<QuePosition::GM, QuePosition::VECIN, float>(first);
    x2Queue.EnQue(second);

    for (uint32_t task = 0U; task < tileCount; ++task) {
        auto& currentQueue = x1OutQueues[task & 1U];
        auto& nextQueue = x1OutQueues[(task + 1U) & 1U];
        const uint32_t offset = start + task * step;
        const uint32_t length = Min(end - offset, tile);
        const uint32_t vectorLength = VectorLength<float>(length);
        LocalTensor<float> a =
            currentQueue.template DeQue<QuePosition::GM, QuePosition::VECIN, float>();
        LocalTensor<float> b = x2Queue.DeQue<float>();
        LocalTensor<uint8_t> out = a.ReinterpretCast<uint8_t>();

        if (task + 1U < tileCount) {
            const uint32_t nextOffset = offset + step;
            const uint32_t nextLength = Min(end - nextOffset, tile);
            LocalTensor<float> next = x2Queue.AllocTensor<float>();
            CopyIn(next, x2Gm, nextOffset, nextLength);
            x2Queue.EnQue(next);
            LocalTensor<float> nextA = nextQueue.AllocTensor<float>();
            CopyIn(nextA, x1Gm, nextOffset, nextLength);
            nextQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, float>(nextA);
        }

        NativeCompare<false>(mask, a, b, out, vectorLength);
        LocalTensor<half> values = a.ReinterpretCast<half>();
        MaskToBoolRepeatedConstants(values, mask, out, ones, zeros, vectorLength);
        currentQueue.template EnQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>(out);
        x2Queue.FreeTensor(b);
        out = currentQueue.template DeQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>();
        CopyOut(outGm, out, offset, length);
        currentQueue.FreeTensor(out);
    }
}

template <typename T, bool sync = false>
__aicore__ inline void ExpandRows(LocalTensor<T>& local, uint32_t rowLength, uint32_t rows)
{
    const uint32_t target = rowLength * rows;
    const uint32_t vectorBlock = 256U / sizeof(T);
    CopyRepeatParams params = {1, 1, 8, 8};
    for (uint32_t filled = rowLength; filled < target;) {
        const uint32_t copyLength = Min(filled, target - filled);
        if (copyLength < vectorBlock) {
            Copy(local[filled], local, copyLength, 1, params);
        } else {
            const uint8_t repeats = static_cast<uint8_t>(CeilDiv(copyLength, vectorBlock));
            Copy(local[filled], local, vectorBlock, repeats, params);
        }
        if constexpr (sync) PipeBarrier<PIPE_V>();
        filled += copyLength;
    }
}

template <typename T>
__aicore__ inline void DuplicatePattern(LocalTensor<T>& local, uint32_t length)
{
    const uint32_t block = 256U / sizeof(T);
    const uint32_t main = length / block * block;
    CopyRepeatParams params = {1, 1, 8, 8};
    if (main != 0U) {
        Copy(local[length], local, block, static_cast<uint8_t>(main / block), params);
    }
    if (main != length) {
        Copy(local[length + main], local[main], length - main, 1, params);
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t Count, int32_t Step>
__aicore__ inline void BuildFastProgression(LocalTensor<int32_t>& values)
{
    ArithProgression(values, 0, Step, 64U);
#define EXPAND_PROGRESSION(N)                                                        \
    if constexpr (Count > N) {                                                      \
        constexpr uint32_t length = Count - N < N ? Count - N : N;                 \
        Adds(values[N], values, static_cast<int32_t>(N * Step),                     \
             static_cast<int32_t>(length));                                         \
        PipeBarrier<PIPE_V>();                                                       \
    }
    EXPAND_PROGRESSION(64U)
    EXPAND_PROGRESSION(128U)
    EXPAND_PROGRESSION(256U)
    EXPAND_PROGRESSION(512U)
    EXPAND_PROGRESSION(1024U)
    EXPAND_PROGRESSION(2048U)
    EXPAND_PROGRESSION(4096U)
    EXPAND_PROGRESSION(8192U)
#undef EXPAND_PROGRESSION
}

__aicore__ inline bool IsPowerOfTwo(uint32_t value)
{ return value != 0U && (value & (value - 1U)) == 0U; }

__aicore__ inline uint32_t PowerOfTwoShift(uint32_t value)
{
    uint32_t shift = 0U;
    while (value > 1U) {
        value >>= 1U;
        ++shift;
    }
    return shift;
}

__aicore__ inline void BuildGatherOffsets(LocalTensor<uint32_t>& offsets,
                                           LocalTensor<float> scratch,
                                           uint32_t start, float floatStart,
                                           uint32_t count, uint32_t low,
                                           uint32_t elementBytes, float inverseLow)
{
    LocalTensor<int32_t> signedOffsets = offsets.ReinterpretCast<int32_t>();
    if (start == 0U && count == 8192U && elementBytes == 4U && low >= 255U &&
        low < count && IsPowerOfTwo(low + 1U)) {
        const uint32_t shift = PowerOfTwoShift(low + 1U);
        LocalTensor<int32_t> folded = scratch.ReinterpretCast<int32_t>();
        BuildFastProgression<8192U, 1>(signedOffsets);
        ShiftRight(folded, signedOffsets, static_cast<int32_t>(shift), count);
        PipeBarrier<PIPE_V>();
        Muls(folded, folded, static_cast<int32_t>(low), count);
        PipeBarrier<PIPE_V>();
        Sub(signedOffsets, signedOffsets, folded, count);
        PipeBarrier<PIPE_V>();
        Adds(folded, signedOffsets, static_cast<int32_t>(1), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(folded, folded, static_cast<int32_t>(shift), count);
        PipeBarrier<PIPE_V>();
        Muls(folded, folded, static_cast<int32_t>(low), count);
        PipeBarrier<PIPE_V>();
        Sub(signedOffsets, signedOffsets, folded, count);
        PipeBarrier<PIPE_V>();
        Muls(signedOffsets, signedOffsets, static_cast<int32_t>(4), count);
        PipeBarrier<PIPE_V>();
        return;
    }
    constexpr uint32_t reciprocalShift = 28U;
    const uint32_t maxIndex = count == 0U ? 0U : count - 1U;
    const uint32_t reciprocal = low == 0U ? 0U :
        static_cast<uint32_t>(((1ULL << reciprocalShift) + low - 1U) / low);
    if (start == 0U && count <= 8192U && low >= 1024U &&
        static_cast<uint64_t>(maxIndex) * low < (1ULL << reciprocalShift) &&
        static_cast<uint64_t>(maxIndex) * reciprocal <= 0x7fffffffULL) {
        LocalTensor<int32_t> quotient = scratch.ReinterpretCast<int32_t>();
        ArithProgression(signedOffsets, 0, 1, count);
        PipeBarrier<PIPE_V>();
        Muls(quotient, signedOffsets, static_cast<int32_t>(reciprocal), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(quotient, quotient, static_cast<int32_t>(reciprocalShift), count);
        PipeBarrier<PIPE_V>();
        Muls(quotient, quotient, static_cast<int32_t>(low), count);
        PipeBarrier<PIPE_V>();
        Sub(signedOffsets, signedOffsets, quotient, count);
        PipeBarrier<PIPE_V>();
        Muls(signedOffsets, signedOffsets, static_cast<int32_t>(elementBytes), count);
        return;
    }
    ArithProgression(signedOffsets, static_cast<int32_t>(start * elementBytes),
                     static_cast<int32_t>(elementBytes), count);
    ArithProgression(scratch, floatStart, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, inverseLow, count);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> periods = scratch.ReinterpretCast<int32_t>();
    Cast(periods, scratch, RoundMode::CAST_FLOOR, count);
    PipeBarrier<PIPE_V>();
    Muls(periods, periods, static_cast<int32_t>(low * elementBytes), count);
    PipeBarrier<PIPE_V>();
    Sub(signedOffsets, signedOffsets, periods, count);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void RunSingleAxisScalar(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                            uint32_t totalLength, bool x1Broadcast)
{
    using C = CompareT<T>;
    const uint32_t blockUnit = 256U / sizeof(T);
    const uint32_t blocks = CeilDiv(totalLength, blockUnit);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t start = blocks * blockIdx / blockNum * blockUnit;
    const uint32_t end = Min(totalLength, blocks * (blockIdx + 1U) / blockNum * blockUnit);
    if (start >= end) return;

    const uint32_t tile = TileElements<T>();
    GlobalTensor<T> scalarGm, fullGm;
    GlobalTensor<uint8_t> outGm;
    scalarGm.SetGlobalBuffer((__gm__ T*)(x1Broadcast ? x1 : x2));
    fullGm.SetGlobalBuffer((__gm__ T*)(x1Broadcast ? x2 : x1));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    if (static_cast<uint64_t>(totalLength) * sizeof(T) > 1024U * 1024U) {
        fullGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> fullQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> scalarQueue;
    TBuf<QuePosition::VECCALC> castBuf, scalarCastBuf, maskBuf, valueBuf, onesBuf;
    TBuf<QuePosition::VECCALC> scalarVectorBuf, maximumBuf;
    pipe.InitBuffer(fullQueue, 2, tile * sizeof(T));
    pipe.InitBuffer(outQueue, 2, tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(valueBuf, tile * sizeof(half));
    pipe.InitBuffer(onesBuf, tile * sizeof(half));
    if constexpr (kNeedsCast<T>) {
        pipe.InitBuffer(scalarQueue, 1, kAlignBytes);
        pipe.InitBuffer(castBuf, tile * sizeof(C));
        pipe.InitBuffer(scalarCastBuf, kAlignBytes);
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        pipe.InitBuffer(scalarVectorBuf, tile * sizeof(int32_t));
        pipe.InitBuffer(maximumBuf, tile * sizeof(int32_t));
    }

    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> values = valueBuf.Get<half>();
    LocalTensor<half> ones = onesBuf.Get<half>();
    Duplicate(ones, static_cast<half>(1.0f), tile);
    C scalar;
    if constexpr (kNeedsCast<T>) {
        LocalTensor<T> source = scalarQueue.AllocTensor<T>();
        CopyIn(source, scalarGm, 0U, 1U);
        scalarQueue.EnQue(source);
        source = scalarQueue.DeQue<T>();
        LocalTensor<C> converted = scalarCastBuf.Get<C>();
        Cast(converted, source, RoundMode::CAST_NONE, VectorLength<T>(1U));
        TEventID vectorToScalar = pipe.FetchEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        scalar = converted.GetValue(0);
        pipe.ReleaseEventID<HardEvent::V_S>(vectorToScalar);
        scalarQueue.FreeTensor(source);
    } else {
        scalar = static_cast<C>(scalarGm.GetValue(0));
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        Duplicate(scalarVectorBuf.Get<int32_t>(), scalar, tile);
    }

    const uint32_t tileCount = CeilDiv(end - start, tile);
    for (uint32_t step = 0; step < tileCount + 2U; ++step) {
        if (step < tileCount) {
            const uint32_t offset = start + step * tile;
            const uint32_t length = Min(end - offset, tile);
            LocalTensor<T> full = fullQueue.AllocTensor<T>();
            CopyIn(full, fullGm, offset, length);
            fullQueue.EnQue(full);
        }
        if (step > 0U && step <= tileCount) {
            const uint32_t offset = start + (step - 1U) * tile;
            const uint32_t length = Min(end - offset, tile);
            const uint32_t vectorLength = VectorLength<C>(length);
            LocalTensor<T> full = fullQueue.DeQue<T>();
            LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
            if constexpr (kNeedsCast<T>) {
                LocalTensor<C> converted = castBuf.Get<C>();
                Cast(converted, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                CompareScalar(mask, converted, scalar,
                              x1Broadcast ? CMPMODE::LT : CMPMODE::GT, vectorLength);
            } else if constexpr (std::is_same_v<T, int32_t>) {
                LocalTensor<int32_t> scalarVector = scalarVectorBuf.Get<int32_t>();
                LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                if (x1Broadcast) Int32Compare<true>(mask, full, scalarVector, maximum, out, vectorLength);
                else Int32Compare<false>(mask, full, scalarVector, maximum, out, vectorLength);
            } else {
                CompareScalar(mask, full, scalar,
                              x1Broadcast ? CMPMODE::LT : CMPMODE::GT, vectorLength);
            }
            MaskToBool(values, mask, out, ones, vectorLength);
            outQueue.EnQue(out);
            fullQueue.FreeTensor(full);
        }
        if (step > 1U) {
            const uint32_t offset = start + (step - 2U) * tile;
            const uint32_t length = Min(end - offset, tile);
            LocalTensor<uint8_t> out = outQueue.DeQue<uint8_t>();
            CopyOut(outGm, out, offset, length);
            outQueue.FreeTensor(out);
        }
    }
}

__aicore__ inline void RunSingleAxisSmallHigh(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                               const GreaterTilingData& tiling)
{
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t high = tiling.highSize;
    const bool x1Broadcast = tiling.broadcastIsX1 != 0U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t base = high / blockNum;
    const uint32_t extra = high % blockNum;
    const uint32_t myHigh = base + (blockIdx < extra ? 1U : 0U);
    const uint32_t highStart = blockIdx * base + Min(blockIdx, extra);
    if (myHigh == 0U) return;

    const uint32_t rowStride = VectorLength<float>(low);
    const uint32_t outStride = VectorLength<uint8_t>(low);
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> broadcastQueue, fullQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> maskBuf, constantsBuf;
    GlobalTensor<float> broadcastGm, fullGm;
    GlobalTensor<uint8_t> outGm;
    broadcastGm.SetGlobalBuffer((__gm__ float*)(x1Broadcast ? x1 : x2));
    fullGm.SetGlobalBuffer((__gm__ float*)(x1Broadcast ? x2 : x1));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    pipe.InitBuffer(broadcastQueue, 2, rowStride * sizeof(float));
    pipe.InitBuffer(fullQueue, 2, middle * rowStride * sizeof(float));
    pipe.InitBuffer(outQueue, 2, middle * outStride);
    pipe.InitBuffer(maskBuf, MaskBytes(rowStride));
    pipe.InitBuffer(constantsBuf, 512U);
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    PipeBarrier<PIPE_V>();

    for (uint32_t step = 0; step < myHigh + 2U; ++step) {
        if (step < myHigh) {
            const uint32_t h = highStart + step;
            LocalTensor<float> broadcast = broadcastQueue.AllocTensor<float>();
            LocalTensor<float> full = fullQueue.AllocTensor<float>();
            CopyIn(broadcast, broadcastGm, h * low, low);
            CopyInBatch(full, fullGm, h * middle * low, low, rowStride,
                        static_cast<uint16_t>(middle));
            broadcastQueue.EnQue(broadcast);
            fullQueue.EnQue(full);
        }
        if (step > 0U && step <= myHigh) {
            LocalTensor<float> broadcast = broadcastQueue.DeQue<float>();
            LocalTensor<float> full = fullQueue.DeQue<float>();
            LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
            for (uint32_t row = 0; row < middle; ++row) {
                LocalTensor<float> fullRow = full[row * rowStride];
                LocalTensor<uint8_t> outRow = out[row * outStride];
                if (x1Broadcast) NativeCompare<true>(mask, fullRow, broadcast, outRow, rowStride);
                else NativeCompare<false>(mask, fullRow, broadcast, outRow, rowStride);
                LocalTensor<half> values = fullRow.ReinterpretCast<half>();
                MaskToBoolRepeatedConstants(values, mask, outRow, ones, zeros, rowStride);
            }
            outQueue.EnQue(out);
            fullQueue.FreeTensor(full);
            broadcastQueue.FreeTensor(broadcast);
        }
        if (step > 1U) {
            const uint32_t h = highStart + step - 2U;
            LocalTensor<uint8_t> out = outQueue.DeQue<uint8_t>();
            CopyOutBatch(outGm, out, h * middle * low, low, outStride,
                         static_cast<uint16_t>(middle));
            outQueue.FreeTensor(out);
        }
    }
}

__aicore__ inline bool CanUseBf16BatchedRows(uint32_t totalLength, uint32_t low,
                                             uint32_t middle, uint32_t high)
{
    if (low == 0U || high == 0U || middle < kBf16RowsPerTile ||
        middle % kBf16RowsPerTile != 0U || low * sizeof(bfloat16_t) % kAlignBytes == 0U ||
        static_cast<uint64_t>(totalLength) * sizeof(bfloat16_t) <= kLargeStreamBytes) {
        return false;
    }
    const uint32_t tileLength = kBf16RowsPerTile * low;
    const uint32_t fullStorage = VectorLength<bfloat16_t>(tileLength);
    const uint32_t compareLength = VectorLength<float>(tileLength);
    const uint32_t broadcastStorage = VectorLength<bfloat16_t>(low);
    const uint64_t ubBytes = 12ULL * fullStorage + 8ULL * compareLength +
                             4ULL * broadcastStorage + MaskBytes(compareLength) + 768U;
    return ubBytes <= kBf16UbBudget;
}

template <typename T>
__aicore__ inline void RepeatMaskedRows32(const LocalTensor<T>& dst,
                                         const LocalTensor<T>& src,
                                         const LocalTensor<uint32_t>& pattern,
                                         uint32_t width, uint16_t rows)
{
    static_assert(sizeof(T) == sizeof(uint32_t));

    SetMaskCount();
    set_vector_mask(0, width);
    GatherMaskImpl((__ubuf__ uint32_t*)dst.GetPhyAddr(),
                   (__ubuf__ uint32_t*)src.GetPhyAddr(),
                   (__ubuf__ uint32_t*)pattern.GetPhyAddr(), 0U,
                   {1U, rows, 0U, 0U});
    SetMaskNorm();
}

template <bool X1Broadcast>
__aicore__ inline bool TryBf16HighCache(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                      const GreaterTilingData& tiling, float inverseLow)
{
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t high = tiling.highSize;
    const uint32_t block = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t maxHighs = CeilDiv(high, cores);
    const uint32_t tile = kBf16RowsPerTile * low;
    const uint32_t storage = VectorLength<bfloat16_t>(tile);
    const uint32_t count = VectorLength<float>(tile);
    const uint32_t cachedCount = VectorLength<float>(maxHighs * low);
    const uint64_t ubBytes = 6ULL * storage + 12ULL * count +
                             4ULL * cachedCount + MaskBytes(count) + 64U;
    if (maxHighs < 2U || ubBytes > kBf16UbBudget || cachedCount > 2U * count) return false;
    const uint32_t highStart = block * (high / cores) + Min(block, high % cores);
    const uint32_t myHighs = high / cores + (block < high % cores);
    if (myHighs == 0U) return true;

    GlobalTensor<bfloat16_t> fullGm, broadcastGm;
    GlobalTensor<uint8_t> outGm;
    fullGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x2 : x1));
    broadcastGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x1 : x2));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    fullGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    broadcastGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> fullQueue;
    TQue<QuePosition::VECIN, 1> castQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> cachedBuf, repeatedBuf, offsetsBuf, maskBuf, constantsBuf;
    pipe.InitBuffer(fullQueue, 2, storage * sizeof(bfloat16_t));
    pipe.InitBuffer(outQueue, 2, storage);
    pipe.InitBuffer(castQueue, 1, count * sizeof(float));
    pipe.InitBuffer(cachedBuf, cachedCount * sizeof(float));
    pipe.InitBuffer(repeatedBuf, count * sizeof(float));
    pipe.InitBuffer(offsetsBuf, count * sizeof(uint32_t));
    pipe.InitBuffer(maskBuf, MaskBytes(count));
    pipe.InitBuffer(constantsBuf, 64U);
    LocalTensor<float> cached = cachedBuf.Get<float>();
    LocalTensor<float> repeated = repeatedBuf.Get<float>();
    LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[16];


    LocalTensor<bfloat16_t> raw = castQueue.AllocTensor<bfloat16_t>();
    CopyIn(raw, broadcastGm, highStart * low, myHighs * low);
    castQueue.EnQue(raw);
    LocalTensor<bfloat16_t> first = fullQueue.AllocTensor<bfloat16_t>();
    const uint32_t start = highStart * middle * low;
    CopyIn(first, fullGm, start, tile);
    fullQueue.EnQue(first);
    Duplicate(ones, static_cast<half>(1.0f), 16U);
    Duplicate(zeros, static_cast<half>(0.0f), 16U);
    BuildGatherOffsets(offsets, repeated, 0U, 0.0f, count, low, sizeof(float), inverseLow);
    raw = castQueue.DeQue<bfloat16_t>();
    Cast(cached, raw, RoundMode::CAST_NONE, VectorLength<float>(myHighs * low));
    castQueue.FreeTensor(raw);
    LocalTensor<float> fullValue = castQueue.AllocTensor<float>();
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                SELMODE::VSEL_TENSOR_TENSOR_MODE);
    const uint32_t tilesPerHigh = middle / kBf16RowsPerTile;
    const uint32_t tasks = myHighs * tilesPerHigh;
    uint32_t refreshTask = 0U;
    uint32_t cachedOffset = 0U;
    BinaryRepeatParams selectParams(1, 0, 0, 8, 0, 0);
    for (uint32_t task = 0U; task <= tasks; ++task) {
        if (task + 1U < tasks) {
            LocalTensor<bfloat16_t> next = fullQueue.AllocTensor<bfloat16_t>();
            CopyIn(next, fullGm, start + (task + 1U) * tile, tile);
            fullQueue.EnQue(next);
        }
        if (task != 0U) {
            LocalTensor<uint8_t> completed = outQueue.DeQue<uint8_t>();
            CopyOut(outGm, completed, start + (task - 1U) * tile, tile);
            outQueue.FreeTensor(completed);
        }
        if (task == tasks) break;
        if (task == refreshTask) {
            Gather(repeated, cached, offsets, cachedOffset, count);
            cachedOffset += low * sizeof(float);
            refreshTask += tilesPerHigh;
        }
        LocalTensor<bfloat16_t> full = fullQueue.DeQue<bfloat16_t>();
        LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
        Cast(fullValue, full, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, fullValue, repeated, X1Broadcast ? CMPMODE::LT : CMPMODE::GT, count);
        LocalTensor<half> values = full.ReinterpretCast<half>();
        const uint8_t repeats = static_cast<uint8_t>(CeilDiv(count, 128U));
        Select<half, SELMODE::VSEL_TENSOR_TENSOR_MODE>(values, ones, zeros, repeats, selectParams);
        PipeBarrier<PIPE_V>();
        Cast(out, values, RoundMode::CAST_TRUNC, repeats * 128U);
        outQueue.EnQue(out);
        fullQueue.FreeTensor(full);
    }
    castQueue.FreeTensor(fullValue);
    return true;
}

template <bool X1Broadcast>
__aicore__ inline bool TryBf16SixMask(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                    const GreaterTilingData& tiling)
{
    constexpr uint32_t rows = 6U;
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t high = tiling.highSize;
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t maxHighs = CeilDiv(high, cores);
    const uint32_t tile = rows * low;
    const uint32_t storage = VectorLength<bfloat16_t>(tile);
    const uint32_t count = VectorLength<float>(tile);
    const uint32_t padded = CeilDiv(low + 15U, 256U) * 256U;
    const uint32_t cacheCount = VectorLength<bfloat16_t>(maxHighs * low + padded - low);
    const uint32_t patternBytes = MaskBytes(padded);
    const uint64_t ubBytes = 4ULL * storage + 8ULL * count + 2ULL * cacheCount + 4ULL * padded +
                            MaskBytes(count) + 16ULL * patternBytes + 512U;
    if (middle % rows != 0U || low < 1024U || count > 16384U ||
        maxHighs * low > 2U * count || ubBytes > kBf16UbBudget) return false;
    const uint32_t firstHigh = core * (high / cores) + Min(core, high % cores);
    const uint32_t myHighs = high / cores + (core < high % cores);
    if (myHighs == 0U) return true;
    const uint32_t tasksPerHigh = middle / rows;
    const uint32_t tasks = myHighs * tasksPerHigh;
    const uint32_t start = firstHigh * middle * low;

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> slots[2];
    TQue<QuePosition::VECIN, 1> cachedQueue;
    TBuf<QuePosition::VECCALC> fullValueBuf, rowValueBuf, repeatedBuf, maskBuf, patternBuf, constantsBuf;
    pipe.InitBuffer(slots[0], 1, storage * sizeof(bfloat16_t));
    pipe.InitBuffer(slots[1], 1, storage * sizeof(bfloat16_t));
    pipe.InitBuffer(cachedQueue, 1, cacheCount * sizeof(bfloat16_t));
    pipe.InitBuffer(fullValueBuf, count * sizeof(float));
    pipe.InitBuffer(rowValueBuf, padded * sizeof(float));
    pipe.InitBuffer(repeatedBuf, count * sizeof(float));
    pipe.InitBuffer(maskBuf, MaskBytes(count));
    pipe.InitBuffer(patternBuf, 16U * patternBytes);
    pipe.InitBuffer(constantsBuf, 512U);
    GlobalTensor<bfloat16_t> fullGm, broadcastGm;
    GlobalTensor<uint8_t> outGm;
    fullGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x2 : x1));
    broadcastGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x1 : x2));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    fullGm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    broadcastGm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<float> rowValue = rowValueBuf.Get<float>();
    LocalTensor<float> repeated = repeatedBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> patterns = patternBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    LocalTensor<bfloat16_t> first = slots[0].AllocTensor<bfloat16_t>();
    CopyIn(first, fullGm, start, tile);
    slots[0].template EnQue<QuePosition::GM, QuePosition::VECIN, bfloat16_t>(first);
    LocalTensor<bfloat16_t> cached = cachedQueue.AllocTensor<bfloat16_t>();
    CopyIn(cached, broadcastGm, firstHigh * low, myHighs * low);
    cachedQueue.EnQue(cached);
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    ArithProgression(repeated, 0.0f, 1.0f, padded);
    PipeBarrier<PIPE_V>();
    for (uint32_t origin = 0U; origin < 16U; ++origin) {
        LocalTensor<uint8_t> pattern = patterns[origin * patternBytes];
        CompareScalar(pattern, repeated, static_cast<float>(static_cast<int32_t>(origin + low)), CMPMODE::LT, padded);
        CompareScalar(mask, repeated, static_cast<float>(static_cast<int32_t>(origin)), CMPMODE::GE, padded);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint16_t> dst = pattern.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> valid = mask.ReinterpretCast<uint16_t>();
        And(dst, dst, valid, padded / 16U);
        PipeBarrier<PIPE_V>();
    }
    cached = cachedQueue.DeQue<bfloat16_t>();
    LocalTensor<float> fullValue = fullValueBuf.Get<float>();
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                SELMODE::VSEL_TENSOR_TENSOR_MODE);
    uint32_t refresh = 0U;
    uint32_t cachedOffset = 0U;
    for (uint32_t task = 0U; task < tasks; ++task) {
        auto& current = slots[task & 1U];
        auto& nextSlot = slots[(task + 1U) & 1U];
        if (task + 1U < tasks) {
            LocalTensor<bfloat16_t> next = nextSlot.AllocTensor<bfloat16_t>();
            CopyIn(next, fullGm, start + (task + 1U) * tile, tile);
            nextSlot.template EnQue<QuePosition::GM, QuePosition::VECIN, bfloat16_t>(next);
        }
        if (task == refresh) {
            const uint32_t origin = cachedOffset % 16U;
            LocalTensor<bfloat16_t> source = cached[cachedOffset - origin];
            Cast(rowValue, source, RoundMode::CAST_NONE, padded);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> pattern = patterns[origin * patternBytes].ReinterpretCast<uint32_t>();
            RepeatMaskedRows32(repeated, rowValue, pattern, padded, static_cast<uint16_t>(rows));
            PipeBarrier<PIPE_V>();
            cachedOffset += low;
            refresh += tasksPerHigh;
        }
        LocalTensor<bfloat16_t> full = current.template DeQue<QuePosition::GM, QuePosition::VECIN, bfloat16_t>();
        Cast(fullValue, full, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, fullValue, repeated, X1Broadcast ? CMPMODE::LT : CMPMODE::GT, count);
        LocalTensor<half> values = full.ReinterpretCast<half>();
        LocalTensor<uint8_t> out = full.ReinterpretCast<uint8_t>();
        MaskToBoolFixedMask(values, out, ones, zeros, count);
        current.template EnQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>(out);
        out = current.template DeQue<QuePosition::VECOUT, QuePosition::GM, uint8_t>();
        CopyOut(outGm, out, start + task * tile, tile);
        current.FreeTensor(out);
    }
    cachedQueue.FreeTensor(cached);
    return true;
}


template <bool X1Broadcast, bool SingleTile>
__aicore__ inline void RunBf16BatchedRows(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                           const GreaterTilingData& tiling, float inverseLow)
{
    if (TryBf16SixMask<X1Broadcast>(x1, x2, output, tiling)) return;
    if (TryBf16HighCache<X1Broadcast>(x1, x2, output, tiling, inverseLow)) return;
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t high = tiling.highSize;
    const uint32_t tileLength = kBf16RowsPerTile * low;
    const uint32_t fullStorage = VectorLength<bfloat16_t>(tileLength);
    const uint32_t compareLength = VectorLength<float>(tileLength);
    const uint32_t broadcastStorage = VectorLength<bfloat16_t>(low);
    const uint32_t tilesPerBroadcast = middle / kBf16RowsPerTile;

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t highsPerCore = high / blockNum;
    const uint32_t extraCores = high % blockNum;
    const bool denseExtras = 2U * extraCores > blockNum;
    const uint32_t extraBefore = denseExtras ? Min(blockIdx, extraCores) :
                                                Min((blockIdx + 1U) / 2U, extraCores);
    const uint32_t highStart = blockIdx * highsPerCore + extraBefore;
    const uint32_t highCount = highsPerCore +
        (denseExtras ? blockIdx < extraCores :
         (blockIdx < 2U * extraCores && (blockIdx & 1U) == 0U));
    const uint32_t rowStart = highStart * middle;
    const uint32_t rowEnd = rowStart + highCount * middle;
    if (rowStart >= rowEnd) return;

    GlobalTensor<bfloat16_t> fullGm, broadcastGm;
    GlobalTensor<uint8_t> outGm;
    fullGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x2 : x1));
    broadcastGm.SetGlobalBuffer((__gm__ bfloat16_t*)(X1Broadcast ? x1 : x2));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    fullGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    broadcastGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> fullQueue, broadcastQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> maskBuf, constantsBuf, repeatedBuf, offsetsBuf, compareGapBuf;
    TBuf<QuePosition::VECCALC> fullCastBuf, repeatedBf16Buf;
    pipe.InitBuffer(fullQueue, 2, fullStorage * sizeof(bfloat16_t));
    pipe.InitBuffer(broadcastQueue, 2, broadcastStorage * sizeof(bfloat16_t));
    pipe.InitBuffer(outQueue, 2, fullStorage);
    pipe.InitBuffer(maskBuf, MaskBytes(compareLength));
    pipe.InitBuffer(constantsBuf, 512U);
    pipe.InitBuffer(repeatedBuf, compareLength * sizeof(float));
    pipe.InitBuffer(offsetsBuf, compareLength * sizeof(uint32_t));
    pipe.InitBuffer(compareGapBuf, 256U);
    pipe.InitBuffer(fullCastBuf, fullStorage * sizeof(float));
    pipe.InitBuffer(repeatedBf16Buf, fullStorage * sizeof(bfloat16_t));

    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    LocalTensor<float> repeated = repeatedBuf.Get<float>();
    LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
    LocalTensor<float> fullValue = fullCastBuf.Get<float>();
    LocalTensor<bfloat16_t> repeatedBf16 = repeatedBf16Buf.Get<bfloat16_t>();
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                SELMODE::VSEL_TENSOR_TENSOR_MODE);

    if constexpr (SingleTile) {
        Duplicate(ones, static_cast<half>(1.0f), 128U);
        Duplicate(zeros, static_cast<half>(0.0f), 128U);
        BuildGatherOffsets(offsets, repeated, 0U, 0.0f, compareLength,
                           low, sizeof(bfloat16_t), inverseLow);
        for (uint32_t h = highStart; h < highStart + highCount; ++h) {
            LocalTensor<bfloat16_t> broadcast = broadcastQueue.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> full = fullQueue.AllocTensor<bfloat16_t>();
            CopyIn(broadcast, broadcastGm, h * low, low);
            CopyIn(full, fullGm, h * tileLength, tileLength);
            broadcastQueue.EnQue(broadcast);
            fullQueue.EnQue(full);
            broadcast = broadcastQueue.DeQue<bfloat16_t>();
            full = fullQueue.DeQue<bfloat16_t>();
            Gather(repeatedBf16, broadcast, offsets, 0U, compareLength);
            PipeBarrier<PIPE_V>();
            Cast(repeated, repeatedBf16, RoundMode::CAST_NONE, compareLength);
            broadcastQueue.FreeTensor(broadcast);

            LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
            Cast(fullValue, full, RoundMode::CAST_NONE, compareLength);
            PipeBarrier<PIPE_V>();
            Compare(mask, fullValue, repeated,
                    X1Broadcast ? CMPMODE::LT : CMPMODE::GT, compareLength);
            LocalTensor<half> reusedValues = full.ReinterpretCast<half>();
            MaskToBoolFixedMask(reusedValues, out, ones, zeros, compareLength);
            outQueue.EnQue(out);
            fullQueue.FreeTensor(full);
            out = outQueue.DeQue<uint8_t>();
            CopyOut(outGm, out, h * tileLength, tileLength);
            outQueue.FreeTensor(out);
        }
        return;
    }

    LocalTensor<bfloat16_t> broadcast = broadcastQueue.AllocTensor<bfloat16_t>();
    CopyIn(broadcast, broadcastGm, rowStart / middle * low, low);
    broadcastQueue.EnQue(broadcast);
    LocalTensor<bfloat16_t> firstFull = fullQueue.AllocTensor<bfloat16_t>();
    CopyIn(firstFull, fullGm, rowStart * low, tileLength);
    fullQueue.EnQue(firstFull);

    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    BuildGatherOffsets(offsets, repeated, 0U, 0.0f, compareLength,
                       low, sizeof(bfloat16_t), inverseLow);

    broadcast = broadcastQueue.DeQue<bfloat16_t>();
    Gather(repeatedBf16, broadcast, offsets, 0U, compareLength);
    PipeBarrier<PIPE_V>();
    Cast(repeated, repeatedBf16, RoundMode::CAST_NONE, compareLength);
    broadcastQueue.FreeTensor(broadcast);

    const uint32_t tileCount = (rowEnd - rowStart) / kBf16RowsPerTile;
    uint32_t prefetchTask = tilesPerBroadcast - 1U;
    uint32_t refreshTask = tilesPerBroadcast;
    uint32_t nextBroadcastHigh = highStart + 1U;
    for (uint32_t task = 0U; task <= tileCount; ++task) {
        if (task + 1U < tileCount) {
            const uint32_t nextTask = task + 1U;
            const uint32_t nextRow = rowStart + nextTask * kBf16RowsPerTile;
            LocalTensor<bfloat16_t> nextFull = fullQueue.AllocTensor<bfloat16_t>();
            CopyIn(nextFull, fullGm, nextRow * low, tileLength);
            fullQueue.EnQue(nextFull);
            if (nextTask == prefetchTask && nextTask + 1U < tileCount) {
                LocalTensor<bfloat16_t> nextBroadcast =
                    broadcastQueue.AllocTensor<bfloat16_t>();
                CopyIn(nextBroadcast, broadcastGm, nextBroadcastHigh * low, low);
                broadcastQueue.EnQue(nextBroadcast);
                ++nextBroadcastHigh;
                prefetchTask += tilesPerBroadcast;
            }
        }

        if (task != 0U) {
            const uint32_t completedRow = rowStart + (task - 1U) * kBf16RowsPerTile;
            LocalTensor<uint8_t> completed = outQueue.DeQue<uint8_t>();
            CopyOut(outGm, completed, completedRow * low, tileLength);
            outQueue.FreeTensor(completed);
        }
        if (task == tileCount) break;

        const bool refreshBroadcast = task == refreshTask;
        if (refreshBroadcast) {
            LocalTensor<bfloat16_t> nextBroadcast =
                broadcastQueue.DeQue<bfloat16_t>();
            Gather(repeatedBf16, nextBroadcast, offsets, 0U, compareLength);
            PipeBarrier<PIPE_V>();
            Cast(repeated, repeatedBf16, RoundMode::CAST_NONE, compareLength);
            broadcastQueue.FreeTensor(nextBroadcast);
            refreshTask += tilesPerBroadcast;
        }

        LocalTensor<bfloat16_t> full = fullQueue.DeQue<bfloat16_t>();
        LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
        Cast(fullValue, full, RoundMode::CAST_NONE, compareLength);
        if (task == 0U || refreshBroadcast) PipeBarrier<PIPE_V>();
        Compare(mask, fullValue, repeated,
                X1Broadcast ? CMPMODE::LT : CMPMODE::GT, compareLength);
        LocalTensor<half> reusedValues = full.ReinterpretCast<half>();
        MaskToBoolFixedMask(reusedValues, out, ones, zeros, compareLength);
        outQueue.EnQue(out);
        fullQueue.FreeTensor(full);
    }
}

template <typename T>
__aicore__ inline bool CanUseInt32Batch3(uint32_t totalLength, uint32_t low,
                                         uint32_t middle, uint32_t high)
{
    if constexpr (!std::is_same_v<T, int32_t>) return false;
    if (low == 0U || middle == 0U || high == 0U ||
        low * sizeof(int32_t) % kAlignBytes == 0U ||
        static_cast<uint64_t>(totalLength) * sizeof(int32_t) <= kLargeStreamBytes) {
        return false;
    }
    constexpr uint32_t tile = 8192U;
    const uint32_t rowsPerTile = tile / low;
    uint32_t gcd = low;
    uint32_t divisor = kAlignBytes;
    while (divisor != 0U) {
        const uint32_t next = gcd % divisor;
        gcd = divisor;
        divisor = next;
    }
    const uint32_t rowGroup = kAlignBytes / gcd;
    return rowsPerTile != 0U && middle >= 3U * rowsPerTile &&
        middle % rowsPerTile == 0U && rowGroup % rowsPerTile == 0U &&
        VectorLength<int32_t>(rowsPerTile * low) == tile;
}

__aicore__ inline void BuildFlatOffsets(LocalTensor<uint32_t>& offsets,
                                       LocalTensor<int32_t>& scratch, uint32_t low)
{
    constexpr uint32_t count = 9216U;
    LocalTensor<int32_t> index = offsets.ReinterpretCast<int32_t>();
    const uint32_t reciprocal = CeilDiv(1U << 24U, low);
    BuildFastProgression<count, 1>(index);
    Muls(scratch, index, static_cast<int32_t>(reciprocal), count);
    PipeBarrier<PIPE_V>();
    ShiftRight(scratch, scratch, static_cast<int32_t>(24), count);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, static_cast<int32_t>(low), count);
    PipeBarrier<PIPE_V>();
    Sub(index, index, scratch, count);
    PipeBarrier<PIPE_V>();
    Muls(index, index, static_cast<int32_t>(sizeof(int32_t)), count);
    PipeBarrier<PIPE_V>();
}

template <bool X1Broadcast>
__aicore__ inline void RunInt32FlatStream(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                        const GreaterTilingData& tiling)
{
    constexpr uint32_t tile = 8192U;
    const uint32_t low = tiling.lowSize;
    const uint32_t period = low * tiling.middleSize;
    const uint32_t paddedWidth = CeilDiv(low + 7U, 256U) * 256U;
    const uint32_t patternBytes = MaskBytes(paddedWidth);
    const uint32_t patternRows = CeilDiv(tile + low, low);
    const uint32_t repeatedCount = VectorLength<int32_t>(patternRows * low);
    const uint32_t broadcastCount = VectorLength<int32_t>(low + paddedWidth);
    const uint32_t circularCount = VectorLength<int32_t>(2U * low);
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t groups = CeilDiv(tiling.totalLength, 3U * tile);
    if (core >= groups) return;
    const uint32_t myGroups = groups / cores + ((core + groups / cores * 3U) % cores < groups % cores);
    const uint32_t lastGroup = (myGroups - 1U) * cores + (core + (myGroups - 1U) * 3U) % cores;
    const uint32_t lastTiles = Min(3U, CeilDiv(tiling.totalLength - lastGroup * 3U * tile, tile));
    const uint32_t begin = core * 3U * tile;
    const uint32_t end = tiling.totalLength;
    if (begin >= end) return;

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;
    TQue<QuePosition::VECIN, 1> broadcastQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> repeatedBuf, maskBuf, constantsBuf, circularBuf, patternBuf;
    pipe.InitBuffer(inputQueue, 2, tile * sizeof(int32_t));
    pipe.InitBuffer(broadcastQueue, 1, broadcastCount * sizeof(int32_t));
    pipe.InitBuffer(outputQueue, 1, 3U * tile);
    pipe.InitBuffer(repeatedBuf, repeatedCount * sizeof(int32_t));
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(constantsBuf, 512U);
    pipe.InitBuffer(circularBuf, circularCount * sizeof(int32_t));
    pipe.InitBuffer(patternBuf, 8U * patternBytes);
    GlobalTensor<int32_t> fullGm, broadcastGm;
    GlobalTensor<uint8_t> outGm;
    fullGm.SetGlobalBuffer((__gm__ int32_t*)(X1Broadcast ? x2 : x1));
    broadcastGm.SetGlobalBuffer((__gm__ int32_t*)(X1Broadcast ? x1 : x2));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    fullGm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<int32_t> repeated = repeatedBuf.Get<int32_t>();
    LocalTensor<int32_t> circular = circularBuf.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    PipeBarrier<PIPE_V>();
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                              SELMODE::VSEL_TENSOR_TENSOR_MODE);
    LocalTensor<int32_t> first = inputQueue.AllocTensor<int32_t>();
    CopyIn(first, fullGm, begin, Min(tile, end - begin));
    inputQueue.EnQue(first);
    LocalTensor<float> index = repeated.ReinterpretCast<float>();
    ArithProgression(index, 0.0f, 1.0f, paddedWidth);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> patterns = patternBuf.Get<uint8_t>();
    for (uint32_t origin = 0U; origin < 8U; ++origin) {
        LocalTensor<uint8_t> pattern = patterns[origin * patternBytes];
        CompareScalar(pattern, index, static_cast<float>(static_cast<int32_t>(origin + low)), CMPMODE::LT, paddedWidth);
        CompareScalar(mask, index, static_cast<float>(static_cast<int32_t>(origin)), CMPMODE::GE, paddedWidth);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint16_t> dst = pattern.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> valid = mask.ReinterpretCast<uint16_t>();
        And(dst, dst, valid, paddedWidth / 16U);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint32_t> firstPattern = patterns.ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> nextPattern = patterns[(low % 8U) * patternBytes].ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> broadcast;
    uint32_t previousHigh = UINT32_MAX;
    uint32_t previousOrigin = UINT32_MAX;
    const uint32_t tasks = (myGroups - 1U) * 3U + lastTiles;
    LocalTensor<uint8_t> batch;
    for (uint32_t task = 0U; task <= tasks; ++task) {
        if (task != 0U && (task % 3U == 0U || task == tasks)) {
            LocalTensor<uint8_t> out = outputQueue.DeQue<uint8_t>();
            const uint32_t completed = task % 3U == 0U ? 3U : task % 3U;
            const uint32_t offset = ((task - completed) / 3U * cores + (core + (task - completed) / 3U * 3U) % cores) * 3U * tile;
            CopyOut(outGm, out, offset, Min(completed * tile, end - offset));
            outputQueue.FreeTensor(out);
        }
        if (task == tasks) break;
        if (task % 3U == 0U) batch = outputQueue.AllocTensor<uint8_t>();
        const uint32_t offset = (task / 3U * cores + (core + task / 3U * 3U) % cores) * 3U * tile + task % 3U * tile;
        const uint32_t length = Min(tile, end - offset);
        const uint32_t vectorLength = VectorLength<int32_t>(length);
        const uint32_t high = offset / period;
        if (high != previousHigh) {
            if (previousHigh != UINT32_MAX) broadcastQueue.FreeTensor(broadcast);
            broadcast = broadcastQueue.AllocTensor<int32_t>();
            CopyIn(broadcast, broadcastGm, high * low,
                   Min(2U, tiling.highSize - high) * low);
            broadcastQueue.EnQue(broadcast);
            broadcast = broadcastQueue.DeQue<int32_t>();
            uint64_t gathered = 0U;
            GatherMask(circular, broadcast, firstPattern, true, paddedWidth,
                       {1U, 2U, 0U, 0U}, gathered);
            PipeBarrier<PIPE_V>();
            previousHigh = high;
            previousOrigin = UINT32_MAX;
        }
        LocalTensor<int32_t> full = inputQueue.DeQue<int32_t>();
        if (task + 1U < tasks) {
            LocalTensor<int32_t> next = inputQueue.AllocTensor<int32_t>();
            const uint32_t nextTask = task + 1U;
            const uint32_t nextOffset = (nextTask / 3U * cores + (core + nextTask / 3U * 3U) % cores) * 3U * tile + nextTask % 3U * tile;
            CopyIn(next, fullGm, nextOffset, Min(tile, end - nextOffset));
            inputQueue.EnQue(next);
        }
        const uint32_t boundary = period - offset % period;
        const uint32_t phase = offset % low;
        const uint32_t origin = phase % 8U;
        if (previousOrigin != origin) {
            LocalTensor<uint32_t> pattern = patterns[origin * patternBytes].ReinterpretCast<uint32_t>();
            uint64_t gathered = 0U;
            GatherMask(repeated, circular, pattern, true, paddedWidth,
                       {1U, static_cast<uint16_t>(patternRows), 0U, 0U}, gathered);
            PipeBarrier<PIPE_V>();
            previousOrigin = origin;
        }
        LocalTensor<int32_t> compareValues = repeated[phase - origin];
        if (boundary < length) {
            uint64_t gathered = 0U;
            GatherMask(compareValues[boundary], broadcast[low / 8U * 8U], nextPattern,
                       true, paddedWidth,
                       {1U, static_cast<uint16_t>(CeilDiv(length - boundary, low)), 0U, 0U}, gathered);
            PipeBarrier<PIPE_V>();
        }
        VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                    SELMODE::VSEL_TENSOR_TENSOR_MODE);
        Int32CompareInPlace<X1Broadcast>(mask, full, compareValues, vectorLength);
        LocalTensor<uint8_t> out = batch[task % 3U * tile];
        LocalTensor<half> values = full.ReinterpretCast<half>();
        MaskToBoolFixedMask(values, out, ones, zeros, vectorLength);
        if ((task + 1U) % 3U == 0U || task + 1U == tasks) outputQueue.EnQue(batch);
        inputQueue.FreeTensor(full);
    }
    if (previousHigh != UINT32_MAX) broadcastQueue.FreeTensor(broadcast);
}

template <bool X1Broadcast>
__aicore__ inline void RunInt32Batch3(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                      const GreaterTilingData& tiling, float inverseLow)
{
    constexpr uint32_t tile = 8192U;
    constexpr uint32_t tilesPerBatch = 3U;
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t rowsPerTile = tile / low;
    const uint32_t tileLength = rowsPerTile * low;
    const uint32_t rows = tiling.highSize * middle;
    uint32_t gcd = low;
    uint32_t divisor = kAlignBytes;
    while (divisor != 0U) {
        const uint32_t next = gcd % divisor;
        gcd = divisor;
        divisor = next;
    }
    const uint32_t rowGroup = kAlignBytes / gcd;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t rowsPerCore = CeilDiv(CeilDiv(rows, blockNum), rowGroup) * rowGroup;
    const uint32_t rowStart = blockIdx * rowsPerCore;
    const uint32_t rowEnd = Min(rows, rowStart + rowsPerCore);
    if (rowStart >= rowEnd) return;

    GlobalTensor<int32_t> broadcastGm, fullGm;
    GlobalTensor<uint8_t> outGm;
    broadcastGm.SetGlobalBuffer((__gm__ int32_t*)(X1Broadcast ? x1 : x2));
    fullGm.SetGlobalBuffer((__gm__ int32_t*)(X1Broadcast ? x2 : x1));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    fullGm.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> broadcastQueue;
    TQue<QuePosition::VECIN, 2> fullQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> maskBuf, constantsBuf, repeatedBuf, offsetsBuf;
    pipe.InitBuffer(broadcastQueue, 1, VectorLength<int32_t>(low) * sizeof(int32_t));
    pipe.InitBuffer(fullQueue, 2, tile * sizeof(int32_t));
    pipe.InitBuffer(outQueue, 2, tilesPerBatch * tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(constantsBuf, 512U);
    pipe.InitBuffer(repeatedBuf, tile * sizeof(int32_t));
    pipe.InitBuffer(offsetsBuf, tile * sizeof(uint32_t));

    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> ones = constantsBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    LocalTensor<int32_t> repeated = repeatedBuf.Get<int32_t>();
    LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
    LocalTensor<int32_t> first = fullQueue.AllocTensor<int32_t>();
    CopyIn(first, fullGm, rowStart * low, tileLength);
    fullQueue.EnQue(first);
    LocalTensor<int32_t> firstBroadcast = broadcastQueue.AllocTensor<int32_t>();
    CopyIn(firstBroadcast, broadcastGm, rowStart / middle * low, low);
    broadcastQueue.EnQue(firstBroadcast);
    Duplicate(ones, static_cast<half>(1.0f), 128U);
    Duplicate(zeros, static_cast<half>(0.0f), 128U);
    PipeBarrier<PIPE_V>();
    BuildGatherOffsets(offsets, repeatedBuf.Get<float>(), 0U, 0.0f, tile,
                       low, sizeof(int32_t), inverseLow);
    VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                SELMODE::VSEL_TENSOR_TENSOR_MODE);

    for (uint32_t segment = rowStart; segment < rowEnd;) {
        const uint32_t segmentEnd = Min(rowEnd, (segment / middle + 1U) * middle);
        LocalTensor<int32_t> broadcast;
        if (segment == rowStart) {
            broadcast = broadcastQueue.DeQue<int32_t>();
        } else {
            broadcast = broadcastQueue.AllocTensor<int32_t>();
            CopyIn(broadcast, broadcastGm, segment / middle * low, low);
            broadcastQueue.EnQue(broadcast);
            broadcast = broadcastQueue.DeQue<int32_t>();
        }
        Gather(repeated, broadcast, offsets, 0U, tile);
        PipeBarrier<PIPE_V>();

        const uint32_t tileCount = (segmentEnd - segment) / rowsPerTile;
        const uint32_t batchCount = CeilDiv(tileCount, tilesPerBatch);
        uint32_t previousRow = segment;
        uint16_t previousTiles = 0U;
        for (uint32_t batchIndex = 0U; batchIndex <= batchCount; ++batchIndex) {
            const uint32_t firstTile = batchIndex * tilesPerBatch;
            const uint32_t currentTiles = batchIndex < batchCount ?
                Min(tilesPerBatch, tileCount - firstTile) : 0U;
            const uint32_t currentRow = segment + firstTile * rowsPerTile;
            LocalTensor<uint8_t> batch;
            if (currentTiles != 0U) {
                batch = outQueue.AllocTensor<uint8_t>();
                if (currentRow != rowStart) {
                    LocalTensor<int32_t> current = fullQueue.AllocTensor<int32_t>();
                    CopyIn(current, fullGm, currentRow * low, tileLength);
                    fullQueue.EnQue(current);
                }
            }
            if (previousTiles != 0U) {
                LocalTensor<uint8_t> previous = outQueue.DeQue<uint8_t>();
                CopyOutBatch(outGm, previous, previousRow * low, tileLength,
                             tile, previousTiles);
                outQueue.FreeTensor(previous);
            }
            for (uint32_t task = 0U; task < currentTiles; ++task) {
                const uint32_t row = currentRow + task * rowsPerTile;
                if (task + 1U < currentTiles) {
                    LocalTensor<int32_t> next = fullQueue.AllocTensor<int32_t>();
                    CopyIn(next, fullGm, (row + rowsPerTile) * low, tileLength);
                    fullQueue.EnQue(next);
                }
                LocalTensor<int32_t> full = fullQueue.DeQue<int32_t>();
                LocalTensor<uint8_t> out = batch[task * tile];
                Int32CompareInPlace<X1Broadcast>(mask, full, repeated, tile);
                LocalTensor<half> values = full.ReinterpretCast<half>();
                MaskToBoolFixedMask(values, out, ones, zeros, tile);
                fullQueue.FreeTensor(full);
            }
            if (currentTiles != 0U) outQueue.EnQue(batch);
            previousRow = currentRow;
            previousTiles = static_cast<uint16_t>(currentTiles);
        }
        broadcastQueue.FreeTensor(broadcast);
        segment = segmentEnd;
    }
}

template <typename T>
__aicore__ inline void RunSingleAxis(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                      const GreaterTilingData& tiling)
{
    const uint32_t low = tiling.lowSize;
    const uint32_t middle = tiling.middleSize;
    const uint32_t high = tiling.highSize;
    const bool x1Broadcast = tiling.broadcastIsX1 != 0U;
    float inverseLow = 1.0f;
    uint32_t stride = 1U;
    for (uint32_t axis = 0; axis < tiling.rank; ++axis) {
        if (stride == low) {
            inverseLow = tiling.inverseStrides[axis];
            break;
        }
        stride *= tiling.outDims[axis];
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        constexpr uint32_t scale = 1U << 24U;
        const uint32_t reciprocal = low == 0U ? 0U : CeilDiv(scale, low);
        if (low >= 256U && low <= 1024U && low % 8U != 0U && middle * low >= 16384U &&
            (middle * low) % 8U == 0U &&
            tiling.totalLength >= GetBlockNum() * 3U * 8192U &&
            static_cast<uint64_t>(8192U + low) * (reciprocal * low - scale) < scale &&
            static_cast<uint64_t>(tiling.totalLength) * sizeof(T) > kLargeStreamBytes) {
            if (x1Broadcast) RunInt32FlatStream<true>(x1, x2, output, tiling);
            else RunInt32FlatStream<false>(x1, x2, output, tiling);
            return;
        }
        if (CanUseInt32Batch3<T>(tiling.totalLength, low, middle, high)) {
            if (x1Broadcast) RunInt32Batch3<true>(x1, x2, output, tiling, inverseLow);
            else RunInt32Batch3<false>(x1, x2, output, tiling, inverseLow);
            return;
        }
    }
    if (low == 1U && high == 1U) {
        RunSingleAxisScalar<T>(x1, x2, output, tiling.totalLength, x1Broadcast);
        return;
    }
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        if (CanUseBf16BatchedRows(tiling.totalLength, low, middle, high)) {
            const bool singleTile = middle == kBf16RowsPerTile;
            if (x1Broadcast) {
                if (singleTile) RunBf16BatchedRows<true, true>(x1, x2, output, tiling, inverseLow);
                else RunBf16BatchedRows<true, false>(x1, x2, output, tiling, inverseLow);
            } else {
                if (singleTile) RunBf16BatchedRows<false, true>(x1, x2, output, tiling, inverseLow);
                else RunBf16BatchedRows<false, false>(x1, x2, output, tiling, inverseLow);
            }
            return;
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        const uint32_t rowStride = VectorLength<float>(low);
        const uint32_t outStride = VectorLength<uint8_t>(low);
        const uint64_t batchBytes =
            2ULL * rowStride * sizeof(float) +
            2ULL * middle * rowStride * sizeof(float) +
            2ULL * middle * outStride + MaskBytes(rowStride) + 512U;
        if (middle > 1U && middle <= 0xffffU && low * sizeof(float) % kAlignBytes != 0U &&
            batchBytes <= kBf16UbBudget) {
            RunSingleAxisSmallHigh(x1, x2, output, tiling);
            return;
        }
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    uint32_t a = low;
    uint32_t b = kAlignBytes;
    while (b != 0U) {
        const uint32_t next = a % b;
        a = b;
        b = next;
    }
    const uint32_t rowGroup = kAlignBytes / a;
    const uint32_t rows = high * middle;
    const uint32_t rowsPerCore = CeilDiv(CeilDiv(rows, blockNum), rowGroup) * rowGroup;
    const uint32_t rowStart = blockIdx * rowsPerCore;
    const uint32_t rowEnd = Min(rows, rowStart + rowsPerCore);
    if (rowStart >= rowEnd) {
        return;
    }

    const uint32_t baseTile = TileElements<T>();
    const uint32_t vectorBlock = 256U / sizeof(T);
    constexpr uint32_t baseGatherTile = std::is_same_v<T, float> ? 8448U : 8192U;
    const bool alignedRow = low * sizeof(T) % kAlignBytes == 0U;
    const uint32_t compareBlock = 256U / sizeof(CompareT<T>);
    uint32_t gcd = low;
    uint32_t divisor = compareBlock;
    while (divisor != 0U) {
        const uint32_t next = gcd % divisor;
        gcd = divisor;
        divisor = next;
    }
    const uint32_t gatherPeriod = low / gcd * compareBlock;
    const bool widePeriodic = std::is_same_v<T, float> && !alignedRow &&
        gatherPeriod > 4096U && gatherPeriod <= baseGatherTile;
    const uint32_t candidateTile = widePeriodic ? 16512U : baseTile;
    const uint32_t gatherTile = widePeriodic ? 16512U : baseGatherTile;
    const bool periodicGather = !alignedRow && gatherPeriod <= gatherTile;
    const bool smallDirect = std::is_same_v<T, half> ||
        (std::is_same_v<T, int32_t> && low > 4096U);
    const uint32_t directCapacity = !smallDirect ? gatherTile : 4096U;
    const uint32_t gatherCapacity = periodicGather ? gatherTile : directCapacity;
    const bool directGather = !alignedRow && !periodicGather && low <= Min(gatherCapacity, candidateTile);
    const bool directRow = std::is_same_v<T, float> && directGather && low > gatherCapacity / 3U;
    const uint32_t directRows = directGather ? gatherCapacity / low : 0U;
    const uint32_t alignedDirectRows = std::is_same_v<T, float> && directRows >= rowGroup ?
        directRows / rowGroup * rowGroup : directRows;
    const uint32_t gatherRows = periodicGather ?
        gatherTile / gatherPeriod * (gatherPeriod / low) :
        alignedDirectRows;
    const uint32_t gatherLength = gatherRows * low;
    const bool compactDoubleDirect = std::is_same_v<T, float> && directGather && !directRow &&
        gatherLength != 0U && gatherLength * 2U <= 16512U;
    const bool extendedDoubleDirect = std::is_same_v<T, float> && directGather && !directRow &&
        !compactDoubleDirect && gatherLength * 2U <= 17408U;
    const bool doubleDirect = compactDoubleDirect || extendedDoubleDirect;
    const bool gatherDoubleDirect = doubleDirect &&
        gatherLength % (kAlignBytes / sizeof(CompareT<T>)) != 0U;
    const bool extendedFullDirect = compactDoubleDirect && gatherDoubleDirect &&
        17408U / low > 16512U / low;
    const bool extendedLayout = extendedDoubleDirect || extendedFullDirect;
    const uint32_t extendedTile = extendedLayout ? 17408U :
        (compactDoubleDirect ? 16512U : candidateTile);
    const bool tailFullDirect = gatherDoubleDirect && 17920U / low > extendedTile / low;
    const uint32_t middleTile = VectorLength<uint8_t>(middle * low);
    const uint32_t eightRowTile = VectorLength<uint8_t>(8U * low);
    const bool smallMiddlePattern = gatherDoubleDirect && middle > 1U && middle < 8U &&
        middleTile <= 17920U && VectorLength<float>(middle * low) <= middleTile;
    const bool alignedEightRows = gatherDoubleDirect && !smallMiddlePattern &&
        eightRowTile <= 17920U && VectorLength<float>(8U * low) <= eightRowTile;
    const uint32_t tile = smallMiddlePattern ? VectorLength<uint8_t>(middle * low) :
        (alignedEightRows ? VectorLength<uint8_t>(8U * low) :
         (tailFullDirect ? 17920U : extendedTile));
    uint32_t directPatternRows = gatherRows;
    if (gatherDoubleDirect) {
        directPatternRows = smallMiddlePattern ? middle :
            (alignedEightRows ? 8U : tile / low);
        if (VectorLength<CompareT<T>>(directPatternRows * low) > tile) --directPatternRows;
    }
    const uint32_t doubleGatherCount = gatherDoubleDirect ?
        VectorLength<CompareT<T>>(directPatternRows * low) : 0U;
    const bool reducedOffsets = extendedLayout || tailFullDirect;
    const uint32_t doubleOffsetTile = reducedOffsets ? 6144U : 8192U;
    const float doubleOffsetTileFloat = reducedOffsets ? 6144.0f : 8192.0f;
    const uint32_t candidatePatternRows = directRow ? 0U : directPatternRows;
    const uint32_t patternRows = std::is_same_v<T, int32_t> &&
        candidatePatternRows != 0U && middle % candidatePatternRows != 0U ?
        0U : candidatePatternRows;
    const uint32_t rowTile = patternRows == 0U ? 1U :
        (doubleDirect && !gatherDoubleDirect ? patternRows * 2U : patternRows);
    const uint32_t offsetCount = gatherDoubleDirect ?
        (smallMiddlePattern ? doubleGatherCount : Min(doubleOffsetTile, doubleGatherCount)) :
        (patternRows == 0U ? 8U :
         (periodicGather ? gatherPeriod : VectorLength<CompareT<T>>(gatherLength)));
    constexpr bool supportsPipelining = std::is_same_v<T, float> ||
        std::is_same_v<T, int32_t> || std::is_same_v<T, bfloat16_t>;
    constexpr uint32_t queueDepth = supportsPipelining ? 2U : 1U;
    const bool int32Direct = std::is_same_v<T, int32_t> && directGather;
    const bool bf16Gather = std::is_same_v<T, bfloat16_t> &&
        !alignedRow && (periodicGather || directGather);
    const uint32_t bufferCount = widePeriodic || doubleDirect ||
        (std::is_same_v<T, int32_t> && !directGather) ||
        (std::is_same_v<T, bfloat16_t> && !bf16Gather) ? 1U : queueDepth;
    const bool reuseFullValues = widePeriodic || doubleDirect || directRow ||
        int32Direct || bf16Gather;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> broadcastQueue;
    TQue<QuePosition::VECIN, queueDepth> fullQueue;
    TQue<QuePosition::VECOUT, queueDepth> outQueue;
    TBuf<QuePosition::VECCALC> maskBuf, valueBuf, onesBuf, cast1Buf, cast2Buf;
    TBuf<QuePosition::VECCALC> compareGapBuf, repeatedBuf, offsetsBuf;
    GlobalTensor<T> broadcastGm, fullGm;
    GlobalTensor<uint8_t> outGm;
    broadcastGm.SetGlobalBuffer((__gm__ T*)(x1Broadcast ? x1 : x2));
    fullGm.SetGlobalBuffer((__gm__ T*)(x1Broadcast ? x2 : x1));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    if (static_cast<uint64_t>(tiling.totalLength) * sizeof(T) > 1024U * 1024U) {
        fullGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }
    const uint32_t broadcastElements = !alignedRow && low <= tile ?
        VectorLength<T>(low) : tile + vectorBlock;
    pipe.InitBuffer(broadcastQueue, 1, broadcastElements * sizeof(T));
    pipe.InitBuffer(fullQueue, bufferCount, tile * sizeof(T));
    const bool batchOutput = alignedEightRows && middle >= 32U;
    const uint32_t bf16BatchTiles = middle % rowTile == 0U ? middle / rowTile : 0U;
    const bool bf16BatchOutput = std::is_same_v<T, bfloat16_t> && bf16Gather &&
        bf16BatchTiles >= 2U && bf16BatchTiles <= 3U;
    pipe.InitBuffer(outQueue, bufferCount,
                    (batchOutput ? 4U : (bf16BatchOutput ? bf16BatchTiles : 1U)) * tile);
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(valueBuf, reuseFullValues ? kAlignBytes : tile * sizeof(half));
    pipe.InitBuffer(onesBuf, reuseFullValues ? 512U : tile * sizeof(half));
    if (directGather && std::is_same_v<T, float>) {
        pipe.InitBuffer(compareGapBuf, 256U);
    }
    pipe.InitBuffer(repeatedBuf, directRow ? kAlignBytes :
                    (doubleDirect ? tile : gatherTile) * sizeof(CompareT<T>));
    pipe.InitBuffer(offsetsBuf, offsetCount * sizeof(uint32_t));
    if constexpr (kNeedsCast<T>) {
        pipe.InitBuffer(cast1Buf, tile * sizeof(CompareT<T>));
        pipe.InitBuffer(cast2Buf, Min(tile, VectorLength<T>(low)) * sizeof(CompareT<T>));
    }
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> values = valueBuf.Get<half>();
    LocalTensor<half> ones = onesBuf.Get<half>();
    LocalTensor<half> zeros = ones[128];
    LocalTensor<CompareT<T>> repeated = repeatedBuf.Get<CompareT<T>>();
    LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
    const uint32_t firstWindowRows = patternRows == 0U ? 0U :
        Min(rowEnd - rowStart, patternRows);
    const bool prefetchFirst = int32Direct && x1Broadcast && patternRows != 0U &&
        patternRows <= 4U * queueDepth &&
        static_cast<uint64_t>(tiling.totalLength) * sizeof(T) > kLargeStreamBytes &&
        rowStart % middle + firstWindowRows <= middle;
    const bool fixedTensorMask = prefetchFirst;
    const uint32_t prefetchedRows = prefetchFirst ? firstWindowRows : 0U;
    if (prefetchFirst) {
        for (uint32_t rows = 0U; rows < prefetchedRows; rows += patternRows) {
            const uint32_t length = low * Min(patternRows, prefetchedRows - rows);
            LocalTensor<T> full = fullQueue.template AllocTensor<T>();
            CopyIn(full, fullGm, (rowStart + rows) * low, length);
            fullQueue.EnQue(full);
        }
        LocalTensor<T> broadcast = broadcastQueue.template AllocTensor<T>();
        CopyIn(broadcast, broadcastGm, rowStart / middle * low, low);
        broadcastQueue.EnQue(broadcast);
    }
    if (reuseFullValues) {
        Duplicate(ones, static_cast<half>(1.0f), 128U);
        Duplicate(zeros, static_cast<half>(0.0f), 128U);
        PipeBarrier<PIPE_V>();
    } else {
        Duplicate(ones, static_cast<half>(1.0f), tile);
    }
    if (smallMiddlePattern) {
        BuildGatherOffsets(offsets, repeatedBuf.Get<float>(), 0U, 0.0f,
                           doubleGatherCount, low, sizeof(float), inverseLow);
    } else if (patternRows != 0U && !gatherDoubleDirect) {
        const uint32_t scratchBytes = (doubleDirect ? tile : gatherTile) * sizeof(CompareT<T>);
        if (offsetCount * sizeof(float) <= scratchBytes) {
            float floatStart = 0.0f;
            for (uint32_t start = 0U; start < offsetCount; start += 8192U) {
                const uint32_t length = Min(8192U, offsetCount - start);
                LocalTensor<uint32_t> chunkOffsets = offsets[start];
                LocalTensor<float> chunkScratch = repeatedBuf.Get<float>()[start];
                BuildGatherOffsets(chunkOffsets, chunkScratch, start, floatStart,
                                   length, low, sizeof(CompareT<T>), inverseLow);
                floatStart += 8192.0f;
            }
        } else {
            for (uint32_t i = 0; i < offsetCount; ++i) {
                offsets.SetValue(i, i % low * sizeof(CompareT<T>));
            }
            PipeBarrier<PIPE_V>();
        }
    }
    if (fixedTensorMask) {
        VselIntrinsicsImplPre<half>((__ubuf__ half*)mask.GetPhyAddr(),
                                    SELMODE::VSEL_TENSOR_TENSOR_MODE);
    }
    if (alignedRow && low <= tile) {
        for (uint32_t segment = rowStart; segment < rowEnd;) {
            const uint32_t h = segment / middle;
            const uint32_t segmentEnd = Min(rowEnd, (h + 1U) * middle);
            const uint32_t cachedRows = Min(segmentEnd - segment, tile / low);
            LocalTensor<T> broadcast = broadcastQueue.AllocTensor<T>();
            CopyIn(broadcast, broadcastGm, h * low, low);
            broadcastQueue.EnQue(broadcast);
            broadcast = broadcastQueue.DeQue<T>();
            LocalTensor<CompareT<T>> broadcastValue;
            if constexpr (kNeedsCast<T>) {
                broadcastValue = cast2Buf.Get<CompareT<T>>();
                Cast(broadcastValue, broadcast, RoundMode::CAST_NONE, VectorLength<T>(low));
                PipeBarrier<PIPE_V>();
                if (cachedRows > 1U) {
                    ExpandRows<CompareT<T>, true>(broadcastValue, low, cachedRows);
                }
            } else {
                broadcastValue = broadcast.template ReinterpretCast<CompareT<T>>();
                if (cachedRows > 1U) ExpandRows<T, true>(broadcast, low, cachedRows);
            }

            for (uint32_t row = segment; row < segmentEnd;) {
                const uint32_t groupRows = Min(segmentEnd - row, cachedRows);
                const uint32_t length = low * groupRows;
                const uint32_t vectorLength = VectorLength<CompareT<T>>(length);
                LocalTensor<T> full = fullQueue.template AllocTensor<T>();
                CopyIn(full, fullGm, row * low, length);
                fullQueue.EnQue(full);
                full = fullQueue.template DeQue<T>();
                LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
                if constexpr (kNeedsCast<T>) {
                    LocalTensor<CompareT<T>> fullValue = cast1Buf.Get<CompareT<T>>();
                    Cast(fullValue, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                    if (x1Broadcast) Compare(mask, fullValue, broadcastValue, CMPMODE::LT, vectorLength);
                    else Compare(mask, fullValue, broadcastValue, CMPMODE::GT, vectorLength);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    if (x1Broadcast) Int32CompareInPlace<true>(mask, full, broadcastValue, vectorLength);
                    else Int32CompareInPlace<false>(mask, full, broadcastValue, vectorLength);
                } else if (x1Broadcast) {
                    NativeCompare<true>(mask, full, broadcastValue, out, vectorLength);
                } else {
                    NativeCompare<false>(mask, full, broadcastValue, out, vectorLength);
                }
                MaskToBool(values, mask, out, ones, vectorLength);
                outQueue.EnQue(out);
                fullQueue.FreeTensor(full);
                out = outQueue.template DeQue<uint8_t>();
                CopyOut(outGm, out, row * low, length);
                outQueue.FreeTensor(out);
                row += groupRows;
            }
            broadcastQueue.FreeTensor(broadcast);
            segment = segmentEnd;
        }
        return;
    }
    if (low <= tile) {
        for (uint32_t segment = rowStart; segment < rowEnd;) {
            const uint32_t h = segment / middle;
            const uint32_t segmentEnd = Min(rowEnd, (h + 1U) * middle);
            LocalTensor<T> broadcast;
            if (prefetchFirst && segment == rowStart) {
                broadcast = broadcastQueue.template DeQue<T>();
            } else {
                broadcast = broadcastQueue.template AllocTensor<T>();
                CopyIn(broadcast, broadcastGm, h * low, low);
                broadcastQueue.EnQue(broadcast);
                broadcast = broadcastQueue.template DeQue<T>();
            }
            LocalTensor<CompareT<T>> broadcastValue =
                broadcast.template ReinterpretCast<CompareT<T>>();
            if constexpr (kNeedsCast<T>) {
                broadcastValue = cast2Buf.Get<CompareT<T>>();
            }
            if (patternRows != 0U) {
                if constexpr (kNeedsCast<T>) {
                    Cast(broadcastValue, broadcast, RoundMode::CAST_NONE, VectorLength<T>(low));
                    PipeBarrier<PIPE_V>();
                }
                if (gatherDoubleDirect) {
                    if constexpr (std::is_same_v<T, float>) {
                        if (smallMiddlePattern) {
                            Gather(repeated, broadcastValue, offsets, 0U, doubleGatherCount);
                            PipeBarrier<PIPE_V>();
                        } else {
                            float floatStart = 0.0f;
                            for (uint32_t start = 0U; start < doubleGatherCount;
                                 start += doubleOffsetTile) {
                                const uint32_t length = Min(doubleOffsetTile, doubleGatherCount - start);
                                LocalTensor<float> chunk = repeatedBuf.Get<float>()[start];
                                BuildGatherOffsets(offsets, chunk, start, floatStart,
                                                   length, low, sizeof(float), inverseLow);
                                Gather(chunk, broadcastValue, offsets, 0U, length);
                                PipeBarrier<PIPE_V>();
                                floatStart += doubleOffsetTileFloat;
                            }
                        }
                    }
                } else {
                    Gather(repeated, broadcastValue, offsets, 0U, offsetCount);
                    PipeBarrier<PIPE_V>();
                    if (doubleDirect) DuplicatePattern(repeated, gatherLength);
                }
                if (periodicGather && gatherLength > gatherPeriod) {
                    ExpandRows<CompareT<T>, true>(repeated, gatherPeriod,
                                                  gatherLength / gatherPeriod);
                }
                broadcastValue = repeated;
            } else if constexpr (kNeedsCast<T>) {
                Cast(broadcastValue, broadcast, RoundMode::CAST_NONE, VectorLength<T>(low));
            }

            if constexpr (std::is_same_v<T, bfloat16_t>) {
                if (bf16BatchOutput) {
                    const uint32_t tileCount = CeilDiv(segmentEnd - segment, rowTile);
                    LocalTensor<uint8_t> batch = outQueue.template AllocTensor<uint8_t>();
                    for (uint32_t step = 0U; step <= tileCount; ++step) {
                        if (step < tileCount) {
                            const uint32_t row = segment + step * rowTile;
                            LocalTensor<T> full = fullQueue.template AllocTensor<T>();
                            CopyIn(full, fullGm, row * low, rowTile * low);
                            fullQueue.EnQue(full);
                        }
                        if (step > 0U) {
                            const uint32_t task = step - 1U;
                            const uint32_t length = rowTile * low;
                            const uint32_t vectorLength = VectorLength<CompareT<T>>(length);
                            LocalTensor<T> full = fullQueue.template DeQue<T>();
                            LocalTensor<CompareT<T>> fullValue = cast1Buf.Get<CompareT<T>>();
                            LocalTensor<uint8_t> out = batch[task * tile];
                            Cast(fullValue, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                            Compare(mask, fullValue, broadcastValue,
                                    x1Broadcast ? CMPMODE::LT : CMPMODE::GT, vectorLength);
                            LocalTensor<half> reusedValues = full.template ReinterpretCast<half>();
                            MaskToBoolRepeatedConstants(reusedValues, mask, out, ones, zeros,
                                                        vectorLength);
                            fullQueue.FreeTensor(full);
                        }
                    }
                    outQueue.EnQue(batch);
                    batch = outQueue.template DeQue<uint8_t>();
                    CopyOutBatch(outGm, batch, segment * low, rowTile * low,
                                 tile, static_cast<uint16_t>(tileCount));
                    outQueue.FreeTensor(batch);
                    broadcastQueue.FreeTensor(broadcast);
                    segment = segmentEnd;
                    continue;
                }
            }
            if ((std::is_same_v<T, float> &&
                 ((periodicGather && !widePeriodic) || directRow)) || int32Direct || bf16Gather) {
                const uint32_t tileCount = CeilDiv(segmentEnd - segment, rowTile);
                for (uint32_t step = 0; step < tileCount + 2U; ++step) {
                    if (step < tileCount) {
                        const uint32_t row = segment + step * rowTile;
                        const uint32_t length = low * Min(segmentEnd - row, rowTile);
                        if (!prefetchFirst || row >= rowStart + prefetchedRows) {
                            LocalTensor<T> full = fullQueue.template AllocTensor<T>();
                            CopyIn(full, fullGm, row * low, length);
                            fullQueue.EnQue(full);
                        }
                    }
                    if (step > 0U && step <= tileCount) {
                        const uint32_t row = segment + (step - 1U) * rowTile;
                        const uint32_t length = low * Min(segmentEnd - row, rowTile);
                        const uint32_t vectorLength = VectorLength<CompareT<T>>(length);
                        LocalTensor<T> full = fullQueue.template DeQue<T>();
                        LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
                        if constexpr (kNeedsCast<T>) {
                            LocalTensor<CompareT<T>> fullValue = cast1Buf.Get<CompareT<T>>();
                            Cast(fullValue, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                            if (x1Broadcast) {
                                Compare(mask, fullValue, broadcastValue, CMPMODE::LT, vectorLength);
                            } else {
                                Compare(mask, fullValue, broadcastValue, CMPMODE::GT, vectorLength);
                            }
                        } else if constexpr (std::is_same_v<T, int32_t>) {
                            if (x1Broadcast) Int32CompareInPlace<true>(mask, full, broadcastValue,
                                                                      vectorLength);
                            else Int32CompareInPlace<false>(mask, full, broadcastValue, vectorLength);
                        } else if (x1Broadcast) {
                            NativeCompare<true>(mask, full, broadcastValue, out, vectorLength);
                        } else {
                            NativeCompare<false>(mask, full, broadcastValue, out, vectorLength);
                        }
                        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
                                      std::is_same_v<T, bfloat16_t>) {
                            if (reuseFullValues) {
                                LocalTensor<half> reusedValues = full.template ReinterpretCast<half>();
                                if (fixedTensorMask) {
                                    MaskToBoolFixedMask(reusedValues, out, ones, zeros, vectorLength);
                                } else {
                                    MaskToBoolRepeatedConstants(reusedValues, mask, out, ones, zeros,
                                                                vectorLength);
                                }
                            } else {
                                MaskToBool(values, mask, out, ones, vectorLength);
                            }
                        } else {
                            MaskToBool(values, mask, out, ones, vectorLength);
                        }
                        outQueue.EnQue(out);
                        fullQueue.FreeTensor(full);
                    }
                    if (step > 1U) {
                        const uint32_t row = segment + (step - 2U) * rowTile;
                        const uint32_t length = low * Min(segmentEnd - row, rowTile);
                        LocalTensor<uint8_t> out = outQueue.template DeQue<uint8_t>();
                        CopyOut(outGm, out, row * low, length);
                        outQueue.FreeTensor(out);
                    }
                }
            } else {
                for (uint32_t row = segment; row < segmentEnd;) {
                    if constexpr (std::is_same_v<T, float>) {
                        if (batchOutput && segmentEnd - row >= 32U) {
                            LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
                            for (uint32_t batch = 0U; batch < 4U; ++batch) {
                                const uint32_t batchRow = row + batch * 8U;
                                const uint32_t length = 8U * low;
                                const uint32_t vectorLength = VectorLength<float>(length);
                                LocalTensor<float> full = fullQueue.template AllocTensor<float>();
                                CopyIn(full, fullGm, batchRow * low, length);
                                fullQueue.EnQue(full);
                                full = fullQueue.template DeQue<float>();
                                LocalTensor<uint8_t> batchOut = out[batch * tile];
                                if (x1Broadcast) {
                                    NativeCompare<true>(mask, full, broadcastValue,
                                                        batchOut, vectorLength);
                                } else {
                                    NativeCompare<false>(mask, full, broadcastValue,
                                                         batchOut, vectorLength);
                                }
                                LocalTensor<half> reusedValues = full.template ReinterpretCast<half>();
                                MaskToBoolRepeatedConstants(reusedValues, mask, batchOut,
                                                            ones, zeros, vectorLength);
                                fullQueue.FreeTensor(full);
                            }
                            outQueue.EnQue(out);
                            out = outQueue.template DeQue<uint8_t>();
                            CopyOutBatch(outGm, out, row * low, 8U * low, tile, 4U);
                            outQueue.FreeTensor(out);
                            row += 32U;
                            continue;
                        }
                    }
                    const uint32_t groupRows = Min(segmentEnd - row, rowTile);
                    const uint32_t length = low * groupRows;
                    const uint32_t vectorLength = VectorLength<CompareT<T>>(length);
                    LocalTensor<T> full = fullQueue.template AllocTensor<T>();
                    CopyIn(full, fullGm, row * low, length);
                    fullQueue.EnQue(full);
                    full = fullQueue.template DeQue<T>();
                    LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
                    if constexpr (kNeedsCast<T>) {
                        LocalTensor<CompareT<T>> fullValue = cast1Buf.Get<CompareT<T>>();
                        Cast(fullValue, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                        if (x1Broadcast) Compare(mask, fullValue, broadcastValue, CMPMODE::LT, vectorLength);
                        else Compare(mask, fullValue, broadcastValue, CMPMODE::GT, vectorLength);
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        if (x1Broadcast) Int32CompareInPlace<true>(mask, full, broadcastValue, vectorLength);
                        else Int32CompareInPlace<false>(mask, full, broadcastValue, vectorLength);
                    } else if (x1Broadcast) {
                        NativeCompare<true>(mask, full, broadcastValue, out, vectorLength);
                    } else {
                        NativeCompare<false>(mask, full, broadcastValue, out, vectorLength);
                    }
                    if constexpr (std::is_same_v<T, float>) {
                        LocalTensor<half> reusedValues = full.template ReinterpretCast<half>();
                        if (reuseFullValues) {
                            MaskToBoolRepeatedConstants(reusedValues, mask, out, ones, zeros,
                                                        vectorLength);
                        }
                        else MaskToBool(reusedValues, mask, out, ones, vectorLength);
                    } else {
                        MaskToBool(values, mask, out, ones, vectorLength);
                    }
                    outQueue.EnQue(out);
                    fullQueue.FreeTensor(full);
                    out = outQueue.template DeQue<uint8_t>();
                    CopyOut(outGm, out, row * low, length);
                    outQueue.FreeTensor(out);
                    row += groupRows;
                }
            }
            broadcastQueue.FreeTensor(broadcast);
            segment = segmentEnd;
        }
        return;
    }
    for (uint32_t segment = rowStart; segment < rowEnd;) {
        const uint32_t h = segment / middle;
        const uint32_t segmentEnd = Min(rowEnd, (h + 1U) * middle);
        for (uint32_t lowBase = 0; lowBase < low; lowBase += tile) {
            const uint32_t length = Min(low - lowBase, tile);
            const uint32_t vectorLength = VectorLength<CompareT<T>>(length);
            LocalTensor<T> broadcast = broadcastQueue.AllocTensor<T>();
            CopyIn(broadcast, broadcastGm, h * low + lowBase, length);
            broadcastQueue.EnQue(broadcast);
            broadcast = broadcastQueue.DeQue<T>();
            LocalTensor<CompareT<T>> broadcastValue;
            if constexpr (kNeedsCast<T>) {
                broadcastValue = cast2Buf.Get<CompareT<T>>();
                Cast(broadcastValue, broadcast, RoundMode::CAST_NONE, VectorLength<T>(length));
            } else {
                broadcastValue = broadcast.template ReinterpretCast<CompareT<T>>();
            }

            for (uint32_t row = segment; row < segmentEnd; ++row) {
                LocalTensor<T> full = fullQueue.template AllocTensor<T>();
                CopyIn(full, fullGm, row * low + lowBase, length);
                fullQueue.EnQue(full);
                full = fullQueue.template DeQue<T>();
                LocalTensor<uint8_t> out = outQueue.template AllocTensor<uint8_t>();
                if constexpr (kNeedsCast<T>) {
                    LocalTensor<CompareT<T>> fullValue = cast1Buf.Get<CompareT<T>>();
                    Cast(fullValue, full, RoundMode::CAST_NONE, VectorLength<T>(length));
                    if (x1Broadcast) {
                        Compare(mask, fullValue, broadcastValue, CMPMODE::LT, vectorLength);
                    } else {
                        Compare(mask, fullValue, broadcastValue, CMPMODE::GT, vectorLength);
                    }
                    MaskToBool(values, mask, out, ones, vectorLength);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    if (x1Broadcast) Int32CompareInPlace<true>(mask, full, broadcast, vectorLength);
                    else Int32CompareInPlace<false>(mask, full, broadcast, vectorLength);
                    MaskToBool(values, mask, out, ones, vectorLength);
                } else {
                    if (x1Broadcast) NativeCompare<true>(mask, full, broadcast, out, vectorLength);
                    else NativeCompare<false>(mask, full, broadcast, out, vectorLength);
                    MaskToBool(values, mask, out, ones, vectorLength);
                }
                outQueue.EnQue(out);
                fullQueue.FreeTensor(full);
                out = outQueue.template DeQue<uint8_t>();
                CopyOut(outGm, out, row * low + lowBase, length);
                outQueue.FreeTensor(out);
            }
            broadcastQueue.FreeTensor(broadcast);
        }
        segment = segmentEnd;
    }
}

template <typename T>
__aicore__ inline void RunFullMiddleRows(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                         const GreaterTilingData& tiling)
{
    using C = CompareT<T>;
    const bool middleIsX1 = tiling.broadcastIsX1 != 0U;
    const uint32_t cols = tiling.outDims[0];
    const uint32_t middle = tiling.outDims[1];
    const uint32_t rows = tiling.outDims[2] * middle;
    uint32_t a = cols;
    uint32_t b = kAlignBytes;
    while (b != 0U) {
        const uint32_t next = a % b;
        a = b;
        b = next;
    }
    const uint32_t rowGroup = kAlignBytes / a;
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    const uint32_t rowsPerCore = CeilDiv(CeilDiv(rows, blockNum), rowGroup) * rowGroup;
    const uint32_t rowStart = GetBlockIdx() * rowsPerCore;
    const uint32_t rowEnd = Min(rows, rowStart + rowsPerCore);
    if (rowStart >= rowEnd) return;

    const uint32_t tile = TileElements<T>();
    GlobalTensor<T> fullGm, middleGm;
    GlobalTensor<uint8_t> outGm;
    fullGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x2 : x1));
    middleGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x1 : x2));
    outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
    if (static_cast<uint64_t>(rows) * cols * sizeof(T) > 1024U * 1024U) {
        fullGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> fullQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> castBuf, scalarInputBuf, scalarCastBuf;
    TBuf<QuePosition::VECCALC> maskBuf, valueBuf, maximumBuf, scalarBuf;
    pipe.InitBuffer(fullQueue, 1, tile * sizeof(T));
    pipe.InitBuffer(outQueue, 1, tile);
    if constexpr (kNeedsCast<T>) {
        pipe.InitBuffer(castBuf, tile * sizeof(C));
        pipe.InitBuffer(scalarInputBuf, kAlignBytes);
        pipe.InitBuffer(scalarCastBuf, kAlignBytes);
    }
    pipe.InitBuffer(maskBuf, MaskBytes(tile));
    pipe.InitBuffer(valueBuf, tile * sizeof(half));
    if constexpr (std::is_same_v<T, int32_t>) {
        pipe.InitBuffer(maximumBuf, tile * sizeof(int32_t));
        pipe.InitBuffer(scalarBuf, tile * sizeof(int32_t));
    }
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<half> values = valueBuf.Get<half>();
    TEventID scalarToVector;
    TEventID vectorToScalar;
    if constexpr (kNeedsCast<T>) {
        scalarToVector = pipe.FetchEventID<HardEvent::S_V>();
        vectorToScalar = pipe.FetchEventID<HardEvent::V_S>();
    }

    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        C scalar;
        if constexpr (kNeedsCast<T>) {
            LocalTensor<T> scalarInput = scalarInputBuf.Get<T>();
            LocalTensor<C> scalarCast = scalarCastBuf.Get<C>();
            scalarInput.SetValue(0, middleGm.GetValue(row % middle));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            Cast(scalarCast, scalarInput, RoundMode::CAST_NONE, VectorLength<T>(1U));
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            scalar = scalarCast.GetValue(0);
        } else {
            scalar = static_cast<C>(middleGm.GetValue(row % middle));
        }
        for (uint32_t colBase = 0; colBase < cols; colBase += tile) {
            const uint32_t length = Min(cols - colBase, tile);
            const uint32_t vectorLength = VectorLength<C>(length);
            LocalTensor<T> source = fullQueue.AllocTensor<T>();
            CopyIn(source, fullGm, row * cols + colBase, length);
            fullQueue.EnQue(source);
            source = fullQueue.DeQue<T>();
            LocalTensor<C> full;
            if constexpr (kNeedsCast<T>) {
                full = castBuf.Get<C>();
                Cast(full, source, RoundMode::CAST_NONE, length);
            } else {
                full = source.template ReinterpretCast<C>();
            }

            LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
            if constexpr (std::is_same_v<T, int32_t>) {
                LocalTensor<int32_t> scalarVector = scalarBuf.Get<int32_t>();
                LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                Duplicate(scalarVector, scalar, vectorLength);
                if (middleIsX1) Int32Compare<true>(mask, full, scalarVector, maximum, out, vectorLength);
                else Int32Compare<false>(mask, full, scalarVector, maximum, out, vectorLength);
            } else {
                CompareScalar(mask, full, scalar, middleIsX1 ? CMPMODE::LT : CMPMODE::GT, vectorLength);
            }
            MaskToBool(values, mask, out, vectorLength);
            outQueue.EnQue(out);
            fullQueue.FreeTensor(source);
            out = outQueue.DeQue<uint8_t>();
            CopyOut(outGm, out, row * cols + colBase, length);
            outQueue.FreeTensor(out);
        }
    }
    if constexpr (kNeedsCast<T>) {
        pipe.ReleaseEventID<HardEvent::S_V>(scalarToVector);
        pipe.ReleaseEventID<HardEvent::V_S>(vectorToScalar);
    }
}

template <typename T>
class FullMiddleKernel {
    static constexpr uint32_t kMiddleTile = 1024;
    static constexpr uint32_t kBlockElems = kAlignBytes / sizeof(T);
    static constexpr uint32_t kLocalElems = kMiddleTile * kBlockElems;
    static constexpr uint32_t AlignUb(uint32_t bytes)
    { return (bytes + 1023U) / 1024U * 1024U; }
    static constexpr uint32_t kScalarBytes = kMiddleTile / 8U * kAlignBytes;
    static constexpr uint32_t kScalarOffset = AlignUb(kLocalElems * sizeof(T));
    static constexpr uint32_t kExpandedOffset = AlignUb(kScalarOffset + kScalarBytes);
    static constexpr uint32_t kPackedOffset = AlignUb(kExpandedOffset + kLocalElems * sizeof(T));
    static constexpr uint32_t kPatternOffset = AlignUb(kPackedOffset + kLocalElems * sizeof(T));
    static constexpr uint32_t kOutputOffset = kPatternOffset + 1024U;
    static constexpr uint32_t kMaskBytes = (kLocalElems + 255U) / 256U * kAlignBytes;
    static constexpr uint32_t kMaskOffset = AlignUb(kOutputOffset + kLocalElems);
    static constexpr uint32_t kOnesOffset = AlignUb(kMaskOffset + kMaskBytes);
    static constexpr uint32_t kValueOffset = AlignUb(kOnesOffset + kLocalElems * sizeof(half));

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        cols_ = tiling.outDims[0];
        middle_ = tiling.outDims[1];
        outer_ = tiling.outDims[2];
        middleIsX1_ = tiling.broadcastIsX1 != 0U;
        fullGm_.SetGlobalBuffer((__gm__ T*)(middleIsX1_ ? x2 : x1));
        middleGm_.SetGlobalBuffer((__gm__ T*)(middleIsX1_ ? x1 : x2));
        outGm_.SetGlobalBuffer((__gm__ uint8_t*)output);
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        outerStart_ = outer_ * blockIdx / blockNum;
        outerEnd_ = outer_ * (blockIdx + 1U) / blockNum;

        LocalTensor<T> full = Local<T>(0, kLocalElems);
        LocalTensor<T> scalar = Local<T>(kScalarOffset, kScalarBytes / sizeof(T));
        LocalTensor<T> expanded = Local<T>(kExpandedOffset, kLocalElems);
        LocalTensor<T> packed = Local<T>(kPackedOffset, kLocalElems);
        LocalTensor<uint8_t> out = Local<uint8_t>(kOutputOffset, kLocalElems);
        LocalTensor<uint8_t> mask = Local<uint8_t>(kMaskOffset, kMaskBytes);
        LocalTensor<half> ones = Local<half>(kOnesOffset, kLocalElems);
        LocalTensor<half> values = Local<half>(kValueOffset, kLocalElems);
        Duplicate(ones, static_cast<half>(1.0f), kLocalElems);
        InitPattern();

        uint32_t iteration = 0;
        const uint32_t middleTiles = CeilDiv(middle_, kMiddleTile);
        const uint32_t totalIterations = middleTiles * (outerEnd_ - outerStart_);
        for (uint32_t middleBase = 0; middleBase < middle_; middleBase += kMiddleTile) {
            const uint32_t currentMiddle = Min(middle_ - middleBase, kMiddleTile);
            if (iteration != 0U) WaitFlag<HardEvent::V_MTE2>(0);
            CopyScalars(scalar, middleBase, currentMiddle);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Brcb(expanded, scalar, static_cast<uint8_t>(currentMiddle / 8U), BrcbRepeatParams(1, 8));
            GatherScalars(packed, expanded, currentMiddle);
            const uint32_t length = currentMiddle * cols_;
            const uint32_t vectorLength = VectorLength<T>(length);

            for (uint32_t outer = outerStart_; outer < outerEnd_; ++outer) {
                const uint32_t offset = (outer * middle_ + middleBase) * cols_;
                const bool hasPrevious = iteration != 0U;
                const bool hasNext = iteration + 1U < totalIterations;
                if (outer != outerStart_) WaitFlag<HardEvent::V_MTE2>(0);
                DataCopy(full, fullGm_[offset], length);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                if (middleIsX1_) {
                    Compare(mask, packed, full, CMPMODE::GT, vectorLength);
                } else {
                    Compare(mask, full, packed, CMPMODE::GT, vectorLength);
                }
                if (hasNext) SetFlag<HardEvent::V_MTE2>(0);
                Select(values, mask, ones, static_cast<half>(0.0f),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorLength);
                if (hasPrevious) WaitFlag<HardEvent::MTE3_V>(0);
                Cast(out, values, RoundMode::CAST_TRUNC, vectorLength);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(outGm_[offset], out, length);
                if (hasNext) SetFlag<HardEvent::MTE3_V>(0);
                ++iteration;
            }
        }
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> Local(uint32_t byteOffset, uint32_t elements) const
    { return LocalTensor<U>(TPosition::VECCALC, byteOffset, elements); }

    __aicore__ inline void InitPattern()
    {
        LocalTensor<uint8_t> bytes = Local<uint8_t>(kPatternOffset, 1024U);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> pattern = bytes.ReinterpretCast<uint16_t>();
            const uint16_t bits = cols_ == 16U ? static_cast<uint16_t>(0xFFFFU) :
                                  static_cast<uint16_t>((1U << cols_) - 1U);
            Duplicate(pattern, bits, 256U / sizeof(uint16_t));
        } else {
            LocalTensor<uint32_t> pattern = bytes.ReinterpretCast<uint32_t>();
            const uint32_t lanes = (1U << cols_) - 1U;
            const uint32_t bits = lanes | (lanes << 8U) | (lanes << 16U) | (lanes << 24U);
            Duplicate(pattern, bits, 256U / sizeof(uint32_t));
        }
    }

    __aicore__ inline void GatherScalars(LocalTensor<T>& packed, LocalTensor<T>& expanded,
                                          uint32_t currentMiddle)
    {
        uint64_t gathered = 0;
        GatherMaskParams params = {1, static_cast<uint16_t>(currentMiddle / 8U), 8, 0};
        LocalTensor<uint8_t> bytes = Local<uint8_t>(kPatternOffset, 1024U);
        if constexpr (std::is_same_v<T, half>) {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint16_t>(), false, 0, params, gathered);
        } else {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint32_t>(), false, 0, params, gathered);
        }
    }

    __aicore__ inline void CopyScalars(LocalTensor<T>& local, uint32_t offset, uint32_t length)
    {
        const uint32_t bytes = length * sizeof(T);
        if ((offset * sizeof(T)) % kAlignBytes == 0U && bytes % kAlignBytes == 0U) {
            DataCopy(local, middleGm_[offset], length);
        } else {
            DataCopyExtParams copy = {1, bytes, 0, 0, 0};
            DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, middleGm_[offset], copy, pad);
        }
    }

    GlobalTensor<T> fullGm_, middleGm_;
    GlobalTensor<uint8_t> outGm_;
    uint32_t cols_, middle_, outer_, outerStart_, outerEnd_;
    bool middleIsX1_;
};

template <typename T>
class OuterMiddleKernel {
    static constexpr uint32_t kMiddleTile = 1536U;
    static constexpr uint32_t kBlockElems = kAlignBytes / sizeof(T);
    static constexpr uint32_t kCachedRows = 32U;

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        cols_ = tiling.outDims[0];
        middle_ = tiling.outDims[1];
        outer_ = tiling.outDims[2];
        middleIsX1_ = tiling.broadcastIsX1 != 0U;
        middleGm_.SetGlobalBuffer((__gm__ T*)(middleIsX1_ ? x1 : x2));
        rowGm_.SetGlobalBuffer((__gm__ T*)(middleIsX1_ ? x2 : x1));
        outGm_.SetGlobalBuffer((__gm__ uint8_t*)output);
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        outerStart_ = outer_ * blockIdx / blockNum;
        outerEnd_ = outer_ * (blockIdx + 1U) / blockNum;
        if (outerStart_ >= outerEnd_) return;
        if (cols_ <= 8U && middle_ % 8U == 0U) {
            RunPacked();
        } else {
            RunRowVector();
        }
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> Local(uint32_t byteOffset, uint32_t elements) const
    { return LocalTensor<U>(TPosition::VECCALC, byteOffset, elements); }

    __aicore__ inline void RunPacked()
    {
        const uint32_t middleTile = kMiddleTile;
        const uint32_t expandedElems = middleTile * kBlockElems;
        const uint32_t packedElems = middleTile * cols_;
        const uint32_t repeatedOffset = 0U;
        const uint32_t scalarOffset = Align(packedElems * sizeof(T));
        const uint32_t scalarBytes = middleTile / 8U * kAlignBytes;
        const uint32_t expandedOffset = Align(scalarOffset + scalarBytes);
        const uint32_t packedOffset = Align(expandedOffset + expandedElems * sizeof(T));
        const uint32_t patternOffset = Align(packedOffset + packedElems * sizeof(T));
        const uint32_t rowOffset = patternOffset + 1024U;
        const uint32_t rowElems = kCachedRows * 8U;
        const uint32_t valueOffset = expandedOffset;
        const uint32_t outputOffset = Align(valueOffset + packedElems * sizeof(half));
        const uint32_t outputStride = Align(packedElems);
        const uint32_t maskOffset = Align(rowOffset + rowElems * sizeof(T));
        const uint32_t maskBytes = MaskBytes(packedElems);
        const uint32_t offsetsOffset = Align(maskOffset + maskBytes);
        const uint32_t offsetsElems = 8U * 256U / sizeof(T);
        const uint32_t onesOffset = Align(offsetsOffset + offsetsElems * sizeof(uint32_t));

        LocalTensor<T> repeatedRow = Local<T>(repeatedOffset, packedElems);
        LocalTensor<T> scalar = Local<T>(scalarOffset, scalarBytes / sizeof(T));
        LocalTensor<T> expanded = Local<T>(expandedOffset, expandedElems);
        LocalTensor<T> packed = Local<T>(packedOffset, packedElems);
        LocalTensor<T> row = Local<T>(rowOffset, rowElems);
        LocalTensor<uint8_t> mask = Local<uint8_t>(maskOffset, maskBytes);
        LocalTensor<half> values = Local<half>(valueOffset, packedElems);
        LocalTensor<uint32_t> offsets = Local<uint32_t>(offsetsOffset, offsetsElems);
        LocalTensor<half> ones = Local<half>(onesOffset, packedElems);
        const uint32_t maxLength = Min(middle_, middleTile) * cols_;
        BuildOffsets(offsets, Min(maxLength, offsetsElems));
        Duplicate(ones, static_cast<half>(1.0f), packedElems);
        const uint32_t coreRows = outerEnd_ - outerStart_;
        const bool cacheRows = coreRows <= kCachedRows;
        if (cacheRows) {
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            CopyIn(row, rowGm_, outerStart_ * cols_, coreRows * cols_);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }

        for (uint32_t middleBase = 0; middleBase < middle_; middleBase += middleTile) {
            const uint32_t currentMiddle = Min(middle_ - middleBase, middleTile);
            const uint32_t length = currentMiddle * cols_;
            const uint32_t vectorLength = VectorLength<T>(length);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            CopyScalars(scalar, middleBase, currentMiddle);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            InitPattern(patternOffset);
            Brcb(expanded, scalar, static_cast<uint8_t>(currentMiddle / 8U), BrcbRepeatParams(1, 8));
            GatherScalars(packed, expanded, currentMiddle, patternOffset);

            for (uint32_t outer = outerStart_; outer < outerEnd_; ++outer) {
                const uint32_t localOuter = outer - outerStart_;
                const uint32_t outputIndex = localOuter & 1U;
                LocalTensor<T> currentRow = row;
                if (cacheRows) {
                    currentRow = row[(outer - outerStart_) * cols_];
                } else {
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    CopyIn(row, rowGm_, outer * cols_, cols_);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                }
                RepeatRow(repeatedRow, currentRow, offsets, length);
                if (middleIsX1_) {
                    Compare(mask, packed, repeatedRow, CMPMODE::GT, vectorLength);
                } else {
                    Compare(mask, repeatedRow, packed, CMPMODE::GT, vectorLength);
                }
                LocalTensor<uint8_t> currentOut = Local<uint8_t>(
                    outputOffset + outputIndex * outputStride, packedElems);
                if (localOuter >= 2U) WaitFlag<HardEvent::MTE3_V>(outputIndex);
                MaskToBool(values, mask, currentOut, ones, vectorLength);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                const uint32_t offset = (outer * middle_ + middleBase) * cols_;
                CopyOut(outGm_, currentOut, offset, length);
                SetFlag<HardEvent::MTE3_V>(outputIndex);
            }
            if (coreRows != 0U) WaitFlag<HardEvent::MTE3_V>(0);
            if (coreRows > 1U) WaitFlag<HardEvent::MTE3_V>(1);
        }
    }

    __aicore__ inline uint32_t Align(uint32_t bytes) const
    { return CeilDiv(bytes, kAlignBytes) * kAlignBytes; }

    __aicore__ inline void RepeatRow(LocalTensor<T>& repeated, LocalTensor<T>& row,
                                      LocalTensor<uint32_t>& offsets, uint32_t length)
    {
        const uint32_t block = 256U / sizeof(T);
        const uint32_t seed = Min(length, cols_ * block);
        Gather(repeated, row, offsets, 0U, seed);
        if (seed == length) return;
        PipeBarrier<PIPE_V>();
        const uint32_t copies = length / seed;
        if (copies > 1U) ExpandRows<T, true>(repeated, seed, copies);
        const uint32_t filled = copies * seed;
        if (filled < length) {
            CopyRepeatParams params = {1, 1, 8, 8};
            Copy(repeated[filled], repeated, block,
                 static_cast<uint8_t>(CeilDiv(length - filled, block)), params);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void BuildOffsets(LocalTensor<uint32_t>& offsets, uint32_t length)
    {
        uint32_t a = cols_;
        uint32_t b = 8U;
        while (b != 0U) {
            const uint32_t next = a % b;
            a = b;
            b = next;
        }
        const uint32_t period = cols_ / a * 8U;
        for (uint32_t i = 0; i < period; ++i) {
            offsets.SetValue(i, i % cols_ * sizeof(T));
        }
        PipeBarrier<PIPE_V>();
        if (length > period) ExpandRows<uint32_t, true>(offsets, period, length / period);
    }

    __aicore__ inline void RunRowVector()
    {
        const uint32_t tile = TileElements<T>();
        const uint32_t vectorAlign = 256U / sizeof(T);
        const bool canGroup = cols_ <= tile && cols_ % vectorAlign == 0U;
        const uint32_t group = canGroup ? Min(middle_, Min(32U, 65536U / cols_)) : 1U;
        const uint32_t outCapacity = canGroup ? group * cols_ : tile;
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> rowQueue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> maskBuf, valueBuf;
        pipe.InitBuffer(rowQueue, 1, tile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, outCapacity);
        pipe.InitBuffer(maskBuf, MaskBytes(tile));
        pipe.InitBuffer(valueBuf, tile * sizeof(half));
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valueBuf.Get<half>();

        for (uint32_t outer = outerStart_; outer < outerEnd_; ++outer) {
            for (uint32_t colBase = 0; colBase < cols_; colBase += tile) {
                const uint32_t length = Min(cols_ - colBase, tile);
                const uint32_t vectorLength = VectorLength<T>(length);
                LocalTensor<T> row = rowQueue.AllocTensor<T>();
                CopyIn(row, rowGm_, outer * cols_ + colBase, length);
                rowQueue.EnQue(row);
                row = rowQueue.DeQue<T>();
                const uint32_t middleGroup = length == cols_ ? group : 1U;
                for (uint32_t middleBase = 0; middleBase < middle_; middleBase += middleGroup) {
                    const uint32_t count = Min(middle_ - middleBase, middleGroup);
                    LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
                    for (uint32_t i = 0; i < count; ++i) {
                        const T scalar = middleGm_.GetValue(middleBase + i);
                        if (middleIsX1_) {
                            CompareScalar(mask, row, scalar, CMPMODE::LT, vectorLength);
                        } else {
                            CompareScalar(mask, row, scalar, CMPMODE::GT, vectorLength);
                        }
                        LocalTensor<uint8_t> slice = count == 1U ? out : out[i * vectorLength];
                        MaskToBool(values, mask, slice, vectorLength);
                    }
                    outQueue.EnQue(out);
                    out = outQueue.DeQue<uint8_t>();
                    const uint32_t offset = (outer * middle_ + middleBase) * cols_ + colBase;
                    CopyOut(outGm_, out, offset, count * length);
                    outQueue.FreeTensor(out);
                }
                rowQueue.FreeTensor(row);
            }
        }
    }

    __aicore__ inline void InitPattern(uint32_t patternOffset)
    {
        LocalTensor<uint8_t> bytes = Local<uint8_t>(patternOffset, 1024U);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> pattern = bytes.ReinterpretCast<uint16_t>();
            const uint16_t bits = cols_ == 16U ? static_cast<uint16_t>(0xFFFFU) :
                                  static_cast<uint16_t>((1U << cols_) - 1U);
            Duplicate(pattern, bits, 256U / sizeof(uint16_t));
        } else {
            LocalTensor<uint32_t> pattern = bytes.ReinterpretCast<uint32_t>();
            const uint32_t lanes = (1U << cols_) - 1U;
            const uint32_t bits = lanes | (lanes << 8U) | (lanes << 16U) | (lanes << 24U);
            Duplicate(pattern, bits, 256U / sizeof(uint32_t));
        }
    }

    __aicore__ inline void GatherScalars(LocalTensor<T>& packed, LocalTensor<T>& expanded,
                                          uint32_t currentMiddle, uint32_t patternOffset)
    {
        uint64_t gathered = 0;
        GatherMaskParams params = {1, static_cast<uint16_t>(currentMiddle / 8U), 8, 0};
        LocalTensor<uint8_t> bytes = Local<uint8_t>(patternOffset, 1024U);
        if constexpr (std::is_same_v<T, half>) {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint16_t>(), false, 0, params, gathered);
        } else {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint32_t>(), false, 0, params, gathered);
        }
    }

    __aicore__ inline void CopyScalars(LocalTensor<T>& local, uint32_t offset, uint32_t length)
    {
        const uint32_t bytes = length * sizeof(T);
        if ((offset * sizeof(T)) % kAlignBytes == 0U && bytes % kAlignBytes == 0U) {
            DataCopy(local, middleGm_[offset], length);
        } else {
            DataCopyExtParams copy = {1, bytes, 0, 0, 0};
            DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, middleGm_[offset], copy, pad);
        }
    }

    GlobalTensor<T> middleGm_, rowGm_;
    GlobalTensor<uint8_t> outGm_;
    uint32_t cols_, middle_, outer_, outerStart_, outerEnd_;
    bool middleIsX1_;
};

template <typename T>
class PromotedFullMiddleKernel {
    using C = CompareT<T>;
    static constexpr uint32_t kMiddleTile = std::is_same_v<T, bfloat16_t> ? 1024U : 768U;
    static constexpr uint32_t kBlockElems = kAlignBytes / sizeof(C);
    static constexpr uint32_t kLocalElems = kMiddleTile * kBlockElems;

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        const bool middleIsX1 = tiling.broadcastIsX1 != 0U;
        const uint32_t cols = tiling.outDims[0];
        const uint32_t middle = tiling.outDims[1];
        const uint32_t outer = tiling.outDims[2];
        if (middle % 8U != 0U || cols > kBlockElems ||
            (static_cast<uint64_t>(middle) * cols) % kAlignBytes != 0U) {
            RunFullMiddleRows<T>(x1, x2, output, tiling);
            return;
        }
        const uint32_t outerStart = outer * GetBlockIdx() / GetBlockNum();
        const uint32_t outerEnd = outer * (GetBlockIdx() + 1U) / GetBlockNum();
        if (outerStart >= outerEnd) return;

        GlobalTensor<T> fullGm, middleGm;
        GlobalTensor<uint8_t> outGm;
        fullGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x2 : x1));
        middleGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x1 : x2));
        outGm.SetGlobalBuffer((__gm__ uint8_t*)output);

        TPipe pipe;
        TQue<QuePosition::VECIN, 1> fullQueue, scalarQueue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> fullCastBuf, scalarCastBuf, expandedBuf, packedBuf;
        TBuf<QuePosition::VECCALC> patternBuf, maskBuf, valueBuf, maximumBuf;
        pipe.InitBuffer(fullQueue, 1, kLocalElems * sizeof(T));
        pipe.InitBuffer(scalarQueue, 1, kMiddleTile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kLocalElems);
        if constexpr (kNeedsCast<T>) {
            pipe.InitBuffer(fullCastBuf, kLocalElems * sizeof(C));
            pipe.InitBuffer(scalarCastBuf, kMiddleTile * sizeof(C));
        }
        pipe.InitBuffer(expandedBuf, kLocalElems * sizeof(C));
        pipe.InitBuffer(packedBuf, kLocalElems * sizeof(C));
        pipe.InitBuffer(patternBuf, 1024U);
        pipe.InitBuffer(maskBuf, MaskBytes(kLocalElems));
        pipe.InitBuffer(valueBuf, kLocalElems * sizeof(half));
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(maximumBuf, kLocalElems * sizeof(int32_t));
        }

        LocalTensor<C> expanded = expandedBuf.Get<C>();
        LocalTensor<C> packed = packedBuf.Get<C>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valueBuf.Get<half>();
        InitPattern(patternBuf.Get<uint8_t>(), cols);

        for (uint32_t middleBase = 0; middleBase < middle; middleBase += kMiddleTile) {
            const uint32_t currentMiddle = Min(middle - middleBase, kMiddleTile);
            LocalTensor<T> sourceScalar = scalarQueue.AllocTensor<T>();
            CopyIn(sourceScalar, middleGm, middleBase, currentMiddle);
            scalarQueue.EnQue(sourceScalar);
            sourceScalar = scalarQueue.DeQue<T>();
            LocalTensor<C> scalar;
            if constexpr (kNeedsCast<T>) {
                scalar = scalarCastBuf.Get<C>();
                Cast(scalar, sourceScalar, RoundMode::CAST_NONE, currentMiddle);
            } else {
                scalar = sourceScalar.template ReinterpretCast<C>();
            }
            Brcb(expanded, scalar, static_cast<uint8_t>(currentMiddle / 8U), BrcbRepeatParams(1, 8));
            GatherScalars(packed, expanded, patternBuf.Get<uint8_t>(), currentMiddle);

            const uint32_t length = currentMiddle * cols;
            const uint32_t vectorLength = VectorLength<C>(length);
            for (uint32_t outerIndex = outerStart; outerIndex < outerEnd; ++outerIndex) {
                const uint32_t offset = (outerIndex * middle + middleBase) * cols;
                LocalTensor<T> sourceFull = fullQueue.AllocTensor<T>();
                CopyIn(sourceFull, fullGm, offset, length);
                fullQueue.EnQue(sourceFull);
                sourceFull = fullQueue.DeQue<T>();
                LocalTensor<C> full;
                if constexpr (kNeedsCast<T>) {
                    full = fullCastBuf.Get<C>();
                    Cast(full, sourceFull, RoundMode::CAST_NONE, length);
                } else {
                    full = sourceFull.template ReinterpretCast<C>();
                }

                LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                    if (middleIsX1) Int32Compare<false>(mask, packed, full, maximum, out, vectorLength);
                    else Int32Compare<false>(mask, full, packed, maximum, out, vectorLength);
                } else if (middleIsX1) {
                    Compare(mask, packed, full, CMPMODE::GT, vectorLength);
                } else {
                    Compare(mask, full, packed, CMPMODE::GT, vectorLength);
                }
                MaskToBool(values, mask, out, vectorLength);
                outQueue.EnQue(out);
                fullQueue.FreeTensor(sourceFull);
                out = outQueue.DeQue<uint8_t>();
                CopyOut(outGm, out, offset, length);
                outQueue.FreeTensor(out);
            }
            scalarQueue.FreeTensor(sourceScalar);
        }
    }

private:
    __aicore__ inline void InitPattern(LocalTensor<uint8_t> bytes, uint32_t cols)
    {
        if constexpr (std::is_same_v<C, half>) {
            const uint16_t bits = cols == 16U ? 0xFFFFU : static_cast<uint16_t>((1U << cols) - 1U);
            Duplicate(bytes.ReinterpretCast<uint16_t>(), bits, 256U / sizeof(uint16_t));
        } else {
            const uint32_t lanes = (1U << cols) - 1U;
            Duplicate(bytes.ReinterpretCast<uint32_t>(), lanes * 0x01010101U, 256U / sizeof(uint32_t));
        }
    }

    __aicore__ inline void GatherScalars(LocalTensor<C>& packed, LocalTensor<C>& expanded,
                                         LocalTensor<uint8_t> bytes, uint32_t middle)
    {
        uint64_t gathered = 0;
        GatherMaskParams params = {1, static_cast<uint16_t>(middle / 8U), 8, 0};
        if constexpr (std::is_same_v<C, half>) {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint16_t>(), false, 0, params, gathered);
        } else {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint32_t>(), false, 0, params, gathered);
        }
    }
};

template <typename T>
class PromotedOuterMiddleKernel {
    using C = CompareT<T>;
    static constexpr uint32_t kMiddleTile = std::is_same_v<T, int8_t> ? 1376U : 1024U;
    static constexpr uint32_t kBlockElems = kAlignBytes / sizeof(C);
    static constexpr uint32_t kPackedElems = kMiddleTile * 8U;
    static constexpr uint32_t kCachedRows = 32U;

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        const bool middleIsX1 = tiling.broadcastIsX1 != 0U;
        const uint32_t cols = tiling.outDims[0];
        const uint32_t middle = tiling.outDims[1];
        const uint32_t outer = tiling.outDims[2];
        const uint32_t outerStart = outer * GetBlockIdx() / GetBlockNum();
        const uint32_t outerEnd = outer * (GetBlockIdx() + 1U) / GetBlockNum();
        if (outerStart >= outerEnd) return;

        GlobalTensor<T> middleGm, rowGm;
        GlobalTensor<uint8_t> outGm;
        middleGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x1 : x2));
        rowGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x2 : x1));
        outGm.SetGlobalBuffer((__gm__ uint8_t*)output);
        if (cols > 8U || middle % 8U != 0U) {
            RunRows(middleGm, rowGm, outGm, middleIsX1, cols, middle, outerStart, outerEnd);
            return;
        }

        TPipe pipe;
        TQue<QuePosition::VECIN, 1> scalarQueue, rowQueue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> scalarCastBuf, rowCastBuf, expandedBuf, packedBuf, repeatedBuf;
        TBuf<QuePosition::VECCALC> offsetsBuf, patternBuf, maskBuf, valueBuf, onesBuf, maximumBuf;
        pipe.InitBuffer(scalarQueue, 1, kMiddleTile * sizeof(T));
        pipe.InitBuffer(rowQueue, 1, kCachedRows * 8U * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kPackedElems);
        if constexpr (kNeedsCast<T>) {
            pipe.InitBuffer(scalarCastBuf, kMiddleTile * sizeof(C));
            pipe.InitBuffer(rowCastBuf, kCachedRows * 8U * sizeof(C));
        }
        pipe.InitBuffer(expandedBuf, kMiddleTile * kBlockElems * sizeof(C));
        pipe.InitBuffer(packedBuf, kPackedElems * sizeof(C));
        pipe.InitBuffer(repeatedBuf, kPackedElems * sizeof(C));
        constexpr uint32_t kOffsetElems = 8U * 256U / sizeof(C);
        pipe.InitBuffer(offsetsBuf, kOffsetElems * sizeof(uint32_t));
        pipe.InitBuffer(patternBuf, 1024U);
        pipe.InitBuffer(maskBuf, MaskBytes(kPackedElems));
        pipe.InitBuffer(valueBuf, kPackedElems * sizeof(half));
        pipe.InitBuffer(onesBuf, kPackedElems * sizeof(half));
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(maximumBuf, kPackedElems * sizeof(int32_t));
        }

        LocalTensor<C> expanded = expandedBuf.Get<C>();
        LocalTensor<C> packed = packedBuf.Get<C>();
        LocalTensor<C> repeated = repeatedBuf.Get<C>();
        LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
        LocalTensor<uint8_t> pattern = patternBuf.Get<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valueBuf.Get<half>();
        LocalTensor<half> ones = onesBuf.Get<half>();
        InitPattern(pattern, cols);
        BuildOffsets(offsets, cols, Min(Min(middle, kMiddleTile) * cols, kOffsetElems));
        Duplicate(ones, static_cast<half>(1.0f), kPackedElems);
        const uint32_t coreRows = outerEnd - outerStart;
        const bool cacheRows = coreRows <= kCachedRows;
        LocalTensor<T> cachedSource;
        LocalTensor<C> cachedRow;
        if (cacheRows) {
            cachedSource = rowQueue.AllocTensor<T>();
            CopyIn(cachedSource, rowGm, outerStart * cols, coreRows * cols);
            rowQueue.EnQue(cachedSource);
            cachedSource = rowQueue.DeQue<T>();
            if constexpr (kNeedsCast<T>) {
                cachedRow = rowCastBuf.Get<C>();
                Cast(cachedRow, cachedSource, RoundMode::CAST_NONE, coreRows * cols);
            } else {
                cachedRow = cachedSource.template ReinterpretCast<C>();
            }
        }

        for (uint32_t middleBase = 0; middleBase < middle; middleBase += kMiddleTile) {
            const uint32_t currentMiddle = Min(middle - middleBase, kMiddleTile);
            const uint32_t length = currentMiddle * cols;
            const uint32_t vectorLength = VectorLength<C>(length);
            LocalTensor<T> sourceScalar = scalarQueue.AllocTensor<T>();
            CopyIn(sourceScalar, middleGm, middleBase, currentMiddle);
            scalarQueue.EnQue(sourceScalar);
            sourceScalar = scalarQueue.DeQue<T>();
            LocalTensor<C> scalar;
            if constexpr (kNeedsCast<T>) {
                scalar = scalarCastBuf.Get<C>();
                Cast(scalar, sourceScalar, RoundMode::CAST_NONE, currentMiddle);
            } else {
                scalar = sourceScalar.template ReinterpretCast<C>();
            }
            Brcb(expanded, scalar, static_cast<uint8_t>(currentMiddle / 8U), BrcbRepeatParams(1, 8));
            GatherScalars(packed, expanded, pattern, currentMiddle);

            for (uint32_t outerIndex = outerStart; outerIndex < outerEnd; ++outerIndex) {
                LocalTensor<T> sourceRow;
                LocalTensor<C> row;
                if (cacheRows) {
                    row = cachedRow[(outerIndex - outerStart) * cols];
                } else {
                    sourceRow = rowQueue.AllocTensor<T>();
                    CopyIn(sourceRow, rowGm, outerIndex * cols, cols);
                    rowQueue.EnQue(sourceRow);
                    sourceRow = rowQueue.DeQue<T>();
                    if constexpr (kNeedsCast<T>) {
                        row = rowCastBuf.Get<C>();
                        Cast(row, sourceRow, RoundMode::CAST_NONE, cols);
                    } else {
                        row = sourceRow.template ReinterpretCast<C>();
                    }
                }
                RepeatRow(repeated, row, offsets, cols, length);

                LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                    if (middleIsX1) Int32Compare<false>(mask, packed, repeated, maximum, out, vectorLength);
                    else Int32Compare<false>(mask, repeated, packed, maximum, out, vectorLength);
                } else if (middleIsX1) {
                    Compare(mask, packed, repeated, CMPMODE::GT, vectorLength);
                } else {
                    Compare(mask, repeated, packed, CMPMODE::GT, vectorLength);
                }
                MaskToBool(values, mask, out, ones, vectorLength);
                outQueue.EnQue(out);
                if (!cacheRows) rowQueue.FreeTensor(sourceRow);
                out = outQueue.DeQue<uint8_t>();
                CopyOut(outGm, out, (outerIndex * middle + middleBase) * cols, length);
                outQueue.FreeTensor(out);
            }
            scalarQueue.FreeTensor(sourceScalar);
        }
        if (cacheRows) rowQueue.FreeTensor(cachedSource);
    }

private:
    __aicore__ inline void RepeatRow(LocalTensor<C>& repeated, LocalTensor<C>& row,
                                      LocalTensor<uint32_t>& offsets,
                                      uint32_t cols, uint32_t length)
    {
        const uint32_t block = 256U / sizeof(C);
        const uint32_t seed = Min(length, cols * block);
        Gather(repeated, row, offsets, 0U, seed);
        if (seed == length) return;
        PipeBarrier<PIPE_V>();
        const uint32_t copies = length / seed;
        if (copies > 1U) ExpandRows<C, true>(repeated, seed, copies);
        const uint32_t filled = copies * seed;
        if (filled < length) {
            CopyRepeatParams params = {1, 1, 8, 8};
            Copy(repeated[filled], repeated, block,
                 static_cast<uint8_t>(CeilDiv(length - filled, block)), params);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void RunRows(GlobalTensor<T>& middleGm, GlobalTensor<T>& rowGm,
                                   GlobalTensor<uint8_t>& outGm, bool middleIsX1,
                                   uint32_t cols, uint32_t middle,
                                   uint32_t outerStart, uint32_t outerEnd)
    {
        const uint32_t tile = TileElements<T>();
        if (static_cast<uint64_t>(outerEnd - outerStart) * cols * sizeof(T) > 1024U * 1024U) {
            rowGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> rowQueue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> castBuf, scalarInputBuf, scalarCastBuf;
        TBuf<QuePosition::VECCALC> maskBuf, valueBuf, maximumBuf, scalarBuf;
        pipe.InitBuffer(rowQueue, 1, tile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, tile);
        if constexpr (kNeedsCast<T>) {
            pipe.InitBuffer(castBuf, tile * sizeof(C));
            pipe.InitBuffer(scalarInputBuf, kAlignBytes);
            pipe.InitBuffer(scalarCastBuf, kAlignBytes);
        }
        pipe.InitBuffer(maskBuf, MaskBytes(tile));
        pipe.InitBuffer(valueBuf, tile * sizeof(half));
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(maximumBuf, tile * sizeof(int32_t));
            pipe.InitBuffer(scalarBuf, tile * sizeof(int32_t));
        }
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valueBuf.Get<half>();
        TEventID scalarToVector;
        TEventID vectorToScalar;
        if constexpr (kNeedsCast<T>) {
            scalarToVector = pipe.FetchEventID<HardEvent::S_V>();
            vectorToScalar = pipe.FetchEventID<HardEvent::V_S>();
        }

        for (uint32_t outer = outerStart; outer < outerEnd; ++outer) {
            for (uint32_t colBase = 0; colBase < cols; colBase += tile) {
                const uint32_t length = Min(cols - colBase, tile);
                const uint32_t vectorLength = VectorLength<C>(length);
                LocalTensor<T> source = rowQueue.AllocTensor<T>();
                CopyIn(source, rowGm, outer * cols + colBase, length);
                rowQueue.EnQue(source);
                source = rowQueue.DeQue<T>();
                LocalTensor<C> row;
                if constexpr (kNeedsCast<T>) {
                    row = castBuf.Get<C>();
                    Cast(row, source, RoundMode::CAST_NONE, length);
                } else {
                    row = source.template ReinterpretCast<C>();
                }

                for (uint32_t middleIndex = 0; middleIndex < middle; ++middleIndex) {
                    C scalar;
                    if constexpr (kNeedsCast<T>) {
                        LocalTensor<T> scalarInput = scalarInputBuf.Get<T>();
                        LocalTensor<C> scalarCast = scalarCastBuf.Get<C>();
                        scalarInput.SetValue(0, middleGm.GetValue(middleIndex));
                        SetFlag<HardEvent::S_V>(scalarToVector);
                        WaitFlag<HardEvent::S_V>(scalarToVector);
                        Cast(scalarCast, scalarInput, RoundMode::CAST_NONE, VectorLength<T>(1U));
                        SetFlag<HardEvent::V_S>(vectorToScalar);
                        WaitFlag<HardEvent::V_S>(vectorToScalar);
                        scalar = scalarCast.GetValue(0);
                    } else {
                        scalar = static_cast<C>(middleGm.GetValue(middleIndex));
                    }
                    LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
                    if constexpr (std::is_same_v<T, int32_t>) {
                        LocalTensor<int32_t> scalarVector = scalarBuf.Get<int32_t>();
                        LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                        Duplicate(scalarVector, scalar, vectorLength);
                        if (middleIsX1) Int32Compare<true>(mask, row, scalarVector, maximum, out, vectorLength);
                        else Int32Compare<false>(mask, row, scalarVector, maximum, out, vectorLength);
                    } else {
                        CompareScalar(mask, row, scalar,
                                      middleIsX1 ? CMPMODE::LT : CMPMODE::GT, vectorLength);
                    }
                    MaskToBool(values, mask, out, vectorLength);
                    outQueue.EnQue(out);
                    out = outQueue.DeQue<uint8_t>();
                    const uint32_t offset = (outer * middle + middleIndex) * cols + colBase;
                    CopyOut(outGm, out, offset, length);
                    outQueue.FreeTensor(out);
                }
                rowQueue.FreeTensor(source);
            }
        }
        if constexpr (kNeedsCast<T>) {
            pipe.ReleaseEventID<HardEvent::S_V>(scalarToVector);
            pipe.ReleaseEventID<HardEvent::V_S>(vectorToScalar);
        }
    }

    __aicore__ inline void InitPattern(LocalTensor<uint8_t> bytes, uint32_t cols)
    {
        if constexpr (std::is_same_v<C, half>) {
            const uint16_t bits = static_cast<uint16_t>((1U << cols) - 1U);
            Duplicate(bytes.ReinterpretCast<uint16_t>(), bits, 256U / sizeof(uint16_t));
        } else {
            const uint32_t lanes = (1U << cols) - 1U;
            Duplicate(bytes.ReinterpretCast<uint32_t>(), lanes * 0x01010101U, 256U / sizeof(uint32_t));
        }
    }

    __aicore__ inline void BuildOffsets(LocalTensor<uint32_t>& offsets, uint32_t cols, uint32_t length)
    {
        uint32_t a = cols;
        uint32_t b = 8U;
        while (b != 0U) {
            const uint32_t next = a % b;
            a = b;
            b = next;
        }
        const uint32_t period = cols / a * 8U;
        for (uint32_t i = 0; i < period; ++i) offsets.SetValue(i, i % cols * sizeof(C));
        PipeBarrier<PIPE_V>();
        if (length > period) ExpandRows<uint32_t, true>(offsets, period, length / period);
    }

    __aicore__ inline void GatherScalars(LocalTensor<C>& packed, LocalTensor<C>& expanded,
                                         LocalTensor<uint8_t> bytes, uint32_t middle)
    {
        uint64_t gathered = 0;
        GatherMaskParams params = {1, static_cast<uint16_t>(middle / 8U), 8, 0};
        if constexpr (std::is_same_v<C, half>) {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint16_t>(), false, 0, params, gathered);
        } else {
            GatherMask(packed, expanded, bytes.ReinterpretCast<uint32_t>(), false, 0, params, gathered);
        }
    }

};

template <typename T>
class CheckerboardKernel {
    using C = CompareT<T>;
    static constexpr uint32_t kMiddleTile = 64U;
    static constexpr uint32_t kBlockElems = kAlignBytes / sizeof(C);
    static constexpr uint32_t kColTile = 64U;
    static constexpr uint32_t kPackedElems = kMiddleTile * kColTile;

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        const bool middleIsX1 = tiling.broadcastIsX1 != 0U;
        const uint32_t cols = tiling.outDims[0];
        const uint32_t middle = tiling.outDims[1];
        const uint32_t rightOuter = tiling.outDims[2];
        const uint32_t leftOuter = tiling.outDims[3];
        const uint32_t pairCount = leftOuter * rightOuter;
        const uint32_t middleTiles = CeilDiv(middle, kMiddleTile);
        const uint32_t colTiles = CeilDiv(cols, kColTile);
        const uint32_t tasksPerPair = middleTiles * colTiles;
        const uint32_t taskCount = pairCount * tasksPerPair;
        const uint32_t taskStart = taskCount * GetBlockIdx() / GetBlockNum();
        const uint32_t taskEnd = taskCount * (GetBlockIdx() + 1U) / GetBlockNum();
        if (taskStart >= taskEnd) return;

        GlobalTensor<T> middleGm, rowGm;
        GlobalTensor<uint8_t> outGm;
        middleGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x1 : x2));
        rowGm.SetGlobalBuffer((__gm__ T*)(middleIsX1 ? x2 : x1));
        outGm.SetGlobalBuffer((__gm__ uint8_t*)output);

        TPipe pipe;
        TQue<QuePosition::VECIN, 1> scalarQueue, rowQueue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> scalarCastBuf, rowCastBuf, expandedBuf, repeatedBuf;
        TBuf<QuePosition::VECCALC> offsetsBuf, scalarOffsetsBuf, offsetTmpBuf;
        TBuf<QuePosition::VECCALC> maskBuf, valueBuf, maximumBuf;
        pipe.InitBuffer(scalarQueue, 1, kMiddleTile * sizeof(T));
        pipe.InitBuffer(rowQueue, 1, kColTile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kPackedElems);
        if constexpr (kNeedsCast<T>) {
            pipe.InitBuffer(scalarCastBuf, kMiddleTile * sizeof(C));
            pipe.InitBuffer(rowCastBuf, kColTile * sizeof(C));
        }
        pipe.InitBuffer(expandedBuf, kPackedElems * sizeof(C));
        pipe.InitBuffer(repeatedBuf, kPackedElems * sizeof(C));
        pipe.InitBuffer(offsetsBuf, kPackedElems * sizeof(uint32_t));
        pipe.InitBuffer(scalarOffsetsBuf, kPackedElems * sizeof(uint32_t));
        pipe.InitBuffer(offsetTmpBuf, kPackedElems * sizeof(float));
        pipe.InitBuffer(maskBuf, MaskBytes(kPackedElems));
        pipe.InitBuffer(valueBuf, kPackedElems * sizeof(half));
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(maximumBuf, kPackedElems * sizeof(int32_t));
        }

        LocalTensor<C> expanded = expandedBuf.Get<C>();
        LocalTensor<C> repeated = repeatedBuf.Get<C>();
        LocalTensor<uint32_t> offsets = offsetsBuf.Get<uint32_t>();
        LocalTensor<uint32_t> scalarOffsets = scalarOffsetsBuf.Get<uint32_t>();
        LocalTensor<float> offsetTmp = offsetTmpBuf.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valueBuf.Get<half>();
        const bool compactOutput = cols < kColTile;
        if (compactOutput) {
            BuildCompactOffsets(offsets, scalarOffsets, offsetTmp, cols,
                                tiling.inversePeriods[0]);
        }
        else BuildOffsets(offsets);

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t pair = task / tasksPerPair;
            const uint32_t tile = task % tasksPerPair;
            const uint32_t middleBase = tile / colTiles * kMiddleTile;
            const uint32_t colBase = tile % colTiles * kColTile;
            const uint32_t left = pair / rightOuter;
            const uint32_t right = pair % rightOuter;
            const uint32_t currentMiddle = Min(middle - middleBase, kMiddleTile);
            const uint32_t paddedMiddle = CeilDiv(currentMiddle, 8U) * 8U;
            const uint32_t currentCols = Min(cols - colBase, kColTile);
            const uint32_t packedLength = currentMiddle * currentCols;
            const uint32_t length = compactOutput ? VectorLength<C>(packedLength) : paddedMiddle * kColTile;

            LocalTensor<T> sourceScalar = scalarQueue.AllocTensor<T>();
            CopyIn(sourceScalar, middleGm, left * middle + middleBase, currentMiddle);
            scalarQueue.EnQue(sourceScalar);
            sourceScalar = scalarQueue.DeQue<T>();
            LocalTensor<C> scalar;
            if constexpr (kNeedsCast<T>) {
                scalar = scalarCastBuf.Get<C>();
                Cast(scalar, sourceScalar, RoundMode::CAST_NONE, currentMiddle);
            } else {
                scalar = sourceScalar.template ReinterpretCast<C>();
            }
            if (compactOutput) Gather(expanded, scalar, scalarOffsets, 0U, length);
            else BroadcastScalars(expanded, scalar, paddedMiddle);

            LocalTensor<T> sourceRow = rowQueue.AllocTensor<T>();
            CopyIn(sourceRow, rowGm, right * cols + colBase, currentCols);
            rowQueue.EnQue(sourceRow);
            sourceRow = rowQueue.DeQue<T>();
            LocalTensor<C> row;
            if constexpr (kNeedsCast<T>) {
                row = rowCastBuf.Get<C>();
                Cast(row, sourceRow, RoundMode::CAST_NONE, currentCols);
            } else {
                row = sourceRow.template ReinterpretCast<C>();
            }
            Gather(repeated, row, offsets, 0U, length);

            LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
            if constexpr (std::is_same_v<T, int32_t>) {
                LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                if (middleIsX1) Int32Compare<false>(mask, expanded, repeated, maximum, out, length);
                else Int32Compare<false>(mask, repeated, expanded, maximum, out, length);
            } else if (middleIsX1) {
                Compare(mask, expanded, repeated, CMPMODE::GT, length);
            } else {
                Compare(mask, repeated, expanded, CMPMODE::GT, length);
            }
            MaskToBool(values, mask, out, length);
            outQueue.EnQue(out);
            rowQueue.FreeTensor(sourceRow);
            scalarQueue.FreeTensor(sourceScalar);
            out = outQueue.DeQue<uint8_t>();
            const uint32_t offset = (pair * middle + middleBase) * cols + colBase;
            if (compactOutput) {
                CopyOut(outGm, out, offset, packedLength);
            } else {
                const uint32_t sourceStride = kColTile / kAlignBytes - CeilDiv(currentCols, kAlignBytes);
                DataCopyExtParams copy = {static_cast<uint16_t>(currentMiddle), currentCols,
                                          sourceStride, cols - currentCols, 0};
                DataCopyPad(outGm[offset], out, copy);
            }
            outQueue.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void BroadcastScalars(LocalTensor<C>& expanded, LocalTensor<C>& scalar,
                                             uint32_t middle)
    {
        constexpr uint32_t blocksPerRow = kColTile / kBlockElems;
        for (uint32_t block = 0; block < blocksPerRow; ++block) {
            Brcb(expanded[block * kBlockElems], scalar, static_cast<uint8_t>(middle / 8U),
                 BrcbRepeatParams(blocksPerRow, 8U * blocksPerRow));
        }
    }

    __aicore__ inline void BuildOffsets(LocalTensor<uint32_t>& offsets)
    {
        for (uint32_t i = 0; i < kColTile; ++i) offsets.SetValue(i, i * sizeof(C));
        PipeBarrier<PIPE_V>();
        ExpandRows<uint32_t, true>(offsets, kColTile, kMiddleTile);
    }

    __aicore__ inline void BuildCompactOffsets(LocalTensor<uint32_t>& rowOffsets,
                                                LocalTensor<uint32_t>& scalarOffsets,
                                                LocalTensor<float>& tmp,
                                                uint32_t cols, float inverseCols)
    {
        const uint32_t length = kMiddleTile * cols;
        Duplicate(rowOffsets, 0U, kPackedElems);
        Duplicate(scalarOffsets, 0U, kPackedElems);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> index = rowOffsets.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> quotient = scalarOffsets.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> rowStart = tmp.ReinterpretCast<int32_t>();
        CreateVecIndex(index, 0, length);
        PipeBarrier<PIPE_V>();
        Cast(tmp, index, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        Muls(tmp, tmp, inverseCols, length);
        PipeBarrier<PIPE_V>();
        Cast(quotient, tmp, RoundMode::CAST_FLOOR, length);
        PipeBarrier<PIPE_V>();
        Muls(rowStart, quotient, static_cast<int32_t>(cols), length);
        Muls(quotient, quotient, static_cast<int32_t>(sizeof(C)), length);
        PipeBarrier<PIPE_V>();
        Sub(index, index, rowStart, length);
        PipeBarrier<PIPE_V>();
        Muls(index, index, static_cast<int32_t>(sizeof(C)), length);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T>
class CachedBroadcastKernel {
    using C = CompareT<T>;
    static constexpr uint32_t kTile =
        std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int32_t> ? 2048U : 4096U;

public:
    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                               const GreaterTilingData& tiling)
    {
        const uint32_t x1Elements = Elements(tiling.x1Dims, tiling.rank);
        const uint32_t x2Elements = Elements(tiling.x2Dims, tiling.rank);
        const uint32_t taskStart = tiling.highSize * GetBlockIdx() / GetBlockNum();
        const uint32_t taskEnd = tiling.highSize * (GetBlockIdx() + 1U) / GetBlockNum();
        if (taskStart >= taskEnd) return;

        GlobalTensor<T> x1Gm, x2Gm;
        GlobalTensor<uint8_t> outGm;
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        outGm.SetGlobalBuffer((__gm__ uint8_t*)output);

        TPipe pipe;
        TQue<QuePosition::VECIN, 1> x1Queue, x2Queue;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<QuePosition::VECCALC> x1CastBuf, x2CastBuf, gathered1Buf, gathered2Buf;
        TBuf<QuePosition::VECCALC> offsets1Buf, offsets2Buf, indexBuf, quotientBuf;
        TBuf<QuePosition::VECCALC> periodBuf, tempBuf, maskBuf, valuesBuf, maximumBuf;
        pipe.InitBuffer(x1Queue, 1, VectorLength<T>(x1Elements) * sizeof(T));
        pipe.InitBuffer(x2Queue, 1, VectorLength<T>(x2Elements) * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kTile);
        if constexpr (kNeedsCast<T>) {
            pipe.InitBuffer(x1CastBuf, VectorLength<C>(x1Elements) * sizeof(C));
            pipe.InitBuffer(x2CastBuf, VectorLength<C>(x2Elements) * sizeof(C));
        }
        pipe.InitBuffer(gathered1Buf, kTile * sizeof(C));
        pipe.InitBuffer(gathered2Buf, kTile * sizeof(C));
        pipe.InitBuffer(offsets1Buf, kTile * sizeof(uint32_t));
        pipe.InitBuffer(offsets2Buf, kTile * sizeof(uint32_t));
        pipe.InitBuffer(indexBuf, kTile * sizeof(int32_t));
        pipe.InitBuffer(quotientBuf, kTile * sizeof(int32_t));
        pipe.InitBuffer(periodBuf, kTile * sizeof(int32_t));
        pipe.InitBuffer(tempBuf, kTile * sizeof(float));
        pipe.InitBuffer(maskBuf, MaskBytes(kTile));
        pipe.InitBuffer(valuesBuf, kTile * sizeof(half));
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(maximumBuf, kTile * sizeof(int32_t));
        }

        LocalTensor<T> source1 = x1Queue.AllocTensor<T>();
        LocalTensor<T> source2 = x2Queue.AllocTensor<T>();
        CopyIn(source1, x1Gm, 0U, x1Elements);
        CopyIn(source2, x2Gm, 0U, x2Elements);
        x1Queue.EnQue(source1);
        x2Queue.EnQue(source2);
        source1 = x1Queue.DeQue<T>();
        source2 = x2Queue.DeQue<T>();
        LocalTensor<C> cached1;
        LocalTensor<C> cached2;
        if constexpr (kNeedsCast<T>) {
            cached1 = x1CastBuf.Get<C>();
            cached2 = x2CastBuf.Get<C>();
            Cast(cached1, source1, RoundMode::CAST_NONE, VectorLength<T>(x1Elements));
            Cast(cached2, source2, RoundMode::CAST_NONE, VectorLength<T>(x2Elements));
            PipeBarrier<PIPE_V>();
        } else {
            cached1 = source1.template ReinterpretCast<C>();
            cached2 = source2.template ReinterpretCast<C>();
        }

        LocalTensor<C> gathered1 = gathered1Buf.Get<C>();
        LocalTensor<C> gathered2 = gathered2Buf.Get<C>();
        LocalTensor<uint32_t> offsets1 = offsets1Buf.Get<uint32_t>();
        LocalTensor<uint32_t> offsets2 = offsets2Buf.Get<uint32_t>();
        LocalTensor<int32_t> index = indexBuf.Get<int32_t>();
        LocalTensor<int32_t> quotient = quotientBuf.Get<int32_t>();
        LocalTensor<int32_t> period = periodBuf.Get<int32_t>();
        LocalTensor<float> temp = tempBuf.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> values = valuesBuf.Get<half>();

        for (uint32_t tileBase = 0; tileBase < tiling.lowSize; tileBase += kTile) {
            const uint32_t length = Min(tiling.lowSize - tileBase, kTile);
            const uint32_t computeLength = VectorLength<C>(length);
            CreateVecIndex(index, static_cast<int32_t>(tileBase), computeLength);
            PipeBarrier<PIPE_V>();
            BuildOffsets(offsets1, index, quotient, period, temp, tiling.x1Dims,
                         tiling.outDims, tiling.inverseStrides, tiling.inversePeriods,
                         tiling.middleSize, computeLength);
            BuildOffsets(offsets2, index, quotient, period, temp, tiling.x2Dims,
                         tiling.outDims, tiling.inverseStrides, tiling.inversePeriods,
                         tiling.middleSize, computeLength);

            uint32_t previousBase1 = UINT32_MAX;
            uint32_t previousBase2 = UINT32_MAX;
            for (uint32_t task = taskStart; task < taskEnd; ++task) {
                const uint32_t base1 = InputBase(task, tiling.x1Dims, tiling.outDims,
                                                 tiling.middleSize, tiling.rank);
                const uint32_t base2 = InputBase(task, tiling.x2Dims, tiling.outDims,
                                                 tiling.middleSize, tiling.rank);
                if (base1 != previousBase1) {
                    Gather(gathered1, cached1[base1], offsets1, 0U, computeLength);
                    previousBase1 = base1;
                }
                if (base2 != previousBase2) {
                    Gather(gathered2, cached2[base2], offsets2, 0U, computeLength);
                    previousBase2 = base2;
                }
                LocalTensor<uint8_t> out = outQueue.AllocTensor<uint8_t>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    LocalTensor<int32_t> maximum = maximumBuf.Get<int32_t>();
                    Int32Compare<false>(mask, gathered1, gathered2, maximum, out, computeLength);
                } else {
                    Compare(mask, gathered1, gathered2, CMPMODE::GT, computeLength);
                }
                MaskToBool(values, mask, out, computeLength);
                outQueue.EnQue(out);
                out = outQueue.DeQue<uint8_t>();
                CopyOut(outGm, out, task * tiling.lowSize + tileBase, length);
                outQueue.FreeTensor(out);
            }
        }
        x2Queue.FreeTensor(source2);
        x1Queue.FreeTensor(source1);
    }

private:
    __aicore__ inline uint32_t Elements(const uint32_t* dims, uint32_t rank)
    {
        uint32_t result = 1U;
        for (uint32_t axis = 0; axis < rank; ++axis) result *= dims[axis];
        return result;
    }

    __aicore__ inline uint32_t InputBase(uint32_t task, const uint32_t* dims,
                                         const uint32_t* out, uint32_t innerRank,
                                         uint32_t rank)
    {
        uint32_t stride = 1U;
        for (uint32_t axis = 0; axis < innerRank; ++axis) stride *= dims[axis];
        uint32_t base = 0U;
        for (uint32_t axis = innerRank; axis < rank; ++axis) {
            const uint32_t coordinate = task % out[axis];
            task /= out[axis];
            if (dims[axis] != 1U) base += coordinate * stride;
            stride *= dims[axis];
        }
        return base;
    }

    __aicore__ inline void BuildOffsets(LocalTensor<uint32_t>& offsets,
                                         LocalTensor<int32_t>& index,
                                         LocalTensor<int32_t>& quotient,
                                         LocalTensor<int32_t>& period,
                                         LocalTensor<float>& temp,
                                         const uint32_t* dims, const uint32_t* out,
                                         const float* inverseStrides,
                                         const float* inversePeriods,
                                         uint32_t innerRank, uint32_t length)
    {
        LocalTensor<int32_t> result = offsets.ReinterpretCast<int32_t>();
        Duplicate(result, 0, length);
        PipeBarrier<PIPE_V>();
        uint32_t outStride = 1U;
        uint32_t inputStride = 1U;
        for (uint32_t axis = 0; axis < innerRank; ++axis) {
            if (dims[axis] != 1U) {
                Cast(temp, index, RoundMode::CAST_NONE, length);
                PipeBarrier<PIPE_V>();
                Muls(temp, temp, inverseStrides[axis], length);
                PipeBarrier<PIPE_V>();
                Cast(quotient, temp, RoundMode::CAST_FLOOR, length);
                PipeBarrier<PIPE_V>();
                Cast(temp, index, RoundMode::CAST_NONE, length);
                PipeBarrier<PIPE_V>();
                Muls(temp, temp, inversePeriods[axis], length);
                PipeBarrier<PIPE_V>();
                Cast(period, temp, RoundMode::CAST_FLOOR, length);
                PipeBarrier<PIPE_V>();
                Muls(period, period, static_cast<int32_t>(out[axis]), length);
                PipeBarrier<PIPE_V>();
                Sub(quotient, quotient, period, length);
                PipeBarrier<PIPE_V>();
                Muls(quotient, quotient,
                     static_cast<int32_t>(inputStride * sizeof(C)), length);
                PipeBarrier<PIPE_V>();
                Add(result, result, quotient, length);
                PipeBarrier<PIPE_V>();
            }
            outStride *= out[axis];
            inputStride *= dims[axis];
        }
    }
};

__aicore__ inline uint32_t BroadcastCount(const uint32_t* dims, const uint32_t* out, uint32_t rank)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        count += dims[i] != out[i];
    }
    return count;
}

template <typename T>
__aicore__ inline void DoBroadcast(GM_ADDR input, GM_ADDR scratch, GM_ADDR final,
                                    const uint32_t* dims, const uint32_t* outDims, uint32_t rank,
                                    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>& queue)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t dataBlock = 512U / sizeof(T);
    const uint32_t maxTile = kWorkspaceTileBytes / sizeof(T);
    uint32_t remaining = BroadcastCount(dims, outDims, rank);
    GM_ADDR source = input;
    GM_ADDR target = remaining % 2U == 0U ? scratch : final;
    uint32_t lowSize = 1;

    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (dims[axis] != outDims[axis]) {
            uint32_t highSize = 1;
            for (uint32_t i = axis + 1U; i < rank; ++i) {
                highSize *= dims[i];
            }
            uint32_t highFrom = 0;
            uint32_t highTo = highSize;
            uint32_t lowFrom = 0;
            uint32_t lowTo = lowSize;
            if (lowSize > maxTile) {
                const uint32_t lowBlocks = CeilDiv(lowSize, dataBlock);
                lowFrom = lowBlocks * blockIdx / blockNum * dataBlock;
                lowTo = Min(lowSize, lowBlocks * (blockIdx + 1U) / blockNum * dataBlock);
            } else {
                highFrom = highSize * blockIdx / blockNum;
                highTo = highSize * (blockIdx + 1U) / blockNum;
            }

            GlobalTensor<T> sourceGm, targetGm;
            sourceGm.SetGlobalBuffer((__gm__ T*)source);
            targetGm.SetGlobalBuffer((__gm__ T*)target);
            for (uint32_t high = highFrom; high < highTo; ++high) {
                for (uint32_t low = lowFrom; low < lowTo; low += maxTile) {
                    const uint32_t length = Min(lowTo - low, maxTile);
                    LocalTensor<T> local = queue.AllocTensor<T>();
                    DataCopyExtParams copy = {1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
                    DataCopyPad(local, sourceGm[high * lowSize + low], copy, pad);
                    queue.EnQue(local);
                    local = queue.DeQue<T>();
                    for (uint32_t repeat = 0; repeat < outDims[axis]; ++repeat) {
                        DataCopyPad(targetGm[(high * outDims[axis] + repeat) * lowSize + low], local, copy);
                    }
                    queue.FreeTensor(local);
                }
            }
            if (blockNum > 1U) {
                CrossCoreSetFlag<0, PIPE_MTE3>(0);
                CrossCoreWaitFlag(0);
            }
            source = target;
            --remaining;
            target = remaining % 2U == 0U ? scratch : final;
        }
        lowSize *= outDims[axis];
    }
}

template <typename T>
__aicore__ inline void RunWorkspace(GM_ADDR x1, GM_ADDR x2, GM_ADDR output, GM_ADDR workspace,
                                     const GreaterTilingData& tiling)
{
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queue;
    pipe.InitBuffer(queue, 1, kWorkspaceTileBytes);
    GM_ADDR fullX1 = x1;
    GM_ADDR fullX2 = x2;

    if (tiling.broadcastOffsets[0] >= 0) {
        GM_ADDR final = workspace + tiling.broadcastOffsets[0];
        GM_ADDR scratch = tiling.broadcastOffsets[2] >= 0 ? workspace + tiling.broadcastOffsets[2] : final;
        DoBroadcast<T>(x1, scratch, final, tiling.x1Dims, tiling.outDims, tiling.rank, queue);
        fullX1 = final;
    }
    if (tiling.broadcastOffsets[1] >= 0) {
        GM_ADDR final = workspace + tiling.broadcastOffsets[1];
        GM_ADDR scratch = tiling.broadcastOffsets[2] >= 0 ? workspace + tiling.broadcastOffsets[2] : final;
        DoBroadcast<T>(x2, scratch, final, tiling.x2Dims, tiling.outDims, tiling.rank, queue);
        fullX2 = final;
    }
    pipe.Destroy();
    RunLinearSimple<T>(fullX1, fullX2, output, tiling.totalLength);
}

}

extern "C" __global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR output,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            if (static_cast<uint64_t>(tilingData.totalLength) * sizeof(DTYPE_X1) >
                kLargeStreamBytes) {
                RunLargeInt8Linear(x1, x2, output, tilingData.totalLength);
            } else {
                RunLinear<DTYPE_X1>(x1, x2, output, tilingData.totalLength);
            }
        } else if constexpr (std::is_same_v<DTYPE_X1, float>) {
            if (static_cast<uint64_t>(tilingData.totalLength) * sizeof(DTYPE_X1) >
                kLargeStreamBytes) {
                RunLargeFp32Linear(x1, x2, output, tilingData.totalLength);
            } else {
                RunLinear<DTYPE_X1, true>(x1, x2, output, tilingData.totalLength);
            }
        } else {
            RunLinear<DTYPE_X1>(x1, x2, output, tilingData.totalLength);
        }
    } else if (TILING_KEY_IS(1)) {
        RunSingleAxis<DTYPE_X1>(x1, x2, output, tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIV_1_0);
        RunWorkspace<DTYPE_X1>(x1, x2, output, workspace, tilingData);
    } else if (TILING_KEY_IS(3)) {
        if constexpr (kNeedsCast<DTYPE_X1> || std::is_same_v<DTYPE_X1, int32_t>) {
            PromotedFullMiddleKernel<DTYPE_X1> kernel;
            kernel.Run(x1, x2, output, tilingData);
        } else {
            const bool packed = tilingData.outDims[1] % 8U == 0U &&
                tilingData.outDims[0] <= kAlignBytes / sizeof(DTYPE_X1) &&
                (static_cast<uint64_t>(tilingData.outDims[1]) * tilingData.outDims[0]) % kAlignBytes == 0U;
            if (packed) {
                InitSocState();
                FullMiddleKernel<DTYPE_X1> kernel;
                kernel.Run(x1, x2, output, tilingData);
            } else {
                RunFullMiddleRows<DTYPE_X1>(x1, x2, output, tilingData);
            }
        }
    } else if (TILING_KEY_IS(4)) {
        if (tilingData.rank == 2U || tilingData.rank == 4U) {
            CheckerboardKernel<DTYPE_X1> kernel;
            kernel.Run(x1, x2, output, tilingData);
        } else if constexpr (kNeedsCast<DTYPE_X1> || std::is_same_v<DTYPE_X1, int32_t>) {
            PromotedOuterMiddleKernel<DTYPE_X1> kernel;
            kernel.Run(x1, x2, output, tilingData);
        } else {
            InitSocState();
            OuterMiddleKernel<DTYPE_X1> kernel;
            kernel.Run(x1, x2, output, tilingData);
        }
    } else if (TILING_KEY_IS(5)) {
        CachedBroadcastKernel<DTYPE_X1> kernel;
        kernel.Run(x1, x2, output, tilingData);
    }
}
