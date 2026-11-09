#ifndef __GREATER_FP32_RING_H__
#define __GREATER_FP32_RING_H__

namespace NsGreater {

class GreaterFp32TBufRing {
    static constexpr uint32_t TILE_ELEMENTS = 23808;
    static constexpr uint32_t FP32_BLOCK_ELEMENTS =
        DATA_BLOCK_BYTES / sizeof(float);
    static constexpr uint32_t PREDICATE_ELEMENTS = TILE_ELEMENTS / 8;
    static constexpr uint32_t HALF_REPEAT_ELEMENTS =
        VECTOR_ALIGN_BYTES / sizeof(half);

public:
    __aicore__ inline GreaterFp32TBufRing() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
        uint64_t ubPartDataNum, uint64_t bigClassCount)
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (bigClassCount != 0 && blockIdx < bigClassCount) {
            coreDataNum = bigCoreDataNum;
            blockOffset = blockIdx * bigCoreDataNum;
        } else {
            coreDataNum = smallCoreDataNum;
            blockOffset = bigClassCount * bigCoreDataNum +
                (blockIdx - bigClassCount) * smallCoreDataNum;
        }
        if (ubPartDataNum != TILE_ELEMENTS) {
            coreDataNum = 0;
        }

        x1Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(x1) + blockOffset,
            coreDataNum);
        x2Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(x2) + blockOffset,
            coreDataNum);
        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(y) + blockOffset,
            coreDataNum);

        pipe.InitBuffer(x1OutputBuf, TILE_ELEMENTS * sizeof(float));
        pipe.InitBuffer(
            x2Fp16Buf,
            TILE_ELEMENTS * sizeof(float) + VECTOR_ALIGN_BYTES);
        pipe.InitBuffer(predicateBuf, PREDICATE_ELEMENTS);
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0) {
            return;
        }
        AllocateEvents();
        InitialLoad();
        InitPredicateConstants();

        const uint64_t tileCount =
            (coreDataNum + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        for (uint64_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t logicalLength = TileLength(tile);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                static_cast<event_t>(inputReady));
            Compute(logicalLength);

            const uint64_t nextTile = tile + 1;
            if (nextTile < tileCount) {
                Reload(tile, logicalLength, nextTile);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    static_cast<event_t>(x2Reusable));
                Store(tile, logicalLength);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                    static_cast<event_t>(x1Reusable));
            }
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline void AllocateEvents()
    {
        inputReady = pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        x2Reusable = pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
        outputReady = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        x1Reusable = pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void ReleaseEvents()
    {
        pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(x2Reusable);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(x1Reusable);
    }

    __aicore__ inline uint32_t TileLength(uint64_t tile) const
    {
        const uint64_t offset = tile * TILE_ELEMENTS;
        const uint64_t remaining = coreDataNum - offset;
        return static_cast<uint32_t>(
            remaining < TILE_ELEMENTS ? remaining : TILE_ELEMENTS);
    }

    __aicore__ inline void CopyInput(
        AscendC::LocalTensor<float> destination,
        AscendC::GlobalTensor<float>& source,
        uint64_t sourceOffset, uint32_t logicalLength)
    {
        const uint32_t alignedLength =
            logicalLength / FP32_BLOCK_ELEMENTS * FP32_BLOCK_ELEMENTS;
        if (alignedLength != 0) {
            AscendC::DataCopy(
                destination, source[sourceOffset], alignedLength);
        }
        if (alignedLength == logicalLength) {
            return;
        }
        AscendC::DataCopyParams tailParams;
        tailParams.blockCount = 1;
        tailParams.blockLen = static_cast<uint16_t>(
            (logicalLength - alignedLength) * sizeof(float));
        tailParams.srcStride = 0;
        tailParams.dstStride = 0;
        AscendC::DataCopyPad(
            destination[alignedLength],
            source[sourceOffset + alignedLength], tailParams,
            {false, 0, 0, 0});
    }

    __aicore__ inline void CopyOutput(
        uint64_t destinationOffset,
        AscendC::LocalTensor<int8_t> source,
        uint32_t logicalLength)
    {
        const uint32_t alignedLength =
            logicalLength / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        if (alignedLength != 0) {
            AscendC::DataCopy(
                yGm[destinationOffset], source, alignedLength);
        }
        if (alignedLength == logicalLength) {
            return;
        }
        AscendC::DataCopyParams tailParams;
        tailParams.blockCount = 1;
        tailParams.blockLen = static_cast<uint16_t>(
            logicalLength - alignedLength);
        tailParams.srcStride = 0;
        tailParams.dstStride = 0;
        AscendC::DataCopyPad(
            yGm[destinationOffset + alignedLength],
            source[alignedLength], tailParams);
    }

    __aicore__ inline void InitialLoad()
    {
        AscendC::LocalTensor<float> x1 = x1OutputBuf.Get<float>();
        AscendC::LocalTensor<float> x2 = x2Fp16Buf.Get<float>();
        const uint32_t logicalLength = TileLength(0);
        CopyInput(x2, x2Gm, 0, logicalLength);
        CopyInput(x1, x1Gm, 0, logicalLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            static_cast<event_t>(inputReady));
    }

    __aicore__ inline void Reload(
        uint64_t currentTile, uint32_t currentLength,
        uint64_t nextTile)
    {
        const uint64_t nextOffset = nextTile * TILE_ELEMENTS;
        const uint32_t nextLength = TileLength(nextTile);
        AscendC::LocalTensor<float> x1 = x1OutputBuf.Get<float>();
        AscendC::LocalTensor<float> x2 = x2Fp16Buf.Get<float>();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(x2Reusable));
        CopyInput(x2, x2Gm, nextOffset, nextLength);
        Store(currentTile, currentLength);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            static_cast<event_t>(x1Reusable));
        CopyInput(x1, x1Gm, nextOffset, nextLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            static_cast<event_t>(inputReady));
    }

    __aicore__ inline void Compute(uint32_t logicalLength)
    {
        const uint32_t computeLength =
            (logicalLength + 63) / 64 * 64;
        AscendC::LocalTensor<float> x1 = x1OutputBuf.Get<float>();
        AscendC::LocalTensor<float> x2 = x2Fp16Buf.Get<float>();
        AscendC::LocalTensor<int8_t> output = x1OutputBuf.Get<int8_t>();
        AscendC::LocalTensor<half> result = x2Fp16Buf.Get<half>();
        AscendC::LocalTensor<half> oneBlock = result[2 * TILE_ELEMENTS];
        AscendC::LocalTensor<int8_t> predicate = predicateBuf.Get<int8_t>();

        AscendC::Compare(
            predicate, x1, x2, AscendC::CMPMODE::GT, computeLength);
        AscendC::PipeBarrier<PIPE_V>();
        MaterializePreloadedInt8Predicate(
            output, predicate, result, oneBlock, computeLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(x2Reusable));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
            static_cast<event_t>(outputReady));
    }

    __aicore__ inline void Store(
        uint64_t tile, uint32_t logicalLength)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            static_cast<event_t>(outputReady));
        AscendC::LocalTensor<int8_t> output = x1OutputBuf.Get<int8_t>();
        CopyOutput(tile * TILE_ELEMENTS, output, logicalLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            static_cast<event_t>(x1Reusable));
    }

    __aicore__ inline void InitPredicateConstants()
    {
        AscendC::LocalTensor<half> result = x2Fp16Buf.Get<half>();
        AscendC::LocalTensor<half> oneBlock = result[2 * TILE_ELEMENTS];
        AscendC::Duplicate<half>(
            oneBlock, (half)0, HALF_REPEAT_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(oneBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(
            oneBlock, (half)1, HALF_REPEAT_ELEMENTS);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x1OutputBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2Fp16Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> predicateBuf;
    AscendC::GlobalTensor<float> x1Gm;
    AscendC::GlobalTensor<float> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::TEventID inputReady;
    AscendC::TEventID x2Reusable;
    AscendC::TEventID outputReady;
    AscendC::TEventID x1Reusable;
    uint64_t coreDataNum = 0;
    uint64_t blockOffset = 0;
};

} // namespace NsGreater

#endif // __GREATER_FP32_RING_H__
