#ifndef __GREATER_INT32_PACKED_GM_H__
#define __GREATER_INT32_PACKED_GM_H__

#include "greater.h"

namespace NsGreater {

// Each AIV compares its entire contiguous partition into a private bit-packed
// GM interval, then expands that same interval. Both phases are one invocation.
// Admission is structural and capacity-based; no tensor values are inspected.
class GreaterInt32PackedGm {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        GM_ADDR workspace, const GreaterInt32DirectTilingData& data)
    {
        rowLength = data.rowLength;
        piecesPerGroup = data.rowCount / 8;
        paddedRow = (rowLength + 31) / 32 * 32;
        elements = 8 * paddedRow;
        logicalElements = 8 * rowLength;
        x1Rows = data.x1Rows != 0;
        const uint64_t pieces = data.naturalGroupCount * piecesPerGroup;
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t q = pieces / blocks;
        const uint64_t rem = pieces % blocks;
        firstPiece = block * q + (block < rem ? block : rem);
        endPiece = firstPiece + q + (block < rem ? 1 : 0);
        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        // The official generated wrapper already passes user workspace.
        packedGm.SetGlobalBuffer((__gm__ int8_t*)workspace);
        if (x1Rows) {
            x2Gm.SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            x1Gm.SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        pipe.InitBuffer(rawBuf, 16 * elements);
        pipe.InitBuffer(broadcastBuf, 4 * elements);
        pipe.InitBuffer(rowBuf, 4 * paddedRow);
        pipe.InitBuffer(packedBuf, 16 * paddedRow);
        pipe.InitBuffer(predicateBuf, 2 * paddedRow);
        pipe.InitBuffer(constantsBuf, 256);
    }

    __aicore__ inline void Process()
    {
        if (firstPiece == endPiece) { return; }
        rowReady = pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        rowReuse = pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
        transition = pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        for (uint32_t s = 0; s < 2; ++s) {
            inputReady[s] = pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
            inputReuse[s] = pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
            outputReady[s] = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
            outputReuse[s] = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        }
        AscendC::Duplicate<int32_t>(rowBuf.Get<int32_t>(), 0, paddedRow);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(rowReuse));
        ComparePartition();
        DrainInputOutput();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(rowReuse));
        // No cross-AIV read exists. Order this AIV's GM writes before its reads.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(transition));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(transition));
        InitConstants();
        ExpandPartition();
        DrainInputOutput();
        for (uint32_t s = 0; s < 2; ++s) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady[s]);
            pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputReuse[s]);
            pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady[s]);
            pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(outputReuse[s]);
        }
        pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(rowReady);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(rowReuse);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(transition);
    }

private:
    __aicore__ inline uint64_t MacroCount(uint64_t first)
    {
        const uint64_t groupRemaining = piecesPerGroup - first % piecesPerGroup;
        const uint64_t remaining = endPiece - first;
        const uint64_t count = groupRemaining < remaining ? groupRemaining : remaining;
        return count < 2 ? count : 2;
    }

    __aicore__ inline void WaitInputReuse(uint32_t slot)
    {
        if (inputPending[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(inputReuse[slot]));
            inputPending[slot] = false;
        }
    }

    __aicore__ inline void WaitOutputReuse(uint32_t slot)
    {
        if (outputPending[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(outputReuse[slot]));
            outputPending[slot] = false;
        }
    }

    __aicore__ inline void DrainInputOutput()
    {
        for (uint32_t s = 0; s < 2; ++s) {
            WaitInputReuse(s);
            WaitOutputReuse(s);
        }
    }

    __aicore__ inline void LoadRaw(uint32_t slot, uint64_t first, uint64_t count)
    {
        WaitInputReuse(slot);
        AscendC::GlobalTensor<int32_t>& stream = x1Rows ? x2Gm : x1Gm;
        auto raw = rawBuf.Get<int32_t>()[slot * 2 * elements];
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * logicalElements * sizeof(int32_t));
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPad(raw, stream[first * logicalElements], params,
            AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputReady[slot]));
    }

    __aicore__ inline void CacheRow(uint64_t group)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(rowReuse));
        AscendC::GlobalTensor<int32_t>& rowGm = x1Rows ? x1Gm : x2Gm;
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint16_t>(rowLength * sizeof(int32_t));
        params.srcGap = 0;
        params.dstGap = 0;
        AscendC::DataCopyPad(rowBuf.Get<int32_t>(), rowGm[group * rowLength],
            params, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(rowReady));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(rowReady));
        auto tile = broadcastBuf.Get<int32_t>();
        const uint32_t dstShape[2] = {8, static_cast<uint32_t>(paddedRow)};
        const uint32_t srcShape[2] = {1, static_cast<uint32_t>(paddedRow)};
        AscendC::Broadcast<int32_t, 2, 0>(tile, rowBuf.Get<int32_t>(), dstShape, srcShape);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMaskParams gather(1, 8,
            static_cast<uint16_t>(paddedRow * sizeof(int32_t) / 32), 0);
        uint64_t kept = 0;
        AscendC::GatherMask<int32_t>(tile, tile, 7, true,
            static_cast<uint32_t>(rowLength), gather, kept);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(rowReuse));
    }

    __aicore__ inline void CompareMacro(uint32_t slot, uint64_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputReady[slot]));
        auto raw = rawBuf.Get<int32_t>()[slot * 2 * elements];
        auto tile = broadcastBuf.Get<int32_t>();
        if (elements != logicalElements) {
            AscendC::Duplicate<int32_t>(raw[count * logicalElements], 0,
                static_cast<uint32_t>(elements - logicalElements));
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint64_t remaining = count; remaining != 0; --remaining) {
            const uint64_t piece = remaining - 1;
            auto data = raw[piece * logicalElements];
            auto pred = predicateBuf.Get<int8_t>()[piece * paddedRow];
            if (x1Rows) {
                AscendC::Min(data, tile, data, static_cast<int32_t>(elements));
            } else {
                AscendC::Max(data, data, tile, static_cast<int32_t>(elements));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(pred, tile, data, AscendC::CMPMODE::EQ,
                static_cast<uint32_t>(elements));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(inputReuse[slot]));
        inputPending[slot] = true;
    }

    __aicore__ inline void FlushPacked(uint32_t slot, uint64_t first, uint64_t count)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputReady[slot]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputReady[slot]));
        AscendC::DataCopy(packedGm[first * paddedRow],
            packedBuf.Get<int8_t>()[slot * 8 * paddedRow], count * paddedRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(outputReuse[slot]));
        outputPending[slot] = true;
    }

    __aicore__ inline void ComparePartition()
    {
        uint64_t cursor = firstPiece;
        uint64_t starts[2] = {endPiece, endPiece};
        uint64_t counts[2] = {0, 0};
        for (uint32_t s = 0; s < 2 && cursor < endPiece; ++s) {
            starts[s] = cursor;
            counts[s] = MacroCount(cursor);
            LoadRaw(s, cursor, counts[s]);
            cursor += counts[s];
        }
        uint64_t cachedGroup = static_cast<uint64_t>(-1);
        uint64_t packedFirst = firstPiece;
        uint64_t packedFill = 0;
        uint32_t packedSlot = 0;
        for (uint32_t slot = 0; starts[slot] < endPiece; slot ^= 1) {
            const uint64_t current = starts[slot];
            const uint64_t count = counts[slot];
            const uint64_t group = current / piecesPerGroup;
            if (group != cachedGroup) {
                CacheRow(group);
                cachedGroup = group;
            }
            CompareMacro(slot, count);
            if (cursor < endPiece) {
                starts[slot] = cursor;
                counts[slot] = MacroCount(cursor);
                LoadRaw(slot, cursor, counts[slot]);
                cursor += counts[slot];
            } else {
                starts[slot] = endPiece;
            }
            for (uint64_t p = 0; p < count; ++p) {
                if (packedFill == 0) { WaitOutputReuse(packedSlot); }
                AscendC::DataCopy(
                    packedBuf.Get<int8_t>()[(packedSlot * 8 + packedFill) * paddedRow],
                    predicateBuf.Get<int8_t>()[p * paddedRow], paddedRow);
                AscendC::PipeBarrier<PIPE_V>();
                ++packedFill;
                if (packedFill == 8) {
                    FlushPacked(packedSlot, packedFirst, packedFill);
                    packedFirst += packedFill;
                    packedFill = 0;
                    packedSlot ^= 1;
                }
            }
        }
        if (packedFill != 0) { FlushPacked(packedSlot, packedFirst, packedFill); }
    }

    __aicore__ inline void InitConstants()
    {
        auto constants = constantsBuf.Get<half>();
        AscendC::Duplicate<half>(constants, (half)1, 128);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(constants);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(constants, (half)0, 128);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadPacked(uint32_t slot, uint64_t first, uint64_t count)
    {
        WaitInputReuse(slot);
        AscendC::DataCopy(broadcastBuf.Get<int8_t>()[slot * 16 * paddedRow],
            packedGm[first * paddedRow], count * paddedRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputReady[slot]));
    }

    __aicore__ inline void ExpandChunk(uint32_t inputSlot, uint64_t offset, uint64_t count,
        uint32_t outputSlot, uint64_t first)
    {
        // Count-mode GatherMask restores the mode, not VECTOR_MASK bits.
        // This Select overload consumes the current normal-mode mask.
        AscendC::ResetMask();
        auto scratch = rawBuf.Get<int8_t>();
        auto packed = broadcastBuf.Get<int8_t>()[(inputSlot * 16 + offset) * paddedRow];
        const AscendC::BinaryRepeatParams repeats(1, 1, 0, 8, 0, 0);
        for (uint64_t p = 0; p < count; ++p) {
            auto bytes = scratch[p * elements];
            auto halves = bytes.ReinterpretCast<half>();
            AscendC::Select(halves, packed[p * paddedRow], constantsBuf.Get<half>(),
                static_cast<uint8_t>(elements / 128), repeats);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(bytes, halves, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        WaitOutputReuse(outputSlot);
        auto output = rawBuf.Get<int8_t>()[(5 + outputSlot * 4) * elements];
        if (logicalElements == elements) {
            AscendC::DataCopy(output, scratch, count * elements);
        } else {
            AscendC::GatherMaskParams gather(1, static_cast<uint16_t>(count),
                static_cast<uint16_t>(elements / 32), 0);
            uint64_t kept = 0;
            AscendC::GatherMask<uint16_t>(output.ReinterpretCast<uint16_t>(),
                scratch.ReinterpretCast<uint16_t>(), 7, true,
                static_cast<uint32_t>(logicalElements / 2), gather, kept);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputReady[outputSlot]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputReady[outputSlot]));
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * logicalElements);
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPad(yGm[first * logicalElements], output, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(outputReuse[outputSlot]));
        outputPending[outputSlot] = true;
    }

    __aicore__ inline void ExpandPartition()
    {
        uint64_t cursor = firstPiece;
        uint64_t starts[2] = {endPiece, endPiece};
        uint64_t counts[2] = {0, 0};
        for (uint32_t s = 0; s < 2 && cursor < endPiece; ++s) {
            starts[s] = cursor;
            counts[s] = endPiece - cursor < 16 ? endPiece - cursor : 16;
            LoadPacked(s, cursor, counts[s]);
            cursor += counts[s];
        }
        uint32_t outputSlot = 0;
        for (uint32_t slot = 0; starts[slot] < endPiece; slot ^= 1) {
            const uint64_t current = starts[slot];
            const uint64_t count = counts[slot];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(inputReady[slot]));
            for (uint64_t p = 0; p < count; p += 4) {
                const uint64_t n = count - p < 4 ? count - p : 4;
                ExpandChunk(slot, p, n, outputSlot, current + p);
                outputSlot ^= 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(inputReuse[slot]));
            inputPending[slot] = true;
            if (cursor < endPiece) {
                starts[slot] = cursor;
                counts[slot] = endPiece - cursor < 16 ? endPiece - cursor : 16;
                LoadPacked(slot, cursor, counts[slot]);
                cursor += counts[slot];
            } else {
                starts[slot] = endPiece;
            }
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf, broadcastBuf, rowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf, predicateBuf, constantsBuf;
    AscendC::GlobalTensor<int32_t> x1Gm, x2Gm;
    AscendC::GlobalTensor<int8_t> yGm, packedGm;
    uint64_t rowLength = 0, piecesPerGroup = 0, paddedRow = 0;
    uint64_t elements = 0, logicalElements = 0, firstPiece = 0, endPiece = 0;
    bool x1Rows = false;
    int32_t rowReady = 0, rowReuse = 0, transition = 0;
    int32_t inputReady[2] = {}, inputReuse[2] = {}, outputReady[2] = {}, outputReuse[2] = {};
    bool inputPending[2] = {}, outputPending[2] = {};
};

} // namespace NsGreater
#endif
