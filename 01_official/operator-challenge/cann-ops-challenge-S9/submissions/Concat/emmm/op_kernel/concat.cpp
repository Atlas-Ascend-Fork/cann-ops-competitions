#include "kernel_operator.h"
#include "basic_api/kernel_operator_list_tensor_intf.h"

namespace {

using namespace AscendC;
constexpr uint32_t VECTOR_BYTES = 256U;
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t MAX_DYNAMIC_INPUTS = 256U;
constexpr uint32_t PACKED_WIDTH_WORDS = MAX_DYNAMIC_INPUTS / 4U;
constexpr uint32_t FOUR_BUFFER_MAX_TILE_BYTES = 32U * 1024U;
constexpr uint32_t FIVE_BUFFER_MAX_TILE_BYTES = 24U * 1024U;
constexpr uint32_t COMPACT_GATHER_MIN_ROW_BYTES = 48U;
constexpr uint32_t COMPACT_GATHER_ROW_BYTE_LIMIT = 128U;
constexpr uint32_t COMPACT_GATHER_MAX_STORAGE_ROW_BYTES = 124U;
constexpr uint32_t COMPACT_GATHER_OFFSET_BYTES = 1024U;
constexpr uint32_t DYNAMIC_SLAB_HEADER_WORD = 74U;
constexpr uint32_t DYNAMIC_SLAB_METADATA_VERSION = 1U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_METADATA_VERSION = 7U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_GROUPS = 20U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS = 16U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD =
    DYNAMIC_SLAB_HEADER_WORD + 1U +
    (DYNAMIC_SLAB_GROUP_OWNER_GROUPS + 1U) / 2U;
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR = 0xffU;
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR = 0x00U;
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA = 0xefU;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_TIGHT_BUDGET_WORD = 90U;
constexpr uint32_t DYNAMIC_SLAB_GATHER_SEED_ROWS = 8U;
constexpr uint32_t DYNAMIC_SLAB_GATHER_ROWS =
    DYNAMIC_SLAB_GATHER_SEED_ROWS;
constexpr uint32_t DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT =
    VECTOR_BYTES / sizeof(uint32_t);
constexpr uint32_t DYNAMIC_SLAB_GATHER_OFFSET_BYTES =
    DYNAMIC_SLAB_GATHER_ROWS * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT *
    sizeof(uint32_t);
// Key 92 issues sixteen-row Gather repeats.  Its persistent tables must use
// the full sixteen records as both their allocation size and their address
// stride.  Reusing DYNAMIC_SLAB_GATHER_OFFSET_BYTES here would retain the
// older eight-row (2048-byte) footprint and make adjacent tables overlap.
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES =
    16U * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(uint32_t);
static_assert(DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES == 4096U);
constexpr uint32_t DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS = 32U;
constexpr uint32_t DYNAMIC_SLAB_SCRATCH_OFFSET32_BYTES =
    DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS *
    DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(uint32_t);
constexpr uint32_t DYNAMIC_SLAB_COMPACT_GATHER_ROW_BYTE_LIMIT = 224U;
constexpr uint32_t DYNAMIC_SLAB_COMPACT_GATHER_MAX_ROW_BYTES =
    DYNAMIC_SLAB_COMPACT_GATHER_ROW_BYTE_LIMIT - sizeof(uint32_t);
constexpr uint32_t DYNAMIC_SLAB_COMPACT_GATHER_MAX_PREFIX_BYTES =
    DATA_BLOCK_BYTES - sizeof(uint32_t);
constexpr uint32_t DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES = 224U;
constexpr uint32_t DYNAMIC_SLAB_SEGMENT_BYTES = 192U;
constexpr uint32_t DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES = 224U;
constexpr uint32_t DYNAMIC_SLAB_MAX_PACKED_ROW_BYTES =
    0xffU * sizeof(uint32_t);
constexpr uint32_t DEDICATED_DIRECT_MAX_INPUTS = 8U;
constexpr uint32_t DEDICATED_DIRECT_MAX_SLOTS = 5U;
constexpr uint32_t DEDICATED_DIRECT_SLOT_PITCH_WORD =
    DEDICATED_DIRECT_MAX_INPUTS;
static_assert(
    DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES >= 2U * DATA_BLOCK_BYTES &&
        DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES % DATA_BLOCK_BYTES == 0U,
    "boundary scratch must keep head and tail in disjoint data blocks");
static_assert(
    (DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES + DATA_BLOCK_BYTES -
     sizeof(uint32_t)) / sizeof(uint32_t) <=
        DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT,
    "one Gather record must cover a compact row plus prefix residue");
static_assert(
    (DYNAMIC_SLAB_COMPACT_GATHER_MAX_ROW_BYTES +
     DYNAMIC_SLAB_COMPACT_GATHER_MAX_PREFIX_BYTES) /
        sizeof(uint32_t) <= DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT,
    "dynamic slab Gather offset record must cover row plus prefix residue");
static_assert(
    DYNAMIC_SLAB_SEGMENT_BYTES % DATA_BLOCK_BYTES == 0U,
    "dynamic slab Gather segments must preserve the destination prefix lane");
static_assert(
    (DYNAMIC_SLAB_SEGMENT_BYTES +
     DYNAMIC_SLAB_COMPACT_GATHER_MAX_PREFIX_BYTES) /
        sizeof(uint32_t) <= DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT,
    "dynamic slab segmented Gather record must cover one segment plus prefix");
static_assert(
    DYNAMIC_SLAB_MAX_PACKED_ROW_BYTES >=
        DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES,
    "dynamic slab packed row metadata must represent segmented Gather rows");
template <uint32_t DIRECT_BUFFER_COUNT, bool EXACT_BATCH = false,
          bool RAW_UB = false, bool NARROW_B32 = false,
          bool NARROW_B8 = false, bool INPUT_PARALLEL = false,
          bool NARROW_B16 = false, bool NARROW_CHUNK = false,
          bool ALIGNED_ASSEMBLY = false,
          bool UNIFORM_ALIGNED_ASSEMBLY = false,
          bool BATCH_NARROW_CHUNK = false,
          bool CACHE_NARROW_POINTERS = false,
          bool BALANCED_ROW_GROUPS = false,
          bool PACKED_ROW_WIDTHS = false,
          bool BALANCED_ROWS = false,
          bool PACKED_BALANCED_SCHEDULE = false,
          bool COMPACT_GATHER_ROWS = false>
class ConcatCopyKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        if constexpr (BATCH_NARROW_CHUNK || CACHE_NARROW_POINTERS) {
            const __gm__ uint64_t* descriptorStart =
                reinterpret_cast<__gm__ uint64_t*>(inputs);
            inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
                descriptorStart[0] / sizeof(uint64_t);
        }
        // folded_with_desc stores one [rank/offset, shape...] record per input.
        // Decode row widths on demand so the launch tiling stays independent of
        // the dynamic input count.
        if constexpr (!UNIFORM_ALIGNED_ASSEMBLY) {
            shapeStart_ = reinterpret_cast<__gm__ uint64_t*>(inputs) + 1U;
            descStructSize_ = static_cast<uint32_t>(*shapeStart_ & 0xffffffffULL) + 1U;
        }
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        if (RAW_UB) {
            return;
        }
        if (NARROW_CHUNK || ALIGNED_ASSEMBLY) {
            return;
        }
        if (DIRECT_BUFFER_COUNT != 0U) {
            pipe_.InitBuffer(inputBuffer_, tiling_->tileBytes);
            pipe_.InitBuffer(outputBuffer_, tiling_->tileBytes);
            if (DIRECT_BUFFER_COUNT >= 3U) {
                pipe_.InitBuffer(directBuffer2_, tiling_->tileBytes);
            }
            if (DIRECT_BUFFER_COUNT >= 4U) {
                pipe_.InitBuffer(directBuffer3_, tiling_->tileBytes);
            }
            if (DIRECT_BUFFER_COUNT == 5U) {
                pipe_.InitBuffer(directBuffer4_, tiling_->tileBytes);
            }
            return;
        }
        if (tiling_->rowsPerBatch != 0U && tiling_->blockFactorRows != 0U) {
            pipe_.InitBuffer(inputBuffer_, tiling_->tileBytes);
            pipe_.InitBuffer(outputBuffer_, tiling_->tileBytes);
            if (tiling_->elementBytes != 1U &&
                tiling_->outputRowBytes % DATA_BLOCK_BYTES == 0U &&
                tiling_->tileBytes <= FOUR_BUFFER_MAX_TILE_BYTES) {
                pipe_.InitBuffer(directBuffer2_, tiling_->tileBytes);
                pipe_.InitBuffer(directBuffer3_, tiling_->tileBytes);
                if (tiling_->tileBytes <= FIVE_BUFFER_MAX_TILE_BYTES) {
                    pipe_.InitBuffer(directBuffer4_, tiling_->tileBytes);
                }
            }
        } else {
            pipe_.InitBuffer(copyBuffer_, tiling_->tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (EXACT_BATCH) {
            ProcessExactBatch();
            return;
        }
        if (INPUT_PARALLEL) {
            ProcessInputParallel();
            return;
        }
        if (ALIGNED_ASSEMBLY) {
            ProcessAlignedAssembly();
            return;
        }
        if (DIRECT_BUFFER_COUNT != 0U) {
            ProcessBatchedRows();
            return;
        }
        if (tiling_->totalBytes == 0U || tiling_->outputRowBytes == 0U || tiling_->blockFactorBytes == 0U) {
            return;
        }
        if (tiling_->rowsPerBatch != 0U && tiling_->blockFactorRows != 0U) {
            ProcessBatchedRows();
            return;
        }
        ProcessByteRange();
    }

private:
    __aicore__ inline void ProcessAlignedAssembly()
    {
        const uint64_t blockFactorRows = tiling_->blockFactorRows;
        const uint64_t outerRows = tiling_->outerRows;
        const uint64_t outputRowBytes = tiling_->outputRowBytes;
        const uint32_t rowsPerBatch = tiling_->rowsPerBatch;
        const uint32_t inputNum = tiling_->inputNum;
        const uint32_t uniformInputRowBytes = tiling_->uniformInputRowBytes;
        uint64_t row = blockFactorRows * static_cast<uint64_t>(GetBlockIdx());
        if (row >= outerRows) {
            return;
        }
        const uint64_t endRow = Min(row + blockFactorRows, outerRows);
        LocalTensor<uint8_t> outputLocal(TPosition::VECCALC, 0U, tiling_->tileBytes);
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(rowsPerBatch)));
            uint32_t outputPrefix = 0U;
            for (uint32_t inputIdx = 0U; inputIdx < inputNum; ++inputIdx) {
                uint32_t rowBytes = uniformInputRowBytes;
                if constexpr (!UNIFORM_ALIGNED_ASSEMBLY) {
                    rowBytes = InputRowBytes(inputIdx);
                }
                GlobalTensor<uint8_t> inputGm;
                inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
                DataCopyParams copyParams{
                    rows, static_cast<uint16_t>(rowBytes / DATA_BLOCK_BYTES), 0U,
                    static_cast<uint16_t>((outputRowBytes - rowBytes) /
                                          DATA_BLOCK_BYTES)};
                DataCopy(outputLocal[outputPrefix], inputGm[row * rowBytes], copyParams);
                outputPrefix += rowBytes;
            }
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            const uint32_t outputBytes = static_cast<uint32_t>(rows) *
                static_cast<uint32_t>(outputRowBytes);
            DataCopy(outputGm_[row * outputRowBytes], outputLocal, outputBytes);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            row += rows;
        }
    }

    __aicore__ inline void ProcessInputParallel()
    {
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint64_t inputsPerBlock = tiling_->inputNum / blockCount;
        const uint64_t extraInputs = tiling_->inputNum % blockCount;
        const uint32_t startInput = static_cast<uint32_t>(
            blockIndex * inputsPerBlock + Min(blockIndex, extraInputs));
        const uint32_t assignedInputs = static_cast<uint32_t>(inputsPerBlock +
            static_cast<uint64_t>(blockIndex < extraInputs));
        if (assignedInputs == 0U) {
            return;
        }

        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < startInput; ++inputIdx) {
            outputPrefix += InputRowBytes(inputIdx);
        }
        LocalTensor<uint8_t> local0 = inputBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> local1 = outputBuffer_.Get<uint8_t>();
        bool pending0 = false;
        bool pending1 = false;
        uint32_t copyCount = 0U;
        for (uint32_t inputIdx = startInput;
             inputIdx < startInput + assignedInputs; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            for (uint64_t startRow = 0U; startRow < tiling_->outerRows;
                 startRow += tiling_->rowsPerBatch) {
                const uint16_t rows = static_cast<uint16_t>(Min(
                    tiling_->outerRows - startRow,
                    static_cast<uint64_t>(tiling_->rowsPerBatch)));
                if ((copyCount & 1U) == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyAlignedRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                          rowBytes, local0, EVENT_ID0);
                    pending0 = true;
                } else {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyAlignedRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                          rowBytes, local1, EVENT_ID1);
                    pending1 = true;
                }
                ++copyCount;
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void ProcessExactBatch()
    {
        const uint64_t startRow = tiling_->blockFactorRows * static_cast<uint64_t>(GetBlockIdx());
        const uint16_t rows = static_cast<uint16_t>(tiling_->blockFactorRows);
        if (RAW_UB) {
            const uint32_t tileBytes = tiling_->tileBytes;
            LocalTensor<uint8_t> local0(TPosition::VECCALC, 0U, tileBytes);
            LocalTensor<uint8_t> local1(TPosition::VECCALC, tileBytes, tileBytes);
            LocalTensor<uint8_t> local2(TPosition::VECCALC, 2U * tileBytes, tileBytes);
            LocalTensor<uint8_t> local3(TPosition::VECCALC, 3U * tileBytes, tileBytes);
            LocalTensor<uint8_t> local4(TPosition::VECCALC, 4U * tileBytes, tileBytes);
            CopyRowsPipelined5(startRow, rows, local0, local1, local2, local3, local4);
            return;
        }
        CopyRowsPipelined5(startRow, rows,
                           inputBuffer_.Get<uint8_t>(),
                           outputBuffer_.Get<uint8_t>(),
                           directBuffer2_.Get<uint8_t>(),
                           directBuffer3_.Get<uint8_t>(),
                           directBuffer4_.Get<uint8_t>());
    }

    __aicore__ inline void ProcessBatchedRows()
    {
        uint64_t startRow = tiling_->blockFactorRows * static_cast<uint64_t>(GetBlockIdx());
        uint64_t endRow = Min(startRow + tiling_->blockFactorRows, tiling_->outerRows);
        if constexpr (PACKED_BALANCED_SCHEDULE) {
            const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
            const uint32_t packedSchedule = tiling_->launchPadding[
                PACKED_WIDTH_WORDS + static_cast<uint32_t>(blockIndex) / 4U];
            const uint32_t scheduleByte =
                (packedSchedule >> ((blockIndex % 4U) * 8U)) & 0xffU;
            const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
            startRow = blockIndex * rowsPerBlock + (scheduleByte & 0x3fU);
            endRow = startRow + rowsPerBlock + ((scheduleByte >> 6U) & 1U);
        } else if constexpr (BALANCED_ROWS) {
            const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
            const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
            const uint64_t extraRows = tiling_->outerRows % blockCount;
            startRow = blockIndex * rowsPerBlock + Min(blockIndex, extraRows);
            endRow = startRow + rowsPerBlock +
                static_cast<uint64_t>(blockIndex < extraRows);
        } else if constexpr (BALANCED_ROW_GROUPS) {
            const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
            uint64_t lhs = tiling_->outputRowBytes;
            uint64_t rhs = DATA_BLOCK_BYTES;
            while (rhs != 0U) {
                const uint64_t remainder = lhs % rhs;
                lhs = rhs;
                rhs = remainder;
            }
            const uint64_t rowsPerAlignedGroup = DATA_BLOCK_BYTES / lhs;
            const uint64_t rowGroups = tiling_->outerRows / rowsPerAlignedGroup;
            const uint64_t groupsPerBlock = rowGroups / blockCount;
            const uint64_t extraGroups = rowGroups % blockCount;
            const uint64_t startGroup = blockIndex * groupsPerBlock +
                Min(blockIndex, extraGroups);
            const uint64_t assignedGroups = groupsPerBlock +
                static_cast<uint64_t>(blockIndex < extraGroups);
            startRow = startGroup * rowsPerAlignedGroup;
            endRow = startRow + assignedGroups * rowsPerAlignedGroup;
            if (blockIndex + 1U == blockCount) {
                endRow += tiling_->outerRows % rowsPerAlignedGroup;
            }
        }
        if (startRow >= tiling_->outerRows) {
            return;
        }
        if (NARROW_B16 || NARROW_CHUNK) {
            const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
            const uint64_t rowGroups = tiling_->outerRows / 16U;
            const uint64_t groupsPerBlock = rowGroups / blockCount;
            const uint64_t extraGroups = rowGroups % blockCount;
            const uint64_t startGroup = blockIndex * groupsPerBlock +
                Min(blockIndex, extraGroups);
            const uint64_t assignedGroups = groupsPerBlock +
                static_cast<uint64_t>(blockIndex < extraGroups);
            startRow = 16U * startGroup;
            endRow = startRow + 16U * assignedGroups;
            if (blockIndex + 1U == blockCount) {
                endRow += tiling_->outerRows % 16U;
            }
        }
        LocalTensor<uint64_t> cachedInputPtrs;
        if constexpr (BATCH_NARROW_CHUNK || CACHE_NARROW_POINTERS) {
            const uint32_t inputPointerBytes = AlignUp(
                tiling_->inputNum * sizeof(uint64_t), DATA_BLOCK_BYTES);
            const uint32_t pointerOffset = BATCH_NARROW_CHUNK ?
                tiling_->tileBytes + tiling_->rowsPerBatch * 768U :
                (DIRECT_BUFFER_COUNT == 0U ? 2U : DIRECT_BUFFER_COUNT) *
                    tiling_->tileBytes;
            cachedInputPtrs = LocalTensor<uint64_t>(
                TPosition::VECCALC, pointerOffset, inputPointerBytes);
            GlobalTensor<uint64_t> inputPtrGm;
            inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
            DataCopy(cachedInputPtrs, inputPtrGm,
                     inputPointerBytes / sizeof(uint64_t));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        }
        LocalTensor<uint8_t> compactInputLocal;
        LocalTensor<uint8_t> gatherOffsetLocal;
        if constexpr (COMPACT_GATHER_ROWS) {
            const uint32_t inputPointerBytes = AlignUp(
                tiling_->inputNum * sizeof(uint64_t), DATA_BLOCK_BYTES);
            const uint32_t compactInputOffset =
                DIRECT_BUFFER_COUNT * tiling_->tileBytes + inputPointerBytes;
            const uint32_t compactInputBytes = AlignUp(
                tiling_->rowsPerBatch * COMPACT_GATHER_MAX_STORAGE_ROW_BYTES,
                DATA_BLOCK_BYTES);
            compactInputLocal = LocalTensor<uint8_t>(
                TPosition::VECCALC, compactInputOffset, compactInputBytes);
            gatherOffsetLocal = LocalTensor<uint8_t>(
                TPosition::VECCALC, compactInputOffset + compactInputBytes,
                COMPACT_GATHER_OFFSET_BYTES);
        }
        uint32_t persistentPipelineState = 0U;
        uint64_t row = startRow;
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(endRow - row,
                static_cast<uint64_t>(tiling_->rowsPerBatch)));
            if (NARROW_CHUNK) {
                const uint32_t rowBytes = InputRowBytes(0U);
                const uint32_t scratchBytesPerRow = BATCH_NARROW_CHUNK ? 768U :
                    (rowBytes == 8U ? 160U : 96U);
                LocalTensor<uint8_t> outputLocal(
                    TPosition::VECCALC, 0U, tiling_->tileBytes);
                LocalTensor<uint8_t> inputLocal(
                    TPosition::VECCALC, tiling_->tileBytes,
                    tiling_->rowsPerBatch * scratchBytesPerRow);
                const uint16_t transposeRows = rows >= 32U ?
                    static_cast<uint16_t>(rows / 16U * 16U) : 0U;
                if (transposeRows != 0U) {
                    AssembleNarrowChunkRows(row, transposeRows, inputLocal,
                                            outputLocal, cachedInputPtrs);
                    const uint32_t outputBytes = static_cast<uint32_t>(transposeRows) *
                        static_cast<uint32_t>(tiling_->outputRowBytes);
                    const uint64_t rowsAfterBatch = endRow - row - rows;
                    const bool outputReusedByMte2 = transposeRows != rows ||
                        (rowsAfterBatch != 0U && rowsAfterBatch < 32U);
                    CopyOutput(row * tiling_->outputRowBytes, outputLocal, outputBytes,
                               outputReusedByMte2);
                }
                if (transposeRows != rows) {
                    CopyRowsPipelined(row + transposeRows,
                                      static_cast<uint16_t>(rows - transposeRows),
                                      inputLocal, outputLocal);
                }
                row += rows;
                continue;
            }
            LocalTensor<uint8_t> inputLocal;
            LocalTensor<uint8_t> outputLocal;
            if constexpr (RAW_UB) {
                inputLocal = LocalTensor<uint8_t>(
                    TPosition::VECCALC, 0U, tiling_->tileBytes);
                outputLocal = LocalTensor<uint8_t>(
                    TPosition::VECCALC, tiling_->tileBytes,
                    tiling_->tileBytes);
            } else {
                inputLocal = inputBuffer_.Get<uint8_t>();
                outputLocal = outputBuffer_.Get<uint8_t>();
            }
            if (NARROW_B16) {
                const uint16_t transposeRows = rows >= 32U ?
                    static_cast<uint16_t>(rows / 16U * 16U) : 0U;
                if (transposeRows != 0U) {
                    AssembleNarrowB16Rows(row, transposeRows, inputLocal, outputLocal,
                                          cachedInputPtrs);
                    const uint32_t outputBytes = static_cast<uint32_t>(transposeRows) *
                        static_cast<uint32_t>(tiling_->outputRowBytes);
                    CopyOutput(row * tiling_->outputRowBytes, outputLocal, outputBytes,
                               transposeRows != rows);
                }
                if (transposeRows != rows) {
                    CopyRowsPipelined(row + transposeRows,
                                      static_cast<uint16_t>(rows - transposeRows),
                                      inputLocal, outputLocal);
                }
                row += rows;
                continue;
            }
            if (NARROW_B32) {
                AssembleNarrowB32Rows(row, rows, inputLocal, outputLocal,
                                      cachedInputPtrs);
                const uint32_t outputBytes = static_cast<uint32_t>(rows) *
                    static_cast<uint32_t>(tiling_->outputRowBytes);
                CopyOutput(row * tiling_->outputRowBytes, outputLocal, outputBytes, false);
                row += rows;
                continue;
            }
            if (NARROW_B8) {
                if (rows >= 64U) {
                    AssembleNarrowB8Rows(row, rows, inputLocal, outputLocal,
                                         cachedInputPtrs);
                } else {
                    AssembleByteRows(row, rows, inputLocal, outputLocal);
                }
                const uint32_t outputBytes = static_cast<uint32_t>(rows) *
                    static_cast<uint32_t>(tiling_->outputRowBytes);
                CopyOutput(row * tiling_->outputRowBytes, outputLocal, outputBytes, false);
                row += rows;
                continue;
            }
            if (DIRECT_BUFFER_COUNT == 5U) {
                CopyRowsPipelined5(row, rows, inputLocal, outputLocal,
                                   directBuffer2_.Get<uint8_t>(),
                                   directBuffer3_.Get<uint8_t>(),
                                   directBuffer4_.Get<uint8_t>());
                row += rows;
                continue;
            } else if (DIRECT_BUFFER_COUNT == 4U) {
                CopyRowsPipelined4(row, rows, inputLocal, outputLocal,
                                   directBuffer2_.Get<uint8_t>(),
                                   directBuffer3_.Get<uint8_t>());
                row += rows;
                continue;
            } else if (DIRECT_BUFFER_COUNT == 3U) {
                if constexpr (RAW_UB) {
                    LocalTensor<uint8_t> local2(
                        TPosition::VECCALC, 2U * tiling_->tileBytes,
                        tiling_->tileBytes);
                    persistentPipelineState = CopyRowsPipelined3Persistent(
                        row, rows, inputLocal, outputLocal, local2,
                        persistentPipelineState);
                } else {
                    if constexpr (COMPACT_GATHER_ROWS) {
                        CopyRowsPipelined3Compact(
                            row, rows, inputLocal, outputLocal,
                            directBuffer2_.Get<uint8_t>(), cachedInputPtrs,
                            compactInputLocal, gatherOffsetLocal);
                    } else {
                        CopyRowsPipelined3(row, rows, inputLocal, outputLocal,
                                           directBuffer2_.Get<uint8_t>(),
                                           cachedInputPtrs);
                    }
                }
                row += rows;
                continue;
            } else if (DIRECT_BUFFER_COUNT == 2U) {
                CopyRowsPipelined(row, rows, inputLocal, outputLocal);
                row += rows;
                continue;
            }
            const bool byteBlockMode = tiling_->elementBytes == 1U;
            const bool fastDirectMode = !byteBlockMode &&
                tiling_->outputRowBytes % DATA_BLOCK_BYTES == 0U;
            const bool batchedVectorMode = !byteBlockMode &&
                tiling_->outputRowBytes % DATA_BLOCK_BYTES == 0U;
            if (fastDirectMode) {
                if (tiling_->tileBytes <= FIVE_BUFFER_MAX_TILE_BYTES) {
                    CopyRowsPipelined5(row, rows, inputLocal, outputLocal,
                                       directBuffer2_.Get<uint8_t>(),
                                       directBuffer3_.Get<uint8_t>(),
                                       directBuffer4_.Get<uint8_t>());
                } else if (tiling_->tileBytes <= FOUR_BUFFER_MAX_TILE_BYTES) {
                    CopyRowsPipelined4(row, rows, inputLocal, outputLocal,
                                       directBuffer2_.Get<uint8_t>(),
                                       directBuffer3_.Get<uint8_t>());
                } else {
                    CopyRowsPipelined(row, rows, inputLocal, outputLocal);
                }
                row += rows;
                continue;
            } else if (tiling_->outputRowBytes % DATA_BLOCK_BYTES != 0U) {
                CopyRowsPipelined(row, rows, inputLocal, outputLocal);
                row += rows;
                continue;
            } else if (byteBlockMode) {
                AssembleByteRows(row, rows, inputLocal, outputLocal);
            } else {
                uint64_t outputPrefix = 0U;
                for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
                    const uint32_t rowBytes = InputRowBytes(inputIdx);
                    if (rowBytes != 0U && batchedVectorMode) {
                        CopyInputToOutputLocal(inputIdx, row, rows, outputPrefix, rowBytes,
                                               inputLocal, outputLocal);
                    } else if (rowBytes != 0U) {
                        CopyUnalignedVectorRows(inputIdx, row, rows, outputPrefix, rowBytes,
                                                inputLocal, outputLocal);
                    }
                    outputPrefix += rowBytes;
                }
            }
            const uint32_t outputBytes = static_cast<uint32_t>(rows) *
                static_cast<uint32_t>(tiling_->outputRowBytes);
            CopyOutput(row * tiling_->outputRowBytes, outputLocal, outputBytes, false);
            row += rows;
        }
        if constexpr (RAW_UB && DIRECT_BUFFER_COUNT == 3U) {
            const uint32_t pendingMask = persistentPipelineState & 7U;
            if ((pendingMask & 1U) != 0U) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            if ((pendingMask & 2U) != 0U) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
            if ((pendingMask & 4U) != 0U) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            }
        }
    }

    __aicore__ inline void CopyRowsPipelined(uint64_t startRow, uint16_t rows,
                                             LocalTensor<uint8_t> local0,
                                             LocalTensor<uint8_t> local1)
    {
        bool pending0 = false;
        bool pending1 = false;
        uint32_t copyCount = 0U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes != 0U) {
                const bool useSecond = (copyCount & 1U) != 0U;
                if (useSecond) {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local1, EVENT_ID1);
                    pending1 = true;
                } else {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local0, EVENT_ID0);
                    pending0 = true;
                }
                ++copyCount;
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void CopyRowsPipelined4(uint64_t startRow, uint16_t rows,
                                              LocalTensor<uint8_t> local0,
                                              LocalTensor<uint8_t> local1,
                                              LocalTensor<uint8_t> local2,
                                              LocalTensor<uint8_t> local3)
    {
        bool pending0 = false;
        bool pending1 = false;
        bool pending2 = false;
        bool pending3 = false;
        uint32_t copyCount = 0U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes != 0U) {
                const uint32_t slot = copyCount & 3U;
                if (slot == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local0, EVENT_ID0);
                    pending0 = true;
                } else if (slot == 1U) {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local1, EVENT_ID1);
                    pending1 = true;
                } else if (slot == 2U) {
                    if (pending2) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local2, EVENT_ID2);
                    pending2 = true;
                } else {
                    if (pending3) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local3, EVENT_ID3);
                    pending3 = true;
                }
                ++copyCount;
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if (pending2) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (pending3) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
    }

    __aicore__ inline void CopyRowsPipelined3(uint64_t startRow, uint16_t rows,
                                              LocalTensor<uint8_t> local0,
                                              LocalTensor<uint8_t> local1,
                                              LocalTensor<uint8_t> local2,
                                              LocalTensor<uint64_t> cachedInputPtrs)
    {
        bool pending0 = false;
        bool pending1 = false;
        bool pending2 = false;
        uint32_t slot = 0U;
        uint32_t packedRowWidths = 0U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            uint32_t rowBytes = 0U;
            if constexpr (PACKED_ROW_WIDTHS) {
                if ((inputIdx & 3U) == 0U) {
                    packedRowWidths = tiling_->launchPadding[inputIdx / 4U];
                }
                const uint32_t rowElements =
                    (packedRowWidths >> ((inputIdx & 3U) * 8U)) & 0xffU;
                rowBytes = rowElements * tiling_->elementBytes;
            } else {
                rowBytes = InputRowBytes(inputIdx);
            }
            if (rowBytes != 0U) {
                if (slot == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyRowsDirect3(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                    local0, EVENT_ID0, cachedInputPtrs);
                    pending0 = true;
                } else if (slot == 1U) {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyRowsDirect3(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                    local1, EVENT_ID1, cachedInputPtrs);
                    pending1 = true;
                } else {
                    if (pending2) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                    CopyRowsDirect3(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                    local2, EVENT_ID2, cachedInputPtrs);
                    pending2 = true;
                }
                slot = slot == 2U ? 0U : slot + 1U;
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if (pending2) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
    }

    __aicore__ inline void CopyRowsPipelined3Compact(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint8_t> local0, LocalTensor<uint8_t> local1,
        LocalTensor<uint8_t> local2, LocalTensor<uint64_t> cachedInputPtrs,
        LocalTensor<uint8_t> compactInputLocal,
        LocalTensor<uint8_t> gatherOffsetLocal)
    {
        uint32_t pendingMask = 0U;
        uint32_t slot = 0U;
        uint32_t packedRowWidths = 0U;
        uint64_t outputPrefix = 0U;
        bool compactPending = false;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            if ((inputIdx & 3U) == 0U) {
                packedRowWidths = tiling_->launchPadding[inputIdx / 4U];
            }
            const uint32_t rowElements =
                (packedRowWidths >> ((inputIdx & 3U) * 8U)) & 0xffU;
            const uint32_t rowBytes = rowElements * tiling_->elementBytes;
            if (rowBytes != 0U) {
                const uint32_t pendingBit = 1U << slot;
                if ((pendingMask & pendingBit) != 0U) {
                    if (slot == 0U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    } else if (slot == 1U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    } else {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                }
                LocalTensor<uint8_t> outputLocal = local0;
                event_t outputEvent = EVENT_ID0;
                if (slot == 1U) {
                    outputLocal = local1;
                    outputEvent = EVENT_ID1;
                } else if (slot == 2U) {
                    outputLocal = local2;
                    outputEvent = EVENT_ID2;
                }
                const bool useCompactGather =
                    rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
                    rowBytes < COMPACT_GATHER_ROW_BYTE_LIMIT &&
                    rowBytes % DATA_BLOCK_BYTES != 0U;
                if (useCompactGather) {
                    if (compactPending) {
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
                    }
                    CopyRowsCompactGather(
                        inputIdx, startRow, rows, outputPrefix, rowBytes,
                        outputLocal, outputEvent, cachedInputPtrs,
                        compactInputLocal, gatherOffsetLocal);
                    compactPending = true;
                } else {
                    CopyRowsDirect3(
                        inputIdx, startRow, rows, outputPrefix, rowBytes,
                        outputLocal, outputEvent, cachedInputPtrs);
                }
                pendingMask |= pendingBit;
                slot = slot == 2U ? 0U : slot + 1U;
            }
            outputPrefix += rowBytes;
        }
        if (compactPending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        }
        if ((pendingMask & 1U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if ((pendingMask & 2U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if ((pendingMask & 4U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
    }

    __aicore__ inline void CopyRowsCompactGather(
        uint32_t inputIdx, uint64_t startRow, uint16_t rows,
        uint64_t outputPrefix, uint32_t rowBytes,
        LocalTensor<uint8_t> outputLocal, event_t outputEvent,
        LocalTensor<uint64_t> cachedInputPtrs,
        LocalTensor<uint8_t> compactInputLocal,
        LocalTensor<uint8_t> gatherOffsetLocal)
    {
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
            cachedInputPtrs.GetValue(inputIdx)));
        const uint32_t compactBytes = static_cast<uint32_t>(rows) * rowBytes;
        DataCopyExtParams copyInParams{1U, compactBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> noPadding{false, 0U, 0U, 0U};
        DataCopyPad(compactInputLocal, inputGm[startRow * rowBytes],
                    copyInParams, noPadding);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);

        const uint32_t rowElements = rowBytes / sizeof(uint32_t);
        const uint32_t paddedRowElements =
            AlignUp(rowBytes, DATA_BLOCK_BYTES) / sizeof(uint32_t);
        LocalTensor<uint32_t> compactWords =
            compactInputLocal.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> outputWords =
            outputLocal.ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> gatherOffsetValues =
            gatherOffsetLocal.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> gatherOffsets =
            gatherOffsetLocal.ReinterpretCast<uint32_t>();

        Duplicate<int32_t>(gatherOffsetValues, 0,
                           8U * paddedRowElements);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < 8U; ++row) {
            ArithProgression<int32_t>(
                gatherOffsetValues[row * paddedRowElements],
                static_cast<int32_t>(row * rowBytes),
                static_cast<int32_t>(sizeof(uint32_t)), rowElements);
        }
        PipeBarrier<PIPE_V>();
        uint32_t row = 0U;
        for (; row + 8U <= rows; row += 8U) {
            Gather<uint32_t>(
                outputWords[row * paddedRowElements],
                compactWords[row * rowElements], gatherOffsets, 0U,
                8U * paddedRowElements);
        }
        PipeBarrier<PIPE_V>();
        ArithProgression<int32_t>(
            gatherOffsetValues, 0, static_cast<int32_t>(sizeof(uint32_t)),
            rowElements);
        PipeBarrier<PIPE_V>();
        for (; row < rows; ++row) {
            Gather<uint32_t>(
                outputWords[row * paddedRowElements], compactWords,
                gatherOffsets, row * rowBytes, rowElements);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::V_MTE3>(outputEvent);
        WaitFlag<HardEvent::V_MTE3>(outputEvent);

        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{
            rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(
            outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
            outputLocal, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(outputEvent);
    }

    __aicore__ inline uint32_t CopyRowsPipelined3Persistent(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint8_t> local0, LocalTensor<uint8_t> local1,
        LocalTensor<uint8_t> local2, uint32_t pipelineState)
    {
        uint32_t pendingMask = pipelineState & 7U;
        uint32_t slot = (pipelineState >> 3U) & 3U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes != 0U) {
                const uint32_t pendingBit = 1U << slot;
                if ((pendingMask & pendingBit) != 0U) {
                    if (slot == 0U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    } else if (slot == 1U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    } else {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                }
                if (slot == 0U) {
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local0, EVENT_ID0);
                } else if (slot == 1U) {
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local1, EVENT_ID1);
                } else {
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local2, EVENT_ID2);
                }
                pendingMask |= pendingBit;
                slot = slot == 2U ? 0U : slot + 1U;
            }
            outputPrefix += rowBytes;
        }
        return pendingMask | (slot << 3U);
    }

    __aicore__ inline void CopyRowsPipelined5(uint64_t startRow, uint16_t rows,
                                              LocalTensor<uint8_t> local0,
                                              LocalTensor<uint8_t> local1,
                                              LocalTensor<uint8_t> local2,
                                              LocalTensor<uint8_t> local3,
                                              LocalTensor<uint8_t> local4)
    {
        bool pending0 = false;
        bool pending1 = false;
        bool pending2 = false;
        bool pending3 = false;
        bool pending4 = false;
        uint32_t slot = 0U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes != 0U) {
                if (slot == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local0, EVENT_ID0);
                    pending0 = true;
                } else if (slot == 1U) {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local1, EVENT_ID1);
                    pending1 = true;
                } else if (slot == 2U) {
                    if (pending2) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local2, EVENT_ID2);
                    pending2 = true;
                } else if (slot == 3U) {
                    if (pending3) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local3, EVENT_ID3);
                    pending3 = true;
                } else {
                    if (pending4) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix, rowBytes,
                                   local4, EVENT_ID4);
                    pending4 = true;
                }
                ++slot;
                if (slot == 5U) {
                    slot = 0U;
                }
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if (pending2) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (pending3) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if (pending4) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
    }

    __aicore__ inline void CopyRowsDirect(uint32_t inputIdx, uint64_t startRow,
                                          uint16_t rows, uint64_t outputPrefix,
                                          uint32_t rowBytes, LocalTensor<uint8_t> local,
                                          event_t eventId)
    {
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
        if (rowBytes % DATA_BLOCK_BYTES == 0U) {
            const uint32_t inputBlocks =
                static_cast<uint32_t>(rows) * rowBytes / DATA_BLOCK_BYTES;
            DataCopyParams copyInParams{
                1U, static_cast<uint16_t>(inputBlocks), 0U, 0U};
            DataCopy(local, inputGm[startRow * rowBytes], copyInParams);
        } else {
            DataCopyExtParams copyInParams{rows, rowBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(local, inputGm[startRow * rowBytes],
                        copyInParams, padParams);
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                    local, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopyRowsDirect3(uint32_t inputIdx, uint64_t startRow,
                                           uint16_t rows, uint64_t outputPrefix,
                                           uint32_t rowBytes,
                                           LocalTensor<uint8_t> local,
                                           event_t eventId,
                                           LocalTensor<uint64_t> cachedInputPtrs)
    {
        GlobalTensor<uint8_t> inputGm;
        if constexpr (CACHE_NARROW_POINTERS) {
            inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                cachedInputPtrs.GetValue(inputIdx)));
        } else {
            inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
        }
        if (rowBytes % DATA_BLOCK_BYTES == 0U) {
            const uint32_t inputBlocks =
                static_cast<uint32_t>(rows) * rowBytes / DATA_BLOCK_BYTES;
            DataCopyParams copyInParams{
                1U, static_cast<uint16_t>(inputBlocks), 0U, 0U};
            DataCopy(local, inputGm[startRow * rowBytes], copyInParams);
        } else {
            DataCopyExtParams copyInParams{rows, rowBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(local, inputGm[startRow * rowBytes],
                        copyInParams, padParams);
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                    local, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopyAlignedRowsDirect(uint32_t inputIdx, uint64_t startRow,
                                                 uint16_t rows, uint64_t outputPrefix,
                                                 uint32_t rowBytes,
                                                 LocalTensor<uint8_t> local,
                                                 event_t eventId)
    {
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
        const uint32_t inputBytes = static_cast<uint32_t>(rows) * rowBytes;
        DataCopyExtParams copyInParams{1U, inputBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(local, inputGm[startRow * rowBytes], copyInParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                    local, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopyInputToOutputLocal(uint32_t inputIdx, uint64_t startRow,
                                                  uint16_t rows, uint64_t outputPrefix,
                                                  uint32_t rowBytes,
                                                  LocalTensor<uint8_t> inputLocal,
                                                  LocalTensor<uint8_t> outputLocal)
    {
        __gm__ uint8_t* source = inputList_.GetDataPtr<uint8_t>(inputIdx);
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(source);

        const uint32_t leftPadding = static_cast<uint32_t>(outputPrefix % DATA_BLOCK_BYTES);
        const uint32_t inputRowStride = AlignUp(leftPadding + rowBytes, DATA_BLOCK_BYTES);
        DataCopyExtParams copyInParams{rows, rowBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{
            leftPadding != 0U, static_cast<uint8_t>(leftPadding), 0U, 0U};
        DataCopyPad(inputLocal, inputGm[startRow * rowBytes],
                    copyInParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (tiling_->elementBytes == 2U) {
            CopyVectorRows<int16_t>(outputLocal, inputLocal, rows, outputPrefix,
                                    rowBytes, inputRowStride);
        } else {
            CopyVectorRows<int32_t>(outputLocal, inputLocal, rows, outputPrefix,
                                    rowBytes, inputRowStride);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void CopyUnalignedVectorRows(uint32_t inputIdx, uint64_t startRow,
                                                   uint16_t rows, uint64_t outputPrefix,
                                                   uint32_t rowBytes,
                                                   LocalTensor<uint8_t> inputLocal,
                                                   LocalTensor<uint8_t> outputLocal)
    {
        __gm__ uint8_t* source = inputList_.GetDataPtr<uint8_t>(inputIdx);
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(source);
        for (uint16_t localRow = 0U; localRow < rows; ++localRow) {
            const uint64_t destinationOffset = localRow * tiling_->outputRowBytes + outputPrefix;
            const uint32_t leftPadding = static_cast<uint32_t>(destinationOffset % DATA_BLOCK_BYTES);
            const uint32_t inputRowStride = AlignUp(leftPadding + rowBytes, DATA_BLOCK_BYTES);
            DataCopyExtParams copyInParams{1U, rowBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> padParams{
                leftPadding != 0U, static_cast<uint8_t>(leftPadding), 0U, 0U};
            const uint64_t sourceOffset = (startRow + localRow) * rowBytes;
            DataCopyPad(inputLocal, inputGm[sourceOffset], copyInParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (tiling_->elementBytes == 2U) {
                CopyVectorRows<int16_t>(outputLocal, inputLocal, 1U, destinationOffset,
                                        rowBytes, inputRowStride);
            } else {
                CopyVectorRows<int32_t>(outputLocal, inputLocal, 1U, destinationOffset,
                                        rowBytes, inputRowStride);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void AssembleByteRows(uint64_t startRow, uint16_t rows,
                                            LocalTensor<uint8_t> scratchLocal,
                                            LocalTensor<uint8_t> outputLocal)
    {
        LocalTensor<uint16_t> scratch = scratchLocal.template ReinterpretCast<uint16_t>();
        const uint32_t totalBytes = static_cast<uint32_t>(rows) *
            static_cast<uint32_t>(tiling_->outputRowBytes);
        const uint32_t alignedTotalBytes = AlignUp(totalBytes, DATA_BLOCK_BYTES);
        Duplicate<uint16_t>(outputLocal.template ReinterpretCast<uint16_t>(),
                            static_cast<uint16_t>(0), alignedTotalBytes / sizeof(uint16_t));

        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes == 0U) {
                continue;
            }
            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
            for (uint16_t localRow = 0U; localRow < rows; ++localRow) {
                const uint32_t destinationOffset = static_cast<uint32_t>(
                    static_cast<uint64_t>(localRow) * tiling_->outputRowBytes + outputPrefix);
                const uint32_t leftPadding = destinationOffset % DATA_BLOCK_BYTES;
                const uint32_t destinationBase = destinationOffset - leftPadding;
                const uint32_t scratchBytes = AlignUp(leftPadding + rowBytes, DATA_BLOCK_BYTES);
                const uint32_t rightPadding = scratchBytes - leftPadding - rowBytes;
                const uint64_t sourceOffset = (startRow + localRow) * rowBytes;
                DataCopyExtParams copyParams{1U, rowBytes, 0U, 0U, 0U};
                DataCopyPadExtParams<uint8_t> padParams{
                    true, static_cast<uint8_t>(leftPadding),
                    static_cast<uint8_t>(rightPadding), 0U};
                DataCopyPad(scratchLocal, inputGm[sourceOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                LocalTensor<uint16_t> output =
                    outputLocal[destinationBase].template ReinterpretCast<uint16_t>();
                Or<uint16_t>(output, output, scratch, scratchBytes / sizeof(uint16_t));
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            outputPrefix += rowBytes;
        }
    }

    __aicore__ inline void AssembleNarrowB32Rows(uint64_t startRow, uint16_t rows,
                                                 LocalTensor<uint8_t> scratchLocal,
                                                 LocalTensor<uint8_t> outputLocal,
                                                 LocalTensor<uint64_t> cachedInputPtrs)
    {
        constexpr uint32_t MATRIX_INPUTS = 16U;
        constexpr uint32_t MATRIX_ROWS = 8U;
        constexpr uint32_t OUTPUT_HALF_BYTES = 8U * sizeof(uint32_t);
        const uint32_t outputRowBytes = static_cast<uint32_t>(tiling_->outputRowBytes);
        const uint32_t inputBytes = static_cast<uint32_t>(rows) * sizeof(uint32_t);
        const uint32_t inputStorageBytes = AlignUp(inputBytes, DATA_BLOCK_BYTES);
        DataCopyExtParams copyParams{1U, inputBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            GlobalTensor<uint8_t> inputGm;
            if constexpr (CACHE_NARROW_POINTERS) {
                inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    cachedInputPtrs.GetValue(inputIdx)));
            } else {
                inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
            }
            DataCopyPad(scratchLocal[inputIdx * inputStorageBytes],
                        inputGm[startRow * sizeof(uint32_t)], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const uint8_t repeatTimes = static_cast<uint8_t>(rows / MATRIX_ROWS);
        const uint16_t destinationRepeatStride = static_cast<uint16_t>(
            MATRIX_ROWS * outputRowBytes / DATA_BLOCK_BYTES);
        for (uint32_t inputGroup = 0U; inputGroup < tiling_->inputNum;
             inputGroup += MATRIX_INPUTS) {
            uint64_t sourceAddresses[MATRIX_INPUTS];
            for (uint32_t lane = 0U; lane < MATRIX_INPUTS; ++lane) {
                sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[(inputGroup + lane) * inputStorageBytes].GetPhyAddr());
            }
            uint64_t destinationAddresses[MATRIX_INPUTS];
            const uint32_t groupOutputOffset = inputGroup * sizeof(uint32_t);
            for (uint32_t addressIdx = 0U; addressIdx < MATRIX_INPUTS; ++addressIdx) {
                const uint32_t destinationOffset = (addressIdx >> 1U) * outputRowBytes +
                    groupOutputOffset + (addressIdx & 1U) * OUTPUT_HALF_BYTES;
                destinationAddresses[addressIdx] = reinterpret_cast<uint64_t>(
                    outputLocal[destinationOffset].GetPhyAddr());
            }
            set_va_reg_sb(VA0, destinationAddresses);
            set_va_reg_sb(VA1, destinationAddresses + 8U);
            set_va_reg_sb(VA2, sourceAddresses);
            set_va_reg_sb(VA3, sourceAddresses + 8U);
            scatter_vnchwconv_b32(VA0, VA2, repeatTimes, destinationRepeatStride, 1U);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void AssembleNarrowB16Rows(uint64_t startRow, uint16_t rows,
                                                 LocalTensor<uint8_t> scratchLocal,
                                                 LocalTensor<uint8_t> outputLocal,
                                                 LocalTensor<uint64_t> cachedInputPtrs)
    {
        constexpr uint32_t MATRIX_INPUTS = 16U;
        constexpr uint32_t MATRIX_ROWS = 16U;
        constexpr uint32_t ELEMENT_BYTES = sizeof(uint16_t);
        const uint32_t outputRowBytes = static_cast<uint32_t>(tiling_->outputRowBytes);
        const uint32_t inputBytes = static_cast<uint32_t>(rows) * ELEMENT_BYTES;
        const uint32_t inputStorageBytes = AlignUp(inputBytes, DATA_BLOCK_BYTES);
        DataCopyExtParams copyParams{1U, inputBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            GlobalTensor<uint8_t> inputGm;
            if constexpr (CACHE_NARROW_POINTERS) {
                inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    cachedInputPtrs.GetValue(inputIdx)));
            } else {
                inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
            }
            DataCopyPad(scratchLocal[inputIdx * inputStorageBytes],
                        inputGm[startRow * ELEMENT_BYTES], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const uint8_t repeatTimes = static_cast<uint8_t>(rows / MATRIX_ROWS);
        const uint16_t destinationRepeatStride = static_cast<uint16_t>(
            MATRIX_ROWS * outputRowBytes / DATA_BLOCK_BYTES);
        for (uint32_t inputGroup = 0U; inputGroup < tiling_->inputNum;
             inputGroup += MATRIX_INPUTS) {
            uint64_t sourceAddresses[MATRIX_INPUTS];
            uint64_t destinationAddresses[MATRIX_INPUTS];
            const uint32_t groupOutputOffset = inputGroup * ELEMENT_BYTES;
            for (uint32_t lane = 0U; lane < MATRIX_INPUTS; ++lane) {
                sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[(inputGroup + lane) * inputStorageBytes].GetPhyAddr());
                destinationAddresses[lane] = reinterpret_cast<uint64_t>(
                    outputLocal[lane * outputRowBytes + groupOutputOffset].GetPhyAddr());
            }
            set_va_reg_sb(VA0, destinationAddresses);
            set_va_reg_sb(VA1, destinationAddresses + 8U);
            set_va_reg_sb(VA2, sourceAddresses);
            set_va_reg_sb(VA3, sourceAddresses + 8U);
            scatter_vnchwconv_b16(VA0, VA2, repeatTimes,
                                  destinationRepeatStride, 1U);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void AssembleNarrowChunkRows(uint64_t startRow, uint16_t rows,
                                                   LocalTensor<uint8_t> scratchLocal,
                                                   LocalTensor<uint8_t> outputLocal,
                                                   LocalTensor<uint64_t> cachedInputPtrs)
    {
        if constexpr (BATCH_NARROW_CHUNK) {
            AssembleBatchedNarrowChunkRows(startRow, rows, scratchLocal,
                                           outputLocal, cachedInputPtrs);
            return;
        }
        constexpr uint32_t MATRIX_ROWS = 16U;
        constexpr uint32_t ELEMENT_BYTES = sizeof(uint16_t);
        const uint32_t rowBytes = InputRowBytes(0U);
        const uint32_t elementsPerInput = rowBytes / ELEMENT_BYTES;
        const uint32_t inputsPerGroup = DATA_BLOCK_BYTES / rowBytes;
        const uint32_t repeats = static_cast<uint32_t>(rows) / MATRIX_ROWS;
        const uint32_t inputStorageBytes = AlignUp(
            static_cast<uint32_t>(rows) * rowBytes, DATA_BLOCK_BYTES);
        const uint32_t firstStageStride = repeats * DATA_BLOCK_BYTES;
        const uint32_t intermediateBase = inputsPerGroup * inputStorageBytes;
        const uint32_t outputRowBytes = static_cast<uint32_t>(tiling_->outputRowBytes);
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(rows) * rowBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

        for (uint32_t inputGroup = 0U; inputGroup < tiling_->inputNum;
             inputGroup += inputsPerGroup) {
            for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                GlobalTensor<uint8_t> inputGm;
                inputGm.SetGlobalBuffer(
                    inputList_.GetDataPtr<uint8_t>(inputGroup + inputSlot));
                DataCopyPad(scratchLocal[inputSlot * inputStorageBytes],
                            inputGm[startRow * rowBytes], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                uint64_t sourceAddresses[MATRIX_ROWS];
                uint64_t destinationAddresses[MATRIX_ROWS];
                const uint32_t inputBase = inputSlot * inputStorageBytes;
                const uint32_t stageBase = intermediateBase +
                    inputSlot * MATRIX_ROWS * firstStageStride;
                for (uint32_t lane = 0U; lane < MATRIX_ROWS; ++lane) {
                    sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                        scratchLocal[inputBase + lane * rowBytes].GetPhyAddr());
                    destinationAddresses[lane] = reinterpret_cast<uint64_t>(
                        scratchLocal[stageBase + lane * firstStageStride].GetPhyAddr());
                }
                set_va_reg_sb(VA0, destinationAddresses);
                set_va_reg_sb(VA1, destinationAddresses + 8U);
                set_va_reg_sb(VA2, sourceAddresses);
                set_va_reg_sb(VA3, sourceAddresses + 8U);
                scatter_vnchwconv_b16(VA0, VA2, static_cast<uint8_t>(repeats),
                                      1U, static_cast<uint16_t>(rowBytes / 2U));
            }

            uint64_t sourceAddresses[MATRIX_ROWS];
            uint64_t destinationAddresses[MATRIX_ROWS];
            for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                const uint32_t stageBase = intermediateBase +
                    inputSlot * MATRIX_ROWS * firstStageStride;
                for (uint32_t element = 0U; element < elementsPerInput; ++element) {
                    const uint32_t lane = inputSlot * elementsPerInput + element;
                    sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                        scratchLocal[stageBase + element * firstStageStride].GetPhyAddr());
                }
            }
            const uint32_t groupOutputOffset = inputGroup * rowBytes;
            for (uint32_t row = 0U; row < MATRIX_ROWS; ++row) {
                destinationAddresses[row] = reinterpret_cast<uint64_t>(
                    outputLocal[row * outputRowBytes + groupOutputOffset].GetPhyAddr());
            }
            set_va_reg_sb(VA0, destinationAddresses);
            set_va_reg_sb(VA1, destinationAddresses + 8U);
            set_va_reg_sb(VA2, sourceAddresses);
            set_va_reg_sb(VA3, sourceAddresses + 8U);
            scatter_vnchwconv_b16(
                VA0, VA2, static_cast<uint8_t>(repeats),
                static_cast<uint16_t>(MATRIX_ROWS * outputRowBytes / DATA_BLOCK_BYTES),
                1U);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void AssembleBatchedNarrowChunkRows(
        uint64_t startRow, uint16_t rows, LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint8_t> outputLocal, LocalTensor<uint64_t> cachedInputPtrs)
    {
        constexpr uint32_t MATRIX_ROWS = 16U;
        constexpr uint32_t ELEMENT_BYTES = sizeof(uint16_t);
        constexpr uint32_t GROUPS_PER_STAGE = 8U;
        constexpr uint32_t STAGE_BUFFERS = 2U;
        const uint32_t rowBytes = InputRowBytes(0U);
        const uint32_t elementsPerInput = rowBytes / ELEMENT_BYTES;
        const uint32_t inputsPerGroup = DATA_BLOCK_BYTES / rowBytes;
        const uint32_t repeats = static_cast<uint32_t>(rows) / MATRIX_ROWS;
        const uint32_t inputStorageBytes = AlignUp(
            static_cast<uint32_t>(rows) * rowBytes, DATA_BLOCK_BYTES);
        const uint32_t firstStageStride = repeats * DATA_BLOCK_BYTES;
        const uint32_t inputBytesPerGroup = inputsPerGroup * inputStorageBytes;
        const uint32_t intermediateBytesPerGroup =
            inputsPerGroup * MATRIX_ROWS * firstStageStride;
        const uint32_t inputBytesPerStage = GROUPS_PER_STAGE * inputBytesPerGroup;
        const uint32_t scratchBytesPerStage =
            inputBytesPerStage + intermediateBytesPerGroup;
        const uint32_t outputRowBytes = static_cast<uint32_t>(tiling_->outputRowBytes);
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(rows) * rowBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

        const uint32_t inputsPerStage = inputsPerGroup * GROUPS_PER_STAGE;
        uint32_t inputStage = 0U;
        uint32_t stageIndex = 0U;
        uint32_t activeGroups = Min32(
            tiling_->inputNum / inputsPerGroup, GROUPS_PER_STAGE);
        for (uint32_t groupSlot = 0U; groupSlot < activeGroups; ++groupSlot) {
            const uint32_t scratchBase = groupSlot * inputBytesPerGroup;
            const uint32_t inputGroup = groupSlot * inputsPerGroup;
            for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                GlobalTensor<uint8_t> inputGm;
                inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    cachedInputPtrs.GetValue(inputGroup + inputSlot)));
                DataCopyPad(scratchLocal[scratchBase + inputSlot * inputStorageBytes],
                            inputGm[startRow * rowBytes], copyParams, padParams);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        while (inputStage < tiling_->inputNum) {
            const uint32_t nextInputStage = inputStage + inputsPerStage;
            const uint32_t stageSlot = stageIndex % STAGE_BUFFERS;
            const event_t stageEvent = stageSlot == 0U ? EVENT_ID0 : EVENT_ID1;
            if (nextInputStage < tiling_->inputNum) {
                const uint32_t nextStageIndex = stageIndex + 1U;
                const uint32_t nextStageSlot = nextStageIndex % STAGE_BUFFERS;
                const event_t nextStageEvent =
                    nextStageSlot == 0U ? EVENT_ID0 : EVENT_ID1;
                if (nextStageIndex >= STAGE_BUFFERS) {
                    WaitFlag<HardEvent::V_MTE2>(nextStageEvent);
                }
                const uint32_t nextScratchBufferBase =
                    nextStageSlot * scratchBytesPerStage;
                const uint32_t remainingGroups =
                    (tiling_->inputNum - nextInputStage) / inputsPerGroup;
                const uint32_t nextActiveGroups =
                    Min32(remainingGroups, GROUPS_PER_STAGE);
                for (uint32_t groupSlot = 0U; groupSlot < nextActiveGroups;
                     ++groupSlot) {
                    const uint32_t scratchBase = nextScratchBufferBase +
                        groupSlot * inputBytesPerGroup;
                    const uint32_t inputGroup =
                        nextInputStage + groupSlot * inputsPerGroup;
                    for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup;
                         ++inputSlot) {
                        GlobalTensor<uint8_t> inputGm;
                        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                            cachedInputPtrs.GetValue(inputGroup + inputSlot)));
                        DataCopyPad(
                            scratchLocal[scratchBase + inputSlot * inputStorageBytes],
                            inputGm[startRow * rowBytes], copyParams, padParams);
                    }
                }
                SetFlag<HardEvent::MTE2_V>(nextStageEvent);
            }

            WaitFlag<HardEvent::MTE2_V>(stageEvent);
            const uint32_t scratchBufferBase = stageSlot * scratchBytesPerStage;
            const uint32_t remainingGroups =
                (tiling_->inputNum - inputStage) / inputsPerGroup;
            activeGroups = Min32(remainingGroups, GROUPS_PER_STAGE);
            for (uint32_t groupSlot = 0U; groupSlot < activeGroups; ++groupSlot) {
                const uint32_t scratchBase = scratchBufferBase +
                    groupSlot * inputBytesPerGroup;
                const uint32_t intermediateBase =
                    scratchBufferBase + inputBytesPerStage;
                for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                    uint64_t sourceAddresses[MATRIX_ROWS];
                    uint64_t destinationAddresses[MATRIX_ROWS];
                    const uint32_t inputBase =
                        scratchBase + inputSlot * inputStorageBytes;
                    const uint32_t stageBase = intermediateBase +
                        inputSlot * MATRIX_ROWS * firstStageStride;
                    for (uint32_t lane = 0U; lane < MATRIX_ROWS; ++lane) {
                        sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                            scratchLocal[inputBase + lane * rowBytes].GetPhyAddr());
                        destinationAddresses[lane] = reinterpret_cast<uint64_t>(
                            scratchLocal[stageBase + lane * firstStageStride].GetPhyAddr());
                    }
                    set_va_reg_sb(VA0, destinationAddresses);
                    set_va_reg_sb(VA1, destinationAddresses + 8U);
                    set_va_reg_sb(VA2, sourceAddresses);
                    set_va_reg_sb(VA3, sourceAddresses + 8U);
                    scatter_vnchwconv_b16(VA0, VA2, static_cast<uint8_t>(repeats),
                                          1U, static_cast<uint16_t>(rowBytes / 2U));
                }

                uint64_t sourceAddresses[MATRIX_ROWS];
                uint64_t destinationAddresses[MATRIX_ROWS];
                for (uint32_t inputSlot = 0U; inputSlot < inputsPerGroup; ++inputSlot) {
                    const uint32_t stageBase = intermediateBase +
                        inputSlot * MATRIX_ROWS * firstStageStride;
                    for (uint32_t element = 0U; element < elementsPerInput; ++element) {
                        const uint32_t lane = inputSlot * elementsPerInput + element;
                        sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                            scratchLocal[stageBase + element * firstStageStride].GetPhyAddr());
                    }
                }
                const uint32_t inputGroup = inputStage + groupSlot * inputsPerGroup;
                const uint32_t groupOutputOffset = inputGroup * rowBytes;
                for (uint32_t row = 0U; row < MATRIX_ROWS; ++row) {
                    destinationAddresses[row] = reinterpret_cast<uint64_t>(
                        outputLocal[row * outputRowBytes + groupOutputOffset].GetPhyAddr());
                }
                set_va_reg_sb(VA0, destinationAddresses);
                set_va_reg_sb(VA1, destinationAddresses + 8U);
                set_va_reg_sb(VA2, sourceAddresses);
                set_va_reg_sb(VA3, sourceAddresses + 8U);
                scatter_vnchwconv_b16(
                    VA0, VA2, static_cast<uint8_t>(repeats),
                    static_cast<uint16_t>(MATRIX_ROWS * outputRowBytes / DATA_BLOCK_BYTES),
                    1U);
            }
            SetFlag<HardEvent::V_MTE2>(stageEvent);
            inputStage = nextInputStage;
            ++stageIndex;
        }
        // The last use of each generation emits a reuse token with no later
        // MTE2 producer to consume it.  Drain both generations before the UB
        // scratch leaves scope.
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void AssembleNarrowB8Rows(uint64_t startRow, uint16_t rows,
                                                LocalTensor<uint8_t> scratchLocal,
                                                LocalTensor<uint8_t> outputLocal,
                                                LocalTensor<uint64_t> cachedInputPtrs)
    {
        constexpr uint32_t MATRIX_INPUTS = 16U;
        constexpr uint32_t MATRIX_ROWS = 16U;
        const uint32_t outputRowBytes = static_cast<uint32_t>(tiling_->outputRowBytes);
        const uint32_t inputStorageBytes = AlignUp(static_cast<uint32_t>(rows),
            DATA_BLOCK_BYTES);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(rows), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            GlobalTensor<uint8_t> inputGm;
            if constexpr (CACHE_NARROW_POINTERS) {
                inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    cachedInputPtrs.GetValue(inputIdx)));
            } else {
                inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
            }
            DataCopyPad(scratchLocal[inputIdx * inputStorageBytes],
                        inputGm[startRow], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const uint8_t lowRepeats = static_cast<uint8_t>(
            (static_cast<uint32_t>(rows) + 31U) / 32U);
        const uint8_t highRepeats = rows <= MATRIX_ROWS ? 0U :
            static_cast<uint8_t>((static_cast<uint32_t>(rows) + 15U) / 32U);
        const uint16_t destinationRepeatStride = static_cast<uint16_t>(outputRowBytes);
        for (uint32_t inputGroup = 0U; inputGroup < tiling_->inputNum;
             inputGroup += MATRIX_INPUTS) {
            uint64_t sourceAddresses[MATRIX_INPUTS];
            for (uint32_t lane = 0U; lane < MATRIX_INPUTS; ++lane) {
                sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[(inputGroup + lane) * inputStorageBytes].GetPhyAddr());
            }
            const uint32_t destinationBase = inputGroup / DATA_BLOCK_BYTES *
                DATA_BLOCK_BYTES;
            uint64_t lowDestinationAddresses[MATRIX_INPUTS];
            uint64_t highDestinationAddresses[MATRIX_INPUTS];
            for (uint32_t row = 0U; row < MATRIX_ROWS; ++row) {
                lowDestinationAddresses[row] = reinterpret_cast<uint64_t>(
                    outputLocal[row * outputRowBytes + destinationBase].GetPhyAddr());
                highDestinationAddresses[row] = reinterpret_cast<uint64_t>(
                    outputLocal[(row + MATRIX_ROWS) * outputRowBytes +
                                destinationBase].GetPhyAddr());
            }
            set_va_reg_sb(VA0, lowDestinationAddresses);
            set_va_reg_sb(VA1, lowDestinationAddresses + 8U);
            set_va_reg_sb(VA2, sourceAddresses);
            set_va_reg_sb(VA3, sourceAddresses + 8U);
            if ((inputGroup & 16U) == 0U) {
                scatter_vnchwconv_b8(VA0, VA2, lowRepeats,
                                     destinationRepeatStride, 1U, false, false);
            } else {
                scatter_vnchwconv_b8(VA0, VA2, lowRepeats,
                                     destinationRepeatStride, 1U, true, false);
            }
            if (highRepeats != 0U) {
                set_va_reg_sb(VA0, highDestinationAddresses);
                set_va_reg_sb(VA1, highDestinationAddresses + 8U);
                if ((inputGroup & 16U) == 0U) {
                    scatter_vnchwconv_b8(VA0, VA2, highRepeats,
                                         destinationRepeatStride, 1U, false, true);
                } else {
                    scatter_vnchwconv_b8(VA0, VA2, highRepeats,
                                         destinationRepeatStride, 1U, true, true);
                }
            }
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    template <typename T>
    __aicore__ inline void CopyVectorRows(LocalTensor<uint8_t> destinationBytes,
                                           LocalTensor<uint8_t> sourceBytes, uint16_t rows,
                                           uint64_t outputPrefix, uint32_t rowBytes,
                                           uint32_t inputRowStride)
    {
        constexpr uint32_t ELEMENT_BYTES = sizeof(T);
        constexpr uint32_t VECTOR_ELEMENTS = VECTOR_BYTES / ELEMENT_BYTES;
        const uint32_t leftElements = static_cast<uint32_t>(outputPrefix % DATA_BLOCK_BYTES) /
            ELEMENT_BYTES;
        const uint32_t destinationBaseBytes = static_cast<uint32_t>(
            outputPrefix - outputPrefix % DATA_BLOCK_BYTES);
        LocalTensor<T> source = sourceBytes.template ReinterpretCast<T>();
        LocalTensor<T> destination = destinationBytes[destinationBaseBytes].template ReinterpretCast<T>();
        const uint32_t outputStrideBlocks = static_cast<uint32_t>(
            tiling_->outputRowBytes / DATA_BLOCK_BYTES);
        const uint32_t inputStrideBlocks = inputRowStride / DATA_BLOCK_BYTES;
        const bool repeatRows = rows <= 255U && outputStrideBlocks <= 255U && inputStrideBlocks <= 255U;

        uint32_t copiedElements = 0U;
        const uint32_t totalElements = rowBytes / ELEMENT_BYTES;
        while (copiedElements < totalElements) {
            const uint32_t physicalElement = leftElements + copiedElements;
            const uint32_t vectorBase = physicalElement / VECTOR_ELEMENTS * VECTOR_ELEMENTS;
            const uint32_t lane = physicalElement - vectorBase;
            const uint32_t count = Min32(totalElements - copiedElements, VECTOR_ELEMENTS - lane);
            uint64_t mask[2] = {0U, 0U};
            MakeMask(mask, lane, count);
            if (repeatRows) {
                UnaryRepeatParams params{1U, 1U,
                    static_cast<uint8_t>(outputStrideBlocks),
                    static_cast<uint8_t>(inputStrideBlocks)};
                Adds<T>(destination[vectorBase], source[vectorBase], static_cast<T>(0),
                        mask, static_cast<uint8_t>(rows), params);
            } else {
                for (uint16_t row = 0U; row < rows; ++row) {
                    const uint32_t destinationRow = static_cast<uint32_t>(
                        row * tiling_->outputRowBytes / ELEMENT_BYTES);
                    const uint32_t sourceRow = row * inputRowStride / ELEMENT_BYTES;
                    UnaryRepeatParams params{1U, 1U, 8U, 8U};
                    Adds<T>(destination[destinationRow + vectorBase],
                            source[sourceRow + vectorBase], static_cast<T>(0), mask, 1U, params);
                }
            }
            copiedElements += count;
        }
    }

    __aicore__ inline void CopyOutput(uint64_t outputOffset, LocalTensor<uint8_t> outputLocal,
                                      uint32_t outputBytes, bool sourceReusedByMte2)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyOutParams{1U, outputBytes, 0U, 0U, 0U};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyOutParams);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        if (sourceReusedByMte2) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void MakeMask(uint64_t mask[2], uint32_t start, uint32_t count) const
    {
        const uint32_t lowStart = start < 64U ? start : 64U;
        const uint32_t lowEnd = Min32(start + count, 64U);
        if (lowEnd > lowStart) {
            mask[0] = LowBits(lowEnd - lowStart) << lowStart;
        }
        const uint32_t highStart = start > 64U ? start - 64U : 0U;
        const uint32_t highEndAbsolute = Min32(start + count, 128U);
        const uint32_t highEnd = highEndAbsolute > 64U ? highEndAbsolute - 64U : 0U;
        if (highEnd > highStart) {
            mask[1] = LowBits(highEnd - highStart) << highStart;
        }
    }

    __aicore__ inline uint64_t LowBits(uint32_t count) const
    {
        return count >= 64U ? ~static_cast<uint64_t>(0U) :
            (count == 0U ? 0U : (static_cast<uint64_t>(1U) << count) - 1U);
    }

    __aicore__ inline uint32_t Min32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment) const
    {
        return alignment == 0U ? value : (value + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline void ProcessByteRange()
    {
        uint64_t globalOffset = tiling_->blockFactorBytes * static_cast<uint64_t>(GetBlockIdx());
        if (globalOffset >= tiling_->totalBytes) {
            return;
        }
        const uint64_t endOffset =
            Min(globalOffset + tiling_->blockFactorBytes, tiling_->totalBytes);

        while (globalOffset < endOffset) {
            const uint64_t outerIdx = globalOffset / tiling_->outputRowBytes;
            const uint64_t rowOffset = globalOffset - outerIdx * tiling_->outputRowBytes;
            uint64_t inputPrefix = 0U;
            uint32_t inputIdx = 0U;
            uint32_t inputRowBytes = InputRowBytes(inputIdx);
            while (inputIdx < tiling_->inputNum &&
                   rowOffset >= inputPrefix + inputRowBytes) {
                inputPrefix += inputRowBytes;
                ++inputIdx;
                if (inputIdx < tiling_->inputNum) {
                    inputRowBytes = InputRowBytes(inputIdx);
                }
            }
            if (inputIdx >= tiling_->inputNum) {
                return;
            }

            const uint64_t offsetInInputRow = rowOffset - inputPrefix;
            const uint64_t contiguousBytes = inputRowBytes - offsetInInputRow;
            uint64_t copyBytes = Min(contiguousBytes, endOffset - globalOffset);
            uint64_t sourceOffset = outerIdx * inputRowBytes + offsetInInputRow;
            __gm__ uint8_t* source = inputList_.GetDataPtr<uint8_t>(inputIdx);

            while (copyBytes > 0U) {
                const uint32_t currentBytes = static_cast<uint32_t>(Min(copyBytes,
                    static_cast<uint64_t>(tiling_->tileBytes)));
                Copy(source, sourceOffset, globalOffset, currentBytes);
                sourceOffset += currentBytes;
                globalOffset += currentBytes;
                copyBytes -= currentBytes;
            }
        }
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t InputRowBytes(uint32_t inputIdx) const
    {
        const uint64_t shapeOffset = static_cast<uint64_t>(inputIdx) * descStructSize_ +
            1U + tiling_->concatDim;
        const uint64_t axisElements = shapeStart_[shapeOffset];
        return static_cast<uint32_t>(axisElements * tiling_->innerElements *
                                     tiling_->elementBytes);
    }

    __aicore__ inline void Copy(__gm__ uint8_t* source, uint64_t sourceOffset,
                                uint64_t outputOffset, uint32_t bytes)
    {
        LocalTensor<uint8_t> local = copyBuffer_.Get<uint8_t>();
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(source);
        DataCopyExtParams copyParams{1U, bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(local, inputGm[sourceOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(outputGm_[outputOffset], local, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> copyBuffer_;
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    TBuf<TPosition::VECCALC> directBuffer2_;
    TBuf<TPosition::VECCALC> directBuffer3_;
    TBuf<TPosition::VECCALC> directBuffer4_;
    ListTensorDesc inputList_;
    __gm__ uint64_t* inputDataPtrs_ = nullptr;
    __gm__ uint64_t* shapeStart_ = nullptr;
    uint32_t descStructSize_ = 0U;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

class ConcatFlatPipelineKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        shapeStart_ = reinterpret_cast<__gm__ uint64_t*>(inputs) + 1U;
        descStructSize_ = static_cast<uint32_t>(*shapeStart_ & 0xffffffffULL) + 1U;
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        uint64_t globalOffset = tiling_->blockFactorBytes *
            static_cast<uint64_t>(GetBlockIdx());
        if (globalOffset >= tiling_->totalBytes) {
            return;
        }
        const uint64_t endOffset = Min(
            globalOffset + tiling_->blockFactorBytes, tiling_->totalBytes);
        const uint32_t tileBytes = tiling_->tileBytes;
        LocalTensor<uint8_t> local0(TPosition::VECCALC, 0U, tileBytes);
        LocalTensor<uint8_t> local1(TPosition::VECCALC, tileBytes, tileBytes);
        bool pending0 = false;
        bool pending1 = false;
        uint32_t slot = 0U;

        while (globalOffset < endOffset) {
            const uint64_t outerIdx = globalOffset / tiling_->outputRowBytes;
            const uint64_t rowOffset =
                globalOffset - outerIdx * tiling_->outputRowBytes;
            uint64_t inputPrefix = 0U;
            uint32_t inputIdx = 0U;
            uint32_t inputRowBytes = InputRowBytes(inputIdx);
            while (inputIdx < tiling_->inputNum &&
                   rowOffset >= inputPrefix + inputRowBytes) {
                inputPrefix += inputRowBytes;
                ++inputIdx;
                if (inputIdx < tiling_->inputNum) {
                    inputRowBytes = InputRowBytes(inputIdx);
                }
            }
            if (inputIdx >= tiling_->inputNum) {
                break;
            }

            const uint64_t offsetInInputRow = rowOffset - inputPrefix;
            uint64_t copyBytes = Min(
                static_cast<uint64_t>(inputRowBytes) - offsetInInputRow,
                endOffset - globalOffset);
            uint64_t sourceOffset =
                outerIdx * inputRowBytes + offsetInInputRow;
            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));

            while (copyBytes != 0U) {
                const uint32_t currentBytes = static_cast<uint32_t>(Min(
                    copyBytes, static_cast<uint64_t>(tileBytes)));
                if (slot == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    Copy(inputGm, sourceOffset, globalOffset, currentBytes,
                         local0, EVENT_ID0);
                    pending0 = true;
                } else {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    Copy(inputGm, sourceOffset, globalOffset, currentBytes,
                         local1, EVENT_ID1);
                    pending1 = true;
                }
                slot ^= 1U;
                sourceOffset += currentBytes;
                globalOffset += currentBytes;
                copyBytes -= currentBytes;
            }
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

private:
    __aicore__ inline uint32_t InputRowBytes(uint32_t inputIdx) const
    {
        const uint64_t shapeOffset = static_cast<uint64_t>(inputIdx) *
            descStructSize_ + 1U + tiling_->concatDim;
        const uint64_t axisElements = shapeStart_[shapeOffset];
        return static_cast<uint32_t>(axisElements * tiling_->innerElements *
                                     tiling_->elementBytes);
    }

    __aicore__ inline void Copy(GlobalTensor<uint8_t> inputGm,
                                uint64_t sourceOffset, uint64_t outputOffset,
                                uint32_t bytes, LocalTensor<uint8_t> local,
                                event_t eventId)
    {
        DataCopyExtParams copyParams{1U, bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(local, inputGm[sourceOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        DataCopyPad(outputGm_[outputOffset], local, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    ListTensorDesc inputList_;
    __gm__ uint64_t* shapeStart_ = nullptr;
    uint32_t descStructSize_ = 0U;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

// Two complete, flat input batches feed one full-row or two partial slabs.
// All addresses below are bytes unless a LocalTensor<uint32_t> is indexed.
template <bool PartialSlabs>
class ConcatWideRowAssemblyKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
#if defined(L2_CACHE_HINT)
        l2BypassOffset_ = g_opSystemRunCfg.l2Cacheoffset;
#endif
    }

    __aicore__ inline void Process()
    {
        const uint64_t first = tiling_->blockFactorRows * GetBlockIdx();
        if (first >= tiling_->outerRows ||
            tiling_->launchPadding[9U] != (PartialSlabs ? 2U : 1U)) {
            return;
        }
        const uint64_t end = Min(first + tiling_->blockFactorRows,
                                 tiling_->outerRows);
        rows_ = tiling_->rowsPerBatch;
        arenaBytes_ = tiling_->launchPadding[8U];
        outputBytes_ = static_cast<uint32_t>(tiling_->outputRowBytes);
        const uint32_t bandBytes = PartialSlabs ? tiling_->launchPadding[10U] : outputBytes_;
        const uint32_t slabStorage = Align32(bandBytes);
        const uint32_t outputStorage = (PartialSlabs ? 2U : 1U) * slabStorage;
        LocalTensor<uint8_t> arena0(TPosition::VECCALC, 0U, arenaBytes_);
        LocalTensor<uint8_t> arena1(TPosition::VECCALC, arenaBytes_, arenaBytes_);
        LocalTensor<uint8_t> slabs(TPosition::VECCALC, 2U * arenaBytes_, outputStorage);
        LocalTensor<uint32_t> indices(TPosition::VECCALC,
            2U * arenaBytes_ + outputStorage, INDEX_BYTES / sizeof(uint32_t));
        auto signedIndices = indices.ReinterpretCast<int32_t>();
        for (uint32_t lane = 0U; lane < DATA_BLOCK_BYTES / sizeof(uint32_t); ++lane) {
            signedIndices.SetValue(lane, static_cast<int32_t>(lane * sizeof(uint32_t)));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        // Double the initialized prefix. Every consumer follows its Vector
        // producer; no TPipe-owned scratch or implicit event allocation.
        for (uint32_t words = DATA_BLOCK_BYTES / sizeof(uint32_t);
             words < INDEX_BYTES / sizeof(uint32_t); words *= 2U) {
            Adds(signedIndices[words], signedIndices,
                 static_cast<int32_t>(words * sizeof(uint32_t)), words);
            PipeBarrier<PIPE_V>();
        }

        uint32_t arenaOffset = 0U;
        for (uint32_t input = 0U; input < tiling_->inputNum; ++input) {
            const uint32_t bytes = tiling_->launchPadding[input];
            if (bytes == 0U) {
                continue;
            }
            addresses_[active_] = reinterpret_cast<uint64_t>(
                inputList_.GetDataPtr<uint8_t>(input)) + l2BypassOffset_;
            widths_[active_] = bytes;
            slices_[active_] = arenaOffset;
            arenaOffset += Align32(rows_ * bytes);
            ++active_;
        }

        LoadBatch(arena0, first, static_cast<uint32_t>(Min(rows_, end - first)), EVENT_ID1);
        uint32_t releasedMask = 0U;
        uint32_t slot = 0U;
        uint32_t outputPendingMask = 0U;
        for (uint64_t row = first; row < end; row += rows_, slot ^= 1U) {
            const uint64_t nextRow = row + rows_;
            if (nextRow < end) {
                const uint32_t nextSlot = slot ^ 1U;
                const event_t nextEvent = nextSlot == 0U ? EVENT_ID1 : EVENT_ID2;
                if ((releasedMask & (1U << nextSlot)) != 0U) {
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    releasedMask &= ~(1U << nextSlot);
                }
                LoadBatch(nextSlot == 0U ? arena0 : arena1, nextRow,
                    static_cast<uint32_t>(Min(rows_, end - nextRow)), nextEvent);
            }
            const event_t event = slot == 0U ? EVENT_ID1 : EVENT_ID2;
            const LocalTensor<uint8_t> arena = slot == 0U ? arena0 : arena1;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t count = static_cast<uint32_t>(Min(rows_, end - row));
            for (uint32_t innerRow = 0U; innerRow < count; ++innerRow) {
                for (uint32_t band = 0U; band < (PartialSlabs ? 2U : 1U); ++band) {
                    const event_t outputEvent = band == 0U ? EVENT_ID0 : EVENT_ID5;
                    if ((outputPendingMask & (1U << band)) != 0U) {
                        WaitFlag<HardEvent::MTE3_V>(outputEvent);
                    }
                    const uint32_t begin = band * bandBytes;
                    const uint32_t finish = static_cast<uint32_t>(Min(begin + bandBytes, outputBytes_));
                    const LocalTensor<uint8_t> slab = slabs[band * slabStorage];
                    uint32_t outputPrefix = 0U;
                    for (uint32_t input = 0U; input < active_; ++input) {
                        const uint32_t left = outputPrefix > begin ? outputPrefix : begin;
                        const uint32_t right = static_cast<uint32_t>(Min(outputPrefix + widths_[input], finish));
                        if (left < right) {
                            CopyWide(slab.ReinterpretCast<uint32_t>(),
                                arena.ReinterpretCast<uint32_t>(), indices,
                                left - begin, slices_[input] + innerRow * widths_[input] + left - outputPrefix,
                                right - left);
                        }
                        outputPrefix += widths_[input];
                    }
                    SetFlag<HardEvent::V_MTE3>(outputEvent);
                    WaitFlag<HardEvent::V_MTE3>(outputEvent);
                    const DataCopyExtParams params{1U, finish - begin, 0U, 0U, 0U};
                    DataCopyPad(outputGm_[(row + innerRow) * outputBytes_ + begin], slab, params);
                    SetFlag<HardEvent::MTE3_V>(outputEvent);
                    outputPendingMask |= 1U << band;
                }
            }
            SetFlag<HardEvent::V_MTE2>(event);
            releasedMask |= 1U << slot;
        }
        if ((releasedMask & 1U) != 0U) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        if ((releasedMask & 2U) != 0U) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
        if ((outputPendingMask & 1U) != 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        if (PartialSlabs && (outputPendingMask & 2U) != 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        }
    }

private:
    static constexpr uint32_t INDEX_BYTES = 16U * 1024U;

    __aicore__ inline void CopyWide(
        LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src,
        LocalTensor<uint32_t> indices, uint32_t destination,
        uint32_t source, uint32_t bytes)
    {
        const uint32_t left = destination % DATA_BLOCK_BYTES;
        if (left != 0U) {
            const uint32_t head = static_cast<uint32_t>(Min(bytes, DATA_BLOCK_BYTES - left));
            const uint32_t lane = left / sizeof(uint32_t);
            const uint32_t lanes = head / sizeof(uint32_t);
            uint64_t mask[2] = {((1ULL << lanes) - 1U) << lane, 0U};
            // The preceding input's masked tail shares this destination
            // block. Complete it before this independent-lane writer.
            PipeBarrier<PIPE_V>();
            Gather(dst[(destination - left) / sizeof(uint32_t)], src,
                indices, source - left, mask, 1U, 8U);
            destination += head;
            source += head;
            bytes -= head;
        }
        // The aligned subset does not need an index-table read per word.
        // UB-to-UB DataCopy is a Vector-pipeline bitwise block movement on
        // C220, covered by the same arena/slab lifetime events as Gather.
        if (source % DATA_BLOCK_BYTES == 0U) {
            while (bytes >= DATA_BLOCK_BYTES) {
                const uint32_t blocks = static_cast<uint32_t>(Min(
                    bytes / DATA_BLOCK_BYTES, 4095U));
                const DataCopyParams params{1U, static_cast<uint16_t>(blocks), 0U, 0U};
                DataCopy(dst[destination / sizeof(uint32_t)],
                         src[source / sizeof(uint32_t)], params);
                const uint32_t copied = blocks * DATA_BLOCK_BYTES;
                destination += copied;
                source += copied;
                bytes -= copied;
            }
        }
        while (bytes >= VECTOR_BYTES) {
            const uint32_t repeats = static_cast<uint32_t>(Min(
                bytes / VECTOR_BYTES, INDEX_BYTES / VECTOR_BYTES));
            Gather(dst[destination / sizeof(uint32_t)], src, indices,
                source, static_cast<uint64_t>(64U), static_cast<uint8_t>(repeats), 8U);
            const uint32_t copied = repeats * VECTOR_BYTES;
            destination += copied;
            source += copied;
            bytes -= copied;
        }
        if (bytes != 0U) {
            Gather(dst[destination / sizeof(uint32_t)], src, indices,
                source, static_cast<uint64_t>(bytes / sizeof(uint32_t)), 1U, 8U);
        }
    }

    __aicore__ inline void LoadBatch(LocalTensor<uint8_t> arena,
        uint64_t row, uint32_t count, event_t event)
    {
        for (uint32_t input = 0U; input < active_; ++input) {
            GlobalTensor<uint8_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(addresses_[input]));
            const DataCopyExtParams params{1U, count * widths_[input], 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> noPad{false, 0U, 0U, 0U};
            DataCopyPad(arena[slices_[input]], gm[row * widths_[input]], params, noPad);
        }
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline uint64_t Min(uint64_t a, uint64_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align32(uint32_t bytes) const { return (bytes + 31U) / 32U * 32U; }
    ListTensorDesc inputList_;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
    uint64_t l2BypassOffset_ = 0U;
    uint64_t addresses_[DEDICATED_DIRECT_MAX_INPUTS] = {};
    uint32_t widths_[DEDICATED_DIRECT_MAX_INPUTS] = {};
    uint32_t slices_[DEDICATED_DIRECT_MAX_INPUTS] = {};
    uint32_t active_ = 0U;
    uint32_t rows_ = 0U;
    uint32_t arenaBytes_ = 0U;
    uint32_t outputBytes_ = 0U;
};

class ConcatDedicatedDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
#if defined(L2_CACHE_HINT)
        // The direct-row path issues many small descriptors.  Cache the
        // runtime L2 alias once during initialization instead of re-reading
        // the system configuration for every descriptor submission.
        l2BypassOffset_ = g_opSystemRunCfg.l2Cacheoffset;
#endif
    }

    __aicore__ inline void Process()
    {
        const uint64_t startRow = tiling_->blockFactorRows *
            static_cast<uint64_t>(GetBlockIdx());
        if (startRow >= tiling_->outerRows) {
            return;
        }
        const uint64_t endRow = Min(
            startRow + tiling_->blockFactorRows, tiling_->outerRows);
        const uint32_t rowsPerBatch = tiling_->rowsPerBatch;
        const uint32_t slotBytes0 = rowsPerBatch *
            tiling_->launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD];
        const uint32_t slotBytes1 = rowsPerBatch *
            tiling_->launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD + 1U];
        const uint32_t slotBytes2 = rowsPerBatch *
            tiling_->launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD + 2U];
        const uint32_t slotBytes3 = rowsPerBatch *
            tiling_->launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD + 3U];
        const uint32_t slotBytes4 = rowsPerBatch *
            tiling_->launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD + 4U];
        LocalTensor<uint8_t> local0(TPosition::VECCALC, 0U, slotBytes0);
        LocalTensor<uint8_t> local1(
            TPosition::VECCALC, slotBytes0, slotBytes1);
        LocalTensor<uint8_t> local2(
            TPosition::VECCALC, slotBytes0 + slotBytes1, slotBytes2);
        LocalTensor<uint8_t> local3(
            TPosition::VECCALC, slotBytes0 + slotBytes1 + slotBytes2,
            slotBytes3);
        LocalTensor<uint8_t> local4(
            TPosition::VECCALC,
            slotBytes0 + slotBytes1 + slotBytes2 + slotBytes3,
            slotBytes4);

        uint64_t inputAddresses[DEDICATED_DIRECT_MAX_INPUTS] = {};
        uint32_t activeInputs[DEDICATED_DIRECT_MAX_SLOTS] = {};
        uint32_t activeRowBytes[DEDICATED_DIRECT_MAX_SLOTS] = {};
        uint64_t activeOutputOffsets[DEDICATED_DIRECT_MAX_SLOTS] = {};
        uint32_t activeCount = 0U;
        uint64_t activeOutputPrefix = 0U;
        for (uint32_t input = 0U; input < tiling_->inputNum; ++input) {
            inputAddresses[input] = reinterpret_cast<uint64_t>(
                inputList_.GetDataPtr<uint8_t>(input));
            const uint32_t rowBytes = tiling_->launchPadding[input];
            if (rowBytes != 0U && activeCount < DEDICATED_DIRECT_MAX_SLOTS) {
                activeInputs[activeCount] = input;
                activeRowBytes[activeCount] = rowBytes;
                activeOutputOffsets[activeCount] = activeOutputPrefix;
                ++activeCount;
            }
            activeOutputPrefix += rowBytes;
        }

        uint32_t pendingMask = 0U;
        for (uint64_t row = startRow; row < endRow;
             row += rowsPerBatch) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(rowsPerBatch)));
            uint32_t slot = 0U;
            // Submit every input-side DMA for this batch before exposing the
            // first completion to Vector/MTE3.  The old loop waited inside
            // CopyRows after each descriptor, which left MTE2 with only one
            // request in flight while Scalar drained the preceding output.
            // Slots remain independently protected by the persistent
            // MTE3->MTE2 events at the next batch boundary.
            for (slot = 0U; slot < activeCount; ++slot) {
                const uint32_t input = activeInputs[slot];
                const uint32_t rowBytes = activeRowBytes[slot];
                const uint32_t pendingBit = 1U << slot;
                if ((pendingMask & pendingBit) != 0U) {
                    WaitSlot(slot);
                    pendingMask &= ~pendingBit;
                }
                LocalTensor<uint8_t> local = local0;
                event_t event = EVENT_ID0;
                if (slot == 1U) {
                    local = local1;
                    event = EVENT_ID1;
                } else if (slot == 2U) {
                    local = local2;
                    event = EVENT_ID2;
                } else if (slot == 3U) {
                    local = local3;
                    event = EVENT_ID3;
                } else if (slot == 4U) {
                    local = local4;
                    event = EVENT_ID4;
                }
                LoadRows(inputAddresses[input], row, rows, rowBytes,
                         local, event);
                pendingMask |= pendingBit;
            }

            for (slot = 0U; slot < activeCount; ++slot) {
                const uint32_t rowBytes = activeRowBytes[slot];
                LocalTensor<uint8_t> local = local0;
                event_t event = EVENT_ID0;
                if (slot == 1U) {
                    local = local1;
                    event = EVENT_ID1;
                } else if (slot == 2U) {
                    local = local2;
                    event = EVENT_ID2;
                } else if (slot == 3U) {
                    local = local3;
                    event = EVENT_ID3;
                } else if (slot == 4U) {
                    local = local4;
                    event = EVENT_ID4;
                }
                StoreRows(row, rows, activeOutputOffsets[slot], rowBytes,
                          local, event);
            }
        }
        Drain(pendingMask);
    }

private:
    __aicore__ inline void WaitSlot(uint32_t slot)
    {
        if (slot == 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else if (slot == 1U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else if (slot == 2U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        } else if (slot == 3U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
    }

    __aicore__ inline void Drain(uint32_t pendingMask)
    {
        if ((pendingMask & 1U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if ((pendingMask & 2U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if ((pendingMask & 4U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if ((pendingMask & 8U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if ((pendingMask & 16U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
    }

    __aicore__ inline void LoadRows(
        uint64_t inputAddress, uint64_t startRow, uint16_t rows,
        uint32_t rowBytes, LocalTensor<uint8_t> local, event_t event)
    {
#if defined(L2_CACHE_HINT)
        // Dedicated direct rows are consumed exactly once.  Keep those GM
        // reads out of L2, as in the dynamic GroupOwner path, so the long
        // unaligned input stream cannot evict reusable runtime metadata or
        // output lines.  The alias is applied only to input reads; output
        // stores retain the normal cache policy.
        inputAddress += l2BypassOffset_;
#endif
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress));
        const uint64_t inputBlocks = static_cast<uint64_t>(rows) *
            rowBytes / DATA_BLOCK_BYTES;
        if (rowBytes % DATA_BLOCK_BYTES == 0U && inputBlocks <= 0xffffU) {
            DataCopyParams copyInParams{
                1U, static_cast<uint16_t>(inputBlocks), 0U, 0U};
            DataCopy(local, inputGm[startRow * rowBytes], copyInParams);
        } else {
            DataCopyExtParams copyInParams{rows, rowBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> noPadding{false, 0U, 0U, 0U};
            DataCopyPad(local, inputGm[startRow * rowBytes],
                        copyInParams, noPadding);
        }
        SetFlag<HardEvent::MTE2_MTE3>(event);
    }

    __aicore__ inline void StoreRows(
        uint64_t startRow, uint16_t rows, uint64_t outputPrefix,
        uint32_t rowBytes, LocalTensor<uint8_t> local, event_t event)
    {
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{
            rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(
            outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
            local, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(event);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    ListTensorDesc inputList_;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
    uint64_t l2BypassOffset_ = 0U;
};

class ConcatIncrementalFlatPipelineKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
#if defined(L2_CACHE_HINT)
        l2BypassOffset_ = g_opSystemRunCfg.l2Cacheoffset;
#endif
    }

    __aicore__ inline void Process()
    {
        uint64_t globalOffset = tiling_->blockFactorBytes *
            static_cast<uint64_t>(GetBlockIdx());
        if (globalOffset >= tiling_->totalBytes) {
            return;
        }
        const uint64_t endOffset = Min(
            globalOffset + tiling_->blockFactorBytes, tiling_->totalBytes);
        const uint32_t tileBytes = tiling_->tileBytes;
        LocalTensor<uint8_t> local0(TPosition::VECCALC, 0U, tileBytes);
        LocalTensor<uint8_t> local1(
            TPosition::VECCALC, tileBytes, tileBytes);

        uint64_t outerIdx = globalOffset / tiling_->outputRowBytes;
        const uint64_t rowOffset =
            globalOffset - outerIdx * tiling_->outputRowBytes;
        uint64_t inputPrefix = 0U;
        uint32_t inputIdx = 0U;
        uint32_t inputRowBytes = tiling_->launchPadding[0U];
        while (inputIdx < tiling_->inputNum &&
               rowOffset >= inputPrefix + inputRowBytes) {
            inputPrefix += inputRowBytes;
            ++inputIdx;
            if (inputIdx < tiling_->inputNum) {
                inputRowBytes = tiling_->launchPadding[inputIdx];
            }
        }
        if (inputIdx >= tiling_->inputNum) {
            return;
        }

        const uint64_t offsetInInput = rowOffset - inputPrefix;
        uint64_t sourceOffset = outerIdx * inputRowBytes + offsetInInput;
        uint64_t inputRemaining = inputRowBytes - offsetInInput;
        uint32_t pendingMask = 0U;
        uint32_t slot = 0U;
        while (globalOffset < endOffset) {
            if (inputRemaining == 0U) {
                do {
                    ++inputIdx;
                    if (inputIdx == tiling_->inputNum) {
                        inputIdx = 0U;
                        ++outerIdx;
                    }
                    inputRowBytes = tiling_->launchPadding[inputIdx];
                } while (inputRowBytes == 0U);
                sourceOffset = outerIdx * inputRowBytes;
                inputRemaining = inputRowBytes;
            }

            GlobalTensor<uint8_t> inputGm;
            uint64_t inputAddress = reinterpret_cast<uint64_t>(
                inputList_.GetDataPtr<uint8_t>(inputIdx));
#if defined(L2_CACHE_HINT)
            inputAddress += l2BypassOffset_;
#endif
            inputGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t*>(inputAddress));
            while (inputRemaining != 0U && globalOffset < endOffset) {
                const uint32_t currentBytes = static_cast<uint32_t>(Min(
                    Min(inputRemaining, endOffset - globalOffset),
                    static_cast<uint64_t>(tileBytes)));
                const uint32_t pendingBit = 1U << slot;
                if ((pendingMask & pendingBit) != 0U) {
                    if (slot == 0U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    } else {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                }
                if (slot == 0U) {
                    Copy(inputGm, sourceOffset, globalOffset, currentBytes,
                         local0, EVENT_ID0);
                } else {
                    Copy(inputGm, sourceOffset, globalOffset, currentBytes,
                         local1, EVENT_ID1);
                }
                pendingMask |= pendingBit;
                slot ^= 1U;
                sourceOffset += currentBytes;
                globalOffset += currentBytes;
                inputRemaining -= currentBytes;
            }
        }
        if ((pendingMask & 1U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if ((pendingMask & 2U) != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

private:
    __aicore__ inline void Copy(
        GlobalTensor<uint8_t> inputGm, uint64_t sourceOffset,
        uint64_t outputOffset, uint32_t bytes, LocalTensor<uint8_t> local,
        event_t event)
    {
        DataCopyExtParams copyParams{1U, bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> noPadding{false, 0U, 0U, 0U};
        DataCopyPad(local, inputGm[sourceOffset], copyParams, noPadding);
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        DataCopyPad(outputGm_[outputOffset], local, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(event);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    ListTensorDesc inputList_;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
    uint64_t l2BypassOffset_ = 0U;
};

class ConcatExactRawUbKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        inputList_.Init(reinterpret_cast<__gm__ void*>(inputs));
        shapeStart_ = reinterpret_cast<__gm__ uint64_t*>(inputs) + 1U;
        descStructSize_ = static_cast<uint32_t>(*shapeStart_ & 0xffffffffULL) + 1U;
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileBytes = tiling_->tileBytes;
        const uint64_t startRow = tiling_->blockFactorRows *
            static_cast<uint64_t>(GetBlockIdx());
        const uint16_t rows = static_cast<uint16_t>(tiling_->blockFactorRows);
        LocalTensor<uint8_t> local0(TPosition::VECCALC, 0U, tileBytes);
        LocalTensor<uint8_t> local1(TPosition::VECCALC, tileBytes, tileBytes);
        LocalTensor<uint8_t> local2(TPosition::VECCALC, 2U * tileBytes, tileBytes);
        LocalTensor<uint8_t> local3(TPosition::VECCALC, 3U * tileBytes, tileBytes);
        LocalTensor<uint8_t> local4(TPosition::VECCALC, 4U * tileBytes, tileBytes);
        CopyRowsPipelined5(startRow, rows, local0, local1, local2, local3,
                           local4);
    }

private:
    __aicore__ inline void CopyRowsPipelined5(
        uint64_t startRow, uint16_t rows, LocalTensor<uint8_t> local0,
        LocalTensor<uint8_t> local1, LocalTensor<uint8_t> local2,
        LocalTensor<uint8_t> local3, LocalTensor<uint8_t> local4)
    {
        bool pending0 = false;
        bool pending1 = false;
        bool pending2 = false;
        bool pending3 = false;
        bool pending4 = false;
        uint32_t slot = 0U;
        uint64_t outputPrefix = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < tiling_->inputNum; ++inputIdx) {
            const uint32_t rowBytes = InputRowBytes(inputIdx);
            if (rowBytes != 0U) {
                if (slot == 0U) {
                    if (pending0) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local0, EVENT_ID0);
                    pending0 = true;
                } else if (slot == 1U) {
                    if (pending1) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local1, EVENT_ID1);
                    pending1 = true;
                } else if (slot == 2U) {
                    if (pending2) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local2, EVENT_ID2);
                    pending2 = true;
                } else if (slot == 3U) {
                    if (pending3) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local3, EVENT_ID3);
                    pending3 = true;
                } else {
                    if (pending4) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
                    }
                    CopyRowsDirect(inputIdx, startRow, rows, outputPrefix,
                                   rowBytes, local4, EVENT_ID4);
                    pending4 = true;
                }
                slot = slot == 4U ? 0U : slot + 1U;
            }
            outputPrefix += rowBytes;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        if (pending2) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (pending3) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if (pending4) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
    }

    __aicore__ inline void CopyRowsDirect(
        uint32_t inputIdx, uint64_t startRow, uint16_t rows,
        uint64_t outputPrefix, uint32_t rowBytes, LocalTensor<uint8_t> local,
        event_t eventId)
    {
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(inputList_.GetDataPtr<uint8_t>(inputIdx));
        if (rowBytes % DATA_BLOCK_BYTES == 0U) {
            const uint32_t inputBlocks =
                static_cast<uint32_t>(rows) * rowBytes / DATA_BLOCK_BYTES;
            DataCopyParams copyInParams{
                1U, static_cast<uint16_t>(inputBlocks), 0U, 0U};
            DataCopy(local, inputGm[startRow * rowBytes], copyInParams);
        } else {
            DataCopyExtParams copyInParams{rows, rowBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(local, inputGm[startRow * rowBytes], copyInParams,
                        padParams);
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        const uint32_t outputStride = static_cast<uint32_t>(
            tiling_->outputRowBytes - rowBytes);
        DataCopyExtParams copyOutParams{
            rows, rowBytes, 0U, outputStride, 0U};
        DataCopyPad(
            outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
            local, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline uint32_t InputRowBytes(uint32_t inputIdx) const
    {
        const uint64_t shapeOffset =
            static_cast<uint64_t>(inputIdx) * descStructSize_ + 1U +
            tiling_->concatDim;
        const uint64_t axisElements = shapeStart_[shapeOffset];
        return static_cast<uint32_t>(axisElements * tiling_->innerElements *
                                     tiling_->elementBytes);
    }

    ListTensorDesc inputList_;
    __gm__ uint64_t* shapeStart_ = nullptr;
    uint32_t descStructSize_ = 0U;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

class ConcatSegmentedB32Kernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        const __gm__ uint64_t* descriptorStart =
            reinterpret_cast<__gm__ uint64_t*>(inputs);
        inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
            descriptorStart[0] / sizeof(uint64_t);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t INPUT_POINTER_BYTES =
            MAX_DYNAMIC_INPUTS * sizeof(uint64_t);
        const uint32_t scratchBytes =
            tiling_->rowsPerBatch * SCRATCH_BYTES_PER_ROW;
        LocalTensor<uint8_t> segmentLocal(
            TPosition::VECCALC, 0U, tiling_->tileBytes);
        LocalTensor<uint8_t> scratchLocal(
            TPosition::VECCALC, tiling_->tileBytes, scratchBytes);
        LocalTensor<uint64_t> cachedInputPtrs(
            TPosition::VECCALC, tiling_->tileBytes + scratchBytes,
            INPUT_POINTER_BYTES);

        GlobalTensor<uint64_t> inputPtrGm;
        inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
        DataCopy(cachedInputPtrs, inputPtrGm, MAX_DYNAMIC_INPUTS);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);

        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint64_t rowGroups = tiling_->outerRows / MATRIX_ROWS;
        const uint64_t groupsPerBlock = rowGroups / blockCount;
        const uint64_t extraGroups = rowGroups % blockCount;
        const uint64_t startGroup = blockIndex * groupsPerBlock +
            Min(blockIndex, extraGroups);
        const uint64_t assignedGroups = groupsPerBlock +
            static_cast<uint64_t>(blockIndex < extraGroups);
        uint64_t row = MATRIX_ROWS * startGroup;
        const uint64_t endRow = row + MATRIX_ROWS * assignedGroups;

        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatch(row, rows, segmentLocal, scratchLocal,
                            cachedInputPtrs);
            row += rows;
        }
    }

private:
    static constexpr uint32_t MATRIX_ROWS = 8U;
    static constexpr uint32_t ROW_BYTES = sizeof(uint32_t);
    static constexpr uint32_t INPUTS_PER_GROUP = 16U;
    static constexpr uint32_t INPUTS_PER_SEGMENT = 128U;
    static constexpr uint32_t GROUPS_PER_SEGMENT =
        INPUTS_PER_SEGMENT / INPUTS_PER_GROUP;
    static constexpr uint32_t SEGMENT_BYTES = INPUTS_PER_SEGMENT * ROW_BYTES;
    static constexpr uint32_t SCRATCH_BYTES_PER_ROW =
        2U * INPUTS_PER_GROUP * ROW_BYTES;

    __aicore__ inline void ProcessRowBatch(
        uint64_t startRow, uint16_t rows, LocalTensor<uint8_t> segmentLocal,
        LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        for (uint32_t segmentInput = 0U; segmentInput < tiling_->inputNum;
             segmentInput += INPUTS_PER_SEGMENT) {
            AssembleSegment(startRow, rows, segmentInput, segmentLocal,
                            scratchLocal, cachedInputPtrs);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopyExtParams outputParams{
                rows, SEGMENT_BYTES, 0U,
                static_cast<uint32_t>(tiling_->outputRowBytes) - SEGMENT_BYTES,
                0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes +
                          static_cast<uint64_t>(segmentInput) * ROW_BYTES],
                segmentLocal, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void AssembleSegment(
        uint64_t startRow, uint16_t rows, uint32_t segmentInput,
        LocalTensor<uint8_t> segmentLocal, LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t inputStorageBytes =
            static_cast<uint32_t>(rows) * ROW_BYTES;
        const uint32_t inputBytesPerGroup =
            INPUTS_PER_GROUP * inputStorageBytes;
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(rows) * ROW_BYTES, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

        LoadInputGroup(startRow, segmentInput, 0U, inputStorageBytes,
                       inputBytesPerGroup, copyParams, padParams, scratchLocal,
                       cachedInputPtrs);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t groupSlot = 0U; groupSlot < GROUPS_PER_SEGMENT;
             ++groupSlot) {
            const uint32_t nextGroupSlot = groupSlot + 1U;
            const uint32_t generation = groupSlot & 1U;
            const event_t generationEvent = generation == 0U ?
                EVENT_ID0 : EVENT_ID1;
            if (nextGroupSlot < GROUPS_PER_SEGMENT) {
                const uint32_t nextGeneration = nextGroupSlot & 1U;
                const event_t nextEvent = nextGeneration == 0U ?
                    EVENT_ID0 : EVENT_ID1;
                if (nextGroupSlot >= 2U) {
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                }
                LoadInputGroup(
                    startRow,
                    segmentInput + nextGroupSlot * INPUTS_PER_GROUP,
                    nextGeneration, inputStorageBytes, inputBytesPerGroup,
                    copyParams, padParams, scratchLocal, cachedInputPtrs);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            WaitFlag<HardEvent::MTE2_V>(generationEvent);
            TransposeInputGroup(groupSlot, generation, rows,
                                inputStorageBytes, inputBytesPerGroup,
                                segmentLocal, scratchLocal);
            SetFlag<HardEvent::V_MTE2>(generationEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void LoadInputGroup(
        uint64_t startRow, uint32_t inputGroup, uint32_t generation,
        uint32_t inputStorageBytes, uint32_t inputBytesPerGroup,
        const DataCopyExtParams& copyParams,
        const DataCopyPadExtParams<uint8_t>& padParams,
        LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t generationBase = generation * inputBytesPerGroup;
        for (uint32_t inputSlot = 0U; inputSlot < INPUTS_PER_GROUP;
             ++inputSlot) {
            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                cachedInputPtrs.GetValue(inputGroup + inputSlot)));
            DataCopyPad(
                scratchLocal[generationBase + inputSlot * inputStorageBytes],
                inputGm[startRow * ROW_BYTES], copyParams, padParams);
        }
    }

    __aicore__ inline void TransposeInputGroup(
        uint32_t groupSlot, uint32_t generation, uint16_t rows,
        uint32_t inputStorageBytes, uint32_t inputBytesPerGroup,
        LocalTensor<uint8_t> segmentLocal, LocalTensor<uint8_t> scratchLocal)
    {
        constexpr uint32_t OUTPUT_HALF_BYTES = 8U * ROW_BYTES;
        const uint32_t generationBase = generation * inputBytesPerGroup;
        uint64_t sourceAddresses[INPUTS_PER_GROUP];
        uint64_t destinationAddresses[INPUTS_PER_GROUP];
        for (uint32_t lane = 0U; lane < INPUTS_PER_GROUP; ++lane) {
            sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                scratchLocal[generationBase + lane * inputStorageBytes].GetPhyAddr());
            const uint32_t destinationOffset =
                (lane >> 1U) * SEGMENT_BYTES +
                groupSlot * INPUTS_PER_GROUP * ROW_BYTES +
                (lane & 1U) * OUTPUT_HALF_BYTES;
            destinationAddresses[lane] = reinterpret_cast<uint64_t>(
                segmentLocal[destinationOffset].GetPhyAddr());
        }
        set_va_reg_sb(VA0, destinationAddresses);
        set_va_reg_sb(VA1, destinationAddresses + 8U);
        set_va_reg_sb(VA2, sourceAddresses);
        set_va_reg_sb(VA3, sourceAddresses + 8U);
        scatter_vnchwconv_b32(
            VA0, VA2, static_cast<uint8_t>(rows / MATRIX_ROWS),
            static_cast<uint16_t>(MATRIX_ROWS * SEGMENT_BYTES /
                                  DATA_BLOCK_BYTES),
            1U);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __gm__ uint64_t* inputDataPtrs_ = nullptr;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

template <uint32_t SEGMENT_ROW_BYTES,
          uint32_t SEGMENT_INPUTS_PER_SEGMENT>
class ConcatSegmentedNarrowKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        const __gm__ uint64_t* descriptorStart =
            reinterpret_cast<__gm__ uint64_t*>(inputs);
        inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
            descriptorStart[0] / sizeof(uint64_t);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t INPUT_POINTER_BYTES =
            MAX_DYNAMIC_INPUTS * sizeof(uint64_t);
        const uint32_t scratchBytes = tiling_->rowsPerBatch * SCRATCH_BYTES_PER_ROW;
        LocalTensor<uint8_t> segmentLocal(
            TPosition::VECCALC, 0U, tiling_->tileBytes);
        LocalTensor<uint8_t> scratchLocal(
            TPosition::VECCALC, tiling_->tileBytes, scratchBytes);
        LocalTensor<uint64_t> cachedInputPtrs(
            TPosition::VECCALC, tiling_->tileBytes + scratchBytes,
            INPUT_POINTER_BYTES);

        GlobalTensor<uint64_t> inputPtrGm;
        inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
        DataCopy(cachedInputPtrs, inputPtrGm, MAX_DYNAMIC_INPUTS);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);

        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint64_t rowGroups = tiling_->outerRows / MATRIX_ROWS;
        const uint64_t groupsPerBlock = rowGroups / blockCount;
        const uint64_t extraGroups = rowGroups % blockCount;
        const uint64_t startGroup = blockIndex * groupsPerBlock +
            Min(blockIndex, extraGroups);
        const uint64_t assignedGroups = groupsPerBlock +
            static_cast<uint64_t>(blockIndex < extraGroups);
        uint64_t row = MATRIX_ROWS * startGroup;
        const uint64_t endRow = row + MATRIX_ROWS * assignedGroups;

        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatch(row, rows, segmentLocal, scratchLocal,
                            cachedInputPtrs);
            row += rows;
        }
    }

private:
    static constexpr uint32_t MATRIX_ROWS = 16U;
    static constexpr uint32_t ROW_BYTES = SEGMENT_ROW_BYTES;
    static constexpr uint32_t INPUTS_PER_GROUP = DATA_BLOCK_BYTES / ROW_BYTES;
    static constexpr uint32_t INPUTS_PER_SEGMENT =
        SEGMENT_INPUTS_PER_SEGMENT;
    static constexpr uint32_t GROUPS_PER_SEGMENT =
        INPUTS_PER_SEGMENT / INPUTS_PER_GROUP;
    static constexpr uint32_t SEGMENT_BYTES = INPUTS_PER_SEGMENT * ROW_BYTES;
    static constexpr uint32_t SCRATCH_BYTES_PER_ROW =
        2U * DATA_BLOCK_BYTES + INPUTS_PER_GROUP * DATA_BLOCK_BYTES;

    __aicore__ inline void ProcessRowBatch(
        uint64_t startRow, uint16_t rows, LocalTensor<uint8_t> segmentLocal,
        LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        for (uint32_t segmentInput = 0U; segmentInput < tiling_->inputNum;
             segmentInput += INPUTS_PER_SEGMENT) {
            AssembleSegment(startRow, rows, segmentInput, segmentLocal,
                            scratchLocal, cachedInputPtrs);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopyExtParams outputParams{
                rows, SEGMENT_BYTES, 0U,
                static_cast<uint32_t>(tiling_->outputRowBytes) - SEGMENT_BYTES,
                0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes +
                          static_cast<uint64_t>(segmentInput) * ROW_BYTES],
                segmentLocal, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void AssembleSegment(
        uint64_t startRow, uint16_t rows, uint32_t segmentInput,
        LocalTensor<uint8_t> segmentLocal, LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t repeats = static_cast<uint32_t>(rows) / MATRIX_ROWS;
        const uint32_t inputStorageBytes = static_cast<uint32_t>(rows) * ROW_BYTES;
        const uint32_t inputBytesPerGroup =
            INPUTS_PER_GROUP * inputStorageBytes;
        const uint32_t intermediateBase = 2U * inputBytesPerGroup;
        const uint32_t firstStageStride = repeats * DATA_BLOCK_BYTES;
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(rows) * ROW_BYTES, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

        LoadInputGroup(startRow, segmentInput, 0U, inputStorageBytes,
                       inputBytesPerGroup, copyParams, padParams, scratchLocal,
                       cachedInputPtrs);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t groupSlot = 0U; groupSlot < GROUPS_PER_SEGMENT;
             ++groupSlot) {
            const uint32_t nextGroupSlot = groupSlot + 1U;
            const uint32_t generation = groupSlot & 1U;
            const event_t generationEvent = generation == 0U ?
                EVENT_ID0 : EVENT_ID1;
            if (nextGroupSlot < GROUPS_PER_SEGMENT) {
                const uint32_t nextGeneration = nextGroupSlot & 1U;
                const event_t nextEvent = nextGeneration == 0U ?
                    EVENT_ID0 : EVENT_ID1;
                if (nextGroupSlot >= 2U) {
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                }
                LoadInputGroup(
                    startRow,
                    segmentInput + nextGroupSlot * INPUTS_PER_GROUP,
                    nextGeneration, inputStorageBytes, inputBytesPerGroup,
                    copyParams, padParams, scratchLocal, cachedInputPtrs);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            WaitFlag<HardEvent::MTE2_V>(generationEvent);
            TransposeInputGroup(
                groupSlot, generation, repeats, inputStorageBytes,
                inputBytesPerGroup, intermediateBase, firstStageStride,
                segmentLocal, scratchLocal);
            SetFlag<HardEvent::V_MTE2>(generationEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void LoadInputGroup(
        uint64_t startRow, uint32_t inputGroup, uint32_t generation,
        uint32_t inputStorageBytes, uint32_t inputBytesPerGroup,
        const DataCopyExtParams& copyParams,
        const DataCopyPadExtParams<uint8_t>& padParams,
        LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t generationBase = generation * inputBytesPerGroup;
        for (uint32_t inputSlot = 0U; inputSlot < INPUTS_PER_GROUP;
             ++inputSlot) {
            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                cachedInputPtrs.GetValue(inputGroup + inputSlot)));
            DataCopyPad(
                scratchLocal[generationBase + inputSlot * inputStorageBytes],
                inputGm[startRow * ROW_BYTES], copyParams, padParams);
        }
    }

    __aicore__ inline void TransposeInputGroup(
        uint32_t groupSlot, uint32_t generation, uint32_t repeats,
        uint32_t inputStorageBytes, uint32_t inputBytesPerGroup,
        uint32_t intermediateBase, uint32_t firstStageStride,
        LocalTensor<uint8_t> segmentLocal, LocalTensor<uint8_t> scratchLocal)
    {
        constexpr uint32_t ELEMENT_BYTES = sizeof(uint16_t);
        constexpr uint32_t ELEMENTS_PER_INPUT = ROW_BYTES / ELEMENT_BYTES;
        const uint32_t generationBase = generation * inputBytesPerGroup;
        for (uint32_t inputSlot = 0U; inputSlot < INPUTS_PER_GROUP;
             ++inputSlot) {
            uint64_t sourceAddresses[MATRIX_ROWS];
            uint64_t destinationAddresses[MATRIX_ROWS];
            const uint32_t inputBase = generationBase +
                inputSlot * inputStorageBytes;
            const uint32_t stageBase = intermediateBase +
                inputSlot * MATRIX_ROWS * firstStageStride;
            for (uint32_t lane = 0U; lane < MATRIX_ROWS; ++lane) {
                sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[inputBase + lane * ROW_BYTES].GetPhyAddr());
                destinationAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[stageBase + lane * firstStageStride].GetPhyAddr());
            }
            set_va_reg_sb(VA0, destinationAddresses);
            set_va_reg_sb(VA1, destinationAddresses + 8U);
            set_va_reg_sb(VA2, sourceAddresses);
            set_va_reg_sb(VA3, sourceAddresses + 8U);
            scatter_vnchwconv_b16(
                VA0, VA2, static_cast<uint8_t>(repeats), 1U,
                static_cast<uint16_t>(ROW_BYTES / ELEMENT_BYTES));
        }

        uint64_t sourceAddresses[MATRIX_ROWS];
        uint64_t destinationAddresses[MATRIX_ROWS];
        for (uint32_t inputSlot = 0U; inputSlot < INPUTS_PER_GROUP;
             ++inputSlot) {
            const uint32_t stageBase = intermediateBase +
                inputSlot * MATRIX_ROWS * firstStageStride;
            for (uint32_t element = 0U; element < ELEMENTS_PER_INPUT;
                 ++element) {
                const uint32_t lane = inputSlot * ELEMENTS_PER_INPUT + element;
                sourceAddresses[lane] = reinterpret_cast<uint64_t>(
                    scratchLocal[stageBase + element * firstStageStride].GetPhyAddr());
            }
        }
        const uint32_t groupOutputOffset = groupSlot * DATA_BLOCK_BYTES;
        for (uint32_t row = 0U; row < MATRIX_ROWS; ++row) {
            destinationAddresses[row] = reinterpret_cast<uint64_t>(
                segmentLocal[row * SEGMENT_BYTES + groupOutputOffset].GetPhyAddr());
        }
        set_va_reg_sb(VA0, destinationAddresses);
        set_va_reg_sb(VA1, destinationAddresses + 8U);
        set_va_reg_sb(VA2, sourceAddresses);
        set_va_reg_sb(VA3, sourceAddresses + 8U);
        scatter_vnchwconv_b16(
            VA0, VA2, static_cast<uint8_t>(repeats),
            static_cast<uint16_t>(MATRIX_ROWS * SEGMENT_BYTES /
                                  DATA_BLOCK_BYTES),
            1U);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __gm__ uint64_t* inputDataPtrs_ = nullptr;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

template <bool TRIPLE_BUFFER_128 = false>
class ConcatSegmentedAligned32Kernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        const __gm__ uint64_t* descriptorStart =
            reinterpret_cast<__gm__ uint64_t*>(inputs);
        inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
            descriptorStart[0] / sizeof(uint64_t);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t INPUT_POINTER_BYTES =
            MAX_DYNAMIC_INPUTS * sizeof(uint64_t);
        const uint32_t scratchGenerationBytes =
            tiling_->uniformInputRowBytes == 128U ?
            tiling_->rowsPerBatch * tiling_->uniformInputRowBytes : 0U;
        constexpr uint32_t SCRATCH_GENERATIONS =
            TRIPLE_BUFFER_128 ? 3U : 2U;
        const uint32_t segmentBufferBytes =
            tiling_->rowsPerBatch * SEGMENT_BYTES;
        const bool doubleBufferedStridedSegments =
            tiling_->uniformInputRowBytes != 128U &&
            tiling_->tileBytes >= 2U * segmentBufferBytes;
        LocalTensor<uint8_t> segmentLocal0(
            TPosition::VECCALC, 0U, segmentBufferBytes);
        LocalTensor<uint8_t> segmentLocal1(
            TPosition::VECCALC, segmentBufferBytes,
            doubleBufferedStridedSegments ? segmentBufferBytes : 0U);
        LocalTensor<uint8_t> scratchLocal(
            TPosition::VECCALC, tiling_->tileBytes,
            SCRATCH_GENERATIONS * scratchGenerationBytes);
        LocalTensor<uint64_t> cachedInputPtrs(
            TPosition::VECCALC,
            tiling_->tileBytes +
                SCRATCH_GENERATIONS * scratchGenerationBytes,
            INPUT_POINTER_BYTES);

        GlobalTensor<uint64_t> inputPtrGm;
        inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
        DataCopy(cachedInputPtrs, inputPtrGm, tiling_->inputNum);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);

        uint64_t row = tiling_->blockFactorRows *
            static_cast<uint64_t>(GetBlockIdx());
        if (row >= tiling_->outerRows) {
            return;
        }
        const uint64_t endRow = Min(
            row + tiling_->blockFactorRows, tiling_->outerRows);
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row,
                static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatch(row, rows, segmentLocal0, segmentLocal1,
                            doubleBufferedStridedSegments, scratchLocal,
                            scratchGenerationBytes, cachedInputPtrs);
            row += rows;
        }
    }

private:
    static constexpr uint32_t SEGMENT_BYTES = 512U;

    __aicore__ inline void ProcessRowBatch(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint8_t> segmentLocal0,
        LocalTensor<uint8_t> segmentLocal1,
        bool doubleBufferedStridedSegments,
        LocalTensor<uint8_t> scratchLocal,
        uint32_t scratchGenerationBytes,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t rowBytes = tiling_->uniformInputRowBytes;
        if (rowBytes != 128U) {
            ProcessStridedRowBatch(
                startRow, rows, rowBytes, segmentLocal0, segmentLocal1,
                doubleBufferedStridedSegments, cachedInputPtrs);
            return;
        }
        if constexpr (TRIPLE_BUFFER_128) {
            ProcessTripleBuffered128RowBatch(
                startRow, rows, rowBytes, segmentLocal0, scratchLocal,
                scratchGenerationBytes, cachedInputPtrs);
            return;
        }
        const uint32_t inputsPerSegment = SEGMENT_BYTES / rowBytes;
        const uint32_t inputStorageBytes =
            static_cast<uint32_t>(rows) * rowBytes;
        DataCopyParams scatterParams{
            rows, static_cast<uint16_t>(rowBytes / DATA_BLOCK_BYTES), 0U,
            static_cast<uint16_t>(
                (SEGMENT_BYTES - rowBytes) / DATA_BLOCK_BYTES)};
        DataCopyParams outputParams{
            rows, static_cast<uint16_t>(SEGMENT_BYTES / DATA_BLOCK_BYTES),
            0U, static_cast<uint16_t>(
                (tiling_->outputRowBytes - SEGMENT_BYTES) /
                DATA_BLOCK_BYTES)};
        for (uint32_t segmentInput = 0U; segmentInput < tiling_->inputNum;
             segmentInput += inputsPerSegment) {
            LoadInput(startRow, segmentInput, 0U, rowBytes,
                      inputStorageBytes, scratchGenerationBytes,
                      scratchLocal, cachedInputPtrs);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t inputSlot = 0U;
                 inputSlot < inputsPerSegment; ++inputSlot) {
                const uint32_t nextInputSlot = inputSlot + 1U;
                const uint32_t generation = inputSlot & 1U;
                const event_t generationEvent = generation == 0U ?
                    EVENT_ID0 : EVENT_ID1;
                if (nextInputSlot < inputsPerSegment) {
                    const uint32_t nextGeneration = nextInputSlot & 1U;
                    const event_t nextEvent = nextGeneration == 0U ?
                        EVENT_ID0 : EVENT_ID1;
                    if (nextInputSlot >= 2U) {
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    }
                    LoadInput(
                        startRow, segmentInput + nextInputSlot,
                        nextGeneration, rowBytes, inputStorageBytes,
                        scratchGenerationBytes, scratchLocal,
                        cachedInputPtrs);
                    SetFlag<HardEvent::MTE2_V>(nextEvent);
                }
                WaitFlag<HardEvent::MTE2_V>(generationEvent);
                DataCopy(
                    segmentLocal0[inputSlot * rowBytes],
                    scratchLocal[generation * scratchGenerationBytes],
                    scatterParams);
                SetFlag<HardEvent::V_MTE2>(generationEvent);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopy(
                outputGm_[startRow * tiling_->outputRowBytes +
                          static_cast<uint64_t>(segmentInput) * rowBytes],
                segmentLocal0, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void ProcessTripleBuffered128RowBatch(
        uint64_t startRow, uint16_t rows, uint32_t rowBytes,
        LocalTensor<uint8_t> segmentLocal,
        LocalTensor<uint8_t> scratchLocal,
        uint32_t scratchGenerationBytes,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t inputsPerSegment = SEGMENT_BYTES / rowBytes;
        const uint32_t inputStorageBytes =
            static_cast<uint32_t>(rows) * rowBytes;
        DataCopyParams scatterParams{
            rows, static_cast<uint16_t>(rowBytes / DATA_BLOCK_BYTES), 0U,
            static_cast<uint16_t>(
                (SEGMENT_BYTES - rowBytes) / DATA_BLOCK_BYTES)};
        DataCopyParams outputParams{
            rows, static_cast<uint16_t>(SEGMENT_BYTES / DATA_BLOCK_BYTES),
            0U, static_cast<uint16_t>(
                (tiling_->outputRowBytes - SEGMENT_BYTES) /
                DATA_BLOCK_BYTES)};
        for (uint32_t segmentInput = 0U; segmentInput < tiling_->inputNum;
             segmentInput += inputsPerSegment) {
            LoadInput(startRow, segmentInput, 0U, rowBytes,
                      inputStorageBytes, scratchGenerationBytes,
                      scratchLocal, cachedInputPtrs);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t inputSlot = 0U;
                 inputSlot < inputsPerSegment; ++inputSlot) {
                const uint32_t nextInputSlot = inputSlot + 1U;
                const uint32_t generation = inputSlot % 3U;
                const event_t generationEvent = generation == 0U ?
                    EVENT_ID0 : (generation == 1U ? EVENT_ID1 : EVENT_ID2);
                if (nextInputSlot < inputsPerSegment) {
                    const uint32_t nextGeneration = nextInputSlot % 3U;
                    const event_t nextEvent = nextGeneration == 0U ?
                        EVENT_ID0 : (nextGeneration == 1U ?
                        EVENT_ID1 : EVENT_ID2);
                    if (nextInputSlot >= 3U) {
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    }
                    LoadInput(
                        startRow, segmentInput + nextInputSlot,
                        nextGeneration, rowBytes, inputStorageBytes,
                        scratchGenerationBytes, scratchLocal,
                        cachedInputPtrs);
                    SetFlag<HardEvent::MTE2_V>(nextEvent);
                }
                WaitFlag<HardEvent::MTE2_V>(generationEvent);
                DataCopy(
                    segmentLocal[inputSlot * rowBytes],
                    scratchLocal[generation * scratchGenerationBytes],
                    scatterParams);
                SetFlag<HardEvent::V_MTE2>(generationEvent);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            DataCopy(
                outputGm_[startRow * tiling_->outputRowBytes +
                          static_cast<uint64_t>(segmentInput) * rowBytes],
                segmentLocal, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

    __aicore__ inline void ProcessStridedRowBatch(
        uint64_t startRow, uint16_t rows, uint32_t rowBytes,
        LocalTensor<uint8_t> segmentLocal0,
        LocalTensor<uint8_t> segmentLocal1,
        bool doubleBufferedStridedSegments,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        const uint32_t inputsPerSegment = SEGMENT_BYTES / rowBytes;
        DataCopyParams inputParams{
            rows, static_cast<uint16_t>(rowBytes / DATA_BLOCK_BYTES), 0U,
            static_cast<uint16_t>(
                (SEGMENT_BYTES - rowBytes) / DATA_BLOCK_BYTES)};
        DataCopyParams outputParams{
            rows, static_cast<uint16_t>(SEGMENT_BYTES / DATA_BLOCK_BYTES),
            0U, static_cast<uint16_t>(
                (tiling_->outputRowBytes - SEGMENT_BYTES) /
                DATA_BLOCK_BYTES)};
        bool pending0 = false;
        bool pending1 = false;
        uint32_t segmentIndex = 0U;
        for (uint32_t segmentInput = 0U; segmentInput < tiling_->inputNum;
             segmentInput += inputsPerSegment) {
            const bool useSlot1 = doubleBufferedStridedSegments &&
                (segmentIndex & 1U) != 0U;
            const event_t segmentEvent = useSlot1 ? EVENT_ID1 : EVENT_ID0;
            if ((useSlot1 && pending1) || (!useSlot1 && pending0)) {
                WaitFlag<HardEvent::MTE3_MTE2>(segmentEvent);
                if (useSlot1) {
                    pending1 = false;
                } else {
                    pending0 = false;
                }
            }
            LocalTensor<uint8_t> segmentLocal = useSlot1 ?
                segmentLocal1 : segmentLocal0;
            for (uint32_t inputSlot = 0U;
                 inputSlot < inputsPerSegment; ++inputSlot) {
                GlobalTensor<uint8_t> inputGm;
                inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    cachedInputPtrs.GetValue(segmentInput + inputSlot)));
                DataCopy(segmentLocal[inputSlot * rowBytes],
                         inputGm[startRow * rowBytes], inputParams);
            }
            SetFlag<HardEvent::MTE2_MTE3>(segmentEvent);
            WaitFlag<HardEvent::MTE2_MTE3>(segmentEvent);
            DataCopy(
                outputGm_[startRow * tiling_->outputRowBytes +
                          static_cast<uint64_t>(segmentInput) * rowBytes],
                segmentLocal, outputParams);
            SetFlag<HardEvent::MTE3_MTE2>(segmentEvent);
            if (useSlot1) {
                pending1 = true;
            } else {
                pending0 = true;
            }
            ++segmentIndex;
        }
        if (pending0) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (pending1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void LoadInput(
        uint64_t startRow, uint32_t inputIdx, uint32_t generation,
        uint32_t rowBytes, uint32_t inputStorageBytes,
        uint32_t scratchGenerationBytes, LocalTensor<uint8_t> scratchLocal,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
            cachedInputPtrs.GetValue(inputIdx)));
        DataCopy(scratchLocal[generation * scratchGenerationBytes],
                 inputGm[startRow * rowBytes], inputStorageBytes);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __gm__ uint64_t* inputDataPtrs_ = nullptr;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
};

template <bool COMPACT_GATHER_ROWS = false,
          bool SEGMENTED_GATHER_ROWS = false,
          bool ADAPTIVE_SCRATCH_ROWS = false,
          uint32_t GATHER_ROWS = DYNAMIC_SLAB_GATHER_SEED_ROWS,
          uint32_t OFFSET_GENERATIONS = 1U,
          bool VECTOR_GATHER_OFFSETS = false,
          bool DIRECT_GM_POINTERS = false,
          bool PRECOMPUTED_GATHER_BASE = false,
          bool WIDE_GATHER_SEGMENTS = false,
          bool SCRATCH_OFFSET_GATHER32 = false,
          uint32_t COMPACT_GATHER_MIN_BYTES = COMPACT_GATHER_MIN_ROW_BYTES,
          bool GATHER_MISALIGNED_BLOCK_ROWS = false,
          bool BYPASS_L2_READS = false>
class ConcatDynamicSlabKernel {
public:
    static_assert(
        GATHER_ROWS == DYNAMIC_SLAB_GATHER_SEED_ROWS || GATHER_ROWS == 16U,
        "dynamic slab Gather supports eight or sixteen offset records");
    static_assert(
        OFFSET_GENERATIONS == 1U || OFFSET_GENERATIONS == 2U,
        "dynamic slab Gather supports one or two offset generations");
    static_assert(
        !VECTOR_GATHER_OFFSETS || OFFSET_GENERATIONS == 1U,
        "Vector-only Gather offsets use one pipeline-ordered table");
    static_assert(
        !PRECOMPUTED_GATHER_BASE ||
            (VECTOR_GATHER_OFFSETS && DIRECT_GM_POINTERS),
        "precomputed Gather base requires the Scalar-UB-free route");
    static_assert(
        !WIDE_GATHER_SEGMENTS ||
            (SEGMENTED_GATHER_ROWS && GATHER_ROWS == 16U),
        "wide Gather segments require the repeat-16 segmented route");
    static_assert(
        !SCRATCH_OFFSET_GATHER32 ||
            (WIDE_GATHER_SEGMENTS && VECTOR_GATHER_OFFSETS &&
             DIRECT_GM_POINTERS && PRECOMPUTED_GATHER_BASE),
        "scratch-offset repeat-32 requires the scalar-free wide route");
    static_assert(
        COMPACT_GATHER_MIN_BYTES != 0U &&
            COMPACT_GATHER_MIN_BYTES % sizeof(uint32_t) == 0U,
        "compact Gather threshold must contain complete FP32 elements");

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        const __gm__ uint64_t* descriptorStart =
            reinterpret_cast<__gm__ uint64_t*>(inputs);
        inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
            descriptorStart[0] / sizeof(uint64_t);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
#if defined(L2_CACHE_HINT)
        if constexpr (BYPASS_L2_READS) {
            l2BypassOffset_ = g_opSystemRunCfg.l2Cacheoffset;
        }
#endif

        const uint32_t header =
            tiling_->launchPadding[DYNAMIC_SLAB_HEADER_WORD];
        groupCount_ = header & 0xffU;
        slabStride_ = ((header >> 8U) & 0xffU) * DATA_BLOCK_BYTES;
        scratchStride_ = ((header >> 16U) & 0xffU) * DATA_BLOCK_BYTES;
        metadataVersion_ = header >> 24U;
        scratchStride1_ = scratchStride_;
        scratchStride2_ = scratchStride_;
        scratchStride3_ = scratchStride_;
        if constexpr (ADAPTIVE_SCRATCH_ROWS) {
            const uint32_t layoutIndex = DYNAMIC_SLAB_HEADER_WORD + 1U +
                (groupCount_ + 1U) / 2U;
            const uint32_t layout = tiling_->launchPadding[layoutIndex];
            scratchStride1_ = (layout & 0xffU) * DATA_BLOCK_BYTES;
            scratchStride2_ = ((layout >> 8U) & 0xffU) * DATA_BLOCK_BYTES;
            scratchStride3_ = ((layout >> 16U) & 0xffU) * DATA_BLOCK_BYTES;
            adaptiveMetadataVersion_ = layout >> 24U;
        }
        const uint32_t slabBytes = tiling_->rowsPerBatch * slabStride_;
        const uint32_t scratchBytes0 =
            tiling_->rowsPerBatch * scratchStride_;
        const uint32_t scratchBytes1 =
            tiling_->rowsPerBatch * scratchStride1_;
        pointerBytes_ = AlignUp(
            tiling_->inputNum * sizeof(uint64_t), DATA_BLOCK_BYTES);
        pipe_.InitBuffer(slabBuffer_, slabBytes);
        pipe_.InitBuffer(scratchBuffer0_, scratchBytes0);
        pipe_.InitBuffer(scratchBuffer1_, scratchBytes1);
        if constexpr (COMPACT_GATHER_ROWS || SEGMENTED_GATHER_ROWS) {
            pipe_.InitBuffer(gatherOffsetBuffer_,
                             OFFSET_GENERATIONS * GATHER_ROWS *
                                 DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT *
                                 sizeof(uint32_t));
        }
        if constexpr (!DIRECT_GM_POINTERS) {
            pipe_.InitBuffer(pointerBuffer_, pointerBytes_);
        }
        if constexpr (PRECOMPUTED_GATHER_BASE) {
            pipe_.InitBuffer(
                gatherBaseBuffer_,
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (metadataVersion_ != DYNAMIC_SLAB_METADATA_VERSION ||
            groupCount_ == 0U || groupCount_ > 32U ||
            slabStride_ == 0U || scratchStride_ == 0U ||
            tiling_->rowsPerBatch == 0U) {
            return;
        }
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint32_t scheduleWord = tiling_->launchPadding[
            PACKED_WIDTH_WORDS + static_cast<uint32_t>(blockIndex) / 4U];
        const uint32_t scheduleByte =
            (scheduleWord >> ((blockIndex % 4U) * 8U)) & 0xffU;
        const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
        uint64_t row = blockIndex * rowsPerBlock + (scheduleByte & 0x3fU);
        const uint64_t endRow = row + rowsPerBlock +
            ((scheduleByte >> 6U) & 1U);
        if (row >= tiling_->outerRows) {
            return;
        }

        LocalTensor<uint64_t> cachedInputPtrs;
        if constexpr (!DIRECT_GM_POINTERS) {
            cachedInputPtrs = pointerBuffer_.Get<uint64_t>();
            GlobalTensor<uint64_t> inputPtrGm;
            inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
            DataCopy(cachedInputPtrs, inputPtrGm,
                     pointerBytes_ / sizeof(uint64_t));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        }
        InitializeGatherOffsetBase();

        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatch(row, rows, cachedInputPtrs);
            row += rows;
        }
    }

protected:
    __aicore__ inline void InitializeGatherOffsetBase()
    {
        if constexpr (PRECOMPUTED_GATHER_BASE) {
            // CANN 8.5 implements ArithProgression with Scalar UB seed
            // stores. Execute that implementation once before any data-path
            // Vector work, then keep the complete [0, 4, ..., 252] record
            // read-only for every input in this block.
            LocalTensor<int32_t> gatherBase =
                gatherBaseBuffer_.Get<int32_t>();
            ArithProgression<int32_t>(
                gatherBase, 0, static_cast<int32_t>(sizeof(int32_t)),
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t GroupEnd(uint32_t group) const
    {
        const uint32_t packed = tiling_->launchPadding[
            DYNAMIC_SLAB_HEADER_WORD + 1U + group / 2U];
        return (packed >> ((group & 1U) * 16U)) & 0xffffU;
    }

    __aicore__ inline uint32_t RowBytes(uint32_t inputIdx,
                                        uint32_t& packedWidths) const
    {
        if ((inputIdx & 3U) == 0U) {
            packedWidths = tiling_->launchPadding[inputIdx / 4U];
        }
        const uint32_t elements =
            (packedWidths >> ((inputIdx & 3U) * 8U)) & 0xffU;
        return elements * tiling_->elementBytes;
    }

    __aicore__ inline uint64_t LowBits(uint32_t count) const
    {
        return count >= 64U ? ~static_cast<uint64_t>(0U) :
            (count == 0U ? 0U :
             (static_cast<uint64_t>(1U) << count) - 1U);
    }

    __aicore__ inline void MakeMask(uint64_t mask[2], uint32_t start,
                                    uint32_t count) const
    {
        mask[0] = 0U;
        mask[1] = 0U;
        const uint32_t lowStart = start < 64U ? start : 64U;
        const uint32_t lowEnd = start + count < 64U ?
            start + count : 64U;
        if (lowEnd > lowStart) {
            mask[0] = LowBits(lowEnd - lowStart) << lowStart;
        }
        const uint32_t highStart = start > 64U ? start - 64U : 0U;
        const uint32_t absoluteEnd = start + count < 128U ?
            start + count : 128U;
        const uint32_t highEnd = absoluteEnd > 64U ?
            absoluteEnd - 64U : 0U;
        if (highEnd > highStart) {
            mask[1] = LowBits(highEnd - highStart) << highStart;
        }
    }

    __aicore__ inline bool UseCompactGather(
        uint32_t rowBytes, uint32_t localPrefix = 0U) const
    {
        return COMPACT_GATHER_ROWS &&
            tiling_->elementBytes == sizeof(uint32_t) &&
            rowBytes >= COMPACT_GATHER_MIN_BYTES &&
            rowBytes < DYNAMIC_SLAB_COMPACT_GATHER_ROW_BYTE_LIMIT &&
            (rowBytes % DATA_BLOCK_BYTES != 0U ||
             (GATHER_MISALIGNED_BLOCK_ROWS &&
              localPrefix % DATA_BLOCK_BYTES != 0U));
    }

    __aicore__ inline bool UseSegmentedGather(
        uint32_t rowBytes, uint32_t localPrefix = 0U) const
    {
        return SEGMENTED_GATHER_ROWS &&
            tiling_->elementBytes == sizeof(uint32_t) &&
            rowBytes >= DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES &&
            (rowBytes % DATA_BLOCK_BYTES != 0U ||
             (GATHER_MISALIGNED_BLOCK_ROWS &&
              localPrefix % DATA_BLOCK_BYTES != 0U));
    }

    __aicore__ inline uint32_t LoadScratch(
        uint32_t inputIdx, uint64_t startRow, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes,
        const LocalTensor<uint8_t>& scratch, event_t event,
        const LocalTensor<uint64_t>& cachedInputPtrs)
    {
        uint64_t inputAddress = 0U;
        if constexpr (DIRECT_GM_POINTERS) {
            // A cached pointer is a Scalar UB load.  After the first Vector
            // pack CANN 8.5 conservatively drains PIPE_V before every such
            // load, even though the pointer and data buffers do not overlap.
            // Read the folded descriptor's pointer table from GM instead so
            // Scalar no longer aliases any Vector-produced UB region.
            inputAddress = inputDataPtrs_[inputIdx];
        } else {
            inputAddress = cachedInputPtrs.GetValue(inputIdx);
        }
        const uint64_t sourceOffset = startRow * rowBytes;
        if constexpr (BYPASS_L2_READS) {
            inputAddress += l2BypassOffset_;
        }
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(inputAddress) + sourceOffset);
        uint32_t rowStride = rowBytes;
        if (UseCompactGather(rowBytes, localPrefix) ||
            UseSegmentedGather(rowBytes, localPrefix)) {
            const DataCopyExtParams copyParams{
                1U, static_cast<uint32_t>(rows) * rowBytes, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> noPadding{
                false, 0U, 0U, 0U};
            DataCopyPad(scratch, inputGm,
                        copyParams, noPadding);
        } else {
            const uint32_t leftPadding =
                localPrefix % DATA_BLOCK_BYTES;
            const bool alignedCopy = leftPadding == 0U &&
                rowBytes % DATA_BLOCK_BYTES == 0U &&
                (inputAddress + sourceOffset) % DATA_BLOCK_BYTES == 0U;
            if (alignedCopy) {
                DataCopy(scratch, inputGm,
                         static_cast<uint32_t>(rows) * rowBytes);
            } else {
                rowStride = AlignUp(
                    leftPadding + rowBytes, DATA_BLOCK_BYTES);
                const DataCopyExtParams copyParams{
                    rows, rowBytes, 0U, 0U, 0U};
                const DataCopyPadExtParams<uint8_t> padParams{
                    leftPadding != 0U, static_cast<uint8_t>(leftPadding),
                    0U, 0U};
                DataCopyPad(scratch, inputGm,
                            copyParams, padParams);
            }
        }
        SetFlag<HardEvent::MTE2_V>(event);
        return rowStride;
    }

    __aicore__ inline void VectorPack(
        const LocalTensor<uint8_t>& slabBytes,
        const LocalTensor<uint8_t>& scratchBytes,
        uint16_t rows, uint32_t localPrefix, uint32_t rowBytes,
        uint32_t sourceStride)
    {
        constexpr uint32_t VECTOR_ELEMENTS =
            VECTOR_BYTES / sizeof(uint32_t);
        const uint32_t leftElements =
            localPrefix % DATA_BLOCK_BYTES / sizeof(uint32_t);
        const uint32_t destinationBase =
            localPrefix - localPrefix % DATA_BLOCK_BYTES;
        LocalTensor<int32_t> source =
            scratchBytes.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> destination =
            slabBytes[destinationBase].ReinterpretCast<int32_t>();
        const uint8_t destinationRepeatStride = static_cast<uint8_t>(
            slabStride_ / DATA_BLOCK_BYTES);
        const uint8_t sourceRepeatStride = static_cast<uint8_t>(
            sourceStride / DATA_BLOCK_BYTES);
        // A complete, prefix-aligned cell has no packing work to do.  Use
        // the C220 UB-to-UB 2-D copy engine to move all rows directly into
        // the slab.  This preserves the same row ownership and source
        // lifetime as VectorPack, but avoids issuing masked Adds for every
        // 128-byte vector chunk.  The predicate is entirely runtime-derived;
        // unaligned rows retain the established masked-vector path because
        // their neighbouring cell bytes belong to adjacent inputs.
        const uint64_t copyBytes = static_cast<uint64_t>(rows) * rowBytes;
        if (leftElements == 0U && rowBytes <= 1024U &&
            copyBytes >= 8U * 1024U &&
            rowBytes % DATA_BLOCK_BYTES == 0U &&
            sourceStride >= rowBytes &&
            sourceStride % DATA_BLOCK_BYTES == 0U &&
            slabStride_ >= rowBytes &&
            (slabStride_ - rowBytes) % DATA_BLOCK_BYTES == 0U &&
            rows != 0U) {
            const DataCopyParams copyParams{
                rows,
                static_cast<uint16_t>(rowBytes / DATA_BLOCK_BYTES),
                static_cast<uint16_t>((sourceStride - rowBytes) /
                                       DATA_BLOCK_BYTES),
                static_cast<uint16_t>((slabStride_ - rowBytes) /
                                       DATA_BLOCK_BYTES)};
            DataCopy(slabBytes[destinationBase], scratchBytes, copyParams);
            return;
        }
        uint32_t copied = 0U;
        const uint32_t total = rowBytes / sizeof(uint32_t);
        while (copied < total) {
            const uint32_t physical = leftElements + copied;
            const uint32_t vectorBase =
                physical / VECTOR_ELEMENTS * VECTOR_ELEMENTS;
            const uint32_t lane = physical - vectorBase;
            const uint32_t remaining = total - copied;
            const uint32_t count = remaining < VECTOR_ELEMENTS - lane ?
                remaining : VECTOR_ELEMENTS - lane;
            uint64_t mask[2];
            MakeMask(mask, lane, count);
            UnaryRepeatParams params{
                1U, 1U, destinationRepeatStride, sourceRepeatStride};
            Adds<int32_t>(destination[vectorBase], source[vectorBase], 0,
                          mask, static_cast<uint8_t>(rows), params);
            copied += count;
        }
    }

    __aicore__ inline void GenerateGatherOffsetsWavefront(
        LocalTensor<int32_t> offsetValues, uint32_t rowBytes,
        uint32_t leftBytes, uint32_t count, uint32_t offsetSlot = 0U,
        uint32_t* offsetPendingMask = nullptr)
    {
        constexpr uint32_t BLOCK_ELEMENTS =
            DATA_BLOCK_BYTES / sizeof(int32_t);
        constexpr uint8_t RECORD_REPEAT_STRIDE = static_cast<uint8_t>(
            DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT / BLOCK_ELEMENTS);
        static_assert(
            DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT % BLOCK_ELEMENTS == 0U,
            "dynamic slab Gather record must contain complete data blocks");

        if constexpr (VECTOR_GATHER_OFFSETS) {
            // Derive the complete first record on Vector, then derive the
            // other seven records independently from it. Key 58/59 retain
            // the dynamic ArithProgression control; key 60 reads a constant
            // base record so its input loop performs no Scalar UB access.
            const UnaryRepeatParams completeRecord{
                1U, 1U,
                static_cast<uint8_t>(
                    DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT /
                    BLOCK_ELEMENTS),
                static_cast<uint8_t>(
                    DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT /
                    BLOCK_ELEMENTS)};
            if constexpr (PRECOMPUTED_GATHER_BASE) {
                LocalTensor<int32_t> gatherBase =
                    gatherBaseBuffer_.Get<int32_t>();
                SetVectorMask<int32_t>(0U, LowBits(count));
                Adds<int32_t, false>(
                    offsetValues, gatherBase,
                    -static_cast<int32_t>(leftBytes), MASK_PLACEHOLDER,
                    1U, completeRecord);
            } else {
                ArithProgression<int32_t>(
                    offsetValues, -static_cast<int32_t>(leftBytes),
                    static_cast<int32_t>(sizeof(int32_t)), count);
            }
            PipeBarrier<PIPE_V>();

            SetVectorMask<int32_t>(0U, LowBits(count));
            for (uint32_t row = 1U;
                 row < DYNAMIC_SLAB_GATHER_SEED_ROWS; ++row) {
                Adds<int32_t, false>(
                    offsetValues[
                        row * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
                    offsetValues, static_cast<int32_t>(row * rowBytes),
                    MASK_PLACEHOLDER, 1U, completeRecord);
            }
            PipeBarrier<PIPE_V>();

            if constexpr (GATHER_ROWS == 16U) {
                Adds<int32_t, false>(
                    offsetValues[DYNAMIC_SLAB_GATHER_SEED_ROWS *
                        DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
                    offsetValues,
                    static_cast<int32_t>(
                        DYNAMIC_SLAB_GATHER_SEED_ROWS * rowBytes),
                    MASK_PLACEHOLDER,
                    static_cast<uint8_t>(
                        DYNAMIC_SLAB_GATHER_SEED_ROWS),
                    completeRecord);
                PipeBarrier<PIPE_V>();
            }
        } else {
            event_t tableEvent = EVENT_ID0;
            if constexpr (OFFSET_GENERATIONS == 1U) {
                // Preserve the established single-table dependency exactly
                // for keys 47--49 and 56.
                tableEvent = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(tableEvent);
                WaitFlag<HardEvent::V_S>(tableEvent);
            } else {
                // A table-local release is published after its Gather reader.
                // Scalar can therefore seed the alternate table while Vector
                // is still consuming this one, and waits only on actual reuse.
                tableEvent = offsetSlot == 0U ? EVENT_ID0 : EVENT_ID1;
                const uint32_t bit = 1U << offsetSlot;
                if ((*offsetPendingMask & bit) != 0U) {
                    WaitFlag<HardEvent::V_S>(tableEvent);
                    *offsetPendingMask &= ~bit;
                }
            }

            for (uint32_t row = 0U;
                 row < DYNAMIC_SLAB_GATHER_SEED_ROWS; ++row) {
                LocalTensor<int32_t> record = offsetValues[
                    row * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT];
                const int32_t first =
                    static_cast<int32_t>(row * rowBytes) -
                    static_cast<int32_t>(leftBytes);
                for (uint32_t lane = 0U; lane < BLOCK_ELEMENTS; ++lane) {
                    record.SetValue(
                        lane,
                        first +
                            static_cast<int32_t>(lane * sizeof(int32_t)));
                }
            }

            event_t scalarToVector = tableEvent;
            if constexpr (OFFSET_GENERATIONS == 1U) {
                scalarToVector = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::S_V));
            }
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            SetVectorMask<int32_t>(
                0U, (static_cast<uint64_t>(1U) << BLOCK_ELEMENTS) - 1U);
            PipeBarrier<PIPE_V>();

            // One repeat advances by one 256-byte offset record. Each VADDS
            // stage therefore updates the same 32-byte block in all eight
            // records, followed by one barrier for the stage-to-stage RAW.
            const UnaryRepeatParams recordWavefront{
                1U, 1U, RECORD_REPEAT_STRIDE, RECORD_REPEAT_STRIDE};
            const uint32_t blockCount =
                (count + BLOCK_ELEMENTS - 1U) / BLOCK_ELEMENTS;
            for (uint32_t block = 1U; block < blockCount; ++block) {
                Adds<int32_t, false>(
                    offsetValues[block * BLOCK_ELEMENTS],
                    offsetValues[(block - 1U) * BLOCK_ELEMENTS],
                    static_cast<int32_t>(DATA_BLOCK_BYTES),
                    MASK_PLACEHOLDER,
                    static_cast<uint8_t>(
                        DYNAMIC_SLAB_GATHER_SEED_ROWS),
                    recordWavefront);
                PipeBarrier<PIPE_V>();
            }

            if constexpr (GATHER_ROWS == 16U) {
                // Copy the eight complete wavefront records in one Vector
                // instruction. Every destination offset differs from its
                // source by exactly eight input rows, replacing 64 Scalar
                // seed stores while doubling each Gather repeat.
                SetVectorMask<int32_t>(0U, ~static_cast<uint64_t>(0U));
                Adds<int32_t, false>(
                    offsetValues[DYNAMIC_SLAB_GATHER_SEED_ROWS *
                        DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
                    offsetValues,
                    static_cast<int32_t>(
                        DYNAMIC_SLAB_GATHER_SEED_ROWS * rowBytes),
                    MASK_PLACEHOLDER,
                    static_cast<uint8_t>(
                        DYNAMIC_SLAB_GATHER_SEED_ROWS),
                    recordWavefront);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline bool CanUseScratchOffsetGather32(
        uint16_t rows, uint32_t rowBytes,
        uint32_t scratchCapacity) const
    {
        if constexpr (!SCRATCH_OFFSET_GATHER32) {
            return false;
        }
        if (rows <= 16U ||
            rowBytes < DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES ||
            scratchCapacity < rowBytes) {
            return false;
        }
        const uint32_t tableOffset = AlignUp(
            static_cast<uint32_t>(rows) * rowBytes, DATA_BLOCK_BYTES);
        return tableOffset + DYNAMIC_SLAB_SCRATCH_OFFSET32_BYTES <=
            static_cast<uint32_t>(rows) * scratchCapacity;
    }

    __aicore__ inline uint32_t ScratchOffsetGather32TableOffset(
        uint16_t rows, uint32_t rowBytes) const
    {
        return AlignUp(
            static_cast<uint32_t>(rows) * rowBytes, DATA_BLOCK_BYTES);
    }

    __aicore__ inline void GenerateScratchGatherOffsets32(
        LocalTensor<int32_t> offsetValues, uint32_t rowBytes,
        uint32_t leftBytes, uint32_t count)
    {
        constexpr uint32_t BLOCK_ELEMENTS =
            DATA_BLOCK_BYTES / sizeof(int32_t);
        const UnaryRepeatParams completeRecord{
            1U, 1U,
            static_cast<uint8_t>(
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT / BLOCK_ELEMENTS),
            static_cast<uint8_t>(
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT / BLOCK_ELEMENTS)};
        LocalTensor<int32_t> gatherBase =
            gatherBaseBuffer_.Get<int32_t>();
        SetVectorMask<int32_t>(0U, LowBits(count));
        Adds<int32_t, false>(
            offsetValues, gatherBase, -static_cast<int32_t>(leftBytes),
            MASK_PLACEHOLDER, 1U, completeRecord);
        PipeBarrier<PIPE_V>();

        for (uint32_t row = 1U;
             row < DYNAMIC_SLAB_GATHER_SEED_ROWS; ++row) {
            Adds<int32_t, false>(
                offsetValues[
                    row * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
                offsetValues, static_cast<int32_t>(row * rowBytes),
                MASK_PLACEHOLDER, 1U, completeRecord);
        }
        PipeBarrier<PIPE_V>();

        Adds<int32_t, false>(
            offsetValues[DYNAMIC_SLAB_GATHER_SEED_ROWS *
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
            offsetValues,
            static_cast<int32_t>(
                DYNAMIC_SLAB_GATHER_SEED_ROWS * rowBytes),
            MASK_PLACEHOLDER,
            static_cast<uint8_t>(DYNAMIC_SLAB_GATHER_SEED_ROWS),
            completeRecord);
        PipeBarrier<PIPE_V>();

        Adds<int32_t, false>(
            offsetValues[16U * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
            offsetValues, static_cast<int32_t>(16U * rowBytes),
            MASK_PLACEHOLDER, 16U, completeRecord);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool PrepareScratchGatherOffsets32(
        LocalTensor<uint8_t> scratchBytes, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes,
        uint32_t scratchCapacity)
    {
        if (!CanUseScratchOffsetGather32(
                rows, rowBytes, scratchCapacity)) {
            return false;
        }
        const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;
        uint32_t segmentSpan = DYNAMIC_SLAB_SEGMENT_BYTES;
        if constexpr (WIDE_GATHER_SEGMENTS) {
            const uint32_t widenedSpan =
                leftBytes == 0U ? 256U : 224U;
            const uint32_t baselineSegments =
                (rowBytes + DYNAMIC_SLAB_SEGMENT_BYTES - 1U) /
                    DYNAMIC_SLAB_SEGMENT_BYTES;
            const uint32_t widenedSegments =
                (rowBytes + widenedSpan - 1U) / widenedSpan;
            // Prefer the widest legal segment even when the descriptor count
            // is unchanged.  This fills more lanes in the leading Gather and
            // leaves a shorter tail while preserving the same source bytes,
            // destination ownership, repeat count, and UB footprint.
            if (widenedSegments <= baselineSegments) {
                segmentSpan = widenedSpan;
            }
        }
        const uint32_t tableOffset =
            ScratchOffsetGather32TableOffset(rows, rowBytes);
        LocalTensor<int32_t> scratchOffsetValues =
            scratchBytes[tableOffset].ReinterpretCast<int32_t>();
        GenerateScratchGatherOffsets32(
            scratchOffsetValues, rowBytes, leftBytes,
            leftBytes / sizeof(uint32_t) +
                segmentSpan / sizeof(uint32_t));
        return true;
    }

    __aicore__ inline void PublishGatherOffsetRelease(
        uint32_t offsetSlot, uint32_t* offsetPendingMask)
    {
        if constexpr (OFFSET_GENERATIONS == 2U) {
            const event_t tableEvent =
                offsetSlot == 0U ? EVENT_ID0 : EVENT_ID1;
            SetFlag<HardEvent::V_S>(tableEvent);
            *offsetPendingMask |= 1U << offsetSlot;
        }
    }

    __aicore__ inline void DrainGatherOffsetReleases(
        uint32_t& offsetPendingMask)
    {
        if constexpr (OFFSET_GENERATIONS == 2U) {
            for (uint32_t slot = 0U; slot < OFFSET_GENERATIONS; ++slot) {
                const uint32_t bit = 1U << slot;
                if ((offsetPendingMask & bit) != 0U) {
                    const event_t tableEvent =
                        slot == 0U ? EVENT_ID0 : EVENT_ID1;
                    WaitFlag<HardEvent::V_S>(tableEvent);
                    offsetPendingMask &= ~bit;
                }
            }
        }
    }

    __aicore__ inline void CompactGatherToSlab(
        const LocalTensor<uint8_t>& slabBytes,
        const LocalTensor<uint8_t>& scratchBytes,
        const LocalTensor<uint8_t>& offsetBytes, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes, uint32_t offsetSlot = 0U,
        uint32_t* offsetPendingMask = nullptr,
        uint32_t scratchCapacity = 0U, bool offsetsReady = false)
    {
        const uint32_t rowElements = rowBytes / sizeof(uint32_t);
        const uint32_t leftElements =
            localPrefix % DATA_BLOCK_BYTES / sizeof(uint32_t);
        const uint32_t destinationBase =
            localPrefix - localPrefix % DATA_BLOCK_BYTES;
        LocalTensor<uint32_t> destination =
            slabBytes[destinationBase].ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> source =
            scratchBytes.ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> offsetValues =
            offsetBytes.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets =
            offsetBytes.ReinterpretCast<uint32_t>();
        const bool useScratchOffset32 = !offsetsReady &&
            CanUseScratchOffsetGather32(rows, rowBytes, scratchCapacity);

        if (!useScratchOffset32 && !offsetsReady) {
            if constexpr (SEGMENTED_GATHER_ROWS) {
                GenerateGatherOffsetsWavefront(
                    offsetValues, rowBytes, leftElements * sizeof(uint32_t),
                    leftElements + rowElements, offsetSlot,
                    offsetPendingMask);
            } else {
                // Preserve key46's established compact-only generator. The
                // wavefront candidate is isolated to segmented keys 47 and 48.
                for (uint32_t row = 0U; row < GATHER_ROWS; ++row) {
                    ArithProgression<int32_t>(
                        offsetValues[
                            row * DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT],
                        static_cast<int32_t>(row * rowBytes) -
                            static_cast<int32_t>(
                                leftElements * sizeof(uint32_t)),
                        static_cast<int32_t>(sizeof(uint32_t)),
                        leftElements + rowElements);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        uint64_t mask[2];
        MakeMask(mask, leftElements, rowElements);
        const uint16_t destinationRepeatStride =
            static_cast<uint16_t>(slabStride_ / DATA_BLOCK_BYTES);
        if (useScratchOffset32) {
            const uint32_t tableOffset =
                ScratchOffsetGather32TableOffset(rows, rowBytes);
            LocalTensor<int32_t> scratchOffsetValues =
                scratchBytes[tableOffset].ReinterpretCast<int32_t>();
            LocalTensor<uint32_t> scratchOffsets =
                scratchBytes[tableOffset].ReinterpretCast<uint32_t>();
            GenerateScratchGatherOffsets32(
                scratchOffsetValues, rowBytes,
                leftElements * sizeof(uint32_t),
                leftElements + rowElements);
            uint32_t scratchRow = 0U;
            for (; scratchRow + DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS <= rows;
                 scratchRow += DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS) {
                Gather<uint32_t>(
                    destination[scratchRow * slabStride_ / sizeof(uint32_t)],
                    source, scratchOffsets, scratchRow * rowBytes, mask,
                    static_cast<uint8_t>(
                        DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS),
                    destinationRepeatStride);
            }
            const uint32_t scratchTailRows =
                static_cast<uint32_t>(rows) - scratchRow;
            if (scratchTailRows != 0U) {
                Gather<uint32_t>(
                    destination[scratchRow * slabStride_ / sizeof(uint32_t)],
                    source, scratchOffsets, scratchRow * rowBytes, mask,
                    static_cast<uint8_t>(scratchTailRows),
                    destinationRepeatStride);
            }
            return;
        }
        uint32_t row = 0U;
        for (; row + GATHER_ROWS <= rows; row += GATHER_ROWS) {
            Gather<uint32_t>(
                destination[row * slabStride_ / sizeof(uint32_t)],
                source, offsets, row * rowBytes, mask,
                static_cast<uint8_t>(GATHER_ROWS),
                destinationRepeatStride);
        }
        const uint32_t tailRows = static_cast<uint32_t>(rows) - row;
        if (tailRows != 0U) {
            Gather<uint32_t>(
                destination[row * slabStride_ / sizeof(uint32_t)],
                source, offsets, row * rowBytes, mask,
                static_cast<uint8_t>(tailRows),
                destinationRepeatStride);
        }
        if (!offsetsReady) {
            PublishGatherOffsetRelease(offsetSlot, offsetPendingMask);
        }
    }

    __aicore__ inline void SegmentedGatherToSlab(
        const LocalTensor<uint8_t>& slabBytes,
        const LocalTensor<uint8_t>& scratchBytes,
        const LocalTensor<uint8_t>& offsetBytes, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes, uint32_t offsetSlot = 0U,
        uint32_t* offsetPendingMask = nullptr,
        uint32_t scratchCapacity = 0U, bool offsetsReady = false)
    {
        LocalTensor<uint32_t> source =
            scratchBytes.ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> offsetValues =
            offsetBytes.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets =
            offsetBytes.ReinterpretCast<uint32_t>();
        const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;
        const uint32_t leftElements = leftBytes / sizeof(uint32_t);
        const uint32_t destinationStart = localPrefix - leftBytes;
        const uint16_t destinationRepeatStride =
            static_cast<uint16_t>(slabStride_ / DATA_BLOCK_BYTES);

        // The segment size is block aligned, so each segment begins at the
        // same destination lane. Key 67 widens only when doing so strictly
        // reduces the segment count. A 256-byte span is legal at residue zero;
        // other residues reserve one 32-byte lane block and use at most 224
        // bytes, keeping every active Gather offset inside one 64-lane record.
        uint32_t segmentSpan = DYNAMIC_SLAB_SEGMENT_BYTES;
        if constexpr (WIDE_GATHER_SEGMENTS) {
            const uint32_t widenedSpan =
                leftBytes == 0U ? 256U : 224U;
            const uint32_t baselineSegments =
                (rowBytes + DYNAMIC_SLAB_SEGMENT_BYTES - 1U) /
                    DYNAMIC_SLAB_SEGMENT_BYTES;
            const uint32_t widenedSegments =
                (rowBytes + widenedSpan - 1U) / widenedSpan;
            if (widenedSegments <= baselineSegments) {
                segmentSpan = widenedSpan;
            }
        }
        const bool useScratchOffset32 = !offsetsReady &&
            CanUseScratchOffsetGather32(rows, rowBytes, scratchCapacity);
        if (useScratchOffset32) {
            const uint32_t tableOffset =
                ScratchOffsetGather32TableOffset(rows, rowBytes);
            offsetValues =
                scratchBytes[tableOffset].ReinterpretCast<int32_t>();
            offsets =
                scratchBytes[tableOffset].ReinterpretCast<uint32_t>();
        } else if (!offsetsReady) {
            GenerateGatherOffsetsWavefront(
                offsetValues, rowBytes, leftBytes,
                leftElements + segmentSpan / sizeof(uint32_t),
                offsetSlot, offsetPendingMask);
        }

        uint32_t segmentOffset = 0U;
        while (segmentOffset < rowBytes) {
            const uint32_t remaining = rowBytes - segmentOffset;
            const uint32_t segmentBytes =
                remaining < segmentSpan ? remaining : segmentSpan;
            const uint32_t segmentElements =
                segmentBytes / sizeof(uint32_t);
            LocalTensor<uint32_t> destination =
                slabBytes[destinationStart + segmentOffset]
                    .ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> segmentSource =
                source[segmentOffset / sizeof(uint32_t)];
            uint64_t mask[2];
            MakeMask(mask, leftElements, segmentElements);

            uint32_t row = 0U;
            if (useScratchOffset32) {
                for (;
                     row + DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS <= rows;
                     row += DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS) {
                    Gather<uint32_t>(
                        destination[row * slabStride_ / sizeof(uint32_t)],
                        segmentSource, offsets, row * rowBytes, mask,
                        static_cast<uint8_t>(
                            DYNAMIC_SLAB_SCRATCH_OFFSET32_ROWS),
                        destinationRepeatStride);
                }
            } else {
                for (; row + GATHER_ROWS <= rows; row += GATHER_ROWS) {
                    Gather<uint32_t>(
                        destination[row * slabStride_ / sizeof(uint32_t)],
                        segmentSource, offsets, row * rowBytes, mask,
                        static_cast<uint8_t>(GATHER_ROWS),
                        destinationRepeatStride);
                }
            }
            const uint32_t tailRows = static_cast<uint32_t>(rows) - row;
            if (tailRows != 0U) {
                Gather<uint32_t>(
                    destination[row * slabStride_ / sizeof(uint32_t)],
                    segmentSource, offsets, row * rowBytes, mask,
                    static_cast<uint8_t>(tailRows),
                    destinationRepeatStride);
            }
            segmentOffset += segmentBytes;
        }
        if (!useScratchOffset32 && !offsetsReady) {
            PublishGatherOffsetRelease(offsetSlot, offsetPendingMask);
        }
    }

    __aicore__ inline uint32_t NextNonzero(
        uint32_t input, uint32_t end, uint32_t& packedWidths,
        uint32_t& rowBytes) const
    {
        while (input < end) {
            rowBytes = RowBytes(input, packedWidths);
            if (rowBytes != 0U) {
                return input;
            }
            ++input;
        }
        rowBytes = 0U;
        return end;
    }

    __aicore__ inline event_t TripleScratchEvent(uint32_t slot) const
    {
        if (slot == 0U) {
            return EVENT_ID1;
        }
        if (slot == 1U) {
            return EVENT_ID2;
        }
        return EVENT_ID3;
    }

    __aicore__ inline LocalTensor<uint8_t> TripleScratchForSlot(
        uint32_t slot, LocalTensor<uint8_t> scratch0,
        LocalTensor<uint8_t> scratch1,
        LocalTensor<uint8_t> scratch2) const
    {
        if (slot == 0U) {
            return scratch0;
        }
        if (slot == 1U) {
            return scratch1;
        }
        return scratch2;
    }

    __aicore__ inline void ConsumeTripleScratchRelease(
        uint32_t slot, uint32_t& pendingMask) const
    {
        const uint32_t bit = 1U << slot;
        if ((pendingMask & bit) != 0U) {
            WaitFlag<HardEvent::V_MTE2>(TripleScratchEvent(slot));
            pendingMask &= ~bit;
        }
    }

    __aicore__ inline void PackTripleScratchToSlab(
        LocalTensor<uint8_t> slab, LocalTensor<uint8_t> scratch,
        LocalTensor<uint8_t> gatherOffsets, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes, uint32_t sourceStride)
    {
        if (UseCompactGather(rowBytes, localPrefix)) {
            CompactGatherToSlab(
                slab, scratch, gatherOffsets, rows, localPrefix, rowBytes);
        } else if (UseSegmentedGather(rowBytes, localPrefix)) {
            SegmentedGatherToSlab(
                slab, scratch, gatherOffsets, rows, localPrefix, rowBytes);
        } else {
            VectorPack(slab, scratch, rows, localPrefix, rowBytes,
                       sourceStride);
        }
    }

    // Three independent scratch generations keep two future input MTE2
    // transfers queued while Vector packs the current input. EVENT_ID3 is
    // reused only after Process() has consumed the pointer-table MTE2_S token;
    // within this function every generation is protected by its own
    // MTE2_V/V_MTE2 pair and the output slab retains the key-47 V/MTE3 chain.
    __aicore__ inline void ProcessRowBatchTriple(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint64_t> cachedInputPtrs,
        LocalTensor<uint8_t> scratch2)
    {
        LocalTensor<uint8_t> slab = slabBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> scratch0 = scratchBuffer0_.Get<uint8_t>();
        LocalTensor<uint8_t> scratch1 = scratchBuffer1_.Get<uint8_t>();
        LocalTensor<uint8_t> gatherOffsets =
            gatherOffsetBuffer_.Get<uint8_t>();
        uint32_t pendingMask = 0U;
        bool slabPending = false;
        uint32_t groupStart = 0U;
        uint64_t outputPrefix = 0U;

        for (uint32_t group = 0U; group < groupCount_; ++group) {
            const uint32_t groupEnd = GroupEnd(group);
            if (groupStart >= tiling_->inputNum || groupEnd <= groupStart ||
                groupEnd > tiling_->inputNum) {
                break;
            }
            uint32_t packedWidths =
                tiling_->launchPadding[groupStart / 4U];
            uint32_t scan = groupStart;
            uint32_t localPrefix = 0U;
            uint32_t slotPrefix[3] = {0U, 0U, 0U};
            uint32_t slotRowBytes[3] = {0U, 0U, 0U};
            uint32_t slotStride[3] = {0U, 0U, 0U};
            uint32_t loaded = 0U;
            uint32_t processed = 0U;

            // Fill all available generations before the first Vector wait.
            for (uint32_t slot = 0U; slot < 3U; ++slot) {
                uint32_t rowBytes = 0U;
                const uint32_t input = NextNonzero(
                    scan, groupEnd, packedWidths, rowBytes);
                if (input == groupEnd) {
                    break;
                }
                ConsumeTripleScratchRelease(slot, pendingMask);
                LocalTensor<uint8_t> scratch = TripleScratchForSlot(
                    slot, scratch0, scratch1, scratch2);
                slotPrefix[slot] = localPrefix;
                slotRowBytes[slot] = rowBytes;
                slotStride[slot] = LoadScratch(
                    input, startRow, rows, localPrefix, rowBytes, scratch,
                    TripleScratchEvent(slot), cachedInputPtrs);
                localPrefix += rowBytes;
                scan = input + 1U;
                ++loaded;
            }
            if (slabPending) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                slabPending = false;
            }

            // Steady state: consume one generation and immediately refill it
            // behind the other two queued or ready generations.
            while (processed < loaded) {
                const uint32_t slot = processed % 3U;
                const event_t event = TripleScratchEvent(slot);
                LocalTensor<uint8_t> scratch = TripleScratchForSlot(
                    slot, scratch0, scratch1, scratch2);
                WaitFlag<HardEvent::MTE2_V>(event);
                PackTripleScratchToSlab(
                    slab, scratch, gatherOffsets, rows, slotPrefix[slot],
                    slotRowBytes[slot], slotStride[slot]);
                SetFlag<HardEvent::V_MTE2>(event);
                pendingMask |= 1U << slot;
                ++processed;

                uint32_t rowBytes = 0U;
                const uint32_t input = NextNonzero(
                    scan, groupEnd, packedWidths, rowBytes);
                if (input < groupEnd) {
                    ConsumeTripleScratchRelease(slot, pendingMask);
                    slotPrefix[slot] = localPrefix;
                    slotRowBytes[slot] = rowBytes;
                    slotStride[slot] = LoadScratch(
                        input, startRow, rows, localPrefix, rowBytes,
                        scratch, event, cachedInputPtrs);
                    localPrefix += rowBytes;
                    scan = input + 1U;
                    ++loaded;
                }
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t alignedGroup =
                AlignUp(localPrefix, DATA_BLOCK_BYTES);
            const DataCopyExtParams outputParams{
                rows, localPrefix,
                (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
                static_cast<uint32_t>(
                    tiling_->outputRowBytes - localPrefix),
                0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                slab, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            slabPending = true;
            outputPrefix += localPrefix;
            groupStart = groupEnd;
        }

        if (slabPending) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for (uint32_t slot = 0U; slot < 3U; ++slot) {
            ConsumeTripleScratchRelease(slot, pendingMask);
        }
    }

    __aicore__ inline void ProcessRowBatch(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint64_t> cachedInputPtrs)
    {
        LocalTensor<uint8_t> slab = slabBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> scratch0 = scratchBuffer0_.Get<uint8_t>();
        LocalTensor<uint8_t> scratch1 = scratchBuffer1_.Get<uint8_t>();
        LocalTensor<uint8_t> gatherOffsets;
        if constexpr (COMPACT_GATHER_ROWS || SEGMENTED_GATHER_ROWS) {
            gatherOffsets = gatherOffsetBuffer_.Get<uint8_t>();
        }
        bool slabPending = false;
        bool scratchPending0 = false;
        bool scratchPending1 = false;
        uint32_t groupStart = 0U;
        uint64_t outputPrefix = 0U;

        for (uint32_t group = 0U; group < groupCount_; ++group) {
            const uint32_t groupEnd = GroupEnd(group);
            if (groupStart >= tiling_->inputNum || groupEnd <= groupStart ||
                groupEnd > tiling_->inputNum) {
                break;
            }
            uint32_t packedWidths =
                tiling_->launchPadding[groupStart / 4U];
            uint32_t rowBytes = 0U;
            uint32_t current = NextNonzero(
                groupStart, groupEnd, packedWidths, rowBytes);
            if (current == groupEnd) {
                groupStart = groupEnd;
                continue;
            }
            if (scratchPending0) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                scratchPending0 = false;
            }
            uint32_t currentStride = LoadScratch(
                current, startRow, rows, 0U, rowBytes, scratch0,
                EVENT_ID1, cachedInputPtrs);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (slabPending) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                slabPending = false;
            }

            uint32_t localPrefix = 0U;
            uint32_t currentSlot = 0U;
            while (true) {
                const uint32_t nextPrefix = localPrefix + rowBytes;
                uint32_t nextRowBytes = 0U;
                uint32_t nextPackedWidths = packedWidths;
                const uint32_t next = NextNonzero(
                    current + 1U, groupEnd, nextPackedWidths,
                    nextRowBytes);
                const uint32_t nextSlot = currentSlot ^ 1U;
                uint32_t nextStride = 0U;
                if (next < groupEnd) {
                    if (nextSlot == 0U) {
                        if (scratchPending0) {
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                            scratchPending0 = false;
                        }
                        nextStride = LoadScratch(
                            next, startRow, rows, nextPrefix,
                            nextRowBytes, scratch0, EVENT_ID1,
                            cachedInputPtrs);
                    } else {
                        if (scratchPending1) {
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                            scratchPending1 = false;
                        }
                        nextStride = LoadScratch(
                            next, startRow, rows, nextPrefix,
                            nextRowBytes, scratch1, EVENT_ID2,
                            cachedInputPtrs);
                    }
                }

                if (currentSlot == 0U) {
                    if (UseCompactGather(rowBytes, localPrefix)) {
                        CompactGatherToSlab(
                            slab, scratch0, gatherOffsets, rows,
                            localPrefix, rowBytes);
                    } else if (UseSegmentedGather(rowBytes, localPrefix)) {
                        SegmentedGatherToSlab(
                            slab, scratch0, gatherOffsets, rows,
                            localPrefix, rowBytes);
                    } else {
                        VectorPack(slab, scratch0, rows, localPrefix,
                                   rowBytes, currentStride);
                    }
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    scratchPending0 = true;
                } else {
                    if (UseCompactGather(rowBytes, localPrefix)) {
                        CompactGatherToSlab(
                            slab, scratch1, gatherOffsets, rows,
                            localPrefix, rowBytes);
                    } else if (UseSegmentedGather(rowBytes, localPrefix)) {
                        SegmentedGatherToSlab(
                            slab, scratch1, gatherOffsets, rows,
                            localPrefix, rowBytes);
                    } else {
                        VectorPack(slab, scratch1, rows, localPrefix,
                                   rowBytes, currentStride);
                    }
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
                    scratchPending1 = true;
                }
                localPrefix = nextPrefix;
                if (next == groupEnd) {
                    break;
                }
                if (nextSlot == 0U) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                } else {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }
                current = next;
                currentSlot = nextSlot;
                currentStride = nextStride;
                rowBytes = nextRowBytes;
                packedWidths = nextPackedWidths;
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t alignedGroup =
                AlignUp(localPrefix, DATA_BLOCK_BYTES);
            const DataCopyExtParams outputParams{
                rows, localPrefix,
                (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
                static_cast<uint32_t>(tiling_->outputRowBytes - localPrefix),
                0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                slab, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            slabPending = true;
            outputPrefix += localPrefix;
            groupStart = groupEnd;
        }

        if (slabPending) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        if (scratchPending0) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        if (scratchPending1) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value,
                                       uint32_t alignment) const
    {
        return (value + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> slabBuffer_;
    TBuf<TPosition::VECCALC> scratchBuffer0_;
    TBuf<TPosition::VECCALC> scratchBuffer1_;
    TBuf<TPosition::VECCALC> gatherOffsetBuffer_;
    TBuf<TPosition::VECCALC> gatherBaseBuffer_;
    TBuf<TPosition::VECCALC> pointerBuffer_;
    __gm__ uint64_t* inputDataPtrs_ = nullptr;
    GlobalTensor<uint8_t> outputGm_;
    const __gm__ ConcatTilingData* tiling_ = nullptr;
    uint32_t groupCount_ = 0U;
    uint32_t slabStride_ = 0U;
    uint32_t scratchStride_ = 0U;
    uint32_t scratchStride1_ = 0U;
    uint32_t scratchStride2_ = 0U;
    uint32_t scratchStride3_ = 0U;
    uint32_t pointerBytes_ = 0U;
    uint32_t metadataVersion_ = 0U;
    uint32_t adaptiveMetadataVersion_ = 0U;
    uint64_t l2BypassOffset_ = 0U;
};

// Keep the established key-47/key-46 class and its two-slot Process() intact.
// Key 48 owns the extra UB handle and dispatch loop independently. Key 67
// instantiates the same three-generation event schedule with sixteen Gather
// records and strictly beneficial 224/256-byte segments. Key 68 keeps that
// data path while moving pointer reads to GM and offset expansion to Vector.
template <uint32_t GATHER_ROWS = DYNAMIC_SLAB_GATHER_SEED_ROWS,
          bool WIDE_GATHER_SEGMENTS = false,
          bool SCALAR_FREE_METADATA = false>
class ConcatDynamicSlabTripleKernel :
    public ConcatDynamicSlabKernel<true, true, false, GATHER_ROWS, 1U,
                                   SCALAR_FREE_METADATA,
                                   SCALAR_FREE_METADATA,
                                   SCALAR_FREE_METADATA,
                                   WIDE_GATHER_SEGMENTS> {
    using Base = ConcatDynamicSlabKernel<
        true, true, false, GATHER_ROWS, 1U,
        SCALAR_FREE_METADATA, SCALAR_FREE_METADATA, SCALAR_FREE_METADATA,
        WIDE_GATHER_SEGMENTS>;

protected:
    using Base::groupCount_;
    using Base::inputDataPtrs_;
    using Base::InitializeGatherOffsetBase;
    using Base::metadataVersion_;
    using Base::Min;
    using Base::pipe_;
    using Base::pointerBuffer_;
    using Base::pointerBytes_;
    using Base::PrepareScratchGatherOffsets32;
    using Base::ProcessRowBatchTriple;
    using Base::scratchStride_;
    using Base::slabStride_;
    using Base::tiling_;

public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        Base::Init(inputs, output, tilingData);
        const uint32_t scratchBytes =
            tilingData->rowsPerBatch * scratchStride_;
        pipe_.InitBuffer(scratchBuffer2_, scratchBytes);
    }

    __aicore__ inline void Process()
    {
        if (metadataVersion_ != DYNAMIC_SLAB_METADATA_VERSION ||
            groupCount_ == 0U || groupCount_ > 32U ||
            slabStride_ == 0U || scratchStride_ == 0U ||
            tiling_->rowsPerBatch == 0U) {
            return;
        }
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint32_t scheduleWord = tiling_->launchPadding[
            PACKED_WIDTH_WORDS + static_cast<uint32_t>(blockIndex) / 4U];
        const uint32_t scheduleByte =
            (scheduleWord >> ((blockIndex % 4U) * 8U)) & 0xffU;
        const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
        uint64_t row = blockIndex * rowsPerBlock + (scheduleByte & 0x3fU);
        const uint64_t endRow = row + rowsPerBlock +
            ((scheduleByte >> 6U) & 1U);
        if (row >= tiling_->outerRows) {
            return;
        }

        LocalTensor<uint64_t> cachedInputPtrs;
        if constexpr (!SCALAR_FREE_METADATA) {
            cachedInputPtrs = pointerBuffer_.template Get<uint64_t>();
            GlobalTensor<uint64_t> inputPtrGm;
            inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
            DataCopy(cachedInputPtrs, inputPtrGm,
                     pointerBytes_ / sizeof(uint64_t));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        }
        InitializeGatherOffsetBase();

        LocalTensor<uint8_t> scratch2 = scratchBuffer2_.Get<uint8_t>();
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row,
                static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatchTriple(
                row, rows, cachedInputPtrs, scratch2);
            row += rows;
        }
    }

private:
    TBuf<TPosition::VECCALC> scratchBuffer2_;
};

// Key 49 preserves key 48's row ownership and output-slab geometry. Key 56
// reuses the identical four-slot pipeline with sixteen Gather records. Key 57
// probes two Scalar-generated tables; key 58 instead synthesizes one table
// entirely on Vector. Key 59 also reads input pointers directly from GM. Key
// 60 precomputes the ArithProgression base before the input loop so the loop
// itself contains no Scalar UB accesses. Key 69 retains that metadata path
// and the four-slot FIFO while widening segmented Gather spans as in key 68.
template <uint32_t GATHER_ROWS = DYNAMIC_SLAB_GATHER_SEED_ROWS,
          uint32_t OFFSET_GENERATIONS = 1U,
          bool VECTOR_GATHER_OFFSETS = false,
          bool DIRECT_GM_POINTERS = false,
          bool PRECOMPUTED_GATHER_BASE = false,
          bool WIDE_GATHER_SEGMENTS = false,
          bool SCRATCH_OFFSET_GATHER32 = false>
class ConcatDynamicSlabAdaptiveKernel :
    public ConcatDynamicSlabKernel<true, true, true, GATHER_ROWS,
                                   OFFSET_GENERATIONS,
                                   VECTOR_GATHER_OFFSETS,
                                   DIRECT_GM_POINTERS,
                                   PRECOMPUTED_GATHER_BASE,
                                   WIDE_GATHER_SEGMENTS,
                                   SCRATCH_OFFSET_GATHER32> {
    using Base =
        ConcatDynamicSlabKernel<true, true, true, GATHER_ROWS,
                                OFFSET_GENERATIONS,
                                VECTOR_GATHER_OFFSETS,
                                DIRECT_GM_POINTERS,
                                PRECOMPUTED_GATHER_BASE,
                                WIDE_GATHER_SEGMENTS,
                                SCRATCH_OFFSET_GATHER32>;

protected:
    using Base::adaptiveMetadataVersion_;
    using Base::AlignUp;
    using Base::CompactGatherToSlab;
    using Base::DrainGatherOffsetReleases;
    using Base::gatherOffsetBuffer_;
    using Base::groupCount_;
    using Base::GroupEnd;
    using Base::inputDataPtrs_;
    using Base::InitializeGatherOffsetBase;
    using Base::LoadScratch;
    using Base::metadataVersion_;
    using Base::Min;
    using Base::NextNonzero;
    using Base::outputGm_;
    using Base::pipe_;
    using Base::pointerBuffer_;
    using Base::pointerBytes_;
    using Base::scratchBuffer0_;
    using Base::scratchBuffer1_;
    using Base::scratchStride_;
    using Base::scratchStride1_;
    using Base::scratchStride2_;
    using Base::scratchStride3_;
    using Base::SegmentedGatherToSlab;
    using Base::slabBuffer_;
    using Base::slabStride_;
    using Base::tiling_;
    using Base::UseCompactGather;
    using Base::UseSegmentedGather;
    using Base::VectorPack;

public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        Base::Init(inputs, output, tilingData);
        pipe_.InitBuffer(
            scratchBuffer2_, tilingData->rowsPerBatch * scratchStride2_);
        pipe_.InitBuffer(
            scratchBuffer3_, tilingData->rowsPerBatch * scratchStride3_);
    }

    __aicore__ inline void Process()
    {
        if (metadataVersion_ != DYNAMIC_SLAB_METADATA_VERSION ||
            adaptiveMetadataVersion_ != 1U ||
            groupCount_ == 0U || groupCount_ >= 32U ||
            slabStride_ == 0U || scratchStride_ == 0U ||
            scratchStride1_ == 0U || scratchStride2_ == 0U ||
            scratchStride3_ == 0U || tiling_->rowsPerBatch == 0U) {
            return;
        }
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint32_t scheduleWord = tiling_->launchPadding[
            PACKED_WIDTH_WORDS + static_cast<uint32_t>(blockIndex) / 4U];
        const uint32_t scheduleByte =
            (scheduleWord >> ((blockIndex % 4U) * 8U)) & 0xffU;
        const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
        uint64_t row = blockIndex * rowsPerBlock + (scheduleByte & 0x3fU);
        const uint64_t endRow = row + rowsPerBlock +
            ((scheduleByte >> 6U) & 1U);
        if (row >= tiling_->outerRows) {
            return;
        }

        LocalTensor<uint64_t> cachedInputPtrs;
        if constexpr (!DIRECT_GM_POINTERS) {
            cachedInputPtrs = pointerBuffer_.template Get<uint64_t>();
            GlobalTensor<uint64_t> inputPtrGm;
            inputPtrGm.SetGlobalBuffer(inputDataPtrs_);
            DataCopy(cachedInputPtrs, inputPtrGm,
                     pointerBytes_ / sizeof(uint64_t));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        }
        InitializeGatherOffsetBase();

        LocalTensor<uint8_t> scratch2 = scratchBuffer2_.Get<uint8_t>();
        LocalTensor<uint8_t> scratch3 = scratchBuffer3_.Get<uint8_t>();
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row,
                static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatchAdaptive(
                row, rows, cachedInputPtrs, scratch2, scratch3);
            row += rows;
        }
    }

private:
    __aicore__ inline event_t AdaptiveEvent(uint32_t slot) const
    {
        if (slot == 0U) {
            return EVENT_ID1;
        }
        if (slot == 1U) {
            return EVENT_ID2;
        }
        if (slot == 2U) {
            return EVENT_ID3;
        }
        return EVENT_ID4;
    }

    __aicore__ inline uint32_t AdaptiveCapacity(uint32_t slot) const
    {
        if (slot == 0U) {
            return scratchStride_;
        }
        if (slot == 1U) {
            return scratchStride1_;
        }
        if (slot == 2U) {
            return scratchStride2_;
        }
        return scratchStride3_;
    }

    __aicore__ inline LocalTensor<uint8_t> AdaptiveScratch(
        uint32_t slot, LocalTensor<uint8_t> scratch0,
        LocalTensor<uint8_t> scratch1, LocalTensor<uint8_t> scratch2,
        LocalTensor<uint8_t> scratch3) const
    {
        if (slot == 0U) {
            return scratch0;
        }
        if (slot == 1U) {
            return scratch1;
        }
        if (slot == 2U) {
            return scratch2;
        }
        return scratch3;
    }

    __aicore__ inline uint32_t ScratchRequirement(
        uint32_t localPrefix, uint32_t rowBytes) const
    {
        if (UseCompactGather(rowBytes, localPrefix) ||
            UseSegmentedGather(rowBytes, localPrefix)) {
            return AlignUp(rowBytes, DATA_BLOCK_BYTES);
        }
        return AlignUp(
            localPrefix % DATA_BLOCK_BYTES + rowBytes, DATA_BLOCK_BYTES);
    }

    __aicore__ inline uint32_t SelectScratch(
        uint32_t freeMask, uint32_t requirement) const
    {
        uint32_t selected = 4U;
        uint32_t selectedCapacity = 0xffffffffU;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            const uint32_t capacity = AdaptiveCapacity(slot);
            if ((freeMask & (1U << slot)) != 0U &&
                capacity >= requirement && capacity < selectedCapacity) {
                selected = slot;
                selectedCapacity = capacity;
            }
        }
        return selected;
    }

    __aicore__ inline void ConsumeScratchRelease(
        uint32_t slot, uint32_t& pendingMask) const
    {
        const uint32_t bit = 1U << slot;
        if ((pendingMask & bit) != 0U) {
            WaitFlag<HardEvent::V_MTE2>(AdaptiveEvent(slot));
            pendingMask &= ~bit;
        }
    }

    __aicore__ inline void ProcessRowBatchAdaptive(
        uint64_t startRow, uint16_t rows,
        LocalTensor<uint64_t> cachedInputPtrs,
        LocalTensor<uint8_t> scratch2,
        LocalTensor<uint8_t> scratch3)
    {
        LocalTensor<uint8_t> slab =
            slabBuffer_.template Get<uint8_t>();
        LocalTensor<uint8_t> scratch0 =
            scratchBuffer0_.template Get<uint8_t>();
        LocalTensor<uint8_t> scratch1 =
            scratchBuffer1_.template Get<uint8_t>();
        LocalTensor<uint8_t> gatherOffsets =
            gatherOffsetBuffer_.template Get<uint8_t>();
        uint32_t pendingMask = 0U;
        uint32_t offsetPendingMask = 0U;
        uint32_t offsetSequence = 0U;
        bool slabPending = false;
        uint32_t groupStart = 0U;
        uint64_t outputPrefix = 0U;

        for (uint32_t group = 0U; group < groupCount_; ++group) {
            const uint32_t groupEnd = GroupEnd(group);
            uint32_t packedWidths =
                tiling_->launchPadding[groupStart / 4U];
            uint32_t scan = groupStart;
            uint32_t localPrefix = 0U;
            uint32_t freeMask = 0xfU;
            uint32_t queueSlot[4] = {};
            uint32_t queuePrefix[4] = {};
            uint32_t queueRowBytes[4] = {};
            uint32_t queueStride[4] = {};
            uint32_t queueHead = 0U;
            uint32_t queueCount = 0U;
            bool slabReady = !slabPending;

            while (scan < groupEnd || queueCount != 0U) {
                while (scan < groupEnd && queueCount < 4U) {
                    uint32_t rowBytes = 0U;
                    const uint32_t input = NextNonzero(
                        scan, groupEnd, packedWidths, rowBytes);
                    if (input == groupEnd) {
                        scan = groupEnd;
                        break;
                    }
                    const uint32_t slot = SelectScratch(
                        freeMask,
                        ScratchRequirement(localPrefix, rowBytes));
                    if (slot == 4U) {
                        break;
                    }
                    ConsumeScratchRelease(slot, pendingMask);
                    const event_t event = AdaptiveEvent(slot);
                    LocalTensor<uint8_t> scratch = AdaptiveScratch(
                        slot, scratch0, scratch1, scratch2, scratch3);
                    const uint32_t tail =
                        (queueHead + queueCount) & 3U;
                    queueSlot[tail] = slot;
                    queuePrefix[tail] = localPrefix;
                    queueRowBytes[tail] = rowBytes;
                    queueStride[tail] = LoadScratch(
                        input, startRow, rows, localPrefix, rowBytes,
                        scratch, event, cachedInputPtrs);
                    if constexpr (SCRATCH_OFFSET_GATHER32) {
                        this->PrepareScratchGatherOffsets32(
                            scratch, rows, localPrefix, rowBytes,
                            AdaptiveCapacity(slot));
                    }
                    freeMask &= ~(1U << slot);
                    ++queueCount;
                    scan = input + 1U;
                    localPrefix += rowBytes;
                }

                if (queueCount == 0U) {
                    return;
                }
                const uint32_t head = queueHead;
                const uint32_t slot = queueSlot[head];
                const event_t event = AdaptiveEvent(slot);
                LocalTensor<uint8_t> scratch = AdaptiveScratch(
                    slot, scratch0, scratch1, scratch2, scratch3);
                WaitFlag<HardEvent::MTE2_V>(event);
                if (!slabReady) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    slabPending = false;
                    slabReady = true;
                }
                if constexpr (OFFSET_GENERATIONS == 1U) {
                    if (UseCompactGather(
                            queueRowBytes[head], queuePrefix[head])) {
                        CompactGatherToSlab(
                            slab, scratch, gatherOffsets, rows,
                            queuePrefix[head], queueRowBytes[head], 0U,
                            nullptr, AdaptiveCapacity(slot));
                    } else if (UseSegmentedGather(
                                   queueRowBytes[head], queuePrefix[head])) {
                        SegmentedGatherToSlab(
                            slab, scratch, gatherOffsets, rows,
                            queuePrefix[head], queueRowBytes[head], 0U,
                            nullptr, AdaptiveCapacity(slot));
                    } else {
                        VectorPack(
                            slab, scratch, rows, queuePrefix[head],
                            queueRowBytes[head], queueStride[head]);
                    }
                } else {
                    const bool compact =
                        UseCompactGather(
                            queueRowBytes[head], queuePrefix[head]);
                    const bool segmented =
                        UseSegmentedGather(
                            queueRowBytes[head], queuePrefix[head]);
                    if (compact || segmented) {
                        const uint32_t offsetSlot =
                            offsetSequence % OFFSET_GENERATIONS;
                        constexpr uint32_t OFFSET_TABLE_BYTES =
                            GATHER_ROWS *
                            DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT *
                            sizeof(uint32_t);
                        LocalTensor<uint8_t> activeOffsets =
                            gatherOffsets[offsetSlot * OFFSET_TABLE_BYTES];
                        if (compact) {
                            CompactGatherToSlab(
                                slab, scratch, activeOffsets, rows,
                                queuePrefix[head], queueRowBytes[head],
                                offsetSlot, &offsetPendingMask);
                        } else {
                            SegmentedGatherToSlab(
                                slab, scratch, activeOffsets, rows,
                                queuePrefix[head], queueRowBytes[head],
                                offsetSlot, &offsetPendingMask);
                        }
                        ++offsetSequence;
                    } else {
                        VectorPack(
                            slab, scratch, rows, queuePrefix[head],
                            queueRowBytes[head], queueStride[head]);
                    }
                }
                SetFlag<HardEvent::V_MTE2>(event);
                pendingMask |= 1U << slot;
                freeMask |= 1U << slot;
                queueHead = (queueHead + 1U) & 3U;
                --queueCount;
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t alignedGroup =
                AlignUp(localPrefix, DATA_BLOCK_BYTES);
            const DataCopyExtParams outputParams{
                rows, localPrefix,
                (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
                static_cast<uint32_t>(
                    tiling_->outputRowBytes - localPrefix), 0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                slab, outputParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            slabPending = true;
            outputPrefix += localPrefix;
            groupStart = groupEnd;
        }

        if (slabPending) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            ConsumeScratchRelease(slot, pendingMask);
        }
        DrainGatherOffsetReleases(offsetPendingMask);
    }

    TBuf<TPosition::VECCALC> scratchBuffer2_;
    TBuf<TPosition::VECCALC> scratchBuffer3_;
};

// Key 92 changes the ownership grid: one block owns one contiguous output
// group across all of its assigned rows.  The Host gives the eight heaviest
// groups a second row shard on a 40-AIV launch.  Every block derives a local
// slab and four scratch capacities from its own group, so row batches grow
// beyond key 79's global 50/51-row allocation without enlarging launch tiling.
// If two compact inputs have identical runtime Gather geometry and the full
// batch fits in UB, one shared offset table funds a persistent input arena.
// The arena keeps every source live while two disjoint row bands of the slab
// form a Vector/MTE3 software pipeline; input DMA granularity and Gather
// repeat count remain unchanged.
class ConcatDynamicSlabGroupOwnerKernel :
    public ConcatDynamicSlabKernel<true, true, true, 16U, 1U,
                                   true, true, true, true, true, 4U, true,
                                   true> {
    using Base = ConcatDynamicSlabKernel<
        true, true, true, 16U, 1U, true, true, true, true, true, 4U, true,
        true>;

protected:
    using Base::AlignUp;
    using Base::CompactGatherToSlab;
    using Base::gatherBaseBuffer_;
    using Base::gatherOffsetBuffer_;
    using Base::GenerateGatherOffsetsWavefront;
    using Base::GroupEnd;
    using Base::groupCount_;
    using Base::InitializeGatherOffsetBase;
    using Base::inputDataPtrs_;
    using Base::l2BypassOffset_;
    using Base::LoadScratch;
    using Base::metadataVersion_;
    using Base::Min;
    using Base::NextNonzero;
    using Base::outputGm_;
    using Base::pipe_;
    using Base::RowBytes;
    using Base::scratchBuffer0_;
    using Base::scratchBuffer1_;
    using Base::scratchStride_;
    using Base::scratchStride1_;
    using Base::scratchStride2_;
    using Base::scratchStride3_;
    using Base::SegmentedGatherToSlab;
    using Base::slabBuffer_;
    using Base::slabStride_;
    using Base::tiling_;
    using Base::UseCompactGather;
    using Base::UseSegmentedGather;
    using Base::VectorPack;

public:
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatTilingData* tilingData)
    {
        tiling_ = tilingData;
        const __gm__ uint64_t* descriptorStart =
            reinterpret_cast<__gm__ uint64_t*>(inputs);
        inputDataPtrs_ = const_cast<__gm__ uint64_t*>(descriptorStart) +
            descriptorStart[0] / sizeof(uint64_t);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
#if defined(L2_CACHE_HINT)
        l2BypassOffset_ = g_opSystemRunCfg.l2Cacheoffset;
#endif

        const uint32_t header =
            tiling_->launchPadding[DYNAMIC_SLAB_HEADER_WORD];
        groupCount_ = header & 0xffU;
        availableUbBytes_ =
            ((header >> 8U) & 0xffffU) * DATA_BLOCK_BYTES;
        metadataVersion_ = header >> 24U;
        const uint32_t blockIndex = static_cast<uint32_t>(GetBlockIdx());
        uint32_t scheduleByte = 0U;
        if (blockIndex < 40U) {
            const uint32_t scheduleWord = tiling_->launchPadding[
                PACKED_WIDTH_WORDS + blockIndex / 4U];
            scheduleByte =
                (scheduleWord >> ((blockIndex & 3U) * 8U)) & 0xffU;
        }
        selectedGroup_ = scheduleByte & 0x1fU;
        shardIndex_ = (scheduleByte >> 5U) & 1U;
        shardCount_ = 1U + ((scheduleByte >> 6U) & 1U);
        groupStart_ = selectedGroup_ == 0U ? 0U :
            GroupEnd(selectedGroup_ - 1U);
        groupEnd_ = selectedGroup_ < groupCount_ ?
            GroupEnd(selectedGroup_) : 0U;

        uint32_t arenaPairByte = DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR;
        if (selectedGroup_ < DYNAMIC_SLAB_GROUP_OWNER_GROUPS) {
            const uint32_t pairWord = tiling_->launchPadding[
                DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD +
                selectedGroup_ / 4U];
            arenaPairByte =
                (pairWord >> ((selectedGroup_ & 3U) * 8U)) & 0xffU;
        }
        arenaPairFirst_ = arenaPairByte & 0xfU;
        arenaPairSecond_ = arenaPairByte >> 4U;
        const bool tightArena =
            arenaPairByte == DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA;
        const uint32_t encodedTightBudget =
            tiling_->launchPadding[
                DYNAMIC_SLAB_GROUP_OWNER_TIGHT_BUDGET_WORD] & 0xffffU;
        const uint32_t arenaAvailableUbBytes = tightArena &&
                encodedTightBudget != 0U ?
            encodedTightBudget * DATA_BLOCK_BYTES : availableUbBytes_;

        uint32_t packedWidths = tiling_->launchPadding[0U];
        uint32_t localPrefix = 0U;
        uint32_t nonzero = 0U;
        gatherInputCount_ = 0U;
        outputPrefix_ = 0U;
        scratchStride_ = 0U;
        scratchStride1_ = 0U;
        scratchStride2_ = 0U;
        scratchStride3_ = 0U;
        uint32_t firstPairRowBytes = 0U;
        uint32_t firstPairLeftBytes = 0U;
        uint32_t secondPairRowBytes = 0U;
        uint32_t secondPairLeftBytes = 0U;
        bool firstPairCompact = false;
        bool secondPairCompact = false;
        for (uint32_t input = 0U; input < groupEnd_; ++input) {
            const uint32_t rowBytes = RowBytes(input, packedWidths);
            if (input < groupStart_) {
                outputPrefix_ += rowBytes;
                continue;
            }
            if (rowBytes == 0U) {
                continue;
            }
            const uint32_t requirement = ScratchRequirement(
                localPrefix, rowBytes);
            // Size the four scratch generations by their runtime input
            // phase, not by the four largest rows in the complete group.
            // Inputs in one phase cannot be live simultaneously: the FIFO
            // consumes the older owner before the phase recurs.  The normal
            // best-fit allocator may still borrow a larger free generation;
            // when it does, its existing release loop eventually exposes
            // the phase owner, so progress and event ownership are unchanged.
            // This keeps four in-flight generations while spending UB only
            // on widths that can overlap in the ordered input stream.
            const uint32_t phase = nonzero & 3U;
            if (requirement > ScratchCapacity(phase)) {
                SetScratchCapacity(phase, requirement);
            }
            const bool compact = UseCompactGather(rowBytes, localPrefix);
            const bool segmented =
                UseSegmentedGather(rowBytes, localPrefix);
            if (compact || segmented) {
                if (gatherInputCount_ == arenaPairFirst_) {
                    firstPairRowBytes = rowBytes;
                    firstPairLeftBytes = localPrefix % DATA_BLOCK_BYTES;
                    firstPairCompact = compact;
                }
                if (gatherInputCount_ == arenaPairSecond_) {
                    secondPairRowBytes = rowBytes;
                    secondPairLeftBytes = localPrefix % DATA_BLOCK_BYTES;
                    secondPairCompact = compact;
                }
                ++gatherInputCount_;
            }
            localPrefix += rowBytes;
            ++nonzero;
        }
        groupBytes_ = localPrefix;
        slabStride_ = AlignUp(localPrefix, DATA_BLOCK_BYTES);
        FillScratchCapacities();
        offsetTableCount_ = gatherInputCount_ == 0U ?
            1U : gatherInputCount_;
        fixedUbBytes_ = offsetTableCount_ *
                DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES +
            DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t);
        const uint32_t perRowBytes = slabStride_ + scratchStride_ +
            scratchStride1_ + scratchStride2_ + scratchStride3_;
        const uint32_t rowStorageBytes =
            availableUbBytes_ >= fixedUbBytes_ ?
            availableUbBytes_ - fixedUbBytes_ : 0U;
        rowsPerBatch_ = perRowBytes == 0U ? 0U :
            rowStorageBytes / perRowBytes;
        if (rowsPerBatch_ > 255U) {
            rowsPerBatch_ = 255U;
        }
        const bool validArenaPair = arenaPairByte !=
                DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR &&
            arenaPairByte !=
                DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR &&
            arenaPairByte != DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA &&
            arenaPairFirst_ < arenaPairSecond_ &&
            arenaPairSecond_ < gatherInputCount_ &&
            firstPairCompact && secondPairCompact &&
            firstPairRowBytes == secondPairRowBytes &&
            firstPairLeftBytes == secondPairLeftBytes;
        const bool arenaWithoutPair = arenaPairByte ==
                DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR ||
            tightArena;
        arenaHasSharedPair_ = validArenaPair;
        const uint32_t arenaTableCount = validArenaPair &&
                gatherInputCount_ > 1U ?
            gatherInputCount_ - 1U :
            (gatherInputCount_ == 0U ? 1U : gatherInputCount_);
        uint32_t arenaRows = rowsPerBatch_;
        uint64_t arenaUbBytes = 0U;
        if (validArenaPair || arenaWithoutPair) {
            while (arenaRows >= 32U) {
                arenaBytes_ = ComputeArenaBytes(arenaRows);
                arenaUbBytes =
                    static_cast<uint64_t>(arenaRows) * slabStride_ +
                    arenaBytes_ +
                    static_cast<uint64_t>(arenaTableCount) *
                        DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES +
                    DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t);
                if (arenaBytes_ != 0U &&
                    arenaUbBytes <= arenaAvailableUbBytes) {
                    break;
                }
                --arenaRows;
            }
        }
        arenaEnabled_ = (validArenaPair || arenaWithoutPair) &&
            arenaRows >= 32U && arenaBytes_ != 0U &&
            arenaUbBytes <= arenaAvailableUbBytes;
        if (arenaEnabled_) {
            rowsPerBatch_ = arenaRows;
            offsetTableCount_ = arenaTableCount;
            fixedUbBytes_ = offsetTableCount_ *
                    DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES +
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t);
        }
        valid_ = metadataVersion_ ==
                DYNAMIC_SLAB_GROUP_OWNER_METADATA_VERSION &&
            (scheduleByte & 0x80U) != 0U &&
            groupCount_ == DYNAMIC_SLAB_GROUP_OWNER_GROUPS &&
            selectedGroup_ < groupCount_ &&
            groupStart_ < groupEnd_ && groupEnd_ <= tiling_->inputNum &&
            shardIndex_ < shardCount_ && nonzero >= 3U &&
            gatherInputCount_ <= DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS &&
            offsetTableCount_ <= DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS &&
            fixedUbBytes_ <= availableUbBytes_ &&
            (!tightArena || arenaEnabled_) &&
            slabStride_ != 0U && rowsPerBatch_ >= 32U;
        if (!valid_) {
            slabStride_ = DATA_BLOCK_BYTES;
            groupBytes_ = DATA_BLOCK_BYTES;
            scratchStride_ = DATA_BLOCK_BYTES;
            scratchStride1_ = DATA_BLOCK_BYTES;
            scratchStride2_ = DATA_BLOCK_BYTES;
            scratchStride3_ = DATA_BLOCK_BYTES;
            rowsPerBatch_ = 1U;
            gatherInputCount_ = 0U;
            offsetTableCount_ = 1U;
            arenaBytes_ = 0U;
            arenaEnabled_ = false;
            arenaHasSharedPair_ = false;
            fixedUbBytes_ = DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES +
                DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t);
        }

        pipe_.InitBuffer(slabBuffer_, rowsPerBatch_ * slabStride_);
        if (arenaEnabled_) {
            pipe_.InitBuffer(arenaBuffer_, arenaBytes_);
        } else {
            pipe_.InitBuffer(scratchBuffer0_, rowsPerBatch_ * scratchStride_);
            pipe_.InitBuffer(scratchBuffer1_, rowsPerBatch_ * scratchStride1_);
            pipe_.InitBuffer(scratchBuffer2_, rowsPerBatch_ * scratchStride2_);
            pipe_.InitBuffer(scratchBuffer3_, rowsPerBatch_ * scratchStride3_);
        }
        pipe_.InitBuffer(
            gatherOffsetBuffer_,
            offsetTableCount_ * DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES);
        pipe_.InitBuffer(
            gatherBaseBuffer_,
            DYNAMIC_SLAB_GATHER_OFFSETS_PER_REPEAT * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        const uint64_t baseRows = tiling_->outerRows / shardCount_;
        const uint64_t extraRows = tiling_->outerRows % shardCount_;
        uint64_t row = static_cast<uint64_t>(shardIndex_) * baseRows +
            Min(static_cast<uint64_t>(shardIndex_), extraRows);
        uint64_t endRow = row + baseRows +
            static_cast<uint64_t>(shardIndex_ < extraRows);

        // The output row stride can expose a repeating address residue to
        // every MTE3 row descriptor.  When the Host has already selected two
        // shards, move only their common boundary to the nearest period of
        // that residue.  The rule is derived from the runtime output stride
        // and outer-row count; it changes no bytes and keeps the two ranges
        // contiguous.  A one-row adjustment is preferable to presenting two
        // otherwise identical streams with different MTE3 phase.  Aligned
        // output strides have period one and therefore retain the original
        // partition exactly.
        if (shardCount_ == 2U && tiling_->outerRows > 1U &&
            tiling_->outputRowBytes % DATA_BLOCK_BYTES != 0U) {
            const uint32_t residue = static_cast<uint32_t>(
                tiling_->outputRowBytes % DATA_BLOCK_BYTES);
            uint32_t gcd = DATA_BLOCK_BYTES;
            uint32_t value = residue;
            while (value != 0U) {
                const uint32_t next = gcd % value;
                gcd = value;
                value = next;
            }
            const uint32_t period = DATA_BLOCK_BYTES / gcd;
            const uint64_t originalBoundary = baseRows +
                static_cast<uint64_t>(extraRows != 0U);
            uint64_t boundary = ((originalBoundary + period / 2U) /
                period) * period;
            if (boundary == 0U) {
                boundary = period;
            }
            if (boundary >= tiling_->outerRows) {
                boundary = tiling_->outerRows - 1U;
            }
            row = shardIndex_ == 0U ? 0U : boundary;
            endRow = shardIndex_ == 0U ? boundary : tiling_->outerRows;
        }
        if (row >= endRow || row >= tiling_->outerRows) {
            return;
        }
        InitializeGatherOffsetBase();
        if (!InitializePersistentGatherOffsets()) {
            return;
        }
        if (arenaEnabled_) {
            ProcessArenaRows(row, endRow);
            return;
        }
        bool slabPending = false;
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(rowsPerBatch_)));
            if (!ProcessRowBatch(row, rows, slabPending)) {
                break;
            }
            row += rows;
        }
        if (slabPending) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline event_t ScratchEvent(uint32_t slot) const
    {
        if (slot == 0U) {
            return EVENT_ID1;
        }
        if (slot == 1U) {
            return EVENT_ID2;
        }
        if (slot == 2U) {
            return EVENT_ID3;
        }
        return EVENT_ID4;
    }

    __aicore__ inline uint32_t ScratchCapacity(uint32_t slot) const
    {
        if (slot == 0U) {
            return scratchStride_;
        }
        if (slot == 1U) {
            return scratchStride1_;
        }
        if (slot == 2U) {
            return scratchStride2_;
        }
        return scratchStride3_;
    }

    __aicore__ inline void SetScratchCapacity(
        uint32_t slot, uint32_t value)
    {
        if (slot == 0U) {
            scratchStride_ = value;
        } else if (slot == 1U) {
            scratchStride1_ = value;
        } else if (slot == 2U) {
            scratchStride2_ = value;
        } else {
            scratchStride3_ = value;
        }
    }

    __aicore__ inline void InsertScratchCapacity(uint32_t requirement)
    {
        uint32_t insert = 4U;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            if (requirement > ScratchCapacity(slot)) {
                insert = slot;
                break;
            }
        }
        if (insert == 4U) {
            return;
        }
        for (uint32_t slot = 3U; slot > insert; --slot) {
            SetScratchCapacity(slot, ScratchCapacity(slot - 1U));
        }
        SetScratchCapacity(insert, requirement);
    }

    __aicore__ inline void FillScratchCapacities()
    {
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            if (ScratchCapacity(slot) == 0U) {
                SetScratchCapacity(slot, DATA_BLOCK_BYTES);
            }
        }
    }

    __aicore__ inline LocalTensor<uint8_t> ScratchTensor(uint32_t slot)
    {
        if (slot == 0U) {
            return scratchBuffer0_.Get<uint8_t>();
        }
        if (slot == 1U) {
            return scratchBuffer1_.Get<uint8_t>();
        }
        if (slot == 2U) {
            return scratchBuffer2_.Get<uint8_t>();
        }
        return scratchBuffer3_.Get<uint8_t>();
    }

    __aicore__ inline uint32_t ScratchRequirement(
        uint32_t localPrefix, uint32_t rowBytes) const
    {
        if (UseCompactGather(rowBytes, localPrefix) ||
            UseSegmentedGather(rowBytes, localPrefix)) {
            return AlignUp(rowBytes, DATA_BLOCK_BYTES);
        }
        return AlignUp(
            localPrefix % DATA_BLOCK_BYTES + rowBytes, DATA_BLOCK_BYTES);
    }

    __aicore__ inline uint32_t ArenaSourceStride(
        uint32_t localPrefix, uint32_t rowBytes) const
    {
        if (UseCompactGather(rowBytes, localPrefix) ||
            UseSegmentedGather(rowBytes, localPrefix)) {
            return rowBytes;
        }
        return AlignUp(
            localPrefix % DATA_BLOCK_BYTES + rowBytes, DATA_BLOCK_BYTES);
    }

    __aicore__ inline uint32_t ArenaSliceBytes(
        uint32_t capacityRows, uint32_t localPrefix,
        uint32_t rowBytes) const
    {
        const uint32_t stride = ArenaSourceStride(localPrefix, rowBytes);
        const uint64_t bytes = static_cast<uint64_t>(capacityRows) * stride;
        if (UseCompactGather(rowBytes, localPrefix) ||
            UseSegmentedGather(rowBytes, localPrefix)) {
            return bytes <= 0xffffffffU - (DATA_BLOCK_BYTES - 1U) ?
                AlignUp(static_cast<uint32_t>(bytes), DATA_BLOCK_BYTES) : 0U;
        }
        return bytes <= 0xffffffffU ? static_cast<uint32_t>(bytes) : 0U;
    }

    __aicore__ inline uint32_t ComputeArenaBytes(
        uint32_t capacityRows) const
    {
        uint32_t packedWidths =
            tiling_->launchPadding[groupStart_ / 4U];
        uint32_t localPrefix = 0U;
        uint64_t total = 0U;
        for (uint32_t input = groupStart_; input < groupEnd_; ++input) {
            const uint32_t rowBytes = RowBytes(input, packedWidths);
            if (rowBytes == 0U) {
                continue;
            }
            const uint32_t slice = ArenaSliceBytes(
                capacityRows, localPrefix, rowBytes);
            if (slice == 0U) {
                return 0U;
            }
            total += slice;
            if (total > 0xffffffffU) {
                return 0U;
            }
            localPrefix += rowBytes;
        }
        return static_cast<uint32_t>(total);
    }

    __aicore__ inline uint32_t GatherTableSlot(uint32_t table) const
    {
        if (!arenaHasSharedPair_) {
            return table;
        }
        if (table == arenaPairFirst_ || table == arenaPairSecond_) {
            return gatherInputCount_ - 2U;
        }
        return table - static_cast<uint32_t>(table > arenaPairFirst_) -
            static_cast<uint32_t>(table > arenaPairSecond_);
    }

    __aicore__ inline bool InitializePersistentGatherOffsets()
    {
        uint32_t packedWidths =
            tiling_->launchPadding[groupStart_ / 4U];
        uint32_t localPrefix = 0U;
        uint32_t table = 0U;
        for (uint32_t input = groupStart_; input < groupEnd_; ++input) {
            const uint32_t rowBytes = RowBytes(input, packedWidths);
            if (rowBytes == 0U) {
                continue;
            }
            const bool compact = UseCompactGather(rowBytes, localPrefix);
            const bool segmented =
                UseSegmentedGather(rowBytes, localPrefix);
            if (compact || segmented) {
                if (table >= gatherInputCount_) {
                    return false;
                }
                const uint32_t tableSlot = GatherTableSlot(table);
                if (tableSlot >= offsetTableCount_) {
                    return false;
                }
                const uint32_t leftBytes =
                    localPrefix % DATA_BLOCK_BYTES;
                uint32_t count = leftBytes / sizeof(uint32_t) +
                    rowBytes / sizeof(uint32_t);
                if (segmented) {
                    uint32_t segmentSpan = DYNAMIC_SLAB_SEGMENT_BYTES;
                    const uint32_t widenedSpan =
                        leftBytes == 0U ? 256U : 224U;
                    const uint32_t baselineSegments =
                        (rowBytes + DYNAMIC_SLAB_SEGMENT_BYTES - 1U) /
                            DYNAMIC_SLAB_SEGMENT_BYTES;
                    const uint32_t widenedSegments =
                        (rowBytes + widenedSpan - 1U) / widenedSpan;
                    if (widenedSegments <= baselineSegments) {
                        segmentSpan = widenedSpan;
                    }
                    count = leftBytes / sizeof(uint32_t) +
                        segmentSpan / sizeof(uint32_t);
                }
                if (!arenaHasSharedPair_ || table != arenaPairSecond_) {
                    LocalTensor<int32_t> offsets =
                        gatherOffsetBuffer_.Get<uint8_t>()[
                            tableSlot *
                                DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES]
                            .ReinterpretCast<int32_t>();
                    GenerateGatherOffsetsWavefront(
                        offsets, rowBytes, leftBytes, count);
                }
                ++table;
            }
            localPrefix += rowBytes;
        }
        return table == gatherInputCount_ &&
            AlignUp(localPrefix, DATA_BLOCK_BYTES) == slabStride_;
    }

    __aicore__ inline event_t ArenaSlabEvent(uint32_t band) const
    {
        return band == 0U ? EVENT_ID0 : EVENT_ID5;
    }

    __aicore__ inline void PackArenaSlice(
        const LocalTensor<uint8_t>& slab,
        const LocalTensor<uint8_t>& source, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes, uint32_t sourceStride,
        uint32_t gatherTable)
    {
        if (UseCompactGather(rowBytes, localPrefix)) {
            const uint32_t tableSlot = GatherTableSlot(gatherTable);
            LocalTensor<uint8_t> offsets =
                gatherOffsetBuffer_.Get<uint8_t>()[
                    tableSlot * DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES];
            CompactGatherToSlab(
                slab, source, offsets, rows, localPrefix, rowBytes,
                0U, nullptr, 0U, true);
        } else if (UseSegmentedGather(rowBytes, localPrefix)) {
            const uint32_t tableSlot = GatherTableSlot(gatherTable);
            LocalTensor<uint8_t> offsets =
                gatherOffsetBuffer_.Get<uint8_t>()[
                    tableSlot * DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES];
            SegmentedGatherToSlab(
                slab, source, offsets, rows, localPrefix, rowBytes,
                0U, nullptr, rowBytes, true);
        } else {
            VectorPack(
                slab, source, rows, localPrefix, rowBytes, sourceStride);
        }
    }

    __aicore__ inline void EmitArenaBand(
        uint64_t startRow, uint16_t rows,
        const LocalTensor<uint8_t>& slab, uint32_t band,
        bool pending[2])
    {
        const event_t event = ArenaSlabEvent(band);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        const uint32_t logicalBytes = groupBytes_;
        const uint32_t alignedGroup =
            AlignUp(logicalBytes, DATA_BLOCK_BYTES);
        const DataCopyExtParams outputParams{
            rows, logicalBytes,
            (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
            static_cast<uint32_t>(
                tiling_->outputRowBytes - logicalBytes), 0U};
        GlobalTensor<uint32_t> outputView;
        outputView.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            outputGm_.GetPhyAddr(
                startRow * tiling_->outputRowBytes + outputPrefix_)));
        const uint64_t outputAddress = startRow *
            static_cast<uint64_t>(tiling_->outputRowBytes) + outputPrefix_;
        const bool directOutput = logicalBytes == alignedGroup &&
            logicalBytes != 0U &&
            (outputAddress % DATA_BLOCK_BYTES) == 0U &&
            (tiling_->outputRowBytes % DATA_BLOCK_BYTES) == 0U &&
            logicalBytes / DATA_BLOCK_BYTES <= 0xffffU &&
            (slabStride_ - logicalBytes) / DATA_BLOCK_BYTES <= 0xffffU &&
            (tiling_->outputRowBytes - logicalBytes) /
                DATA_BLOCK_BYTES <= 0xffffU;
        if (directOutput) {
            const DataCopyParams directParams{
                rows,
                static_cast<uint16_t>(logicalBytes / DATA_BLOCK_BYTES),
                static_cast<uint16_t>((slabStride_ - logicalBytes) /
                                       DATA_BLOCK_BYTES),
                static_cast<uint16_t>(
                    (tiling_->outputRowBytes - logicalBytes) /
                        DATA_BLOCK_BYTES)};
            DataCopy(outputView, slab.ReinterpretCast<uint32_t>(),
                     directParams);
        } else {
            DataCopyPad(
                outputView, slab.ReinterpretCast<uint32_t>(), outputParams);
        }
        SetFlag<HardEvent::MTE3_V>(event);
        pending[band] = true;
    }

    __aicore__ inline bool ProcessArenaBatch(
        uint64_t startRow, uint16_t rows, bool slabPending[2],
        bool& arenaReadPending)
    {
        if (arenaReadPending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            arenaReadPending = false;
        }
        uint16_t firstRows = rows;
        if (rowsPerBatch_ >= 32U) {
            // Keep the two slab generations at a fixed, repeat-aligned
            // boundary, including the final partial batch. Front-load the
            // first band so the second band's reads/output have a short tail.
            const uint32_t firstCapacity =
                ((static_cast<uint32_t>(rowsPerBatch_) + 31U) / 32U) * 16U;
            firstRows = static_cast<uint16_t>(
                Min(static_cast<uint32_t>(rows), firstCapacity));
        }
        const uint16_t secondRows = rows - firstRows;
        LocalTensor<uint8_t> slab = slabBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> firstSlab = slab;
        LocalTensor<uint8_t> secondSlab =
            slab[static_cast<uint32_t>(firstRows) * slabStride_];
        LocalTensor<uint8_t> arena = arenaBuffer_.Get<uint8_t>();
        LocalTensor<uint64_t> unusedCachedPointers;

        uint32_t pendingMask = 0U;
        uint32_t packedWidths =
            tiling_->launchPadding[groupStart_ / 4U];
        uint32_t scan = groupStart_;
        uint32_t localPrefix = 0U;
        uint32_t gatherTable = 0U;
        uint32_t arenaOffset = 0U;
        uint32_t issueSlot = 0U;
        uint32_t queueSlot[4];
        uint32_t queuePrefix[4];
        uint32_t queueRowBytes[4];
        uint32_t queueStride[4];
        uint32_t queueGatherTable[4];
        uint32_t queueArenaOffset[4];
        uint32_t queueHead = 0U;
        uint32_t queueCount = 0U;
        bool firstSlabReady = !slabPending[0];

        while (scan < groupEnd_ || queueCount != 0U) {
            while (scan < groupEnd_ && queueCount < 4U) {
                uint32_t rowBytes = 0U;
                const uint32_t input = NextNonzero(
                    scan, groupEnd_, packedWidths, rowBytes);
                if (input == groupEnd_) {
                    scan = groupEnd_;
                    break;
                }
                const uint32_t slot = issueSlot;
                issueSlot = (issueSlot + 1U) & 3U;
                ConsumeScratchRelease(slot, pendingMask);
                const uint32_t tail = (queueHead + queueCount) & 3U;
                const uint32_t stride =
                    ArenaSourceStride(localPrefix, rowBytes);
                const uint32_t sliceBytes = ArenaSliceBytes(
                    rowsPerBatch_, localPrefix, rowBytes);
                if (sliceBytes == 0U ||
                    arenaOffset + sliceBytes > arenaBytes_) {
                    return false;
                }
                queueSlot[tail] = slot;
                queuePrefix[tail] = localPrefix;
                queueRowBytes[tail] = rowBytes;
                queueStride[tail] = stride;
                queueArenaOffset[tail] = arenaOffset;
                const bool usesGather =
                    UseCompactGather(rowBytes, localPrefix) ||
                    UseSegmentedGather(rowBytes, localPrefix);
                queueGatherTable[tail] = gatherTable;
                gatherTable += static_cast<uint32_t>(usesGather);
                LoadScratch(
                    input, startRow, rows, localPrefix, rowBytes,
                    arena[arenaOffset], ScratchEvent(slot),
                    unusedCachedPointers);
                arenaOffset += sliceBytes;
                ++queueCount;
                scan = input + 1U;
                localPrefix += rowBytes;
            }
            if (queueCount == 0U) {
                return false;
            }
            const uint32_t head = queueHead;
            const uint32_t slot = queueSlot[head];
            const event_t event = ScratchEvent(slot);
            WaitFlag<HardEvent::MTE2_V>(event);
            if (!firstSlabReady) {
                WaitFlag<HardEvent::MTE3_V>(ArenaSlabEvent(0U));
                slabPending[0] = false;
                firstSlabReady = true;
            }
            PackArenaSlice(
                firstSlab, arena[queueArenaOffset[head]], firstRows,
                queuePrefix[head], queueRowBytes[head],
                queueStride[head], queueGatherTable[head]);
            SetFlag<HardEvent::V_MTE2>(event);
            pendingMask |= 1U << slot;
            queueHead = (queueHead + 1U) & 3U;
            --queueCount;
        }
        if (gatherTable != gatherInputCount_ ||
            arenaOffset != arenaBytes_ ||
            AlignUp(localPrefix, DATA_BLOCK_BYTES) != slabStride_) {
            return false;
        }
        EmitArenaBand(startRow, firstRows, firstSlab, 0U, slabPending);
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            ConsumeScratchRelease(slot, pendingMask);
        }

        if (secondRows == 0U) {
            return true;
        }
        if (slabPending[1]) {
            WaitFlag<HardEvent::MTE3_V>(ArenaSlabEvent(1U));
            slabPending[1] = false;
        }
        packedWidths = tiling_->launchPadding[groupStart_ / 4U];
        localPrefix = 0U;
        gatherTable = 0U;
        arenaOffset = 0U;
        for (uint32_t input = groupStart_; input < groupEnd_; ++input) {
            const uint32_t rowBytes = RowBytes(input, packedWidths);
            if (rowBytes == 0U) {
                continue;
            }
            const uint32_t stride =
                ArenaSourceStride(localPrefix, rowBytes);
            const uint32_t sliceBytes = ArenaSliceBytes(
                rowsPerBatch_, localPrefix, rowBytes);
            if (sliceBytes == 0U ||
                arenaOffset + sliceBytes > arenaBytes_) {
                return false;
            }
            const bool usesGather =
                UseCompactGather(rowBytes, localPrefix) ||
                UseSegmentedGather(rowBytes, localPrefix);
            PackArenaSlice(
                secondSlab,
                arena[arenaOffset +
                    static_cast<uint32_t>(firstRows) * stride],
                secondRows, localPrefix, rowBytes, stride, gatherTable);
            gatherTable += static_cast<uint32_t>(usesGather);
            arenaOffset += sliceBytes;
            localPrefix += rowBytes;
        }
        if (gatherTable != gatherInputCount_ ||
            arenaOffset != arenaBytes_ ||
            AlignUp(localPrefix, DATA_BLOCK_BYTES) != slabStride_) {
            return false;
        }
        // Retire every second-band Vector read before publishing arena
        // ownership.  The barrier alone does not constrain MTE2, while the
        // cross-pipe event alone can be scheduled ahead of independent Vector
        // work.  Together they form the complete V -> V-barrier -> MTE2
        // dependency required before the next batch overwrites this arena.
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        arenaReadPending = true;
        EmitArenaBand(
            startRow + firstRows, secondRows, secondSlab, 1U,
            slabPending);
        return true;
    }

    __aicore__ inline void ProcessArenaRows(
        uint64_t row, uint64_t endRow)
    {
        bool slabPending[2] = {false, false};
        bool arenaReadPending = false;
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row, static_cast<uint64_t>(rowsPerBatch_)));
            if (!ProcessArenaBatch(
                    row, rows, slabPending, arenaReadPending)) {
                break;
            }
            row += rows;
        }
        if (arenaReadPending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        for (uint32_t band = 0U; band < 2U; ++band) {
            if (slabPending[band]) {
                WaitFlag<HardEvent::MTE3_V>(ArenaSlabEvent(band));
            }
        }
    }

    __aicore__ inline uint32_t SelectScratch(
        uint32_t freeMask, uint32_t requirement) const
    {
        uint32_t selected = 4U;
        uint32_t selectedCapacity = 0xffffffffU;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            const uint32_t capacity = ScratchCapacity(slot);
            if ((freeMask & (1U << slot)) != 0U &&
                capacity >= requirement && capacity < selectedCapacity) {
                selected = slot;
                selectedCapacity = capacity;
            }
        }
        return selected;
    }

    __aicore__ inline void ConsumeScratchRelease(
        uint32_t slot, uint32_t& pendingMask) const
    {
        const uint32_t bit = 1U << slot;
        if ((pendingMask & bit) != 0U) {
            WaitFlag<HardEvent::V_MTE2>(ScratchEvent(slot));
            pendingMask &= ~bit;
        }
    }

    __aicore__ inline bool ProcessRowBatch(
        uint64_t startRow, uint16_t rows, bool& slabPending)
    {
        LocalTensor<uint8_t> slab = slabBuffer_.Get<uint8_t>();
        LocalTensor<uint64_t> unusedCachedPointers;
        uint32_t pendingMask = 0U;
        uint32_t packedWidths =
            tiling_->launchPadding[groupStart_ / 4U];
        uint32_t scan = groupStart_;
        uint32_t localPrefix = 0U;
        uint32_t gatherTable = 0U;
        uint32_t freeMask = 0xfU;
        uint32_t queueSlot[4] = {};
        uint32_t queuePrefix[4] = {};
        uint32_t queueRowBytes[4] = {};
        uint32_t queueStride[4] = {};
        uint32_t queueGatherTable[4] = {};
        uint32_t queueHead = 0U;
        uint32_t queueCount = 0U;
        bool slabReady = !slabPending;

        while (scan < groupEnd_ || queueCount != 0U) {
            while (scan < groupEnd_ && queueCount < 4U) {
                uint32_t rowBytes = 0U;
                const uint32_t input = NextNonzero(
                    scan, groupEnd_, packedWidths, rowBytes);
                if (input == groupEnd_) {
                    scan = groupEnd_;
                    break;
                }
                const uint32_t slot = SelectScratch(
                    freeMask, ScratchRequirement(localPrefix, rowBytes));
                if (slot == 4U) {
                    break;
                }
                ConsumeScratchRelease(slot, pendingMask);
                const event_t event = ScratchEvent(slot);
                LocalTensor<uint8_t> scratch = ScratchTensor(slot);
                const uint32_t tail = (queueHead + queueCount) & 3U;
                queueSlot[tail] = slot;
                queuePrefix[tail] = localPrefix;
                queueRowBytes[tail] = rowBytes;
                const bool usesGather =
                    UseCompactGather(rowBytes, localPrefix) ||
                    UseSegmentedGather(rowBytes, localPrefix);
                queueGatherTable[tail] = gatherTable;
                gatherTable += static_cast<uint32_t>(usesGather);
                queueStride[tail] = LoadScratch(
                    input, startRow, rows, localPrefix, rowBytes,
                    scratch, event, unusedCachedPointers);
                freeMask &= ~(1U << slot);
                ++queueCount;
                scan = input + 1U;
                localPrefix += rowBytes;
            }
            if (queueCount == 0U) {
                return false;
            }
            const uint32_t head = queueHead;
            const uint32_t slot = queueSlot[head];
            const event_t event = ScratchEvent(slot);
            LocalTensor<uint8_t> scratch = ScratchTensor(slot);
            WaitFlag<HardEvent::MTE2_V>(event);
            if (!slabReady) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                slabPending = false;
                slabReady = true;
            }
            if (UseCompactGather(
                    queueRowBytes[head], queuePrefix[head])) {
                if (queueGatherTable[head] >= gatherInputCount_) {
                    return false;
                }
                LocalTensor<uint8_t> gatherOffsets =
                    gatherOffsetBuffer_.Get<uint8_t>()[
                        queueGatherTable[head] *
                            DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES];
                CompactGatherToSlab(
                    slab, scratch, gatherOffsets, rows,
                    queuePrefix[head], queueRowBytes[head], 0U,
                    nullptr, ScratchCapacity(slot), true);
            } else if (UseSegmentedGather(
                           queueRowBytes[head], queuePrefix[head])) {
                if (queueGatherTable[head] >= gatherInputCount_) {
                    return false;
                }
                LocalTensor<uint8_t> gatherOffsets =
                    gatherOffsetBuffer_.Get<uint8_t>()[
                        queueGatherTable[head] *
                            DYNAMIC_SLAB_GROUP_OWNER_OFFSET_BYTES];
                SegmentedGatherToSlab(
                    slab, scratch, gatherOffsets, rows,
                    queuePrefix[head], queueRowBytes[head], 0U,
                    nullptr, ScratchCapacity(slot), true);
            } else {
                VectorPack(
                    slab, scratch, rows, queuePrefix[head],
                    queueRowBytes[head], queueStride[head]);
            }
            SetFlag<HardEvent::V_MTE2>(event);
            pendingMask |= 1U << slot;
            freeMask |= 1U << slot;
            queueHead = (queueHead + 1U) & 3U;
            --queueCount;
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint32_t alignedGroup =
            AlignUp(localPrefix, DATA_BLOCK_BYTES);
        const DataCopyExtParams outputParams{
            rows, localPrefix,
            (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
            static_cast<uint32_t>(
                tiling_->outputRowBytes - localPrefix), 0U};
        GlobalTensor<uint32_t> outputView;
        outputView.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            outputGm_.GetPhyAddr(
                startRow * tiling_->outputRowBytes + outputPrefix_)));
        // When the runtime-computed group width and both row bases are
        // block aligned, the slab already contains exactly the bytes that
        // belong to this group.  Use the ordinary 2-D DMA in that case so
        // MTE3 does not pay the padding descriptor path.  Unaligned group
        // ownership keeps DataCopyPad because neighbouring inputs may share
        // the edge block and padding is semantically required.
        const uint64_t outputAddress = startRow *
            static_cast<uint64_t>(tiling_->outputRowBytes) + outputPrefix_;
        const bool directOutput = localPrefix == alignedGroup &&
            localPrefix != 0U &&
            (outputAddress % DATA_BLOCK_BYTES) == 0U &&
            (tiling_->outputRowBytes % DATA_BLOCK_BYTES) == 0U &&
            localPrefix / DATA_BLOCK_BYTES <= 0xffffU &&
            (slabStride_ - localPrefix) / DATA_BLOCK_BYTES <= 0xffffU &&
            (tiling_->outputRowBytes - localPrefix) / DATA_BLOCK_BYTES <= 0xffffU;
        if (directOutput) {
            const DataCopyParams directParams{
                rows,
                static_cast<uint16_t>(localPrefix / DATA_BLOCK_BYTES),
                static_cast<uint16_t>((slabStride_ - localPrefix) /
                                       DATA_BLOCK_BYTES),
                static_cast<uint16_t>((tiling_->outputRowBytes - localPrefix) /
                                       DATA_BLOCK_BYTES)};
            DataCopy(outputView, slab.ReinterpretCast<uint32_t>(), directParams);
        } else {
            DataCopyPad(
                outputView, slab.ReinterpretCast<uint32_t>(), outputParams);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        slabPending = true;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            ConsumeScratchRelease(slot, pendingMask);
        }
        return gatherTable == gatherInputCount_ &&
            localPrefix + outputPrefix_ <= tiling_->outputRowBytes;
    }

    TBuf<TPosition::VECCALC> scratchBuffer2_;
    TBuf<TPosition::VECCALC> scratchBuffer3_;
    TBuf<TPosition::VECCALC> arenaBuffer_;
    uint64_t outputPrefix_ = 0U;
    uint32_t availableUbBytes_ = 0U;
    uint32_t fixedUbBytes_ = 0U;
    uint32_t arenaBytes_ = 0U;
    uint32_t groupBytes_ = 0U;
    uint32_t rowsPerBatch_ = 0U;
    uint32_t gatherInputCount_ = 0U;
    uint32_t offsetTableCount_ = 1U;
    uint32_t selectedGroup_ = 0U;
    uint32_t groupStart_ = 0U;
    uint32_t groupEnd_ = 0U;
    uint32_t shardIndex_ = 0U;
    uint32_t shardCount_ = 0U;
    uint32_t arenaPairFirst_ = 0xfU;
    uint32_t arenaPairSecond_ = 0xfU;
    bool arenaEnabled_ = false;
    bool arenaHasSharedPair_ = false;
    bool valid_ = false;
};

// Key 64 removes key 63's duplicate full-row scratch load.  Complete,
// output-aligned cells of wide rows move directly from GM to one of two
// alternating slabs.  Short fragmented rows use one compact 2D MTE2 transfer
// and one Gather; wide rows isolate only their two partial cells.  The second
// slab lets MTE2 assemble group g+1 while MTE3 writes group g.
class ConcatDynamicDoubleSlabBoundaryKernel :
    public ConcatDynamicSlabKernel<true, true, false, 8U, 1U,
                                   true, true, true> {
    using Base =
        ConcatDynamicSlabKernel<true, true, false, 8U, 1U,
                                true, true, true>;

protected:
    using Base::AlignUp;
    using Base::gatherBaseBuffer_;
    using Base::gatherOffsetBuffer_;
    using Base::GenerateGatherOffsetsWavefront;
    using Base::groupCount_;
    using Base::GroupEnd;
    using Base::InitializeGatherOffsetBase;
    using Base::inputDataPtrs_;
    using Base::MakeMask;
    using Base::metadataVersion_;
    using Base::Min;
    using Base::NextNonzero;
    using Base::outputGm_;
    using Base::pipe_;
    using Base::scratchBuffer0_;
    using Base::scratchBuffer1_;
    using Base::scratchStride_;
    using Base::scratchStride1_;
    using Base::slabBuffer_;
    using Base::slabStride_;
    using Base::tiling_;

public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatTilingData* tilingData)
    {
        Base::Init(inputs, output, tilingData);
        pipe_.InitBuffer(
            slabBuffer1_, tilingData->rowsPerBatch * slabStride_);
    }

    __aicore__ inline void Process()
    {
        if (metadataVersion_ != DYNAMIC_SLAB_METADATA_VERSION ||
            groupCount_ == 0U || groupCount_ >= 32U ||
            slabStride_ == 0U ||
            scratchStride_ != DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES ||
            scratchStride1_ != DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES ||
            tiling_->rowsPerBatch == 0U) {
            return;
        }
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockCount = static_cast<uint64_t>(GetBlockNum());
        const uint32_t scheduleWord = tiling_->launchPadding[
            PACKED_WIDTH_WORDS + static_cast<uint32_t>(blockIndex) / 4U];
        const uint32_t scheduleByte =
            (scheduleWord >> ((blockIndex % 4U) * 8U)) & 0xffU;
        const uint64_t rowsPerBlock = tiling_->outerRows / blockCount;
        uint64_t row = blockIndex * rowsPerBlock + (scheduleByte & 0x3fU);
        const uint64_t endRow = row + rowsPerBlock +
            ((scheduleByte >> 6U) & 1U);
        if (row >= tiling_->outerRows) {
            return;
        }

        InitializeGatherOffsetBase();
        while (row < endRow) {
            const uint16_t rows = static_cast<uint16_t>(Min(
                endRow - row,
                static_cast<uint64_t>(tiling_->rowsPerBatch)));
            ProcessRowBatch(row, rows);
            row += rows;
        }
    }

private:
    __aicore__ inline event_t SlabEvent(uint32_t slot) const
    {
        return slot == 0U ? EVENT_ID0 : EVENT_ID5;
    }

    __aicore__ inline void BoundaryLayout(
        uint32_t localPrefix, uint32_t rowBytes,
        uint32_t& headBytes, uint32_t& interiorStart,
        uint32_t& interiorBytes, uint32_t& tailStart,
        uint32_t& tailBytes) const
    {
        const uint32_t absoluteEnd = localPrefix + rowBytes;
        const uint32_t absoluteInteriorStart =
            AlignUp(localPrefix, DATA_BLOCK_BYTES);
        const uint32_t absoluteInteriorEnd =
            absoluteEnd / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        if (absoluteInteriorStart < absoluteInteriorEnd) {
            headBytes = absoluteInteriorStart - localPrefix;
            interiorStart = headBytes;
            interiorBytes =
                absoluteInteriorEnd - absoluteInteriorStart;
            tailStart = interiorStart + interiorBytes;
            tailBytes = rowBytes - tailStart;
            return;
        }

        const uint32_t firstLineEnd =
            (localPrefix / DATA_BLOCK_BYTES + 1U) * DATA_BLOCK_BYTES;
        const uint32_t firstLineBytes = firstLineEnd - localPrefix;
        headBytes = rowBytes < firstLineBytes ? rowBytes : firstLineBytes;
        interiorStart = headBytes;
        interiorBytes = 0U;
        tailStart = headBytes;
        tailBytes = rowBytes - headBytes;
    }

    __aicore__ inline bool UseCompactWholeRow(
        uint32_t localPrefix, uint32_t rowBytes) const
    {
        const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;
        return leftBytes + rowBytes <= DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES &&
            (localPrefix % DATA_BLOCK_BYTES != 0U ||
             rowBytes % DATA_BLOCK_BYTES != 0U);
    }

    __aicore__ inline void LoadBoundaryAndInterior(
        uint32_t input, uint64_t startRow, uint16_t rows,
        uint32_t localPrefix, uint32_t rowBytes,
        LocalTensor<uint8_t> slab, LocalTensor<uint8_t> scratch,
        event_t event)
    {
        uint32_t headBytes = 0U;
        uint32_t interiorStart = 0U;
        uint32_t interiorBytes = 0U;
        uint32_t tailStart = 0U;
        uint32_t tailBytes = 0U;
        BoundaryLayout(localPrefix, rowBytes, headBytes, interiorStart,
                       interiorBytes, tailStart, tailBytes);

        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
            inputDataPtrs_[input]));
        const uint64_t sourceOffset = startRow * rowBytes;
        const DataCopyPadExtParams<uint8_t> noPadding{
            false, 0U, 0U, 0U};
        const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;

        if (UseCompactWholeRow(localPrefix, rowBytes)) {
            const uint32_t alignedRowBytes =
                AlignUp(leftBytes + rowBytes, DATA_BLOCK_BYTES);
            const DataCopyExtParams compactParams{
                rows, rowBytes, 0U,
                (DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES - alignedRowBytes) /
                    DATA_BLOCK_BYTES,
                0U};
            const DataCopyPadExtParams<uint8_t> compactPadding{
                leftBytes != 0U, static_cast<uint8_t>(leftBytes),
                0U, 0U};
            DataCopyPad(scratch, inputGm[sourceOffset],
                        compactParams, compactPadding);
            SetFlag<HardEvent::MTE2_V>(event);
            return;
        }

        if (headBytes != 0U) {
            const DataCopyExtParams headParams{
                rows, headBytes, rowBytes - headBytes,
                (DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES - DATA_BLOCK_BYTES) /
                    DATA_BLOCK_BYTES,
                0U};
            const DataCopyPadExtParams<uint8_t> headPadding{
                leftBytes != 0U, static_cast<uint8_t>(leftBytes),
                0U, 0U};
            DataCopyPad(scratch, inputGm[sourceOffset],
                        headParams, headPadding);
        }
        if (tailBytes != 0U) {
            const DataCopyExtParams tailParams{
                rows, tailBytes, rowBytes - tailBytes,
                (DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES - DATA_BLOCK_BYTES) /
                    DATA_BLOCK_BYTES,
                0U};
            DataCopyPad(
                scratch[DATA_BLOCK_BYTES],
                inputGm[sourceOffset + tailStart],
                tailParams, noPadding);
        }
        if (interiorBytes != 0U) {
            const DataCopyExtParams interiorParams{
                rows, interiorBytes, rowBytes - interiorBytes,
                (slabStride_ - interiorBytes) / DATA_BLOCK_BYTES, 0U};
            DataCopyPad(
                slab[localPrefix + interiorStart],
                inputGm[sourceOffset + interiorStart],
                interiorParams, noPadding);
        }
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void GatherBoundaryRows(
        LocalTensor<uint32_t> destination,
        LocalTensor<uint32_t> source,
        LocalTensor<uint32_t> offsets,
        uint16_t rows, const uint64_t mask[2])
    {
        const uint16_t destinationRepeatStride =
            static_cast<uint16_t>(slabStride_ / DATA_BLOCK_BYTES);
        uint32_t row = 0U;
        for (; row + 8U <= rows; row += 8U) {
            Gather<uint32_t>(
                destination[row * slabStride_ / sizeof(uint32_t)],
                source, offsets,
                row * DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES,
                mask, 8U, destinationRepeatStride);
        }
        const uint32_t tailRows = static_cast<uint32_t>(rows) - row;
        if (tailRows != 0U) {
            Gather<uint32_t>(
                destination[row * slabStride_ / sizeof(uint32_t)],
                source, offsets,
                row * DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES,
                mask, static_cast<uint8_t>(tailRows),
                destinationRepeatStride);
        }
    }

    __aicore__ inline void RestoreBoundaries(
        LocalTensor<uint8_t> slabBytes,
        LocalTensor<uint8_t> scratchBytes,
        LocalTensor<uint8_t> offsetBytes,
        uint16_t rows, uint32_t localPrefix, uint32_t rowBytes)
    {
        uint32_t headBytes = 0U;
        uint32_t interiorStart = 0U;
        uint32_t interiorBytes = 0U;
        uint32_t tailStart = 0U;
        uint32_t tailBytes = 0U;
        BoundaryLayout(localPrefix, rowBytes, headBytes, interiorStart,
                       interiorBytes, tailStart, tailBytes);
        if (UseCompactWholeRow(localPrefix, rowBytes)) {
            const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;
            LocalTensor<int32_t> offsetValues =
                offsetBytes.ReinterpretCast<int32_t>();
            LocalTensor<uint32_t> offsets =
                offsetBytes.ReinterpretCast<uint32_t>();
            const uint32_t activeLanes =
                (leftBytes + rowBytes) / sizeof(uint32_t);
            GenerateGatherOffsetsWavefront(
                offsetValues, DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES,
                0U, activeLanes);
            uint64_t compactMask[2];
            MakeMask(compactMask, leftBytes / sizeof(uint32_t),
                     rowBytes / sizeof(uint32_t));
            GatherBoundaryRows(
                slabBytes[localPrefix - leftBytes]
                    .ReinterpretCast<uint32_t>(),
                scratchBytes.ReinterpretCast<uint32_t>(),
                offsets, rows, compactMask);
            return;
        }
        if (headBytes == 0U && tailBytes == 0U) {
            return;
        }

        const uint32_t leftBytes = localPrefix % DATA_BLOCK_BYTES;
        LocalTensor<int32_t> offsetValues =
            offsetBytes.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets =
            offsetBytes.ReinterpretCast<uint32_t>();
        GenerateGatherOffsetsWavefront(
            offsetValues, DYNAMIC_BOUNDARY_SCRATCH_ROW_BYTES,
            0U, DATA_BLOCK_BYTES / sizeof(uint32_t));

        if (headBytes != 0U) {
            uint64_t headMask[2];
            MakeMask(headMask, leftBytes / sizeof(uint32_t),
                     headBytes / sizeof(uint32_t));
            GatherBoundaryRows(
                slabBytes[localPrefix - leftBytes]
                    .ReinterpretCast<uint32_t>(),
                scratchBytes.ReinterpretCast<uint32_t>(),
                offsets, rows, headMask);
        }
        if (tailBytes != 0U) {
            uint64_t tailMask[2];
            MakeMask(tailMask, 0U, tailBytes / sizeof(uint32_t));
            GatherBoundaryRows(
                slabBytes[localPrefix + tailStart]
                    .ReinterpretCast<uint32_t>(),
                scratchBytes[DATA_BLOCK_BYTES]
                    .ReinterpretCast<uint32_t>(),
                offsets, rows, tailMask);
        }
    }

    __aicore__ inline void ProcessRowBatch(
        uint64_t startRow, uint16_t rows)
    {
        LocalTensor<uint8_t> slabs[2] = {
            slabBuffer_.Get<uint8_t>(), slabBuffer1_.Get<uint8_t>()};
        LocalTensor<uint8_t> scratches[2] = {
            scratchBuffer0_.Get<uint8_t>(),
            scratchBuffer1_.Get<uint8_t>()};
        LocalTensor<uint8_t> gatherOffsets =
            gatherOffsetBuffer_.Get<uint8_t>();
        bool scratchPending[2] = {false, false};
        bool slabPending[2] = {false, false};
        uint32_t groupStart = 0U;
        uint64_t outputPrefix = 0U;

        for (uint32_t group = 0U; group < groupCount_; ++group) {
            const uint32_t groupEnd = GroupEnd(group);
            if (groupStart >= tiling_->inputNum ||
                groupEnd <= groupStart || groupEnd > tiling_->inputNum) {
                break;
            }
            const uint32_t slabSlot = group & 1U;
            const event_t slabEvent = SlabEvent(slabSlot);
            LocalTensor<uint8_t> slab = slabs[slabSlot];
            if (slabPending[slabSlot]) {
                WaitFlag<HardEvent::MTE3_MTE2>(slabEvent);
                WaitFlag<HardEvent::MTE3_V>(slabEvent);
                slabPending[slabSlot] = false;
            }

            uint32_t packedWidths =
                tiling_->launchPadding[groupStart / 4U];
            uint32_t rowBytes = 0U;
            uint32_t current = NextNonzero(
                groupStart, groupEnd, packedWidths, rowBytes);
            if (current == groupEnd) {
                groupStart = groupEnd;
                continue;
            }

            uint32_t currentSlot = 0U;
            if (scratchPending[currentSlot]) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                scratchPending[currentSlot] = false;
            }
            LoadBoundaryAndInterior(
                current, startRow, rows, 0U, rowBytes,
                slab, scratches[currentSlot], EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

            uint32_t localPrefix = 0U;
            while (true) {
                const uint32_t nextPrefix = localPrefix + rowBytes;
                uint32_t nextRowBytes = 0U;
                uint32_t nextPackedWidths = packedWidths;
                const uint32_t next = NextNonzero(
                    current + 1U, groupEnd,
                    nextPackedWidths, nextRowBytes);
                const uint32_t nextSlot = currentSlot ^ 1U;
                const event_t nextEvent =
                    nextSlot == 0U ? EVENT_ID1 : EVENT_ID2;
                if (next < groupEnd) {
                    if (scratchPending[nextSlot]) {
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        scratchPending[nextSlot] = false;
                    }
                    LoadBoundaryAndInterior(
                        next, startRow, rows, nextPrefix,
                        nextRowBytes, slab, scratches[nextSlot], nextEvent);
                }

                RestoreBoundaries(
                    slab, scratches[currentSlot], gatherOffsets,
                    rows, localPrefix, rowBytes);
                const event_t currentEvent =
                    currentSlot == 0U ? EVENT_ID1 : EVENT_ID2;
                SetFlag<HardEvent::V_MTE2>(currentEvent);
                scratchPending[currentSlot] = true;
                localPrefix = nextPrefix;
                if (next == groupEnd) {
                    break;
                }
                WaitFlag<HardEvent::MTE2_V>(nextEvent);
                current = next;
                currentSlot = nextSlot;
                rowBytes = nextRowBytes;
                packedWidths = nextPackedWidths;
            }

            SetFlag<HardEvent::V_MTE3>(slabEvent);
            WaitFlag<HardEvent::V_MTE3>(slabEvent);
            const uint32_t alignedGroup =
                AlignUp(localPrefix, DATA_BLOCK_BYTES);
            const DataCopyExtParams outputParams{
                rows, localPrefix,
                (slabStride_ - alignedGroup) / DATA_BLOCK_BYTES,
                static_cast<uint32_t>(
                    tiling_->outputRowBytes - localPrefix), 0U};
            DataCopyPad(
                outputGm_[startRow * tiling_->outputRowBytes + outputPrefix],
                slab, outputParams);
            SetFlag<HardEvent::MTE3_MTE2>(slabEvent);
            SetFlag<HardEvent::MTE3_V>(slabEvent);
            slabPending[slabSlot] = true;
            outputPrefix += localPrefix;
            groupStart = groupEnd;
        }

        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slabPending[slot]) {
                const event_t event = SlabEvent(slot);
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                WaitFlag<HardEvent::MTE3_V>(event);
            }
        }
        if (scratchPending[0]) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        if (scratchPending[1]) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
    }

    TBuf<TPosition::VECCALC> slabBuffer1_;
};

}  // namespace

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
    if (TILING_KEY_IS(15U)) {
        ConcatExactRawUbKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(5U)) {
        ConcatCopyKernel<5U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(4U)) {
        ConcatCopyKernel<4U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(3U)) {
        ConcatCopyKernel<3U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(2U)) {
        ConcatCopyKernel<2U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(16U)) {
        ConcatCopyKernel<0U, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(17U)) {
        ConcatCopyKernel<0U, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(25U)) {
        ConcatCopyKernel<0U, false, false, false, true, false, false, false,
                         false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(18U)) {
        ConcatCopyKernel<2U, false, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(19U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(26U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, true, false,
                         false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(20U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(24U)) {
        ConcatSegmentedNarrowKernel<8U, 64U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(29U)) {
        ConcatSegmentedNarrowKernel<16U, 32U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(30U)) {
        ConcatSegmentedAligned32Kernel<> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(31U)) {
        ConcatSegmentedAligned32Kernel<true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(32U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(34U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(37U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, true, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(35U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, true, false, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(64U)) {
        ConcatDynamicDoubleSlabBoundaryKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(60U)) {
        ConcatDynamicSlabAdaptiveKernel<16U, 1U, true, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(59U)) {
        ConcatDynamicSlabAdaptiveKernel<16U, 1U, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(58U)) {
        ConcatDynamicSlabAdaptiveKernel<16U, 1U, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(57U)) {
        ConcatDynamicSlabAdaptiveKernel<16U, 2U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(56U)) {
        ConcatDynamicSlabAdaptiveKernel<16U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(49U)) {
        ConcatDynamicSlabAdaptiveKernel<8U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(92U)) {
        ConcatDynamicSlabGroupOwnerKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(79U)) {
        ConcatDynamicSlabAdaptiveKernel<
            16U, 1U, true, true, true, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(69U)) {
        ConcatDynamicSlabAdaptiveKernel<16U, 1U, true, true, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(68U)) {
        ConcatDynamicSlabTripleKernel<16U, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(67U)) {
        ConcatDynamicSlabTripleKernel<16U, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(48U)) {
        ConcatDynamicSlabTripleKernel<> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(47U)) {
        ConcatDynamicSlabKernel<true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(46U)) {
        ConcatDynamicSlabKernel<true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(44U)) {
        ConcatDynamicSlabKernel<> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(43U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, true, false, true, true, true,
                         true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(39U)) {
        ConcatCopyKernel<3U, false, false, false, false, false, false, false,
                         false, false, false, true, false, true, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(128U)) {
        ConcatWideRowAssemblyKernel<false> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(130U)) {
        ConcatWideRowAssemblyKernel<true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(91U)) {
        ConcatDedicatedDirectKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(40U)) {
        ConcatCopyKernel<3U, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(28U)) {
        ConcatSegmentedB32Kernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(23U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, false, true,
                         false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(21U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, false, false, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(22U)) {
        ConcatCopyKernel<0U, false, false, false, false, false, false, false, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(78U)) {
        ConcatIncrementalFlatPipelineKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(33U)) {
        ConcatFlatPipelineKernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(0U)) {
        ConcatCopyKernel<0U> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    }
}
