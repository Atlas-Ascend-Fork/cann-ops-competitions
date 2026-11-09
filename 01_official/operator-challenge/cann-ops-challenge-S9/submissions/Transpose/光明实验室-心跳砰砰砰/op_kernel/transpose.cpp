#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t kScheduleGeneral2D = 0;
constexpr int64_t kScheduleAligned2D = 1;
constexpr int64_t kScheduleCopy2D = 2;
constexpr int64_t kScheduleRank3Perm120 = 12;
constexpr int64_t kScheduleRank4Perm0231 = 13;
constexpr int64_t kScheduleRank4Perm0312 = 14;
constexpr int64_t kScheduleByteTile2D = 15;
constexpr int64_t kScheduleTileGrid2D = 16;
constexpr int64_t kScheduleRank4Perm0231Wide = 17;
constexpr int64_t kScheduleRank4Perm0312Wide = 18;
constexpr int64_t kScheduleTileGrid32x32 = 19;
constexpr int64_t kScheduleTileGrid2DPad = 20;
constexpr int64_t kScheduleGenericNd = 21;
constexpr int64_t kSchedulePlaneNd = 22;
constexpr int64_t kScheduleReverseNd = 23;
constexpr int64_t kSchedulePlaneCompactB16 = 24;
constexpr int64_t kSchedulePlaneB32 = 25;
constexpr int64_t kSchedulePlaneB32Wide = 26;
constexpr int64_t kSchedulePlaneB32Grouped = 27;
constexpr int64_t kSchedulePlaneB32WideTail = 28;
constexpr int64_t kScheduleGenericRowsReordered = 29;
constexpr int64_t kSchedulePlaneB32FullRows = 30;
constexpr int64_t kScheduleRank5GroupedB32 = 31;
constexpr int64_t kAlignedTileRows = 16;
constexpr int64_t kAlignedTileCols = 32;
constexpr int64_t kWideTileRows = 64;
constexpr int64_t kWideTileCols = 32;
constexpr int64_t kGrid32TileRows = 32;
constexpr int64_t kGrid32TileCols = 32;
constexpr int64_t kByteTileRows = 32;
constexpr int64_t kByteTileCols = 64;
constexpr int64_t kMediumPadTileCols = 64;
constexpr int64_t kLargePadTileColsB16 = 128;
constexpr int64_t kLargePadTileColsB32 = 184;
constexpr int64_t kWidePadThreshold = 2048;
constexpr int64_t kLargePadElements = 80 * 1024 * 1024;
constexpr int64_t kGenericTileRows = 32;
constexpr int64_t kGenericNdTileElems = 1024;
constexpr int64_t kGenericReorderedGroup = 20;
constexpr int64_t kGenericReorderedFusedGroup = 10;
constexpr int64_t kDataBlockBytes = 32;
constexpr int64_t kPlaneNdTileCols = 512;
constexpr int64_t kPlaneNdTileElemsB16 = 16 * 1024;
constexpr int64_t kPlaneNdTileElemsB32 = 10 * 1024;
constexpr int64_t kPlaneB32WideTileRows = 48;
constexpr int64_t kPlaneB32WideTileCols = 288;
constexpr int64_t kPlaneB32WideTailRows = 48;
constexpr int64_t kPlaneB32WideTailCols = 336;
constexpr int64_t kPlaneB32GroupedTileCols = 264;
constexpr int64_t kPlaneB32GroupedPlanes = 4;
constexpr int64_t kRank5InnerTile = 6;
constexpr int64_t kRank5OuterGroup = 6;
constexpr int64_t kCompactFullRows = 64;
constexpr int64_t kCompactTileCols = 80;
constexpr int64_t kCompactInputGuard = 32;
constexpr int64_t kPlaneB32FullTileElements = 48 * 336 * 3;
constexpr int64_t kPlaneNdTileElemsB8 = 16 * 1024;
constexpr int64_t kPlaneCompactTileElemsB16 = 21 * 1024;
constexpr int64_t kPlaneCompactGroupCols = 4;
constexpr int64_t kReverseIndexChunk = 256;
constexpr int64_t kGenericNdInputBytes = 32 * 1024;
constexpr int64_t kFullTripleBufferMinCols = 640;
constexpr int64_t kFullTripleBufferMaxCols = 800;
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kFullOutputBufferNum = 3;
constexpr uint32_t kGenericRowsBufferNum = 4;

__aicore__ inline void TransposeB32Blocks(uint64_t dst[16], uint64_t src[16],
                                         uint8_t repeats, uint16_t dstStride)
{

    TransDataTo5HD<uint32_t>(dst, src, {false, false, repeats,
        static_cast<uint16_t>(repeats == 1 ? 0 : dstStride),
        static_cast<uint16_t>(repeats == 1 ? 0 : 1)});
}

struct TileInfo {
    int64_t rowBase;
    int64_t colBase;
    int64_t rows;
    int64_t cols;
};

struct PlaneConfig {
    int64_t transposedDim;
    int64_t contiguousDim;
    int64_t contiguousOutputAxis;
    int64_t inputRowStride;
    int64_t outputRowStride;
    int64_t tileCols;
    int64_t localTileCols;
    int64_t rowTiles;
    int64_t colTiles;
    int64_t tileLength;
    bool compactInputRows;
};

struct PlaneTileInfo {
    int64_t inputBase;
    int64_t outputBase;
    int64_t rows;
    int64_t cols;
};

struct GroupedPlaneTileInfo {
    int64_t basePlane;
    int64_t colBase;
    int64_t cols;
    int64_t planes;
    int64_t inputBase;
    int64_t outputBase;
};

struct GenericRowTileInfo {
    int64_t inputBase;
    int64_t outputBase;
    int64_t rows;
};

struct ReorderedRowTaskInfo {
    int64_t colBase;
    int64_t groupCols;
    int64_t batch;
};

struct Rank5TaskInfo {
    int64_t d3Base;
    int64_t d0Base;
    int64_t d1;
    int64_t group;
    int64_t cols;
};

template <typename T>
class KernelTranspose {
public:
    __aicore__ inline KernelTranspose() {}

    template <bool FullRows = false>
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int64_t rows, int64_t cols, int64_t totalLength,
                                int64_t blockRows, int64_t smallCoreTasks, int64_t bigCoreCount, int64_t schedule)
    {
        inputPtr_ = input;
        outputPtr_ = output;
        rows_ = rows;
        cols_ = cols;
        blockRows_ = blockRows;
        smallCoreTasks_ = smallCoreTasks;
        bigCoreCount_ = bigCoreCount;
        schedule_ = schedule;
        totalLength_ = totalLength;
        if constexpr (!FullRows) {
            padTileCols_ = cols < kWidePadThreshold ? kAlignedTileCols :
                           totalLength < kLargePadElements ? kMediumPadTileCols :
                           sizeof(T) == 2 ? kLargePadTileColsB16 : kLargePadTileColsB32;
            totalRowTasks_ = InitTaskCount();
        }
        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalLength_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, totalLength_);
        if constexpr (FullRows) {
            InitPlaneB32FullRowsBuffers();
        } else if (schedule_ == kScheduleRank4Perm0231Wide) {
            pipe_.InitBuffer(tileQueue_, kBufferNum, kWideTileRows * kWideTileCols * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, kWideTileRows * kWideTileCols * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, kWideTileRows * kWideTileCols * sizeof(uint32_t));
            BuildTransposeOffsets(kWideTileRows, kWideTileCols);
        } else if (schedule_ == kScheduleRank4Perm0312Wide) {
            const int64_t tileLength = blockRows_ * kWideTileCols;
            pipe_.InitBuffer(tileQueue_, kBufferNum, tileLength * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, tileLength * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, tileLength * sizeof(uint32_t));
            BuildTransposeOffsets(blockRows_, kWideTileCols);
        } else if (schedule_ == kScheduleTileGrid32x32) {
            pipe_.InitBuffer(tileQueue_, kBufferNum, kGrid32TileRows * kGrid32TileCols * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, kGrid32TileRows * kGrid32TileCols * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, kGrid32TileRows * kGrid32TileCols * sizeof(uint32_t));
            BuildTransposeOffsets(kGrid32TileRows, kGrid32TileCols);
        } else if (schedule_ == kScheduleRank3Perm120) {
            const int64_t tileLength = blockRows_ * kAlignedTileCols;
            pipe_.InitBuffer(tileQueue_, kBufferNum, tileLength * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, tileLength * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, tileLength * sizeof(uint32_t));
            BuildTransposeOffsets(blockRows_, kAlignedTileCols);
        } else if (schedule_ == kScheduleAligned2D && sizeof(T) == 2 && blockRows_ >= 32) {
            pipe_.InitBuffer(tileQueue_, 1, rows_ * 32 * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 2, rows_ * 32 * sizeof(T));
        } else if (schedule_ == kScheduleAligned2D || schedule_ == kScheduleTileGrid2D ||
            schedule_ == kScheduleRank4Perm0231 || schedule_ == kScheduleRank4Perm0312) {
            pipe_.InitBuffer(tileQueue_, kBufferNum, kAlignedTileRows * kAlignedTileCols * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, kAlignedTileRows * kAlignedTileCols * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, kAlignedTileRows * kAlignedTileCols * sizeof(uint32_t));
            BuildTransposeOffsets(kAlignedTileRows, kAlignedTileCols);
        } else if (schedule_ == kScheduleTileGrid2DPad) {
            const int64_t tileLength = blockRows_ * padTileCols_;
            pipe_.InitBuffer(tileQueue_, kBufferNum, tileLength * sizeof(T));
            pipe_.InitBuffer(outputQueue_, sizeof(T) == 2 ? kBufferNum : 1, tileLength * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, (tileLength + blockRows_) * sizeof(uint32_t));
            BuildTransposeOffsets(blockRows_, padTileCols_);
        } else if (schedule_ == kSchedulePlaneB32FullRows) {
            InitPlaneB32FullRowsBuffers();
        } else if (schedule_ == kSchedulePlaneB32Grouped) {
            const int64_t tileElements =
                (blockRows_ + 15) / 16 * 16 * kPlaneB32GroupedTileCols;
            pipe_.InitBuffer(tileQueue_, 1, tileElements * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 1, tileElements * sizeof(T));
        } else if (schedule_ == kScheduleRank5GroupedB32) {
            const int64_t localCols = (kRank5InnerTile * cols_ + 7) / 8 * 8;
            const int64_t outputRows = (kRank5OuterGroup * rows_ + 15) / 16 * 16;
            pipe_.InitBuffer(tileQueue_, 1, outputRows * localCols * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 1, outputRows * localCols * sizeof(T));
        } else if (schedule_ == kSchedulePlaneNd || schedule_ == kSchedulePlaneCompactB16 ||
                   schedule_ == kSchedulePlaneB32 || schedule_ == kSchedulePlaneB32Wide ||
                   schedule_ == kSchedulePlaneB32WideTail) {
            const int64_t tileElements = schedule_ == kSchedulePlaneCompactB16 ? kPlaneCompactTileElemsB16 :
                schedule_ == kSchedulePlaneB32WideTail ? kPlaneB32WideTailRows * kPlaneB32WideTailCols :
                schedule_ == kSchedulePlaneB32Wide ? kPlaneB32WideTileRows * kPlaneB32WideTileCols :
                sizeof(T) == 4 ? kPlaneNdTileElemsB32 :
                sizeof(T) == 1 ? kPlaneNdTileElemsB8 : kPlaneNdTileElemsB16;
            pipe_.InitBuffer(tileQueue_, schedule_ == kSchedulePlaneB32Wide ||
                             schedule_ == kSchedulePlaneB32WideTail ? 1 : kBufferNum,
                             tileElements * sizeof(T));
            pipe_.InitBuffer(outputQueue_, sizeof(T) == 2 || schedule_ == kSchedulePlaneB32 ||
                             schedule_ == kSchedulePlaneB32Wide || schedule_ == kSchedulePlaneB32WideTail ?
                                 kBufferNum : 1,
                             tileElements * (sizeof(T) == 1 ? 3 : sizeof(T)));
            const int64_t indexElements = schedule_ == kSchedulePlaneB32 ||
                schedule_ == kSchedulePlaneB32Wide || schedule_ == kSchedulePlaneB32WideTail ? 8 :
                schedule_ == kSchedulePlaneCompactB16 ? blockRows_ * kPlaneCompactGroupCols : tileElements;
            pipe_.InitBuffer(gatherIndexBuf_, indexElements * sizeof(uint32_t));
            if constexpr (sizeof(T) == 1) {
                pipe_.InitBuffer(genericCastBuf_, tileElements * sizeof(half));
            }
        } else if (schedule_ == kScheduleReverseNd) {
            const int64_t localC = (blockRows_ * sizeof(T) + kDataBlockBytes - 1) /
                                   kDataBlockBytes * kDataBlockBytes / sizeof(T);
            const int64_t gatherC = sizeof(T) == 2 ? 16 : 32;
            const int64_t inputElements = rows_ * cols_ * localC;
            const int64_t outputElements = blockRows_ * cols_ * rows_;
            const int64_t indexElements = (blockRows_ < gatherC ? blockRows_ : gatherC) * cols_ * rows_;
            pipe_.InitBuffer(tileQueue_, 1, inputElements * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 1, outputElements * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, indexElements * sizeof(uint32_t));
            pipe_.InitBuffer(reverseScratchBuf_, 3 * kReverseIndexChunk * sizeof(int32_t));
            BuildReverseOffsets(rows_, cols_, localC, blockRows_ < gatherC ? blockRows_ : gatherC);
        } else if (schedule_ == kScheduleGeneral2D) {
            pipe_.InitBuffer(tileQueue_, kBufferNum, blockRows_ * cols_ * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, blockRows_ * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, blockRows_ * sizeof(uint32_t));
            BuildRowOffsets(blockRows_, cols_);
        } else if (schedule_ == kScheduleGenericRowsReordered) {
            const bool fusedGather = blockRows_ == kGenericReorderedFusedGroup;
            const int64_t groupSize = fusedGather ?
                kGenericReorderedFusedGroup : kGenericReorderedGroup;
            const int64_t inputRowBytes =
                (groupSize * cols_ * sizeof(T) + kDataBlockBytes - 1) /
                kDataBlockBytes * kDataBlockBytes;
            const int64_t outputGroupBytes =
                (rows_ * cols_ * sizeof(T) + kDataBlockBytes - 1) /
                kDataBlockBytes * kDataBlockBytes;
            pipe_.InitBuffer(tileQueue_, kBufferNum, rows_ * inputRowBytes);
            pipe_.InitBuffer(outputQueue_, fusedGather ? kBufferNum : 1, groupSize * outputGroupBytes);
            pipe_.InitBuffer(gatherIndexBuf_, (fusedGather ?
                groupSize * outputGroupBytes / sizeof(T) : rows_ * cols_) * sizeof(uint32_t));
            pipe_.InitBuffer(reorderedScratchBuf_, 2 * kReverseIndexChunk * sizeof(int32_t));
        } else if (schedule_ == kScheduleGenericNd) {
            pipe_.InitBuffer(tileQueue_, 1, kGenericNdTileElems * kDataBlockBytes);
            pipe_.InitBuffer(outputQueue_, 1, kGenericNdTileElems * (sizeof(T) == 1 ? 3 : sizeof(T)));
            pipe_.InitBuffer(copyQueue_, kBufferNum, kGenericNdInputBytes);
            pipe_.InitBuffer(gatherIndexBuf_, kGenericNdTileElems * sizeof(uint32_t));
            BuildRowOffsets(kGenericNdTileElems,
                            sizeof(T) == 1 ? kDataBlockBytes * sizeof(half) : kDataBlockBytes / sizeof(T));
            if constexpr (sizeof(T) == 1) {
                pipe_.InitBuffer(genericCastBuf_, kGenericNdTileElems * kDataBlockBytes * sizeof(half));
            }
        } else if (schedule_ == kScheduleCopy2D) {
            pipe_.InitBuffer(copyQueue_, kBufferNum, blockRows_ * sizeof(T));
        } else if (schedule_ == kScheduleByteTile2D) {
            pipe_.InitBuffer(tileQueue_, kBufferNum, kByteTileRows * kByteTileCols * sizeof(T));
            pipe_.InitBuffer(outputQueue_, kBufferNum, kByteTileRows * kByteTileCols * sizeof(T));
        }
    }

    __aicore__ inline int64_t PlaneTileCols() const
    {
        const int64_t contiguousDim = inputShape_[rank_ - 1];
        return schedule_ == kSchedulePlaneB32WideTail ? kPlaneB32WideTailCols :
               schedule_ == kSchedulePlaneB32Wide ? kPlaneB32WideTileCols :
               (contiguousDim < kPlaneNdTileCols ? contiguousDim : kPlaneNdTileCols);
    }

    __aicore__ inline void InitNdMetadata(const TransposeTilingData& tilingData)
    {
        rank_ = tilingData.rank;
        totalLength_ = tilingData.totalLength;
        inputShape_[0] = tilingData.in0;
        inputShape_[1] = tilingData.in1;
        inputShape_[2] = tilingData.in2;
        inputShape_[3] = tilingData.in3;
        inputShape_[4] = tilingData.in4;
        inputShape_[5] = tilingData.in5;
        inputShape_[6] = tilingData.in6;
        inputShape_[7] = tilingData.in7;
        perm_[0] = tilingData.perm0;
        perm_[1] = tilingData.perm1;
        perm_[2] = tilingData.perm2;
        perm_[3] = tilingData.perm3;
        perm_[4] = tilingData.perm4;
        perm_[5] = tilingData.perm5;
        perm_[6] = tilingData.perm6;
        perm_[7] = tilingData.perm7;
        for (int64_t axis = 0; axis < rank_; ++axis) {
            outputShape_[axis] = inputShape_[perm_[axis]];
        }
        inputStrides_[rank_ - 1] = 1;
        outputStrides_[rank_ - 1] = 1;
        for (int64_t axis = rank_ - 2; axis >= 0; --axis) {
            inputStrides_[axis] = inputStrides_[axis + 1] * inputShape_[axis + 1];
            outputStrides_[axis] = outputStrides_[axis + 1] * outputShape_[axis + 1];
        }
        if (schedule_ == kSchedulePlaneNd || schedule_ == kSchedulePlaneCompactB16 ||
            schedule_ == kSchedulePlaneB32 || schedule_ == kSchedulePlaneB32Wide ||
            schedule_ == kSchedulePlaneB32WideTail) {
            const int64_t tileCols = PlaneTileCols();
            const bool compactInputRows = perm_[rank_ - 1] == rank_ - 2 &&
                                          inputShape_[rank_ - 1] <= tileCols;
            const int64_t localTileCols = compactInputRows ? tileCols :
                (tileCols * sizeof(T) + 31) / 32 * 32 / sizeof(T);
            const int64_t offsetCols = schedule_ == kSchedulePlaneCompactB16 ?
                kPlaneCompactGroupCols : localTileCols;
            if (schedule_ != kSchedulePlaneB32 && schedule_ != kSchedulePlaneB32Wide &&
                schedule_ != kSchedulePlaneB32WideTail) {
                BuildTransposeOffsetsBytes(blockRows_, localTileCols, offsetCols,
                                           sizeof(T) == 1 ? sizeof(half) : sizeof(T));
            }
        }
        if (schedule_ == kScheduleRank3Perm120) {
            totalRowTasks_ = inputShape_[1] *
                             ((inputShape_[0] + blockRows_ - 1) / blockRows_) *
                             ((inputShape_[2] + kAlignedTileCols - 1) / kAlignedTileCols);
        } else if (schedule_ == kScheduleRank4Perm0231) {
            totalRowTasks_ = inputShape_[0] * inputShape_[2] *
                             ((inputShape_[1] + kAlignedTileRows - 1) / kAlignedTileRows) *
                             ((inputShape_[3] + kAlignedTileCols - 1) / kAlignedTileCols);
        } else if (schedule_ == kScheduleRank4Perm0312) {
            totalRowTasks_ = inputShape_[0] * inputShape_[1] *
                             ((inputShape_[2] + kAlignedTileRows - 1) / kAlignedTileRows) *
                             ((inputShape_[3] + kAlignedTileCols - 1) / kAlignedTileCols);
        } else if (schedule_ == kScheduleRank4Perm0231Wide) {
            totalRowTasks_ = inputShape_[0] * inputShape_[2] *
                            (inputShape_[1] / kWideTileRows) * (inputShape_[3] / kWideTileCols);
        } else if (schedule_ == kScheduleRank4Perm0312Wide) {
            totalRowTasks_ = inputShape_[0] *
                             ((inputShape_[1] * inputShape_[2] + blockRows_ - 1) / blockRows_) *
                             (inputShape_[3] / kWideTileCols);
        }
        inputGm_.SetGlobalBuffer((__gm__ T*)inputPtr_, totalLength_);
        outputGm_.SetGlobalBuffer((__gm__ T*)outputPtr_, totalLength_);
    }

    template <bool FullRows = false>
    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t extraTask = blockIdx < bigCoreCount_ ? 1 : 0;
        const int64_t firstTask = blockIdx * smallCoreTasks_ +
                                  (blockIdx < bigCoreCount_ ? blockIdx : bigCoreCount_);
        const int64_t lastTask = firstTask + smallCoreTasks_ + extraTask;
        if constexpr (FullRows) {
            const int64_t rowTiles = (rows_ + blockRows_ - 1) / blockRows_;
            const int64_t planes = totalLength_ / rows_ / cols_;
            const int64_t totalTasks = planes * rowTiles;
            const int64_t coreCount = (totalTasks - bigCoreCount_) / smallCoreTasks_;
            ProcessPlaneB32FullRows(
                blockIdx, blockIdx + (smallCoreTasks_ + extraTask) * coreCount, coreCount);
        } else if (schedule_ == kScheduleAligned2D) {
            if (sizeof(T) == 2 && blockRows_ >= 32) {
                ProcessAlignedColumnSlabs(firstTask, lastTask);
            } else {
                ProcessAligned2D(firstTask, lastTask);
            }
        } else if (schedule_ == kScheduleCopy2D) {
            ProcessCopy(firstTask, lastTask);
        } else if (schedule_ == kScheduleByteTile2D) {
            ProcessByteTile2D(firstTask, lastTask);
        } else if (schedule_ == kScheduleTileGrid2D) {
            ProcessTileGrid2D(firstTask, lastTask);
        } else if (schedule_ == kScheduleTileGrid32x32) {
            ProcessTileGrid32x32(firstTask, lastTask);
        } else if (schedule_ == kScheduleTileGrid2DPad) {
            ProcessTileGrid2DPad(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank3Perm120) {
            ProcessRank3Perm120(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank4Perm0231) {
            ProcessRank4Perm0231(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank4Perm0231Wide) {
            ProcessRank4Perm0231Wide(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank4Perm0312Wide) {
            ProcessRank4Perm0312Wide(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank4Perm0312) {
            ProcessRank4Perm0312(firstTask, lastTask);
        } else if (schedule_ == kScheduleGenericRowsReordered) {
            ProcessGenericRowsReordered(firstTask, lastTask);
        } else if (schedule_ == kScheduleGenericNd) {
            ProcessGenericNd(firstTask, lastTask);
        } else if (schedule_ == kSchedulePlaneB32FullRows) {
            ProcessPlaneB32FullRows(firstTask, lastTask);
        } else if (schedule_ == kSchedulePlaneB32Grouped) {
            ProcessPlaneB32Grouped(firstTask, lastTask);
        } else if (schedule_ == kScheduleRank5GroupedB32) {
            ProcessRank5Grouped(firstTask, lastTask);
        } else if (schedule_ == kSchedulePlaneNd || schedule_ == kSchedulePlaneCompactB16 ||
                   schedule_ == kSchedulePlaneB32 || schedule_ == kSchedulePlaneB32Wide ||
                   schedule_ == kSchedulePlaneB32WideTail) {
            ProcessPlaneNd(firstTask, lastTask);
        } else if (schedule_ == kScheduleReverseNd) {
            ProcessReverseNd(firstTask, lastTask);
        } else {
            ProcessGeneral(firstTask, lastTask);
        }
    }

    __aicore__ inline void ProcessBlockedFullRows(int64_t taskCount)
    {
        blockedOutput_ = true;
        ProcessPlaneB32FullRows(0, taskCount);
    }

private:
    __aicore__ inline bool IsCompactFullRows() const
    {
        return blockRows_ == kCompactFullRows;
    }

    __aicore__ inline uint32_t FullRowsOutputBuffers() const
    {
        return cols_ < kFullTripleBufferMinCols || cols_ >= kFullTripleBufferMaxCols ?
            kBufferNum : kFullOutputBufferNum;
    }

    __aicore__ inline int64_t FullRowsInputCols() const
    {
        return IsCompactFullRows() ? cols_ : (cols_ + 7) / 8 * 8;
    }

    __aicore__ inline int64_t FullRowsOutputTileCols(int64_t inputCols, uint32_t outputBuffers) const
    {
        const int64_t maxCols = (kPlaneB32FullTileElements - blockRows_ * inputCols) /
                                (outputBuffers * blockRows_) / 8 * 8;
        if (IsCompactFullRows()) {
            return kCompactTileCols;
        }
        const int64_t tiles = (cols_ + maxCols - 1) / maxCols;
        return ((cols_ + tiles - 1) / tiles + 7) / 8 * 8;
    }

    __aicore__ inline void InitPlaneB32FullRowsBuffers()
    {
        const int64_t inputCols = FullRowsInputCols();
        if (IsCompactFullRows()) {
            pipe_.InitBuffer(tileQueue_, 1,
                             (blockRows_ * inputCols + kCompactInputGuard) * sizeof(T));
            pipe_.InitBuffer(outputQueue_, 2, blockRows_ * kCompactTileCols * sizeof(T));
            pipe_.InitBuffer(gatherIndexBuf_, blockRows_ * kCompactTileCols * sizeof(uint32_t));
            return;
        }
        const uint32_t outputBuffers = FullRowsOutputBuffers();
        const int64_t outputCols = FullRowsOutputTileCols(inputCols, outputBuffers);
        pipe_.InitBuffer(tileQueue_, 1, blockRows_ * inputCols * sizeof(T));
        pipe_.InitBuffer(outputQueue_, outputBuffers, blockRows_ * outputCols * sizeof(T));
    }

    __aicore__ inline void BuildCompactFullRowsOffsets()
    {
        BuildTransposeOffsetsBytes(blockRows_, cols_, kCompactTileCols, sizeof(T));
    }

    __aicore__ inline void BuildRowOffsets(int64_t count, int64_t strideElems)
    {
        BuildRowOffsetsAt(0, count, strideElems);
    }

    __aicore__ inline void BuildRowOffsetsAt(int64_t base, int64_t count, int64_t strideElems)
    {
        LocalTensor<int32_t> offsets = gatherIndexBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsets[base], 0, static_cast<int32_t>(strideElems * sizeof(T)),
                                  static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildTransposeOffsets(int64_t tileRows, int64_t tileCols)
    {
        BuildTransposeOffsetsBytes(tileRows, tileCols, tileCols, sizeof(T));
    }

    __aicore__ inline void BuildReorderedRowOffsets(int64_t inputRowElems, int64_t outputGroupElems,
                                                    int64_t groups)
    {
        LocalTensor<int32_t> offsets = gatherIndexBuf_.Get<int32_t>();
        LocalTensor<int32_t> quotient = reorderedScratchBuf_.Get<int32_t>();
        LocalTensor<float> temp = quotient[kReverseIndexChunk].ReinterpretCast<float>();
        const int64_t count = rows_ * cols_;
        Duplicate(offsets, 0, static_cast<uint32_t>(outputGroupElems));
        PipeBarrier<PIPE_V>();
        for (int64_t base = 0; base < count; base += kReverseIndexChunk) {
            const uint32_t length = static_cast<uint32_t>(count - base < kReverseIndexChunk ?
                                                           count - base : kReverseIndexChunk);
            CreateVecIndex(offsets[base], static_cast<int32_t>(base), length);
            PipeBarrier<PIPE_V>();
            Cast(temp, offsets[base], RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, 1.0f / static_cast<float>(cols_), length);
            Adds(temp, temp, 1.0e-4f, length);
            PipeBarrier<PIPE_V>();
            Cast(quotient, temp, RoundMode::CAST_FLOOR, length);
            PipeBarrier<PIPE_V>();
            Muls(quotient, quotient,
                 static_cast<int32_t>((inputRowElems - cols_) * sizeof(T)), length);
            Muls(offsets[base], offsets[base], static_cast<int32_t>(sizeof(T)), length);
            PipeBarrier<PIPE_V>();
            Add(offsets[base], offsets[base], quotient, length);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t group = 1; group < groups; ++group) {
            Duplicate(offsets[group * outputGroupElems], 0, static_cast<uint32_t>(outputGroupElems));
            PipeBarrier<PIPE_V>();
            Adds(offsets[group * outputGroupElems], offsets,
                 static_cast<int32_t>(group * cols_ * sizeof(T)), static_cast<uint32_t>(count));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildTransposeOffsetsBytes(int64_t tileRows, int64_t inputStride,
                                                      int64_t offsetCols, int64_t elementBytes)
    {
        LocalTensor<int32_t> offsets = gatherIndexBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsets, 0, static_cast<int32_t>(inputStride * elementBytes),
                                  static_cast<uint32_t>(tileRows));
        PipeBarrier<PIPE_V>();
        for (int64_t c = 1; c < offsetCols; ++c) {
            Adds(offsets[c * tileRows], offsets, static_cast<int32_t>(c * elementBytes),
                 static_cast<uint32_t>(tileRows));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildReverseOffsets(int64_t dimA, int64_t tileB, int64_t localC, int64_t gatherCols)
    {
        LocalTensor<int32_t> offsets = gatherIndexBuf_.Get<int32_t>();
        LocalTensor<int32_t> scratch = reverseScratchBuf_.Get<int32_t>();
        LocalTensor<int32_t> quotient = scratch;
        LocalTensor<int32_t> coordinateC = scratch[kReverseIndexChunk];
        LocalTensor<float> temp = scratch[2 * kReverseIndexChunk].ReinterpretCast<float>();
        const int64_t count = gatherCols * tileB * dimA;
        for (int64_t base = 0; base < count; base += kReverseIndexChunk) {
            const uint32_t length = static_cast<uint32_t>(count - base < kReverseIndexChunk ?
                                                           count - base : kReverseIndexChunk);
            CreateVecIndex(offsets[base], static_cast<int32_t>(base), length);
            PipeBarrier<PIPE_V>();
            Cast(temp, offsets[base], RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, 1.0f / static_cast<float>(dimA), length);
            Adds(temp, temp, 1.0e-4f, length);
            PipeBarrier<PIPE_V>();
            Cast(quotient, temp, RoundMode::CAST_FLOOR, length);
            PipeBarrier<PIPE_V>();
            Muls(temp.ReinterpretCast<int32_t>(), quotient, static_cast<int32_t>(dimA), length);
            Sub(offsets[base], offsets[base], temp.ReinterpretCast<int32_t>(), length);
            PipeBarrier<PIPE_V>();
            Muls(offsets[base], offsets[base],
                 static_cast<int32_t>(tileB * localC * sizeof(T)), length);

            Cast(temp, quotient, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, 1.0f / static_cast<float>(tileB), length);
            PipeBarrier<PIPE_V>();
            Cast(coordinateC, temp, RoundMode::CAST_FLOOR, length);
            PipeBarrier<PIPE_V>();
            Muls(temp.ReinterpretCast<int32_t>(), coordinateC, static_cast<int32_t>(tileB), length);
            Sub(quotient, quotient, temp.ReinterpretCast<int32_t>(), length);
            PipeBarrier<PIPE_V>();
            Muls(quotient, quotient, static_cast<int32_t>(localC * sizeof(T)), length);
            Muls(coordinateC, coordinateC, static_cast<int32_t>(sizeof(T)), length);
            PipeBarrier<PIPE_V>();
            Add(offsets[base], offsets[base], quotient, length);
            PipeBarrier<PIPE_V>();
            Add(offsets[base], offsets[base], coordinateC, length);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline int64_t InitTaskCount()
    {
        if (schedule_ == kScheduleCopy2D) {
            return (totalLength_ + blockRows_ - 1) / blockRows_;
        }
        if (schedule_ == kScheduleByteTile2D) {
            return ((rows_ + kByteTileRows - 1) / kByteTileRows) *
                   ((cols_ + kByteTileCols - 1) / kByteTileCols);
        }
        if (schedule_ == kScheduleTileGrid2D) {
            return ((rows_ + kAlignedTileRows - 1) / kAlignedTileRows) *
                   ((cols_ + kAlignedTileCols - 1) / kAlignedTileCols);
        }
        if (schedule_ == kScheduleTileGrid2DPad) {
            return ((rows_ + blockRows_ - 1) / blockRows_) *
                   ((cols_ + padTileCols_ - 1) / padTileCols_);
        }
        if (schedule_ == kScheduleTileGrid32x32) {
            return ((rows_ + kGrid32TileRows - 1) / kGrid32TileRows) *
                   ((cols_ + kGrid32TileCols - 1) / kGrid32TileCols);
        }
        return (rows_ + blockRows_ - 1) / blockRows_;
    }

    __aicore__ inline void ProcessAligned2D(int64_t firstTask, int64_t lastTask)
    {
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t rowBase = task * kAlignedTileRows;
            QueueAligned2DInput(rowBase, 0);
            for (int64_t colBase = 0; colBase < cols_; colBase += kAlignedTileCols) {
                const int64_t nextColBase = colBase + kAlignedTileCols;
                if (nextColBase < cols_) {
                    QueueAligned2DInput(rowBase, nextColBase);
                }
                LocalTensor<T> tile = tileQueue_.DeQue<T>();
                StoreAligned2DOutput(tile, rowBase, colBase);
                tileQueue_.FreeTensor(tile);
            }
        }
    }

    __aicore__ inline void QueueAligned2DInput(int64_t rowBase, int64_t colBase)
    {
        LocalTensor<T> tile = tileQueue_.AllocTensor<T>();
        DataCopyParams copyInParams = {
            static_cast<uint16_t>(kAlignedTileRows),
            static_cast<uint16_t>(kAlignedTileCols * sizeof(T) / 32),
            static_cast<uint16_t>((cols_ - kAlignedTileCols) * sizeof(T) / 32),
            0};
        DataCopy(tile, inputGm_[rowBase * cols_ + colBase], copyInParams);
        tileQueue_.EnQue(tile);
    }

    __aicore__ inline void ProcessAlignedColumnSlabs(int64_t firstTask, int64_t lastTask)
    {
        constexpr int64_t tileCols = 32;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t column = task * tileCols;
            LocalTensor<T> input = tileQueue_.AllocTensor<T>();
            DataCopy(input, inputGm_[column],
                DataCopyParams{static_cast<uint16_t>(rows_),
                    static_cast<uint16_t>(tileCols * sizeof(T) / 32),
                    static_cast<uint16_t>((cols_ - tileCols) * sizeof(T) / 32), 0});
            tileQueue_.EnQue(input);
            input = tileQueue_.DeQue<T>();
            LocalTensor<T> output = outputQueue_.AllocTensor<T>();
            const uint64_t srcBase = reinterpret_cast<uint64_t>(input.GetPhyAddr());
            const uint64_t dstBase = reinterpret_cast<uint64_t>(output.GetPhyAddr());
            for (int64_t row = 0; row < rows_; row += 16) {
                uint64_t src[16], dst[16];
                for (int64_t lane = 0; lane < 16; ++lane) {
                    src[lane] = srcBase + (row + lane) * tileCols * sizeof(T);
                    dst[lane] = dstBase + (lane * rows_ + row) * sizeof(T);
                }
                TransDataTo5HD<half>(dst, src, {false, false, 2,
                    static_cast<uint16_t>(rows_), 1});
            }
            PipeBarrier<PIPE_V>();
            tileQueue_.FreeTensor(input);
            outputQueue_.EnQue(output);
            output = outputQueue_.DeQue<T>();
            DataCopy(outputGm_[column * rows_], output, static_cast<uint32_t>(tileCols * rows_));
            outputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void StoreAligned2DOutput(LocalTensor<T>& tile, int64_t rowBase, int64_t colBase)
    {
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        Gather(outputTile, tile, gatherOffsets, 0, kAlignedTileRows * kAlignedTileCols);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        outputTile = outputQueue_.DeQue<T>();
        DataCopyParams copyOutParams = {
            static_cast<uint16_t>(kAlignedTileCols),
            static_cast<uint16_t>(kAlignedTileRows * sizeof(T) / 32),
            0,
            static_cast<uint16_t>((rows_ - kAlignedTileRows) * sizeof(T) / 32)};
        DataCopy(outputGm_[colBase * rows_ + rowBase], outputTile, copyOutParams);
        outputQueue_.FreeTensor(outputTile);
    }

    __aicore__ inline void ProcessTileGrid2D(int64_t firstTask, int64_t lastTask)
    {
        const int64_t colTiles = cols_ / kAlignedTileCols;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t colTile = task % colTiles;
            const int64_t rowTile = task / colTiles;
            const int64_t rowBase = rowTile * kAlignedTileRows;
            const int64_t colBase = colTile * kAlignedTileCols;
            if (rowBase >= rows_) {
                break;
            }
            QueueAligned2DInput(rowBase, colBase);
            LocalTensor<T> tile = tileQueue_.DeQue<T>();
            StoreAligned2DOutput(tile, rowBase, colBase);
            tileQueue_.FreeTensor(tile);
        }
    }

    __aicore__ inline void ProcessTileGrid2DPad(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        TileInfo current = DecodeTile(firstTask, blockRows_, padTileCols_);
        CopyInPadTile(current);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            TileInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeTile(task + 1, blockRows_, padTileCols_);
                CopyInPadTile(next);
            }
            ComputePadTile(current);
            CopyOutPadTile(current);
            current = next;
        }
    }

    __aicore__ inline void ProcessTileGrid32x32(int64_t firstTask, int64_t lastTask)
    {
        const int64_t colTiles = cols_ / kGrid32TileCols;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t colTile = task % colTiles;
            const int64_t rowTile = task / colTiles;
            const int64_t rowBase = rowTile * kGrid32TileRows;
            const int64_t colBase = colTile * kGrid32TileCols;
            if (rowBase >= rows_) {
                break;
            }
            TransposeGrid32Tile2D(rowBase, colBase);
        }
    }

    __aicore__ inline void TransposeGrid32Tile2D(int64_t rowBase, int64_t colBase)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        DataCopyParams copyInParams = {
            static_cast<uint16_t>(kGrid32TileRows),
            static_cast<uint16_t>(kGrid32TileCols * sizeof(T) / 32),
            static_cast<uint16_t>((cols_ - kGrid32TileCols) * sizeof(T) / 32),
            0};
        DataCopy(inputTile, inputGm_[rowBase * cols_ + colBase], copyInParams);
        tileQueue_.EnQue(inputTile);
        inputTile = tileQueue_.DeQue<T>();
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        Gather(outputTile, inputTile, gatherOffsets, 0, kGrid32TileRows * kGrid32TileCols);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        outputTile = outputQueue_.DeQue<T>();
        DataCopyParams copyOutParams = {
            static_cast<uint16_t>(kGrid32TileCols),
            static_cast<uint16_t>(kGrid32TileRows * sizeof(T) / 32),
            0,
            static_cast<uint16_t>((rows_ - kGrid32TileRows) * sizeof(T) / 32)};
        DataCopy(outputGm_[colBase * rows_ + rowBase], outputTile, copyOutParams);
        outputQueue_.FreeTensor(outputTile);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline int64_t AlignLocalElems(int64_t elems)
    {
        const int64_t bytes = elems * static_cast<int64_t>(sizeof(T));
        return ((bytes + 31) / 32) * 32 / static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline TileInfo DecodeTile(int64_t task, int64_t tileRows, int64_t tileCols)
    {
        const int64_t rowTiles = (rows_ + tileRows - 1) / tileRows;
        const int64_t rowBase = task % rowTiles * tileRows;
        const int64_t colBase = task / rowTiles * tileCols;
        return {rowBase, colBase,
                rows_ - rowBase < tileRows ? rows_ - rowBase : tileRows,
                cols_ - colBase < tileCols ? cols_ - colBase : tileCols};
    }

    __aicore__ inline void CopyInPadTile(const TileInfo& tile)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(tile.rows),
            static_cast<uint32_t>(tile.cols * sizeof(T)),
            static_cast<uint32_t>((cols_ - tile.cols) * sizeof(T)),
            0,
            0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputTile, inputGm_[tile.rowBase * cols_ + tile.colBase], copyInParams, padParams);
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void ComputePadTile(const TileInfo& tile)
    {
        LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        const int64_t inputStride = AlignLocalElems(tile.cols);
        const int64_t outputStride = AlignLocalElems(tile.rows);
        if (inputStride == padTileCols_ && outputStride == blockRows_) {
            LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
            Gather(outputTile, inputTile, gatherOffsets, 0, blockRows_ * padTileCols_);
        } else {
            const int64_t rowOffsetBase = blockRows_ * padTileCols_;
            BuildRowOffsetsAt(rowOffsetBase, tile.rows, inputStride);
            LocalTensor<uint32_t> rowOffsets = gatherIndexBuf_.Get<uint32_t>()[rowOffsetBase];
            for (int64_t c = 0; c < tile.cols; ++c) {
                Gather(outputTile[c * outputStride], inputTile, rowOffsets,
                       static_cast<uint32_t>(c * sizeof(T)), static_cast<uint32_t>(tile.rows));
            }
        }
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline void CopyOutPadTile(const TileInfo& tile)
    {
        LocalTensor<T> outputTile = outputQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(tile.cols),
            static_cast<uint32_t>(tile.rows * sizeof(T)),
            0,
            static_cast<uint32_t>((rows_ - tile.rows) * sizeof(T)),
            0};
        DataCopyPad(outputGm_[tile.colBase * rows_ + tile.rowBase], outputTile, copyOutParams);
        outputQueue_.FreeTensor(outputTile);
    }

    __aicore__ inline void ProcessGeneral(int64_t firstTask, int64_t lastTask)
    {
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t rowBase = task * blockRows_;
            if (rowBase >= rows_) {
                break;
            }
            int64_t currentRows = rows_ - rowBase;
            if (currentRows > blockRows_) {
                currentRows = blockRows_;
            }
            LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
            const uint32_t inputElems = static_cast<uint32_t>(currentRows * cols_);
            if ((static_cast<int64_t>(inputElems) * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                DataCopy(inputTile, inputGm_[rowBase * cols_], inputElems);
            } else {
                DataCopyExtParams params = {1, static_cast<uint32_t>(inputElems * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(inputTile, inputGm_[rowBase * cols_], params, padParams);
            }
            tileQueue_.EnQue(inputTile);
            inputTile = tileQueue_.DeQue<T>();

            for (int64_t col = 0; col < cols_; ++col) {
                LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
                LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
                Gather(outputTile, inputTile, gatherOffsets, static_cast<uint32_t>(col * sizeof(T)),
                       static_cast<uint32_t>(currentRows));
                PipeBarrier<PIPE_V>();
                outputQueue_.EnQue(outputTile);
                outputTile = outputQueue_.DeQue<T>();
                DataCopyParams copyOutParams = {
                    1,
                    static_cast<uint16_t>(currentRows * sizeof(T) / 32),
                    0,
                    0};
                if ((currentRows * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                    DataCopy(outputGm_[col * rows_ + rowBase], outputTile, copyOutParams);
                } else {
                    DataCopyExtParams padParams = {
                        1,
                        static_cast<uint32_t>(currentRows * sizeof(T)),
                        0,
                        0,
                        0};
                    DataCopyPad(outputGm_[col * rows_ + rowBase], outputTile, padParams);
                }
                outputQueue_.FreeTensor(outputTile);
            }
            tileQueue_.FreeTensor(inputTile);
        }
    }

    __aicore__ inline int64_t GenericInputRowBase(int64_t outputRow)
    {
        int64_t inputBase = 0;
        for (int64_t axis = rank_ - 2; axis >= 0; --axis) {
            const int64_t coordinate = outputRow % outputShape_[axis];
            outputRow /= outputShape_[axis];
            inputBase += coordinate * inputStrides_[perm_[axis]];
        }
        return inputBase;
    }

    __aicore__ inline void CopyGenericContiguous(int64_t inputBase, int64_t outputBase, int64_t count)
    {
        LocalTensor<T> tile = copyQueue_.AllocTensor<T>();
        DataCopyExtParams params = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(tile, inputGm_[inputBase], params, padding);
        copyQueue_.EnQue(tile);
        tile = copyQueue_.DeQue<T>();
        DataCopyPad(outputGm_[outputBase], tile, params);
        copyQueue_.FreeTensor(tile);
    }

    __aicore__ inline void ProcessGenericNd(int64_t firstTask, int64_t lastTask)
    {
        const int64_t lastDim = outputShape_[rank_ - 1];
        const int64_t inputStride = inputStrides_[perm_[rank_ - 1]];
        if (inputStride == 1 && lastDim * sizeof(T) <= kGenericNdInputBytes) {
            ProcessGenericContiguousRows(firstTask, lastTask, lastDim);
            return;
        }
        const int64_t colTiles = (lastDim + blockRows_ - 1) / blockRows_;
        const int64_t tileRows = lastDim <= blockRows_ ? blockRows_ / lastDim : 1;
        const int64_t totalRows = totalLength_ / lastDim;
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t rowBase = task / colTiles * tileRows;
            const int64_t colBase = task % colTiles * blockRows_;
            const int64_t rows = totalRows - rowBase < tileRows ? totalRows - rowBase : tileRows;
            const int64_t cols = lastDim - colBase < blockRows_ ? lastDim - colBase : blockRows_;
            const int64_t tileElements = rows * cols;
            const int64_t outputBase = rowBase * lastDim + colBase;
            if (inputStride == 1) {
                for (int64_t row = 0; row < rows; ++row) {
                    const int64_t inputBase = GenericInputRowBase(rowBase + row) + colBase;
                    CopyGenericContiguous(inputBase, outputBase + row * lastDim, cols);
                }
                continue;
            }
            LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
            DataCopyExtParams copyIn = {static_cast<uint16_t>(cols), static_cast<uint32_t>(sizeof(T)),
                                        static_cast<uint32_t>((inputStride - 1) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padding{
                true, 0, static_cast<uint8_t>(kDataBlockBytes / sizeof(T) - 1),
                static_cast<T>(0)};
            const int64_t localRowStride = cols * kDataBlockBytes / sizeof(T);
            for (int64_t row = 0; row < rows; ++row) {
                const int64_t inputBase = GenericInputRowBase(rowBase + row) + colBase * inputStride;
                DataCopyPad(inputTile[row * localRowStride], inputGm_[inputBase], copyIn, padding);
            }
            tileQueue_.EnQue(inputTile);
            inputTile = tileQueue_.DeQue<T>();

            LocalTensor<T> outputStorage = outputQueue_.AllocTensor<T>();
            LocalTensor<T> outputTile = outputStorage;
            if constexpr (sizeof(T) == 1) {
                LocalTensor<half> expanded = genericCastBuf_.Get<half>();
                LocalTensor<half> gathered = outputStorage.template ReinterpretCast<half>();
                Cast(expanded, inputTile, RoundMode::CAST_NONE,
                     static_cast<uint32_t>(tileElements * kDataBlockBytes));
                PipeBarrier<PIPE_V>();
                Gather(gathered, expanded, gatherOffsets, 0, static_cast<uint32_t>(tileElements));
                PipeBarrier<PIPE_V>();
                outputTile = outputStorage[kGenericNdTileElems * sizeof(half)];
                Cast(outputTile, gathered, RoundMode::CAST_RINT, static_cast<uint32_t>(tileElements));
            } else {
                Gather(outputTile, inputTile, gatherOffsets, 0, static_cast<uint32_t>(tileElements));
            }
            PipeBarrier<PIPE_V>();
            outputQueue_.EnQue(outputStorage);
            outputStorage = outputQueue_.DeQue<T>();
            if constexpr (sizeof(T) == 1) {
                outputTile = outputStorage[kGenericNdTileElems * sizeof(half)];
            } else {
                outputTile = outputStorage;
            }
            DataCopyExtParams copyOut = {1, static_cast<uint32_t>(tileElements * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outputBase], outputTile, copyOut);
            outputQueue_.FreeTensor(outputStorage);
            tileQueue_.FreeTensor(inputTile);
        }
    }

    __aicore__ inline GenericRowTileInfo DecodeGenericRowTile(int64_t task, int64_t tileRows,
                                                              int64_t rowAxis, int64_t rowTiles)
    {
        const int64_t rowInAxis = task % rowTiles * tileRows;
        int64_t outputOuter = task / rowTiles;
        if (schedule_ == kScheduleGenericRowsReordered) {
            const int64_t dim0 = outputShape_[0];
            outputOuter = outputOuter % dim0 * outputShape_[1] + outputOuter / dim0;
        }
        const int64_t outputRow = outputOuter * rowAxis + rowInAxis;
        return {GenericInputRowBase(outputRow), outputRow,
                rowAxis - rowInAxis < tileRows ? rowAxis - rowInAxis : tileRows};
    }

    __aicore__ inline ReorderedRowTaskInfo DecodeReorderedRowTask(
        int64_t task, int64_t groupTiles, int64_t groupSize)
    {
        const int64_t batches = inputShape_[1];
        const int64_t batch = task % batches;
        const int64_t tile = task / batches;
        const int64_t deficit = groupTiles * groupSize - inputShape_[2];
        const int64_t prefixTiles =
            (bigCoreCount_ * (smallCoreTasks_ + 1) + batches - 1) / batches;
        if (deficit > 0 && prefixTiles >= deficit && prefixTiles < groupTiles) {
            const int64_t phase = batch * prefixTiles / batches;
            const int64_t shortBefore = tile < prefixTiles ?
                (tile * deficit + phase) / prefixTiles : deficit;
            const int64_t shortAfter = tile + 1 < prefixTiles ?
                ((tile + 1) * deficit + phase) / prefixTiles : deficit;
            return {tile * groupSize - shortBefore,
                    groupSize - (shortAfter > shortBefore ? 1 : 0), batch};
        }
        const int64_t colBase = tile * groupSize;
        return {colBase, inputShape_[2] - colBase < groupSize ?
                         inputShape_[2] - colBase : groupSize, batch};
    }

    __aicore__ inline void QueueReorderedRows(const ReorderedRowTaskInfo& info,
                                               int64_t inputRowBytes)
    {
        const int64_t blockBytes = info.groupCols * cols_ * sizeof(T);
        const int64_t alignedBlockBytes = (blockBytes + kDataBlockBytes - 1) /
                                          kDataBlockBytes * kDataBlockBytes;
        const int64_t inputBase = (info.batch * inputShape_[2] + info.colBase) * cols_;
        LocalTensor<T> input = tileQueue_.AllocTensor<T>();
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(rows_), static_cast<uint32_t>(blockBytes),
            static_cast<uint32_t>((inputStrides_[0] - info.groupCols * cols_) * sizeof(T)),
            static_cast<uint32_t>((inputRowBytes - alignedBlockBytes) / kDataBlockBytes), 0};
        DataCopyPadExtParams<T> padding{
            true, 0, static_cast<uint8_t>(alignedBlockBytes / sizeof(T) - info.groupCols * cols_),
            static_cast<T>(0)};
        DataCopyPad(input, inputGm_[inputBase], copyIn, padding);
        tileQueue_.EnQue(input);
    }

    __aicore__ inline void ProcessReorderedRows(const ReorderedRowTaskInfo& info,
                                                 int64_t outputGroupElems, int64_t groupSize)
    {
        LocalTensor<T> input = tileQueue_.DeQue<T>();
        LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> offsets = gatherIndexBuf_.Get<uint32_t>();
        if (groupSize == kGenericReorderedFusedGroup) {
            Gather(output, input, offsets, 0,
                   static_cast<uint32_t>(info.groupCols * outputGroupElems));
        } else {
            for (int64_t group = 0; group < info.groupCols; ++group) {
                Gather(output[group * outputGroupElems], input, offsets,
                       static_cast<uint32_t>(group * cols_ * sizeof(T)),
                       static_cast<uint32_t>(rows_ * cols_));
            }
        }
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        tileQueue_.FreeTensor(input);
        output = outputQueue_.DeQue<T>();
        const int64_t outputBase =
            (info.colBase * inputShape_[1] + info.batch) * rows_ * cols_;
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(info.groupCols),
            static_cast<uint32_t>(rows_ * cols_ * sizeof(T)), 0,
            static_cast<uint32_t>((inputShape_[1] - 1) * rows_ * cols_ * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], output, copyOut);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessGenericRowsReordered(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        const int64_t groupSize = blockRows_ == kGenericReorderedFusedGroup ?
            kGenericReorderedFusedGroup : kGenericReorderedGroup;
        const int64_t groupTiles = (inputShape_[2] + groupSize - 1) / groupSize;
        const int64_t inputRowBytes =
            (groupSize * cols_ * sizeof(T) + kDataBlockBytes - 1) /
            kDataBlockBytes * kDataBlockBytes;
        const int64_t outputGroupElems =
            (rows_ * cols_ * sizeof(T) + kDataBlockBytes - 1) /
            kDataBlockBytes * kDataBlockBytes / sizeof(T);
        ReorderedRowTaskInfo current =
            DecodeReorderedRowTask(firstTask, groupTiles, groupSize);
        QueueReorderedRows(current, inputRowBytes);
        BuildReorderedRowOffsets(inputRowBytes / sizeof(T), outputGroupElems,
                                 groupSize == kGenericReorderedFusedGroup ? groupSize : 1);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            ReorderedRowTaskInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeReorderedRowTask(task + 1, groupTiles, groupSize);
                QueueReorderedRows(next, inputRowBytes);
            }
            ProcessReorderedRows(current, outputGroupElems, groupSize);
            current = next;
        }
    }

    __aicore__ inline void CopyInGenericRows(const GenericRowTileInfo& info, int64_t lastDim,
                                             int64_t localRowBytes, int64_t inputRowStride)
    {
        LocalTensor<T> tile = copyQueue_.AllocTensor<T>();
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(info.rows), static_cast<uint32_t>(lastDim * sizeof(T)),
            info.rows > 1 ? static_cast<uint32_t>((inputRowStride - lastDim) * sizeof(T)) : 0, 0, 0};
        DataCopyPadExtParams<T> padding{
            true, 0, static_cast<uint8_t>(localRowBytes / sizeof(T) - lastDim), static_cast<T>(0)};
        DataCopyPad(tile, inputGm_[info.inputBase], copyIn, padding);
        copyQueue_.EnQue(tile);
    }

    __aicore__ inline void CopyOutGenericRows(const GenericRowTileInfo& info, int64_t lastDim)
    {
        LocalTensor<T> tile = copyQueue_.DeQue<T>();
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(info.rows), static_cast<uint32_t>(lastDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[info.outputBase * lastDim], tile, copyOut);
        copyQueue_.FreeTensor(tile);
    }

    __aicore__ inline void ProcessGenericContiguousRows(int64_t firstTask, int64_t lastTask, int64_t lastDim)
    {
        if (firstTask >= lastTask) {
            return;
        }
        if (schedule_ != kScheduleGenericRowsReordered && totalLength_ * sizeof(T) >= 1024 * 1024) {
            inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        const int64_t localRowBytes = (lastDim * sizeof(T) + kDataBlockBytes - 1) /
                                      kDataBlockBytes * kDataBlockBytes;
        const int64_t tileRows = blockRows_;
        const int64_t rowAxis = outputShape_[rank_ - 2];
        const int64_t rowTiles = (rowAxis + tileRows - 1) / tileRows;
        const int64_t inputRowStride = inputStrides_[perm_[rank_ - 2]];
        GenericRowTileInfo current = DecodeGenericRowTile(firstTask, tileRows, rowAxis, rowTiles);
        CopyInGenericRows(current, lastDim, localRowBytes, inputRowStride);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            GenericRowTileInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeGenericRowTile(task + 1, tileRows, rowAxis, rowTiles);
                CopyInGenericRows(next, lastDim, localRowBytes, inputRowStride);
            }
            CopyOutGenericRows(current, lastDim);
            current = next;
        }
    }

    __aicore__ inline void PlaneBases(int64_t planeIndex, int64_t contiguousOutputAxis,
                                      int64_t& inputBase, int64_t& outputBase)
    {
        inputBase = 0;
        outputBase = 0;
        for (int64_t axis = rank_ - 2; axis >= 0; --axis) {
            if (axis == contiguousOutputAxis) {
                continue;
            }
            const int64_t coordinate = planeIndex % outputShape_[axis];
            planeIndex /= outputShape_[axis];
            inputBase += coordinate * inputStrides_[perm_[axis]];
            outputBase += coordinate * outputStrides_[axis];
        }
    }

    __aicore__ inline GroupedPlaneTileInfo DecodeGroupedPlaneTile(
        int64_t task, int64_t innerPlanes, int64_t groupPlanes,
        int64_t contiguousDim, int64_t colTiles)
    {
        if (colTiles == 1 && innerPlanes % groupPlanes == 0) {
            return {task * groupPlanes, 0, contiguousDim, groupPlanes, 0, 0};
        }
        const int64_t colBase = task % colTiles * kPlaneB32GroupedTileCols;
        const int64_t groupTask = task / colTiles;
        const int64_t cols = contiguousDim - colBase < kPlaneB32GroupedTileCols ?
                             contiguousDim - colBase : kPlaneB32GroupedTileCols;
        if (innerPlanes % groupPlanes == 0) {
            return {groupTask * groupPlanes, colBase, cols, groupPlanes, 0, 0};
        }
        const int64_t groupTiles = (innerPlanes + groupPlanes - 1) / groupPlanes;
        const int64_t groupIndex = groupTask % groupTiles;
        const int64_t smallGroups = innerPlanes / groupTiles;
        const int64_t bigGroups = innerPlanes % groupTiles;
        const int64_t planes = smallGroups + (groupIndex < bigGroups ? 1 : 0);
        const int64_t groupBase = groupIndex * smallGroups +
                                  (groupIndex < bigGroups ? groupIndex : bigGroups);
        return {groupTask / groupTiles * innerPlanes + groupBase, colBase, cols, planes, 0, 0};
    }

    __aicore__ inline void CopyInGroupedPlaneTile(
        GroupedPlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        PlaneBases(tile.basePlane, config.contiguousOutputAxis, tile.inputBase, tile.outputBase);
        const int64_t groupInputStride = inputStrides_[perm_[rank_ - 2]];
        const int64_t alignedColsBytes = (tile.cols * sizeof(T) + 31) / 32 * 32;
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(config.transposedDim), static_cast<uint32_t>(tile.cols * sizeof(T)),
            static_cast<uint32_t>((config.inputRowStride - tile.cols) * sizeof(T)),
            static_cast<uint32_t>((config.localTileCols * sizeof(T) - alignedColsBytes) / kDataBlockBytes), 0};
        DataCopyPadExtParams<T> padding{
            false, 0, static_cast<uint8_t>(alignedColsBytes / sizeof(T) - tile.cols), static_cast<T>(0)};
        for (int64_t plane = 0; plane < tile.planes; ++plane) {
            DataCopyPad(inputTile[plane * config.transposedDim * config.localTileCols],
                        inputGm_[tile.inputBase + plane * groupInputStride + tile.colBase], copyIn, padding);
        }
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void ComputeGroupedPlaneTile(
        const GroupedPlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        const int64_t groupedRows = tile.planes * config.transposedDim;
        const int64_t outputStride = (groupedRows + 15) / 16 * 16;
        const uint64_t srcBase = reinterpret_cast<uint64_t>(inputTile.GetPhyAddr());
        const uint64_t dstBase = reinterpret_cast<uint64_t>(outputTile.GetPhyAddr());
        for (int64_t rowBase = 0; rowBase < outputStride; rowBase += 16) {
            uint64_t dst[16];
            uint64_t src[16];
            for (int64_t lane = 0; lane < 16; ++lane) {
                src[lane] = srcBase + (rowBase + lane) * config.localTileCols * sizeof(T);
                dst[lane] = dstBase +
                    (rowBase + (lane / 2) * outputStride + (lane % 2) * 8) * sizeof(T);
            }
            TransposeB32Blocks(dst, src, static_cast<uint8_t>((tile.cols + 7) / 8),
                               static_cast<uint16_t>(outputStride));
        }
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline void CopyOutGroupedPlaneTile(
        const GroupedPlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> outputTile = outputQueue_.DeQue<T>();
        const int64_t groupedRows = tile.planes * config.transposedDim;
        const int64_t outputStride = (groupedRows + 15) / 16 * 16;
        const int64_t alignedRowsBytes =
            (groupedRows * sizeof(T) + kDataBlockBytes - 1) /
            kDataBlockBytes * kDataBlockBytes;
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(tile.cols), static_cast<uint32_t>(groupedRows * sizeof(T)),
            static_cast<uint32_t>((outputStride * sizeof(T) - alignedRowsBytes) / kDataBlockBytes),
            static_cast<uint32_t>((config.outputRowStride - groupedRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[tile.outputBase + tile.colBase * config.outputRowStride], outputTile, copyOut);
        outputQueue_.FreeTensor(outputTile);
    }

    __aicore__ inline void ProcessPlaneB32Grouped(int64_t firstTask, int64_t lastTask)
    {
        const int64_t transposedAxis = perm_[rank_ - 1];
        const int64_t transposedDim = inputShape_[transposedAxis];
        const int64_t contiguousDim = inputShape_[rank_ - 1];
        int64_t contiguousOutputAxis = 0;
        while (perm_[contiguousOutputAxis] != rank_ - 1) {
            ++contiguousOutputAxis;
        }
        const int64_t innerPlanes = outputShape_[rank_ - 2];
        const int64_t groupPlanes = innerPlanes < kPlaneB32GroupedPlanes ?
                                    innerPlanes : kPlaneB32GroupedPlanes;
        const int64_t colTiles = (contiguousDim + kPlaneB32GroupedTileCols - 1) /
                                 kPlaneB32GroupedTileCols;
        const int64_t outputRowStride = outputStrides_[contiguousOutputAxis];
        const int64_t tileLength = (blockRows_ + 15) / 16 * 16 * kPlaneB32GroupedTileCols;
        const PlaneConfig config = {
            transposedDim, contiguousDim, contiguousOutputAxis, inputStrides_[transposedAxis], outputRowStride,
            kPlaneB32GroupedTileCols, kPlaneB32GroupedTileCols, groupPlanes, colTiles, tileLength, false};
        GroupedPlaneTileInfo current = DecodeGroupedPlaneTile(
            firstTask, innerPlanes, groupPlanes, contiguousDim, colTiles);
        CopyInGroupedPlaneTile(current, config);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            ComputeGroupedPlaneTile(current, config);
            GroupedPlaneTileInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeGroupedPlaneTile(
                    task + 1, innerPlanes, groupPlanes, contiguousDim, colTiles);
                CopyInGroupedPlaneTile(next, config);
            }
            CopyOutGroupedPlaneTile(current, config);
            current = next;
        }
    }

    __aicore__ inline Rank5TaskInfo DecodeRank5Task(int64_t task)
    {
        const int64_t d3Tiles = (inputShape_[3] + kRank5InnerTile - 1) / kRank5InnerTile;
        const int64_t d3Base = task % d3Tiles * kRank5InnerTile;
        int64_t outer = task / d3Tiles;
        const int64_t d0Groups = (inputShape_[0] + kRank5OuterGroup - 1) / kRank5OuterGroup;
        const int64_t d0Base = outer % d0Groups * kRank5OuterGroup;
        const int64_t d1 = outer / d0Groups;
        const int64_t group = inputShape_[0] - d0Base < kRank5OuterGroup ?
                              inputShape_[0] - d0Base : kRank5OuterGroup;
        const int64_t currentD3 = inputShape_[3] - d3Base < kRank5InnerTile ?
                                  inputShape_[3] - d3Base : kRank5InnerTile;
        const int64_t cols = currentD3 * inputShape_[4];
        return {d3Base, d0Base, d1, group, cols};
    }

    __aicore__ inline void CopyInRank5Grouped(const Rank5TaskInfo& info)
    {
        const int64_t localCols = (kRank5InnerTile * inputShape_[4] + 7) / 8 * 8;
        const int64_t alignedCols = (info.cols + 7) / 8 * 8;
        LocalTensor<T> input = tileQueue_.AllocTensor<T>();
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(inputShape_[2]), static_cast<uint32_t>(info.cols * sizeof(T)),
            static_cast<uint32_t>((inputStrides_[2] - info.cols) * sizeof(T)),
            static_cast<uint32_t>((localCols - alignedCols) * sizeof(T) / kDataBlockBytes), 0};
        DataCopyPadExtParams<T> padding{
            true, 0, static_cast<uint8_t>(alignedCols - info.cols), static_cast<T>(0)};
        for (int64_t plane = 0; plane < info.group; ++plane) {
            const int64_t inputBase = (info.d0Base + plane) * inputStrides_[0] +
                                      info.d1 * inputStrides_[1] + info.d3Base * inputStrides_[3];
            DataCopyPad(input[plane * inputShape_[2] * localCols], inputGm_[inputBase], copyIn, padding);
        }
        tileQueue_.EnQue(input);
    }

    __aicore__ inline void ComputeRank5Grouped(const Rank5TaskInfo& info)
    {
        const int64_t localCols = (kRank5InnerTile * inputShape_[4] + 7) / 8 * 8;
        const int64_t outputRows = (kRank5OuterGroup * inputShape_[2] + 15) / 16 * 16;
        LocalTensor<T> input = tileQueue_.DeQue<T>();
        LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        const uint64_t srcBase = reinterpret_cast<uint64_t>(input.GetPhyAddr());
        const uint64_t dstBase = reinterpret_cast<uint64_t>(output.GetPhyAddr());
        for (int64_t rowBase = 0; rowBase < outputRows; rowBase += 16) {
            uint64_t dst[16];
            uint64_t src[16];
            for (int64_t lane = 0; lane < 16; ++lane) {
                src[lane] = srcBase + (rowBase + lane) * localCols * sizeof(T);
                dst[lane] = dstBase +
                    (rowBase + (lane / 2) * outputRows + (lane % 2) * 8) * sizeof(T);
            }
            TransposeB32Blocks(dst, src, static_cast<uint8_t>((info.cols + 7) / 8),
                               static_cast<uint16_t>(outputRows));
        }
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        tileQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOutRank5Grouped(const Rank5TaskInfo& info)
    {
        const int64_t groupRows = info.group * inputShape_[2];
        const int64_t outputRows = (kRank5OuterGroup * inputShape_[2] + 15) / 16 * 16;
        const int64_t outputBase = info.d3Base * outputStrides_[0] +
                                   info.d1 * outputStrides_[2] + info.d0Base * outputStrides_[3];
        const int64_t alignedRowsBytes = (groupRows * sizeof(T) + 31) / 32 * 32;
        LocalTensor<T> output = outputQueue_.DeQue<T>();
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(info.cols), static_cast<uint32_t>(groupRows * sizeof(T)),
            static_cast<uint32_t>((outputRows * sizeof(T) - alignedRowsBytes) / kDataBlockBytes),
            static_cast<uint32_t>((outputStrides_[1] - groupRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], output, copyOut);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessRank5Grouped(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        Rank5TaskInfo current = DecodeRank5Task(firstTask);
        CopyInRank5Grouped(current);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            ComputeRank5Grouped(current);
            Rank5TaskInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeRank5Task(task + 1);
                CopyInRank5Grouped(next);
            }
            CopyOutRank5Grouped(current);
            current = next;
        }
    }

    __aicore__ inline void ProcessPlaneB32FullRows(
        int64_t firstTask, int64_t lastTask, int64_t taskStride = 1)
    {
        if (firstTask >= lastTask) {
            return;
        }
        const int64_t transposedDim = rows_;
        const int64_t contiguousDim = cols_;
        const int64_t inputCols = FullRowsInputCols();
        const uint32_t outputBuffers = FullRowsOutputBuffers();
        const int64_t outputTileCols = FullRowsOutputTileCols(inputCols, outputBuffers);
        const int64_t rowTiles = (transposedDim + blockRows_ - 1) / blockRows_;
        const int64_t planeStride = transposedDim * contiguousDim;
        inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        QueuePlaneB32FullRowsInput(firstTask, rowTiles, planeStride, contiguousDim, inputCols);
        if (IsCompactFullRows()) {
            BuildCompactFullRowsOffsets();
        }
        for (int64_t task = firstTask; task < lastTask; task += taskStride) {
            const int64_t rowBase = task % rowTiles * blockRows_;
            const int64_t rows = transposedDim - rowBase < blockRows_ ?
                                 transposedDim - rowBase : blockRows_;
            const int64_t planeBase = task / rowTiles * planeStride;
            const int64_t alignedRowsBytes = (rows * sizeof(T) + kDataBlockBytes - 1) /
                                             kDataBlockBytes * kDataBlockBytes;
            LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
            for (int64_t colBase = 0; colBase < contiguousDim; colBase += outputTileCols) {
                const int64_t cols = contiguousDim - colBase < outputTileCols ?
                                     contiguousDim - colBase : outputTileCols;
                LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
                if (IsCompactFullRows()) {
                    TransposeCompactFullRows(outputTile, inputTile, colBase, cols);
                } else {
                    TransposePlaneB32(outputTile, inputTile[colBase], inputCols, cols);
                }
                outputQueue_.EnQue(outputTile);
                if (colBase + outputTileCols >= contiguousDim) {
                    tileQueue_.FreeTensor(inputTile);
                    if (task + taskStride < lastTask) {
                        QueuePlaneB32FullRowsInput(
                            task + taskStride, rowTiles, planeStride, contiguousDim, inputCols);
                    }
                }
                outputTile = outputQueue_.DeQue<T>();
                if (blockedOutput_) {
                    const int64_t colTiles = (contiguousDim + outputTileCols - 1) / outputTileCols;
                    const int64_t outputBase = ((task / rowTiles * colTiles + colBase / outputTileCols) *
                        rowTiles + task % rowTiles) * outputTileCols * blockRows_;
                    DataCopy(outputGm_[outputBase], outputTile, outputTileCols * blockRows_);
                    outputQueue_.FreeTensor(outputTile);
                    continue;
                }
                DataCopyExtParams copyOut = {
                    static_cast<uint16_t>(cols), static_cast<uint32_t>(rows * sizeof(T)),
                    static_cast<uint32_t>((blockRows_ * sizeof(T) - alignedRowsBytes) / kDataBlockBytes),
                    static_cast<uint32_t>((transposedDim - rows) * sizeof(T)), 0};
                DataCopyPad(outputGm_[planeBase + colBase * transposedDim + rowBase], outputTile, copyOut);
                outputQueue_.FreeTensor(outputTile);
            }
        }
    }

    __aicore__ inline void QueuePlaneB32FullRowsInput(int64_t task, int64_t rowTiles,
                                                       int64_t planeStride, int64_t contiguousDim,
                                                       int64_t inputCols)
    {
        const int64_t rowBase = task % rowTiles * blockRows_;
        const int64_t rows = rows_ - rowBase < blockRows_ ? rows_ - rowBase : blockRows_;
        const int64_t planeBase = task / rowTiles * planeStride;
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        if (IsCompactFullRows()) {
            const int64_t elements = rows * contiguousDim;
            const int64_t alignedElements = (elements + 7) / 8 * 8;
            DataCopyExtParams copyIn = {1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padding{
                true, 0, static_cast<uint8_t>(alignedElements - elements), static_cast<T>(0)};
            DataCopyPad(inputTile, inputGm_[planeBase + rowBase * contiguousDim], copyIn, padding);
            tileQueue_.EnQue(inputTile);
            return;
        }
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(rows), static_cast<uint32_t>(contiguousDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{
            true, 0, static_cast<uint8_t>(inputCols - contiguousDim), static_cast<T>(0)};
        DataCopyPad(inputTile, inputGm_[planeBase + rowBase * contiguousDim], copyIn, padding);
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void TransposeCompactFullRows(const LocalTensor<T>& output,
                                                    const LocalTensor<T>& input,
                                                    int64_t colBase, int64_t cols)
    {
        Gather(output, input, gatherIndexBuf_.Get<uint32_t>(),
               static_cast<uint32_t>(colBase * sizeof(T)), cols * blockRows_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessPlaneNd(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        const int64_t transposedAxis = perm_[rank_ - 1];
        const int64_t transposedDim = inputShape_[transposedAxis];
        const int64_t contiguousDim = inputShape_[rank_ - 1];
        int64_t contiguousOutputAxis = 0;
        while (perm_[contiguousOutputAxis] != rank_ - 1) {
            ++contiguousOutputAxis;
        }
        const int64_t inputRowStride = inputStrides_[transposedAxis];
        const int64_t outputRowStride = outputStrides_[contiguousOutputAxis];
        const int64_t tileCols = PlaneTileCols();
        const bool compactInputRows = transposedAxis == rank_ - 2 && contiguousDim <= tileCols;
        const int64_t localTileCols = compactInputRows ? tileCols :
            (tileCols * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        const int64_t rowTiles = (transposedDim + blockRows_ - 1) / blockRows_;
        const int64_t colTiles = (contiguousDim + tileCols - 1) / tileCols;
        const int64_t tileLength = blockRows_ * localTileCols;
        const PlaneConfig config = {
            transposedDim, contiguousDim, contiguousOutputAxis, inputRowStride, outputRowStride,
            tileCols, localTileCols, rowTiles, colTiles, tileLength, compactInputRows};
        if (schedule_ == kSchedulePlaneB32 && totalLength_ * sizeof(T) >= 1024 * 1024) {
            inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        if (schedule_ == kSchedulePlaneB32Wide || schedule_ == kSchedulePlaneB32WideTail) {
            const int64_t tailCols = config.contiguousDim - (config.colTiles - 1) * config.tileCols;
            const bool rowsFirst = schedule_ == kSchedulePlaneB32WideTail ?
                config.colTiles >= 4 || tailCols >= 256 :
                config.colTiles >= 4 && tailCols >= 64;
            if (config.colTiles >= 3 && (schedule_ == kSchedulePlaneB32WideTail || !rowsFirst ||
                config.contiguousDim % (kDataBlockBytes / sizeof(T)) == 0)) {
                inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            }
            const int64_t taskCount = lastTask - firstTask;
            int64_t task = firstTask;
            PlaneTileInfo current = rowsFirst ? DecodePlaneTileRowsFirst(task, config) :
                                                DecodePlaneTile(task, config);
            CopyInPlaneTile(current, config);
            for (int64_t localTask = 0; localTask < taskCount; ++localTask) {
                ComputePlaneTile(current, config);
                PlaneTileInfo next = current;
                ++task;
                if (localTask + 1 < taskCount) {
                    next = rowsFirst ? DecodePlaneTileRowsFirst(task, config) : DecodePlaneTile(task, config);
                    CopyInPlaneTile(next, config);
                }
                CopyOutPlaneTile(current, config);
                current = next;
            }
            return;
        }
        PlaneTileInfo current = DecodePlaneTile(firstTask, config);
        CopyInPlaneTile(current, config);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            PlaneTileInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodePlaneTile(task + 1, config);
                CopyInPlaneTile(next, config);
            }
            ComputePlaneTile(current, config);
            CopyOutPlaneTile(current, config);
            current = next;
        }
    }

    __aicore__ inline PlaneTileInfo DecodePlaneTile(int64_t task, const PlaneConfig& config)
    {
        int64_t remaining = task;
        const int64_t colBase = remaining % config.colTiles * config.tileCols;
        remaining /= config.colTiles;
        const int64_t rowBase = remaining % config.rowTiles * blockRows_;
        return MakePlaneTile(remaining / config.rowTiles, rowBase, colBase, config);
    }

    __aicore__ inline PlaneTileInfo DecodePlaneTileRowsFirst(int64_t task, const PlaneConfig& config)
    {
        int64_t remaining = task;
        const int64_t rowBase = remaining % config.rowTiles * blockRows_;
        remaining /= config.rowTiles;
        const int64_t colBase = remaining % config.colTiles * config.tileCols;
        return MakePlaneTile(remaining / config.colTiles, rowBase, colBase, config);
    }

    __aicore__ inline PlaneTileInfo MakePlaneTile(int64_t planeIndex, int64_t rowBase, int64_t colBase,
                                                  const PlaneConfig& config)
    {
        int64_t inputBase = 0;
        int64_t outputBase = 0;
        PlaneBases(planeIndex, config.contiguousOutputAxis, inputBase, outputBase);
        inputBase += rowBase * config.inputRowStride + colBase;
        outputBase += colBase * config.outputRowStride + rowBase;
        return {inputBase, outputBase,
                config.transposedDim - rowBase < blockRows_ ? config.transposedDim - rowBase : blockRows_,
                config.contiguousDim - colBase < config.tileCols ? config.contiguousDim - colBase : config.tileCols};
    }

    __aicore__ inline void CopyInPlaneTile(const PlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        if (config.compactInputRows && tile.cols == config.contiguousDim) {
            const int64_t elements = tile.rows * tile.cols;
            const int64_t alignedElements = (elements * sizeof(T) + 31) / 32 * 32 / sizeof(T);
            DataCopyExtParams copyIn = {1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padding{
                true, 0, static_cast<uint8_t>(alignedElements - elements), static_cast<T>(0)};
            DataCopyPad(inputTile, inputGm_[tile.inputBase], copyIn, padding);
        } else {
            const int64_t alignedColsBytes = (tile.cols * sizeof(T) + 31) / 32 * 32;
            DataCopyExtParams copyIn = {
                static_cast<uint16_t>(tile.rows), static_cast<uint32_t>(tile.cols * sizeof(T)),
                static_cast<uint32_t>((config.inputRowStride - tile.cols) * sizeof(T)),
                static_cast<uint32_t>((config.localTileCols * sizeof(T) - alignedColsBytes) / kDataBlockBytes), 0};
            DataCopyPadExtParams<T> padding{
                true, 0, static_cast<uint8_t>(alignedColsBytes / sizeof(T) - tile.cols), static_cast<T>(0)};
            DataCopyPad(inputTile, inputGm_[tile.inputBase], copyIn, padding);
        }
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void TransposePlaneB32(const LocalTensor<T>& outputTile,
                                             const LocalTensor<T>& inputTile, int64_t inputStride, int64_t cols)
    {
        uint64_t dst[16];
        uint64_t src[16];
        const uint64_t dstBase = reinterpret_cast<uint64_t>(outputTile.GetPhyAddr());
        const uint64_t srcBase = reinterpret_cast<uint64_t>(inputTile.GetPhyAddr());
        for (int64_t rowBase = 0; rowBase < blockRows_; rowBase += 16) {
            for (int64_t lane = 0; lane < 16; ++lane) {
                src[lane] = srcBase + (rowBase + lane) * inputStride * sizeof(T);
                dst[lane] = dstBase + (rowBase + (lane / 2) * blockRows_ + (lane % 2) * 8) * sizeof(T);
            }
            TransposeB32Blocks(dst, src, static_cast<uint8_t>((cols + 7) / 8),
                               static_cast<uint16_t>(blockRows_));
        }
    }

    __aicore__ inline void ComputePlaneTile(const PlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
        LocalTensor<T> outputStorage = outputQueue_.AllocTensor<T>();
        LocalTensor<T> outputTile = outputStorage;
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        const uint32_t gatheredElements = static_cast<uint32_t>(tile.cols * blockRows_);
        if (schedule_ == kSchedulePlaneCompactB16) {
            for (int64_t colBase = 0; colBase < tile.cols; colBase += kPlaneCompactGroupCols) {
                const int64_t groupCols = tile.cols - colBase < kPlaneCompactGroupCols ?
                                          tile.cols - colBase : kPlaneCompactGroupCols;
                Gather(outputTile[colBase * blockRows_], inputTile, gatherOffsets,
                       static_cast<uint32_t>(colBase * sizeof(T)),
                       static_cast<uint32_t>(groupCols * blockRows_));
            }
        } else if constexpr (sizeof(T) == 1) {
            LocalTensor<half> expanded = genericCastBuf_.Get<half>();
            LocalTensor<half> gathered = outputStorage.template ReinterpretCast<half>();
            Cast(expanded, inputTile, RoundMode::CAST_NONE, static_cast<uint32_t>(config.tileLength));
            PipeBarrier<PIPE_V>();
            Gather(gathered, expanded, gatherOffsets, 0, gatheredElements);
            PipeBarrier<PIPE_V>();
            outputTile = outputStorage[config.tileLength * sizeof(half)];
            Cast(outputTile, gathered, RoundMode::CAST_RINT, gatheredElements);
        } else if constexpr (sizeof(T) == 4) {
            if (schedule_ == kSchedulePlaneB32 || schedule_ == kSchedulePlaneB32Wide ||
                schedule_ == kSchedulePlaneB32WideTail ||
                (tile.rows == blockRows_ && blockRows_ % 16 == 0 && config.localTileCols % 8 == 0)) {
                TransposePlaneB32(outputTile, inputTile, config.localTileCols, config.localTileCols);
            } else {
                Gather(outputTile, inputTile, gatherOffsets, 0, gatheredElements);
            }
        } else {
            Gather(outputTile, inputTile, gatherOffsets, 0, gatheredElements);
        }
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputStorage);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline void CopyOutPlaneTile(const PlaneTileInfo& tile, const PlaneConfig& config)
    {
        LocalTensor<T> outputStorage = outputQueue_.DeQue<T>();
        LocalTensor<T> outputTile = sizeof(T) == 1 ?
            outputStorage[config.tileLength * sizeof(half)] : outputStorage;
        const int64_t alignedRowsBytes = (tile.rows * sizeof(T) + kDataBlockBytes - 1) /
                                         kDataBlockBytes * kDataBlockBytes;
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(tile.cols), static_cast<uint32_t>(tile.rows * sizeof(T)),
            static_cast<uint32_t>((blockRows_ * sizeof(T) - alignedRowsBytes) / kDataBlockBytes),
            static_cast<uint32_t>((config.outputRowStride - tile.rows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[tile.outputBase], outputTile, copyOut);
        outputQueue_.FreeTensor(outputStorage);
    }

    __aicore__ inline void ReversePlaneBases(int64_t planeIndex, int64_t& inputBase, int64_t& outputBase)
    {
        inputBase = 0;
        outputBase = 0;
        for (int64_t axis = rank_ - 2; axis >= 2; --axis) {
            const int64_t coordinate = planeIndex % inputShape_[axis];
            planeIndex /= inputShape_[axis];
            inputBase += coordinate * inputStrides_[axis];
            outputBase += coordinate * outputStrides_[rank_ - 1 - axis];
        }
    }

    __aicore__ inline void ProcessReverseNd(int64_t firstTask, int64_t lastTask)
    {
        const int64_t dimA = inputShape_[0];
        const int64_t dimB = inputShape_[1];
        const int64_t dimC = inputShape_[rank_ - 1];
        const int64_t tileB = cols_;
        const int64_t gatherC = sizeof(T) == 2 ? 16 : 32;
        const int64_t bTiles = (dimB + tileB - 1) / tileB;
        const int64_t cTiles = (dimC + blockRows_ - 1) / blockRows_;
        const int64_t localC = (blockRows_ * sizeof(T) + kDataBlockBytes - 1) /
                               kDataBlockBytes * kDataBlockBytes / sizeof(T);
        LocalTensor<uint32_t> offsets = gatherIndexBuf_.Get<uint32_t>();

        for (int64_t task = firstTask; task < lastTask; ++task) {
            int64_t remaining = task;
            const int64_t cBase = remaining % cTiles * blockRows_;
            remaining /= cTiles;
            const int64_t bBase = remaining % bTiles * tileB;
            const int64_t planeIndex = remaining / bTiles;
            const int64_t currentB = dimB - bBase < tileB ? dimB - bBase : tileB;
            const int64_t currentC = dimC - cBase < blockRows_ ? dimC - cBase : blockRows_;
            int64_t inputBase = 0;
            int64_t outputBase = 0;
            ReversePlaneBases(planeIndex, inputBase, outputBase);
            inputBase += bBase * inputStrides_[1] + cBase;
            outputBase += cBase * outputStrides_[0] + bBase * dimA;

            LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
            const int64_t alignedCBytes = (currentC * sizeof(T) + kDataBlockBytes - 1) /
                                          kDataBlockBytes * kDataBlockBytes;
            DataCopyExtParams copyIn = {
                static_cast<uint16_t>(currentB), static_cast<uint32_t>(currentC * sizeof(T)),
                static_cast<uint32_t>((inputStrides_[1] - currentC) * sizeof(T)),
                static_cast<uint32_t>((localC * sizeof(T) - alignedCBytes) / kDataBlockBytes), 0};
            DataCopyPadExtParams<T> padding{
                true, 0, static_cast<uint8_t>(alignedCBytes / sizeof(T) - currentC), static_cast<T>(0)};
            for (int64_t a = 0; a < dimA; ++a) {
                DataCopyPad(inputTile[a * tileB * localC],
                            inputGm_[inputBase + a * inputStrides_[0]], copyIn, padding);
            }
            tileQueue_.EnQue(inputTile);
            inputTile = tileQueue_.DeQue<T>();

            LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
            const int64_t outputRowBytes = (currentB * dimA * sizeof(T) + kDataBlockBytes - 1) /
                                           kDataBlockBytes * kDataBlockBytes;
            const int64_t outputRowElements = outputRowBytes / sizeof(T);
            for (int64_t groupBase = 0; groupBase < currentC; groupBase += gatherC) {
                const int64_t groupCols = currentC - groupBase < gatherC ? currentC - groupBase : gatherC;
                if (currentB == tileB) {
                    Gather(outputTile[groupBase * outputRowElements], inputTile, offsets,
                           static_cast<uint32_t>(groupBase * sizeof(T)),
                           static_cast<uint32_t>(groupCols * tileB * dimA));
                } else {
                    for (int64_t c = 0; c < groupCols; ++c) {
                        Gather(outputTile[(groupBase + c) * outputRowElements], inputTile,
                               offsets[c * tileB * dimA],
                               static_cast<uint32_t>(groupBase * sizeof(T)),
                               static_cast<uint32_t>(currentB * dimA));
                    }
                }
            }
            PipeBarrier<PIPE_V>();
            outputQueue_.EnQue(outputTile);
            outputTile = outputQueue_.DeQue<T>();
            DataCopyExtParams copyOut = {
                static_cast<uint16_t>(currentC), static_cast<uint32_t>(currentB * dimA * sizeof(T)), 0,
                static_cast<uint32_t>((outputStrides_[0] - currentB * dimA) * sizeof(T)), 0};
            DataCopyPad(outputGm_[outputBase], outputTile, copyOut);
            outputQueue_.FreeTensor(outputTile);
            tileQueue_.FreeTensor(inputTile);
        }
    }

    __aicore__ inline void ProcessCopy(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        QueueCopyInput(firstTask);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            if (task + 1 < lastTask) {
                QueueCopyInput(task + 1);
            }
            const int64_t offset = task * blockRows_;
            const int64_t currentElems = totalLength_ - offset < blockRows_ ? totalLength_ - offset : blockRows_;
            LocalTensor<T> tile = copyQueue_.DeQue<T>();
            if ((currentElems * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
                DataCopy(outputGm_[offset], tile, static_cast<uint32_t>(currentElems));
            } else {
                DataCopyExtParams params = {1, static_cast<uint32_t>(currentElems * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm_[offset], tile, params);
            }
            copyQueue_.FreeTensor(tile);
        }
    }

    __aicore__ inline void QueueCopyInput(int64_t task)
    {
        const int64_t offset = task * blockRows_;
        const int64_t currentElems = totalLength_ - offset < blockRows_ ? totalLength_ - offset : blockRows_;
        LocalTensor<T> tile = copyQueue_.AllocTensor<T>();
        if ((currentElems * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(tile, inputGm_[offset], static_cast<uint32_t>(currentElems));
        } else {
            DataCopyExtParams params = {1, static_cast<uint32_t>(currentElems * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(tile, inputGm_[offset], params, padParams);
        }
        copyQueue_.EnQue(tile);
    }

    __aicore__ inline void ProcessByteTile2D(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        TileInfo current = DecodeTile(firstTask, kByteTileRows, kByteTileCols);
        CopyInByteTile(current);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            TileInfo next = current;
            if (task + 1 < lastTask) {
                next = DecodeTile(task + 1, kByteTileRows, kByteTileCols);
                CopyInByteTile(next);
            }
            ComputeByteTile(current);
            CopyOutByteTile(current);
            current = next;
        }
    }

    __aicore__ inline void CopyInByteTile(const TileInfo& tile)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        if (tile.rows == kByteTileRows && tile.cols == kByteTileCols && cols_ % 32 == 0) {
            DataCopyParams params = {
                static_cast<uint16_t>(kByteTileRows),
                static_cast<uint16_t>(kByteTileCols / 32),
                static_cast<uint16_t>((cols_ - kByteTileCols) / 32),
                0};
            DataCopy(inputTile, inputGm_[tile.rowBase * cols_ + tile.colBase], params);
        } else {
            const int64_t localCols = (tile.cols + 31) / 32 * 32;
            DataCopyExtParams params = {
                static_cast<uint16_t>(tile.rows),
                static_cast<uint32_t>(tile.cols * sizeof(T)),
                static_cast<uint32_t>((cols_ - tile.cols) * sizeof(T)),
                0,
                0};
            DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(localCols - tile.cols), static_cast<T>(0)};
            DataCopyPad(inputTile, inputGm_[tile.rowBase * cols_ + tile.colBase], params, padParams);
        }
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void ComputeByteTile(const TileInfo& tile)
    {
        LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        TransposeByteTile2D(outputTile, inputTile, tile.cols);
        outputQueue_.EnQue(outputTile);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline void CopyOutByteTile(const TileInfo& tile)
    {
        LocalTensor<T> outputTile = outputQueue_.DeQue<T>();
        if (tile.rows == kByteTileRows && tile.cols == kByteTileCols && rows_ % 32 == 0) {
            DataCopyParams params = {
                static_cast<uint16_t>(kByteTileCols), 1, 0,
                static_cast<uint16_t>((rows_ - kByteTileRows) / 32)};
            DataCopy(outputGm_[tile.colBase * rows_ + tile.rowBase], outputTile, params);
        } else {
            DataCopyExtParams params = {
                static_cast<uint16_t>(tile.cols),
                static_cast<uint32_t>(tile.rows * sizeof(T)),
                0,
                static_cast<uint32_t>((rows_ - tile.rows) * sizeof(T)),
                0};
            DataCopyPad(outputGm_[tile.colBase * rows_ + tile.rowBase], outputTile, params);
        }
        outputQueue_.FreeTensor(outputTile);
    }

    __aicore__ inline void TransposeByteTile2D(const LocalTensor<T>& outputTile, const LocalTensor<T>& inputTile,
                                               int64_t tileCols)
    {
        uint64_t dst[16] = {0};
        uint64_t src[16] = {0};
        const uint64_t dstBase = reinterpret_cast<uint64_t>(outputTile.GetPhyAddr());
        const uint64_t srcBase = reinterpret_cast<uint64_t>(inputTile.GetPhyAddr());
        const uint8_t repeat = static_cast<uint8_t>((tileCols + 31) / 32);
        const int64_t srcRowStride = static_cast<int64_t>(repeat) * 32;
        const uint16_t dstRepStride = repeat > 1 ? 32 : 0;
        const uint16_t srcRepStride = repeat > 1 ? 1 : 0;
        for (int64_t i = 0; i < 16; ++i) {
            dst[i] = dstBase + i * kByteTileRows;
            src[i] = srcBase + i * srcRowStride;
        }

        TransDataTo5HD<uint8_t>(dst, src, {false, false, repeat, dstRepStride, srcRepStride});
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < 16; ++i) {
            dst[i] += 16 * kByteTileRows;
        }
        TransDataTo5HD<uint8_t>(dst, src, {false, true, repeat, dstRepStride, srcRepStride});
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < 16; ++i) {
            src[i] += 16 * srcRowStride;
        }
        TransDataTo5HD<uint8_t>(dst, src, {true, true, repeat, dstRepStride, srcRepStride});
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < 16; ++i) {
            dst[i] -= 16 * kByteTileRows;
        }
        TransDataTo5HD<uint8_t>(dst, src, {true, false, repeat, dstRepStride, srcRepStride});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransposeTile2D(int64_t inputBase, int64_t inputRowStride,
                                           int64_t outputBase, int64_t outputRowStride, int64_t tileRows)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        DataCopyParams copyInParams = {
            static_cast<uint16_t>(tileRows),
            static_cast<uint16_t>(kAlignedTileCols * sizeof(T) / 32),
            static_cast<uint16_t>((inputRowStride - kAlignedTileCols) * sizeof(T) / 32),
            0};
        DataCopy(inputTile, inputGm_[inputBase], copyInParams);
        tileQueue_.EnQue(inputTile);
        inputTile = tileQueue_.DeQue<T>();

        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        Gather(outputTile, inputTile, gatherOffsets, 0, tileRows * kAlignedTileCols);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        outputTile = outputQueue_.DeQue<T>();
        DataCopyParams copyOutParams = {
            static_cast<uint16_t>(kAlignedTileCols),
            static_cast<uint16_t>(tileRows * sizeof(T) / 32),
            0,
            static_cast<uint16_t>((outputRowStride - tileRows) * sizeof(T) / 32)};
        DataCopy(outputGm_[outputBase], outputTile, copyOutParams);
        outputQueue_.FreeTensor(outputTile);
        tileQueue_.FreeTensor(inputTile);
    }

    __aicore__ inline void ProcessRank3Perm120(int64_t firstTask, int64_t lastTask)
    {
        const int64_t dim0 = inputShape_[0];
        const int64_t dim1 = inputShape_[1];
        const int64_t dim2 = inputShape_[2];
        const int64_t rowTiles = dim0 / blockRows_;
        const int64_t colTiles = dim2 / kAlignedTileCols;
        const int64_t inputRowStride = dim1 * dim2;
        const int64_t outputRowStride = dim0;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            int64_t remaining = task;
            const int64_t colTile = remaining % colTiles;
            remaining /= colTiles;
            const int64_t rowTile = remaining % rowTiles;
            const int64_t dim1Idx = remaining / rowTiles;
            if (dim1Idx >= dim1) {
                break;
            }
            const int64_t rowBase = rowTile * blockRows_;
            const int64_t colBase = colTile * kAlignedTileCols;
            const int64_t inputBase = rowBase * inputRowStride + dim1Idx * dim2 + colBase;
            const int64_t outputBase = dim1Idx * dim2 * dim0 + colBase * dim0 + rowBase;
            TransposeTile2D(inputBase, inputRowStride, outputBase, outputRowStride, blockRows_);
        }
    }

    __aicore__ inline void ProcessRank4Perm0231(int64_t firstTask, int64_t lastTask)
    {
        const int64_t dim0 = inputShape_[0];
        const int64_t dim1 = inputShape_[1];
        const int64_t dim2 = inputShape_[2];
        const int64_t dim3 = inputShape_[3];
        const int64_t rowTiles = dim1 / kAlignedTileRows;
        const int64_t colTiles = dim3 / kAlignedTileCols;
        const int64_t inputRowStride = dim2 * dim3;
        const int64_t outputRowStride = dim1;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            int64_t remaining = task;
            const int64_t colTile = remaining % colTiles;
            remaining /= colTiles;
            const int64_t rowTile = remaining % rowTiles;
            remaining /= rowTiles;
            const int64_t dim2Idx = remaining % dim2;
            const int64_t dim0Idx = remaining / dim2;
            if (dim0Idx >= dim0) {
                break;
            }
            const int64_t channelBase = rowTile * kAlignedTileRows;
            const int64_t widthBase = colTile * kAlignedTileCols;
            const int64_t inputBase = dim0Idx * dim1 * dim2 * dim3 +
                                      channelBase * dim2 * dim3 +
                                      dim2Idx * dim3 + widthBase;
            const int64_t outputBase = dim0Idx * dim2 * dim3 * dim1 +
                                       dim2Idx * dim3 * dim1 +
                                       widthBase * dim1 + channelBase;
            TransposeTile2D(inputBase, inputRowStride, outputBase, outputRowStride, kAlignedTileRows);
        }
    }

    __aicore__ inline void ProcessRank4Perm0231Wide(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        const int64_t dim1 = inputShape_[1];
        const int64_t dim2 = inputShape_[2];
        const int64_t dim3 = inputShape_[3];
        const int64_t inputRowStride = dim2 * dim3;
        QueueRank4Perm0231WideInput(firstTask, dim1, dim2, dim3, inputRowStride);
        int64_t nextTask = firstTask + 1;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t colTile = task % (dim3 / kWideTileCols);
            const int64_t rowTask = task / (dim3 / kWideTileCols);
            const int64_t rowTile = rowTask % (dim1 / kWideTileRows);
            const int64_t plane = rowTask / (dim1 / kWideTileRows);
            const int64_t dim2Idx = plane % dim2;
            const int64_t dim0Idx = plane / dim2;
            if (dim0Idx >= inputShape_[0]) {
                break;
            }
            const int64_t outputBase = (dim0Idx * dim2 * dim3 + dim2Idx * dim3 +
                                       colTile * kWideTileCols) * dim1 + rowTile * kWideTileRows;
            if (nextTask < lastTask) {
                QueueRank4Perm0231WideInput(nextTask, dim1, dim2, dim3, inputRowStride);
                ++nextTask;
            }
            LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
            StoreWideOutput(inputTile, outputBase, dim1);
            tileQueue_.FreeTensor(inputTile);
        }
    }

    __aicore__ inline void QueueRank4Perm0231WideInput(int64_t task, int64_t dim1, int64_t dim2, int64_t dim3,
                                                       int64_t inputRowStride)
    {
        const int64_t colTile = task % (dim3 / kWideTileCols);
        const int64_t rowTask = task / (dim3 / kWideTileCols);
        const int64_t rowTile = rowTask % (dim1 / kWideTileRows);
        const int64_t plane = rowTask / (dim1 / kWideTileRows);
        const int64_t inputBase = ((plane / dim2 * dim1 + rowTile * kWideTileRows) * dim2 +
                                  plane % dim2) * dim3 + colTile * kWideTileCols;
        QueueWideInput(inputBase, inputRowStride);
    }

    __aicore__ inline void QueueWideInput(int64_t inputBase, int64_t inputRowStride)
    {
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        DataCopyParams copyInParams = {
            static_cast<uint16_t>(kWideTileRows),
            static_cast<uint16_t>(kWideTileCols * sizeof(T) / 32),
            static_cast<uint16_t>((inputRowStride - kWideTileCols) * sizeof(T) / 32),
            0};
        DataCopy(inputTile, inputGm_[inputBase], copyInParams);
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void StoreWideOutput(LocalTensor<T>& inputTile, int64_t outputBase, int64_t outputStride)
    {
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        Gather(outputTile, inputTile, gatherOffsets, 0, kWideTileRows * kWideTileCols);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        outputTile = outputQueue_.DeQue<T>();
        DataCopy(outputGm_[outputBase], outputTile,
            DataCopyParams{static_cast<uint16_t>(kWideTileCols),
                static_cast<uint16_t>(kWideTileRows * sizeof(T) / 32), 0,
                static_cast<uint16_t>((outputStride - kWideTileRows) * sizeof(T) / 32)});
        outputQueue_.FreeTensor(outputTile);
    }

    __aicore__ inline void ProcessRank4Perm0312(int64_t firstTask, int64_t lastTask)
    {
        const int64_t dim0 = inputShape_[0];
        const int64_t dim1 = inputShape_[1];
        const int64_t dim2 = inputShape_[2];
        const int64_t dim3 = inputShape_[3];
        const int64_t rowTiles = dim2 / kAlignedTileRows;
        const int64_t colTiles = dim3 / kAlignedTileCols;
        const int64_t inputRowStride = dim3;
        const int64_t outputRowStride = dim1 * dim2;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            int64_t remaining = task;
            const int64_t colTile = remaining % colTiles;
            remaining /= colTiles;
            const int64_t rowTile = remaining % rowTiles;
            remaining /= rowTiles;
            const int64_t dim1Idx = remaining % dim1;
            const int64_t dim0Idx = remaining / dim1;
            if (dim0Idx >= dim0) {
                break;
            }
            const int64_t heightBase = rowTile * kAlignedTileRows;
            const int64_t widthBase = colTile * kAlignedTileCols;
            const int64_t inputBase = dim0Idx * dim1 * dim2 * dim3 +
                                      dim1Idx * dim2 * dim3 +
                                      heightBase * dim3 + widthBase;
            const int64_t outputBase = dim0Idx * dim3 * dim1 * dim2 +
                                       widthBase * dim1 * dim2 +
                                       dim1Idx * dim2 + heightBase;
            TransposeTile2D(inputBase, inputRowStride, outputBase, outputRowStride, kAlignedTileRows);
        }
    }

    __aicore__ inline void ProcessRank4Perm0312Wide(int64_t firstTask, int64_t lastTask)
    {
        if (firstTask >= lastTask) {
            return;
        }
        const int64_t dim1 = inputShape_[1];
        const int64_t dim2 = inputShape_[2];
        const int64_t dim3 = inputShape_[3];
        const int64_t outputRowStride = dim1 * dim2;
        const int64_t groupsPerBatch = (outputRowStride + blockRows_ - 1) / blockRows_;
        QueueRank4Perm0312WideInput(firstTask, dim1, dim2, dim3, groupsPerBatch);
        int64_t nextTask = firstTask + 1;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t colTile = task % (dim3 / kWideTileCols);
            const int64_t rowTask = task / (dim3 / kWideTileCols);
            const int64_t groupIdx = rowTask % groupsPerBatch;
            const int64_t dim0Idx = rowTask / groupsPerBatch;
            if (dim0Idx >= inputShape_[0]) {
                break;
            }
            const int64_t rowBase = groupIdx * blockRows_;
            const int64_t currentRows = outputRowStride - rowBase < blockRows_ ? outputRowStride - rowBase : blockRows_;
            const int64_t outputBase = (dim0Idx * dim3 + colTile * kWideTileCols) * outputRowStride + rowBase;
            if (nextTask < lastTask) {
                QueueRank4Perm0312WideInput(nextTask, dim1, dim2, dim3, groupsPerBatch);
                ++nextTask;
            }
            LocalTensor<T> inputTile = tileQueue_.DeQue<T>();
            StoreWideOutputStrided(inputTile, outputBase, outputRowStride, currentRows);
            tileQueue_.FreeTensor(inputTile);
        }
    }

    __aicore__ inline void QueueRank4Perm0312WideInput(int64_t task, int64_t dim1, int64_t dim2, int64_t dim3,
                                                      int64_t groupsPerBatch)
    {
        const int64_t colTile = task % (dim3 / kWideTileCols);
        const int64_t rowTask = task / (dim3 / kWideTileCols);
        const int64_t groupIdx = rowTask % groupsPerBatch;
        const int64_t dim0Idx = rowTask / groupsPerBatch;
        const int64_t rowBase = groupIdx * blockRows_;
        const int64_t totalRows = dim1 * dim2;
        const int64_t currentRows = totalRows - rowBase < blockRows_ ? totalRows - rowBase : blockRows_;
        LocalTensor<T> inputTile = tileQueue_.AllocTensor<T>();
        DataCopy(inputTile, inputGm_[(dim0Idx * totalRows + rowBase) * dim3 + colTile * kWideTileCols],
            DataCopyParams{static_cast<uint16_t>(currentRows),
                static_cast<uint16_t>(kWideTileCols * sizeof(T) / 32),
                static_cast<uint16_t>((dim3 - kWideTileCols) * sizeof(T) / 32), 0});
        tileQueue_.EnQue(inputTile);
    }

    __aicore__ inline void StoreWideOutputStrided(LocalTensor<T>& inputTile, int64_t outputBase,
                                                  int64_t outputRowStride, int64_t currentRows)
    {
        LocalTensor<T> outputTile = outputQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> gatherOffsets = gatherIndexBuf_.Get<uint32_t>();
        Gather(outputTile, inputTile, gatherOffsets, 0, blockRows_ * kWideTileCols);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(outputTile);
        outputTile = outputQueue_.DeQue<T>();
        DataCopyParams copyOutParams = {
            static_cast<uint16_t>(kWideTileCols),
            static_cast<uint16_t>(currentRows * sizeof(T) / 32),
            static_cast<uint16_t>((blockRows_ - currentRows) * sizeof(T) / 32),
            static_cast<uint16_t>((outputRowStride - currentRows) * sizeof(T) / 32)};
        DataCopy(outputGm_[outputBase], outputTile, copyOutParams);
        outputQueue_.FreeTensor(outputTile);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> tileQueue_;
    TQue<QuePosition::VECOUT, kFullOutputBufferNum> outputQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kGenericRowsBufferNum> copyQueue_;
    TBuf<QuePosition::VECCALC> gatherIndexBuf_;
    TBuf<QuePosition::VECCALC> genericCastBuf_;
    TBuf<QuePosition::VECCALC> reverseScratchBuf_;
    TBuf<QuePosition::VECCALC> reorderedScratchBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GM_ADDR inputPtr_ = nullptr;
    GM_ADDR outputPtr_ = nullptr;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t totalLength_ = 0;
    bool blockedOutput_ = false;
    int64_t blockRows_ = 1;
    int64_t padTileCols_ = kAlignedTileCols;
    int64_t smallCoreTasks_ = 1;
    int64_t bigCoreCount_ = 0;
    int64_t totalRowTasks_ = 0;
    int64_t schedule_ = 0;
    int64_t rank_ = 2;
    int64_t inputShape_[8];
    int64_t outputShape_[8];
    int64_t inputStrides_[8];
    int64_t outputStrides_[8];
    int64_t perm_[8];
};

template <typename T, bool FullRows = false>
__aicore__ inline void RunTranspose(GM_ADDR input, GM_ADDR output, const TransposeTilingData& tilingData,
                                    int64_t localTasks = 0)
{
    KernelTranspose<T> kernel;
    kernel.template Init<FullRows>(input, output, tilingData.dim0, tilingData.dim1, tilingData.totalLength,
        tilingData.blockRows, tilingData.smallCoreTasks, tilingData.bigCoreCount, tilingData.schedule);
    if constexpr (!FullRows) {
        if (tilingData.schedule == kScheduleRank3Perm120 ||
            tilingData.schedule == kScheduleRank4Perm0231 ||
            tilingData.schedule == kScheduleRank4Perm0231Wide ||
            tilingData.schedule == kScheduleRank4Perm0312Wide ||
            tilingData.schedule == kScheduleRank4Perm0312 ||
            tilingData.schedule == kScheduleGenericNd ||
            tilingData.schedule == kScheduleGenericRowsReordered ||
            tilingData.schedule == kSchedulePlaneNd ||
            tilingData.schedule == kSchedulePlaneCompactB16 ||
            tilingData.schedule == kSchedulePlaneB32 ||
            tilingData.schedule == kSchedulePlaneB32Wide ||
            tilingData.schedule == kSchedulePlaneB32WideTail ||
            tilingData.schedule == kSchedulePlaneB32Grouped ||
            tilingData.schedule == kScheduleRank5GroupedB32 ||
            tilingData.schedule == kScheduleReverseNd) {
            kernel.InitNdMetadata(tilingData);
        }
    }
    if (localTasks != 0) {
        kernel.ProcessBlockedFullRows(localTasks);
        PipeBarrier<PIPE_ALL>();
    } else {
        kernel.template Process<FullRows>();
    }
}

template <typename T>
__aicore__ inline void PackBlockedRows(GM_ADDR workspace, GM_ADDR output,
                                        const TransposeTilingData& tilingData)
{
    const int64_t rows = tilingData.dim0;
    constexpr int64_t columnsPerTask = 10;
    const int64_t paddedRows = (rows + 63) / 64 * 64;
    const int64_t colTiles = (tilingData.dim1 + columnsPerTask - 1) / columnsPerTask;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;
    TQue<QuePosition::VECOUT, 2> outputQueue;
    TBuf<QuePosition::VECCALC> indexBuffer;
    pipe.InitBuffer(inputQueue, 2, columnsPerTask * paddedRows * sizeof(T));
    pipe.InitBuffer(outputQueue, 2, columnsPerTask * paddedRows * sizeof(T));
    pipe.InitBuffer(indexBuffer, columnsPerTask * paddedRows * sizeof(uint32_t));
    LocalTensor<int32_t> offsets = indexBuffer.Get<int32_t>();
    LocalTensor<float> scratch = outputQueue.AllocTensor<float>();
    LocalTensor<int32_t> columnIndex = outputQueue.AllocTensor<int32_t>();

    const int64_t patternCols = rows % 2 != 0 ? 8 : rows % 4 != 0 ? 4 : 2;
    const uint32_t indexCount = patternCols * rows;
    CreateVecIndex(offsets, 0, indexCount);
    PipeBarrier<PIPE_V>();
    Cast(scratch, offsets, RoundMode::CAST_NONE, indexCount);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, 1.0f / static_cast<float>(rows), indexCount);
    Adds(scratch, scratch, 1.0e-4f, indexCount);
    PipeBarrier<PIPE_V>();
    Cast(columnIndex, scratch, RoundMode::CAST_FLOOR, indexCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> temp = scratch.ReinterpretCast<int32_t>();
    Muls(temp, columnIndex, static_cast<int32_t>(rows), indexCount);
    PipeBarrier<PIPE_V>();
    Sub(offsets, offsets, temp, indexCount);
    PipeBarrier<PIPE_V>();
    ShiftRight(temp, offsets, static_cast<int32_t>(6), indexCount);
    PipeBarrier<PIPE_V>();
    Muls(temp, temp, static_cast<int32_t>((columnsPerTask - 1) * 64), indexCount);
    Muls(columnIndex, columnIndex, static_cast<int32_t>(64), indexCount);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, temp, indexCount);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, columnIndex, indexCount);
    PipeBarrier<PIPE_V>();
    Muls(offsets, offsets, static_cast<int32_t>(sizeof(T)), indexCount);
    PipeBarrier<PIPE_V>();
    for (int64_t column = patternCols; column < columnsPerTask; column += patternCols) {
        const int64_t count = columnsPerTask - column < patternCols ? columnsPerTask - column : patternCols;
        Adds(offsets[column * rows], offsets, static_cast<int32_t>(column * 64 * sizeof(T)),
             static_cast<uint32_t>(count * rows));
    }
    PipeBarrier<PIPE_V>();
    outputQueue.FreeTensor(scratch);
    outputQueue.FreeTensor(columnIndex);
    GlobalTensor<T> source;
    GlobalTensor<T> destination;
    source.SetGlobalBuffer((__gm__ T*)workspace);
    destination.SetGlobalBuffer((__gm__ T*)output);
    GlobalTensor<T> cachedDestination;
    cachedDestination.SetGlobalBuffer((__gm__ T*)output);

    destination.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    for (int64_t task = -1; task < colTiles; ++task) {
        if (task + 1 < colTiles) {
            const int64_t nextColumn = (task + 1) * columnsPerTask;
            const int64_t sourceBase = nextColumn / kCompactTileCols * paddedRows * kCompactTileCols +
                                      nextColumn % kCompactTileCols * 64;
            LocalTensor<T> next = inputQueue.AllocTensor<T>();
            DataCopyParams copyIn{static_cast<uint16_t>(paddedRows / 64),
                static_cast<uint16_t>(columnsPerTask * 64 * sizeof(T) / 32),
                static_cast<uint16_t>((kCompactTileCols - columnsPerTask) * 64 * sizeof(T) / 32), 0};
            DataCopy(next, source[sourceBase], copyIn);
            inputQueue.EnQue(next);
        }
        if (task < 0) {
            continue;
        }
        const int64_t column = task * columnsPerTask;
        const int64_t columns = tilingData.dim1 - column < columnsPerTask ?
                                tilingData.dim1 - column : columnsPerTask;
        LocalTensor<T> input = inputQueue.DeQue<T>();
        LocalTensor<T> local = outputQueue.AllocTensor<T>();
        Gather(local, input, offsets.ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(columns * rows));
        PipeBarrier<PIPE_V>();
        inputQueue.FreeTensor(input);
        outputQueue.EnQue(local);
        local = outputQueue.DeQue<T>();
        DataCopyExtParams copyOut{1, static_cast<uint32_t>(columns * rows * sizeof(T)), 0, 0, 0};
        if (task < (colTiles * 25 + 99) / 100) {
            DataCopyPad(cachedDestination[column * rows], local, copyOut);
        } else {
            DataCopyPad(destination[column * rows], local, copyOut);
        }
        outputQueue.FreeTensor(local);
    }
}

template <typename T>
__aicore__ inline void RunFullRows(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                   const TransposeTilingData& tilingData)
{
    if (tilingData.blockRows == kCompactFullRows && tilingData.dim0 <= 768 && tilingData.dim0 % 8 != 0) {
        const int64_t planeElements = tilingData.dim0 * tilingData.dim1;
        const int64_t planes = tilingData.totalLength / planeElements;
        const int64_t rowTiles = (tilingData.dim0 + 63) / 64;
        const int64_t workspaceElements = rowTiles * 64 *
            ((tilingData.dim1 + kCompactTileCols - 1) / kCompactTileCols * kCompactTileCols);
        GM_ADDR temporary = GetUserWorkspace(workspace) + GetBlockIdx() * workspaceElements * sizeof(T);
        TransposeTilingData local = tilingData;
        local.totalLength = planeElements;

        for (int64_t plane = GetBlockIdx(); plane < planes; plane += GetBlockNum()) {
            RunTranspose<T, true>(input + plane * planeElements * sizeof(T), temporary, local, rowTiles);
            PackBlockedRows<T>(temporary, output + plane * planeElements * sizeof(T), local);
        }
    } else {
        RunTranspose<T, true>(input, output, tilingData);
    }
}

template <typename T>
__aicore__ inline void StageRank5Planes(GM_ADDR input, GM_ADDR workspace,
                                       const TransposeTilingData& data)
{
    constexpr int64_t tileCols = 480;
    const int64_t rows = data.in2;
    const int64_t paddedRows = (rows + 15) / 16 * 16;
    const int64_t cols = data.in3 * data.in4;
    const int64_t colTiles = (cols + tileCols - 1) / tileCols;
    const int64_t tasks = data.in0 * data.in1 * colTiles;
    const int64_t core = GetBlockIdx();
    const int64_t small = tasks / GetBlockNum();
    const int64_t extra = tasks % GetBlockNum();
    const int64_t first = core * small + (core < extra ? core : extra);
    const int64_t end = first + small + (core < extra ? 1 : 0);
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> in;
    TQue<QuePosition::VECOUT, 2> out;
    pipe.InitBuffer(in, 1, paddedRows * tileCols * sizeof(T));
    pipe.InitBuffer(out, 2, paddedRows * tileCols * sizeof(T));
    GlobalTensor<T> source;
    GlobalTensor<T> destination;
    source.SetGlobalBuffer((__gm__ T*)input);
    destination.SetGlobalBuffer((__gm__ T*)workspace);
    source.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    for (int64_t task = first; task < end; ++task) {
        const int64_t plane = task / colTiles;
        const int64_t column = task % colTiles * tileCols;
        const int64_t validCols = cols - column < tileCols ? cols - column : tileCols;
        const int64_t alignedCols = (validCols + 7) / 8 * 8;
        LocalTensor<T> x = in.AllocTensor<T>();
        DataCopyExtParams load{static_cast<uint16_t>(rows),
            static_cast<uint32_t>(validCols * sizeof(T)),
            static_cast<uint32_t>((cols - validCols) * sizeof(T)),
            static_cast<uint32_t>((tileCols - alignedCols) * sizeof(T) / 32), 0};
        DataCopyPad(x, source[plane * rows * cols + column], load,
            DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>(alignedCols - validCols), 0});
        in.EnQue(x);
        x = in.DeQue<T>();
        LocalTensor<T> y = out.AllocTensor<T>();
        const uint64_t srcBase = reinterpret_cast<uint64_t>(x.GetPhyAddr());
        const uint64_t dstBase = reinterpret_cast<uint64_t>(y.GetPhyAddr());
        for (int64_t row = 0; row < paddedRows; row += 16) {
            uint64_t src[16];
            uint64_t dst[16];
            for (int64_t lane = 0; lane < 16; ++lane) {
                src[lane] = srcBase + (row + lane) * tileCols * sizeof(T);
                dst[lane] = dstBase + (row + lane / 2 * paddedRows + lane % 2 * 8) * sizeof(T);
            }
            TransDataTo5HD<uint32_t>(dst, src, {false, false,
                static_cast<uint8_t>(tileCols / 8), static_cast<uint16_t>(paddedRows), 1});
        }
        PipeBarrier<PIPE_V>();
        in.FreeTensor(x);
        out.EnQue(y);
        y = out.DeQue<T>();
        const int64_t reorderedPlane = plane % data.in1 * data.in0 + plane / data.in1;
        DataCopy(destination[(reorderedPlane * colTiles + task % colTiles) * paddedRows * tileCols],
                 y, static_cast<uint32_t>(paddedRows * tileCols));
        out.FreeTensor(y);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <typename T>
__aicore__ inline void PackRank5Planes(GM_ADDR workspace, GM_ADDR output,
                                      const TransposeTilingData& data)
{
    constexpr int64_t columnsPerTask = 8;
    const int64_t planes = data.in0 * data.in1;
    const int64_t rows = data.in2;
    const int64_t paddedRows = (rows + 15) / 16 * 16;
    const int64_t outputRows = planes * rows;
    const int64_t cols = data.in3 * data.in4;
    const int64_t planeElements = (cols + 479) / 480 * 480 * paddedRows;
    const int64_t tasks = (cols + columnsPerTask - 1) / columnsPerTask;
    const int64_t core = GetBlockIdx();
    const int64_t small = tasks / GetBlockNum();
    const int64_t extra = tasks % GetBlockNum();
    const int64_t first = core * small + (core < extra ? core : extra);
    const int64_t end = first + small + (core < extra ? 1 : 0);
    const uint32_t fullCount = columnsPerTask * outputRows;
    const int64_t period = outputRows % 8 == 0 ? 1 : outputRows % 4 == 0 ? 2 : outputRows % 2 == 0 ? 4 : 8;
    const uint32_t count = period * outputRows;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> in;
    TQue<QuePosition::VECOUT, 2> out;
    TBuf<QuePosition::VECCALC> indexBuffer;
    pipe.InitBuffer(in, 2, columnsPerTask * planes * paddedRows * sizeof(T));
    pipe.InitBuffer(out, 2, (fullCount + 7) / 8 * 8 * sizeof(T));
    pipe.InitBuffer(indexBuffer, (fullCount + 7) / 8 * 8 * sizeof(uint32_t));
    LocalTensor<int32_t> offsets = indexBuffer.Get<int32_t>();
    LocalTensor<float> scratch = out.AllocTensor<float>();
    LocalTensor<int32_t> temp = out.AllocTensor<int32_t>();
    LocalTensor<int32_t> columnIndex = in.AllocTensor<int32_t>();
    CreateVecIndex(offsets, 0, count);
    PipeBarrier<PIPE_V>();
    Cast(scratch, offsets, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, 1.0f / static_cast<float>(outputRows), count);
    Adds(scratch, scratch, 1.0e-4f, count);
    PipeBarrier<PIPE_V>();
    Cast(columnIndex, scratch, RoundMode::CAST_FLOOR, count);
    PipeBarrier<PIPE_V>();
    Muls(temp, columnIndex, static_cast<int32_t>(outputRows), count);
    PipeBarrier<PIPE_V>();
    Sub(offsets, offsets, temp, count);
    PipeBarrier<PIPE_V>();
    Cast(scratch, offsets, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, 1.0f / static_cast<float>(rows), count);
    Adds(scratch, scratch, 1.0e-4f, count);
    PipeBarrier<PIPE_V>();
    Cast(temp, scratch, RoundMode::CAST_FLOOR, count);
    PipeBarrier<PIPE_V>();
    Muls(temp, temp, static_cast<int32_t>(columnsPerTask * paddedRows - rows), count);
    Muls(columnIndex, columnIndex, static_cast<int32_t>(paddedRows), count);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, temp, count);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, columnIndex, count);
    PipeBarrier<PIPE_V>();
    Muls(offsets, offsets, static_cast<int32_t>(sizeof(T)), count);
    PipeBarrier<PIPE_V>();
    for (int64_t column = period; column < columnsPerTask; column += period) {
        Adds(offsets[column * outputRows], offsets, static_cast<int32_t>(column * paddedRows * sizeof(T)), count);
    }
    PipeBarrier<PIPE_V>();
    in.FreeTensor(columnIndex);
    out.FreeTensor(scratch);
    out.FreeTensor(temp);
    GlobalTensor<T> source;
    GlobalTensor<T> destination;
    source.SetGlobalBuffer((__gm__ T*)workspace);
    destination.SetGlobalBuffer((__gm__ T*)output);
    for (int64_t task = first - 1; task < end; ++task) {
        if (task + 1 < end) {
            const int64_t column = (task + 1) * columnsPerTask;
            LocalTensor<T> next = in.AllocTensor<T>();
            DataCopyExtParams load{static_cast<uint16_t>(planes),
                static_cast<uint32_t>(columnsPerTask * paddedRows * sizeof(T)),
                static_cast<uint32_t>((planeElements - columnsPerTask * paddedRows) * sizeof(T)), 0, 0};
            DataCopyPad(next, source[column * paddedRows], load, DataCopyPadExtParams<T>{false, 0, 0, 0});
            in.EnQue(next);
        }
        if (task < first) {
            continue;
        }
        const int64_t column = task * columnsPerTask;
        const int64_t validCols = cols - column < columnsPerTask ? cols - column : columnsPerTask;
        LocalTensor<T> x = in.DeQue<T>();
        LocalTensor<T> y = out.AllocTensor<T>();
        Gather(y, x, offsets.ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(validCols * outputRows));
        PipeBarrier<PIPE_V>();
        in.FreeTensor(x);
        out.EnQue(y);
        y = out.DeQue<T>();
        DataCopyPad(destination[column * outputRows], y,
            DataCopyExtParams{1, static_cast<uint32_t>(validCols * outputRows * sizeof(T)), 0, 0, 0});
        out.FreeTensor(y);
    }
}

template <typename T>
__aicore__ inline void RunRank5Cached(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                     const TransposeTilingData& data)
{
    GM_ADDR temporary = GetUserWorkspace(workspace);
    StageRank5Planes<T>(input, temporary, data);
    PackRank5Planes<T>(temporary, output, data);
}

}

extern "C" __global__ __aicore__ void transpose(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        RunTranspose<half>(input, output, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RunTranspose<float>(input, output, tilingData);
    } else if (TILING_KEY_IS(2)) {
        RunTranspose<int32_t>(input, output, tilingData);
    } else if (TILING_KEY_IS(3)) {
        RunTranspose<int8_t>(input, output, tilingData);
    } else if (TILING_KEY_IS(4)) {
        RunFullRows<float>(input, output, workspace, tilingData);
    } else if (TILING_KEY_IS(5)) {
        RunFullRows<int32_t>(input, output, workspace, tilingData);
    } else if (TILING_KEY_IS(6)) {
        RunRank5Cached<float>(input, output, workspace, tilingData);
    } else if (TILING_KEY_IS(7)) {
        RunRank5Cached<int32_t>(input, output, workspace, tilingData);
    }
}
