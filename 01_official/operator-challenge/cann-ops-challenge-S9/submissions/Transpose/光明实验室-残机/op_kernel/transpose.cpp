// Transpose 设备侧实现：根据 Host 下发的 mode 和 tiling key 选择搬运/转置路径。
// 索引默认以元素为单位；DataCopyExtParams 的 GM 步长和 blockLen 以字节为单位，
// UB 步长以 32B 数据块为单位；Gather 索引最终转换为字节偏移。
// KernelTranspose 负责通用路径和各类二维/分组转置；
// KernelTransposeAlignedWorkspace 负责先转置到带间隔的 scratch，再压紧并对齐写回。
// TPipe/TQue 路径由队列保护数据依赖；静态 UB 路径用显式事件保护槽位复用。
// 不使用 ShapeInfo，因此在包含 Ascend C 头文件之前关闭其默认维度存储。
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kAlignBytes = 32;
// mode 编号必须与 op_host/transpose.cpp 一致；它与入口的 tiling key 含义不同。
constexpr uint32_t kModeBlockCopy = 0;
constexpr uint32_t kModeGeneric = 1;
constexpr uint32_t kModeTranspose021Fp16 = 2;
constexpr uint32_t kModeTranspose2DMacro = 3;
constexpr uint32_t kModeTransposeFlattenTiled = 4;
constexpr uint32_t kModeLastAxisTransposeTiled = 5;
constexpr uint32_t kModeReverse3DSuffixGather = 6;
constexpr uint32_t kModeTranspose021Macro = 7;
constexpr uint32_t kModeTranspose021Batch = 8;
constexpr uint32_t kModeTranspose021ResidentOutput = 9;
// 以下是按数据位宽定义的基础转置块；B8/B32 分别表示每元素 8/32 位。
constexpr uint32_t kTile2D16 = 16;
constexpr uint32_t kTileB8Rows = 16;
constexpr uint32_t kTileB8Cols = 32;
constexpr uint32_t kTileB32Rows = 16;
constexpr uint32_t kTileB32Cols = 8;
// 静态 UB 槽位的容量和地址间隔；按路径分别使用，不能把不同布局的槽位混用。
constexpr uint32_t kStaticTileSlotBytes = 1024;
constexpr uint32_t kStaticMacroSlotBytes = 32 * 1024;
constexpr uint32_t kLastAxisStaticSlotBytes = 48 * 1024;
constexpr uint32_t kStaticTripleSlotBytes = 61 * 1024;
constexpr uint32_t kTranspose021B32WideRowBytes = 3 * 1024;
constexpr uint32_t kTranspose021B32LegacyInputSlotBytes = 76 * 1024;
constexpr uint32_t kTranspose021B32LegacyOutputSlotBytes = 32 * 1024;
constexpr uint32_t kTranspose021B32CoreStridedInputSlotBytes = 76 * 1024;
constexpr uint32_t kTranspose021B32CoreStridedOutputSlotBytes = 40 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided48Rows = 48;
// 32 位 021 按核交错路径的整缓存行写布局：64 行对应每列 256B。
// 输出行距和基址满足 128B 对齐时才选择该缓存策略；实际 tile 行数还受输入槽容量限制。
constexpr uint32_t kTranspose021B32CoreStrided64Rows = 64;
constexpr uint32_t kTranspose021B32CoreStrided64InputSlotBytes = 132 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided64OutputPingOffsetBytes = 132 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided64OutputSlotBytes = 30 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided64OutputPongOffsetBytes = 162 * 1024;

constexpr uint32_t kCacheLineBytes = 128;
// 32 个 32 位元素共 128B，是该写布局采用的整缓存行行数步长。
constexpr uint32_t kWholeLineStoreRowStep = 32;
constexpr uint32_t kUbBytes = 192 * 1024;
// 32 位非对齐行采用 packed 输入：连续读入多行，再用 Gather 转成列段，减少逐行读尾部请求。
// 每个输出分段最多 64 列；这是本路径的粒度选择，不能推广成所有形状的最优参数。
constexpr uint32_t kTranspose021B32GatherChunkCols = 64;
// 完整输入块的行数取 8 的倍数，因此 rowStep*sizeof(T) 为 32B 的倍数，整块读取可对齐。
constexpr uint32_t kTranspose021B32GatherRowStep = 8;
// 根据历史 B3 列宽扫描建立的 bank 冲突筛选参数；属于路径启发式，不是公开硬件规格。
constexpr uint32_t kUbBankCount = 32;
constexpr uint32_t kMinBankSpreadRows = 8;
constexpr uint32_t kTranspose021B32CoreStrided48InputOffsetBytes = 4 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided48InputSlotBytes = 100 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided48OutputPingOffsetBytes = 104 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided48OutputPongOffsetBytes = 148 * 1024;
constexpr uint32_t kTranspose021B32CoreStrided48OutputSlotBytes = 44 * 1024;
constexpr uint32_t kTranspose021B32PipelineSlotBytes = 64 * 1024;
constexpr uint32_t kTranspose021B32FullRowInputSlotBytes = 128 * 1024;
constexpr uint32_t kTranspose021B32FullRowOutputSlotBytes = 64 * 1024;
constexpr uint32_t kStaticBlockCopySlotBytes = 64 * 1024;
constexpr uint32_t kStaticUnalignedBlockCopySlotBytes = 40 * 1024;
constexpr uint32_t kLastAxisB32FixedGroup = 4;
constexpr uint32_t kLastAxisRank5DGroup = 16;
constexpr uint32_t kLastAxisRank5AGroup = 4;
constexpr uint32_t kLastAxisRank5ADGroup = 4;
constexpr uint32_t kLastAxisRank5MinAForADGroup = 8;
constexpr uint32_t kLastAxisRank5FullADGroup = 5;
// FullA 分组把 A×R 展平后每次读 64 行，输出在 UB 累积成较长的连续段。
// 输入行额外增加 8 个 32 位元素，即 32B 间隔，用于改变转置读取的 UB bank 分布。
constexpr uint32_t kLastAxisRank5FullARowGroup = 64;
constexpr uint32_t kLastAxisRank5FullASourceRowPadCols = 8;
constexpr uint32_t kLastAxisRank5FullAMin = 8;
constexpr uint32_t kLastAxisRank5FullAMax = 12;
constexpr uint32_t kLastAxisRank5MinRC = 17;
constexpr uint32_t kLastAxisRank5MaxRC = 25;
constexpr uint32_t kLastAxisRank5PaddedRows = 32;
constexpr uint32_t kLastAxisRank5MaxGroupedCols =
    kLastAxisRank5DGroup * kLastAxisRank5MaxRC;
constexpr uint32_t kLastAxisRank5SourceSlotBytes =
    kLastAxisRank5MaxRC * kLastAxisRank5MaxGroupedCols * sizeof(int32_t);
constexpr uint32_t kLastAxisRank5OutputSlotBytes =
    kLastAxisRank5PaddedRows * kLastAxisRank5MaxGroupedCols * sizeof(int32_t);
constexpr uint32_t kLastAxisRank5DstPingOffsetBytes = kLastAxisRank5SourceSlotBytes;
constexpr uint32_t kLastAxisRank5SrcPongOffsetBytes =
    kLastAxisRank5DstPingOffsetBytes + kLastAxisRank5OutputSlotBytes;
constexpr uint32_t kLastAxisRank5DstPongOffsetBytes =
    kLastAxisRank5SrcPongOffsetBytes + kLastAxisRank5SourceSlotBytes;
constexpr uint32_t kLastAxisRank5ADSourceSlotBytes = 41 * 1024;
constexpr uint32_t kLastAxisRank5ADOutputSlotBytes = 50 * 1024;
constexpr uint32_t kLastAxisRank5ADDstPingOffsetBytes = kLastAxisRank5ADSourceSlotBytes;
constexpr uint32_t kLastAxisRank5ADSrcPongOffsetBytes =
    kLastAxisRank5ADDstPingOffsetBytes + kLastAxisRank5ADOutputSlotBytes;
constexpr uint32_t kLastAxisRank5ADDstPongOffsetBytes =
    kLastAxisRank5ADSrcPongOffsetBytes + kLastAxisRank5ADSourceSlotBytes;
constexpr uint32_t kLastAxisRank5FullASourceSlotBytes = 68 * 512;
constexpr uint32_t kLastAxisRank5FullAOutputOffsetBytes = kLastAxisRank5FullASourceSlotBytes;
constexpr uint32_t kLastAxisRank5FullAOutputSlotBytes = 304 * 512;
constexpr uint32_t kLastAxisRank5FullAOutputPitchRows = 304;
constexpr uint32_t kDataCopyPadMaxBlockCount = 4095;
constexpr uint32_t kTranspose021GatherCols = 32;
constexpr uint32_t kTranspose021StreamCols = 16;
constexpr uint64_t kReverse3DInputMajorThresholdBytes = 192ULL * 1024 * 1024;
constexpr uint32_t kReverse3DInputRowSkewBytes = 32;
constexpr uint32_t kReverse3DGatherGroupBatch = 8;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
    return AlignUp(bytes, kAlignBytes);
}

template <typename T>
__aicore__ inline void CopyInPad(GlobalTensor<T>& inputGm, uint32_t offset, LocalTensor<T>& inputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T>& outputGm, uint32_t offset, LocalTensor<T>& outputLocal, uint32_t len)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[offset], outputLocal, copyParams);
}

// 普通入口的共享实现。kInputL2Bypass 保留既有 key1 输入策略；key0 中只有
// Reverse3DSuffixGather 的一次性输入流按实测关闭读方向 L2。具体算法仍由 mode_ 决定，
// 输出缓存提示还需检查物理地址和写入跨度。
template <typename T, bool kInputL2Bypass>
class KernelTranspose {
public:
    // 读取 Host 参数并建立实际输入/输出步长；只为命中的执行路径分配 UB 缓冲。
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const TransposeTilingData& tilingData)
    {
        rank_ = tilingData.rank;
        mode_ = tilingData.mode;
        totalLength_ = tilingData.totalLength;
        outputBufferLength_ = tilingData.outputBufferLength;
        taskLength_ = tilingData.taskLength;
        directDimA_ = tilingData.directDimA;
        directDimB_ = tilingData.directDimB;
        coreSplitA_ = tilingData.coreSplitA;
        coreSplitB_ = tilingData.coreSplitB;
        tileLength_ = tilingData.tileLength;
        innerBlock_ = tilingData.innerBlock;
        swapAxis_ = tilingData.swapAxis;
        swapOuter_ = tilingData.swapOuter;
        swapDimA_ = tilingData.swapDimA;
        swapDimB_ = tilingData.swapDimB;
        swapInner_ = tilingData.swapInner;
        swapTileA_ = tilingData.swapTileA;
        swapTileB_ = tilingData.swapTileB;
        for (uint32_t i = 0; i < kMaxRank; ++i) {
            inputDims_[i] = tilingData.inputDims[i];
            perm_[i] = tilingData.perm[i];
        }
        uint32_t inputStride = 1;
        uint32_t outputStride = 1;
        for (int32_t i = static_cast<int32_t>(rank_) - 1; i >= 0; --i) {
            inputStrides_[i] = inputStride;
            inputStride *= inputDims_[i];
            outputStrides_[i] = outputStride;
            outputDims_[i] = inputDims_[perm_[i]];
            outputStride *= outputDims_[i];
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalLength_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, outputBufferLength_);
        outputLineAligned_ =
            (reinterpret_cast<uint64_t>(output) % kCacheLineBytes) == 0;
        if constexpr (kInputL2Bypass) {
            inputGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else if (mode_ == kModeReverse3DSuffixGather) {
            // 该结构族的输入只顺序消费一次；方向化 READ 禁止读分配，不改变写方向或其他模式。
            inputGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        // 避免对部分缓存行的写入直接旁路 L2；本路径仅在整行写条件满足时关闭输出缓存。
        if (UseTranspose021B32AlignedStore()) {
            outputGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (mode_ == kModeReverse3DSuffixGather) {
            const uint32_t packedInputRowBytes = AlignUpBytes(swapTileB_ * swapInner_ * sizeof(T));
            const bool useInputRowSkew =
                static_cast<uint64_t>(totalLength_) * sizeof(T) > kReverse3DInputMajorThresholdBytes &&
                ((packedInputRowBytes / kAlignBytes) & 1U) != 0;
            const uint32_t inputRowBytes = packedInputRowBytes +
                (useInputRowSkew ? kReverse3DInputRowSkewBytes : 0);
            const uint32_t inputBytes = swapTileA_ * inputRowBytes;
            if constexpr (sizeof(T) == 1) {
                pipe->InitBuffer(batchCopyQueue_, 2, AlignUpBytes(inputBytes));
            } else {
                pipe->InitBuffer(batchInQueue_, 2, AlignUpBytes(inputBytes));
                const uint32_t groupIndexElems = swapTileA_ * swapInner_;
                const uint32_t alignedGroupIndexElems =
                    AlignUpBytes(groupIndexElems * sizeof(uint32_t)) / sizeof(uint32_t);
                const uint32_t paddedOutputElems =
                    AlignUpBytes(swapTileA_ * swapInner_ * sizeof(T)) / sizeof(T);
                const bool outputRowsAligned =
                    paddedOutputElems == swapTileA_ * swapInner_;
                const uint32_t offsetGroupStride = alignedGroupIndexElems > paddedOutputElems ?
                    alignedGroupIndexElems : paddedOutputElems;
                const bool useBatchedGatherIndex =
                    static_cast<uint64_t>(totalLength_) * sizeof(T) > kReverse3DInputMajorThresholdBytes;
                const uint32_t gatherGroupBatch = kReverse3DGatherGroupBatch;
                const uint32_t offsetGroups = outputRowsAligned ?
                    (useBatchedGatherIndex ? MinU32(swapTileB_, gatherGroupBatch) : swapTileB_) : 1;
                const uint32_t outputBytes = swapTileB_ *
                    AlignUpBytes(swapTileA_ * swapInner_ * sizeof(T));
                pipe->InitBuffer(batchOutQueue_, 2, AlignUpBytes(outputBytes));
                pipe->InitBuffer(gatherIndexBuf_,
                                 AlignUpBytes(offsetGroups * offsetGroupStride * sizeof(uint32_t)));
            }
        } else if (mode_ == kModeBlockCopy || mode_ == kModeGeneric) {
            pipe->InitBuffer(copyQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
        } else {
            const uint32_t directTileBytes = swapTileA_ * swapTileB_ * sizeof(T);
            if (mode_ == kModeLastAxisTransposeTiled &&
                directTileBytes <= kLastAxisStaticSlotBytes) {
                // 该分支直接绑定四个静态 UB 槽位，无需在此重复分配队列缓冲。
            } else if (mode_ == kModeTranspose021Batch) {
                pipe->InitBuffer(tileInQueue_, swapTileB_, AlignUpBytes(tileLength_ * sizeof(T)));
                pipe->InitBuffer(tileOutQueue_, swapTileB_, AlignUpBytes(tileLength_ * sizeof(T)));
            } else if (mode_ == kModeTranspose021ResidentOutput) {
                // 输入块和常驻输出直接绑定静态 UB 地址。
            } else if (UseTranspose021B32PackedGather()) {
                const uint32_t gatherTileRows = PackedGatherTileRows();
                pipe->InitBuffer(batchInQueue_, 1,
                                 AlignUpBytes(gatherTileRows * inputDims_[2] * sizeof(T)));
                pipe->InitBuffer(batchOutQueue_, 2,
                                 AlignUpBytes(kTranspose021B32GatherChunkCols * gatherTileRows *
                                              sizeof(T)));
                pipe->InitBuffer(gatherIndexBuf_,
                                 AlignUpBytes(kTranspose021GatherCols * gatherTileRows *
                                              sizeof(uint32_t)));
            } else if (UseTranspose021B32FullRow()) {
                // 该路径直接绑定两份输入槽和一份输出槽。
            } else if (mode_ == kModeTranspose021Fp16 && inputDims_[2] < swapTileB_) {
                pipe->InitBuffer(batchInQueue_, 2, AlignUpBytes(tileLength_ * sizeof(T)));
                if constexpr (sizeof(T) == 1) {
                    const uint32_t outputElems = swapTileA_ > 512 ?
                        kTranspose021StreamCols * swapTileA_ : tileLength_;
                    pipe->InitBuffer(batchOutQueue_, 2, AlignUpBytes(outputElems * sizeof(T)));
                    const uint32_t offsetBytes = AlignUpBytes(swapTileA_ * sizeof(uint32_t));
                    const uint32_t wordBytes = AlignUpBytes(swapTileA_ * sizeof(uint16_t));
                    const uint32_t maskBytes = AlignUpBytes((swapTileA_ / 2) * sizeof(uint16_t));
                    pipe->InitBuffer(gatherIndexBuf_, 2 * offsetBytes + 6 * wordBytes + 2 * maskBytes);
                } else {
                    pipe->InitBuffer(batchOutQueue_, 2, AlignUpBytes(tileLength_ * sizeof(T)));
                    pipe->InitBuffer(gatherIndexBuf_, AlignUpBytes(
                        kTranspose021GatherCols * swapTileA_ * sizeof(uint32_t)));
                }
            } else if (mode_ == kModeTranspose021Fp16 && sizeof(T) == 1 &&
                       tileLength_ * sizeof(T) <= kStaticTripleSlotBytes) {
                // 常规 INT8 路径直接使用三份静态 UB 槽位。
            } else {
                if constexpr (sizeof(T) != 1) {
                    pipe->InitBuffer(tileInQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
                    pipe->InitBuffer(tileOutQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
                } else {
                    pipe->InitBuffer(tileInQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
                    pipe->InitBuffer(tileOutQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
                    const bool flattenInt8Macro =
                        mode_ == kModeTransposeFlattenTiled && swapTileB_ > kTileB8Cols;
                    if (mode_ != kModeTranspose2DMacro && mode_ != kModeTranspose021Macro &&
                        mode_ != kModeTranspose021Batch && !flattenInt8Macro) {
                        pipe->InitBuffer(copyQueue_, 1, AlignUpBytes(tileLength_ * sizeof(T)));
                    }
                }
            }
        }
    }

    // task 的单位随 mode 改变，可以是一段连续数据、一个二维 tile 或一个完整 batch。
    // 先确定核拥有的任务区间，再调用相应路径；按核交错的路径额外传递 taskStride。
    __aicore__ inline void Process()
    {
        if (directDimA_ != 0 && directDimB_ != 0 && coreSplitA_ != 0 && coreSplitB_ != 0) {
            ProcessDirect2D();
            return;
        }
        uint32_t taskStart = GetBlockIdx() * taskLength_;
        uint32_t taskTotal = 0;
        if (mode_ == kModeBlockCopy) {
            taskTotal = totalLength_ / innerBlock_;
        } else if (mode_ == kModeReverse3DSuffixGather) {
            taskTotal = swapOuter_ * (swapDimA_ / swapTileA_) *
                        ((swapDimB_ + swapTileB_ - 1) / swapTileB_);
        } else if (mode_ == kModeGeneric) {
            taskTotal = totalLength_;
        } else if (mode_ == kModeTranspose2DMacro) {
            taskTotal = (inputDims_[0] / swapTileA_) * (inputDims_[1] / swapTileB_);
        } else if (mode_ == kModeTranspose021Macro) {
            taskTotal = inputDims_[0] * (inputDims_[1] / swapTileA_) * (inputDims_[2] / swapTileB_);
        } else if (mode_ == kModeTranspose021Batch) {
            taskTotal = inputDims_[0];
        } else if (mode_ == kModeTranspose021ResidentOutput) {
            taskTotal = inputDims_[0];
        } else if (mode_ == kModeTranspose021Fp16) {
            if constexpr (sizeof(T) == 1) {
                taskTotal = inputDims_[0] * ((inputDims_[1] + kTileB8Cols - 1) / kTileB8Cols) *
                            ((inputDims_[2] + kTileB8Cols - 1) / kTileB8Cols);
            } else if constexpr (sizeof(T) == 4) {
                taskTotal = inputDims_[0] * ((inputDims_[1] + kTileB32Rows - 1) / kTileB32Rows) *
                            ((inputDims_[2] + kTileB32Cols - 1) / kTileB32Cols);
            } else {
                taskTotal = inputDims_[0] * ((inputDims_[1] + kTile2D16 - 1) / kTile2D16) *
                            ((inputDims_[2] + kTile2D16 - 1) / kTile2D16);
            }
        } else if (mode_ == kModeTransposeFlattenTiled) {
            if constexpr (sizeof(T) == 1) {
                const uint32_t tileRows = swapTileB_ > kTileB8Cols ? swapTileA_ : kTileB8Cols;
                const uint32_t tileCols = swapTileB_ > kTileB8Cols ? swapTileB_ : kTileB8Cols;
                taskTotal = swapOuter_ * ((swapDimA_ + tileRows - 1) / tileRows) *
                            ((swapDimB_ + tileCols - 1) / tileCols);
            } else if constexpr (sizeof(T) == 4) {
                taskTotal = swapOuter_ * ((swapDimA_ + kTileB32Rows - 1) / kTileB32Rows) *
                            ((swapDimB_ + kTileB32Cols - 1) / kTileB32Cols);
            } else {
                taskTotal = swapOuter_ * ((swapDimA_ + kTile2D16 - 1) / kTile2D16) *
                            ((swapDimB_ + kTile2D16 - 1) / kTile2D16);
            }
        } else if (mode_ == kModeLastAxisTransposeTiled) {
            const uint32_t tileBlocksA = (swapDimA_ + swapTileA_ - 1) / swapTileA_;
            const uint32_t tileBlocksB = (swapDimB_ + swapTileB_ - 1) / swapTileB_;
            taskTotal = (totalLength_ / (swapDimA_ * swapDimB_)) * tileBlocksA * tileBlocksB;
        } else {
            return;
        }

        uint32_t taskStride = 1;
        uint32_t taskCount = 0;
        if (mode_ == kModeTranspose021Batch || mode_ == kModeTranspose021ResidentOutput) {
            GetSplitRange(inputDims_[0], GetBlockNum(), GetBlockIdx(), taskStart, taskCount);
        } else if (mode_ == kModeTranspose021Fp16 && sizeof(T) == 1 &&
            inputDims_[2] < swapTileB_) {
            uint32_t batchStart = 0;
            uint32_t batchCount = 0;
            GetSplitRange(inputDims_[0], GetBlockNum(), GetBlockIdx(), batchStart, batchCount);
            const uint32_t rowTiles = (inputDims_[1] + swapTileA_ - 1) / swapTileA_;
            taskStart = batchStart * rowTiles;
            taskCount = batchCount * rowTiles;
        } else if (mode_ == kModeReverse3DSuffixGather && sizeof(T) != 1 &&
            static_cast<uint64_t>(totalLength_) * sizeof(T) > kReverse3DInputMajorThresholdBytes) {
            taskStart = GetBlockIdx();
            taskStride = GetBlockNum();
            if (taskStart >= taskTotal) {
                return;
            }
            taskCount = (taskTotal - taskStart + taskStride - 1) / taskStride;
        } else if (UseTranspose021B32CoreStrided()) {
            const uint32_t coreStridedRows = CoreStridedTileRows();
            const uint32_t rowTiles =
                (inputDims_[1] + coreStridedRows - 1) / coreStridedRows;
            taskTotal = inputDims_[0] * rowTiles;
            taskStart = GetBlockIdx();
            taskStride = GetBlockNum();
            if (taskStart >= taskTotal) {
                return;
            }
            taskCount = (taskTotal - taskStart + taskStride - 1) / taskStride;
        } else {
            if (taskStart >= taskTotal) {
                return;
            }
            taskCount = taskTotal - taskStart;
            if (taskCount > taskLength_) {
                taskCount = taskLength_;
            }
        }

        if (mode_ == kModeBlockCopy) {
            ProcessBlockCopy(taskStart, taskCount);
        } else if (mode_ == kModeReverse3DSuffixGather) {
            if constexpr (sizeof(T) == 1) {
                ProcessReverse3DSuffixPadCopy(taskStart, taskCount);
            } else {
                ProcessReverse3DSuffixGather(taskStart, taskCount, taskStride);
            }
        } else if (mode_ == kModeGeneric) {
            ProcessGeneric(taskStart, taskCount);
        } else if (mode_ == kModeTranspose2DMacro) {
            if constexpr (sizeof(T) == 2) {
                if (swapTileA_ == 64 && swapTileB_ == 64) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 64, 64);
                } else if (swapTileA_ == 128 && swapTileB_ == 128) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 128, 128);
                } else if (swapTileA_ == 128 && swapTileB_ == 256) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 128, 256);
                } else if (swapTileA_ == 192 && swapTileB_ == 192) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 192, 192);
                } else if (swapTileA_ == 256 && swapTileB_ == 128) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 256, 128);
                } else if (swapTileA_ == 256 && swapTileB_ == 192) {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, 256, 192);
                } else {
                    ProcessTranspose2DMacroTile(taskStart, taskCount, swapTileA_, swapTileB_);
                }
            } else if constexpr (sizeof(T) == 4) {
                ProcessTranspose2DMacroTileB32(taskStart, taskCount, swapTileA_, swapTileB_);
            } else if constexpr (sizeof(T) == 1) {
                ProcessTranspose2DMacroTileB8(taskStart, taskCount, swapTileA_, swapTileB_);
            }
        } else if (mode_ == kModeTranspose021Macro) {
            ProcessTranspose021Macro(taskStart, taskCount, swapTileA_, swapTileB_);
        } else if (mode_ == kModeTranspose021Batch) {
            ProcessTranspose021Batch(taskStart, taskCount);
        } else if (mode_ == kModeTranspose021ResidentOutput) {
            ProcessTranspose021ResidentOutput(taskStart, taskCount);
        } else if (mode_ == kModeTranspose021Fp16) {
            taskStride_ = taskStride;
            ProcessTranspose021Tiled(taskStart, taskCount);
        } else if (mode_ == kModeTransposeFlattenTiled) {
            ProcessTransposeBatchFlattenTiled(taskStart, taskCount);
        } else {
            ProcessLastAxisTransposeTiled(taskStart, taskCount);
        }
    }

private:
    // 将连续任务均分给各核，余数分给前面的核；返回的区间互不重叠。
    __aicore__ inline void GetSplitRange(uint32_t total, uint32_t parts, uint32_t index,
                                         uint32_t& start, uint32_t& count)
    {
        const uint32_t base = total / parts;
        const uint32_t remainder = total - base * parts;
        if (index < remainder) {
            count = base + 1;
            start = index * count;
        } else {
            count = base;
            start = remainder * (base + 1) + (index - remainder) * base;
        }
    }

    // 把核号解码为二维核网格坐标，分别切分两条任务轴，再选择块拷贝或末轴转置。
    __aicore__ inline void ProcessDirect2D()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t coreA = blockIdx / coreSplitB_;
        const uint32_t coreB = blockIdx - coreA * coreSplitB_;
        if (coreA >= coreSplitA_ || coreB >= coreSplitB_) {
            return;
        }
        uint32_t startA = 0;
        uint32_t countA = 0;
        uint32_t startB = 0;
        uint32_t countB = 0;
        GetSplitRange(directDimA_, coreSplitA_, coreA, startA, countA);
        GetSplitRange(directDimB_, coreSplitB_, coreB, startB, countB);
        if (countA == 0 || countB == 0) {
            return;
        }
        if (mode_ == kModeBlockCopy) {
            ProcessBlockCopyDirect(startA, countA, startB, countB);
        } else if (mode_ == kModeLastAxisTransposeTiled) {
            if (UseLastAxisRank5FullAGroup()) {
                ProcessLastAxisRank5FullAGroup(startA, countA);
            } else if (UseLastAxisRank5ADGroup()) {
                ProcessLastAxisRank5ADGroup(startA, countA);
            } else if (UseLastAxisRank5DGroup()) {
                ProcessLastAxisRank5DGroup(startA, countA);
            } else {
                ProcessLastAxisTransposeDirect(startA, countA, startB, countB);
            }
        }
    }

    __aicore__ inline bool UseLastAxisRank5FullAGroup() const
    {
        if constexpr (sizeof(T) != sizeof(int32_t)) {
            return false;
        }
        return mode_ == kModeLastAxisTransposeTiled && swapOuter_ == kLastAxisRank5FullADGroup &&
            rank_ == 5 && perm_[0] == 3 && perm_[1] == 4 && perm_[2] == 1 &&
            perm_[3] == 0 && perm_[4] == 2 &&
            inputDims_[0] >= kLastAxisRank5FullAMin && inputDims_[0] <= kLastAxisRank5FullAMax &&
            inputDims_[2] >= kLastAxisRank5MinRC && inputDims_[2] <= kLastAxisRank5MaxRC &&
            inputDims_[4] >= kLastAxisRank5MinRC && inputDims_[4] <= kLastAxisRank5MaxRC;
    }

    // 对 [A,B,R,D,C]→[D,C,B,A,R]，一个任务固定 B 并覆盖一组 D，合并全部 A×R。
    __aicore__ inline void GetLastAxisRank5FullAGroupTaskInfo(
        uint32_t task, uint32_t& b, uint32_t& dStart, uint32_t& outputBase,
        uint32_t& validCols, uint32_t& paddedCols)
    {
        const uint32_t aDim = inputDims_[0];
        const uint32_t bDim = inputDims_[1];
        const uint32_t rDim = inputDims_[2];
        const uint32_t dDim = inputDims_[3];
        const uint32_t cDim = inputDims_[4];
        const uint32_t dTiles = (dDim + kLastAxisRank5FullADGroup - 1) /
            kLastAxisRank5FullADGroup;
        b = task / dTiles;
        const uint32_t dTile = task - b * dTiles;
        dStart = dTile * kLastAxisRank5FullADGroup;
        const uint32_t dCount = MinU32(kLastAxisRank5FullADGroup, dDim - dStart);
        validCols = dCount * cDim;
        paddedCols = AlignUp(validCols, kTileB32Cols);
        outputBase = (((dStart * cDim) * bDim + b) * aDim) * rDim;
    }

    // A×R 在输出中连续、在输入中跨 A 时有跳跃；按 R 的边界分段读取并放入带行间隔的 UB。
    __aicore__ inline void CopyInLastAxisRank5FullARowGroup(
        LocalTensor<T>& srcLocal, uint32_t b, uint32_t dStart,
        uint32_t rowStart, uint32_t rowCount, uint32_t paddedCols, uint32_t validCols)
    {
        const uint32_t bDim = inputDims_[1];
        const uint32_t rDim = inputDims_[2];
        const uint32_t dDim = inputDims_[3];
        const uint32_t cDim = inputDims_[4];
        const uint32_t rowStride = dDim * cDim;
        const uint32_t sourcePitchCols = paddedCols + kLastAxisRank5FullASourceRowPadCols;
        uint32_t flatRow = rowStart;
        uint32_t localRow = 0;
        uint32_t remainingRows = rowCount;
        while (remainingRows > 0) {
            const uint32_t a = flatRow / rDim;
            const uint32_t r = flatRow - a * rDim;
            const uint32_t segmentRows = MinU32(remainingRows, rDim - r);
            const uint32_t inputBase =
                ((((a * bDim + b) * rDim + r) * dDim) + dStart) * cDim;
            LocalTensor<T> segmentLocal = srcLocal[localRow * sourcePitchCols];
            CopyInStaticTile(segmentLocal, inputBase, rowStride,
                             segmentRows, sourcePitchCols, segmentRows, validCols);
            flatRow += segmentRows;
            localRow += segmentRows;
            remainingRows -= segmentRows;
        }
    }

    // 用 16×8 的 32 位转置块写入常驻输出的指定行段；不足 16 行的地址指向有效行，写回时裁掉补齐部分。
    __aicore__ inline void TransposeLastAxisRank5FullARowGroup(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
        uint32_t dstRowOffset, uint32_t validRows, uint32_t paddedCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint32_t sourcePitchCols = paddedCols + kLastAxisRank5FullASourceRowPadCols;
        const uint32_t paddedRows = AlignUp(validRows, kTileB32Rows);
        const uint8_t repeatTimes = static_cast<uint8_t>(paddedCols / kTileB32Cols);
        const uint16_t dstRepStride = static_cast<uint16_t>(
            repeatTimes == 1 ? 0 : kLastAxisRank5FullAOutputPitchRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t rowBlock = 0; rowBlock < paddedRows / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                const uint32_t srcRow = srcRowBase + i;
                const uint32_t safeRow = srcRow < validRows ? srcRow : 0;
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[safeRow * sourcePitchCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[dstRowOffset + srcRowBase +
                             i * kLastAxisRank5FullAOutputPitchRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[dstRowOffset + srcRowBase +
                             i * kLastAxisRank5FullAOutputPitchRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    // 每个输出块包含完整 A×R，UB 行距为固定 pitch，GM 行距包含 B；仅搬运有效数据。
    __aicore__ inline void CopyOutLastAxisRank5FullAGroup(
        LocalTensor<T>& dstLocal, uint32_t outputBase, uint32_t validCols)
    {
        const uint32_t validRows = inputDims_[0] * inputDims_[2];
        const uint32_t outputRowStride = inputDims_[0] * inputDims_[1] * inputDims_[2];
        const uint32_t blockBytes = validRows * sizeof(T);
        const uint32_t localRowBytes = kLastAxisRank5FullAOutputPitchRows * sizeof(T);
        const uint32_t srcStride =
            (localRowBytes - AlignUpBytes(blockBytes)) / kAlignBytes;
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(validCols), blockBytes, srcStride,
            static_cast<uint32_t>((outputRowStride - validRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
    }

    // 此路径是一份输入槽加一份常驻输出槽，不是完整的输入/输出双缓冲。
    // 输入每完成一组 Vector 读取便释放；下一任务的首组读取可与当前任务 MTE3 写回交叠。
    // MTE3_V 保护输出槽，避免下一个任务的转置覆盖尚未写回的数据。
    __aicore__ inline void ProcessLastAxisRank5FullAGroup(uint32_t taskStart, uint32_t taskCount)
    {
        LocalTensor<T> srcLocal(TPosition::VECCALC, 0,
                                kLastAxisRank5FullASourceSlotBytes / sizeof(T));
        LocalTensor<T> dstLocal(TPosition::VECCALC, kLastAxisRank5FullAOutputOffsetBytes,
                                kLastAxisRank5FullAOutputSlotBytes / sizeof(T));

        uint32_t b = 0;
        uint32_t dStart = 0;
        uint32_t outputBase = 0;
        uint32_t validCols = 0;
        uint32_t paddedCols = 0;
        GetLastAxisRank5FullAGroupTaskInfo(
            taskStart, b, dStart, outputBase, validCols, paddedCols);

        const uint32_t aDim = inputDims_[0];
        const uint32_t rDim = inputDims_[2];
        const uint32_t totalRows = aDim * rDim;
        const uint32_t firstRowCount = MinU32(kLastAxisRank5FullARowGroup, totalRows);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInLastAxisRank5FullARowGroup(
            srcLocal, b, dStart, 0, firstRowCount, paddedCols, validCols);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t offset = 0; offset < taskCount; ++offset) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            uint32_t rowStart = 0;
            uint32_t rowCount = firstRowCount;
            while (true) {
                TransposeLastAxisRank5FullARowGroup(
                    dstLocal, srcLocal, rowStart, rowCount, paddedCols);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                rowStart += rowCount;
                if (rowStart >= totalRows) {
                    break;
                }
                rowCount = MinU32(kLastAxisRank5FullARowGroup, totalRows - rowStart);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInLastAxisRank5FullARowGroup(
                    srcLocal, b, dStart, rowStart, rowCount,
                    paddedCols, validCols);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }

            uint32_t nextB = 0;
            uint32_t nextDStart = 0;
            uint32_t nextOutputBase = 0;
            uint32_t nextValidCols = 0;
            uint32_t nextPaddedCols = 0;
            if (offset + 1 < taskCount) {
                GetLastAxisRank5FullAGroupTaskInfo(
                    taskStart + offset + 1, nextB, nextDStart, nextOutputBase,
                    nextValidCols, nextPaddedCols);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInLastAxisRank5FullARowGroup(
                    srcLocal, nextB, nextDStart, 0, firstRowCount,
                    nextPaddedCols, nextValidCols);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            CopyOutLastAxisRank5FullAGroup(dstLocal, outputBase, validCols);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);

            b = nextB;
            dStart = nextDStart;
            outputBase = nextOutputBase;
            validCols = nextValidCols;
            paddedCols = nextPaddedCols;
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline bool UseLastAxisRank5ADGroup() const
    {
        if constexpr (sizeof(T) != sizeof(int32_t)) {
            return false;
        }
        return mode_ == kModeLastAxisTransposeTiled && swapOuter_ == kLastAxisRank5ADGroup &&
            rank_ == 5 && perm_[0] == 3 && perm_[1] == 4 && perm_[2] == 1 &&
            perm_[3] == 0 && perm_[4] == 2 &&
            inputDims_[0] >= kLastAxisRank5MinAForADGroup &&
            inputDims_[2] >= kLastAxisRank5MinRC && inputDims_[2] <= kLastAxisRank5MaxRC &&
            inputDims_[4] >= kLastAxisRank5MinRC && inputDims_[4] <= kLastAxisRank5MaxRC;
    }

    __aicore__ inline bool UseLastAxisRank5DGroup() const
    {
        if constexpr (sizeof(T) != sizeof(int32_t)) {
            return false;
        }
        return mode_ == kModeLastAxisTransposeTiled && swapOuter_ == kLastAxisRank5DGroup &&
            rank_ == 5 && perm_[0] == 3 && perm_[1] == 4 && perm_[2] == 1 &&
            perm_[3] == 0 && perm_[4] == 2 &&
            inputDims_[2] >= kLastAxisRank5MinRC && inputDims_[2] <= kLastAxisRank5MaxRC &&
            inputDims_[4] >= kLastAxisRank5MinRC && inputDims_[4] <= kLastAxisRank5MaxRC;
    }

    // 固定 A、B，将连续 D×C 合为列维；R 为行维，尾 D 组只包含有效列。
    __aicore__ inline void GetLastAxisRank5DGroupTaskInfo(
        uint32_t task, uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& validCols, uint32_t& paddedCols)
    {
        const uint32_t aDim = inputDims_[0];
        const uint32_t bDim = inputDims_[1];
        const uint32_t rDim = inputDims_[2];
        const uint32_t dDim = inputDims_[3];
        const uint32_t cDim = inputDims_[4];
        const uint32_t dTiles = (dDim + kLastAxisRank5DGroup - 1) / kLastAxisRank5DGroup;
        const uint32_t pair = task / dTiles;
        const uint32_t dTile = task - pair * dTiles;
        const uint32_t a = pair / bDim;
        const uint32_t b = pair - a * bDim;
        const uint32_t dStart = dTile * kLastAxisRank5DGroup;
        const uint32_t dCount = MinU32(kLastAxisRank5DGroup, dDim - dStart);
        validCols = dCount * cDim;
        paddedCols = AlignUp(validCols, kTileB32Cols);
        inputBase = (((a * bDim + b) * rDim * dDim) + dStart) * cDim;
        outputBase = (((dStart * cDim) * bDim + b) * aDim + a) * rDim;
    }

    // 把 R 补到 32 行的物理布局，用两组 16×8 转置处理；源尾行复用有效地址以避免越界读取。
    __aicore__ inline void TransposeLastAxisRank5DGroup(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal, uint32_t paddedCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint32_t validRows = inputDims_[2];
        const uint8_t repeatTimes = static_cast<uint8_t>(paddedCols / kTileB32Cols);
        const uint16_t dstRepStride =
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : kLastAxisRank5PaddedRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t rowBlock = 0; rowBlock < 2; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                const uint32_t srcRow = srcRowBase + i;
                const uint32_t safeRow = srcRow < validRows ? srcRow : 0;
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[safeRow * paddedCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * kLastAxisRank5PaddedRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * kLastAxisRank5PaddedRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutLastAxisRank5DGroup(
        LocalTensor<T>& dstLocal, uint32_t outputBase, uint32_t validCols)
    {
        const uint32_t validRows = inputDims_[2];
        const uint32_t outputRowStride = inputDims_[0] * inputDims_[1] * validRows;
        const uint32_t blockBytes = validRows * sizeof(T);
        const uint32_t localRowBytes = kLastAxisRank5PaddedRows * sizeof(T);
        const uint32_t srcStride =
            (localRowBytes - AlignUpBytes(blockBytes)) / kAlignBytes;
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(validCols), blockBytes, srcStride,
            static_cast<uint32_t>((outputRowStride - validRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
    }

    // 输入和输出各有 ping/pong 两槽；输入由 V_MTE2 释放，输出由 MTE3_V 释放。
    __aicore__ inline void ProcessLastAxisRank5DGroup(uint32_t taskStart, uint32_t taskCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0,
                               kLastAxisRank5SourceSlotBytes / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, kLastAxisRank5DstPingOffsetBytes,
                               kLastAxisRank5OutputSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kLastAxisRank5SrcPongOffsetBytes,
                               kLastAxisRank5SourceSlotBytes / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, kLastAxisRank5DstPongOffsetBytes,
                               kLastAxisRank5OutputSlotBytes / sizeof(T));

        uint32_t inputBase[2] = {0, 0};
        uint32_t outputBase[2] = {0, 0};
        uint32_t validCols[2] = {0, 0};
        uint32_t paddedCols[2] = {0, 0};
        GetLastAxisRank5DGroupTaskInfo(taskStart, inputBase[0], outputBase[0],
                                       validCols[0], paddedCols[0]);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInStaticTile(srcPing, inputBase[0], inputDims_[3] * inputDims_[4],
                         inputDims_[2], paddedCols[0], inputDims_[2], validCols[0]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t offset = 0; offset < taskCount; ++offset) {
            const uint32_t slot = offset & 1U;
            const uint32_t nextSlot = slot ^ 1U;
            const int32_t inputEvent = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t nextInputEvent = nextSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t outputEvent = slot == 0 ? EVENT_ID2 : EVENT_ID3;
            LocalTensor<T>& srcLocal = slot == 0 ? srcPing : srcPong;
            LocalTensor<T>& nextSrcLocal = nextSlot == 0 ? srcPing : srcPong;
            LocalTensor<T>& dstLocal = slot == 0 ? dstPing : dstPong;

            WaitFlag<HardEvent::MTE2_V>(inputEvent);
            if (offset + 1 < taskCount) {
                GetLastAxisRank5DGroupTaskInfo(taskStart + offset + 1,
                                               inputBase[nextSlot], outputBase[nextSlot],
                                               validCols[nextSlot], paddedCols[nextSlot]);
                WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                CopyInStaticTile(nextSrcLocal, inputBase[nextSlot],
                                 inputDims_[3] * inputDims_[4],
                                 inputDims_[2], paddedCols[nextSlot],
                                 inputDims_[2], validCols[nextSlot]);
                SetFlag<HardEvent::MTE2_V>(nextInputEvent);
            }

            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            TransposeLastAxisRank5DGroup(dstLocal, srcLocal, paddedCols[slot]);
            SetFlag<HardEvent::V_MTE2>(inputEvent);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            CopyOutLastAxisRank5DGroup(dstLocal, outputBase[slot], validCols[slot]);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }

    // 同时合并一组 A 和一组 D，将更多有效行放进同一转置任务以扩大写请求。
    __aicore__ inline void GetLastAxisRank5ADGroupTaskInfo(
        uint32_t task, uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& aCount, uint32_t& validRows, uint32_t& paddedRows,
        uint32_t& validCols, uint32_t& paddedCols)
    {
        const uint32_t aDim = inputDims_[0];
        const uint32_t bDim = inputDims_[1];
        const uint32_t rDim = inputDims_[2];
        const uint32_t dDim = inputDims_[3];
        const uint32_t cDim = inputDims_[4];
        const uint32_t aTiles = (aDim + kLastAxisRank5AGroup - 1) / kLastAxisRank5AGroup;
        const uint32_t dTiles = (dDim + kLastAxisRank5ADGroup - 1) / kLastAxisRank5ADGroup;
        const uint32_t tasksPerB = dTiles * aTiles;
        const uint32_t b = task / tasksPerB;
        const uint32_t taskInB = task - b * tasksPerB;
        const uint32_t dTile = taskInB / aTiles;
        const uint32_t aTile = taskInB - dTile * aTiles;
        const uint32_t aStart = aTile * kLastAxisRank5AGroup;
        const uint32_t dStart = dTile * kLastAxisRank5ADGroup;
        aCount = MinU32(kLastAxisRank5AGroup, aDim - aStart);
        const uint32_t dCount = MinU32(kLastAxisRank5ADGroup, dDim - dStart);
        validRows = aCount * rDim;
        paddedRows = AlignUp(validRows, kTileB32Rows);
        validCols = dCount * cDim;
        paddedCols = AlignUp(validCols, kTileB32Cols);
        inputBase = (((aStart * bDim + b) * rDim * dDim) + dStart) * cDim;
        outputBase = (((dStart * cDim * bDim + b) * aDim + aStart) * rDim);
    }

    __aicore__ inline void CopyInLastAxisRank5ADGroup(
        LocalTensor<T>& srcLocal, uint32_t inputBase, uint32_t aCount,
        uint32_t paddedCols, uint32_t validCols)
    {
        const uint32_t bDim = inputDims_[1];
        const uint32_t rDim = inputDims_[2];
        const uint32_t dDim = inputDims_[3];
        const uint32_t cDim = inputDims_[4];
        const uint32_t aStride = bDim * rDim * dDim * cDim;
        const uint32_t rowStride = dDim * cDim;
        for (uint32_t aOffset = 0; aOffset < aCount; ++aOffset) {
            LocalTensor<T> aLocal = srcLocal[aOffset * rDim * paddedCols];
            CopyInStaticTile(aLocal, inputBase + aOffset * aStride, rowStride,
                             rDim, paddedCols, rDim, validCols);
        }
    }

    __aicore__ inline void TransposeLastAxisRank5ADGroup(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
        uint32_t validRows, uint32_t paddedRows, uint32_t paddedCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(paddedCols / kTileB32Cols);
        const uint16_t dstRepStride =
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : paddedRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t rowBlock = 0; rowBlock < paddedRows / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                const uint32_t srcRow = srcRowBase + i;
                const uint32_t safeRow = srcRow < validRows ? srcRow : 0;
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[safeRow * paddedCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * paddedRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * paddedRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutLastAxisRank5ADGroup(
        LocalTensor<T>& dstLocal, uint32_t outputBase, uint32_t validRows,
        uint32_t paddedRows, uint32_t validCols)
    {
        const uint32_t outputRowStride = inputDims_[0] * inputDims_[1] * inputDims_[2];
        const uint32_t blockBytes = validRows * sizeof(T);
        const uint32_t localRowBytes = paddedRows * sizeof(T);
        const uint32_t srcStride =
            (localRowBytes - AlignUpBytes(blockBytes)) / kAlignBytes;
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(validCols), blockBytes, srcStride,
            static_cast<uint32_t>((outputRowStride - validRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
    }

    // A/D 分组沿用双输入、双输出的显式事件流水；尾组依据解码后的有效行列裁剪。
    __aicore__ inline void ProcessLastAxisRank5ADGroup(uint32_t taskStart, uint32_t taskCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0,
                               kLastAxisRank5ADSourceSlotBytes / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, kLastAxisRank5ADDstPingOffsetBytes,
                               kLastAxisRank5ADOutputSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kLastAxisRank5ADSrcPongOffsetBytes,
                               kLastAxisRank5ADSourceSlotBytes / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, kLastAxisRank5ADDstPongOffsetBytes,
                               kLastAxisRank5ADOutputSlotBytes / sizeof(T));

        uint32_t inputBase[2] = {0, 0};
        uint32_t outputBase[2] = {0, 0};
        uint32_t aCount[2] = {0, 0};
        uint32_t validRows[2] = {0, 0};
        uint32_t paddedRows[2] = {0, 0};
        uint32_t validCols[2] = {0, 0};
        uint32_t paddedCols[2] = {0, 0};
        GetLastAxisRank5ADGroupTaskInfo(
            taskStart, inputBase[0], outputBase[0], aCount[0], validRows[0],
            paddedRows[0], validCols[0], paddedCols[0]);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInLastAxisRank5ADGroup(
            srcPing, inputBase[0], aCount[0], paddedCols[0], validCols[0]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t offset = 0; offset < taskCount; ++offset) {
            const uint32_t slot = offset & 1U;
            const uint32_t nextSlot = slot ^ 1U;
            const int32_t inputEvent = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t nextInputEvent = nextSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t outputEvent = slot == 0 ? EVENT_ID2 : EVENT_ID3;
            LocalTensor<T>& srcLocal = slot == 0 ? srcPing : srcPong;
            LocalTensor<T>& nextSrcLocal = nextSlot == 0 ? srcPing : srcPong;
            LocalTensor<T>& dstLocal = slot == 0 ? dstPing : dstPong;

            WaitFlag<HardEvent::MTE2_V>(inputEvent);
            if (offset + 1 < taskCount) {
                GetLastAxisRank5ADGroupTaskInfo(
                    taskStart + offset + 1, inputBase[nextSlot], outputBase[nextSlot],
                    aCount[nextSlot], validRows[nextSlot], paddedRows[nextSlot],
                    validCols[nextSlot], paddedCols[nextSlot]);
                WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                CopyInLastAxisRank5ADGroup(
                    nextSrcLocal, inputBase[nextSlot], aCount[nextSlot],
                    paddedCols[nextSlot], validCols[nextSlot]);
                SetFlag<HardEvent::MTE2_V>(nextInputEvent);
            }

            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            TransposeLastAxisRank5ADGroup(
                dstLocal, srcLocal, validRows[slot], paddedRows[slot], paddedCols[slot]);
            SetFlag<HardEvent::V_MTE2>(inputEvent);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            CopyOutLastAxisRank5ADGroup(
                dstLocal, outputBase[slot], validRows[slot], paddedRows[slot], validCols[slot]);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }

    // 对保留连续内层的轴交换，用多块 DataCopy 直接重组 GM 顺序，避免逐元素 Gather。
    __aicore__ inline void ProcessBlockCopyDirect(uint32_t cStart, uint32_t cCount,
                                                   uint32_t bStart, uint32_t bCount)
    {
        const uint32_t blockLen = swapInner_ * sizeof(T);
        if (blockLen % kAlignBytes != 0) {
            ProcessBlockCopyDirectUnaligned(cStart, cCount, bStart, bCount, blockLen);
            return;
        }
        const uint32_t srcStride = (swapOuter_ * swapDimB_ - 1) * blockLen;
        const uint32_t localLength = swapDimA_ * swapInner_;
        if (cCount * bCount > 1 && localLength * sizeof(T) <= kStaticBlockCopySlotBytes) {
            ProcessBlockCopyStaticDoubleBuffer(cStart, cCount, bStart, bCount, blockLen, srcStride,
                                               localLength);
            return;
        }
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(swapDimA_), blockLen, srcStride, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        for (uint32_t cOffset = 0; cOffset < cCount; ++cOffset) {
            const uint32_t c = cStart + cOffset;
            for (uint32_t bOffset = 0; bOffset < bCount; ++bOffset) {
                const uint32_t b = bStart + bOffset;
                const uint32_t inputBase = (b * swapDimB_ + c) * swapInner_;
                const uint32_t outputBase = (c * swapOuter_ + b) * localLength;
                LocalTensor<T> local = copyQueue_.AllocTensor<T>();
                DataCopyPad(local, inputGm_[inputBase], inParams, padParams);
                copyQueue_.EnQue(local);
                local = copyQueue_.DeQue<T>();
                CopyOutPad(outputGm_, outputBase, local, localLength);
                copyQueue_.FreeTensor(local);
            }
        }
    }

    // 纯搬运的两个静态槽位可分别处于读入和写出阶段，不需要 Vector 转置。
    // MTE3_MTE2 保证槽位上次写出已完成；MTE2_MTE3 保证本次读入完成后才开始写出。
    __aicore__ inline void ProcessBlockCopyStaticDoubleBuffer(uint32_t cStart, uint32_t cCount,
                                                               uint32_t bStart, uint32_t bCount,
                                                               uint32_t blockLen, uint32_t srcStride,
                                                               uint32_t localLength)
    {
        LocalTensor<T> ping(TPosition::VECCALC, 0, kStaticBlockCopySlotBytes / sizeof(T));
        LocalTensor<T> pong(TPosition::VECCALC, kStaticBlockCopySlotBytes,
                            kStaticBlockCopySlotBytes / sizeof(T));
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(swapDimA_), blockLen, srcStride, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t loopIndex = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t cOffset = 0; cOffset < cCount; ++cOffset) {
            const uint32_t c = cStart + cOffset;
            for (uint32_t bOffset = 0; bOffset < bCount; ++bOffset, ++loopIndex) {
                const uint32_t b = bStart + bOffset;
                const bool usePing = (loopIndex & 1U) == 0;
                const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T>& local = usePing ? ping : pong;
                const uint32_t inputBase = (b * swapDimB_ + c) * swapInner_;
                const uint32_t outputBase = (c * swapOuter_ + b) * localLength;

                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopyPad(local, inputGm_[inputBase], inParams, padParams);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);

                CopyOutPad(outputGm_, outputBase, local, localLength);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void GetBlockCopyDirectUnalignedBatch(uint32_t batchIndex, uint32_t cStart,
                                                             uint32_t bStart, uint32_t bCount,
                                                             uint32_t batchesPerA, uint32_t rowsPerBatch,
                                                             uint32_t& inputBase, uint32_t& outputBase,
                                                             uint32_t& validRows)
    {
        const uint32_t batchesPerC = bCount * batchesPerA;
        const uint32_t cOffset = batchIndex / batchesPerC;
        const uint32_t innerIndex = batchIndex - cOffset * batchesPerC;
        const uint32_t bOffset = innerIndex / batchesPerA;
        const uint32_t aBatch = innerIndex - bOffset * batchesPerA;
        const uint32_t aStart = aBatch * rowsPerBatch;
        const uint32_t c = cStart + cOffset;
        const uint32_t b = bStart + bOffset;
        validRows = MinU32(rowsPerBatch, swapDimA_ - aStart);
        inputBase = ((aStart * swapOuter_ + b) * swapDimB_ + c) * swapInner_;
        outputBase = ((c * swapOuter_ + b) * swapDimA_ + aStart) * swapInner_;
    }

    // 连续内层不足 32B 对齐时按 UB 容量分批，使用带填充的多块搬运保留有效数据边界。
    __aicore__ inline void ProcessBlockCopyDirectUnaligned(uint32_t cStart, uint32_t cCount,
                                                            uint32_t bStart, uint32_t bCount,
                                                            uint32_t blockLen)
    {
        const uint32_t paddedBlockLen = AlignUpBytes(blockLen);
        const uint32_t rowsPerBatch = kStaticUnalignedBlockCopySlotBytes / paddedBlockLen;
        if (rowsPerBatch == 0) {
            return;
        }
        const uint32_t batchesPerA = (swapDimA_ + rowsPerBatch - 1) / rowsPerBatch;
        const uint32_t totalBatches = cCount * bCount * batchesPerA;

        LocalTensor<T> ping(TPosition::VECCALC, 0, kStaticUnalignedBlockCopySlotBytes / sizeof(T));
        LocalTensor<T> pong(TPosition::VECCALC, kStaticUnalignedBlockCopySlotBytes,
                            kStaticUnalignedBlockCopySlotBytes / sizeof(T));
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        const uint32_t inputStride = (swapOuter_ * swapDimB_ - 1) * blockLen;

        uint32_t inputBase = 0;
        uint32_t outputBase = 0;
        uint32_t validRows = 0;
        GetBlockCopyDirectUnalignedBatch(0, cStart, bStart, bCount, batchesPerA, rowsPerBatch,
                                         inputBase, outputBase, validRows);
        DataCopyExtParams firstInParams = {
            static_cast<uint16_t>(validRows), blockLen, inputStride, 0, 0};
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopyPad(ping, inputGm_[inputBase], firstInParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        for (uint32_t batch = 0; batch < totalBatches; ++batch) {
            const bool usePing = (batch & 1U) == 0;
            const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T>& currentLocal = usePing ? ping : pong;
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);

            if (batch + 1 < totalBatches) {
                const bool nextUsePing = !usePing;
                const int32_t nextEventId = nextUsePing ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T>& nextLocal = nextUsePing ? ping : pong;
                uint32_t nextValidRows = 0;
                GetBlockCopyDirectUnalignedBatch(batch + 1, cStart, bStart, bCount, batchesPerA, rowsPerBatch,
                                                 inputBase, outputBase, nextValidRows);
                DataCopyExtParams nextInParams = {
                    static_cast<uint16_t>(nextValidRows), blockLen, inputStride, 0, 0};
                WaitFlag<HardEvent::MTE3_MTE2>(nextEventId);
                DataCopyPad(nextLocal, inputGm_[inputBase], nextInParams, padParams);
                SetFlag<HardEvent::MTE2_MTE3>(nextEventId);
            }

            GetBlockCopyDirectUnalignedBatch(batch, cStart, bStart, bCount, batchesPerA, rowsPerBatch,
                                             inputBase, outputBase, validRows);
            DataCopyExtParams outParams = {
                static_cast<uint16_t>(validRows), blockLen, 0, 0, 0};
            DataCopyPad(outputGm_[outputBase], currentLocal, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    // 构造 [A,B,inner]→[B,A,inner] 所需的 UB 字节偏移。
    // 先生成一组索引 a*srcRowStride+inner，再沿 B 复制并加偏移；GM 间隔已在读入时消除。
    __aicore__ inline void PrepareReverse3DSuffixGatherOffsets(LocalTensor<uint32_t>& offsets,
                                                               uint32_t tileA, uint32_t tileB,
                                                               uint32_t srcRowStride,
                                                               uint32_t offsetGroupStride)
    {
        LocalTensor<int32_t> offsetI32 = offsets.ReinterpretCast<int32_t>();
        const uint32_t groupElems = tileA * swapInner_;
        const bool packedGroups = offsetGroupStride == groupElems;
        if (!packedGroups) {
            Duplicate(offsetI32, static_cast<int32_t>(0), tileB * offsetGroupStride);
            PipeBarrier<PIPE_V>();
        }
        // VSEED07：先生成足够多的连续种子行，使后续倍增写入从 8 个 INT32 即 32B 边界开始。
        // 将线性索引 a*inner+i 加上 a*(srcRowStride-inner)，得到目标 a*srcRowStride+i。
        uint32_t seedRows = 1;
        while (seedRows < tileA && (seedRows * swapInner_) % 8 != 0) {
            ++seedRows;
        }
        CreateVecIndex(offsetI32, static_cast<int32_t>(0), seedRows * swapInner_);
        PipeBarrier<PIPE_V>();
        for (uint32_t a = 1; a < seedRows; ++a) {
            const uint32_t begin = a * swapInner_;
            const uint32_t end = begin + swapInner_;
            const int32_t correction = static_cast<int32_t>(a * (srcRowStride - swapInner_));
            // Adds 的 UB 基址向下对齐到 32B，位掩码仅修改当前种子行，保护首尾相邻行的索引。
            for (uint32_t base = begin / 8 * 8; base < end; base += 64) {
                const uint32_t low = begin > base ? begin - base : 0;
                const uint32_t high = MinU32(64, end - base);
                const uint64_t upper = high == 64 ? ~0ULL : (1ULL << high) - 1;
                const uint64_t lower = low == 0 ? 0 : (1ULL << low) - 1;
                uint64_t lanes[2] = {upper & ~lower, 0};
                Adds(offsetI32[base], offsetI32[base], correction, lanes, 1,
                     UnaryRepeatParams{1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
            }
        }
        // 后续行利用已生成索引成倍扩展；每次 Vector 完成后，下一次才能将其作为输入。
        uint32_t filledRows = seedRows;
        while (filledRows < tileA) {
            const uint32_t chunkRows = MinU32(filledRows, tileA - filledRows);
            Adds(offsetI32[filledRows * swapInner_], offsetI32,
                 static_cast<int32_t>(filledRows * srcRowStride), chunkRows * swapInner_);
            PipeBarrier<PIPE_V>();
            filledRows += chunkRows;
        }
        if (packedGroups) {
            uint32_t filledGroups = 1;
            while (filledGroups < tileB) {
                const uint32_t chunkGroups = MinU32(filledGroups, tileB - filledGroups);
                Adds(offsetI32[filledGroups * groupElems], offsetI32,
                     static_cast<int32_t>(filledGroups * swapInner_), chunkGroups * groupElems);
                PipeBarrier<PIPE_V>();
                filledGroups += chunkGroups;
            }
            Muls(offsetI32, offsetI32, static_cast<int32_t>(sizeof(T)), tileA * tileB * swapInner_);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t group = 1; group < tileB; ++group) {
                Adds(offsetI32[group * offsetGroupStride], offsetI32,
                     static_cast<int32_t>(group * swapInner_), groupElems);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t group = 0; group < tileB; ++group) {
                Muls(offsetI32[group * offsetGroupStride], offsetI32[group * offsetGroupStride],
                     static_cast<int32_t>(sizeof(T)), groupElems);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // 从任务号还原固定中间轴及 A/B 分块坐标；解码顺序与核间任务切分保持一致。
    __aicore__ inline void DecodeReverse3DSuffixTile(uint32_t tileIndex, uint32_t tileBlocksA,
                                                     uint32_t& middle, uint32_t& aBlock,
                                                     uint32_t& bBlock)
    {
        if (static_cast<uint64_t>(totalLength_) * sizeof(T) <= kReverse3DInputMajorThresholdBytes) {
            const uint32_t tilesPerBBlock = swapOuter_ * tileBlocksA;
            bBlock = tileIndex / tilesPerBBlock;
            const uint32_t innerTile = tileIndex - bBlock * tilesPerBBlock;
            middle = innerTile / tileBlocksA;
            aBlock = innerTile - middle * tileBlocksA;
            return;
        }
        const uint32_t tileBlocksB = (swapDimB_ + swapTileB_ - 1) / swapTileB_;
        const uint32_t tilesPerABlock = swapOuter_ * tileBlocksB;
        aBlock = tileIndex / tilesPerABlock;
        const uint32_t innerTile = tileIndex - aBlock * tilesPerABlock;
        middle = innerTile / tileBlocksB;
        bBlock = innerTile - middle * tileBlocksB;
    }

    // INT8 的反向三维后缀路径以多块带填充搬运组织连续内层，采用独立的字节数据布局。
    __aicore__ inline void ProcessReverse3DSuffixPadCopy(uint32_t tileStart, uint32_t tileCount)
    {
        const uint32_t tileBlocksA = swapDimA_ / swapTileA_;
        const uint32_t innerBytes = swapInner_ * sizeof(T);
        const uint32_t fullRowBytes = AlignUpBytes(swapTileB_ * innerBytes);
        const uint32_t srcRowStride = fullRowBytes / sizeof(T);

        for (uint32_t step = 0; step < tileCount + 1; ++step) {
            if (step < tileCount) {
                const uint32_t tileIndex = tileStart + step;
                uint32_t middle = 0;
                uint32_t aBlock = 0;
                uint32_t bBlock = 0;
                DecodeReverse3DSuffixTile(tileIndex, tileBlocksA, middle, aBlock, bBlock);
                const uint32_t aStart = aBlock * swapTileA_;
                const uint32_t bStart = bBlock * swapTileB_;
                const uint32_t tileB = MinU32(swapTileB_, swapDimB_ - bStart);
                const uint32_t inputBase =
                    ((aStart * swapOuter_ + middle) * swapDimB_ + bStart) * swapInner_;
                const uint32_t paddedRowBytes = AlignUpBytes(tileB * innerBytes);

                LocalTensor<T> srcLocal = batchCopyQueue_.AllocTensor<T>();
                DataCopyExtParams inParams = {
                    static_cast<uint16_t>(swapTileA_), tileB * innerBytes,
                    static_cast<uint32_t>((swapOuter_ * swapDimB_ - tileB) * innerBytes),
                    static_cast<uint32_t>((fullRowBytes - paddedRowBytes) / kAlignBytes), 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
                batchCopyQueue_.EnQue(srcLocal);
            }

            if (step >= 1) {
                const uint32_t tileIndex = tileStart + step - 1;
                uint32_t middle = 0;
                uint32_t aBlock = 0;
                uint32_t bBlock = 0;
                DecodeReverse3DSuffixTile(tileIndex, tileBlocksA, middle, aBlock, bBlock);
                const uint32_t aStart = aBlock * swapTileA_;
                const uint32_t bStart = bBlock * swapTileB_;
                const uint32_t tileB = MinU32(swapTileB_, swapDimB_ - bStart);
                LocalTensor<T> srcLocal = batchCopyQueue_.DeQue<T>();
                DataCopyExtParams outParams = {1, innerBytes, 0, 0, 0};
                for (uint32_t bOffset = 0; bOffset < tileB; ++bOffset) {
                    const uint32_t outputBase =
                        (((bStart + bOffset) * swapOuter_ + middle) * swapDimA_ + aStart) * swapInner_;
                    for (uint32_t aOffset = 0; aOffset < swapTileA_; ++aOffset) {
                        const uint32_t localOffset = aOffset * srcRowStride + bOffset * swapInner_;
                        DataCopyPad(outputGm_[outputBase + aOffset * swapInner_], srcLocal[localOffset], outParams);
                    }
                }
                batchCopyQueue_.FreeTensor(srcLocal);
            }
        }
    }

    // 固定中间轴，将 [A,B,inner] 的分块搬入 UB，经 Gather 生成 [B,A,inner]。
    // step 读当前块、转置前一块、写出前两块；额外两轮用于排空流水。
    // 输入/输出各两份物理缓冲，EnQue/DeQue/FreeTensor 承担跨 MTE2、Vector、MTE3 的依赖。
    __aicore__ inline void ProcessReverse3DSuffixGather(uint32_t tileStart, uint32_t tileCount,
                                                        uint32_t tileStride)
    {
        const uint32_t tileBlocksA = swapDimA_ / swapTileA_;
        const uint32_t innerBytes = swapInner_ * sizeof(T);
        const uint32_t packedInputRowBytes = AlignUpBytes(swapTileB_ * innerBytes);
        const bool useInputRowSkew =
            static_cast<uint64_t>(totalLength_) * sizeof(T) > kReverse3DInputMajorThresholdBytes &&
            ((packedInputRowBytes / kAlignBytes) & 1U) != 0;
        const uint32_t fullRowBytes = packedInputRowBytes +
            (useInputRowSkew ? kReverse3DInputRowSkewBytes : 0);
        const uint32_t srcRowStride = fullRowBytes / sizeof(T);
        const uint32_t outputRowBytes = AlignUpBytes(swapTileA_ * innerBytes);
        const uint32_t outputRowElems = outputRowBytes / sizeof(T);
        const bool outputRowsAligned = outputRowBytes == swapTileA_ * innerBytes;
        const uint32_t groupElems = swapTileA_ * swapInner_;
        const uint32_t alignedGroupIndexElems =
            AlignUpBytes(groupElems * sizeof(uint32_t)) / sizeof(uint32_t);
        const uint32_t offsetGroupStride = alignedGroupIndexElems > outputRowElems ?
            alignedGroupIndexElems : outputRowElems;
        const bool useBatchedGatherIndex =
            static_cast<uint64_t>(totalLength_) * sizeof(T) > kReverse3DInputMajorThresholdBytes;
        const uint32_t gatherGroupBatch = kReverse3DGatherGroupBatch;
        const uint32_t offsetGroups = outputRowsAligned ?
            (useBatchedGatherIndex ? MinU32(swapTileB_, gatherGroupBatch) : swapTileB_) : 1;
        LocalTensor<uint32_t> offsets = gatherIndexBuf_.Get<uint32_t>();
        PrepareReverse3DSuffixGatherOffsets(
            offsets, swapTileA_, offsetGroups, srcRowStride, offsetGroupStride);

        for (uint32_t step = 0; step < tileCount + 2; ++step) {
            if (step < tileCount) {
                const uint32_t tileIndex = tileStart + step * tileStride;
                uint32_t middle = 0;
                uint32_t aBlock = 0;
                uint32_t bBlock = 0;
                DecodeReverse3DSuffixTile(tileIndex, tileBlocksA, middle, aBlock, bBlock);
                const uint32_t aStart = aBlock * swapTileA_;
                const uint32_t bStart = bBlock * swapTileB_;
                const uint32_t tileB = MinU32(swapTileB_, swapDimB_ - bStart);
                const uint32_t inputBase =
                    ((aStart * swapOuter_ + middle) * swapDimB_ + bStart) * swapInner_;
                const uint32_t paddedRowBytes = AlignUpBytes(tileB * innerBytes);

                LocalTensor<T> srcLocal = batchInQueue_.AllocTensor<T>();
                DataCopyExtParams inParams = {
                    static_cast<uint16_t>(swapTileA_), tileB * innerBytes,
                    static_cast<uint32_t>((swapOuter_ * swapDimB_ - tileB) * innerBytes),
                    static_cast<uint32_t>((fullRowBytes - paddedRowBytes) / kAlignBytes), 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
                batchInQueue_.EnQue(srcLocal);
            }

            if (step >= 1 && step - 1 < tileCount) {
                const uint32_t tileIndex = tileStart + (step - 1) * tileStride;
                uint32_t middle = 0;
                uint32_t aBlock = 0;
                uint32_t bBlock = 0;
                DecodeReverse3DSuffixTile(tileIndex, tileBlocksA, middle, aBlock, bBlock);
                const uint32_t bStart = bBlock * swapTileB_;
                const uint32_t tileB = MinU32(swapTileB_, swapDimB_ - bStart);

                LocalTensor<T> srcLocal = batchInQueue_.DeQue<T>();
                LocalTensor<T> dstLocal = batchOutQueue_.AllocTensor<T>();
                if (outputRowsAligned) {
                    for (uint32_t groupStart = 0; groupStart < tileB; groupStart += offsetGroups) {
                        const uint32_t groupCount = MinU32(offsetGroups, tileB - groupStart);
                        Gather(dstLocal[groupStart * outputRowElems], srcLocal, offsets,
                               groupStart * innerBytes, groupCount * outputRowElems);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
#if __NPU_ARCH__ == 2201
                    if constexpr (sizeof(T) == 2) {
                        // MASK09：保持 C220 的完整段/尾段 Gather 分解，改为先处理所有行的完整段，再处理所有尾段。
                        // 同一组共享一次掩码设置；各行输出区间互不重叠，输入和索引只读，因此组内无需逐行屏障。
                        // 这里的 tailElems 是 UB 输出行的尾段，可含物理补齐位置；GM 写回仍只使用有效元素。
                        // 仅对 C220 的 2 字节类型使用已核对的 vgather，其他类型/架构保留公共 Gather 接口。
                        const uint32_t fullElems = outputRowElems / 128 * 128;
                        const uint8_t fullRepeats = fullElems / 128;
                        const uint32_t tailElems = outputRowElems - fullElems;
                        auto dstPtr = (__ubuf__ uint16_t*)dstLocal.GetPhyAddr();
                        auto offsetPtr = (__ubuf__ uint32_t*)offsets.GetPhyAddr();
                        const uint32_t srcAddr = static_cast<uint32_t>(
                            reinterpret_cast<uint64_t>(srcLocal.GetPhyAddr()));
                        SetMaskNorm();
                        if (fullRepeats > 0) {
                            SetVectorMask<T>(128);
                            for (uint32_t bOffset = 0; bOffset < tileB; ++bOffset) {
                                vgather(dstPtr + bOffset * outputRowElems, offsetPtr,
                                        srcAddr + bOffset * innerBytes, 8, fullRepeats);
                            }
                        }
                        if (tailElems > 0) {
                            SetVectorMask<T>(static_cast<int32_t>(tailElems));
                            for (uint32_t bOffset = 0; bOffset < tileB; ++bOffset) {
                                vgather(dstPtr + bOffset * outputRowElems + fullElems,
                                        offsetPtr + fullElems,
                                        srcAddr + bOffset * innerBytes, 8, 1);
                            }
                        }
                        PipeBarrier<PIPE_V>();
                        // 恢复默认向量掩码，防止当前尾段掩码影响后续向量操作。
                        ResetMask();
                    } else
#endif
                    {
                        // GBAR08：各行只读同一输入/索引、写不同输出区间，组末统一屏障即可。
                        for (uint32_t bOffset = 0; bOffset < tileB; ++bOffset) {
                            Gather(dstLocal[bOffset * outputRowElems], srcLocal,
                                   offsets, bOffset * innerBytes, outputRowElems);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                batchOutQueue_.EnQue(dstLocal);
                batchInQueue_.FreeTensor(srcLocal);
            }

            if (step >= 2) {
                const uint32_t tileIndex = tileStart + (step - 2) * tileStride;
                uint32_t middle = 0;
                uint32_t aBlock = 0;
                uint32_t bBlock = 0;
                DecodeReverse3DSuffixTile(tileIndex, tileBlocksA, middle, aBlock, bBlock);
                const uint32_t aStart = aBlock * swapTileA_;
                const uint32_t bStart = bBlock * swapTileB_;
                const uint32_t tileB = MinU32(swapTileB_, swapDimB_ - bStart);
                const uint32_t outputBase =
                    ((bStart * swapOuter_ + middle) * swapDimA_ + aStart) * swapInner_;

                LocalTensor<T> dstLocal = batchOutQueue_.DeQue<T>();
                DataCopyExtParams outParams = {
                    static_cast<uint16_t>(tileB), swapTileA_ * innerBytes, 0,
                    static_cast<uint32_t>((swapOuter_ * swapDimA_ - swapTileA_) * innerBytes), 0};
                DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
                batchOutQueue_.FreeTensor(dstLocal);
            }
        }
    }

    // 固定未参与交换的轴，在两条交换轴上执行 Host 选择的二维分块和尾块处理。
    __aicore__ inline void ProcessLastAxisTransposeDirect(uint32_t majorStart, uint32_t majorCount,
                                                           uint32_t colTileStart, uint32_t colTileCount)
    {
        const uint32_t tileBytes = swapTileA_ * swapTileB_ * sizeof(T);
        if constexpr (sizeof(T) == 4) {
            if (rank_ == 3 && perm_[0] == 2 && perm_[1] == 1 && perm_[2] == 0 &&
                swapTileA_ == swapDimA_ && tileBytes <= kStaticMacroSlotBytes) {
                ProcessLastAxisReverse3DB32Grouped(majorStart, majorCount, colTileStart, colTileCount);
                return;
            }
        }
        if (tileBytes <= kLastAxisStaticSlotBytes) {
            ProcessLastAxisTransposeStaticDoubleBuffer(majorStart, majorCount, colTileStart, colTileCount);
            return;
        }
        const uint32_t rowAxis = perm_[rank_ - 1];
        const uint32_t colAxis = rank_ - 1;
        const uint32_t outAxisOfCol = swapAxis_;
        const uint32_t rowTiles = (swapDimA_ + swapTileA_ - 1) / swapTileA_;
        const uint32_t colStart = colTileStart * swapTileB_;
        uint32_t fixedCoords[kMaxRank] = {};
        for (uint32_t majorOffset = 0; majorOffset < majorCount; ++majorOffset) {
            const uint32_t major = majorStart + majorOffset;
            const uint32_t fixedIndex = major / rowTiles;
            const uint32_t rowTile = major - fixedIndex * rowTiles;
            const uint32_t rowStart = rowTile * swapTileA_;
            const uint32_t validRows = MinU32(swapTileA_, swapDimA_ - rowStart);
            DecodeLastAxisFixedCoords(fixedIndex, rowAxis, colAxis, fixedCoords);
            uint32_t inputBase = rowStart * inputStrides_[rowAxis] + colStart;
            uint32_t outputBase = rowStart * outputStrides_[rank_ - 1] +
                                  colStart * outputStrides_[outAxisOfCol];
            for (uint32_t axis = 0; axis < rank_; ++axis) {
                if (axis != rowAxis && axis != colAxis) {
                    inputBase += fixedCoords[axis] * inputStrides_[axis];
                }
            }
            for (uint32_t outAxis = 0; outAxis < rank_; ++outAxis) {
                const uint32_t inAxis = perm_[outAxis];
                if (inAxis != rowAxis && inAxis != colAxis) {
                    outputBase += fixedCoords[inAxis] * outputStrides_[outAxis];
                }
            }
            for (uint32_t colOffset = 0; colOffset < colTileCount; ++colOffset) {
                const uint32_t currentCol = colStart + colOffset * swapTileB_;
                const uint32_t validCols = MinU32(swapTileB_, swapDimB_ - currentCol);
                const uint32_t baseTileRows = sizeof(T) == 1 ? kTileB8Cols :
                                              (sizeof(T) == 4 ? kTileB32Rows : kTile2D16);
                const uint32_t baseTileCols = sizeof(T) == 1 ? kTileB8Cols :
                                              (sizeof(T) == 4 ? kTileB32Cols : kTile2D16);
                if (validRows == swapTileA_ && validCols == swapTileB_ &&
                    (swapTileA_ > baseTileRows || swapTileB_ > baseTileCols)) {
                    ProcessTransposeMacroByDtypePad(inputBase, outputBase, inputStrides_[rowAxis],
                                                    outputStrides_[outAxisOfCol], swapTileA_, swapTileB_);
                } else {
                    ProcessTransposeTileByDtype(inputBase, outputBase, inputStrides_[rowAxis],
                                                outputStrides_[outAxisOfCol], validRows, validCols);
                }
                inputBase += swapTileB_;
                outputBase += swapTileB_ * outputStrides_[outAxisOfCol];
            }
        }
    }

    __aicore__ inline void TransposeLastAxisB32Grouped(LocalTensor<T>& dstLocal,
                                                       LocalTensor<T>& srcLocal,
                                                       uint32_t fixedOffset)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(swapTileB_ / kTileB32Cols);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes,
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : swapTileA_ * kLastAxisB32FixedGroup),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < swapTileA_ / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[(srcRowBase + i) * swapTileB_].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                const uint32_t dstBase =
                    (i * kLastAxisB32FixedGroup + fixedOffset) * swapTileA_ + srcRowBase;
                dstList[i * 2] = reinterpret_cast<uint64_t>(dstLocal[dstBase].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[dstBase + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    // 将多个固定坐标的小矩阵组合处理，在较大的输出槽内拼接各组结果后写回。
    __aicore__ inline void ProcessLastAxisReverse3DB32Grouped(uint32_t fixedStart,
                                                              uint32_t fixedCount,
                                                              uint32_t colTileStart,
                                                              uint32_t colTileCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0,
                               kStaticMacroSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kStaticMacroSlotBytes,
                               kStaticMacroSlotBytes / sizeof(T));
        LocalTensor<T> dstLocal(TPosition::VECCALC, kStaticMacroSlotBytes * 2,
                                kStaticMacroSlotBytes * kLastAxisB32FixedGroup / sizeof(T));
        const uint32_t colStart = colTileStart * swapTileB_;
        const uint32_t outputColStride = inputDims_[1] * inputDims_[0];

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        for (uint32_t fixedOffset = 0; fixedOffset < fixedCount;
             fixedOffset += kLastAxisB32FixedGroup) {
            const uint32_t groupCount =
                MinU32(kLastAxisB32FixedGroup, fixedCount - fixedOffset);
            const uint32_t currentFixed = fixedStart + fixedOffset;
            for (uint32_t colOffset = 0; colOffset < colTileCount; ++colOffset) {
                const uint32_t currentCol = colStart + colOffset * swapTileB_;
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInStaticFullTile(srcPing, currentFixed * inputDims_[2] + currentCol,
                                     inputStrides_[0], swapTileA_, swapTileB_);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                for (uint32_t groupOffset = 0; groupOffset < groupCount; ++groupOffset) {
                    const bool usePing = (groupOffset & 1U) == 0;
                    const int32_t inputEvent = usePing ? EVENT_ID0 : EVENT_ID1;
                    LocalTensor<T>& srcLocal = usePing ? srcPing : srcPong;
                    WaitFlag<HardEvent::MTE2_V>(inputEvent);
                    if (groupOffset + 1 < groupCount) {
                        const bool nextUsePing = !usePing;
                        const int32_t nextInputEvent = nextUsePing ? EVENT_ID0 : EVENT_ID1;
                        LocalTensor<T>& nextSrcLocal = nextUsePing ? srcPing : srcPong;
                        const uint32_t nextInputBase =
                            (currentFixed + groupOffset + 1) * inputDims_[2] + currentCol;
                        WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                        CopyInStaticFullTile(nextSrcLocal, nextInputBase, inputStrides_[0],
                                             swapTileA_, swapTileB_);
                        SetFlag<HardEvent::MTE2_V>(nextInputEvent);
                    }
                    if (groupOffset == 0) {
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                    }
                    TransposeLastAxisB32Grouped(dstLocal, srcLocal, groupOffset);
                    SetFlag<HardEvent::V_MTE2>(inputEvent);
                }

                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const uint32_t blockElems = groupCount * swapTileA_;
                const uint32_t srcStride =
                    (kLastAxisB32FixedGroup - groupCount) * swapTileA_ * sizeof(T) / kAlignBytes;
                DataCopyExtParams outParams = {
                    static_cast<uint16_t>(swapTileB_),
                    static_cast<uint32_t>(blockElems * sizeof(T)), srcStride,
                    static_cast<uint32_t>((outputColStride - blockElems) * sizeof(T)), 0};
                const uint32_t outputBase =
                    currentCol * outputColStride + currentFixed * inputDims_[0];
                DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    // 在静态 UB 中按数据位宽构造 TransDataTo5HD 地址列表；大块由基础转置块重复覆盖。
    __aicore__ inline void TransposeStaticMacroTile(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
                                                     uint32_t tileRows, uint32_t tileCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        if constexpr (sizeof(T) == 1) {
            const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB8Cols);
            const uint16_t dstRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows);
            const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
            for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB8Rows; ++rowBlock) {
                const uint32_t srcRowBase = rowBlock * kTileB8Rows;
                const uint32_t dstColBase = (srcRowBase / kTileB8Cols) * kTileB8Cols;
                const bool dstHighHalf = (srcRowBase % kTileB8Cols) != 0;
                for (uint32_t i = 0; i < kTileB8Rows; ++i) {
                    srcList[i] = reinterpret_cast<uint64_t>(
                        srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
                }
                for (uint32_t i = 0; i < kTile2D16; ++i) {
                    dstList[i] = reinterpret_cast<uint64_t>(
                        dstLocal[i * tileRows + dstColBase].GetPhyAddr());
                }
                TransDataTo5HDParams lowParams{
                    dstHighHalf, false, repeatTimes, dstRepStride, srcRepStride};
                TransDataTo5HD<T>(dstList, srcList, lowParams);

                for (uint32_t i = 0; i < kTile2D16; ++i) {
                    dstList[i] = reinterpret_cast<uint64_t>(
                        dstLocal[(kTile2D16 + i) * tileRows + dstColBase].GetPhyAddr());
                }
                TransDataTo5HDParams highParams{
                    dstHighHalf, true, repeatTimes, dstRepStride, srcRepStride};
                TransDataTo5HD<T>(dstList, srcList, highParams);
            }
            PipeBarrier<PIPE_V>();
        } else if constexpr (sizeof(T) == 4) {
            const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
            TransDataTo5HDParams transParams{
                false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
                static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
            for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB32Rows; ++rowBlock) {
                const uint32_t srcRowBase = rowBlock * kTileB32Rows;
                for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                    srcList[i] = reinterpret_cast<uint64_t>(
                        srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
                }
                for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                    dstList[i * 2] = reinterpret_cast<uint64_t>(
                        dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
                    dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                        dstLocal[srcRowBase + i * tileRows + kTileB32Cols].GetPhyAddr());
                }
                TransDataTo5HD<T>(dstList, srcList, transParams);
                PipeBarrier<PIPE_V>();
            }
        } else {
            const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTile2D16);
            TransDataTo5HDParams transParams{
                false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
                static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
            for (uint32_t rowBlock = 0; rowBlock < tileRows / kTile2D16; ++rowBlock) {
                const uint32_t srcRowBase = rowBlock * kTile2D16;
                for (uint32_t i = 0; i < kTile2D16; ++i) {
                    srcList[i] = reinterpret_cast<uint64_t>(
                        srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
                    dstList[i] = reinterpret_cast<uint64_t>(
                        dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
                }
                TransDataTo5HD<T>(dstList, srcList, transParams);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    // 静态输入/输出槽按奇偶任务交替；预取下一块前等待对应输入槽可复用。
    __aicore__ inline void ProcessLastAxisTransposeStaticDoubleBuffer(uint32_t majorStart, uint32_t majorCount,
                                                                      uint32_t colTileStart,
                                                                      uint32_t colTileCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, kLastAxisStaticSlotBytes / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, kLastAxisStaticSlotBytes,
                               kLastAxisStaticSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kLastAxisStaticSlotBytes * 2,
                               kLastAxisStaticSlotBytes / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, kLastAxisStaticSlotBytes * 3,
                               kLastAxisStaticSlotBytes / sizeof(T));

        const uint32_t rowAxis = perm_[rank_ - 1];
        const uint32_t outAxisOfCol = swapAxis_;
        const uint32_t taskCount = majorCount * colTileCount;
        if (taskCount == 0) {
            return;
        }

        uint32_t inputBase[2] = {0, 0};
        uint32_t outputBase[2] = {0, 0};
        uint32_t validRows[2] = {0, 0};
        uint32_t validCols[2] = {0, 0};
        GetLastAxisStaticTileInfo(0, majorStart, colTileStart, colTileCount,
                                  inputBase[0], outputBase[0],
                                  validRows[0], validCols[0]);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInStaticTile(srcPing, inputBase[0], inputStrides_[rowAxis],
                         swapTileA_, swapTileB_, validRows[0], validCols[0]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t task = 0; task < taskCount; ++task) {
            const uint32_t slot = task & 1U;
            const uint32_t nextSlot = slot ^ 1U;
            const int32_t inputEvent = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t nextInputEvent = nextSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t outputEvent = slot == 0 ? EVENT_ID2 : EVENT_ID3;
            LocalTensor<T>& srcLocal = slot == 0 ? srcPing : srcPong;
            LocalTensor<T>& nextSrcLocal = nextSlot == 0 ? srcPing : srcPong;
            LocalTensor<T>& dstLocal = slot == 0 ? dstPing : dstPong;

            WaitFlag<HardEvent::MTE2_V>(inputEvent);
            if (task + 1 < taskCount) {
                GetLastAxisStaticTileInfo(task + 1, majorStart, colTileStart, colTileCount,
                                          inputBase[nextSlot], outputBase[nextSlot],
                                          validRows[nextSlot], validCols[nextSlot]);
                WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                CopyInStaticTile(nextSrcLocal, inputBase[nextSlot], inputStrides_[rowAxis],
                                 swapTileA_, swapTileB_,
                                 validRows[nextSlot], validCols[nextSlot]);
                SetFlag<HardEvent::MTE2_V>(nextInputEvent);
            }

            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            TransposeStaticMacroTile(dstLocal, srcLocal, swapTileA_, swapTileB_);
            SetFlag<HardEvent::V_MTE2>(inputEvent);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            CopyOutStaticTile(dstLocal, outputBase[slot], outputStrides_[outAxisOfCol],
                              swapTileA_, swapTileB_, validRows[slot], validCols[slot]);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }

    __aicore__ inline void GetLastAxisStaticTileInfo(
        uint32_t taskOffset, uint32_t majorStart, uint32_t colTileStart,
        uint32_t colTileCount, uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& validRows, uint32_t& validCols)
    {
        const uint32_t rowAxis = perm_[rank_ - 1];
        const uint32_t colAxis = rank_ - 1;
        const uint32_t outAxisOfCol = swapAxis_;
        const uint32_t rowTiles = (swapDimA_ + swapTileA_ - 1) / swapTileA_;
        const uint32_t majorOffset = taskOffset / colTileCount;
        const uint32_t colOffset = taskOffset - majorOffset * colTileCount;
        const uint32_t major = majorStart + majorOffset;
        const uint32_t fixedIndex = major / rowTiles;
        const uint32_t rowTile = major - fixedIndex * rowTiles;
        const uint32_t rowStart = rowTile * swapTileA_;
        const uint32_t colStart = (colTileStart + colOffset) * swapTileB_;
        validRows = MinU32(swapTileA_, swapDimA_ - rowStart);
        validCols = MinU32(swapTileB_, swapDimB_ - colStart);

        uint32_t fixedCoords[kMaxRank] = {};
        DecodeLastAxisFixedCoords(fixedIndex, rowAxis, colAxis, fixedCoords);
        inputBase = rowStart * inputStrides_[rowAxis] + colStart;
        outputBase = rowStart * outputStrides_[rank_ - 1] +
                     colStart * outputStrides_[outAxisOfCol];
        for (uint32_t axis = 0; axis < rank_; ++axis) {
            if (axis != rowAxis && axis != colAxis) {
                inputBase += fixedCoords[axis] * inputStrides_[axis];
            }
        }
        for (uint32_t outAxis = 0; outAxis < rank_; ++outAxis) {
            const uint32_t inAxis = perm_[outAxis];
            if (inAxis != rowAxis && inAxis != colAxis) {
                outputBase += fixedCoords[inAxis] * outputStrides_[outAxis];
            }
        }
    }

    __aicore__ inline uint32_t OutputIndexToInputOffset(uint32_t outputIndex)
    {
        uint32_t srcOffset = 0;
        uint32_t remaining = outputIndex;
        for (uint32_t axis = 0; axis < rank_; ++axis) {
            const uint32_t stride = outputStrides_[axis];
            const uint32_t coord = stride == 0 ? 0 : remaining / stride;
            remaining = stride == 0 ? 0 : remaining - coord * stride;
            srcOffset += coord * inputStrides_[perm_[axis]];
        }
        return srcOffset;
    }

    // 将连续有效区间按 tileLength_ 分段搬运；输入与输出绑定同一队列 Tensor，省去 UB 内中转。
    __aicore__ inline void CopyContiguousSpan(uint32_t inputOffset, uint32_t outputOffset, uint32_t len)
    {
        for (uint32_t offset = 0; offset < len; offset += tileLength_) {
            uint32_t tileLen = len - offset;
            if (tileLen > tileLength_) {
                tileLen = tileLength_;
            }
            LocalTensor<T> local = copyQueue_.AllocTensor<T>();
            CopyInPad(inputGm_, inputOffset + offset, local, tileLen);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<T>();
            CopyOutPad(outputGm_, outputOffset + offset, local, tileLen);
            copyQueue_.FreeTensor(local);
        }
    }

    // 枚举输出的连续后缀块，映射其输入起点后执行整段拷贝。
    __aicore__ inline void ProcessBlockCopy(uint32_t rowStart, uint32_t rowCount)
    {
        for (uint32_t row = 0; row < rowCount; ++row) {
            const uint32_t outputBase = (rowStart + row) * innerBlock_;
            const uint32_t inputBase = OutputIndexToInputOffset(outputBase);
            CopyContiguousSpan(inputBase, outputBase, innerBlock_);
        }
    }

    // 处理 [batch,rows,cols]→[batch,cols,rows]；按位宽和行形态分派整行、packed 或普通 tile 路径。
    __aicore__ inline void ProcessTranspose021Tiled(uint32_t tileStart, uint32_t tileCount)
    {
        const uint32_t dimH = inputDims_[1];
        const uint32_t dimW = inputDims_[2];
        uint32_t tileRows = kTile2D16;
        uint32_t tileCols = kTile2D16;
        if constexpr (sizeof(T) == 1) {
            tileRows = kTileB8Cols;
            tileCols = kTileB8Cols;
        } else if constexpr (sizeof(T) == 4) {
            tileRows = kTileB32Rows;
            tileCols = kTileB32Cols;
        }
        if (swapTileA_ > tileRows || swapTileB_ > tileCols) {
            ProcessTranspose021MacroTiled(tileStart, tileCount, tileRows, tileCols);
            return;
        }
        if (tileCount > 1) {
            ProcessTranspose021StaticDoubleBuffer(tileStart, tileCount, tileRows, tileCols);
            return;
        }
        const uint32_t hTiles = (dimH + tileRows - 1) / tileRows;
        const uint32_t wTiles = (dimW + tileCols - 1) / tileCols;
        const uint32_t tilesPerBatch = hTiles * wTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t hTile = innerTile / wTiles;
            const uint32_t wTile = innerTile - hTile * wTiles;
            const uint32_t hStart = hTile * tileRows;
            const uint32_t wStart = wTile * tileCols;
            const uint32_t validH = MinU32(tileRows, dimH - hStart);
            const uint32_t validW = MinU32(tileCols, dimW - wStart);
            const uint32_t inputBase = (batch * dimH + hStart) * dimW + wStart;
            const uint32_t outputBase = (batch * dimW + wStart) * dimH + hStart;
            ProcessTransposeTileByDtype(inputBase, outputBase, dimW, dimH, validH, validW);
        }
    }

    __aicore__ inline void ProcessTranspose021MacroTiled(uint32_t tileStart, uint32_t tileCount,
                                                         uint32_t baseTileRows, uint32_t baseTileCols)
    {
        const uint32_t macroBytes = swapTileA_ * swapTileB_ * sizeof(T);
        const uint32_t cols = inputDims_[2];
        if constexpr (sizeof(T) == 1) {
            if (cols < swapTileB_) {
                ProcessTranspose021PackedInt8Gather(tileStart, tileCount, swapTileA_, swapTileB_);
                return;
            }
            if (macroBytes <= kStaticTripleSlotBytes) {
                ProcessTranspose021MacroStaticTripleBuffer(tileStart, tileCount);
                return;
            }
        }
        if constexpr (sizeof(T) == 4) {
            if (UseTranspose021B32FullRow()) {
                ProcessTranspose021B32FullRow(tileStart, tileCount);
                return;
            }
        }
        const bool balancedMacro =
            static_cast<uint64_t>(swapTileA_) <= static_cast<uint64_t>(swapTileB_) * 2 &&
            static_cast<uint64_t>(swapTileB_) <= static_cast<uint64_t>(swapTileA_) * 2;
        if (tileCount > 1 && macroBytes <= kStaticMacroSlotBytes &&
            (balancedMacro || sizeof(T) == 1)) {
            ProcessTranspose021MacroStaticDoubleBuffer(tileStart, tileCount);
            return;
        }

        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t macroRows = swapTileA_;
        const uint32_t macroCols = swapTileB_;
        if constexpr (sizeof(T) != 1) {
            if (cols < macroCols) {
                ProcessTranspose021PackedGather(tileStart, tileCount, macroRows, macroCols);
                return;
            }
        }
        const uint32_t rowTiles = (rows + macroRows - 1) / macroRows;
        const uint32_t colTiles = (cols + macroCols - 1) / macroCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            if (batch >= batchCount) {
                return;
            }
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * macroRows;
            const uint32_t colStart = colTile * macroCols;
            const uint32_t validRows = MinU32(macroRows, rows - rowStart);
            const uint32_t validCols = MinU32(macroCols, cols - colStart);
            const uint32_t inputBase = (batch * rows + rowStart) * cols + colStart;
            const uint32_t outputBase = (batch * cols + colStart) * rows + rowStart;
            if (validRows == macroRows && validCols == macroCols) {
                ProcessTransposeMacroByDtypePad(inputBase, outputBase, cols, rows, macroRows, macroCols);
                continue;
            }
            ProcessTransposeMacroTailPad(inputBase, outputBase, cols, rows,
                                         macroRows, macroCols, validRows, validCols);
        }
    }

    __aicore__ inline void Transpose021B32ToResidentOutput(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
        uint32_t srcStride, uint32_t outputRows,
        uint32_t rowStart, uint32_t tileCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : outputRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t i = 0; i < kTileB32Rows; ++i) {
            srcList[i] = reinterpret_cast<uint64_t>(srcLocal[i * srcStride].GetPhyAddr());
        }
        for (uint32_t i = 0; i < kTileB32Cols; ++i) {
            dstList[i * 2] = reinterpret_cast<uint64_t>(
                dstLocal[i * outputRows + rowStart].GetPhyAddr());
            dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                dstLocal[i * outputRows + rowStart + kTileB32Cols].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstList, srcList, transParams);
    }

    // 输出列段在 UB 常驻，分批读入并转置各行段，拼成较完整输出后再写回。
    __aicore__ inline void ProcessTranspose021ResidentOutput(uint32_t taskStart, uint32_t taskCount)
    {
        if constexpr (sizeof(T) != sizeof(uint32_t)) {
            return;
        }
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t inputTileRows = swapTileA_;
        const uint32_t paddedCols = swapTileB_;
        const uint32_t residentCols = swapOuter_;
        const uint32_t inputSlotBytes = inputTileRows * paddedCols * sizeof(T);
        const uint32_t residentElems = residentCols * rows;
        LocalTensor<T> srcLocal(TPosition::VECCALC, 0, inputSlotBytes / sizeof(T));
        LocalTensor<T> residentLocal(TPosition::VECCALC, inputSlotBytes, residentElems);

        const uint32_t bodyRows = (rows / inputTileRows) * inputTileRows;
        const uint32_t rowTail = rows - bodyRows;
        const uint32_t tailCols = cols - residentCols;
        const uint32_t tailPaddedCols = AlignUp(tailCols, kTileB32Cols);
        for (uint32_t task = 0; task < taskCount; ++task) {
            const uint32_t batch = taskStart + task;
            const uint32_t batchInputBase = batch * rows * cols;
            const uint32_t batchOutputBase = batch * cols * rows;

            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            for (uint32_t rowStart = 0; rowStart < bodyRows; rowStart += inputTileRows) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInStaticTile(srcLocal, batchInputBase + rowStart * cols,
                                 cols, inputTileRows, paddedCols,
                                 inputTileRows, cols);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Transpose021B32ToResidentOutput(residentLocal, srcLocal, paddedCols,
                                                rows, rowStart, residentCols);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            CopyOutPad(outputGm_, batchOutputBase, residentLocal, residentElems);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

            if (rowTail != 0) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInStaticTile(srcLocal, batchInputBase + bodyRows * cols,
                                 cols, inputTileRows, paddedCols,
                                 rowTail, residentCols);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Transpose021B32FullRowSlice(residentLocal, srcLocal, paddedCols,
                                            inputTileRows, 0, residentCols);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                CopyOutStaticTile(residentLocal, batchOutputBase + bodyRows,
                                  rows, inputTileRows, residentCols,
                                  rowTail, residentCols);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }

            if (tailCols == 0) {
                continue;
            }

            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            for (uint32_t rowStart = 0; rowStart < rows; rowStart += inputTileRows) {
                const uint32_t validRows = MinU32(inputTileRows, rows - rowStart);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInStaticTile(srcLocal,
                                 batchInputBase + rowStart * cols + residentCols,
                                 cols, inputTileRows, tailPaddedCols,
                                 validRows, tailCols);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                Transpose021B32FullRowSlice(residentLocal, srcLocal, tailPaddedCols,
                                            inputTileRows, 0, tailPaddedCols);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                CopyOutStaticTile(residentLocal,
                                  batchOutputBase + residentCols * rows + rowStart,
                                  rows, inputTileRows, tailPaddedCols,
                                  validRows, tailCols);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    __aicore__ inline bool UseTranspose021B32FullRow() const
    {
        if constexpr (sizeof(T) != 4) {
            return false;
        }
        const uint32_t paddedCols = AlignUpBytes(inputDims_[2] * sizeof(T)) / sizeof(T);
        return mode_ == kModeTranspose021Fp16 && inputDims_[1] >= 32 && inputDims_[2] >= 128 &&
            swapTileB_ <= paddedCols && swapTileA_ % kTileB32Rows == 0 &&
            swapTileB_ % kTileB32Cols == 0 &&
            tileLength_ * sizeof(T) <= kTranspose021B32FullRowInputSlotBytes;
    }

    __aicore__ inline bool UseTranspose021B32CoreStrided() const
    {
        if constexpr (sizeof(T) != 4) {
            return false;
        }
        return UseTranspose021B32FullRow() && inputDims_[1] > 751 &&
            inputDims_[2] >= 505 && inputDims_[2] < 576;
    }

    // 只在基址、输出行距和每次写入长度均覆盖完整 128B 缓存行时使用该写路径。
    // 32 位元素要求有效行数为 32 的倍数；完整 tile 与尾 tile 都须满足此布局。
    __aicore__ inline bool UseTranspose021B32AlignedStore() const
    {
        if constexpr (sizeof(T) != 4) {
            return false;
        }
        // 总行数乘 4 是 128 的倍数，配合 64 行主块可保证尾块也是 32 行的倍数。
        return UseTranspose021B32CoreStrided() &&
            (inputDims_[1] * sizeof(T)) % kCacheLineBytes == 0 &&
            outputLineAligned_;
    }

    // 按一份 packed 输入、两份输出列段和一份索引表计算 UB 容量。
    // 这里实际采用单输入槽；扩大输入块可以减少列段写出的轮数，输出双槽保留计算/写出交叠。
    __aicore__ inline uint32_t PackedGatherTileRows() const
    {
        const uint32_t perRowBytes =
            inputDims_[2] * sizeof(T) +
            2U * kTranspose021B32GatherChunkCols * sizeof(T) +
            kTranspose021GatherCols * sizeof(uint32_t);
        uint32_t tileRows = kUbBytes / perRowBytes;
        tileRows = (tileRows / kTranspose021B32GatherRowStep) * kTranspose021B32GatherRowStep;
        return tileRows;
    }

    // Gather 连续输出元素在输入中相隔 rowBytes；用整 32B 步进和余数字节估计连续行是否集中访问同一 bank。
    // 这是历史 B3 数据支持的冲突筛选条件，不能由此推断其他路径的实际带宽。
    __aicore__ inline bool PackedGatherBankAliased() const
    {
        const uint32_t rowBytes = inputDims_[2] * sizeof(T);
        const uint32_t blockAdvance = rowBytes / kAlignBytes;
        const uint32_t leftoverBytes = rowBytes % kAlignBytes;
        return blockAdvance % kUbBankCount == 0 &&
            leftoverBytes * kMinBankSpreadRows <= kAlignBytes;
    }

    __aicore__ inline bool UseTranspose021B32PackedGather() const
    {
        if constexpr (sizeof(T) != 4) {
            return false;
        }
        // 仅对非 32B 对齐行启用 packed Gather；已对齐行保留原有较宽的写出粒度。
        return UseTranspose021B32CoreStrided() &&
            inputDims_[2] % kTileB32Cols != 0 &&
            !UseTranspose021B32AlignedStore() &&
            !PackedGatherBankAliased() &&
            PackedGatherTileRows() >= kTranspose021B32GatherRowStep;
    }

    // 依据输入槽容量限制行数，避免补齐后的行宽侵入输出槽；整缓存行写还要求行数为 32 的倍数。
    __aicore__ inline uint32_t CoreStridedTileRows() const
    {
        if (UseTranspose021B32PackedGather()) {
            return PackedGatherTileRows();
        }
        const uint32_t rowBytes = AlignUpBytes(inputDims_[2] * sizeof(T));
        if (rowBytes == 0) {
            return kTileB32Rows;
        }
        if (UseTranspose021B32AlignedStore()) {
            uint32_t alignedRows =
                kTranspose021B32CoreStrided64InputSlotBytes / rowBytes;
            if (alignedRows > kTranspose021B32CoreStrided64Rows) {
                alignedRows = kTranspose021B32CoreStrided64Rows;
            }
            // 保持行数为 32 的倍数，使每个列段写入覆盖整 128B 缓存行。
            alignedRows = (alignedRows / kWholeLineStoreRowStep) * kWholeLineStoreRowStep;
            if (alignedRows != 0) {
                return alignedRows;
            }
        }
        uint32_t tileRows = kTranspose021B32CoreStrided48InputSlotBytes / rowBytes;
        if (tileRows > kTranspose021B32CoreStrided48Rows) {
            tileRows = kTranspose021B32CoreStrided48Rows;
        }
        tileRows = (tileRows / kTileB32Rows) * kTileB32Rows;
        return tileRows == 0 ? kTileB32Rows : tileRows;
    }

    __aicore__ inline void Transpose021B32FullRowSlice(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal, uint32_t srcStride,
        uint32_t tileRows, uint32_t colStart, uint32_t tileCols)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[(srcRowBase + i) * srcStride + colStart].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline bool GetTranspose021B32FullRowTileInfo(
        uint32_t tileIndex, uint32_t tileRows,
        uint32_t& inputBase, uint32_t& outputBase, uint32_t& validRows)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        const uint32_t batch = tileIndex / rowTiles;
        if (batch >= batchCount) {
            return false;
        }
        const uint32_t rowTile = tileIndex - batch * rowTiles;
        const uint32_t rowStart = rowTile * tileRows;
        validRows = MinU32(tileRows, rows - rowStart);
        inputBase = (batch * rows + rowStart) * cols;
        outputBase = batch * cols * rows + rowStart;
        return true;
    }

    __aicore__ inline void ProcessTranspose021B32FullRowLegacy(uint32_t tileStart, uint32_t tileCount)
    {
        const bool coreStrided = UseTranspose021B32CoreStrided();
        const uint32_t inputSlotBytes = coreStrided ?
            kTranspose021B32CoreStridedInputSlotBytes : kTranspose021B32LegacyInputSlotBytes;
        const uint32_t outputSlotBytes = coreStrided ?
            kTranspose021B32CoreStridedOutputSlotBytes : kTranspose021B32LegacyOutputSlotBytes;
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, inputSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, inputSlotBytes, inputSlotBytes / sizeof(T));
        LocalTensor<T> dstLocal(TPosition::VECCALC,
                                inputSlotBytes * 2, outputSlotBytes / sizeof(T));

        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t tileRows = swapTileA_;
        const uint32_t paddedCols = swapTileB_;
        uint32_t chunkCols = MinU32(
            paddedCols,
            outputSlotBytes / (tileRows * sizeof(T)));
        chunkCols = (chunkCols / kTileB32Cols) * kTileB32Cols;
        if (chunkCols == 0) {
            return;
        }

        uint32_t inputBase[2] = {0, 0};
        uint32_t outputBase[2] = {0, 0};
        uint32_t validRows[2] = {0, 0};
        if (!GetTranspose021B32FullRowTileInfo(tileStart, tileRows,
                                               inputBase[0], outputBase[0], validRows[0])) {
            return;
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInStaticTile(srcPing, inputBase[0], cols, tileRows, paddedCols,
                         validRows[0], cols);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t slot = tile & 1U;
            const uint32_t nextSlot = slot ^ 1U;
            const int32_t inputEvent = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t nextInputEvent = nextSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T>& srcLocal = slot == 0 ? srcPing : srcPong;
            LocalTensor<T>& nextSrcLocal = nextSlot == 0 ? srcPing : srcPong;

            WaitFlag<HardEvent::MTE2_V>(inputEvent);
            if (tile + 1 < tileCount) {
                GetTranspose021B32FullRowTileInfo(tileStart + (tile + 1) * taskStride_, tileRows,
                                                  inputBase[nextSlot], outputBase[nextSlot],
                                                  validRows[nextSlot]);
                WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                CopyInStaticTile(nextSrcLocal, inputBase[nextSlot], cols, tileRows, paddedCols,
                                 validRows[nextSlot], cols);
                SetFlag<HardEvent::MTE2_V>(nextInputEvent);
            }

            for (uint32_t colStart = 0; colStart < cols; colStart += chunkCols) {
                const uint32_t validChunkCols = MinU32(chunkCols, cols - colStart);
                const uint32_t transposeCols = AlignUp(validChunkCols, kTileB32Cols);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                Transpose021B32FullRowSlice(dstLocal, srcLocal, paddedCols,
                                            tileRows, colStart, transposeCols);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                CopyOutStaticTile(dstLocal, outputBase[slot] + colStart * rows,
                                  rows, tileRows, transposeCols,
                                  validRows[slot], validChunkCols);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            SetFlag<HardEvent::V_MTE2>(inputEvent);
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    // 将 validRows 行作为一个连续块读入；完整块行数按 8 行对齐，最后不足块单独处理。
    __aicore__ inline void CopyInPackedGatherTile(uint32_t inputBase, uint32_t validRows,
                                                 uint32_t cols)
    {
        LocalTensor<T> srcLocal = batchInQueue_.AllocTensor<T>();
        const uint32_t copyBytes = validRows * cols * sizeof(T);
        const uint32_t alignedBytes = AlignUpBytes(copyBytes);
        const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
        DataCopyExtParams inParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        batchInQueue_.EnQue(srcLocal);
    }

    // 单输入槽保留较高的 packed tile，两个输出槽让后一列段 Gather 与前一列段写出交叠。
    __aicore__ inline void ProcessTranspose021B32PackedGather(uint32_t tileStart,
                                                             uint32_t tileCount)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t tileRows = PackedGatherTileRows();
        LocalTensor<uint32_t> offsets = gatherIndexBuf_.Get<uint32_t>();
        PrepareTranspose021PackedGatherOffsets(offsets, tileRows, cols);

        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            uint32_t inputBase = 0;
            uint32_t outputBase = 0;
            uint32_t validRows = 0;
            if (!GetTranspose021B32FullRowTileInfo(
                    tileStart + tile * taskStride_, tileRows,
                    inputBase, outputBase, validRows)) {
                break;
            }
            CopyInPackedGatherTile(inputBase, validRows, cols);

            LocalTensor<T> srcLocal = batchInQueue_.DeQue<T>();
            for (uint32_t colStart = 0; colStart < cols;
                 colStart += kTranspose021B32GatherChunkCols) {
                const uint32_t validChunkCols =
                    MinU32(kTranspose021B32GatherChunkCols, cols - colStart);
                LocalTensor<T> dstLocal = batchOutQueue_.AllocTensor<T>();
                for (uint32_t group = 0; group < validChunkCols;
                     group += kTranspose021GatherCols) {
                    const uint32_t groupCols =
                        MinU32(kTranspose021GatherCols, validChunkCols - group);
                    Gather(dstLocal[group * tileRows], srcLocal, offsets,
                           (colStart + group) * sizeof(T), groupCols * tileRows);
                    PipeBarrier<PIPE_V>();
                }
                batchOutQueue_.EnQue(dstLocal);
                // 两份输出槽允许下一列段的 Gather 与当前列段的 MTE3 写出交叠。
                LocalTensor<T> outLocal = batchOutQueue_.DeQue<T>();
                CopyOutStaticTile(outLocal, outputBase + colStart * rows,
                                  rows, tileRows, kTranspose021B32GatherChunkCols,
                                  validRows, validChunkCols);
                batchOutQueue_.FreeTensor(outLocal);
            }
            batchInQueue_.FreeTensor(srcLocal);
        }
    }

    // 整行读取配合双输出槽；满足整缓存行条件时使用更宽的 64 行写布局。
    __aicore__ inline void ProcessTranspose021B32CoreStrided48Pipeline(
        uint32_t tileStart, uint32_t tileCount)
    {
        // 整缓存行布局将容量偏向 64 行输入，使每列写出由 192B 扩大为 256B。
        const bool wholeLineStore = UseTranspose021B32AlignedStore();
        const uint32_t inputOffset = wholeLineStore ?
            0U : kTranspose021B32CoreStrided48InputOffsetBytes;
        const uint32_t inputSlot = wholeLineStore ?
            kTranspose021B32CoreStrided64InputSlotBytes :
            kTranspose021B32CoreStrided48InputSlotBytes;
        const uint32_t outPing = wholeLineStore ?
            kTranspose021B32CoreStrided64OutputPingOffsetBytes :
            kTranspose021B32CoreStrided48OutputPingOffsetBytes;
        const uint32_t outPong = wholeLineStore ?
            kTranspose021B32CoreStrided64OutputPongOffsetBytes :
            kTranspose021B32CoreStrided48OutputPongOffsetBytes;
        const uint32_t outSlot = wholeLineStore ?
            kTranspose021B32CoreStrided64OutputSlotBytes :
            kTranspose021B32CoreStrided48OutputSlotBytes;
        LocalTensor<T> srcLocal(TPosition::VECCALC, inputOffset, inputSlot / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, outPing, outSlot / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, outPong, outSlot / sizeof(T));
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t tileRows = CoreStridedTileRows();
        const uint32_t paddedCols = AlignUpBytes(cols * sizeof(T)) / sizeof(T);
        uint32_t chunkCols = outSlot / (tileRows * sizeof(T));
        chunkCols = (MinU32(chunkCols, paddedCols) / kTileB32Cols) * kTileB32Cols;
        if (chunkCols == 0) {
            return;
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        uint32_t chunkIndex = 0;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            uint32_t inputBase = 0;
            uint32_t outputBase = 0;
            uint32_t validRows = 0;
            const uint32_t tileIndex = tileStart + tile * taskStride_;
            if (!GetTranspose021B32FullRowTileInfo(
                    tileIndex, tileRows, inputBase, outputBase, validRows)) {
                break;
            }

            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyInStaticTile(srcLocal, inputBase, cols, tileRows, paddedCols,
                             validRows, cols);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t colStart = 0; colStart < cols;
                 colStart += chunkCols, ++chunkIndex) {
                const uint32_t validChunkCols = MinU32(chunkCols, cols - colStart);
                const uint32_t transposeCols = AlignUp(validChunkCols, kTileB32Cols);
                const bool usePing = (chunkIndex & 1U) == 0;
                const int32_t outputEvent = usePing ? EVENT_ID1 : EVENT_ID2;
                LocalTensor<T>& dstLocal = usePing ? dstPing : dstPong;
                WaitFlag<HardEvent::MTE3_V>(outputEvent);
                Transpose021B32FullRowSlice(dstLocal, srcLocal, paddedCols,
                                            tileRows, colStart, transposeCols);
                if (colStart + chunkCols >= cols) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                SetFlag<HardEvent::V_MTE3>(outputEvent);
                WaitFlag<HardEvent::V_MTE3>(outputEvent);
                CopyOutStaticTile(dstLocal, outputBase + colStart * rows,
                                  rows, tileRows, transposeCols,
                                  validRows, validChunkCols);
                SetFlag<HardEvent::MTE3_V>(outputEvent);
            }
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }


    // 三个静态槽由一份固定输入和两份交替输出组成，Vector 可与前一块 MTE3 写出交叠。
    // 输入等待 V_MTE2 才能覆盖，两个输出槽分别等待各自的 MTE3_V。
    __aicore__ inline void ProcessTranspose021B32TriplePipeline(uint32_t tileStart, uint32_t tileCount)
    {
        LocalTensor<T> srcLocal(TPosition::VECCALC, 0,
                                kTranspose021B32PipelineSlotBytes / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, kTranspose021B32PipelineSlotBytes,
                               kTranspose021B32PipelineSlotBytes / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, kTranspose021B32PipelineSlotBytes * 2,
                               kTranspose021B32PipelineSlotBytes / sizeof(T));

        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t tileRows = swapTileA_;
        const uint32_t tileCols = swapTileB_;
        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        const uint32_t colTiles = (cols + tileCols - 1) / tileCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            if (batch >= inputDims_[0]) {
                break;
            }
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            const uint32_t validRows = MinU32(tileRows, rows - rowStart);
            const uint32_t validCols = MinU32(tileCols, cols - colStart);
            const uint32_t inputBase = (batch * rows + rowStart) * cols + colStart;
            const uint32_t outputBase = (batch * cols + colStart) * rows + rowStart;
            const bool usePing = (tile & 1U) == 0;
            const int32_t outputEvent = usePing ? EVENT_ID1 : EVENT_ID2;
            LocalTensor<T>& dstLocal = usePing ? dstPing : dstPong;

            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyInStaticTile(srcLocal, inputBase, cols, tileRows, tileCols, validRows, validCols);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(outputEvent);

            const uint32_t transposeCols = AlignUp(validCols, kTileB32Cols);
            Transpose021B32FullRowSlice(dstLocal, srcLocal, tileCols,
                                        tileRows, 0, transposeCols);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            CopyOutStaticTile(dstLocal, outputBase, rows, tileRows, transposeCols,
                              validRows, validCols);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    // 32 位整行路径的二次分派入口，区分窄行、宽行以及按核交错分配的任务。
    __aicore__ inline void ProcessTranspose021B32FullRow(uint32_t tileStart, uint32_t tileCount)
    {
        if (UseTranspose021B32PackedGather()) {
            ProcessTranspose021B32PackedGather(tileStart, tileCount);
            return;
        }
        if (UseTranspose021B32CoreStrided()) {
            ProcessTranspose021B32CoreStrided48Pipeline(tileStart, tileCount);
            return;
        }
        const uint32_t paddedInputCols = AlignUpBytes(inputDims_[2] * sizeof(T)) / sizeof(T);
        if (swapTileA_ == 64 && swapTileB_ == 256) {
            ProcessTranspose021B32TriplePipeline(tileStart, tileCount);
            return;
        }
        if (paddedInputCols * sizeof(T) < kTranspose021B32WideRowBytes) {
            ProcessTranspose021B32FullRowLegacy(tileStart, tileCount);
            return;
        }
        LocalTensor<T> srcLocal(TPosition::VECCALC, 0,
                                kTranspose021B32FullRowInputSlotBytes / sizeof(T));
        LocalTensor<T> dstLocal(TPosition::VECCALC, kTranspose021B32FullRowInputSlotBytes,
                                kTranspose021B32FullRowOutputSlotBytes / sizeof(T));

        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t tileRows = swapTileA_;
        const uint32_t tileCols = swapTileB_;
        uint32_t chunkCols = MinU32(
            tileCols,
            kTranspose021B32FullRowOutputSlotBytes / (tileRows * sizeof(T)));
        chunkCols = (chunkCols / kTileB32Cols) * kTileB32Cols;
        if (chunkCols == 0) {
            return;
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);

        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        const uint32_t colTiles = (cols + tileCols - 1) / tileCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            if (batch >= inputDims_[0]) {
                break;
            }
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colTileStart = colTile * tileCols;
            const uint32_t validRows = MinU32(tileRows, rows - rowStart);
            const uint32_t validCols = MinU32(tileCols, cols - colTileStart);
            const uint32_t inputBase = (batch * rows + rowStart) * cols + colTileStart;
            const uint32_t outputBase = (batch * cols + colTileStart) * rows + rowStart;
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyInStaticTile(srcLocal, inputBase, cols, tileRows, tileCols, validRows, validCols);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t colStart = 0; colStart < validCols; colStart += chunkCols) {
                const uint32_t validChunkCols = MinU32(chunkCols, validCols - colStart);
                const uint32_t transposeCols = AlignUp(validChunkCols, kTileB32Cols);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                Transpose021B32FullRowSlice(dstLocal, srcLocal, tileCols,
                                            tileRows, colStart, transposeCols);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                CopyOutStaticTile(dstLocal, outputBase + colStart * rows,
                                  rows, tileRows, transposeCols,
                                  validRows, validChunkCols);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    // 为 packed 输入生成按行跨度递增的字节索引；连续读取的输入在 UB 内不插入逐行填充。
    __aicore__ inline void PrepareTranspose021PackedGatherOffsets(LocalTensor<uint32_t>& offsets,
                                                                   uint32_t tileRows,
                                                                   uint32_t validCols)
    {
        LocalTensor<int32_t> offsetI32 = offsets.ReinterpretCast<int32_t>();
        CreateVecIndex(offsetI32, static_cast<int32_t>(0), tileRows);
        PipeBarrier<PIPE_V>();
        Muls(offsetI32, offsetI32, static_cast<int32_t>(validCols * sizeof(T)), tileRows);
        PipeBarrier<PIPE_V>();
        const uint32_t targetCols = MinU32(kTranspose021GatherCols, validCols);
        uint32_t filledCols = 1;
        while (filledCols < targetCols) {
            const uint32_t chunkCols = MinU32(filledCols, targetCols - filledCols);
            Adds(offsetI32[filledCols * tileRows], offsetI32,
                 static_cast<int32_t>(filledCols * sizeof(T)), chunkCols * tileRows);
            PipeBarrier<PIPE_V>();
            filledCols += chunkCols;
        }
    }

    // 先连续读取多行，再按索引收集输出列段；输出仍依据有效行列写回。
    __aicore__ inline void ProcessTranspose021PackedGather(uint32_t tileStart, uint32_t tileCount,
                                                            uint32_t tileRows, uint32_t tileCols)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        LocalTensor<uint32_t> offsets = gatherIndexBuf_.Get<uint32_t>();
        PrepareTranspose021PackedGatherOffsets(offsets, tileRows, cols);

        for (uint32_t step = 0; step < tileCount + 2; ++step) {
            if (step < tileCount) {
                const uint32_t tileIndex = tileStart + step;
                const uint32_t batch = tileIndex / rowTiles;
                if (batch >= batchCount) {
                    return;
                }
                const uint32_t rowTile = tileIndex - batch * rowTiles;
                const uint32_t rowStart = rowTile * tileRows;
                const uint32_t validRows = MinU32(tileRows, rows - rowStart);
                const uint32_t inputBase = (batch * rows + rowStart) * cols;

                LocalTensor<T> srcLocal = batchInQueue_.AllocTensor<T>();
                DataCopyExtParams inParams = {
                    1, static_cast<uint32_t>(validRows * cols * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
                batchInQueue_.EnQue(srcLocal);
            }

            if (step >= 1 && step - 1 < tileCount) {
                LocalTensor<T> srcLocal = batchInQueue_.DeQue<T>();
                LocalTensor<T> dstLocal = batchOutQueue_.AllocTensor<T>();
                for (uint32_t colStart = 0; colStart < cols; colStart += kTranspose021GatherCols) {
                    const uint32_t colCount = MinU32(kTranspose021GatherCols, cols - colStart);
                    Gather(dstLocal[colStart * tileRows], srcLocal, offsets,
                           colStart * sizeof(T), colCount * tileRows);
                    PipeBarrier<PIPE_V>();
                }
                batchOutQueue_.EnQue(dstLocal);
                batchInQueue_.FreeTensor(srcLocal);
            }

            if (step >= 2) {
                const uint32_t tileIndex = tileStart + step - 2;
                const uint32_t batch = tileIndex / rowTiles;
                const uint32_t rowTile = tileIndex - batch * rowTiles;
                const uint32_t rowStart = rowTile * tileRows;
                const uint32_t validRows = MinU32(tileRows, rows - rowStart);
                const uint32_t outputBase = batch * cols * rows + rowStart;

                LocalTensor<T> dstLocal = batchOutQueue_.DeQue<T>();
                CopyOutStaticTile(dstLocal, outputBase, rows, tileRows, tileCols, validRows, cols);
                batchOutQueue_.FreeTensor(dstLocal);
            }
        }
    }

    __aicore__ inline void PrepareTranspose021PackedInt8Gather(
        LocalTensor<uint32_t>& gatherOffsets,
        LocalTensor<uint16_t>& lowMask, LocalTensor<uint16_t>& highMask,
        uint32_t pairCount, uint32_t cols)
    {
        LocalTensor<int32_t> gatherOffsetsI32 = gatherOffsets.ReinterpretCast<int32_t>();
        CreateVecIndex(gatherOffsetsI32, static_cast<int32_t>(0), pairCount);
        PipeBarrier<PIPE_V>();
        Muls(gatherOffsetsI32, gatherOffsetsI32, static_cast<int32_t>(2 * cols), pairCount);
        PipeBarrier<PIPE_V>();
        if ((cols & 1U) != 0) {
            Adds(gatherOffsetsI32[pairCount], gatherOffsetsI32,
                 static_cast<int32_t>(cols - 1), pairCount);
            Adds(gatherOffsetsI32[2 * pairCount], gatherOffsetsI32,
                 static_cast<int32_t>(cols + 1), pairCount);
        } else {
            Adds(gatherOffsetsI32[pairCount], gatherOffsetsI32,
                 static_cast<int32_t>(cols), pairCount);
        }
        Duplicate(lowMask, static_cast<uint16_t>(0x00FFU), pairCount);
        Duplicate(highMask, static_cast<uint16_t>(0xFF00U), pairCount);
        PipeBarrier<PIPE_V>();
    }

    // INT8 数据通过成对组织适配向量转置粒度，按原始字节位置重排。
    __aicore__ inline void Transpose021PackedInt8Pair(
        LocalTensor<uint16_t>& dstPair, LocalTensor<uint16_t>& srcB16,
        LocalTensor<uint32_t>& gatherOffsets, LocalTensor<uint16_t>& gatherWords,
        LocalTensor<uint16_t>& tempWords,
        LocalTensor<uint16_t>& lowMask, LocalTensor<uint16_t>& highMask,
        uint32_t pairCount, uint32_t cols, uint32_t colStart)
    {
        LocalTensor<uint16_t> evenWords = gatherWords;
        LocalTensor<uint16_t> oddWords = gatherWords[pairCount];
        const bool hasSecondCol = colStart + 1 < cols;
        const uint32_t gatherCount = hasSecondCol && (cols & 1U) != 0 ?
            3 * pairCount : 2 * pairCount;
        Gather(gatherWords, srcB16, gatherOffsets, colStart, gatherCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> dstFirst = dstPair;
        LocalTensor<uint16_t> tempFirst = tempWords;
        if ((cols & 1U) != 0) {
            And(dstFirst, evenWords, lowMask, pairCount);
            And(tempFirst, oddWords, highMask, pairCount);
            if (hasSecondCol) {
                ShiftRight(dstPair[pairCount], evenWords, static_cast<uint16_t>(8), pairCount);
                ShiftLeft(tempWords[pairCount], gatherWords[2 * pairCount],
                          static_cast<uint16_t>(8), pairCount);
            }
        } else {
            And(dstFirst, evenWords, lowMask, pairCount);
            ShiftLeft(tempFirst, oddWords, static_cast<uint16_t>(8), pairCount);
            if (hasSecondCol) {
                ShiftRight(dstPair[pairCount], evenWords, static_cast<uint16_t>(8), pairCount);
                And(tempWords[pairCount], oddWords, highMask, pairCount);
            }
        }
        PipeBarrier<PIPE_V>();

        Or(dstFirst, dstFirst, tempFirst, pairCount);
        if (hasSecondCol) {
            Or(dstPair[pairCount], dstPair[pairCount], tempWords[pairCount], pairCount);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Transpose021PackedInt8TwoPairs(
        LocalTensor<uint16_t>& dstFour, LocalTensor<uint16_t>& srcB16,
        LocalTensor<uint32_t>& gatherOffsets,
        LocalTensor<uint16_t>& gatherWordsA, LocalTensor<uint16_t>& gatherWordsB,
        LocalTensor<uint16_t>& tempWordsA, LocalTensor<uint16_t>& tempWordsB,
        LocalTensor<uint16_t>& lowMask, LocalTensor<uint16_t>& highMask,
        uint32_t pairCount, uint32_t cols, uint32_t colStart)
    {
        LocalTensor<uint16_t> evenWordsA = gatherWordsA;
        LocalTensor<uint16_t> oddWordsA = gatherWordsA[pairCount];
        LocalTensor<uint16_t> evenWordsB = gatherWordsB;
        LocalTensor<uint16_t> oddWordsB = gatherWordsB[pairCount];
        const uint32_t gatherCount = (cols & 1U) != 0 ? 3 * pairCount : 2 * pairCount;
        Gather(gatherWordsA, srcB16, gatherOffsets, colStart, gatherCount);
        Gather(gatherWordsB, srcB16, gatherOffsets, colStart + 2, gatherCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> dstPairB = dstFour[2 * pairCount];
        if ((cols & 1U) != 0) {
            And(dstFour, evenWordsA, lowMask, pairCount);
            And(tempWordsA, oddWordsA, highMask, pairCount);
            ShiftRight(dstFour[pairCount], evenWordsA, static_cast<uint16_t>(8), pairCount);
            ShiftLeft(tempWordsA[pairCount], gatherWordsA[2 * pairCount],
                      static_cast<uint16_t>(8), pairCount);

            And(dstPairB, evenWordsB, lowMask, pairCount);
            And(tempWordsB, oddWordsB, highMask, pairCount);
            ShiftRight(dstPairB[pairCount], evenWordsB, static_cast<uint16_t>(8), pairCount);
            ShiftLeft(tempWordsB[pairCount], gatherWordsB[2 * pairCount],
                      static_cast<uint16_t>(8), pairCount);
        } else {
            And(dstFour, evenWordsA, lowMask, pairCount);
            ShiftLeft(tempWordsA, oddWordsA, static_cast<uint16_t>(8), pairCount);
            ShiftRight(dstFour[pairCount], evenWordsA, static_cast<uint16_t>(8), pairCount);
            And(tempWordsA[pairCount], oddWordsA, highMask, pairCount);

            And(dstPairB, evenWordsB, lowMask, pairCount);
            ShiftLeft(tempWordsB, oddWordsB, static_cast<uint16_t>(8), pairCount);
            ShiftRight(dstPairB[pairCount], evenWordsB, static_cast<uint16_t>(8), pairCount);
            And(tempWordsB[pairCount], oddWordsB, highMask, pairCount);
        }
        PipeBarrier<PIPE_V>();

        Or(dstFour, dstFour, tempWordsA, pairCount);
        Or(dstFour[pairCount], dstFour[pairCount], tempWordsA[pairCount], pairCount);
        Or(dstPairB, dstPairB, tempWordsB, pairCount);
        Or(dstPairB[pairCount], dstPairB[pairCount], tempWordsB[pairCount], pairCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Transpose021PackedInt8Tile(
        LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal,
        LocalTensor<uint32_t>& gatherOffsets,
        LocalTensor<uint16_t>& gatherWordsA, LocalTensor<uint16_t>& gatherWordsB,
        LocalTensor<uint16_t>& tempWordsA, LocalTensor<uint16_t>& tempWordsB,
        LocalTensor<uint16_t>& lowMask, LocalTensor<uint16_t>& highMask,
        uint32_t tileRows, uint32_t cols)
    {
        const uint32_t pairCount = tileRows / 2;
        LocalTensor<uint16_t> srcB16 = srcLocal.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> dstB16 = dstLocal.template ReinterpretCast<uint16_t>();
        uint32_t colStart = 0;
        for (; colStart + 3 < cols; colStart += 4) {
            LocalTensor<uint16_t> dstFour = dstB16[colStart * pairCount];
            Transpose021PackedInt8TwoPairs(dstFour, srcB16, gatherOffsets,
                                           gatherWordsA, gatherWordsB,
                                           tempWordsA, tempWordsB,
                                           lowMask, highMask, pairCount, cols, colStart);
        }
        for (; colStart < cols; colStart += 2) {
            LocalTensor<uint16_t> dstPair = dstB16[colStart * pairCount];
            Transpose021PackedInt8Pair(dstPair, srcB16, gatherOffsets, gatherWordsA,
                                      tempWordsA, lowMask, highMask, pairCount, cols, colStart);
        }
    }

    // 按列段流式组织 INT8 转置，限制同时驻留的 UB 工作集。
    __aicore__ inline void ProcessTranspose021PackedInt8Streamed(
        uint32_t tileStart, uint32_t tileCount, uint32_t tileRows,
        LocalTensor<uint32_t>& gatherOffsets,
        LocalTensor<uint16_t>& gatherWordsA, LocalTensor<uint16_t>& gatherWordsB,
        LocalTensor<uint16_t>& tempWordsA, LocalTensor<uint16_t>& tempWordsB,
        LocalTensor<uint16_t>& lowMask, LocalTensor<uint16_t>& highMask)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        const uint32_t pairCount = tileRows / 2;

        for (uint32_t step = 0; step < tileCount + 1; ++step) {
            if (step < tileCount) {
                const uint32_t tileIndex = tileStart + step;
                const uint32_t batch = tileIndex / rowTiles;
                if (batch >= batchCount) {
                    return;
                }
                const uint32_t rowTile = tileIndex - batch * rowTiles;
                const uint32_t rowStart = rowTile * tileRows;
                const uint32_t validRows = MinU32(tileRows, rows - rowStart);
                const uint32_t inputBase = (batch * rows + rowStart) * cols;
                const uint32_t copyBytes = validRows * cols;
                const uint32_t rightPadding = AlignUpBytes(copyBytes) - copyBytes;

                LocalTensor<T> srcLocal = batchInQueue_.AllocTensor<T>();
                DataCopyExtParams inParams = {1, copyBytes, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{
                    rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
                DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
                batchInQueue_.EnQue(srcLocal);
            }

            if (step == 0) {
                continue;
            }

            const uint32_t tileIndex = tileStart + step - 1;
            const uint32_t batch = tileIndex / rowTiles;
            const uint32_t rowTile = tileIndex - batch * rowTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t validRows = MinU32(tileRows, rows - rowStart);
            const uint32_t outputTileBase = batch * cols * rows + rowStart;
            const uint32_t blockBytes = validRows * sizeof(T);
            const uint32_t srcStride =
                (tileRows * sizeof(T) - AlignUpBytes(blockBytes)) / kAlignBytes;

            LocalTensor<T> srcLocal = batchInQueue_.DeQue<T>();
            LocalTensor<uint16_t> srcB16 = srcLocal.template ReinterpretCast<uint16_t>();
            for (uint32_t colGroup = 0; colGroup < cols; colGroup += kTranspose021StreamCols) {
                const uint32_t validGroupCols = MinU32(kTranspose021StreamCols, cols - colGroup);
                LocalTensor<T> dstLocal = batchOutQueue_.AllocTensor<T>();
                LocalTensor<uint16_t> dstB16 = dstLocal.template ReinterpretCast<uint16_t>();
                uint32_t localCol = 0;
                for (; localCol + 3 < validGroupCols; localCol += 4) {
                    const uint32_t colStart = colGroup + localCol;
                    LocalTensor<uint16_t> dstFour = dstB16[localCol * pairCount];
                    Transpose021PackedInt8TwoPairs(dstFour, srcB16, gatherOffsets,
                                                   gatherWordsA, gatherWordsB,
                                                   tempWordsA, tempWordsB,
                                                   lowMask, highMask,
                                                   pairCount, cols, colStart);
                }
                for (; localCol < validGroupCols; localCol += 2) {
                    const uint32_t colStart = colGroup + localCol;
                    LocalTensor<uint16_t> dstPair = dstB16[localCol * pairCount];
                    Transpose021PackedInt8Pair(dstPair, srcB16, gatherOffsets, gatherWordsA,
                                              tempWordsA, lowMask, highMask,
                                              pairCount, cols, colStart);
                }
                batchOutQueue_.EnQue(dstLocal);

                dstLocal = batchOutQueue_.DeQue<T>();
                DataCopyExtParams outParams = {
                    static_cast<uint16_t>(validGroupCols), blockBytes, srcStride,
                    static_cast<uint32_t>((rows - validRows) * sizeof(T)), 0};
                DataCopyPad(outputGm_[outputTileBase + colGroup * rows], dstLocal, outParams);
                batchOutQueue_.FreeTensor(dstLocal);
            }
            batchInQueue_.FreeTensor(srcLocal);
        }
    }

    // 根据 packed INT8 布局选择收集与重排方式，再裁掉行列补齐部分。
    __aicore__ inline void ProcessTranspose021PackedInt8Gather(
        uint32_t tileStart, uint32_t tileCount, uint32_t tileRows, uint32_t tileCols)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
        const uint32_t pairCount = tileRows / 2;
        LocalTensor<uint8_t> workspace = gatherIndexBuf_.Get<uint8_t>();
        const uint32_t offsetBytes = AlignUpBytes(tileRows * sizeof(uint32_t));
        const uint32_t wordBytes = AlignUpBytes(tileRows * sizeof(uint16_t));
        const uint32_t maskBytes = AlignUpBytes(pairCount * sizeof(uint16_t));
        uint32_t workspaceOffset = 0;
        LocalTensor<uint32_t> gatherOffsets = workspace[workspaceOffset].template ReinterpretCast<uint32_t>();
        workspaceOffset += 2 * offsetBytes;

        LocalTensor<uint16_t> gatherWordsA = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        workspaceOffset += 2 * wordBytes;
        LocalTensor<uint16_t> gatherWordsB = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        workspaceOffset += 2 * wordBytes;
        LocalTensor<uint16_t> tempWordsA = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        workspaceOffset += wordBytes;
        LocalTensor<uint16_t> tempWordsB = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        workspaceOffset += wordBytes;
        LocalTensor<uint16_t> lowMask = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        workspaceOffset += maskBytes;
        LocalTensor<uint16_t> highMask = workspace[workspaceOffset].template ReinterpretCast<uint16_t>();
        (void)maskBytes;
        PrepareTranspose021PackedInt8Gather(gatherOffsets, lowMask, highMask, pairCount, cols);

        if (tileRows > 512) {
            ProcessTranspose021PackedInt8Streamed(tileStart, tileCount, tileRows,
                                                  gatherOffsets, gatherWordsA, gatherWordsB,
                                                  tempWordsA, tempWordsB,
                                                  lowMask, highMask);
            return;
        }

        for (uint32_t step = 0; step < tileCount + 2; ++step) {
            if (step < tileCount) {
                const uint32_t tileIndex = tileStart + step;
                const uint32_t batch = tileIndex / rowTiles;
                if (batch >= batchCount) {
                    return;
                }
                const uint32_t rowTile = tileIndex - batch * rowTiles;
                const uint32_t rowStart = rowTile * tileRows;
                const uint32_t validRows = MinU32(tileRows, rows - rowStart);
                const uint32_t inputBase = (batch * rows + rowStart) * cols;
                const uint32_t copyBytes = validRows * cols;
                const uint32_t rightPadding = AlignUpBytes(copyBytes) - copyBytes;

                LocalTensor<T> srcLocal = batchInQueue_.AllocTensor<T>();
                DataCopyExtParams inParams = {1, copyBytes, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{
                    rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
                DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
                batchInQueue_.EnQue(srcLocal);
            }

            if (step >= 1 && step - 1 < tileCount) {
                LocalTensor<T> srcLocal = batchInQueue_.DeQue<T>();
                LocalTensor<T> dstLocal = batchOutQueue_.AllocTensor<T>();
                Transpose021PackedInt8Tile(dstLocal, srcLocal, gatherOffsets,
                                           gatherWordsA, gatherWordsB,
                                           tempWordsA, tempWordsB, lowMask, highMask,
                                           tileRows, cols);
                batchOutQueue_.EnQue(dstLocal);
                batchInQueue_.FreeTensor(srcLocal);
            }

            if (step >= 2) {
                const uint32_t tileIndex = tileStart + step - 2;
                const uint32_t batch = tileIndex / rowTiles;
                const uint32_t rowTile = tileIndex - batch * rowTiles;
                const uint32_t rowStart = rowTile * tileRows;
                const uint32_t validRows = MinU32(tileRows, rows - rowStart);
                const uint32_t outputBase = batch * cols * rows + rowStart;

                LocalTensor<T> dstLocal = batchOutQueue_.DeQue<T>();
                CopyOutStaticTile(dstLocal, outputBase, rows, tileRows, tileCols, validRows, cols);
                batchOutQueue_.FreeTensor(dstLocal);
            }
        }
    }

    __aicore__ inline bool GetTranspose021MacroTileInfo(
        uint32_t tileIndex, uint32_t macroRows, uint32_t macroCols,
        uint32_t& inputBase, uint32_t& outputBase, uint32_t& validRows, uint32_t& validCols)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = (rows + macroRows - 1) / macroRows;
        const uint32_t colTiles = (cols + macroCols - 1) / macroCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;
        const uint32_t batch = tileIndex / tilesPerBatch;
        if (batch >= batchCount) {
            return false;
        }
        const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
        const uint32_t rowTile = innerTile / colTiles;
        const uint32_t colTile = innerTile - rowTile * colTiles;
        const uint32_t rowStart = rowTile * macroRows;
        const uint32_t colStart = colTile * macroCols;
        validRows = MinU32(macroRows, rows - rowStart);
        validCols = MinU32(macroCols, cols - colStart);
        inputBase = (batch * rows + rowStart) * cols + colStart;
        outputBase = (batch * cols + colStart) * rows + rowStart;
        return true;
    }

    // 大块 021 使用两份交替输入和一份固定输出；提前读入下一块，但复用输出前仍需等待 MTE3。
    __aicore__ inline void ProcessTranspose021MacroStaticTripleBuffer(
        uint32_t tileStart, uint32_t tileCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, kStaticTripleSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kStaticTripleSlotBytes,
                               kStaticTripleSlotBytes / sizeof(T));
        LocalTensor<T> dstLocal(TPosition::VECCALC, kStaticTripleSlotBytes * 2,
                                kStaticTripleSlotBytes / sizeof(T));

        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t macroRows = swapTileA_;
        const uint32_t macroCols = swapTileB_;
        uint32_t inputBase[2] = {0, 0};
        uint32_t outputBase[2] = {0, 0};
        uint32_t validRows[2] = {0, 0};
        uint32_t validCols[2] = {0, 0};

        if (!GetTranspose021MacroTileInfo(tileStart, macroRows, macroCols,
                                          inputBase[0], outputBase[0], validRows[0], validCols[0])) {
            return;
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInStaticTile(srcPing, inputBase[0], cols, macroRows, macroCols,
                         validRows[0], validCols[0]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t slot = tile & 1U;
            const uint32_t nextSlot = slot ^ 1U;
            const int32_t inputEvent = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            const int32_t nextInputEvent = nextSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T>& srcLocal = slot == 0 ? srcPing : srcPong;
            LocalTensor<T>& nextSrcLocal = nextSlot == 0 ? srcPing : srcPong;

            WaitFlag<HardEvent::MTE2_V>(inputEvent);
            if (tile + 1 < tileCount) {
                GetTranspose021MacroTileInfo(tileStart + tile + 1, macroRows, macroCols,
                                             inputBase[nextSlot], outputBase[nextSlot],
                                             validRows[nextSlot], validCols[nextSlot]);
                WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                CopyInStaticTile(nextSrcLocal, inputBase[nextSlot], cols, macroRows, macroCols,
                                 validRows[nextSlot], validCols[nextSlot]);
                SetFlag<HardEvent::MTE2_V>(nextInputEvent);
            }

            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            TransposeStaticMacroTile(dstLocal, srcLocal, macroRows, macroCols);
            SetFlag<HardEvent::V_MTE2>(inputEvent);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            CopyOutStaticTile(dstLocal, outputBase[slot], rows, macroRows, macroCols,
                              validRows[slot], validCols[slot]);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    // 以静态双缓冲组织整块 021 转置；尾块由下层带有效尺寸的搬运处理。
    __aicore__ inline void ProcessTranspose021MacroStaticDoubleBuffer(uint32_t tileStart, uint32_t tileCount)
    {
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, kStaticMacroSlotBytes / sizeof(T));
        LocalTensor<T> dstPing(TPosition::VECCALC, kStaticMacroSlotBytes,
                               kStaticMacroSlotBytes / sizeof(T));
        LocalTensor<T> srcPong(TPosition::VECCALC, kStaticMacroSlotBytes * 2,
                               kStaticMacroSlotBytes / sizeof(T));
        LocalTensor<T> dstPong(TPosition::VECCALC, kStaticMacroSlotBytes * 3,
                               kStaticMacroSlotBytes / sizeof(T));

        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t macroRows = swapTileA_;
        const uint32_t macroCols = swapTileB_;
        const uint32_t rowTiles = (rows + macroRows - 1) / macroRows;
        const uint32_t colTiles = (cols + macroCols - 1) / macroCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const bool usePing = (tile & 1U) == 0;
            const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T>& srcLocal = usePing ? srcPing : srcPong;
            LocalTensor<T>& dstLocal = usePing ? dstPing : dstPong;

            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            if (batch >= batchCount) {
                break;
            }
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * macroRows;
            const uint32_t colStart = colTile * macroCols;
            const uint32_t validRows = MinU32(macroRows, rows - rowStart);
            const uint32_t validCols = MinU32(macroCols, cols - colStart);
            const uint32_t inputBase = (batch * rows + rowStart) * cols + colStart;
            const uint32_t outputBase = (batch * cols + colStart) * rows + rowStart;

            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CopyInStaticTile(srcLocal, inputBase, cols, macroRows, macroCols, validRows, validCols);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            TransposeStaticMacroTile(dstLocal, srcLocal, macroRows, macroCols);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);

            CopyOutStaticTile(dstLocal, outputBase, rows, macroRows, macroCols, validRows, validCols);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    // 把尾块放入满足基础转置要求的 UB 布局，最后只将有效矩形写回 GM。
    __aicore__ inline void ProcessTransposeMacroTailPad(uint32_t inputBase, uint32_t outputBase,
                                                        uint32_t inputStride, uint32_t outputStride,
                                                        uint32_t tileRows, uint32_t tileCols,
                                                        uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        CopyInStaticTile(srcLocal, inputBase, inputStride, tileRows, tileCols, validRows, validCols);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        if constexpr (sizeof(T) == 1) {
            TransposeStaticMacroTile(dstLocal, srcLocal, tileRows, tileCols);
        } else {
            uint64_t srcList[kTile2D16];
            uint64_t dstList[kTile2D16];
            if constexpr (sizeof(T) == 4) {
                const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
                TransDataTo5HDParams transParams{
                    false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
                    static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
                for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB32Rows; ++rowBlock) {
                    const uint32_t srcRowBase = rowBlock * kTileB32Rows;
                    for (uint32_t row = 0; row < kTileB32Rows; ++row) {
                        srcList[row] = reinterpret_cast<uint64_t>(
                            srcLocal[(srcRowBase + row) * tileCols].GetPhyAddr());
                    }
                    for (uint32_t col = 0; col < kTileB32Cols; ++col) {
                        dstList[col * 2] = reinterpret_cast<uint64_t>(
                            dstLocal[srcRowBase + col * tileRows].GetPhyAddr());
                        dstList[col * 2 + 1] = reinterpret_cast<uint64_t>(
                            dstLocal[srcRowBase + col * tileRows + kTileB32Cols].GetPhyAddr());
                    }
                    TransDataTo5HD<T>(dstList, srcList, transParams);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTile2D16);
                TransDataTo5HDParams transParams{
                    false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
                    static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
                for (uint32_t rowBlock = 0; rowBlock < tileRows / kTile2D16; ++rowBlock) {
                    const uint32_t srcRowBase = rowBlock * kTile2D16;
                    for (uint32_t row = 0; row < kTile2D16; ++row) {
                        srcList[row] = reinterpret_cast<uint64_t>(
                            srcLocal[(srcRowBase + row) * tileCols].GetPhyAddr());
                        dstList[row] = reinterpret_cast<uint64_t>(
                            dstLocal[srcRowBase + row * tileRows].GetPhyAddr());
                    }
                    TransDataTo5HD<T>(dstList, srcList, transParams);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        CopyOutStaticTile(dstLocal, outputBase, outputStride, tileRows, tileCols, validRows, validCols);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    // 基础 021 tile 的静态双缓冲实现，读入和写出分别等待相应完成事件。
    __aicore__ inline void ProcessTranspose021StaticDoubleBuffer(uint32_t tileStart, uint32_t tileCount,
                                                                  uint32_t tileRows, uint32_t tileCols)
    {
        constexpr uint32_t staticTileElements =
            sizeof(T) == 1 ? kTileB8Rows * kTileB8Cols :
            (sizeof(T) == 4 ? kTileB32Rows * kTileB32Cols : kTile2D16 * kTile2D16);
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, staticTileElements);
        LocalTensor<T> dstPing(TPosition::VECCALC, kStaticTileSlotBytes, staticTileElements);
        LocalTensor<T> srcPong(TPosition::VECCALC, kStaticTileSlotBytes * 2, staticTileElements);
        LocalTensor<T> dstPong(TPosition::VECCALC, kStaticTileSlotBytes * 3, staticTileElements);

        const uint32_t dimH = inputDims_[1];
        const uint32_t dimW = inputDims_[2];
        const uint32_t hTiles = (dimH + tileRows - 1) / tileRows;
        const uint32_t wTiles = (dimW + tileCols - 1) / tileCols;
        const uint32_t tilesPerBatch = hTiles * wTiles;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const bool usePing = (tile & 1U) == 0;
            const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T>& srcLocal = usePing ? srcPing : srcPong;
            LocalTensor<T>& dstLocal = usePing ? dstPing : dstPong;

            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            const uint32_t hTile = innerTile / wTiles;
            const uint32_t wTile = innerTile - hTile * wTiles;
            const uint32_t hStart = hTile * tileRows;
            const uint32_t wStart = wTile * tileCols;
            const uint32_t validH = MinU32(tileRows, dimH - hStart);
            const uint32_t validW = MinU32(tileCols, dimW - wStart);
            const uint32_t inputBase = (batch * dimH + hStart) * dimW + wStart;
            const uint32_t outputBase = (batch * dimW + wStart) * dimH + hStart;

            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CopyInStaticTile(srcLocal, inputBase, dimW, tileRows, tileCols, validH, validW);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            TransposeStaticFullTile(dstLocal, srcLocal);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);

            CopyOutStaticTile(dstLocal, outputBase, dimH, tileRows, tileCols, validH, validW);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    // 把连续轴段折叠为二维矩阵后复用转置 tile，外层仍按独立批次寻址。
    __aicore__ inline void ProcessTransposeBatchFlattenTiled(uint32_t tileStart, uint32_t tileCount)
    {
        uint32_t tileRows = kTile2D16;
        uint32_t tileCols = kTile2D16;
        if constexpr (sizeof(T) == 1) {
            if (swapTileB_ > kTileB8Cols) {
                ProcessTransposeBatchFlattenInt8MacroPad(tileStart, tileCount);
                return;
            }
            tileRows = kTileB8Cols;
            tileCols = kTileB8Cols;
        } else if constexpr (sizeof(T) == 4) {
            tileRows = kTileB32Rows;
            tileCols = kTileB32Cols;
        }
        if (tileCount > 1 && swapDimA_ % tileRows == 0 && swapDimB_ % tileCols == 0) {
            ProcessTransposeBatchFlattenStaticDoubleBuffer(tileStart, tileCount, tileRows, tileCols);
            return;
        }
        ProcessTransposeBatchFlattenTiledCore(tileStart, tileCount, tileRows, tileCols);
    }

    __aicore__ inline void ProcessTransposeBatchFlattenInt8MacroPad(uint32_t tileStart,
                                                                    uint32_t tileCount)
    {
        const uint32_t tileRows = swapTileA_;
        const uint32_t tileCols = swapTileB_;
        const uint32_t colTiles = (swapDimB_ + tileCols - 1) / tileCols;
        const uint32_t rowTiles = (swapDimA_ + tileRows - 1) / tileRows;
        const uint32_t tilesPerOuter = rowTiles * colTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t outer = tileIndex / tilesPerOuter;
            const uint32_t innerTile = tileIndex - outer * tilesPerOuter;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            const uint32_t validRows = MinU32(tileRows, swapDimA_ - rowStart);
            const uint32_t validCols = MinU32(tileCols, swapDimB_ - colStart);
            const uint32_t inputBase = (outer * swapDimA_ + rowStart) * swapDimB_ + colStart;
            const uint32_t outputBase = (outer * swapDimB_ + colStart) * swapDimA_ + rowStart;
            ProcessTransposeTileMacroB8PadTail(inputBase, outputBase, swapDimB_, swapDimA_,
                                               tileRows, tileCols, validRows, validCols);
        }
    }

    // 完整矩形直接使用多块 DMA；GM 跨行间隔以字节计算，UB 内采用连续的行布局。
    __aicore__ inline void CopyInStaticFullTile(LocalTensor<T>& srcLocal, uint32_t inputBase,
                                                 uint32_t inputStride, uint32_t tileRows, uint32_t tileCols)
    {
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint32_t>(tileCols * sizeof(T)),
            static_cast<uint32_t>((inputStride - tileCols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
    }

    // 读入有效矩形并补齐 UB 行尾；GM 行距保持原始 stride，补齐元素仅服务于物理布局。
    __aicore__ inline void CopyInStaticTile(LocalTensor<T>& srcLocal, uint32_t inputBase,
                                            uint32_t inputStride, uint32_t tileRows, uint32_t tileCols,
                                            uint32_t validRows, uint32_t validCols)
    {
        if (validRows == tileRows && validCols == tileCols) {
            CopyInStaticFullTile(srcLocal, inputBase, inputStride, tileRows, tileCols);
            return;
        }
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if (validCols == tileCols) {
            DataCopyExtParams inParams = {
                static_cast<uint16_t>(validRows), static_cast<uint32_t>(tileCols * sizeof(T)),
                static_cast<uint32_t>((inputStride - tileCols) * sizeof(T)), 0, 0};
            DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
            return;
        }
        const uint32_t copyBytes = validCols * sizeof(T);
        const uint32_t alignedCopyBytes = AlignUpBytes(copyBytes);
        const uint32_t rightPadding = (alignedCopyBytes - copyBytes) / sizeof(T);
        const uint32_t dstStride = (tileCols * sizeof(T) - alignedCopyBytes) / kAlignBytes;
        DataCopyPadExtParams<T> tailPadParams{
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
        for (uint32_t row = 0; row < validRows; row += kDataCopyPadMaxBlockCount) {
            const uint32_t blockCount = MinU32(kDataCopyPadMaxBlockCount, validRows - row);
            DataCopyExtParams inParams = {
                static_cast<uint16_t>(blockCount), copyBytes,
                static_cast<uint32_t>((inputStride - validCols) * sizeof(T)), dstStride, 0};
            DataCopyPad(srcLocal[row * tileCols], inputGm_[inputBase + row * inputStride],
                        inParams, tailPadParams);
        }
    }

    // 按元素位宽选择基础转置指令布局；转置不对输入数值做算术变换。
    __aicore__ inline void TransposeStaticFullTile(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal)
    {
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        if constexpr (sizeof(T) == 1) {
            for (uint32_t rowBlock = 0; rowBlock < 2; ++rowBlock) {
                const uint32_t srcRowBase = rowBlock * kTile2D16;
                const bool dstHighHalf = rowBlock != 0;
                for (uint32_t i = 0; i < kTile2D16; ++i) {
                    srcList[i] = reinterpret_cast<uint64_t>(
                        srcLocal[(srcRowBase + i) * kTileB8Cols].GetPhyAddr());
                    dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * kTileB8Cols].GetPhyAddr());
                }
                TransDataTo5HDParams lowParams{dstHighHalf, false, 1, 0, 0};
                TransDataTo5HD<T>(dstList, srcList, lowParams);
                PipeBarrier<PIPE_V>();

                for (uint32_t i = 0; i < kTile2D16; ++i) {
                    dstList[i] = reinterpret_cast<uint64_t>(
                        dstLocal[(kTile2D16 + i) * kTileB8Cols].GetPhyAddr());
                }
                TransDataTo5HDParams highParams{dstHighHalf, true, 1, 0, 0};
                TransDataTo5HD<T>(dstList, srcList, highParams);
                PipeBarrier<PIPE_V>();
            }
        } else if constexpr (sizeof(T) == 4) {
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[i * kTileB32Cols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(dstLocal[i * kTileB32Rows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[i * kTileB32Rows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HDParams transParams{false, false, 1, 0, 0};
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[i * kTile2D16].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * kTile2D16].GetPhyAddr());
            }
            TransDataTo5HDParams transParams{false, false, 1, 0, 0};
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutStaticFullTile(LocalTensor<T>& dstLocal, uint32_t outputBase,
                                                  uint32_t outputStride, uint32_t tileRows, uint32_t tileCols)
    {
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint32_t>(tileRows * sizeof(T)), 0,
            static_cast<uint32_t>((outputStride - tileRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
    }

    // 从物理 tile 中提取有效矩形，UB 端跨行间隔按 32B 块换算，GM 端按字节换算。
    __aicore__ inline void CopyOutStaticTile(LocalTensor<T>& dstLocal, uint32_t outputBase,
                                             uint32_t outputStride, uint32_t tileRows, uint32_t tileCols,
                                             uint32_t validRows, uint32_t validCols)
    {
        if (validRows == tileRows && validCols == tileCols) {
            CopyOutStaticFullTile(dstLocal, outputBase, outputStride, tileRows, tileCols);
            return;
        }
        if (validRows == tileRows) {
            DataCopyExtParams outParams = {
                static_cast<uint16_t>(validCols), static_cast<uint32_t>(tileRows * sizeof(T)), 0,
                static_cast<uint32_t>((outputStride - tileRows) * sizeof(T)), 0};
            DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
            return;
        }
        const uint32_t blockBytes = validRows * sizeof(T);
        const uint32_t localRowBytes = tileRows * sizeof(T);
        const uint32_t srcStride = (localRowBytes - AlignUpBytes(blockBytes)) / kAlignBytes;
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(validCols), blockBytes, srcStride,
            static_cast<uint32_t>((outputStride - validRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
    }

    __aicore__ inline void ProcessTransposeBatchFlattenStaticDoubleBuffer(uint32_t tileStart, uint32_t tileCount,
                                                                          uint32_t tileRows, uint32_t tileCols)
    {
        constexpr uint32_t staticTileElements =
            sizeof(T) == 1 ? kTileB8Rows * kTileB8Cols :
            (sizeof(T) == 4 ? kTileB32Rows * kTileB32Cols : kTile2D16 * kTile2D16);
        LocalTensor<T> srcPing(TPosition::VECCALC, 0, staticTileElements);
        LocalTensor<T> dstPing(TPosition::VECCALC, kStaticTileSlotBytes, staticTileElements);
        LocalTensor<T> srcPong(TPosition::VECCALC, kStaticTileSlotBytes * 2, staticTileElements);
        LocalTensor<T> dstPong(TPosition::VECCALC, kStaticTileSlotBytes * 3, staticTileElements);

        const uint32_t colTiles = swapDimB_ / tileCols;
        const uint32_t rowTiles = swapDimA_ / tileRows;
        const uint32_t tilesPerOuter = rowTiles * colTiles;
        uint32_t outer = tileStart / tilesPerOuter;
        const uint32_t innerStart = tileStart - outer * tilesPerOuter;
        uint32_t rowTile = innerStart / colTiles;
        uint32_t colTile = innerStart - rowTile * colTiles;
        uint32_t remaining = tileCount;
        uint32_t loopIndex = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (remaining != 0) {
            const uint32_t tilesThisRow = MinU32(remaining, colTiles - colTile);
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            uint32_t inputBase = (outer * swapDimA_ + rowStart) * swapDimB_ + colStart;
            uint32_t outputBase = (outer * swapDimB_ + colStart) * swapDimA_ + rowStart;
            for (uint32_t colOffset = 0; colOffset < tilesThisRow; ++colOffset, ++loopIndex) {
                const bool usePing = (loopIndex & 1U) == 0;
                const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T>& srcLocal = usePing ? srcPing : srcPong;
                LocalTensor<T>& dstLocal = usePing ? dstPing : dstPong;

                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                CopyInStaticFullTile(srcLocal, inputBase, swapDimB_, tileRows, tileCols);
                SetFlag<HardEvent::MTE2_V>(eventId);
                WaitFlag<HardEvent::MTE2_V>(eventId);

                TransposeStaticFullTile(dstLocal, srcLocal);
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);

                CopyOutStaticFullTile(dstLocal, outputBase, swapDimA_, tileRows, tileCols);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
                inputBase += tileCols;
                outputBase += tileCols * swapDimA_;
            }
            remaining -= tilesThisRow;
            colTile = 0;
            ++rowTile;
            if (rowTile == rowTiles) {
                rowTile = 0;
                ++outer;
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void ProcessTransposeBatchFlattenTiledCore(uint32_t tileStart, uint32_t tileCount,
                                                                 uint32_t tileRows, uint32_t tileCols)
    {
        const uint32_t colTiles = (swapDimB_ + tileCols - 1) / tileCols;
        const uint32_t tilesPerOuter = ((swapDimA_ + tileRows - 1) / tileRows) * colTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t outer = tileIndex / tilesPerOuter;
            const uint32_t innerTile = tileIndex - outer * tilesPerOuter;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            const uint32_t validRows = MinU32(tileRows, swapDimA_ - rowStart);
            const uint32_t validCols = MinU32(tileCols, swapDimB_ - colStart);
            const uint32_t inputBase = (outer * swapDimA_ + rowStart) * swapDimB_ + colStart;
            const uint32_t outputBase = (outer * swapDimB_ + colStart) * swapDimA_ + rowStart;
            ProcessTransposeTileByDtype(inputBase, outputBase, swapDimB_, swapDimA_, validRows, validCols);
        }
    }

    // 解码固定轴与交换轴的分块坐标，分别处理完整块和边界块。
    __aicore__ inline void ProcessLastAxisTransposeTiled(uint32_t tileStart, uint32_t tileCount)
    {
        const uint32_t rowAxis = perm_[rank_ - 1];
        const uint32_t colAxis = rank_ - 1;
        const uint32_t outAxisOfCol = swapAxis_;
        const uint32_t tileRows = swapTileA_;
        const uint32_t tileCols = swapTileB_;
        const uint32_t rowTiles = (swapDimA_ + tileRows - 1) / tileRows;
        const uint32_t colTiles = (swapDimB_ + tileCols - 1) / tileCols;
        const uint32_t tilesPerFixed = rowTiles * colTiles;
        uint32_t fixedCoords[kMaxRank] = {};
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t fixedIndex = tileIndex / tilesPerFixed;
            const uint32_t innerTile = tileIndex - fixedIndex * tilesPerFixed;
            const uint32_t rowTile = innerTile / colTiles;
            const uint32_t colTile = innerTile - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            const uint32_t validRows = MinU32(tileRows, swapDimA_ - rowStart);
            const uint32_t validCols = MinU32(tileCols, swapDimB_ - colStart);

            DecodeLastAxisFixedCoords(fixedIndex, rowAxis, colAxis, fixedCoords);
            uint32_t inputBase = rowStart * inputStrides_[rowAxis] + colStart;
            uint32_t outputBase = rowStart * outputStrides_[rank_ - 1] + colStart * outputStrides_[outAxisOfCol];
            for (uint32_t axis = 0; axis < rank_; ++axis) {
                if (axis == rowAxis || axis == colAxis) {
                    continue;
                }
                inputBase += fixedCoords[axis] * inputStrides_[axis];
            }
            for (uint32_t outAxis = 0; outAxis < rank_; ++outAxis) {
                const uint32_t inAxis = perm_[outAxis];
                if (inAxis == rowAxis || inAxis == colAxis) {
                    continue;
                }
                outputBase += fixedCoords[inAxis] * outputStrides_[outAxis];
            }
            const uint32_t baseTileRows = sizeof(T) == 1 ? kTileB8Cols :
                                          (sizeof(T) == 4 ? kTileB32Rows : kTile2D16);
            const uint32_t baseTileCols = sizeof(T) == 1 ? kTileB8Cols :
                                          (sizeof(T) == 4 ? kTileB32Cols : kTile2D16);
            if (validRows == tileRows && validCols == tileCols &&
                (tileRows > baseTileRows || tileCols > baseTileCols)) {
                ProcessTransposeMacroByDtypePad(inputBase, outputBase, inputStrides_[rowAxis],
                                                outputStrides_[outAxisOfCol], tileRows, tileCols);
            } else {
                ProcessTransposeTileByDtype(inputBase, outputBase, inputStrides_[rowAxis],
                                            outputStrides_[outAxisOfCol], validRows, validCols);
            }
        }
    }

    // 仅展开不参与本次二维交换的轴，将固定坐标映射到输入/输出 GM 基址。
    __aicore__ inline void DecodeLastAxisFixedCoords(uint32_t fixedIndex, uint32_t rowAxis, uint32_t colAxis,
                                                     uint32_t (&coords)[kMaxRank])
    {
        for (uint32_t axis = 0; axis < kMaxRank; ++axis) {
            coords[axis] = 0;
        }
        for (int32_t axis = static_cast<int32_t>(rank_) - 1; axis >= 0; --axis) {
            if (static_cast<uint32_t>(axis) == rowAxis || static_cast<uint32_t>(axis) == colAxis) {
                continue;
            }
            const uint32_t dim = inputDims_[axis];
            coords[axis] = fixedIndex % dim;
            fixedIndex /= dim;
        }
    }

    __aicore__ inline void ProcessTransposeTile32x32B8(uint32_t inputBase, uint32_t outputBase,
                                                       uint32_t inputStride, uint32_t outputStride)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(kTileB8Cols), static_cast<uint32_t>(kTileB8Cols * sizeof(T)),
            static_cast<uint32_t>((inputStride - kTileB8Cols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        for (uint32_t rowBlock = 0; rowBlock < 2; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTile2D16;
            const bool dstHighHalf = rowBlock != 0;
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * kTileB8Cols].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * kTileB8Cols].GetPhyAddr());
            }
            TransDataTo5HDParams transParamsLow{dstHighHalf, false, 1, 0, 0};
            TransDataTo5HD<T>(dstList, srcList, transParamsLow);
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[(kTile2D16 + i) * kTileB8Cols].GetPhyAddr());
            }
            TransDataTo5HDParams transParamsHigh{dstHighHalf, true, 1, 0, 0};
            TransDataTo5HD<T>(dstList, srcList, transParamsHigh);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        if (outputStride == kTileB8Cols) {
            DataCopyExtParams outParams = {1, 32 * 32 * sizeof(T), 0, 0, 0};
            DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        } else {
            DataCopyExtParams outParams = {
                static_cast<uint16_t>(kTileB8Cols), static_cast<uint32_t>(kTileB8Cols * sizeof(T)), 0,
                static_cast<uint32_t>((outputStride - kTileB8Cols) * sizeof(T)), 0};
            DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        }
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    // 按数据位宽分派基础 tile：16 位、32 位和 INT8 使用不同地址组织及尾块策略。
    __aicore__ inline void ProcessTransposeTileByDtype(uint32_t inputBase, uint32_t outputBase,
                                                       uint32_t inputStride, uint32_t outputStride,
                                                       uint32_t validRows, uint32_t validCols)
    {
        if constexpr (sizeof(T) == 1) {
            if (validRows == kTileB8Cols && validCols == kTileB8Cols) {
                ProcessTransposeTile32x32B8(inputBase, outputBase, inputStride, outputStride);
            } else {
                ProcessTransposeTileB8TailStrided(inputBase, outputBase, inputStride, outputStride,
                                                  validRows, validCols);
            }
        } else if constexpr (sizeof(T) == 4) {
            ProcessTransposeTileB32(inputBase, outputBase, inputStride, outputStride, validRows, validCols);
        } else {
            ProcessTransposeTile16(inputBase, outputBase, inputStride, outputStride, validRows, validCols);
        }
    }

    __aicore__ inline void ProcessTransposeMacroByDtypePad(uint32_t inputBase, uint32_t outputBase,
                                                           uint32_t inputStride, uint32_t outputStride,
                                                           uint32_t tileRows, uint32_t tileCols)
    {
        if constexpr (sizeof(T) == 1) {
            ProcessTransposeTileMacroB8Pad(inputBase, outputBase, inputStride, outputStride, tileRows, tileCols);
        } else if constexpr (sizeof(T) == 4) {
            ProcessTransposeTileMacroB32Pad(inputBase, outputBase, inputStride, outputStride, tileRows, tileCols);
        } else {
            ProcessTransposeTileMacroPad(inputBase, outputBase, inputStride, outputStride, tileRows, tileCols);
        }
    }

    // 16 位基础转置：搬入有效 tile，构造 16 路地址表，执行转置后写回有效范围。
    __aicore__ inline void ProcessTransposeTile16(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                  uint32_t outputStride, uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if (validCols == kTile2D16) {
            DataCopyExtParams inParams = {
                static_cast<uint16_t>(validRows), static_cast<uint32_t>(kTile2D16 * sizeof(T)),
                static_cast<uint32_t>((inputStride - kTile2D16) * sizeof(T)), 0, 0};
            DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        } else {
            for (uint32_t row = 0; row < validRows; ++row) {
                DataCopyExtParams inParams = {1, static_cast<uint32_t>(validCols * sizeof(T)), 0, 0, 0};
                DataCopyPad(srcLocal[row * kTile2D16], inputGm_[inputBase + row * inputStride], inParams, padParams);
            }
        }
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        for (uint32_t i = 0; i < kTile2D16; ++i) {
            srcList[i] = reinterpret_cast<uint64_t>(srcLocal[i * kTile2D16].GetPhyAddr());
            dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * kTile2D16].GetPhyAddr());
        }
        TransDataTo5HDParams transParams{false, false, 1, 0, 0};
        TransDataTo5HD<T>(dstList, srcList, transParams);
        PipeBarrier<PIPE_V>();
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        if (validRows == kTile2D16) {
            DataCopyExtParams outParams = {
                static_cast<uint16_t>(validCols), static_cast<uint32_t>(kTile2D16 * sizeof(T)), 0,
                static_cast<uint32_t>((outputStride - kTile2D16) * sizeof(T)), 0};
            DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        } else {
            for (uint32_t row = 0; row < validCols; ++row) {
                DataCopyExtParams outParams = {1, static_cast<uint32_t>(validRows * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm_[outputBase + row * outputStride], dstLocal[row * kTile2D16], outParams);
            }
        }
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTranspose2DMacroTile(uint32_t tileStart, uint32_t tileCount, uint32_t tileRows,
                                                       uint32_t tileCols)
    {
        const uint32_t rows = inputDims_[0];
        const uint32_t cols = inputDims_[1];
        const uint32_t colTiles = cols / tileCols;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t rowTile = tileIndex / colTiles;
            const uint32_t colTile = tileIndex - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            ProcessTransposeTileMacro(rowStart * cols + colStart, colStart * rows + rowStart, cols, rows,
                                      tileRows, tileCols);
        }
    }

    __aicore__ inline void ProcessTranspose021Macro(uint32_t tileStart, uint32_t tileCount,
                                                    uint32_t tileRows, uint32_t tileCols)
    {
        const uint32_t batchCount = inputDims_[0];
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t rowTiles = rows / tileRows;
        const uint32_t colTiles = cols / tileCols;
        const uint32_t tilesPerBatch = rowTiles * colTiles;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t batch = tileIndex / tilesPerBatch;
            if (batch >= batchCount) {
                return;
            }
            const uint32_t innerTile = tileIndex - batch * tilesPerBatch;
            uint32_t rowTile = 0;
            uint32_t colTile = 0;
            if constexpr (sizeof(T) == 2) {
                colTile = innerTile / rowTiles;
                rowTile = innerTile - colTile * rowTiles;
            } else {
                rowTile = innerTile / colTiles;
                colTile = innerTile - rowTile * colTiles;
            }
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            const uint32_t inputBase = (batch * rows + rowStart) * cols + colStart;
            const uint32_t outputBase = (batch * cols + colStart) * rows + rowStart;
            if constexpr (sizeof(T) == 1) {
                ProcessTransposeTileMacroB8(inputBase, outputBase, cols, rows, tileRows, tileCols);
            } else if constexpr (sizeof(T) == 4) {
                ProcessTransposeTileMacroB32(inputBase, outputBase, cols, rows, tileRows, tileCols);
            } else {
                ProcessTransposeTileMacro(inputBase, outputBase, cols, rows, tileRows, tileCols);
            }
        }
    }

    // 将若干完整矩阵作为一批搬入 UB，在两份工作 Tensor 间完成填充、重排与去填充。
    __aicore__ inline void ProcessTranspose021Batch(uint32_t batchStart, uint32_t batchCount)
    {
        for (uint32_t offset = 0; offset < batchCount; offset += swapTileA_) {
            const uint32_t tasks = MinU32(swapTileA_, batchCount - offset);
            const uint32_t currentBatch = batchStart + offset;
            CopyInTranspose021Batch(currentBatch, tasks);
            ComputeTranspose021Batch(tasks);
            CopyOutTranspose021Batch(currentBatch, tasks);
        }
    }

    __aicore__ inline void CopyInTranspose021Batch(uint32_t batchStart, uint32_t tasks)
    {
        const uint32_t matrixElems = inputDims_[1] * inputDims_[2];
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(tasks * matrixElems * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[batchStart * matrixElems], copyParams, padParams);
        tileInQueue_.EnQue(srcLocal);
    }

    // 把行/列是否满足物理块对齐转成模板参数，消除当前批次不需要的填充步骤。
    __aicore__ inline void ComputeTranspose021Batch(uint32_t tasks)
    {
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const bool isWAlign = inputDims_[2] == AlignUp(inputDims_[2], blockElems);
        const bool isHAlign = inputDims_[1] == AlignUp(inputDims_[1], kTile2D16);
        if (isWAlign && isHAlign) {
            ComputeTranspose021BatchImpl<true, true>(tasks);
        } else if (isWAlign) {
            ComputeTranspose021BatchImpl<true, false>(tasks);
        } else if (isHAlign) {
            ComputeTranspose021BatchImpl<false, true>(tasks);
        } else {
            ComputeTranspose021BatchImpl<false, false>(tasks);
        }
    }

    template <bool kWAlign, bool kHAlign>
    // 两份 Tensor 交替充当中间结果；每个 PIPE_V 保护前一步输出被下一步读取的依赖。
    __aicore__ inline void ComputeTranspose021BatchImpl(uint32_t tasks)
    {
        LocalTensor<T> srcLocal = tileInQueue_.DeQue<T>();
        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        if constexpr (kWAlign) {
            ReshapeTranspose021Batch(srcLocal, dstLocal, tasks);
            PipeBarrier<PIPE_V>();
            TransposeCompute021Batch(dstLocal, srcLocal, tasks);
            PipeBarrier<PIPE_V>();
            if constexpr (kHAlign) {
                CopyTranspose021BatchLocal(srcLocal, dstLocal, tasks);
            } else {
                UnPadHTranspose021Batch(srcLocal, dstLocal, tasks);
            }
        } else {
            PadWTranspose021Batch(srcLocal, dstLocal, tasks);
            PipeBarrier<PIPE_V>();
            ReshapeTranspose021Batch(dstLocal, srcLocal, tasks);
            PipeBarrier<PIPE_V>();
            TransposeCompute021Batch(srcLocal, dstLocal, tasks);
            PipeBarrier<PIPE_V>();
            UnPadWTranspose021Batch(dstLocal, srcLocal, tasks);
            PipeBarrier<PIPE_V>();
            if constexpr (kHAlign) {
                CopyTranspose021BatchLocal(srcLocal, dstLocal, tasks);
            } else {
                UnPadHTranspose021Batch(srcLocal, dstLocal, tasks);
            }
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);
    }

    // 整批写回紧凑输出；物理行有间隔时用多块 DMA 跳过间隔，并单独处理最后不足块的字节。
    __aicore__ inline void CopyOutTranspose021Batch(uint32_t batchStart, uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t matrixElems = rows * cols;
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t rowsAlign = AlignUp(rows, kTile2D16);
        const uint32_t colsAlign = AlignUp(cols, blockElems);
        LocalTensor<T> dstLocal = tileOutQueue_.DeQue<T>();
        if (rows == rowsAlign) {
            DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(tasks * matrixElems * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[batchStart * matrixElems], dstLocal, copyParams);
        } else {
            const uint32_t unPadHNum = tasks * colsAlign / blockElems;
            const uint32_t blockLen = unPadHNum * rows * sizeof(T);
            const uint32_t totalBytes = tasks * matrixElems * sizeof(T);
            const uint16_t blockCount = static_cast<uint16_t>(totalBytes / blockLen);
            const uint32_t tailBytes = totalBytes % blockLen;
            const uint32_t srcStride =
                unPadHNum * (rowsAlign - rows) * sizeof(T) / kAlignBytes;
            DataCopyExtParams copyParams = {blockCount, blockLen, srcStride, 0, 0};
            DataCopyPad(outputGm_[batchStart * matrixElems], dstLocal, copyParams);
            if (tailBytes > 0) {
                copyParams.blockCount = 1;
                copyParams.blockLen = tailBytes;
                const uint32_t srcOffset = blockCount * unPadHNum * rowsAlign;
                const uint32_t dstOffset = batchStart * matrixElems +
                    blockCount * blockLen / sizeof(T);
                DataCopyPad(outputGm_[dstOffset], dstLocal[srcOffset], copyParams);
            }
        }
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ReshapeTranspose021Batch(LocalTensor<T>& srcLocal,
                                                    LocalTensor<T>& dstLocal,
                                                    uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        DataCopyParams copyParams = {
            static_cast<uint16_t>(rows), static_cast<uint16_t>(colsAlign / blockElems), 0,
            static_cast<uint16_t>(colsAlign * (tasks - 1) / blockElems)};
        for (uint32_t task = 0; task < tasks; ++task) {
            DataCopy(dstLocal[task * colsAlign], srcLocal[task * rows * colsAlign], copyParams);
        }
    }

    __aicore__ inline void TransposeCompute021Batch(LocalTensor<T>& srcLocal,
                                                     LocalTensor<T>& dstLocal,
                                                     uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint8_t repeatTimes = static_cast<uint8_t>(tasks * colsAlign / blockElems);
        const uint16_t dstRepStride = repeatTimes == 1 ? 0 : rowsAlign;
        const uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t rowBlock = 0; rowBlock < rowsAlign / kTile2D16; ++rowBlock) {
            uint64_t srcList[kTile2D16];
            uint64_t dstList[kTile2D16];
            for (uint32_t row = 0; row < kTile2D16; ++row) {
                const uint32_t offset =
                    rowBlock * tasks * colsAlign * kTile2D16 + row * colsAlign * tasks;
                srcList[row] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            for (uint32_t col = 0; col < blockElems; ++col) {
                for (uint32_t half = 0; half < kTile2D16 / blockElems; ++half) {
                    const uint32_t offset = rowBlock * kTile2D16 + col * rowsAlign +
                        half * blockElems;
                    dstList[col * (kTile2D16 / blockElems) + half] =
                        reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
        }
    }

    __aicore__ inline void PadWTranspose021Batch(LocalTensor<T>& srcLocal,
                                                 LocalTensor<T>& dstLocal,
                                                 uint32_t tasks)
    {
        TransposeForPadW1Transpose021Batch(srcLocal, dstLocal, tasks);
        PipeBarrier<PIPE_V>();
        PadWCopyTranspose021Batch(dstLocal, srcLocal, tasks);
        PipeBarrier<PIPE_V>();
        TransposeForPadW2Transpose021Batch(srcLocal, dstLocal, tasks);
    }

    __aicore__ inline void TransposeForPadW1Transpose021Batch(LocalTensor<T>& srcLocal,
                                                              LocalTensor<T>& dstLocal,
                                                              uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t hOnce = kTile2D16 * blockElems;
        const uint32_t transLoop = AlignUp(tasks * rows, hOnce) / hOnce;
        const uint8_t repeatTimes = static_cast<uint8_t>(cols);
        const uint16_t dstRepStride = repeatTimes == 1 ? 0 : kTile2D16;
        const uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t loop = 0; loop < transLoop; ++loop) {
            uint64_t srcList[kTile2D16];
            uint64_t dstList[kTile2D16];
            for (uint32_t row = 0; row < kTile2D16; ++row) {
                const uint32_t offset = loop * hOnce * cols + blockElems * cols * row;
                srcList[row] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                dstList[row] = reinterpret_cast<uint64_t>(
                    dstLocal[loop * hOnce * cols + blockElems * row].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
        }
    }

    __aicore__ inline void PadWCopyTranspose021Batch(LocalTensor<T>& srcLocal,
                                                     LocalTensor<T>& dstLocal,
                                                     uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t cols = inputDims_[2];
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(cols, blockElems);
        const uint32_t hOnce = kTile2D16 * blockElems;
        const uint32_t transLoop = AlignUp(tasks * rows, hOnce) / hOnce;
        DataCopyParams copyParams = {
            static_cast<uint16_t>(transLoop * blockElems),
            static_cast<uint16_t>(cols * kTile2D16 * sizeof(T) / kAlignBytes), 0,
            static_cast<uint16_t>((colsAlign - cols) * kTile2D16 * sizeof(T) / kAlignBytes)};
        DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline void TransposeForPadW2Transpose021Batch(LocalTensor<T>& srcLocal,
                                                              LocalTensor<T>& dstLocal,
                                                              uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint32_t hOnce = kTile2D16 * blockElems;
        const uint32_t transLoop = AlignUp(tasks * rows, hOnce) / hOnce;
        const uint8_t repeatTimes = static_cast<uint8_t>(colsAlign);
        const uint16_t dstRepStride = repeatTimes == 1 ? 0 : 1;
        const uint16_t srcRepStride = repeatTimes == 1 ? 0 : kTile2D16;
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint32_t loop = 0; loop < transLoop; ++loop) {
            uint64_t srcList[kTile2D16];
            uint64_t dstList[kTile2D16];
            for (uint32_t half = 0; half < kTile2D16 / blockElems; ++half) {
                for (uint32_t col = 0; col < blockElems; ++col) {
                    const uint32_t offset = loop * hOnce * colsAlign +
                        half * blockElems + kTile2D16 * col;
                    srcList[half * blockElems + col] =
                        reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                }
            }
            if constexpr (sizeof(T) == 4) {
                for (uint32_t row = 0; row < kTile2D16; row += 2) {
                    const uint32_t offset =
                        loop * hOnce * colsAlign + (row / 2) * blockElems * colsAlign;
                    dstList[row] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
                for (uint32_t row = 1; row < kTile2D16; row += 2) {
                    const uint32_t offset = loop * hOnce * colsAlign +
                        (row / 2 + blockElems) * blockElems * colsAlign;
                    dstList[row] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            } else {
                for (uint32_t row = 0; row < kTile2D16; ++row) {
                    const uint32_t offset =
                        loop * hOnce * colsAlign + row * blockElems * colsAlign;
                    dstList[row] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
        }
    }

    __aicore__ inline void UnPadWTranspose021Batch(LocalTensor<T>& srcLocal,
                                                   LocalTensor<T>& dstLocal,
                                                   uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t cols = inputDims_[2];
        const uint32_t colsAlign = AlignUp(cols, blockElems);
        DataCopyParams copyParams = {
            static_cast<uint16_t>(tasks),
            static_cast<uint16_t>(cols * rowsAlign * sizeof(T) / kAlignBytes),
            static_cast<uint16_t>((colsAlign - cols) * rowsAlign * sizeof(T) / kAlignBytes), 0};
        DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline void UnPadHTranspose021Batch(LocalTensor<T>& srcLocal,
                                                   LocalTensor<T>& dstLocal,
                                                   uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint32_t unPadHNum = tasks * colsAlign / blockElems;
        if (unPadHNum * rowsAlign / kTile2D16 == 1) {
            CopyTranspose021BatchLocal(srcLocal, dstLocal, tasks);
            return;
        }
        TransposeForUnPadH1Transpose021Batch(srcLocal, dstLocal, tasks);
        PipeBarrier<PIPE_V>();
        UnPadHCopyTranspose021Batch(dstLocal, srcLocal, tasks);
        PipeBarrier<PIPE_V>();
        TransposeForUnPadH2Transpose021Batch(srcLocal, dstLocal, tasks);
    }

    __aicore__ inline void TransposeForUnPadH1Transpose021Batch(LocalTensor<T>& srcLocal,
                                                                LocalTensor<T>& dstLocal,
                                                                uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint32_t unPadHNum = tasks * colsAlign / blockElems;
        const uint8_t repeatTimes =
            static_cast<uint8_t>(unPadHNum * rowsAlign / kTile2D16);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, kTile2D16, 2};
        if (repeatTimes == 1) {
            transParams.dstRepStride = 0;
            transParams.srcRepStride = 0;
        }
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        for (uint32_t half = 0; half < kTile2D16 / blockElems; ++half) {
            for (uint32_t col = 0; col < blockElems; ++col) {
                const uint32_t offset = half * blockElems + col * unPadHNum * rowsAlign;
                srcList[half * blockElems + col] =
                    reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
        }
        if constexpr (sizeof(T) == 4) {
            for (uint32_t row = 0; row < kTile2D16; row += 2) {
                dstList[row] = reinterpret_cast<uint64_t>(
                    dstLocal[(row / 2) * blockElems].GetPhyAddr());
            }
            for (uint32_t row = 1; row < kTile2D16; row += 2) {
                dstList[row] = reinterpret_cast<uint64_t>(
                    dstLocal[(row / 2 + blockElems) * blockElems].GetPhyAddr());
            }
        } else {
            transParams.srcRepStride = repeatTimes == 1 ? 0 : 1;
            for (uint32_t row = 0; row < kTile2D16; ++row) {
                dstList[row] = reinterpret_cast<uint64_t>(
                    dstLocal[row * blockElems].GetPhyAddr());
            }
        }
        TransDataTo5HD<T>(dstList, srcList, transParams);
    }

    __aicore__ inline void UnPadHCopyTranspose021Batch(LocalTensor<T>& srcLocal,
                                                       LocalTensor<T>& dstLocal,
                                                       uint32_t tasks)
    {
        const uint32_t rows = inputDims_[1];
        const uint32_t rowsAlign = AlignUp(rows, kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint32_t unPadHNum = tasks * colsAlign / blockElems;
        DataCopyParams copyParams = {
            static_cast<uint16_t>(unPadHNum), static_cast<uint16_t>(rows),
            static_cast<uint16_t>(rowsAlign - rows), 0};
        DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline void TransposeForUnPadH2Transpose021Batch(LocalTensor<T>& srcLocal,
                                                                LocalTensor<T>& dstLocal,
                                                                uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t blockElems = kAlignBytes / sizeof(T);
        const uint32_t colsAlign = AlignUp(inputDims_[2], blockElems);
        const uint32_t unPadHNum = tasks * colsAlign / blockElems;
        const uint8_t repeatTimes =
            static_cast<uint8_t>(unPadHNum * rowsAlign / kTile2D16);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, 2, kTile2D16};
        if (repeatTimes == 1) {
            transParams.dstRepStride = 0;
            transParams.srcRepStride = 0;
        }
        if constexpr (sizeof(T) != 4) {
            transParams.dstRepStride = repeatTimes == 1 ? 0 : 1;
        }
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        for (uint32_t row = 0; row < kTile2D16; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(
                srcLocal[row * blockElems].GetPhyAddr());
        }
        for (uint32_t col = 0; col < blockElems; ++col) {
            for (uint32_t half = 0; half < kTile2D16 / blockElems; ++half) {
                const uint32_t offset = col * unPadHNum * rowsAlign + blockElems * half;
                dstList[col * (kTile2D16 / blockElems) + half] =
                    reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
        }
        TransDataTo5HD<T>(dstList, srcList, transParams);
    }

    __aicore__ inline void CopyTranspose021BatchLocal(LocalTensor<T>& srcLocal,
                                                      LocalTensor<T>& dstLocal,
                                                      uint32_t tasks)
    {
        const uint32_t rowsAlign = AlignUp(inputDims_[1], kTile2D16);
        const uint32_t copyBytes = tasks * inputDims_[2] * rowsAlign * sizeof(T);
        DataCopyParams copyParams = {
            1, static_cast<uint16_t>(AlignUpBytes(copyBytes) / kAlignBytes), 0, 0};
        DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline void ProcessTransposeTileMacro(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                     uint32_t outputStride, uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint16_t>(tileCols * sizeof(T) / kAlignBytes),
            static_cast<uint16_t>((inputStride - tileCols) * sizeof(T) / kAlignBytes), 0};
        DataCopy(srcLocal, inputGm_[inputBase], inParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTile2D16);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTile2D16; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTile2D16;
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
        }
        PipeBarrier<PIPE_V>();
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint16_t>(tileRows * sizeof(T) / kAlignBytes), 0,
            static_cast<uint16_t>((outputStride - tileRows) * sizeof(T) / kAlignBytes)};
        DataCopy(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileMacroPad(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                        uint32_t outputStride, uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint32_t>(tileCols * sizeof(T)),
            static_cast<uint32_t>((inputStride - tileCols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTile2D16);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTile2D16; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTile2D16;
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
        }
        PipeBarrier<PIPE_V>();
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint32_t>(tileRows * sizeof(T)), 0,
            static_cast<uint32_t>((outputStride - tileRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTranspose2DMacroTileB32(uint32_t tileStart, uint32_t tileCount, uint32_t tileRows,
                                                          uint32_t tileCols)
    {
        const uint32_t rows = inputDims_[0];
        const uint32_t cols = inputDims_[1];
        const uint32_t colTiles = cols / tileCols;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t rowTile = tileIndex / colTiles;
            const uint32_t colTile = tileIndex - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            ProcessTransposeTileMacroB32(rowStart * cols + colStart, colStart * rows + rowStart, cols, rows,
                                         tileRows, tileCols);
        }
    }

    // 32 位宏块由基础转置块组合，地址表中的成对位置对应完整 32 位元素的布局。
    __aicore__ inline void ProcessTransposeTileMacroB32(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                        uint32_t outputStride, uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint16_t>(tileCols * sizeof(T) / kAlignBytes),
            static_cast<uint16_t>((inputStride - tileCols) * sizeof(T) / kAlignBytes), 0};
        DataCopy(srcLocal, inputGm_[inputBase], inParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint16_t>(tileRows * sizeof(T) / kAlignBytes), 0,
            static_cast<uint16_t>((outputStride - tileRows) * sizeof(T) / kAlignBytes)};
        DataCopy(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileMacroB32Pad(uint32_t inputBase, uint32_t outputBase,
                                                           uint32_t inputStride, uint32_t outputStride,
                                                           uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint32_t>(tileCols * sizeof(T)),
            static_cast<uint32_t>((inputStride - tileCols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB32Cols);
        TransDataTo5HDParams transParams{
            false, false, repeatTimes, static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows),
            static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1)};
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB32Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB32Rows;
            for (uint32_t i = 0; i < kTileB32Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTileB32Cols; ++i) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dstLocal[srcRowBase + i * tileRows + kTileB32Cols].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transParams);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint32_t>(tileRows * sizeof(T)), 0,
            static_cast<uint32_t>((outputStride - tileRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTranspose2DMacroTileB8(uint32_t tileStart, uint32_t tileCount, uint32_t tileRows,
                                                         uint32_t tileCols)
    {
        const uint32_t rows = inputDims_[0];
        const uint32_t cols = inputDims_[1];
        const uint32_t colTiles = cols / tileCols;
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t tileIndex = tileStart + tile;
            const uint32_t rowTile = tileIndex / colTiles;
            const uint32_t colTile = tileIndex - rowTile * colTiles;
            const uint32_t rowStart = rowTile * tileRows;
            const uint32_t colStart = colTile * tileCols;
            ProcessTransposeTileMacroB8(rowStart * cols + colStart, colStart * rows + rowStart, cols, rows,
                                        tileRows, tileCols);
        }
    }

    // INT8 宏块按字节转置的地址组织覆盖整个矩形，维持输入字节的原始值。
    __aicore__ inline void ProcessTransposeTileMacroB8(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                       uint32_t outputStride, uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint16_t>(tileCols * sizeof(T) / kAlignBytes),
            static_cast<uint16_t>((inputStride - tileCols) * sizeof(T) / kAlignBytes), 0};
        DataCopy(srcLocal, inputGm_[inputBase], inParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB8Cols);
        const uint16_t dstRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB8Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB8Rows;
            const uint32_t dstColBase = (srcRowBase / kTileB8Cols) * kTileB8Cols;
            const bool dstHighHalf = (srcRowBase % kTileB8Cols) != 0;
            for (uint32_t i = 0; i < kTileB8Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
            }

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams transParams00{dstHighHalf, false, repeatTimes, dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, transParams00);
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(
                    dstLocal[(kTile2D16 + i) * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams transParams01{dstHighHalf, true, repeatTimes, dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, transParams01);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint16_t>(tileRows * sizeof(T) / kAlignBytes), 0,
            static_cast<uint16_t>((outputStride - tileRows) * sizeof(T) / kAlignBytes)};
        DataCopy(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileMacroB8Pad(uint32_t inputBase, uint32_t outputBase,
                                                          uint32_t inputStride, uint32_t outputStride,
                                                          uint32_t tileRows, uint32_t tileCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(tileRows), static_cast<uint32_t>(tileCols * sizeof(T)),
            static_cast<uint32_t>((inputStride - tileCols) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB8Cols);
        const uint16_t dstRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB8Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB8Rows;
            const uint32_t dstColBase = (srcRowBase / kTileB8Cols) * kTileB8Cols;
            const bool dstHighHalf = (srcRowBase % kTileB8Cols) != 0;
            for (uint32_t i = 0; i < kTileB8Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
            }

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(dstLocal[i * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams transParams00{dstHighHalf, false, repeatTimes, dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, transParams00);
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(
                    dstLocal[(kTile2D16 + i) * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams transParams01{dstHighHalf, true, repeatTimes, dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, transParams01);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(tileCols), static_cast<uint32_t>(tileRows * sizeof(T)), 0,
            static_cast<uint32_t>((outputStride - tileRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileMacroB8PadTail(
        uint32_t inputBase, uint32_t outputBase, uint32_t inputStride, uint32_t outputStride,
        uint32_t tileRows, uint32_t tileCols, uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        const uint32_t inputBytes = validCols * sizeof(T);
        const uint32_t alignedInputBytes = AlignUpBytes(inputBytes);
        const uint32_t rightPadding = (alignedInputBytes - inputBytes) / sizeof(T);
        const uint32_t inputDstStride =
            (tileCols * sizeof(T) - alignedInputBytes) / kAlignBytes;
        DataCopyExtParams inParams = {
            static_cast<uint16_t>(validRows), inputBytes,
            static_cast<uint32_t>((inputStride - validCols) * sizeof(T)), inputDstStride, 0};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
        DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / kTileB8Cols);
        const uint16_t dstRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : tileRows);
        const uint16_t srcRepStride = static_cast<uint16_t>(repeatTimes == 1 ? 0 : 1);
        for (uint32_t rowBlock = 0; rowBlock < tileRows / kTileB8Rows; ++rowBlock) {
            const uint32_t srcRowBase = rowBlock * kTileB8Rows;
            const uint32_t dstColBase = (srcRowBase / kTileB8Cols) * kTileB8Cols;
            const bool dstHighHalf = (srcRowBase % kTileB8Cols) != 0;
            for (uint32_t i = 0; i < kTileB8Rows; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(
                    srcLocal[(srcRowBase + i) * tileCols].GetPhyAddr());
            }
            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(
                    dstLocal[i * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams lowParams{dstHighHalf, false, repeatTimes,
                                           dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, lowParams);
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < kTile2D16; ++i) {
                dstList[i] = reinterpret_cast<uint64_t>(
                    dstLocal[(kTile2D16 + i) * tileRows + dstColBase].GetPhyAddr());
            }
            TransDataTo5HDParams highParams{dstHighHalf, true, repeatTimes,
                                            dstRepStride, srcRepStride};
            TransDataTo5HD<T>(dstList, srcList, highParams);
            PipeBarrier<PIPE_V>();
        }
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        const uint32_t outputBytes = validRows * sizeof(T);
        const uint32_t localOutputRowBytes = tileRows * sizeof(T);
        const uint32_t outputSrcStride =
            (localOutputRowBytes - AlignUpBytes(outputBytes)) / kAlignBytes;
        DataCopyExtParams outParams = {
            static_cast<uint16_t>(validCols), outputBytes, outputSrcStride,
            static_cast<uint32_t>((outputStride - validRows) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileB32(uint32_t inputBase, uint32_t outputBase, uint32_t inputStride,
                                                   uint32_t outputStride, uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<T> srcLocal = tileInQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if (validCols == kTileB32Cols) {
            DataCopyExtParams inParams = {
                static_cast<uint16_t>(validRows), static_cast<uint32_t>(kTileB32Cols * sizeof(T)),
                static_cast<uint32_t>((inputStride - kTileB32Cols) * sizeof(T)), 0, 0};
            DataCopyPad(srcLocal, inputGm_[inputBase], inParams, padParams);
        } else {
            for (uint32_t row = 0; row < validRows; ++row) {
                DataCopyExtParams inParams = {1, static_cast<uint32_t>(validCols * sizeof(T)), 0, 0, 0};
                DataCopyPad(srcLocal[row * kTileB32Cols], inputGm_[inputBase + row * inputStride], inParams,
                            padParams);
            }
        }
        tileInQueue_.EnQue(srcLocal);
        srcLocal = tileInQueue_.DeQue<T>();

        LocalTensor<T> dstLocal = tileOutQueue_.AllocTensor<T>();
        uint64_t srcList[kTile2D16];
        uint64_t dstList[kTile2D16];
        for (uint32_t i = 0; i < kTile2D16; ++i) {
            srcList[i] = reinterpret_cast<uint64_t>(srcLocal[i * kTileB32Cols].GetPhyAddr());
        }
        for (uint32_t i = 0; i < kTileB32Cols; ++i) {
            dstList[i * 2] = reinterpret_cast<uint64_t>(dstLocal[i * kTileB32Rows].GetPhyAddr());
            dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dstLocal[i * kTileB32Rows + kTileB32Cols].GetPhyAddr());
        }
        TransDataTo5HDParams transParams{false, false, 1, 0, 0};
        TransDataTo5HD<T>(dstList, srcList, transParams);
        PipeBarrier<PIPE_V>();
        tileOutQueue_.EnQue(dstLocal);
        tileInQueue_.FreeTensor(srcLocal);

        dstLocal = tileOutQueue_.DeQue<T>();
        if (validRows == kTileB32Rows) {
            DataCopyExtParams outParams = {
                static_cast<uint16_t>(validCols), static_cast<uint32_t>(kTileB32Rows * sizeof(T)), 0,
                static_cast<uint32_t>((outputStride - kTileB32Rows) * sizeof(T)), 0};
            DataCopyPad(outputGm_[outputBase], dstLocal, outParams);
        } else {
            for (uint32_t row = 0; row < validCols; ++row) {
                DataCopyExtParams outParams = {1, static_cast<uint32_t>(validRows * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm_[outputBase + row * outputStride], dstLocal[row * kTileB32Rows], outParams);
            }
        }
        tileOutQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessTransposeTileB8TailStrided(uint32_t inputBase, uint32_t outputBase,
                                                             uint32_t inputStride, uint32_t outputStride,
                                                             uint32_t validRows, uint32_t validCols)
    {
        for (uint32_t col = 0; col < validCols; ++col) {
            LocalTensor<T> local = copyQueue_.AllocTensor<T>();
            for (uint32_t row = 0; row < validRows; ++row) {
                local.SetValue(row, inputGm_.GetValue(inputBase + row * inputStride + col));
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<T>();
            CopyOutPad(outputGm_, outputBase + col * outputStride, local, validRows);
            copyQueue_.FreeTensor(local);
        }
    }

    // 通用兜底：按输出坐标逐元素读取输入，先收集到 UB，再连续写出；适用性优先于热点性能。
    __aicore__ inline void ProcessGeneric(uint32_t start, uint32_t count)
    {
        for (uint32_t offset = 0; offset < count; offset += tileLength_) {
            uint32_t tileLen = count - offset;
            if (tileLen > tileLength_) {
                tileLen = tileLength_;
            }
            LocalTensor<T> local = copyQueue_.AllocTensor<T>();
            uint32_t coords[kMaxRank] = {};
            uint32_t srcOffset = OutputIndexToInputOffsetWithCoords(start + offset, coords);
            for (uint32_t i = 0; i < tileLen; ++i) {
                local.SetValue(i, inputGm_.GetValue(srcOffset));
                AdvanceOutputCoords(coords, srcOffset);
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<T>();
            CopyOutPad(outputGm_, start + offset, local, tileLen);
            copyQueue_.FreeTensor(local);
        }
    }

    // 输出线性下标先按 outputStrides_ 展开，再用 perm_ 对应的输入步长累加源地址。
    __aicore__ inline uint32_t OutputIndexToInputOffsetWithCoords(uint32_t outputIndex,
                                                                  uint32_t (&coords)[kMaxRank])
    {
        uint32_t srcOffset = 0;
        uint32_t remaining = outputIndex;
        for (uint32_t axis = 0; axis < rank_; ++axis) {
            const uint32_t stride = outputStrides_[axis];
            const uint32_t coord = remaining / stride;
            remaining -= coord * stride;
            coords[axis] = coord;
            srcOffset += coord * inputStrides_[perm_[axis]];
        }
        return srcOffset;
    }

    // 按混合进制对输出坐标加一，并增量更新输入偏移，避免每个元素都重新做全部除法。
    __aicore__ inline void AdvanceOutputCoords(uint32_t (&coords)[kMaxRank], uint32_t& srcOffset)
    {
        for (int32_t axis = static_cast<int32_t>(rank_) - 1; axis >= 0; --axis) {
            const uint32_t inputStride = inputStrides_[perm_[axis]];
            ++coords[axis];
            srcOffset += inputStride;
            if (coords[axis] < outputDims_[axis]) {
                return;
            }
            srcOffset -= coords[axis] * inputStride;
            coords[axis] = 0;
        }
    }

private:
    // 绑定队列让纯搬运输入/输出共享同一 UB Tensor；独立输入/输出队列用于需要转置的路径。
    // 模板队列深度与 InitBuffer 的物理块数分别描述在途管理和实际存储容量。
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copyQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> batchCopyQueue_;
    TQue<TPosition::VECIN, 1> tileInQueue_;
    TQue<TPosition::VECOUT, 1> tileOutQueue_;
    TQue<TPosition::VECIN, 2> batchInQueue_;
    TQue<TPosition::VECOUT, 2> batchOutQueue_;
    TBuf<TPosition::VECCALC> gatherIndexBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    uint32_t rank_ = 0;
    uint32_t mode_ = 0;
    uint32_t totalLength_ = 0;
    uint32_t outputBufferLength_ = 0;
    uint32_t taskLength_ = 0;
    uint32_t taskStride_ = 1;
    uint32_t directDimA_ = 0;
    uint32_t directDimB_ = 0;
    uint32_t coreSplitA_ = 0;
    uint32_t coreSplitB_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t innerBlock_ = 1;
    uint32_t swapAxis_ = 0;
    uint32_t swapOuter_ = 1;
    uint32_t swapDimA_ = 1;
    uint32_t swapDimB_ = 1;
    uint32_t swapInner_ = 1;
    uint32_t swapTileA_ = 1;
    uint32_t swapTileB_ = 1;
    bool outputLineAligned_ = false;
    uint32_t inputDims_[kMaxRank] = {};
    uint32_t outputDims_[kMaxRank] = {};
    uint32_t inputStrides_[kMaxRank] = {};
    uint32_t outputStrides_[kMaxRank] = {};
    uint32_t perm_[kMaxRank] = {};
};

// 32 位 021 的 workspace 路径：每核独占 scratch 和 LUT，不修改输入 GM。
// 第一阶段把转置结果放入每列 768 个元素的物理行距，第二阶段去掉列尾间隔并连续写出。
// 128 个 32 位元素对应 512B；输出边界按真实 GM 地址对齐，最后一段受 total_ 限制。
// kAlignedBases 由入口检查真实指针决定，非 512B 基址通过偏移修正继续使用该算法。
template <bool kAlignedBases>
class KernelTransposeAlignedWorkspace {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const TransposeTilingData& data)
    {
        rows_ = data.inputDims[1];
        cols_ = data.inputDims[2];
        batches_ = data.inputDims[0];
        batchElems_ = rows_ * cols_;
        total_ = data.totalLength;
        if constexpr (!kAlignedBases) {
            inputBias_ = (reinterpret_cast<uint64_t>(input) % 512) / 4;
            outputBias_ = (reinterpret_cast<uint64_t>(output) % 512) / 4;
        }
        // workspace 先按 512B 对齐，再按核划分 scratch；后接每核独立 LUT。
        // 额外列空间为压缩窗口及边界搬运预留，Host 申请大小必须与这里的布局一致。
        const uint32_t scratchElems = AlignUp(cols_ + 12, 16) * 768;
        const uint64_t scratchBase = (reinterpret_cast<uint64_t>(workspace) + 511) & ~511ULL;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), total_);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), total_);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scratchBase) +
                                GetBlockIdx() * scratchElems, scratchElems);
        lut_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratchBase) +
                            GetBlockNum() * scratchElems + GetBlockIdx() * 8960, 8960);
        // 主阶段输入/最终输出是流式访问；scratch 和 LUT 保持默认缓存以服务当前调用内的二次读取。
        input_.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        output_.SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void Process()
    {
        PreparePackedLut();
        const uint32_t cores = GetBlockNum();
        // 完整轮次按 batch 交错分配给各核；scratch 写完后用 MTE3_MTE2 连接压紧阶段。
        const uint32_t complete = batches_ / cores * cores;
        for (uint32_t batch = GetBlockIdx(); batch < complete; batch += cores) {
            TransposeToScratch(batch, 0, cols_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            PackOutput(batch, OutputBoundary(batch * batchElems_),
                       OutputBoundary((batch + 1) * batchElems_));
        }
        const uint32_t remaining = batches_ - complete;
        if (remaining == 0) {
            return;
        }
        // 尾批次由多核读取同一份只读输入，因此在此阶段恢复输入 L2 缓存。
        input_.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_NORMAL);
        // 最后不足一轮的 batch 按物理对齐的输出区间分摊到多核，各核写区间互不重叠。
        const uint32_t group = GetBlockIdx() % remaining;
        const uint32_t lane = GetBlockIdx() / remaining;
        const uint32_t lanes = (cores - 1 - group) / remaining + 1;
        const uint32_t batch = complete + group;
        const uint32_t batchBegin = OutputBoundary(batch * batchElems_);
        const uint32_t batchEnd = OutputBoundary((batch + 1) * batchElems_);
        const uint32_t units = (batchEnd - batchBegin + 127) / 128;
        const uint32_t begin = lane == 0 ? batchBegin :
            Min(batchEnd, OutputBoundary(batchBegin + units * lane / lanes * 128));
        const uint32_t end = lane + 1 == lanes ? batchEnd :
            Min(batchEnd, OutputBoundary(batchBegin + units * (lane + 1) / lanes * 128));
        if (begin < end) {
            const uint32_t columnBegin = (begin - batch * batchElems_) / rows_;
            const uint32_t columnEnd = (Min(end, (batch + 1) * batchElems_) -
                                       batch * batchElems_ + rows_ - 1) / rows_;
            TransposeTailToScratch(batch, columnBegin, columnEnd);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            PackOutput(batch, begin, end);
        }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    // 在总输出范围内上取整到物理 512B 边界；非对齐基址先加 outputBias_，对齐后再减回。
    __aicore__ inline uint32_t OutputBoundary(uint32_t pos)
    {
        if constexpr (kAlignedBases) {
            return Min(total_, AlignUp(pos, 128));
        }
        return pos == 0 ? 0 : Min(total_, AlignUp(pos + outputBias_, 128) - outputBias_);
    }

    // 正常输出块最多 8192 个元素，即 32KiB；非对齐首块先写到第一个物理 512B 边界。
    __aicore__ inline uint32_t ChunkCount(uint32_t pos, uint32_t end)
    {
        if constexpr (!kAlignedBases) {
            if (pos == 0 && outputBias_ != 0) {
                return Min(128 - outputBias_, end);
            }
        }
        return Min(8192, end - pos);
    }

    // 生成紧凑逻辑位置 p 对应的 scratch 字节偏移：(p + floor(p/rows_)*(768-rows_))*4。
    // 浮点运算仅用于这个有界索引表的生成，不作用于待转置的数据值。
    // LUT 每核生成一次并写入自己的 workspace；MTE3_MTE2 确保后续读取能看到已写完的表。
    __aicore__ inline void PreparePackedLut()
    {
        LocalTensor<float> temp(TPosition::VECCALC, 0, 8960);
        LocalTensor<int32_t> ix(TPosition::VECCALC, 35840, 8960);
        auto q = temp.ReinterpretCast<int32_t>();
        CreateVecIndex(temp, 0.0f, 8960);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, 1.0f / static_cast<float>(static_cast<int32_t>(rows_)), 8960);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, 0.0001f, 8960);
        PipeBarrier<PIPE_V>();
        Cast(ix, temp, RoundMode::CAST_FLOOR, 8960);
        PipeBarrier<PIPE_V>();
        Muls(ix, ix, static_cast<int32_t>(768 - rows_), 8960);
        CreateVecIndex(q, int32_t(0), 8960);
        PipeBarrier<PIPE_V>();
        Add(ix, ix, q, 8960);
        PipeBarrier<PIPE_V>();
        Muls(ix, ix, int32_t(4), 8960);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyExtParams write{1, 35840, 0, 0, 0};
        DataCopyPad(lut_, ix, write);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    }

    // 构造 32 列×64 行的 Gather 字节索引；不足 64 行时把无效行索引夹到最后一条有效行。
    __aicore__ inline void PrepareOffsets(LocalTensor<uint32_t>& offsets, uint32_t validRows)
    {
        auto ix = offsets.ReinterpretCast<int32_t>();
        CreateVecIndex(ix, int32_t(0), 64);
        PipeBarrier<PIPE_V>();
        if (validRows != 64) {
            Mins(ix, ix, static_cast<int32_t>(validRows - 1), 64);
            PipeBarrier<PIPE_V>();
        }
        Muls(ix, ix, static_cast<int32_t>(cols_ * 4), 64);
        PipeBarrier<PIPE_V>();
        for (uint32_t filled = 1; filled < 32; filled *= 2) {
            Adds(ix[filled * 64], ix, static_cast<int32_t>(filled * 4), filled * 64);
            PipeBarrier<PIPE_V>();
        }
    }

    // 主批次使用单输入槽、双输出槽：一次读取多行，按 64 列分段 Gather 后写到独占 scratch。
    // 对齐扩展的输入范围始终裁剪到合法 GM 范围；Gather 起点补偿对齐产生的前缀。
    __aicore__ inline void TransposeToScratch(uint32_t batch, uint32_t columnBegin,
                                             uint32_t columnEnd)
    {
        LocalTensor<float> src(TPosition::VECCALC, 0, 136192 / 4);
        LocalTensor<uint32_t> ix(TPosition::VECCALC, 168960, 2048);
        PrepareOffsets(ix, 64);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        uint32_t chunk = 0;
        for (uint32_t row = 0; row < rows_; row += 64) {
            const uint32_t validRows = Min(64, rows_ - row);
            if (validRows != 64) {
                PrepareOffsets(ix, validRows);
            }
            const uint32_t begin = batch * batchElems_ + row * cols_;
            uint32_t aligned = begin / 128 * 128;
            uint32_t end = Min(total_, AlignUp(begin + validRows * cols_, 128));
            uint32_t srcShift = 0;
            if constexpr (!kAlignedBases) {
                const uint32_t physicalStart = (begin + inputBias_) / 128 * 128;
                aligned = physicalStart < inputBias_ ? 0 : physicalStart - inputBias_;
                end = Min(total_, AlignUp(begin + validRows * cols_ + inputBias_, 128) - inputBias_);
                if (physicalStart < inputBias_) {
                    srcShift = inputBias_ % 8;
                }
            }
            const uint32_t count = end - aligned;
            DataCopyExtParams read{1, count * 4, 0, 0, 0};
            const uint32_t pad = AlignUp(count, 8) - count;
            DataCopyPadExtParams<float> padding{pad != 0, 0, static_cast<uint8_t>(pad), 0};
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            if constexpr (kAlignedBases) {
                DataCopyPad(src, input_[aligned], read, padding);
            } else {
                if (aligned == 0 && inputBias_ != 0) {
                    const uint32_t headCount = Min(count, 128 - inputBias_);
                    DataCopyExtParams headRead{1, headCount * 4, 0, 0, 0};
                    DataCopyPadExtParams<float> headPad{true, static_cast<uint8_t>(srcShift), 0, 0};
                    DataCopyPad(src, input_[0], headRead, headPad);
                    if (headCount < count) {
                        const uint32_t bodyCount = count - headCount;
                        const uint32_t bodyPadCount = AlignUp(bodyCount, 8) - bodyCount;
                        DataCopyExtParams bodyRead{1, bodyCount * 4, 0, 0, 0};
                        DataCopyPadExtParams<float> bodyPad{bodyPadCount != 0, 0,
                            static_cast<uint8_t>(bodyPadCount), 0};
                        DataCopyPad(src[headCount + srcShift], input_[headCount], bodyRead, bodyPad);
                    }
                } else {
                    DataCopyPad(src, input_[aligned], read, padding);
                }
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t col = columnBegin; col < columnEnd; col += 64, ++chunk) {
                const int32_t event = (chunk & 1) ? EVENT_ID1 : EVENT_ID0;
                LocalTensor<float> dst(TPosition::VECCALC, 136192 + (chunk & 1) * 16384, 4096);
                const uint32_t validCols = Min(64, columnEnd - col);
                WaitFlag<HardEvent::MTE3_V>(event);
                for (uint32_t group = 0; group < validCols; group += 32) {
                    Gather(dst[group * 64], src, ix, (begin - aligned + srcShift + col + group) * 4,
                           Min(32, validCols - group) * 64);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(event);
                WaitFlag<HardEvent::V_MTE3>(event);
                DataCopyExtParams write{static_cast<uint16_t>(validCols), validRows * 4,
                    (64 - AlignUp(validRows, 8)) / 8, (768 - validRows) * 4, 0};
                DataCopyPad(scratch_[col * 768 + row], dst, write);
                SetFlag<HardEvent::MTE3_V>(event);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    // 尾波读取的 UB 行距固定为 264 个元素，与主路径的原始 cols_ 行距不同，须重新生成索引。
    __aicore__ inline void PrepareTailOffsets(LocalTensor<uint32_t>& offsets, uint32_t validRows)
    {
        auto ix = offsets.ReinterpretCast<int32_t>();
        CreateVecIndex(ix, int32_t(0), 64);
        PipeBarrier<PIPE_V>();
        if (validRows != 64) {
            Mins(ix, ix, static_cast<int32_t>(validRows - 1), 64);
            PipeBarrier<PIPE_V>();
        }
        Muls(ix, ix, static_cast<int32_t>(264 * 4), 64);
        PipeBarrier<PIPE_V>();
        for (uint32_t filled = 1; filled < 32; filled *= 2) {
            Adds(ix[filled * 64], ix, static_cast<int32_t>(filled * 4), filled * 64);
            PipeBarrier<PIPE_V>();
        }
    }

    // 读取当前核负责的列带，最多 256 列；物理行距保留到 264 个元素，输入槽由 V_MTE2 保护。
    __aicore__ inline void LoadTailColumnBand(uint32_t batch, uint32_t row,
                                          uint32_t col, uint32_t width, uint32_t slot)
    {
        const int32_t event = slot ? EVENT_ID1 : EVENT_ID0;
        LocalTensor<float> src(TPosition::VECCALC, slot * 67584, 16896);
        const uint32_t validRows = Min(64, rows_ - row);
        WaitFlag<HardEvent::V_MTE2>(event);
        DataCopyExtParams read{static_cast<uint16_t>(validRows), width * 4,
                              (cols_ - width) * 4, (264 - AlignUp(width, 8)) / 8, 0};
        const uint32_t pad = AlignUp(width, 8) - width;
        DataCopyPadExtParams<float> padding{pad != 0, 0, static_cast<uint8_t>(pad), 0};
        DataCopyPad(src, input_[batch * batchElems_ + row * cols_ + col], read, padding);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    // 尾波按列区间分担 batch，输入列带 ping/pong 预取、输出列段 ping/pong 写回，避免只让少数核收尾。
    __aicore__ inline void TransposeTailToScratch(uint32_t batch, uint32_t columnBegin,
                                             uint32_t columnEnd)
    {
        LocalTensor<uint32_t> ix(TPosition::VECCALC, 167936, 2048);
        PrepareTailOffsets(ix, 64);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        LoadTailColumnBand(batch, 0, columnBegin, Min(256, columnEnd - columnBegin), 0);
        uint32_t inputSlot = 0;
        uint32_t chunk = 0;
        for (uint32_t row = 0; row < rows_; row += 64) {
            const uint32_t validRows = Min(64, rows_ - row);
            if (validRows != 64) { PrepareTailOffsets(ix, validRows); }
            for (uint32_t band = columnBegin; band < columnEnd; band += 256) {
                const uint32_t width = Min(256, columnEnd - band);
                const int32_t inputEvent = inputSlot ? EVENT_ID1 : EVENT_ID0;
                LocalTensor<float> src(TPosition::VECCALC, inputSlot * 67584, 16896);
                if (band + width < columnEnd) {
                    LoadTailColumnBand(batch, row, band + width,
                                   Min(256, columnEnd - band - width), inputSlot ^ 1);
                } else if (row + validRows < rows_) {
                    LoadTailColumnBand(batch, row + validRows, columnBegin,
                                   Min(256, columnEnd - columnBegin), inputSlot ^ 1);
                }
                WaitFlag<HardEvent::MTE2_V>(inputEvent);
                for (uint32_t col = 0; col < width; col += 64, ++chunk) {
                    const int32_t event = (chunk & 1) ? EVENT_ID1 : EVENT_ID0;
                    LocalTensor<float> dst(TPosition::VECCALC, 135168 + (chunk & 1) * 16384, 4096);
                    const uint32_t validCols = Min(64, width - col);
                    WaitFlag<HardEvent::MTE3_V>(event);
                    for (uint32_t group = 0; group < validCols; group += 32) {
                        Gather(dst[group * 64], src, ix, (col + group) * 4,
                               Min(32, validCols - group) * 64);
                        PipeBarrier<PIPE_V>();
                    }
                    SetFlag<HardEvent::V_MTE3>(event);
                    WaitFlag<HardEvent::V_MTE3>(event);
                    DataCopyExtParams write{static_cast<uint16_t>(validCols), validRows * 4,
                        (64 - AlignUp(validRows, 8)) / 8, (768 - validRows) * 4, 0};
                    DataCopyPad(scratch_[(band + col) * 768 + row], dst, write);
                    SetFlag<HardEvent::MTE3_V>(event);
                }
                SetFlag<HardEvent::V_MTE2>(inputEvent);
                inputSlot ^= 1;
            }
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    // 按逻辑输出块覆盖的列范围读取带 768 行距的 scratch。
    // 若对齐写区间跨过 batch 边界，额外从下一 batch 的只读输入提取前缀，避免依赖其他核的 scratch。
    __aicore__ inline void LoadPackedChunk(uint32_t batch, uint32_t pos, uint32_t count,
                                          uint32_t slot)
    {
        const int32_t event = slot ? EVENT_ID1 : EVENT_ID0;
        LocalTensor<float> src(TPosition::VECCALC, slot * 40960, 10240);
        const uint32_t relative = pos - batch * batchElems_;
        const uint32_t column = relative / rows_;
        const uint32_t inBatch = Min(count, (batch + 1) * batchElems_ - pos);
        const uint32_t columns = (relative % rows_ + inBatch + rows_ - 1) / rows_;
        WaitFlag<HardEvent::V_MTE2>(event);
        DataCopyExtParams read{1, columns * 768 * 4, 0, 0, 0};
        DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        DataCopyPad(src, scratch_[column * 768], read, padding);
        if (count > inBatch) {
            DataCopyExtParams boundary{static_cast<uint16_t>(count - inBatch), 4,
                                      (cols_ - 1) * 4, 0, 0};
            DataCopyPadExtParams<float> boundaryPad{true, 0, 7, 0};
            DataCopyPad(src[9216], input_[(batch + 1) * batchElems_], boundary, boundaryPad);
        }
        SetFlag<HardEvent::MTE2_V>(event);
    }

    // 一列 768 个位置使用 24 个 32 位掩码字；最后一字仅保留有效行，屏蔽列尾物理间隔。
    __aicore__ inline uint32_t PackedMaskWord(uint32_t word)
    {
        return word % 24 != 23 || rows_ == 768 ? 0xffffffffU : (1U << (rows_ - 736)) - 1;
    }

    // 预先准备可重复使用的列尾掩码；常规输出块复用此表，减少逐块构造索引的工作。
    __aicore__ inline void PreparePackedMask(LocalTensor<uint32_t>& mask)
    {
        Duplicate(mask, uint32_t(0xffffffff), 320);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t column = 0; column < 12; ++column) {
            mask.SetValue(column * 24 + 23, PackedMaskWord(23));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    // GMASK：用 GatherMask 去掉 scratch 列尾间隔，同时屏蔽当前起点之前的元素。
    // 源起点下取整到 256 个元素以匹配掩码寻址；结束后恢复临时改动的掩码字。
    // extent 限制扫描终点，因此只需临时清除首部掩码，无需为每块重新生成整张表。
    // V_S/S_V 分别保护向量读取结束和标量写入完成，不能直接删去。
    __aicore__ inline void CompactPackedChunk(LocalTensor<float>& dst,
                                              LocalTensor<float>& src,
                                              LocalTensor<uint32_t>& mask,
                                              uint32_t rowOffset, uint32_t count)
    {
        const uint32_t start = rowOffset / 256 * 256;
        const uint32_t firstWord = start / 32;
        const uint32_t prefix = rowOffset - start;
        const uint32_t words = (prefix + 31) / 32;
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = 0; i < words; ++i) {
            const uint32_t remaining = prefix - i * 32;
            const uint32_t original = PackedMaskWord(firstWord + i);
            mask.SetValue(firstWord + i, remaining >= 32 ? 0 : original & (0xffffffffU << remaining));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const uint32_t last = rowOffset + count - 1;
        const uint32_t extent = last / rows_ * 768 + last % rows_ + 1 - start;
        uint64_t selected = 0;
        GatherMask(dst, src[start], mask[firstWord], true, extent,
                   GatherMaskParams{1, 1, 8, 0}, selected);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = 0; i < words; ++i) {
            mask.SetValue(firstWord + i, PackedMaskWord(firstWord + i));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    // 第二阶段使用双输入/双输出槽：预取下一块 scratch，压紧当前块，MTE3 写出前一块。
    // 正常块走 GatherMask；跨 batch 边界的块改用 LUT+Gather，补齐下一 batch 前缀后完成同一输出区间。
    // V_MTE2 释放输入槽，MTE3_V 释放输出槽，EVENT_ID2 单独保护边界 Gather 的索引表。
    __aicore__ inline void PackOutput(uint32_t batch, uint32_t begin, uint32_t end)
    {
        LocalTensor<int32_t> ix(TPosition::VECCALC, 147456, 8192);
        auto gatherIx = ix.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> mask(TPosition::VECCALC, 180224, 320);
        PreparePackedMask(mask);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        LoadPackedChunk(batch, begin, ChunkCount(begin, end), 0);
        uint32_t chunk = 0;
        for (uint32_t pos = begin; pos < end; ++chunk) {
            const uint32_t slot = chunk & 1;
            const int32_t event = slot ? EVENT_ID1 : EVENT_ID0;
            const uint32_t count = ChunkCount(pos, end);
            const uint32_t inBatch = Min(count, (batch + 1) * batchElems_ - pos);
            const uint32_t rowOffset = (pos - batch * batchElems_) % rows_;
            LocalTensor<float> src(TPosition::VECCALC, slot * 40960, 10240);
            LocalTensor<float> dst(TPosition::VECCALC, 81920 + slot * 32768, 8192);
            WaitFlag<HardEvent::MTE2_V>(event);
            if (count > inBatch) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                const uint32_t indexPad = AlignUp(count, 8) - count;
                DataCopyExtParams indexRead{1, count * 4, 0, 0, 0};
                DataCopyPadExtParams<int32_t> indexPadding{
                    indexPad != 0, 0, static_cast<uint8_t>(indexPad), 0};
                DataCopyPad(ix, lut_[rowOffset], indexRead, indexPadding);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            }
            if (pos + count < end) {
                LoadPackedChunk(batch, pos + count, ChunkCount(pos + count, end), slot ^ 1);
            }
            WaitFlag<HardEvent::MTE3_V>(event);
            if (count > inBatch) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t i = inBatch; i < count; ++i) {
                    ix.SetValue(i, static_cast<int32_t>(36864 + (i - inBatch) * 32));
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Gather(dst, src, gatherIx, uint32_t(0), count);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
            } else {
                CompactPackedChunk(dst, src, mask, rowOffset, count);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(event);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            DataCopyExtParams write{1, count * 4, 0, 0, 0};
            DataCopyPad(output_[pos], dst, write);
            SetFlag<HardEvent::MTE3_V>(event);
            if constexpr (kAlignedBases) {
                pos += 8192;
            } else {
                pos += count;
            }
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    GlobalTensor<float> input_, output_, scratch_;
    GlobalTensor<int32_t> lut_;
    uint32_t rows_, cols_, batches_, batchElems_, total_;
    uint32_t inputBias_ = 0, outputBias_ = 0;
};

// 将 TPipe 放在 Kernel 对象之外，统一拥有当前调用的缓冲和事件资源。
template <typename T, bool kInputL2Bypass>
__aicore__ inline void RunTranspose(GM_ADDR input, GM_ADDR output, const TransposeTilingData& tilingData)
{
    TPipe pipe;
    KernelTranspose<T, kInputL2Bypass> kernel;
    kernel.Init(&pipe, input, output, nullptr, tilingData);
    kernel.Process();
}

}  // namespace

// 入口 key0/key1 对应普通实现的输入 L2 默认/旁路策略，key2 对应 workspace 对齐写路径。
// 即使 key2 的主体使用静态 Tensor，仍保留 TPipe，供 CreateVecIndex 等接口内部事件分配使用。
extern "C" __global__ __aicore__ void transpose(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }
    if (TILING_KEY_IS(0)) {
        RunTranspose<DTYPE_INPUT, false>(input, output, tilingData);
        return;
    }
    if (TILING_KEY_IS(1)) {
        RunTranspose<DTYPE_INPUT, true>(input, output, tilingData);
    }
    if (TILING_KEY_IS(2)) {
        // 真实 GM 指针不满足 4B 元素对齐时退回普通实现；512B 对齐与否在下一级分别处理。
        if ((reinterpret_cast<uint64_t>(input) % 4) != 0 ||
            (reinterpret_cast<uint64_t>(output) % 4) != 0) {
            RunTranspose<DTYPE_INPUT, true>(input, output, tilingData);
            return;
        }
        SetSysWorkspace(workspace);
        TPipe pipe;
        if (((reinterpret_cast<uint64_t>(input) | reinterpret_cast<uint64_t>(output)) % 512) == 0) {
            KernelTransposeAlignedWorkspace<true> kernel;
            kernel.Init(input, output, GetUserWorkspace(workspace), tilingData);
            kernel.Process();
        } else {
            KernelTransposeAlignedWorkspace<false> kernel;
            kernel.Init(input, output, GetUserWorkspace(workspace), tilingData);
            kernel.Process();
        }
    }
}
