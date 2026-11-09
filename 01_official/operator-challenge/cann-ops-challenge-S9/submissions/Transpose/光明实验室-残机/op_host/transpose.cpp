// Transpose 算子 host 侧：tiling 计算与算子注册。
// 整体数据流：TilingFunc 读取输入 shape 与 dims 属性，先剔除 size=1 的轴，
// 再依 perm 形态、dtype 与 UB 容量依次尝试各专用路径(2D macro / 021 系列 /
// flatten / reverse3d gather / block copy / block swap / last-axis 等)，
// 为命中路径计算任务数、tile 尺寸、核间切分与 L2 bypass 标记，全部结果
// 打包进 TransposeTilingData(见 transpose_tiling.h)下发给 device；所有专用
// 形态都不命中时退回 kModeGeneric 逐元素路径。
// 文件结构：
//   1. 匿名 namespace：模式常量、容量/对齐参数与各路径的 tile 选择 helper；
//   2. optiling::TilingFunc：主决策流程(路径匹配 -> blockDim -> tileLength -> 下发)；
//   3. ge::InferShape / InferDataType：输出 shape 与 dtype 推导；
//   4. ops::Transpose：OpDef 注册(输入输出、dims 属性、tiling 绑定)。
// device 侧对应实现在 op_kernel/transpose.cpp，两边的 mode 编号必须一致。

#include "tiling/tiling_api.h"
#include "transpose_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

// 执行路径编号，写入 tiling.mode，kernel 侧据此分发到对应实现：
// 0 BlockCopy：输出末尾连续块的整体拷贝(含 4D reverse 的二维块拷贝形态)；
// 1 Generic：通用逐元素路径，按 inputStrides/outputStrides 计算 GM 地址；
// 2 Transpose021Fp16：rank3 [0,2,1] 的行列分块 tile 化路径(名称沿袭 fp16，覆盖各 dtype)；
// 3 Transpose2DMacro：rank2 [1,0] 的整除型 macro tile 转置；
// 4 TransposeFlattenTiled：把若干轴折叠成二维转置(如 [1,2,0]/[2,0,1])；
// 5 LastAxisTransposeTiled：最后一轴被换走的 tile 化转置；
// 6 Reverse3DSuffixGather：rank4 [2,1,0,3] 的 gather 路径(内层不对齐时启用)；
// 7 Transpose021Macro：rank3 [0,2,1] 的整除型 macro tile 转置；
// 8 Transpose021Batch：rank3 [0,2,1] 按 batch 批处理的双缓冲流水路径；
// 9 Transpose021ResidentOutput：rank3 [0,2,1] 输出列常驻 UB 的窄形状专用路径。
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
// 当前实现最多使用 40 个 AIV，实际 blockDim 还受平台可用核数及任务数量限制。
constexpr uint32_t kMaxBlockDim = 40;
// Host 用于 tileLength 收缩的 512B 性能粒度；不是所有 MTE 指令的硬性地址对齐要求。
constexpr uint32_t kAlignBytes = 512;
// DMA(DataCopy 系列)指令的最小对齐/块粒度(32B)，决定各处地址与长度的取整。
constexpr uint32_t kDmaAlignBytes = 32;
// 为系统保留的 UB 字节数，计算可用容量时先扣除。
constexpr uint32_t kReservedUbBytes = 8 * 1024;
// 021 系列路径在 UB 预算中额外保留的字节数。
constexpr uint32_t kTranspose021ReservedUbBytes = 256;
// 021-B32 整行路径 ping-pong 流水下单份输入 slot 的容量(64KB)。
constexpr uint32_t kTranspose021B32PipelineInputSlotBytes = 64 * 1024;
// 021-B32 整行路径宽行场景下单份输入 slot 的容量上限(128KB)。
constexpr uint32_t kTranspose021B32FullRowInputSlotBytes = 128 * 1024;
// last-axis 静态 kernel 单份缓冲的容量，限制该路径 tile 的行列上限。
constexpr uint32_t kLastAxisStaticSlotBytes = 48 * 1024;
// rank5 INT32 专用分组的常量：分组大小与 A/D 维的适用区间门槛。
constexpr uint32_t kLastAxisRank5DGroup = 16;
constexpr uint32_t kLastAxisRank5AGroup = 4;
constexpr uint32_t kLastAxisRank5ADGroup = 4;
constexpr uint32_t kLastAxisRank5MinAForADGroup = 8;
constexpr uint32_t kLastAxisRank5FullADGroup = 5;
constexpr uint32_t kLastAxisRank5FullAMin = 8;
constexpr uint32_t kLastAxisRank5FullAMax = 12;
constexpr uint32_t kLastAxisRank5MinRC = 17;
constexpr uint32_t kLastAxisRank5MaxRC = 25;
constexpr uint32_t kLastAxisRank5MinD = 128;
constexpr uint32_t kFlattenInt8BaseTileCols = 32;
constexpr uint32_t kFlattenInt8MacroRows = 64;
constexpr uint32_t kFlattenInt8MacroMinCols = 128;
constexpr uint32_t kFlattenInt8MacroMaxCols = 256;

constexpr uint32_t kDataCopyPadMaxBlockCount = 4095;
constexpr uint32_t kStaticUnalignedBlockCopySlotBytes = 40 * 1024;
constexpr uint64_t kReverse3DInputMajorThresholdBytes = 192ULL * 1024 * 1024;
constexpr uint32_t kReverse3DInputRowSkewBytes = 32;
constexpr uint32_t kReverse3DGatherGroupBatch = 8;
// 该路径的 tile 成本模型对离散输出 DMA 块赋予更高权重；这是调度启发式，不是硬件周期常数。
constexpr uint32_t kReverse3DOutputDmaWeight = 3;

// 元素宽度用于 UB 容量、DMA 字节数和基础转置块选择，返回 0 表示不支持该类型。
uint32_t GetDataTypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

// 先校验支持的 dtype；最终 key0/key1/key2 由 TilingFunc 的缓存策略与 workspace 条件覆盖。
uint32_t GetTilingKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_INT8:
            return 0;
        default:
            return UINT32_MAX;
    }
}

// 按元素位宽选择基础转置块，匹配 Kernel 的 16 位、32 位和 INT8 地址组织。
void SelectBaseTransposeTile(ge::DataType dt, uint32_t& tileRows, uint32_t& tileCols)
{
    if (dt == ge::DT_INT8) {
        tileRows = 32;
        tileCols = 32;
    } else if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        tileRows = 16;
        tileCols = 8;
    } else {
        tileRows = 16;
        tileCols = 16;
    }
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return align == 0 ? value : (value / align) * align;
}

// 在 INT8 列宽族内搜索能装入静态 UB 的大块，允许尾部补齐以减少小 tile 数量。
bool SelectFlattenInt8PaddedMacroTile(uint32_t outer, uint32_t rows, uint32_t cols,
                                     uint32_t coreNum, uint64_t usableUbBytes,
                                     uint32_t& tileRows, uint32_t& tileCols)
{
    if (cols < kFlattenInt8MacroMinCols || cols > kFlattenInt8MacroMaxCols) {
        return false;
    }
    const uint32_t candidateCols = AlignUp(cols, kDmaAlignBytes);
    const uint64_t tileBytes = static_cast<uint64_t>(kFlattenInt8MacroRows) * candidateCols;
    if (2 * tileBytes > usableUbBytes) {
        return false;
    }
    const uint64_t taskCount = static_cast<uint64_t>(outer) *
        ((rows + kFlattenInt8MacroRows - 1) / kFlattenInt8MacroRows);
    const uint32_t minUsefulCores = std::max(1U, coreNum / 2);
    if (taskCount < minUsefulCores || taskCount > UINT32_MAX) {
        return false;
    }
    tileRows = kFlattenInt8MacroRows;
    tileCols = candidateCols;
    return true;
}

uint64_t GetAlignedSegmentSpan(uint32_t length, uint32_t tileLength, uint32_t align)
{
    const uint32_t fullTiles = length / tileLength;
    const uint32_t tail = length - fullTiles * tileLength;
    uint64_t span = static_cast<uint64_t>(fullTiles) * AlignUp(tileLength, align);
    if (tail != 0) {
        span += AlignUp(tail, align);
    }
    return span;
}

// 在 UB、DMA blockCount 和 Vector repeat 上限内枚举行列分块。
// 优先减少按 128B 边界估计的访问跨度，再比较填充工作量和 tile 数；该估计不是实测 HBM 流量。
bool SelectTranspose021TailMacroTile(uint32_t rows, uint32_t cols, uint64_t usableUbBytes,
                                    uint32_t dataWidth, uint32_t rowAlign, uint32_t colAlign,
                                    uint32_t repeatCols, uint32_t minRowBytes, uint32_t minColBytes,
                                    uint32_t bufferCount,
                                    uint32_t& tileRows, uint32_t& tileCols)
{
    constexpr uint32_t kCacheLineBytes = 128;
    constexpr uint32_t kMaxRowBlocks = 4095;
    constexpr uint32_t kMaxRepeatTimes = UINT8_MAX;
    constexpr uint32_t kMaxPartitions = 64;
    const uint64_t maxTileBytes = usableUbBytes / bufferCount;
    const uint32_t cacheLineElems = kCacheLineBytes / dataWidth;

    bool found = false;
    uint64_t bestTraffic = UINT64_MAX;
    uint64_t bestPaddedElems = UINT64_MAX;
    uint32_t bestTileCount = UINT32_MAX;
    for (uint32_t rowParts = 1; rowParts <= kMaxPartitions; ++rowParts) {
        const uint32_t candidateRows = AlignUp((rows + rowParts - 1) / rowParts, rowAlign);
        if (static_cast<uint64_t>(candidateRows) * dataWidth < minRowBytes ||
            candidateRows > kMaxRowBlocks) {
            continue;
        }
        const uint32_t actualRowTiles = (rows + candidateRows - 1) / candidateRows;
        for (uint32_t colParts = 1; colParts <= kMaxPartitions; ++colParts) {
            const uint32_t candidateCols = AlignUp((cols + colParts - 1) / colParts, colAlign);
            if (static_cast<uint64_t>(candidateCols) * dataWidth < minColBytes ||
                candidateCols > cols ||
                candidateCols / repeatCols > kMaxRepeatTimes) {
                continue;
            }
            const uint64_t tileBytes =
                static_cast<uint64_t>(candidateRows) * candidateCols * dataWidth;
            if (tileBytes > maxTileBytes) {
                continue;
            }

            const uint32_t actualColTiles = (cols + candidateCols - 1) / candidateCols;
            const uint64_t inputTraffic = static_cast<uint64_t>(rows) * dataWidth *
                GetAlignedSegmentSpan(cols, candidateCols, cacheLineElems);
            const uint64_t outputTraffic = static_cast<uint64_t>(cols) * dataWidth *
                GetAlignedSegmentSpan(rows, candidateRows, cacheLineElems);
            const uint64_t traffic = inputTraffic + outputTraffic;
            const uint32_t tileCount = actualRowTiles * actualColTiles;
            const uint64_t paddedElems =
                static_cast<uint64_t>(candidateRows) * candidateCols * tileCount;
            if (!found || traffic < bestTraffic ||
                (traffic == bestTraffic && paddedElems < bestPaddedElems) ||
                (traffic == bestTraffic && paddedElems == bestPaddedElems && tileCount < bestTileCount)) {
                found = true;
                bestTraffic = traffic;
                bestPaddedElems = paddedElems;
                bestTileCount = tileCount;
                tileRows = candidateRows;
                tileCols = candidateCols;
            }
        }
    }
    return found;
}

// 32 位数据优先读取完整行；较宽行改用受输入槽容量约束的列分块，防止物理行补齐后超出 UB。
bool SelectTranspose021B32FullRowTile(uint32_t rows, uint32_t cols,
                                     uint32_t& tileRows, uint32_t& tileCols)
{
    constexpr uint32_t kDataWidth = 4;
    constexpr uint32_t kRowAlign = 16;
    constexpr uint32_t kColAlign = kDmaAlignBytes / kDataWidth;
    constexpr uint32_t kWideRowBytes = 3 * 1024;
    constexpr uint32_t kNarrowInputTileBytes = 76 * 1024;
    if (rows < 32 || cols < 128) {
        return false;
    }

    tileCols = AlignUp(cols, kColAlign);
    const bool useWideTile = tileCols * kDataWidth >= kWideRowBytes;
    const uint32_t inputTileBytes = useWideTile ?
        kTranspose021B32FullRowInputSlotBytes : kNarrowInputTileBytes;
    if (useWideTile && rows >= 64) {
        tileRows = AlignDown(std::min(rows, 64U), kRowAlign);
        const uint32_t maxCols = static_cast<uint32_t>(
            kTranspose021B32PipelineInputSlotBytes /
            (static_cast<uint64_t>(tileRows) * kDataWidth));
        tileCols = AlignDown(std::min(tileCols, maxCols), kColAlign);
        return tileCols >= kColAlign;
    }
    if (static_cast<uint64_t>(kRowAlign) * tileCols * kDataWidth >
        inputTileBytes) {
        return false;
    }
    const uint32_t maxRows = static_cast<uint32_t>(
        inputTileBytes /
        (static_cast<uint64_t>(tileCols) * kDataWidth));
    tileRows = AlignDown(std::min(rows, maxRows), kRowAlign);
    return tileRows >= kRowAlign;
}

// 为二维任务选择核网格的两维切分，结果交给 Kernel 的 GetSplitRange 生成不重叠区间。
void SelectDirectCoreSplit(uint32_t dimA, uint32_t dimB, uint32_t maxCores,
                           uint32_t& splitA, uint32_t& splitB)
{
    if (dimA >= maxCores) {
        splitA = maxCores;
        splitB = 1;
        return;
    }
    splitA = 1;
    splitB = 1;
    uint32_t bestCoreCount = 1;
    uint64_t bestMaxWork = static_cast<uint64_t>(dimA) * dimB;
    const uint32_t maxSplitA = std::min(dimA, maxCores);
    for (uint32_t candidateA = 1; candidateA <= maxSplitA; ++candidateA) {
        const uint32_t maxSplitB = std::min(dimB, maxCores / candidateA);
        for (uint32_t candidateB = 1; candidateB <= maxSplitB; ++candidateB) {
            const uint32_t coreCount = candidateA * candidateB;
            const uint64_t maxWork =
                static_cast<uint64_t>((dimA + candidateA - 1) / candidateA) *
                ((dimB + candidateB - 1) / candidateB);
            if (coreCount > bestCoreCount ||
                (coreCount == bestCoreCount && maxWork < bestMaxWork)) {
                bestCoreCount = coreCount;
                bestMaxWork = maxWork;
                splitA = candidateA;
                splitB = candidateB;
            }
        }
    }
}

// 将负轴号归一化，并检查 dims 长度、范围及重复轴，确保 perm 是合法排列。
bool BuildPerm(const gert::Shape& inputShape, const gert::RuntimeAttrs* attrs,
               uint32_t (&perm)[optiling::kTransposeMaxRank])
{
    const uint32_t rank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (rank == 0 || rank > optiling::kTransposeMaxRank || attrs == nullptr) {
        return false;
    }
    const auto* dimsAttr = attrs->GetListInt(0);
    if (dimsAttr == nullptr || dimsAttr->GetSize() != rank) {
        return false;
    }

    bool seen[optiling::kTransposeMaxRank] = {};
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t axis = dimsAttr->GetData()[i];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || seen[axis]) {
            return false;
        }
        perm[i] = static_cast<uint32_t>(axis);
        seen[axis] = true;
    }
    return true;
}

// 为收缩后的 shape 计算连续存储步长；所有 stride 都以元素计，字节换算在搬运处进行。
void BuildShapeInfoFromDims(uint32_t rank, const uint32_t (&perm)[optiling::kTransposeMaxRank],
                            const uint32_t (&inputDims)[optiling::kTransposeMaxRank],
                            uint32_t (&inputStrides)[optiling::kTransposeMaxRank],
                            uint32_t (&outputDims)[optiling::kTransposeMaxRank],
                            uint32_t (&outputStrides)[optiling::kTransposeMaxRank],
                            uint32_t& totalLength)
{
    for (uint32_t i = 0; i < optiling::kTransposeMaxRank; ++i) {
        inputStrides[i] = 1;
        outputDims[i] = 1;
        outputStrides[i] = 1;
    }

    uint64_t total = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        outputDims[i] = inputDims[perm[i]];
        total *= static_cast<uint64_t>(inputDims[i]);
    }
    totalLength = static_cast<uint32_t>(total);

    if (rank > 0) {
        inputStrides[rank - 1] = 1;
        outputStrides[rank - 1] = 1;
        for (int32_t i = static_cast<int32_t>(rank) - 2; i >= 0; --i) {
            inputStrides[i] = inputStrides[i + 1] * inputDims[i + 1];
            outputStrides[i] = outputStrides[i + 1] * outputDims[i + 1];
        }
    }
}

// size=1 的轴不影响线性存储顺序，删除后重建 perm；全为 1 时保留一维标量等价表示。
bool CollapseSingletonAxes(const gert::Shape& inputShape, uint32_t rawRank,
                           const uint32_t (&rawPerm)[optiling::kTransposeMaxRank],
                           uint32_t& rank,
                           uint32_t (&perm)[optiling::kTransposeMaxRank],
                           uint32_t (&inputDims)[optiling::kTransposeMaxRank])
{
    uint32_t oldToNew[optiling::kTransposeMaxRank];
    uint32_t rawDims[optiling::kTransposeMaxRank];
    for (uint32_t i = 0; i < optiling::kTransposeMaxRank; ++i) {
        oldToNew[i] = UINT32_MAX;
        rawDims[i] = 1;
        perm[i] = 0;
        inputDims[i] = 1;
    }

    rank = 0;
    for (uint32_t axis = 0; axis < rawRank; ++axis) {
        const int64_t dim = inputShape.GetDim(axis);
        if (dim <= 0 || dim > UINT32_MAX) {
            return false;
        }
        rawDims[axis] = static_cast<uint32_t>(dim);
        if (rawDims[axis] != 1) {
            oldToNew[axis] = rank;
            inputDims[rank] = rawDims[axis];
            ++rank;
        }
    }

    if (rank == 0) {
        rank = 1;
        inputDims[0] = 1;
        perm[0] = 0;
        return true;
    }

    uint32_t outAxis = 0;
    for (uint32_t rawOutAxis = 0; rawOutAxis < rawRank; ++rawOutAxis) {
        const uint32_t inputAxis = rawPerm[rawOutAxis];
        if (rawDims[inputAxis] == 1) {
            continue;
        }
        if (oldToNew[inputAxis] == UINT32_MAX || outAxis >= rank) {
            return false;
        }
        perm[outAxis] = oldToNew[inputAxis];
        ++outAxis;
    }
    return outAxis == rank;
}

// 找出输出末尾仍保持输入连续顺序的轴段，将其乘积作为整块拷贝的 innerBlock。
uint32_t GetContiguousOutputSuffix(uint32_t rank, const uint32_t (&perm)[optiling::kTransposeMaxRank],
                                   const uint32_t (&outputDims)[optiling::kTransposeMaxRank])
{
    uint32_t innerBlock = 1;
    int32_t expectedAxis = static_cast<int32_t>(rank) - 1;
    for (int32_t outAxis = static_cast<int32_t>(rank) - 1; outAxis >= 0; --outAxis) {
        if (static_cast<int32_t>(perm[outAxis]) != expectedAxis) {
            break;
        }
        innerBlock *= outputDims[outAxis];
        --expectedAxis;
    }
    return innerBlock;
}

// 识别仅交换相邻轴且保留连续后缀的排列，为块搬运提供结构性判定。
bool DetectAdjacentSwapWithSuffix(uint32_t rank, const uint32_t (&perm)[optiling::kTransposeMaxRank],
                                  uint32_t& swapAxis)
{
    swapAxis = 0;
    bool found = false;
    for (uint32_t outAxis = 0; outAxis < rank; ++outAxis) {
        const bool swappedWithNext = outAxis + 1 < rank && perm[outAxis] == outAxis + 1 &&
                                     perm[outAxis + 1] == outAxis;
        const bool unchanged = perm[outAxis] == outAxis;
        if (swappedWithNext) {
            if (found) {
                return false;
            }
            found = true;
            swapAxis = outAxis;
            ++outAxis;
        } else if (!unchanged) {
            return false;
        }
    }
    return found;
}

// 识别两个相邻连续轴段整体互换，允许先折叠轴段再执行二维或块拷贝路径。
bool DetectAdjacentBlockSwap(uint32_t rank, const uint32_t (&perm)[optiling::kTransposeMaxRank],
                             uint32_t& prefixLen, uint32_t& splitAxis, uint32_t& suffixStart)
{
    prefixLen = 0;
    splitAxis = 0;
    suffixStart = 0;
    for (uint32_t prefix = 0; prefix + 1 < rank; ++prefix) {
        for (uint32_t split = prefix + 1; split < rank; ++split) {
            for (uint32_t suffix = split + 1; suffix <= rank; ++suffix) {
                uint32_t outAxis = 0;
                bool matched = true;
                for (uint32_t axis = 0; axis < prefix && matched; ++axis, ++outAxis) {
                    matched = perm[outAxis] == axis;
                }
                for (uint32_t axis = split; axis < suffix && matched; ++axis, ++outAxis) {
                    matched = perm[outAxis] == axis;
                }
                for (uint32_t axis = prefix; axis < split && matched; ++axis, ++outAxis) {
                    matched = perm[outAxis] == axis;
                }
                for (uint32_t axis = suffix; axis < rank && matched; ++axis, ++outAxis) {
                    matched = perm[outAxis] == axis;
                }
                if (matched && outAxis == rank) {
                    prefixLen = prefix;
                    splitAxis = split;
                    suffixStart = suffix;
                    return true;
                }
            }
        }
    }
    return false;
}

void UseMacroTile(uint32_t macroMode, uint32_t tileRows, uint32_t tileCols, uint32_t rows, uint32_t cols,
                  uint32_t& mode, uint32_t& taskTotal, uint32_t& swapTileA, uint32_t& swapTileB)
{
    mode = macroMode;
    swapTileA = tileRows;
    swapTileB = tileCols;
    taskTotal = (rows / tileRows) * (cols / tileCols);
}

bool IsAtLeastRatio(uint32_t lhs, uint32_t rhs, uint32_t ratio)
{
    return rhs != 0 && static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs) * ratio;
}

bool TryUseFp16MacroTile(uint32_t tileRows, uint32_t tileCols, uint32_t rows, uint32_t cols, uint32_t macroMode,
                         uint32_t& mode, uint32_t& taskTotal, uint32_t& swapTileA, uint32_t& swapTileB)
{
    if (tileRows % 16 != 0 || tileCols % 16 != 0 || rows % tileRows != 0 || cols % tileCols != 0) {
        return false;
    }
    UseMacroTile(macroMode, tileRows, tileCols, rows, cols, mode, taskTotal, swapTileA, swapTileB);
    return true;
}

bool TryUseB32EquivalentTile(uint32_t fp16TileRows, uint32_t fp16TileCols, uint32_t rows, uint32_t cols,
                             uint32_t& mode, uint32_t& taskTotal, uint32_t& swapTileA, uint32_t& swapTileB,
                             uint32_t macroMode = kModeTranspose2DMacro)
{
    if (fp16TileCols < 16 || fp16TileRows % 16 != 0 || fp16TileCols % 16 != 0) {
        return false;
    }
    const uint32_t tileRows = fp16TileRows;
    const uint32_t tileCols = fp16TileCols / 2;
    if (tileCols % 8 != 0 || rows % tileRows != 0 || cols % tileCols != 0) {
        return false;
    }
    UseMacroTile(macroMode, tileRows, tileCols, rows, cols, mode, taskTotal, swapTileA, swapTileB);
    return true;
}

bool TryUseInt8EquivalentTile(uint32_t fp16TileRows, uint32_t fp16TileCols, uint32_t rows, uint32_t cols,
                              uint32_t& mode, uint32_t& taskTotal, uint32_t& swapTileA, uint32_t& swapTileB,
                              uint32_t macroMode = kModeTranspose2DMacro)
{
    if (fp16TileRows % 16 != 0 || fp16TileCols % 16 != 0 || fp16TileCols > UINT32_MAX / 2) {
        return false;
    }
    const uint32_t tileRows = fp16TileRows;
    const uint32_t tileCols = fp16TileCols * 2;
    if (tileCols % 32 != 0 || rows % tileRows != 0 || cols % tileCols != 0) {
        return false;
    }
    UseMacroTile(macroMode, tileRows, tileCols, rows, cols, mode, taskTotal, swapTileA, swapTileB);
    return true;
}

bool TryUseDtypeMacroTile(ge::DataType dt, uint32_t tileRows, uint32_t tileCols, uint32_t rows, uint32_t cols,
                          uint32_t macroMode, uint32_t& mode, uint32_t& taskTotal,
                          uint32_t& swapTileA, uint32_t& swapTileB)
{
    if (rows % tileRows != 0 || cols % tileCols != 0) {
        return false;
    }
    if (dt == ge::DT_FLOAT16) {
        if (tileRows % 16 != 0 || tileCols % 16 != 0) {
            return false;
        }
    } else if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        if (tileRows % 16 != 0 || tileCols % 8 != 0) {
            return false;
        }
    } else if (dt == ge::DT_INT8) {
        if (tileRows % 32 != 0 || tileCols % 32 != 0) {
            return false;
        }
    } else {
        return false;
    }
    UseMacroTile(macroMode, tileRows, tileCols, rows, cols, mode, taskTotal, swapTileA, swapTileB);
    return true;
}

// 按 dtype 和整除条件尝试既有宏块布局，不能装入对应物理布局时继续后续分支。
bool TryUsePreferredMacroTile(ge::DataType dt, uint32_t rows, uint32_t cols, uint32_t macroMode,
                              uint32_t& mode, uint32_t& taskTotal, uint32_t& swapTileA, uint32_t& swapTileB)
{
    if (dt == ge::DT_FLOAT16) {
        if (IsAtLeastRatio(rows, cols, 64) &&
            TryUseFp16MacroTile(256, 128, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
        if (IsAtLeastRatio(cols, rows, 64) &&
            TryUseFp16MacroTile(128, 256, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
        if (std::min(rows, cols) >= 8192 &&
            TryUseFp16MacroTile(256, 192, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
        if (std::min(rows, cols) >= 8192 &&
            TryUseFp16MacroTile(256, 128, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
        if (std::min(rows, cols) >= 6144 &&
            TryUseFp16MacroTile(192, 192, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
        if (TryUseFp16MacroTile(128, 128, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB) ||
            TryUseFp16MacroTile(128, 256, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB) ||
            TryUseFp16MacroTile(256, 128, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB) ||
            TryUseFp16MacroTile(64, 64, rows, cols, macroMode, mode, taskTotal, swapTileA, swapTileB)) {
            return true;
        }
    } else if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        if ((IsAtLeastRatio(rows, cols, 64) &&
             TryUseB32EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (IsAtLeastRatio(cols, rows, 64) &&
             TryUseB32EquivalentTile(128, 256, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 8192 &&
             TryUseB32EquivalentTile(256, 192, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 8192 &&
             TryUseB32EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 6144 &&
             TryUseB32EquivalentTile(192, 192, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            TryUseB32EquivalentTile(128, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseB32EquivalentTile(128, 256, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseB32EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseB32EquivalentTile(64, 64, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) {
            return true;
        }
    } else if (dt == ge::DT_INT8) {
        if ((IsAtLeastRatio(rows, cols, 64) &&
             TryUseInt8EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (IsAtLeastRatio(cols, rows, 64) &&
             TryUseInt8EquivalentTile(128, 256, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 8192 &&
             TryUseInt8EquivalentTile(256, 192, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 8192 &&
             TryUseInt8EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            (std::min(rows, cols) >= 6144 &&
             TryUseInt8EquivalentTile(192, 192, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) ||
            TryUseInt8EquivalentTile(128, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseInt8EquivalentTile(128, 256, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseInt8EquivalentTile(256, 128, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode) ||
            TryUseInt8EquivalentTile(64, 64, rows, cols, mode, taskTotal, swapTileA, swapTileB, macroMode)) {
            return true;
        }
    }
    return false;
}

// 先尝试整除宏块，再搜索允许尾块的静态布局；候选须同时满足 UB 和重复次数限制。
bool TryUseLastAxisMacroTile(ge::DataType dt, uint32_t rows, uint32_t cols, uint32_t maxTileElems,
                             uint32_t& mode, uint32_t& taskTotal,
                             uint32_t& swapTileA, uint32_t& swapTileB)
{
    const uint32_t dataWidth = GetDataTypeSize(dt);
    const uint32_t rowAlign = dt == ge::DT_INT8 ? 32U : 16U;
    const uint32_t colAlign = dt == ge::DT_INT8 ? 32U : (dt == ge::DT_FLOAT16 ? 16U : 8U);
    uint64_t bestTileElems = 0;
    uint32_t bestRows = 0;
    uint32_t bestCols = 0;
    if (dataWidth != 0 && rows <= 64) {
        for (uint32_t candidateRows = AlignDown(std::min(rows, 64U), rowAlign);
             candidateRows >= rowAlign; candidateRows -= rowAlign) {
            if (rows % candidateRows != 0) {
                continue;
            }
            const uint32_t maxColsByUb = kLastAxisStaticSlotBytes / (candidateRows * dataWidth);
            const uint32_t maxColsByRepeat = colAlign * UINT8_MAX;
            const uint32_t maxCols = AlignDown(std::min(cols, std::min(maxColsByUb, maxColsByRepeat)), colAlign);
            for (uint32_t candidateCols = maxCols; candidateCols >= colAlign; candidateCols -= colAlign) {
                if (cols % candidateCols != 0) {
                    continue;
                }
                const uint64_t tileElems = static_cast<uint64_t>(candidateRows) * candidateCols;
                if (tileElems > bestTileElems ||
                    (tileElems == bestTileElems && candidateRows > bestRows)) {
                    bestTileElems = tileElems;
                    bestRows = candidateRows;
                    bestCols = candidateCols;
                }
                break;
            }
        }
    }
    if (bestTileElems != 0) {
        UseMacroTile(kModeLastAxisTransposeTiled, bestRows, bestCols, rows, cols,
                     mode, taskTotal, swapTileA, swapTileB);
        return true;
    }

    if (TryUsePreferredMacroTile(dt, rows, cols, kModeLastAxisTransposeTiled,
                                 mode, taskTotal, swapTileA, swapTileB)) {
        return true;
    }

    // 尾 tile 复用同一带填充的静态 Kernel，因此可在容量范围内保留较大的块尺寸。
    uint32_t bestTileCount = UINT32_MAX;
    uint64_t bestTailTileElems = 0;
    uint32_t bestTailRows = 0;
    uint32_t bestTailCols = 0;
    const uint32_t maxRows = std::min(AlignUp(rows, rowAlign), kDataCopyPadMaxBlockCount);
    const uint32_t maxColsByRepeat = colAlign * UINT8_MAX;
    for (uint32_t candidateRows = rowAlign; candidateRows <= maxRows; candidateRows += rowAlign) {
        const uint32_t maxColsByUb = kLastAxisStaticSlotBytes / (candidateRows * dataWidth);
        const uint32_t maxCols = AlignDown(
            std::min(maxColsByUb, std::min(maxColsByRepeat, kDataCopyPadMaxBlockCount)), colAlign);
        if (maxCols < colAlign) {
            break;
        }
        const uint32_t candidateCols = AlignDown(
            std::min(maxCols, AlignUp(cols, colAlign)), colAlign);
        const uint32_t tileCount =
            ((rows + candidateRows - 1) / candidateRows) *
            ((cols + candidateCols - 1) / candidateCols);
        const uint64_t tileElems = static_cast<uint64_t>(candidateRows) * candidateCols;
        // 先最小化 tile 数；同数时，32 位类型优先更多列，以增加输入 DMA 连续块长度。
        // 较窄类型仍优先更多行，以保留输出块的字节长度；这是不同位宽访问代价下的经验选择。
        const bool preferMoreCols = dataWidth == 4;
        const bool better = preferMoreCols ?
            (candidateCols > bestTailCols ||
             (candidateCols == bestTailCols && tileElems > bestTailTileElems)) :
            (candidateRows > bestTailRows ||
             (candidateRows == bestTailRows && tileElems > bestTailTileElems));
        if (tileCount < bestTileCount || (tileCount == bestTileCount && better)) {
            bestTileCount = tileCount;
            bestTailTileElems = tileElems;
            bestTailRows = candidateRows;
            bestTailCols = candidateCols;
        }
    }
    if (bestTailTileElems != 0) {
        mode = kModeLastAxisTransposeTiled;
        taskTotal = bestTileCount;
        swapTileA = bestTailRows;
        swapTileB = bestTailCols;
        return true;
    }

    const uint32_t maxBufferedElems = std::max(1U, maxTileElems / 2);
    if (static_cast<uint64_t>(rows) * cols > maxBufferedElems) {
        return false;
    }
    const uint32_t repeatCols = dt == ge::DT_INT8 ? 32U : (dt == ge::DT_FLOAT16 ? 16U : 8U);
    if (cols / repeatCols > UINT8_MAX) {
        return false;
    }
    return TryUseDtypeMacroTile(dt, rows, cols, rows, cols, kModeLastAxisTransposeTiled,
                                mode, taskTotal, swapTileA, swapTileB);
}

// 估算批处理填充和转置阶段的最大中间 Tensor，取峰值而不是仅按有效数据量预算。
uint64_t GetTranspose021BatchBufferBytes(uint32_t tasks, uint32_t rows, uint32_t cols,
                                         uint32_t dataWidth)
{
    const uint32_t blockElems = kDmaAlignBytes / dataWidth;
    const uint32_t rowsAlign = AlignUp(rows, 16);
    const uint32_t colsAlign = AlignUp(cols, blockElems);
    const uint32_t hOnce = 16 * blockElems;
    const uint64_t flatRows = static_cast<uint64_t>(tasks) * rows;
    const uint64_t flatRowsAlign = ((flatRows + hOnce - 1) / hOnce) * hOnce;
    const uint64_t padWBytes = flatRowsAlign * colsAlign * dataWidth;
    const uint64_t transposeBytes =
        static_cast<uint64_t>(tasks) * rowsAlign * colsAlign * dataWidth;
    return cols == colsAlign ? transposeBytes : std::max(padWBytes, transposeBytes);
}

// 在单槽容量允许范围内选择一批容纳的矩阵数。
uint32_t GetTranspose021TasksOnce(uint32_t maxTasks, uint32_t rows, uint32_t cols,
                                  uint32_t dataWidth, uint64_t slotBytes)
{
    const uint32_t blockElems = kDmaAlignBytes / dataWidth;
    const uint32_t colsAlign = AlignUp(cols, blockElems);
    const uint32_t hOnce = 16 * blockElems;
    uint32_t bestTasks = 0;
    uint64_t bestFlatRows = 0;
    uint64_t bestFlatRowsAlign = 1;
    for (uint32_t tasks = maxTasks; tasks > 0; --tasks) {
        if (static_cast<uint64_t>(tasks) * colsAlign / blockElems > UINT8_MAX) {
            continue;
        }
        if (GetTranspose021BatchBufferBytes(tasks, rows, cols, dataWidth) > slotBytes) {
            continue;
        }
        const uint64_t flatRows = static_cast<uint64_t>(tasks) * rows;
        const uint64_t flatRowsAlign = ((flatRows + hOnce - 1) / hOnce) * hOnce;
        const uint64_t lhs = flatRows * bestFlatRowsAlign;
        const uint64_t rhs = bestFlatRows * flatRowsAlign;
        if (bestTasks == 0 || lhs > rhs || (lhs == rhs && tasks > bestTasks)) {
            bestTasks = tasks;
            bestFlatRows = flatRows;
            bestFlatRowsAlign = flatRowsAlign;
        }
    }
    return bestTasks;
}

// 仅在批次聚合的中间布局满足 UB 预算时启用 021 批处理。
bool TryUseTranspose021Batch(ge::DataType dt, uint32_t batch, uint32_t rows, uint32_t cols,
                             uint64_t ubSize, uint32_t coreNum, uint32_t dataWidth,
                             uint32_t& mode, uint32_t& taskTotal,
                             uint32_t& tasksOnce, uint32_t& doubleBuffer)
{
    if ((dt != ge::DT_FLOAT16 && dt != ge::DT_FLOAT && dt != ge::DT_INT32) ||
        batch == 0 || rows == 0 || cols == 0) {
        return false;
    }

    const uint32_t blockElems = kDmaAlignBytes / dataWidth;
    const uint32_t colsAlign = AlignUp(cols, blockElems);
    if (colsAlign / blockElems > UINT8_MAX || (cols != colsAlign && colsAlign > UINT8_MAX)) {
        return false;
    }

    const uint32_t blockDim = std::min(coreNum, batch);
    const uint32_t maxTasksPerCore = (batch + blockDim - 1) / blockDim;
    const uint64_t ubBudget =
        ubSize > kTranspose021ReservedUbBytes ? ubSize - kTranspose021ReservedUbBytes : ubSize;
    const uint64_t singleSlotBytes = AlignDown(static_cast<uint32_t>(ubBudget / 2), kDmaAlignBytes);
    const uint64_t doubleSlotBytes = AlignDown(static_cast<uint32_t>(ubBudget / 4), kDmaAlignBytes);
    const uint32_t singleTasks =
        GetTranspose021TasksOnce(maxTasksPerCore, rows, cols, dataWidth, singleSlotBytes);
    if (singleTasks == 0) {
        return false;
    }

    const uint32_t doubleTasks =
        GetTranspose021TasksOnce(maxTasksPerCore, rows, cols, dataWidth, doubleSlotBytes);
    if (doubleTasks > 0 && maxTasksPerCore > doubleTasks) {
        tasksOnce = doubleTasks;
        doubleBuffer = 2;
    } else {
        tasksOnce = singleTasks;
        doubleBuffer = 1;
    }

    mode = kModeTranspose021Batch;
    taskTotal = batch;
    return true;
}

// 为窄形状选择输出常驻 UB 的列段；读取分批进行，输出积累后统一写回。
bool TryUseTranspose021ResidentOutput(ge::DataType dt, uint32_t batch,
                                      uint32_t rows, uint32_t cols, uint64_t ubSize,
                                      uint32_t& mode, uint32_t& taskTotal,
                                      uint32_t& tileRows, uint32_t& tileCols,
                                      uint32_t& residentCols)
{
    constexpr uint32_t kDataWidth = sizeof(uint32_t);
    constexpr uint32_t kInputTileRows = 16;
    constexpr uint32_t kColAlign = kDmaAlignBytes / kDataWidth;
    constexpr uint32_t kMinResidentCols = 64;
    const uint32_t paddedCols = AlignUp(cols, kColAlign);
    if ((dt != ge::DT_FLOAT && dt != ge::DT_INT32) || batch == 0 ||
        rows < 48 || rows > 144 || rows % kColAlign != 0 ||
        paddedCols < 512 || cols >= 640) {
        return false;
    }

    const uint64_t inputBytes =
        static_cast<uint64_t>(kInputTileRows) * paddedCols * kDataWidth;
    if (inputBytes >= ubSize) {
        return false;
    }
    const uint32_t maxResidentCols = AlignDown(static_cast<uint32_t>(
        (ubSize - inputBytes) / (static_cast<uint64_t>(rows) * kDataWidth)), kColAlign);
    residentCols = AlignDown(std::min(cols, maxResidentCols), kColAlign);
    if (residentCols < kMinResidentCols) {
        return false;
    }

    mode = kModeTranspose021ResidentOutput;
    taskTotal = batch;
    tileRows = kInputTileRows;
    tileCols = paddedCols;
    return true;
}

uint32_t AlignedInnerElems(uint32_t innerElems, uint32_t dataWidth);

uint32_t AlignedInnerElems(uint32_t innerElems, uint32_t dataWidth)
{
    const uint32_t alignedBytes = AlignUp(innerElems * dataWidth, kDmaAlignBytes);
    return std::max(1U, alignedBytes / dataWidth);
}

// 为 [A,middle,B,inner]→[B,middle,A,inner] 选择 Gather 分块。
// A 方向要求整除，B 方向允许尾块；UB 预算包含双输入、双输出和索引表，须与 Kernel Init 一致。
bool TryUseReverse3DSuffixGatherTile(uint32_t swapDimA, uint32_t swapDimB, uint32_t swapInner,
                                     uint64_t usableUbBytes, uint64_t totalBytes, uint32_t dataWidth,
                                     uint32_t& swapTileA, uint32_t& swapTileB)
{
    const uint32_t innerBytes = swapInner * dataWidth;
    if (innerBytes % kDmaAlignBytes == 0) {
        return false;
    }

    if (dataWidth == 1) {
        for (uint32_t candA = std::min(swapDimA, 64U); candA > 0; --candA) {
            if (swapDimA % candA != 0) {
                continue;
            }
            const uint64_t inputBytes =
                static_cast<uint64_t>(candA) * AlignUp(innerBytes, kDmaAlignBytes);
            if (2 * inputBytes <= usableUbBytes) {
                swapTileA = candA;
                swapTileB = 1;
                return true;
            }
        }
        return false;
    }

    // 以每次 Vector 的元素数和 8 位 repeat 上限限制 Gather 总元素数；补齐元素也计入预算。
    const uint32_t gatherCountLimit = dataWidth == 2 ? 128U * UINT8_MAX : 64U * UINT8_MAX;
    uint32_t bestA = 0;
    uint32_t bestB = 0;
    uint64_t bestScheduleCost = UINT64_MAX;
    uint64_t bestDmaBlocks = UINT64_MAX;
    uint64_t bestTileCount = UINT64_MAX;
    uint64_t bestProduct = 0;
    const bool useBatchedGatherIndex = totalBytes > kReverse3DInputMajorThresholdBytes;
    const uint32_t gatherGroupBatch = kReverse3DGatherGroupBatch;
    constexpr uint32_t outputBufferCount = 2;
    const uint32_t maxA = std::min(swapDimA, static_cast<uint32_t>(UINT16_MAX));
    for (uint32_t candA = 1; candA <= maxA; ++candA) {
        if (swapDimA % candA != 0) {
            continue;
        }
        const uint32_t maxB = std::min(
            std::min(swapDimB, static_cast<uint32_t>(UINT16_MAX)),
            gatherCountLimit / std::max(1U, candA * swapInner));
        for (uint32_t candB = 1; candB <= maxB; ++candB) {
            const uint32_t packedInputRowBytes = AlignUp(candB * innerBytes, kDmaAlignBytes);
            const bool useInputRowSkew = useBatchedGatherIndex &&
                ((packedInputRowBytes / kDmaAlignBytes) & 1U) != 0;
            const uint64_t inputBytes = static_cast<uint64_t>(candA) *
                (packedInputRowBytes + (useInputRowSkew ? kReverse3DInputRowSkewBytes : 0));
            const uint64_t outputRowBytes = AlignUp(candA * innerBytes, kDmaAlignBytes);
            const uint64_t groupIndexElems = static_cast<uint64_t>(candA) * swapInner;
            const uint64_t alignedGroupIndexElems =
                AlignUp(groupIndexElems * sizeof(uint32_t), kDmaAlignBytes) / sizeof(uint32_t);
            const uint64_t paddedOutputElems =
                AlignUp(candA * swapInner * dataWidth, kDmaAlignBytes) / dataWidth;
            if (static_cast<uint64_t>(candB) * paddedOutputElems > gatherCountLimit) {
                break;
            }
            const bool outputRowsAligned = (candA * innerBytes) % kDmaAlignBytes == 0;
            const uint64_t indexGroups = outputRowsAligned ?
                (useBatchedGatherIndex ? std::min(candB, gatherGroupBatch) : candB) : 1;
            const uint64_t outputBytes = static_cast<uint64_t>(candB) * outputRowBytes;
            const uint64_t indexBytes =
                indexGroups * std::max(alignedGroupIndexElems, paddedOutputElems) * sizeof(uint32_t);
            if (2 * inputBytes + outputBufferCount * outputBytes + indexBytes > usableUbBytes) {
                break;
            }
            const uint64_t tileBlocksA = swapDimA / candA;
            const uint64_t tileBlocksB = (swapDimB + candB - 1) / candB;
            const uint64_t tileCount = tileBlocksA * tileBlocksB;
            const uint64_t inputDmaBlocks = static_cast<uint64_t>(swapDimA) * tileBlocksB;
            const uint64_t outputDmaBlocks = tileBlocksA * swapDimB;
            const uint64_t dmaBlocks = inputDmaBlocks + outputDmaBlocks;
            // 调度代价同时考虑读写 DMA 块、Gather 次数和输入补齐，避免只追求更大的 tile 面积。
            const uint64_t weightedDmaBlocks =
                inputDmaBlocks + kReverse3DOutputDmaWeight * outputDmaBlocks;
            const uint64_t gatherCalls = outputRowsAligned ?
                tileCount * ((candB + indexGroups - 1) / indexGroups) : tileBlocksA * swapDimB;
            const uint64_t inputPaddingPenalty =
                totalBytes > kReverse3DInputMajorThresholdBytes &&
                (candB * innerBytes) % kDmaAlignBytes != 0
                    ? static_cast<uint64_t>(swapDimA) * tileBlocksB
                    : 0;
            const uint64_t scheduleCost = weightedDmaBlocks + gatherCalls + inputPaddingPenalty;
            const uint64_t product = static_cast<uint64_t>(candA) * candB;
            bool better = scheduleCost < bestScheduleCost;
            if (scheduleCost == bestScheduleCost) {
                better = dmaBlocks < bestDmaBlocks ||
                    (dmaBlocks == bestDmaBlocks &&
                     (tileCount < bestTileCount ||
                      (tileCount == bestTileCount &&
                       (product > bestProduct ||
                        (product == bestProduct && (bestA == 0 || candA < bestA))))));
            }
            if (better) {
                bestA = candA;
                bestB = candB;
                bestScheduleCost = scheduleCost;
                bestDmaBlocks = dmaBlocks;
                bestTileCount = tileCount;
                bestProduct = product;
            }
        }
    }
    if (bestA == 0 || bestB == 0) {
        return false;
    }
    swapTileA = bestA;
    swapTileB = bestB;
    return true;
}

// 在连续 inner 块的前提下选择 A/B 分块，控制单次搬运和局部 Tensor 容量。
void SelectBlockSwapTile(uint32_t swapDimA, uint32_t swapDimB, uint32_t swapInner, uint32_t maxTileElems,
                         uint32_t dataWidth, uint32_t& swapTileA, uint32_t& swapTileB)
{
    const uint32_t maxBufferedElems = std::max(1U, maxTileElems / 2);
    const uint32_t innerAlignedElems = AlignedInnerElems(swapInner, dataWidth);
    const uint32_t maxPairs = std::max(1U, maxBufferedElems / innerAlignedElems);
    const uint32_t innerBlocks = std::max(1U, (innerAlignedElems * dataWidth) / kDmaAlignBytes);
    const uint32_t maxGap = UINT16_MAX / innerBlocks;
    uint32_t bestA = 1;
    uint32_t bestB = 1;
    uint64_t bestTileCount = UINT64_MAX;
    uint64_t bestProduct = 0;
    const uint32_t maxA = std::min(swapDimA, maxPairs);
    for (uint32_t candA = 1; candA <= maxA; ++candA) {
        uint32_t candB = std::min(swapDimB, maxPairs / candA);
        if (candB == 0) {
            continue;
        }
        if (candA > maxGap + 1) {
            candB = std::min(candB, std::min(candA - 1, maxGap + 1));
            if (candB == 0) {
                continue;
            }
        }
        const uint64_t tileCount =
            static_cast<uint64_t>((swapDimA + candA - 1) / candA) * ((swapDimB + candB - 1) / candB);
        const uint64_t product = static_cast<uint64_t>(candA) * candB;
        if (tileCount < bestTileCount || (tileCount == bestTileCount && product > bestProduct)) {
            bestA = candA;
            bestB = candB;
            bestTileCount = tileCount;
            bestProduct = product;
        }
    }
    swapTileA = bestA;
    swapTileB = bestB;
}

// 将前缀、两个交换轴段和连续后缀折叠成 swapOuter/A/B/inner，供设备侧统一寻址。
void BuildAdjacentBlockSwapParams(uint32_t rank, uint32_t prefixLen, uint32_t splitAxis, uint32_t suffixStart,
                                  const uint32_t (&inputDims)[optiling::kTransposeMaxRank], uint32_t maxTileElems,
                                  uint32_t dataWidth, uint32_t& swapOuter, uint32_t& swapDimA,
                                  uint32_t& swapDimB, uint32_t& swapInner, uint32_t& swapTileA,
                                  uint32_t& swapTileB)
{
    swapOuter = 1;
    for (uint32_t i = 0; i < prefixLen; ++i) {
        swapOuter *= inputDims[i];
    }
    swapDimA = 1;
    for (uint32_t i = prefixLen; i < splitAxis; ++i) {
        swapDimA *= inputDims[i];
    }
    swapDimB = 1;
    for (uint32_t i = splitAxis; i < suffixStart; ++i) {
        swapDimB *= inputDims[i];
    }
    swapInner = 1;
    for (uint32_t i = suffixStart; i < rank; ++i) {
        swapInner *= inputDims[i];
    }
    SelectBlockSwapTile(swapDimA, swapDimB, swapInner, maxTileElems, dataWidth, swapTileA, swapTileB);
}

}  // namespace

namespace optiling {

// 主入口依次完成合法性检查、轴收缩、结构性路径选择、核数与容量预算以及 tiling 序列化。
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr || context->GetInputTensor(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    const uint32_t rawRank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (rawRank == 0 || rawRank > kTransposeMaxRank) {
        return ge::GRAPH_FAILED;
    }

    auto dt = context->GetInputTensor(0)->GetDataType();
    const uint32_t dataWidth = GetDataTypeSize(dt);
    uint32_t tilingKey = GetTilingKey(dt);
    if (dataWidth == 0 || tilingKey == UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    uint32_t rawPerm[kTransposeMaxRank] = {};
    if (!BuildPerm(inputShape, context->GetAttrs(), rawPerm)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputDims[kTransposeMaxRank];
    uint32_t inputStrides[kTransposeMaxRank];
    uint32_t outputDims[kTransposeMaxRank];
    uint32_t outputStrides[kTransposeMaxRank];
    uint32_t totalLength = 1;
    uint32_t perm[kTransposeMaxRank] = {};
    uint32_t rank = rawRank;
    if (!CollapseSingletonAxes(inputShape, rawRank, rawPerm, rank, perm, inputDims)) {
        return ge::GRAPH_FAILED;
    }
    BuildShapeInfoFromDims(rank, perm, inputDims, inputStrides, outputDims, outputStrides, totalLength);
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t innerBlock = GetContiguousOutputSuffix(rank, perm, outputDims);
    uint32_t mode = kModeGeneric;
    uint32_t taskTotal = totalLength;
    uint32_t swapAxis = 0;
    uint32_t swapOuter = 1;
    uint32_t swapDimA = 1;
    uint32_t swapDimB = 1;
    uint32_t swapInner = 1;
    uint32_t swapTileA = 1;
    uint32_t swapTileB = 1;
    uint32_t directDimA = 0;
    uint32_t directDimB = 0;
    uint32_t coreSplitA = 0;
    uint32_t coreSplitB = 0;
    uint32_t blockSwapPrefix = 0;
    uint32_t blockSwapSplit = 0;
    uint32_t blockSwapSuffix = 0;
    const uint32_t outputBufferLength = AlignUp(totalLength, std::max(1U, kDmaAlignBytes / dataWidth));
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t usableUbBytes = ubSize > kReservedUbBytes ? ubSize - kReservedUbBytes : ubSize;
    uint32_t elemPerAlign = std::max(1U, kAlignBytes / dataWidth);
    uint32_t maxTile = static_cast<uint32_t>(std::max<uint64_t>(elemPerAlign, usableUbBytes / dataWidth));
    maxTile = AlignDown(maxTile, elemPerAlign);
    if (maxTile == 0) {
        maxTile = elemPerAlign;
    }
    const uint32_t coreNum =
        std::min(std::max(1U, ascendcPlatform.GetCoreNumAiv()), kMaxBlockDim);
    if (rank == 2 && perm[0] == 1 && perm[1] == 0) {
        const uint32_t rows = inputDims[0];
        const uint32_t cols = inputDims[1];
        TryUsePreferredMacroTile(dt, rows, cols, kModeTranspose2DMacro,
                                 mode, taskTotal, swapTileA, swapTileB);
    } else if (rank == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        const uint32_t batch = inputDims[0];
        const uint32_t rows = inputDims[1];
        const uint32_t cols = inputDims[2];
        uint32_t matrixTasks = 0;
        if (TryUseTranspose021Batch(dt, batch, rows, cols, ubSize, coreNum, dataWidth,
                                    mode, taskTotal, swapTileA, swapTileB)) {
        } else if (TryUseTranspose021ResidentOutput(dt, batch, rows, cols, ubSize,
                                                    mode, taskTotal, swapTileA, swapTileB,
                                                    swapOuter)) {
        } else if (TryUsePreferredMacroTile(dt, rows, cols, kModeTranspose021Macro,
                                     mode, matrixTasks, swapTileA, swapTileB)) {
            taskTotal = batch * matrixTasks;
        } else {
            mode = kModeTranspose021Fp16;
            if (dt == ge::DT_FLOAT16 && rows >= 64 && cols >= 256 &&
                SelectTranspose021TailMacroTile(rows, cols, usableUbBytes, dataWidth,
                                                16, 16, 16, 128, 512,
                                                2,
                                                swapTileA, swapTileB)) {
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) &&
                       SelectTranspose021B32FullRowTile(rows, cols, swapTileA, swapTileB)) {
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) &&
                       rows >= 32 && cols >= 128 &&
                       SelectTranspose021TailMacroTile(rows, cols, usableUbBytes, dataWidth,
                                                       16, 8, 8, 512, 512,
                                                       2,
                                                       swapTileA, swapTileB)) {
            } else if (dt == ge::DT_FLOAT16 && rows >= 384 && cols >= 384) {
                swapTileA = 128;
                swapTileB = 128;
            } else if (dt == ge::DT_FLOAT16 && rows >= 128 && cols >= 128) {
                swapTileA = 64;
                swapTileB = 64;
            } else if (dt == ge::DT_FLOAT16 && rows >= 384 && cols >= 32) {
                swapTileA = AlignDown(std::min(rows, 256U), 16U);
                swapTileB = 64;
            } else if (dt == ge::DT_FLOAT16 && rows >= 32 && cols >= 384) {
                swapTileA = 64;
                swapTileB = AlignDown(std::min(cols, 256U), 16U);
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) && rows >= 384 && cols >= 192) {
                swapTileA = 128;
                swapTileB = 64;
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) && rows >= 128 && cols >= 64) {
                swapTileA = 64;
                swapTileB = 32;
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) && rows >= 384 && cols >= 16) {
                swapTileA = AlignDown(std::min(rows, 256U), 16U);
                swapTileB = 32;
            } else if ((dt == ge::DT_FLOAT || dt == ge::DT_INT32) && rows >= 16 && cols >= 384) {
                swapTileA = 32;
                swapTileB = AlignDown(std::min(cols, 256U), 8U);
            } else if (dt == ge::DT_INT8 && rows >= 128 && cols >= 128 &&
                       SelectTranspose021TailMacroTile(rows, cols, usableUbBytes, dataWidth,
                                                       32, 32, 32, 128, 128,
                                                       3,
                                                       swapTileA, swapTileB)) {
            } else if (dt == ge::DT_INT8 && rows >= 384 && cols >= 32 && cols < 128) {
                if (cols < 64 && rows > 512) {
                    swapTileA = 512;
                    uint32_t bestTasks = (rows + swapTileA - 1) / swapTileA;
                    uint32_t bestPaddedRows = bestTasks * swapTileA;
                    const uint32_t candidates[2] = {768U, 1024U};
                    for (uint32_t candidate : candidates) {
                        const uint32_t candidateTasks = (rows + candidate - 1) / candidate;
                        const uint32_t candidatePaddedRows = candidateTasks * candidate;
                        if (candidatePaddedRows < bestPaddedRows ||
                            (candidatePaddedRows == bestPaddedRows && candidateTasks < bestTasks)) {
                            swapTileA = candidate;
                            bestTasks = candidateTasks;
                            bestPaddedRows = candidatePaddedRows;
                        }
                    }
                } else {
                    swapTileA = AlignDown(std::min(rows, 512U), 16U);
                }
                swapTileB = 64;
            } else if (dt == ge::DT_INT8 && rows >= 32 && rows < 128 && cols >= 384) {
                swapTileA = 64;
                swapTileB = AlignDown(std::min(cols, 512U), 32U);
            } else {
                SelectBaseTransposeTile(dt, swapTileA, swapTileB);
            }
            taskTotal = inputDims[0] * ((rows + swapTileA - 1) / swapTileA) *
                        ((cols + swapTileB - 1) / swapTileB);
        }
    } else if (rank == 3 &&
               ((perm[0] == 1 && perm[1] == 2 && perm[2] == 0) ||
                (perm[0] == 2 && perm[1] == 0 && perm[2] == 1))) {
        const bool flatten120 = perm[0] == 1;
        swapOuter = 1;
        swapDimA = flatten120 ? inputDims[0] : inputDims[0] * inputDims[1];
        swapDimB = flatten120 ? inputDims[1] * inputDims[2] : inputDims[2];
        mode = kModeTransposeFlattenTiled;
        if (dt == ge::DT_INT8 &&
            SelectFlattenInt8PaddedMacroTile(swapOuter, swapDimA, swapDimB, coreNum,
                                             usableUbBytes, swapTileA, swapTileB)) {
            taskTotal = swapOuter * ((swapDimA + swapTileA - 1) / swapTileA) *
                        ((swapDimB + swapTileB - 1) / swapTileB);
        } else if (dataWidth == 4) {
            taskTotal = ((swapDimA + 15) / 16) * ((swapDimB + 7) / 8);
        } else if (dataWidth == 1) {
            taskTotal = ((swapDimA + 31) / 32) * ((swapDimB + 31) / 32);
        } else {
            taskTotal = ((swapDimA + 15) / 16) * ((swapDimB + 15) / 16);
        }
    } else if (rank == 4 && perm[0] == 2 && perm[1] == 1 && perm[2] == 0 && perm[3] == 3 &&
               TryUseReverse3DSuffixGatherTile(inputDims[0], inputDims[2], inputDims[3], usableUbBytes,
                                               static_cast<uint64_t>(totalLength) * dataWidth,
                                               dataWidth, swapTileA, swapTileB)) {
        mode = kModeReverse3DSuffixGather;
        swapOuter = inputDims[1];
        swapDimA = inputDims[0];
        swapDimB = inputDims[2];
        swapInner = inputDims[3];
        taskTotal = swapOuter * (swapDimA / swapTileA) * ((swapDimB + swapTileB - 1) / swapTileB);
    } else if (rank == 4 && perm[0] == 2 && perm[1] == 1 && perm[2] == 0 && perm[3] == 3 &&
               inputDims[0] <= kDataCopyPadMaxBlockCount &&
               (((static_cast<uint64_t>(inputDims[3]) * dataWidth) % kDmaAlignBytes == 0 &&
                 static_cast<uint64_t>(inputDims[0]) * inputDims[3] <= maxTile) ||
                ((static_cast<uint64_t>(inputDims[3]) * dataWidth) % kDmaAlignBytes != 0 &&
                 static_cast<uint64_t>(inputDims[3]) * dataWidth <= kStaticUnalignedBlockCopySlotBytes))) {
        mode = kModeBlockCopy;
        swapOuter = inputDims[1];
        swapDimA = inputDims[0];
        swapDimB = inputDims[2];
        swapInner = inputDims[3];
        directDimA = swapDimB;
        directDimB = swapOuter;
        taskTotal = directDimA * directDimB;
    } else if (DetectAdjacentBlockSwap(rank, perm, blockSwapPrefix, blockSwapSplit, blockSwapSuffix)) {
        BuildAdjacentBlockSwapParams(rank, blockSwapPrefix, blockSwapSplit, blockSwapSuffix, inputDims, maxTile,
                                     dataWidth, swapOuter, swapDimA, swapDimB, swapInner, swapTileA, swapTileB);
        if (swapInner == 1) {
            mode = kModeTransposeFlattenTiled;
            if (dataWidth == 4) {
                taskTotal = swapOuter * ((swapDimA + 15) / 16) * ((swapDimB + 7) / 8);
            } else if (dataWidth == 1) {
                taskTotal = swapOuter * ((swapDimA + 31) / 32) * ((swapDimB + 31) / 32);
            } else {
                taskTotal = swapOuter * ((swapDimA + 15) / 16) * ((swapDimB + 15) / 16);
            }
        }
    } else if (DetectAdjacentSwapWithSuffix(rank, perm, swapAxis)) {
        mode = kModeGeneric;
    } else if (perm[rank - 1] != rank - 1) {
        const uint32_t rowAxis = perm[rank - 1];
        uint32_t outAxisOfLastInput = 0;
        for (uint32_t axis = 0; axis < rank; ++axis) {
            if (perm[axis] == rank - 1) {
                outAxisOfLastInput = axis;
                break;
            }
        }
        mode = kModeLastAxisTransposeTiled;
        swapAxis = outAxisOfLastInput;
        swapDimA = inputDims[rowAxis];
        swapDimB = inputDims[rank - 1];
        const bool useRank5GroupFamily =
            dt == ge::DT_INT32 && rank == 5 &&
            perm[0] == 3 && perm[1] == 4 && perm[2] == 1 &&
            perm[3] == 0 && perm[4] == 2 &&
            inputDims[2] >= kLastAxisRank5MinRC && inputDims[2] <= kLastAxisRank5MaxRC &&
            inputDims[4] >= kLastAxisRank5MinRC && inputDims[4] <= kLastAxisRank5MaxRC &&
            inputDims[3] >= kLastAxisRank5MinD;
        const uint64_t rank5ADGroupTasks = static_cast<uint64_t>(inputDims[1]) *
            ((inputDims[0] + kLastAxisRank5AGroup - 1) / kLastAxisRank5AGroup) *
            ((inputDims[3] + kLastAxisRank5ADGroup - 1) / kLastAxisRank5ADGroup);
        const uint64_t rank5FullAGroupTasks = static_cast<uint64_t>(inputDims[1]) *
            ((inputDims[3] + kLastAxisRank5FullADGroup - 1) / kLastAxisRank5FullADGroup);
        const uint64_t rank5DGroupTasks = static_cast<uint64_t>(inputDims[0]) * inputDims[1] *
            ((inputDims[3] + kLastAxisRank5DGroup - 1) / kLastAxisRank5DGroup);
        const bool useRank5FullAGroup = useRank5GroupFamily &&
            inputDims[0] >= kLastAxisRank5FullAMin &&
            inputDims[0] <= kLastAxisRank5FullAMax &&
            rank5FullAGroupTasks >= coreNum && rank5FullAGroupTasks <= UINT32_MAX;
        const bool useRank5ADGroup = useRank5GroupFamily &&
            inputDims[0] >= kLastAxisRank5MinAForADGroup &&
            rank5ADGroupTasks >= coreNum && rank5ADGroupTasks <= UINT32_MAX;
        const bool useRank5DGroup = useRank5GroupFamily &&
            rank5DGroupTasks >= coreNum && rank5DGroupTasks <= UINT32_MAX;
        if (useRank5FullAGroup) {
            swapOuter = kLastAxisRank5FullADGroup;
            swapTileA = 32;
            swapTileB = 32;
            taskTotal = static_cast<uint32_t>(rank5FullAGroupTasks);
            directDimA = taskTotal;
            directDimB = 1;
        } else if (useRank5ADGroup) {
            swapOuter = kLastAxisRank5ADGroup;
            swapTileA = 32;
            swapTileB = 32;
            taskTotal = static_cast<uint32_t>(rank5ADGroupTasks);
            directDimA = taskTotal;
            directDimB = 1;
        } else if (useRank5DGroup) {
            swapOuter = kLastAxisRank5DGroup;
            swapTileA = 32;
            swapTileB = 32;
            taskTotal = static_cast<uint32_t>(rank5DGroupTasks);
            directDimA = taskTotal;
            directDimB = 1;
        } else {
            const uint32_t fixedTasks = totalLength / std::max(1U, swapDimA * swapDimB);
            uint32_t matrixTasks = 0;
            if (TryUseLastAxisMacroTile(dt, swapDimA, swapDimB, maxTile,
                                        mode, matrixTasks, swapTileA, swapTileB)) {
                taskTotal = fixedTasks * matrixTasks;
            } else {
                SelectBaseTransposeTile(dt, swapTileA, swapTileB);
                taskTotal = fixedTasks * ((swapDimA + swapTileA - 1) / swapTileA) *
                            ((swapDimB + swapTileB - 1) / swapTileB);
            }
            directDimA = fixedTasks * ((swapDimA + swapTileA - 1) / swapTileA);
            directDimB = (swapDimB + swapTileB - 1) / swapTileB;
        }
    } else if (innerBlock > 1 && innerBlock != totalLength) {
        mode = kModeBlockCopy;
        taskTotal = totalLength / innerBlock;
    } else {
        mode = kModeGeneric;
    }

    // 普通分块路径的短读/非对齐读保留 L2，仅对下面明确选中的流式输入族使用旁路策略。
    uint32_t transpose021B32FullRowTileA = 0;
    uint32_t transpose021B32FullRowTileB = 0;
    const bool transpose021B32FullRow =
        mode == kModeTranspose021Fp16 &&
        (dt == ge::DT_FLOAT || dt == ge::DT_INT32) &&
        SelectTranspose021B32FullRowTile(inputDims[1], inputDims[2],
                                         transpose021B32FullRowTileA,
                                         transpose021B32FullRowTileB) &&
        swapTileA == transpose021B32FullRowTileA &&
        swapTileB == transpose021B32FullRowTileB;
    const bool transpose021B32CoreStrided =
        transpose021B32FullRow && inputDims[1] > 751 &&
        inputDims[2] >= 505 && inputDims[2] < 576;
    const bool transpose021PackedGatherInputL2Bypass =
        mode == kModeTranspose021Fp16 && inputDims[2] < swapTileB &&
        !transpose021B32FullRow;
    const bool transpose021InputL2Bypass =
        mode == kModeTranspose021Macro || transpose021PackedGatherInputL2Bypass ||
        transpose021B32CoreStrided;
    const bool inputL2Bypass =
        (mode == kModeBlockCopy && swapDimA <= 32) || transpose021InputL2Bypass;
    tilingKey = inputL2Bypass ? 1 : 0;

    uint32_t blockDim = 1;
    const bool transpose021PackedInt8 =
        mode == kModeTranspose021Fp16 && dt == ge::DT_INT8 &&
        inputDims[2] < swapTileB;
    if (directDimA != 0 && directDimB != 0) {
        if (mode == kModeLastAxisTransposeTiled && directDimA >= coreNum &&
            directDimB > 1 && directDimB <= coreNum) {
            coreSplitB = directDimB;
            coreSplitA = std::min(directDimA, coreNum / coreSplitB);
        } else {
            SelectDirectCoreSplit(directDimA, directDimB, coreNum, coreSplitA, coreSplitB);
        }
        blockDim = coreSplitA * coreSplitB;
    } else if (transpose021PackedInt8) {
        blockDim = std::min(coreNum, std::max(1U, inputDims[0]));
    } else {
        blockDim = std::min(coreNum, std::max(1U, taskTotal));
    }
    uint32_t taskLength = (taskTotal + blockDim - 1) / blockDim;
    const uint32_t lastTaskStart = taskLength * (blockDim - 1);
    uint32_t lastTaskLength = lastTaskStart < taskTotal ? taskTotal - lastTaskStart : 0;

    uint32_t tileLength = 16 * 16;
    if (mode == kModeTranspose2DMacro || mode == kModeTranspose021Macro ||
        mode == kModeLastAxisTransposeTiled) {
        tileLength = swapTileA * swapTileB;
    } else if (mode == kModeTranspose021Batch) {
        const uint64_t ubBudget = ubSize > kTranspose021ReservedUbBytes ?
            ubSize - kTranspose021ReservedUbBytes : ubSize;
        const uint32_t slotBytes = AlignDown(
            static_cast<uint32_t>(ubBudget / (2 * swapTileB)), kDmaAlignBytes);
        tileLength = slotBytes / dataWidth;
    } else if (mode == kModeTranspose021ResidentOutput) {
        tileLength = swapTileA * swapTileB;
    } else if (mode == kModeReverse3DSuffixGather) {
        const uint32_t inputRow = AlignUp(swapTileB * swapInner * dataWidth, kDmaAlignBytes) / dataWidth;
        tileLength = std::max(swapTileA * inputRow, swapTileA * swapTileB * swapInner);
    } else if (mode == kModeTranspose021Fp16) {
        tileLength = swapTileA * swapTileB;
    } else if (mode == kModeTransposeFlattenTiled) {
        if (dataWidth == 1 && swapTileB > kFlattenInt8BaseTileCols) {
            tileLength = swapTileA * swapTileB;
        } else if (dataWidth == 1) {
            tileLength = 32 * 32;
        } else if (dataWidth == 4) {
            tileLength = 16 * 8;
        } else {
            tileLength = 16 * 16;
        }
    } else if (mode == kModeBlockCopy) {
        uint32_t directBlockLength = innerBlock;
        if (directDimA != 0) {
            directBlockLength = (swapInner * dataWidth) % kDmaAlignBytes == 0
                                    ? swapDimA * swapInner
                                    : 2 * (kStaticUnalignedBlockCopySlotBytes / dataWidth);
        }
        tileLength = std::min(directBlockLength, maxTile);
        if (tileLength > elemPerAlign) {
            tileLength = AlignDown(tileLength, elemPerAlign);
        }
        if (tileLength == 0) {
            tileLength = std::min(innerBlock, elemPerAlign);
        }
    } else {
        tileLength = std::min(taskLength, maxTile);
        if (tileLength == 0) {
            tileLength = 1;
        }
    }

    TransposeTilingData tiling;
    tiling.set_rank(rank);
    tiling.set_mode(mode);
    tiling.set_totalLength(totalLength);
    tiling.set_outputBufferLength(outputBufferLength);
    tiling.set_taskLength(taskLength);
    tiling.set_lastTaskLength(lastTaskLength);
    tiling.set_directDimA(directDimA);
    tiling.set_directDimB(directDimB);
    tiling.set_coreSplitA(coreSplitA);
    tiling.set_coreSplitB(coreSplitB);
    tiling.set_tileLength(tileLength);
    tiling.set_innerBlock(innerBlock);
    tiling.set_swapAxis(swapAxis);
    tiling.set_swapOuter(swapOuter);
    tiling.set_swapDimA(swapDimA);
    tiling.set_swapDimB(swapDimB);
    tiling.set_swapInner(swapInner);
    tiling.set_swapTileA(swapTileA);
    tiling.set_swapTileB(swapTileB);
    tiling.set_inputDims(inputDims);
    tiling.set_inputStrides(inputStrides);
    tiling.set_outputDims(outputDims);
    tiling.set_outputStrides(outputStrides);
    tiling.set_perm(perm);

    // 该 32 位 021 结构族采用 768 行距的 scratch；行列范围与 Kernel 的静态 UB 布局匹配。
    // 这里只选择算法，GM 基址的 4B/512B 对齐由设备入口检查；不能仅凭 shape 假定物理指针对齐。
    const bool alignedWorkspaceOutput = transpose021B32FullRow && rank == 3 &&
        perm[0] == 0 && perm[1] == 2 && perm[2] == 1 &&
        inputDims[1] >= 752 && inputDims[1] <= 768 &&
        inputDims[2] >= 505 && inputDims[2] <= 528;
    if (alignedWorkspaceOutput) {
        tilingKey = 2;
        size_t* sizes = context->GetWorkspaceSizes(1);
        if (sizes == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // 预留系统 workspace、最多 511B 的地址对齐空间，以及每核 scratch 和 8960 个 INT32 LUT。
        // 这里的元素数乘 4 转成字节；布局须与 KernelTransposeAlignedWorkspace::Init 同步维护。
        sizes[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + 512ULL +
            static_cast<size_t>(blockDim) * (AlignUp(inputDims[2] + 12, 16) * 768 + 8960) * 4;
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

// 按原始 dims 推导外部可见的输出 shape；Kernel 内部删除单例轴不会改变此接口契约。
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t perm[optiling::kTransposeMaxRank] = {};
    if (!BuildPerm(*inputShape, context->GetAttrs(), perm)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rank = inputShape->GetDimNum();
    outputShape->SetDimNum(0);
    for (int64_t i = 0; i < rank; ++i) {
        outputShape->AppendDim(inputShape->GetDim(perm[i]));
    }
    return ge::GRAPH_SUCCESS;
}

// Transpose 只改变元素位置，输出数据类型与输入一致。
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("dims").AttrType(REQUIRED).ListInt();

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);

}  // namespace ops
