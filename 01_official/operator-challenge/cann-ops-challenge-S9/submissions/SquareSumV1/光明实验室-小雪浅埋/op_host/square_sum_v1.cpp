#include "square_sum_v1_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
// -----------------------------------------------------------------------------
// 协议常量定义 (与 Kernel 内部布局一一对应)
// -----------------------------------------------------------------------------
enum SchemeTopology : uint32_t {
    SCHEME_AR = 0,
    SCHEME_ARA = 1,
    SCHEME_RAR = 2,
    SCHEME_RA = 3,
    SCHEME_FALLBACK = 4
};

constexpr uint64_t BYTES_PER_BLOCK = 32;
constexpr uint64_t BANK_PADDING_SPAN = 256;
constexpr uint32_t MAX_BURST_ROWS = 4095;
constexpr uint32_t TILING_KEY_SYNC = 1;
constexpr uint32_t TILING_KEY_DIRECT_AR = 2;
constexpr uint32_t MAX_VECTOR_REPEAT_STRIDE = 65535;
constexpr uint32_t MAX_VECTOR_REPEATS = 255;

constexpr uint32_t COMPACT_RA_MIN_GROUPS = 4;
constexpr uint32_t COMPACT_ARA_PHASE_LANES = 2;
constexpr uint32_t RA_MODE_TILED = 1;
constexpr uint32_t RA_MODE_PHASE = 2;
constexpr uint32_t RA_PHASE_SLAB_UNIT = 1;
constexpr uint32_t RA_PHASE_LANES = 1;
constexpr uint32_t RA_SEGMENT_MAX = 2;
constexpr uint32_t RA_SCHEDULE_WAVE = 1;
constexpr bool FLAG_ENABLE_RA_WAVE = true;

constexpr uint64_t FP32_UNIT_ROUNDOFF_BASE = 1ULL << 24;
constexpr uint64_t FP32_BUDGET_DIVISOR = 15000;
constexpr uint32_t MAX_PHASE_SUM_DEPTH = static_cast<uint32_t>(
    (FP32_UNIT_ROUNDOFF_BASE - FP32_BUDGET_DIVISOR) / (FP32_BUDGET_DIVISOR + 1));

constexpr uint32_t RA_CORE_FACTOR = 4;
constexpr uint32_t RA_CORE_RESERVED = 1;
constexpr uint32_t RA_CORE_ACTIVE_RATIO = RA_CORE_FACTOR - RA_CORE_RESERVED;
constexpr uint32_t RA_STAGGER_PARTITIONS = 4;
constexpr uint32_t RA_STAGGER_MIN_ROUNDS = 8;
constexpr uint32_t RA_STAGGER_OFFSET_ROWS = RA_STAGGER_PARTITIONS - 1;
constexpr uint32_t RAR_SCHEDULE_WAVE = 1;
constexpr uint32_t RAR_WAVE_MIN_PASSES = 4;

constexpr uint32_t DMA_AMORTIZE_BLOCK_THRESHOLD = 1024;
constexpr uint64_t MIN_RAR_WAVE_CHUNK_BYTES = BYTES_PER_BLOCK * DMA_AMORTIZE_BLOCK_THRESHOLD;
constexpr uint64_t MIN_RA_WAVE_CHUNK_BYTES = BYTES_PER_BLOCK * DMA_AMORTIZE_BLOCK_THRESHOLD;
constexpr uint32_t MIN_RA_WAVE_FULL_PASSES = RA_STAGGER_MIN_ROUNDS;
constexpr uint64_t MONO_CORE_MEM_THRESHOLD = BYTES_PER_BLOCK * DMA_AMORTIZE_BLOCK_THRESHOLD;
constexpr uint32_t MIN_RAR_CORES = 4;

// -----------------------------------------------------------------------------
// 算术与对齐辅助
// -----------------------------------------------------------------------------
template <typename T>
inline bool MultiplyOverflow(T a, T b, T& out)
{
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        return true;
    }
    out = a * b;
    return false;
}

inline uint64_t AlignTo(uint64_t val, uint64_t align)
{
    if (align == 0) return val;
    const uint64_t rem = val % align;
    if (rem == 0) return val;
    const uint64_t diff = align - rem;
    return (val > std::numeric_limits<uint64_t>::max() - diff) ?
        std::numeric_limits<uint64_t>::max() : val + diff;
}

inline uint64_t DivCeil(uint64_t num, uint64_t den)
{
    return den == 0 ? 0 : num / den + (num % den != 0 ? 1 : 0);
}

inline uint32_t GreatestCommonDivisor(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

inline uint32_t LeastCommonMultiple(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0) return 0;
    const uint32_t g = GreatestCommonDivisor(a, b);
    if (g == 0) return 0;
    const uint64_t prod = static_cast<uint64_t>(a / g) * b;
    return prod > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(prod);
}

// -----------------------------------------------------------------------------
// 维度解析与规约轴提取
// -----------------------------------------------------------------------------
bool ParseReductionAxes(const gert::Shape& tensorShape, const gert::ContinuousVector* attr,
                        std::array<bool, optiling::SQUARE_SUM_MAX_RANK>& flags)
{
    const int32_t dims = tensorShape.GetDimNum();
    flags.fill(false);
    if (dims <= 0 || dims > static_cast<int32_t>(optiling::SQUARE_SUM_MAX_RANK) || attr == nullptr) {
        return false;
    }
    if (attr->GetSize() == 0) {
        std::fill(flags.begin(), flags.begin() + dims, true);
        return true;
    }
    const auto* indices = static_cast<const int64_t*>(attr->GetData());
    if (indices == nullptr) return false;
    for (size_t i = 0; i < attr->GetSize(); ++i) {
        int64_t ax = indices[i];
        if (ax < 0) ax += dims;
        if (ax < 0 || ax >= dims || flags[static_cast<size_t>(ax)]) {
            return false;
        }
        flags[static_cast<size_t>(ax)] = true;
    }
    return true;
}

inline uint64_t ResolveMonoThreshold(uint64_t ubCapacity)
{
    return std::min<uint64_t>(MONO_CORE_MEM_THRESHOLD, ubCapacity);
}

inline bool CheckStreamingNeeded(uint64_t totalBytes, uint64_t cacheL2)
{
    return cacheL2 > 0 && totalBytes > cacheL2;
}

uint32_t DetermineCoreAllocation(uint32_t topology, uint64_t totalBytes, uint64_t length0,
                                uint64_t length1, uint32_t devCores, uint64_t monoThreshold)
{
    if (devCores == 0 || totalBytes <= monoThreshold) {
        return 1;
    }
    uint64_t workload = 1;
    if (topology == SCHEME_AR || topology == SCHEME_ARA || topology == SCHEME_RA) {
        workload = length0;
    } else if (topology == SCHEME_RAR) {
        workload = length1;
    }
    const uint64_t perCoreCap = workload / devCores + (workload % devCores != 0 ? 1 : 0);
    const uint64_t active = perCoreCap == 0 ? 1 : workload / perCoreCap + (workload % perCoreCap != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::max<uint64_t>(1, active));
}

inline uint32_t ScalePhaseCoreLimit(uint32_t requestedCores, uint32_t hardwareCores)
{
    if (requestedCores == 0 || hardwareCores < RA_CORE_FACTOR) {
        return requestedCores;
    }
    const uint32_t bound = static_cast<uint32_t>(
        static_cast<uint64_t>(hardwareCores) * RA_CORE_ACTIVE_RATIO / RA_CORE_FACTOR);
    return std::min(requestedCores, std::max(1U, bound));
}

// -----------------------------------------------------------------------------
// Phase 分解与累加深度分析
// -----------------------------------------------------------------------------
bool ValidateRaPhaseFeasibility(uint64_t redDim, uint32_t innerDim, uint32_t rowsPerPhase,
                                uint32_t numCores, uint32_t tileRows)
{
    if (redDim == 0 || innerDim == 0 || rowsPerPhase == 0 || numCores == 0 || tileRows == 0) {
        return false;
    }
    const uint64_t phaseVolume = static_cast<uint64_t>(rowsPerPhase) * innerDim;
    if (phaseVolume > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        innerDim > std::numeric_limits<uint32_t>::max() / tileRows) {
        return false;
    }
    const uint64_t peakCoreRows = DivCeil(redDim, numCores);
    const uint64_t fullPasses = peakCoreRows / tileRows;
    const uint64_t remRows = peakCoreRows % tileRows;
    const uint64_t groupsInPass = DivCeil(tileRows, rowsPerPhase);
    const uint64_t groupsInTail = DivCeil(remRows, rowsPerPhase);
    uint64_t aggregateGroups = 0;
    if (MultiplyOverflow(fullPasses, groupsInPass, aggregateGroups) ||
        aggregateGroups > std::numeric_limits<uint64_t>::max() - groupsInTail) {
        return false;
    }
    aggregateGroups += groupsInTail;
    const uint64_t laneHeight = DivCeil(aggregateGroups, RA_PHASE_LANES);
    const uint64_t overhead = (RA_PHASE_LANES - 1U) + (rowsPerPhase - 1U) + (numCores - 1U);
    if (laneHeight > std::numeric_limits<uint64_t>::max() - overhead) {
        return false;
    }
    const uint64_t loopRounds = DivCeil(peakCoreRows, tileRows);
    return laneHeight + overhead <= MAX_PHASE_SUM_DEPTH &&
           tileRows >= rowsPerPhase * 2U && loopRounds >= RA_STAGGER_MIN_ROUNDS;
}

bool AnalyzePhaseGrouping(uint64_t redDim, uint32_t rowsPerPhase, uint32_t numCores,
                          uint64_t& lowRows, uint64_t& highRows, uint64_t& maxGroups)
{
    lowRows = 0;
    highRows = 0;
    maxGroups = 0;
    if (redDim == 0 || rowsPerPhase == 0 || numCores == 0) {
        return false;
    }
    const uint64_t totalGroups = redDim / rowsPerPhase;
    const uint64_t remElements = redDim % rowsPerPhase;
    if (totalGroups < numCores) return false;

    const uint64_t uniformGroups = totalGroups / numCores;
    const uint64_t overflowGroups = totalGroups % numCores;
    const uint64_t topGroups = uniformGroups + (overflowGroups > 0 ? 1U : 0U);
    const uint64_t finalGroups = uniformGroups + (remElements > 0 ? 1U : 0U);
    maxGroups = std::max(topGroups, finalGroups);

    uint64_t baseVolume = 0;
    uint64_t topVolume = 0;
    uint64_t tailVolume = 0;
    if (MultiplyOverflow(uniformGroups, static_cast<uint64_t>(rowsPerPhase), baseVolume) ||
        MultiplyOverflow(topGroups, static_cast<uint64_t>(rowsPerPhase), topVolume) ||
        MultiplyOverflow(uniformGroups, static_cast<uint64_t>(rowsPerPhase), tailVolume) ||
        tailVolume > std::numeric_limits<uint64_t>::max() - remElements) {
        return false;
    }
    tailVolume += remElements;
    highRows = std::max(topVolume, tailVolume);
    lowRows = tailVolume;
    if (overflowGroups > 0) {
        lowRows = std::min(lowRows, topVolume);
    }
    if (overflowGroups < static_cast<uint64_t>(numCores - 1U)) {
        lowRows = std::min(lowRows, baseVolume);
    }
    return lowRows > 0 && lowRows <= highRows && highRows <= redDim;
}

bool EvaluateCheckpointRequirement(uint64_t redDim, uint32_t rowsPerPhase, uint32_t numCores)
{
    uint64_t lowR = 0, highR = 0, peakG = 0;
    if (!AnalyzePhaseGrouping(redDim, rowsPerPhase, numCores, lowR, highR, peakG)) {
        return false;
    }
    const uint64_t overhead = (RA_PHASE_LANES - 1U) + (rowsPerPhase - 1U) + (numCores - 1U);
    return peakG > MAX_PHASE_SUM_DEPTH || overhead > MAX_PHASE_SUM_DEPTH - peakG;
}

bool ComputeSegmentPartition(uint64_t minG, uint64_t maxG, uint32_t rowsPerPhase,
                             uint32_t numCores, uint32_t tileRows, uint32_t& numSegments,
                             uint32_t& snapGroups)
{
    numSegments = 0;
    snapGroups = 0;
    if (minG == 0 || maxG < minG || rowsPerPhase == 0 || numCores == 0 || tileRows == 0 ||
        tileRows % rowsPerPhase != 0) {
        return false;
    }
    const uint64_t overhead = (RA_PHASE_LANES - 1U) + (rowsPerPhase - 1U) + (numCores - 1U);
    if (maxG <= std::numeric_limits<uint64_t>::max() - overhead &&
        maxG + overhead <= MAX_PHASE_SUM_DEPTH) {
        numSegments = 1;
        return true;
    }
    const uint64_t stepGroups = tileRows / rowsPerPhase;
    const uint64_t guardWindow = 2U * stepGroups;
    if (stepGroups == 0 || minG < 2U * guardWindow) {
        return false;
    }
    const uint64_t median = DivCeil(maxG, 2U);
    const uint64_t ptLower = median / stepGroups * stepGroups;
    const uint64_t ptUpper = (ptLower == median) ? ptLower :
        (ptLower <= std::numeric_limits<uint64_t>::max() - stepGroups ? ptLower + stepGroups : 0U);

    uint64_t pickedCheckpoint = 0;
    uint64_t optimalSpan = std::numeric_limits<uint64_t>::max();
    const uint64_t attempts[2] = {ptLower, ptUpper};
    for (uint64_t probe : attempts) {
        if (probe < guardWindow || probe > minG - guardWindow ||
            probe > std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        const uint64_t currentMax = std::max(probe, maxG - probe);
        if (currentMax < optimalSpan) {
            pickedCheckpoint = probe;
            optimalSpan = currentMax;
        }
    }
    if (pickedCheckpoint == 0) return false;

    const uint64_t segOverhead = overhead + 1U;
    if (optimalSpan > std::numeric_limits<uint64_t>::max() - segOverhead ||
        optimalSpan + segOverhead > MAX_PHASE_SUM_DEPTH) {
        return false;
    }
    numSegments = RA_SEGMENT_MAX;
    snapGroups = static_cast<uint32_t>(pickedCheckpoint);
    return true;
}

bool CheckRaPhaseAlignment(uint64_t redDim, uint32_t innerDim, uint32_t rowsPerPhase,
                           uint32_t numCores, uint32_t tileRows, uint32_t* segOut = nullptr,
                           uint32_t* chkOut = nullptr)
{
    if (innerDim == 0 || rowsPerPhase == 0 || tileRows == 0 ||
        rowsPerPhase > std::numeric_limits<uint32_t>::max() / 2U ||
        tileRows % rowsPerPhase != 0 || innerDim > std::numeric_limits<uint32_t>::max() / tileRows) {
        return false;
    }
    uint64_t lowR = 0, highR = 0, peakG = 0;
    if (!AnalyzePhaseGrouping(redDim, rowsPerPhase, numCores, lowR, highR, peakG)) {
        return false;
    }
    uint64_t tileCount = 0, tileBytes = 0;
    if (MultiplyOverflow(static_cast<uint64_t>(tileRows), static_cast<uint64_t>(innerDim), tileCount) ||
        MultiplyOverflow(tileCount, static_cast<uint64_t>(sizeof(float)), tileBytes) ||
        tileBytes % BYTES_PER_BLOCK != 0) {
        return false;
    }
    const uint64_t minG = DivCeil(lowR, rowsPerPhase);
    uint32_t sCount = 0, cCount = 0;
    if (!ComputeSegmentPartition(minG, peakG, rowsPerPhase, numCores, tileRows, sCount, cCount)) {
        return false;
    }
    const uint64_t rounds = DivCeil(highR, tileRows);
    const bool valid = (DivCeil(lowR, tileRows) == rounds) &&
                       (tileRows >= rowsPerPhase * 2U) && (rounds >= RA_STAGGER_MIN_ROUNDS);
    if (valid) {
        if (segOut) *segOut = sCount;
        if (chkOut) *chkOut = cCount;
    }
    return valid;
}

bool GaugePhaseWaveDimensions(uint64_t redDim, uint32_t rowsPerPhase, uint32_t numCores,
                              uint32_t tileRows, uint64_t& minG, uint64_t& maxG)
{
    minG = 0;
    maxG = 0;
    if (redDim == 0 || rowsPerPhase == 0 || numCores == 0 || tileRows == 0 ||
        rowsPerPhase > std::numeric_limits<uint32_t>::max() / 2U ||
        tileRows % rowsPerPhase != 0 || tileRows < rowsPerPhase * 2U) {
        return false;
    }
    const uint64_t passes = redDim / tileRows;
    if (passes / numCores < MIN_RA_WAVE_FULL_PASSES) return false;

    const uint64_t gPerTile = tileRows / rowsPerPhase;
    const uint64_t baseline = passes / numCores;
    const uint64_t remPass = passes % numCores;
    if (MultiplyOverflow(baseline, gPerTile, minG) || minG == 0) return false;

    uint64_t topPass = baseline;
    if (remPass != 0) {
        if (topPass == std::numeric_limits<uint64_t>::max()) return false;
        ++topPass;
    }
    uint64_t topG = 0, tailG = 0;
    if (MultiplyOverflow(topPass, gPerTile, topG) ||
        MultiplyOverflow(baseline, gPerTile, tailG)) {
        return false;
    }
    const uint64_t tailRows = redDim % tileRows;
    if (tailRows != 0) {
        const uint64_t tailGroups = DivCeil(tailRows, rowsPerPhase);
        if (tailG > std::numeric_limits<uint64_t>::max() - tailGroups) return false;
        tailG += tailGroups;
    }
    maxG = std::max(topG, tailG);
    return minG <= maxG;
}

bool ValidatePhaseWaveCompatibility(uint64_t redDim, uint32_t innerDim, uint32_t rowsPerPhase,
                                    uint32_t numCores, uint32_t tileRows, uint32_t* segOut = nullptr,
                                    uint32_t* chkOut = nullptr)
{
    if (innerDim == 0 || tileRows == 0 || innerDim > std::numeric_limits<uint32_t>::max() / tileRows) {
        return false;
    }
    uint64_t count = 0, bytes = 0;
    if (MultiplyOverflow(static_cast<uint64_t>(tileRows), static_cast<uint64_t>(innerDim), count) ||
        MultiplyOverflow(count, static_cast<uint64_t>(sizeof(float)), bytes) ||
        bytes % BYTES_PER_BLOCK != 0 || bytes < MIN_RA_WAVE_CHUNK_BYTES) {
        return false;
    }
    uint64_t minG = 0, maxG = 0;
    if (!GaugePhaseWaveDimensions(redDim, rowsPerPhase, numCores, tileRows, minG, maxG)) {
        return false;
    }
    uint32_t sCount = 0, cCount = 0;
    if (!ComputeSegmentPartition(minG, maxG, rowsPerPhase, numCores, tileRows, sCount, cCount)) {
        return false;
    }
    if (segOut) *segOut = sCount;
    if (chkOut) *chkOut = cCount;
    return true;
}

uint32_t FineTuneAlignedTile(uint32_t ceiling, uint64_t redDim, uint32_t innerDim,
                             uint32_t rowsPerPhase, uint32_t numCores)
{
    if (ceiling == 0 || rowsPerPhase == 0 || rowsPerPhase > std::numeric_limits<uint32_t>::max() / 2U) {
        return 0;
    }
    const uint32_t floorLimit = rowsPerPhase * 2U;
    uint32_t iter = ceiling / rowsPerPhase * rowsPerPhase;
    while (iter >= floorLimit) {
        if (CheckRaPhaseAlignment(redDim, innerDim, rowsPerPhase, numCores, iter)) {
            return iter;
        }
        if (iter < floorLimit + rowsPerPhase) break;
        iter -= rowsPerPhase;
    }
    return 0;
}

uint32_t OptimizeDmaAlignmentTile(uint32_t ceiling, uint64_t redDim, uint32_t innerDim,
                                 uint32_t rowsPerPhase, uint32_t numCores)
{
    if (ceiling == 0 || innerDim == 0 || rowsPerPhase == 0 || numCores == 0) return ceiling;
    const uint32_t floorBound = static_cast<uint32_t>(DivCeil(static_cast<uint64_t>(ceiling) * 3U, 4U));
    const uint64_t strideBytes = static_cast<uint64_t>(innerDim) * sizeof(float);
    uint32_t optimal = ceiling;
    uint32_t primeDiv = GreatestCommonDivisor(
        static_cast<uint32_t>((static_cast<uint64_t>(ceiling) * strideBytes) % 512U), 512U);

    uint32_t candidate = ceiling / rowsPerPhase * rowsPerPhase;
    while (candidate >= rowsPerPhase * 2U) {
        if (candidate < floorBound) break;
        uint64_t totalBytes = 0;
        if (!MultiplyOverflow(static_cast<uint64_t>(candidate), strideBytes, totalBytes) &&
            totalBytes >= MIN_RA_WAVE_CHUNK_BYTES &&
            CheckRaPhaseAlignment(redDim, innerDim, rowsPerPhase, numCores, candidate)) {
            const uint32_t currentDiv = GreatestCommonDivisor(
                static_cast<uint32_t>(totalBytes % 512U), 512U);
            if (currentDiv > primeDiv) {
                optimal = candidate;
                primeDiv = currentDiv;
            }
        }
        if (candidate < rowsPerPhase * 3U) break;
        candidate -= rowsPerPhase;
    }
    return optimal;
}

uint32_t SelectSafeRaTileFallback(uint32_t ceiling, uint64_t redDim, uint32_t innerDim,
                                  uint32_t elemBytes, uint32_t rowsPerPhase, uint32_t numCores)
{
    if (rowsPerPhase == 0 || elemBytes == 0) return 0;
    for (uint32_t round = 0; round < 2; ++round) {
        for (uint32_t cand = ceiling; cand >= rowsPerPhase * 2U; --cand) {
            if (!ValidateRaPhaseFeasibility(redDim, innerDim, rowsPerPhase, numCores, cand)) {
                continue;
            }
            uint64_t volume = 0, bytes = 0;
            const bool valid =
                !MultiplyOverflow(static_cast<uint64_t>(cand), static_cast<uint64_t>(innerDim), volume) &&
                !MultiplyOverflow(volume, static_cast<uint64_t>(elemBytes), bytes);
            if (round != 0 || (valid && bytes % BYTES_PER_BLOCK != 0)) {
                return cand;
            }
        }
    }
    return 0;
}

bool CheckStaggerRaFeasible(uint64_t redDim, uint32_t numCores, uint32_t tilePrimary)
{
    if (numCores < RA_STAGGER_PARTITIONS || tilePrimary == 0) return false;
    const uint64_t shortRows = redDim / numCores;
    const uint64_t longRows = DivCeil(redDim, numCores);
    const uint64_t shortRounds = DivCeil(shortRows, tilePrimary);
    const uint64_t longRounds = DivCeil(longRows, tilePrimary);
    if (shortRounds != longRounds || shortRounds < RA_STAGGER_MIN_ROUNDS) return false;

    const uint64_t shortRem = shortRows - (shortRounds - 1) * tilePrimary;
    const uint64_t longRem = longRows - (longRounds - 1) * tilePrimary;
    return tilePrimary - shortRem >= RA_STAGGER_OFFSET_ROWS &&
           tilePrimary - longRem >= RA_STAGGER_OFFSET_ROWS;
}

bool CheckWaveRarFeasible(uint64_t redDim, uint64_t innerDim, uint32_t elemBytes,
                          uint32_t numCores, uint32_t tilePrimary, uint32_t quantum)
{
    if (numCores < MIN_RAR_CORES || quantum == 0 || tilePrimary == 0 ||
        innerDim == 0 || elemBytes == 0 || tilePrimary % quantum != 0) {
        return false;
    }
    uint64_t waveSpan = 0, tileElems = 0, tileBytes = 0;
    if (MultiplyOverflow(static_cast<uint64_t>(numCores), static_cast<uint64_t>(tilePrimary), waveSpan) ||
        MultiplyOverflow(static_cast<uint64_t>(tilePrimary), innerDim, tileElems) ||
        MultiplyOverflow(tileElems, static_cast<uint64_t>(elemBytes), tileBytes)) {
        return false;
    }
    return waveSpan > 0 && redDim / waveSpan >= RAR_WAVE_MIN_PASSES &&
           tileBytes >= MIN_RAR_WAVE_CHUNK_BYTES;
}

// -----------------------------------------------------------------------------
// UB 预算二分与 Tile 规划
// -----------------------------------------------------------------------------
inline uint32_t FetchReductionScratch(uint32_t rows, uint32_t cols, AscendC::ReducePattern p)
{
    ge::Shape shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    uint32_t maxS = 0, minS = 0;
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT, p, true, true, maxS, minS);
    return static_cast<uint32_t>(AlignTo(minS, BYTES_PER_BLOCK));
}

template <typename Predicate>
inline uint32_t BinarySearchCapacity(uint64_t limit, uint64_t workload, Predicate&& pred)
{
    uint32_t l = 1;
    uint32_t r = static_cast<uint32_t>(std::min<uint64_t>(workload, MAX_BURST_ROWS));
    uint32_t res = 0;
    while (l <= r) {
        const uint32_t m = l + (r - l) / 2;
        if (pred(m) <= limit) {
            res = m;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    return res;
}

uint32_t PickDenseTile(uint64_t ubCapacity, uint64_t workload, uint32_t alignedInner,
                       uint32_t elemBytes, uint32_t queueDepth, AscendC::ReducePattern p,
                       uint32_t& scratchOut)
{
    if (workload == 0 || alignedInner == 0) return 0;
    uint32_t l = 1;
    uint32_t r = static_cast<uint32_t>(std::min<uint64_t>(workload, MAX_BURST_ROWS));
    uint32_t optimal = 0;
    uint32_t primeScratch = 0;
    while (l <= r) {
        const uint32_t m = l + (r - l) / 2;
        const uint32_t tmp = FetchReductionScratch(m, alignedInner, p);
        const uint64_t count = static_cast<uint64_t>(m) * alignedInner;
        const uint64_t peakDim = std::max<uint64_t>(m, alignedInner);
        uint64_t usage = queueDepth * (count * elemBytes + BANK_PADDING_SPAN);
        if (elemBytes != sizeof(float)) {
            usage += count * sizeof(float) + BANK_PADDING_SPAN;
        }
        usage += 2 * AlignTo(peakDim * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(peakDim * elemBytes, BYTES_PER_BLOCK);
        usage += tmp;
        if (usage <= ubCapacity) {
            optimal = m;
            primeScratch = tmp;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    scratchOut = primeScratch;
    return optimal;
}

uint32_t PickCompactTile(uint64_t ubCapacity, uint64_t workload, uint32_t innerDim,
                         uint32_t elemBytes, uint32_t queueDepth, uint32_t extraBytes = 0)
{
    if (workload == 0 || innerDim == 0) return 0;
    return BinarySearchCapacity(ubCapacity, workload, [&](uint32_t m) {
        const uint64_t count = static_cast<uint64_t>(m) * innerDim;
        const uint64_t peakDim = std::max<uint64_t>(m, innerDim);
        uint64_t usage = queueDepth * (count * elemBytes + BANK_PADDING_SPAN);
        if (elemBytes != sizeof(float)) {
            usage += count * sizeof(float) + BANK_PADDING_SPAN;
        }
        usage += 2 * AlignTo(peakDim * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(peakDim * elemBytes, BYTES_PER_BLOCK);
        usage += AlignTo(extraBytes, BYTES_PER_BLOCK);
        return usage;
    });
}

uint32_t PickRaTile(uint64_t ubCapacity, uint64_t workload, uint32_t innerDim,
                    uint32_t alignedInner, uint32_t elemBytes, uint32_t extraBytes, uint32_t numCores)
{
    if (workload == 0 || innerDim == 0 || alignedInner == 0 || numCores == 0) return 0;
    uint32_t optimal = BinarySearchCapacity(ubCapacity, workload, [&](uint32_t m) {
        const uint64_t count = static_cast<uint64_t>(m) * innerDim;
        const uint64_t peakDim = std::max<uint64_t>(m, alignedInner);
        uint64_t usage = count * elemBytes + BANK_PADDING_SPAN;
        usage += 2 * AlignTo(peakDim * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(extraBytes, BYTES_PER_BLOCK);
        return usage;
    });

    while (optimal > 0) {
        const uint32_t mergeRows = std::min(optimal, numCores - 1U);
        const uint64_t qMem = static_cast<uint64_t>(optimal) * innerDim * elemBytes + BANK_PADDING_SPAN;
        const uint64_t mMem = static_cast<uint64_t>(mergeRows) * alignedInner * sizeof(float);
        if (qMem >= mMem) break;
        --optimal;
    }
    return optimal;
}

uint32_t PickRaPhaseTile(uint64_t ubCapacity, uint64_t workload, uint32_t innerDim,
                         uint32_t alignedInner, uint32_t elemBytes, uint32_t queueDepth,
                         uint32_t numCores, uint32_t rowsPerPhase, bool hasSegments)
{
    if (workload == 0 || innerDim == 0 || alignedInner == 0 || elemBytes == 0 ||
        queueDepth == 0 || numCores == 0 || rowsPerPhase == 0) {
        return 0;
    }
    const uint64_t pVolume = static_cast<uint64_t>(rowsPerPhase) * innerDim;
    const uint64_t fVolume = 3ULL * alignedInner;
    uint32_t l = 1;
    uint32_t r = static_cast<uint32_t>(std::min<uint64_t>(workload, MAX_BURST_ROWS));
    uint32_t optimal = 0;
    while (l <= r) {
        const uint32_t m = l + (r - l) / 2;
        const uint64_t count = static_cast<uint64_t>(m) * innerDim;
        uint64_t usage = queueDepth * AlignTo(count * elemBytes + BYTES_PER_BLOCK, BYTES_PER_BLOCK);
        usage += AlignTo(RA_PHASE_LANES * pVolume * sizeof(float), BYTES_PER_BLOCK);
        if (hasSegments) {
            usage += AlignTo(fVolume * sizeof(float), BYTES_PER_BLOCK);
            usage += AlignTo(static_cast<uint64_t>(alignedInner) * sizeof(float), BYTES_PER_BLOCK);
        }
        if (usage <= ubCapacity) {
            optimal = m;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    if (static_cast<uint64_t>(optimal) * innerDim < fVolume) return 0;
    return optimal;
}

uint32_t PickAraPhaseTile(uint64_t ubCapacity, uint64_t workload, uint32_t innerDim,
                          uint32_t elemBytes, uint32_t queueDepth, uint32_t rowsPerPhase)
{
    if (workload == 0 || innerDim == 0 || rowsPerPhase == 0) return 0;
    const uint64_t pVolume = static_cast<uint64_t>(rowsPerPhase) * innerDim;
    const uint64_t wVolume = std::max<uint64_t>(pVolume, 3 * AlignTo(innerDim, BYTES_PER_BLOCK / sizeof(float)));
    uint32_t l = 1;
    uint32_t r = static_cast<uint32_t>(std::min<uint64_t>(workload, MAX_BURST_ROWS));
    uint32_t optimal = 0;
    while (l <= r) {
        const uint32_t m = l + (r - l) / 2;
        const uint64_t count = static_cast<uint64_t>(m) * innerDim;
        uint64_t usage = queueDepth * AlignTo(count * elemBytes + BYTES_PER_BLOCK, BYTES_PER_BLOCK);
        usage += AlignTo(wVolume * sizeof(float), BYTES_PER_BLOCK);
        usage += AlignTo(COMPACT_ARA_PHASE_LANES * pVolume * sizeof(float), BYTES_PER_BLOCK);
        usage += AlignTo(innerDim * elemBytes, BYTES_PER_BLOCK);
        if (usage <= ubCapacity) {
            optimal = m;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    if (optimal < workload) {
        optimal = optimal / rowsPerPhase * rowsPerPhase;
    }
    return optimal;
}

uint32_t PickArTile(uint64_t ubCapacity, uint64_t workload, uint32_t innerDim,
                    uint32_t elemBytes, uint32_t queueDepth, uint32_t packStride)
{
    if (workload == 0 || innerDim < 16 || packStride == 0) return 0;
    uint32_t l = 1;
    uint32_t r = static_cast<uint32_t>(std::min<uint64_t>(
        workload, std::min<uint32_t>(MAX_BURST_ROWS, MAX_VECTOR_REPEATS)));
    uint32_t optimal = 0;
    while (l <= r) {
        const uint32_t m = l + (r - l) / 2;
        const uint64_t count = static_cast<uint64_t>(m) * innerDim;
        const uint64_t alignedRows = AlignTo(m, BYTES_PER_BLOCK / sizeof(float));
        uint64_t usage = queueDepth * AlignTo(count * elemBytes + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(count * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        const uint64_t blockSums = (count + 63) / 64 * 8 + 8;
        usage += AlignTo(blockSums * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(static_cast<uint64_t>(m) * packStride * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += 2 * AlignTo(2 * alignedRows * sizeof(float) + BANK_PADDING_SPAN, BYTES_PER_BLOCK);
        usage += AlignTo(static_cast<uint64_t>(m) * packStride * sizeof(uint32_t), BYTES_PER_BLOCK);
        usage += AlignTo(static_cast<uint64_t>(m) * elemBytes, BYTES_PER_BLOCK);
        if (usage <= ubCapacity) {
            optimal = m;
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    return optimal;
}
} // namespace

// -----------------------------------------------------------------------------
// GE 接口实现：形状与数据类型推导
// -----------------------------------------------------------------------------
namespace ge {
static ge::graphStatus DeriveShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    const auto* axAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    const bool* keepDims = context->GetAttrs()->GetAttrPointer<bool>(1);
    std::array<bool, optiling::SQUARE_SUM_MAX_RANK> flags{};
    if (keepDims == nullptr || !ParseReductionAxes(*inShape, axAttr, flags)) {
        return ge::GRAPH_FAILED;
    }
    const int32_t dims = inShape->GetDimNum();
    outShape->SetDimNum(0);
    for (int32_t i = 0; i < dims; ++i) {
        if (flags[static_cast<size_t>(i)]) {
            if (*keepDims) outShape->AppendDim(1);
        } else {
            outShape->AppendDim(inShape->GetDim(i));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DeriveType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

// -----------------------------------------------------------------------------
// 算子 Tiling 主逻辑
// -----------------------------------------------------------------------------
namespace optiling {
static ge::graphStatus BuildTiling(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputDesc(0) == nullptr || context->GetAttrs() == nullptr ||
        context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& tensorShape = context->GetInputShape(0)->GetStorageShape();
    const int32_t dims = tensorShape.GetDimNum();
    const auto* axAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    std::array<bool, SQUARE_SUM_MAX_RANK> axisFlags{};
    if (!ParseReductionAxes(tensorShape, axAttr, axisFlags)) {
        return ge::GRAPH_FAILED;
    }

    std::array<uint32_t, SQUARE_SUM_MAX_RANK> shapeVec{};
    std::array<uint64_t, SQUARE_SUM_MAX_RANK> strideVec{};
    uint64_t totalInputs = 1;
    uint64_t totalOutputs = 1;
    uint64_t totalReduces = 1;
    uint32_t maskBits = 0;

    for (int32_t i = dims - 1; i >= 0; --i) {
        const int64_t curDim = tensorShape.GetDim(i);
        if (curDim <= 0 || static_cast<uint64_t>(curDim) > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        shapeVec[static_cast<size_t>(i)] = static_cast<uint32_t>(curDim);
        strideVec[static_cast<size_t>(i)] = totalInputs;
        if (MultiplyOverflow(totalInputs, static_cast<uint64_t>(curDim), totalInputs)) {
            return ge::GRAPH_FAILED;
        }
        if (axisFlags[static_cast<size_t>(i)]) {
            maskBits |= 1U << i;
            if (MultiplyOverflow(totalReduces, static_cast<uint64_t>(curDim), totalReduces)) {
                return ge::GRAPH_FAILED;
            }
        } else if (MultiplyOverflow(totalOutputs, static_cast<uint64_t>(curDim), totalOutputs)) {
            return ge::GRAPH_FAILED;
        }
    }

    std::vector<std::pair<bool, uint64_t>> contiguousBlocks;
    for (int32_t i = 0; i < dims; ++i) {
        const bool isRed = axisFlags[static_cast<size_t>(i)];
        if (contiguousBlocks.empty() || contiguousBlocks.back().first != isRed) {
            contiguousBlocks.emplace_back(isRed, shapeVec[static_cast<size_t>(i)]);
        } else if (MultiplyOverflow(contiguousBlocks.back().second,
                   static_cast<uint64_t>(shapeVec[static_cast<size_t>(i)]), contiguousBlocks.back().second)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t chosenTopology = SCHEME_FALLBACK;
    uint64_t length0 = 1, length1 = 1, length2 = 1;
    if (contiguousBlocks.size() == 1 && contiguousBlocks[0].first) {
        chosenTopology = SCHEME_AR;
        length1 = contiguousBlocks[0].second;
    } else if (contiguousBlocks.size() == 2 && !contiguousBlocks[0].first && contiguousBlocks[1].first) {
        chosenTopology = SCHEME_AR;
        length0 = contiguousBlocks[0].second;
        length1 = contiguousBlocks[1].second;
    } else if (contiguousBlocks.size() == 3 && !contiguousBlocks[0].first && contiguousBlocks[1].first && !contiguousBlocks[2].first) {
        chosenTopology = SCHEME_ARA;
        length0 = contiguousBlocks[0].second;
        length1 = contiguousBlocks[1].second;
        length2 = contiguousBlocks[2].second;
    } else if (contiguousBlocks.size() == 3 && contiguousBlocks[0].first && !contiguousBlocks[1].first && contiguousBlocks[2].first) {
        chosenTopology = SCHEME_RAR;
        length0 = contiguousBlocks[0].second;
        length1 = contiguousBlocks[1].second;
        length2 = contiguousBlocks[2].second;
    } else if (contiguousBlocks.size() == 2 && contiguousBlocks[0].first && !contiguousBlocks[1].first) {
        chosenTopology = SCHEME_RA;
        length0 = contiguousBlocks[0].second;
        length1 = contiguousBlocks[1].second;
    }

    const ge::DataType dType = context->GetInputDesc(0)->GetDataType();
    const int64_t rawTypeBytes = ge::GetSizeByDataType(dType);
    if (rawTypeBytes != 2 && rawTypeBytes != 4) return ge::GRAPH_FAILED;
    const uint32_t elemBytes = static_cast<uint32_t>(rawTypeBytes);

    uint64_t inputByteSize = 0;
    if (MultiplyOverflow(totalInputs, static_cast<uint64_t>(elemBytes), inputByteSize)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC devPlatform(context->GetPlatformInfo());
    const uint32_t devCores = static_cast<uint32_t>(devPlatform.GetCoreNumAiv());
    uint64_t ubCapacity = 0, cacheL2Capacity = 0;
    devPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCapacity);
    devPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, cacheL2Capacity);
    if (devCores == 0 || ubCapacity < 32 * 1024) return ge::GRAPH_FAILED;

    const uint64_t soloCutoff = ResolveMonoThreshold(ubCapacity);
    const bool isStreaming = CheckStreamingNeeded(inputByteSize, cacheL2Capacity);
    uint32_t activeCores = (chosenTopology == SCHEME_FALLBACK) ? 1 :
        DetermineCoreAllocation(chosenTopology, inputByteSize, length0, length1, devCores, soloCutoff);

    uint64_t innerDim = 1;
    uint64_t workUnits = 1;
    AscendC::ReducePattern redPattern = AscendC::ReducePattern::AR;

    if (chosenTopology == SCHEME_AR) {
        innerDim = length1;
        workUnits = DivCeil(length0, activeCores);
    } else if (chosenTopology == SCHEME_ARA) {
        innerDim = length2;
        workUnits = length1;
        redPattern = AscendC::ReducePattern::RA;
    } else if (chosenTopology == SCHEME_RAR) {
        innerDim = length2;
        workUnits = DivCeil(length1, activeCores);
    } else if (chosenTopology == SCHEME_RA) {
        innerDim = length1;
        workUnits = DivCeil(length0, activeCores);
        redPattern = AscendC::ReducePattern::RA;
    } else {
        innerDim = shapeVec[static_cast<size_t>(dims - 1)];
        workUnits = 1;
    }

    const uint64_t alignUnit = BYTES_PER_BLOCK / elemBytes;
    uint64_t innerPadded64 = AlignTo(innerDim, alignUnit);
    if (innerPadded64 == 0 || innerPadded64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t standardInnerPadded = static_cast<uint32_t>(innerPadded64);
    uint32_t innerPadded = standardInnerPadded;
    uint32_t queueSlots = 1;
    uint32_t extraScratchBytes = 0;

    const uint32_t strideGCD = static_cast<uint32_t>(innerDim) /
        GreatestCommonDivisor(static_cast<uint32_t>(innerDim), BYTES_PER_BLOCK / sizeof(float));
    const uint32_t compactPackStride = static_cast<uint32_t>(AlignTo(
        innerDim / (BYTES_PER_BLOCK / sizeof(float)), BYTES_PER_BLOCK / sizeof(float)));

    bool flagCompactAr = (chosenTopology == SCHEME_AR && elemBytes == 2 &&
                          inputByteSize > soloCutoff && innerPadded > innerDim &&
                          innerDim >= BYTES_PER_BLOCK / elemBytes &&
                          strideGCD <= MAX_VECTOR_REPEAT_STRIDE);

    const uint64_t rarRowBytes = (innerDim > std::numeric_limits<uint64_t>::max() / elemBytes) ?
        std::numeric_limits<uint64_t>::max() : innerDim * elemBytes;
    bool flagCompactRar = (chosenTopology == SCHEME_RAR && dType == ge::DT_FLOAT16 &&
                           rarRowBytes < BANK_PADDING_SPAN && innerPadded > innerDim &&
                           strideGCD <= MAX_VECTOR_REPEAT_STRIDE);

    bool flagCompactRa = (chosenTopology == SCHEME_RA && dType == ge::DT_FLOAT &&
                          innerPadded > innerDim && isStreaming);

    const uint32_t raPhaseBase = BYTES_PER_BLOCK / sizeof(float) /
        GreatestCommonDivisor(static_cast<uint32_t>(innerDim), BYTES_PER_BLOCK / sizeof(float));

    uint32_t modeCompactRa = 0;
    uint32_t raPhaseRows = 0;
    bool flagRaPhaseSplit = false;
    uint32_t raPhaseSegments = 0;
    uint32_t raPhaseCheckpoints = 0;
    uint32_t raPhaseSchedule = 0;

    bool flagCompactAra = (chosenTopology == SCHEME_ARA && innerPadded > innerDim);
    bool flagPhaseAra = false;
    uint32_t rarRowQuantum = 1;
    uint32_t optimalTile = 0;

    if (flagCompactAr) {
        innerPadded = static_cast<uint32_t>(innerDim);
        queueSlots = 1;
        optimalTile = PickArTile(ubCapacity, workUnits, innerPadded, elemBytes, queueSlots, compactPackStride);
        if (optimalTile < 2) {
            flagCompactAr = false;
            innerPadded = standardInnerPadded;
            queueSlots = 1;
            extraScratchBytes = 0;
            optimalTile = 0;
        } else {
            if (workUnits > optimalTile) {
                const uint32_t dbTile = PickArTile(ubCapacity, workUnits, innerPadded, elemBytes, 2, compactPackStride);
                if (dbTile >= 2) {
                    queueSlots = 2;
                    optimalTile = dbTile;
                }
            }
            extraScratchBytes = static_cast<uint32_t>(AlignTo(
                static_cast<uint64_t>(optimalTile) * compactPackStride * sizeof(uint32_t), BYTES_PER_BLOCK));
        }
    }

    if (flagCompactRar) {
        innerPadded = static_cast<uint32_t>(innerDim);
        queueSlots = (length0 > 1) ? 2 : 1;
        optimalTile = PickCompactTile(ubCapacity, workUnits, innerPadded, elemBytes, queueSlots);
        if (optimalTile > 0) {
            const uint32_t vQuantum = 8U / GreatestCommonDivisor(innerPadded, 8U);
            const uint32_t modB = static_cast<uint32_t>((static_cast<uint64_t>(innerPadded) * elemBytes) % BYTES_PER_BLOCK);
            const uint32_t cQuantum = static_cast<uint32_t>(BYTES_PER_BLOCK) /
                GreatestCommonDivisor(modB, static_cast<uint32_t>(BYTES_PER_BLOCK));
            const uint32_t jointQuantum = LeastCommonMultiple(vQuantum, cQuantum);
            rarRowQuantum = jointQuantum;
            const uint32_t alignedM = (jointQuantum == 0) ? 0 : optimalTile / jointQuantum * jointQuantum;
            const uint64_t origPasses = DivCeil(workUnits, optimalTile);
            const uint64_t newPasses = (alignedM == 0) ? 0 : DivCeil(workUnits, alignedM);
            if (alignedM > 0 && newPasses == origPasses) {
                optimalTile = alignedM;
            }
        }
        if (optimalTile == 0) {
            flagCompactRar = false;
            innerPadded = standardInnerPadded;
            queueSlots = 1;
        }
    }

    if (flagCompactRa) {
        queueSlots = 1;
        const uint64_t gatherByteReq = AlignTo(innerDim * static_cast<uint64_t>(sizeof(uint32_t)), BYTES_PER_BLOCK);
        uint32_t fallbackTiledTile = 0;
        if (gatherByteReq <= std::numeric_limits<uint32_t>::max()) {
            fallbackTiledTile = PickRaTile(ubCapacity, workUnits, static_cast<uint32_t>(innerDim),
                                           innerPadded, elemBytes, static_cast<uint32_t>(gatherByteReq), activeCores);
        }
        constexpr uint32_t phaseQueueDepth = 2;
        const uint32_t curPhaseRows = raPhaseBase * RA_PHASE_SLAB_UNIT;
        uint32_t pTile = PickRaPhaseTile(ubCapacity, workUnits, static_cast<uint32_t>(innerDim),
                                         innerPadded, elemBytes, phaseQueueDepth, activeCores, curPhaseRows, false);
        uint32_t pCores = activeCores;
        uint64_t pUnits = workUnits;
        bool pEligible = ValidateRaPhaseFeasibility(length0, static_cast<uint32_t>(innerDim), curPhaseRows, pCores, pTile);

        if (pEligible && !flagRaPhaseSplit) {
            const uint32_t cappedCores = ScalePhaseCoreLimit(activeCores, devCores);
            if (cappedCores < activeCores) {
                const uint64_t cappedUnits = DivCeil(length0, cappedCores);
                const bool needsCheckpoints = EvaluateCheckpointRequirement(length0, curPhaseRows, cappedCores);
                uint32_t cappedTile = PickRaPhaseTile(ubCapacity, cappedUnits, static_cast<uint32_t>(innerDim),
                                                      innerPadded, elemBytes, phaseQueueDepth, cappedCores,
                                                      curPhaseRows, needsCheckpoints);
                uint32_t tunedTile = FineTuneAlignedTile(cappedTile, length0, static_cast<uint32_t>(innerDim),
                                                         curPhaseRows, cappedCores);
                if (tunedTile > 0) {
                    tunedTile = OptimizeDmaAlignmentTile(tunedTile, length0, static_cast<uint32_t>(innerDim),
                                                         curPhaseRows, cappedCores);
                    uint64_t dummyLow = 0, dummyHigh = 0, dummyMaxG = 0;
                    if (!AnalyzePhaseGrouping(length0, curPhaseRows, cappedCores, dummyLow, dummyHigh, dummyMaxG)) {
                        return ge::GRAPH_FAILED;
                    }
                    pCores = cappedCores;
                    pUnits = dummyHigh;
                    pTile = tunedTile;
                    flagRaPhaseSplit = true;
                    if (!CheckRaPhaseAlignment(length0, static_cast<uint32_t>(innerDim), curPhaseRows, cappedCores,
                                               tunedTile, &raPhaseSegments, &raPhaseCheckpoints)) {
                        return ge::GRAPH_FAILED;
                    }
                } else {
                    if (!ValidateRaPhaseFeasibility(length0, static_cast<uint32_t>(innerDim), curPhaseRows, cappedCores, cappedTile)) {
                        cappedTile = SelectSafeRaTileFallback(cappedTile, length0, static_cast<uint32_t>(innerDim),
                                                              elemBytes, curPhaseRows, cappedCores);
                    }
                    if (ValidateRaPhaseFeasibility(length0, static_cast<uint32_t>(innerDim), curPhaseRows, cappedCores, cappedTile)) {
                        pCores = cappedCores;
                        pUnits = cappedUnits;
                        pTile = cappedTile;
                    }
                }
            }
        }
        const uint64_t tiledPasses = (fallbackTiledTile == 0) ? 0 : DivCeil(workUnits, fallbackTiledTile);
        const bool tiledEligible = (fallbackTiledTile >= raPhaseBase * COMPACT_RA_MIN_GROUPS) &&
                                   (tiledPasses >= RA_STAGGER_MIN_ROUNDS);

        if (pEligible) {
            activeCores = pCores;
            workUnits = pUnits;
            modeCompactRa = RA_MODE_PHASE;
            raPhaseRows = curPhaseRows;
            if (raPhaseSegments == 0) raPhaseSegments = 1;
            queueSlots = phaseQueueDepth;
            extraScratchBytes = 0;
            optimalTile = pTile;
            uint32_t waveSegs = 0, waveChks = 0;
            if (FLAG_ENABLE_RA_WAVE && flagRaPhaseSplit && isStreaming && queueSlots == 2 &&
                ValidatePhaseWaveCompatibility(length0, static_cast<uint32_t>(innerDim), raPhaseRows,
                                               activeCores, optimalTile, &waveSegs, &waveChks) &&
                waveSegs == raPhaseSegments) {
                raPhaseSchedule = RA_SCHEDULE_WAVE;
                raPhaseCheckpoints = waveChks;
            }
        } else if (tiledEligible) {
            modeCompactRa = RA_MODE_TILED;
            extraScratchBytes = static_cast<uint32_t>(gatherByteReq);
            optimalTile = fallbackTiledTile;
        } else {
            flagCompactRa = false;
            queueSlots = 1;
            extraScratchBytes = 0;
            optimalTile = 0;
        }
    }

    if (flagCompactAra) {
        innerPadded = static_cast<uint32_t>(innerDim);
        queueSlots = (length1 > 1) ? 2 : 1;
        uint64_t tmp64 = 0;
        if (MultiplyOverflow(innerDim, sizeof(uint32_t), tmp64)) return ge::GRAPH_FAILED;
        tmp64 = AlignTo(tmp64, BYTES_PER_BLOCK);
        if (tmp64 > std::numeric_limits<uint32_t>::max()) return ge::GRAPH_FAILED;
        extraScratchBytes = static_cast<uint32_t>(tmp64);

        const uint32_t araPhaseRows = BYTES_PER_BLOCK / elemBytes /
            GreatestCommonDivisor(static_cast<uint32_t>(innerDim), BYTES_PER_BLOCK / elemBytes);
        if (elemBytes == 2 && workUnits >= araPhaseRows * COMPACT_RA_MIN_GROUPS) {
            optimalTile = PickAraPhaseTile(ubCapacity, workUnits, innerPadded, elemBytes, queueSlots, araPhaseRows);
            flagPhaseAra = (optimalTile >= araPhaseRows);
            if (flagPhaseAra) extraScratchBytes = 0;
        }
        if (!flagPhaseAra) {
            optimalTile = PickCompactTile(ubCapacity, workUnits, innerPadded, elemBytes, queueSlots, extraScratchBytes);
        }
        if (optimalTile < raPhaseBase * COMPACT_RA_MIN_GROUPS) {
            flagCompactAra = false;
            flagPhaseAra = false;
            innerPadded = standardInnerPadded;
            queueSlots = 1;
            extraScratchBytes = 0;
            optimalTile = 0;
        }
    }

    if (optimalTile == 0) {
        optimalTile = PickDenseTile(ubCapacity, workUnits, innerPadded, elemBytes, queueSlots, redPattern, extraScratchBytes);
    }

    if (!flagCompactAr && !flagCompactRar && !flagCompactRa && !flagCompactAra &&
        optimalTile > 0 && chosenTopology != SCHEME_FALLBACK) {
        const bool allowPipeline = (chosenTopology == SCHEME_AR || chosenTopology == SCHEME_ARA || chosenTopology == SCHEME_RAR);
        const bool multiStep = allowPipeline && (workUnits > optimalTile || (chosenTopology == SCHEME_RAR && length0 > 1));
        if (multiStep) {
            uint32_t dbExtra = 0;
            const uint32_t dbTile = PickDenseTile(ubCapacity, workUnits, innerPadded, elemBytes, 2, redPattern, dbExtra);
            if (dbTile > 0) {
                queueSlots = 2;
                optimalTile = dbTile;
                extraScratchBytes = dbExtra;
            }
        }
    }

    if (optimalTile == 0 && chosenTopology != SCHEME_FALLBACK) {
        chosenTopology = SCHEME_FALLBACK;
        flagCompactAr = false;
        flagCompactRar = false;
        flagCompactRa = false;
        modeCompactRa = 0;
        raPhaseRows = 0;
        flagRaPhaseSplit = false;
        raPhaseSegments = 0;
        raPhaseCheckpoints = 0;
        raPhaseSchedule = 0;
        flagCompactAra = false;
        flagPhaseAra = false;
        activeCores = 1;
        innerDim = shapeVec[static_cast<size_t>(dims - 1)];
        workUnits = 1;
        redPattern = AscendC::ReducePattern::AR;
        innerPadded64 = AlignTo(innerDim, alignUnit);
        if (innerPadded64 == 0 || innerPadded64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        innerPadded = static_cast<uint32_t>(innerPadded64);
        optimalTile = PickDenseTile(ubCapacity, workUnits, innerPadded, elemBytes, 1, redPattern, extraScratchBytes);
        queueSlots = 1;
    }
    if (optimalTile == 0) return ge::GRAPH_FAILED;

    uint32_t raStaggerCount = 0;
    if (chosenTopology == SCHEME_RA && dType == ge::DT_FLOAT &&
        modeCompactRa != RA_MODE_PHASE && queueSlots == 1 &&
        isStreaming && CheckStaggerRaFeasible(length0, activeCores, optimalTile)) {
        raStaggerCount = RA_STAGGER_PARTITIONS;
    }

    const bool singlePassStream = flagPhaseAra || flagCompactRar || (modeCompactRa == RA_MODE_PHASE);
    const bool arStreamingActive = flagCompactAr && cacheL2Capacity > 0 && inputByteSize >= cacheL2Capacity / 4U;
    const bool flagBypassL2 = (singlePassStream && isStreaming) || arStreamingActive;

    uint32_t rarSchedule = 0;
    if (flagCompactRar && queueSlots == 2 && flagBypassL2 &&
        CheckWaveRarFeasible(length1, innerDim, elemBytes, activeCores, optimalTile, rarRowQuantum)) {
        rarSchedule = RAR_SCHEDULE_WAVE;
    }

    const bool flagArDirect = flagCompactAr && dType == ge::DT_BF16 &&
        innerDim <= std::numeric_limits<uint32_t>::max() && (innerDim % 8U) != 0 &&
        compactPackStride >= 64U;

    SquareSumV1TilingData tilingPlan;
    tilingPlan.set_shape(shapeVec.data());
    tilingPlan.set_stride(strideVec.data());
    tilingPlan.set_inputSize(totalInputs);
    tilingPlan.set_outputSize(totalOutputs);
    tilingPlan.set_dim0(length0);
    tilingPlan.set_dim1(length1);
    tilingPlan.set_dim2(length2);
    tilingPlan.set_reduceSize(totalReduces);
    tilingPlan.set_topology(chosenTopology);
    tilingPlan.set_rank(static_cast<uint32_t>(dims));
    tilingPlan.set_axisMask(maskBits);
    tilingPlan.set_blockDim(activeCores);
    tilingPlan.set_tilePrimary(optimalTile);
    tilingPlan.set_alignedInner(innerPadded);
    tilingPlan.set_sharedTmpBytes(extraScratchBytes);
    tilingPlan.set_elementSize(elemBytes);
    tilingPlan.set_inputBufferNum(queueSlots);
    tilingPlan.set_raStaggerGroups(raStaggerCount);
    tilingPlan.set_compactAr(flagCompactAr ? 1U : 0U);
    tilingPlan.set_compactRar(flagCompactRar ? 1U : 0U);
    tilingPlan.set_compactAra(flagPhaseAra ? 2U : (flagCompactAra ? 1U : 0U));
    tilingPlan.set_bypassL2(flagBypassL2 ? 1U : 0U);
    tilingPlan.set_rarScheduleMode(rarSchedule);
    tilingPlan.set_compactRa(modeCompactRa);
    tilingPlan.set_compactRaPhaseRows(raPhaseRows);
    tilingPlan.set_compactRaPhaseAlignedSplit(flagRaPhaseSplit ? 1U : 0U);
    tilingPlan.set_compactRaPhaseSegments(raPhaseSegments);
    tilingPlan.set_compactRaPhaseCheckpointGroups(raPhaseCheckpoints);
    tilingPlan.set_compactRaPhaseScheduleMode(raPhaseSchedule);

    size_t* workspaceLens = context->GetWorkspaceSizes(1);
    auto* rawBuffer = context->GetRawTilingData();
    if (workspaceLens == nullptr || rawBuffer == nullptr || rawBuffer->GetData() == nullptr ||
        rawBuffer->GetCapacity() < tilingPlan.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }

    if (chosenTopology == SCHEME_RA) {
        const uint64_t spanStride = AlignTo(innerPadded * sizeof(float), 512);
        const uint64_t sysWork = devPlatform.GetLibApiWorkSpaceSize();
        uint64_t usrWork = 0;
        if (MultiplyOverflow(spanStride, static_cast<uint64_t>(activeCores), usrWork) ||
            usrWork > std::numeric_limits<size_t>::max() - sysWork) {
            return ge::GRAPH_FAILED;
        }
        tilingPlan.set_workspaceStride(spanStride);
        workspaceLens[0] = static_cast<size_t>(sysWork + usrWork);
        context->SetTilingKey(GET_TPL_TILING_KEY(TILING_KEY_SYNC));
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        tilingPlan.set_workspaceStride(0);
        workspaceLens[0] = 0;
        context->SetTilingKey(GET_TPL_TILING_KEY(flagArDirect ? TILING_KEY_DIRECT_AR : 0));
    }

    context->SetBlockDim(activeCores);
    tilingPlan.SaveToBuffer(rawBuffer->GetData(), rawBuffer->GetCapacity());
    rawBuffer->SetDataSize(tilingPlan.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

// -----------------------------------------------------------------------------
// 算子注册与原型定义
// -----------------------------------------------------------------------------
namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::DeriveShape).SetInferDataType(ge::DeriveType);
        this->AICore().SetTiling(optiling::BuildTiling).AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
} // namespace ops
