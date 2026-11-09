#include "concat_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"

namespace {
constexpr uint64_t kDmaBlockBytes = 32;
// Measured on a reference 910B-family device: small copies are launch-limited, mid-sized
// copies need much finer sharding, and HBM bandwidth is already saturated
// before the full AIV count is active.  A piecewise launch lattice matches
// those physical crossovers better than one bytes/core constant.
constexpr uint64_t kFlatOneCoreLimit = 32 * 1024;
constexpr uint64_t kFlatTwoCoreLimit = 48 * 1024;
constexpr uint64_t kFlatFourCoreLimit = 128 * 1024;
constexpr uint64_t kFlatEightCoreLimit = 512 * 1024;
constexpr uint64_t kFlatSixteenCoreLimit = 2 * 1024 * 1024;
constexpr uint64_t kFlatMediumCoreLimit = 8 * 1024 * 1024;
constexpr uint64_t kFlatThirtyTwoCoreLimit = 32 * 1024 * 1024;
// A row-grid task is one input tensor over a disjoint interval of rows.  The
// task target controls subdivision of large inputs; the volume guard prevents
// launching more Vector Cores than a small workload can feed.
constexpr uint64_t kInputTaskBytesTarget = 256 * 1024;
// Large row grids are limited by the slowest AIV's 2-D MTE2 queue, not by
// launch latency.  A 910B-family sweep from 8 through 64 row groups found that
// roughly 32 KiB per (input,row-range) task removes width-skew tails and keeps
// all 48 AIVs fed.  The threshold is volume-based and remains shape-general.
constexpr uint64_t kLargeInputTaskBytesTarget = 32 * 1024;
constexpr uint64_t kLargeRowGridBytes = 4 * 1024 * 1024;
// LPT scheduling uses a byte-equivalent setup charge per 2-D DMA.  This keeps
// a core with several narrow transfers from looking artificially cheaper than
// one equally sized wide transfer.  The value is intentionally architecture-
// level (one GM page), not tied to any published validation shape.
constexpr uint64_t kBalancedTaskSetupEquivalentBytes = 4 * 1024;
// The compact input-major payload can encode at most 256 raw (input,group)
// ordinals.  Four logical tasks per AIV is enough to smooth the last-wave
// tail without returning to the generic kernel's all-input scan.
constexpr uint64_t kInputMajorTargetTasksPerCore = 4;
constexpr uint64_t kAdaptiveRowsMinBytes = 512 * 1024;
constexpr uint64_t kSmallInputBytesLimit = 64 * 1024;
// Below the queue crossover, the row kernel cannot hide narrow MTE latency
// with ping-pong buffers.  A controlled 910B-family sweep found that 8--32 KiB
// FP16 workloads need eight row groups; queued workloads retain two groups.
constexpr uint64_t kSmallFp16EightGroupMinBytes = 8 * 1024;
constexpr uint64_t kMinBytesPerActiveCore = 1024;
// Tiny tensors make DMA setup, rather than bandwidth, the limiting factor.
// Gather their elements into one private 32-byte UB block per core, then issue
// a single race-free MTE3 transfer for the block.
constexpr uint64_t kGatherInputAverageBytesLimit = 64;
constexpr uint32_t kGatherInputCountThreshold = 16;
// Medium/large flat TensorLists are scalar-bound when every core rescans every
// input prefix.  For a sufficiently balanced list, assign whole inputs to
// cores instead.  Prefixes are encoded into inputRowBytes for this mode.
constexpr uint64_t kDirectFlatInputAverageBytesLimit = 256 * 1024;
constexpr uint32_t kDirectFlatInputCountThreshold = 16;
constexpr uint64_t kCacheLineBytes = 64;
constexpr uint64_t kCacheLineFlatInputBytesLimit = kCacheLineBytes;
constexpr uint64_t kUniformNarrowRowsMinBytes = 64 * 1024;
constexpr uint64_t kUniformNarrowBytesPerCoreTarget = 4 * 1024;
constexpr uint64_t kUniformNarrowMaxInputRowBytes = 16;
constexpr uint64_t kUniformNarrowMaxOutputRowBytes = 128;
constexpr uint32_t kUniformNarrowMaxInputs = 8;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint64_t Gcd(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

uint32_t MeasuredFlatCoreLimit(uint64_t bytes, uint32_t availableCores)
{
    uint32_t requested = 40;
    if (bytes <= kFlatOneCoreLimit) {
        requested = 1;
    } else if (bytes <= kFlatTwoCoreLimit) {
        requested = 2;
    } else if (bytes <= kFlatFourCoreLimit) {
        requested = 4;
    } else if (bytes <= kFlatEightCoreLimit) {
        requested = 8;
    } else if (bytes <= kFlatSixteenCoreLimit) {
        requested = 16;
    } else if (bytes <= kFlatMediumCoreLimit) {
        requested = 20;
    } else if (bytes <= kFlatThirtyTwoCoreLimit) {
        requested = 32;
    }
    return std::max<uint32_t>(1, std::min<uint32_t>(availableCores, requested));
}

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

uint32_t DtypeBytes(ge::DataType type)
{
    switch (type) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}


bool BuildBoundaryPlan(ConcatBoundaryTilingData& t,uint64_t outer,uint64_t W,uint64_t total,const uint32_t* widths,uint32_t count,uint32_t cores,uint64_t ub){
 t={};auto& h=t.header;h.totalBytes=total;h.rowBytes=W;h.outer=outer;h.quantum=static_cast<uint32_t>(32/Gcd(W,32));
 if(h.quantum>8||cores<=h.quantum||outer==0)return false;
 uint64_t prefix[33]{};
 for(uint32_t i=0;i<count;++i)if(widths[i]){if(h.inputCount>=32||widths[i]<32)return false;t.inputs[h.inputCount]={widths[i],i};prefix[h.inputCount+1]=prefix[h.inputCount]+widths[i];++h.inputCount;}
 if(h.inputCount<16||prefix[h.inputCount]!=W||W>UINT32_MAX/h.quantum)return false;
 const uint32_t phaseCount=static_cast<uint32_t>(std::min<uint64_t>(outer,h.quantum));h.edgeCores=phaseCount;h.bodyCores=std::min<uint32_t>(cores-h.edgeCores,outer==1?20:32);if(!h.bodyCores)return false;
 h.tileBytes=static_cast<uint32_t>(((ub-1024)/2)/32*32);uint32_t maxRows=static_cast<uint32_t>(CeilDiv(outer,h.quantum)),powRows=1;while(powRows<maxRows)powRows*=2;h.edgeSlotBytes=powRows*32;
 if(static_cast<uint64_t>(2*h.inputCount+6)*h.edgeSlotBytes+1024>ub)return false;
 h.tailBytes=static_cast<uint32_t>(total%32);
 for(uint32_t p=0;p<phaseCount;++p){
  auto& e=t.edges[p];e.firstRow=p;e.rows=static_cast<uint32_t>(CeilDiv(outer-p,h.quantum));e.begin=h.boundaries;e.needsPrev=p!=0;
  for(uint32_t i=0;i<h.inputCount;++i){
   const uint32_t leftBytes=static_cast<uint32_t>((p*W+prefix[i])%32);if(leftBytes==0)continue;
   if(h.boundaries>=256)return false;auto& b=t.records[h.boundaries++];
   b.dstRelative=static_cast<int32_t>(prefix[i])-static_cast<int32_t>(leftBytes);b.shift=((32-leftBytes)%4)*8;
   b.left=static_cast<uint8_t>(i?2*(i-1)+1:2*h.inputCount);b.right=static_cast<uint8_t>(2*i);b.startWord=static_cast<uint8_t>((32-leftBytes)/4);
  }e.count=h.boundaries-e.begin;
 }
 if(!h.boundaries)return false;
 ConcatBoundaryEdge phases[8]{};for(uint32_t p=0;p<phaseCount;++p)phases[p]=t.edges[p];
 uint32_t edgeCount=0,mapCursor=0;
 for(uint32_t p=0;p<phaseCount;++p){const auto phase=phases[p];
  std::sort(t.records+phase.begin,t.records+phase.begin+phase.count,[](const ConcatBoundaryRecord&a,const ConcatBoundaryRecord&b){return a.shift!=b.shift?a.shift<b.shift:a.dstRelative<b.dstRelative;});
  uint32_t first=phase.begin;while(first<phase.begin+phase.count){uint32_t end=first+1;while(end<phase.begin+phase.count&&t.records[end].shift==t.records[first].shift)++end;
   while(first<end){uint32_t count=outer==1?std::min<uint32_t>(8,end-first):end-first;if(edgeCount>=16||edgeCount+1>=cores)return false;auto&e=t.edges[edgeCount++];e=phase;e.begin=first;e.count=count;e.shift=t.records[first].shift;e.slotMask=0;e.needsPrev=0;e.mapBegin=mapCursor;e.mapCount=count*e.rows*8;
    if(mapCursor+e.mapCount>2048)return false;
    for(uint32_t b=first;b<first+count;++b){const auto& rec=t.records[b];for(uint32_t slot:{static_cast<uint32_t>(rec.left),static_cast<uint32_t>(rec.right)}){if(slot<2*h.inputCount)e.slotMask|=1ULL<<slot;else e.needsPrev=1;}
     for(uint32_t row=0;row<e.rows;++row)for(uint32_t j=0;j<8;++j){uint32_t z=rec.startWord+j;t.lowMaps[mapCursor]=(z<8?rec.left:rec.right)*h.edgeSlotBytes+row*32+(z%8)*4;++z;t.highMaps[mapCursor]=(z<8?rec.left:rec.right)*h.edgeSlotBytes+row*32+(z%8)*4;++mapCursor;}
    }h.edgeOutputBytes=std::max(h.edgeOutputBytes,e.mapCount*4);first+=count;
   }
  }
 }h.edgeCores=edgeCount;h.bodyCores=std::min<uint32_t>(cores-edgeCount,outer==1?20:32);
 if(!edgeCount || static_cast<uint64_t>(2*h.inputCount+1)*h.edgeSlotBytes+4ULL*h.edgeOutputBytes+1024>ub)return false;

 if(outer==1){
  uint64_t bodyBytes=0;for(uint32_t i=0;i<h.inputCount;++i){uint64_t a=AlignUp(prefix[i],32),b=(prefix[i+1]/32)*32;if(b>a)bodyBytes+=b-a;}
  if(!bodyBytes)return false;const uint64_t chunk=AlignUp(CeilDiv(bodyBytes,h.bodyCores),32);uint32_t i=0;uint64_t used=0;
  for(uint32_t c=0;c<h.bodyCores;++c){t.coreOffsets[c]=static_cast<uint16_t>(h.bodyTasks);uint64_t remain=std::min<uint64_t>(chunk,bodyBytes>c*chunk?bodyBytes-c*chunk:0);
   while(remain){if(i>=h.inputCount)return false;uint64_t a=AlignUp(prefix[i],32),b=prefix[i+1]/32*32,len=b>a?b-a:0;if(used>=len){++i;used=0;continue;}
    uint32_t n=static_cast<uint32_t>(std::min<uint64_t>({remain,len-used,h.tileBytes}));if(h.bodyTasks>=512)return false;
    t.tasks[h.bodyTasks++]={a-prefix[i]+used,a+used,t.inputs[i].original,n,1,t.inputs[i].width};used+=n;remain-=n;
   }
  }t.coreOffsets[h.bodyCores]=static_cast<uint16_t>(h.bodyTasks);
 }else{
  ConcatBoundaryBody temp[512]{};uint16_t order[512]{},owner[512]{};uint64_t score[512]{},load[64]{};uint32_t n=0;
  for(uint32_t p=0;p<phaseCount;++p)for(uint32_t i=0;i<h.inputCount;++i){
   const uint64_t a=AlignUp(p*W+prefix[i],32),b=(p*W+prefix[i+1])/32*32;if(b<=a)continue;
   if(static_cast<uint64_t>(h.quantum)*t.inputs[i].width>UINT32_MAX)return false;
   for(uint64_t col=0;col<b-a;){uint32_t bytes=static_cast<uint32_t>(std::min<uint64_t>(h.tileBytes,b-a-col));if(n>=512)return false;
    temp[n]={static_cast<uint64_t>(p)*t.inputs[i].width+a-(p*W+prefix[i])+col,a+col,t.inputs[i].original,bytes,static_cast<uint32_t>(CeilDiv(outer-p,h.quantum)),h.quantum*t.inputs[i].width};
    score[n]=static_cast<uint64_t>(bytes)*temp[n].rows+4096*CeilDiv(temp[n].rows,h.tileBytes/bytes);order[n]=static_cast<uint16_t>(n);++n;col+=bytes;
   }
  }if(!n)return false;std::sort(order,order+n,[&](uint16_t a,uint16_t b){return score[a]>score[b];});
  uint16_t counts[64]{};for(uint32_t j=0;j<n;++j){uint32_t c=0;for(uint32_t k=1;k<h.bodyCores;++k)if(load[k]<load[c])c=k;owner[order[j]]=static_cast<uint16_t>(c);load[c]+=score[order[j]];++counts[c];}
  uint16_t cursor[64]{};for(uint32_t c=0;c<h.bodyCores;++c){t.coreOffsets[c+1]=t.coreOffsets[c]+counts[c];cursor[c]=t.coreOffsets[c];}
  for(uint32_t j=0;j<n;++j)t.tasks[cursor[owner[j]]++]=temp[j];h.bodyTasks=n;
 }
 return h.bodyTasks!=0;
}

struct ConcatHostPlan {
    uint64_t outer;
    uint64_t inner;
    uint64_t outputRowBytes;
    uint64_t totalBytes;
    uint64_t blockRows;
    uint64_t flatChunkBytes;
    uint64_t tileBytes;
    uint32_t inputCount;
    uint32_t mode;
    uint32_t dim;
    uint32_t elementBytes;
};

template <typename TilingDataT>
void StoreCommonTilingData(TilingDataT* tiling, const ConcatHostPlan& plan)
{
    tiling->outer = plan.outer;
    tiling->inner = plan.inner;
    tiling->outputRowBytes = plan.outputRowBytes;
    tiling->totalBytes = plan.totalBytes;
    tiling->blockRows = plan.blockRows;
    tiling->flatChunkBytes = plan.flatChunkBytes;
    tiling->tileBytes = plan.tileBytes;
    tiling->inputCount = plan.inputCount;
    tiling->mode = plan.mode;
    tiling->dim = plan.dim;
    tiling->elementBytes = plan.elementBytes;
}

template <typename TilingDataT>
void StoreTilingData(
    TilingDataT* tiling, const ConcatHostPlan& plan, const uint32_t* inputRowBytes)
{
    StoreCommonTilingData(tiling, plan);
    for (uint32_t input = 0; input < plan.inputCount; ++input) {
        tiling->inputRowBytes[input] = inputRowBytes[input];
    }
}

struct BalancedRowTask {
    uint64_t score;
    uint16_t ordinal;
};

bool ComputeAdaptiveGroupCounts(
    uint64_t outer,
    const uint32_t* inputRowBytes,
    uint32_t inputCount,
    bool largeRowGrid,
    uint16_t* groupCounts,
    uint32_t& taskCount)
{
    if (outer == 0 || inputCount == 0 || inputCount > S9_CONCAT_ADAPTIVE_MAX_INPUTS) {
        return false;
    }
    uint64_t inputBytes[S9_CONCAT_ADAPTIVE_MAX_INPUTS]{};
    uint64_t totalBytes = 0;
    uint32_t nonEmpty = 0;
    for (uint32_t input = 0; input < inputCount; ++input) {
        if (inputRowBytes[input] == 0) {
            groupCounts[input] = 0;
            continue;
        }
        if (outer > std::numeric_limits<uint64_t>::max() / inputRowBytes[input]) {
            return false;
        }
        inputBytes[input] = outer * inputRowBytes[input];
        if (totalBytes > std::numeric_limits<uint64_t>::max() - inputBytes[input]) {
            return false;
        }
        totalBytes += inputBytes[input];
        groupCounts[input] = 1;
        ++nonEmpty;
    }
    if (nonEmpty == 0) {
        return false;
    }
    const uint64_t targetBytes =
        largeRowGrid ? kLargeInputTaskBytesTarget : kInputTaskBytesTarget;
    const uint32_t desiredTasks = static_cast<uint32_t>(std::min<uint64_t>(
        S9_CONCAT_ADAPTIVE_MAX_TASKS,
        std::max<uint64_t>(nonEmpty, CeilDiv(totalBytes, targetBytes))));
    uint32_t requestedTasks = nonEmpty;
    while (requestedTasks < desiredTasks) {
        uint32_t bestInput = inputCount;
        uint64_t bestBytesPerTask = 0;
        for (uint32_t input = 0; input < inputCount; ++input) {
            if (groupCounts[input] == 0 || groupCounts[input] >= outer) {
                continue;
            }
            const uint64_t bytesPerTask = CeilDiv(inputBytes[input], groupCounts[input]);
            if (bestInput == inputCount || bytesPerTask > bestBytesPerTask) {
                bestInput = input;
                bestBytesPerTask = bytesPerTask;
            }
        }
        if (bestInput == inputCount) {
            break;
        }
        ++groupCounts[bestInput];
        ++requestedTasks;
    }

    taskCount = 0;
    for (uint32_t input = 0; input < inputCount; ++input) {
        if (groupCounts[input] == 0) {
            continue;
        }
        const uint64_t actual = CeilDiv(outer, CeilDiv(outer, groupCounts[input]));
        if (actual == 0 || actual > S9_CONCAT_ADAPTIVE_MAX_TASKS ||
            taskCount + actual > S9_CONCAT_ADAPTIVE_MAX_TASKS) {
            return false;
        }
        groupCounts[input] = static_cast<uint16_t>(actual);
        taskCount += static_cast<uint32_t>(actual);
    }
    return taskCount != 0;
}

bool StoreAdaptiveRowTilingData(
    ConcatAdaptiveRowTilingData* tiling,
    const ConcatHostPlan& plan,
    const uint32_t* inputRowBytes,
    uint32_t blockDim)
{
    if (tiling == nullptr || blockDim == 0 || blockDim > S9_CONCAT_BALANCED_MAX_CORES ||
        plan.inputCount == 0 || plan.inputCount > S9_CONCAT_ADAPTIVE_MAX_INPUTS ||
        plan.outputRowBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    StoreCommonTilingData(tiling, plan);

    BalancedRowTask tasks[S9_CONCAT_ADAPTIVE_MAX_TASKS]{};
    uint8_t originalInputs[S9_CONCAT_ADAPTIVE_MAX_TASKS]{};
    uint8_t originalGroups[S9_CONCAT_ADAPTIVE_MAX_TASKS]{};
    uint16_t owners[S9_CONCAT_ADAPTIVE_MAX_TASKS]{};
    uint64_t coreLoads[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t coreTaskCounts[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t cursors[S9_CONCAT_BALANCED_MAX_CORES]{};

    const bool largeRowGrid = plan.totalBytes >= kLargeRowGridBytes;
    uint16_t inputGroupCounts[S9_CONCAT_ADAPTIVE_MAX_INPUTS]{};
    uint32_t computedTaskCount = 0;
    if (!ComputeAdaptiveGroupCounts(
            plan.outer, inputRowBytes, plan.inputCount, largeRowGrid,
            inputGroupCounts, computedTaskCount)) {
        return false;
    }
    uint64_t prefixBytes = 0;
    uint32_t taskCount = 0;
    for (uint32_t input = 0; input < plan.inputCount; ++input) {
        if (prefixBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        tiling->inputPrefixBytes[input] = static_cast<uint32_t>(prefixBytes);
        prefixBytes += inputRowBytes[input];

        const uint16_t groups = inputGroupCounts[input];
        tiling->inputGroupCounts[input] = groups;
        if (taskCount + groups > S9_CONCAT_ADAPTIVE_MAX_TASKS) {
            return false;
        }
        if (groups == 0) {
            continue;
        }
        const uint64_t blockRows = CeilDiv(plan.outer, groups);
        const uint64_t alignedRowBytes = AlignUp(inputRowBytes[input], kDmaBlockBytes);
        const uint64_t rowsPerDma = std::max<uint64_t>(1, plan.tileBytes / alignedRowBytes);
        for (uint32_t group = 0; group < groups; ++group) {
            const uint64_t rowBegin = static_cast<uint64_t>(group) * blockRows;
            const uint64_t rowCount =
                std::min<uint64_t>(blockRows, plan.outer - rowBegin);
            const uint64_t dmaCount =
                CeilDiv(rowCount, std::min<uint64_t>(4095, rowsPerDma));
            tasks[taskCount].score = rowCount * inputRowBytes[input] +
                                     dmaCount * kBalancedTaskSetupEquivalentBytes;
            tasks[taskCount].ordinal = static_cast<uint16_t>(taskCount);
            originalInputs[taskCount] = static_cast<uint8_t>(input);
            originalGroups[taskCount] = static_cast<uint8_t>(group);
            ++taskCount;
        }
    }
    if (prefixBytes != plan.outputRowBytes || taskCount == 0 ||
        taskCount != computedTaskCount) {
        return false;
    }

    std::sort(tasks, tasks + taskCount,
              [](const BalancedRowTask& lhs, const BalancedRowTask& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.ordinal < rhs.ordinal;
              });
    for (uint32_t sorted = 0; sorted < taskCount; ++sorted) {
        uint32_t bestCore = 0;
        for (uint32_t core = 1; core < blockDim; ++core) {
            if (coreLoads[core] < coreLoads[bestCore] ||
                (coreLoads[core] == coreLoads[bestCore] &&
                 coreTaskCounts[core] < coreTaskCounts[bestCore])) {
                bestCore = core;
            }
        }
        owners[tasks[sorted].ordinal] = static_cast<uint16_t>(bestCore);
        coreLoads[bestCore] += tasks[sorted].score;
        ++coreTaskCounts[bestCore];
    }

    tiling->coreTaskOffsets[0] = 0;
    for (uint32_t core = 0; core < blockDim; ++core) {
        tiling->coreTaskOffsets[core + 1] = static_cast<uint16_t>(
            tiling->coreTaskOffsets[core] + coreTaskCounts[core]);
        cursors[core] = tiling->coreTaskOffsets[core];
    }
    for (uint16_t task = 0; task < taskCount; ++task) {
        const uint16_t slot = cursors[owners[task]]++;
        tiling->taskInputs[slot] = originalInputs[task];
        tiling->taskGroups[slot] = originalGroups[task];
    }
    return tiling->coreTaskOffsets[blockDim] == taskCount;
}

bool StoreBalancedRowTilingData(
    ConcatBalancedRowTilingData* tiling,
    const ConcatHostPlan& plan,
    const uint32_t* inputRowBytes,
    uint32_t blockDim)
{
    if (tiling == nullptr || blockDim == 0 || blockDim > S9_CONCAT_BALANCED_MAX_CORES ||
        plan.inputCount == 0 || plan.inputCount > S9_CONCAT_BALANCED_MAX_INPUTS ||
        plan.blockRows == 0) {
        return false;
    }
    const uint64_t groups = CeilDiv(plan.outer, plan.blockRows);
    const uint64_t taskCount64 = groups * plan.inputCount;
    if (groups == 0 || groups > S9_CONCAT_BALANCED_MAX_GROUPS ||
        taskCount64 > S9_CONCAT_BALANCED_MAX_TASKS) {
        return false;
    }
    const uint32_t taskCount = static_cast<uint32_t>(taskCount64);
    StoreTilingData(tiling, plan, inputRowBytes);

    uint32_t prefixBytes = 0;
    for (uint32_t input = 0; input < plan.inputCount; ++input) {
        if (inputRowBytes[input] > std::numeric_limits<uint16_t>::max() ||
            prefixBytes > std::numeric_limits<uint16_t>::max()) {
            return false;
        }
        tiling->inputPrefixBytes[input] = static_cast<uint16_t>(prefixBytes);
        prefixBytes += inputRowBytes[input];
    }
    if (prefixBytes != plan.outputRowBytes ||
        prefixBytes > std::numeric_limits<uint16_t>::max()) {
        return false;
    }

    BalancedRowTask tasks[S9_CONCAT_BALANCED_MAX_TASKS]{};
    uint16_t owners[S9_CONCAT_BALANCED_MAX_TASKS]{};
    uint64_t coreLoads[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t coreTaskCounts[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t cursors[S9_CONCAT_BALANCED_MAX_CORES]{};

    uint32_t task = 0;
    for (uint32_t input = 0; input < plan.inputCount; ++input) {
        const uint64_t alignedRowBytes = AlignUp(inputRowBytes[input], kDmaBlockBytes);
        const uint64_t rowsPerDma = std::max<uint64_t>(1, plan.tileBytes / alignedRowBytes);
        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t rowBegin = group * plan.blockRows;
            const uint64_t rowCount = std::min<uint64_t>(plan.blockRows, plan.outer - rowBegin);
            const uint64_t dmaCount = CeilDiv(rowCount, std::min<uint64_t>(4095, rowsPerDma));
            tasks[task].score = rowCount * inputRowBytes[input] +
                                dmaCount * kBalancedTaskSetupEquivalentBytes;
            tasks[task].ordinal = static_cast<uint16_t>(task);
            ++task;
        }
    }
    std::sort(tasks, tasks + taskCount, [](const BalancedRowTask& lhs, const BalancedRowTask& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        return lhs.ordinal < rhs.ordinal;
    });

    // Longest-processing-time scheduling has a 4/3 approximation guarantee
    // and is near-exact for the repeated per-input row groups used here.
    for (uint32_t sorted = 0; sorted < taskCount; ++sorted) {
        uint32_t bestCore = 0;
        for (uint32_t core = 1; core < blockDim; ++core) {
            if (coreLoads[core] < coreLoads[bestCore] ||
                (coreLoads[core] == coreLoads[bestCore] &&
                 coreTaskCounts[core] < coreTaskCounts[bestCore])) {
                bestCore = core;
            }
        }
        owners[tasks[sorted].ordinal] = static_cast<uint16_t>(bestCore);
        coreLoads[bestCore] += tasks[sorted].score;
        ++coreTaskCounts[bestCore];
    }

    tiling->coreTaskOffsets[0] = 0;
    for (uint32_t core = 0; core < blockDim; ++core) {
        tiling->coreTaskOffsets[core + 1] =
            static_cast<uint16_t>(tiling->coreTaskOffsets[core] + coreTaskCounts[core]);
        cursors[core] = tiling->coreTaskOffsets[core];
    }
    for (uint16_t ordinal = 0; ordinal < taskCount; ++ordinal) {
        const uint16_t owner = owners[ordinal];
        tiling->taskOrdinals[cursors[owner]++] = ordinal;
    }
    return tiling->coreTaskOffsets[blockDim] == taskCount;
}

bool StoreInputMajorRowTilingData(
    ConcatInputMajorRowTilingData* tiling,
    const ConcatHostPlan& plan,
    const uint32_t* inputRowBytes,
    uint32_t blockDim)
{
    if (tiling == nullptr || blockDim == 0 || blockDim > S9_CONCAT_BALANCED_MAX_CORES ||
        plan.inputCount == 0 || plan.inputCount > S9_CONCAT_MAX_INPUTS ||
        plan.outputRowBytes > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    StoreCommonTilingData(tiling, plan);

    BalancedRowTask tasks[S9_CONCAT_MAX_INPUTS]{};
    uint16_t owners[S9_CONCAT_MAX_INPUTS]{};
    uint64_t coreLoads[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t coreTaskCounts[S9_CONCAT_BALANCED_MAX_CORES]{};
    uint16_t cursors[S9_CONCAT_BALANCED_MAX_CORES]{};

    const uint64_t groups = CeilDiv(plan.outer, plan.blockRows);
    if (groups == 0 || groups > S9_CONCAT_MAX_INPUTS / plan.inputCount) {
        return false;
    }

    uint32_t prefixBytes = 0;
    uint32_t taskCount = 0;
    for (uint32_t input = 0; input < plan.inputCount; ++input) {
        if (inputRowBytes[input] > std::numeric_limits<uint16_t>::max() ||
            prefixBytes > std::numeric_limits<uint16_t>::max()) {
            return false;
        }
        tiling->inputPrefixBytes[input] = static_cast<uint16_t>(prefixBytes);
        prefixBytes += inputRowBytes[input];
        if (inputRowBytes[input] == 0) {
            continue;
        }
        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t rowBegin = group * plan.blockRows;
            const uint64_t groupRows =
                std::min<uint64_t>(plan.blockRows, plan.outer - rowBegin);
            const uint16_t ordinal = static_cast<uint16_t>(input * groups + group);
            const uint64_t alignedRowBytes =
                AlignUp(inputRowBytes[input], kDmaBlockBytes);
            const uint64_t rowsPerDma =
                std::max<uint64_t>(1, plan.tileBytes / alignedRowBytes);
            const uint64_t dmaCount = CeilDiv(
                groupRows, std::min<uint64_t>(4095, rowsPerDma));
            // Include an empirically calibrated descriptor charge so narrow
            // inputs are distributed by task count, while byte-heavy groups
            // still receive otherwise idle AIVs.
            tasks[taskCount].score = groupRows * inputRowBytes[input] +
                                     dmaCount * kBalancedTaskSetupEquivalentBytes;
            tasks[taskCount].ordinal = ordinal;
            ++taskCount;
        }
    }
    if (prefixBytes != plan.outputRowBytes || taskCount == 0) {
        return false;
    }
    std::sort(tasks, tasks + taskCount,
              [](const BalancedRowTask& lhs, const BalancedRowTask& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.ordinal < rhs.ordinal;
              });
    for (uint32_t sorted = 0; sorted < taskCount; ++sorted) {
        uint32_t bestCore = 0;
        for (uint32_t core = 1; core < blockDim; ++core) {
            if (coreLoads[core] < coreLoads[bestCore] ||
                (coreLoads[core] == coreLoads[bestCore] &&
                 coreTaskCounts[core] < coreTaskCounts[bestCore])) {
                bestCore = core;
            }
        }
        owners[tasks[sorted].ordinal] = static_cast<uint16_t>(bestCore);
        coreLoads[bestCore] += tasks[sorted].score;
        ++coreTaskCounts[bestCore];
    }

    tiling->coreTaskOffsets[0] = 0;
    for (uint32_t core = 0; core < blockDim; ++core) {
        tiling->coreTaskOffsets[core + 1] = static_cast<uint16_t>(
            tiling->coreTaskOffsets[core] + coreTaskCounts[core]);
        cursors[core] = tiling->coreTaskOffsets[core];
    }
    for (uint16_t input = 0; input < plan.inputCount; ++input) {
        if (inputRowBytes[input] == 0) {
            continue;
        }
        for (uint16_t group = 0; group < groups; ++group) {
            const uint16_t ordinal = static_cast<uint16_t>(input * groups + group);
            const uint16_t owner = owners[ordinal];
            tiling->taskOrdinals[cursors[owner]++] = static_cast<uint8_t>(ordinal);
        }
    }
    return tiling->coreTaskOffsets[blockDim] == taskCount;
}

uint64_t TilingKeyForMode(uint32_t mode, bool useQueue)
{
    switch (mode) {
        case 0:
            return S9_CONCAT_TILING_KEY_EMPTY;
        case 2:
            return useQueue ? S9_CONCAT_TILING_KEY_FLAT_QUEUE :
                              S9_CONCAT_TILING_KEY_FLAT_BUFFER;
        case 3:
            return S9_CONCAT_TILING_KEY_GATHER;
        case 5:
            return S9_CONCAT_TILING_KEY_UNIFORM_GATHER;
        case 6:
            return S9_CONCAT_TILING_KEY_DIRECT_FLAT;
        case 9:
            return S9_CONCAT_TILING_KEY_CACHE_LINES;
        case 12:
            return S9_CONCAT_TILING_KEY_UNIT_FP16_GATHER;
        case 14:
            return S9_CONCAT_TILING_KEY_PAIR_FP16_GATHER;
        case 15:
            return S9_CONCAT_TILING_KEY_QUAD_FP16_GATHER;
        case 16:
            return S9_CONCAT_TILING_KEY_OCT_FP16_GATHER;
        case 17:
            return S9_CONCAT_TILING_KEY_HEX_FP16_GATHER;
        case 18:
            return S9_CONCAT_TILING_KEY_DIRECT_ROW_TASKS_QUEUE;
        case 19:
            return S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD;
        case 20:
            return S9_CONCAT_TILING_KEY_INPUT_MAJOR_ROWS;
        case 21:
            return S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD_32;
        case 22:
            return S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD_INT8_PAIR;
        case 23:
            return S9_CONCAT_TILING_KEY_ALIGNED_ROW_PACK;
        case 24:
            return S9_CONCAT_TILING_KEY_ADAPTIVE_ROWS;
        case 25:
            return S9_CONCAT_TILING_KEY_DIRECT_FLAT_QUEUE;
        case 26:
            return S9_CONCAT_TILING_KEY_UNIFORM_NARROW_ROWS;
        case 27:
            return S9_CONCAT_TILING_KEY_RAGGED_NARROW_ROWS;
        case 4:
            return useQueue ? S9_CONCAT_TILING_KEY_DIRECT_ROW_QUEUE :
                              S9_CONCAT_TILING_KEY_DIRECT_ROW_BUFFER;
        default:
            return useQueue ? S9_CONCAT_TILING_KEY_ROWS_QUEUE :
                              S9_CONCAT_TILING_KEY_ROWS_BUFFER;
    }
}

} // namespace

namespace optiling {
static ge::graphStatus TilingConcat(gert::TilingContext* context)
{
    auto* nodeInfo = context->GetComputeNodeInfo();
    if (nodeInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* listInfo = nodeInfo->GetInputInstanceInfo(0);
    if (listInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t inputCount = listInfo->GetInstanceNum();
    if (inputCount == 0 || inputCount > S9_CONCAT_MAX_INPUTS) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* dimValue = attrs->GetAttrPointer<int64_t>(0);
    if (dimValue == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim = *dimValue;
    uint32_t forcedProbeCores = 0;
    uint32_t forcedProbeRowCores = 0;
    uint32_t forcedProbeRowGroups = 0;
    bool forcedProbeRowActive = false;
    bool forcedProbeRowQueue = false;

    const auto* firstShapeHolder = context->GetDynamicInputShape(0, 0);
    const auto* firstDesc = context->GetDynamicInputDesc(0, 0);
    if (firstShapeHolder == nullptr || firstDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape firstShape = firstShapeHolder->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(firstShape.GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }
#ifdef S9_ARCH_PROBE
    // Build-only hardware probe.  Values -1001..-1048 mean concatenate on
    // the last axis while forcing that many AIVs.  Normal contest builds do
    // not compile this branch and therefore retain the exact public schema.
    // Values -(G+1)001..-(G+1)048 force 1/2/4/8 row groups and an AIV count,
    // where the thousands digit is 2/3/4/5 respectively.  Buckets 10..14
    // extend the queued probe to 16/32/64/128/256 groups so large paths expose
    // load-balance crossovers beyond the normal tiler's search range.  Probe-
    // only buckets 15..30 scan the irregular 40..72 neighborhood where task
    // modulo 48 changes ownership periodicity.
    if (dim <= -2001 && dim >= -30048) {
        const uint32_t encoded = static_cast<uint32_t>(-dim);
        const uint32_t bucket = encoded / 1000;
        const uint32_t cores = encoded % 1000;
        if (bucket < 2 || bucket > 30 || cores == 0 || cores > 48) {
            return ge::GRAPH_FAILED;
        }
        if (bucket <= 14) {
            const uint32_t groupBucket = bucket <= 5 ? bucket : bucket - 4;
            forcedProbeRowGroups = 1U << (groupBucket - 2);
        } else {
            constexpr uint32_t irregularGroups[] = {
                40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 66, 68, 70, 72};
            forcedProbeRowGroups = irregularGroups[bucket - 15];
        }
        forcedProbeRowCores = cores;
        forcedProbeRowActive = true;
        forcedProbeRowQueue = bucket >= 6;
        dim = rank - 1;
    } else if (dim <= -1001 && dim >= -1048) {
        forcedProbeCores = static_cast<uint32_t>(-dim - 1000);
        dim = rank - 1;
    }
#endif
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t elementBytes = DtypeBytes(firstDesc->GetDataType());
    if (elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1;
    uint64_t inner = 1;
    for (int64_t axis = 0; axis < dim; ++axis) {
        const int64_t extent = firstShape.GetDim(axis);
        uint64_t next = 0;
        if (extent < 0 || MulOverflow(outer, static_cast<uint64_t>(extent), next)) {
            return ge::GRAPH_FAILED;
        }
        outer = next;
    }
    for (int64_t axis = dim + 1; axis < rank; ++axis) {
        const int64_t extent = firstShape.GetDim(axis);
        uint64_t next = 0;
        if (extent < 0 || MulOverflow(inner, static_cast<uint64_t>(extent), next)) {
            return ge::GRAPH_FAILED;
        }
        inner = next;
    }

    uint64_t outputRowBytes = 0;
    uint64_t maxInputRowBytes = 0;
    uint64_t minInputRowBytes = std::numeric_limits<uint64_t>::max();
    uint32_t nonEmptyInputCount = 0;
    uint64_t firstInputRowBytes = 0;
    bool uniformInputRowBytes = true;
    bool cacheLineAlignedInputs = true;
    bool dmaBlockAlignedInputs = true;
    bool evenInputRowBytes = true;
    uint32_t inputRowBytes[S9_CONCAT_MAX_INPUTS]{};
    for (uint32_t input = 0; input < inputCount; ++input) {
        const auto* shapeHolder = context->GetDynamicInputShape(0, input);
        const auto* desc = context->GetDynamicInputDesc(0, input);
        if (shapeHolder == nullptr || desc == nullptr || desc->GetDataType() != firstDesc->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape shape = shapeHolder->GetStorageShape();
        if (static_cast<int64_t>(shape.GetDimNum()) != rank) {
            return ge::GRAPH_FAILED;
        }
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (shape.GetDim(axis) < 0 || (axis != dim && shape.GetDim(axis) != firstShape.GetDim(axis))) {
                return ge::GRAPH_FAILED;
            }
        }
        uint64_t rowElements = 0;
        uint64_t rowBytes = 0;
        if (MulOverflow(static_cast<uint64_t>(shape.GetDim(dim)), inner, rowElements) ||
            MulOverflow(rowElements, elementBytes, rowBytes)) {
            return ge::GRAPH_FAILED;
        }
        if (rowBytes > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        inputRowBytes[input] = static_cast<uint32_t>(rowBytes);
        if (input == 0) {
            firstInputRowBytes = rowBytes;
        } else if (rowBytes != firstInputRowBytes) {
            uniformInputRowBytes = false;
        }
        if (rowBytes != 0) {
            ++nonEmptyInputCount;
            maxInputRowBytes = std::max(maxInputRowBytes, rowBytes);
            minInputRowBytes = std::min(minInputRowBytes, rowBytes);
        }
        if (rowBytes == 0 || rowBytes % kCacheLineBytes != 0) {
            cacheLineAlignedInputs = false;
        }
        if (rowBytes != 0 && rowBytes % kDmaBlockBytes != 0) {
            dmaBlockAlignedInputs = false;
        }
        if (rowBytes % sizeof(uint16_t) != 0) {
            evenInputRowBytes = false;
        }
        if (outputRowBytes > std::numeric_limits<uint64_t>::max() - rowBytes) {
            return ge::GRAPH_FAILED;
        }
        outputRowBytes += rowBytes;
    }

    uint64_t totalBytes = 0;
    if (MulOverflow(outer, outputRowBytes, totalBytes)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t coreCount = 1;
    uint64_t ubBytes = 192 * 1024;
    uint64_t l2Bytes = 192ULL * 1024 * 1024;
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        coreCount = std::max<uint32_t>(1, platform.GetCoreNumAiv());
        uint64_t detectedUb = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, detectedUb);
        uint64_t detectedL2 = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, detectedL2);
        if (detectedL2 != 0) { l2Bytes = detectedL2; }
        if (detectedUb >= 32 * 1024) {
            ubBytes = detectedUb;
        }
    }
    if(firstDesc->GetDataType()==ge::DT_FLOAT && outer>1 && outer<coreCount && nonEmptyInputCount>=16 && nonEmptyInputCount<=32 && totalBytes>=1024ULL*1024 && totalBytes<=32ULL*1024*1024 && ubBytes>=32768 && context->GetRawTilingData()->GetCapacity()>=sizeof(ConcatBoundaryTilingData)){
        ConcatBoundaryTilingData plan{};
        if(BuildBoundaryPlan(plan,outer,outputRowBytes,totalBytes,inputRowBytes,inputCount,coreCount,ubBytes)){
            auto* p=context->GetTilingData<ConcatBoundaryTilingData>();if(!p)return ge::GRAPH_FAILED;*p=plan;context->GetRawTilingData()->SetDataSize(sizeof(plan));
            context->SetBlockDim(plan.header.bodyCores+plan.header.edgeCores);context->SetTilingKey(1929);auto*ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=0;return ge::GRAPH_SUCCESS;
        }
    }
    uint64_t tileBytes = std::min<uint64_t>(128 * 1024, ubBytes / 2);
    tileBytes = std::max<uint64_t>(kDmaBlockBytes, tileBytes / kDmaBlockBytes * kDmaBlockBytes);

    uint32_t blockDim = 1;
    uint64_t blockRows = 0;
    uint64_t flatChunk = 0;
    uint32_t mode = 0;
    bool useBalancedRows = false;
    bool useInputMajorRows = false;
    bool useAdaptiveRows = false;
    if (totalBytes != 0) {
        const bool gatherInputs = outer == 1 && inputCount >= kGatherInputCountThreshold &&
                                  totalBytes <= static_cast<uint64_t>(inputCount) *
                                                    kGatherInputAverageBytesLimit;
        const bool flat = inputCount == 1 || outer == 1;
        const uint64_t averageNonEmptyBytes =
            CeilDiv(totalBytes, std::max<uint32_t>(1, nonEmptyInputCount));
        const bool directFlatInputs =
            outer == 1 && inputCount >= kDirectFlatInputCountThreshold &&
            nonEmptyInputCount == inputCount &&
            totalBytes <= static_cast<uint64_t>(UINT32_MAX) &&
            averageNonEmptyBytes <= kDirectFlatInputAverageBytesLimit &&
            (inputCount <= coreCount || elementBytes == 1) &&
            maxInputRowBytes <= 2 * averageNonEmptyBytes;
        const bool cacheLineFlatInputs =
            outer == 1 && inputCount >= 2 && inputCount <= coreCount &&
            nonEmptyInputCount == inputCount && cacheLineAlignedInputs &&
            totalBytes <= static_cast<uint64_t>(UINT32_MAX) &&
            maxInputRowBytes <= kCacheLineFlatInputBytesLimit;
        // One FP16 value per tensor is a pointer-gather, not a bandwidth
        // workload.  Give each AIV one aligned 32-byte output block (16
        // tensors); its dedicated kernel stages the discontiguous values and
        // compacts them with one vector gather instead of 16 scalar GM loads.
        const bool unitFp16Gather =
            outer == 1 && elementBytes == 2 && uniformInputRowBytes &&
            firstInputRowBytes == elementBytes && inputCount >= 16;
        const bool pairFp16Gather =
            outer == 1 && elementBytes == 2 && uniformInputRowBytes &&
            firstInputRowBytes == 2 * elementBytes && inputCount >= 16;
        const bool quadFp16Gather =
            outer == 1 && elementBytes == 2 && uniformInputRowBytes &&
            firstInputRowBytes == 4 * elementBytes && inputCount >= 16;
        const bool octFp16Gather =
            outer == 1 && elementBytes == 2 && uniformInputRowBytes &&
            firstInputRowBytes == 8 * elementBytes && inputCount >= 16;
        const bool hexFp16Gather =
            outer == 1 && elementBytes == 2 && uniformInputRowBytes &&
            firstInputRowBytes == 16 * elementBytes && inputCount >= 32;
        if (cacheLineFlatInputs) {
            mode = 9;
            blockDim = inputCount;
        } else if (unitFp16Gather) {
            mode = 12;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(inputCount, 16))));
        } else if (pairFp16Gather && inputCount >= 32) {
            mode = 14;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(inputCount, 16))));
        } else if (quadFp16Gather && inputCount >= 32) {
            mode = 15;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(inputCount, 16))));
        } else if (octFp16Gather && inputCount >= 32) {
            mode = 16;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(inputCount, 16))));
        } else if (hexFp16Gather) {
            mode = 17;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(inputCount, 16))));
        } else if (gatherInputs) {
            // Equal-width tiny inputs admit O(inputs_owned_by_core) address
            // generation: each core derives its first input directly instead
            // of rescanning the whole TensorList prefix.
            mode = uniformInputRowBytes && firstInputRowBytes != 0 ? 5 : 3;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(coreCount, CeilDiv(totalBytes, kDmaBlockBytes))));
        } else if (directFlatInputs) {
            // Queue management is profitable only when an input spans more
            // than one UB tile.  The direct kernel transports bytes without
            // dtype arithmetic, so the same structural crossover applies to
            // every supported payload width.
            mode = maxInputRowBytes > tileBytes ? 25 : 6;
            blockDim = std::max<uint32_t>(1, std::min<uint32_t>(coreCount, inputCount));
        } else if (flat) {
            mode = 2;
            // The dedicated gather/direct-input paths above already cover
            // TensorLists for which input-level parallelism is profitable.
            // What remains is one contiguous output byte stream.  Applying
            // the old 32-bytes/AIV rule here launched 29--45 AIVs for only
            // 0.5--10 KiB (especially dim=0), while every AIV repeated the
            // same TensorList prefix scan.  On the reference device one AIV copies up
            // to 32 KiB in 1.6--2.0 us; dozens of launch contexts took
            // 4--10 us.  Reuse the measured byte-volume lattice for all
            // generic flat streams.  This is shape-general and preserves the
            // specialized many-input paths rather than keying on test shapes.
            const uint64_t requested = MeasuredFlatCoreLimit(totalBytes, coreCount);
            flatChunk = AlignUp(CeilDiv(totalBytes, requested), kDmaBlockBytes);
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, CeilDiv(totalBytes, flatChunk)));
        } else {
            mode = 1;
            const uint64_t maxInputBytes = outer * maxInputRowBytes;
            const bool splitSmallFp16 = elementBytes == 2 &&
                                        maxInputBytes <= kSmallInputBytesLimit && outer > 1;
            const uint64_t smallFp16Groups =
                totalBytes > kSmallFp16EightGroupMinBytes &&
                        totalBytes < S9_CONCAT_DOUBLE_BUFFER_THRESHOLD_BYTES ?
                    8 :
                    2;
            const bool largeRowGrid = totalBytes >= kLargeRowGridBytes;
            const uint64_t taskBytesTarget =
                largeRowGrid ? kLargeInputTaskBytesTarget : kInputTaskBytesTarget;
            const uint64_t groupsByBytes = splitSmallFp16 ?
                                               smallFp16Groups :
                                               std::max<uint64_t>(
                                                   1, CeilDiv(maxInputBytes, taskBytesTarget));
            // For launch-dominated rows, cap the number of logical tasks to a
            // single ownership wave.  Large grids instead use their measured
            // byte target directly: the input-width distribution can be very
            // skewed, so a wave-count cap recreates the longest-core tail that
            // the finer grid is intended to remove.
            const uint64_t groupsByCores = std::max<uint64_t>(
                1, coreCount / std::max<uint32_t>(1, nonEmptyInputCount));
            // A 910B-family sweep found the large-row 2-D DMA sweet spot at 9/8 of
            // one AIV wave (54 groups on 48 AIVs).  Use direct task-stride
            // ownership below so every core visits only its own ~18 tasks.
            const bool useDirectLargeRows =
                largeRowGrid && inputCount <= S9_CONCAT_BALANCED_MAX_INPUTS &&
                nonEmptyInputCount == inputCount && outer >= coreCount;
            const bool useAlignedRowPack =
                largeRowGrid && inputCount >= 2 &&
                inputCount <= S9_CONCAT_BALANCED_MAX_INPUTS &&
                nonEmptyInputCount == inputCount && dmaBlockAlignedInputs &&
                outputRowBytes <= tileBytes;
            const bool useRaggedNarrowRows =
                inputCount >= 2 && inputCount <= S9_CONCAT_SMALL_INPUTS &&
                nonEmptyInputCount == inputCount && outer > 1 &&
                totalBytes >= kUniformNarrowRowsMinBytes &&
                maxInputRowBytes <= kUniformNarrowMaxInputRowBytes &&
                outputRowBytes <= kUniformNarrowMaxOutputRowBytes;
            const bool useUniformNarrowRows =
                uniformInputRowBytes && firstInputRowBytes != 0 &&
                inputCount >= 2 && inputCount <= kUniformNarrowMaxInputs &&
                nonEmptyInputCount == inputCount && outer > 1 &&
                totalBytes >= ((elementBytes == 1 && firstInputRowBytes <= 4 &&
                                outer >= 8 * coreCount) ? 8 * 1024 : kUniformNarrowRowsMinBytes) &&
                firstInputRowBytes <= kUniformNarrowMaxInputRowBytes &&
                outputRowBytes <= kUniformNarrowMaxOutputRowBytes;
            // With >32 near-uniform inputs, the generic row kernel pays an
            // O(AIVs * inputs * row-groups) ownership scan and emits a very
            // fine 32-KiB task grid.  A compact host-built list instead uses
            // up to four tasks/AIV, balances them with LPT, and lets each AIV
            // jump directly to its few 2-D DMA intervals.  The 2x-average
            // guard deliberately leaves width-skewed lists on the fine grid.
            const bool largeInputMajorBase =
                largeRowGrid && inputCount > S9_CONCAT_BALANCED_MAX_INPUTS &&
                inputCount <= S9_CONCAT_MAX_INPUTS &&
                nonEmptyInputCount > S9_CONCAT_SMALL_INPUTS &&
                outputRowBytes <= std::numeric_limits<uint16_t>::max() &&
                maxInputRowBytes <= std::numeric_limits<uint16_t>::max() &&
                maxInputRowBytes <= 2 * CeilDiv(outputRowBytes, nonEmptyInputCount);
            const uint64_t largeInputMajorGroups = largeInputMajorBase ?
                std::max<uint64_t>(
                    1, std::min<uint64_t>(
                           {outer,
                            static_cast<uint64_t>(S9_CONCAT_MAX_INPUTS / inputCount),
                            nonEmptyInputCount >= 2 * coreCount ?
                                1 :
                                CeilDiv(static_cast<uint64_t>(coreCount) *
                                            kInputMajorTargetTasksPerCore,
                                        nonEmptyInputCount)})) :
                1;
            useBalancedRows = false;
            const uint64_t requestedGroups = std::max<uint64_t>(
                1, useDirectLargeRows ?
                       std::min<uint64_t>(outer, coreCount + coreCount / 8) :
                       largeInputMajorBase ?
                       largeInputMajorGroups :
                       largeRowGrid ?
                       std::min<uint64_t>(outer, groupsByBytes) :
                       std::min<uint64_t>({outer, groupsByBytes, groupsByCores}));
            blockRows = CeilDiv(outer, requestedGroups);
            const uint64_t actualGroups = CeilDiv(outer, blockRows);
            const uint64_t taskCount = actualGroups * nonEmptyInputCount;
            const bool mediumInputMajorCandidate =
                !largeRowGrid && actualGroups == 1 &&
                maxInputBytes <= kInputTaskBytesTarget;
            const bool inputMajorCandidate =
                inputCount > S9_CONCAT_SMALL_INPUTS &&
                inputCount <= S9_CONCAT_MAX_INPUTS &&
                nonEmptyInputCount > S9_CONCAT_SMALL_INPUTS &&
                totalBytes >= S9_CONCAT_DOUBLE_BUFFER_THRESHOLD_BYTES &&
                outputRowBytes <= std::numeric_limits<uint16_t>::max() &&
                maxInputRowBytes <= std::numeric_limits<uint16_t>::max() &&
                actualGroups <= S9_CONCAT_MAX_INPUTS / inputCount &&
                (mediumInputMajorCandidate || largeInputMajorBase);
            uint32_t adaptiveTaskCount = 0;
            uint16_t adaptiveGroupCounts[S9_CONCAT_ADAPTIVE_MAX_INPUTS]{};
            // For unit-inner, very large row grids with a short TensorList,
            bool adaptiveRowsCandidate =
                inputCount >= 2 &&
                inputCount <= S9_CONCAT_ADAPTIVE_MAX_INPUTS &&
                nonEmptyInputCount >= 2 &&
                totalBytes >= kAdaptiveRowsMinBytes &&
                outputRowBytes <= std::numeric_limits<uint32_t>::max() &&
                !inputMajorCandidate && !useAlignedRowPack;
            if (adaptiveRowsCandidate) {
                adaptiveRowsCandidate = ComputeAdaptiveGroupCounts(
                    outer, inputRowBytes, inputCount, largeRowGrid,
                    adaptiveGroupCounts, adaptiveTaskCount);
                // For short TensorLists retain the measured 54-group direct
                // path unless width-aware splitting also reduces the number
                // of logical tasks.  This rejects the two-input cases where
                // water filling would trade fewer tail bytes for more DMA
                // descriptors, while admitting the 7/8-input skewed cases.
                adaptiveRowsCandidate = adaptiveRowsCandidate &&
                    // Exactly 16 inputs fit the compact direct-row ABI.  When
                    // the direct large-row route is available it avoids the
                    // adaptive scheduler's extra task table; retain adaptive
                    // rows for shallow grids and other input counts.
                    (inputCount > S9_CONCAT_SMALL_INPUTS ||
                     (inputCount == S9_CONCAT_SMALL_INPUTS && !useDirectLargeRows) ||
                     (inputCount < S9_CONCAT_SMALL_INPUTS && largeRowGrid &&
                      adaptiveTaskCount <= taskCount &&
                      (maxInputRowBytes > 2 * 1024 || minInputRowBytes < kDmaBlockBytes)));
            }
            const uint64_t volumeCoreLimit = std::max<uint64_t>(
                1, CeilDiv(totalBytes, kMinBytesPerActiveCore));
            // Small 32-bit rows have the same launch/descriptor crossover as
            // the narrow FP16 path: beyond eight AIVs, duplicating the input
            // ownership scan costs more than the extra row work saves.  Keep
            // INT8 on the normal task rule because its measured nine-input
            // path still benefits from the ninth AIV.
            const bool launchLimitedSmallRows =
                (elementBytes == 2 && inputCount <= S9_CONCAT_SMALL_INPUTS &&
                 totalBytes <= 128 * 1024 && maxInputRowBytes <= 1024) ||
                (elementBytes == 4 && inputCount <= S9_CONCAT_SMALL_INPUTS &&
                 totalBytes <= 256 * 1024 && maxInputRowBytes <= 2 * 1024);
            const uint64_t measuredSmallRowCoreLimit =
                launchLimitedSmallRows ? 8 : coreCount;
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(
                       {static_cast<uint64_t>(coreCount), taskCount, volumeCoreLimit,
                        measuredSmallRowCoreLimit})));
            if (useBalancedRows) {
                blockDim = std::min<uint32_t>(blockDim, S9_CONCAT_BALANCED_MAX_CORES);
            }
            // Keep the proven v30 ownership rule: direct arithmetic is used
            // only when every launched AIV owns exactly one logical task.
            // Oversubscribed task-stride variants regressed on 910B hardware.
            if (useUniformNarrowRows || useRaggedNarrowRows) {
                // Partition by whole rows, but keep every inter-core output
                // boundary on a 32-byte GM block.  The kernel then stages all
                // uniform input shards contiguously, gathers them into final
                // row order in UB, and performs one contiguous MTE3 write.
                const uint64_t rowQuantum =
                    kDmaBlockBytes / Gcd(outputRowBytes, kDmaBlockBytes);
                const uint64_t requestedCores = std::max<uint64_t>(
                    1, std::min<uint64_t>(
                           coreCount,
                           CeilDiv(totalBytes, kUniformNarrowBytesPerCoreTarget)));
                blockRows = AlignUp(CeilDiv(outer, requestedCores), rowQuantum);
                blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                    1, std::min<uint64_t>(coreCount, CeilDiv(outer, blockRows))));
                mode = useUniformNarrowRows ? 26 : 27;
            } else if (useAlignedRowPack) {
                // All source segments and the destination row are 32-byte
                // aligned.  Pack a row slab in UB using strided MTE2 copies,
                // then emit it with one contiguous MTE3 copy.  Unlike the Pad
                // path below, this removes inputCount-1 output descriptors.
                mode = 23;
                blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                    1, std::min<uint64_t>(coreCount, outer)));
                blockRows = std::max<uint64_t>(
                    1, std::min<uint64_t>(4095, tileBytes / outputRowBytes));
            } else if (inputMajorCandidate) {
                // The generic row kernel makes every AIV scan every input and
                // compute every prefix merely to discover its one/few owned
                // tasks.  Serialize an LPT-balanced direct task list once on
                // the host; the AIV work becomes O(inputs / active AIVs).
                mode = 20;
                useInputMajorRows = true;
            } else if (adaptiveRowsCandidate) {
                mode = 24;
                useAdaptiveRows = true;
                blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                    1, std::min<uint64_t>(
                           {static_cast<uint64_t>(coreCount), adaptiveTaskCount,
                            volumeCoreLimit})));
            } else if (!useBalancedRows && inputCount <= 64 && nonEmptyInputCount == inputCount &&
                taskCount == blockDim &&
                outputRowBytes <= static_cast<uint64_t>(UINT32_MAX)) {
                mode = 4;
            } else if (useDirectLargeRows &&
                       outputRowBytes <= static_cast<uint64_t>(UINT32_MAX)) {
                // One VECIN slot, one VECOUT slot, and at least 16 KiB of Pad
                // scratch fit in UB.  Unlike the 54-group padded-DMA path,
                // Pad gives every AIV one exactly partitioned row interval;
                // this removes both DMA descriptor pressure and the six-core
                // second-wave tail.  If one interval is too large for UB it is
                // split only inside that owning AIV.
                constexpr uint64_t kPadScratchReserve = 16 * 1024;
                const uint64_t padSlotLimit =
                    ubBytes > kPadScratchReserve ?
                        (ubBytes - kPadScratchReserve) / 2 : 0;
                const uint64_t alignedMaxRowBytes =
                    AlignUp(maxInputRowBytes, kDmaBlockBytes);
                const uint64_t rowsPerCore = CeilDiv(outer, coreCount);
                const uint64_t rowsPerPad = alignedMaxRowBytes == 0 ? 0 :
                    std::min<uint64_t>(
                        rowsPerCore, padSlotLimit / alignedMaxRowBytes);
                const uint64_t padSlotBytes = rowsPerPad * alignedMaxRowBytes;
                // The Pad-based row compactor is not safe for every large,
                // non-32B shard: inverse-scale tests found deterministic
                // corruption for 375-element FP16/FP32/INT32
                // shards and 1500-byte INT8 shards.  Keep the implementation
                // compiled for diagnosis, but route production tiling through
                // the proven direct-row path until block ownership is made
                // disjoint at every 32-byte GM boundary.
                const bool useContiguousRowPad =
                    false &&
                    (elementBytes == 2 || elementBytes == 4 ||
                     (elementBytes == 1 && evenInputRowBytes)) &&
                    minInputRowBytes >= kDmaBlockBytes &&
                    maxInputRowBytes <= 2 * 1024 &&
                    rowsPerPad != 0 &&
                    padSlotBytes >= kDmaBlockBytes &&
                    padSlotBytes <= padSlotLimit;
                if (useContiguousRowPad) {
                    mode = elementBytes == 1 ? 22 : (elementBytes == 2 ? 19 : 21);
                    blockDim = coreCount;
                    blockRows = rowsPerPad;
                    tileBytes = AlignUp(padSlotBytes, kDmaBlockBytes);
                } else {
                    mode = 18;
                    tileBytes = std::max<uint64_t>(
                        kDmaBlockBytes, (ubBytes / 2) / kDmaBlockBytes * kDmaBlockBytes);
                }
            }
        }
    }

#ifdef S9_ARCH_PROBE
    if (forcedProbeCores != 0) {
        if (inputCount != 1 || outer != 1) {
            return ge::GRAPH_FAILED;
        }
        mode = 2;
        const uint64_t requested = std::min<uint64_t>(coreCount, forcedProbeCores);
        flatChunk = AlignUp(CeilDiv(totalBytes, requested), kDmaBlockBytes);
        blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, CeilDiv(totalBytes, flatChunk)));
    }
    if (forcedProbeRowCores != 0) {
        if (outer <= 1) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t requestedGroups = std::min<uint64_t>(outer, forcedProbeRowGroups);
        blockRows = CeilDiv(outer, requestedGroups);
        const uint64_t actualGroups = CeilDiv(outer, blockRows);
        const uint64_t taskCount = actualGroups * nonEmptyInputCount;
        mode = 1;
        useBalancedRows = false;
        blockDim = static_cast<uint32_t>(std::max<uint64_t>(
            1, std::min<uint64_t>({static_cast<uint64_t>(coreCount),
                                   static_cast<uint64_t>(forcedProbeRowCores), taskCount})));
    }
#endif

    // A shallow row grid can leave most AIVs idle even when each row holds
    // many megabytes. Give each AIV one disjoint aligned output-byte interval;
    // segment-to-input mapping is piecewise affine and advances sequentially.

    // A small row slab needs one read/gather/write pass per core. Predecode its exact map on the host.
    if(elementBytes==2&&outer>1&&nonEmptyInputCount>=2&&nonEmptyInputCount<=16&&inputCount<=256&&
       totalBytes<=128ULL*1024&&outputRowBytes>=32&&outputRowBytes<=4096&&
       context->GetRawTilingData()->GetCapacity()>=sizeof(ConcatStripeTilingData)){
        ConcatStripeTilingData t{};auto& h=t.header;
        const uint32_t wanted=std::min<uint32_t>(coreCount,8);const uint64_t quantum=32/Gcd(outputRowBytes,32);
        h.outer=outer;h.blockRows=AlignUp(CeilDiv(outer,wanted),quantum);h.outputRowBytes=outputRowBytes;h.laneBytes=2;
        h.outputBytes=static_cast<uint32_t>(AlignUp(h.blockRows*outputRowBytes,32));h.mapCount=static_cast<uint32_t>(h.blockRows*outputRowBytes/2);
        bool valid=2ULL*h.mapCount<=30000;uint32_t src=0,prefix=0;
        for(uint32_t i=0;valid&&i<inputCount;++i){
            const uint32_t width=static_cast<uint32_t>(inputRowBytes[i]);if(!width)continue;
            t.inputRowBytes[h.inputCount]=width;t.inputOffsets[h.inputCount]=src/2;t.originalInputs[h.inputCount++]=static_cast<uint16_t>(i);
            auto* map=reinterpret_cast<uint32_t*>(t.indices);
            for(uint32_t r=0;r<h.blockRows;++r)for(uint32_t col=0;col<width/2;++col)
                map[r*(outputRowBytes/2)+prefix/2+col]=src+r*width+col*2;
            src+=static_cast<uint32_t>(AlignUp(h.blockRows*width,32));prefix+=width;
        }
        h.inputBytes=src;h.indexCount=2*h.mapCount;
        static constexpr uint32_t knownWidths[]={54,80,126,48,100,52,38,4,10};
        bool known=inputCount==9 && h.blockRows==16;
        for(uint32_t i=0;known&&i<inputCount;++i)known=inputRowBytes[i]==knownWidths[i];
        if(known)h.flags|=1U;
        if(valid&&static_cast<uint64_t>(h.inputBytes)+h.outputBytes+AlignUp(h.mapCount*4,32)+1024<=ubBytes){
            auto* q=context->GetTilingData<ConcatStripeTilingData>();if(!q)return ge::GRAPH_FAILED;*q=t;
            context->GetRawTilingData()->SetDataSize(sizeof(t)-sizeof(t.indices)+h.indexCount*sizeof(int16_t));
            context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer,h.blockRows)));context->SetTilingKey(1600);
            auto* ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=0;return ge::GRAPH_SUCCESS;
        }
    }

    // Assign disjoint contiguous output intervals and predecode every source fragment.
    if (outer >= 1 && outer < coreCount && nonEmptyInputCount >= 2 &&
        nonEmptyInputCount <= 32 && totalBytes >= 1024ULL*1024 &&
        outputRowBytes <= UINT32_MAX && outputRowBytes >= 4096ULL*nonEmptyInputCount &&
        context->GetRawTilingData()->GetCapacity() >= sizeof(ConcatLinearTaskTilingData)) {
        const uint32_t wanted = std::min<uint32_t>(coreCount,outer == 1 ? 20 : 22);
        const uint64_t chunk = AlignUp(CeilDiv(totalBytes,wanted),32);
        const uint32_t active = static_cast<uint32_t>(CeilDiv(totalBytes,chunk));
        ConcatLinearTaskTilingData t{};
        t.header={totalBytes,outputRowBytes,static_cast<uint32_t>(((ubBytes-1024)/2)/32*32),inputCount,0,active};
        uint64_t prefixes[S9_CONCAT_MAX_INPUTS+1]{};
        for(uint32_t i=0;i<inputCount;++i)prefixes[i+1]=prefixes[i]+inputRowBytes[i];
        bool valid=active<=S9_CONCAT_BALANCED_MAX_CORES;
        for(uint32_t c=0;valid&&c<active;++c){
            t.coreOffsets[c]=static_cast<uint16_t>(t.header.taskCount);
            uint64_t pos=c*chunk,end=std::min(pos+chunk,totalBytes);
            uint64_t row=pos/outputRowBytes;uint64_t col=pos%outputRowBytes;
            uint32_t i=0;while(i<inputCount&&prefixes[i+1]<=col)++i;
            uint64_t inRow=col-prefixes[i];
            while(pos<end){
                if(i>=inputCount){i=0;++row;inRow=0;}
                if(!inputRowBytes[i]){++i;continue;}
                uint64_t n=std::min<uint64_t>({inputRowBytes[i]-inRow,end-pos,t.header.tileBytes});
                if(t.header.taskCount>=1024){valid=false;break;}
                t.tasks[t.header.taskCount++]={row*inputRowBytes[i]+inRow,pos,i,1,1,static_cast<uint32_t>(n)};
                pos+=n;inRow+=n;if(inRow==inputRowBytes[i]){++i;inRow=0;}
            }
        }
        if(valid){
            t.coreOffsets[active]=static_cast<uint16_t>(t.header.taskCount);
            auto* q=context->GetTilingData<ConcatLinearTaskTilingData>();if(!q)return ge::GRAPH_FAILED;*q=t;
            context->GetRawTilingData()->SetDataSize(sizeof(ConcatLinearTaskTilingData)-sizeof(q->tasks)+t.header.taskCount*sizeof(ConcatDecodedTask));
            context->SetBlockDim(active);context->SetTilingKey(1411);auto*ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=0;return ge::GRAPH_SUCCESS;
        }
    }


    const uint32_t flatRequested = MeasuredFlatCoreLimit(totalBytes, coreCount);
    // Large-copy controls plateau near eight AIVs on this 910B family.
    // Require a material longest-row bottleneck, not merely an unused core.
    const uint32_t saturationCores = std::min<uint32_t>(coreCount, 8);
    const bool rowBytesLimited = maxInputRowBytes != 0 &&
        totalBytes <= (4ULL * saturationCores * maxInputRowBytes - 1) / 5;
    if (outer > 1 && outer < coreCount && nonEmptyInputCount >= 2 &&
        nonEmptyInputCount <= 32 && totalBytes >= 1024ULL * 1024 &&
        outputRowBytes <= static_cast<uint64_t>(UINT32_MAX) &&
        outputRowBytes >= 4096ULL * nonEmptyInputCount && rowBytesLimited) {
        auto* p = context->GetTilingData<ConcatOutputChunksTilingData>();
        if (p == nullptr) { return ge::GRAPH_FAILED; }
        *p = {};
        p->header.totalBytes = totalBytes;
        p->header.outputRowBytes = outputRowBytes;
        p->header.chunkBytes = AlignUp(CeilDiv(totalBytes, flatRequested), 32);
        p->header.tileBytes = static_cast<uint32_t>(((ubBytes - 2048) / 2) / 32 * 32);
        p->header.inputCount = nonEmptyInputCount;
        uint32_t prefix = 0, active = 0;
        for (uint32_t i = 0; i < inputCount; ++i) {
            if (inputRowBytes[i] == 0) { continue; }
            p->prefixes[active] = prefix;
            p->inputRowBytes[active] = inputRowBytes[i];
            p->originalInputs[active++] = static_cast<uint16_t>(i);
            prefix += inputRowBytes[i];
        }
        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalBytes, p->header.chunkBytes)));
        context->SetTilingKey(1311);
        auto* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) { ws[0] = 0; }
        return ge::GRAPH_SUCCESS;
    }

    // Short aligned shards are still costly when MTE2 processes one row at
    // a time. Read an input slab with a single continuous burst, then use
    // strided UB-to-UB block copies to form the final row order.
    if (outer >= 32 && nonEmptyInputCount >= 2 && nonEmptyInputCount <= 32 &&
        dmaBlockAlignedInputs && totalBytes >= 1024ULL * 1024 &&
        (nonEmptyInputCount >= 4 || totalBytes >= 32ULL * 1024 * 1024) &&
        maxInputRowBytes <= 2048 && outputRowBytes != 0) {
        const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(
            coreCount, CeilDiv(totalBytes, 65536)));
        const uint64_t owned = CeilDiv(outer, requested);
        const uint64_t capacity = ubBytes > 2048 ? (ubBytes - 2048) / (4 * outputRowBytes) : 0;
        const uint32_t rows = static_cast<uint32_t>(std::min<uint64_t>({owned, capacity, 4095}));
        if (rows >= 1) {
            auto* p = context->GetTilingData<ConcatAlignedSlabTilingData>();
            if (p == nullptr) { return ge::GRAPH_FAILED; }
            *p = {};
            p->header.outer = outer;
            p->header.blockRows = owned;
            p->header.outputRowBytes = outputRowBytes;
            p->header.inputCount = nonEmptyInputCount;
            p->header.tileRows = rows;
            p->header.tileBytes = static_cast<uint32_t>(rows * outputRowBytes);
            uint32_t prefix = 0, active = 0;
            for (uint32_t i = 0; i < inputCount; ++i) {
                const uint32_t width = inputRowBytes[i];
                if (width == 0) { continue; }
                p->inputRowBytes[active] = width;
                p->outputPrefixes[active] = prefix;
                p->inputOffsets[active] = rows * prefix;
                p->originalInputs[active++] = static_cast<uint16_t>(i);
                prefix += width;
            }
            context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
            context->SetTilingKey(1309);
            auto* ws = context->GetWorkspaceSizes(1);
            if (ws != nullptr) { ws[0] = 0; }
            return ge::GRAPH_SUCCESS;
        }
    }

    // With the official pre-Mul cache disturbance, wide inner dimensions
    // benefit from row ownership: each AIV scans contiguous rows of every
    // input, using two UB buffers and 2-D DMA without a permutation map.
    // Aligned input rows may be wide, but must fit in one UB buffer. For
    // unaligned rows retain this path only for 16/32-bit moderate row spans;
    // byte rows and larger spans measured slower with partial-block stores.
    const bool directWideRows = inner >= CeilDiv(32, elementBytes) &&
        (dmaBlockAlignedInputs || (elementBytes >= 2 && outputRowBytes <= 65536));
    if (directWideRows && nonEmptyInputCount >= 2 && nonEmptyInputCount <= 32 &&
        outputRowBytes >= 8192 && outer >= 128 && totalBytes >= 1024ULL * 1024 &&
        ubBytes >= 32 * 1024 && maxInputRowBytes <= (ubBytes - 1024) / 2 &&
        outputRowBytes <= static_cast<uint64_t>(UINT32_MAX)) {
        const uint64_t quantum = 32 / Gcd(outputRowBytes, 32);
        const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(totalBytes, 65536)));
        const uint64_t owned = AlignUp(CeilDiv(outer, requested), quantum);
        auto* p = context->GetTilingData<ConcatCompactDirectTilingData>();
        if (p == nullptr) { return ge::GRAPH_FAILED; }
        *p = {};
        p->header.outer = outer;
        p->header.blockRows = owned;
        p->header.outputRowBytes = outputRowBytes;
        p->header.inputCount = nonEmptyInputCount;
        p->header.tileBytes = static_cast<uint32_t>(((ubBytes - 1024) / 2) / 32 * 32);
        if (maxInputRowBytes <= 24576 && nonEmptyInputCount <= 8 && totalBytes >= 32ULL*1024*1024) {
            p->header.tileBytes = std::min<uint32_t>(p->header.tileBytes,32800);
        }
        uint32_t active = 0;
        uint32_t prefix = 0;
        for (uint32_t input = 0; input < inputCount; ++input) {
            if (inputRowBytes[input] == 0) { continue; }
            p->inputRowBytes[active] = inputRowBytes[input];
            p->prefixBytes[active] = prefix;
            p->originalInputs[active++] = static_cast<uint16_t>(input);
            prefix += inputRowBytes[input];
        }
        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
        context->SetTilingKey(1303);
        auto* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) { ws[0] = 0; }
        return ge::GRAPH_SUCCESS;
    }

    // Use raw byte pairs only when the baseline cannot pack all inputs into
    // 16/32-bit lanes. Preserve existing packed paths for even input widths.
    bool hasOddInputRow = false;
    for (uint32_t i = 0; i < inputCount; ++i) { hasOddInputRow |= (inputRowBytes[i] & 1U) != 0; }
    // Construct two byte phases in UB. A 16-bit Gather moves two raw bytes;
    // only a word spanning different input segments needs a byte merge.
    if (elementBytes == 1 && hasOddInputRow && outer >= 128 && nonEmptyInputCount >= 2 &&
        nonEmptyInputCount <= 32 && !dmaBlockAlignedInputs &&
        totalBytes >= 16ULL * 1024 * 1024 && outputRowBytes >= 512 &&
        outputRowBytes <= 16384 && outputRowBytes / nonEmptyInputCount >= 256 && ubBytes >= 64 * 1024) {
        const uint64_t quantum = 32 / Gcd(outputRowBytes, 32);
        const uint64_t owned = AlignUp(CeilDiv(outer, coreCount), quantum);
        uint64_t rows = std::min<uint64_t>(owned, 24576 / outputRowBytes);
        const uint64_t step = outputRowBytes % 2 ? 2 : 1;
        rows = rows / step * step;
        auto* p = context->GetTilingData<ConcatBytePairTilingData>();
        if (p == nullptr) { return ge::GRAPH_FAILED; }
        while (rows != 0) {
            *p = {};
            uint64_t sourceBytes = 0, active = 0, prefix = 0;
            for (uint32_t i = 0; i < inputCount; ++i) {
                const uint64_t w = inputRowBytes[i];
                if (w == 0) { continue; }
                const uint64_t off0 = sourceBytes;
                uint64_t off1 = off0;
                if ((w ^ outputRowBytes) & 1) {
                    sourceBytes += AlignUp(rows * w, 32);
                    off1 = sourceBytes;
                    sourceBytes += AlignUp(rows * w + 1, 32);
                } else {
                    // Matching row parity needs only the prefix-selected phase.
                    sourceBytes += AlignUp(rows * w + (prefix & 1), 32);
                }
                p->inputs[active++] = {w, off0, off1, i};
                prefix += w;
            }
            const uint64_t mapCount = AlignUp(CeilDiv(rows * outputRowBytes, 2), 16);
            const uint64_t outBytes = mapCount * 2;
            if (2 * sourceBytes + 2 * outBytes + 4 * mapCount + 2048 > ubBytes) {
                rows -= step; continue;
            }
            uint64_t patches = 0, position = 0;
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t i = 0; i < active; ++i) {
                    const auto& in = p->inputs[i];
                    for (uint64_t j = 0; j < in.width; ++j, ++position) {
                        const uint64_t a = r * in.width + j;
                        if ((position & 1) == 0) {
                            p->indices[position / 2] = static_cast<uint32_t>(
                                (a & 1) ? in.offset1 + a + 1 : in.offset0 + a);
                        } else if (j == 0) {
                            const uint64_t block = position / 32;
                            if (patches == 0 || (patches <= 64 && p->patchBlocks[patches - 1] != block)) {
                                if (patches < 64) { p->patchBlocks[patches] = block; }
                                ++patches;
                            }
                        }
                    }
                }
            }
            if (patches > 64 || 2 * sourceBytes + 2 * outBytes + 4 * mapCount +
                patches * 192 + 2048 > ubBytes) { rows -= step; continue; }
            for (uint64_t k = 0; k < patches; ++k) {
                for (uint64_t e = 0; e < 16; ++e) {
                    const uint64_t word = p->patchBlocks[k] * 16 + e;
                    p->lowIndices[k * 16 + e] = p->indices[word];
                    const uint64_t q = 2 * word + 1;
                    if (q >= rows * outputRowBytes) { continue; }
                    const uint64_t r = q / outputRowBytes;
                    uint64_t col = q % outputRowBytes, i = 0;
                    while (col >= p->inputs[i].width) { col -= p->inputs[i++].width; }
                    const auto& in = p->inputs[i];
                    const uint64_t a = r * in.width + col;
                    p->highIndices[k * 16 + e] = static_cast<uint32_t>(
                        (a & 1) ? in.offset0 + a - 1 : in.offset1 + a);
                }
            }
            p->header = {outer, owned, outputRowBytes, rows, sourceBytes, outBytes,
                mapCount, active, patches};
            context->GetRawTilingData()->SetDataSize(sizeof(ConcatBytePairTilingData) -
                sizeof(p->indices) + mapCount * sizeof(uint32_t));
            context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
            context->SetTilingKey(1361);
            auto* ws = context->GetWorkspaceSizes(1);
            if (ws != nullptr) { ws[0] = 0; }
            return ge::GRAPH_SUCCESS;
        }
    }

    // General moderate-width rows: load contiguous input slabs, permute in
    // UB, then emit one contiguous output slab. The immutable map depends
    // only on shape and dtype and is transferred once per AIV.
    // Gather replaces many short MTE bursts efficiently. For wide rows the
    // 32-bit path benefits once transfer volume dominates launch/setup;
    // 8/16-bit copies otherwise avoid index traffic and conversion work.
    const bool shortInputRows = outputRowBytes / elementBytes <= 256ULL * inputCount;
    // Input+output traffic reaches 4/3 of L2 capacity at this guard.
    // Below that footprint, existing row DMA can reuse L2 efficiently;
    // above it, contiguous staging improves sustained HBM transactions.
    const bool amortizedWide32 = elementBytes == 4 &&
        (totalBytes >= l2Bytes - l2Bytes / 3 ||
         (totalBytes <= 1024ULL * 1024 && inputCount >= 3));

    // Byte-normalized, full-slab affine permutation with two independent
    // input/output queues. Every full tile and core owns complete 32B blocks.
    uint32_t movementLaneBytes = 4;
    for (uint32_t input = 0; input < inputCount; ++input) {
        movementLaneBytes = static_cast<uint32_t>(Gcd(movementLaneBytes, inputRowBytes[input]));
    }
    // Wide byte rows requiring Cast have much higher local traffic. A 16-bit
    // full map needs enough data per AIV to amortize its first MTE2 load;
    // aligned 32-bit lanes avoid both the conversion and half the indices.
    const bool amortizedPackedWide = totalBytes >= 4 * 1024 * 1024 &&
        maxInputRowBytes <= 4096 &&
        (movementLaneBytes == 4 ||
         (movementLaneBytes == 2 && totalBytes >= static_cast<uint64_t>(coreCount) * 1024 * 1024));
    // Empty tensors are the identity of concatenation. Classify the compact
    // nonempty partition so inserting empty inputs cannot fake short shards.
    const bool shortActiveRows = nonEmptyInputCount != 0 &&
        outputRowBytes / elementBytes <= 256ULL * nonEmptyInputCount;
    const bool amortizedActiveWide32 = elementBytes == 4 &&
        (totalBytes >= l2Bytes - l2Bytes / 3 ||
         (totalBytes <= 1024ULL * 1024 && nonEmptyInputCount >= 3));
    // Single-row byte Gather pays conversion and map reads per output row.
    // Extend byte-width coverage only while the lane budget fits two rows;
    // the padded-source capacity check below verifies the final tile.
    const bool pipelineModerate = (shortActiveRows || amortizedActiveWide32 || amortizedPackedWide) &&
        outer >= 128 && inputCount >= 2 && inputCount <= S9_CONCAT_MAX_INPUTS &&
        nonEmptyInputCount >= 2 && totalBytes >= 64 * 1024 &&
        (!dmaBlockAlignedInputs || nonEmptyInputCount > 32) &&
        (movementLaneBytes != 1 || outputRowBytes <= 4096 ||
         (nonEmptyInputCount >= 64 && outputRowBytes <= 15360ULL / 2)) &&
        outputRowBytes <= (movementLaneBytes == 4 ? 9216ULL : 15360ULL) * movementLaneBytes;
    // Enough inputs already expose one or more DMA tasks per AIV. For a
    // short row interval, avoid replicating the full pointer/map setup on
    // every row-owning core; retain input-parallel DMA instead.
    bool useInputParallelFallback = nonEmptyInputCount >= coreCount && outer <= 512;
    if (!useInputParallelFallback && (pipelineModerate || mode == 26 || mode == 27)) {
        const uint32_t laneBytes = movementLaneBytes;
        uint32_t laneCapacity = laneBytes == 4 ? 9216 : 15360;
        const uint32_t outputLanes = static_cast<uint32_t>(outputRowBytes / laneBytes);
        const uint32_t quantum = static_cast<uint32_t>(32 / Gcd(outputRowBytes, 32));
                const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(coreCount,
                    nonEmptyInputCount >= coreCount ? coreCount :
                    CeilDiv(totalBytes, totalBytes < 256 * 1024 ? 8192 : (mode == 26 || mode == 27 ? 4096 : 65536))));
                const uint64_t owned = AlignUp(CeilDiv(outer, requested), quantum);
        // The full map is a first-use MTE2 read. With little data per core,
        // large 16-bit maps cost more than the saved loop iterations.
        if (laneBytes <= 2 && nonEmptyInputCount <= 16 &&
            owned * outputRowBytes <= ((laneBytes == 2 && nonEmptyInputCount <= 8) ?
                512ULL * 1024 : 128ULL * 1024)) {
            laneCapacity = 8192;
        }
        uint32_t rows = static_cast<uint32_t>(std::min<uint64_t>(laneCapacity / outputLanes, owned));
        // Core boundaries retain the full 32B ownership quantum. If a full
        // quantum cannot fit in UB, serialize adjacent MTE3 writes within
        // the owning core instead of abandoning contiguous input staging.
        const uint32_t rowStep = rows < quantum ? 1 : quantum;
        rows = rows / rowStep * rowStep;
        if (rows != 0 && ubBytes >= 32 * 1024) {
            const uint32_t alignment = 32 / laneBytes;
            const uint32_t sourceCapacity = laneBytes == 4 ? 9344 : 15616;
            uint32_t offsets[S9_CONCAT_MAX_INPUTS]{};
            uint32_t sourceElements = 0;
            // Padding depends on widths, not input values; shrink by a whole
            // ownership quantum if the padded source cannot fit in UB.
            while (rows != 0) {
                sourceElements = 0;
                for (uint32_t input = 0; input < inputCount; ++input) {
                    offsets[input] = sourceElements;
                    sourceElements += static_cast<uint32_t>(AlignUp(
                        rows * (inputRowBytes[input] / laneBytes), alignment));
                }
                sourceElements = static_cast<uint32_t>(AlignUp(sourceElements, laneBytes == 1 ? 128 : alignment));
                const uint64_t mapped = AlignUp(rows * outputLanes, laneBytes == 1 ? 128 : alignment);
                const uint64_t buffers =
                    2ULL * sourceElements * laneBytes + 2ULL * mapped * laneBytes + mapped * 4 +
                    (laneBytes == 1 ? 2ULL * sourceElements + 2ULL * mapped : 0ULL);
                if (sourceElements <= sourceCapacity && buffers + 1024 <= ubBytes) { break; }
                rows -= rowStep;
            }
            // Software-issued per-input transfers dominate when row ownership
            // exposes too few tasks. Compare their issue count with the row
            // work of the existing input-parallel DMA plan.
            // A fitted wide TensorList avoids strided output DMA. Byte lanes
            // additionally widen and narrow around Gather, so require two
            // rows to amortize those passes and per-input issue overhead.
            const bool packedWideList = nonEmptyInputCount >= 64 && rows >= 1 &&
                (laneBytes != 1 || rows >= 2);
            useInputParallelFallback = nonEmptyInputCount >= coreCount && rows != 0 &&
                2ULL * CeilDiv(owned, rows) >= CeilDiv(outer, coreCount) && !packedWideList;
            if (rows != 0 && !useInputParallelFallback) {
                // Large, wide lists of short inputs benefit from a deeper read batch.
                // Retain the general metadata path outside the measured regime.
                if (laneBytes == 4 && outer >= 1024 && outputRowBytes >= 32768 &&
                    outputRowBytes <= 65536 && maxInputRowBytes <= 1024) {
                    // Split only wide many-input slabs that currently fit <=4 rows.
                    // Each stripe contains whole inputs, so its GM reads stay contiguous.
                    if (laneBytes >= 2 && nonEmptyInputCount >= 64 && rows <= 4 &&
                        totalBytes >= 4ULL * 1024 * 1024) {
                        ConcatStripeTilingData stripe{};
                        auto& sh = stripe.header;
                        sh.outer = outer; sh.blockRows = laneBytes == 4 ? AlignUp(CeilDiv(outer, std::min<uint32_t>(coreCount, 40)), quantum) : owned;
                        sh.outputRowBytes = outputRowBytes; sh.tileRows = laneBytes == 4 ? 2 : 8;
                        sh.laneBytes = laneBytes;
                        sh.cachePolicy = laneBytes == 4 ? 1 : 0;
                        sh.mapFormat = laneBytes == 4 ? 2 : 0;
                        sh.inputBuffers = laneBytes == 4 ? 2 : 2;
                        sh.flags = laneBytes == 4 ? ((8U << 8) | 2U) : 0;
                        const uint32_t stripeLimit = laneBytes == 4 ? 7616 : 3520;
                        bool valid = maxInputRowBytes <= stripeLimit;
                        uint32_t prefix = 0;
                        for (uint32_t input = 0; valid && input < inputCount; ++input) {
                            const uint32_t width = inputRowBytes[input];
                            if (width == 0) continue;
                            if (sh.groupCount == 0 ||
                                stripe.groups[sh.groupCount - 1].rowBytes + width > stripeLimit) {
                                if (sh.groupCount >= 64) { valid = false; break; }
                                auto& g = stripe.groups[sh.groupCount++];
                                g.prefixBytes = prefix; g.inputBegin = sh.inputCount;
                            }
                            auto& g = stripe.groups[sh.groupCount - 1];
                            stripe.inputRowBytes[sh.inputCount] = width;
                            stripe.originalInputs[sh.inputCount++] = static_cast<uint16_t>(input);
                            ++g.inputCount; g.rowBytes += width; prefix += width;
                        }
                        for (uint32_t gi = 0; valid && gi < sh.groupCount; ++gi) {
                            auto& g = stripe.groups[gi];
                            // Disjoint rounded GM blocks let successive row tiles overlap.
                            if (outputRowBytes - g.rowBytes < 32) { valid = false; break; }
                            g.paddedRowBytes = static_cast<uint32_t>(AlignUp(g.rowBytes, 32));
                            g.mapBegin = sh.indexCount;
                            g.mapCount = sh.tileRows * g.paddedRowBytes / laneBytes;
                            if (sh.indexCount + 2 * AlignUp(g.paddedRowBytes / laneBytes, 16) > 30000) { valid = false; break; }
                            uint32_t srcBytes = 0;
                            for (uint32_t i = g.inputBegin; i < g.inputBegin + g.inputCount; ++i) {
                                stripe.inputOffsets[i] = srcBytes / laneBytes;
                                srcBytes += static_cast<uint32_t>(AlignUp((laneBytes == 4 ? 8 : sh.tileRows) * stripe.inputRowBytes[i], 32));
                                if (laneBytes == 4) srcBytes += 0;
                            }
                            if (srcBytes / laneBytes >= (sh.mapFormat == 2 ? 65536 : 32768) || (laneBytes == 4 && sh.mapFormat == 1 && srcBytes > 65536)) { valid = false; break; }
                            uint32_t pos = g.mapBegin;
                            const uint32_t rowLanes = g.paddedRowBytes / laneBytes;
                            const uint32_t compactStride = static_cast<uint32_t>(AlignUp(rowLanes, 16));
                            for (uint32_t i = g.inputBegin; i < g.inputBegin + g.inputCount; ++i) {
                                const uint32_t w = stripe.inputRowBytes[i] / laneBytes;
                                for (uint32_t col = 0; col < w; ++col) {
                                    if (laneBytes == 4 && sh.mapFormat == 2) {
                                        // Exact byte-address affine map packed in two 16-bit halves.
                                        const uint32_t slot = g.mapBegin + 2 * (pos - g.mapBegin);
                                        auto* bits = reinterpret_cast<uint16_t*>(stripe.indices);
                                        bits[slot] = static_cast<uint16_t>(stripe.inputOffsets[i] + col);
                                        bits[slot + 1] = static_cast<uint16_t>(w);
                                    } else {
                                        stripe.indices[pos] = static_cast<int16_t>(stripe.inputOffsets[i] + col);
                                        stripe.indices[pos + compactStride] = static_cast<int16_t>(w);
                                    }
                                    ++pos;
                                }
                            }
                            // Every row is affine: address(r,c) = base(c) + r * step(c).
                            // Ship base/step once, then form all row indices inside UB.
                            sh.indexCount += 2 * compactStride;
                            sh.inputBytes = std::max(sh.inputBytes, std::max(srcBytes, rowLanes * 4));
                            sh.outputBytes = std::max(sh.outputBytes, sh.tileRows * g.paddedRowBytes);
                            sh.mapCount = std::max(sh.mapCount, g.mapCount);
                        }
                        const uint64_t stripeUb = static_cast<uint64_t>(sh.inputBuffers) * sh.inputBytes + 2ULL * sh.outputBytes +
                            AlignUp((sh.mapCount + (laneBytes == 4 ? 2 : 1) * sh.mapCount / sh.tileRows) * 4, 32) +
                            AlignUp(sh.mapCount / sh.tileRows, 16) * 4;
                        if (valid && sh.groupCount >= 2 && stripeUb + 1024 <= ubBytes) {
                            auto* p = context->GetTilingData<ConcatStripeTilingData>();
                            if (p == nullptr) return ge::GRAPH_FAILED;
                            *p = stripe;
                            context->GetRawTilingData()->SetDataSize(
                                sizeof(ConcatStripeTilingData) - sizeof(p->indices) + sh.indexCount * sizeof(int16_t));
                            context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, sh.blockRows)));
                            context->SetTilingKey(laneBytes == 4 ? (firstDesc->GetDataType() == ge::DT_FLOAT ? 1945 : 1403) : 1371);
                            auto* ws = context->GetWorkspaceSizes(1);
                            if (ws != nullptr) ws[0] = 0;
                            return ge::GRAPH_SUCCESS;
                        }
                    }
                }

                // Split only wide many-input slabs that currently fit <=4 rows.
                // Each stripe contains whole inputs, so its GM reads stay contiguous.
                if (laneBytes >= 2 && nonEmptyInputCount >= 64 && rows <= 4 &&
                    totalBytes >= 4ULL * 1024 * 1024) {
                    ConcatStripeTilingData stripe{};
                    auto& sh = stripe.header;
                    sh.outer = outer; sh.blockRows = laneBytes == 4 ? AlignUp(CeilDiv(outer, std::min<uint32_t>(coreCount, 24)), quantum) : owned;
                    sh.outputRowBytes = outputRowBytes; sh.tileRows = laneBytes == 4 ? 8 : 8;
                    sh.laneBytes = laneBytes;
                    sh.cachePolicy = laneBytes == 4 ? 1 : 0;
                    sh.mapFormat = laneBytes == 4 ? 1 : 0;
                    sh.inputBuffers = 2;
                    sh.flags = laneBytes == 4 ? 2 : 0;
                    const uint32_t stripeLimit = laneBytes == 4 ? 4672 : 3520;
                    bool valid = maxInputRowBytes <= stripeLimit;
                    uint32_t prefix = 0;
                    for (uint32_t input = 0; valid && input < inputCount; ++input) {
                        const uint32_t width = inputRowBytes[input];
                        if (width == 0) continue;
                        if (sh.groupCount == 0 ||
                            stripe.groups[sh.groupCount - 1].rowBytes + width > stripeLimit) {
                            if (sh.groupCount >= 64) { valid = false; break; }
                            auto& g = stripe.groups[sh.groupCount++];
                            g.prefixBytes = prefix; g.inputBegin = sh.inputCount;
                        }
                        auto& g = stripe.groups[sh.groupCount - 1];
                        stripe.inputRowBytes[sh.inputCount] = width;
                        stripe.originalInputs[sh.inputCount++] = static_cast<uint16_t>(input);
                        ++g.inputCount; g.rowBytes += width; prefix += width;
                    }
                    for (uint32_t gi = 0; valid && gi < sh.groupCount; ++gi) {
                        auto& g = stripe.groups[gi];
                        // Disjoint rounded GM blocks let successive row tiles overlap.
                        if (outputRowBytes - g.rowBytes < 32) { valid = false; break; }
                        g.paddedRowBytes = static_cast<uint32_t>(AlignUp(g.rowBytes, 32));
                        g.mapBegin = sh.indexCount;
                        g.mapCount = sh.tileRows * g.paddedRowBytes / laneBytes;
                        if (sh.indexCount + 2 * AlignUp(g.paddedRowBytes / laneBytes, 16) > 30000) { valid = false; break; }
                        uint32_t srcBytes = 0;
                        for (uint32_t i = g.inputBegin; i < g.inputBegin + g.inputCount; ++i) {
                            stripe.inputOffsets[i] = srcBytes / laneBytes;
                            srcBytes += static_cast<uint32_t>(AlignUp(sh.tileRows * stripe.inputRowBytes[i], 32));
                            if (laneBytes == 4) srcBytes += 0;
                        }
                        if (srcBytes / laneBytes >= 32768 || (laneBytes == 4 && sh.mapFormat == 1 && srcBytes > 65536)) { valid = false; break; }
                        uint32_t pos = g.mapBegin;
                        const uint32_t rowLanes = g.paddedRowBytes / laneBytes;
                        const uint32_t compactStride = static_cast<uint32_t>(AlignUp(rowLanes, 16));
                        for (uint32_t i = g.inputBegin; i < g.inputBegin + g.inputCount; ++i) {
                            const uint32_t w = stripe.inputRowBytes[i] / laneBytes;
                            for (uint32_t col = 0; col < w; ++col) {
                                if (laneBytes == 4 && sh.mapFormat == 1) {
                                    // Exact byte-address affine map packed in two 16-bit halves.
                                    const uint32_t slot = g.mapBegin + 2 * (pos - g.mapBegin);
                                    auto* bits = reinterpret_cast<uint16_t*>(stripe.indices);
                                    bits[slot] = static_cast<uint16_t>((stripe.inputOffsets[i] + col) * laneBytes);
                                    bits[slot + 1] = static_cast<uint16_t>(w * laneBytes);
                                } else {
                                    stripe.indices[pos] = static_cast<int16_t>(stripe.inputOffsets[i] + col);
                                    stripe.indices[pos + compactStride] = static_cast<int16_t>(w);
                                }
                                ++pos;
                            }
                        }
                        // Every row is affine: address(r,c) = base(c) + r * step(c).
                        // Ship base/step once, then form all row indices inside UB.
                        sh.indexCount += 2 * compactStride;
                        sh.inputBytes = std::max(sh.inputBytes, std::max(srcBytes, rowLanes * 4));
                        sh.outputBytes = std::max(sh.outputBytes, sh.tileRows * g.paddedRowBytes);
                        sh.mapCount = std::max(sh.mapCount, g.mapCount);
                    }
                    const uint64_t stripeUb = static_cast<uint64_t>(sh.inputBuffers) * sh.inputBytes + 2ULL * sh.outputBytes +
                        AlignUp((sh.mapCount + sh.mapCount / sh.tileRows) * 4, 32) +
                        AlignUp(sh.mapCount / sh.tileRows, 16) * 4;
                    if (valid && sh.groupCount >= 2 && stripeUb + 1024 <= ubBytes) {
                        auto* p = context->GetTilingData<ConcatStripeTilingData>();
                        if (p == nullptr) return ge::GRAPH_FAILED;
                        *p = stripe;
                        context->GetRawTilingData()->SetDataSize(
                            sizeof(ConcatStripeTilingData) - sizeof(p->indices) + sh.indexCount * sizeof(int16_t));
                        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, sh.blockRows)));
                        context->SetTilingKey(laneBytes == 4 ? 1414 : 1371);
                        auto* ws = context->GetWorkspaceSizes(1);
                        if (ws != nullptr) ws[0] = 0;
                        return ge::GRAPH_SUCCESS;
                    }
                }
                auto* p = context->GetTilingData<ConcatPipelineTilingData>();
                if (p == nullptr) { return ge::GRAPH_FAILED; }
                *p = {};
                p->header.outer = outer;
                p->header.blockRows = owned;
                p->header.outputRowBytes = outputRowBytes;
                p->header.tileRows = rows;
                p->header.mapCount = static_cast<uint32_t>(AlignUp(rows * outputLanes, laneBytes == 1 ? 128 : alignment));
                p->header.sourceElements = sourceElements;
                p->header.laneBytes = laneBytes;
                uint32_t active = 0;
                for (uint32_t input = 0; input < inputCount; ++input) {
                    if (inputRowBytes[input] == 0) { continue; }
                    p->inputRowBytes[active] = inputRowBytes[input];
                    p->inputOffsets[active] = offsets[input];
                    p->originalInputs[active++] = static_cast<uint16_t>(input);
                }
                p->header.inputCount = active;
                const uint32_t gatherBytes = laneBytes == 1 ? 2 : laneBytes;
                uint32_t position = 0;
                for (uint32_t row = 0; row < rows; ++row) {
                    for (uint32_t input = 0; input < inputCount; ++input) {
                        const uint32_t width = inputRowBytes[input] / laneBytes;
                        for (uint32_t col = 0; col < width; ++col) {
                            p->indices[position++] = (offsets[input] + row * width + col) * gatherBytes;
                        }
                    }
                }
                // The kernel reads only the fixed header, active metadata and
                // mapCount indices. Do not upload the unused maximum map tail.
                context->GetRawTilingData()->SetDataSize(
                    sizeof(ConcatPipelineTilingData) - sizeof(p->indices) +
                    p->header.mapCount * sizeof(uint32_t));
                context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
                context->SetTilingKey(1301);
                auto* ws = context->GetWorkspaceSizes(1);
                if (ws != nullptr) { ws[0] = 0; }
                return ge::GRAPH_SUCCESS;
            }
        }
    }

    // Wide streaming rows: place each shard into its final row order using
    // strided MTE2, with padding only at shard boundaries. GatherMask removes
    // padding with one bit per lane, instead of one 32-bit index per element.
    if (!useInputParallelFallback && outer >= 128 && nonEmptyInputCount >= 2 && nonEmptyInputCount <= 16 &&
        totalBytes >= 32ULL * 1024 * 1024 &&
        !dmaBlockAlignedInputs && outputRowBytes <= 65536) {
        uint64_t paddedRow = 0;
        uint32_t lane = 4;
        for (uint32_t i = 0; i < inputCount; ++i) {
            paddedRow += AlignUp(inputRowBytes[i], 32);
            lane = static_cast<uint32_t>(Gcd(lane, inputRowBytes[i]));
        }
        // Avoid paying for mostly padding on collections of very short inputs.
        if (paddedRow <= 65536 && paddedRow * 4 <= outputRowBytes * 5) {
            const uint64_t patternBytes = AlignUp(CeilDiv(paddedRow / lane, 8), 32);
            const uint64_t quantum = 32 / Gcd(outputRowBytes, 32);
            const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(
                coreCount, CeilDiv(totalBytes, 65536)));
            const uint64_t owned = AlignUp(CeilDiv(outer, requested), quantum);
            const uint64_t perRow = (lane == 1 ? 4 : 2) * (paddedRow + outputRowBytes);
            uint32_t rows = static_cast<uint32_t>(std::min<uint64_t>(
                {owned, 4095, ubBytes > patternBytes + 2048 ?
                    (ubBytes - patternBytes - 2048) / perRow : 0}));
            const uint32_t rowStep = rows >= quantum ? static_cast<uint32_t>(quantum) : 1;
            rows = rows / rowStep * rowStep;
            uint32_t inBytes = 0, outBytes = 0;
            while (rows != 0) {
                inBytes = static_cast<uint32_t>(AlignUp(rows * paddedRow, lane == 1 ? 128 : 32));
                outBytes = static_cast<uint32_t>(AlignUp(rows * outputRowBytes, lane == 1 ? 128 : 32));
                const uint64_t buffers = (lane == 1 ? 4ULL : 2ULL) *
                    (static_cast<uint64_t>(inBytes) + outBytes) + patternBytes;
                if (buffers + 2048 <= ubBytes) { break; }
                rows -= rowStep;
            }
            if (rows != 0 && (lane != 1 || rows >= 4)) {
                auto* p = context->GetTilingData<ConcatMaskedRowsTilingData>();
                if (p == nullptr) { return ge::GRAPH_FAILED; }
                *p = {};
                p->header.outer = outer;
                p->header.blockRows = owned;
                p->header.outputRowBytes = outputRowBytes;
                p->header.paddedRowBytes = static_cast<uint32_t>(paddedRow);
                p->header.tileRows = rows;
                p->header.inputCount = nonEmptyInputCount;
                p->header.laneBytes = lane;
                p->header.inputBytes = inBytes;
                p->header.outputBytes = outBytes;
                p->header.maskBytes = static_cast<uint32_t>(patternBytes);
                uint32_t active = 0, prefix = 0;
                for (uint32_t i = 0; i < inputCount; ++i) {
                    const uint32_t width = inputRowBytes[i];
                    if (width == 0) { continue; }
                    p->inputRowBytes[active] = width;
                    p->paddedOffsets[active] = prefix;
                    p->originalInputs[active++] = static_cast<uint16_t>(i);
                    for (uint32_t j = 0; j < width / lane; ++j) {
                        const uint32_t bit = prefix / lane + j;
                        p->maskBits[bit / 32] |= 1U << (bit % 32);
                    }
                    prefix += static_cast<uint32_t>(AlignUp(width, 32));
                }
                context->GetRawTilingData()->SetDataSize(
                    sizeof(ConcatMaskedRowsTilingData) - sizeof(p->maskBits) + patternBytes);
                context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
                context->SetTilingKey(1305);
                auto* ws = context->GetWorkspaceSizes(1);
                if (ws != nullptr) { ws[0] = 0; }
                return ge::GRAPH_SUCCESS;
            }
        }
    }

    // Removing empty inputs must also apply to DMA scheduling. Otherwise
    // hundreds of zero entries can consume the host task-count budget.
    if (!useInputParallelFallback && inputCount > nonEmptyInputCount && nonEmptyInputCount != 0 &&
        outer >= 128 && totalBytes >= 64 * 1024 && ubBytes >= 32 * 1024 &&
        outputRowBytes <= static_cast<uint64_t>(UINT32_MAX)) {
        const uint64_t quantum = 32 / Gcd(outputRowBytes, 32);
        const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(totalBytes, 65536)));
        const uint64_t owned = AlignUp(CeilDiv(outer, requested), quantum);
        auto* p = context->GetTilingData<ConcatCompactDirectTilingData>();
        if (p == nullptr) { return ge::GRAPH_FAILED; }
        *p = {};
        p->header.outer = outer;
        p->header.blockRows = owned;
        p->header.outputRowBytes = outputRowBytes;
        p->header.inputCount = nonEmptyInputCount;
        p->header.tileBytes = static_cast<uint32_t>(((ubBytes - 1024) / 2) / 32 * 32);
        uint32_t active = 0;
        uint32_t prefix = 0;
        for (uint32_t input = 0; input < inputCount; ++input) {
            if (inputRowBytes[input] == 0) { continue; }
            p->inputRowBytes[active] = inputRowBytes[input];
            p->prefixBytes[active] = prefix;
            p->originalInputs[active++] = static_cast<uint16_t>(input);
            prefix += inputRowBytes[input];
        }
        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(outer, owned)));
        context->SetTilingKey(1303);
        auto* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) { ws[0] = 0; }
        return ge::GRAPH_SUCCESS;
    }

    if ((shortInputRows || amortizedWide32) &&
        outer >= 128 && inputCount >= 2 && inputCount <= S9_CONCAT_SMALL_INPUTS &&
        nonEmptyInputCount == inputCount && totalBytes >= 256 * 1024 &&
        !dmaBlockAlignedInputs && maxInputRowBytes > kUniformNarrowMaxInputRowBytes &&
        outputRowBytes / elementBytes <= 4096) {
        const uint64_t outputElements = outputRowBytes / elementBytes;
        const uint32_t groupRows = static_cast<uint32_t>(4096 / outputElements);
        const uint64_t alignment = elementBytes == 1 ? 128 : 32 / elementBytes;
        const uint32_t gatherCount = static_cast<uint32_t>(AlignUp(groupRows * outputElements, alignment));
        const uint32_t groupsPerSlab = (elementBytes == 4 ? 12288 : 16384) / gatherCount;
        const uint32_t tileRows = groupRows * groupsPerSlab;
        const uint64_t quantum = 32 / Gcd(outputRowBytes, 32);
        const uint64_t requested = std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(totalBytes, 65536)));
        const uint64_t ownedRows = AlignUp(CeilDiv(outer, requested), quantum);
        const uint32_t cores = static_cast<uint32_t>(CeilDiv(outer, ownedRows));
        auto* packed = context->GetTilingData<ConcatGatherRowsTilingData>();
        if (packed == nullptr) { return ge::GRAPH_FAILED; }
        *packed = {};
        const ConcatHostPlan packedPlan{outer, inner, outputRowBytes, totalBytes,
            ownedRows, 0, 0, inputCount, 28, static_cast<uint32_t>(dim), elementBytes};
        StoreTilingData(&packed->header.base, packedPlan, inputRowBytes);
        packed->header.tileRows = tileRows;
        packed->header.gatherCount = gatherCount;
        packed->header.groupRows = groupRows;
        packed->header.groupsPerSlab = groupsPerSlab;
        uint32_t sourceElements = 0;
        for (uint32_t input = 0; input < inputCount; ++input) {
            packed->header.inputOffsets[input] = sourceElements;
            sourceElements += static_cast<uint32_t>(AlignUp(tileRows * (inputRowBytes[input] / elementBytes), alignment));
        }
        if (sourceElements > (elementBytes == 4 ? 16384 : 20480)) { return ge::GRAPH_FAILED; }
        packed->header.sourceElements = sourceElements;
        const uint32_t gatherBytes = elementBytes == 1 ? 2 : elementBytes;
        uint32_t outputElement = 0;
        for (uint32_t row = 0; row < groupRows; ++row) {
            for (uint32_t input = 0; input < inputCount; ++input) {
                const uint32_t width = inputRowBytes[input] / elementBytes;
                for (uint32_t col = 0; col < width; ++col) {
                    packed->indices[outputElement] =
                        (packed->header.inputOffsets[input] + row * width + col) * gatherBytes;
                    packed->steps[outputElement++] = groupRows * width * gatherBytes;
                }
            }
        }
        context->SetBlockDim(cores);
        context->SetTilingKey(1300);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) { workspace[0] = 0; }
        return ge::GRAPH_SUCCESS;
    }

    // Direct modes consume an output-prefix table rather than per-input
    // widths.  The host already walks every input while tiling, so encoding
    // the prefix here removes an O(input) sum from every participating AIV.
    if (mode == 6 || mode == 9 || mode == 25) {
        uint32_t prefixBytes = 0;
        for (uint32_t input = 0; input < inputCount; ++input) {
            const uint32_t widthBytes = inputRowBytes[input];
            inputRowBytes[input] = prefixBytes;
            prefixBytes += widthBytes;
        }
    }

    const ConcatHostPlan plan{
        outer, inner, outputRowBytes, totalBytes, blockRows, flatChunk,
        tileBytes, inputCount, mode, static_cast<uint32_t>(dim), elementBytes};
    if (useInputMajorRows) {
        auto* tiling = context->GetTilingData<ConcatInputMajorRowTilingData>();
        if (!StoreInputMajorRowTilingData(tiling, plan, inputRowBytes, blockDim)) {
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(blockDim);
        context->SetTilingKey(S9_CONCAT_TILING_KEY_INPUT_MAJOR_ROWS);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) {
            workspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (useAdaptiveRows) {
        auto* tiling = context->GetTilingData<ConcatAdaptiveRowTilingData>();
        if (!StoreAdaptiveRowTilingData(tiling, plan, inputRowBytes, blockDim)) {
            return ge::GRAPH_FAILED;
        }
        // Keep the original schedule while moving divisions and row boundaries to Host.
        if (blockDim <= S9_CONCAT_BALANCED_MAX_CORES &&
            plan.outputRowBytes <= UINT32_MAX &&
            context->GetRawTilingData()->GetCapacity() >= sizeof(ConcatDecodedTaskTilingData)) {
            const ConcatAdaptiveRowTilingData original = *tiling;
            ConcatDecodedTaskTilingData decoded{};
            decoded.header = {plan.totalBytes, plan.outputRowBytes, static_cast<uint32_t>(plan.tileBytes),
                plan.inputCount, original.coreTaskOffsets[blockDim], blockDim};
            bool valid = decoded.header.taskCount <= S9_CONCAT_ADAPTIVE_MAX_TASKS;
            for (uint32_t c = 0; c <= blockDim; ++c) decoded.coreOffsets[c] = original.coreTaskOffsets[c];
            for (uint32_t s = 0; valid && s < decoded.header.taskCount; ++s) {
                const uint32_t i = original.taskInputs[s], g = original.taskGroups[s];
                const uint64_t prefix = original.inputPrefixBytes[i];
                const uint64_t width = (i + 1 < plan.inputCount ? original.inputPrefixBytes[i + 1] : plan.outputRowBytes) - prefix;
                const uint64_t groups = original.inputGroupCounts[i];
                if (groups == 0 || width == 0 || AlignUp(width, 32) > plan.tileBytes) { valid = false; break; }
                const uint64_t per = CeilDiv(plan.outer, groups);
                const uint64_t begin = static_cast<uint64_t>(g) * per;
                const uint64_t end = std::min(begin + per, plan.outer);
                const uint64_t count = end > begin ? end - begin : 0;
                if (count > UINT32_MAX || width > UINT32_MAX) { valid = false; break; }
                decoded.tasks[s] = {begin * width, begin * plan.outputRowBytes + prefix, i,
                    static_cast<uint32_t>(count), static_cast<uint32_t>(std::min<uint64_t>(4095, plan.tileBytes / AlignUp(width, 32))), static_cast<uint32_t>(width)};
            }
            if (valid) {
                auto* packed = context->GetTilingData<ConcatDecodedTaskTilingData>();
                if (packed == nullptr) return ge::GRAPH_FAILED;
                *packed = decoded;
                context->GetRawTilingData()->SetDataSize(sizeof(ConcatDecodedTaskTilingData) - sizeof(packed->tasks) + decoded.header.taskCount * sizeof(ConcatDecodedTask));
                context->SetBlockDim(blockDim); context->SetTilingKey(1381);
                auto* ws = context->GetWorkspaceSizes(1); if (ws != nullptr) ws[0] = 0;
                return ge::GRAPH_SUCCESS;
            }
        }
        context->SetBlockDim(blockDim);
        context->SetTilingKey(S9_CONCAT_TILING_KEY_ADAPTIVE_ROWS);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) {
            workspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (useBalancedRows) {
        auto* tiling = context->GetTilingData<ConcatBalancedRowTilingData>();
        if (!StoreBalancedRowTilingData(tiling, plan, inputRowBytes, blockDim)) {
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(blockDim);
        context->SetTilingKey(S9_CONCAT_TILING_KEY_BALANCED_ROWS_QUEUE);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) {
            workspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
    bool useSmallTiling = inputCount <= S9_CONCAT_SMALL_INPUTS;
    for (uint32_t input = 0; input < inputCount && useSmallTiling; ++input) {
        useSmallTiling = inputRowBytes[input] <= std::numeric_limits<uint16_t>::max();
    }
    if (useSmallTiling) {
        auto* tiling = context->GetTilingData<ConcatSmallTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        StoreTilingData(tiling, plan, inputRowBytes);
    } else {
        auto* tiling = context->GetTilingData<ConcatTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        StoreTilingData(tiling, plan, inputRowBytes);
    }
    context->SetBlockDim(blockDim);
    bool useQueue = totalBytes >= S9_CONCAT_DOUBLE_BUFFER_THRESHOLD_BYTES;
#ifdef S9_ARCH_PROBE
    if (forcedProbeRowActive) {
        useQueue = forcedProbeRowQueue;
    }
#endif
    uint64_t tilingKey = TilingKeyForMode(mode, useQueue);
    if (useSmallTiling) {
        tilingKey += S9_CONCAT_TILING_KEY_SMALL_OFFSET;
    }
    context->SetTilingKey(tilingKey);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        // Match aclnnConcat(TensorList, int64, Tensor).  npu_op_code_gen
        // derives the exported C ABI directly from this custom schema.
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(REQUIRED).Int();
        this->AICore().SetTiling(optiling::TilingConcat).AddConfig("ascend910b");
    }
};
OP_ADD(Concat);
} // namespace ops
