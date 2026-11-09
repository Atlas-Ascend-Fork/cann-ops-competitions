#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

namespace {
#include "paths/config.h"

template <typename T>
class GreaterKernel {
public:
#include "paths/dispatch.h"

private:
#include "paths/buffers.h"
#include "paths/bf16_broadcast.h"
#include "paths/bf16_flat_gather.h"
#include "paths/dense.h"
#include "paths/channel_scalar.h"
#include "paths/channel_row.h"
#include "paths/int32_row.h"
#include "paths/generic.h"

    TPipe* pipe = nullptr;
    GlobalTensor<T> selfGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<uint8_t> outGm;
    TQue<QuePosition::VECIN, 1> selfQueue;
    TQue<QuePosition::VECIN, 1> otherQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 2> cScalarPipelineInQueue;
    TQue<QuePosition::VECOUT, 2> cScalarPipelineOutQueue;
    TQue<QuePosition::VECIN, 2> cRowPipelineInQueue;
    TQue<QuePosition::VECOUT, 2> cRowPipelineOutQueue;
    TQue<QuePosition::VECIN, 2> densePipelineSelfQueue;
    TQue<QuePosition::VECIN, 2> densePipelineOtherQueue;
    TQue<QuePosition::VECOUT, 2> densePipelineOutQueue;
    TQue<QuePosition::VECIN, 2> int32RowOtherQueue;
    TQue<QuePosition::VECOUT, 2> int32RowOutQueue;
    TQue<QuePosition::VECIN, 2> bf16CRowSelfQueue;
    TQue<QuePosition::VECIN, 2> bf16CRowOtherQueue;
    TQue<QuePosition::VECOUT, 2> bf16CRowOutQueue;
    TQue<QuePosition::VECIN, 2> bf16PipelineSelfQueue;
    TQue<QuePosition::VECIN, 2> bf16PipelineOtherQueue;
    TQue<QuePosition::VECOUT, 2> bf16PipelineOutQueue;
    TBuf<TPosition::VECCALC> selfFloatBuf;
    TBuf<TPosition::VECCALC> otherFloatBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> broadcastBuf;
    TBuf<TPosition::VECCALC> bankPadBuf;
    TBuf<TPosition::VECCALC> bankTailPadBuf;
    TBuf<TPosition::VECCALC> bankScratchPadBuf;
    TBuf<TPosition::VECCALC> bankMaskPadBuf;
    TBuf<TPosition::VECCALC> int32RowGatherPatternBuf;
    uint32_t totalSize = 0U;
    uint32_t blockElems = 0U;
    uint32_t totalBlocks = 0U;
    uint32_t tileElems = 0U;
    uint32_t startBlock = 0U;
    uint32_t endBlock = 0U;
    uint32_t startElem = 0U;
    uint32_t endElem = 0U;
    bool sameShape = false;
    bool selfDense = false;
    bool otherDense = false;
    bool selfScalar = true;
    bool otherScalar = true;
    bool cScalarOther = false;
    bool cRowOther = false;
    uint32_t outDims[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t selfDims[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t otherDims[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t outStrides[kDimNum] = {1U, 1U, 1U, 1U, 1U};
    uint32_t selfStrides[kDimNum] = {0U, 0U, 0U, 0U, 0U};
    uint32_t otherStrides[kDimNum] = {0U, 0U, 0U, 0U, 0U};
};

#include "paths/type_compute.h"

template <typename T>
__aicore__ inline void RunGreater(GM_ADDR self, GM_ADDR other, GM_ADDR output,
                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GreaterKernel<T> kernel;
    kernel.Init(self, other, output, workspace, tiling, &pipe);
    kernel.Process();
}
}  // namespace

extern "C" __global__ __aicore__ void greater(GM_ADDR self, GM_ADDR other,
                                              GM_ADDR output, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        RunGreater<float>(self, other, output, workspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunGreater<half>(self, other, output, workspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        RunGreater<bfloat16_t>(self, other, output, workspace, tiling);
    } else if (TILING_KEY_IS(3)) {
        RunGreater<int32_t>(self, other, output, workspace, tiling);
    } else if (TILING_KEY_IS(4)) {
        RunGreater<int8_t>(self, other, output, workspace, tiling);
    }
}
