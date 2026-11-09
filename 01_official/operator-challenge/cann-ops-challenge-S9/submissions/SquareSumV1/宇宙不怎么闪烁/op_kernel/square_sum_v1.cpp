#define K_MAX_SHAPE_DIM 0
#include "square_sum_kernel.h"

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    // Pure Vector kernels using hardware SyncAll: CANN 8.5 MIX AIV:AIC 1:0.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(0)) {
        RunSquareSum<float>(input, output, workspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunSquareSum<half>(input, output, workspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        RunSquareSum<bfloat16_t>(input, output, workspace, tiling);
    }
}
