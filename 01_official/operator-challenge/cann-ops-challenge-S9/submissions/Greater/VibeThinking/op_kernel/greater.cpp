#include "kernel_operator.h"
using namespace AscendC;

#define S9G_INLINE __aicore__ inline
#include "greater_tiling.h"
#include "runtime.h"
#include "flat_paths.h"
#include "row_paths.h"

__global__ __aicore__ void greater(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                                   GM_ADDR workspace,GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GreaterRows);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 0 || TILING_KEY_VAR == 1",GreaterExact);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2",GreaterOneShot);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 3 || TILING_KEY_VAR == 4",GreaterFlat);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6",GreaterPairF32);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 7 || TILING_KEY_VAR == 8",GreaterLinearRows);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 16 || TILING_KEY_VAR == 17 || TILING_KEY_VAR == 18 || TILING_KEY_VAR == 19 || TILING_KEY_VAR == 20",
        GreaterExact);

    InitSocState();

    if constexpr (GreaterTypeInfo<DTYPE_X1>::index==2) {
        if (TILING_KEY_IS(16)) {
            RunFixedHalfPacket<128>(x1,x2,output);
            return;
        }
        if (TILING_KEY_IS(17)) {
            RunFixedHalfPacket<256>(x1,x2,output);
            return;
        }
        if (TILING_KEY_IS(18)) {
            RunFixedHalfPacket<512>(x1,x2,output);
            return;
        }
        if (TILING_KEY_IS(19)) {
            RunFixedHalfPacket<1024>(x1,x2,output);
            return;
        }
        if (TILING_KEY_IS(20)) {
            RunFixedHalfPacket<2048>(x1,x2,output);
            return;
        }
    }

    if (TILING_KEY_IS(0)) return;

    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterExact,plan,tiling);
        RunSmallTensor<DTYPE_X1,true>(x1,x2,output,plan.perCore);
        return;
    }
    if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterOneShot,plan,tiling);
        RunSmallTensor<DTYPE_X1,false>(
            x1,x2,output,plan.perCore,plan.total);
        return;
    }
    if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterFlat,plan,tiling);
        if constexpr (GreaterTypeInfo<DTYPE_X1>::index==4) {
            GreaterInt8Stream int8;
            if (int8.Run(x1,x2,output,plan)) return;
        }
        GreaterDensePipeline<DTYPE_X1,false> dense;
        dense.Run(x1,x2,output,plan);
        return;
    }
    if (TILING_KEY_IS(4)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterFlat,plan,tiling);
        GreaterDensePipeline<DTYPE_X1,true> scalar;
        scalar.Run(x1,x2,output,plan);
        return;
    }
    if (TILING_KEY_IS(5)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterRows,plan,tiling);
        GreaterBroadcastPath<DTYPE_X1> rows;
        rows.Run(x1,x2,output,plan);
        return;
    }
    if (TILING_KEY_IS(6)) {
        if constexpr (GreaterTypeInfo<DTYPE_X1>::index==0) {
            GET_TILING_DATA_WITH_STRUCT(GreaterPairF32,plan,tiling);
            GreaterFp32PairPipeline pairs;
            pairs.Run(x1,x2,output,plan);
        }
        return;
    }
    if (TILING_KEY_IS(7)) {
        if constexpr (GreaterTypeInfo<DTYPE_X1>::index==3) {
            GET_TILING_DATA_WITH_STRUCT(GreaterLinearRows,plan,tiling);
            if(plan.side==1)RunGreaterGeometry<1>(x1,x2,output,plan);
            else RunGreaterGeometry<2>(x1,x2,output,plan);
        }
    }
    if(TILING_KEY_IS(8)){
        if constexpr(GreaterTypeInfo<DTYPE_X1>::index==1){
            GET_TILING_DATA_WITH_STRUCT(GreaterLinearRows,plan,tiling);
            if(plan.side==1)RunGreaterBf16Geometry<1>(x1,x2,output,plan);
            else RunGreaterBf16Geometry<2>(x1,x2,output,plan);
        }
    }

}
