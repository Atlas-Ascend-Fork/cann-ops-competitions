/*!
 * \file inplace_update.cpp
 * \brief InplaceUpdate 算子 kernel 入口
 */

#include "inplace_update.h"

enum class InplaceUpdateTilingKey : uint32_t
{
    TILING_KEY_INPLACEUPDATE_MODE_0 = 0,
    TILING_KEY_INPLACEUPDATE_MODE_1 = 1,
    TILING_KEY_INPLACEUPDATE_MODE_2 = 2,
    TILING_KEY_INPLACEUPDATE_MODE_3 = 3,
    TILING_KEY_INPLACEUPDATE_MODE_4 = 4,
    TILING_KEY_INPLACEUPDATE_MODE_5 = 5,
    TILING_KEY_INPLACEUPDATE_MODE_6 = 6,
    TILING_KEY_INPLACEUPDATE_MODE_7 = 7,
    TILING_KEY_INPLACEUPDATE_MODE_8 = 8,
    TILING_KEY_INPLACEUPDATE_MODE_9 = 9,
    TILING_KEY_INPLACEUPDATE_MODE_10 = 10,
    TILING_KEY_INPLACEUPDATE_MODE_11 = 11,
    TILING_KEY_INPLACEUPDATE_MODE_12 = 12,
    TILING_KEY_INPLACEUPDATE_MODE_13 = 13,
    TILING_KEY_INPLACEUPDATE_MODE_14 = 14,
    TILING_KEY_INPLACEUPDATE_MODE_15 = 15,
    TILING_KEY_INPLACEUPDATE_MODE_16 = 16,
    TILING_KEY_INPLACEUPDATE_MODE_17 = 17,
    TILING_KEY_INPLACEUPDATE_MODE_18 = 18,
    TILING_KEY_INPLACEUPDATE_MODE_19 = 19,
    TILING_KEY_INPLACEUPDATE_MODE_20 = 20,
};

template <uint32_t schMode>
__global__ __aicore__ void inplace_update(GM_ADDR x, GM_ADDR i, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(InplaceUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(InplaceUpdateTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_0)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData);
        op.ProcessWithRowFilter();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_1)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData, true);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_2)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData);
        op.ProcessTinySingleCore();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_3)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.InitSmall(x, i, v, y, &tilingData);
        op.ProcessSmallDirect();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_4)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.InitLargeIndex(x, i, v, y, &tilingData);
        op.ProcessLargeIndexRowAligned<false>();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_5)) {
        NsInplaceUpdate::ProcessSmallStaticFastPath(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_6)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData);
        op.ProcessWideRowBalanced();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_7)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData, true);
        op.ProcessSingleIndexScalar();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_8)) {
        NsInplaceUpdate::ProcessTinyStaticIndex2NarrowFastPath(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_9)) {
        NsInplaceUpdate::ProcessPackedStaticIndex3FastPath(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_10)) {
        NsInplaceUpdate::ProcessTinyStaticIndex3NarrowFastPath<false>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_11)) {
        NsInplaceUpdate::ProcessTinyStaticIndex3NarrowFastPath<true>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_12)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.InitLargeIndex(x, i, v, y, &tilingData);
        op.ProcessLargeIndexRowAligned<true>();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_13)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.Init(x, i, v, y, &tilingData);
        op.ProcessWideRowBalancedIndex4();
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_14)) {
        NsInplaceUpdate::ProcessTinyStaticIndex3UbComposeFastPath<3>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_15)) {
        NsInplaceUpdate::ProcessTinyStaticIndex3UbComposeFastPath<4>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_16)) {
        NsInplaceUpdate::ProcessTinyStaticIndex2UbComposeFastPath<16>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_17)) {
        NsInplaceUpdate::ProcessTinyStaticIndex2UbComposeFastPath<24>(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_18)) {
        NsInplaceUpdate::ProcessTinyStaticIndex1UbComposeFastPath(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_19)) {
        NsInplaceUpdate::ProcessTinyStaticIndex1UbCompose8BFastPath(x, i, v, y, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(InplaceUpdateTilingKey::TILING_KEY_INPLACEUPDATE_MODE_20)) {
        NsInplaceUpdate::InplaceUpdate op;
        op.InitLargeIndex(x, i, v, y, &tilingData);
        op.ProcessLargeIndexRowAlignedStatic2048();
    }
}
