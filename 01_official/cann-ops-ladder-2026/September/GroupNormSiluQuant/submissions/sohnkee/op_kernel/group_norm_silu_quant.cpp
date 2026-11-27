/*!
 * \file group_norm_silu_quant.cpp
 * \brief GroupNormSiluQuant 算子 kernel 入口
 */

#include "group_norm_silu_quant.h"

enum class GroupNormSiluQuantTilingKey : uint32_t
{
    TILING_KEY_GROUPNORMSILUQUANT_MODE_0 = 0,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_1 = 1,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_2 = 2,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_3 = 3,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_4 = 4,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_5 = 5,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_6 = 6,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_7 = 7,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_8 = 8,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_9 = 9,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_10 = 10,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_12 = 12,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_14 = 14,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_16 = 16,
};

template <uint32_t schMode>
__global__ __aicore__ void group_norm_silu_quant(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(GroupNormSiluQuantTilingData);
    if constexpr (schMode == GROUPNORMSILUQUANT_TPL_SCH_MODE_18) {
        const auto* src = reinterpret_cast<const __gm__ GroupNormSiluQuantTilingData*>(tiling);
        const int64_t workers = src->workerNum;
        if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(workers)) return;
        GroupNormSiluQuantTilingData tiny;
        tiny.numGroups = src->numGroups;
        tiny.channelsPerGroup = src->channelsPerGroup;
        tiny.spatial = src->spatial;
        tiny.groupElements = src->groupElements;
        tiny.epsilon = src->epsilon;
        AscendC::InitSocState();
        NsGroupNormSiluQuant::Fp16TinyGroupNormSiluQuant op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tiny);
        op.Process();
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(GroupNormSiluQuantTilingData, tilingData, tiling);
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData.workerNum)) {
        return;
    }
    if constexpr ((schMode >= 32 && schMode <= 41) || schMode == 48) {
        constexpr int32_t elements = schMode == 48 ? 0 : (512 << ((schMode - 32) / 2));
        constexpr bool padded = schMode != 48 && (schMode & 1U) != 0;
        NsGroupNormSiluQuant::ExactResidentFp32GroupNormSiluQuant<elements, padded> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
        return;
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_0)) {
        if (tilingData.tileElements == 4096 &&
            ((tilingData.spatial + 31) / 32 * 32) * tilingData.channelsPerGroup <= 4096) {
            NsGroupNormSiluQuant::FixedResidentGroupNormSiluQuant<half> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<half> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_1)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_2)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_3)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, true, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_4)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<half, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_5)) {
        if (tilingData.tileElements == 4096 && tilingData.groupElements <= 4096 &&
            tilingData.spatial % 32 == 0) {
            NsGroupNormSiluQuant::FixedResidentGroupNormSiluQuant<float> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<float, false, false, true> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_6)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, false, true, false, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_7)) {
        // Host routes every eligible FP16 tiny group to key18. The remaining
        // key7 owners use the original full resident or generic fallback.
        if (tilingData.tileElements == 4096) {
            NsGroupNormSiluQuant::FixedResidentGroupNormSiluQuant<half> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<half, false, false, true> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_8)) {
        // EnableDoubleBuffer distinguishes the short full-resident path at
        // compile time; the resident input itself still owns one queue slot.
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, true, false, false, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_9)) {
        if (tilingData.tileElements == 4096 &&
            ((tilingData.spatial + 31) / 32 * 32) * tilingData.channelsPerGroup <= 4096) {
            NsGroupNormSiluQuant::FixedResidentGroupNormSiluQuant<half> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<half, false, false, false, false, true> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_10)) {
        if (tilingData.tileElements == 4096 && tilingData.groupElements <= 4096 &&
            tilingData.spatial % 32 == 0) {
            NsGroupNormSiluQuant::FixedResidentGroupNormSiluQuant<float> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<float, false, false, true, false, true, true> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_12)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, true, false, false, true, false, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_14)) {
        if (tilingData.groupElements <= 64 && tilingData.channelsPerGroup <= 4 &&
            tilingData.spatial % 8 == 0 && tilingData.activateSilu != 0 &&
            tilingData.workerNum == tilingData.totalGroups) {
            NsGroupNormSiluQuant::Fp32TinyGroupNormSiluQuant op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        } else {
            NsGroupNormSiluQuant::GroupNormSiluQuant<float, false, false, false, false, false, true> op;
            op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_16)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float, false, true, false, true, true> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
}
