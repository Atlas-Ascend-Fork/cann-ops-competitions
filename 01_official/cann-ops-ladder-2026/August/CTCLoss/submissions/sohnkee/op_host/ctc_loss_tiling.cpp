/*
 * Copyright (c) 2026.
 * CtcLoss host tiling: batch parallelism, FP32 rolling alpha and class slicing.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/ctc_loss_tiling_data.h"
#include "../op_kernel/ctc_loss_tiling_key.h"

#include <algorithm>
#include <cstring>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

namespace {
constexpr int64_t T_DIM = 0;
constexpr int64_t N_DIM = 1;
constexpr int64_t C_DIM = 2;
constexpr int64_t S_DIM = 1;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t VECTOR_ALIGN = 64;
constexpr int64_t PAIR_MIN_TIME = 2 * BLOCK_BYTES / FP32_BYTES;
constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;

ge::graphStatus CtcLossTilingFunc(gert::TilingContext* context)
{
    auto log_probs = context->GetInputShape(0);
    auto targets = context->GetInputShape(1);
    auto input_lengths = context->GetInputShape(2);
    auto target_lengths = context->GetInputShape(3);
    auto log_desc = context->GetInputDesc(0);
    auto target_desc = context->GetInputDesc(1);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, log_probs);
    OP_CHECK_NULL_WITH_CONTEXT(context, targets);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_lengths);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_lengths);
    OP_CHECK_NULL_WITH_CONTEXT(context, log_desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::Shape& log_shape = log_probs->GetStorageShape();
    const gert::Shape& target_shape = targets->GetStorageShape();
    const gert::Shape& input_len_shape = input_lengths->GetStorageShape();
    const gert::Shape& target_len_shape = target_lengths->GetStorageShape();
    OP_CHECK_IF(log_shape.GetDimNum() != 3, OP_LOGE(context, "log_probs must be rank 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(target_shape.GetDimNum() != 2, OP_LOGE(context, "targets must be rank 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_len_shape.GetDimNum() != 1 || target_len_shape.GetDimNum() != 1,
                OP_LOGE(context, "length tensors must be rank 1"), return ge::GRAPH_FAILED);

    const int64_t time_step = log_shape.GetDim(T_DIM);
    const int64_t batch_size = log_shape.GetDim(N_DIM);
    const int64_t num_classes = log_shape.GetDim(C_DIM);
    const int64_t target_stride = target_shape.GetDim(S_DIM);
    OP_CHECK_IF(time_step <= 0 || batch_size <= 0 || num_classes < 2 || target_stride <= 0,
                OP_LOGE(context, "invalid static shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(target_shape.GetDim(0) != batch_size || input_len_shape.GetDim(0) != batch_size ||
                    target_len_shape.GetDim(0) != batch_size,
                OP_LOGE(context, "batch dimensions do not match"), return ge::GRAPH_FAILED);

    const int64_t* blank_ptr = attrs->GetAttrPointer<int64_t>(0);
    const char* reduction_ptr = attrs->GetAttrPointer<char>(1);
    const bool* zero_infinity_ptr = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, blank_ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction_ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, zero_infinity_ptr);
    OP_CHECK_IF(*blank_ptr < 0 || *blank_ptr >= num_classes,
                OP_LOGE(context, "blank is outside [0, C)"), return ge::GRAPH_FAILED);

    int64_t reduction = REDUCTION_NONE;
    if (std::strcmp(reduction_ptr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else if (std::strcmp(reduction_ptr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else {
        OP_CHECK_IF(std::strcmp(reduction_ptr, "none") != 0,
                    OP_LOGE(context, "reduction must be none, mean or sum"), return ge::GRAPH_FAILED);
    }

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
    auto platform = platform_ascendc::PlatformAscendC(platform_info);
    const int64_t aiv_core_num = platform.GetCoreNumAiv();
    uint64_t ub_size_u64 = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_u64);
    OP_CHECK_IF(aiv_core_num <= 0 || ub_size_u64 == 0, OP_LOGE(context, "invalid platform resources"),
                return ge::GRAPH_FAILED);
    const int64_t worker_num = std::min(batch_size, aiv_core_num);
    const bool pad_launch = batch_size < 8;

    const int64_t alpha_align = CeilAlign(2 * target_stride + 1, VECTOR_ALIGN);
    const int64_t target_align = CeilAlign(target_stride, VECTOR_ALIGN);
    const int64_t reduction_align = CeilAlign(batch_size, VECTOR_ALIGN);
    const int64_t target_bytes = target_desc->GetDataType() == ge::DT_INT64 ? 8 : 4;
    const int64_t prob_bytes = log_desc->GetDataType() == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t max_tasks_per_core = (batch_size + worker_num - 1) / worker_num;
    const int64_t result_bytes = CeilAlign(max_tasks_per_core, BLOCK_BYTES / FP32_BYTES) * FP32_BYTES;
    const int64_t reduction_bytes = reduction != REDUCTION_NONE && batch_size > 1 ?
                                        reduction_align * FP32_BYTES : 0;

    // Four alpha-sized FP32 tensors plus emission, s-1 and legal s-2 Gather indices.
    // Multi-slice emission temporarily reuses the alpha candidate tensors.
    const int64_t fixed_ub_bytes = 7 * alpha_align * FP32_BYTES + target_align * target_bytes +
                                   reduction_bytes + 3 * BLOCK_BYTES +
                                   std::max<int64_t>(0, result_bytes - BLOCK_BYTES);
    const int64_t available_for_slice = static_cast<int64_t>(ub_size_u64) - fixed_ub_bytes;
    const int64_t max_slice_align = FloorAlign(available_for_slice / prob_bytes, VECTOR_ALIGN);
    OP_CHECK_IF(max_slice_align < VECTOR_ALIGN,
                OP_LOGE(context, "UB is insufficient for the maximum target shape"), return ge::GRAPH_FAILED);
    const int64_t slice_length = std::min(num_classes, max_slice_align);
    const int64_t slice_align = CeilAlign(slice_length, VECTOR_ALIGN);
    const int64_t slice_count = (num_classes + slice_length - 1) / slice_length;
    const int64_t slice_tail = num_classes - (slice_count - 1) * slice_length;
    // E2: 32-element row alignment for the Ping-Pong slot (128B/64B >= 32B block
    // granularity) instead of 64, so more large-C single-slice cases fit two rows.
    const int64_t prob_row_align = CeilAlign(num_classes, static_cast<int64_t>(32));
    const bool prob_ping_pong = time_step > 2 && slice_count == 1 &&
                                2 * prob_row_align <= max_slice_align;
    // Aggregate complete, 32-byte-aligned rows into one two-dimensional DMA. The two UB bands retain
    // Ping-Pong overlap while reducing descriptor and event issue rate. Misaligned rows keep the
    // established one-row pipeline.
    const int64_t row_bytes = num_classes * prob_bytes;
    const int64_t max_band_rows = max_slice_align / (2 * prob_row_align);
    const bool band_stride_representable = row_bytes <= static_cast<int64_t>(UINT32_MAX) &&
                                           batch_size - 1 <=
                                               (static_cast<int64_t>(UINT32_MAX) - row_bytes) / row_bytes;
    const int64_t candidate_band_rows = std::min<int64_t>(max_band_rows, 4095);
    const bool prob_band_eligible = prob_ping_pong && row_bytes % BLOCK_BYTES == 0 &&
                                    band_stride_representable && candidate_band_rows >= 4 &&
                                    time_step - 1 > candidate_band_rows;
    const int64_t prob_band_rows = prob_band_eligible ? candidate_band_rows : 0;
    const bool pair_resource = (log_desc->GetDataType() == ge::DT_FLOAT16 ||
                                log_desc->GetDataType() == ge::DT_FLOAT) &&
                               target_stride >= 2 && slice_count == 1 && prob_ping_pong;
    const bool pair_mode = pair_resource && time_step >= PAIR_MIN_TIME && batch_size <= aiv_core_num / 2;
    const bool asym_pair_mode = pair_resource && time_step >= VECTOR_ALIGN && batch_size > aiv_core_num / 2 &&
                                batch_size < aiv_core_num;
    const int64_t block_dim = pair_mode ? 2 * batch_size :
                              (asym_pair_mode ? aiv_core_num :
                               (pad_launch ? std::min<int64_t>(8, aiv_core_num) : worker_num));
    const int64_t reverse_worker_num = asym_pair_mode ? aiv_core_num - batch_size : 0;

    CtcLossTilingData* tiling = context->GetTilingData<CtcLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto* raw_tiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw_tiling);
    raw_tiling->SetDataSize(sizeof(CtcLossTilingData));
    tiling->timeStep = time_step;
    tiling->batchSize = batch_size;
    tiling->numClasses = num_classes;
    tiling->maxTargetLength = target_stride;
    tiling->targetStride = target_stride;
    tiling->alphaAlign = alpha_align;
    tiling->targetAlign = target_align;
    tiling->sliceLength = slice_length;
    tiling->sliceAlign = slice_align;
    tiling->sliceCount = slice_count;
    tiling->sliceTail = slice_tail;
    tiling->tasksPerCore = asym_pair_mode ? batch_size / reverse_worker_num : batch_size / worker_num;
    tiling->tailCoreCount = asym_pair_mode ? batch_size % reverse_worker_num : batch_size % worker_num;
    tiling->reduction = reduction;
    tiling->blank = *blank_ptr;
    tiling->zeroInfinity = *zero_infinity_ptr ? 1 : 0;
    tiling->probPingPong = prob_ping_pong ? 1 : 0;
    tiling->probRowAlign = prob_row_align;
    tiling->probBandRows = prob_band_rows;
    tiling->pairMode = pair_mode ? 1 : (asym_pair_mode ? 2 : 0);

    uint64_t tiling_key = 0;
    if (log_desc->GetDataType() == ge::DT_FLOAT16) {
        tiling_key = target_desc->GetDataType() == ge::DT_INT64 ? CTCLOSS_TPL_SCH_MODE_1 : CTCLOSS_TPL_SCH_MODE_0;
    } else {
        tiling_key = target_desc->GetDataType() == ge::DT_INT64 ? CTCLOSS_TPL_SCH_MODE_3 : CTCLOSS_TPL_SCH_MODE_2;
    }
    if (pair_mode) {
        tiling_key += CTCLOSS_TPL_SCH_MODE_8;
    } else if (asym_pair_mode) {
        tiling_key += CTCLOSS_TPL_SCH_MODE_12;
    } else if (reduction == REDUCTION_NONE || batch_size == 1) {
        tiling_key += CTCLOSS_TPL_SCH_MODE_4;
    }
    if (tiling_key == CTCLOSS_TPL_SCH_MODE_0) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_0));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_1) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_1));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_2) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_2));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_3) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_3));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_4) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_4));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_5) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_5));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_6) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_6));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_7) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_7));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_8) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_8));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_9) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_9));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_10) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_10));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_11) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_11));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_12) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_12));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_13) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_13));
    } else if (tiling_key == CTCLOSS_TPL_SCH_MODE_14) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_14));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(CTCLOSS_TPL_SCH_MODE_15));
    }
    context->SetBlockDim(block_dim);
    if (pair_mode || asym_pair_mode || (reduction != REDUCTION_NONE && batch_size > 1)) {
        context->SetScheduleMode(1);
    }

    size_t* workspace_sizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_sizes);
    const int64_t temp_loss_bytes = CeilAlign(batch_size * FP32_BYTES, BLOCK_BYTES);
    const int64_t midpoint_bytes = pair_mode || asym_pair_mode ? batch_size * alpha_align * FP32_BYTES : 0;
    workspace_sizes[0] = platform.GetLibApiWorkSpaceSize() + temp_loss_bytes + midpoint_bytes;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForCtcLoss(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct CtcLossCompileInfo {};
} // namespace

IMPL_OP_OPTILING(CtcLoss).Tiling(CtcLossTilingFunc).TilingParse<CtcLossCompileInfo>(TilingParseForCtcLoss);

} // namespace optiling
