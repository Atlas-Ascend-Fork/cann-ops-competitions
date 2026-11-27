/*
 * Copyright (c) 2026.
 * CTCLoss forward kernel for Ascend 910B.
 */

#ifndef CTCLOSS_H
#define CTCLOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "ctc_loss_tiling_data.h"
#include "ctc_loss_tiling_key.h"

#include <type_traits>

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace NsCtcLoss {

using namespace AscendC;

namespace {
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t VECTOR_ALIGN = 64;
constexpr int64_t PAIR_MIN_TIME = 2 * BLOCK_BYTES / sizeof(float);
constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr uint16_t MIDPOINT_SYNC_FLAG = 8;
constexpr uint16_t LOSS_SYNC_FLAG = 9;
constexpr float FLOAT_LOWEST = -3.402823466e+38F;

template <typename T1, typename T2>
__aicore__ inline T1 AlignUp(T1 value, T2 align)
{
    return align == 0 ? 0 : (value + align - 1) / align * align;
}
} // namespace

template <typename TProb, typename TTarget, bool PAIR_COMPILED, bool ASYM_PAIR_COMPILED>
class CtcLoss {
public:
    __aicore__ inline CtcLoss() {}

    __aicore__ inline void Init(GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths,
                                GM_ADDR target_lengths, GM_ADDR neg_log_likelihood, GM_ADDR workspace,
                                const CtcLossTilingData* tiling_data)
    {
        timeStep_ = tiling_data->timeStep;
        batchSize_ = tiling_data->batchSize;
        numClasses_ = tiling_data->numClasses;
        maxTargetLength_ = tiling_data->maxTargetLength;
        targetStride_ = tiling_data->targetStride;
        alphaAlign_ = tiling_data->alphaAlign;
        targetAlign_ = tiling_data->targetAlign;
        sliceLength_ = tiling_data->sliceLength;
        sliceAlign_ = tiling_data->sliceAlign;
        sliceCount_ = tiling_data->sliceCount;
        sliceTail_ = tiling_data->sliceTail;
        tasksPerCore_ = tiling_data->tasksPerCore;
        tailCoreCount_ = tiling_data->tailCoreCount;
        reduction_ = tiling_data->reduction;
        blank_ = tiling_data->blank;
        zeroInfinity_ = tiling_data->zeroInfinity;
        probPingPong_ = tiling_data->probPingPong;
        probRowAlign_ = tiling_data->probRowAlign;
        probBandRows_ = tiling_data->probBandRows;

        const int64_t block_idx = GetBlockIdx();
        if constexpr (PAIR_COMPILED) {
            taskCount_ = block_idx < 2 * batchSize_ ? 1 : 0;
            firstBatch_ = block_idx / 2;
            reverseCore_ = (block_idx & 1) != 0;
        } else if constexpr (ASYM_PAIR_COMPILED) {
            if (block_idx < batchSize_) {
                taskCount_ = 1;
                firstBatch_ = block_idx;
                reverseCore_ = false;
            } else {
                const int64_t reverse_idx = block_idx - batchSize_;
                reverseCore_ = true;
                if (reverse_idx < tailCoreCount_) {
                    taskCount_ = tasksPerCore_ + 1;
                    firstBatch_ = reverse_idx * taskCount_;
                } else {
                    taskCount_ = tasksPerCore_;
                    firstBatch_ = reverse_idx * tasksPerCore_ + tailCoreCount_;
                }
            }
        } else if (block_idx >= batchSize_) {
            taskCount_ = 0;
            firstBatch_ = 0;
        } else if (block_idx < tailCoreCount_) {
            taskCount_ = tasksPerCore_ + 1;
            firstBatch_ = block_idx * taskCount_;
        } else {
            taskCount_ = tasksPerCore_;
            firstBatch_ = block_idx * tasksPerCore_ + tailCoreCount_;
        }

        logProbsGm_.SetGlobalBuffer((__gm__ TProb*)log_probs, timeStep_ * batchSize_ * numClasses_);
        targetsGm_.SetGlobalBuffer((__gm__ TTarget*)targets, batchSize_ * targetStride_);
        inputLengthsGm_.SetGlobalBuffer((__gm__ int32_t*)input_lengths, batchSize_);
        targetLengthsGm_.SetGlobalBuffer((__gm__ int32_t*)target_lengths, batchSize_);
        outputGm_.SetGlobalBuffer((__gm__ float*)neg_log_likelihood,
                                  reduction_ == REDUCTION_NONE ? batchSize_ : 1);
        tempLossGm_.SetGlobalBuffer((__gm__ float*)workspace, batchSize_);
        const int64_t temp_loss_align = AlignUp(batchSize_, BLOCK_BYTES / sizeof(float));
        if constexpr (PAIR_COMPILED || ASYM_PAIR_COMPILED) {
            midpointGm_.SetGlobalBuffer((__gm__ float*)workspace + temp_loss_align, batchSize_ * alphaAlign_);
        }

        const int64_t target_bytes = timeStep_ > 1 ? targetAlign_ * sizeof(TTarget) : BLOCK_BYTES;
        const int64_t alpha_aux_bytes = timeStep_ > 1 ? 3 * alphaAlign_ * sizeof(float) : 0;
        const int64_t alpha_b_bytes = timeStep_ > 1 ? alphaAlign_ * sizeof(float) : 0;
        const int64_t gather_index_bytes = timeStep_ > 1 ? alphaAlign_ * sizeof(uint32_t) : 0;
        const int64_t transition_index_bytes = timeStep_ > 1 ? 2 * alphaAlign_ * sizeof(uint32_t) : 0;
        const int64_t prob_storage = timeStep_ > 1 ?
                                         (probBandRows_ > 1 ? 2 * probBandRows_ * probRowAlign_ :
                                          (probPingPong_ != 0 ? 2 * probRowAlign_ : sliceAlign_)) : 0;
        const int64_t prob_slice_bytes = prob_storage * sizeof(TProb);
        constexpr int64_t prob_scalar_bytes = 2 * BLOCK_BYTES;
        const int64_t max_task_count = tasksPerCore_ + (tailCoreCount_ > 0 ? 1 : 0);
        const int64_t result_bytes = AlignUp(max_task_count, BLOCK_BYTES / sizeof(float)) * sizeof(float);
        const int64_t reduction_bytes = reduction_ != REDUCTION_NONE && batchSize_ > 1 ?
                                            AlignUp(batchSize_, VECTOR_ALIGN) * sizeof(float) : 0;
        const int64_t arena_bytes = target_bytes + alpha_aux_bytes + alpha_b_bytes + gather_index_bytes +
                                    transition_index_bytes + prob_slice_bytes + prob_scalar_bytes + result_bytes +
                                    reduction_bytes;
        pipe_.InitBuffer(arenaBuf_, arena_bytes);

        LocalTensor<uint8_t> arena = arenaBuf_.Get<uint8_t>();
        int64_t arena_offset = 0;
        targets_ = arena[arena_offset].ReinterpretCast<TTarget>();
        arena_offset += target_bytes;
        if (timeStep_ > 1) {
            alphaA_ = arena[arena_offset].ReinterpretCast<float>();
            arena_offset += alpha_aux_bytes;
            alphaB_ = arena[arena_offset].ReinterpretCast<float>();
            arena_offset += alpha_b_bytes;
            aux_ = alphaA_[alphaAlign_];
            max_ = alphaA_[2 * alphaAlign_];
            gatherIndex_ = arena[arena_offset].ReinterpretCast<uint32_t>();
            arena_offset += gather_index_bytes;
            transitionIndex_ = arena[arena_offset].ReinterpretCast<uint32_t>();
            arena_offset += transition_index_bytes;
            transitionIndex2_ = transitionIndex_[alphaAlign_];
            probSlice_ = arena[arena_offset].ReinterpretCast<TProb>();
            arena_offset += prob_slice_bytes;
        }
        probScalar_ = arena[arena_offset].ReinterpretCast<TProb>();
        arena_offset += prob_scalar_bytes;
        result_ = arena[arena_offset].ReinterpretCast<float>();
        arena_offset += result_bytes;
        if (reduction_ != REDUCTION_NONE && batchSize_ > 1) {
            reductionTensor_ = arena[arena_offset].ReinterpretCast<float>();
        }

    }

    __aicore__ inline void Process()
    {
        if constexpr (PAIR_COMPILED) {
            ProcessPaired();
            return;
        } else if constexpr (ASYM_PAIR_COMPILED) {
            ProcessAsymmetricPaired();
            return;
        }
        GlobalTensor<float> loss_dst = reduction_ == REDUCTION_NONE || batchSize_ == 1 ? outputGm_ : tempLossGm_;
        for (int64_t task = 0; task < taskCount_; ++task) {
            const int64_t batch_id = firstBatch_ + task;
            const int64_t target_length = targetLengthsGm_.GetValue(batch_id);
            float loss = ComputeBatch(batch_id, inputLengthsGm_.GetValue(batch_id), target_length);
            if (reduction_ == REDUCTION_MEAN) {
                float divisor = target_length > 0 ? static_cast<float>(target_length) : 1.0f;
                if (batchSize_ > 1) {
                    divisor *= static_cast<float>(batchSize_);
                }
                loss /= divisor;
            }
            result_.SetValue(task, loss);
        }
        if (taskCount_ > 0) {
            SToMte3();
            DataCopyParams copy_params{1, static_cast<uint16_t>(taskCount_ * sizeof(float)), 0, 0};
            DataCopyPad(loss_dst[firstBatch_], result_, copy_params);
        }

        if (reduction_ != REDUCTION_NONE && batchSize_ > 1) {
            SyncMte3Published<LOSS_SYNC_FLAG>();
            if (GetBlockIdx() == 0) {
                ReduceLosses();
            }
        }
    }

private:
    template <uint16_t FLAG_ID>
    __aicore__ inline void SyncMte3Published()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(FLAG_ID);
        CrossCoreWaitFlag(FLAG_ID);
    }

    __aicore__ inline void Mte2ToS()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(id);
        WaitFlag<HardEvent::MTE2_S>(id);
    }

    __aicore__ inline void SToMte2()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(id);
        WaitFlag<HardEvent::S_MTE2>(id);
    }

    __aicore__ inline void Mte2ToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }

    __aicore__ inline void VToMte2()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
    }

    __aicore__ inline void SToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
    }

    __aicore__ inline void VToS()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
    }

    __aicore__ inline void SToMte3()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(id);
        WaitFlag<HardEvent::S_MTE3>(id);
    }

    __aicore__ inline void VToMte3()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }

    __aicore__ inline void Mte3ToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
    }

    __aicore__ inline float ReadProb(int64_t time, int64_t batch, int64_t label)
    {
        SToMte2();
        DataCopyParams copy_params{1, static_cast<uint16_t>(sizeof(TProb)), 0, 0};
        DataCopyPadParams pad_params{false, 0, 0, 0};
        const int64_t offset = (time * batchSize_ + batch) * numClasses_ + label;
        DataCopyPad(probScalar_, logProbsGm_[offset], copy_params, pad_params);
        Mte2ToS();
        return static_cast<float>(probScalar_.GetValue(0));
    }

    __aicore__ inline void BeginLoadInitialProbs(int64_t batch, int64_t target_length)
    {
        SToMte2();
        DataCopyParams copy_params{1, static_cast<uint16_t>(sizeof(TProb)), 0, 0};
        DataCopyPadParams pad_params{false, 0, 0, 0};
        const int64_t row_offset = batch * numClasses_;
        DataCopyPad(probScalar_, logProbsGm_[row_offset + blank_], copy_params, pad_params);
        if (target_length > 0) {
            constexpr int64_t second_slot = BLOCK_BYTES / sizeof(TProb);
            const int64_t label = static_cast<int64_t>(targets_.GetValue(0));
            DataCopyPad(probScalar_[second_slot], logProbsGm_[row_offset + label], copy_params, pad_params);
        }
    }

    __aicore__ inline void EndLoadInitialProbs(int64_t target_length)
    {
        Mte2ToS();
        alphaA_.SetValue(0, static_cast<float>(probScalar_.GetValue(0)));
        if (target_length > 0) {
            constexpr int64_t second_slot = BLOCK_BYTES / sizeof(TProb);
            alphaA_.SetValue(1, static_cast<float>(probScalar_.GetValue(second_slot)));
        }
    }

    __aicore__ inline void LoadTargets(int64_t batch, int64_t target_length)
    {
        if (target_length == 0) {
            return;
        }
        SToMte2();
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(target_length * sizeof(TTarget)), 0, 0, 0};
        DataCopyPadExtParams<TTarget> pad_params{false, 0, 0, 0};
        DataCopyPad(targets_, targetsGm_[batch * targetStride_], copy_params, pad_params);
        Mte2ToS();
    }

    __aicore__ inline void BeginLoadProbSlice(int64_t time, int64_t batch, int64_t slice_id,
                                              int64_t slice_length)
    {
        BeginLoadProbSliceTo(time, batch, slice_id, slice_length, 0);
    }

    __aicore__ inline void BeginLoadProbSliceTo(int64_t time, int64_t batch, int64_t slice_id,
                                                int64_t slice_length, int64_t local_offset)
    {
        VToMte2();
        const int64_t offset = (time * batchSize_ + batch) * numClasses_ + slice_id * sliceLength_;
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(slice_length * sizeof(TProb)), 0, 0, 0};
        DataCopyPadExtParams<TProb> pad_params{false, 0, 0, 0};
        DataCopyPad(probSlice_[local_offset], logProbsGm_[offset], copy_params, pad_params);
    }

    __aicore__ inline void EndLoadProbSlice(int64_t slice_length)
    {
        Mte2ToV();
    }

    __aicore__ inline void BeginLoadProbBand(int64_t time, int64_t batch, int64_t row_count,
                                             int64_t slot)
    {
        VToMte2();
        const int64_t row_bytes = numClasses_ * sizeof(TProb);
        const int64_t offset = (time * batchSize_ + batch) * numClasses_;
        const uint32_t src_stride = static_cast<uint32_t>((batchSize_ - 1) * row_bytes);
        const uint32_t dst_stride = static_cast<uint32_t>((probRowAlign_ * sizeof(TProb) - row_bytes) /
                                                          BLOCK_BYTES);
        DataCopyExtParams copy_params{static_cast<uint16_t>(row_count), static_cast<uint32_t>(row_bytes),
                                      src_stride, dst_stride, 0};
        DataCopyPadExtParams<TProb> pad_params{false, 0, 0, 0};
        const int64_t slot_offset = slot * probBandRows_ * probRowAlign_;
        DataCopyPad(probSlice_[slot_offset], logProbsGm_[offset], copy_params, pad_params);
    }

    __aicore__ inline void GatherProb(LocalTensor<float> dst, LocalTensor<TProb> src,
                                      LocalTensor<uint32_t> indices, int64_t count)
    {
        if constexpr (std::is_same<TProb, float>::value) {
            Gather(dst, src, indices, 0, count);
        } else {
            LocalTensor<TProb> gathered = alphaB_.ReinterpretCast<TProb>();
            PipeBarrier<PIPE_V>();
            Gather(gathered, src, indices, 0, count);
            PipeBarrier<PIPE_V>();
            Cast(dst, gathered, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherHalfProbAndAdd(LocalTensor<float> dst, LocalTensor<TProb> src,
                                                LocalTensor<uint32_t> indices, int64_t active_count,
                                                int64_t vector_count)
    {
        LocalTensor<TProb> gathered = alphaB_.ReinterpretCast<TProb>();
        PipeBarrier<PIPE_V>();
        const int64_t repeat_count = vector_count / VECTOR_ALIGN;
        if (vector_count > 0 && vector_count % VECTOR_ALIGN == 0 && repeat_count <= 255) {
            Gather(gathered, src, indices, 0, vector_count);
            PipeBarrier<PIPE_V>();
            constexpr uint64_t full_fp32_mask = VECTOR_ALIGN;
            const UnaryRepeatParams axpy_params(1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
            if (vector_count % (2 * VECTOR_ALIGN) == VECTOR_ALIGN) {
                Axpy<float, TProb, false>(dst, gathered, static_cast<TProb>(1.0f), full_fp32_mask,
                                          static_cast<uint8_t>(repeat_count), axpy_params);
            } else {
                Axpy<float, TProb, true>(dst, gathered, static_cast<TProb>(1.0f), full_fp32_mask,
                                         static_cast<uint8_t>(repeat_count), axpy_params);
            }
            return;
        }
        Gather(gathered, src, indices, 0, active_count);
        PipeBarrier<PIPE_V>();
        Axpy(dst, gathered, static_cast<TProb>(1.0f), static_cast<int32_t>(active_count));
    }

    __aicore__ inline void GatherFloatProbAndAddAligned(LocalTensor<float> dst, LocalTensor<float> gathered,
                                                        LocalTensor<TProb> src, LocalTensor<uint32_t> indices,
                                                        int64_t active_count, int64_t vector_count)
    {
        const int64_t repeat_count = vector_count / VECTOR_ALIGN;
        if (vector_count > 0 && vector_count % VECTOR_ALIGN == 0 && repeat_count <= 255) {
            GatherProb(gathered, src, indices, vector_count);
            constexpr uint64_t full_fp32_mask = VECTOR_ALIGN;
            const BinaryRepeatParams binary_params;
            Add<float, false>(dst, dst, gathered, full_fp32_mask, static_cast<uint8_t>(repeat_count), binary_params);
            return;
        }
        GatherProb(gathered, src, indices, active_count);
        Add(dst, dst, gathered, active_count);
    }

    __aicore__ inline void LoadProbSlice(int64_t time, int64_t batch, int64_t slice_id, int64_t slice_length)
    {
        BeginLoadProbSlice(time, batch, slice_id, slice_length);
        EndLoadProbSlice(slice_length);
    }

    __aicore__ inline float GatherTargetProbs(int64_t time, int64_t batch, int64_t target_length)
    {
        Duplicate(max_, 0.0f, targetAlign_);
        PipeBarrier<PIPE_V>();
        VToS();
        float blank_prob = -INFINITY;
        for (int64_t slice_id = 0; slice_id < sliceCount_; ++slice_id) {
            const int64_t slice_start = slice_id * sliceLength_;
            const int64_t slice_length = slice_id == sliceCount_ - 1 ? sliceTail_ : sliceLength_;
            const int64_t slice_end = slice_start + slice_length;
            for (int64_t i = 0; i < target_length; ++i) {
                const int64_t label = static_cast<int64_t>(targets_.GetValue(i));
                const uint32_t byte_offset = (label >= slice_start && label < slice_end)
                                                 ? static_cast<uint32_t>((label - slice_start) * sizeof(TProb))
                                                 : 0U;
                gatherIndex_.SetValue(i, byte_offset);
            }
            if (slice_id == 0) {
                EndLoadProbSlice(slice_length);
            } else {
                LoadProbSlice(time, batch, slice_id, slice_length);
            }
            SToV();
            GatherProb(aux_, probSlice_, gatherIndex_, target_length);
            VToS();
            if (blank_ >= slice_start && blank_ < slice_end) {
                blank_prob = static_cast<float>(probSlice_.GetValue(blank_ - slice_start));
            }
            for (int64_t i = 0; i < target_length; ++i) {
                const int64_t label = static_cast<int64_t>(targets_.GetValue(i));
                if (label >= slice_start && label < slice_end) {
                    max_.SetValue(i, aux_.GetValue(i));
                }
            }
        }
        return blank_prob;
    }

    __aicore__ inline void BuildStateMetadata(int64_t target_length, int64_t state_count, int64_t state_align,
                                              bool exact_mode)
    {
        const uint32_t sentinel_offset = static_cast<uint32_t>(state_count * sizeof(float));
        const int64_t metadata_count = exact_mode ? state_count : state_align;
        ArithProgression<int32_t>(transitionIndex_.ReinterpretCast<int32_t>(), -static_cast<int32_t>(sizeof(float)),
                                  static_cast<int32_t>(sizeof(float)), metadata_count);
        Duplicate(transitionIndex2_, sentinel_offset, metadata_count);
        if (sliceCount_ == 1) {
            Duplicate(gatherIndex_, static_cast<uint32_t>(blank_ * sizeof(TProb)), metadata_count);
        }
        VToS();
        transitionIndex_.SetValue(0, sentinel_offset);
        for (int64_t state = state_count; state < metadata_count; ++state) {
            transitionIndex_.SetValue(state, sentinel_offset);
        }
        int64_t previous_label = 0;
        for (int64_t target_id = 0; target_id < target_length; ++target_id) {
            const int64_t state = 2 * target_id + 1;
            const int64_t label = static_cast<int64_t>(targets_.GetValue(target_id));
            if (sliceCount_ == 1) {
                gatherIndex_.SetValue(state, static_cast<uint32_t>(label * sizeof(TProb)));
            }
            if (target_id > 0 && label != previous_label) {
                transitionIndex2_.SetValue(state, static_cast<uint32_t>((state - 2) * sizeof(float)));
            }
            previous_label = label;
        }
        SToV();
    }

    __aicore__ inline void BuildReverseMetadata(int64_t target_length, int64_t state_count, int64_t state_align)
    {
        const uint32_t sentinel_offset = static_cast<uint32_t>(state_count * sizeof(float));
        ArithProgression<int32_t>(transitionIndex_.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(float)),
                                  static_cast<int32_t>(sizeof(float)), state_align);
        Duplicate(transitionIndex2_, sentinel_offset, state_align);
        Duplicate(gatherIndex_, static_cast<uint32_t>(blank_ * sizeof(TProb)), state_align);
        VToS();
        for (int64_t state = state_count - 1; state < state_align; ++state) {
            transitionIndex_.SetValue(state, sentinel_offset);
        }
        int64_t label = target_length > 0 ? static_cast<int64_t>(targets_.GetValue(0)) : 0;
        for (int64_t target_id = 0; target_id < target_length; ++target_id) {
            const int64_t state = 2 * target_id + 1;
            gatherIndex_.SetValue(state, static_cast<uint32_t>(label * sizeof(TProb)));
            if (target_id + 1 < target_length) {
                const int64_t next_label = static_cast<int64_t>(targets_.GetValue(target_id + 1));
                if (label != next_label) {
                    transitionIndex2_.SetValue(state, static_cast<uint32_t>((state + 2) * sizeof(float)));
                }
                label = next_label;
            }
        }
        SToV();
    }

    __aicore__ inline void LogAddThree(LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float> c,
                                       LocalTensor<float> scratch, int64_t count)
    {
        const int64_t repeat_count = count / VECTOR_ALIGN;
        const bool use_normal_mask = count > 0 && count % VECTOR_ALIGN == 0 && repeat_count <= 255 &&
                                     (count != alphaAlign_ || 3 * repeat_count <= 255);
        if (use_normal_mask) {
            constexpr uint64_t full_fp32_mask = VECTOR_ALIGN;
            const uint8_t repeats = static_cast<uint8_t>(repeat_count);
            const BinaryRepeatParams binary_params;
            const UnaryRepeatParams unary_params;
            Max<float, false>(scratch, a, b, full_fp32_mask, repeats, binary_params);
            PipeBarrier<PIPE_V>();
            Max<float, false>(scratch, scratch, c, full_fp32_mask, repeats, binary_params);
            PipeBarrier<PIPE_V>();
            Maxs<float, false>(scratch, scratch, FLOAT_LOWEST, full_fp32_mask, repeats, unary_params);
            PipeBarrier<PIPE_V>();
            if (count == VECTOR_ALIGN && alphaAlign_ == VECTOR_ALIGN) {
                const BinaryRepeatParams plane_broadcast_params(1, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                                 DEFAULT_REPEAT_STRIDE, 0);
                Sub<float, false>(a, a, scratch, full_fp32_mask, 3, plane_broadcast_params);
            } else {
                Sub<float, false>(a, a, scratch, full_fp32_mask, repeats, binary_params);
                Sub<float, false>(b, b, scratch, full_fp32_mask, repeats, binary_params);
                Sub<float, false>(c, c, scratch, full_fp32_mask, repeats, binary_params);
            }
            PipeBarrier<PIPE_V>();
            if (count == alphaAlign_) {
                Exp<float, false>(a, a, full_fp32_mask, static_cast<uint8_t>(3 * repeat_count), unary_params);
            } else {
                Exp<float, false>(a, a, full_fp32_mask, repeats, unary_params);
                Exp<float, false>(b, b, full_fp32_mask, repeats, unary_params);
                Exp<float, false>(c, c, full_fp32_mask, repeats, unary_params);
            }
            PipeBarrier<PIPE_V>();
            Add<float, false>(b, a, b, full_fp32_mask, repeats, binary_params);
            PipeBarrier<PIPE_V>();
            Add<float, false>(c, b, c, full_fp32_mask, repeats, binary_params);
            PipeBarrier<PIPE_V>();
            Ln<float, false>(a, c, full_fp32_mask, repeats, unary_params);
            PipeBarrier<PIPE_V>();
            Add<float, false>(a, a, scratch, full_fp32_mask, repeats, binary_params);
            PipeBarrier<PIPE_V>();
            return;
        }

        Max(scratch, a, b, count);
        PipeBarrier<PIPE_V>();
        Max(scratch, scratch, c, count);
        PipeBarrier<PIPE_V>();
        Maxs(scratch, scratch, FLOAT_LOWEST, count);
        PipeBarrier<PIPE_V>();
        Sub(a, a, scratch, count);
        Sub(b, b, scratch, count);
        Sub(c, c, scratch, count);
        PipeBarrier<PIPE_V>();
        if (count == alphaAlign_) {
            Exp(a, a, 3 * count);
        } else {
            Exp(a, a, count);
            Exp(b, b, count);
            Exp(c, c, count);
        }
        PipeBarrier<PIPE_V>();
        Add(b, a, b, count);
        PipeBarrier<PIPE_V>();
        Add(c, b, c, count);
        PipeBarrier<PIPE_V>();
        Log(a, c, count);
        PipeBarrier<PIPE_V>();
        Add(a, a, scratch, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LogAddTwo(LocalTensor<float> a, LocalTensor<float> b,
                                     LocalTensor<float> scratch, int64_t count)
    {
        Max(scratch, a, b, count);
        PipeBarrier<PIPE_V>();
        Maxs(scratch, scratch, FLOAT_LOWEST, count);
        PipeBarrier<PIPE_V>();
        Sub(a, a, scratch, count);
        Sub(b, b, scratch, count);
        PipeBarrier<PIPE_V>();
        Exp(a, a, count);
        Exp(b, b, count);
        PipeBarrier<PIPE_V>();
        Add(b, a, b, count);
        PipeBarrier<PIPE_V>();
        Log(a, b, count);
        PipeBarrier<PIPE_V>();
        Add(a, a, scratch, count);
        PipeBarrier<PIPE_ALL>();
    }

    template <bool TWO_WAY>
    __aicore__ inline void AdvanceAlpha(LocalTensor<float> alpha, LocalTensor<float> scratch,
                                        int64_t time, int64_t batch, int64_t target_length,
                                        int64_t active_count, bool exact_mode)
    {
        const int64_t vector_count = exact_mode ? active_count : AlignUp(active_count, VECTOR_ALIGN);
        if (sliceCount_ == 1 || target_length > 0) {
            const int64_t first_slice_length = sliceCount_ == 1 ? numClasses_ : sliceLength_;
            BeginLoadProbSlice(time, batch, 0, first_slice_length);
        }
        if constexpr (TWO_WAY) {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddTwo(alpha, aux_, scratch, vector_count);
        } else if (!exact_mode && vector_count == alphaAlign_ && active_count == 2 * target_length + 1) {
            Gather(aux_, alpha, transitionIndex_, 0, 2 * vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        } else {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            Gather(max_, alpha, transitionIndex2_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        }

        if (sliceCount_ == 1) {
            EndLoadProbSlice(numClasses_);
            if constexpr (std::is_same<TProb, half>::value) {
                GatherHalfProbAndAdd(alpha, probSlice_, gatherIndex_, active_count, vector_count);
            } else {
                GatherFloatProbAndAddAligned(alpha, aux_, probSlice_, gatherIndex_, active_count, vector_count);
            }
        } else {
            const int64_t active_target_count = active_count / 2;
            const float blank_prob = target_length == 0 ? ReadProb(time, batch, blank_) :
                                                          GatherTargetProbs(time, batch, active_target_count);
            Duplicate(aux_, blank_prob, vector_count);
            PipeBarrier<PIPE_V>();
            VToS();
            for (int64_t target_id = 0; target_id < active_target_count; ++target_id) {
                aux_.SetValue(2 * target_id + 1, max_.GetValue(target_id));
            }
            SToV();
            Add(alpha, alpha, aux_, active_count);
            PipeBarrier<PIPE_V>();
        }
    }

    template <bool TWO_WAY>
    __aicore__ inline void AdvanceAlphaPingPong(LocalTensor<float> alpha, LocalTensor<float> scratch,
                                                int64_t time, int64_t batch, int64_t target_length,
                                                int64_t active_count, bool exact_mode)
    {
        const int64_t vector_count = exact_mode ? active_count : AlignUp(active_count, VECTOR_ALIGN);
        if constexpr (TWO_WAY) {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddTwo(alpha, aux_, scratch, vector_count);
        } else if (!exact_mode && vector_count == alphaAlign_ && active_count == 2 * target_length + 1) {
            Gather(aux_, alpha, transitionIndex_, 0, 2 * vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        } else {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            Gather(max_, alpha, transitionIndex2_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        }

        EndLoadProbSlice(numClasses_);
        const int64_t current_slot = (time - 1) & 1;
        if (time + 1 < inputLength_) {
            const int64_t next_slot = current_slot ^ 1;
            BeginLoadProbSliceTo(time + 1, batch, 0, numClasses_, next_slot * probRowAlign_);
        }
        if constexpr (std::is_same<TProb, half>::value) {
            GatherHalfProbAndAdd(alpha, probSlice_[current_slot * probRowAlign_], gatherIndex_, active_count,
                                 vector_count);
        } else {
            GatherFloatProbAndAddAligned(alpha, aux_, probSlice_[current_slot * probRowAlign_], gatherIndex_,
                                         active_count, vector_count);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool TWO_WAY>
    __aicore__ inline void AdvanceAlphaBand(LocalTensor<float> alpha, LocalTensor<float> scratch,
                                            int64_t batch, int64_t target_length, int64_t active_count,
                                            bool exact_mode, LocalTensor<TProb> prob_row, bool first_in_band,
                                            int64_t next_time, int64_t next_count, int64_t next_slot)
    {
        const int64_t vector_count = exact_mode ? active_count : AlignUp(active_count, VECTOR_ALIGN);
        if constexpr (TWO_WAY) {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddTwo(alpha, aux_, scratch, vector_count);
        } else if (!exact_mode && vector_count == alphaAlign_ && active_count == 2 * target_length + 1) {
            Gather(aux_, alpha, transitionIndex_, 0, 2 * vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        } else {
            Gather(aux_, alpha, transitionIndex_, 0, vector_count);
            Gather(max_, alpha, transitionIndex2_, 0, vector_count);
            PipeBarrier<PIPE_V>();
            LogAddThree(alpha, aux_, max_, scratch, vector_count);
        }

        if (first_in_band) {
            EndLoadProbSlice(numClasses_);
            if (next_count > 0) {
                BeginLoadProbBand(next_time, batch, next_count, next_slot);
            }
        }
        if constexpr (std::is_same<TProb, half>::value) {
            GatherHalfProbAndAdd(alpha, prob_row, gatherIndex_, active_count, vector_count);
        } else {
            GatherFloatProbAndAddAligned(alpha, aux_, prob_row, gatherIndex_, active_count, vector_count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateForwardFrontier(int64_t& frontier, int64_t state_count)
    {
        if (frontier + 1 >= state_count) {
            return;
        }
        if ((frontier & 1) != 0 && frontier + 2 < state_count &&
            targets_.GetValue(frontier / 2) != targets_.GetValue(frontier / 2 + 1)) {
            frontier += 2;
        } else {
            frontier += 1;
        }
    }

    __aicore__ inline void RecenterPairedAlpha(int64_t state_count, int64_t state_align,
                                               int64_t transition_steps, float& center_scale)
    {
        if constexpr (std::is_same<TProb, float>::value) {
            if (transition_steps % VECTOR_ALIGN != 0) {
                return;
            }
            ReduceMax(max_, alphaA_, aux_, static_cast<uint32_t>(state_align), false);
            VToS();
            const float center = max_.GetValue(0);
            if (center != -INFINITY) {
                center_scale += center;
                Adds(alphaA_, alphaA_, -center, state_align);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void RestorePairedCenter(int64_t state_align, float center_scale)
    {
        if constexpr (std::is_same<TProb, float>::value) {
            if (center_scale != 0.0f) {
                Adds(alphaA_, alphaA_, center_scale, state_align);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeForwardMidpoint(int64_t batch, int64_t input_length, int64_t target_length,
                                                   int64_t midpoint)
    {
        inputLength_ = midpoint + 1;
        const int64_t state_count = 2 * target_length + 1;
        const int64_t state_align = AlignUp(state_count, VECTOR_ALIGN);
        LoadTargets(batch, target_length);
        BeginLoadInitialProbs(batch, target_length);
        BuildStateMetadata(target_length, state_count, state_align, false);
        Duplicate(alphaA_, -INFINITY, state_align);
        PipeBarrier<PIPE_V>();
        VToS();
        EndLoadInitialProbs(target_length);

        int64_t frontier = 1;
        int64_t transition_steps = 0;
        float center_scale = 0.0f;
        if (probBandRows_ > 1) {
            int64_t band_start = 1;
            int64_t band_count = midpoint < probBandRows_ ? midpoint : probBandRows_;
            int64_t slot = 0;
            BeginLoadProbBand(band_start, batch, band_count, slot);
            while (band_start <= midpoint) {
                const int64_t next_start = band_start + band_count;
                const int64_t next_remaining = midpoint - next_start + 1;
                const int64_t next_count = next_remaining > 0 ?
                                               (next_remaining < probBandRows_ ? next_remaining : probBandRows_) : 0;
                const int64_t next_slot = slot ^ 1;
                const int64_t slot_offset = slot * probBandRows_ * probRowAlign_;
                for (int64_t row = 0; row < band_count; ++row) {
                    UpdateForwardFrontier(frontier, state_count);
                    AdvanceAlphaBand<false>(alphaA_, alphaB_, batch, target_length, frontier + 1, false,
                                            probSlice_[slot_offset + row * probRowAlign_], row == 0,
                                            next_start, next_count, next_slot);
                    ++transition_steps;
                    RecenterPairedAlpha(state_count, state_align, transition_steps, center_scale);
                }
                band_start = next_start;
                band_count = next_count;
                slot = next_slot;
            }
            RestorePairedCenter(state_align, center_scale);
            return;
        }

        BeginLoadProbSliceTo(1, batch, 0, numClasses_, 0);
        for (int64_t time = 1; time <= midpoint; ++time) {
            UpdateForwardFrontier(frontier, state_count);
            AdvanceAlphaPingPong<false>(alphaA_, alphaB_, time, batch, target_length, frontier + 1, false);
            ++transition_steps;
            RecenterPairedAlpha(state_count, state_align, transition_steps, center_scale);
        }
        RestorePairedCenter(state_align, center_scale);
    }

    __aicore__ inline int64_t ReverseVectorStart(int64_t state_count, int64_t transition_steps) const
    {
        int64_t semantic_start = state_count - 2 - 2 * transition_steps;
        if (semantic_start < 0) {
            semantic_start = 0;
        }
        return semantic_start / VECTOR_ALIGN * VECTOR_ALIGN;
    }

    __aicore__ inline int64_t ReverseLogAdd(int64_t state_count, int64_t state_align, int64_t transition_steps)
    {
        const int64_t vector_start = ReverseVectorStart(state_count, transition_steps);
        const int64_t vector_count = state_align - vector_start;
        if (vector_start == 0 && vector_count == alphaAlign_) {
            Gather(aux_, alphaA_, transitionIndex_, 0, 2 * vector_count);
        } else {
            Gather(aux_[vector_start], alphaA_, transitionIndex_[vector_start], 0, vector_count);
            Gather(max_[vector_start], alphaA_, transitionIndex2_[vector_start], 0, vector_count);
        }
        PipeBarrier<PIPE_V>();
        LogAddThree(alphaA_[vector_start], aux_[vector_start], max_[vector_start], alphaB_[vector_start],
                    vector_count);
        return vector_start;
    }

    __aicore__ inline void AddReverseEmission(LocalTensor<TProb> prob_row, int64_t state_count,
                                               int64_t state_align, int64_t vector_start)
    {
        const int64_t active_count = state_count - vector_start;
        const int64_t vector_count = state_align - vector_start;
        if constexpr (std::is_same<TProb, half>::value) {
            GatherHalfProbAndAdd(alphaA_[vector_start], prob_row, gatherIndex_[vector_start], active_count,
                                 vector_count);
        } else {
            GatherFloatProbAndAddAligned(alphaA_[vector_start], aux_[vector_start], prob_row,
                                         gatherIndex_[vector_start], active_count, vector_count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeReverse(LocalTensor<TProb> final_row, int64_t target_length,
                                              int64_t state_count, int64_t state_align)
    {
        Duplicate(alphaA_, -INFINITY, state_align);
        PipeBarrier<PIPE_V>();
        GatherProb(aux_, final_row, gatherIndex_, state_count);
        VToS();
        alphaA_.SetValue(state_count - 2, aux_.GetValue(state_count - 2));
        alphaA_.SetValue(state_count - 1, aux_.GetValue(state_count - 1));
        SToV();
    }

    __aicore__ inline void ComputeBackwardSuffix(int64_t batch, int64_t input_length, int64_t target_length,
                                                  int64_t midpoint)
    {
        inputLength_ = input_length;
        const int64_t state_count = 2 * target_length + 1;
        const int64_t state_align = AlignUp(state_count, VECTOR_ALIGN);
        LoadTargets(batch, target_length);
        BuildReverseMetadata(target_length, state_count, state_align);
        int64_t transition_steps = 0;
        float center_scale = 0.0f;

        if (probBandRows_ > 1) {
            int64_t band_end = input_length;
            int64_t band_start = band_end - probBandRows_;
            if (band_start < midpoint + 1) {
                band_start = midpoint + 1;
            }
            int64_t band_count = band_end - band_start;
            int64_t slot = 0;
            BeginLoadProbBand(band_start, batch, band_count, slot);
            EndLoadProbSlice(numClasses_);
            int64_t slot_offset = slot * probBandRows_ * probRowAlign_;
            InitializeReverse(probSlice_[slot_offset + (band_count - 1) * probRowAlign_], target_length,
                              state_count, state_align);
            int64_t first_row = band_count - 2;

            while (true) {
                const int64_t previous_end = band_start;
                int64_t previous_start = previous_end - probBandRows_;
                if (previous_start < midpoint + 1) {
                    previous_start = midpoint + 1;
                }
                const int64_t previous_count = previous_end - previous_start;
                const int64_t previous_slot = slot ^ 1;
                if (previous_count > 0) {
                    BeginLoadProbBand(previous_start, batch, previous_count, previous_slot);
                }
                for (int64_t row = first_row; row >= 0; --row) {
                    ++transition_steps;
                    const int64_t vector_start = ReverseLogAdd(state_count, state_align, transition_steps);
                    AddReverseEmission(probSlice_[slot_offset + row * probRowAlign_], state_count, state_align,
                                       vector_start);
                    RecenterPairedAlpha(state_count, state_align, transition_steps, center_scale);
                }
                if (previous_count == 0) {
                    break;
                }
                EndLoadProbSlice(numClasses_);
                band_end = previous_end;
                band_start = previous_start;
                band_count = previous_count;
                slot = previous_slot;
                slot_offset = slot * probBandRows_ * probRowAlign_;
                first_row = band_count - 1;
            }
        } else {
            BeginLoadProbSliceTo(input_length - 1, batch, 0, numClasses_, 0);
            EndLoadProbSlice(numClasses_);
            InitializeReverse(probSlice_, target_length, state_count, state_align);
            if (input_length - 2 > midpoint) {
                BeginLoadProbSliceTo(input_length - 2, batch, 0, numClasses_, probRowAlign_);
            }
            for (int64_t time = input_length - 2; time > midpoint; --time) {
                ++transition_steps;
                const int64_t vector_start = ReverseLogAdd(state_count, state_align, transition_steps);
                const int64_t slot = (input_length - 1 - time) & 1;
                EndLoadProbSlice(numClasses_);
                if (time - 1 > midpoint) {
                    BeginLoadProbSliceTo(time - 1, batch, 0, numClasses_, (slot ^ 1) * probRowAlign_);
                }
                AddReverseEmission(probSlice_[slot * probRowAlign_], state_count, state_align, vector_start);
                RecenterPairedAlpha(state_count, state_align, transition_steps, center_scale);
            }
        }

        ++transition_steps;
        ReverseLogAdd(state_count, state_align, transition_steps);
        RecenterPairedAlpha(state_count, state_align, transition_steps, center_scale);
        RestorePairedCenter(state_align, center_scale);
    }

    __aicore__ inline float CombineMidpoint(int64_t batch, int64_t target_length)
    {
        const int64_t state_count = 2 * target_length + 1;
        const int64_t state_align = AlignUp(state_count, VECTOR_ALIGN);
        VToMte2();
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(state_align * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad_params{false, 0, 0, 0};
        DataCopyPad(alphaB_, midpointGm_[batch * alphaAlign_], copy_params, pad_params);
        Mte2ToV();
        Add(alphaA_, alphaA_, alphaB_, state_align);
        PipeBarrier<PIPE_V>();
        ReduceMax(max_, alphaA_, aux_, static_cast<uint32_t>(state_align), false);
        VToS();
        const float max_value = max_.GetValue(0);
        if (max_value == -INFINITY) {
            return zeroInfinity_ != 0 ? 0.0f : INFINITY;
        }
        Duplicate(aux_, max_value, state_align);
        PipeBarrier<PIPE_V>();
        Sub(alphaA_, alphaA_, aux_, state_align);
        PipeBarrier<PIPE_V>();
        Exp(alphaA_, alphaA_, state_align);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, true>(max_, alphaA_, aux_, static_cast<int32_t>(state_align));
        VToS();
        max_.SetValue(8, max_.GetValue(0));
        SToV();
        Log(max_[8], max_[8], 8);
        VToS();
        return -(max_.GetValue(8) + max_value);
    }

    __aicore__ inline void WriteMidpoint(int64_t batch, int64_t state_align)
    {
        VToMte3();
        DataCopyParams copy_params{1, static_cast<uint16_t>(state_align * sizeof(float)), 0, 0};
        DataCopyPad(midpointGm_[batch * alphaAlign_], alphaA_, copy_params);
    }

    __aicore__ inline void ProcessPaired()
    {
        if (taskCount_ == 0) {
            return;
        }
        const int64_t batch = firstBatch_;
        const int64_t input_length = inputLengthsGm_.GetValue(batch);
        const int64_t target_length = targetLengthsGm_.GetValue(batch);
        const bool pair_runtime = input_length >= PAIR_MIN_TIME && input_length <= timeStep_ &&
                                  target_length >= 2 && target_length <= maxTargetLength_;
        const int64_t midpoint = pair_runtime ? (input_length - 1) / 2 : 0;
        float loss = 0.0f;
        if (pair_runtime) {
            if (reverseCore_) {
                ComputeBackwardSuffix(batch, input_length, target_length, midpoint);
                WriteMidpoint(batch, AlignUp(2 * target_length + 1, VECTOR_ALIGN));
            } else {
                ComputeForwardMidpoint(batch, input_length, target_length, midpoint);
            }
        } else if (!reverseCore_) {
            loss = ComputeBatch(batch, input_length, target_length);
        }

        SyncMte3Published<MIDPOINT_SYNC_FLAG>();
        if (!reverseCore_) {
            if (pair_runtime) {
                loss = CombineMidpoint(batch, target_length);
            }
            if (reduction_ == REDUCTION_MEAN) {
                float divisor = target_length > 0 ? static_cast<float>(target_length) : 1.0f;
                if (batchSize_ > 1) {
                    divisor *= static_cast<float>(batchSize_);
                }
                loss /= divisor;
            }
            GlobalTensor<float> loss_dst = reduction_ == REDUCTION_NONE || batchSize_ == 1 ? outputGm_ : tempLossGm_;
            WriteOne(loss_dst, batch, loss);
        }

        if (reduction_ != REDUCTION_NONE && batchSize_ > 1) {
            SyncMte3Published<LOSS_SYNC_FLAG>();
            if (GetBlockIdx() == 0) {
                ReduceLosses();
            }
        }
    }

    __aicore__ inline int64_t AsymmetricReverseTaskCount(int64_t batch) const
    {
        const int64_t wide_batch_count = (tasksPerCore_ + 1) * tailCoreCount_;
        return batch < wide_batch_count ? tasksPerCore_ + 1 : tasksPerCore_;
    }

    __aicore__ inline int64_t AsymmetricMidpoint(int64_t input_length, int64_t reverse_task_count) const
    {
        return reverse_task_count * (input_length - 1) / (reverse_task_count + 1);
    }

    __aicore__ inline bool CanPairRuntime(int64_t input_length, int64_t target_length) const
    {
        return input_length >= VECTOR_ALIGN && input_length <= timeStep_ &&
               target_length >= 2 && target_length <= maxTargetLength_;
    }

    __aicore__ inline void ProcessAsymmetricPaired()
    {
        float loss = 0.0f;
        bool pair_runtime = false;
        int64_t forward_batch = firstBatch_;
        int64_t forward_target_length = 0;

        if (reverseCore_) {
            for (int64_t task = 0; task < taskCount_; ++task) {
                const int64_t batch = firstBatch_ + task;
                const int64_t input_length = inputLengthsGm_.GetValue(batch);
                const int64_t target_length = targetLengthsGm_.GetValue(batch);
                if (CanPairRuntime(input_length, target_length)) {
                    const int64_t midpoint = AsymmetricMidpoint(input_length, taskCount_);
                    ComputeBackwardSuffix(batch, input_length, target_length, midpoint);
                    WriteMidpoint(batch, AlignUp(2 * target_length + 1, VECTOR_ALIGN));
                    if (task + 1 < taskCount_) {
                        Mte3ToV();
                    }
                }
            }
        } else {
            const int64_t input_length = inputLengthsGm_.GetValue(forward_batch);
            forward_target_length = targetLengthsGm_.GetValue(forward_batch);
            pair_runtime = CanPairRuntime(input_length, forward_target_length);
            if (pair_runtime) {
                const int64_t reverse_task_count = AsymmetricReverseTaskCount(forward_batch);
                const int64_t midpoint = AsymmetricMidpoint(input_length, reverse_task_count);
                ComputeForwardMidpoint(forward_batch, input_length, forward_target_length, midpoint);
            } else {
                loss = ComputeBatch(forward_batch, input_length, forward_target_length);
            }
        }

        SyncMte3Published<MIDPOINT_SYNC_FLAG>();
        if (!reverseCore_) {
            if (pair_runtime) {
                loss = CombineMidpoint(forward_batch, forward_target_length);
            }
            if (reduction_ == REDUCTION_MEAN) {
                float divisor = forward_target_length > 0 ? static_cast<float>(forward_target_length) : 1.0f;
                if (batchSize_ > 1) {
                    divisor *= static_cast<float>(batchSize_);
                }
                loss /= divisor;
            }
            GlobalTensor<float> loss_dst = reduction_ == REDUCTION_NONE || batchSize_ == 1 ? outputGm_ : tempLossGm_;
            WriteOne(loss_dst, forward_batch, loss);
        }

        if (reduction_ != REDUCTION_NONE && batchSize_ > 1) {
            SyncMte3Published<LOSS_SYNC_FLAG>();
            if (GetBlockIdx() == 0) {
                ReduceLosses();
            }
        }
    }

    __aicore__ inline float FinalLoss(LocalTensor<float> alpha, int64_t target_length, bool need_vector_sync)
    {
        if (need_vector_sync) {
            VToS();
        }
        if (target_length == 0) {
            const float log_likelihood = alpha.GetValue(0);
            return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
        }
        const float l1 = alpha.GetValue(2 * target_length);
        const float l2 = alpha.GetValue(2 * target_length - 1);
        float max_value = l1 > l2 ? l1 : l2;
        if (max_value == -INFINITY) {
            max_value = 0.0f;
        }
        max_.SetValue(0, l1 - max_value);
        max_.SetValue(8, l2 - max_value);
        SToV();
        Exp(max_, max_, 16);
        PipeBarrier<PIPE_V>();
        Add(max_, max_, max_[8], 8);
        PipeBarrier<PIPE_V>();
        Log(max_[8], max_, 8);
        PipeBarrier<PIPE_V>();
        VToS();
        const float log_likelihood = max_.GetValue(8) + max_value;
        return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
    }

    __aicore__ inline float ComputeBatch(int64_t batch, int64_t input_length, int64_t target_length)
    {
        inputLength_ = input_length;
        if (input_length < 0 || input_length > timeStep_ || target_length < 0 ||
            target_length > maxTargetLength_) {
            return zeroInfinity_ != 0 ? 0.0f : INFINITY;
        }
        if (input_length == 0) {
            const float log_likelihood = target_length == 0 ? 0.0f : -INFINITY;
            return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
        }

        LoadTargets(batch, target_length);
        BeginLoadInitialProbs(batch, target_length);
        if (target_length == 0) {
            Mte2ToS();
            float log_likelihood = static_cast<float>(probScalar_.GetValue(0));
            for (int64_t time = 1; time < input_length; ++time) {
                log_likelihood += ReadProb(time, batch, blank_);
            }
            return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
        }
        if (input_length <= 2 * target_length - 1) {
            int64_t repeat_count = 0;
            for (int64_t target_id = 1; target_id < target_length; ++target_id) {
                repeat_count += targets_.GetValue(target_id) == targets_.GetValue(target_id - 1) ? 1 : 0;
            }
            const int64_t min_input_length = target_length + repeat_count;
            if (input_length <= min_input_length) {
                if (input_length < min_input_length) {
                    Mte2ToS();
                    return zeroInfinity_ != 0 ? 0.0f : INFINITY;
                }
                constexpr int64_t second_slot = BLOCK_BYTES / sizeof(TProb);
                if (input_length == 2 && target_length == 2 && repeat_count == 0) {
                    const int64_t label = static_cast<int64_t>(targets_.GetValue(1));
                    const int64_t offset = (batchSize_ + batch) * numClasses_ + label;
                    DataCopyParams copy_params{1, static_cast<uint16_t>(sizeof(TProb)), 0, 0};
                    DataCopyPadParams pad_params{false, 0, 0, 0};
                    DataCopyPad(probScalar_, logProbsGm_[offset], copy_params, pad_params);
                    Mte2ToS();
                    const float log_likelihood = static_cast<float>(probScalar_.GetValue(second_slot)) +
                                                 static_cast<float>(probScalar_.GetValue(0));
                    return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
                }
                Mte2ToS();
                float log_likelihood = static_cast<float>(probScalar_.GetValue(second_slot));
                int64_t time = 1;
                for (int64_t target_id = 1; target_id < target_length; ++target_id) {
                    const int64_t label = static_cast<int64_t>(targets_.GetValue(target_id));
                    if (label == static_cast<int64_t>(targets_.GetValue(target_id - 1))) {
                        log_likelihood += ReadProb(time++, batch, blank_);
                    }
                    log_likelihood += ReadProb(time++, batch, label);
                }
                return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
            }
        }
        const int64_t state_count = 2 * target_length + 1;
        const int64_t state_align = AlignUp(state_count, VECTOR_ALIGN);
        const bool exact_mode = input_length <= 5 && state_count < VECTOR_ALIGN;
        if (input_length > 1) {
            BuildStateMetadata(target_length, state_count, state_align, exact_mode);
            Duplicate(alphaA_, -INFINITY, state_align);
            PipeBarrier<PIPE_V>();
            VToS();
        }
        if (input_length == 1) {
            Mte2ToS();
            constexpr int64_t second_slot = BLOCK_BYTES / sizeof(TProb);
            const float log_likelihood = static_cast<float>(probScalar_.GetValue(second_slot));
            return zeroInfinity_ != 0 && log_likelihood == -INFINITY ? 0.0f : -log_likelihood;
        }
        EndLoadInitialProbs(target_length);
        const bool use_prob_band = probBandRows_ > 1 && input_length > probBandRows_ + 1;
        if (use_prob_band) {
            BeginLoadProbBand(1, batch, probBandRows_, 0);
        } else if (probPingPong_ != 0 && input_length > 2) {
            BeginLoadProbSliceTo(1, batch, 0, numClasses_, 0);
        }

        if (target_length == 1) {
            if (use_prob_band) {
                int64_t band_start = 1;
                int64_t band_count = probBandRows_;
                int64_t slot = 0;
                while (band_start < input_length) {
                    const int64_t next_start = band_start + band_count;
                    const int64_t next_remaining = input_length - next_start;
                    const int64_t next_count = next_remaining > 0 ?
                                                   (next_remaining < probBandRows_ ? next_remaining : probBandRows_) : 0;
                    const int64_t next_slot = slot ^ 1;
                    const int64_t slot_offset = slot * probBandRows_ * probRowAlign_;
                    for (int64_t row = 0; row < band_count; ++row) {
                        AdvanceAlphaBand<true>(alphaA_, alphaB_, batch, target_length, 3, exact_mode,
                                               probSlice_[slot_offset + row * probRowAlign_], row == 0,
                                               next_start, next_count, next_slot);
                    }
                    band_start = next_start;
                    band_count = next_count;
                    slot = next_slot;
                }
            } else {
                for (int64_t time = 1; time < input_length; ++time) {
                    if (probPingPong_ != 0 && input_length > 2) {
                        AdvanceAlphaPingPong<true>(alphaA_, alphaB_, time, batch, target_length, 3, exact_mode);
                    } else {
                        AdvanceAlpha<true>(alphaA_, alphaB_, time, batch, target_length, 3, exact_mode);
                    }
                }
            }
        } else {
            int64_t frontier = 1;
            if (use_prob_band) {
                int64_t band_start = 1;
                int64_t band_count = probBandRows_;
                int64_t slot = 0;
                while (band_start < input_length) {
                    const int64_t next_start = band_start + band_count;
                    const int64_t next_remaining = input_length - next_start;
                    const int64_t next_count = next_remaining > 0 ?
                                                   (next_remaining < probBandRows_ ? next_remaining : probBandRows_) : 0;
                    const int64_t next_slot = slot ^ 1;
                    const int64_t slot_offset = slot * probBandRows_ * probRowAlign_;
                    for (int64_t row = 0; row < band_count; ++row) {
                        if (frontier + 1 < state_count) {
                            if ((frontier & 1) != 0 && frontier + 2 < state_count &&
                                targets_.GetValue(frontier / 2) != targets_.GetValue(frontier / 2 + 1)) {
                                frontier += 2;
                            } else {
                                frontier += 1;
                            }
                        }
                        AdvanceAlphaBand<false>(alphaA_, alphaB_, batch, target_length, frontier + 1, exact_mode,
                                                probSlice_[slot_offset + row * probRowAlign_], row == 0,
                                                next_start, next_count, next_slot);
                    }
                    band_start = next_start;
                    band_count = next_count;
                    slot = next_slot;
                }
            } else {
                for (int64_t time = 1; time < input_length; ++time) {
                    if (frontier + 1 < state_count) {
                        if ((frontier & 1) != 0 && frontier + 2 < state_count &&
                            targets_.GetValue(frontier / 2) != targets_.GetValue(frontier / 2 + 1)) {
                            frontier += 2;
                        } else {
                            frontier += 1;
                        }
                    }
                    const int64_t active_count = frontier + 1;
                    if (probPingPong_ != 0 && input_length > 2) {
                        AdvanceAlphaPingPong<false>(alphaA_, alphaB_, time, batch, target_length, active_count,
                                                    exact_mode);
                    } else {
                        AdvanceAlpha<false>(alphaA_, alphaB_, time, batch, target_length, active_count, exact_mode);
                    }
                }
            }
        }
        return FinalLoss(alphaA_, target_length, input_length > 1);
    }

    __aicore__ inline void WriteOne(GlobalTensor<float> dst, int64_t index, float value)
    {
        result_.SetValue(0, value);
        SToMte3();
        DataCopyParams copy_params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPad(dst[index], result_, copy_params);
    }

    __aicore__ inline void ReduceLosses()
    {
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(batchSize_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad_params{false, 0, 0, 0};
        DataCopyPad(reductionTensor_, tempLossGm_, copy_params, pad_params);
        Mte2ToS();
        float sum = 0.0f;
        for (int64_t i = 0; i < batchSize_; ++i) {
            sum += reductionTensor_.GetValue(i);
        }
        WriteOne(outputGm_, 0, sum);
    }

private:
    TPipe pipe_;
    GlobalTensor<TProb> logProbsGm_;
    GlobalTensor<TTarget> targetsGm_;
    GlobalTensor<int32_t> inputLengthsGm_;
    GlobalTensor<int32_t> targetLengthsGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<float> tempLossGm_;
    GlobalTensor<float> midpointGm_;

    TBuf<TPosition::VECCALC> arenaBuf_;

    LocalTensor<float> alphaA_;
    LocalTensor<float> alphaB_;
    LocalTensor<float> aux_;
    LocalTensor<float> max_;
    LocalTensor<TTarget> targets_;
    LocalTensor<uint32_t> gatherIndex_;
    LocalTensor<uint32_t> transitionIndex_;
    LocalTensor<uint32_t> transitionIndex2_;
    LocalTensor<TProb> probSlice_;
    LocalTensor<TProb> probScalar_;
    LocalTensor<float> result_;
    LocalTensor<float> reductionTensor_;

    int64_t timeStep_ = 0;
    int64_t batchSize_ = 0;
    int64_t numClasses_ = 0;
    int64_t maxTargetLength_ = 0;
    int64_t targetStride_ = 0;
    int64_t alphaAlign_ = 0;
    int64_t targetAlign_ = 0;
    int64_t sliceLength_ = 0;
    int64_t sliceAlign_ = 0;
    int64_t sliceCount_ = 0;
    int64_t sliceTail_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t tailCoreCount_ = 0;
    int64_t reduction_ = 0;
    int64_t blank_ = 0;
    int64_t zeroInfinity_ = 0;
    int64_t probPingPong_ = 0;
    int64_t probRowAlign_ = 0;
    int64_t probBandRows_ = 0;
    int64_t inputLength_ = 0;
    int64_t taskCount_ = 0;
    int64_t firstBatch_ = 0;
    bool reverseCore_ = false;
};

} // namespace NsCtcLoss

#endif // CTCLOSS_H
