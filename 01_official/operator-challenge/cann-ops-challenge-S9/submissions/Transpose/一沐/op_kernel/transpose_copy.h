#ifndef COMPETITION_TRANSPOSE_COPY_H
#define COMPETITION_TRANSPOSE_COPY_H

// Two independent slots overlap GM reads and writes. No vector arithmetic.
template <typename T> class KernelTransposeCopy {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
        __gm__ const CompetitionTransposeTilingData *data)
    {
        data_ = data;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
        input_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(buffer_, 2 * 65536);
        ready_ = pipe_.FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        for (uint32_t s = 0; s < 2; ++s)
            free_[s] = pipe_.AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        for (uint32_t a = 0; a < data->rank; ++a) {
            dims_[a] = data->outputDims[a];
            strides_[a] = data->inputStrides[a];
        }
    }
    __aicore__ inline void Process()
    {
        const uint64_t tasks = data_->taskCount;
        const uint64_t rowSize = data_->contiguousElements;
        const uint64_t chunks = data_->chunksPerRow;
        const uint32_t rank = data_->rank;
        const uint64_t step = AscendC::GetBlockNum();
        constexpr uint32_t capacity = 65536 / sizeof(T);
        uint32_t iteration = 0;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks; task += step, ++iteration) {
            const uint32_t slot = iteration & 1;
            if (iteration >= 2) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
            const uint64_t row = task / chunks;
            const uint64_t start = (task % chunks) * capacity;
            uint64_t index = row * rowSize, offset = 0;
            for (uint32_t rev = 0; rev < rank; ++rev) {
                const uint32_t a = rank - 1 - rev;
                offset += (index % dims_[a]) * strides_[a];
                index /= dims_[a];
            }
            const uint32_t n = rowSize - start < capacity ? rowSize - start : capacity;
            auto local = buffer_.Get<T>()[slot * capacity];
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, input_[offset + start], cp, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ready_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ready_);
            AscendC::DataCopyPad(output_[row * rowSize + start], local, cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
        }
        for (uint32_t slot = 0; slot < 2 && slot < iteration; ++slot)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
        for (uint32_t slot = 0; slot < 2; ++slot)
            pipe_.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::GlobalTensor<T> input_, output_;
    __gm__ const CompetitionTransposeTilingData *data_;
    uint64_t dims_[16], strides_[16];
    int32_t ready_, free_[2];
};
#endif
