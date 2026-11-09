#ifndef COMPETITION_TRANSPOSE_PIPELINE_H
#define COMPETITION_TRANSPOSE_PIPELINE_H

// Two 64 KB input slots and one 64 KB output buffer fit in the 192 KB UB.
// Large tiles improve transaction utilization in both memory directions. b32/b16
// transpose eliminates Gather indices. Int8 conversion is exact for all values.
template <typename T, bool MergeBatch = false, bool ByteNative = false> class KernelTransposePipeline {
    struct Job { uint64_t input, output; uint32_t rows, cols; };
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
        __gm__ const CompetitionTransposeTilingData *data)
    {
        data_ = data;
        rank_ = data->rank;
        fast_ = data->inputDims[0];
        row_ = data->inputDims[3];
        width_ = data->inputDims[4];
        height_ = data->tileSide;
        group_ = data->inputDims[2];
        // Optional third axis: contiguous in input, possibly separated in output.
        // Its extent is flattened into columns, but output stores remain per batch.
        if constexpr (MergeBatch) {
            batchAxis_ = data->inputDims[6] - 1;
            fastSize_ = data->outputDims[fast_];
            batchStride_ = data->inputDims[8];
        }
        tall_ = group_ == 1 && height_ > width_;
        rows_ = data->outputDims[row_];
        cols_ = data->outputDims[fast_];
        if constexpr (MergeBatch) cols_ *= data->outputDims[batchAxis_];
        inStride_ = data->inputStrides[row_];
        outStride_ = data->inputDims[1];
        rowTiles_ = (rows_ + height_ - 1) / height_;
        colTiles_ = (cols_ + width_ - 1) / width_;
        // Cache descriptors once, rather than reading tiling GM in every tile.
        for (uint32_t a = 0; a < rank_; ++a) {
            dims_[a] = data->outputDims[a];
            strides_[a] = data->inputStrides[a];
        }
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
        if (data->inputDims[5] & 1) input_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (data->inputDims[5] & 2) output_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(inBuffer_, 131072);
        pipe_.InitBuffer(outBuffer_, 65536);
        for (uint32_t s = 0; s < 2; ++s) {
            const uint64_t ib = reinterpret_cast<uint64_t>(inBuffer_.Get<uint8_t>()[s * 65536].GetPhyAddr());
            const uint64_t ob = reinterpret_cast<uint64_t>(outBuffer_.Get<uint8_t>().GetPhyAddr());
            if constexpr (ByteNative) {
                // b8 consumes a selected 16-byte half of each 32-byte row block.
                // Destination halves alternate across 16-row input strips.
                const uint32_t blocks = tall_ ? width_ / 16 : height_ / 16;
                for (uint32_t rb = 0; rb < blocks; ++rb) {
                    for (uint32_t a = 0; a < 16; ++a) {
                        src_[s][rb][a] = ib + (tall_ ?
                            ((rb % 2) * 16 + a) * width_ + (rb / 2) * 32 :
                            (rb * 16 + a) * width_);
                        dst_[s][rb][a] = ob + (tall_ ?
                            ((rb / 2) * 32 + a) * height_ : a * height_ + (rb / 2) * 32);
                        dstHigh_[rb][a] = dst_[s][rb][a] + 16 * height_;
                    }
                }
            } else {
            constexpr uint32_t bytes = sizeof(T) == 1 ? 2 : sizeof(T);
            constexpr uint32_t offset = sizeof(T) == 1 ? 32768 : 0;
            constexpr uint32_t vectorCols = sizeof(T) == 4 ? 8 : 16;
            const uint32_t addressBlocks = tall_ ? width_ / vectorCols : height_ / 16;
            for (uint32_t rb = 0; group_ == 1 && rb < addressBlocks; ++rb) {
                for (uint32_t a = 0; a < 16; ++a) {
                    src_[s][rb][a] = ib + offset + (tall_ ?
                        a * width_ + rb * vectorCols : (rb * 16 + a) * width_) * bytes;
                    if constexpr (sizeof(T) == 4)
                        dst_[s][rb][a] = ob + ((a / 2) * height_ +
                            (tall_ ? rb * vectorCols * height_ : rb * 16) + (a % 2) * 8) * bytes;
                    else dst_[s][rb][a] = ob + offset + (a * height_ +
                        (tall_ ? rb * vectorCols * height_ : rb * 16)) * bytes;
                }
            }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(s);
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t first = AscendC::GetBlockIdx(), step = AscendC::GetBlockNum();
        const uint64_t tasks = data_->taskCount;
        if (first >= tasks) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            return;
        }
        Job jobs[2];
        jobs[0] = Describe(first);
        Load(jobs[0], 0);
        uint32_t iteration = 0;
        for (uint64_t task = first; task < tasks; task += step, ++iteration) {
            const uint32_t slot = iteration & 1, next = slot ^ 1;
            if (task + step < tasks) {
                // The previous vector use of the next input slot must finish.
                if (iteration > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(next);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next);
                }
                jobs[next] = Describe(task + step);
                Load(jobs[next], next);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            auto src = inBuffer_.Get<T>()[slot * (65536 / sizeof(T))];
            auto dst = outBuffer_.Get<T>();
            if (group_ > 1) {
                // Transpose whole 32-byte-aligned blocks in UB with vector copy.
                AscendC::DataCopyParams cp{static_cast<uint16_t>(width_),
                    static_cast<uint16_t>(group_ * sizeof(T) / 32), 0,
                    static_cast<uint16_t>((height_ - 1) * group_ * sizeof(T) / 32)};
                for (uint32_t r = 0; r < height_; ++r)
                    AscendC::DataCopy(dst[r * group_], src[r * width_ * group_], cp);
            } else {
            if constexpr (ByteNative) {
                const uint32_t blocks = tall_ ? width_ / 16 : height_ / 16;
                for (uint32_t rb = 0; rb < blocks; ++rb) {
                    AscendC::TransDataTo5HDParams p{static_cast<bool>(rb & 1), false,
                        static_cast<uint8_t>(width_ / 32), static_cast<uint16_t>(height_), 1};
                    if (tall_) {
                        p.repeatTimes = height_ / 32;
                        p.dstRepStride = 1;
                        p.srcRepStride = width_;
                    }
                    if (p.repeatTimes == 1) p.dstRepStride = p.srcRepStride = 0;
                    AscendC::TransDataTo5HD<int8_t>(dst_[slot][rb], src_[slot][rb], p);
                    p.srcHighHalf = true;
                    AscendC::TransDataTo5HD<int8_t>(dstHigh_[rb], src_[slot][rb], p);
                }
            } else {
            if constexpr (sizeof(T) == 1) {
                auto halfIn = inBuffer_.Get<half>()[slot * 32768 + 16384];
                AscendC::Cast(halfIn, src, AscendC::RoundMode::CAST_NONE, height_ * width_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            constexpr uint32_t vectorCols = sizeof(T) == 4 ? 8 : 16;
            AscendC::TransDataTo5HDParams p{false, false,
                static_cast<uint8_t>(width_ / vectorCols), static_cast<uint16_t>(height_), 1};
            if (tall_) {
                constexpr uint32_t bytes = sizeof(T) == 1 ? 2 : sizeof(T);
                p.repeatTimes = height_ / 16;
                p.dstRepStride = 16 * bytes / 32;
                p.srcRepStride = 16 * width_ * bytes / 32;
            }
            const uint32_t addressBlocks = tall_ ? width_ / vectorCols : height_ / 16;
            for (uint32_t rb = 0; rb < addressBlocks; ++rb) {
                if constexpr (sizeof(T) == 4)
                    AscendC::TransDataTo5HD<uint32_t>(dst_[slot][rb], src_[slot][rb], p);
                else AscendC::TransDataTo5HD<uint16_t>(dst_[slot][rb], src_[slot][rb], p);
            }
            if constexpr (sizeof(T) == 1) {
                AscendC::PipeBarrier<PIPE_V>();
                auto halfOut = outBuffer_.Get<half>()[16384];
                AscendC::Cast(dst, halfOut, AscendC::RoundMode::CAST_NONE, height_ * width_);
            }
            }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
            const Job &j = jobs[slot];
            const uint32_t paddedRow = ((j.rows * group_ * sizeof(T) + 31) / 32) * 32;
            // A 16-byte int8 row is dispatched only when the entire output
            // strip is contiguous and the descriptor below collapses to one block.
            const uint32_t pad = height_ * group_ * sizeof(T) >= paddedRow ?
                (height_ * group_ * sizeof(T) - paddedRow) / 32 : 0;
            AscendC::DataCopyExtParams pOut{static_cast<uint16_t>(j.cols),
                static_cast<uint32_t>(j.rows * group_ * sizeof(T)), pad,
                static_cast<uint32_t>((outStride_ - j.rows * group_) * sizeof(T)), 0};
            if (j.rows == height_ && outStride_ == height_ * group_) {
                pOut.blockCount = 1;
                pOut.blockLen = j.cols * height_ * group_ * sizeof(T);
                pOut.srcStride = pOut.dstStride = 0;
            }
            if constexpr (MergeBatch) {
                pOut.blockCount = fastSize_;
                if (j.rows == height_ && outStride_ == height_) {
                    pOut.blockCount = 1;
                    pOut.blockLen = fastSize_ * height_ * sizeof(T);
                    pOut.srcStride = pOut.dstStride = 0;
                }
                for (uint32_t b = 0; b < j.cols / fastSize_; ++b)
                    AscendC::DataCopyPad(output_[j.output + b * batchStride_],
                        dst[b * fastSize_ * height_], pOut);
            } else AscendC::DataCopyPad(output_[j.output], dst, pOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
private:
    __aicore__ inline Job Describe(uint64_t task)
    {
        uint64_t batch = task / (rowTiles_ * colTiles_);
        const uint64_t tile = task % (rowTiles_ * colTiles_);
        const uint64_t r = tile / colTiles_ * height_, c = tile % colTiles_ * width_;
        uint64_t ib = r * inStride_ + c * group_, ob = c * outStride_ + r * group_, os = 1;
        // Host guarantees width is a multiple of the full fast-axis extent.
        if constexpr (MergeBatch) ob = (c / fastSize_) * batchStride_ + r;
        for (uint32_t rev = 0; rev < rank_; ++rev) {
            const uint32_t a = rank_ - 1 - rev;
            if (a != fast_ && a != row_ && (!MergeBatch || a != batchAxis_) && a <= row_) {
                const uint64_t x = batch % dims_[a];
                batch /= dims_[a];
                ib += x * strides_[a];
                ob += x * os;
            }
            os *= dims_[a];
        }
        return {ib, ob, static_cast<uint32_t>(rows_ - r < height_ ? rows_ - r : height_),
            static_cast<uint32_t>(cols_ - c < width_ ? cols_ - c : width_)};
    }
    __aicore__ inline void Load(const Job &j, uint32_t slot)
    {
        auto local = inBuffer_.Get<T>()[slot * (65536 / sizeof(T))];
        const uint32_t pad = (width_ * group_ * sizeof(T) - ((j.cols * group_ * sizeof(T) + 31) / 32) * 32) / 32;
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(j.rows),
            static_cast<uint32_t>(j.cols * group_ * sizeof(T)),
            static_cast<uint32_t>((inStride_ - j.cols * group_) * sizeof(T)), pad, 0};
        if (j.cols == width_ && inStride_ == width_ * group_) {
            p.blockCount = 1;
            p.blockLen = j.rows * width_ * group_ * sizeof(T);
            p.srcStride = p.dstStride = 0;
        }
        AscendC::DataCopyPadExtParams<T> zeros{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, input_[j.input], p, zeros);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
    }
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuffer_, outBuffer_;
    AscendC::GlobalTensor<T> input_, output_;
    __gm__ const CompetitionTransposeTilingData *data_;
    uint32_t rank_, fast_, row_, width_, height_, group_;
    bool tall_;
    uint32_t batchAxis_, fastSize_;
    uint64_t batchStride_;
    uint64_t rows_, cols_, inStride_, outStride_, rowTiles_, colTiles_;
    uint64_t dims_[16], strides_[16];
    uint64_t src_[2][16][16], dst_[2][16][16];
    uint64_t dstHigh_[16][16];
};
#endif
