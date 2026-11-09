#ifndef S9_TRANSPOSE_COLUMN_OWNER_H
#define S9_TRANSPOSE_COLUMN_OWNER_H
namespace s9transpose {
template<typename T> class KernelTransposeColumnOwner {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeColumnOwnerTilingData& t) {
        data_ = t; x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (t.cachePolicy & 1U) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (t.cachePolicy & 2U) y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint32_t buffers = t.bufferCount == 0 ? 2 : t.bufferCount;
        const uint32_t inputRows = t.rows < 16 ? 16 : t.rows;
        pipe_.InitBuffer(in_, buffers, inputRows * (t.cols + alignment) * sizeof(T));
        pipe_.InitBuffer(out_, buffers, t.cols * (t.rows + alignment) * sizeof(T));
    }

    __aicore__ inline void ProcessRowPermutation() {
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint64_t h = data_.base.matrixH, w = data_.base.matrixW;
        const uint32_t inputPitch = data_.cols + alignment, outputPitch = data_.rows + alignment;
        const uint64_t units = (w + alignment - 1) / alignment;
        const uint64_t cores = AscendC::GetBlockNum(), bid = AscendC::GetBlockIdx();
        const uint64_t small = units / cores, extra = units % cores;
        const uint64_t first = (bid * small + (bid < extra ? bid : extra)) * alignment;
        const uint64_t last0 = first + (small + (bid < extra ? 1 : 0)) * alignment;
        const uint64_t last = last0 < w ? last0 : w;
        for (uint64_t col = first; col < last; col += data_.cols) {
            const uint32_t nc = static_cast<uint32_t>(last-col < data_.cols ? last-col : data_.cols);
            const uint32_t alignedCols = (nc + alignment - 1) / alignment * alignment;
            for (uint64_t row = 0; row < h; row += data_.rows) {
                const uint32_t nr = static_cast<uint32_t>(h-row < data_.rows ? h-row : data_.rows);
                auto input = in_.template AllocTensor<T>();
                for (uint32_t done = 0; done < nr;) {
                    const uint64_t rr = row + done, cell = rr / data_.rowInner;
                    const uint64_t srcRow = ((cell % data_.rowA) * data_.rowB + cell / data_.rowA) * data_.rowInner + rr % data_.rowInner;
                    const uint32_t run = static_cast<uint32_t>(data_.rowInner - rr % data_.rowInner < nr-done ? data_.rowInner - rr % data_.rowInner : nr-done);
                    AscendC::DataCopyExtParams ip{static_cast<uint16_t>(run), static_cast<uint32_t>(nc * sizeof(T)),
                        static_cast<uint32_t>((w - nc) * sizeof(T)), (inputPitch - alignedCols) / alignment, 0};
                    AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};
                    AscendC::DataCopyPad(input[done * inputPitch], x_[srcRow * w + col], ip, pp);
                    done += run;
                }
                in_.EnQue(input); input = in_.template DeQue<T>(); auto output = out_.template AllocTensor<T>();
            uint64_t src[16], dst[16];
            const uint32_t repeats = (nc + alignment - 1) / alignment;
            const uint32_t rtiles = (nr + 15) / 16;
            for (uint32_t rt = 0; rt < rtiles; ++rt) {
                for (uint32_t i = 0; i < 16; ++i)
                    src[i] = reinterpret_cast<uint64_t>(input[(rt * 16 + i) * inputPitch].GetPhyAddr());
                if constexpr (sizeof(T) == 1) {
                    // Native INT8 low/high source halves; paired row tiles fill
                    // both halves of each destination datablock without casts.
                    for (uint32_t halfCol = 0; halfCol < 2; ++halfCol) {
                        for (uint32_t i = 0; i < 16; ++i)
                            dst[i] = reinterpret_cast<uint64_t>(output[(halfCol * 16 + i) * outputPitch + (rt / 2) * 32].GetPhyAddr());
                        AscendC::TransDataTo5HDParams tp{rt % 2 != 0, halfCol != 0,
                            static_cast<uint8_t>(repeats), static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                            static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                        AscendC::TransDataTo5HD<T>(dst, src, tp);
                    }
                } else {
                    constexpr uint32_t groups = 16 / alignment;
                    for (uint32_t c = 0; c < alignment; ++c)
                        for (uint32_t g = 0; g < groups; ++g)
                            dst[c * groups + g] = reinterpret_cast<uint64_t>(output[c * outputPitch + rt * 16 + g * alignment].GetPhyAddr());
                    AscendC::TransDataTo5HDParams tp{false, false, static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                        static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                    AscendC::TransDataTo5HD<T>(dst, src, tp);
                }
            }
                out_.EnQue(output); in_.FreeTensor(input); output = out_.template DeQue<T>();
                const uint32_t alignedRows = (nr + alignment - 1) / alignment * alignment;
                AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc), static_cast<uint32_t>(nr * sizeof(T)),
                    (outputPitch - alignedRows) / alignment, static_cast<uint32_t>((h-nr)*sizeof(T)), 0};
                AscendC::DataCopyPad(y_[col*h+row],output,op); out_.FreeTensor(output);
            }
        }
    }
    __aicore__ inline void Process() {
        if (data_.rowMode != 0) { ProcessRowPermutation(); return; }
        const auto& b = data_.base;
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint32_t inputPitch = data_.cols + alignment;
        const uint32_t outputPitch = data_.rows + alignment;
        const uint64_t h = b.outputShape[b.rank - 1];
        const uint64_t w = b.shape[b.rank - 1];
        const uint64_t rowStride = b.inputStride[b.perm[b.rank - 1]];
        const bool wholeRange = data_.rangeMode != 0;
        const uint64_t unit = wholeRange ? alignment : data_.cols;
        const uint64_t tasks = (data_.totalColumns + unit - 1) / unit;
        const uint64_t small = tasks / AscendC::GetBlockNum(), extra = tasks % AscendC::GetBlockNum();
        const uint64_t bid = AscendC::GetBlockIdx();
        const uint64_t start = bid * small + (bid < extra ? bid : extra);
        const uint64_t end = start + small + (bid < extra ? 1 : 0);
        for (uint64_t task = start; task < end; task = wholeRange ? end : task + 1) {
            uint64_t flat = task * unit;
            const uint64_t limit = wholeRange ? end * unit : flat + data_.cols;
            const uint64_t stop = limit < data_.totalColumns ? limit : data_.totalColumns;
            while (flat < stop) {
                const uint64_t col = flat % w;
                uint64_t span = w - col < stop - flat ? w - col : stop - flat;
                if (span > data_.cols) span = data_.cols;
                const uint32_t nc = static_cast<uint32_t>(span);
                uint64_t coordinates[kMaxRank] = {};
                uint64_t remaining = flat * data_.innerMatrices * h, inputBase = 0;
                for (int32_t a = static_cast<int32_t>(b.rank) - 1; a >= 0; --a) {
                    coordinates[a] = remaining % b.outputShape[a];
                    remaining /= b.outputShape[a];
                    inputBase += coordinates[a] * b.inputStride[b.perm[a]];
                }
                for (uint64_t inner = 0; inner < data_.innerMatrices; ++inner) {
                    const uint64_t outputBase = (flat * data_.innerMatrices + inner) * h;
                    for (uint64_t row = 0; row < h; row += data_.rows) {
                        const uint32_t nr = static_cast<uint32_t>(h - row < data_.rows ? h - row : data_.rows);
                        auto input = in_.template AllocTensor<T>();
                        const uint32_t alignedCols = (nc + alignment - 1) / alignment * alignment;
                        AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr), static_cast<uint32_t>(nc * sizeof(T)),
                            static_cast<uint32_t>((rowStride - nc) * sizeof(T)), (inputPitch - alignedCols) / alignment, 0};
                        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                        AscendC::DataCopyPad(input, x_[inputBase + row * rowStride], ip, pad);
                        in_.EnQue(input); input = in_.template DeQue<T>();
                        auto output = out_.template AllocTensor<T>();
            uint64_t src[16], dst[16];
            const uint32_t repeats = (nc + alignment - 1) / alignment;
            const uint32_t rtiles = (nr + 15) / 16;
            for (uint32_t rt = 0; rt < rtiles; ++rt) {
                for (uint32_t i = 0; i < 16; ++i)
                    src[i] = reinterpret_cast<uint64_t>(input[(rt * 16 + i) * inputPitch].GetPhyAddr());
                if constexpr (sizeof(T) == 1) {
                    // Native INT8 low/high source halves; paired row tiles fill
                    // both halves of each destination datablock without casts.
                    for (uint32_t halfCol = 0; halfCol < 2; ++halfCol) {
                        for (uint32_t i = 0; i < 16; ++i)
                            dst[i] = reinterpret_cast<uint64_t>(output[(halfCol * 16 + i) * outputPitch + (rt / 2) * 32].GetPhyAddr());
                        AscendC::TransDataTo5HDParams tp{rt % 2 != 0, halfCol != 0,
                            static_cast<uint8_t>(repeats), static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                            static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                        AscendC::TransDataTo5HD<T>(dst, src, tp);
                    }
                } else {
                    constexpr uint32_t groups = 16 / alignment;
                    for (uint32_t c = 0; c < alignment; ++c)
                        for (uint32_t g = 0; g < groups; ++g)
                            dst[c * groups + g] = reinterpret_cast<uint64_t>(output[c * outputPitch + rt * 16 + g * alignment].GetPhyAddr());
                    AscendC::TransDataTo5HDParams tp{false, false, static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                        static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                    AscendC::TransDataTo5HD<T>(dst, src, tp);
                }
            }

                        out_.EnQue(output); in_.FreeTensor(input);
                        output = out_.template DeQue<T>();
                        const uint32_t alignedRows = (nr + alignment - 1) / alignment * alignment;
                        AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc), static_cast<uint32_t>(nr * sizeof(T)),
                            (outputPitch - alignedRows) / alignment,
                            static_cast<uint32_t>((data_.columnStride - nr) * sizeof(T)), 0};
                        AscendC::DataCopyPad(y_[outputBase + row], output, op);
                        out_.FreeTensor(output);
                    }
                    // The innermost output axis is complete; carry through
                    // outer coordinates instead of decoding outputBase again.
                    for (int32_t a = static_cast<int32_t>(b.rank) - 2; a >= 0; --a) {
                        const uint64_t stride = b.inputStride[b.perm[a]];
                        inputBase += stride;
                        if (++coordinates[a] < b.outputShape[a]) break;
                        coordinates[a] = 0;
                        inputBase -= b.outputShape[a] * stride;
                    }
                }
                flat += nc;
            }
        }
    }
private:
    TransposeColumnOwnerTilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_;
    AscendC::GlobalTensor<T> x_,y_;
};
}
#endif
