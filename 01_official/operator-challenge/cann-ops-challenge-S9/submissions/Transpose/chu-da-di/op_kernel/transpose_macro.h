#ifndef S9_TRANSPOSE_MACRO_H
#define S9_TRANSPOSE_MACRO_H
namespace s9transpose {
// Independent key: preserve the legacy register footprint and instruction stream.
template<typename T, uint32_t kBuffers = 2> class KernelTransposeMacro {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeMacroTilingData& t) {
        data_ = t; x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (t.reserved & 1U) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (t.reserved & 2U) y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        constexpr uint32_t align = 32 / sizeof(T);
        pipe_.InitBuffer(in_, kBuffers, t.rows * (t.cols + align) * sizeof(T));
        pipe_.InitBuffer(out_, kBuffers, t.cols * (t.rows + align) * sizeof(T));
    }
    __aicore__ inline void Process() {
        const auto& b = data_.base;
        const uint32_t inputPitch = data_.cols + 32 / sizeof(T);
        const uint32_t outputPitch = data_.rows + 32 / sizeof(T);
        const uint64_t rowTiles = (b.matrixH + data_.rows - 1) / data_.rows;
        const uint64_t colTiles = (b.matrixW + data_.cols - 1) / data_.cols;
        const uint64_t tasksPerMatrix = rowTiles * colTiles;
        const uint64_t total = b.outer * tasksPerMatrix;
        const uint64_t small = total / AscendC::GetBlockNum(), extra = total % AscendC::GetBlockNum();
        const uint64_t bid = AscendC::GetBlockIdx();
        const uint64_t start = bid * small + (bid < extra ? bid : extra);
        const uint64_t end = start + small + (bid < extra ? 1 : 0);
        for (uint64_t task = start; task < end; ++task) {
            uint64_t outer = task / tasksPerMatrix;
            const uint64_t tile = task % tasksPerMatrix;
            const uint64_t row = tile / colTiles * data_.rows;
            const uint64_t col = tile % colTiles * data_.cols;
            const uint32_t nr = b.matrixH - row < data_.rows ? b.matrixH - row : data_.rows;
            const uint32_t nc = b.matrixW - col < data_.cols ? b.matrixW - col : data_.cols;
            uint64_t inputBase = outer * b.matrixH * b.matrixW;
            uint64_t outputBase = inputBase;
            uint64_t rowStride = b.matrixW, colStride = b.matrixH;
            if (data_.permuted != 0) {
                inputBase = 0; outputBase = 0; uint64_t os = 1;
                const uint32_t rowAxis = b.perm[b.rank - 1];
                rowStride = b.inputStride[rowAxis]; colStride = data_.columnStride;
                for (int32_t axis = b.rank - 1; axis >= 0; --axis) {
                    const uint32_t a = b.perm[axis];
                    if (a != rowAxis && a != b.rank - 1) {
                        const uint64_t c = outer % b.outputShape[axis]; outer /= b.outputShape[axis];
                        inputBase += c * b.inputStride[a]; outputBase += c * os;
                    }
                    os *= b.outputShape[axis];
                }
            }
            auto input = in_.template AllocTensor<T>();
            constexpr uint32_t alignment = 32 / sizeof(T);
            const uint32_t alignedCols = (nc + alignment - 1) / alignment * alignment;
            AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr), static_cast<uint32_t>(nc * sizeof(T)),
                static_cast<uint32_t>((rowStride - nc) * sizeof(T)),
                (inputPitch - alignedCols) / alignment, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(input, x_[inputBase + row * rowStride + col], ip, pad);
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
                static_cast<uint32_t>((colStride - nr) * sizeof(T)), 0};
            AscendC::DataCopyPad(y_[outputBase + col * colStride + row], output, op);
            out_.FreeTensor(output);
        }
    }
private:
    TransposeMacroTilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_;
    AscendC::GlobalTensor<T> x_, y_;
};
}
#endif
