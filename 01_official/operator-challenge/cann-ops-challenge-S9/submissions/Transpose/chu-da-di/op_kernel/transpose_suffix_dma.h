#ifndef S9_TRANSPOSE_SUFFIX_DMA_H
#define S9_TRANSPOSE_SUFFIX_DMA_H
namespace s9transpose {
// P(prefix) tensor I(inner): copy whole contiguous fibers with strided MTE2.
// The bound queue synchronizes MTE2 directly with MTE3; no vector arithmetic.
template<typename T> class KernelTransposeSuffixDma {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData& t) {
        data_ = t; x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(queue_, 2, 32768);
    }
    __aicore__ inline void Process() {
        const uint64_t inner = data_.innerElements;
        const uint32_t bytes = static_cast<uint32_t>(inner * sizeof(T));
        const uint32_t rowBytes = (bytes + 31) / 32 * 32;
        const uint32_t capacity = 32768 / rowBytes;
        const uint64_t quantum = data_.rowOwnershipQuantum;
        const uint64_t units = (data_.outputBlocks + quantum - 1) / quantum;
        const uint64_t small = units / AscendC::GetBlockNum(), extra = units % AscendC::GetBlockNum();
        const uint64_t bid = AscendC::GetBlockIdx();
        const uint64_t first = bid * small + (bid < extra ? bid : extra);
        const uint64_t last = first + small + (bid < extra ? 1 : 0);
        uint64_t block = first * quantum;
        const uint64_t end = last * quantum < data_.outputBlocks ? last * quantum : data_.outputBlocks;
        const uint64_t width = data_.outputShape[data_.rank - 2];
        const uint64_t stride = data_.inputStride[data_.perm[data_.rank - 2]];
        while (block < end) {
            uint64_t rows = width - block % width;
            if (rows > end - block) rows = end - block;
            if (rows > capacity) rows = capacity;
            uint64_t linear = block * inner, offset = 0;
            for (int32_t a = data_.rank - 1; a >= 0; --a) {
                const uint64_t c = linear % data_.outputShape[a]; linear /= data_.outputShape[a];
                offset += c * data_.inputStride[data_.perm[a]];
            }
            auto tile = queue_.template AllocTensor<T>();
            AscendC::DataCopyExtParams ip{static_cast<uint16_t>(rows), bytes,
                static_cast<uint32_t>((stride - inner) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(tile, x_[offset], ip, pad);
            queue_.EnQue(tile); tile = queue_.template DeQue<T>();
            AscendC::DataCopyExtParams op{static_cast<uint16_t>(rows), bytes, 0, 0, 0};
            AscendC::DataCopyPad(y_[block * inner], tile, op);
            queue_.FreeTensor(tile); block += rows;
        }
    }
private:
    TransposeTilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> queue_;
    AscendC::GlobalTensor<T> x_,y_;
};
}
#endif
