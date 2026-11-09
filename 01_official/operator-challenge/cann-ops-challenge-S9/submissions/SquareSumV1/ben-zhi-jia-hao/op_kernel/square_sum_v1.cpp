/*
 * SquareSumV1 kernel:  y = sum(x * x, axis, keep_dims)
 *
 * The host has already collapsed the shape into an alternating chain of kept
 * and reduced groups (see op_host/square_sum_v1.cpp).  Only two shapes of work
 * reach the kernel:
 *
 *   innerIsReduce == 1  a contiguous run folds into ONE output scalar
 *   innerIsReduce == 0  a contiguous run folds ELEMENT-WISE onto an output run
 *
 * Squaring and accumulating are one instruction:  MulAddDst(acc, x, x, n)
 * computes acc += x*x.  For float16 the mixed form MulAddDst<float, half>
 * also folds in the widening, so a half input costs a single vector pass and
 * never materialises the squares.  bfloat16 has no mixed form, so it pays one
 * extra Cast.  All accumulation is fp32 regardless of the tensor dtype.
 */

#include "kernel_operator.h"
#include "cols_raw_f32.h"
#include "cols_raw_bf16.h"
#include "layer_pairs.h"

#include <type_traits>

using namespace AscendC;

namespace {
constexpr uint32_t STAGE_SLOTS = 240;
constexpr uint32_t STAGE_PITCH = 8;   // floats between staged scalars (32 B apart)
constexpr uint32_t STAGE_FLOATS = STAGE_SLOTS * STAGE_PITCH;
constexpr uint32_t MAX_GROUP = 4;
// Padded element capacity for KEY_ROWS_ONE. It must match
// squaresum::ONE_TILE_ELEMS in the host planner because the kernel UB map
// and the host promotion bound use the same capacity.
constexpr uint32_t SS_ONE_TILE_ELEMS = 2048;

__aicore__ inline uint64_t CeilDivU(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

// floats rounded up to a whole 32 B block -- the same rounding TPipe::InitBuffer
// applies to every buffer length, so an arena laid out with this reproduces the
// address each separate TBuf would have been given.
constexpr uint32_t BLK_FLOATS = 32 / sizeof(float);
__aicore__ inline uint32_t AlignFloats(uint32_t n)
{
    return ((n + BLK_FLOATS - 1) / BLK_FLOATS) * BLK_FLOATS;
}

template <typename T, bool kAtomicRed = false>
class SquareSumV1Kernel {
public:
    __aicore__ inline SquareSumV1Kernel() {}

    // Read tiling fields directly through the GM pointer. Keep frequently
    // branched-on values as members and read other fields at their use sites.
    // KEY is a compile-time tiling key, so route-specific initialization omits
    // unneeded fields. Remaining copies have static bounds.
    template <uint64_t KEY>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                                TPipe* pipe)
    {
        const __gm__ SquareSumV1TilingData* t =
            reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);

        usedCores_ = t->usedCores;
        accLen_ = t->accLen;
        ubChunk_ = t->ubChunk;
        workLen_ = t->workLen;
        outStage_ = t->outStage;
        tmpLen_ = t->tmpLen;
        innerLen_ = t->innerLen;
        outTiles_ = t->outTiles;

        if constexpr (KEY == 1 || KEY == 14) {
            // ProcessRows walks rows of a `K R` form directly; it never touches
            // the odometer, the reduction split or the workspace.
            rowBatch_ = t->rowBatch;
            rowPitch_ = t->rowPitch;
        } else {
            // Only what this route branches on is copied.  Everything else is
            // read through t_ where it is used -- once per function, hoisted
            // into a local when a loop needs it -- so the prologue neither
            // loads nor spills it.
            nK_ = t->nOuterK;
            if constexpr (KEY != 2) {
                // VectorTasksContig walks one contiguous span, so it never
                // enters RunInit/RunAdvance/RunNext and never asks how many
                // outer reduced groups there are.
                nR_ = t->nOuterR;
            }
            redChunks_ = t->redChunks;
            // The four odometer arrays are NOT copied.  They are 128 bytes,
            // half the tiling struct, and copying them is 16 loads and 16
            // stores in the prologue whichever of them the shape actually
            // has -- most shapes have nOuterK <= 1 and nOuterR <= 1, so 12 to
            // 16 of the 16 slots are never read.  The odometer helpers read
            // them through this pointer at the point of use instead, which is
            // once per task (TileBase) and once per DMA batch (RunAdvance).
            t_ = t;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (redChunks_ > 1) {
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
            if constexpr (kAtomicRed) {
                wsU32_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint32_t*>(GetUserWorkspace(workspace)));
                nonce_ = reinterpret_cast<uint64_t>(y);
            }
        }

        // Use one VECCALC arena with disjoint, 32-byte-aligned acc, tmp, work, and
        // stage regions. All four can be live simultaneously.
        //
        // TBuf does not supply queue events. Synchronization remains in the input
        // and output queues and in the explicit barriers around scalar/vector
        // hand-offs; combining these regions does not replace those dependencies.
        tmpAt_ = AlignFloats(accLen_);
        workAt_ = tmpAt_ + AlignFloats(tmpLen_);
        stageAt_ = workAt_ + AlignFloats(workLen_);

        pipe->InitBuffer(inQue_, 2, ubChunk_ * sizeof(T));
        pipe->InitBuffer(ubBuf_, (stageAt_ + ((KEY == 2 || KEY == 10 || KEY == 14) ? 0U : STAGE_FLOATS)) * sizeof(float));
        pipe->InitBuffer(outQue_, 1, outStage_ * sizeof(float));
    }

    // The four regions of the arena.  Sizes are derived from the offsets rather
    // than stored, so the merge adds three stack words and not seven.
    __aicore__ inline LocalTensor<float> AccT()
    {
        return ubBuf_.template GetWithOffset<float>(tmpAt_, 0);
    }
    __aicore__ inline LocalTensor<float> TmpT()
    {
        return ubBuf_.template GetWithOffset<float>(workAt_ - tmpAt_, tmpAt_ * sizeof(float));
    }
    __aicore__ inline LocalTensor<float> WorkT()
    {
        return ubBuf_.template GetWithOffset<float>(stageAt_ - workAt_, workAt_ * sizeof(float));
    }
    __aicore__ inline LocalTensor<float> StageT()
    {
        return ubBuf_.template GetWithOffset<float>(STAGE_FLOATS, stageAt_ * sizeof(float));
    }

    // ---------------------------------------------------------------- routes
    template <uint64_t KEY>
    __aicore__ inline void ProcessGeneric()
    {
        ProcessGenericStage1<KEY>();
        if constexpr (kAtomicRed) {
            if (redChunks_ > 1) {
                SyncAll<true>();
                if constexpr (KEY == 2 || KEY == 3) {
                    CombinePolledVec(static_cast<uint32_t>(innerLen_));
                } else {
                    CombinePolled();
                }
            }
        } else {
            if (redChunks_ > 1) {
                SyncAll();
                ProcessGenericStage2<KEY>();
            }
        }
    }

    // Separate stage entry points allow the two halves of the generic reduction
    // to be invoked with external ordering. ProcessGeneric retains the normal
    // combined entry. Compile-time KEY selection instantiates only the route
    // bodies used by the selected kernel variant.
    template <uint64_t KEY>
    __aicore__ inline void ProcessGenericStage1()
    {
        const uint64_t tasks = t_->indepUnits * static_cast<uint64_t>(redChunks_);
        uint64_t lo = 0;
        uint64_t hi = 0;
        SplitRange(tasks, lo, hi);

        if constexpr (KEY == 0) {
            ScalarTasks(lo, hi);
        } else if constexpr (KEY == 2) {
            VectorTasksContig(lo, hi);
        } else {
            VectorTasks(lo, hi);
        }
    }

    template <uint64_t KEY>
    __aicore__ inline void ProcessGenericStage2()
    {
        if (redChunks_ > 1) {
            Combine();
        }
    }

    // Many short contiguous rows: one padded multi-row DMA, one reduce burst.
    __aicore__ inline void ProcessRows()
    {
        uint64_t lo = 0;
        uint64_t hi = 0;
        SplitRange(outTiles_, lo, hi);

        const uint32_t rowElems = static_cast<uint32_t>(innerLen_);
        const uint32_t dmaPitch = ((rowElems * sizeof(T) + 31U) & ~31U) / sizeof(T);
        const uint32_t gap = rowPitch_ - dmaPitch;
        const uint32_t gapBlocks = gap * sizeof(T) / 32U;
        const uint8_t rightPad = static_cast<uint8_t>(dmaPitch - rowElems);
        DataCopyPadExtParams<T> padParams{rightPad != 0, 0, rightPad, static_cast<T>(0)};
        // Rows of a K R layout are adjacent in GM. If the row length is a multiple
        // of 32 bytes, padded UB pitch equals the row length and the batch can use
        // one contiguous transfer with the same destination layout.
        const bool contigRows = (rowPitch_ == rowElems);

        const bool blockReduction = rowPitch_ > 64 && rowPitch_ <= 8192;
        const uint32_t reduceRows = (rowBatch_ + 7U) & ~7U;
        uint32_t rowBlocks = rowPitch_ / 8U;
        uint32_t pairPasses = 0;
        while ((rowBlocks & 1U) == 0) { rowBlocks >>= 1; ++pairPasses; }
        if (blockReduction && rowBlocks > 1) {
            BuildTransposeIndex(reduceRows, rowBlocks, reduceRows * rowBlocks);
        }

        const uint32_t firstRows = static_cast<uint32_t>((hi - lo) < rowBatch_ ? hi - lo : rowBatch_);
        auto first = inQue_.template AllocTensor<T>();
        DataCopyExtParams firstCp = contigRows
            ? DataCopyExtParams{1, static_cast<uint32_t>(firstRows * rowElems * sizeof(T)), 0, 0, 0}
            : DataCopyExtParams{static_cast<uint16_t>(firstRows), static_cast<uint32_t>(rowElems * sizeof(T)), 0, gapBlocks, 0};
        DataCopyPad(first, xGm_[lo * innerLen_], firstCp, padParams);
        inQue_.EnQue(first);

        for (uint64_t t = lo; t < hi; t += rowBatch_) {
            const uint32_t nb = static_cast<uint32_t>(
                (hi - t < static_cast<uint64_t>(rowBatch_)) ? (hi - t) : static_cast<uint64_t>(rowBatch_));

            if (t + nb < hi) {
                const uint32_t next = static_cast<uint32_t>((hi - t - nb) < rowBatch_ ? hi - t - nb : rowBatch_);
                auto ahead = inQue_.template AllocTensor<T>();
                DataCopyExtParams cp = contigRows
                    ? DataCopyExtParams{1, static_cast<uint32_t>(next * rowElems * sizeof(T)), 0, 0, 0}
                    : DataCopyExtParams{static_cast<uint16_t>(next), static_cast<uint32_t>(rowElems * sizeof(T)), 0, gapBlocks, 0};
                DataCopyPad(ahead, xGm_[(t + nb) * innerLen_], cp, padParams);
                inQue_.EnQue(ahead);
            }
            auto in = inQue_.template DeQue<T>();

            if (gap != 0) {
                const uint8_t stride = rowPitch_ * sizeof(T) / 32U;
                if constexpr (sizeof(T) == 2) {
                    Duplicate<uint16_t>(in.template ReinterpretCast<uint16_t>()[dmaPitch],
                        uint16_t(0), uint64_t(gap), uint8_t(nb), uint16_t(1), stride);
                } else {
                    Duplicate<float>(in.template ReinterpretCast<float>()[dmaPitch],
                        0.0f, uint64_t(gap), uint8_t(nb), uint16_t(1), stride);
                }
                pipe_barrier(PIPE_V);
            }

            LocalTensor<float> sq = AccT();
            const uint32_t n = nb * rowPitch_;
            if constexpr (std::is_same_v<T, float>) {
                Mul(sq, in, in, static_cast<int32_t>(n));
            } else {
                Cast(sq, in, RoundMode::CAST_NONE, static_cast<int32_t>(n));
                Mul(sq, sq, sq, static_cast<int32_t>(n));
            }
            inQue_.FreeTensor(in);

            LocalTensor<T> out = outQue_.template AllocTensor<T>();
            LocalTensor<float> dst = (std::is_same_v<T, float>)
                                         ? out.template ReinterpretCast<float>()
                                         : TmpT();
            if (rowPitch_ <= 64) {
                // one vector repeat covers a whole padded row
                WholeReduceSum<float>(dst, sq, static_cast<int32_t>(rowPitch_),
                                      static_cast<int32_t>(nb), 1, 1,
                                      static_cast<int32_t>(rowPitch_ * sizeof(float) / 32));
            } else if (blockReduction) {
                auto partial = TmpT();
                ReduceBlocks(partial, sq, n);
                uint32_t partialCount = n / 8U;
                for (uint32_t pass = 0; pass < pairPasses; ++pass) {
                    ReducePairs(partial, partialCount);
                    partialCount >>= 1;
                }
                if (rowBlocks == 1U) {
                    if constexpr (std::is_same_v<T, float>) Adds(dst, partial, 0.0f, nb);
                    else dst = partial;
                } else {
                Gather(sq, partial, WorkT().template ReinterpretCast<uint32_t>(), 0U, rowBlocks * reduceRows);
                pipe_barrier(PIPE_V);
                uint32_t power = 1;
                while ((power << 1) <= rowBlocks) power <<= 1;
                if (rowBlocks > power) {
                    Add(sq, sq, sq[power * reduceRows], (rowBlocks-power) * reduceRows);
                    pipe_barrier(PIPE_V);
                }
                for (uint32_t half = power >> 1; half; half >>= 1) {
                    Add(sq, sq, sq[half * reduceRows], half * reduceRows);
                    pipe_barrier(PIPE_V);
                }
                if constexpr (std::is_same_v<T, float>) {
                    Adds(dst, sq, 0.0f, nb);
                } else {
                    dst = sq;
                }
                }
            } else {
                LocalTensor<float> stage = StageT();
                const uint32_t stride = rowPitch_ / 8U;
                WholeReduceSum<float>(dst, sq, 64, nb, 1, 1, stride);
                for (uint32_t k = 64; k < rowPitch_; k += 64) {
                    const uint32_t count = (rowPitch_ - k) < 64 ? rowPitch_ - k : 64;
                    WholeReduceSum<float>(stage, sq[k], count, nb, 1, 1, stride);
                    pipe_barrier(PIPE_V);
                    Add(dst, dst, stage, nb);
                    pipe_barrier(PIPE_V);
                }
            }
            CastToOut(out, dst, nb);
            outQue_.EnQue(out);
            out = outQue_.template DeQue<T>();
            DataCopyExtParams wp{1, static_cast<uint32_t>(nb * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[t], out, wp);
            outQue_.FreeTensor(out);
        }
    }

    // Read the matrix as one flat span. Only the small array of 8-element
    // block sums is rearranged; at most 15 boundary values per row are kept
    // separately. Every input participates exactly once, including odd rows.
    __aicore__ inline void ProcessFlatRows()
    {
        uint64_t lo = 0, hi = 0;
        SplitRange(outTiles_, lo, hi);
        if (lo == hi) return;
        const uint32_t width = static_cast<uint32_t>(innerLen_);
        constexpr bool packed = std::is_same_v<T,bfloat16_t>;
        constexpr uint32_t unit = packed ? 16 : 8;
        const uint32_t blocks = width/unit-1U;
        const uint32_t edges = width-blocks*unit;
        const uint32_t pitch = rowPitch_;
        const uint32_t partialLen = packed ? ((ubChunk_/unit+63U)/64U)*72U
                                          : (ubChunk_/unit+7U)&~7U;
        uint32_t firstCount = static_cast<uint32_t>(hi-lo < rowBatch_ ? hi-lo : rowBatch_)*width;
        auto first = inQue_.template AllocTensor<T>();
        CopyFlatRows(first, lo*width, firstCount);
        inQue_.EnQue(first);
        BuildFlatRowIndices(width, blocks, edges, pitch);
        for (uint64_t row = lo; row < hi; row += rowBatch_) {
            const uint32_t nb = static_cast<uint32_t>(hi-row < rowBatch_ ? hi-row : rowBatch_);
            if (row+nb < hi) {
                const uint32_t next = static_cast<uint32_t>(hi-row-nb < rowBatch_ ? hi-row-nb : rowBatch_)*width;
                auto ahead = inQue_.template AllocTensor<T>();
                CopyFlatRows(ahead, (row+nb)*width, next);
                inQue_.EnQue(ahead);
            }
            auto in = inQue_.template DeQue<T>();
            auto sq = AccT();
            const uint32_t count = (nb*width+unit-1U)&~(unit-1U);
            auto partial = TmpT();
            auto offsets = WorkT().template ReinterpretCast<uint32_t>();
            const uint32_t rawEdgeLen = packed ? ((edges*pitch/2U+7U)&~7U) : 0U;
            auto edge = partial[partialLen+rawEdgeLen];
            if constexpr (packed) {
                auto raw = partial[partialLen].template ReinterpretCast<uint16_t>();
                auto inputBits = in.template ReinterpretCast<uint16_t>();
                Gather(raw, inputBits, offsets[blocks*pitch], 0U, edges*pitch);
                pipe_barrier(PIPE_V);
                Cast(edge, raw.template ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, edges*pitch);
                pipe_barrier(PIPE_V);
                Mul(edge, edge, edge, edges*pitch);
                auto mask = partial[tmpLen_-8].template ReinterpretCast<uint32_t>();
                Duplicate<uint32_t>(mask, 0xffff0000U, 8);
                auto inputPairs = in.template ReinterpretCast<float>();
                auto even = sq.template ReinterpretCast<uint32_t>();
                const BinaryRepeatParams rp{1,1,0,8,8,0};
                // The pair result occupies exactly the original two BF16
                // elements. Only a bounded even-value temporary is needed;
                // the queues can therefore hold more complete input rows.
                for (uint32_t off=0; off<count/2U; off+=accLen_) {
                    const uint32_t pairs = count/2U-off < accLen_ ? count/2U-off : accLen_;
                    ShiftLeft<uint32_t>(even, in.template ReinterpretCast<uint32_t>()[off], 16, pairs);
                    pipe_barrier(PIPE_V);
                    uint32_t done = 0;
                    const uint32_t halfCount = pairs*2U;
                    while (done+128U <= halfCount) {
                        uint32_t repeats = (halfCount-done)/128U;
                        if (repeats > 255U) repeats = 255U;
                        And<uint16_t>(inputBits[off*2U+done], inputBits[off*2U+done],
                            mask.template ReinterpretCast<uint16_t>(), 128, repeats, rp);
                        done += repeats*128U;
                    }
                    if (done < halfCount) And<uint16_t>(inputBits[off*2U+done], inputBits[off*2U+done],
                        mask.template ReinterpretCast<uint16_t>(), halfCount-done, 1, rp);
                    pipe_barrier(PIPE_V);
                    Mul(inputPairs[off], inputPairs[off], inputPairs[off], pairs);
                    pipe_barrier(PIPE_V);
                    MulAddDst<float,float>(inputPairs[off], sq, sq, pairs);
                    pipe_barrier(PIPE_V);
                }
            } else {
                if constexpr (std::is_same_v<T,float>) Mul(sq, in, in, count);
                else {
                    Cast(sq, in, RoundMode::CAST_NONE, count);
                    pipe_barrier(PIPE_V);
                    Mul(sq, sq, sq, count);
                }
                pipe_barrier(PIPE_V);
                Gather(edge, sq, offsets[blocks*pitch], 0U, edges*pitch);
            }
            pipe_barrier(PIPE_V);
            FoldFlatPlanes(edge, edges, pitch);
            if constexpr (packed) ReducePaddedBlocks(partial, in.template ReinterpretCast<float>(), count/2U);
            else ReduceBlocks(partial, sq, count);
            inQue_.FreeTensor(in);
            Gather(sq, partial, offsets, 0U, blocks*pitch);
            pipe_barrier(PIPE_V);
            FoldFlatPlanes(sq, blocks, pitch);
            Add(sq, sq, edge, nb);
            pipe_barrier(PIPE_V);
            StoreTyped(sq, nb, row);
        }
    }

private:
    __aicore__ inline void CopyFlatRows(LocalTensor<T> dst, uint64_t start, uint32_t count)
    {
        constexpr uint32_t alignment = 32U/sizeof(T);
        const uint32_t rounded = (count+alignment-1U)&~(alignment-1U);
        if (start+rounded <= outTiles_*innerLen_) {
            // Extra lanes lie inside the input tensor, outside every live
            // row's partition; their block sums and edges are never emitted.
            DataCopy(dst, xGm_[start], rounded);
        } else {
            const uint32_t prefix = count&~(alignment-1U);
            if (prefix) DataCopy(dst, xGm_[start], prefix);
            if (prefix < count) {
                DataCopyExtParams cp{1,(count-prefix)*static_cast<uint32_t>(sizeof(T)),0,0,0};
                DataCopyPadExtParams<T> pad{true,0,
                    static_cast<uint8_t>(rounded-count),static_cast<T>(0)};
                DataCopyPad(dst[prefix], xGm_[start+prefix], cp, pad);
            }
        }
    }

    __aicore__ inline void BuildFlatRowIndices(uint32_t width, uint32_t blocks,
                                               uint32_t edges, uint32_t pitch)
    {
        auto dst = WorkT().template ReinterpretCast<int32_t>();
        auto q = AccT().template ReinterpretCast<int32_t>();
        auto r = TmpT().template ReinterpretCast<int32_t>();
        constexpr bool packed = std::is_same_v<T,bfloat16_t>;
        constexpr uint32_t unit = packed ? 16 : 8;
        constexpr uint32_t unitShift = packed ? 4 : 3;
        uint32_t shift = 0;
        while ((1U<<shift) < pitch) ++shift;
        const uint32_t count = blocks*pitch;
        ArithProgression<int32_t>(dst, 0, 1, count);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q, dst, shift, count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r, q, shift, count);
        pipe_barrier(PIPE_V);
        Sub(dst, dst, r, count);
        pipe_barrier(PIPE_V);
        Mins<int32_t>(dst, dst, static_cast<int32_t>(rowBatch_-1), count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(dst, dst, static_cast<int32_t>(width), count);
        pipe_barrier(PIPE_V);
        Adds<int32_t>(dst, dst, unit-1U, count);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(dst, dst, unitShift, count);
        pipe_barrier(PIPE_V);
        Add(dst, dst, q, count);
        pipe_barrier(PIPE_V);
        if constexpr (packed) {
            // Add one bank-sized gap per 64 compact sums. A near-power-of-two
            // input row no longer gives the gather a power-of-two row stride.
            ShiftRight<int32_t>(r, dst, 6, count);
            pipe_barrier(PIPE_V);
            ShiftLeft<int32_t>(r, r, 3, count);
            pipe_barrier(PIPE_V);
            Add(dst, dst, r, count);
            pipe_barrier(PIPE_V);
        }
        ShiftLeft<int32_t>(dst, dst, 2, count);
        pipe_barrier(PIPE_V);

        auto tail = dst[count];
        const uint32_t tailCount = edges*pitch;
        ArithProgression<int32_t>(tail, 0, 1, tailCount);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q, tail, shift, tailCount);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r, q, shift, tailCount);
        pipe_barrier(PIPE_V);
        Sub(tail, tail, r, tailCount);
        pipe_barrier(PIPE_V);
        Mins<int32_t>(tail, tail, static_cast<int32_t>(rowBatch_-1), tailCount);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(tail, tail, static_cast<int32_t>(width), tailCount);
        pipe_barrier(PIPE_V);
        Adds<int32_t>(r, tail, unit-1U, tailCount);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(r, r, unitShift, tailCount);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r, r, unitShift, tailCount);
        pipe_barrier(PIPE_V);
        Sub(r, r, tail, tailCount);
        pipe_barrier(PIPE_V);
        Sub(r, r, q, tailCount);
        pipe_barrier(PIPE_V);
        Adds<int32_t>(r, r, -1, tailCount);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(r, r, 31, tailCount);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(r, r, -static_cast<int32_t>(unit*blocks), tailCount);
        pipe_barrier(PIPE_V);
        Add(tail, tail, r, tailCount);
        pipe_barrier(PIPE_V);
        Add(tail, tail, q, tailCount);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(tail, tail, packed ? 1 : 2, tailCount);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void FoldFlatPlanes(LocalTensor<float> data, uint32_t planes, uint32_t pitch)
    {
        uint32_t power = 1;
        while ((power<<1) <= planes) power <<= 1;
        if (planes > power) {
            Add(data, data, data[power*pitch], (planes-power)*pitch);
            pipe_barrier(PIPE_V);
        }
        for (uint32_t half = power>>1; half; half >>= 1) {
            Add(data, data, data[half*pitch], half*pitch);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ReducePaddedBlocks(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
    {
        const uint32_t groups = count/512U;
        if (groups) {
            for (uint32_t b=0; b<8; ++b) {
                BlockReduceSum<float>(dst[b*8U], src[b*64U], groups, 64, 9, 1, 64);
            }
        }
        if (count%512U) ReduceBlocks(dst[groups*72U], src[groups*512U], count%512U);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReduceBlocks(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
    {
        uint32_t offset = 0;
        while (offset + 64U <= count) {
            uint32_t reps = (count-offset)/64U;
            if (reps > 255U) reps = 255U;
            BlockReduceSum<float>(dst[offset/8U], src[offset], reps, 64, 1, 1, 8);
            offset += reps * 64U;
        }
        if (offset < count) BlockReduceSum<float>(dst[offset/8U], src[offset], 1, count-offset, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReducePairs(LocalTensor<float> data, uint32_t count)
    {
        uint32_t offset = 0;
        while (offset + 64U <= count) {
            uint32_t reps = (count-offset)/64U;
            if (reps > 255U) reps = 255U;
            PairReduceSum<float>(data[offset/2U], data[offset], reps, 64, 1, 1, 8);
            offset += reps * 64U;
        }
        if (offset < count) PairReduceSum<float>(data[offset/2U], data[offset], 1, count-offset, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReduceResidueRows(LocalTensor<float> acc, uint32_t width,
                                              uint32_t rows, uint32_t period,
                                              uint32_t groupStride, uint64_t outOffset)
    {
        auto sums = TmpT();
        auto scratch = sums[period * groupStride];
        const uint32_t srcStride = period * width / 8U;
        for (uint32_t residue = 0; residue < period && residue < rows; ++residue) {
            const uint32_t groupRows = (rows - residue + period - 1) / period;
            const uint32_t start = residue * width;
            const uint32_t head = start & 7U;
            const uint32_t aligned = start & ~7U;
            for (uint32_t q = 0; q < groupRows;) {
                const uint32_t repeat = groupRows-q > 255 ? 248 : groupRows-q;
                auto dst = sums[residue * groupStride + q];
                bool first = true;
                for (uint32_t seg = 0; seg < head + width; seg += 64) {
                    const uint32_t end = head + width - seg < 64 ? head + width - seg : 64;
                    const uint32_t begin = seg == 0 ? head : 0;
                    const uint64_t bits = (end == 64 ? ~0ULL : ((1ULL << end)-1)) & (~0ULL << begin);
                    uint64_t mask[2] = {bits, 0ULL};
                    WholeReduceSum<float>(first ? dst : scratch,
                        acc[aligned + q * period * width + seg], mask, repeat, 1, 1, srcStride);
                    pipe_barrier(PIPE_V);
                    if (!first) {
                        Add(dst, dst, scratch, repeat);
                        pipe_barrier(PIPE_V);
                    }
                    first = false;
                }
                q += repeat;
            }
        }
        if (period == 1) {
            StoreTyped(sums, rows, outOffset);
        } else {
            Gather(acc, sums, WorkT().template ReinterpretCast<uint32_t>(), 0U, rows);
            pipe_barrier(PIPE_V);
            StoreTyped(acc, rows, outOffset);
        }
    }

    __aicore__ inline void BuildTransposeIndex(uint32_t span, uint32_t multiple, uint32_t count)
    {
        auto index = WorkT().template ReinterpretCast<int32_t>();
        auto q = AccT().template ReinterpretCast<int32_t>();
        auto r = TmpT().template ReinterpretCast<int32_t>();
        uint32_t shift = 0;
        uint64_t scale = 1;
        while (scale <= static_cast<uint64_t>(count) * span) { scale <<= 1; ++shift; }
        const int32_t magic = static_cast<int32_t>((scale + span - 1) / span);
        ArithProgression<int32_t>(index, 0, 1, count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(q, index, magic, count);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q, q, static_cast<int32_t>(shift), count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(r, q, static_cast<int32_t>(span), count);
        pipe_barrier(PIPE_V);
        Sub<int32_t>(index, index, r, count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(index, index, static_cast<int32_t>(multiple), count);
        pipe_barrier(PIPE_V);
        Add<int32_t>(index, index, q, count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(index, index, 2, count);
        pipe_barrier(PIPE_V);
    }

    // Polled combine, pass 2. After publishing all local tasks, read each
    // chunk slot until its nonce and checksum validate, sum float32 partials,
    // and use the normal single-cast output path.
    //
    // The nonce is the output GM address, not a unique launch identifier.
    // For reused addresses, a previously valid slot must have an equivalent
    // partial value or be overwritten before the combine accepts it.
    __aicore__ inline void CombinePolled()
    {
        const uint64_t tasks = t_->indepUnits * static_cast<uint64_t>(redChunks_);
        uint64_t lo = 0;
        uint64_t hi = 0;
        SplitRange(tasks, lo, hi);
        LocalTensor<float> acc = AccT();
        LocalTensor<float> work = WorkT();
        LocalTensor<uint32_t> pb = work.template ReinterpretCast<uint32_t>();
        DataCopyPadExtParams<uint32_t> noPad{false, 0, 0, 0U};
        const uint32_t n0 = static_cast<uint32_t>(nonce_);
        const uint32_t n1 = static_cast<uint32_t>(nonce_ >> 32);
        for (uint64_t task = lo; task < hi; ++task) {
            if ((task % redChunks_) != 0) {
                continue;
            }
            const uint64_t unit = task / redChunks_;
            // The unit's slots are packed 16-byte records, so ONE contiguous
            // read collects them all; record j lands at pb[j*4].  The form
            // this replaces polled each record separately behind its own
            // PIPE_ALL drain -- redChunks_ round trips per poll round, paid
            // by every full-reduce split (now planned from 128 KB up).
            float total = 0.0f;
            {
                const uint64_t base = static_cast<uint64_t>(unit) * redChunks_ * 4ULL;
                DataCopyExtParams rp{1, static_cast<uint32_t>(redChunks_) * 16U, 0, 0, 0};
                for (;;) {
                    DataCopyPad(pb, wsU32_[base], rp, noPad);
                    pipe_barrier(PIPE_ALL);
                    bool all = true;
                    for (uint32_t j = 0; j < redChunks_; ++j) {
                        const uint32_t o = j * 4U;
                        if (pb.GetValue(o) != n0 || pb.GetValue(o + 1U) != n1 ||
                            (n0 ^ n1 ^ pb.GetValue(o + 2U) ^ 0x5A5AA5A5u) !=
                                pb.GetValue(o + 3U)) {
                            all = false;
                            break;
                        }
                    }
                    if (!all) {
                        continue;
                    }
                    for (uint32_t j = 0; j < redChunks_; ++j) {
                        union { float f; uint32_t u; } pun;
                        pun.u = pb.GetValue(j * 4U + 2U);
                        total += pun.f;
                    }
                    break;
                }
            }
            // One write, one cast: park the fp32 total in acc[0] and reuse the
            // ordinary store, which handles every output dtype.
            acc.SetValue(0, total);
            pipe_barrier(PIPE_ALL);
            StoreTyped(acc, 1, unit);
        }
    }

    __aicore__ inline void SplitRange(uint64_t total, uint64_t& lo, uint64_t& hi) const
    {
        const uint64_t idx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t nc = (usedCores_ == 0) ? 1 : usedCores_;
        if (idx >= nc) {
            lo = 0;
            hi = 0;
            return;
        }
        const uint64_t per = total / nc;
        const uint64_t rem = total % nc;
        if (idx < rem) {
            lo = idx * (per + 1);
            hi = lo + per + 1;
        } else {
            lo = rem * (per + 1) + (idx - rem) * per;
            hi = lo + per;
        }
    }

    __aicore__ inline uint64_t TileBase(uint64_t t) const
    {
        if (nK_ == 0) {
            return 0;
        }
        if (nK_ == 1) {
            return t * t_->kStride[0];
        }
        uint64_t rem = t;
        uint64_t base = 0;
        for (int32_t j = static_cast<int32_t>(nK_) - 1; j >= 0; --j) {
            base += (rem % t_->kSize[j]) * t_->kStride[j];
            rem /= t_->kSize[j];
        }
        return base;
    }

    __aicore__ inline uint64_t RunInit(uint64_t r, uint64_t* ctr) const
    {
        uint64_t off = 0;
        uint64_t rem = r;
        for (int32_t j = static_cast<int32_t>(nR_) - 1; j >= 0; --j) {
            const uint64_t c = rem % t_->rSize[j];
            ctr[j] = c;
            off += c * t_->rStride[j];
            rem /= t_->rSize[j];
        }
        return off;
    }

    // advance by `step` positions of the innermost outer-reduced group; the
    // caller guarantees the step stays inside that group
    __aicore__ inline void RunAdvance(uint64_t* ctr, uint64_t& off, uint64_t step) const
    {
        if (nR_ == 0 || step == 0) {
            return;
        }
        const int32_t last = static_cast<int32_t>(nR_) - 1;
        ctr[last] += step;
        off += step * t_->rStride[last];
        if (ctr[last] < t_->rSize[last]) {
            return;
        }
        off -= t_->rSize[last] * t_->rStride[last];
        ctr[last] = 0;
        for (int32_t j = last - 1; j >= 0; --j) {
            ctr[j] += 1;
            off += t_->rStride[j];
            if (ctr[j] < t_->rSize[j]) {
                return;
            }
            off -= t_->rSize[j] * t_->rStride[j];
            ctr[j] = 0;
        }
    }

    __aicore__ inline void RunNext(uint64_t* ctr, uint64_t& off) const
    {
        for (int32_t j = static_cast<int32_t>(nR_) - 1; j >= 0; --j) {
            ctr[j] += 1;
            off += t_->rStride[j];
            if (ctr[j] < t_->rSize[j]) {
                return;
            }
            off -= t_->rSize[j] * t_->rStride[j];
            ctr[j] = 0;
        }
    }

    // acc[0..n) += src[off..off+n)^2, accumulating in fp32
    __aicore__ inline void AccumSquare(const LocalTensor<float>& acc, const LocalTensor<T>& src,
                                       uint32_t off, uint32_t n)
    {
#ifdef SS_NO_FUSED_MULADD
        // Materialize float32 squares before adding when fused accumulation is disabled.
        {
            LocalTensor<float> tmp = TmpT();
            if constexpr (std::is_same_v<T, float>) {
                Mul(tmp, src[off], src[off], static_cast<int32_t>(n));
            } else {
                Cast(tmp, src[off], RoundMode::CAST_NONE, static_cast<int32_t>(n));
                Mul(tmp, tmp, tmp, static_cast<int32_t>(n));
            }
            Add(acc, acc, tmp, static_cast<int32_t>(n));
        }
#else
        if constexpr (std::is_same_v<T, float>) {
            MulAddDst<float, float>(acc, src[off], src[off], static_cast<int32_t>(n));
        } else if constexpr (std::is_same_v<T, half>) {
            // one vmla does widen + square + accumulate; no fp32 square is ever
            // materialised and the fp32 UB copy of the tile disappears
            MulAddDst<float, half>(acc, src[off], src[off], static_cast<int32_t>(n));
        } else {
            // bfloat16 has no mixed-precision vmla form, so it pays one Cast
            LocalTensor<float> tmp = TmpT();
            Cast(tmp, src[off], RoundMode::CAST_NONE, static_cast<int32_t>(n));
            MulAddDst<float, float>(acc, tmp, tmp, static_cast<int32_t>(n));
        }
#endif
    }

    __aicore__ inline void CastToOut(const LocalTensor<T>& out, const LocalTensor<float>& src, uint32_t n)
    {
        if constexpr (std::is_same_v<T, float>) {
            (void)out;
            (void)src;
            (void)n;
        } else if constexpr (std::is_same_v<T, half>) {
            Cast(out, src, RoundMode::CAST_NONE, static_cast<int32_t>(n));
        } else {
            Cast(out, src, RoundMode::CAST_RINT, static_cast<int32_t>(n));
        }
    }

    // stream one contiguous source range into acc (chunked by the UB tile size)
    __aicore__ inline void StreamRange(const LocalTensor<float>& acc, uint64_t src, uint64_t count)
    {
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        uint64_t remain = count;
        uint64_t cursor = src;
        while (remain > 0) {
            const uint32_t n = static_cast<uint32_t>((remain < ubChunk_) ? remain : ubChunk_);
            LocalTensor<T> in = inQue_.template AllocTensor<T>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            DataCopyPad(in, xGm_[cursor], cp, noPad);
            inQue_.EnQue(in);
            in = inQue_.template DeQue<T>();
            for (uint32_t k = 0; k < n; k += accLen_) {
                const uint32_t m = ((n - k) < accLen_) ? (n - k) : accLen_;
                AccumSquare(acc, in, k, m);
            }
            inQue_.FreeTensor(in);
            cursor += n;
            remain -= n;
        }
    }

 public:
    // ---------------------------------- batched strided scalar (key 9) ----
    // One multi-block DataCopyPad per BATCH of outer runs instead of one
    // micro-DMA per run.  Layout after the copy is the same contract the
    // VectorTasks batch documents: each run right-padded with zeros to
    // `pitch`, runs at multiples of pitch, so one AccumSquare covers the
    // whole batch and the pads contribute nothing.  Everything is summed in
    // the end, so lanes may mix runs freely: ReduceSum of the accumulator IS
    // the output.  Outputs stage and flush exactly like ScalarTasks.
    __aicore__ inline void ProcessStridedScalar()
    {
        uint64_t lo = 0;
        uint64_t hi = 0;
        SplitRange(outTiles_, lo, hi);
        LocalTensor<float> acc = AccT();
        LocalTensor<float> work = WorkT();
        LocalTensor<float> stage = StageT();
        const uint64_t redPerTile = t_->redPerTile;
        const uint64_t stride = t_->rStride[0];
        const uint32_t n = static_cast<uint32_t>(innerLen_);
        const uint32_t pitch =
            static_cast<uint32_t>(((n * sizeof(T) + 31U) / 32U) * 32U / sizeof(T));
        uint32_t maxB = ubChunk_ / pitch;
        if (maxB == 0U) {
            maxB = 1U;
        }
        if (maxB > 4095U) {
            maxB = 4095U;
        }
        uint32_t grp = 1U;
        while (grp * 2U <= maxB && static_cast<uint64_t>(grp) * 2U <= redPerTile &&
               static_cast<uint64_t>(grp) * 2U * pitch <= accLen_) {
            grp *= 2U;
        }
        const uint32_t accLanes = grp * pitch;
        const uint32_t gapBytes = static_cast<uint32_t>((stride - n) * sizeof(T));
        const uint8_t rightPad = static_cast<uint8_t>(pitch - n);
        DataCopyPadExtParams<T> pad{rightPad != 0U, 0, rightPad, static_cast<T>(0)};

        uint32_t staged = 0;
        uint64_t stageFirst = 0;
        for (uint64_t tsk = lo; tsk < hi; ++tsk) {
            const uint64_t base = TileBase(tsk);
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(accLanes));
            uint64_t r = 0;
            while (r < redPerTile) {
                const uint32_t nb = static_cast<uint32_t>(
                    ((redPerTile - r) < grp) ? (redPerTile - r) : grp);
                LocalTensor<T> in = inQue_.template AllocTensor<T>();
                DataCopyExtParams cp{static_cast<uint16_t>(nb),
                                     static_cast<uint32_t>(n * sizeof(T)), gapBytes, 0, 0};
                DataCopyPad(in, xGm_[base + r * stride], cp, pad);
                inQue_.EnQue(in);
                in = inQue_.template DeQue<T>();
                AccumSquare(acc, in, 0, nb * pitch);
                inQue_.FreeTensor(in);
                r += nb;
            }
            if (staged == 0U) {
                stageFirst = tsk;
            }
            ReduceSum<float>(stage[staged * STAGE_PITCH], acc, work,
                             static_cast<int32_t>(accLanes));
            ++staged;
            if (staged == STAGE_SLOTS) {
                FlushStage(staged, stageFirst);
                staged = 0;
            }
        }
        if (staged > 0U) {
            FlushStage(staged, stageFirst);
        }
    }

    // -------------------------------------- layered sequential (key 10) ----
    // The outer reduced dim is outermost, so layer l = x[l] is one contiguous
    // outTiles*innerLen block.  Each core owns a contiguous run of OUTPUTS
    // [oLo,oHi) -- i.e. the byte range [oLo*innerLen, oHi*innerLen) inside
    // every layer -- and streams that range layer by layer with plain
    // sequential DMA, squaring into a float accumulator of the same layout.
    // The tail folds each innerLen segment to its output: one Gather
    // transposes the slice [nOut x innerLen] -> [innerLen x nOut] (slice
    // <= 8192 elements, the Gather ceiling), then innerLen wide Adds collapse
    // the rows and the nOut results store through the ordinary staged path.
    // The offset table is built once per core, in the work region (8192 u32),
    // and reused by every slice: entries are BYTE offsets, table[k*nOut+o] =
    // (o*innerLen+k)*4.
    __aicore__ inline void ProcessLayered()
    {
        uint64_t oLo = 0;
        uint64_t oHi = 0;
        SplitRange(outTiles_, oLo, oHi);
        if (oHi <= oLo) {
            return;
        }
        const uint64_t redPerTile = t_->redPerTile;
        const uint64_t layer = t_->rStride[0];
        const uint32_t n = static_cast<uint32_t>(innerLen_);
        LocalTensor<float> acc = AccT();
        LocalTensor<float> gDst = TmpT();
        LocalTensor<uint32_t> tab = WorkT().template ReinterpretCast<uint32_t>();

        const bool residueReduction = n <= 256;
        uint32_t period = 1;
        while ((period * n & 7U) != 0) period <<= 1;
        uint32_t reduceSlice = 8192U / n;
        if (reduceSlice > outStage_) reduceSlice = outStage_;
        reduceSlice &= ~7U;
        const uint32_t rowAlignment = 32U/sizeof(T);
        const uint32_t outPerSlice = residueReduction
            ? ((accLen_ / n < outStage_ ? accLen_ / n : outStage_) & ~(rowAlignment-1U))
            : (accLen_ / (n * reduceSlice)) * reduceSlice;
        const uint32_t groupStride = (((outPerSlice + period - 1) / period) + 7U) & ~7U;
        if (residueReduction) {
            if (period > 1) BuildTransposeIndex(period, groupStride, outPerSlice);
        } else {
            auto ti = tab.template ReinterpretCast<int32_t>();
            ArithProgression<int32_t>(ti, 0, n * 4U, reduceSlice);
            pipe_barrier(PIPE_V);
            for (uint32_t k = 1; k < n; ++k) {
                Adds<int32_t>(ti[k * reduceSlice], ti, k * 4U, reduceSlice);
            }
        }

        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};

        uint64_t sliceStep = outPerSlice;
        if (outTiles_ >= static_cast<uint64_t>(outPerSlice) * usedCores_) {
            oLo = static_cast<uint64_t>(GetBlockIdx()) * outPerSlice;
            oHi = outTiles_;
            sliceStep *= usedCores_;
        }
        for (uint64_t o0 = oLo; o0 < oHi; o0 += sliceStep) {
            const uint32_t nOut = static_cast<uint32_t>(
                ((oHi - o0) < outPerSlice) ? (oHi - o0) : outPerSlice);
            const uint32_t sliceElems = nOut * n;
            const uint64_t sliceBase = o0 * n;
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(sliceElems));
            // Load consecutive layers with one multi-block DataCopyPad. Its UB pitch
            // is ceil32(sliceBytes), so block b starts at b*pitchElems, not b*sliceElems.
            // Carry the padded pitch through indexing even for unaligned slice lengths.
            const uint32_t sliceBytes = static_cast<uint32_t>(sliceElems * sizeof(T));
            const uint32_t pitchB = (sliceBytes + 31U) & ~31U;
            const uint32_t pitchElems = pitchB / static_cast<uint32_t>(sizeof(T));
            uint32_t lb = (pitchElems <= ubChunk_) ? (ubChunk_ / pitchElems) : 1U;
            // Limit the layer batch by UB capacity, the 4095-block descriptor bound,
            // and the number of remaining layers.
            const uint32_t lbCap = (redPerTile < 4095ULL)
                                       ? static_cast<uint32_t>(redPerTile)
                                       : 4095U;
            if (lb > lbCap) {
                lb = lbCap;
            }
            if (pitchElems <= ubChunk_) {
                const uint32_t gap =
                    static_cast<uint32_t>((layer - sliceElems) * sizeof(T));
                uint64_t remain = redPerTile;
                uint64_t l = static_cast<uint64_t>(GetBlockIdx()) % redPerTile;
                uint32_t current = static_cast<uint32_t>(redPerTile-l < lb ? redPerTile-l : lb);
                auto first = inQue_.template AllocTensor<T>();
                DataCopyExtParams firstCp{static_cast<uint16_t>(current), sliceBytes, gap, 0, 0};
                if (current == 1 && (sliceBytes & 31U) == 0) DataCopy(first, xGm_[l * layer + sliceBase], sliceElems);
                else DataCopyPad(first, xGm_[l * layer + sliceBase], firstCp, noPad);
                inQue_.EnQue(first);
                l += current; remain -= current;
                if (l == redPerTile) l = 0;
                while (current) {
                    uint32_t next = 0;
                    if (remain) {
                        next = static_cast<uint32_t>(remain < lb ? remain : lb);
                        if (next > redPerTile-l) next = static_cast<uint32_t>(redPerTile-l);
                        auto ahead = inQue_.template AllocTensor<T>();
                        DataCopyExtParams cp{static_cast<uint16_t>(next), sliceBytes, gap, 0, 0};
                        if (next == 1 && (sliceBytes & 31U) == 0) DataCopy(ahead, xGm_[l * layer + sliceBase], sliceElems);
                        else DataCopyPad(ahead, xGm_[l * layer + sliceBase], cp, noPad);
                        inQue_.EnQue(ahead);
                        l += next; remain -= next;
                        if (l == redPerTile) l = 0;
                    }
                    auto in = inQue_.template DeQue<T>();
                    for (uint32_t b = 0; b < current; ++b) {
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            for (uint32_t k = 0; k < sliceElems; k += tmpLen_) {
                                const uint32_t count = (sliceElems-k) < tmpLen_ ? sliceElems-k : tmpLen_;
                                AccumSquare(acc[k], in[b * pitchElems + k], 0, count);
                            }
                        } else {
                            AccumSquare(acc, in[b * pitchElems], 0, sliceElems);
                        }
                    }
                    inQue_.FreeTensor(in);
                    current = next;
                }
            } else {
                for (uint64_t l = 0; l < redPerTile; ++l) {
                    uint64_t done = 0;
                    while (done < sliceElems) {
                        const uint32_t c = static_cast<uint32_t>(
                            ((sliceElems - done) < ubChunk_) ? (sliceElems - done) : ubChunk_);
                        LocalTensor<T> in = inQue_.template AllocTensor<T>();
                        DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(T)), 0, 0, 0};
                        DataCopyPad(in, xGm_[l * layer + sliceBase + done], cp, noPad);
                        inQue_.EnQue(in);
                        in = inQue_.template DeQue<T>();
                        AccumSquare(acc[static_cast<uint32_t>(done)], in, 0, c);
                        inQue_.FreeTensor(in);
                        done += c;
                    }
                }
            }
            if (residueReduction) {
                ReduceResidueRows(acc, n, nOut, period, groupStride, o0);
                continue;
            }
            // Tail: transpose the slice, collapse the innerLen rows, and the
            // first nOut floats of gDst ARE the outputs, contiguous -- store
            // them straight through the single-cast path.  No staging, no
            // scalar copies, no extra events.
            for (uint32_t sub = 0; sub < nOut; sub += reduceSlice) {
                Gather(gDst, acc, tab, sub * n * 4U, n * reduceSlice);
                pipe_barrier(PIPE_V);
                uint32_t power = 1;
                while ((power << 1) <= n) power <<= 1;
                if (n > power) {
                    Add(gDst, gDst, gDst[power * reduceSlice], (n - power) * reduceSlice);
                    pipe_barrier(PIPE_V);
                }
                for (uint32_t halfRows = power >> 1; halfRows; halfRows >>= 1) {
                    Add(gDst, gDst, gDst[halfRows * reduceSlice], halfRows * reduceSlice);
                    pipe_barrier(PIPE_V);
                }
                const uint32_t count = (nOut-sub) < reduceSlice ? nOut-sub : reduceSlice;
                StoreTyped(gDst, count, o0 + sub);
            }
        }
    }

    // ---------------------------- K R K layered polled (key 12) ------------
    // The outermost group is REDUCED, so layer l = everything below it is one
    // contiguous slotW * innerRuns block (slotW = outTiles*innerLen).  Chunks
    // split the layer range; each chunk streams its layers sequentially, and
    // inside a layer each kept-outer's sub-block (innerRuns runs of innerLen,
    // contiguous) folds through the run-group ladder into the chunk's slot
    // accumulator at [k1*innerLen, +innerLen).  One slot publish per chunk,
    // the ordinary polled combine, one cast at the store.  The accumulator
    // splits: lanes region at acc[0..lanesCap), slot region at
    // acc[lanesCap..lanesCap+slotW) -- the plan gate keeps slotW <= 4096 and
    // accLen is 8192.
    __aicore__ inline void ProcessKRKPolled()
    {
        const uint32_t rcTotal = redChunks_;
        const uint32_t rc = static_cast<uint32_t>(GetBlockIdx());
        if (rc >= rcTotal) {
            return;
        }
        const uint64_t layers = t_->rSize[0];
        const uint64_t layerStride = t_->rStride[0];
        const uint64_t outT = t_->outTiles;
        const uint32_t n = static_cast<uint32_t>(innerLen_);
        // Row pitch inside the slot: 8-float aligned so every kept-outer's
        // row starts on a 32-byte boundary (the k1*29 fault of the first
        // build).  Publish/combine carry the padded width; the store walks
        // rows and writes only the n live floats of each.
        const uint32_t slotP = (n + 7U) & ~7U;
        const uint32_t slotW = static_cast<uint32_t>(outT * slotP);
        const uint64_t innerRuns = t_->redPerTile / (layers ? layers : 1ULL);
        const uint64_t subBlock = static_cast<uint64_t>(n) * innerRuns;  // 连续
        const uint32_t lanes = t_->runLanes;
        const uint32_t fold = t_->runFold;
        const uint32_t lanesCap = 4096U;
        LocalTensor<float> acc = AccT();                 // [0..lanesCap) lanes
        LocalTensor<float> slot = acc[lanesCap];         // [.. +slotW)
        LocalTensor<float> work = WorkT();
        const uint32_t slab = (lanes != 0U) ? (ubChunk_ / lanes) * lanes : ubChunk_;
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};

        // Distribute (layer, kept-outer) pairs as work units to balance the
        // remaining work more finely than a layer-only split.
        const uint64_t units = layers * outT;
        const uint64_t per = (units + rcTotal - 1U) / rcTotal;
        const uint64_t uLo = static_cast<uint64_t>(rc) * per;
        const uint64_t uHi = ((uLo + per) < units) ? (uLo + per) : units;
        Duplicate<float>(slot, 0.0f, static_cast<int32_t>(slotW));
        for (uint64_t u = uLo; u < uHi; ++u) {
            const uint64_t l = u / outT;
            const uint64_t k1 = u % outT;
            const uint64_t layerBase = l * layerStride;
            {
                // 子块: innerRuns×n 连续
                Duplicate<float>(acc, 0.0f, static_cast<int32_t>(lanes));
                uint64_t remain = subBlock;
                uint64_t cursor = layerBase + k1 * subBlock;
                while (remain > 0) {
                    const uint32_t c =
                        static_cast<uint32_t>((remain < slab) ? remain : slab);
                    LocalTensor<T> in = inQue_.template AllocTensor<T>();
                    DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(T)), 0, 0, 0};
                    DataCopyPad(in, xGm_[cursor], cp, noPad);
                    inQue_.EnQue(in);
                    in = inQue_.template DeQue<T>();
                    for (uint32_t k = 0; k < c; k += lanes) {
                        const uint32_t m = ((c - k) < lanes) ? (c - k) : lanes;
                        AccumSquare(acc, in, k, m);
                    }
                    inQue_.FreeTensor(in);
                    cursor += c;
                    remain -= c;
                }
                FoldRunGroups(acc, n, lanes, fold);
                Add(slot[static_cast<uint32_t>(k1) * slotP],
                    slot[static_cast<uint32_t>(k1) * slotP], acc,
                    static_cast<int32_t>(n));
            }
        }
        PublishVecSlot(slot, slotW, 0ULL, rc, slotW);
        SyncAll<true>();
        // 合并: 只有 rc==0 折叠并存 slotW 宽
        if (rc == 0U) {
            const uint64_t pitch = VecSlotPitch(slotW);
            LocalTensor<uint32_t> hb = work.template ReinterpretCast<uint32_t>();
            DataCopyPadExtParams<uint32_t> noPadU{false, 0, 0, 0U};
            DataCopyPadExtParams<float> noPadF{false, 0, 0, 0.0f};
            const uint32_t n0 = static_cast<uint32_t>(nonce_);
            const uint32_t n1 = static_cast<uint32_t>(nonce_ >> 32);
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(slotW));
            for (uint32_t j = 0; j < rcTotal; ++j) {
                const uint64_t slotB = static_cast<uint64_t>(j) * pitch;
                LocalTensor<float> pay = inQue_.template AllocTensor<float>();
                DataCopyExtParams pl{1, static_cast<uint32_t>(slotW * sizeof(float)), 0, 0, 0};
                DataCopyPad(pay, wsGm_[slotB / 4ULL + 4ULL], pl, noPadF);
                inQue_.EnQue(pay);
                pay = inQue_.template DeQue<float>();
                Add(acc, acc, pay, static_cast<int32_t>(slotW));
                inQue_.FreeTensor(pay);
            }
            for (uint64_t k1 = 0; k1 < outT; ++k1) {
                StoreTyped(acc[static_cast<uint32_t>(k1) * slotP], n, k1 * n);
            }
        }
    }

 private:
    // ------------------------------------------------- trailing (row) reduce
    __aicore__ inline void ScalarTasks(uint64_t lo, uint64_t hi)
    {
        LocalTensor<float> acc = AccT();
        LocalTensor<float> work = WorkT();
        LocalTensor<float> stage = StageT();
        uint32_t staged = 0;
        uint64_t stageFirst = 0;
        const uint64_t redPerTile = t_->redPerTile;
        const uint64_t outAligned = t_->outAligned;
        const uint32_t splitRuns = t_->splitRuns;

        for (uint64_t task = lo; task < hi; ++task) {
            const uint64_t unit = (redChunks_ > 1) ? (task / redChunks_) : task;
            const uint32_t rc = (redChunks_ > 1) ? static_cast<uint32_t>(task % redChunks_) : 0U;
            const uint64_t base = TileBase(unit);

            uint64_t runLo = 0;
            uint64_t runHi = redPerTile;
            uint64_t inLo = 0;
            uint64_t inHi = innerLen_;
            if (redChunks_ > 1) {
                if (splitRuns != 0) {
                    const uint64_t per = CeilDivU(redPerTile, redChunks_);
                    runLo = rc * per;
                    runHi = (runLo + per < redPerTile) ? (runLo + per) : redPerTile;
                    if (runLo > redPerTile) {
                        runLo = redPerTile;
                    }
                } else {
                    uint64_t per = CeilDivU(innerLen_, redChunks_);
                    per = ((per + 255) / 256) * 256;
                    inLo = rc * per;
                    inHi = (inLo + per < innerLen_) ? (inLo + per) : innerLen_;
                    if (inLo > innerLen_) {
                        inLo = innerLen_;
                    }
                }
            }

            const uint64_t span = (inHi > inLo) ? (inHi - inLo) : 0ULL;
            uint32_t used = (span < ubChunk_) ? static_cast<uint32_t>(span) : ubChunk_;
            if (used > accLen_) {
                used = accLen_;
            }
            if (used == 0) {
                used = 1;
            }
            const uint32_t effLen = used;
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(effLen));
            if (runHi > runLo && inHi > inLo) {
                uint64_t ctr[MAX_GROUP] = {0, 0, 0, 0};
                uint64_t runOff = RunInit(runLo, ctr);
                for (uint64_t r = runLo; r < runHi; ++r) {
                    StreamRange(acc, base + runOff + inLo, inHi - inLo);
                    RunNext(ctr, runOff);
                }
            }

            if (redChunks_ > 1) {
                if constexpr (kAtomicRed) {
                    // Polled combine, pass 1. Every (unit, chunk), including chunk zero,
                    // publishes a 16-byte slot: nonce low/high words, value bits, and checksum.
                    // The nonce is the output tensor GM address; address reuse alone does not
                    // distinguish launches. See CombinePolled for the reuse condition.
                    // Publish every local task before polling, since polling inside this loop
                    // could wait for a future task assigned to the same core.
                    ReduceSum<float>(stage, acc, work, static_cast<int32_t>(effLen));
                    pipe_barrier(PIPE_ALL);
                    union { float f; uint32_t u; } pun;
                    pun.f = stage.GetValue(0);
                    const uint32_t n0 = static_cast<uint32_t>(nonce_);
                    const uint32_t n1 = static_cast<uint32_t>(nonce_ >> 32);
                    LocalTensor<T> out = outQue_.template AllocTensor<T>();
                    LocalTensor<uint32_t> w = out.template ReinterpretCast<uint32_t>();
                    w.SetValue(0, n0);
                    w.SetValue(1, n1);
                    w.SetValue(2, pun.u);
                    w.SetValue(3, n0 ^ n1 ^ pun.u ^ 0x5A5AA5A5u);
                    pipe_barrier(PIPE_ALL);
                    const uint64_t slot =
                        (unit * static_cast<uint64_t>(redChunks_) + rc) * 4ULL;
                    DataCopyExtParams sp{1, 16U, 0, 0, 0};
                    DataCopyPad(wsU32_[slot], w, sp);
                    pipe_barrier(PIPE_ALL);
                    outQue_.FreeTensor(out);
                    continue;
                }
                LocalTensor<T> out = outQue_.template AllocTensor<T>();
                LocalTensor<float> dst = out.template ReinterpretCast<float>();
                ReduceSum<float>(dst, acc, work, static_cast<int32_t>(effLen));
                outQue_.EnQue(out);
                out = outQue_.template DeQue<T>();
                LocalTensor<float> src = out.template ReinterpretCast<float>();
                DataCopyExtParams wp{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(wsGm_[static_cast<uint64_t>(rc) * outAligned + unit], src, wp);
                outQue_.FreeTensor(out);
            } else {
                if (staged == 0) {
                    stageFirst = unit;
                }
                ReduceSum<float>(stage[staged * STAGE_PITCH], acc, work,
                                 static_cast<int32_t>(effLen));
                ++staged;
                if (staged == STAGE_SLOTS) {
                    FlushStage(staged, stageFirst);
                    staged = 0;
                }
            }
        }
        if (staged > 0) {
            FlushStage(staged, stageFirst);
        }
    }

    __aicore__ inline void FlushStage(uint32_t n, uint64_t outOff)
    {
        LocalTensor<float> stage = StageT();
        LocalTensor<T> out = outQue_.template AllocTensor<T>();
        LocalTensor<float> dst = (std::is_same_v<T, float>) ? out.template ReinterpretCast<float>()
                                                            : TmpT();
        // gather the 32 B spaced scalars into a packed vector (mask = 1 -> identity)
        WholeReduceSum<float>(dst, stage, 1, static_cast<int32_t>(n), 1, 1, 1);
        CastToOut(out, dst, n);
        outQue_.EnQue(out);
        out = outQue_.template DeQue<T>();
        DataCopyExtParams wp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[outOff], out, wp);
        outQue_.FreeTensor(out);
    }

    // --------------------------------- strided reduce over a contiguous span
    //
    // Taken when the planner has established that the sole outer-reduced group
    // steps by exactly one run, so every source run of a task is adjacent to
    // the next and the whole task is ONE contiguous byte range.
    //
    // The batched route below spends one DMA descriptor AND one MulAddDst per
    // source run.  On [2048,32] axis 0 that is 2048 vector issues for 64 KB,
    // and the simulator charges each a VMLA, four MOVEMASKs and a VECTOR BAR:
    // 227,807 cycles of vector work and 949,933 of barrier, against 12,637 of
    // MTE2.  Per-run issue, not bytes, is the cost.
    //
    // Here the span is read one descriptor per UB tile and accumulated in
    // slabs of `runLanes = runGroup * innerLen_` lanes, so ONE MulAddDst
    // absorbs runGroup runs.  Lane `m*innerLen_ + j` then holds column j
    // summed over every run congruent to m modulo runGroup, and the partials
    // are collapsed once per output tile.
    __aicore__ inline void VectorTasksContig(uint64_t lo, uint64_t hi)
    {
        LocalTensor<float> acc = AccT();
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        const uint32_t n = static_cast<uint32_t>(innerLen_);
        const uint32_t lanes = t_->runLanes;
        const uint32_t fold = t_->runFold;
        const uint64_t redPerTile = t_->redPerTile;
        const uint64_t outAligned = t_->outAligned;
        const uint32_t slab = (ubChunk_ / lanes) * lanes;
        const uint32_t mathChunk=std::is_same_v<T,bfloat16_t> && tmpLen_<lanes ? tmpLen_ : lanes;
        const bool denseFold = sizeof(T) == 2 && fold >= 8 && n * fold <= t_->workLen && n * fold <= 8192;
        if (denseFold) BuildTransposeIndex(fold, n, n * fold);

        for (uint64_t task = lo; task < hi; ++task) {
            const uint64_t t = (redChunks_ > 1) ? (task / redChunks_) : task;
            const uint32_t rc = (redChunks_ > 1) ? static_cast<uint32_t>(task % redChunks_) : 0U;
            const uint64_t base = TileBase(t);

            // Interleave complete slabs among reduction workers. Every slab
            // begins at the same column phase and adjacent workers read adjacent GM.
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(lanes));
            const uint64_t total = redPerTile * innerLen_;
            const uint64_t step = static_cast<uint64_t>(slab) * redChunks_;
            uint64_t pos = static_cast<uint64_t>(rc) * slab;
            uint32_t current = 0;
            if (pos < total) {
                current = static_cast<uint32_t>((total - pos) < slab ? (total - pos) : slab);
                auto first = inQue_.template AllocTensor<T>();
                DataCopyExtParams cp{1, static_cast<uint32_t>(current * sizeof(T)), 0, 0, 0};
                if ((current * sizeof(T) & 31U) == 0) DataCopy(first, xGm_[base + pos], current);
                else DataCopyPad(first, xGm_[base + pos], cp, noPad);
                inQue_.EnQue(first);
                pos += step;
            }
            if constexpr (std::is_same_v<T,bfloat16_t>) {
                SetMaskCount();
                SetVectorMask<float,MaskMode::COUNTER>(0,mathChunk);
            }
            while (current) {
                uint32_t next = 0;
                if (pos < total) {
                    next = static_cast<uint32_t>((total - pos) < slab ? (total - pos) : slab);
                    auto ahead = inQue_.template AllocTensor<T>();
                    DataCopyExtParams cp{1, static_cast<uint32_t>(next * sizeof(T)), 0, 0, 0};
                    if ((next * sizeof(T) & 31U) == 0) DataCopy(ahead, xGm_[base + pos], next);
                    else DataCopyPad(ahead, xGm_[base + pos], cp, noPad);
                    inQue_.EnQue(ahead);
                    pos += step;
                }
                auto in = inQue_.template DeQue<T>();
                uint32_t accOffset=0;
                for (uint32_t k = 0; k < current; k += mathChunk) {
                    const uint32_t m = ((current - k) < mathChunk) ? (current - k) : mathChunk;
                    if constexpr (std::is_same_v<T,bfloat16_t>) {
                        if (m!=mathChunk) SetVectorMask<float,MaskMode::COUNTER>(0,m);
                        auto tmp=TmpT();
                        const UnaryRepeatParams cvt{1,1,8,4};
                        const BinaryRepeatParams fma{1,1,1,8,8,8};
                        Cast<float,bfloat16_t,false>(tmp,in[k],RoundMode::CAST_NONE,uint64_t{0},1,cvt);
                        pipe_barrier(PIPE_V);
                        MulAddDst<float,float,false>(acc[accOffset],tmp,tmp,uint64_t{0},1,fma);
                        pipe_barrier(PIPE_V);
                    } else AccumSquare(acc[accOffset], in, k, m);
                    accOffset+=mathChunk;
                    if (accOffset==lanes) accOffset=0;
                }
                inQue_.FreeTensor(in);
                current = next;
            }

            if constexpr (std::is_same_v<T,bfloat16_t>) {
                SetMaskNorm();
                ResetMask();
            }
            FoldRunGroups(acc, n, lanes, fold, denseFold);
            const uint64_t outOff = t * innerLen_;
            if (redChunks_ > 1) {
                if constexpr (kAtomicRed) {
                    // VECTOR POLLED COMBINE, pass 1.  Same contract as the
                    // scalar form at ScalarTasks, widened to an n-float
                    // payload: payload first, 16-byte validated header last.
                    // Both ride the same MTE3 queue, which completes in
                    // order, so a reader that sees a valid header sees the
                    // payload.  All chunks publish -- chunk 0 included,
                    // because a core that owns two chunks of one unit reuses
                    // acc between them, so nothing survives to pass 2 except
                    // the workspace.
                    PublishVecSlot(acc, n, t, rc, n);
                    continue;
                }
                StoreFloat(acc, n, static_cast<uint64_t>(rc) * outAligned + outOff);
            } else {
                StoreTyped(acc, n, outOff);
            }
        }
    }

    // Slot pitch for the vector polled combine: 16-byte header + n floats,
    // rounded to a 32-byte GM block.
    __aicore__ inline uint64_t VecSlotPitch(uint32_t n) const
    {
        return ((16ULL + static_cast<uint64_t>(n) * 4ULL) + 31ULL) & ~31ULL;
    }

    // slotN fixes the slot PITCH; n is the payload width.  They differ only
    // on a ragged final column piece, where every unit's slot must still sit
    // at the uniform colPiece pitch the combiner walks.
    __aicore__ inline void PublishVecSlot(const LocalTensor<float>& acc, uint32_t n,
                                          uint64_t unit, uint32_t rc, uint32_t slotN)
    {
        const uint64_t pitch = VecSlotPitch(slotN);
        const uint64_t slotB = (unit * static_cast<uint64_t>(redChunks_) + rc) * pitch;
        // Payload: the n-float partial, straight from acc (PIPE_V wrote it;
        // the queue pair orders V -> MTE3).
        LocalTensor<T> out = outQue_.template AllocTensor<T>();
        LocalTensor<float> pay = out.template ReinterpretCast<float>();
        Adds<float>(pay, acc, 0.0f, static_cast<int32_t>(n));
        outQue_.EnQue(out);
        out = outQue_.template DeQue<T>();
        DataCopyExtParams pp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsU32_[(slotB + 16ULL) / 4ULL], out.template ReinterpretCast<uint32_t>(), pp);
        outQue_.FreeTensor(out);
    }

    // VECTOR POLLED COMBINE, pass 2: for every chunk-0 task this core owns,
    // wait for the unit's redChunks_ headers, vector-add the payloads in
    // fp32, and store the unit's n outputs through the single-cast path.
    __aicore__ inline void CombinePolledVec(uint32_t n)
    {
        uint64_t lo, hi;
        SplitRange(t_->indepUnits * static_cast<uint64_t>(redChunks_), lo, hi);
        const uint64_t colChunks = t_->colChunks;
        const uint64_t colPiece = t_->colPiece;
        const uint32_t slotN = static_cast<uint32_t>(colChunks > 1 ? colPiece : n);
        const uint64_t pitch = VecSlotPitch(slotN);
        const uint32_t cap = (ubChunk_ * sizeof(T) / sizeof(float)) & ~7U;
        auto acc = AccT();
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        for (uint64_t task = lo; task < hi; ++task) {
            if (task % redChunks_) continue;
            const uint64_t unit = task / redChunks_;
            const uint64_t tile = unit / colChunks;
            const uint64_t col = (unit % colChunks) * colPiece;
            const uint32_t w = static_cast<uint32_t>(
                (innerLen_ - col) < slotN ? (innerLen_ - col) : slotN);
            Duplicate(acc, 0.0f, w);
            for (uint32_t off = 0; off < w; off += cap) {
                const uint32_t width = (w - off) < cap ? (w - off) : cap;
                const uint32_t row = (width + 7U) & ~7U;
                const uint32_t batch = cap / row;
                for (uint32_t j = 0; j < redChunks_; j += batch) {
                    const uint32_t nb = (redChunks_ - j) < batch ? (redChunks_ - j) : batch;
                    auto pay = inQue_.template AllocTensor<float>();
                    DataCopyExtParams cp{static_cast<uint16_t>(nb), width * 4U,
                        static_cast<uint32_t>(pitch - width * 4ULL), 0, 0};
                    DataCopyPad(pay, wsGm_[(unit * redChunks_ + j) * pitch / 4ULL + 4ULL + off], cp, noPad);
                    inQue_.EnQue(pay);
                    pay = inQue_.template DeQue<float>();
                    uint32_t power = 1;
                    while ((power << 1) <= nb) power <<= 1;
                    if (nb > power) {
                        Add(pay, pay, pay[power * row], (nb - power) * row);
                        pipe_barrier(PIPE_V);
                    }
                    for (uint32_t half = power >> 1; half; half >>= 1) {
                        Add(pay, pay, pay[half * row], half * row);
                        pipe_barrier(PIPE_V);
                    }
                    Add(acc[off], acc[off], pay, width);
                    inQue_.FreeTensor(pay);
                }
            }
            StoreTyped(acc, w, tile * innerLen_ + col);
        }
    }

    // acc[0..runGroup*n) holds runGroup partial output vectors, partial m at
    // float offset m*n.  Collapse them onto acc[0..n).
    //
    // Halving (acc[0..w) += acc[w..2w)) pairs partial m with m + count/2 and is
    // a plain vector Add while w is a whole number of 32 B blocks.  That holds
    // down to runFold*n, the smallest multiple of the run length that is
    // aligned; below it the partials sit at offsets the vector unit cannot
    // address.  runFold == 1 when a run is itself a whole number of blocks,
    // and then the fold is entirely vector.
    //
    // The residual is the only place in this kernel where a pipe other than
    // PIPE_V touches a TBuf, so it is fenced on both sides:
    //   PIPE_V (Add) -> PIPE_S (GetValue/SetValue) -> PIPE_V (StoreTyped)
    // It costs real money -- ~8,500 cycles of scalar and ~46,500 of barrier per
    // output tile on [50,997,5] -- and is still the better trade, because the
    // padded route it replaces costs ~290,000 per tile against ~61,000 here.
    // The obvious next step is to fence with a targeted V_S/S_V event pair
    // instead of PipeBarrier<PIPE_ALL>, since it is the barrier and not the
    // scalar work that dominates; that is new hazard surface and wants its own
    // verification pass.
    __aicore__ inline void FoldRunGroups(LocalTensor<float> acc, uint32_t n, uint32_t runLanes,
                                         uint32_t runFold, bool dense = false)
    {
        uint32_t width = runLanes;
        const uint32_t stopAt = runFold * n;
        while (width > stopAt) {
            width >>= 1;
            Add(acc, acc, acc[width], static_cast<int32_t>(width));
        }
        if (runFold <= 1) {
            return;
        }
        LocalTensor<float> tmp = TmpT();
        LocalTensor<uint32_t> offsets = WorkT().template ReinterpretCast<uint32_t>();
        if (dense) {
            pipe_barrier(PIPE_V);
            Gather(tmp, acc, offsets, 0U, stopAt);
            pipe_barrier(PIPE_V);
            ReduceBlocks(acc, tmp, stopAt);
            uint32_t groups = runFold / 8U;
            uint32_t count = n * groups;
            while (groups > 1U) { ReducePairs(acc, count); count >>= 1; groups >>= 1; }
            return;
        }
        const uint32_t block = 128;
        ArithProgression<int32_t>(offsets.template ReinterpretCast<int32_t>(), 0, 4, block);
        pipe_barrier(PIPE_V);
        for (uint32_t j = 0; j < n; j += block) {
            const uint32_t count = ((n - j) < block) ? (n - j) : block;
            for (uint32_t m = 1; m < runFold; ++m) {
                Gather(tmp, acc, offsets, (m * n + j) * 4U, count);
                pipe_barrier(PIPE_V);
                Add(acc[j], acc[j], tmp, count);
                pipe_barrier(PIPE_V);
            }
        }
    }

    // ------------------------------------------------ strided (column) reduce
    __aicore__ inline void VectorTasks(uint64_t lo, uint64_t hi)
    {
        LocalTensor<float> acc = AccT();
        const uint64_t colPiece = t_->colPiece;
        const uint64_t colChunks = t_->colChunks;
        const uint64_t redPerTile = t_->redPerTile;
        const uint64_t outAligned = t_->outAligned;

        for (uint64_t task = lo; task < hi; ++task) {
            const uint64_t unit = (redChunks_ > 1) ? (task / redChunks_) : task;
            const uint32_t rc = (redChunks_ > 1) ? static_cast<uint32_t>(task % redChunks_) : 0U;
            const uint64_t t = (colChunks > 1) ? (unit / colChunks) : unit;
            const uint64_t cc = (colChunks > 1) ? (unit % colChunks) : 0ULL;
            const uint64_t base = TileBase(t);
            const uint64_t colLo = cc * colPiece;
            uint64_t colHi = colLo + colPiece;
            if (colHi > innerLen_) {
                colHi = innerLen_;
            }

            uint64_t runLo = 0;
            uint64_t runHi = redPerTile;
            if (redChunks_ > 1) {
                const uint64_t per = CeilDivU(redPerTile, redChunks_);
                runLo = rc * per;
                runHi = (runLo + per < redPerTile) ? (runLo + per) : redPerTile;
                if (runLo > redPerTile) {
                    runLo = redPerTile;
                }
            }

            for (uint64_t b = colLo; b < colHi; b += ubChunk_) {
                const uint32_t n =
                    static_cast<uint32_t>(((colHi - b) < ubChunk_) ? (colHi - b) : ubChunk_);
                uint64_t ctr[MAX_GROUP] = {0, 0, 0, 0};
                uint64_t runOff = RunInit(runLo, ctr);

                // Source runs that differ only in the innermost outer-reduced
                // group sit at a constant stride, so many of them ride in one
                // multi-block DataCopyPad instead of one tiny DMA each.  When
                // the innermost kept run is short (the strided-reduction case)
                // this is the difference between 142-byte and 32 KB bursts.
                const uint32_t pitch =
                    static_cast<uint32_t>(((n * sizeof(T) + 31) / 32) * 32 / sizeof(T));
                uint32_t maxBatch = ubChunk_ / pitch;
                if (maxBatch == 0) {
                    maxBatch = 1;
                }
                if (maxBatch > 4095) {
                    maxBatch = 4095;
                }
                const uint32_t gapBytes =
                    (nR_ == 0) ? 0U
                               : static_cast<uint32_t>((t_->rStride[nR_ - 1] - n) * sizeof(T));

                // ...and one MulAddDst per BATCH rather than per run.  Issuing
                // one per run is what actually costs: on [2048,32] axis 0 the
                // simulator charged 2048 of them a VMLA, four MOVEMASKs and a
                // VECTOR BAR each - 227,806 cycles of vector work and 957,816
                // of barrier, against 12,650 of MTE2.
                //
                // So the accumulator keeps one partial per slot of the padded
                // tile instead of one total, and the whole tile is squared and
                // accumulated in a single call.  The slots sit at multiples of
                // `pitch`, which is a whole number of 32 B blocks by
                // construction, so folding them at the end is log2(grp) plain
                // vector Adds - no scalar, and no pipe but PIPE_V involved.
                // The DMA right-pads each run with zeros so the lanes between
                // n and pitch contribute nothing.
                uint32_t grp = 1;
                while (grp * 2 <= maxBatch &&
                       static_cast<uint64_t>(grp) * 2 <= (runHi - runLo)) {
                    grp *= 2;
                }
                const uint32_t accLanes = grp * pitch;
                const uint8_t rightPad = static_cast<uint8_t>(pitch - n);
                DataCopyPadExtParams<T> pad{rightPad != 0, 0, rightPad, static_cast<T>(0)};
                Duplicate<float>(acc, 0.0f, static_cast<int32_t>(accLanes));

                uint64_t r = runLo;
                while (r < runHi) {
                    uint64_t batch = runHi - r;
                    if (nR_ > 0) {
                        const uint64_t leftInGroup = t_->rSize[nR_ - 1] - ctr[nR_ - 1];
                        if (leftInGroup < batch) {
                            batch = leftInGroup;
                        }
                    } else {
                        batch = 1;
                    }
                    if (batch > grp) {
                        batch = grp;
                    }
                    const uint32_t nb = static_cast<uint32_t>(batch);

                    LocalTensor<T> in = inQue_.template AllocTensor<T>();
                    DataCopyExtParams cp{static_cast<uint16_t>(nb),
                                         static_cast<uint32_t>(n * sizeof(T)), gapBytes, 0, 0};
                    DataCopyPad(in, xGm_[base + runOff + b], cp, pad);
                    inQue_.EnQue(in);
                    in = inQue_.template DeQue<T>();
                    AccumSquare(acc, in, 0, nb * pitch);
                    inQue_.FreeTensor(in);
                    RunAdvance(ctr, runOff, batch);
                    r += batch;
                }
                // collapse the per-slot partials onto acc[0..pitch)
                uint32_t w = accLanes;
                while (w > pitch) {
                    w >>= 1;
                    Add(acc, acc, acc[w], static_cast<int32_t>(w));
                }
                const uint64_t outOff = t * innerLen_ + b;
                if (redChunks_ > 1) {
                    if constexpr (kAtomicRed) {
                        // KEY_PAD_POLLED: the retag gate guarantees one
                        // segment per unit (colPiece<=ubChunk), so b==colLo
                        // and this is the unit's whole partial.  The slot
                        // OWNER is the (tile, column-piece) unit and the
                        // pitch is the uniform colPiece width -- a ragged
                        // tail publishes fewer floats at the same pitch.
                        const uint32_t slotN = static_cast<uint32_t>(
                            (colChunks > 1) ? colPiece : n);
                        PublishVecSlot(acc, n, unit, rc, slotN);
                    } else {
                        StoreFloat(acc, n,
                                   static_cast<uint64_t>(rc) * outAligned + outOff);
                    }
                } else {
                    StoreTyped(acc, n, outOff);
                }
            }
        }
    }

    __aicore__ inline void StoreFloat(const LocalTensor<float>& acc, uint32_t n, uint64_t off)
    {
        LocalTensor<T> out = outQue_.template AllocTensor<T>();
        LocalTensor<float> dst = out.template ReinterpretCast<float>();
        Adds<float>(dst, acc, 0.0f, static_cast<int32_t>(n));
        outQue_.EnQue(out);
        out = outQue_.template DeQue<T>();
        LocalTensor<float> src = out.template ReinterpretCast<float>();
        DataCopyExtParams wp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[off], src, wp);
        outQue_.FreeTensor(out);
    }

    __aicore__ inline void StoreTyped(const LocalTensor<float>& acc, uint32_t n, uint64_t off)
    {
        LocalTensor<T> out = outQue_.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<float> dst = out.template ReinterpretCast<float>();
            Adds<float>(dst, acc, 0.0f, static_cast<int32_t>(n));
        } else {
            CastToOut(out, acc, n);
        }
        outQue_.EnQue(out);
        out = outQue_.template DeQue<T>();
        DataCopyExtParams wp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[off], out, wp);
        outQue_.FreeTensor(out);
    }

    // ------------------------------------------------------- 2nd stage merge
    __aicore__ inline void Combine()
    {
        uint64_t lo = 0;
        uint64_t hi = 0;
        SplitRange(t_->totalOut, lo, hi);
        if (hi <= lo) {
            return;
        }
        const uint32_t queFloats = static_cast<uint32_t>(ubChunk_ * sizeof(T) / sizeof(float));
        uint32_t step = (accLen_ < queFloats) ? accLen_ : queFloats;
        if (outStage_ < step) {
            step = outStage_;
        }
        if (step == 0) {
            step = 1;
        }
        LocalTensor<float> acc = AccT();
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        const uint64_t outAligned = t_->outAligned;

        for (uint64_t q = lo; q < hi; q += step) {
            const uint32_t n = static_cast<uint32_t>(((hi - q) < step) ? (hi - q) : step);
            Duplicate<float>(acc, 0.0f, static_cast<int32_t>(n));
            for (uint32_t p = 0; p < redChunks_; ++p) {
                LocalTensor<float> in = inQue_.template AllocTensor<float>();
                DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
                DataCopyPad(in, wsGm_[static_cast<uint64_t>(p) * outAligned + q], cp, noPad);
                inQue_.EnQue(in);
                in = inQue_.template DeQue<float>();
                Add(acc, acc, in, static_cast<int32_t>(n));
                inQue_.FreeTensor(in);
            }
            StoreTyped(acc, n, q);
        }
    }

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> wsGm_;
    // kAtomicRed only: the same user workspace as 16-byte validated slots, and
    // the per-launch nonce (the output tensor's GM address).
    GlobalTensor<uint32_t> wsU32_;
    uint64_t nonce_ = 0;
    TQue<QuePosition::VECIN, 2> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> ubBuf_;

    const __gm__ SquareSumV1TilingData* t_ = nullptr;
    uint64_t innerLen_ = 1;
    uint64_t outTiles_ = 1;
    uint32_t nK_ = 0;
    uint32_t nR_ = 0;
    uint32_t usedCores_ = 1;
    uint32_t redChunks_ = 1;
    uint32_t rowBatch_ = 1;
    uint32_t rowPitch_ = 1;
    uint32_t accLen_ = 1;
    uint32_t ubChunk_ = 1;
    uint32_t workLen_ = 1;
    uint32_t outStage_ = 1;
    uint32_t tmpLen_ = 1;
    uint32_t tmpAt_ = 0;    // float index of the tmp region inside ubBuf_
    uint32_t workAt_ = 0;
    uint32_t stageAt_ = 0;
};

// KEY_ROWS_ONE is a separate straight-line row-reduction variant.
//
// Host preconditions:
// the collapsed layout is K R with no outer reduced group or reduction split;
// the busiest core fits one padded tile of ONE_TILE_ELEMS;
// rowPitch <= ONE_MAX_PITCH, allowing one WholeReduceSum repeat per row;
// rows-per-core <= OneMaxRows(sizeof(T)), keeping repeat counts legal.
//
// Explicit MTE2_V and V_MTE3 flags replace the queue-supplied ordering.
// Both use event id 0 in different HardEvent pools; this route allocates no
// queue that would consume those event identifiers.
template <typename T>
class SquareSumV1RowsOne {
public:
    static constexpr uint32_t TILE_ELEMS = SS_ONE_TILE_ELEMS;
    // Rows are at least 32 B of padded pitch, and the repeat count is capped
    // the way rowBatch is.  Mirrors OneMaxRows() in the planner.
    static constexpr uint32_t MAX_ROWS =
        (TILE_ELEMS * sizeof(T) / 32 < STAGE_SLOTS) ? (TILE_ELEMS * sizeof(T) / 32) : STAGE_SLOTS;

    // The UB map, entirely at compile time.  Making these immediates rather
    // than expressions in blob fields is most of the point of the route: the
    // general path derives tmpAt_/workAt_/stageAt_ from three tiling reads and
    // spills all three to the frame before the first descriptor.
    static constexpr uint32_t OFF_IN = 0;
    static constexpr uint32_t OFF_ACC = OFF_IN + TILE_ELEMS * sizeof(T);
    static constexpr uint32_t OFF_DST = OFF_ACC + TILE_ELEMS * 4;   // fp32 row sums
    static constexpr uint32_t OFF_OUT = OFF_DST + MAX_ROWS * 4;     // T, absent for float
    static constexpr bool OUT_IS_DST = std::is_same_v<T, float>;
    static constexpr uint32_t UB_BYTES =
        OUT_IS_DST ? OFF_OUT : (OFF_OUT + (MAX_ROWS * sizeof(T) + 31) / 32 * 32);
    // Every offset is a multiple of 32 B: TILE_ELEMS is 2048 and MAX_ROWS a
    // multiple of 8, so each region length is, which is what DataCopyPad and
    // the vector unit both require.
    static_assert(OFF_ACC % 32 == 0 && OFF_DST % 32 == 0 && OFF_OUT % 32 == 0,
                  "one-shot UB regions must stay 32 B aligned");

    __aicore__ inline SquareSumV1RowsOne() {}

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
    {
        // Four tiling fields, read where they are.  They span two 64-byte GM
        // lines (usedCores at 12 and rowPitch at 28 in the first, innerLen at
        // 64 and outTiles at 88 in the second), which is two round trips and
        // is the floor for this struct layout, not for this route.
        const __gm__ SquareSumV1TilingData* t =
            reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        const uint32_t nc = t->usedCores;
        const uint32_t pitch = t->rowPitch;
        const uint64_t rowElems = t->innerLen;
        const uint64_t rows = t->outTiles;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        // Same split as SplitRange, open coded because there is one call.
        const uint64_t idx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t cores = (nc == 0) ? 1 : nc;
        if (idx >= cores) {
            return;
        }
        const uint64_t per = rows / cores;
        const uint64_t rem = rows % cores;
        const uint64_t lo = (idx < rem) ? idx * (per + 1) : rem * (per + 1) + (idx - rem) * per;
        const uint32_t nb = static_cast<uint32_t>(per + ((idx < rem) ? 1 : 0));
        if (nb == 0) {
            return;
        }

        pipe->InitBuffer(ub_, UB_BYTES);
        LocalTensor<uint8_t> a = ub_.template Get<uint8_t>();
        LocalTensor<T> in = a[OFF_IN].template ReinterpretCast<T>();

        // One descriptor.  When the row is already a whole number of 32 B
        // blocks the padded pitch equals the row, so the nb-block copy
        // describes one contiguous byte range landing at the same UB
        // addresses; issuing it as a single block costs nb-fold fewer MTE2
        // blocks, and blocks are what the DMA costs (~60 cycles each).
        const uint32_t rowBytes = static_cast<uint32_t>(rowElems) * sizeof(T);
        const uint8_t rightPad = static_cast<uint8_t>(pitch - rowElems);
        DataCopyExtParams cp;
        if (rightPad == 0) {
            cp.blockCount = 1;
            cp.blockLen = nb * rowBytes;
        } else {
            cp.blockCount = static_cast<uint16_t>(nb);
            cp.blockLen = rowBytes;
        }
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPadExtParams<T> pad{rightPad != 0, 0, rightPad, static_cast<T>(0)};
        DataCopyPad(in, xGm_[lo * rowElems], cp, pad);

        // MTE2 -> V read-after-write dependency on input. The flag is raised
        // after the input transfers retire and waited before the vector unit
        // computes squares; one pair covers the complete batch.
        SetFlag<HardEvent::MTE2_V>(ONE_EVT);
        WaitFlag<HardEvent::MTE2_V>(ONE_EVT);

        const uint32_t n = nb * pitch;
        LocalTensor<float> sq = a[OFF_ACC].template ReinterpretCast<float>();
        if constexpr (std::is_same_v<T, float>) {
            Mul(sq, in, in, static_cast<int32_t>(n));
        } else {
            Cast(sq, in, RoundMode::CAST_NONE, static_cast<int32_t>(n));
            Mul(sq, sq, sq, static_cast<int32_t>(n));
        }
        // V -> V below needs no flag: one pipe, program order, and this is the
        // same argument the arena audit in SquareSumV1Kernel already makes.
        // The pad bytes DataCopyPad wrote are zero, so their squares add zero.
        LocalTensor<float> dst = a[OFF_DST].template ReinterpretCast<float>();
        WholeReduceSum<float>(dst, sq, static_cast<int32_t>(pitch), static_cast<int32_t>(nb), 1, 1,
                              static_cast<int32_t>(pitch * sizeof(float) / 32));

        // float32 needs no narrowing, so the store reads the fp32 row sums in
        // place and the `out` region does not exist for that dtype at all.
        LocalTensor<T> out;
        if constexpr (OUT_IS_DST) {
            out = dst.template ReinterpretCast<T>();
        } else {
            out = a[OFF_OUT].template ReinterpretCast<T>();
            if constexpr (std::is_same_v<T, half>) {
                Cast(out, dst, RoundMode::CAST_NONE, static_cast<int32_t>(nb));
            } else {
                Cast(out, dst, RoundMode::CAST_RINT, static_cast<int32_t>(nb));
            }
        }

        // V -> MTE3 read-after-write dependency on output. Wait for the final
        // vector write before storing. No further UB consumer or input DMA follows
        // this work, so no additional reuse edge is needed within the route.
        SetFlag<HardEvent::V_MTE3>(ONE_EVT);
        WaitFlag<HardEvent::V_MTE3>(ONE_EVT);
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = nb * sizeof(T);
        wp.srcStride = 0;
        wp.dstStride = 0;
        DataCopyPad(yGm_[lo], out, wp);
    }

private:
    static constexpr int32_t ONE_EVT = 0;   // see the event-id note above

    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
};

// KEY_ROWS_HDR and KEY_ROWS_BARE use the same row arithmetic and explicit
// MTE2_V / V_MTE3 dependencies as the straight-line row route.
// The host packs oneHdr with row lengths and the quotient/remainder of the
// row split; the kernel reads one word and derives per-core ranges without
// division. Appending the field preserves earlier offsets.
//
// The bare variant builds LocalTensor views directly over the same UB layout.
// It explicitly performs InitSocState and the final PIPE_ALL barrier normally
// provided by TPipe. Both producer/consumer flag pairs remain required.
template <typename T, bool BARE>
class SquareSumV1RowsHdr {
public:
    static constexpr uint32_t TILE_ELEMS = SS_ONE_TILE_ELEMS;
    static constexpr uint32_t MAX_ROWS =
        (TILE_ELEMS * sizeof(T) / 32 < STAGE_SLOTS) ? (TILE_ELEMS * sizeof(T) / 32) : STAGE_SLOTS;

    // Use the same UB layout as SquareSumV1RowsOne.
    static constexpr uint32_t OFF_IN = 0;
    static constexpr uint32_t OFF_ACC = OFF_IN + TILE_ELEMS * sizeof(T);
    static constexpr uint32_t OFF_DST = OFF_ACC + TILE_ELEMS * 4;
    static constexpr uint32_t OFF_OUT = OFF_DST + MAX_ROWS * 4;
    static constexpr bool OUT_IS_DST = std::is_same_v<T, float>;
    static constexpr uint32_t UB_BYTES =
        OUT_IS_DST ? OFF_OUT : (OFF_OUT + (MAX_ROWS * sizeof(T) + 31) / 32 * 32);
    static_assert(OFF_ACC % 32 == 0 && OFF_DST % 32 == 0 && OFF_OUT % 32 == 0,
                  "one-shot UB regions must stay 32 B aligned");

    __aicore__ inline SquareSumV1RowsHdr() {}

    // `pipe` is nullptr and unused when BARE.
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
    {
        if constexpr (BARE) {
            // The one thing TPipe's constructor does that this route needs.
            InitSocState();
        }

        // Read the packed tiling word next to the global-buffer setup.
        // The following unpacking operates on register values.
        const __gm__ SquareSumV1TilingData* t =
            reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        const uint64_t hdr = t->oneHdr;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        // PackOneHdr, op_host/square_sum_v1_plan.h.  Shifts and masks on a
        // value already in a register: no memory touched.
        const uint32_t rowElems = static_cast<uint32_t>(hdr & 0xffffu);
        const uint32_t pitch = static_cast<uint32_t>((hdr >> 16) & 0xffffu);
        const uint32_t per = static_cast<uint32_t>((hdr >> 32) & 0xffffu);
        const uint32_t rem = static_cast<uint32_t>((hdr >> 48) & 0xffu);
        const uint32_t cores = static_cast<uint32_t>((hdr >> 56) & 0xffu);

        // The same split KEY_ROWS_ONE open-codes, with the divide and the
        // modulo already done on the host.  For idx < rem this is idx*(per+1);
        // for idx >= rem it is rem*(per+1) + (idx-rem)*per.
        const uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        if (idx >= cores) {
            return;
        }
        const uint32_t nb = per + ((idx < rem) ? 1u : 0u);
        if (nb == 0) {
            return;
        }
        const uint32_t lo = idx * per + ((idx < rem) ? idx : rem);

        LocalTensor<uint8_t> a;
        if constexpr (BARE) {
            a = LocalTensor<uint8_t>(TPosition::VECCALC, 0, UB_BYTES);
        } else {
            pipe->InitBuffer(ub_, UB_BYTES);
            a = ub_.template Get<uint8_t>();
        }
        LocalTensor<T> in = a[OFF_IN].template ReinterpretCast<T>();

        const uint32_t rowBytes = rowElems * sizeof(T);
        const uint8_t rightPad = static_cast<uint8_t>(pitch - rowElems);
        DataCopyExtParams cp;
        if (rightPad == 0) {
            cp.blockCount = 1;
            cp.blockLen = nb * rowBytes;
        } else {
            cp.blockCount = static_cast<uint16_t>(nb);
            cp.blockLen = rowBytes;
        }
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPadExtParams<T> pad{rightPad != 0, 0, rightPad, static_cast<T>(0)};
        DataCopyPad(in, xGm_[static_cast<uint64_t>(lo) * rowElems], cp, pad);

        // ---- EDGE 1, MTE2 -> V (RAW on `in`) ----
        // The vector unit below reads exactly the bytes the DataCopyPad above
        // writes.  MTE2 raises the flag only after every previously issued MTE2
        // instruction has retired, so one pair covers the whole copy whether it
        // went out as one block or as nb of them.
        SetFlag<HardEvent::MTE2_V>(ONE_EVT);
        WaitFlag<HardEvent::MTE2_V>(ONE_EVT);

        const uint32_t n = nb * pitch;
        LocalTensor<float> sq = a[OFF_ACC].template ReinterpretCast<float>();
        if constexpr (std::is_same_v<T, float>) {
            Mul(sq, in, in, static_cast<int32_t>(n));
        } else {
            Cast(sq, in, RoundMode::CAST_NONE, static_cast<int32_t>(n));
            Mul(sq, sq, sq, static_cast<int32_t>(n));
        }
        // V -> V needs no flag: one pipe, program order.  The pad bytes
        // DataCopyPad wrote are zero, so their squares add zero.
        LocalTensor<float> dst = a[OFF_DST].template ReinterpretCast<float>();
        WholeReduceSum<float>(dst, sq, static_cast<int32_t>(pitch), static_cast<int32_t>(nb), 1, 1,
                              static_cast<int32_t>(pitch * sizeof(float) / 32));

        LocalTensor<T> out;
        if constexpr (OUT_IS_DST) {
            out = dst.template ReinterpretCast<T>();
        } else {
            out = a[OFF_OUT].template ReinterpretCast<T>();
            if constexpr (std::is_same_v<T, half>) {
                Cast(out, dst, RoundMode::CAST_NONE, static_cast<int32_t>(nb));
            } else {
                Cast(out, dst, RoundMode::CAST_RINT, static_cast<int32_t>(nb));
            }
        }

        // ---- EDGE 2, V -> MTE3 (RAW on `out`) ----
        // MTE3 reads the bytes the last vector instruction wrote.  Greater
        // traced this exact pair on its own one-shot route and found MTE3
        // arriving at the store 55 ns BEFORE the vector pipe had finished
        // writing, then sitting on the flag.  There is no MTE2 after the vector
        // work and nothing reads UB afterwards, so no WAR pair is needed.
        SetFlag<HardEvent::V_MTE3>(ONE_EVT);
        WaitFlag<HardEvent::V_MTE3>(ONE_EVT);
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = nb * sizeof(T);
        wp.srcStride = 0;
        wp.dstStride = 0;
        DataCopyPad(yGm_[lo], out, wp);

        if constexpr (BARE) {
            // The one thing ~TPipe() does that is load-bearing.  The keys that
            // still build a TPipe inherit it from the destructor instead.
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    static constexpr int32_t ONE_EVT = 0;

    // The BARE specialization discards this branch at compile time and never
    // uses the TBuf or writes an InitBuffer record.
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
};
}  // namespace

// ================================ KEY 13 ====================================
// MICRO: whole tensor <= 8 KB.  One core, one load, one vector square, a
// scalar odometer fold, one store.  No TPipe (key 6's trick), no queues, no
// workspace.  Exact for every shape the parser admits: the fold walks the
// same group arrays every other route decodes, in TileBase order (group
// [nK-1] fastest), so output ordering is identical by construction.
template <typename T>
class SquareSumV1Micro {
public:
    static constexpr uint32_t IN_BYTES = 8192;
    static constexpr uint32_t OFF_IN = 0;
    static constexpr uint32_t OFF_W = IN_BYTES;                // float squares
    static constexpr uint32_t OFF_OF = OFF_W + IN_BYTES * 2;   // float outputs
    static constexpr uint32_t OFF_OT = OFF_OF + IN_BYTES * 2;  // typed outputs
    static constexpr uint32_t UB_BYTES = OFF_OT + IN_BYTES;
    static constexpr int32_t EVT = 0;

    __aicore__ inline SquareSumV1Micro() {}

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        InitSocState();
        if (GetBlockIdx() != 0) {
            return;
        }
        const __gm__ SquareSumV1TilingData* t =
            reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        GlobalTensor<T> xGm;
        GlobalTensor<T> yGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        const uint32_t innerRed = t->innerIsReduce;
        const uint64_t inner = t->innerLen;
        const uint64_t totalOut = t->totalOut;
        const uint64_t totalIn = t->outTiles * t->redPerTile * inner;
        const uint64_t ks0 = t->kSize[0], ks1 = t->kSize[1], ks2 = t->kSize[2], ks3 = t->kSize[3];
        const uint64_t kt0 = t->kStride[0], kt1 = t->kStride[1], kt2 = t->kStride[2], kt3 = t->kStride[3];
        const uint64_t rs0 = t->rSize[0], rs1 = t->rSize[1], rs2 = t->rSize[2], rs3 = t->rSize[3];
        const uint64_t rt0 = t->rStride[0], rt1 = t->rStride[1], rt2 = t->rStride[2], rt3 = t->rStride[3];

        LocalTensor<uint8_t> a(TPosition::VECCALC, 0, UB_BYTES);
        LocalTensor<T> in = a[OFF_IN].template ReinterpretCast<T>();
        LocalTensor<float> w = a[OFF_W].template ReinterpretCast<float>();
        LocalTensor<float> oF = a[OFF_OF].template ReinterpretCast<float>();

        DataCopyExtParams cp{1, static_cast<uint32_t>(totalIn * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(in, xGm[0], cp, noPad);
        SetFlag<HardEvent::MTE2_V>(EVT);
        WaitFlag<HardEvent::MTE2_V>(EVT);
        if constexpr (std::is_same_v<T, float>) {
            Mul(w, in, in, static_cast<int32_t>(totalIn));
        } else {
            Cast(w, in, RoundMode::CAST_NONE, static_cast<int32_t>(totalIn));
            Mul(w, w, w, static_cast<int32_t>(totalIn));
        }
        SetFlag<HardEvent::V_S>(EVT);
        WaitFlag<HardEvent::V_S>(EVT);

        uint64_t tI = 0;
        for (uint64_t a0 = 0; a0 < ks0; ++a0)
        for (uint64_t a1 = 0; a1 < ks1; ++a1)
        for (uint64_t a2 = 0; a2 < ks2; ++a2)
        for (uint64_t a3 = 0; a3 < ks3; ++a3) {
            const uint64_t kOff = a0 * kt0 + a1 * kt1 + a2 * kt2 + a3 * kt3;
            if (innerRed != 0) {
                float accv = 0.0f;
                for (uint64_t b0 = 0; b0 < rs0; ++b0)
                for (uint64_t b1 = 0; b1 < rs1; ++b1)
                for (uint64_t b2 = 0; b2 < rs2; ++b2)
                for (uint64_t b3 = 0; b3 < rs3; ++b3) {
                    const uint64_t base = kOff + b0 * rt0 + b1 * rt1 + b2 * rt2 + b3 * rt3;
                    for (uint64_t i = 0; i < inner; ++i) {
                        accv += w.GetValue(static_cast<uint32_t>(base + i));
                    }
                }
                oF.SetValue(static_cast<uint32_t>(tI), accv);
                ++tI;
            } else {
                const uint32_t ob = static_cast<uint32_t>(tI * inner);
                for (uint64_t i = 0; i < inner; ++i) {
                    oF.SetValue(ob + static_cast<uint32_t>(i), 0.0f);
                }
                for (uint64_t b0 = 0; b0 < rs0; ++b0)
                for (uint64_t b1 = 0; b1 < rs1; ++b1)
                for (uint64_t b2 = 0; b2 < rs2; ++b2)
                for (uint64_t b3 = 0; b3 < rs3; ++b3) {
                    const uint64_t base = kOff + b0 * rt0 + b1 * rt1 + b2 * rt2 + b3 * rt3;
                    for (uint64_t i = 0; i < inner; ++i) {
                        const uint32_t oi = ob + static_cast<uint32_t>(i);
                        oF.SetValue(oi,
                                    oF.GetValue(oi) + w.GetValue(static_cast<uint32_t>(base + i)));
                    }
                }
                ++tI;
            }
        }

        DataCopyExtParams wp{1, static_cast<uint32_t>(totalOut * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same_v<T, float>) {
            SetFlag<HardEvent::S_MTE3>(EVT);
            WaitFlag<HardEvent::S_MTE3>(EVT);
            DataCopyPad(yGm[0], oF, wp);
        } else {
            SetFlag<HardEvent::S_V>(EVT);
            WaitFlag<HardEvent::S_V>(EVT);
            LocalTensor<T> oT = a[OFF_OT].template ReinterpretCast<T>();
            if constexpr (std::is_same_v<T, half>) {
                Cast(oT, oF, RoundMode::CAST_NONE, static_cast<int32_t>(totalOut));
            } else {
                Cast(oT, oF, RoundMode::CAST_RINT, static_cast<int32_t>(totalOut));
            }
            SetFlag<HardEvent::V_MTE3>(EVT);
            WaitFlag<HardEvent::V_MTE3>(EVT);
            DataCopyPad(yGm[0], oT, wp);
        }
    }
};

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    // Initialize the system workspace before any route calls no-argument
    // SyncAll, which reads GetSysWorkSpacePtr().
    AscendC::SetSysWorkspace(workspace);

    // Every key returned by TilingFunc needs an explicit TILING_KEY_IS(N)
    // entry so the toolchain emits its launchable variant. A bare else branch
    // does not define another key.
    //
    // Construct each kernel class and TPipe inside the branches that use them.
    // The bare row variant manages its own initialization and final barrier
    // and must not acquire a function-scope TPipe.
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<0>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<0>();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<1>(x, y, workspace, tiling, &pipe);
        op.ProcessRows();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<2>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<2>();
    } else if (TILING_KEY_IS(29)) {
        KERNEL_TASK_TYPE(29,KERNEL_TYPE_AIV_ONLY);
        ss_layer_pairs::Kernel op;
        op.Run(x,y,tiling);
    } else if (TILING_KEY_IS(26)) {
        KERNEL_TASK_TYPE(26,KERNEL_TYPE_MIX_AIV_1_0);
        ss_cols_raw_bf16::Kernel op;
        op.Run(x,y,workspace,tiling);
    } else if (TILING_KEY_IS(25)) {
        KERNEL_TASK_TYPE(25,KERNEL_TYPE_MIX_AIV_1_0);
        ss_cols_raw_f32::Kernel op;
        op.Run(x,y,workspace,tiling);
    } else if (TILING_KEY_IS(14)) {
        KERNEL_TASK_TYPE(14, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<14>(x, y, workspace, tiling, &pipe);
        op.ProcessFlatRows();
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<3>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<3>();
    } else if (TILING_KEY_IS(8)) {
        // The vector polled combine: the CONTIG body splits the run range
        // across chunks, publishes n-float partial slots, and chunk 0 folds
        // them with vector Adds.  Same no-zeroed-memory contract as key 7.
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIV_1_0);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X, true> op;
        op.Init<2>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<2>();
    } else if (TILING_KEY_IS(12)) {
        // K R K layered polled: contiguous layer streams, one slot per chunk.
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_MIX_AIV_1_0);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X, true> op;
        op.Init<3>(x, y, workspace, tiling, &pipe);
        op.ProcessKRKPolled();
    } else if (TILING_KEY_IS(11)) {
        // PAD-polled: VectorTasks batches the strided runs, chunks publish
        // n-float slots, chunk 0 folds -- the K R K family at full width.
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIV_1_0);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X, true> op;
        op.Init<3>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<3>();
    } else if (TILING_KEY_IS(10)) {
        // Layered sequential walk: contiguous layers, Gather-transposed tail.
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<10>(x, y, workspace, tiling, &pipe);
        op.ProcessLayered();
    } else if (TILING_KEY_IS(9)) {
        // Batched strided scalar: R K R with a short reduced inner run.
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init<0>(x, y, workspace, tiling, &pipe);
        op.ProcessStridedScalar();
    } else if (TILING_KEY_IS(7)) {
        // The atomic-split reduction: generic stage-1 partials fold in GM
        // through the MTE3 atomic adder; no SyncAll, no stage 2.  The host
        // sets the framework atomic-clean flag for exactly this key.
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIV_1_0);
        TPipe pipe;
        SquareSumV1Kernel<DTYPE_X, true> op;
        op.Init<0>(x, y, workspace, tiling, &pipe);
        op.ProcessGeneric<0>();
    } else if (TILING_KEY_IS(4)) {
        // The straight-line row route.  A separate class, not a branch inside
        // SquareSumV1Kernel: each key is its own compiled variant, so a
        // variant that never loops should not carry the loop, the three
        // InitBuffers or the two TQues.
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1RowsOne<DTYPE_X> one;
        one.Run(x, y, tiling, &pipe);
    } else if (TILING_KEY_IS(5)) {
        // Rung one: one packed tiling word, no division.  TPipe kept, so the
        // only difference from key 4 is how the tiling reaches the kernel.
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_AIV_ONLY);
        TPipe pipe;
        SquareSumV1RowsHdr<DTYPE_X, false> one;
        one.Run(x, y, tiling, &pipe);
    } else if (TILING_KEY_IS(13)) {
        // Tiny-total direct: no TPipe, one core (see the class comment).
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_AIV_ONLY);
        SquareSumV1Micro<DTYPE_X> mic;
        mic.Run(x, y, tiling);
    } else if (TILING_KEY_IS(6)) {
        // Rung two: the same, with TPipe/TBuf/InitBuffer gone as well.  No
        // TPipe is constructed in this variant at all.
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
        SquareSumV1RowsHdr<DTYPE_X, true> one;
        one.Run(x, y, tiling, nullptr);
    }

}
