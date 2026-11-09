#ifndef TP_CLOCK_TIMER
#define TP_CLOCK_TIMER 0
#endif
// Transpose (torch.permute) device side.
//
// Four routes, all driven from the same canonical description the host builds
// in transpose_plan.h:
//
//   key 1 (bulk) - rowCount rows of `rowLen` contiguous elements. Source base
//                  of row r comes from an odometer over the outer output
//                  axes; destination base is exactly r * rowLen. Rows wider
//                  than UB are split into chunks, and (row, chunk) pairs are
//                  what gets spread over the cores. Covers identity /
//                  reshape-only permutations and every long-contiguous-suffix
//                  shape, including suffixes larger than UB.
//
//   key 2 (rows) - the same row abstraction, but a row is assembled from
//                  `blkPerRow` DMA blocks of `blkLenElems` elements taken at
//                  `srcBlkPitch` spacing. Several rows are staged in UB and
//                  flushed with a single contiguous store, because the
//                  destination is provably contiguous across rows.
//                  With blkPerRow == 1 this is the batched short-segment
//                  route; with blkLenElems == 1 it is the general transpose.
//
//   key 4 (fold) - the short-contiguous-suffix family again, but a run of
//                  `foldLen` consecutive rows that share one constant source
//                  stride is issued as a SINGLE strided DataCopyPad instead of
//                  one per row. The host picks which odometer axis supplies
//                  that run; when it is the innermost one the destination is
//                  still contiguous and several runs share one store, and
//                  otherwise the store is strided by whole rows.
//
//   key 3 (tile) - a real 2-D transpose. A tile is read as contiguous rows,
//                  transposed inside UB by one Gather, and written as
//                  contiguous rows of the other axis, so no DMA burst is ever
//                  shorter than a 32-byte data block. Only for b16/b32 with
//                  both exchanged extents at least a data block wide.
//
//   key 5 (tile-one) - the key-3 decomposition for a core that owns exactly one
//                  tile, run straight through with no odometer or loop.
//
//   key 6 (tile-i8) - the tile route for int8, which has no native 1-byte
//                  Gather. Same tile decomposition as key 3, but each tile is
//                  Cast int8->half, transposed by the half Gather, and Cast
//                  half->int8 before the store. int8 in [-128,127] is exact in
//                  float16, so the bridge is bit-exact; see TransposeTileI8Kernel.
//
// Keys 1 and 2 alone cover every legal input, so a correct fallback is always
// resident in the binary; keys 3, 4, 5 and 6 are strictly opt-in.

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/index/arithprogression.h"

// MIX code generation calls matmul::clearWorkspace on the AIC before entry.
// This operator uses its own workspace protocol and no KFC matmul server,
// so the adapter below is a no-op. Keep this kernel flags in the 4..11 range
// to avoid the framework-reserved flag 15.
namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t*) {}
}

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr int32_t MAX_ODO = 8;

// Compile-time element capacity of the one-shot tile UB map. It must match
// kOneTileElems in the host planner, which enforces the launch bound.
constexpr uint32_t ONE_TILE_ELEMS = 8192;

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

// Declare the tiling fields consumed by each route. KEY == 0 reads all fields.
// Route-specific initialization writes only fields used by that route;
// unused members intentionally have no default initializer.
template <uint32_t KEY>
struct RouteNeeds {
    static constexpr bool kAll = (KEY == 0);
    static constexpr bool kBulk = kAll || (KEY == 1);
    static constexpr bool kRows = kAll || (KEY == 2);
    static constexpr bool kTile = kAll || (KEY == 3);
    static constexpr bool kFold = kAll || (KEY == 4);
    // SetupRows sizes the staging buffer from rowsPerBatch * chunkBlks *
    // slotBytes and is shared by the rows and fold routes.
    static constexpr bool kStage = kRows || kFold;
    static constexpr bool kPitch = kRows || kTile;
};

// CPU-debug poisoning fills fields omitted by RouteNeeds with zero.
// This exposes unintended reads of route state while avoiding unbounded
// allocation or loop counts from arbitrary uninitialized values.
#ifndef TRANSPOSE_POISON_UNUSED
#if defined(ASCENDC_CPU_DEBUG)
#define TRANSPOSE_POISON_UNUSED 1
#else
#define TRANSPOSE_POISON_UNUSED 0
#endif
#endif

}  // namespace

template <typename T>
class TransposeKernel {
public:
    __aicore__ inline TransposeKernel() {}

    // Template on the tiling type and KEY. Route-specific initialization writes
    // only required fields; KEY == 0 writes them all. See RouteNeeds for the
    // field sets and the intentionally uninitialized members.
    template <uint32_t KEY, typename TD>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TD& t, TPipe* pipe)
    {
        InitP<KEY>(x, y, &t, pipe);
    }

    // The pointer-based initializer reads tiling fields from the caller-selected
    // address space. A GM pointer avoids materializing the full structure locally.
    template <uint32_t KEY, typename PtrT>
    __aicore__ inline void InitP(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        using N = RouteNeeds<KEY>;
#if TRANSPOSE_POISON_UNUSED
        if (!N::kPitch) { srcBlkPitch_ = 0; }
        if (!N::kBulk) { chunkElems_ = 0; }
        if (!N::kRows) { blkPerRow_ = 0; blkLenElems_ = 0; }
        if (!N::kStage) { rowsPerBatch_ = 0; chunkBlks_ = 0; slotBytes_ = 0; }
        if (!N::kTile) {
            tileK_ = 0; dstRowStride_ = 0; tileMT_ = 0; tileKT_ = 0; tilePrefetch_ = 0;
        }
        if (!N::kFold) {
            foldLen_ = 0; foldExt_ = 0; foldSrcStride_ = 0; foldDstRowStride_ = 0;
        }
#endif
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        outerRank_ = t->outerRank;
        usedCores_ = t->usedCores;
        if (N::kPitch) {
            srcBlkPitch_ = t->srcBlkPitch;
        }
        if (N::kBulk) {
            chunkElems_ = t->chunkElems;
        }
        if (N::kRows) {
            blkPerRow_ = t->blkPerRow;
            blkLenElems_ = t->blkLenElems;
        }
        if (N::kStage) {
            rowsPerBatch_ = t->rowsPerBatch;
            chunkBlks_ = t->chunkBlks;
            slotBytes_ = t->slotBytes;
        }
        if (N::kTile) {
            tileK_ = t->tileK;
            dstRowStride_ = t->dstRowStride;
            tileMT_ = t->tileMT;
            tileKT_ = t->tileKT;
            tilePrefetch_ = t->tilePrefetch;
        }
        if (N::kFold) {
            foldLen_ = t->foldLen;
            foldExt_ = t->foldExt;
            foldSrcStride_ = t->foldSrcStride;
            foldDstRowStride_ = t->foldDstRowStride;
        }
        // Copy only the active odometer axes; later code reads no axis past outerRank_.
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);
#if !defined(ASCENDC_CPU_DEBUG)
        if constexpr(KEY==4) {
            if(t->tilePrefetch==101u)xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
#endif
        pipe_ = pipe;
    }

    __aicore__ inline void SetupBulk()
    {
        uint32_t bytes = static_cast<uint32_t>(chunkElems_ * sizeof(T));
        bytes = (bytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        if (bytes == 0) {
            bytes = DATA_BLOCK_BYTES;
        }
        pipe_->InitBuffer(queue_, BUFFER_NUM, bytes);
    }

    __aicore__ inline void SetupRows()
    {
        uint64_t bytes = static_cast<uint64_t>(rowsPerBatch_) *
                         static_cast<uint64_t>(chunkBlks_) *
                         static_cast<uint64_t>(slotBytes_);
        if (bytes == 0) {
            bytes = DATA_BLOCK_BYTES;
        }
        pipe_->InitBuffer(queue_, BUFFER_NUM, static_cast<uint32_t>(bytes));
    }

    // Key 1: chunk rows of contiguous elements. Build descriptors in the loop.
    // The optional kSeg form handles chunksPerRow == 1 with segmented index ranges.
    template <bool kSeg = false>
    __aicore__ inline void ProcessBulk()
    {
        if (numel_ == 0 || rowCount_ == 0 || rowLen_ == 0 || chunkElems_ == 0) {
            return;
        }
        const uint64_t chunksPerRow = (rowLen_ + chunkElems_ - 1) / chunkElems_;
        const uint64_t totalChunks = rowCount_ * chunksPerRow;

        uint64_t begin;
        uint64_t end;
        SplitRange(totalChunks, begin, end);
        if (begin >= end) {
            return;
        }

        uint64_t row = begin / chunksPerRow;
        OdoSet(row);
        uint64_t rowFirstChunk = row * chunksPerRow;

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};

        if (kSeg && chunksPerRow == 1) {
            const uint32_t bytes = static_cast<uint32_t>(rowLen_ * sizeof(T));
            DataCopyExtParams desc{1, bytes, 0, 0, 0};
            uint64_t dst = begin * rowLen_;
            for (uint64_t r = begin; r < end; ++r) {
                LocalTensor<T> local = queue_.template AllocTensor<T>();
                DataCopyPad(local, xGm_[odoBase_], desc, pad);
                queue_.EnQue(local);
                local = queue_.template DeQue<T>();
                DataCopyPad(yGm_[dst], local, desc);
                queue_.FreeTensor(local);
                dst += rowLen_;
                OdoNext();
            }
            return;
        }

        for (uint64_t idx = begin; idx < end; ++idx) {
            if (idx >= rowFirstChunk + chunksPerRow) {
                OdoNext();
                ++row;
                rowFirstChunk += chunksPerRow;
            }
            const uint64_t c = idx - rowFirstChunk;
            const uint64_t offset = c * chunkElems_;
            const uint64_t len = MinU64(chunkElems_, rowLen_ - offset);
            const uint32_t bytes = static_cast<uint32_t>(len * sizeof(T));

            LocalTensor<T> local = queue_.template AllocTensor<T>();
            DataCopyExtParams in{1, bytes, 0, 0, 0};
            DataCopyPad(local, xGm_[odoBase_ + offset], in, pad);
            queue_.EnQue(local);
            local = queue_.template DeQue<T>();
            DataCopyExtParams out{1, bytes, 0, 0, 0};
            DataCopyPad(yGm_[row * rowLen_ + offset], local, out);
            queue_.FreeTensor(local);
        }
    }

    // -----------------------------------------------------------------
    // key 2: rows assembled from blocks, batched into one store.
    // -----------------------------------------------------------------
    __aicore__ inline void ProcessRows()
    {
        if (numel_ == 0 || rowCount_ == 0 || rowLen_ == 0 ||
            blkPerRow_ == 0 || blkLenElems_ == 0 || chunkBlks_ == 0 ||
            rowsPerBatch_ == 0) {
            return;
        }
        const uint32_t chunksPerRow =
            (blkPerRow_ + chunkBlks_ - 1) / chunkBlks_;
        const uint32_t blockBytes = static_cast<uint32_t>(blkLenElems_ * sizeof(T));
        const uint32_t slotElems = slotBytes_ / static_cast<uint32_t>(sizeof(T));
        const uint32_t srcGapBytes = (srcBlkPitch_ > blkLenElems_)
            ? static_cast<uint32_t>((srcBlkPitch_ - blkLenElems_) * sizeof(T))
            : 0;
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};

        if (chunksPerRow <= 1) {
            // A whole row fits in UB: gather `rows` rows, flush them with a
            // single contiguous store. Work unit is one row.
            uint64_t begin;
            uint64_t end;
            SplitRange(rowCount_, begin, end);
            if (begin >= end) {
                return;
            }
            OdoSet(begin);
            for (uint64_t rowStart = begin; rowStart < end; rowStart += rowsPerBatch_) {
                const uint32_t rows =
                    static_cast<uint32_t>(MinU64(rowsPerBatch_, end - rowStart));
                LocalTensor<T> local = queue_.template AllocTensor<T>();
                for (uint32_t r = 0; r < rows; ++r) {
                    // Construct descriptors for the current transfer.
                    DataCopyExtParams in{static_cast<uint16_t>(blkPerRow_), blockBytes,
                                         srcGapBytes, 0, 0};
                    DataCopyPad(local[static_cast<uint64_t>(r) * blkPerRow_ * slotElems],
                                xGm_[odoBase_], in, pad);
                    OdoNext();
                }
                queue_.EnQue(local);
                local = queue_.template DeQue<T>();
                DataCopyExtParams out{static_cast<uint16_t>(rows * blkPerRow_),
                                      blockBytes, 0, 0, 0};
                DataCopyPad(yGm_[rowStart * rowLen_], local, out);
                queue_.FreeTensor(local);
            }
        } else {
            // A row is split into chunks, either because it is wider than UB
            // or because there are fewer rows than cores. Work unit is one
            // (row, chunk) pair. blkLenElems_ is 1 whenever blkPerRow_ > 1,
            // so a chunk of `blks` blocks is exactly `blks` destination
            // elements.
            uint64_t begin;
            uint64_t end;
            SplitRange(rowCount_ * chunksPerRow, begin, end);
            if (begin >= end) {
                return;
            }
            uint64_t row = begin / chunksPerRow;
            OdoSet(row);
            uint64_t rowFirst = row * chunksPerRow;
            for (uint64_t idx = begin; idx < end; ++idx) {
                if (idx >= rowFirst + chunksPerRow) {
                    OdoNext();
                    ++row;
                    rowFirst += chunksPerRow;
                }
                const uint32_t c =
                    static_cast<uint32_t>(idx - rowFirst) * chunkBlks_;
                const uint32_t blks =
                    (blkPerRow_ - c < chunkBlks_) ? (blkPerRow_ - c) : chunkBlks_;
                LocalTensor<T> local = queue_.template AllocTensor<T>();
                DataCopyExtParams in{static_cast<uint16_t>(blks), blockBytes,
                                     srcGapBytes, 0, 0};
                DataCopyPad(local,
                            xGm_[odoBase_ + static_cast<uint64_t>(c) * srcBlkPitch_],
                            in, pad);
                queue_.EnQue(local);
                local = queue_.template DeQue<T>();
                DataCopyExtParams out{static_cast<uint16_t>(blks), blockBytes, 0, 0, 0};
                DataCopyPad(yGm_[row * rowLen_ +
                                 static_cast<uint64_t>(c) * blkLenElems_],
                            local, out);
                queue_.FreeTensor(local);
            }
        }
    }

    // -----------------------------------------------------------------
    // key 4: a run of constant-stride rows as one strided DataCopyPad.
    //
    // The host has pulled one odometer axis out of the odometer and handed it
    // over as (foldExt_, foldSrcStride_, foldDstRowStride_). A work unit is
    // (odometer position g, chunk c of that axis), and covers
    //   rows  odoOutBase_ + (c*foldLen_ + j) * foldDstRowStride_,  j < len
    //   from  odoBase_    + (c*foldLen_ + j) * foldSrcStride_
    // which is one DataCopyPad on each side: `len` blocks of rowLen_ elements,
    // a constant gap between them. Where the whole row route spends one
    // descriptor and one DataCopyExtParams construction per row, this spends
    // one per `len` rows.
    // -----------------------------------------------------------------
    __aicore__ inline void ProcessFold()
    {
        if (numel_ == 0 || rowCount_ == 0 || rowLen_ == 0 || foldLen_ == 0 ||
            foldExt_ == 0 || rowsPerBatch_ == 0) {
            return;
        }
        const uint32_t rowBytes = static_cast<uint32_t>(rowLen_ * sizeof(T));
        const uint32_t slotElems = slotBytes_ / static_cast<uint32_t>(sizeof(T));
        const uint32_t srcGapBytes = static_cast<uint32_t>(
            (foldSrcStride_ - rowLen_) * sizeof(T));
        const uint64_t chunks = (foldExt_ + foldLen_ - 1) / foldLen_;
        const uint64_t groups = rowCount_ / foldExt_;

        uint64_t begin;
        uint64_t end;
        SplitRange(groups * chunks, begin, end);
        if (begin >= end) {
            return;
        }
        uint64_t g = begin / chunks;
        uint64_t c = begin - g * chunks;
        OdoSet(g);

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};

        if (foldDstRowStride_ == 1) {
            // Destination rows are consecutive across chunks and across
            // odometer positions alike, so one contiguous store can flush as
            // many runs as UB holds.
            uint64_t unit = begin;
            while (unit < end) {
                LocalTensor<T> local = queue_.template AllocTensor<T>();
                const uint64_t firstRow = odoOutBase_ + c * foldLen_;
                uint32_t staged = 0;
                // `staged == 0` guarantees forward progress even if a future
                // planner change ever sized the batch below one run; without it
                // the loop could spin, which under the simulator means a kill,
                // not a failure.
                while (unit < end && (staged == 0 || staged + foldLen_ <= rowsPerBatch_)) {
                    const uint64_t off = c * foldLen_;
                    const uint32_t len =
                        static_cast<uint32_t>(MinU64(foldLen_, foldExt_ - off));
                    DataCopyExtParams in{static_cast<uint16_t>(len), rowBytes,
                                         srcGapBytes, 0, 0};
                    DataCopyPad(local[static_cast<uint64_t>(staged) * slotElems],
                                xGm_[odoBase_ + off * foldSrcStride_], in, pad);
                    staged += len;
                    ++unit;
                    if (++c == chunks) {
                        c = 0;
                        OdoNext();
                    }
                }
                queue_.EnQue(local);
                local = queue_.template DeQue<T>();
                DataCopyExtParams out{static_cast<uint16_t>(staged), rowBytes, 0, 0, 0};
                DataCopyPad(yGm_[firstRow * rowLen_], local, out);
                queue_.FreeTensor(local);
            }
        } else {
            const uint32_t dstGapBytes =
                static_cast<uint32_t>((foldDstRowStride_ - 1) * rowLen_ * sizeof(T));
            for (uint64_t unit = begin; unit < end; ++unit) {
                const uint64_t off = c * foldLen_;
                const uint32_t len =
                    static_cast<uint32_t>(MinU64(foldLen_, foldExt_ - off));
                LocalTensor<T> local = queue_.template AllocTensor<T>();
                DataCopyExtParams in{static_cast<uint16_t>(len), rowBytes, srcGapBytes, 0, 0};
                DataCopyPad(local, xGm_[odoBase_ + off * foldSrcStride_], in, pad);
                queue_.EnQue(local);
                local = queue_.template DeQue<T>();
                DataCopyExtParams out{static_cast<uint16_t>(len), rowBytes, 0,
                                      dstGapBytes, 0};
                DataCopyPad(yGm_[(odoOutBase_ + off * foldDstRowStride_) * rowLen_],
                            local, out);
                queue_.FreeTensor(local);
                if (++c == chunks) {
                    c = 0;
                    OdoNext();
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // key 3: 2-D tile transposed inside UB.
    //
    // The two axes being exchanged are i (extent rowLen_, input stride
    // srcBlkPitch_, output stride 1) and k (extent tileK_, input stride 1,
    // output stride dstRowStride_). A tile is read as mi rows of kk
    // contiguous elements (>= 32 B per burst), transposed by one Gather, and
    // written as kk rows of mi contiguous elements (>= 32 B per burst). No
    // single-element DMA block occurs anywhere on this route.
    // -----------------------------------------------------------------
    __aicore__ inline void SetupTile()
    {
        const uint32_t tileBytes = tileMT_ * tileKT_ * static_cast<uint32_t>(sizeof(T));
        const uint32_t offBytes = tileMT_ * tileKT_ * static_cast<uint32_t>(sizeof(uint32_t));
        // Double-buffered queues rather than raw TBuf: the framework inserts
        // every hazard edge, including the write-after-read from FreeTensor to
        // the next AllocTensor that a hand-rolled flag set is easy to miss,
        // and the second buffer restores MTE2 / V / MTE3 overlap.
        pipe_->InitBuffer(tileInQue_, BUFFER_NUM, tileBytes);
        pipe_->InitBuffer(tileOutQue_, BUFFER_NUM, tileBytes);
        // Initialize the input, output, and offset-table regions together before DMA.
        pipe_->InitBuffer(tileOff_, offBytes);
    }

    // Derive tile coordinates from the linear unit index.
    __aicore__ inline void ProcessTile()
    {
        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileMT_ == 0 || tileKT_ == 0) {
            return;
        }
        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t pitchInBytes = tileKT_ * ds;   // multiple of 32 by construction
        const uint32_t pitchOutBytes = tileMT_ * ds;  // multiple of 32 by construction

        LocalTensor<uint32_t> ubOff = tileOff_.template Get<uint32_t>();

        // Build the offset table once from tile geometry before issuing tile loads.
        // ubOut[k * tileMT_ + i] reads ubIn[i * tileKT_ + k].
        {
            // off[k * tileMT_ + i] = i * pitchInBytes + k * ds.
            // Build rows by repeated doubling: an existing row block is copied with
            // a built * ds offset. The barriers preserve dependencies between steps.
            LocalTensor<int32_t> offI = tileOff_.template Get<int32_t>();
            Arange<int32_t>(offI, 0, static_cast<int32_t>(pitchInBytes),
                            static_cast<int32_t>(tileMT_));
            PipeBarrier<PIPE_V>();
            uint32_t built = 1;
            while (built < tileKT_) {
                const uint32_t take =
                    (tileKT_ - built < built) ? (tileKT_ - built) : built;
                // Source and destination never overlap: take <= built.
                Adds<int32_t>(offI[built * tileMT_], offI,
                              static_cast<int32_t>(built * ds),
                              static_cast<int32_t>(take * tileMT_));
                PipeBarrier<PIPE_V>();
                built += take;
            }
        }

        const uint64_t tilesM = (rowLen_ + tileMT_ - 1) / tileMT_;
        const uint64_t tilesK = (tileK_ + tileKT_ - 1) / tileKT_;
        const uint64_t perBatch = tilesM * tilesK;
        uint64_t begin;
        uint64_t end;
        SplitRange(rowCount_ * perBatch, begin, end);
        if (begin >= end) {
            return;
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        uint64_t batch = begin / perBatch;
        OdoSet(batch);
        uint64_t batchFirst = batch * perBatch;

        if (tilePrefetch_ != 0) {
            // Software-pipelined loop: issue tile n+1 input before waiting for tile n
            // Gather, keeping a load in flight during the previous tile vector work
            // and output transfer.
            uint32_t pMi = 0;
            uint32_t pKk = 0;
            uint64_t pI0 = 0;
            uint64_t pK0 = 0;
            uint64_t pOut = 0;
            TileIssueLoad(begin - batchFirst, tilesK, pitchInBytes, ds, pad, pMi, pKk, pI0,
                          pK0, pOut);

            uint64_t unit = begin;
            while (unit < end) {
                const uint32_t mi = pMi;
                const uint32_t kk = pKk;
                const uint64_t i0 = pI0;
                const uint64_t k0 = pK0;
                const uint64_t outBase = pOut;

                const uint64_t next = unit + 1;
                if (next < end) {
                    if (next >= batchFirst + perBatch) {
                        OdoNext();
                        ++batch;
                        batchFirst += perBatch;
                    }
                    TileIssueLoad(next - batchFirst, tilesK, pitchInBytes, ds, pad, pMi, pKk,
                                  pI0, pK0, pOut);
                }

                LocalTensor<T> ubIn = tileInQue_.template DeQue<T>();
                LocalTensor<T> ubOut = tileOutQue_.template AllocTensor<T>();
                Gather(ubOut, ubIn, ubOff, static_cast<uint32_t>(0), tileMT_ * tileKT_);
                tileInQue_.FreeTensor(ubIn);
                tileOutQue_.EnQue(ubOut);
                ubOut = tileOutQue_.template DeQue<T>();

                const uint32_t miBytes = mi * ds;
                const uint32_t miSlot = (miBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
                                        DATA_BLOCK_BYTES;
                DataCopyExtParams out{static_cast<uint16_t>(kk), miBytes,
                                      (pitchOutBytes - miSlot) / DATA_BLOCK_BYTES,
                                      static_cast<uint32_t>((dstRowStride_ - mi) * ds), 0};
                DataCopyPad(yGm_[outBase + k0 * dstRowStride_ + i0], ubOut, out);
                tileOutQue_.FreeTensor(ubOut);
                unit = next;
            }
            return;
        }

        for (uint64_t unit = begin; unit < end; ++unit) {
            if (unit >= batchFirst + perBatch) {
                OdoNext();
                ++batch;
                batchFirst += perBatch;
            }
            const uint64_t within = unit - batchFirst;
            const uint64_t tm = within / tilesK;
            const uint64_t tk = within - tm * tilesK;
            const uint64_t i0 = tm * tileMT_;
            const uint64_t k0 = tk * tileKT_;
            const uint32_t mi = static_cast<uint32_t>(MinU64(tileMT_, rowLen_ - i0));
            const uint32_t kk = static_cast<uint32_t>(MinU64(tileKT_, tileK_ - k0));

            const uint32_t kkBytes = kk * ds;
            const uint32_t kkSlot = (kkBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
                                    DATA_BLOCK_BYTES;
            LocalTensor<T> ubIn = tileInQue_.template AllocTensor<T>();
            DataCopyExtParams in{static_cast<uint16_t>(mi), kkBytes,
                                 static_cast<uint32_t>((srcBlkPitch_ - kk) * ds),
                                 (pitchInBytes - kkSlot) / DATA_BLOCK_BYTES, 0};
            DataCopyPad(ubIn, xGm_[odoBase_ + i0 * srcBlkPitch_ + k0], in, pad);
            tileInQue_.EnQue(ubIn);
            ubIn = tileInQue_.template DeQue<T>();

            LocalTensor<T> ubOut = tileOutQue_.template AllocTensor<T>();
            Gather(ubOut, ubIn, ubOff, static_cast<uint32_t>(0), tileMT_ * tileKT_);
            tileInQue_.FreeTensor(ubIn);
            tileOutQue_.EnQue(ubOut);
            ubOut = tileOutQue_.template DeQue<T>();

            const uint32_t miBytes = mi * ds;
            const uint32_t miSlot = (miBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
                                    DATA_BLOCK_BYTES;
            DataCopyExtParams out{static_cast<uint16_t>(kk), miBytes,
                                  (pitchOutBytes - miSlot) / DATA_BLOCK_BYTES,
                                  static_cast<uint32_t>((dstRowStride_ - mi) * ds), 0};
            DataCopyPad(yGm_[odoOutBase_ + k0 * dstRowStride_ + i0], ubOut, out);
            tileOutQue_.FreeTensor(ubOut);
        }
    }

private:
    // Computes tile `within`'s geometry from the odometer's current position
    // and issues its load. Split out so the pipelined loop can run one load
    // ahead without duplicating the index arithmetic. Returns the geometry the
    // caller must keep in order to place the store once that load lands.
    __aicore__ inline void TileIssueLoad(uint64_t within, uint64_t tilesK,
                                         uint32_t pitchInBytes, uint32_t ds,
                                         const DataCopyPadExtParams<T>& pad, uint32_t& mi,
                                         uint32_t& kk, uint64_t& i0, uint64_t& k0,
                                         uint64_t& outBase)
    {
        const uint64_t tm = within / tilesK;
        const uint64_t tk = within - tm * tilesK;
        i0 = tm * tileMT_;
        k0 = tk * tileKT_;
        mi = static_cast<uint32_t>(MinU64(tileMT_, rowLen_ - i0));
        kk = static_cast<uint32_t>(MinU64(tileKT_, tileK_ - k0));
        outBase = odoOutBase_;

        const uint32_t kkBytes = kk * ds;
        const uint32_t kkSlot =
            (kkBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        LocalTensor<T> ubIn = tileInQue_.template AllocTensor<T>();
        DataCopyExtParams in{static_cast<uint16_t>(mi), kkBytes,
                             static_cast<uint32_t>((srcBlkPitch_ - kk) * ds),
                             (pitchInBytes - kkSlot) / DATA_BLOCK_BYTES, 0};
        DataCopyPad(ubIn, xGm_[odoBase_ + i0 * srcBlkPitch_ + k0], in, pad);
        tileInQue_.EnQue(ubIn);
    }

    // Contiguous split of `work` items over usedCores_ blocks.
    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    // Odometer over the outer output axes; index 0 is the slowest axis.
    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tileInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tileOutQue_;
    TBuf<QuePosition::VECCALC> tileOff_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    // Members are intentionally uninitialized. Init<KEY> or OdoSet writes every
    // field before its route reads it. RouteNeeds defines the required field set.
    // TRANSPOSE_POISON_UNUSED can fill unused fields with sentinel values to
    // expose an omitted dependency during CPU checks.
    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t chunkElems_;
    uint32_t blkPerRow_;
    uint32_t blkLenElems_;
    uint32_t rowsPerBatch_;
    uint32_t chunkBlks_;
    uint32_t slotBytes_;
    uint32_t outerRank_;
    uint32_t usedCores_;

    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t tileKT_;

    uint32_t foldLen_;
    uint64_t foldExt_;
    uint64_t foldSrcStride_;
    uint64_t foldDstRowStride_;
    uint32_t tilePrefetch_;

    // Init() fills [0, outerRank_) and OdoSet() zeroes the same range; nothing
    // ever reads past it. Zero-initialising all eight cost 32 scalar stores.
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// Key 5: straight-line transpose for a core that owns exactly one tile.
// Use one input transfer, Gather, and one output transfer with a fixed UB map.
//
// The host must establish all preconditions:
// units == blockDim, so GetBlockIdx() is the tile index;
// outerRank == 0, so no batch odometer is needed;
// tileMT * tileKT <= ONE_TILE_ELEMS, so the UB map fits.
template <typename T>
class TransposeOneKernel {
public:
    static constexpr uint32_t S = sizeof(T);
    // Compile-time UB regions are each ONE_TILE_ELEMS wide. Their fixed offsets
    // allow each selected tile to use the same bounded layout.
    static constexpr uint32_t OFF_IN = 0;
    static constexpr uint32_t OFF_OUT = OFF_IN + ONE_TILE_ELEMS * S;
    static constexpr uint32_t OFF_TAB = OFF_OUT + ONE_TILE_ELEMS * S;
    static constexpr uint32_t UB_BYTES = OFF_TAB + ONE_TILE_ELEMS * 4;

    __aicore__ inline TransposeOneKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        // Read tiling fields next to the global-buffer setup. This field set spans
        // the first two tiling cache lines.
        const uint64_t numel = t->numel;
        const uint64_t rowLen = t->rowLen;
        const uint64_t srcBlkPitch = t->srcBlkPitch;
        const uint32_t usedCores = t->usedCores;
        const uint64_t tileK = t->tileK;
        const uint64_t dstRowStride = t->dstRowStride;
        const uint32_t tileMT = t->tileMT;
        const uint32_t tileKT = t->tileKT;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        const uint64_t me = GetBlockIdx();
        if (numel == 0 || rowLen == 0 || tileK == 0 || tileMT == 0 || tileKT == 0 ||
            me >= static_cast<uint64_t>(usedCores)) {
            return;
        }

        // This core's tile. `units == usedCores` is the host's precondition, and
        // SplitRange over an equal split would give exactly [me, me+1), so the
        // unit index IS the block index and the division that produced it in the
        // general route disappears.
        const uint64_t tilesK = (tileK + tileKT - 1) / tileKT;
        const uint64_t tm = me / tilesK;
        const uint64_t tk = me - tm * tilesK;
        const uint64_t i0 = tm * tileMT;
        const uint64_t k0 = tk * tileKT;
        const uint32_t mi = static_cast<uint32_t>(MinU64(tileMT, rowLen - i0));
        const uint32_t kk = static_cast<uint32_t>(MinU64(tileKT, tileK - k0));

        pipe->InitBuffer(ub_, UB_BYTES);
        LocalTensor<uint8_t> a = ub_.template Get<uint8_t>();
        LocalTensor<T> ubIn = a[OFF_IN].template ReinterpretCast<T>();
        LocalTensor<T> ubOut = a[OFF_OUT].template ReinterpretCast<T>();

        const uint32_t pitchInBytes = tileKT * S;   // multiple of 32 by construction
        const uint32_t pitchOutBytes = tileMT * S;  // Multiple of 32 by construction.
        //
        // Build the same offset arithmetic as ProcessTile before the first input load.
        // The table uses a separate UB region, so its vector writes do not overlap DMA.
        {
            LocalTensor<int32_t> offI = a[OFF_TAB].template ReinterpretCast<int32_t>();
            Arange<int32_t>(offI, 0, static_cast<int32_t>(pitchInBytes),
                            static_cast<int32_t>(tileMT));
            PipeBarrier<PIPE_V>();
            uint32_t built = 1;
            while (built < tileKT) {
                const uint32_t take = (tileKT - built < built) ? (tileKT - built) : built;
                Adds<int32_t>(offI[built * tileMT], offI, static_cast<int32_t>(built * S),
                              static_cast<int32_t>(take * tileMT));
                PipeBarrier<PIPE_V>();
                built += take;
            }
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t kkBytes = kk * S;
        const uint32_t kkSlot =
            (kkBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        DataCopyExtParams in{static_cast<uint16_t>(mi), kkBytes,
                             static_cast<uint32_t>((srcBlkPitch - kk) * S),
                             (pitchInBytes - kkSlot) / DATA_BLOCK_BYTES, 0};
        DataCopyPad(ubIn, xGm_[i0 * srcBlkPitch + k0], in, pad);

        // MTE2 -> V read-after-write dependency on ubIn. Raise the flag after the
        // input transfer and wait before Gather reads the loaded bytes. Without
        // queue-provided events, this explicit pair supplies the required ordering.
        SetFlag<HardEvent::MTE2_V>(ONE_EVT);
        WaitFlag<HardEvent::MTE2_V>(ONE_EVT);

        LocalTensor<uint32_t> ubOff = a[OFF_TAB].template ReinterpretCast<uint32_t>();
        Gather(ubOut, ubIn, ubOff, static_cast<uint32_t>(0), tileMT * tileKT);

        // ---- EDGE 2, V -> MTE3 (read-after-write on ubOut) ----
        // Raised by the vector pipe after the Gather retires, waited by MTE3
        // before the store reads ubOut. Greater's trace of the same structure
        // caught MTE3 arriving 55 ns early and sitting on this flag, so it is
        // load-bearing rather than defensive.
        SetFlag<HardEvent::V_MTE3>(ONE_EVT);
        WaitFlag<HardEvent::V_MTE3>(ONE_EVT);

        const uint32_t miBytes = mi * S;
        const uint32_t miSlot =
            (miBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        DataCopyExtParams out{static_cast<uint16_t>(kk), miBytes,
                              (pitchOutBytes - miSlot) / DATA_BLOCK_BYTES,
                              static_cast<uint32_t>((dstRowStride - mi) * S), 0};
        DataCopyPad(yGm_[k0 * dstRowStride + i0], ubOut, out);
    }

private:
    // Event id 0 for both edges. The TPipe event pool is indexed per HardEvent,
    // so MTE2_V id 0 and V_MTE3 id 0 are different hardware flags, and this
    // variant allocates no TQue, so nothing else has taken an id.
    static constexpr int32_t ONE_EVT = 0;

    TBuf<QuePosition::VECCALC> ub_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
};

// Key 6: int8 tile transpose through a half-precision intermediate.
// Gather supports two- and four-byte elements, so cast int8 to half, transpose
// the half buffer, then cast back using CAST_RINT. Every int8 value is exactly
// representable in half and remains integral through the permutation.
//
// The class has separate half buffers and uses half-layout byte offsets.
// The template compiles with every dtype object, but only sizeof(T)==1
// executes this body; the host selects this key only for int8.
template <typename T>
class TransposeTileI8Kernel {
public:
    __aicore__ inline TransposeTileI8Kernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 1) {
            RunImpl(x, y, t, pipe);
        }
    }

private:
    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        // Read the key-3 field set once from the GM tiling pointer.
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        tileMT_ = t->tileMT;
        tileKT_ = t->tileKT;
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);

        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileMT_ == 0 || tileKT_ == 0) {
            return;
        }

        // UB layout: two double-buffered int8 queues (in and out, one work unit
        // in flight each) plus three single TBufs -- the two half work buffers
        // and the Gather offset table. The host sized the tile so this all fits
        // (transpose_plan.h, the isI8 arm of tileBytes).
        const uint32_t elems = tileMT_ * tileKT_;
        pipe->InitBuffer(inQue_, BUFFER_NUM, elems * static_cast<uint32_t>(sizeof(T)));
        pipe->InitBuffer(outQue_, BUFFER_NUM, elems * static_cast<uint32_t>(sizeof(T)));
        pipe->InitBuffer(halfInBuf_, elems * static_cast<uint32_t>(sizeof(half)));
        pipe->InitBuffer(halfOutBuf_, elems * static_cast<uint32_t>(sizeof(half)));
        pipe->InitBuffer(offBuf_, elems * static_cast<uint32_t>(sizeof(uint32_t)));

        // Offset table for the HALF buffer: off[k*tileMT + i] = i*(tileKT*2) +
        // k*2. Same doubling construction as ProcessTile (1 + ceil(log2(tileKT))
        // vector calls), but the element stride is sizeof(half) because the
        // Gather reads the half copy. All PIPE_V, writing a region no DMA on
        // this route touches.
        const uint32_t halfDs = static_cast<uint32_t>(sizeof(half));  // 2
        const uint32_t pitchInHalfBytes = tileKT_ * halfDs;
        {
            LocalTensor<int32_t> offI = offBuf_.template Get<int32_t>();
            Arange<int32_t>(offI, 0, static_cast<int32_t>(pitchInHalfBytes),
                            static_cast<int32_t>(tileMT_));
            PipeBarrier<PIPE_V>();
            uint32_t built = 1;
            while (built < tileKT_) {
                const uint32_t take = (tileKT_ - built < built) ? (tileKT_ - built) : built;
                Adds<int32_t>(offI[built * tileMT_], offI, static_cast<int32_t>(built * halfDs),
                              static_cast<int32_t>(take * tileMT_));
                PipeBarrier<PIPE_V>();
                built += take;
            }
        }

        const uint64_t tilesM = (rowLen_ + tileMT_ - 1) / tileMT_;
        const uint64_t tilesK = (tileK_ + tileKT_ - 1) / tileKT_;
        const uint64_t perBatch = tilesM * tilesK;
        uint64_t begin;
        uint64_t end;
        SplitRange(rowCount_ * perBatch, begin, end);
        if (begin >= end) {
            return;
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        uint64_t batch = begin / perBatch;
        OdoSet(batch);
        uint64_t batchFirst = batch * perBatch;

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));  // 1
        const uint32_t pitchInBytes = tileKT_ * ds;   // multiple of 32 by construction
        const uint32_t pitchOutBytes = tileMT_ * ds;  // multiple of 32 by construction

        LocalTensor<uint32_t> ubOff = offBuf_.template Get<uint32_t>();
        LocalTensor<half> ubHalfIn = halfInBuf_.template Get<half>();
        LocalTensor<half> ubHalfOut = halfOutBuf_.template Get<half>();

        for (uint64_t unit = begin; unit < end; ++unit) {
            if (unit >= batchFirst + perBatch) {
                OdoNext();
                ++batch;
                batchFirst += perBatch;
            }
            const uint64_t within = unit - batchFirst;
            const uint64_t tm = within / tilesK;
            const uint64_t tk = within - tm * tilesK;
            const uint64_t i0 = tm * tileMT_;
            const uint64_t k0 = tk * tileKT_;
            const uint32_t mi = static_cast<uint32_t>(MinU64(tileMT_, rowLen_ - i0));
            const uint32_t kk = static_cast<uint32_t>(MinU64(tileKT_, tileK_ - k0));

            const uint32_t kkBytes = kk * ds;
            const uint32_t kkSlot =
                (kkBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            LocalTensor<T> ubIn = inQue_.template AllocTensor<T>();
            DataCopyExtParams in{static_cast<uint16_t>(mi), kkBytes,
                                 static_cast<uint32_t>((srcBlkPitch_ - kk) * ds),
                                 (pitchInBytes - kkSlot) / DATA_BLOCK_BYTES, 0};
            DataCopyPad(ubIn, xGm_[odoBase_ + i0 * srcBlkPitch_ + k0], in, pad);
            inQue_.EnQue(ubIn);
            ubIn = inQue_.template DeQue<T>();

            // int8 -> half over the whole tile region. The int8 tile is
            // contiguous at pitch tileKT (== pitchInBytes), so element (i,k) sits
            // at i*tileKT + k, and the half copy inherits that layout at pitch
            // tileKT*2 -- exactly what the offset table addresses. Padding lanes
            // (i >= mi or k >= kk) hold stale int8 and are cast too, but the
            // store below only reads the [0,kk) x [0,mi) region that was loaded,
            // so they never reach the output.
            Cast(ubHalfIn, ubIn, RoundMode::CAST_NONE, elems);
            inQue_.FreeTensor(ubIn);
            PipeBarrier<PIPE_V>();

            Gather(ubHalfOut, ubHalfIn, ubOff, static_cast<uint32_t>(0), elems);
            PipeBarrier<PIPE_V>();

            LocalTensor<T> ubOut = outQue_.template AllocTensor<T>();
            // half -> int8. Every lane holds an exact integer in [-128,127], so
            // CAST_RINT (round to nearest) is the identity on the values.
            Cast(ubOut, ubHalfOut, RoundMode::CAST_RINT, elems);
            outQue_.EnQue(ubOut);
            ubOut = outQue_.template DeQue<T>();

            const uint32_t miBytes = mi * ds;
            const uint32_t miSlot =
                (miBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            DataCopyExtParams out{static_cast<uint16_t>(kk), miBytes,
                                  (pitchOutBytes - miSlot) / DATA_BLOCK_BYTES,
                                  static_cast<uint32_t>((dstRowStride_ - mi) * ds), 0};
            DataCopyPad(yGm_[odoOutBase_ + k0 * dstRowStride_ + i0], ubOut, out);
            outQue_.FreeTensor(ubOut);
        }
    }

    // Contiguous split of `work` items over usedCores_ blocks. Mirrors
    // TransposeKernel::SplitRange; duplicated so the b16/b32 class stays
    // untouched.
    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    // Odometer over the outer output axes; index 0 is the slowest axis. Mirrors
    // TransposeKernel's OdoSet/OdoNext.
    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<QuePosition::VECCALC> halfInBuf_;
    TBuf<QuePosition::VECCALC> halfOutBuf_;
    TBuf<QuePosition::VECCALC> offBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t tileKT_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// Key 7: transpose into packed UB output and store each tile as one block.
// A zero GM gap does not merge padded UB rows; the Gather table must produce
// a contiguous output sequence. For M == rowLen:
// off[j] = j*P - floor(j/M)*(M*P - ds).
//
// The table computes floor(j/M) as floor((j + 0.5f)/M). Its rounding margin
// requires N < 4194304; the host caps N at 8192. Offset-zero vector operations
// avoid unaligned vector destinations when M*ds is not a multiple of 32.
//
// Host preconditions:
// tilesM == 1; rowLen == dstRowStride; tileKT*ds is a multiple of 32;
// tileKT*rowLen <= kMaxGatherElems;
// (tileKT*rowLen - 1) * tileKT*ds fits in int32.
template <typename T>
class TransposeTilePackKernel {
public:
    __aicore__ inline TransposeTilePackKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        tileKT_ = t->tileKT;
        // tilePrefetch carries the number of adjacent K tiles per input slab.
        // Gather keeps a one-tile table and selects each tile through srcBase.
        slabT_ = t->tilePrefetch == 0 ? 1u : t->tilePrefetch;
        // Mode 9 uses two TransDataTo5HD calls for M <= 32. The unused source
        // pointers fill panel columns that the store discards. DataCopyPad uses
        // blockLen=M*ds and a padded UB source stride to write the output rows.
        vnchwMode_ = (slabT_ == 9u);
        if (vnchwMode_) {
            slabT_ = 6u;   // keep the T-slab read aggregation
        }
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);

        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileKT_ == 0) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t M = static_cast<uint32_t>(rowLen_);   // tilesM == 1
        const uint32_t T_ = slabT_;
        const uint32_t pitchInBytes = tileKT_ * T_ * ds;     // slab row pitch
        const uint32_t nOff = tileKT_ * M;                   // Gather element count

        // UB: the in queue holds M rows of the T-tile slab (single-buffered
        // when T > 1 -- the wide load amortizes the lost overlap), the out
        // queue the PACKED tile, and the table/scratch one tile's offsets.
        if (vnchwMode_ && M > 32u) {
            return;   // host gate guarantees this; fail closed anyway
        }
        pipe->InitBuffer(inQue_, T_ > 1u ? 1 : BUFFER_NUM, M * pitchInBytes);
        pipe->InitBuffer(outQue_, BUFFER_NUM,
                         vnchwMode_ ? tileKT_ * 32u * ds : nOff * ds);
        if (!vnchwMode_) {
            pipe->InitBuffer(offBuf_, nOff * static_cast<uint32_t>(sizeof(uint32_t)));
            pipe->InitBuffer(scrBuf_, nOff * static_cast<uint32_t>(sizeof(uint32_t)));
        }
        evHalfRdy_[0] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        evHalfRdy_[1] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        evHalfFree_[0] = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        evHalfFree_[1] = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());

        // ---- the offset table, full-length ops at offset zero only ----------
        if (!vnchwMode_) {
            LocalTensor<int32_t> tab = offBuf_.template Get<int32_t>();
            LocalTensor<int32_t> scrI = scrBuf_.template Get<int32_t>();
            LocalTensor<float> scrF = scrBuf_.template Get<float>();
            // scr = j
            Arange<int32_t>(scrI, 0, 1, nOff);
            PipeBarrier<PIPE_V>();
            // scr = (float)j, then (j + 0.5) / M, then floor -> k = j / M
            Cast(scrF, scrI, RoundMode::CAST_NONE, nOff);
            PipeBarrier<PIPE_V>();
            Adds<float>(scrF, scrF, 0.5f, nOff);
            PipeBarrier<PIPE_V>();
            Muls<float>(scrF, scrF, 1.0f / static_cast<float>(static_cast<int32_t>(M)), nOff);
            PipeBarrier<PIPE_V>();
            Cast(scrI, scrF, RoundMode::CAST_FLOOR, nOff);
            PipeBarrier<PIPE_V>();
            // scr = -k * (M*P - ds)
            Muls<int32_t>(scrI, scrI,
                          -static_cast<int32_t>(M * pitchInBytes - ds), nOff);
            PipeBarrier<PIPE_V>();
            // tab = j*P, then tab += scr  ->  off[j] = i*P + k*ds
            Arange<int32_t>(tab, 0, static_cast<int32_t>(pitchInBytes), nOff);
            PipeBarrier<PIPE_V>();
            Add<int32_t>(tab, tab, scrI, nOff);
            PipeBarrier<PIPE_V>();
        }

        const uint64_t slabSpan = static_cast<uint64_t>(tileKT_) * T_;
        const uint64_t tilesK = (tileK_ + slabSpan - 1) / slabSpan;
        uint64_t begin;
        uint64_t end;
        SplitRange(rowCount_ * tilesK, begin, end);
        if (begin >= end) {
            return;
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        uint64_t batch = begin / tilesK;
        OdoSet(batch);
        uint64_t batchFirst = batch * tilesK;

        LocalTensor<uint32_t> ubOff;
        if (!vnchwMode_) {
            ubOff = offBuf_.template Get<uint32_t>();
        }

        for (uint64_t unit = begin; unit < end; ++unit) {
            if (unit >= batchFirst + tilesK) {
                OdoNext();
                ++batch;
                batchFirst += tilesK;
            }
            const uint64_t k0 = (unit - batchFirst) * slabSpan;
            const uint32_t kkT = static_cast<uint32_t>(MinU64(slabSpan, tileK_ - k0));

            // T > 1 splits the slab load in two HALVES (ceil(T/2) tiles
            // each) landing in the two halves of the one slab buffer, and
            // overlaps half B's load with half A's gathers via two events:
            // the T=4 pipe put vec at 60.9us fully SERIAL behind an 81.9us
            // read pipe; the ping-pong buys that overlap back without the
            // second slab copy UB cannot afford.
            const uint32_t tilesHere = (kkT + tileKT_ - 1) / tileKT_;
            const uint32_t halfTiles = (T_ > 1u) ? ((T_ + 1u) / 2u) : 1u;
            LocalTensor<T> ubIn = inQue_.template AllocTensor<T>();
            uint32_t loaded = 0;
            uint32_t halfIdx = 0;
            while (loaded < tilesHere) {
                const uint32_t hTiles =
                    (tilesHere - loaded) < halfTiles ? (tilesHere - loaded)
                                                     : halfTiles;
                const uint32_t hk0 = loaded * tileKT_;
                const uint32_t hkk = static_cast<uint32_t>(MinU64(
                    static_cast<uint64_t>(hTiles) * tileKT_,
                    static_cast<uint64_t>(kkT) - hk0));
                const uint32_t hkBytes = hkk * ds;
                const uint32_t hkSlot = (hkBytes + DATA_BLOCK_BYTES - 1) /
                                        DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
                if (halfIdx >= 2u) {
                    AscendC::WaitFlag<HardEvent::V_MTE2>(evHalfFree_[halfIdx & 1u]);
                }
                DataCopyExtParams in{static_cast<uint16_t>(M), hkBytes,
                                     static_cast<uint32_t>((srcBlkPitch_ - hkk) * ds),
                                     (pitchInBytes - hkSlot) / DATA_BLOCK_BYTES, 0};
                DataCopyPad(ubIn[hk0], xGm_[odoBase_ + k0 + hk0], in, pad);
                AscendC::SetFlag<HardEvent::MTE2_V>(evHalfRdy_[halfIdx & 1u]);
                AscendC::WaitFlag<HardEvent::MTE2_V>(evHalfRdy_[halfIdx & 1u]);
                for (uint32_t tt = loaded; tt < loaded + hTiles; ++tt) {
                    const uint32_t kk = static_cast<uint32_t>(MinU64(
                        tileKT_, static_cast<uint64_t>(kkT) - tt * tileKT_));
                    LocalTensor<T> ubOut = outQue_.template AllocTensor<T>();
                    if (vnchwMode_) {
                        // Transpose kt x M into a 32-column panel: call one
                        // covers source rows 0..15 -> columns 0..15, call
                        // two rows 16..M-1 (padded with dummy row-0 sources)
                        // -> columns 16..31. kSnapV = ceil8(kk).
                        const uint32_t kSnapV = (kk + 7u) / 8u * 8u;
                        TransDataTo5HDParams cnvV{false, false,
                                                  static_cast<uint8_t>(kSnapV / 8u),
                                                  32, 1};
                        uint64_t srcL[16];
                        uint64_t dstL[16];
                        for (uint32_t half = 0; half < 2u; ++half) {
                            const uint32_t base = half * 16u;
                            if (base >= M) {
                                break;
                            }
                            for (uint32_t r = 0; r < 16u; ++r) {
                                const uint32_t row = (base + r < M) ? (base + r) : 0u;
                                srcL[r] = reinterpret_cast<uint64_t>(
                                    ubIn[static_cast<uint64_t>(row) * (pitchInBytes / ds) +
                                         tt * tileKT_]
                                        .GetPhyAddr());
                            }
                            for (uint32_t r = 0; r < 8u; ++r) {
                                const uint64_t rowAddr = reinterpret_cast<uint64_t>(
                                    ubOut[static_cast<uint64_t>(r) * 32u + base]
                                        .GetPhyAddr());
                                dstL[2u * r] = rowAddr;
                                dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                            }
                            TransDataTo5HD<T>(dstL, srcL, cnvV);
                        }
                        outQue_.EnQue(ubOut);
                        ubOut = outQue_.template DeQue<T>();
                        // One descriptor, kk packed rows: blockLen M*ds with
                        // the ceil32 pitch bridged by srcStride (32B units);
                        // the framework discards the pad bytes at GM.
                        const uint32_t rowB = M * ds;
                        const uint32_t rowSlot =
                            (rowB + DATA_BLOCK_BYTES - 1u) / DATA_BLOCK_BYTES *
                            DATA_BLOCK_BYTES;
                        DataCopyExtParams outV{static_cast<uint16_t>(kk), rowB,
                                               (32u * 4u - rowSlot) /
                                                   DATA_BLOCK_BYTES,
                                               0, 0};
                        DataCopyPad(
                            yGm_[odoOutBase_ + (k0 + tt * tileKT_) * dstRowStride_],
                            ubOut, outV);
                        outQue_.FreeTensor(ubOut);
                    } else {
                    Gather(ubOut, ubIn, ubOff,
                           static_cast<uint32_t>(tt * tileKT_ * ds), nOff);
                    outQue_.EnQue(ubOut);
                    ubOut = outQue_.template DeQue<T>();
                    DataCopyExtParams out{1, kk * M * ds, 0, 0, 0};
                    DataCopyPad(
                        yGm_[odoOutBase_ + (k0 + tt * tileKT_) * dstRowStride_],
                        ubOut, out);
                    outQue_.FreeTensor(ubOut);
                    }
                }
                AscendC::SetFlag<HardEvent::V_MTE2>(evHalfFree_[halfIdx & 1u]);
                loaded += hTiles;
                ++halfIdx;
            }
            // Balance the trailing half flags before the buffer goes back.
            for (uint32_t h = 0; h < (halfIdx < 2u ? halfIdx : 2u); ++h) {
                AscendC::WaitFlag<HardEvent::V_MTE2>(
                    evHalfFree_[(halfIdx - 1u - h) & 1u]);
            }
            inQue_.FreeTensor(ubIn);
        }
    }

private:
    // Mirrors TransposeKernel's; duplicated so keys 3/5/6 stay byte-identical.
    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<QuePosition::VECCALC> offBuf_;
    TBuf<QuePosition::VECCALC> scrBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileKT_;
    uint32_t slabT_;
    bool vnchwMode_ = false;
    event_t evHalfRdy_[2];
    event_t evHalfFree_[2];
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// Key 8: pack a batch of staged fold rows before storing it as one block.
// The load stages rows at a 32-byte-aligned slotBytes pitch. Gather removes
// that padding with:
// off[j] = j*ds + floor(j/rowLen)*(slotBytes - rowLen*ds).
// The table uses the same bounded float quotient construction as key 7.
//
// Host preconditions: foldDstRowStride == 1;
// rowsPerBatch * rowLen <= kMaxGatherElems;
// staging, packed output, offset table, and scratch must all fit in UB.
template <typename T>
class TransposeFoldPackKernel {
public:
    __aicore__ inline TransposeFoldPackKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        usedCores_ = t->usedCores;
        rowsPerBatch_ = t->rowsPerBatch;
        slotBytes_ = t->slotBytes;
        foldLen_ = t->foldLen;
        foldExt_ = t->foldExt;
        foldSrcStride_ = t->foldSrcStride;
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);
#if !defined(ASCENDC_CPU_DEBUG)
        if(t->tilePrefetch==101u)xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif

        if (numel_ == 0 || rowCount_ == 0 || rowLen_ == 0 || foldLen_ == 0 ||
            foldExt_ == 0 || rowsPerBatch_ == 0) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t rowBytes = static_cast<uint32_t>(rowLen_) * ds;
        const uint32_t R = rowsPerBatch_;
        const uint32_t nOff = R * static_cast<uint32_t>(rowLen_);

        pipe->InitBuffer(stageQue_, BUFFER_NUM, R * slotBytes_);
        pipe->InitBuffer(packQue_, BUFFER_NUM, R * rowBytes);
        pipe->InitBuffer(offBuf_, nOff * static_cast<uint32_t>(sizeof(uint32_t)));
        pipe->InitBuffer(scrBuf_, nOff * static_cast<uint32_t>(sizeof(uint32_t)));

        // ---- the pack table, full-length ops at offset zero only ------------
        {
            LocalTensor<int32_t> tab = offBuf_.template Get<int32_t>();
            LocalTensor<int32_t> scrI = scrBuf_.template Get<int32_t>();
            LocalTensor<float> scrF = scrBuf_.template Get<float>();
            Arange<int32_t>(scrI, 0, 1, nOff);
            PipeBarrier<PIPE_V>();
            Cast(scrF, scrI, RoundMode::CAST_NONE, nOff);
            PipeBarrier<PIPE_V>();
            Adds<float>(scrF, scrF, 0.5f, nOff);
            PipeBarrier<PIPE_V>();
            Muls<float>(scrF, scrF, 1.0f / static_cast<float>(static_cast<int32_t>(rowLen_)), nOff);
            PipeBarrier<PIPE_V>();
            Cast(scrI, scrF, RoundMode::CAST_FLOOR, nOff);   // k = j / rowLen
            PipeBarrier<PIPE_V>();
            Muls<int32_t>(scrI, scrI, static_cast<int32_t>(slotBytes_ - rowBytes), nOff);
            PipeBarrier<PIPE_V>();
            Arange<int32_t>(tab, 0, static_cast<int32_t>(ds), nOff);
            PipeBarrier<PIPE_V>();
            Add<int32_t>(tab, tab, scrI, nOff);
            PipeBarrier<PIPE_V>();
        }

        const uint32_t slotElems = slotBytes_ / ds;
        const uint64_t chunks = (foldExt_ + foldLen_ - 1) / foldLen_;
        const uint64_t groups = rowCount_ / foldExt_;

        uint64_t begin;
        uint64_t end;
        SplitRange(groups * chunks, begin, end);
        if (begin >= end) {
            return;
        }
        uint64_t g = begin / chunks;
        uint64_t c = begin - g * chunks;
        OdoSet(g);

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t srcGapBytes =
            static_cast<uint32_t>((foldSrcStride_ - rowLen_) * ds);
        LocalTensor<uint32_t> ubOff = offBuf_.template Get<uint32_t>();

        uint64_t unit = begin;
        while (unit < end) {
            LocalTensor<T> stage = stageQue_.template AllocTensor<T>();
            const uint64_t firstRow = odoOutBase_ + c * foldLen_;
            uint32_t staged = 0;
            while (unit < end && (staged == 0 || staged + foldLen_ <= R)) {
                const uint64_t off = c * foldLen_;
                const uint32_t len =
                    static_cast<uint32_t>(MinU64(foldLen_, foldExt_ - off));
                DataCopyExtParams in{static_cast<uint16_t>(len), rowBytes,
                                     srcGapBytes, 0, 0};
                DataCopyPad(stage[static_cast<uint64_t>(staged) * slotElems],
                            xGm_[odoBase_ + off * foldSrcStride_], in, pad);
                staged += len;
                ++unit;
                if (++c == chunks) {
                    c = 0;
                    OdoNext();
                }
            }
            stageQue_.EnQue(stage);
            stage = stageQue_.template DeQue<T>();

            LocalTensor<T> packed = packQue_.template AllocTensor<T>();
            // Packs the whole R-row buffer. A short batch (staged < R) gathers
            // stale rows into the tail, which the blockLen below never reaches --
            // the same arrangement as key 7's runt tile.
            Gather(packed, stage, ubOff, static_cast<uint32_t>(0), nOff);
            stageQue_.FreeTensor(stage);
            packQue_.EnQue(packed);
            packed = packQue_.template DeQue<T>();

            // THE POINT OF THE ROUTE: one block, not `staged` of them.
            DataCopyExtParams out{1, staged * rowBytes, 0, 0, 0};
            DataCopyPad(yGm_[firstRow * rowLen_], packed, out);
            packQue_.FreeTensor(packed);
        }
    }

private:
    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> stageQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> packQue_;
    TBuf<QuePosition::VECCALC> offBuf_;
    TBuf<QuePosition::VECCALC> scrBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t foldExt_;
    uint64_t foldSrcStride_;
    uint32_t rowsPerBatch_;
    uint32_t slotBytes_;
    uint32_t foldLen_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// Wide b32 panel transpose through TransDataTo5HD. Each unit spans the full
// input-inner extent and an mt-wide strip of output columns. Load 16 source
// rows per slab, transpose into the panel, and store the panel rows.
// The final strip is shifted left to retain a uniform, aligned UB pitch;
// overlapping output columns receive the same values.
//
// For dav_c220 b32 vnchwconv, each repeat consumes eight floats from each
// of 16 source pointers. Destination pointers are paired for the two
// 32-byte halves of each output row. Source and destination repeat strides
// are expressed in 32-byte units.
//
// Host preconditions: dtypeSize == 4; tileKT == kSnap == ceil(bigK, 8);
// kSnap/8 <= 255; tileMT is a multiple of 16 with 16 <= tileMT <= bigM;
// bigK <= 4095; slabs and output panel fit UB.
template <typename T>
class TransposeTileWideKernel {
public:
    __aicore__ inline TransposeTileWideKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 4) {
            RunImpl(x, y, t, pipe);
        }
    }

private:
    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        // Read the key-3 field set once from the GM tiling pointer.
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        tileMT_ = t->tileMT;
        tileKT_ = t->tileKT;
        // On key 9 tilePrefetch carries the SLAB BUFFER COUNT (2 = double
        // buffered loads, 1 = single slab traded for panel width); the route
        // has no prefetch arm of its own.
        tilePrefetch_ = t->tilePrefetch;
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);
#if defined(ASCENDC_CPU_DEBUG)
#else
        // Disable input allocation in L2 for this streaming read path.
        xGm_.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
            AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif

        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileMT_ == 0 || tileKT_ == 0 ||
            rowLen_ < tileMT_) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t mt = tileMT_;
        const uint32_t kSnap = tileKT_;
        // Mode 3 pairs double-buffered eight-row slabs to supply the 16 source
        // pointers of each vnchwconv operation.
        const bool pair8 =
            (tilePrefetch_ == 3u || tilePrefetch_ == 4u || tilePrefetch_ == 6u);
        // Mode 5 uses double-buffered 16-row slabs with wave scheduling.
        // Mode 4 pairs eight-row slabs and assigns units round-robin so neighboring
        // cores write adjacent output strips.
        const bool wave =
            (tilePrefetch_ == 4u || tilePrefetch_ == 5u || tilePrefetch_ == 6u ||
             tilePrefetch_ == 7u || tilePrefetch_ == 8u || tilePrefetch_ == 10u || tilePrefetch_ == 11u);
        // Mode 7 retains the mode-5 cache behavior.
        // Mode 8 disables output writeback caching through set_cache_notwriteback.
#if !defined(ASCENDC_CPU_DEBUG)
        if (tilePrefetch_ == 7u) {
            yGm_.template SetL2CacheHint<AscendC::CacheRwMode::WRITE>(
                AscendC::CacheMode::CACHE_MODE_NORMAL);
        }
        if (tilePrefetch_ == 8u) {
            yGm_.template SetL2CacheHint<AscendC::CacheRwMode::WRITE>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
#endif
        // Mode 10 loads a slab as one contiguous block and uses the native packed
        // row pitch for vnchwconv source pointers. Source row addresses may lack
        // 32-byte alignment; this optional mode relies on the intrinsic accepting them.
        const bool repairPacked = (tilePrefetch_ == 11u);
        const uint32_t padRows=t->rowsPerBatch==2u?2u:4u;
        const bool packedSlab = (tilePrefetch_ == 10u || repairPacked);
        const uint32_t slabRows = pair8 ? 8u : 16u;   // tp5/tp7 keep 16-row slabs
        const uint32_t slabBytes = slabRows * kSnap * ds;
        const uint32_t panelBytes = kSnap * mt * ds;  // kSnap rows: vnchwconv may
                                                      // write rows [bigK, kSnap)
        // 6 = pair8 x3 + wave: three 8-row slab buffers (24 rows of
        // prefetch) against the mt64 panel -- 183.1 KB, the deepest prefetch
        // the UB admits at 256 B stores.
        const int32_t slabBufs = (tilePrefetch_ == 1u) ? 1
                                 : (tilePrefetch_ == 6u) ? 3
                                                         : BUFFER_NUM;
        pipe->InitBuffer(slabQue_, slabBufs, slabBytes);
        pipe->InitBuffer(panelQue_, 1, panelBytes);
        LocalTensor<T> padTmp;
        LocalTensor<uint32_t> padIndex;
        if(repairPacked) {
            // Four-row staging fits with the original mt48 panel and both
            // slabs: 183040 bytes at kSnap520. The host checks this budget.
            pipe->InitBuffer(padTmpBuf_,padRows*kSnap*ds);
            pipe->InitBuffer(padIndexBuf_,padRows*kSnap*4u);
            padTmp=padTmpBuf_.template Get<T>();
            padIndex=padIndexBuf_.template Get<uint32_t>();
            auto idx=padIndex.template ReinterpretCast<int32_t>();
            SetMaskNorm();
            for(uint32_t r=0;r<padRows;++r)
                ArithProgression(idx[r*kSnap],static_cast<int32_t>(r*tileK_*ds),static_cast<int32_t>(ds),kSnap);
            PipeBarrier<PIPE_V>();
        }

        const uint64_t tilesM = (rowLen_ + mt - 1) / mt;
        const uint64_t workAll = rowCount_ * tilesM;
        uint64_t begin;
        uint64_t end;
        uint64_t step = 1;
        if (wave) {
            uint64_t cores = usedCores_ == 0 ? GetBlockNum() : usedCores_;
            if (cores == 0) {
                cores = 1;
            }
            begin = GetBlockIdx();
            end = workAll;
            step = cores;
            if (begin >= end) {
                return;
            }
        } else {
            SplitRange(workAll, begin, end);
            if (begin >= end) {
                return;
            }
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t loadGap = static_cast<uint32_t>((srcBlkPitch_ - tileK_) * ds);
        const uint32_t storeGap = static_cast<uint32_t>((dstRowStride_ - mt) * ds);
        const uint32_t subSlabs = mt / 16u;
        TransDataTo5HDParams cnv{false, false, static_cast<uint8_t>(kSnap / 8u),
                                 static_cast<uint16_t>(mt), 1};

        uint64_t batch = begin / tilesM;
        OdoSet(batch);
        uint64_t batchFirst = batch * tilesM;

        for (uint64_t unit = begin; unit < end; unit += step) {
            if (unit >= batchFirst + tilesM) {
                batch = unit / tilesM;
                OdoSet(batch);   // wave steps can cross several batches
                batchFirst = batch * tilesM;
            }
            uint64_t i0 = (unit - batchFirst) * mt;
            if (i0 + mt > rowLen_) {
                i0 = rowLen_ - mt;  // uniform-width tail: idempotent overlap
            }

            LocalTensor<T> panel = panelQue_.template AllocTensor<T>();
            for (uint32_t s = 0; s < subSlabs; ++s) {
                LocalTensor<T> slab;
                LocalTensor<T> slabB;
                if (pair8) {
                    DataCopyExtParams inA{8, static_cast<uint32_t>(tileK_ * ds),
                                          loadGap, 0, 0};
                    slab = slabQue_.template AllocTensor<T>();
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_],
                                inA, pad);
                    slabQue_.EnQue(slab);
                    slabB = slabQue_.template AllocTensor<T>();
                    DataCopyPad(slabB,
                                xGm_[odoBase_ + (i0 + 16u * s + 8u) * srcBlkPitch_],
                                inA, pad);
                    slabQue_.EnQue(slabB);
                    slab = slabQue_.template DeQue<T>();
                    slabB = slabQue_.template DeQue<T>();
                } else if (packedSlab) {
                    slab = slabQue_.template AllocTensor<T>();
                    // One fat contiguous block: 16 rows x tileK elems, packed.
                    DataCopyExtParams in{1,
                                         static_cast<uint32_t>(16ull * tileK_ * ds), 0,
                                         0, 0};
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_], in,
                                pad);
                    slabQue_.EnQue(slab);
                    slab = slabQue_.template DeQue<T>();
                } else {
                    slab = slabQue_.template AllocTensor<T>();
                    DataCopyExtParams in{16, static_cast<uint32_t>(tileK_ * ds), loadGap, 0, 0};
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_], in, pad);
                    slabQue_.EnQue(slab);
                    slab = slabQue_.template DeQue<T>();
                }

                if(repairPacked) {
                    const uint32_t count=padRows*kSnap, full=count/64u, tail=count%64u;
                    const CopyRepeatParams cp(1,1,8,8);
                    // Expand from the end like memmove. Stage each group before
                    // writing it; a destination never overlaps an unread group.
                    for(int32_t r=16-static_cast<int32_t>(padRows);r>=0;r-=static_cast<int32_t>(padRows)) {
                        Gather(padTmp,slab,padIndex,static_cast<uint32_t>(r)*tileK_*ds,count);
                        PipeBarrier<PIPE_V>();
                        auto dst=slab[static_cast<uint32_t>(r)*kSnap];
                        if(full) Copy(dst,padTmp,static_cast<uint64_t>(64),static_cast<uint8_t>(full),cp);
                        if(tail) Copy(dst[full*64],padTmp[full*64],static_cast<uint64_t>(tail),static_cast<uint8_t>(1),cp);
                        PipeBarrier<PIPE_V>();
                    }
                }
                uint64_t srcL[16];
                uint64_t dstL[16];
                for (uint32_t r = 0; r < 16u; ++r) {
                    if (pair8) {
                        const LocalTensor<T>& half = (r < 8u) ? slab : slabB;
                        srcL[r] = reinterpret_cast<uint64_t>(
                            half[static_cast<uint32_t>(r & 7u) * kSnap].GetPhyAddr());
                    } else if (packedSlab && !repairPacked) {
                        // Packed pitch is tileK elements; individual source rows may be unaligned.
                        srcL[r] = reinterpret_cast<uint64_t>(slab.GetPhyAddr()) +
                                  static_cast<uint64_t>(r) * tileK_ * ds;
                    } else {
                        srcL[r] = reinterpret_cast<uint64_t>(
                            slab[static_cast<uint32_t>(r) * kSnap].GetPhyAddr());
                    }
                }
                for (uint32_t r = 0; r < 8u; ++r) {
                    const uint64_t rowAddr = reinterpret_cast<uint64_t>(
                        panel[r * mt + 16u * s].GetPhyAddr());
                    dstL[2u * r] = rowAddr;
                    dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                }
                TransDataTo5HD<T>(dstL, srcL, cnv);
                slabQue_.FreeTensor(slab);
                if (pair8) {
                    slabQue_.FreeTensor(slabB);
                }
            }
            panelQue_.EnQue(panel);
            panel = panelQue_.template DeQue<T>();
            DataCopyExtParams out{static_cast<uint16_t>(tileK_),
                                  static_cast<uint32_t>(mt * ds), 0, storeGap, 0};
            DataCopyPad(yGm_[odoOutBase_ + i0], panel, out);
            panelQue_.FreeTensor(panel);
        }
    }

    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TBuf<QuePosition::VECCALC> padTmpBuf_,padIndexBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> slabQue_;
    TQue<QuePosition::VECOUT, 1> panelQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t tileKT_;
    uint32_t tilePrefetch_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// Key 12: two-pass transpose through alternating per-core batch images.
// Pass A stores the transposed image in scratch; pass B copies it contiguously
// to output. Interleave A(s) with B(s-1) using disjoint parity images.
// ImgReady orders image writes before reads; ImgFree orders reads before
// reusing that image two batches later. The two phases have disjoint UB regions.
//
// Host preconditions: dtypeSize == 4; dstRowStride == bigM;
// srcBlkPitch == bigK; kSnap/8 <= 255; bigM >= 96; bigK >= 8;
// slabs, panel, and at least four chunk rows fit UB;
// the per-core pair of images stays within the scratch budget.
template <typename T>
class TransposeTileL2Kernel {
public:
    __aicore__ inline TransposeTileL2Kernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 4) {
            RunImpl(x, y, ws, t, pipe);
        }
    }

private:
    static constexpr uint64_t kUbBudgetBytes = 184320ull - 4096ull;

    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        usedCores_ = t->usedCores;
        const uint32_t mt = t->tileMT;     // 48
        const uint32_t kSnap = t->tileKT;
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        if (numel_ == 0 || rowLen_ < 2ull * mt || tileK_ < 8ull || mt == 0 || kSnap == 0) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint64_t tilesP = (rowLen_ + mt - 1) / mt;
        const uint64_t panelElems = static_cast<uint64_t>(kSnap) * mt;
        const uint64_t slabElems = 8ull * kSnap;
        const uint64_t imageElems = tileK_ * rowLen_;
        const uint64_t aElems = 2ull * slabElems + panelElems;
        uint64_t chunkRows = (kUbBudgetBytes / ds - aElems) / (2ull * rowLen_);
        if (chunkRows > tileK_) {
            chunkRows = tileK_;
        }
        if (chunkRows == 0) {
            return;
        }
        const uint64_t chunks = (tileK_ + chunkRows - 1) / chunkRows;

        uint64_t cores = usedCores_ == 0 ? GetBlockNum() : usedCores_;
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores || me >= rowCount_) {
            return;
        }
        scratchGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(GetUserWorkspace(ws)) + me * 2ull * imageElems,
            2ull * imageElems);

        pipe->InitBuffer(arena_,
                         static_cast<uint32_t>((aElems + chunkRows * rowLen_) * ds));
        LocalTensor<T> base = arena_.template Get<T>();
        LocalTensor<T> slabT[2] = {base[0], base[slabElems]};
        LocalTensor<T> panelT = base[2ull * slabElems];
        const uint64_t chunkSlot2 =
            aElems + (((chunkRows * rowLen_ * ds + 31ull) & ~31ull) / ds);
        LocalTensor<T> chunkT[2] = {base[aElems], base[chunkSlot2]};

        const event_t eSlabRdy[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>())};
        const event_t eSlabFree[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>())};
        const event_t ePanelRdy =
            static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        const event_t ePanelFree =
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        const event_t eChunkRdy[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>())};
        const event_t eChunkFree[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>())};
        const event_t eImgRdy[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>())};
        const event_t eImgFree[2] = {
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>()),
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>())};

        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eImgFree[0]);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eImgFree[1]);

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t subSlabs = mt / 16u;
        TransDataTo5HDParams cnv{false, false, static_cast<uint8_t>(kSnap / 8u),
                                 static_cast<uint16_t>(mt), 1};
        const uint32_t storeGap = static_cast<uint32_t>((rowLen_ - mt) * ds);

        // Number of batches this core owns, and the pipeline over them.
        uint64_t nb = 0;
        for (uint64_t b = me; b < rowCount_; b += cores) {
            ++nb;
        }
        bool slabPrimed[2] = {false, false};
        bool chunkPrimed[2] = {false, false};
        bool panelPrimed = false;
        uint64_t chunkTick = 0;   // global chunk counter -> slot parity

        for (uint64_t sStep = 0; sStep <= nb; ++sStep) {
            const bool hasA = sStep < nb;
            const bool hasB = sStep > 0;
            const uint64_t bA = me + sStep * cores;
            const uint64_t bB = me + (sStep - 1) * cores;
            const uint64_t imgA = (sStep & 1ull) * imageElems;
            const uint64_t imgB = ((sStep - 1) & 1ull) * imageElems;
            const uint64_t odoInA = hasA ? OdoIn(bA) : 0;
            const uint64_t odoOutB = hasB ? OdoOut(bB) : 0;

            uint64_t pa = 0;
            uint64_t ch = 0;
            const uint64_t aTot = hasA ? tilesP : 0;
            const uint64_t bTot = hasB ? chunks : 0;
            if (hasA) {
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eImgFree[sStep & 1ull]);
            }
            if (hasB) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eImgRdy[(sStep - 1) & 1ull]);
            }
            // Issue each B block before its A block. Fine-grained interleaving can
            // create a cycle through the in-order DMA queues. This ordering lets chunk
            // stores drain while the next slab loads use the other engine.
            while (pa < aTot || ch < bTot) {
                const bool pickA = ch >= bTot;
                if (pickA) {
                    uint64_t j0 = pa * mt;
                    if (j0 + mt > rowLen_) {
                        j0 = rowLen_ - mt;
                    }
                    if (panelPrimed) {
                        AscendC::WaitFlag<HardEvent::MTE3_V>(ePanelFree);
                    }
                    uint32_t slot = 0;
                    for (uint32_t sSub = 0; sSub < subSlabs; ++sSub) {
                        const uint32_t slA = slot & 1u;
                        const uint32_t slB = (slot + 1u) & 1u;
                        DataCopyExtParams in{8, static_cast<uint32_t>(tileK_ * ds), 0, 0,
                                             0};
                        if (slabPrimed[slA]) {
                            AscendC::WaitFlag<HardEvent::V_MTE2>(eSlabFree[slA]);
                        }
                        DataCopyPad(slabT[slA],
                                    xGm_[odoInA + (j0 + 16ull * sSub) * srcBlkPitch_],
                                    in, pad);
                        AscendC::SetFlag<HardEvent::MTE2_V>(eSlabRdy[slA]);
                        if (slabPrimed[slB]) {
                            AscendC::WaitFlag<HardEvent::V_MTE2>(eSlabFree[slB]);
                        }
                        DataCopyPad(
                            slabT[slB],
                            xGm_[odoInA + (j0 + 16ull * sSub + 8ull) * srcBlkPitch_],
                            in, pad);
                        AscendC::SetFlag<HardEvent::MTE2_V>(eSlabRdy[slB]);
                        AscendC::WaitFlag<HardEvent::MTE2_V>(eSlabRdy[slA]);
                        AscendC::WaitFlag<HardEvent::MTE2_V>(eSlabRdy[slB]);
                        uint64_t srcL[16];
                        uint64_t dstL[16];
                        for (uint32_t r = 0; r < 16u; ++r) {
                            const LocalTensor<T>& half =
                                (r < 8u) ? slabT[slA] : slabT[slB];
                            srcL[r] = reinterpret_cast<uint64_t>(
                                half[static_cast<uint64_t>(r & 7u) * kSnap]
                                    .GetPhyAddr());
                        }
                        for (uint32_t r = 0; r < 8u; ++r) {
                            const uint64_t rowAddr = reinterpret_cast<uint64_t>(
                                panelT[static_cast<uint64_t>(r) * mt + 16ull * sSub]
                                    .GetPhyAddr());
                            dstL[2u * r] = rowAddr;
                            dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                        }
                        TransDataTo5HD<T>(dstL, srcL, cnv);
                        AscendC::SetFlag<HardEvent::V_MTE2>(eSlabFree[slA]);
                        AscendC::SetFlag<HardEvent::V_MTE2>(eSlabFree[slB]);
                        slabPrimed[slA] = true;
                        slabPrimed[slB] = true;
                        slot += 2u;
                    }
                    AscendC::SetFlag<HardEvent::V_MTE3>(ePanelRdy);
                    AscendC::WaitFlag<HardEvent::V_MTE3>(ePanelRdy);
                    DataCopyExtParams outP{static_cast<uint16_t>(tileK_),
                                           static_cast<uint32_t>(mt * ds), 0, storeGap,
                                           0};
                    DataCopyPad(scratchGm_[imgA + j0], panelT, outP);
                    AscendC::SetFlag<HardEvent::MTE3_V>(ePanelFree);
                    panelPrimed = true;
                    ++pa;
                    if (pa == aTot) {
                        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eImgRdy[sStep & 1ull]);
                    }
                } else {
                    const uint64_t k0 = ch * chunkRows;
                    uint64_t rows = chunkRows;
                    if (k0 + rows > tileK_) {
                        rows = tileK_ - k0;
                    }
                    const uint32_t cs = static_cast<uint32_t>(chunkTick & 1ull);
                    // Depth-2 wait: this references the store TWO chunks
                    // back, long complete at steady state, so the MTE2
                    // queue never head-of-line blocks the slab loads (the
                    // single-buffered flight serialized everything to
                    // 1548 us on exactly that stall).
                    if (chunkPrimed[cs]) {
                        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eChunkFree[cs]);
                    }
                    DataCopyExtParams inC{
                        1, static_cast<uint32_t>(rows * rowLen_ * ds), 0, 0, 0};
                    DataCopyPad(chunkT[cs], scratchGm_[imgB + k0 * rowLen_], inC, pad);
                    AscendC::SetFlag<HardEvent::MTE2_MTE3>(eChunkRdy[cs]);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eChunkRdy[cs]);
                    DataCopyExtParams outC{
                        1, static_cast<uint32_t>(rows * rowLen_ * ds), 0, 0, 0};
                    DataCopyPad(yGm_[odoOutB + k0 * dstRowStride_], chunkT[cs], outC);
                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eChunkFree[cs]);
                    chunkPrimed[cs] = true;
                    ++chunkTick;
                    ++ch;
                    if (ch == bTot) {
                        AscendC::SetFlag<HardEvent::MTE2_MTE3>(
                            eImgFree[(sStep - 1) & 1ull]);
                    }
                }
            }
        }
        // Balance every primed or pre-set flag before TPipe reclaims events.
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eImgFree[nb & 1ull]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eImgFree[(nb + 1ull) & 1ull]);
        for (uint32_t i = 0; i < 2u; ++i) {
            if (slabPrimed[i]) {
                AscendC::WaitFlag<HardEvent::V_MTE2>(eSlabFree[i]);
            }
        }
        for (uint32_t i = 0; i < 2u; ++i) {
            if (chunkPrimed[i]) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eChunkFree[i]);
            }
        }
        if (panelPrimed) {
            AscendC::WaitFlag<HardEvent::MTE3_V>(ePanelFree);
        }
    }

    __aicore__ inline uint64_t OdoIn(uint64_t row)
    {
        uint64_t acc = 0;
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            acc += (row % ext) * outerInStride_[a];
            row /= ext;
        }
        return acc;
    }

    __aicore__ inline uint64_t OdoOut(uint64_t row)
    {
        uint64_t acc = 0;
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            acc += (row % ext) * outerOutStride_[a];
            row /= ext;
        }
        return acc;
    }

    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> scratchGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t usedCores_;
    uint32_t outerRank_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
};

// Key 10: two-stage b32 transpose through per-core scratch.
// A unit contains one batch and a J-wide output-column group. Stage 1
// transposes 16-column panels into scratch. Stage 2 assembles Ib-row chunks
// from those panels and writes J-wide output segments.
// MTE3->MTE2 flags order scratch writes before reads; MTE2->MTE3 flags
// protect scratch reuse. Each unit belongs to one core, so the route requires
// no cross-core barrier.
//
// Host preconditions: dtypeSize == 4; tileKT == kSnap; kSnap/8 <= 255;
// J is a multiple of 16 and at least 32; 8 <= Ib <= bigK;
// Ib*16*ds satisfies the transfer layout; bigM >= 16; UB buffers fit;
// workspace holds one unit-sized scratch region per launched core.
template <typename T>
class TransposeTileViaKernel {
public:
    __aicore__ inline TransposeTileViaKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 4) {
            RunImpl(x, y, ws, t, pipe);
        }
    }

private:
    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        tileMT_ = t->tileMT;        // J: output columns per unit
        tileKT_ = t->tileKT;        // kSnap: padded slab/panel row count
        rowsPerBatch_ = t->rowsPerBatch;  // Ib: stage-2 output rows per block
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);

        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileMT_ == 0 || tileKT_ == 0 ||
            rowsPerBatch_ == 0 || rowLen_ < 16 || tileK_ < rowsPerBatch_) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t J = tileMT_;
        const uint32_t kSnap = tileKT_;
        const uint32_t Ib = rowsPerBatch_;
        const uint32_t panelsFull = J / 16u;           // panels in a FULL group
        const uint64_t groups = (rowLen_ + J - 1) / J;
        const uint32_t slabBytes = 16u * kSnap * ds;
        const uint32_t panelBytes = kSnap * 16u * ds;
        const uint32_t asmPitch = panelsFull * 16u * ds;   // bytes, multiple of 64
        const uint32_t asmBytes = Ib * asmPitch;
        const uint64_t unitWsElems = static_cast<uint64_t>(panelsFull) * kSnap * 16u;

        pipe->InitBuffer(slabQue_, BUFFER_NUM, slabBytes);
        // Use a single panel buffer to leave room for the assembled-row buffers.
        // The two phases remain serialized within each unit, keeping scratch writes
        // and subsequent reads adjacent.
        pipe->InitBuffer(panelQue_, 1, panelBytes);
        pipe->InitBuffer(asmQue_, BUFFER_NUM, asmBytes);

        uint64_t begin;
        uint64_t end;
        SplitRange(rowCount_ * groups, begin, end);
        if (begin >= end) {
            return;
        }
        wsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(GetUserWorkspace(ws)) +
                GetBlockIdx() * unitWsElems,
            unitWsElems);

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t loadGap = static_cast<uint32_t>((srcBlkPitch_ - tileK_) * ds);
        // Panel pitch is 16 elements = 64 B: 8 rows per repeat = 512 B = 16
        // dst stride units of 32 B.
        TransDataTo5HDParams cnv{false, false, static_cast<uint8_t>(kSnap / 8u), 16, 1};

        // Allocate dedicated events for scratch reuse. asmQue also uses
        // MTE2_MTE3 / MTE3_MTE2 events; sharing an identifier with its internal
        // buffer events could release the write-after-read fence too early.
        const event_t evS1S2 =
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        const event_t evS2S1 =
            static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());

        uint64_t batch = begin / groups;
        OdoSet(batch);
        uint64_t batchFirst = batch * groups;
        bool scratchInUse = false;

        for (uint64_t unit = begin; unit < end; ++unit) {
            if (unit >= batchFirst + groups) {
                OdoNext();
                ++batch;
                batchFirst += groups;
            }
            const uint64_t g = unit - batchFirst;
            const uint64_t j0 = g * J;
            const uint32_t jw = static_cast<uint32_t>(MinU64(J, rowLen_ - j0));
            const uint32_t panels = (jw + 15u) / 16u;

            // WAR: the previous unit's stage-2 reads must land before this
            // unit's stage-1 stores overwrite the same scratch.
            if (scratchInUse) {
                SetFlag<HardEvent::MTE2_MTE3>(evS2S1);
                WaitFlag<HardEvent::MTE2_MTE3>(evS2S1);
            }
            scratchInUse = true;

            // ---- stage 1: J-group -> scratch panels ------------------------
            for (uint32_t p = 0; p < panels; ++p) {
                const uint64_t jp = j0 + 16ull * p;
                const uint32_t rows = static_cast<uint32_t>(MinU64(16, rowLen_ - jp));
                LocalTensor<T> slab = slabQue_.template AllocTensor<T>();
                DataCopyExtParams in{static_cast<uint16_t>(rows),
                                     static_cast<uint32_t>(tileK_ * ds), loadGap, 0, 0};
                DataCopyPad(slab, xGm_[odoBase_ + jp * srcBlkPitch_], in, pad);
                slabQue_.EnQue(slab);
                slab = slabQue_.template DeQue<T>();

                LocalTensor<T> panel = panelQue_.template AllocTensor<T>();
                uint64_t srcL[16];
                uint64_t dstL[16];
                for (uint32_t r = 0; r < 16u; ++r) {
                    const uint32_t rr = (r < rows) ? r : (rows - 1u);
                    srcL[r] = reinterpret_cast<uint64_t>(
                        slab[static_cast<uint32_t>(rr) * kSnap].GetPhyAddr());
                }
                for (uint32_t r = 0; r < 8u; ++r) {
                    const uint64_t rowAddr =
                        reinterpret_cast<uint64_t>(panel[r * 16u].GetPhyAddr());
                    dstL[2u * r] = rowAddr;
                    dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                }
                TransDataTo5HD<T>(dstL, srcL, cnv);
                slabQue_.FreeTensor(slab);

                panelQue_.EnQue(panel);
                panel = panelQue_.template DeQue<T>();
                DataCopyExtParams outP{1, panelBytes, 0, 0, 0};
                DataCopyPad(wsGm_[static_cast<uint64_t>(p) * kSnap * 16u], panel, outP);
                panelQue_.FreeTensor(panel);
            }

            // RAW: stage-2 reads wait for stage-1's scratch stores.
            SetFlag<HardEvent::MTE3_MTE2>(evS1S2);
            WaitFlag<HardEvent::MTE3_MTE2>(evS1S2);

            // ---- stage 2: scratch panels -> output rows --------------------
            const uint64_t iBlocks = (tileK_ + Ib - 1) / Ib;
            for (uint64_t ib = 0; ib < iBlocks; ++ib) {
                uint64_t i0 = ib * Ib;
                if (i0 + Ib > tileK_) {
                    i0 = tileK_ - Ib;   // uniform-height tail: idempotent overlap
                }
                LocalTensor<T> rowsBuf = asmQue_.template AllocTensor<T>();
                for (uint32_t p = 0; p < panels; ++p) {
                    DataCopyExtParams inC{static_cast<uint16_t>(Ib), 16u * ds, 0,
                                          (asmPitch - 16u * ds) / DATA_BLOCK_BYTES, 0};
                    DataCopyPad(rowsBuf[p * 16u],
                                wsGm_[static_cast<uint64_t>(p) * kSnap * 16u + i0 * 16u],
                                inC, pad);
                }
                asmQue_.EnQue(rowsBuf);
                rowsBuf = asmQue_.template DeQue<T>();
                const uint32_t asmElemsPerRow = asmPitch / ds;
                for (uint32_t r = 0; r < Ib; ++r) {
                    DataCopyExtParams outR{1, jw * ds, 0, 0, 0};
                    DataCopyPad(yGm_[odoOutBase_ + (i0 + r) * dstRowStride_ + j0],
                                rowsBuf[r * asmElemsPerRow], outR);
                }
                asmQue_.FreeTensor(rowsBuf);
            }
        }
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(evS1S2);
        pipe->ReleaseEventID<HardEvent::MTE2_MTE3>(evS2S1);
    }

    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> slabQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> panelQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> asmQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> wsGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t tileKT_;
    uint32_t rowsPerBatch_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};

// ---------------------------------------------------------------------------
// key 11 (tile-cube): the cube-assisted two-stage wide transpose.
//
// The DMA engine itself can transpose the GRID: copy_gm_to_cbuf_multi_nd2nz_b32s
// (reached through DataCopy(L1, GM, Nd2NzParams)) reads an ND matrix with
// CONTIGUOUS row-major bursts and lands it in L1 as NZ -- 8-lane column
// panels, each panel n-major contiguous. That is exactly the panel structure
// key 10's stage 1 built by hand with slab loads and vnchwconv, minus the
// in-fractal (8x8) transpose, which stays with the AIV where vnchwconv is
// ~free. So the split is:
//
//   AIC (20 of them, MIX 1:2): per batch, per 192-row chunk: GM -> L1 with
//   nd2nz (hardware grid transpose, fat contiguous reads), then L1 -> GM
//   scratch as ONE contiguous ~400 KB store (copy_cbuf_to_gm). The whole of
//   stage 1 leaves the vector cores.
//
//   AIV (2 per AIC): waits the group flag, reads 4-panel blocks (~24 KB
//   CONTIGUOUS -- no L2 crutch needed, streams at gap-0 rates), vnchwconv
//   the 8x8 fractals, stores 32 CONSECUTIVE output rows with ONE descriptor
//   (blockCount 32, blockLen Nc*4, UB pitch == blockLen).
//
// Chunks are UNIFORM 192 rows via the left-shifted tail (idempotent overlap,
// as keys 9/10), so no ragged geometry exists anywhere except the last
// panel group (65th panel: 7 valid lanes; its pad lane row is simply not
// stored). Sync is two batch-parity scratch regions per group and mode-2
// (intra-group, the only AIC<->AIV routing GetffstMsg's `& 0x3` admits) flags: AIC sets one ready flag per AIV, each AIV
// sets its own free flag back, AIC waits both before overwriting a region.
// Flag ids 8..15; ALLOCATED hardware event ids are not involved.
// ---------------------------------------------------------------------------
constexpr uint16_t CUBE_FLAG_READY = 0;   // +parity*2+sub  -> 0..3
constexpr uint16_t CUBE_FLAG_PAIR = 8;    // +sub -> 8..9 (AIV<->AIV barrier)
// ids 10..15 stay clear: the framework reserves 11..14 (SYNC_* in
// kernel_operator_sync_impl.h) and 15 (KFC WORKSPACE_SYNC_ID).
// S4 static split: every third served batch is produced by the AIV pair
// itself (V-pipe 32 B block-interleave into the SAME NZ scratch layout, one
// fat store per slab), returning stage-1 to 40-core aggregate bandwidth.
__aicore__ inline bool CubeAivBatch(uint64_t served)
{
    return (served % 3ull) == 2ull;
}
// The host chooses chunk rows from plan.tileMT as a multiple of 16
// whose slot layout fits the L1 budget.
constexpr uint32_t CUBE_PG = 4;           // panels per AIV unit (32 output rows)

template <typename T>
class TransposeTileCubeKernel {
public:
    __aicore__ inline TransposeTileCubeKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 4) {
            RunImpl(x, y, ws, t, pipe);
        }
    }

private:
    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, GM_ADDR ws, PtrT t, TPipe* pipe)
    {
        numel_ = t->numel;
        rowCount_ = t->rowCount;      // batches
        rowLen_ = t->rowLen;          // bigM: input rows, output columns
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;    // GROUP count (AICs)
        tileK_ = t->tileK;            // bigK: input cols, output rows
        dstRowStride_ = t->dstRowStride;
        tileMT_ = t->tileMT;          // Nc: uniform chunk rows (host-solved)
        rowsPerBatch_ = t->rowsPerBatch;  // L1 slot count (1 or 2)
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);
        const uint32_t nc = tileMT_;
        const uint32_t slots = rowsPerBatch_;
        if (numel_ == 0 || nc == 0 || rowLen_ < nc || tileK_ == 0 || rowCount_ == 0 ||
            slots == 0) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t lanes = 8u;                       // C0 for b32
        const uint32_t panels = static_cast<uint32_t>((tileK_ + lanes - 1) / lanes);
        const uint32_t colsPad = panels * lanes;
        const uint32_t chunks =
            static_cast<uint32_t>((rowLen_ + nc - 1) / nc);
        const uint64_t chunkElems = static_cast<uint64_t>(nc) * colsPad;
        const uint64_t regionElems = static_cast<uint64_t>(chunks) * chunkElems;

        const uint64_t grpCount = usedCores_ == 0 ? 1 : usedCores_;
#if defined(ASCENDC_CPU_DEBUG)
        // The CPU-debug path does not execute hardware nd2nz or cross-core flags.
        const uint64_t grp = GetBlockIdx();
        const uint32_t sub = 0;
        if (grp >= grpCount) {
            return;
        }
        (void)ds;
        (void)colsPad;
        (void)sub;
#else
        // ASCEND_IS_AIV expands to a constexpr condition for if syntax;
        // do not use it as a ternary operand.
        uint64_t rat = 1;
        uint32_t sub = 0;
        if ASCEND_IS_AIV {
            rat = GetTaskRation();
            sub = static_cast<uint32_t>(GetSubBlockIdx());
        }
        const uint64_t grp = GetBlockIdx() / (rat == 0 ? 1 : rat);
        if (grp >= grpCount) {
            return;
        }
        const uint64_t perGroup = (rowCount_ + grpCount - 1) / grpCount;
        wsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(GetUserWorkspace(ws)) +
                grp * perGroup * regionElems,
            perGroup * regionElems);

        if ASCEND_IS_AIC {
            RunProducer(pipe, grp, grpCount, nc, slots, chunks, chunkElems,
                        regionElems);
        }
        if ASCEND_IS_AIV {
            RunConsumer(pipe, grp, grpCount, sub, nc, panels, chunks, chunkElems,
                        regionElems, ds);
        }
#endif
    }

    // ---- AIC: GM -> L1 (nd2nz) -> GM scratch, one batch per flag round ----
    __aicore__ inline void RunProducer(TPipe* pipe, uint64_t grp, uint64_t grpCount,
                                       uint32_t nc, uint32_t slots,
                                       uint32_t chunks, uint64_t chunkElems,
                                       uint64_t regionElems)
    {
#if !defined(ASCENDC_CPU_DEBUG)
        // TWO L1 chunk buffers: nd2nz(next) overlaps the cbuf->GM drain of
        // the previous chunk, so the producer's two DMA legs cost max()
        // rather than sum. Per-buffer event pairs; the WAR guards each
        // buffer against ITS OWN previous store, batch boundaries included.
        TBuf<TPosition::A1> l1Buf;
        pipe->InitBuffer(l1Buf,
                         static_cast<uint32_t>(static_cast<uint64_t>(slots) *
                                               chunkElems * sizeof(T)));
        LocalTensor<T> l1all = l1Buf.template Get<T>();
        event_t evWr[2];
        event_t evRd[2];
        evWr[0] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        evWr[1] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        evRd[0] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        evRd[1] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        uint32_t slotUsed = 0;   // bitmask: slot has an in-flight/старый store
        uint64_t chunkSeq = 0;

        uint64_t served = 0;
        for (uint64_t b = grp; b < rowCount_; b += grpCount, ++served) {
            if (CubeAivBatch(served)) {
                continue;   // the AIV pair produces this batch itself
            }
            const uint32_t parity = static_cast<uint32_t>(served & 1u);
            // Use a separate scratch region per batch so the producer does not wait
            // for a consumer before reusing GM storage.
            const uint64_t inBase = b * rowLen_ * srcBlkPitch_;
            const uint64_t regBase = served * regionElems;
            for (uint32_t c = 0; c < chunks; ++c) {
                uint64_t n0 = static_cast<uint64_t>(c) * nc;
                if (n0 + nc > rowLen_) {
                    n0 = rowLen_ - nc;   // uniform chunk: idempotent overlap
                }
                // Protect L1 write-after-read hazards across batch boundaries as well as
                // within a batch. Each slot waits for its own previous store.
                const uint32_t slot = static_cast<uint32_t>(chunkSeq % slots);
                ++chunkSeq;
                LocalTensor<T> l1 = l1all[static_cast<uint64_t>(slot) * chunkElems];
                if (slotUsed & (1u << slot)) {
                    WaitFlag<HardEvent::MTE3_MTE2>(evRd[slot]);
                }
                slotUsed |= (1u << slot);
                Nd2NzParams nz{1,
                               static_cast<uint16_t>(nc),
                               static_cast<uint16_t>(tileK_),
                               0,
                               static_cast<uint16_t>(srcBlkPitch_),
                               static_cast<uint16_t>(nc),
                               1,
                               0};
                DataCopy(l1, xGm_[inBase + n0 * srcBlkPitch_], nz);
                SetFlag<HardEvent::MTE2_MTE3>(evWr[slot]);
                WaitFlag<HardEvent::MTE2_MTE3>(evWr[slot]);
                DataCopyParams out{1,
                                   static_cast<uint16_t>(chunkElems * sizeof(T) /
                                                         DATA_BLOCK_BYTES),
                                   0, 0};
                DataCopy(wsGm_[regBase + static_cast<uint64_t>(c) * chunkElems], l1,
                         out);
                SetFlag<HardEvent::MTE3_MTE2>(evRd[slot]);
                // Signal readiness per chunk so AIV processing can overlap the next AIC
                // chunk. Producer and consumers use identical batch/chunk flag order: one
                // set and one wait per chunk per sub-block, with no state across launches.
                CrossCoreSetFlag<2, PIPE_MTE3>(
                    static_cast<uint16_t>(CUBE_FLAG_READY + parity * 2u));
                CrossCoreSetFlag<2, PIPE_MTE3>(
                    static_cast<uint16_t>(CUBE_FLAG_READY + parity * 2u + 1u));
            }
        }
        if (slotUsed & 1u) {
            WaitFlag<HardEvent::MTE3_MTE2>(evRd[0]);
        }
        if (slotUsed & 2u) {
            WaitFlag<HardEvent::MTE3_MTE2>(evRd[1]);
        }
        pipe->ReleaseEventID<HardEvent::MTE2_MTE3>(evWr[0]);
        pipe->ReleaseEventID<HardEvent::MTE2_MTE3>(evWr[1]);
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(evRd[0]);
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(evRd[1]);
#endif
    }

    // ---- AIV: scratch panels -> vnchwconv -> packed 32-row output stores ---
    __aicore__ inline void RunConsumer(TPipe* pipe, uint64_t grp, uint64_t grpCount,
                                       uint32_t sub, uint32_t nc, uint32_t panels,
                                       uint32_t chunks, uint64_t chunkElems,
                                       uint64_t regionElems, uint32_t ds)
    {
        const uint32_t inBlk = CUBE_PG * nc * 8u;               // elems per unit read
        const uint32_t outPitch = nc * ds;                      // mult of 32
        const uint32_t colsPadB = panels * 8u * ds;             // slab row bytes
        const uint32_t slabB = 16u * colsPadB;                  // S4 producer slab
        const uint32_t inQB = inBlk * ds > slabB ? inBlk * ds : slabB;
        const uint32_t outQB =
            CUBE_PG * 8u * outPitch > slabB ? CUBE_PG * 8u * outPitch : slabB;
        pipe->InitBuffer(cubeInQue_, BUFFER_NUM, inQB);
        pipe->InitBuffer(cubeOutQue_, BUFFER_NUM, outQB);
        colsPadB_ = colsPadB;

        const uint32_t pGroups = (panels + CUBE_PG - 1) / CUBE_PG;
        // In-fractal transpose uses 16 source row pointers with eight floats each.
        // Per repeat the source advances 16 rows (srcRepStride 16), and each
        // destination row gains 16 floats (dstRepStride 2). Repeat count is nc/16.
        TransDataTo5HDParams cnv{false, false, static_cast<uint8_t>(nc / 16u),
                                 2, 16};

        uint64_t served = 0;
        for (uint64_t b = grp; b < rowCount_; b += grpCount, ++served) {
            const uint32_t parity = static_cast<uint32_t>(served & 1u);
            const uint64_t regBase = served * regionElems;
            const uint64_t outBatch = b * tileK_ * dstRowStride_;
            const bool selfMade = CubeAivBatch(served);

            if (selfMade) {
                // S4: the pair produces this batch. Sub s relayouts chunks
                // c % 2 == s -- slab in (fat contiguous read), V-pipe 32 B
                // block-interleave into the NZ layout in UB, ONE fat store
                // per slab -- then a pair barrier so both consume all of it.
                const uint64_t inBase = b * rowLen_ * srcBlkPitch_;
                for (uint32_t c = sub; c < chunks; c += 2) {
                    uint64_t n0 = static_cast<uint64_t>(c) * nc;
                    if (n0 + nc > rowLen_) {
                        n0 = rowLen_ - nc;
                    }
                    ProduceChunkAiv(inBase, regBase + static_cast<uint64_t>(c) * chunkElems,
                                    n0, nc, panels, ds);
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(
                    static_cast<uint16_t>(CUBE_FLAG_PAIR + sub));
                CrossCoreWaitFlag<2, PIPE_MTE2>(
                    static_cast<uint16_t>(CUBE_FLAG_PAIR + (1u - sub)));
            }

            for (uint32_t c = 0; c < chunks; ++c) {
                if (!selfMade) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(
                    static_cast<uint16_t>(CUBE_FLAG_READY + parity * 2u + sub));
                }
                uint64_t n0 = static_cast<uint64_t>(c) * nc;
                if (n0 + nc > rowLen_) {
                    n0 = rowLen_ - nc;
                }
                for (uint32_t pg = sub; pg < pGroups; pg += 2) {
                const uint32_t p0 = pg * CUBE_PG;
                const uint32_t pN = (p0 + CUBE_PG <= panels) ? CUBE_PG : (panels - p0);

                LocalTensor<T> ubIn = cubeInQue_.template AllocTensor<T>();
                DataCopyExtParams in{1, pN * nc * 8u * ds, 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(ubIn,
                            wsGm_[regBase + static_cast<uint64_t>(c) * chunkElems +
                                  static_cast<uint64_t>(p0) * nc * 8u],
                            in, pad);
                cubeInQue_.EnQue(ubIn);
                ubIn = cubeInQue_.template DeQue<T>();

                LocalTensor<T> ubOut = cubeOutQue_.template AllocTensor<T>();
                for (uint32_t p = 0; p < pN; ++p) {
                    // [nc x 8] -> [8 x nc]: 16 n-rows per repeat.
                    uint64_t srcL[16];
                    uint64_t dstL[16];
                    for (uint32_t r = 0; r < 16u; ++r) {
                        srcL[r] = reinterpret_cast<uint64_t>(
                            ubIn[p * nc * 8u + r * 8u].GetPhyAddr());
                    }
                    for (uint32_t r = 0; r < 8u; ++r) {
                        const uint64_t rowAddr = reinterpret_cast<uint64_t>(
                            ubOut[(p * 8u + r) * nc].GetPhyAddr());
                        dstL[2u * r] = rowAddr;
                        dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                    }
                    TransDataTo5HD<T>(dstL, srcL, cnv);
                }
                cubeOutQue_.EnQue(ubOut);
                ubOut = cubeOutQue_.template DeQue<T>();

                // Output rows 8*p0 .. 8*p0+8*pN-1 are consecutive; the pad
                // lanes of the last panel (rows >= tileK_) are not stored.
                uint32_t nRows = 8u * pN;
                const uint64_t i0 = 8ull * p0;
                if (i0 + nRows > tileK_) {
                    nRows = static_cast<uint32_t>(tileK_ - i0);
                }
                DataCopyExtParams outP{static_cast<uint16_t>(nRows),
                                       nc * ds, 0,
                                       static_cast<uint32_t>(
                                           (dstRowStride_ - nc) * ds),
                                       0};
                DataCopyPad(yGm_[outBatch + i0 * dstRowStride_ + n0], ubOut, outP);
                cubeOutQue_.FreeTensor(ubOut);
                cubeInQue_.FreeTensor(ubIn);
                }
            }
        }
    }

    // S4 stage-1 on an AIV: one 16-row slab at a time -- fat contiguous
    // load, V-pipe interleave of 8-element pieces into NZ panel order
    // (panel p's 16x8 block = 512 B contiguous in staging), then ONE
    // panels-block store landing at the chunk's panel stride. The V pipe is
    // ~idle on this route (30-40 us of ~1150), which is exactly why the
    // interleave lives there and not on the already-loaded MTE3.
    __aicore__ inline void ProduceChunkAiv(uint64_t inBase, uint64_t chunkBase,
                                           uint64_t n0, uint32_t nc, uint32_t panels,
                                           uint32_t ds)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t slabs = nc / 16u;
        for (uint32_t sIdx = 0; sIdx < slabs; ++sIdx) {
            const uint64_t rowTop = n0 + 16ull * sIdx;
            LocalTensor<T> slab = cubeInQue_.template AllocTensor<T>();
            DataCopyExtParams in{16, static_cast<uint32_t>(tileK_ * ds),
                                 static_cast<uint32_t>((srcBlkPitch_ - tileK_) * ds),
                                 0, 0};
            DataCopyPad(slab, xGm_[inBase + rowTop * srcBlkPitch_], in, pad);
            cubeInQue_.EnQue(slab);
            slab = cubeInQue_.template DeQue<T>();

            LocalTensor<T> nz = cubeOutQue_.template AllocTensor<T>();
            DataCopyParams il{16, 1,
                              static_cast<uint16_t>((colsPadB_ - 8u * ds) /
                                                    DATA_BLOCK_BYTES),
                              0};
            for (uint32_t pI = 0; pI < panels; ++pI) {
                DataCopy(nz[static_cast<uint64_t>(pI) * 16u * 8u],
                         slab[static_cast<uint64_t>(pI) * 8u], il);
            }
            cubeOutQue_.EnQue(nz);
            nz = cubeOutQue_.template DeQue<T>();
            DataCopyExtParams st{static_cast<uint16_t>(panels), 16u * 8u * ds, 0,
                                 static_cast<uint32_t>((static_cast<uint64_t>(nc) -
                                                        16ull) * 8ull * ds),
                                 0};
            DataCopyPad(wsGm_[chunkBase + (rowTop - n0) * 8ull], nz, st);
            cubeOutQue_.FreeTensor(nz);
            cubeInQue_.FreeTensor(slab);
        }
    }

    TQue<QuePosition::VECIN, BUFFER_NUM> cubeInQue_;
    // VECOUT, not TQueBind: this buffer is WRITTEN BY V (vnchwconv), and a
    // TQueBind<VECIN,VECOUT> pairs its recycle events MTE2<->MTE3 only -- it
    // never fences V against the in-flight store, and at 9 units per AIV the
    // two-deep buffer came back around while MTE3 still read it (out rows of
    // pg0 carrying pg4's data, the [2,300,264] fingerprint). Same defect
    // class as key 10's FetchEventID collision: queue type must match the
    // pipes that actually touch the buffer.
    TQue<QuePosition::VECOUT, BUFFER_NUM> cubeOutQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> wsGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t rowsPerBatch_;
    uint32_t colsPadB_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
};

#include "transpose_output.h"
#include "transpose_rect.h"
#include "transpose_group.h"
#include "transpose_whole_suffix.h"

// TRANSPOSE_TWIN_BUILD suppresses the operator entry point so alternate
// launch wrappers can include the kernel classes.
#ifndef TRANSPOSE_TWIN_BUILD

extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
#if TP_CLOCK_TIMER && !defined(ASCENDC_CPU_DEBUG)
    const uint64_t diagStart=static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
    TPipe pipe;

    // Read needed fields directly through the GM TransposeTilingData pointer.
    // Using the shared type preserves the compiler-defined layout and avoids
    // copying the complete tiling structure to local storage.
    __gm__ TransposeTilingData* td = reinterpret_cast<__gm__ TransposeTilingData*>(tiling);

    // Initialize inside each TILING_KEY_IS branch so every compiled variant
    // writes only the state its route consumes.
    if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
        TransposeKernel<DTYPE_X> op;
        op.InitP<4>(x, y, td, &pipe);
        op.SetupRows();
        op.ProcessFold();
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        TransposeKernel<DTYPE_X> op;
        op.InitP<3>(x, y, td, &pipe);
        op.SetupTile();
        op.ProcessTile();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        TransposeKernel<DTYPE_X> op;
        op.InitP<2>(x, y, td, &pipe);
        op.SetupRows();
        op.ProcessRows();
    } else if (TILING_KEY_IS(1)) {
        // Use an explicit TILING_KEY_IS literal for each launchable key.
        // A default else branch does not generate a kernel variant for key 1.
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        TransposeKernel<DTYPE_X> op;
        op.InitP<1>(x, y, td, &pipe);
        op.SetupBulk();
        op.ProcessBulk();
    } else if (TILING_KEY_IS(5)) {
        // The one-shot route uses its separate straight-line class, with no
        // queue-based tile loop or batch odometer.
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_AIV_ONLY);
        TransposeOneKernel<DTYPE_X> one;
        one.Run(x, y, td, &pipe);
    } else if (TILING_KEY_IS(6)) {
        // The int8 tile route has its own class and half-intermediate UB layout.
        // The body is enabled only for the one-byte dtype; other dtype objects
        // retain an empty variant. The host selects this key only for int8.
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
        TransposeTileI8Kernel<DTYPE_X> i8;
        i8.Run(x, y, td, &pipe);
    } else if (TILING_KEY_IS(7)) {
        // The packed-store tile route uses its own class and UB layout.
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_AIV_ONLY);
        TransposeTilePackKernel<DTYPE_X> pk;
        pk.Run(x, y, td, &pipe);
    } else if (TILING_KEY_IS(8)) {
        // The packed-store FOLD route. Emitted LAST for the same reason key 7
        // was: the seven existing variants keep their order and .text offsets.
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_AIV_ONLY);
        TransposeFoldPackKernel<DTYPE_X> fp;
        fp.Run(x, y, td, &pipe);
    } else if (TILING_KEY_IS(9)) {
        // The full-K panel route uses its own class. The body is enabled only for
        // four-byte dtypes; the host selects the route only for those types.
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_AIV_ONLY);
        TransposeTileWideKernel<DTYPE_X> tw;
        tw.Run(x, y, td, &pipe);
    } else if (TILING_KEY_IS(10)) {
        // The two-stage wide route through the user workspace. Emitted LAST;
        // keys 1..9 keep their order and .text offsets. SetSysWorkspace is
        // SquareSumV1's one-line lesson: nothing on this route calls SyncAll,
        // but the pointer is set anyway so no future edit can reintroduce the
        // wild-GM poll by accident.
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIV_ONLY);
        AscendC::SetSysWorkspace(workspace);
        TransposeTileViaKernel<DTYPE_X> tv;
        tv.Run(x, y, workspace, td, &pipe);
    } else if (TILING_KEY_IS(12)) {
        // The L2-resident two-pass: every burst fat, every address ours.
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_AIV_ONLY);
        AscendC::SetSysWorkspace(workspace);
        TransposeTileL2Kernel<DTYPE_X> tl;
        tl.Run(x, y, workspace, td, &pipe);
    } else if (TILING_KEY_IS(17)) {
        KERNEL_TASK_TYPE(17,KERNEL_TYPE_AIV_ONLY);
        if(td->tilePrefetch==204){TransposeCompactSuffixKernel<DTYPE_X> tsw;tsw.Run(x,y,td,&pipe);}else{TransposeWholeSuffixKernel<DTYPE_X> tsw;tsw.Run(x,y,td,&pipe);}
    } else if (TILING_KEY_IS(15)) {
        KERNEL_TASK_TYPE(15,KERNEL_TYPE_AIV_ONLY);
        if(td->tilePrefetch==205){TransposeLongGroupKernel<DTYPE_X> tg;tg.Run(x,y,td,&pipe);}else{TransposeGroupKernel<DTYPE_X> tg;tg.Run(x,y,td,&pipe);}
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16,KERNEL_TYPE_AIV_ONLY);
        if(td->tilePrefetch==197 || td->tilePrefetch==198){AscendC::SetSysWorkspace(workspace);TransposeDenseBlockedL2Kernel<DTYPE_X> ts;ts.Run(x,y,workspace,td,&pipe);}else if(td->foldExt==174 && td->tilePrefetch==174){TransposePackedDenseKernel<DTYPE_X> ts;ts.Run(x,y,td,&pipe);}else if(td->foldExt==173 && td->tilePrefetch==10){TransposeAlignedWideKernel<DTYPE_X> ts;ts.Run(x,y,td,&pipe);}else{TransposeSlabKernel<DTYPE_X> ts;ts.Run(x,y,td,&pipe);}
    } else if (TILING_KEY_IS(14)) {
        KERNEL_TASK_TYPE(14,KERNEL_TYPE_AIV_ONLY);
        TransposeRectKernel<DTYPE_X> tr;
        tr.Run(x,y,td,&pipe);
    } else if (TILING_KEY_IS(13)) {
        KERNEL_TASK_TYPE(13,KERNEL_TYPE_AIV_ONLY);
        TransposeOutputKernel<DTYPE_X> to;
        to.Run(x,y,td,&pipe);
    } else if (TILING_KEY_IS(11)) {
        // The cube-assisted two-stage route: the repo's first MIX variant.
        // 1 AIC produces (nd2nz relayout to scratch), its 2 AIVs consume.
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
        AscendC::SetSysWorkspace(workspace);
        TransposeTileCubeKernel<DTYPE_X> tc;
        tc.Run(x, y, workspace, td, &pipe);
    }

    // Optional diagnostic delay after output computation. Only block 0 runs
    // the dependent arithmetic chain; zero diagSpin disables it. CPU-debug
    // builds exclude the trailing diagnostic field and this block.
#if !defined(ASCENDC_CPU_DEBUG)
    if (AscendC::GetBlockIdx() == 0) {
        const uint64_t ds = td->diagSpin;
#if TP_CLOCK_TIMER
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t deadline=(10000ULL+100ULL*ds)*50ULL;
        while(static_cast<uint64_t>(AscendC::GetSystemCycle())-diagStart<deadline) {}
#else
        if (ds != 0) {
            uint64_t v = ds | 1ULL;
            for (uint64_t i = 0; i < ds; ++i) {
                v = v * 6364136223846793005ULL + 1442695040888963407ULL;
            }
            if (v == 0x5A5AA5A5DEADBEEFULL) {
                *reinterpret_cast<__gm__ uint32_t*>(y) = 0U;
            }
        }
#endif
    }
#endif
}

#endif  // TRANSPOSE_TWIN_BUILD
