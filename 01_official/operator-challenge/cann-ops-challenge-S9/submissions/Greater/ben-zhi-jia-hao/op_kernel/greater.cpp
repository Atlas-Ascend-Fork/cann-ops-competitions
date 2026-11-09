/*
 * Greater computes elementwise comparisons with broadcast support.
 * The kernels use dtype-appropriate comparisons, mask selection and conversion
 * to Boolean output. Broadcast rows may use padded batches or cached operands.
 */
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t GT_MAX_DIM = 8;
// Queue depth is bounded at compile time; Init selects the required slot count.
constexpr uint32_t BUF_NUM = 2;
constexpr uint32_t TILE_GRAN = 256;    // tile length granularity, in elements
constexpr uint32_t UB_RESERVE = 8192;  // slack left to TPipe bookkeeping and buffer overhang
constexpr uint32_t MASK_GRAN = 256;    // mask buffer granularity, in bytes

constexpr uint32_t FAM_FLAT = 0;
constexpr uint32_t FAM_BCAST = 1;
// The one-tile route uses a separate kernel with a fixed UB layout.
constexpr uint32_t FAM_ONE = 2;
// The one-tile allocation is bounded by GT_ONE_TILE. The host checks tile
// and UB capacity before selecting this route.
constexpr uint32_t GT_ONE_TILE = 2048;

// ---- row batching (see ProcessRowBatch) --------------------------------------
// One UB datablock, in bytes. Every stride expressed to DataCopyPad on the UB
// side and to Copy as a repeat stride is counted in these.
constexpr uint32_t BLK_BYTES = 32;
// One full vector repeat, in bytes. Copy addresses at most this much per repeat,
// so filling a wider row slot takes one Copy per 256 bytes of it.
constexpr uint32_t REPEAT_BYTES = 256;
// Row pitch granularity, in ELEMENTS. The bool output is one byte per element,
// so a pitch of 32 elements is exactly one datablock of output - the smallest
// pitch for which the UB->GM store can express the row gap at all.
constexpr uint32_t ROW_PITCH_GRAN = 32;
// Both Copy and Brcb take repeatTimes as a uint8.
constexpr uint32_t MAX_REPEAT = 255;

constexpr uint32_t RB_ROWS = 0;   // operand is dense on the inner axis and advances per row
constexpr uint32_t RB_HOLD = 1;   // operand is invariant across the batched rows
constexpr uint32_t RB_SPLAT = 2;  // operand is one scalar per row (broadcast on the inner axis)


// Bit-exact integer view used for the strided Copy. Copy and Brcb only accept
// b16 / b32, and going through an integer view keeps NaN payloads, signed zeros
// and denormals untouched - an arithmetic "copy" such as Adds(x, 0) would not.
template <uint32_t N>
struct GtView;
template <>
struct GtView<4> {
    using type = int32_t;
};
template <>
struct GtView<2> {
    using type = int16_t;
};
template <>
struct GtView<1> {
    using type = int16_t;
};

template <typename T>
struct GtTraits;
template <>
struct GtTraits<float> {
    using CT = float;
    static constexpr bool kNeedCast = false;
    static constexpr bool kIntPath = false;
};
template <>
struct GtTraits<half> {
    using CT = half;
    static constexpr bool kNeedCast = false;
    static constexpr bool kIntPath = false;
};
template <>
struct GtTraits<bfloat16_t> {
    using CT = float;
    static constexpr bool kNeedCast = true;
    static constexpr bool kIntPath = false;
};
template <>
struct GtTraits<int8_t> {
    using CT = half;
    static constexpr bool kNeedCast = true;
    static constexpr bool kIntPath = false;
};
template <>
struct GtTraits<int32_t> {
    using CT = int32_t;
    static constexpr bool kNeedCast = false;
    static constexpr bool kIntPath = true;
};

__aicore__ inline uint64_t CeilDivU(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t RoundUpU32(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

}  // namespace

// The row-batch state is empty on flat routes. SetupRowBatch initializes the
// valid flag first and assigns other fields before they are used.
template <uint32_t FAM>
struct GtRowBatch {
    uint64_t rowStride1;  // stride of the innermost OUTER axis, per operand
    uint64_t rowStride2;
    uint32_t pitch;       // UB elements per row slot = RoundUp(innerLen, 32)
    uint32_t pitchBlk;    // that pitch in 32-byte datablocks, operand side
    uint32_t gapBlk;      // datablocks of padding after each loaded row
    uint32_t vtPerRow;    // integer-view elements in one row slot
    uint32_t repeats;     // Copy calls needed to fill one row slot
    uint32_t maxRows;     // rows one tile can hold
    uint32_t mode1;
    uint32_t mode2;
    bool on;
    bool keepHold;
};
template <>
struct GtRowBatch<FAM_FLAT> {};

// Broadcast state stores integer views of the input bits for scalar broadcasts.
// Flat routes do not use these fields.
template <uint32_t FAM>
struct GtBcast {
    TBuf<QuePosition::VECCALC> bufX, bufY;
    GlobalTensor<int32_t> rawX32, rawY32;
    GlobalTensor<uint16_t> rawX16, rawY16;
    uint64_t stride1[GT_MAX_DIM];
    uint64_t stride2[GT_MAX_DIM];
    uint32_t shape[GT_MAX_DIM];
    uint64_t innerLen = 1;
    uint32_t dimNum = 1;
    uint32_t innerMode = 0;
    uint64_t off1 = ~0ULL;  // operand-1 offset currently resident in bufX (sentinel = none)
    uint64_t off2 = ~0ULL;
    bool cacheX = false;
    bool cacheY = false;
};
template <>
struct GtBcast<FAM_FLAT> {};

// GT_DIRECT_TILING selects direct metadata reads; disabling it uses GET_TILING_DATA.
#ifndef GT_TILING_GM
#define GT_TILING_GM 1
#endif

#if GT_TILING_GM
using GtTiling = const __gm__ GreaterTilingData *;
#else
using GtTiling = const GreaterTilingData *;
#endif

template <typename T, uint32_t FAM>
class KernelGreater {
public:
    using CT = typename GtTraits<T>::CT;
    using VT = typename GtView<sizeof(T)>::type;
    static constexpr uint32_t S = sizeof(T);
    static constexpr uint32_t CS = sizeof(CT);
    static constexpr bool NEED_CAST = GtTraits<T>::kNeedCast;
    static constexpr bool INT_PATH = GtTraits<T>::kIntPath;
    // Integer-view elements a single vector repeat can address: 64 for b32, 128 for b16.
    static constexpr uint32_t VT_PER_REPEAT = REPEAT_BYTES / sizeof(VT);

    __aicore__ inline KernelGreater() {}

    // Derive tile capacity from the platform UB size and the allocation cost.
    static __aicore__ inline constexpr uint32_t UbPerElem()
    {
        uint32_t b = 0;
        b += BUF_NUM * S;  // x1 queue
        b += BUF_NUM * S;  // x2 queue
        b += BUF_NUM * 1;  // bool output queue
        b += 2;            // ones (half)
        b += 2;            // select result (half)
        b += 1;            // compare mask (needs 1/8, rounded up generously)
        if (NEED_CAST) {
            b += 2 * CS;
        }
        if (INT_PATH) {
            b += CS;
        }
        if (FAM == FAM_BCAST) {
            b += 2 * S;  // resident broadcast buffers for x1 and x2 (plus a fixed overhang)
        }
        return b;
    }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GtTiling t, TPipe *pipe)
    {
        // Read the required tiling metadata and bind the input/output buffers together
        // before submitting independent input transfers.
        const uint64_t total = t->totalNum;
        const uint64_t perCore = t->perCore;
        const uint32_t ubSize = t->ubSize;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
        if constexpr(FAM==FAM_FLAT) {
            alignedCopy_=(t->pad0 & 1024u)!=0;
            prefetch_=(t->pad0 & 16384u)!=0;
            roundRobin_=(t->pad0 & 131072u)!=0;
            totalFlat_=total; flatCores_=t->usedCores;
            if(t->pad0 & 2048u) yGm_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
        }
        if(t->pad0 & 8u) {
            if constexpr(FAM==FAM_FLAT) {
                x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            } else {
                if(t->strides[2]!=0) x1Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                else x2Gm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        if constexpr (FAM == FAM_BCAST) {
            // Bind the broadcast integer views before they are used.
            if constexpr (S == 4) {
                bc_.rawX32.SetGlobalBuffer((__gm__ int32_t *)x1);
                bc_.rawY32.SetGlobalBuffer((__gm__ int32_t *)x2);
            } else if constexpr (S == 2) {
                bc_.rawX16.SetGlobalBuffer((__gm__ uint16_t *)x1);
                bc_.rawY16.SetGlobalBuffer((__gm__ uint16_t *)x2);
            }
        }
        valid_ = false;
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * perCore;
        if (total == 0 || start >= total) {
            return;
        }
        valid_ = true;
        myStart_ = start;
        myLen_ = (total - start < perCore) ? (total - start) : perCore;

        // Read dimension and stride arrays only for broadcast routes.
        if constexpr (FAM == FAM_BCAST) {
            bc_.innerLen = t->innerLen;
            bc_.dimNum = t->dimNum;
            bc_.innerMode = t->innerMode;
            for (uint32_t i = 0; i < bc_.dimNum; ++i) {
                bc_.shape[i] = t->shape[i];
                bc_.stride1[i] = t->strides[2 * i];
                bc_.stride2[i] = t->strides[2 * i + 1];
            }
        }

        // ---- tile length ----
        uint32_t avail = (ubSize > UB_RESERVE) ? (ubSize - UB_RESERVE) : 0;
        uint32_t tl = avail / UbPerElem();
        tl = tl / TILE_GRAN * TILE_GRAN;
        if constexpr(FAM==FAM_FLAT) {
            if(t->pad0 & 32u) {
                const uint32_t slots=(t->pad0 & 128u)?1:BUF_NUM;
                const uint32_t bytes=UbPerElem()-1-(BUF_NUM-slots)*(2*S+1);
                const uint32_t grain=(t->pad0 & 256u)?512:TILE_GRAN;
                tl=(avail/bytes)/grain*grain;
                while(tl && bytes*tl+RoundUpU32(tl/8,MASK_GRAN)>avail) tl-=grain;
            }
        }
        if (tl == 0) {
            tl = TILE_GRAN;
        }
        uint64_t need = myLen_;
        if constexpr(FAM==FAM_FLAT) { if(roundRobin_) need=total; }
        if constexpr (FAM == FAM_BCAST) {
            if (bc_.innerLen > need) {
                need = bc_.innerLen;
            }
            // Account for the padded row pitch when bounding the tile allocation.
            rb_.pitch = RoundUpU32(static_cast<uint32_t>(bc_.innerLen), ROW_PITCH_GRAN);
            if (rb_.pitch > bc_.innerLen) {
                const uint64_t padded = CeilDivU(myLen_, bc_.innerLen) * rb_.pitch;
                if (padded > need) {
                    need = padded;
                }
            }
        }
        const uint64_t needAligned = CeilDivU(need, TILE_GRAN) * TILE_GRAN;
        if (needAligned < static_cast<uint64_t>(tl)) {
            tl = static_cast<uint32_t>(needAligned);
        }
        tileLen_ = tl;
        maskBytes_ = RoundUpU32(tileLen_ / 8, MASK_GRAN);

        // Use one queue slot when the complete core share fits one tile. Broadcast
        // work must also fit the current row; BUF_NUM remains the compile-time bound.
        bool oneRound = (myLen_ <= static_cast<uint64_t>(tileLen_));
        if constexpr(FAM==FAM_FLAT) { if(roundRobin_) oneRound=false; }
        if constexpr (FAM == FAM_BCAST) {
            const uint64_t pos0 = (bc_.innerLen != 0) ? (myStart_ % bc_.innerLen) : 0;
            oneRound = oneRound && (myLen_ + pos0 <= bc_.innerLen);
        }
        const uint8_t bufNum = (oneRound || (FAM==FAM_FLAT && (t->pad0 & 128u))) ? 1 : static_cast<uint8_t>(BUF_NUM);

        pipe->InitBuffer(qX_, bufNum, tileLen_ * S);
        pipe->InitBuffer(qY_, bufNum, tileLen_ * S);
        pipe->InitBuffer(qO_, bufNum, tileLen_ * 1);
        // On flat routes, allocate vector scratch during the first iteration after
        // CopyIn. TBuf does not allocate queue events; retain the queue allocation
        // order. Broadcast routes initialize their scratch in Init.
        pipe_ = pipe;
        // Use one arena for disjoint aligned vector-only scratch regions.
        // Keep input and output DMA buffers separate.
        if constexpr (FAM == FAM_BCAST) {
            pipe->InitBuffer(bufArena_, ArenaBytes());
            arena_ = bufArena_.template Get<uint8_t>();
            // One TILE_GRAN of overhang: a chunk that lands on a row tail is still
            // rounded up to a whole vector repeat, so the compute reads up to
            // TILE_GRAN elements past the end of the cached row block.
            pipe->InitBuffer(bc_.bufX, (tileLen_ + TILE_GRAN) * S);
            pipe->InitBuffer(bc_.bufY, (tileLen_ + TILE_GRAN) * S);
        }

        if constexpr (FAM == FAM_BCAST) {
            onesT_ = Slice<half>(0);
            Duplicate(onesT_, (half)1.0, tileLen_);

            // An operand whose inner block repeats across consecutive rows is held
            // resident in UB and reloaded only when its offset actually changes.
            // That is the difference between one DMA and shape[D-2] small ones.
            const uint32_t cacheMode = t->cacheMode;
            bc_.cacheX = ((cacheMode & 1u) != 0) && (bc_.innerLen <= tileLen_);
            bc_.cacheY = ((cacheMode & 2u) != 0) && (bc_.innerLen <= tileLen_);
            SetupRowBatch((t->pad0 & 16u)!=0,(t->pad0 & 32u)!=0);
        }
    }

    // Batch rows with an output pitch rounded up to 32 elements so Boolean output
    // row gaps are 32-byte aligned. Load the row batch, compare, then store it.
    __aicore__ inline void SetupRowBatch(bool force,bool keep)
    {
        rb_.on = false;
        rb_.keepHold = keep;
        if (bc_.dimNum < 2 || bc_.innerLen == 0) {
            return;
        }
        rb_.pitchBlk = rb_.pitch * S / BLK_BYTES;
        rb_.gapBlk = rb_.pitchBlk - RoundUpU32(static_cast<uint32_t>(bc_.innerLen) * S, BLK_BYTES) / BLK_BYTES;
        rb_.maxRows = tileLen_ / rb_.pitch;
        if (rb_.maxRows > MAX_REPEAT) {
            rb_.maxRows = MAX_REPEAT;
        }
        if (rb_.maxRows < 2) {
            return;
        }
        // Use segment copies or Brcb expansion according to the row batch and segment
        // counts. Brcb works on 256-byte units.
        rb_.vtPerRow = rb_.pitch * S / static_cast<uint32_t>(sizeof(VT));
        rb_.repeats = (rb_.vtPerRow + VT_PER_REPEAT - 1) / VT_PER_REPEAT;
        if (!force && rb_.repeats > rb_.maxRows) {
            return;
        }

        const uint32_t d = bc_.dimNum;
        rb_.rowStride1 = bc_.stride1[d - 2];
        rb_.rowStride2 = bc_.stride2[d - 2];
        rb_.mode1 = (bc_.innerMode == 1) ? RB_SPLAT : ((rb_.rowStride1 == 0) ? RB_HOLD : RB_ROWS);
        rb_.mode2 = (bc_.innerMode == 2) ? RB_SPLAT : ((rb_.rowStride2 == 0) ? RB_HOLD : RB_ROWS);
        // The collapse guarantees these, but the descriptors would be silently
        // wrong rather than merely slow if they ever failed, so they are checked
        // rather than assumed. RB_SPLAT: the inner axis is broadcast for this
        // operand, so the running stride is still 1 when the next axis out is
        // assigned, giving one contiguous scalar per row. RB_ROWS: the inner axis
        // is dense, so the next axis out has stride exactly innerLen.
        if ((rb_.mode1 == RB_SPLAT && rb_.rowStride1 != 1) || (rb_.mode1 == RB_ROWS && rb_.rowStride1 < bc_.innerLen) ||
            (rb_.mode2 == RB_SPLAT && rb_.rowStride2 != 1) || (rb_.mode2 == RB_ROWS && rb_.rowStride2 < bc_.innerLen)) {
            return;
        }
        rb_.on = true;
        // In batched mode nothing stays resident across tiles, so bc_.bufX/bc_.bufY
        // stop being DataCopy endpoints entirely: they are written by Copy or
        // Duplicate (vector) and read by the compare (vector). Turning the resident
        // block off is what removes the last cross-pipe hazard from this route.
        bc_.cacheX = false;
        bc_.cacheY = false;
    }

    // Map the scratch arena to aligned regions for constants, selection values,
    // comparison masks and optional cast buffers.
    __aicore__ inline uint32_t OffSel() { return tileLen_ * 2; }
    __aicore__ inline uint32_t OffMask() { return tileLen_ * 4; }
    __aicore__ inline uint32_t OffCA() { return tileLen_ * 4 + maskBytes_; }
    __aicore__ inline uint32_t OffCB() { return OffCA() + tileLen_ * CS; }
    __aicore__ inline uint32_t ArenaBytes()
    {
        uint32_t b = OffCA();
        if constexpr (NEED_CAST) {
            b += 2 * tileLen_ * CS;
        } else if constexpr (INT_PATH) {
            b += tileLen_ * CS;
        }
        return b;
    }
    // Address arithmetic only: operator[] shifts the tensor's start, and
    // ReinterpretCast changes its element type. Neither touches memory, which is
    // the whole point of not calling TBuf::Get per region.
    template <typename U>
    __aicore__ inline LocalTensor<U> Slice(uint32_t byteOff)
    {
        return arena_[byteOff].template ReinterpretCast<U>();
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        if constexpr (FAM == FAM_FLAT) {
            ProcessFlat();
        } else {
            ProcessBcast();
        }
    }

private:
    // ---------------- dense elementwise, no broadcast ----------------
    __aicore__ inline void ProcessFlat()
    {
        if(roundRobin_) { ProcessFlatRoundRobin(); return; }
        if(prefetch_) { ProcessFlatPrefetch(); return; }
        uint64_t off = 0;
        while (off < myLen_) {
            const uint64_t left = myLen_ - off;
            const uint32_t n = static_cast<uint32_t>(left < tileLen_ ? left : tileLen_);
            const uint32_t na = RoundUpU32(n, TILE_GRAN);
            const uint64_t g = myStart_ + off;

            CopyIn(qX_, x1Gm_, g, n);
            CopyIn(qY_, x2Gm_, g, n);
            if (off == 0) {
                // Initialize vector scratch after submitting the input descriptors.
                pipe_->InitBuffer(bufArena_, ArenaBytes());
                arena_ = bufArena_.template Get<uint8_t>();
                onesT_ = Slice<half>(0);
                Duplicate(onesT_, (half)1.0, tileLen_);
            }
            LocalTensor<T> a = qX_.template DeQue<T>();
            LocalTensor<T> b = qY_.template DeQue<T>();
            Compute(a, b, na);
            qX_.FreeTensor(a);
            qY_.FreeTensor(b);
            CopyOut(g, n);
            off += n;
        }
    }

    __aicore__ inline void ProcessFlatPrefetch()
    {
        const uint32_t first=static_cast<uint32_t>(myLen_<tileLen_?myLen_:tileLen_);
        CopyIn(qX_,x1Gm_,myStart_,first);
        CopyIn(qY_,x2Gm_,myStart_,first);
        pipe_->InitBuffer(bufArena_,ArenaBytes());
        arena_=bufArena_.template Get<uint8_t>();
        onesT_=Slice<half>(0);
        Duplicate(onesT_,(half)1,tileLen_);
        for(uint64_t off=0;off<myLen_;) {
            const uint32_t n=static_cast<uint32_t>(myLen_-off<tileLen_?myLen_-off:tileLen_);
            const uint64_t next=off+n;
            if(next<myLen_) {
                const uint32_t nn=static_cast<uint32_t>(myLen_-next<tileLen_?myLen_-next:tileLen_);
                CopyIn(qX_,x1Gm_,myStart_+next,nn);
                CopyIn(qY_,x2Gm_,myStart_+next,nn);
            }
            auto a=qX_.template DeQue<T>();
            auto b=qY_.template DeQue<T>();
            Compute(a,b,RoundUpU32(n,TILE_GRAN));
            qX_.FreeTensor(a); qY_.FreeTensor(b);
            CopyOut(myStart_+off,n);
            off=next;
        }
    }

    __aicore__ inline void ProcessFlatRoundRobin()
    {
        uint64_t g=static_cast<uint64_t>(GetBlockIdx())*tileLen_;
        if(g>=totalFlat_) return;
        const uint64_t stride=static_cast<uint64_t>(flatCores_)*tileLen_;
        const uint32_t first=static_cast<uint32_t>(totalFlat_-g<tileLen_?totalFlat_-g:tileLen_);
        CopyIn(qX_,x1Gm_,g,first); CopyIn(qY_,x2Gm_,g,first);
        pipe_->InitBuffer(bufArena_,ArenaBytes());
        arena_=bufArena_.template Get<uint8_t>(); onesT_=Slice<half>(0);
        Duplicate(onesT_,(half)1,tileLen_);
        bool ready=true;
        while(g<totalFlat_) {
            const uint32_t n=static_cast<uint32_t>(totalFlat_-g<tileLen_?totalFlat_-g:tileLen_);
            if(!ready) { CopyIn(qX_,x1Gm_,g,n); CopyIn(qY_,x2Gm_,g,n); }
            const uint64_t next=g+stride;
            ready=prefetch_ && next<totalFlat_;
            if(ready) {
                const uint32_t nn=static_cast<uint32_t>(totalFlat_-next<tileLen_?totalFlat_-next:tileLen_);
                CopyIn(qX_,x1Gm_,next,nn); CopyIn(qY_,x2Gm_,next,nn);
            }
            auto a=qX_.template DeQue<T>(); auto b=qY_.template DeQue<T>();
            Compute(a,b,RoundUpU32(n,TILE_GRAN));
            qX_.FreeTensor(a); qY_.FreeTensor(b); CopyOut(g,n);
            g=next;
        }
    }

    // ---------------- broadcast: odometer over the collapsed outer dims ----------------
    __aicore__ inline void ProcessBcast()
    {
        uint32_t idx[GT_MAX_DIM];
        const uint32_t d = bc_.dimNum;

        uint64_t row = myStart_ / bc_.innerLen;
        uint32_t pos = static_cast<uint32_t>(myStart_ - row * bc_.innerLen);

        uint64_t rem = row;
        for (int i = static_cast<int>(d) - 2; i >= 0; --i) {
            const uint32_t e = bc_.shape[i];
            idx[i] = static_cast<uint32_t>(rem % e);
            rem /= e;
        }
        uint64_t off1 = 0;
        uint64_t off2 = 0;
        for (uint32_t i = 0; i + 1 < d; ++i) {
            off1 += static_cast<uint64_t>(idx[i]) * bc_.stride1[i];
            off2 += static_cast<uint64_t>(idx[i]) * bc_.stride2[i];
        }

        bool newRow = true;
        uint64_t remaining = myLen_;
        uint64_t g = myStart_;
        while (remaining > 0) {
            // ---- batched: many whole rows in one iteration ----
            // Entered only on a row boundary with at least one whole row left, so
            // a core that starts or ends mid-row still gets that partial row from
            // the general path below.
            if (rb_.on && pos == 0 && remaining >= bc_.innerLen) {
                uint32_t k = rb_.maxRows;
                const uint32_t inAxis = bc_.shape[d - 2] - idx[d - 2];  // no odometer carry inside a batch
                if (inAxis < k) {
                    k = inAxis;
                }
                const uint64_t whole = remaining / bc_.innerLen;
                if (whole < static_cast<uint64_t>(k)) {
                    k = static_cast<uint32_t>(whole);
                }
                ProcessRowBatch(k, off1, off2);
                CopyOutRows(g, k);
                const uint64_t done = static_cast<uint64_t>(k) * bc_.innerLen;
                g += done;
                remaining -= done;
                StepOdometer(idx, off1, off2, d, k);
                newRow = true;
                continue;
            }
            if (newRow) {
                // A broadcast inner axis makes the operand a single value for the
                // whole row: splat it once instead of re-reading it per chunk.
                if (bc_.innerMode == 1) {
                    FillScalarX(off1);
                } else if (bc_.innerMode == 2) {
                    FillScalarY(off2);
                }
                if (bc_.cacheX && off1 != bc_.off1) {
                    LoadRowBlock(bc_.bufX.template Get<T>(), x1Gm_, off1);
                    bc_.off1 = off1;
                }
                if (bc_.cacheY && off2 != bc_.off2) {
                    LoadRowBlock(bc_.bufY.template Get<T>(), x2Gm_, off2);
                    bc_.off2 = off2;
                }
                newRow = false;
            }
            const uint64_t avail = bc_.innerLen - pos;
            uint64_t take = (avail < remaining) ? avail : remaining;
            if (take > tileLen_) {
                take = tileLen_;
            }
            const uint32_t n = static_cast<uint32_t>(take);
            const uint32_t na = RoundUpU32(n, TILE_GRAN);

            // Vector ops need a 32-byte aligned UB offset. Only the very first
            // chunk of a core can start mid-row (innerLen <= tileLen means a chunk
            // always finishes its row), so at most one chunk per core falls back
            // to a direct read.
            const bool posAligned = ((pos & 31u) == 0u);
            const bool xFromQ = (bc_.innerMode != 1) && !(bc_.cacheX && posAligned);
            const bool yFromQ = (bc_.innerMode != 2) && !(bc_.cacheY && posAligned);
            if (xFromQ) {
                CopyIn(qX_, x1Gm_, off1 + pos, n);
            }
            if (yFromQ) {
                CopyIn(qY_, x2Gm_, off2 + pos, n);
            }

            LocalTensor<T> a;
            LocalTensor<T> b;
            if (xFromQ) {
                a = qX_.template DeQue<T>();
            } else if (bc_.innerMode == 1) {
                a = bc_.bufX.template Get<T>();
            } else {
                a = bc_.bufX.template Get<T>()[pos];
            }
            if (yFromQ) {
                b = qY_.template DeQue<T>();
            } else if (bc_.innerMode == 2) {
                b = bc_.bufY.template Get<T>();
            } else {
                b = bc_.bufY.template Get<T>()[pos];
            }

            Compute(a, b, na);
            if (xFromQ) {
                qX_.FreeTensor(a);
            }
            if (yFromQ) {
                qY_.FreeTensor(b);
            }
            CopyOut(g, n);

            g += n;
            remaining -= n;
            pos += n;
            if (pos == bc_.innerLen) {
                pos = 0;
                StepOdometer(idx, off1, off2, d, 1);
                newRow = true;
            }
        }
    }

    // Advance the odometer by k rows. k is always clamped to what is left in the
    // innermost outer axis, so only that axis can be stepped by more than one and
    // the carry above it is still the ordinary +1 walk: no division, no modulo.
    __aicore__ inline void StepOdometer(uint32_t *idx, uint64_t &off1, uint64_t &off2, uint32_t d, uint32_t k)
    {
        if (d < 2) {
            return;  // a single collapsed axis has no rows to step
        }
        int i = static_cast<int>(d) - 2;
        idx[i] += k;
        off1 += static_cast<uint64_t>(k) * bc_.stride1[i];
        off2 += static_cast<uint64_t>(k) * bc_.stride2[i];
        while (i >= 0 && idx[i] >= bc_.shape[i]) {
            off1 -= static_cast<uint64_t>(idx[i]) * bc_.stride1[i];
            off2 -= static_cast<uint64_t>(idx[i]) * bc_.stride2[i];
            idx[i] = 0;
            --i;
            if (i < 0) {
                break;
            }
            idx[i] += 1;
            off1 += bc_.stride1[i];
            off2 += bc_.stride2[i];
        }
    }

    // ---------------- one batched tile: k whole rows at rb_.pitch -------------
    __aicore__ inline void ProcessRowBatch(uint32_t k, uint64_t off1, uint64_t off2)
    {
        bool aFromQ = false;
        bool bFromQ = false;
        LocalTensor<T> a = PrepareBatch(qX_, x1Gm_, bc_.bufX, rb_.mode1, rb_.rowStride1, off1, k, aFromQ,bc_.off1);
        LocalTensor<T> b = PrepareBatch(qY_, x2Gm_, bc_.bufY, rb_.mode2, rb_.rowStride2, off2, k, bFromQ,bc_.off2);
        Compute(a, b, RoundUpU32(k * rb_.pitch, TILE_GRAN));
        if (aFromQ) {
            qX_.FreeTensor(a);
        }
        if (bFromQ) {
            qY_.FreeTensor(b);
        }
    }

    // Bring one operand into the padded row layout and return the tensor to
    // compare against. Everything a DMA writes goes through a TQue, so every
    // MTE2->V edge in this route is framework-generated; the two raw broadcast
    // buffers are only ever written by the vector pipe here.
    __aicore__ inline LocalTensor<T> PrepareBatch(TQue<QuePosition::VECIN, BUF_NUM> &q, const GlobalTensor<T> &gm,
                                                  TBuf<QuePosition::VECCALC> &buf, uint32_t mode, uint64_t rowStride,
                                                  uint64_t off, uint32_t k, bool &fromQ,uint64_t &heldOff)
    {
        if (mode == RB_ROWS) {
            LocalTensor<T> l = q.template AllocTensor<T>();
            CopyRowsPitched(l, gm, off, k, rowStride);
            q.EnQue(l);
            fromQ = true;
            return q.template DeQue<T>();
        }
        fromQ = false;
        LocalTensor<T> dst = buf.template Get<T>();
        if(mode==RB_HOLD && rb_.keepHold && heldOff==off) return dst;
        if constexpr (S == 1) {
            if (mode == RB_SPLAT) {
                // Brcb has no b8 form, so the one-byte dtype splats per row from a
                // scalar GM read - the same mechanism the unbatched path uses, but
                // the DMA, the compare and the store around it are still batched.
                SplatRowsByte(dst, gm, off, k);
                return dst;
            }
        }
        LocalTensor<T> stage = q.template AllocTensor<T>();
        if (mode == RB_HOLD) {
            CopyRowsPitched(stage, gm, off, 1, rowStride);
        } else {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = k * S;  // RB_SPLAT has row stride 1: the k scalars are contiguous
            p.srcStride = 0;
            p.dstStride = 0;
            DataCopyPadExtParams<T> pad;
            DataCopyPad(stage, gm[off], p, pad);
        }
        q.EnQue(stage);
        LocalTensor<T> src = q.template DeQue<T>();
        if (mode == RB_HOLD) {
            ReplicateRow(dst, src, rb_.keepHold ? rb_.maxRows : k);
            if(rb_.keepHold) heldOff=off;
        } else if constexpr (S != 1) {
            SplatRows(dst, src, k);
        }
        q.FreeTensor(src);
        return dst;
    }

    // k rows of innerLen elements each, laid down at rb_.pitch.
    __aicore__ inline void CopyRowsPitched(const LocalTensor<T> &dst, const GlobalTensor<T> &gm, uint64_t gmOff,
                                           uint32_t k, uint64_t rowStride)
    {
        DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(k);
        p.blockLen = static_cast<uint32_t>(bc_.innerLen) * S;
        // A single-block descriptor has no gap to express, and RB_HOLD would
        // otherwise underflow this subtraction with its zero row stride.
        p.srcStride = (k > 1) ? static_cast<uint32_t>((rowStride - bc_.innerLen) * S) : 0u;
        p.dstStride = rb_.gapBlk;  // UB gap, in datablocks
        DataCopyPadExtParams<T> pad;
        DataCopyPad(dst, gm[gmOff], p, pad);
    }

    // One row, invariant across the batch, written into all k row slots.
    // srcRepeatSize 0 makes every repeat re-read the same row.
    __aicore__ inline void ReplicateRow(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t k)
    {
        LocalTensor<VT> d = dst.template ReinterpretCast<VT>();
        LocalTensor<VT> s = src.template ReinterpretCast<VT>();
        for (uint32_t o = 0; o < rb_.vtPerRow; o += VT_PER_REPEAT) {
            const uint32_t m = (rb_.vtPerRow - o < VT_PER_REPEAT) ? (rb_.vtPerRow - o) : VT_PER_REPEAT;
            Copy(d[o], s[o], static_cast<uint64_t>(m), static_cast<uint8_t>(k),
                 CopyRepeatParams(1, 1, static_cast<uint16_t>(rb_.pitchBlk), 0));
        }
    }

    // k scalars, one per row, each filling its whole row slot. Brcb puts scalar r
    // into datablock r; the Copy then re-reads that one datablock for every block
    // of row r. Two vector instructions for the whole tile, and no scalar reads.
    __aicore__ inline void SplatRows(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t k)
    {
        LocalTensor<VT> tmp = Slice<VT>(OffSel());
        Brcb(tmp, src.template ReinterpretCast<VT>(), static_cast<uint8_t>((k + 7) / 8), BrcbRepeatParams(1, 8));
        LocalTensor<VT> d = dst.template ReinterpretCast<VT>();
        for (uint32_t o = 0; o < rb_.vtPerRow; o += VT_PER_REPEAT) {
            const uint32_t m = (rb_.vtPerRow - o < VT_PER_REPEAT) ? (rb_.vtPerRow - o) : VT_PER_REPEAT;
            Copy(d[o], tmp, static_cast<uint64_t>(m), static_cast<uint8_t>(k),
                 CopyRepeatParams(1, 0, static_cast<uint16_t>(rb_.pitchBlk), 1));
        }
    }

    // b8 splat: duplicate the doubled byte over each row slot. rb_.pitch is a
    // multiple of 32, so every row start is 32-byte aligned and even.
    __aicore__ inline void SplatRowsByte(const LocalTensor<T> &dst, const GlobalTensor<T> &gm, uint64_t off, uint32_t k)
    {
        for (uint32_t r = 0; r < k; ++r) {
            const uint16_t v = static_cast<uint16_t>(static_cast<uint8_t>(gm.GetValue(off + r)));
            LocalTensor<uint16_t> d = dst[r * rb_.pitch].template ReinterpretCast<uint16_t>();
            Duplicate(d, static_cast<uint16_t>((v << 8) | v), rb_.pitch / 2);
        }
    }

    // k result rows, taken from a rb_.pitch-byte UB pitch into contiguous GM.
    __aicore__ inline void CopyOutRows(uint64_t gmOff, uint32_t k)
    {
        LocalTensor<int8_t> o = qO_.template DeQue<int8_t>();
        DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(k);
        p.blockLen = static_cast<uint32_t>(bc_.innerLen);  // bytes: one byte per element
        p.srcStride = 0;  // rb_.pitch bytes is exactly RoundUp(innerLen, 32), so no extra gap
        p.dstStride = 0;  // output rows are adjacent in GM
        DataCopyPad(yGm_[gmOff], o, p);
        qO_.FreeTensor(o);
    }

    // Before replacing a cached row, drain outstanding pipelines. Ensure the
    // load completes before its vector consumer. Reload only when the row offset changes.
    __aicore__ inline void LoadRowBlock(const LocalTensor<T> &dst, const GlobalTensor<T> &gm, uint64_t gmOff)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(bc_.innerLen) * S;
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<T> pad;
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst, gm[gmOff], p, pad);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, BUF_NUM> &q, const GlobalTensor<T> &gm, uint64_t gmOff,
                                  uint32_t n)
    {
        LocalTensor<T> l = q.template AllocTensor<T>();
        if constexpr(FAM==FAM_FLAT) {
            if(alignedCopy_ && (gmOff*S)%32==0 && (n*S)%32==0) {
                DataCopy(l,gm[gmOff],n);
                q.EnQue(l);
                return;
            }
        }
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = n * S;
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<T> pad;
        DataCopyPad(l, gm[gmOff], p, pad);
        q.EnQue(l);
    }

    __aicore__ inline void CopyOut(uint64_t gmOff, uint32_t n)
    {
        LocalTensor<int8_t> o = qO_.template DeQue<int8_t>();
        if constexpr(FAM==FAM_FLAT) {
            if(alignedCopy_ && gmOff%32==0 && n%32==0) {
                DataCopy(yGm_[gmOff],o,n);
                qO_.FreeTensor(o);
                return;
            }
        }
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = n;
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(yGm_[gmOff], o, p);
        qO_.FreeTensor(o);
    }

    __aicore__ inline void Compute(const LocalTensor<T> &a, const LocalTensor<T> &b, uint32_t na)
    {
        LocalTensor<uint8_t> mask = Slice<uint8_t>(OffMask());
        if constexpr (INT_PATH) {
            LocalTensor<int32_t> ca = Slice<int32_t>(OffCA());
            Min(ca, a, b, na);                            // min(x1, x2)
            Compare(mask, ca, a, CMPMODE::EQ, na);        // (min == x1)  <=>  (x1 <= x2)
            LocalTensor<uint16_t> m16 = mask.template ReinterpretCast<uint16_t>();
            Not(m16, m16, maskBytes_ / 2);                // negate -> (x1 > x2)
        } else if constexpr (NEED_CAST) {
            LocalTensor<CT> ca = Slice<CT>(OffCA());
            LocalTensor<CT> cb = Slice<CT>(OffCB());
            Cast(ca, a, RoundMode::CAST_NONE, na);
            Cast(cb, b, RoundMode::CAST_NONE, na);
            Compare(mask, ca, cb, CMPMODE::GT, na);
        } else {
            Compare(mask, a, b, CMPMODE::GT, na);
        }
        LocalTensor<half> sel = Slice<half>(OffSel());
        Select(sel, mask, onesT_, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, na);
        LocalTensor<int8_t> o = qO_.template AllocTensor<int8_t>();
        Cast(o, sel, RoundMode::CAST_NONE, na);
        qO_.EnQue(o);
    }

    // Splat one operand element across the whole tile, copying raw bits so that
    // NaN payloads and signed zeros survive untouched.
    //
    // GetValue reads GM from the scalar pipe. The value reaches Duplicate as a
    // register operand, so the S->V dependency is a register interlock rather
    // than a memory ordering edge, and the GM location read is a read-only
    // operand tensor that nothing in this kernel writes.
    __aicore__ inline void FillScalarX(uint64_t idx)
    {
        LocalTensor<T> dst = bc_.bufX.template Get<T>();
        if constexpr (S == 4) {
            LocalTensor<int32_t> d = dst.template ReinterpretCast<int32_t>();
            Duplicate(d, bc_.rawX32.GetValue(idx), tileLen_);
        } else if constexpr (S == 2) {
            LocalTensor<uint16_t> d = dst.template ReinterpretCast<uint16_t>();
            Duplicate(d, bc_.rawX16.GetValue(idx), tileLen_);
        } else {
            LocalTensor<uint16_t> d = dst.template ReinterpretCast<uint16_t>();
            const uint16_t v = static_cast<uint16_t>(static_cast<uint8_t>(x1Gm_.GetValue(idx)));
            Duplicate(d, static_cast<uint16_t>((v << 8) | v), tileLen_ / 2);
        }
    }

    __aicore__ inline void FillScalarY(uint64_t idx)
    {
        LocalTensor<T> dst = bc_.bufY.template Get<T>();
        if constexpr (S == 4) {
            LocalTensor<int32_t> d = dst.template ReinterpretCast<int32_t>();
            Duplicate(d, bc_.rawY32.GetValue(idx), tileLen_);
        } else if constexpr (S == 2) {
            LocalTensor<uint16_t> d = dst.template ReinterpretCast<uint16_t>();
            Duplicate(d, bc_.rawY16.GetValue(idx), tileLen_);
        } else {
            LocalTensor<uint16_t> d = dst.template ReinterpretCast<uint16_t>();
            const uint16_t v = static_cast<uint16_t>(static_cast<uint8_t>(x2Gm_.GetValue(idx)));
            Duplicate(d, static_cast<uint16_t>((v << 8) | v), tileLen_ / 2);
        }
    }

    // ---- UB buffer / pipe audit -------------------------------------------
    // bufArena_ holds ones / sel / mask / castA / castB at fixed disjoint
    // offsets. All five are touched by the VECTOR pipe only (Duplicate /
    // Compare / Not / Cast / Select write them, Compare / Select / Cast read
    // them) and none is ever a DataCopy endpoint. Single pipe, program order, so
    // no synchronisation edge is required for them - which is precisely why they
    // could be merged into one slot at all. The regions a DMA does touch
    // (bc_.bufX / bc_.bufY and every TQue) are deliberately NOT in the arena.
    //
    // The sel region doubles as the Brcb landing area in SplatRows. That is
    // still one pipe: Brcb writes it (V), the following Copy reads it (V), and
    // the Select later in the same tile overwrites it (V). Program order covers
    // all three, and Brcb's destination never overlaps the ones region, which is
    // the only arena region that has to survive across tiles.
    //
    // bc_.bufX / bc_.bufY have two writers, and which one is live depends on the
    // route:
    //   * batched route (rb_.on): written by Copy or Duplicate, i.e. the
    //     VECTOR pipe, and read by the compare on the VECTOR pipe. Not a
    //     DataCopy endpoint at all - every DMA in that route lands in a TQue and
    //     gets its MTE2->V edge from the framework. SetupRowBatch forces
    //     bc_.cacheX/bc_.cacheY off precisely so this stays true.
    //   * unbatched route: LoadRowBlock writes them from MTE2 while the vector
    //     pipe reads them across many output tiles. Both directions are covered
    //     by the PipeBarrier pair inside LoadRowBlock, which also covers the
    //     write-after-write against any vector write left by the other route.
    // The remaining writer, FillScalarX/Y, is a vector-pipe Duplicate; it can
    // never race the DMA writer because the host makes the two mutually
    // exclusive - innerMode==1 requires x1 broadcast on the inner axis, and
    // cacheMode bit0 requires it dense there.
    TQue<QuePosition::VECIN, BUF_NUM> qX_;
    TQue<QuePosition::VECIN, BUF_NUM> qY_;
    TQue<QuePosition::VECOUT, BUF_NUM> qO_;
    TBuf<QuePosition::VECCALC> bufArena_;
    LocalTensor<uint8_t> arena_;
    LocalTensor<half> onesT_;

    GlobalTensor<T> x1Gm_, x2Gm_;
    GlobalTensor<int8_t> yGm_;

    // Init assigns the valid flag first and initializes each remaining field before
    // use. Process returns when the object is invalid; flat routes use only their
    // initialized state.
    TPipe *pipe_;
    uint64_t myStart_;
    uint64_t myLen_;
    uint32_t tileLen_;
    uint32_t maskBytes_;
    // Row-batch state is used only for broadcast processing.
    GtRowBatch<FAM> rb_;
    // Every field the broadcast route needs and the flat family does not. Empty
    // for FAM_FLAT; see GtBcast.
    GtBcast<FAM> bc_;
    bool valid_ = false;
    bool alignedCopy_ = false;
    bool prefetch_ = false;
    bool roundRobin_ = false;
    uint64_t totalFlat_;
    uint32_t flatCores_;
};

// The one-tile route requires each core share to fit GT_ONE_TILE and the host
// UB bound. Its fixed UB layout keeps input, output and vector scratch regions
// disjoint and aligned. Input regions transfer from MTE2 to V; output transfers
// from V to MTE3. Explicit events enforce both dependencies. Event ID zero is
// independent for each HardEvent type. Initialize the SoC state before work and
// drain all pipelines before returning.
template <typename T>
class KernelGreaterOne {
public:
    using CT = typename GtTraits<T>::CT;
    static constexpr uint32_t S = sizeof(T);
    static constexpr uint32_t CS = sizeof(CT);
    static constexpr bool NEED_CAST = GtTraits<T>::kNeedCast;
    static constexpr bool INT_PATH = GtTraits<T>::kIntPath;

    // The fixed UB layout uses 256-element vector bounds and 32-byte alignment.
    static constexpr uint32_t N_ELEM = GT_ONE_TILE;
    static constexpr uint32_t OFF_X = 0;
    static constexpr uint32_t OFF_Y = OFF_X + N_ELEM * S;
    static constexpr uint32_t OFF_O = OFF_Y + N_ELEM * S;              // int8 result
    static constexpr uint32_t OFF_ONES = OFF_O + N_ELEM;               // half 1.0
    static constexpr uint32_t OFF_SEL = OFF_ONES + N_ELEM * 2;         // half
    static constexpr uint32_t OFF_MASK = OFF_SEL + N_ELEM * 2;
    static constexpr uint32_t MASK_BYTES = (N_ELEM / 8 + MASK_GRAN - 1) / MASK_GRAN * MASK_GRAN;
    static constexpr uint32_t OFF_CA = OFF_MASK + MASK_BYTES;
    static constexpr uint32_t OFF_CB = OFF_CA + N_ELEM * CS;
    static constexpr uint32_t UB_BYTES =
        NEED_CAST ? (OFF_CB + N_ELEM * CS) : (INT_PATH ? (OFF_CA + N_ELEM * CS) : OFF_CA);
    // Worst case over the five dtypes is int32 and bfloat16 at 35,072 bytes.
    // The host refuses this family unless N_ELEM * 32 + UB_RESERVE fits, which
    // is a strictly larger figure, so this can only ever be the smaller bound.

    __aicore__ inline KernelGreaterOne() {}

    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GtTiling t)
    {
        // The one thing TPipe's constructor does that this route needs:
        // set_atomic_none / set_mask_norm / set_vector_mask.  Everything else it
        // does is bookkeeping for services this route does not use.
        InitSocState();

        // Read the first two uint64_t tiling fields and bind the GM buffers before copying.
        const uint64_t total = t->totalNum;
        const uint64_t perCore = t->perCore;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);

        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * perCore;
        if (total == 0 || start >= total) {
            return;
        }
        const uint64_t left = total - start;
        // n <= perCore <= GT_ONE_TILE is the host's precondition for selecting
        // this family (greater_tiling_core.h, ONE_TILE_ELEMS), so na = round-up
        // of n to TILE_GRAN can never exceed N_ELEM and no region can overrun
        // into its neighbour.
        const uint32_t n = static_cast<uint32_t>(left < perCore ? left : perCore);
        const uint32_t na = RoundUpU32(n, TILE_GRAN);

        // Construct LocalTensor views over aligned, bounded regions of the fixed UB layout.
        LocalTensor<uint8_t> a(TPosition::VECCALC, 0, UB_BYTES);
        LocalTensor<T> xL = a[OFF_X].template ReinterpretCast<T>();
        LocalTensor<T> yL = a[OFF_Y].template ReinterpretCast<T>();

        // One descriptor shape for both operands - same element count, same
        // dtype - so the struct is built once instead of once per CopyIn.
        DataCopyExtParams pin;
        pin.blockCount = 1;
        pin.blockLen = n * S;
        pin.srcStride = 0;
        pin.dstStride = 0;
        DataCopyPadExtParams<T> pad;
        DataCopyPad(xL, x1Gm_[start], pin, pad);
        DataCopyPad(yL, x2Gm_[start], pin, pad);

        // Issued after both descriptors on purpose: the vector unit builds the
        // ones vector while MTE2 is still moving the operands.  It writes `ones`,
        // which no DMA in this route touches, so it needs no edge in either
        // direction against the two loads above.
        LocalTensor<half> ones = a[OFF_ONES].template ReinterpretCast<half>();
        Duplicate(ones, (half)1.0, na);

        // Wait for both MTE2 input transfers before the first vector operand read.
        SetFlag<HardEvent::MTE2_V>(GT_EVT_ID);
        WaitFlag<HardEvent::MTE2_V>(GT_EVT_ID);

        LocalTensor<uint8_t> mask = a[OFF_MASK];
        if constexpr (INT_PATH) {
            LocalTensor<int32_t> ca = a[OFF_CA].template ReinterpretCast<int32_t>();
            Min(ca, xL, yL, na);                        // min(x1, x2)
            Compare(mask, ca, xL, CMPMODE::EQ, na);     // (min == x1)  <=>  (x1 <= x2)
            LocalTensor<uint16_t> m16 = mask.template ReinterpretCast<uint16_t>();
            Not(m16, m16, MASK_BYTES / 2);              // negate -> (x1 > x2)
        } else if constexpr (NEED_CAST) {
            LocalTensor<CT> ca = a[OFF_CA].template ReinterpretCast<CT>();
            LocalTensor<CT> cb = a[OFF_CB].template ReinterpretCast<CT>();
            Cast(ca, xL, RoundMode::CAST_NONE, na);
            Cast(cb, yL, RoundMode::CAST_NONE, na);
            Compare(mask, ca, cb, CMPMODE::GT, na);
        } else {
            Compare(mask, xL, yL, CMPMODE::GT, na);
        }
        LocalTensor<half> sel = a[OFF_SEL].template ReinterpretCast<half>();
        Select(sel, mask, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, na);
        LocalTensor<int8_t> oL = a[OFF_O].template ReinterpretCast<int8_t>();
        Cast(oL, sel, RoundMode::CAST_NONE, na);

        // Wait for vector conversion to complete before the MTE3 output transfer.
        SetFlag<HardEvent::V_MTE3>(GT_EVT_ID);
        WaitFlag<HardEvent::V_MTE3>(GT_EVT_ID);
        DataCopyExtParams pout;
        pout.blockCount = 1;
        pout.blockLen = n;      // the output is one byte per element
        pout.srcStride = 0;
        pout.dstStride = 0;
        DataCopyPad(yGm_[start], oL, pout);

        // Drain all pipelines before returning so outstanding transfers are complete.
        pipe_barrier(PIPE_ALL);
    }

private:
    // See the event-id note in the class comment: nothing else in this variant
    // allocates one, and the pool is per HardEvent.
    static constexpr int32_t GT_EVT_ID = 0;

    GlobalTensor<T> x1Gm_, x2Gm_;
    GlobalTensor<int8_t> yGm_;
};

// Aligned one-tile route. The host validates exact core splitting, alignment
// and UB capacity before selecting it; other work uses the general routes.
#include "greater_fast.h"
#include "greater_repeat.h"
#include "greater_flat_reuse.h"
#include "greater_row_cached.h"
#include "greater_phase_cache.h"

// Optional diagnostic delay, excluded when GT_DIAG is zero. The host supplies
// its code in the route-specific payload.
#include "greater_diag.h"

// Optional diagnostic ladder for eligible fp16 aligned work. Runtime variants
// 10 through 16 select keys 900 through 906. These prefixes intentionally do
// not compute full Greater output. The default route does not use the ladder;
// GT_LADDER=0 excludes it.
#ifndef GT_LADDER
#define GT_LADDER 1
#endif
#if GT_LADDER
#include "greater_ladder.h"
#endif

extern "C" __global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    const uint64_t gtDiagStart = GtDiagStart();
#if GT_TILING_GM
    // Read the tiling fields where they are, instead of copying all 224 bytes of
    // the struct into stack first. Every route reads only the handful of fields
    // it actually uses.
    GtTiling tp = reinterpret_cast<GtTiling>(tiling);
#else
    GET_TILING_DATA(tilingData, tiling);
    GtTiling tp = &tilingData;
#endif

    // Instantiate TPipe only in general routes that use queue-managed buffers.

    // The framework supplies the tiling key. GT_TWIN reads it from launch metadata.
#ifdef GT_TWIN
    // The launch adapter reserves the first eight workspace bytes for the tiling key.
    __gm__ uint64_t *gtKeySrc = reinterpret_cast<__gm__ uint64_t *>(workspace);
    const uint64_t gtKey = gtKeySrc[0];
#define GT_KEY_IS(k) (gtKey == static_cast<uint64_t>(k))
#else
#define GT_KEY_IS(k) TILING_KEY_IS(k)
#endif

// Run the enabled diagnostic after the computation route drains its output.
#define GT_RUN(TYPE, KEY)                          \
    if (GT_KEY_IS(KEY)) {                          \
        if constexpr ((KEY)==0) {                 \
            if(tp->pad0 & 4096u) {                \
                TPipe reusePipe;                  \
                KernelGreaterFlatReuse reuse;     \
                reuse.Init(x1,x2,y,tp,&reusePipe); \
                reuse.Process();                  \
                GtDiagSpin(tiling,workspace,GT_DIAG_OFF_GEN,gtDiagStart); \
                return;                           \
            }                                     \
        }                                         \
        if constexpr (((KEY)%10)==FAM_BCAST && GtTraits<TYPE>::kIntPath) { \
            if(tp->pad0 & 32768u) {                \
                TPipe phasePipe;                  \
                KernelGreaterPhaseCache phase;    \
                phase.Init(x1,x2,y,tp,&phasePipe); \
                phase.Process();                  \
                GtDiagSpin(tiling,workspace,GT_DIAG_OFF_GEN,gtDiagStart); \
                return;                           \
            }                                     \
            if (tp->pad0 & 512u) {                 \
                TPipe repeatPipe;                 \
                KernelGreaterRepeat<TYPE,true> repeat; \
                repeat.Init(x1,x2,y,tp,&repeatPipe); \
                repeat.Process();                 \
                GtDiagSpin(tiling,workspace,GT_DIAG_OFF_GEN,gtDiagStart); \
                return;                           \
            }                                     \
        }                                         \
        if constexpr (((KEY)%10)==FAM_BCAST && (GtTraits<TYPE>::kIntPath || (GtTraits<TYPE>::kNeedCast && sizeof(TYPE)==2))) { \
            if(tp->pad0 & 8192u) {                 \
                TPipe rowPipe;                    \
                KernelGreaterRowCached<TYPE> row;  \
                row.Init(x1,x2,y,tp,&rowPipe);     \
                row.Process();                    \
                GtDiagSpin(tiling,workspace,GT_DIAG_OFF_GEN,gtDiagStart); \
                return;                           \
            }                                     \
            if (tp->pad0 & 1u) {                   \
                TPipe repeatPipe;                  \
                KernelGreaterRepeat<TYPE> repeat;  \
                repeat.Init(x1,x2,y,tp,&repeatPipe); \
                repeat.Process();                  \
                GtDiagSpin(tiling,workspace,GT_DIAG_OFF_GEN,gtDiagStart); \
                return;                            \
            }                                      \
        }                                          \
        TPipe pipe;                                \
        KernelGreater<TYPE, (KEY) % 10> op;        \
        op.Init(x1, x2, y, tp, &pipe);             \
        op.Process();                              \
        GtDiagSpin(tiling, workspace, GT_DIAG_OFF_GEN, gtDiagStart); \
        return;                                    \
    }

// The device toolchain extracts literal tiling keys to compile route variants.
// Every key the host emits needs a corresponding literal dispatch branch.
#define GT_RUN_ONE(TYPE, KEY)                      \
    if (GT_KEY_IS(KEY)) {                          \
        KernelGreaterOne<TYPE> op;                 \
        op.Run(x1, x2, y, tp);                     \
        GtDiagSpin(tiling, workspace, GT_DIAG_OFF_ONE, gtDiagStart); \
        return;                                    \
    }

// The aligned route reads perCore from its eight-byte payload at offset zero.
#define GT_RUN_FAST(TYPE, KEY)                     \
    if (GT_KEY_IS(KEY)) {                          \
        KernelGreaterFast<TYPE> op;                \
        op.Run(x1, x2, y, tiling);                 \
        GtDiagSpin(tiling, workspace, GT_DIAG_OFF_FAST, gtDiagStart); \
        return;                                    \
    }

    GT_RUN(float, 0)
    GT_RUN(float, 1)
    GT_RUN_ONE(float, 2)
    GT_RUN_FAST(float, 3)
    GT_RUN(half, 10)
    GT_RUN(half, 11)
    GT_RUN_ONE(half, 12)
    GT_RUN_FAST(half, 13)
    GT_RUN(bfloat16_t, 20)
    GT_RUN(bfloat16_t, 21)
    GT_RUN_ONE(bfloat16_t, 22)
    GT_RUN_FAST(bfloat16_t, 23)
    GT_RUN(int32_t, 30)
    GT_RUN(int32_t, 31)
    GT_RUN_ONE(int32_t, 32)
    GT_RUN_FAST(int32_t, 33)
    GT_RUN(int8_t, 40)
    GT_RUN(int8_t, 41)
    GT_RUN_ONE(int8_t, 42)
    GT_RUN_FAST(int8_t, 43)

#if GT_LADDER
// Use literal keys for diagnostic ladder dispatch so the toolchain can discover
// them. The host selects these keys only for eligible fp16 aligned work.
// Static assertions keep the selected key and prefix level consistent.
#define GT_RUN_LADDER(TYPE, KEY, LEVEL)                     \
    if (GT_KEY_IS(KEY)) {                                   \
        static_assert((KEY) == 900 + (LEVEL),               \
                      "ladder key literal and level disagree"); \
        KernelGreaterLadder<TYPE, (LEVEL)> op;              \
        op.Run(x1, x2, y, tiling);                          \
        return;                                             \
    }
    GT_RUN_LADDER(half, 900, 0)   // L0   enter, return
    GT_RUN_LADDER(half, 901, 1)   // L0w  + the observation write
    GT_RUN_LADDER(half, 902, 2)   // L1   + the 8-byte tiling read
    GT_RUN_LADDER(half, 903, 3)   // L2   + address setup
    GT_RUN_LADDER(half, 904, 4)   // L3   + DataCopy in x1
    GT_RUN_LADDER(half, 905, 5)   // L4   + DataCopy in x2
    GT_RUN_LADDER(half, 906, 6)   // L5   + the compare/select/cast
#undef GT_RUN_LADDER
#endif
#undef GT_RUN_FAST
#undef GT_RUN_ONE
#undef GT_RUN
#undef GT_KEY_IS
}

// The toolchain extracts literal tiling keys separately from compiling their
// bodies. Keep this guard consistent with all dispatch branches, including
// the conditional diagnostic ladder.
#define GT_TKCAT_(a, b) a##b
#define GT_TKCAT(a, b) GT_TKCAT_(a, b)
#define GT_TK_RUNTIME_g_tilingKey 1
#if !GT_TKCAT(GT_TK_RUNTIME_, TILING_KEY_VAR)
#if GT_LADDER
#define GT_LADDER_KEYS_OK                                                        \
    (TILING_KEY_IS(900) || TILING_KEY_IS(901) || TILING_KEY_IS(902) ||           \
     TILING_KEY_IS(903) || TILING_KEY_IS(904) || TILING_KEY_IS(905) ||           \
     TILING_KEY_IS(906))
#else
#define GT_LADDER_KEYS_OK 0
#endif
#if !(TILING_KEY_IS(0)  || TILING_KEY_IS(1)  || TILING_KEY_IS(2)  || TILING_KEY_IS(3)  || \
      TILING_KEY_IS(10) || TILING_KEY_IS(11) || TILING_KEY_IS(12) || TILING_KEY_IS(13) || \
      TILING_KEY_IS(20) || TILING_KEY_IS(21) || TILING_KEY_IS(22) || TILING_KEY_IS(23) || \
      TILING_KEY_IS(30) || TILING_KEY_IS(31) || TILING_KEY_IS(32) || TILING_KEY_IS(33) || \
      TILING_KEY_IS(40) || TILING_KEY_IS(41) || TILING_KEY_IS(42) || TILING_KEY_IS(43) || \
      GT_LADDER_KEYS_OK)
#error "Greater: a kernel variant was emitted for a tiling key that greater() has no GT_RUN/GT_RUN_ONE/GT_RUN_LADDER for. Either give the key a route, or remove the stray literal that registered it. Shipping it means a launch that writes nothing."
#endif
#undef GT_LADDER_KEYS_OK
#endif
#undef GT_TK_RUNTIME_g_tilingKey
#undef GT_TKCAT
#undef GT_TKCAT_
