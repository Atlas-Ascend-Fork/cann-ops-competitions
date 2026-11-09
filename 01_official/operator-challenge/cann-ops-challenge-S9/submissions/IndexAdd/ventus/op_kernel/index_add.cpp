/**
 * IndexAdd 设备侧实现。
 * 沿指定轴将 source 累加到索引指定的输出位置；重复索引的处理顺序与相应累加路径保持一致。
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../op_host/index_add_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t kAccElements = 16384U;
constexpr uint32_t kSrcElements = 4096U;
constexpr uint32_t kIndexTile = 2048U;
constexpr uint32_t kResidentIndexElements = 8192U;
constexpr uint32_t kResidentMaxRows = 2048U;
constexpr uint32_t kMaxBatch = 64U;
constexpr uint32_t kUbBlockBytes = 32U;

constexpr uint32_t kAtomicTile = 8192U;
constexpr uint32_t kAtomicTileMax = 20480U;

template <typename A, typename B>
struct SameType {
    static constexpr bool value = false;
};
template <typename A>
struct SameType<A, A> {
    static constexpr bool value = true;
};

template <typename T>
struct AccTraits {
    using Type = T;
};
template <>
struct AccTraits<int8_t> {
    using Type = float;
};
template <>
struct AccTraits<bfloat16_t> {
    using Type = float;
};

template <typename T>
// 根据类型和分块信息选择输出驻留、索引分桶或原子散射累加路径。
class KernelIndexAdd {
public:
    using Acc = typename AccTraits<T>::Type;
    static constexpr bool kDirect = SameType<T, Acc>::value;
    static constexpr bool kIsInt8 = SameType<T, int8_t>::value;

    // BF16 使用 FP32 执行加法，并按相应路径转换回 BF16，以匹配逐步舍入要求。
    static constexpr bool kIsBf16 = SameType<T, bfloat16_t>::value;

    static constexpr bool kIsInteger = kIsInt8 || SameType<T, int32_t>::value;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                GM_ADDR workspace, const IndexAddTilingData& tiling)
    {
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));

        outer_ = tiling.outer;
        axisLen_ = tiling.axisLen;
        inner_ = tiling.inner;
        indexCount_ = tiling.indexCount;
        innerChunk_ = tiling.innerChunk;
        innerChunkCount_ = tiling.innerChunkCount;
        axisChunk_ = tiling.axisChunk;
        axisChunkCount_ = tiling.axisChunkCount;
        taskCount_ = tiling.taskCount;
        useBucket_ = tiling.useBucket;
        numSlices_ = tiling.numSlices;
        sliceRows_ = tiling.sliceRows;
        sliceShift_ = tiling.sliceShift;
        bucketCap_ = tiling.bucketCap;
        bucketCores_ = tiling.bucketCores;
        useResident_ = tiling.useResident;
        blockDim_ = tiling.blockDim;
        singleSubblock_ = tiling.singleSubblock;

        if (!kIsInteger && useBucket_ != 0) {

            GM_ADDR user = GetUserWorkspace(workspace);
            const uint32_t blocks = blockDim_;
            syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(user), blocks * 8);
            GM_ADDR countsAddr = user + static_cast<uint64_t>(blocks) * 32;
            countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countsAddr),
                                      bucketCores_ * numSlices_);
            GM_ADDR bucketAddr = countsAddr + bucketCores_ * numSlices_ * sizeof(int32_t);
            bucketGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bucketAddr),
                                      bucketCores_ * numSlices_ * bucketCap_ * 2);

            pipe_.InitBuffer(idxBuf_, kIndexTile * sizeof(int32_t));
            pipe_.InitBuffer(accBuf_, kAccElements * sizeof(float));
            pipe_.InitBuffer(srcAccBuf_, kSrcElements * sizeof(float));
            pipe_.InitBuffer(syncUb_, 64U * 32U);
            pipe_.InitBuffer(bcntBuf_, 256U * sizeof(int32_t));
            if constexpr (!kDirect) {

                pipe_.InitBuffer(selfTBuf_, kAccElements * sizeof(T));
                pipe_.InitBuffer(srcTBuf_, kSrcElements * sizeof(T));
                pipe_.InitBuffer(roundBuf_, kSrcElements * sizeof(T));
            }
            {
                uint32_t caBytes =
                    static_cast<uint32_t>(bucketCores_ * numSlices_ * sizeof(int32_t));
                caBytes = ((caBytes + 31U) / 32U) * 32U;
                pipe_.InitBuffer(countsAllBuf_, caBytes < 32U ? 32U : caBytes);
            }
            InitGlobalMemory(syncGm_, blocks * 8, 0);
            return;
        }

        if constexpr (kIsInteger) {

          ownPath_ = false;
          if constexpr (kDirect && kIsInteger) {
            if (useBucket_ == 2U && axisChunkCount_ == 1U && indexCount_ > 0U &&
                innerChunk_ > 0U) {
                const uint32_t p =
                    Pitch(static_cast<uint32_t>(innerChunk_));
                const uint32_t accBytes = static_cast<uint32_t>(axisLen_) * p *
                    static_cast<uint32_t>(sizeof(Acc));
                uint32_t ibBytes = static_cast<uint32_t>(
                    ((indexCount_ + 7U) / 8U * 8U) * sizeof(int32_t));
                if (ibBytes < 32U) { ibBytes = 32U; }
                const uint32_t fixed = accBytes + ibBytes + 4096U;
                const uint32_t oneSlot =
                    2U * p * static_cast<uint32_t>(sizeof(Acc));
                if (fixed + oneSlot <= 196608U) {
                    uint32_t ob = (196608U - fixed) / oneSlot;
                    if (ob > 255U) { ob = 255U; }
                    if (static_cast<uint64_t>(ob) > indexCount_) {
                        ob = static_cast<uint32_t>(indexCount_);
                    }
                    if (ob < 1U) { ob = 1U; }
                    ownBatch_ = ob;
                    ownSlotElems_ = ob * p;
                    ownPath_ = true;
                    ownShards_ = numSlices_ < 1U ? 1U : numSlices_;
                    pipe_.InitBuffer(idxBuf_, ibBytes);
                    pipe_.InitBuffer(accBuf_, accBytes < 32U ? 32U : accBytes);
                    pipe_.InitBuffer(srcAccBuf_,
                        2U * ownSlotElems_ * static_cast<uint32_t>(sizeof(Acc)));
                    if (ownShards_ > 1U) {

                        GM_ADDR user = GetUserWorkspace(workspace);
                        syncGm_.SetGlobalBuffer(
                            reinterpret_cast<__gm__ int32_t*>(user),
                            blockDim_ * 8);
                        pipe_.InitBuffer(syncUb_, 64U * 32U);
                        InitGlobalMemory(syncGm_, blockDim_ * 8, 0);

                    }
                }
            }
          }
          if (ownPath_) {

          } else {
            GM_ADDR user = GetUserWorkspace(workspace);
            syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(user),
                                    blockDim_ * 8);
            const uint32_t ib =
                static_cast<uint32_t>(((indexCount_ + 7) / 8 * 8) * sizeof(int32_t));
            pipe_.InitBuffer(atomicIdxBuf_, ib < 32U ? 32U : ib);

            {
                const uint32_t avail = 196608U - (ib < 32U ? 32U : ib) - 2048U - 4096U;
                uint32_t at = avail / (2U * static_cast<uint32_t>(sizeof(T)));
                if (at > kAtomicTileMax) { at = kAtomicTileMax; }
                if (at < kAtomicTile) { at = kAtomicTile; }
                atomicTile_ = at;
                pipe_.InitBuffer(atomicDataBuf_,
                                 2U * at * sizeof(T) < 32U ? 32U
                                     : 2U * at * sizeof(T));
            }
            pipe_.InitBuffer(syncUb_, 64U * 32U);
            InitGlobalMemory(syncGm_, blockDim_ * 8, 0);
          }
        } else {
            const bool sortedResident = SortedResidentEligible();
            if constexpr (SameType<T, half>::value) {
                if (sortedResident) {

                    output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }
            }
            const bool narrowPath = NarrowEligible();

            const bool atomicFloat =
                !(useResident_ != 0U && narrowPath) && AtomicFloatEligible();
            if (atomicFloat) {
                GM_ADDR user2 = GetUserWorkspace(workspace);
                syncGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t*>(user2), blockDim_ * 8);
                const uint32_t ib2 = static_cast<uint32_t>(
                    ((indexCount_ + 7) / 8 * 8) * sizeof(int32_t));
                pipe_.InitBuffer(atomicIdxBuf_, ib2 < 32U ? 32U : ib2);
                {
                    const uint32_t avail2 =
                        196608U - (ib2 < 32U ? 32U : ib2) - 2048U - 4096U;
                    uint32_t at2 = avail2 / (2U * static_cast<uint32_t>(sizeof(T)));
                    if (at2 > kAtomicTileMax) { at2 = kAtomicTileMax; }
                    if (at2 < kAtomicTile) { at2 = kAtomicTile; }
                    atomicTile_ = at2;
                    pipe_.InitBuffer(atomicDataBuf_,
                                     2U * at2 * sizeof(T) < 32U ? 32U
                                         : 2U * at2 * sizeof(T));
                }
                pipe_.InitBuffer(syncUb_, 64U * 32U);
                InitGlobalMemory(syncGm_, blockDim_ * 8, 0);
            }

            pipe_.InitBuffer(
                idxBuf_, (sortedResident ? kResidentIndexElements
                          : (narrowPath ? 2U * kIndexTile : kIndexTile)) *
                sizeof(int32_t));

            if (narrowPath && SameType<T, float>::value) {
                const uint32_t pitchA =
                    AlignUpBlockElems(static_cast<uint32_t>(axisLen_) *
                                      static_cast<uint32_t>(sizeof(Acc))) /
                    static_cast<uint32_t>(sizeof(Acc));
                if (pitchA != 0U) {
                    const uint32_t bSmall =
                        ((kNarrowAccElems / pitchA) / 16U) * 16U;

                    const uint32_t cSmall = bSmall > kNarrowBMax ? kNarrowBMax : bSmall;


                    if (cSmall != 0U && (cSmall % 16U) == 0U) {

                        const uint32_t bp = cSmall;
                        uint32_t ck = kNarrowSrcDead / bp;
                        if (ck > kNarrowStageDead / bp) { ck = kNarrowStageDead / bp; }
                        if (ck > kIndexTile) { ck = kIndexTile; }
                        const uint32_t unit0 = kUbBlockBytes /
                            static_cast<uint32_t>(sizeof(Acc));
                        if (ck >= unit0) { ck = (ck / unit0) * unit0; }
                        if (ck >= 16U) { ck = (ck / 16U) * 16U; }
                        const uint32_t rc = NarrowRowChunk(ck);
                        const uint32_t rem =
                            static_cast<uint32_t>(axisLen_) % rc;
                        const uint32_t headRows = rem == 0U ? rc : rem;
                        const uint32_t headPitch =
                            AlignUpBlockElems(headRows *
                                static_cast<uint32_t>(sizeof(Acc))) /
                            static_cast<uint32_t>(sizeof(Acc));
                        const uint32_t rcPitch =
                            AlignUpBlockElems(rc *
                                static_cast<uint32_t>(sizeof(Acc))) /
                            static_cast<uint32_t>(sizeof(Acc));

                        const uint32_t pitchAll =
                            AlignUpBlockElems(static_cast<uint32_t>(axisLen_) *
                                static_cast<uint32_t>(sizeof(Acc))) /
                            static_cast<uint32_t>(sizeof(Acc));
                        const bool accRoom =
                            (pitchAll + (rcPitch - rc)) * bp <= narrowAccElems_;
                        if (rc != 0U && (rcPitch % 16U) == 0U &&
                            (headPitch % 16U) == 0U && accRoom) {
                            narrowSrcElems_ = kNarrowSrcDead;
                            narrowStageElems_ = kNarrowStageDead;
                            narrowTabElems_ = kNarrowTabDead;
                            narrowTabDead_ = true;
                        }
                    }
                }
            }

            if (narrowPath && SameType<T, float>::value) {
                const uint32_t pitchA =
                    AlignUpBlockElems(static_cast<uint32_t>(axisLen_) *
                                      static_cast<uint32_t>(sizeof(Acc))) /
                    static_cast<uint32_t>(sizeof(Acc));
                const uint32_t bpNow =
                    pitchA == 0U ? 0U
                                 : ((narrowAccElems_ / pitchA) / 16U) * 16U;
                uint32_t bp48 =
                    pitchA == 0U ? 0U : ((kNarrowAcc48 / pitchA) / 16U) * 16U;
                if (bp48 > kNarrowBMax) { bp48 = kNarrowBMax; }

                const uint32_t perCore = blockDim_ == 0U ? 0U :
                    static_cast<uint32_t>((outer_ + blockDim_ - 1U) / blockDim_);
                const bool groupsDrop = bpNow != 0U && bp48 != 0U &&
                    ((perCore + bp48 - 1U) / bp48) <
                    ((perCore + bpNow - 1U) / bpNow);
                if (pitchA != 0U && bp48 > bpNow && bp48 != 0U && groupsDrop) {
                    uint32_t ck = kNarrowSrc48 / bp48;
                    if (ck > kNarrowStage48 / bp48) { ck = kNarrowStage48 / bp48; }
                    if (ck > kIndexTile) { ck = kIndexTile; }
                    const uint32_t unit0 = kUbBlockBytes /
                        static_cast<uint32_t>(sizeof(Acc));
                    if (ck >= unit0) { ck = (ck / unit0) * unit0; }
                    if (ck >= 16U) { ck = (ck / 16U) * 16U; }
                    if (ck != 0U) {
                        const uint32_t rc = NarrowRowChunk(ck);
                        const uint32_t rem =
                            static_cast<uint32_t>(axisLen_) % rc;
                        const uint32_t headRows = rem == 0U ? rc : rem;
                        const uint32_t esz2 =
                            static_cast<uint32_t>(sizeof(Acc));
                        const uint32_t headPitch =
                            AlignUpBlockElems(headRows * esz2) / esz2;
                        const uint32_t rcPitch =
                            AlignUpBlockElems(rc * esz2) / esz2;
                        const bool accRoom =
                            (pitchA + (rcPitch - rc)) * bp48 <= kNarrowAcc48;
                        const bool stageRoom =
                            bp48 * rcPitch <= kNarrowStage48;
                        if ((rcPitch % 16U) == 0U && (headPitch % 16U) == 0U &&
                            accRoom && stageRoom) {
                            narrowAccElems_ = kNarrowAcc48;
                            narrowSrcElems_ = kNarrowSrc48;
                            narrowStageElems_ = kNarrowStage48;
                            narrowTabElems_ = kNarrowTabDead;
                            narrowTabDead_ = true;
                        }
                    }
                }
            }
            pipe_.InitBuffer(accBuf_,
                narrowPath ? narrowAccElems_ * static_cast<uint32_t>(sizeof(Acc))
                           : kAccElements * static_cast<uint32_t>(sizeof(Acc)));
            const uint32_t residentSourceElements =
                sortedResident && SameType<T, half>::value ?
                2U * kSrcElements : kSrcElements;
            pipe_.InitBuffer(srcAccBuf_,
                narrowPath ? narrowSrcElems_ * static_cast<uint32_t>(sizeof(Acc))
                           : residentSourceElements * static_cast<uint32_t>(sizeof(Acc)));

            if (narrowPath) {
                pipe_.InitBuffer(narrowStageBuf_, narrowStageElems_ * sizeof(Acc));

                const uint32_t tb = sizeof(Acc) == 2U
                    ? kNarrowTabBytes
                    : narrowTabElems_ * static_cast<uint32_t>(sizeof(int32_t));
                pipe_.InitBuffer(narrowFwdBuf_, tb);
                pipe_.InitBuffer(narrowBwdBuf_, tb);

                if (NarrowSplitShards() > 1U) {
                    GM_ADDR user = GetUserWorkspace(workspace);
                    syncGm_.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(user), blockDim_ * 8);
                    pipe_.InitBuffer(syncUb_, 64U * 32U);
                    InitGlobalMemory(syncGm_, blockDim_ * 8, 0);
                }
            }
            if constexpr ((kDirect || kIsBf16) && !kIsInteger) {
                if (sortedResident) {
                    pipe_.InitBuffer(
                        residentSortedBuf_, kResidentIndexElements * sizeof(int32_t));
                    pipe_.InitBuffer(
                        residentRowStartBuf_, (kResidentMaxRows + 1U) * sizeof(int32_t));
                    pipe_.InitBuffer(
                        residentCursorBuf_, kResidentMaxRows * sizeof(int32_t));
                }
            }
            if constexpr (kIsBf16) {
                if (sortedResident && bucketCores_ != 0U) {
                    GM_ADDR user = GetUserWorkspace(workspace);
                    GM_ADDR sortedAddr = user;
                    residentSortedGm_.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(sortedAddr),
                        indexCount_);
                    const uint64_t sortedBytes =
                        ((indexCount_ * sizeof(int32_t) + 31U) / 32U) * 32U;
                    residentStartsGm_.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(sortedAddr + sortedBytes),
                        axisLen_ + 1U);
                }
            }
            if constexpr (!kDirect) {
                const uint32_t selfElements = sortedResident && kIsBf16 ?
                    kSrcElements : kAccElements;
                pipe_.InitBuffer(selfTBuf_, selfElements * sizeof(T));
                pipe_.InitBuffer(srcTBuf_, kSrcElements * sizeof(T));
            }
            if constexpr (kIsInt8) {
                pipe_.InitBuffer(halfBuf_, kAccElements * sizeof(half));
                pipe_.InitBuffer(wrapFBuf_, kAccElements * sizeof(float));
                pipe_.InitBuffer(wrapIBuf_, kAccElements * sizeof(int32_t));
            }
            if constexpr (kIsBf16) {
                pipe_.InitBuffer(roundBuf_, kSrcElements * sizeof(T));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (kIsInteger) {
            if constexpr (kDirect) {
                if (ownPath_) {
                    IntegerOwnedProcess();
                    return;
                }
            }
            AtomicProcess();
            return;
        }
        if constexpr (SameType<T, float>::value || SameType<T, half>::value || SameType<T, bfloat16_t>::value) {

            if (useResident_ != 0 && NarrowEligible()) {
                NarrowResidentProcess();
                return;
            }

            if (AtomicFloatEligible()) {
                AtomicProcess();
                return;
            }
            if (useResident_ != 0) {
                if constexpr (kDirect || kIsBf16) {
                    const bool sortedResident = SortedResidentEligible();
                    if (sortedResident) {
                        if constexpr (kIsBf16) {
                            ResidentBf16SortedProcess();
                        } else {
                            ResidentSortedProcess();
                        }
                        return;
                    }
                }
                ResidentProcess();
                return;
            }
            if (useBucket_ != 0) {
                BucketProcess();
                return;
            }
        }
        if (taskCount_ == 0) {
            return;
        }
        const uint64_t blocks = static_cast<uint64_t>(blockDim_);
        const uint64_t block = LogicalBlockIndex();
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        for (uint64_t task = begin; task < end; ++task) {
            RunTask(task);
        }
    }

private:
    __aicore__ inline bool SortedResidentEligible() const
    {
        if constexpr (!(kDirect || kIsBf16)) {
            return false;
        }
        const uint32_t residentPitch = static_cast<uint32_t>(
            (((inner_ * sizeof(T) + kUbBlockBytes - 1U) /
              kUbBlockBytes) * kUbBlockBytes) / sizeof(T));
        const bool bf16Capacity = !kIsBf16 ||
            (residentPitch != 0U && kSrcElements / residentPitch >= 8U);
        const uint64_t outerThreshold = kIsBf16 ? 40U : 384U;
        return useResident_ != 0U && outer_ >= outerThreshold &&
            indexCount_ >= 128U && indexCount_ <= kResidentIndexElements &&
            inner_ * sizeof(T) >= 128U &&
            ((inner_ * sizeof(T) + 31U) / 32U) <= 255U &&
            axisLen_ <= kResidentMaxRows && bf16Capacity;
    }

    __aicore__ inline uint64_t LogicalBlockIndex() const
    {
        if constexpr (SameType<T, half>::value) {
            if (singleSubblock_ == 0U) {
                return static_cast<uint64_t>(GetBlockIdx());
            }
            return static_cast<uint64_t>(GetBlockIdx()) / GetTaskRation();
        }
        return static_cast<uint64_t>(GetBlockIdx());
    }

    __aicore__ inline void BuildResidentStableOrder()
    {
        if constexpr (kDirect || kIsBf16) {
            LocalTensor<int32_t> indices = idxBuf_.template Get<int32_t>();
            LocalTensor<int32_t> sorted = residentSortedBuf_.template Get<int32_t>();
            LocalTensor<int32_t> starts = residentRowStartBuf_.template Get<int32_t>();
            LocalTensor<int32_t> cursor = residentCursorBuf_.template Get<int32_t>();
            const DataCopyPadExtParams<int32_t> pad;
            const DataCopyExtParams copy(
                1, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)),
                0, 0, 0);
            DataCopyPad(indices, index_[0], copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            Duplicate(cursor, 0, static_cast<int32_t>(axisLen_));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            for (uint32_t j = 0; j < static_cast<uint32_t>(indexCount_); ++j) {
                const uint32_t row = static_cast<uint32_t>(indices.GetValue(j));
                if (row < axisLen_) {
                    cursor.SetValue(row, cursor.GetValue(row) + 1);
                }
            }
            uint32_t sum = 0U;
            for (uint32_t row = 0; row < static_cast<uint32_t>(axisLen_); ++row) {
                const uint32_t count = static_cast<uint32_t>(cursor.GetValue(row));
                starts.SetValue(row, static_cast<int32_t>(sum));
                cursor.SetValue(row, static_cast<int32_t>(sum));
                sum += count;
            }
            starts.SetValue(static_cast<uint32_t>(axisLen_), static_cast<int32_t>(sum));
            for (uint32_t j = 0; j < static_cast<uint32_t>(indexCount_); ++j) {
                const uint32_t row = static_cast<uint32_t>(indices.GetValue(j));
                if (row < axisLen_) {
                    const uint32_t pos = static_cast<uint32_t>(cursor.GetValue(row));
                    sorted.SetValue(pos, static_cast<int32_t>(j));
                    cursor.SetValue(row, static_cast<int32_t>(pos + 1U));
                }
            }
        }
    }

    __aicore__ inline void PrepareResidentBf16Order()
    {
        if constexpr (kIsBf16) {
            if (bucketCores_ == 0U) {
                BuildResidentStableOrder();
                return;
            }

            LocalTensor<int32_t> sorted =
                residentSortedBuf_.template Get<int32_t>();
            LocalTensor<int32_t> starts =
                residentRowStartBuf_.template Get<int32_t>();
            const uint64_t block = LogicalBlockIndex();
            if (block == 0U) {
                BuildResidentStableOrder();
                const DataCopyExtParams sortedCopy(
                    1U, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)),
                    0U, 0U, 0U);
                const DataCopyExtParams startsCopy(
                    1U, static_cast<uint32_t>((axisLen_ + 1U) * sizeof(int32_t)),
                    0U, 0U, 0U);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyPad(residentSortedGm_[0], sorted, sortedCopy);
                DataCopyPad(residentStartsGm_[0], starts, startsCopy);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }

            SyncAll();

            const DataCopyPadExtParams<int32_t> pad;
            const DataCopyExtParams sortedCopy(
                1U, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)),
                0U, 0U, 0U);
            const DataCopyExtParams startsCopy(
                1U, static_cast<uint32_t>((axisLen_ + 1U) * sizeof(int32_t)),
                0U, 0U, 0U);
            DataCopyPad(sorted, residentSortedGm_[0], sortedCopy, pad);
            DataCopyPad(starts, residentStartsGm_[0], startsCopy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }

    __aicore__ inline void ResidentSortedProcess()
    {
        if constexpr (kDirect) {
            BuildResidentStableOrder();
            const uint64_t blocks = static_cast<uint64_t>(blockDim_);
            const uint64_t block = LogicalBlockIndex();
            const uint32_t innerLen = static_cast<uint32_t>(inner_);
            const uint32_t pitch = Pitch(innerLen);
            uint32_t maxBatch = kSrcElements / pitch;
            if (maxBatch > kMaxBatch) {
                maxBatch = kMaxBatch;
            }
            LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
            LocalTensor<Acc> sourceRows = srcAccBuf_.template Get<Acc>();
            LocalTensor<int32_t> sorted = residentSortedBuf_.template Get<int32_t>();
            LocalTensor<int32_t> starts = residentRowStartBuf_.template Get<int32_t>();
            const DataCopyPadExtParams<T> pad;
            const DataCopyExtParams rowCopy(
                1, innerLen * sizeof(T), 0, 0, 0);
            constexpr uint32_t vectorWidth = sizeof(Acc) == 4U ? 64U : 128U;
            const uint8_t sourceRepStride = static_cast<uint8_t>(
                pitch * sizeof(Acc) / kUbBlockBytes);
            const BinaryRepeatParams repeatParams{
                1, 1, 1, 0, sourceRepStride, 0};

            for (uint64_t outerIndex = block; outerIndex < outer_; outerIndex += blocks) {
                const uint64_t selfBase = outerIndex * axisLen_ * inner_;
                const uint64_t sourceBase = outerIndex * indexCount_ * inner_;
                LoadSelfBlock(
                    selfBase, static_cast<uint32_t>(axisLen_), innerLen, pitch);

                for (uint32_t row = 0; row < static_cast<uint32_t>(axisLen_); ++row) {
                    uint32_t pos = static_cast<uint32_t>(starts.GetValue(row));
                    const uint32_t end = static_cast<uint32_t>(starts.GetValue(row + 1U));
                    while (pos < end) {
                        const uint32_t batch =
                            (end - pos) < maxBatch ? (end - pos) : maxBatch;
                        if constexpr (SameType<T, half>::value) {
                            uint32_t j = 0U;
                            for (; j + 1U < batch; j += 2U) {
                                const uint32_t firstIndex = static_cast<uint32_t>(
                                    sorted.GetValue(pos + j));
                                const uint32_t secondIndex = static_cast<uint32_t>(
                                    sorted.GetValue(pos + j + 1U));
                                const uint32_t sourceGap = static_cast<uint32_t>(
                                    (static_cast<uint64_t>(secondIndex - firstIndex) * inner_ -
                                     innerLen) * sizeof(T));
                                const DataCopyExtParams pairCopy(
                                    2U, innerLen * sizeof(T), sourceGap, 0, 0);
                                DataCopyPad(
                                    sourceRows[j * pitch],
                                    source_[sourceBase + firstIndex * inner_],
                                    pairCopy, pad);
                            }
                            if (j < batch) {
                                const uint32_t sourceIndex = static_cast<uint32_t>(
                                    sorted.GetValue(pos + j));
                                DataCopyPad(
                                    sourceRows[j * pitch],
                                    source_[sourceBase + sourceIndex * inner_],
                                    rowCopy, pad);
                            }
                        } else {

                            for (uint32_t j = 0U; j < batch; ++j) {
                                const uint32_t sourceIndex = static_cast<uint32_t>(
                                    sorted.GetValue(pos + j));
                                DataCopyPad(
                                    sourceRows[j * pitch],
                                    source_[sourceBase + sourceIndex * inner_],
                                    rowCopy, pad);
                            }

                        }
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                        for (uint32_t lane = 0; lane < innerLen; lane += vectorWidth) {
                            const uint32_t part =
                                (innerLen - lane) < vectorWidth ?
                                (innerLen - lane) : vectorWidth;
                            Add(
                                acc[row * pitch + lane], sourceRows[lane],
                                acc[row * pitch + lane],
                                static_cast<uint64_t>(part),
                                static_cast<uint8_t>(batch), repeatParams);
                        }
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                        pos += batch;
                    }
                }
                StoreBlock(
                    selfBase, static_cast<uint32_t>(axisLen_), innerLen,
                    pitch, static_cast<uint32_t>(axisLen_ * pitch));
            }
        }
    }

    __aicore__ inline void ResidentBf16SortedProcess()
    {
        if constexpr (kIsBf16) {
            PrepareResidentBf16Order();
            const uint64_t blocks = static_cast<uint64_t>(blockDim_);
            const uint64_t block = LogicalBlockIndex();
            const uint32_t innerLen = static_cast<uint32_t>(inner_);
            const uint32_t pitch = Pitch(innerLen);
            uint32_t outerBatch = kSrcElements / pitch;
            const uint32_t accCapacity = kAccElements / pitch;
            if (outerBatch > accCapacity) {
                outerBatch = accCapacity;
            }
            if (outerBatch > 255U) {
                outerBatch = 255U;
            }
            const uint32_t outerBatches = static_cast<uint32_t>(
                (outer_ + outerBatch - 1U) / outerBatch);
            const uint64_t tasks =
                static_cast<uint64_t>(outerBatches) * axisLen_;
            LocalTensor<float> acc = accBuf_.template Get<float>();
            LocalTensor<float> sourceAcc0 = srcAccBuf_.template Get<float>();
            LocalTensor<T> selfT = selfTBuf_.template Get<T>();
            LocalTensor<T> sourceT0 = srcTBuf_.template Get<T>();
            LocalTensor<uint8_t> indexScratch =
                idxBuf_.template Get<uint8_t>();
            LocalTensor<T> sourceT1 =
                indexScratch.template ReinterpretCast<T>();
            LocalTensor<float> sourceAcc1 =
                indexScratch[kSrcElements * sizeof(T)]
                    .template ReinterpretCast<float>();
            LocalTensor<T> rounded = roundBuf_.template Get<T>();
            LocalTensor<int32_t> sorted =
                residentSortedBuf_.template Get<int32_t>();
            LocalTensor<int32_t> starts =
                residentRowStartBuf_.template Get<int32_t>();
            const DataCopyPadExtParams<T> pad;
            const uint8_t repeatStride = static_cast<uint8_t>(
                pitch * sizeof(float) / kUbBlockBytes);
            const BinaryRepeatParams parallelRows{
                1, 1, 1, repeatStride, repeatStride, repeatStride};
            constexpr uint32_t vectorWidth = 64U;
            const uint32_t selfGap = static_cast<uint32_t>(
                (axisLen_ * inner_ - innerLen) * sizeof(T));
            const uint32_t sourceGap = static_cast<uint32_t>(
                (indexCount_ * inner_ - innerLen) * sizeof(T));

            for (uint64_t task = block; task < tasks; task += blocks) {
                const uint32_t outerBatchIndex = static_cast<uint32_t>(
                    task / axisLen_);
                const uint32_t row = static_cast<uint32_t>(
                    task - static_cast<uint64_t>(outerBatchIndex) * axisLen_);
                const uint64_t outerBase =
                    static_cast<uint64_t>(outerBatchIndex) * outerBatch;
                const uint32_t planes = static_cast<uint32_t>(
                    (outer_ - outerBase) < outerBatch ?
                    (outer_ - outerBase) : outerBatch);
                const uint32_t elements = planes * pitch;
                const uint64_t selfBase =
                    outerBase * axisLen_ * inner_ +
                    static_cast<uint64_t>(row) * inner_;
                const uint64_t sourceBase =
                    outerBase * indexCount_ * inner_;

                const DataCopyExtParams selfCopy(
                    planes, innerLen * sizeof(T), selfGap, 0, 0);
                DataCopyPad(selfT, self_[selfBase], selfCopy, pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(acc, selfT, RoundMode::CAST_NONE,
                     static_cast<int32_t>(elements));
                PipeBarrier<PIPE_V>();

                uint32_t pos = static_cast<uint32_t>(starts.GetValue(row));
                const uint32_t end = static_cast<uint32_t>(
                    starts.GetValue(row + 1U));
                SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
                const DataCopyExtParams sourceCopy(
                    planes, innerLen * sizeof(T), sourceGap, 0, 0);
                if (pos < end) {
                    const uint32_t firstSource = static_cast<uint32_t>(
                        sorted.GetValue(pos));
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    DataCopyPad(
                        sourceT0,
                        source_[sourceBase +
                                static_cast<uint64_t>(firstSource) * inner_],
                        sourceCopy, pad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                }
                uint32_t ordinal = 0U;
                while (pos < end) {
                    const uint32_t ping = ordinal & 1U;
                    const event_t event = ping == 0U ? EVENT_ID1 : EVENT_ID2;
                    if (pos + 1U < end) {
                        const uint32_t nextPing = ping ^ 1U;
                        const event_t nextEvent =
                            nextPing == 0U ? EVENT_ID1 : EVENT_ID2;
                        const uint32_t nextSource = static_cast<uint32_t>(
                            sorted.GetValue(pos + 1U));
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        LocalTensor<T> nextT = nextPing == 0U ?
                            sourceT0 : sourceT1;
                        DataCopyPad(
                            nextT,
                            source_[sourceBase +
                                    static_cast<uint64_t>(nextSource) * inner_],
                            sourceCopy, pad);
                        SetFlag<HardEvent::MTE2_V>(nextEvent);
                    }
                    WaitFlag<HardEvent::MTE2_V>(event);
                    LocalTensor<T> currentT = ping == 0U ? sourceT0 : sourceT1;
                    LocalTensor<float> currentAcc =
                        ping == 0U ? sourceAcc0 : sourceAcc1;
                    Cast(currentAcc, currentT, RoundMode::CAST_NONE,
                         static_cast<int32_t>(elements));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(event);
                    for (uint32_t lane = 0U; lane < innerLen;
                         lane += vectorWidth) {
                        const uint32_t part =
                            (innerLen - lane) < vectorWidth ?
                            (innerLen - lane) : vectorWidth;
                        Add(
                            acc[lane], acc[lane], currentAcc[lane],
                            static_cast<uint64_t>(part),
                            static_cast<uint8_t>(planes), parallelRows);
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(rounded, acc, RoundMode::CAST_RINT,
                         static_cast<int32_t>(elements));
                    PipeBarrier<PIPE_V>();
                    Cast(acc, rounded, RoundMode::CAST_NONE,
                         static_cast<int32_t>(elements));
                    PipeBarrier<PIPE_V>();
                    ++pos;
                    ++ordinal;
                }
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                Cast(selfT, acc, RoundMode::CAST_RINT,
                     static_cast<int32_t>(elements));
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                const DataCopyExtParams store(
                    planes, innerLen * sizeof(T), 0, selfGap, 0);
                DataCopyPad(output_[selfBase], selfT, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline uint32_t Pitch(uint32_t chunkLen) const
    {
        const uint32_t bytes = ((chunkLen * sizeof(T) + kUbBlockBytes - 1) / kUbBlockBytes) * kUbBlockBytes;
        return bytes / sizeof(T);
    }

    __aicore__ inline void RunTask(uint64_t task)
    {
        const uint64_t perOuter = innerChunkCount_ * axisChunkCount_;
        const uint64_t outerIndex = task / perOuter;
        const uint64_t rest = task - outerIndex * perOuter;
        const uint64_t innerIndex = rest / axisChunkCount_;
        const uint64_t axisIndex = rest - innerIndex * axisChunkCount_;

        const uint64_t innerBegin = innerIndex * innerChunk_;
        const uint32_t chunkLen =
            static_cast<uint32_t>((inner_ - innerBegin) < innerChunk_ ? (inner_ - innerBegin) : innerChunk_);
        const uint64_t axisBegin = axisIndex * axisChunk_;
        const uint32_t rows =
            static_cast<uint32_t>((axisLen_ - axisBegin) < axisChunk_ ? (axisLen_ - axisBegin) : axisChunk_);
        if (chunkLen == 0 || rows == 0) {
            return;
        }
        const uint32_t pitch = Pitch(chunkLen);
        const uint32_t blockElements = rows * pitch;
        const uint64_t selfBase = outerIndex * axisLen_ * inner_ + axisBegin * inner_ + innerBegin;

        LoadSelfBlock(selfBase, rows, chunkLen, pitch);
        Accumulate(outerIndex, axisBegin, rows, innerBegin, chunkLen, pitch);
        StoreBlock(selfBase, rows, chunkLen, pitch, blockElements);
    }

    __aicore__ inline void LoadSelfBlock(uint64_t base, uint32_t rows, uint32_t chunkLen, uint32_t pitch)
    {
        const DataCopyPadExtParams<T> pad;
        if constexpr (kDirect) {
            LocalTensor<T> dst = accBuf_.template Get<T>();
            const DataCopyExtParams copy(
                rows, chunkLen * sizeof(T), (inner_ - chunkLen) * sizeof(T), 0, 0);
            DataCopyPad(dst, self_[base], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            LocalTensor<T> staging = selfTBuf_.template Get<T>();
            const DataCopyExtParams copy(
                rows, chunkLen * sizeof(T), (inner_ - chunkLen) * sizeof(T), 0, 0);
            DataCopyPad(staging, self_[base], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ToAccumulator(accBuf_.template Get<Acc>(), staging, rows * pitch);
        }
    }

    __aicore__ inline void ToAccumulator(const LocalTensor<Acc>& dst, const LocalTensor<T>& src, uint32_t count)
    {
        if constexpr (kIsInt8) {
            LocalTensor<half> mid = halfBuf_.template Get<half>();
            Cast(mid, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Cast(dst, mid, RoundMode::CAST_NONE, static_cast<int32_t>(count));
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(uint64_t outerIndex, uint64_t axisBegin, uint32_t rows,
                                      uint64_t innerBegin, uint32_t chunkLen, uint32_t pitch)
    {
        const uint64_t sourceBase = outerIndex * indexCount_ * inner_ + innerBegin;
        LocalTensor<int32_t> indices = idxBuf_.template Get<int32_t>();
        const int32_t low = static_cast<int32_t>(axisBegin);
        const int32_t high = static_cast<int32_t>(axisBegin + rows);
        const DataCopyPadExtParams<int32_t> indexPad;

        for (uint64_t base = 0; base < indexCount_; base += kIndexTile) {
            const uint64_t left = indexCount_ - base;
            const uint32_t count = static_cast<uint32_t>(left < kIndexTile ? left : kIndexTile);
            const DataCopyExtParams copy(1, count * sizeof(int32_t), 0, 0, 0);
            DataCopyPad(indices, index_[base], copy, indexPad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AccumulateRows(indices, base, count, sourceBase, axisBegin, low, high, chunkLen, pitch);
        }
    }

    __aicore__ inline void AccumulateRows(const LocalTensor<int32_t>& indices, uint64_t base, uint32_t count,
                                          uint64_t sourceBase, uint64_t axisBegin, int32_t low, int32_t high,
                                          uint32_t chunkLen, uint32_t pitch)
    {
        uint32_t maxBatch = kSrcElements / pitch;
        if (maxBatch == 0) {
            maxBatch = 1;
        }
        if (maxBatch > kMaxBatch) {
            maxBatch = kMaxBatch;
        }
        LocalTensor<T> staging;
        if constexpr (kDirect) {
            staging = srcAccBuf_.template Get<T>();
        } else {
            staging = srcTBuf_.template Get<T>();
        }
        const DataCopyPadExtParams<T> pad;
        const DataCopyExtParams copy(1, chunkLen * sizeof(T), 0, 0, 0);
        uint32_t rowLocal[kMaxBatch];

        const bool noFilter = low <= 0 &&
            high >= static_cast<int32_t>(axisLen_);
        if (noFilter && count > 0U && maxBatch >= 2U) {
            const uint32_t half = maxBatch >> 1U;
            const uint32_t slotElems = half * pitch;
            const int32_t evB[2] = {EVENT_ID6, EVENT_ID7};
            uint32_t iIssue = 0U;
            uint32_t iUse = 0U;
            uint32_t sIssue = 0U;
            uint32_t sUse = 0U;
            uint32_t inflight = 0U;
            bool war[2] = {false, false};
            uint32_t cnt2[2] = {0U, 0U};
            uint32_t at2[2] = {0U, 0U};
            while (iUse < count) {
                while (inflight < 2U && iIssue < count) {
                    uint32_t b = count - iIssue;
                    if (b > half) { b = half; }
                    if (war[sIssue]) {
                        WaitFlag<HardEvent::V_MTE2>(evB[sIssue]);
                        war[sIssue] = false;
                    }

                    const DataCopyExtParams bcp(
                        b, chunkLen * sizeof(T),
                        (inner_ - chunkLen) * sizeof(T), 0, 0);
                    DataCopyPad(staging[sIssue * slotElems],
                                source_[sourceBase + (base + iIssue) * inner_],
                                bcp, pad);
                    SetFlag<HardEvent::MTE2_V>(evB[sIssue]);
                    cnt2[sIssue] = b;
                    at2[sIssue] = iIssue;
                    iIssue += b;
                    sIssue ^= 1U;
                    ++inflight;
                }
                WaitFlag<HardEvent::MTE2_V>(evB[sUse]);
                const uint32_t b = cnt2[sUse];
                const uint32_t j0 = at2[sUse];
                for (uint32_t t = 0; t < b; ++t) {
                    rowLocal[t] = static_cast<uint32_t>(
                        indices.GetValue(j0 + t)) -
                        static_cast<uint32_t>(axisBegin);
                }
                AddBatchNoDrain(rowLocal, b, chunkLen, pitch,
                                sUse * slotElems);
                SetFlag<HardEvent::V_MTE2>(evB[sUse]);
                war[sUse] = true;
                iUse += b;
                sUse ^= 1U;
                --inflight;
            }
            for (uint32_t d = 0U; d < 2U; ++d) {
                if (war[d]) {
                    WaitFlag<HardEvent::V_MTE2>(evB[d]);
                    war[d] = false;
                }
            }
            return;
        }
        uint32_t batch = 0;
        for (uint32_t j = 0; j < count; ++j) {
            const int32_t target = indices.GetValue(j);
            if (target < low || target >= high) {
                continue;
            }
            DataCopyPad(staging[batch * pitch], source_[sourceBase + (base + j) * inner_], copy, pad);
            rowLocal[batch] = static_cast<uint32_t>(target) - static_cast<uint32_t>(axisBegin);
            ++batch;
            if (batch == maxBatch) {
                FlushBatch(rowLocal, batch, chunkLen, pitch);
                batch = 0;
            }
        }
        if (batch != 0) {
            FlushBatch(rowLocal, batch, chunkLen, pitch);
        }
    }

    __aicore__ inline void AddBatchNoDrain(
        const uint32_t* rowLocal, uint32_t batch, uint32_t chunkLen,
        uint32_t pitch, uint32_t slotOff)
    {
        LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
        LocalTensor<Acc> src = srcAccBuf_.template Get<Acc>();
        if constexpr (!kDirect) {
            ToAccumulator(src[slotOff],
                          srcTBuf_.template Get<T>()[slotOff],
                          batch * pitch);
        }
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t row = rowLocal[b];
            Add(acc[row * pitch], acc[row * pitch],
                src[slotOff + b * pitch], static_cast<int32_t>(chunkLen));
            if constexpr (kIsBf16) {

                LocalTensor<T> rounded = roundBuf_.template Get<T>();
                Cast(rounded, acc[row * pitch], RoundMode::CAST_RINT,
                     static_cast<int32_t>(chunkLen));
                Cast(acc[row * pitch], rounded,
                     RoundMode::CAST_NONE, static_cast<int32_t>(chunkLen));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushBatch(const uint32_t* rowLocal, uint32_t batch,
                                          uint32_t chunkLen, uint32_t pitch)
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
        LocalTensor<Acc> src = srcAccBuf_.template Get<Acc>();
        if constexpr (!kDirect) {
            ToAccumulator(src, srcTBuf_.template Get<T>(), batch * pitch);
        }
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t row = rowLocal[b];
            Add(acc[row * pitch], acc[row * pitch], src[b * pitch], static_cast<int32_t>(chunkLen));
            if constexpr (kIsBf16) {
                LocalTensor<T> rounded = roundBuf_.template Get<T>();
                PipeBarrier<PIPE_V>();
                Cast(rounded, acc[row * pitch], RoundMode::CAST_RINT, static_cast<int32_t>(chunkLen));
                PipeBarrier<PIPE_V>();
                Cast(acc[row * pitch], rounded, RoundMode::CAST_NONE, static_cast<int32_t>(chunkLen));
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void StoreBlock(uint64_t base, uint32_t rows, uint32_t chunkLen,
                                      uint32_t pitch, uint32_t blockElements)
    {
        LocalTensor<T> out;
        if constexpr (kDirect) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            out = accBuf_.template Get<T>();
        } else {
            LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
            out = selfTBuf_.template Get<T>();
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if constexpr (kIsInt8) {
                WrapToInt8(acc, out, blockElements);
            } else {
                Cast(out, acc, RoundMode::CAST_RINT, static_cast<int32_t>(blockElements));
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        const DataCopyExtParams copy(
            rows, chunkLen * sizeof(T), 0, (inner_ - chunkLen) * sizeof(T), 0);
        DataCopyPad(output_[base], out, copy);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
    }

    static constexpr uint32_t kNarrowLanes =
        kUbBlockBytes / static_cast<uint32_t>(sizeof(Acc));

    static constexpr uint32_t kNarrowTable =
        sizeof(Acc) == 2U ? 14336U : 10496U;

    static constexpr uint32_t kNarrowAccElems =
        sizeof(Acc) == 2U ? 30720U : 20992U;
    static constexpr uint32_t kNarrowTabElems =
        sizeof(Acc) == 2U ? kNarrowTable : 4096U;

    static constexpr uint32_t kNarrowTabBytes =
        sizeof(Acc) == 2U ? 1024U : kNarrowTabElems * 4U;

    static constexpr uint32_t kNarrowSrcElems = sizeof(Acc) == 2U ? 24576U : 4096U;

    static constexpr uint32_t kNarrowStage =
        sizeof(Acc) == 2U ? 24576U : 8192U;

    static constexpr uint32_t kNarrowSrcDead = 10240U;
    static constexpr uint32_t kNarrowStageDead = 10240U;
    static constexpr uint32_t kNarrowTabDead = 256U;


    static constexpr uint32_t kNarrowAcc48 = 31744U;
    static constexpr uint32_t kNarrowSrc48 = 6144U;
    static constexpr uint32_t kNarrowStage48 = 6144U;

    bool narrowTabDead_ = false;
    uint32_t atomicTile_ = kAtomicTile;
    uint32_t narrowAccElems_ = kNarrowAccElems;
    uint32_t narrowSrcElems_ = kNarrowSrcElems;
    uint32_t narrowStageElems_ = kNarrowStage;
    uint32_t narrowTabElems_ = kNarrowTabElems;

    static constexpr uint32_t kNarrowBMax = sizeof(Acc) == 2U ? 128U : 64U;

    static constexpr uint32_t kNarrowMinShard = 512U;

    __aicore__ inline bool AtomicFloatEligible() const
    {
        if constexpr (!SameType<T, float>::value) {
            return false;
        } else {

            const uint64_t total = outer_ * axisLen_ * inner_;

            return useBucket_ == 0U && indexCount_ > 0U && axisLen_ > 0U &&
                   inner_ > 0U && total >= (1ULL << 16);
        }
    }

    __aicore__ inline bool NarrowEligible() const
    {
        if constexpr (!kDirect) {
            return false;
        } else {

            return useResident_ != 0U && inner_ == 1U && axisLen_ >= 1U &&
                   indexCount_ >= 1U &&
                   axisLen_ * kNarrowLanes <= kNarrowTable &&
                   axisLen_ <= kNarrowAccElems / kNarrowLanes;
        }
    }

    __aicore__ inline uint32_t NarrowSplitShards() const
    {
        if constexpr (!(kDirect && SameType<T, float>::value)) {
            return 1U;
        } else {
            if (!NarrowEligible() || blockDim_ == 0U) { return 1U; }
            const uint32_t esz = static_cast<uint32_t>(sizeof(Acc));
            const uint32_t pitchElems =
                AlignUpBlockElems(static_cast<uint32_t>(axisLen_) * esz) / esz;
            if (pitchElems == 0U) { return 1U; }

            uint32_t bpadCap = narrowAccElems_ / pitchElems;
            bpadCap = (bpadCap / kNarrowLanes) * kNarrowLanes;
            if (bpadCap > kNarrowBMax) { bpadCap = kNarrowBMax; }
            if (bpadCap == 0U) { return 1U; }
            uint64_t g = (outer_ + bpadCap - 1U) / bpadCap;
            if (g == 0U) { g = 1U; }
            if (g >= blockDim_) { return 1U; }
            uint32_t shards = static_cast<uint32_t>(blockDim_ / g);
            const uint32_t byIndex =
                static_cast<uint32_t>(indexCount_ / kNarrowMinShard);
            if (shards > byIndex) { shards = byIndex; }
            if (shards == 0U) { shards = 1U; }
            return shards;
        }
    }

    __aicore__ inline uint32_t NarrowRowChunk(uint32_t ck) const
    {
        uint32_t rc = (ck / kNarrowLanes) * kNarrowLanes;
        if (rc == 0U) { rc = kNarrowLanes; }
        if (rc > static_cast<uint32_t>(axisLen_)) {
            rc = static_cast<uint32_t>(axisLen_);
        }
        if constexpr (sizeof(Acc) != 4U) {
            return rc;
        } else {
            const uint32_t esz = static_cast<uint32_t>(sizeof(Acc));
            const uint32_t start = rc;
            for (uint32_t r = start; r >= kNarrowLanes; --r) {
                if ((AlignUpBlockElems(r * esz) / esz) % 16U != 0U) { continue; }
                const uint32_t rem = static_cast<uint32_t>(axisLen_) % r;
                const uint32_t hr = rem == 0U ? r : rem;
                if ((AlignUpBlockElems(hr * esz) / esz) % 16U == 0U) {
                    return r;
                }
            }
            return start;
        }
    }

    __aicore__ inline uint32_t NarrowBpad(uint32_t b) const
    {
        uint32_t bpad = ((b + kNarrowLanes - 1U) / kNarrowLanes) * kNarrowLanes;
        if (bpad > kNarrowBMax) { bpad = kNarrowBMax; }

        if constexpr (sizeof(Acc) == 4U) {
            if ((bpad % 16U) != 0U) {
                const uint32_t up = ((bpad + 15U) / 16U) * 16U;
                if (up <= kNarrowBMax) { bpad = up; }
            }
        }

        const uint32_t pitchElems =
            AlignUpBlockElems(static_cast<uint32_t>(axisLen_) *
                              static_cast<uint32_t>(sizeof(Acc))) /
            static_cast<uint32_t>(sizeof(Acc));

        uint32_t cap = static_cast<uint32_t>(narrowAccElems_ / pitchElems);

        constexpr uint32_t kCapUnit = SameType<T, float>::value ? 16U : kNarrowLanes;
        cap = (cap / kCapUnit) * kCapUnit;
        if (cap == 0U) { cap = kNarrowLanes; }
        if (bpad > cap) { bpad = cap; }
        return bpad;
    }

    __aicore__ inline void NarrowAffineTable(
        const LocalTensor<int32_t>& off, uint32_t n,
        int32_t a, int32_t b, uint32_t d)
    {
        LocalTensor<float> f0 =
            narrowStageBuf_.template Get<float>();
        uint32_t kChunk = f0.GetSize() / 3U;
        if (kChunk > 1024U) { kChunk = 1024U; }
        kChunk = (kChunk / 64U) * 64U;
        if (kChunk == 0U) {
            for (uint32_t k = 0; k < n; ++k) {
                off.SetValue(k, a * static_cast<int32_t>(k) +
                                b * static_cast<int32_t>(k / d));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID3);
            WaitFlag<HardEvent::S_V>(EVENT_ID3);
            return;
        }
        LocalTensor<float> f1 = f0[kChunk];
        LocalTensor<float> f2 = f0[2U * kChunk];
        const float invD = 1.0f / static_cast<float>(static_cast<int32_t>(d));
        for (uint32_t base = 0; base < n; base += kChunk) {
            const int32_t c = static_cast<int32_t>(
                (n - base) < kChunk ? (n - base) : kChunk);
            CreateVecIndex(f0, static_cast<float>(static_cast<int32_t>(base)), c);
            PipeBarrier<PIPE_V>();
            Muls(f1, f0, invD, c);
            PipeBarrier<PIPE_V>();
            Adds(f1, f1, 1.0e-4f, c);
            PipeBarrier<PIPE_V>();
            Cast(f1, f1, RoundMode::CAST_FLOOR, c);
            PipeBarrier<PIPE_V>();
            Muls(f0, f0, static_cast<float>(a), c);
            PipeBarrier<PIPE_V>();
            Muls(f1, f1, static_cast<float>(b), c);
            PipeBarrier<PIPE_V>();
            Add(f0, f0, f1, c);
            PipeBarrier<PIPE_V>();
            Cast(off[base], f0, RoundMode::CAST_RINT, c);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t AlignUpBlockElems(uint32_t bytes) const
    {
        return ((bytes + kUbBlockBytes - 1U) / kUbBlockBytes) * kUbBlockBytes;
    }

    __aicore__ inline void EnsureFwd(uint32_t pitch, uint32_t bpad,
                                     uint32_t& curPitch, uint32_t& curBpad)
    {
        if (pitch == curPitch && bpad == curBpad) { return; }
        const int32_t esz = static_cast<int32_t>(sizeof(Acc));
        LocalTensor<int32_t> fwdI = narrowFwdBuf_.template Get<int32_t>();
        const int32_t n = static_cast<int32_t>(pitch * bpad);
        NarrowAffineTable(fwdI, static_cast<uint32_t>(n),
                          esz * static_cast<int32_t>(pitch),
                          esz * (1 - static_cast<int32_t>(pitch * bpad)), bpad);
        curPitch = pitch;
        curBpad = bpad;

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void EnsureBwd(uint32_t pitch, uint32_t bpad,
                                     uint32_t& curPitch, uint32_t& curBpad)
    {
        if (pitch == curPitch && bpad == curBpad) { return; }
        const int32_t esz = static_cast<int32_t>(sizeof(Acc));
        LocalTensor<int32_t> bwdI = narrowBwdBuf_.template Get<int32_t>();
        const int32_t n = static_cast<int32_t>(pitch * bpad);
        NarrowAffineTable(bwdI, static_cast<uint32_t>(n),
                          esz * static_cast<int32_t>(bpad),
                          esz * (1 - static_cast<int32_t>(bpad * pitch)), pitch);
        curPitch = pitch;
        curBpad = bpad;

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NarrowCopySelf(uint64_t total, uint64_t blocks,
                                          uint64_t block)
    {
        const uint64_t share = total / blocks;
        const uint64_t extra = total % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        LocalTensor<Acc> buf = narrowStageBuf_.template Get<Acc>();
        const DataCopyPadExtParams<Acc> pd;

        const uint32_t tile = narrowStageElems_;
        for (uint64_t pos = begin; pos < end; pos += tile) {
            const uint32_t cnt = static_cast<uint32_t>(
                (end - pos) < tile ? (end - pos) : tile);
            const DataCopyExtParams cp(1, cnt * sizeof(Acc), 0, 0, 0);
            DataCopyPad(buf, self_[pos], cp, pd);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(output_[pos], buf, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void NarrowResidentProcess()
    {


        source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        self_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        if constexpr (kDirect) {
            const uint64_t blocks = static_cast<uint64_t>(blockDim_);
            const uint64_t block = LogicalBlockIndex();
            if (blocks == 0U || block >= blocks) { return; }
            const uint32_t axis = static_cast<uint32_t>(axisLen_);

            const uint32_t shards = NarrowSplitShards();
            uint64_t groups = blocks;
            uint64_t grp = block;
            uint64_t jBegin = 0;
            uint64_t jEnd = indexCount_;
            if (shards > 1U) {
                groups = blocks / shards;
                grp = block % groups;
                const uint64_t sIdx = block / groups;
                NarrowCopySelf(outer_ * axisLen_, blocks, block);

                SyncAll();

                if (sIdx >= shards) { return; }
                const uint64_t share = indexCount_ / shards;
                const uint64_t extra = indexCount_ % shards;
                jBegin = sIdx * share + (sIdx < extra ? sIdx : extra);
                jEnd = jBegin + share + (sIdx < extra ? 1 : 0);
                if (jBegin >= jEnd) { return; }
            }
            const uint64_t per = (outer_ + groups - 1U) / groups;
            const uint64_t o0 = grp * per;

            const uint64_t o1 = (o0 + per) < outer_ ? (o0 + per) : outer_;

            LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
            LocalTensor<Acc> srcT = srcAccBuf_.template Get<Acc>();
            LocalTensor<Acc> stage = narrowStageBuf_.template Get<Acc>();
            LocalTensor<int32_t> idxAll = idxBuf_.template Get<int32_t>();
            LocalTensor<int32_t> fwdI = narrowFwdBuf_.template Get<int32_t>();
            LocalTensor<int32_t> bwdI = narrowBwdBuf_.template Get<int32_t>();
            LocalTensor<uint32_t> fwd = fwdI.template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> bwd = bwdI.template ReinterpretCast<uint32_t>();
            const DataCopyPadExtParams<Acc> pad;
            const DataCopyPadExtParams<int32_t> ipad;
            const uint32_t esz = static_cast<uint32_t>(sizeof(Acc));

            const uint32_t phaseMod = esz == 2U ? 128U : 32U;
            uint64_t period = 1U;
            if (phaseMod != 0U && shards == 1U &&
                indexCount_ * esz >= 4U * phaseMod) {
                const uint32_t rem =
                    static_cast<uint32_t>((indexCount_ * esz) % phaseMod);
                uint32_t gg = phaseMod;
                uint32_t xx = rem;
                while (xx != 0U) { const uint32_t t = gg % xx; gg = xx; xx = t; }
                period = phaseMod / gg;
                if (outer_ < period) { period = 1U; }
            }

            uint32_t curBpad = 0U;
            uint32_t curPitch = 0U;
            uint32_t curBwdBpad = 0U;
            uint32_t curBwdPitch = 0U;

            uint32_t roundPlanes = 1U;
            uint64_t itBegin = o0;
            uint64_t itEnd = o1;
            if (period > 1U) {
                roundPlanes = NarrowBpad(kNarrowBMax);
                if (roundPlanes > kNarrowBMax) { roundPlanes = kNarrowBMax; }
                if (roundPlanes == 0U) { roundPlanes = 1U; }
                uint64_t totalRounds = 0U;
                for (uint64_t g = 0U; g < period; ++g) {
                    const uint64_t nG = (outer_ - g + period - 1U) / period;
                    totalRounds += (nG + roundPlanes - 1U) / roundPlanes;
                }
                const uint64_t rShare = totalRounds / groups;
                const uint64_t rExtra = totalRounds % groups;
                itBegin = grp * rShare + (grp < rExtra ? grp : rExtra);
                itEnd = itBegin + rShare + (grp < rExtra ? 1U : 0U);
            }
            uint32_t step = 0U;
            uint64_t cls = 0U;
            uint64_t clsRounds = 0U;
            for (uint64_t it = itBegin; it < itEnd;
                 it += (period > 1U ? 1U : step)) {
                uint64_t ob = it;
                uint64_t planesLeft = o1 - it;
                if (period > 1U) {
                    uint64_t nG = (outer_ - cls + period - 1U) / period;
                    while (cls < period &&
                           it >= clsRounds + (nG + roundPlanes - 1U) / roundPlanes) {
                        clsRounds += (nG + roundPlanes - 1U) / roundPlanes;
                        ++cls;
                        nG = (outer_ - cls + period - 1U) / period;
                    }
                    const uint64_t first = (it - clsRounds) * roundPlanes;
                    ob = cls + first * period;
                    planesLeft = nG - first;
                    if (planesLeft > roundPlanes) { planesLeft = roundPlanes; }
                }
                const uint32_t b = static_cast<uint32_t>(
                    planesLeft < kNarrowBMax ? planesLeft : kNarrowBMax);
                const uint32_t bpad = NarrowBpad(b);

                uint32_t chunk = ((bpad % 16U) == 0U &&
                                  (sizeof(Acc) == 2U || narrowTabDead_))
                                     ? (narrowSrcElems_ / bpad)
                                     : (narrowTabElems_ / bpad);
                if (chunk > narrowSrcElems_ / bpad) {
                    chunk = narrowSrcElems_ / bpad;
                }
                if (chunk > narrowStageElems_ / bpad) {
                    chunk = narrowStageElems_ / bpad;
                }
                if (chunk > kIndexTile) { chunk = kIndexTile; }

                {
                    const uint32_t unit = kUbBlockBytes / esz;
                    if (chunk >= unit) { chunk = (chunk / unit) * unit; }
                }

                if (narrowTabDead_ && chunk >= 16U) {
                    chunk = (chunk / 16U) * 16U;
                }
                if (period > 1U) {

                    const uint32_t unitM = phaseMod / esz;
                    if (chunk >= unitM) { chunk = (chunk / unitM) * unitM; }
                }
                if (chunk == 0U) { return; }
                const uint32_t bReal = b < bpad ? b : bpad;
                step = bReal;

                uint32_t zeroedSpitch = 0U;

                uint32_t rowChunk = NarrowRowChunk(chunk);
                if (rowChunk == 0U) { rowChunk = kNarrowLanes; }

                const uint32_t accSpan =
                    (AlignUpBlockElems(axis * esz) / esz) * bpad;
                const uint32_t rowHead =
                    (axis % rowChunk) == 0U ? rowChunk : (axis % rowChunk);
                if (shards > 1U) {
                    Duplicate(acc, static_cast<Acc>(0),
                              static_cast<int32_t>(accSpan));
                    PipeBarrier<PIPE_ALL>();
                }
                for (uint32_t r0 = 0; shards == 1U && r0 < axis; ) {
                    const uint32_t rows = r0 == 0U ? rowHead : rowChunk;
                    const uint32_t pitch =
                        AlignUpBlockElems(rows * esz) / esz;

                    const bool hwSelf = (bpad % 16U) == 0U && (pitch % 16U) == 0U;
                    if (!hwSelf) { EnsureFwd(pitch, bpad, curPitch, curBpad); }
                    const DataCopyExtParams cp(
                        bReal, rows * esz, (period * axisLen_ - rows) * esz, 0, 0);
                    DataCopyPad(stage, self_[ob * axisLen_ + r0], cp, pad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                    if (bReal < bpad) {
                        Duplicate(stage[bReal * pitch],
                                  static_cast<Acc>(0),
                                  static_cast<int32_t>((bpad - bReal) * pitch));
                        PipeBarrier<PIPE_V>();
                    }
                    if (hwSelf) {
                        AscendC::tiling::ConfusionTransposeTiling ct;
                        ct.param0 = sizeof(Acc) == 4U ? 8U : 0U;
                        ct.param1 = bpad;
                        ct.param2 = pitch;
                        ct.param3 = bpad / 16U;
                        ct.param4 = bpad;
                        ct.param5 = sizeof(Acc) == 4U ? (pitch / 8U)
                                                      : (pitch / 16U);
                        LocalTensor<uint8_t> ctTmp =
                            narrowBwdBuf_.template Get<int32_t>()
                                .template ReinterpretCast<uint8_t>();
                        if constexpr (sizeof(Acc) == 4U) {
                            Transpose(acc[r0 * bpad].template ReinterpretCast<float>(),
                                      stage.template ReinterpretCast<float>(),
                                      ctTmp,
                                      TransposeType::TRANSPOSE_ND2ND_ONLY, ct);
                        } else {
                            Transpose(acc[r0 * bpad].template ReinterpretCast<half>(),
                                      stage.template ReinterpretCast<half>(),
                                      ctTmp,
                                      TransposeType::TRANSPOSE_ND2ND_ONLY, ct);
                        }
                    } else {
                        Gather(acc[r0 * bpad], stage, fwd, 0U, rows * bpad);
                    }
                    PipeBarrier<PIPE_ALL>();
                    r0 += rows;
                }

                uint32_t idxHalf = 0U;
                bool stageBusy = false;

                uint64_t jA = 0U;
                uint32_t lead = 0U;
                if (period > 1U) {
                    const uint64_t ph = (ob * indexCount_ * esz) % phaseMod;
                    jA = ((phaseMod - ph) % phaseMod) / esz;
                    if (jA != 0U) {
                        lead = phaseMod / esz;
                        if (lead < 16U) { lead = 16U; }
                    }
                }
                const uint64_t lastPlane =
                    ob + static_cast<uint64_t>(bReal - 1U) * period;
                const int64_t jStart = static_cast<int64_t>(jBegin + jA) -
                    static_cast<int64_t>(lead);
                for (int64_t j0 = jStart, wEnd = jStart;
                     j0 < static_cast<int64_t>(jEnd); j0 = wEnd) {
                    const uint32_t wlen =
                        (lead != 0U && j0 < static_cast<int64_t>(jA)) ? lead : chunk;
                    wEnd = j0 + static_cast<int64_t>(wlen);
                    const uint64_t vs = j0 < 0 ? 0U : static_cast<uint64_t>(j0);
                    const uint64_t ve = wEnd < static_cast<int64_t>(jEnd) ?
                        static_cast<uint64_t>(wEnd) : jEnd;

                    const uint32_t cnt = static_cast<uint32_t>(ve - vs);
                    const uint32_t lo = static_cast<uint32_t>(static_cast<int64_t>(vs) - j0);
                    const uint32_t rlen = static_cast<uint32_t>(static_cast<int64_t>(ve) - j0);

                    const uint32_t rlenRead = period > 1U ?
                        AlignUpBlockElems(rlen * esz) / esz : rlen;
                    if (stageBusy) {
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
                        stageBusy = false;
                    }
                    LocalTensor<int32_t> idx = idxAll[idxHalf * kIndexTile];
                    const DataCopyExtParams icp(1, cnt * sizeof(int32_t), 0, 0, 0);
                    DataCopyPad(idx, index_[vs], icp, ipad);
                    const uint32_t spitch = AlignUpBlockElems(rlenRead * esz) / esz;
                    const uint32_t prevPitch = curPitch;

                    if ((bpad % 16U) != 0U) {
                        EnsureFwd(spitch, bpad, curPitch, curBpad);
                    }

                    if (curPitch != prevPitch) { zeroedSpitch = 0U; }
                    const uint64_t srcBase = static_cast<uint64_t>(
                        static_cast<int64_t>(ob * indexCount_) + j0);

                    const bool exactTail = rlenRead != rlen && lastPlane + 1U == outer_;
                    const uint32_t alignedRows = exactTail ? bReal - 1U : bReal;
                    if (alignedRows != 0U) {
                        const DataCopyExtParams scp(alignedRows, rlenRead * esz,
                            (period * indexCount_ - rlenRead) * esz, 0, 0);
                        DataCopyPad(stage, source_[srcBase], scp, pad);
                    }
                    if (exactTail) {
                        const DataCopyExtParams tcp(1, rlen * esz, 0, 0, 0);
                        DataCopyPad(stage[(bReal - 1U) * spitch],
                                    source_[srcBase + static_cast<uint64_t>(bReal - 1U) *
                                            period * indexCount_], tcp, pad);
                    }

                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    if (bReal < bpad && zeroedSpitch != spitch) {
                        Duplicate(stage[bReal * spitch],
                                  static_cast<Acc>(0),
                                  static_cast<int32_t>((bpad - bReal) * spitch));
                        PipeBarrier<PIPE_V>();
                        zeroedSpitch = spitch;
                    }

                    if ((bpad % 16U) == 0U) {
                        AscendC::tiling::ConfusionTransposeTiling ct;
                        ct.param0 = sizeof(Acc) == 4U ? 8U : 0U;
                        ct.param1 = bpad;
                        ct.param2 = spitch;
                        ct.param3 = bpad / 16U;
                        ct.param4 = bpad;
                        ct.param5 = sizeof(Acc) == 4U ? (spitch / 8U)
                                                      : (spitch / 16U);
                        LocalTensor<uint8_t> ctTmp =
                            narrowBwdBuf_.template Get<int32_t>()
                                .template ReinterpretCast<uint8_t>();
                        if constexpr (sizeof(Acc) == 4U) {
                            Transpose(srcT.template ReinterpretCast<float>(),
                                      stage.template ReinterpretCast<float>(),
                                      ctTmp,
                                      TransposeType::TRANSPOSE_ND2ND_ONLY, ct);
                        } else {
                            Transpose(srcT.template ReinterpretCast<half>(),
                                      stage.template ReinterpretCast<half>(),
                                      ctTmp,
                                      TransposeType::TRANSPOSE_ND2ND_ONLY, ct);
                        }
                    } else {
                        Gather(srcT, stage, fwd, 0U, rlen * bpad);
                    }

                    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
                    stageBusy = true;
                    PipeBarrier<PIPE_V>();

                    {

                        const uint64_t maskLow = bpad >= 64U ?
                            0xFFFFFFFFFFFFFFFFULL :
                            ((1ULL << bpad) - 1ULL);
                        const uint64_t maskHigh = bpad >= 128U ?
                            0xFFFFFFFFFFFFFFFFULL :
                            (bpad > 64U ? ((1ULL << (bpad - 64U)) - 1ULL) : 0ULL);
                        SetVectorMask<Acc, MaskMode::NORMAL>(maskHigh, maskLow);
                    }
                    BinaryRepeatParams brp;
                    brp.dstBlkStride = 1;
                    brp.src0BlkStride = 1;
                    brp.src1BlkStride = 1;
                    brp.dstRepStride = 8;
                    brp.src0RepStride = 8;
                    brp.src1RepStride = 8;

                    LocalTensor<int64_t> idx64 =
                        idx.template ReinterpretCast<int64_t>();
                    uint32_t j = 0U;
                    for (; j + 4U <= cnt; j += 4U) {
                        const uint64_t w0 =
                            static_cast<uint64_t>(idx64.GetValue(j >> 1U));
                        const uint64_t w1 =
                            static_cast<uint64_t>(idx64.GetValue((j >> 1U) + 1U));
                        const uint32_t r0 = static_cast<uint32_t>(w0 & 0xFFFFFFFFULL);
                        const uint32_t r1 = static_cast<uint32_t>(w0 >> 32U);
                        const uint32_t r2 = static_cast<uint32_t>(w1 & 0xFFFFFFFFULL);
                        const uint32_t r3 = static_cast<uint32_t>(w1 >> 32U);
                        if (r0 < axisLen_) {
                            Add<Acc, false>(acc[r0 * bpad], acc[r0 * bpad],
                                            srcT[(lo + j) * bpad], MASK_PLACEHOLDER, 1, brp);
                        }
                        if (r1 < axisLen_) {
                            Add<Acc, false>(acc[r1 * bpad], acc[r1 * bpad],
                                            srcT[(lo + j + 1U) * bpad], MASK_PLACEHOLDER, 1, brp);
                        }
                        if (r2 < axisLen_) {
                            Add<Acc, false>(acc[r2 * bpad], acc[r2 * bpad],
                                            srcT[(lo + j + 2U) * bpad], MASK_PLACEHOLDER, 1, brp);
                        }
                        if (r3 < axisLen_) {
                            Add<Acc, false>(acc[r3 * bpad], acc[r3 * bpad],
                                            srcT[(lo + j + 3U) * bpad], MASK_PLACEHOLDER, 1, brp);
                        }
                    }
                    for (; j < cnt; ++j) {
                        const uint32_t row =
                            static_cast<uint32_t>(idx.GetValue(j));
                        if (row >= axisLen_) { continue; }

                        Add<Acc, false>(acc[row * bpad], acc[row * bpad],
                                        srcT[(lo + j) * bpad], MASK_PLACEHOLDER,
                                        1, brp);

                    }
                    ResetMask();
                    idxHalf ^= 1U;
                }
                if (stageBusy) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
                    stageBusy = false;
                }

                PipeBarrier<PIPE_ALL>();

                for (uint32_t r0 = 0; r0 < axis; ) {
                    const uint32_t rows = r0 == 0U ? rowHead : rowChunk;
                    const uint32_t pitch =
                        AlignUpBlockElems(rows * esz) / esz;

                    const bool hwBack = (pitch % 16U) == 0U && (bpad % 16U) == 0U;
                    if (hwBack) {
                        AscendC::tiling::ConfusionTransposeTiling cb;
                        cb.param0 = sizeof(Acc) == 4U ? 8U : 0U;
                        cb.param1 = pitch;
                        cb.param2 = bpad;
                        cb.param3 = pitch / 16U;
                        cb.param4 = pitch;
                        cb.param5 = sizeof(Acc) == 4U ? (bpad / 8U)
                                                      : (bpad / 16U);
                        LocalTensor<uint8_t> cbTmp =
                            narrowFwdBuf_.template Get<int32_t>()
                                .template ReinterpretCast<uint8_t>();
                        if constexpr (sizeof(Acc) == 4U) {
                            Transpose(stage.template ReinterpretCast<float>(),
                                      acc[r0 * bpad].template ReinterpretCast<float>(),
                                      cbTmp, TransposeType::TRANSPOSE_ND2ND_ONLY, cb);
                        } else {
                            Transpose(stage.template ReinterpretCast<half>(),
                                      acc[r0 * bpad].template ReinterpretCast<half>(),
                                      cbTmp, TransposeType::TRANSPOSE_ND2ND_ONLY, cb);
                        }
                    } else {
                        EnsureBwd(pitch, bpad, curBwdPitch, curBwdBpad);
                        Gather(stage, acc[r0 * bpad], bwd, 0U, bpad * pitch);
                    }
                    PipeBarrier<PIPE_ALL>();
                    const DataCopyExtParams cp(
                        bReal, rows * esz, 0, (period * axisLen_ - rows) * esz, 0);
                    if (shards > 1U) { SetAtomicAdd<T>(); }
                    DataCopyPad(output_[ob * axisLen_ + r0], stage, cp);
                    if (shards > 1U) { SetAtomicNone(); }
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    r0 += rows;
                }

                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ResidentProcess()
    {
        if constexpr (SameType<T, float>::value || SameType<T, half>::value || SameType<T, bfloat16_t>::value) {
            const uint64_t blocks = static_cast<uint64_t>(blockDim_);
            const uint64_t block = LogicalBlockIndex();
            const uint32_t innerLen = static_cast<uint32_t>(inner_);
            const uint32_t pitch = Pitch(innerLen);
            uint32_t maxBatch = kSrcElements / pitch;
            if (maxBatch > kMaxBatch) {
                maxBatch = kMaxBatch;
            }
            if (maxBatch == 0) {
                return;
            }

            LocalTensor<int32_t> indices = idxBuf_.template Get<int32_t>();
            LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
            const DataCopyPadExtParams<int32_t> indexPad;
            const DataCopyPadExtParams<T> sourcePad;
            for (uint64_t outerIndex = block; outerIndex < outer_; outerIndex += blocks) {
                const uint64_t selfBase = outerIndex * axisLen_ * inner_;
                const uint64_t sourceBase = outerIndex * indexCount_ * inner_;
                LoadSelfBlock(selfBase, static_cast<uint32_t>(axisLen_), innerLen, pitch);

                for (uint64_t base = 0; base < indexCount_; base += maxBatch) {
                    const uint32_t count = static_cast<uint32_t>(
                        (indexCount_ - base) < maxBatch ? (indexCount_ - base) : maxBatch);
                    const DataCopyExtParams indexCopy(
                        1, count * sizeof(int32_t), 0, 0, 0);
                    DataCopyPad(indices, index_[base], indexCopy, indexPad);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

                    LocalTensor<T> sourceStaging;
                    if constexpr (kDirect) {
                        sourceStaging = srcAccBuf_.template Get<T>();
                    } else {
                        sourceStaging = srcTBuf_.template Get<T>();
                    }
                    const DataCopyExtParams sourceCopy(
                        count, innerLen * sizeof(T), 0, 0, 0);
                    DataCopyPad(sourceStaging, source_[sourceBase + base * inner_],
                                sourceCopy, sourcePad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

                    LocalTensor<Acc> sourceAcc = srcAccBuf_.template Get<Acc>();
                    if constexpr (!kDirect) {
                        ToAccumulator(sourceAcc, sourceStaging, count * pitch);
                    }
                    for (uint32_t j = 0; j < count; ++j) {
                        const uint32_t row = static_cast<uint32_t>(indices.GetValue(j));
                        if (row >= axisLen_) {
                            continue;
                        }
                        Add(acc[row * pitch], acc[row * pitch], sourceAcc[j * pitch],
                            static_cast<int32_t>(innerLen));
                        if constexpr (kIsBf16) {
                            LocalTensor<T> rounded = roundBuf_.template Get<T>();
                            PipeBarrier<PIPE_V>();
                            Cast(rounded, acc[row * pitch], RoundMode::CAST_RINT,
                                 static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                            Cast(acc[row * pitch], rounded, RoundMode::CAST_NONE,
                                 static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                }
                StoreBlock(selfBase, static_cast<uint32_t>(axisLen_), innerLen,
                           pitch, static_cast<uint32_t>(axisLen_ * pitch));
            }
        }
    }

    __aicore__ inline void WrapToInt8(const LocalTensor<Acc>& acc, const LocalTensor<T>& out, uint32_t count)
    {
        if constexpr (kIsInt8) {
            const int32_t n = static_cast<int32_t>(count);
            LocalTensor<float> value = acc.template ReinterpretCast<float>();
            LocalTensor<float> scratch = wrapFBuf_.template Get<float>();
            LocalTensor<int32_t> whole = wrapIBuf_.template Get<int32_t>();
            LocalTensor<half> mid = halfBuf_.template Get<half>();
            Adds(scratch, value, 128.0f, n);
            PipeBarrier<PIPE_V>();
            Muls(scratch, scratch, 1.0f / 256.0f, n);
            PipeBarrier<PIPE_V>();
            Cast(whole, scratch, RoundMode::CAST_FLOOR, n);
            PipeBarrier<PIPE_V>();
            Cast(scratch, whole, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Muls(scratch, scratch, -256.0f, n);
            PipeBarrier<PIPE_V>();
            Add(scratch, value, scratch, n);
            PipeBarrier<PIPE_V>();
            Cast(mid, scratch, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Cast(out, mid, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Phase1Bucket(uint64_t P, uint64_t k)
    {
        if constexpr (SameType<T, float>::value || SameType<T, half>::value || SameType<T, bfloat16_t>::value) {
            const uint64_t S = numSlices_;
            const uint64_t cap = bucketCap_;
            const uint32_t shift = static_cast<uint32_t>(sliceShift_);
            const uint64_t iBegin = k * cap;
            uint64_t iEnd = iBegin + cap;
            if (iEnd > indexCount_) {
                iEnd = indexCount_;
            }
            uint32_t cnt[256];
            for (uint64_t s = 0; s < S; ++s) {
                cnt[s] = 0;
            }
            LocalTensor<int32_t> idx = idxBuf_.template Get<int32_t>();
            const DataCopyPadExtParams<int32_t> pd;

            for (uint64_t base = iBegin; base < iEnd; base += kIndexTile) {
                const uint32_t c =
                    static_cast<uint32_t>((iEnd - base) < kIndexTile ? (iEnd - base) : kIndexTile);
                const DataCopyExtParams cp(1, c * sizeof(int32_t), 0, 0, 0);
                DataCopyPad(idx, index_[base], cp, pd);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t j = 0; j < c; ++j) {
                    const uint32_t s = static_cast<uint32_t>(idx.GetValue(j)) >> shift;
                    ++cnt[s];
                }
            }
            uint32_t prefix[256];
            uint32_t w[256];
            uint32_t running = 0;
            for (uint64_t s = 0; s < S; ++s) {

                running = ((running + 3U) / 4U) * 4U;
                prefix[s] = running;
                running += cnt[s];
                w[s] = 0;
            }

            LocalTensor<int32_t> stage = srcAccBuf_.template Get<int32_t>();
            for (uint64_t base = iBegin; base < iEnd; base += kIndexTile) {
                const uint32_t c =
                    static_cast<uint32_t>((iEnd - base) < kIndexTile ? (iEnd - base) : kIndexTile);
                const DataCopyExtParams cp(1, c * sizeof(int32_t), 0, 0, 0);
                DataCopyPad(idx, index_[base], cp, pd);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t j = 0; j < c; ++j) {
                    const uint32_t t = static_cast<uint32_t>(idx.GetValue(j));
                    const uint32_t s = t >> shift;
                    const uint32_t localRow = t - (s << shift);
                    const uint32_t e = prefix[s] + w[s];
                    ++w[s];
                    stage.SetValue(2 * e, static_cast<int32_t>(localRow));
                    stage.SetValue(2 * e + 1, static_cast<int32_t>(base + j));
                }
            }

            LocalTensor<int32_t> cntUB = bcntBuf_.template Get<int32_t>();
            for (uint64_t s = 0; s < S; ++s) {
                cntUB.SetValue(s, static_cast<int32_t>(cnt[s]));
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            {
                const DataCopyExtParams cp(1, static_cast<uint32_t>(S * sizeof(int32_t)), 0, 0, 0);
                DataCopyPad(countsGm_[k * S], cntUB, cp);
            }

            for (uint64_t s = 0; s < S; ++s) {
                if (cnt[s] == 0) {
                    continue;
                }
                const DataCopyExtParams cp(1, cnt[s] * 2U * sizeof(int32_t), 0, 0, 0);
                DataCopyPad(bucketGm_[(k * S + s) * cap * 2], stage[prefix[s] * 2], cp);
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void Phase2Bucket(uint64_t P, uint64_t k)
    {
        if constexpr (SameType<T, float>::value || SameType<T, half>::value || SameType<T, bfloat16_t>::value) {
            const uint64_t S = numSlices_;
            const uint64_t cap = bucketCap_;
            const uint64_t sr = sliceRows_;
            const uint32_t innerLen = static_cast<uint32_t>(inner_);
            const uint32_t pitch = Pitch(innerLen);
            LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
            LocalTensor<int32_t> ent = idxBuf_.template Get<int32_t>();
            LocalTensor<int32_t> counts = countsAllBuf_.template Get<int32_t>();
            const DataCopyPadExtParams<T> pdf;
            const DataCopyPadExtParams<int32_t> pdi;

            {
                const DataCopyExtParams cp(1, static_cast<uint32_t>(P * S * sizeof(int32_t)), 0, 0, 0);
                DataCopyPad(counts, countsGm_[0], cp, pdi);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            }
            const uint64_t phase2Tasks = outer_ * S;
            for (uint64_t task = k; task < phase2Tasks; task += P) {
                const uint64_t outerIndex = task / S;
                const uint64_t s = task - outerIndex * S;
                const uint64_t rowBegin = s * sr;
                const uint32_t rows =
                    static_cast<uint32_t>((axisLen_ - rowBegin) < sr ? (axisLen_ - rowBegin) : sr);
                if (rows == 0) {
                    continue;
                }
                const uint64_t selfBase =
                    outerIndex * axisLen_ * inner_ + rowBegin * inner_;
                const uint64_t sourceBase = outerIndex * indexCount_ * inner_;
                {
                    const DataCopyExtParams cp(rows, innerLen * sizeof(T), 0, 0, 0);
                    if constexpr (kDirect) {
                        DataCopyPad(acc, self_[selfBase], cp, pdf);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    } else {
                        LocalTensor<T> stg = selfTBuf_.template Get<T>();
                        DataCopyPad(stg, self_[selfBase], cp, pdf);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        Cast(acc, stg, RoundMode::CAST_NONE, static_cast<int32_t>(rows * pitch));
                        PipeBarrier<PIPE_V>();
                    }
                }

                SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                for (uint64_t kk = 0; kk < P; ++kk) {
                    const int32_t c = counts.GetValue(kk * S + s);
                    if (c <= 0) {
                        continue;
                    }
                    {
                        const DataCopyExtParams cp(1, static_cast<uint32_t>(c) * 2U * sizeof(int32_t), 0, 0, 0);
                        DataCopyPad(ent, bucketGm_[(kk * S + s) * cap * 2], cp, pdi);
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    }
                    for (int32_t e = 0; e < c; ++e) {
                        uint32_t localRow = static_cast<uint32_t>(ent.GetValue(2 * e));
                        uint32_t pos = static_cast<uint32_t>(ent.GetValue(2 * e + 1));
                        if (localRow >= rows || pos >= indexCount_) {

                            continue;
                        }
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                        const DataCopyExtParams cps(1, innerLen * sizeof(T), 0, 0, 0);
                        if constexpr (kDirect) {
                            LocalTensor<T> src = srcAccBuf_.template Get<T>();
                            DataCopyPad(src, source_[sourceBase + static_cast<uint64_t>(pos) * inner_], cps, pdf);
                            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                            Add(acc[localRow * pitch], acc[localRow * pitch], src,
                                static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                        } else {
                            LocalTensor<T> srcT = srcTBuf_.template Get<T>();
                            LocalTensor<float> srcF = srcAccBuf_.template Get<float>();
                            DataCopyPad(srcT, source_[sourceBase + static_cast<uint64_t>(pos) * inner_], cps, pdf);
                            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                            Cast(srcF, srcT, RoundMode::CAST_NONE, static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();

                            Add(acc[localRow * pitch], acc[localRow * pitch], srcF,
                                static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                            LocalTensor<T> rnd = roundBuf_.template Get<T>();
                            Cast(rnd, acc[localRow * pitch], RoundMode::CAST_RINT,
                                 static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                            Cast(acc[localRow * pitch], rnd, RoundMode::CAST_NONE,
                                 static_cast<int32_t>(innerLen));
                            PipeBarrier<PIPE_V>();
                        }
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    }
                }
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

                {
                    const DataCopyExtParams cp(rows, innerLen * sizeof(T), 0, 0, 0);
                    if constexpr (kDirect) {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        DataCopyPad(output_[selfBase], acc, cp);
                    } else {
                        LocalTensor<T> outT = selfTBuf_.template Get<T>();
                        Cast(outT, acc, RoundMode::CAST_RINT, static_cast<int32_t>(rows * pitch));
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        DataCopyPad(output_[selfBase], outT, cp);
                    }
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                }
            }
        }
    }

    __aicore__ inline void BucketProcess()
    {
        if constexpr (SameType<T, float>::value || SameType<T, half>::value || SameType<T, bfloat16_t>::value) {
            const uint64_t P = static_cast<uint64_t>(blockDim_);
            const uint64_t k = LogicalBlockIndex();
            Phase1Bucket(P, k);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<int32_t> us = syncUb_.template Get<int32_t>();
            SyncAll(syncGm_, us, static_cast<int32_t>(P));
            Phase2Bucket(P, k);
        }
    }

    TPipe pipe_;
    bool ownPath_ = false;
    uint32_t ownBatch_ = 1U;
    uint32_t ownSlotElems_ = 0U;
    uint64_t ownShards_ = 1U;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> srcAccBuf_;
    TBuf<TPosition::VECCALC> selfTBuf_;
    TBuf<TPosition::VECCALC> srcTBuf_;
    TBuf<TPosition::VECCALC> halfBuf_;
    TBuf<TPosition::VECCALC> wrapFBuf_;
    TBuf<TPosition::VECCALC> wrapIBuf_;
    TBuf<TPosition::VECCALC> roundBuf_;
    TBuf<TPosition::VECCALC> atomicIdxBuf_;
    TBuf<TPosition::VECCALC> atomicDataBuf_;
    TBuf<TPosition::VECCALC> syncUb_;
    TBuf<TPosition::VECCALC> bcntBuf_;
    TBuf<TPosition::VECCALC> countsAllBuf_;
    TBuf<TPosition::VECCALC> residentSortedBuf_;
    TBuf<TPosition::VECCALC> residentRowStartBuf_;
    TBuf<TPosition::VECCALC> residentCursorBuf_;
    TBuf<TPosition::VECCALC> narrowStageBuf_;
    TBuf<TPosition::VECCALC> narrowFwdBuf_;
    TBuf<TPosition::VECCALC> narrowBwdBuf_;
    GlobalTensor<int32_t> syncGm_;
    GlobalTensor<int32_t> residentSortedGm_;
    GlobalTensor<int32_t> residentStartsGm_;
    GlobalTensor<T> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<T> source_;
    GlobalTensor<T> output_;
    uint64_t outer_;
    uint64_t axisLen_;
    uint64_t inner_;
    uint64_t indexCount_;
    uint64_t innerChunk_;
    uint64_t innerChunkCount_;
    uint64_t axisChunk_;
    uint64_t axisChunkCount_;
    uint64_t taskCount_;

    GlobalTensor<int32_t> countsGm_;
    GlobalTensor<int32_t> bucketGm_;
    uint64_t numSlices_;
    uint64_t sliceRows_;
    uint64_t sliceShift_;
    uint64_t bucketCap_;
    uint64_t bucketCores_;
    uint32_t useBucket_;
    uint32_t useResident_;
    uint32_t blockDim_;
    uint32_t singleSubblock_;

    __aicore__ inline void CopyRange(uint64_t total, uint64_t blocks, uint64_t block)
    {
        const uint64_t share = total / blocks;
        const uint64_t extra = total % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        LocalTensor<T> buf = atomicDataBuf_.template Get<T>();
        const DataCopyPadExtParams<T> pd;

        if (begin < end) {
            const int32_t ev2[2] = {EVENT_ID0, EVENT_ID1};
            bool wpend[2] = {false, false};
            uint64_t pos = begin;
            uint32_t cur = 0U;
            {
                const uint32_t c0 = static_cast<uint32_t>(
                    (end - pos) < atomicTile_ ? (end - pos) : atomicTile_);
                const DataCopyExtParams cp0(1, c0 * sizeof(T), 0, 0, 0);
                DataCopyPad(buf, self_[pos], cp0, pd);
                SetFlag<HardEvent::MTE2_MTE3>(ev2[0]);
            }
            while (pos < end) {
                const uint32_t cnt = static_cast<uint32_t>(
                    (end - pos) < atomicTile_ ? (end - pos) : atomicTile_);
                const uint64_t nxtPos = pos + cnt;
                const uint32_t nxt = cur ^ 1U;
                WaitFlag<HardEvent::MTE2_MTE3>(ev2[cur]);

                if (nxtPos < end) {
                    const uint32_t ncnt = static_cast<uint32_t>(
                        (end - nxtPos) < atomicTile_ ? (end - nxtPos)
                                                     : atomicTile_);
                    if (wpend[nxt]) {
                        WaitFlag<HardEvent::MTE3_MTE2>(ev2[nxt]);
                        wpend[nxt] = false;
                    }
                    const DataCopyExtParams ncp(1, ncnt * sizeof(T), 0, 0, 0);
                    DataCopyPad(buf[nxt * atomicTile_], self_[nxtPos], ncp, pd);
                    SetFlag<HardEvent::MTE2_MTE3>(ev2[nxt]);
                }
                const DataCopyExtParams cp(1, cnt * sizeof(T), 0, 0, 0);
                DataCopyPad(output_[pos], buf[cur * atomicTile_], cp);
                SetFlag<HardEvent::MTE3_MTE2>(ev2[cur]);
                wpend[cur] = true;
                pos = nxtPos;
                cur = nxt;
            }
            for (uint32_t d = 0U; d < 2U; ++d) {
                if (wpend[d]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(ev2[d]);
                    wpend[d] = false;
                }
            }
        }
    }

    __aicore__ inline void CopyRangeOwn(uint64_t total, uint64_t blocks,
                                        uint64_t block)
    {
      if constexpr (kDirect && kIsInteger) {
        if (total == 0U || blocks == 0U) { return; }
        const uint64_t share = total / blocks;
        const uint64_t extra = total % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        if (begin >= end) { return; }
        LocalTensor<Acc> buf = srcAccBuf_.template Get<Acc>();
        const DataCopyPadExtParams<T> pd;
        const uint32_t tile = ownSlotElems_;
        const int32_t ev2[2] = {EVENT_ID4, EVENT_ID5};
        bool wpend[2] = {false, false};
        uint64_t pos = begin;
        uint32_t cur = 0U;
        {
            const uint64_t left0 = end - pos;
            const uint32_t c0 = static_cast<uint32_t>(left0 < tile ? left0 : tile);
            const DataCopyExtParams cp0(1, c0 * sizeof(T), 0, 0, 0);
            DataCopyPad(buf, self_[pos], cp0, pd);
            SetFlag<HardEvent::MTE2_MTE3>(ev2[0]);
        }
        while (pos < end) {
            const uint64_t left = end - pos;
            const uint32_t cnt = static_cast<uint32_t>(left < tile ? left : tile);
            const uint64_t nxtPos = pos + cnt;
            const uint32_t nxt = cur ^ 1U;
            WaitFlag<HardEvent::MTE2_MTE3>(ev2[cur]);
            if (nxtPos < end) {
                const uint64_t left2 = end - nxtPos;
                const uint32_t nc =
                    static_cast<uint32_t>(left2 < tile ? left2 : tile);
                const DataCopyExtParams ncp(1, nc * sizeof(T), 0, 0, 0);
                if (wpend[nxt]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(ev2[nxt]);
                    wpend[nxt] = false;
                }
                DataCopyPad(buf[nxt * tile], self_[nxtPos], ncp, pd);
                SetFlag<HardEvent::MTE2_MTE3>(ev2[nxt]);
            }
            const DataCopyExtParams cp(1, cnt * sizeof(T), 0, 0, 0);
            DataCopyPad(output_[pos], buf[cur * tile], cp);
            SetFlag<HardEvent::MTE3_MTE2>(ev2[cur]);
            wpend[cur] = true;
            pos = nxtPos;
            cur = nxt;
        }
        for (uint32_t d = 0U; d < 2U; ++d) {
            if (wpend[d]) {
                WaitFlag<HardEvent::MTE3_MTE2>(ev2[d]);
                wpend[d] = false;
            }
        }
      }
    }

    __aicore__ inline void IntegerOwnedProcess()
    {

      if constexpr (kDirect && kIsInteger) {
        const uint64_t blocks = static_cast<uint64_t>(blockDim_);
        const uint64_t block = LogicalBlockIndex();
        if (taskCount_ == 0U || blocks == 0U) { return; }
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        const bool sharded = ownShards_ > 1U;
        if (sharded) {

            CopyRangeOwn(outer_ * axisLen_ * inner_, blocks, block);
            SyncAll();
        }
        if (begin >= end) { return; }

        LocalTensor<int32_t> idx = idxBuf_.template Get<int32_t>();
        {
            const DataCopyPadExtParams<int32_t> pd;
            const DataCopyExtParams cp(
                1, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)), 0, 0, 0);
            DataCopyPad(idx, index_[0], cp, pd);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        LocalTensor<Acc> acc = accBuf_.template Get<Acc>();
        LocalTensor<Acc> src = srcAccBuf_.template Get<Acc>();
        const DataCopyPadExtParams<T> pad;
        const int32_t ev[2] = {EVENT_ID2, EVENT_ID3};
        if (sharded) { SetAtomicAdd<T>(); }

        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t perOuter = innerChunkCount_ * ownShards_;
            const uint64_t outerIndex = task / perOuter;
            const uint64_t rest = task - outerIndex * perOuter;
            const uint64_t innerIndex = rest / ownShards_;
            const uint64_t shard = rest - innerIndex * ownShards_;
            const uint64_t jShare = indexCount_ / ownShards_;
            const uint64_t jExtra = indexCount_ % ownShards_;
            const uint64_t jBegin =
                shard * jShare + (shard < jExtra ? shard : jExtra);
            const uint64_t jEnd =
                jBegin + jShare + (shard < jExtra ? 1U : 0U);
            const uint64_t innerBegin = innerIndex * innerChunk_;
            if (innerBegin >= inner_) { continue; }
            const uint64_t left = inner_ - innerBegin;
            const uint32_t chunkLen = static_cast<uint32_t>(
                left < innerChunk_ ? left : innerChunk_);
            const uint32_t pitch = Pitch(chunkLen);
            const uint32_t rows = static_cast<uint32_t>(axisLen_);
            const uint64_t selfBase =
                outerIndex * axisLen_ * inner_ + innerBegin;
            const uint64_t srcBase =
                outerIndex * indexCount_ * inner_ + innerBegin;

            if (sharded) {
                Duplicate(acc, static_cast<Acc>(0),
                          static_cast<int32_t>(rows * pitch));
                PipeBarrier<PIPE_V>();
            } else {
                const DataCopyExtParams cp(
                    rows, chunkLen * sizeof(T),
                    (inner_ - chunkLen) * sizeof(T), 0, 0);
                DataCopyPad(acc, self_[selfBase], cp, pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }

            uint64_t jIssue = jBegin;
            uint64_t jUse = jBegin;
            uint32_t sIssue = 0U;
            uint32_t sUse = 0U;
            uint32_t inflight = 0U;
            bool war[2] = {false, false};
            uint32_t batchRows[2] = {0U, 0U};
            uint64_t batchJ[2] = {0U, 0U};
            while (jUse < jEnd) {
                while (inflight < 2U && jIssue < jEnd) {
                    uint64_t bb = jEnd - jIssue;
                    if (bb > ownBatch_) { bb = ownBatch_; }
                    const uint32_t B = static_cast<uint32_t>(bb);
                    if (war[sIssue]) {
                        WaitFlag<HardEvent::V_MTE2>(ev[sIssue]);
                        war[sIssue] = false;
                    }
                    const DataCopyExtParams scp(
                        B, chunkLen * sizeof(T),
                        (inner_ - chunkLen) * sizeof(T), 0, 0);
                    DataCopyPad(src[sIssue * ownSlotElems_],
                                source_[srcBase + jIssue * inner_], scp, pad);
                    SetFlag<HardEvent::MTE2_V>(ev[sIssue]);
                    batchRows[sIssue] = B;
                    batchJ[sIssue] = jIssue;
                    jIssue += B;
                    sIssue ^= 1U;
                    ++inflight;
                }
                WaitFlag<HardEvent::MTE2_V>(ev[sUse]);
                const uint32_t B = batchRows[sUse];
                const uint64_t j0 = batchJ[sUse];
                LocalTensor<Acc> cur = src[sUse * ownSlotElems_];
                for (uint32_t b = 0U; b < B; ++b) {
                    const uint32_t row =
                        static_cast<uint32_t>(idx.GetValue(j0 + b));
                    Add(acc[row * pitch], acc[row * pitch], cur[b * pitch],
                        static_cast<int32_t>(chunkLen));
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(ev[sUse]);
                war[sUse] = true;
                jUse += B;
                sUse ^= 1U;
                --inflight;
            }
            for (uint32_t d = 0U; d < 2U; ++d) {
                if (war[d]) {
                    WaitFlag<HardEvent::V_MTE2>(ev[d]);
                    war[d] = false;
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams ocp(
                rows, chunkLen * sizeof(T), 0,
                (inner_ - chunkLen) * sizeof(T), 0);

            DataCopyPad(output_[selfBase], acc, ocp);

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        if (sharded) { SetAtomicNone(); }
      }
    }

    __aicore__ inline void AtomicProcess()
    {


        const uint64_t selfBytes = outer_ * axisLen_ * inner_ * sizeof(T);
        if (selfBytes >= (1ULL << 20)) {
            self_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (inner_ * sizeof(T) >= 512U) {
            source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        const uint64_t blocks = static_cast<uint64_t>(blockDim_);
        const uint64_t block = LogicalBlockIndex();

        CopyRange(outer_ * axisLen_ * inner_, blocks, block);

        SyncAll();

        LocalTensor<int32_t> idx = atomicIdxBuf_.template Get<int32_t>();
        if (indexCount_ > 0) {
            const DataCopyPadExtParams<int32_t> pd;
            const DataCopyExtParams cp(1, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)), 0, 0, 0);
            DataCopyPad(idx, index_[0], cp, pd);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }

        const uint64_t total = outer_ * indexCount_;
        const uint64_t share = total / blocks;
        const uint64_t extra = total % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        LocalTensor<T> buf = atomicDataBuf_.template Get<T>();
        const DataCopyPadExtParams<T> pd;
        SetAtomicAdd<T>();

        uint64_t o = indexCount_ != 0 ? begin / indexCount_ : 0;
        uint64_t j = indexCount_ != 0 ? begin % indexCount_ : 0;
        uint64_t srcRowBase = o * indexCount_ * inner_ + j * inner_;
        uint64_t outRowBase = o * axisLen_ * inner_;

        uint32_t slot = 0U;
        uint32_t issued = 0U;

        const uint32_t innerBytes = static_cast<uint32_t>(inner_) * sizeof(T);

        const uint32_t pitchBytes = ((innerBytes + 31U) / 32U) * 32U;
        const uint32_t pitchElems =
            pitchBytes / static_cast<uint32_t>(sizeof(T));
        const uint32_t rowsPerBatch =
            (inner_ != 0 && pitchElems != 0U && pitchElems <= atomicTile_) ?
            (atomicTile_ / pitchElems) : 1U;
        if (rowsPerBatch >= 2U) {

            static constexpr uint32_t kAlignedRowsMax = 32U;
            const uint32_t eszB = static_cast<uint32_t>(sizeof(T));

            const bool alignedRows = kIsInteger && sizeof(T) >= 2U && (innerBytes % 32U) != 0U;
            const uint32_t slotElems = (pitchBytes + 32U) / eszB;
            uint32_t alignedCap = atomicTile_ / slotElems;
            if (alignedCap > kAlignedRowsMax) { alignedCap = kAlignedRowsMax; }
            if (alignedCap == 0U) { alignedCap = 1U; }
            const uint64_t totalElems = outer_ * indexCount_ * inner_;
            const int32_t evS[2] = {EVENT_ID0, EVENT_ID1};
            uint32_t rowAt[kAlignedRowsMax];
            uint32_t phAt[kAlignedRowsMax];
            uint64_t task = begin;
            while (task < end) {
                uint64_t runRows = indexCount_ - j;
                if (runRows > (end - task)) { runRows = end - task; }
                if (runRows > rowsPerBatch) { runRows = rowsPerBatch; }
                if (alignedRows && runRows > alignedCap) { runRows = alignedCap; }
                const uint32_t rr = static_cast<uint32_t>(runRows);

                bool aligned = alignedRows;
                if (aligned) {
                    for (uint32_t t = 0; t < rr; ++t) {
                        rowAt[t] = static_cast<uint32_t>(idx.GetValue(j + t));
                        const uint64_t e0 = srcRowBase + static_cast<uint64_t>(t) * inner_;
                        phAt[t] = static_cast<uint32_t>((e0 * eszB) % 32U);
                        const uint64_t dElem = outRowBase + static_cast<uint64_t>(rowAt[t]) * inner_;
                        if (dElem < phAt[t] / eszB) { aligned = false; }
                    }
                }

                const DataCopyExtParams rcp(rr, innerBytes, 0, 0, 0);
                if (issued >= 2U) {
                    WaitFlag<HardEvent::MTE3_MTE2>(evS[slot]);
                }
                LocalTensor<T> cur = buf[slot * atomicTile_];
                if (aligned) {
                    for (uint32_t t = 0; t < rr; ++t) {
                        const uint64_t e0 = srcRowBase + static_cast<uint64_t>(t) * inner_;
                        const uint32_t ph = phAt[t];
                        uint32_t len = ((ph + innerBytes + 31U) / 32U) * 32U;

                        if (e0 + (len - ph) / eszB > totalElems) { len = ph + innerBytes; }
                        const DataCopyExtParams acp(1, len, 0, 0, 0);
                        DataCopyPad(cur[t * slotElems], source_[e0 - ph / eszB], acp, pd);
                    }
                    SetFlag<HardEvent::MTE2_V>(evS[slot]);
                    WaitFlag<HardEvent::MTE2_V>(evS[slot]);

                    if constexpr (sizeof(T) >= 2U) {
                        for (uint32_t t = 0; t < rr; ++t) {
                            if (phAt[t] != 0U) {
                                Duplicate(cur[t * slotElems], static_cast<T>(0),
                                          static_cast<int32_t>(phAt[t] / eszB));
                            }
                        }
                    }
                    SetFlag<HardEvent::V_MTE3>(evS[slot]);
                    WaitFlag<HardEvent::V_MTE3>(evS[slot]);
                    for (uint32_t t = 0; t < rr; ++t) {
                        const DataCopyExtParams wa(1, phAt[t] + innerBytes, 0, 0, 0);
                        DataCopyPad(output_[outRowBase + static_cast<uint64_t>(rowAt[t]) * inner_ -
                                            phAt[t] / eszB],
                                    cur[t * slotElems], wa);
                    }
                } else {
                    DataCopyPad(cur, source_[srcRowBase], rcp, pd);
                    SetFlag<HardEvent::MTE2_MTE3>(evS[slot]);
                    WaitFlag<HardEvent::MTE2_MTE3>(evS[slot]);
                    const DataCopyExtParams wcp(1, innerBytes, 0, 0, 0);
                    for (uint32_t t = 0; t < rr; ++t) {
                        const uint32_t row =
                            static_cast<uint32_t>(idx.GetValue(j + t));
                        DataCopyPad(output_[outRowBase +
                                        static_cast<uint64_t>(row) * inner_],
                                    cur[t * pitchElems], wcp);
                    }
                }
                SetFlag<HardEvent::MTE3_MTE2>(evS[slot]);
                slot ^= 1U;
                if (issued < 2U) { ++issued; }
                task += runRows;
                j += runRows;
                srcRowBase += runRows * inner_;
                if (j == indexCount_) {
                    j = 0;
                    ++o;
                    srcRowBase = o * indexCount_ * inner_;
                    outRowBase = o * axisLen_ * inner_;
                }
            }
            if (issued >= 2U) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else if (issued == 1U) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            SetAtomicNone();
            return;
        }
        for (uint64_t task = begin; task < end; ++task) {
            const uint32_t row = static_cast<uint32_t>(idx.GetValue(j));
            const uint64_t outBase = outRowBase + static_cast<uint64_t>(row) * inner_;
            for (uint64_t ic = 0; ic < inner_; ic += kAtomicTile) {
                const uint32_t cnt = static_cast<uint32_t>((inner_ - ic) < kAtomicTile ? (inner_ - ic) : kAtomicTile);
                const DataCopyExtParams cp(1, cnt * sizeof(T), 0, 0, 0);
                if (issued >= 2U) {
                    if (slot == 0U) {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    } else {
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                }
                LocalTensor<T> cur = buf[slot * kAtomicTile];
                DataCopyPad(cur, source_[srcRowBase + ic], cp, pd);
                if (slot == 0U) {
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                } else {
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                }
                DataCopyPad(output_[outBase + ic], cur, cp);
                if (slot == 0U) {
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                } else {
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                }
                slot ^= 1U;
                if (issued < 2U) { ++issued; }
            }

            ++j;
            srcRowBase += inner_;
            if (j == indexCount_) {
                j = 0;
                ++o;
                srcRowBase = o * indexCount_ * inner_;
                outRowBase = o * axisLen_ * inner_;
            }
        }

        if (issued >= 2U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else if (issued == 1U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        SetAtomicNone();
    }
};

__aicore__ inline TCubeTiling PrepareIndexAddCubeTiling(
    uint32_t m, uint32_t n, uint32_t k)
{
    TCubeTiling tiling;
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = m;
    tiling.singleCoreN = n;
    tiling.singleCoreK = k;
    tiling.baseM = m < 64U ? m : 64U;
    tiling.baseN = n < 64U ? n : 64U;
    tiling.baseK = 64U;
    tiling.depthA1 = 2;
    tiling.depthB1 = 2;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;
    return tiling;
}

}

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(IndexAddTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (SameType<DTYPE_SELF, half>::value) {

        if ASCEND_IS_AIC {
            return;
        }
        if (tilingData.singleSubblock != 0U && GetSubBlockIdx() != 0U) {
            return;
        }
    }
    KernelIndexAdd<DTYPE_SELF> op;
    op.Init(self, index, source, output, workspace, tilingData);
    op.Process();
}
