// Own runtime-index-derived layout. Included inside the anonymous namespace.
// Caller scopes the sole TPipe before constructing an optional fallback.
class IndexAddHalfSourcePrefetchPrefix {
    static constexpr uint32_t kUbBytes = 188U * 1024U;
    static constexpr uint32_t kMaxHeight = 32U;
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output, const IndexAddTilingData& td,
                                TPipe* pipe)
    {
        axis_ = td.axisLength;
        updates_ = td.indexLength;
        outer_ = td.totalLength / axis_;
        axisStride_ = Align(axis_, 32U);
        sortCount_ = Align(updates_, 64U);
        sourceStride_ = Align(updates_, 16U);
        planeCapacity_ = Align(updates_ + 7U * kMaxHeight, 8U);
        const uint32_t metadata =
            4U * (planeCapacity_ + 3U * axisStride_ + 40U + 32U);
        // B16 control: isolate batch-size cost before separating output
        // lifetime from Gather scratch. The admitted resource family fits
        // this layout throughout its axis/index bounds.
        batch_ = 16U;
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(source));
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(self));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        pipe->InitBuffer(arena_, kUbBytes);
        auto h = arena_.Get<half>();
        sourceUb_ = h;
        columns_ = h[batch_ * sourceStride_];
        accum_ = columns_[batch_ * sourceStride_];
        // Gather/output has its own storage, outside the raw source range.
        planes_ = accum_[batch_ * axisStride_].ReinterpretCast<uint32_t>();
        axes_ = planes_[planeCapacity_];
        inverse_ = axes_[axisStride_];
        work_ = inverse_[axisStride_];
        starts_ = work_[axisStride_];
        active_ = starts_[40U];
        gathered_ = active_[32U].ReinterpretCast<half>();
        // A disjoint row-major buffer removes per-row UB padding before
        // the output DMA. The existing output-generation event also
        // protects this buffer; raw source prefetch never aliases it.
        packedOutput_ = gathered_[batch_ * axisStride_];
        // These tables are used only during Prepare and fit inside future
        // gathered storage (12*axisStride <= 2*batch*axisStride).
        heads_ = gathered_.ReinterpretCast<float>();
        counts_ = heads_[axisStride_];
        otherWork_ = counts_[axisStride_].ReinterpretCast<uint32_t>();
    }

    __aicore__ inline bool Prepare()
    {
        const uint32_t n = sortCount_;
        auto f = arena_.Get<float>();
        auto u = arena_.Get<uint32_t>();
        auto indices = u[4U * n].ReinterpretCast<int32_t>();
        auto scores = f[5U * n];
        auto positions = u[6U * n];
        DataCopyPad(indices, indexGm_, DataCopyExtParams{1U, updates_ * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
        Duplicate(scores, -16777216.0F, n);
        CreateVecIndex(positions.ReinterpretCast<int32_t>(), int32_t{0}, n);
        PipeBarrier<PIPE_V>();
        Cast(f[2U * n], positions.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, updates_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Cast(scores, indices, RoundMode::CAST_NONE, updates_);
        PipeBarrier<PIPE_V>();
        uint32_t keyStride = 1U;
        while (keyStride < updates_) { keyStride <<= 1U; }
        Muls(scores, scores, static_cast<float>(static_cast<int32_t>(keyStride)), updates_);
        PipeBarrier<PIPE_V>();
        Add(scores, scores, f[2U * n], updates_);
        PipeBarrier<PIPE_V>();
        Muls(scores, scores, -1.0F, updates_);
        PipeBarrier<PIPE_V>();
        auto records = SortRecords(f, f[2U * n], scores, positions, n);
        uint64_t retained = 0U;
        auto sortedPositions = scores.ReinterpretCast<uint32_t>();
        GatherMask(sortedPositions, records.ReinterpretCast<uint32_t>(), uint8_t{2}, true,
                   2U * n, {1U, 1U, 0U, 0U}, retained);
        PipeBarrier<PIPE_V>();
        // Convert the sorted position to an index-address offset. Scratch
        // below4*n is disjoint from the index and position arrays.
        auto byteOffsets = u;
        ShiftLeft(byteOffsets, sortedPositions, uint32_t{2}, updates_);
        PipeBarrier<PIPE_V>();
        auto sortedIndices = u[n].ReinterpretCast<int32_t>();
        Gather(sortedIndices, indices, byteOffsets, 0U, updates_);
        PipeBarrier<PIPE_V>();
        auto sortedAxes = positions.ReinterpretCast<float>();
        // Keep one initialized sentinel even when updates_ equals n. The old
        // sort-mask region above 7*n is dead; source DMA has not started.
        Duplicate(sortedAxes, 16777216.0F, n + 8U);
        PipeBarrier<PIPE_V>();
        Cast(sortedAxes, sortedIndices, RoundMode::CAST_NONE, updates_);
        PipeBarrier<PIPE_V>();
        auto csr = u[4U * n];
        DataCopy(csr, sortedPositions, n);
        Duplicate(inverse_, uint32_t{0}, axisStride_);
        // P includes the final real axis's right boundary, and Compare's
        // full 64-float groups. All search scratch lies below CSR at 4*n.
        const uint32_t p = Align(axis_ + 1U, 64U);
        auto lower = f;
        auto targets = f[p];
        auto probeOffsets = f[2U * p];
        auto offsetWords = u[3U * p];
        auto probes = f[4U * p];
        auto advanced = f[5U * p];
        auto searchMask = u[6U * p].ReinterpretCast<uint8_t>();
        Duplicate(lower, 0.0F, p);
        CreateVecIndex(targets, 0.0F, p);
        PipeBarrier<PIPE_V>();
        // Binary lifting computes lower_bound(axis) for every axis together.
        // Clamping to the +2^24 sentinel makes every Gather address legal.
        uint32_t step = 1U;
        while (step <= updates_ / 2U) { step <<= 1U; }
        for (; step != 0U; step >>= 1U) {
            Adds(probeOffsets, lower, static_cast<float>(static_cast<int32_t>(step - 1U)), p);
            Adds(advanced, lower, static_cast<float>(static_cast<int32_t>(step)), p);
            PipeBarrier<PIPE_V>();
            Mins(probeOffsets, probeOffsets, static_cast<float>(static_cast<int32_t>(updates_)), p);
            PipeBarrier<PIPE_V>();
            Muls(probeOffsets, probeOffsets, 4.0F, p);
            PipeBarrier<PIPE_V>();
            Cast(offsetWords.ReinterpretCast<int32_t>(), probeOffsets, RoundMode::CAST_RINT, p);
            PipeBarrier<PIPE_V>();
            Gather(probes, sortedAxes, offsetWords, 0U, p);
            PipeBarrier<PIPE_V>();
            Compare(searchMask, probes, targets, CMPMODE::LT, p);
            PipeBarrier<PIPE_V>();
            Select(lower, searchMask, advanced, lower, SELMODE::VSEL_TENSOR_TENSOR_MODE, p);
            PipeBarrier<PIPE_V>();
        }
        DataCopy(heads_, lower, axisStride_);
        Adds(probeOffsets, targets, 1.0F, p);
        Duplicate(advanced, 0.0F, p);
        PipeBarrier<PIPE_V>();
        Muls(probeOffsets, probeOffsets, 4.0F, p);
        PipeBarrier<PIPE_V>();
        Cast(offsetWords.ReinterpretCast<int32_t>(), probeOffsets, RoundMode::CAST_RINT, p);
        PipeBarrier<PIPE_V>();
        // Only real axes consume the next lower bound. In particular, do not
        // Gather the padded last lane at offset p when axisStride_ equals p.
        Gather(probes, lower, offsetWords, 0U, axis_);
        PipeBarrier<PIPE_V>();
        Sub(advanced, probes, lower, axis_);
        PipeBarrier<PIPE_V>();
        DataCopy(counts_, advanced, axisStride_);
        WholeReduceMax(probes, advanced, int32_t{64}, static_cast<int32_t>(p / 64U),
                       int32_t{1}, int32_t{1}, int32_t{8}, ReduceOrder::ORDER_ONLY_VALUE);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        height_ = 0U;
        for (uint32_t group = 0U; group < p / 64U; ++group) {
            const uint32_t maximum = static_cast<uint32_t>(static_cast<int32_t>(probes.GetValue(group)));
            height_ = height_ > maximum ? height_ : maximum;
        }
        if (height_ == 0U || height_ > kMaxHeight) { return false; }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);

        // Sort axes by count descending; zero-count real axes precede padding.
        // This small sort uses only[0,6*axisStride), below the CSR at4*n.
        const uint32_t a = axisStride_;
        auto axisScores = f[5U * a];
        auto axisPositions = u[4U * a];
        CreateVecIndex(axisPositions.ReinterpretCast<int32_t>(), int32_t{0}, a);
        CreateVecIndex(f[2U * a], 0.0F, a);
        Muls(axisScores, counts_, static_cast<float>(static_cast<int32_t>(a + 1U)), a);
        PipeBarrier<PIPE_V>();
        Sub(axisScores, axisScores, f[2U * a], a);
        PipeBarrier<PIPE_V>();
        records = SortRecords(f, f[2U * a], axisScores, axisPositions, a);
        GatherMask(axes_, records.ReinterpretCast<uint32_t>(), uint8_t{2}, true,
                   2U * a, {1U, 1U, 0U, 0U}, retained);
        PipeBarrier<PIPE_V>();
        ShiftLeft(work_, axes_, uint32_t{2}, a);
        PipeBarrier<PIPE_V>();
        Gather(otherWork_.ReinterpretCast<float>(), heads_, work_, 0U, a);
        Gather(f, counts_, work_, 0U, a);
        PipeBarrier<PIPE_V>();
        DataCopy(heads_, otherWork_.ReinterpretCast<float>(), a);
        DataCopy(counts_, f, a);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t rank = 0U; rank < axis_; ++rank) {
            inverse_.SetValue(axes_.GetValue(rank), rank * batch_ * 2U);
        }
        uint32_t count = axis_;
        uint32_t cursor = 0U;
        for (uint32_t level = 0U; level < height_; ++level) {
            while (count != 0U && counts_.GetValue(count - 1U) <= static_cast<float>(static_cast<int32_t>(level))) { --count; }
            starts_.SetValue(level, cursor);
            active_.SetValue(level, count);
            const uint32_t padded = Align(count, 8U);
            if (cursor + padded > planeCapacity_) { return false; }
            Adds(work_.ReinterpretCast<float>(), heads_, static_cast<float>(static_cast<int32_t>(level)), count);
            PipeBarrier<PIPE_V>();
            Muls(work_.ReinterpretCast<float>(), work_.ReinterpretCast<float>(), 4.0F, count);
            PipeBarrier<PIPE_V>();
            Cast(otherWork_.ReinterpretCast<int32_t>(), work_.ReinterpretCast<float>(), RoundMode::CAST_RINT, count);
            Duplicate(planes_[cursor], uint32_t{0}, padded);
            PipeBarrier<PIPE_V>();
            Gather(planes_[cursor], csr, otherWork_, 0U, count);
            PipeBarrier<PIPE_V>();
            ShiftLeft(planes_[cursor], planes_[cursor], batch_ == 32U ? uint32_t{6} : uint32_t{5}, count);
            PipeBarrier<PIPE_V>();
            cursor += padded;
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        return true;
    }

    __aicore__ inline void Process()
    {
        const uint64_t first = outer_ * GetBlockIdx() / GetBlockNum();
        const uint64_t end = outer_ * (GetBlockIdx() + 1U) / GetBlockNum();
        for (uint64_t row = first; row < end; row += batch_) {
            const uint32_t rows = static_cast<uint32_t>(end - row < batch_ ? end - row : batch_);
            CopyAlignedRows(columns_, selfGm_, row, rows, axis_);
            if (row == first) {
                // Raw is independent of self's columns/accumulator staging.
                // Prepare retired all former raw readers before Process.
                CopyAlignedRows(sourceUb_, sourceGm_, row, rows, updates_, 1U);
            }
            WaitFlag<HardEvent::MTE2_V>(0);
            TransposeRows(accum_, columns_, rows, Align(axis_, 16U));
            ShiftLeft(work_, axes_, batch_ == 32U ? uint32_t{6} : uint32_t{5}, axisStride_);
            PipeBarrier<PIPE_V>();
            GatherColumnBlocks(columns_, accum_, work_, axis_);
            PipeBarrier<PIPE_V>();
            // Gatherb initialized only this padded prefix. Do not copy the
            // larger axisStride padding, which this self path has not written.
            DataCopy(accum_, columns_, Align(axis_, 8U) * batch_);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::MTE2_V>(1);
            TransposeRows(columns_, sourceUb_, rows, sourceStride_);
            PipeBarrier<PIPE_V>();
            const uint64_t nextRow = row + batch_;
            if (nextRow < end) {
                // This transpose is the last current reader of raw. Its
                // completion also follows consumption of source-ready1.
                SetFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(1);
                const uint32_t nextRows = static_cast<uint32_t>(
                    end - nextRow < batch_ ? end - nextRow : batch_);
                CopyAlignedRows(sourceUb_, sourceGm_, nextRow, nextRows, updates_, 1U);
            }
            if (row != first) {
                // The next writer of dedicated gathered_ is Vector, not
                // MTE2. Source transpose/prefetch above do not touch it.
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            for (uint32_t level = 0U; level < height_; ++level) {
                const uint32_t count = active_.GetValue(level);
                GatherColumnBlocks(gathered_, columns_, planes_[starts_.GetValue(level)], count);
                PipeBarrier<PIPE_V>();
                Add(accum_, accum_, gathered_, count * batch_);
                PipeBarrier<PIPE_V>();
            }
            // Source columns are dead after the last chronological Add.
            // Reorder complete columns before a regular matrix transpose.
            GatherColumnBlocks(columns_, accum_, inverse_, axisStride_);
            PipeBarrier<PIPE_V>();
            TransposeOutput();
            LocalTensor<half> outputValues = gathered_;
            if (axis_ != axisStride_) {
                PipeBarrier<PIPE_V>();
                uint64_t retained = 0U;
                // A2 counter mode selects axis_ real elements per row.
                // Row starts retain their aligned UB stride; selected
                // values are concatenated in the disjoint destination.
                GatherMask(packedOutput_, gathered_, uint8_t{7}, true, axis_,
                    {uint8_t{1}, static_cast<uint8_t>(rows),
                     static_cast<uint8_t>(axisStride_ / 16U), uint8_t{0}},
                    retained);
                PipeBarrier<PIPE_V>();
                outputValues = packedOutput_;
            }
            // Next self DMA overwrites columns_, read by output transpose.
            // Retire the Vector reader independently of the MTE3 consumer.
            SetFlag<HardEvent::V_MTE2>(0);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(outputGm_[row * axis_], outputValues,
                DataCopyExtParams{1U, rows * axis_ * 2U, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        // One output generation remains outstanding after the final batch.
        if (first < end) {
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }

private:
    __aicore__ inline uint32_t Align(uint32_t value, uint32_t alignment)
    { return (value + alignment - 1U) / alignment * alignment; }

    __aicore__ inline LocalTensor<float> SortRecords(LocalTensor<float> records,
        LocalTensor<float> other, LocalTensor<float> scores,
        LocalTensor<uint32_t> positions, uint32_t n)
    {
        Sort32(records, scores, positions, n / 32U);
        PipeBarrier<PIPE_V>();
        for (uint32_t length = 32U; length < n; length *= 4U) {
            const uint32_t group = 4U * length;
            const uint32_t full = n / group;
            if (full != 0U) {
                MrgSort4Info p;
                for (uint32_t k = 0U; k < 4U; ++k) { p.elementLengths[k] = static_cast<uint16_t>(length); }
                p.ifExhaustedSuspension = false; p.validBit = 15U;
                p.repeatTimes = static_cast<uint8_t>(full);
                MrgSortSrcList<float> lists(records, records[2U * length], records[4U * length], records[6U * length]);
                MrgSort(other, lists, p);
            }
            const uint32_t first = full * group;
            const uint32_t left = n - first;
            if (left != 0U) {
                if (left <= length) {
                    DataCopy(other[2U * first], records[2U * first], 2U * left);
                } else {
                    MrgSort4Info p;
                    uint32_t valid = 0U;
                    for (uint32_t k = 0U; k < 4U; ++k) {
                        const uint32_t start = k * length;
                        const uint32_t size = left > start ? (left - start < length ? left - start : length) : 0U;
                        p.elementLengths[k] = static_cast<uint16_t>(size);
                        if (size != 0U) { valid |= 1U << k; }
                    }
                    p.ifExhaustedSuspension = false; p.validBit = static_cast<uint8_t>(valid); p.repeatTimes = 1U;
                    MrgSortSrcList<float> lists(records[2U * first], records[2U * (first + length)],
                        records[2U * (left > 2U * length ? first + 2U * length : first)],
                        records[2U * (left > 3U * length ? first + 3U * length : first)]);
                    MrgSort(other[2U * first], lists, p);
                }
            }
            PipeBarrier<PIPE_V>();
            auto swap = records; records = other; other = swap;
        }
        return records;
    }

    __aicore__ inline void CopyRows(LocalTensor<half> destination,
                                     GlobalTensor<half> gm, uint64_t first,
                                     uint32_t rows, uint32_t stride, uint32_t readyEvent)
    {
        DataCopyPad(destination, gm[first * stride],
            DataCopyExtParams{1U, rows * stride * 2U, 0U, 0U, 0U},
            DataCopyPadExtParams<half>{false, 0U, 0U, static_cast<half>(0)});
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(readyEvent));
    }

    __aicore__ inline void CopyAlignedRows(LocalTensor<half> destination, GlobalTensor<half> gm,
        uint64_t first, uint32_t rows, uint32_t width, uint32_t readyEvent = 0U)
    {
        const uint32_t stride = Align(width, 16U);
        if (width == stride) {
            CopyRows(destination, gm, first, rows, width, readyEvent);
            return;
        }
        const uint32_t rowBytes = width * 2U;
        DataCopyPad(destination, gm[first * width],
            DataCopyExtParams{static_cast<uint16_t>(rows), rowBytes, 0U, 0U, 0U},
            DataCopyPadExtParams<half>{true, 0U,
                static_cast<uint8_t>(stride - width), static_cast<half>(0)});
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(readyEvent));
    }

    __aicore__ inline void TransposeRows(LocalTensor<half> destination,
        LocalTensor<half> source, uint32_t rows, uint32_t rowStride)
    {
        const TransDataTo5HDParams params(false, false,
            static_cast<uint8_t>(rowStride / 16U),
            static_cast<uint16_t>(batch_), uint16_t{1});
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        for (uint32_t first = 0U; first < batch_; first += 16U) {
            for (uint32_t lane = 0U; lane < 16U; ++lane) {
                const uint32_t row = first + lane < rows ? first + lane : rows - 1U;
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[row * rowStride].GetPhyAddr());
                destinationList[lane] = reinterpret_cast<uint64_t>(
                    destination[lane * batch_ + first].GetPhyAddr());
            }
            TransDataTo5HD<half>(destinationList, sourceList, params);
        }
    }

    __aicore__ inline void GatherColumnBlocks(LocalTensor<half> destination,
        LocalTensor<half> source, LocalTensor<uint32_t> plane, uint32_t count)
    {
        const uint8_t repeats = static_cast<uint8_t>((count + 7U) / 8U);
        const GatherRepeatParams params{
            static_cast<uint8_t>(batch_ / 16U), static_cast<uint8_t>(batch_ / 2U)};
        // Each offset is an aligned column base. Callers exclude padding
        // from numerical updates or the final GM output. Self permutation
        // can read the zero-padded columns up to Align(axis, 8).
        Gatherb(destination.ReinterpretCast<uint16_t>(), source.ReinterpretCast<uint16_t>(),
                plane, repeats, params);
        if (batch_ == 32U) {
            // Disjoint odd32-byte blocks complete the64-byte columns.
            Gatherb(destination[16U].ReinterpretCast<uint16_t>(),
                    source[16U].ReinterpretCast<uint16_t>(), plane, repeats, params);
        }
    }

    __aicore__ inline void TransposeOutput()
    {
        const TransDataTo5HDParams params(false, false,
            static_cast<uint8_t>(axisStride_ / 16U), uint16_t{1},
            static_cast<uint16_t>(batch_));
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        for (uint32_t first = 0U; first < batch_; first += 16U) {
            for (uint32_t lane = 0U; lane < 16U; ++lane) {
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    columns_[lane * batch_ + first].GetPhyAddr());
                destinationList[lane] = reinterpret_cast<uint64_t>(
                    gathered_[(first + lane) * axisStride_].GetPhyAddr());
            }
            TransDataTo5HD<half>(destinationList, sourceList, params);
        }
    }

    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<half> sourceGm_, selfGm_, outputGm_;
    GlobalTensor<int32_t> indexGm_;
    LocalTensor<half> sourceUb_, columns_, accum_, gathered_, packedOutput_;
    LocalTensor<uint32_t> planes_, axes_, inverse_, work_, otherWork_, starts_, active_;
    LocalTensor<float> heads_, counts_;
    uint32_t axis_, updates_, axisStride_, sortCount_, sourceStride_, batch_, planeCapacity_, height_;
    uint64_t outer_;
};
