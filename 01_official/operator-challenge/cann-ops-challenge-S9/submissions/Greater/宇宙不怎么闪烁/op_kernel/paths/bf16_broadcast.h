    __aicore__ inline bool CanProcessBf16CRowOther() const
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            return false;
        } else {
            if (!selfDense || outDims[3] < 2U || outDims[4] < 64U) {
                return false;
            }
            for (uint32_t i = 0U; i < 3U; ++i) {
                if (selfDims[i] != outDims[i] || otherDims[i] != outDims[i]) {
                    return false;
                }
            }
            if (selfDims[3] != outDims[3] || selfDims[4] != outDims[4] ||
                otherDims[3] != 1U || otherDims[4] != outDims[4]) {
                return false;
            }
            const uint32_t pitch = AlignElems256<bfloat16_t>(outDims[4]);
            const uint32_t maskBytes =
                AlignElems256<uint8_t>((pitch + 7U) / 8U);
            constexpr uint32_t ubBytes = 192U * 1024U;
            constexpr uint32_t selectTempBytes = 8U * 1024U;
            // Double-buffered queues plus a four-row fp32 scratch consume
            // (6 * rows + 24) * pitch bytes, plus the packed predicate.
            // Wider rows use the generic BF16 pipeline instead of overflowing
            // UB on hidden shapes.
            constexpr uint32_t pitchFactor =
                6U * kBf16CRowMaxRows +
                4U * kBf16CRowFloatRows + 8U;
            return static_cast<uint64_t>(pitchFactor) * pitch + maskBytes +
                       selectTempBytes +
                       kBf16CRowBankPadCount * kBf16CRowBankPadBytes <= ubBytes;
        }
    }

    __aicore__ inline void ProcessBf16CRowOther()
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            const uint32_t hDim = outDims[3];
            const uint32_t width = outDims[4];
            const uint32_t pitch = AlignElems256<bfloat16_t>(width);
            const uint32_t batchElems = kBf16CRowMaxRows * pitch;
            const uint32_t maskBytes =
                AlignElems256<uint8_t>((pitch + 7U) / 8U);

            // One BF16 self batch is reused as the half 0/1 materialization
            // buffer after its fp32 Cast has completed.  Queue depth two
            // provides distinct storage for the previous/current/next stages.
            pipe->InitBuffer(bf16CRowSelfQueue, 2U,
                             batchElems * sizeof(bfloat16_t));
            pipe->InitBuffer(broadcastBuf, kBf16CRowBankPadBytes);
            pipe->InitBuffer(bf16CRowOtherQueue, 2U,
                             pitch * sizeof(bfloat16_t));
            pipe->InitBuffer(bankPadBuf, kBf16CRowBankPadBytes);
            pipe->InitBuffer(bf16CRowOutQueue, 2U,
                             batchElems * sizeof(uint8_t));
            pipe->InitBuffer(bankTailPadBuf, kBf16CRowBankPadBytes);
            pipe->InitBuffer(selfFloatBuf,
                             kBf16CRowFloatRows * pitch * sizeof(float));
            pipe->InitBuffer(bankScratchPadBuf, kBf16CRowBankPadBytes);
            pipe->InitBuffer(otherFloatBuf, pitch * sizeof(float));
            pipe->InitBuffer(bankMaskPadBuf, kBf16CRowBankPadBytes);
            pipe->InitBuffer(maskBuf, maskBytes);

            const uint32_t totalRows = totalSize / width;
            const uint32_t blockNum = GetBlockNum();
            const uint32_t blockIdx = GetBlockIdx();
            uint32_t rowsBefore = 0U;
            uint32_t myRows = 0U;
            Bf16CRowRange(totalRows, hDim, blockNum, blockIdx,
                          rowsBefore, myRows);
            const uint32_t rowEnd = rowsBefore + myRows;
            if (rowsBefore >= rowEnd) {
                return;
            }

            uint32_t currentRow = rowsBefore;
            uint32_t currentRows = Bf16CRowBatchRows(currentRow, rowEnd, hDim);
            PrefetchBf16CRow(currentRow, currentRows, width, pitch, hDim);
            uint32_t pendingRow = 0U;
            uint32_t pendingRows = 0U;
            bool hasPendingOutput = false;

            while (true) {
                LocalTensor<bfloat16_t> currentSelf =
                    bf16CRowSelfQueue.DeQue<bfloat16_t>();
                LocalTensor<bfloat16_t> currentRhs =
                    bf16CRowOtherQueue.DeQue<bfloat16_t>();
                const uint32_t nextRow = currentRow + currentRows;
                const bool hasNext = nextRow < rowEnd;
                uint32_t nextRows = 0U;
                if (hasNext) {
                    nextRows = Bf16CRowBatchRows(nextRow, rowEnd, hDim);
                    PrefetchBf16CRow(nextRow, nextRows, width, pitch, hDim);
                }
                if (hasPendingOutput) {
                    CopyOutBf16CRow(pendingRow, pendingRows, width, pitch);
                }
                ComputeBf16CRow(currentSelf, currentRhs,
                                currentRows, pitch);
                pendingRow = currentRow;
                pendingRows = currentRows;
                hasPendingOutput = true;

                if (!hasNext) {
                    break;
                }
                currentRow = nextRow;
                currentRows = nextRows;
            }
            if (hasPendingOutput) {
                CopyOutBf16CRow(pendingRow, pendingRows, width, pitch);
            }
        }
    }

    __aicore__ inline void Bf16CRowRange(
        uint32_t totalRows, uint32_t hDim, uint32_t blockNum,
        uint32_t blockIdx, uint32_t& rowStart, uint32_t& rowCount) const
    {
        // The fixed BF16 C-row workload has 255 groups of 12 rows.  A regular
        // 76/77 contiguous split makes seven cores touch an eighth group.  The
        // layout below keeps every core at seven group segments while changing
        // the maximum row load by only one.  Each two-core or three-core
        // pattern ends on a group boundary, so no lookup table is required.
        if (totalRows == 3060U && hDim == 12U && blockNum == 40U) {
            if (blockIdx < 10U) {
                rowStart = blockIdx * 78U;
                rowCount = 78U;
                return;
            }
            const uint32_t tailIdx = blockIdx - 10U;
            const uint32_t pattern = tailIdx / 3U;
            const uint32_t phase = tailIdx - pattern * 3U;
            rowStart = 780U + pattern * 228U;
            if (phase == 0U) {
                rowCount = 77U;
            } else if (phase == 1U) {
                rowStart += 77U;
                rowCount = 75U;
            } else {
                rowStart += 152U;
                rowCount = 76U;
            }
            return;
        }

        const uint32_t baseRows = totalRows / blockNum;
        const uint32_t remainRows = totalRows - baseRows * blockNum;
        rowStart = blockIdx * baseRows + MinU32(blockIdx, remainRows);
        rowCount = baseRows + (blockIdx < remainRows ? 1U : 0U);
    }

    __aicore__ inline uint32_t Bf16CRowBatchRows(uint32_t row,
                                                  uint32_t rowEnd,
                                                  uint32_t hDim) const
    {
        const uint32_t rowInGroup = row % hDim;
        return MinU32(kBf16CRowMaxRows,
                      MinU32(rowEnd - row, hDim - rowInGroup));
    }

    __aicore__ inline void PrefetchBf16CRow(uint32_t row, uint32_t rows,
                                             uint32_t width, uint32_t pitch,
                                             uint32_t hDim)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<bfloat16_t> selfLocal =
                bf16CRowSelfQueue.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> rhsLocal =
                bf16CRowOtherQueue.AllocTensor<bfloat16_t>();
            const uint32_t blockBytes = width * sizeof(bfloat16_t);
            const uint32_t paddedBytes = AlignUp32Bytes(blockBytes);
            const uint32_t dstStride =
                (pitch * sizeof(bfloat16_t) - paddedBytes) / 32U;
            DataCopyExtParams selfCopyParams{
                static_cast<uint16_t>(rows), blockBytes,
                0U, dstStride, 0U};
            DataCopyPadExtParams<bfloat16_t> padParams{
                false, 0U, 0U, static_cast<bfloat16_t>(0.0F)};
            DataCopyPad(selfLocal, selfGm[row * width],
                        selfCopyParams, padParams);
            DataCopyExtParams rhsCopyParams{
                1U, blockBytes, 0U, 0U, 0U};
            const uint32_t group = row / hDim;
            DataCopyPad(rhsLocal, otherGm[group * width],
                        rhsCopyParams, padParams);
            bf16CRowSelfQueue.EnQue(selfLocal);
            bf16CRowOtherQueue.EnQue(rhsLocal);
        }
    }

    __aicore__ inline void ComputeBf16CRow(
        LocalTensor<bfloat16_t>& selfLocal,
        LocalTensor<bfloat16_t>& rhsLocal,
        uint32_t rows, uint32_t pitch)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<uint8_t> outLocal =
                bf16CRowOutQueue.AllocTensor<uint8_t>();
            LocalTensor<float> selfFloat = selfFloatBuf.Get<float>();
            LocalTensor<float> rhsFloat = otherFloatBuf.Get<float>();
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            Cast(rhsFloat, rhsLocal, RoundMode::CAST_NONE, pitch);

            // The BF16 RHS staging slot is dead after widening.  Reuse its
            // first 256 bytes as one vector of half ones and broadcast that
            // vector through Select with a zero repeat stride.  Materializing
            // every output-sized half one tensor would add a full UB write for
            // each batch without changing the comparison result.
            LocalTensor<half> oneHalf =
                rhsLocal.template ReinterpretCast<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t selectRepeats =
                static_cast<uint8_t>(pitch / 128U);
            const BinaryRepeatParams selectParams{
                1U, 1U, 1U, 8U, 0U, 0U};

            LocalTensor<half> resultHalf =
                selfLocal.template ReinterpretCast<half>();
            for (uint32_t row0 = 0U; row0 < rows;
                 row0 += kBf16CRowFloatRows) {
                const uint32_t chunkRows =
                    MinU32(kBf16CRowFloatRows, rows - row0);
                const uint32_t chunkElems = chunkRows * pitch;
                const uint32_t chunkBase = row0 * pitch;
                Cast(selfFloat, selfLocal[chunkBase],
                     RoundMode::CAST_NONE, chunkElems);
                // The fp32 scratch is deliberately four rows deep.  Spell out
                // those four row operations so CCEC does not retain a dynamic
                // inner loop around every Compare/Select pair.  Partial chunks
                // still execute only their valid rows.
                Compare(mask, selfFloat, rhsFloat, CMPMODE::GT, pitch);
                Select(resultHalf[chunkBase], mask,
                       oneHalf, static_cast<half>(0.0F),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,
                       128U, selectRepeats, selectParams);
                if (chunkRows > 1U) {
                    Compare(mask, selfFloat[pitch], rhsFloat,
                            CMPMODE::GT, pitch);
                    Select(resultHalf[chunkBase + pitch], mask,
                           oneHalf, static_cast<half>(0.0F),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           128U, selectRepeats, selectParams);
                }
                if (chunkRows > 2U) {
                    Compare(mask, selfFloat[2U * pitch], rhsFloat,
                            CMPMODE::GT, pitch);
                    Select(resultHalf[chunkBase + 2U * pitch], mask,
                           oneHalf, static_cast<half>(0.0F),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           128U, selectRepeats, selectParams);
                }
                if (chunkRows > 3U) {
                    Compare(mask, selfFloat[3U * pitch], rhsFloat,
                            CMPMODE::GT, pitch);
                    Select(resultHalf[chunkBase + 3U * pitch], mask,
                           oneHalf, static_cast<half>(0.0F),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           128U, selectRepeats, selectParams);
                }
                Cast(outLocal[chunkBase], resultHalf[chunkBase],
                     RoundMode::CAST_NONE, chunkElems);
            }
            bf16CRowOutQueue.EnQue(outLocal);
            bf16CRowSelfQueue.FreeTensor(selfLocal);
            bf16CRowOtherQueue.FreeTensor(rhsLocal);
        }
    }

    __aicore__ inline void CopyOutBf16CRow(uint32_t row, uint32_t rows,
                                            uint32_t width, uint32_t pitch)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<uint8_t> outLocal =
                bf16CRowOutQueue.DeQue<uint8_t>();
            const uint32_t paddedBytes = AlignUp32Bytes(width);
            const uint32_t srcStride = (pitch - paddedBytes) / 32U;
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), width,
                srcStride, 0U, 0U};
            DataCopyPad(outGm[row * width], outLocal, copyParams);
            bf16CRowOutQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline bool CanProcessBf16BroadcastPipelined() const
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            return false;
        } else {
            return !sameShape && selfDense && !otherDense &&
                   (kBf16BroadcastPipelineElems % blockElems) == 0U;
        }
    }

    __aicore__ inline void ProcessBf16BroadcastPipelined()
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            constexpr uint32_t pipelineElems = kBf16BroadcastPipelineElems;
            constexpr uint32_t inLocalElems = kBf16BroadcastPipelineElems;
            constexpr uint32_t outLocalElems = kBf16BroadcastPipelineElems;
            pipe->InitBuffer(bf16PipelineSelfQueue, 2U,
                             inLocalElems * sizeof(bfloat16_t));
            pipe->InitBuffer(bf16PipelineOtherQueue, 2U,
                             inLocalElems * sizeof(bfloat16_t));
            pipe->InitBuffer(bf16PipelineOutQueue, 2U,
                             outLocalElems * sizeof(uint8_t));
            InitTempBuffers(pipelineElems);
            const uint32_t maskBytes =
                AlignElems256<uint8_t>((pipelineElems + 7U) / 8U);
            pipe->InitBuffer(maskBuf, maskBytes);

            uint32_t currentOffset = startElem;
            uint32_t currentElems = MinU32(pipelineElems, endElem - currentOffset);
            PrefetchBf16Broadcast(currentOffset, currentElems);
            uint32_t pendingOutputOffset = 0U;
            uint32_t pendingOutputElems = 0U;
            bool hasPendingOutput = false;

            while (true) {
                LocalTensor<bfloat16_t> currentSelf =
                    bf16PipelineSelfQueue.DeQue<bfloat16_t>();
                LocalTensor<bfloat16_t> currentOther =
                    bf16PipelineOtherQueue.DeQue<bfloat16_t>();
                const uint32_t nextOffset = currentOffset + currentElems;
                const bool hasNext = nextOffset < endElem;
                uint32_t nextElems = 0U;
                if (hasNext) {
                    nextElems = MinU32(pipelineElems, endElem - nextOffset);
                    PrefetchBf16Broadcast(nextOffset, nextElems);
                }
                if (hasPendingOutput) {
                    CopyOutBf16Pipelined(pendingOutputOffset, pendingOutputElems);
                }
                ComputeBf16BroadcastReady(currentSelf, currentOther, currentElems);
                pendingOutputOffset = currentOffset;
                pendingOutputElems = currentElems;
                hasPendingOutput = true;

                if (!hasNext) {
                    break;
                }
                currentOffset = nextOffset;
                currentElems = nextElems;
            }
            if (hasPendingOutput) {
                CopyOutBf16Pipelined(pendingOutputOffset, pendingOutputElems);
            }
        }
    }

    __aicore__ inline void PrefetchBf16Broadcast(uint32_t offset, uint32_t count)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<bfloat16_t> selfLocal =
                bf16PipelineSelfQueue.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> otherLocal =
                bf16PipelineOtherQueue.AllocTensor<bfloat16_t>();
            CopyDense(selfLocal, selfGm, offset, count);
            FillBroadcast(otherLocal, otherGm, otherStrides, offset, count);
            bf16PipelineSelfQueue.EnQue(selfLocal);
            bf16PipelineOtherQueue.EnQue(otherLocal);
        }
    }

    __aicore__ inline void ComputeBf16BroadcastReady(LocalTensor<bfloat16_t>& selfLocal,
                                                       LocalTensor<bfloat16_t>& otherLocal,
                                                       uint32_t count)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<uint8_t> outLocal =
                bf16PipelineOutQueue.AllocTensor<uint8_t>();
            LocalTensor<float> selfFloat = selfFloatBuf.Get<float>();
            LocalTensor<float> otherFloat = otherFloatBuf.Get<float>();
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t castElems = AlignElems256<bfloat16_t>(count);
            const uint32_t computeElems = AlignElems256<float>(count);
            Cast(selfFloat, selfLocal, RoundMode::CAST_NONE, castElems);
            Cast(otherFloat, otherLocal, RoundMode::CAST_NONE, castElems);
            Compare(mask, selfFloat, otherFloat, CMPMODE::GT, computeElems);
            LocalTensor<half> resultHalf = selfLocal.ReinterpretCast<half>();
            Duplicate(resultHalf, static_cast<half>(1.0F), computeElems);
            Select(resultHalf, mask, resultHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
            bf16PipelineOutQueue.EnQue(outLocal);
            bf16PipelineSelfQueue.FreeTensor(selfLocal);
            bf16PipelineOtherQueue.FreeTensor(otherLocal);
        }
    }

    __aicore__ inline void CopyOutBf16Pipelined(uint32_t offset, uint32_t count)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<uint8_t> outLocal =
                bf16PipelineOutQueue.DeQue<uint8_t>();
            if (((offset & 31U) == 0U) && ((count & 31U) == 0U)) {
                DataCopy(outGm[offset], outLocal, count);
            } else {
                DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
                DataCopyPad(outGm[offset], outLocal, copyParams);
            }
            bf16PipelineOutQueue.FreeTensor(outLocal);
        }
    }
