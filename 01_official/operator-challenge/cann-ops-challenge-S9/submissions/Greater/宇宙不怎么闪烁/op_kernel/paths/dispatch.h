    __aicore__ inline GreaterKernel() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
                                GM_ADDR workspace, GM_ADDR tiling, TPipe* pipeIn)
    {
        (void)workspace;
        pipe = pipeIn;
        GET_TILING_DATA(td, tiling);
        totalSize = td.total_size;
        blockElems = td.block_elems;
        totalBlocks = td.total_blocks;
        tileElems = td.tile_elems;
        sameShape = td.same_shape != 0U;
        selfDense = td.self_dense != 0U;
        otherDense = td.other_dense != 0U;
        for (uint32_t i = 0U; i < kDimNum; ++i) {
            outDims[i] = td.out_dims[i];
            selfDims[i] = td.self_dims[i];
            otherDims[i] = td.other_dims[i];
            selfStrides[i] = td.self_strides[i];
            otherStrides[i] = td.other_strides[i];
            outStrides[i] = td.out_strides[i];
            selfScalar = selfScalar && td.self_strides[i] == 0U;
            otherScalar = otherScalar && td.other_strides[i] == 0U;
        }
        cScalarOther = IsRank3CScalarOther();
        cRowOther = IsRank4CRowOther();

        selfGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        otherGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(other));
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output), totalSize);

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        if (blockNum == 0U || blockIdx >= blockNum) {
            startBlock = 0U;
            endBlock = 0U;
            startElem = 0U;
            endElem = 0U;
            return;
        }
        const uint32_t baseBlocks = totalBlocks / blockNum;
        const uint32_t remainBlocks = totalBlocks - baseBlocks * blockNum;
        const uint32_t extraBefore = blockIdx < remainBlocks ? blockIdx : remainBlocks;
        const uint32_t myBlocks = baseBlocks + (blockIdx < remainBlocks ? 1U : 0U);
        startBlock = blockIdx * baseBlocks + extraBefore;
        endBlock = startBlock + myBlocks;
        const uint64_t start = static_cast<uint64_t>(startBlock) * blockElems;
        const uint64_t end = static_cast<uint64_t>(endBlock) * blockElems;
        startElem = static_cast<uint32_t>(start);
        endElem = end < totalSize ? static_cast<uint32_t>(end) : totalSize;
    }

    __aicore__ inline void Process()
    {
        if (startElem >= endElem || tileElems == 0U) {
            return;
        }
        if constexpr (std::is_same_v<T, int32_t>) {
            if (CanProcessInt32RowSelfBroadcast()) {
                ProcessInt32RowSelfBroadcast();
                return;
            }
        }
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            if (cScalarOther) {
                ProcessCScalarOther();
                return;
            }
            if (cRowOther) {
                ProcessCRowOther();
                return;
            }
            if (CanProcessDenseSameShapePipelined()) {
                ProcessDenseSameShapePipelined();
                return;
            }
        }
        if constexpr (std::is_same_v<T, int8_t>) {
            if (CanProcessDenseSameShapePipelined()) {
                ProcessDenseSameShapePipelined();
                return;
            }
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (CanProcessBf16CRowOther()) {
                ProcessBf16CRowFlatGather();
                return;
            }
            if (CanProcessBf16BroadcastPipelined()) {
                ProcessBf16BroadcastPipelined();
                return;
            }
        }
        const uint32_t inLocalElems = AlignElems256<T>(tileElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(tileElems);
        pipe->InitBuffer(selfQueue, kBufferNum, inLocalElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, inLocalElems * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, outLocalElems * sizeof(uint8_t));
        InitTempBuffers();
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants();

        for (uint32_t offset = startElem; offset < endElem; offset += tileElems) {
            const uint32_t curElems = MinU32(tileElems, endElem - offset);
            CopyIn(offset, curElems);
            Compute(curElems);
            CopyOut(offset, curElems);
        }
    }
