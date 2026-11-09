    __aicore__ inline void InitTempBuffers(uint32_t localTileElems = 0U)
    {
        const uint32_t bufferElems = localTileElems == 0U ? tileElems : localTileElems;
        if constexpr (std::is_same_v<T, float>) {
            if (!sameShape) {
                pipe->InitBuffer(selfFloatBuf,
                                 AlignElems256<half>(bufferElems) * sizeof(half));
                pipe->InitBuffer(otherFloatBuf,
                                 AlignElems256<half>(bufferElems) * sizeof(half));
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if (cScalarOther || cRowOther) {
                pipe->InitBuffer(otherFloatBuf, AlignElems256<half>(bufferElems) * sizeof(half));
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(selfFloatBuf, AlignElems256<float>(bufferElems) * sizeof(float));
            pipe->InitBuffer(otherFloatBuf, AlignElems256<float>(bufferElems) * sizeof(float));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            pipe->InitBuffer(selfFloatBuf, AlignElems256<int32_t>(bufferElems) * sizeof(int32_t));
            pipe->InitBuffer(otherFloatBuf, AlignElems256<uint8_t>(bufferElems) * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            const uint32_t halfBytes = AlignElems256<half>(bufferElems) * sizeof(half);
            pipe->InitBuffer(selfFloatBuf,
                             halfBytes + kInt8BankPadSelfHalfToOtherHalfBytes);
            pipe->InitBuffer(otherFloatBuf, halfBytes);
        }
    }

    __aicore__ inline void InitLoopConstants(uint32_t localTileElems = 0U)
    {
        const uint32_t bufferElems = localTileElems == 0U ? tileElems : localTileElems;
        if constexpr (std::is_same_v<T, float>) {
            if (!sameShape) {
                LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
                Duplicate(oneHalf, static_cast<half>(1.0F),
                          AlignElems256<float>(bufferElems));
            }
        }
    }
