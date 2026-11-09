template <>
__aicore__ inline void GreaterKernel<float>::Compute(uint32_t count)
{
    LocalTensor<float> selfLocal = selfQueue.DeQue<float>();
    LocalTensor<float> otherLocal = otherQueue.DeQue<float>();
    LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    const uint32_t computeElems = AlignElems256<float>(count);
    Compare(mask, selfLocal, otherLocal, CMPMODE::GT, computeElems);
    LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
    LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
    Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
    Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
    outQueue.EnQue(outLocal);
    selfQueue.FreeTensor(selfLocal);
    otherQueue.FreeTensor(otherLocal);
}

template <>
__aicore__ inline void GreaterKernel<half>::Compute(uint32_t count)
{
    LocalTensor<half> selfLocal = selfQueue.DeQue<half>();
    LocalTensor<half> otherLocal = otherQueue.DeQue<half>();
    LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    const uint32_t computeElems = AlignElems256<half>(count);
    Compare(mask, selfLocal, otherLocal, CMPMODE::GT, computeElems);
    Duplicate(otherLocal, static_cast<half>(1.0F), 128U);
    const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
    const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
    Select(selfLocal, mask, otherLocal, static_cast<half>(0.0F),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
           repeatParams);
    Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
    outQueue.EnQue(outLocal);
    selfQueue.FreeTensor(selfLocal);
    otherQueue.FreeTensor(otherLocal);
}

template <>
__aicore__ inline void GreaterKernel<bfloat16_t>::Compute(uint32_t count)
{
    LocalTensor<bfloat16_t> selfLocal = selfQueue.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> otherLocal = otherQueue.DeQue<bfloat16_t>();
    LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
    LocalTensor<float> selfFloat = selfFloatBuf.Get<float>();
    LocalTensor<float> otherFloat = otherFloatBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    const uint32_t computeElems = AlignElems256<bfloat16_t>(count);
    Cast(selfFloat, selfLocal, RoundMode::CAST_NONE, computeElems);
    Cast(otherFloat, otherLocal, RoundMode::CAST_NONE, computeElems);
    Compare(mask, selfFloat, otherFloat, CMPMODE::GT, AlignElems256<float>(count));
    LocalTensor<half> oneHalf = selfLocal.template ReinterpretCast<half>();
    const uint32_t computeFloatElems = AlignElems256<float>(count);
    Duplicate(oneHalf, static_cast<half>(1.0F), computeFloatElems);
    Select(oneHalf, mask, oneHalf, static_cast<half>(0.0F),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, computeFloatElems);
    Cast(outLocal, oneHalf, RoundMode::CAST_NONE, computeFloatElems);
    outQueue.EnQue(outLocal);
    selfQueue.FreeTensor(selfLocal);
    otherQueue.FreeTensor(otherLocal);
}

template <>
__aicore__ inline void GreaterKernel<int32_t>::Compute(uint32_t count)
{
    LocalTensor<int32_t> selfLocal = selfQueue.DeQue<int32_t>();
    LocalTensor<int32_t> otherLocal = otherQueue.DeQue<int32_t>();
    LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
    LocalTensor<int32_t> maxLocal = selfFloatBuf.Get<int32_t>();
    LocalTensor<uint8_t> eqSelfMask = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> eqOtherMask = otherFloatBuf.Get<uint8_t>();
    const uint32_t computeElems = AlignElems256<int32_t>(count);
    const uint32_t logicalSize = AlignElems256<uint8_t>(count) / 16U;
    Max(maxLocal, selfLocal, otherLocal, computeElems);
    Compare(eqSelfMask, selfLocal, maxLocal, CMPMODE::EQ, computeElems);
    Compare(eqOtherMask, selfLocal, otherLocal, CMPMODE::EQ, computeElems);
    Not(eqOtherMask.template ReinterpretCast<uint16_t>(), eqOtherMask.template ReinterpretCast<uint16_t>(), logicalSize);
    And(eqSelfMask.template ReinterpretCast<uint16_t>(), eqSelfMask.template ReinterpretCast<uint16_t>(),
        eqOtherMask.template ReinterpretCast<uint16_t>(), logicalSize);
    LocalTensor<half> oneHalf = selfLocal.template ReinterpretCast<half>();
    LocalTensor<half> zeroHalf = otherLocal.template ReinterpretCast<half>();
    Duplicate(oneHalf, static_cast<half>(1.0F), computeElems);
    Duplicate(zeroHalf, static_cast<half>(0.0F), computeElems);
    Select(oneHalf, eqSelfMask, oneHalf, zeroHalf, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeElems);
    Cast(outLocal, oneHalf, RoundMode::CAST_NONE, computeElems);
    outQueue.EnQue(outLocal);
    selfQueue.FreeTensor(selfLocal);
    otherQueue.FreeTensor(otherLocal);
}

template <>
__aicore__ inline void GreaterKernel<int8_t>::Compute(uint32_t count)
{
    LocalTensor<int8_t> selfLocal = selfQueue.DeQue<int8_t>();
    LocalTensor<int8_t> otherLocal = otherQueue.DeQue<int8_t>();
    LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
    LocalTensor<half> selfHalf = selfFloatBuf.Get<half>();
    LocalTensor<half> otherHalf = otherFloatBuf.Get<half>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    const uint32_t computeElems = AlignElems256<int8_t>(count);
    Cast(selfHalf, selfLocal, RoundMode::CAST_NONE, computeElems);
    Cast(otherHalf, otherLocal, RoundMode::CAST_NONE, computeElems);
    Compare(mask, selfHalf, otherHalf, CMPMODE::GT, computeElems);
    Duplicate(selfHalf, static_cast<half>(1.0F), computeElems);
    Duplicate(otherHalf, static_cast<half>(0.0F), computeElems);
    Select(selfHalf, mask, selfHalf, otherHalf, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeElems);
    Cast(outLocal, selfHalf, RoundMode::CAST_NONE, computeElems);
    outQueue.EnQue(outLocal);
    selfQueue.FreeTensor(selfLocal);
    otherQueue.FreeTensor(otherLocal);
}
