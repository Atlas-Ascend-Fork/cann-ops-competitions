#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_ctc_loss.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}


static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor
    aclTensor* log_probs = nullptr;
    void* log_probsDeviceAddr = nullptr;
    aclTensor* targets = nullptr;
    void* targetsDeviceAddr = nullptr;
    aclTensor* input_lengths = nullptr;
    void* input_lengthsDeviceAddr = nullptr;
    aclTensor* target_lengths = nullptr;
    void* target_lengthsDeviceAddr = nullptr;
    std::vector<int64_t> log_probsShape = {2, 1, 3};
    std::vector<float> log_probsHostData(6, 1);
    ret = CreateAclTensor(log_probsHostData, log_probsShape, &log_probsDeviceAddr, aclDataType::ACL_FLOAT, &log_probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> targetsShape = {1, 2};
    std::vector<int32_t> targetsHostData(2, 1);
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT32, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> input_lengthsShape = {1};
    std::vector<int32_t> input_lengthsHostData(1, 1);
    ret = CreateAclTensor(input_lengthsHostData, input_lengthsShape, &input_lengthsDeviceAddr, aclDataType::ACL_INT32, &input_lengths);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> target_lengthsShape = {1};
    std::vector<int32_t> target_lengthsHostData(1, 1);
    ret = CreateAclTensor(target_lengthsHostData, target_lengthsShape, &target_lengthsDeviceAddr, aclDataType::ACL_INT32, &target_lengths);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* neg_log_likelihood = nullptr;
    void* neg_log_likelihoodDeviceAddr = nullptr;
    std::vector<int64_t> neg_log_likelihoodShape = {1};
    std::vector<float> neg_log_likelihoodHostData(1, 0);
    ret = CreateAclTensor(neg_log_likelihoodHostData, neg_log_likelihoodShape, &neg_log_likelihoodDeviceAddr, aclDataType::ACL_FLOAT, &neg_log_likelihood);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    

    // 调用 aclnnCtcLossGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCtcLossGetWorkspaceSize(log_probs, targets, input_lengths, target_lengths, 0, "none", false, neg_log_likelihood, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnCtcLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnCtcLoss 第二段接口
    ret = aclnnCtcLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnCtcLoss failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclDestroyTensor(log_probs);
    aclrtFree(log_probsDeviceAddr);
    aclDestroyTensor(targets);
    aclrtFree(targetsDeviceAddr);
    aclDestroyTensor(input_lengths);
    aclrtFree(input_lengthsDeviceAddr);
    aclDestroyTensor(target_lengths);
    aclrtFree(target_lengthsDeviceAddr);
    
    aclDestroyTensor(neg_log_likelihood);
    aclrtFree(neg_log_likelihoodDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
