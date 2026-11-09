#include "aclnn/acl_meta.h"
#include <cstdint>

extern "C" aclnnStatus aclnnSquareSumV1CanonicalGetWorkspaceSize(
    const aclTensor *, const aclIntArray *, bool, const aclTensor *,
    uint64_t *, aclOpExecutor **);

namespace {
// Executors may retain attribute descriptors until execution completes. Keep
// immutable all-axis descriptors alive for the entire loaded library lifetime,
// rather than destroying a temporary immediately after GetWorkspaceSize.
// Initialization is thread-safe; the cache has a fixed bound of 64 small arrays.
// They are process-lifetime metadata, reclaimed by the OS: an ACL destructor
// cannot safely run after torch_npu has already finalized its ACL runtime.
struct AllAxes {
    aclIntArray *arrays[65]{};
    AllAxes() {
        int64_t values[64];
        for (int64_t i = 0; i < 64; ++i) values[i] = i;
        for (uint64_t rank = 1; rank <= 64; ++rank)
            arrays[rank] = aclCreateIntArray(values, rank);
    }
};
}

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor *x, const aclIntArray *axis, bool keepDims,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
    uint64_t axisCount = 0;
    if (axis) {
        const aclnnStatus status = aclGetIntArraySize(axis, &axisCount);
        if (status != 0) return status;
    }
    if (axisCount != 0)
        return aclnnSquareSumV1CanonicalGetWorkspaceSize(
            x, axis, keepDims, out, workspaceSize, executor);

    // torch.sum(dim=()) reduces all axes. The generated CANN 8.5 interface
    // rejects a length-zero aclIntArray, so pass the equivalent explicit axis list.
    int64_t *dims = nullptr;
    uint64_t rank = 0;
    const aclnnStatus shapeStatus = aclGetViewShape(x, &dims, &rank);
    delete[] dims; // Required by the public aclGetViewShape ownership contract.
    if (shapeStatus != 0) return shapeStatus;
    if (rank > 64) return 161002; // ACLNN_ERR_PARAM_INVALID
    static const AllAxes cache;
    const aclIntArray *all = cache.arrays[rank == 0 ? 1 : rank];
    if (!all) return 161001; // ACLNN_ERR_PARAM_NULLPTR
    return aclnnSquareSumV1CanonicalGetWorkspaceSize(
        x, all, keepDims, out, workspaceSize, executor);
}
