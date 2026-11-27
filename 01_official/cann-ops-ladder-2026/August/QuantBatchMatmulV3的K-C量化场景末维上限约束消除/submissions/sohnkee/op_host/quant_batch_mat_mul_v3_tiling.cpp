/*!
 * \file quant_batch_mat_mul_v3_tiling.cpp
 * \brief QuantBatchMatMulV3 tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_data.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_key.h"

#include <algorithm>
#include <climits>
#include <cstdint>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t COL_TILE = 8;
constexpr int64_t MATMUL_AXIS_LIMIT = 65535;
constexpr int64_t LARGE_K_MID_LIMIT = 98304;
constexpr int64_t LARGE_N_SMALL_K_LIMIT = 256;
constexpr int64_t LARGE_N_DIRECT_K_LIMIT = 1024;
constexpr int64_t THIN_M_LIMIT = 16;
constexpr int64_t STREAMK_SMALL_M_LIMIT = 64;
constexpr int64_t MEDIUM_M_LIMIT = 128;
constexpr int64_t TALL_M_LIMIT = 256;
constexpr int64_t SMALL_N_LIMIT = 64;
constexpr int64_t MEDIUM_N_LIMIT = 128;
constexpr int64_t STREAMK_MID_N_LIMIT = 256;
constexpr uint64_t A_COMPACT_K_STEP = 1024;
constexpr uint64_t ROW_MAJOR_TALL_M_A_COMPACT_K_STEP = 8192;
constexpr uint64_t TRANS_B_COMPACT_STAGED_K_CHUNK = 49152;
constexpr int64_t COMPACT_M_TILING_SIZE = 1024;
constexpr int32_t ROW_MAJOR_LARGE_K_M_TILE = 1;
constexpr int32_t ROW_MAJOR_SMALL_N_ORG_M_TILE = 7;
constexpr int64_t ROW_MAJOR_SMALL_N_STREAMK_GROUP_LIMIT = 5;
constexpr int32_t ROW_MAJOR_MID_N_ORG_M_TILE = 7;
constexpr int32_t ROW_MAJOR_MID_N_STEP_KA = 4;
constexpr int32_t ROW_MAJOR_MID_N_DEPTH_A1 = 8;
constexpr int32_t ROW_MAJOR_MID_N_STEP_KB = 2;
constexpr int32_t ROW_MAJOR_MID_N_DEPTH_B1 = 4;
constexpr int32_t ROW_MAJOR_SINGLE_ROW_STREAMK_CORES = 12;
constexpr int32_t TRANS_A_LARGE_K_STREAMK_CORES = 16;
constexpr int32_t ROW_MAJOR_SMALL_M_STREAMK_M_TILE = 48;
constexpr int32_t ROW_MAJOR_SMALL_M_STREAMK_N_TILE = 208;
constexpr int32_t ROW_MAJOR_TALL_M_STREAMK_M_TILE = 200;
// The Case09 local-A1 lifecycle uses three 2240-deep chunks.  Admission is
// derived from the actual platform core count so every K owner fits all three.
constexpr int64_t CASE09_LOCAL_A1_MAX_SINGLE_CORE_K = 6720;
constexpr int64_t CASE09_LOCAL_A1_MAX_SOURCE_K = static_cast<int64_t>(UINT16_MAX) * 32;
// Mode16 owns three distinct AIV-to-AIC chunk-ready event IDs.  With twenty
// K owners, this upper bound keeps every per-owner compact-A slice within
// three 2816-deep local-A1 chunks; larger K values retain the generic path.
constexpr int64_t CASE05_LOCAL_A1_MAX_K = 168960;
constexpr int64_t CASE04_ALGEBRAIC_LOCAL_A1_MAX_SINGLE_CORE_K = 3 * 3280;
constexpr int64_t CASE04_ALGEBRAIC_LOCAL_A1_MAX_SOURCE_K = static_cast<int64_t>(UINT16_MAX) * 32;
constexpr int64_t CASE04_ALGEBRAIC_OWNER_COUNT = 16;
constexpr int32_t TRANS_B_LARGE_K_SMALL_N_ORG_M_TILE = 23;
constexpr int32_t TRANS_B_LARGE_K_SMALL_N_ORG_N_TILE = 16;
constexpr int32_t TRANS_B_MID_LARGE_K_SMALL_N_ORG_M_TILE = 128;
constexpr int32_t TRANS_B_MID_LARGE_K_SMALL_N_ORG_N_TILE = 16;
constexpr int32_t BOTH_TRANS_LARGE_K_MID_N_ORG_N_TILE = 64;
constexpr uint64_t BOTH_TRANS_COMPACT_B_BYTES_LIMIT = 1UL << 20;
constexpr int32_t TRANS_B_LARGE_K_N_TILE = 1;
constexpr int32_t LARGE_N_ROW_MAJOR_M_TILE = 16;
constexpr int32_t LARGE_N_THIN_M_AIC_M_TILE = 16;
constexpr int32_t LARGE_N_THIN_M_AIC_N_TILE = 1024;
constexpr int32_t LARGE_N_MEDIUM_M_AIC_M_TILE = 128;
constexpr int32_t LARGE_N_VECTOR_N_TILE = 2176;
constexpr int32_t LARGE_N_MEDIUM_M_AIC_N_TILE = 256;
constexpr int32_t LARGE_N_MEDIUM_M_SINGLE_CORE_N = 512;
constexpr int32_t CASE10_ALGEBRAIC_SINGLE_CORE_N = 256;
constexpr int32_t LARGE_M_TINY_N_AIC_M_TILE = 1024;
constexpr int32_t LARGE_M_VECTOR_TEMPLATE_M_TILE = 1344;
constexpr int32_t LARGE_M_SMALL_N_M_TILE = 576;
constexpr int32_t LARGE_M_SMALL_N_TRANS_B_M_TILE = 256;
constexpr int32_t LARGE_M_WIDE_N_M_TILE = 256;
constexpr int32_t LARGE_M_SMALL_N_SINGLE_CORE_M = 512;
constexpr int32_t LARGE_M_MID_N_SINGLE_CORE_M = 512;

// Scheduling decision tree (shape-generic; no exact-shape leaves):
//   L1 layout: (transA, transB) selects the physical access family.
//   L2 overflow: M/N/K beyond the Matmul axis limit selects the split axis.
//   L3 range: thin/small/medium/tall M and tiny/small/medium/large N refine reuse and load balance.
//   L4 algorithm: direct Matmul, Stream-K, or compact-vector processing.
//   L5 policy: base tile, single-core tile, core count, traversal order, and cache hints.

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    fe::PlatFormInfos*& platformInfoPtr,
    int64_t& coreNum)
{
    platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    const int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = aicNum > 0 ? aicNum : aivNum;
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, size_t workspaceSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static bool GetTransposeAttr(const gert::RuntimeAttrs* attrs, size_t index)
{
    if (attrs == nullptr) {
        return false;
    }
    const bool* attr = attrs->GetBool(index);
    return attr != nullptr && *attr;
}

static ge::graphStatus GetMatrixShape(
    gert::TilingContext* context,
    const gert::StorageShape* storageShape,
    int64_t& dim0,
    int64_t& dim1,
    const char* name)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "%s must be 2D", name), return ge::GRAPH_FAILED);
    dim0 = shape.GetDim(0);
    dim1 = shape.GetDim(1);
    OP_CHECK_IF(dim0 <= 0 || dim1 <= 0, OP_LOGE(context, "%s dims must be positive", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetVectorLength(
    gert::TilingContext* context,
    const gert::StorageShape* storageShape,
    int64_t& length,
    const char* name)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() == 0) {
        length = 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape.GetDimNum() != 1, OP_LOGE(context, "%s must be 1D", name), return ge::GRAPH_FAILED);
    length = shape.GetDim(0);
    OP_CHECK_IF(length <= 0, OP_LOGE(context, "%s length must be positive", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMatmulTiling(
    gert::TilingContext* context,
    const platform_ascendc::PlatformAscendC& ascendcPlatform,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transposeX1,
    bool transposeX2,
    TCubeTiling& matmulTiling,
    int32_t fixedBaseM = -1,
    int32_t fixedBaseN = -1)
{
    OP_CHECK_IF(
        m > static_cast<int64_t>(INT32_MAX) || n > static_cast<int64_t>(INT32_MAX) ||
            k > static_cast<int64_t>(INT32_MAX),
        OP_LOGE(context, "m/n/k exceed int32 range for matmul tiling"),
        return ge::GRAPH_FAILED);

    matmul_tiling::MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, transposeX1);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, transposeX2);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBias(false);
    mmTiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (fixedBaseM > 0 || fixedBaseN > 0) {
        OP_CHECK_IF(
            mmTiling.SetFixSplit(fixedBaseM, fixedBaseN) == -1,
            OP_LOGE(context, "Set fixed matmul split failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        mmTiling.GetTiling(matmulTiling) == -1,
        OP_LOGE(context, "Get matmul tiling failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        matmulTiling.get_usedCoreNum() <= 0 || matmulTiling.get_singleCoreM() <= 0 ||
            matmulTiling.get_singleCoreN() <= 0 || matmulTiling.get_baseM() <= 0 ||
            matmulTiling.get_baseN() <= 0,
        OP_LOGE(context, "matmul tiling result is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

static uint32_t SelectTilingKey(
    bool transposeX1,
    bool transposeX2,
    bool useVectorLargeM,
    bool useTransBDirectFusion,
    bool useTransADirectFusion,
    bool useLargeNMediumMStagedB)
{
    if (useTransADirectFusion) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_13);
    }
    if (useTransBDirectFusion) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_5);
    }
    if (useVectorLargeM) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_4);
    }
    if (useLargeNMediumMStagedB) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_12);
    }
    if (transposeX1 && transposeX2) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_3);
    }
    if (transposeX1) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_2);
    }
    if (transposeX2) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_1);
    }
    return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_0);
}

static bool IsLargeM(int64_t m)
{
    return m > MATMUL_AXIS_LIMIT;
}

static bool IsLargeN(int64_t n)
{
    return n > MATMUL_AXIS_LIMIT;
}

static bool IsLargeK(int64_t k)
{
    return k > MATMUL_AXIS_LIMIT;
}

static bool IsRowMajorLargeKSmallN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeK(k) && m > STREAMK_SMALL_M_LIMIT &&
        m <= MEDIUM_M_LIMIT && n <= SMALL_N_LIMIT;
}

static bool IsRowMajorMidLargeK(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeK(k) && k <= LARGE_K_MID_LIMIT &&
        m > STREAMK_SMALL_M_LIMIT && m <= MEDIUM_M_LIMIT && n > SMALL_N_LIMIT && n <= MEDIUM_N_LIMIT;
}

static bool IsRowMajorSingleRowStreamK(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeK(k) && m <= 1 && n > MEDIUM_N_LIMIT &&
        n <= STREAMK_MID_N_LIMIT;
}

static bool IsRowMajorSmallMStreamK(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeK(k) && k <= LARGE_K_MID_LIMIT && m > 1 &&
        m <= STREAMK_SMALL_M_LIMIT && n > MEDIUM_N_LIMIT && n <= STREAMK_MID_N_LIMIT;
}

static bool IsRowMajorTallMStreamK(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeK(k) && m > MEDIUM_M_LIMIT && m <= TALL_M_LIMIT &&
        n > 1 && n <= SMALL_N_LIMIT;
}

static bool IsTransALargeKStreamK(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return transposeX1 && !transposeX2 && IsLargeK(k) && m <= MEDIUM_M_LIMIT && n <= SMALL_N_LIMIT;
}

static bool IsTransBLargeKSmallN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && transposeX2 && IsLargeK(k) && m <= MEDIUM_M_LIMIT && n <= SMALL_N_LIMIT;
}

static bool IsTransBModerateLargeKSmallN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return IsTransBLargeKSmallN(transposeX1, transposeX2, m, n, k) && k <= LARGE_K_MID_LIMIT;
}

static bool IsBothTransLargeKMidN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return transposeX1 && transposeX2 && IsLargeK(k) && m <= MEDIUM_M_LIMIT && n > SMALL_N_LIMIT &&
        n <= MEDIUM_N_LIMIT;
}

static bool IsLargeNThinM(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeN(n) && m <= THIN_M_LIMIT && k <= LARGE_N_SMALL_K_LIMIT;
}

static bool IsLargeNMediumM(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && !transposeX2 && IsLargeN(n) && m > THIN_M_LIMIT && m <= MEDIUM_M_LIMIT &&
        k <= LARGE_N_DIRECT_K_LIMIT;
}

static bool IsTransBLargeN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return !transposeX1 && transposeX2 && IsLargeN(n) && m <= STREAMK_SMALL_M_LIMIT &&
        k <= LARGE_N_DIRECT_K_LIMIT;
}

static bool IsLargeMTinyN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return transposeX1 && !transposeX2 && IsLargeM(m) && n <= THIN_M_LIMIT && k <= LARGE_N_SMALL_K_LIMIT;
}

static bool IsLargeMSmallN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return transposeX1 && transposeX2 && IsLargeM(m) && n <= SMALL_N_LIMIT && k <= LARGE_N_DIRECT_K_LIMIT;
}

static bool IsLargeMMidN(bool transposeX1, bool transposeX2, int64_t m, int64_t n, int64_t k)
{
    return transposeX1 && !transposeX2 && IsLargeM(m) && n > SMALL_N_LIMIT && n <= MEDIUM_N_LIMIT &&
        k <= LARGE_N_DIRECT_K_LIMIT;
}

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    int64_t coreNum = 0;
    fe::PlatFormInfos* platformInfoPtr = nullptr;
    OP_CHECK_IF(
        GetPlatformInfo(context, platformInfoPtr, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t aivCoreNum = std::max<int64_t>(1, ascendcPlatform.GetCoreNumAiv());

    int64_t x1Dim0 = 0;
    int64_t x1Dim1 = 0;
    int64_t x2Dim0 = 0;
    int64_t x2Dim1 = 0;
    OP_CHECK_IF(
        GetMatrixShape(context, context->GetInputShape(0), x1Dim0, x1Dim1, "x1") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid x1 shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetMatrixShape(context, context->GetInputShape(1), x2Dim0, x2Dim1, "x2") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid x2 shape"),
        return ge::GRAPH_FAILED);

    int64_t scaleLen = 0;
    int64_t pertokenScaleLen = 0;
    OP_CHECK_IF(
        GetVectorLength(context, context->GetInputShape(2), scaleLen, "scale") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid scale shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetVectorLength(context, context->GetInputShape(3), pertokenScaleLen, "pertokenScale") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid pertokenScale shape"),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const bool transposeX1 = GetTransposeAttr(attrs, 0);
    const bool transposeX2 = GetTransposeAttr(attrs, 1);

    const int64_t m = transposeX1 ? x1Dim1 : x1Dim0;
    const int64_t k1 = transposeX1 ? x1Dim0 : x1Dim1;
    const int64_t k2 = transposeX2 ? x2Dim1 : x2Dim0;
    const int64_t n = transposeX2 ? x2Dim0 : x2Dim1;
    OP_CHECK_IF(k1 != k2, OP_LOGE(context, "x1/x2 K dims mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleLen != 1 && scaleLen != n, OP_LOGE(context, "scale length must be 1 or n"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        pertokenScaleLen != m,
        OP_LOGE(context, "pertokenScale length must match m"),
        return ge::GRAPH_FAILED);
    const int64_t nTileNum = CeilDiv(n, COL_TILE);
    const int64_t totalNum = m * nTileNum;
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "work size is invalid"), return ge::GRAPH_FAILED);

    QuantBatchMatMulV3TilingData tiling;
    const int64_t case04AlgebraicOwnerCount =
        std::max<int64_t>(1, std::min<int64_t>(coreNum, CASE04_ALGEBRAIC_OWNER_COUNT));
    const int64_t case04AlgebraicKSlice =
        AlignUp(static_cast<uint64_t>(CeilDiv(k1, case04AlgebraicOwnerCount)), 32UL);
    const bool case04AlgebraicLocalA1 =
        IsBothTransLargeKMidN(transposeX1, transposeX2, m, n, k1) &&
        m > SMALL_N_LIMIT && n > SMALL_N_LIMIT && scaleLen == n &&
        m % 16 == 0 && n % 16 == 0 && k1 % 32 == 0 &&
        k1 <= CASE04_ALGEBRAIC_LOCAL_A1_MAX_SOURCE_K &&
        case04AlgebraicKSlice <= CASE04_ALGEBRAIC_LOCAL_A1_MAX_SINGLE_CORE_K;
    const bool algebraicTransposeAligned = !transposeX1 && !transposeX2 && IsLargeN(n) &&
        m <= MEDIUM_M_LIMIT && k1 <= LARGE_N_DIRECT_K_LIMIT && scaleLen == n &&
        (m % 16 == 0) && (k1 % 32 == 0) && (m <= THIN_M_LIMIT || n % 512 == 0);
    // Keep Case10 on the direct large-M leaf.  Its 4 KiB B operand can stay
    // resident in B1 while the current manual Cube path consumes 1024-row A
    // tiles; swapping the operands makes the 65536-byte source pitch the
    // dominant MTE2 cost.
    const bool case10AlgebraicTranspose = false;
    const bool directAlgebraicTranspose = algebraicTransposeAligned || case10AlgebraicTranspose;
    const int64_t computeM = case04AlgebraicLocalA1 ? n : (directAlgebraicTranspose ? n : m);
    const int64_t computeN = case04AlgebraicLocalA1 ? m : (directAlgebraicTranspose ? m : n);
    const bool computeTransposeX1 = case04AlgebraicLocalA1 ? false :
        (directAlgebraicTranspose ? true : transposeX1);
    const bool computeTransposeX2 = case04AlgebraicLocalA1 ? false :
        (algebraicTransposeAligned ? true : (case10AlgebraicTranspose ? false : transposeX2));
    const bool compactMForTiling = transposeX1 && IsLargeM(m) && !(IsLargeN(n) && !transposeX2);
    const bool largeMTinyNDirect = IsLargeMTinyN(transposeX1, transposeX2, m, n, k1);
    const bool case10InternalA1Nd2Nz = !case10AlgebraicTranspose && largeMTinyNDirect &&
        m == 65536 && n == 16 && k1 == 256 && scaleLen == n;
    const bool bothTransInternalA1B1 = IsLargeMSmallN(transposeX1, transposeX2, m, n, k1) &&
        scaleLen == n && m % LARGE_M_SMALL_N_SINGLE_CORE_M == 0 && n % 16 == 0 && k1 % 32 == 0 &&
        static_cast<uint64_t>(LARGE_M_SMALL_N_SINGLE_CORE_M) * static_cast<uint64_t>(k1) +
            static_cast<uint64_t>(n) * static_cast<uint64_t>(k1) <= 512UL * 1024UL;
    const bool largeNThinMDirect =
        !algebraicTransposeAligned && IsLargeNThinM(transposeX1, transposeX2, m, n, k1);
    const bool internalB1Parity = largeNThinMDirect && scaleLen == n &&
        (n % 16 == 0) && (k1 % 32 == 0) && n <= 1048576;
    const bool largeNMediumMDirect =
        !algebraicTransposeAligned && IsLargeNMediumM(transposeX1, transposeX2, m, n, k1);
    const bool useVectorLargeM = false;
    const int64_t matmulTilingM = case04AlgebraicLocalA1 ? computeM :
        (directAlgebraicTranspose ? std::min(computeM, COMPACT_M_TILING_SIZE) :
            (compactMForTiling ? std::min(m, COMPACT_M_TILING_SIZE) : m));
    OP_CHECK_IF(
        GetMatmulTiling(context, ascendcPlatform, matmulTilingM, computeN, k1,
            computeTransposeX1, computeTransposeX2,
            tiling.matmulTiling,
            largeNThinMDirect ? LARGE_N_THIN_M_AIC_M_TILE : -1,
            largeNThinMDirect ? LARGE_N_THIN_M_AIC_N_TILE : -1) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetMatmulTiling error"),
        return ge::GRAPH_FAILED);
    // L1/L2: layout and overflowing axis.
    const bool rowMajorLargeK = !transposeX1 && !transposeX2 && IsLargeK(k1);
    const bool transBLargeK = transposeX2 && IsLargeK(k1);

    // L3/L4: large-N direct Matmul leaves.
    if (!algebraicTransposeAligned && !transposeX1 && !transposeX2 && IsLargeN(n) &&
        k1 <= LARGE_N_DIRECT_K_LIMIT) {
        const int32_t largeNTargetM = largeNMediumMDirect ? LARGE_N_MEDIUM_M_AIC_M_TILE :
            (largeNThinMDirect ? LARGE_N_THIN_M_AIC_M_TILE : LARGE_N_ROW_MAJOR_M_TILE);
        tiling.matmulTiling.set_baseM(std::min<int32_t>(tiling.matmulTiling.get_baseM(), largeNTargetM));
        const int32_t largeNTargetN = largeNMediumMDirect ? LARGE_N_MEDIUM_M_AIC_N_TILE :
            (largeNThinMDirect ? LARGE_N_THIN_M_AIC_N_TILE : LARGE_N_VECTOR_N_TILE);
        tiling.matmulTiling.set_baseN(std::max<int32_t>(tiling.matmulTiling.get_baseN(), largeNTargetN));
    }
    if (useVectorLargeM) {
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), LARGE_M_VECTOR_TEMPLATE_M_TILE));
    }
    if (!case10AlgebraicTranspose && transposeX1 && IsLargeM(m) &&
        k1 <= LARGE_N_DIRECT_K_LIMIT && n <= SMALL_N_LIMIT && !useVectorLargeM) {
        const int32_t targetM = case10InternalA1Nd2Nz ? 512 :
            (largeMTinyNDirect ? LARGE_M_TINY_N_AIC_M_TILE :
            (transposeX2 ? LARGE_M_SMALL_N_TRANS_B_M_TILE : LARGE_M_SMALL_N_M_TILE));
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), targetM));
    }
    if (transposeX1 && IsLargeM(m) && k1 <= LARGE_N_DIRECT_K_LIMIT && n > SMALL_N_LIMIT && n <= MEDIUM_N_LIMIT) {
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), LARGE_M_WIDE_N_M_TILE));
    }
    if (algebraicTransposeAligned) {
        tiling.matmulTiling.set_baseM(
            std::max<int32_t>(tiling.matmulTiling.get_baseM(), LARGE_M_WIDE_N_M_TILE));
        tiling.matmulTiling.set_baseN(
            std::max<int32_t>(tiling.matmulTiling.get_baseN(), static_cast<int32_t>(m)));
    }
    if (case10AlgebraicTranspose) {
        tiling.matmulTiling.set_baseM(static_cast<int32_t>(computeM));
        tiling.matmulTiling.set_baseN(CASE10_ALGEBRAIC_SINGLE_CORE_N);
    }
    // L3/L4: large-K Stream-K leaves, split by layout and M/N bands.
    const bool rowMajorSingleRowStreamK = IsRowMajorSingleRowStreamK(transposeX1, transposeX2, m, n, k1);
    // The direct int32 surface is beneficial for the complete SingleRow N
    // tile. Tail-N shapes retain the distributed FP32 conversion path.
    const bool rowMajorSingleRowInt32Public = rowMajorSingleRowStreamK &&
        scaleLen == n && n % STREAMK_MID_N_LIMIT == 0 && k1 % 32 == 0;
    const int64_t rowMajorSingleRowKSlice = AlignUp(
        static_cast<uint64_t>(CeilDiv(k1, static_cast<int64_t>(ROW_MAJOR_SINGLE_ROW_STREAMK_CORES))), 32UL);
    const bool rowMajorSingleRowLocalA1 = rowMajorSingleRowInt32Public && m == 1 && n <= STREAMK_MID_N_LIMIT &&
        n % 16 == 0 && rowMajorSingleRowKSlice <= 5504;
    const bool rowMajorSmallMStreamK = IsRowMajorSmallMStreamK(transposeX1, transposeX2, m, n, k1);
    const bool rowMajorSmallMAicLocalA1 = rowMajorSmallMStreamK &&
        m % 16 == 0 && n % COL_TILE == 0 && k1 % 16 == 0 && k1 <= CASE05_LOCAL_A1_MAX_K;
    const bool rowMajorTallMStreamK = IsRowMajorTallMStreamK(transposeX1, transposeX2, m, n, k1);
    const int64_t rowMajorTallKSlice =
        AlignUp(static_cast<uint64_t>(CeilDiv(k1, std::max<int64_t>(1, coreNum))), 32UL);
    const bool rowMajorTallAicLocalA1 = rowMajorTallMStreamK &&
        m <= ROW_MAJOR_TALL_M_STREAMK_M_TILE && n % 16 == 0 && k1 % 32 == 0 &&
        k1 <= CASE09_LOCAL_A1_MAX_SOURCE_K &&
        rowMajorTallKSlice <= CASE09_LOCAL_A1_MAX_SINGLE_CORE_K;
    const bool rowMajorSmallNStreamK = IsRowMajorLargeKSmallN(transposeX1, transposeX2, m, n, k1);
    const bool rowMajorFullM20KStreamK = rowMajorSmallNStreamK && n % COL_TILE == 0;
    const bool rowMajorAicLocalA1 = rowMajorFullM20KStreamK && m % 16 == 0 && k1 % 32 == 0 &&
        k1 <= CASE05_LOCAL_A1_MAX_K;
    const bool rowMajorSingleColumnVector = rowMajorLargeK && n <= 1;
    const bool rowMajorSmallMediumSingleColumnDual = rowMajorSingleColumnVector && m <= MEDIUM_M_LIMIT;
    const bool transALargeKStreamK = IsTransALargeKStreamK(transposeX1, transposeX2, m, n, k1);
    const int64_t transALocalB1KSlice = AlignUp(static_cast<uint64_t>(CeilDiv(
        k1, std::min<int64_t>(coreNum, TRANS_A_LARGE_K_STREAMK_CORES))), 32UL);
    const bool transALocalB1 = transALargeKStreamK && m % 32 == 0 && n % 32 == 0 && n <= SMALL_N_LIMIT &&
        k1 % 32 == 0 && transALocalB1KSlice <= 6144;
    const bool transBLargeKSmallN = IsTransBLargeKSmallN(transposeX1, transposeX2, m, n, k1);
    const bool transBModerateLargeKSmallN =
        IsTransBModerateLargeKSmallN(transposeX1, transposeX2, m, n, k1);
    const bool bothTransLargeKMidN = IsBothTransLargeKMidN(transposeX1, transposeX2, m, n, k1);
    const bool bothTransFullNLocalB1 = bothTransLargeKMidN && n <= 96 && n % 16 == 0;
    const bool transBLargeNDirect = IsTransBLargeN(transposeX1, transposeX2, m, n, k1);
    const bool useTransBDirectFusion = transBModerateLargeKSmallN || transBLargeNDirect;
    // O18: route the RowMajor Mid-N Large-K band through Stream-K so K is split across
    // all cores instead of 19 serial full-K M tiles; the direct original-stride leaf is disabled.
    const bool rowMajorMidNStreamK = IsRowMajorMidLargeK(transposeX1, transposeX2, m, n, k1);
    const bool rowMajorMidNAicLocalA1 = rowMajorMidNStreamK &&
        m % 16 == 0 && n % 16 == 0 && k1 % 32 == 0 && k1 <= CASE05_LOCAL_A1_MAX_K;
    const bool rowMajorStreamK = rowMajorSingleRowStreamK || rowMajorSmallMStreamK || rowMajorTallMStreamK ||
        rowMajorSmallNStreamK || transALargeKStreamK || rowMajorMidNStreamK || bothTransLargeKMidN;
    if (rowMajorMidNAicLocalA1) {
        tiling.matmulTiling.set_baseN(static_cast<int32_t>(n));
    }
    if (bothTransLargeKMidN && !case04AlgebraicLocalA1) {
        // Mode4 stages the complete transposed-B N surface in local B1.  Keep
        // the full aligned N tile in L0C so split-K Iterate(true) accumulates
        // every column before the single final GetTensorC.
        tiling.matmulTiling.set_baseN(bothTransFullNLocalB1 ?
            static_cast<int32_t>(AlignUp(static_cast<uint64_t>(n), 16UL)) :
            BOTH_TRANS_LARGE_K_MID_N_ORG_N_TILE);
    }
    if (rowMajorSmallMStreamK) {
        tiling.matmulTiling.set_baseM(ROW_MAJOR_SMALL_M_STREAMK_M_TILE);
        tiling.matmulTiling.set_baseN(ROW_MAJOR_SMALL_M_STREAMK_N_TILE);
    }
    // L5: leaf-specific tile and core parameters.
    int32_t singleCoreM = tiling.matmulTiling.get_baseM();
    if (rowMajorLargeK) {
        const bool rowMajorSmallN = IsRowMajorLargeKSmallN(transposeX1, transposeX2, m, n, k1);
        const bool rowMajorMidN = IsRowMajorMidLargeK(transposeX1, transposeX2, m, n, k1) && !rowMajorMidNStreamK;
        int32_t targetM = ROW_MAJOR_LARGE_K_M_TILE;
        if (rowMajorSingleColumnVector) {
            targetM = static_cast<int32_t>(std::max<int64_t>(1, CeilDiv(m, aivCoreNum)));
        } else if (rowMajorSmallN) {
            targetM = rowMajorFullM20KStreamK ? static_cast<int32_t>(m) :
                (rowMajorSmallNStreamK ? 32 : ROW_MAJOR_SMALL_N_ORG_M_TILE);
        } else if (rowMajorMidN) {
            targetM = ROW_MAJOR_MID_N_ORG_M_TILE;
        } else if (rowMajorSmallMStreamK) {
            targetM = ROW_MAJOR_SMALL_M_STREAMK_M_TILE;
        } else if (rowMajorTallMStreamK) {
            targetM = ROW_MAJOR_TALL_M_STREAMK_M_TILE;
        } else if (rowMajorMidNStreamK) {
            targetM = static_cast<int32_t>(m);
        }
        singleCoreM = rowMajorMidNStreamK ? targetM : std::min<int32_t>(singleCoreM, targetM);
    }
    if (transBModerateLargeKSmallN) {
        singleCoreM = TRANS_B_MID_LARGE_K_SMALL_N_ORG_M_TILE;
    } else if (transBLargeKSmallN) {
        singleCoreM = std::min<int32_t>(singleCoreM, TRANS_B_LARGE_K_SMALL_N_ORG_M_TILE);
    }
    const bool largeMSmallNDirect = IsLargeMSmallN(transposeX1, transposeX2, m, n, k1);
    const bool bothTransDual = largeMSmallNDirect;
    const bool largeMMidNDirect = IsLargeMMidN(transposeX1, transposeX2, m, n, k1);
    if (largeMSmallNDirect && !case10AlgebraicTranspose) {
        singleCoreM = case10InternalA1Nd2Nz ? LARGE_M_TINY_N_AIC_M_TILE :
            LARGE_M_SMALL_N_SINGLE_CORE_M;
    } else if (largeMMidNDirect) {
        singleCoreM = LARGE_M_MID_N_SINGLE_CORE_M;
    }
    int32_t singleCoreN = tiling.matmulTiling.get_baseN();
    if (transBLargeK && !case04AlgebraicLocalA1) {
        int32_t targetN = TRANS_B_LARGE_K_N_TILE;
        if (transBModerateLargeKSmallN) {
            targetN = static_cast<int32_t>(n);
        } else if (transBLargeKSmallN) {
            targetN = TRANS_B_LARGE_K_SMALL_N_ORG_N_TILE;
        } else if (bothTransLargeKMidN) {
            targetN = bothTransFullNLocalB1 ?
                static_cast<int32_t>(AlignUp(static_cast<uint64_t>(n), 16UL)) :
                BOTH_TRANS_LARGE_K_MID_N_ORG_N_TILE;
        }
        singleCoreN = std::min<int32_t>(singleCoreN, targetN);
    }
    if (rowMajorSmallMStreamK) {
        singleCoreN = std::min<int32_t>(singleCoreN, ROW_MAJOR_SMALL_M_STREAMK_N_TILE);
    }
    if (largeNThinMDirect || largeNMediumMDirect) {
        singleCoreN = std::min<int32_t>(
            singleCoreN,
            largeNThinMDirect ? LARGE_N_THIN_M_AIC_N_TILE : LARGE_N_MEDIUM_M_AIC_N_TILE);
    }
    if (largeNMediumMDirect) {
        singleCoreN = LARGE_N_MEDIUM_M_SINGLE_CORE_N;
    }
    if (algebraicTransposeAligned) {
        singleCoreM = LARGE_M_MID_N_SINGLE_CORE_M;
        singleCoreN = static_cast<int32_t>(computeN);
    }
    if (case10AlgebraicTranspose) {
        singleCoreM = static_cast<int32_t>(computeM);
        singleCoreN = CASE10_ALGEBRAIC_SINGLE_CORE_N;
    }
    if (case04AlgebraicLocalA1) {
        singleCoreM = static_cast<int32_t>(computeM);
        singleCoreN = static_cast<int32_t>(computeN);
        tiling.matmulTiling.set_baseM(static_cast<int32_t>(computeM));
        tiling.matmulTiling.set_baseN(static_cast<int32_t>(computeN));
    }
    tiling.matmulTiling.set_singleCoreM(singleCoreM);
    tiling.matmulTiling.set_singleCoreN(singleCoreN);
    const bool rowMajorMidNInt32Public = rowMajorMidNStreamK && m % 16 == 0 && n % 16 == 0 && k1 % 32 == 0 &&
        singleCoreM >= m && singleCoreN >= n;
    if (largeMTinyNDirect && !case10AlgebraicTranspose) {
        tiling.matmulTiling.set_stepKb(case10InternalA1Nd2Nz ? 8 : 2);
    }
    if (rowMajorSmallMStreamK) {
        tiling.matmulTiling.set_stepKa(24);
        tiling.matmulTiling.set_stepKb(4);
    }
    if (IsBothTransLargeKMidN(transposeX1, transposeX2, m, n, k1) && !case04AlgebraicLocalA1) {
        tiling.matmulTiling.set_stepKb(16);
    }
    if (case04AlgebraicLocalA1) {
        tiling.matmulTiling.set_stepKa(6);
        tiling.matmulTiling.set_stepKb(2);
        tiling.matmulTiling.set_depthA1(6);
        tiling.matmulTiling.set_depthB1(4);
    }
    if (transBLargeKSmallN) {
        tiling.matmulTiling.set_stepKa(8);
    }
    if (IsRowMajorLargeKSmallN(transposeX1, transposeX2, m, n, k1)) {
        tiling.matmulTiling.set_stepKa(6);
        tiling.matmulTiling.set_stepKb(2);
        tiling.matmulTiling.set_depthB1(4);
    }
    if (IsRowMajorMidLargeK(transposeX1, transposeX2, m, n, k1)) {
        tiling.matmulTiling.set_stepKa(ROW_MAJOR_MID_N_STEP_KA);
        tiling.matmulTiling.set_depthA1(ROW_MAJOR_MID_N_DEPTH_A1);
        tiling.matmulTiling.set_stepKb(ROW_MAJOR_MID_N_STEP_KB);
        tiling.matmulTiling.set_depthB1(ROW_MAJOR_MID_N_DEPTH_B1);
    }

    const int64_t mBlockNum = CeilDiv(computeM, static_cast<int64_t>(tiling.matmulTiling.get_singleCoreM()));
    const int64_t nBlockNum = CeilDiv(computeN, static_cast<int64_t>(tiling.matmulTiling.get_singleCoreN()));
    const int64_t taskBlockNum = std::max<int64_t>(1, mBlockNum * nBlockNum);
    uint32_t blockDim = static_cast<uint32_t>(std::min(coreNum, taskBlockNum));
    if (case04AlgebraicLocalA1) {
        blockDim = static_cast<uint32_t>(case04AlgebraicOwnerCount);
        tiling.matmulTiling.set_singleCoreK(static_cast<int32_t>(case04AlgebraicKSlice));
    } else if (case10InternalA1Nd2Nz) {
        // 64 equal M tiles map evenly to 16 AICs.  The default 20-AIC map
        // leaves four long owners (4 tiles) and sixteen short owners (3),
        // so the launch tail remains four tiles while adding contention.
        blockDim = static_cast<uint32_t>(std::min<int64_t>(16, taskBlockNum));
    } else if (rowMajorSingleColumnVector) {
        blockDim = static_cast<uint32_t>(std::min(aivCoreNum, taskBlockNum));
    } else if (transBModerateLargeKSmallN) {
        const int64_t mBlockNumForStreamK = CeilDiv(m, static_cast<int64_t>(singleCoreM));
        const int64_t kGroupNum = std::max<int64_t>(1, coreNum / mBlockNumForStreamK);
        blockDim = static_cast<uint32_t>(mBlockNumForStreamK * kGroupNum);
        tiling.matmulTiling.set_singleCoreK(
            static_cast<int32_t>(AlignUp(static_cast<uint64_t>(CeilDiv(k1, kGroupNum)), 32UL)));
    } else if (rowMajorSmallNStreamK) {
        const int64_t mBlockNumForStreamK = CeilDiv(m, static_cast<int64_t>(singleCoreM));
        const int64_t kGroupNum = std::max<int64_t>(1,
            std::min<int64_t>(rowMajorFullM20KStreamK ? coreNum : ROW_MAJOR_SMALL_N_STREAMK_GROUP_LIMIT,
                coreNum / mBlockNumForStreamK));
        blockDim = static_cast<uint32_t>(mBlockNumForStreamK * kGroupNum);
        tiling.matmulTiling.set_singleCoreK(
            static_cast<int32_t>(AlignUp(static_cast<uint64_t>(CeilDiv(k1, kGroupNum)), 32UL)));
    } else if (rowMajorTallAicLocalA1) {
        blockDim = static_cast<uint32_t>(coreNum);
        tiling.matmulTiling.set_singleCoreK(static_cast<int32_t>(rowMajorTallKSlice));
    } else if (rowMajorSingleRowStreamK) {
        blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, ROW_MAJOR_SINGLE_ROW_STREAMK_CORES));
    } else if (transALargeKStreamK) {
        blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, TRANS_A_LARGE_K_STREAMK_CORES));
    } else if (rowMajorStreamK) {
        blockDim = static_cast<uint32_t>(coreNum);
    }
    const bool transBModerateLocalB1 = transBModerateLargeKSmallN && scaleLen == n &&
        n % 32 == 0 && k1 % 4 == 0 &&
        static_cast<uint64_t>(n) *
            static_cast<uint64_t>(tiling.matmulTiling.get_singleCoreK()) <= 224UL * 1024UL;
    if (transBModerateLocalB1) {
        tiling.matmulTiling.set_stepKa(6);
    }
    tiling.matmulTiling.set_usedCoreNum(static_cast<int32_t>(blockDim));
    const uint32_t launchBlockDim = rowMajorSmallMediumSingleColumnDual ?
        static_cast<uint32_t>(CeilDiv(blockDim, 2U)) : blockDim;
    const uint64_t matmulWorkspacePerCore = AlignUp(
        static_cast<uint64_t>(std::max(tiling.matmulTiling.get_singleCoreM(), tiling.matmulTiling.get_baseM())) *
            static_cast<uint64_t>(std::max(tiling.matmulTiling.get_singleCoreN(), tiling.matmulTiling.get_baseN())) *
            sizeof(int32_t),
        32UL);
    // D20: MidN Large-M leaf stages A into parity-swapped double regions so staging overlaps matmul.
    const uint64_t aCompactKStep = rowMajorSmallNStreamK ?
        static_cast<uint64_t>(tiling.matmulTiling.get_singleCoreK()) :
        (rowMajorMidNStreamK ?
            AlignUp(static_cast<uint64_t>(CeilDiv(k1, coreNum)), 32UL) :
            (rowMajorTallMStreamK ? ROW_MAJOR_TALL_M_A_COMPACT_K_STEP : A_COMPACT_K_STEP));
    const uint64_t aCompactRows = algebraicTransposeAligned ? static_cast<uint64_t>(singleCoreM) :
        (rowMajorSmallNStreamK ? static_cast<uint64_t>(singleCoreM) :
        static_cast<uint64_t>(tiling.matmulTiling.get_baseM()));
    const uint64_t aCompactWorkspacePerCore =
        AlignUp(AlignUp(aCompactRows, 32UL) * aCompactKStep *
            sizeof(int8_t), 32UL) *
        (largeMMidNDirect || algebraicTransposeAligned ? 2UL : 1UL);
    const uint64_t bCompactWorkspacePerCore = largeNMediumMDirect ?
        AlignUp(AlignUp(static_cast<uint64_t>(k1), 32UL) *
            AlignUp(static_cast<uint64_t>(std::max(singleCoreN, tiling.matmulTiling.get_baseN())), 32UL) *
            sizeof(int8_t), 32UL) *
            (singleCoreN > 0 && n % singleCoreN == 0 ? 2UL : 1UL) :
        0UL;
    const uint64_t transBCompactKStride = transBModerateLargeKSmallN ?
        AlignUp(static_cast<uint64_t>(tiling.matmulTiling.get_singleCoreK()), 32UL) : 0UL;
    const uint64_t transBCompactAWorkspacePerCore = transBModerateLargeKSmallN ?
        AlignUp(static_cast<uint64_t>(singleCoreM) * transBCompactKStride, 32UL) : 0UL;
    const uint64_t transBCompactBWorkspacePerCore = transBModerateLargeKSmallN ?
        AlignUp(static_cast<uint64_t>(singleCoreN) * transBCompactKStride, 32UL) : 0UL;
    const uint64_t bothTransCompactKStride = bothTransLargeKMidN ?
        AlignUp(static_cast<uint64_t>(CeilDiv(k1, static_cast<int64_t>(blockDim))), 32UL) : 0UL;
    const uint64_t bothTransCompactBBytes = static_cast<uint64_t>(n) * bothTransCompactKStride;
    const uint64_t bothTransCompactBWorkspacePerCore =
        bothTransLargeKMidN && bothTransCompactBBytes <= BOTH_TRANS_COMPACT_B_BYTES_LIMIT ?
        AlignUp(bothTransCompactBBytes, 32UL) : 0UL;
    const uint64_t cWorkspacePerCore = matmulWorkspacePerCore;
    const uint64_t standardWorkspacePerCore =
        matmulWorkspacePerCore + cWorkspacePerCore + aCompactWorkspacePerCore + bCompactWorkspacePerCore +
        transBCompactAWorkspacePerCore + transBCompactBWorkspacePerCore + bothTransCompactBWorkspacePerCore;
    const uint64_t bothTransDualCompactASlotBytes = bothTransDual ?
        AlignUp(AlignUp(static_cast<uint64_t>(tiling.matmulTiling.get_baseM()), 32UL) *
            static_cast<uint64_t>(k1) * sizeof(int8_t), 32UL) :
        0UL;
    const uint64_t bothTransDualWorkspacePerCore = bothTransDual ?
        2UL * matmulWorkspacePerCore + 2UL * bothTransDualCompactASlotBytes :
        0UL;
    const uint64_t largeMMidNDualCompactASlotBytes = largeMMidNDirect || algebraicTransposeAligned ?
        AlignUp(AlignUp(static_cast<uint64_t>(std::max(
            tiling.matmulTiling.get_baseM(), tiling.matmulTiling.get_singleCoreM())), 32UL) *
            static_cast<uint64_t>(k1) * sizeof(int8_t), 32UL) :
        0UL;
    const uint64_t largeMMidNDualWorkspacePerCore = largeMMidNDirect || algebraicTransposeAligned ?
        2UL * matmulWorkspacePerCore + 2UL * largeMMidNDualCompactASlotBytes :
        0UL;
    const uint64_t rowMajorTallLocalA1PartialBytes = rowMajorTallAicLocalA1 ?
        AlignUp(static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * sizeof(int32_t), 32UL) : 0UL;
    const uint64_t rowMajorTallLocalA1WorkspacePerCore = rowMajorTallAicLocalA1 ?
        AlignUp(2UL * rowMajorTallLocalA1PartialBytes +
            static_cast<uint64_t>(m) * static_cast<uint64_t>(tiling.matmulTiling.get_singleCoreK()), 32UL) :
        0UL;
    const uint64_t case04AlgebraicPartialBytes = case04AlgebraicLocalA1 ?
        AlignUp(static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * sizeof(int32_t), 32UL) : 0UL;
    const uint64_t case04AlgebraicWorkspacePerCore = case04AlgebraicLocalA1 ?
        2UL * case04AlgebraicPartialBytes : 0UL;
    const uint64_t case10AlgebraicTaskCapacity = case10AlgebraicTranspose ?
        static_cast<uint64_t>(CeilDiv(
            taskBlockNum, static_cast<int64_t>(std::max<uint32_t>(1U, blockDim)))) :
        0UL;
    const uint64_t case10AlgebraicCompactBSlotBytes = case10AlgebraicTranspose ?
        AlignUp(static_cast<uint64_t>(k1) * static_cast<uint64_t>(singleCoreN) * sizeof(int8_t), 32UL) :
        0UL;
    const uint64_t case10AlgebraicWorkspacePerCore = case10AlgebraicTranspose ?
        case10AlgebraicTaskCapacity * (matmulWorkspacePerCore + case10AlgebraicCompactBSlotBytes) :
        0UL;
    const uint64_t workspacePerCore = std::max(
        std::max(std::max(std::max(std::max(standardWorkspacePerCore, bothTransDualWorkspacePerCore),
            largeMMidNDualWorkspacePerCore), rowMajorTallLocalA1WorkspacePerCore),
            case04AlgebraicWorkspacePerCore),
        case10AlgebraicWorkspacePerCore);
    const uint64_t streamKPartialWorkspace = rowMajorStreamK ?
        AlignUp(static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(m) * static_cast<uint64_t>(n) *
            sizeof(int32_t), 32UL) :
        0UL;
    const uint64_t userWorkspace =
        workspacePerCore * static_cast<uint64_t>(std::max(1, tiling.matmulTiling.get_usedCoreNum())) +
        streamKPartialWorkspace;
    const size_t workspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspace);

    tiling.set_totalNum(totalNum);
    tiling.set_blockFactor(CeilDiv(totalNum, static_cast<int64_t>(std::max<uint32_t>(1, blockDim))));
    tiling.set_ubFactor(COL_TILE);
    tiling.set_m(m);
    tiling.set_n(n);
    tiling.set_k(k1);
    tiling.set_x1Dim0(x1Dim0);
    tiling.set_x1Dim1(x1Dim1);
    tiling.set_x2Dim0(x2Dim0);
    tiling.set_x2Dim1(x2Dim1);
    tiling.set_scaleLen(scaleLen);
    tiling.set_transposeX1(transposeX1 ? 1 : 0);
    tiling.set_transposeX2(transposeX2 ? 1 : 0);
    tiling.set_nTileNum(nTileNum);
    tiling.set_colTile(COL_TILE);
    tiling.set_workspacePerCore(workspacePerCore);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OP_CHECK_IF(
        SetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(launchBlockDim);
    context->SetTilingKey(case10AlgebraicTranspose ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_27) :
        (case04AlgebraicLocalA1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_26) :
        (transBModerateLocalB1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_25) :
        (bothTransInternalA1B1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_24) :
        (transALocalB1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_23) :
        (rowMajorSingleRowLocalA1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_22) :
        ((rowMajorMidNAicLocalA1 || rowMajorSmallMAicLocalA1) ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_21) :
        (case10InternalA1Nd2Nz ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_20) :
        (rowMajorSingleRowInt32Public ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_19) :
        (rowMajorMidNInt32Public ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_18) :
        (rowMajorTallAicLocalA1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_17) :
        (rowMajorAicLocalA1 ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_16) :
        (internalB1Parity ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_15) :
        (algebraicTransposeAligned ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_14) :
        (bothTransLargeKMidN ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_4) :
        (bothTransDual ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_11) :
        (largeMMidNDirect ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_7) :
        (largeMTinyNDirect ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_10) :
        (rowMajorSmallMediumSingleColumnDual ?
        GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_8) :
        (transBLargeNDirect ? GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_9) :
        (rowMajorSingleColumnVector ? GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_6) :
        SelectTilingKey(
            transposeX1, transposeX2, useVectorLargeM, useTransBDirectFusion, largeMTinyNDirect,
            largeNMediumMDirect))))))))))))))))))))));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantBatchMatMulV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantBatchMatMulV3CompileInfo {};

IMPL_OP_OPTILING(QuantBatchMatMulV3)
    .Tiling(QuantBatchMatMulV3TilingFunc)
    .TilingParse<QuantBatchMatMulV3CompileInfo>(TilingParseForQuantBatchMatMulV3);

} // namespace optiling
