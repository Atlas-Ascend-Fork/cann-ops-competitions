// Portable geometry for row-major permutation with torch.permute semantics.
// This header has no CANN dependency. It validates the permutation, removes
// unit axes, merges adjacent contiguous axes, derives the shared contiguous
// suffix, and detects layouts that need only a flat copy.

#ifndef ASCEND_S9_TRANSPOSE_GEOMETRY_H
#define ASCEND_S9_TRANSPOSE_GEOMETRY_H

#include <cstdint>

namespace s9 {
namespace transpose {

// The published S9 spec tops out at rank 5. Keep a little headroom and fail
// closed above it rather than writing out of bounds.
constexpr int kMaxRank = 8;

enum class GeometryStatus : int {
    kOk = 0,
    kRankOutOfRange = 1,      // rank < 1 or rank > kMaxRank
    kNegativeExtent = 2,      // an extent < 0
    kPermLengthMismatch = 3,  // len(perm) != rank
    kPermEntryOutOfRange = 4, // entry outside [-rank, rank)
    kPermNotBijective = 5,    // repeats an axis after negative normalization
};

struct Geometry {
    GeometryStatus status = GeometryStatus::kRankOutOfRange;

    int rank = 0;
    int64_t shape[kMaxRank] = {};
    int32_t perm[kMaxRank] = {};          // normalized, all non-negative
    int64_t outputShape[kMaxRank] = {};
    int64_t inputStride[kMaxRank] = {};   // row-major, in elements
    int64_t outputStride[kMaxRank] = {};  // row-major over outputShape

    // Axes with extent 1 contribute a constant 0 to the source offset, so they
    // can be dropped without changing the element order.
    int activeRank = 0;
    int32_t activeInput[kMaxRank] = {};   // input axis ids, ascending
    int32_t activeOutput[kMaxRank] = {};  // input axis ids, in output order

    // Adjacent axes that stay adjacent and ascending in both orders behave as
    // one fatter axis. Collapsing them is what lets one kernel serve a whole
    // family of shapes instead of one published case.
    int collapsedRank = 0;
    int64_t collapsedShape[kMaxRank] = {};  // in input-group order
    int32_t collapsedPerm[kMaxRank] = {};   // output-group order -> input group

    int64_t numel = 0;
    // Elements guaranteed contiguous in the source AND written contiguously.
    int64_t innermostContiguous = 0;
    int64_t contiguousSegments = 0;

    bool exactIdentity = false;  // perm is 0,1,2,... over the full rank
    bool reshapeOnly = false;    // element order is unchanged; a flat copy works

    bool ok() const { return status == GeometryStatus::kOk; }
};

namespace detail {

inline void RowMajorStrides(const int64_t* extents, int n, int64_t* out)
{
    int64_t stride = 1;
    for (int axis = n - 1; axis >= 0; --axis) {
        out[axis] = stride;
        stride *= extents[axis];
    }
}

}  // namespace detail

// Builds the geometry. `rawPerm` may contain PyTorch-style negative axes.
// Returns false and leaves `status` describing the reason when the request is
// not a legal permutation. Zero extents are accepted so that empty-tensor
// paths can be exercised; they yield numel 0.
//
// `permRank` is passed separately and checked against `rank` on purpose. In a
// real host tiling function the shape and the `dims` attribute arrive from two
// independent sources and may disagree; trusting the shape's rank while
// indexing the attribute reads out of bounds.
inline bool BuildGeometry(
    const int64_t* shape, int rank, const int64_t* rawPerm, int permRank, Geometry& g)
{
    g = Geometry{};

    if (rank < 1 || rank > kMaxRank) {
        g.status = GeometryStatus::kRankOutOfRange;
        return false;
    }
    if (permRank != rank) {
        g.status = GeometryStatus::kPermLengthMismatch;
        return false;
    }
    g.rank = rank;
    for (int i = 0; i < rank; ++i) {
        if (shape[i] < 0) {
            g.status = GeometryStatus::kNegativeExtent;
            return false;
        }
        g.shape[i] = shape[i];
    }

    // Normalize negative axes first, then check bijectivity. This is the
    // PyTorch order, and it is why rank-2 (0, -2) must be rejected: it
    // normalizes to (0, 0).
    bool seen[kMaxRank] = {};
    for (int i = 0; i < rank; ++i) {
        int64_t value = rawPerm[i];
        if (value < -rank || value >= rank) {
            g.status = GeometryStatus::kPermEntryOutOfRange;
            return false;
        }
        int axis = static_cast<int>(value < 0 ? value + rank : value);
        if (seen[axis]) {
            g.status = GeometryStatus::kPermNotBijective;
            return false;
        }
        seen[axis] = true;
        g.perm[i] = static_cast<int32_t>(axis);
    }

    for (int i = 0; i < rank; ++i) {
        g.outputShape[i] = g.shape[g.perm[i]];
    }
    detail::RowMajorStrides(g.shape, rank, g.inputStride);
    detail::RowMajorStrides(g.outputShape, rank, g.outputStride);

    g.numel = 1;
    for (int i = 0; i < rank; ++i) {
        g.numel *= g.shape[i];
    }

    // Active axes: everything that is not size 1.
    int activePosition[kMaxRank];
    for (int i = 0; i < kMaxRank; ++i) {
        activePosition[i] = -1;
    }
    for (int axis = 0; axis < rank; ++axis) {
        if (g.shape[axis] != 1) {
            activePosition[axis] = g.activeRank;
            g.activeInput[g.activeRank] = static_cast<int32_t>(axis);
            ++g.activeRank;
        }
    }
    int activeOutRank = 0;
    for (int i = 0; i < rank; ++i) {
        const int axis = g.perm[i];
        if (g.shape[axis] != 1) {
            g.activeOutput[activeOutRank++] = static_cast<int32_t>(axis);
        }
    }

    // Cut the output order into maximal runs that are also adjacent and
    // ascending in the input order. Each run becomes one collapsed axis.
    int runFirst[kMaxRank];
    int runLast[kMaxRank];
    int64_t runExtent[kMaxRank];
    int runCount = 0;
    for (int i = 0; i < activeOutRank; ++i) {
        const int axis = g.activeOutput[i];
        const bool extendsRun =
            runCount > 0 && activePosition[axis] == activePosition[runLast[runCount - 1]] + 1;
        if (extendsRun) {
            runLast[runCount - 1] = axis;
            runExtent[runCount - 1] *= g.shape[axis];
        } else {
            runFirst[runCount] = axis;
            runLast[runCount] = axis;
            runExtent[runCount] = g.shape[axis];
            ++runCount;
        }
    }

    // Rank the runs by where they start in the input order; that ranking is
    // the collapsed input axis numbering.
    int order[kMaxRank];
    for (int i = 0; i < runCount; ++i) {
        order[i] = i;
    }
    for (int i = 1; i < runCount; ++i) {  // insertion sort; runCount <= kMaxRank
        const int key = order[i];
        int j = i - 1;
        while (j >= 0 && activePosition[runFirst[order[j]]] > activePosition[runFirst[key]]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    int inputGroupOfOutputGroup[kMaxRank];
    for (int inputGroup = 0; inputGroup < runCount; ++inputGroup) {
        inputGroupOfOutputGroup[order[inputGroup]] = inputGroup;
        g.collapsedShape[inputGroup] = runExtent[order[inputGroup]];
    }
    for (int outputGroup = 0; outputGroup < runCount; ++outputGroup) {
        g.collapsedPerm[outputGroup] = static_cast<int32_t>(inputGroupOfOutputGroup[outputGroup]);
    }
    g.collapsedRank = runCount;

    // The contiguous suffix is the trailing stretch where the active input and
    // active output orders agree. Belonging to one ascending run is NOT enough:
    // for (A,B,C) -> (C,A,B) the run (A,B) is ascending, yet consecutive B
    // values are still C elements apart in the source.
    int suffix = 0;
    while (suffix < g.activeRank && suffix < activeOutRank &&
           g.activeInput[g.activeRank - 1 - suffix] == g.activeOutput[activeOutRank - 1 - suffix]) {
        ++suffix;
    }
    if (g.numel == 0) {
        g.innermostContiguous = 0;
        g.contiguousSegments = 0;
    } else {
        int64_t inner = 1;
        for (int i = 0; i < suffix; ++i) {
            inner *= g.shape[g.activeInput[g.activeRank - 1 - i]];
        }
        if (inner < 1) {
            inner = 1;  // no preserved suffix: one element is the only safe unit
        }
        g.innermostContiguous = inner;
        g.contiguousSegments = g.numel / inner;
    }

    g.exactIdentity = true;
    for (int i = 0; i < rank; ++i) {
        if (g.perm[i] != i) {
            g.exactIdentity = false;
            break;
        }
    }
    g.reshapeOnly = (g.activeRank == activeOutRank);
    if (g.reshapeOnly) {
        for (int i = 0; i < g.activeRank; ++i) {
            if (g.activeInput[i] != g.activeOutput[i]) {
                g.reshapeOnly = false;
                break;
            }
        }
    }

    g.status = GeometryStatus::kOk;
    return true;
}

}  // namespace transpose
}  // namespace s9

#endif  // ASCEND_S9_TRANSPOSE_GEOMETRY_H
