#ifndef GREATER_TILING_FIELDS_H
#define GREATER_TILING_FIELDS_H
/*
 * Shared ordered tiling schema. X declares scalar fields and A declares fixed
 * arrays. perCore is the first field; diagCode is the last field.
 */

// Collapse adjacent dimensions with equal broadcast patterns. The shared
// maximum dimension count is GT_MAX_DIM_VALUE.
#ifndef GT_MAX_DIM_VALUE
#define GT_MAX_DIM_VALUE 8
#endif

#define GT_TILING_FIELDS(X, A)                                                                        \
    /* perCore is the first field, also used by the eight-byte aligned-route payload. */ \
    /* The aligned-route payload contains one uint64_t field. */ \
    /* Keep declaration order consistent between host and kernel readers. */ \
    /* Scalar field declaration. */ \
    X(uint64_t, perCore)   /* output elements handled by one core */                                  \
    X(uint64_t, totalNum)  /* total output elements */                                                \
    X(uint64_t, innerLen)  /* extent of the innermost collapsed dim */                                \
    X(uint64_t, rowNum)    /* totalNum / innerLen */                                                  \
    X(uint32_t, ubSize)    /* Usable UB bytes queried from the platform. */         \
    X(uint32_t, usedCores)                                                                            \
    X(uint32_t, dimNum)     /* number of collapsed dims (D) */                                        \
    X(uint32_t, innerMode)  /* 0 = both dense, 1 = x1 scalar per row, 2 = x2 scalar per row */         \
    X(uint32_t, cacheMode)  /* bit0: x1 row block invariant, bit1: x2 row block invariant */          \
    X(uint32_t, pad0)                                                                                 \
    A(uint32_t, GT_MAX_DIM_VALUE, shape) /* collapsed output shape */                                 \
    /* Input strides are interleaved by dimension. */ \
    /* x1 strides occupy entries 2*i. */ \
    /* x2 strides occupy entries 2*i+1. */ \
    /* A zero stride denotes a broadcast dimension. */ \
    /* Use the same array extent on both sides. */ \
    /* Preserve field order in host and kernel declarations. */ \
    A(uint64_t, 2 * GT_MAX_DIM_VALUE, strides)                                                        \
    X(uint64_t, tilingKey) /* This key matches the value passed to SetTilingKey. */ \
                           /* The key identifies the metadata reader and computation route. */               \
    /* Keep diagCode as the final field. */ \
    /* Optional diagnostic payload. */ \
    /* Each serialized route uses its own diagnostic-word offset. */ \
    /* Zero when diagnostics are disabled. */ \
    X(uint64_t, diagCode)

#endif  // GREATER_TILING_FIELDS_H
