// CANN 8.5.0 DTYPE_MAP defines half and float for FP16 and FP32.
// Compile each specialized implementation only for the dtype that uses it.
#define INDEXADD_KIND_half 1
#define INDEXADD_KIND___fp16 1
#define INDEXADD_KIND_float 2
#define INDEXADD_CAT_INNER(a,b) a##b
#define INDEXADD_CAT(a,b) INDEXADD_CAT_INNER(a,b)
#if INDEXADD_CAT(INDEXADD_KIND_,DTYPE_SELF) == 1
#include "index_add_half_impl.h"
#elif INDEXADD_CAT(INDEXADD_KIND_,DTYPE_SELF) == 2
#include "index_add_float_impl.h"
#else
#include "index_add_common_impl.h"
#endif
