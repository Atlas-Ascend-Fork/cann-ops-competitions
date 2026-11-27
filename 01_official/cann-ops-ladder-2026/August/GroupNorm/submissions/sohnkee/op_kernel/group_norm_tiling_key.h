/*!
 * \file group_norm_tiling_key.h
 * \brief GroupNorm compile-time scheduling leaves.
 */

#ifndef GROUP_NORM_TILING_KEY_H
#define GROUP_NORM_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define GROUPNORM_TPL_SCH_MODE_0 0
#define GROUPNORM_TPL_SCH_MODE_1 1
#define GROUPNORM_TPL_SCH_MODE_2 2
#define GROUPNORM_TPL_SCH_MODE_3 3

ASCENDC_TPL_ARGS_DECL(
    GroupNorm,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        2,
        ASCENDC_TPL_UI_LIST,
        GROUPNORM_TPL_SCH_MODE_0,
        GROUPNORM_TPL_SCH_MODE_1,
        GROUPNORM_TPL_SCH_MODE_2,
        GROUPNORM_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        schMode,
        ASCENDC_TPL_UI_LIST,
        GROUPNORM_TPL_SCH_MODE_0,
        GROUPNORM_TPL_SCH_MODE_1,
        GROUPNORM_TPL_SCH_MODE_2,
        GROUPNORM_TPL_SCH_MODE_3)));

#endif // GROUP_NORM_TILING_KEY_H
