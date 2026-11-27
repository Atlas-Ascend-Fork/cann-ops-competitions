/*!
 * \file inplace_update_tiling_data.h
 * \brief Tiling data for InplaceUpdate.
 */

#ifndef _INPLACEUPDATE_TILING_DATA_H_
#define _INPLACEUPDATE_TILING_DATA_H_

#include <cstdint>

struct InplaceUpdateTilingData {
    int64_t totalNum = 0;       // Number of x/y elements.
    int64_t blockFactor = 1;    // Elements owned by each core.
    int64_t ubFactor = 0;       // Elements copied by one data tile.
    int64_t rowNum = 0;         // x.shape[0].
    int64_t innerNum = 0;       // Elements in one row.
    int64_t indexNum = 0;       // Number of update indices.
    int64_t indexUbCount = 8;   // int32 indices staged per tile.
    int64_t typeSize = 4;       // Element byte size of x/v/y.
};

#endif
