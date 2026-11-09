/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file transpose_tiling.h
 * \brief Shared Transpose tiling-data declaration for host compilation.
 */
#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

#include <cstdint>

struct TransposeTilingData {
    int64_t fastPath;
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
    int64_t dims;
    int64_t totalnumber;
    int64_t shape[8];
    uint64_t vaConfigPadding;
    uint64_t dstVaConfig[32];
    uint64_t srcVaConfig[32];
    int64_t perm[8];
};

#endif // TRANSPOSE_TILING_H
