/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file greater_tiling.h
 * \brief Shared Greater tiling-data declaration for host compilation.
 */
#ifndef GREATER_TILING_H
#define GREATER_TILING_H

// Keep one authoritative ABI declaration. The kernel-local header is packaged
// with device sources, while this canonical host header satisfies the contest
// source-layout contract and is consumed by greater_tiling.cpp.
#include "../op_kernel/greater_tiling_data.h"

#endif // GREATER_TILING_H
