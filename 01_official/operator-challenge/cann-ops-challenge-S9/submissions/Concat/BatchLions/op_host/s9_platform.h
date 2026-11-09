/* Concat Host 平台核数查询；与纯算术头分离以支持离线测试。
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef S9_PLATFORM_H
#define S9_PLATFORM_H

/* ## 包含路径要探，不能写死
 *
 * 这个头在不同工程里的相对位置不一样。语料里两种写法都有：
 *     "tiling/platform/platform_ascendc.h"   —— 1391 处（主流，CANN 的
 *                                              include/utils 作根时就是这个）
 *     "platform/platform_ascendc.h"          ——  378 处（仓内另一套根）
 * 我第一版只写了后者，结果 msopgen 生成的自定义算子工程里**根本找不到**，
 * 整个编译在 `fatal error: platform/platform_ascendc.h: No such file` 上挂掉。
 *
 * 现在用 `__has_include` 依次探，两个都没有就**退回常量**而不是编译失败 ——
 * 查平台只是优化（少用核只是慢），**不该成为编不出包的理由**。 */
#if defined(__has_include)
#  if __has_include("tiling/platform/platform_ascendc.h")
#    include "tiling/platform/platform_ascendc.h"
#    define S9_HAVE_PLATFORM 1
#  elif __has_include("platform/platform_ascendc.h")
#    include "platform/platform_ascendc.h"
#    define S9_HAVE_PLATFORM 1
#  elif __has_include(<tiling/platform/platform_ascendc.h>)
#    include <tiling/platform/platform_ascendc.h>
#    define S9_HAVE_PLATFORM 1
#  endif
#endif

#include "s9_tiling_calc.h"

namespace s9 {

/* 查本次运行的 AIV 核数。查不到就回落到 AIV_MAX。
 *
 * 写成模板是为了不依赖具体的 context 类型（TilingContext 等都有 GetPlatformInfo）。
 * 取值顺序按语料里官方算子的通行写法：先 GetCoreNumAiv，为 0 再退 GetCoreNum。 */
template <typename Ctx>
inline uint32_t QueryAivCores(Ctx *context)
{
#ifdef S9_HAVE_PLATFORM
    if (context == nullptr) { return AIV_MAX; }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) { return AIV_MAX; }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t n = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (n == 0) { n = static_cast<uint32_t>(ascendcPlatform.GetCoreNum()); }
    return n == 0 ? AIV_MAX : n;
#else
    // 找不到平台头：退回兜底常量（40，见 s9_tiling_calc.h 的 AIV_MAX）。
    // **不是编译失败** —— 少用核只是慢，编不出包才是致命的。
    (void)context;
    return AIV_MAX;
#endif
}

/* 查本次运行的 AIC（Cube）核数。查不到就按 AIV 的一半估（A2 上恒为 1:2）。
 *
 * 混合核 `KERNEL_TYPE_MIX_AIC_1_2` 下 `SetBlockDim(N)` 启动 **N 个 Cube ＋ 2N 个
 * Vector**（《设置Kernel类型》表 2），所以 blockDim 要按 AIC 数给，不是按 AIV 数。 */
template <typename Ctx>
inline uint32_t QueryAicCores(Ctx *context)
{
#ifdef S9_HAVE_PLATFORM
    if (context == nullptr) { return AIV_MAX / 2; }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) { return AIV_MAX / 2; }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t n = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    return n == 0 ? (AIV_MAX / 2) : n;
#else
    (void)context;
    return AIV_MAX / 2;
#endif
}

}  // namespace s9
#endif  // S9_PLATFORM_H
