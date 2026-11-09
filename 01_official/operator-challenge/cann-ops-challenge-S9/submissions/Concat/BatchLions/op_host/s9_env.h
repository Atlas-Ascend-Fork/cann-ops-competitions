/* Host 侧环境开关的**统一解析**（只给 op_host 用；kernel 里没有 getenv）。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * ## 为什么要有这个文件
 *
 * `TilingFunc` 里原先手抄了三十多段同样的三行模板：
 *
 *     const char* x = getenv("S9_FOO");
 *     if (x != nullptr) { int v = atoi(x); if (v >= lo && v <= hi) { var = (uint32_t)v; } }
 *
 * 语义都是同一条：**没设 / 非数字 / 越界 ⇒ 保持默认**，只有落进 `[lo, hi]`
 * 才覆盖。抄三十遍的代价不是"慢"，是**改一处忘一处** —— 仓库里
 * "`S9_GBUF` 允许 1..8 而 kernel 的事件号只到 3"那个真机同步错位就是这么来的。
 * 收成一个来源之后，这张"开关表"还能被**脱机单测**（`test/test_s9_env.cpp`）。
 *
 * ## 有两类刻意**不**走这里
 *
 * 1. 纯 `atoi` 不过滤的（`S9_L2` / `S9_L2O` / `S9_COLBASIS` / `S9_STRAT`）——
 *    它们的**越界值本身有语义**：例如 `l2Mode` 只要不等于 0/1/4，kernel 就走
 *    DISABLE 分支（`else { DISABLE }`）。用区间把它过滤掉反而是改行为。
 * 2. 带附加条件的（`S9_BANDS` 必须整除核数、`S9_BLK` / `S9_MAXINB` / `S9_AIC`
 *    有多分支或还要联动别的字段）—— 这些保留各自的写法，就地注释说明。
 */
#ifndef S9_ENV_H
#define S9_ENV_H

#include <climits>
#include <cstdlib>

namespace s9 {

/* 环境变量 → uint32。没设 / 非数字 / 不在 [lo, hi] ⇒ 返回 def。
 * `hi` 传 `INT_MAX` 表示"只设下界"。
 * 注意 `atoi("abc") == 0`：下界 > 0 的开关遇到非数字会自动落回默认，与原写法一致。 */
inline uint32_t EnvU32(const char *name, uint32_t def, int lo, int hi)
{
    const char *s = std::getenv(name);
    if (s == nullptr) { return def; }
    const int v = std::atoi(s);
    return (v >= lo && v <= hi) ? (uint32_t)v : def;
}

/* 布尔开关：没设 ⇒ def；设了 ⇒ `atoi != 0`。
 * 注意"设了 `S9_X=` 空串"会得到 `atoi("") == 0` ⇒ false，与原写法一致。 */
inline bool EnvFlag(const char *name, bool def)
{
    const char *s = std::getenv(name);
    return (s == nullptr) ? def : (std::atoi(s) != 0);
}

}  // namespace s9

#endif  // S9_ENV_H
