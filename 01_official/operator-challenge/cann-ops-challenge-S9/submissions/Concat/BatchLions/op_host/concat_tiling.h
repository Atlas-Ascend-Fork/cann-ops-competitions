/* Concat 的 TilingData。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * 归一形式（与 ops/concat/concat_kernel.h 一致）：
 *   rows = ∏shape[:dim]、n_i = ∏shape_i[dim:]、nOut = Σn_i
 * 每个输入占输出的列区间 [off_i, off_i + n_i)。
 */
#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
/* torch.cat 的输入个数上限。
 *
 * 赛题的 `test_op.py` 用 `generate_random_split_sizes(total_len, max_step)`
 * 把一个轴随机切成若干片（每片长度可为 0），片数 ≈ `2·total_len/max_step`：
 *     256/64 ⇒ 9 路（样例 case1）   4096/64 ⇒ 123 路   4096/16 ⇒ 511 路
 * 原来取 64「留足余量」—— **余量根本不够**，提交时 5 个 case 全挂。
 *
 * **这个常量必须与 `s9::CONCAT_MAX_IN` 完全一致**：
 * 那边是纯算术侧（能单测）用的，这边定的是 TilingData 里数组的长度。
 * 两者分叉的后果是 host 按大的写、结构按小的存，
 * kernel 读第 N 项以后全是垃圾 —— 而且**编译不报错**。
 * 所以在 op_host/concat.cpp 里加了 static_assert 把它们钉死。
 *
 * 数组因此占 2048×4 = 8 KB。**不影响每次 launch 的开销**：
 * kernel 用 `S9TilingRd` 按需读 GM（`td.At(offset)`），不走 GET_TILING_DATA 的整体拷贝。 */
/* **实测：aclnn 对动态张量列表的硬上限是 256**（910B4 上喂 511 路，
 * 框架报 EZ1001 "exceeds limit: 256"）。超过 256 路根本到不了本算子，
 * 所以这里也收到 256，nIn 数组从 8 KB 缩到 1 KB。
 * 详见 s9_tiling_calc.h 里 CONCAT_MAX_IN 的说明。 */
constexpr uint32_t CONCAT_MAX_IN = 256;

BEGIN_TILING_DATA_DEF(ConcatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);       // ∏shape[:dim]
  TILING_DATA_FIELD_DEF(uint32_t, nOut);       // Σn_i
  TILING_DATA_FIELD_DEF(uint32_t, numIn);      // 实际输入个数
  TILING_DATA_FIELD_DEF(uint32_t, strategy);   // 0=按行切 1=按列切
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);   // 参与的核数
  TILING_DATA_FIELD_DEF(uint32_t, ubBudget);   // 单块 UB 预算（字节）
  TILING_DATA_FIELD_DEF(uint32_t, l2Mode);     // 输入侧 L2 模式：0=DISABLE 1=NORMAL 4=PERSISTENT
  TILING_DATA_FIELD_DEF(uint32_t, l2ModeOut);  // 输出侧 L2 模式，同上取值
  TILING_DATA_FIELD_DEF(uint32_t, pipeMode);   // 0=按策略自动 1=强制前瞻 2=强制不前瞻
  // 向量拼装路的暂存缓冲块数。msprof 上 mte2 只有 0.664 —— 搬运流水有三分之一
  // 时间在等暂存区腾出来，加深流水可能比继续压读放大更值钱。S9_GBUF 可调。
  TILING_DATA_FIELD_DEF(uint32_t, gBuf);
  // 向量拼装路的**列带数**：blockDim 个核排成 bands × (blockDim/bands) 的网格，
  // 每核只管 nOut/bands 列。1 ＝ 不分带。见 s9_tiling_calc.h 的实测依据。S9_BANDS 可调。
  TILING_DATA_FIELD_DEF(uint32_t, bands);
  // 最宽的一路 —— kernel 用它给暂存区留边界余量
  TILING_DATA_FIELD_DEF(uint32_t, maxIn);
  // 拼装区的缓冲块数。只有 1 块时 Gather(k+1) 要等 tile k 写完 ——
  // 向量与 MTE3 完全串行。S9_ABUF 可调（上限 BUF_N=3）。
  TILING_DATA_FIELD_DEF(uint32_t, aBuf);
  // 按列切时每核每路至少搬多少元素（＝ COL_BURST_BYTES / esz）。
  // **host 和 kernel 必须用同一个值**，所以由 host 算好传下来；
  // 以前 kernel 侧是写死的移位常量，S9_COLEL 改不动它。
  TILING_DATA_FIELD_DEF(uint32_t, colBurstEl);
  // 列带宽度的取整粒度（**元素**）。和 colBurstEl 同一个道理由 host 下发：
  // 以前 kernel 侧写死 S9_BAND_GRAIN=512，两边各存一份，旋钮只改得动 host 那份。
  TILING_DATA_FIELD_DEF(uint32_t, bandGrain);
  // 任一条带里最多几路非空输入。kernel 按它开暂存余量 —— host 精确数出来的，
  // 估计值会低估（各路宽度随机），低估就是真机 UB 越界。
  TILING_DATA_FIELD_DEF(uint32_t, maxInBand);
  // 拼装路每个 tile 的行数上限（0 ＝ 不限，按 UB 顶满）。tileG 有最优点，见 op_kernel。
  TILING_DATA_FIELD_DEF(uint32_t, tileGMax);
  // AIC 分到的搬运比例（百分数，0 = 不启用混合核）。见 op_kernel 的 RunConcatMix。
  TILING_DATA_FIELD_DEF(uint32_t, aicPct);
  // 序幕取 TilingData 的方式：0 = 逐字段 GM 标量读（旧行为）、1 = 一条 DMA 整块进 UB。
  // 只为做配对 A/B 而存在，见 op_kernel 的 S9TilingUb。S9_PRE 覆盖。
  TILING_DATA_FIELD_DEF(uint32_t, preMode);
  // 列带边界：1 = 均衡切分（默认）、0 = 统一带宽＋末带吃剩下（旧行为，留作 A/B）。
  // 见 s9_tiling_calc.h 的 BandRange。S9_BAL 覆盖。
  TILING_DATA_FIELD_DEF(uint32_t, bandBal);
  // 1 = 照旧每路每 tile 调一次 SetL2CacheHint；0 = l2Mode==NORMAL 时跳过（源码证明是空操作）。
  // 只为配对 A/B 而存在，见 op_kernel 里的说明。S9_HINT 覆盖。
  TILING_DATA_FIELD_DEF(uint32_t, hintMode);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_IN, nIn);    // 每个输入的列数
  /* ── 每带执行计划（strategy 3 专用；`S9_BANDPLAN=1`）──
   *
   * 每条带相交的输入下标区间 `[bandLo[b], bandHi[b])` 与区间起点的输出列前缀
   * `bandOff[b]`。**放在 nIn[] 之后**：标量头与 nIn[] 的偏移一个字节都不动，
   * 序幕那条「头 ＋ nIn[]」的 DMA（前 S9_PRE_BYTES 字节）也就不受影响。
   *
   * kernel 按它跳过「每核扫一遍全部 numIn 路」—— 246 路那档 msprof 量到
   * `aiv_scalar_ratio 0.97`，主因就是那场扫描。计划由 host 用
   * `s9::CalcConcatBandPlan` 算出来，与 kernel 的带边界同源。
   *
   * `planOn = 0` 时这三个数组不读（旧路径一个字节没变）。 */
  TILING_DATA_FIELD_DEF(uint32_t, planOn);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, bandLo);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, bandHi);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, bandOff);
  /* ── 逐核布局表：**布局下沉**（`S9_LAYOUT=1`）──
   *
   * 一个核一行、每行 `s9::S9L_CORE_FIELDS`(14) 个 uint32：本核的行区间、列带区间、
   * 以及 `PlanConcatStage` 算出来的整份 UB 布局（tileG / 各缓冲跨度 / 偏移表大小…）。
   * 字段顺序与 `s9::ConcatCoreLayoutRow` 一一对应（host 按它平铺）。
   *
   * **放在整个结构的最后**：前面所有字段的偏移（尤其是序幕那条
   * `S9_PRE_BYTES` 的 DMA 覆盖的"标量头 ＋ nIn[]"）一个字节都不动。
   *
   * `layOn = 0` 时这几个数不读，kernel 照旧自己算 —— 两边的布局由
   * `test_concat_layout.cpp` 的 `CheckCoreLayoutTable` 钉成逐字段一致。 */
  TILING_DATA_FIELD_DEF(uint32_t, layOn);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 48 * 14, layRows);
  /* 边界路裁剪（`S9_TRIM`）：0=关、1=只裁 kernel 的读（UB 账不变 ⇒ tileG 不变）、
   * 2=读与 UB 账一起裁。**1 与 2 的 kernel 行为完全一样**，差别只在 host 怎么算账 ——
   * 这样"少读字节"与"UB 变大 ⇒ tileG 变大"能分两次 A/B 量。 */
  TILING_DATA_FIELD_DEF(uint32_t, trimOn);
  /* 直写路径（`S9_DIRECT`）：0=向量拼装（Gather），1=不拼装、每路直接写出。
   * **故意不参与 tiling 运算** —— 两条路拿到逐字节相同的分核/分带/tileG，
   * 于是配对 A/B 量的正好是"拼装 vs 直写"这一个自变量。
   * 见 op_kernel 的 `DirectBanded` 注释（为什么旧的 `S9_GATHER=0` 不算量过它）。 */
  TILING_DATA_FIELD_DEF(uint32_t, directOn);
  /* Gather 偏移表：0 = 摊开 tileG 行（原样）；1 = **只用 1 行、逐行就地推进**。
   * 表的字节数是 `tblRow·4·tileG` —— 每个输出元素 4 字节 int32，而数据本身
   * f16 才 2 字节，实测它和暂存区一样大（bw=1638、tileG=9 时各约 59 KB）。
   * 缩成 1 行之后表与 tileG **无关**，省下的 UB 可以全给 tileG。
   * **这一版只改 kernel、不动 UB 账** ⇒ 量到的是「per-row Gather 贵不贵」，
   * 不是收益。中性才说明后面改 `PlanConcatStage` 是安全的。 */
  TILING_DATA_FIELD_DEF(uint32_t, tblKOn);
  /* 序幕重叠（`S9_PRERD`）：把**第一趟 MTE2** 提到建偏移表之前发。
   * 读不依赖表，而建表期间 MTE2 完全空着 —— profiling 量到
   * `aiv_scalar_time = 78.8 µs` 与 `Task Duration − aiv_time = 74.5 µs`
   * 几乎相等，这段就是那 21% 空转最后一个没被否掉的嫌疑。 */
  TILING_DATA_FIELD_DEF(uint32_t, preRdOn);
  /* 二维分核的列组数（仅 strategy 0，`S9_GRID`）。1 ＝ 旧行为（纯按行切）。
   * 按行切的 blockDim = rows，而 dim 在中间轴时 rows 常常只有十几 —— 48 个核
   * 只用得上十几个。这个字段让每个行组再切 colParts 份列，**搬运形态不变**。 */
  TILING_DATA_FIELD_DEF(uint32_t, colParts);
  /* 标量瘦身（`S9_HOIST`）：`CopyIn` 里那对 `SetGlobalBuffer`/`SetL2CacheHint`
   * 是**可证明冗余**的 —— `Emit` 只有 `One()` 一个调用点，而 `One()` 在发 Job
   * 之前已经为本路设过 `srcGm_`。留旋钮是为了配对 A/B，不是因为不确定它等价。
   * 依据：按行切那条路实测 `aiv_scalar_ratio = 0.77`、`mte2 = 0.87` ——
   * 标量是紧挨着的次瓶颈，一旦翻倍（`S9_GRID` 那次）就立刻变成主瓶颈 0.99。 */
  TILING_DATA_FIELD_DEF(uint32_t, hoistOn);
  /* 单带时一个 tile 只写一块（`S9_ONEWRITE`）：bands=1 且拼装区一行恰好是 32B 的整数倍时，
   * 拼装区里的行首尾相接、输出行在 GM 里也首尾相接（dstStride=0），原来仍按「一行一块」写。
   * 910B1 实测每写一块约 40 ns、与字节数无关（见 docs/submission-env.md）。**只改写法，地址一个不变。**
   * 放在尾部：前面所有字段的偏移不动。 */
  TILING_DATA_FIELD_DEF(uint32_t, oneWriteOn);
  // 单带密排开关，追加尾字段，不移动既有序幕和数组偏移。
  TILING_DATA_FIELD_DEF(uint32_t, denseOn);
  // **不传 offOut**：它就是 nIn 的前缀和，kernel 边循环边累加即可。
  // 省下来的一半空间用来把路数上限从 1024 提到 2048，
  // TilingData 仍是 8 KB（扫 756 个可能形状，最多见到 4124 路，
  // 但那要 total=16384 且 max_step=8；样例用的是 max_step=64，
  // 同样 total 只有 ~500 路）。
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}
#endif  // CONCAT_TILING_H
