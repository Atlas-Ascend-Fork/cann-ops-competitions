/* Concat 的纯 C++ Tiling 算术，与 Host、布局和 CPU 回放共用。
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef S9_TILING_CALC_H
#define S9_TILING_CALC_H

#include <cstdint>
// 拼装路的 UB 逐块布局（含逐块 32B 对齐与序幕/指针暂存的实账）。
// **与 op_kernel 的 Init 同源** —— 那份账以前在 host、kernel、孪生各写一份，
// 三份对不上的后果是 `InitBuffer` 越界（真机 vector core exception 507035）。
#include "s9_concat_layout.h"

namespace s9 {

// 平台查询失败时使用的保守 AIV 核数；正常路径由 Host 传入平台值。
constexpr uint32_t AIV_MAX = 40;
// Concat 的核数：`clamp(输出字节 / MIN_BURST, CONCAT_MIN_CORES, CONCAT_MAX_CORES)`。
//
// ── 实测（910B3 · CANN 8.5.2 · L2 输入输出均 NORMAL · 公平口径 · 已含 tiling 直读
//    与拼装快路），六个 shape 各扫一遍核数，单位 us，**〔〕为该行最优** ──
//
//  代号      shape                 输出量   行数  路数    2     4     6     8    11/12  16   24    32    40
//  small   [128,27..5]x9 fp16     63 KB    128   9    -   8.26  6.80〔5.90〕6.54  7.16 10.04   -     -
//  mid     [512,55]x3    fp16    165 KB    512   3    -    -    9.68  6.70〔5.94〕6.26  6.18  8.22   -
//  longmid [55,512]x3    fp16    165 KB     55   3  5.28  3.92〔3.22〕3.48  3.48  3.90   -     -     -
//  big     [2024,1000]x3 fp16   11.6 MB  2024   3    -     -     -     -     -   32.42 22.42 20.43〔19.02〕
//  shortbig[20240,100]x3 fp16   11.6 MB 20240   3    -     -     -     -     -     -   88.41 73.40〔64.12〕
//  huge    [4096,2000]x3 fp16   47.0 MB  4096   3    -     -     -     -     -     -   79.48 78.46〔77.44〕
//
// 本规则给出：small 8 · mid 11 · longmid 11 · big 40 · shortbig 40 · huge 40。
// **六个里五个正中最优**，只有 longmid 差 8%（规则 11 核 3.48，最优 6 核 3.22），
// 平均代价 1.3%。
//
// ── 为什么是这三个常数：一个能解释它们的机器模型 ──
//
// 把每核并行、取均分近似，单次 launch 的耗时可以写成
//
//     T(p) = t0                          启动与固定开销
//          + a · ⌈rows/p⌉ · blkPerRow    每核的 DataCopyPad 块代价
//          + 2W / min(p·bc, Bmax)        搬运；总带宽**饱和**，不是随核数线性叠加
//          + c · p · numIn               争用：每核对每输入各开一条访存流
//
// 其中 blkPerRow = numIn（读）+ numIn 或 1（写；拼装快路把写合成一条）。
// 用上表 28 个点做最小二乘（对数空间），拟合出来的参数与硬件规格对得上，
// 这是模型没跑偏的旁证：
//     t0 = 1.70 us · a = 11.2 ns/块 · bc = 76.5 GB/s（A2A3.ini 推算 59.2）
//     Bmax = 1.59 TB/s（ini 推算 ddr_rate 32 B/cycle × 24 核 × 1.85 GHz = 1.42）
//     c = 0.031 us/(核·路)
//
// 模型解释了三条经验规律：
// ① **上限**：`c·p·numIn` 与饱和的搬运项合起来，使得核数过了某点净亏 ——
//    48 核在六个 shape 上一次都不是最优。取 40：big/shortbig/huge 上模型与实测都选 40。
// ② **下限 8**：小 shape 由 `a·⌈rows/p⌉·blkPerRow` 主导，要靠多核摊块数。
// ③ **每核 16 KB**：中段是块代价与搬运项的交叉点。
//
// **但模型本身不该拿去替掉这三个常数**：同样在这 28 个点上，
// 模型的平均代价是 **3.2%**，比三常数规则的 1.3% 更差（它在 small/mid 上偏多、
// 在 longmid 上偏多）。试过把读块与写块的代价拆成两个系数，拟合把写的系数压到 0，
// 说明这批数据分辨不出那个差异。**所以模型的价值是解释，不是替代。**
//
// ── 三件会让这组标定失效的事，都踩过 ──
// - **改 L2 CacheMode**：MIN_BURST 曾按 L2=DISABLE 调成 4096，改 NORMAL 后曲线整体左移。
// - **改测量口径**：赛题 custom_op 每轮 concat 前跑一次 201 MB 的 aclnnMul，
//   给 torch.cat 对比时必须插同一个 Mul，否则官方 L2 命中、自研 L2 冷。
// - **改 kernel 的固定开销**：绕开 GET_TILING_DATA 的整体拷贝之后，
//   mid 的最优核数从 16 回到 11、longmid 从 4 移到 6 —— **每核的固定成本直接决定最优核数**。
//
// ── 还有一个别忘的前提 ──
// 同一配置的**run-to-run 抖动有 ±10~22%**（big 同配置两次测到 22.72 与 18.58）。
// 上表已取多次中位数，但这意味着「规则差最优 8%」这种量级**单次测量分辨不出来**，
// 再往下调没有可验证的收益。
// **这个 40 是在 910B3 上标定的，而 910B3 恰好只有 40 个 AIV** —— 当时它等于「用满」。
// 评测机是 910B1、有 48 个 AIV。
//
// ── 第一次放开（40 → 64）被榜单证伪 ──
//     40 上限（2026-09-08 那版）→ 603.32 µs / 第 52
//     64 上限（2026-09-09 07:27 那版）→ 646.36 µs / 第 71
// 当时的机理解释是：每个核都要**建一遍全表**、给 **numIn 条路各发一次 DMA**，
// 这部分固定开销正比于核数，48 核比 40 核多 20%，而它约占三分之一 ⇒ +7%。
// 注意那次提交**同时还带了一处 int8 门槛改动**，所以严格说它不是单变量的。
//
// ── 现在重新放到 48，理由是那条机理已经被拆掉了 ──
// 上面那段注释自己写了复检条件：「真要用满 48 个核，得先把每核的固定开销压下去
//（列带分核就是在做这件事），并且在评测机上用一次提交单独验证」。
// 列带分核已经落地，而且读循环只走本带：**每核现在只碰 [iLo,iHi) 那几路**，
// 表也只有 O(bandCols) 而不是 O(nOut)，固定开销不再正比于 numIn。
// 另一半证据来自 msprof：(4096,16384) fp16 那档 **aiv_mte2_ratio = 0.936**，
// 搬运流水已经按核打满 ⇒ 吞吐直接受核数限制，多 8 个核就是多 8 套 DMA 引擎。
//
// **本机（910B4，40 个 AIV）量不出这条**：min(平台报的 40, 48) 仍是 40，
// 全套本机回归一个数都不会动。所以它只能靠一次单变量提交来判。
// 主机孪生里每个用例按 40 和 48 各跑一遍，覆盖这半边的分带/判据组合。
constexpr uint32_t CONCAT_MAX_CORES = 48;
constexpr uint32_t CONCAT_MIN_CORES = 8;
constexpr uint32_t MIN_BURST = 16384;
constexpr uint32_t TARGET_ITERS = 4;        // 每核跑几轮（实测最优，见 ops/README.md）
constexpr uint32_t UB_MIN = 8 * 1024;
// UB 共 192 KB，kernel 开 BUF_N 块，所以单块上限 ＝ 192/BUF_N 留点余量。
// BUF_N 从 2 提到 3 之后（见 op_kernel/concat.cpp 的说明），这里从 92 KB 降到 60 KB。
// **两处必须一起改**，否则 InitBuffer 会超出 UB。
constexpr uint32_t UB_MAX = 60 * 1024;
/* **输入路数的上限。**
 *
 * 赛题的输入是 `generate_random_split_sizes(total_len, max_step)` 随机切出来的，
 * 路数约等于 `2·total_len / max_step`。样例 case1 是 total_len=256、max_step=64
 * ⇒ 9 路；但 total_len=4096、max_step=64 就是 **123 路**，
 * total_len=4096、max_step=16 是 **511 路**。
 *
 * 原来这个数是 64，而且 host 侧的读取循环写成
 * `for (k = 0; k < CONCAT_MAX_IN; ++k)`，超过 64 路时**只读前 64 路、
 * numIn 停在 64、而且完全不知道还有更多** —— nOut 只算了前 64 路，
 * 输出后半段留着 `at::empty` 的垃圾，校验必挂。**这是提交失败的根因。**
 *
 * **上限取 256 —— 这是 aclnn 框架自己的硬限制，实测出来的。**
 * 在 910B4 上二分：256 路通过、257 路起报
 *     `AclNN_Parameter_Error(EZ1001): Size of tensorList in dynamic input[0] is N,
 *      which exceeds limit`
 * 这条检查在**生成的 aclnn 包装代码里**，发生在我的 tiling 之前，改不到。
 * 所以再往上开没有意义（原来写 1024，那段范围根本到不了）。
 *
 * 现在：上限 256，且 host 用 `GetComputeNodeInputNum()` 取真实路数，
 * 超过上限**显式失败**而不是静默截断。
 * TilingData 里的两个数组因此变成 1024×2×4 = 8 KB —— 不影响每次 launch 的开销，
 * 因为 kernel 用 `S9TilingRd` **按需读 GM**（`td.At(offset)`），不整体拷贝。 *
 * **实测把上限钉死在 256** —— 那是 aclnn 框架自己的限制，不是我定的。
 * 910B4 上喂 511 路时框架直接拒：
 *     AclNN_Parameter_Error(EZ1001): Size of tensorList in dynamic input[0]
 *     is 511, which exceeds limit: 256
 * 也就是说超过 256 路的调用**根本到不了本算子**，支持更多是死代码。
 * 于是失败区间正好是 **65~256**：样例 case1 只有 9 路，单点测试永远碰不到。
 * 收到 256 后 nIn 数组只占 1 KB（2048 时是 8 KB）。 */
constexpr uint32_t CONCAT_MAX_IN = 256;
// 拼装路径的输入路数上限，与 kernel 的 S9_ASM_MAX 一致
// 拼装路径的路数上限。**从 16 提到 256**（＝ aclnn 的动态输入硬上限），
// 因为 Case5 那类形状有 123 路，原来的 16 把最该用拼装的场景全挡在门外。
// kernel 侧 srcs/nIns/offs 三个栈数组随之变成 256×(8+4+4)=4 KB。
constexpr uint32_t ASM_MAX_IN = S9L_ASM_MAX_IN;   // 唯一来源在 s9_concat_layout.h
/* 按行切时每条 DMA 低于这个字节数，就改成按输入切核（strategy 4）。
 * **没有实测依据**，只有方向站得住：低于这个量级时 DMA 的发射开销
 * 压过搬运本身。用 S9_INSPLIT 覆盖以便上板扫（同 MIN_BURST 的做法）。 */
// 按输入切核（strategy 4）的门槛：按行切时每条指令搬的字节数低于它，才认为
// 按行切"退化成了大量小 DMA"、值得改按输入切。
//
// **512 是拼装路还很弱的时候标的，已经过期。** 拼装路装上「摊开的多行偏移表」
// ＋「列带分核」＋「读循环只走本带」之后，同样的形状它比按输入切好得多。
// 910B4 上按门槛扫（合计）：**384 → 293.66** · 256 → 295.14 · 512 → 373.35。
// 差的那 80 µs 全在两档 bytesPerInstr = 416 的形状上，它们正好卡在 384 与 512 之间：
//     (512,2048) int32 248 路   门槛 512 → 80.64（按输入切） · 384 → **42.54**（拼装）
//     (2048,2048) int8 248 路   门槛 512 → 99.84            · 384 → **59.05**
//
// **再往下调到 128 试过，被榜单证伪，别再试。** 910B4 上「只改这一处」量到的是
// 大赢：(128,4096) f16 123路 25.58 → **19.37**、(512,2048) f16 248路 42.62 →
// **30.10**、(256,4096) f16 243路 48.24 → **29.08**，0/32/64/96/128 全等价。
// 可是同一份代码上板（第 14 次提交，单变量）**五档齐退**：
//     5.66→7.37 · 30.87→33.11 · 19.56→20.19 · 103.30→104.69 · 245.64→261.05
//     合计 405.03 → **426.41**
// 两台机的核数都被 CONCAT_MAX_CORES 封在 40，差的是 L2（96 vs 192 MB）与主频
// （1500 vs 1850）—— 评测机上「按输入切核」的整行大块写更划算，本机量不出来。
// ⇒ **这个门槛属于「本机测量不可迁移」的量，以榜单为准，停在 384。**
constexpr uint32_t IN_SPLIT_BYTES = 384;
// 按输入切核（strategy 4）**每路写出去的块宽下限**（字节）。低于它就让给拼装路。
//
// 依据是 msprof 的流量放大，不是推的：(128,4096) fp16 123 路那档走按输入切时
//     写 2928 KB / 需要 1024 KB ＝ **2.86×**，读 1980 / 1024 ＝ 1.93×
// 而每路只有 nOut/numIn·esz ＝ 66 字节 —— 每块都要碰 4 个 32B 块。
// 同一形状交给拼装路（写块 ＝ 带宽·esz ＝ 1024 字节）：**26.51 → 18.84 µs（−29%）**。
//
// **这条和 IN_SPLIT_BYTES 是两个不同的判据，别合并**：
// 前者问「按行切是不是退化成小 DMA 了」（要不要离开按行切），
// 这条问「按输入切自己的写块够不够宽」（离开之后该去 4 还是 3）。
// 榜单证伪过"把 IN_SPLIT_BYTES 整体调低"（405.03 → 426.41），
// 那是**全面**减少按输入切；这条只挪走写块 < 128 B 的那一小撮，且要拼装路接得住。
constexpr uint32_t IN_SPLIT_MIN_WRITE = 128;
// 拼装区开双缓冲（aBuf=2）的**路数上限**。
//
// aBuf=2 让 Gather(k+1) 与 tile k 的写出重叠 —— 向量受限的档赚，读受限的档亏
//（它多占一份 outPitch/行，把 tile 压小）。910B4 + CANN 8.5.0 上逐档量：
//
//   档                    路数   aBuf=1    aBuf=2
//   c  1024×1024  f32      27    15.56   **14.64**   −6%
//   d  2048×2048  f16      58    31.41   **29.34**   −6.6%
//   g  512×2048   i32      58    20.74   **19.61**   −5.4%
//   e  2048×8192  f16     123    73.69   **64.98**   −11.8%
//   p4 4096×4096  f16     123    74.43   **65.60**   −11.9%
//   I5 2048×4096  i8      123    47.29   **44.05**   −6.8%
//   f  128×4096   f16     123    18.95     18.81      0
//   B2 8192×8192  f16    ~128   481.56    481.47      0
//   G16 4096×16384 f16    246   502.92     534.45   **+6.3%**
//   I2 2048×2048  i8      248    45.86      47.38   **+3.3%**
//
// **判据是路数，不是数据量** —— B2 与 G16 同为 256 MB，B2（128 路）中性、
// G16（246 路）亏 6.3%；128 与 246 之间是干净的分界，取 160。
// 机理：路数越多，`slack = numIn·32` 这块固定 UB 越大（246 路是 7.9 KB，
// 乘 gBuf 再乘 int8 的副本），tile 本来就被压得紧；而且每 tile 的读指令条数
// 正比于路数 ⇒ 读先成为瓶颈，这时把 UB 让给 aBuf 是纯亏。
constexpr uint32_t ABUF2_MAX_IN = 160;
// ── 列带分核（strategy 3）的两个标定量 ──
// 一条带写出去是 blockCount=cnt、blockLen=带宽字节 的跨步块。带太窄写就退化成小块，
// 所以给带宽一个下限。512 字节的依据只是「DMA 块小于这个量级时发射开销压过搬运」，
// 与 IN_SPLIT_BYTES 同源，**同样没有独立实测**，用 S9_BANDMIN 扫。
constexpr uint32_t BAND_MIN_BYTES = 512;
// **带宽必须是 512 字节的整数倍。** 这是扫出来的，不是推的：
// (4096,4096) fp16 123 路，只改带数 ——
//     1 带 171.13 · 2 带 144.20 · 4 带 133.23 · 5 带 139.62 ·
//     **8 带 128.05** · 10 带 153.05
// 对得上的只有一条：带宽字节数是不是 512 的整数倍。
//   8 带 → 512 列 ＝ 1024 B ✓   4 带 → 1024 列 ＝ 2048 B ✓
//   5 带 → 1632 B ✗   10 带 → 832 B ✗   2 带 → 4064 B ✗
// 每核写出的是 blockLen=带宽、按 nOut·esz 跨步的块，起点 cb0·esz 也是带宽的整数倍；
// 带宽不是 512 的倍数时每一块都横跨缓存行边界。fp32 那档同样对得上
//（c 档 4 带 21.64 最快，而 5/10 带 26.60/29.90）。
// 列带宽度的取整粒度。**取粗不是浪费，是把带首钉在 512 B 上** —— 试过调细，全线变差。
//
// 起初以为 512 太粗、自己制造了负载不均：(4096,16384) fp16 · 10 带时
// bandCols = AlignUp(1639,256) = 1792，前 9 条各 1792、末带只剩 256，
// 最忙的核比平均多干 9.4%。改细粒度确实把不均压到 0.6%，但 910B4 上**更慢**：
//     自动选带   512 → G16 468.39 · p4 116.88 · e 115.46
//                128 → G16 515.32 · p4 141.55 · e 128.18
//     带数钉 10  512 → G16 468.07 ｜ 256 → 496.65 ｜ 128 → 499.23 ｜ 32 → **530.44**
// 带数钉死之后仍然单调变差 ⇒ **不是带数的事，是带首的对齐**：
// 粒度 512 B 时 cb0·esz 必是 512 的倍数，一条带写出去的
// blockLen=bw·esz、dstStride=(nOut−bw)·esz 两头都压在整线上；
// 粒度 32 B 只保证 32 对齐，每个写块两端各多压进半条线。
// 均衡那点收益（9.4%）远不抵这个代价。
//
// 顺带：均分本身也救不了这一档 —— 最忙的带仍是 7 个 512B 粒度（1792），
// 均分只是让末带那几个核不那么闲，**关键路径一模一样**。
// 列带宽度的取整粒度。**取粗不是浪费，是把带首钉在 512 B 上。**
//
// 910B4 上调细全线变差（自动选带 512 → G16 468.39 · 128 → 515.32；
// 带数钉 10 时 512 → 468.07 · 32 → 530.44，单调）。机理是带首的对齐：
// 粒度 512 B 时 cb0·esz 必是 512 的倍数，一条带写出去两头都压在整线上。
//
// **上榜验过一次 512 → 16（第 18 次提交），退步但很有信息量**：
// 同一个改动本机让 G16 退 11%，榜上 Case5 只退 2.0%（248.18 → 253.14）。
// 它把带数从 16 提到 48、每条暂存读从 1.8 KB 提到 6.5 KB ——
// 也就是说**读块变大在 B1 上确实赚回了约 9%**，只是被失去对齐的代价抵消。
// ⇒ 要的是「读块大 **且** 带首仍 512B 对齐」，而不是单纯调细粒度。
constexpr uint32_t BAND_GRAIN_BYTES = 512;
// 带数只在 blockDim 的因数里挑 —— 网格要填满，否则有核完全空转。
constexpr uint32_t BAND_MAX       = 64;
// Gather 拼装的 nOut 上限。
//
// **这个上限的理由已经过期了。** 它原本是因为偏移表要按 nOut 建一遍、
// 是 O(nOut) 的一次性开销，行短时才划算（1024 → 8192 两次都是按量级推的）。
// 加了**列带分核**之后表只覆盖本核那一带，是 O(bandCols) 而不是 O(nOut)，
// 而 nOut 越大、按行切越烂（每行每路一条小 DMA）。910B4 实测（左＝上限 8192
// 走按行切，右＝放开走拼装路）：
//     4096×16384 fp16 134 MB  1065.03 → **588.28**（1.81×）
//     2048×32768 fp16         1001.00 → **589.42**（1.70×）
//     2048×16384 fp32          996.90 → **504.98**（1.97×）
//      8192×16384 fp16 58 路  1557.37 → **880.26**（1.77×）
//     16384×16384 fp16 512 MB 4196.76 → **1843.73**（2.28×）
// 少路数的大形状（12 路）持平，没有一档变差，全部逐位相同。
//
// 真正的约束不是 nOut 而是 **UB 放不放得下一行**（fixed + perTile ≤ UB_TOTAL_HOST），
// 那个判据本来就在，让它自己去卡。这里留一个大到不起作用的值。
// S9_GMAXOUT 可以调回去做 A/B。
constexpr uint32_t GATHER_MAX_OUT = 1u << 20;
// **`UB_TOTAL_HOST` 已删。** 它自称"留给孪生做对照"，而孪生一次都没引用过 ——
// 一个没人用的第二份 UB 口径，正是"两边算不到一起"的现成来源。
// 唯一口径在 `s9_concat_layout.h`（那边还要再扣序幕的 1152 B，并把指针暂存
// 逐项算进实账）。
// **下面这段讲的是 `COL_BURST_BYTES`。** 原先这里还压着一句
// 「列切每个核至少分到多少**元素**（不是字节）」—— 它和三行之后的
// 「**单位是字节，不是元素**」**直接矛盾**，是 COL_BURST_ELEMS 时代的残留。
// 两句相反的注释比没有注释更糟，删掉旧的那句。
// 910B4 上扫 (2024,3000)/dim=0 两种 dtype：最优核数**都是 13**，
// 而 13 核对应 fp16 每核 16.4 KB、fp32 每核 32.8 KB —— 字节数差一倍、
// 元素数一样（≈8192）。所以决定核数的是元素数，按字节反推会让 fp32 多开一倍核。
// 只测过一组形状 × 两种 dtype，外推请当心。
// 按列切时每个核至少分到多少 —— **单位是字节，不是元素**。
//
// 原先是 `COL_BURST_ELEMS = 8192` 个元素。那个数是在 fp32 上标的，
// 换算成载荷是 32 KB；而同一个常数用在 fp16 上只有 16 KB，**载荷小了一半**，
// 于是 fp16 的 dim=0 档一律切得太碎。910B4 上扫核数（dim=0，五档）：
//   档                    默认核数/耗时      最优核数/耗时     每核每路载荷
//   (2024,3000) f16 57路      13 / 47.96      **10 / 40.53**      21.3 KB
//   (4048,3000) f16 123路     12 / 106.34     ** 8 / 82.48**      24.6 KB
//   (1024,4096) f16 27路      18 / ~36        **12 / 27.04**      25.9 KB
//   (8192,4096) f16 247路     16 / ~220       **13 / 200.08**     20.9 KB
//   (2024,3000) f32 57路      13 / 69.81      **20 / 68.78**      21.3 KB
// **五档的最优点全部落在 21~26 KB**，与 dtype、路数、总量都无关 ——
// 它就是"每核每路一条 DMA 该搬多少字节"。fp32 那档一直没暴露问题，
// 正是因为 8192 个 fp32 元素恰好是 32 KB，离最优不远。
// 取 24576（24 KB）：五档里有三档正好落在实测最优点上。
//
// **换 dtype、换长宽比又验了四档**（同样扫核数，左＝旧规则给的核数/耗时，
// 右＝新规则给的核数/耗时）：
//   (4096,2048) i32 123路   8 / 106.26  → **11 / 95.76**   −9.9%（最优点就是 11）
//   (2048,4096) i8   58路  17 / ~55     → ** 5 / ~35.3**   −36%
//   (512,8192)  f32  27路  18 / ~48.6   → **25 / ~49**     持平
//   (8192,2048) f16 247路   8 / **114.01** → 5 / ~127      **+11%（这一档变差）**
//
// **已知的例外**：最后那档的最优载荷只有 16.6 KB，比其余档小一截。
// 它和 (8192,4096) f16 247路 一样是"路数很多、每路又不大"的形状 ——
// 这类形状偏好更小的载荷（更多核）。24 KB 在它上面亏 11%，
// 但九档合计仍然是 −11%（480 → 425）与 −5%（324 → 307）。
// 想连这一档一起吃掉，得让载荷随路数变，那是另一条需要单独标定的规则，
// **别凭"看起来该这样"就加**。
// **2026-09-15 在 910B1（48 核）上按规格抽样试过 16384 —— 合并回归否掉了，退回 24576。**
// 40 个 rows < 核数 的形状（`op/notebook/batch.sh colel`，含已经开满核的 strategy 1）× 4 档：
//     8192  几何平均 1.121×，赢 19 输 9（最差 0.59：已开满核的档被过切）
//     12288 几何平均 1.087×，赢 16 输 4
//     16384 几何平均 1.115×，**赢 20 输 1**（最差 0.83，7 µs 的 1.9 MB 小张量）
// 上面那组 24 KB 的依据是 910B3 上 9 个手挑二维形状，这一组覆盖中间轴与四种 dtype。
// ⚠️ **随后的合并回归（78 个规格形状，`regr.sh`）把 16384 否掉了**：旧选路是 strategy 1 的 8 个里
// **5 个变慢 8%~29%**（`2,4933,404 int32` 23→34 核 25.36→35.83 µs、`2,754,2954 int32` 30→45 核 29.96→39.52），
// 规律一致 —— 原来只开几个核的赢、原来已经开到 20~30 核的被过切。上面那组 40 形状里这类太少，没暴露。
// ⇒ 单一常量调不对，要分档（只给核开得少的加核）；在那条规则单独验证之前**保持 24576**。
constexpr uint32_t COL_BURST_BYTES = 24576;
// 列切核数的基准：0 = 最短的一路（旧行为），1 = 平均一路（默认）。S9_COLBASIS 可覆盖。
/* ═══ 输出线性切分（strategy 5）的启用门槛 ═══
 *
 * 低于这个输出字节数就不换 —— 多核的启动与尾巴代价在小形状上抵不过并行收益，
 * 而 `rows == 1` 的小形状本来也不慢。**这个数是选的，不是量的**：
 * 已量到的最小一档是 1 MB（f32、128 路，2.59×），64 KB 到 1 MB 之间没有数据点。
 * 要收紧或放宽请先扫，别凭"看起来该这样"改。 */
constexpr uint32_t LINEAR_MIN_BYTES = 64u * 1024u;
/* 「掉出拼装路时放开 blockDim」这条救援的行数上限。
 * **8 是量出来的，不是选的**：910B4 上 rows=2/4/8 分别 5.63×/4.04×/5.13×，
 * 而 rows ≥ 9 的那一段本机根本触发不到（40 核时它们已经 gatherOk）。
 * 评测机 48 核上唯一一次落进那段的观测是 **C4 退化 1.8%**。⇒ 先只留量过的。 */
constexpr uint32_t LINEAR_RESCUE_MAX_ROWS = 8;
constexpr uint32_t COL_BASIS_MIN = 0;
constexpr uint32_t COL_BASIS_AVG = 1;
// **`GATHER_ITERS` 与 `ConcatKnobs::gatherIters` 已删 —— 那是个死旋钮。**
// 它从 `S9_GITERS` 解析、进 ConcatKnobs、传进 CalcConcatLens，然后被
// `(void)` 掉，**一次都没参与计算**。用户设了它什么都不会发生、也不报错。
// 它描述的"向量拼装的目标轮数"早已被 kernel 侧按整块 UB 反推 tileG 取代
//（见 op_kernel 的 Init / PlanConcatStage）。
// 那一行尾部还挂着第二条注释（"向量拼装的 nOut 上限"），是删 GATHER_MAX_OUT 时
// 漏掉的残骸，一并清掉。

inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b : 0; }
inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
inline uint32_t Clamp(uint32_t v, uint32_t lo, uint32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ═════════════ 带检查的维度累乘 ═════════════
 *
 * 这些 tiling 函数原来一律写 `t.rows *= (uint32_t)shape[i]`。
 * 两处都会**静默出错**：
 *   ① `(uint32_t)shape[i]` —— shape 是 int64_t，负数或 > 2^32 直接截断；
 *   ② 累乘本身在 uint32 里回绕。
 * 后果不是崩，是**分核参数变成一个错的小数字** ⇒ 少算一部分元素、
 * 或者每核区间算错越界写。这类"算错但不报错"正是本仓库最怕的。
 *
 * 改成：越界一律让 Calc* 返回 false，由 host 转成 INVALID_VALUE。
 * **宁可拒绝，也不给静默错的结果** —— 与 CalcSquareSum 拒绝
 * 不连续归约轴是同一条原则。 */
inline bool DimOk(int64_t v)
{
    return v >= 0 && v <= 0xFFFFFFFFll;
}

/* acc *= v，溢出或 v 越界返回 false（acc 保持不变）。 */
inline bool MulDim(uint32_t &acc, int64_t v)
{
    if (!DimOk(v)) { return false; }
    uint64_t prod = (uint64_t)acc * (uint64_t)v;
    if (prod > 0xFFFFFFFFull) { return false; }
    acc = (uint32_t)prod;
    return true;
}

/* 从 int64_t 收窄成 uint32_t，越界返回 false。 */
inline bool NarrowDim(uint32_t &out, int64_t v)
{
    if (!DimOk(v)) { return false; }
    out = (uint32_t)v;
    return true;
}

inline bool NarrowU64(uint32_t &out, uint64_t v)
{
    if (v > 0xFFFFFFFFull) { return false; }
    out = (uint32_t)v;
    return true;
}

/* ═════════════ Concat ═════════════ */
struct ConcatTiling {
    uint32_t rows, nOut, numIn, strategy, blockDim, ubBudget;
    // 向量拼装路（strategy 3）的**列带数**：把输出的列切成 bands 条带，
    // blockDim 个核排成 bands × (blockDim/bands) 的二维网格。1 ＝ 不分带（旧行为）。
    uint32_t bands;
    // 最宽的一路。kernel 侧的暂存区要按它留边界余量（相交的路一律整行入暂存，
    // 一条带最多切开两条，所以最多多出 2·maxIn·esz 一行）。
    uint32_t maxIn;
    // 按列切时每核每路至少搬多少**元素** ＝ COL_BURST_BYTES / esz。
    // host 和 kernel 必须用同一个值，所以由 host 算好经 TilingData 传下去；
    // 以前是两边各写一份（host 用常量、kernel 用移位），S9_COLEL 只改得动 host 那份，
    // 于是那个旋钮扫出来的全是"两边不一致"的配置，结论作废。
    uint32_t colBurstEl;
    // 列带取整的粒度（**元素**，＝ bandGrainBytes / esz）。host 算好经 TilingData 下发，
    // kernel 不再自带一份常量 —— 这是 colBurstEl 踩过的同一个坑：
    // 两边各写一份时旋钮只改得动 host 那份，扫出来的全是"两边不一致"的配置。
    uint32_t bandGrain;
    uint32_t bandBal;    // 1 = 均衡切分（默认）、0 = 旧的「统一带宽＋末带吃剩下」
    // 拼装区的缓冲块数（1 或 2）。**它进 UB 账**，所以必须和 GatherFitsBands
    // 用的是同一个值 —— 算得下、真机放不下就是 vector core exception 507035。
    uint32_t aBuf;
    // 任一条带里最多几路非空输入 —— kernel 按它开暂存余量（见 MaxInputsPerBand）。
    uint32_t maxInBand;
    // **host 实际发布给 kernel 的余量路数**（slack 的口径）。默认取全部 numIn，
    // 依据是实测「精确值反而更慢」（见 op_host 里 S9_MAXINB 那段：G16 精确 516 / 全路数 503）。
    // **可行性判据必须用发布的这个数**：原先 host 按精确的 maxInBand 算账、
    // kernel 按 numIn 开余量，于是"账算得下、真机放不下"（vector core exception 507035）。
    uint32_t slackIn;
    // 拼装路暂存区的缓冲块数（进 UB 账，也是 kernel 事件号的上限 —— 见 S9L_BUF_N）。
    uint32_t gBuf;
    /* 边界路裁剪（`S9_TRIM`）：
     *   0 ＝ 关（默认，全部整行入暂存）
     *   1 ＝ 只裁 kernel 的**读**（UB 账仍按旧口径 ⇒ **tileG 不变**，
     *       这样"少读字节"的收益与"UB 变大 ⇒ tileG 变大"的收益能分开量）
     *   2 ＝ 读与 UB 账一起裁（tileG 会变大 —— tier 2 的单独实验）
     * 1 与 2 的 kernel 行为完全一样，差别只在 host 怎么算账。 */
    uint32_t trimOn;
    uint32_t tblK;            // 偏移表只留 K+3 行（S9_TBLK）——**进 UB 账**
    uint32_t denseOn{0};      // 单带非对齐行按超行密排；与 kernel 共用布局账
    /* 二维分核的**列组数**（仅 strategy 0）。1 ＝ 旧行为（纯按行切）。
     * 核编号拆成 `rowIdx = bi / colParts`、`colIdx = bi % colParts`：
     * 行仍按 `rows` 组切（与旧行为逐字相同），每组再把**每一路的列**切 colParts 份。 */
    uint32_t colParts;
    uint32_t nIn[CONCAT_MAX_IN], offOut[CONCAT_MAX_IN];
};

/* 某个带数下，**任何一条带里最多有几路非空输入**。
 *
 * 暂存区按「每路一块、块起点 32B 对齐」布局，所以余量是 32B × 本带路数。
 * 原来按全部 numIn 留，248 路时是 15872 B（UB 的 8.4%），而一条带通常只有几十路 ——
 * 这块是 `fixed`，直接把 tileG 压小，而暂存读的指令条数是 `numIn × rows / tileG`。
 *
 * **不能用 ⌈numIn/bands⌉+2 这种估计**：各路宽度是随机的（赛题按 randint(0,step) 切），
 * 一条带里完全可能挤进远超均分数量的窄路。host 手上有全部宽度，精确数一遍就好，
 * O(numIn)，而且 kernel 侧照这个数开缓冲，两边必须同源。
 */
/* ═══ 列带的边界：**均衡切分**，不是「统一带宽 ＋ 末带吃剩下」═══
 *
 * 原来 `bandCols = AlignUp(⌈nOut/bands⌉, grain)`，前 bands−1 条都取这个宽度、
 * 末带只剩零头。零头可以小得离谱：
 *     nOut=16384、bands=10、grain=256 ⇒ bandCols=1792，
 *     **前 9 条各 1792、末带只有 256** —— 末带那一组核只有别人 1/7 的活。
 * 总活是定的，于是另外 36 个核每人扛 1/36 而不是 1/40 ⇒ **白丢约 10%**。
 * msprof 上看得见：(4096,16384) fp16 那档 `aiv_time` 393.69 而
 * `Task Duration` 465.72，中位核有 15% 的时间不忙 —— 就是在等末带那几个核之外的人。
 *
 * 改成把 **G = ⌈nOut/grain⌉ 个粒度**均分给 bands 条带：
 * 第 b 条拿 [b·G/bands, (b+1)·G/bands) 这些粒度。于是各带最多差一个粒度
 *（上例变成 6 条 1536 ＋ 4 条 1792，极差 1.17× 而不是 7×）。
 *
 * **host 与 kernel 必须用同一条公式**，否则两边算出的列区间不重合，
 * 会互相写到对方的地盘上（结果错且不报错）。所以只留这一个来源。 */
inline uint32_t BandGrains(uint32_t nOut, uint32_t grainEl)
{
    return grainEl ? CeilDiv(nOut, grainEl) : nOut;
}
/* 第 b 条带的列区间 [c0, c1)。b ∈ [0, bands)。 */
inline void BandRange(uint32_t nOut, uint32_t bands, uint32_t grainEl, uint32_t b,
                      uint32_t &c0, uint32_t &c1, uint32_t bal = 1)
{
    if (bands <= 1) { c0 = 0; c1 = nOut; return; }
    const uint32_t g = grainEl ? grainEl : 1;
    if (bal == 0) {                       // 旧行为：统一带宽 ＋ 末带吃剩下（留作 A/B 对照）
        const uint32_t bc = AlignUp(CeilDiv(nOut, bands), g);
        c0 = b * bc;
        c1 = c0 + bc;
        if (c0 > nOut) { c0 = nOut; }
        if (c1 > nOut) { c1 = nOut; }
        return;
    }
    const uint32_t G = BandGrains(nOut, g);
    const uint32_t g0 = (uint32_t)((uint64_t)b * G / bands);
    const uint32_t g1 = (uint32_t)((uint64_t)(b + 1) * G / bands);
    c0 = g0 * g;
    c1 = g1 * g;
    if (c0 > nOut) { c0 = nOut; }
    if (c1 > nOut) { c1 = nOut; }
}
/* 最宽的一条带（UB 账按它开）。 */
inline uint32_t BandColsMax(uint32_t nOut, uint32_t bands, uint32_t grainEl, uint32_t bal = 1)
{
    if (bands <= 1) { return nOut; }
    const uint32_t g = grainEl ? grainEl : 1;
    if (bal == 0) { return AlignUp(CeilDiv(nOut, bands), g); }
    return CeilDiv(BandGrains(nOut, g), bands) * g;
}

inline uint32_t MaxInputsPerBand(uint32_t nOut, const uint32_t *lens, uint32_t numIn,
                                 uint32_t bands, uint32_t grainEl, uint32_t alignEl,
                                 uint32_t bal = 1)
{
    if (numIn == 0) { return 0; }
    if (bands <= 1) { return numIn; }
    const uint32_t g  = grainEl ? grainEl : (alignEl ? alignEl : 1);
    if (g == 0) { return numIn; }
    /* **一遍扫完，O(numIn + bands)**（原先是 B×numIn：每条带都把前缀和从 0 重算）。
     *
     * 依据是 `BandRange` 的两个性质：
     *   · 各带的列区间**互不重叠**（g1(b) = g0(b+1)，相邻带首尾相接）；
     *   · `b` 增大时区间**单调右移**。
     * 而 `lens` 的前缀和也是单调不减的 ⇒「与本带相交」的那些路在两条带之间
     * 只会右移、不会回退：**每条路最多跨两条带**，所以内层计数加起来是 O(numIn)。
     *
     * `i0/off0` 是「第一个右端 > c0 的非空路」及其起始列。空片（`n == 0`）
     * 按定义不相交，直接跳过 —— 它加进前缀和也是加 0，与旧写法逐位等价。 */
    uint32_t best = 0, i0 = 0, off0 = 0;
    for (uint32_t b = 0; b < bands; ++b) {
        uint32_t c0 = 0, c1 = 0;
        BandRange(nOut, bands, g, b, c0, c1, bal);     // 与 kernel 同源
        if (c0 >= c1) { continue; }
        while (i0 < numIn) {                           // 单调推进，不回退
            const uint32_t n = lens[i0];
            if (n == 0) { ++i0; continue; }
            if (off0 + n > c0) { break; }
            off0 += n;
            ++i0;
        }
        uint32_t cnt = 0, off = off0;
        for (uint32_t i = i0; i < numIn; ++i) {
            const uint32_t n = lens[i];
            if (n == 0) { continue; }
            if (off >= c1) { break; }
            ++cnt;
            off += n;
        }
        if (cnt > best) { best = cnt; }
    }
    return best ? best : 1;
}

/* ═══ 带数的**合法性**（拼装路 strategy 3）：这一个候选带数能不能上？ ═══
 *
 * **只留这一个来源。** op_host 选带（CalcConcatLens 里那个 `for (B = 2 …)`）与
 * 脱机的候选枚举工具（op/src/test/cand_concat.cpp）都调它 —— 工具里"合法"而
 * 线上不合法（或反过来）会把取舍表变成一张看不懂的废纸。
 *
 * 判据逐条（每一条都来自一次真机教训，见各自的原文注释）：
 *   · 带数必须是核数的因数 —— 否则 kernel 悄悄退回 bands=1，而 UB 账按分带算；
 *   · 行要分得出这么多组（`blockDim/B ≤ rows`）；
 *   · 最宽那条带至少 BAND_MIN_BYTES，且不窄于最宽一路的 4 倍
 *     （边界路整行入暂存，带宽太窄不值）；
 *   · 均衡切分要求粒度数 ≥ 带数；旧切法要求前 bands−1 条铺不满
 *     （否则末带整条空）。
 *
 * **`B == 1` 直接放行**：单带是 host 的兜底默认，带宽/末带那两条判据是为
 * "切一条带来"设的，对整行不成立。漏掉这一条会让工具把**线上真跑的**单带配置
 * 从表里筛掉（小块形状上 host 选的正是单带）。
 *
 * 注意它**不含 UB 实账** —— 那是 GatherFitsBands / PlanConcatStage 的事，
 * 两者一起才构成"这个候选能不能上"。 */
inline bool ConcatBandLegal(const ConcatTiling &t, uint32_t esz, uint32_t B,
                            uint32_t grainEl, uint32_t alignEl, uint32_t bandMinBytes)
{
    if (B == 0 || B > t.blockDim || B > BAND_MAX || t.blockDim % B != 0) { return false; }
    if (B == 1) { return true; }                 // host 的兜底默认，判据不适用
    if (t.blockDim / B > t.rows) { return false; }
    const uint32_t gEl = grainEl ? grainEl : (alignEl ? alignEl : 1);
    const uint32_t bc = BandColsMax(t.nOut, B, gEl, t.bandBal);
    if (bc == 0 || (uint64_t)bc * esz < bandMinBytes) { return false; }
    if (t.bandBal != 0) {
        if (BandGrains(t.nOut, gEl) < B) { return false; }
    } else if ((uint64_t)bc * (B - 1) >= t.nOut) {
        return false;
    }
    if ((uint64_t)bc < 4ull * (t.maxIn ? t.maxIn : 1)) { return false; }
    return true;
}

/* 拼装路（strategy 3）的 UB 账：给定带数，**这个配置放不放得下**。
 *
 * 算术本体在 `s9_concat_layout.h`（`PlanConcatStage`），**与 op_kernel 的 Init 同源**。
 * 这里只负责把「带数」翻译成「一条带多宽、本组最多几行」，再去问那一份账。
 *
 *   slackIn ＝ host **实际发布**给 kernel 的余量路数（0 ⇒ 全部 numIn，与默认一致）。
 *             必须用发布值而不是精确值：kernel 是按 TilingData 里的数开缓冲的。
 *   myRows  ＝ 本组最多几行（0 ⇒ 不限）。只影响返回的 tileG，不影响"放不放得下" ——
 *             PlanConcatStage 会在放不下时逐行回缩，缩到 1 行仍放不下才判不可行。
 *   gBuf    ＝ 暂存块数（默认 2，与 op_host 的默认一致）。
 */
inline bool GatherFitsBands(uint32_t nOut, uint32_t numIn, uint32_t maxIn,
                            uint32_t esz, uint32_t bands,
                            uint32_t grainElIn = 0, uint32_t aBuf = 1,
                            uint32_t slackIn = 0, uint32_t bal = 1,
                            uint32_t myRows = 0, uint32_t gBuf = 2,
                            const uint32_t *lens = nullptr, bool trimAccount = false,
                            uint32_t tblK = 0, uint32_t *tileGOut = nullptr, uint32_t dense = 0)
{
    if (esz == 0 || nOut == 0 || bands == 0 || numIn == 0) { return false; }
    const uint32_t grainEl = grainElIn ? grainElIn : (BAND_GRAIN_BYTES / esz);
    const uint32_t fallbackAlign = 32 / esz;
    const uint32_t bandCols = BandColsMax(nOut, bands,
                                          grainEl ? grainEl : (fallbackAlign ? fallbackAlign : 1),
                                          bal);
    if (bandCols == 0) { return false; }
    const uint32_t nSlack = (slackIn && slackIn < numIn) ? slackIn : numIn;
    /* 裁剪版的 UB 账：逐路行宽之和要按**最宽的那条带**取（host 就是按它开的缓冲）。
     * 只有在 trimAccount 且拿得到 nIn[] 时才算；否则传 0 ⇒ PlanConcatStage 走旧口径。 */
    uint64_t trimRowBytes = 0;
    if (trimAccount && lens != nullptr) {
        for (uint32_t b = 0; b < bands; ++b) {
            uint32_t c0 = 0, c1 = 0;
            BandRange(nOut, bands, grainEl ? grainEl : fallbackAlign, b, c0, c1, bal);
            if (c0 >= c1) { continue; }
            const uint64_t s = ConcatBandStagedBytes(esz, lens, numIn, c0, c1, /*trim=*/true);
            if (s > trimRowBytes) { trimRowBytes = s; }
        }
    }
    const ConcatStageLayout L = PlanConcatStage(bandCols, esz, numIn, maxIn, nSlack, gBuf, aBuf,
                                                0, myRows, tblK, bands == 1 ? dense : 0u, trimRowBytes, bands == 1);
    if (tileGOut != nullptr) { *tileGOut = L.fits ? L.tileG : 0; }
    return L.fits;
}

/* ═══ 逐核布局表：**布局下沉**（`S9_LAYOUT=1`）═══
 *
 * 原先每个核在 `Init()` 里自己算一遍「我是哪个带、管哪些列/行」再调
 * `PlanConcatStage` 反推 tileG 与整份 UB 布局。这些量**只依赖 tiling**，
 * host 上算一次就够 —— 这里把它预先算好，kernel 只读自己那一行。
 *
 * **首版必须与 kernel 现算的逐字段一致**（否则就不是"只改计算位置"了）：
 *   · 核 → (带, 组, 列区间, 行区间) 走 `MapConcatCore`，与 kernel 同一份；
 *   · tileG / 各缓冲跨度走 `PlanConcatStage`，与 kernel 同一份；
 *   · `myRows` 用**本核实际行数**（不是组的上界），所以连"尾组少一行"这种
 *     差别都不引入。
 *
 * 返回放不下的核数（0 ＝ 全部可行）。**放不下就调用方别用这张表** ——
 * 静默降级会让"这一版为什么慢"永远查不出来。
 *
 * `outCap` 给足 blockDim 行；`*rowsWritten` 回填实际写了几行。 */
inline uint32_t CalcConcatCoreLayouts(const ConcatTiling &t, uint32_t esz, uint32_t blockDim,
                                      uint32_t gBuf, uint32_t aBuf, uint32_t maxNIn,
                                      uint32_t nSlack, uint32_t tileGMax, uint32_t numInputs,
                                      uint32_t maxIn, uint32_t bands, uint32_t bandGrain,
                                      ConcatCoreLayoutRow *out, uint32_t outCap,
                                      uint32_t *rowsWritten)
{
    if (rowsWritten) { *rowsWritten = 0; }
    if (esz == 0 || out == nullptr) { return blockDim; }
    const uint32_t alignEl = (32 / esz) ? (32 / esz) : 1;
    const uint32_t grainEl = bandGrain ? bandGrain : alignEl;
    uint32_t bad = 0;
    const uint32_t n = (blockDim < outCap) ? blockDim : outCap;
    for (uint32_t k = 0; k < n; ++k) {
        const ConcatCoreMap m = MapConcatCore(t.rows, t.nOut, blockDim, bands, grainEl,
                                              t.bandBal, k, alignEl);
        // 模式 2 时 UB 账也按裁剪后的逐路行宽算（模式 1 传 0 ⇒ 旧口径 ⇒ tileG 不变）。
        // **与 ConcatBandStagedBytes 同源**：kernel 的前缀和用的是同一个函数。
        const uint64_t trimRowBytes =
            (t.trimOn >= 2) ? ConcatBandStagedBytes(esz, t.nIn, numInputs, m.cb0, m.cb1, true)
                            : 0;
        const ConcatStageLayout lay = PlanConcatStage(m.bw, esz, numInputs, maxIn, nSlack,
                                                      gBuf, aBuf, tileGMax, m.myRows,
                                                      t.tblK, m.bands == 1 ? t.denseOn : 0u, trimRowBytes, m.bands == 1);
        ConcatCoreLayoutRow &r = out[k];
        r.rowBeg = m.rowBeg; r.rowEnd = m.rowEnd;
        r.bw = m.bw; r.cb0 = m.cb0; r.cb1 = m.cb1;
        r.outPitch = lay.outPitch; r.tblRow = lay.tblRow; r.tileG = lay.tileG;
        r.stageSpan = lay.stageSpan; r.asmSpan = lay.asmSpan;
        r.offTblBytes = lay.offTblBytes; r.dltBytes = lay.dltBytes;
        r.castSpan = lay.castSpan; r.asmGBytes = lay.asmGBytes;
        if (!lay.fits || lay.tileG == 0) { ++bad; }
    }
    if (rowsWritten) { *rowsWritten = n; }
    return bad + (blockDim > outCap ? (blockDim - outCap) : 0);
}

/* ═══ Tiling 的旋钮 ═══
 *
 * 这 15 个量原先是 `CalcConcatLens` 的**位置参数**，默认值抄在两处
 *（它自己的声明，以及 `CalcConcat` 转发时写死的那一串常量）。两个后果：
 *   ① 加一个旋钮要改三处（声明、定义、转发），漏一处就是**默认值分叉** ——
 *      host 走一套、孪生走另一套，而两边都编得过；
 *   ② 15 个同类型的位置参数**调换两个不报错**。调用点看到的是
 *      `…, 512, 2, true, 40, 16384, 0, 32768, …` 这样一串裸数字，
 *      读不出哪个是哪个；而这些旋钮改的是切核与选路，错了只是"慢"或"选了别的分支"，
 *      不会报错。
 *
 * 收成一个带默认成员初始化的结构体：默认值**只有一处**，调用点按名字指定
 *（`ConcatKnobs kn; kn.trimOn = 1;`），少写的字段自动取发布默认值。 */
struct ConcatKnobs {
    uint32_t minBurst     = MIN_BURST;
    // **默认与 op_host 里发布的默认值一致（开）。**
    // 原先默认 false，于是所有脱机测试都在测一个不发布的配置 ——
    // 向量拼装路（含列带分核）整条从来没被孪生跑到过。
    bool     gatherOn     = true;
    uint32_t maxCores     = AIV_MAX;
    uint32_t inSplitBytes = IN_SPLIT_BYTES;
    uint32_t colBasis     = COL_BASIS_AVG;
    uint32_t colBurstBytes = COL_BURST_BYTES;
    uint32_t bandMinBytes  = BAND_MIN_BYTES;
    // int8 是否借道 half 走拼装路。**默认开** ——
    // 910B4 实测 (2048,4096) 180.81 → 61.40、
    // (8192,4096) 693.12 → 156.20、(1024,1024) 83.71 → 18.30，
    // 三档 2.9~4.6×，全部逐位相同。S9_I8G=0 可关掉做 A/B。
    bool     i8Gather     = true;
    uint32_t gatherMaxOut = GATHER_MAX_OUT;
    uint32_t bandBal      = 1;
    uint32_t bandGrainBytes  = BAND_GRAIN_BYTES;
    uint32_t inSplitMinWrite = IN_SPLIT_MIN_WRITE;
    uint32_t abuf2MaxIn   = ABUF2_MAX_IN;
    uint32_t trimOn       = 0;
    /* Gather 偏移表只留 2 行（`S9_TBL1`）。**进 UB 账** —— 表从「每行一份」变成
     * 固定 2 行之后，省下的 UB 让 tileG / gBuf 有得可分。host、kernel、孪生
     * 三方必须用同一个值，见 `s9_concat_layout.h` 里 `offTbl` 的说明。 */
    uint32_t tblK         = 0;
    uint32_t denseOn      = 1;  // 0 关闭；1 自动保护；2 强制诊断
    /* 二维分核的每列块字节下限（0 ＝ 关，旧行为）。**与 COL_BURST_BYTES 是两回事**：
     * 那个是 strategy 1「把一路切给多个核」的下限（24 KB，为二维窄路标定）；
     * 这里是「行不够核数时，把每一路的列再切几份」的下限，路本来就宽得多。 */
    uint32_t gridBurstBytes = 0;
    /* 输出线性切分（strategy 5，`rows == 1` 专用）。**默认开** ——
     * 910B4 实测 2.59×~3.21×（见 docs/submission-env.md）。`S9_LIN=0` 可关掉做 A/B。 */
    bool     linearOn     = true;
    /* 拼装路（strategy 3）的**最小 tileG**（`S9_S3MINTG`，0 ＝ 关，旧行为）。
     * 910B1 实测 strategy 3 输给 strategy 0 的只有「行宽 32 KB、路数少」的两档：
     *     4096,16384 f16 22 路  371.88 vs 289.48 µs（1.28×）
     *     2048,8192  f32 14 路  167.12 vs 138.53 µs（1.21×）
     * 按**带数**划线是错的：`1024,4096 f16 16 路`同样是 2 带，strategy 3 反而快 2.7×
     *（23.98 vs 64.80），而按带数 < 3 改道会把 151 个规格用例（含 Case1 形状）一起推走。
     * 两个输家的共同点是**一行本身就把 UB 占满** ⇒ tileG 只有 1~2，每行每路一条 DMA。 */
    uint32_t s3MinTileG   = 0;   // 2026-09-15 规格抽样 50 形状验证：默认 2 时几何平均 0.85×（22 赢 23 输）⇒ 退回 0
    /* 上一条只在**带数 ≤ 这个值**时生效（`S9_S3MINTGB`，默认 4）。910B1 扩大验证（12 形状，全部 exact）：
     * tileG=1 且 1~4 带时 s0 赢或持平（16384,4096 f32 1.36×、2048,8192 f32 1.22×、1000,999,33 f16 1.19×、
     * 4096,16384 f16 1.05×、8192,8192 f16 1.00），而 `256,512,128 f16 dim=1` 58 路 8 带 tileG=1
     * 让给 s0 反而**慢 2×**（78.82 → 161.43 µs）—— 带多时每核只碰几路，s3 仍然划算。 */
    uint32_t s3MinTileGMaxBands = 4;
    /* DMA 拼装快路（strategy 2）开关（`S9_ASM2`，默认开 ＝ 旧行为）。
     * 2026-09-15 按规格抽样量到它在多路时一致地慢（113~297 GB/s，标量占满）：
     * 它的门只看「偏移 32B 对齐、一行放得下」，不看 tile 有多小。关掉它是为了配对 A/B，
     * 让这些形状落到 strategy 3（能接时）或 strategy 0。 */
    bool     asm2On       = true;
    /* **strategy 3 能开到这么多带时，让出 strategy 2**（`S9_S2YIELD`，默认 2；0 ＝ 旧行为）。
     * 910B1 按规格抽样的配对（默认 vs `S9_ASM2=0`，17 档全部 CHECK exact）分得毫无例外：
     *   落到 ≥2 带的 strategy 3：7/7 档 strategy 2 输 1.5×~66×
     *     （`510,2,370,24 int32` 101 路 3901.65 → 71.06 µs）；
     *   落到单带 strategy 3 或 strategy 0：10/10 档 strategy 2 赢或持平（对方 0.30~0.92×）。
     * strategy 2 每 tile 每路一条 DMA、`tile = UB预算/行宽`，多带 strategy 3 的读指令少 bands 倍。 */
    uint32_t s2YieldBands = 2;
    /* **strategy 3 能开到这么多带时，strategy 4 也让出来**（`S9_S4YIELD`，默认 3；0 ＝ 旧行为）。
     * 910B1 规格抽样配对（默认 vs `S9_INSPLIT=0`）：落到 5~16 带 strategy 3 的 8 档里 7 档 s3 快
     * 1.38~2.72×、1 档持平；落到 s2/s0 的 2 档 s4 大赢。
     * 补量 2~4 带：≥3 带合计 13 档 10 赢 2 平 1 负（慢 7%），2 带一赢一输，单带 6 档全负 1.4× ⇒ 门槛 3。 */
    uint32_t s4YieldBands = 3;
};

/* shapes[k] 是第 k 个输入的形状（rank 维），dim 已转正。 */
inline bool CalcConcatLens(uint32_t rows, const uint32_t *lens, uint32_t numIn,
                           uint32_t esz, ConcatTiling &t,
                           const ConcatKnobs &kn = ConcatKnobs{});

inline bool CalcConcat(const int64_t *const *shapes, const int64_t *ranks,
                       uint32_t numIn, int64_t dim, uint32_t esz, ConcatTiling &t,
                       const ConcatKnobs &kn = ConcatKnobs{})
{
    if (numIn == 0 || numIn > CONCAT_MAX_IN || esz == 0) { return false; }
    int64_t rank = ranks[0];
    if (dim < 0 || dim >= rank) { return false; }

    uint32_t rows = 1;
    for (int64_t i = 0; i < dim; ++i) { if (!MulDim(rows, shapes[0][i])) { return false; } }

    // 各路在拼接轴之后的元素数
    uint32_t lens[CONCAT_MAX_IN];      // 4 KB 栈，可接受（原来是 128 KB 的 shapes 数组）
    for (uint32_t k = 0; k < numIn; ++k) {
        uint32_t n = 1;
        for (int64_t i = dim; i < ranks[k]; ++i) { if (!MulDim(n, shapes[k][i])) { return false; } }
        lens[k] = n;
    }
    return CalcConcatLens(rows, lens, numIn, esz, t, kn);
}

/* 按**各路长度**算 tiling —— host 侧路数很多时走这条，
 * 免得为了凑 `shapes/ranks` 开一个 CONCAT_MAX_IN×16 的大栈数组
 *（1024 路时那是 128 KB，直接爆栈）。 */
inline bool CalcConcatLens(uint32_t rows, const uint32_t *lens, uint32_t numIn,
                           uint32_t esz, ConcatTiling &t, const ConcatKnobs &kn)
{
    if (numIn == 0 || numIn > CONCAT_MAX_IN || esz == 0) { return false; }
    t.rows = rows;
    t.trimOn = kn.trimOn;
    t.tblK   = kn.tblK;
    t.denseOn = kn.tblK == 0 ? kn.denseOn : 0;
    t.colParts = 1;

    t.nOut = 0;
    uint32_t nMin = 0xFFFFFFFFu, nMax = 0;
    for (uint32_t k = 0; k < numIn; ++k) {
        uint32_t n = lens[k];
        t.nIn[k] = n;
        t.offOut[k] = t.nOut;
        // 累加也会溢出：9 路各 2^29 列就顶到头了
        if ((uint64_t)t.nOut + n > 0xFFFFFFFFull) { return false; }
        t.nOut += n;
        // **长度为 0 的输入是合法的**（赛题的随机切分会切出空片），
        // 但不能拿它去估 burst，否则核数会被压到 0
        if (n > 0 && n < nMin) { nMin = n; }
        if (n > nMax) { nMax = n; }
    }
    for (uint32_t k = numIn; k < CONCAT_MAX_IN; ++k) { t.nIn[k] = 0; t.offOut[k] = 0; }
    if (nMin == 0xFFFFFFFFu) { nMin = 1; }
    t.maxIn = nMax;
    t.numIn = numIn;

    // **一整行的字节数必须装得进 DataCopyPad 的 stride 字段**（uint32）。
    // CopyOut 填的是 `dstStride = (nOut − cols)·esz`，nOut 是所有输入的
    // 列数之和 —— 16 路各 2^26 列就到 2^30，×4 字节正好溢出。
    // 溢出的后果是**跨步变成一个小得离谱的数**，于是每行写到前一行的中间，
    // 结果错但不报错。宁可拒绝。
    if ((uint64_t)t.nOut * esz > 0xFFFFFFFFull) { return false; }

    // 默认按行切（burst 最长）；按行切用不满核时才考虑按列切，
    // 且列切能用几个核由 burst 下限反推
    uint64_t outBytes = (uint64_t)t.rows * t.nOut * esz;
    const uint32_t coreCap = (kn.maxCores && kn.maxCores < CONCAT_MAX_CORES)
                                 ? kn.maxCores : CONCAT_MAX_CORES;
    uint32_t blockDim = Clamp((uint32_t)CeilDiv((uint32_t)(outBytes > 0xFFFFFFFFull
                                                           ? 0xFFFFFFFFu : outBytes), kn.minBurst),
                              CONCAT_MIN_CORES,
                              // **标定上限与实际核数取小的那个。**
                              // CONCAT_MAX_CORES=40 是在 910B3 上真机标定的，
                              // 但换个型号（910B1/B2 有 48 个 AIV）就不该被它压住；
                              // 反过来核数更少的型号上也不能超。
                              coreCap);
    t.strategy = 0;
    t.bands = 1;
    // **拼装区的缓冲块数要在 UB 账之前定下来** —— 它是 perTile 的一项。
    // 判据见 ABUF2_MAX_IN 的实测表：路数少时开双缓冲赚 6~12%，路数多时亏。
    t.aBuf = (numIn <= kn.abuf2MaxIn) ? 2u : 1u;
    t.maxInBand = numIn;                     // 选带之前先按最坏情况，选中带数后再收紧
    // **发布给 kernel 的余量路数与暂存块数**。默认就是 op_host 会下发的值
    //（余量按全部路数、暂存两块），可行性判据按这一份算 —— 见字段说明。
    t.slackIn = numIn;
    t.gBuf = 2;
    // 粒度按元素下发；不足一个元素时退回 32B 对齐（与 kernel 的 fallback 一致）
    t.bandBal   = kn.bandBal;
    t.bandGrain = (esz && kn.bandGrainBytes >= esz) ? (kn.bandGrainBytes / esz) : (32 / esz);
    if (t.bandGrain == 0) { t.bandGrain = 1; }
    t.colBurstEl = (esz && kn.colBurstBytes >= esz) ? (kn.colBurstBytes / esz) : 1;
    if (t.colBurstEl == 0) { t.colBurstEl = 1; }
    if (t.rows < blockDim) {
        // 列切能开几个核，由 burst 下限反推。**基准取哪一路是关键。**
        //
        // 原先取 `nMin`（最短的一路）。碰上 (2024,3000) 沿 dim=0 拼这种形状就崩了：
        // rows=1、57 路、最短一路 3000 个 fp16 ＝ 6000 字节 < kn.minBurst=16384，
        // 于是 colCores=0，既不 >blockDim 也不 >rows，落进 else ⇒
        // **blockDim = rows = 1** —— 12 MB 的搬运整个压在一个 AIV 上。
        // 这不是推断：S9_DBG 在真机上打出 `strategy=0 blockDim=1`，
        // 而同形状 146 µs（fp16）／259 µs（fp32）也正是单核的量级。
        //
        // 最短的一路短，不构成"整体不该切列"的理由 —— 它自己就那么点数据，
        // 分到的核空转一点无所谓；决定核数的应该是**典型的一路**。
        // 改用平均一路做基准。`S9_COLBASIS=0` 切回旧行为，便于上板 A/B。
        uint32_t basis = (kn.colBasis == COL_BASIS_MIN || t.numIn == 0)
                             ? nMin : (t.nOut / t.numIn);
        // 载荷按字节算，再换回元素 —— kernel 侧要用同一个 colBurstEl，
        // 所以它经 TilingData 传过去，两边不可能算出不同的值。
        uint32_t colCores = t.colBurstEl ? (basis / t.colBurstEl) : 0;
        if (colCores > coreCap) { colCores = coreCap; }
        // 核数仍然用 colCores，**不要开满**。
        //
        // 曾经改成 `blockDim = coreCap`（想着"核数交给 kernel 按路各自定就行"），
        // 910B3 上一测就打脸：(2024,3000)/dim=0
        //   blockDim=40 → fp16 96.94 µs / fp32 125.46 µs
        //   blockDim=13 → fp16 **43.91** / fp32 **54.81**   ⇒ 快 2.2×
        // 同时验过不是 UB 预算的锅（`S9_UBX=94208` 强行调回大 ub，96.87／125.20，纹丝不动）。
        // 机理没查清（40 个 block 里多数只碰得到少数几路，调度/尾巴的代价压过了并行），
        // **但结论是量出来的**，先按量到的走。
        //
        // kernel 侧的按路分核照旧有效：coresK = nIn/COL_BURST_ELEMS 再被 bn_ 夹住，
        // 短路仍然只占一个核 —— 那部分是净赚，不受这里影响。
        if (colCores > t.rows) { t.strategy = 1; blockDim = colCores; }
        else { blockDim = t.rows ? t.rows : 1; }
    }
    t.blockDim = blockDim < 1 ? 1 : blockDim;

    // ═══ 向量拼装路（strategy = 3）的可行性 —— **要在 strategy 4 之前算出来** ═══
    //
    // 两条路都想接"多路小片"这类形状，谁先判谁就抢走。原先 strategy 4 在前，
    // 于是 int8 放宽门槛之后它把所有多路 int8 都吃掉了，拼装路再也轮不到。
    // 现在先把"拼装路能不能接"算清楚，strategy 4 只在它接不了时才放宽。
    //
    // **int8 借道 half**：《Gather》的 A2 支持类型里没有 int8，但《Cast》有
    // `int8_t ↔ half` 且 int8 值域在 half 里是精确整数（见 op_kernel 的说明），
    // 所以 int8 可以先 Cast 上去再 Gather。载体宽度因此是 2 而不是 1 ——
    // **但那个换算在 `s9_concat_layout.h` 的 `ConcatRowBudgetOf` 里**，
    // 这里原先另存了一份 `gsz`，声明之后只被 `(void)` 掉，从未参与计算。已删。
    bool     gatherOk    = false;
    uint32_t gatherBands = 1;
    if (kn.gatherOn && t.numIn > 0 && t.numIn <= ASM_MAX_IN && t.nOut > 0 &&
        t.nOut <= kn.gatherMaxOut && (esz == 2 || esz == 4 || (esz == 1 && kn.i8Gather))) {
        const uint32_t alignEl = esz ? (32 / esz) : 1;
        const uint32_t grainEl = kn.bandGrainBytes / esz;
        // ═══ 带数：在 blockDim 的因数里取满足下限的最大值 ═══
        //
        // **"带宽钉在 1024 B、带数由 nOut 反推"这条试过，是错的。** 依据是
        // (4096,4096) fp16 那档扫带数时 1024 B 最快（8 带 128.05 vs 4 带 2048 B 133.23），
        // 我把它外推到别的形状，上板一量三档全退：
        //     (2048,2048) int8 248路  4 带 512 B  58.74 → 2 带 1024 B  **76.46（+30%）**
        //     (16384,16384) f16 126路              1846.66 → **1999.68（+8%）**
        //     (4096,16384) f16 246路  10 带 3584 B 470.86 → 32 带 1024 B **491.21（+4%）**
        // 机理是**写密度**：宽行档一条带写出去是 blockLen=带宽、按 nOut·esz 跨步的块，
        // nOut=16384 fp16 时 1024/32768 只有 3% 密度，而 3584 B 有 11%。
        // 带越窄读指令越少，但写越稀 —— 两者的平衡点随 nOut 变，**不是一个常数带宽**。
        // 所以退回原来的选法，并把这条负面结论留在这里。
        //
        // 「必须整除核数」这一条也一起退回：它是"网格填满、没有核空转"的简单保证，
        // 而放开它带来的收益（能取到更窄的带）恰恰是上面证明有害的那个方向。
        // **判据全在 ConcatBandLegal 里**（与 op/src/test/cand_concat.cpp 同一份）——
        // 原先这里是内联的一串 continue，工具那边抄了一份，两边随时可能分叉。
        //
        // 「末带会整条空」那条只对**旧切法**成立：它把前 B−1 条都按 bc 铺，
        // 铺不下就把末带挤没了。**均衡切法下不会有空带** —— 只要粒度数 G ≥ B，
        // 每条至少分到一个粒度。**这才是均衡切分真正的价值**：不是"把末带那点
        // 零头摊平"（那不动关键路径，已被配对 A/B 证伪），而是**它能合法化更高的
        // 带数**。而每核读指令数 ∝ numIn·rows/(blockDim·bands)，带数直接进分母。
        // 顺带一个好性质：粒度是 512 B 的整数倍时，每条带都是整数个粒度
        // ⇒ **带宽自动保持 512 B 对齐**，不违反 p4 上量出来的那条判据。
        for (uint32_t B = 2; B <= t.blockDim && B <= BAND_MAX; ++B) {
            if (ConcatBandLegal(t, esz, B, grainEl, alignEl, kn.bandMinBytes)) {
                gatherBands = B;              // 取满足全部条件的**最大**带数
            }
        }

        // **至少要放得下一行（tileG ≥ 1）。** 账在 GatherFitsBands → PlanConcatStage 里，
        // 与 kernel 的 Init 同源，而且按 **host 发布的那份余量**（t.slackIn）算。
        t.maxInBand = MaxInputsPerBand(t.nOut, lens, t.numIn, gatherBands,
                                       grainEl, alignEl, kn.bandBal);
        // 本组最多几行：bands 条带 × 每条 blockDim/bands 组（host 只产出因数，
        // 不整除时 kernel 会把余数分给前几条带，取上界即可）。
        uint32_t bandRows = 0;
        if (gatherBands > 0 && t.blockDim && t.blockDim % gatherBands == 0) {
            const uint32_t grp = t.blockDim / gatherBands;
            bandRows = grp ? CeilDiv(t.rows, grp) : t.rows;
        }
        uint32_t mainTileG = 0;
        gatherOk = GatherFitsBands(t.nOut, t.numIn, nMax, esz, gatherBands,
                                   grainEl ? grainEl : alignEl, t.aBuf, t.slackIn, kn.bandBal,
                                   bandRows, t.gBuf, lens, /*trimAccount=*/kn.trimOn >= 2,
                                   /*tblK=*/kn.tblK, &mainTileG, t.denseOn);
        const bool mainOk = gatherOk;

        /* ═══ 救一把：`rows < 核数` 且**这个配置根本进不了拼装路** ═══
         *
         * `blockDim` 在判拼装路**之前**就被「按行切用不满核 ⇒ 按列切；列切也用不满
         * ⇒ `blockDim = rows`」压过一道。后果是连锁的：rows=4 ⇒ blockDim=4
         * ⇒ 带数最多 4 ⇒ 每带 16384 列 ＝ 64 KB/行 ⇒ **一行都放不进 UB**
         * ⇒ 拼装路判不可行 ⇒ 掉回按行切。910B4 实测这一族只有 **18~52 GB/s**。
         *
         * 而**列带分核不需要 `rows ≥ blockDim`** —— 只需要 `rows ≥ blockDim/bands`
         *（组数），`bands == blockDim` 时组数恒为 1。所以对每个候选带数 B 用
         * `bd = B · min(rows, coreCap/B)` 重算一遍，取能用上最多核的那个。
         *
         * **只在 `!gatherOk` 时做，这一条是实测逼出来的，不是保守起见。**
         * 无条件放开时量到（旧包 vs 新包）：
         *     rows=2  58.09 → **10.45**（5.56×）   rows=4  60.11 → **14.59**（4.12×）
         *     rows=8  91.96 → **16.76**（5.49×）   rows=16 23.96 → 22.59（1.06×）
         *     rows=39 16.59 → 16.66（1.00×）
         *     **rows=24 20.04 → 23.55（0.85×，慢 17%）**  ← 回归
         * 已经在拼装路上的档（rows≥12，300~500 GB/s）本来就不该碰；
         * 真正病的是掉出拼装路的那些（18~52 GB/s），对它们"任何配置都比现状好"。
         * ⇒ 用"进不了拼装路"当门，比拿 6 个点去拟一个 rows 阈值有原理得多。 */
        /* **只救量过的那一段。** 本机实测的赢全部来自 `rows ≤ 8`
         * （2 行 5.63× · 4 行 4.04× · 8 行 5.13×，都是 strategy 0 → 3）；
         * `rows ∈ [9, coreCap)` 那一段**本机一次都量不到** —— 40 核时它们
         * 本来就 `gatherOk`（救援不触发），只有 48 核才会落进来。
         * 而评测机（48 核）上那一次观测是**退化**：v35→v36 的 kernel 逐字节相同，
         * 唯一差别就是这条救援，而 C4 涨了 1.8%（89.237 vs 87.692，
         * 而 C4 的同版本重现精度是 0.06%）。
         * ⇒ 没量过的区间不放行。`LINEAR_RESCUE_MAX_ROWS` 要往上抬，
         * 先在 48 核的机器上扫，别拿一个远端观测去猜。 */
        if (!gatherOk && t.rows <= LINEAR_RESCUE_MAX_ROWS && t.rows < coreCap) {
            uint32_t bestBd = 0, bestB = 0;
            for (uint32_t B = 1; B <= coreCap && B <= BAND_MAX; ++B) {
                uint32_t grp = coreCap / B;
                if (grp > t.rows) { grp = t.rows; }
                if (grp == 0) { continue; }
                const uint32_t bd = B * grp;
                if (bd <= bestBd || bd <= t.blockDim) { continue; }   // 用不上更多核就不换
                ConcatTiling probe = t;
                probe.blockDim = bd;
                if (!ConcatBandLegal(probe, esz, B, grainEl, alignEl, kn.bandMinBytes)) { continue; }
                if (!GatherFitsBands(t.nOut, t.numIn, nMax, esz, B,
                                     grainEl ? grainEl : alignEl, t.aBuf, t.slackIn,
                                     kn.bandBal, CeilDiv(t.rows, grp), t.gBuf, lens,
                                     /*trimAccount=*/kn.trimOn >= 2,
                                     /*tblK=*/kn.tblK, nullptr, t.denseOn)) { continue; }
                bestBd = bd; bestB = B;
            }
            if (bestBd > 0) {
                t.blockDim = bestBd;
                gatherBands = bestB;
                t.maxInBand = MaxInputsPerBand(t.nOut, lens, t.numIn, gatherBands,
                                               grainEl, alignEl, kn.bandBal);
                gatherOk = true;
            }
        }
        /* **一行就占满 UB 时让出拼装路**（`kn.s3MinTileG`，默认 0 ＝ 不生效）。
         * 只看主路算出来的 tileG；走过救援的形状（rows ≤ 8，strategy 3 赢 4~5×）不碰。 */
        /* **只认"UB 装不下"造成的 tileG 小**：本组总共就只有 1 行时 tileG 天然是 1，那不是这条规则要管的
         *（`[37,65]+[11,65] dim=0` 12 KB 单核那档就是这样，让给 s0 毫无依据）。 */
        if (mainOk && kn.s3MinTileG > 0 && mainTileG < kn.s3MinTileG &&
            gatherBands <= kn.s3MinTileGMaxBands && (bandRows == 0 || mainTileG < bandRows)) {
            gatherOk = false;
        }
    }

    // ═══ 按输入切核（strategy = 4）═══
    //
    // **路数很多时，按行切会退化成"大量小 DMA"。** 算一笔账：
    // 250 路、rows=128、blockDim=40 ⇒ 每核约 3 行、每路平均 nOut/250 个元素，
    // 于是每核要发 **250×2 = 500 条 DMA，每条只搬几十字节**。
    //
    // 改成每个核包干若干**整路**（全部行）：每核约 6 路 × 2 = **12 条**，
    // 每条 `blockCount = rows`。搬的字节数一模一样，**指令数少一个 blockDim 倍**。
    // 两条路的 CopyOut 形状也完全相同（都是 blockLen=nIn·esz、
    // dstStride=(nOut−nIn)·esz 的跨步块），只是后者一条指令里的块更多 ——
    // 所以这不是"用别的代价换指令数"，是**纯赚**。
    //
    // 代价只有一个：**负载按路分，均衡不如按行分**。所以只在两个条件同时满足时用：
    //   ① `numIn ≥ 2·blockDim` —— 每核至少摊到两路，否则空核太多
    //      （样例 case1 是 9 路 / 8 核，**不满足**，仍走已验证的按行切）；
    //   ② 按行切时每条指令搬的字节数低于门限 —— 那才说明它真的退化了。
    //
    // 门限见 IN_SPLIT_BYTES 的说明（384，扫出来的）。用 `S9_INSPLIT` 覆盖以便上板扫。
    // ③ **每一路的写块不能太窄** —— 见 IN_SPLIT_MIN_WRITE。
    //    这一条是 2026-09-10 用 msprof 的流量放大量出来的，不是推的。
    const bool s4Yield = (kn.s4YieldBands > 0 && gatherOk && gatherBands >= kn.s4YieldBands);
    if (!s4Yield && t.strategy == 0 && t.numIn >= 2 * t.blockDim && t.blockDim > 1 && t.nOut > 0) {
        uint32_t rpc  = CeilDiv(t.rows, t.blockDim);
        uint32_t avgN = t.nOut / t.numIn;
        uint64_t bytesPerInstr = (uint64_t)rpc * avgN * esz;
        // 按输入切核是**每路一块**地写出去的，块宽就是 avgN·esz。
        // 块太窄时写会被 32B 粒度放大：(128,4096) fp16 123 路那档
        // msprof 量到 **写 2.86×、读 1.93×**（写 2928 KB 而只需 1024 KB），
        // 每路只有 66 字节。同一形状交给拼装路（写块 = 带宽 × esz = 1024 字节）
        // 是 **26.51 → 18.84 µs（−29%）**。
        // 所以窄写块的形状让给拼装路 —— 但**只在拼装路真的接得住时**，
        // 接不住就还是按输入切（那仍比按行切好得多）。
        const bool tooNarrowToWrite = ((uint64_t)avgN * esz < kn.inSplitMinWrite) && gatherOk;
        if (tooNarrowToWrite) { bytesPerInstr = kn.inSplitBytes; }   // 落选，交给下面的 strategy 3
        // **int8 曾经在这里放宽过门槛，现在撤掉了。**
        //
        // 当时的理由是「int8 进不了 Gather，按输入切是它唯一的好路」，
        // 依据是 (2048,4096) int8 123 路：按行切 322 · 按输入切 178。
        // **那个标定点选窄了。** 行数一多结论就反过来 ——
        // (8192,4096) int8 123 路：按行切 498 · 按输入切 **693**。
        // 榜单也印证了：放宽的那一版 Case5 从 428.99 涨到 476.21（+47 µs），
        // 按型号折算正好落在 334 → 464 这条线上。
        //
        // 而且前提本身没了：int8 现在借道 half 就能进拼装路
        //（(8192,4096) 那档 **156 µs**，比两条老路都好 3~4 倍），
        // 拼装路接不了的窄片档（如 248 路、每路 8 列）本来就满足 512 的原门槛。
        // 所以门槛对所有 dtype 一视同仁。
        if (bytesPerInstr < kn.inSplitBytes) { t.strategy = 4; }
    }

    // 单块 UB 预算：每核约 4 轮
    // 按列切时每个核一次最多碰 ceil(nIn/coresK) 列，coresK 向下取整
    // ⇒ 单核切片最大约 2·COL_BURST_ELEMS。取它与 nOut/blockDim 的较大者：
    // 实测 UB 预算调大调小对这两档没影响（96.94 vs 96.87 µs），
    // 但没有理由把预算估小，估大只是多留余量。
    uint32_t colsPerCore = t.nOut;
    if (t.strategy == 1) {
        uint32_t byBurst = 2 * t.colBurstEl;
        uint32_t byCore  = CeilDiv(t.nOut, t.blockDim ? t.blockDim : 1);
        colsPerCore = byBurst > byCore ? byBurst : byCore;
    }
    uint32_t rowBytes = AlignUp(colsPerCore * esz, 32);
    uint32_t rowsPerCore = (t.strategy == 1) ? t.rows : CeilDiv(t.rows, t.blockDim);
    uint32_t tileTarget = CeilDiv(rowsPerCore, TARGET_ITERS);
    if (tileTarget < 1) { tileTarget = 1; }
    uint64_t want = (uint64_t)tileTarget * rowBytes;
    t.ubBudget = want < UB_MIN ? UB_MIN : (want > UB_MAX ? UB_MAX : (uint32_t)want);

    // ═══ 拼装快路（strategy = 2）═══
    // 把一整个输出行在 UB 里拼好再一次写出，**把 numIn 条跨步写合成 1 条连续写**。
    // 只在四个条件都成立时启用，否则退回常规路径：
    //   ① 按行切（列切时每核只碰输出的一段，本就没有"整行"可拼）；
    //   ② **每个 offOut[i]·esz 是 32 的倍数** —— DataCopyPad 要求 UB 侧落点 32B 对齐。
    //      这一条最挑剔：big（偏移 1000 元素 ＝ 2000 字节）不满足，
    //      huge（偏移 2000 元素 ＝ 4000 字节）满足。
    //   ③ 一整行放得进 UB 预算（放不下就得按列再切，连续写的好处也就没了）；
    //   ④ 输入路数 ≤ S9_ASM_MAX（kernel 侧要把指针放栈上）。
    const bool s2Yield = (kn.s2YieldBands > 0 && gatherOk && gatherBands >= kn.s2YieldBands);
    if (kn.asm2On && !s2Yield && t.strategy == 0 && t.numIn > 0 && t.numIn <= ASM_MAX_IN && t.nOut > 0) {
        uint64_t pitch = AlignUp((uint32_t)((uint64_t)t.nOut * esz), 32);
        bool aligned = pitch <= t.ubBudget;
        for (uint32_t k = 0; k < t.numIn && aligned; ++k) {
            if (((uint64_t)t.offOut[k] * esz) % 32 != 0) { aligned = false; }
        }
        if (aligned) { t.strategy = 2; }
    }

    /* ═══ 向量拼装路（strategy = 3）—— **这是默认主路径** ═══
     *
     * ⚠️ **这里原先有一大段说它"实测更慢、默认关闭、只留在 S9_STRAT=3 后面备查"，
     * 那段里有五条现在都是假的**，而且就压在做相反事情的代码上方。逐条更正：
     *
     *  1. **默认开，而且赢。** 那组 910B3 的 A/B（small +40%、mid +30%）是
     *     **统一行距 + 每行一块**那一代实现上测的，读放大 1.91× 没治好，
     *     白搭上 Gather 的钱。改成打包读之后 910B4 上 9 档重测：
     *     c 档 92.73 → 32.61（2.8×）、Case5 那类 359.62 → 177.14（2.0×），
     *     其余持平，**没有一档变差**。`S9_GATHER=0` 可关掉做 A/B。
     *  2. **int8 能走。** 旧前提写「dtype 必须 16/32 位，int8 不在 Gather 的支持表里」——
     *     现在 int8 **借道 half**（`S9GatherAs`，±127 在 half 里是精确整数 ⇒ 往返恒等），
     *     默认开（`i8Gather`）。910B4 实测 (2048,4096) 180.81 → 61.40 等三档 2.9~4.6×。
     *  3. **建表没有标量路。** 旧说法「偏移表要 nOut 次标量写、这一步没能向量化」
     *     已被推翻：《Adds》《Duplicate》在 A2 上有逐 bit mask 的高维切分版，
     *     落点按 32B 对齐取 64 通道窗口、参与通道由 mask 选 ⇒ **全向量建表**
     *     （见 op_kernel 的 `GatherAssembled`）。那条 O(nOut) 标量开销不存在了。
     *  4. **`GATHER_MAX_OUT` 不是 1024，是 1<<20。** 真正的约束是"UB 放不放得下一行"，
     *     那个判据本来就在，让它自己去卡（见上面 GATHER_MAX_OUT 的说明）。
     *  5. 旧段落据此推的"要救得上三四级缓冲 + 更大 UB"也一并作废。
     *
     * **仍然成立的那一条**（外部事实，与我们的实现无关）：910B 上官方 ConcatD 走
     * TBE DSL（`concat_v2_d.py`：`api_check_support("tbe.dsl.vexp","float32")` 为真
     * ⇒ DSL 而非 TIK），其 schedule 的默认发射指令是 **`vector_concat`**，
     * 只有读对齐时才退化成 `dma_copy`
     *（`tbe/dsl/unify_schedule/vector/concat/concat_schedule.py:_calc_emit_insn`）。
     * 也就是说**官方默认走向量、DMA 才是特例** —— 这正是当初一律走 DMA 时
     * 在「行短且不对齐」形状上落后的原因。
     *
     * **为什么用 Gather**：strategy 2（DMA 拼装）要求每个 `offOut[i]·esz` 是 32 的倍数；
     * `Gather` 按**元素粒度**收集（`dst[i] = src[(base + srcOffset[i]) / sizeof(T)]`），
     * **绕开 DataCopyPad 的 32B 落点约束**，于是任意列偏移都能在 UB 里拼成整行、
     * 再一次连续写出。
     *
     * 现在的前提（不满足就退回常规路径）：① 按行切；② 路数 ≤ `ASM_MAX_IN`
     *（kernel 侧指针放栈上，上限在共享头 `S9L_ASM_MAX_IN`）；
     * ③ UB 放得下「暂存 ＋ 拼装 ＋ 两张表」（由 `GatherFitsBands` → `PlanConcatStage` 判）。 */
    // ═══ 向量拼装路（strategy = 3）═══
    // 可行性与带数在上面就算好了（要赶在 strategy 4 之前），这里只负责落选。
    //
    // **为什么要分带。** 按行切时每个核都要碰**全部** numIn 路，而且一行占的 UB
    // 是整行（nOut·esz），tile 只装得下几行 ⇒ 每 tile 发 numIn 条很小的 DataCopyPad。
    // 真机量到的就是这个形态：(4096,4096) fp16 33.5 MB，只改路数不改字节数 ——
    //     123 路 151.27 · 33 路 116.03 · 14 路 104.70 · 8 路 104.33 µs
    // 每多一路 **+0.41 µs**，123 路白花 50 µs；8 路时 104.33 已经贴住带宽底 97.6。
    //
    // 分带一次治两头：每核只碰 numIn/bands 路，且每行 UB 占用小 bands 倍
    // ⇒ tile 行数涨 bands 倍 ⇒ 每条 DMA 又大 bands 倍。
    // 代价是写从整行连续变成 blockCount=cnt、blockLen=带宽 的跨步块。
    if (gatherOk && t.strategy == 0) {
        t.strategy = 3;
        t.bands = gatherBands;
        // **这条路的 UB 不再由 host 的 ubBudget 决定** —— kernel 自己按整块 UB
        // 反推 tile（见 op_kernel 的 Init）。host 只负责判「至少放得下一行」。
    }

    /* ═══ 输出线性切分（strategy = 5）—— `rows == 1` 的兜底覆盖 ═══
     *
     * **放在最后、无条件覆盖**，因为 `rows == 1` 时上面每一条路都是坏的：
     *   · 按行切只开得出 1 个核（行只有一行）；
     *   · 按列切要求"平均一路 ≥ 24 KB"，多路形状几乎不满足 ⇒ 退回 blockDim = rows = 1；
     *   · 拼装路（2/3）以"按行切、每核拿若干整行"为前提，一行时分不出核；
     *   · 按输入切核的 blockDim 已被上面压到 1。
     * 910B4 实测：默认单核 34~71 GB/s，而**强开核数反而慢 3~4×**
     *（现有策略按"路内的列"分核，249 路 × 40 ≈ 一万条碎 DMA）。
     * 换成按输出字节切之后 **2.59×~3.21×**。
     *
     * 只覆盖 `rows == 1`：`rows > 1` 时列带分核已经跑到 692 GB/s，没有理由换。
     * ⇒ **下行风险为零**（不满足条件的形状一个字节都不变）。 */
    if (kn.linearOn && t.rows == 1 && t.nOut > 0 &&
        (uint64_t)t.nOut * esz >= LINEAR_MIN_BYTES) {
        const uint32_t alignEl = (32 / esz) ? (32 / esz) : 1;
        // 每核至少分到一个 32B 粒度，否则多出来的核只会空转
        uint32_t lin = t.nOut / alignEl;
        if (lin > coreCap) { lin = coreCap; }
        if (lin >= 2) {                       // 只有一个核可分就没必要换路
            t.strategy = 5;
            t.bands = 1;
            t.blockDim = lin;
        }
    }

    /* ═══ 二维分核（行组 × 列组）—— 只对 strategy 0，放在**所有策略选完之后** ═══
     *
     * **为什么需要它。** 按行切的 `blockDim = rows`，而 `dim` 在中间轴时
     * `rows = N4` 常常只有十几 —— 评测机 48 个 AIV 只用得上 16 个。
     * 本机扫 3700 个合法的中间轴形状，**96.6% 落在按行切上**，其中大量是
     * `blockDim = rows ≪ 48` 这种。
     *
     * **为什么不是 strategy 1。** 列切的判据是 `colCores > rows`，而 `colCores`
     * 由 `COL_BURST_BYTES = 24 KB` 反推；中间轴形状上平均一路才 31 KB/行 ⇒
     * `colCores = 1`，永远进不去。那个 24 KB 下限是为**二维窄路**标定的
     *（"短路被切碎"那组实测），不适用于这里 —— 这里一条路本身就有几十 KB。
     *
     * **为什么保留按行切而不换路。** 按行切在宽路上的数据形态本来就好：
     * 读是整块连续（srcStride=0），写是 `blockLen = 路宽·esz` 的大块。
     * 缺的只有核数。所以只补核数，搬运形态一个字节都不动。
     *
     * `gridBurstBytes = 0` ⇒ 整块跳过，退回旧行为。 */
    if (kn.gridBurstBytes && t.strategy == 0 && t.rows > 0 && t.rows < coreCap &&
        t.numIn > 0 && t.nOut > 0) {
        uint32_t parts = coreCap / t.rows;
        const uint32_t burstEl = (esz && kn.gridBurstBytes >= esz)
                                     ? (kn.gridBurstBytes / esz) : 1;
        // 基准取**平均一路**（与 strategy 1 的 colBasis 同源）：最短的一路短，
        // 不构成"整体不该切列"的理由 —— 它自己就那么点数据。
        uint32_t cap = burstEl ? ((t.nOut / t.numIn) / burstEl) : 1;
        if (cap < 1) { cap = 1; }
        if (parts > cap) { parts = cap; }
        if (parts > 1) { t.colParts = parts; t.blockDim = t.rows * parts; }
    }
    return true;
}

/* ═══ 每带执行计划（strategy 3 专用；`S9_BANDPLAN=1`）═══
 *
 * **每个核原先要先扫一遍全部 numIn 路**，才能知道"哪些路与我这条带相交"
 *（kernel 的 iLo/iHi 与 stBase 都是这么来的）。246 路那档 msprof 量到
 * `aiv_scalar_ratio 0.97` —— 这场扫描就是主因。
 *
 * 这些量**只依赖 tiling**，在 host 上算一次就行：第 b 条带相交的输入下标区间
 * `[lo, hi)`（相交的路必然连成一段：offs[] 是 nIn 的前缀和、单调不减），
 * 以及这一段起点的输出前缀 `off`。
 *
 * 三条不变式（自检里逐条验）：① 区间连续；② [lo,hi) 恰好覆盖所有与本带相交的
 * 非空路；③ 相邻带之间 lo/hi 单调不减。空带用 lo=hi=0 表示"不干活"。
 *
 * **host 与 kernel 必须同源**：kernel 按它跳过扫描，算错一格就会漏掉整条路
 *（输出留着 at::empty 的垃圾，不报错）。 */
struct ConcatBandPlan {
    uint32_t bands{0};
    uint32_t lo[BAND_MAX];
    uint32_t hi[BAND_MAX];
    uint32_t off[BAND_MAX];      // 第 b 带第一个相交输入的**输出列前缀**
};

inline bool CalcConcatBandPlan(const ConcatTiling &t, ConcatBandPlan &p)
{
    if (t.nOut == 0 || t.bands == 0 || t.bands > BAND_MAX || t.numIn > CONCAT_MAX_IN) {
        return false;
    }
    p.bands = t.bands;
    const uint32_t grainEl = t.bandGrain ? t.bandGrain : 1;
    /* **一遍扫完，O(numIn + bands)**（原先是 B×numIn：每条带都把全部路重扫一遍）。
     *
     * 与 `MaxInputsPerBand` 同一条依据：带区间单调且互不重叠、`offOut` 是前缀和
     * ⇒「与带相交」的路是一条**连续下标段**，两条带之间只右移。`i0` 只前进不回退，
     * 于是每条路最多被跨两条带。零宽路（`n == 0`）按定义不相交，跳过。 */
    uint32_t i0 = 0;
    for (uint32_t b = 0; b < t.bands; ++b) {
        uint32_t c0 = 0, c1 = 0;
        BandRange(t.nOut, t.bands, grainEl, b, c0, c1, t.bandBal);   // 与 kernel 同源
        // **零宽带谁都不相交**（kernel 的 Init 也是这么收的：`cb0_ >= cb1_ ⇒ 都置 0`）。
        // 少了这一条，[4,4) 这种带会被判成"与跨过它的那条路相交"。
        if (c0 >= c1) { p.lo[b] = 0; p.hi[b] = 0; p.off[b] = 0; continue; }
        while (i0 < t.numIn &&
               (t.nIn[i0] == 0 || t.offOut[i0] + t.nIn[i0] <= c0)) {
            ++i0;                                  // 单调推进：空片 / 整条在本带左侧
        }
        uint32_t lo = t.numIn, hi = 0;
        for (uint32_t i = i0; i < t.numIn; ++i) {
            const uint32_t n = t.nIn[i];
            if (n == 0) { continue; }                 // 空片不算相交（与 kernel 一致）
            const uint32_t o = t.offOut[i];
            if (o >= c1) { break; }                   // 整条在本带右侧 ⇒ 后面都更右
            if (lo == t.numIn) { lo = i; }
            hi = i + 1;
        }
        if (lo >= hi) { lo = 0; hi = 0; p.off[b] = 0; }   // 空带
        else          { p.off[b] = t.offOut[lo]; }
        p.lo[b] = lo;
        p.hi[b] = hi;
    }
    return true;
}

/* ═══ 发布前的最终配置校验（纯算术，可单测）═══
 *
 * `TilingFunc` 里有一堆 `S9_*` 环境开关，它们都在 `CalcConcatLens` **之后**改
 * 核数、带数、buffer 数、UB 预算。原先这些覆盖各改各的、改完直接下发：
 *   · 局部 `strategy` 是在 S9_STRAT / S9_BLK **之前**拷出来的 ⇒ 下发的是旧值；
 *   · S9_GBUF 允许 1..8，而 kernel 的事件号只到 3（EvB 只区分 0/1/2）
 *     ⇒ 第 4 块与第 1 块共用事件，同步错位；
 *   · 改核数后带数不整除 ⇒ kernel **悄悄退回 bands=1**，而 UB 账是按分带算的
 *     ⇒ 真机 vector core exception 507035（不是错值，是崩）；
 *   · UB 预算能覆盖到超过容量 ⇒ `InitBuffer` 直接踩别人的缓冲。
 *
 * 所以所有覆盖做完之后必须过这一关：**要么明确失败（带诊断），要么安全降级**。
 * 负例见 `test/test_concat_layout.cpp` 的 CheckFinalConfigValidation。 */
struct ConcatFinal {
    uint32_t strategy{0};
    uint32_t bands{1};
    uint32_t gBuf{2};
    uint32_t aBuf{1};
    bool     ok{true};
    bool     changed{false};     // 发生过收敛/降级 —— 调用方应当记一笔
    const char *why{"ok"};       // 失败原因，或改动的理由
};

/* 不超过 cap 的最大因数（n ≥ 1，cap ≥ 1）。 */
/* 不超过 cap 的最大因数（n ≥ 1，cap ≥ 1）。
 *
 * **O(√n) 而不是 O(cap) 的线性试除**：枚举 d ≤ √n 的所有因数，它和配对的
 * n/d 一起判定；两个都 ≤ cap 时取更大的那个。返回值与原写法逐位相同
 *（"n 的不超过 cap 的最大因数"），但循环次数从 cap 变成 √n。
 * cap ≤ BAND_MAX ≈ 48 时两者的绝对耗时都微不足道 —— 收这一刀是为了
 * **不让"线性试除"这种写法留在关键路径旁边当模板**（`sweep_concat.cpp`
 * 那类工具会把 tiling 跑几十万次）。
 * 单测：`test/test_s9_env.cpp` 同目录的 `test_tiling.cpp` 里对
 * `CheckFinalConfigFinal` 的用例覆盖了它；另有穷举对照见提交说明。 */
inline uint32_t LargestFactorAtMost(uint32_t n, uint32_t cap)
{
    if (n < 1 || cap < 1) { return 1; }
    if (cap > n) { cap = n; }
    uint32_t best = 1;
    for (uint32_t d = 1; (uint64_t)d * d <= (uint64_t)n; ++d) {
        if (n % d != 0) { continue; }
        const uint32_t hi = n / d;                 // 配对的那个因数（≥ d）
        if (hi <= cap && hi > best) { best = hi; }
        if (d  <= cap && d  > best) { best = d;  }
        if (best == cap) { return cap; }           // 已到上界，不可能更大
    }
    return best;
}

inline ConcatFinal CheckConcatFinal(const ConcatTiling &t, uint32_t esz, uint32_t blockDim,
                                    uint32_t bandsIn, uint32_t gBufIn, uint32_t aBufIn,
                                    uint32_t slackIn, uint32_t ubBudget,
                                    uint32_t tileGMax, uint32_t maxCores)
{
    (void)tileGMax;      // 只在 kernel 侧参与 tileG 推导，校验不需要它
    ConcatFinal f;
    f.strategy = t.strategy;
    f.bands    = bandsIn ? bandsIn : 1;
    f.gBuf     = gBufIn ? gBufIn : 2;
    f.aBuf     = aBufIn ? aBufIn : 1;
    auto fail = [&](const char *why) { f.ok = false; f.why = why; return f; };

    if (t.numIn == 0 || t.numIn > CONCAT_MAX_IN) { return fail("输入路数越界"); }
    {
        uint64_t sum = 0;
        for (uint32_t k = 0; k < t.numIn; ++k) { sum += t.nIn[k]; }
        if (sum != t.nOut) { return fail("nIn[] 之和不等于 nOut"); }
    }
    if (blockDim < 1) { return fail("核数为 0"); }
    if (maxCores && blockDim > maxCores) {
        return fail("核数超过平台查询值（用 SyncAll 的算子会死锁）");
    }
    if (slackIn > t.numIn) { return fail("余量路数超过输入路数"); }
    // **暂存/拼装块数受事件号上限约束**：kernel 的 EvB 只区分 0/1/2
    //（《SetFlag/WaitFlag》建议只用 ID0~5），更大的值会让两块共用事件。
    if (f.gBuf < 1 || f.gBuf > S9L_BUF_N) {
        f.gBuf = Clamp(f.gBuf, 1u, S9L_BUF_N);
        f.changed = true; f.why = "暂存块数超出事件号上限 ⇒ 收敛到上限";
    }
    if (f.aBuf < 1 || f.aBuf > S9L_BUF_N) {
        f.aBuf = Clamp(f.aBuf, 1u, S9L_BUF_N);
        f.changed = true; f.why = "拼装块数超出事件号上限 ⇒ 收敛到上限";
    }
    // UB 预算：常规路要 `BUF_N 块 × ubBudget`，外加序幕与指针暂存
    if (ubBudget < 32) { return fail("UB 预算小于一个 32B 块"); }
    {
        const uint64_t td = S9LAlignUp64((uint64_t)t.numIn * 8, S9L_BLOCK_BYTES)
                          + S9L_BLOCK_BYTES;
        const uint64_t need = (uint64_t)S9L_BUF_N * ubBudget + td + S9L_PRE_BYTES;
        if (need > (uint64_t)S9L_UB_BYTES - S9L_UB_MARGIN) {
            return fail("UB 预算 × 块数 ＋ 序幕/指针暂存 超过单核 UB");
        }
    }
    if (f.strategy != 3) {
        // 0/1/2/4 的前提已在 CalcConcatLens 里判过。策略号到 4 为止
        //（曾经的 5 ＝"按输入切核 + 大块"，2026-09-13 实测 4.33× 慢后整体删除）。
        if (f.strategy > 5) { return fail("策略编号越界"); }
        return f;
    }
    // ── 拼装路（strategy 3）的最终校验 ──
    if (t.numIn > ASM_MAX_IN) { return fail("拼装路的路数超过栈上指针上限"); }
    if (!(esz == 1 || esz == 2 || esz == 4)) {
        return fail("拼装路的 dtype 不支持（只有 8/16/32 位；64 位会把表的行起点弄成非 32B 对齐）");
    }
    if (t.nOut > GATHER_MAX_OUT) { return fail("拼装路的 nOut 超过上限"); }
    if (f.bands > blockDim) { return fail("带数超过核数（会有带分不到核）"); }
    if (blockDim % f.bands != 0) {
        // kernel 的守卫是"不整除就退回 1"，而 host 的 UB 账按分带算 ⇒ 两边不一致。
        // 这里显式收敛成因数，绝不把不整除的值下发。
        f.bands = LargestFactorAtMost(blockDim, f.bands);
        f.changed = true;
        f.why = "带数不整除核数（kernel 会退回 1，账就对不上）⇒ 收敛成因数";
    }
    if ((uint64_t)t.bandGrain * esz % S9L_BLOCK_BYTES != 0) {
        return fail("带粒度不是 32B 的整数倍 ⇒ 带首不保证 32B 对齐");
    }
    /* **余量路数不能低估。** 暂存区按「每路一块、块起点 32B 对齐」布局，一块带里
     * 到底挤进几路要**精确数**（各路宽度是随机的，按均分估必然偏小）。
     * host 默认按全部路数留余量（那是最宽的上界），但 S9_MAXINB 能把它调小 ——
     * 调小了这里必须拦住：真机表现是 vector core exception 507035，不是错值。 */
    {
        const uint32_t alignEl = (esz && 32 / esz) ? (32 / esz) : 1;
        const uint32_t need = MaxInputsPerBand(t.nOut, t.nIn, t.numIn, f.bands, t.bandGrain,
                                               alignEl, t.bandBal);
        const uint32_t nSlack = slackIn ? slackIn : t.numIn;
        if (nSlack < need) {
            return fail("暂存余量路数低于该带实际挤进的输入数（低估 ⇒ 真机 UB 越界）");
        }
    }
    {
        const uint32_t grp = blockDim / f.bands;
        const uint32_t myRows = grp ? CeilDiv(t.rows, grp) : t.rows;
        const uint32_t nSlack = slackIn ? slackIn : t.numIn;
        if (!GatherFitsBands(t.nOut, t.numIn, t.maxIn, esz, f.bands, t.bandGrain,
                             f.aBuf, nSlack, t.bandBal, myRows, f.gBuf, t.nIn,
                             /*trimAccount=*/t.trimOn >= 2, 0, nullptr, t.denseOn)) {
            // 前提被某个覆盖破坏 ⇒ **安全降级**到最保守的按行切（那是验证过的活路）。
            f.strategy = 0;
            f.bands    = 1;
            f.changed  = true;
            f.why      = "拼装路的 UB 实账放不下（含序幕与指针暂存）⇒ 降级按行切";
        }
    }
    return f;
}

}  // namespace s9
#endif  // S9_TILING_CALC_H
