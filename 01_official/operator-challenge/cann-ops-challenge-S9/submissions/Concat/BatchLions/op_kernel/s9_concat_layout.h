/* Concat 向量拼装路（strategy 3）的 UB 布局 —— 纯算术、不依赖任何 CANN 类型。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * ## 为什么要把这份账剥出来
 *
 * 这条路的 UB 账原先散在三处，各写一份公式：
 *   · `op_kernel/concat.cpp` 的 `Init()`  —— 真正调 `InitBuffer` 分配的地方
 *   · `s9_tiling_calc.h` 的 `GatherFitsBands` —— host 的可行性判据
 *   · `test/twin_concat.cpp` 的 `Strat3`   —— 孪生里复刻的那一份
 * 三份对不上的后果不是"慢"，是 `InitBuffer` 越界（真机 vector core
 * exception 507035）。所以这里的函数是**唯一**的来源：host 判可行性、
 * kernel 反推 tileG 并分配、孪生回放绝对地址，全走它。
 *
 * ## 这次修的那一处：手动轮转的块跨度必须逐块 32B 对齐
 *
 * 文档《InitBuffer》（8.3.RC1）：「TBuf 的 len 为总分配字节数，非对齐时补齐」——
 * 也就是说 `InitBuffer` 只保证**这一块自己**从 32B 边界开始。
 * 而拼装路开的是**一个** TBuf 装 `gBuf` 块、块间靠手动偏移轮转
 *（`stgAll[ub * stgSpanEl]`），这个手动跨度是 kernel 自己算的，
 * 框架不管 —— `stageSpan_ = tileG_*rowUse + slack` 就没对齐过。
 *
 * 可复现的反例（不是推的，本文件的自检把它钉死了）：
 *   fp16、dim=1、`(4096,63)×4 ＋ (4096,4)`、40 核、默认参数
 *   ⇒ host 选 strategy=3、bands=1、aBuf=2；tileG=51、rowUse=764、slack=160
 *   ⇒ 旧跨度 51·764+160 = **39124**，39124 % 32 = **20**
 *   ⇒ 第二块暂存起点落在 20 mod 32 上，而 DataCopyPad 的 UB 侧目的地址
 *     要求 32B 对齐（《DataCopyPad》8.1.RC1：LocalTensor 起始地址需要 32B 对齐）。
 * 这个形状**至今没在设备上验过**（本机无卡），所以它先以"合法反例"的身份
 * 停在回归里，等上板复现。
 *
 * ## 账目的口径（与 kernel 一一对应，改一处要改两处）
 *
 *   hd   = S9_PRE_BYTES                     序幕：标量头 ＋ nIn[] 整块取进 UB
 *   td   = align32(numIn*8) + 32            指针/nIn 暂存（q：不轮转，只有一块）
 *   off  = tblRow*4*tileG                   摊开 tileG 行的偏移表
 *   dlt  = tblRow*4                         每列的行步长表
 *   stg  = gBuf * stageSpan_                暂存队列（**手动轮转，跨度要对齐**）
 *   asm  = aBuf * outPitch*tileG            拼装区
 *   cast = align32(stageSpan*gsz)           int8 借道 half：暂存的副本（TBuf，一块）
 *   asmG = align32(tblRow*gsz*tileG)        int8 借道 half：拼装结果的副本（一块）
 *
 * 全部相加必须 ≤ 192 KB − 8 KB 余量。**序幕与指针暂存也算在内** ——
 * 原先 kernel 的 `UB_TOTAL` 只减了序幕的 1152 B，指针暂存那最多 2080 B
 * 是藏在 8 KB 余量里的，而 host 的可行性判据根本没算这一项。
 */
#ifndef S9_CONCAT_LAYOUT_H
#define S9_CONCAT_LAYOUT_H

#include <cstdint>

/* **设备侧调用约定。** 这个头同时被 host（`op_host/concat.cpp`）、CPU 单测与
 * kernel 包含。kernel 调用的函数必须标 `__aicore__` —— 不标的话 bisheng 会把它们
 * 当 host 函数，只编译不发射设备码，**链接时**报 `undefined symbol:
 * s9::PlanConcatStage(...)`（2026-09-11 用官方 CANN 8.5.0 编 ConcatCustom 时就
 * 是这个错，见 AscendS9/docs/plans 里的 T7 记录）。
 *
 * host 与 CPU 单测里没有 `__aicore__`，所以由**包含方**决定它的含义：
 * kernel 侧在 include 之前 `#define S9_LAYOUT_AICORE __aicore__`，
 * 其余地方不定义 ⇒ 展开成空。CANN 自己生成的入口代码用的也是同一套写法
 *（`op_project_templates/.../kernel_entry.py`：`#ifndef __aicore__ #define ...`）。 */
#ifndef S9_LAYOUT_AICORE
#define S9_LAYOUT_AICORE
#endif

namespace s9 {

/* 不可行的原因用**枚举**而不是字符串：这个结构体会进设备码，
 * 而设备侧不该出现字符串字面量。文本由 WhyText 在 host 侧翻译。 */
enum ConcatWhy : uint32_t {
    C9_OK = 0,
    C9_EMPTY,            // bw/esz/numIn/tileG 为 0
    C9_TILEG_TOO_BIG,    // tileG 超过 blockCount 上限
    C9_TABLE_UNALIGNED,  // 偏移表的行起点无法 32B 对齐（8 字节 dtype）
    C9_OVERFLOW,         // 某个布局量溢出 uint32
    C9_UB_ACCOUNT,       // 实账（含序幕与指针暂存）超过 UB 上限
    C9_ROW_TOO_WIDE,     // 连 1 行都放不下
};

inline const char *WhyText(uint32_t why)
{
    switch (why) {
        case C9_OK:            return "ok";
        case C9_EMPTY:         return "空配置";
        case C9_TILEG_TOO_BIG: return "tileG 超过 blockCount 上限";
        case C9_TABLE_UNALIGNED:
            return "偏移表行起点无法 32B 对齐（8 字节 dtype 未支持）";
        case C9_OVERFLOW:      return "布局量溢出 uint32";
        case C9_UB_ACCOUNT:    return "实账超过 UB 上限（含序幕与指针暂存）";
        default:               return "1 行也放不下（含序幕与指针暂存）";
    }
}

/* ── 与 kernel 侧的常量必须一致（改一处要改两处）── */
constexpr uint32_t S9L_BLOCK_BYTES = 32;    // DataCopyPad / InitBuffer 的硬对齐
constexpr uint32_t S9L_BUF_N       = 3;     // kernel 的 BUF_N：拼装路缓冲块数上限
/* **拼装路的路数上限 —— 唯一来源。**
 *
 * host 用它决定「这个形状走不走拼装路」（`numIn <= ASM_MAX_IN`），
 * kernel 用它开栈数组（`srcs/nIns/offs/stBase/aN/aW/aShift`，每个 S9_ASM_MAX 项）。
 * 两边原先各写一个 256，只靠一句注释「与 s9::ASM_MAX_IN 一致」维系 ——
 * **一旦有人只改一边，host 就会把 numIn > 数组长度的形状放进拼装路，
 * kernel 在栈上越界写**（真机 vector core exception 507035，编译期毫无察觉）。
 * 仓库对 `CONCAT_MAX_IN` 与 `UB_TOTAL` 都钉过同一类口径，唯独漏了这一对。
 * 放进共享头之后两边都从它派生，**分叉在语法上就不可能**。 */
constexpr uint32_t S9L_ASM_MAX_IN  = 256;
constexpr uint32_t S9L_HDR_BYTES   = 128;   // 标量头整块取时的字节数
constexpr uint32_t S9L_PRE_IN      = 256;   // 与 CONCAT_MAX_IN 一致
constexpr uint32_t S9L_PRE_BYTES   = S9L_HDR_BYTES + S9L_PRE_IN * 4;    // 1152
constexpr uint32_t S9L_UB_BYTES    = 192 * 1024;   // 910B 单核 UB（架构规格）
constexpr uint32_t S9L_UB_MARGIN   = 8 * 1024;     // 系统/框架占用的余量
constexpr uint32_t S9L_TILEG_HARD_MAX = 4095;      // DataCopyPad blockCount 上限

/* 与 kernel 的 `UB_TOTAL` 同口径：扣掉余量与序幕，剩下的才是拼装路能用的。
 * 指针暂存（最多 2080 B）不在这个数里扣，而是在实账里逐项累加。 */
constexpr uint32_t S9L_UB_TOTAL = S9L_UB_BYTES - S9L_UB_MARGIN - S9L_PRE_BYTES;

/* `tblK` 模式下偏移表除 K 行工作区之外还要的行数：母版、cur（推进到本 chunk 的
 * 起始行）、dltK（＝dlt·K）。**kernel 与账必须用同一个常量** —— 差一行就是
 * 工作区落在缓冲外，真机 UB 越界且不一定当场炸。 */
static const uint32_t S9L_TBL_EXTRA_ROWS = 3;

S9_LAYOUT_AICORE inline uint64_t S9LCeilDiv64(uint64_t a, uint64_t b)
{
    return b ? (a + b - 1) / b : 0;
}
S9_LAYOUT_AICORE inline uint64_t S9LAlignUp64(uint64_t a, uint64_t b)
{
    return S9LCeilDiv64(a, b) * b;
}

/* ═══ 编译期策略特化：**试过、量过、撤掉了** ═══
 *
 * 曾经的编码是 `key = 100 + strategy·10 + log2(esz)`，host 由 `S9_SPECIALIZE=1`
 * 下发、kernel 用 `if constexpr` 只实例化那一条路。2026-09-11 上板（910B4 +
 * CANN 8.3.RC1，同一份包交替配对）：
 *   · int8 的 2048×2048 那档能跑，B/A = 1.009 —— **没有收益**；
 *   · fp16 的 128×256 与 512×4096 **跑不出结果**：`RuntimeError: ... current working
 *     operator name is aclnnConcat`（设备侧内部错误，机理未查，怀疑 if constexpr 之后
 *     helper 被内联、设备侧栈帧变大）。
 * ⇒ 没有收益 + 会崩，两条都不占理，整条实验撤掉。键回到元素字节宽度（1/2/4/8）与
 * 混合核的 +16；kernel 侧回到运行时分支。
 *
 * 留这段记录的用处：**下次有人想"把策略编进 key"时，先看这里** —— 官方文档确实支持
 * TilingKey 常量折叠（《多分支策略》），但那说的是"分支能被折叠"，不等于"折叠之后更快"。 */

/* ═══ 核 → (带, 组, 列区间, 行区间) 的映射 —— **host 与 kernel 唯一的一份** ═══
 *
 * 这段算术原先只写在 `op_kernel/concat.cpp` 的 `Init()` 里。布局下沉（host 预先
 * 算好每个核的布局再下发）要求 host 也能算出**一模一样**的映射 —— 两边各写一份
 * 就会变成"host 按 A 核的账算、kernel 跑 B 核的配置"，那不是慢，是互相写对方
 * 的地盘（结果错且不报错）。所以剥到这里，两边都调它。
 *
 * 语义（与 kernel 原实现逐字对齐）：
 *   · `bands` 必须整除核数，否则**退回 1**（网格要填满）；
 *   · 核按「带内连续」编号：前 `bn % bands` 条带各多一组；
 *   · 列带按 `bandBal` 二分：0 ＝ 统一带宽＋末带吃剩下（旧行为），1 ＝ 均衡切分；
 *   · 行按组切，余数分给前几组；
 *   · 空带（`cb0 ≥ cb1`）不干活，但 `bw` 要留一个对齐宽度让缓冲合法。
 * `MapConcatCore` 只回答"我是谁、我管哪一块"，**不放不放得下**（那是
 * PlanConcatStage / GatherFitsBands 的事）。 */
struct ConcatCoreMap {
    uint32_t bands{1}, bandIdx{0}, grpIdx{0}, grp{1};
    uint32_t bandCols{0}, cb0{0}, cb1{0}, bw{0};
    uint32_t rowBeg{0}, rowEnd{0};
    uint32_t alignEl{1}, myRows{1};
    bool     empty{false};
};

S9_LAYOUT_AICORE inline ConcatCoreMap MapConcatCore(uint32_t rows, uint32_t nOut,
                                                   uint32_t blockDim, uint32_t bandsIn,
                                                   uint32_t grainElIn, uint32_t bandBal,
                                                   uint32_t blockIdx, uint32_t alignElIn)
{
    ConcatCoreMap m;
    m.alignEl = alignElIn ? alignElIn : 1;
    const uint32_t bn = blockDim ? blockDim : 1;
    m.bands = (bandsIn >= 1 && bandsIn <= bn && bn % bandsIn == 0) ? bandsIn : 1;

    const uint32_t gBase = bn / m.bands;
    const uint32_t gExtra = bn % m.bands;
    uint32_t start = 0;
    for (uint32_t b = 0; b < m.bands; ++b) {
        const uint32_t gcnt = gBase + (b < gExtra ? 1u : 0u);
        if (blockIdx < start + gcnt) { m.bandIdx = b; m.grpIdx = blockIdx - start; m.grp = gcnt; break; }
        start += gcnt;
    }

    const uint32_t grain = grainElIn ? grainElIn : m.alignEl;
    if (bandBal == 0) {
        m.bandCols = (m.bands == 1)
                         ? nOut
                         : (uint32_t)S9LAlignUp64(S9LCeilDiv64(nOut, m.bands), grain);
        m.cb0 = m.bandIdx * m.bandCols;
        m.cb1 = (m.cb0 + m.bandCols < nOut) ? (m.cb0 + m.bandCols) : nOut;
        if (m.cb0 > nOut) { m.cb0 = nOut; }
    } else if (m.bands == 1) {
        m.bandCols = nOut;
        m.cb0 = 0;
        m.cb1 = nOut;
    } else {
        const uint32_t G = (uint32_t)S9LCeilDiv64(nOut, grain);
        m.bandCols = (uint32_t)(S9LCeilDiv64(G, m.bands) * grain);
        const uint32_t g0 = (uint32_t)((uint64_t)m.bandIdx * G / m.bands);
        const uint32_t g1 = (uint32_t)((uint64_t)(m.bandIdx + 1) * G / m.bands);
        m.cb0 = g0 * grain;
        m.cb1 = g1 * grain;
        if (m.cb0 > nOut) { m.cb0 = nOut; }
        if (m.cb1 > nOut) { m.cb1 = nOut; }
    }
    if (m.cb0 >= m.cb1) { m.cb0 = 0; m.cb1 = 0; }
    m.bw = m.cb1 - m.cb0;
    {
        const uint32_t q = rows / m.grp, rr = rows % m.grp;
        m.rowBeg = q * m.grpIdx + (m.grpIdx < rr ? m.grpIdx : rr);
        m.rowEnd = m.rowBeg + q + (m.grpIdx < rr ? 1u : 0u);
    }
    if (m.bw == 0) { m.empty = true; m.rowBeg = 0; m.rowEnd = 0; m.bw = m.alignEl; }
    m.myRows = (m.rowEnd > m.rowBeg) ? (m.rowEnd - m.rowBeg) : 1;
    return m;
}

/* ═══ 逐核布局表 —— 布局下沉（`S9_LAYOUT=1`）下发的那张表 ═══
 *
 * 一个核一行，字段就是 kernel 的 `Init()` 从 `ConcatStageLayout` 里取的每一个量。
 * `MapConcatCore` → `PlanConcatStage` 都由 host 先跑一遍，kernel 只读自己那一行。
 *
 * **首版必须与 kernel 现算的完全一致**：所以这里的字段与顺序都照
 * `ConcatStageLayout` 来，不多不少；`tileG` 用的是本核**实际行数**夹过的那个值。
 * 校验见 `test_concat_layout.cpp` 的 `CoreLayoutTableMatchesKernel`。 */
struct ConcatCoreLayoutRow {
    /* 字段顺序**固定**（host 按它平铺进 TilingData，kernel 按同一个序号读）。
     * 前四项就是 `MapConcatCore` 的答案，后十项是 `PlanConcatStage` 的。 */
    uint32_t rowBeg{0}, rowEnd{0};
    uint32_t bw{0}, cb0{0}, cb1{0};
    uint32_t outPitch{0}, tblRow{0}, tileG{0};
    uint32_t stageSpan{0}, asmSpan{0};
    uint32_t offTblBytes{0}, dltBytes{0}, castSpan{0}, asmGBytes{0};
};
constexpr uint32_t S9L_CORE_FIELDS = 14;    // ConcatCoreLayoutRow 的 uint32 个数
constexpr uint32_t S9L_CORE_MAX    = 48;    // 单核布局表的行数上限（AIV 数上限）

/* 平铺进 TilingData 时的字段序号。**host 写、kernel 读，只能用这一份** ——
 * 两边各写一串数字就会错位，而错位的表现是"布局诡异"（慢）甚至越界。 */
enum ConcatCoreField : uint32_t {
    C9F_ROW_BEG = 0, C9F_ROW_END, C9F_BW, C9F_CB0, C9F_CB1, C9F_OUT_PITCH, C9F_TBL_ROW,
    C9F_TILE_G, C9F_STAGE_SPAN, C9F_ASM_SPAN, C9F_OFF_TBL, C9F_DLT, C9F_CAST, C9F_ASM_G,
};

/* 把一行布局摊成 14 个 uint32（顺序即 ConcatCoreField）。 */
S9_LAYOUT_AICORE inline void ConcatCoreLayoutToFlat(const ConcatCoreLayoutRow &r,
                                                    uint32_t *out)
{
    out[C9F_ROW_BEG]     = r.rowBeg;
    out[C9F_ROW_END]     = r.rowEnd;
    out[C9F_BW]          = r.bw;
    out[C9F_CB0]         = r.cb0;
    out[C9F_CB1]         = r.cb1;
    out[C9F_OUT_PITCH]   = r.outPitch;
    out[C9F_TBL_ROW]     = r.tblRow;
    out[C9F_TILE_G]      = r.tileG;
    out[C9F_STAGE_SPAN]  = r.stageSpan;
    out[C9F_ASM_SPAN]    = r.asmSpan;
    out[C9F_OFF_TBL]     = r.offTblBytes;
    out[C9F_DLT]         = r.dltBytes;
    out[C9F_CAST]        = r.castSpan;
    out[C9F_ASM_G]       = r.asmGBytes;
}

/* 一次布局的完整描述。**kernel 的每一个 InitBuffer 形参都从这里取**。 */
struct ConcatStageLayout {
    // 入参（回放用）
    uint32_t bw{0}, esz{1}, numIn{0}, maxNIn{0}, nSlack{0}, gBuf{2}, aBuf{1};
    // 行/表
    uint32_t tileG{0};
    uint32_t outPitch{0};      // 拼装区一行的字节数（32B 对齐）
    uint32_t tblRow{0};        // 拼装区一行的元素数（含补白）
    uint32_t extra{0};         // 边界路整行入暂存的额外字节（2·maxNIn·esz）
    uint32_t rowUse{0};        // 暂存区一行的字节数
    uint32_t slack{0};         // 每路 32B 余量的总和
    uint32_t stageSpan{0};     // **逐块对齐后**的暂存跨度（字节）
    uint32_t asmSpan{0};       // 一块拼装区的字节数（outPitch·tileG）
    uint32_t offTblBytes{0};   // 摊开 tileG 行的偏移表
    uint32_t dltBytes{0};      // 每列的行步长表
    uint32_t castSpan{0};      // int8：暂存的 half 副本
    uint32_t asmGBytes{0};     // int8：拼装结果的 half 副本
    uint32_t tdBytes{0};       // 指针/nIn 暂存（不轮转）
    uint32_t hdBytes{0};       // 序幕
    uint64_t stgBytes{0};      // gBuf · stageSpan
    uint64_t asmBytes{0};      // aBuf · asmSpan
    uint64_t totalBytes{0};    // 全部相加（含序幕与指针暂存）
    uint64_t limitBytes{0};    // 上限 = S9L_UB_BYTES − S9L_UB_MARGIN
    bool     fits{false};
    // 放不下时"把 tileG 缩小"是否有意义。结构性不可行（dtype/溢出/空配置）
    // 缩多少行都一样，PlanConcatStage 不该白缩一圈把原因覆盖掉。
    bool     retrySmaller{true};
    // PlanConcatStage 里**回缩之前**按预算反推出的 tileG（0 ＝ 没走 Plan）。
    // 只用于诊断/统计：新账（含序幕、指针暂存与逐块对齐）比旧口径严多少。
    uint32_t tileGBudget{0};
    /* 密排超行（`dense`）：`superG` 行拼成一个 32B 对齐的超行，拼装区与偏移表都按超行密排、
     * 行与行之间不留补白。1 ＝ 旧布局（每行按 32B 对齐占一格）。
     * 此时 `outPitch` 是**真实行宽**（`bw·esz`，不一定对齐）、`tblRow` 是**一个超行**的表项数
     *（`superG·bw`）—— 两个字段的含义随它变，读的一方必须先看它。 */
    uint32_t superG{1};
    uint32_t why{C9_OK};       // 不可行时的原因（枚举，见 WhyText）

    /* 第 b 块暂存的**绝对起点**（字节，相对这条路拿到的整块 UB）。
     * 孪生与 kernel 都按它算地址 —— 对齐了才有意义。 */
    S9_LAYOUT_AICORE uint64_t StgBase(uint32_t b) const { return (uint64_t)b * stageSpan; }
    S9_LAYOUT_AICORE uint64_t AsmBase(uint32_t b) const { return (uint64_t)b * asmSpan; }
    /* 第 b 块暂存是否整个落在自己那一块里（越界检查用）。 */
    S9_LAYOUT_AICORE bool StgBlockOk(uint32_t b) const
    {
        return stageSpan != 0 && (uint64_t)(b + 1) * stageSpan <= stgBytes;
    }
};

/* ═══ 给定 tileG 的逐块布局 ═══
 *
 * 全部用 uint64 中间量算，任何一项超出 uint32（它们都要传进 `InitBuffer` 的
 * uint32 形参）就判不可行 —— **宁可拒绝，也不给静默截断的结果**。 */
/* ═══ 边界路裁剪：一条带里逐路的**入暂存行宽** ═══
 *
 * 「一条带里与它相交的路一律整行入暂存」这条规矩会让**被带切开的那 ≤2 条路**
 * 把带外的列也搬进来（读放大）。裁剪就是让这两路只搬相交的那一段：
 *
 *     lo = max(off_i, cb0)   hi = min(off_i + n_i, cb1)   W = hi − lo
 *     行宽（字节） = align32(W·esz)      ← 32B 是 DataCopyPad 的 UB 侧硬要求
 *     整段在带内的路仍然按整路走：n_i·esz
 *
 * **host 的 UB 账与 kernel 的暂存前缀和都必须用这一份** —— 两边算出的行宽不同，
 * 就是"账算得下、真机放不下"（vector core exception 507035）。
 * `trim = false` 退化成"全部按整路"，即裁剪之前的口径。 */
/* **该不该裁这一路：只有"裁完更省"才裁。**
 *
 * 紧凑槽的行宽是 `align32(W·esz)`，而整行是 `n·esz`。窄路会**反过来变大**：
 * 33 个 fp16 ＝ 66 B，补到 32B 之后是 96 B —— 每行多占 30 B。
 * 于是"裁剪后的暂存 ≤ 裁剪前"这条**不是自动成立**的，必须在这里挡一道：
 * 只有 `align32(W·esz) < n·esz` 才裁；不裁就一个字都不动（读整行、行距 n）。
 *
 * 这条护栏还顺带保住了一件事：**模式 1（UB 账不变）永远安全** ——
 * 逐路行宽只会变小，旧的账自然还是上界。（孪生里的 `TrimCompare` 就是先踩到
 * 123 路等宽那档越界才发现这一点的。） */
/* **把一路 [off, off+n) 夹到带 [cb0, cb1) 上，返回夹完还剩几列**（0 ＝ 不相交）。
 *
 * 这一段算术原先在下面**三个函数里各抄了一遍**（`ConcatRouteTrimmed` 的裁剪判据、
 * `ConcatBandStagedBytes` 的字节账、`ConcatRoutePitchEl` 的重排行距），
 * 三处必须**逐位一致** —— 不一致就是"host 说放得下、真机 507035"
 *（`s9_concat_layout.h` 的文件头就是为这件事写的）。收成一份口径之后，
 * "夹"的定义只有一个地方，另外两处不可能漂。
 *
 * 顺带一个恒等式，下面三处都在用：**`w == n` ⟺ 这一路整段落在带内**
 *（`n > 0` 时）。所以"整段在带内就没什么可裁"那个判断不必再写一遍夹取。 */
S9_LAYOUT_AICORE inline uint32_t ConcatBandClipWidth(uint32_t n, uint32_t off,
                                                     uint32_t cb0, uint32_t cb1)
{
    const uint32_t lo = (off > cb0) ? off : cb0;
    uint32_t hi = off + n;
    if (hi > cb1) { hi = cb1; }
    return (hi > lo) ? (hi - lo) : 0;
}

S9_LAYOUT_AICORE inline bool ConcatRouteTrimmed(uint32_t esz, uint32_t n, uint32_t off,
                                                uint32_t cb0, uint32_t cb1)
{
    if (n == 0 || esz == 0) { return false; }
    const uint32_t w = ConcatBandClipWidth(n, off, cb0, cb1);
    if (w == 0 || w == n) { return false; }   // 不相交 / 整段在带内 ⇒ 没什么可裁
    return S9LAlignUp64((uint64_t)w * esz, S9L_BLOCK_BYTES) < (uint64_t)n * esz;
}

S9_LAYOUT_AICORE inline uint64_t ConcatBandStagedBytes(uint32_t esz, const uint32_t *nIn,
                                                       uint32_t numIn, uint32_t cb0, uint32_t cb1,
                                                       bool trim)
{
    uint64_t sum = 0, off = 0;
    for (uint32_t i = 0; i < numIn; ++i) {
        const uint32_t n = nIn[i];
        const uint32_t o = (uint32_t)off;                  // offOut[i] ＝ nIn 的前缀和
        off += n;
        if (n == 0) { continue; }
        if (o >= cb1 || o + n <= cb0) { continue; }        // 与本带不相交
        if (!trim || !ConcatRouteTrimmed(esz, n, o, cb0, cb1)) {
            sum += (uint64_t)n * esz;
            continue;
        }
        sum += S9LAlignUp64((uint64_t)ConcatBandClipWidth(n, o, cb0, cb1) * esz,
                            S9L_BLOCK_BYTES);
    }
    return sum;
}

/* 单路的重排后**行距**（元素）。裁剪时被切开的那些路只留 W 列、并按 32B 补齐
 * —— 这是让 `DataCopyPad(blockCount=cnt, blockLen=W·esz, dstStride=0)` 每行
 * 落在 32B 边界上的充要条件（GMToUB 文档：dstStride 的单位是 32B dataBlock，
 * 且 UB 侧目的地址必须 32B 对齐）。
 * 整段在带内的路返回 n（一个字都不变），所以不裁剪时这个函数是恒等映射。 */
S9_LAYOUT_AICORE inline uint32_t ConcatRoutePitchEl(uint32_t esz, uint32_t n, uint32_t off,
                                                    uint32_t cb0, uint32_t cb1, bool trim)
{
    if (!trim || !ConcatRouteTrimmed(esz, n, off, cb0, cb1)) { return n; }
    const uint32_t w = ConcatBandClipWidth(n, off, cb0, cb1);
    if (w == 0) { return n; }
    return (uint32_t)(S9LAlignUp64((uint64_t)w * esz, S9L_BLOCK_BYTES) / esz);
}

/* ═══ 一条带的**行账** ═══
 *
 * 这 7 个量只依赖 (bw, esz, maxNIn, nSlack, trimRowBytes)，与 tileG 无关。
 * 而它们原先在 `LayoutConcatStage`（按给定 tileG 排账）与 `PlanConcatStage`
 * （反推 tileG）里**各抄了一遍**。抄漂的后果不是编译错，是
 * **"反推时以为放得下、排账时放不下"** —— tileG 被 `while (!L.fits)` 一格一格
 * 退回去，最坏塌到 1，而账面上一切正常。收成一份之后两边不可能再漂。 */
struct ConcatRowBudget {
    uint64_t gsz{0};        // Gather 载体宽度（int8 借道 half ⇒ 2）
    uint64_t outPitch{0};   // 输出行距（32B 对齐）
    uint64_t tblRow{0};     // 偏移表一行的表项数
    uint64_t extra{0};      // 边界路整行入暂存的额外字节（2·maxNIn·esz）
    uint64_t rowUse{0};     // 暂存区一行占的字节
    uint64_t slack{0};      // 逐路 32B 补齐的余量
    bool     kI8{false};
};

S9_LAYOUT_AICORE inline ConcatRowBudget ConcatRowBudgetOf(uint32_t bw, uint32_t esz,
                                                          uint32_t maxNIn, uint32_t nSlack,
                                                          uint64_t trimRowBytes)
{
    ConcatRowBudget r;
    r.gsz      = (esz == 1) ? 2ull : (uint64_t)esz;
    r.kI8      = (esz == 1);
    r.outPitch = S9LAlignUp64((uint64_t)bw * esz, S9L_BLOCK_BYTES);
    r.tblRow   = esz ? (r.outPitch / esz) : 0;
    r.extra    = 2ull * maxNIn * esz;
    // 裁剪时逐路行宽之和（由 ConcatBandStagedBytes 给出）就是这一带要搬的字节数，
    // "最宽一路 ×2"的那笔余量不再需要 —— 边界路只搬相交的那一段了。
    r.rowUse   = trimRowBytes ? trimRowBytes : ((uint64_t)bw * esz + r.extra);
    r.slack    = (uint64_t)nSlack * S9L_BLOCK_BYTES;
    return r;
}

/* ═══ 密排超行的行数：**唯一来源**（host 账、kernel 建表、孪生回放都调它）═══
 *
 * 行宽 `bw·esz` 不是 32 的倍数时，旧布局让每行在 UB 里按 32B 对齐占一格，
 * 写出就成了「每行一块」—— 910B1 实测**每块约 40 ns、与字节数无关**
 *（`2,297,9612,1 int32 dim=3`：21.8 MB、每行 4 B，4988 µs 里 mte3 占 98%）。
 * 取最小的 2 的幂 `g` 使 `g·bw·esz` 是 32 的倍数（`g ≤ 32`），g 行拼一个超行，
 * 超行内部与超行之间都不留补白 ⇒ 拼装区整段连续。
 * `g·bw·4` 也必是 32 的倍数（esz ≤ 4），偏移表按超行推进时向量 `Add` 的落点合法。
 * `tblK` 分块模式不支持密排（返回 1）。 */
S9_LAYOUT_AICORE inline uint32_t S9LDenseG(uint32_t bw, uint32_t esz, bool dense, uint32_t tblK)
{
    if (!dense || tblK != 0 || bw == 0 || esz == 0 || esz > 4) { return 1; }
    const uint64_t v = (uint64_t)bw * esz;
    uint32_t g = 1;
    while (g < S9L_BLOCK_BYTES && ((v * g) % S9L_BLOCK_BYTES) != 0) { g <<= 1; }
    return g;
}

/* 密排建超行表的工作量随 superG*numIn 增长。短行组、多路时可能得不偿失：
 * 910B1 覆盖集初测：211B/97路/每核72行，21us→92us；大行组则显著受益。
 * mode=1 用行数覆盖建表规模作为保守门槛；mode=2 强制实验，0关闭。
 * 这是依据A/B设置的选路规则，不是硬件指令耗时公式。 */
S9_LAYOUT_AICORE inline bool S9LUseDense(uint32_t bw, uint32_t esz, uint32_t numIn,
                                       uint32_t myRows, uint32_t mode, uint32_t tblK)
{
    const uint32_t g = S9LDenseG(bw, esz, mode != 0, tblK);
    return g > 1 && (mode == 2 || myRows == 0 || (uint64_t)myRows >= (uint64_t)g * numIn);
}

S9_LAYOUT_AICORE inline ConcatStageLayout LayoutConcatStage(uint32_t bw, uint32_t esz, uint32_t numIn,
                                           uint32_t maxNIn, uint32_t nSlack,
                                           uint32_t gBuf, uint32_t aBuf, uint32_t tileG,
                                           uint32_t tblK, bool dense,
                                           uint64_t trimRowBytes = 0)
{
    ConcatStageLayout L;
    L.bw = bw; L.esz = esz; L.numIn = numIn; L.maxNIn = maxNIn; L.nSlack = nSlack;
    L.gBuf = (gBuf >= 1 && gBuf <= S9L_BUF_N) ? gBuf : 2;
    L.aBuf = (aBuf >= 1 && aBuf <= S9L_BUF_N) ? aBuf : 1;
    L.tileG = tileG;
    L.limitBytes = (uint64_t)S9L_UB_BYTES - S9L_UB_MARGIN;
    if (esz == 0 || bw == 0 || numIn == 0 || tileG == 0) {
        L.why = C9_EMPTY; L.retrySmaller = false; return L;
    }
    if (tileG > S9L_TILEG_HARD_MAX) {
        L.why = C9_TILEG_TOO_BIG; L.retrySmaller = false; return L;
    }

    // 行账只有一处来源（ConcatRowBudgetOf）；下面几行是取别名，不是重算。
    const ConcatRowBudget rb = ConcatRowBudgetOf(bw, esz, maxNIn, nSlack, trimRowBytes);
    const uint64_t gsz = rb.gsz, outPitch = rb.outPitch, tblRow = rb.tblRow;
    const uint64_t extra = rb.extra, rowUse = rb.rowUse, slack = rb.slack;
    const bool     kI8 = rb.kI8;
    /* **两张表的落点也要 32B 对齐。** 偏移表是**按行推进**用的
     *（`off[j·tblRow]`，见 kernel 的"把第 0 行推成 tileG 行"），
     * 行起点是 `j·tblRow·4` 字节，而向量指令的目的地址要求 32B 对齐。
     * `tblRow = outPitch/esz`、outPitch 又是 32 的倍数 ⇒
     *   esz=1：tblRow·4 = 4·outPitch ≡ 0    esz=2：= 2·outPitch ≡ 0
     *   esz=4：= outPitch ≡ 0               **esz=8：= outPitch/2 —— 不保证**
     * 8 字节 dtype 本算子并没有注册（OpDef 只声明 fp16/fp32/int32/int8，
     * `DtypeBytes` 里那条 DT_DOUBLE/DT_INT64 ⇒ 8 是死代码），所以这里
     * **显式拒绝**：真要用 int64，得先把表改成对齐的行距（tblRow 向上取到 8 的倍数），
     * 那是另一处改动，不能靠"反正现在到不了"蒙过去。 */
    const uint32_t superG = S9LDenseG(bw, esz, dense, tblK);
    const bool     dn     = (superG > 1);
    // 密排：拼装区一行 = 真实行宽；表按超行（superG·bw 项）推进
    const uint64_t rowPitchD = dn ? (uint64_t)bw * esz : outPitch;
    const uint64_t tblRowD   = dn ? (uint64_t)superG * bw : tblRow;
    if ((tblRowD * 4) % S9L_BLOCK_BYTES != 0) {
        L.why = C9_TABLE_UNALIGNED;
        L.retrySmaller = false;
        return L;
    }
    // **逐块对齐**：手动轮转的跨度只能自己管，框架只保证每块自己的起点。
    const uint64_t stageSpan = S9LAlignUp64((uint64_t)tileG * rowUse + slack, S9L_BLOCK_BYTES);
    const uint64_t asmSpan   = dn ? S9LAlignUp64(rowPitchD * tileG, S9L_BLOCK_BYTES)
                                  : outPitch * tileG;
    /* ═══ 偏移表：`tblK` 决定它是 `tileG` 行还是 **K + 3 行** ═══
     *
     * 表是每个输出元素一个 int32 —— f16 那档数据本身才 2 字节，**表比数据还大**。
     * 实测（4096×16384 f16、bw≈1638、tileG≈9）：表 ~59 KB，与暂存区（~59 KB）
     * 一样大，占满 192 KB UB 的三分之一。这也是为什么**每一个旋钮都变差**：
     * UB 是满的，bands/gBuf/aBuf/ubBudget 想多要一点都得从 tileG 里扣。
     *
     * `tblK != 0` 时 kernel 只留 **K + 3 行**：母版、cur、dltK、以及 K 行工作区（见
     * `GatherAssembled` 的说明）。每个 chunk 铺 K 行、发**一条**覆盖 K 行的 `Gather`，
     * chunk 之间 `cur += dlt·K`。表的行数与 `tileG` **无关**。
     *
     * **K 是折中出来的，不是拍的**：K=1 在 `4096×16384`（bw≈1638）上快 2.4%，
     * 但在 `4096×4096`（bw≈1024、kernel 只有 100 µs）上慢 **17.3%** ——
     * per-row `Gather` 的**每条调用固定开销**在短 kernel 上权重高 4 倍。
     * K 把调用数从 `cnt` 降到 `cnt/K`，同时表仍是 `O(K)`。
     *
     * **这个参数是必填的，故意不给默认值** —— host、kernel、孪生三方必须算出
     * 同一个 tileG，漏传一处就是"账对不上、真机 UB 越界且不报错"。
     * 加默认值等于把这个错留给运行期。 */
    /* **不比摊开更省就不用。** `tileG` 本来就只有 3 行时，K=4 的表（K+3=7 行）
     * 反而更大，tileG 会被它挤小 —— 自检里 `rows=512 esz=2` 那档正是这么被抓到的。
     * `min` 让这件事在账里自动挡住，**kernel 用同一个判据**（见 `GatherAssembled`）。 */
    const uint64_t tblRows   = (tblK && (uint64_t)(tblK + S9L_TBL_EXTRA_ROWS) < tileG)
                             ? (uint64_t)(tblK + S9L_TBL_EXTRA_ROWS) : (uint64_t)tileG;
    const uint64_t superRows = dn ? ((uint64_t)tileG + superG - 1) / superG : tblRows;
    const uint64_t offTbl    = tblRowD * 4 * superRows;
    const uint64_t dlt       = tblRowD * 4;
    const uint64_t castSpan  = kI8 ? S9LAlignUp64(stageSpan * gsz, S9L_BLOCK_BYTES) : 0;
    const uint64_t asmG      = kI8 ? S9LAlignUp64((dn ? (uint64_t)bw : tblRow) * gsz * tileG,
                                              S9L_BLOCK_BYTES) : 0;
    const uint64_t tdBytes   = S9LAlignUp64((uint64_t)numIn * 8, S9L_BLOCK_BYTES)
                             + S9L_BLOCK_BYTES;
    const uint64_t hdBytes   = S9L_PRE_BYTES;
    const uint64_t stgBytes  = (uint64_t)L.gBuf * stageSpan;
    const uint64_t asmBytes  = (uint64_t)L.aBuf * asmSpan;
    const uint64_t total     = hdBytes + tdBytes + offTbl + dlt + stgBytes + asmBytes
                             + castSpan + asmG;

    /* 溢出：每一项都要塞进 uint32 形参，total 之外还要单独查 ——
     * total 装得下不代表每一项都装得下。 */
    if (stageSpan > 0xFFFFFFFFull || asmSpan > 0xFFFFFFFFull ||
        offTbl > 0xFFFFFFFFull || dlt > 0xFFFFFFFFull ||
        castSpan > 0xFFFFFFFFull || asmG > 0xFFFFFFFFull ||
        tdBytes > 0xFFFFFFFFull || stgBytes > 0xFFFFFFFFull ||
        asmBytes > 0xFFFFFFFFull || total > 0xFFFFFFFFull) {
        L.why = C9_OVERFLOW;
        L.retrySmaller = false;
        return L;
    }

    L.outPitch    = (uint32_t)rowPitchD;
    L.tblRow      = (uint32_t)tblRowD;
    L.superG      = superG;
    L.extra       = (uint32_t)extra;
    L.rowUse      = (uint32_t)rowUse;
    L.slack       = (uint32_t)slack;
    L.stageSpan   = (uint32_t)stageSpan;
    L.asmSpan     = (uint32_t)asmSpan;
    L.offTblBytes = (uint32_t)offTbl;
    L.dltBytes    = (uint32_t)dlt;
    L.castSpan    = (uint32_t)castSpan;
    L.asmGBytes   = (uint32_t)asmG;
    L.tdBytes     = (uint32_t)tdBytes;
    L.hdBytes     = (uint32_t)hdBytes;
    L.stgBytes    = stgBytes;
    L.asmBytes    = asmBytes;
    L.totalBytes  = total;
    L.fits        = (total <= L.limitBytes);
    if (!L.fits) { L.why = C9_UB_ACCOUNT; }
    return L;
}

/* ═══ 反推 tileG：与 kernel 的顺序一字不差 ═══
 *
 * ① 按**未对齐**的 perTile/fixed 从预算反推（这一步故意保持原样：
 *    改口径会让每个形状的 tileG 都动一点，而 tileG 有实测最优点，
 *    要动它得单独上板扫，不能混在一次"修对齐"里）；
 * ② 夹 blockCount 上限与 host 下发的`tileGMax`（**字节**上限，见它的实测说明）；
 * ③ 夹本核实际行数（`myRows`，0 ＝ 不限）—— 表是摊开 tileG 行的，
 *    行数比它少时那些展开全是白做；
 * ④ 用**对齐后的实账**复核，超了就往回缩一格，直到放得下或缩到 1 行。
 *
 * `myRows = 0` 会让 tileG 取到接近 4095 —— 那只适合问"最少一行放不放得下"，
 * host 判可行性时应当传真实的行数（见 op_host 的调用点）。 */
S9_LAYOUT_AICORE inline ConcatStageLayout PlanConcatStageImpl(uint32_t bw, uint32_t esz, uint32_t numIn,
                                         uint32_t maxNIn, uint32_t nSlack,
                                         uint32_t gBuf, uint32_t aBuf,
                                         uint32_t tileGMaxBytes, uint32_t myRows,
                                         uint32_t tblK, bool dense,
                                         uint64_t trimRowBytes = 0)
{
    if (esz == 0 || bw == 0 || numIn == 0) {
        ConcatStageLayout L; L.why = C9_EMPTY; return L;
    }
    // 同上：与 LayoutConcatStage 共用同一份行账
    const ConcatRowBudget rb = ConcatRowBudgetOf(bw, esz, maxNIn, nSlack, trimRowBytes);
    const uint64_t gsz = rb.gsz, outPitch = rb.outPitch, tblRow = rb.tblRow;
    const uint64_t rowUse = rb.rowUse, slack = rb.slack;
    const bool     kI8 = rb.kI8;
    const uint32_t gB = (gBuf >= 1 && gBuf <= S9L_BUF_N) ? gBuf : 2;
    const uint32_t aB = (aBuf >= 1 && aBuf <= S9L_BUF_N) ? aBuf : 1;
    /* `tblK` 把偏移表从「每行一份」挪到「固定 K + 3 行」：
     * 它退出 `perTile`、进入 `fixed`。反推错了只会多绕几圈回缩循环（不影响正确性），
     * 但错得多了就会从一个离谱的 tileG 起步，白绕几十圈。 */
    /* **两种口径各推一次、取大的。**
     * `LayoutConcatStage` 里的护栏是「表不比摊开更省就用摊开」，而这里推 tileG 时
     * 还不知道 tileG 是多少（鸡生蛋）。若无条件按「分块」收费，`tileG` 本来就小的
     * 档（例如 3 行）会被 `K+3` 行的固定开销压到 2 —— 而回缩循环**只会更小、不会更大**，
     * 于是永远回不去。自检里 `rows=512 esz=2 K=4` 那条「缩表之后 tileG 反而变小」
     * 抓的就是这个。取两者的大值当起点，真实账由 `LayoutConcatStage` 判、放不下再回缩。 */
    // 密排时每行只占真实行宽与 bw 个表项；超行向上取整多出的至多 (superG−1) 行表、以及拼装区尾部对齐，进 fixed
    const uint32_t dG       = S9LDenseG(bw, esz, dense, tblK);
    const uint64_t pitchP   = (dG > 1) ? (uint64_t)bw * esz : outPitch;
    const uint64_t tblP     = (dG > 1) ? (uint64_t)bw : tblRow;
    const uint64_t denseFix = (dG > 1) ? ((uint64_t)(dG - 1) * bw * 4 + (uint64_t)aB * S9L_BLOCK_BYTES) : 0;
    const uint64_t vecFix   = (uint64_t)gB * slack + (kI8 ? slack * gsz : 0) + denseFix;
    const uint64_t vecTile  = (uint64_t)gB * rowUse + (uint64_t)aB * pitchP
                            + (kI8 ? (rowUse * gsz + tblP * gsz) : 0);
    // ① 摊开：表进 perTile    ② 分块：表进 fixed（K + 3 行）
    const uint64_t perTile0 = tblP * 4 + vecTile,             fixed0 = tblP * 4 + vecFix;
    const uint64_t perTileK = vecTile,
                   fixedK   = tblRow * 4 * (uint64_t)(tblK + S9L_TBL_EXTRA_ROWS + 1) + vecFix;
    const uint64_t budget0  = (S9L_UB_TOTAL > fixed0) ? (S9L_UB_TOTAL - fixed0) : 0;
    const uint64_t budgetK  = (S9L_UB_TOTAL > fixedK) ? (S9L_UB_TOTAL - fixedK) : 0;
    const uint64_t tg0      = perTile0 ? budget0 / perTile0 : 0;
    const uint64_t tgK      = (tblK && perTileK) ? budgetK / perTileK : 0;
    const uint64_t perTile  = perTile0;                       // 只用于下面的 0 判断
    const uint64_t fixed    = fixed0;
    (void)fixed;
    uint64_t tileG = (tgK > tg0) ? tgK : tg0;
    if (tileG == 0) { tileG = 1; }
    if (tileG > S9L_TILEG_HARD_MAX) { tileG = S9L_TILEG_HARD_MAX; }
    if (tileGMaxBytes > 0) {
        uint64_t cap = tileGMaxBytes / (pitchP ? pitchP : 1);
        if (cap == 0) { cap = 1; }
        if (tileG > cap) { tileG = cap; }
    }
    if (myRows > 0 && tileG > myRows) { tileG = myRows; }
    const uint32_t derived = (uint32_t)tileG;

    ConcatStageLayout L = LayoutConcatStage(bw, esz, numIn, maxNIn, nSlack, gB, aB,
                                            (uint32_t)tileG, tblK, dense, trimRowBytes);
    /* ④ 复核：对齐只会让账变大（最多 31·gBuf 字节），缩几格就够。
     * 缩到 1 行还放不下 ⇒ 这个配置本身不可行，如实返回。 */
    while (!L.fits && L.retrySmaller && L.tileG > 1) {
        L = LayoutConcatStage(bw, esz, numIn, maxNIn, nSlack, gB, aB, L.tileG - 1,
                              tblK, dense, trimRowBytes);
    }
    if (!L.fits && L.retrySmaller) { L.why = C9_ROW_TOO_WIDE; }
    L.tileGBudget = derived;
    return L;
}

S9_LAYOUT_AICORE inline ConcatStageLayout PlanConcatStage(uint32_t bw, uint32_t esz, uint32_t numIn,
                                         uint32_t maxNIn, uint32_t nSlack,
                                         uint32_t gBuf, uint32_t aBuf,
                                         uint32_t tileGMaxBytes, uint32_t myRows,
                                         uint32_t tblK, uint32_t denseMode,
                                         uint64_t trimRowBytes = 0, bool singleBand = false)
{
    // 单带覆盖整行，输入行宽之和恰为 bw，不需要两条跨带边界路的额外预算。
    // 逐路 32B 补齐余量及所有缓冲跨度对齐仍由共享布局函数保留。
    if (singleBand && trimRowBytes == 0) { trimRowBytes = (uint64_t)bw * esz; }
    const bool eligible = S9LUseDense(bw, esz, numIn, myRows, denseMode, tblK);
    ConcatStageLayout candidate = PlanConcatStageImpl(bw, esz, numIn, maxNIn, nSlack,
        gBuf, aBuf, tileGMaxBytes, myRows, tblK, eligible, trimRowBytes);
    if (!eligible || denseMode == 2) { return candidate; }
    // 自动模式还需比较实账：超行表的固定开销可能压缩 tileG。
    // 例如 3108 B 行、81 行/核时，密排把 tileG 从 7 降到 4，B1 对照出现退化。
    // 用布局结果保护所有同类输入，不能只按行宽设一个经验上限。
    const ConcatStageLayout padded = PlanConcatStageImpl(bw, esz, numIn, maxNIn, nSlack,
        gBuf, aBuf, tileGMaxBytes, myRows, tblK, false, trimRowBytes);
    if (!candidate.fits || (padded.fits && candidate.tileG < padded.tileG)) { return padded; }
    return candidate;
}

}  // namespace s9
#endif  // S9_CONCAT_LAYOUT_H
