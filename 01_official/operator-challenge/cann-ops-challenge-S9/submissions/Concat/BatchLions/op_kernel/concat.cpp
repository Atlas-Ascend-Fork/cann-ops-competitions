/* Concat 的 Kernel 侧。
 *
 * Copyright (c) 2026 ascend_simula contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * 算法与 `ops/concat/concat_kernel.h` 同源（那份在 CANN 8.5.0 的 CPU 域孪生调试下
 * 验过：12 个用例 × 3 种切核策略全部逐位相同）。这里的差别只有两处：
 *   1. 参数从**标量入参**改成 `GET_TILING_DATA` 从 GM 读 TilingData；
 *   2. 动态输入用 `ListTensorDesc` 解包，**一次 launch 吃完全部输入**，
 *      不再像自测版那样按 4 个一批。
 *
 * TilingKey ＝ 元素字节宽度（1/2/4/8）。Concat 只搬不算，
 * fp16/bf16/int16 走同一份实现。**`TILING_KEY_IS` 只能 if / else if，不支持 else**，
 * 所以四个宽度各写一条。
 */
#include "kernel_operator.h"
/* 混合核（KERNEL_TYPE_MIX_AIC_1_2）必须包含它。
 *
 * 构建期 `compile_op.py` 会给 mix 算子自动生成一段包装：
 *     matmul::clearWorkspace(workspace);
 *（`asc_op_compiler/compile_op.py:644-647`：`if is_mix and not is_c310()`）。
 * 那个符号是 `AscendC::clearWorkspace`（`impl/adv_api/detail/kfc/kernel_kfc.h`），
 * 靠 `include/adv_api/matmul/matmul.h:404` 的 `namespace matmul = AscendC;` 别名过去。
 * 不 include 就是 `error: use of undeclared identifier 'matmul'` —— 而且报错发生在
 * **框架生成的临时文件**里，看不出跟自己的源码有什么关系。
 *
 * 本算子一条 Matmul 都没有，纯粹是为了这个自动生成的包装。 */
#include "lib/matmul_intf.h"
// 动态输入（张量列表）的解包接口不在 kernel_operator.h 里，要单独包含
#include "kernel_operator_list_tensor_intf.h"
/* **逐块 UB 布局的共享算术**（纯 C++，不依赖 CANN，因而 host 与孪生用的是同一份）。
 * 三处各写一份公式曾经导致"账按分带算、缓冲按不分带开"（真机 507035），
 * 还漏了"手动轮转的块跨度要逐块 32B 对齐"这一条。见 s9_concat_layout.h 顶部。 */
// **设备侧要标 __aicore__**：不标的话 bisheng 把那些函数当 host 函数，只编译不发射
// 设备码，链接时报 `undefined symbol: s9::PlanConcatStage(...)`（官方 CANN 8.5.0
// 编 ConcatCustom 时踩过）。host 与 CPU 单测不定义这个宏，展开成空。
#define S9_LAYOUT_AICORE __aicore__
#include "s9_concat_layout.h"

using namespace AscendC;

namespace {
constexpr uint32_t BLOCK_BYTES = 32;
/* 缓冲块数。**从 2 提到 3 是 msprof 量出来的**：
 * d 档（2024,3000 fp16 dim=0）mte2_ratio 只有 0.469、mte3_ratio 0.306 ——
 * 两条搬运流水都闲着一半以上，说明不是吞吐受限而是**延迟受限**：
 * BUF_N=2 时任一时刻只有一趟在飞，每轮都在等上一轮的 flag。
 * 三块缓冲让 DMA 队列能同时排两趟，把等待藏进去。
 * 代价是每块要同比压小（UB 共 192 KB），见 s9_tiling_calc.h 的 UB_MAX。
 * p4 那种 mte2 已经 0.894 的档位吃不到这个好处，但也不该变差。 */
constexpr uint32_t BUF_N = 3;
// 拼装路径一次要把全部输入指针放在栈上，给个上限；超过就退回常规路径（host 判）
// 路数低于这个值就不整块取指针/nIn。
//
// 门槛不是"抵得过一条 DMA"那么简单：路数少时逐个 GetDataPtr 是**和大块 DMA 交错**
// 发生的，基本被搬运掩盖掉；整块取反而把它们全提到前面、串行在搬运之前。
// 实测（910B4，同一份代码只改门槛）：
//     57 路 dim=0   fp16 42.3 → 44.6 · fp32 71.6 → 74.5   **退 5%**（被串行化）
//     248 路        int32 43.1 → **33.4** · int8 58.8 → **46.0**   赢 22%
//     123 路        int8 58.1 → **49.0** 赢 16% · fp16 123.5 → 118.9 赢 4%
// 取 64：把 57 路那档排除在外，123 路及以上都吃到收益。
constexpr uint32_t S9_PTR_BULK_MIN = 64;
// 列带取整粒度**不再在这里写死** —— 它由 host 经 TilingData 的 bandGrain（元素）下发。
// 以前两边各存一份 512，S9_BANDGRAIN 之类的旋钮只改得动 host 那份，
// 扫出来的全是"两边不一致"的配置（colBurstEl 踩过同一个坑）。

/* Gather 的载体类型。
 *
 * 《Gather》的产品支持表里，Atlas A2 只列了 int16/uint16/half/bfloat16/int32/
 * uint32/float —— **int8 不在其中**。于是 int8 只能走"每路一条跨步写"的老路，
 * 而它的块长常常只有几十字节：(2048,4096) int8 123 路实测 178 µs，
 * 而带宽底只有 24 µs，是所有 dtype 里离底最远的一族。
 *
 * 借道 half 就能进拼装路：《Cast》的 Atlas A2 支持表里
 * `int8_t → half` 是 CAST_NONE、`half → int8_t` 有 CAST_RINT。
 * int8 的值域是 ±127，而 half 能精确表示到 2048 的整数 ⇒ **往返逐位无损**，
 * 不是"精度够用"而是恒等。两趟 Cast 是逐元素向量op（约 64 元素/拍），
 * 相对 Gather（实测约 4 元素/拍）可以忽略。 */
template <typename T> struct S9GatherAs      { using type = T; };
template <>           struct S9GatherAs<int8_t> { using type = half; };
// 拼装路的路数上限。**值在共享头里（`s9::S9L_ASM_MAX_IN`），两边都从它派生** ——
// 原先 host 与 kernel 各写一个 256，只靠注释维系；改单边就是栈上越界写（507035）。
constexpr uint32_t S9_ASM_MAX = s9::S9L_ASM_MAX_IN;
// 910B 单核 UB 容量（架构规格）。向量拼装路自己按这个数分配，
// 不走 host 的 ubBudget —— 它那个是给「双缓冲 × 单块」的常规路算的，口径不同。
/* TilingData 的**标量头**整块取进 UB 时占的字节数（32B 对齐，必须 ≥ nIn[] 之前的长度）。
 * 见下面 S9TilingUb 的说明。 */
constexpr uint32_t S9_HDR_BYTES = 128;
/* 序幕那一条 DMA 一次取多少：**标量头 ＋ 整个 nIn[] 数组**。
 *
 * nIn[] 紧跟在头后面，而 `CONCAT_MAX_IN = 256` ⇒ 数组最多 1024 字节。
 * 一条 DataCopyPad 取 128+1024 = 1152 字节和取 128 字节**是同一条指令**
 *（1152 B 按 32 B/拍也才 36 拍），却把「每路一次 GM 标量读」整段消掉了。
 * 代价是 1152 B UB（占 0.6%）。 */
constexpr uint32_t S9_PRE_IN    = 256;                                 // 与 CONCAT_MAX_IN 一致
constexpr uint32_t S9_PRE_BYTES = S9_HDR_BYTES + S9_PRE_IN * 4;        // 1152
constexpr uint32_t UB_TOTAL = 192 * 1024 - 8 * 1024 - S9_PRE_BYTES;   // 留 8 KB 余量 ＋ 序幕
/* 这个口径**必须**与共享布局头的 `S9L_UB_TOTAL` 一致 —— 预算算两遍正是
 * "host 说放得下、kernel 放不下"的来源。钉不住就是编译期错误。 */
static_assert(UB_TOTAL == s9::S9L_UB_TOTAL,
              "kernel 的 UB 预算必须与 s9_concat_layout.h 的 S9L_UB_TOTAL 一致");
// 按列切时每核每路至少搬多少元素 —— **不再是编译期常量，由 host 经 TilingData 给**。
//
// 以前这里写死 8192 并用移位（`nIn >> 13`），host 那边是另一份常量。
// 两份分开写有两个后果：① S9_COLEL 只改得动 host 那份，扫出来的全是
// "两边不一致"的配置，结论作废；② 载荷该按**字节**定而不是元素
//（fp32 的 8192 元素是 32 KB，fp16 只有 16 KB），一改就得两边同时改。
// 现在由 host 按 `COL_BURST_BYTES / esz` 算好传下来，两边不可能不一致。
// 除法换掉移位的代价：每路一次除法，只在按列切的入口，不在热路径。

__aicore__ inline uint32_t S9CeilDiv(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b : 0; }
__aicore__ inline uint32_t S9AlignUp(uint32_t a, uint32_t b) { return S9CeilDiv(a, b) * b; }
__aicore__ inline uint32_t S9Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
/* **UB 侧的块间跨距**（单位：32B dataBlock）＝ 一行 UB 的字节数 − 本块补齐后的长度。
 *
 * `CopyIn` 把它填进 `dstStride`、`CopyOut` 填进 `srcStride` —— **同一个量、两个方向**。
 * 原先两处各抄一遍同一个表达式（含那个整数除法），改行距时漏改一处就是 UB 越界
 *（真机 vector core exception 507035），而且编译期毫无察觉。 */
__aicore__ inline uint32_t S9UbGapBlocks(uint32_t rowUbBytes, uint32_t blockLen)
{
    return (rowUbBytes - S9AlignUp(blockLen, BLOCK_BYTES)) / BLOCK_BYTES;
}

/* ═══ 标量瘦身：这个 kernel 在好几档上是**标量受限**的 ═══
 *
 * msprof（PipeUtilization）在 910B4 上量到：
 *   c 档 1024,1024 fp32  aiv_scalar_ratio **0.989**（mte2 才 0.344）
 *   d 档 2024,3000 fp16  aiv_scalar_ratio **0.997**（mte2 才 0.469）
 *   f 档 128,4096 fp16   aiv_scalar_ratio 0.729
 * 也就是说搬运流水在**等标量算地址**，不是带宽不够。
 *
 * 罪魁是每一路都要做的整数除法与取模。AIV 的标量单元上一次除法要几十拍，
 * 而这些除数**全都是 2 的幂或很小的数**，可以换成移位和条件减。
 *
 * 下面这些辅助只在「除数保证是 2 的幂」时用，调用点都标了理由。 */
__aicore__ inline uint32_t S9AlignUpP2(uint32_t a, uint32_t mask)  // mask = 2^k − 1
{
    return (a + mask) & ~mask;
}
/* a、b 都在 [0, m) 内时的模加：一次条件减顶掉一次取模 */
__aicore__ inline uint32_t S9AddMod(uint32_t a, uint32_t b, uint32_t m)
{
    uint32_t s = a + b;
    return s >= m ? s - m : s;
}

/* ═══ TilingData 的 l2Mode → CacheMode：**两张表，一个来源** ═══
 *
 * 原先这段映射在 kernel 里抄了 **5 遍**（`Init` 的输出侧、`One`、
 * `Assembled`、`GatherAssembled` 的 `cm`、`CopyIn`），而且是**三种写法**：
 *   · 输出侧：`==4 PERSISTENT / ==1 NORMAL / else DISABLE`
 *   · 输入侧：`==0 DISABLE / ==4 PERSISTENT / else NORMAL`（两处 if-else + 一处三元）
 * 抄五遍的代价不是慢，是**改一处忘一处**（`S9_GBUF` 踩过同一个坑：host 允许 1..8
 * 而 kernel 事件号只到 3）。收成两个函数之后，"输入与输出用的是不同的表"
 * 这件事也变得一眼能看见 —— 它**是有意的**，不是笔误。
 *
 * **实测依据（2026-09-13 配对 A/B，`ab_knob.sh` R=4，四轮离散度 <1%）：**
 *   · 输入改 NORMAL：4096×4096 f16 101.70→110.56（1.087）、
 *                   4096×16384 f16 411.17→457.60（1.113）⇒ **DISABLE 快 8.7~11.3%**
 *   · 输出改 DISABLE：101.89→132.63（1.302）、410.23→503.12（1.226）
 *                   ⇒ **NORMAL 快 23~30%**
 * 机理：**输入是流式读一次**，让它进 L2 只会占着缓冲过一遍就被冲掉；
 * **输出写一次、马上可能被下一个算子读**，留在 L2 才有意义。 */
__aicore__ inline CacheMode S9InCacheMode(uint32_t l2)
{
    return (l2 == 0) ? CacheMode::CACHE_MODE_DISABLE
         : (l2 == 4) ? CacheMode::CACHE_MODE_PERSISTENT
                     : CacheMode::CACHE_MODE_NORMAL;
}
__aicore__ inline CacheMode S9OutCacheMode(uint32_t l2)
{
    return (l2 == 4) ? CacheMode::CACHE_MODE_PERSISTENT
         : (l2 == 1) ? CacheMode::CACHE_MODE_NORMAL
                     : CacheMode::CACHE_MODE_DISABLE;
}

/* **这一路（宽 n 列、起点 off）与带 [cb0, cb1) 有没有交集。**
 *
 * 这个三元判断在 `GatherAssembled` 里抄了 **4 遍**（每带计划的前缀和、
 * 全扫描的前缀和、两张表的逐路循环、路由压紧的逐路循环）。四处必须给出
 * **同一个**"本带路由集合" —— 只要有一处漂了，前缀和算的是 A 集合、
 * 偏移表铺的是 B 集合，输出会**静默错列**（不报错、不越界，只是数据不对）。
 * 收成一份之后不可能再漂。 */
__aicore__ inline bool S9RouteHitsBand(uint32_t n, uint32_t off, uint32_t cb0, uint32_t cb1)
{
    return n != 0 && off < cb1 && off + n > cb0;
}

/* **逐路累加 offOut，并夹出这一路与本核列区间 [c0, c1) 相交的那一段 [a, b)。**
 *
 * `off` 是**进/出参**：无论这一路用不用得上都要往前推（offOut 是 nIn 的前缀和）。
 * 返回 false ＝ 这一路与本核无关（零宽 / 完全在左 / 完全在右）。
 *
 * 混合核那条路的 **AIC 半边与 AIV 半边各抄了一遍**同一段（取 n、推 off、判相交、
 * 夹出 a/b），而两边抄出来的边界条件还不一样（AIC 多一个 `pct == 0` 的前置短路）。
 * 收成一个来源之后，"两边看到的是同一段列"这件事不再靠人眼比对。 */
__aicore__ inline bool S9ClipRoute(uint32_t n, uint32_t &off, uint32_t c0, uint32_t c1,
                                   uint32_t &s, uint32_t &a, uint32_t &b)
{
    s = off;
    off += n;
    if (n == 0 || s >= c1 || off <= c0) { return false; }
    a = (s > c0) ? s : c0;
    b = (off < c1) ? off : c1;
    return b > a;
}
}

/* 按字节偏移从 GM 直读 TilingData 的一个 uint32 字段。
 *
 * **这里刻意不用 `GET_TILING_DATA`。** 那个宏由构建期的
 * `cmake/util/tiling_data_def_build.py` 生成，展开成
 *     ConcatTilingData t; InitConcatTilingData(tiling, &t);
 * 而生成的 Init 是
 *     for (i = 0; i < sizeof(Struct)/4; i++) dst[i] = src[i];
 * —— **逐个 uint32 从 GM 标量读**。本算子的 TilingData 有两个 64 元素数组
 * （nIn / offOut），共 544 字节，也就是**每次 launch 136 次标量 GM 读**，
 * 而 3 路输入实际只用到 14 个字，其余 122 次纯属浪费。
 *
 * profiling 上这表现为小 shape 的 scalar 占比奇高（longmid 最优点 0.84、mte2 才 0.42）。
 * 把 CONCAT_MAX_IN 临时改成 8（544→96 字节）做过判定性实验：
 * longmid 最优点 3.74 → 3.04 us（−19%），确认代价就在这里。
 *
 * 偏移用 offsetof 取，结构体字段顺序变了会自动跟着变，不写死数字。
 */
struct S9TilingRd {
    __gm__ const uint32_t *w_;
    __aicore__ inline explicit S9TilingRd(GM_ADDR t) : w_((__gm__ const uint32_t *)t) {}
    __aicore__ inline uint32_t At(uint32_t byteOff) const { return w_[byteOff >> 2]; }
};

/* **标量头也整块取。** 上面那段说的是 nIn[] 数组，这里说的是它前面的十几个标量字段。
 *
 * `S9TilingRd::At` 每调一次就是一次 **GM 标量读**，而 `RunConcat` 的序幕里
 * 光是拼 `op.Init(...)` 的实参就要读 17 个字段。按同一份 msprof 拟合出来的
 * 「每次 GM 标量读约 0.15 µs」算，这一项就是 **2.5 µs** —— 而小档（样例 case1
 * 是 128×256 fp16，只有 64 KB）实测总耗时才 4.90 µs，其中 scalar 占 3.37。
 * 路数扫描（同样字节、只改路数）给出 **固定 3.90 µs ＋ 每路 0.104 µs**，
 * 而厂商 `ConcatD` 在小张量上的 Task Duration 中位只有 **1.86 µs**。
 *
 * 一条 DataCopyPad 把 nIn[] 之前的整个头搬进 UB，之后全是 UB 标量读（几个拍）。
 * 代价是一次 DMA ＋ 一对 MTE2→S 同步 ＋ 128 字节 UB。 */
struct S9TilingUb {
    LocalTensor<uint32_t> w_;
    __aicore__ inline uint32_t At(uint32_t byteOff) const { return w_.GetValue(byteOff >> 2); }
};

/* 两种取法二选一（`preMode`）。**只为做配对 A/B**：公共资源池上单次测量的离散度
 * 约 ±20%，"改一版跑一遍"分不出 3% 的差别，必须同一份包里交替跑。 */
struct S9TilingRdAny {
    S9TilingRd            g_;
    LocalTensor<uint32_t> u_;
    bool                  pre_;
    __aicore__ inline uint32_t At(uint32_t byteOff) const
    {
        return pre_ ? u_.GetValue(byteOff >> 2) : g_.At(byteOff);
    }
};

#define S9_TD(fld) td.At((uint32_t)offsetof(ConcatTilingData, fld))

/* 一次 launch 把全部输入搬到输出的各自列区间。
 * T 只用来定字节宽度（1/2/4/8），不参与计算。 */
template <typename T>
class ConcatMulti {
public:
    __aicore__ inline ConcatMulti() {}

    __aicore__ inline LocalTensor<uint32_t> TdBuf() { return tdBuf_.template Get<uint32_t>(); }
    /* 每带计划（`S9_BANDPLAN=1`）里的那一段：本核只碰 [planLo_, planHi_) 这几路。
     * 计划关掉时 PlanOn() 为假，调用方走原来的全扫描路径。 */
    __aicore__ inline bool     PlanOn()   const { return planOn_; }
    __aicore__ inline uint32_t PlanLo()   const { return planLo_; }
    __aicore__ inline uint32_t PlanHi()   const { return planHi_; }
    __aicore__ inline uint32_t PlanOff()  const { return planOff_; }
    __aicore__ inline bool     Direct()   const { return directMode_ != 0; }
    /* **参数只有四个：输出、TilingData、读取器、TPipe。**
     *
     * 原先是 **28 个位置参数**，调用点要按顺序摆 20 个 `S9_TD(...)`、3 个
     * 计划表指针、2 个"必须走 GM 读"的尾部字段。三件坏事同时存在：
     *   ① 28 个同类型（uint32_t）的位置参数，**调换两个不报错** ——
     *      而这些量决定切核、缓冲块数与列带，错了是"写到别的核的地盘"，
     *      不是崩溃；
     *   ② "哪些字段必须走 `tdg.At`（GM）而不是 `td.At`（UB 序幕）"这条知识
     *      写在**调用点**，离用它的地方几十行远 —— `layOn`/`trimOn` 在
     *      TilingData 尾部、超出序幕那条 `S9_PRE_BYTES` 的 DMA 覆盖范围，
     *      走错就读到 UB 缓冲之外（不报错）；
     *   ③ 加一个 tiling 字段要同时改签名、调用点、函数体三处。
     * 现在字段名与它的读法写在一起，调用点只剩一行。
     *
     * `td` 是序幕那份可能走 UB 的读取器；`tiling` 是 GM 基址，尾部字段
     * （`layOn`/`trimOn`）与三张计划表都从它取。 */
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR tiling, const S9TilingRdAny &td,
                                TPipe *pipe)
    {
        // 序幕覆盖得到的标量字段：走 td（preMode=1 时是 UB 读）
        const uint32_t rows = S9_TD(rows), nOut = S9_TD(nOut), blkNum = S9_TD(blockDim);
        const uint32_t strategy = S9_TD(strategy), ubBudget = S9_TD(ubBudget);
        const uint32_t l2Mode = S9_TD(l2Mode), l2ModeOut = S9_TD(l2ModeOut);
        // **路数按拼装路的栈数组上限夹住** —— 超了就不是慢，是写到栈外
        const uint32_t numIn = S9Min(S9_TD(numIn), S9_ASM_MAX);
        // maxNIn 由 host 算好传下来：kernel 侧再扫一遍要 numIn 次 GM 标量读
        const uint32_t maxNIn = S9_TD(maxIn), pipeMode = S9_TD(pipeMode);
        const uint32_t gBuf = S9_TD(gBuf), bands = S9_TD(bands), aBuf = S9_TD(aBuf);
        const uint32_t bandGrain = S9_TD(bandGrain), bandBal = S9_TD(bandBal);
        const uint32_t hintMode = S9_TD(hintMode), maxInBand = S9_TD(maxInBand);
        const uint32_t tileGMax = S9_TD(tileGMax), colBurstEl = S9_TD(colBurstEl);
        const uint32_t planOn = S9_TD(planOn);
        // 每带计划的三张表在 TilingData 里（nIn[] 之后）；越界防护在下面
        //（lo ≤ hi ≤ numIn 才采信）。
        __gm__ const uint32_t *bandLo =
            (__gm__ const uint32_t *)(tiling + offsetof(ConcatTilingData, bandLo));
        __gm__ const uint32_t *bandHi =
            (__gm__ const uint32_t *)(tiling + offsetof(ConcatTilingData, bandHi));
        __gm__ const uint32_t *bandOff =
            (__gm__ const uint32_t *)(tiling + offsetof(ConcatTilingData, bandOff));
        // 逐核布局表（布局下沉）：**整条表在 TilingData 的最后**，所以上面那些字段的
        // 偏移一个字节都没动，序幕那条 S9_PRE_BYTES 的 DMA 也不受影响。
        __gm__ const uint32_t *layRows =
            (__gm__ const uint32_t *)(tiling + offsetof(ConcatTilingData, layRows));
        /* **`layOn` 与 `trimOn` 必须走 GM 直读**（`S9TilingRd`），不能走 `td`：
         * 它们在 TilingData 尾部，超出序幕那条 `S9_PRE_BYTES` 的 DMA 覆盖范围，
         * preMode=1 时 `td.At` 会读到 UB 缓冲之外。`trimOn` 排在 `layRows`
         * 后面，所以只多一次 GM 标量读，不挪任何已有偏移。 */
        S9TilingRd tdg(tiling);
        const uint32_t layOn  = tdg.At((uint32_t)offsetof(ConcatTilingData, layOn));
        const uint32_t trimOn = tdg.At((uint32_t)offsetof(ConcatTilingData, trimOn));
        // `directOn` 紧跟 `trimOn`，同样在 TilingData 尾部 ⇒ 只能 GM 直读。
        const uint32_t directOn = tdg.At((uint32_t)offsetof(ConcatTilingData, directOn));
        const uint32_t tblKOn = tdg.At((uint32_t)offsetof(ConcatTilingData, tblKOn));
        const uint32_t preRdOn = tdg.At((uint32_t)offsetof(ConcatTilingData, preRdOn));
        const uint32_t colParts = tdg.At((uint32_t)offsetof(ConcatTilingData, colParts));
        const uint32_t hoistOn = tdg.At((uint32_t)offsetof(ConcatTilingData, hoistOn));
        // `oneWriteOn` 在 `hoistOn` 之后、TilingData 尾部 ⇒ 同样只能 GM 直读
        const uint32_t oneWriteOn = tdg.At((uint32_t)offsetof(ConcatTilingData, oneWriteOn));
        const uint32_t denseOn = tdg.At((uint32_t)offsetof(ConcatTilingData, denseOn));

        rows_ = rows; nOut_ = nOut; strategy_ = strategy; pipeMode_ = pipeMode;
        // 边界路裁剪（S9_TRIM）：1/2 **kernel 行为完全一样**，差别只在 host 怎么算
        // UB 账（见 op_host 的说明）。kernel 只关心"开不开"。
        trim_ = (trimOn != 0);
        directMode_ = directOn;
        tblK_ = tblKOn;
        preRd_ = (preRdOn != 0);
        colParts_ = (colParts >= 1) ? colParts : 1;
        hoist_ = (hoistOn != 0);
        oneWrite_ = (oneWriteOn != 0);
        dense_ = directOn == 0 && tblKOn == 0 ? denseOn : 0;
        bal_ = bandBal;
        hint_ = hintMode;
        gBuf_ = (gBuf >= 1 && gBuf <= 8) ? gBuf : 2;
        aBuf_ = (aBuf >= 1 && aBuf <= BUF_N) ? aBuf : 1;
        colBurstEl_ = colBurstEl ? colBurstEl : 1;
        bn_ = blkNum ? blkNum : 1;
        bi_ = GetBlockIdx();
        ubBudget_ = ubBudget ? ubBudget : (48 * 1024);
        l2Mode_ = l2Mode;
        dstGm_.SetGlobalBuffer((__gm__ T *)dst);
        // 输出侧的 L2 模式由 tiling 给，默认 **NORMAL**（映射见 S9OutCacheMode）。
        // 本节初稿把这里写成"默认 DISABLE、且 DISABLE 才对"，**被配对 A/B 推翻**：
        // 输出改 DISABLE 在两个大档上慢 23%~30%。理由见 S9OutCacheMode 的注释。
        dstGm_.SetL2CacheHint(S9OutCacheMode(l2ModeOut));
        // **与输入无关的量在这里算一次就够。**
        // Profiling 说这个 kernel 是**标量受限**的（aiv_scalar_ratio 0.907，
        // 而 MTE2 只有 0.58、MTE3 只有 0.21 —— 搬运流水在等标量）。
        // 原先 rows_/bn_ 与 rows_%bn_ 这两个**整数除法**在每个输入里各算一遍，
        // 9 个输入就是 18 次除法，而它们**根本不随输入变**。
        alignEl_ = BLOCK_BYTES / sizeof(T);
        if (alignEl_ == 0) { alignEl_ = 1; }
        // alignEl_ 恒为 2 的幂（32/1、32/2、32/4、32/8 ＝ 32、16、8、4），
        // 预存掩码，热路径上的对齐就不用除法了（见文件头 S9AlignUpP2 的说明）。
        alignMask_ = alignEl_ - 1;
        colChunk_ = (ubBudget_ / sizeof(T)) & ~alignMask_;
        if (colChunk_ == 0) { colChunk_ = alignEl_; }
        if (strategy_ == 4) {
            // **按输入切核：本核包干若干整路，行是全的。**
            // 路数很多时按行切会退化成"大量小 DMA"（250 路、rows=128、40 核 ⇒
            // 每核 500 条、每条几十字节）。改成一核一批整路之后，
            // 每条 DMA 的 blockCount 从 rows/核 变成 rows，
            // 搬运字节数不变、**指令数少一个 bn_ 倍**。
            // 哪些路归本核，由 RunConcat 按累计列数判（见那边的说明）。
            rowBeg_ = 0;
            rowEnd_ = rows_;
        } else if (strategy_ != 1) {                // 按行切：区间与输入无关
            /* 二维分核：核编号拆成「行组 × 列组」。`colParts_ == 1` 时
             * `rowParts == bn_`、`ri == bi_`，与旧写法**逐字相同**。 */
            const uint32_t rp = (colParts_ > 1) ? (bn_ / colParts_) : bn_;
            const uint32_t rowParts = rp ? rp : 1;
            const uint32_t ri = (colParts_ > 1) ? (bi_ / colParts_) : bi_;
            colIdx_ = (colParts_ > 1) ? (bi_ % colParts_) : 0;
            uint32_t q = rows_ / rowParts, r = rows_ % rowParts;
            rowBeg_ = q * ri + (ri < r ? ri : r);
            rowEnd_ = rowBeg_ + q + (ri < r ? 1 : 0);
        }
        // nIn[] 数组的落脚点：所有策略都要逐路读它，整块搬进 UB 比逐个读 GM 快两个数量级
        // 这块 UB 先后装两样东西：nIn[]（每路 4 字节）与动态输入的指针数组（每路 8 字节）。
        // 按大的那个开，两者不同时在用（先取指针、再取 nIn，或反之）。
        pipe->InitBuffer(tdBuf_, S9AlignUpP2(numIn * 8, BLOCK_BYTES - 1) + BLOCK_BYTES);
        if (strategy_ == 3) {
            // ═══ 向量拼装：暂存区（每路一段，**行内打包**）＋ 拼装区 ＋ 两张表 ═══
            //
            // **暂存区从「统一行距」改成「每路按自己的宽度打包」**，这是 msprof 逼出来的：
            //   统一行距时每一路每行发一条 DataCopyPad（Case5 那档 blockLen 只有 66 字节），
            //   而 66 字节的非对齐读要碰 2~3 条 64B 缓存行 ——
            //   实测 aiv_GM_to_UB = 62720 KB，而张量本身只有 32768 KB，**读放大 1.91×**。
            //   Gather 路把写放大治好了（UB_to_GM 93696 → 32768 KB，正好 1.0×，
            //   mte3 从 0.475 掉到 0.040），可 mte2 还是 0.861 —— 读成了唯一的瓶颈。
            //
            // 打包之后，一路一轮只发**一条**连续 DMA（blockLen = tile·nIn·esz），
            // 放大降到 (tile·nIn·esz + 63)/(tile·nIn·esz)。
            //
            // 代价：每一路的行距变成各自的 nIn·esz，"一张表服务所有行"不再成立。
            // 补法是再存一张**每列的行步长**表 dlt[c] = nIn(c)·esz，
            // 每处理完一行就 `Add(off, off, dlt)` 把整张表推进一行 —— 一次向量加而已
            //（vec 现在才 0.118，加得起）。一轮结束再减回去，省掉一份原始表。
            //
            // ═══ 列带分核：核排成 bands × (bn_/bands) 的网格 ═══
            //
            // 打包读把**读放大**治好之后，剩下的是**发射开销**：每核每 tile 仍要发
            // numIn 条 DataCopyPad。真机上同字节数只改路数（4096×4096 fp16 33.5 MB）：
            //     123 路 151.27 · 33 路 116.03 · 14 路 104.70 · 8 路 104.33 µs
            // 每多一路 +0.41 µs；8 路时已经贴住带宽底（97.6）。msprof 佐证：
            // mte2_ratio 0.68 却只有 4.6 B/拍（硬件 32 B/拍）⇒ 每条 DMA 约 37 拍是纯开销。
            //
            // 分带一次治两头：本核只碰与自己列带相交的 numIn/bands 路，
            // 而且一行只占 bw·esz 而不是 nOut·esz ⇒ 同样的 UB 装得下 bands 倍的行
            // ⇒ 每条 DMA 又大 bands 倍。代价是写从整行连续变成
            // blockCount=cnt、blockLen=bw·esz 的跨步块，所以 host 给带宽定了下限。
            // ═══ 核 →（带, 组, 列区间, 行区间）：**与 host 同一份** ═══
            //
            // 这段算术原先只写在这里。布局下沉（`S9_LAYOUT=1`，host 预先算好每个核的
            // 布局再下发）要求 host 也能算出**一模一样**的映射 —— 两边各写一份就会
            // 变成"host 按 A 核的账算、kernel 跑 B 核的配置"，那不是慢，是互相写对方
            // 的地盘（结果错且不报错）。所以剥到 `common/s9_concat_layout.h` 的
            // `MapConcatCore` 里，两边都调它。
            //
            // **带数不必整除核数**（除不尽时把余数分给前几条带各多一组，核全部用上），
            // 所以核号 → 带号要走一遍前缀和。循环最多 BAND_MAX(64) 次，每核一次，
            // 不在热路径。
            const uint32_t grainElPre = bandGrain ? bandGrain : alignEl_;
            const s9::ConcatCoreMap cm = s9::MapConcatCore(rows_, nOut_, bn_, bands,
                                                           grainElPre, bal_, bi_, alignEl_);
            bands_    = cm.bands;
            bandCols_ = cm.bandCols;
            cb0_ = cm.cb0; cb1_ = cm.cb1; bw_ = cm.bw;
            rowBeg_ = cm.rowBeg; rowEnd_ = cm.rowEnd;      // 行按组切（覆盖上面按 bn_ 切的那对）
            const uint32_t bandIdx = cm.bandIdx;
            /* **每带计划**（`S9_BANDPLAN=1`）：本核这条带要碰哪几路输入，host 已经算好。
             * 校验一遍再采信 —— 越界的数据只可能来自"两边不同源"，那种情况下宁可
             * 当空带（不干活），也不越界读（结果错且不报错）。 */
            if (planOn != 0 && bandLo != nullptr && bandHi != nullptr && bandIdx < bands_) {
                const uint32_t lo = bandLo[bandIdx];
                const uint32_t hi = bandHi[bandIdx];
                if (lo <= hi && hi <= numIn) {
                    planOn_  = 1;
                    planLo_  = lo;
                    planHi_  = hi;
                    planOff_ = (bandOff != nullptr) ? bandOff[bandIdx] : 0;
                }
            }
            // 带宽按 host 下发的粒度取整（bandGrain 已是**元素**）。
            // 两边算出不同的 bandCols 就会写到别的核的地盘上，所以只能有一个来源。
            // ── 列带边界：**均衡切分**（与 s9_tiling_calc.h 的 BandRange 一字一样）──
            //
            // 原来是「统一带宽 ＋ 末带吃剩下」：bandCols = AlignUp(⌈nOut/bands⌉, grain)，
            // 末带只剩零头。nOut=16384、bands=10、grain=256 时前 9 条各 1792、
            // **末带只有 256** —— 那一组核只有别人 1/7 的活，另外 36 个核每人扛 1/36
            // 而不是 1/40，**白丢约 10%**。msprof 上就是 aiv_time 393.69 而
            // Task Duration 465.72 那 15% 的空档。
            //
            // 现在把 G = ⌈nOut/grain⌉ 个粒度均分给 bands 条：第 b 条拿
            // [b·G/bands, (b+1)·G/bands)。各带最多差一个粒度（上例 6×1536 ＋ 4×1792）。
            // **两边必须同源** —— 算出不同的区间就会互相写对方的地盘（错且不报错）。
            // **上面那段算术现在在 `MapConcatCore` 里**（`cm.bandCols/cb0_/cb1_/bw_`
            // 已经取好）：空带（`cb0 ≥ cb1`）在那里被收成"不干活但缓冲合法"。
            // ── 偏移表**一次摊开 tileG 行**，一个 tile 只发一次 Gather ──
            //
            // 原先是每行一次 Gather ＋ 一次 Add 把表推进一行。分带之后行数涨了
            // bands 倍、每次 Gather 的 count 却小了 bands 倍，于是**向量指令的发射开销
            // 反噬**：msprof 实测 bands=1 → 10 时 mte2 97.2 → 55.1 µs（读侧如愿），
            // 可 vec 68.1 → 113.7 µs，净亏 76 µs。折算每条多余向量指令约 37 拍。
            //
            // 表只依赖暂存区的布局，而暂存区每个 tile 都填在同样的地址上
            // ⇒ **整张表全程不变**，可以一次建好 tileG 行：第 0 行照旧建，
            // 后面每行一条 `Add(off[j·tblRow], off[(j−1)·tblRow], dlt)` 推出来。
            // 于是每 tile 只剩一条 Gather，行内的 Add/Sub 全部消失。
            // 代价是表占 tileG·tblRow·4 字节，会把 tile 压小 —— 所以它和分带是**一对**：
            // 不分带时表太大反而亏（bands=1 时 tile 从 6 掉到 3）。
            //
            // **"只摊开 R 行、组间用向量 Add 推进"这条试过，是负面的。**
            // 动机看着很站得住：表占每行 UB 的三到四成，把 tile 压在 8 行，
            // 而 msprof 上 (4096,16384) fp16 那档 mte2_ratio 0.938、vec 只有 0.498,
            // 像是"向量有余量、可以拿它换读指令"。实测（R=0 即摊开整个 tile 为基准）：
            //     (4096,16384) f16  R=0 **471.30** · R=2 480.59 · R=4 490.70
            //                       · R=8 508.42 · R=16 1246.42
            //     (4096,4096)  f16  R=0 **118.93** · R=2 144.51 · R=4 132.90 · R=8 120.10
            //     (16384,16384) f16 R=0 **1833.47** · R=2 1871 · R=4 1909 · R=8 1938
            // 每档都是 R=0 最快 —— 多出来的向量指令（每 tile 2·组数 条）比省下的
            // 读指令更贵，和「每条向量指令约 37 拍」那笔账一致。
            //
            // 顺带一个会静默出错的坑：推进量**必须铺成 R 行**、一次推进整张表。
            // 只推第 0 行时 R=1 碰巧对、R≥2 直接算错（真机 MISMATCH，不报错）。
            //
            // 补白列（bw 不是 32B 整数倍时的末带）：off 取 cl·esz、dlt 取 0 ——
            // 落在暂存区内、且逐行不动，读到的是垃圾但写出时按 blockLen=bw·esz 丢掉。
            // **余量只按「任一条带里最多几路」留，不是按全部路数。**
            //
            // 暂存区里只放与本带相交的那几路（读循环只走 [iLo,iHi)），每路块起点按
            // 32B 对齐，最坏多出 32B/路。原来按 numIn 留 —— 248 路时 15872 B，
            // 占 UB 的 8.4%，而一条带通常只有几十路。这块是 `fixed`，直接压小 tileG，
            // 而 msprof 指出暂存读的指令条数是 `numIn × rows / tileG`。
            //
            // **这个数由 host 决定并经 TilingData 传下来**（默认是全部路数 ——
            // 实测比按 MaxInputsPerBand 精确数出来的那个值更快，见 op_host 的 S9_MAXINB）。
            // 关键不变量：**host 的可行性判据用同一个数**，否则就是
            // "host 按精确值算账、kernel 按全路数开缓冲" —— 账算得下、真机放不下
            //（vector core exception 507035）。这条现在由 GatherFitsBands →
            // PlanConcatStage 的同一份账保证。
            constexpr bool kI8I = (sizeof(T) == 1);
            const uint32_t nSlack = (maxInBand > 0 && maxInBand < numIn) ? maxInBand : numIn;
            // **tileG 有最优点，不是越大越好，而且最优点用「字节」表达才统一。**
            //
            // 910B4 上扫出来（同一份包，只改上限）：
            //     TILEG(行)      G16      p4      I2
            //       顶满        518.02   64.06   39.50
            //         8       **499.99**  96.01   77.97
            //        16         516.74   64.93   50.72
            //        32         517.20   64.47 **39.61**
            // 三档的带宽不同（G16 的 bw·esz ＝ 2048 B，p4/I2 ＝ 512 B），
            // 换算成**每个 tile 的拼装区字节数**，三个最优点都是 **16 KB**
            //（8×2048、32×512、32×512）。所以上限按字节给，除以 outPitch 得行数。
            //
            // 拼装区一个 tile 的字节数 ＝ 一条 CopyOut 指令的载荷
            //（blockCount=tileG、blockLen=bw·esz），超过某个量之后
            // 那条写指令跨的输出地址范围太大，收益转负。
            // **还要被本核实际的行数夹住。** 表是一次摊开 tileG 行的，
            // 展开靠 tileG−1 条向量 Add —— 行数比 tileG 少时那些 Add 全是白做，
            // 而且大表和拼装区也按 tileG 白占 UB。
            // 4 个元素、dim=0 那档就栽在这里：UB 反推出 tileG≈1466，
            // 一行的问题发了 1465 条 Add，5.6 µs 变成 30.3 µs。
            // ═══ 逐块布局：算术在 `common/s9_concat_layout.h`，host 与 kernel 共用 ═══
            //
            // **旧写法在这里手算 `stageSpan_ = tileG_*rowUse + slack`，块跨度没有
            // 逐块 32B 对齐** —— 第二块（第三块…）暂存的起点就落在非 32B 边界上，
            // 而 DataCopyPad 对 UB 侧的目的地址硬要求 32B 对齐。可复现的反例：
            // fp16、dim=1、`(4096,63)×4 ＋ (4096,4)`、40 核 ⇒ tileG=51、rowUse=764、
            // slack=160 ⇒ 旧跨度 **39124**（%32 = 20）。见 s9_concat_layout.h 顶部的说明。
            //
            // 边界路是**整行读**（blockCount=1）而不是每行一块：一条带最多切开两条路，
            // 若按带内宽度跨步读，那两条路每 tile 就要发 cnt 个几十字节的小块 ——
            // 正是打包读之前的老毛病。整行读多搬 (nIn−w)·esz/行，但块数从 2·cnt 降到 2。
            // int8 每 tile 行还要多一份 half 副本（暂存 ＋ 拼装结果）：**只多一份，
            // 不随 gBuf/aBuf 翻倍** —— 它们是 TBuf，不入队、没有块轮转
            //（《TBuf》：TPipe 只会为 TBuf 分配一块内存）。这两项都在那份账里，
            // 而且**整份账要连序幕与指针暂存一起算**，比原先的 UB_TOTAL 口径更严。
            const uint32_t myRows = (rowEnd_ > rowBeg_) ? (rowEnd_ - rowBeg_) : 1;
            s9::ConcatStageLayout lay{};
            bool layFromTable = false;
            // ═══ 布局下沉（`S9_LAYOUT=1`）：host 已经把这一核的整份布局算好了 ═══
            //
            // 校验三条再采信 —— 表是 host 按**它以为的**核号/带区间算的，错位时
            // 宁可退回自算（慢一点，但不会让两个核互相覆盖）：
            //   ① `tileG` 与暂存跨度非 0（那是"这个核放不下"的信号）；
            //   ② 表里这一核的列区间**与本核算出来的逐位相同**。
            // 逐字段一致性另有脱机检查（`test_concat_layout.cpp` 的 CheckCoreLayoutTable
            // 对 3000 个核逐字段比过），这里只是运行期的最后一道。
            if (layOn != 0 && layRows != nullptr && bi_ < s9::S9L_CORE_MAX) {
                const __gm__ const uint32_t *r =
                    layRows + (size_t)bi_ * (size_t)s9::S9L_CORE_FIELDS;
                const uint32_t lCb0 = r[s9::C9F_CB0];
                const uint32_t lCb1 = r[s9::C9F_CB1];
                const uint32_t lTileG = r[s9::C9F_TILE_G];
                if (lTileG != 0 && lCb0 == cb0_ && lCb1 == cb1_) {
                    lay.outPitch    = r[s9::C9F_OUT_PITCH];
                    lay.tblRow      = r[s9::C9F_TBL_ROW];
                    lay.tileG       = lTileG;
                    lay.stageSpan   = r[s9::C9F_STAGE_SPAN];
                    lay.asmSpan     = r[s9::C9F_ASM_SPAN];
                    lay.offTblBytes = r[s9::C9F_OFF_TBL];
                    lay.dltBytes    = r[s9::C9F_DLT];
                    lay.castSpan    = r[s9::C9F_CAST];
                    lay.asmGBytes   = r[s9::C9F_ASM_G];
                    // outPitch 已编码 host 最终选择：非对齐行只有密排才等于真实行宽。
                    // 不能仅用 dense_ 重算，否则会漏掉 host 的 tileG 比较保护。
                    lay.superG      = s9::S9LDenseG(bw_, sizeof(T),
                        lay.outPitch == bw_ * sizeof(T), tblK_);
                    lay.fits        = true;
                    layFromTable    = true;
                }
            }
            if (!layFromTable) {
                lay = s9::PlanConcatStage(bw_, (uint32_t)sizeof(T), numIn, maxNIn, nSlack,
                                          gBuf_, aBuf_, tileGMax, myRows,
                                          tblK_, bands_ == 1 ? dense_ : 0u, 0, bands_ == 1);
                if (!lay.fits || lay.tileG == 0) {
                    // host 的 GatherFitsBands 走的是同一份账，正常到不了这里。真到了
                    //（例如某个只在 host 生效的 S9_* 旋钮把两边拉开），退回**一行的最小布局**：
                    // 宁可慢，也不越界。这些覆盖的最终校验在 host 侧（见 TilingFunc 末尾）。
                    lay = s9::LayoutConcatStage(bw_, (uint32_t)sizeof(T), numIn, maxNIn, nSlack,
                                                gBuf_, aBuf_, 1, tblK_, false);
                }
            }
            outPitch_   = lay.outPitch;       // 拼装区一行（32B 对齐）
            tblRow_     = lay.tblRow;         // 拼装区一行的元素数（含补白）
            tileG_      = lay.tileG;
            superG_     = lay.superG;
            stageSpan_  = lay.stageSpan;      // **逐块 32B 对齐后的**暂存跨度
            asmSpan_    = lay.asmSpan;
            stagePitch_ = 0;                  // 打包布局用不到统一行距
            numIn_      = numIn;
            pipe->InitBuffer(stgBuf_, gBuf_ * stageSpan_);
            pipe->InitBuffer(asmBuf_, aBuf_ * asmSpan_);
            pipe->InitBuffer(offBuf_, lay.offTblBytes);
            pipe->InitBuffer(dltBuf_, lay.dltBytes);
            if constexpr (kI8I) {
                // 副本要覆盖整个暂存跨度（Cast 一条指令把 accEl 个元素整段搬上去），
                // 拼装的 half 结果按一个 tile 的行数算。
                pipe->InitBuffer(castBuf_, lay.castSpan);
                pipe->InitBuffer(asmGBuf_, lay.asmGBytes);
            }
        } else {
            // 一次分配，所有输入共用
            pipe->InitBuffer(que_, BUF_N, ubBudget_);
        }
    }

    /* 搬一个输入：它占输出的列区间 [offOut, offOut + nIn)。 */
    /* 一次搬运的全部参数。**从成员变量改成显式传参**，
     * 因为软件流水要同时持有「当前这趟」和「上一趟」两组参数，
     * 共用成员会被新的那趟覆盖掉。 */
    struct Job {
        __gm__ T *src;
        uint32_t nIn, offOut, c0, cols, rowUb, r0, cnt;
    };

    /* **软件流水：把写出延后一拍。**
     *
     * 原来是 `CopyIn(k); CopyOut(k);` 顺着来，而 CopyOut 开头要
     * `WaitFlag<MTE2_MTE3>` —— 标量单元卡在那里，于是
     * **MTE2(k+1) 必须等 MTE2(k) 整个落地才发得出去**，
     * 每趟的关键路径就是一次 MTE2 往返延迟。
     *
     * 实测正是这个形态：耗时几乎只跟**路数**走、不跟数据量走 ——
     * (128,4096) 2 MB / 123 路要 185.6 us，而 (2024,3000) **24 MB** / 88 路
     * 只要 142.8 us；每路稳定 ~1.5 us（1.8 GHz 下约 2700 拍，而一路只搬 1~2 KB）。
     *
     * Emit 展开后是：CopyIn(0) / CopyIn(1) CopyOut(0) / CopyIn(2) CopyOut(1) / …
     * 也就是**下一趟的装载先于本趟的写出**，等 MTE2(k) 时 MTE2(k+1) 已在飞。
     *
     * 缓冲仍是 BUF_N=2 就够：任一时刻最多两趟在手（k−1 已 EnQue 未 Free、
     * k 刚 Alloc），Emit 结束时立刻降回一趟。 */
    __aicore__ inline void Emit(const Job &j)
    {
        // **前瞻只在按输入切核（strategy 4）上开。** 这是实测分出来的：
        //
        //   形状            路数   前瞻后    基线    策略
        //   128,4096        123    22.91   185.61     4   ← 快 8.1 倍
        //   512,2048        248    77.21   373.79     4   ← 快 4.8 倍
        //   2024,3000        88   225.24   142.80     0   ← 慢 58%
        //   1024,1024        27    85.58    46.77     0   ← 慢 83%
        //   128,256           9     7.89     7.00     0   ← 慢 13%
        //
        // 完全按策略分野。原因是 BUF_N=2 时前瞻**挤掉了反向依赖的余量**：
        // CopyIn(k) 要等 MTE3_MTE2(k−2)，而那条写出在前瞻方案里是**上一步**
        // 才发出的（原来是两步前），搬运量大时还没落地就撞上去。
        // strategy 4 每核只有 3~6 趟、每趟 blockCount=rows 很大，不吃这个亏。
        //
        // 想两头都要就得加缓冲块（BUF_N≥3），但那要同比压小 ubBudget（UB 只有 192 KB），
        // 是另一组取舍，**要量过再定**。在那之前按实测走。
        // pipeMode：0=按策略自动、1=强制开、2=强制关（S9_PIPE 覆盖，便于 A/B）。
        /* ═══ 前瞻默认打开（`S9_PIPE=2` 关掉）═══
         *
         * **原先只有 strategy 4 默认开**，按行切（strategy 0）是关的 ——
         * 于是每一块都是「读 → 等 → 写」串起来，MTE2 与 MTE3 零重叠。
         *
         * 910B1 实测（单次 msprof，中位数，远超噪声）：
         *     32,512,16,64 f16 dim=1   458 → **601 GB/s**（1.31×）
         *     256,512,128  f16 dim=1   603 → **748**     （1.24×）
         *     128,1000,256 int8 dim=1  756 →   779       （1.03×）
         * 而 dim 在中间轴的形状占合法空间的 96.6%，正是走这一条。
         *
         * **为什么一直没发现**：早先只在 `16,1000,8,64`（38 µs 的小档）上量过，
         * 得到 0.963 而噪声地板 9% ⇒ 判成"无结论"就放过了。
         * 换成够大的形状才看得出来 —— 这是"测试集选错、真收益被噪声埋掉"的实例。 */
        const bool lookahead = (pipeMode_ != 2);
        if (!lookahead) { CopyIn(j); CopyOut(j); return; }
        CopyIn(j);
        if (hasPend_) { CopyOut(pend_); }
        pend_ = j;
        hasPend_ = true;
    }

    /* 收尾：把最后一趟的写出补上。**不能漏** —— 漏了那一趟的数据永远不落盘。 */
    __aicore__ inline void Flush()
    {
        if (hasPend_) { CopyOut(pend_); hasPend_ = false; }
    }

    __aicore__ inline void One(__gm__ T *src, uint32_t nIn, uint32_t offOut)
    {
        if (nIn == 0 || rows_ == 0) { return; }
        srcGm_.SetGlobalBuffer(src);
        // 输入的 L2 模式由 tiling 给（见 op_host 的说明）
        srcGm_.SetL2CacheHint(S9InCacheMode(l2Mode_));
        nIn_ = nIn; offOut_ = offOut;

        uint32_t beg, end, c0, c1;
        if (strategy_ == 1) {                       // 按列切：只有这条要按输入算
            // **每一路自己能撑几个核，就用几个核** —— 不再让所有路共用 bn_。
            //
            // 原先是 `per = AlignUp(CeilDiv(nIn_, bn_), alignEl_)`，也就是把**每一路**
            // 都切成 bn_ 份。路长差得多时这会把短路切碎：(2024,3000)/dim=0 的 57 路里
            // 最短一路 3000 个元素，40 核下每核只剩 75 个 ＝ 150 字节的 DMA。
            //
            // 真机上这一条是量出来的，不是推的：同样 12 MB、同样 40 个核，
            //   路路都大（8 路）  → 38.15 µs
            //   路有大有小（57 路）→ 146.0 µs
            // 差 3.8×。所以慢的不是核多，是短路被切碎。
            // 载荷由 host 给（COL_BURST_BYTES / esz），见文件头的说明
            uint32_t coresK = colBurstEl_ ? (nIn_ / colBurstEl_) : 1;
            if (coresK == 0)   { coresK = 1; }      // 短到撑不满一个核 ⇒ 整路给一个核
            if (coresK > bn_)  { coresK = bn_; }
            // **起点要轮转**。否则每一路都是 0..coresK-1 号核在干活，
            // 短路全压在低编号核上，高编号的核一点活都没有。
            //
            // 种子用**已经分出去的核数累计**（colBase_），不是 offOut_ ——
            // 主机孪生上试过 offOut_ % bn_，在真实数据上直接失效：
            // 各路的偏移常常都是 bn_ 的倍数（长度是 1000 的倍数、bn_=40），
            // 轮转量恒为 0，40 个核里 17 个白站着。累计核数则是紧密排布，
            // 且只依赖各路长度 —— 每个核自己算得出，不用通信。
            // colBase_ < bn_ 且 coresK ≤ bn_ ⇒ 和 < 2·bn_，一次条件减就够，不用取模。
            // 同理 bi_ < bn_、base < bn_ ⇒ bi_ + bn_ − base ∈ [1, 2·bn_)。
            uint32_t base = colBase_;
            colBase_ = S9AddMod(colBase_, coresK, bn_);   // **必须在提前返回之前推进**
            uint32_t rel = bi_ + bn_ - base;
            if (rel >= bn_) { rel -= bn_; }
            if (rel >= coresK) { return; }          // 这一路轮不到本核
            // coresK == 1 是最常见的情形（短路整路给一个核），直接跳过除法
            // **均衡切分，不是「统一份额 ＋ 末核吃剩下」。** 这是与列带同一类的坑：
            //   per = AlignUp(⌈nIn/coresK⌉, alignEl) 之后 per·coresK 可能远大于 nIn，
            //   于是尾巴上的几个核 c0 ≥ nIn **直接空转**。
            //   nIn=1000 个 fp32、alignEl=8、coresK=40 ⇒ per=32、32×40=1280 > 1000
            //   ⇒ 32~39 号核什么都不做，31 号只拿到 8 个元素。
            // 改成把 G = ⌈nIn/alignEl⌉ 个对齐粒度均分：第 rel 份拿
            //   [rel·G/coresK, (rel+1)·G/coresK)。各核最多差一个粒度，且没有空核
            //  （coresK = nIn/colBurstEl ≤ nIn/alignEl = G）。
            if (coresK == 1) {
                c0 = 0;
                c1 = nIn_;
            } else if (bal_ == 0) {          // 旧行为，只为配对 A/B 留着
                uint32_t per = S9AlignUpP2(S9CeilDiv(nIn_, coresK), alignMask_);
                c0 = per * rel;
                c1 = S9Min(c0 + per, nIn_);
            } else {
                const uint32_t G  = S9CeilDiv(nIn_, alignEl_);
                const uint32_t g0 = (uint32_t)((uint64_t)rel * G / coresK);
                const uint32_t g1 = (uint32_t)((uint64_t)(rel + 1) * G / coresK);
                c0 = g0 * alignEl_;
                c1 = S9Min(g1 * alignEl_, nIn_);
            }
            if (c0 >= nIn_ || c1 <= c0) { return; }
            beg = 0; end = rows_;
        } else {                                    // 按行切：Init 里已算好
            beg = rowBeg_; end = rowEnd_;
            if (colParts_ > 1) {
                /* 二维分核：本核只做这一路的第 `colIdx_` 份列。
                 * **按 32B 粒度数均分，不是「统一份额 ＋ 末核吃剩下」** ——
                 * 后者会让尾部的核 c0 ≥ nIn 直接空转（列带那边踩过同一个坑）。 */
                const uint32_t G  = S9CeilDiv(nIn_, alignEl_);
                const uint32_t g0 = (uint32_t)((uint64_t)colIdx_ * G / colParts_);
                const uint32_t g1 = (uint32_t)((uint64_t)(colIdx_ + 1) * G / colParts_);
                c0 = g0 * alignEl_;
                c1 = S9Min(g1 * alignEl_, nIn_);
            } else {
                c0 = 0; c1 = nIn_;
            }
        }
        if (beg >= end || c1 <= c0) { return; }

        // **列方向也要分块**：一行本身就可能比 UB 预算宽
        //（dim=0 时 rows≡1，(16,2048) fp32 单行就是 131072 字节）。colChunk_ 在 Init 里算好。
        for (uint32_t cc = c0; cc < c1; cc += colChunk_) {
            cols_ = S9Min(c1 - cc, colChunk_);
            c0_ = cc;
            // BLOCK_BYTES(32) 是 2 的幂 ⇒ 掩码对齐；再除以 sizeof(T)（编译期常量）
            rowUb_ = S9AlignUpP2(cols_ * sizeof(T), BLOCK_BYTES - 1) / sizeof(T);
            uint32_t tile = ubBudget_ / (rowUb_ * sizeof(T));
            if (tile == 0) { tile = 1; }
            if (tile > 4095) { tile = 4095; }        // blockCount 上限
            for (uint32_t r = beg; r < end; r += tile) {
                Job j;
                j.src = src; j.nIn = nIn_; j.offOut = offOut_;
                j.c0 = cc; j.cols = cols_; j.rowUb = rowUb_;
                j.r0 = r; j.cnt = S9Min(end - r, tile);
                Emit(j);
            }
        }
    }

    /* ═══ 拼装路径（strategy == 2）═══
     *
     * 常规路径每个输入单独走一趟 MTE2＋MTE3，**写是跨步的**：
     * 大 shape 上一次写 nIn·esz 字节、然后跳过 nOut·esz 字节，
     * 3 路输入就是 1/3 密度的三条跨步写流。读反而是完全连续的（srcStride=0）。
     * profiling 上 huge 是 MTE2 受限（mte2 0.91）、有效带宽 1.20 TB/s，
     * 而 HBM 峰值约 1.42 TB/s（ddr_rate=32 B/cycle × 24 核 × 1.85 GHz），
     * 官方在同一用例上跑到 2.08 TB/s —— 说明还有空间。
     *
     * 这里把一整行在 UB 里拼好，**3 条跨步写合成 1 条整行连续写**。
     *
     * **能这么做的前提是对齐**（`DataCopyPad` 要求 UB 侧地址与 stride 都按 32B）：
     *   ① 每个 offOut[i]·esz 是 32 的倍数 —— 输入 i 在 UB 行内的落点要对齐；
     *   ② UB 行距 pitch = alignUp(nOut·esz, 32) 本身是 32 的倍数（恒成立）；
     *   ③ (pitch − alignUp(nIn_i·esz, 32)) 是 32 的倍数（由 ① ② 推出）。
     * huge 满足（偏移 0/4000/8000 字节），big 不满足（偏移 2000 字节）——
     * **所以这是快路而不是唯一路**，由 host 判定，不满足就退回常规路径。
     *
     * 真机 A/B（910B3，S9_STRAT 强制两条路，各三次取中位数，单位 us）：
     *     shape       块数    常规    拼装
     *     longmid      165    3.72    3.80   +2%（噪声内，无收益）
     *     bigA        6072   13.42   12.89   −4%
     *     huge       12288   82.73   73.49   −11%
     *     shortbigA  60720   54.99   41.79   **−24%**
     * 收益随**块数**增长而不是字节数 —— 正因为省下的是「每行每输入一条指令」。
     * longmid 没收益是因为它标量受限（scalar 0.90、mte2 才 0.21），
     * 把写变连续帮不上忙。低于约 200 KB 时中性，没有实测到的回退。
     */
    __aicore__ inline void Assembled(__gm__ T **srcs, const uint32_t *nIns,
                                     const uint32_t *offs, uint32_t numIn)
    {
        if (rows_ == 0 || nOut_ == 0) { return; }
        uint32_t pitch = S9AlignUp(nOut_ * sizeof(T), BLOCK_BYTES);   // UB 行距（字节）
        uint32_t tile  = ubBudget_ / pitch;
        if (tile == 0) { return; }                    // 放不下整行 —— host 不该选这条路
        if (tile > 4095) { tile = 4095; }
        /* ═══ 前瞻（`S9_PIPE=1`）：把写延后一拍 ═══
         *
         * **这条路原先一点重叠都没有**：每个 tile 都是「读 numIn 条 → `DeQue` →
         * `WaitFlag<MTE2_MTE3>` 等本块读落地 → 写」串起来的。
         * 910B1 上实测（`16,256,32,64 f32 dim=2`，32 MB，走的正是这条路）：
         *     mte2_time 62.5 µs ＋ mte3_time 23.8 µs ＝ 86.3
         *     而 Task Duration 是 **137 µs** ⇒ 两条流水完全没有重叠
         *     `aiv_mte2_ratio` 只有 **0.455**，MTE2 一半时间在空转
         * 同一个张量换成 `dim=1`（走按行切）是 1635 GB/s，这一档只有 479 —— 3.4×。
         *
         * 前瞻之后第 i 块的读与第 i−1 块的写重叠，上限是
         * `max(mte2, mte3)` 而不是 `mte2 + mte3`。 */
        const bool la = (pipeMode_ == 1);
        bool     pend = false;
        uint32_t pb = 0, pr = 0, pcnt = 0;
        for (uint32_t r = rowBeg_; r < rowEnd_; r += tile) {
            uint32_t cnt = S9Min(rowEnd_ - r, tile);
            LocalTensor<T> ub = que_.template AllocTensor<T>();
            uint32_t b = it_ % BUF_N;
            if (it_ >= BUF_N) { WaitFlag<HardEvent::MTE3_MTE2>(EvOut(b)); }
            DataCopyPadExtParams<T> pad{true, 0, 0, (T)0};
            for (uint32_t i = 0; i < numIn; ++i) {
                if (nIns[i] == 0) { continue; }
                DataCopyExtParams q;
                q.blockCount = (uint16_t)cnt;
                q.blockLen   = nIns[i] * sizeof(T);
                q.srcStride  = 0;                     // 输入整块连续
                q.dstStride  = (pitch - S9AlignUp(q.blockLen, BLOCK_BYTES)) / BLOCK_BYTES;
                q.rsv        = 0;
                GlobalTensor<T> g;
                g.SetGlobalBuffer(srcs[i]);
                // NORMAL 删不得，见 GatherAssembled 里那段（删掉是 8~16 倍的回退）
                g.SetL2CacheHint(S9InCacheMode(l2Mode_));
                // 落点：UB 行内偏移 offs[i]（元素），由 host 保证 ·esz 后 32B 对齐
                DataCopyPad(ub[offs[i]], g[(uint64_t)r * nIns[i]], q, pad);
            }
            SetFlag<HardEvent::MTE2_MTE3>(EvIn(b));
            que_.EnQue(ub);

            if (!la) {
                AsmWrite(b, r, cnt, pitch);
            } else if (pend) {
                AsmWrite(pb, pr, pcnt, pitch);       // 写上一块，与本块的读重叠
            }
            if (la) { pb = b; pr = r; pcnt = cnt; pend = true; }
            ++it_;
        }
        if (la && pend) { AsmWrite(pb, pr, pcnt, pitch); }   // 收尾：最后一块不能漏
    }

    /* `Assembled` 的一次写出。抽成方法只为**前瞻与非前瞻共用同一份** ——
     * 抄两遍的话跨距算术会漂，而漂了不报错、只是输出错行。 */
    __aicore__ inline void AsmWrite(uint32_t b, uint32_t r, uint32_t cnt, uint32_t pitch)
    {
        LocalTensor<T> ob = que_.template DeQue<T>();
        WaitFlag<HardEvent::MTE2_MTE3>(EvIn(b));
        DataCopyExtParams w;
        // `S9_ONEWRITE`：pitch 恰好是一整行时，cnt 块紧挨着的整行 ≡ 一块（同一段地址）
        const bool one = oneWrite_ && pitch == nOut_ * sizeof(T);
        w.blockCount = one ? (uint16_t)1 : (uint16_t)cnt;
        w.blockLen   = one ? cnt * nOut_ * (uint32_t)sizeof(T) : nOut_ * (uint32_t)sizeof(T);   // **一整行**
        w.srcStride  = one ? 0u : (pitch - S9AlignUp(nOut_ * (uint32_t)sizeof(T), BLOCK_BYTES)) / BLOCK_BYTES;
        w.dstStride  = 0;                         // 输出行紧挨着 —— 连续写
        w.rsv        = 0;
        DataCopyPad(dstGm_[(uint64_t)r * nOut_], ob, w);
        SetFlag<HardEvent::MTE3_MTE2>(EvOut(b));
        que_.FreeTensor(ob);
    }

    /* ═══ 直写路径（`S9_DIRECT=1`，与 strategy 3 共用全部 tiling）═══
     *
     * **它与 `S9_GATHER=0` 那条退路不是一回事，这一点是本函数存在的全部理由。**
     *
     * `S9_GATHER=0` 退回的是常规路径 `S9RunRows`，那条路是**路在外、行在内**：
     * `One()` 里 `for (r = beg; r < end; r += tile)` 把**整条路的所有行**走完才换
     * 下一路。于是「路 i 写第 r 行」与「路 i+1 写第 r 行」之间隔了整整
     * `rows · nOut · esz` 字节的输出跨度 —— `512×2048 int32 248 路`那档
     * ＝ 512 行 × 8 KB = 4 MB/核 × 40 核 = **160 MB**，超过 L2。
     * 每一行的 cache line 在两路之间必被逐出，于是每路每行都是一次
     * read-modify-write，而每块才 8 个 int32 ＝ 32 B。
     * **实测的 1.8×~28.9× 退步量的是这个循环次序，不是"每路各写各的"本身。**
     *
     * 本函数把次序换成**行块在外、路在内**（与 `Assembled` 同）：
     * 路 i 与路 i+1 对同一行的写只隔 `tileG · align32(w·esz)` 字节的输出跨度
     *（每核几百 KB），行一直留在 L2 里，相邻路的窄写在 L2 内自然合并 ——
     * **不需要在 UB 里拼装，也就不需要 Gather。**
     *
     * 换来的是**每路的读块变大**。设 UB 预算 `U`、带宽 `W`、路宽 `L`：
     *   · Gather 路要同时装下**整条带**的暂存＋拼装区 ⇒ 每路读 `U·L/(2W)`；
     *   · 本路一次只装**一路** ⇒ 每路读 `U/2`（受 L2 驻留约束，但那一项有 ~50×
     *     裕度：`2·L2/(核数·U)` ＝ 2×192 MB/(40×192 KB)）。
     * 比值 ＝ `W/L` ＝ **每带路数**。已量过的路数扫描是「读块大 30× ⇒ 快 1.24×」，
     * 所以这一项按次线性外推是 ~1.2×，再加去掉向量开销与 4→2 次 UB 访问。
     *
     * **`directOn` 不参与任何 tiling 运算**：分核、分带、`tileG_`、UB 账与
     * Gather 路逐字节相同，所以配对 A/B 量的正好是这一个自变量。
     * 代价是本路拿到的 `tileG_` 是按 Gather 的 UB 账算的（偏小），
     * 也就是说**这一版量到的是收益的下界** —— 读块那一项还没兑现。
     *
     * 缓冲：借 `stgBuf_`（Gather 路的暂存区，共 `gBuf_ · stageSpan_` 字节），
     * 切成 `nb` 个槽，每槽 `tileG_ · align32(maxW·esz)`。`nb ≥ 2` 时走前瞻
     *（先发本轮的读、再发上一轮的写），读写才真正重叠。 */
    __aicore__ inline bool DirectBanded(__gm__ T **srcs, const uint32_t *nIns,
                                        const uint32_t *offs, uint32_t numIn)
    {
        if (rows_ == 0 || nOut_ == 0 || tileG_ == 0 || bw_ == 0) { return true; }
        if (rowBeg_ >= rowEnd_) { return true; }
        const uint32_t esz = (uint32_t)sizeof(T);

        /* ── 与本带相交的路压成密集数组 ──
         * 判据、夹取算术都与 `GatherAssembled` 用同一份（`S9RouteHitsBand` /
         * `ConcatBandClipWidth`），**不另写一遍** —— 两处漂开就是两条路看到
         * 不同的列区间，输出静默错列。
         * 就地压紧安全：`nAct ≤ i − iBeg ≤ i`，写回的是本槽或更前面的槽。 */
        uint32_t aN[S9_ASM_MAX], aW[S9_ASM_MAX], aSh[S9_ASM_MAX], aOut[S9_ASM_MAX];
        uint32_t nAct = 0, maxRow = 0;
        const uint32_t iBeg = planOn_ ? planLo_ : 0u;
        const uint32_t iEnd = planOn_ ? planHi_ : numIn;
        for (uint32_t i = iBeg; i < iEnd; ++i) {
            const uint32_t n = nIns[i], o = offs[i];
            if (!S9RouteHitsBand(n, o, cb0_, cb1_)) { continue; }
            const uint32_t lo = (o > cb0_) ? o : cb0_;
            const uint32_t cw = s9::ConcatBandClipWidth(n, o, cb0_, cb1_);
            if (cw == 0) { continue; }
            // UB 侧每块补到 32B（`dstStride=0` 的含义），槽宽按最宽的那一路算
            const uint32_t rb = S9AlignUpP2(cw * esz, BLOCK_BYTES - 1);
            if (rb > maxRow) { maxRow = rb; }
            aN[nAct] = n; aW[nAct] = cw; aSh[nAct] = lo - o; aOut[nAct] = lo;
            srcs[nAct] = srcs[i];
            ++nAct;
        }
        if (nAct == 0 || maxRow == 0) { return true; }

        /* ── 行块大小 ──
         * `directMode_ == 1`：沿用 Gather 的 `tileG_`（读块大小与 Gather 完全一样，
         *   于是 A/B 量到的纯粹是"拼不拼装"）。
         * `directMode_ == 2`：**把整个暂存区用满**。Gather 要同时装下整条带的
         *   全部路，直写一次只装一路，所以同样的 UB 能装下大得多的行块 ——
         *   放大倍数约是 `每带路数 × (平均路宽/最宽路宽)`。路长在 [0, max_step]
         *   上均匀 ⇒ 均值/最大 ≈ 0.5，所以约是「每带路数的一半」倍。
         * **只放大不缩小**：`tileG_` 本来就装得下，取 max 保证不会变差。 */
        const uint32_t total = gBuf_ * stageSpan_;
        const uint32_t myRows = rowEnd_ - rowBeg_;
        uint32_t tg = tileG_;
        if (directMode_ >= 2) {
            uint32_t t2 = (total / 3u) / maxRow;      // 先按 3 个槽留
            if (t2 < 1) { t2 = 1; }
            if (t2 > myRows) { t2 = myRows; }
            if (t2 > 4095)   { t2 = 4095; }           // blockCount 硬上限
            if (t2 > tg)     { tg = t2; }
        }
        /* **可行性护栏。** 一个槽至少要装得下一个行块，否则就是写出 UB 区域
         * （真机 507035，而且不一定当场炸）。`tileG_` 那一档理论上恒成立
         *（Gather 装得下全部路 ⇒ 更装得下一路），但窄路补到 32B 之后
         * `align32(w·esz)` 可能大于 `w·esz` 好几倍（w·esz=2 时是 16 倍），
         * 所以不能只靠推理。放不下就**逐次减半**，减到 1 行还放不下就
         * 交回 Gather 路 —— 这里一个字节都还没写，退回是安全的。 */
        while (tg > 1 && (uint64_t)tg * maxRow > (uint64_t)total) { tg >>= 1; }
        if ((uint64_t)tg * maxRow > (uint64_t)total) { return false; }
        const uint32_t slotB = tg * maxRow;
        uint32_t nb = total / slotB;
        if (nb == 0) { nb = 1; }
        if (nb > 3)  { nb = 3; }
        const bool la = (nb >= 2);          // 能前瞻才前瞻；nb==1 时读写只能串起来
        const uint32_t slotEl = slotB / esz;

        LocalTensor<T> stgAll = stgBuf_.template Get<T>();
        DataCopyPadExtParams<T> pad{false, 0, 0, (T)0};   // 补白不出门，不必清零
        const bool skipHint = (hint_ == 0) && (l2Mode_ == 1);
        const CacheMode cm = S9InCacheMode(l2Mode_);

        uint32_t it = 0;
        // 前瞻挂起的那一笔写（`la` 为真时用）
        bool hasPend = false;
        uint32_t pb = 0, pCnt = 0, pW = 0;
        uint64_t pDst = 0;
        for (uint32_t r = rowBeg_; r < rowEnd_; r += tg) {
            const uint32_t cnt = S9Min(rowEnd_ - r, tg);
            for (uint32_t j = 0; j < nAct; ++j) {
                const uint32_t b = (nb == 1) ? 0u : (it % nb);
                // 这个槽上一次的写出完了没有
                if (it >= nb) { WaitFlag<HardEvent::MTE3_MTE2>(EvB(b)); }
                GlobalTensor<T> g;
                g.SetGlobalBuffer(srcs[j]);
                if (!skipHint) { g.SetL2CacheHint(cm); }
                DataCopyExtParams q;
                q.blockCount = (uint16_t)cnt;
                q.blockLen   = aW[j] * esz;
                q.srcStride  = (aN[j] - aW[j]) * esz;   // GM 侧，单位字节
                q.dstStride  = 0;                       // UB 侧，逐块补到 32B 紧密排
                q.rsv        = 0;
                DataCopyPad(stgAll[b * slotEl], g[(uint64_t)r * aN[j] + aSh[j]], q, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EvB(b));

                if (la) {
                    if (hasPend) { DirectWrite(stgAll, pb, slotEl, pCnt, pW, pDst); }
                    pb = b; pCnt = cnt; pW = aW[j];
                    pDst = (uint64_t)r * nOut_ + aOut[j];
                    hasPend = true;
                } else {
                    DirectWrite(stgAll, b, slotEl, cnt, aW[j],
                                (uint64_t)r * nOut_ + aOut[j]);
                }
                ++it;
            }
        }
        if (hasPend) { DirectWrite(stgAll, pb, slotEl, pCnt, pW, pDst); }

        /* 收尾：`MTE3_MTE2` 设了 `it` 次（每笔写一次）、在循环头等了
         * `max(0, it − nb)` 次 ⇒ 还要补 `min(it, nb)` 次。
         * **悬空 flag 单次调用看不出来，harness 跑 30 轮会累积成挂死。** */
        const uint32_t nd = (it < nb) ? it : nb;
        for (uint32_t k = 0; k < nd; ++k) {
            WaitFlag<HardEvent::MTE3_MTE2>(EvB((it - 1 - k) % nb));
        }
        return true;
    }

    /* 直写路的一笔写出。**读与写的 UB 口径必须同源**：读用 `dstStride = 0`
     * ⇒ 每行补到 32B 后紧密排 ⇒ 写就得 `srcStride = 0`。
     * `dstStride` 在 GM 侧、单位字节（DataCopyPad_UBToGM）。 */
    __aicore__ inline void DirectWrite(LocalTensor<T> stgAll, uint32_t b, uint32_t slotEl,
                                       uint32_t cnt, uint32_t w, uint64_t dstEl)
    {
        WaitFlag<HardEvent::MTE2_MTE3>(EvB(b));
        DataCopyExtParams p;
        p.blockCount = (uint16_t)cnt;
        p.blockLen   = w * (uint32_t)sizeof(T);
        p.srcStride  = 0;
        p.dstStride  = (nOut_ - w) * (uint32_t)sizeof(T);
        p.rsv        = 0;
        DataCopyPad(dstGm_[dstEl], stgAll[b * slotEl], p);
        SetFlag<HardEvent::MTE3_MTE2>(EvB(b));
    }

    /* ═══ 向量拼装路径（strategy == 3）═══
     *
     * DMA 拼装（strategy 2）要求每个 offOut[i]·esz 是 32 的倍数，
     * mid（偏移 0/110/220 字节）这类「行短且不对齐」的形状用不上，
     * 只能退回常规路径 —— 于是每行每路各发一条 DataCopyPad 块，
     * profiling 上就是标量/搬运双高而带宽很低（mid 最优点 scalar 0.55、mte2 0.53）。
     *
     * `Gather` 按**元素粒度**收集：`dst[i] = src[(srcBaseAddr + srcOffset[i]) / sizeof(T)]`，
     * 偏移是逐元素的字节量（文档《Gather》），**没有 32B 落点约束**。
     * 于是：numIn 条连续读 → 一次 Gather 拼成整行 → 一条连续写。
     *
     * **偏移表为什么能只建一次**：暂存区里每一路都用同一个行距 stagePitch_，
     * 所以「输出行内第 j 个元素在暂存区的字节偏移」对所有行只差一个 r·stagePitch_，
     * 把它交给 srcBaseAddr 即可，表本身与 r 无关。
     *
     * **表是全向量建的**（见下面 GatherAssembled 里的窗口循环）。
     * 曾经的说法是「每一路的段起点 offOut[i]·4 不满足 32B 对齐，向量指令写不到那个
     * 位置，所以只能标量」—— 那条只对**不带 mask** 的向量接口成立。
     * 《Adds》《Duplicate》在 A2 上都有逐 bit mask 的高维切分版：
     * 落点仍按 32B 对齐（取 64 通道窗口的首地址），**参与计算的通道由 mask 选**，
     * 于是任意起止的一段都能一条指令写完。
     */
    /* 偏移表的分段基址：off[cl] = (cl + stBaseEl + cb0 − offOut)·gsz。
     * 全部按**元素**算完再乘载体宽度 gsz —— int8 借道 half 时暂存区的字节偏移
     * 是元素下标的 2 倍，而 DMA 落点仍按 int8 算，两个口径必须分开。
     * cb0_ 可能小于 offOut（这一路从带中间开始），所以必须走有符号运算：
     * 用无符号会绕成天文数字，Gather 直接读到暂存区外。 */
    __aicore__ inline int32_t S9TblBase(uint32_t stBaseEl, uint32_t offOut, uint32_t gsz) const
    {
        return ((int32_t)stBaseEl + (int32_t)cb0_ - (int32_t)offOut) * (int32_t)gsz;
    }

    /* `tblK` 模式下的一趟拼装：每 chunk 铺 K 行，发**一条**覆盖 K 行的 `Gather`。
     *
     * 行布局见 `GatherAssembled` 里建表那一段。每 chunk 的向量指令是
     * 「1 条 cur→work[0] ＋ (m−1) 条 work[i]=work[i−1]+dlt ＋ 1 条 cur+=dltK」，
     * 一个 tile 合计约 `cnt + 2·cnt/K` 条 —— 与摊开 tileG 行时的 `cnt` 条同量级，
     * 而表的行数从 `tileG` 降到 `K + 3`。
     *
     * **`dst`/`src` 是模板参数**：int8 借道 half 时两边都是 `half`，其余是 `T`。
     * 两个调用点共用这一份 —— 抄两遍就会在某一次改动里漂开，而漂开的后果是
     * 输出静默错列（Gather 不越界、不报错）。 */
    template <typename D, typename Sx>
    __aicore__ inline void GatherChunk(LocalTensor<D> dst, LocalTensor<Sx> src,
                                       LocalTensor<int32_t> off, LocalTensor<int32_t> dlt,
                                       uint32_t cnt, uint32_t K)
    {
        LocalTensor<uint32_t> offU = off.template ReinterpretCast<uint32_t>();
        constexpr uint32_t W = 3;                                   // 工作区起始行
        Adds(off[tblRow_], off, (int32_t)0, (int32_t)tblRow_);      // cur ← 母版
        for (uint32_t j = 0; j < cnt; j += K) {
            const uint32_t m = S9Min(K, cnt - j);
            Adds(off[W * tblRow_], off[tblRow_], (int32_t)0, (int32_t)tblRow_);
            for (uint32_t i = 1; i < m; ++i) {
                Add(off[(W + i) * tblRow_], off[(W + i - 1) * tblRow_], dlt, (int32_t)tblRow_);
            }
            Gather(dst[j * tblRow_], src, offU[W * tblRow_], 0, tblRow_ * m);
            if (j + K < cnt) {
                Add(off[tblRow_], off[tblRow_], off[2 * tblRow_], (int32_t)tblRow_);
            }
        }
    }

    /* 带内每一路一条连续 `DataCopyPad`（整行入暂存）—— 读循环本体。
     *
     * **抽成方法是为了序幕重叠**（`S9_PRERD`）：`preRd` 时第 0 趟的读要在
     * **建偏移表之前**发出去。读不依赖表，而建表是向量活，两条流水本可以并行；
     * 现在的发射顺序却是「标量压紧 ＋ 向量建表 → MTE2 读」，于是建表期间
     * MTE2 完全空着。profiling 量到 `aiv_scalar_time = 78.8 µs`，而
     * `Task Duration − aiv_time = 74.5 µs` —— 两个数几乎相等。
     *
     * 抄第二份是不行的：这段里三处跨步算术（`srcStride`、`dstStride`、
     * `blockCount` 的两条分支）漂开一处就是输出静默错列。 */
    __aicore__ inline void GatherReadTile(LocalTensor<T> stg, uint64_t r, uint32_t cnt,
                                          __gm__ T **srcs, const uint32_t *aN,
                                          const uint32_t *aW, const uint32_t *aShift,
                                          const uint32_t *stBase, uint32_t nAct,
                                          bool skipHint, CacheMode cm)
    {
        constexpr uint32_t esz = (uint32_t)sizeof(T);
        DataCopyPadExtParams<T> pad{true, 0, 0, (T)0};
        DataCopyExtParams q;
        q.rsv = 0;
        for (uint32_t j = 0; j < nAct; ++j) {
            GlobalTensor<T> g;
            g.SetGlobalBuffer(srcs[j]);
            if (!skipHint) { g.SetL2CacheHint(cm); }
            if (aShift[j] != 0 || aW[j] != aN[j]) {
                /* **边界路：只搬相交的那一段。** 原来是整行入暂存
                 *（`blockCount=1、blockLen=cnt·nIn·esz`），带外的列白搬 ——
                 * 一条带最多切开两条路，那多搬的部分正是"冗余读"。
                 *
                 * 现在按需跨步读：`blockCount=cnt` 行、每行 `W·esz` 字节、
                 * 源行距 `(nIn−W)·esz`（GM 侧单位是字节），目的侧
                 * `dstStride=0` —— 框架会把每块补到 32B 再紧密排，
                 * 于是紧凑槽的行距恰好是 `align32(W·esz)`，与暂存前缀和
                 * （`ConcatRoutePitchEl`）和表里的 dlt 三处同一口径。
                 *
                 * `blockLen=cnt·nIn·esz` 那条老路**只留一行指令**，所以
                 * 指令条数不变（每路仍是一条 DataCopyPad）。 */
                q.blockCount = (uint16_t)cnt;
                q.blockLen   = aW[j] * esz;
                q.srcStride  = (aN[j] - aW[j]) * esz;
                q.dstStride  = 0;
                DataCopyPad(stg[stBase[j]], g[(uint64_t)r * aN[j] + aShift[j]], q, pad);
            } else {
                q.blockCount = 1;
                q.blockLen   = cnt * aN[j] * esz;
                q.srcStride  = 0;
                q.dstStride  = 0;
                DataCopyPad(stg[stBase[j]], g[(uint64_t)r * aN[j]], q, pad);
            }
        }
    }

    __aicore__ inline void GatherAssembled(__gm__ T **srcs, const uint32_t *nIns,
                                           const uint32_t *offs, uint32_t numIn)
    {
        if (rows_ == 0 || nOut_ == 0 || tileG_ == 0 || bw_ == 0) { return; }
        if (rowBeg_ >= rowEnd_) { return; }        // 本核这一行组是空的
        const uint32_t esz = (uint32_t)sizeof(T);

        // Gather 的载体：int8 借道 half（见文件头 S9GatherAs 的文档依据），其余就是 T 本身
        using G = typename S9GatherAs<T>::type;
        constexpr uint32_t gsz  = (uint32_t)sizeof(G);
        constexpr bool     kI8  = (sizeof(T) == 1);

        // ── 只有与本核列带 [cb0_, cb1_) 相交的路才要处理 ──
        // 相交的路一律**整行入暂存**（blockCount=1 的一条连续 DMA），
        // 于是行步长恒为 nIn 个元素，被带切开的那两条路也不例外。
        // 只有 stBase 要存数组（它是前缀和，算不回来），**单位是元素不是字节** ——
        // int8 借道 half 时同一个元素下标在两个缓冲里对应不同的字节偏移。
        //
        // **与带相交的路是连续的一段下标。** offs[] 是 nIn 的前缀和、单调不减，
        // 所以「offs[i] < cb1 且 offs[i]+nIn[i] > cb0」的 i 必然连成一段。
        // 把这段 [iLo, iHi) 算一次（O(numIn)，每核一次），后面每个 tile 的读循环
        // 就只走这一段，而不是把全部 numIn 路都过一遍做区间判断。
        //
        // 这一条是 msprof 逼出来的：(4096,16384) fp16 246 路那档
        // **aiv_scalar_ratio 0.97**（462 µs / 476），而 vec 只有 196、mte2 218。
        // 每核 128 个 tile × 246 次迭代 ＝ 31488 次，其中约 90% 是"这一路不在我带里"。
        uint32_t stBase[S9_ASM_MAX];
        uint32_t accEl = 0;
        uint32_t iLo = numIn, iHi = 0;
        /* **边界路裁剪**（`S9_TRIM`）：被带切开的那 ≤2 条路只把相交的那一段搬进来，
         * 所以它们的**行距**不再是整路宽度，而是 `align32(W·esz)` 个元素
         * —— `DataCopyPad(blockCount=cnt, blockLen=W·esz, dstStride=0)` 每行落在
         * 32B 边界上的充要条件（文档：dstStride 的单位是 32B dataBlock，
         * UB 侧目的地址必须 32B 对齐）。整段在带内的路一个字都不变。
         *
         * 暂存跨度与**host 的 UB 账必须同源**：host 走 `ConcatBandStagedBytes`
         * （逐路行宽之和），这里逐路累加 —— 两边算出的行宽不同就是
         * "账算得下、真机放不下"（507035）。 */
        /* 单路行距：直接调共用算术，**不套 lambda** —— bisheng 在 `__aicore__`
         * 函数里对闭包的支持不明确（本次一试就报 "no matching function for call to
         * object of type (lambda)"），而这里本来也不需要闭包。 */
        /* 扫哪一段、要不要顺手记 [iLo, iHi)，是**唯一的**一处差别：
         *   · `planOn_`（每带计划，默认）：host 已经把本带的路由区间算好了，
         *     `stBase[iLo]` 天然是 0 —— iLo 之前没有任何一路与这条带相交，
         *     暂存区里也只放相交的路 ⇒ 只扫 [planLo_, planHi_) 就够；
         *   · 没有计划：只能扫全部 numIn 路，边扫边把相交的那一段记成 [iLo, iHi)。
         * 原先这是两段几乎逐字相同的循环（各抄一遍同一个累加式），
         * 现在合成一段 —— 两个分支的累加式不可能再漂开。 */
        const uint32_t iBeg = planOn_ ? planLo_ : 0u;
        const uint32_t iEnd = planOn_ ? planHi_ : numIn;
        if (planOn_) { iLo = planLo_; iHi = planHi_; }
        // 段起点要同时满足两边的 32B 对齐：DMA 落在 T 缓冲（每元素 esz 字节）、
        // Gather 读载体缓冲（每元素 gsz 字节）。**按窄的那个算**才两边都成立 ——
        // int8 借道 half 时 esz=1、gsz=2，取 32/2=16 个元素只有 16 字节，
        // DataCopyPad 的 UB 落点就不是 32B 对齐了；取 32/1=32 个元素才对。
        constexpr uint32_t alignEl = (BLOCK_BYTES / ((esz < gsz) ? esz : gsz));
        for (uint32_t i = iBeg; i < iEnd; ++i) {
            stBase[i] = accEl;
            if (!S9RouteHitsBand(nIns[i], offs[i], cb0_, cb1_)) { continue; }
            if (!planOn_) {
                if (i < iLo) { iLo = i; }
                iHi = i + 1;
            }
            accEl += S9AlignUpP2(
                tileG_ * s9::ConcatRoutePitchEl(esz, nIns[i], offs[i], cb0_, cb1_, trim_),
                alignEl - 1);
        }

        if (iLo >= iHi) { return; }        // 本核这一带没有任何一路（末带可能整条空）

        // ── 第 0 行的两张表（下标是**带内相对列** cl = c − cb0_）──
        //   off[cl]：该列在暂存区第 0 行的字节偏移 ＝ cl·esz + (stBase[k] + (cb0−off_k)·esz)
        //   dlt[cl]：该列所属输入的行步长 ＝ nIn[k]·esz
        // 补白列（cl ≥ bw_）：off 留着 cl·esz、dlt 置 0 —— 落在暂存区内且逐行不动。
        //
        // **每一路整段都用带 mask 的向量写，没有标量路。**
        //
        // 原来是「32B 对齐的主体走 Adds/Duplicate，首尾零头逐列 SetValue」——
        // int32 一个 DataBlock 是 8 个元素，所以每路最多 7＋7 列零头、两张表就是 28 次
        // 标量 UB 写。msprof 把这笔账摆出来了（2048×2048 fp16，**恒 8 MB 只改路数**）：
        //     248 路  scalar 26.96 µs · vec 14.14 · 总 39.72
        //      16 路  scalar 10.31 µs · vec 12.91 · 总 20.58
        // 多出来的 232 路里，**+16.65 µs 全在标量**，向量只多 1.23 —— 正是这段零头。
        //
        // 《Adds》《Duplicate》在 A2 上都有**逐 bit mask 的高维切分版**，
        // 且 dst 只要 32B 对齐。int32 一次迭代处理 **64 个元素**，所以按 64 通道
        // 的窗口切：窗口首地址 w·4 必是 256 的倍数，天然对齐；
        // 一路落在一个窗口内（STEP=16 时平均才 8 列）就只要**两条**向量指令。
        //
        // 顺带把 V→S / S→V 两道同步也删了 —— 没有标量段就不需要它们。
        LocalTensor<int32_t> off = offBuf_.template Get<int32_t>();
        LocalTensor<int32_t> dlt = dltBuf_.template Get<int32_t>();
        // 底子：off 先铺成 cl·gsz 的斜坡、dlt 清零。补白列就停在这个底子上。
        CreateVecIndex(off, 0, (int32_t)tblRow_);
        Muls(off, off, (int32_t)gsz, (int32_t)tblRow_);
        Duplicate(dlt, (int32_t)0, (int32_t)tblRow_);
        constexpr uint32_t kWin = 64;                    // int32 一次迭代的元素数
        const UnaryRepeatParams kRep{1, 1, 8, 8};
        /* ── 同一趟里顺手把「与本带相交且非空」的路压成密集下标 ──
         * 热循环里每路每 tile 要做 3 次数组取 ＋ 3 次比较 ＋ 一个分支来判断
         * "这一路在不在我的带里"，而**这个判断与 r 无关**，每个 tile 都重算一遍。
         * 248 路那档每核约 4 个 tile ⇒ 白做约 750 次。就地压紧（nAct ≤ i 恒成立），
         * 不额外占栈：aN[] 是唯一新增的一份（S9_ASM_MAX·4 ＝ 1 KB）。
         *
         * **压紧原先是紧跟其后的第二趟同样的循环** —— 同一个
         * `S9RouteHitsBand` 判据、同一段夹取算术各抄一遍，而"裁没裁"
         * 一边问 `ConcatRoutePitchEl != n`、另一边问 `ConcatRouteTrimmed`。
         * 两处只要有一处漂了，表铺的是 A 集合、读循环走的是 B 集合，
         * 输出**静默错列**。合成一趟之后不可能再漂。 */
        uint32_t aN[S9_ASM_MAX];
        uint32_t aW[S9_ASM_MAX];        // 本带里这一路的**相交宽度**（元素）
        uint32_t aShift[S9_ASM_MAX];    // 相交段在整行里的起点（元素，0 ＝ 整段在带内）
        uint32_t nAct = 0;
        for (uint32_t i = iLo; i < iHi; ++i) {
            const uint32_t n = nIns[i], o = offs[i];
            if (!S9RouteHitsBand(n, o, cb0_, cb1_)) { continue; }
            // 夹取走共用算术，只算一次：lo 是相交段起点，cw 是它的宽度。
            // 相交（`S9RouteHitsBand` 为真、带非空）⇒ cw > 0。
            const uint32_t lo = (o > cb0_) ? o : cb0_;
            const uint32_t cw = s9::ConcatBandClipWidth(n, o, cb0_, cb1_);
            const uint32_t l0 = lo - cb0_, l1 = l0 + cw;
            /* **表基址用的"输入起点"要跟着裁剪走。**
             * 未裁剪（或这一路整段在带内、或它是右边界）：暂存行里是整条路，
             * 列 `c` 落在元素 `stBase + (c − offs[i])` ⇒ 基址照旧传 `offs[i]`。
             * 裁剪后的**左边界路**：紧凑行只从 `lo` 开始存 ⇒ 列 `c` 落在
             * `stBase + (c − lo)` ⇒ 基址要传 `lo`（不然会指到槽前一个路的地盘里，
             * 甚至指到缓冲之外 —— 而且**不报错**）。 */
            const uint32_t pitchEl = s9::ConcatRoutePitchEl(esz, n, o, cb0_, cb1_, trim_);
            /* **"裁没裁"只问行距这一个量。** `ConcatRoutePitchEl` 只在
             * `trim && ConcatRouteTrimmed` 时才改行距，而 `ConcatRouteTrimmed`
             * 的判据本身就是 `align32(w·esz) < n·esz` ⇒ 改了必然严格变小。
             * 所以 `pitchEl != n` ⟺ `trim_ && ConcatRouteTrimmed(…)`，
             * 不必再把后者算第二遍。 */
            const bool bnd = (pitchEl != n);
            const int32_t k = S9TblBase(stBase[i], bnd ? lo : o, gsz);
            // 行步长取**这一路自己的**（紧凑槽是补到 32B 的行距，不是整路宽度）
            const int32_t d = (int32_t)(pitchEl * gsz);
            // 超行第kk行在输出表中紧邻上一行；输入仍按该路自己的行距推进。
            // 官网 Gather 按字节偏移收集元素；超行表起点仍32B对齐。
            for (uint32_t kk = 0; kk < superG_; ++kk) {
              const uint32_t first = l0 + kk * bw_, last = l1 + kk * bw_;
              const int32_t base = k + (int32_t)kk * ((int32_t)pitchEl - (int32_t)bw_) * (int32_t)gsz;
              for (uint32_t w = first & ~(kWin - 1); w < last; w += kWin) {
                const uint32_t s0 = (first > w) ? (first - w) : 0;
                const uint32_t e0 = ((last - w) < kWin) ? (last - w) : kWin;
                if (e0 <= s0) { continue; }
                // 低 64 bit 逐位对应本次迭代的 64 个 int32 通道；高 64 bit 必须清零。
                // `1ULL << 64` 是未定义行为，满窗要单独写成 ~0ULL。
                uint64_t m0 = ((e0 - s0) >= kWin) ? ~(uint64_t)0
                                                  : ((((uint64_t)1 << (e0 - s0)) - 1) << s0);
                uint64_t msk[2] = {m0, 0};
                Adds(off[w], off[w], base, msk, (uint8_t)1, kRep);
                Duplicate(dlt[w], d * (int32_t)superG_, msk, (uint8_t)1, (uint16_t)1, (uint8_t)8);
              }
            }
            /* 压紧**必须排在读完 stBase[i]/srcs[i] 之后**：nAct ≤ i，
             * 写回的是本槽或更前面的槽，本轮要用的那两个值已经取出来了。
             * 只有"裁完更省"的路走紧凑槽（窄路补到 32B 反而更大，
             * `ConcatRouteTrimmed` 已经把它们挡在外面）；不裁的路
             * aShift=0、aW=n ⇒ 读循环走原来的整行连续读。 */
            aN[nAct]     = n;
            aW[nAct]     = bnd ? cw : n;
            aShift[nAct] = bnd ? (lo - o) : 0;
            stBase[nAct] = stBase[i];
            srcs[nAct]   = srcs[i];
            ++nAct;
        }
        // **这里原来有一对 S→V 同步**，用来让标量补的那些列对后面的向量可见。
        // 标量段没有了，依赖也就没有了：建表与展开都在向量流水上，天然有序。
        // ── 把第 0 行推成 tileG_ 行 —— 每行一条向量 Add，全程只做一次 ──
        // off[j·tblRow] = off[(j−1)·tblRow] + dlt。tblRow_·4 必是 32 的倍数
        //（outPitch_ 是 32B 对齐的，除以 esz 再乘 4 仍然对齐），落点合法。
        /* `tbl1_`：**不摊开**。表只保留第 0 行（原样，当母版），
         * 每个 tile 把母版拷进第 1 行当工作行，逐行 Gather、逐行就地 `Add(dlt)`。
         * 加法次数与摊开时一模一样（tileG−1 次），Gather 的元素总数也一样，
         * 只是 Gather 从「一条覆盖 cnt 行」变成「cnt 条各覆盖 1 行」。
         * 需要 tileG ≥ 2（要有第 1 行放工作副本）；否则退回摊开（它本来也只有 1 行）。 */
        const bool skipHint = (hint_ == 0) && (l2Mode_ == 1);
        const CacheMode cm = S9InCacheMode(l2Mode_);
        LocalTensor<T> stgAll = stgBuf_.template Get<T>();
        const uint32_t stgSpanEl = stageSpan_ / esz;

        /* ═══ 序幕重叠（`S9_PRERD`）═══
         *
         * **第一趟的读不依赖偏移表**，而表是向量活、读是 MTE2 —— 两条流水本可以
         * 并行。现在的发射顺序却是「压紧 ＋ 建表 → 读」，于是建表期间 MTE2 空着。
         * 把第 0 趟提到这里发，MTE2 就在建表的同时跑起来。
         *
         * 这是"把计算移出测量窗口"的**正确形式：重叠，不是删掉**。
         * 上限就是实测的那 21% 空转（`Task Duration 409.2` vs `aiv_time 334.7`）。
         *
         * 与循环里**共用同一份** `GatherReadTile` —— 抄第二份就会漂。
         * 循环里第 0 趟靠 `preRd` 跳过（连同它的 `SetFlag`）。 */
        const bool preRd = preRd_ && (rowBeg_ < rowEnd_);
        if (preRd) {
            const uint32_t cnt0 = S9Min(rowEnd_ - rowBeg_, tileG_);
            GatherReadTile(stgAll, (uint64_t)rowBeg_, cnt0, srcs, aN, aW, aShift, stBase,
                           nAct, skipHint, cm);
            SetFlag<HardEvent::MTE2_V>(EvB(0));
        }

        /* `tblK_ != 0`：**不摊开 tileG 行**，只留 `K + 3` 行 ——
         *   行 0        母版（与 r 无关）
         *   行 1        cur：母版推进到本 chunk 的起始行
         *   行 2        dltK ＝ dlt·K
         *   行 3..3+K-1 工作区（本 chunk 的 K 行）
         * 每个 chunk 铺 K 行、发**一条**覆盖 K 行的 `Gather`，chunk 之间 `cur += dltK`。
         * K=1 时退化成逐行 —— 那一档实测在 bw≈1638 上快 2.4%、在 bw≈1024 上慢 17.3%
         *（per-row `Gather` 的每条调用固定开销），所以 K 要能调。 */
        /* **判据与 UB 账逐字相同**（`s9_concat_layout.h` 的 `tblRows`）：
         * 只有 `K + 3 < tileG` 才值得分块，否则表比摊开还大。 */
        const uint32_t K = (tblK_ && tblK_ + s9::S9L_TBL_EXTRA_ROWS < tileG_)
                         ? S9Min(tblK_, tileG_) : 0u;
        if (K == 0) {
            for (uint32_t j = 1; j < (tileG_ + superG_ - 1) / superG_; ++j) {
                Add(off[j * tblRow_], off[(j - 1) * tblRow_], dlt, (int32_t)tblRow_);
            }
        } else {
            Muls(off[2 * tblRow_], dlt, (int32_t)K, (int32_t)tblRow_);
        }
        LocalTensor<uint32_t> offU = off.template ReinterpretCast<uint32_t>();

        DataCopyPadExtParams<T> pad{true, 0, 0, (T)0};

        // ── L2 模式只算一次；`hint_ == 0` 时对 NORMAL 干脆不调 ──
        //
        // **源码证明：A2 上 `SetL2CacheHint(NORMAL)` 是空操作。**
        //   `kernel_tensor.h:294`      CacheMode cacheMode_ = CACHE_MODE_NORMAL;（类内默认）
        //   `kernel_tensor_impl.h:1277` SetGlobalBuffer: cacheMode_==NORMAL ⇒ address_ = buf；
        //                               oriAddress_ = buf
        //   `kernel_tensor_impl.h:1700` SetL2CacheHint(NORMAL): cacheMode_ = NORMAL；
        //                               address_ = oriAddress_
        // 新构造的 GlobalTensor 调完 SetGlobalBuffer 之后，再调一次 NORMAL **逐字段相同**。
        //
        // 但这与文档里记的「删掉是 8~16 倍回退（G16 503 → 4066）」直接冲突 ——
        // 那次一定还动了别的东西，或者测量被污染了。**不凭读源码就删**，
        // 留 S9_HINT 做配对 A/B：为 0 时只在 l2Mode≠1 才调。
        LocalTensor<T> asmAll = asmBuf_.template Get<T>();
        const uint32_t asmSpanEl = asmSpan_ / esz;

        uint32_t it = 0;
        for (uint32_t r = rowBeg_; r < rowEnd_; r += tileG_, ++it) {
            uint32_t cnt = S9Min(rowEnd_ - r, tileG_);
            const uint32_t gatherCount = (superG_ > 1 ? bw_ : tblRow_) * cnt;
            const uint32_t ub = (gBuf_ == 1) ? 0u : (it % gBuf_);
            const uint32_t ab = (aBuf_ == 1) ? 0u : (it % aBuf_);
            LocalTensor<T> stg  = stgAll[ub * stgSpanEl];
            LocalTensor<T> asmb = asmAll[ab * asmSpanEl];

            // ① 读：带内每一路一条连续 DataCopyPad（整行入暂存）
            //    等这块暂存上一次被向量读完（第 it−gBuf_ 趟）
            //    **`preRd` 时第 0 趟已经在建表之前发过了**，这里跳过。
            if (!(it == 0 && preRd)) {
                if (it >= gBuf_) { WaitFlag<HardEvent::V_MTE2>(EvB(ub)); }
                GatherReadTile(stg, r, cnt, srcs, aN, aW, aShift, stBase, nAct,
                               skipHint, cm);
                SetFlag<HardEvent::MTE2_V>(EvB(ub));
            }



            // ② 拼：**整个 tile 一次 Gather**，表里已经含了这 cnt 行
            //    int8 借道 half：整段一条 Cast 上去、Gather、再整段一条 Cast 回来。
            //    两条 Cast 都是**每 tile 一条指令**（不是每路一条）。
            WaitFlag<HardEvent::MTE2_V>(EvB(ub));
            if (it >= aBuf_) { WaitFlag<HardEvent::MTE3_V>(EvB(ab)); }
            if constexpr (kI8) {
                LocalTensor<G> srcG = castBuf_.template Get<G>();
                LocalTensor<G> asmG = asmGBuf_.template Get<G>();
                Cast(srcG, stg, RoundMode::CAST_NONE, (int32_t)accEl);
                if (K) { GatherChunk(asmG, srcG, off, dlt, cnt, K); }
                else    { Gather(asmG, srcG, offU, 0, gatherCount); }
                // int8 的值域是 ±127，在 half 里是精确整数 ⇒ 舍入模式取哪个都一样。
                Cast(asmb, asmG, RoundMode::CAST_RINT, (int32_t)gatherCount);
            } else if (K) {
                GatherChunk(asmb, stg, off, dlt, cnt, K);
            } else {
                Gather(asmb, stg, offU, 0, gatherCount);
            }
            SetFlag<HardEvent::V_MTE2>(EvB(ub));      // 暂存块可以被下一趟覆盖了
            SetFlag<HardEvent::V_MTE3>(EvB(ab));

            // ③ 写：cnt 条跨步块，每条就是本核这一带
            //    srcStride 在 UB 侧、单位 32B block；dstStride 在 GM 侧、单位字节
            //    （DataCopyPad_UBToGM.md:114-115）。不分带时 dstStride=0，退回整行连续写。
            WaitFlag<HardEvent::V_MTE3>(EvB(ab));
            DataCopyExtParams w;
            /* **单带、行宽恰为 32B 整数倍 ⇒ 一块写完**（`S9_ONEWRITE`）。
             * 此时 `bw_ == nOut_`（dstStride=0，输出行首尾相接）且 `outPitch_ == bw_·esz`
             *（srcStride=0，拼装区行首尾相接），`cnt` 块紧挨着的 `bw·esz` 与一块 `cnt·bw·esz`
             * 搬的是**同一段地址**。910B1：每写一块约 40 ns、与字节数无关。 */
            const bool one = (oneWrite_ || superG_ > 1) && bands_ == 1 && bw_ == nOut_ && outPitch_ == bw_ * esz;
            w.blockCount = one ? (uint16_t)1 : (uint16_t)cnt;
            w.blockLen   = one ? cnt * bw_ * esz : bw_ * esz;
            w.srcStride  = one ? 0u : (outPitch_ - S9AlignUp(bw_ * esz, BLOCK_BYTES)) / BLOCK_BYTES;
            w.dstStride  = one ? 0u : (nOut_ - bw_) * esz;
            w.rsv        = 0;
            DataCopyPad(dstGm_[(uint64_t)r * nOut_ + cb0_], asmb, w);
            SetFlag<HardEvent::MTE3_V>(EvB(ab));
        }
        // **收尾：把没人等的 flag 等掉。** 悬空的 flag 会留给下一次 launch，
        // 单次调用看不出来，harness 跑 30 轮时会累积成挂死。
        //   V_MTE2  设了 it 次、等了 max(0, it−gBuf_) 次 ⇒ 补 min(it, gBuf_) 次
        //   MTE3_V  设了 it 次、等了 max(0, it−aBuf_) 次 ⇒ 补 min(it, aBuf_) 次
        //   MTE2_V / V_MTE3 同趟内设等成对，天然平衡。
        {
            const uint32_t ng = it < gBuf_ ? it : gBuf_;
            for (uint32_t k = 0; k < ng; ++k) {
                WaitFlag<HardEvent::V_MTE2>(EvB((it - 1 - k) % gBuf_));
            }
            const uint32_t na = it < aBuf_ ? it : aBuf_;
            for (uint32_t k = 0; k < na; ++k) {
                WaitFlag<HardEvent::MTE3_V>(EvB((it - 1 - k) % aBuf_));
            }
        }
    }

    /* **收尾：把最后一次 SetFlag 等掉。**
     * 每个 SetFlag 都必须有配对的 WaitFlag —— 悬空的 flag 会留给下一次 launch，
     * 单次调用看不出来，但 harness 跑 30 轮时会累积成挂死
     *（run.sh 撞它自己的 180 秒上限就是这么来的）。 */
    __aicore__ inline void Drain()
    {
        Flush();                    // 补上最后一趟的写出（软件流水把它延后了一拍）
        // 第 k 轮置的 MTE3→MTE2 由第 k+BUF_N 轮的 CopyIn 消费掉；
        // 跑完之后只剩最后 min(轮数, BUF_N) 个没人等，这里补齐。
        // **两条路各有自己的轮次计数，都要收**：
        //   · 行切/列切路 → itOut_（写出的趟数）
        //   · 拼装路（strategy 2）→ it_
        // 漏掉任一条，悬空的 SetFlag 会留给下一次 launch，多轮累积成挂死。
        // 计数为 0 时下面就是空循环，所以两条都写是安全的。
        uint32_t n = itOut_ < BUF_N ? itOut_ : BUF_N;
        for (uint32_t i = 0; i < n; ++i) {
            WaitFlag<HardEvent::MTE3_MTE2>(EvOut((itOut_ - 1 - i) % BUF_N));
        }
        itIn_ = 0; itOut_ = 0;
        uint32_t m = it_ < BUF_N ? it_ : BUF_N;
        for (uint32_t i = 0; i < m; ++i) {
            WaitFlag<HardEvent::MTE3_MTE2>(EvOut((it_ - 1 - i) % BUF_N));
        }
        it_ = 0;
    }

private:
    /* 一次搬 cnt 行 × cols_ 列。两个方向的 stride 单位不同：
     * GM→UB 的 srcStride 是**字节**、dstStride 是 **32B 块**；UB→GM 反过来。 */
    __aicore__ inline void CopyIn(const Job &j)
    {
        /* **这一对是冗余的**：`Emit` 只有 `One()` 一个调用点，而 `One()` 在发 Job
         * 之前已经为本路设过 `srcGm_`（同一路的每个 Job 用的是同一个指针）。
         * 每个 Job 重做一遍，在按行切那条路上是纯浪费 —— 那条路实测
         * `aiv_scalar_ratio = 0.77`、`mte2 = 0.87`，标量是紧挨着的次瓶颈。
         * `S9_HOIST=0` 保留旧行为，便于配对 A/B。 */
        if (!hoist_) {
            srcGm_.SetGlobalBuffer(j.src);
            srcGm_.SetL2CacheHint(S9InCacheMode(l2Mode_));
        }
        LocalTensor<T> ub = que_.template AllocTensor<T>();
        // **反向依赖按 buffer 配对，不按轮次串。**
        // FreeTensor 是在 MTE3 发出搬运之后立刻调的，块被标记可用时 MTE3 可能还没读完，
        // 所以写 UB 之前必须等"这一块"上一次被读完 —— 而**这一块**上一次被读是第 k−BUF_N 轮，
        // 不是第 k−1 轮。等 k−1 会把 MTE2 和 MTE3 完全串起来，双缓冲的重叠就没了
        //（那一版实测 23.5 us，官方 aclnnCat 是 10.7 us）。
        // **装载与写出各自计数。**
        // 软件流水把写出延后了一拍，若两者共用一个 it_，
        // CopyIn(k) 取到的 it_ 还停在 k−1 ⇒ 装载与写出会落到**同一块 buffer**，
        // MTE2 直接盖掉 MTE3 正在读的数据 —— 静默损坏，CPU 域看不出来。
        uint32_t b = itIn_ % BUF_N;
        if (itIn_ >= BUF_N) { WaitFlag<HardEvent::MTE3_MTE2>(EvOut(b)); }
        DataCopyExtParams p;
        p.blockCount = (uint16_t)j.cnt;
        p.blockLen   = j.cols * sizeof(T);
        p.srcStride  = (j.nIn - j.cols) * sizeof(T);
        p.dstStride  = S9UbGapBlocks(j.rowUb * sizeof(T), p.blockLen);
        p.rsv        = 0;
        DataCopyPadExtParams<T> pad{true, 0, 0, (T)0};
        DataCopyPad(ub, srcGm_[(uint64_t)j.r0 * j.nIn + j.c0], p, pad);
        // **MTE2 → MTE3 必须显式同步。**
        // `TQue<VECIN>` 的 EnQue/DeQue 插的是 MTE2→**V** 的同步（队列位置决定同步对象），
        // 而这里是纯搬运、中间没有 Vector 步骤，真正的消费者是 MTE3 ——
        // 少了这条，真机上 MTE3 会抢在 MTE2 写完之前读 UB，读到的是残留
        //（实测输出头部正是 TilingData 的 rows/nOut）。CPU 域顺序执行，看不出来。
        // **事件 ID 要按轮次交替**：BUF_N=2 时两轮可以同时在飞，
        // 共用一个 ID 会让第 k 轮的 Wait 配到第 k+1 轮的 Set。
        SetFlag<HardEvent::MTE2_MTE3>(EvIn(b));
        ++itIn_;
        que_.EnQue(ub);
    }

    __aicore__ inline void CopyOut(const Job &j)
    {
        LocalTensor<T> ub = que_.template DeQue<T>();
        uint32_t b = itOut_ % BUF_N;
        WaitFlag<HardEvent::MTE2_MTE3>(EvIn(b));        // 等本块的 CopyIn 真正落地
        DataCopyExtParams p;
        p.blockCount = (uint16_t)j.cnt;
        p.blockLen   = j.cols * sizeof(T);
        p.srcStride  = S9UbGapBlocks(j.rowUb * sizeof(T), p.blockLen);
        p.dstStride  = (nOut_ - j.cols) * sizeof(T);   // 跳过别的输入占的那些列
        p.rsv        = 0;
        DataCopyPad(dstGm_[(uint64_t)j.r0 * nOut_ + j.offOut + j.c0], ub, p);
        SetFlag<HardEvent::MTE3_MTE2>(EvOut(b));
        ++itOut_;
        que_.FreeTensor(ub);
    }

    TQue<TPosition::VECIN, BUF_N> que_;
    GlobalTensor<T> srcGm_, dstGm_;
    uint32_t rows_{0}, nOut_{0}, strategy_{0}, bn_{1}, bi_{0}, ubBudget_{0};
    // 向量拼装（strategy 3）专用
    uint32_t stagePitch_{0}, stageSpan_{0}, outPitch_{0}, tileG_{0}, numIn_{0};
    uint32_t asmSpan_{0};                 // 拼装区一块的字节数（outPitch_·tileG_）
    // 列带分核：本核负责的输出列区间 [cb0_, cb1_)，宽 bw_；bandCols_ 是标称带宽
    uint32_t bands_{1}, bandCols_{0}, cb0_{0}, cb1_{0}, bw_{0}, tblRow_{0}, aBuf_{1};
    uint32_t colBurstEl_{1};   // 按列切的每核载荷（元素），由 host 经 TilingData 给
    /* **拼装路不用 TQue，用 TBuf ＋ 手动 SetFlag/WaitFlag。**
     *
     * TQue 的 `DeQue()` 会**挡住标量单元**：下一趟的 DataCopyPad 要等本趟的
     * 向量算完才发得出去，于是读和算彻底串行。msprof 上看得一清二楚 ——
     * (2048,2048) int8 248 路那档 `dur 43.13 ≈ mte2 17.89 + vec 16.30`，
     * 两条流水的忙时几乎正好相加，说明完全没有重叠。
     * `WaitFlag` 只把等待指令塞进对应流水的队列，标量单元照常往下发，
     * 于是第 k+1 趟的读能和第 k 趟的算真正并行。 */
    TBuf<TPosition::VECCALC> stgBuf_;      // gBuf_ 块，每块 stageSpan_ 字节
    TBuf<TPosition::VECCALC> asmBuf_;      // aBuf_ 块，每块 outPitch_·tileG_ 字节
    TBuf<TPosition::VECCALC> offBuf_;
    TBuf<TPosition::VECCALC> dltBuf_;
    // int8 借道 half 时多出来的两块：暂存的 half 副本、拼装的 half 结果
    TBuf<TPosition::VECCALC> castBuf_;
    TBuf<TPosition::VECCALC> asmGBuf_;
    TBuf<TPosition::VECCALC> tdBuf_;    // TilingData 的 nIn[] 整块落到 UB   // 每列的行步长（打包暂存下逐行推进偏移表用）
    uint32_t nIn_{0}, offOut_{0}, cols_{0}, c0_{0}, rowUb_{0};
    /* 拼装路手动双缓冲的事件号。
     * `(事件类型, eventID)` 才是键，所以不同类型可以复用同一个 ID；
     * 《SetFlag/WaitFlag》建议只用 ID0~5（ID6 系统预留、ID7 给 TPipe 自动同步）。
     *   暂存块 b：V_MTE2 / MTE2_V 用 ID0~2
     *   拼装块 b：MTE3_V / V_MTE3 用 ID0~2（与上面类型不同，不冲突） */
    __aicore__ static inline event_t EvB(uint32_t b)
    {
        return b == 0 ? EVENT_ID0 : (b == 1 ? EVENT_ID1 : EVENT_ID2);
    }

    /* 每块 buffer 一对事件：EvIn 管 MTE2→MTE3、EvOut 管 MTE3→MTE2。 */
    // BUF_N 块缓冲各要一对事件号，两组不能重叠（ID0~2 装载、ID3~5 写出）
    __aicore__ static inline event_t EvIn(uint32_t b)
    {
        return b == 0 ? EVENT_ID0 : (b == 1 ? EVENT_ID1 : EVENT_ID2);
    }
    __aicore__ static inline event_t EvOut(uint32_t b)
    {
        return b == 0 ? EVENT_ID3 : (b == 1 ? EVENT_ID4 : EVENT_ID5);
    }
    uint32_t it_{0};                      // 拼装路（strategy 2）自己的轮次计数
    uint32_t gBuf_{2};                    // 向量拼装路的暂存缓冲块数
    uint32_t alignMask_{0};               // alignEl_ − 1，热路径对齐用
    uint32_t colBase_{0};                 // 按列切的轮转起点：已分出去的核数累计 % bn_
    uint32_t itIn_{0};                    // 已发出的装载趟数，决定 CopyIn 用哪块 buffer
    uint32_t itOut_{0};                   // 已发出的写出趟数，决定 CopyOut 用哪块 buffer
    Job pend_{};                          // 被延后一拍、还没写出的那一趟
    bool hasPend_{false};
    uint32_t alignEl_{1}, colChunk_{1};      // Init 里算一次
    uint32_t rowBeg_{0}, rowEnd_{0};         // 按行切的区间，与输入无关
    uint32_t l2Mode_{0};
    // 每带计划（S9_BANDPLAN=1）：本核只碰 [planLo_, planHi_) 这几路输入
    uint32_t planOn_{0}, planLo_{0}, planHi_{0}, planOff_{0};
    // 边界路裁剪（S9_TRIM）：被带切开的那 ≤2 条路只搬相交的那一段
    bool     trim_{false};
    uint32_t directMode_{0};      // 0=Gather 1=直写(沿用 tileG) 2=直写(放大 tileG)
    uint32_t tblK_{0};            // 偏移表只用 K+3 行、每条 Gather 覆盖 K 行（S9_TBLK）
    bool     oneWrite_{false};    // 单带且行宽 32B 整数倍时一个 tile 只写一块（S9_ONEWRITE）
    uint32_t dense_{0};       // 单带非对齐行密排，布局与建表使用相同开关
    uint32_t superG_{1};          // 一个对齐超行包含的逻辑行数
    bool     preRd_{false};       // 序幕重叠：建表之前先发第一趟 MTE2（S9_PRERD）
    uint32_t colParts_{1};        // 二维分核的列组数（S9_GRID），1 ＝ 纯按行切
    uint32_t colIdx_{0};          // 本核在行组内的列序号
    bool     hoist_{false};       // 跳过 CopyIn 里冗余的 SetGlobalBuffer/Hint（S9_HOIST）
    uint32_t bal_{1};      // 均衡切分开关（列带与按列切共用，S9_BAL）
    uint32_t hint_{1};     // 1 = 照旧每路每 tile 调一次 SetL2CacheHint；0 = NORMAL 时跳过（S9_HINT）
    uint32_t pipeMode_{0};   // 0=按策略自动 1=强制前瞻 2=强制不前瞻
};


/* 把 TilingData 里的 nIn[] 整块搬进 UB 再逐个读。
 *
 * **每一路一次 GM 标量读是很贵的。** msprof 的拟合给出「每路 0.15 µs」的固定开销
 *（A 档 123 路 vs B 档 8 路，同数据量差 17.23 µs）。原因是 `S9TilingRd::At`
 * 直接读 GM（`w_[byteOff>>2]`），而 kernel 里每路要读它两次（求 maxNIn 一次、
 * 填 nIns[] 一次），加上 `GetDataPtr` 的一次，248 路就是 744 次 GM 标量读。
 * 一条 DataCopyPad 把整个数组搬进 UB，之后是 UB 读，快两个数量级。 */
/* 把**动态输入的指针数组**整块搬进 UB 再逐个读。
 *
 * `ListTensorDesc::GetDataPtr(i)` 每调用一次就是一次 GM 标量读，而 kernel 每路都要
 * 调一次、**每个核都要走一遍全部 numIn 路**。msprof 把这笔账摊开了：
 * (512,2048) int32 248 路那档 **aiv_scalar_ratio 0.876**（30.17 µs / 34.4），
 * 而 mte2 只有 0.164、vec 0.123，带宽只有 208 GB/s ——
 * 248 路 × 每路约 0.15 µs ＝ 37 µs，与那 30 µs 对得上。
 *
 * 张量列表描述符的布局（`kernel_operator_list_tensor_impl.h`，dav_c220 与 dav_3510
 * 逐字相同）：首个 uint64 是「头地址到指针数组的字节偏移」，其后是连续的 uint64 指针。
 * 所以一条 DataCopyPad 就能把 numIn 个指针取进 UB。
 *
 * **但这依赖的是内部布局，不是公开约定**，所以取完要用公开 API 抽验两项
 *（首尾各一次 GetDataPtr）；对不上就整体退回逐个调 API 的老路。
 * 两次 GM 读换 numIn 次，代价可以忽略，而万一哪天布局变了不会静默出错。 */
/* **从 GM 连续搬 blockLen 字节进 UB，并就地等它落地。**
 *
 * `S9LoadSrcPtrs`（张量列表的指针数组，每项 8 字节）与 `S9LoadU32Arr`（nIn[]，
 * 每项 4 字节）是同一段 MTE2→S 的样板：建 GlobalTensor、填 DataCopyExtParams、
 * `DataCopyPad`、`SetFlag/WaitFlag(MTE2_S)`。原先各抄一遍 —— 抄错的后果是
 * **等错事件号**，真机上表现为偶发读到旧数据（不报错）。 */
__aicore__ inline void S9DmaToUb(GM_ADDR src, uint32_t blockLen, LocalTensor<uint32_t> &buf)
{
    GlobalTensor<uint32_t> g;
    g.SetGlobalBuffer((__gm__ uint32_t *)src);
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen   = blockLen;
    p.srcStride  = 0;
    p.dstStride  = 0;
    p.rsv        = 0;
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    DataCopyPad(buf, g, p, pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}

template <uint32_t N>
__aicore__ inline bool S9LoadSrcPtrs(GM_ADDR x, uint32_t n, LocalTensor<uint32_t> &buf,
                                     AscendC::ListTensorDesc &lst, __gm__ uint8_t *(&out)[N])
{
    // **路数少时不值。** 整块取要一条 DataCopyPad ＋ 一次 MTE2→S 同步 ＋ 两次抽验，
    // 大约相当于 3~4 次 GM 标量读；路数低于这个量级时逐个调 API 更快
    //（9 路那档实测 5.56 → 6.51 µs，就是这么亏的）。
    if (n < S9_PTR_BULK_MIN) { return false; }
    if (n == 0) { return true; }
    __gm__ uint64_t *head = (__gm__ uint64_t *)x;
    uint64_t byteOff = head[0];
    if (byteOff == 0 || (byteOff & 7u) != 0) { return false; }   // 偏移必须是 8 的倍数
    S9DmaToUb(x + byteOff, n * 8, buf);      // 每个指针 8 字节
    for (uint32_t i = 0; i < n; ++i) {
        // UB 侧按 uint32 读两半再拼 —— LocalTensor<uint64_t> 的标量读不是所有版本都有
        uint64_t lo = (uint64_t)buf.GetValue(i * 2);
        uint64_t hi = (uint64_t)buf.GetValue(i * 2 + 1);
        out[i] = (__gm__ uint8_t *)((lo) | (hi << 32));
    }
    // 抽验：首尾各一项必须与公开 API 一致
    if (out[0] != (__gm__ uint8_t *)lst.GetDataPtr<uint8_t>(0)) { return false; }
    if (n > 1 && out[n - 1] != (__gm__ uint8_t *)lst.GetDataPtr<uint8_t>(n - 1)) { return false; }
    return true;
}

template <uint32_t N>
__aicore__ inline bool S9LoadU32Arr(GM_ADDR tiling, uint32_t byteOff, uint32_t n,
                                    LocalTensor<uint32_t> &buf, uint32_t (&out)[N])
{
    if (n == 0) { return true; }
    if (n < S9_PTR_BULK_MIN) { return false; }   // 路数少时逐个读 GM 更快，见上面的说明
    S9DmaToUb(tiling + byteOff, n * 4, buf);
    for (uint32_t i = 0; i < n; ++i) { out[i] = buf.GetValue(i); }
    return true;
}

/* ═══ 三条路的执行体 ═══
 *
 * 从 `RunConcat` 里剥出来，只为读起来清楚：原来那段是三个内联的大块，一个函数几百行。
 * **策略仍然在运行时由 TilingData.strategy 决定** —— 曾经把它编进 key、用
 * `if constexpr` 做编译期折叠（`S9_SPECIALIZE=1`），上板量下来既没收益、又在 fp16
 * 形状上崩，已整条撤掉（见 RunConcat 与算子入口处的说明）。
 */

/* 两条拼装路（strat=2 DMA 拼装 / strat=3 向量拼装）。 */
template <typename T>
__aicore__ inline void S9RunAsm(ConcatMulti<T> &op, ListTensorDesc &srcList, GM_ADDR x,
                                GM_ADDR tiling, const S9TilingRdAny &td, uint32_t numIn,
                                uint32_t offN, uint32_t strat)
{
    __gm__ T *srcs[S9_ASM_MAX];
    uint32_t nIns[S9_ASM_MAX], offs[S9_ASM_MAX];
    uint32_t k = S9Min(numIn, S9_ASM_MAX);
    LocalTensor<uint32_t> tb = op.TdBuf();
    // **指针也整块取。** 每路一次 GetDataPtr 就是一次 GM 标量读，每个核都要走一遍
    // 全部 numIn 路 —— msprof 上 248 路那档 scalar_ratio 0.876、30 µs 全在这里。
    __gm__ uint8_t *ptrs[S9_ASM_MAX];
    const bool bulk = S9LoadSrcPtrs(x, k, tb, srcList, ptrs);
    const bool bulkN = S9LoadU32Arr(tiling, offN, k, tb, nIns);
    uint32_t acc = 0;
    for (uint32_t i = 0; i < k; ++i) {
        srcs[i] = bulk ? (__gm__ T *)ptrs[i] : srcList.GetDataPtr<T>(i);
        if (!bulkN) { nIns[i] = td.At(offN + i * 4); }
        offs[i] = acc;                       // 前缀和
        acc += nIns[i];
    }
    if (strat == 2) { op.Assembled(srcs, nIns, offs, k); }
    // 直写路放不下就**原地退回** Gather（那时一个字节都还没写出去）
    else if (!op.Direct() || !op.DirectBanded(srcs, nIns, offs, k)) {
        op.GatherAssembled(srcs, nIns, offs, k);
    }
}

/* 向量拼装路 + **每带计划**（`S9_BANDPLAN=1`）。
 *
 * 与 `S9RunAsm` 的区别只有一处：**只取本带那几路的指针与 nIn**。
 * 原先是每核把全部 numIn 路的指针与 nIn 都取一遍（指针还整块 DMA 进 UB），
 * 而分带之后每个核其实只碰 numIn/bands 路 —— 246 路 / 10 带那档，
 * 95% 的 `GetDataPtr` 与随之而来的 GM 标量读都是白花的。
 *
 * `offs[]` 从计划里的 `PlanOff()`（本带第一路在输出里的列前缀）开始累加，
 * 所以数组按**绝对下标**填，只有 [PlanLo, PlanHi) 这一段是有效的 ——
 * `GatherAssembled` 也只读这一段（它的扫描同样按计划跳过前面）。 */
template <typename T>
__aicore__ inline void S9RunAsmPlanned(ConcatMulti<T> &op, ListTensorDesc &srcList,
                                       GM_ADDR tiling, const S9TilingRdAny &td,
                                       uint32_t numIn, uint32_t offN)
{
    const uint32_t k  = S9Min(numIn, S9_ASM_MAX);
    const uint32_t lo = op.PlanLo(), hi = op.PlanHi();
    if (lo >= hi) { return; }                  // 空带：本核这一带没有任何一路
    __gm__ T *srcs[S9_ASM_MAX];
    uint32_t nIns[S9_ASM_MAX], offs[S9_ASM_MAX];
    uint32_t acc = op.PlanOff();
    for (uint32_t i = lo; i < hi; ++i) {
        srcs[i] = srcList.GetDataPtr<T>(i);
        nIns[i] = td.At(offN + i * 4);
        offs[i] = acc;
        acc += nIns[i];
    }
    if (!op.Direct() || !op.DirectBanded(srcs, nIns, offs, k)) {
        op.GatherAssembled(srcs, nIns, offs, k);
    }
}

/* 按输入切核（strategy 4）。
 *
 * 每一路归哪个核，由它的**起始列**落在哪一段决定：
 *     owner = offOut · blockDim / nOut
 * 这样每核拿到的是一段**连续的路**，而且是按**列数**（≈字节数）分的，
 * 不是按路数分 —— 各路长短差别很大，按个数分会很不均。
 * 每一路只归一个核 ⇒ 输出区间互不重叠，不需要任何核间同步。
 *
 * 代价是每个核都要把 numIn 项 nIn 走一遍 —— 那是 numIn 次 **GM** 标量读，
 * 248 路就是 37 µs（每路约 0.15 µs）。所以这里也整块取进 UB 再逐个读。
 * 指针不必整块取：本核只碰自己那几路，GetDataPtr 只调 numIn/blockDim 次。 */
/* **不收 `x`。** 这条路逐个调 `srcList.GetDataPtr` 取指针，用不上张量列表的原始
 * 基址（另外两条要它是为了 `S9LoadSrcPtrs` 的整块取）。原先收了一个 `x` 再
 * `(void)x;` 掩掉 —— 那种写法会让"这个参数是不是该用而漏用了"永远说不清。 */
template <typename T>
__aicore__ inline void S9RunInSplit(ConcatMulti<T> &op, ListTensorDesc &srcList,
                                    GM_ADDR tiling, const S9TilingRdAny &td, uint32_t numIn,
                                    uint32_t offN)
{
    const uint32_t bd = S9_TD(blockDim) ? S9_TD(blockDim) : 1;
    const uint32_t nOutAll = S9_TD(nOut);
    const uint32_t me = GetBlockIdx();
    uint32_t nIns4[S9_ASM_MAX];
    const uint32_t k4 = S9Min(numIn, S9_ASM_MAX);
    LocalTensor<uint32_t> tb4 = op.TdBuf();
    const bool bulkN4 = S9LoadU32Arr(tiling, offN, k4, tb4, nIns4);
    uint32_t off = 0;
    for (uint32_t i = 0; i < numIn; ++i) {
        uint32_t n = (bulkN4 && i < k4) ? nIns4[i] : td.At(offN + i * 4);
        uint32_t owner = nOutAll ? (uint32_t)((uint64_t)off * bd / nOutAll) : 0;
        if (owner >= bd) { owner = bd - 1; }
        if (owner == me) { op.One(srcList.GetDataPtr<T>(i), n, off); }
        off += n;
    }
}

/* 按行切/按列切（strategy 0/1）：每个核都要走全部 numIn 路，
 * 所以 nIn[] 和指针都整块取（每路一次 GM 标量读 × 两处，248 路就是 74 µs）。 */
template <typename T>
__aicore__ inline void S9RunRows(ConcatMulti<T> &op, ListTensorDesc &srcList, GM_ADDR x,
                                 GM_ADDR tiling, const S9TilingRdAny &td, uint32_t numIn,
                                 uint32_t offN)
{
    uint32_t nIns0[S9_ASM_MAX];
    __gm__ uint8_t *ptrs0[S9_ASM_MAX];
    const uint32_t k0 = S9Min(numIn, S9_ASM_MAX);
    LocalTensor<uint32_t> tb0 = op.TdBuf();
    const bool bulk0 = S9LoadSrcPtrs(x, k0, tb0, srcList, ptrs0);
    const bool bulkN0 = S9LoadU32Arr(tiling, offN, k0, tb0, nIns0);
    uint32_t off = 0;
    for (uint32_t i = 0; i < numIn; ++i) {
        // offOut 就地累加，不占 TilingData。
        uint32_t n = (bulkN0 && i < k0) ? nIns0[i] : td.At(offN + i * 4);
        __gm__ T *sp = (bulk0 && i < k0) ? (__gm__ T *)ptrs0[i] : srcList.GetDataPtr<T>(i);
        op.One(sp, n, off);
        off += n;
    }
}

/* 混合路一次搬多大。可用 -DS9_MIX_UB_CHUNK=... 在编译期改，用来二分定位。 */
#ifndef S9_MIX_UB_CHUNK
#define S9_MIX_UB_CHUNK (16u * 1024u)
#endif
#ifndef S9_MIX_L1_CHUNK
#define S9_MIX_L1_CHUNK (64u * 1024u)
#endif

/* 混合路的事件号：两块缓冲各一对（MTE2→MTE3 用 ID0/1，MTE3→MTE2 用 ID2/3）。
 *《SetFlag/WaitFlag》建议只用 ID0~5（ID6 系统预留、ID7 给 TPipe 自动同步）。 */
__aicore__ inline event_t S9MixEvIn(uint32_t b)  { return b == 0 ? EVENT_ID0 : EVENT_ID1; }
__aicore__ inline event_t S9MixEvOut(uint32_t b) { return b == 0 ? EVENT_ID2 : EVENT_ID3; }
/* 每个 SetFlag 都要有配对的 WaitFlag —— 悬空 flag 单次调用看不出来，
 * harness 跑 30 轮会累积成挂死（run.sh 撞 180 秒上限就是这么来的）。 */
__aicore__ inline void S9MixDrain(uint32_t it)
{
    const uint32_t n = it < 2 ? it : 2;
    for (uint32_t i = 0; i < n; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(S9MixEvOut((it - 1 - i) & 1u));
    }
}

/* **混合路的一整条半边**（AIC 或 AIV），只有三处差别用 `kAiv` 挑：
 *   ① 这半边分到哪一段（AIC 拿 [a, a+aicEl)，两个 AIV 对半分剩下的 [a+aicEl, b)）；
 *   ② 暂存在 L1 还是 UB —— 由调用方传进来的 `base` 决定，本函数不关心；
 *   ③ 连续搬 `DataCopy` 还是补齐搬 `DataCopyPad`（L1 侧长度必须 32B 对齐，
 *      UB 侧的零头只能走 Pad）。
 *
 * 原先两半边**各抄了一遍**同一段骨架：取 n、推 off、夹段、双缓冲分块、
 * 等上上一块的 MTE3→MTE2、搬进、MTE2→MTE3 设等、搬出、设 MTE3→MTE2、轮次 +1。
 * 抄错的后果是**等错事件号** —— 真机上表现为偶发读到旧数据，不报错；
 * 而且两边的前置短路条件本来就不一样（AIC 多一个 `pct == 0` 的提前 continue），
 * 眼睛比不出"它们看到的是同一段列"。收成一份之后不可能再漂。 */
/* 本组（一个 AIC ＋ 两个 AIV）**共同**看到的那一段：路数、nIn[] 偏移、
 * 列区间、AIC 的份额、对齐粒度。**两个半边必须逐字相同** —— 差一个就是
 * 两边算出不同的列区间，互相写对方的地盘（错且不报错）。所以只建一份传下去。 */
struct S9MixSlice {
    uint32_t numIn{0}, offN{0}, c0{0}, c1{0}, pct{0}, alignEl{1};
};

/* `kPair`：这一段是不是由**两个 AIV 对半分**。
 *   · 混合核路（`RunConcatMix`）一组里有两个 AIV，要分 ⇒ kPair = true；
 *   · 输出线性切分（strategy 5）每核就是自己的整段，**不分** ⇒ kPair = false。
 * `RD`：TilingData 的读取器。混合核路只有 GM 直读（AIC 没有 UB），
 * 而 strategy 5 走的是普通 AIV 路，序幕已经把标量头整块取进 UB ⇒ 用 `S9TilingRdAny`
 * 能省掉每路一次 GM 标量读。模板化之后两个调用点共用同一份实现。 */
template <typename T, bool kAiv, bool kPair, typename RD>
__aicore__ inline void S9MixRunHalf(const RD &td, ListTensorDesc &srcList,
                                    GlobalTensor<T> dst, LocalTensor<T> base,
                                    uint32_t chunkEl, const S9MixSlice &sl, uint32_t sub)
{
    constexpr uint32_t esz = (uint32_t)sizeof(T);
    const uint32_t alignEl = sl.alignEl;
    DataCopyPadExtParams<T> pad{false, 0, 0, (T)0};
    uint32_t off = 0, it = 0;
    for (uint32_t i = 0; i < sl.numIn; ++i) {
        const uint32_t n = td.At(sl.offN + i * 4);
        uint32_t s = 0, a = 0, b = 0;
        if (!S9ClipRoute(n, off, sl.c0, sl.c1, s, a, b)) { continue; }
        // AIC 那一份的长度向下取整到 32B（`pct == 0` ＝ 对照组：AIC 不搬数据）
        const uint32_t aicEl =
            (sl.pct == 0) ? 0u
                          : (uint32_t)(((uint64_t)(b - a) * sl.pct / 100) / alignEl * alignEl);
        uint32_t q0 = a, q1 = a + aicEl;
        if constexpr (kAiv) {
            const uint32_t r0 = a + aicEl;
            if (r0 >= b) { continue; }
            if constexpr (kPair) {
                // 两个 AIV 对半分剩余段；分界按 32B 取整，免得两边都写同一个 block
                uint32_t mid = S9AlignUp(r0 + (b - r0) / 2, alignEl);
                if (mid > b) { mid = b; }
                q0 = (sub == 0) ? r0 : mid;
                q1 = (sub == 0) ? mid : b;
            } else {
                q0 = r0; q1 = b;          // 每核就是自己的整段，不再对半
            }
        }
        if (q1 <= q0) { continue; }
        GlobalTensor<T> src;
        src.SetGlobalBuffer(srcList.GetDataPtr<T>(i) + (q0 - s));
        src.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        for (uint32_t p = 0; p < q1 - q0; p += chunkEl) {
            uint32_t cnt = (q1 - q0) - p;
            if (cnt > chunkEl) { cnt = chunkEl; }     // chunkEl·esz 是 32 的倍数
            const uint32_t bb = it & 1u;
            LocalTensor<T> buf = base[bb * chunkEl];
            if (it >= 2) { WaitFlag<HardEvent::MTE3_MTE2>(S9MixEvOut(bb)); }
            if constexpr (kAiv) {
                DataCopyExtParams e{1, cnt * esz, 0, 0, 0};
                DataCopyPad(buf, src[p], e, pad);
                SetFlag<HardEvent::MTE2_MTE3>(S9MixEvIn(bb));
                WaitFlag<HardEvent::MTE2_MTE3>(S9MixEvIn(bb));
                DataCopyPad(dst[q0 + p], buf, e);
            } else {
                DataCopy(buf, src[p], cnt);
                SetFlag<HardEvent::MTE2_MTE3>(S9MixEvIn(bb));
                WaitFlag<HardEvent::MTE2_MTE3>(S9MixEvIn(bb));
                DataCopy(dst[q0 + p], buf, cnt);
            }
            SetFlag<HardEvent::MTE3_MTE2>(S9MixEvOut(bb));
            ++it;
        }
    }
    S9MixDrain(it);
}

/* ═══ 输出线性切分（strategy == 5）—— `rows == 1`（所有 `dim=0`）专用 ═══
 *
 * **为什么需要它。** `dim=0` 的 concat 必然 `rows == 1`（`rows = ∏shape[:0]` ＝ 空积）。
 * 而按行切在 `rows == 1` 时只开得出**一个核**，kernel 侧再按"路内的列"分核又会把
 * 大块切碎 —— 910B4 实测：强开到 40 核反而慢 3~4×（单调），而默认的单核跑在
 * **34~71 GB/s**，是列带分核主路径（692 GB/s）的 1/10 ~ 1/19。
 *
 * **正确的切法是按输出的字节位置切**：每核拿一段连续的 `[c0, c1)` 列，
 * 夹出与它相交的那几路，各自发大块 DMA。负载完全均衡（每核字节数只差一个 32B 粒度）、
 * 写完全连续、每条 DMA 大。实测 **2.59× ~ 3.21×**（36.87 / 23.86 / 36.96 µs）。
 *
 * **不需要混合核。** 这条路与 `RunConcatMix` 的 AIV 半边是同一个函数
 *（`S9MixRunHalf<T, kAiv=true, kPair=false>`）；实测 `S9_AIC` 取 0/30/50/70
 * 结果完全一样（36.87/36.36/36.74/36.98）⇒ **AIC 一点贡献都没有**，
 * 所以这里走默认的 `AIV_ONLY` 核型，**不动 TilingKey、不碰 MIX**。
 *
 * 与混合核路的唯一差别：那边一组两个 AIV 要对半分（`kPair=true`），
 * 这边每核就是自己的整段（`kPair=false`）。 */
template <typename T>
__aicore__ inline void S9RunLinear(ListTensorDesc &srcList, GM_ADDR y,
                                   const S9TilingRdAny &td, uint32_t numIn,
                                   uint32_t offN, TPipe *pipe)
{
    constexpr uint32_t esz = (uint32_t)sizeof(T);
    constexpr uint32_t alignEl = (BLOCK_BYTES / esz) ? (BLOCK_BYTES / esz) : 1u;
    const uint32_t nOut = S9_TD(nOut);
    // **组数取 GetBlockNum()，不读 TilingData 的 blockDim** —— 同 RunConcatMix 的理由：
    // host 可能在 set_blockDim 之后又改过核数，而框架按 SetBlockDim 的实参启动。
    const uint32_t bn = (uint32_t)GetBlockNum() ? (uint32_t)GetBlockNum() : 1u;
    const uint32_t bi = (uint32_t)GetBlockIdx();
    // 本核的输出列区间（按 32B 取整；末核收到 nOut）。取整保证两个核不会写同一个 block。
    uint32_t c0 = S9AlignUp((uint32_t)((uint64_t)nOut * bi / bn), alignEl);
    uint32_t c1 = S9AlignUp((uint32_t)((uint64_t)nOut * (bi + 1) / bn), alignEl);
    if (c0 > nOut) { c0 = nOut; }
    if (c1 > nOut) { c1 = nOut; }
    if (c1 <= c0) { return; }      // 这个核没分到列（nOut 小于核数时的尾巴）

    GlobalTensor<T> dst;
    dst.SetGlobalBuffer((__gm__ T *)y);
    dst.SetL2CacheHint(S9OutCacheMode(S9_TD(l2ModeOut)));

    constexpr uint32_t kChunkB = S9_MIX_UB_CHUNK;
    TBuf<TPosition::VECCALC> ubb;      // 手动双缓冲，理由同 RunConcatMix
    pipe->InitBuffer(ubb, 2 * kChunkB);

    S9MixSlice sl;
    sl.numIn = numIn; sl.offN = offN; sl.c0 = c0; sl.c1 = c1;
    sl.pct = 0;                        // 没有 AIC 那一份
    sl.alignEl = alignEl;
    S9MixRunHalf<T, true, false>(td, srcList, dst, ubb.template Get<T>(),
                                 kChunkB / esz, sl, 0);
}

/* **策略在运行时由 TilingData.strategy 决定。**
 *
 * 曾经试过"把策略编进 TilingKey、用 `if constexpr` 让没走到的路整段不实例化"
 *（`S9_SPECIALIZE=1`，key = 100 + strategy·10 + log2(esz)）。2026-09-11 上板结论：
 * **没收益、而且在 fp16 形状上直接把算子打崩**（int8 那档能跑，B/A = 1.009；
 * fp16 的 128×256 与 512×4096 都是 "Inner error ... current working operator name
 * is aclnnConcat"）。机理没查（怀疑是 if constexpr 之后 helper 被内联、
 * 设备侧栈帧变大），但既然**同机 A/B 量不出收益**，就没有理由留着这个地雷。
 * ⇒ 这条实验整体撤掉：分支回到运行时，键回到元素字节宽度。 */
template <typename T>
__aicore__ inline void RunConcat(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    // **标量头 ＋ 整个 nIn[] 一条 DMA 取进 UB**（见 S9TilingUb / S9_PRE_BYTES）。
    // 之后 `S9_TD(...)` 和 `td.At(offN + i*4)` 全是 UB 标量读（几个拍），
    // 序幕里再没有一次 GM 标量读。
    static_assert(offsetof(ConcatTilingData, nIn) <= S9_HDR_BYTES,
                  "S9_HDR_BYTES 必须覆盖 nIn[] 之前的全部标量字段");
    static_assert(offsetof(ConcatTilingData, nIn) + S9_ASM_MAX * 4 <= S9_PRE_BYTES,
                  "S9_PRE_BYTES 必须覆盖整个 nIn[] —— 否则超出的路读到的是垃圾");
    S9TilingRd tdg(tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> hdBuf;
    pipe.InitBuffer(hdBuf, S9_PRE_BYTES);
    // 开关本身只花**一次** GM 标量读，两组 A/B 都付，不影响比较。
    const bool pre = tdg.At((uint32_t)offsetof(ConcatTilingData, preMode)) != 0;
    S9TilingRdAny td{tdg, hdBuf.Get<uint32_t>(), pre};
    if (pre) {
        GlobalTensor<uint32_t> gh;
        gh.SetGlobalBuffer((__gm__ uint32_t *)tiling);
        DataCopyExtParams hp{1, S9_PRE_BYTES, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> hpad{false, 0, 0, 0};
        DataCopyPad(td.u_, gh, hp, hpad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }
    const uint32_t numIn = S9_TD(numIn);
    const uint32_t offN  = (uint32_t)offsetof(ConcatTilingData, nIn);
    // **offOut 不再由 TilingData 传** —— 它是 nIn 的前缀和，这里边走边累加。
    // 省掉的那一半空间换成路数上限翻倍（1024 → 2048）。
    const uint32_t strat = S9_TD(strategy);
    // 动态输入是**一个 GM_ADDR 指向张量列表描述符**，用 ListTensorDesc 解包
    ListTensorDesc srcList((__gm__ void *)x);

    /* ═══ 输出线性切分（strategy 5）**在 ConcatMulti 之前就分出去** ═══
     *
     * 它那套「行/带/暂存/拼装」的账在 `rows == 1` 上没有意义，而且
     * `ConcatMulti` 与这条路各有一套事件号与轮次计数 —— 让两者同时在场，
     * "谁该收哪个 flag"就要靠人去推。直接早返回，交互面归零。 */
    if (strat == 5) {
        S9RunLinear<T>(srcList, y, td, numIn, offN, &pipe);
        return;
    }

    ConcatMulti<T> op;
    // 二十几个 tiling 字段的读法**写在 Init 里**（含"哪几个必须走 GM 直读"），
    // 这里只交出源头：输出地址、TilingData 基址、序幕读取器、TPipe。
    op.Init(y, tiling, td, &pipe);
    {
        // 策略运行时才知道，所以四条路都在（但 `One` 内部仍按
        // `strategy_` 走各自的分支，行为与改动之前**逐字节相同**）。
        // 全输入计划且路数较多时复用已验证的批量元数据入口。
        // 部分输入带和小路数仍逐项读取，避免引入额外 DMA/同步开销。
        const bool fullPlanBulk = strat == 3 && op.PlanOn() && numIn >= S9_PTR_BULK_MIN &&
                                  op.PlanLo() == 0 && op.PlanHi() == numIn;
        if (strat == 3 && op.PlanOn() && !fullPlanBulk) {
            S9RunAsmPlanned<T>(op, srcList, tiling, td, numIn, offN);
        } else if (strat == 2 || strat == 3) {
            S9RunAsm<T>(op, srcList, x, tiling, td, numIn, offN, strat);
        } else if (strat == 4) {
            S9RunInSplit<T>(op, srcList, tiling, td, numIn, offN);
        } else {
            S9RunRows<T>(op, srcList, x, tiling, td, numIn, offN);
        }
    }
    op.Drain();
}


/* ═══ AIC＋AIV 混合整块搬运（TilingKey ＝ esz + 16）═══
 *
 * ## 为什么 AIC 能参与
 *
 * 曾经据「A2 上 `DataCopyPad_GMToL1` 不支持、`DataCopy_L1ToGM` 要求 32B 粒度」
 * 断言 AIC 搬不了 Concat 的落点。**那个结论是错的**，读漏了对齐要求的主语：
 *   ·《DataCopy（GMToL1连续数据搬运）》：**源地址（GM）需要 1 字节对齐**，
 *     目的地址（L1）32 字节对齐；`count*sizeof(T)` 需 32 字节对齐，
 *     不对齐会**向下取整**（静默少搬，不报错）。
 *   ·《DataCopy（L1ToGM连续数据搬运）》：**目的地址（GM）需要 1 字节对齐**，
 *     源地址（L1）32 字节对齐；`count*sizeof(T)` 同样要 32 字节对齐。
 * 也就是说 **32B 约束只落在 L1 侧和「长度」上，GM 两头都是任意字节**。
 * 于是任意偏移的连续块 [s, s+L) → [d, d+L) 可以这样切：
 *     AIC 搬前 ⌊L/32⌋·32 字节（长度对齐，两端 GM 地址任意）
 *     AIV 搬剩下的 ≤31 字节（DataCopyPad 无对齐要求）
 * **连头都不用切，只剩一个尾巴。**
 *
 * ## 适用范围
 *
 * 只用于 `rows == 1`（dim=0 归一之后每一路在输出里就是一整个连续块）。
 * dim=-1 那类「每行一条窄带」的落点仍然只能走 Gather 拼装路 —— AIC 没有 Gather。
 *
 * ## 分工
 *
 * `KERNEL_TYPE_MIX_AIC_1_2` 下 `SetBlockDim(N)` 启动 N 个 Cube ＋ 2N 个 Vector
 *（《设置Kernel类型》表 2），`GetBlockIdx()` 给组号、`GetSubBlockIdx()` 给组内的
 * AIV 编号（0/1）。输出按组切成 N 段（段界按 32B 取整），组 g 内：
 *     每条与本段相交的路 [a, b)：AIC 拿前 `aicPct%`（长度向下取整到 32B），
 *                                余下 [a+aicEl, b) 由两个 AIV 对半分。
 * `aicPct = 0` 就是**同一份启动配置下的对照组**（AIC 空转，全部由 AIV 做），
 * 这样「混合核本身的启动开销」和「AIC 到底有没有带宽」两件事能分开量。
 *
 * 三条路互不重叠地写输出 ⇒ **不需要任何核间同步**（不用 CrossCoreSetFlag）。
 *
 * ## 已知的一次性开销
 *
 * AIC 没有 UB，`S9LoadU32Arr` 那条「整块取进 UB 再读」的路走不了，
 * 所以 nIn[] 只能逐个 GM 标量读（每路约 0.1 µs）。247 路那档约 25 µs。
 * 先接受它 —— 要量的是带宽，这笔开销单独可见、也单独可修（让 host 下发每组的
 * 起始路号即可）。
 */
template <typename T>
__aicore__ inline void RunConcatMix(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    S9TilingRd td(tiling);
    const uint32_t numIn = S9_TD(numIn);
    const uint32_t offN  = (uint32_t)offsetof(ConcatTilingData, nIn);
    const uint32_t nOut  = S9_TD(nOut);
    // **组数取 GetBlockNum()，不读 TilingData 的 blockDim。**
    // host 侧 `tiling.set_blockDim()` 在混合路把 blockDim 改成 AIC 数**之前**就调过了，
    // TilingData 里留的是旧值 —— kernel 按 24 分组、框架按 20 启动，
    // 于是最后 4/24 的输出没人写（真机 CHECK MISMATCH，不报错）。
    // GetBlockNum() 返回的就是 SetBlockDim 的实参，两边不可能对不上。
    const uint32_t ng    = (uint32_t)GetBlockNum() ? (uint32_t)GetBlockNum() : 1u;
    const uint32_t pct   = S9_TD(aicPct) > 100 ? 100u : S9_TD(aicPct);
    constexpr uint32_t esz = (uint32_t)sizeof(T);
    constexpr uint32_t alignEl = (BLOCK_BYTES / esz) ? (BLOCK_BYTES / esz) : 1u;
    // **AIV 上的 GetBlockIdx() 已经是展平的 AIV 序号，不是组号。**
    //《GetBlockIdx》原文：「当AIC与AIV比例为1:2时，AIC上取值范围为[0, numBlocks)，
    // **AIV上取值范围为[0, 2 * numBlocks)**」。
    // 一开始两边都直接拿它当组号，于是 AIV 的 g 跑到 [0, 2N) —— g ≥ N 的那一半
    // 算出 c0 = c1 = nOut，什么都不做，**输出有一半没人写**（真机 CHECK MISMATCH，
    // 不报错）。CLAUDE.md 里记的 `logic_idx = block_idx*sub_block_num + sub_block_idx`
    // 描述的是另一套口径，以 API 文档为准。
    // `ASCEND_IS_AIC` 只能用在 `if` 的位置（它展开成带初始化的 if constexpr），
    // 写成三元表达式会报 "default initialization of an object of const type"。
    const uint32_t flat = (uint32_t)GetBlockIdx();
    uint32_t g = flat, sub = 0;
    if ASCEND_IS_AIV { g = flat >> 1; sub = flat & 1u; }

    // 本组的输出列区间（按 32B 取整；末组收到 nOut）
    uint32_t c0 = S9AlignUp((uint32_t)((uint64_t)nOut * g / ng), alignEl);
    uint32_t c1 = S9AlignUp((uint32_t)((uint64_t)nOut * (g + 1) / ng), alignEl);
    if (c0 > nOut) { c0 = nOut; }
    if (c1 > nOut) { c1 = nOut; }

    TPipe pipe;
    ListTensorDesc srcList((__gm__ void *)x);
    GlobalTensor<T> dst;
    dst.SetGlobalBuffer((__gm__ T *)y);
    dst.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

    // 两个半边共用的那一段（只建一份，见 S9MixSlice）
    S9MixSlice sl;
    sl.numIn = numIn; sl.offN = offN; sl.c0 = c0; sl.c1 = c1;
    sl.pct = pct;     sl.alignEl = alignEl;

    if ASCEND_IS_AIC {
        // L1 512 KB，取两块 64 KB 做双缓冲。
        // **不用 TQue，手动双缓冲。** 两个原因：
        //  ① 这条路是 MTE2 → MTE3（中间没有向量指令），而 TQue 的 EnQue/DeQue
        //     给的是「按 TPosition 推出来的」那对同步（VECIN 给 MTE2→V），
        //     方向不对 —— 写出时数据还没落地，真机 CHECK MISMATCH 且不报错。
        //  ② TQue 的事件号由 TPipe 的 AllocEventID 分配，和这里写死的 ID0~3 有撞号风险。
        constexpr uint32_t kChunkB = S9_MIX_L1_CHUNK;
        TBuf<TPosition::A1> l1b;
        pipe.InitBuffer(l1b, 2 * kChunkB);
        S9MixRunHalf<T, false, true>(td, srcList, dst, l1b.template Get<T>(),
                                     kChunkB / esz, sl, 0);
    }
    if ASCEND_IS_AIV {
        constexpr uint32_t kChunkB = S9_MIX_UB_CHUNK;
        TBuf<TPosition::VECCALC> ubb;      // 同上：手动双缓冲，不用 TQue
        pipe.InitBuffer(ubb, 2 * kChunkB);
        S9MixRunHalf<T, true, true>(td, srcList, dst, ubb.template Get<T>(),
                                    kChunkB / esz, sl, sub);
    }
}

extern "C" __global__ __aicore__ void concat(GM_ADDR x, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 混合核只给 key = esz+16 那几支开；其余分支仍是纯 AIV，启动配置一个字节没变。
    KERNEL_TASK_TYPE(17, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(18, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(20, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(24, KERNEL_TYPE_MIX_AIC_1_2);
    // TilingKey ＝ 元素字节宽度（+16 ＝ 走 AIC＋AIV 混合路）。
    // **只能 if / else if，不支持 else**
    if (TILING_KEY_IS(1)) {
        RunConcat<int8_t>(x, y, tiling);
    } else if (TILING_KEY_IS(2)) {
        RunConcat<half>(x, y, tiling);
    } else if (TILING_KEY_IS(4)) {
        RunConcat<float>(x, y, tiling);
    } else if (TILING_KEY_IS(8)) {
        RunConcat<int64_t>(x, y, tiling);
    } else if (TILING_KEY_IS(17)) {
        RunConcatMix<int8_t>(x, y, tiling);
    } else if (TILING_KEY_IS(18)) {
        RunConcatMix<half>(x, y, tiling);
    } else if (TILING_KEY_IS(20)) {
        RunConcatMix<float>(x, y, tiling);
    } else if (TILING_KEY_IS(24)) {
        RunConcatMix<int64_t>(x, y, tiling);
    }
    /* 这里曾经有过一组"编译期策略特化"的 key（100 + strategy·10 + log2(esz)）。
     * 2026-09-11 上板：int8 能跑但 B/A = 1.009（无收益），fp16 形状直接崩
     *（见 RunConcat 头上那段说明）⇒ 整条实验撤掉，键回到元素字节宽度。 */
}
