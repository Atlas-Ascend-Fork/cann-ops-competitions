# T5QKV 拆分与缓存追加算子（T5SplitQkvAppendCache）设计文档

> 任务：10 月社区任务——T5QKV 拆分与缓存追加算子开发（题目3）
> 硬件：昇腾 910C（Atlas A3 / Ascend910_93 soc，arch22，单逻辑设备 48 Vector Core + 24 Cube Core）
> 代码仓：`t5-split-qkv-append-cache`（实现基线 commit `9aecd19`）
> 本文为竞赛提交设计文档；更细的内部设计草案见仓内 `expert/01_design.md`（地址映射、搬运量表、分核演算的完整推导均以该文档 §2 为准，此处直接引用）。

---

# 需求背景

## 需求来源

- **任务来源**：10 月社区任务——T5QKV 拆分与缓存追加算子开发，任务书《题目3-T5QKV拆分与缓存追加算子开发任务书》（版本 2026-10-23）。
- **接口与语义定义**：任务书 §2.3「接口定义」与本包 `op.json`（3 输入 / 3 输出 / 2 标量属性）。
- **验收定义**：任务书 §3（35 个公开精度用例逐位一致、10 个性能用例固定耗时上限、内存峰值增量预算、35 节自验矩阵），测试数据与标杆由任务包 `case.json` / `generate_data.py` / `golden.py` 提供。

## 背景介绍

大模型单 token 解码（decode）阶段，每层 Transformer 需要把当前 token 的 Q/K/V 投影结果拆开：Q 参与本步注意力计算，K/V 则要**追加进 KV 缓存**供注意力读取历史上下文。训练/推理框架常将三者的线性投影输出融合为一块连续的 `qkv` 张量（Q、K、V 三段顺序拼接），解码时需要一个算子完成两件事：

1. 把融合 QKV 拆成独立的 Q、K、V（重排 + 复制，输出独立连续存储）；
2. 把当前 token 的 K/V 拼接到历史缓存 `pastKey/pastValue` 的末尾，得到 `L+1` 长度的新缓存。

该算子位于解码热路径上，每步每层都要执行，且形状跨度极大（B/H/L/D 覆盖 1~64 / 1~128 / 0~8192 / 1~256，总搬运量跨 5 个数量级，从几十字节到约 1 GB）。纯数据搬运、无数值计算，性能竞争点在于多核切分、DMA 效率与 host 侧每次调用开销。

## 现状分析

- **基线实现**（任务书 §3.3 定义）：视图拆分 Q/K/V，Q clone 为独立连续输出，K/V 分别与历史缓存做 `torch.cat`。该实现每步产生 3 个独立算子（clone + 2 次 concat），多次 kernel 启动与中间排空间隙；且 concat 的 K/V 写出不可避免地对每 head 片做跨 stride 写。逐用例基线耗时 64~205 μs（见 §与标杆差异）。
- **平台约束**：目标平台为 910C（Ascend910_9382 / DAV_2201 / arch22），AIC:AIV = 1:2，每 AIV UB 192KB；**Ascend C 无 GM→GM 直拷**，DataCopy 必须经 UB 中转。fp16/bf16 位宽一致（2B/元素），本算子无任何数值运算，可统一按 uint16 位拷贝，天然满足"逐位一致（含 ±0 / Inf / NaN 位模式）"的精度要求。
- **非对齐现实**：公开用例中 D 取值含 1/3/5/65/80/255 等非 32B 对齐值，且额外种子用例的输入基址带 +2B 偏移；DMA 搬运必须同时具备对齐快路与运行期自动降级路径。
- **L=0 特殊性**：首次解码时历史缓存元素数为 0，指针可能为 null，算子必须结构化跳过 past 复制且不得触碰空指针。

---

# 需求分析

## 需求描述

单 token 解码场景：输入按 Q、K、V 三段顺序拼接的融合 QKV `qkv[B,1,3HD]`，以及两份连续历史缓存 `pastKey/pastValue[B,H,L,D]`；输出独立连续的 `query[B,H,1,D]`、`key[B,H,L+1,D]`、`value[B,H,L+1,D]`，其中 K/V 的前 L 个 token 逐位复制历史缓存，末尾 token 分别取 qkv 的 K 段与 V 段。全部输出重新分配、互不重叠、不修改输入；L=0 时不得读取空历史缓存。

工程形态要求（任务书 §2.2）：

- ACLNN 算子工程化：Ascend C 内核、OpDef 注册、shape/dtype 推导、tiling、可构建安装包（安装到用户目录，不覆盖系统 CANN）、两阶段 ACLNN API、独立 C++ 调用样例；
- Kernel 直调工程，与 ACLNN 复用同一 tiling 计算函数；
- PyTorch NPU 接口 `run(inputs, attributes) -> tuple[Tensor,...]` 作为精度/性能/内存验收入口，另提供 `prepare` 与 `run_direct`（Kernel 直调交叉验证）；
- 显式全局工作区为 0；框架入口须校验输入并拒绝非法/未知属性。

## 输入输出规格

| 方向 | 名称 | 形状 | dtype | 说明 |
|---|---|---|---|---|
| 输入 | `qkv` | `[B,1,3*H*D]` | float16 / bfloat16 | 按 Q、K、V 三段排列，ND 连续 |
| 输入 | `pastKey` | `[B,H,L,D]` | 与 qkv 一致 | 连续历史缓存（L 可为 0） |
| 输入 | `pastValue` | `[B,H,L,D]` | 与 qkv 一致 | 同上 |
| 输出 | `query` | `[B,H,1,D]` | 与 qkv 一致 | 独立连续，`qkv[b,0,h*D+d] → query[b,h,0,d]` |
| 输出 | `key` | `[B,H,L+1,D]` | 与 qkv 一致 | 前 L 片复制 pastKey，第 L 片取 `qkv[b,0,H*D+h*D+d]` |
| 输出 | `value` | `[B,H,L+1,D]` | 与 qkv 一致 | 前 L 片复制 pastValue，第 L 片取 `qkv[b,0,2*H*D+h*D+d]` |
| 属性 | `headNum` | 标量 int | — | 必须等于 H（缺省按形状推导） |
| 属性 | `headDim` | 标量 int | — | 必须等于 D（缺省按形状推导） |

约束：`1≤B≤64`、`1≤H≤128`、`1≤D≤256`、`0≤L≤8192`；总输出元素数 `B*H*D + 2*B*H*(L+1)*D ≤ 2^28`；支持非对齐 D；所有输入连续、同 dtype、同设备；拒绝尺寸、类型、布局、属性不匹配。两阶段 ACLNN API：

```cpp
aclnnStatus aclnnT5SplitQkvAppendCacheGetWorkspaceSize(
    const aclTensor *qkv, const aclTensor *pastKey, const aclTensor *pastValue,
    int64_t headNum, int64_t headDim,
    aclTensor *query, aclTensor *key, aclTensor *value,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnT5SplitQkvAppendCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor);
```

## 需求拆解

| # | 子需求 | 关键点 | 本方案对应章节 |
|---|---|---|---|
| R1 | 地址映射与布局分析 | 五类子问题全部退化为连续段拷贝 | 详细设计 §算子分析 |
| R2 | 多核切分与负载均衡 | 48 AIV，past 沿 L 分片铺满 | 详细设计 §分核策略 |
| R3 | Tiling 设计 | 纯 host 可单测函数，ACLNN/直调共享 | 详细设计 §TilingData 与 ComputeTiling |
| R4 | UB 缓冲与流水 | 192KB/核，双缓冲 MTE2/MTE3 软件流水 | 详细设计 §UB 双缓冲设计 |
| R5 | 对齐/非对齐双路 | 对齐快路 + 运行期逐描述符 pad 降级，正确性不依赖 tilingKey | 详细设计 §UB 双缓冲设计 / 实现状态 |
| R6 | 输入校验与拒绝 | 28 非法场景 × 3 入口 = 84 条全数拒绝，固定校验顺序 | 详细设计 §校验顺序 |
| R7 | 精度逐位一致 | uint16 位拷贝，无数值运算 | 全篇（架构性保证） |
| R8 | 内存纪律 | workspace=0、峰值增量 ≤ 输出字节+16KiB、20 次调用回收 | 详细设计 §需求拆解 / 实现状态 |
| R9 | 双入口交叉验证 | run（ACLNN）与 run_direct（Kernel 直调）同源同结果 | 详细设计 §TilingData / 实现状态 |

## 与标杆差异

标杆为任务书 §3.3 定义的 `views_concat_q_copy`：视图拆分 + Q clone + K/V 两次 `torch.cat`。差异点：

| 维度 | 标杆 | 本实现 |
|---|---|---|
| kernel 次数/调用 | 3 个算子（clone + 2×concat） | 1 个融合 kernel |
| K/V 写模式 | concat 逐 head 片跨 stride 写 | past 段连续大块 memcpy；末尾 token 单条多块 DMA（strided 写） |
| 多核利用 | 框架内部分核 | 48 AIV 显式单元化 round-robin，past 沿 L 自适应分片 |
| 位一致性 | 浮点拷贝路径 | uint16 位拷贝，NaN/Inf/±0 按构造保留 |
| 每调用 host 开销 | 3 次 Python→算子 dispatch | 1 次 `torch.ops` C++ dispatch 内完成校验后全流程 |

任务书 10 个性能用例的固定上限（同机同输入实测比值制）：001/011≤28μs、005/015≤28μs、006/016≤29μs、008/018≤29μs、007/017≤75/77μs（基线 64.48~204.52μs）。搬运量最大的 007/017（16,16,2048,64）读+写合计约 256 MiB/次，是带宽竞争主战场；001/011（16KB）类是纯启动/框架开销竞争（详见 `expert/01_design.md` §2.3/§9）。

---

# 详细设计

## 算子分析

### 逐元素地址映射公式

记 `HD=H*D`、`LD=L*D`、`SL=(L+1)*D`（key/value 中每个 head 的一片），各张量按行优先展平（元素下标，fp16/bf16 每元素 2 字节；引自 `expert/01_design.md` §2.1）：

```text
query[b,h,0,d]           :  src = qkv + b*3HD + 0   + h*D + d   →  dst = query + b*HD  + h*D + d
key[b,h,l,d]   (0≤l<L)   :  src = pk  + b*HLD + h*LD + l*D + d   →  dst = key   + b*HSL + h*SL + l*D + d
value[b,h,l,d] (0≤l<L)   :  src = pv  + b*HLD + h*LD + l*D + d   →  dst = value + b*HSL + h*SL + l*D + d
key[b,h,L,d]   (末尾)    :  src = qkv + b*3HD + HD  + h*D + d    →  dst = key   + b*HSL + h*SL + L*D + d
value[b,h,L,d] (末尾)    :  src = qkv + b*3HD + 2HD + h*D + d    →  dst = value + b*HSL + h*SL + L*D + d
```

该公式已用 numpy 按 flat 索引复刻，与 `golden.py` 在 (B=2,H=3,D=5,L=2) 随机数据上逐位一致（uint16 视图比较）。

### 连续性分析

关键结构：**K/V 段在 qkv 内是 `[B,1,H,D]` 的 head-major 连续布局，与输出 `[B,H,·,D]` 的 head-major 排布一致**；pastKey/pastValue 与 key/value 的前 L 片同为 `[B,H,L,D]` 同布局（引自 `expert/01_design.md` §2.2）。因此五类子问题全部退化为"连续段 → 连续段（或规则 strided 段）"的纯 memcpy：

| 子问题 | 源侧 | 目的侧 | 整块性 |
|---|---|---|---|
| Q 重排 | 每 b：qkv 内连续 HD 元素 | query 内连续 | B 个连续→连续整块（H 维天然内联） |
| K 末尾 | 每 b：qkv K 段连续 HD | 每 h 一个 D 片，片间距 SL−D=L*D 元素 | 源连续；目的 H 块规则 strided（blockCount=H, blockLen=D, dstStride=L*D 字节）；L=0 退化连续 |
| V 末尾 | 同上（V 段） | 同上 | 同上 |
| pastKey 复制 | 每 (b,h)：连续 LD 元素 | 连续 | B*H 个连续→连续整块（最大 4MB，需 UB 分片） |
| pastValue 复制 | 同上 | 同上 | 同上 |

推论：① 无任何逐元素散搬；② past 复制是纯 memcpy（如 007 用例 past 段占总搬运量 95%+，是带宽主路径）；③ 沿 L 切分不破坏连续性，可任意分片用于多核负载均衡。

### 搬运量

每用例总搬运字节（`S=2` 字节/元素，L=0 时不含 past 项；引自 `expert/01_design.md` §2.3）：

```text
readEl  = 3*B*H*D + (L>0 ? 2*B*H*L*D : 0)
writeEl = B*H*D + 2*B*H*(L+1)*D          (= 输出元素数)
totalBytes = (readEl + writeEl) * 2
```

35 个公开用例搬运量跨 5 个数量级（12 B ~ 1065 MB）。10 个性能用例全部满足 D%16==0（对齐快路可用）；非对齐 D 只出现在非性能用例 → 通用路径以正确性优先。

## 分核策略（48 AIV）

**只用 AIV**：本算子是纯 MTE2/MTE3 数据搬运、零 Cube 计算，注册为 vector 算子，`blockDim` 按 `GetCoreNumAiv()`（目标环境 48）分配，AIC 空闲。

将全部搬运工作显式划分为三类**目的区间互斥、并集恰为全部输出字节**的单元（unit），round-robin 铺满 blockDim 个 AIV：

```text
unit 编号空间（kernel 侧 div/mod 反解，无数组）：
  [0, numPast)                : past 单元   = 2*B*H*S 个（K/V 两缓存 × (b,h) × L 分片 s）
  [numPast, numPast+numTail)  : 尾token单元 = 2*B 个（K/V × b，每单元 H 个 head 的 strided 写）
  [numPast+numTail, +numQ)    : query 单元  = B 个（每单元 HD 元素连续拷）

Host 侧计算：
  S        = L==0 ? 0 : clamp( ceil(aivNum / (2*B*H)), 1, min(ceil(L*D*2/MIN_UNIT_BYTES), L) )
  lLen     = ceil(L / S)                        # 每 L 分片 token 数（末片截断）
  totalUnits = numPast + numTail + numQ
  blockDim   = min(aivNum, totalUnits)          # 空核立即返回

Device 侧：core c 处理 u % blockDim == c 的所有单元 u。
```

设计取舍：仅按 B 或 (B,H) 切分在 B*H<48 时（如 008 用例 B*H=1）大量核空转，故叠加 L 分片（连续性不破坏）使单元数 ≈ 核数；单元下界 `MIN_UNIT_BYTES=4096` 保证 DMA 效率。负载均衡：单元数 ≥ blockDim 时 round-robin 各核单元数差 ≤1。多核正确性：各单元目的区间互斥（由地址映射公式可证），无跨核写冲突；GM 写侧永远按精确字节长度，不越界覆盖邻居字节。

## TilingData 与 ComputeTiling 纯函数

**TilingData**（Host/Kernel 共享 ABI，扁平标量 POD、显式初值，见 `op_kernel/arch22/t5_split_qkv_append_cache_tiling_data.h`）：

```cpp
struct T5SplitQkvAppendCacheTilingData {
    int32_t b, h, d, l;        // B/H/D/L（l=0 表示无 past 拷贝）
    int32_t blockDim;          // 实际启用核数
    int32_t pastSplit;         // S：past 的 L 分片数
    int32_t lLen;              // 每 L 分片 token 数
    int32_t numPast, numTail, numQ;  // 三类单元数
    int32_t ubTileElems;       // UB 分片元素数（≤32768，双槽 2×64KB）
    int32_t hdElems;           // H*D
};
```

**ComputeTiling** 是无任何 CANN 上下文依赖的纯函数（平台参数 `aivNum`/`ubSizeBytes` 显式传参），ACLNN 的 TilingFunc 与 Kernel 直调 host（及 torch 扩展）都调用它——满足任务书"可复用同一 tiling 计算函数"要求，并使单测同时覆盖两条入口。计算逻辑即上文分核公式，另含 UB 分片元素数决策：

```text
tile = clamp(向下对齐16到 max(HD, lLen*D), 下限16, 上限32768)
再按 UB 预算钳制：2*(tile*2B + 128B 防护带) + 8KB 预留 ≤ ubSizeBytes(192KB)
```

UT（`tests/ut/`）对设计表 + 1002 个 (B,H,L,D) 网格组合断言 blockDim/S/lLen/tile，无 NPU 亦可运行（真机实测 1002 combos PASSED）。

## UB 双缓冲设计

平台约束：arch22 每 AIV UB 192KB，AscendC 无 GM→GM 直拷，必须经 UB 中转。

| Buffer | 用途 | 大小 | 生命周期 |
|---|---|---|---|
| `ubStage[2]`（TBuf, VECALC） | GM→UB→GM staging 双槽 | 2 × ubTileElems×2B = 2×64KB = 128KB | 整 kernel 常驻复用，所有单元共用 |
| 同步事件/标量 | — | <1KB | — |
| 合计 | | ≈129KB ≤ 192KB（利用率 67%） | |

- **双缓冲流水**：MTE2 装 tile i+1 与 MTE3 写 tile i 重叠。tile ≥ 数 KB 时单条 MTE 时长显著大于事件同步开销，双缓冲有正收益（`expert/01_design.md` §4.3 对 chunkcat"细粒度双缓冲反模式"结论做了适用边界限定：该结论针对 1 行/tile 的场景，本设计 tile 粒度相反）。
- **同步纪律**：`SetFlag/WaitFlag` 按事件类型（MTE2_MTE3 / MTE3_MTE2）各取独立事件 ID，严格 1:1 配对；每函数返回时所有已下发 MTE3 全部排空，跨单元无残留占用。
- **对齐快路 + 运行期降级**：fp16/bf16 统一按 uint16 位拷贝。`D%16==0` 且源/目的基址 32B 对齐时装载走 `isPad=false`（对齐快路，含尾 token 单条多块 strided DMA）；任何非对齐（D=3/5/65/255，或额外种子用例 +2B 基址）逐描述符运行期判定，装载用 `DataCopyPad(isPad=true, rightPadding 补齐到 32B)`，**写侧永远按精确字节数**（不用 GM padding，不越界覆盖邻居字节）。正确性不依赖 tilingKey。
- **L=0 双保险**：tiling 层 `numPast=0`，kernel 层 `if (l>0)` 才对 pastKey/pastValue `SetGlobalBuffer`——空指针既不 Set 更不读。

## 校验顺序（C0–C12 固定链）

三个入口（`prepare` / `run` / `run_direct`）共享同一条 O(1) Python 校验链（`solution.py`），ACLNN CheckParams 与直调 host 做 C++ 层等价校验。顺序固定：**容器与个数 → 设备/dtype → 结构 → 尺寸域 → 属性 → 总量**：

| 规则 | 内容 | 拦截场景 |
|---|---|---|
| C0 | attributes 必须为 dict（None 视为空 dict） | attributes_type |
| C1 | 输入个数为 3 | input_count |
| C2 | 全部在 NPU 且同设备 | cpu_input |
| C3 | dtype ∈ {fp16,bf16} 且三输入一致 | float32 / cache_dtype |
| C4 | qkv 结构 [B,1,3HD] | packed_tokens / packed_mismatch |
| C5 | pastKey/Value rank4 且互等 [B,H,L,D] | cache_rank / cache_mismatch |
| C6 | 逐输入 is_contiguous() | noncontiguous |
| C7 | 1≤B≤64、1≤H≤128、1≤D≤256、0≤L≤8192 | shape_limit_* 5 项 |
| C8 | 属性 `type(v) is int`（显式排除 bool——Python `isinstance(True,int)` 的坑） | headNum/headDim 传 str/float/bool |
| C9 | 属性值域 1≤headNum≤128、1≤headDim≤256 | headNum=-1/0、headDim=-1/0 |
| C10 | 属性与形状等值 headNum==H、headDim==D | headNum=9（H=8）、headDim=65（D=64） |
| C11 | 键集合 ⊆ {headNum, headDim} | unknown_attribute |
| C12 | 总输出元素 ≤ 2^28 | 超大量 |

两个固定顺序陷阱：`shape_limit_*` 场景属性为空字典，尺寸域检查（C7）必须先于属性检查；bool 是 int 子类，C8 必须用 `type(v) is int` 而非 isinstance。28 非法场景 × {prepare, run, run_direct} = 84 条全部拒绝，抛 `ValueError/TypeError`，不静默容错。

---

# 实现状态与真机验证

> 以下为 as-built 事实记录，全部数字可由仓内报告复现。

## 交付形态

- ACLNN 算子工程（`T5SplitQkvAppendCache/`）：op_host / op_kernel（arch22）/ op_api / op_graph，`build.sh --soc=ascend910_93` 产出可安装 `.run` 包，`scripts/install_to_user.sh` 安装到用户目录（不覆盖系统 CANN）。
- Kernel 直调工程（`kernel_direct/`）：与 ACLNN 共用同一 `ComputeTiling` 纯函数与同一 kernel 源（`op_kernel/arch22/t5_split_qkv_append_cache_impl.h`），产出 `libt5_split_qkv_ops.so`（torch 扩展，提供 `torch.ops.npu.t5_split_qkv_append_cache` 与 `..._direct` 两个算子）。
- `solution.py`：验收入口 `run` / `prepare` / `run_direct`。
- 实现基线 commit `9aecd19`（本地提交，未 push 远端）。

## ST 真机验证（八段全过）

报告：`T5SplitQkvAppendCache/tests/st/st_report.json`（2026-11-11 12:41，status=passed，465 条记录）。

| 段 | 内容 | 结果 |
|---|---|---|
| A | 35 公开用例 × {run, run_direct} 精度（uint16 逐位）+ 存储独立 + 输入不变 | PASSED（381 条精度记录 0 mismatch） |
| B | 12 额外种子/存储偏移用例（基址 +2B）× 2 入口 | PASSED |
| C | 28 非法场景 × 3 入口 = 84 条拒绝 | 84/84 rejected |
| D | 双 stream（001 fp16、018 bf16 各 12 次） | PASSED |
| E | 20 次重复调用回收 + 峰值预算 | PASSED（run peak_delta=0 / run_direct=512B，预算 out_bytes 12,619,776 + 16KiB） |
| F | 生命周期：8 保留输出 + 32 次输入提前释放/同流写压力 | PASSED |
| G | 连续两次缓存追加（L=31→32→33 链式对照 golden） | PASSED |
| H | run 与 run_direct 交叉验证（35 用例逐位一致） | PASSED |

UT：`bash tests/ut/run.sh` → ComputeTiling 设计表 + 1002 grid combos PASSED。

性能验收说明：ST 不含性能段；任务书 10 个性能用例的正式测量须按 §3.3 协议（预热 50 + 5 轮×200 次取中位）由 `benchmark.py` 完成，本仓库尚未提交正式性能报告，设计期带宽/开销预估见 `expert/01_design.md` §9。

## 关键实现修复（真机实测发现）

1. **DataCopyPad rightPadding 参数单位**（commit `9aecd19`）：CANN 8.5.1 `copy_gm_to_ubuf_align_b16` 的 rightPadding 单位是**元素数**而非字节；按字节传值会使 blockLen+padding 总量非 32B 整倍数，硬件报 "The instruction configuration of MTE is illegal"（error 507035，向量核异常）。修复为 `(32 - blockLen%32)/2` 元素，恒补齐到 32B 对齐；UB 槽防护带（tile+64 元素）吸收 padded 写入。
2. **aclCreateTensor 9 参签名 + storageDims**（commit `906ed13`）：CANN 8.5.1 为 9 参签名，storageDims 传 nullptr 会导致 tiling 阶段报 rank=0，改传 sizes.data()。
3. **输出分配 aclrtMalloc + from_blob**（commit `acd1017`）：ST E 段以"单次调用峰值增量 ≤ 输出字节+16KiB"考核，而调用方 `out = f(out)` 重绑定期间新旧输出并存，输出若走 torch caching allocator（`at::empty`）峰值恒为 2×输出字节。改为 `aclrtMalloc` 直配 + `at::from_blob(deleter=aclrtFree)`，生命周期/存储独立语义不变（F 段验证），实测 peak_delta run=0 / run_direct=512B（仅 tiling 小张量）。
4. **CANN 8.5.1 兼容补丁**（commit `6d6e11b`）：CMake 的 torch_npu 头/库路径推导（`$ENV{T5_OP_INSTALL_PATH}/dav-2201/torch_npu`）；BF16 枚举名 `ACL_BF16`（9.x 才有 ACL_BFLOAT16）；`acl/acl.h` 必须先于 torch_npu 头（旧版 acl_rt.h include guard 抢占）；`.asc` 生成的 host 桩为 C++ 修饰名（`extern "C"` 不生效）；`ACLNN_SUCCESS` 判定改 0。
5. **kernel config json 别名**（commit `0ddf692`）：aclnn 运行时按内核符号名推导查找 config json，与打包工具生成名不一致；落地在 `scripts/install_to_user.sh` 安装后自动补齐 `t_5_split_qkv_append_cache.json` / `t5split_qkv_append_cache.json` 两个别名，重装不丢。
6. **CopyPastUnit 源表索引**（commit `60c74f5`）：cache(0/1) 直接当 GlobalTensor 表索引用导致 key 单元读 qkv、value 单元读 pastKey，改 `cache+1` 对齐表（0=qkv, 1=pastKey, 2=pastValue）。

## 环境

| 项 | 任务书目标口径 | 本仓实测验证链 |
|---|---|---|
| 硬件 | 昇腾 910C（Atlas A3），单逻辑设备 48 AIV + 24 AIC | Atlas A3 真机（soc ascend910_93，`acl.get_soc_name()`=Ascend910_9362，构建目标 ascend910_9391） |
| CANN | 9.1.0 | 系统 CANN 8.5.1（兼容补丁见上节；9.1.0 为目标版本，切换需按任务书"其他版本须重新验证"重跑全量 ST） |
| Python / 框架 | Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4 | Python 3.11.14、torch 2.9.0+cpu、torch_npu 2.9.0.post1 |
| 自定义算子包 | 安装用户目录，不覆盖系统 CANN | `/opt/atomgit/t5_opp`（`vendors/t5_split_qkv_append_cache_custom`） |

证据链命令（可复制重跑）：

```bash
source /opt/atomgit/t5_opp/vendors/t5_split_qkv_append_cache_custom/bin/set_env.bash
export T5_OPS_LIB=$PWD/T5SplitQkvAppendCache/kernel_direct/build/libt5_split_qkv_ops.so ASCEND_DEVICE_ID=0
python3 T5SplitQkvAppendCache/tests/st/run_all.py --solution-file solution.py \
    --report T5SplitQkvAppendCache/tests/st/st_report.json
```
