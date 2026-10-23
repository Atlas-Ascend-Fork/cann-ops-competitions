# SlidingTileAttention 算子设计文档

| 项 | 内容 |
| --- | --- |
| 算子 | SlidingTileAttention（STA），算子类型 `SlidingTileAttention` |
| 平台 | Atlas A2 / 910B3（`ascend910b`），CANN 9.1.0 |
| 形态 | Ascend C aclnn 算子工程（vendor `sta`，接口库 `libsta_opapi.so`） |
| 任务 | CANN 社区任务 2026 `04-1-SlidingTileAttention` |
| 状态 | 设计定稿；实现已完成，精度 54/54 PASS，官方 4 例性能为 GPU A100 的 1.36~1.87× |

# 需求背景（required）

## 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务来源 | CANN 社区任务 2026（`cann-ops-competitions/04_tasks/01_community-task-2026`）任务书《SlidingTileAttention 算子开发任务书》 |
| 对标实现 | FastVideo `fastvideo_kernel` 中的 `sliding_tile_attention`（commit `4ddcdf5`） |
| 交付形态 | Ascend C aclnn 算子工程 + 自测用例 + 自测报告 + 算子 README + 本设计文档 |
| 合入去向 | `ops-transformer` 仓 `experimental/attention` 目录 |

## 背景介绍

### SlidingTileAttention算子实现优化

扩散/视频生成模型（DiT）的自注意力作用在 $(T,H,W)$ 三维 token 网格上，序列长度 $S=T\cdot H\cdot W$（外加 text token）。分辨率或时长上升时，full attention 的 $O(S^2)$ 计算量与 KV 体积都不可接受。Sliding Tile Attention（STA）把注意力稀疏化为两部分：

- **局部**：Image/Video Query 只关注三维空间中相邻 Tile Window 内的 Image/Video KV；
- **全局**：所有 Text token 对所有 Query 可见（text 数量远小于视频 token 数）。

该稀疏结构天然以 3D Tile 为计算单位，单个 Tile 内的 token 在序列中连续（tile-major），因此可以按 Tile 组织 GM 访问与矩阵乘的 K 维，避免逐 token 的随机 gather。FastVideo 在 GPU 侧以 Triton/CUDA kernel 实现该算子，昇腾侧此前无对应实现。

本算子基于 Ascend C 与 CATLASS 的 FlashAttentionInfer block 组件实现，工程路径与依赖如下：

| 类别 | 路径 |
| --- | --- |
| 算子工程 | 本算子目录（`op_host` / `op_host/op_api` / `op_kernel` / `examples` / `tests`） |
| catlass 依赖 | `op_kernel/attn_infra/`：从上游抽取的 48 个头文件（非 submodule、非全量拷贝） |
| QK、PV 矩阵乘 | `op_kernel/attn_infra/include/catlass/gemm/block/block_mmad.hpp`（主模板）+ `block_mmad_fai_qk_normal.hpp` / `block_mmad_fai_pv_normal.hpp`（`MmadAtlasA2FAIQK` / `MmadAtlasA2FAIPV`） |
| softmax、O 累加 | `.../catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp`、`block_epilogue_rescale_o_no_split_row.hpp` |
| 跨核同步 | `.../catlass/arch/cross_core_sync.hpp`（`CrossCoreFlag` / `CrossCoreSetFlag` / `CrossCoreWaitFlag`） |

### SlidingTileAttention参考实现现状分析

本算子为昇腾侧新增算子，无历史 TBE 版本，故模板中「TBE 实现现状分析」一节改为对标 GPU 参考实现（FastVideo `fastvideo_kernel`）的能力与约束分析。

FastVideo 参考实现（GPU，Triton/CUDA）的接口：

```python
def sliding_tile_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window_size: list, text_length: int,
    has_text: bool = True, seq_shape: str = "30x48x80",
) -> torch.Tensor
```

其参数与约束（摘自任务书 §2.4）：

| 参数 | 类别 | 数据类型 | 形状 | 约束 |
| --- | --- | --- | --- | --- |
| q / k / v | 输入 | float16、bfloat16 | `[B,N,S,D]`（BNSD） | 四者 shape、dtype 完全一致；`B,N,S,D > 0`；`S >= T·H·W + text_length`；`has_text=false` 时须 `S == T·H·W`（否则尾行无 task 覆盖）；不支持 broadcast |
| output | 输出 | 同 q | `[B,N,S,D]` | 与 q 一致 |
| window_size | 属性 | list_list_int | — | 每 head 的 `(t,h,w)`，单位 Tile，正奇数；长度 1 广播到所有 head，长度 N 时逐 head 生效 |
| text_length | 属性 | int | 标量 | `>= 0`；`has_text=false` 时按 0 处理 |
| has_text | 属性 | bool | 标量 | `True` / `False` |
| seq_shape | 属性 | string | `"T x H x W"` | Image/Video token 的三维网格尺寸 |

参考实现的语义要点（本算子必须逐条对齐）：

| 要点 | 描述 |
| --- | --- |
| Tile 尺寸 | 固定 $(B_t,B_h,B_w)=(6,8,8)$，单 Tile 384 token |
| 序列排布 | Tile-major：同一 Tile 内 token 连续 |
| 窗口单位 | `window_size` 以 Tile 个数计，逐 head 独立 |
| 可见性 | image query：窗口内 image tile ∪ 全部 text；text query：全局 |
| 边界 | 窗口整体平移（clamp），保持窗口长度不变，不截断 |
| Causal Mask | 无（需要时由上层叠加） |

GPU A100 基线（任务书 §3.3 给定，作为性能判定基线）：

| 用例 | B | H | D | Img_seq | Text_len | Window | GPU A100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 8 | 64 | 30×48×80 | 128 | 逐 head，各维 1~7 | 9 calls，Total 64.204 ms（avg 7.134 ms） |
| B | 1 | 8 | 64 | 36×48×48 | 0 | `(3,3,3)` × H | 8 calls，Total 11.691 ms（avg 1.461 ms） |

移植到 NPU 的四个结构性难点：

1. **可见 KV 集合不规则**：每个 task 的 KV 由「若干条 image run + text 块」拼接而成，无法像规则 FlashAttention 那样按连续区间推进；窗口逐 head 不同，同一 kernel 内的 run 数可相差两个数量级（窗口 `(1,1,1)` 为 3 块，`(7,7,7)` 为 1029 块）。
2. **短 run 放大同步开销**：一个 Tile = 384 token = 3 个 128-token block；当 $M_w=1$ 时一条 run 只有 3 块，若按 run 起流水，QK/softmax/PV 的跨核握手会成为主要开销。
3. **S 必须 fp32 落 GM**：QK 结果由 fixpipe 以 fp32 写入 GM，再由 AIV 读回做 softmax。16 bit S 无法满足 `2^-9` 精度门限（bf16 虽可过门限，但 softmax 前需多一趟 `Cast`，把省下的搬运又还给向量链），因此这笔流量在本结构下无法通过换 dtype 消除。
4. **task 代价不均**：窗口逐 head 不同、text 长度可变，单 task 的 KV 长度从 3 块到 1029 块（未 clamp 的最大窗口）不等，按核静态切分必然失衡，需要细粒度 task + 轮转调度。

### SlidingTileAttention算子功能分析

功能：对 `[B,N,S,D]` 的张量完成一次带 3D tile 稀疏 mask 的 attention。

- 输入：`q`、`k`、`v`（BNSD，float16 / bfloat16）
- 属性：`window_size`、`text_length`、`has_text`、`seq_shape`
- 输出：`output`（与 q 同 shape / dtype）

| Query 类型 | 可见 KV |
| --- | --- |
| Image token（`i < S_img`） | 窗口内 image tile ∪ 全部 text token |
| Text token（`i >= S_img`） | 全部 image tile ∪ 全部 text token |

- 窗口长度保持不变，边界处整体平移（clamp），不做截断；
- 算子本身不含 Causal Mask；
- `has_text = false` 时 `text_length` 视为 0，text 相关搬运与累加分支整体跳过；
- `S > S_img + text_length` 的尾部 token 在 `has_text = true` 时按 text query 参与计算（不报错，语义由调用方保证）；
  `has_text = false` 时 `textBlocks = 0`、不生成任何 text task，因此 host 要求 `S == S_img`，否则报 `GRAPH_FAILED`（否则尾行输出未初始化）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言、以 aclnn 算子工程化方式实现 SlidingTileAttention 算子：

- 支持 float16 / bfloat16，数据格式 BNSD；
- 功能与 FastVideo `sliding_tile_attention` 完全对齐（Tile 语义、窗口 clamp、text 全局、无 causal mask）；
- 精度满足生态算子开源精度标准（fp16 `rtol=atol=2^-9`、bf16 `rtol=atol=2^-6`，`matched_ratio >= 0.99`）；
- 性能不低于 GPU A100 的 0.8 倍（等价于 NPU 耗时 ≤ 1.25× GPU）；
- 适配 Atlas A2（910B3），交付可安装的算子包与可复现的自测用例。

## 需求拆解

1. **算子原型与 aclnn 接口**：OpDef（3 输入 / 1 输出 / 4 属性）、InferShape、InferDataType；`window_size` 为 `list_list_int`，框架无法自动生成接口桩，需手写 aclnn 层。
2. **Tiling**：参数校验、逐 head 窗口表、task 划分、workspace 切分、启动配置。
3. **Kernel**：QK → online softmax → PV → rescale-O 四段计算，AIC + AIV 混合核流水。
4. **不规则 KV 的装载**：image run 与 text 块打包成 stack，逐 block 描述符装载（含跨栈预取）。
5. **dtype 支持**：float16、bfloat16 各编译一份 kernel 二进制。
6. **工程化交付**：CMake 工程、算子包（`.run`）、算子 README。
7. **精度自测**：任务书 2 组样例（fp16/bf16 共 4 例）+ ATK 生成的 50 例，合计 54 例。
8. **性能自测**：`msprof op` 采集 kernel 耗时，与 GPU A100 基线对比，覆盖 fp16/bf16。

# 详细设计（required）

## 算子分析

### 数学公式

**（1）输入与统计量**

$$Q,K,V\in\mathbb{R}^{B\times N\times S\times D},\qquad S=S_{\mathrm{img}}+S_{\mathrm{text}},\qquad S_{\mathrm{img}}=T\times H\times W$$

$$\mathrm{scale}=\frac{1}{\sqrt{D}}$$

**（2）Tile 网格与序列排布**

$$G_T=\frac{T}{B_t},\quad G_H=\frac{H}{B_h},\quad G_W=\frac{W}{B_w},\quad N_{\mathrm{tile}}=G_TG_HG_W,\quad S_{\mathrm{img}}=N_{\mathrm{tile}}\cdot B_tB_hB_w$$

token $(t,h,w)$ 映射到序列下标（Tile-major）：

$$p=\big((q_tG_H+q_h)G_W+q_w\big)\cdot B_tB_hB_w+\big((t\bmod B_t)B_h+(h\bmod B_h)\big)B_w+(w\bmod B_w)$$

其中 $q_t=\lfloor t/B_t\rfloor$，$q_h=\lfloor h/B_h\rfloor$，$q_w=\lfloor w/B_w\rfloor$。

**（3）Sliding Window（clamp）**

窗口以 Tile 个数给出，对应 Tile 窗口尺寸：

$$M_t^{(n)}=\frac{W_t^{(n)}}{B_t},\quad M_h^{(n)}=\frac{W_h^{(n)}}{B_h},\quad M_w^{(n)}=\frac{W_w^{(n)}}{B_w}$$

边界处理为整体平移而非截断：

$$\mathrm{extent}=\min\!\left(M,\,G\right),\qquad \mathrm{center}=\mathrm{clamp}\!\left(q,\ \frac{\mathrm{extent}}{2},\ G-1-\frac{\mathrm{extent}}{2}\right)$$

$$\mathrm{Window}=[\mathrm{center}-\mathrm{extent}/2,\ \mathrm{center}+\mathrm{extent}/2]$$

**（4）可见集合**

$$\mathcal{K}^{(n)}(i)=\begin{cases}\mathcal{K}^{(n)}_{\mathrm{local\ image}}(i)\cup\mathcal{K}_{\mathrm{text}}, & i<S_{\mathrm{img}}\\[2pt] \{0,1,\dots,S-1\}, & i\ge S_{\mathrm{img}}\end{cases}$$

**（5）Attention**

$$s_{i,j}^{(n)}=\frac{Q_i^{(n)}\cdot K_j^{(n)}}{\sqrt{D}},\qquad j\notin\mathcal{K}^{(n)}(i)\iff s_{i,j}^{(n)}=-\infty$$

$$P_{i,j}^{(n)}=\frac{\exp\!\big(s_{i,j}^{(n)}\big)}{\sum_{k\in\mathcal{K}^{(n)}(i)}\exp\!\big(s_{i,k}^{(n)}\big)},\qquad O_i^{(n)}=\sum_{j\in\mathcal{K}^{(n)}(i)}P_{i,j}^{(n)}V_j^{(n)}$$

**（6）在线 softmax 递推（实现形式）**

按 stack 逐块推进，维护行最大值 $m$、行和 $l$、输出累加 $O$：

$$m_{new}=\max\big(m_{old},\ \mathrm{rowmax}(S_s)\big)$$

$$l_{new}=l_{old}\cdot e^{m_{old}-m_{new}}+\mathrm{rowsum}\!\Big(e^{S_s\cdot \mathrm{scale}-m_{new}}\Big)$$

$$O_{new}=O_{old}\cdot e^{m_{old}-m_{new}}+e^{S_s\cdot \mathrm{scale}-m_{new}}\cdot V_s$$

最后一个 stack 结束后做一次归一化 $O\leftarrow O/l$，并转换为输出 dtype。

### 支持数据类型

| 位置 | dtype | 说明 |
| --- | --- | --- |
| q / k / v / output | float16、bfloat16 | 对外仅支持这两种；q/k/v/output 必须一致 |
| S（QK 结果） | float32 | 数值必需：`\|s\| ~ 1e2` 时 fp16 的 ULP 为 0.06，乘 scale 后 logit 误差已贴住 `2^-9` 门限，无法留出余量 |
| P（softmax 输出） | 与输入同 dtype | $P\in[0,1]$，16 bit 提供的是相对精度，误差按比例进入 PV；同时省掉一份 V 的预转换平面 |
| OTmp（PV 结果） | float32 | 供 rescale 累加，末栈归一化后落 16 bit |
| 归约（rowmax / rowsum）与累加 | float32 | 依赖 `BlockReduceSum<float>` / `Max<float>` / `Exp<float>` |

### 支持形状

| 维度 | 约束 |
| --- | --- |
| 排布 | BNSD（`[B,N,S,D]`），q/k/v/output 必须均为 4 维且 shape 完全一致 |
| B | `B >= 1` |
| N | `N <= 128`（窗口表以 `windowTable[3*128]` 内嵌在 TilingData 中） |
| S | `S >= S_img + text_length` |
| D | `1 <= D <= 128`（由 `OTMP_SLOT_ELEMS / Q_ROWS = 128` 决定） |
| seq_shape | `"T x H x W"`，各维须分别被 Tile 尺寸 `6 / 8 / 8` 整除，否则 Tile 网格不整齐，直接报参数校验错误 |

自测覆盖的取值：`D ∈ {16,32,64}`、`B / N ∈ [1,16]`、`S ∈ [69120,115456]`、`text_length ∈ [0,239]`、窗口各维 ∈ `[1,9]`（单位 Tile）、`seq_shape ∈ {18x48x80, 30x48x80, 36x48x48}`；用例构成与判定口径见「精度标准」。

## 算子实现

### 实现方案

整体为「OpDef + Tiling + 单 kernel 入口」的 aclnn 工程：host 侧完成校验、窗口表、task 划分、workspace 切分与启动配置；kernel 侧以 `MIX_AIC_1_2` 方式，由 AIC 完成两次矩阵乘、AIV 完成 softmax 与 O 累加，S/P/OTmp 三块中间结果经 GM 交换。

单个 stack 的数据流：

```
            GM: q / k / v（BNSD，16 bit）
                     │
  AIC   ┌────────────▼─────────────┐  BlockMmad<MmadAtlasA2FAIQK>
        │ QK: Q(128×D) · K(128×D)ᵀ │  → S(128×128, fp32)，fixpipe 直接写 GM
        └────────────┬─────────────┘
                     │ S[slot]
  AIV   ┌────────────▼─────────────┐  BlockEpilogue<EpilogueAtlasA2OnlineSoftmax>
        │ softmax: S → P(16 bit)    │  更新行 max / 行和，写 P 到 GM
        └────────────┬─────────────┘
                     │ P[slot]
  AIC   ┌────────────▼─────────────┐  BlockMmad<MmadAtlasA2FAIPV>
        │ PV: P(128×128) · V(128×D) │  → OTmp(128×D, fp32)，fixpipe 写 GM
        └────────────┬─────────────┘
                     │ OTmp[slot]
  AIV   ┌────────────▼─────────────┐  BlockEpilogue<EpilogueAtlasA2RescaleO>
        │ O ← O·exp(Δm) + O_tmp     │  末栈做 1/l 归一化并落 16 bit
        └───────────────────────────┘
```

关键设计决策：

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| S 的存储精度 | float32 | 见「支持数据类型」：fp16 存 S 的 logit 误差已贴住/超过门限；bf16 虽可过门限，但 softmax 前需多一趟 `Cast`，把省下的搬运还给向量链，净收益不成立 |
| P 的存储精度 | 与输入同 dtype | 相对精度足够；省掉 V 的 fp16 转换平面与一份 workspace |
| 核类型 | `MIX_AIC_1_2` | 每个 block 的向量工作量（softmax 与 O 累加）明显重于 cube（两次 128×128×D 矩阵乘），按 1 AIC : 2 AIV 配比铺满 40 个 AIV，与 20 个 AIC 同拍推进 |
| 流水槽位 | 3 槽 + `PRE_LAUNCH = 2` | QK 与 softmax、PV 与 rescale 各错开 2 个 stack，使 cube / vector 两条链重叠；槽位数必须大于预发射深度，否则会覆盖未读完的槽 |
| KV 打包 | 跨 run 每 4 块一栈 | 避免 $M_w=1$ 的短 run（3 块）独占一轮流水，把跨核握手次数降到约 1/4 |
| S 的行 stride | 512 元素（2 KB） | 满栈固定 4×128 列（末栈可能更少），整栈一次写入；32 B 对齐，尾块空洞靠 `blkCols` 排除而非改 stride |

算子依赖 CATLASS 的三个补丁。本包已把 catlass 用到的代码抽取到 `op_kernel/attn_infra/`，
三个补丁均已**固化进抽取的源码**（构建时不再施加，`patches_baked_in/` 仅作对照）：

| 补丁 | 作用 | 不加的后果 |
| --- | --- | --- |
| `m3_catlass_singleC0_fastpath.patch` | `D = 16` 时源与目标布局逐字节同构，用一条 1D `DataCopy` 代替 `Nd2Nz` | 按 32 B 行拆成多次 transaction，MTE2 事务效率显著下降 |
| `d1_3_spectile384_catlass.patch` | 384 列专用归约（`RowsumSPECTILE384` / `RowmaxSPECTILE384`） | 归约退化到更小 Tile，softmax 变慢 |
| `d2_pack_catlass.patch` | `BlockMmad` 的 QK/PV 增加逐 block 描述符重载（`blkOff` / `blkCols` / `nBlocks` / `hasNext` / `seqStart`） | 无法跨 run 打包，kernel 编译失败（PV 侧缺少 13 参数重载） |

#### 3.2.1 host侧设计：

tiling 策略：host 不按张量维度切分，而是构造「task 图」。一次调用先由属性还原 3D 网格与逐 head 窗口表，再把每 $(b,n)$ 的 query 行按 128 行切成 task；所有几何量（网格、task 数、窗口表、workspace 偏移、scale）一次性写入 `SlidingTileAttentionTilingData`，kernel 只读不算，避免两侧公式漂移。

**aclnn 接口适配**：`window_size` 为 `list_list_int` 形态（`const aclIntArray *const *windowSize, uint64_t windowSizeLen`），CANN 9.x 的 opbuild 无法为该形态生成接口桩，因此 `op_host/op_api/` 下为手写实现，并在 `op_host/CMakeLists.txt` 中以 `ACLNNTYPE aclnn_exclude` 声明该算子（只生成 op proto，不生成 aclnn 桩），手写源码由 `add_modules_sources` 的 `op_api/*.cpp` 规则纳入编译。手写层把 `windowSizeLen` 组 `(t,h,w)` 展平为一维 int 数组，经 `NnopbaseAddIntArrayAttr` 传入；`TilingFunc` 按数组总长度 `3`（广播）/ `3*N`（逐 head）解析。

**TilingData 组成**（Host 填、Kernel 读）：

| 分组 | 字段 | 说明 |
| --- | --- | --- |
| 问题规模 | `batch`、`numHeads`、`seqLen`、`headDim` | B、N、S、D |
| 序列切分 | `imgSeqLen`、`textLen`、`textBlocks`、`textLastCount`、`hasText` | text 末块有效列数用于跳过空洞 |
| tile 网格 | `gridT`、`gridH`、`gridW`、`numTiles` | 由 `seq_shape` 与 Tile 尺寸得到 |
| task 划分 | `imageTasksPerBn`、`textTasksPerBn`、`totalTaskNum` | 分别为 `numTiles*3`、`hasText` 时的 `ceil((S-S_img)/128)`（否则 0）、两者之和乘 `B*N` |
| 数值 | `scaleValue` | $1/\sqrt{D}$ |
| 窗口 | `windowTable[3*128]` | 逐 head 的 `(M_t,M_h,M_w)`，定长内嵌，避免额外 GM 读 |
| workspace | `sBase`、`pBase`、`otBase`、`workSpaceSize` | 相对用户 workspace 起点的字节偏移 |

单个 task 的 4 个参数（`(b,n)` 索引、query 起始行、image 起点与块数、text 起点与块数）由 kernel 按同一套公式从 TilingData 重算，不把 task 表整体搬上 GM。

##### 1. 分核策略：

- **核数**：`nc = platform.GetCoreNumAic()`（910B3 为 20），`context->SetSimdNumBlocks(nc)`；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 使 AIC : AIV = 1 : 2，即 20 AIC + 40 AIV。
- **任务粒度**：1 个 task = 128 行 query。image 侧每个 Tile（384 token）恰好 3 个 task；text 侧每 128 token 1 个 task：

$$\mathrm{totalTaskNum}=B\cdot N\cdot\Big(N_{\mathrm{tile}}\cdot 3+\Big\lceil\frac{S-S_{\mathrm{img}}}{128}\Big\rceil\Big)$$

- **分配方式**：不静态切分区间，AIC 与 AIV 各自按 `taskIdx += coreNum` 轮转取 task。原因是窗口逐 head 不同，单 task 的 KV 长度从 3 块到 1029 块不等，静态切分必然失衡；细粒度 task + 轮转可把差异摊平。
- **同 task 的核内一致性**：AIV 用 `GetBlockIdx() / GetSubBlockNum()` 折算到与 AIC 相同的核序号，两侧对同一 task 使用同一组 workspace 槽位。
- **AIV 的行内拆分**：同一 AIC 对下的 2 个 AIV（`GetSubBlockIdx()` 为 0/1）处理同一个 task，行维由 epilogue 对半拆（`qSBlockSize / 2`，`qSBlockSize` 传入的就是该 task 的行数），因此 AIV 侧的任务轮转步长与 AIC 一致（均为 `nc`）。
- **典型规模**：A 例（`B=1, N=8`，网格 `5×6×10`）共 7208 个 task（每核约 360 个）；B 例（网格 `6×6×6`）共 5184 个（每核约 259 个）。
- **workspace 随核数复制**：S/P/OTmp 三平面按 `nc` 份切分，核间无写冲突。

##### 2. 数据分块和内存优化策略：

三级切分，粒度由外到内：

| 层级 | 粒度 | 说明 |
| --- | --- | --- |
| task | 128 行 query | `Q_ROWS = 128`，与 `L1TileShape::M` 对齐 |
| stack | 4 个 KV block = 512 token | `STACK_BLOCKS = 4`，一次 QK 覆盖 128×512 |
| KV block | 128 token | `KV_BLOCK = 128`，与 `L1TileShape::N/K` 对齐 |

- **L1/L0 分块**：`L1TileShape = L0TileShape = GemmShape<128,128,128>`；QK 的 K 维是 headDim（≤128），PV 的 K 维是 KV 列（128）。L0C 单块 128×128 fp32 = 64 KB，一次 fixpipe 正好落一个 KV block 的 S。
- **S 行 stride = 512 元素（2 KB）**：一个 stack 固定 4×128 列，S/P 每槽实际使用 `Q_ROWS × S_ROW_STRIDE = 128×512 = 65536` 元素，槽容量 `SLOT_ELEMS = 131072`（约为占用量的 2 倍）；512 列整栈写入保证 32 B 对齐，无需按有效列重排。
- **workspace 三平面**（每平面按 `nc` 复制）：

| 平面 | 类型 | 每核元素数 | 20 核合计 |
| --- | --- | --- | --- |
| S | float32 | 3 × 131072 | 30.0 MiB |
| P | 输入 dtype（2 B） | 3 × 131072 | 15.0 MiB |
| OTmp | float32 | 3 × 16384（128×128） | 3.75 MiB |

  合计 48.75 MiB，另加 `GetLibApiWorkSpaceSize()` 的系统区；kernel 侧用 `GetUserWorkspace()` 跳过框架保留的 16 MB 区。

- **UB 分配（192 KB）**：softmax 与 RescaleO 两个 epilogue 共用同一块 `resource.ubBuf`，按 16 KB 为单位静态划分：

| 区间（×16 KB） | 用途 |
| --- | --- |
| `[0,4)` | LS：S 的 UB 暂存（fp32，双缓冲 2 × 32 KB） |
| `[4,6)` | LP：P 的 UB 暂存（16 bit，双缓冲 2 × 16 KB） |
| `[6,8)` | LO：RescaleO 的 O_local（fp32） |
| `[8,10)` | GO：O 的搬出缓冲（16 bit 与 fp32 视图复用） |
| `10*16K` 起 14 KB | 行归约标量（LM/HM/GM/LL/GL/DM）与 1 KB 临时区 |
| `[11,12)` | mask 区：仅 masked 重载使用，本 kernel 走 NO_MASK 重载，不占用 |

- **双缓冲**：L1/L0 的 ping-pong 由 CATLASS `BlockMmad` 内部管理，描述符中的 `hasNext` 把下一栈首块一并纳入预取窗口；S/P/OTmp 的 3 槽环由 kernel 用 flag 管理（见 3.2.2）。
- **两处刻意不做**：① Q 不常驻 L1（窗口内 K/V 的复用率远高于 Q）；② 不为 V 建 fp16 转换平面（bf16 直接用 bf16 参与 mma），省一份 workspace 与一次全量转换。

##### 3. tilingkey规划策略：

本算子**不使用 tilingkey**：kernel 只有一个入口，所有差异由 `TilingData` 承载。

| 差异维度 | 承载方式 |
| --- | --- |
| dtype（fp16 / bf16） | 编译期宏 `DTYPE_Q` / `DTYPE_K`，构建期展开为两份 kernel 二进制（产物 2 个 `.o`，运行期由 binary_info_config 的 key 选择） |
| 是否有 text | `hasText` + `textBlocks = 0`，text 分支按 0 次循环自然跳过 |
| 窗口逐 head 不同 | `windowTable[3*N]`，task 解码时按 head 查表 |
| text 末块不满 128 | `textLastCount`，只影响最后一栈的归约列数与 PV 的 K 上界 |
| 网格与序列长度 | `gridT/H/W`、`seqLen`、`imgSeqLen` |

这样取舍的原因是：形状差异只出现在**外层 task/stack 解码**，内层主体循环（128×128 的 QK/softmax/PV）与形状无关，用 tilingkey 拆分支只会放大二进制数量而不带来内层收益；反之 dtype 必须拆成两份二进制，因为它决定 mma 指令与 P 的位宽。

**数据检测**：`TilingFunc` 对输入与属性做显式校验，违反即返回 `GRAPH_FAILED` 并打印原因：

| 检查项 | 规则 |
| --- | --- |
| 维度与形状 | q/k/v/output 均为 4 维、各维 > 0，四者 shape 与 dtype 完全一致（不支持 broadcast） |
| dtype | 仅 float16 / bfloat16 |
| head 数 | `N <= 128` |
| headDim | `1 <= D <= 128` |
| seq_shape | `T x H x W`，各维分别被 `6 / 8 / 8` 整除 |
| 序列长度 | `S >= T·H·W + text_length`；`has_text = false` 时须 `S == T·H·W`（尾行无 task 覆盖，不放过） |
| 窗口 | 展平后长度 ∈ {3, 3N}；每组 `(t,h,w)` 为正奇数 |
| text | `text_length >= 0`；`has_text = true` 时须 `> 0` |
| SoC | `ascend910b` |

#### 3.2.2 kernel侧设计：

kernel 入口分两段：**Init**（取 tiling、置混合核类型、按块预置硬件 event flag）与 **Process**（按 task 轮转，task 内按 stack 走四段计算）。AIC 与 AIV 是同一入口下的两条编译路径，由 `operator()<g_coreType>` 的显式特化（`AIC` / `AIV`）分派。

1. **dtype 与组件**：`ElementQ = DTYPE_Q`、`ElementK = DTYPE_K`、`ElementP = ElementQ`、`ElementS = ElementOTmp = float`；四段计算分别实例化 `BlockMmad<MmadAtlasA2FAIQK>`、`BlockEpilogue<EpilogueAtlasA2OnlineSoftmax>`、`BlockMmad<MmadAtlasA2FAIPV>`、`BlockEpilogue<EpilogueAtlasA2RescaleO>`。
2. **task / stack 解码（CopyIn 前置）**：每个 task 先算出「可见 KV 的 block 列表」——image 侧每 run 固定 `(t,h)`，沿 w 方向连续 `3·M_w` 块；text 侧为 `textBlocks` 块。再把 block 列表按 4 块一栈打包，得到 `nStacks`。每栈生成描述符 `blkOff[]`（块在序列内的 token 起点）、`blkCols[]`（有效列数）、`nBlocks`、`hasNext`、`seqStart`，交给 `d2` 补丁提供的 `BlockMmad` 逐 block 重载。**不变量**：只有 text 末块可能 `blkCols < 128`，且它固定在栈尾，因此空洞不进入 softmax 归约区间，也不进入 PV 的 K 维。
3. **三槽流水与跨核同步**：S/P/OTmp 各 3 个槽位，AIC 与 AIV 用 3 个 `CrossCoreFlag` 握手（`qkReady=1`、`softmaxReady=2`、`pvReady=3`）：
   - AIC：`QK(s)` 完成后 `CrossCoreSetFlag(qkReady, PIPE_FIX)`；`PV(s-2)` 把 `softmaxReady` 作为形参交给 `BlockMmadPV` 内部 `CrossCoreWaitFlag`；完成后 `CrossCoreSetFlag(pvReady, PIPE_FIX)`；
   - AIV：`CrossCoreWaitFlag(qkReady)` → `softmax(s)` → `CrossCoreSetFlag(softmaxReady, PIPE_MTE3)`；`CrossCoreWaitFlag(pvReady)` → `rescale(s-2)`；
   - `static_assert(NUM_SLOTS >= PRE_LAUNCH + 1)`：同一槽位被复用前，另一侧的读必须已完成；
   - 入口还按块预置一组硬件 event（AIC：`M_MTE1` / `FIX_M` / `MTE1_MTE2`；AIV：`MTE3_V` / `MTE3_MTE2` / `V_MTE2`），出口逐一 `WaitFlag` 配对，避免核退出时留下未消费的 flag。
4. **边界处理**：
   - image 网格各维均可被 Tile 尺寸整除（host 已校验），不存在不满 Tile 的行；
   - text 末块 `textLastCount < 128`：只影响该栈 softmax 的归约列数与 PV 的 K 循环上界；
   - `has_text = false`：`textBlocks = 0`，可见 KV 仅由 image run 组成；
   - 栈首/栈尾的 L1 预取由 `seqStart` / `hasNext` 决定，避免跨栈重复搬运。
5. **流程**（单个 task）：

```
# AIC 与 AIV 各自跑自己的循环，下面按时间叠加展示两边的相位
for task = blockIdx; task < totalTaskNum; task += coreNum:
    decode(task) -> (b, n, qRowStart, imageRuns, textBlocks)
    pack run/text blocks -> stacks (每栈 <= 4 块)
    for s in 0 .. nStacks-1:
        AIC: QK(s)                      // Q(128×D)·K(128×D)ᵀ -> S(fp32), slot s%3
        AIV: softmax(s)                 // S -> P(16bit)，更新 m / l
        AIC: if s >= 2: PV(s-2)         // P(128×128)·V(128×D) -> OTmp(fp32)
        AIV: if s >= 2: rescale(s-2)    // O = O·exp(Δm) + OTmp
    tail: 各自排空流水（剩余 PV / rescale），末栈 rescale 做 1/l 归一化后写 O
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas 800I/T A2（`ascend910b`） | √ | 已适配：910B3（20 AIC / 40 AIV），CANN 9.1.0 |
| Atlas A3 / 950 | — | 未适配：算子注册的 SoC 仅 `ascend910b`，kernel 依赖 `Arch::AtlasA2` |

## 算子约束限制

1. **平台**：仅 `ascend910b`（Atlas A2 系列），未适配 A3 / 950。
2. **dtype**：仅 float16 / bfloat16，且 q/k/v/output 必须一致。
3. **形状**：4 维 BNSD；`N <= 128`；`D <= 128`；`seq_shape` 各维须分别被 `6 / 8 / 8` 整除。
4. **序列长度**：`S >= S_img + text_length`；`has_text = true` 时多出的尾部 token 按 text query 计算（不报错，语义由调用方保证）；`has_text = false` 时**必须 `S == S_img`**——此时不生成 text task，`[S_img, S)` 的尾行没有任何 task 覆盖，host 直接报错而不返回未初始化输出。
5. **窗口**：长度 3（广播）或 3N（逐 head），各维为正奇数；实际窗口会被 clamp 到网格内，长度不变。
6. **无 Causal Mask**：与参考实现一致，需要时必须由上层叠加。
7. **精度边界**：`max_abs_error` 的理论上界为输出 dtype 的末位量化噪声（fp16 3.9062e-03、bf16 3.1250e-02）；显著超出通常意味着窗口或 text 语义用错，而非数值误差。判定见「精度标准」。
8. **依赖版本**：catlass 代码以抽取件形式随算子仓交付（`op_kernel/attn_infra/`），3 个补丁已固化进抽取源码；抽取件与上游文件结构绑定，升级 catlass 需重新验证补丁可施加。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

| 项 | 内容 |
| --- | --- |
| 比对方法 | 逐元素判定 `\|a - g\| <= atol + rtol · \|g\|`（`a` 为 NPU 输出，`g` 为 golden），统计 `matched_ratio` 与 `max_abs_error` |
| float16 门限 | `rtol = atol = 2^-9`；`matched_ratio >= 0.99`；`max_abs_error <= 1e-1` |
| bfloat16 门限 | `rtol = atol = 2^-6`；`matched_ratio >= 0.99`；`max_abs_error <= 1e0` |
| 判定口径 | `matched_ratio` 与 `max_abs_error` 两项同时满足才算 PASS。`max_abs_error` 的理论上界是输出 dtype 的末位量化噪声（fp16 3.9062e-03、bf16 3.1250e-02）；显著超出按语义错误（窗口 / text / 排布）排查，不放宽阈值 |
| golden | GPU 上 FastVideo `sliding_tile_attention`（`scripts/make_fixture_triton.py` 生成 `meta.txt` + `input/{q,k,v}.bin` + `output/golden.bin`） |
| 用例覆盖 | 任务书 §3.3 的 2 组形状 × fp16/bf16 = 4 例；ATK 按任务书输入分布生成 50 例（`D ∈ {16,32,64}`、`B/N ∈ [1,16]`、窗口各维 ∈ [1,9]、`has_text` 两种取值均覆盖），合计 54 例 |
| 复现 | `bash scripts/build.sh` 构建算子包后，用 `sta_test --root <fixture根>` 逐例比对；54 例 fixture（4 例任务书形状 + 50 例 ATK 生成）与测试代码随算子仓交付 |

### 性能标准

| 项 | 内容 |
| --- | --- |
| 指标 | 单次 aclnn 调用（`[B,N,S,D]` 一次算完全部 head）的 kernel 时间 |
| 基线 | GPU A100 上 FastVideo 参考实现的 Total time（任务书 §3.3）：A 例 64.204 ms（9 calls）、B 例 11.691 ms（8 calls） |
| 目标 | 官方 4 例 NPU 耗时 ≤ **1.25×** GPU A100（与 0.8 倍 GPU 持平），fp16 / bf16 分别判定 |
| 采集 | `msprof op --launch-count=1 --warm-up=1 --aic-metrics=Default` 取 `OpBasicInfo.csv` 的 `Task Duration`；`sta_test --perf --repeat 20` 的 aclrtEvent 计时用于交叉校核 |
| 判定口径 | 以同会话、同 fixture 的 `min` 为准；`mean` 受首发降频影响，不作判据 |
| 扩展口径 | 54 例全量另统计「聚合比 ΣNPU/ΣGPU」「中位比」「达标例数」，用于观察分档差异，不替代官方 4 例的达标判定 |

### 测试方案

| 项 | 内容 |
| --- | --- |
| 精度入口 | `sta_test --root <数据根>`：逐例输出 `matched_ratio` 与 `max_abs_error`，按上表门限判定 |
| 性能入口 | `scripts/eval.sh`：逐例 `msprof op` 采集并汇总到 `out/msprof_op` |
| 交叉校核 | `sta_test --perf --repeat 20` 的 aclrtEvent 计时与 `Task Duration` 对照，偏差 > 5% 时复采 |
| 复现前提 | `ASCEND_CUSTOM_OPP_PATH` 指向本次构建的 `vendors/sta`；改过 catlass 头文件须先删 kernel 产物再构建 |

> 判定结果：54 例精度全部 PASS（fp16 `max_abs_error` ≤ 3.9e-3、bf16 ≤ 3.1e-2，`matched_ratio` = 1.0）；官方 4 例性能为 87.44 / 90.70 / 21.16 / 21.88 ms，对应 GPU A100 的 1.36 / 1.41 / 1.81 / 1.87×，均未达到 1.25× 目标，优化方向的实验与否证台账随交付材料单独提供。

## 兼容性分析

- **新增算子**：不修改、不替代既有算子，OpDef 与 aclnn 接口均为新增，不涉及既有图/模型的兼容性问题。
- **与参考实现的语义兼容**：Tile 尺寸、tile-major 排布、窗口 clamp、text 全局、无 causal mask 均与 FastVideo 对齐；`has_text = false` 与 `text_length = 0` 等价。
- **接口兼容**：`window_size` 支持长度 3（广播）与 3N（逐 head）两种形态；`seq_shape` 同时接受 `x` 与 `X` 分隔符。
- **命名隔离**：vendor 名为 `sta`、aclnn 库为 `libsta_opapi.so`，安装到 `vendors/sta` 目录，与内置算子及其他自定义算子互不覆盖；卸载即删除该目录。
- **版本依赖**：依赖 CANN ≥ 9.0（当前 9.1.0）与随包交付的 catlass 抽取件；升级 CANN 或替换 catlass 抽取件需重跑 54 例回归。
