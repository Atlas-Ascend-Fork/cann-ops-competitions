# VarLenNdRotaryPosEmb算子

## 一、赛题背景

视频生成大模型在推理时会把一批视频按帧打包成一条长序列送进 attention：所有 token 连续排布在最后一个维度之前，**每帧的 token 数由该帧的分辨率决定**（动态分辨率 / 多镜头混跑时各帧宽高并不相同），**每帧的时间戳由采集时间决定**（可变帧率下帧间隔不相等），而**帧在张量里的存储顺序未必等于它的时间顺序**（稀疏采样、关键帧优先、多镜头时间轴重叠都会让存储序与时间序解耦）。

于是每个 token 的位置坐标不再能从下标直接算出：它要先在一张"每帧 token 数"表上定位自己属于哪一帧，再在该帧的宽高上反解行列，最后按存储帧记下的位置索引去取时间戳。这就是 **n-d RoPE 的变长/变帧率形态（VarLen ND RoPE）**：位置编码的三个轴坐标全部由输入数据决定。

本题要求基于视频生成模型的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 VarLenNdRotaryPosEmb 算子。

> 本题沿用 3D RoPE 的配对与旋转规则，并把网格尺寸、帧时间戳、帧存储顺序全部改为由输入数据决定：帧长可变、每帧分辨率可变、存储序与时间序解耦。

## 二、算子功能描述

实现的 VarLenNdRotaryPosEmb 算子需完成以下核心计算：

1. **帧定位**：由每帧 token 数构造帧起点表（前缀和），把每个 token 定位到它所属的存储帧
2. **时间基准**：由每帧时间增量构造时基表（前缀和），并按"该存储帧的位置索引"取出它的时间戳
3. **帧内坐标**：按该帧自己的宽度反解出行、列坐标
4. **三轴频率表**：按段长 `[d_t | d_h | d_w]` 分别计算逆频率，由三轴坐标乘逆频率得到旋转角，进而生成 cos/sin（段内交错布局）
5. **分段旋转**：张量最后一维 D 按 `[d_t | d_h | d_w]` 三段切分，每段内部按**相邻对**配对旋转（角度取对应轴、对应频率的旋转角）
6. **输出**与输入同 shape 同 dtype 的结果

## 三、核心定义与约束

### 3.1 参考算子

变长帧 3D RoPE 的等价实现（FP32 中间精度）：

```python
import torch


def var_len_nd_rotary_pos_emb(x, frame_tokens, frame_widths, frame_deltas, frame_pos,
                              rope_dim_list, theta=10000.0):
    """x: (S, N, D)；F = len(frame_tokens)；S = sum(frame_tokens)"""
    S, N, D = x.shape
    F = frame_tokens.shape[0]

    # 1. 存储帧起点（排他前缀和）与时间基准（排他前缀和）
    starts = torch.cat([torch.zeros(1), torch.cumsum(frame_tokens, 0)[:-1]]).long()
    times = torch.cat([torch.zeros(1), torch.cumsum(frame_deltas, 0)[:-1]]).float()

    # 2. 每个 token：定位帧 -> 帧内反解 (h, w) + 取出时间戳
    sid = torch.arange(S)
    j = torch.searchsorted(starts, sid, right=True) - 1     # 所属存储帧
    idx = sid - starts[j]                                   # 帧内序号
    w = (idx % frame_widths[j]).float()
    h = (idx // frame_widths[j]).float()
    t = times[frame_pos[j]]                                 # 存储序与时间序解耦

    # 3. 三轴各自的 cos/sin（段内按 repeat_interleave(2) 交错布局）
    cos_parts, sin_parts = [], []
    for axis, d_a in enumerate(rope_dim_list):
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_a, 2)[: d_a // 2].float() / d_a))
        angle = [t, h, w][axis].unsqueeze(-1) * inv_freq     # (S, d_a/2)
        cos_parts.append(angle.cos().repeat_interleave(2, dim=-1))
        sin_parts.append(angle.sin().repeat_interleave(2, dim=-1))
    cos = torch.cat(cos_parts, dim=-1).reshape(S, 1, D)
    sin = torch.cat(sin_parts, dim=-1).reshape(S, 1, D)

    # 4. 相邻对旋转（FP32）
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    y1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
    y2 = x2 * cos[..., 1::2] + x1 * sin[..., 1::2]
    return torch.stack((y1, y2), dim=-1).flatten(-2).to(x.dtype)
```

### 3.2 数学公式

记 **F** 为存储帧数，第 j 个存储帧的 token 数 `n_j`、宽 `W_j`（高 `H_j = n_j / W_j`）、时间增量 `Δ_j`、位置索引 `p_j`；记 S = Σ n_j。

**存储帧起点（前缀和）**：

$$
\text{start}_j = \sum_{k<j} n_k, \qquad
\text{token } s \in [\text{start}_j,\ \text{start}_j + n_j) \Leftrightarrow \text{帧号 } j
$$

**时间基准与时间戳（前缀和 + 位置索引查表）**：

$$
\text{time}_j = \sum_{k<j} \Delta_k \ (\text{time}_0 = 0), \qquad
t(s) = \text{time}_{p_j}, \quad j = \text{帧号}(s)
$$

**帧内坐标反解**（除数为该帧自己的宽）：

$$
\text{idx} = s - \text{start}_j, \qquad
w = \text{idx} \bmod W_j, \qquad h = \text{idx} \div W_j
$$

**三轴逆频率**（第 a 段长 $d_a$）与旋转角：

$$
\text{inv\_freq}_a[i] = \theta^{-2i/d_a}, \qquad
\varphi_a(s) = \text{coord}_a(s) \cdot \text{inv\_freq}_a[i], \quad \text{coord} = (t, h, w)
$$

**cos/sin 段内交错布局**（段 a 占 D 维中的第 $\sum_{b<a} d_b$ 起 $d_a$ 个位置）：

$$
\cos_{\text{seg}_a}[2i] = \cos_{\text{seg}_a}[2i+1] = \cos(\varphi_a[i]), \qquad
\sin_{\text{seg}_a}[2i] = \sin_{\text{seg}_a}[2i+1] = \sin(\varphi_a[i])
$$

**相邻对旋转**（对每个相邻通道对 $(2m, 2m+1)$）：

$$
y[2m] = x[2m]\cdot\cos[2m] - x[2m+1]\cdot\sin[2m], \qquad
y[2m+1] = x[2m+1]\cdot\cos[2m+1] + x[2m]\cdot\sin[2m+1]
$$

注意：**配对方式为相邻对（interleave），不是前半/后半配对（rotate-half）**，与 HuggingFace Llama 风格不同。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | (S, N, D) | float16, float32 | ND | 待旋转张量，S = Σ n_j |
| INPUT(必选) | frame_tokens | tensor | (F,) | int32 | ND | 第 j 个存储帧的 token 数 n_j = H_j·W_j |
| INPUT(必选) | frame_widths | tensor | (F,) | int32 | ND | 第 j 个存储帧的宽 W_j（须整除 n_j）|
| INPUT(必选) | frame_deltas | tensor | (F,) | float32 | ND | 帧时长，时基为其排他前缀和 |
| INPUT(必选) | frame_pos | tensor | (F,) | int32 | ND | 第 j 个存储帧在时基表中的位置索引（可重复、可乱序）|
| ATTR(属性) | rope_dim_list | list_int | [3] | - | - | 三轴分段 [d_t, d_h, d_w]，各段偶数、和为 D |
| ATTR(属性) | theta | float | - | - | - | RoPE 基础频率基数，默认 10000.0 |
| OUTPUT(输出) | y | tensor | (S, N, D) | 与 x 一致 | ND | 旋转位置编码结果 |

### 3.4 关键输入约束

- **数据类型**: float16, float32（三个索引/时基输入固定 int32/int32/int32/float32）

- **维度场景**: 3 维输入（S, N, D），N 为 head 数

- **维度取值范围（均为正整数）**:
  - S = Σ n_j（token 总数）: [1, 262144]
  - F（存储帧数）: [1, 128]
  - N（head 数）: [1, 128]
  - D（head 维度）: [8, 4096]，且为偶数
  - n_j（每帧 token 数）: [2, 65536]
  - W_j（每帧宽）: [1, n_j]，且 `n_j % W_j == 0`
  - d_t, d_h, d_w: 均为偶数且 ≥ 2，满足 `d_t + d_h + d_w = D`
  - Δ_j（帧时长）: ≥ 0
  - p_j（位置索引）: [0, F)，**允许重复、允许乱序**

- **一致性约束**: `len(frame_tokens) == len(frame_widths) == len(frame_deltas) == len(frame_pos) == F`；`sum(frame_tokens) == S`；`x.shape[0] == S`

- **数值语义**: x 为任意实数，无范围限制，建议取值 [-10, 10]

### 3.5 核心属性说明

- **rope_dim_list**（list_int）: 三个轴的频率段长。段 i 的旋转角按第 i 轴坐标（依次为 t、h、w）计算；分段边界为偶数位，相邻对不会跨段
- **theta**（float，默认 10000.0）: 基础频率基数。越大则整体旋转角越小，各段内部频率跨度不变；测试用例中取到 1e6

### 3.6 输出严格要求

- **形状约束**: y.shape == x.shape
- **类型约束**: y 的数据类型与 x 一致
- **数值等价**: 输出与参考实现在合理浮点误差范围内一致
- **中间精度要求**: 前缀和、帧定位、坐标反解、inv_freq、角度、cos/sin 及旋转乘加必须使用 FP32 精度；仅输入输出为 float16 时按对应精度存取
- **顺序约束**: 输出与输入 **token 顺序一一对应**（不做任何重排）

### 3.7 特殊值处理规则

- **零坐标 token**: 存在存储帧满足 `p_j = 0` 且其帧内序号为 0 的 token，其 $(t,h,w)=(0,0,0)$，输出与输入逐位一致
- **存储序与时间序解耦**: `frame_pos` 可以乱序、可以重复（多个存储帧共享同一时间戳），算子不得假设 `p_j` 单调、不得假设时间戳互不相同
- **每帧尺寸不同**: `W_j`、`n_j` 逐帧变化，帧内反解的除数是数据依赖的，算子不得假设全局统一的 H/W
- **配对方式**: 相邻对配对（`2m` 与 `2m+1`），角度在段内交错复制；不是 rotate-half 的前后半配对
- **分段与轴对应**: 第 i 段使用第 i 轴坐标；分段顺序不可交换
- **非对齐**: D、N、S 与各帧边界的任意组合可能产生非对齐访问，算子需正确且高效地处理
- **大坐标/大 theta**: 时间戳可达数百（Δ 可达 5、F 可达 128），theta 可达 1e6，旋转角精度依赖 FP32 计算
- **NaN/Inf 输入**: 输入含 NaN/Inf 时输出对应位置为 NaN/Inf，算子不应崩溃
- **确定性**: 相同输入与属性必须产生完全相同的输出

## 四、规则要求

1. **cos/sin 生成规则**: 不得预先物化全量 cos/sin 作为常量绕过计算（允许在 kernel 内按等价结构计算/复用）

2. **实现方式规则**: 旋转的具体实现方式不限（如复数乘法、相邻对公式、重排后 rotate-half 等任意数学等价写法），任何满足数值要求的实现均可

3. **输入不可变**: 算子执行不得修改任何输入的内容；workspace 自行管理

4. **赛事规则**: 本赛题的实现与提交必须遵守《赛事规则》，完整内容如下：

   1、请按照题目要求完成对应的AscendC算子，补齐算子Host侧的Tiling逻辑和Device侧的Kernel逻辑，其中Host侧仅负责参数校验、shape推导、tiling逻辑和启动配置；Kernel侧负责完成该算子完整的计算流程。

   2、请按照规则完成算子开发，严禁出现违规情况，包括但不限于：

      - 1）在Host侧读取输入张量进行计算。
      - 2）在Host侧对输入数据进行重排。
      - 3）在Host侧生成预查表，传入Kernel侧直接取值计算。
      - 4）在Host侧完成向量、矩阵计算，将结果传入Kernel侧直接用于计算。
      - 5）硬编码、按测试用例或者数据生成规律跳算。
      - 6）写入标记跳过计算。
      - 7）利用旧输出或者未初始化内存跳过计算。
      - 8）逐条匹配评测用例编写专属实现。
      - 9）修改或者干扰判题、精度校验和计时流程。
      - 10）注入空kernel、识别预热轮次或把实际计算移出计时范围等行为。
      - 11）在多轮次的判题中加缓存来减少耗时。
      - 12）一次计算通过多次kernel来完成计算。

   3、比赛结束后，会进行最终的人工核验，如果出现未按照规则提交的代码，赛事官方会按违规代码作废成绩。

## 五、精度判断规则

精度要求：计算结果需满足以下精度误差要求：

- float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
- float16、bfloat16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
- int32：要求计算结果完全准确，无误差

## 六、示例说明

**示例1**: 变长帧 + 时间序解耦（F=2, frame_tokens=[6,6], frame_widths=[3,3], frame_deltas=[1.0,2.0], frame_pos=[1,0], D=8, rope_dim_list=[4,2,2], theta=10000.0）

- 输入 x: shape=[12, 1, 8]，所有 token 取 `[1, 0, 1, 0, 1, 0, 1, 0]`（每对起点为 1，便于直接读出旋转角）
- 前缀和: `starts=[0,6]`（帧 0 占 token 0~5，帧 1 占 token 6~11）；`times=[0.0, 1.0]`
- 时间戳: 存储帧 0 的 `frame_pos=1` → t=1.0；存储帧 1 的 `frame_pos=0` → t=0.0（**存储序与时间序相反**）
- 帧内坐标（W=3）: idx 0~5 → (h,w) = (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
- 输出 y（节选）:

| token | 存储帧/帧内序号 | (t, h, w) | y |
|---|---|---|---|
| 0 | 帧0 / 0 | (1.0, 0, 0) | [0.540302, 0.841471, 0.999950, 0.00999980, 1, 0, 1, 0]（T 段两个频率：角度 1 与 0.01）|
| 1 | 帧0 / 1 | (1.0, 0, 1) | [0.540302, 0.841471, 0.999950, 0.00999980, 1, 0, 0.540302, 0.841471]（再叠加 W 段旋转 1 rad）|
| 4 | 帧0 / 4 | (1.0, 1, 1) | [0.540302, 0.841471, 0.999950, 0.00999980, 0.540302, 0.841471, 0.540302, 0.841471]（三轴同时旋转）|
| 6 | 帧1 / 0 | (0.0, 0, 0) | [1, 0, 1, 0, 1, 0, 1, 0]（零坐标 → 恒等）|

- 校验: 对任意 token，每对 $(x[2m], x[2m+1])$ 的平方和不变（正交旋转）；零坐标 token 与输入逐位一致

**示例2**: 视频实际场景（F=13, x shape=[20280, 24, 128], dims=[16,56,56]，帧长不等长、帧内宽度逐帧不同、frame_pos 块内乱序）

- 校验性质: 同一存储帧内所有 token 共享同一时间戳；帧内 (h,w) 覆盖 `[0,H_j)×[0,W_j)`；任意 token 各分段内相邻对的平方和不变

**示例3**: 大规模压测（F=64, x shape=[131072, 24, 128], dims=[44,42,42], theta=10000.0，frame_pos 全随机）

- 覆盖: 131072 个 token 的帧定位 + 逐帧不同网格 + 时间戳乱序复用