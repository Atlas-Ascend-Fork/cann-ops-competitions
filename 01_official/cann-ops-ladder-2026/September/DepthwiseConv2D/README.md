# DepthwiseConv2D算子

## 一、赛题背景

深度可分离卷积（Depthwise Separable Convolution）是轻量级卷积神经网络（如 MobileNet 系列、EfficientNet、ShuffleNet）的核心构件。它将标准卷积拆分为两步：**深度卷积（Depthwise Convolution）**与**逐点卷积（Pointwise Convolution）**。其中深度卷积对每个输入通道独立施加一个 2D 卷积核，不进行跨通道加权，极大降低了参数量与计算量。

本题聚焦于深度卷积部分（DepthwiseConv2D），要求基于 PyTorch 中 `torch.nn.Conv2d`（groups=C 时）的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 DepthwiseConv2D 算子。

## 二、算子功能描述

实现的 DepthwiseConv2D 算子需完成以下核心计算：

1. **零填充**：对输入张量在 H、W 方向两侧各填充 `pad` 行/列零值。
2. **滑动窗口卷积**：以步长 `stride`、膨胀率 `dilation` 滑动卷积核，对每个通道独立做 2D 卷积。
3. **偏置叠加**：将 per-channel 偏置 `b` 加到对应通道的所有输出元素上。

算子的核心难点在于：每通道独立卷积导致 Cube 单元难以直接利用（无跨通道矩阵乘），需高效设计 Vector 向量化乘加或 im2col + Cube 混合策略；滑动窗口带来非连续内存访问，需要合理的 Tiling 分块与 Double Buffer 流水设计。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch 原生算子：`torch.nn.Conv2d`（`groups = in_channels`，即 depthwise 模式）

等价 numpy 实现：

```python
import numpy as np

def depthwise_conv2d(x, w, b=None, stride=1, pad=0, dilation=1):
    # x: (N, C, H, W), w: (C, 1, kH, kW), b: (C,)
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape

    if pad > 0:
        x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    H_out = (H + 2*pad - dilation*(kH-1) - 1) // stride + 1
    W_out = (W + 2*pad - dilation*(kW-1) - 1) // stride + 1

    x32 = x.astype(np.float32)
    w32 = w.astype(np.float32)
    y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    for kh in range(kH):
        for kw in range(kW):
            hs, ws = kh * dilation, kw * dilation
            patch = x32[:, :, hs:hs+(H_out-1)*stride+1:stride, ws:ws+(W_out-1)*stride+1:stride]
            y += patch * w32[:, 0, kh, kw].reshape(1, C, 1, 1)

    if b is not None:
        y += b.astype(np.float32).reshape(1, C, 1, 1)

    return y.astype(x.dtype)
```

### 3.2 数学公式

#### 3.2.1 输出尺寸计算

卷积输出在空间维度上的尺寸由输入尺寸 $H$/$W$、卷积核尺寸 $k_H$/$k_W$、步长 $\mathit{stride}$、填充量 $\mathit{pad}$ 和膨胀率 $\mathit{dilation}$ 共同决定：

$$H_{out} = \left\lfloor \frac{H + 2 \cdot \mathit{pad} - \mathit{dilation} \cdot (k_H - 1) - 1}{\mathit{stride}} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W + 2 \cdot \mathit{pad} - \mathit{dilation} \cdot (k_W - 1) - 1}{\mathit{stride}} \right\rfloor + 1$$

其中 $\lfloor \cdot \rfloor$ 为向下取整。公式含义：填充后的有效输入尺寸为 $H + 2 \cdot \mathit{pad}$，膨胀卷积核的实际感受野为 $\mathit{dilation} \cdot (k_H - 1) + 1$（膨胀率 >1 时卷积核元素之间插入 $\mathit{dilation} - 1$ 个间隔），滑动窗口需在有效输入范围内每次前进 $\mathit{stride}$ 步，能滑动的次数即为 $H_{out} - 1$。

#### 3.2.2 零填充定义

对输入张量 $x \in \mathbb{R}^{N \times C \times H \times W}$，在空间维度 H、W 的上下左右各填充 $\mathit{pad}$ 行/列零值，得到填充后张量 $x^{\mathit{pad}} \in \mathbb{R}^{N \times C \times (H+2\cdot\mathit{pad}) \times (W+2\cdot\mathit{pad})}$：

$$x^{\mathit{pad}}_{n,c,\,h,\,w} = \begin{cases} x_{n,c,\,h-\mathit{pad},\,w-\mathit{pad}} & \text{若 } \mathit{pad} \le h < H+\mathit{pad} \text{ 且 } \mathit{pad} \le w < W+\mathit{pad} \\ 0 & \text{否则（填充区域）} \end{cases}$$

#### 3.2.3 深度卷积计算

深度卷积的核心特征是**每个通道独立卷积，不跨通道加权**。对于输出张量 $y \in \mathbb{R}^{N \times C \times H_{out} \times W_{out}}$ 中的任意一个元素 $y_{n,c,i,j}$，其计算过程为：

$$y_{n,c,i,j} = b_c + \sum_{kh=0}^{k_H-1} \sum_{kw=0}^{k_W-1} x^{\mathit{pad}}_{n,c,\,h_s,\,w_s} \cdot w_{c,0,kh,kw}$$

其中采样位置由滑动窗口位置 $(i, j)$、卷积核偏移 $(kh, kw)$、步长与膨胀率共同确定：

$$h_s = i \cdot \mathit{stride} + kh \cdot \mathit{dilation}$$

$$w_s = j \cdot \mathit{stride} + kw \cdot \mathit{dilation}$$

**公式逐项说明**：

| 符号 | 含义 |
|------|------|
| $y_{n,c,i,j}$ | 输出张量在 batch $n$、通道 $c$、空间位置 $(i, j)$ 处的值 |
| $b_c$ | 通道 $c$ 的偏置（若未提供则视为 0） |
| $kh, kw$ | 卷积核在 H、W 方向的索引，分别取 $0, 1, \dots, k_H-1$ 与 $0, 1, \dots, k_W-1$ |
| $x^{\mathit{pad}}_{n,c,h_s,w_s}$ | 填充后输入在通道 $c$ 上、空间位置 $(h_s, w_s)$ 处的值 |
| $w_{c,0,kh,kw}$ | 卷积核在通道 $c$ 上、空间偏移 $(kh, kw)$ 处的权重（第 1 维固定为 1，即 per-channel 单卷积核） |
| $h_s, w_s$ | 实际采样位置，由输出位置、卷积核偏移、步长、膨胀率联合确定 |

**卷积过程解读**：

1. **定位输出窗口**：对输出位置 $(i, j)$，窗口左上角在填充输入中的坐标为 $(i \cdot \mathit{stride},\; j \cdot \mathit{stride})$，即每移动一个输出像素，采样起点在输入上前进 $\mathit{stride}$ 步。

2. **遍历卷积核**：对卷积核的每个位置 $(kh, kw)$，计算其在输入中的实际采样偏移 $(kh \cdot \mathit{dilation},\; kw \cdot \mathit{dilation})$，膨胀率 $>1$ 时等效于在卷积核元素间插入间隔，扩大感受野而不增加参数量。

3. **逐元素乘加**：将采样到的输入值 $x^{\mathit{pad}}_{n,c,h_s,w_s}$ 与对应卷积核权重 $w_{c,0,kh,kw}$ 相乘，累加到输出 $y_{n,c,i,j}$ 中。整个双重求和等价于一个 $k_H \times k_W$ 窗口内的加权求和。

4. **叠加偏置**：所有卷积核位置的乘加完成后，将 per-channel 偏置 $b_c$ 加到输出上。

5. **通道独立**：注意权重第 0 维为 $C$、第 1 维固定为 1，意味着通道 $c$ 的输出仅依赖通道 $c$ 的输入与通道 $c$ 的卷积核，不存在跨通道信息混合——这是深度卷积与标准卷积的根本区别。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | (N, C, H, W) | float16, float32 | ND | 输入张量 |
| INPUT(必选) | w | tensor | (C, 1, kH, kW) | float16, float32 | ND | 深度卷积权重，与 x 同 dtype |
| INPUT(可选) | b | tensor | (C,) | float32 | ND | per-channel 偏置，默认全零 |
| OUTPUT(输出) | y | tensor | (N, C, H_out, W_out) | float16, float32 | ND | 卷积输出，与 x 同 dtype |

### 3.4 关键输入约束

- **数据类型**: x、w 支持 float16 / float32 且必须一致；b 固定为 float32。
- **维度场景**: 4 维 (N, C, H, W)。
- **维度取值范围(均为正整数)**:
  - N（batch）：N ∈ [1, 4096]
  - C（通道）：C ∈ [1, 4096]
  - H / W（空间维度）：H, W ∈ [1, 1024]
  - kH / kW（卷积核尺寸）：kH, kW ∈ [1, 15]，kH ≤ H，kW ≤ W
- **其他约束**: 不支持空 Tensor；w 的第 0 维必须与 C 一致，第 1 维固定为 1；b 的维度必须与 C 一致（当提供时）。

### 3.5 核心属性说明

| 属性名 | 类型 | 必选/可选 | 默认值 | 说明 |
|--------|------|-----------|--------|------|
| stride | int | 可选 | 1 | 滑动步长（H、W 方向相同） |
| pad | int | 可选 | 0 | 空间维度零填充量（四周相同） |
| dilation | int | 可选 | 1 | 膨胀率（H、W 方向相同） |

### 3.6 输出严格要求

- **形状约束**: y 的 shape 为 (N, C, H_out, W_out)，其中 H_out、W_out 由 3.2 公式确定。
- **类型约束**: y 与 x 同 dtype（float16/float32）。
- **数值等价**: 输出结果应与参考实现（上述 numpy 实现）的计算结果在精度允许范围内一致。

### 3.7 特殊值处理规则

- **零填充区域**: 填充区域视为 0，参与卷积累乘时不贡献到输出值（0 乘任何数为 0）。
- **偏置为空**: 当未提供偏置 b 时，输出仅含卷积加权求和结果，不加偏置。
- **dilation > 1**: 膨胀卷积核在空间上稀疏采样，等效卷积核尺寸为 `dilation * (k - 1) + 1`。

## 四、规则要求

1. **数值一致性规则**: 给定相同输入，算子多次执行结果应完全一致（确定性算子，不含随机性）。

2. **精度要求**: 计算结果与更高精度参考实现相比需满足第五节的精度判断规则。

3. **性能要求**: 在保证数值正确性的前提下，充分利用 NPU 硬件特性：
   - 合理设计滑动窗口的数据搬运与 Tiling 分块策略
   - 支持向量化计算，合理利用 Cube（im2col 方案）或 Vector（直接卷积方案）计算单元
   - 优化内存访问模式，减少 GM 与 UB 之间的数据搬运
   - 合理利用多核并行，沿 batch 和/或通道维分配任务
   - 鼓励使用 Double Buffer / Ping-Pong 机制实现计算与搬运流水重叠
   - 代码质量：正确使用 AscendC API；Tiling 策略合理（多核负载均衡、UB 内存不溢出）；代码可读性与注释完整性

## 五、精度判断规则

- **精度要求**：计算结果需满足以下精度误差要求：
    - float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
    - float16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
    - int32：要求计算结果完全准确，无误差

## 六、示例说明

**示例1**: 小规模示例

- 输入 x: shape=(1, 4, 8, 8), dtype=float32
- 权重 w: shape=(4, 1, 3, 3), dtype=float32
- 偏置 b: shape=(4,), dtype=float32
- 属性: stride=1, pad=1, dilation=1
- 计算过程: 对每个通道独立做 3×3 卷积，stride=1、pad=1 时输出尺寸不变 (8×8)
- 输出: y shape=(1, 4, 8, 8)

**示例2**: 中等规模示例

- 输入 x: shape=(2, 32, 32, 32), dtype=float32
- 权重 w: shape=(32, 1, 3, 3), dtype=float32
- 偏置 b: shape=(32,), dtype=float32
- 属性: stride=2, pad=1, dilation=1
- 计算过程: stride=2 下采样，输出尺寸为 16×16，32 个通道独立卷积
- 输出: y shape=(2, 32, 16, 16)

**示例3**: 真实场景示例

- 输入 x: shape=(8, 512, 64, 64), dtype=float32（MobileNet 大 batch 特征提取层）
- 权重 w: shape=(512, 1, 3, 3), dtype=float32
- 偏置 b: shape=(512,), dtype=float32
- 属性: stride=1, pad=1, dilation=1
- 计算过程: 512 个通道各做 3×3 卷积，输出 64×64，总计算量约 8×512×64×64×9 ≈ 151M MACs
- 输出: y shape=(8, 512, 64, 64)
