# BatchNorm3D算子

## 一、赛题背景

Batch Normalization（批量归一化）是深度卷积网络中最基础的归一化算子之一，广泛应用于 3D 卷积神经网络（如视频理解 C3D/I3D、医学影像 3D-UNet、点云处理等）。与 2D 卷积的 BatchNorm2d 相比，BatchNorm3d 沿 (N, D, H, W) 四个维度计算均值和方差，通道维度 C 上的每个通道有独立的 scale(γ) 和 offset(β) 参数。

本题要求基于 PyTorch 中 `torch.nn.BatchNorm3d` 的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 BatchNorm3D 算子。

## 二、算子功能描述

实现的 BatchNorm3D 算子需完成以下核心计算：

1. **多维归约求统计量**：沿归约维 (N, D, H, W) 对每个通道分别计算均值 μ_c 与方差 σ_c²。
2. **归一化**：对每个元素计算 `(x - μ) / sqrt(σ² + ε)`。
3. **仿射变换**：`y = 归一化结果 * γ + β`（γ、β 为 per-channel 参数）。

算子的核心难点在于：归约维度 (N, D, H, W) 在内存中非连续，需要高效的跨维归约策略，以及 per-channel 广播仿射的向量化实现。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch 原生算子：`torch.nn.BatchNorm3d`（训练模式）

等价 numpy 实现：

```python
import numpy as np

def batch_norm_3d(x, scale, offset, epsilon=1e-5):
    # x: (N, C, D, H, W)
    N, C = x.shape[0], x.shape[1]
    x32 = x.astype(np.float32)
    scale32 = scale.astype(np.float32).reshape(1, C, 1, 1, 1)
    offset32 = offset.astype(np.float32).reshape(1, C, 1, 1, 1)

    mean = x32.mean(axis=(0, 2, 3, 4), keepdims=True)   # (1, C, 1, 1, 1)
    var = x32.var(axis=(0, 2, 3, 4), keepdims=True)     # (1, C, 1, 1, 1)

    y = (x32 - mean) / np.sqrt(var + epsilon) * scale32 + offset32
    return y.astype(x.dtype), mean.reshape(C).astype(np.float32), var.reshape(C).astype(np.float32)
```

### 3.2 数学公式

$$\mu_c = \frac{1}{|S|}\sum_{i \in S} x_{c,i}, \quad \sigma_c^2 = \frac{1}{|S|}\sum_{i \in S}(x_{c,i} - \mu_c)^2$$

$$y_{c,i} = \frac{x_{c,i} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \cdot \gamma_c + \beta_c$$

其中 S 为 (N, D, H, W) 的笛卡尔积，|S| = N × D × H × W，方差采用有偏估计（除以 |S|）。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | (N, C, D, H, W) | float16, float32 | ND | 输入张量 |
| INPUT(必选) | scale | tensor | (C,) | float32 | ND | 通道缩放 γ |
| INPUT(必选) | offset | tensor | (C,) | float32 | ND | 通道偏置 β |
| OUTPUT(输出) | y | tensor | (N, C, D, H, W) | float16, float32 | ND | 归一化结果，与 x 同 dtype |
| OUTPUT(输出) | save_mean | tensor | (C,) | float32 | ND | 均值 |
| OUTPUT(输出) | save_variance | tensor | (C,) | float32 | ND | 方差 |

### 3.4 关键输入约束

- **数据类型**: x 支持 float16 / float32；scale、offset 固定为 float32。
- **维度场景**: 5 维 (N, C, D, H, W)。
- **维度取值范围(均为正整数)**:
  - N（batch）：N ∈ [1, 4096]
  - C（通道）：C ∈ [1, 4096]
  - D / H / W（空间维度）：D, H, W ∈ [1, 1024]
- **其他约束**: 不支持空 Tensor；scale、offset 的维度必须与 C 一致。

### 3.5 核心属性说明

- **epsilon**（float，默认 1e-5）：数值稳定常数，加入方差后开方，避免方差为 0 时除零。

### 3.6 输出严格要求

- **形状约束**: y 与 x 同 shape；save_mean、save_variance 为 (C,)。
- **类型约束**: y 与 x 同 dtype（float16/float32）；save_mean、save_variance 固定为 float32。
- **数值等价**: 输出结果应与参考实现（上述 numpy 实现）的计算结果在精度允许范围内一致。

### 3.7 特殊值处理规则

- **方差为 0 处理**: 当某通道所有元素相等（方差为 0）时，`sqrt(σ² + ε)` 退化为 `sqrt(ε)`，归一化结果恒为 0，经仿射后输出该通道 offset，避免除零产生 NaN/Inf。

## 四、规则要求

1. **数值一致性规则**: 给定相同输入，算子多次执行结果应完全一致（确定性算子，不含随机性）。

2. **精度要求**: 计算结果与更高精度参考实现相比需满足第五节的精度判断规则。

3. **性能要求**: 在保证数值正确性的前提下，充分利用 NPU 硬件特性：
   - 合理设计多维归约（(N, D, H, W) 非连续内存）的分块与并行策略
   - 支持向量化计算，合理利用 Cube 与 Vector 计算单元
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

- 输入 x: shape=(1, 4, 2, 2, 2), dtype=float32
- 输入 scale: shape=(4,), dtype=float32
- 输入 offset: shape=(4,), dtype=float32
- 计算过程: 对每个通道沿 (N=1, D=2, H=2, W=2) 共 8 个元素计算均值与有偏方差，再归一化并做仿射变换
- 输出: y shape=(1, 4, 2, 2, 2)，save_mean/save_variance shape=(4,)

**示例2**: 中等规模示例

- 输入 x: shape=(2, 32, 8, 8, 8), dtype=float32
- 每个通道沿 (N=2, D=8, H=8, W=8) 共 1024 个元素归约，32 个通道独立统计
- 输出: y shape=(2, 32, 8, 8, 8)，save_mean/save_variance shape=(32,)

**示例3**: 真实场景示例

- 输入 x: shape=(8, 512, 8, 32, 32), dtype=float32（视频理解常见大 batch 多通道场景）
- 每个通道沿 (N=8, D=8, H=32, W=32) 共 65536 个元素归约，512 个通道独立统计
- 输出: y shape=(8, 512, 8, 32, 32)，save_mean/save_variance shape=(512,)
