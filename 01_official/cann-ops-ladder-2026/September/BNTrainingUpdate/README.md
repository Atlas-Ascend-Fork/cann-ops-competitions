# BNTrainingUpdate算子

## 一、赛题背景

在分布式 Batch Normalization 训练中，为了减少通信次数并提高效率，CANN 将 BatchNorm 的前向计算拆分为两个阶段：先由 `bn_training_reduce` 算子在各设备上计算本地的 sum（元素和）与 square_sum（元素平方和），经 all-reduce 得到全局统计量后，再由 **BNTrainingUpdate** 算子完成均值/方差计算与归一化输出。这种拆分使统计量的跨卡归约与归一化计算解耦，是大规模分布式训练中的关键优化。

本题要求基于上述分段实现的业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 BNTrainingUpdate 算子。

## 二、算子功能描述

对已完成统计量归约的批次数据，完成以下核心计算：

1. **统计量换算**：由全局 sum 与 square_sum 计算均值 μ 与方差 σ²。
2. **归一化**：对输入 x 计算 `(x - μ) / sqrt(σ² + ε)`。
3. **仿射变换**：`y = 归一化结果 * γ + β`。

算子的核心难点在于：平方和公式 `σ² = sum_sq/N - μ²` 的数值稳定性，以及 per-channel 广播仿射的向量化实现。

## 三、核心定义与约束

### 3.1 参考算子

等价 numpy 实现：

```python
import numpy as np

def bn_training_update(x, sum, square_sum, scale, offset, epsilon=1e-5):
    # x: (N, C, H, W); sum/square_sum/scale/offset: (C,)
    N, C, H, W = x.shape
    n_elem = N * H * W
    x32 = x.astype(np.float32)
    sum32 = sum.astype(np.float32)
    sq32 = square_sum.astype(np.float32)
    scale32 = scale.astype(np.float32).reshape(1, C, 1, 1)
    offset32 = offset.astype(np.float32).reshape(1, C, 1, 1)

    mean = sum32 / n_elem                        # (C,)
    var = sq32 / n_elem - mean * mean            # (C,)

    y = (x32 - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + epsilon) * scale32 + offset32
    return y.astype(x.dtype), mean.astype(np.float32), var.astype(np.float32)
```

### 3.2 数学公式

设每通道全局元素数为 N_c（= N × H × W），sum_c 为元素和，sum_sq_c 为元素平方和：

$$\mu_c = \frac{sum_c}{N_c}, \quad \sigma_c^2 = \frac{sum\_sq_c}{N_c} - \mu_c^2$$

$$y_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \cdot \gamma_c + \beta_c$$

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | (N, C, H, W) | float16, float32 | ND | 批次数据 |
| INPUT(必选) | sum | tensor | (C,) | float32 | ND | 元素和（前置 reduce 归约结果） |
| INPUT(必选) | square_sum | tensor | (C,) | float32 | ND | 元素平方和（前置 reduce 归约结果） |
| INPUT(必选) | scale | tensor | (C,) | float32 | ND | 通道缩放 γ |
| INPUT(必选) | offset | tensor | (C,) | float32 | ND | 通道偏置 β |
| OUTPUT(输出) | y | tensor | (N, C, H, W) | float16, float32 | ND | 归一化结果，与 x 同 dtype |
| OUTPUT(输出) | save_mean | tensor | (C,) | float32 | ND | 本批均值 |
| OUTPUT(输出) | save_variance | tensor | (C,) | float32 | ND | 本批方差 |

### 3.4 关键输入约束

- **数据类型**: x 支持 float16 / float32；sum、square_sum、scale、offset 固定为 float32。
- **维度场景**: 4 维 (N, C, H, W)。
- **维度取值范围(均为正整数)**:
  - N（batch）：N ∈ [1, 4096]
  - C（通道）：C ∈ [1, 4096]
  - H / W（空间维度）：H, W ∈ [1, 1024]
- **其他约束**: 不支持空 Tensor；sum、square_sum、scale、offset 的维度必须与 C 一致。

### 3.5 核心属性说明

- **epsilon**（float，默认 1e-5）：数值稳定常数，加入方差后开方，避免方差为 0 时除零。

### 3.6 输出严格要求

- **形状约束**: y 与 x 同 shape；save_mean、save_variance 为 (C,)。
- **类型约束**: y 与 x 同 dtype（float16/float32）；save_mean、save_variance 固定为 float32。
- **数值等价**: 输出结果应与参考实现（上述 numpy 实现）的计算结果在精度允许范围内一致。

### 3.7 特殊值处理规则

- **方差为 0 处理**: 当某通道所有元素相等（方差为 0）时，`sqrt(σ² + ε)` 退化为 `sqrt(ε)`，避免除零产生 NaN/Inf。
- **平方和公式数值稳定性**: 方差由 `sum_sq/N - μ²` 计算，实现时需注意大数值下两项相减的精度损失，必要时采用更稳定的方差计算方式。

## 四、规则要求

1. **数值一致性规则**: 给定相同输入，算子多次执行结果应完全一致（确定性算子，不含随机性）。

2. **精度要求**: 计算结果与更高精度参考实现相比需满足第五节的精度判断规则。

3. **性能要求**: 在保证数值正确性的前提下，充分利用 NPU 硬件特性：
   - 合理设计 per-channel 统计量换算与广播仿射的向量化策略
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

- 输入 x: shape=(1, 4, 2, 2), dtype=float32
- 输入 sum/square_sum/scale/offset: shape=(4,), dtype=float32
- 计算过程: 每个通道元素数 N_c = 1×2×2 = 4，由 sum 与 square_sum 计算均值与方差，再归一化并做仿射变换
- 输出: y shape=(1, 4, 2, 2)，save_mean/save_variance shape=(4,)

**示例2**: 中等规模示例

- 输入 x: shape=(2, 32, 16, 16), dtype=float32
- 每个通道元素数 N_c = 2×16×16 = 512，32 个通道独立换算
- 输出: y shape=(2, 32, 16, 16)，save_mean/save_variance shape=(32,)

**示例3**: 真实场景示例

- 输入 x: shape=(8, 512, 64, 64), dtype=float32（大规模分布式训练常见多通道大空间场景）
- 每个通道元素数 N_c = 8×64×64 = 32768，512 个通道独立换算
- 输出: y shape=(8, 512, 64, 64)，save_mean/save_variance shape=(512,)
