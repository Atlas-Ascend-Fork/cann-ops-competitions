# GroupNorm算子

## 一、赛题背景

Group Normalization（分组归一化）是计算机视觉和生成模型中广泛使用的归一化算子，在目标检测（Mask R-CNN、YOLO系列）、图像生成（Stable Diffusion、DiT）以及视频理解模型中替代Batch Normalization的默认选择。与BatchNorm不同，GroupNorm沿通道维度分组计算统计量，消除了对batch size的依赖，在小batch和变长输入场景下表现稳定。

GroupNorm的计算流程为：将输入张量的通道维度划分为G个组，在每组内沿(H, W)或(C/G, H, W)维度计算均值和方差，然后进行归一化和仿射变换。其核心挑战在于：分组逻辑导致不规则的内存访问模式，且均值/方差的计算需要在非连续区域上执行归约操作。传统的逐元素标量循环实现在GPU/NPU上性能极差，需要通过向量化归约、分组并行化和内存访问优化来充分发挥硬件性能。

本题要求基于PyTorch中`torch.nn.GroupNorm`的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能的GroupNorm算子。

相似类型算子的实现可参考开源仓：

https://gitcode.com/cann/ops-nn

## 二、算子功能描述

实现的GroupNorm算子需完成以下操作：

1. 将输入张量的通道维度C划分为G个组（每组包含C/G个通道）
2. 在每组内，沿指定的归约维度计算均值 μ 和方差 σ²
3. 使用 μ 和 σ² 对组内数据进行归一化：`(x - μ) / sqrt(σ² + ε)`
4. 应用可学习的缩放系数 gamma 和偏置 beta（均为per-channel参数，shape=[C]）

算子需正确处理分组维度的归约、非连续内存区域的访问，以及gamma/beta的per-channel广播。

> **前置说明**: 输入 x 为原始张量，无预处理要求，算子内部完成分组→统计→归一化→仿射变换全流程，接口与 `torch.nn.functional.group_norm` 一致。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch原生算子：`torch.nn.functional.group_norm(x, num_groups, gamma, beta, eps)`

等价实现：

```python
def group_norm(x, gamma, beta, G, eps):
    # x: (N, C, *) — 支持任意空间维度（2D/3D）
    N, C = x.shape[:2]
    spatial_dims = x.shape[2:]             # (H, W) 或 (D, H, W)
    x = x.view(N, G, C // G, *spatial_dims)  # 分组
    reduce_dims = tuple(range(2, 2 + len(spatial_dims) + 1))  # (C/G, H, W) 或 (C/G, D, H, W)
    mean = x.mean(dim=reduce_dims, keepdim=True)
    var  = x.var(dim=reduce_dims, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.view(N, C, *spatial_dims)         # 恢复形状
    gamma = gamma.view(1, C, *([1] * len(spatial_dims)))
    beta  = beta.view(1, C, *([1] * len(spatial_dims)))
    return x * gamma + beta
```

> **前置说明**: GroupNorm 无预处理要求，输入 x 为原始张量，接口与 `torch.nn.functional.group_norm` 一致。

### 3.2 数学公式

将输入 x 按通道维度分为 G 组，每组 S_i 包含 C/G 个通道的所有空间位置：

$$\mu_i = \frac{1}{|S_i|}\sum_{j \in S_i} x_j$$

$$\sigma_i^2 = \frac{1}{|S_i|}\sum_{j \in S_i} (x_j - \mu_i)^2$$

$$y_j = \frac{x_j - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma_j + \beta_j$$

其中 |S_i| = (C/G) × H × W 为第 i 组的元素总数，γ_j 和 β_j 为第 j 通道对应的缩放和偏置参数。

### 3.3 输入输出与属性总览

| 类型         | 参数名     | 类型   | 维度形状                        | 支持数据类型      | 数据格式    | 备注                   |
| ------------ | ---------- | ------ | ------------------------------- | ----------------- | ----------- | ---------------------- |
| INPUT(必选)  | x          | tensor | (N, C, H, W) 或 (N, C, D, H, W) | float16, bfloat16 | ND, NC1HWC0 | 输入张量               |
| INPUT(必选)  | gamma      | tensor | (C,)                            | float32           | ND          | 逐通道缩放系数         |
| INPUT(必选)  | beta       | tensor | (C,)                            | float32           | ND          | 逐通道偏置             |
| ATTR(属性)   | num_groups | int    | -                               | -                 | -           | 分组数，必须整除C      |
| ATTR(属性)   | epsilon    | float  | -                               | -                 | -           | 数值稳定常数，默认1e-5 |
| OUTPUT(输出) | y          | tensor | 与x一致                         | 与x一致           | 与x一致     | 归一化后的输出         |

### 3.4 关键输入约束

- **维度取值范围(均为正整数)**:
  - N（batch）: [1, 64]
  - C（通道数）: [1, 5120]
  - H、W（空间维度）: [1, 4096]
  - 含3D变体时 D（深度）: [1, 256]
  - G（分组数）: [1, 32]，且必须整除 C

- **典型shape**:
  - ResNet/FPN场景: (N, 256, H, W), G=32
  - ViT/DiT场景: (N, 512, H, W), G=32
  - 小通道场景: (N, 4, H, W), G=1（退化为LayerNorm）

- **分组约束**: C % G == 0（组大小 CpG = C/G 为正整数）

- **非对齐场景**: CpG、H、W 的任意组合可能导致非32字节对齐的内存访问，算子需正确处理

### 3.5 核心属性说明

- **num_groups**（int）: 将通道维划分为G组。当 G=1 时退化为 LayerNorm（沿所有通道+空间归一化），当 G=C 时退化为 InstanceNorm（逐通道归一化）

- **epsilon**（float，默认1e-5）: 方差归一化中的平滑项

### 3.6 输出严格要求

- **形状约束**: y.shape == x.shape，所有维度尺寸不变
- **类型约束**: y 的数据类型与 x 一致
- **数值等价**: 输出与 PyTorch `F.group_norm(x, G, gamma, beta, eps)` 在合理浮点误差范围内一致
- **中间精度要求**: 均值、方差计算及归一化除法必须使用 FP32 精度

### 3.7 特殊值处理规则

- **NaN/Inf处理**: 输入含NaN/Inf时，输出对应位置为NaN/Inf，算子不应崩溃
- **全零输入**: 均值=0，方差=0，归一化后输出 = beta
- **单元素组**: 当 |S_i| = 1 时，方差 = 0（但加eps后安全），输出 = beta + 修正项

## 四、规则要求

1. **分组归约规则**: 均值/方差的计算仅在每组内独立进行，不同组使用各自的统计量。不可跨组归约

2. **多维归约规则**: 归约可沿所有空间维度展开，不同维度展开顺序影响内存访问模式。选手需选择合适的归约维度顺序以优化访存

3. **多核并行规则**: 当输入较大时需多核并行。分组边界可能与核分配边界不对齐，需处理跨核归约

## 五、示例说明

**示例1**: 基础场景（N=1, C=4, G=2）

- 输入 x: shape=[1, 4, 2, 2], dtype=float32
- gamma: [1.0, 1.0, 2.0, 2.0], beta: [0, 0, 1.0, 1.0]
- 属性: num_groups=2, epsilon=1e-5
- 计算过程: 通道0,1为第一组，通道2,3为第二组，各组内独立计算均值/方差后归一化
- 输出 y: shape=[1, 4, 2, 2]

**示例2**: 典型ResNet场景

- 输入 x: shape=[2, 256, 64, 64], dtype=float16
- gamma/beta: shape=[256]
- 属性: num_groups=32, epsilon=1e-5（每组8个通道）
- 输出 y: shape=[2, 256, 64, 64]

**示例3**: InstanceNorm等价验证（G=C）

- 输入 x: shape=[1, 16, 32, 32], dtype=float16
- gamma/beta: shape=[16]
- 属性: num_groups=16, epsilon=1e-5（每组1个通道，即InstanceNorm）
- 输出与 torch.nn.InstanceNorm2d(16, affine=True) 设置 gamma/beta 后的结果一致

## 六、测试用例覆盖范围

- **数据类型覆盖**: float16, bfloat16

- **维度场景覆盖**:
  - 4维输入(N,C,H,W): 小空间(7×7)到 大空间(256×256)
  - 5维输入(N,C,D,H,W): 3D GroupNorm
  - 典型 C: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048

- **分组场景覆盖**:
  - G=1（LayerNorm等价验证）
  - G=C（InstanceNorm等价验证）
  - G=C/2, C/4, C/8 等常见配置
  - CpG (=C/G) 非2的幂次（如C=60, G=12 → CpG=5）

- **数值场景**: 正常范围、极端值、全零输入、NaN/Inf

- **正确性验证基准**: 以 PyTorch `F.group_norm()` 在 FP32 下的结果为 golden
