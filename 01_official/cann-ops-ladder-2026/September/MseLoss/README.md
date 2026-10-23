# MseLoss算子

## 一、赛题背景

均方误差（Mean Squared Error，MSE）是回归任务和图像重建任务中最基础的损失函数，衡量预测值与目标值之间的平方误差。MSE Loss 广泛应用于目标检测的边界框回归、图像超分/去噪、自编码器等场景。

本题要求基于 PyTorch 中 `torch.nn.MSELoss` 的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 MseLoss 算子。

## 二、算子功能描述

计算输入 self 与目标 target 每个元素之间的均方误差：

1. **逐元素计算平方差**：`(self - target)²`。
2. **按 reduction 归约**：
   - `none`（0）：不归约，输出与输入同形状。
   - `mean`（1）：求均值，输出标量（shape [1]）。
   - `sum`（2）：求和，输出标量（shape [1]）。

算子的核心难点在于：reduction 三种模式的分支处理，以及 mean 模式下的归约与除法实现。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch 原生算子：`torch.nn.MSELoss`（`reduction` 为 none/mean/sum）

等价 numpy 实现：

```python
import numpy as np

def mse_loss(self, target, reduction=1):
    # self/target 同 shape；reduction: 0=none, 1=mean, 2=sum
    x32 = self.astype(np.float32)
    y32 = target.astype(np.float32)
    l = (x32 - y32) ** 2
    if reduction == 0:
        out = l
    elif reduction == 1:
        out = np.array([l.mean()], dtype=np.float32)
    else:
        out = np.array([l.sum()], dtype=np.float32)
    return out.astype(self.dtype)
```

### 3.2 数学公式

$$\ell(x, y) = \{(x_n - y_n)^2\}_{n=1}^{N}$$

$$\text{out} = \begin{cases} \ell, & \text{reduction} = \text{none} \\ \text{mean}(\ell), & \text{reduction} = \text{mean} \\ \text{sum}(\ell), & \text{reduction} = \text{sum} \end{cases}$$

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | self | tensor | 任意形状 | float16, float32 | ND | 预测值 |
| INPUT(必选) | target | tensor | 与 self 相同 | float16, float32 | ND | 目标值 |
| OUTPUT(输出) | out | tensor | 同 self（none）/ [1]（mean/sum） | float16, float32 | ND | 损失，与 self 同 dtype |

### 3.4 关键输入约束

- **数据类型**: self、target 支持 float16 / float32，二者 dtype 必须一致。
- **维度场景**: 任意维（1D / 2D / 3D 等），self 与 target 必须同 shape。
- **维度取值范围(均为正整数)**: 各维度 ∈ [1, 16384]，元素总数需在硬件可处理范围内。
- **其他约束**: 不支持空 Tensor；self 与 target shape 必须完全一致。

### 3.5 核心属性说明

- **reduction**（int，默认 1）：缩减方式，0=none（不归约）、1=mean（求均值）、2=sum（求和）。

### 3.6 输出严格要求

- **形状约束**: reduction=0 时 out 与 self 同 shape；reduction=1/2 时 out 为 shape [1] 的标量。
- **类型约束**: out 与 self 同 dtype（float16/float32）。
- **数值等价**: 输出结果应与参考实现（上述 numpy 实现）的计算结果在精度允许范围内一致。

### 3.7 特殊值处理规则

- **无特殊值**: 平方差 `(x-y)²` 恒为非负，无除零或溢出风险；mean 模式需注意大输入下的归约精度。

## 四、规则要求

1. **数值一致性规则**: 给定相同输入，算子多次执行结果应完全一致（确定性算子，不含随机性）。

2. **精度要求**: 计算结果与更高精度参考实现相比需满足第五节的精度判断规则。

3. **性能要求**: 在保证数值正确性的前提下，充分利用 NPU 硬件特性：
   - 支持向量化计算，逐元素平方差充分利用 Vector 计算单元
   - mean/sum 模式的归约采用分块归约，减少多核间的同步开销
   - 优化内存访问模式，减少 GM 与 UB 之间的数据搬运
   - 合理利用多核并行，沿元素维分配任务
   - 鼓励使用 Double Buffer / Ping-Pong 机制实现计算与搬运流水重叠
   - 代码质量：正确使用 AscendC API；Tiling 策略合理（多核负载均衡、UB 内存不溢出）；代码可读性与注释完整性

## 五、精度判断规则

- **精度要求**：计算结果需满足以下精度误差要求：
    - float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
    - float16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
    - int32：要求计算结果完全准确，无误差

## 六、示例说明

**示例1**: 小规模示例

- 输入 self: shape=(8,), dtype=float32；target: shape=(8,), dtype=float32；reduction=mean
- 计算过程: 逐元素平方差后求均值，输出标量
- 输出: out shape=(1,), dtype=float32

**示例2**: 不归约示例

- 输入 self: shape=(7, 13), dtype=float16；target: shape=(7, 13), dtype=float16；reduction=none
- 计算过程: 逐元素平方差，不归约
- 输出: out shape=(7, 13), dtype=float16

**示例3**: 真实场景示例

- 输入 self: shape=(16384, 4096), dtype=float32；target 同 shape；reduction=none（图像重建大张量场景）
- 计算过程: 逐元素平方差，不归约
- 输出: out shape=(16384, 4096), dtype=float32
