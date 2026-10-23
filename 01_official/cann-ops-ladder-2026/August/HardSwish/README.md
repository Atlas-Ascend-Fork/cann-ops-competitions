# HardSwish算子

**难度系数：1.0**

## 一、赛题背景

HardSwish是深度学习中一种高效的激活函数，是Swish函数的分段线性近似。通过用简单的分段线性函数替代Sigmoid计算，在保持与Swish相近的表达能力的同时，大幅降低了计算复杂度，广泛应用于移动端和嵌入式场景下的轻量级神经网络（如MobileNetV3）中。

本题要求基于PyTorch原生torch.nn.functional.hardswish算子的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的HardSwish算子。

相似类型算子的实现可参考开源仓：

https://gitcode.com/cann/ops-math

https://gitcode.com/cann/ops-nn

https://gitcode.com/cann/ops-cv

https://gitcode.com/cann/ops-transformer

https://gitcode.com/cann/cann-samples

## 二、算子功能描述

实现的HardSwish算子需对输入多维张量的每个元素执行HardSwish激活操作，返回与输入形状相同的结果张量。

算子为逐元素计算，输出形状与输入形状完全一致，且需兼容非32整倍数的维度非对齐场景。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch原生算子：torch.nn.functional.hardswish（算子行为、计算结果需与该算子完全对齐）

### 3.2 数学公式

$$
\\text{hardswish}(x) = x \\times \\frac{\\text{ReLU6}(x + 3)}{6}
$$

其中：

$$
\\text{ReLU6}(x) = \\min(\\max(0, x), 6)
$$

等价分段表示：

$$
\\text{hardswish}(x) = \\begin{cases} 0, \& x \\leq -3 \\ x \\times (x + 3) / 6, \& -3 < x < 3 \\ x, \& x \\geq 3 \\end{cases}
$$

### 3.3 输入输出与属性总览

|类型|参数名|类型|维度形状|支持数据类型|数据格式|备注|
|-|-|-|-|-|-|-|
|INPUT(必选)|self|tensor|(..., N4, N3, N2, N)|float32|ND|输入张量|
|OUTPUT(输出)|out|tensor|(..., N4, N3, N2, N)|float32|ND|激活后的结果张量，形状与输入一致|

### 3.4 关键输入约束

* **维度取值范围(均为正整数)**:

  * N ∈ [1, 10000]
  * N2 ∈ [1, 10000]
  * N3 ∈ [1, 1000]
  * N4 ∈ [1, 1000]
* 输入为任意多维张量，最终维度可拆解为(..., N4, N3, N2, N)，前序...为任意合法批次维度，shape支持0-8维
* **非对齐场景兼容**: N、N2、N3、N4 均可能为非32的整倍数，算子需适配内存/数据非32字节对齐的场景
* **数值取值约束**: 输入张量的数值取值范围不超出float32数据类型的原生表达范围

### 3.5 输出严格要求

* **形状约束**: 输出形状与输入形状完全一致
* **类型约束**: 输出数据类型为float32
* **数值含义**: 输出张量中每个位置的值为输入张量对应位置元素经HardSwish激活函数计算后的结果
* **精度要求**: 要求与torch.nn.functional.hardswish标杆精度误差满足浮点数精度要求（相对误差不超过1e-5或绝对误差不超过1e-6）

### 3.6 特殊值处理规则

* **NaN处理**: 如果输入包含NaN，输出对应位置为NaN（NaN参与任何算术运算结果均为NaN）
* **Inf处理**:

  * 输入为+Inf时，ReLU6(Inf + 3) = ReLU6(Inf) = 6，hardswish(Inf) = Inf × 6 / 6 = +Inf
  * 输入为-Inf时，ReLU6(-Inf + 3) = ReLU6(-Inf) = 0，hardswish(-Inf) = -Inf × 0 / 6 = NaN
* **零值处理**:

  * 输入为0时，hardswish(0) = 0 × ReLU6(3) / 6 = 0 × 3 / 6 = 0

## 四、规则要求

1. **逐元素计算规则**: 对输入张量的每个元素独立执行HardSwish激活函数计算，元素之间无依赖关系
2. **形状不变规则**: 输出张量形状与输入张量形状完全一致，不改变维度信息
3. **分段计算规则**: 根据输入值所在的区间采用不同的计算方式，避免不必要的乘法和除法运算：

   * x ≤ -3：直接输出0
   * -3 < x < 3：计算 x × (x + 3) / 6
   * x ≥ 3：直接输出x
4. **ReLU6计算规则**: 中间结果ReLU6(x + 3)的值域为\[0, 6]，需正确处理上界截断和下界截断

## 五、示例说明

**示例1**: 一维张量基础计算

* 输入self: tensor([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]), dtype=float32, shape=[7] (N=7)
* 输出out: tensor([0.0, 0.0, -0.375, 0.0, 1.125, 3.0, 4.0]), dtype=float32, shape=[7]
* 结果解释:

  * hardswish(-4.0) = 0（x ≤ -3，输出0）
  * hardswish(-3.0) = 0（x ≤ -3，输出0）
  * hardswish(-1.5) = -1.5 × (-1.5 + 3) / 6 = -1.5 × 1.5 / 6 = -0.375
  * hardswish(0.0) = 0 × 3 / 6 = 0
  * hardswish(1.5) = 1.5 × (1.5 + 3) / 6 = 1.5 × 4.5 / 6 = 1.125
  * hardswish(3.0) = 3.0（x ≥ 3，输出x）
  * hardswish(4.0) = 4.0（x ≥ 3，输出x）

**示例2**: 二维张量计算

* 输入self: tensor([[0.0, -2.0], [2.0, 5.0]]), dtype=float32, shape=[2, 2] (N2=2, N=2)
* 输出out: tensor([[0.0, -0.3333], [1.6667, 5.0]]), dtype=float32, shape=[2, 2]
* 结果解释:

  * hardswish(-2.0) = -2.0 × 1.0 / 6 ≈ -0.3333
  * hardswish(2.0) = 2.0 × 5.0 / 6 ≈ 1.6667
  * hardswish(5.0) = 5.0（x ≥ 3，直接输出）

**示例3**: 特殊值处理

* 输入self: tensor([inf, -inf, nan]), dtype=float32, shape=[3] (N=3)
* 输出out: tensor([inf, nan, nan]), dtype=float32, shape=[3]
* 结果解释: +Inf输出+Inf，-Inf输出-Inf×0=NaN，NaN输出NaN

**示例4**: 临界值验证

* 输入self: tensor([-3.0, 3.0]), dtype=float32, shape=[2] (N=2)
* 输出out: tensor([0.0, 3.0]), dtype=float32, shape=[2]
* 结果解释:

  * hardswish(-3.0) = -3.0 × ReLU6(0) / 6 = -3.0 × 0 / 6 = 0
  * hardswish(3.0) = 3.0 × ReLU6(6) / 6 = 3.0 × 6 / 6 = 3.0

## 六、测试用例覆盖范围

* **数据类型覆盖**: float32
* **维度场景覆盖**:

  * 0维（标量）场景
  * 1维、2维、3维、4维(N4,N3,N2,N)、含批次维度的多维场景(..., N4,N3,N2,N)
  * N\~N4 为非32整倍数的非对齐场景
* **数值区间覆盖**:

  * x ≤ -3：零值区域
  * -3 < x < 3：二次函数区域（重点覆盖）
  * x ≥ 3：线性区域
* **临界值场景覆盖**:

  * x = -3（左区间边界）
  * x = 3（右区间边界）
  * x = 0（零点）
* **特殊值场景覆盖**:

  * NaN值
  * +Inf、-Inf
  * 零值
