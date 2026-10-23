# ConfusionMatrix 算子竞赛题目

## 算子名称
**ConfusionMatrix**（混淆矩阵）

## 算子功能
ConfusionMatrix 用于计算分类任务的混淆矩阵，矩阵的行表示真实标签，列表示预测标签，位置 (i, j) 的值表示真实标签为 i 被预测为 j 的样本数量。

## 输入输出描述

| 名称 | 类别 | 数据类型 | 形状 | 说明 |
|------|------|----------|------|------|
| `labels` | 必需输入 | `int32` / `int64` | `[N]` | 真实标签序列，每个元素为类别索引 |
| `predictions` | 必需输入 | `int32` / `int64` | `[N]` | 预测标签序列，与 labels 形状相同 |
| `weights` | 可选输入 | `int32` / `int64` / `float32` | `[N]` | 每个样本的权重，默认为 1 |
| `y` | 必需输出 | `int32` （默认int32） | `[num_classes, num_classes]` | 混淆矩阵 |

## 属性参数

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_classes` | `int` | `None` | 类别数量，若为 None 则自动计算为 `max(labels, predictions) + 1` |
| `dtype` | `DataType` | `int32` | 输出矩阵的数据类型 |


## 功能示例

### 示例1：基本混淆矩阵
```python
import tensorflow as tf

labels = [1, 2, 4]
predictions = [2, 2, 4]
cm = tf.math.confusion_matrix(labels, predictions)
# 输出：
# [[0 0 0 0 0]
#  [0 0 1 0 0]   <- 真实标签1被预测为2的次数=1
#  [0 0 1 0 0]   <- 真实标签2被预测为2的次数=1
#  [0 0 0 0 0]
#  [0 0 0 0 1]]  <- 真实标签4被预测为4的次数=1
```

### 示例2：限制类别数
```python
import tensorflow as tf

labels = [0, 1, 2]
predictions = [0, 2, 1]
cm = tf.math.confusion_matrix(labels, predictions, num_classes=3)
# 输出：3x3 矩阵
# [[1 0 0]
#  [0 0 1]
#  [0 1 0]]
```

### 示例3：加权混淆矩阵
```python
import tensorflow as tf

labels = [0, 1, 2]
predictions = [0, 1, 2]
weights = [1, 2, 3]
cm = tf.math.confusion_matrix(labels, predictions, weights=weights)
# 输出：
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

## 任务要求
- 针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确。
- 充分发挥系统带宽能力和并行计算能力，算子性能更优。
