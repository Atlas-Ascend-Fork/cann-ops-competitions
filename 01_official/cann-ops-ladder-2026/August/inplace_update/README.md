# InplaceUpdate 算子竞赛题目

## 算子名称
**InplaceUpdate**（原位更新）

## 算子功能
InplaceUpdate 是一个数组更新算子，用于对张量的指定行进行更新操作。该算子支持负索引，允许从数组末尾索引。

## 输入输出描述

| 名称 | 类别 | 数据类型 | 形状 | 说明 |
|------|------|----------|------|------|
| `x` | 必需输入 | `T` | `[rows, ...]` | 原张量，待更新的目标 |
| `i` | 必需输入 | `int32` | `[n]` | 索引向量，指定要更新的行位置 |
| `v` | 必需输入 | 与 `x` 一致 | `[n, ...]` | 更新值张量，除第0维外形状与x相同 |
| `y` | 必需输出 | 与 `x` 一致 | 与 `x` 相同 | 输出张量，形状与x相同 |

## 属性参数
无额外属性参数。

## 形状约束
1. `x.dims() == v.dims()`（维度数相同）
2. `x.dim_size(j) == v.dim_size(j)` 对于 `j≥1`（除第0维外各维度匹配）
3. `i.dim_size(0) == v.dim_size(0)`（索引数量与更新值行数相等）

## 功能示例

### 示例1：单行更新（标量索引）
```python
import tensorflow as tf
from tensorflow.python.ops import inplace_ops

x = tf.ones([7, 3], dtype=tf.float32)
v = tf.ones([3], dtype=tf.float32) * 5  # v比x低一维
y = inplace_ops.inplace_update(x, 5, v)  # 更新第5行
# 结果：y[5, :] = 5
```

### 示例2：批量行更新（向量索引）
```python
import tensorflow as tf
from tensorflow.python.ops import inplace_ops

x = tf.ones([7, 3], dtype=tf.float32)
v = tf.ones([2, 3], dtype=tf.float32) * 3
y = inplace_ops.inplace_update(x, [3, 4], v)  # 更新第3、4行
# 结果：y[3, :] = 3, y[4, :] = 3
```

### 示例3：负索引更新
```python
import tensorflow as tf
from tensorflow.python.ops import inplace_ops

x = tf.ones([7, 3], dtype=tf.float32)
v = tf.ones([1, 3], dtype=tf.float32) * 2
y = inplace_ops.inplace_update(x, [-1], v)  # 更新最后一行
# 结果：y[6, :] = 2
```

## 任务要求
- 针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确。
- 充分发挥系统带宽能力和并行计算能力，算子性能更优。
