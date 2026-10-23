# index算子

## 一、赛题背景

在深度学习模型中，Index算子用于根据索引 `indices` 从输入张量 `x` 的指定维度中取出对应坐标的数据，是稀疏注意力、MoE路由、RAG检索等场景中紧随TopK之后的关键数据搬运算子，对端到端访存性能影响显著。

本题要求基于Ascend官方Index算子接口规范，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的index算子，需同时支持连续轴与非连续轴两类索引场景。

## 二、算子功能描述

- 算子功能：Index算子等同于竞品 PyTorch 中重载的中括号，对 tensor 进行索引操作，例如 `y = x[1, :, 3]`。按 indices 指定的下标对输入 x 进行多维高级索引（advanced indexing），取出对应坐标的数据得到输出 y。

通过 `indexed_sizes`（长度为 N 的 0/1 掩码）标记 `x` 的哪些维度参与索引，被索引的维度在输出中首先被 `indices` 的形状替换，未被索引的维度按原顺序保留。根据 `indexed_sizes` 中 1 是否连续，分为两种场景：

- **连续轴场景**：`indexed_sizes` 中的 1 连续。例如 `x` 形状为 `(a, b, c, d)`，`indexed_sizes = [0, 1, 1, 0]`，`indices` 形状为 `(e, f)`，则 `y` 形状为 `(a, e, f, d)`。
- **非连续轴场景**：`indexed_sizes` 中的 1 不连续。例如 `x` 形状为 `(a, b, c, d)`，`indexed_sizes = [1, 0, 1, 0]`，`indices` 形状为 `(e, f)`，则 `y` 形状为 `(e, f, b, d)`。


## 三、核心定义与约束

### 3.1 参考实现
```python
import numpy as np
import torch

def impl(x, mask, indices):
    x_torch = torch.from_numpy(x)
    indices_torch = torch.from_numpy(np.array(indices))
    
    cmd = "x_torch["
    idx = 0
    for i in range(mask.size):
        if mask[i]:
            cmd += "indices_torch"
            idx += 1
        else:
            cmd += ":"
        if i < mask.size - 1:
            cmd += ","
    cmd += "]"
    
    res = eval(cmd)
    res = res.numpy()
    return res
```

### 3.2 输入输出与属性总览

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|----------------|------|----------|----------|
| x | 输入 | N 维输入张量 | INT64、INT32、FLOAT32、FLOAT16、INT8、UINT8、BOOL | ND |
| indexed_sizes | 输入 | 长度为 N 的 0/1 掩码，1 表示对应维度被索引 | INT64 | ND |
| indices | 输入 | 动态输入，任意维张量，每个元素为被索引维度乘积空间内的扁平索引 | INT32、INT64 | ND |
| y | 输出 | 根据索引取出后的数据，dtype 与 x 相同 | 与 x 相同 | ND |

### 3.3 关键输入约束

- **参数解释**
    - x：输入张量，N 维，连续（ND）布局；算子内部依据其连续布局自动计算各维度步长
    - indexed_sizes：一维 INT64 张量，长度等于 x 的维度数，逐维标记 x 的每一维是否参与索引：取值为 1 表示该维由索引选择（消耗一个 indices 输入），取值大于 1 表示该维整体保留。
    - indices：任意维张量，元素需满足 `0 ≤ indices[i] < P`，其中 `P = prod(x.shape[i] for indexed_dims)`；indexed_sizes为1的数量跟indices的数量要一致
    - y：与 x 同 dtype；形状由 `indices.shape` 替换被索引维度块、非索引维度按原顺序保留得到

### 3.4 输出严格要求

- **精度要求**: 与参考代码的精度误差满足浮点数精度要求（float32误差不超过1e-4，float16 放宽至1e-3）；整数类型输出需与参考实现逐元素完全一致
- **边界要求**: indices 元素必须落在 `[0, P)` 区间内，越界行为不做保证
- **形状要求**: y 的形状严格遵循 3.3 节连续轴/非连续轴场景规则

## 四、规则要求

1. 设计算子逻辑，保证算子精度正确
2. 充分发挥系统带宽能力，算子性能更优
