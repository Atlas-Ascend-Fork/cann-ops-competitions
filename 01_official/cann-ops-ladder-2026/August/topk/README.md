# topk算子

## 一、赛题背景

在深度学习模型中，TopK算子被广泛应用于注意力稀疏化、采样、检索排序（如Beam Search）、MoE专家路由等场景，用于从大规模候选中快速筛选出最有价值的若干元素。在大模型推理框架（如vllm）中，TopK常配合Index算子完成对关键token或专家特征的稀疏提取，是影响端到端性能的关键算子之一。

本题要求基于PyTorch标准实现的topk算子，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的topk算子。

## 二、算子功能描述


沿给定输入张量 `x` 的指定维度 `dim`，返回该维度上取值最大的 `k` 个元素构成的张量 `values`，以及这些元素在原维度上的索引 `indices`。当 `largest=True` 时取最大的 k 个；当 `largest=False` 时取最小的 k 个。当 `sorted=True` 时，结果按目标顺序（降序或升序）排列；当 `sorted=False` 时，结果顺序不做保证。


## 三、核心定义与约束

### 3.1 参考实现
```
PyTorch原生算子：torch.topk

参考文档：https://pytorch.org/docs/stable/generated/torch.topk.html

```
```python
import torch

def golden_topk(x, k, dim=-1, largest=True, sorted=True):
    x_t = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
    return torch.topk(x_t, k, dim=dim, largest=largest, sorted=sorted)

```

### 3.2 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | 任意形状| float16、float32、int64 | ND | 输入张量 |
| INPUT(必选) | k | int32 | 标量  | int32 | - | 要取出的元素个数 |
| ATTR(可选) | dim | int | 标量 | int | - | 指定沿哪个维度进行操作，默认为-1（最后一维） |
| ATTR(可选) | largest | bool | 标量 | bool | - | 是否取最大值，默认为true。若为false则取最小值 |
| ATTR(可选) | sorted | bool | 标量  | bool | - | 	是否对输出结果排序，默认为true |
| OUTPUT(输出) | values | tensor | 任意形状 | float16、float32、int64 | ND | 	输出的top-k个元素值 |
| OUTPUT(输出) | indices | tensor | 任意形状 | int32 | ND | 输出的top-k个元素在输入张量中的索引 |

### 3.3 关键输入约束
- dim必须在[-x维度数, x维度数)范围内
- k必须大于等于0且小于等于x在dim维度上的大小


### 3.4 输出严格要求

- **精度要求**: 
    - values：与参考代码的精度误差满足浮点数精度要求（float32误差不超过1e-4，float16放宽至1e-3）
    - indices：与参考实现逐元素完全一致；若存在相同数值元素，按其在原dim轴上的原始位置升序排列（与 PyTorch 稳定排序语义一致）

## 四、规则要求

1. 设计算子逻辑，保证算子精度正确
2. 充分发挥系统带宽能力，算子性能更优
