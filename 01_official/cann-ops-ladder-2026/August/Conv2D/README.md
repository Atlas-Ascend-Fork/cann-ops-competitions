# Conv类算子开发任务

## 一、算子描述

### 1.1 算子名称
**Conv2D** - 二维卷积运算

### 1.2 功能描述
实现标准的二维卷积操作，对输入特征图进行卷积核滑动窗口计算，输出卷积结果。支持多种卷积配置，包括不同步长、填充方式和分组卷积。
相关实现可参考https://gitcode.com/cann/ops-nn/tree/master/conv

### 1.3 数学定义

对于输入特征图 `input` 和卷积核 `weight`，输出 `output` 的计算公式为：

```
output[n, c_out, h_out, w_out] = bias[c_out] +
    Σ_{c_in=0}^{C_in-1} Σ_{kh=0}^{kernel_h-1} Σ_{kw=0}^{kernel_w-1}
        input[n, c_in, h_in, w_in] × weight[ c_out, c_in, kh, kw]
```

**输入特征图坐标计算**：

```
h_in = h_out × stride_h + kh × dilation_h - padding_top
w_in = w_out × stride_w + kw × dilation_w - padding_left
```

**参数说明**：
- `n`: batch 索引，范围 [0, N)
- `c_out`: 输出通道索引，范围 [0, C_out)
- `c_in`: 输入通道索引，范围 [0, C_in)
- `h_out, w_out`: 输出特征图的空间坐标
- `h_in, w_in`: 输入特征图的空间坐标（通过上述公式计算）
- `kh, kw`: 卷积核内的相对坐标，范围 [0, kernel_h) 和 [0, kernel_w)

**计算过程说明**：
1. 对于输出特征图的每个位置 `(n, c_out, h_out, w_out)`
2. 遍历卷积核的所有位置 `(kh, kw)` 和对应的输入通道 `c_in`
3. 计算对应的输入特征图坐标 `(h_in, w_in)`
4. 将输入值与卷积核权重相乘并累加
5. 最后加上偏置 `bias[c_out]`

### 1.4 输入输出规格

**输入张量**：
- `input`: `[N, C_in, H_in, W_in]` - 输入特征图
- `weight`: `[C_out, C_in/groups, kernel_h, kernel_w]` - 卷积核
- `bias`: `[C_out]` - 偏置（可选）

**输出张量**：
- `output`: `[N, C_out, H_out, W_out]` - 输出特征图

**输出维度计算**：
```
H_out = floor((H_in + padding_top + padding_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
W_out = floor((W_in + padding_left + padding_right - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
```

### 1.5 支持的数据类型要求
- `half` (FP16)
- `float` (FP32)
- `bfloat16` (BF16)

---

## 二、约束条件要求

### 2.1 任务维度范围

| 参数 | 约束条件 | 说明 |
|------|---------|------|
| N | 1 ≤ N ≤ 16 | Batch size |
| C_in | 1 ≤ C_in ≤ 4096 | 输入通道数 |
| C_out | 1 ≤ C_out ≤ 4096 | 输出通道数 |
| H_in, W_in | 1 ≤ H_in, W_in ≤ 4096 | 输入高度和宽度 |
| kernel_h, kernel_w | 1 ≤ kernel_h, kernel_w ≤ 7 | 卷积核大小 |
| stride_h, stride_w | 1 ≤ stride_h, stride_w ≤ 4 | 步长 |
| groups | groups = 1 | 分组数 |
| dilation_h, dilation_w | 1 ≤ dilation_h, dilation_w ≤ 2 | 膨胀系数 |
