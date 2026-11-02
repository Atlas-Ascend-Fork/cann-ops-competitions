# 双分支 LayerNorm（DualAffineLayerNorm）算子设计文档

作者：TreamTik

# 需求背景（required）

## 需求来源

本设计对应《题目1-双分支LayerNorm算子开发任务书》（2026-10-23 固定耗时目标版附件）。使用 Ascend C 实现共享统计量的双分支 LayerNorm，适配 Atlas A3（910C / Ascend910_9382）。代码合入目标为 [xLLM-AI/xllm](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops)。

## 背景介绍

同一输入分别执行两次 LayerNorm，会重复读取输入并计算均值、方差。本算子在一次 Kernel 中完成行归一化，再应用两套仿射参数，减少重复归约、全局内存访问和 Kernel 启动开销。

# 需求分析（required）

## 需求描述

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | [N,H] | FP16 / BF16 | ND 连续输入 |
| gamma1、beta1、gamma2、beta2 | [H] | 与 x 一致 | 两套独立仿射参数 |
| epsilon | 标量属性 | float | 默认 1e-5，范围 [1e-7,1e-2] |
| y1、y2 | [N,H] | 与 x 一致 | 每次调用新分配的独立连续输出 |

支持 `1<=N<=4096`、`1<=H<=8192`，包括非对齐 H 和连续张量的非零 storage offset。输入位于同一 NPU，数据为有限值，`abs(x)<=10000`，仿射参数绝对值不超过 8。仅实现前向。

## 需求拆解

1. 实现共享 FP32 统计量、归一化和两路仿射计算。
2. 提供 OpDef 注册、shape/dtype 推导、tiling 和两阶段 ACLNN API。
3. 提供 PyTorch NPU 正式入口及复用同一内核的 Kernel 直调入口。
4. 提供私有安装包、独立 C++ 样例及精度、性能、内存和工程测试。

# 详细设计（required）

## 算子分析

对每行长度 H 的输入计算：

```text
mu = sum(x) / H
variance = sum((x - mu)^2) / H
z = (x - mu) / sqrt(variance + epsilon)
y1 = cast(z * gamma1 + beta1)
y2 = cast(z * gamma2 + beta2)
```

统计量、归一化及仿射运算使用 FP32，最后转换为输入类型。方差分母为 H。常量行及 H=1 时，输出须精确等于对应 beta。

## 算子实现

### host 侧设计

框架入口校验输入个数、设备、维度、dtype、连续性、尺寸和属性。非法 epsilon、未知属性和不匹配的仿射参数直接报错。输出分别分配，底层存储互不重叠，也不复用输入存储。

OpDef 和推导函数声明五个输入、两个输出和 epsilon 属性。ACLNN 采用 workspace 查询与执行两阶段接口；显式 GM workspace 为 0。正式 `run(inputs, attributes)` 调用 ACLNN，`run_direct` 使用相同计算语义和 tiling 参数进行交叉验证。

设可用 Vector 核数为 C，`blockDim=min(N,C)`。第 k 个核处理行 `k, k+blockDim, ...`，每行只由一个核计算，不使用跨核归约或原子操作。tiling 传递 N、H、对齐长度、核数及 epsilon，dtype 通过模板选择。

### kernel 侧设计

1. 每核加载四个仿射参数，后续处理多行时复用。
2. 搬入一行 x 并转换为 FP32。先减去该行首元素得到平移数据，再求均值和中心化结果；方差使用中心化结果的平方和除以 H。
3. 计算 `rsqrt(variance+epsilon)`，得到共享归一化向量 z。
4. 依次计算两路仿射，转换为输入类型并分别写回。

平移不改变数学上的中心化结果，可降低大偏移、小方差输入的消减误差。常量行平移后为精确零，保证输出 beta 的要求。

行 r、列 j 的输入及输出元素偏移均为 `r*H+j`，仿射参数偏移为 j；字节偏移为元素偏移乘以 2。地址从张量当前有效数据指针开始计算，保留 storage offset。偏移计算使用 64 位整数。

令 `A=ceil(H/16)*16`，对应 16 位元素的 32 字节对齐。非对齐搬运使用 DataCopyPad，有效读写长度为 H；归约仅覆盖有效元素，补齐区置零。CopyIn、Vector 计算和 CopyOut 按数据依赖同步，缓冲区在消费者完成后复用。

### 片上缓冲与工作区

| 区域 | 字节数 | 用途 |
|---|---:|---|
| 四个仿射参数 | 8A | 核内跨行复用 |
| 输入缓冲 | 2A | 单行搬入 |
| 输出缓冲 | 2A | 两路依次转换及搬出 |
| 三个 FP32 缓冲 | 12A | 中心化、平方归约及仿射计算 |
| 合计 | 24A | H=8192 时为 192 KiB |

归约临时区复用计算缓冲，不额外分配 GM 中间张量。Host 根据目标平台 UB 容量核对上述预算。输出张量由调用方分配，不计入片上缓冲。

### 性能优化方案

共享一次输入搬运、均值、方差和归一化计算；中间向量保留在 UB。按行分核保证连续访问，每核复用仿射参数，减少多行场景的重复搬运。单次 Kernel 完成两路输出，避免将中间统计量写入 GM。

## 支持硬件与约束

目标为 Atlas A3 / Ascend910_9382，48 个 Vector Core、24 个 Cube Core，CANN 9.1.0。计算使用 Vector Core。软件及驱动按任务附件的指定环境核验。

计算使用调用方当前 stream，不修改输入。框架封装将调用参数提交到 torch_npu 任务队列，记录输入的 stream 使用关系，支持调用返回后提前释放输入。输出按框架分配器的 stream 规则回收，支持多次调用保留独立结果。

# 可维可测分析

## 精度标准/性能标准

CPU FP64 公式转换至输出类型作为 Golden。逐元素满足 `abs(actual-reference)<=atol+rtol*abs(reference)`：FP16 的 atol、rtol 均为 2e-3，BF16 均为 2e-2。常量行及 H=1 逐元素精确比较 beta。

测试覆盖 46 个公开输入、12 个额外种子及存储偏移输入，并在 ACLNN 和直调入口交叉验证。20 类非法场景分别检查 prepare、run、run_direct。另检查输出存储独立、输入不变、非法元数据、两个独立 stream 以及输入和输出生命周期。

10 个性能用例按附件 `performance_targets.json` 的固定微秒上限逐项判定。预热 50 次，测量 5 轮、每轮 200 次，以轮均耗时的中位数比较。计时从提交前到该轮设备操作全部完成，包含每次新输出分配，排除输入传输和 Golden 比较。

内存检查要求显式 GM workspace 为 0，活跃张量峰值增量不超过输出字节数加 16 KiB。释放并同步后，重复 20 次调用须回到调用前的活跃张量字节数。真实测试结果及环境指纹单独记录。

## 兼容性分析

沿用 xllm_ops 的构建和框架接入方式，ACLNN 与直调共用 Ascend C 计算内核。私有 .run 安装到用户目录，不覆盖系统 CANN。保持 FP16/BF16、ND 连续布局、当前 stream 和独立输出语义；不增加广播或反向接口。
