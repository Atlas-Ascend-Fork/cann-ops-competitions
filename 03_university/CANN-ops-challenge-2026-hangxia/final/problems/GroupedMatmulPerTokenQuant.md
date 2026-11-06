# GroupedMatmulPerTokenQuant算子

## 一、赛题背景

在混合专家模型（Mixture of Experts，MoE）推理中，路由后的 token 按专家连续排列，不同专家接收到的 token 数通常差异较大。GroupedMatmul（GMM）将多个专家的矩阵乘统一调度，避免逐专家调用带来的启动开销。

本赛题采用 A8W8 全量化计算：激活和专家权重均为 INT8，激活使用 per-token 量化因子，权重使用 per-channel 量化因子，结果反量化后输出 FLOAT16 或 BFLOAT16。本题要求基于 PyTorch 中分组矩阵乘、per-token/per-channel 反量化的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 GroupedMatmulPerTokenQuant 算子。

本题采用 M 轴分组、单 Tensor、ND 格式，输入固定不转置，不包含 bias 或激活函数。本文中的“per-token 量化”指使用输入量化因子完成反量化；输出浮点矩阵 y 。

## 二、算子功能描述

实现的 GroupedMatmulPerTokenQuant 算子需完成以下核心计算：

1. **分组矩阵乘**：所有专家共用 K、N，各专家的 token 数 M_e 可以不同，允许 M_e=0。
2. **全量化反量化**：INT8×INT8 使用 INT32 累加；权重 scale 按专家、按输出通道使用，激活 scale 按全局 token 行使用。
3. **输出类型转换**：反量化的中间计算使用 FLOAT32，最后转换为 FLOAT16 或 BFLOAT16，输出按原 token 顺序拼接。

## 三、核心定义与约束

### 3.1 参考算子

以下为 PyTorch CPU 参考实现（输入输出均为 CPU Tensor）：

```python
import torch

@torch.no_grad()
def grouped_matmul_pertoken_quant_cpu(
        x, weight, xScale, weightScale, groupList,
        groupListType=0, outputDtype="float16"):
    M, K = x.shape
    E, K_w, N = weight.shape
    assert all(t.device.type == "cpu"
               for t in (x, weight, xScale, weightScale, groupList))
    assert x.dtype == weight.dtype == torch.int8
    assert K == K_w and xScale.shape == (M,)
    assert xScale.dtype == torch.float32
    assert weightScale.shape == (E, N)
    assert groupList.shape == (E,) and groupList.dtype == torch.int64
    assert outputDtype in ("float16", "bfloat16")
    assert groupListType in (0, 1)
    assert torch.all((groupList >= 0) & (groupList <= M))
    scale_dtype = torch.float32 if outputDtype == "float16" else torch.bfloat16
    assert weightScale.dtype == scale_dtype
    if groupListType == 0:
        ends = groupList
        assert torch.all(ends[1:] >= ends[:-1])
    else:
        ends = torch.cumsum(groupList, dim=0)
    assert ends[-1].item() == M

    output_dtype = getattr(torch, outputDtype)
    y = torch.empty((M, N), dtype=output_dtype)
    start = 0
    for e, end in enumerate(ends.tolist()):
        # 分块计算，避免为整个专家生成大尺寸 INT64 临时矩阵。
        for m0 in range(start, end, 32):
            m1 = min(m0 + 32, end)
            for n0 in range(0, N, 128):
                n1 = min(n0 + 128, N)
                acc = torch.zeros((m1 - m0, n1 - n0), dtype=torch.int64)
                for k0 in range(0, K, 256):
                    k1 = min(k0 + 256, K)
                    acc += (x[m0:m1, k0:k1].to(torch.int64)
                            @ weight[e, k0:k1, n0:n1].to(torch.int64))
                assert torch.all((acc >= -(2**31)) & (acc <= 2**31 - 1))
                value = acc.to(torch.int32).to(torch.float32)
                value = value * weightScale[e, n0:n1].float().reshape(1, -1)
                value = value * xScale[m0:m1].reshape(-1, 1)
                y[m0:m1, n0:n1] = value.to(output_dtype)
        start = end
    return y
```

### 3.2 数学公式

**阶段一 — 确定 token 所属专家：**

输入 x 的 M 行 token 已按专家顺序连续排列，groupList 指定每个专家处理哪些行。

- groupListType=0 时，groupList 表示各专家的累计结束位置。例如 groupList=[2,3]，专家 0 处理第 0、1 行，专家 1 处理第 2 行。
- groupListType=1 时，groupList 表示各专家的 token 数。例如 groupList=[2,1] 表示同样的分组。
- 专家没有 token 时跳过计算。以下公式中，$m$ 始终是 x 的全局行下标，$e$ 是该行所属的专家编号。

**阶段二 — 量化矩阵乘与反量化：**

$$
\text{matmulOut}_{fp32}[m,n]
=\text{cast}_{fp32}\!\left(\sum_{k=0}^{K-1}x[m,k]\times weight[e,k,n]\right)
\times\text{cast}_{fp32}(weightScale[e,n])\times xScale[m]
$$

- $x$ 为 INT8 类型，形状 $(M,K)$，固定不转置。
- $weight$ 为 INT8 类型，形状 $(E,K,N)$，每个专家使用自己的权重矩阵 $weight[e,:,:]$，固定不转置。
- 矩阵乘内部使用 INT32 累加，随后 `cast` 为 FLOAT32。
- $weightScale$ 为 per-channel（列级）反量化因子，形状 $(E,N)$；$weightScale[e,n]$ 对应专家 e 的第 n 个输出通道。
- $xScale$ 为 per-token（行级）反量化因子，FLOAT32，形状 $(M,)$；$xScale[m]$ 对应输入的第 m 行 token。
- CPU 参考实现先乘 weightScale，再乘 xScale，两次乘法均以 FLOAT32 计算，BFLOAT16 类型的 weightScale 先转换为 FLOAT32。参赛实现允许调整乘法顺序或采用数学等价的融合方式，最终输出须满足第五节的精度要求，不要求与参考实现逐位一致。

**阶段三 — 输出类型转换：**

$$
y[m,n]=\text{cast}_{outputDtype}(\text{matmulOut}_{fp32}[m,n])
$$

- outputDtype 为 float16 时，输出 y 为 FLOAT16。
- outputDtype 为 bfloat16 时，输出 y 为 BFLOAT16。
- 最终类型转换采用就近偶数舍入（round-half-to-even）。
- 输出 y 的形状为 $(M,N)$，行顺序与输入 x 一致，保留计算结果中的正值、负值和零值。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | x | tensor | (M, K) | INT8 | ND | 按专家连续排列的激活，固定不转置 |
| INPUT(必选) | weight | tensor | (E, K, N) | INT8 | ND | 每个专家一份权重，固定不转置 |
| INPUT(必选) | xScale | tensor | (M,) | FLOAT32 | ND | 每个 token 的反量化因子 |
| INPUT(必选) | weightScale | tensor | (E, N) | FLOAT32 / BFLOAT16 | ND | 每个专家、每个输出通道的反量化因子 |
| INPUT(必选) | groupList | tensor | (E,) | INT64 | ND | 专家累计结束位置或各专家 token 数 |
| OUTPUT(输出) | y | tensor | (M, N) | FLOAT16 / BFLOAT16 | ND | 反量化后的浮点结果 |
| ATTR(属性) | groupListType | int | 标量 | INT64 | — | 0：累计结束位置；1：各专家 token 数 |
| ATTR(属性) | outputDtype | string | 标量 | — | — | float16 或 bfloat16 |

**数据类型组合必须满足：**

| x | weight | xScale | weightScale | y |
|---|--------|--------|-------------|---|
| INT8 | INT8 | FLOAT32 | FLOAT32 | FLOAT16 |
| INT8 | INT8 | FLOAT32 | BFLOAT16 | BFLOAT16 |

两种输出模式都必须支持，weightScale 必须按对应的输入类型参与计算。

### 3.4 关键输入约束

- **数据类型**: x、weight 为 INT8；xScale、weightScale 为 FLOAT32（weightScale 亦可为 BFLOAT16）；groupList 为 INT64；输出 y 为 FLOAT16 或 BFLOAT16。
- **维度场景**: 2 维 x (M, K)；3 维 weight (E, K, N)；1 维 xScale (M,)、weightScale (E, N)；1 维 groupList (E,)；2 维输出 y (M, N)。
- **维度取值范围（均为正整数）**: 1≤E≤128，1≤M≤32768，1≤K≤8192，1≤N≤8192。
- **内存范围**：每个测试用例全部输入与输出 Tensor 的逻辑字节数之和不超过 1 GiB，workspace 单独统计。
- **分组要求**：只支持 M 轴分组，所有专家共用 K、N，所有 token 已按专家顺序连续排列，不需要选手执行路由排序。
- **groupListType=0**：groupList 为非负、单调非递减的累计结束位置，最后一个值必须等于 M；相邻值相等表示空专家。
- **groupListType=1**：groupList 为非负的专家 token 数，所有值之和必须等于 M；值为 0 表示空专家。
- **空专家**：允许首部、中间、尾部专家为空；M≥1，允许 E>M。
- **布局要求**：所有 Tensor 均连续，x、weight 固定不转置；仅要求逻辑 shape 正确，不要求 M_e、K、N 为 16/32/64 的倍数。尾块不足部分必须正确处理。
- **scale 数值**：xScale、weightScale 为有限正数，存储后的数值位于 [1e-5, 1.0]；不包含 NaN、Inf、负数或零 scale。
- **整数范围**：x、weight 覆盖 INT8 全范围 [-128,127]。K≤8192 时任一部分和绝对值不超过 8192×16384=134217728，保证 INT32 累加不溢出。
- **输出范围**：测试数据保证每个反量化 FP32 结果有限、绝对值不超过 60000，避免 FLOAT16 输出溢出；允许负值、零和很小的输出值。
- **全覆盖要求**：groupList 覆盖全部 M 行，输出没有未定义的尾部行。不支持 bias、offset、激活函数、NZ 格式、INT4、K 轴分组或 Tensor 列表输入。

### 3.5 核心属性说明

- **groupListType**（int，默认 0）: 支持 0、1；同一 token 分组的两种表示应产生一致输出。
- **outputDtype**（string，默认 float16）: 支持 float16、bfloat16，由评测输入指定；选手不得自行改变输出类型。
- **其他**: 分组轴固定为 M 轴，计算结果不经过激活函数；赛题可变属性仅包含 groupListType 和 outputDtype。
- xScale、weightScale、groupList 都是运行时输入。相同 shape 下它们的值可以变化，不允许依赖固定专家负载、固定 scale 或预先缓存的结果。

### 3.6 输出严格要求

- y 的 shape 必须为 (M,N)，dtype 必须与 outputDtype 一致，数据格式为连续 ND。
- 专家输出写入对应的全局 token 区间，不能按执行完成顺序拼接，也不能改变 token 顺序。
- 每次调用必须写完全部 M×N 个输出元素；空专家不读写任何 token 行。
- 所有输入均为只读，不得修改 x、weight、scale 或 groupList。
- 输出应在第五章规定的误差范围内与 CPU golden 一致；不要求不同合法实现逐位相同。

### 3.7 特殊值处理规则

- **空专家**：跳过矩阵乘和反量化，后续专家仍使用正确的权重下标及 token 区间。
- **零输入行**：当某个 token 的 x 全为 0 时，对应 y 全为数值 0；不输出任何新的 scale。
- **负结果**：保持负号，不进行激活截断。
- **非对齐尾块**：尾块处理不得影响有效结果，不得越界读取或写出。
- **INT32→FP32 舍入**：绝对值大于 2^24 的整数可能无法由 FP32 精确表示，应按正常类型转换处理，不能声称转换永不丢失精度。
- **BF16 输入 scale**：以实际 BF16 存储值参与计算。

## 四、规则要求

1. **正确性和确定性**：给定相同输入，多次执行结果应一致；专家调度顺序不能改变分组归属或产生跨专家数据污染。

2. **计算与实现要求**：使用 Ascend C 实现；整数矩阵乘保证正确累加，若采用 K 轴切分，完整整数结果在反量化前合并。允许使用 Ascend C Matmul 等基础 API，不允许直接调用实现本题完整功能的现成算子完成参赛实现，也不允许在 CPU 上计算输出。

3. **资源与提交要求**：提交 Host/Tiling、Kernel、构建与调用入口及必要说明。workspace 由 Host 明确声明，不能越界或跨调用依赖缓存；不得写死用例结果、根据用例编号选择输出或读取评测器的 golden。

4. **赛事规则**: 本赛题的实现与提交必须遵守《赛事规则》，完整内容如下：

   1、请按照题目要求完成对应的AscendC算子，补齐算子Host侧的Tiling逻辑和Device侧的Kernel逻辑，其中Host侧仅负责参数校验、shape推导、tiling逻辑和启动配置；Kernel侧负责完成该算子完整的计算流程。

   2、请按照规则完成算子开发，严禁出现违规情况，包括但不限于：

      - 1）在Host侧读取输入张量进行计算。
      - 2）在Host侧对输入数据进行重排。
      - 3）在Host侧生成预查表，传入Kernel侧直接取值计算。
      - 4）在Host侧完成向量、矩阵计算，将结果传入Kernel侧直接用于计算。
      - 5）硬编码、按测试用例或者数据生成规律跳算。
      - 6）写入标记跳过计算。
      - 7）利用旧输出或者未初始化内存跳过计算。
      - 8）逐条匹配评测用例编写专属实现。
      - 9）修改或者干扰判题、精度校验和计时流程。
      - 10）注入空kernel、识别预热轮次或把实际计算移出计时范围等行为。
      - 11）在多轮次的判题中加缓存来减少耗时。
      - 12）一次计算通过多次kernel来完成计算。

   3、比赛结束后，会进行最终的人工核验，如果出现未按照规则提交的代码，赛事官方会按违规代码作废成绩。

## 五、精度判断规则

精度要求：计算结果需满足以下精度误差要求：

- float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
- float16、bfloat16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
- int32：要求计算结果完全准确，无误差

## 六、示例说明

**示例 1：两个专家、不同 per-token/per-channel scale（FP16 输出）**

- E=2，M=3，K=2，N=2。
- x=[[1,2],[3,4],[-1,2]]。
- weight[0]=[[1,2],[3,4]]，weight[1]=[[2,-1],[1,3]]。
- xScale=[0.5,0.25,0.125]，weightScale=[[0.5,0.25],[0.25,0.5]]，均为 FLOAT32。
- groupList=[2,3]，groupListType=0，outputDtype=float16。
- 专家 0 处理全局第 0、1 行，INT32 结果为 [[7,10],[15,22]]。
- 专家 1 处理全局第 2 行，INT32 结果为 [[0,7]]。
- 先乘各专家的 weightScale，再乘全局对应 xScale，得到：

```text
y = [[1.75, 1.25],
     [1.875, 1.375],
     [0.0, 0.4375]]
```

上述结果均可由 FP16 精确表示。把 groupList 改为 [2,1]、groupListType 改为 1，输出应保持一致。

**示例 2：首部、中间、尾部空专家（BF16 输出）**

- E=5，M=2，K=2，N=1。
- x=[[1,2],[-1,1]]。
- weight=[[[9],[9]],[[2],[1]],[[8],[8]],[[1],[3]],[[7],[7]]]。
- xScale=[0.5,0.25]，weightScale=[[0.5],[0.5],[0.5],[0.25],[0.5]]；weightScale 存储为 BFLOAT16。
- groupList=[0,1,1,2,2]，groupListType=0，outputDtype=bfloat16。
- 专家 0、2、4 为空；专家 1 处理第 0 行，专家 3 处理第 1 行。
- 输出 y=[[1.0],[0.125]]，dtype 为 BFLOAT16。
- 等价计数表示为 groupList=[0,1,0,1,0]、groupListType=1。

**示例 3：保留负结果和零输入行**

- E=1，M=2，K=2，N=2，x=[[-1,0],[0,0]]。
- weight[0]=[[2,-3],[1,4]]，xScale=[0.5,0.25]，weightScale=[[0.25,0.5]]。
- groupList=[2]，groupListType=0，outputDtype=float16。
- 输出 y=[[-0.25,0.75],[0.0,0.0]]。
- 第一行负值必须保留，第二行为零输入行，结果为全零。
