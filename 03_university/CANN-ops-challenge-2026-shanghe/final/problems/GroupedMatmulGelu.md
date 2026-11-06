# GroupedMatmulGelu算子

## 一、赛题背景

在包含多专家线性层的网络中，各专家处理的 token 数可能不同。分组矩阵乘（Grouped Matmul，GMM）将这些变长矩阵乘组织为一次设备调用；INT8 量化进一步降低输入和权重的存储及搬运开销。矩阵乘后的反量化与 GELU 激活若分别执行，会产生中间结果的 GM 读写和额外 kernel 调用。

本题要求基于 PyTorch 中 `torch.matmul` 与 `torch.nn.functional.gelu(..., approximate="tanh")` 的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 GroupedMatmulGelu 算子。

## 二、算子功能描述

实现的 GroupedMatmulGelu 算子需完成以下核心计算：

1. **变长分组矩阵乘**：x 和 weight 均为 INT8，矩阵乘采用 INT32 累加。每个专家的 K、N 相同，组内 token 数 M_e 可以不同。
2. **per-channel 反量化**：每个专家、每个输出通道使用独立的 FP32 系数 scale[e,n]，沿该专家的 M_e 维广播。
3. **tanh GELU 激活**：对反量化的 FP32 中间结果计算固定 tanh 近似定义的 GELU。
4. **输出转换**：激活结束后才转换为 FP16，保持输入 token 的分组排列顺序。

> **前置说明**：x、weight 已完成对称 INT8 量化，零点均为 0；x 已按专家排列。算子不执行路由、重排或输入量化。scale 为合并后的反量化因子：若激活采用 per-tensor 系数 s_x、权重采用 per-channel 系数 s_w[e,n]，则由调用方预先提供 scale[e,n] = s_x * s_w[e,n]。本题不含 per-token 系数、bias 和输出再量化。

## 三、核心定义与约束

### 3.1 参考算子

参考语义为整数矩阵乘、per-channel 广播乘法和 PyTorch `torch.nn.functional.gelu(..., approximate="tanh")` 的组合；单个普通 batch matmul 无法表达本题的变长专家分组。

以下为清晰表达计算顺序的 PyTorch CPU 参考实现（输入为 CPU Tensor）：

```python
import math

import torch

def grouped_matmul_gelu_cpu(x, weight, scale, group_list):
    # x: [M,K], weight: [E,K,N], scale: [E,N], group_list: [E]
    y = torch.empty((x.shape[0], weight.shape[2]), dtype=torch.float16)
    start = 0
    for e, end_value in enumerate(group_list):
        end = int(end_value)
        if start < end:
            c = x[start:end].to(torch.int64) @ weight[e].to(torch.int64)
            z = c.to(torch.int32).to(torch.float32) * scale[e]
            # 将FP32反量化结果升精度求GELU，降低golden函数自身误差。
            z64 = z.to(torch.float64)
            inner = math.sqrt(2.0 / math.pi) * (z64 + 0.044715 * z64 ** 3)
            tail = torch.exp(-2.0 * torch.abs(inner))
            factor = torch.where(inner >= 0, 1.0 / (1.0 + tail), tail / (1.0 + tail))
            y[start:end] = (z64 * factor).to(torch.float16)
        start = end
    return y
```

参赛实现可采用 FP32 计算路径，按第五节的精度要求与 golden 逐元素比对。

### 3.2 数学公式

令 $p_{-1}=0$、$p_e=\mathrm{group\_list}[e]$，专家 e 的 token 数为 $M_e=p_e-p_{e-1}$。对于 $0\le i<M_e$、$0\le n<N$：

$$
C_e[i,n]=\sum_{k=0}^{K-1}x[p_{e-1}+i,k]\cdot weight[e,k,n]
$$

$$
Z_e[i,n]=\operatorname{FP32}(C_e[i,n])\cdot scale[e,n]
$$

$$
\operatorname{GELU}_{tanh}(z)
=\frac{z}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}
\left(z+0.044715z^3\right)\right)\right)
$$

$$
y[p_{e-1}+i,n]=\operatorname{FP16}(\operatorname{GELU}_{tanh}(Z_e[i,n]))
$$

每个 scale[e,n] 仅沿专家 e 的 token 行广播，不沿 K 维量化，也不在专家之间共享。反量化发生在完整 K 维累加之后。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| INPUT(必选) | x | tensor | (M, K) | INT8 | ND | 按专家连续分组的激活，固定不转置 |
| INPUT(必选) | weight | tensor | (E, K, N) | INT8 | ND | 各专家权重，固定不转置 |
| INPUT(必选) | scale | tensor | (E, N) | FLOAT32 | ND | 合并后的 per-channel 反量化因子 |
| INPUT(必选) | group_list | tensor | (E,) | INT64 | ND | 各专家 token 的累积结束位置 |
| OUTPUT(输出) | y | tensor | (M, N) | FLOAT16 | ND | GELU 后的最终结果 |

### 3.4 关键输入约束

- **数据类型**: x、weight 为 INT8（对称量化，零点为 0）；scale 为 FLOAT32；group_list 为 INT64；输出 y 为 FLOAT16。
- **维度场景**: 2 维 x (M, K)；3 维 weight (E, K, N)；2 维 scale (E, N)；1 维 group_list (E,)；2 维输出 y (M, N)。
- **维度取值范围（均为正整数）**: $E\in[1,128]$、$M\in[1,65536]$、$K\in[1,8192]$、$N\in[1,4096]$。
- **形状一致**：weight.shape[1] = x.shape[1]，scale.shape = (E,N)，group_list.shape = (E,)。
- **分组约束**：group_list[0] ≥ 0；group_list 非递减；group_list[E-1] = M。允许 M_e=0，禁止整体 M=0。M、M_e、K、N 均不要求整块对齐，也不要求 M 被 E 整除。
- **数值范围**：x、weight 可覆盖 INT8 全值域 [-128,127]。scale 为有限非负 FP32，且 $\mathrm{scale}[e,n]\le\min(1,60000/(16384K))$，保证任意合法输入的反量化绝对值不超过 60000，最终 FP16 输出有限。
- **存储**：仅支持连续 ND Tensor，不支持 stride 变体、NZ 权重或转置属性。
- **组合规模**：INT8 权重总量 E×K×N 不超过 256 MiB；四个输入及最终输出的总字节数不超过 512 MiB。上面的各维上限不能无条件同时取最大值。

### 3.5 核心属性说明

- **属性参数**: 无。GELU 固定使用 tanh 近似定义，不支持 erf 模式切换。
- **group_list 语义**（固定为累积结束位置，而非各组长度）: 例如各组长度为 [2,0,3] 时，group_list 应传 [2,2,5]；对应专家 0 处理前两行，专家 1 为空，专家 2 处理后三行。调用方必须按这个语义提供合法输入。

### 3.6 输出严格要求

- 输出为一个连续 ND Tensor，shape=(M,N)，dtype=FLOAT16。
- 输出第 i 行与输入第 i 行对应，不增加 padding 行、不重排专家、不输出空专家占位行。
- 每个有效输出元素都必须写入；不得依赖输出缓冲区的初始内容。
- 禁止在 GELU 前将反量化结果转为 FP16；允许符合第五节精度的 FP32 等价实现和函数逼近。
- 必须根据实际输入数据和 group_list 计算，不得根据 M/E 猜测均分，也不得缓存固定输出。

### 3.7 特殊值处理规则

- **空专家**：连续相等的累积位置表示空组。跳过该组计算及输出写入，但其 weight/scale 专家索引仍然存在，后续专家不能错位。
- **零输入**：整行 x 为 0 或对应 weight 通道为 0，则该输出的累加、反量化和 GELU 结果均为 0。
- **零 scale**：scale[e,n]=0 时，专家 e 的第 n 个输出通道为 0；其他专家的同名通道不受影响。
- **负值**：GELU 在小负数区间输出负数，不能替换为 ReLU 或 FastGELU。
- **负数尾部精度**：直接计算 FP32 的 `1 + tanh(u)` 可能因相消而丢失小负数结果，不能保证通过第五节相对误差要求。可使用数学等价的稳定计算：令 `u = sqrt(2/pi) * (z + 0.044715*z^3)`、`t = exp(-2*abs(u))`；当 `u >= 0` 时返回 `z/(1+t)`，否则返回 `z*t/(1+t)`。该写法同时避免正指数溢出；golden 使用其 FP64 形式。PyTorch 的普通 FP32 GELU 输出不能直接视为本题高精度 golden。
- **整数累加边界**：$K\le8192$ 时，$\sum_k|x_k w_k|\le8192\times128\times128=2^{27}<2^{31}$，INT32 累加不会溢出。转换到 FP32 时允许该类型正常的舍入。
- **NaN/Inf**：合法输入不含 NaN/Inf，并受输出安全范围约束；合法输入的输出不得含 NaN/Inf。非法输入的设备侧报错行为不纳入赛题评测。

## 四、规则要求

1. **原生开发与融合**：采用 Ascend C 原生开发，不得拼装 aclnn 等既有算子提交；全部计算阶段在同一个 kernel 内完成。允许使用 GM workspace 暂存中间结果，中间结果的存储、搬运及必要的同步开销均计入算子执行耗时。计算路径、分块方式和同步策略由参赛者自行选择，最终输出须满足题目规定的功能和精度要求。可利用 Cube 与 Vector 协同、分块流水及片上数据复用优化性能，不限定具体实现方式。

2. **泛化要求**：测试时可能在同一 shape 下改变输入值和 group_list 内容，实现不得依赖特定用例数据。

3. **确定性与访问合法性**：相同输入重复执行应得到确定结果；正确处理尾块和跨专家偏移，不越界访问，不遗漏写出，不修改四个输入 Tensor。

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

本算子的最终输出为 float16，与 golden 逐元素比较，每个元素均需满足上述误差要求；golden 为 0 时仅判断绝对误差，避免相对误差除零。输出中不得出现 NaN 或 Inf。

## 六、示例说明

**示例 1：per-channel 系数与负数 GELU**

- x=[[1,-1],[-1,1]]，weight[0]=[[1,0],[0,1]]。
- scale[0]=[1.0,0.5]，group_list=[2]。
- INT32 累加结果为 [[1,-1],[-1,1]]。
- 反量化结果为 [[1,-0.5],[-1,0.5]]。
- tanh GELU 的浮点结果约为 [[0.841192,-0.154286],[-0.158808,0.345714]]，最终转换为 FP16。
- 第二个通道只使用 scale[0,1]；负数结果不能被 ReLU 截断。

**示例 2：中间空专家**

- E=3，M=5，各组长度 [2,0,3]，group_list=[2,2,5]。
- x[0:2] 使用 weight[0] 和 scale[0]。
- 专家 1 不输出任何行。
- x[2:5] 使用 weight[2] 和 scale[2]，不能错误使用专家 1 的参数。
