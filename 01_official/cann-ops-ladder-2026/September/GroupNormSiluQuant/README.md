# GroupNormSiluQuant算子

## 一、赛题背景

GroupNormSiluQuant 是将 **组归一化（GroupNorm）**、**SiLU 激活** 与 **量化（Quant）** 三个操作融合的推理算子，广泛应用于 Stable Diffusion、DiT 等扩散生成模型的推理加速场景。相比分开执行三个算子，融合实现消除中间结果的全局内存往返，显著降低推理延迟。

本题要求基于上述融合逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 GroupNormSiluQuant 算子。

## 二、算子功能描述

对输入 self 完成三个操作的融合计算：

1. **GroupNorm**：将通道维度 C 分为 group 组，每组内沿 (C/group, 空间维) 计算均值与方差，做归一化与仿射变换，同时输出均值 meanOut 与标准差倒数 rstdOut。
2. **SiLU**：对归一化结果施加 `x / (1 + e^(-x))` 激活。
3. **Quant**：`round(siluOut / quantScale)` 量化并截断到 int8 范围，输出 out。

算子的核心难点在于：分组归约的高效实现、SiLU 的 exp 数值稳定、量化 round + int8 clamp，以及三算子融合的访存优化。

## 三、核心定义与约束

### 3.1 参考算子

等价 numpy 实现：

```python
import numpy as np

def group_norm_silu_quant(self, gamma, beta, quantScale, group, eps=1e-5, activateSilu=True):
    N, C = self.shape[0], self.shape[1]
    Cpg = C // group
    spatial = self.shape[2:]
    x32 = self.astype(np.float32)

    x_g = x32.reshape(N, group, Cpg, *spatial)
    axes = tuple(range(2, 2 + len(spatial) + 1))
    mean = x_g.mean(axis=axes, keepdims=True)        # (N, group, 1, ...)
    var = x_g.var(axis=axes, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)

    gamma_r = gamma.astype(np.float32).reshape(1, group, Cpg, *([1] * len(spatial)))
    beta_r = beta.astype(np.float32).reshape(1, group, Cpg, *([1] * len(spatial)))
    gn_out = (x_g - mean) * rstd * gamma_r + beta_r

    if activateSilu:
        silu_out = gn_out / (1.0 + np.exp(-gn_out))
    else:
        silu_out = gn_out

    qs_r = quantScale.astype(np.float32)
    if qs_r.size != 1:
        qs_r = qs_r.reshape(1, group, Cpg, *([1] * len(spatial)))
    out = np.clip(np.rint(silu_out / qs_r), -128, 127).astype(np.int8).reshape(N, C, *spatial)

    return out, mean.reshape(N, group), rstd.reshape(N, group)
```

### 3.2 数学公式

GroupNorm（每组 S_i 含 C/group 个通道的空间位置）：

$$\mu = E[x], \quad \sigma^2 = Var[x], \quad \text{groupNormOut} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

$$\text{meanOut} = \mu, \quad \text{rstdOut} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}$$

SiLU 激活：

$$\text{siluOut} = \frac{\text{groupNormOut}}{1 + e^{-\text{groupNormOut}}}$$

量化：

$$\text{out} = \text{clamp}\left(\text{round}\left(\frac{\text{siluOut}}{\text{quantScale}}\right),\ -128,\ 127\right)$$

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | self | tensor | (N, C, H, W) | float16, float32 | ND | 输入张量 |
| INPUT(必选) | gamma | tensor | (C,) | float16, float32 | ND | 缩放 |
| INPUT(必选) | beta | tensor | (C,) | float16, float32 | ND | 偏置 |
| INPUT(必选) | quantScale | tensor | 标量 或 (C,) | float32 | ND | 量化尺度 |
| OUTPUT(输出) | out | tensor | (N, C, H, W) | int8 | ND | 量化结果 |
| OUTPUT(输出) | meanOut | tensor | (N, group) | float16, float32 | ND | 均值，与 self 同 dtype |
| OUTPUT(输出) | rstdOut | tensor | (N, group) | float16, float32 | ND | 标准差倒数，与 self 同 dtype |

### 3.4 关键输入约束

- **数据类型**: self、gamma、beta 支持 float16 / float32；quantScale 固定为 float32。
- **维度场景**: 4 维 (N, C, H, W)。
- **维度取值范围(均为正整数)**:
  - N（batch）：N ∈ [1, 4096]
  - C（通道）：C ∈ [1, 4096]
  - H / W（空间维度）：H, W ∈ [1, 1024]
- **其他约束**: group 必须整除 C；不支持空 Tensor；gamma、beta 的维度必须与 C 一致。

### 3.5 核心属性说明

- **group**（int，必选）：分组数，须整除 C，取值范围 [1, C]。
- **eps**（float，默认 1e-5）：数值稳定常数，加入方差后开方，避免方差为 0 时除零。
- **activateSilu**（bool，默认 true）：是否开启 SiLU 激活。

### 3.6 输出严格要求

- **形状约束**: out 与 self 同 shape；meanOut、rstdOut 为 (N, group)。
- **类型约束**: out 固定为 int8；meanOut、rstdOut 与 self 同 dtype（float16/float32）。
- **数值等价**: 输出结果应与参考实现（上述 numpy 实现）的计算结果在精度允许范围内一致。

### 3.7 特殊值处理规则

- **方差为 0 处理**: 当某组所有元素相等（方差为 0）时，`sqrt(σ² + ε)` 退化为 `sqrt(ε)`，避免除零产生 NaN/Inf。
- **量化截断**: round 后 clamp 到 int8 表示范围 [-128, 127]，超出范围的截断到边界。
- **SiLU 数值稳定**: 对较大正值，`x/(1+e^(-x))` 趋近 x，实现时需避免 `e^(-x)` 溢出。

## 四、规则要求

1. **数值一致性规则**: 给定相同输入，算子多次执行结果应完全一致（确定性算子，不含随机性）。

2. **舍入与截断规则**: 量化除法后采用就近舍入（`rint` 模式），随后 clamp 到 [-128, 127]，先 round 后 clamp，不可调换。

3. **性能要求**: 在保证数值正确性的前提下，充分利用 NPU 硬件特性：
   - 合理设计分组归约与三算子融合的分块与并行策略
   - 支持向量化计算，合理利用 Cube 与 Vector 计算单元
   - 优化内存访问模式，减少 GM 与 UB 之间的数据搬运（融合消除中间结果）
   - 合理利用多核并行，沿 batch 和/或分组维分配任务
   - 鼓励使用 Double Buffer / Ping-Pong 机制实现计算与搬运流水重叠
   - 代码质量：正确使用 AscendC API；Tiling 策略合理（多核负载均衡、UB 内存不溢出）；代码可读性与注释完整性

## 五、精度判断规则

- **精度要求**：计算结果需满足以下精度误差要求：
    - float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
    - float16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
    - int32：要求计算结果完全准确，无误差

## 六、示例说明

**示例1**: 小规模示例

- 输入 self: shape=(1, 4, 2, 2), dtype=float16, group=2（每组 C/group=2 个通道）
- 输入 gamma/beta: shape=(4,), dtype=float16；quantScale: shape=(1,), dtype=float32
- 计算过程: 每组沿 (2 通道, H, W) 共 8 个元素计算均值方差并归一化，SiLU 激活后按 quantScale 量化
- 输出: out shape=(1, 4, 2, 2) int8，meanOut/rstdOut shape=(1, 2)

**示例2**: 中等规模示例

- 输入 self: shape=(2, 32, 16, 16), dtype=float32, group=8（每组 4 通道）
- 每组沿 (4 通道, H, W) 共 1024 个元素归约，8 组独立统计
- 输出: out shape=(2, 32, 16, 16) int8，meanOut/rstdOut shape=(2, 8)

**示例3**: 真实场景示例

- 输入 self: shape=(8, 512, 64, 64), dtype=float32, group=64（每组 8 通道，扩散模型常见配置）
- 每组沿 (8 通道, H, W) 共 32768 个元素归约，64 组独立统计
- 输出: out shape=(8, 512, 64, 64) int8，meanOut/rstdOut shape=(8, 64)
