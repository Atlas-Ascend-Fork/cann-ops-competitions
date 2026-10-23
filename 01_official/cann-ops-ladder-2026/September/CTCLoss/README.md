# CTCLoss算子

## 一、赛题背景

CTC（Connectionist Temporal Classification，联结时序分类）Loss是语音识别（ASR）、手写识别和视频理解中经典的序列对齐损失函数。它解决的核心问题是：在输入序列长度和目标序列长度不一致的情况下，无需显式对齐标注即可端到端训练。CTC通过动态规划枚举所有可能的对齐路径，计算目标序列的条件概率，在DeepSpeech、Whisper、WeNet等主流语音模型中广泛应用。

CTC Loss的GPU实现极具挑战性：其核心是沿时间轴的前向-后向动态规划（Forward-Backward Algorithm），具有天然的时序依赖——当前时刻的计算依赖上一时刻的结果。在GPU/NPU上，这意味着无法简单地将所有时间步完全并行。如何在保持动态规划数据依赖正确性的前提下，最大化硬件并行度和减少全局内存往返，是CTC算子性能优化的核心。

本题要求基于PyTorch中`torch.nn.CTCLoss`的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能的CTCLoss算子。

相似类型算子的实现可参考开源仓：

https://gitcode.com/cann/ops-nn

## 二、算子功能描述

实现的CTCLoss算子需完成以下核心计算：

1. **序列展开**: 将目标序列展开为含blank间隔的扩展序列（插入blank符号）
2. **前向动态规划**: 沿时间轴正向递推计算alpha矩阵（所有合法对齐路径的累积log概率）
3. **损失计算**: 由alpha计算负对数似然损失

> **前置说明**: 输入 `log_probs` 为已沿类别维做完 log_softmax 的 log 概率，算子内部不做 log_softmax 预处理，与 PyTorch `torch.nn.CTCLoss` 接口一致。

算子的核心难点在于：动态规划的时序依赖要求alpha/beta在时间步间传递，且扩展序列中的blank和重复字符规则增加了合法状态的判定复杂度。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch原生算子：`torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)`

等价python实现：
```python
import torch

def log_add_exp(a, b):
    """数值稳定的log-sum-exp"""
    max_val = torch.max(a, b)
    return max_val + torch.log(1 + torch.exp(-torch.abs(a - b)))

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    # log_probs: (T, N, C) — 时间步, batch, 类别数(含blank)
    # targets: (N, S) — batch, 最大目标序列长度
    T, N, C = log_probs.shape
    losses = []
    for b in range(N):
        T_b = input_lengths[b]
        S_b = target_lengths[b]
        # 扩展序列: 每个label前后插入blank
        extended = [blank] + [c for t in targets[b, :S_b] for c in (t, blank)]
        S_ext = len(extended)
        # Forward DP
        alpha = torch.full((T_b, S_ext), -float('inf'))
        alpha[0, 0] = log_probs[0, b, blank]
        if S_ext > 1:
            alpha[0, 1] = log_probs[0, b, extended[1]]
        for t in range(1, T_b):
            for s in range(S_ext):
                # 从同状态转移 (s来源)
                alpha[t, s] = log_add_exp(alpha[t, s], alpha[t-1, s])
                # 从s-1转移
                if s >= 1:
                    alpha[t, s] = log_add_exp(alpha[t, s], alpha[t-1, s-1])
                # 从s-2跳过blank (skip condition)
                if s >= 2 and extended[s] != blank and extended[s] != extended[s-2]:
                    alpha[t, s] = log_add_exp(alpha[t, s], alpha[t-1, s-2])
                alpha[t, s] += log_probs[t, b, extended[s]]
        # Loss from last timestep
        loss = -log_add_exp(alpha[-1, -1], alpha[-1, -2])
        losses.append(loss)
    if len(losses) == 0:
        return torch.tensor(0.0)
    return torch.mean(torch.stack(losses))
```

### 3.2 数学公式

> 输入 `l_{t,i}` 为已做完 log_softmax 的 log 概率。

**序列扩展:** 长度为S的目标序列扩展为S'=2S+1的扩展序列，第k个元素（targets为0-indexed）:
- k=0: blank
- k=2j-1: targets[j-1]（j=1,2,...,S，对应目标序列第j个元素）
- k=2j: blank

**前向DP (alpha):**
$$\alpha(t, s) = l_{t, ext[s]} + \log\begin{cases}
e^{\alpha(t-1, s)} & \text{if } s = 0 \\
e^{\alpha(t-1, s)} + e^{\alpha(t-1, s-1)} & \text{if } s \geq 1, \delta = 0 \\
e^{\alpha(t-1, s)} + e^{\alpha(t-1, s-1)} + e^{\alpha(t-1, s-2)} & \text{if } s \geq 2, \delta = 1
\end{cases}$$
其中 δ = 1（当且仅当 s ≥ 2 且 ext[s] ≠ blank 且 ext[s] ≠ ext[s-2]），否则δ = 0

**损失:**
$$L = -\log\left(e^{\alpha(T-1, S'-1)} + e^{\alpha(T-1, S'-2)}\right)$$

其中 log_add_exp(x, y) = max(x, y) + log(1 + exp(-|x-y|))（数值稳定的log-sum-exp）

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | log_probs | tensor | (T, N, C) | float16, float32 | ND | 输入log概率（已沿类别维做log_softmax） |
| INPUT(必选) | targets | tensor | (N, S) | int32, int64 | ND | 目标标签序列(不含blank)，padding值用-1或≥C的值表示 |
| INPUT(必选) | input_lengths | tensor | (N,) | int32 | ND | 每个batch样本的有效时间步数 |
| INPUT(必选) | target_lengths | tensor | (N,) | int32 | ND | 每个batch样本的有效目标序列长度 |
| ATTR(属性) | blank | int | - | - | - | blank标签索引，默认0 |
| ATTR(属性) | reduction | string | - | - | - | 归约方式: 'none', 'mean', 'sum'，默认'mean' |
| ATTR(属性) | zero_infinity | bool | - | - | - | 当正无穷时是否置零，默认false |
| OUTPUT(输出) | neg_log_likelihood | tensor | (N,) 或 (1,) | float32 | ND | 负对数似然损失（reduction='none'时为(N,)，否则为标量） |

### 3.4 关键输入约束

- **维度取值范围**: T(时间步)∈[1, 16384], N(batch)∈[1, 512], C(类别)∈[2, 20001], S(目标长度)∈[1, 2048]
- **序列长度**: input_lengths[i] ≤ T, target_lengths[i] ≤ S, 且 target_lengths[i] ≤ input_lengths[i]（CTC的合法性约束）
- **targets填充**: targets中超出target_lengths[i]的位置填充为无效值
- **blank与C的关系**: blank ∈ [0, C-1]

### 3.5 核心属性说明

- **blank**（int，默认0）: 空白标签索引。blank不能是目标序列中的有效标签
- **reduction**（string，默认'mean'）: 'none'返回逐样本损失，'mean'返回平均值，'sum'返回总和
- **zero_infinity**（bool，默认false）: 当损失为+∞时（通常由非法对齐导致），是否将损失和梯度置零

### 3.6 输出严格要求

- neg_log_likelihood为负对数似然，非负值。loss=0表示完美预测
- 中间alpha矩阵使用FP32 log-space存储，避免下溢
- reduction='none' 时输出shape为(N,)，其他归约方式输出标量shape(1,)

### 3.7 特殊值处理规则

- **无限对齐**: 某些输入下合法对齐路径可能不存在（概率→0），损失为+∞。当zero_infinity=true时，此类样本损失置0
- **blank规则**: 连续两个相同的label之间必须至少有一个blank。DP递推时的skip条件需正确实现

## 四、规则要求

1. **log-space计算规则**: 所有概率计算必须在log空间进行（使用log_add_exp），禁止在普通空间做乘法（会导致下溢至0）

2. **DP依赖规则**: DP计算具有真实的时序依赖——alpha[t]依赖alpha[t-1]。此数据依赖不可打破，但选手可通过tiling策略（如按batch并行、按S'维并行t-1内的独立状态）最大化并行度

3. **扩展序列规则**: 每个目标序列扩展后长度为2S+1，其中blank出现在第0,2,4,...位，labels出现在第1,3,5,...位

4. **skip条件规则**: 从s-2跳跃到s仅在ext[s]≠blank且ext[s]≠ext[s-2]时合法，防止跳过blank导致连续相同label被合并

## 五、示例说明

**示例1**: 最短有效序列（单字符识别）

- log_probs: shape=[2, 1, 3]（T=2, N=1, C=3含blank=0），dtype=float32
- targets: [[1]], dtype=int32, target_lengths=[1]
- input_lengths=[2]
- 扩展序列: [0, 1, 0] (S'=3)
- alpha递推:
  - t=0: alpha[0] = [log_prob(0,blank), log_prob(0,1), -inf]
  - t=1: alpha[1,s] = log_add_exp(alpha[0,s], alpha[0,s-1]) + log_prob(1,ext[s])（s=2时额外加alpha[0,0]项）
- loss = -log_add_exp(alpha[1,1], alpha[1,2])
- 输出 loss: 标量 ≈ 非负值

**示例2**: 多batch变长序列

- T=50, N=4, C=30
- targets: 4个不同长度的序列
- input_lengths=[50, 35, 42, 40], target_lengths=[20, 12, 18, 15]
- 每batch独立做DP，扩展后S'分别为41, 25, 37, 31
- 输出: shape=[4]（每样本loss） 或 标量（reduction='mean'）

**示例3**: 典型语音识别场景

- T=1500, N=16, C=5001（含blank，对应5000个汉字/token）
- 实际目标长度约50-100个token
- 需大规模DP矩阵: T×S'≈1500×200，FP32存储
- 核心瓶颈: T维度无法并行化，如何在每个时间步内最大化并行

## 六、测试用例覆盖范围

- **数值精度**: float32, float16（log_prob）

- **合法对齐**: CTC约束满足的所有合法对齐路径概率之和=1（可验证概率归一化）

- **维度覆盖**:
  - 短序列: T=2-10, S=1-3
  - 中等序列: T=100-500, S=20-100
  - 长序列（真实ASR场景）: T=500-16384, S=100-2048

- **边界覆盖**:
  - T=target_length（最短有效T，每个时间步恰好输出一个label）
  - 单字符目标序列+TPadding
  - 所有目标相同字符（考验skip规则）
  - target_length=0（空白目标）

- **正确性验证基准**: 以PyTorch `torch.nn.CTCLoss` 在FP64精度下的结果为golden
