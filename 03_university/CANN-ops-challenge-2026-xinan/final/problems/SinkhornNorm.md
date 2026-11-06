# SinkhornNorm算子

## 一、赛题背景

DeepSeek V4 用 Hyper-Connections（HC）替代传统残差连接：每个 block 维护 `hc_mult` 份 hidden state 副本，block 间通过一组混合权重完成「塌缩-展开」传递。其核心约束是 Manifold-Constrained Hyper-Connections（mHC）：把副本间的组合矩阵通过 Sinkhorn-Knopp 迭代归一化投影到**双随机矩阵流形**上，保证信号在深层堆叠中非扩张传播。

Sinkhorn-Knopp 是运筹学与最优传输中的经典迭代算法，通过交替的按行/按列归一化把非负矩阵推近双随机矩阵。本题要求基于 DeepSeek V4 的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 SinkhornNorm 算子。

## 二、算子功能描述

实现的 SinkhornNorm 算子需完成以下核心计算：

1. **切分混合 logits**：`mixes` 最后一维按 `[H | H | H×H]` 三段切分（pre logits、post logits、comb logits），comb logits 按行主序 reshape 成 H×H 矩阵
2. **门控**：`pre = sigmoid(pre logits) + eps`；`post = 2 × sigmoid(post logits)`
3. **comb 初始化**：对 comb logits 按行（最后一维）做 softmax 后加 eps
4. **Sinkhorn-Knopp 迭代**：先做 1 次列归一化，再按（行归一化 → 列归一化）重复 `hc_sinkhorn_iters-1` 次，每次归一化的分母均为对应行/列和加 eps
5. **输出** pre、post、comb

## 三、核心定义与约束

### 3.1 参考算子

DeepSeek V4 Hyper-Connections（mHC）中 Sinkhorn 归一化的等价实现（线性映射与 scale/base 仿射已归并到输入 `mixes`，即 `mixes` 为三段混合 logits 的拼接）：

```python
import torch


def sinkhorn_norm(mixes, hc_mult=4, hc_sinkhorn_iters=20, eps=1e-6):
    H = hc_mult
    z_pre, z_post, z_comb = mixes.split([H, H, H * H], dim=-1)

    pre = torch.sigmoid(z_pre) + eps                                  # (..., H)
    post = 2 * torch.sigmoid(z_post)                                  # (..., H)

    # comb 的行方向为最后一维（softmax 沿 -1），列方向为倒数第二维（sum 沿 -2）
    comb = torch.softmax(z_comb.reshape(*mixes.shape[:-1], H, H), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)              # 第 1 次：列归一化
    for _ in range(hc_sinkhorn_iters - 1):                            # 再重复 iters-1 轮
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)          #   行归一化
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)          #   列归一化
    return pre, post, comb
```

### 3.2 数学公式

记输入 `mixes` 按 `[H | H | H²]` 切分为 `z_pre`、`z_post`、`z_comb`，`z_comb` 按行主序 reshape 为 `Z ∈ R^{H×H}`；记 `ε = eps`。

**门控**：

$$
\text{pre}_i = \sigma(z_{\text{pre},i}) + \epsilon, \qquad
\text{post}_i = 2\,\sigma(z_{\text{post},i})
$$

**comb 初始化**（按行 softmax）：

$$
A_{ij} = \frac{e^{Z_{ij}}}{\sum_{k} e^{Z_{ik}}} + \epsilon
$$

**Sinkhorn-Knopp 迭代**（先列归一化 1 次；其后每轮先行归一化、再列归一化）：

$$
M^{(1)}_{ij} = \frac{A_{ij}}{\sum_{i'} A_{i'j} + \epsilon}
$$

其后第 $t = 2, \ldots, \text{iters}$ 轮：

$$
M_{ij} \leftarrow \frac{M_{ij}}{\sum_{j'} M_{ij'} + \epsilon}
\quad(\text{行归一化}), \qquad
M_{ij} \leftarrow \frac{M_{ij}}{\sum_{i'} M_{i'j} + \epsilon}
\quad(\text{列归一化})
$$

**输出**：`comb = M^{(\text{iters})}`；`iters >= 2` 时共执行 `1 + 2*(iters-1)` 次归一化，最后一步为列归一化。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | mixes | tensor | (B, S, mix_hc) | float32 | ND | 混合 logits，mix_hc=(2+H)*H |
| ATTR(属性) | hc_mult | int | - | - | - | 副本数 H，默认 4，取值 [2, 128] |
| ATTR(属性) | hc_sinkhorn_iters | int | - | - | - | 迭代轮数，默认 20，取值 [1, 32] |
| ATTR(属性) | eps | float | - | - | - | 数值稳定常数，默认 1e-6 |
| OUTPUT(输出) | pre | tensor | (B, S, H) | float32 | ND | 塌缩权重 |
| OUTPUT(输出) | post | tensor | (B, S, H) | float32 | ND | 展开权重，值域 (0, 2) |
| OUTPUT(输出) | comb | tensor | (B, S, H, H) | float32 | ND | 双随机组合矩阵 |

### 3.4 关键输入约束

- **数据类型**: float32（输入输出及中间计算全程 fp32）

- **维度场景**: 3 维输入（B, S, mix_hc）

- **维度取值范围（均为正整数）**:
  - B（batch）: [1, 128]
  - S（seq）: [1, 8192]
  - H = hc_mult: [2, 128]
  - hc_sinkhorn_iters: [1, 32]

- **一致性约束**: mixes 最后一维必须等于 `(2 + H) * H`

- **数值语义**: mixes 为任意实数（混合 logits），不要求非负，建议取值 [-2, 2]

### 3.5 核心属性说明

- **hc_mult**（int，默认 4）: 副本数 H。决定切分位置（前 H 个、中间 H 个、后 H² 个）与 comb 矩阵尺寸
- **hc_sinkhorn_iters**（int，默认 20）: 迭代轮数。注意语义：iters=1 时仅做 1 次列归一化；iters≥2 时执行「列 1 次 + (行→列)×(iters-1)」共 1+2×(iters-1) 次归一化
- **eps**（float，默认 1e-6）: 数值稳定常数。出现在三处：`pre` 的 `+ε`、comb 初值的 `+ε`、以及**每次**归一化分母的 `+ε`

### 3.6 输出严格要求

- **形状约束**: pre.shape == (B, S, H)，post.shape == (B, S, H)，comb.shape == (B, S, H, H)
- **类型约束**: 三个输出均为 float32
- **数值等价**: 与参考实现在合理浮点误差范围内一致
- **中间精度要求**: sigmoid、softmax、归一化全程 FP32
- **值域**: pre ∈ (ε, 1+ε]，post ∈ (0, 2]，comb 逐元素为正
- **迭代语义**: 输出为第 `iters` 轮迭代末的**精确轨迹状态**，不是收敛极限；iters=20 时最后一步为列归一化，故列和 ≈ 1（含 ~1e-6 量级的 ε 系统性残差），行和 ≈ 1（收敛程度取决于数据分布）

### 3.7 特殊值处理规则

- **softmax 数值稳定**: 需减去最大值（等价实现即可）
- **迭代轨迹即定义**: 「列 1 次 + (行→列)×(iters-1)」的次序不可交换、不可合并；输出是固定轮数迭代的状态，而非完全收敛的双随机矩阵（数据分布较宽时，20 轮与收敛极限可有 1e-2 量级差异，属正常）
- **eps 的作用**: 防止分母为 0，并使 pre、comb 严格为正
- **极端值**: logits 可为任意实数；softmax 保证每行至少一项非零，分母 +ε 后恒大于 0，不产生 NaN
- **常数输入**: 全常数 logits 会归一化为均匀矩阵
- **负值 logits**: 输入允许负值，无范围限制

## 四、规则要求

1. **实现方式规则**: 可用向量化、流水、分块、数学等价改写（如缩放向量因子化）等任意加速手段

2. **确定性规则**: 同一输入与属性组合必须产生完全相同的输出（轮数固定、无数据相关分支）

3. **输入不可变**: 算子执行不得修改输入 mixes 的内容；workspace 自行管理

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

**示例1**: 极小规模调试（B=1, S=1, H=2, iters=2, eps=1e-6）

- 输入 mixes: shape=[1, 1, 8]，值 `[0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 0.4, -0.4]`
- 切分: `z_pre=[0.5, -0.5]`，`z_post=[1.0, -1.0]`，`z_comb=[[0.2, -0.2], [0.4, -0.4]]`
- 门控输出: `pre = [0.622460, 0.377542]`，`post = [1.462117, 0.537883]`
- comb 初值（softmax 行 + 1e-6）: `[[0.598689, 0.401313], [0.689976, 0.310027]]`
- 迭代（iters=2，共 3 次归一化：列→行→列）:

| 步骤 | comb |
|---|---|
| 第1次（列） | [[0.464581, 0.564165], [0.535419, 0.435834]] |
| 第2次（行） | [[0.451599, 0.548400], [0.551266, 0.448733]] |
| 第3次（列）= 输出 | [[0.450309, 0.549976], [0.549691, 0.450023]] |

- 输出: `pre shape=[1,1,2]`，`post shape=[1,1,2]`，`comb shape=[1,1,2,2]`
- 校验: comb 列和 = [1.000000, 1.000000]（ε 残差），行和 = [1.000285, 0.999713]（未收敛）
- 对照 iters=20: comb = [[0.450166, 0.549833], [0.549833, 0.450166]]，行和、列和均 ≈ 1.000000

**示例2**: V4 实际场景（B=4, S=4096, H=4, iters=20, eps=1e-6）

- 输入 mixes: shape=[4, 4096, 24]（mix_hc=(2+4)×4=24）
- 输出: pre shape=[4, 4096, 4]，post shape=[4, 4096, 4]，comb shape=[4, 4096, 4, 4]
- 校验性质: comb 逐元素为正、列和 ≈ 1（1e-6 量级 ε 残差）、行和 ≈ 1；comb 为非对称矩阵（组合方向敏感）