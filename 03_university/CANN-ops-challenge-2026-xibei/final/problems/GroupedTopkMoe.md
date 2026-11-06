# GroupedTopkMoe算子

## 一、赛题背景

Mixture-of-Experts（MoE）是当前大语言模型的主流扩展架构，模型由大量专家子网络构成，每个 token 在推理时只激活其中少量专家。MoE 路由（gating）决定每个 token 激活哪些专家以及对应的组合权重。

DeepSeek V3 的 MoE 路由比简单 top-k 多一层「分组受限」结构：全部专家被划分为若干个 group，每个 token 先从 group 中选出一部分，再在选中的 group 内选专家，且每个 token 只能从有限的 group 中选取。这种分层受限选择在保证专家负载均衡的同时，比全量 top-k 更有结构。

本题要求基于 DeepSeek V3 分组受限 MoE 路由（sigmoid 评分 + 分层 top-k）的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 GroupedTopkMoe 算子。

## 二、算子功能描述

实现的 GroupedTopkMoe 算子需完成以下核心计算：

1. **sigmoid 评分**：对 router_logits 逐元素计算 `sigmoid(x)` 评分
2. **分组评分**：将专家按连续区间划分为 n_expert_groups 个组，每组取组内 top-2 评分求和作为组分数
3. **受限选组**：按组分数选出 top n_limited_groups 个组（受每 token 最多选组数约束）
4. **组内选专家**：只在选中组内按选择分数选 top router_top_k 个专家
5. **权重归一化**：对选中专家的 sigmoid 评分归一化，并乘 route_scale

## 三、核心定义与约束

### 3.1 参考算子

PyTorch 中 DeepSeek V3 分组受限路由的等价实现：

```python
import torch


def grouped_topk_moe(router_logits, expert_bias=None, n_expert_groups=8,
                     n_limited_groups=3, router_top_k=8, route_scale=2.5):
    T, E = router_logits.shape
    G = n_expert_groups
    group_size = E // G

    # 1. sigmoid 评分（作为最终权重的基础分数）
    s = torch.sigmoid(router_logits.float())                # (T, E)

    # 2. 选择分数 = sigmoid + bias（bias 仅参与选择，不进入权重）
    sel = s + expert_bias.float().unsqueeze(0) if expert_bias is not None else s

    # 3. 组评分 = 每组内 top-2 求和（专家按连续区间分组）
    sel_group = sel.view(T, G, group_size)
    top2 = torch.sort(sel_group, dim=-1, descending=True, stable=True).values[..., :2]
    group_score = top2.sum(dim=-1)                    # (T, G)

    # 4. 选 top n_limited_groups 个组（分数降序，并列取组索引升序）
    gorder = torch.argsort(group_score, dim=-1, descending=True, stable=True)
    selected = gorder[..., :n_limited_groups]                # (T, G_lim)
    group_mask = torch.zeros_like(group_score, dtype=torch.bool)
    group_mask.scatter_(1, selected, True)

    # 5. 组掩码广播到专家维度，未选中组专家置 -inf
    expert_mask = group_mask.unsqueeze(-1).expand(T, G, group_size).reshape(T, E)
    sel_masked = sel.masked_fill(~expert_mask, float('-inf'))

    # 6. 全局 top-k 选专家（分数降序，并列取专家索引升序）
    ids = torch.argsort(sel_masked, dim=-1, descending=True, stable=True)[..., :router_top_k]   # (T, k)

    # 7. 权重 = 选中专家 sigmoid 分数归一化后乘 route_scale
    w = s.gather(1, ids)
    w = w / w.sum(dim=-1, keepdim=True) * route_scale

    return w, ids.int()
```

### 3.2 数学公式

设输入 router_logits 为 `x`，形状 `(T, E)`（T 为 token 数，E 为专家数）。

**sigmoid 评分**：

$$\text{score}[t, e] = \sigma(x[t, e]) = \frac{1}{1 + e^{-x[t, e]}}$$

**选择分数**（bias 仅参与选择，不进入权重）：

$$\text{sel}[t, e] = \text{score}[t, e] + \text{bias}[e]$$

**组评分**：专家 0..E-1 按连续区间划分为 G 组，第 g 组包含专家 `[g·E/G, (g+1)·E/G)`，组分数取组内 top-2 之和：

$$\text{gscore}[t, g] = \sum_{e \in \text{top2 of group } g} \text{sel}[t, e]$$

**受限选组**：每 token 选组分数最大的 `G_lim` 个组（并列取组索引升序）。

**组内选专家**：仅在选中组内，按 sel 降序选 `k` 个专家（并列取专家索引升序），得 `ids[t, :]`。

**权重归一化**：

$$w[t, j] = \frac{\text{score}[t, \text{ids}[t, j]]}{\sum_{j'} \text{score}[t, \text{ids}[t, j']]} \times \text{route\_scale}$$

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| INPUT(必选) | router_logits | tensor | (T, E) | float16, float32 | ND | 路由 logits |
| INPUT(可选) | expert_bias | tensor | (E,) | float32 | ND | 负载均衡偏置，仅参与选择 |
| ATTR(属性) | n_expert_groups | int | - | - | - | 分组数 G，默认 8 |
| ATTR(属性) | n_limited_groups | int | - | - | - | 每 token 最多选组数 G_lim，默认 3 |
| ATTR(属性) | router_top_k | int | - | - | - | 每 token 选中专家数 k，默认 8 |
| ATTR(属性) | route_scale | float | - | - | - | 归一化后缩放系数，默认 2.5 |
| OUTPUT(输出) | topk_weights | tensor | (T, k) | 与 logits 一致 | ND | 选中专家的归一化权重 |
| OUTPUT(输出) | topk_ids | tensor | (T, k) | int32 | ND | 选中专家的 id |

### 3.4 关键输入约束

- **数据类型**: float16, float32（expert_bias 固定为 float32）

- **维度场景**: 2 维输入（T, E），T 为 token 数，E 为专家数

- **维度取值范围（均为正整数）**:
  - T（token 数）: [1, 262144]
  - E（专家数）: [64, 8192]（需被 G 整除）
  - G（分组数）: [2, 16]
  - G_lim（最多选组数）: [1, G]
  - k（选中专家数）: [1, E]

- **整除约束**: E 必须能被 G 整除（组内专家数 group_size = E // G，专家按连续区间分组）

- **选满约束**: G_lim × (E // G) ≥ k，保证每个 token 在受限组内能选满 k 个专家

- **expert_bias 约束**: 若提供，shape 必须为 (E,)，仅参与选择，不影响最终权重值

### 3.5 核心属性说明

- **n_expert_groups**（int，默认 8）: 专家分组数 G。E 个专家被等分为 G 组，每组 group_size = E // G 个专家，专家按连续区间分组
- **n_limited_groups**（int，默认 3）: 每 token 最多选组数 G_lim。每个 token 只从 G_lim 个组中选专家
- **router_top_k**（int，默认 8）: 每 token 激活的专家数量 k，即最终选中的专家数
- **route_scale**（float，默认 2.5）: 归一化后权重再乘的缩放系数，使选中权重之和为 route_scale

### 3.6 输出严格要求

- **形状约束**: topk_weights.shape == (T, k)，topk_ids.shape == (T, k)
- **类型约束**: topk_weights 数据类型与 router_logits 一致，topk_ids 为 int32
- **数值等价**: 输出与参考实现在合理浮点误差范围内一致
- **中间精度要求**: sigmoid 计算必须使用 FP32 精度，避免大 logits 下溢出
- **权重和约束**: 未缩放前选中权重之和为 1，乘 route_scale 后权重之和为 route_scale

### 3.7 特殊值处理规则

- **sigmoid 数值稳定**: 当 logits 较大时，sigmoid 计算需避免 exp 溢出（如 x≥0 时用 `1/(1+e^{-x})`，x<0 时用 `e^x/(1+e^x)`）
- **负无穷 logits**: logits 为负无穷时，sigmoid 趋近 0，score 趋近 0
- **G_lim=1 边界**: 每 token 只从一个组选专家，退化为「选最优组再组内 top-k」
- **G_lim=G 边界**: 每 token 可从所有组选专家，退化为「全局 top-k」
- **并列值（tie）处理**: 组分数并列时取组索引升序；专家分数并列时取专家索引升序
- **expert_bias 语义**: 仅加到选择分数上影响分组与选专家，最终权重只由 sigmoid 分数决定

## 四、规则要求

1. **实现方式规则**: sigmoid 可用数值稳定近似、top-k 选择可用 bitonic sort、堆、分块等方式，任何与数学定义等价且结果正确的实现均可

2. **赛事规则**: 本赛题的实现与提交必须遵守《赛事规则》，完整内容如下：

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
- int32：要求计算结果完全准确，无误差（topk_ids 需与参考实现完全一致）

## 六、示例说明

**示例1**: 极小规模调试（T=1, E=8, G=4, G_lim=2, k=3, route_scale=2.5）

- 输入 router_logits: shape=[1, 8]
- 假设经 sigmoid 评分后各专家分数为（专家 0..7 分为 4 组，每组 2 个）:

| 组 | 专家 | sigmoid 分数 | 组内 top-2 和 |
|---|---|---|---|
| 0 | e0, e1 | 0.9, 0.1 | 1.0 |
| 1 | e2, e3 | 0.3, 0.8 | 1.1 |
| 2 | e4, e5 | 0.7, 0.2 | 0.9 |
| 3 | e6, e7 | 0.05, 0.04 | 0.09 |

- 选组: top-2 组 = 组1（1.1）、组0（1.0）
- 掩码: 组2、组3 的专家分数置 -inf
- 选专家: 剩余专家 e0=0.9, e1=0.1, e2=0.3, e3=0.8 → top-3 = e0, e3, e2
- 权重: sum=2.0，归一化后 [0.45, 0.40, 0.15]，乘 route_scale=2.5 后 [1.125, 1.0, 0.375]
- 输出: topk_ids=[0, 3, 2]，topk_weights=[1.125, 1.0, 0.375]

**示例2**: DeepSeek V3 实际场景（T=512, E=256, G=8, G_lim=3, k=8, route_scale=2.5）

- 输入 router_logits: shape=[512, 256]
- 属性: n_expert_groups=8, n_limited_groups=3, router_top_k=8, route_scale=2.5
- 输出: topk_weights shape=[512, 8]，topk_ids shape=[512, 8]