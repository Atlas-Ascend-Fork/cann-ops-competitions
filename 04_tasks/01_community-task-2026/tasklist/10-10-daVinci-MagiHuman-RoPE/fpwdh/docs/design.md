# 需求背景（required）

## 需求来源

10月社区任务-daVinci-MagiHuman RoPE 优化任务书（A3）。
目标仓：https://gitcode.com/cann/cann-recipes-infer
RFC/Issue：https://gitcode.com/cann/cann-recipes-infer/issues/442
代码 PR：https://gitcode.com/cann/cann-recipes-infer/merge_requests/1039

## 背景介绍

### daVinci-MagiHuman RoPE 现状

daVinci-MagiHuman 是将文本、参考图像、视频和音频 token 放入统一序列进行联合去噪的单流 Transformer。
仓内 `inference/model/dit/dit_module.py` 已在部分 RoPE 路径使用 `torch_npu.npu_rotary_mul`，但仍同时保留：

1. **双路径并存**：`apply_rotary_emb_torch`（4D 分支已用 `npu_rotary_mul`，但 interleaved/非4D 走小算子）与 `apply_rotary_emb_npu`（预扩展版）语义重复，按 `isinstance(rope, tuple)` 分叉。
2. **每 forward 重复数据准备**：`DiTModel.forward` 每次执行 `tensor_split→cat→unsqueeze→contiguous`；`coords_mapping` 是纯几何函数（与 timestep 无关），32 steps × CFG=2 ≈ 64+ 次前向全部重复同一计算。
3. **partial rotary 的 Slice+Cat**：rotary_dim=96 < head_dim=128，每次融合调用前需 Slice 切片、调用后需 Cat 拼回，多付 2 个 kernel launch。
4. **无约束校验/无显式回退**：直接调 `npu_rotary_mul`，不满足算子约束时行为未定义。

### npu_rotary_mul 算子约束（官方文档）

| 参数 | 约束 |
|---|---|
| input | 3D/4D，dtype fp16/bf16/fp32，D<896 且 D%2==0，B,N<1000 |
| r1/r2 | 末维 == D，范围 [-1,1]，支持 1S1D/BS1D/BSND 广播 |
| rotary_mode | 'half'(NeoX) / 'interleave'(GPT-J) |
| rotate | 可选 2D 矩阵，`out = r1*x + r2*(x@rotate)`，官方推荐用于"末维切分多份时一次调用" |

# 需求分析（required）

## 需求描述

1. 收敛双 RoPE 路径到 `npu_rotary_mul` 单一融合路径
2. 预计算/复用展开的 cos/sin，避免每层重复 `cat/unsqueeze/to`
3. 对不支持的布局/维度/interleaved 保留显式回退，不静默产生错误结果
4. 对 partial rotary 用 rotate 参数单次调用消灭 Slice+Cat
5. 默认使能，不设配置开关
6. 算子级数值对比 + 端到端质量对比 + A3 性能复测

## 需求拆解

1. 约束校验 + 显式回退（5 类场景）
2. cos/sin 预扩展缓存（coords_mapping 纯几何函数，跨 step 复用）
3. partial rotary rotate 路径（seqlen≥2048 自适应切换）
4. padding 缓存（r1/r2 尾部 (1/0) 预计算）
5. 精度/性能/内存三维度自验

# 详细设计（required）

## 算子分析

### 数学公式

```
half 模式: out = r1 * x + r2 * rotate_half(x)
rotate 模式: out = r1 * x + r2 * (x @ rotate)  # rotate = diag(R_rotary_dim, I_tail)
```

### 支持数据类型

float16、bfloat16、float32

### 支持形状

3D/4D，BSND layout，D<896 且 D%2==0，B,N<1000

## 算子实现

### 实现方案

#### host 侧设计（DiTModel._prepare_rotary）：

1. **cos/sin 预扩展缓存**：`coords_mapping` 由 (t,h,w,T,H,W,ref_*) 纯几何生成（`get_coords`），与 timestep 无关 → 同一次生成内 cos/sin 恒定。缓存键 = (seqlen, device) + `torch.equal` 全等闸门（任何输入变化即重建）。
2. **padding 缓存**：cos/sin 尾部预 padding 到全 headdim（cos tail=1, sin tail=0），每生成 1 次（替代每调用 1 次 = 160 次/步 × 64 步）。
3. **rotate 矩阵缓存**：模块级 `_ROTATE_MATRIX_CACHE`，形状/dtype/device 固定，每生成构造 1 次。

#### 调用边界设计（apply_rotary_emb_npu）：

1. **自适应双路径**（实测阈值 ROTATE_MIN_SEQLEN=2048）：
   - seqlen ≥ 2048 且 partial rotary → rotate 路径（块对角矩阵单次调用，消 Slice+Cat）
   - seqlen < 2048 或 full rotary → slice 路径（免费视图切片 + Cat 拼回）
2. **约束校验**（`_npu_rotary_mul_supported`）：逐条校验 dtype/D/B,N/维数，结果按 shape 签名 memoize（热路径 dict 查询）。
3. **显式回退**：不满足约束时 `logging.warning` + 小算子路径（`_rotary_small_op`），行为与基线一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A3 (Ascend910) | √ |

## 算子约束限制

- D ≥ 896 → 显式回退
- D 奇数 → 显式回退
- dtype 非 fp16/bf16/fp32 → 显式回退
- 维数非 3D/4D → 显式回退
- interleaved=True → 小算子路径（不支持融合）
- partial rotary（rotary_dim < headdim）→ rotate 或 slice 路径（自适应）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 融合与小算子输出一致（1 ULP bf16），无 NaN/Inf | 任务书 3.2 |
| 性能标准 | 端到端中位下降超基线波动 + RoPE 路径累计耗时下降 | 任务书 3.3 |

## 实测结果（Atlas A3，base 448×256/5s/32steps/CFG=2/seed=42，每档预热 1+正式 3 次）

### 性能

| 指标 | baseline | 优化后 | 变化 |
|---|---|---|---|
| RoPE 路径累计耗时 | 30398 us | 23520 us | **-22.6%** |
| 端到端中位耗时 | 42.491 s | 41.164 s | **-3.12%（-1.327s，超基线波动 0.188s）** |
| Slice 调用数 | 428 次/步 | 108 次/步 | -75% |
| ConcatD 调用数 | 336 次/步 | 180 次/步 | -47% |

### 精度（8 组真实 shape，含 partial 96/128 / full rotary、rotate/slice 双路径）

- max_abs_err ≤ 1 ULP (bf16 3.125e-2)；大值域 MERE=0、MARE≈1.6e-3；无 NaN/Inf
- partial rotary 尾部逐位保真；多次调用 bitwise 确定性
- 回退路径与基线 bitexact（8 类边界/异常场景）

### 内存

| 指标 | baseline | 优化后 |
|---|---|---|
| HBM 峰值 | 55050 MB | 55050 MB（一致，无回归） |
| 连续 5 次生成 | — | HBM 无增长（无泄漏） |
| rope 缓存增量 | — | <50 MB |

## 兼容性分析

- 8 种模式全部 `MAGI_COMPILE_COMPILE_MODE: NONE`，magi_compile 不编译
- 回退路径行为与基线一致（8 类场景实测 bitexact）
- 不改 `models/davinci-magihuman` 以外代码
