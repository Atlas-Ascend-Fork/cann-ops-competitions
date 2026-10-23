# CompressorGrad算子

## 一、赛题背景

CompressorGrad 是 Transformer 长上下文压缩机制 Compressor 算子的反向传播算子。在长序列推理与训练场景中（如 DeepSeek-V4 的 MLA），需将 `cmp_ratio` 个 token 通过分组 softmax 加权融合为 1 个压缩 token，从而显著降低 KV-cache 内存占用与注意力计算量。前向在 `gradEnabled=true` 时导出 `softmax_score`（分组 softmax 结果）与 `kv`（softmax 结果与 kv_state 的 Hadamard 乘积）两个中间结果，作为本算子的输入。

本题要求采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 CompressorGrad 算子。

## 二、算子功能描述

实现的 CompressorGrad 算子需完成以下核心计算：

1. **步骤1 (Hadamard 反向)**: 逐压缩块将上游梯度 `dC` 与前向保存的 `softmax_score`、`kv` 逐元素相乘，得到 `dK` 与 `dS'`。
2. **步骤2 (softmax 反向)**: 对 `dS'` 沿压缩轴（块内 `cmp_ratio` 行）做 softmax 反向，得到 `dZ`。
3. **步骤3 (APE 梯度)**: 按 token 序列位置 `pos % cmp_ratio` 将 `dZ` 行 ScatterAdd 累加到 `dApe`。
4. **步骤4 (matmul 反向)**: 将 `dK`、`dZ` 按压缩块映射回全局 token 行，与权重做矩阵乘法反向，得到 `dX`、`dW_KV`、`dW_Gate`。

## 三、核心定义与约束

### 3.1 参考算子

等价 python 实现核心逻辑：

```python
import numpy as np


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_float32(t):
    """bf16/fp16 → float32（输入可能已经是 float32）"""
    if t.dtype == np.float32:
        return t
    return t.astype(np.float32)


def impl(x, wkv, wgate, d_cmp_kv, softmax_score, kv, cmp_ratio=4):
    """CompressorGrad 算子 golden 实现（coff=1 固定、D=512 固定、BSH 布局）

    返回 (d_x, d_wkv, d_wgate, d_ape)，dtype 与算子输出规格一致。
    """
    orig_dtype = x.dtype
    is_bf16 = ('bfloat16' in str(orig_dtype) or 'bf16' in str(orig_dtype))

    # 统一转 float32 计算（bf16/fp16 精度不足，内部用 FP32）
    x_f32 = _to_float32(x)
    wkv_f32 = _to_float32(wkv)
    wgate_f32 = _to_float32(wgate)
    d_cmp_kv_f32 = _to_float32(d_cmp_kv)
    sm_f32 = softmax_score.astype(np.float32) if softmax_score.dtype != np.float32 else softmax_score
    kv_f32 = kv.astype(np.float32) if kv.dtype != np.float32 else kv

    # 解析维度（coff=1 → D = wkv.shape[0]）
    head_dim = wkv_f32.shape[0]          # 固定 512
    hidden_size = wkv_f32.shape[1]
    batch_size, seq_len, _ = x_f32.shape
    total_tokens = batch_size * seq_len

    # 分配梯度累积缓冲（FP32）
    d_new_kv = np.zeros((total_tokens, head_dim), dtype=np.float32)
    d_new_score = np.zeros((total_tokens, head_dim), dtype=np.float32)
    d_ape = np.zeros((cmp_ratio, head_dim), dtype=np.float32)

    # 逐 batch、逐压缩块反向（coff=1，start_pos=0，seqused=S）
    for batch_idx in range(batch_size):
        # 压缩上限（向下对齐到 cmp_ratio 整数倍）
        compress_limit = (seq_len // cmp_ratio) * cmp_ratio
        batch_seq_idx = 0
        bsh_block_idx = 0
        while batch_seq_idx < seq_len:
            block_seq_start = batch_seq_idx
            block_seq_end = block_seq_start + cmp_ratio
            if block_seq_end > seq_len:
                block_seq_end = seq_len
            batch_seq_idx = block_seq_end

            # 尾块跳过（不足 cmp_ratio，不参与压缩）
            if block_seq_start >= compress_limit:
                break

            # 取当前块上游梯度与中间量
            dC = d_cmp_kv_f32[batch_idx, bsh_block_idx]              # (D,)
            sk = kv_f32[batch_idx, bsh_block_idx]                    # (cmp_ratio, D)
            sm = sm_f32[batch_idx, bsh_block_idx]                    # (cmp_ratio, D)
            bsh_block_idx += 1

            # 反步1: Hadamard 反向
            dC2d = dC[None, :]                                        # (1, D) 广播
            d_kv_block = dC2d * sm                                    # (cmp_ratio, D)
            d_sw = dC2d * sk                                          # (cmp_ratio, D)

            # 反步2: softmax 反向（沿 axis=0，每列独立）
            col_sum = np.sum(sm * d_sw, axis=0, keepdims=True)        # (1, D)
            d_score_block = sm * (d_sw - col_sum)                     # (cmp_ratio, D)

            # 反步3: 路由回 d_new_kv/d_new_score + ScatterAdd → d_ape
            off_s = batch_idx * seq_len + block_seq_start
            off_e = batch_idx * seq_len + block_seq_end
            d_new_kv[off_s:off_e, :] += d_kv_block
            d_new_score[off_s:off_e, :] += d_score_block
            ape_off = block_seq_start % cmp_ratio
            n_pos = block_seq_end - block_seq_start
            d_ape[ape_off:ape_off + n_pos, :] += d_score_block

    # 反步4: matmul 反向
    x_flat = x_f32.reshape(total_tokens, hidden_size)                 # (T, H)

    if is_bf16:
        from ml_dtypes import bfloat16 as bf16_np
        iodtype = bf16_np
    else:
        iodtype = np.float16

    d_new_kv_q = d_new_kv.astype(iodtype).astype(np.float32)
    d_new_score_q = d_new_score.astype(iodtype).astype(np.float32)

    # dX = d_new_kv @ wkv + d_new_score @ wgate  → (T, H)
    d_x_flat = d_new_kv_q @ wkv_f32 + d_new_score_q @ wgate_f32
    # dWkv = d_new_kv.T @ x_flat  → (D, H)
    d_wkv = d_new_kv_q.T @ x_flat
    # dWgate = d_new_score.T @ x_flat → (D, H)
    d_wgate = d_new_score_q.T @ x_flat

    # 恢复 d_x shape + 转回原始 dtype
    d_x = d_x_flat.reshape(batch_size, seq_len, hidden_size)

    if is_bf16:
        from ml_dtypes import bfloat16 as bf16_np
        d_x_out = d_x.astype(bf16_np)
        d_wkv_out = d_wkv.astype(bf16_np)
        d_wgate_out = d_wgate.astype(bf16_np)
    else:
        d_x_out = d_x.astype(np.float16)
        d_wkv_out = d_wkv.astype(np.float16)
        d_wgate_out = d_wgate.astype(np.float16)

    # d_ape 保持 float32
    return d_x_out, d_wkv_out, d_wgate_out, d_ape.astype(np.float32)
```

### 3.2 数学公式

记 `cmp_ratio` 为压缩率，`i` 为压缩块序号，`dC_i / S_i / K_i` 分别为第 `i` 块的上游梯度 / softmax_score / kv，`k` 为块内行序号，`pos` 为 token 序列位置。

**反步1 — Hadamard 反向**:

$$
dK_i = dC_i \odot S_i, \quad dS'_i = dC_i \odot K_i, \quad i=1,\cdots,N_{blocks}
$$

**反步2 — softmax 反向**（沿压缩轴 `cmp_ratio` 求和）:

$$
dZ_i = S_i \odot \left( dS'_i - \sum_{k=1}^{cmp\_ratio} \left( S_i \odot dS'_i \right)_{k,:} \right)
$$

**反步3 — APE 梯度**（ScatterAdd 按 `pos % cmp_ratio`）:

$$
dApe = \text{ScatterAdd}\left( dZ, \; pos \bmod cmp\_ratio \right)
$$

**反步4 — matmul 反向**:

$$
dX = dNewKv \cdot W^{KV} + dNewScore \cdot W^{Gate}
$$

$$
dW^{KV} = dNewKv^{T} \cdot X, \quad dW^{Gate} = dNewScore^{T} \cdot X
$$

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
|------|--------|------|----------|--------------|----------|------|
| 输入 | x | required | [B, S, H] | bfloat16, float16 | ND | 前向输入原始数据 |
| 输入 | wkv | required | [D, H] | bfloat16, float16 | ND | KV 压缩权重（D=512 固定） |
| 输入 | wgate | required | [D, H] | bfloat16, float16 | ND | Gate 压缩权重（D=512 固定） |
| 输入 | d_cmp_kv | required | [B, ceil(S/cmp_ratio), D] | bfloat16, float16 | ND | 上游梯度 dC |
| 输入 | softmax_score | required | [B, ceil(S/cmp_ratio), cmp_ratio, D] | float32, float32 | ND | 前向导出中间量 S |
| 输入 | kv | required | 同 softmax_score | float32, float32 | ND | 前向导出中间量 K |
| 属性 | cmp_ratio | required | - | int | - | 压缩率 [2, 128] |
| 输出 | d_x | required | [B, S, H] | bfloat16, float16 | ND | x 的梯度 |
| 输出 | d_wkv | required | [D, H] | bfloat16, float16 | ND | wkv 的梯度 |
| 输出 | d_wgate | required | [D, H] | bfloat16, float16 | ND | wgate 的梯度 |
| 输出 | d_ape | required | [cmp_ratio, D] | float32, float32 | ND | APE 位置编码梯度 |

### 3.4 关键输入约束

- **数据类型**: x/wkv/wgate/d_cmp_kv/d_x/d_wkv/d_wgate 支持 bfloat16 / float16；softmax_score/kv/d_ape 为 float32。
- **维度场景**: x 为 3 维 [B, S, H]（BSH 布局）。
- **维度取值范围(均为正整数)**:
  - B (Batch Size) ∈ [1, 4096]
  - S (Sequence Length) ∈ [1, 131072]
  - H (Hidden Size) ∈ [1024, 10240]，且必须 512 对齐
  - D (Head Dim) 固定为 512
  - cmp_ratio ∈ [2, 128]
- **其他约束**:
  - 不支持 B、S 为 0 的空 Tensor（所有输入/输出 shapeSize 必须 > 0）。

### 3.5 核心属性说明

- **cmp_ratio**（int，必填）: 压缩率，每 `cmp_ratio` 个 token 压缩为 1 个，取值范围 [2, 128]，与前向一致。

### 3.6 输出严格要求

- d_x shape 与 x 完全一致（保持 3 维 [B, S, H]）。
- d_wkv / d_wgate shape 为 [D, H]，与 wkv / wgate 一致。
- d_ape shape 为 [cmp_ratio, D]，dtype 为 float32。
- 无压缩块（尾块不足 cmp_ratio）时对应 d_x 行输出显式刷 0。

### 3.7 特殊值处理规则

- **尾块跳过**: 不足 `cmp_ratio` 个 token 的尾块不参与压缩，对应 d_x 行输出显式刷 0。

## 四、精度判断规则

- **精度要求**：计算结果需满足以下精度误差要求：
    - float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
    - float16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
    - int32：要求计算结果完全准确，无误差
