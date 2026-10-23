# recompute_w_u算子

## 一、赛题背景

Qwen3.5模型与Qwen3-next相同采用GDN(Gated Delta NetWorks)结构，在Prefill阶段使用chunk_wise算法在块内并行计算注意力，这个过程涉及到WY分解、下三角求解等额外步骤。即将序列长度T划分为 $ceil(T/(chunk_size ))$，在实际业务场景下，序列长度是变长序列的总长度。

本题要求基于vllm的triton实现的recompute_w_u算子，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的recompute_w_u算子。

## 二、算子功能描述

recompute_w_u算子的计算公式：
$$ u = A · diag(β) · v $$
$$ u = A · diag(β) · exp(g) · k $$

## 三、核心定义与约束

### 3.1 参考实现

```
import torch
import numpy as np
import random
import math

import numpy as np
from ml_dtypes import bfloat16

def _to_bfloat16_bits(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _to_bfloat16(value):
    bf16_bits = _to_bfloat16_bits(value)
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def run(k, v, A, beta, g, global_chunk_offsets):
    A_float = np.asarray(A, dtype=np.float32)
    k_float = np.asarray(k, dtype=np.float32)
    v_float = np.asarray(v, dtype=np.float32)
    beta_float = np.asarray(beta, dtype=np.float32)
    g_float = np.asarray(g, dtype=np.float32)
    chunk_offsets = np.asarray(global_chunk_offsets, dtype=np.int64).tolist()

    B_dim, H, T, BT = A_float.shape
    Hg = k_float.shape[2]
    K = k_float.shape[3]
    V = v_float.shape[-1]
    group_size = H // Hg

    A_p = A_float
    k_p = k_float.transpose(0, 2, 1, 3)
    v_p = v_float.transpose(0, 2, 1, 3)
    beta_p = beta_float
    g_p = g_float

    u = np.zeros((B_dim, H, T, V), dtype=np.float32)
    w = np.zeros((B_dim, H, T, K), dtype=np.float32)

    for i in range(1, len(chunk_offsets)):
        start = chunk_offsets[i - 1]
        end = chunk_offsets[i]
        current_chunk_len = end - start

        for h in range(H):
            hg = h // group_size

            A_chunk = A_p[:, h, start:end, :current_chunk_len]
            beta_chunk = beta_p[:, h, start:end]
            g_chunk = g_p[:, h, start:end]
            v_chunk = v_p[:, h, start:end, :]
            k_chunk = k_p[:, hg, start:end, :]

            u_scaled = _to_bfloat16(A_chunk * beta_chunk[:, np.newaxis, :])
            u_chunk = np.matmul(u_scaled, v_chunk)
            u[:, h, start:end, :] = _to_bfloat16(u_chunk)

            w_scaled = _to_bfloat16(A_chunk * beta_chunk[:, np.newaxis, :] * np.exp(g_chunk)[:, np.newaxis, :])
            w_chunk = np.matmul(w_scaled, k_chunk)
            w[:, h, start:end, :] = _to_bfloat16(w_chunk)

    return w.astype(bfloat16), u.astype(bfloat16)


B = 1
T = 10016
H = 8
Chunk_size = 64
D = 128
H_g = 2

k = torch.randn((B, T, H_g, D), dtype=torch.bfloat16)  #  k未转置
v = torch.randn((B, T, H, D), dtype=torch.bfloat16)  #  v未转置
A = torch.randn((B, H, T, Chunk_size), dtype=torch.float32)
g = torch.randn((B, H, T), dtype=torch.float32)

transpose_beta = False
if transpose_beta:
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
else:
    beta = torch.randn((B, H, T), dtype=torch.bfloat16)

chunk_offsets = torch.tensor([0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024,1088,1152,1216,1280,1344,1357,1421,1485,1549,1613,1677,1741,1805,1869,1933,1997,2061,2125,2189,2253,2317,2381,2445,2509,2573,2637,2701,2765,2829,2893,2957,3021,3085,3149,3213,3277,3341,3405,3469,3533,3597,3661,3725,3789,3853,3917,3921,3985,4049,4113,4177,4241,4305,4369,4433,4497,4561,4625,4689,4753,4817,4881,4945,5009,5073,5137,5201,5265,5329,5393,5457,5521,5585,5649,5713,5777,5841,5905,5969,6033,6097,6161,6225,6289,6353,6417,6481,6545,6609,6673,6737,6801,6865,6929,6993,7057,7121,7185,7249,7313,7377,7441,7505,7569,7633,7697,7761,7825,7889,7953,8017,8081,8145,8209,8273,8337,8401,8465,8529,8593,8657,8721,8785,8849,8913,8977,9041,9105,9169,9233,9297,9361,9425,9489,9553,9617,9681,9745,9809,9873,9937,10001,10016] , dtype=torch.int32)

w_golden, u_golden = run(
    k.detach().cpu().to(torch.float32).numpy(),
    v.detach().cpu().to(torch.float32).numpy(),
    A.detach().cpu().numpy(),
    beta.detach().cpu().to(torch.float32).numpy(),
    g.detach().cpu().numpy(),
    chunk_offsets.detach().cpu().numpy(),
)
```


### 3.2 输入输出与属性总览

| 类型         | 参数名        | 类型   | 维度形状      | 支持数据类型 | 数据格式 | 备注                                    |
| ------------ | ------------- | ------ | ------------- | ------------ | -------- | --------------------------------------- |
| INPUT(必选)  | A             | tensor | [B, H, T, BT] | bfloat16     | ND       | 注意力权重矩阵                          |
| INPUT(必选)  | k             | tensor | [B,T,Hg,D]    | bfloat16     | ND       | Key 特征张量                            |
| INPUT(必选)  | v             | tensor | [B,T,H,D]     | bfloat16     | ND       | Value 特征张量                          |
| INPUT(必选)  | beta          | tensor | [B, H, T]     | bfloat16     | ND       | 缩放系数                                |
| INPUT(必选)  | g             | tensor | [B, H, T]     | float        | ND       | 门控/累积项                             |
| INPUT(必选)  | chunk_offsets | tensor | [len(NT) + 1] | int32        | ND       | 分块索引列表, NT表示T内的chunk_size个数 |
| OUTPUT(输出) | w             | tensor | [B,H,T,D]     | bfloat16     | ND       | k的重计算结果                           |
| OUTPUT(输出) | u             | tensor | [B,H,T,D]     | bfloat16     | ND       | v的重计算结果                           |

### 3.2 关键输入约束

- **参数解释**
  - B：批次大小，当前B=1
  - T：序列长度，在变长序列场景下T为多个序列长度之和
  - H：多头注意力中的头数
  - Hg：多头注意力分组
  - BT：分块大小，当前BT=64
  - D：每个注意力头的维度
  - chunk_offsets: 分块索引列表，如 [0, 64, 128, ..., T],单调递增且递增大小<= BT

- **维度取值范围(均为正整数)**:
  - B：1
  - T：包含序列个数[1,4], 单个序列长度[1k, 64K]，T长度范围[1k, 256k]
  - H：[1, 128]
  - Hg：[1, 32]
  - D：[32,256]
  - chunk_offsets：单调递增，举例：T = 300，其中包含序列长度3，cu_seqlens = [0, 100, 240, 300], 则chunk_offsets = [0, 64, 100, 164, 228, 240, 300]

### 3.3 输出严格要求

- **精度要求**: 要求与参考代码的精度误差满足浮点数精度要求（相对误差不超过1e-5或绝对误差不超过1e-6）


## 四、规则要求

1. 设计算子逻辑，保证算子精度正确
2. 充分发挥系统带宽能力，算子性能更优

## 五、 Triton参考

注：triton的实现中仅供参考，具体以 3.1参考实现为主

```
import torch
from vllm.triton_utils import tl, triton
import math

def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    token_batch = len(cu_seqlens) - 1
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    chunk_indices = []
    for tb in range(token_batch):
        curr_chunks = math.ceil((cu_seqlens[tb + 1] - cu_seqlens[tb]) / chunk_size)
        num_chunks += curr_chunks
        for c in range(curr_chunks):
            chunk_indices.append([tb, c])
    return torch.Tensor(chunk_indices).to(cu_seqlens.dtype).npu()


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "H", "Hg", "K", "V"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_max = T
    i_t_o = tl.program_id(0)

    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_o * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_g = g + bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)

            ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)

            ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            ptr_k = k + (bos * Hg + i_h // (H // Hg)) * K + offs_t_2d * (Hg * K) + offs_k * 1
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None] * b_g[:, None]
            b_w = tl.dot(b_A, b_kb)

            ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)

# beta 和 g的 输入shape为 [B, T, H] 与3.1不同，需要转置
# 输出 w: [B, T, H, K]; u: [B, T, H, V] 和3.1之间不同，需要转置
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 64
    BV = 64

    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    beta = beta.transpose(1, 2).contiguous()
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()
    recompute_w_u_fwd_kernel[(NT, B)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    return w, u

```
