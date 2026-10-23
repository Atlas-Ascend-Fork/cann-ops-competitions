# chunk_scaled_dot_kkt算子

## 一、赛题背景

Qwen3.5模型与Qwen3-next相同，采用了GDN(Gated Delta Networks)结构，在Prefill阶段使用chunkwise算法在块内并行计算注意力，这个过程涉及到WY分解、下三角求解等额外步骤。即将序列长度T划分为 $ceil(T/(chunk\_size ))$，在实际业务场景下，序列长度是变长序列的总长度。

本题要求基于vllm的triton实现的chunk_scaled_dot_kkt算子，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的chunk_scaled_dot_kkt算子。

## 二、算子功能描述

chunk_scaled_dot_kkt算子的计算公式：
$$
\mathrm{strictLower}\big(\mathrm{diag}(beta) \left( \Gamma \odot kk^\mathrm{T} \right)\big)
$$

$$
\Gamma = \exp\left( g\_cumsum[:, \mathrm{None}] - g\_cumsum[\mathrm{None}, :] \right)
$$

其中，strictLower 表示取矩阵的“严格下三角部分”，即只保留行索引大于列索引的元素（$i>j$），而将对角线（$i=j$）和上三角（$i<j$）的元素置为零。

## 三、核心定义与约束

### 3.1 参考实现

```python
import torch
import golden_chunk_scaled_dot_kkt
import numpy as np

def run(k, beta, g_cumsum, chunk_offsets, chunk_size=64):
    k_float = np.asarray(k, dtype=np.float32)
    beta_float = np.asarray(beta, dtype=np.float32)
    g_float = np.asarray(g_cumsum, dtype=np.float32)
    chunk_offsets_list = np.asarray(chunk_offsets, dtype=np.int64).tolist()

    B, T, Hg, K_dim = k_float.shape
    H = beta_float.shape[1]
    BT = chunk_size
    num_repeat = H // Hg

    out = np.zeros((B, H, T, BT), dtype=np.float32)

    num_chunks = len(chunk_offsets_list) - 1
    for c in range(num_chunks):
        t_start = chunk_offsets_list[c]
        t_end = chunk_offsets_list[c + 1]
        L = t_end - t_start

        k_chunk = k_float[:, t_start:t_end].astype(np.float32)
        b_chunk = beta_float[:, :, t_start:t_end].astype(np.float32)
        g_chunk = g_float[:, :, t_start:t_end].astype(np.float32)

        for i_h in range(H):
            kv_h = i_h // num_repeat
            k_h = k_chunk[:, :, kv_h, :]
            b_h = b_chunk[:, i_h, :]
            g_h = g_chunk[:, i_h, :]
            kkt = np.matmul(k_h, k_h.transpose(0, 2, 1))
            gi = g_h[:, :, np.newaxis]
            gj = g_h[:, np.newaxis, :]
            diff = gi - gj
            gamma = np.exp(np.where(diff < 0, diff, float("-inf")))
            gamma_beta = gamma * b_h[:, :, np.newaxis]
            out_mid = gamma_beta * kkt
            out[:, i_h, t_start:t_end, :L] = out_mid

    return out.astype(np.float32)

def generate_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks_per_seq = (seq_lens + chunk_size - 1) // chunk_size

    seq_ids = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_chunks_per_seq)
        ]
    )
    chunk_ids = torch.cat(
        [torch.arange(n, dtype=torch.long) for n in num_chunks_per_seq]
    )
    chunk_indices = torch.stack([seq_ids, chunk_ids], dim=1)
    return chunk_indices.to(cu_seqlens.device)

def generate_chunk_offsets(cu_seqlens, chunk_indices, chunk_size=64):
    num_chunks = chunk_indices.shape[0]
    chunk_offsets = torch.zeros(num_chunks + 1, dtype=torch.int32)

    for i in range(num_chunks):
        seq_id = chunk_indices[i, 0].item()
        chunk_id = chunk_indices[i, 1].item()
        seq_start = cu_seqlens[seq_id].item()
        chunk_start_in_seq = chunk_id * chunk_size
        global_start = seq_start + chunk_start_in_seq
        chunk_offsets[i] = global_start

    chunk_offsets[-1] = cu_seqlens[-1].item()
    return chunk_offsets

testcase = (1, 10016, 8, 2, 128, 64, False, torch.bfloat16, True, torch.float32, [0, 10016])
B, T, H, Hg, K, chunk_size, transpose_beta, dtype, use_g, output_dtype, cu_seqlens_list = testcase

transpose_beta = False

k = torch.randn(B, T, Hg, K, device="cpu", dtype=dtype)
if transpose_beta:
    beta = torch.randn(B, T, H, device="cpu", dtype=dtype)
else:
    beta = torch.randn(B, H, T, device="cpu", dtype=dtype)

g_org = None
if use_g:
    g_org = torch.nn.functional.logsigmoid(
        torch.rand(B, T, H, dtype=torch.float32, device="cpu")
    )

cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device="cpu")

if cu_seqlens is not None:
    cu_seqlens_cpu = cu_seqlens.cpu()
    chunk_indices = generate_chunk_indices(cu_seqlens_cpu, chunk_size)
    chunk_offsets = generate_chunk_offsets(
        cu_seqlens_cpu, chunk_indices, chunk_size
    )
    chunk_indices = chunk_indices.cpu()
    cu_seqlens = cu_seqlens_cpu.cpu()
else:
    chunk_offsets = None
g_cumsum = torch.zeros_like(g_org)
for s, e in zip(chunk_offsets, chunk_offsets[1:]):
    g_cumsum[:, s:e] = g_org[:, s:e].cumsum(dim=1)
g_cumsum = torch.permute(g_cumsum, (0, 2, 1)).contiguous()

golden_A = run(
    k.detach().cpu().to(torch.float32).numpy(),
    beta.detach().cpu().to(torch.float32).numpy(),
    g_cumsum.detach().cpu().to(torch.float32).numpy(),
    chunk_offsets.detach().cpu().numpy(),
    chunk_size)
```


### 3.2 输入输出与属性总览

| 输入/输出    | 参数名        | 类型   | 维度形状      | 支持数据类型 | 数据格式 | 描述                  |
| ------------ | ------------- | ------ | ------------- | ------------ | -------- | --------------------- |
| INPUT(必选)  | k             | tensor | [B, T, Hg, K] | bfloat16     | ND       | Key特征张量           |
| INPUT(必选)  | beta          | tensor | [B, H, T]     | bfloat16     | ND       | 缩放系数              |
| INPUT(必选)  | g_cumsum      | tensor | [B, H, T]     | float32      | ND       | 门控/累积项           |
| INPUT(必选)  | chunk_offsets | tensor | [NT+1]        | int32        | ND       | 变长模式下的chunk索引 |
| INPUT(必选)  | chunk_size    | scalar | 零维          | int32        | ND       | T轴切分的chunk大小    |
| OUTPUT(输出) | A             | tensor | [B, H, T, BT] | float32      | ND       | 输出张量              |

其中，`chunk_offsets[i]`表示第`i`个chunk在T轴上的起始位置，`chunk_offsets[i+1] - chunk_offsets[i]`为该chunk的有效长度 `chunklen`。递增序列，最后一个数值等于T，`chunklen<=chunk_size`。

### 3.3 关键输入约束

- **参数解释**
  - B：批次大小，当前B=1
  - T：序列长度
  - H：多头注意力中的头数
  - Hg：多头注意力分组数
  - BT：分块大小chunk_size，当前BT=64
  - K：每个注意力头的键向量维度
  - NT：序列chunk数
- chunk_offsets: 分块索引列表，如 [0, 64, 128, ..., T]，单调递增且递增大小<= BT

- **维度取值范围(均为正整数)**:
  - B：1
  - T：包含序列个数[1,4]，单个序列长度[1k, 64K]，T长度范围[1k, 256k]
  - H：[1, 128]
  - Hg：[1, 32]，Hg是H的因子
  - K：128和256
  - chunk_offsets：单调递增，举例：T = 300，其中包含序列长度3，cu_seqlens = [0, 100, 240, 300]，则chunk_offsets = [0, 64, 100, 164, 228, 240, 300]
  - chunk_size：64

### 3.4 输出严格要求

- **精度要求**: 要求与参考代码的精度误差满足浮点数精度要求（相对误差不超过1e-5或绝对误差不超过1e-6）


## 四、规则要求

1. 设计算子逻辑，保证算子精度正确
2. 充分发挥系统带宽能力，算子性能更优

## 五： Triton参考

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

@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,  # [H, B, T]
    g_cumsum,  # [H, B, T]
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    bt_stride = B * T
    i_t_i, _ = tl.program_id(0), tl.program_id(1)

    for i_bh in range(B * H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_i * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_t_i
        o_t = tl.arange(0, BT)
        o_t_fp32 = o_t.to(tl.float32)

        p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * Hg + i_h // (H // Hg)) * K, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))

        if USE_G:
            p_g = tl.make_block_ptr(g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A *= safe_exp(b_g_diff)

        b_A *= b_beta[:, None]
        b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, Hg, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, Hg, K = k.shape

    H = beta.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    chunk_scaled_dot_kkt_fwd_kernel[(NT, 1)](
        k=k,
        beta=torch.permute(beta, (2, 0, 1)).contiguous(),
        g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous(),
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=128,
        num_warps=8,
        num_stages=3,
        multibuffer=True,
    )
    return A
```
