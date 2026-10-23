# chunk_gated_delta_rule_fwd_h算子

## 一、赛题背景

Qwen3.5模型与Qwen3-next相同采用GDN(Gated Delta NetWorks)结构，在Prefill阶段使用chunk_wise算法在块内并行计算注意力，这个过程涉及到WY分解、下三角求解等额外步骤。即将序列长度T划分为 $ceil(T/(chunk_size ))$，在实际业务场景下，序列长度是变长序列的总长度。

本题要求基于vllm的triton实现的chunk_gated_delta_rule_fwd_kernel_h_blockdim64算子，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高兼容性、高适配性的chunk_gated_delta_rule_fwd_h算子。

## 二、算子功能描述

单个chunk计算公式如下：
$$
\begin{aligned}
\mathbf{v} &= (\mathbf{u} - \mathbf{w} \cdot \mathbf{h}_t) * \exp(g_{\text{last}} - g) \\
\mathbf{h_{\text{t}}} &= \mathbf{h_{\text{t}}} * \exp(g_{\text{last}}) \\
\mathbf{h}_{t+1} &= \mathbf{h}_t + \mathbf{K}^T \cdot \mathbf{v}
\end{aligned}
$$

## 三、核心定义与约束

### 3.1 参考实现

```
import torch
import math
import random

def run(k, w, u, g, initial_state, cu_seqlens, chunk_indices, chunk_size):    
    k = k.transpose(1,2).contiguous() # 转置后shape：[B,HK, T,K]
    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = u.shape[1], u.shape[3]
    BT = chunk_size  # 固定为64
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) # 等长序列
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT) # 变长序列
    if initial_state is not None:
        initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float32)  # 保持 float32 用于初始化
    else:
        initial_state = None

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=k.dtype)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=u.dtype)
    final_state = torch.zeros((N, HV, K, V), device=k.device, dtype=u.dtype)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None:
            bos = 0
            eos = T
            T_inner = T
            NT_inner = NT
            boh = 0
        else:
            bos = cu_seqlens[n]
            eos = cu_seqlens[n + 1]
            T_inner = eos - bos
            NT_inner = (T_inner + BT - 1) // BT
            boh = chunk_offsets[n]

        for h in range(HV):
            for i in range(NT_inner):
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None:
                    k_sel = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel = w[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel = g[n, h, bos + i * BT : bos + i * BT + actual_len]

                    if initial_state is not None and i == 0:
                        S[n, h, boh+i] = initial_state[n, h]
                else:
                    k_sel = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel = w[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel = u[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    
                    if initial_state is not None and i == 0:
                        S[0, h, boh+i] = initial_state[n, h]

                # if h==1 and i==1:
                #     breakpoint()
                ws = w_sel @ S[0, h, boh+i]
                ws_fp16 = ws.to(torch.float16)
                ws_fp16 = torch.nan_to_num(ws_fp16, nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                v_new = cast_to_float16(u_sel).float() - ws_fp16.float()
                v_new_fp16 = cast_to_float16(v_new)      
                g_last = g_sel[actual_len-1:actual_len]
                v_new_decay = v_new_fp16.float() * cast_to_float16((g_last - g_sel).exp().float())[..., None]
                v_new_decay_fp16 = cast_to_float16(v_new_decay)
                v_new_decay = v_new_decay_fp16.to(u.dtype)
                h_decay = cast_to_float16(S[0, h, boh+i]).float() * cast_to_float16(g_last).exp().float()[..., None]
                h_decay_fp16 = cast_to_float16(h_decay)

                k_vnew = (k_sel.transpose(-1, -2) @ v_new_decay)
                k_vnew_fp16 = k_vnew.to(torch.float16)
                k_vnew_fp16 = torch.nan_to_num(k_vnew_fp16, nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                next_h = h_decay_fp16.to(torch.bfloat16) + k_vnew_fp16.to(torch.bfloat16)

                if cu_seqlens is None:
                    if i != NT_inner-1:
                        S[n, h, boh+i+1] = next_h.to(S.dtype)
                    else:
                        final_state[n, h] = next_h.to(S.dtype)
                    v_new_output[n, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new_fp16.to(u.dtype)
                else:
                    if i != NT_inner-1:
                        S[0, h, boh+i+1] = next_h.to(S.dtype)
                    else:
                        final_state[n, h] = next_h.to(S.dtype)
                    v_new_output[0, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new_fp16.to(u.dtype) 

    return S, v_new_output, final_state

def cast_to_float16(x):
    is_special = torch.isinf(x) | torch.isnan(x)
    x_f16 = x.to(torch.float16)
    clamped = torch.clamp(x_f16, -65504.0, 65504.0)
    return torch.where(is_special, x_f16, clamped)

def cdiv_torch(a, b):
    return (a + b - 1) // b

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv_torch(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1) 

B = 1
T = 64
Hg = 2
H = 16
D = 128
chunk_size = 64
dtype = torch.bfloat16
token_batch = 2
cu_seqlens = torch.tensor([ 0, 61, 64])
k = torch.rand(B, T, Hg, D, dtype=dtype)*2-1
w = (torch.rand(B, H, T, D, dtype=dtype)*2-1) / 128
u = torch.rand(B, H, T, D, dtype=dtype)*2-1
initial_state = torch.randn(len(cu_seqlens)-1, H, D, D, dtype=dtype)
shape_batch, v_head_num, seqlen, _ = u.shape
each_g = -torch.arange(1, T + 1, dtype=torch.float32)
g = each_g.view(1, 1, T).expand(B, H, T)
chunk_offsets = torch.tensor([[0, 0],
        [1, 0]])

cpu_output_0,cpu_output_1,cpu_output_2 = run(
    k, 
    w,
    u,
    g,
    initial_state,
    cu_seqlens,
    chunk_offsets, 64)
```

### 3.2 输入输出与属性总览

| 参数名        | 输入 | 描述                                                         | 数据类型 | 数据格式 | 维度 | Shape                                |
| ------------- | ---: | ------------------------------------------------------------ | -------- | -------- | ---- | ------------------------------------ |
| k             | 输入 | attention结构中的键向量(key)                                 | bfloat16 | ND       | 4维  | [B,T,HK,K]                           |
| w             | 输入 | w和u是WY变换的中间产出结果。w表示有效键投影 (Effective Key Projection)，存储了经过门控g调制的键k的聚合信息。 | bfloat16 | ND       | 4维  | [B,HV,T,V]                           |
| u             | 输入 | u表示修正后的值投影 (Corrected Value Projection)，<br/>存储了值v中需要用于更新隐藏状态的新信息部分。 | bfloat16 | ND       | 4维  | [B,HV,T,V]                           |
| g             | 输入 | 门控参数，包含每个chunk的累积门控衰减值，控制前一步记忆状态的衰减率。T维度chunk内严格递减，均为负数。 | Float32  | ND       | 3维  | [B,HV,T]                             |
| initial_state | 输入 | 每个序列的初始隐状态，是后续递推每个chunk的隐状态的初始起点  | bfloat16 | ND       | 4维  | [token_batch,HV,K,V]                 |
| cu_seqlens    | 输入 | 变长序列场景下，记录每个序列的累积长度，用于将扁平化的 q/k/v张量划分回独立序列 | Int64    | ND       |      |                                      |
| chunk_indices | 输入 | chunk索引表，用于变长序列场景下，快速定位每个chunk所属的序列编号及其在序列内的chunk序号。 | int64    | ND       | 2维  | [NT,2]                               |
| chunk_size    | 输入 | 64                                                           | int64    |          |      |                                      |
| h             | 输出 | 每个chunk的隐状态                                            | bfloat16 | ND       | 5维  | [B,HV,NT,K,V]  NT：一共多少个chunk数 |
| v             | 输出 | 输出经过g衰减和w/u修正后的值向量                             | bfloat16 | ND       | 4维  | [B,HV,T,V]                           |
| final_state   | 输出 | 每个序列经过整个序列递推更新后，在末端位置对应的最终隐状态   | bfloat16 | ND       | 4维  | [token_batch,HV,K,V]                 |

### 3.3 关键输入约束

- **参数解释**
  - B：批次大小，当前B=1
  - T：序列长度，在变长序列场景下T为多个序列长度之和，token_batch为子序列个数
  - HV：多头注意力中的头数
  - HK：多头注意力分组
  - BT：分块大小，当前BT=64
  - K/V：每个注意力头的维度
  - NT：所有序列包含的chunk数
  - chunk_indices : 第一列为当前chunk所属序列编号，第二列为当前chunk在所属序列中的chunk编号
- **维度取值范围(均为正整数)**:
  - B：1
  - T：包含序列个数[1,4], 单个序列长度[1k, 64K]，T长度范围[1k, 256k]，即token_batch范围为[1,4]
  - HV：[1, 128]
  - HK：[1, 32]
  - K/V：[32,256]； K与V相等

### 3.3 输出严格要求

- **精度要求**: 要求与参考代码的精度误差满足浮点数精度要求（相对误差不超过1e-5或绝对误差不超过1e-6）

## 四、规则要求

1. 设计算子逻辑，保证算子精度正确
2. 充分发挥系统带宽能力，算子性能更优

## 五、 Triton参考

注：triton的实现中仅供参考，具体以 3.1参考实现为主 

```
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

def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.cat([
        torch.tensor([0], dtype=torch.long, device=cu_seqlens.device),
        num_chunks.cumsum(dim=0)
    ])
    return torch.Tensor(chunk_offsets).to(cu_seqlens.dtype).npu()

@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "H", "Hg", "K", "V"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
    b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)
    # create b_hupd_bv1 and b_hupd_bv2

    v_start1 = 0
    v_start2 = 64

    offs_k = tl.arange(0, 128)[:, None]
    offs_v1 = v_start1 + tl.arange(0, 64)[None, :]
    offs_v2 = v_start2 + tl.arange(0, 64)[None, :]
    mask_kv1 = (offs_k < K) & (offs_v1 < V)
    mask_kv2 = (offs_k < K) & (offs_v2 < V)

    # load initial state
    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv1 = h0_ptr + offs_k * V + offs_v1 * 1
        b_h1_bv1 += tl.load(ptr_h0_bv1, mask=mask_kv1, other=0.0).to(tl.float32)

        ptr_h0_bv2 = h0_ptr + offs_k * V + offs_v2 * 1
        b_h1_bv2 += tl.load(ptr_h0_bv2, mask=mask_kv2, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V

        p_h1_bv1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_h1_bv1, b_h1_bv1.to(p_h1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_h1_bv2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_h1_bv2, b_h1_bv2.to(p_h1_bv2.dtype.element_ty), boundary_check=(0, 1))

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        offs_k_wv = tl.arange(0, 128)[None, :]
        mask_w = (offs_t_wv < T) & (offs_k_wv < K)

        w_base = w + bos * H * K + i_h * K
        ptr_w = w_base + offs_t_wv * stride_w + offs_k_wv * 1
        b_w = tl.load(ptr_w, mask=mask_w, other=0.0)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        offs_t_v = (i_t * BT + tl.arange(0, BT))[:, None]
        mask_v1 = (offs_t_v < T) & (offs_v1 < V)

        v_base = v + bos * H * V + i_h * V
        ptr_v1 = v_base + offs_t_v * stride_v + offs_v1 * 1
        b_v1 = tl.load(ptr_v1, mask=mask_v1, other=0.0)
        b_v_new1 = b_v1.to(tl.float32)
        b_v_new1 -= tl.dot(b_w, b_h1_bv1.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new1 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start1), (BT, 64), (1, 0))
            tl.store(p_v_new1, b_v_new1.to(p_v_new1.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new1 = b_v_new1 * b_g[:, None]
            b_h1_bv1 = b_h1_bv1 * b_g_last

        b_v_new1 = b_v_new1.to(k.dtype.element_ty)
        b_h1_bv1 += tl.dot(b_k, b_v_new1)

        mask_v2 = (offs_t_v < T) & (offs_v2 < V)
        ptr_v2 = v_base + offs_t_v * stride_v + offs_v2 * 1
        b_v2 = tl.load(ptr_v2, mask=mask_v2, other=0.0)
        b_v_new2 = b_v2.to(tl.float32)
        b_v_new2 -= tl.dot(b_w, b_h1_bv2.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new2 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start2), (BT, 64), (1, 0))
            tl.store(p_v_new2, b_v_new2.to(p_v_new2.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new2 = b_v_new2 * b_g[:, None]
            b_h1_bv2 = b_h1_bv2 * b_g_last

        b_v_new2 = b_v_new2.to(k.dtype.element_ty)
        b_h1_bv2 += tl.dot(b_k, b_v_new2)

    # epilogue
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V

        p_ht1_bv1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_ht1_bv1, b_h1_bv1.to(p_ht1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_ht1_bv2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_ht1_bv2, b_h1_bv2.to(p_ht1_bv2.dtype.element_ty), boundary_check=(0, 1))

# k shape [B, T, HK, K]; w shape [B, T, HV, K]; u shape: [B, T, HV, V] 与3.1实现的shape需要注意区别（其他输入相同）
# 输出中的 h和 v_new 需要 dim1与dim2交换 （dim从0开始计数）才与3.1实现匹配
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_offsets is None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            chunk_offsets,
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    h_update = k.new_empty(B, NT, H, K, K)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32)

    v_new = torch.empty_like(u) if save_new_value else None
    # g = g.transpose(1, 2).contiguous()

    def grid(meta):
        return (1, N * H)
    print(chunk_indices.shape)
    print(chunk_offsets.shape)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=h_update,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state

```
