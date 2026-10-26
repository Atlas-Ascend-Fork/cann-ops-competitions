# TurboQuantAttentionScore 算子设计文档

> 目标平台：Atlas 800T A2（开发自测于 Atlas 910B / dav_c220 / arch22）；CANN 9.1.0；目标仓 ops-transformer `experimental/attention`。
> 本文档反映**实际实现现状**：Decode 纯 Vector 路径已完成并 5-case 精度全绿；Prefill 的 Cube 混合路径为设计方案（当前未实现，纯 Vector prefill 实测远不达标，见 §7）。

## 1 需求背景（required）

### 1.1 需求来源
华为昇腾 CANN 社区任务（9 月）——`turbo_quant_attention_score` 算子开发，交付合入 ops-transformer 开源仓。

### 1.2 背景介绍
TurboQuant 是一种 KV Cache 量化方案：Key 向量经随机旋转（Hadamard/rotation matrix）后做 Lloyd-Max 主量化（mse_bits，默认 3-bit）+ QJL 残差量化（1-bit 符号）。本算子在**旋转空间直接消费编码后的 Paged Key cache** 计算 attention score，无需把 Key 完整反量化回原空间，长上下文下显著降低显存与带宽。

score 语义（`golden.py` 权威）：

```
score[t,h,l] = norm[l]·⟨Hq[t,h], CENT[idx[l]]⟩ + (sqrt(pi/2)/HEAD_DIM)·gamma[l]·⟨S(Hq)[t,h], sign[l]⟩
```

其中 `HEAD_DIM=128`，`kv_head = q_head / gqa_ratio`（GQA，本任务 q_heads=32, kv_heads=8, ratio=4）。

### 1.3 功能分析
**输入**：
- `query_rotated` (bf16) `[T, 32, 128]`：预旋转 query Hq
- `query_qjl` (bf16) `[T, 32, 128]`：旋转 query 的 QJL 投影 S(Hq)
- `key_cache_idx` (u8) `[NB, 128, 8, ceil(128*mse_bits/8)]`：主量化 bit-packed（3-bit LSB-first，8 维拼 24-bit 小端字）
- `key_cache_qjl` (u8) `[NB, 128, 8, 16]`：QJL 符号（1-bit/维）
- `key_cache_norm` (bf16) `[NB, 128, 8]`：Key L2 范数
- `key_cache_gamma` (bf16) `[NB, 128, 8]`：残差范数
- `block_table` (i32) `[B, max_blocks]`：Paged 逻辑→物理块映射
- `seq_lens` (i32) `[B]`：各请求 KV 长度

**输出**：`attn_scores` (fp32) `[T, 32, max_kv_len]`，尾位（l ≥ seq_len）填 `-inf` 哨兵。

**属性**：`mse_bits` (int, 2/3/4)。

## 2 需求分析（required）

1. **功能**：按 `mse_bits` 解包主量化 → 查 Lloyd-Max 质心表 → `norm·⟨Hq, dequant⟩`；解包 QJL 符号 → `coeff·gamma·⟨S(Hq), sign⟩`；相加，尾位 -inf。
2. **Paged KV Cache**：`block_table` 逻辑→物理块映射，支持乱序块 + 变长。
3. **GQA**：q head 按 group 映射到 kv head（h → kvh = h/ratio）。
4. **精度**：5 个官方 case 逐元素误差 ≤ 1e-3，零失败点。
5. **性能**：≤ 全精度 baseline × 1.2。
6. **工程化**：aclnn 两段式接口（原型/Tiling/Kernel/Host）+ ops-transformer PR + 设计文档。

## 3 总体架构

任务按 **(batch b, kv_head g, 64-key tile)** 划分（一个物理 block 128 key 拆成 2 个 64-key tile）。单 kernel 内：

1. **CopyIn**：从 GM 读取本 tile 的 idx/qjl 字节、norm/gamma。
2. **Unpack**：bit-packing 反量化 → 质心/符号向量。
3. **ComputeHead**：对该 tile 覆盖的每个 query token、每个 gqa 组内 head，计算两路点积并合成 score。
4. **CopyOut**：写 `attn_scores` 对应 64 列。

**Decode（T=B）**：纯 Vector AIV（已实现）。**Prefill（B=1, T=512）**：需 MIX（AIV 解包 + AIC Cube GEMM，见 §7）。

## 4 详细设计（required）

### 4.1 Host 侧（Tiling）
`turbo_quant_attention_score_tiling.cpp`：
- 从 `query_rotated`/`key_cache_idx`/`block_table` 形状推 `numTokens/numQHeads/numKvHeads/gqaRatio/batch/maxBlocksPerSeq/idxBytes/qjlBytes`；`maxKvLen = maxBlocksPerSeq*128`。
- 校验：仅支持 `batch==1`（prefill）或 `T==B`（decode）；`numBlocks ≥ batch*maxBlocksPerSeq`。
- 分核：`totalTasks = batch*numKvHeads*tilesPerBatch`（tilesPerBatch = maxBlocksPerSeq*2），`usedCoreNum = min(totalTasks, GetCoreNumAiv())`。
- **`COEFF_SCALE = sqrt(pi/2)/128 = 0.009791516697777345F`**（关键常量，曾误设致 4–14% 残差，已修）。
- TilingKey：按 `mse_bits`（2/3/4）与 norm/gamma dtype 编码。

### 4.2 Kernel 侧（`turbo_quant_attention_score_vector.h`）

#### 4.2.1 数据读取：GM 直读策略（关键设计决策）
dav_c220 上对 **非 32B 对齐的 strided `DataCopy`/`DataCopyPad`** 落位不可靠（实测 blockLen=48/2 的分块拷贝会把行放错、末行读 0）。故采用：
- idx/qjl **直接 `GlobalTensor::GetValue` 逐字节读 GM**；
- norm/gamma 直接 `GlobalTensor<uint16_t>::GetValue` 逐 key 读，`union{uint32;float}` 位 `<<16` **bitcast** 成 f32（bf16 = f32 高 16 位，精确零扩展），绕开 bisheng **禁止 bf16/整型标量↔浮点强转**；
- `ReadGmScalar`（读 block_table/seq_lens 单个 int32）必须 **32B 对齐批量拷 8 个 int32 再按偏移取值**（单 4B 非对齐 DataCopy 会静默返回 0）。

#### 4.2.2 解包（Unpack）
3-bit LSB-first：每 3 字节拼 24-bit 小端字 `w = b0 | b1<<8 | b2<<16`，8 个相位各取 3 bit `code=(w>>3p)&7`。质心查表用**矢量 LUT `Gather`**（src=8 元质心表，idx=code*4 字节偏移），同族 `turbo_quant_sparse_flash_attention` 反量化范式。QJL 符号同理（`SIGN2` 2 元 LUT，bit=1→+1）。

#### 4.2.3 点积（ComputeHead，已矢量化）
`decMain/decSign` 为 row-major `[64][128]` fp32。点积 `dotM[k]=Σ_d decM[k][d]·qf[d]`：逐行
`Mul(128) → Add 折半(128→64) → WholeReduceSum(64,1,1,1,8)(64→1)`。
**关键**：解包/dot 内标量 `SetValue` 写 UB 后、矢量指令读同缓冲前必须 `PipeBarrier<PIPE_ALL>`（否则每 tile 末元素读到旧值）。
最终 `Mul(score,normF,dotM)` + `Mul(tmpV,gammaF,dotQ)` + `Axpy(score,tmpV,coeff_)`。

#### 4.2.4 尾位
`valid<64` 时对 l≥seq_len 写 FP32-min/-inf 哨兵。

### 4.3 支持的 arch22（dav_c220）原语约束（实测）
- `Scatter` 为空桩（`REPORT_NOT_SUPPORT`，编译过但不写）；`Interleave`/通用 `Transpose` 不可用 → 不能用其做 phase-major→row-major 转置。
- `Gather`：idx 为字节偏移（u32），**大源（≥KB）scatter-gather 会 UB 越界**；小源 LUT gather 安全。
- `vshr`（ShiftRight）为算术移位；`vsub` 不支持 uint32（需 int32 或 `ReinterpretCast`）。
- 标量整型↔浮点强转、bf16 标量 cast 均被禁 → 用 `Cast` 矢量 intrinsic 或 union bitcast。

## 5 精度验证（已达成）
跨机 C++ aclnn harness（本机 numpy golden + 设备 C++ runner 逐元素比对，阈 abs/rel ≤1e-3）：

| case | 比对规模 | 结果 | maxRel |
| --- | --- | --- | --- |
| gqa_decode_kv4k | 131,072 | PASSED | 4.5e-06 |
| gqa_decode_kv64k | 2,097,152 | PASSED | 7.0e-06 |
| gqa_decode_kv128k | 4,194,304 | PASSED | 7.7e-06 |
| gqa_prefill_kv4k | slice(T=512) | PASSED | 5.9e-06 |
| gqa_prefill_kv32k | slice(T=512) | PASSED | 6.5e-06 |

5/5 全绿，误差 ~1e-5（100× 余量）。三处关键数值修复：① COEFF 常量；② norm/gamma GM 直读+bitcast；③ 标量写→矢量读 barrier。

## 6 性能现状与剖析
ACL event 纯算子计时（本机 910B）：

| case | V0 纯标量 | 当前(M4 两步矢量化) | 上限(×1.2) |
| --- | --- | --- | --- |
| decode kv4k | 9099 µs | ~1000 µs | 152 |
| decode kv64k | 142,722 µs | ~15,500 µs | 2,786 |

已做优化：点积矢量化（~3×）、解包 LUT Gather 化（~3×，累计 ~9×）。**成本剖析**（kv4k 996µs）：主路字节提取~288 + 点积~376 + Gather/符号/scale/输出~332，**分散无单一银弹**。距上限仍 ~6.6×。

## 7 Prefill 与后续性能方案（设计，未实现）
纯 Vector 的 prefill（T=512，逐 token×head 点积）实测远超预算（社区前人 48ms vs 预算 0.75ms）。**达标方案 = MIX**：AIV 解包产出量化 Key 矩阵写 GM，AIC 单 Cube GEMM `A=[Hq‖Hq‖S(Hq)] × B^T`（B 为 fp16 拼接，双 fp16 保精度 K=384），acc fp32 fixpipe 直写 attn_scores。精度红线：单 fp16 GEMM 误差 ~2e-3 不可用，双 fp16 ~4e-5 可用。

**Decode 达标路径 = 整体 V1 重写**：批量 gather（跨多 key 一次 Gather）+ 矢量化归约（降 64 行串行 WholeReduceSum/barrier）+ 双缓冲流水（MTE2/计算重叠）。

## 8 支持的硬件
| 芯片 | 支持 |
| --- | --- |
| Atlas 800T A2（910B，dav_c220/arch22） | √（已开发自测） |

## 9 约束与限制
- 仅 `batch==1`（prefill）或 `T==B`（decode）；`numBlocks ≥ batch*maxBlocksPerSeq`。
- norm/gamma 定为 bf16（op.json 权威）。
- GQA：`numQHeads % numKvHeads == 0`。
- 当前性能未达任务基线×1.2（见 §6/§7），需后续 V1 重写与 Prefill Cube 专项。
