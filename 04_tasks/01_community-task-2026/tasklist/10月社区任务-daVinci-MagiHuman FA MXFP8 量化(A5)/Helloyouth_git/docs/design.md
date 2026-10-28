# daVinci-MagiHuman FA MXFP8 量化设计文档

> 任务来源：昇腾社区 2026 年 10 月社区任务「daVinci-MagiHuman FA MXFP8 量化任务书（A5）」
> 代码合入仓：https://gitcode.com/cann/cann-recipes-infer
> 设计文档提交仓：cann-ops-competitions（本文档）

---

# 需求背景（required）

## 需求来源

10 月社区任务 A5：在 Ascend 950 上为 daVinci-MagiHuman 视频生成模型的 Attention 路径接入 MXFP8 量化，使目标 Attention 真实执行 MXFP8 计算，在保持生成质量的前提下取得端到端性能提升与峰值显存下降。

- 任务书：`daVinci-MagiHuman FA MXFP8 量化任务书(A5).md`（本文唯一语义真值源；配套 `test_cases/README.md` 是自验执行口径，冲突时以任务书为准）
- 设计文档提交路径：`04_tasks/01_community-task-2026/tasklist/10月社区任务-daVinci-MagiHuman FA MXFP8 量化(A5)/Helloyouth_git/docs/design.md`
- 代码合入仓：`https://gitcode.com/cann/cann-recipes-infer`（本文简称 infer 仓）
  - 模型侧改动：`models/davinci-magihuman/`（配置 + `inference/model/dit/dit_module.py` 等）
  - 公共模块扩展：`module/fa_quant/`（仅在超出既有契约时，且保持全部现有调用方兼容）
- 复用能力：infer 仓 `module/fa_quant/fa_quant.py` 的 `npu_mxfp8_quant` / `npu_mxfp8_attn`（Q/K/V 动态 MXFP8 量化 + `npu_fused_infer_attention_score_v2` 封装）
- 已有接入样例：`models/hunyuan-video/hyvideo/modules/attention.py`（`mode="mxfp8"`）与 `models/hunyuan-video/config/single_fp8.yaml`（`fa-mxfp8: true`）
- 工程模式：模型 MXFP8 FA 接入 + 必要的公共封装扩展，**不新建独立 kernel 工程**，不新建平行公共目录

## 背景介绍

### daVinci-MagiHuman Attention 路径现状分析

daVinci-MagiHuman 是多模态（视频/音频/文本）统一序列的 DiT 视频生成模型。Attention 相关实现集中在 `models/davinci-magihuman/inference/model/dit/dit_module.py`，当前全部走 BF16 `torch_npu.npu_fused_infer_attention_score`（FA v1），共三条执行路径：

| 路径 | 注册名 / 触发条件 | 输入 layout | 稀疏 / 变长 | 现状实现 |
| --- | --- | --- | --- | --- |
| `flash_attn_func` | `infra::flash_attn_func`；`use_local_attn=False` 层的标准 Attention | BSND `(b, s, n, d)` | 非稀疏、定长 | FA v1，`sparse_mode=0`，`scale=1/sqrt(d)`，输出转回原 dtype |
| `flex_flash_attn_func` | `infra::flex_flash_attn_func`；`use_local_attn=True` 层的局部/变长注意力 | 段内 BNSD（由 `(t, n, d)` permute 而来） | `q_ranges`/`k_ranges` 描述的局部块 | 无重叠 q 段切分（`_split_q_range_with_no_overlap`）后逐段调 FA v1（`pre_tokens=65535, next_tokens=65535`，段内全注意力），同一 q 段命中多个 k 段时按 LSE 加权合并 |
| `flash_attn_with_cp` / `flex_flash_attn_with_cp` | context-parallel 包装 | 同上 | 同上 | cp_size>1 时先 all-to-all（头↔序列重排），再调用上两个函数；`cp_size=1` 时直通 |

模型侧关键事实：

1. Q/K/V 在进入 Attention 前已完成 per-head RMSNorm（`q_norm`/`k_norm`）与 RoPE，量化点在 RoPE 之后、FA 之前，作用于真实参与注意力计算的数值；
2. 模型存在 GQA：`num_heads_q` 与 `num_heads_kv` 由权重 config 决定，二者可不相等；
3. 统一多模态序列（视频/音频/文本 token 拼接）经 `ModalityDispatcher` 重排，序列长度为各模态长度之和，不是固定值——因此**实际序列长度（actual sequence length）契约必须显式处理**，不能假设定长；
4. 两个 FA 函数通过 `magi_register_custom_op` 注册（`is_subgraph_boundary=True`）；base 模式下 `TORCH_COMPILE_DISABLE=1`、`MAGI_COMPILE_COMPILE_MODE=NONE`，实际 eager 执行；
5. `flex_flash_attn_func` 的 LSE 合并依赖 FA 返回 softmax_lse，MXFP8 FA v2 能否同时返回 LSE 须在 Ascend 950 实机确认（见「开放问题」）。

### 仓内公共 MXFP8 FA 模块现状分析

`module/fa_quant/fa_quant.py` 当前能力：

| 接口 | 现有契约 | 限制 |
| --- | --- | --- |
| `npu_mxfp8_quant(x, dst_type, output_layout, axis)` | 输入 BSND `(b, s, n, d)`，view 成 `(b*s, n, d)` 调 `torch_npu.npu_dynamic_mx_quant`，返回 `(量化张量, scale)` | 仅 `output_layout="TND"`；Q/K 沿 `axis=-1`（head_dim 分组）、V 沿 `axis=0` 的用法在 `npu_mxfp8_attn` 内固定 |
| `npu_mxfp8_attn(q, k, v, dst_type, softmax_scale)` | Q/K/V 动态量化后调 `torch_npu.npu_fused_infer_attention_score_v2`：`input_layout="TND"`、`sparse_mode=0`、`query/key/value_quant_mode=6/6/8`、scale dtype `float8_e8m0fnu`、`out_dtype=输入 dtype`；**`actual_seq_qlen/actual_seq_kvlen` 由均匀 `arange` 生成（等长序列假设）** | 仅支持等长、非稀疏基础路径；不支持显式变长 actual_seq_lens；异常时统一抛「需 CANN 9.1.0+」 |

HunyuanVideo 已有接入样例（`attention.py` 的 `mode="mxfp8"` 分支直接调 `npu_mxfp8_attn`），证明标准定长路径的公共封装可直接复用。

结论：

1. **标准定长路径（`flash_attn_func`）**：输入 BSND、非稀疏，与 `npu_mxfp8_attn` 既有契约一致，可直接复用，公共模块零改动；
2. **变长/局部路径（`flex_flash_attn_func`）**：超出公共模块「等长 arange」契约，须**扩展公共接口**支持显式 `actual_seq_qlen/actual_seq_kvlen`（逐段传入真实段长），并保持 HunyuanVideo 等现有调用方行为不变；
3. **GQA**：公共接口已透传 `num_query_heads`/`num_key_value_heads`（取自 `q.shape`/`k.shape`），天然兼容 GQA/MQA；
4. **依赖能力**：`npu_dynamic_mx_quant` 与 `npu_fused_infer_attention_score_v2` 需 CANN 9.1.0+，且 Ascend 950 上的目标算子能力（支持矩阵、LSE 返回、head_dim 支持）须在 RFC 阶段实机验证后锁定。

# 需求分析（required）

## 需求描述

在 Ascend 950 上，以 `fa-mxfp8` 配置开关为 daVinci-MagiHuman 的目标 Attention 分支（RFC 选定：标准 Attention 必做；局部/变长分支按实机验证结论接入或明确回退）接入仓内公共 MXFP8 FA 能力：Q/K/V 动态量化为 MXFP8 后真实执行 `npu_fused_infer_attention_score_v2`，输出恢复 BF16；同 batch 端到端性能优于 BF16 基线、峰值显存下降、生成质量无可稳定辨识退化；不支持的组合 fail-fast 或可审计回退 BF16；公共 `module/fa_quant/` 如有改动须通过全部直接调用方回归。

## 需求拆解

1. **RFC/Issue**：固定基线（infer 仓 commit、上游模型 commit `209209b7086eba2020c5439265221495a8357322`、四组权重 revision/hash）、软硬件版本（Ascend 950 实机型号、CANN/PyTorch/torch_npu）、生成模式与配置（base 文生视频、448×256、5 秒、32 steps、CFG=2、seed=42、`dit_cache.method: NoCache`）、输入（prompt 及 hash）、性能测量方法、量化精度评测策略与数值门限；
2. **profiling 梳理**：采集 BF16 base 基线，记录 `flash_attn_func`、`flex_flash_attn_func`、局部注意力和 context-parallel 路径的 Q/K/V shape、layout、head 数、实际序列长度、sparse 参数与各路径耗时，形成 RFC 的张量契约表；
3. **公共模块扩展**：`npu_mxfp8_attn` 增加可选显式 `actual_seq_qlen/actual_seq_kvlen` 入参（缺省保持现行为），必要时新增变长分段封装；扩展点全部带测试；
4. **模型侧接入**：`dit_module.py` 两个 FA 函数内按开关分发 BF16 FA / MXFP8 FA；配置开关 `fa-mxfp8`、`fa-mxfp8-fallback` 经 `config/fa_mxfp8.yaml` 注入，`bash infer.sh fa_mxfp8` 成为可执行入口；
5. **量化真实性**：profiling 同时观察到 `npu_dynamic_mx_quant` 与 MXFP8 FA v2；不得使用伪造/全 1 scale，不得量化后立即反量化再调用 BF16 FA；
6. **数值与质量**：量化前 Q/K/V、scale、FA 输出、至少一个 DiT block 输出做数值/分布检查（无 NaN/Inf、满足 RFC 门限）；≥5 组固定 prompt、2 名验收人盲测无稳定可辨识退化；
7. **性能与内存**：同配置端到端中位耗时下降且超出 BF16 三次波动范围；FA 算子累计耗时下降，计入量化与 layout 转换后 Attention 总耗时仍下降；NPU 峰值内存低于 BF16 基线；
8. **回退与兼容**：关闭开关恢复 BF16 基线；公共模块改动后 HunyuanVideo `single_fp8.yaml` 代表配置精度/性能回归不低于基线波动下界；非法参数与不支持组合产生可审计日志。

# 详细设计（required）

## 基线锁定（RFC 基线契约）

| 项 | 取值 / 锁定方式 |
| --- | --- |
| infer 仓 commit | 开发分支基于合入时 master 最新 commit，RFC 记录具体 SHA |
| 上游模型 commit | `SandAI-org/daVinci-MagiHuman` 固定 `209209b7086eba2020c5439265221495a8357322`，非覆盖方式合并其 `inference/` |
| 权重 | daVinci-MagiHuman base、T5Gemma、Stable Audio、Wan2.2 VAE 四组，记录各自 revision/hash；路径改于 `base_config.json` |
| 硬件 | Ascend 950 单卡；记录芯片/整机型号、可用 HBM、Host 内存、OS、驱动与固件 |
| 软件栈 | 以仓内 A2/A3 参考（Python 3.12 / CANN 9.0.0 / PyTorch 2.10.0 / torch_npu v26.0.0）为起点，实机确认 Ascend 950 可用组合；`npu_dynamic_mx_quant`、`npu_fused_infer_attention_score_v2` 能力验证通过后在 RFC 锁定，不足时先满足算子仓合入与版本要求 |
| 生成模式 | base 文生视频；448×256、5 秒、32 steps、CFG=2、seed=42、`dit_cache.method: NoCache`；BF16/MXFP8 使用同一 prompt、同配置、同权重 |
| 配置起点 | `config/base.yaml` + `config/base_config.json`（`engine_config.load` 指向 base 权重、`cp_size: 1`） |
| 性能测量 | `ENABLE_PROFILER=1 bash infer.sh base` / `bash infer.sh fa_mxfp8`；每档预热 ≥1 次、正式 ≥3 次，每次完整生成并在计时结束前完成 NPU 同步；记录 Base/Decode/Total 及均值、中位数、极差；profiler 产物在 `prof/<mode>_prof/`，统计量化、layout 转换、FA、Attention 总耗时与调用次数 |
| 精度评测 | 中间张量数值门限（见「数值契约」）+ 固定样本盲测（≥5 prompt：单人/多人/肢体动作/镜头运动/复杂背景，同 seed 双版本，2 名指定验收人）双口径 |

## Attention 路径与张量契约

RFC 阶段以 profiling 实测数据填充下表（shape 为预期形式，`T≈H/8×W/8×t` 量级的统一多模态序列长度，具体数值实机锁定）：

| 契约项 | `flash_attn_func` | `flex_flash_attn_func` |
| --- | --- | --- |
| Q shape | `(b, s, n_q, d)` BSND | 段内 `(seg_t, n_q, d)`（TND 语义） |
| K/V shape | `(b, s_kv, n_kv, d)` BSND | 段内 `(seg_tkv, n_kv, d)` |
| layout 转换 | BSND → view `(b*s, n, d)`（TND），零拷贝 | 段张量原生 `(t, n, d)` 直接作 TND 输入，**取消 BF16 路径中的 permute/unsqueeze** |
| head 数 | `n_q` / `n_kv`（GQA 透传） | 同左 |
| actual_seq_lens | 等长 `[s, 2s, …, b*s]`（公共模块现行为） | 显式 `[seg_t]` / `[seg_tkv]`（batch=1 逐段调用） |
| sparse_mode | 0 | 0（段内全注意力语义与 BF16 路径 `pre/next_tokens=65535` 等价） |
| 量化 axis | Q/K 沿 head_dim（mode 6），V 沿序列（mode 8） | 同左 |

## 量化契约

| 项 | 契约 |
| --- | --- |
| 量化对象 | RoPE/RMSNorm 之后的 Q、K、V 激活（真实参与 FA 计算的数值） |
| 量化算子 | `torch_npu.npu_dynamic_mx_quant`（每次 FA 调用前动态执行，非离线/伪造 scale） |
| 量化 dtype | Q/K/V → `torch.float8_e4m3fn`；scale → `torch_npu.float8_e8m0fnu` |
| 量化模式 | `query_quant_mode=6`（per-channel group）、`key_quant_mode=6`、`value_quant_mode=8`（per-token group），与公共模块现契约一致 |
| softmax_scale | `1/sqrt(head_dim)`，与 BF16 路径一致 |
| 输出 dtype | FA v2 `out_dtype=输入 dtype`（BF16），FA 出来即恢复，后续 gating/proj 不感知量化 |
| FA 算子 | `torch_npu.fused_infer_attention_score_v2`，`input_layout="TND"`、`sparse_mode=0` |
| 禁止项 | 全 1/伪造 scale；量化后立即反量化回 BF16 再走 FA v1；只改配置不走量化算子 |

## 实现方案

### 总体结构

```
config/fa_mxfp8.yaml (fa-mxfp8 / fa-mxfp8-fallback)
        │  infer.sh fa_mxfp8 → mm_launch → pipeline(model_args)
        ▼
DiTModel / Attention（models/davinci-magihuman/）
        │  运行时上下文 fa_quant_context（模块级，不改动 magi 注册算子签名）
        ▼
dit_module.py: flash_attn_func / flex_flash_attn_func   ←─ 开关分发点
        │  fa-mxfp8=false → BF16 FA v1（现行为）
        │  fa-mxfp8=true  → module.fa_quant 扩展接口
        ▼
module/fa_quant/npu_mxfp8_attn（+ 显式 actual_seq_lens 扩展）
        ▼
npu_dynamic_mx_quant ×3 → npu_fused_infer_attention_score_v2
```

### 模型侧设计（models/davinci-magihuman/）

1. **开关注入**：`config/fa_mxfp8.yaml` 从 `base.yaml` 复制全部字段并追加 `model_args.fa-mxfp8: true`、`model_args.fa-mxfp8-fallback: true`；`infer.sh` 的 mode 白名单增加 `fa_mxfp8`，使 `bash infer.sh fa_mxfp8` 直接可执行。开关随 model_args 进入 pipeline，在 DiT 构建阶段写入 `fa_quant` 运行时上下文；
2. **分发点**：`flash_attn_func` / `flex_flash_attn_func` 函数体内部按上下文开关分发。选择在函数体内分发而非改注册签名，原因是两函数已通过 `magi_register_custom_op` 注册（`is_subgraph_boundary=True`），保持签名不变可避免影响 MagiCompiler 的图边界；base 模式 eager 执行（`MAGI_COMPILE_COMPILE_MODE=NONE`），运行时读上下文无编译期分支问题；
3. **标准路径**：`fa-mxfp8=true` 时，`flash_attn_func` 将 BSND 的 Q/K/V 直接交给 `npu_mxfp8_attn`（契约一致，零 layout 额外拷贝），输出 `(b, s, n_q, d)` BF16，后续与 BF16 路径完全一致；
4. **变长/局部路径**：`flex_flash_attn_func` 保留 `_split_q_range_with_no_overlap` 无重叠 q 段切分；每段以 `(seg_t, n_q, d)` 原生 TND 形式直接调用扩展后的公共接口，显式传 `actual_seq_qlen=[seg_t]`、`actual_seq_kvlen=[seg_tkv]`，替代 BF16 路径的 BNSD permute/unsqueeze；同一 q 段多个 k 段的处理见「LSE 合并」；
5. **CP 路径**：`flash_attn_with_cp` / `flex_flash_attn_with_cp` 不改动，量化发生在其内部调用的 FA 函数中，cp_size=1（本任务验收口径）下自动生效；cp_size>1 组合首期不承诺支持，按回退策略处理；
6. **范围控制**：仅模型自身逻辑留在 `models/davinci-magihuman/`；通用 layout/变长/量化能力一律下沉 `module/fa_quant/`，不在模型目录复制通用实现。

### 公共模块扩展（module/fa_quant/）

1. `npu_mxfp8_attn` 增加可选参数 `actual_seq_qlen=None, actual_seq_kvlen=None`：为 `None` 时维持现有 `arange` 等长行为（HunyuanVideo 等现有调用方零感知）；非 `None` 时校验其为 int64 device 张量且末元素等于 `b*s`/`b*s_kv` 后透传 FA v2；
2. `npu_mxfp8_quant` 保持现有签名与行为；如变长路径需要按段量化，复用 `(b*s, n, d)` view 的等价形式（段内 batch=1），不新增接口；
3. 异常处理细化：区分「CANN/torch_npu 版本不足（RuntimeError）」「layout/sparse/seq_len 契约不满足（ValueError）」，供模型侧决定 fail-fast 还是回退；
4. 同步更新 `module/fa_quant/README.md`（契约、扩展参数、回归结论）；所有扩展点补充 UT/功能用例（含显式 seq_lens 与缺省路径的等价性测试）。

### LSE 合并（flex 路径多 k 段）

BF16 路径对「同一 q 段命中多个 k 段」用 FA v1 返回的 softmax_lse 做加权合并。MXFP8 路径：

- 优先验证 FA v2 在 MXFP8 输入下 `softmax_lse_flag=True` 可用且数值正确（RFC 实机确认项）；
- 若多 k 段在序列上连续/可合并，先做区间合并减少 FA 调用次数（段间语义不变）；
- 若两者均不可用，该组合回退 BF16 并记录可审计日志（`fa-mxfp8-fallback=true` 时），不静默使用错误参数。

### 配置与回退行为

| 场景 | 行为 |
| --- | --- |
| `fa-mxfp8: false`（或缺省） | BF16 基线路径，行为与 base 完全一致 |
| `fa-mxfp8: true`，标准路径 | MXFP8 FA v2 执行 |
| `fa-mxfp8: true`，CANN/torch_npu 版本不足、head_dim/序列契约超出 FA v2 支持矩阵、LSE 不可用且 k 段不可合并、cp_size>1 | `fa-mxfp8-fallback: true` → 回退 BF16 FA v1 并输出一次性 WARN 日志（含原因）；`false` → fail-fast 抛错，禁止静默错误 |
| 非法参数（seq_lens 与 shape 不一致、layout 非法等） | 直接抛错（不属于回退范畴） |

### 涉及文件清单

```text
models/davinci-magihuman/
├── config/fa_mxfp8.yaml                          # 新增：开关配置（复制 base.yaml 全字段）
├── infer.sh                                      # 修改：mode 白名单 + fa_mxfp8
└── inference/
    ├── model/dit/dit_module.py                   # 修改：两个 FA 函数分发 + 上下文
    └── pipeline/…                                # 修改：开关读取与注入
module/fa_quant/
├── fa_quant.py                                   # 修改：显式 actual_seq_lens 扩展 + 错误分类
├── README.md                                     # 修改：契约与回归说明
docs/models/davinci-magihuman/
└── davinci-magihuman_optimization.md             # 更新：MXFP8 FA 优化章节
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（PR/DT，具体型号以实机记录为准） | √ |
| Atlas A2/A3 | 不涉及本任务新增；公共模块扩展须保证不劣化既有调用方（含 A2/A3 上的 HunyuanVideo 回归） |

## 算子约束限制

- 本任务不开发独立算子；仅接入与封装扩展，能力边界由 `npu_dynamic_mx_quant` / `npu_fused_infer_attention_score_v2` 支持矩阵决定；
- 不将 MatMul 权重量化、RoPE 融合或缓存算法收益计入本任务；
- 收益不得来自改变分辨率、时长、steps、CFG、prompt 或 seed；
- 不提交模型权重、输入数据或生成视频大文件；
- 保留必要的 BF16 回退路径，不为量化覆盖率移除。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 平台与功能 | Ascend 950 实机开关可启停，目标模式完整生成；不支持组合行为明确、无静默错误 | 任务书 §3.5 |
| 量化真实性 | 同一 profiling 中同时观察到 Q/K/V `npu_dynamic_mx_quant` 与 MXFP8 `npu_fused_infer_attention_score_v2`；dtype/scale/layout/actual_seq_lens/head 数与本文契约一致 | 任务书 §3.5、test_cases/README |
| 数值稳定性 | Q/K/V 量化前张量与 BF16 路径逐位一致；scale 为合法 e8m0 非全 1；FA 输出与 DiT block 输出无 NaN/Inf、无异常饱和；FA 输出与 BF16 参考的误差门限（余弦相似度 ≥0.999、逐元素 `abs ≤ atol + rtol·|ref|` 的具体 atol/rtol）在 RFC 中以首个 case 实测分布锁定 | 任务书 §3.2/§3.5 |
| 最终质量 | ≥5 组固定 prompt、2 名指定验收人盲测，主体/语义/构图/动作/时序/色彩/伪影无可稳定辨识退化 | 任务书 §3.2 |
| 性能 | 同机同配置下端到端中位耗时下降且超过 BF16 三次结果波动范围；FA 算子累计耗时下降；计入动态量化与 layout 转换后 Attention 总耗时仍下降 | 任务书 §3.3、test_cases/README |
| 内存 | 同配置下 MXFP8 NPU 峰值低于 BF16 基线，提交原始值（加载后/生成峰值/连续 5 次生成后） | 任务书 §3.4 |
| 回退与兼容 | 关闭特性恢复 BF16 基线行为；`module/fa_quant/` 全部直接调用方 API/功能回归通过，HunyuanVideo `single_fp8.yaml` 代表配置精度与性能不低于基线波动下界 | 任务书 §3.5、§4 |
| 工程质量 | `pre-commit run --all-files`、仓库 CI、Ascend 950 实机验收通过；无权重/大文件 | 任务书 §3.5 |

## 自测用例规划

| case | 路径 | 验证点 |
| --- | --- | --- |
| base 标准 Attention（RFC 默认 case） | `flash_attn_func` | 端到端精度/性能/内存 + 中间张量门限 |
| RFC 指定局部/变长 case | `flex_flash_attn_func` | 显式 actual_seq_lens 契约、段量化、多 k 段策略；若模型配置无法触发则提交 profiling 证据列为不支持项 |
| GQA head case | 任一命中路径 | `n_q ≠ n_kv` 透传正确性 |
| 开关关闭 | 全路径 | 与 BF16 基线逐位一致 |
| 非法参数 / 不支持组合 | 分发层 | fail-fast 报错信息或可审计回退日志 |
| 公共模块回归 | HunyuanVideo `single_fp8.yaml` | API/功能/精度/代表性性能不低于波动下界 |

## 兼容性分析

公共模块改动仅为**可选参数扩展**，缺省行为与现契约逐位一致，HunyuanVideo 等现有调用方零改动零感知；模型侧改动全部限定在 `models/davinci-magihuman/`，不影响仓内其他模型；`config/fa_mxfp8.yaml` 为新增文件，不修改既有配置语义；`module/fa_quant/README.md` 与模型文档同步更新。CANN 版本门槛（9.1.0+）在文档中显式声明并在运行时校验。

## 与任务方的开放问题

| # | 事项 | 影响 | 应对措施 |
| --- | --- | --- | --- |
| 1 | Ascend 950 上 FA v2 MXFP8 是否支持 `softmax_lse_flag` 返回 LSE | 决定 flex 路径多 k 段走 MXFP8 合并还是回退 BF16 | RFC 首轮实机验证；不可用时按「区间合并 → 回退」策略降级，不影响标准路径收益 |
| 2 | Ascend 950 可用 CANN/PyTorch/torch_npu 组合及 FA v2 MXFP8 head_dim 支持矩阵 | 决定实际软件栈与 fail-fast 条件集合 | 实机确认后在 RFC 锁定；不足时先满足算子仓合入与版本要求 |
| 3 | 权重包内模型 config 的 `num_heads_q/num_heads_kv/head_dim/local_attn_layers` 具体取值 | 张量契约表数值 | 权重准备完成后实机读取并回填 RFC，本文仅锁契约形式 |
