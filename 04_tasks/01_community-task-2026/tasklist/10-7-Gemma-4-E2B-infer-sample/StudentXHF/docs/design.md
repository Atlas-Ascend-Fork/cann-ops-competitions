# Gemma-4-E2B 纯文本推理接入设计文档

作者：StudentXHF。修订日期：2026-10-22。状态：提交设计评审，尚未获得设计批准。

本文是社区任务的独立设计交付，不包含实现代码、模型权重或大日志。
代码提交与设计审批分别记录，不能用代码 PR 已提交或 CI 通过代替设计审核通过。

## 1. 需求背景

### 1.1 需求来源及关联提交

- [社区任务：Gemma-4-E2B 推理样例](https://www.hiascend.com/activities/task-center/details/69423188360d481bab8669b7b9f817e6)。
- 本设计依据 [2026-10-20 下载的任务书及测试附件](https://www.hiascend.com/p/resource/202610/7dcd689564c14ff7a227b55f4c1ad7ce.zip)，ZIP SHA-256：`3c9cbf791c3f42c9dac7cc2352d9b8967ce7954274abfa0bc46b9e109a5db9f`。
- [代码实现 PR #1106](https://gitcode.com/cann/cann-recipes-infer/pull/1106)。该 PR 包含旧版设计说明，但不是本次独立设计 PR。
- 已测试实现提交：[`a122087e23cd38fe865dc77c15832969514893ae`](https://gitcode.com/StudentXHF/cann-recipes-infer/commit/a122087e23cd38fe865dc77c15832969514893ae?ref=feat/gemma4-e2b-a3-test)。历史分支名含 `a3`，不代表当前验收硬件；本设计目标为 Atlas A2。
- [相关 RFC 讨论 #447](https://gitcode.com/cann/cann-recipes-infer/issues/447)：现有精度审计脚本引用其中 v0.9 判据；引用不代表该讨论的所有方案或阈值已获维护者批准，也不代替本文评审。

### 1.2 背景与设计目标

仓内已有 Gemma-4-26B-A4B MoE 样例，但 E2B 是独立新模型，不能只替换旧样例 YAML 或权重路径。
本任务基于统一 executor 接入 E2B 的 BF16 纯文本 Offline 推理，补齐逐层输入（PLE）、跨层 KV 共享和分层双宽 Dense MLP。

最新任务书已明确独立目录、注册名和入口，不再保留旧文档的目录待决项 C01：

| 项目 | 设计要求 |
| --- | --- |
| 模型目录 | `models/gemma_4_e2b/` |
| 注册名 | `gemma_4_e2b`，不得覆盖 `gemma_4` |
| 启动参数 | `--model gemma_4_e2b` |
| 代码、配置、测试、用户/设计/验证文档 | 归入独立 E2B 目录；本文件为社区任务仓的设计评审副本 |
| 可复用范围 | 公共执行器、通用算子及基础组件，不复用旧 MoE 实现、模型配置、权重映射或测试基线 |

## 2. 需求分析与支持边界

首期为 Atlas A2 单逻辑设备、1TP、batch=1、BF16、纯文本、Offline，执行模式为 eager 和 `npugraph_ex` Decode。
视觉/音频、量化、MoE、MTP、Online PD、多卡、极限长上下文和同批次变长 Prefill 不在本期范围。
不新增私有调度器或独立 aclnn/kernel 工程；若公共算子存在必要功能缺口，另行提出方案并取得维护者确认。

### 2.1 模型配置

固定官方模型 `google/gemma-4-E2B` base（非 `-it`），revision 为 `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`。
解析顶层多模态配置的 `text_config`，不从旧 26B-A4B YAML 推导 E2B 结构。

| 配置项 | 固定 revision 对应值/语义 |
| --- | --- |
| 层数、hidden size | 35、1536 |
| Attention 排列 | 每组 4 个 sliding + 1 个 full；从 `layer_types` 读取 |
| Q heads / KV heads | 8 / 1 |
| sliding / full head dimension | 256 / 512 |
| 滑窗 | 512 token |
| PLE 维度 | 每层 256 |
| KV 共享 | 后 20 层消费共享区之前最后一个同类型层的 KV |
| FFN intermediate | 前 15 层 6144，后 20 层 12288；GELU 门控 Dense MLP |
| 输出 | embedding/lm-head 权重绑定，final logit softcap=30 |

这些数值用于校验固定模型身份；层型、共享来源和尺寸仍由配置驱动。配置不匹配、非法共享边界或缺失权重应失败，不降级为旧模型。

### 2.2 统一接口

实现构造函数 `(config, infer_config, comm_manager, prefix)`、`forward`、`load_weights`、`get_cache_info`。
输入为纯文本经固定 tokenizer/template 生成的 token IDs；输出通过统一执行器处理 EOS、最大生成长度和解码。
首期拒绝不支持的 batch/并行配置；参考精度用例对超长输入明确报错，不静默截断。
容量为 2048 输入 + 最多 128 输出，不能据此声称支持配置文件标称的全部长上下文。

## 3. 详细设计

### 3.1 总体数据流

```text
固定 tokenizer/template → input_ids → 缩放主 embedding
                                  └→ 专用 PLE embedding
主 embedding → packed PLE projection → 缩放/RMSNorm → 与 PLE embedding 合成
    ↓
35 层 Decoder：attention → Dense MLP → 本层 PLE 门控/投影/残差
    ↓
最终 RMSNorm → tied LM head → softcap → logits → 统一生成循环
```

### 3.2 逐层输入与数值语义

主 embedding 乘 `sqrt(hidden_size)`，专用 PLE embedding 乘 `sqrt(hidden_size_per_layer_input)`。
主 hidden 经一次打包投影得到 `L × ple_dim`，乘 `hidden_size**-0.5` 后按 PLE 维度 RMSNorm。
该结果与专用 PLE embedding 相加，再乘 `1/sqrt(2)`；各 Decoder 只取本层 slice，依次执行 gate、GELU-tanh、逐元素乘、projection、归一化和残差融合。

RMSNorm 统计采用 FP32，归一化后恢复输入 dtype；保持官方权重乘法语义，不套用其他 Gemma 变体的 `1 + weight`。
embedding 的缩放常数按目标 dtype 转换，避免标量精度差在多层累计。
保持 `layer_scalar`、残差顺序、RoPE、MLP 激活和最终 softcap 与固定 Transformers 参考一致。

### 3.3 跨层 KV 来源与所有权

令 `boundary = num_hidden_layers - num_kv_shared_layers`。
共享区前每层是 owner；consumer 的来源是共享区之前同类型的最后一个 owner，由配置计算，不在 forward 中写死索引。
固定模型对应 owner=0～14、consumer=15～34，其中 sliding consumer 读第 13 层，full consumer 读第 14 层。
consumer 始终计算自己的 Q，不重复计算、写入或释放来源 K/V。

`get_cache_info()` 只声明实际持有缓存的 15 个 owner，每个声明包含有效缓存项，`num_layers` 对应物理 owner 数。
不声明 35 层后附加 20 个空缓存列表，也不让 consumer 重复承担物理缓存生命周期。

| 阶段 | owner | consumer | 生命周期责任 |
| --- | --- | --- | --- |
| Prefill | 计算完整本轮 K/V，按 slot 写框架缓存；在本轮局部字典保留完整 K/V 引用 | 读取对应来源层的完整本轮 K/V，不用截断后的窗口尾替代完整 Prefill | 字典仅存在于本次 forward，不跨请求保存 |
| Decode | 写入当前 token，使用框架 block table / slot mapping | 使用自身 Q，读取来源 owner 的缓存与相同映射 | 由统一框架分配、释放和复用；consumer 不单独 free |
| EOS/上限/异常/下一个请求 | 清理由执行器控制；验证回收后可安全复用 | 不保留跨请求 Python side channel | 覆盖非连续 slot、释放重用、取消与连续请求 |

sliding/full attention 使用对应窗口/全局语义，物理 block size=128。
分别验证 511、512、513 边界、共享区前后层、Prefill 全序列和逐步 Decode，防止窗口 off-by-one 与提前回收。
共享消费者 checkpoint 中存在的 K/V projection、K norm 仍严格加载；运行时不使用不等于可以静默丢弃。

### 3.4 权重加载与异常处理

语言参数从 `model.language_model.*` 映射到独立 E2B 模型。
严格检查重复来源名、未知语言张量、shape 不符和目标缺失；失败立即报错，不以部分加载继续推理。
主 embedding 同时满足 tied `lm_head.weight` 别名，必要 buffer 按官方语义加载或明确重建。

仅允许跳过明确在任务范围外的组件前缀：

- `model.audio_tower.`
- `model.vision_tower.`
- `model.embed_audio.`
- `model.embed_vision.`

固定权重文件为 10,246,621,918 字节；SHA-256：`76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`。
config SHA-256：`e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8`。
tokenizer、模板和权重逐文件摘要由 `reference_hf.py` 写入参考记录，两端必须使用相同文件与输入 token。

### 3.5 图模式

先验证 eager 精度，再通过统一 executor 捕获并执行 `npugraph_ex` Decode。
固定 shape 下区分初次编译/预热与稳态；图内不得依赖 `.item()` 读取动态设备值驱动 host 分支。
两模式使用相同 prompt、模板、EOS、seed 和输出上限，要求完整 token 序列一致；未经批准不启用放宽阈值的替代验收。
重捕获计数、日志和 Profiler 必须保留。历史验证日志存在 excessive recapture 警告，见第 6 节；退出码为零不能代替“稳态无重复编译”的证明。

### 3.6 独立性与现有实现差异

设计要求不复用旧 MoE 模型实现，原 `gemma_4` 注册项及文件保持不变。
但已测试代码 `a122087` 仍从 `models.gemma_4.models.modeling_gemma4` 引用 `_AttnInputs`、`Gemma4Attention`、`Gemma4ForCausalLM` 和 `Gemma4RotaryEmbedding`，并存在类继承。
这是最新独立模型要求下需要评审的实际依赖，不能仅因目录独立就宣布完全符合要求。
后续应按评审将真正通用的接口/基础组件与旧模型实现解耦；若改变实现或公共组件，需重采受影响精度、性能和调用方回归。
本设计 PR 不混入该代码重构，不改写历史已测 commit 和证据。

## 4. 基线、环境和运行配置

### 4.1 已记录环境与复现身份

| 项目 | 2026-10-18 自验记录 |
| --- | --- |
| 硬件 | 单 Ascend 910B3（Atlas A2），HBM 65,536 MiB |
| CANN / Python | 9.0.0 / 3.11.4 |
| 候选运行时 | PyTorch 2.7.1+cpu、torch-npu 2.7.1.post4、Transformers 5.0.0 |
| 官方参考 | 独立 Transformers 5.5.0 环境，NPU 运行 |
| 上游代码基线 | `93527538f1567a4cf42db58883231e4e00f8b213` |
| 已测候选 | `a122087e23cd38fe865dc77c15832969514893ae` |

完整 OS、驱动、固件、依赖锁、设备映射和日志随环境证据交付；上表不构成对未实测 A3/A5 的支持声明。

### 4.2 配置与命令

```yaml
model_config:
  model_name: "gemma_4_e2b"
  model_path: "/data/models/gemma-4-E2B"
  exe_mode: "eager"
  with_ckpt: true
  enable_profiler: false
  enable_weight_nz: false
  enable_cache_compile: false
  enable_static_kernel: false
data_config:
  dataset: "default"
  input_truncated_len: 2048
parallel_config:
  world_size: 1
  attn_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
  dense_tp_size: 1
scheduler_config:
  max_new_tokens: 128
  max_prefill_tokens: 2048
  batch_size: 1
  block_size: 128
```

模型路径为示例占位路径。使用同一固定 revision 获取权重并核验摘要，不以变化中的 `main` 作为基线。
从代码仓根目录运行：

```bash
bash executor/scripts/infer.sh --model gemma_4_e2b --mode offline \
  --yaml gemma_4_e2b_bf16_1tp.yaml

python models/gemma_4_e2b/tests/reference_hf.py \
  --model-path /data/models/gemma-4-E2B \
  --prompt-file models/gemma_4_e2b/tests/data/prompts.json \
  --max-new-tokens 128 --seed 1234 --output reference_results.jsonl
```

固定 base 模型与经过校验的 `chat_template.jinja`，不默认 base checkpoint 含可用模板。
参考端显式渲染同一用户模板，保留输入 token、贪心生成参数、EOS 和完整输出；不引入额外 thinking 行为或采样差异。
图模式修改 `exe_mode` 为 `npugraph_ex`；性能/内存采集另开启 Profiler，不能把不同配置结果混为同一基线。

## 5. 可维可测分析

### 5.1 精度与数据集协议

使用固定 12 个文本样本，覆盖中英文、问答、摘要/改写和简单推理，每类不少于 2 条，保留两端输入/输出 token 和完整文本。
模块用例覆盖 embedding、PLE、sliding/full attention、owner/consumer、双宽 MLP、归一化、softcap 和最终 logits。
合成模块输入采用 seed=0/1/2、正态分布裁剪至 [-4,4]，不能替代真实自然语言样本。

以下是待维护者确认的 RFC v0.9 提案判据，不标记为官方已批准：

| 项目 | 提案与记录方法 |
| --- | --- |
| 模块 BF16 对齐 | `abs(a-b) <= 1e-2 + 2e-2*abs(b)`，全部元素检查 |
| 与参考自然生成 | 审计前 10 个固定样本，至少 8 个序列完全一致，同时保存全部 12 个结果 |
| 同轨迹 logits | teacher-force 使用同一参考前缀，校验 conditioning 摘要；记录全词表误差，审计 max-abs≤0.2、top-32 重合≥24 |
| eager 与图模式 | 完整生成 token 序列一致；未批准的替代规则不生效 |
| MMLU | 固定 test split 全量 14,042 题作为候选首采；参考配对分层 500 题，分数差≤3 个百分点、翻转率≤5% |

MMLU revision=`c30699e8356da336a370243923dbaf21066bb9fe`，固定 prompt 与题目 ID。
完整候选成绩与 500 题配对参考必须分开报告；参考子集不冒充官方参考的全量评测。
若评审要求参考也覆盖全量，则补采全量后再判定，不以子集通过自动豁免。
测试协议变化须更新版本、重新采集受影响基线，不能根据运行结果倒调阈值。

### 5.2 性能、内存及生命周期矩阵

| 类别 | 用例与通过条件 |
| --- | --- |
| 性能 | 128/512/1024/2048 输入 × eager/npugraph_ex，共 8 组；最多输出 128、batch=1 |
| 采集口径 | 同一已加载进程，每组≥1 warmup、≥3 steady；分开模型加载、编译、Prefill、Decode，记录实际 token 数、均值和中位数 |
| 图模式 | 固定 shape 稳态不反复编译；保留捕获事件，不以退出码代替该判据 |
| 最大容量 | 2048+128、batch=1 无 OOM；权重、KV、激活、workspace、图缓存均计入实测峰值 |
| 内存采样 | 加载后、Prefill 峰值、Decode 峰值、连续 20 请求后；NPU allocated/reserved/peak 与 Host RSS 分开记录 |
| KV 正确性 | 511/512/513、共享区前后、非连续 slot、EOS、输出上限、取消、释放重用，无重复写/释放和跨请求污染 |
| 错误路径 | 非法结构、缺失/重复/不匹配权重、不支持的 batch/模式/并行配置明确失败 |
| 回归 | 不修改既有注册行为；如提取公共组件或修改共享缓存，覆盖全部直接调用方，含既有 26B-A4B 受影响路径 |

本任务不预设加速比例，首次有效实机结果作为后续性能基线；回归允许波动范围拟定为 ±10%，待评审确认。
静态权重约 10.2 GB 仅为预算输入，不能推导运行峰值；不得用吞吐改善掩盖精度或生命周期失败。

## 6. 已有自验摘要与尚待闭环事项

以下摘自绑定 `a122087` 的 2026-10-18 自验报告，是历史结果，不代表本独立设计已批准或平台已验收：

| 项目 | 历史记录 |
| --- | --- |
| 固定生成 | eager、npugraph_ex 各 12/12 token/text 一致 |
| 同轨迹 logits | 各 780/780 argmax，一致性报告 max-abs=0 |
| MMLU 候选全量 | 8,051/14,042，57.3351% |
| 500 题配对 | 参考 296/500，候选 293/500；差 0.6 个百分点、翻转 11/500 |
| eager mean TPS（128/512/1024/2048） | 7.5299 / 7.5955 / 7.3566 / 7.4472 |
| graph mean TPS（相同长度） | 3.6509 / 4.0296 / 3.8510 / 4.0629；本轮慢于 eager |
| 2048+128 峰值 | NPU allocated 9,870,131,200 字节，reserved 10,343,153,664 字节 |
| 连续 20 请求后 | NPU allocated/reserved 保留增长 0/0 字节；Host RSS 保留增长 180,224 字节 |

原始 Profiler 记录 212 个文件；完整证据归档 `gemma4_final_full_evidence_a122087.tar.zst` 为 1,030,565,399 字节，SHA-256=`6fcdfaf3a23d447b233955f6d2cc3d08d54edc4aac37f5c5c8057e8c556581b2`。
大文件不提交此 PR，评审需要时提供归档及清单；没有公开下载地址时不虚构地址。

必须单独闭环：

1. 本设计与所引 RFC 的阈值、模板、参考子集和缓存契约需维护者正式评审，不能用自验 PASS 代替。
2. `a122087` 仍依赖旧 `models/gemma_4` 类，见 3.6；需按最新任务书审查和解耦，再补测受影响范围。
3. 旧设计说明位于 `docs/models/gemma_4_e2b/` 且残留旧目录冲突/图模式 gate 叙述，需在代码 PR 文档整改时同步到独立模型目录；本次只提交社区设计。
4. 图模式日志有 excessive recapture 警告；需定位发生阶段，证明固定 shape 稳态无重复编译，不能只依据最终退出码为零宣告该项完成。
5. 官方附件是任务书及测试说明，未附可直接执行的原版 case；自编自验脚本不能称为“官方原始脚本”。
6. 修改代码、权重、环境或协议后，重新绑定 commit/config/日志，重采受影响基线。

## 7. 交付与版本关系

本 PR 仅新增本文件至 `cann/cann-ops-competitions` 对应任务的 `StudentXHF/docs/`。
实现继续由 `cann/cann-recipes-infer` PR #1106 承载，二者分别评审，不关闭或替换已有代码 PR。
社区设计合入后，平台设计文档字段填写本设计 PR；代码字段填写最终经过验证的完整 commit 链接，两者不可混填。

AI 参与：本文整理与交叉核对使用 OpenAI Codex 辅助；维护者审批与真实测试结果不由 AI 声明替代。
模型和参考实现使用须遵守其各自许可，权重、tokenizer、数据集与运行日志不随本设计 PR 提交。

## 8. 参考资料

- [官方固定 revision](https://huggingface.co/google/gemma-4-E2B/tree/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)
- [Transformers v5.5.0 Gemma4 实现](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py)
- [cann-recipes-infer 贡献指南](https://gitcode.com/cann/cann-recipes-infer/blob/master/CONTRIBUTION.md)
- [新模型 Checklist](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/common/new_model_checklist.md)
- [社区任务设计提交目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
