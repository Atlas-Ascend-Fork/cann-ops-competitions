# Qwen3-1.7B BF16 A2 单卡模型样例设计

状态：设计评审稿，2026-10-18。已有本地实现和实机自测，尚未取得社区设计评审通过或合入记录。本稿如实提交现有方案，不补写已评审的历史。验收结论以任务书及维护者评审为准。

# 需求背景

## 需求来源

- [10月社区任务—Qwen3-1.7B 模型样例](https://www.hiascend.com/activities/task-center/details/73500202c71f4d30bc512daeb749f00d)。
- [本任务讨论帖](https://gitcode.com/cann/cann-recipes-infer/discussions/15)。
- [官方任务书压缩包](https://www.hiascend.com/p/resource/202610/6167013f90e2467daddfc9afd27be9b1.zip)，任务书第2至5章及配套测试说明。
- [目标代码仓](https://gitcode.com/cann/cann-recipes-infer)，开发起点 `93527538f1567a4cf42db58883231e4e00f8b213`。

## 背景与现状

现有 Qwen3-8B 已使用 `QwenForCausalLM` 与 `Qwen3Config`。本任务复用该实现，增加独立的 `qwen3_1_7b` 注册、BF16/1TP 配置及文档，完成 Prefill、Decode 和文本生成。

官方 1.7B 配置启用 `tie_word_embeddings`。原加载路径没有处理此共享值，同时未完整审计融合权重的缺失分片。仅增加注册名和 YAML 无法证明全部必需参数已正确初始化。另在输出上限1的实机用例发现公共离线输出方法返回两个内部 token，需要限制返回长度。

# 需求分析

## 范围

1. 官方 Qwen3-1.7B BF16 原始权重，Atlas A2 单卡、1TP、离线推理。
2. 交付 `npugraph_ex` 配置，并以 eager 作功能和质量对照。
3. 支持 Paged KV Cache、Packed Sequence、batch1及至少两个不等长请求。
4. 官方 tokenizer、chat template、默认 thinking；保留完整输入输出 token、文本和生成参数。
5. 建立1.7B首份性能基线，并对受影响的 Qwen3-8B、Qwen2.5-7B-Instruct 路径回归。

不扩展量化、Online PD、多卡优化或长上下文能力。不新增模型 forward 或重复实现模型算法。

## 配置与接口

使用既有入口，仓库根目录启动：

```bash
bash executor/scripts/infer.sh --model qwen --mode offline --yaml qwen3_1_7b_1tp.yaml
```

`--model qwen` 是模型目录，YAML 中 `model_name: qwen3_1_7b` 是独立注册名。模型尺寸、Q/KV heads、RoPE、QK-Norm、attention bias 与 embedding/lm-head 关系由官方 config 决定，不硬编码1.7B维度。

候选 YAML 如下。运行前将路径替换为经校验的官方固定 revision 本地目录；batch2用例还将 `batch_size` 改为2、`max_prefill_tokens` 改为4096，并归档实际 YAML。

```yaml
model_config:
  model_name: "qwen3_1_7b"
  model_path: "/path/to/Qwen3-1.7B"
  dtype: "bfloat16"
  platform_version: "A2"
  exe_mode: "npugraph_ex"
  with_ckpt: true
  enable_profiler: false
  enable_cache_compile: false
  enable_static_kernel: false
data_config:
  dataset: "default"
  dataset_path: "models/qwen/tests/qwen3_1_7b"
  input_truncated_len: 2048
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  seed: 42
parallel_config:
  world_size: 1
  attn_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
scheduler_config:
  max_new_tokens: 1024
  batch_size: 1
  max_prefill_tokens: 2048
  block_size: 128
```

# 详细设计

## 模型注册与执行链路

`executor/core/support_models.py` 新增 `qwen3_1_7b`，映射到已有 `QwenForCausalLM` 和 `Qwen3Config`。执行链路沿用 tokenizer → Scheduler → Prefill/Decode → Paged KV Cache → 离线返回结果，不改变模型 forward。

模型注册后，Qwen共享加载路径的直接模型使用者为 `qwen3_1_7b`、`qwen3_8b`、`qwen25_7b_instruct`。`DefaultModelLoader` 调用 `load_weights`，模型 worker 执行权重后处理和 forward；因此已有两个模型必须回归。

## 权重加载

- 使用官方 config 驱动各层 shape 和 TP 分片，保留既有 QKV、gate/up 融合布局。
- BF16 dense 路径检查缺失、重复、未知键、shape不符及融合分片完整性，异常明确报错，拒绝静默使用未初始化权重。
- `tie_word_embeddings=true` 时，以 embedding 的逻辑权重初始化 lm_head；若检查点显式给出 lm_head，加载前核对一致性。
- embedding运行布局为 `[vocab, hidden]`，已有 Linear 后处理会转置 lm_head，故分别保存逻辑相同而运行布局独立的权重，避免把同一个 Parameter 交给两种布局。
- 原量化加载分支及 forward 保持原有实现；此次验证范围仅BF16。

## 输出长度边界

实机发现上限1时执行节拍产生两个内部 token。修复位于 `OfflineInference.get_valid_output`：返回长度取配置输出上限、请求输出上限及已有EOS/有效长度的最小值，不修改 Scheduler 节拍。直接调用方是 `OfflineInference.generate` 和本任务验证脚本的结果捕获钩子。

新增模型无关边界用例，并对两个旧模型执行真实权重回归。此公共方法的更广泛兼容性仍须仓库CI检验，不据此宣称其他模型全部支持。

## 输入与参考映射

权重和tokenizer固定为官方 `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。不转换、不量化。网络受限时从Qwen官方ModelScope传输，对每个文件核对HF固定revision的Git blob或LFS哈希。

精度参考为同设备上的 Transformers 5.0.0 `AutoModelForCausalLM.generate`，不采用8B输出作标杆。双方使用官方默认thinking、相同chat template、实际截断后的输入IDs、greedy/seed42及相同EOS配置。质量输出上限1024，性能输出上限128；此前128上限未结束thinking的记录保留，不充作完整回答验收。

原入口默认batch1只提交首条提示词，因此五类样本逐条或分批运行。验证脚本直接捕获请求对象中的token，不重新tokenize输出文本来还原token。

## 硬件与软件

| 项目 | 已测试环境及证据归属 |
| --- | --- |
| 初始设备 | Atlas 800I A2 / Ascend910B4，物理卡4映射逻辑0，32 GiB HBM；1.7B质量和四组首份性能来自此设备 |
| 后续设备 | Atlas 800T A2 / Ascend910B3，物理卡2映射逻辑0，标称64 GiB HBM；补采内存、入口及20组旧模型同机对照来自此设备 |
| Host | 两环境容器上限32 GiB；800T cgroup v1实测34359738368字节 |
| 系统 | Ubuntu 22.04.5、aarch64、Python 3.11.15 |
| 框架 | CANN 9.0.0、torch 2.8.0+cpu、torch_npu 2.8.0.post4、Transformers 5.0.0 |
| 其他依赖 | NumPy 1.26.4、PyArrow 19.0.1、Pandas 2.2.3；完整依赖清单随证据提供 |
| 驱动/固件 | 驱动26.1.1、固件9.0.0.9.220 |

两台设备分别记录，不用跨机器时延作性能回归判定。800T runtime可用HBM总量65452113920字节，与标称64 GiB分列记录。

## 性能及容量方案

复用仓内融合算子、Paged KV Cache及 `npugraph_ex`，本任务建立首份1.7B基线，不宣称性能提升。固定输入128/512/1024/2048、输出上限128、batch1，每组预热1次、正式3次；分开统计加载、编译/预热与稳态整网Prefill/Decode，保存原始逐次值、均值和中位数及Profiler索引。

最大配置为输入2048、输出上限1024、batch2，连续20请求记录Host RSS、NPU allocated/reserved及KV回收。最初32 GiB HBM环境已完成该负载。NPU allocator不等于整卡含驱动的全部占用，有限20请求窗口不能证明长期绝无泄漏。

# 可维可测分析

## 验证矩阵

| 项目 | 方法与判定 | 当前状态 |
| --- | --- | --- |
| 权重/工具 | 官方文件哈希、加载错误处理、TP分片、返回长度等单元用例 | 57项通过 |
| 质量 | 中英、知识、摘要、推理5类，保存HF/eager/图batch1/2完整token和文本；对照任务书3.2 | 已采齐；整体质量判定待评审 |
| Packed/KV | batch1/2不等长请求、最大长度、连续请求及缓存回收 | 已执行，未见跨请求污染或OOM |
| 输出边界 | 上限1/128/1024、自然EOS、输入截断 | 已执行；首次边界失败与修复后记录均保留 |
| 非法配置 | 模型名、权重路径、batch、长度、TP、模式等 | 七类负例明确报错、非零退出 |
| 图执行 | warmup后Decode API及正式区间重编译日志 | 有图执行API证据，正式区间未发现Dynamo重编译 |
| 1.7B性能 | 四种精确输入长度，固定128输出上限、1预热/3正式 | 首份性能已采集，不设不存在的优化前比值 |
| 内存 | 最大2048/1024/batch2、20请求及profile_memory | 观察窗口未见持续增长；Profiler缺失记录限制见下 |
| 旧模型回归 | Qwen3-8B、Qwen2.5-7B-Instruct，原始/修改后同机同配置 | 20组通过，包含一次固定两组稳定性诊断 |
| 风格/CI | 任务改动文件检查、全仓检查、平台CI | 18个改动文件通过；全仓有上游失败，平台CI未运行 |

旧模型权重分别固定为8B `b968826d9c46dd6066d109eabc6255188de91218`、2.5-7B `a09a35458c702b33eeacc393d103063234e8bc28`。执行前固定：修改后Prefill/Decode中位时延不高于原始1.05倍，原始重复波动不超过±5%。首次失败及不稳定记录保留，后续一次相邻对照同时满足新配对和保留原基线门槛；不放宽阈值，不混用跨机数据。

## 必须公开的限制及待评审事项

1. **输出一致的口径**：任务书允许后端文本差异并要求质量无稳定退化，配套测试说明另写eager/图“输出结果需要保持一致”。现有完整token序列存在差异，不能声称逐token一致。请评审明确口径；若要求严格一致，按要求定位并修复，保留原始记录。
2. **完整质量**：五类回答均可读、相关、核心任务正确且自然EOS，但知识题附加事实有错误。HF称木星距离太阳最远；eager称其质量超过太阳70%；图batch1称其唯一有显著光环；图batch2错误描述其最大卫星和核心。不同后端附加错误不同，现有证据不足以宣布无稳定退化，也不足以直接归因为加载错误。完整回答随证据提供，不只展示正确片段。
3. **内存Profiler**：补采收尾后CSV/JSON及哈希检查通过，20请求结束NPU allocated为4927810048、reserved为6079643648字节，Host RSS为2951737344字节，KV49总/48空闲。但仍有32条不完整记录告警、8条operator记录缺释放时间；不据此推断长期无泄漏。
4. **全仓检查与CI**：全仓pre-commit因上游格式和脚本权限失败，490个无关自动格式改动已恢复；任务18文件检查通过。尚无GitCode CI结果，不能写成全仓或平台CI已通过。
5. **提交位置**：公共活动提供竞赛仓设计目录，模型任务书指定代码仓并允许 `docs/models/qwen/` 独立设计文档。设计稿可按维护者约定迁移；不为模型任务套建Ascend C算子kernel目录。

## 交付与追溯

公开交付为注册、YAML、必要兼容代码、测试脚本、README、RFC及baseline metadata。详细复现方法见 `models/qwen/agentic/RFC_qwen3_1_7b.md`；大文件不入Git。

现有本地候选包可用于评审准备，尚无公开下载地址和提交commit：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| Qwen3-1.7B-候选源码.zip | 134992 | `69818d1189eb0a2f8ebd61bfb71ad026d9b5ebbda30855d0b94f738b768fcb28` |
| Qwen3-1.7B-验证证据.zip | 8401999 | `42d2556c90082b5d26d534e7c3906cd55abc7bcae404226e1263272e286edc5b` |

正式验收前须固定候选代码commit，提供可访问的代码/证据位置，并完成设计PR评审合入和待确认项处理。当前不能以开发起点commit冒充包含新增实现的验收commit。
