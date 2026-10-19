# Cosmos3 补齐 T2I、I2I 与 T2V+Sound 的昇腾 NPU 适配设计文档

| 项目 | 内容 |
| --- | --- |
| 文档状态 | 设计评审稿 v0.3.0，2026-10-16 补齐社区审核入口、CheckList 对照和待审差异；未取得设计审核通过结论 |
| RFC / Issue | [#137：Cosmos3 补齐 T2I、I2I 与 T2V+Sound 场景适配方案](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/137) |
| 作者 / GitCode | 钟才智 / `gcw_HNGPtE0T`（个人提交） |
| 本次修订 AI 辅助 | OpenAI Codex；模型 `gpt-6-astra`（依据本地会话记录核对）；辅助文档整理和证据核查，不代表人工评审结论 |
| 社区任务编号 | 2026 年 10 月第 19 项：`10-19-Cosmos3-T2I-I2I-T2V-Sound` |
| 社区设计审核目录 | `cann/cann-ops-competitions` 的 `04_tasks/01_community-task-2026/tasklist/10-19-Cosmos3-T2I-I2I-T2V-Sound/gcw_HNGPtE0T/docs/design.md` |
| Recipes 技术设计路径 | `docs/world_model/cosmos3/design.md` |
| 编写日期 | 2026-10-14 |
| 任务 | 10 月社区任务——Cosmos3 模型 T2I、I2I、T2V+Sound 场景适配 |
| 目标代码仓 | `cann/cann-recipes-embodied-ai` |
| 目标样例 | `world_model/cosmos3` |
| 方案核对时的样例仓基线 | `db9b2453ad70ac919f8c42e02a61d16db63f4e66` |
| 锁定的 Cosmos 上游基线 | `a61b292ccbc337c25c0c82d2a8cca75feef96cee` |
| 计划完成时间 | 2026-11-01（社区任务截止日期；以评审及平台安排为准） |
| 实现状态 | 实现 PR #207 保持 Draft；A2/A3 六场景及回归证据已采集。A2 首轮 V2V 性能失败及 AVAE 初始化加载差异仍待审，不声明整体验收完成。 |

## 审核入口与证据边界

本稿按[社区任务流程 #39](https://gitcode.com/org/cann/discussions/39)和[设计文档 CheckList](https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001)补交到 competitions 的专属任务目录。社区设计审核、Recipes 技术评审和实现验收是不同环节；代码目标仓仍遵循任务书，不因补交设计文档而迁移实现代码。

- [RFC Issue #137](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/137)：方案、资产和验收口径讨论。
- [社区设计 PR #2247](https://gitcode.com/cann/cann-ops-competitions/pull/2247)：本任务在 competitions 仓库的纯设计文档评审入口。
- [Recipes 设计 PR #206](https://gitcode.com/cann/cann-recipes-embodied-ai/pull/206)：保留原技术设计讨论，并同步本次修订；不能以该 PR 的 CI 代替社区设计审核。
- [实现 PR #207](https://gitcode.com/cann/cann-recipes-embodied-ai/pull/207)：代码、自验记录和已知限制；保持 Draft，待设计审核和验收差异确认。
- 本文证据链接固定到实现提交 `5d36522df2c9d08f587fef9ebe181e4b317a2e9c`。本次只更新设计文档，不重跑模型，不修改原报告、媒体哈希、测量参数或通过条件。

### 请求评审确认

1. **AVAE 生命周期差异**：任务书要求音频模块按需加载；固定 Nano 的 `sound_gen=true` 在模型初始化时构造 AVAE，`enable_sound` 只控制声音生成/解码。本稿请求确认是否接受保留该上游语义；若不接受，需先评审生命周期改造方案并追加针对性验证，不能把该项勾为已满足。
2. **性能口径与 A2 首轮失败**：既有自验使用 `median(candidate) <= max(baseline)`，它是已执行的自验计算规则，不是维护者已批准的最终解释。A2 V2V 首轮超出基线上限 `0.01059991994405 s`，原 `acceptance_pass=false` 保留；固定一次反向顺序复测通过，不覆盖首轮。请结合两轮数据及任务书原文确认最终验收结论。
3. **流程状态**：实现和设备验证已先行开展，但社区设计尚未审核通过；这不构成追认。取得明确设计审核结论并解决待审项前，不申请将实现标为最终验收完成。

# 需求背景（required）

## 需求来源

本设计依据 2026 年 10 月社区任务《Cosmos3 适配 T2I、I2I 与 T2V+Sound 场景》的任务书和自验说明，并参考 cann/cann-recipes-embodied-ai 的贡献指南、现有 Cosmos3 样例，以及 NVIDIA cosmos-framework 固定提交 a61b292。

该任务要求沿用上游已有模型模式和推理语义，在现有昇腾适配样例上以最小改动补齐文生图、图生图和带声音的视频生成；同时证明已支持的 T2V、I2V、V2V 行为没有回归。A2 和 A3 均属于验收平台要求；两端实际环境、锁定资产、六场景和回归结果已记录于 [实测记录](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/validation.md)。有实测记录不等于所有验收条件均满足，A2 性能和音频加载行为仍需评审。

## 背景介绍

当前样例以 NVIDIA cosmos-framework 的固定提交 a61b292 为基础，通过覆盖式 overlay、npu_adapt.sh、CANN attention 后端和本地 checkpoint patch 适配昇腾。现有 README 记录了 T2V、I2V、V2V 的运行方式，并指出样例原先验证的软件栈为 CANN 9.0.0、torch_npu 2.10.0、Python 3.13。

本次核对的上游提交已经包含 text2image、image2image、text2video 模式、对应模式默认参数，以及 text2video 的 enable_sound 开关；同一提交也包含 T2I 和 T2V+Sound 输入样例。image2image 的默认参数存在，但该提交没有提供 inputs/omni/i2i.json，因此本任务需新增符合上游字段契约的 I2I 示例。上游在 enable_sound 为 true 而模型配置没有 sound_gen 时会报错；本地 checkpoint patch 已为 checkpoint 内 tokenizer、Wan VAE 和 AVAE 的离线加载提供适配基础。

镜像名称不作为运行时证据。本轮两端实际使用 CANN 9.1.0、Python 3.12.13、PyTorch/torch_npu 2.10.0、PyAV 18.1.0 和单卡；型号、驱动、实际导入路径及 smoke 结果见实测环境矩阵。原样例的 CANN 9.0.0/Python 3.13 仅保留为历史说明，不作为本轮复现命令。

## 当前上游能力与场景需求分析

| 场景 | 上游模式和输入契约 | 计划复用的模型/路径 | 输出及验收重点 |
| --- | --- | --- | --- |
| T2I 文生图 | model_mode=text2image；沿用上游 prompt、name 等输入结构 | 首选现有样例所用 Cosmos3-Nano；复用图像生成模式和单帧输出分支 | 每条输入生成一张可解码的 vision.jpg；检查尺寸、格式和非有限值 |
| I2I 图生图 | model_mode=image2image；必须提供本地 vision_path 和 prompt | Cosmos3-Nano；复用上游图像条件预处理、条件编码和 image2image 默认参数 | 输出有效图像；固定 seed 后更换输入图，确认输入条件确实参与生成 |
| T2V+Sound | model_mode=text2video 且 enable_sound=true；prompt 按上游 T2V+Sound 样例表达视频内容和声音描述 | Cosmos3-Nano checkpoint 与其 sound tokenizer/AVAE；复用 text2video 声音生成、解码和封装路径 | 输出同时含有效视频轨和音频轨的制品；音频非空、非静音且与视频对齐 |
| 旧场景回归 | T2V、I2V、V2V 继续沿用现有 JSON 输入和默认路径 | 与改动前相同的 checkpoint、上游提交、输入和运行参数 | 同平台相同 seed 下核对模型原始输出、最终制品、端到端耗时和峰值显存 |

a61b292 中的 text2image 默认参数包括 50 步、guidance 4.0、shift 3.0 和 1:1 宽高比；image2image 默认参数包括 35 步、guidance 6.0、shift 5.0。图像模式的单帧语义由上游推理参数处理，不在昇腾适配层重新实现。最终运行分辨率和采样参数以自验说明及锁定 RFC 为准。

## 模型与资产基线

三个新增场景使用同一 Cosmos3-Nano checkpoint，revision 固定为 e59a53c25979a090fa8706c9acc0c254a6e89b92。上游 a61b292 的 Nano 配置将 sound_gen 设为 true，并配置 AVAE。两端资产已经逐文件校验，来源和 SHA256 见 [资产清单](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/assets.sha256.json)；具体运行证据见实测记录，不以在线目录存在代替实机验证。

| 资产 | 本轮固定值 | 使用约束 |
| --- | --- | --- |
| Cosmos3 主 checkpoint | nvidia/Cosmos3-Nano，revision e59a53c25979a090fa8706c9acc0c254a6e89b92 | 本地保存获取来源、revision 和逐文件 SHA256；不提交权重 |
| Sound tokenizer / AVAE | 使用上述 Nano checkpoint 随附的 sound_tokenizer | 加载失败或目录/权重缺失时明确报错；不得退化为无声视频 |
| Vision VAE | 现有样例要求的 Wan2.2_VAE.pth；获取说明参考 Wan-AI/Wan2.2-TI2V-5B，固定 revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e | 记录实际文件 SHA256 和放置路径；不提交权重 |
| 输入媒体 | 使用任务自验固定的 T2I prompt、I2I 图像及三种旧场景输入；T2V+Sound 固定 prompt | 输入地址、尺寸、参数和 SHA256 写入 RFC；大文件和真实媒体不入库 |

Cosmos3-Edge 的公开模型快照未见 sound_tokenizer 目录，不能用于 T2V+Sound 验收。若 Nano 在固定上游提交、锁定权重和目标平台上无法满足某个模式，不临时替换模型或伪造兼容分支；应暂停该模式，核实上游正式支持的专用 checkpoint、硬件资源和任务范围，并更新 RFC 后再继续。a61b292 注册了 Cosmos3-Super-Text2Image 专用变体，但它属于 32B 级模型候选，只有确认任务允许、平台显存足够并通过设计评审后才考虑，初始方案不依赖它。

Cosmos framework 的固定提交使用 OpenMDW-1.1；cann-recipes-embodied-ai 仓库使用 Apache-2.0。PR 只交付按目标仓许可编写的适配代码、patch、配置、测试和文档，不复制完整上游源码，不提交权重、输入媒体或生成制品，并保留上游及模型资产的来源和许可说明。

### RFC 锁定项

以下为本轮已记录的代码、资产和测量协议，最终评审状态仍以 RFC/PR 为准；不因本次文档更新追认未取得的维护者批准。

| 锁定项 | 当前值/状态（最终口径待审） | 锁定证据 |
| --- | --- | --- |
| Recipes 代码基线 | `cann/cann-recipes-embodied-ai`，设计依据提交 `db9b2453ad70ac919f8c42e02a61d16db63f4e66`；贡献仓许可证 Apache-2.0 | RFC 记录实现分支及最终代码提交 |
| Cosmos 上游 | `NVIDIA/cosmos-framework`，`a61b292ccbc337c25c0c82d2a8cca75feef96cee`；上游声明 OpenMDW-1.1 | 固定提交号；若升级，附 overlay、patch 和 attention 兼容性评估 |
| 模型变体 | Cosmos3-Nano，三个新增模式已有双平台运行记录 | 模型配置、文件清单和实测摘要 |
| Checkpoint / tokenizer | Nano revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`；逐文件 SHA256 已核对 | assets.sha256.json |
| Vision VAE | Wan2.2 VAE revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`；文件 SHA256 已核对 | assets.sha256.json 和 validation.md |
| A2/A3 环境 | 实际型号、软件版本、驱动、卡数和 smoke 已采集 | validation.md 的双平台环境矩阵 |
| 推理输入与协议 | 六场景固定参数和输入媒体哈希已记录，预热一次、正式三次 | inputs.sha256.json、双平台 acceptance-summary.json、README 命令 |
| 验收容差 | 保留原 `1/24 s` 声画容差及精度/性能规则；新增时间轴起止检查不放宽阈值 | validation.md；A2 首轮失败和固定复测结果同时保留待审 |

# 需求分析（required）

## 需求描述

在 Atlas A2 和 Atlas A3 上，通过现有 Cosmos3 统一推理入口，使用 upstream-compatible 输入和参数实现 T2I、I2I、T2V+Sound 三类推理。改动限定在 world_model/cosmos3 样例 overlay、必要的本地 patch、场景示例、测试及文档中。默认路径继续关闭声音，公共 attention 行为、checkpoint 加载语义和多卡参数不得因新增模式而变化。

新功能必须复用上游模型模式、预处理、条件编码、采样、VAE/AVAE 解码及输出封装。NPU 适配集中在设备初始化、torch_npu 运行时、HCCL、CANN attention 和离线 checkpoint 资源定位。不得训练或蒸馏 checkpoint，不实现新的声音 tokenizer，不通过改动生成语义绕过模型资产限制，也不把完整 Cosmos 源码复制进适配仓。

## 需求拆解

1. 固定 Cosmos 上游提交、仓库 License、模型变体、checkpoint revision、VAE 与声音 tokenizer 资产及其哈希。
2. 固定 A2/A3 型号、CANN、PyTorch、torch_npu、Python、驱动、卡数、精度、输入、seed、分辨率、帧数、采样步数、编码依赖和 guardrail 参数。
3. 提供符合上游输入契约的 T2I、I2I、T2V+Sound 示例，使用现有统一推理入口和独立输出目录。
4. 给缺失或错误的输入、checkpoint、tokenizer、VAE、编码器和非法参数提供可定位错误；必需音频模块缺失时非零失败，不静默生成无声制品。
5. 在 A2、A3 上完成三种新增模式的功能和输出检查，在两个平台分别完成 T2V、I2V、V2V 改动前后精度、性能和内存回归。
6. 连续进行至少五次新旧场景混合调用，验证状态清理、输出隔离和内存释放；音频模块按需加载仍是任务要求，Nano 初始化行为与该要求的差异单独提请评审，不因混合调用通过而视为已满足。
7. 交付 T2I、I2I、T2V+Sound 的 JSON/YAML 场景示例和 A2/A3 环境说明，示例媒体路径使用相对路径或占位符；更新 README 中的支持矩阵、模型与资产、双平台环境、命令、输出结构、A/V 检查、限制和排障；补齐适配说明、自验步骤、环境/实验记录，并完成 pre-commit、仓库 CI 与平台验收。

# 详细设计（required）

## 场景数据流与统一推理入口

三个新增场景和已有场景都通过 cosmos_framework.scripts.inference 进入；不新增另一套推理 CLI。

数据流与模块生命周期如下；预检失败时返回非零状态，不生成成功占位制品。

```text
JSON / local media / fixed checkpoint
  -> upstream arguments and mode defaults
  -> input, asset and encoder preflight
  -> NPU / HCCL / CANN attention initialization
  -> model initialization (Nano sound_gen=true: AVAE constructed here)
  -> mode dispatch and sampling -> original latent / model output
     |-- T2I / I2I: image VAE -> vision.jpg -> size / decode checks
     |-- T2V / I2V / V2V: video VAE -> vision.mp4 -> regression checks
     '-- T2V + Sound: video VAE + AVAE decode -> mux -> A/V timeline checks
  -> isolated output directory and traceable run metadata
```

图像生成和编辑保持单帧输出，不将图像扩展为视频帧。I2I 复用上游 vision_path 预处理与图像条件编码。固定 Nano 的 sound_gen=true 会在模型初始化时构造 AVAE；enable_sound=true 控制声音生成和解码，不控制音频权重按请求懒加载。本次保留该上游行为，与任务书按需加载要求的差异明确待维护者评审，不声明该项已经满足。T2V 默认 enable_sound=false 时路径保持现状；音视频复用上游输出实现，不自定义无声占位音轨或改变同步语义。

## 适配实现方案

### 上游基线和 overlay

- 固定 cosmos-framework 为 a61b292ccbc337c25c0c82d2a8cca75feef96cee。现有 npu_adapt.sh 校验短提交 a61b292，适配脚本的字符串替换和 patch 都依赖该版本源码布局；未经 RFC 审核和完整回归不得升级上游。
- 以 cann-recipes-embodied-ai/world_model/cosmos3/ 为唯一模型样例代码变更范围，继续用 overlay 和 adaptor_patches 管理差异，不把上游仓源码复制到贡献仓。
- 先对 npu_adapt.sh、adaptor_patches/inference_local_checkpoint.patch、CANN attention overlay 做逐文件适用性检查。保留检查上游 HEAD 的保护逻辑；升级或变更脚本前先证明本地修改不会误伤其他上游逻辑。
- 新场景示例放入样例的独立 `examples/inputs/` 目录；输入媒体路径使用相对路径或明确占位符，不提交媒体，也不覆盖 Cosmos 上游已有输入。示例字段与 a61b292 的 `inputs/omni/t2i.json`、`inputs/omni/t2vs.json` 及 image2image 参数定义一致。新增 `i2i.json` 必须提供本地 `vision_path` 占位符。

### NPU 设备与 attention

- 沿用现有 torch_npu 默认设备、NPU memory probe、NPU stream/event、分布式 HCCL 初始化、device placement 和 CANN attention 注册适配。
- NPU 执行时选择样例已集成的 CANN FIA attention backend；只在 cosmos3 overlay 和该样例上做必要变更。不得修改共享 framework/module 层；若核查发现必须触及共享层，应先列全直接调用样例并增加覆盖，再通过 RFC 更新范围。
- 检查 autocast/精度上下文、device、stream、attention backend 在三种新模式都走同一适配路径；不得为新模式新增改变 dtype、随机数、scheduler 或 prompt 语义的分支。
- 先做单卡 smoke；多卡参数仅验证旧行为不回归，本任务不要求优化 CP、CFGP、FSDP。

### 本地 checkpoint 与音频链路

- 沿用本地 checkpoint patch，从 checkpoint 自带 tokenizer、vision encoder 和 sound_tokenizer 读取资源；Wan2.2_VAE.pth 可按现有样例约定放置并从本地路径加载。
- 启动预检应核验所选模式需要的权重和资源，错误信息包含具体资源路径及补救说明，不在离线时静默联网或退回其他 checkpoint。
- T2V+Sound 预检 sound_gen、sound_tokenizer 配置、AVAE 权重和编码/封装依赖。缺少任一必需能力都在采样前失败并返回非零状态。
- 生成后确认 AVAE 输出有样本、幅度非零，采样率及声道与实际 tokenizer 配置一致；探测封装产物中确有视频和音频轨，验证均可解码。不可只依据文件存在或日志声称成功。
- 图像编码和音视频封装沿用 upstream 依赖与函数。检查环境中的 PyAV、imageio-ffmpeg/FFmpeg 实际可用性；不引入无必要的替代编码器。

### 错误处理与状态隔离

输入类型与扩展名、vision_path 存在性、分辨率约束、模型资源和编码依赖应在尽可能早的阶段校验。异常必须保留可定位的原始原因并令目标调用失败；不吞掉异常后返回伪成功状态。音频生成仅由 enable_sound 显式打开；AVAE 模型对象随模型初始化创建，临时对象按上游生命周期释放，不声称按请求创建或释放音频权重。每次运行使用独立输出目录，固定输入、seed、模型和环境参数写入 sample_args.json 及运行日志。

## 适配范围与约束

- 仅支持任务书列明的 Cosmos3 T2I、I2I、T2V+Sound 新场景，以及 T2V、I2V、V2V 回归；目标平台为 Atlas A2 和 Atlas A3。
- 继续使用锁定 Cosmos 上游的模型模式、输入字段、默认参数和生成语义；不训练/蒸馏模型、不实现新的 tokenizer、不以无声输出替代失败的音频链路。
- 初始实现不要求优化 CP、CFGP、FSDP；多卡参数只做不回归验证。不得提交 checkpoint、VAE、tokenizer、输入媒体或生成制品。

## 支持硬件

目标硬件为 Atlas A2、Atlas A3；本轮实际设备和软件组合见下表及固定实测记录，不将该结果推广到未验证型号或版本。

### 环境与初始运行参数

| 记录项 | A2 | A3 |
| --- | --- | --- |
| 设备身份 | 单卡 Ascend910B3 | 单个可见 NPU，设备名 Ascend910_9382；npu-smi 管理查询异常单独记录 |
| Python | 独立 venv，3.12.13 | 独立 venv，3.12.13 |
| PyTorch | 2.10.0，实际模块路径/git version 见 validation.md | 同左 |
| torch_npu | 2.10.0，导入与真实 NPU smoke 通过 | 同左 |
| CANN / 驱动 | CANN 9.1.0，驱动 26.1.1 | CANN 9.1.0，驱动 26.1.1 |
| 输入和 checkpoint | 与 A3 共用同一哈希和推理配置 | 与 A2 共用同一哈希和推理配置 |

初始可复现设置沿用自验说明：parallelism-preset=latency、resolution=480、seed=0、NPUs=1、--no-guardrails。T2I 固定 1:1 图像输入/输出；I2I 固定输入图像及 prompt，记录实际尺寸和宽高比；T2V+Sound 固定视频 prompt、audio_description、fps、duration 和输出编码参数。现有场景使用仓内 t2v.json、i2v.json、v2v.json。两端使用同一已归档输入清单、checkpoint 哈希和场景参数；具体数值见下方固定证据链接。归档自验配置不等于维护者已经批准验收口径。

本轮复现命令统一为已运行的 Python 3.12.13/CANN 9.1.0 组合，并固定 PyAV 18.1.0。更换平台镜像或依赖后仍须重新采集环境并验证兼容性，不把历史实测推广为其他版本已验证。

# 可维可测分析

## 精度标准/性能标准

| 指标 | 设计标准 | 依据/锁定方式 |
| --- | --- | --- |
| 新场景精度与功能 | T2I/I2I 输出符合输入契约且可解码；T2V+Sound 同时产生有效的视频轨和非静音音频轨；均无 NaN/Inf | 任务书及自验说明；固定输入、seed、checkpoint 和格式 |
| 旧场景精度 | T2V、I2V、V2V 改动前后模型原始 latent/输出一致；仅封装元数据变化时解码后逐帧像素一致 | 原始输出数值判据及计算方法在 RFC/Issue 中预先锁定 |
| 旧场景性能 | 耗时越小越好；既有自验按 `median(candidate) <= max(baseline)` 判断，保留原始测量和失败；峰值显存增加须解释 | 任务书要求不退化；同平台、同资产/参数，预热 1 次、正式 3 次；该计算口径与任务书原文的对应关系待评审 |
| 新场景性能 | T2I、I2I、T2V+Sound 记录正式运行耗时，不要求相对上游加速 | 自验说明；每场景预热 1 次、正式测量 3 次 |
| 内存与稳定性 | 六场景记录 NPU/Host 峰值；新旧场景混合调用至少 5 次，无 OOM、输出覆盖或扣除常驻缓存后的持续增长 | 自验说明；固定混合顺序并保留逐次 profiling 记录 |

## 测试用例规划

### 功能与精度验收

| 测试对象 | 方法 | 通过条件 |
| --- | --- | --- |
| T2I | A2、A3 分别以固定 prompt、checkpoint、seed 和 480 配置运行 | 仅输出预期单张 vision.jpg；文件可解码，尺寸/格式符合配置；模型中间值与输出无 NaN/Inf |
| I2I | 锁定验收集中的一张固定输入图与 prompt/seed；另用一张替代图做条件敏感性补充检查 | 固定输入输出可解码；固定 prompt/seed 后替换输入图，生成输出必须变化；同时确认条件张量变化，并在 RFC 中预先锁定差异计算方法，证明不是忽略输入的 T2I 路径 |
| T2V+Sound | 以 text2video、enable_sound=true 运行固定声音描述样例 | 最终制品有可解码视频轨和音频轨；音频采样数大于零且非全零；记录采样率、声道、视频/音频时长与 A/V 差值 |
| T2V / I2V / V2V 回归 | 每平台以改动前后相同的 checkpoint、输入、seed、参数各运行 | 模型原始 latent/输出保持一致，数值比较阈值与计算方法在 RFC 预先锁定；若仅封装元数据变化，解码后逐帧像素一致 |
| 混合调用 | 新场景与旧场景交错连续执行至少 5 次 | 输出目录互不覆盖；无跨请求状态污染；扣除框架常驻缓存后显存不持续增长，无 OOM；音频按需加载与 Nano 初始化行为的差异单独待审 |

本轮自验沿用既有 `1/24 s` 声画容差（24 FPS 下约 41.7 ms），同时检查时长差、起点差和终点差；缺失或非单调时间戳应失败。该数值已用于历史检查，最终验收口径仍待评审，不因结果产生而放宽阈值。音频非静音同时用样本数、峰值/均方根和听检/播放检查确认，避免只依赖容器轨道元数据。

### 异常输入与资源检查

以下六类异常均须在 A2/A3 覆盖；每项产生非零退出状态和可定位错误，不得留下被判定为成功的空制品：

| 异常类别 | 验证内容 |
| --- | --- |
| 缺少 sound tokenizer | enable_sound=true 但 checkpoint 不含 sound_gen 或 AVAE 目录/权重 |
| 缺少编码器或封装依赖 | 必需图像/视频/音频编码能力不可用 |
| 错误输入类型 | image2image 的 vision_path 指向不支持或无法解码的媒体类型 |
| 缺少视觉输入 | image2image 未提供 vision_path 或文件不可访问 |
| 非法尺寸/分辨率 | 使用不被模型或 VAE 支持的尺寸，检查前置校验和错误说明 |
| 错误或不完整 checkpoint | checkpoint 路径错误、必需权重/配置不完整 |

### 性能与内存测试

- 改动前后在 A2、A3 分别测量 T2V、I2V、V2V：预热 1 次、正式 3 次。令基线延迟为 `b1,b2,b3`，候选延迟为 `c1,c2,c3`；既有自验判据为 `median(c1,c2,c3) <= max(b1,b2,b3)`，不额外增加百分比容差。记录输入准备到媒体保存的批次墙钟耗时（含原始张量落盘，不含模型加载），基线和候选均开启相同观测，profiling 单独运行。
- 任务书原文为“稳态端到端中位耗时不劣于各自基线重复测量波动下界”。本稿修正初稿将“耗时”写成“不低于”的表述错误，**不把 `max(baseline)` 冒称为该原文已经获批的等价解释**；具体验收公式仍请维护者确认。本文不修改工具、原报告或历史通过/失败值。
- A2 首轮 V2V 候选中位耗时为 `272.89704328996595 s`，基线上限为 `272.8864433700219 s`，超出 `0.01059991994405 s`；固定一次复测通过，首轮失败仍待审。峰值显存不得出现无法解释的增长；不能通过更改分辨率、帧数、步数、精度或 guardrail 设置规避。
- 新增 T2I、I2I、T2V+Sound 各预热 1 次、正式执行 3 次并留存每次耗时。本任务不要求新增场景相对上游获得加速。
- 记录六个场景单卡 NPU 峰值显存与 Host 峰值内存；启用框架 memory profiler。混合调用每轮记录推理后显存，扣除框架常驻缓存后不应持续增长。
- 不通过降低分辨率、帧数、采样步数、精度或改变 guardrail 设置来规避性能与精度回归。基线和候选参数必须相同。

## 兼容性分析

新增能力限定在 Cosmos3 样例 overlay、必要 patch、场景示例、测试及文档内。默认未开启 `enable_sound` 的既有调用仍沿用原分支，不改变默认 `model_mode`、CANN attention、本地 checkpoint、多卡参数及输出目录语义。新旧模式通过同一推理入口调用；在 A2/A3 上以同一代码基线、checkpoint、输入和参数完成改动前后验证。静态输入校验、模块导入和输出结构检查只证明各自覆盖的环节，不能代替真实 NPU 功能、精度和性能测试。

如确需修改共享 framework/module/attention 代码，先列出全部直接调用样例，并通过 RFC 扩大范围；合入前补齐受影响样例在 A2/A3 上的代表性功能、精度和性能回归证据。

## 工程质量与可维护性

- 新增测试覆盖三种新模式的输入契约、输出结构和资源错误；旧模式有 A2/A3 双平台回归。
- 运行 `pre-commit run --all-files`、仓库 CI 和目标平台实机验证。分别记录配置/输入检查、模块导入、NPU smoke、完整模式验收；静态检查不能替代设备实测。
- README 和适配说明覆盖支持矩阵、模型变体及资产、A2/A3 实际软件栈、复现命令、输出结构、A/V 检查、已知限制与排障。实验记录包含 Recipes/Cosmos 提交号、checkpoint/VAE/输入哈希、两平台完整环境、六个场景命令与逐次结果、异常用例、输出检查、精度对比、端到端耗时、Host/NPU 内存和 profiling 数据；大文件仅保留可追溯的脱敏地址和哈希。
- PR 内容仅包含一个 Cosmos3 场景样例适配；关联设计 RFC/Issue，并在 PR 描述中逐项对应本设计文档验收表。

## 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 重建环境或变更依赖后偏离已验证组合 | torch_npu/CANN ABI、Python 轮子或编码依赖不兼容 | 复用已归档双平台矩阵；变更后重新采集运行时信息并验证，历史结果不自动继承 |
| AVAE 初始化加载与按需加载要求不一致 | 混合调用通过不能证明该要求满足 | 如实披露并请求维护者决策；需要修改生命周期时先更新方案再验证 |
| A2 V2V 首轮性能临界失败 | 不能仅用一次通过复测宣布最终验收通过 | 同时保留两轮逐次耗时、原失败和复测，等待口径及结果审核 |
| Nano 的模式支持或固定 revision 与任务资产要求不匹配 | 不能以该 checkpoint 验收 T2I/I2I/T2V+Sound | 先核对 pinned source 与权重配置；模式不符时停下并提交 RFC 更新，不用 Edge 替代声音模型 |
| tokenizer/VAE 需联网或不在本地 checkpoint | 离线运行失败或误用不同资产 | 本地路径预检，固定 revision 和 SHA256；缺少资源时明确失败 |
| CANN attention、流或后处理差异影响旧场景 | 精度、性能或同步回归 | 改动前在两个平台保存基线；集中修改 overlay，旧场景逐项复测 |
| 音频编码、轨道封装和视频时钟精度不同 | 输出无声、解码失败或 A/V 不同步 | 先锁编码依赖和容差；使用探测工具检查轨道、采样率、声道、时长并实际播放 |

## 计划阶段与交付件

| 阶段 | 计划 | 阶段出口 |
| --- | --- | --- |
| 设计评审与基线锁定 | 提交 RFC/Issue 讨论并锁定 License、上游提交、模型/资产 revision、A2/A3 软件栈及固定输入；实现 PR 关联 RFC 并包含本设计稿 | 已达成方案共识，形成可复现环境矩阵 |
| 改动前实测 | A2/A3 跑 T2V、I2V、V2V 基线并归档日志、输出、耗时、内存 | 六场景输入/权重哈希和旧场景平台基线齐全 |
| 最小实现与单场景验证 | 增加输入示例、必要兼容、前置错误检查和 smoke；先逐一验证新模式 | 三种新模式在每个平台产出满足结构检查的制品 |
| 回归与工程检查 | 完成精度、性能、内存、异常及五次混合调用；运行 pre-commit 与 CI | 第三章验收项逐项有日志或报告证据 |
| PR 与任务验收 | 提交单一 Cosmos3 样例代码 PR，关联 RFC，整理自验材料并按任务平台申请验收 | 代码 PR、A2/A3 实测记录和任务验收申请均可追溯 |

社区任务截止日期为 2026-11-01（沿用任务计划，以平台安排为准）。上述表格是阶段出口要求，不代表批准记录；当前证据采集已开展，但设计审核和两项验收差异尚未闭环。后续若维护者要求补验，应重新连接对应环境；本次纯文档提交不启动模型或性能重跑。

## CheckList 适用性对照

| 检查类别 | 本任务对应内容 | 适用性说明 |
| --- | --- | --- |
| 提交位置、标题、CLA、CI | competitions 的 10-19 任务目录；PR 使用“【社区任务】...设计文档”标题，提交后单独检查 CLA 并评论 `/compile` | Recipes 的 CI 不替代本设计 PR 的 CI；审核通过以维护者明确意见为准 |
| 需求来源、标杆实现 | 本文需求背景、固定 Cosmos 提交、上游输入与默认参数文件 | 本任务是模型样例适配，不是将某个 TBE 算子改写为 Ascend C |
| 外部依赖、内部模块 | 资产基线、双平台环境、overlay/patch、CANN attention、输入与音频链路 | 对应模型任务要求，列出固定版本和错误处理 |
| TBE 信息库、ACLNN 原型 | 不适用 | 无新增 TBE/ACLNN 算子接口；统一调用 Python 推理入口并复用上游模式 |
| 分核、LocalMemory、TilingKey、Ascend C kernel | 不适用 | 不新增自定义 Ascend C kernel，不重写 CANN attention；不能虚构分核或 UB 公式 |
| 总体流程、差异及约束 | 数据流图、AVAE 生命周期、适配范围、六场景回归及风险表 | AVAE 与性能口径单列待审，不以“复用上游”为由免除任务要求 |
| 硬件、交叉特性、精度和性能 | A2/A3 矩阵；默认无声路径、多卡参数、共享 attention 兼容性；固定输入、原始张量、媒体和耗时检查 | 不优化 CP/CFGP/FSDP；扩大共享层修改范围前须追加影响分析和回归 |

## 固定证据与复现索引

以下链接均指向已有实现提交，不依赖本设计目录存在同名文件；原始大体积日志、权重和媒体不随设计文档提交。

- [复现命令及六场景示例](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/world_model/cosmos3/README.md)
- [输入 JSON 目录](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/tree/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/world_model/cosmos3/examples/inputs)
- [双平台实测记录、环境和测量方法](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/validation.md)
- [资产 SHA256 清单](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/assets.sha256.json) / [输入 SHA256 清单](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/inputs.sha256.json)
- [A2 原验收及固定复测摘要](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/a2-acceptance-summary.json) / [A3 原验收摘要](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/a3-acceptance-summary.json)
- [A2 既有产物复核](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/a2-output-recheck-20261016.json) / [A3 既有产物复核](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/docs/world_model/cosmos3/a3-output-recheck-20261016.json)
- [性能比较工具](https://gitcode.com/gcw_HNGPtE0T/cann-recipes-embodied-ai/blob/5d36522df2c9d08f587fef9ebe181e4b317a2e9c/world_model/cosmos3/tools/compare_runs.py)

`03d0a48` 版本的双平台结果为各 99 项测试通过、各 16 份既有产物复核通过；`5d36522` 的本地结果为 109 通过、2 跳过，官方 CI [#262](https://gitcode.com/cann/cann-recipes-embodied-ai/actions/runs/0f4e5c1ae7114784994c75410eb2c7d2) 通过。两组证据不混写：后者未重跑模型，也未重新生成前者报告中的校验器哈希。上述 CI 只证明对应实现提交的流水线状态，不代表本设计 PR 的 CI 或人工审核通过。

## 参考资料

1. [NVIDIA cosmos-framework 固定提交 a61b292](https://github.com/NVIDIA/cosmos-framework/tree/a61b292)
2. [上游 T2I 输入样例](https://github.com/NVIDIA/cosmos-framework/blob/a61b292/inputs/omni/t2i.json)
3. [上游 T2V+Sound 输入样例](https://github.com/NVIDIA/cosmos-framework/blob/a61b292/inputs/omni/t2vs.json)
4. [上游 image2image 默认参数](https://github.com/NVIDIA/cosmos-framework/blob/a61b292/cosmos_framework/inference/defaults/image2image/sample_args.json)
5. [现有 Cosmos3 昇腾样例](https://gitcode.com/cann/cann-recipes-embodied-ai/tree/db9b2453ad70ac919f8c42e02a61d16db63f4e66/world_model/cosmos3)
6. [仓库贡献指南](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/CONTRIBUTION.md)
7. [Cosmos3-Nano 候选 checkpoint 快照](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92)
8. [Wan2.2 VAE 候选来源](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e)

## 附录：版本修订记录

| 日期 | 版本 | 修改说明 |
| --- | --- | --- |
| 2026-10-14 | v0.1.0 | 根据社区任务书、设计文档模板、现有 Cosmos3 overlay 和固定 Cosmos 上游提交形成设计评审稿；未包含实机验证结论 |
| 2026-10-16 | v0.2.0 | 实现分支更新双平台环境、既有证据及 AVAE/A2 性能待审说明；未取得设计审核通过结论 |
| 2026-10-16 | v0.3.0 | 同步 Recipes 设计稿并补交 competitions 社区设计审核；修正耗时方向，说明既有计算口径与任务书解释待审，补充 CheckList 适用性及固定证据链接；未修改模型、原报告或阈值 |
