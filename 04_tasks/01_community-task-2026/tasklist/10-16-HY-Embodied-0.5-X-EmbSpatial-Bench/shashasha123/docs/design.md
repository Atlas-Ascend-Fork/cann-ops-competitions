# HY-Embodied-0.5-X 模型接入与 EmbSpatial-Bench 评测设计

## 1. 需求背景与范围

任务来源为[10 月社区任务](https://www.hiascend.com/activities/task-center/details/3d47bc05b1c043ae935a009b1d452ba6?menu=guide)，官方任务目录为 `10-16-HY-Embodied-0.5-X-EmbSpatial-Bench`。目标是在 Atlas A3 单卡上提供 HY-Embodied-0.5-X 本地图像与文本推理样例，并完成官方 human-verified EmbSpatial-Bench 全量 3,640 题评测，样本级 Accuracy 不低于 80.0%（至少 2,912 题正确）。

开发仓为 [cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai)，新增目录为 `manipulation/hy-embodied-0.5-x/infer_with_torch/`，技术文档放在该样例的 `doc/`。设计 PR 提交至 `cann/cann-ops-competitions` 的 `master` 分支，本设计同时作为开发仓 RFC 的协议依据。

本任务交付单样本与顺序批处理 CLI、NPU 适配、独立评测与计分、稳定性及性能采集工具、可复现说明。不增加训练能力，不修改其他样例的默认行为，不要求部署可选的在线服务。通用算子模板中的 Host/Tiling 和 Kernel 章节不直接适用于本模型接入任务；注意力适配、资源约束和测试设计在下文说明。

## 2. 版本、平台与协议锁定

### 2.1 依赖基线

| 项目 | 锁定值 |
| --- | --- |
| 平台 | Atlas A3，单个 NPU 逻辑设备，`npu:0`；不启用多卡或 CPU offload |
| Recipes 起点 | `89fa8aec52b920acc351c9aa4586db7e20683092` |
| HY 上游 | `Tencent-Hunyuan/HY-Embodied-0.5-X`，`cf23678a76dc9c445ab7f7aea7a9a8f60e910a63` |
| Transformers | `9293856c419762ebf98fbe2bd9440f9ce7069f1a`，只对该版本施加可核对的最小补丁 |
| 权重 | `tencent/HY-Embodied-0.5-X`，revision `79f8301d8a80dd03ba778c58d6429e414e9ef5d3` |
| EmbSpatial 官方代码 | `mengfeidu/EmbSpatial-Bench`，`75792fc5957b0142fab8ff1d47574ffcf73e02c9` |
| 数据 | `Phineas476/EmbSpatial-Bench`，revision `e2733cacf0f20371518c687176e881d79644e61e`，仅 `embspatial_bench.json` |
| 数据文件 SHA-256 | `f9427218bb69afa218ca075e7efc1ca7a0a2379f87c0b55e4813ca19b9de0fa8` |
| Python / CANN | Python 3.12.13 / CANN 9.1.0 |
| PyTorch / torch_npu | PyTorch 2.10.0 / torch_npu 2.10.0.post7；安装和运行阶段核对匹配性 |
| 精度 | 模型权重与核心计算 BF16；按上游保留归一化、softmax 等必要的 FP32 中间计算 |
| 显存目标 | 单逻辑设备总 HBM 至少 64 GiB；进程峰值 allocated 目标不超过 48 GiB，reserved 不超过 56 GiB |

依赖安装使用独立虚拟环境。完整运行依赖及其版本须写入锁定文件；不安装 CUDA FlashAttention、训练用 Liger/DeepSpeed，不自动升级 Transformers。启动时核对 Python、框架版本、上游 commit、补丁状态、权重及 processor 文件清单与 SHA-256，任一不符立即报错。权重下载后的文件哈希构成运行身份，不以目录名称代替版本。

上述环境是设计配置，不代表兼容性测试结论。依赖适配如需变更，先更新 RFC 与配置版本，重新采集基线；不得沿用旧运行结果冒充新协议。平台、版本或模型身份未确认时禁止最终验收。

### 2.2 正式评测协议

- 方法固定为 generation，`enable_thinking=true`，单题单图、batch size 1；顺序使用原 JSON 数组顺序。
- 数据仅使用 human-verified evaluation 文件，禁止读取 `embspatial_sft.json`。题号须唯一，分布必须为 close 612、far 594、above 596、under 602、left 616、right 620，合计 3,640。
- 解码参数固定为 `max_new_tokens=32768`、`temperature=0.05`、`top_p=1.0`、`top_k=0`、`num_beams=1`、`do_sample=true`、`use_cache=true`、`repetition_penalty=1.0`，使用权重 tokenizer 定义的 EOS/PAD。
- 全局 seed 为 0；每题 seed 为 `SHA256("0:" + question_id)` 的前 8 个十六进制数对 `2**31` 取模。Python、NumPy、PyTorch 与 NPU RNG 在每题前重置，断点续跑不改变后续题的随机状态。
- 单题执行超时设为 600 秒，超时、OOM、NaN/Inf、运行异常、生成达到上限且未形成完整答案均记录为错误，不重试同一题到正确。
- 使用 revision 内的 processor/chat template。图像解码为 RGB，保持原图比例，由 processor 执行 bicubic resize、`patch_size=16`、`merge_size=2`、`temporal_patch_size=1`、`min_pixels=25088`、`max_pixels=4194304`、`rescale_factor=1/255`、mean/std 各为 `[0.5,0.5,0.5]`；不按答案裁剪图像或传入对象坐标。

提示词由白名单字段 `question`、`answer_options` 构造，选项原序保留，模板精确为：

```text
{question}
A. {answer_options[0]}
B. {answer_options[1]}
C. {answer_options[2]}
D. {answer_options[3]}
Select the best option. Put only its uppercase letter (A, B, C, or D) in the final <answer>...</answer> field.
```

通过上游 `build_messages` 构造一条用户消息：一张图像在前、以上文本在后。不给模型传入 relation、ground truth、objects、scene 标注或题号。题号只用于审计、顺序和 seed。

正式输出只接受一个完整 `<answer>...</answer>` 标签内严格匹配 `A`、`B`、`C`、`D` 的内容（允许首尾空白），或在闭合 `</think>` 之后最后一个非空行严格匹配 `Final answer: A` 等四种形式。若有多个答案标签、未闭合思考区、矛盾答案或其他形式，按不可解析计错。上游模板在输入末尾可能预填 `<think>`，解析时须明确区分提示前缀与生成内容，不能把思考区的选项字母当成答案。direct 模式仅用于功能测试，可接受裸单字母或一个完整 answer 标签；其结果不混入正式 thinking 总分。

## 3. 模型与 NPU 适配设计

### 3.1 数据流与边界

流程为：校验运行清单 → 加载 `HyEmbodiedPipeline` → 本地图像解码和 chat template → 视觉编码器 → MoT 路由及语言主干 → KV Cache 自回归生成 → 保存原始输出 → 独立解析计分。

复用上游 `HyEmbodiedPipeline` 的模型加载、消息构造和 processor 契约，不复制完整上游模型。以一个限域适配模块和版本受控补丁替换 CUDA 依赖。CLI 支持显式模型/processor 路径、device、dtype、最大 token、temperature、top-p、seed、thinking 开关及本地图片；批处理采用逐样本调用，保持单题状态可清理。正式评测参数由锁定配置提供，CLI 覆盖会生成新的配置身份，不能继续写入旧结果。

### 3.2 注意力语义

指定 Transformers 实现直接导入 `flash_attn_func` 和 `flash_attn_varlen_func`，不能只设置 `attn_implementation` 就认为完成适配。设计以 NPU 注意力适配器实现这两个调用所需的有限接口，并在指定文件中用显式补丁替换导入；不伪装或全局注入 CUDA 软件包。

保留上游 MoT 的两阶段计算：文本走因果注意力；视觉连续片段用片段内双向注意力覆盖对应结果。图像编码器使用双向注意力；decode 的 query 长度小于 KV 长度时保持上游非因果单步路径。Q/K/V 分组头数、缩放系数、变长累计偏移及输出形状必须保持一致。

基线适配器将变长序列累计长度传给原生 TND 路径，定长路径采用 BNSD 布局，通过 `torch_npu.npu_fusion_attention` 完成 BF16 计算；prefill 因果掩码与视觉无掩码路径分别处理。对不支持的 dropout、滑窗、张量布局及 dtype 显式报错。设备上保存 Q/K/V、掩码、注意力输出，CPU 仅接收长度等控制元数据。以独立 FP32 小规模数学参考对照注意力输出（CPU 参考仅用于数值测试，模型路径始终在 NPU），覆盖视觉片段、GQA、prefill、单 token decode 和可变序列长度，检查最大绝对误差和相对误差，并结合完整模型评测判断精度。

单 token、batch size 1 的纯文本推理，在 decoder 入口检查一次 modality mask 后直达文本投影与非因果 KV 注意力，省去每层重复的元数据同步和训练占位视觉分支。优化仅在 eval 且无梯度时启用，保留图像 prefill、多样本及训练原路径；以路由输出精确对比和同 seed 完整 token 对比检验等价性。

若融合算子约束导致无法保持原语义，可提供明确选择、全在 NPU 执行的分块数学参考后端用于定位；不得静默切换 CPU 或混用两种后端的评测结果。后端选择及实现版本纳入协议哈希，正式运行中不能更改。

### 3.3 设备、精度与生命周期

模型显式放到单个 `npu:0`，避免 `device_map=auto` 或按显存自动 offload。适配 NPU seed、同步、autocast 和峰值内存接口；不调用 CUDA stream/RNG。保持上游视觉/文本路由及 KV Cache 结构，不跨题复用 cache。

加载后逐项核对参数/缓冲区设备；对视觉编码器、MoT 主干与 lm head 安装抽查 hook，检查输入输出设备及有限值。结合 profiler 验证核心算子在 NPU 执行，CPU 图像解码、tokenization、控制元数据处理不冒充核心模型计算。实际回退警告必须定位，不能过滤后宣称无 CPU 回退。

正式评测与独立连续100题检查在每次新worker加载后，先使用既定10个smoke输入按相同生成配置完成预热。预热输出单独保存于warmup.jsonl且不计分；每次请求重新设置固定seed，正式评测仍按原序覆盖全部3640题，不使用预热答案替换任何正式结果。记录加载后、预热后和正式运行期间的内存，区分缓存建立与跨题状态残留。

每题结束释放输入、输出张量、KV Cache 和临时图像引用，记录同步后的 allocated/reserved 与 Host RSS；不通过每题清空 allocator 掩盖内存泄漏。单题等待满600秒即判timeout并通知生成流程取消；最多30秒返回宽限仅用于采集部分输出、TTFT、NPU/Host峰值和清理后内存，期间产生的EOS或正确答案仍计错，不延长答题得分窗口。streamer在CPU token回调入口响应取消，退出异常处理块后再释放输入/KV并测量结束内存，避免traceback保留张量；宽限后仍无响应则强制终止并明确标注缺测。OOM或超时结束后，后续题在干净进程继续，已失败题不重做。程序中断时，已开始而未完成的题记录为 interrupted 错误；仅尚未开始的题允许恢复执行。

## 4. 评测记录、续跑与独立复算

### 4.1 运行身份

配置按 UTF-8、键排序、固定 JSON 序列化计算 SHA-256。哈希覆盖平台与依赖版本、实现 commit、上游版本、权重/processor 文件哈希、数据哈希及顺序、prompt、图像参数、attention 后端、seed 规则、解码和解析规则。路径仅用于定位，身份由内容确定；不得因迁移目录改变语义，也不得省略内容哈希。

每次运行创建不可变 manifest 与追加写 JSONL，先写入并 fsync 题目开始事件，再记录结果。恢复前核对 manifest、数据题号全集和所有已有行的配置哈希，拒绝重复题号、冲突结果、未知题号及损坏行。尾部不完整写入保留到审计附件，不能静默丢弃已开始题的责任记录。同一输出目录用进程锁阻止并发写入。

### 4.2 每题与汇总输出

JSONL 保存 `question_id`、relation、原始生成文本、独立解析答案、ground truth、correct、status/error、input/output token 数、seed、TTFT、生成和端到端耗时、NPU allocated/reserved/peak、Host RSS/peak、配置哈希。输入图像只记录来源索引和 SHA-256，不把 base64 复制进日志。标准答案仅在模型推理结束后的计分侧使用。

独立复算程序无需模型或 NPU，仅读取 JSONL 和运行清单，重新解析 raw output，校验保存的解析结果、correct 与统计一致；总分固定为正确数 / 3640 × 100%，同时输出六类正确数、分母和 Accuracy，以及异常题清单。缺题导致完整性失败，不能缩小分母。每次完整运行均保留，若多 seed 或调参则报告每次协议及结果，不只展示最高分。

## 5. 验证方案与性能基线

| 项目 | 用例与判定 |
| --- | --- |
| 纯逻辑测试 | A/B/C/D、空输出、多答案、思考区含字母、未闭合标签、缺图、重复题、SFT/schema 错误、哈希变化、损坏日志、OOM/超时记录及续跑等 |
| NPU 功能 | 模型加载、上游 demo 单图推理、相同配置 NPU 基线、thinking/direct 输出、顺序多样本、设备 hook 与 NaN/Inf 检查 |
| 固定 smoke | 按原始顺序取各 relation 首题，再按原序补足 10 个不同真实样本；固定题号清单。另加 direct、缺图、不可解析、强制超长输出边界用例，不将人工边界结果计入 Accuracy |
| profiling | 10 个固定真实样本各重复至少 5 次，每次保留采集数据；开启 `profile_memory`，固定采集 prefill 与前两个 decode step，每次仍执行完整生成，记录采样边界、预热和重复序号，不只取最快一次 |
| 全量精度 | 3,640 题每题一条结果，至少 2,912 题正确；六类样本数与官方数据一致，所有异常计错 |
| 连续内存 | 连续 100 题记录 NPU/Host 内存；20 个固定样本连续两轮，检查两轮末端及逐题曲线，不残留上一轮状态，不出现持续单调增长或 OOM |
| 回归 | 独立样例不改公共代码；如必须修改公共模块，先枚举所有直接调用样例，再补全对应功能与精度回归 |
| 工程检查 | 对最终提交执行 `pre-commit run --all-files`，设计 PR 按仓库 workflow 评论 `compile`，记录真实 CI 结果 |

TTFT 从预处理开始到第一个输出 token 可用，并额外记录模型输入就绪至首 token 的区间，避免不同计时口径混合。以 streamer/逐步生成回调记录首 token；计时边界进行 NPU 同步。逐题记录端到端延迟、输入/输出 token，汇总均值、P50、P95。输出吞吐同时给出 token 总数 / 总生成时间及逐题统计，明确是否包含首 token。初始化、预热、profiler 与正式评测分开，不把采集开销混入无 profiler 基线。profiling 采用固定前三个生成 step 的 CPU/NPU 原始 trace 窗口，包含首轮视觉与语言计算及两个 KV decode step；逐样本完整生成的计时、token 和峰值内存仍全部保留。原始 trace 可作无损 gzip 压缩，以满足 ZIP 体积限制，报告须明确窗口与采集开销。

内存覆盖权重加载后、预热后、连续运行和完整评测阶段，记录 allocated/reserved、NPU 峰值以及进程树 Host RSS/峰值。对持续增长检查趋势和引用释放原因，不仅比较两个瞬时点。达到 token 上限、异常重启等事件单独标注。

## 6. 许可证与交付复现

上游 HY 代码标注 Apache-2.0，权重使用 Tencent HY Community License Agreement；权重不按 Apache-2.0 重新标注。EmbSpatial 数据卡标注 CC-BY-4.0。依赖许可证、上游来源、版本及修改说明纳入第三方说明；使用权重遵循其地域、使用和分发条款。仓库不提交权重、评测数据、base64 图片、缓存、大型原始输出或凭据。

实现、依赖安装、版本/文件校验、数据获取、推理、smoke、全量评测、profiling、内存测试、独立复算和报告生成入口均提交至验收完整 commit。验收包提供实际配置、固定样本索引、原始逐题结果、性能/内存输出及任务所需采集数据；《复现附件说明》区分可重跑输入与实际输出对照，逐项说明来源、相对路径、对应题号和命令。官方数据与权重按固定 revision 获取并校验，不依赖开发机目录。

自验证说明从全新检出指定完整 commit 开始，依次给出 CANN 环境、Python 虚拟环境、依赖安装、数据/权重准备、编译或补丁安装、全部自验、复算和报告生成命令。报告仅从本次运行输出生成；说明性人工字段与测量数据分开。交付前重新获取任务书及全部附件并核对变化，再从独立检出目录执行最终说明。

ZIP 控制在 100,000,000 字节以内，去除可重新下载的大型权重、重复日志和与验收无关文件；必交原始采集材料不得用截图代替。若必交材料仍超过限制，明确体积来源并请验收方确认处理方式。包内提供文件清单和 SHA-256 校验，不加入无关图片或视频。

## 7. 阶段安排与风险处理

阶段顺序为设计 PR 与 RFC 评议、基础环境准备、NPU 适配及参考核对、固定 smoke、完整精度/性能/内存自验、独立检出复现与材料整理。维护者评议、开发检查和验收状态分别记录；目标在任务页面截止日期前完成自验与交付，实际评议等待时间及评测耗时需保留余量。

主要风险是变长注意力语义、BF16 数值差异、长 thinking 生成耗时、依赖版本兼容以及显存增长。以小规模注意力对照、设备与有限值检查、固定 smoke、连续内存测试逐层定位。精度不足时从图像预处理、模板、attention、生成与解析逐项排查，禁止使用测试答案微调、手工纠正或删题提高分数。协议或依赖发生变化须更新 RFC 并重新完整评测；未达到阈值或未完成项如实报告。
