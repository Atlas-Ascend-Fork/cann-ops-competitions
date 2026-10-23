# HY-Embodied-0.5-X 昇腾推理样例与 EmbSpatial-Bench 评测设计

依赖：本文的术语与取值来自任务书《10月社区任务-新增HY-Embodied-0.5-X并完成EmbSpatial-Bench评测》及其 `test_cases/README.md` 自验说明、本仓 `resources/design_template.md`，以及目标代码仓 `cann-recipes-embodied-ai` 当前提交下的 `README.md` 与 `CONTRIBUTION.md`。

与模板章节的对应：本任务交付的是模型推理样例与评测流程，不交付算子，因此模板"算子分析"下的数学公式、支持数据类型、支持形状三节，以及"算子实现"下的分核策略、数据分块与内存优化策略、tilingkey 规划、kernel 侧设计四节都不适用。模板中这些位置的内容由"适配点分析"与"实现方案"两节承担：前者说明昇腾适配必须替换的计算入口与计算位置的判定，后者说明目录划分、Host 侧与 NPU 侧设计、prompt 构造、答案解析与评测编排。模板的其余章节逐一对应。

# 需求背景（required）

## 需求来源

需求来自 CANN 社区任务 2026 的 10 月任务《10月社区任务-新增HY-Embodied-0.5-X并完成EmbSpatial-Bench评测》。任务要求在目标代码仓 `cann-recipes-embodied-ai` 中新增 HY-Embodied-0.5-X 的昇腾推理样例，并在锁定的评测协议下完成 EmbSpatial-Bench 全量评测，Accuracy 不低于 80.0%。方案已按 `CONTRIBUTION.md` 的要求提交 RFC，即该仓库的 Issue 162（https://gitcode.com/cann/cann-recipes-embodied-ai/issues/162 ），它记录与本文相同的版本与评测协议。

## 背景介绍

上游提供 Python 推理类 `HyEmbodiedPipeline` 与命令行入口 `python -m hy_embodied.cli.infer`，运行在 CUDA 上，模型结构在锁定提交下 transformers 自带的 `transformers/models/hunyuan_vl_mot/` 中。与昇腾适配相关的现状如下：

| 项 | 上游现状 | 对昇腾适配的影响 |
| --- | --- | --- |
| 模型结构的位置 | 结构代码在上述 transformers 模块中，权重目录中没有结构代码文件 | 加载模型与 config 都不需要 `trust_remote_code`，样例不执行权重目录中的代码 |
| 注意力 | `modeling_hunyuan_vl_mot` 在模块顶层执行 `from flash_attn import flash_attn_func, flash_attn_varlen_func`，语言模型与视觉编码器的 forward 直接调用这两个函数 | 该导入在模块加载时执行，设置 `attn_implementation` 不改变调用路径，因此需要在导入上游模型之前替换 `flash_attn` 模块 |
| 设备 | 推理接口接收设备参数 | 样例传入 `npu`，不使用 CUDA 设备、autocast 与 stream 接口 |
| PyTorch 版本 | `requirements.txt` 推荐 torch 2.10.0 | 该版本没有配套的 torch_npu，样例使用 torch_npu 配套的 torch 2.7.1 |
| tokenizers | 上游 `requirements.txt` 钉死 `tokenizers==0.22.2`，该版本也落在锁定的 transformers 提交在导入时校验的 0.22.0 到 0.23.0 区间内 | 样例沿用 0.22.2 |
| Liger Kernel、DeepSpeed | 出现在上游依赖列表中，只被训练代码调用 | 推理不调用，样例不安装 |
| 生成参数 | 上游 `GenerationConfig` 有 `max_new_tokens`、`temperature`、`use_cache`、`enable_thinking` 四项，其中影响采样的只有 `temperature`，没有 top-p、top-k 与 seed；权重自带的 `generation_config.json` 也不含采样项，因此未显式传入的采样参数取 transformers 的默认值 | 样例的生成参数对象在上游四项之外补 top-p、top-k 与 seed，前两者透传给 `generate`，使采样不依赖库的默认值；可复现性由样例在每题推理前重置随机种子实现 |
| 精度 | 权重以 BF16 发布 | 样例以 BF16 加载与推理 |

EmbSpatial-Bench 的官方仓在锁定的提交下只提供数据与指标定义，本设计固定输入模型的 prompt 模板，和从模型输出解析答案的方案，见"prompt 构造"与"答案解析"。

## 与目标代码仓现有样例的差距

本样例补齐目标代码仓中身多模态理解模型的样例和可复现的多选题评测流程。

# 需求分析（required）

## 需求描述

在 Atlas A2 单卡上新增 HY-Embodied-0.5-X 的昇腾推理样例，样例从本地权重完成图像加文本推理，支持 direct 与 thinking 两种输出格式；在固定的评测协议下对 EmbSpatial-Bench 人工核验评测集的全部 3640 题完成评测，Accuracy 不低于 80.0%，并交付可复现的配置、脚本、文档与实验记录。

## 需求拆解

| 编号 | 需求 | 对应的验收维度 |
| --- | --- | --- |
| 1 | 样例在 Atlas A2 单卡上从本地权重完成图像加文本推理，direct 与 thinking 输出格式正确 | 模型功能 |
| 2 | 视觉编码器、语言主干、MoT 模态路由、KV Cache 与 lm head 在 NPU 上执行，这些模块中的矩阵乘与注意力算子不回落 CPU；样例不使用 CUDA 设备与接口 | 模型功能 |
| 3 | "锁定的版本与评测协议"一节列出的取值全部固定在样例配置中，开发与验收期间不变 | 可复现 |
| 4 | 对 3640 题按数据集顺序分发给 6 个工作进程推理，谁空闲谁领下一题，每个工作进程 batch 为 1，每题使用一张对应图像、原问题与原始四个候选项；标准答案与对象坐标字段不进入模型输入 | 数据完整性 |
| 5 | 答案解析只接受固定的字母写法，不可解析、超时、OOM 与运行异常的题计为错误并保留在分母中 | 数据完整性 |
| 6 | `Accuracy = 正确题数 / 3640 × 100% >= 80.0%`，即正确题数不少于 2912；同时报告六类 relation 各自的正确数、总数与 Accuracy | 精度指标 |
| 7 | 逐题记录结果，中断后按 `question_id` 续跑，续跑不改变已完成的题与最终分母；全量评测期间内存不持续增长，不 OOM | 稳定性 |
| 8 | 交付固定 smoke 样本的功能与 profiling 结果、首 token 延迟、输出 token 吞吐、端到端耗时与 NPU/Host 峰值内存记录 | 稳定性、文档记录 |
| 9 | 交付样例代码、评测脚本、独立复算脚本、测试代码、README、适配与评测过程文档、实验记录；权重、数据集与图像不入库 | 文档记录、工程质量 |
| 10 | 公共代码的默认行为不变；通过 `pre-commit run --all-files`、仓库 CI 与 Atlas A2 实机验收 | 公共回归测试、工程质量 |

## 锁定的版本与评测协议

下表中除"峰值 HBM 目标"与"工作进程数"以外的取值固定在样例配置中，开发与验收期间不变。峰值 HBM 目标是本设计承诺的目标值，它在验收中的定位见"精度标准/性能标准"一节；工作进程数是按实测扩展性设定的运行参数，不参与配置哈希，续跑时可以与上一次不同，见"评测编排"与"配置哈希"两节。

| 项 | 锁定值 | 来源 |
| --- | --- | --- |
| 硬件 | Atlas A2，单卡 Ascend 910B3，64 GB HBM | 任务书允许 A2 或 A3，本样例取 A2 |
| `cann-recipes-embodied-ai` 基线提交 | `89fa8aec52b920acc351c9aa4586db7e20683092` | 本样例开发分支的起点，即提交 RFC 时 `master` 的头提交 |
| CANN、驱动 | 9.1.0、25.5.0 | 验收用实机的环境 |
| PyTorch、torch_npu、Python | 2.7.1、2.7.1.post4、3.11.4 | torch_npu 的配套版本 |
| transformers | 提交 `9293856c419762ebf98fbe2bd9440f9ce7069f1a`，版本号 4.57.0 | 上游 `requirements.txt` 指定 |
| tokenizers | 0.22.2 | 上游 `requirements.txt` 指定；该版本也在上述 transformers 提交校验的区间内 |
| HY-Embodied-0.5-X 代码 | 提交 `cf23678a76dc9c445ab7f7aea7a9a8f60e910a63` | 自验说明指定 |
| HY-Embodied-0.5-X 权重 | `tencent/HY-Embodied-0.5-X`，revision `79f8301d8a80dd03ba778c58d6429e414e9ef5d3`，文件 sha256 见下方"权重文件哈希"表 | 该 revision 下的权重文件 |
| 精度 | BF16 | 权重的发布精度 |
| 峰值 HBM 目标 | 单个工作进程不超过 32 GB；另报告整卡峰值（6 个工作进程合计） | 本设计自设的目标值 |
| EmbSpatial-Bench 代码 | 提交 `75792fc5957b0142fab8ff1d47574ffcf73e02c9` | 自验说明指定 |
| EmbSpatial-Bench 数据 | `Phineas476/EmbSpatial-Bench`，revision `e2733cacf0f20371518c687176e881d79644e61e`；评测只读取 `embspatial_bench.json`，其 sha256 为 `f9427218bb69afa218ca075e7efc1ca7a0a2379f87c0b55e4813ca19b9de0fa8`，共 3640 题，六类题数为 close 612、far 594、above 596、under 602、left 616、right 620 | 自验说明指定的 human-verified 评测集 |
| 评测方式 | generation，thinking 开启 | 任务书的默认建议 |
| 评测 split | human-verified evaluation，命令行中的标识符为 `human_verified_eval`。该数据集在 Hugging Face 上不划分 split，该 split 对应文件 `embspatial_bench.json` 的全部 3640 题；另一个文件 `embspatial_sft.json` 不读取 | 自验说明指定的 split 与该 revision 下的文件构成 |
| prompt 模板 | 见"prompt 构造"一节，全文固定在样例配置中 | EvalScope 的 EmbSpatial-Bench 文档中"样例示例"给出的实际模型输入，地址为 https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/emb_spatial_bench.html |
| 图像预处理 | `min_pixels` 25088，`max_pixels` 4194304，`patch_size` 16，`merge_size` 2，各通道均值与标准差均为 0.5，重采样方式 bicubic；样例不另做缩放或裁剪 | 权重自带 `preprocessor_config.json` 的默认值 |
| 生成参数 | temperature 0.05；`max_new_tokens` 8192 | temperature 取上游命令行的默认值；`max_new_tokens` 不取上游的 32768，依据见 RFC "协议决定与依据"第 4 条 |
| top-p | 1.0 | 上游 `GenerationConfig` 没有该项，但它调用的 `model.generate` 接受该参数；任务书要求 top-p 可显式设置，因此样例的配置与命令行都含该项并透传给 `generate`，锁定值取 `generate` 的默认值 1.0，即不截断候选分布 |
| top-k | 50 | temperature 0.05 下 `do_sample` 为真，top-k 参与采样。上游与权重自带的 `generation_config.json` 都不设该项，不显式传入时取 transformers 当前的默认值 50，该默认值可能随库版本变化，因此锁定为 50 并与 top-p 一同显式传入，参考侧与 NPU 侧取同一值 |
| 随机种子 | 每题推理前把 NPU 与 CPU 的随机种子重置为 1234 | 本设计定，用于让每题的结果不依赖它在评测中的位置 |
| 单题时限 | 3600 秒 | 本设计定，见"结果记录、断点续跑与异常处理"一节 |
| 工作进程数 | 6 | 单卡上并发的工作进程数，由实测扩展性决定：在与单进程相同的 256 题子集上，2、4、6 个工作进程分别取得 1.91×、3.62×、5.59× 的吞吐提升，每题的输出 token 与解析答案都与单进程逐题一致；6 个进程时整卡 HBM 峰值为 51.9 GB（共 64 GB），每个进程的 NPU 已分配内存峰值约 7.2 GB，全量评测耗时从单进程约 25 小时（在前 256 题上实测每题平均 25.1 秒）降到约 4.5 小时，因此取 6。每个工作进程仍 batch 为 1、每题前重置种子；不参与配置哈希，见"评测编排"与"配置哈希"两节 |
| 答案写法 | 只用字母体系，不用候选索引；接受 `A`、`(A)`、`[A]`、`A.`、`A)` 五种写法，字母只接受大写 A 到 D | 任务书要求在字母与候选索引之间取一种，本设计取字母，因为 prompt 本身要求模型输出字母 |
| 答案解析 | 见"答案解析"一节 | 任务书要求只解析 `<answer>` 或最终答案行 |

权重文件哈希：下表列出锁定 revision 下权重目录中除 `.gitattributes`、`README.md`、`LICENSE` 与 `video_preprocessor_config.json` 以外的全部文件，前三者不被加载，最后一个只在视频输入时使用，而本样例只输入单张图像。样例在加载模型前逐一校验这些文件的 sha256，任一不一致时报错退出。

| 文件 | sha256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `f78e77184ab4f3d53e28b85ffda8b328185cc40fcfaab59d17c3df0f2524621e` |
| `model-00002-of-00002.safetensors` | `a5ad94eb2fc98b41bd89b8cf33f446e86c472ccd15382c0f17a53028af90258d` |
| `model.safetensors.index.json` | `7d87e4a5b5f6da33a5689f9a0883fbeb663616b0616f620f975ca5f71c5a77c2` |
| `config.json` | `637d4da72b09e8b92af9e62185b0cdc7d172437ac73c5c790d5811f0bcfe9469` |
| `generation_config.json` | `e50d8075692c7b863faed5b8b0a79ec71e3f69098cb99d88aa608874b5c45734` |
| `preprocessor_config.json` | `1929c3a757543d069398007ad9033620f626f33ae552dd6e379a31cf00418e01` |
| `chat_template.jinja` | `c1bb404124c4b5b7ac4f4a6b6d2a5e8f3116d57aded58b9b38406384f1bafbfc` |
| `tokenizer.json` | `1ac285056de1f8bbc1145fac3078be7867c79d83a521c387335603da98244ed9` |
| `tokenizer_config.json` | `c3e74613a06954b66ef95ec7ca855704356d47da3faeb2417c857686f0930a0d` |
| `special_tokens_map.json` | `bb9f59990034dae326581b9c62471523975417869f78a244b7ae2ce8cbb085eb` |

其中 `chat_template.jinja` 决定对话渲染与 thinking 开关的展开方式，"答案解析"一节的第 1 步依赖它的行为；`preprocessor_config.json` 决定锁定表中"图像预处理"一行的取值；`tokenizer.json` 与 `tokenizer_config.json` 决定分词与特殊 token。这三类文件的改动都会改变模型输入，因此与权重文件同样纳入校验与配置哈希。

配置哈希的算法：取下列各项组成一个键固定的字典，按键名排序序列化为 UTF-8 的 JSON，分隔符不含多余空白，再取 sha256。

参与哈希的项：芯片型号；依赖版本（CANN、驱动、PyTorch、torch_npu、Python、tokenizers、transformers 提交号）；上游代码提交号；权重 revision 与"权重文件哈希"表中全部文件的 sha256；精度；数据集 revision、评测 split 与评测数据文件的 sha256；评测方式与 thinking 开关；prompt 模板全文；锁定表"图像预处理"一行列出的全部取值；生成参数，即 temperature、top-p、top-k、`max_new_tokens`、种子与单题时限；接受的答案写法；答案解析的规则版本号。

不参与哈希的项：`cann-recipes-embodied-ai` 基线提交、样例自身的实现代码、权重与数据的本地路径、输出文件路径、机器标识、峰值 HBM 目标、工作进程数——每个工作进程 batch 为 1，每题前重置种子，进程数不改变任何一题的输入与生成参数。因此在芯片型号与依赖版本相同的另一台机器上续跑，配置哈希不变；续跑时工作进程数也可以与上一次不同。

答案解析的规则版本号初值为 1，与下文"答案解析"一节的五个步骤对应。该节的规则任何改动都把版本号加 1，并在 README 的变更记录中写明改动的内容与原因；正式评测开始后不再改动。

配置哈希标识的是评测协议，样例自身的适配层代码不进入它。实现改动由另一个值标识：每次运行把样例代码所在仓库的 git commit 写入运行元数据，与配置哈希并列。续跑的门槛只有配置哈希，与自验说明一致，见“结果记录、断点续跑与异常处理”。

# 详细设计（required）

## 适配点分析

### 注意力调用路径

transformers 的 `modeling_hunyuan_vl_mot` 在模块顶层导入 `flash_attn_func` 与 `flash_attn_varlen_func`，并在语言模型与视觉编码器的 forward 中直接调用，所以这两个函数是昇腾适配必须替换的计算入口。替换方式是：在导入上游模型之前，把一个实现了这两个函数的模块对象注册到 `sys.modules["flash_attn"]`，使上游的顶层导入取到该模块。两个函数都用 `torch_npu.npu_fusion_attention` 实现，其中 `flash_attn_varlen_func` 按累积序列长度把变长批次拆成各条序列后调用。本样例不新增也不修改昇腾算子，`npu_fusion_attention` 是 torch_npu 提供的接口。

这种替换方式不修改上游 HY-Embodied-0.5-X 与 transformers 的任何文件，也不复制上游代码。

### CUDA 残留的核查

任务书要求关键模型计算无 CUDA 残留，核查范围是设备、autocast、FlashAttention、stream、随机数与分布式六项。在锁定的提交下逐项核查上游推理路径，即 `hy_embodied.inference.pipeline`、`hy_embodied.cli.infer` 与 transformers 的 `models/hunyuan_vl_mot/`，结论如下：

| 项 | 上游推理路径中的现状 | 本样例的处理 |
| --- | --- | --- |
| 设备 | `HyEmbodiedPipeline.from_pretrained` 与 CLI 的设备参数默认值是 `cuda`，但设备只作为字符串传给 `model.to(device)`；路径中没有 `torch.cuda.*` 调用 | 传入 `npu` |
| autocast | 只有旋转位置编码的 forward 调用 `torch.autocast`，其 `device_type` 取自输入张量的 `device.type`，在 NPU 上取到 `npu`，且 `enabled` 为 `False`；此外 `_flash_attention_forward_mot` 中有一处 `torch.get_autocast_gpu_dtype()`，它只在 query 为 float32 且外层开启了 autocast 时执行 | 前者在 NPU 上自洽，不适配；后者所在分支在 BF16 且不开启 autocast 时不进入，样例在注册的 `flash_attn` 替身入口断言 query 不是 float32，进入即报错退出，使该分支不会被静默执行 |
| FlashAttention | 见"注意力调用路径" | 用 `npu_fusion_attention` 替换 |
| stream | 推理路径中没有 `torch.cuda.Stream` 或 `Event`；上游 `cli/server.py` 中的 `stream` 指 HTTP 流式响应，与 CUDA stream 无关，且本样例不交付该服务 | 无需适配 |
| 随机数 | 上游推理路径不设置随机种子 | 样例用 `torch.manual_seed` 与 torch_npu 的 NPU 侧种子接口重置，不调用 `torch.cuda.manual_seed` |
| 分布式 | 推理路径不导入也不调用 `torch.distributed` | 无需适配 |

核查方式有两步，结果都写入实验记录：静态扫描在上述三处上游代码与样例代码中检索 `torch.cuda.`、`autocast`、`Stream`、`torch.distributed` 与 `flash_attn`，逐个命中点给出判定；运行时核查在全量评测中确认 profiler 采集到的算子不含 CUDA 后端，且进程未加载 CUDA 运行时库。上游后续提交若引入新的命中点，按"锁定的版本与评测协议"重新核查。

### 模态路由与计算位置

MoT 架构按 token 的模态选择参数，路由本身是张量索引与参数选择，没有自定义算子。视觉编码器、语言主干、MoT 模态路由、KV Cache 与 lm head 都在 NPU 上执行。判定方式有四项：

1. 样例导出各模块参数所在的设备。
2. 用 profiler 采集算子的设备归属，检查范围包含上述模块中的矩阵乘与注意力算子，也包含模态路由使用的索引与 gather 算子。
3. 检查推理过程中的 device 到 host 拷贝与强制同步调用，例如对 NPU 张量取 `.item()` 后用于 Python 分支判断。路由按模态掩码在 NPU 上完成，不把掩码取回 host。
4. 在生成过程中导出 KV Cache 张量的设备与 dtype。

矩阵乘沿用 PyTorch 的 `torch.matmul` 与 `nn.Linear` 调用，由 torch_npu 调度到昇腾算子，适配层只替换注意力的两个函数。profiler 那一项的检查范围按模块逐一列出：视觉编码器与语言主干每一层的 QKV 投影、注意力、FFN，lm head 的输出投影，以及 embedding 查找。

四项结果都写入实验记录。任一项显示核心计算在 CPU 上执行，或路由依赖 host 侧分支，都按缺陷处理：先修改适配层，再重新完整评测，然后才提交 PR。

### 评测数据与计分的隔离

数据完整性有两个风险点，分别对应两项设计：

1. 评测数据文件与 SFT 数据文件同在数据集目录下。评测入口只按配置中的文件名读取 `embspatial_bench.json`，并在运行前校验该文件的 sha256 与锁定值一致，一致后才开始推理；不一致时报错退出，不做任何推理。文件哈希写入运行元数据。
2. 模型输入只由问题、四个候选项与对应图像构成。下表逐字段列出样例对数据的用法。

`embspatial_bench.json` 是一个长度为 3640 的数组，每个元素的字段与本样例的用法如下：

| 字段 | 本样例的用法 |
| --- | --- |
| `question_id` | 结果文件的主键与续跑去重的依据，取值形如 `mp3d_0`，由数据集提供 |
| `question` | 填入 prompt 模板的 `{question}` |
| `answer_options` | 四个候选项，按数组顺序依次填入 `(A)` 到 `(D)` |
| `image` | 该题图像，以 base64 编码的 JPEG 内嵌在该字段中。样例在内存中解码后交给 processor，不写出图像文件，因此制品中不含图像数据 |
| `answer` | 标准答案，取值是 0 到 3 的下标，指向 `answer_options` 中的一项。只用于计分 |
| `relation` | 六类空间关系之一，用于分类统计 |
| `objects`、`scene_id`、`data_source` | 不读取。其中 `objects` 保存每个物体的名称与 bbox，即任务书所说的对象坐标字段 |

计分时把解析出的字母按 A 到 D 映射为 0 到 3 的下标，再与 `answer` 比较。结果行同时记录解析出的字母与映射后的下标。

计分与推理分开：推理过程只负责写出每题的原始输出，Accuracy 由独立的复算脚本从结果文件重新解析并计算，复算脚本不读取模型与权重。推理进程与复算脚本调用同一个解析模块，规则版本号也定义在该模块中，模块只依赖标准库。

复算脚本的输入只有逐题结果 JSONL 一个文件，与自验说明"只读取结果 JSONL 即可完成复算"一致。结果 JSONL 只含逐题结果行，正式评测下恰为 3640 行，不含元数据行；运行元数据写在同目录的 `run_meta.json` 中。复算脚本在算分之前校验三项：行数为 3640、`question_id` 唯一、六类 relation 的题数与锁定表一致；任一项不符就报错退出，不输出分数。校验通过后，它把总 Accuracy 与六类 relation 的正确数、总数、Accuracy 写成一份 JSON 与一份 CSV。

配置哈希的校验不由复算脚本承担，而分在另外两处：评测入口在续跑时比对 `run_meta.json` 中的配置哈希，见"结果记录、断点续跑与异常处理"；审计脚本在 PR 评审时比对 `run_meta.json` 与锁定配置，并核对结果 JSONL 的 sha256。这样复算脚本保持只读一个文件，而配置一致性仍有人核对。

计分规则中禁止的做法在实现上明文落到三条：解析失败、超时、OOM 与运行异常的题一律写入结果并计错，评测入口不对任何题重试；结果文件写入后，复算脚本与审计脚本都不接受对 `answer` 字段或解析结果的人工修改，任何修正只能以重跑整个固定协议的方式进行；任何题都不从分母中删除，分母固定为 3640。

## 实现方案

### 目录与模块划分

本样例占用目标代码仓的两个位置：

| 位置 | 内容 |
| --- | --- |
| `manipulation/hy-embodied-0.5-x/infer_with_torch/` | 适配层、单图推理入口、批量评测入口、独立复算脚本、依赖列表、NPU 运行配置、示例输入、EmbSpatial 评测配置，以及本样例的 `README.md`。配置中的路径使用占位符 |
| `manipulation/hy-embodied-0.5-x/infer_with_torch/tests/` | 本样例的测试代码，见"测试"一节 |
| `docs/manipulation/hy-embodied-0.5-x/infer_with_torch/` | 适配与评测过程文档、配图 |

样例目录内平铺放置代码与配置文件，文档放在顶层 `docs/<类别>/<模型>/<入口>/`。

本样例的 `README.md` 说明模型、平台、权重与 License、环境、安装、运行、数据获取、评测命令、指标、结果、性能与限制；`docs/` 下的过程文档说明上游架构、依赖替换、NPU 适配点、精度定位、评测协议、各 relation 的误差分析与适配前后的实验。

样例路径由任务书指定。仓库 README 对操作类模型的描述是输出动作序列，本模型输出文本，因此 PR 在模型入口表新增的那一行注明这是多模态理解模型；同时在"目录结构说明"中补上这两个路径。本样例不引入新的原子化优化能力，不改动能力入口表。

权重、数据集、图像与逐题结果文件都不提交。README 给出权重与数据的获取方式，并列出用于校验的 revision 与文件哈希。全量评测的逐题 JSONL 压缩后作为 RFC Issue 162 的附件上传，PR 中给出链接、压缩文件与解压后文件各自的 sha256，以及行数，供评审期间复算与审计。该文件含每题的原始输出，thinking 模式下体积较大；超过平台单附件上限时按 relation 分成六个文件分别上传，PR 中列出每个文件的 sha256 与行数。

### Host 侧设计

Host 侧指进程内的适配层与批量评测编排，职责分为四项：

1. **环境校验**。启动时读取 CANN、驱动、torch、torch_npu、Python、tokenizers 与 transformers 的版本，与锁定值比对；任一项不一致时报错退出，不自动安装也不升级。
2. **注意力模块注册**。在导入上游模型之前完成 `sys.modules["flash_attn"]` 的注册，注册后再导入 `HyEmbodiedPipeline`。顺序颠倒会让上游取到真实的 `flash_attn` 包，因此注册与导入在同一个函数内按固定顺序执行。
3. **模型加载**。设备为 `npu`，精度为 BF16，权重路径、processor 路径与 thinking 开关来自配置。
4. **评测编排**。主进程按数据集顺序把题目动态分发给 6 个常驻工作进程，谁空闲谁领下一题，每个工作进程 batch 为 1；每题推理前重置随机种子；每题结束后由完成该题的工作进程写一行结果（带 `worker` 编号）并采样 NPU 与 Host 内存，结果行按完成顺序写出。复算与续跑都按 `question_id` 去重与匹配，与结果行的写出顺序无关。

### NPU 侧执行设计

NPU 侧指模型在昇腾上的执行，设计点有三项：

1. **注意力**。`flash_attn_func` 按固定长度批次调用 `npu_fusion_attention`；`flash_attn_varlen_func` 先按累积序列长度切分，再逐条调用。两者的 head 数、缩放系数与因果掩码参数由调用方传入的张量形状与参数决定。
2. **KV Cache**。沿用 transformers 的 cache 实现，cache 张量随模型在 NPU 上分配。
3. **内存**。单题推理结束后释放该题的 cache 与图像张量。单条序列的 KV Cache 上限由 `max_new_tokens`、层数、KV head 数与 head 维度共同决定。

### 单图推理入口与上游 CLI 契约

单图推理入口是对上游 `HyEmbodiedPipeline` 的薄封装，命令行参数与上游 `python -m hy_embodied.cli.infer` 同名同义：`--model`、`--image`、`--prompt`、`--temperature`、`--max-new-tokens`、`--enable-thinking`、`--device`，其中 `--device` 的默认值改为 `npu`。在此之上补四项上游命令行没有的参数：任务书要求可显式设置的 `--top-p`、`--seed` 与 `--dtype`，以及锁定表中与 top-p 同样需要显式给出的 `--top-k`，默认值都取锁定表。上游的 `--batch` 不提供，原因见"约束限制"。入口在调用上游接口之前完成"适配点分析"中的注册与环境校验，因此其余行为与上游一致。

### 图像尺寸与坐标、回答格式

上游 `docs/inference.md` 规定：grounding 输出的坐标按每个轴归一化到 `(0, 1000)`，与输入图像的像素尺寸无关；thinking 模式把最终答案写在 `<answer>` 中，内容可能是含 bbox、point 或 trajectory 字段的 JSON。据此本样例的处理是：

1. **图像尺寸**。processor 按锁定表"图像预处理"一行的取值把图像缩放到 patch 对齐的尺寸，样例不另做缩放、裁剪或填充。坐标既然是归一化值，这一缩放不改变它的含义，样例也不对坐标做任何换算。
2. **回答格式**。单图入口原样返回 `<answer>` 内的文本，其中的坐标字段保持上游的回答格式；README 用一个 grounding 提示词的例子展示该形态。EmbSpatial-Bench 是多选题，评测路径上的答案解析只接受字母写法，见"答案解析"，因此评测中出现坐标类输出时按不可解析计错。

### 测试

测试代码放在样例目录的 `tests/` 下，用 pytest 组织，按是否需要 NPU 分为两组：不需要 NPU 的用例在仓库 CI 中运行，需要 NPU 的用例在 Atlas A2 实机上运行，两组的结果都写入实验记录。

| 用例 | 内容 | 需要 NPU |
| --- | --- | --- |
| 模型加载 | 校验权重文件哈希、按锁定依赖完成加载，断言各模块参数的设备与 dtype | 是 |
| 单图推理 | 用示例图像与提示词各跑一次，断言返回非空文本 | 是 |
| thinking/direct | 同一输入下两种模式各跑一次，断言 direct 输出不含 `<think>`、thinking 输出含闭合的 `<think>` 块与最终答案字段 | 是 |
| 答案解析的四种选项 | 构造解析为 A、B、C、D 的四组文本，断言解析结果正确，并覆盖锁定表中的五种写法 | 否 |
| 不可解析输出 | 小写字母、候选索引数字、空输出、多个 `<answer>`、未闭合 `<answer>`、未结束的 `<think>`，断言都判为不可解析并计错 | 否 |
| 重复 question | 含重复 `question_id` 的结果文件交给复算脚本，断言报错退出且不输出分数；把同一题重复交给评测入口，断言只计一次 | 否 |
| 缺图 | 图像字段为空、base64 不可解码、解码后不是图像三种输入，断言记为 `image_error` 并计错，评测继续 | 否 |
| OOM 与超时的记录 | 用桩替换推理调用，分别抛出显存不足与超时，断言记为 `oom`、`timeout`，进入异常清单且分母不变 | 否 |
| 真实数据 smoke | "smoke 样本"一节的固定样本集，断言每题产出完整结果行与预期的错误类型 | 是 |

### prompt 构造

每题送入模型的文本固定为下列格式，其中 `{question}` 是题目的问题，`(A)` 到 `(D)` 依次是数据集中原始四个候选项：

```
{question}
(A) {候选项 1}
(B) {候选项 2}
(C) {候选项 3}
(D) {候选项 4}
Answer with only the letter of the correct option. The last line of your response should be of the format: ANSWER: [LETTER] where LETTER is one of A, B, C, D.
```

该文本与该题的图像一起，按上游 chat template 组成对话，thinking 开启。图像按锁定表中的预处理参数处理，base64 解码后的位图在内存中直接交给 processor。

`apply_chat_template` 接到 `enable_thinking` 时会打印一条该参数不合法、将被忽略的警告，但该参数实际生效：开启与关闭渲染出的对话不同，token 数也不同。因此样例不以这条警告判断开关失效，也不把它当作缺陷；thinking 开关是否生效以渲染结果与输出中是否出现推理块为准，由 smoke 的格式检查确认。README 的常见问题中写明这一点。

### 答案解析

解析的输入是推理接口返回的新增生成文本，按以下步骤取答案：

1. 删除输出中闭合的 `<think>` 到 `</think>` 块。删除后文本中仍有 `<think>` 时，说明推理没有结束，判为不可解析。生成文本中还会出现没有配对开始标记的 `</think>`：权重自带的 `chat_template.jinja` 在 `add_generation_prompt` 为真且 `enable_thinking` 为真时，以 `<think>` 结尾，生成因此从推理内部开始，第一个 `</think>` 没有配对的开始标记。这种情况下 `</think>` 之前的文本仍是推理，因此连同该标记一起删除其之前的全部文本，使后续步骤读不到推理文本。文本中有多个未配对的 `</think>` 时，以其中最后一个为界，删除它与它之前的全部文本。
2. 文本中出现 `<answer>` 或 `</answer>` 时，必须恰有一个闭合的 `<answer>` 到 `</answer>`，取其内容；出现多个或未闭合时判为不可解析，不回退到最后一行。
3. 文本中既没有 `<answer>` 也没有 `</answer>` 时，取最后一个非空行；没有非空行时判为不可解析。
4. 取出的文本去掉首尾空白；以 `ANSWER:` 开头时去掉该前缀与其后的空白，前缀的比较不区分大小写。
5. 剩余文本恰为 `A`、`(A)`、`[A]`、`A.`、`A)` 之一时解析为对应选项；字母只接受大写 A 到 D，不做大小写转换。其余一律判为不可解析，包括小写字母、候选索引数字、多于一个字母，以及附带其他文字。

不可解析的题计为错误，仍计入 3640 的分母。

### smoke 样本

smoke 集合共 18 题，不少于自验说明要求的 10 个，由四组构成。选取规则只依据 `question_id`，因此结果唯一：取评测集中 `question_id` 以 `ai2thor_` 开头的题，按 `question_id` 的字典序排列，先按 close、far、above、under、left、right 的次序为六类 relation 各取最先的两题，组成 relation 覆盖组并保持这个次序；再依次为后面三组各取一题最先未被取用的 close 与一题最先未被取用的 far。选出的清单固定在样例目录的 smoke 配置文件中，并在 README 与实验记录中列出，选定后不再更换。

| 组 | 题数 | 运行方式与用途 |
| --- | --- | --- |
| relation 覆盖 | 12 | thinking 模式，按正式协议运行，六类空间关系各两题 |
| direct | 2 | direct 模式，只验证输出格式，不计分 |
| 缺图 | 2 | 把该题的 `image` 字段在内存中分别替换为空串与一段非图像字节，数据文件保持原样，评测入口对它的哈希校验照常通过；预期记为 `image_error` |
| 超长输出 | 2 | 把 `max_new_tokens` 临时置为 16，使输出在最终答案字段之前被截断；预期正常结束、判为不可解析并计错。锁定的 8192 下只有少数 thinking 不收敛的题达到上限，且事先无法指定是哪几题，截断场景由此构造；该改动只作用于这一组 smoke 样本，正式评测的配置哈希仍按锁定表计算 |

六项检查如下，任一项不通过都先修改实现再重跑 smoke，通过后才进入正式评测：

1. **NaN 与 Inf**。权重加载后遍历全部参数，断言都是有限值；smoke 运行时开启 `output_scores`，逐步对输出 logits 做有限性检查，出现非有限值即失败。该开关只在 smoke 阶段开启，正式评测的耗时与内存记录因此不含它的开销。
2. **乱码**。输出能按 UTF-8 解码，不含替换字符 `U+FFFD`，不含换行与制表以外的 C0 控制字符，且 ASCII 可打印字符与 CJK 字符合计占比不低于 99%；不满足时在报告中列出该题原文并复核。
3. **空答案**。按"答案解析"第 1 步删除推理文本后仍有非空文本。
4. **跨样本状态污染**。relation 覆盖组按正序与逆序各运行一次，逐题比较解析出的答案；不一致的题在报告中列出并复核原始输出。
5. **异常计错**。缺图组与超长输出组的结果行带有预期的错误类型，且评测在该题之后继续运行。
6. **格式**。direct 输出不含 `<think>`，thinking 输出含闭合的 `<think>` 块与最终答案字段。

性能采集按自验说明用 10 题：relation 覆盖组按上述次序的前 10 题，即 right 以外五类各两题，按正式协议以 thinking 模式运行，每题重复 5 次，统计量见"精度标准/性能标准"。

### 结果记录、断点续跑与异常处理

每题的结果写一行 JSONL，字段为 `question_id`、relation、执行该题的工作进程编号 `worker`、送入模型的 prompt、原始输出、解析出的字母、该字母映射后的下标、标准答案下标、是否正确、输入 token 数、输出 token 数、首 token 延迟、端到端耗时、错误类型。prompt 字段只记文本，不含图像的 base64 数据。结果文件中 `question_id` 唯一，结果行按完成顺序写出。

续跑按 `question_id` 去重：入口启动时读取已有结果文件，跳过其中已出现的 `question_id`，已完成的题不重复推理也不重复计分。续跑前比对配置哈希与同目录 `run_meta.json` 中记录的值，不一致时报错退出，要求重新完整评测；此时入口不覆盖也不追加写已有结果文件，报错信息要求先把该文件改名归档，再用新的输出路径重新开始，新文件与新的 `run_meta.json` 记录新的配置哈希。git commit 与已有运行元数据记录的不同时续跑照常进行，两个 commit 都写入运行元数据与实验记录，报告据此说明两段结果分别由哪个实现产出。续跑时工作进程数可以与上一次不同，因为它不参与配置哈希；实际进程数写入 `run_meta.json` 的 `workers` 字段。

某题写出带错误类型的结果行之后，续跑把它视为已完成而跳过，因此"评测数据与计分的隔离"一节的不重试规则在续跑路径上同样成立。重新运行该题的唯一方式是按"精度标准/性能标准"一节重跑整个固定协议。

每题结果行的错误类型取下表中的一个值：

| 错误类型 | 含义 | 输出是否交给解析规则 | 计分 |
| --- | --- | --- | --- |
| `none` | 推理正常完成且输出解析出一个选项 | 交给 | 按解析出的选项与标准答案比较 |
| `unparsable` | 推理正常完成，输出交给解析规则后判为不可解析 | 交给 | 错误 |
| `timeout` | 该题超过锁定的单题时限 | 不交给 | 错误 |
| `oom` | NPU 显存不足 | 不交给 | 错误 |
| `infer_error` | 图像已解码为位图之后，上游推理接口抛出异常，例如 processor 处理该位图失败或 chat template 渲染失败 | 不交给 | 错误 |
| `npu_error` | torch_npu 或 CANN 报出算子执行错误或运行时错误，且不是显存不足 | 不交给 | 错误 |
| `image_error` | 没有得到位图，即 base64 解码失败，或解码出的字节不是可解码的图像 | 不交给 | 错误 |
| `killed` | 执行该题的工作进程直接死亡，没有报出错误类型，例如段错误、驱动崩溃或被内核的 OOM killer 终止 | 不交给 | 错误 |

`timeout`、`oom`、`infer_error`、`npu_error`、`image_error`、`killed` 这六类合称异常，汇总为异常清单。锁定的评测集中每题都有对应图像，`image_error` 因此只在 smoke 的缺图样本中出现。

单题级的终止与恢复按下列机制实现。模型加载在 6 个常驻的工作进程中完成，主进程按数据集顺序把题目动态分发给这些工作进程，谁空闲谁领下一题，并按锁定的单题时限为每个工作进程当前所领的题分别设置截止时间。错误类型由两方分别判定：工作进程捕获 Python 层能捕获的异常，包括显存不足、上游接口抛出的异常与 torch_npu 报出的运行时错误，把对应的错误类型通过与主进程的通道报出后退出；主进程在某个工作进程的当前题超过截止时间时判为 `timeout`，在该工作进程没有报出错误类型就退出时判为 `killed`，后者对应 Python 层捕获不到的情形，例如段错误与驱动崩溃。两种情况下主进程都只终止出问题的那一个工作进程，为它当前那道题写入错误类型，重启它并重新加载权重，然后让它继续领下一题；其他工作进程不受影响，各自继续推理。主进程本身不加载模型。

输出达到 `max_new_tokens` 而正常结束的题按"答案解析"一节的规则解析，截断后通常没有最终答案字段，因而多数判为不可解析。

单题时限用来发现挂起的推理，取值要满足一个条件：一道正常跑满 8192 个输出 token 的题不被判为超时。smoke 阶段在正式评测之前完成，实测的单题耗时分布写入 README，用来确认 3600 秒满足该条件；若不满足，在正式评测开始之前修订该取值、重新计算配置哈希，并在 RFC Issue 162 中补记。正式评测开始后不再改动。

### 执行流程

| 步骤 | 命令或动作 | 产物 |
| --- | --- | --- |
| 1 | 安装依赖，按锁定的提交号安装 transformers | 环境 |
| 2 | 下载权重与数据集，校验 revision 与文件哈希 | 本地权重与 `embspatial_bench.json` |
| 3 | 按上游 README 的默认配置跑通参考实现并采集参考基线 | 参考输出，见"自验矩阵"的"参考实现基线"一行 |
| 4 | 单图推理，分别用 direct 与 thinking 模式 | 两种模式的输出，用于确认格式 |
| 5 | 运行 smoke 样本的功能检查，再用其中 10 题各重复 5 次采集 profiling | smoke 结果、首 token 延迟与吞吐 |
| 6 | 运行全量评测 | 3640 行逐题 JSONL、`run_meta.json`、异常清单、内存曲线 |
| 7 | 运行独立复算脚本 | 总 Accuracy、六类统计 |
| 8 | 导出模块设备归属与 profiler 结果，执行 CUDA 残留的静态扫描与运行时核查 | 计算位置与无 CUDA 残留的证据 |
| 9 | 汇总实验记录 | "实验记录"一节列出的清单 |

评测入口的命令行按自验说明的接口约定，参数含义与下列示例一致：

```bash
python evaluate_embspatial.py \
  --model-path /path/to/HY-Embodied-0.5-X \
  --dataset-path /path/to/embspatial_bench.json \
  --dataset-revision e2733cacf0f20371518c687176e881d79644e61e \
  --split human_verified_eval \
  --output-dir results/ \
  --mode generation \
  --enable-thinking \
  --temperature 0.05 \
  --top-p 1.0 \
  --top-k 50 \
  --max-new-tokens 8192 \
  --seed 1234 \
  --resume
```

自验说明的示例用 `--output-path` 指一个结果文件，本样例改为 `--output-dir`，因为一次运行除逐题 JSONL 外还写出 `run_meta.json` 与异常清单，三者放在同一目录便于归档。`--dataset-revision` 与 `--split` 按交付件要求提供：入口用它们校验本地数据与锁定值一致，取值与锁定表不同时报错退出，不自动下载其他版本。各参数省略时取锁定表中的值。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 | √ |
| Atlas A3 | |

任务允许 Atlas A2 或 A3 单卡，本样例在 Atlas A2 单卡上开发与验收。

## 约束限制

1. 样例按单卡设计，不含分布式推理。
2. 任务书要求支持本地图像加 prompt 的批或单样本推理，本样例按单样本交付：单图入口一次处理一张图像；全量评测在单卡上由 6 个工作进程并发执行，每个工作进程仍按数据集顺序领题、batch 为 1。上游 `generate_batch` 的批处理不在交付范围内，原因是 thinking 模式的输出长度是长尾的，同一批次内不同长度的序列相互等待，使每题的耗时无法独立归因；并行改为多进程而不是 batch，每个进程仍是单请求，逐题的首 token 延迟与端到端耗时仍可逐题记录。
3. 精度评测只覆盖 thinking 模式的 generation 方式，不含 likelihood 方式。
4. 本样例只交付推理与评测，不交付训练与微调能力。
5. 任务书把 OpenAI 兼容服务列为可选项，本样例不交付它，README 的"服务"一节写明这一点。
6. 样例不承诺与上游 CUDA 实现逐位一致的输出，原因是昇腾算子的浮点累加顺序可能带来微小差异。因此跨轮与跨样本的检查不以输出逐位相同为门槛，答案与原始文本的差异在报告中列出并复核，判定方式见"内存与稳定性"与"smoke 样本"。
7. 依赖的 License：样例安装的运行期依赖及其 License 如下，取自各包在 PyPI 上的发行元数据：torch_npu、torchvision 为 BSD，PyTorch 的元数据给出 Apache-2.0、BSD-2-Clause、BSD-3-Clause、BSL-1.0、MIT 的组合，NumPy 为 BSD-3-Clause 与 0BSD、MIT、Zlib、CC0-1.0 的组合；transformers、tokenizers、huggingface-hub、safetensors、accelerate、timm 为 Apache-2.0；einops 为 MIT；Pillow 为 MIT-CMU；Jinja2 为 BSD；regex 为 Apache-2.0 与 CNRI-Python；tqdm 为 MPL-2.0 与 MIT。这些 License 都与目标代码仓的 Apache-2.0 相容，且样例只在运行期引用它们。上游依赖列表中的 flash-attn、DeepSpeed、Liger Kernel 不安装。README 列出依赖清单与各自的 License，版本以样例的依赖文件为准；安装后按各发行包内的 LICENSE 文件逐一核对锁定版本的实际 License，核对结果写入实验记录。
8. 模型、数据与仓库的 License：HY-Embodied-0.5-X 的代码与权重采用 Tencent HY Community License，该协议不适用于欧盟、英国和韩国；EmbSpatial-Bench 数据采用 CC-BY-4.0；目标代码仓采用 Apache-2.0。本样例的开发与实机验收都在中国境内的昇腾算力环境上进行，不在该协议排除的地域内。样例代码全部新写，以运行时注册模块的方式适配上游，不包含上游代码片段，因此样例本身以 Apache-2.0 发布。README 写明这三种 License、该地域限制与各自的原文链接；同时检查仓库的 `Third_Party_Open_Source_Software_Notice`，需要补充时在同一 PR 中更新。提交 PR 前完成 CLA 签署。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 全部 3640 题上 `Accuracy = 正确题数 / 3640 × 100% >= 80.0%`，即正确题数不少于 2912；总分按这个样本级公式计算，不用六类 Accuracy 的算术平均替代；同时报告六类 relation 各自的正确数、总数与 Accuracy。Accuracy 只用 thinking 模式的结果，direct 模式只在 smoke 样本中验证输出格式 | 任务书第 3 章与自验说明"精度测试要求" |
| 性能标准 | 首 token 延迟、端到端耗时的平均值、P50、P95 与输出 token 吞吐的基线，来自单进程 smoke profiling（10 题各重复 5 次）采集；全量评测仍逐题记录这几项，但它们是在 6 个工作进程并发下测得的，报告中注明并发数，只作参考，不作为性能基线。本任务不要求相对上游实现的性能提升 | 自验说明"性能测试" |
| 内存标准 | NPU 与 Host 峰值内存可记录，全量评测期间不 OOM，内存不持续单调增长 | 任务书第 3 章与自验说明"存储测试" |

峰值 HBM 目标 32 GB 指单个工作进程，不是验收门槛：任务书的验收标准表对内存只要求不 OOM、无持续增长且记录齐全，该要求针对整卡。单个工作进程实测峰值超过 32 GB 时在报告中给出实测值与原因，不阻塞合入；另报告整卡峰值（6 个工作进程合计），整卡出现 OOM 或内存持续增长，则不满足内存标准。

正式评测在代码与配置冻结后至少跑一次。任务书允许多次运行，也允许用完整重跑同一固定协议的方式替换含异常样本的那一次，因此重跑不受限制：既可以在发现实现缺陷并改动代码后重跑，也可以在代码与配置都未改动时重跑同一配置。每次重跑在实验记录中写明触发它的原因，即缺陷与对应改动、异常样本，或只是重复运行。每一次全量评测的结果都记入实验记录并在 PR 中列出，汇总给出每次的总 Accuracy、六类 Accuracy 以及跨次的均值与极差，以最后一次作为验收结论。

性能指标：首 token 延迟指从调用推理接口到生成第一个输出 token 的时间，包含图像预处理；端到端耗时指从调用到最后一个 token 生成完毕的时间；输出 token 吞吐按 batch 为 1 的单请求计算，即 `(输出 token 数 - 1) / (端到端耗时 - 首 token 延迟)`，该定义按单个工作进程内的单次请求计算，不因进程数 6 增减而改变；多个工作进程并发时，各进程仍各自独立计时，但彼此共用同一张卡，计时结果可能受其他进程竞争的影响，因此全量评测期间测得的值只作参考，不作为性能基线。

## 自验矩阵

本表的"对应验收维度"列指向任务书第 3 章验收标准表中的维度，PR 中的自验结果按本表逐行给出，从而与第 3 章逐项对应。

| 验证项 | 对应验收维度 | 方法 | 通过依据 |
| --- | --- | --- | --- |
| 参考实现基线 | 模型功能、可复现 | 按上游 README 的默认配置跑通单图推理，用相同权重、图片、prompt 与生成参数在 NPU 上采集首份基线；参考侧与 NPU 侧都按锁定表显式传入 top-p 与 top-k，并同样在每题生成前重置一次随机种子，否则两侧的采样分布不可比。参考侧在能取得 CUDA GPU 时在 GPU 上运行并附对照；本任务的算力来自昇腾在线开发环境，不含 CUDA GPU，取不到时只采集 NPU 基线，并在报告中写明缺少 GPU 对照 | 已采集到完整的基线输出与配置记录；两侧答案的差异在报告中列出 |
| 模型加载 | 模型功能、可复现 | 校验"权重文件哈希"表中全部文件的 sha256 后加载模型，导出各模块参数的设备与 dtype | 哈希全部一致；加载成功；参数都在 NPU 上且为 BF16 |
| 单图推理 | 模型功能 | direct 与 thinking 各运行一次 | direct 输出不含 `<think>`；thinking 输出含闭合的 `<think>` 块与最终答案字段 |
| smoke 功能 | 稳定性、模型功能 | 按"smoke 样本"一节运行四组样本，并执行该节列出的六项检查 | 六项检查全部通过 |
| smoke profiling | 文档记录 | 按"smoke 样本"一节的 10 题各重复 5 次 | 每次都产出完整的耗时与 token 数记录，并汇总为性能标准一行列出的统计量 |
| 计算位置 | 模型功能 | 按"模态路由与计算位置"一节列出的判定执行 | 每一项都没有发现 CPU 上的核心计算与 host 侧的路由分支 |
| CUDA 残留 | 模型功能 | 按"CUDA 残留的核查"一节执行静态扫描与运行时核查 | 该节表中六项逐一有判定结论；运行时 profiler 采集到的算子不含 CUDA 后端，进程未加载 CUDA 运行时库 |
| 数据完整性 | 数据完整性 | 评测入口校验 `embspatial_bench.json` 的 sha256；复算脚本统计结果文件的题数与六类分布；另写一个检查脚本逐行读结果文件的 prompt 字段 | 哈希与锁定值一致；题数为 3640 且六类分布与锁定表一致；prompt 与锁定的模板按该题的 `question` 与四个候选项渲染出的文本逐字符相同，因而它只含题面与候选项，`answer` 下标、`objects` 的名称与 bbox 以及 SFT 数据都在其外。候选项文本含有正确选项的文本，它是题面的一部分 |
| 重复 question | 数据完整性 | 构造一个含重复 `question_id` 的结果文件交给复算脚本；另把同一题重复交给评测入口 | 复算脚本报错退出且不输出分数；评测入口按 `question_id` 跳过已完成的题，该题只计一次 |
| 答案解析 | 数据完整性、精度指标 | 固定样例覆盖四种选项、五种写法、无 `<answer>` 时取最后一行、多个 `<answer>`、未闭合 `<answer>`、游离 `</think>` 后出现字母、小写字母、候选索引数字、空输出 | 每个样例的解析结果与"答案解析"一节的规则一致 |
| 独立复算 | 精度指标 | 复算脚本只读取结果 JSONL，重新解析并计算 | 总 Accuracy 与六类统计与评测过程的输出一致，且 `Accuracy >= 80.0%` |
| 断点续跑 | 稳定性 | 评测中途终止后以 `--resume` 继续 | 已完成的 `question_id` 不重复出现；最终题数仍为 3640；配置哈希不一致时报错退出 |
| 异常处理 | 数据完整性、稳定性 | 注入缺图、超时与 OOM 三种情况 | 三种情况都计为错误、写入错误类型并进入异常清单，评测继续运行 |
| 内存与跨轮残留 | 稳定性 | 按"内存与稳定性"一节执行全量评测的内存采样、连续 100 题的单独结论与 20 样本两轮运行 | 该节列出的判据全部满足 |
| 实验记录 | 文档记录 | 按"实验记录"一节逐项核对清单 | 清单中每一项都有对应内容且可追溯 |
| 公共回归 | 公共回归测试 | 运行目标代码仓中直接调用模型的功能与精度测试 | 公共代码默认行为不变，测试通过 |
| 工程质量 | 工程质量 | 样例 `tests/` 全部用例、`pre-commit run --all-files`、仓库 CI、Atlas A2 实机完整评测 | 四项全部通过 |

## 内存与稳定性

内存采样覆盖全量评测：每题结束后各采样一次 NPU 与 Host 内存，两条曲线连同每次工作进程重启的位置一并记入实验记录。判据按自验说明与任务书的措辞取"不呈持续单调增长、不 OOM"，具体判定如下，三条都满足才算通过：

1. 全程不出现 OOM，也不出现 `killed`。
2. 每条曲线按"工作进程 × 重启段"分别切成若干段：先按采样所属的工作进程编号分组，再在组内按该进程的重启位置切段；每段内部单独判定，因为重启会重新加载权重并把内存读数拉回起点，不同工作进程之间或跨段比较都没有意义。段内把样本按顺序均分为前后两半，后半的中位数不超过前半的中位数加上权重占用的 1%。
3. 最长的那一段另做一次线性拟合，把斜率与它的 95% 置信区间记入报告。斜率显著为正时，结合第 2 条与该段的绝对增量复核，并在报告中说明；这一项只作为报告中的诊断量。

自验说明单独要求的"连续 100 题"改由一次单进程的 memory-100 运行单独给出，不取全量评测开头的连续 100 题；该运行按同样判据单独给出一次结论。profiler 开启 `profile_memory`，分别记录权重加载后、预热后与正式评测期间的 NPU 与 Host 峰值内存。

跨轮无残留的判据：同一批 20 个固定样本连续运行两轮，两轮之间复用同一个常驻工作进程，不重启它也不重新加载权重。自验说明要求的是"第二轮结束后不得残留上一轮的 KV Cache、图像或输出状态"，因此判据取第二轮结束后的状态：

1. 第二轮结束后的 NPU 已分配内存与第一轮开始前相差不超过权重占用的 1%，即两轮产生的 KV Cache 与图像张量都已释放。
2. 第二轮结束后导出常驻进程内的缓存对象，断言不持有上一题的 KV Cache、图像张量与输出文本。
3. 逐题比较两轮解析出的答案，不一致的题复核其原始输出，确认差异来自采样与浮点累加顺序，并连同原始文本的差异一并在报告中列出；跨轮残留的判定以第 1、2 条为准，依据见"约束限制"中关于输出一致性的那一条。

## 实验记录

实验记录是一份文档，放在 `docs/manipulation/hy-embodied-0.5-x/infer_with_torch/` 下，与适配过程文档并列，PR 中给出链接。体积大的制品不入库：逐题 JSONL 按"目录与模块划分"一节的办法作为 RFC Issue 162 的附件，profiler 原始数据同样以附件或可审计链接提供，实验记录中给出链接与 sha256。记录的内容按下表，与任务书交付件 7 和自验说明"输出结果"逐项对应。

| 项 | 内容 |
| --- | --- |
| 平台与卡数 | 芯片型号、单卡、驱动与固件版本、实际工作进程数（即 `run_meta.json` 的 `workers` 字段） |
| 提交号 | `cann-recipes-embodied-ai` 基线提交与本样例的实现 commit、HY-Embodied-0.5-X 上游提交、EmbSpatial-Bench 官方仓提交、transformers 提交 |
| 权重与数据 | 权重 revision 与"权重文件哈希"表全部文件的 sha256；数据集 revision、split 与 `embspatial_bench.json` 的 sha256 |
| 配置 | 锁定表的全部取值、配置哈希，以及计算该哈希所用的序列化输入 |
| 环境 | CANN、PyTorch、torch_npu、Python、tokenizers 版本与依赖清单，依赖 License 的核对结果 |
| 命令 | 每一步的完整命令行，与"执行流程"一节的步骤对应 |
| smoke | smoke 样本的 `question_id` 清单、每题结果与六项检查的结论 |
| 全量结果 | 逐题 JSONL 的链接、sha256 与行数；`run_meta.json`；总 Accuracy 与六类 relation 的正确数、总数、Accuracy |
| 性能 | 首 token 延迟与端到端耗时的平均值、P50、P95，输出 token 吞吐 |
| 内存 | 权重加载后、预热后与评测期间的 NPU/Host 峰值内存，单个工作进程的峰值 HBM 与整卡峰值 HBM（6 个工作进程合计），全量内存曲线与"内存与稳定性"一节各判据的结论 |
| 计算位置与 CUDA 残留 | 模块设备归属、profiler 结论、静态扫描的逐点判定 |
| 异常清单 | 每个异常样本的 `question_id`、错误类型与原始输出摘要 |
| 多次运行 | 每一次全量评测的触发原因与结果，跨次的均值与极差 |
| 复算命令 | 只读取逐题 JSONL 的复算命令及其输出 |

## 兼容性分析

本样例新增独立目录，不修改目标代码仓的公共代码，因此对仓库中其他样例没有影响。样例不涉及算子的新增或修改，因此不需要按 `CONTRIBUTION.md` 的要求先向算子仓提交变更。如 PR 评审过程中出现对公共代码的改动，则列出全部直接调用样例并附回归结果。

## 交付内容与计划

交付内容：样例代码与依赖文件；EmbSpatial-Bench 评测脚本与独立复算脚本；"测试"一节的测试代码；README；适配与评测过程文档；"实验记录"一节的实验记录；全量评测的总 Accuracy、六类统计、性能与内存记录；逐题结果以可审计的制品链接提供。PR 关联 RFC Issue 162，说明支持平台、改动影响与已知限制。


