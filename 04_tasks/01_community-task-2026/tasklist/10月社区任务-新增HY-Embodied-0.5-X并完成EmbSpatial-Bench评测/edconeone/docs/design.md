# HY-Embodied-0.5-X NPU 接入与 EmbSpatial-Bench 评测设计说明书

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 10月社区任务：新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测 |
| 贡献者 / SIG | edconeone / Recipes SIG |
| 关联 RFC | [cann-recipes-embodied-ai #134](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/134)，待维护者评审 |
| 设计状态 | 提交评审；已完成 Atlas 800T A3 单图整模与独立 Attention 探针，完整验收仍待执行 |
| 设计仓 / 目标分支 | `cann/cann-ops-competitions`（原 `cann/cann-competitions`）/ `master` |
| 实现仓 / 计划目录 | `cann/cann-recipes-embodied-ai` / `manipulation/hy-embodied-0.5-x/infer_with_torch/` |
| 核心门槛 | 官方 human-verified evaluation 全量 3,640 题，正确数 ≥ 2,912，Accuracy ≥ 80.0% |
| 开发前出口 | Recipes RFC 明确平台、版本、License 与评测协议并达成维护者共识后，冻结正式验收配置；并行原型开发不代表方案已获批准 |

# 1. 需求背景（required）

## 1.1 需求来源与边界

依据随任务提供的《新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测》及 `test_cases/README.md`，新增标准化 NPU 图像加文本推理样例，支持 direct/thinking 输出并建立可独立复算的空间理解基线。设计遵循社区设计模板和 Recipes 贡献指南 [R1][R2]。

HY-Embodied-0.5-X 基于 MoT 架构，任务书给出的规模为总参数约 4B、每次激活约 2B。上游支持图像、视频和服务接口；本期验收范围为单图单轮推理、按样本顺序执行的批处理，以及 EmbSpatial-Bench 多选生成评测。视频、多图对话、训练/SFT、量化、多卡并行、机器人动作闭环和服务部署均不作为本期必要交付。可选服务只有复用同一推理/解析协议并单独测试后才开放。

首期保持新增样例内部实现，不修改公共 `module/`、其他样例或公共默认行为；确需抽取公共能力时，另列调用方清单并完成所有直接调用样例的功能与精度回归。若需要新增 CANN 算子，先独立提交并合入算子仓，再推进 Recipes 代码 PR。

## 1.2 已核验事实与参考版本

以下 SHA 为设计调查基线；分支移动不自动改变它们。代码交付时另记录实际实现 commit 和相对基线的 patch 哈希。

| 对象 | 固定版本 / 已核验情况 |
| --- | --- |
| 设计仓调查基线 | `83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469`；通用设计模板，不存在本任务专用占位目录 |
| Recipes 调查基线 | `db9b2453ad70ac919f8c42e02a61d16db63f4e66`；当前无本任务模型样例 |
| HY 官方源码 | `cf23678a76dc9c445ab7f7aea7a9a8f60e910a63`，与任务自验说明一致 [R3] |
| HY 定制 Transformers | `9293856c419762ebf98fbe2bd9440f9ce7069f1a`，由 HY 安装依赖锁定；不能替换为同版本号的 PyPI 包 [R4] |
| 权重 | `tencent/HY-Embodied-0.5-X`，revision `79f8301d8a80dd03ba778c58d6429e414e9ef5d3`；已下载并逐文件核对官方 Git blob/LFS 指纹，两个权重分片 SHA256 校验通过 [R5] |
| EmbSpatial 官方仓 | `75792fc5957b0142fab8ff1d47574ffcf73e02c9`，与任务自验说明一致；此版本提供说明而非可直接复用的评测程序 [R6] |
| 数据 | `Phineas476/EmbSpatial-Bench`，revision `e2733cacf0f20371518c687176e881d79644e61e`，仅使用 `embspatial_bench.json` [R7] |
| Bench 文件指纹 | 官方 LFS SHA256 `f9427218bb69afa218ca075e7efc1ca7a0a2379f87c0b55e4813ca19b9de0fa8`，大小 253,512,493 bytes；本地下载已再次核验 SHA256、3640 题及六类数量 |

该数据仓同时有 `embspatial_sft.json`；不得使用自动发现所有 JSON 的默认加载方式。数据卡所述 evaluation 文件与 HF 自动产生的 `train` 标签不是同一概念；本设计以固定 revision、精确文件名和内容哈希定义 human-verified evaluation split。

## 1.3 上游现状与差距

`HyEmbodiedPipeline.from_pretrained()` 以 BF16 加载 `AutoModelForImageTextToText`，使用 `trust_remote_code=False`，将模型移至指定设备并置为 eval。processor/chat template 由权重目录决定，`generate` 与 `generate_batch` 提供多模态输入契约。CLI 默认 thinking 关闭、temperature 0.05、max_new_tokens 32768；正式评测按本任务选择 thinking 开启。

关键差距在固定版本 Transformers 的 `modeling_hunyuan_vl_mot.py`：存在 FlashAttention 硬导入、MoT 专用 `_flash_attention_forward_mot`，视觉编码器也直接调用 FlashAttention。仅 `.to('npu')` 或设置 `attn_implementation='sdpa'` 不能覆盖这些调用。固定推理路径未发现 Liger 调用；最小推理环境排除训练专属 Liger/DeepSpeed 和 CUDA FlashAttention/Triton 依赖。包装层还需显式扩展 seed、top-p、dtype、逐题记录、断点恢复与错误分类。

License 存在元数据冲突：HY README/包 classifier 的 Apache 标注与根目录实际 `TENCENT HY COMMUNITY LICENSE AGREEMENT` 不一致，权重数据卡标为 `other`。以实际 LICENSE 文本逐项审查，保留原版权与许可；不得把上游模型/权重笼统标为 Apache-2.0。Bench 数据卡标注 CC-BY-4.0，但固定 GitHub 仓未提供独立 LICENSE 文件，不能据此给整仓源码推定相同许可。具体依赖、模型使用范围及补丁发布方式由维护者在 RFC 中确认。

# 2. 需求分析（required）

## 2.1 需求拆解与验收映射

| 编号 | 需求 | 设计落点 | 计划验证 / 交付证据 |
| --- | --- | --- | --- |
| RQ01 | 平台、环境、源码、权重、数据及协议锁定 | §3.1、§4 | RFC、lock manifest、SHA256、环境清单 |
| RQ02 | 单样本/批处理图文推理、direct/thinking | §3.2–3.4 | CLI、固定 smoke、输出格式测试 |
| RQ03 | 核心计算全程 NPU、无 CUDA 残留/CPU 静默回退 | §3.3、§6.2 | 张量设备断言、调用扫描、Profiler trace |
| RQ04 | 全量 3,640 题、无答案泄漏、六类统计 | §3.5 | 输入 manifest、逐题 JSONL、独立复算 |
| RQ05 | Accuracy ≥ 80%、异常计错、不得择优挑分 | §3.6–3.7 | ≥ 2,912 正确、错误清单、完整 run 索引 |
| RQ06 | 可恢复、无重复计分和跨样本状态污染 | §3.7、§6.1 | 崩溃/恢复测试、配置哈希校验、20 题两轮 |
| RQ07 | ≥ 10 smoke，每例 profiling ≥ 5 次 | §6.1–6.2 | 固定用例 manifest、55 次以上采集结果 |
| RQ08 | TTFT、吞吐、端到端与 NPU/Host 内存 | §6.2–6.3 | 原始计时、均值/P50/P95、峰值与内存曲线 |
| RQ09 | 100 题连续稳定、完整正式评测 | §6.3 | 100 题内存采样、完整正式运行记录 |
| RQ10 | README、优化过程、测试、CI及公共回归 | §3.2、§6.4–6.5 | 单样例代码 PR、文档、pre-commit/CI与实机结果 |

## 2.2 支持矩阵与明确限制

| 项目 | 本期协议 |
| --- | --- |
| 输入 | 每请求 1 张本地可解码图片与非空 UTF-8 prompt；正式评测使用原图、原问题和原顺序 4 个候选项 |
| 输出 | 模型原始 completion、独立最终答案、结束原因、token 数与耗时；推理文本不进入选择题解析器 |
| dtype | 模型主干/视觉编码/权重与 KV Cache 为 BF16；归一化、softmax 等保持上游要求的 FP32 累加后转回 |
| batch | 单请求 batch=1；批量接口顺序遍历输入列表、每题独立状态；向量化 batch>1 不进入首版正式评测 |
| 设备 | 指定 `npu:0`，单进程、单卡、单请求在途；禁止 `device_map=auto` 带来的 CPU offload |
| thinking | 单图功能支持 true/false；正式得分仅 generation + thinking=true，direct 另列功能结果 |
| 长度 | max_new_tokens=32768；输入长度及图像预处理由冻结配置控制，超限报错，不自动裁剪问题/选项 |
| 随机性 | 正式 run_seed=42；按 question_id 派生每题 seed，固定采样协议，不承诺跨设备逐 token 位相同 |
| 非法输入 | 缺图、坏图、空 prompt、选项数不为 4、重复 ID、错误版本均有显式错误；正式样本异常保留在分母中 |
| 并发/服务 | Pipeline 非线程安全；可选服务串行调用同一适配器、显式生命周期，不复用上一请求 KV Cache |

# 3. 详细设计（required）

## 3.1 RFC 配置冻结与运行前检查

本设计提出单逻辑 NPU 方案，供维护者确认。开发机已通过系统 DMI 与 `dmidecode -s system-product-name` 双来源确认公开型号 Atlas 800T A3。固定资源、隔离软件栈和单图整模已完成初步验证；完整容量、稳定性、全量精度及协议评审仍需分别取得证据，不能以初步验证代替最终验收。

| 参数 | 提案 / 锁定值 | 冻结条件 |
| --- | --- | --- |
| 平台/卡数 | Atlas 800T A3，1 个逻辑 NPU，HBM 64 GiB | 型号来自系统 DMI 双来源，容量来自设备工具；单逻辑设备方案与预算仍供 RFC 确认 |
| 实测软件栈 | aarch64、Python 3.13.13、CANN 8.5.0、torch 2.8.0+cpu、torch_npu 2.8.0.post4、torchvision 0.23.0；驱动 25.3.rc1.2、固件 7.8.0.6.201 | 隔离 venv 的依赖检查、实际 libhccl 来源及固定 HY/Transformers 来源验证通过；已跑单图 direct/thinking，不代表完整验收 |
| 模型依赖 | §1.2 固定 HY/Transformers SHA，加本样例可审查补丁 | 记录基线 SHA、补丁 SHA256、包安装来源；禁止自动升级 |
| 权重/processor | §1.2 固定 HF revision；模型、tokenizer、processor、chat template 同一 snapshot | 逐文件 SHA256、文件集合及 safetensors index 校验通过 |
| 评测协议 | generation；thinking=true；batch_size=1；use_cache=true | RFC 评审确认后写入 `protocol.lock.json` |
| 解码 | do_sample=true、temperature=0.05、top_p=1.0、top_k=0、num_beams=1、repetition_penalty=1.0、max_new_tokens=32768 | 显式覆盖继承默认值，导出完整有效 generation config；EOS/pad 来自冻结 tokenizer/config |
| seed / 超时 | run_seed=42；每题硬超时 1800 秒（从开始处理该题到结果完成持久化前） | 固定为本设计提案；若 smoke 表明不可行，在正式 run 前经 RFC 修订，而非中途修改 |
| 内存预算 | NPU 峰值 ≤ 64 GiB 的 90%，即 57.6 GiB；Host RSS 目标 ≤ 32 GiB，主机可用内存 ≥ 64 GiB | 属工程预算；目标设备变更必须重算，32768 token 容量测试通过后冻结 |

已使用独立任务 venv，未覆盖共享环境。原参考 XVLA 软件组合 [R8] 不作为本任务运行事实；本任务按上表实测组合冻结包版本和动态库来源。独立融合 Attention 的 15 项探针覆盖 BF16/FP16、D72、GQA、packed 段、8K prefill 与 32K KV decode，并在 Python 3.13.13 环境复验；32K 单算子探针不等同于完整模型容量验证。

`protocol.lock.json` 保存平台、精确软件构建版本、所有 revision/哈希、评测方式、样本顺序摘要、prompt、processor、生成参数、解析器版本、样例代码 commit/patch、超时与内存预算。规范化 JSON（UTF-8、键排序、固定数值表示）计算 SHA256；排除输出目录、日志时间等无关位置元数据，纳入所有影响模型输出/计分的参数。恢复必须比对该哈希及设备/环境信息。

预检按以下顺序执行：配置/schema → 环境与安装来源 → 权重/数据文件指纹 → 数据 ID/字段/数量/relation → NPU 可用性与容量 → 补丁生效与后端探针 → 启动推理。任何缺失或冲突均停止正式运行并输出错误，不能自动下载另一 revision、升级依赖、更换后端或改参数。源码扫描与 import 探针覆盖真实被加载文件路径，防止 editable 安装误用另一份代码。

## 3.2 目录、入口与模块职责

计划新增目录如下（下列脚本属于后续实现接口，当前设计 PR 不声明它们已可执行）：

```text
manipulation/hy-embodied-0.5-x/infer_with_torch/
├── README.md
├── requirements-npu.txt
├── configs/{inference.yaml,embspatial-generation.yaml}
├── patches/{hy-npu.patch,transformers-hunyuan-mot-npu.patch}
├── src/{pipeline_npu.py,attention_npu.py,environment.py}
├── src/{dataset.py,prompts.py,answer_parser.py,result_store.py}
├── infer.py
├── evaluate_embspatial.py
├── recompute_metrics.py
├── tests/
└── doc/{adaptation.md,experiments.md}
```

外部 snapshot、缓存图片、逐题原始输出与 profiler 制品放在用户指定目录，Git 中只放获取步骤、manifest/schema、小型合成测试与制品链接。复用上游 Pipeline/processor；补丁只覆盖设备、依赖与必要后端接口，不复制完整 HY/Transformers 源码。新能力成熟且有第二个实际调用方后再考虑进入公共 `module/`。

接口职责：`load_pipeline(config)` 一次加载模型/processor并返回单设备实例；`infer_one(image_path, prompt, config)` 返回 `raw_output/final_answer/finish_reason/token_counts/timing`；`infer_many(requests)` 保持输入顺序并逐题调用；`evaluate` 管理数据、事务日志和推理，不将标签传给模型；`recompute_metrics` 仅凭结果 JSONL 即可复算，另支持可选 manifest 加强审计，不加载模型。参数及输入归调用方所有，处理函数不就地修改源问题、候选项或图像文件。

拟交付命令示例：

```bash
python infer.py --config configs/inference.yaml \
  --model-path /path/to/HY-Embodied-0.5-X --device npu:0 --dtype bfloat16 \
  --image /path/to/image.jpg --prompt 'Describe the spatial relation.' \
  --max-new-tokens 32768 --temperature 0.05 --top-p 1.0 --seed 42 --enable-thinking

python evaluate_embspatial.py --config configs/embspatial-generation.yaml \
  --model-path /path/to/HY-Embodied-0.5-X \
  --dataset-path /path/to/embspatial_bench.json --mode generation \
  --enable-thinking --run-dir /path/to/results/run-001 --resume

python recompute_metrics.py --results /path/to/results/run-001/results.jsonl \
  --expected-count 3640

# 完整制品验收另增加源ID/标签核对（不影响上述独立复算能力）
python recompute_metrics.py --results /path/to/results/run-001/results.jsonl \
  --expected-count 3640 --manifest /path/to/results/run-001/manifest.json
```

CLI 与配置同时提供同一参数时，先显示合成后的有效配置再冻结；恢复运行不允许 CLI 覆盖冻结值。输出 JSONL 追加失败即停止，不可只在内存保留未落盘成绩。

## 3.3 NPU 模型接入与 Attention 方案

### 3.3.1 Host、模型与设备边界

Host 负责文件校验、图片解码、tokenization、prompt 构造、日志与计分。模型主干、视觉编码器、MoT 路由张量操作、Q/K/V 投影、RoPE、attention、MLP、KV Cache 与 lm head 均在 NPU 执行。允许 CPU 做输入控制与最终 token 文本解码；不允许把核心 matmul、softmax 或路由计算搬回 CPU 作为兼容办法。

显式导入 torch_npu 并选择设备；在 inference/no_grad 模式下运行，用 NPU BF16 autocast 包围需要的模型算术，保留上游 FP32 数值稳定步骤。逐模块检查参数、buffer、激活和 KV 的设备；整数 token/位置/索引保留整数类型，不能把所有输入统一转为 BF16。仅替换实际触达的 CUDA device/autocast/stream/synchronize/RNG 路径；训练专属 Liger 功能不进入推理依赖，若推理触达则提供与原数学语义一致的原生 torch NPU 路径。

首期无分布式依赖；检测到不匹配的 world_size/device 配置则拒绝启动。使用单个推理 stream，异步任务在读取结果、记录设备计时和释放请求资源前正确同步；取消、异常和超时不复用未结束的设备上下文。

### 3.3.2 三条 Attention 调用路径

| 调用点 | 设计处理 | 必须保持的语义 |
| --- | --- | --- |
| MoT `_flash_attention_forward_mot` | 小范围替换为 `attention_npu`；消除 flash_attn 硬导入和后端强绑定 | 文本/视觉 token 分段、路由回填顺序、padding、位置编码、causal 与 KV 长度 |
| ViT 普通 FlashAttention | 使用等价非因果 NPU attention | 图像内部 token 可见性、head/layout、scale |
| ViT 变长 FlashAttention | 根据累计序列边界执行分段 NPU attention | 不同图像/分块边界不互相 attention，恢复原输出顺序 |

先在 NPU 原生 torch 组合算子上建立小尺寸正确性 oracle，正式推理候选为 `torch_npu.npu_fused_infer_attention_score` [R10]。文本 head_dim=128，按实际有效长度采用经验证的 TND/BNSD 路径；TND 的累计长度接口与 FlashAttention 的 cu_seqlens（含起始0）不能直接互拷。视觉 head_dim=72，按图像段使用非因果 BNSD 路径；若所选接口需对齐，可在原72维完成Q/K norm后零填Q/K/V至80维，显式保留 scale=1/sqrt(72)，输出截取前72维。该代数等价候选必须真机对拍，不能直接套文本TND限制。`npu_fusion_attention` 的所查文档限定训练场景，不作为本设计默认推理接口。具体融合 API、layout 和支持矩阵以冻结 torch_npu 版本的真实探针为准。后端名称写入 lock，不在正式评测中自动回退。融合实现遇到不支持的 mask、dtype 或长度必须启动前报错；不能以 CPU 计算完成样本。

数学基线为 `softmax(QKᵀ / sqrt(d) + mask)V`。标准张量规范化为 `[B,H,S,D]`；GQA 的 Q/KV head 比例按 checkpoint 校验，在必要时按同组复制 K/V，禁止错误广播。冻结 checkpoint 为 16 个 Q head、4 个 KV head、head_dim=128；使用 config 实际值，不硬编码到通用后端。

带 KV Cache 的解码不是普通左上对齐三角 mask：设已缓存长度 P，当前 query 索引 i 可见 key 索引 `j <= P+i`，再叠加真实 padding/分段约束。prefill P=0 与 decode P>0 都验证；当 `S_q != S_kv` 时不得未经验证直接依赖后端 `is_causal=True` 的默认对齐。MoT 原实现先计算全序列因果 attention，再对 `v_seqlens=[s,e)` 的每个视觉段仅以该段 Q/K/V 做双向 attention，覆盖对应位置输出；视觉段不能被错误改成可访问全序列的双向 attention。保留 `_modality_mask_to_segments` 产物及特殊边界，不能只按 image token 连续区间重建。prefill 左padding、纯文本、多视觉段分别对拍；单token decode 使用已有KV全部有效前缀，与上游非因果调用等价。模态mask还控制独立文本/视觉QKV/O、norm/MLP路由，相关张量选择、回填必须留在NPU。

手写全量 `QKᵀ` 占用 O(S²)，仅用于短序列验证，不能作为 32768 token 正式路径。生产候选采用 NPU 上的分块/融合实现且保留同一语义，通过 §6.2 精度与容量门禁后冻结。新增自定义 Ascend C Kernel、Host tiling、UB 分块目前不在本设计范围，底层调度交由已验证 CANN/torch_npu 算子；若探针证实能力缺口，再提交独立算子方案评审。

### 3.3.3 Cache 与资源生命周期

模型及只读权重随进程复用，每题 KV Cache、路由索引、图像张量、生成 buffer 独立创建，结束后移除全部引用。保存完成的 CPU 字符串、标量和哈希，不在日志列表中保留 NPU Tensor。异常路径同样清理请求资源；设备进入错误状态时，父进程记录该题终态并重启 worker 后处理下一题。

KV 理论预算按 `B × 层数 × 2(K/V) × KV_heads × head_dim × 总序列长度 × 元素字节` 计算。固定 checkpoint 的 B=1、32 层、4 KV heads、128 head_dim、BF16、use_cla=false，对应每 token 65,536 bytes；仅 32,768 个 token 的 K/V 约 2 GiB，还须加输入 token、视觉特征、权重、激活、lm-head 输出及工作空间。以实测峰值决定是否满足 §3.1 预算，不能由约 8GB 权重大小推断整模可运行。

## 3.4 多模态预处理与输出合同

按同一 snapshot 的 processor/chat template 构建输入，保留图像和文本特殊 token。固定预处理读取到的参数为：RGB，bicubic（resample=3），rescale=1/255，mean/std 均 `[0.5,0.5,0.5]`，min_pixels=25088，max_pixels=4194304，patch_size=16，merge_size=2，temporal_patch_size=1。使用 `HunYuanVLMoTProcessor` / `Qwen2VLImageProcessorFast` 的实现，不自行增加裁剪、翻转、坐标提示或改 resize 规则。

processor/device 选择、预处理输出 shape/grid 和 token 数进入记录；图像转 RGB、resize 与 patch packing 在 CPU 预处理阶段可执行。超出 processor 或模型上下文限制时显式报错，不静默截断。模型回答坐标等内容属于一般推理输出，不得转换为空间题的候选答案提示。

模型仅解码新生成 token，不把原 prompt 混入 completion。保留 raw output 与必要的标签/结束标记，验证 tokenizer 的 `skip_special_tokens` 不会抹掉 `<think>/<answer>` 分隔信息；必要时保留 token 级边界再分别解码。固定 chat template 的 generation 前缀已包含 `<think>`，切除 prompt 后的 completion 可能没有开标签；解析器按冻结模板给定的初始 THINKING 状态读取续接内容，遇到 `</think>` 才转入最终答案状态，不能要求模型重复生成开标签。保存模板前缀摘要供审计。thinking=true 的解析器必须隔离思考区和最终答案区；标签缺失、未闭合、多答案、被长度截断均分类记录，不让推理文本中的 A/B/C/D 影响计分。

## 3.5 数据读取、顺序与无泄漏边界

仅打开 §1.2 精确 Bench 文件，并核验 SHA256；以原始数组顺序评测，不 shuffle、不按难度排序。数据中的图像字段为 base64，答案为 0-based 索引；实际字段为 `question_id/data_source/scene_id/question/relation/image/answer_options/answer/objects`。内部 `options` 只映射 `answer_options`，标签映射 `answer`，禁止使用 `objects` 的坐标等答案相关内容；不能把 base64 当作文件路径。启动前对字段类型和答案域 `[0,3]` 做完整校验。

若源数据存在唯一 question_id 则原样保存；若没有，生成 `embspatial:<dataset_sha256>:<zero_based_row_index>` 并记录 source_index，禁止运行时按问题文本去重导致少题。生成输入层只接收 image、question、options 和内部 ID；标签与 relation 交给独立评分层。relation 仅用于分组报表，不作为 prompt 附加提示。解码图片存入仓外内容寻址缓存，摘要绑定原始图像字节，日志不写 base64。

下表为任务自验说明给出的验收合同；固定 Bench 文件现已下载并逐行核验，实际 3,640 题与六类数量均匹配。每次运行仍须预检文件 SHA256 和完整计数，如不一致停止并请维护者核对，不把多出/缺失的数据人为补齐。

| relation | 全量样本数 |
| --- | ---: |
| close | 612 |
| far | 594 |
| above | 596 |
| under | 602 |
| left | 616 |
| right | 620 |
| 总计 | 3,640 |

固定问题模板如下，仅替换 question 与原始 options；答案映射 `A/B/C/D ↔ 0/1/2/3`，不排列重组候选项。以下为本设计拟定的生成输出合同，不冒充指定官方仓已有脚本的默认 prompt。

```text
{question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Select the correct option. Put exactly one letter (A, B, C, or D)
in the final <answer>...</answer> field.
```

thinking 开关通过上游 chat template 传入，不把标准答案、对象坐标、SFT 内容或人工标注加入消息。冻结模板全文及 UTF-8 哈希；generation 与 likelihood 属不同协议，首期拒绝 likelihood，若后续需要则先评审独立协议并完整重跑、单独报分。

## 3.6 严格答案解析与计分

正式协议选择 `<answer>` 标签合同：completion 中要求恰好一个完整且位于思考区外的 `<answer>...</answer>`，末尾只允许空白或已识别的模型结束 token。字段 trim 后必须严格匹配 `^[ABCD]$`。不接受小写、多个字母、`A or B`、选项文字或数字索引；不使用模糊正则在整段文本搜索。`<think>` 中出现答案标签也不能进入最终答案；畸形嵌套、多段答案和缺失标签均记 `parse_error`。direct 功能模式沿用最终标签合同并禁止混入 thinking 区；原始模型输出始终保留用于诊断。

推理状态与计分分开：正常结束且答案可解析时比较标准答案；`length_limit`（即使截断前出现字母）、OOM、timeout、missing_image、bad_image、runtime_error、interrupted_error、parse_error 均令 `is_correct=false`。单次正式 run 每题最多尝试一次，不按答案正确与否重试，不能人工修订结果。全量文件无法读取或全局 schema 错误属于预检失败，不能生成“完整评测通过”结论。

`Accuracy = 100 × Σ is_correct / 3640`，门槛为正确题数 ≥ 2912。每个 relation 单独输出 `correct/total/accuracy`；总分用样本级统计，不以六类 Accuracy 宏平均替代。异常率与正常可解析率另报，不能更改正式分母。

## 3.7 结果、事务与断点恢复

运行制品包括 `manifest.json`、只追加的 `events.jsonl`、规范化 `results.jsonl`、`summary.json`、`relation_metrics.csv` 和 `errors.jsonl`。每题结果至少包含：

```text
run_id, protocol_hash, dataset_revision, parser_version, enable_thinking,
initial_thinking_state, question_id, source_index, relation,
image_sha256, input_hash, raw_output, parsed_answer, ground_truth,
is_correct, status, finish_reason, sample_seed, input_tokens,
input_text_tokens, input_visual_tokens, output_tokens,
ttft_ms, e2e_ms, decode_ms, error_type, error_message
```

不可取得的 token/计时字段写 null 及原因，不伪造 0；summary 报对应有效样本数。运行manifest另存每题的原始ID、relation与ground truth摘要/映射，用于独立核对结果，模型worker只接收不含标签的输入manifest。ground_truth 只在推理完成后由评分层合并入结果，不能传入模型。完整 raw output 在仓外受控制品保留，仓内只存索引/摘要；若需截断面向人的日志，不得截断可复算制品。

父进程在推理前持久化并 fsync `STARTED(question_id)`，完成后持久化唯一终态；规范化结果以临时文件写完再原子替换。恢复时：

1. 核验同一 protocol hash、源码/权重/数据/环境及输入 manifest；不一致拒绝继续。
2. 已有终态（包括错误）全部跳过，不重新生成；只有从未开始的题可以执行。
3. STARTED 无终态的题记 `interrupted_error`，计错并继续后续题，避免崩溃成为隐式重试。
4. 同一 ID 出现冲突终态、未知 ID 或损坏记录时停止并保留证据，不使用 last-write-wins。尾部写入中断只能按事务日志的完整记录边界恢复，不能丢弃已完成终态。
5. 每题 seed 由 `SHA256(str(run_seed)+":"+question_id)` 前 8 bytes 大端转整数再对 `2^63-1` 取模，并设置 Python/NumPy/torch/NPU RNG；NumPy 使用该值对 `2^32` 取模。跳过前题不改变后题 seed。

父进程以固定 watchdog 实现硬超时，超时后标记终态、终止 worker、清理后重载模型；重载时间作为恢复开销单列。完成后要求 results 的 ID 集合等于 manifest 的 3640 个 ID，且 relation 数量一致。独立复算脚本只读 JSONL 即检查3640个唯一ID、六类分布及协议一致性，依据逐行保存的解析器版本/初始thinking状态重新解析 raw_output、复核 parsed_answer 与 is_correct 并重算分数；验收时再以可选 `--manifest` 核对完整源ID集合、标签与relation一致。不只信任结果中的布尔值。检测到遗漏/重复/伪造统计返回非零，未完成运行只能输出进度而不能宣称正式 Accuracy 达标。

多 seed/多轮完整实验各用独立 run_id，保留全部完成与未完成记录、配置差异、分数及均值/标准差。配置修订或重新尝试异常题只能开启一次新的完整 3640 题运行，并保留旧 run，禁止挑选最佳题目拼接结果。

# 4. 支持硬件

当前目标平台为已确认的 Atlas 800T A3，单逻辑 NPU、HBM 64 GiB，驱动/固件和软件栈见 §3.1。已完成单张真实 Bench 图片的 direct/thinking 生成及核心张量/32 层 KV 设备检查；不据此宣称全量精度、完整容量、无 CPU fallback 或其他平台通过。不同平台必须分别完整验证并记录独立结果。

若首选平台不能满足容量/后端能力要求，在开始正式验收前提交 RFC 修订，重新冻结平台、协议和参考结果。不得在运行中通过改成 CPU offload、缩小图片、减少输出长度或多卡拆分来掩盖容量问题。

# 5. 约束限制与风险处理

| 风险 | 检测方式 | 处理与出口 |
| --- | --- | --- |
| 定制 Transformers 与 NPU 软件不兼容 | 固定环境 import、模型加载、短序列真实 NPU 探针 | 保留完整错误和最小复现，按固定基线修补；未经评审不换版本 |
| MoT/ViT mask 或 cache 对齐错误 | prefilling/decode/变长/GQA 分层数值比较 | 按第一个产生偏差的层定位，不能以放宽解析规则修正 |
| NPU attention 不支持长序列或内存超限 | 真实输入极值、32768 输出容量测试及 profiler | 优化 NPU 后端；仍不满足则修订 RFC，保持原始失败记录 |
| thinking 输出缺失/标签被 tokenizer 丢弃 | token/原始字符串对照与解析单测 | 修复输出提取，正式 run 前冻结；不得借助 GT 选择答案 |
| 80% Accuracy 无法达到 | 同权重协议的参考 GPU 与 NPU 结果、六类错误分析 | 区分上游能力/协议/迁移偏差；公开实际分数，修复后完整重跑，不训练评测集 |
| License 元数据冲突 | 实际 LICENSE、权重卡与依赖清单对照 | 维护者确认使用/分发方式；保留补丁和原始许可，避免复制整仓 |
| 数据版本、类别或字段不一致 | SHA256、3640 ID、六类计数、schema 检查 | 启动前拒绝，核实数据源；不删除困难题或改变分母 |
| 隐式 CPU fallback | 核心 tensor 设备断言、算子 trace、dispatch 记录 | 核心计算一旦回退即失败，不作为有效 NPU 实验 |
| 状态泄漏与恢复后随机流漂移 | 20题两轮、逐题 seed、断电恢复与结果复算 | 保留请求局部资源与事务日志，拒绝配置不一致续跑 |

参考 CUDA 运行只用于迁移对照；CPU/mock 测试只能验证解析、计分和日志逻辑，不能证明 NPU 精度、性能或兼容性。

# 6. 可维可测分析

## 6.1 功能、自验与异常矩阵

冻结 smoke manifest 共 11 例：S01–S06 从固定 Bench 各 relation 按原顺序取第一题、thinking=true；S07–S08 取后续不同 ID、thinking=false（direct模式）；S09 缺失/损坏图像的合成请求；S10 使用单独测试配置触发输出长度上限；S11 用固定非法completion fixture触发 `parse_error`，只验证解析/记录链路，明确属于合成测试。选择仅根据 ID/relation/长度，不根据是否答对。smoke 不计入正式 3640 题结果，异常用例预期是正确记录错误，不要求产生合法答案。

| 测试组 | 覆盖与判定 |
| --- | --- |
| 加载与输入 | 固定权重/revision、错误版本快速失败；单图、顺序批处理、RGB预处理、非空prompt、空/坏图、4候选项 |
| 输出与解析 | thinking/direct；A/B/C/D 四种合法输出；数字/小写/空答案/乱码/多个字母/标签嵌套/think 内答案/截断均按合同处理 |
| 数值 | 视觉特征、MoT关键层、attention与lm-head输出无非有限值；mask 的合法负无穷不混同异常激活 |
| 数据隔离 | Prompt 快照只包含问题/选项/图片；标签、relation、坐标和SFT不可达模型调用参数 |
| 事务恢复 | STARTED前、STARTED后、终态落盘前后强制中断；已完成含错误题不重试；未知ID、重复终态、hash变化必须拒绝 |
| 错误记录 | 人为注入 OOM/timeout/服务或运行异常仅用于逻辑测试；另记录真实设备错误，二者证据分开 |
| 复算 | 合成小数据手工可知答案，检查微平均、六类统计、错误入分母和损坏JSONL；完整运行严格3640条 |

S01–S11 每个至少独立重复 5 次并采集 profiling；S09/S10/S11 的预期异常也保留完整 trace/退出记录，其无有效 TTFT 的情况标 N/A 及原因，不将其当作正常成功样本。正式 run 先预热固定有效请求，不用正式题目正确与否决定停止预热。

## 6.2 精度与性能标准

| 标准 | 执行方法 | 门槛 / 来源 |
| --- | --- | --- |
| 功能精度 | 全部 human-verified Bench 单样本生成 | 正确数 ≥2912/3640；任务书硬门槛 |
| 六类统计 | 固定 denominator，独立重算 | 六类各自 correct/total/Accuracy 齐全；无单类80%硬要求 |
| Attention 单元对齐 | 同Q/K/V/mask，短序列FP32 oracle；覆盖Sq=1、Sq≠Skv、prefill、padding、GQA、分段 | BF16工程诊断初始容差 `atol=rtol=2e-2`，同时报最大/均方误差；此为设计建议，不替代最终Accuracy验收 |
| 模型迁移对照 | 同snapshot/processor/prompt，reference GPU与NPU比较视觉特征、逐层输出、teacher-forced logits | 固定比较数据与第一偏差层；不要求采样自由生成逐token完全相同，差异需分析 |
| CPU回退检查 | 有效NPU样本trace + 核心模块设备断言；保留所有dispatch错误 | 核心计算不存在CPU静默回退，无触达CUDA专用路径 |
| 性能 | 全量逐题计时 + 每个smoke至少5次profile | 无强制加速比；完整记录均值、P50、P95、吞吐及样本数 |

定义统一计时边界：E2E 从 worker 接收题目开始（含解码/预处理、H2D、生成、答案提取及设备同步）到生成结果就绪，结果落盘与worker重启耗时另列；TTFT 从同一起点到首个新生成 token 可被 Host 观察，不能把整个 `generate()` 返回时间误报为 TTFT。实现通过不改变采样逻辑的生成回调/streamer 采集，禁止凭空估算。decode 耗时为首 token 到末 token，输出吞吐同时报告 `Σoutput_tokens/ΣE2E_seconds` 及纯 decode 的 `Σmax(output_tokens-1,0)/Σdecode_seconds`，后者仅统计至少2 token且计时有效的题。

端到端使用单调时钟，NPU计时前后明确同步；Profiler 采集与无Profiler的正式性能统计分开，保留采集开销说明。均值用全体有效观测，P50/P95 采用排序后 nearest-rank（下标 `ceil(p*n)-1`），不得仅取最快轮。失败样本单列状态与已观测E2E，并报告其数量；吞吐分母和有效样本集合写入summary，TTFT不存在时不把0混入分位数。

## 6.3 内存与稳定性

开启 `profile_memory`，分别记录模型加载后、预热后、正式评测期间的 NPU allocated/reserved 峰值与 Host RSS 峰值，附采样周期、进程范围和获取工具版本。NPU allocator 峰值不等于整卡占用；同时记录设备监测值，并纳入worker子进程/图片缓存的Host占用。Host RSS 以 1 Hz 和阶段边界采集完整进程树，包含 Profiler 解析子进程，报告实际采样间隔；离散采样峰值不视为连续绝对峰值。数据JSON采用流式读取或有界索引，图片按需解码与释放，不一次性缓存全部 base64/图片。

开发阶段 CPU+NPU 文本 Profiler 诊断的解析进程曾超过 32 GiB Host 预算，保留为不达标记录。每个 smoke 的五次验收采集改用 NPU activities、Level0、独立 DB 导出，仍开启 `profile_memory` 与 shape，不缩短原定生成预算；记录 NPU kernel/CANN 设备内存，allocator 峰值由 `torch.npu` 观测，Host 由上述全树采样器观测。CPU 框架轨迹与 CPU fallback 诊断使用独立 CPU+NPU 记录，不能用 NPU-only trace 独自证明无 CPU 回退。单例完整生成/解析探针已观测 8.74 GiB Host 采样峰值，55 次完整计划仍须逐项实测。

连续 100 题记录逐题结束同步后的 allocated/reserved/RSS及峰值，要求无OOM、无持续增长；分析长度差异和缓存warm-up，不把allocator保留内存直接当作泄漏。固定20题连续两轮，逐题比较相同输入的内存与状态，第二轮结束不得保留上一轮KV/图像/输出引用。检查引用释放与稳态占用，不用每题强行清空allocator掩盖泄漏。

完整 3640 题运行允许对单题异常按合同计错并继续，但必须无未处理异常、无长期内存增长；如出现真实OOM，即使Accuracy达标也须独立判定稳定性门槛未通过。修复后保留失败run，并按同一固定协议重新完整验证。

## 6.4 工程、兼容性与交付

后续代码 PR 执行 `pre-commit run --all-files`、仓库 CI、目标 Atlas 全量实机验证；只读文档检查不等同这些测试通过。保留独立环境，避免覆盖其他样例的 Transformers 或影响公共默认配置。若公共模块发生改动，列出全部直接调用样例并分别提供功能/精度回归；无公共改动时明确说明适用范围。

README 包含模型与License、已验证平台/版本、安装与补丁应用、权重数据获取、单图/批处理/评测/复算命令、结果与限制。`adaptation.md` 记录上游架构、三条attention替换、依赖冲突、精度定位；`experiments.md` 记录每轮环境、协议差异、六类错误分布、优化前后数据与失败实验。

| 交付件 | 内容 |
| --- | --- |
| 设计/RFC | 本文、维护者结论、平台与协议冻结记录；代码PR关联设计PR和RFC |
| 样例 | 依赖、配置、补丁、加载/预处理/推理/解析/错误处理入口 |
| 评测工具 | 精确数据revision、固定split、逐题JSONL、恢复、summary/CSV与独立复算 |
| 测试 | 11个固定smoke及每例5次以上profile、解析/计分/事务测试、100题/20题两轮稳定性 |
| 实验制品 | 精确环境与完整命令、源码/权重/数据/配置哈希、3640逐题记录、分组统计、延迟吞吐、内存与错误列表 |
| 制品索引 | 可访问的外部制品链接、内容哈希、生成命令与独立解包/复算步骤；不提交权重/数据/base64/大型raw日志 |

## 6.5 计划与评审确认点

建议周期四周：第1周确认RFC/License/目标平台，建立固定参考环境、补丁骨架与短序列oracle；第2周完成MoT/ViT NPU路径和单图smoke、容量门禁；第3周完成全量评测、恢复/内存/性能记录并定位误差；第4周复算、公共回归（如涉及）、文档与代码PR验收。以上为计划，无任何阶段被描述为已完成实机验收。

请维护者重点确认：① 已确认 Atlas 800T A3 的单逻辑 NPU 使用方案及内存预算；②实测 NPU 软件栈的验收适用性与上游许可处理；③ generation+thinking、严格标签prompt/解析、解码与1800秒超时合同；④新任务目录归属及代码仓RFC后续交付关系。上述项目确认后形成锁定记录；任何后续协议调整都有新版本和重新完整评测要求。

# 7. 参考资料

- [R1 社区任务流程与设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469/04_tasks/01_community-task-2026/README.md)。
- [R2 Recipes CONTRIBUTION.md](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/db9b2453ad70ac919f8c42e02a61d16db63f4e66/CONTRIBUTION.md)。
- [R3 HY 官方固定源码](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X/tree/cf23678a76dc9c445ab7f7aea7a9a8f60e910a63)，重点为 `src/hy_embodied/inference/pipeline.py`、`src/hy_embodied/cli/infer.py`、`pyproject.toml` 和 `LICENSE`。
- [R4 HY Transformers 安装依赖](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X/blob/cf23678a76dc9c445ab7f7aea7a9a8f60e910a63/pyproject.toml)。
- [R5 固定权重与配置](https://huggingface.co/tencent/HY-Embodied-0.5-X/tree/79f8301d8a80dd03ba778c58d6429e414e9ef5d3)。
- [R6 EmbSpatial-Bench 官方固定说明](https://github.com/mengfeidu/EmbSpatial-Bench/tree/75792fc5957b0142fab8ff1d47574ffcf73e02c9)。
- [R7 固定数据revision与数据卡](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench/tree/e2733cacf0f20371518c687176e881d79644e61e)。
- [R8 Recipes XVLA 环境参考](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/db9b2453ad70ac919f8c42e02a61d16db63f4e66/manipulation/xvla/train/README.md)，仅作为软件栈候选依据。
- [R9 EmbSpatial-Bench 论文](https://arxiv.org/abs/2406.05756)。
- [R10 torch_npu 推理融合Attention文档](https://gitcode.com/Ascend/op-plugin/blob/master/docs/context/torch_npu-npu_fused_infer_attention_score.md)，接口调用以冻结版本再次核验。
- [R11 固定Transformers模型实现](https://github.com/huggingface/transformers/blob/9293856c419762ebf98fbe2bd9440f9ce7069f1a/src/transformers/models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py)。

需求优先级为当前任务书与自验说明、维护者已确认RFC、目标仓现行贡献约定、固定上游源码/数据，最后才是相邻样例。本文的候选配置、工程诊断容差和资源预算均与任务硬门槛明确区分。
