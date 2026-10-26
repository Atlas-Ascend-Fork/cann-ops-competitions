# 需求背景（required）

## 需求来源

本需求来自昇腾任务中心的 CANN 社区任务[“新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测”](https://www.hiascend.com/activities/task-center/details/3d47bc05b1c043ae935a009b1d452ba6)。基于 `cann-recipes-embodied-ai` 仓，在 Atlas A3 上新增可复现的昇腾推理样例，并完成 EmbSpatial-Bench 全量评测，达到 Accuracy ≥ 80.0%。

## 背景介绍

### HY-Embodied-0.5-X 推理接入

HY-Embodied-0.5-X 基于 MoT-2B 架构，总参数约 4B、单次激活约 2B，支持图像/视频输入，以及 direct、thinking 两种生成模式。本任务接入图像+文本推理，使用 EmbSpatial-Bench 验证空间理解能力。

上游代码仓为 [Tencent-Hunyuan/HY-Embodied-0.5-X](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X)，推理接口为 `src/hy_embodied/inference/` 中的 `HyEmbodiedPipeline`，模型实现位于指定 Transformers 提交的 `src/transformers/models/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py`。

### 上游实现现状分析

上游环境面向 CUDA。模型实现直接导入并调用 FlashAttention，视觉编码器和 MoT attention 都需要适配；该版本标记 `_supports_sdpa=False`，仅修改设备或配置 SDPA 无法完成迁移。上游安装脚本还包含 Liger 等依赖，需按实际推理调用链筛选。

目标仓尚未提供本模型样例。本设计复用上游 Pipeline、processor 和 chat template，通过局部封装与 patch 完成 NPU 适配。

### 模型功能分析

| 项目 | 内容 |
| --- | --- |
| 输入 | 本地图像与 prompt；评测输入包含原问题及四个候选项 |
| 输出 | direct 答案，或 thinking 的推理文本及最终答案 |
| 精度 | BF16 |
| 模型计算 | 视觉编码器、MoT、语言主干、KV Cache 和 lm head 在 NPU 执行 |
| 评测数据 | EmbSpatial-Bench human-verified evaluation，全量3,640题 |
| 评测结果 | 逐题JSONL、总分及六类空间关系统计 |

# 需求分析（required）

## 需求描述

在 `manipulation/hy-embodied-0.5-x/infer_with_torch/` 新增 Atlas A3 推理样例，支持单条和批量图文输入。在固定协议下完成 EmbSpatial-Bench 评测，提供断点续跑、答案解析与独立复算，并记录精度、性能和内存数据。

## 需求拆解

1. 在 RFC 中固定平台、卡数、代码/权重/数据版本、依赖、精度、峰值显存目标及评测协议。
2. 适配 CUDA 专属调用及视觉/MoT attention，保持模型结构、权重和原计算语义。
3. 提供推理入口，支持配置权重路径、device、dtype、max_new_tokens、temperature、top-p、seed 和 thinking。
4. 全量评测3,640题，Accuracy ≥80.0%，逐题结果可追溯、可恢复、可独立复算。
5. 完成功能、异常、稳定性及性能/内存自验，交付 README、适配文档与实验记录。预计周期4周。

# 详细设计（required）

## 模型分析

### 推理流程与计算语义

图像与文本经 processor/chat template 转为张量，传入 NPU 完成视觉编码及 MoT/语言主干计算，再通过自回归生成输出文本。

MoT 按模态 mask 将文本、视觉 token 分派至对应投影和计算分支。Attention 适配需要保留以下语义：

- **视觉编码器：**双向 attention；变长输入按原累计长度分别计算，各段独立。
- **MoT prefill：**先对混合序列计算因果 attention，再对每个视觉片段单独计算双向 attention，覆盖该片段的视觉输出；文本位置保留因果输出。
- **单 token decode：**当前 query 访问已缓存的有效历史 KV，保持原 GQA 头映射、RoPE、归一化顺序及 cache 位置。

Attention 计算为 `O = softmax(scale × QKᵀ + mask) × V`，scale 使用上游实际调用的值。padding 不参与有效计算；视觉片段与不同样本之间不交叉访问。

### 支持数据类型

模型参数和主要浮点计算使用 BF16，token ID、位置和路由索引保持所需整数类型。图像解码、分词与结果处理在 Host 完成，关键模型计算在 NPU 执行。

### 支持输入形状

单样本为一张图像与一段文本，批量输入使用左 padding，输出顺序与输入一致。图像尺寸及视觉 token 数由固定 processor 配置决定，文本长度遵守模型上下文约束。正式评测采用 batch size=1。

Attention 的 Q 为 `[B,Hq,Sq,D]`，K/V 为 `[B,Hkv,Sk,D]`；后端封装负责布局转换、变长长度及 mask 处理，保持 query 头与 KV 头的对应关系。

## 模型实现

### 实现方案

#### Host 侧设计

##### 1. 环境与模型加载

依据任务自验说明及上游环境脚本，采用以下版本作为初始方案：

| 对象 | 版本 / 来源 |
| --- | --- |
| HY 上游 | `cf23678a76dc9c445ab7f7aea7a9a8f60e910a63` |
| Transformers | `9293856c419762ebf98fbe2bd9440f9ce7069f1a` |
| 模型权重 | `tencent/HY-Embodied-0.5-X`，固定revision及文件哈希 |
| EmbSpatial代码 | `75792fc5957b0142fab8ff1d47574ffcf73e02c9` |
| 数据 | `Phineas476/EmbSpatial-Bench`，固定evaluation文件revision及哈希 |

A3具体型号、卡数、Recipes提交、CANN/PyTorch/torch_npu/Python组合及峰值显存目标在环境分配后补齐RFC。提供独立NPU依赖清单，启动时校验版本及设备，依赖不匹配直接报错。

在导入模型类前应用绑定源提交的patch，然后复用 `HyEmbodiedPipeline.from_pretrained` 加载本地权重、processor和chat template，配置BF16及NPU设备。保持权重名称和模型结构，不复制完整上游代码。

##### 2. 输入准备与生成

复用上游图像预处理及chat template，支持direct/thinking。上游Pipeline未直接提供top-p、seed、token统计及首token回调，封装时补齐这些接口，并显式记录有效生成参数。

正式评测初始提案为generation + thinking，BF16、batch size=1、max_new_tokens=32768、temperature=0.05、top-p=1.0、seed=42、use_cache=true、do_sample=true。其余参数包括top-k和EOS按权重生成配置导出并锁定。输出上限需结合上下文容量、A3显存和smoke耗时验证，正式运行前完成协议冻结。

评测prompt保留原问题和候选项顺序：

```text
{question}

A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Choose exactly one option. Put only its letter (A, B, C, or D) in the final answer.
```

模型上下文只包含图像、问题和候选项，ground truth及objects/bbox不传入生成模块。

##### 3. 评测、答案解析与恢复

读取官方human-verified evaluation，校验question_id唯一、候选项数量及六类样本统计，按原始顺序逐题推理。

thinking模式仅从唯一完整的 `<answer>` 块解析最终答案；direct模式检查整段最终回答。去除首尾空白后只接受A/B/C/D单字母，按原选项顺序映射。解码时保留答案结构标记，不从 `<think>` 搜索答案。格式不符、缺图、OOM、超时或运行异常均计错。

逐题保存question_id、relation、原始输出、解析答案、ground truth、correct、输入/输出token数、耗时和错误状态，结果原子落盘。独立复算脚本重新解析并计算总分和分类指标，不直接信任correct字段；结果ID集合与固定数据清单核对。

断点续跑检查代码、环境、参数及资产哈希，按question_id跳过已完成记录，包括失败题。每题根据固定全局seed和question_id派生种子，恢复随机状态与该题对应；配置变化则完整重跑。超时通过进程外监控处理，必要时重建推理进程后继续后续题目。已确认失败不重试计好，无法保持协议的中断另建完整run；多次完整结果分别报告。

##### 4. 代码组织

按贡献指南组织样例，代码放入 `src/`，适配及实验说明放入 `doc/`，README提供复现步骤；配置、依赖和测试作为必要文件提供。

```text
manipulation/hy-embodied-0.5-x/infer_with_torch/
├── src/                 # 推理适配、评测、答案解析及复算
├── doc/                 # 适配过程与实验说明
├── config/              # 推理及评测配置
├── tests/               # 功能、异常及恢复测试
├── requirements.txt
└── README.md
```

#### NPU 侧设计

##### 1. Attention 后端替换

patch移除模型文件顶层FlashAttention强制导入，并替换视觉塔的 `flash_attn_func` / `flash_attn_varlen_func` 及 `_flash_attention_forward_mot` 的调用路径。

主路径采用A3软件栈支持的融合attention接口，候选为 `torch_npu.npu_fusion_attention` 或对应融合推理接口。环境到位后验证BF16、GQA、head dimension、布局、变长长度及mask支持，再确定具体分派。

| 路径 | 实现方式 |
| --- | --- |
| 视觉塔定长/变长 | 转换为后端支持布局，按原分段执行非因果attention，再恢复输出布局 |
| MoT prefill | 有padding时unpad，执行因果attention，再repad |
| MoT视觉覆盖 | 按视觉片段gather Q/K/V，执行独立双向attention，scatter回原位置 |
| 单token decode | 当前Q与有效历史KV执行attention，保持cache位置和GQA头映射 |

封装时转换累计序列长度格式，核对mask取值含义、scale及输出布局。NPU显式matmul/softmax仅用于小尺寸数值对照；正式长序列采用融合实现，避免物化完整平方attention矩阵。

##### 2. 设备、MoT与KV Cache

审查device、autocast、stream、同步与随机数调用，替换推理路径中的CUDA专属逻辑。模型参数、视觉特征、MoT投影、attention、KV Cache和lm head保持NPU执行，不启用核心计算CPU offload。

保留模态mask与布尔索引逻辑，检查gather/scatter等操作的设备支持。每题独立KV Cache，完成后释放临时引用；批量输入保持样本隔离。

KV容量估算为 `2 × 层数 × B × 缓存token数 × KV头数 × head_dim × dtype字节数`。峰值另包含权重、视觉及主干临时张量、打包缓冲和算子workspace，以实际A3测量确定显存目标。

##### 3. 正确性验证

使用相同权重、图片、prompt、processor及生成参数对照参考实现，依次检查预处理、视觉特征、模态投影、attention输出和logits。局部张量记录最大绝对误差和相对L2误差，容差经BF16对照验证后固定。

用参数/张量位置及profiler trace核查核心执行设备，检查NaN/Inf及跨样本状态。贪心生成可用于定位数值差异，单独记录诊断配置，不混入正式评测协议。

## 支持硬件

| 支持的平台 | 涉及勾选 |
| --- | --- |
| Atlas A3 | √ |

初始采用单设备推理，具体产品型号、物理卡数、逻辑NPU数及显存分配在RFC中锁定。

## 模型约束限制

1. 图像预处理、模板及生成参数采用固定配置，不能按题目改变输入或解析规则。
2. 使用官方evaluation全量数据，不使用EmbSpatial-SFT或评测答案训练、提示泄漏及规则硬编码。
3. 权重、数据、base64图像及大体积结果不入库，提供获取方式、哈希及可审计制品链接。
4. 本任务不要求训练；OpenAI-compatible server为可选项。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 功能标准 | A3图文推理稳定，direct/thinking正确，核心模型计算在NPU | 任务书第3章 |
| 精度标准 | 正确数/3,640×100% ≥80.0%，即至少正确2,912题；六类分别报告正确数、总数及Accuracy | 任务书第2.3、3章，自验说明 |
| 完整性 | 全量ID与数据清单一致，无漏题、重复计分或答案泄漏；不可解析及异常计错，结果可独立复算 | 任务书第3章 |
| 性能标准 | 记录TTFT、端到端耗时、输出吞吐和token数，报告平均值/P50/P95；10个固定smoke样本每个至少重复5次profiling，无强制加速比例 | 自验说明 |
| 内存标准 | 开启profile_memory，记录加载后、预热后及正式评测NPU/Host峰值，满足RFC显存目标；连续100题及20题两轮无持续泄漏或状态残留 | 自验说明、任务书第2.1章 |
| 工程标准 | 环境与参数可追溯，完整正式评测至少一次，文档齐全，通过pre-commit、CI及A3实机验收 | 任务书第3、4章 |

六类数量按自验说明校验：close 612、far 594、above 596、under 602、left 616、right 620，合计3,640。

### 测试设计

| 分类 | 覆盖内容 | 检查点 |
| --- | --- | --- |
| 加载与依赖 | 固定版本、缺依赖、错误权重路径、不可用NPU及patch不匹配 | 正常加载或明确报错 |
| Attention | 视觉定长/变长、MoT因果与视觉覆盖、GQA、单token decode、左padding | 数值和可见范围与参考一致，输出索引正确 |
| 图文推理 | 六类真实样本、单条/批量、thinking/direct | 输入输出对应，无NaN/Inf、乱码、空答案或污染 |
| 解析与计分 | A/B/C/D、空/多答案、未闭合标签、think中含选项 | 仅最终答案参与计分，复算一致 |
| 数据与故障 | 重复ID、缺图、OOM、超时及进程中断 | 不删题，已确认失败计错 |
| 恢复与稳定性 | 配置改变、原子落盘、断点恢复、100题连续及20题两轮 | 不重复计分，配置变化拒绝恢复，无缓存污染或泄漏 |

固定smoke不少于10项，正常真实题覆盖六类关系及两种输出模式，异常用例覆盖缺图、坏图、不可解析和超长输出。模拟故障与真实模型结果分开记录。正式评测在配置冻结后全量执行；修复适配问题后完整重跑，保留每次结果。

TTFT通过首个新token回调测量，端到端时间包含读图、预处理、生成和解析，NPU计时考虑异步执行与同步。输出token从新生成ID统计，吞吐按总输出token数/总生成时间计算。profiling与正式精度运行分开；内存结合allocated/reserved、设备监控及Host RSS测量，区分allocator缓存与泄漏。

## 兼容性分析

新增独立模型样例，复用上游接口，不改变公共代码默认行为。模型专属适配保留在样例内；如修改公共模块，列出全部直接调用样例并完成其功能和精度回归。

检查代码、权重、数据及依赖License。README提供安装、推理、数据获取、评测、续跑和复算方法，优化文档记录适配点及验证结果，实验记录保存环境、版本、哈希、命令、完整指标及制品索引。支持范围以经过实测的RFC锁定A3环境为准。
