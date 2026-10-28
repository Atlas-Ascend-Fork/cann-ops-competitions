# daVinci-MagiHuman MatMul INT8 量化设计（Atlas A3）

# 需求背景（required）

## 需求来源与范围

本 RFC 面向 [10 月社区任务：daVinci-MagiHuman MatMul INT8 量化（A3）](https://www.hiascend.com/activities/task-center/details/f7dd746fa3694d34b10c38b43ede8aad?menu=guide)，拟在 `cann/cann-recipes-infer` 的 `models/davinci-magihuman/` 中接入真实 W8A8 INT8 线性层，降低权重存储和同配置端到端生成耗时，同时保留原有生成质量与显式 BF16 回退能力。

- 提交账号：`Zhang_E`。
- 当前阶段：开发前设计评议。本文描述拟实现的行为，未完成 A3 实机验收，不声明已获得性能或精度收益。
- 任务范围：Atlas A3、单进程/单设备、base 文生视频；普通线性层和该模式实际启用的专家/多模态线性层均纳入验证。
- 不量化 RoPE、RMSNorm、Softmax/FA、残差、VAE、文本编码器和音频编解码器；首版不扩大到超分辨率、distill、多卡或其他平台。
- 保留 Adapter、最终输出投影等非目标模块原有浮点 dtype。现有 FP32 模块不统一强制转为 BF16。
- 查重说明：仓库已有其他开发者提交的 [RFC #497](https://gitcode.com/cann/cann-recipes-infer/issues/497)。本提案由 Zhang_E 独立提出，不引用其基线、实测性能或校验结果作为本任务证据，请维护者协调任务归属与重复实现范围。

关联方案讨论：[cann-recipes-infer RFC #498](https://gitcode.com/cann/cann-recipes-infer/issues/498)。本文件根据 [CANN 社区任务参与流程](https://www.hiascend.com/developer/activities/cann-community-task) 和本仓设计模板提交设计文档 PR；设计评审通过并合入后开展实现，完成自验并按平台流程提交验收，通过测试后再按通知提交实现 PR。RFC Issue 用于技术讨论，不替代设计文档 PR。

## 背景介绍与现状分析

源码分析基线：`cann/cann-recipes-infer@a06336361c2614c7bcaebcb9046181a9e23008f7`。模型上游基线按任务书和模型 README 固定为 `209209b7086eba2020c5439265221495a8357322`。

| 位置 | 当前行为与接入关注点 |
| --- | --- |
| `inference/model/dit/dit_module.py::BaseLinear` | 参数权重为 BF16 `[E*N, K]`，调用 `_BF16ComputeLinear`；单专家时 `E=1`。`num_layers_for_initialization` 不是权重的额外维度。 |
| `NativeMoELinear` | 按模态分组输入，沿权重第 0 维拆分专家，分别执行线性计算，再按原逻辑拼接。必须保持专家、scale 和输入分组的一一对应。 |
| `Attention` / `MLP` | 候选模块为 `linear_qkv`、`linear_proj`、`up_gate_proj`、`down_proj`。QKV 可能包含 gating 部分；MLP 根据层配置采用 GELU 或门控 SWIGLU，不能改变投影布局。 |
| `TransFormerLayer` / `TransformerBlock` | 通过 `block.layers.{i}` 组织层；`mm_layers` 指定三模态层，其余为单专家。最终层数和专家数量以实际加载配置为准。 |
| `inference/model/dit/dit_model.py::get_dit` | 构造模型后调用 `load_model_checkpoint`，然后迁移设备并进入 eval。量化模块构造、权重加载、post-load 处理必须在此链路闭环。 |
| `inference/common/config.py` | 由固定上游版本非覆盖补入；已读取该版本的配置定义，顶层 `MagiPipelineConfig` 尚无 `quantization_config`，需显式扩展配置模型。 |
| `infer.sh` | 当前模式白名单不包含 `int8`，只增加 JSON/YAML 无法完成接入。 |
| `module/quantization/compressed_tensors/compressed_tensors_w8a8_int8.py` | 已有动态量化与 `npu_quant_matmul` INT8 调用，以及转置、NZ、scale dtype 的 post-load 处理，可参考和评估复用。 |
| `models/sana-video/patches/quant_module.py` | 当前 `a8w8` 分支使用 `npu_dynamic_mx_quant(...float8_e4m3fn)`，属于 MXFP8；不能据模式名认定为本任务的 INT8 实现。 |

以上为静态代码事实；尚未取得实际层清单、热点占比、权重统计和 A3 算子执行证据。

# 需求分析（required）

## 需求描述

在 Atlas A3 上为 daVinci-MagiHuman base 文生视频的目标 Linear/MatMul 引入真实 W8A8 INT8 计算，同时满足生成质量、端到端耗时、存储与内存稳定性的任务要求。

## 需求拆解

| 要求 | 设计与验证方式 |
| --- | --- |
| 真实 INT8 执行 | 离线 INT8 权重、在线动态 INT8 激活与 npu_quant_matmul，结合 profiling 核对实际算子与目标层 |
| 普通及专家路径 | 保持权重行、专家、scale 与输入模态的对应，覆盖空组和实际启用的专家 |
| 精度与质量 | 预先冻结独立开发集和五组评测输入；分别验证线性层、DiT block 与最终视频质量 |
| 性能与存储 | 同配置端到端耗时改善超过 BF16 基线波动；统计完整制品体积及量化额外开销 |
| 兼容与回退 | 量化默认关闭，敏感投影显式 BF16 回退，元数据异常报错，连续生成与重复加载检查内存 |

# 详细设计（required）

## 3. 基线冻结与环境记录

| 项目 | 拟固定值 / 当前状态 |
| --- | --- |
| 仓库与上游版本 | 上述两个 commit；后续改版记录完整 SHA，BF16 与 INT8 使用相同基础代码 |
| 硬件 | 目标 Atlas A3；实际型号、设备可见 ID、物理卡/die 映射、HBM、Host 内存、驱动、固件和 OS 待采集，不预填 |
| 参考软件 | Python 3.12、CANN 9.0.0、PyTorch 2.10.0、torch_npu v26.0.0；安装包真实版本、构建号与 hash 待采集 |
| 生成模式 | `base` 文生视频；`world_size=1`，`cp_size=1` |
| 输入参数 | 448×256、5 秒、32 steps、CFG=2、生成 seed=42、`dit_cache.method=NoCache` |
| 执行设置 | 继承 `base.yaml`，含 `TORCH_COMPILE_DISABLE=1`、`MAGI_COMPILE_COMPILE_MODE=NONE`；offload、缓存、同步和计时边界保持一致 |
| 权重 | daVinci-MagiHuman base、T5Gemma、Stable Audio、Wan2.2 VAE，以及 base 实际使用的 TurboVAE 配置/权重，逐项记录来源、revision、文件清单及 SHA-256 |
| 初始化随机性 | 记录 `engine_config.seed`；BF16/INT8 保持一致。生成 seed 与初始化 seed 分开记录，避免只固定其中一项 |
| 样本 | 第 7 节固定评测样本；开发数据与最终评测数据分开冻结 |

按模型 README 非覆盖合并固定上游 `inference/`，复制 assets/prompts 并安装依赖。第一次实机测试前，将缺失项补入运行 manifest；不得使用占位 hash 进入验收。

BF16 profiling 基线命令沿用任务书：

```bash
cd models/davinci-magihuman
export ENABLE_PROFILER=1
bash infer.sh base
```

基线采集前，在测试配置中显式设置 `model_args.seed: 42`；INT8 配置复制同一生成参数。量化配置、量化权重路径及输出位置以外的差异均须解释并消除。

## 4. W8A8 数值与张量契约

首版提议：权重逐输出通道对称 INT8，激活逐 token 动态对称 INT8，zero-point 为 0；不引入静态激活校准或 SmoothQuant 重标定。以下契约须在实现及最终评测前由 RFC 冻结。

设一个专家的权重 `W[N,K]`，展平输入 `X[M,K]`：

- 离线权重量化：`s_w[n] = max(abs(W[n,:])) / 127`，全零通道以 `s_w[n]=1`、`qW[n,:]=0` 明确定义；其余 `qW = clamp(round_to_nearest_even(W/s_w), -127, 127)`。
- 激活：通过目标版本的 `npu_dynamic_quant` 获得 `qX[M,K]` 与 `s_x[M]`。以实机探针核对其量化范围、舍入和全零行行为，记录算子版本；不能用未经核对的手写模拟公式替代实际算子契约。
- 核心计算：INT8×INT8 MatMul，按 `s_x[m]*s_w[n]` 反量化，目标输出 BF16。语义参考为 `Y = (qX @ qW.T) * s_x[:,None] * s_w[None,:] + bias`。
- scale 保存为 FP32。首选运行时权重 scale FP32 `[N]`、激活 scale FP32 `[M]`；目标软件若要求转换，须在 post-load 中显式处理并记录，先通过探针验证，再冻结配置与误差阈值。
- 当前四类目标投影为 `bias=False`；适配层仍测试有/无 bias。浮点 bias 不直接当作 INT32 accumulator bias，先保持浮点加法语义，再评估目标版本的融合能力。
- 保留原输入前导维度、输出 shape 与 `output_dtype` 接口语义；需要转换输出 dtype 时，在反量化后显式转换。

| 张量 / 字段 | 存储契约 | 运行时契约 |
| --- | --- | --- |
| 普通/专家权重 | `qweight: int8 [E*N,K]`，保持原输出通道与专家顺序 | 先拆为每专家 `[N,K]`，再转置为 `[K,N]`；不得先整体转置后沿错误维度切专家 |
| 权重 scale | `weight_scale: float32 [E*N]`，和权重行逐一对应 | 每专家 `[N]`，与该专家权重共同切分；布局变换不能改变 scale 含义 |
| 激活与 scale | 不保存跨输入复用的静态激活 scale | 每次对当前输入/专家非空分组计算 `qX[M,K]` 和 `s_x[M]` |
| bias | 无 bias 时显式标记；有 bias 时保留源 dtype 与 `[E*N]` shape | 随专家分片，按浮点线性层语义加回 |
| 原始浮点保留层 | 保留原 dtype、名字与形状 | 只对选定敏感投影采用 BF16 回退；非目标 FP32 层保持原状 |

全零输入和权重有专门用例；合法全零张量的确定性 scale 规则不等同于给缺失元数据补占位 scale。NaN/Inf 权重或输入按错误处理。

## 5. 模块接入、产物与回退

**5.1 配置和加载链路**

1. 新增模型私有 `QuantizationConfig`，在 `MagiPipelineConfig` 增加默认关闭的 `quantization_config`，通过 `entry.py → get_dit` 传递，缺少字段时执行原 BF16 链路。
2. `get_dit` 在启用量化时先验证制品 manifest、层名和 shape，再构造/替换目标线性层并加载对应 dtype 的权重。禁止先把 INT8 复制到 BF16 Parameter、或在 `model.to(dtype=...)` 时误转换量化权重。
3. 保留标准 `load_model_checkpoint` 对非目标参数的命名与加载语义；量化加载器检查缺失/多余键。上游 checkpoint 实现需在开发阶段补齐阅读和最小加载测试，不假设它能够直接消费自定义量化制品。
4. 优先评估复用 `module/quantization/` 的 INT8 method；模型私有适配负责原接口、专家切片和量化 manifest 映射。不得为了复用而绕过其 scale、smooth scale 或 post-load 前置条件。若接口不匹配，保留公共模块原样，以模型私有适配实现相同 INT8 算子契约，并说明差异。
5. 保持单专家以及 `mm_layers` 多模态路径的 dispatch/undispatch 顺序。空专家分组直接返回正确 shape 的空输出，不触发不支持的零维 MatMul。
6. 权重转置和可选 NZ 转换仅在 post-load 执行一次；转置、布局、scale dtype 都记录到 manifest，重复加载不二次转置。ND/NZ 选择通过目标 shape 探针与性能证据决定。

**5.2 离线制品**

转换工具逐 shard 处理原权重，保持确定的层名、专家顺序、舍入规则与文件分片顺序。制品包含 `model*.safetensors`、索引、`quant_meta.json`、`CHECKSUMS.sha256`；只提交生成工具和说明，不提交权重。

`quant_meta.json` 至少记录：schema 版本、转换工具/仓库/上游 commit、源模型和各 shard hash、量化方案与舍入规则、层名、`K/N/E`、dtype/shape、weight/scale/bias 键、zero-point、布局、目标/忽略层清单、软件版本、输出文件 hash 和转换命令参数。确定性元数据不嵌入时间戳/绝对机器路径；时间信息单独放运行日志。相同输入及工具版本重复转换须得到相同张量和确定性制品校验结果。

**5.3 失败与回退规则**

- `enabled=false`：走原始 BF16 配置/权重，不要求量化制品存在。
- `ignored_layers`：在模块构造前解析为完整层名，作为敏感层的显式 BF16 保留策略；每条规则必须匹配实际模块，否则报错。专家数量和粒度记录清楚，首版按完整投影模块回退，避免模糊的专家局部匹配。
- 量化目标缺 scale、错误 shape/dtype、非有限或非法非正 scale、hash 不符、版本不兼容、层名不匹配：默认立即报错，信息包含层名、实际值和期望契约。
- 禁止自动生成占位 scale、静默跳过量化、或将 INT8 立即反量化后用 BF16 MatMul 冒充优化。
- 不支持的硬件/算子组合需报错；通过修改已审定的 `ignored_layers` 重新运行可显式回退，所有回退层及原因写入报告。

## 6. 使用方式与交付文件

以下为拟新增接口示例，当前仓库尚不能运行 `bash infer.sh int8`。

`config/int8.yaml` 从基线复制，设置 `config-load-path: config/int8_config.json`、`seed: 42` 和独立输出目录；同时扩展 `infer.sh` 模式白名单。

`int8_config.json` 保留基线全部 `engine_config` / `evaluation_config`，增加如下量化字段：

```json
{
  "quantization_config": {
    "enabled": true,
    "mode": "w8a8",
    "activation_method": "dynamic",
    "weight_granularity": "per_channel",
    "activation_granularity": "per_token",
    "symmetric": true,
    "quantized_weight_path": "/path/to/davinci-magihuman-int8",
    "manifest_file": "quant_meta.json",
    "target_modules": ["linear_qkv", "linear_proj", "up_gate_proj", "down_proj"],
    "ignored_layers": [],
    "fallback_dtype": "bfloat16",
    "on_invalid_metadata": "error"
  }
}
```

`target_modules` 仅作为枚举候选层的后缀筛选条件；转换器输出最终完整层名清单，并与加载后的对象逐项对账。空 `ignored_layers` 是初始方案，是否保留须由预先冻结的开发集敏感度评估决定，不能基于最终验收样本反向挑层。

拟新增转换入口：

```bash
python -m inference.model.quantization_layers.weight_convert \
  --source /path/to/daVinci-MagiHuman/base \
  --output /path/to/davinci-magihuman-int8 \
  --config config/int8_config.json
bash infer.sh int8
```

| 交付位置 | 拟修改内容 |
| --- | --- |
| `models/davinci-magihuman/inference/model/quantization_layers/` | INT8 适配、产物转换、manifest 校验、专用加载逻辑 |
| `inference/common/config.py`、`inference/pipeline/entry.py` | 模型私有配置解析与透传；保留上游许可证与原配置默认行为 |
| `inference/model/dit/dit_model.py`、必要的 `dit_module.py` 改动 | 加载分支、目标模块映射、专家和 bias 语义 |
| `config/int8.yaml`、`config/int8_config.json`、`infer.sh` | 可启动且默认关闭的量化入口 |
| 模型测试目录 | 功能/异常、层与 block 数值、产物确定性、性能统计和内存检查脚本 |
| 模型 README、`docs/models/davinci-magihuman/` | 量化原理、A3 支持范围、权重转换、启停、限制、回退与复现实验 |

# 可维可测分析

## 7. 数据、精度和质量验证

动态激活量化不需要离线拟合激活 scale，但必须准备独立的开发/敏感度数据。拟使用 [VBench](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490) 开源 prompt suite 的固定子集，revision 为 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；开发集拟选 32 条覆盖人物、动作与场景的 prompt。实现前公开源文件、抽样规则、样本索引/文本 hash 和 manifest hash，并在查看 INT8 验收结果前冻结。当前尚未完成抽样，不声明该子集已经验证。

最终验收至少使用下列 5 组原创固定 prompt，生成 seed 均为 42，参数完全一致。开发集不包含这些 prompt；若需要修改样本，必须先修订 RFC 并重新跑全部基线，保留变更原因。

| 编号 | 覆盖 | 固定 prompt |
| --- | --- | --- |
| Q1 | 单人、面部、手部 | A woman in a blue jacket sits at a wooden table, turns toward the camera and waves with her right hand. Natural daylight, fixed camera. |
| Q2 | 多人、交互 | Two adults in a sunny park pass a red ball to each other while talking. Both people remain visible in a medium-wide shot. |
| Q3 | 肢体动作 | A dancer in black sportswear performs a slow full-body turn and raises both arms in an indoor studio. Fixed full-body framing. |
| Q4 | 镜头运动 | A man in a gray coat walks along a riverside path while the camera smoothly tracks sideways, keeping him centered. |
| Q5 | 复杂背景、遮挡、声画 | A woman speaks to the camera in a busy outdoor market with colorful stalls, moving pedestrians and hanging lights in the background. |

建议在首次 INT8 验收前冻结以下数值门限；它们是提案值，不是当前测试结果，也不是任务书给定数值：

- 逐目标线性层：余弦相似度 ≥ 0.995、相对 L2 ≤ 0.03。
- 至少一个完整 DiT block，同时覆盖普通层与多模态层的代表性输出：余弦相似度 ≥ 0.99、相对 L2 ≤ 0.05。
- 相对 L2 定义为 `||Yq-Ybf16||2 / max(||Ybf16||2, 1e-6)`；余弦以 FP32 计算。零范数参考输出单列，检查最大绝对误差 ≤ 1e-6，不将无定义余弦记为通过。
- 普通层、各实际专家和不同去噪时段分别报告，不用跨层平均值掩盖失败；均不得出现 NaN/Inf。纯零输入有 bias 时应输出对应浮点 bias。
- 层级误差比较使用相同的 BF16 录制输入隔离本层误差；block 同时报告固定入口输入差异与实际链路累计差异。探针采样步骤建议 0、15、31，并保存采样规则。
- 五组视频逐项记录主体、语义、构图、动作、时序、色彩、音画一致性及伪影；无可稳定辨识退化。记录评审者、结论及视频获取位置/hash，不以单一张量指标替代最终质量判断。

功能/异常用例覆盖：开关量化、原 BF16 配置、敏感层回退、错误/缺失 scale、非法 dtype/shape/hash、普通线性层、实际专家路径、空专家分组、bias、全零输入、重复加载、连续 5 次生成。失败脚本返回非零状态。

## 8. 性能、存储和内存验收

**量化真实性**：加载后核对全部目标权重/scale 和模块类型；profiling 中将目标模块与实际 INT8 MatMul 事件关联，记录调用次数、输入 dtype、累计耗时和动态量化开销。算子名以目标版本实际输出为准；仅配置开关或 INT8 权重 dtype 不足以证明真实执行。

**热点与范围**：先从 BF16 profiling 生成四类投影的完整候选层表，记录 shape、专家数、调用次数、累计耗时及占比。再依据开发集敏感度确定最终目标/回退清单，在最终性能验收前冻结，不预填层数、调用次数或占比。

**端到端计时**：BF16/INT8 各至少 1 次预热、3 次正式完整生成，固定 batch、prompt、seed、环境和加载状态；建议交替测试以减少温度/系统漂移。计时前后完成 NPU 同步，统一 Base、Decode、Total 的计时边界；文件编码/保存是否包含在 Total 中须明确，另记启动/加载耗时。正式延迟测试关闭 profiler，另开 profiling 运行采集算子证据。

设 BF16 三次 Total 为 `B1..B3`，INT8 为 `Q1..Q3`：

```text
Bmed = median(B1, B2, B3)
Qmed = median(Q1, Q2, Q3)
Rbase = max(B1, B2, B3) - min(B1, B2, B3)
通过：Bmed - Qmed > Rbase，且 Qmed < Bmed
提升百分比 = (Bmed - Qmed) / Bmed * 100%
```

同时要求目标 MatMul 或目标 block 累计耗时下降，报告量化/反量化额外开销。每档提供原始值、均值、中位数、极差；基线明显不稳定时先排除系统干扰、增加样本并保留全部数据，不删除慢样本制造收益。不得修改分辨率、视频时长、steps、CFG、缓存或输入取得表面加速。

**存储**：分别统计 BF16 DiT 制品、INT8 权重、scale/元数据、保留浮点层和总制品大小，保证同范围 INT8 总存储小于 BF16；另报告包含未量化子模型的整套部署存储。不能只比较 int8 数组字节数。

**内存**：记录加载后、生成峰值、连续 5 次生成后和重复加载时的 NPU allocated/reserved 与 Host RSS；同一采样口径报告变化，无持续增长。峰值显存下降为期望收益，不把任务书未要求的固定百分比写成既定门限。

## 9. 兼容性、限制和备选方案

- 默认关闭量化、无量化字段的旧配置继续使用原 BF16 路径；既有 base/其他模式至少完成配置与导入回归，声明支持的目标模式完成实机功能/精度回归。
- 首版把改动限制在模型目录及对应文档。若需改公共 `module/quantization/`，先列出全部直接调用方，再完成其功能、精度及代表性性能回归；未完成前不提交该公共改动合入。
- 动态量化会引入 reduction、scale 和算子启动开销；小 shape/小专家分组可能无收益。精度敏感层回退会降低压缩与加速空间，需如实报告。
- 当前不承诺 A2/A5、图模式、多卡、SR/distill 或任意 checkpoint 通用支持。
- 备选一：静态激活量化，可能降低运行时开销，但需要代表性校准、独立 scale 制品与额外误差验证；只有在动态方案确有瓶颈时另行修订 RFC。
- 备选二：局部 W8A8 + 显式 BF16 回退，以开发集敏感度和热点证据选层。必须同时满足端到端性能、存储和质量要求。
- 仅 W8A16 或 BF16 伪量化不满足本任务真实 INT8 MatMul 的验收目标。

## 10. 实施计划与最终材料

以下为工作量估算，不是已确认的日历排期。官方任务截止时间为 2026-11-02 00:00（UTC+8），验收申请须在此前提交；设计审核通过、A3 与权重就绪后需据实际剩余时间安排并行工作并更新进展。若不能在截止前完成全部自验与验收申请，不承诺取得本轮任务激励，不以未测试材料代替验收。

| 阶段 | 建议时长 | 完成条件 |
| --- | --- | --- |
| 环境、BF16 基线和契约冻结 | 2 个工作日 | manifest、固定样本、基线 profiling、热点清单、A3 算子探针及数值门限就绪 |
| 转换器、配置和加载接入 | 3 个工作日 | 可重复制品、普通/专家路径、异常校验与 BF16 回退通过 |
| 精度、质量、性能和内存 | 3 个工作日 | 逐层/block、5 组样本、连续生成及至少 3 次正式耗时完整 |
| 回归、文档和验收材料 | 2 个工作日 | 贡献规范检查、模型测试、必要公共模块回归与平台验收材料；测试通过后按通知提交实现 PR |

实现 PR 更新模型 README 和 `docs/models/davinci-magihuman/`，说明原理、平台支持矩阵、启用方式、权重生成、限制与回退；附候选/最终层清单、数值/主观质量、profiling、逐次耗时、内存与存储报告。执行 `pre-commit run --all-files`、仓库 CI 和 A3 实机测试，分别记录状态。

开发者私仓 `task_submission/` 按任务书提交：`1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`；不适用项说明原因。数据、权重、视频和大 profiling 文件仅给出可访问获取方式与校验信息，不进入公开代码仓。

## 11. 请维护者评议

1. 是否接受上述动态 per-token / 权重 per-channel W8A8 方案与模型私有适配边界？
2. 是否接受建议的逐层/block 数值门限、最终质量判定及以 BF16 极差为波动范围的性能判据？
3. 开发集采用 VBench 固定子集、最终质量使用上述 5 组独立 prompt 是否合适？样本 manifest 将在测试前补齐并冻结。
4. 与其他开发者的 [RFC #497](https://gitcode.com/cann/cann-recipes-infer/issues/497) 如何协调实现范围？具体进展将在设计评议与算力确认后更新，遵守任务截止时间。

## 12. 参考资料

- [CANN 社区任务提交与评审流程](https://www.hiascend.com/developer/activities/cann-community-task)：设计文档 PR 评论需 @condfuse_3、@fullt、@Ascend-CANN。
- [社区任务页面](https://www.hiascend.com/activities/task-center/details/f7dd746fa3694d34b10c38b43ede8aad?menu=guide)及随任务下载的任务书、`test_cases/README.md`。
- [固定源码版本的模型 README](https://gitcode.com/cann/cann-recipes-infer/blob/a06336361c2614c7bcaebcb9046181a9e23008f7/models/davinci-magihuman/README.md)、[线性层实现](https://gitcode.com/cann/cann-recipes-infer/blob/a06336361c2614c7bcaebcb9046181a9e23008f7/models/davinci-magihuman/inference/model/dit/dit_module.py)。
- [仓内 W8A8 INT8 实现](https://gitcode.com/cann/cann-recipes-infer/blob/a06336361c2614c7bcaebcb9046181a9e23008f7/module/quantization/compressed_tensors/compressed_tensors_w8a8_int8.py)。
- [模型上游固定版本](https://github.com/GAIR-NLP/daVinci-MagiHuman/tree/209209b7086eba2020c5439265221495a8357322)、[其配置定义](https://github.com/GAIR-NLP/daVinci-MagiHuman/blob/209209b7086eba2020c5439265221495a8357322/inference/common/config.py)。
- [官方 npu_quant_matmul API 参考](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)：参考页面可能随主线变化，最终支持性以任务要求软件版本及 A3 实机探针为准。

当前未完成：A3 环境/权重 manifest、开发集抽样 manifest、BF16 基线、算子探针、最终目标/回退层、精度/性能/存储/内存实测、CI。上述项目均作为实施与验收前置条件，不以本 RFC 的提交代表已通过。
