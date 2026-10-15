# 需求背景（required）

## 需求来源

[10 月社区任务：daVinci-MagiHuman MatMul INT8 量化（A3）](https://www.hiascend.com/activities/task-center/details/f7dd746fa3694d34b10c38b43ede8aad)。任务 ID：`f7dd746fa3694d34b10c38b43ede8aad`；开发者：`2511_93680239`。

本方案在 `cann-recipes-infer/models/davinci-magihuman/` 接入模型私有 W8A8 量化，采用 **40 个 MLP up 投影平滑 INT8、120 个 attention/down 投影 BF16** 的混合精度策略，以真实 INT8 MatMul 降低 base 视频生成耗时和 DiT 权重存储，同时保持人物、动作与音画质量。

任务工程形态为模型量化适配。设计沿用官方模板的四个主章节，将 host/kernel 栏目落实为离线转换、配置与加载、NPU 运行时三部分；计算调度使用 CANN/TorchNPU 原生算子。

## 背景介绍与问题分析

模型的 `BaseLinear` 和 `NativeMoELinear` 承载 DiT 的 QKV、attention 输出、MLP up-gate/down 四类投影。BF16 基线需要读取较大的权重并反复执行矩阵乘法，INT8 可降低权重带宽和矩阵计算开销。视频扩散过程会逐步传播量化误差，因此本方案把性能收益、模态专家一致性和视频质量共同作为选层依据。

开发对照覆盖全投影、仅 MLP、平滑全投影、平滑 MLP、仅 down 和仅 up 六种配置。含 down 量化的多个候选出现舞蹈场景退化；保留 attention/down 的 U 配置获得五类开发视频整体正面反馈，并保留明确的端到端收益。因此本次交付采用 U，将目标层与排除层写入配置和量化 manifest。

## 工程版本与参考

- 目标仓研究基点：[`cann-recipes-infer@bbd278fe5ef772daa310df680865a6826ae6d909`](https://gitcode.com/cann/cann-recipes-infer/tree/bbd278fe5ef772daa310df680865a6826ae6d909/models/davinci-magihuman)。
- 上游依赖基点：[`daVinci-MagiHuman@209209b7086eba2020c5439265221495a8357322`](https://github.com/GAIR-NLP/daVinci-MagiHuman/tree/209209b7086eba2020c5439265221495a8357322)。按目标仓 README 非覆盖合并，NPU overlay 优先，逐文件保存来源及 SHA256。
- 接口：`torch_npu.npu_dynamic_quant` 和 `torch_npu.npu_quant_matmul`；[TorchNPU 26.1 动态量化接口](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)。
- 权重组件：DiT base、T5Gemma、Stable Audio、Wan2.2 VAE、TurboVAE；保存下载来源、固定 revision、文件清单与 SHA256，BF16/INT8 共享相同的非 DiT 组件。

# 需求分析（required）

## 需求描述与验收对应

| 需求 | 设计措施 | 验证输出 |
| --- | --- | --- |
| A3 上真实 INT8 MatMul | 动态 per-token 激活量化、per-channel INT8 权重、原生量化 MatMul | 目标层与设备 trace 的映射、输入/权重 dtype、算子累计耗时 |
| 普通层与模态专家 | 按原 dispatcher 分组，每专家独立倍率/权重/scale，再按原顺序合并 | 普通及专家路径数值对照、零 token 用例 |
| 敏感层 BF16 | 精确冻结 40 个目标层与 120 个排除层 | 完整层表、配置、manifest 和逐层加载核对 |
| 精度与音画质量 | 通道平滑、分层数值比较、五类场景双人盲测 | 张量指标、匿名视频及两名验收人的原始结论 |
| 性能改善 | 配对同环境测试、独立 profiling、驻留请求测量 | Base/Decode/Total 及完整请求逐次耗时与统计 |
| 存储与内存 | INT8 制品直接加载，分片转换，buffer 生命周期管理 | 完整 DiT 制品字节数、加载/峰值/连续推理/重复加载内存 |
| 工程交付 | 标准 `infer.sh int8`、模型私有代码、严格校验、默认 BF16 | 自验、模型文档、提交报告及仓库 CI |

## 支持硬件与固定配置

| 项目 | 配置 |
| --- | --- |
| 硬件 | Atlas A3，已测 SoC 为 Ascend910_9382；每次运行只暴露一个逻辑 NPU |
| 并行 | 单进程，`world_size=1`、`cp_size=1`，TP/PP/DP 均为 1 |
| 场景 | base 文生视频，5 秒、448×256、32 steps、CFG=2、seed=42、NoCache |
| 标准入口 | BF16：`bash infer.sh base`；U：`bash infer.sh int8` |
| 当前开发环境 | Ubuntu 22.04、Python 3.12、CANN 9.1.0、PyTorch 2.10.0、torch_npu 2.10.0.post4（TorchNPU 26.1） |
| 任务参考环境 | Python 3.12、CANN 9.0.0、PyTorch 2.10.0、TorchNPU 26.0 |

正式评测采用社区确认的环境组合，BF16 和 INT8 始终在同机、同版本下成对执行。提交环境锁文件，包含 wheel/hash、驱动/固件、MagiCompiler、ffmpeg、Host/NPU 容量与设备映射；当前开发环境的认可问题已提交任务讨论帖。

# 详细设计（required）

## 算子分析与精度策略

### 目标层与排除层

对 `i=0..39`，配置选择 `block.layers.i.mlp.up_gate_proj`，共 40 个模块：32 个普通层和 8 个三模态专家层，合计 56 个专家权重组。专家块为 0–3、36–39。以下三类每类 40 个模块保持 BF16：

1. `block.layers.i.attention.linear_qkv`。
2. `block.layers.i.attention.linear_proj`。
3. `block.layers.i.mlp.down_proj`。

公共配置使用 `target_layers=[]` 选择四类投影候选，再由 `ignored_layers` 精确排除上述 120 个完整路径；结构导出器将实际入选的 40 个层写入 `layer-specs.json`，加载器对照模型逐项核验。RoPE、RMSNorm、Attention/Softmax、残差、CFG、采样器、输入 adapter 和最终输出投影沿用基线计算。

仅 up 的配置同时保留普通线性层和真实模态专家，保留较大的 MLP 矩阵计算收益；attention/down 使用原 BF16 权重抑制敏感路径的误差传播。

### 数学公式与 dtype 契约

对专家 `e`，设权重 `W_e[N,K]`，输入前缀展平为 `X_e[M,K]`。权重量化按输出通道，平滑倍率按专家和输入通道，激活量化按运行时 token。

**离线平滑：**

```text
a_e[k] = max |X_e[:,k]|             # BF16 基线实际输入的校准统计
b_e[k] = max_n |W_e[n,k]|
alpha = 0.5
d_e[k] = b_e[k]^(1-alpha) / a_e[k]^alpha
```

`a_e[k]` 或 `b_e[k]` 为零时取 `d_e[k]=1`。统计保存为 CPU FP32 `[K]`；倍率先保存为 BF16 `[E,K]`，随后用这个实际存储值计算 FP32 `W'_e = W_e / d_e`。倍率要求正且有限。

**离线权重量化：**

```text
s_w[e,n] = max_k |W'_e[n,k]| / 127
q_w[e,n,k] = clamp(round_to_nearest_even(W'_e[n,k] / s_w[e,n]), -127, 127)
```

全零输出通道取 `q_w=0, s_w=1`。`q_w` 保存为 INT8 `[E*N,K]`，`s_w` 为 FP32 `[E*N]`；非有限权重、非法 scale 和无法表示的倍率均使转换失败。

**NPU 前向：**

```python
x_bf16 = input.reshape(-1, K).to(torch.bfloat16).contiguous()
q_x, s_x = torch_npu.npu_dynamic_quant(x_bf16, smooth_scales=d_e)
y_bf16 = torch_npu.npu_quant_matmul(
    q_x, q_w_e.transpose(0, 1), s_w_e,
    pertoken_scale=s_x, output_dtype=torch.bfloat16,
)
```

`q_x` 为 INT8 `[M,K]`，`s_x` 为 FP32 `[M]`。`d_e` 在动态量化内部作用于 BF16 输入，与离线 `W/d_e` 配对；运行时直接使用该融合接口，保持倍率应用的计算次序。MatMul 输出后执行 BF16 bias 加法，再恢复输入前缀及调用接口要求的 `output_dtype`。当前目标层为无 bias 配置，通用适配保留有 bias 支持。

### 形状、布局与边界

| 路径 | K | 每专家 N | 本方案处理 |
| --- | ---: | ---: | --- |
| GELU7 up，8 个三模态专家层，E=3 | 5120 | 20480 | ND INT8 权重，转置视图送 MatMul |
| SwiGLU7 up-gate，32 个普通层，E=1 | 5120 | 27304 | 保留完整 gate 布局与实际 N，使用原生 API |
| QKV | 5120 | 7208 | BF16，保留 Q/K/V 和 40 个 gate 通道 |
| Attention 输出 | 5120 | 5120 | BF16 |
| GELU7 down | 20480 | 5120 | BF16 |
| SwiGLU7 down | 13652 | 5120 | BF16 |

真实层表由 meta 模型和 checkpoint 交叉核对，记录 E/N/K 与专家次序。M=0 返回正确 shape/dtype/device 的空输出；全零 token 接受动态量化 API 的合法零 scale；非连续输入整理后计算，整理成本计入测量。量化权重的 scale 则统一要求严格为正。

## 模型侧实现

### Host：配置、转换与加载

1. `infer.sh` 将标准 `int8` 路由至 U，旧 `int8_up_smooth` 名称归一为 `int8`；配置、输出和 profiler 路径共用。`int8_full`/`int8_mlp` 保留为诊断配置。
2. `inspect_int8_model` 在 meta 上构造模型，导出选中层的完整路径、E/N/K 和 bias，并校验原权重为 BF16，生成转换契约。
3. 转换器逐 shard 读取 BF16 权重，按专家生成 `qweight`、`weight_scale`、`input_multiplier`，保留非目标状态，生成索引与 schema 2 manifest。
4. 转换输出使用新目录，完成 tensor、索引与校验和核对后原子发布；校准文件和源文件在转换前后校验 SHA256。
5. 加载器在 meta 模型上核对 schema、算法、层集合、专家数、shape、dtype 和文件哈希，再替换模块并加载真实状态，随后进行 NPU 迁移及组件 offload 管理；当前 U 完整制品含 411 个状态，逐项核对量化状态及原精度保留状态。

关键配置为 `enabled=true`、`schema_version=2`、`mode=w8a8`、动态 per-token 激活、per-channel 权重、量化制品路径、精确 `ignored_layers`、`fallback_dtype=bfloat16`、`on_metadata_error=error`。旧 schema 1 使用独立协议；加载器显式匹配配置与制品版本。

### NPU：模态专家计算与生命周期

`NativeMoELinear` 使用原 dispatcher 将视频、音频、文本 token 分组。权重按 `[E*N,K]` 分成 E 份，scale 与 multiplier 按相同专家索引取值，各组执行动态量化和 INT8 MatMul，再由原 undispatch 恢复顺序。空组直接返回空输出。

量化权重、scale 和倍率注册为 buffer，bias 保留 BF16；`.to(device)`、`state_dict` 和组件 offload 使用同一套状态。模型加载直接建立量化表示，Host 分片转换和 meta 加载控制峰值内存，NPU 仅保留运行所需的模型状态。

### 文件与接口归属

| 模型目录下的位置 | 职责 |
| --- | --- |
| `infer.sh`、`config/int8.yaml`、`config/int8_config.json` | 标准启动、seed、NoCache 与量化配置 |
| `inference/common/config.py` | 严格配置解析和参数透传 |
| `inference/model/quantization_layers/config.py` | 目标/排除层匹配、量化契约 |
| `weight_convert.py`（同目录） | 校准统计、平滑、INT8 转换及 manifest |
| `loader.py`（同目录） | 元数据、索引、tensor 与 SHA256 严格加载 |
| `int8.py`（同目录） | 普通/专家前向、空组、dtype 与 bias 顺序 |
| `inference/model/dit/dit_model.py`、`dit_module.py` | 模型构造、加载接入与模块兼容 |
| `tools/`、`tests/` | 结构导出、API 探针及契约回归 |
| 模型 README、模型专属优化文档 | 转换/运行命令、方案与测量口径 |

公共量化模块保持现有接口，修改范围集中于本模型。

## 制品与校准数据设计

制品包含 safetensors shards、索引和 `quantization_manifest.json`。manifest 保存 schema/算法、源权重 revision/hash、量化层 E/N/K、tensor dtype/shape、scale 粒度、平滑倍率、校准统计 SHA256、`alpha` 及全部输出文件哈希。实验清单另保存源码快照、配置、依赖、数据来源和执行命令。

数据按校准、开发、最终盲测三种用途分离；开源数据登记源仓、许可证、固定 revision、抽样规则、样本 ID 和内容哈希。按人物、动作和场景分层选样，锁定清单后固定 seed 和生成配置。

校准在 BF16 基线的目标投影输入处采集每专家每通道 `absmax`，覆盖去噪步骤、两条 CFG 分支及实际模态分组；按最大值聚合为 `模块名.expertE.all.absmax`。每个入选专家必须有完整统计，缺项立即报错。评测清单与校准清单保持独立。

本轮 U 的开发制品使用首条开发 prompt 的统计完成选层；正式校准阶段使用评审确认的开源清单重新生成制品，并将新统计、制品和配对评测绑定为同一版本。

正式样本规模提案为五类场景各 4 条、共 20 条校准 prompt，另选每类 2 条、共 10 条开发 prompt；最终盲测至少五组，每类至少一组，由独立清单抽取。校准阶段每条完成 32 steps、两次 CFG，20 条合计 1280 次 DiT 前向，以流式 absmax 聚合控制 Host 存储。数据清单经评审确认后生成固定哈希。

以下命令从模型目录执行，配置中的路径按实际权重位置填写：

```bash
python -m tools.inspect_int8_model \
  --config-load-path config/int8_config.json --output layer-specs.json
python -m inference.model.quantization_layers.weight_convert \
  --source-dir /path/to/models/daVinci-MagiHuman/base \
  --output-dir /path/to/models/daVinci-MagiHuman/int8_up_smooth \
  --layer-specs layer-specs.json \
  --source-revision EXACT_WEIGHT_REPOSITORY_REVISION \
  --activation-stats calibration.safetensors --alpha 0.5
bash infer.sh int8
```

# 可维可测分析

## 功能与精度验证

| 验证层级 | 用例与方法 | 判定 |
| --- | --- | --- |
| CPU 契约 | 开关/层匹配、schema、专家映射、零权重、错误 dtype/shape/hash、缺失/重复 tensor、确定性转换 | 正例逐项一致，异常明确非零退出 |
| 量化算术 | 小尺寸 INT8 用 CPU INT64 独立累加，按相同 scale、bias 和输出 dtype 对照 | 整数点积一致，缩放与输出按对应浮点契约比较 |
| A3 API | 实际 K/N、M=1/小尾部/典型 token 数、有无 bias、非连续、全零、空组 | shape/dtype 正确，结果 finite，空组处理正确 |
| 层与 block | 固定同一 BF16 捕获输入，分别对普通层、实际模态专家及至少一个完整 block 比较 | relL2、cosine、最大/平均绝对误差满足冻结阈值，NaN/Inf=0 |
| 关闭/回退 | 默认 BF16、显式排除层、非法元数据 | 默认行为一致，排除层使用原 BF16，元数据错误失败 |
| 完整音画 | 单人、多人、肢体动作、镜头运动、复杂背景，A/B 匿名随机顺序 | 两名指定验收人均确认没有稳定可辨的质量退化 |

数值采样覆盖去噪早/中/晚阶段及 conditional/unconditional 两条 CFG 分支。相对 L2 定义为 `||q-ref||2/max(||ref||2,eps)`；零范数参考采用绝对误差。正式评测前将抽样层、误差上限、零值容差和数据版本写入 `acceptance` 配置，由评审确认后锁定；评测运行读取该配置并逐项输出结论。

检查分为同输入的单层局部误差，以及 BF16/INT8 各自完整去噪轨迹中的 block 误差；普通层和 video/audio/text 分组分别汇总，保留逐样本指标与最差值。阈值提案依据独立开发集与 BF16 重复误差分布形成，固定后用于最终清单；任一必须项超限即非零退出并输出对应层、专家、step 和 CFG 分支。

## 性能、存储与内存标准

| 项目 | 测量方法 | 通过标准 |
| --- | --- | --- |
| Base/Decode/Total | 同机同源码同配置，每档至少 1 次预热、3 次完整 5 秒生成，关闭 profiler/hooks，NPU 同步 | 阶段 Total 中位下降超过锁定的 BF16 波动 |
| 完整请求 | 模型已加载，在 `run_offline` 前同步计时，音视频写出并同步后结束 | 单独报告中位、均值、极差和改善比例 |
| 目标 MatMul/block | 两档使用相同采样窗口，通过 flow 关联真实设备算子 | 目标 MatMul 或 block 累计耗时下降，确认真实 INT8 执行 |
| 制品存储 | 统计同范围完整 DiT 制品：量化权重、scale、倍率、元数据、BF16 保留状态 | INT8 制品总字节数小于 BF16 |
| 连续生成 | 同进程完成五次生成，在相同清理/同步点记录 NPU allocated/reserved 和 Host RSS | 内存进入稳定区间，排查持续增长 |
| 重复加载 | 同进程至少五次加载、生成、释放，统一 GC/同步条件，记录每轮加载前后和释放后内存 | 释放后占用保持稳定，增长项有定位与修复记录 |

基线波动采用 BF16 三次正式样本的极差；阶段 Total 的设计判据为 `median(T_BF16)-median(T_INT8) > max(T_BF16)-min(T_BF16)`。完整请求墙钟另按相同公式评估。冷启动、加载和离线转换独立计时；变更源码、制品或环境时重建两档配对结果。

阶段 Total 定义为原 timer 的 Step1→8 各阶段间隔之和，保存字段 `stage1_to8_seconds`；同时保存 `base_plus_decode_seconds` 和完整请求墙钟。原始日志、各阶段边界、三种统计口径随报告一并提供。

内存测试分别配置预热轮数、测量轮数和 `max_host_growth_bytes`、`max_npu_allocated_growth_bytes`、`max_npu_reserved_growth_bytes`。每轮在相同清理与同步点计算相对预热末值的增长，并报告最差增长、逐轮序列和趋势；检查器按冻结上限逐项判定，持续增长项进入定位流程。阈值配置随环境与测试清单一起评审固定。

Profiler 使用 `warmup=2, active=1` 捕获第三个去噪步的两次 CFG 前向，分别报告量化、MatMul、投影和 block 范围。融合反量化计入 MatMul；各级嵌套范围分别呈现，采样结果按实际窗口报告。

## 方案可行性依据

U 已完成同源码、同配置的 BF16/U 各 9 次开发生成，以及独立设备 profiling；以下为该版本的设计依据。

| 项目 | BF16 | U |
| --- | ---: | ---: |
| 三次驻留完整请求中位 | 35.060751 s | 30.295579 s，耗时降低 13.59% |
| Base 去噪中位 | 33.695605 s | 28.883562 s，耗时降低 14.28% |
| Decode 中位 | 0.770833 s | 0.771732 s |
| base DiT 制品字节数 | 30,642,072,600 B | 23,239,208,056 B，减少 24.16% |
| 采样步选中 up 投影的设备算子累计 | 316.383 ms | 162.585 ms，降低 48.61% |

Profiler 在选中 up 投影中确认 112 次 QuantMatmul 和 112 次 DynamicQuant，其余 120 个投影使用 BF16。187 项 CPU 回归通过；标准 `bash infer.sh int8` 已在 A3 完成完整生成，其输出与开发 U 对应视频 SHA256 相同。五类开发视频已获用户整体正面反馈。正式验收执行前述独立数据、冻结阈值和指定双人盲测流程。

## 兼容性、风险与处理

| 风险 | 设计处理 |
| --- | --- |
| 视频或音频对量化敏感 | 保留 attention/down BF16，冻结层表；以独立五场景数值与双人盲测评价 |
| 平滑倍率和转换权重不一致 | 用实际存储的 BF16 倍率计算 FP32 W/d，schema/校准哈希严格绑定 |
| 专家顺序或空模态错配 | 使用原 dispatcher/undispatch，逐专家切分权重、scale 和倍率，空组直接返回 |
| 权重加载峰值或持久副本 | 分片转换、meta 构造、量化 buffer 直接加载，配合重复加载释放测试 |
| SDK 版本差异 | 保存环境锁；按社区确认组合重建 BF16/INT8 同机基线 |
| 默认路径回归 | 默认关闭量化，原 base 参数与旧配置兼容，CPU 契约与标准入口分别检查 |

## 评审确认与交付

本次设计提交 U 的固定算法、层表、数据契约、工程接口和验证流程。评审集中确认两项：CANN 9.1/TorchNPU 26.1 的验收环境口径；开源校准/盲测清单、数值阈值及两名验收人的登记。确认项写入版本化评测配置，再执行整套正式验证。

设计 PR 仅包含任务个人目录下的 `docs/design.md`。模型实现交付到 `cann-recipes-infer`，包含标准入口、转换与严格加载工具、测试、README 和模型优化文档。源码、配置、权重/数据清单及日志以 SHA256 关联；权重、视频通过获取方式与校验信息提供。

开发者私仓 `task_submission/` 提供：

- `1 自验证步骤说明.md`
- `2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`
- `3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`
- `4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`

交付检查覆盖修改文件规范、仓库 CI、标准命令复现、量化真实性、数值与音画质量、性能以及内存/存储，逐项关联原始记录和对应源码版本。
