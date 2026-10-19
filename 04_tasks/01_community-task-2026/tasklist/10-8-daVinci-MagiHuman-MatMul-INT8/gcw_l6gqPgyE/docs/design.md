# daVinci-MagiHuman MatMul INT8 量化（A3）设计文档

2026-10-18 修订：按更新版任务书，将指定双人盲测要求同步为至少五组固定样本的主观质量对比记录及结论；其余数值、真实性、性能和存储要求保持不变。本文件仍为设计评审提案，不代替实现与实机验收。

版本：V1.1；日期：2026-10-18；提交者：gcw_l6gqPgyE。

状态：设计评审提案。本文覆盖任务书与源码静态核对，不作为 A3 算子探针、权重转换、模型运行、精度/性能/内存测试或固定样本主观质量对比的验收证明。文中数值阈值和工程接口属于拟实施契约，不是已取得的验收结果。实现前须在 cann-recipes-infer RFC/Issue 中确认并冻结；本设计 PR 不代替实现 PR 或实机验收。

# 1. 需求背景（required）

## 1.1 需求来源与版本

依据提供的《daVinci-MagiHuman MatMul INT8 量化任务书（A3）》及其 `test_cases/README.md`。原任务包 SHA256：`88b944583780e4a4535fde8e5156b1b348b904e123e253e732448fa549aebf57`；质量要求对齐的2026-10-18更新包 SHA256：`6d9a4c8f83df01b2ef5d04f6322eb9e75d89bd4c78b46bdd26aa8d85b2ce79ec`。新版任务正文 SHA256：`b62750d12c7a113e0d157c30a0304e5d03b4e34547f6e71d7f9dc8e3724679a1`；新版测试说明 SHA256：`97bc29fa5660d48dcd6ab95167b9042f326145b3b74d99b85499b4b5a79e60e9`。

| 对象 | 设计核对版本 / 约束 |
| --- | --- |
| infer 源码 | `cann/cann-recipes-infer` commit `93527538f1567a4cf42db58883231e4e00f8b213` |
| 上游补全源码 | daVinci-MagiHuman commit `209209b7086eba2020c5439265221495a8357322`，按 README 非覆盖合并 `inference/` |
| 设计提交仓 | `cann/cann-ops-competitions`，任务目录 `10-8-daVinci-MagiHuman-MatMul-INT8` |
| 后续实现仓 | `cann/cann-recipes-infer`，以模型私有修改为主 |
| 目标环境 | Atlas A3，Python 3.12、CANN 9.0.0、PyTorch 2.10.0、torch_npu v26.0.0 |
| 验收场景 | 单卡、world_size=1、cp_size=1，base 文生视频，448×256、5 秒、32 steps、CFG=2、seed=42、NoCache |

上述 infer commit 是源码设计基点；正式 BF16 基线必须记录其实际运行 commit 和完整补全清单。升级基点须重跑成对基线，不能把旧版本性能与新版本 INT8 对比。硬件 SKU、逻辑设备/物理卡映射、HBM、驱动、固件、MagiCompiler commit 和权重 hash 均待实机采集，不预填为已知事实。

## 1.2 现状及问题

`dit_module.py` 中 `BaseLinear` 调用 `_BF16ComputeLinear`；`NativeMoELinear` 按模态拆分输入、按输出通道拆分权重与 bias，再逐专家计算和拼接。`create_linear` 根据专家数选择类。Attention 的 `linear_qkv`、`linear_proj` 和 MLP 的 `up_gate_proj`、`down_proj` 是候选量化层，最终热点排序必须由 BF16 profiling 得到，不能用参数量代替耗时。

当前实际约束：

- 权重逻辑布局是 `[E*N,K]`；单专家 E=1，多模态层 E=3。这里的专家按模态分发，不额外引入 top-k 路由或改变 token 排列。
- `get_dit(model_config, engine_config, device=None)` 先构造 BF16 模型，再 `load_model_checkpoint`、迁移设备及 `eval()`。单纯把 INT8 tensor 塞入 BF16 Parameter 会发生 dtype 转换，不能得到量化执行。
- 补全源码的 `MagiPipelineConfig` 尚无 `quantization_config` 字段，`entry.py` 也未把量化配置传给 `get_dit`；仅新增 JSON 字段不足以生效。
- `infer.sh` 有模式白名单，当前不接受 `int8`。`base.yaml` 不显式写 seed；验收配置必须两边显式固定为 42。
- `module/quantization/compressed_tensors/compressed_tensors_w8a8_int8.py` 已使用 `npu_dynamic_quant` 与 `npu_quant_matmul`，可复用其接口契约，但依赖 layer 参数、加载回调、布局和 scale dtype，不能不加适配直接替换模型类。
- 当前 `models/sana-video/patches/quant_module.py` 的 `a8w8` 分支使用 `float8_e4m3fn` 和 MX scale，不等于本任务的 INT8。只参考组织形式，不采用该数值格式作为 INT8 证据。

# 2. 需求分析（required）

## 2.1 目标与边界

在原 BF16 base 行为可恢复的前提下，将评审选定的 DiT 线性层替换为真实 INT8×INT8 计算；权重、scale、层选择和转换结果可审计；满足逐层/DiT block 数值检查、至少五组固定样本主观质量对比记录及结论、端到端性能提升和权重存储下降。

不开发独立 kernel、aclnn 接口或 tiling 工程；不改 RoPE、RMSNorm、FA/Softmax、残差、激活函数；不量化文本编码器、音频模型、VAE、输入 adapter 和最终输出投影。图像输入、distill、超分、多卡属于本版未验证场景，开启量化时拒绝未支持配置；关闭量化时保持这些既有模式的原路径。

## 2.2 需求到验证的映射

| 编号 | 任务要求 | 设计落点 | 验收证据 |
| --- | --- | --- | --- |
| R01 | 固定基线/RFC | 第 1、7 节 | 环境、配置、输入、源码与四类权重 manifest |
| R02 | 热点与层范围 | 第 3.1、7.1 节 | 按层/专家/shape 的次数、耗时、占比与最终清单 |
| R03 | 真实 INT8 契约 | 第 3.2–3.4 节 | dtype/shape 断言、算子探针、profiler 对应关系 |
| R04 | 权重转换/加载 | 第 4 节 | 产物 hash、重复转换、缺失/错误注入测试 |
| R05 | 配置/忽略层/回退 | 第 5 节 | 开关、忽略规则、显式 BF16 回退日志 |
| R06 | 普通/专家/多模态 | 第 3.4 节 | 每条实际启用路径及空模态测试 |
| R07 | 中间数值与质量 | 第 6.1–6.3 节 | 误差明细、至少 5 组固定样本主观质量对比记录与结论 |
| R08 | 性能与内存 | 第 6.4–6.5 节 | 原始计时、算子证据、权重字节数、内存曲线 |
| R09 | 公共模块兼容 | 第 5.3、6.6 节 | 影响清单及全部直接调用方回归 |
| R10 | 工程与交付 | 第 7 节 | 文档、CI、测试、私仓七项材料 |

# 3. 详细设计（required）

## 3.1 总体流程

```text
固定源码/权重/环境/输入 -> BF16 base + 热点与数值采样
  -> A3 INT8 能力探针 -> RFC 冻结层清单/阈值/方案
  -> 离线转换 + manifest -> 严格加载 INT8 模型
  -> 动态量化 -> INT8 MatMul -> 恢复 BF16 边界
  -> 数值/异常/质量 -> 独立计时与 profiler -> 内存/制品验证
```

首选方案：对称 W8A8，权重按输出通道离线量化，激活按 token 在线动态量化，INT32 累加，输出恢复原边界 dtype。第一版不使用静态激活 scale、不启用 SmoothQuant，也不引入 grouped MatMul；逐专家复用单个量化线性接口，先确保分发语义正确。若能力/精度/开销不满足，必须通过 RFC 修订方案或层范围，不能静默更换成 FP8、W8A16 或全 BF16 后宣称完成。

初始候选集合为 `block.layers.*.attention.{linear_qkv,linear_proj}` 和 `block.layers.*.mlp.{up_gate_proj,down_proj}`。候选不代表全部通过；将每个全限定层名、E/N/K、调用次数、输入 M 分布、BF16 耗时、输出误差和排除原因写入 `layer_inventory.json`。实际存在的普通层、专家层、QKV/输出/MLP 类别均须评估并在最终清单逐项解释；不可只选最容易的一层作为全部覆盖。

## 3.2 数学与存储契约

对于一个专家，输入 X 为 `[M,K]`，原权重 W 为 `[N,K]`，参考输出为 `Y = X @ W.T + b`。权重与激活均 signed INT8、zero_point=0。

```text
s_w[n] = max_k(abs(W[n,k])) / 127
Q_w[n,k] = clip(round_to_nearest_even(W[n,k] / s_w[n]), -127, 127)
s_x[m] = max_k(abs(X[m,k])) / 127
Q_x[m,k] = clip(round_to_nearest_even(X[m,k] / s_x[m]), -127, 127)
S[m,n] = sum_k(int32(Q_x[m,k]) * int32(Q_w[n,k]))
Y_hat[m,n] = S[m,n] * s_x[m] * s_w[n] + b[n]
```

权重量化归约与舍入在 FP32 中执行；非有限原权重/激活直接报错。仅对确认为全零的行采用 `scale=1, Q=0`，记录零行 mask；这不是为缺失 scale 补占位值。非零行 scale 下溢至零时拒绝转换，不能隐式替换成 1。

| 张量 | 文件逻辑 dtype/shape | 运行时约束 |
| --- | --- | --- |
| 输入 | BF16 `[... ,K]` | reshape 为 `[M,K]`，保存前导 shape；连续性显式处理 |
| 激活 Q_x | INT8 `[M,K]` | 在线计算，禁止转回 BF16 后调用浮点 MatMul |
| 激活 s_x | FP32 `[M]` | 每次输入重算，不复用上一步 token scale |
| 权重 Q_w | INT8 `[E*N,K]` | 每专家 `[N,K]`，加载后转成后端所需 `[K,N]` |
| 权重 s_w | FP32 `[E*N]` | 与输出通道一一对应，专家片段为 `[N]` |
| bias | 保持源 bias dtype/shape `[E*N]` 或不存在 | 首版单独相加，避免混用 INT32 bias 语义 |
| 输出 | 默认 input.dtype；支持显式 output_dtype | 恢复 `[...,N]`，不改变现有层调用签名 |

真实算子的动态量化舍入/饱和及全零行行为以目标版本能力探针核实。若原生 `npu_dynamic_quant` 的契约与上式有差别，必须在 RFC 和 CPU oracle 中明确实际差异并重新验证，不能让文档与后端行为不一致。INT32 溢出保守检查 `K*127*127 <= 2^31-1`，超范围拒绝本版支持。

## 3.3 后端适配及能力门禁

拟复用仓内调用形式：

```python
# 接口示意，不是已经运行成功的实现。
qx, sx = torch_npu.npu_dynamic_quant(x_bf16_2d)
y = torch_npu.npu_quant_matmul(
    qx, qweight_kn, runtime_weight_scale,
    pertoken_scale=sx.reshape(-1), bias=None,
    output_dtype=torch.bfloat16,
)
```

`runtime_weight_scale` 的 dtype 和布局不能凭示例推断。共享实现对 BF16 路径可采用 BF16 scale；本设计文件存 FP32 主 scale，能力探针在 A3/v26.0.0 验证 FP32/BF16 scale 的合法组合，记录实际转换与舍入误差。ND 为首个候选，NZ/格式 29 仅在目标设备确认后启用，并在加载时一次性转置/打包，不在 forward 重复搬运。

探针覆盖所有候选 N/K 与代表性 M（含 M=1、空模态、尾部/非对齐），检查 INT8 输入、输出 dtype、scale 方向、误差、同步错误和 profiler kernel。探针失败则停止量化实施，不以 dtype cast 或假量化代替。实现后 profiler 须将层名/专家 ID、调用次数、shape 与真实量化 MatMul 事件关联；不只凭 Python API 名判断。

bias 在量化 MatMul 后独立处理：无 bias 时直接输出；有 bias 时以 FP32 做 `y.float()+bias.float()` 后转目标 dtype，纳入与 `_BF16ComputeLinear` 的数值比较。FP32 请求以已验证 BF16 MatMul 结果提升得到，记录中间舍入；如不满足约定误差则该层保持原路径，不宣称无损 FP32 累积输出。

## 3.4 专家、模态及布局

保持原 `ModalityDispatcher.dispatch/undispatch` 和外层 permute/inv_permute 顺序。按 `expert_id` 从 Q_w/s_w/bias 取相同 `[e*N:(e+1)*N]` 片段；输入使用原分组 X_e，逐组计算并按原顺序拼接。

M_e=0 时直接生成同 device/dtype 的 `[0,N]` 输出，不调用不支持空张量的量化算子。测试仅文本、文本+视频、文本+音频及三模态混合，验证零长度、顺序恢复和专家数与分组数不一致时的报错。不能把 E 合并进 K，不能将三个专家的 scale 广播为同一份。

QKV 权重的输出通道顺序由现有 `q_size/kv_size/gating_size` 定义；保持原融合顺序，不重新排列。MLP up-gate 的 interleaved 激活布局原样保留；分别测试 gated 与非 gated 层。首版支持单 rank，不承诺 TP/PP 权重切分；manifest 仍记录 shard/rank 和轴信息，检测到不支持拓扑时拒绝加载。

# 4. 权重产物与加载设计

## 4.1 离线转换

新增模型私有 `inference/model/quantization_layers/weight_convert.py`。读取原 safetensors 及索引，按排序后的张量名逐 shard 转换，输出独立目录，绝不覆盖源权重。同输入 hash、工具版本、配置和清单应得到相同张量 hash；序列化键和 shard 划分固定，manifest 不混入时间戳影响内容 hash。运行时间等信息另存日志。

INT8 制品应是可独立加载的混合精度 DiT checkpoint：目标层 Q_w/scale，忽略层及非目标层保留原 dtype，且无已量化层的多余 BF16 副本。原始权重仍可在外部供关闭量化使用，不计入 INT8 制品；依赖的未量化权重不得遗漏于存储统计。

动态激活方案不需要离线拟合 activation scale，但必须有独立的校准/敏感度样本集用于选层及确定误差表现。拟采用固定上游 commit 的开放 prompts 作为来源，核对许可、排除最终固定质量样本后，以固定排序/seed 42 选至少 32 条覆盖人物/动作/场景的样本；不足 32 条则停止冻结，补充经许可的公开来源并更新 RFC，不能虚构样本。保存 source/revision、文本、抽样规则及 SHA256。采集早/中/晚扩散步与实际模态分支，冻结后不得根据最终质量对比结果挑换样本。

拟定命令接口（实现阶段交付，目前不可作为已存在脚本运行）：

```bash
python -m inference.model.quantization_layers.weight_convert \
  --source "$BF16_DIT_DIR" --output "$INT8_DIT_DIR" \
  --layer-manifest "$LAYER_MANIFEST" --calibration-manifest "$CALIB_MANIFEST" \
  --scheme w8a8_dynamic_per_token --seed 42
```

## 4.2 元数据最小字段

`manifest.json` 包含 `schema_version`、源仓/上游/转换器 commit、源权重 revision 与逐文件 SHA256、scheme、signedness、zero_point、rounding、qmin/qmax、scale dtype/粒度/方向、目标平台、原始拓扑、校准清单 hash、目标与 ignored 精确层名、输出文件 hash 和字节数。

每层条目包含 `layer_name`、类名、E/N/K、源/目标 tensor key、logical_shape、layout、expert_offsets、scale_shape、bias dtype/shape、全零通道 mask、权重/scale tensor hash 和保留 BF16 的原因。用于运行时的 NZ 等打包信息与文件逻辑布局分开记录，不能把设备专有格式无说明地写作标准 `[N,K]` 权重。

## 4.3 加载状态机

```text
配置解析 -> 校验模式/拓扑 -> 校验 manifest/文件 hash
 -> 构造目标层及非训练 INT8 buffers -> 严格校验 key/shape/dtype
 -> 分 shard 装载 -> 转置/设备打包 -> READY -> 推理
任一步失败 -> 明确异常、非零退出；不得进入半加载状态
```

`enabled=false` 时完全调用原 `load_model_checkpoint`。`enabled=true` 时由模型私有 loader 读取完整混合制品，不复用其 `strict=False` 来忽略 INT8 key；所有缺失/额外 key、权重浮点误装、scale 缺失/非有限/非正数/shape 错误、错专家数、重复或冲突映射、hash 不符、schema 不兼容均拒绝。禁止用随机初始化权重继续运行。

在 `DiTModel` 构造后、原权重加载前替换精确目标模块；INT8 buffers 使用整数 dtype，scale 转设备时单独处理，避免整体 `.to(bfloat16)` 改写 scale 契约。释放旧 BF16 层引用再加载目标 shard，控制 Host 峰值；若构造期已有 BF16 分配，记录峰值并迭代 meta/lazy 构造，不能只比较稳态 HBM。

首版回退是启动前确定的 `ignored_layers`：该层从混合制品装载原 BF16 权重，并输出带原因的实际路由清单。元数据损坏不是精度回退，不自动降级。运行中捕获异常后静默重算 BF16 不允许。关闭量化重新加载原 checkpoint；不常驻整套双份权重。

# 5. 配置、接口与文件组织

## 5.1 配置契约

从 base 配置复制 `config/int8.yaml`、`config/int8_config.json`，保留所有生成参数及权重路径语义，在 YAML 的 `model_args` 显式写 `seed: 42`。base 对照同样显式固定该 seed。除了量化配置、量化制品路径、配置选择文件及输出目录，最终解析后的两份配置必须一致，包括 turbo VAE 开关和相关权重。

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
    "layer_manifest": "/path/to/frozen_layer_inventory.json",
    "ignored_layers": [],
    "fallback_dtype": "bfloat16",
    "invalid_metadata_policy": "error"
  }
}
```

路径均为示例，由用户配置，禁止硬编码开发机路径。缺少整个 quantization_config 等价于 enabled=false；开启时字段必须完备，不支持的 mode/粒度直接报错。ignored_layers 在冻结清单中展开为精确名称，零匹配/拼写错误/与制品冲突必须报错，防止一条通配符误把所有目标排除。最终 INT8 层为空时拒绝量化模式。

## 5.2 接入点

| 文件（相对 models/davinci-magihuman） | 计划改动 |
| --- | --- |
| `config/int8.yaml`, `config/int8_config.json` | 新增完整可启动配置 |
| `infer.sh` | 增加 int8 白名单/帮助；保持 base 等原模式 |
| `inference/common/config.py` | 对固定上游补全文件作最小模型本地适配：新增有类型的 QuantizationConfig 与顶层字段；默认关闭，序列化可审计 |
| `inference/pipeline/entry.py` | 将解析后的配置传至 get_dit，禁止仅读取原始 JSON 却不接运行路径 |
| `inference/model/dit/dit_model.py` | get_dit 新增可选 quantization_config；关闭走原加载器，开启走量化构造/加载流程 |
| `inference/model/quantization_layers/int8.py` | Int8Linear/Int8MoELinear；兼容 forward(input, output_dtype, modality_dispatcher) |
| `inference/model/quantization_layers/weight_convert.py` | 可复现转换与产物验证 |
| `inference/model/quantization_layers/loader.py` | manifest、精确替换、shard 加载与设备布局 |
| `tests/quantization/` | 数值/异常/配置/专家测试与报告脚本 |
| `README.md` | int8 使用、限制、复现入口 |

另在 `docs/models/davinci-magihuman/` 添加量化说明。上述新增类/路径是设计提案，不表示源码基点已有该实现。由于模型用非覆盖方式补全上游文件，新增本地 `config.py` 必须保留固定上游全部原功能与许可证，提供最小 diff，避免复制不同版本后掩盖差异。

## 5.3 复用与兼容性

先验证能否以模型私有适配器使用 `CompressedTensorsW8A8Int8LinearMethod` 的 create/process/apply 接口。它要求的 layer.weight、weight_scale、smooth_scales、bias 和加载状态必须显式满足；identity smooth_scales 只表示本方案不做平滑，不能代替量化 scale。若其 checkpoint/依赖耦合不适用，则在私有适配器复用同一底层 NPU 接口并记录原因，不为绕过耦合修改所有模型。

本版优先不修改 `module/quantization/`。确需修改公共模块时，先用 import/调用检索列出全部直接调用模型和配置，RFC 扩展影响范围，逐一完成功能、精度及代表性性能回归；旧配置、默认关闭与现有权重格式必须保持。未完成全部直接调用方回归不能申请实现合入。

# 6. 可维可测分析

## 6.1 数值标准（评审提案，实施前冻结）

以同一份 BF16 层输入分别执行参考层和 INT8 层，避免把上游误差误归给当前层；再执行累计误差的完整 block 检查。采集 early/middle/late 扩散步与每个实际专家路径，至少一个完整 DiT block（同时保存 residual 返回值），所有输出必须有限。

采用 FP32 计算 `NRMSE=||yq-yb||2/max(||yb||2,1e-12)`、余弦相似度及 max_abs；逐层逐专家逐采样点判定，不用总体均值掩盖失败。

| 对象 | 拟议数值门限 |
| --- | --- |
| 单线性层 | NRMSE ≤ 0.02 且 cosine ≥ 0.999 |
| 完整 DiT block | NRMSE ≤ 0.03 且 cosine ≥ 0.995 |
| 零参考张量（norm ≤ 1e-12） | 跳过无定义的 cosine，max_abs ≤ 1e-5 |
| 开关关闭 / ignored 层 | 同配置重复运行的 BF16 数值容差内一致，不经过量化路径 |

这些值不是任务书既定阈值，也未经过本模型实测；先做能力/校准阶段验证，再由 RFC 评审确认，之后不依据验收输出放宽。超限先定位 scale/布局/专家映射，再用独立校准集评估敏感层 BF16 回退；冻结清单变化必须重跑全部验收。

## 6.2 功能与异常矩阵

| 类别 | 必测项 | 期望 |
| --- | --- | --- |
| 配置 | 未配置、关闭、开启、未知模式、错误路径、空最终目标 | 关闭与原行为一致；非法启用明确失败 |
| 权重 | 正常 shard、重复 key、缺 key、错 hash/版本、INT8 被转浮点 | 正常加载；其余非零退出 |
| scale | 缺失、零/负/NaN/Inf、错 shape、错专家偏移、真实全零行 | 非法拒绝；零行遵守明确契约 |
| Linear | 真实所有 N/K、M=1、非连续输入、bias 开关、output_dtype | shape/dtype/数值符合契约 |
| MoE/模态 | E=1/E=3、空组、组数错误、置换恢复、各实际分支 | 无串专家/串模态；异常可诊断 |
| 边界 | QKV/gating 顺序、gated/non-gated MLP、残差边界 | 不改原语义 |
| 回退 | ignored 精确匹配、冲突/拼写错误、重新关闭量化 | 日志可审计；不静默自动回退 |
| 稳定性 | 连续至少 5 次生成、重复加载/卸载 | 无泄漏/NaN/Inf，输出正常 |

CPU 单测覆盖格式、转换确定性、公式与错误处理；NPU 测试覆盖后端能力和真实执行，二者证据分开。所有验证脚本遇到失败必须返回非零状态。

## 6.3 最终质量

冻结至少 5 组 prompt：单人、多人、肢体动作、镜头运动、复杂背景；记录完整文本/hash、seed=42 及相同生成参数。校准集与验收集不重叠，验收样本不能参与挑层/调阈值。

至少对五组固定样本进行 BF16/INT8 主观质量对比，覆盖单人、多人、肢体动作、镜头运动和复杂背景，并使用相同 seed 与生成参数。逐组记录主体、语义、构图、动作、时序、色彩、音画一致性和伪影八个维度、比较方法、观察限制及总体结论；不得伪造未完成的记录或结论。匿名 A/B 展示及身份映射可选，任务不强制盲测或指定评审人数。主观质量须与基线保持一致且无可稳定辨识退化；自动指标不能代替质量对比记录。

## 6.4 性能与真实性

同 A3 设备、软件、权重来源、batch、输入与生成设置；BF16/INT8 各至少预热 1 次、正式 3 次，每次完整生成 5 秒视频。正式计时关闭 profiler，成对交错运行，计时边界两侧 NPU 同步，使用单调时钟，记录 Base/Decode/Total 原始耗时、均值、中位数和极差。加载时间单列，Total 的起止边界两版一致，不通过异步未完成缩短时间。

设 BF16 三次 Total 为 B，INT8 为 Q，拟冻结保守波动量 `J=max(B)-min(B)`。通过要求同时满足 `median(Q)<median(B)` 且 `median(B)-median(Q)>J`。预定采用该判据，不在看到 INT8 结果后换分母/去异常点；若噪声过大，按预先确定的相同追加次数整批补测两版并保留全部原始值。3 次只能满足任务最低采样，不宣称统计显著性。

独立 profiler 运行保持两版一致开销；保存目标层、INT8 kernel 的次数和累计耗时、动态量化/布局转换/输出处理开销、目标 block 耗时。要求目标 MatMul 或目标 block 累计耗时下降，且端到端通过上述条件。量化算子减少时间但整体变慢不算通过。实际 BF16/INT8 层占比与回退原因随报告交付。

## 6.5 存储和内存

逐项统计原 BF16 DiT 权重、INT8 权重、scale、bias、未量化层、元数据及总制品实际字节数；采用相同模型边界，INT8 完整混合制品总大小必须小于原 BF16 制品。另报包括文本/音频/VAE 在内的部署总量，不能仅对已量化子集声称整模型减半。

记录加载前后、生成峰值、连续 5 次后的 NPU allocated/reserved 与 Host RSS，峰值统计在每次测量前复位，结合运行时/系统工具记录采样间隔。重复加载/卸载及连续生成若释放后仍持续增长则失败并定位缓存/引用；缓存平台期与线性增长分别展示，不只给一张峰值截图。任务未规定 HBM 必须下降百分比，不捏造指标；权重压缩也不等价于峰值内存下降。

## 6.6 工程验证

实现 PR 运行 `pre-commit run --all-files`、仓库 CI 和目标平台测试；新增文件保留来源与适用许可证，不带权重、视频大文件、密钥或机器私有路径。模型本地默认关闭回归覆盖既有 base 入口及配置解析；公共代码发生变化时执行第 5.3 节全部直接调用方回归。

# 7. 实施阶段与交付

## 7.1 阶段门禁

| 阶段 | 工作与产物 | 进入下一阶段条件 |
| --- | --- | --- |
| G0 基线 | 固定补全源码、环境、四类权重及 turbo VAE 资产；复现 base、采集 BF16 热点/波动 | 能完整生成，配置与证据可复现 |
| G1 RFC | INT8 能力探针、校准清单、数值阈值、层范围、bias/scale/布局契约 | RFC/Issue 评审冻结；不借用其他任务 RFC |
| G2 实现 | 转换/manifest、loader、普通与专家适配、入口和关闭路径 | CPU/NPU 功能异常测试通过 |
| G3 验证 | 数值、固定样本主观质量对比、性能、存储、内存、连续生成 | 全部任务指标达到，失败项不得跳过 |
| G4 提交 | 实现 PR 关联 RFC、完整报告与私仓材料、CI | 维护者评审与合入；设计 PR 本身不代表完成 G4 |

## 7.2 证据及私仓材料

统一证据清单至少包含源码/输入/配置/环境 manifest、四类权重 revision/hash、转换命令与 hash、候选和最终层清单、各层数值结果、异常日志、至少5组固定样本主观质量对比记录及结论、所有原始计时、profiler、内存/存储统计、CI 链接和失败复现命令。`use_turbo_vae=true` 时其配置和 checkpoint 也需固定，不能遗漏。

私仓 `task_submission/` 按任务要求提交：

1. `1 自验证步骤说明.md`
2. `2.1 精度自验证报告.xlsx`
3. `2.2 精度自验证日志.log`
4. `3.1 性能自验证报告.xlsx`
5. `3.2 性能自验证日志.log`
6. `4.1 内存自验证报告.xlsx`
7. `4.2 内存自验证日志.log`

报告每项标明 PASS/FAIL/NOT_RUN，原始大文件提供受控获取位置与 SHA256；不能用模板填充或静态检查替代真实结果。当前仅提交本设计文件。

## 7.3 风险与待确认项

| 风险 / 未决项 | 处理 |
| --- | --- |
| A3 后端 dtype/scale/layout 与参考代码不一致 | G1 能力探针锁定实际签名和支持 shape；不满足则修订 RFC |
| 动态量化与小专家调用开销抵消收益 | 按真实 M 分布评估，将回退选择固定在校准阶段；性能不达标如实报告 |
| 扩散累计误差及音画质量 | 层/block 数值和固定样本主观质量对比同时评估，保留敏感层 BF16 |
| 配置只写入文件但未进入模型 | 验证完整解析链路及运行时有效配置；打印精确实际层清单 |
| 量化加载 dtype 被模型迁移转换 | 检查加载前后 buffers 与算子输入 dtype，不只检查磁盘文件 |
| 上游补全版本漂移 | 固定 commit、非覆盖规则与完整合并 manifest |
| 正式 RFC 编号、权重/数据 hash、固定样本质量对比记录尚未落实 | G1/G3 门禁前补齐；本文不提供虚构编号或结果 |

# 8. 参考资料

- [社区设计提交流程](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [模型 README（设计基点）](https://gitcode.com/cann/cann-recipes-infer/blob/93527538f1567a4cf42db58883231e4e00f8b213/models/davinci-magihuman/README.md)
- [DiT 线性层与模态分发（设计基点）](https://gitcode.com/cann/cann-recipes-infer/blob/93527538f1567a4cf42db58883231e4e00f8b213/models/davinci-magihuman/inference/model/dit/dit_module.py)
- [现有 W8A8 INT8 接口（设计基点）](https://gitcode.com/cann/cann-recipes-infer/blob/93527538f1567a4cf42db58883231e4e00f8b213/module/quantization/compressed_tensors/compressed_tensors_w8a8_int8.py)
- [固定上游源码](https://github.com/GAIR-NLP/daVinci-MagiHuman/tree/209209b7086eba2020c5439265221495a8357322)
- [贡献规范](https://gitcode.com/cann/cann-recipes-infer/blob/93527538f1567a4cf42db58883231e4e00f8b213/CONTRIBUTION.md)
- [模型 Checklist](https://gitcode.com/cann/cann-recipes-infer/blob/93527538f1567a4cf42db58883231e4e00f8b213/docs/common/new_model_checklist.md)

AI 参与说明：本设计文档使用 OpenAI Codex 辅助进行任务拆解、源码静态分析和文档编写；源码事实、方案提议及待实机验证项在正文分别标识。
