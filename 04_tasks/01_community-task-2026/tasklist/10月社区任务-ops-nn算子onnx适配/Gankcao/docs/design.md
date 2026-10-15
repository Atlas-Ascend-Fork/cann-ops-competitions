# ops-nn 算子 ONNX 插件 protobuf 解耦设计计划

提交人：Gankcao。任务：10月社区任务-ops-nn算子onnx适配。

本文为实施前设计计划，说明拟采用的迁移方案、验证方法与交付安排，不包含开发过程、实测成绩或已完成声明。

# 需求背景（required）

## 1.1 需求来源

依据下发的《ops-nn算子onnx插件protobuf解耦任务书》及配套 `test_cases/README.md`、framework UT、OM 模型生成与两轮验证脚本实施。代码目标仓为 [cann/ops-nn](https://gitcode.com/cann/ops-nn)，需求参考 [issue #5115](https://gitcode.com/cann/ops-nn/issues/5115)。设计评审与验收按[社区任务流程](https://gitcode.com/org/cann/discussions/39)执行。

## 1.2 背景与目标

现有 ONNX 插件通过 `ParseParamsFn` 接收 protobuf 消息，依赖 `ge_onnx.pb.h` 和生成类型。拟将参数解析切换至 `ge::Operator` 输入及 JSON 属性读取，解除目标插件对 protobuf 生成代码的依赖，同时保持原有属性解析和注册行为。

本任务属于 C++ Host 侧框架适配，不新增计算算子。功能等价性定义为：对同一 ONNX 节点，除任务书明确要求纠正的失效失败判据外，迁移前后的返回状态、目标算子属性、输入输出及后续转换行为保持一致。

# 需求分析（required）

## 2.1 范围与约束

拟覆盖任务书列出的全部 75 个条目、74 个插件文件，清单见附录。`NPUDynamicQuant` 与 `NPUDynamicQuantV2` 共用一个文件；两个 Celu 插件按各自目录分别覆盖。相同文件中的全部注册版本均纳入核对。

修改限于对应 `framework/*_onnx_plugin.cpp`、framework UT，以及确有必要的公共工具或构建接入。不得修改 kernel、tiling、算子原型；保持 `ParseOpToGraphFn`、`DynamicInputRegister`、ONNX opset 注册范围、子图拓扑及非解耦逻辑不变。

开工时拟记录任务包哈希和源码完整提交号，对照任务书所依据的 `6e5b184` 快照与目标仓当前状态。若清单出现已迁移、移动或删除等变化，将记录差异并向社区核实，不静默减少范围，不以已经迁移的代码冒充迁移前基线。

## 2.2 接口和数据契约

```cpp
domi::Status ParseParamsOp(const ge::Operator& op_src, ge::Operator& op_dest);
```

上式为接口示意，实际保留每个插件原函数名称。注册拟由 `.ParseParamsFn(...)` 替换为 `.ParseParamsByOperatorFn(...)`；其余注册链保持原样。

| 输入/输出 | 含义 | 拟保持的契约 |
| --- | --- | --- |
| op_src | GE 源算子 | 通过 GetAttr("attribute") 获取 JSON 字符串；按需获取原本使用的节点名、输入个数等 |
| attribute | JSON 文本 | 外层对象包含 attribute 数组，每项按 name、type 和值字段解释 |
| op_dest | GE 目标算子 | 只写入原实现规定的属性、名称及描述信息 |
| 返回值 | SUCCESS/FAILED | 保留默认值、必填属性及校验分支的行为；JSON 解析和未知异常记录日志并返回 FAILED |

这层接口不定义新的 Tensor dtype/shape 范围；各算子的 dtype、shape、广播和动态输入能力继续由原算子与框架约束决定。

# 详细设计（required）

## 3.1 解析流程

拟逐插件执行以下步骤：确认原始默认值与失败条件；调整头文件及函数签名；读取 attribute JSON；按原顺序遍历属性；完成类型转换和既有校验；写入原有目标属性；保留注册版本；更新对应 UT。

拟删除目标插件中的 protobuf 消息类型、`ge_onnx.pb.h` 引用及直接 protobuf 解析，使用 `operator.h`、`nlohmann/json.hpp`、仓内工具头和 `register/register.h`。nlohmann/json 使用仓内 third_party 版本，不引入另一版本。公共工具优先复用 `common/inc/framework/plugin_util.h` 中的 `StrToFloat` 和 `GetOpName`；若确需新增公共头，采用仓前缀命名并保持职责最小。

节点名仅在原实现有相应写入行为的位置转换为 `op_dest.SetAttr("name", GetOpName(op_src))`；原本不写 name 的插件不新增该属性。原实现使用的输入数量等信息通过对应 Operator 接口获得，保留动态输入注册逻辑。

## 3.2 属性语义保持

| 情况 | 拟处理方式 |
| --- | --- |
| 属性整体不存在 | 保留算子原默认值；原本必填而缺失失败的属性仍失败 |
| FLOAT 属性 f 存在 | 按字符串读取，经 StrToFloat 转 float；空串、非法内容或越界按要求失败 |
| FLOAT 属性存在但 f 省略 | 按 proto3 零值省略语义取 0.0，不回退算子默认值 |
| INT 属性 i 省略 | 在原有匹配规则下按 proto3 零值语义处理 |
| ints/floats | 按数字数组读取，保留原有长度检查、扩维与重排 |
| type 不符 | 对原本检查 type 的属性按缺失处理；源码及配套 UT 明确不检查 type 的属性保留原行为 |
| 重复同名属性 | 保留首个命中、后者覆盖或累加的原始顺序语义，不统一去重 |
| 未知属性 | 保留原来的忽略行为 |
| JSON/转换异常 | 记录算子与属性上下文，返回 FAILED，不以空属性代替非法输入 |
| dtype 映射失败 | 核对工具函数真实返回约定，以 DT_UNDEFINED 等真实失败值判断，并补充负向 UT |

拟重点核对以下分支：HardSigmoid 的 alpha/beta 默认值与零值省略；Gemm 的 transA/transB 独立性及 fixed_shift_value 的 [34,43] 边界；Conv/Pool 的 pads 奇数长度、交错重排、维度提升；Softmax 不同 opset 的 axis；归一化属性改名与 epsilon；量化 dtype 映射及多输出；循环网络可选输入和属性列表。

## 3.3 工程接入及批量实施

拟建立“任务条目—源码文件—注册版本—属性契约—UT—模型脚本”映射，按基础标量、列表与形状、量化、归一化及循环网络等语义组实施。先以 HardSigmoid、Gemm、Conv、一个量化插件和无属性插件校准模式，再分组迁移其余文件。

批处理仅用于清点、机械替换候选、残留依赖扫描和结果汇总，逐文件复核语义差异。每组修改后运行对应 UT，最终执行完整 framework UT。UT 放在算子对应的 `tests/ut/framework/` 或 `common/tests/ut/framework/`，沿用仓内相对 include 方式，核对 common 集中目录改动能触发 CI。

构建计划执行 `bash build.sh --onnxplugin -j8`，UT 使用仓内 OP_HOST_UT 开关及实际构建入口，运行 `--gtest_filter='Onnx*PluginTest.*'`。提交前执行 `git diff --check`，检查范围内旧注册和 protobuf 引用残留，并核对依赖及符号。不会为消除无关仓库依赖而扩大修改范围。

## 3.4 环境与兼容性

拟使用 Atlas A2（Ascend 910B3）、CANN 9.1.0，同一硬件和软件环境获取迁移前后结果；满足任务书 CANN >9.0.0、A2 或 Ascend 950 任一验证要求。Python 拟使用 3.11，安装符合用例要求的 onnx、onnxruntime、numpy 并锁定实际版本，使用 CANN 自带 pyACL 推理。

迁移前若出现任务包所述 ge::onnx 未定义符号，仅按 compat 指引为旧插件提供兼容库，并记录适配方式及哈希。迁移后插件不加载该兼容库。两轮构建目录和自定义 OPP 目录隔离，防止旧库混入。

# 可维可测分析

## 4.1 验证范围与 UT

配套包含 74 个 UT 文件，源码实际定义 731 条 TEST；任务书正文的 726 与配套 README/源码计数有差异，计划以全部实际用例为覆盖清单，不删减用例。若官方更新用例，将记录版本变化。验收目标为全部执行通过，无未说明的禁用、跳过或过滤遗漏。

每个插件覆盖正常解析、属性缺失、非法 JSON、边界/校验四类场景，并断言返回状态和 op_dest 属性。UT 直接构造 ge::Operator，不构造 protobuf 对象。对失效 dtype 判据等必要修正补充失败用例；不通过修改断言来掩盖行为差异。

## 4.2 两轮 OM 验证

拟执行全部 74 个生成脚本及其自动生成的所有变体，形成实际模型清单，不将脚本数等同于模型数。输入按官方脚本和固定随机种子生成，标准随机输入使用正态 μ=0、σ=1，属性取可区分值；golden 使用脚本指定的 onnxruntime 或 numpy 参考实现。

1. 冻结迁移前源码、构建环境和插件库哈希，保存模型、输入、golden 和 meta.json。
2. 迁移前通过 ASCEND_CUSTOM_OPP_PATH 注入插件，执行 ATC `--framework=5 --soc_version=Ascend910B3`，保存完整转换日志及推理输出。
3. 迁移后使用同一模型和输入；核对文件哈希一致，再注入迁移后插件，重复转换与推理。
4. 逐例保留 LoadCustomOpLib 的本轮实际库路径及对应注册证据，确认不是回落到内置插件；证据不足不得记为完成。
5. 对输出数量、名称对应关系、shape、dtype 和数值逐项比对；汇总清单与预期模型清单对账，缺项不得记 PASS。

主判据为迁移前后输出完全一致，不设 rtol/atol。确定性输出按官方脚本字节一致方式核对，allclose 只作诊断，不能代替主判据。若实际用例涉及非确定性输出，而脚本例外口径与任务书主判据存在冲突，将保留数据并向社区确认，在确认前不按放宽标准宣称通过。

## 4.3 golden 辅助精度标准

逐输出按自身 dtype 选择阈值，避免将某个整型输出的零阈值应用到同模型的浮点输出。

| dtype | rtol | atol | fixed_limit |
| --- | --- | --- | --- |
| FP32 | 2^-10 | 2^-16 | 1e-2 |
| FP16 | 2^-9 | 2^-9 | 1e-1 |
| BF16 | 2^-6 | 2^-6 | 1 |
| 整数/布尔 | 0 | 0 | 精确一致 |

浮点逐元素匹配条件为 `abs(actual-golden) <= atol + rtol*abs(golden)`，整体匹配率须不低于 0.99；每个元素同时满足硬上限 `abs(actual-golden) <= max(fixed_limit, 32*ULP(abs(golden)))`。NaN 位置、Inf 位置和符号按官方标准检查，shape/dtype 不一致直接失败。不得关闭硬上限或自行放宽阈值。

`meta.golden=false` 的用例按任务允许的两轮行为一致性验证并写明原因。预期转换失败需两轮均失败且符合指定报错；平台 op-info 缺失等限制需保留两轮相同原因的报错和设备信息，单独列为行为一致，不计作数值精度通过。一轮成功、一轮失败须定位，不按平台问题豁免。

拟优先使用包内 two_round_compare.sh、run_om_and_compare_acl.py、compare_rounds.py 和 precision_standard.py，并补充完整性核验。对于脚本仅警告插件加载证据缺失、shape 声明数量不符、全局容差影响混合 dtype、未比较实际失败原因等情况，将保留原始工具，记录必要执行补丁及理由，保证判据不弱于任务书。

## 4.4 性能与内存

本任务不改变计算 kernel，任务书明确不设性能和内存验收指标。相关交付报告与日志将标注“不涉及（任务书 §3.3/§3.4）”，不设置 ratio 门槛，不填造测试成绩。

## 4.5 自查和证据

拟逐插件检查头文件、签名、注册、JSON 读取、节点名、默认值、失败分支、全部 opset、UT、构建和 diff。报告记录实际源码提交、插件 SHA256、软硬件版本、模型和输入哈希、各项日志路径。真实截图将从实际服务器终端或端口映射后的内置浏览器采集，展示环境、UT 汇总、两轮对照和插件加载证据。

# 实施安排与交付计划

| 阶段 | 工作 | 阶段完成条件 |
| --- | --- | --- |
| 设计评审 | 提交本文、确认范围及歧义 | 设计意见闭环 |
| 基线准备 | 固定版本、构建旧插件、生成模型和输入 | 基线可追溯，平台限制已记录 |
| 分组迁移 | 代表插件校准后覆盖全部文件 | 分组 UT 通过，范围完整 |
| 全量自验 | framework UT、两轮 OM、golden 辅助检查 | 全量清单对账，异常分类有证据 |
| 交付合入 | 自查、私仓报告、代码 PR 与 CI | 交付齐全，评审意见闭环 |

代码将按仓内规范分组提交到 ops-nn。正式代码 PR 仅包含必要源码、UT 及接入改动；大体积测试证据放入私仓交付目录与验收包。拟提供以下七个标准交付文件，并附单算子迁移自查表、模型生成和复现入口、原始日志及截图：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

# 附录：计划迁移范围

以下按任务书条目列示；每条将关联对应 UT 与 OM 记录，实施时核对实际源码版本。

| 编号 | ONNX 算子 | 插件相对路径 |
| --- | --- | --- |
| 1 | Celu | `activation/celu/framework/celu_onnx_plugin.cpp` |
| 2 | Celu | `activation/celu_v2/framework/celu_onnx_plugin.cpp` |
| 3 | NPUGeglu | `activation/ge_glu_v2/framework/npu_geglu_onnx_plugin.cpp` |
| 4 | HardSigmoid | `activation/hard_sigmoid/framework/hard_sigmoid_onnx_plugin.cpp` |
| 5 | LogSoftmax | `activation/log_softmax_v2/framework/logsoftmax_onnx_plugin.cpp` |
| 6 | Selu | `activation/selu/framework/selu_onnx_plugin.cpp` |
| 7 | Shrink | `activation/shrink/framework/shrink_onnx_plugin.cpp` |
| 8 | NPUSoftmaxCrossEntropyWithLogits | `activation/softmax_cross_entropy_with_logits/framework/npu_softmax_cross_entropy_with_logits_onnx_plugin.cpp` |
| 9 | Softmax | `activation/softmax_v2/framework/softmax_onnx_plugin.cpp` |
| 10 | ThresholdedRelu | `activation/threshold/framework/threshold_relu_onnx_plugin.cpp` |
| 11 | Int8Transpose | `common/src/framework/Int8Transpose_onnx_plugin.cpp` |
| 12 | AscendDequantS16 | `common/src/framework/ascend_dequant_s16_onnx_plugin.cpp` |
| 13 | AscendRequant | `common/src/framework/ascend_requant_onnx_plugin.cpp` |
| 14 | AscendRequantS16 | `common/src/framework/ascend_requant_s16_onnx_plugin.cpp` |
| 15 | AveragePool | `common/src/framework/average_pool_onnx_plugin.cpp` |
| 16 | BasicGRUInplaceFillWindowCache | `common/src/framework/basic_gru_inplace_fill_window_cache_onnx_plugin.cpp` |
| 17 | BasicLSTMInplaceFillWindowCache | `common/src/framework/basic_lstm_inplace_fill_window_cache_onnx_plugin.cpp` |
| 18 | NPUBoundingBoxEncode | `common/src/framework/bounding_box_encode_onnx_plugin.cpp` |
| 19 | ConstantOfShape | `common/src/framework/constantofshape_onnx_plugin.cpp` |
| 20 | Conv | `common/src/framework/conv_onnx_plugin.cpp` |
| 21 | ConvTranspose | `common/src/framework/conv_transpose_onnx_plugin.cpp` |
| 22 | DeformableConv2D | `common/src/framework/deformable_conv2d_onnx_plugin.cpp` |
| 23 | DynSeqOuter | `common/src/framework/dyn_seq_outer_onnx_plugin.cpp` |
| 24 | Gemm | `common/src/framework/gemm_onnx_plugin.cpp` |
| 25 | GlobalLpPool | `common/src/framework/global_lppool_onnx_plugin.cpp` |
| 26 | GlobalMaxPool | `common/src/framework/global_max_pool_onnx_plugin.cpp` |
| 27 | GRU | `common/src/framework/gru_onnx_plugin.cpp` |
| 28 | If | `common/src/framework/if_onnx_plugin.cpp` |
| 29 | Int8Conv | `common/src/framework/int8_conv_onnx_plugin.cpp` |
| 30 | Int8ConvRelu | `common/src/framework/int8_conv_relu_onnx_plugin.cpp` |
| 31 | Int8Dequantize | `common/src/framework/int8_dequantize_onnx_plugin.cpp` |
| 32 | Int8FC | `common/src/framework/int8_fc_onnx_plugin.cpp` |
| 33 | Int8GivenIntTensorFill | `common/src/framework/int8_giveint_tensorfill_onnx_plugin.cpp` |
| 34 | LayerNormalization | `common/src/framework/layer_normalization_onnx_plugin.cpp` |
| 35 | LpPool | `common/src/framework/lppool_onnx_plugin.cpp` |
| 36 | LRN | `common/src/framework/lrn_onnx_plugin.cpp` |
| 37 | LSTM | `common/src/framework/lstm_onnx_plugin.cpp` |
| 38 | MaxPool | `common/src/framework/max_pool_onnx_plugin.cpp` |
| 39 | MaxUnpool | `common/src/framework/maxunpool_onnx_plugin.cpp` |
| 40 | MeanVarianceNormalization | `common/src/framework/mean_variance_normalization_onnx_plugin.cpp` |
| 41 | MovingSumWithSigmoid | `common/src/framework/moving_sum_with_sigmoid_onnx_plugin.cpp` |
| 42 | NPUDropoutWithAddSoftmax | `common/src/framework/npu_dropout_with_add_softmax_onnx_plugin.cpp` |
| 43 | NPUGru | `common/src/framework/npu_gru_onnx_plugin.cpp` |
| 44 | ScatterND | `common/src/framework/scatter_nd_onnx_plugin.cpp` |
| 45 | SoftmaxCrossEntropyLoss | `common/src/framework/softmax_cross_entropy_loss_onnx_plugin.cpp` |
| 46 | SparseToDense | `common/src/framework/sparse_to_dense_onnx_plugin.cpp` |
| 47 | NPUDeformableConv2d | `conv/deformable_conv2d/framework/deformable_conv2d_onnx_plugin.cpp` |
| 48 | NPUFusedAddRmsNorm | `experimental/norm/fused_add_rms_norm/framework/npu_fused_add_rms_norm_onnx_plugin.cpp` |
| 49 | GatherElements | `index/gather_elements/framework/gather_elements_onnx_plugin.cpp` |
| 50 | GatherND | `index/gather_nd/framework/gather_nd_onnx_plugin.cpp` |
| 51 | Compress | `index/gather_v2/framework/compress_onnx_plugin.cpp` |
| 52 | Gather | `index/gather_v2/framework/gather_onnx_plugin.cpp` |
| 53 | NonZero | `index/non_zero/framework/non_zero_onnx_plugin.cpp` |
| 54 | NPUScatter | `index/scatter/framework/npu_scatter_onnx_plugin.cpp` |
| 55 | ScatterElements | `index/scatter_elements/framework/scatter_elements_onnx_plugin.cpp` |
| 56 | Unique | `index/unique_with_counts_and_sorting/framework/unique_onnx_plugin.cpp` |
| 57 | NPUAddRmsNorm | `norm/add_rms_norm/framework/npu_add_rms_norm_onnx_plugin.cpp` |
| 58 | BatchNormalization | `norm/batch_norm/framework/batch_normalization_onnx_plugin.cpp` |
| 59 | NPUGroupNormSilu | `norm/group_norm_silu/framework/npu_group_norm_silu_onnx_plugin.cpp` |
| 60 | InstanceNormalization | `norm/instance_norm_v3/framework/instance_norm_v3_onnx_plugin.cpp` |
| 61 | LpNormalization | `norm/lp_norm_v2/framework/lp_norm_v2_onnx_plugin.cpp` |
| 62 | NPURmsNorm | `norm/rms_norm/framework/npu_rms_norm_onnx_plugin.cpp` |
| 63 | AdaptiveAvgPool2d | `pooling/adaptive_avg_pool2d/framework/adaptive_average_pool2d_onnx_plugin.cpp` |
| 64 | AdaptiveMaxPool2d | `pooling/adaptive_max_pool2d/framework/adaptive_max_pool2d_onnx_plugin.cpp` |
| 65 | AscendAntiQuant | `quant/ascend_anti_quant/framework/ascend_anti_quant_onnx_plugin.cpp` |
| 66 | AscendDequant | `quant/ascend_dequant/framework/ascend_dequant_onnx_plugin.cpp` |
| 67 | AscendQuant | `quant/ascend_quant/framework/ascend_quant_onnx_plugin.cpp` |
| 68 | Int8Quantize | `quant/ascend_quant/framework/int8_quantize_onnx_plugin.cpp` |
| 69 | NPUAscendQuantV2 | `quant/ascend_quant_v2/framework/ascend_quant_v2_onnx_plugin.cpp` |
| 70 | NPUDynamicQuant | `quant/dynamic_quant/framework/npu_dynamic_quant_onnx_plugin.cpp` |
| 71 | NPUDynamicQuantV2 | `quant/dynamic_quant/framework/npu_dynamic_quant_onnx_plugin.cpp` |
| 72 | NPUGroupQuant | `quant/group_quant/framework/group_quant_onnx_plugin.cpp` |
| 73 | NPUQuantize | `quant/quantize/framework/npu_quantize_onnx_plugin.cpp` |
| 74 | NPULstm | `rnn/dynamic_rnn/framework/npu_lstm_onnx_plugin.cpp` |
| 75 | NPULstmCell | `rnn/dynamic_rnnv2/framework/npu_lstm_cell_onnx_plugin.cpp` |
