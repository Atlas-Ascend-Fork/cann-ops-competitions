# 10月社区任务：ops-nn 算子 ONNX 适配设计

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 10月社区任务-ops-nn算子onnx适配 |
| 提交者 | qyueqj |
| 实现目标仓 | [cann/ops-nn](https://gitcode.com/cann/ops-nn) |
| 文档阶段 | 设计评审；CANN 编译、UT 执行及 NPU 实机验证待完成 |
| 迁移范围 | 任务书指定的 74 个 ONNX 插件文件、75 个算子条目 |

# 需求背景（required）

## 需求来源

- [官方任务页面](https://www.hiascend.com/activities/task-center/details/de397754b8b7410790780df572bd62ae)。
- [官方任务书与验证用例包](https://www.hiascend.com/p/resource/202610/6d2e0cdb0bca4b078a271eae5b4a8290.zip)，其中 `ops-nn_onnx_task_doc.md` 为本设计的需求与验收依据。
- [社区任务提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)及[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
- [社区任务流程及注意事项 #39](https://gitcode.com/org/cann/discussions/39)：设计文档以 PR 提交到 competitions 对应目录，并在评论区通知指定评审人员。
- 迁移示例：[ops-nn issue #5115](https://gitcode.com/cann/ops-nn/issues/5115)，以及仓内已迁移的 Elu、LeakyRelu、BoundingBoxDecode、Bernoulli 插件及其 UT。

任务书的源码范围基于提交 `6e5b1844d08baee8c2bc9523bf87a05ec2bd9539`。本地准备基线为 `0fee5dd38b9ab973039a15bc0c2047ee9667b7c0`，已对照确认迁移清单中的旧插件、相关公共工具头和 UT 收集代码与任务书快照一致。正式验证固定迁移前版本，记录迁移前后提交及插件哈希；若上游更新导致清单或语义变化，按任务书要求通过 Issue 说明差异。

## 背景介绍

现有部分 ONNX 插件通过 `ParseParamsFn` 接收 protobuf 消息，并直接访问 `NodeProto`、`AttributeProto` 等类型。插件编译与运行因此依赖 ONNX protobuf 头文件、生成代码和相应符号。

本任务将参数解析入口切换为 `ge::Operator`，从框架传入的 `attribute` JSON 读取属性，并使用 `ParseParamsByOperatorFn` 注册。目标是在移除 ONNX 插件 protobuf 依赖的同时，保持原有参数解析、图构建以及模型转换结果。

本任务是 Host 侧框架适配，不新增算子计算公式、数据类型或 shape 能力。各算子的数学含义、Kernel、Tiling、原型、aclnn 接口和 `ParseOpToGraphFn` 子图构建逻辑保持不变。

# 需求分析（required）

## 需求描述

对任务书列出的全部插件完成入口、注册、属性读取和相关构建依赖迁移。对同一 ONNX 节点，迁移前后的返回状态、目标算子属性、动态输入输出注册及后续图结构应保持一致；对同一模型和输入，两轮 OM 推理输出应完全一致。

## 需求拆解

| 要求 | 设计落实方式 | 验证方式 |
| --- | --- | --- |
| 接口替换 | 保持解析函数原名，入参改为 `const ge::Operator&` 与 `ge::Operator&` | 源码检查、编译、framework UT |
| 注册替换 | 使用 `.ParseParamsByOperatorFn(...)`，保留原注册类型、opset 集合及图回调 | 清单核对、ATC 插件加载日志 |
| JSON 属性读取 | 使用仓内 nlohmann/json，保留属性缺省、类型与顺序语义 | 正常、缺省、异常、边界 UT |
| protobuf 解耦 | 移除 ONNX 公共头及生产插件目标上的 protobuf 依赖 | 源码扫描、实际构建、产物依赖与符号检查 |
| 行为保持 | 不修改 Kernel、Tiling、图构建和与迁移无关的逻辑 | 逐插件自查、迁移前后 OM 对比 |
| 覆盖完整 | 74 个文件全部纳入 UT 与 OM 验证，包含 experimental 插件 | 用例清单与实际执行结果逐项核对 |

# 详细设计（required）

## 算子分析

本设计保持各算子原有数学公式、输入输出 dtype、shape、广播及格式约束。迁移只改变 ONNX 节点属性进入 GE 算子的解析路径：

```text
迁移前：ONNX 节点 → protobuf 消息 → ParseParamsFn → GE 属性 / 原有图回调
迁移后：ONNX 节点 → ge::Operator + attribute JSON
                    → ParseParamsByOperatorFn → GE 属性 / 原有图回调
```

作用域以任务书的文件路径为准，不能仅按算子名称去重。两个 Celu 插件以及两个 DeformableConv2d 插件分别对应不同文件；NPUDynamicQuant 与 NPUDynamicQuantV2 共用一个文件，因此是 75 个条目、74 个文件。完整范围见附录。

## 算子实现

### Host 侧接口与注册

```cpp
domi::Status ParseParamsOp(const ge::Operator& op_src, ge::Operator& op_dest);
```

保持现有 `REGISTER_CUSTOM_OP`、`FrameworkType(ONNX)`、`OriginOpType`、`ImplyType` 以及需要的 `ParseOpToGraphFn`，仅替换参数解析注册接口。

原来使用 `node->name()` 的位置改用仓内 `GetOpName(op_src)`；原来依赖节点输入输出数量的位置使用 `GetInputsSize()` / `GetOutputsSize()`。保持原有 `DynamicInputRegister`、`DynamicOutputRegister`、`original_type` 和目标属性的写入条件，不向本来不写该属性的插件统一添加赋值。

### JSON 获取与公共工具

优先复用仓内 `plugin_util.h` 的 `StrToFloat`、`GetOpName` 等工具。拟新增带仓前缀的 `common/inc/framework/nn_onnx_json.h`，仅承载重复的属性数组获取、ONNX 属性类型常量与标量浮点读取，不抽象或重写各算子的解析逻辑。

通过 `op_src.GetAttr("attribute", ge::AscendString)` 取得 JSON 文本，解析其顶层 `attribute` 数组。例如：

```json
{"attribute":[{"name":"alpha","type":1,"f":"0.25"}]}
```

与仓内参考实现一致，未提供 `attribute`、或解析后的对象没有有效属性数组时按空属性集处理；语法非法、实际读取字段时发生类型或转换异常，由插件入口捕获 `std::exception` 及未知异常，记录日志并返回 `FAILED`。无需读取属性的插件保留原有常量赋值行为；是否解析 JSON 以原插件语义及对应官方 UT 为准，不一刀切新增失败分支。

### 属性语义保持

1. 区分“整个属性不存在”与“属性存在但标量字段缺省”。例如 HardSigmoid 的 alpha 属性不存在时保留 0.2；alpha 的 FLOAT 属性存在但无 `f` 字段时取 proto3 零值 0.0。
2. 标量 `f` 以字符串承载，调用 `StrToFloat`；非法字符串、空串或越界按失败处理。`i`、`ints`、`floats` 按任务书规定的数值表示读取，保留原来的转换及范围检查。
3. 按输入数组顺序处理重复属性，保持原插件覆盖、追加或跳过的顺序。未知属性及原有类型门控按原实现处理，不新增统一的拒绝策略。
4. 逐条核对默认值、显式零值、空列表、负数、维度、布局重排及输入输出数量。Conv 的奇数 pads、Gemm 的 fixed_shift_value 回退等分支分别保持其原有状态与属性结果。
5. 按任务书要求检查失败判据与工具函数真实返回值是否匹配。发现无效判据时单独说明修正，并增加非法取值 UT；不借此扩展到无关重构，也不将已有“成功并返回未定义类型”的兼容语义擅自改为失败。

### ConstantOfShape 的 Tensor 属性

该插件需要从 JSON 的 `t` 对象构造 GE Tensor，不能将 protobuf 指针直接替换为临时 JSON 字符串指针。

- 支持 Tensor 字段的 snake_case 与 protobuf JSON camelCase 表示，如 `data_type` / `dataType`、`raw_data` / `rawData`。
- `raw_data` 按 base64 解码；校验字符、长度与 padding。保留原始字节，不将二进制内容当文本数值转换。
- 非 raw 路径读取对应 typed data，兼容 int64 的十进制字符串，避免经过 double 损失大整数精度。
- 保留单标量、默认 float 零值、dtype 映射及重复 value 属性行为；短载荷等边界按官方 UT 的既定行为处理。
- 使用拥有生命周期的字节缓冲区，构造 Tensor 完成复制后再释放，避免悬空数据。
- 增加 raw float 实际数值、camelCase 字段、int64 极值字符串、double 数值和非法载荷等回归用例。GE Tensor 的真实构造与字节读取仍需 CANN UT 验证。

### 构建依赖与测试入口

从 ONNX 公共头中移除 `ge_onnx.pb.h` 及只服务于旧入口的 protobuf 工具。检查全仓 ONNX 插件对该公共头的使用，清理已迁移插件中残留且无用途的 `NodeProto` 别名，避免公共头变化导致非任务插件编译失败。

ONNX 生产插件目标改用仓内 JSON 依赖，移除 protobuf 生成依赖、include 目录、静态库链接及仅用于 protobuf 的 google 命名空间重映射。TensorFlow 等其他插件目标的依赖保持原状，不做全仓 protobuf 删除。

UT 放入仓库既有 framework 测试目录，补齐缺失的构建入口。按算子筛选测试时同时识别所属算子目录与历史测试文件名，覆盖 `fused_add_rms_norm` 对应 `test_npu_fused_add_rms_norm` 等名称不同的情况；不重复收集已经由公共 UT 入口收集的文件。

标准目录与 `experimental/norm/fused_add_rms_norm` 分别构建并留存产物。验证必须包含两类产物，不能因默认构建未包含 experimental 而漏验。

### Kernel、分核与 Tiling 设计

不涉及。本任务不修改设备端计算、分核策略、UB 切分、TilingKey、广播实现或指令调度，也不新增 aclnn API。

## 支持硬件

| 硬件 | 本任务验证安排 |
| --- | --- |
| Atlas A2 | 首选实机平台 |
| Ascend 950 | 可替代平台；任务书允许任一硬件验证通过即可 |

迁移前后在同一硬件、同一驱动和同一 CANN 环境中验证。CANN 版本大于 9.0.0，JSON 使用仓内版本；OM 推理优先采用 CANN 自带 pyACL，也可使用任务书允许的 ais_bench_infer 或 msame。

## 算子约束限制

保持原算子的类型、shape、格式及平台支持约束，不因 Host 解析迁移扩大设备端支持范围。对于部分 SOC 不具备 op-info 条目的算子，按任务书记录平台、算子及 ATC 原始错误，并比较迁移前后两轮错误原因；不能把任意两次转换失败视为一致。

# 可维可测分析

## 精度标准 / 性能标准

| 验收项 | 标准与证据 |
| --- | --- |
| 参数解析 UT | 全部任务插件覆盖正常、缺省、非法 JSON、边界及校验分支；用例使用 `ge::Operator`，不构造 protobuf 对象 |
| 编译 | `bash build.sh --onnxplugin -j8` 及 experimental 对应构建成功，保存完整日志 |
| protobuf 解耦 | 检查 ONNX 目标构建依赖、链接信息及插件符号，不再直接依赖旧 ONNX protobuf 类型 |
| 主精度判据 | 同一个 ONNX 模型、同一份输入，迁移前后 OM 输出 dtype、shape 和逐数值结果完全一致，不使用 rtol/atol 放宽 |
| 辅助 golden 判据 | 按官方混合容差：fp32 rtol=2^-10、atol=2^-16，逐元素硬上限及整体匹配率遵循用例包 §4.1，整数 / 布尔精确比较 |
| 无 golden 或平台限制 | 按任务书保留两轮行为一致性证据，记录实际原因，不伪造精度结果 |
| 性能与内存 | 不涉及；任务书明确为 Host 侧等价重构，不设此类验收指标 |

## 测试方案

### framework UT

任务包提供 74 个 UT 文件。本地对包内 `TEST` / `TEST_F` 声明计数为 731，任务书正文写作 726；以实际用例清单为准，保留此差异供评审核对，不删减官方用例。另拟补充 7 个 ConstantOfShape 回归用例，共 738 个预期用例。

开启 OP_HOST_UT 构建并运行 `Onnx*PluginTest.*`，分别覆盖标准和 experimental 目录；按测试套件名、用例名及文件建立清单，将实际 XML 与清单比对。缺失、跳过、未执行或失败的用例不能计为通过。除任务范围内的 UT 外，对受公共头修改影响的全仓 ONNX 插件执行构建检查。

### OM 两轮对比

1. 固定迁移前、迁移后源码及软件环境，分别构建并保存 standard / experimental 插件、哈希和编译日志。
2. 使用官方 74 个模型生成脚本及必要属性变体，覆盖全部迁移文件；固化模型与输入数据，校验两轮文件哈希相同。
3. 两轮 ATC 均通过 `ASCEND_CUSTOM_OPP_PATH` 注入对应插件，保存实际加载本轮插件的日志证据，避免误用 CANN 内置插件。
4. 转换成功后分别执行推理，先做迁移前后精确一致性比较，再检查与 onnxruntime / numpy golden 的混合容差。
5. 对不支持转换的算子核对两轮报错原因。脚本退出码为零、两轮都失败或两轮日志都有警告，均不能替代逐项验收结论。

若旧接口插件因 CANN 不导出 `ge::onnx::AttributeProto` 符号而加载失败，仅迁移前一轮按官方 `verify_om/compat/README.md` 使用兼容库，并记录环境。迁移后插件不得挂载该库，否则无法证明 protobuf 解耦。

## 兼容性分析

外部算子名称、原注册 opset、GE 属性名、动态输入输出数量、子图回调及设备端实现保持不变。保留现有异常与缺省行为；新增 JSON 解析失败路径限定在新的接口表示中。公共头和 CMake 改动需同时检查已迁移 ONNX 插件与其他框架目标，防止依赖移除扩大影响面。

对于任务书、官方 UT 与现行源码之间存在的差异，保留源码和用例依据并提交说明，不静默选择会改变验收语义的行为。设计评审如要求调整公共工具边界、兼容行为或提交分组，先修订本设计再更新实现。

## 当前阶段与交付计划

本次只提交设计文档供社区评审。本地已准备实现草稿与静态检查材料，但尚未完成 CANN 链接构建、实际 framework UT、ATC 转换或 NPU 推理；这些检查均不能宣称验收通过。

设计文档评审通过并合入后，根据意见修订实现，完成实机自验证，按社区流程提交验收材料并配合验收；按任务书及社区通知向 `ops-nn` 分组提交代码 PR。验收交付件包括迁移代码、UT 与执行结果、OM 转换及两轮精度记录、逐插件自查表，并在开发者仓 `task_submission` 中按官方模板保存自验证步骤、精度报告及原始日志；性能与内存项注明“不涉及”。

# 附录：迁移范围

以下清单来自本任务书；路径均相对于 `ops-nn` 仓库根目录。评审及执行时逐文件核对，NPUDynamicQuant 的两个条目共用一个插件文件。

| 序号 | 算子条目 | 插件路径 |
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
