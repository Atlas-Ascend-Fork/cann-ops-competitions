# ops-math 算子 ONNX 适配设计

作者：psjplznsm
状态：完整自验证完成；提交环境修订及证据供评审，SCA 审核待完成。
版本：2026-10-23

## 1. 需求背景

- [昇腾社区任务](https://www.hiascend.com/activities/task-center/details/18e1d3beae784812b2562460542ad6b3?menu=tasks)
- [任务书与官方用例包](https://www.hiascend.com/p/resource/202610/21e704e530834ebdb12a5c7a5d2baa51.zip)
- [迁移参考 Issue #2892](https://gitcode.com/cann/ops-math/issues/2892)
- [社区任务提交流程](https://gitcode.com/org/cann/discussions/39)

本任务将指定 ONNX 插件从依赖 Protobuf Message/NodeProto 的解析方式迁移到 GE Operator 属性接口，解除这些插件对 ONNX Protobuf 头文件和数据类型的依赖。范围以任务书清单为准，包含 57 个插件源文件、124 个“文件 × 算子”条目；同一文件可通过宏覆盖多个算子，不能用源代码中注册调用的数量代替算子覆盖数。

实现基线为 ops-math 提交 `b386997ada0f4450fe63fe2652f529807cea5261`。该基线在任务范围内有 79 处字面量 `.ParseParamsFn(` 调用。本次仅修改 Host 侧解析入口、属性读取、必要的公共辅助函数和测试，不修改算子原型、Tiling、Ascend C Kernel 或 `ParseOpToGraph*` 子图构建逻辑。

## 2. 需求分析

### 2.1 接口和数据流

原流程：ONNX 模型 → GE ONNX Parser → `ParseParamsFn(Message*, Operator&)` → NodeProto 属性 → GE 算子属性/动态输入输出 → 既有子图构建或算子编译。

目标流程：ONNX 模型 → GE ONNX Parser → `ParseParamsByOperatorFn(const Operator&, Operator&)` → `op_src.GetAttr("attribute", AscendString&)` 中的 JSON → GE 算子属性/动态输入输出 → 既有子图构建或算子编译。

源 Operator 的 `attribute` 字符串形如 `{"attribute":[{"name":"axis","type":2,"i":1}]}`。JSON 只作为框架与插件之间的属性载体，不引入新的对外参数。算子名称从 `GetOpName(op_src)` 读取；官方契约要求空名称回退的 Max、Min、Xor、Transpose 路径回退到目标名称。输入输出数量优先读取 GE 序列化的 `input` / `output` 字符串列表属性；直接构造 Operator 的调用者没有这些属性时，回退到张量描述符数量。

### 2.2 兼容约束

1. 保持每个算子已注册的 ONNX 版本、OriginOpType、目标算子、ImplyType 和子图回调不变。
2. 保持各插件原有的属性名称、默认值、属性类型筛选、重复属性遍历顺序、动态输入输出登记和返回码。不得以统一校验策略覆盖插件原本的差异。
3. 区分“属性不存在”和“属性存在但 Protobuf 默认字段省略”。例如没有 `max` 属性沿用 Clip 默认上限；存在 FLOAT 类型的 `max` 但缺少 `f` 字段时取 `0.0f`。
4. 无属性解析路径不为了形式统一强制解析 JSON。Max、Min、Xor 等官方用例要求校验 JSON 的路径，按任务测试契约处理。
5. 非法 JSON、字段类型错误及转换异常在解析边界记录错误并返回 `FAILED`；原有业务检查返回的 `PARAM_INVALID` 等状态保持不变。成功路径不会依赖异常控制流。
6. Cast 的无效目标类型使用 `GetOmDtypeFromOnnxDtype()` 的实际失败值 `DT_UNDEFINED` 判断，修正旧代码用 `-1` 判断而遗漏错误的情况，并由官方负向用例验证。

## 3. 详细设计

### 3.1 公共辅助函数

复用已有无 Protobuf 依赖的 `common/inc/framework/math_onnx_plugin_util.h`，避免新建平行公共框架。JSON 使用仓库已提供的 nlohmann/json 依赖及其编译命名空间设置，不另行引入不同版本。

| 辅助函数 | 职责 | 边界行为 |
| --- | --- | --- |
| `GetOnnxAttributes` | 读取源 Operator 的字符串属性，解析并取 `attribute` 数组 | 属性缺失、合法 JSON 无该键、非对象顶层或非数组集合均按无属性处理，符合官方 Mod 等 UT；非法 JSON 语法、非对象属性条目抛出异常，由插件捕获 |
| `GetOnnxAttributeInts` | 读取重复整数字段 | 缺失时返回空数组；非数组拒绝；索引读取使用受检访问 |
| `GetOnnxAttributeFloat` | 解析 GE JSON 中字符串或数字形式的 `f` | 缺失字段为 `0.0f`；拒绝布尔、其他错误类型和不完整数字；保留 float 精度及可表示的次正规数 |
| `GetOnnxInputCount` / `GetOnnxOutputCount` | 从 GE 的节点名称列表获取端口数量 | 名称列表优先，包括显式空列表；仅属性不存在时回退描述符数量 |
| `HasOnnxInput` | 判断指定输入位置是否具有非空名称 | 保留 Clip 可选输入的空占位；不压缩输入位置 |
| 已有 `GetOpName` / `Vec2Tensor` / `CreateScalar` / dtype 和格式函数 | 复用既有名称、常量张量、类型和格式处理 | 不改变其业务语义 |

INT 字段 `i` 和 INTS 数组沿用整数表示；STRING 字段 `s` 沿用字符串表示。`type` 数字标记使用明确的本地枚举常量替代 Protobuf 枚举。不能把 GE 的 `f` 字符串误当成 JSON number；重复 FLOATS 字段如有涉及，应按 GE 的数组表示处理。

每个需要属性的解析函数保持原有局部默认值及遍历结构，使用 `value()` 的显式类型默认值读取字段。只在外部解析边界捕获 `std::exception` 和未知异常，记录算子名及错误，避免异常穿过注册回调。保持原有初始化顺序：原先在遍历前登记动态端口、设置 `original_type` 的路径仍在 JSON 解析前完成这些操作，保留官方 UT 检查的失败后状态；不增添事务或改动子图。

### 3.2 插件迁移

- 将解析签名改为 `const ge::Operator& op_src, ge::Operator& op_dest`，注册改为 `ParseParamsByOperatorFn`。
- 删除 `Message`、`NodeProto`、Protobuf 属性枚举以及 `onnx_common.h` 依赖；显式包含实际使用的 GE、注册和公共辅助头文件。
- 使用 `op::Data`、`op::Const` 的子图文件显式包含已有 `stub_ops.h`，补足原先由 `onnx_common.h` 间接提供的声明。
- 保留属性检查的原有顺序与范围，保留例如首个匹配后退出、后一个重复属性覆盖前一个等差异。
- 无属性的宏注册及辅助入口只替换接口和输入输出来源，不额外添加 JSON 解析。
- `ParseOpToGraph*` 函数保持源码 token 等价；新增格式化导致的空白变化不视为业务变化。

实机补充 Clip 的次正规数属性触发了 GE 的数字回退：`Pb2Json` 使用 10 字节缓冲区格式化 FLOAT，格式化失败时直接把 FLOAT 保存为 JSON number。因此 helper 同时支持字符串和数值载体；增加数字、小数、零、负零和最小次正规数回归检查。依据为 [GE Pb2Json 源码](https://gitcode.com/cann/ge/blob/c723bd6ea62037d215d7a61e0d96bafa1ffa24da/parser/parser/common/convert/pb2json.cc) 与前后真实 ATC 日志。

### 3.3 输入输出与可选输入

GE 的 `Message2Operator::ParseOperatorAttrs` 将 NodeProto 的重复字符串字段保存为 Operator 列表属性；`OnnxParser` 传入回调的临时源 Operator 尚未登记张量描述符，直接调用 `GetInputsSize()` 可得到 0。实机 Concat 测试确认了这一差异，因此对 24 个相关插件使用公开 `GetAttr("input"/"output", vector<AscendString>&)` 读取名称列表。参考 [GE 数据转换实现](https://gitcode.com/cann/ge/blob/c723bd6ea62037d215d7a61e0d96bafa1ffa24da/parser/parser/common/convert/message2operator.cc) 与 [ONNX 调用入口](https://gitcode.com/cann/ge/blob/c723bd6ea62037d215d7a61e0d96bafa1ffa24da/parser/parser/onnx/onnx_parser.cc)。

Clip V11 通过名称列表保留 `['X', '', 'max']` 与 `['X', 'min', '']` 的位置语义；缺少列表时兼容官方 UT 的描述符构造方式。新增 4 项 Clip 回归测试和 17 项公共辅助函数测试，覆盖名称列表优先级、空列表、可选位置及 float 边界。补充 OM 模型覆盖 FP16/FP32 下双边界、仅 min、尾部空 max、仅 max 和无边界，不修改既有子图。

### 3.4 随机算子与历史行为

RandomNormal、RandomNormalLike、RandomUniform、RandomUniformLike 使用固定 seed 做迁移前后的 bitwise 对比，不因属于随机算子就允许精度豁免。

Multinomial 的旧子图存在属性未传递到内部算子的历史行为。本任务不修改该子图；按官方用例说明，使用输出 shape、dtype 和取值范围校验，并在结果表中明确记录适用条件。其他非确定性场景只有在同一实现重复运行已提供真实证据、且属于任务允许范围时才采用预先声明的误差标准。

## 4. 可维可测设计

### 4.1 构建与静态检查

实测环境为 Ascend910B2（A2）、CANN 9.1.0、驱动 26.1.1、Ubuntu 22.04 aarch64、Python 3.12.13、GCC 11.4、ONNX 1.17.0、NumPy 1.26.4；迁移前后使用同一环境。CANN 9.1.0 采用独立安装目录，旧版 CANN 9.0.0 不作为本任务的验收环境。

基础构建命令为 `bash build.sh --onnxplugin -j8`；实验目录相关目标显式启用 `--experimental`。Host UT 按仓库构建入口启用 `OP_HOST_UT`，运行过滤器 `--gtest_filter='Onnx*PluginTest.*'`，保留完整构建和执行日志。

静态检查覆盖 57 个目标文件中的旧接口/Protobuf 残留、注册字符串一致性、子图函数 token 一致性，以及 57 份官方 UT 的用例数与 token 一致性。格式与提交检查使用仓库规定的 clang-format 18.1.8 和 pre-commit。静态检查不能替代 CANN 编译、UT 执行或设备推理。

### 4.2 单元测试

官方用例包包含 57 个 UT 文件、624 个测试。该基线的 `ENABLE_EXPERIMENTAL` 选择实验目录，不会同时包含正常目录，因此分别构建执行两种配置。正常配置运行 669 项（官方 615、新增 21、仓库已有 33），实验配置运行 Max 的 9 项；678 项均通过，官方 624 项按完整测试名称逐一对账，缺失 0、失败 0。

Flatten、Squeeze、Unsqueeze、IsInf、RandomUniform 的旧式目录缺少 framework UT 的 CMake 入口，会漏跑 51 项官方用例。本次仅补充这 5 份必要的 `tests/ut/framework/CMakeLists.txt`，调用仓内 `add_modules_ut_sources`。官方 57 份 UT 的断言和 token 保持不变。

覆盖内容包括有效属性、字段缺失及默认值、非法 JSON、类型错误、边界整数/浮点数、错误返回码、动态输入输出数量和注册回调。官方用例首先保持语义不变。确需修复用例基础设施时，分别记录修改前后、修改原因及不改变验收强度的证据。对任务矩阵中尚未覆盖的 dtype、format、属性和可选输入组合增加针对性测试。

### 4.3 ONNX → OM 与设备推理对比

官方 56 个模型生成器覆盖 57 个插件文件（两处 Max 共享生成器）。逐一记录生成器、ONNX opset、算子属性、输入 shape/dtype/format、模型与输入 SHA-256，并复用同一份模型和输入运行 before/after 两轮。

1. 用基线提交构建旧插件，记录插件哈希；必要的 Protobuf 加载兼容措施仅作用于旧插件阶段，记录具体原因和命令。
2. 使用相同 ATC 参数转换全部模型，保留逐用例命令、返回码及日志。以插件加载日志和注册顺序等证据确认自定义插件真正生效，避免误测内置插件。
3. 用 pyACL 在同一设备运行可执行 OM，保存原始输出、shape/dtype 和执行日志。
4. 替换为迁移后插件，使用相同模型、输入、CANN、硬件和 ATC 参数重复转换及推理。
5. 默认按原始字节做 bitwise 对比，区分正负零及 NaN 位模式；额外保留数值差异统计用于定位。固定 seed 的四个随机算子也使用此标准。
6. 对基线转换失败或设备缺少 kernel 的场景，逐例保留前后错误证据并标明限制，不把未运行或失败用例计为通过。任何特殊比较标准必须在用例元数据和报告中明确列出。

官方生成用例之外的补充用例按任务书矩阵确定，优先覆盖 dtype、format、可选输入、缺省与显式零值，以及可能受 API 迁移影响的组合；不通过大批重复 shape 用例虚增覆盖率。

### 4.4 可观测性与交付物

迁移清单关联 57 个源码文件、124 个算子条目、UT、OM 生成器、前后插件哈希和验证结果。原始日志、模型、输入、输出和比较脚本以可复现目录组织；环境私钥、口令和临时连接信息不进入提交物。

按官方模板提供 `task_submission/` 下的复现步骤、精度结果表与日志，以及性能/内存相关文件。本任务未要求性能或内存优化，相关项目明确标为不适用并说明原因，不填造数值。设计评审意见闭环、真实测试完成后提交 ops-math 实现 PR，再按社区流程发起验收。

## 5. 兼容性与风险

| 风险 | 应对与验收依据 |
| --- | --- |
| JSON 字段类型、零值省略与默认值不同 | 保留原算子逻辑；官方 UT 和显式零值/缺失字段回归 |
| Float 字符串解析对次正规数、特殊值的处理 | float 精度转换、完整字符串检查；边界 UT 及真实模型验证 |
| 去掉公共头文件后缺少间接声明 | 显式 include；在实际 CANN SDK 编译所有目标 |
| 仅测试到内置插件 | 保留自定义插件加载、注册顺序、插件哈希证据 |
| 实验目录未编译造成漏测 | 单独核对 experimental Max 文件及 9 个 UT |
| Clip 可选输入占位丢失 | 额外 OM 用例验证 GE 映射，必要时基于公开接口修正 |
| 基线缺陷或非确定性误判为迁移变化 | 同环境重复基线、逐用例证据和明确的允许比较标准 |
| 基线与开发期间上游漂移 | 固定基线哈希；合并上游后重跑受影响验证并记录最终提交 |

## 6. 完整复验结果（2026-10-23）

161 项完整复验：158 项真实输出逐字节一致，1 项 Multinomial 按官方行为判据通过，2 项补充 Clip 拒绝行为单列。其中 119 个官方模型全部完成转换与推理。

测试环境为 **Ascend910B2、CANN 9.1.0 和下表列出的验证 OPP 配置**。OPP 配置包括原型注册、引擎登记、算子名称适配和形状推导，源码及重建脚本位于 `task_evidence/recovery/`。实现 PR 的修改范围是 Host 解析迁移及 framework UT。三个 Swin 模型的输入类型、形状和输出配置按内核接口修订，具体改动见下表。

使用官方 `two_round_compare.sh`，迁移前后采用相同模型、输入、环境和 ATC 参数 `must_keep_origin_dtype`。159 项正向用例中，158 项输出逐字节一致，Multinomial 按官方行为判据通过。另有 2 项补充 Clip 负向用例，两轮均按原解析逻辑拒绝缺少 min/max 的输入。四个固定 seed 的 Random 算子均逐字节一致。

验证工具的 CRLF 行尾转换为 Linux LF，八份文件的规范化 SHA256 与官方文件一致，记录见 `recovered-comparator-verification.json`。

六个修复模型另外完成 NumPy 数学检查，输出均为有限值；两轮主比较采用逐字节判据。Sequence 的常量 L_empty 经 ATC 折叠，按官方 meta 核对 Y、L 两个实际输出，结果为 Y 与输入 A 完全一致、L=4。
| 项目 | 原因与实际修复 | 本次有效输出 |
| --- | --- | --- |
| RepeatWithPad | 为 SDK 已有 Python 内核补齐 A2 引擎登记；内核源码不变 | FP16 [512,512]，两轮逐字节一致 |
| ReduceLogSumExp / ReduceSumSquare | 注册基线仓库原型，匹配运行时缺失的 noop_with_empty_axes=false 可选属性；不向旧内核添加参数 | 各 FP32 [4,1,512]，两轮逐字节一致 |
| Addcmul | 验证依赖把 GE 要求的限定名称绑定到当前轮实际加载插件的原始解析及子图回调；不翻译属性，不改插件注册字符串 | FP32 [4,256,512]，两轮逐字节一致 |
| SwinTransformerLnQKV | 连接官方固定形状推导规则，默认 head_num=4/head_dim=32/seq_length=64/epsilon=1e-5 与固定内核一致；模型按 [8,256,128] FP16 构造 | Q/K/V 三路 FP16 [32,4,64,32]，两轮逐字节一致 |
| SwinAttentionScore | 选择 SDK 未修改的静态 TIK 实现；Q/K/V 为 FP16 [1,2,1024,1024]，scale=0.03125。按实现契约省略可选 softmax_output，修正合并头后的输出形状 | FP16 [1,1024,2048]，两轮逐字节一致 |
| SwinAttentionFFN | 修正为 FP16，补齐残差输入，x1/x3 [1,64,128]、x2 [128,128]、bias [128] | FP16 [1,64,128]，完整批次重跑逐字节一致 |

实现 c21e107a 仅恢复已有源码行尾，实测插件二进制保持 af57dd2 版本。完整模型、输入、输出与插件哈希见个人 Fork 证据分支 task_evidence/recovery/。678 项 UT、普通及实验插件构建、pre-commit/OAT 已通过；SCA 的两项既有片段匹配仍待平台审核。

## 附录：迁移范围

以下为官方任务清单按源码路径聚合的 57 个文件；重复算子名称保留在各自文件条目中。

| 源文件（相对于 ops-math） | 算子 |
| --- | --- |
| `common/src/framework/concat_from_sequence_onnx_plugin.cpp` | ConcatFromSequence |
| `common/src/framework/deformable_offsets_onnx_plugin.cpp` | DeformableOffsets |
| `common/src/framework/einsum_onnx_plugin.cpp` | Einsum |
| `common/src/framework/eyelike_onnx_plugin.cpp` | EyeLike |
| `common/src/framework/phony_concat_onnx_plugin.cpp` | PhonyConcat |
| `common/src/framework/phony_split_onnx_plugin.cpp` | PhonySplit |
| `common/src/framework/random_normal_like_onnx_plugin.cpp` | RandomNormalLike |
| `common/src/framework/random_uniform_like_onnx_plugin.cpp` | RandomUniformLike |
| `common/src/framework/reduceL1_onnx_plugin.cpp` | ReduceL1 |
| `common/src/framework/reduceL2_onnx_plugin.cpp` | ReduceL2 |
| `common/src/framework/repeat_with_pad_onnx_plugin.cpp` | RepeatWithPad |
| `common/src/framework/sequence_onnx_plugin.cpp` | SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength |
| `common/src/framework/similar_opp_onnx_plugin.cpp` | Abs, Acos, Acosh, Add, And, Asin, Asinh, Atan, Atanh, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BlendFaceBgPartTwo, CastLike, Ceil, Cos, Cosh, Det, Div, Dropout, Equal, Erf, Exp, Expand, Floor, GlobalAveragePool, Greater, GreaterOrEqual, HardSwish, Identity, IsNaN, Less, LessOrEqual, Log, Mul, Neg, Not, Or, PRelu, Pow, Range, Reciprocal, Relu, ResizeBicubic, Round, ScatterND, Sigmoid, Sign, Sin, Sinh, Softplus, Softsign, Sqrt, Sub, SwinAttentionFFN, SwinAttentionScore, SwinTransformerLnQKV, Tan, Tanh, Tile, Triu, Where |
| `common/src/framework/split_to_sequence_onnx_plugin.cpp` | SplitToSequence |
| `conversion/clip_by_value/framework/clip_onnx_plugin.cpp` | Clip |
| `conversion/concat/framework/concat_onnx_plugin.cpp` | Concat |
| `conversion/depth_to_space/framework/depth_to_space_onnx_plugin.cpp` | DepthToSpace |
| `conversion/flatten/framework/flatten_onnx_plugin.cpp` | Flatten |
| `conversion/pad_v3/framework/pad_onnx_plugin.cpp` | Pad |
| `conversion/reshape/framework/reshape_onnx_plugin.cpp` | Reshape |
| `conversion/slice/framework/npu_slice_onnx_plugin.cpp` | NPUSlice |
| `conversion/space_to_depth/framework/space_to_depth_onnx_plugin.cpp` | SpaceToDepth |
| `conversion/split_d/framework/split_onnx_plugin.cpp` | Split |
| `conversion/squeeze/framework/squeeze_onnx_plugin.cpp` | Squeeze |
| `conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp` | Slice |
| `conversion/transpose/framework/transpose_onnx_plugin.cpp` | Transpose |
| `conversion/trilu/framework/trilu_onnx_plugin.cpp` | Trilu |
| `conversion/unsqueeze/framework/unsequeeze_onnx_plugin.cpp` | Unsqueeze |
| `experimental/math/asinh/framework/max_onnx_plugin.cpp` | Max |
| `math/accumulate_nv2/framework/mean_onnx_plugin.cpp` | Mean |
| `math/accumulate_nv2/framework/sum_onnx_plugin.cpp` | Sum |
| `math/addcmul/framework/addcmul_onnx_plugin.cpp` | Addcmul |
| `math/arg_max_v2/framework/argmax_onnx_plugin.cpp` | ArgMax |
| `math/arg_min/framework/argmin_onnx_plugin.cpp` | ArgMin |
| `math/bitwise_xor/framework/xor_onnx_plugin.cpp` | Xor |
| `math/cast/framework/cast_onnx_plugin.cpp` | Cast |
| `math/cumsum/framework/cumsum_onnx_plugin.cpp` | CumSum |
| `math/is_inf/framework/isinf_onnx_plugin.cpp` | IsInf |
| `math/maximum/framework/max_onnx_plugin.cpp` | Max |
| `math/minimum/framework/min_onnx_plugin.cpp` | Min |
| `math/mod/framework/mod_onnx_plugin.cpp` | Mod |
| `math/one_hot/framework/onehot_onnx_plugin.cpp` | OneHot |
| `math/reduce_log_sum_exp/framework/reduce_log_sum_exp_onnx_plugin.cpp` | ReduceLogSumExp |
| `math/reduce_max/framework/reduce_max_onnx_plugin.cpp` | ReduceMax |
| `math/reduce_mean/framework/reduce_mean_onnx_plugin.cpp` | ReduceMean |
| `math/reduce_min/framework/reduce_min_onnx_plugin.cpp` | ReduceMin |
| `math/reduce_prod/framework/reduce_prod_onnx_plugin.cpp` | ReduceProd |
| `math/reduce_sum/framework/reduce_log_sum_onnx_plugin.cpp` | ReduceLogSum |
| `math/reduce_sum/framework/reduce_sum_onnx_plugin.cpp` | ReduceSum |
| `math/right_shift/framework/bitshift_onnx_plugin.cpp` | BitShift |
| `math/sign_bits_pack/framework/npu_sign_bits_pack_onnx_plugin.cpp` | NPUSignBitsPack |
| `math/sign_bits_unpack/framework/npu_sign_bits_unpack_onnx_plugin.cpp` | NPUSignBitsUnpack |
| `math/square_sum_v1/framework/reduce_sum_square_onnx_plugin.cpp` | ReduceSumSquare |
| `math/topk/framework/topk_onnx_plugin.cpp` | TopK |
| `random/dsa_random_uniform/framework/random_uniform_onnx_plugin.cpp` | RandomUniform |
| `random/stateless_random_normal_v2/framework/random_normal_onnx_plugin.cpp` | RandomNormal |
| `random/stateless_sample_multinomial/framework/multinomial_onnx_plugin.cpp` | Multinomial |
