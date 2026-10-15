# 10月社区任务-ops-math算子ONNX适配设计文档

提交帐号：`q137145`。方案：A2 上验证 Host 侧 ONNX 插件 protobuf 解耦。版本日期：2026-10-14。

本文是设计评审稿，描述拟实施方案与验证计划；不声明迁移代码、UT、A2 两轮测试、官方评审或验收已经通过。正文遵循社区设计模板的需求背景、需求分析、详细设计、可维可测及兼容性结构。

# 需求背景（required）

## 需求来源

- [官方任务页](https://www.hiascend.com/activities/task-center/details/18e1d3beae784812b2562460542ad6b3)：10月社区任务-ops-math算子onnx适配，发布日 2026-10-14，页面截止日 2026-11-15；具体截止时刻以平台为准。
- [官方任务包](https://www.hiascend.com/p/resource/202610/a1883f98ce744816a85a78852446666e.zip)：内含 `ops-math_onnx_task_doc.md` 与 `test_cases/`。ZIP SHA256 为 `06470bafb1b2c807fef0ce1f566fc0702973b1161e5edf28c7b7ea31aaaba5ec`，任务书 SHA256 为 `21e620c82e7969e75272ec82e0b92fb1ce71776fd7fc808169917a2b5a3e1877`。
- [ops-math 源码仓](https://gitcode.com/cann/ops-math)、[适配 Issue #2892](https://gitcode.com/cann/ops-math/issues/2892)及其关联迁移 PR。
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)、[社区任务流程](https://www.hiascend.com/developer/activities/cann-community-task)、[流程与注意事项](https://gitcode.com/org/cann/discussions/39)。

设计分析固定在任务书引用的源码快照 `91c6c1715a4c07502479ac03e11c52d7210e60ad`。它是可复核的分析起点；是否同时作为正式验收基线，请评审确认。后续主干范围变化按任务书提交对应 Issue，并更新范围映射，不默默删减迁移项。

## 背景介绍

现有插件通过 `ParseParamsFn` 注册依赖 `google::protobuf::Message` 的解析函数，读取 `ge::onnx::NodeProto` 及 AttributeProto。这使参数解析与 protobuf 生成头文件、运行库及测试对象构造耦合。

新框架支持通过 `ParseParamsByOperatorFn` 接收 `ge::Operator`，用 `GetAttr("attribute")` 取得 JSON 字符串。迁移将替换参数解析载体，保持内部算子映射、属性含义、动态输入输出、拆图及计算逻辑。

### 当前实现与参考路径

旧接口示例为 `math/cast/framework/cast_onnx_plugin.cpp`。仓内 `common/src/framework/reverse_sequence_onnx_plugin.cpp` 已使用新签名、JSON 解析及异常处理，可参考其接口和组织方式；其具体属性语义不能直接套到其他算子。`random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp` 可用于核对浮点字符串读取和 Operator 回调形态，但仍有旧头文件引用，不能作为彻底解耦的完成证明。

公共工具位于 `common/inc/framework/math_onnx_plugin_util.h`，包含 `GetOpName`、`GetOmDtypeFromOnnxDtype`、`ChangeFormatFromOnnx`、`Vec2Tensor` 与 `CreateScalar`。优先复用，不为本次迁移另造一套公共解析框架。

# 需求分析（required）

## 需求描述

任务书范围是 **124 个文件×算子条目、57 个插件文件**。同一文件中的多算子分别覆盖，同一算子的多个 ONNX 版本在任务书表内合为一项；条目数、注册版本数、UT 数及 OM 数分别统计。

迁移仅涉及范围内 `framework/*_onnx_plugin.cpp`、对应 framework UT，以及确有必要的测试接入。kernel、tiling、算子原型与 `ParseOpToGraphFn` 保持原样；不新增 aclnn 或 torch 接口。设计 PR 提交到竞赛设计仓；后续实现 PR 按任务书提交到 ops-math，不混淆两者。

## 需求拆解与验证映射

| 编号 | 任务书要求 | 实现方案 | 验证与交付 |
| --- | --- | --- | --- |
| R01 | §1、§2.1—2.3：全范围新接口与解耦 | 保留函数名，替换签名和注册宏，去除范围内直接与传递 protobuf 引用 | 附录全部 57 文件自查，注册/opset 清单，编译日志及依赖检查 |
| R02 | §2.1、§2.4：属性及名称 | Operator 取 JSON，按每个属性保留默认值、字段缺省、类型、顺序和返回分支；名称用 `GetOpName(op_src)` | 正常、缺失、非法 JSON、边界 UT 与 op_dest 属性断言；无属性分支见评审问题 Q4 |
| R03 | §2.1 第7条：失效判据 | Cast 用 `ge::DT_UNDEFINED` 判失败；同类发现逐项举证 | Cast 两个指定负向 UT，转换失败记录，自查表列明合法输入不变及非法输入失败阶段变化 |
| R04 | §2.2、§7：非迁移逻辑 | 保留 OriginOpType/opset、ImplyType、动态注册和拆图函数 | 基线差异审查；子图与动态输入相关 UT；两轮 OM |
| R05 | §3.1：运行环境 | 一套 A2、CANN >9.0.0、仓内 JSON 版本 | 环境清单、实际 SOC/驱动/工具链版本；A2 真机原始记录 |
| R06 | §3.2：两轮一致性 | 旧/新 DSO 隔离，同模型、同输入、同软硬件；输出逐数值相等 | 模型/输入/产物哈希，输出标识、dtype、shape、数值及转换结果；随机项见 Q1 |
| R07 | §3.2：golden 辅助判据 | 按任务专用判据与独立 ORT/NumPy 参考比对；无 golden 保留行为比对 | 精度报告、逐输出原始结果；引用标准差异见 Q2 |
| R08 | §3.5：自验与覆盖 | 接入全部 57 份 UT 和 56 个模型脚本，并补齐未覆盖条目/版本 | UT 列表与 XML/日志/截图，124 条目覆盖矩阵；共享 OM 见 Q3 |
| R09 | §3.5：构建与自查 | 插件构建、OP_HOST_UT framework 目标、diff 检查 | 干净构建日志，实际执行数，自查表 |
| R10 | §3.3、§3.4 | 性能、内存均不涉及 | 不设新增性能/内存验收指标；保留不适用说明，材料形式见 Q7 |
| R11 | §4：材料 | `task_submission` 存步骤、精度报告/日志、自查表和可追溯测试记录 | 最终包逐项核对，不用设计稿代替自验结果 |
| R12 | §5、§7：仓规范与流程 | 设计评审后实施；自验与平台流程完成后向 ops-math 提交实现 PR | 贡献规范、官方 CI、评审/合入及平台回执分别登记 |

# 详细设计（required）

## 算子分析

### 数学公式、数据类型与形状

本题不新增数学公式，不改变任何算子的计算规则、输入输出 dtype、形状、布局或广播能力。迁移的可观察契约是同一节点映射到相同内部算子及属性，进而保持转换和推理行为。各算子的 Tensor 支持范围沿用其源码与任务用例，不声称本题所有算子统一支持某种 dtype 或广播。

### 接口定义及框架调用路径

```cpp
domi::Status ParseParamsOpName(const ge::Operator& op_src,
                              ge::Operator& op_dest);
```

`OpName` 表示现有函数对应的算子名，不要求重命名已有函数。注册保持现有 `REGISTER_CUSTOM_OP`、`FrameworkType(ONNX)`、`OriginOpType`、`ImplyType` 及已有拆图回调，仅将 `.ParseParamsFn(...)` 改为 `.ParseParamsByOperatorFn(...)`。

调用路径：ONNX 模型 → GE/ATC 读取节点并提供 Operator 属性 → 新接口参数回调 → 写入 op_dest → 原有拆图/内部算子及后续编译 → OM → A2 推理。protobuf 到 JSON 的序列化由框架完成；插件不再接触 NodeProto。框架自身仍使用 protobuf 不等于迁移插件允许继续依赖 protobuf。

## 算子实现

### Host 侧设计

按文件建立“注册项/opset—解析函数—属性—目标属性—默认值—异常—动态注册—拆图”账本，核对任务书表中的每一项。迁移按语义族实施，但每个文件均保持独立核对，不按文件名或相似算子批量替换属性默认值。

1. include 使用 `graph/operator.h`、`nlohmann/json.hpp`、`math_onnx_plugin_util.h`、`register/register.h` 及实际需要的日志/算子声明头。清除范围内 `onnx_common.h`、`ge_onnx.pb.h`、Message/NodeProto/AttributeProto 依赖，检查间接包含链。
2. 保持原默认属性初始化和动态输入输出注册的顺序。通过 `ge::AscendString` 接收 `op_src.GetAttr("attribute", attrs_string)`。属性取不到时走各插件原缺失分支；必需属性如 Cast `to` 缺失返回 FAILED，不能统一返回 SUCCESS。
3. 返回字符串示例为 `{"attribute":[{"name":"axis","type":2,"i":2}]}`，需要解析外层对象的 `attribute` 数组，不能把整个字符串误当成数组。空数组按属性缺失；语法错误捕获并返回 FAILED。有属性解析插件对结构/字段不合法导致的访问、转换异常记录日志并返回 FAILED；合法但缺少顶层属性的情形按原默认路径核对。
4. 匹配 `name` 后检查 ONNX `type`，按任务书处理类型不匹配为缺失。未知属性忽略。保留原扫描顺序、重复属性的首个/最后一个生效规则、列表顺序、负值/空列表的透传或校验；不自行排序、去重、裁剪或归一化。
5. `i` 为整数，`s` 为字符串，`ints`/`floats` 为数值数组。属性存在且对应 proto3 标量字段省略时使用其零值，不能回落到属性整体缺失的默认值。数值转换和写入沿用原目标类型，避免把 int64 属性统一缩窄为 int32；边界用例校验实际旧行为。
6. 标量 `f` 以字符串承载，例如 `"f":"0.25"`；按仓内转换行为读取并转为 float。属性存在但无 `f` 键时取 **0.0**；整个属性不存在才使用算子默认值。不能对所有浮点值统一使用 JSON number，也不能私自改变特殊字符串转换语义；字符串类型错误、JSON 访问异常走 FAILED。非法浮点字符串的判定与仓内 `atof` 行为若冲突，列入评审确认后定稿。
7. 对要求写入名称的迁移回调使用 `op_dest.SetAttr("name", GetOpName(op_src))`，并用不同源/目标名称测试来源是否正确。无属性、原无名称写入的回调与通用名称要求冲突，见 Q4，不直接扩大无属性回调的可观察行为。
8. 用局部 JSON 对象和容器，避免全局可变解析状态。捕获 `nlohmann::json::exception` 及未知异常，记录算子名和错误原因后返回 FAILED；不让 C++ 异常跨框架回调边界，也不为失败伪造 SUCCESS。只核验本次引入的资源生命周期，不新增无依据的峰值内存门槛。

### 关键语义族

| 语义族/例子 | 需要保持的行为 | 必要反例 |
| --- | --- | --- |
| Cast | `to` 映射为 `dst_type`；缺失失败；任务规定修正 DT_UNDEFINED 判据 | `to=10` 正常；`to=20` 不支持；有 `to` 无 `i` 等价零值；不匹配 type 按缺失 |
| Flatten、Squeeze、Unsqueeze、Reduce | axis/axes、keepdims、列表与默认值；区分不同 opset 的属性和 Tensor 输入 | axis=2、负轴、空轴、缺失、类型不匹配；不替换后续形状推导 |
| EyeLike、Random*、Multinomial | dtype 映射、浮点 seed/分布参数、Tensor 标量构造及已有拆图 | 属性缺失对比存在但无 f；非默认参数；随机状态固定；seed 转换沿用旧实现 |
| Concat、Split、Sequence*、Max/Min/Mean/Sum | 动态输入输出数量、original_type、注册版本和拆图参数 | 多输入、多输出；Sequence 文件六个条目分别测试；两份 Max 分别证明插件生效 |
| DepthToSpace、SpaceToDepth、Transpose 等 | 格式变更、perm、blocksize 和已有校验 | ND/NCHW 等现有分支；列表顺序；非法值及原失败返回 |
| similar_opp | 全部 63 个任务条目及各注册版本；保持内部算子映射 | 每项注册/回调 UT；原不读属性的 SUCCESS 语义与通用非法 JSON 规则冲突，见 Q4 |

### Cast 失效判据修正

公共 dtype 工具对未知值返回 `ge::DT_UNDEFINED`，旧 Cast 使用 `om == -1` 无法识别该返回值。迁移使用 `om == static_cast<int32_t>(ge::DT_UNDEFINED)`；命中记录日志并返回 FAILED。任务书指定 `ReturnFailedWhenDtypeUnsupported` 和 `ReturnFailedWhenToIsZeroWithoutIField` 必须覆盖。

这会将某些非法模型的失败前移到解析阶段。它是任务书明确要求的修正，不能强求该项错误阶段和文本与旧版本完全相同；保留两轮原始失败及修正依据。其他同类问题仅在证明工具真实返回约定与旧判据不符后按任务书处理，并逐项记录，不能扩展为任意 bug 修复。

### Kernel、tiling 与拆图设计

kernel/tiling 不涉及。全部现有 `ParseOpToGraphFn` 的函数体、节点构建和连边保持原样；动态注册、属性设置仍可处于参数回调中，属于需要保持的行为。若公共工具复用导致无关文件必须调整，先评估影响并说明必要性，不能以解耦名义顺手重构拆图。

### 构建与单测接入

复用 ops-math 现有 framework UT 目标与 CMake 接入：集中插件对应 `common/tests/ut/framework/`；独立算子对应 `<算子目录>/tests/ut/framework/`。测试直接构造 `ge::Operator` 和 JSON 属性，检查注册所得新接口回调及 op_dest，不构造 protobuf 对象。

插件按任务书运行 `bash build.sh --onnxplugin -j8`。仓内 `OP_HOST_UT` 和 `OP_FRAMEWORK_MODULE_NAME` 已有定义，framework 用例按 `test_*_onnx_plugin.cpp` 收集；需验证最终构建目标确实包含全部用例。`-u --ophost` 设置 OP_HOST_UT，但默认构建目标不应被假定已执行 framework UT：在目标环境核实 framework 可执行文件/目标、其构建与运行命令，保存 `--gtest_list_tests`、实际运行数和 XML。确需 CMake 接入修复时仅调整必要的 framework UT 接线，并单独审查。未找到或零执行不能报通过。

### 实施顺序

先以 Cast、浮点字符串/缺省 f、动态输入与拆图插件、similar_opp 无属性分支验证接口和测试接入；随后按语义族覆盖完整 57 文件。开发期运行受影响 UT，最终冻结一个主候选并完整构建、UT、自查及 A2 两轮验证。基线、候选、构建目录和原始记录隔离；不跨候选拼接 PASS。

## 支持硬件

| 项目 | 设计选择及依据 |
| --- | --- |
| 验证平台 | A2；任务书 §3.1 允许 A2 或 Ascend 950 任一通过，本方案不要求再测 950 |
| 优先复现环境 | 任务书实测 Ascend910B3 + CANN 9.1.0；实际环境尚待采集 |
| CANN | 必须 >9.0.0；驱动/固件兼容性按实际官方安装要求核验 |
| JSON | 目标仓 third_party 自带版本，不引入新版本 |
| Python/模型工具 | Python 3.8+；版本兼容性以任务包和实装核验为准，opset 9/11/12/13/14、IR=8 按各模型脚本 |
| 推理 | 优先 CANN 自带 pyACL；taskbook 也允许 ais_bench_infer 或 msame |

硬件结论仅从实际 A2 记录得出；Host 编译、CPU UT 和源码分析不替代 A2 转换/推理证据。此设计不申请资源、启动租卡或修改运行环境。

## 算子约束限制

迁移范围按任务书表固定，保留各算子现有 opset 与限制。未支持的平台/模型按任务书平台容错规则记录，不能一律跳过；已迁移参考文件不自动纳入本题。附录共享模型只是现有资产映射，不构成代表用例覆盖全部条目的批准。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判据及验证方法 | 来源 |
| --- | --- | --- |
| 两轮主判据 | 同模型同输入，按输出标识对齐，再检查 dtype、shape、逐数值相等；不以 golden 容差替代前后完全一致 | 任务书 §3.2；特殊值与随机项见 Q1 |
| golden 辅助判据 | FP32 rtol=2^-10、atol=2^-16，逐元素硬上限 max(fixed_limit,32*ULP)，整体匹配率≥0.99；整数/布尔精确，其他 dtype 按任务包标准文件；独立检查 dtype/shape | 任务书 §3.2 和附件标准；与引用标准的差异见 Q2 |
| 无 golden | 比较两轮转换/推理结果及可观察行为，不伪造 golden | 任务书 §3.2 |
| 平台不支持 | 保存两轮 ATC 原文和 SOC；普通不支持项要求失败根因一致；Cast 修正项单独记录 | 任务书 §3.2、§2.1；错误比对口径见 Q5 |
| framework UT | 覆盖全范围，`--gtest_filter='Onnx*PluginTest.*'`，核对列出的测试与实际运行；四类属性场景及注册/opset | 任务书 §3.5；612/624 差异见 Q6 |
| 代码自查 | include、签名、注册宏、属性、name、返回分支、UT、干净构建及 `git diff --check` | 任务书 §3.5 |
| 性能、内存 | 不涉及；不设置性能提升目标、Profiler 或新增内存验收门 | 任务书 §3.3、§3.4 |

### 两轮验证与证据设计

模型与输入只生成一次并冻结 SHA256，golden 使用对应脚本声明的 ORT/NumPy 独立来源。两轮使用同一物理 A2、驱动、CANN、SOC 配置、命令与输入；只替换插件版本。用独立目录保存旧/新 DSO、OM、输出和日志，通过 `ASCEND_CUSTOM_OPP_PATH` 注入本轮插件，记录 ATC 命令、加载路径和 DSO 身份，排除加载系统默认插件或混入另一轮的情形。

旧插件出现任务书所述 `undefined symbol` 时，仅在旧轮按 `verify_om/compat/README.md` 使用兼容库；记录来源及哈希。新轮不挂该兼容库，检查其动态依赖和符号。只读核验本次迁移插件不再引用 protobuf；框架/无关插件仍依赖 protobuf 时不能把整库“零 protobuf”强加为本题门槛。

先核对附件比较器是否执行正式判据，补充对输出身份、预期 dtype/shape、失败根因和插件生效的检查。保留官方原脚本与差异，工具修订先用已知反例证明能检出缺失属性、dtype/shape 错误、交换输出和错误插件；不自行改容差，不把返回码为零当成完整覆盖。官方若不接受工具修订，改用指定工具加独立核验并保留问题记录。

每个 case 保存注册项/opset、模型和输入哈希、golden 来源、基线及候选提交、构建参数、DSO/兼容库哈希、设备与软件、命令、输出标识/dtype/shape、比较结论、原始报错及例外依据。统计分别列 PASS、FAIL、未运行、平台例外和证据缺失；不把静态测试声明数称作执行通过数。

### 覆盖策略

附件为 57 份 UT 文件、56 个模型脚本。静态测试声明数核对为 624，任务书正文称 612，实际覆盖以冻结清单和运行日志为准，待官方确认版本。`similar_opp` 的 63 条注册项需要全部 UT；附件只生成 Add/Neg/Where/HardSwish 四个代表 OM，未获代表覆盖批准前按全条目要求补模型或记录未验证。两份 Max 共用脚本时分别构建/隔离注册并留下生效证据，不能一次输出同时替两份插件背书。

### 待评审确认的契约差异

以下为已识别的问题，尚无官方回复。它们用于设计评审；在冲突关闭前不声称对应实现语义或最终验收判据已经确定。

| 问题 | 差异 | 设计处理及所需确认 |
| --- | --- | --- |
| Q1 随机与特殊值 | 主判据完全一致；附件对非确定性项使用容差；字节哈希与逐数值相等对 NaN/±0 未必同义 | 优先固定 seed/随机状态；不可复现项需逐算子官方判据。确认 NaN/Inf/±0 处理及数值/字节口径；不以容差替代主判据或独立样本分布检验 |
| Q2 golden 标准 | 本题阈值与引用 opbase 标准快照有差异 | 记录任务专用公式和标准版本，请官方固定各 dtype、fixed_limit/ULP、特殊值及匹配率实现，不擅自选较松标准 |
| Q3 代表 OM | 正文要求全范围，similar_opp 附件只有四个 OM | 设计默认全条目覆盖；请求明确代表覆盖是否获准及两份 Max 的证明方式 |
| Q4 无属性及 f 字符串 | 无属性 UT 对非法 JSON 要求 SUCCESS，与通用 FAILED/name 写入规则冲突；atof 不严格拒绝所有非法数字字符串 | 无属性回调拟保持不读不写、SUCCESS，并单列请求确认；若评审要求统一解析/名称则同步修订语义与 UT。f 的非法字符串规则须与任务/参考行为统一 |
| Q5 失败等价与工具 | Cast 要求失败前移；普通失败脚本只查标记/可选关键词，插件生效缺证据仅 warning | Cast 按明确例外举证，其他项比较错误码/阶段/根因；请求官方确认去除时间戳等非语义差异的口径和工具修订接受方式 |
| Q6 基线与数量 | 91c6c17 与“发布时仓代码”、612 与624、framework UT 命令待实证 | 固定分析快照、资产哈希和实际列表；请求正式基线及 UT 清单/入口；不能按预计总数倒填 PASS |
| Q7 不适用材料 | 性能/内存不涉及，但交付树列四个文件 | 计划保留四个 N/A 说明文件，不生成假测量；请确认保留或省略方式 |

### 交付、评审和验收

设计文档放入设计仓 `04_tasks/01_community-task-2026/tasklist/10月社区任务-ops-math算子onnx适配/q137145/docs/design.md`。采用完整任务名建立目录，不虚构未公布的数字编号。设计 PR 标题为 `【CANN社区任务】ops-math算子ONNX适配算子设计文档`，目标为 `cann/cann-ops-competitions:master`。

当前设计推送仅包含本设计文件；审计报告和准备记录留在本题本地证据目录。按官方流程，设计链接、代码链接“暂无”、审核人通知及任务进展应绑定同一帐号；平台报名/审核、设计评审通过、设计合入、实现自验、平台验收、实现 PR/CI/合入分别记录。官方设计评审或合入由维护者决定。最终验收前保留可识别设计 PR merged 状态的真实截图。

后续私仓新建 `task_submission`，按任务书提供 `1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`；性能和内存四个文件按 Q7 处理。附上 UT 日志/截图、OM 模型/命令/两轮比对原始记录和逐文件自查表。最终包绑定冻结实现候选及哈希，审阅实际图片和文件；设计文档不替代任一自验结果。

## 兼容性分析

新接口兼容性需在 CANN >9.0.0 的实际目标环境验证；不凭接口声明推断所有发行版都兼容。原注册 opset、内部算子、布局、动态输入输出和拆图保持原行为，除明确 Cast 失效判据修正外不改变计算或非法参数处理。UT 中 Operator 字符串表示须与 GE 实际序列化保持一致；纯手工 JSON 测试还需真实 ATC 覆盖。

JSON 使用仓内依赖，避免额外 ABI/版本风险。旧轮的兼容库不带入新候选。回退时切回冻结旧插件及其独立目录，不覆盖新旧产物或历史失败日志。源码更新、依赖/DSO/环境变化时重新评估受影响验证，未经绑定的旧 PASS 不沿用。

## 参考资料

- [ops-math 当前贡献指南](https://gitcode.com/cann/ops-math/blob/master/CONTRIBUTING.md)及[构建说明](https://gitcode.com/cann/ops-math/blob/master/docs/zh/install/build.md)。
- [新接口 API 文档](https://gitcode.com/cann/metadef/blob/master/docs/zh/api/ge_namespace/opregistrationdata/ParseParamsByOperatorFn.md)：回调形态为 `Status(const ge::Operator&, ge::Operator&)`。
- [固定分析快照的 ReverseSequence 参考实现](https://gitcode.com/cann/ops-math/blob/91c6c1715a4c07502479ac03e11c52d7210e60ad/common/src/framework/reverse_sequence_onnx_plugin.cpp)。
- [固定分析快照的公共工具](https://gitcode.com/cann/ops-math/blob/91c6c1715a4c07502479ac03e11c52d7210e60ad/common/inc/framework/math_onnx_plugin_util.h)。
- [引用的混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)与任务包 `test_cases/verify_om/precision_standard.py`；差异按 Q2 固定。

## 附录：完整文件、算子及测试资产映射

下表按任务书原表顺序归并为 57 行，算子合计 124 条。源码路径相对于 ops-math；UT 和 OM 列是原始任务包 `test_cases/ops-math/ut/`、`test_cases/ops-math/om/` 中的文件名。`similar_opp` 和 Max 的覆盖限制见 Q3；本表表示计划与已有资产对应，不表示执行通过。

| 文件 | 算子条目 | UT 文件 | OM 脚本 |
| --- | --- | --- | --- |
| `common/src/framework/concat_from_sequence_onnx_plugin.cpp` | ConcatFromSequence | `test_concat_from_sequence_onnx_plugin.cpp` | `gen_model_concat_from_sequence.py` |
| `common/src/framework/deformable_offsets_onnx_plugin.cpp` | DeformableOffsets | `test_deformable_offsets_onnx_plugin.cpp` | `gen_model_deformable_offsets.py` |
| `common/src/framework/einsum_onnx_plugin.cpp` | Einsum | `test_einsum_onnx_plugin.cpp` | `gen_model_einsum.py` |
| `common/src/framework/eyelike_onnx_plugin.cpp` | EyeLike | `test_eyelike_onnx_plugin.cpp` | `gen_model_eyelike.py` |
| `common/src/framework/phony_concat_onnx_plugin.cpp` | PhonyConcat | `test_phony_concat_onnx_plugin.cpp` | `gen_model_phony_concat.py` |
| `common/src/framework/phony_split_onnx_plugin.cpp` | PhonySplit | `test_phony_split_onnx_plugin.cpp` | `gen_model_phony_split.py` |
| `common/src/framework/random_normal_like_onnx_plugin.cpp` | RandomNormalLike | `test_random_normal_like_onnx_plugin.cpp` | `gen_model_random_normal_like.py` |
| `common/src/framework/random_uniform_like_onnx_plugin.cpp` | RandomUniformLike | `test_random_uniform_like_onnx_plugin.cpp` | `gen_model_random_uniform_like.py` |
| `common/src/framework/reduceL1_onnx_plugin.cpp` | ReduceL1 | `test_reduceL1_onnx_plugin.cpp` | `gen_model_reduceL1.py` |
| `common/src/framework/reduceL2_onnx_plugin.cpp` | ReduceL2 | `test_reduceL2_onnx_plugin.cpp` | `gen_model_reduceL2.py` |
| `common/src/framework/repeat_with_pad_onnx_plugin.cpp` | RepeatWithPad | `test_repeat_with_pad_onnx_plugin.cpp` | `gen_model_repeat_with_pad.py` |
| `common/src/framework/sequence_onnx_plugin.cpp` | SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength | `test_sequence_onnx_plugin.cpp` | `gen_model_sequence.py` |
| `common/src/framework/similar_opp_onnx_plugin.cpp` | Abs, Acos, Acosh, Add, And, Asin, Asinh, Atan, Atanh, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BlendFaceBgPartTwo, CastLike, Ceil, Cos, Cosh, Det, Div, Dropout, Equal, Erf, Exp, Expand, Floor, GlobalAveragePool, Greater, GreaterOrEqual, HardSwish, Identity, IsNaN, Less, LessOrEqual, Log, Mul, Neg, Not, Or, PRelu, Pow, Range, Reciprocal, Relu, ResizeBicubic, Round, ScatterND, Sigmoid, Sign, Sin, Sinh, Softplus, Softsign, Sqrt, Sub, SwinAttentionFFN, SwinAttentionScore, SwinTransformerLnQKV, Tan, Tanh, Tile, Triu, Where | `test_similar_opp_onnx_plugin.cpp` | `gen_model_similar_opp.py` |
| `common/src/framework/split_to_sequence_onnx_plugin.cpp` | SplitToSequence | `test_split_to_sequence_onnx_plugin.cpp` | `gen_model_split_to_sequence.py` |
| `conversion/clip_by_value/framework/clip_onnx_plugin.cpp` | Clip | `test_clip_onnx_plugin.cpp` | `gen_model_clip.py` |
| `conversion/concat/framework/concat_onnx_plugin.cpp` | Concat | `test_concat_onnx_plugin.cpp` | `gen_model_concat.py` |
| `conversion/depth_to_space/framework/depth_to_space_onnx_plugin.cpp` | DepthToSpace | `test_depth_to_space_onnx_plugin.cpp` | `gen_model_depth_to_space.py` |
| `conversion/flatten/framework/flatten_onnx_plugin.cpp` | Flatten | `test_flatten_onnx_plugin.cpp` | `gen_model_flatten.py` |
| `conversion/pad_v3/framework/pad_onnx_plugin.cpp` | Pad | `test_pad_onnx_plugin.cpp` | `gen_model_pad.py` |
| `conversion/reshape/framework/reshape_onnx_plugin.cpp` | Reshape | `test_reshape_onnx_plugin.cpp` | `gen_model_reshape.py` |
| `conversion/slice/framework/npu_slice_onnx_plugin.cpp` | NPUSlice | `test_npu_slice_onnx_plugin.cpp` | `gen_model_npu_slice.py` |
| `conversion/space_to_depth/framework/space_to_depth_onnx_plugin.cpp` | SpaceToDepth | `test_space_to_depth_onnx_plugin.cpp` | `gen_model_space_to_depth.py` |
| `conversion/split_d/framework/split_onnx_plugin.cpp` | Split | `test_split_onnx_plugin.cpp` | `gen_model_split.py` |
| `conversion/squeeze/framework/squeeze_onnx_plugin.cpp` | Squeeze | `test_squeeze_onnx_plugin.cpp` | `gen_model_squeeze.py` |
| `conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp` | Slice | `test_slice_onnx_plugin.cpp` | `gen_model_slice.py` |
| `conversion/transpose/framework/transpose_onnx_plugin.cpp` | Transpose | `test_transpose_onnx_plugin.cpp` | `gen_model_transpose.py` |
| `conversion/trilu/framework/trilu_onnx_plugin.cpp` | Trilu | `test_trilu_onnx_plugin.cpp` | `gen_model_trilu.py` |
| `conversion/unsqueeze/framework/unsequeeze_onnx_plugin.cpp` | Unsqueeze | `test_unsqueeze_onnx_plugin.cpp` | `gen_model_unsqueeze.py` |
| `experimental/math/asinh/framework/max_onnx_plugin.cpp` | Max | `test_max_experimental_onnx_plugin.cpp` | `gen_model_max.py` |
| `math/accumulate_nv2/framework/mean_onnx_plugin.cpp` | Mean | `test_mean_onnx_plugin.cpp` | `gen_model_mean.py` |
| `math/accumulate_nv2/framework/sum_onnx_plugin.cpp` | Sum | `test_sum_onnx_plugin.cpp` | `gen_model_sum.py` |
| `math/addcmul/framework/addcmul_onnx_plugin.cpp` | Addcmul | `test_addcmul_onnx_plugin.cpp` | `gen_model_addcmul.py` |
| `math/arg_max_v2/framework/argmax_onnx_plugin.cpp` | ArgMax | `test_argmax_onnx_plugin.cpp` | `gen_model_argmax.py` |
| `math/arg_min/framework/argmin_onnx_plugin.cpp` | ArgMin | `test_argmin_onnx_plugin.cpp` | `gen_model_argmin.py` |
| `math/bitwise_xor/framework/xor_onnx_plugin.cpp` | Xor | `test_xor_onnx_plugin.cpp` | `gen_model_xor.py` |
| `math/cast/framework/cast_onnx_plugin.cpp` | Cast | `test_cast_onnx_plugin.cpp` | `gen_model_cast.py` |
| `math/cumsum/framework/cumsum_onnx_plugin.cpp` | CumSum | `test_cumsum_onnx_plugin.cpp` | `gen_model_cumsum.py` |
| `math/is_inf/framework/isinf_onnx_plugin.cpp` | IsInf | `test_isinf_onnx_plugin.cpp` | `gen_model_isinf.py` |
| `math/maximum/framework/max_onnx_plugin.cpp` | Max | `test_max_onnx_plugin.cpp` | `gen_model_max.py` |
| `math/minimum/framework/min_onnx_plugin.cpp` | Min | `test_min_onnx_plugin.cpp` | `gen_model_min.py` |
| `math/mod/framework/mod_onnx_plugin.cpp` | Mod | `test_mod_onnx_plugin.cpp` | `gen_model_mod.py` |
| `math/one_hot/framework/onehot_onnx_plugin.cpp` | OneHot | `test_onehot_onnx_plugin.cpp` | `gen_model_onehot.py` |
| `math/reduce_log_sum_exp/framework/reduce_log_sum_exp_onnx_plugin.cpp` | ReduceLogSumExp | `test_reduce_log_sum_exp_onnx_plugin.cpp` | `gen_model_reduce_log_sum_exp.py` |
| `math/reduce_max/framework/reduce_max_onnx_plugin.cpp` | ReduceMax | `test_reduce_max_onnx_plugin.cpp` | `gen_model_reduce_max.py` |
| `math/reduce_mean/framework/reduce_mean_onnx_plugin.cpp` | ReduceMean | `test_reduce_mean_onnx_plugin.cpp` | `gen_model_reduce_mean.py` |
| `math/reduce_min/framework/reduce_min_onnx_plugin.cpp` | ReduceMin | `test_reduce_min_onnx_plugin.cpp` | `gen_model_reduce_min.py` |
| `math/reduce_prod/framework/reduce_prod_onnx_plugin.cpp` | ReduceProd | `test_reduce_prod_onnx_plugin.cpp` | `gen_model_reduce_prod.py` |
| `math/reduce_sum/framework/reduce_log_sum_onnx_plugin.cpp` | ReduceLogSum | `test_reduce_log_sum_onnx_plugin.cpp` | `gen_model_reduce_log_sum.py` |
| `math/reduce_sum/framework/reduce_sum_onnx_plugin.cpp` | ReduceSum | `test_reduce_sum_onnx_plugin.cpp` | `gen_model_reduce_sum.py` |
| `math/right_shift/framework/bitshift_onnx_plugin.cpp` | BitShift | `test_bitshift_onnx_plugin.cpp` | `gen_model_bitshift.py` |
| `math/sign_bits_pack/framework/npu_sign_bits_pack_onnx_plugin.cpp` | NPUSignBitsPack | `test_npu_sign_bits_pack_onnx_plugin.cpp` | `gen_model_npu_sign_bits_pack.py` |
| `math/sign_bits_unpack/framework/npu_sign_bits_unpack_onnx_plugin.cpp` | NPUSignBitsUnpack | `test_npu_sign_bits_unpack_onnx_plugin.cpp` | `gen_model_npu_sign_bits_unpack.py` |
| `math/square_sum_v1/framework/reduce_sum_square_onnx_plugin.cpp` | ReduceSumSquare | `test_reduce_sum_square_onnx_plugin.cpp` | `gen_model_reduce_sum_square.py` |
| `math/topk/framework/topk_onnx_plugin.cpp` | TopK | `test_topk_onnx_plugin.cpp` | `gen_model_topk.py` |
| `random/dsa_random_uniform/framework/random_uniform_onnx_plugin.cpp` | RandomUniform | `test_random_uniform_onnx_plugin.cpp` | `gen_model_random_uniform.py` |
| `random/stateless_random_normal_v2/framework/random_normal_onnx_plugin.cpp` | RandomNormal | `test_random_normal_onnx_plugin.cpp` | `gen_model_random_normal.py` |
| `random/stateless_sample_multinomial/framework/multinomial_onnx_plugin.cpp` | Multinomial | `test_multinomial_onnx_plugin.cpp` | `gen_model_multinomial.py` |
