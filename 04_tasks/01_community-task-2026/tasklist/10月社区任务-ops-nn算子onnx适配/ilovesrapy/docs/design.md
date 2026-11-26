# ops-nn 算子 ONNX 插件 protobuf 解耦迁移方案

申请人：ilovescrapy，对应社区任务：10 月「ops-nn 算子 onnx 适配」。

这是一份动手改造之前写下的方案文档。它只回答三个问题：要改什么、打算怎么改、改完怎么证明没改坏。文中不包含实施结果、性能数据或任何"已完成"的表述。

# 一、任务来龙去脉

## 1.1 任务出处

本次工作对应下发的《ops-nn算子onnx插件protobuf解耦任务书》，验证手段以任务包附带的 `test_cases/README.md`、framework UT、OM 模型生成脚本和两轮比对脚本为准。目标代码仓为 [cann/ops-nn](https://gitcode.com/cann/ops-nn)，改造动机可参考 [issue #5115](https://gitcode.com/cann/ops-nn/issues/5115)；设计评审环节遵循[社区任务统一流程](https://gitcode.com/org/cann/discussions/39)。

## 1.2 现状与要解决的问题

ops-nn 仓目前有 75 个 ONNX 插件条目（分布在 74 个源文件中）仍走老路子注册参数解析：`ParseParamsFn` 的函数签名里挂着一个 protobuf 消息对象（运行时实际是 `ge::onnx::NodeProto`），插件函数体里手工遍历 attribute 列表，按 i/f/s 等字段抠值。这条路径有三个实际痛点：插件源码 include 了 `ge_onnx.pb.h`，proto 一动全家受牵连；编译要拉 protobuf 头文件和生成代码，链路被拖长；写单测必须先拼 protobuf 对象，成本高。

改造方向是任务书规定的等价重构：注册函数换成 `ParseParamsByOperatorFn`，入参改为 `ge::Operator`；属性不再走 protobuf 遍历，而是 `op_src.GetAttr("attribute")` 拿到 JSON 字符串后用 nlohmann/json 解析。目标只有一个——插件代码与 protobuf 生成代码彻底断开，同时解析语义一个比特都不变。

需要强调"不变"的度量口径：对同一个 ONNX 节点，迁移前后的返回状态、写入 op_dest 的属性集合、输入输出注册、以及后续 ATC 转换行为应完全一致；唯一的例外是任务书点名的失效失败判据（详见 3.2 节），这类照抄会带 bug 的死代码按工具函数真实返回约定修正。

# 二、范围与契约

## 2.1 改什么、不改什么

改造对象即任务书迁移范围表所列的 75 条 × 74 个文件（逐条清单见附录），以开工时对仓内源码的静态扫描结果为准。几个边界情况提前约定：`NPUDynamicQuant` 和 `NPUDynamicQuantV2` 两条对应同一个源文件；celu 相关的两条分属 `activation/celu` 和 `activation/celu_v2` 两个目录，各自独立覆盖；同一文件里注册的多个 ONNX 版本算子全部纳入核对。

改动边界收敛在三类文件：目标插件的 `framework/*_onnx_plugin.cpp`、其对应的 framework UT、以及确实绕不开的公共工具或构建接入点。kernel、tiling、算子原型一律不碰；`ParseOpToGraphFn` 子图拆图逻辑、`DynamicInputRegister` 动态输入注册、ONNX opset 注册区间、子图拓扑结构，以及一切与解耦无关的逻辑，全部保持原样——不趁机重构。

动手前会固定两样东西：任务包哈希、目标仓完整 commit 号，并与任务书依据的 `6e5b184` 快照比对。若清单内的插件已经被别人迁移、移动或删除，差异会如实记录并向社区求证，绝不静默缩减范围，更不会拿已迁移的旧代码冒充迁移前基线。

## 2.2 解析函数契约

```cpp
domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest);
```

这是任务书 §2.3 给出的目标形态；具体到每个插件，函数名沿用原有命名（如 ParseParamsGemm、parse_params_hard_sigmoid），不重命名。注册链上唯一的替换点是 `.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`。

| 位置 | 语义 | 迁移后契约 |
| --- | --- | --- |
| op_src | GE 源算子 | `GetAttr("attribute")` 取 JSON 文本；节点名、输入个数等原本用到的信息改走 Operator 接口 |
| attribute JSON | 属性载体 | 顶层对象内含 attribute 数组，逐项含 name、type 与值字段（f 为字符串承载，i/ints/floats 为数字） |
| op_dest | GE 目标算子 | 只写原实现会写的属性、名称与描述，不多写不少写 |
| 返回值 | SUCCESS / FAILED | 默认值回填、必填缺失失败等分支逐一保持；JSON 非法或未知异常记日志后返回 FAILED |

这份契约不涉及任何新的 dtype/shape 约束：各算子的 dtype、shape、广播规则和动态输入能力仍由算子原型与框架既有约束决定，迁移不动它们。

# 三、迁移方案

## 3.1 单个插件的改造步骤

每个插件按同一套动作走一遍，顺序固定：先通读原实现，把默认值、必填校验、失败条件逐条抄成笔记；再换头文件（删 `ge_onnx.pb.h` 及一切 protobuf 类型引用，加 `operator.h`、`nlohmann/json.hpp`、仓内工具头、`register/register.h`）并改函数签名；然后取 attribute JSON，按原实现同样的顺序遍历属性；逐个属性做类型转换和原有校验；按原来的名字写入 op_dest 属性；注册版本原样保留；最后同步改 UT。

头文件替换后，目标插件里不允许残留任何 protobuf 生成类型。JSON 库用仓内 third_party 自带的 nlohmann/json，不另引入第二个版本。公共工具优先复用 `common/inc/framework/plugin_util.h` 现成的 `StrToFloat` 和 `GetOpName`；实在需要新公共头时，按仓前缀命名并保持最小职责。

**include 供给链必须整体保留，不做"裸替换"。** `onnx_common.h` 在迁移前承担两条唯一的供给链，替换它时必须把这两条供给补到新 include 集合里，否则会出编译级联失败：

1. **REG_OP 宏路径**：`onnx_common.h:25 → stub_ops.h:19 → graph/operator_reg.h`。`common/inc/op_graph/op_nn_proto_extend.h` 自身零 include，却直接使用 `REG_OP(...)` 定义扩展算子原型——仓内现状是靠"先 include onnx_common.h 再 include op_nn_proto_extend.h"的顺序隐式供给。而 `register/register.h` 与 `graph/operator.h` 并不保证包含 `operator_reg.h`（CANN 9.2.0 实测命中为 0），所以解耦后该供给必须显式化：在 `op_nn_proto_extend.h` 内自足 include（`graph/operator_reg.h` 或经 `common/stub/inc/framework/stub_ops.h`），使该头不再依赖任何外部包含顺序。
2. **三个 helper 的定义**：`Vec2Tensor`（onnx_common.h:37）、`CreateScalar`（:47）、`ChangeFormatFromOnnx`（:56）只在这一个头里定义。凡迁移后仍在使用它们的插件，不得各自复制本地副本（此前只在 conv_onnx_plugin.cpp 补过一份，属不应扩散的临时做法）；统一做法是把三个 helper 收编进公共工具头 `plugin_util.h`（连同原有的 `StrToFloat`/`GetOpName` 一起），由使用方插件 include 该头取得，单一来源、无副本。

据此，迁移后的 include 集合按"实际用到什么补什么"确定：需要 REG_OP/算子原型的经自足后的 `op_nn_proto_extend.h`，需要 helper 的经 `plugin_util.h`，其余按任务书取 `register/register.h`、`graph/operator.h`、`nlohmann/json.hpp`。完成后对每个迁移文件做 include 闭包静态核验：逐一展开确认 `REG_OP`、`Vec2Tensor`、`CreateScalar`、`ChangeFormatFromOnnx` 均有可达声明，不允许残留对 `onnx_common.h` 的隐式顺序依赖。另注意 `op_nn_proto_extend.h` 在 master 上已有 20 个既有消费方（全部经由 onnx_common.h 取得 REG_OP），修改它时只做"新增自足 include"，不改动任何既有声明，保证不破坏仓内现有包含方。

同族 PR（对同一批 74 个插件独立迁移的 #12174/#12185/#12118）与本 PR 采用的是同一套替换模式，上述两条供给链问题在这些 PR 中同样存在；收敛方案（`op_nn_proto_extend.h` 自足 include + helper 收编 `plugin_util.h`）将在评审意见闭环后与各同族 PR 对齐，避免同一根因在各仓各修各的。

节点名的处理跟着原实现走：只有原来真的会把 node name 写进属性的地方，才换成 `op_dest.SetAttr("name", GetOpName(op_src))`；原本不写 name 的插件不新增。原实现用到的输入数量等信息改为从 Operator 对应接口获取，动态输入注册逻辑原封不动。

## 3.2 语义逐项保持（含失效判据修正）

属性解析是本次迁移最容易走样的地方，逐类约定如下：

| 场景 | 迁移后的行为 |
| --- | --- |
| attribute 整体缺失 | 沿用算子原默认值；原本必填、缺失即失败的属性照旧失败 |
| FLOAT 属性带 f | f 以字符串承载，按 `StrToFloat` 转 float；空串、非法内容或越界按约定返回 FAILED |
| FLOAT 属性存在但无 f 键 | 按 proto3 零值省略语义取 0.0，不回退算子默认值（如 alpha 存在、无 f → alpha=0） |
| INT 属性 i 省略 | 在原有匹配规则下按 proto3 零值语义处理 |
| ints / floats | 按数字数组读取；原有的长度检查、维度提升、交错重排全部保留 |
| type 与取值键不匹配 | 对原本校验 type 的属性按缺失处理；源码与配套 UT 明确不校验 type 的属性保持原行为 |
| 同名属性重复出现 | 维持原有的顺序语义（取首个、覆盖或累加），不做统一去重 |
| 未认识的属性 | 沿用原来的忽略行为 |
| JSON 非法 / 转换异常 | 带上算子与属性上下文记日志，返回 FAILED，不用空属性顶替非法输入 |
| dtype 映射未命中 | 按工具函数真实返回值（如 DT_UNDEFINED）判定失败，并补负向 UT |

特别说明失效判据问题：迁移前代码里可能存在永远不成立的失败判断（典型如用 `om == -1` 判 dtype 映射失败，而 `GetOmDtypeFromOnnxDtype` 未命中时实际返回 `ge::DT_UNDEFINED`）。照抄等于把死分支带进新代码，这类位置一律按工具函数的真实返回约定重写，并补充"非法取值应返回 FAILED"的 UT 覆盖——这是任务书明确要求纠正的点，不计入"行为变化"。

实施时重点盯防的分支清单：HardSigmoid 的 alpha/beta 默认值与零值省略覆盖关系；Gemm 的 transA/transB 相互独立（单侧出现时另一侧保持 false）、fixed_shift_value 越出 [34,43] 告警回退默认值；Conv/Pool 系的 pads 奇数长度返回 FAILED、交错重排与 4 元 NCHW 提升；kernel_shape 缺失时 dim_size/trans_2d 的推导与始终写出；Softmax 跨 opset 的 axis 语义；归一化系的属性改名与 epsilon；量化系 dtype 映射与多输出；循环网络系的可选输入与属性列表。

## 3.3 批量实施与构建

先建立"任务条目 ↔ 源文件 ↔ 注册版本 ↔ 属性契约 ↔ UT ↔ 模型脚本"六元映射表，再按语义分组推进：基础标量一组、列表与形状一组、量化一组、归一化一组、循环网络一组。每组先挑代表插件试刀——计划用 HardSigmoid、Gemm、Conv、一个量化插件加一个无属性插件校准做法，确认模式后再铺开到组内其余文件。

脚本化只干四件事：清点、机械替换候选、残留依赖扫描、结果汇总；每一处语义差异仍逐文件人工复核。每组改完跑对应 UT，收尾跑全量 framework UT。UT 放各算子 `tests/ut/framework/`（common 系放 `common/tests/ut/framework/`），沿用仓内相对 include；common 集中目录的改动要能触发 CI 执行 framework UT（参照 issue #5115 关联 PR 的做法核对）。

构建命令为 `bash build.sh --onnxplugin -j8`；UT 走仓内 OP_HOST_UT 开关与实际构建入口，过滤条件 `--gtest_filter='Onnx*PluginTest.*'`。提交前跑 `git diff --check`，并扫描范围内是否残留旧注册方式与 protobuf 引用，核对依赖与符号干净；随构建一并执行 §3.1 约定的 include 闭包静态核验（REG_OP 与三个 helper 声明可达），CI 全 Job 编译通过作为供给链修复的最终实证。不为了消除某个无关的仓依赖而扩大改动面。

## 3.4 环境与兼容处理

两轮验证取自同一套环境：Atlas A2（Ascend 910B3）、CANN 9.1.0，满足任务书"CANN >9.0.0、A2 或 Ascend 950 任一通过（推荐 A2）"的要求。Python 用 3.11（任务书实测 3.11.15），onnx/onnxruntime/numpy 锁定实际版本，参考任务书实测组合 onnx 1.23.2、onnxruntime 1.31.0、numpy 2.4.6（建议下限 onnx≥1.14、onnxruntime≥1.16、numpy≥1.21，完整清单见配套《test_cases/README.md》§4.2）；验证模型按算子用 opset 11/13/17、统一 ir_version=8。om 推理选 pyACL（CANN 自带，`python3 -c "import acl"` 能 import 即满足）。

迁移前基线轮若碰到 `undefined symbol: …ge::onnx::AttributeProto…` 之类的报错，属 CANN 版本对旧接口插件的符号导出问题，与迁移代码无关——按任务包 compat 指引给旧插件挂兼容库即可，并把适配方式与库哈希记档；迁移后插件一律不挂该兼容库（不再依赖 protobuf 类型，这正是本任务的结论）。两轮的构建目录与 ASCEND_CUSTOM_OPP_PATH 指向的自定义 OPP 目录严格隔离，防止旧库串进新轮。

# 四、验证与证据

## 4.1 UT 覆盖

配套用例包含 74 个 UT 文件，源码实测定义 731 条 TEST；任务书正文写 726，与配套 README 及源码计数存在出入——覆盖以实际用例全集为准，不删减任何用例。若官方中途更新用例，记录版本变化。验收目标：全部执行通过，不存在未说明原因的禁用、跳过或过滤遗漏。

每个插件的用例补齐四类场景：正常解析、属性缺失、非法 JSON、边界/校验，断言点落在返回状态与 op_dest 属性上。UT 直接构造 ge::Operator，不构造任何 protobuf 对象。对失效判据修正等必要行为变化补充失败用例；绝不通过改弱断言来抹平差异。

## 4.2 两轮 OM 一致性验证

执行全部 74 个模型生成脚本及其自动变体，形成实际模型清单——脚本数不等于模型数，以清单对账。输入按官方脚本与固定随机种子生成：标准随机输入取正态 μ=0、σ=1，属性取可区分值（transA=1、alpha=1.5、pads=[1,2,3,4] 这类），保证解析错误能在输出 dtype/shape/数值上被观测到。golden 取脚本指定来源：onnxruntime 或脚本内 numpy 参考实现；其中 39 个脚本的模型 onnxruntime 不可执行（含全部 17 个 NPU 自定义算子），golden 全部由 numpy 参考实现产出（见《test_cases/README.md》§4.5）。

流程分五步：

1. 迁移前：冻结源码、构建环境与插件库哈希，保存模型、输入、golden 与 meta.json；
2. 基线轮：ASCEND_CUSTOM_OPP_PATH 注入迁移前插件，ATC `--framework=5 --soc_version=Ascend910B3` 转换并推理，保留完整日志与输出；
3. 迁移轮：同一模型同一输入，先核对文件哈希未变，再注入迁移后插件，重复转换与推理；
4. 每例留存本轮插件库实际路径与注册证据（LoadCustomOpLib 日志），证明生效的是本轮插件而非回落内置实现，证据不足不算完成；
5. 对输出数量、名称对应、shape、dtype、数值逐项比对；实际清单与预期模型清单对账，缺项不记 PASS。

主判据：迁移前后输出完全一致，逐数值相等，不设 rtol/atol。确定性输出按官方脚本以字节一致方式核对；allclose 仅作诊断手段，不替代主判据。若个别用例输出非确定，而脚本例外口径与任务书主判据冲突，保留原始数据提交社区确认，确认前不以放宽口径宣称通过。

## 4.3 辅助精度标准

对 om 推理输出与 golden 按混合容差标准做辅助比对，阈值逐输出按其自身 dtype 选取，避免把整型输出的零阈值错套到同模型的浮点输出上：

| dtype | rtol | atol | fixed_limit |
| --- | --- | --- | --- |
| FP32 | 2^-10 | 2^-16 | 1e-2 |
| FP16 | 2^-9 | 2^-9 | 1e-1 |
| BF16 | 2^-6 | 2^-6 | 1 |
| 整数/布尔 | 0 | 0 | 精确一致 |

浮点逐元素匹配条件：`abs(actual-golden) <= atol + rtol*abs(golden)`，整体匹配率 ≥0.99；每个元素另受硬上限 `abs(actual-golden) <= max(fixed_limit, 32*ULP(abs(golden)))` 约束。NaN/Inf 位置与符号按官方标准核对，shape 或 dtype 不符直接判失败。硬上限不可关闭，阈值不可自行放宽。

`meta.golden=false` 的用例退化为两轮行为一致性验证，并写明原因。预期转换失败的用例要求两轮均失败且报错一致；因平台 op-info 缺条目等无法转换的，保留两轮相同报错原文与设备信息，单列为"行为一致"，不计入数值精度通过。一轮成功一轮失败的必须定位原因，不以平台问题豁免。

工具侧计划直接使用任务包内 two_round_compare.sh、run_om_and_compare_acl.py、compare_rounds.py 与 precision_standard.py，另做完整性核验。对脚本只警告不拦截的几类情况——插件加载证据缺失仅告警、shape 声明数量不符、全局容差影响混合 dtype、未比对实际失败原因——保留原始工具不动，以执行补丁方式补足，并记录补丁内容与理由，保证判据不弱于任务书。

## 4.4 性能与内存

本任务不触碰计算 kernel，任务书 §3.3/§3.4 明确不设性能与内存验收指标。对应交付报告与日志标注"不涉及"，不设 ratio 门槛，不编造测试数据。

## 4.5 自查与证据留档

逐插件过一遍自查表：头文件替换（含 REG_OP 供给与三个 helper 的来源核验，见 §3.1）、签名与注册宏、JSON 读取、name 获取方式、默认值分支、失败分支、全部 opset、UT 更新、构建、diff。报告记录实际源码 commit、插件 SHA256、软硬件版本、模型与输入哈希、各日志路径。截图取自真实服务器终端或端口映射后的内置浏览器，覆盖环境信息、UT 汇总、两轮对照与插件加载证据。

# 五、实施安排与交付

| 阶段 | 主要工作 | 完成标志 |
| --- | --- | --- |
| 设计评审 | 提交本文，澄清范围与歧义点 | 评审意见闭环 |
| 基线准备 | 固定版本，构建迁移前插件，生成模型与输入 | 基线可追溯，平台限制已记录 |
| 分组迁移 | 代表插件校准后铺开至全部 74 个文件 | 各组 UT 通过，范围对账完整 |
| 全量自验 | 全量 framework UT、两轮 OM、辅助精度比对 | 清单对账无缺项，异常均有分类证据 |
| 交付合入 | 自查表、私仓交付、代码 PR 与 CI | 交付件齐全，评审意见闭环 |

代码按插件文件/算子分组提交 ops-nn，遵循仓内规范；正式 PR 只收必要源码、UT 与接入改动，大体量测试证据放私仓交付目录与验收包。私仓 task_submission 目录按下发模板组织七个标准交付件（1 自验证步骤说明.md；2.1/2.2 精度自验证报告与日志；3.1/3.2 性能自验证报告与日志，标注不涉及；4.1/4.2 内存自验证报告与日志，标注不涉及），另附单算子迁移自查表、模型生成与复现入口、原始日志与截图。PR 描述与提交内容不携带任何个人敏感信息。

# 附录：迁移范围清单（75 条 / 74 文件）

以下按任务书迁移范围表逐条列示；实施时逐条关联对应 UT 与 OM 记录，并核对实际源码版本。

| 编号 | ONNX 算子 | 插件相对路径 |
| ---- | -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
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

