# ops-nn 算子 ONNX 适配设计文档

作者：Delicate02。本文依据 2026-10-17 版本任务书及配套用例包，按社区设计模板组织；文档修订日期为 2026-10-22。设计范围为 Host 侧 ONNX 插件等价迁移，覆盖 75 个任务条目、74 个插件文件及 76 处参数解析注册。

# 需求背景（required）

## 需求来源

本设计对应“10 月社区任务——ops-nn 算子 ONNX 适配”，目标是解除参数解析插件对 ONNX protobuf 生成类型的直接依赖。依据任务书第 2 章的功能、工程模式和接口定义，以及第 3 章的迁移前后验证规则实施；参考 [社区任务目录](https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026/tasklist/10-13-ops-nn-onnx)、[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 与 [ops-nn 任务 issue #5115](https://gitcode.com/cann/ops-nn/issues/5115)。

任务书第 2.2 节明确：本任务是 ONNX 插件框架适配工程，不涉及算子 Kernel、Tiling、算子原型、ACLNN 或 Torch 接口开发；`ParseOpToGraphFn` 子图拆分函数不在迁移范围。因此，模板中的算子分析、Host、Kernel、硬件和测试章节均保留，具体内容按该工程边界填写。

## 背景介绍

旧参数回调经 `.ParseParamsFn(...)` 注册，接收 protobuf 消息，通过 `NodeProto` 读取属性、节点名称和输入输出列表。插件直接包含 `ge_onnx.pb.h`、使用生成类型并引用 protobuf 符号，增加与框架内部序列化实现及版本的耦合。

新参数回调经 `.ParseParamsByOperatorFn(...)` 注册，接收 `const ge::Operator&`。GE 将 ONNX 节点信息写入源 Operator，插件读取 `attribute` JSON 和 `input`、`output` 名称列表，将等价的属性和端口设置到目标 Operator。GE 仍负责解析模型，本任务消除的是目标插件实现及其 framework UT 对 protobuf 生成类型的依赖。

基线取任务发布时同仓源码 `0ec2ed9ccc7e69895444cdae6f889b33bf47410b`。迁移采用仓内已存在的 Operator/JSON 接口范式，复用 `common/inc/framework/plugin_util.h` 中的 `StrToFloat`、`GetOpName` 和仓内 nlohmann/json。

# 需求分析（required）

## 需求描述

保持同一合法 ONNX 节点的目标属性名称、类型、取值、默认值、动态端口、名称写入、组图行为及返回状态；保留每个插件独立的非法输入和重复属性规则。迁移仅改变信息载体和依赖边界。任务明确要求修正的失效失败判据，按被调用工具的真实返回约定处理，并以非法值 UT 覆盖。

本次不增加算子类型、ONNX domain/opset、dtype、shape 或广播支持，不改变已有设备计算算法。解析成功不代表目标平台一定拥有后续算子的 op-info、Kernel 或运行时实现，平台限制须通过两轮证据单独判断。

## 需求拆解

| 需求 | 实现落点 | 完成判据 |
| --- | --- | --- |
| 解析函数签名替换 | 原函数名保留，源参数改为 `const ge::Operator&` | 全部 74 文件的目标回调完成替换 |
| 注册方式替换 | `.ParseParamsByOperatorFn(...)` | 76 处注册迁移，保留原名称、版本、FrameworkType、ImplyType |
| 属性解析替换 | `GetAttr("attribute", ...)` + nlohmann/json | 按原遍历顺序及属性分支写入目标 Operator |
| 依赖解耦 | `graph/operator.h`、`nlohmann/json.hpp`、仓内公共头及 `register/register.h` | 目标插件和 UT 不引用生成 protobuf 类型；迁移后插件不加载旧接口兼容库 |
| 节点名称替换 | 原使用节点名的位置调用 `op_dest.SetAttr("name", GetOpName(op_src))` | 名称值和原写入位置保持一致 |
| 公共工具复用 | `common/inc/framework/ops_nn_onnx_plugin_util.h` | 使用仓前缀，公共能力不吞并各插件业务分支 |
| 返回值与异常 | 保留默认值、失败条件和忽略规则；捕获解析异常 | 正常、缺失、非法值及未知异常路径有对应检查 |
| framework UT | 各插件配套测试与公共工具测试 | 覆盖全部插件文件，执行名单、日志和结果可回溯 |
| OM 自验证 | 同模型、同输入、同 ATC 参数，两轮注入不同插件 | 按新版任务书及用例 meta 的声明逐例判定 |

## 范围与文件分组

| 目录族 | 任务条目 | 插件文件 | 主要内容 |
| --- | ---: | ---: | --- |
| `activation/` | 10 | 10 | 激活函数、Softmax 版本和损失函数 |
| `common/src/framework/` | 36 | 36 | 常规解析、量化子图、RNN 子图等 |
| `conv/deformable_conv2d/` | 1 | 1 | NPU DeformableConv2D |
| `experimental/norm/fused_add_rms_norm/` | 1 | 1 | experimental 插件 |
| `index/` | 8 | 8 | 索引、Scatter、Unique 等 |
| `norm/` | 6 | 6 | 归一化插件 |
| `pooling/` | 2 | 2 | 池化插件 |
| `quant/` | 9 | 8 | 量化与反量化插件 |
| `rnn/` | 2 | 2 | NPULstm、NPULstmCell |
| 合计 | 75 | 74 | 76 处参数解析注册 |

NPUDynamicQuant 与 NPUDynamicQuantV2 在任务表中各占一条，但共用一个文件；Softmax 包含不同版本的两处注册。附录 A 给出全部条目和准确文件路径，不以目录数量代替文件覆盖。

# 详细设计（required）

## 算子分析

### 语义模型与数学关系

本任务不新增算子数学公式。设旧解析函数为 `P_old`，GE 的节点到 Operator 桥接为 `B`，新解析函数为 `P_new`，设计目标为：

```text
P_new(B(node)) 与 P_old(node)
  在返回状态、目标属性、端口注册、名称及后续子图输入信息上等价。

确定性模型：Infer(Compile(P_new(B(model))), input)
          与 Infer(Compile(P_old(model)), input) 逐字节一致。
```

非确定性或预期失败用例按新版任务书明确允许的声明判定，见可维可测章节。这些例外不授权改变插件语义。

### 输入、输出与支持约束

| 对象 | 内容与类型 | 约束 |
| --- | --- | --- |
| `op_src` | `const ge::Operator&` | 承载 GE 生成的节点信息，不在插件中修改 |
| `attribute` | 字符串，顶层为 `{"attribute":[...]}` | 保留属性数组顺序，不提前转成按名称去重的字典 |
| `input` / `output` | `vector<ge::AscendString>` | 空字符串仍占一个可选端口位置 |
| `op_dest` | `ge::Operator&` | 按旧实现写入属性、Tensor、动态端口及格式 |
| 回调结果 | `domi::SUCCESS` / `domi::FAILED` | 保留各分支状态及失败前的写入顺序 |
| 模型 Tensor | dtype、shape、format 由原算子及原插件定义 | 不统一改成 FP32，不扩大 rank、广播或 dtype 范围 |

### 属性表示与默认值

| 类别 | JSON 载体 | 等价处理 |
| --- | --- | --- |
| FLOAT（type=1） | `"f":"0.25"` | 按字符串解析；属性存在但 `f` 省略时取 0.0 |
| INT（type=2） | `"i":1` | 按原整数类型读取；`i` 省略时取 0 |
| STRING（type=3） | `"s":"..."` | 解开 GE 的 C 转义；`s` 省略时取空串 |
| TENSOR（type=4） | `"t":{...}` | 按原 dtype、原数据载体及消费范围构造 Tensor |
| FLOATS（type=6） | `"floats":[...]` | 数值列表，保留顺序和原校验 |
| INTS（type=7） | `"ints":[...]` | 整数列表，保留顺序、空列表和原长度约束 |

属性整体缺失时执行原默认或必填检查；属性存在但标量字段省略时执行 protobuf 默认零/空串语义。例如 HardSigmoid 的 alpha 缺失为 0.2，alpha 存在且 `f` 缺失为 0.0。原插件不检查 type 的分支仍按原方式处理，不能用公共过滤器统一增加 type 限制。

## 算子实现

### 3.2.1 Host 侧设计

#### 调用链与接口

```mermaid
flowchart LR
    model[ONNX 模型] --> parser[GE ONNX Parser]
    parser --> bridge[Message2Operator 桥接]
    bridge --> src[源 Operator: JSON 和名称列表]
    src --> callback[ParseParamsByOperatorFn]
    callback --> dest[目标 Operator: 属性和端口]
    dest --> graph[原有子图构造与 GE 编译]
    graph --> om[OM]
    om --> infer[既有设备实现推理]
```

参数回调保留原名称，签名统一为：

```cpp
domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest);
```

注册链只替换参数回调注册接口；`OriginOpType` 的 domain/opset 列表、`ImplyType`、原有 `ParseOpToGraphFn` 以及其他注册项按基线保留。`PartitionedCall` 类型仍使用原子图函数。

属性读取通过公共工具集中处理：

```cpp
ge::AscendString encoded;
// GetAttr 失败时，ReadOnnxAttributes 返回空属性数组；各插件继续原默认/必填分支。
op_src.GetAttr("attribute", encoded);
// 对成功读取的字符串使用 nlohmann::json::parse，再按原顺序消费 attribute 数组。
```

实际消费属性的回调包围 `try`，捕获 `std::exception` 和 `catch (...)`，记录节点与错误并返回 `FAILED`。原本完全不读取属性的回调不人为增加 JSON 解析，例如 NPUGru、NPUDynamicQuant/V2；无论属性串是否合法，仍执行原固定参数或空操作路径。

#### 公共工具与资源生命周期

新增 `common/inc/framework/ops_nn_onnx_plugin_util.h`，以 `ops_nn_` 区分其他算子仓公共头。仅添加迁移需要的能力，复用既有 `plugin_util.h`，不改写已有公共接口。

| 工具 | 职责 |
| --- | --- |
| `ReadOnnxAttributes` | `GetAttr` 取 JSON；属性键不存在、包装中无数组时返回空数组；语法异常交给回调捕获 |
| `GetOnnxFloat` | 复用 `StrToFloat`；拒绝非法字符串；对 libc 以 ERANGE 报告但可表示的非零 FP32 次正规数/最小正规数保留实际值 |
| `DecodeOnnxBytes` / `GetOnnxString` | 恢复 GE C 转义，包括八进制、NUL、引号和反斜杠；拒绝截断或非法转义 |
| `GetOnnxInputCount` / `GetOnnxOutputCount` | 获取源名称列表长度；缺失列表取 0；保留空占位并检查 int 范围溢出 |
| `Vec2Tensor` / `CreateScalar` | 向原子图提供不依赖 protobuf 的等价 Tensor 构造工具 |
| `ChangeFormatFromOnnx` | 保留原 TensorDesc 格式更新和返回值处理 |

JSON、字符串和 Tensor payload 使用局部对象管理生命周期；Tensor 构造完成前数据缓冲保持有效。不引入跨节点可变缓存、全局解析状态或设备内存分配。公共工具异常由插件边界统一转换成框架状态。

#### 输入输出、名称与动态端口

源 Operator 的 `input`、`output` 名称列表对应 NodeProto 的原始输入输出列表；`GetInputsSize()` 等描述符数量不能替代它。ATC 参数解析时，源节点描述符可能尚未挂接，数量可为零。缺失源列表按 0 处理，不能为了 UT 退回描述符数量。

空字符串表示 ONNX 可选输入占位，保持位置和总数。原固定动态端口数、名称前缀、`original_type` 写入和注册顺序保持不变。目标描述符仍用于原有格式更新；相关 UT 正例应构造必要的目标描述符，另设缺失描述符负例。

#### 特殊插件的语义保持

| 插件/场景 | 处理策略 |
| --- | --- |
| HardSigmoid、Gemm | 分开处理属性缺失与 `f` 缺失；Gemm transA/transB 独立，fixed_shift_value 越界仍按原分支告警并回退 |
| Conv、DeformableConv2D | 保留 strides/dilations 扩充、pads 重排、长度检查、格式更新及原缺省行为 |
| Softmax | 保留旧/新 opset 注册及对应 axis 默认值；两条版本路径分别验证 |
| Celu/CeluV2 | 保留原注册，在独立插件配置中验证，避免重叠注册遮蔽目标回调 |
| AscendQuant | 保留 round_mode、sqrt_mode、dst_type 的独立类型检查及字符串裁剪规则；按 GE DataType 枚举断言 |
| NPULstm、NPULstmCell | 保留输入占位、方向/标志属性和重复项的原状态变化，不强制套用“最后一项覆盖” |
| NPUGru、NPUDynamicQuant/V2 | 保留不读取 JSON 的原路径，不因添加 JSON 校验改变成功/失败行为 |
| ConstantOfShape | 默认 FP32 标量零，形状 `{}`、NCHW；非空 raw_data 优先，否则按原支持的 typed carrier 消费首元素 |
| 重复属性 | 保留每个插件的首次命中、末次命中、列表追加或遇错即失败，保持属性写入时机 |

ConstantOfShape 的 raw_data 是 GE C 转义字节，不按 Base64 解码。typed carrier 仍使用原实现支持的 INT64、INT32、FLOAT、DOUBLE 分支；不把缺失载荷改写成任意补零数据。数据类型大小为负时不得转成无符号分配长度；raw 载荷不足一个目标元素时返回失败。保留原支持范围及空载荷行为，并通过 Tensor 的 dtype、shape、字节数和非零载荷 UT 检查。

#### 失败判据与依赖边界

逐插件检查失败分支和工具返回约定。任务书所举的 `if (om == -1)` 仅在工具实际返回 -1 时有效；若映射失败返回 `ge::DT_UNDEFINED`，应以该枚举判断并增加非法值负例，不能照抄永不成立的判断。此为逐处核对规则，不把它误写成所有插件共有的错误，也不将其他既有回退分支统一改为失败。

插件文件移除 `ge_onnx.pb.h` 和 protobuf 类型，采用 Operator/JSON 及公共工具头；已有子图代码仅替换所依赖的等价辅助工具，不重构子图算法。构建检查针对目标编译对象的依赖与符号，区分 GE 框架自身仍使用 protobuf 和本次插件是否已解耦。

### 3.2.2 Host/Tiling 设计

这里的 Host 工作是模型转换期的参数解析，不是设备 Tiling。依据任务书第 2.2 节，不新增 TilingData、TilingKey、分核策略、UB 分块、Workspace 计算或调度参数。已有算子的 shape 推导、Tiling 和 Workspace 策略保持原实现；JSON 解析不得修改这些算法以迎合用例。

### 3.2.3 Kernel 侧设计

不涉及 Kernel 新增或修改。算子的数学运算、数据搬运、向量/Cube 指令、double buffer、同步与设备内存访问均由既有实现承担。迁移前后在同一环境、同一模型与输入下执行，用设备输出检验解析等价性；不提出新增 Kernel 性能指标。

### 3.2.4 文件组织、实施与回退

| 位置 | 用途 |
| --- | --- |
| 附录 A 的插件文件 | 参数回调签名、属性读取和注册接口迁移 |
| `common/inc/framework/ops_nn_onnx_plugin_util.h` | 共享解析和等价辅助工具 |
| 各算子及 `common/tests/ut/framework/` | 对应插件 UT；公共工具独立 UT |
| framework UT 构建发现规则 | 纳入 common、experimental 和公共工具测试 |
| 私仓 `task_submission/` | 自验证步骤、报告、逐算子自查表、脚本、修订说明和证据 |

按目录组实施和检视，逐文件核对 baseline → migration 的差异，再构建主组、experimental、celu_v1、celu_v2 配置。完整范围及注册数固定，不能因单个算子平台不支持而漏测。

复现时 pre/post 插件和共同依赖置于独立目录，通过环境变量按进程加载。回退到旧插件目录即可重新运行基线，不覆盖系统 CANN 安装。产品迁移、framework UT、用例工具与 GE 候选修复分别保留来源及变更说明，便于独立评审。

## 支持硬件与算子约束限制

| 层次 | 本设计要求与已验证边界 |
| --- | --- |
| 任务硬件 | Atlas A2 或 Ascend 950 任一平台；任务推荐 A2 |
| 已用硬件 | Ascend910B3（A2）；每个用例两轮在同一物理卡执行 |
| 已用软件 | Linux aarch64、CANN 9.1.0、Python 3.11；满足任务 CANN >9.0.0 要求 |
| Host ABI | 插件须针对对应 Host 架构、C++ ABI、CANN 头文件/库构建；aarch64 二进制不能直接用于 x86_64 |
| ATC 目标 | `--framework=5 --soc_version=Ascend910B3`；两轮 ATC 配置一致 |
| Ascend 950 | 属任务允许平台，本次未实测，不宣称现有 aarch64/910B3 证据覆盖 950 |
| 设备执行 | 沿用现有算子 dtype、shape、动态形状和平台支持范围，不添加 op-info 或 Kernel |

公共 GE 桥接必须保留原属性值。已在 CANN 9.1.0 观察到 scalar FLOAT 格式化丢失有效位，以及 typed DOUBLE 桥接分支问题，反馈于 [GE issue #656](https://gitcode.com/cann/ge/issues/656)。插件收到已丢位字符串后无法恢复原位模式，因此本轮两套插件共同使用独立 GE FLOAT/DOUBLE 桥接候选修复库。修复用于表示转换，不修改 ops-nn 算子计算；源码和构建记录在私仓 `task_submission/delivery/ge-bridge-code-review/` 与 `ge-bridge-fix-candidate/`。

该候选补丁尚不作为“上游已合入”的事实。本轮结果证明共同修复环境中的迁移等价性，不声称原装 CANN 无补丁通过。旧插件若缺少 `ge::onnx::AttributeProto` 符号，使用任务包 compat 方案；此兼容库仅用于 pre，post 不挂载。它与两轮共同使用的 GE 桥接修复是不同依赖。

# 可维可测分析

## 精度标准/性能标准

采用新版任务书第 3.2、3.5 节及配套 `test_cases/README.md` 第 4.1 节。验证只比较迁移前后，不以 golden 或 onnxruntime 替代两轮结论。

| 判据 | 前提与通过条件 |
| --- | --- |
| 默认 bitwise | 相同模型、输入、ATC 参数，两轮输出数量、dtype、shape 及数据字节完全一致；区分正负零和 NaN 载荷 |
| 随机算子 | 按任务要求固定 seed；未声明例外时仍按 bitwise |
| tolerance | 有同 OM 重复推理的非确定性依据，meta 显式声明档位和原因；执行官方混合容差标准，不放宽阈值 |
| behavior | 仅用于任务包明确声明且有依据的不可复现用例；检查两轮行为和 shape/dtype，不宣称数值相等 |
| 预期 ATC/推理失败 | 预先声明的模型/平台条件；两轮失败方式一致，保留阶段、实质错误、要求的关键字和加载证据 |
| 未声明失败或单轮失败 | 判失败；不能通过临时改 meta 或跳过模型抵消 |

`tolerance` 使用任务包原 `precision_standard.py`：FP32 `rtol=atol=2^-13`，FP16 为 `2^-10`，BF16 为 `2^-7`；匹配率至少 0.99，并满足其最大绝对误差上限 `max(固定兜底值, 32×ULP)` 和特殊值规则。整数/布尔按精确规则；标准以用例包引用的 opbase 提交 `a5e8e716743b20bfb2ffa16e0213c4aa73b823b3` 为准。不能使用旧版 FP32 阈值覆盖新版工具。

性能与内存专项验证按任务书第 3.3、3.4 节“不涉及”，报告和日志注明 N/A，不伪造测速、带宽、UB 占用或内存优化结果。

## 分层测试设计

### 静态检查与构建

逐文件检查签名、注册、头依赖、名称取值、默认值、失败分支和原子图差异；核对 75 条目、74 文件、76 注册的覆盖。对目标编译对象检查生成 protobuf 头和类型符号残留，执行 `git diff --check`。四组 pre/post 插件分别构建，避免 Celu 重叠注册和 experimental 构建遗漏。

### framework UT

UT 直接构造源/目标 Operator 和 JSON，不构造 NodeProto。正例测试属性、目标类型及数值；负例测试非法 JSON、浮点载体、非法业务值和原失败条件；边界测试缺失值字段、空列表、重复属性、极值、bytes/Tensor 和可选端口占位。公共工具覆盖 CEscape、浮点边界及输入输出计数。原无 JSON 路径单独断言保持 SUCCESS。

以 `Onnx*PluginTest.*` 及对应公共工具执行清单核对实际 Suite.Test、XML 和日志，要求无遗漏、失败或跳过。全部 74 个插件有对应 UT；公共工具额外覆盖不替代插件测试。目标输入/输出描述符与源名称列表分别构造，不混用。

新版许可修订官方 UT，但须说明原因并给出代码，交验收方复核。实际修订包括源列表构造、DeformableConv2D 目标描述符、GE 枚举断言、NPUGru 不读 JSON 路径、ConstantOfShape 空载荷及新增边界覆盖。逐文件说明和 31 份 UT 差异保存在私仓 `task_submission/acceptance-20261017/调整说明.md`、`ut-adaptations/`；复核应针对这套已披露用例，不能用原包覆盖后仍引用本套运行结果。

### 两轮 OM 验证

1. 获取任务发布时未迁移插件和纯迁移插件；两轮使用相同 CANN、共同 GE 候选库和硬件。旧接口兼容库隔离到 pre。
2. 执行全部 74 个模型生成器，将模型、输入与 meta 生成一次并冻结。官方模型及变体全部纳入；不同用例可多卡并行，同一用例两轮必须同卡。
3. 实际调用任务包 `verify_om/two_round_compare.sh`。通过 `ASCEND_CUSTOM_OPP_PATH` 注入各轮插件，在 ATC 日志记录路径和实际解析回调，避免加载内置插件或同一套插件导致假通过。
4. 按同一 ATC 配置转换后，使用任务包 pyACL 脚本推理与比较。GRU、AscendAntiQuant 按已披露配置使用官方默认精度模式，其余使用 `--precision_mode=must_keep_origin_dtype`；每例两轮配置相同。
5. 检查输出数量、dtype、shape、字节或声明判据，校验插件、共同 GE 动态库及 pre/post 环境隔离；失败保留原始错误与逐例终态。
6. 汇总结果、逐例命令和日志。任一缺失证据、未声明失败、源数据不一致或输出不一致导致流程非零退出。

官方 74 个生成器产生 189 个用例；保留并新增 Softmax V11 两个、DynamicQuant/V2 三个变体，共 194 例。两个生成器的新增代码、四个 shell 文件 CRLF→LF 修订、调度工具及参数说明均在 `调整说明.md` 和 `test-adaptations/`。官方比较、精度标准和 pyACL 实现保留；用例 meta 中原有非确定性和预期失败声明保留，不将本轮失败临时改为预期失败。

## 当前自验证结果与复现入口

以下为私仓已保存的 2026-10-17 自验证记录，不是本次文档改动重新运行的结果，也不代表任务方已验收或设计已合入。

| 项目 | 结果与解释 |
| --- | --- |
| framework UT | 779/779：主组 769，experimental 10；763 个插件用例 + 16 个公共工具用例 |
| bitwise | 155 例两轮输出逐字节一致 |
| tolerance | 1 例：`instance_norm_v3`，按官方声明混合容差通过 |
| behavior | 1 例：`gru/no_hidden_size_attr`，按官方声明行为一致；不声称数值相等 |
| 预期 ATC 失败一致 | 35 例；属于声明的负例/平台限制，不计为 NPU 数值通过 |
| 预期推理失败一致 | 2 例 SparseToDense；不计为 NPU 数值通过 |
| 全部声明判据 | 194/194，覆盖 74 文件、75 条目；实际 388 次 ATC、318 次推理尝试 |
| 环境边界 | 仅共同使用 GE 桥接候选修复，无额外 Host、Unique 推导或 DynamicQuant SDK 修复 |

源码与证据入口为[私仓固定交付提交](https://gitcode.com/Delicate02/ops-nn/commit/760510cfab04119587c374886cc5ff6fbb78f284?ref=task_submission)，需要验收账号的私仓权限。纯迁移产品提交为 `0a0903f7a22d43e4b9f4300cf1813b8fbc8a8954`，新增官方 UT 断言提交为 `f603d145d914421a002326db486952242834c82d`。

在满足上述 910B3/CANN 9.1.0 环境并取得私仓权限后：

```bash
git clone --branch task_submission https://gitcode.com/Delicate02/ops-nn.git
cd ops-nn
git checkout 760510cfab04119587c374886cc5ff6fbb78f284
bash task_submission/run_accuracy.sh --output /data/opsnn-review-001 --devices 0
```

输出目录须为新目录；一张卡即可，八卡可设置 `--devices 0,1,2,3,4,5,6,7`。入口恢复随仓提供的 pre/post 插件、GE 库和 UT 产物，运行 UT 后执行全部模型生成、官方两轮脚本和审计。构建源码、补丁及说明随仓提供；该入口是固定平台二进制重放，跨 Host/CANN 平台应先重新构建。已有独立容器重放记录可供核对。

| 私仓位置（均相对仓根） | 内容 |
| --- | --- |
| `task_submission/source_scope.json` | 条目→插件→UT→OM 的准确映射及单项自查 |
| `task_submission/acceptance-20261017/调整说明.md` | 用例与工具逐文件修订原因 |
| `task_submission/acceptance-20261017/判据与例外.md` | 非 bitwise 用例的逐例声明与依据 |
| `task_submission/acceptance-20261017/verification/result.json` | 汇总状态 |
| `task_submission/acceptance-20261017/verification/comparison.json` | 194 个用例及逐输出判定 |
| `task_submission/acceptance-20261017/verification/command-logs.json` | 实际 ATC/推理命令与两轮日志位置 |
| `task_submission/acceptance-20261017/verification/ut/` | main/experimental 实际日志与 XML |
| `task_submission/acceptance-20261017/verification/execution-audit.json` | 388 次 ATC、318 次推理、加载隔离及同源数据检查 |

仓内 ATC 日志为含原始行号、加载和错误信息的摘录，推理日志完整；全量调试日志、OM 和输出由入口在新输出目录重新生成。部分动态输出按官方 pyACL 工具的声明容量处理，不将声明容量视为独立验证的运行时有效长度。

## 兼容性分析与风险控制

| 风险/交叉影响 | 控制措施 |
| --- | --- |
| GE 桥接浮点信息丢失 | 两轮共同使用已披露候选修复；保留源码、最小反例和构建记录，由维护者独立复核 |
| 旧 protobuf 符号兼容 | 仅 pre 使用任务包 compat；确认 post 未加载兼容库 |
| 公共工具改变个别语义 | 工具只处理载体；重复、默认、类型检查、忽略及失败策略仍在原插件分支 |
| 重叠/多版本注册漏覆盖 | 分组构建 Celu；单独覆盖 Softmax 两版本和 DynamicQuant 两注册 |
| 测试绕开真实输入 | 冻结同模型/输入/meta，记录实际 ATC 命令与插件派发；保留官方主模型及全部变体 |
| 非确定性被误作解析回归或被滥用 | 依赖同 OM 重复运行证据和明确 meta 声明；默认保持 bitwise |
| 原包 UT 与修订 UT 混用 | 提交逐文件差异、原因、执行清单和真实日志，由验收方复核后按同套用例重放 |
| 硬件或 SDK 变更 | 在新环境重新构建和执行；不外推 910B3/CANN 9.1.0 结果 |

设计 PR 仅提交本文一个文件。实现、用例、补丁及证据留在开发者私仓。交付件包含步骤说明、精度报告及日志、性能/内存 N/A 报告及日志、单算子自查表；上传包可只含这些文档，完整重放依赖从私仓取得。流程为设计审核并合入后提交任务验收，工作人员复核私仓后再按要求提交公开产品代码 PR。现有自验证结果不能代替上述审核与合入门槛。

# 参考资料

- [ParseParamsByOperatorFn 接口文档](https://raw.gitcode.com/cann/metadef/raw/af8b521f310f3816ee19c08b96a4e7eb50cef6d1/docs/zh/api/ge_namespace/opregistrationdata/ParseParamsByOperatorFn.md)。
- [GE Message2Operator](https://raw.gitcode.com/cann/ge/raw/d594f984d82c352acb17a797846a25db780a617f/parser/parser/common/convert/message2operator.cc)、[Pb2Json](https://raw.gitcode.com/cann/ge/raw/d594f984d82c352acb17a797846a25db780a617f/parser/parser/common/convert/pb2json.cc)：桥接机制参考，不据此推定已安装二进制的源码提交。
- [生态混合容差标准（用例包引用版本）](https://gitcode.com/cann/opbase/blob/a5e8e716743b20bfb2ffa16e0213c4aa73b823b3/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
- [社区流程](https://gitcode.com/org/cann/discussions/39)；同任务已合入设计 [#2182](https://gitcode.com/cann/cann-ops-competitions/pull/2182)、[#2246](https://gitcode.com/cann/cann-ops-competitions/pull/2246)、[#2142](https://gitcode.com/cann/cann-ops-competitions/pull/2142) 仅作章节组织参考，本文验证数据来自本项目。

KG 调用链定位：id `ge_parser_parser_onnx_onnx_custom_parser_adapter_cc_operator`，source_file `graph-compiler/ge/parser/parser/onnx/onnx_custom_parser_adapter.cc`，score `0.8989787101745605`；具体接口和实现以本任务基线及已核对源码为准。

# 附录 A：完整迁移范围

以下路径均相对 ops-nn 仓根，与私仓 `task_submission/source_scope.json` 一一对应。UT 路径由该清单完整列出：算子目录采用其 `tests/ut/framework/`，common 插件采用 `common/tests/ut/framework/`；表中条目 70、71 共用插件及 UT 文件。公共工具测试另在 `common/tests/ut/framework/test_ops_nn_util_onnx_plugin.cpp`。

| 序号 | ONNX 算子条目 | 插件文件 |
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
