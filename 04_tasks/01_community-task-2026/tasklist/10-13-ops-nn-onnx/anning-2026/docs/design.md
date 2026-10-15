# ops-nn ONNX 插件迁移设计

## 需求背景（required）

### 需求来源

本设计对应“10月社区任务—ops-nn 算子 ONNX 适配”，依据任务书的protobuf解耦要求，对74个插件文件、75个“文件×算子”条目进行迁移。具体范围以任务书清单为准，覆盖各算子目录、common集中目录及experimental目录。

### 背景介绍

原有插件通过 `ParseParamsFn` 注册参数解析函数，以protobuf Message为源参数，读取ONNX NodeProto的属性、输入输出名称和节点名称，依赖 `ge_onnx.pb.h` 及protobuf对象类型。

本次使用 `ParseParamsByOperatorFn`，由 `ge::Operator` 承载源节点信息，通过 `GetAttr` 读取属性JSON和输入输出名称列表，完成目标算子的参数映射。

## 需求分析（required）

### 需求描述

解除插件参数解析对protobuf对象和生成头文件的直接依赖，保持原有算子映射、属性默认值、输入输出处理及子图构建行为。

本任务属于Host侧framework适配，不涉及kernel、tiling、aclnn接口或算子原型开发。`ParseOpToGraphFn` 和 `DynamicInputRegister` 保持原实现，不进行无关重构。

### 需求拆解

| 改造项 | 设计要求 |
| --- | --- |
| 解析接口 | 源参数改为 `const ge::Operator&`，目标参数继续使用 `ge::Operator&`。 |
| 注册接口 | 替换为 `ParseParamsByOperatorFn`，保留算子名称、ONNX版本范围和执行类型。 |
| 属性解析 | 从 `attribute` 读取JSON字符串，使用仓内nlohmann/json解析。 |
| 输入输出 | 读取 `input`、`output` 名称列表及数量，保留空名称占位。 |
| 节点名称 | 使用 `GetOpName(op_src)`，保持原有目标属性写入行为。 |
| 公共能力 | 通用JSON解析、浮点转换及输入输出计数由公共工具提供。 |
| 异常处理 | 捕获解析及转换异常，记录日志并返回 `FAILED`。 |
| 失效判据 | 按映射工具实际失败返回约定检查，并补充非法取值UT。 |
| 单元测试 | 直接构造Operator，验证属性映射、默认值、异常和边界。 |

## 详细设计（required）

### 1. 处理流程与接口

GE将ONNX节点转换为源Operator，调用插件参数解析函数。插件读取源信息、设置目标属性和描述符，由原有算子映射或子图构建流程继续处理。

函数保留原名称，仅替换源参数类型，接口形式如下：

```cpp
domi::Status ParseParams(const ge::Operator& op_src, ge::Operator& op_dest);
```

| 信息 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 节点属性 | `node->attribute()` | `GetAttr` 获取JSON，由公共工具解析。 |
| 输入数量 | `node->input_size()` | 优先读取输入名称列表的长度。 |
| 输出数量 | `node->output_size()` | 优先读取输出名称列表的长度。 |
| 节点名称 | `node->name()` | `GetOpName(op_src)` |
| 参数注册 | `ParseParamsFn` | `ParseParamsByOperatorFn` |

### 2. JSON属性解析

通过 `op_src.GetAttr("attribute", attrs_string)` 获取属性字符串，典型表示如下：

```json
{
  "attribute": [
    {"name": "alpha", "type": 1, "f": "0.25"}
  ]
}
```

公共工具读取和解析JSON；各插件保留原属性名称、类型筛选、默认值、必填检查和取值约束。

| 属性类型 | 类型值 | 取值字段 | 读取方式 |
| --- | --- | --- | --- |
| FLOAT | 1 | f | 字符串转float；载体键缺省时取0。 |
| INT | 2 | i | 按整数读取；载体键缺省时取0。 |
| STRING | 3 | s | 按字符串读取；载体键缺省时取空字符串。 |
| FLOATS | 6 | floats | 按浮点数组读取。 |
| INTS | 7 | ints | 按整数数组读取。 |
| TENSOR | 4 | t | 按原dtype、dims和数据载体构造Tensor。 |

属性整体缺失与属性存在但取值键缺省分别处理。例如HardSigmoid没有alpha属性时使用原默认值0.2；存在FLOAT类型alpha但省略f键时取0.0。类型不匹配、未知属性和重复属性的处理顺序保持原分支语义。

### 3. 浮点属性处理

标量f以字符串承载，由公共工具调用仓内 `StrToFloat` 转换。字段缺省保持proto3零值语义，非法字符串转换失败；业务约束仍由对应插件管理。

### 4. 默认值、异常与校验

保留原属性默认值、目标属性类型、列表长度、维数、pads重排、枚举及范围检查。解析或取值转换异常记录日志并返回失败，正常分支保留原返回状态。

任务要求修正的失效判据按工具真实返回约定处理，例如dtype映射未命中返回 `ge::DT_UNDEFINED`，并以非法取值UT覆盖。合法映射和其他业务逻辑不变。

### 5. 输入输出与名称处理

真实ATC源Operator可能尚未连接描述符，因此数量优先读取 `input`、`output` 名称列表；列表属性不存在时回退到描述符数量，兼容framework UT的挂载方式。

空字符串代表可选端口占位，保留位置和列表长度。动态端口名称、注册顺序、`original_type` 及描述符格式更新沿用原实现。涉及格式更新的测试为目标Operator注册相应端口。

需要记录节点名称的插件使用 `GetOpName(op_src)` 写入目标name属性。

### 6. Tensor属性处理

ConstantOfShape等插件保留原dtype、dims、元素数量、字节长度和TensorDesc构造方式，分别覆盖标量、空载体和非空payload。

实际GE JSON中的raw_data按C转义格式还原字节，不能直接当作base64。typed数据按相应dtype读取，不通过填零或修改dtype改变原Tensor语义。

### 7. 文件组织与依赖

| 位置 | 职责 |
| --- | --- |
| 各算子 `framework/*_onnx_plugin.cpp` | 参数接口、属性读取和注册方式迁移。 |
| `common/inc/framework/nn_onnx_plugin_util.h` | JSON解析、浮点转换、列表读取及输入输出计数。 |
| 各算子及common的 `tests/ut/framework/` | 构造Operator，验证对应插件解析行为。 |
| framework UT构建规则 | 收集common及experimental范围内的用例。 |

使用 `graph/operator.h`、`register/register.h`、公共工具头和仓内 `nlohmann/json.hpp`，移除对 `ge_onnx.pb.h` 的直接依赖。保留原 `OriginOpType`、版本分支和 `ImplyType`，逐一覆盖同文件多处注册。

### 支持硬件与约束

按任务要求选择Atlas A2或Ascend950，CANN版本大于9.0.0，迁移前后使用同一环境。

不扩展算子的dtype、shape、广播能力或ONNX版本支持范围，不新增分核、UB分块和tilingkey设计。

## 可维可测分析

### 单元测试设计

直接构造 `ge::Operator`，通过 `SetAttr` 设置JSON和输入输出名称列表，调用迁移后的解析函数，不构造protobuf对象。

| 场景 | 验证内容 |
| --- | --- |
| 正常解析 | 目标属性名称、类型、数值、节点名及端口数量。 |
| 属性缺失 | 默认值、必填失败分支及proto3零值省略。 |
| 异常输入 | 非法JSON、载体类型错误、浮点转换失败和业务校验。 |
| 边界情况 | 空列表、负值、重复属性、可选输入占位及端口数量。 |
| Tensor载体 | dtype、shape、空数据、非零正负payload及字节还原。 |
| 特殊分支 | 失效dtype判据、多版本注册、动态输入及格式更新。 |

覆盖全部74个插件文件，开启仓内OP_HOST_UT运行 `Onnx*PluginTest.*`，检查构建、旧依赖残留及 `git diff --check`。

### OM验证方法

使用任务包生成模型、输入和golden，分别加载迁移前后插件完成ATC转换和设备推理。两轮使用相同模型、输入、SDK及转换配置，保存插件加载证据，检查所有输出的dtype、shape和值。

主判据要求两轮输出完全一致，golden按任务规定独立检查。动态输出使用执行后的描述符；模型变体和原始失败分别记录，不用修订后的诊断模型替代原用例结论。

平台ATC失败按任务书允许的情形留证，不自动将推理失败或非确定性输出视为豁免。

### 精度、性能与内存标准

| 验收项目 | 设计标准 |
| --- | --- |
| 参数解析 | 保持原默认值、属性映射、输入输出和组图语义。 |
| 两轮输出 | 相同模型与输入，迁移前后输出完全一致。 |
| golden精度 | 使用任务包混合容差、硬上限及匹配率，校验dtype、shape；整数和布尔精确比较。 |
| 性能 | 不涉及，依据任务书3.3节。 |
| 内存 | 不涉及，依据任务书3.4节。 |

### 兼容性与维护

公共工具仅提供解析能力，各插件管理自身默认值和业务校验。保留原动态注册及子图构建，避免公共工具修改影响其他算子。

等价迁移依赖GE桥接保留原属性信息。桥接或测试材料存在差异时，单独留证并请求维护者确认，不在插件中猜测原值、放宽精度或修改算子计算行为；确认及复验完成后再提交验收。
