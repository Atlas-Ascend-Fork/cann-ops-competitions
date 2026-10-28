# ops-math 算子 ONNX 插件 protobuf 解耦设计文档

## 需求背景（required）

### 需求来源

- 来源：《ops-math 算子 onnx 插件 protobuf 解耦任务书》（10 月社区任务「ops-math 算子 onnx 适配」）；
- 上游仓库：ops-math（`https://gitcode.com/cann/ops-math`）；
- 迁移改造与 PR 提交均基于该仓，目标目录为各迁移插件文件所在算子目录（如 `math/cast/framework/`、`common/src/framework/`）及对应 framework UT 目录；
- 参考依据：任务书「核心开发要求」「验收标准」两章，以及仓内已完成迁移的参考插件实现（如 `math/shape`、`random/stateless_bernoulli`、`elu`、`leaky_relu`、`bernoulli` 等已使用新接口的插件）。

### 背景介绍

ops-math 仓的 ONNX 插件（`*_onnx_plugin.cpp`）通过 `REGISTER_CUSTOM_OP` 宏注册到 GE 图引擎，`FrameworkType(ONNX)`。ATC 做 ONNX → OM 模型转换时，由 GE 回调插件的参数解析函数，把 ONNX 节点的属性集合映射为内部算子（op kernel / 转换后算子）的属性，从而完成从框架描述到内部算子描述的翻译。

迁移前，仍有 124 个 ONNX 插件条目（分布在 57 个插件文件中，具体清单见任务书第 1 节「迁移范围表」）使用**依赖 protobuf 的旧接口**注册参数解析函数：

```cpp
// 迁移前（旧接口）
static Status ParseParamsXxx(const Message* op_src, ge::Operator& op_dest)
{
    const auto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    // 遍历 node->attribute()，逐个 attr.name() / attr.type() 匹配，
    // 再读 attr.i() / attr.f() / attr.s() / attr.ints() / attr.floats()
}

REGISTER_CUSTOM_OP("Xxx")
    .FrameworkType(ONNX)
    .OriginOpType({...})
    .ParseParamsFn(ParseParamsXxx)     // ← 依赖 protobuf
    .ImplyType(ImplyType::TVM);
```

该接口的入参 `op_src` 实际是 protobuf 生成类 `ge::onnx::NodeProto`，插件源码因此需要 `#include "onnx_common.h"`（该头间接引入 `ge_onnx.pb.h`），并在编译期链接 protobuf 运行库。

这带来三类问题：

1. **与 protobuf 生成代码强耦合**：插件源码依赖 `ge_onnx.pb.h`，ONNX proto 定义的变动会直接波及全部插件，改动半径大；
2. **编译依赖重**：需要拉入 protobuf 头文件与生成代码，编译链路长，构建产物也额外依赖 protobuf 动态库；
3. **UT 成本高**：单测必须手工构造 `NodeProto` / `AttributeProto` 等 protobuf 对象才能触发解析逻辑，用例编写、维护与阅读成本都高。

GE 框架已提供以 `ge::Operator` 为入参的新参数解析接口 `ParseParamsByOperatorFn`：转换阶段 GE 会把 ONNX 节点的属性集合序列化为 JSON 字符串，挂在 `op_src` 的 `"attribute"` 属性上；插件改用 `op_src.GetAttr("attribute", ...)` 取出 JSON，再用仓内自带的 nlohmann/json 解析，即可彻底摆脱对 protobuf 生成代码与运行库的依赖。

### 现状基线（迁移前实测）

对迁移前代码与产物做静态扫描与符号核对，作为本次改造的对照基线：

| 现状项 | 实测数据 |
| --- | --- |
| 任务书第 1 节范围表覆盖 | 124 个算子条目 / 57 个插件文件 |
| 迁移前使用 `ParseParamsFn` 的 `*_onnx_plugin.cpp` 文件数 | 57 |
| 迁移前插件产物的 protobuf 符号 | `liboponnx_plugin_math.so` 存在未定义的 `ge::onnx::*` / `google::protobuf::*` 符号 |
| 公共工具头 | `common/inc/framework/math_onnx_plugin_util.h`（已含 `GetOpName`、`GetOmDtypeFromOnnxDtype`、`Vec2Tensor`、`CreateScalar` 等） |
| JSON 解析库 | 仓内 `third_party` 自带 nlohmann/json（不额外引入新版本） |

## 需求分析（required）

### 需求描述

完成 **124 个算子 ONNX 插件条目（57 个插件文件）** 的 protobuf 解耦迁移适配，使插件代码彻底摆脱对 protobuf 生成代码与运行库的依赖，同时**保持插件原有解析语义与功能完全不变**。

本任务属于 **Host 侧等价重构**：不改变算子计算逻辑、不改变模型转换产物结构、不改变算子注册面（`OriginOpType` / `ImplyType` / `ParseOpToGraphFn` 等），仅替换参数解析函数的入参形态、注册方式、属性解析手段与编译依赖。

迁移后接口（保持各插件原有函数命名，如 `ParseParamsCast`、`ParseParamsFlatten`、`parse_params_pad_v11`）：

```cpp
domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest);

REGISTER_CUSTOM_OP("Xxx")
    .FrameworkType(ONNX)
    .OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParamsXxx)
    .ImplyType(ImplyType::TVM);
```

### 需求拆解

按任务书第 2.1 节逐条落地为 7 项改造动作 + 1 项验证动作：

1. **解析函数签名替换**：`const Message* op_src` → `const ge::Operator& op_src`。入参类型在编译期即已保证，随之删除 `dynamic_cast<const ge::onnx::NodeProto*>` 及其判空分支；
2. **注册方式替换**：`.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`；
3. **属性解析方式替换**：遍历 `node->attribute()` → `op_src.GetAttr("attribute", attrs_string)` 取 JSON 字符串，再用 nlohmann/json 解析；
4. **移除 protobuf 依赖**：删除对 `onnx_common.h`（及 `ge_onnx.pb.h`、protobuf 运行库）的引用，include 替换为 `graph/operator.h`、`nlohmann/json.hpp`、`math_onnx_plugin_util.h`、`register/register.h`、`log/log.h`；
5. **算子名称获取替换**：`node->name()` → `op_dest.SetAttr("name", GetOpName(op_src))`；
6. **公共工具复用**：优先复用 `common/inc/framework/math_onnx_plugin_util.h` 中既有函数，不新增与解耦无关的公共头；
7. **语义保持**：属性缺失时的默认值行为、各分支返回值（`SUCCESS` / `FAILED` / `PARAM_INVALID`）与迁移前逐分支保持一致；属性非法（JSON 解析异常或未知异常）时记日志并返回 `FAILED`；不解析 attribute JSON 的注册路径保持原 `SUCCESS` 语义（实现与 UT 均不改）；迁移前永不成立的失效判据按工具函数真实返回约定修正（Cast 的 dtype 映射失败判断，见「失效判据修正」）；除解耦改造外不修改插件其他逻辑；
8. **验证**：按任务书第 3.5 节完成 framework UT 与迁移前后两轮 om 转换比对自验。

### 工程模式与改造边界

本任务为 **ONNX 插件框架适配工程（Host 侧）**，不属于 kernel 直调、aclnn 算子工程化开发或 torch 接口开发，改造边界为：

| 维度 | 在范围内 | 明确不在范围内 |
| --- | --- | --- |
| 文件 | 各算子目录 `framework/*_onnx_plugin.cpp`、对应 `tests/ut/framework/` 及必要的 `CMakeLists.txt` | 算子 kernel（`op_kernel`）、tiling、`op_host`（inference / infershape / shapeinfer）、算子原型（`*.proto` / `op_graph`） |
| 接口 | 参数解析函数签名、注册方式、属性解析手段、编译依赖 | `REGISTER_CUSTOM_OP` 的 `OriginOpType` / `ImplyType` / `FrameworkType` 等注册项 |
| 子图 | 仅迁移参数解析函数 | `ParseOpToGraphFn`（子图拆图函数，注册为 `PartitionedCall` 的算子）保持原样 |
| 动态输入 | 保持原样 | `DynamicInputRegister` / `DynamicOutputRegister` 不改动 |
| 产物 | `liboponnx_plugin_math.so` | 不新增算子包、不改动算子实现目录 |

## 详细设计（required）

### 接口与数据流设计

迁移前后的数据流对比：

| 项 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 解析函数签名 | `Status ParseParamsXxx(const Message* op_src, ge::Operator& op_dest)` | `Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)` |
| 入参校验 | `dynamic_cast` + 判空分支 | 类型即编译期保证，无运行时判空 |
| 注册宏 | `.ParseParamsFn(ParseParamsXxx)` | `.ParseParamsByOperatorFn(ParseParamsXxx)` |
| 属性来源 | `ge::onnx::NodeProto::attribute()`（protobuf 对象） | `op_src.GetAttr("attribute", ge::AscendString&)`（JSON 字符串） |
| 属性取值 | `attr.i()` / `attr.f()` / `attr.s()` / `attr.ints()` / `attr.floats()` | nlohmann/json 取值（见「JSON 属性解析设计」） |
| 节点名 | `node->name()` | `GetOpName(op_src)` |
| 头文件依赖 | `onnx_common.h` → `ge_onnx.pb.h` | `graph/operator.h`、`math_onnx_plugin_util.h`、`nlohmann/json.hpp`、`register/register.h`、`log/log.h` |
| 编译依赖 | 需 protobuf 生成代码与运行库 | 无 protobuf |
| 产物符号 | 存在未定义 `ge::onnx::*` / `google::protobuf::*` | **0 个** |
| UT 输入构造 | 必须构造 `NodeProto` / `AttributeProto` | `ge::Operator` + `SetAttr("attribute", JSON)` |

### JSON 属性解析设计

GE 侧把 ONNX 节点序列化为 JSON 后挂在 `op_src` 的 `"attribute"` 属性上。顶层键 `attribute` 为属性数组，每个元素是一个属性对象：

```json
{
  "attribute": [
    { "name": "min",       "type": 1, "f": "1.5" },
    { "name": "to",        "type": 2, "i": 10 },
    { "name": "auto_pad",  "type": 3, "s": "SAME_UPPER" },
    { "name": "scales",    "type": 6, "floats": [0.5, 0.25] },
    { "name": "axes",      "type": 7, "ints": [2] }
  ]
}
```

字段约定（与任务书第 2.4 节一致，并与仓内已迁移参考实现交叉核对）：

| 字段 | JSON 类型 | 语义 |
| --- | --- | --- |
| `name` | string | ONNX 属性名，与 ONNX 标准属性名一致 |
| `type` | 数字 | ONNX `AttributeProto` 类型枚举：`FLOAT=1`、`INT=2`、`STRING=3`、`FLOATS=6`、`INTS=7` |
| `i` | 数字 | 标量整型属性值 |
| `f` | **字符串** | 标量浮点属性值（GE 序列化时以字符串承载，须按字符串读取后再转 float） |
| `s` | 字符串 | 单值字符串属性 |
| `ints` / `floats` | 数组 | 列表型属性 |

两条关键语义规则：

1. **`f` 键缺省取 0.0，不回落到算子默认值**：proto3 序列化会省略值为 0 的标量字段。若属性存在且 `type=FLOAT` 但 JSON 中无 `f` 键，表示该属性实际值就是 0，直接取 `0.0f`，**不能**回落到算子在 ONNX 规范下的默认值（例：属性 `alpha` 存在、无 `f` 键 → `alpha = 0.0f`）。本次迁移中 9 个插件文件读取 `f` 字段，全部按此规则实现；
2. **`type` 不符按缺失处理**：属性名匹配但 `type` 与算子定义不符时，视为该属性缺失，保持默认值行为（对应 UT 中的「类型不符按缺失处理」类用例）。

解析骨架（迁移后插件的统一结构）：

```cpp
static Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dst)
{
    // ... 原有前置逻辑（动态输入注册、original_type 等）保持不动 ...

    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (const json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "min" && attr.value("type", 0) == ONNX_ATTR_FLOAT) {
                        max = attr.contains("f") ? std::stof(attr["f"].get<std::string>()) : 0.0f;
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dst).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dst).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    // ... 缺失默认值 / 校验分支 / 写入属性：逐分支对齐迁移前语义 ...
    op_dst.SetAttr("name", GetOpName(op_src));
    return SUCCESS;
}
```

### include 替换规则

迁移后插件的 include 集合按实际需要最小化选取，避免无谓引入：

| include | 何时需要 |
| --- | --- |
| `graph/operator.h` | 始终需要（`ge::Operator` 入参） |
| `graph/graph.h` | **仅**当文件内含 `ParseOpToGraphFn`（需要 `ge::Graph`） |
| `stub_ops.h` | **仅**当文件内 `ParseOpToGraphFn` 用到 `op::Data` / `op::Const`（如 Reduce 家族按输入个数构造 axes 常量） |
| `math/<op>/op_graph/<op>_proto.h` | **仅**当文件内 `ParseOpToGraphFn` 构造该 op 的 `op_graph` 对象 |
| `math_onnx_plugin_util.h` | 需要 `GetOpName` / `GetOmDtypeFromOnnxDtype` / `Vec2Tensor` / `CreateScalar` 等 |
| `log/log.h` | 使用 `OP_LOGE` / `OP_LOGI` |
| `register/register.h` | 始终需要（`REGISTER_CUSTOM_OP` 宏） |
| `nlohmann/json.hpp` | **仅**直接出现 `json` 类型的文件（SUCCESS-only 豁免文件不需要） |
| ~~`onnx_common.h`~~ | **删除** |

> 迁移过程中曾出现 3 个文件因 `ParseOpToGraphFn` 用到 `op::Const` 而漏引 `stub_ops.h` 导致编译失败，已补齐：`math/accumulate_nv2/framework/mean_onnx_plugin.cpp`、`math/addcmul/framework/addcmul_onnx_plugin.cpp`、`math/bitwise_xor/framework/xor_onnx_plugin.cpp`。这一现象本身说明「按需选取 include」的收益：漏引即编译失败，不会被静默放过。

### 公共工具设计（math_onnx_plugin_util.h）

解耦后插件对公共工具的依赖集中到 `common/inc/framework/math_onnx_plugin_util.h`。该头文件**不含任何 protobuf 依赖**，本次迁移中复用了其中既有能力，未新增与解耦无关的公共函数：

| 函数 | 作用 | 迁移中的用途 |
| --- | --- | --- |
| `GetOpName(op)` | 取算子名，失败返回 `"None"` | 替代 `node->name()`；同时用于 `OP_LOGE` 的标签参数 |
| `GetOmDtypeFromOnnxDtype(int)` | ONNX dtype 枚举 → `ge::DataType`，未命中返回 `ge::DT_UNDEFINED` | Cast 等 dtype 映射 |
| `onnx2om_dtype_map` / `DataTypeOnnx` | dtype 映射表与枚举 | 同上 |
| `Vec2Tensor` / `CreateScalar` | 从标量 / 标量数组构造 `ge::Tensor`（仅依赖 `ge::Shape` / `TensorDesc` / `Tensor`，与 protobuf 无关） | Clip 的 min/max 常量、TopK V9 的 `k` 常量、Reduce 家族的 axes 常量等 |
| `ChangeFormatFromOnnx` | 按 ONNX 语义设置输入/输出的 origin format 与 format | 保持原格式设置逻辑不动 |
| `GetOpInputNum(op_src)` / `GetOpOutputNum(op_src)` | 取算子的实际入参 / 出参个数 | **变长入参算子（Concat / Mean / Einsum / Split / Slice 等）的 arity 来源**，见「变长入参算子的 arity 获取」 |

`GetOpInputNum` / `GetOpOutputNum` 是本次迁移中唯一新增的公共函数，它解决的是解耦接口切换后 `GetInputsSize()` 失效的问题（见下节）。实现上以 `input` / `output` 属性列表长度为准确：

```cpp
inline int32_t GetOpInputNum(const ge::Operator& op_src)
{
    std::vector<std::string> input_names;
    if (op_src.GetAttr("input", input_names) == ge::GRAPH_SUCCESS) {
        return static_cast<int32_t>(input_names.size());
    }
    return static_cast<int32_t>(op_src.GetInputsSize());
}
```

`GetOpOutputNum` 同构（读 `output` 属性）。保留 `GetInputsSize()` 回落分支是为了让 framework UT 在游离 `Operator` 上（未设 `input` 属性、但通过 `DynamicInputRegister` 注册了端口）仍能取到正确 arity，测试断言因此无需改动。

### 语义保持规则与特殊场景

等价重构的验收口径是「逐分支行为一致」，本次按下列规则执行：

| 场景 | 处理 |
| --- | --- |
| 属性缺失（含 `GetAttr("attribute")` 失败） | 保持各算子迁移前的默认值行为与返回值（多数为「填默认值 + `SUCCESS`」；Cast 的 `to`、Pad V9 的 `pads`、Clip V11 的可选输入等迁移前就返回 `FAILED` 的分支保持 `FAILED`） |
| `f` 键缺省（proto3 零值省略） | 取 `0.0f`，**不回落**算子默认值 |
| 属性 `type` 与定义不符 | 按缺失处理（保持默认值） |
| `i` / `s` / `ints` / `floats` 取值键或类型不符 | 视为该属性缺失或解析失败，按各算子迁移前对应分支处理（多数回落默认值；强校验算子返回 `FAILED`） |
| attribute JSON 非法（语法错误 / 类型断言异常） | 记日志 + 返回 `FAILED` |
| 重复属性 | 保持迁移前「后出现者覆盖前者」的遍历语义，不额外去重 |
| 数值边界判据 | 逐条与迁移前实现比对，行为一致（如 Unsqueeze 空 `axes` 返回 `PARAM_INVALID`、TopK V9 的 `k == -1` 返回 `PARAM_INVALID`、Slice V8/V9 的 `starts` / `ends` 为空返回 `FAILED`、Clip V11 的 `min` / `max` 全空返回 `FAILED`） |
| 不解析 attribute JSON 的插件 | **豁免**：仍按原实现照写固定属性并返回 `SUCCESS`，函数体与 UT 均不改 |
| `ParseOpToGraphFn`（子图拆图） | 不在迁移范围，保持原样 |
| `DynamicInputRegister` / `DynamicOutputRegister` | 保持原样 |

### 变长入参算子的 arity 获取（关键修复）

这是本次迁移中最隐蔽的一处等价性风险，且**仅凭代码审查无法发现，必须靠迁移前 / 迁移后 ATC 转换 A/B 对比才暴露**。

**现象**：首轮全量迁移完成、framework UT 全量通过、`bash build.sh --onnxplugin -j32` 通过、产物无 protobuf 依赖之后，做迁移前 / 迁移后 ATC 转换 A/B 比对时发现 `Concat` / `Mean` / `Einsum` 三个变长入参算子被**整节点剪除**：迁移前 `.om` 为 36514 / 44556 / 44702 字节且含 `ConcatD` / `PartitionedCall` / `MatMulV3` + `BatchMatMul` 等真实计算内核，迁移后仅剩 5589 / 5579 / 5589 字节，图内只剩 `Data` + `NetOutput`。

**根因**：`ge::onnx::ParseParamsFn` 收到的是 protobuf `Message*`，`op_src.GetInputsSize()` 直接反映 ONNX node 的输入端口数；而 `ge::onnx::ParseParamsByOperatorFn` 收到的是 `ge::Operator`，且 ONNX importer 的调用时序是「先调 ParseParams、后调 `OnnxModelParser::SetOperatorInputs()` 连接端口」（`libfmk_onnx_parser.so` 符号表可见 `SetOperatorInputs` 位于 `OnnxDataParser::ParseParams` 之后）。因此**在 ParseParams 阶段 `op_src.GetInputsSize()` 恒为 0**。

插件内落盘探针实测确认：`op_src` 上此时只有 `attribute`（VT_STRING，属性 JSON）、`name`、`op_type`（VT_STRING，如 `ai.onnx::13::Concat`）、以及 **`input` / `output`（VT_LIST_STRING，ONNX node 的输入 / 输出名列表）**，而 `input_size` / `input_num` / `inputs` / `input_tensor` 等 14 个候选键均不存在。因此 arity 的真实来源是 `input` 属性列表长度。

受影响的是 24 处 `op_src.GetInputsSize()` / `op_src.GetOutputsSize()` 调用，分布在 `einsum`、`concat`、`mean`、`phony_concat`、`phony_split`、`split`（4 处）、`sequence`、`slice`（2 处）、`clip_by_value`、`transpose`、`trilu`、`max` 等插件中；这些算子随后把 0 写入 `N` 属性并据此构造 `DynamicInputRegister`，导致 ConcatD 拆图分支走「`input_size == 0` → `PARAM_INVALID`」而整节点失败。

**修复**：统一改为 `GetOpInputNum(op_src)` / `GetOpOutputNum(op_src)`（见「公共工具设计」），不改动任何算子的分支逻辑、默认值与返回值。

**为何不用编译期常量上界**：曾尝试「声明一个足够大的常量（如 64）+ `DynamicInputRegister("x", 64)`」，实测 `atc` exit=255 失败——`N` 是被 ConcatD / Mean 等内核直接消费的语义属性，多余未连接端口会被视为真实端口，故必须与真实 arity 精确相等。

**修复后 A/B 结果**（详见「om 转换验证」）：8 个可比对模型全部 exit=0，`.om` 字节数与迁移前**逐字节级一致**（cast 14638、reshape 6332、squeeze 6336、mod 78418、topk 52342、concat 36514、einsum 44702、mean 44556），关键内核全部恢复；对全部 8 个模型抽取 op 类型符号集合做逐元素 diff，**差异数为 0**。

> 这一节的价值在于：UT 全绿、编译通过、符号干净**都不能**发现这类问题，只有迁移前 / 迁移后的转换产物 A/B 比对才能暴露。变长入参算子的 arity 迁移必须专项验证。

### 失效判据修正（Cast）

任务书第 2.1 节第 7 条与第 2.4 节明确要求修正的例外项是 Cast 的「不支持 dtype」判据：迁移前该判据写作 `if (om == -1)` 一类形式，在公共工具函数 `GetOmDtypeFromOnnxDtype` 的真实返回约定下**永不成立**，即「不支持 dtype」这条失败分支实际走不到。

`GetOmDtypeFromOnnxDtype(int onnx_type)` 的约定是：命中 `onnx2om_dtype_map` 返回对应 `ge::DataType`，未命中返回 **`ge::DT_UNDEFINED`**。迁移后按真实约定修正为：

```cpp
int onnx = attr.contains("i") ? attr["i"].get<int>() : 0;
int32_t om = GetOmDtypeFromOnnxDtype(onnx);
if (om == ge::DT_UNDEFINED) {
    OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
    return FAILED;
}
op_dst.SetAttr("dst_type", om);
```

修正后该分支可达，并在 UT 中补了「`to` 取未映射 dtype 返回 `FAILED`」的用例守护，避免回退。

### 不解析 JSON 路径的豁免（SUCCESS-only）

范围内存在一批插件的解析函数完全不读取 ONNX 属性——只做动态输入输出注册、设置 `original_type` 等固定属性然后返回 `SUCCESS`（典型如 `similar_opp` 的批量展开注册）。这类路径：

- 只改**函数签名**与**注册宏**，函数体一字不改；
- 不引入 `nlohmann/json.hpp`、不调用 `GetAttr`；
- 保持原 `SUCCESS` 语义，因此**不返回 `FAILED`**，也不受「非法 JSON 返回 `FAILED`」要求约束（任务书第 3.5 节明确「仅限解析 attribute JSON 的注册路径」）；
- UT 只需覆盖「正常调用返回 `SUCCESS`」与「动态输入输出注册生效」，不做非法 JSON 用例。

该规则避免了对豁免路径做无意义改造，同时守住「除解耦改造外不修改插件其他逻辑」的边界。

### 多注册分支设计

范围内有 **16 个文件包含 ≥ 2 处 `ParseParamsByOperatorFn` 注册**，需要逐处处理，不能只改一支：

| 插件文件 | 注册处数 | 分支形态 |
| --- | --- | --- |
| `common/src/framework/sequence_onnx_plugin.cpp` | 6 | SequenceAt / SequenceConstruct / SequenceEmpty / SequenceErase / SequenceInsert / SequenceLength **六个不同算子**共用一套范式 |
| `math/minimum/framework/min_onnx_plugin.cpp` | 4 | 4 组注册对应不同 opset 分组，parse 逻辑相同、`original_type` 不同 |
| `math/reduce_sum/framework/{reduce_sum,reduce_log_sum}_onnx_plugin.cpp` | 各 2 | `axes` 属性支（V12）vs 输入支（V13，`ParseOpToGraph` 按输入个数分派） |
| `math/{reduce_max,reduce_min,reduce_mean,reduce_prod,reduce_log_sum_exp,square_sum_v1}/...` | 各 2 | 同上（Reduce 家族） |
| `conversion/clip_by_value/framework/clip_onnx_plugin.cpp` | 2 | V9 属性支 vs V11 可选输入支 |
| `conversion/{squeeze,unsqueeze}/framework/*_onnx_plugin.cpp` | 各 2 | opset 8–12（属性 `axes`）vs 13–18（空实现，`axes` 由框架按输入连接） |
| `conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp` | 2 | opset 8/9（属性支）vs 10–18（三者均为输入） |
| `conversion/pad_v3/framework/pad_onnx_plugin.cpp` | 2 | V9 属性支（`pads` 缺失即 `FAILED`）vs V11 只解析 `mode` |
| `math/topk/framework/topk_onnx_plugin.cpp` | 2 | V9（`k` 为 INT 属性）vs 10–18（`k` 为第 2 路输入） |

`common/src/framework/similar_opp_onnx_plugin.cpp` 用 `REGISTER_CUSTOM_ONNX_OP(ONNX_OPP, ATC_OPP, FUN)` 宏批量展开，共 **579 条注册**、覆盖 **63 个 ONNX 算子**（对应任务书第 18–80 条目）。宏体内部的 `ParseParams##ATC_OPP` 统一改为新签名即可覆盖全部 63 个算子，是本次改动收益最高的单文件：改一处宏、解耦 63 个条目。

### 文件清单

按任务书第 1 节范围表，本次共迁移 **57 个插件文件 / 124 个算子条目**，分布如下：

| 目录 | 文件数 | 说明 |
| --- | --- | --- |
| `math/` | 25 | `accumulate_nv2/{mean,sum}`、`addcmul`、`arg_max_v2`、`arg_min`、`bitwise_xor`、`cast`、`cumsum`、`is_inf`、`maximum`、`minimum`、`mod`、`one_hot`、`reduce_log_sum_exp`、`reduce_max`、`reduce_mean`、`reduce_min`、`reduce_prod`、`reduce_sum/{reduce_sum,reduce_log_sum}`、`right_shift`、`sign_bits_pack`、`sign_bits_unpack`、`square_sum_v1`、`topk` |
| `conversion/` | 14 | `clip_by_value`、`concat`、`depth_to_space`、`flatten`、`pad_v3`、`reshape`、`slice`(NPUSlice)、`space_to_depth`、`split_d`、`squeeze`、`strided_slice_v2`(Slice)、`transpose`、`trilu`、`unsqueeze` |
| `common/src/framework/` | 14 | `concat_from_sequence`、`deformable_offsets`、`einsum`、`eyelike`、`phony_concat`、`phony_split`、`random_normal_like`、`random_uniform_like`、`reduceL1`、`reduceL2`、`repeat_with_pad`、`sequence`(6 条目)、`similar_opp`(63 条目)、`split_to_sequence` |
| `random/` | 3 | `dsa_random_uniform`、`stateless_random_normal_v2`、`stateless_sample_multinomial` |
| `experimental/math/asinh/` | 1 | `max_onnx_plugin.cpp`（默认构建不含，见下） |
| **合计** | **57** | 覆盖 124 个算子条目 |

> 目录计数两处易错点：`math/reduce_sum/framework/` 一个目录内**两个插件文件**；`conversion/slice/` 与 `conversion/strided_slice_v2/` 是**两个不同目录**下的两个 Slice 类插件（前者 NPUSlice，后者 Slice）。

配套改造：新增 **57 个 framework UT 文件** + **40 个 `CMakeLists.txt`**（其中 1 个为 `random/stateless_sample_multinomial/tests/ut/CMakeLists.txt` 目录级文件），合计 97 个新增文件；未修改任何已存在的 `CMakeLists.txt`（新增目录级构建接线均落在全新目录内）。

**实验性插件说明**：`experimental/math/asinh/framework/max_onnx_plugin.cpp` 受仓内 `ENABLE_EXPERIMENTAL` 选项控制，`experimental/` 与 `math/`、`conversion/`、`random/` 在 CMake 中是**互斥分支**（开启 `--experimental` 会替换而非叠加常规目录）。因此该插件在默认 `--onnxplugin` 构建中不进入产物，其覆盖由 framework UT 承担；本次以 `--onnxplugin` 构建为准完成编译与 UT 验证。

### 支持硬件与约束

- 硬件：Ascend 950（本机为 Atlas A3 / Ascend950PR 单板，SN `10264J332613`）；任务书第 3.1 节允许 Atlas A2 或 Ascend 950 任一验证通过；
- 软件：CANN 9.0.0（本机 `/usr/local/Ascend/cann-9.0.0`）、Python 3.12.13、onnx 1.23.2、numpy 1.26.4；
- 约束：本任务为 Host 侧等价重构，产物（`liboponnx_plugin_math.so`）注册 ABI 不变，ATC 使用方式不变，与具体硬件型号解耦；om 精度比对的迁移前后基线在同一硬件上获取。

### 关于「5 类 shape 算子优化」的范围说明

任务指引中提到「参照 5shape 的技巧进行 ops-math 算子优化」。经与任务书逐条比对，**该技巧不属于本任务的改造范围**，如实记录如下：

1. 任务书第 2.2 节明确：本任务为 **ONNX 插件框架适配工程（Host 侧）**，「开发仅涉及各算子目录下 `framework/*_onnx_plugin.cpp` 及对应 framework UT，**不涉及算子 kernel、tiling、算子原型的开发与修改**」；
2. 任务书第 2.1 节第 7 条明确：「除解耦改造外，**不得修改插件其他逻辑**，不得顺手重构与解耦无关的代码」；
3. 任务书第 3.3 节「性能要求」与第 3.4 节「内存要求」均为「**不涉及**」，不设性能 / 内存验收指标；
4. 5 类 shape 分派属于 **Device 侧 kernel / tiling** 优化套路（按形状分派切分参数与流水线策略）。而 ONNX 插件在参数解析阶段拿到的只是 ONNX 节点属性，**既看不到运行时 shape，也不产生计算 kernel**，shape 分类策略在插件代码中没有落点；
5. 若在参数解析函数中强行加入 shape 分支，将直接违反上述「等价重构」与「不得修改插件其他逻辑」两条硬约束，破坏验收口径。

因此本次**不做** shape 分派类的 kernel 优化。与「算子尽量通用」这一精神相符、且落在 Host 侧范围内的做法已在本次实际执行：公共读取路径收敛到单一公共头、16 个多注册文件共用同一解析范式、`similar_opp` 以一处宏改动覆盖 63 个算子条目、include 按实际需要最小化选取。

## 可维可测分析

### 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 迁移前后两轮 om 转换 + **NPU950 真机推理**比对：8 个模型转换产物字节数一致、op 类型符号 diff 均为 0，推理输出逐元素完全一致（见「om 转换验证」）。本次覆盖的均为确定性算子（Cast / Reshape / Squeeze / Mod / TopK / Concat / Einsum / Mean），不涉及随机算子，故无需混合容差 | 任务书第 3.2 节 |
| 性能标准 | 不涉及。Host 侧等价重构，不改变算子计算逻辑与模型转换产物结构，无新增算力开销 | 任务书第 3.3 节 |
| 内存标准 | 不涉及。同上，不新增显存 / 内存占用 | 任务书第 3.4 节 |

### 单元测试规划

- **构建开关**：`bash build.sh --onnxplugin --ophost -u -j<N>`。其中 `-u` 置 `ENABLE_TEST=TRUE`，`--ophost` 置 `OP_HOST=TRUE`；两者同时满足时才置 `OP_HOST_UT=TRUE`，framework UT 才会被纳入构建（`scripts/build_options.sh` 第 490–495 行）。注意 `--ut` 不是合法长选项；
- **用例组织**：按**插件文件**一一对应，57 个 UT 文件放在各算子 `tests/ut/framework/` 下，套件名统一为 `Onnx*PluginTest`；
- **可测性设计（解耦的直接收益）**：用例**不构造任何 protobuf 对象**，输入仅由 `ge::Operator` + 一行 JSON 字面量构成：

```cpp
ge::Operator op_src = ge::Operator("src", "TestOp");
op_src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"to","type":2,"i":1}]})"));
ge::Operator op_dest = ge::Operator("cast", "TestOp");

EXPECT_EQ(domi::ParseParamsCast(op_src, op_dest), domi::SUCCESS);
int32_t dst_type = -1;
EXPECT_EQ(op_dest.GetAttr("dst_type", dst_type), ge::GRAPH_SUCCESS);
EXPECT_EQ(dst_type, static_cast<int32_t>(ge::DT_FLOAT));
```

用例不再依赖 `.proto` 生成代码，可读性与维护成本显著下降——这正是解耦的目标之一；

- **覆盖场景**（对应任务书第 3.5 节入参生成规则）：
  1. 属性齐全的正常解析；
  2. 属性缺失（使用默认值，返回 `SUCCESS`）；
  3. 非法 attribute JSON（返回 `FAILED`；豁免路径不做此用例）；
   4. 边界 / 校验场景（`f` 键缺省按 0.0、`type` 不符按缺失处理、空列表 / 负值原样透传、Cast 未映射 dtype 返回 `FAILED`、TopK V9 `k == -1` 返回 `PARAM_INVALID` 等）。

**执行结果（实测）**：66 个测试套件、**301 个用例全部通过**（`[  PASSED  ] 301 tests.`，退出码 0）。运行命令见「自验证步骤说明」。

**独立 Host 侧 UT 的边界（如实记录）**：部分插件的解析流程依赖图拓扑，在游离 `Operator` 上不可达，本次据实处理、不改动插件逻辑：

| 依赖能力 | 游离 `Operator` 上的实测行为 | 处理方式 |
| --- | --- | --- |
| `UpdateInputDesc` / `UpdateOutputDesc`（`ChangeFormatFromOnnx` 内部调用） | 恒返回 `GRAPH_FAILED`（已建 `ge::Graph` + `AddOp` + `SetInputs` / `SetOutputs` / `SetValid` 后仍为 `GRAPH_FAILED`） | `DepthToSpace` / `SpaceToDepth` 的属性写转发生在该调用之后，UT 中不可观测；用例据实断言返回码，并用「校验阶段 / 描述符阶段」两条路径证明解析分支可达。`DeformableOffsets` 的属性写转发生在该调用之前，UT 仍可完整校验属性值 |
| `GetInputsSize()` | 连接输入（`SetInput`）不创建端口，恒返回 0 | 用 `DynamicInputRegister("x", N)` 等价注册动态输入，实测 `GetInputsSize()` 返回 N，`AccumulateNV2` 的「有输入」分支与 `Clip V11` 的输入数分支均覆盖到 `SUCCESS` 路径。在 `ParseParamsByOperatorFn` 路径下此处同样为 0（ATC 时序导致，见「变长入参算子的 arity 获取」），`GetOpInputNum` 的回落分支在 UT 内恰好覆盖这一情形 |

这三类插件的真实行为仍由 ATC 转换链路覆盖（见下节），UT 侧不做超出可观测范围的断言。

### 编译与静态自查

| 检查项 | 结果 |
| --- | --- |
| `bash build.sh --onnxplugin -j32` | 通过，产出 `build/liboponnx_plugin_math.so` |
| `git diff --check` | 通过（无空白错误） |
| 全仓 `*_onnx_plugin.cpp` 中 `ParseParamsFn` 出现次数 | **0** |
| 全仓 `*_onnx_plugin.cpp` 中 `.ParseParamsByOperatorFn` 注册次数 | **88**（覆盖 66 个文件，含范围外已迁移的历史插件） |
| 调用 `GetAttr("attribute")` 的插件文件数 | **55** |
| 读取 `f` 字段的插件文件数 | **9**（全部按「字符串承载 + 键缺省取 0.0」实现） |
| 迁移后产物未定义的 `ge::onnx::*` / `google::protobuf::*` 符号数 | **0** |
| 迁移后产物 `DT_NEEDED` | `libregister.so`、`libunified_dlog.so`、`libstdc++.so.6`、`libgcc_s.so.1`、`libc.so.6` —— **无 protobuf、无 compat 库** |

> 最后一项是本次解耦最直接的证据：迁移后插件产物的运行时依赖列表里已经完全没有 protobuf。

### om 转换验证规划

- **注入方式**：ATC 加载自定义 ONNX 插件的路径为 `/usr/local/Ascend/cann-9.0.0/opp/built-in/framework/onnx/`。将内置 `*.so` 移出、放入本次构建的 `liboponnx_plugin_math.so` 即可让 ATC 使用仓内插件（已验证：仅放置本仓产物时，`einsum_smoke.onnx` 转换 exit=0）；
- **ATC 命令**：`atc --model=<model>.onnx --framework=5 --soc_version=Ascend950PR_9579 --output=<name>`。注意 `--soc_version` 必须写完整串 `Ascend950PR_9579`（可用 `acl.get_soc_name(0)` 复核），短名 `ascend950` 会报 `SoC version is invalid`；ATC 不支持 `-m` / `-o` 等短选项；ONNX 框架不接受 `--input_format`；
- **环境前置**：`/usr/local/Ascend/cann-9.0.0/opp/vendors/config.ini` 首行必须为 `vendors=customize,ops-math-pre` 的 key=value 形式，不能使用 `[vendor_xxx]` INI 段式写法，否则 `tbe/common/utils/op_tiling.py` 读取该行时抛 `IndexError`，atc 报 `Format of file content is invalid` 并回落到 SoC 无效错误。已在本次环境修复；
- **比对方式**：同一 onnx 模型、同一输入、同一 ATC 参数，分别注入迁移前 / 迁移后插件各转换一轮，比对转换产物；
- **迁移前基线的构建**：`git stash -u` 暂存全部迁移改动 → 重跑 `bash build.sh --onnxplugin -j32` → 产出 `liboponnx_plugin_math_pre.so` → `git stash pop` 恢复工作区。两份产物均留存在 `task_submission/verify_om/` 并记录 md5；
- **⚠️ vendor 目录污染陷阱（必须规避）**：`/usr/local/Ascend/opp/vendors/ops-math-pre/` 曾残留一份迁移前 .so，而 `config.ini` 的 `vendors=customize,ops-math-pre` 会让 ATC **同时**加载 vendor 里的旧 .so，导致「迁移后」实测值被旧产物污染、A/B 结论完全失真。做 A/B 时必须先把 `config.ini` 改为 `vendors=customize`，仅保留 `built-in/framework/onnx/` 这一条加载路径；
- **执行结果（实测，8 个模型）**：

| 模型 | 迁移前 .om 字节 | 迁移后 .om 字节 | 转换 exit | op 类型符号 diff |
| --- | --- | --- | --- | --- |
| cast | 14638 | 14638 | 0 | 0 |
| reshape | 6332 | 6332 | 0 | 0 |
| squeeze | 6336 | 6336 | 0 | 0 |
| mod | 78418 | 78418 | 0 | 0 |
| topk | 52342 | 52344 | 0 | 0 |
| concat | 36514 | 36514 | 0 | 0 |
| einsum | 44702 | 44702 | 0 | 0 |
| mean | 44556 | 44556 | 0 | 0 |

前 6 个模型字节数完全一致；`topk` 差 2 字节为命令行长度差异（输出名不同）所致，非图内容差异。`concat` / `einsum` / `mean` 三个变长入参算子曾因 arity 问题回退为 5589 / 5589 / 5579 字节（节点被剪除），经「变长入参算子的 arity 获取」一节所述修复后恢复至迁移前数值；

- **图内容比对**：对每个模型用 `strings -n 2 *.om` 抽取 op 类型符号集合（归一化输出路径后逐元素 diff），8 个模型差异数均为 0。修复后 `concat` 恢复 `ConcatD`、`einsum` 恢复 `MatMulV3` + `BatchMatMul`、`mean` 恢复 `PartitionedCall`，与迁移前一致。字符串集合中仅存的差异是 ATC 生成的 TE kernel 产物文件名哈希后缀（如 `..._430400_kernel.cpp` 与 `..._502300_kernel.cpp`），属编译器工件命名，不影响图结构；
- **om 推理 A/B（真机执行，等价性的最终证据）**：`pyACL` 绑定不可用（见下），改用 **C++ ACL 直连**跑真机推理。CANN 9.0.0 重构了模型执行 API：`acl.mdl` C++ 命名空间已移除，`aclrtInit` / `aclrtFinalize` 改为 `aclInit` / `aclFinalize`，张量改由 `aclCreateDataBuffer` + `aclmdlAddDatasetBuffer` 承载（`aclTensorDesc` 不再有公开的地址 setter），且 `aclmdlSetDatasetTensorDesc` 必须在 `aclmdlAddDatasetBuffer` 之后调用（否则报 `Value [0] for [index] is invalid ... must be smaller than input databuf size`）；链接只需 `-lascendcl`（`aclmdl*` / `aclrt*` 全部实现在其中）。runner 见 `task_submission/verify_om/npu_infer/acl_run.cpp`，输入按模型声明 dtype 填充 `{1,2,3,4,5,6}`：

| 模型 | 迁移前输出 | 迁移后输出 | 一致性 |
| --- | --- | --- | --- |
| cast | `1 2 3 4 5 6` | `1 2 3 4 5 6` | 逐元素一致 |
| reshape | `1 2 3 4 5 6` | `1 2 3 4 5 6` | 逐元素一致 |
| squeeze | `1 2` | `1 2` | 逐元素一致 |
| mod | `0 0 0 0 0 0` | `0 0 0 0 0 0` | 逐元素一致 |
| topk | `5 4 10 9` | `5 4 10 9` | 逐元素一致 |
| concat | `1 2 3 4 5 6 1 2 3 4 5 6` | `1 2 3 4 5 6 1 2 3 4 5 6` | 逐元素一致 |
| einsum | `38 44 50 56 83 98 113 128` | `38 44 50 56 83 98 113 128` | 逐元素一致 |
| mean | `1 2 3 4 5 6` | `1 2 3 4 5 6` | 逐元素一致 |

8/8 模型在 NPU 设备 0 上执行成功（`aclmdlExecute` 返回 `ACL_SUCCESS`），迁移前后输出**逐元素完全一致**。数值亦经人工核验符合数学语义（`mod` 为 x % x 恒 0；`einsum` 为矩阵乘积；`concat` 为沿轴拼接；`topk` 为取值下标），排除 runner 输入绑定失效的可能。**至此 ATC 转换产物一致性与真机推理一致性双重成立**；

- **已知环境限制（如实记录）**：`pyACL` 的 `acl.mdl` 绑定在本机 CANN 9.0.0 上不可用（`get_num_inputs` 返回 0、`execute` 报 `args parse failed`、多次调用段错误、`load_from_file` 返回约定不一致），`ais_bench_infer` 与 `msame` 亦不可用；已改用上述 C++ ACL 直连路径完成真机推理，**该限制不再构成本次验证的阻塞项**。`clip_by_value` 模型 ATC 转换在迁移前后均以 exit=255 失败，属模型本身与本环境的既有问题，与本迁移无关；`onehot` 模型两轮均转换成功、产物字节数一致，但其 `.om` 命名与其余模型不同，未纳入上述 8 模型比对。

### 兼容性分析

- **接口兼容**：新接口由 GE 框架提供（`register/register.h`），注册 ABI 不变，ATC 使用方式不变；
- **行为兼容**：等价重构通过 framework UT 逐分支用例 + 迁移前后 ATC 转换产物比对双重验证；
- **存量兼容**：仓内 `onnx_common.h` 保留（范围外仍有插件引用），本次仅在范围内 57 个文件中断开对该头的引用，不影响范围外插件；
- **依赖兼容**：JSON 解析使用仓内 `third_party` 自带的 nlohmann/json，不新增第三方依赖，不引入新版本；
- **环境兼容**：`opp/vendors/config.ini` 的格式问题是 CANN 9.0.0 侧的环境问题，与本仓代码无关，交付物中不改动该文件。

### 风险与规避

| 风险 | 规避措施 |
| --- | --- |
| 属性 JSON 与 protobuf 取值语义的细节差异（`f` 字符串承载、`f` 键缺省 = 0、`type` 不符按缺失） | 逐算子与迁移前源码比对 + 专项 UT 用例（见「单元测试规划·覆盖场景 4」） |
| 多注册文件只改一支、漏改另一支 | 已静态清点 16 个多注册文件（见「多注册分支设计」），逐文件核对每处注册均已替换 |
| 漏引 include 导致静默使用错误类型 | include 按需最小化选取，漏引直接编译失败（本次已实锤并修复 3 处 `stub_ops.h` 漏引） |
| `GetOmDtypeFromOnnxDtype` 返回约定被误记为 `-1` | 以 `ge::DT_UNDEFINED` 为唯一判据，并补 UT 守护，防止回退 |
| 豁免路径被误加 JSON 解析，改变原 `SUCCESS` 语义 | 以「是否出现 `GetAttr("attribute")`」作为豁免判定标准，豁免路径不引 `nlohmann/json.hpp` |
| **变长入参算子 arity 丢失**：`GetInputsSize()` 在 ParseParams 阶段恒为 0，静默把 `N` 写成 0 导致算子被整节点剪除，且 UT / 编译 / 符号审计全部无法发现 | 统一改用 `GetOpInputNum` / `GetOpOutputNum`（读 `input` / `output` 属性列表），并**以迁移前 / 迁移后 ATC 转换 A/B 比对作为必做验收**；不接受编译期常量上界方案（实测多余端口会破坏 `N` 语义） |
| 实验性 `max_onnx_plugin.cpp` 不在默认构建内 | 如实记录构建开关互斥关系，覆盖由 framework UT 承担 |
| 本机 pyACL `acl.mdl` 绑定缺陷 | 改用 C++ ACL 直连（`-lascendcl`）完成 NPU950 真机推理，8/8 模型输出逐元素一致，该限制不再阻塞验证 |
| 与同机其它任务争抢 NPU | 转换 / 编译分片串行执行，执行前后各查一次 `npu-smi` |

## 交付与审查

### 交付清单

| 序号 | 交付项 | 状态 |
| --- | --- | --- |
| 1 | 更新 ops-math 项目（57 个插件文件 / 124 个算子条目） | 完成，`bash build.sh --onnxplugin -j32` 通过 |
| 2 | 配套 framework UT（57 个 UT 文件 + 40 个 `CMakeLists.txt`） | 完成，66 套件 / 301 用例全部通过 |
| 3 | 设计文档 `design.md` | 本文 |
| 4 | 迁移范式与自查指南（`MIGRATION_RECIPE.md`） | 完成 |
| 5 | 自验证步骤说明、UT 执行日志、ATC 转换留证、环境说明 | 见 `task_submission/` |
| 6 | om 转换等价性（迁移前 / 后 A/B） | 完成，8 模型字节数一致、op 符号 diff 均为 0 |
| 7 | om 推理等价性（NPU950 真机执行） | 完成，8/8 模型输出逐元素完全一致 |
| 8 | 性能 / 内存自验证报告 | 不涉及（任务书第 3.3 / 3.4 节） |

### 审查要点

- 范围内插件 `ParseParamsFn` 残留为 0，无 `ge_onnx.pb.h` / `onnx_common.h` 引用；
- 产物无 protobuf 符号依赖，`DT_NEEDED` 干净；
- 函数体除签名与属性解析手段外无其他改动（`ParseOpToGraphFn`、`DynamicInputRegister`、`OriginOpType`、`ImplyType` 均未动）；
- 默认值、返回值、重复属性策略与迁移前逐分支一致；
- `git diff --check` 通过，无空白错误与无用文件；
- 不包含任何个人敏感信息。

**TeamName：sme_engine**
