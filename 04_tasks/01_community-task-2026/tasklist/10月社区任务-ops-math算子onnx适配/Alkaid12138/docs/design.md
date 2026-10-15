# ops-math ONNX 插件 protobuf 解耦迁移设计文档

> 任务：2026 年 10 月 CANN 社区任务《ops-math 算子 onnx 插件 protobuf 解耦迁移适配》
> 目标仓库：https://gitcode.com/cann/ops-math
> 设计基线：ops-math master（任务书静态扫描基线 commit `91c6c1715`；开发分支基于最新 master `b7a0fd17f`）

---

## 1. 需求背景（required）

### 1.1 需求来源

本任务来自 2026 年 10 月 CANN 社区任务，任务书为《ops-math 算子 onnx 插件 protobuf 解耦任务书》（`ops-math_onnx_task_doc.md`）。目标是在 `ops-math` 开源仓中，将仍在通过 `ParseParamsFn` 注册参数解析函数的 **124 个 ONNX 算子插件条目（57 个插件文件）** 迁移至不依赖 protobuf 的 `ParseParamsByOperatorFn` 接口，使插件代码彻底摆脱对 protobuf 生成代码（`ge_onnx.pb.h`）与 protobuf 运行库的依赖，同时**保持插件原有解析语义与功能不变**（等价重构）。

设计依据如下：

- 任务书：`ops-math_onnx_task_doc.md`（迁移范围表、接口定义、验收标准、交付件要求）；
- 验证用例包：`test_cases/`（57 个 framework UT 文件 / 624 个用例、56 个 om 模型生成脚本、两轮比对工具链）；
- 仓内已合入的迁移参考实现：
  - `common/src/framework/reverse_sequence_onnx_plugin.cpp`（纯解析函数迁移样板，MR !5201）；
  - `conversion/strided_slice/framework/npu_indexing_onnx_plugin.cpp`（含 `ParseOpToGraphFn` 的迁移样板）；
  - `random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp`；
- protobuf-free 公共工具头：`common/inc/framework/math_onnx_plugin_util.h`；
- 迁移适配 issue（含关联 PR 示例）：https://gitcode.com/cann/ops-math/issues/2892 ；
- ONNX 算子开发说明（CANN 官方文档）：ai_framework_adaptation/onnx_framework/call_examples.md。

### 1.2 背景介绍

#### 1.2.1 现状：插件参数解析与 protobuf 强耦合

`ops-math` 仓的 ONNX 适配插件通过 `REGISTER_CUSTOM_OP(...).ParseParamsFn(ParseParamsXxx)` 注册参数解析函数，该函数签名依赖 `google::protobuf::Message`（框架实际传入 `ge::onnx::NodeProto`）。插件内部需遍历 protobuf attribute 数组，并按 `attr.i()` / `attr.f()` / `attr.s()` / `attr.ints()` 等字段逐项取值，再写入目标算子属性。典型形态（迁移前 `math/cast/framework/cast_onnx_plugin.cpp`）：

```cpp
static Status ParseParamsCast(const Message* op_src, ge::Operator& op_dst)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) { /* ... */ return FAILED; }
    for (auto attr : node->attribute()) {
        if (attr.name() == "to") {
            int onnx = attr.i();
            int32_t om = GetOmDtypeFromOnnxDtype(onnx);
            if (om == -1) { /* 永不成立的失效判据，见 3.2.6 */ }
            op_dst.SetAttr("dst_type", om);
            break;
        }
    }
}
```

#### 1.2.2 三类问题

1. **与 protobuf 生成代码强耦合**：插件源码依赖 `ge_onnx.pb.h`，proto 定义变动会波及全部插件；
2. **编译依赖重**：需要拉入 protobuf 头文件与生成代码，编译链路长；
3. **UT 成本高**：单测必须构造 protobuf 对象，用例编写与维护成本高。

#### 1.2.3 迁移目标与范围

将参数解析函数由 `ParseParamsFn`（protobuf 入参）替换为 `ParseParamsByOperatorFn`（`ge::Operator` 入参）；属性解析由 protobuf 遍历改为 `op_src.GetAttr("attribute")` 取 JSON 字符串后用 **nlohmann/json** 解析。迁移范围为对仓内源码静态扫描出的、仍在使用 `ParseParamsFn` 的全部插件：**57 个文件 × 124 个算子条目**（同一算子的多个 ONNX opset 版本合并计一条）。经本次开发前对最新 master 的 `git grep -l ParseParamsFn` 复核，文件集合与任务书迁移范围表**逐条一致**。

#### 1.2.4 参考实现与公共工具

仓内已存在 protobuf-free 公共工具头 `common/inc/framework/math_onnx_plugin_util.h`，提供 `GetOpName`、`GetOmDtypeFromOnnxDtype`、`Vec2Tensor`、`CreateScalar`、`ChangeFormatFromOnnx`、`DataTypeOnnx` 枚举等；该头不含任何 protobuf 依赖，正是为本次解耦准备。`common/inc/framework/onnx_common.h` 目前唯一残留作用就是 `#include "onnx/proto/ge_onnx.pb.h"`，故**迁移后的插件不再 include `onnx_common.h`**：纯解析插件引用 `math_onnx_plugin_util.h` 等；含 `ParseOpToGraphFn`（拆图函数）的插件按已合入先例 `npu_indexing_onnx_plugin.cpp` 直接引用 `stub_ops.h` + `graph/graph.h`。

### 1.3 接口与语义基线

迁移后参数解析函数接口（保持仓内现有函数命名，如 `ParseParamsCast`、`ParseParamsFlatten`、similar_opp 的宏命名 `ParseParams<FUN>`）：

```cpp
domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest);
```

注册接口：

```cpp
REGISTER_CUSTOM_OP("${OpName}")
    .FrameworkType(ONNX)
    .OriginOpType({...})                     // opset 覆盖逐条保持不变
    .ParseParamsByOperatorFn(ParseParams${OpName})
    .ImplyType(ImplyType::TVM);
```

**语义基线**：迁移前各插件对同一 ONNX 节点的解析结果（写入 `op_dest` 的属性序列）必须逐项一致，仅解析实现方式变化。唯一例外见 3.2.6。

---

## 2. 需求分析（required）

### 2.1 需求描述

以等价重构方式完成 57 个 ONNX 插件文件的解耦迁移：替换解析函数签名、注册方式、属性解析方式、节点名获取方式；移除对 `ge_onnx.pb.h` 及 protobuf 运行库的依赖；各分支返回值（`SUCCESS`/`FAILED`/`PARAM_INVALID`）与迁移前保持一致；不修改插件其他逻辑，不做与解耦无关的重构。同时为每个迁移插件补齐 framework UT。

### 2.2 外部组件依赖

| 组件 | 用途 | 版本/来源 |
|---|---|---|
| nlohmann/json | attribute JSON 解析 | **仓内 third_party 自带版本**（不另引新版本） |
| `math_onnx_plugin_util.h` | protobuf-free 公共工具（`GetOpName`/`GetOmDtypeFromOnnxDtype`/`Vec2Tensor`/`CreateScalar`/`ChangeFormatFromOnnx`） | 仓内 `common/inc/framework/` |
| `stub_ops.h` / `graph/graph.h` | 仅含 `ParseOpToGraphFn` 的插件（拆图函数）引用 | 仓内 |
| CANN Toolkit | ATC 转换（`--framework=5`）、pyACL 推理 | > 9.0.0（实测 CANN 9.1.0） |
| Python 3.8+ / onnx / onnxruntime / numpy | om 验证 golden 生成 | 实测 3.11 / 1.23.2 / 1.31.0 / 2.4.6 |

### 2.3 迁移范围清单（57 文件 / 124 条目）

按目录归集：

| 目录大类 | 文件数 | 代表算子 | 特征 |
|---|---|---|---|
| `common/src/framework/` | 14 | similar_opp（63 算子）、sequence（6 算子）、eyelike、einsum、reduceL1/L2、phony_concat/split、repeat_with_pad、random_*_like、concat_from_sequence、split_to_sequence、deformable_offsets | 含宏批量注册与多算子聚合文件；部分含 `ParseOpToGraphFn` |
| `conversion/*/framework/` | 14 | concat、flatten、squeeze、transpose、depth_to_space、space_to_depth、trilu、clip、reshape、pad、split、unsqueeze、slice、npu_slice | 多含双注册分支（opset 分界）与拆图函数 |
| `math/*/framework/` | 26 | cast、addcmul、xor、cumsum、isinf、max/min、mod、onehot、bitshift、sign_bits_pack/unpack、topk、argmax/argmin、reduce_* 系列、mean/sum | 单/少量属性解析为主；reduce 系列写张量属性 |
| `random/*/framework/` | 3 | random_uniform（dsa）、random_normal（stateless）、multinomial | 随机算子，含 seed/dtype 属性 |
| `experimental/math/asinh/framework/` | 1 | Max（实验路径） | 与 `math/maximum` 共用一份 om 脚本 |

（124 个算子条目的逐行对照见任务书第 1 节迁移范围表，本次以该表为唯一范围基准。）

### 2.4 需求拆解

1. **接口迁移**：`ParseParamsFn` → `ParseParamsByOperatorFn`；签名 → `(const ge::Operator&, ge::Operator&)`；删除 `dynamic_cast/reinterpret_cast<NodeProto*>` 与 protobuf 判空分支；
2. **属性解析迁移**：`GetAttr("attribute")` 取 JSON → nlohmann/json 解析 → 逐属性按原语义写 `op_dest`；
3. **节点名迁移**：原 `node->name()` → `GetOpName(op_src)`（仅原本使用节点名的文件）；
4. **依赖清理**：删除 `onnx_common.h`（含 `ge_onnx.pb.h`）依赖，纯解析插件引用 `math_onnx_plugin_util.h` 等；含拆图函数的插件引用 `stub_ops.h` + `graph/graph.h`；
5. **UT 补齐**：57 个 framework UT 文件就位并纳入 CI 编译；
6. **验证**：624 个 UT 通过 + 56 个脚本迁移前后两轮 om 逐位比对一致。

---

## 3. 详细设计（required）

### 3.1 插件分析：attribute JSON 契约

迁移后解析函数通过 `op_src.GetAttr("attribute")` 拿到 JSON 字符串，键名与 ONNX `AttributeProto` 字段同名：

| 属性形态 | type 值 | JSON 示例 | 读取方式 |
|---|---|---|---|
| 标量整数 INT | 2 | `{"name":"to","type":2,"i":10}` | `attr["i"].get<int>()` |
| 标量浮点 FLOAT | 1 | `{"name":"alpha","type":1,"f":"0.25"}`（**值以字符串承载**） | 先取 `std::string` 再 `atof` 转 float |
| 字符串 STRING | 3 | `{"name":"mode","type":3,"s":"NCHW"}` | `attr.value("s", "")` |
| 整数列表 INTS | 7 | `{"name":"axes","type":7,"ints":[1,2]}` | `attr["ints"]` 数组遍历 |
| 浮点列表 FLOATS | 6 | `{"name":"floats","type":6,"floats":[0.1,0.2]}` | `attr["floats"]` 数组遍历 |

整体格式：`{"attribute":[ <属性对象>, ... ]}`。该契约与仓内已迁移参考实现（elu / leaky_relu / bounding_box_decode / bernoulli）的读法及其配套 UT 的输入表示交叉核对一致。

**浮点标量的两点约定**：① 标量浮点属性以**字符串**承载，须按字符串读取后转 float；② 属性存在但 `f` 键缺省（proto3 省略值为 0 的标量）时取值为 **0.0**，**不得回落到算子默认值**（例：`alpha` 属性存在、无 `f` 键 → alpha=0）；同理标量整数 `i` 键缺省取 0。

### 3.2 实现方案

#### 3.2.1 统一代码骨架

对每个解析函数采用如下骨架（对照 `reverse_sequence_onnx_plugin.cpp`）：

```cpp
static Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // 1.（如有）DynamicInputRegister/DynamicOutputRegister/original_type：保持原样与原时序
    // 2.（如有）默认值先行设置（与迁移前默认值路径一致）
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) != ge::GRAPH_SUCCESS) {
        return SUCCESS;   // 无 attribute 键：走默认值路径，返回值按原语义
    }
    try {
        json attrs = json::parse(attrs_string.GetString());
        if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
            for (const json& attr : attrs["attribute"]) {
                const std::string name = attr.value("name", "");
                // 按 name 分发，严格保持迁移前语义
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }
    return SUCCESS;
}
```

要点：无 attribute 键时的返回值、以及"缺某属性即失败"的算子（如 Cast 缺 `to` → FAILED），按迁移前语义处理，不机械套用 SUCCESS。

#### 3.2.2 include 替换

纯解析插件的 include 集：

```cpp
#include "graph/operator.h"          // ge::Operator
#include "math_onnx_plugin_util.h"   // 公共工具
#include "log/log.h"                 // OP_LOGE/OP_LOGW
#include "register/register.h"       // REGISTER_CUSTOM_OP
#include "nlohmann/json.hpp"         // 属性 JSON
```

含 `ParseOpToGraphFn` 的插件（对照 `npu_indexing_onnx_plugin.cpp`）：

```cpp
#include "graph/operator.h"
#include "graph/graph.h"
#include "stub_ops.h"                // op::Data / op::Const / op::XXX 等拆图算子
#include "<算子>_proto.h"            // 拆图函数用到的算子原型头（按需）
#include "math_onnx_plugin_util.h"
#include "register/register.h"
#include "nlohmann/json.hpp"
#include "log/log.h"
```

**统一删除 `onnx_common.h`**（其唯一作用是拉入 `ge_onnx.pb.h`）。迁移后全部 57 个文件均不再引用 protobuf 类型。

#### 3.2.3 属性读取与语义保持规则

1. 整数标量：`attr.contains("i") ? attr["i"].get<int64_t>() : 0`（键缺省取 0，不回落默认值）；
2. 浮点标量：`std::string s = attr["f"]; float v = atof(s.c_str());`（键缺省取 0.0，不回落默认值）；
3. 字符串：`attr.value("s", "")`（`s` 键缺省取空串，不回落默认值）；
4. 列表：`ints`/`floats` 为 JSON 数组，键缺省/空数组按空列表透传；
5. **type 校验**：是否校验、校验失败按缺失还是报错，**逐文件照抄迁移前行为**（多数算子按缺失/忽略；`BitShift` 为严格白名单——出现任何非 `direction`（或 type 非 STRING）属性立即 FAILED；`EyeLike` 不校验 type）；
6. **多个同名属性**：保持迁移前 `break`（取首个）或累加语义；
7. **无关属性**：一律忽略；
8. **属性缺失**：保持原有默认值行为并返回 SUCCESS；**属性非法**（JSON 解析异常或未知异常）：记日志并返回 FAILED；各分支返回值与迁移前一致。

#### 3.2.4 节点名获取替换

原通过 protobuf `node->name()` 获取节点名的 35 个文件，统一改为 `GetOpName(op_src)`（`math_onnx_plugin_util.h`）；原本未使用节点名的文件不新增该行为。原 `node->input_size()` / `node->output_size()` 相应改为 `op_src.GetInputsSize()` / `op_src.GetOutputsSize()`。

#### 3.2.5 特殊机制保持

- **动态输入/输出注册**（`DynamicInputRegister`/`DynamicOutputRegister`）与 `SetAttr("original_type", ...)`：**保持原样与原时序**（包括失败分支前的副作用顺序）；
- **`ParseOpToGraphFn`（子图拆图函数）**：**不在本次迁移范围**，整体保持原样，仅其所在文件的 include 从 `onnx_common.h` 换为 `stub_ops.h` + `graph/graph.h`；
- **双注册分支**（如 `Slice` 的 opset 8/9 属性解析 vs 10~18 恒 SUCCESS、`ReduceMean` 的 opset 8~12 vs 13~18）：两个解析函数分别迁移，分支语义（属性累加、`noop_with_empty_axes` 告警/透传、`input_size` 等）逐条保持；
- **张量属性写入**（如 `ArgMax` 的 `dimension`、reduce 系列的 `axes`）：用 `Vec2Tensor`/`CreateScalar`，dims/dtype 与迁移前一致。

#### 3.2.6 例外：Cast 失效判据修正（唯一允许的行为变化）

`math/cast/framework/cast_onnx_plugin.cpp` 迁移前以 `if (om == -1)` 判断 dtype 映射失败，而 `GetOmDtypeFromOnnxDtype()` 对未命中的 dtype 返回 `ge::DT_UNDEFINED` 而非 -1，该分支**永不触发**——不支持的 dtype（如 `to=20`）迁移前要到引擎选择阶段才失败（`Failed to select engine`）。迁移后必须改判：

```cpp
const int32_t om = static_cast<int32_t>(GetOmDtypeFromOnnxDtype(onnx));
if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) {
    OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
    return FAILED;
}
```

命中即记日志并返回 FAILED 且不写 `dst_type`。该修正不影响等价性判据：非法模型两轮都转换失败，合法模型行为不变。其余若发现同类"判据与工具函数真实返回约定不符"的失效判据，按同一原则处理并在自查表中说明。

#### 3.2.7 文件分组与工程实施

| 组 | 文件数 | 迁移要点 |
|---|---|---|
| A. 宏批量注册 | 1（similar_opp，63 算子 / 578 注册项） | 宏体改为 `(const ge::Operator&, ge::Operator&)` 直接 `return SUCCESS;`；**不读属性、不写属性、非法 JSON 也返回 SUCCESS**（新增 FAILED 会改变可观测语义）；注册表 / FUN 后缀 / ATC 算子名映射逐条不变 |
| B. 常规属性解析 | 47 | 按 3.2.1~3.2.4 统一模式逐文件迁移 |
| C. 聚合多算子文件 | 2 | 逐解析函数迁移，函数名与注册关系不变 |
| D. 含特殊机制 | 7 | 动态注册与拆图函数保持原样（3.2.5） |

工程约束：函数嵌套深度 ≤5（CodeCheck），必要时按 `reverse_sequence` 的做法抽 `ApplyXxxAttr` / `ParseXxxAttrs` 静态辅助函数；单文件改动量控制在迁移必需范围，不做无关重构。

### 3.3 支持硬件

本任务为 **Host 侧框架适配改造**，插件代码与硬件无关。om 转换与推理验证需 Atlas A2 或 Ascend 950 任一（推荐 A2）；CANN 版本 > 9.0.0。

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A3 系列产品 | √（推荐，实测 Ascend910B3 + CANN 9.1.0） |
| Ascend 950 | √（任一即可） |

### 3.4 算子（插件）约束限制

- 迁移为**等价重构**，除 3.2.6 的 Cast 失效判据修正外，不得改变任何插件行为；
- 不涉及 kernel / tiling / 算子原型（op_host / op_kernel / op_api）开发与修改；
- TensorFlow 插件（`*_tf_plugin.cpp`）与已迁移插件不在范围内；
- `ParseOpToGraphFn` 拆图函数逻辑不变；
- nlohmann/json 使用仓内 third_party 自带版本，不引入新版本；不得新增公共头文件（现有工具已覆盖全部需求）。

---

## 4. 可维可测分析

### 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 主判据（等价性） | 同一 onnx 模型、同一输入，迁移前/后插件各自转换 om 并推理，输出**逐数值完全相等**（不设 rtol/atol） | 任务书 3.2 节 |
| 辅助判据（精度） | om 输出与 golden（onnxruntime 或脚本内 numpy 参考实现）按生态**混合容差标准**比对：fp32 rtol=2^-10、atol=2^-16，逐元素硬上限 max(fixed_limit, 32×ULP)，整数/布尔逐位精确，整体匹配率 ≥0.99；并校验输出 dtype/shape | 混合容差标准（opbase） |
| 性能标准 | 不涉及（Host 侧等价重构，不改计算逻辑与转换产物结构） | 任务书 3.3 节 |
| 内存标准 | 不涉及 | 任务书 3.4 节 |

### 4.2 测试用例规划

**（1）framework UT（行为契约）**：任务配套 57 个 UT 文件、624 个用例，放入各算子 `tests/ut/framework/`（`common/src/framework/` 下插件放 `common/tests/ut/framework/`），直接 include 迁移后插件源文件、以 `ge::Operator` + `SetAttr("attribute", <JSON串>)` 构造输入，**不构造 protobuf 对象**。每算子固定覆盖四类场景：

| 场景分类 | 用例描述 | 预期 |
|---|---|---|
| 正常解析 | 属性齐全，`op_dest` 各属性值与迁移前语义一致 | SUCCESS + 属性值匹配 |
| 默认值 | 属性缺失时返回 SUCCESS 并使用默认值（如 Flatten `axis` 默认 1、Squeeze `axes` 默认空列表） | SUCCESS + 默认值 |
| 失败分支 | 非法 attribute JSON 返回 FAILED；原实现中缺属性即失败的（如 Cast 缺 `to`）仍 FAILED | FAILED |
| 边界/校验 | type 与声明不符按缺失处理、`i`/`f` 键缺省取 0、空数组/负值原样透传、不支持 dtype 等 | 按契约 |

构建与运行：开启仓内 `OP_HOST_UT` 开关构建，`--gtest_filter='Onnx*PluginTest.*'`，标准为 **624 个用例全部通过**。

**（2）om 转换精度验证**：任务配套 56 个模型生成脚本（覆盖 57 个插件文件，两份 Max 共用一份脚本）。每个算子三步：生成 onnx 模型 + 随机输入 + golden → ATC 转换（`--soc_version` 按硬件）→ pyACL 推理并按混合容差标准比对。随后执行**迁移前后两轮比对**（主判据）：

- 一键脚本 `two_round_compare.sh`（`--pre`/`--post` 指定两套插件 `.so`，自动经 `ASCEND_CUSTOM_OPP_PATH` 注入 ATC 并抽取插件加载证据）；
- 插件生效证据：日志中 `[Register][Check] Plugin of <算子> already registered, skip`，随验证记录归档；
- 三类用例判据：常规用例两轮转换成功且输出逐位一致；预期失败用例两轮均在转换期失败且报错一致；无 golden 用例判据退化为两轮行为一致。

### 4.3 兼容性分析

- **注册兼容**：`OriginOpType` 的 opset 覆盖、`ImplyType::TVM`、动态输入注册逐条保留，ONNX 模型到内部算子的映射关系不变；
- **产物兼容**：om 转换产物结构与推理结果不变（等价重构），下游无感知；
- **编译兼容**：插件不再依赖 protobuf 头与生成代码，编译链路缩短；nlohmann/json 用仓内版本，无新增三方依赖；
- **对旧接口无破坏**：仅涉及插件内部参数解析实现，不改变对外接口。

### 4.4 风险与降级预案

| 风险 | 降级预案 |
|---|---|
| 迁移前（旧接口）插件在部分 CANN 版本加载报 `undefined symbol: ge::onnx::AttributeProto` | 属 CANN 版本与旧代码适配问题（与迁移无关），按任务包 `verify_om/compat/README.md` 生成兼容库；**迁移后插件不依赖 protobuf，不需该兼容库** |
| 仓内 abseil 内联命名空间与 CANN 运行库不一致（`undefined symbol ...absl...lts_...`） | 改 `third_party/lib_cache/abseil-cpp/absl/base/options.h` 的 `ABSL_OPTION_INLINE_NAMESPACE_NAME` 为 CANN 实测值后重编 |
| 个别算子在部分 SOC 上因 CANN op-info 无条目无法 ATC 转换 | 记录平台/算子/报错原文，该平台此项以"迁移前后两轮行为一致"（均失败且报错一致）为准，不影响其余算子验收 |
| 属性 JSON 实际形态与契约不符 | 迁移后临时打印真实 JSON（或 ATC debug 日志）校准输入表示，UT 期望值不变 |
| 本地无 NPU 环境 | 先完成设计与代码静态验证（结构、注册关系、回归 grep），运行验证在具备 CANN + NPU 的环境执行 |

---

## 附录：修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
|---|---|---|---|
| 2026-10-14 | v1.0.0 | 初稿：明确迁移范围（57 文件/124 条目）、接口替换方案、attribute JSON 契约、语义保持规则、Cast 失效判据修正、UT 与 om 两轮验证方案 | gcw_MIr7oWnV |
