# 需求背景（required）

## 需求来源

10 月社区任务：ops-math 仓 124 个算子 onnx 插件条目的 protobuf 解耦迁移适配（Host 侧框架适配，等价重构）。

- 任务书：`ops-math_onnx_task_doc.md`（本文唯一语义真值源；配套 `test_cases/README.md` 若与任务书冲突，以任务书为准）
- 代码合入仓：`https://gitcode.com/cann/ops-math`（已 fork 至 `https://gitcode.com/cool_cy/ops-math`，迁移改造与 PR 提交均基于该仓）
- 迁移方式参考：ops-math 仓 issue `https://gitcode.com/cann/ops-math/issues/2892`（关联 PR：fix: 迁移onnx插件ParseParams API并补充framework层UT用例、feat(onnx): 迁移 ReverseSequence 插件至 ParseParamsByOperatorFn 并补单测、refactor: npu_indexing onnx 插件迁移至 ParseParamsByOperatorFn 接口）
- 设计文档提交仓：`cann-ops-competitions`（`https://gitcode.com/cann/cann-ops-competitions`）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/10月社区任务-ops-math算子onnx适配/cool_cy/docs/design.md`
- 验证用例包：任务书配套 `test_cases/`（57 个 framework UT 文件共 624 个用例、56 个 om 模型生成脚本、`verify_om/` 两轮比对工具），随任务书下发、全量覆盖迁移范围

## 背景介绍

### ops-math onnx 插件 protobuf 解耦

ops-math 仓现有 124 个 onnx 算子插件条目（文件 × 算子，57 个插件文件）通过 `ParseParamsFn` 注册参数解析函数。该函数签名为 `domi::Status (*)(const Message* op_src, ge::Operator& op_dst)`，入参是 protobuf `Message` 基类（实际传入 `ge::onnx::NodeProto`），插件内部先 `reinterpret_cast`/`dynamic_cast` 回 `NodeProto`，再遍历 protobuf `attribute()` 并按 `i()/f()/s()` 等字段取值。以 `conversion/flatten/framework/flatten_onnx_plugin.cpp` 为例（迁移前）：

```cpp
#include "onnx_common.h"                       // 内部第 30 行: #include "onnx/proto/ge_onnx.pb.h"

static Status ParseParamsFlatten(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) { ... return FAILED; }
    int axis = 1;
    for (auto attr : node->attribute()) {       // 遍历 protobuf attribute
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis = attr.i();
        }
    }
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Flatten") ...
  .ParseParamsFn(ParseParamsFlatten)            // 旧注册接口
  .ImplyType(ImplyType::TVM);
```

protobuf 字段遍历基于反射，`attr.type()`/`attr.i()` 等均依赖 protobuf 生成代码，由此带来三类问题（任务书第 1 节）：

1. 与 protobuf 生成代码强耦合：插件源码依赖 `ge_onnx.pb.h`，proto 定义变动会波及全部插件；
2. 编译依赖重：需要拉入 protobuf 头文件与生成代码，编译链路长；
3. UT 成本高：单测必须构造 protobuf 对象，用例编写与维护成本高。

本任务要求将上述 124 个条目的参数解析函数由 `ParseParamsFn` 替换为 `ParseParamsByOperatorFn`，属性解析由 protobuf 遍历改为 `op_src.GetAttr("attribute")` 获取 JSON 字符串后用 nlohmann/json 解析，使插件代码彻底摆脱对 protobuf 生成代码的依赖，同时保持插件原有解析语义与功能不变。

### 同类改造现状分析（基于 ops-math master，commit `11aa1c75` 源码实测）

**仓内已有 9 个插件文件完成同型迁移**，是本任务的在库先例：

| 已迁移文件 | 迁移后形态 |
| --- | --- |
| `common/src/framework/reverse_sequence_onnx_plugin.cpp` | `ParseParamsByOperatorFn` + nlohmann/json + 配套 framework UT |
| `common/src/framework/correlation_onnx_plugin.cpp` | 同上 |
| `common/src/framework/npu_rotary_mul_onnx_plugin.cpp` | 无属性插件，仅注册宏替换 |
| `conversion/npu_format_cast/framework/npu_format_cast_onnx_plugin.cpp` | 同上 |
| `conversion/strided_slice/framework/npu_indexing_onnx_plugin.cpp` | 含 JSON 解析 + UT |
| `math/cast/framework/npu_dtype_onnx_plugin.cpp` | 同上 |
| `math/one_hot/framework/one_hot_d_onnx_plugin.cpp` | 同上 |
| `math/shape/framework/shape_onnx_plugin.cpp` | 早期迁移（仍 include `onnx_common.h`，过渡形态） |
| `random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp` | 同上（过渡形态） |

两个关键结论：

1. **完全解耦的 include 模式以 `reverse_sequence`/`correlation`/`npu_indexing`/`one_hot_d` 为准**：只包含 `graph/operator.h`、`math_onnx_plugin_util.h`、`log/log.h`、`register/register.h`、`nlohmann/json.hpp` 及算子原型头（如 `cast_proto.h`，属 GE 算子原型注册头，与 google protobuf 无关），**不 include `onnx_common.h`**。`shape`/`bernoulli` 仍带 `onnx_common.h` 属历史过渡形态，本次不效仿。
2. **公共工具已就绪且声明了约束**：`common/inc/framework/math_onnx_plugin_util.h`（protobuf-free）提供 `GetOpName`、`Vec2Tensor`、`CreateScalar`、`DataTypeOnnx` 枚举、`GetOmDtypeFromOnnxDtype`（未命中返回 `ge::DT_UNDEFINED`）、`ChangeFormatFromOnnx`；其文件头注释明确"与 protobuf 解耦（ParseParamsByOperatorFn）的插件不得 include onnx_common.h（其拉入 ge_onnx.pb.h）"。proto 定义变动波及面已被该公共头收敛。

**framework UT 基建已就绪**：`cmake/ut.cmake` 的 `add_framework_ut_modules`/`add_modules_ut_sources` 按目录 GLOB `test_*_onnx_plugin.cpp` 自动收集（算子独立目录 `<prefix>/<op>/tests/ut/framework/` 与 common 集中目录 `common/tests/ut/framework/` 两种放置均支持，`--ops` 模式按目录算子名/文件名过滤），UT 产物为 `math_op_host_ut`；`common/tests/ut/framework/` 下已有 `test_reverse_sequence_onnx_plugin.cpp` 等 3 个参考 UT（gtest，直接 include 插件源文件、用 `ge::Operator` + `SetAttr("attribute", JSON)` 构造输入）。

### 迁移范围分析

对 master 源码静态扫描，仍使用 `.ParseParamsFn(` 的插件共 **57 个文件、124 个条目**（与任务书第 1 节迁移范围表一致），目录分布：

| 目录 | 文件数 | 说明 |
| --- | --- | --- |
| `common/src/framework/` | 14 | 含 `similar_opp_onnx_plugin.cpp`（单文件 63 个条目，宏批量注册）与 sequence 系列（单文件 6 算子） |
| `conversion/*`（14 个算子目录） | 14 | Concat/Flatten/Pad/Slice/Split/Transpose 等，多为带属性解析 |
| `math/*`（23 个算子目录） | 25 | Reduce 系列/Cast/TopK/ArgMax 等；`reduce_sum`、`accumulate_nv2` 各 2 文件 |
| `experimental/math/asinh/` | 1 | `max_onnx_plugin.cpp` |
| `random/*` | 3 | RandomUniform/RandomNormal/Multinomial |

特征分布（影响迁移分类处理）：

- **37 个文件**含 `DynamicInputRegister`/`DynamicOutputRegister` 动态输入输出注册——迁移时保持对应逻辑不变（任务书第 7 节第 4 条）；
- **33 个文件**注册了 `ParseOpToGraphFn`（PartitionedCall 子图拆图函数）——拆图函数**不在本次迁移范围**，仅迁移参数解析函数（任务书 2.2 节第 3 条）；
- 其余为无属性或简单属性的普通插件。

# 需求分析（required）

## 需求描述

对 ops-math 仓 124 个 onnx 插件条目（57 个文件）完成 protobuf 解耦等价重构：解析函数签名替换为 `ge::Operator` 入参、注册方式替换为 `ParseParamsByOperatorFn`、属性解析替换为 `GetAttr("attribute")` + nlohmann/json、移除 `ge_onnx.pb.h`/protobuf 运行库依赖，节点名获取改为 `op_dest.SetAttr("name", GetOpName(op_dest))`；迁移前后解析行为完全一致（唯一例外：Cast 不支持 dtype 的失效判据须修正，见详细设计）。仅涉及各算子目录 `framework/*_onnx_plugin.cpp` 及对应 framework UT，不涉及 kernel、tiling、算子原型。

迁移后接口定义（任务书 2.3 节，函数命名保持仓内现有习惯，如 `ParseParamsCast`、`ParseParamsFlatten`）：

```cpp
domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest);

REGISTER_CUSTOM_OP("${OpName}")
    .FrameworkType(ONNX)
    .OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParams${OpName})
    .ImplyType(ImplyType::TVM);
```

## 需求拆解

1. **签名替换**：`static Status ParseParams${OpName}(const Message*, ge::Operator&)` → `static Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest)`；删除函数开头的 `dynamic_cast`/`reinterpret_cast` 判空分支（`ge::Operator` 入参无此失败模式，该分支在旧代码中本身也不构成可触发路径）。
2. **注册替换**：`.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`；`REGISTER_CUSTOM_OP` 的 `FrameworkType(ONNX)`/`OriginOpType`/`ImplyType` 保持不变。
3. **属性解析替换**：protobuf 遍历 → `op_src.GetAttr("attribute", attrs_string)` 取 JSON 串 → `nlohmann::json::parse` → 遍历 `attrs["attribute"]` 数组按 `name` 分派；非法 JSON 记录日志并返回 FAILED，属性缺失保持原默认值并返回 SUCCESS，属性名匹配但 `type` 值不符按缺失处理。
4. **include 替换**：删除 `onnx_common.h`（protobuf 入口），替换为 `operator.h` + `nlohmann/json.hpp` + `math_onnx_plugin_util.h` + `register/register.h`；确需新增公共头时使用仓前缀命名（沿用 `math_` 前缀）避免与其他算子仓重名。
5. **节点名获取替换**：原 protobuf `node->name()` → `op_dest.SetAttr("name", GetOpName(op_dest))`（`math_onnx_plugin_util.h` 已有 `GetOpName`，走 `op.GetName(AscendString&)`）。
6. **失效判据修正（等价重构的唯一例外）**：`math/cast/framework/cast_onnx_plugin.cpp` 以 `if (om == -1)` 判 dtype 映射失败，而 `GetOmDtypeFromOnnxDtype()` 未命中返回 `ge::DT_UNDEFINED` 而非 -1，该分支永不触发（迁移前不支持的 dtype 要到引擎选择阶段才报 `Failed to select engine`）。迁移后改判 `om == static_cast<int32_t>(ge::DT_UNDEFINED)`，命中即记录日志并返回 FAILED（对应 UT：`OnnxCastPluginTest.ReturnFailedWhenDtypeUnsupported` 等）；其余如发现同类"判据与工具函数真实返回约定不符"的情况同样处理并在自查表中说明。
7. **framework UT**：按配套用例包补齐/更新 57 个 UT 文件（624 个用例，gtest，命名 `Onnx*PluginTest.*`），UT 直接用 `ge::Operator` 构造输入、不构造 protobuf 对象；放置到对应算子目录 `tests/ut/framework/`（common 插件放 `common/tests/ut/framework/`）。
8. **om 两轮比对自验**：用配套 `verify_om/two_round_compare.sh` 对全部迁移算子执行迁移前/迁移后 om 转换 + 推理 + 逐位比对，作为 PR 前自验记录。

# 详细设计（required）

## 算子分析

### 迁移本质

本任务不改变任何计算逻辑：onnx 模型 → om 图的转换中，参数解析环节的**输入**（onnx 节点属性集合）与**输出**（`op_dest` 上的属性键值序列）都保持不变，变的只是解析函数的实现载体（protobuf 反射 → JSON 解析）。因此无数学公式、无 tiling/kernel 设计，设计核心是"**逐属性语义等价**"。

### attribute JSON 输入约定

迁移后解析函数经 `op_src.GetAttr("attribute")` 拿到的 JSON，键名与 ONNX AttributeProto 字段同名，整体格式 `{"attribute":[ <属性对象>, ... ]}`：

| 属性形态 | type 值 | JSON 示例 |
| --- | --- | --- |
| 标量整数（INT） | 2 | `{"name":"to","type":2,"i":10}` |
| 标量浮点（FLOAT） | 1 | `{"name":"alpha","type":1,"f":"0.25"}`（值以**字符串**承载） |
| 字符串（STRING） | 3 | `{"name":"data_format","type":3,"s":"NCHW"}` |
| 整数列表（INTS） | 7 | `{"name":"axes","type":7,"ints":[1,2]}` |
| 浮点列表（FLOATS） | 6 | `{"name":"float_attrs","type":6,"floats":[0.1,0.2]}` |

解析侧两条关键约定（依据仓内已迁移参考实现 elu/leaky_relu/bounding_box_decode/bernoulli 的读法及其配套 UT）：

1. 标量浮点 `f` 以字符串承载，先取 `std::string` 再转 float（参考实现用 `atof` 等价转换），解析失败返回 FAILED；`i`/`ints`/`floats` 仍为数字。
2. 属性存在但 `f`/`i` 键缺省（proto3 省略零值标量）时取值为 0，**不得回落到算子默认值**（例：Flatten 的 `axis` 属性存在但无 `i` 键 → axis=0 而非默认 1）。

## 算子实现

### 实现方案

#### 通用迁移模式（以 Flatten 为例，before/after 完整对照）

```cpp
// ===== 迁移后 =====
#include "graph/operator.h"
#include "math_onnx_plugin_util.h"
#include "log/log.h"
#include "register/register.h"
#include "nlohmann/json.hpp"

namespace domi {
static Status ParseParamsFlatten(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int axis = 1;                                    // 默认值与迁移前一致
    AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        json attrs = json::parse(attrs_string.GetString());   // 非法 JSON 抛异常
        for (json& attr : attrs["attribute"]) {
            if (attr["name"] == "axis" && attr["type"] == 2) {   // INT，type 不符按缺失
                axis = attr.contains("i") ? attr["i"].get<int>() : 0;  // i 键缺省取 0 不落默认
            }
        }
    }
    op_dest.SetAttr("axis", axis);
    op_dest.SetAttr("name", GetOpName(op_dest));     // 原 node->name() 的等价获取
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Flatten") ...
  .ParseParamsByOperatorFn(ParseParamsFlatten)       // 注册宏替换
  .ImplyType(ImplyType::TVM);
}
```

迁移模式逐项对应：默认值在循环外初始化（缺失 → 默认值 → SUCCESS）；`type` 校验等价于迁移前的 `attr.type() == AttributeProto::INT` 判断；`contains("i")` 等价于 proto3 零值省略语义；非法 JSON 用 `try/catch` 包裹 `json::parse` 及取值过程，catch 中 `OP_LOGE` 并返回 FAILED（迁移前 protobuf 解析失败同样返回 FAILED，返回值语义一致）。

#### 分类迁移策略（按 57 个文件的特征分四类处理）

| 类别 | 范围 | 处理方式 |
| --- | --- | --- |
| A. 无属性插件 | `similar_opp_onnx_plugin.cpp`（63 条目）、`npu_rotary_mul` 型等 | 解析函数体为空实现（仅返回 SUCCESS），保留宏批量注册结构，将 `REGISTER_CUSTOM_ONNX_OP` 宏内 `.ParseParamsFn` 改为 `.ParseParamsByOperatorFn`、宏生成的函数签名改为 `ge::Operator` 入参并删除 `dynamic_cast`；文件级改动集中在宏定义一处 |
| B. 简单标量/列表插件 | Flatten、Squeeze、Transpose、Reduce 系列、Concat、Reshape 等 | 按"通用迁移模式"逐属性改写，重点保持默认值表与 type 校验行为 |
| C. 含枚举映射/复杂语义插件 | Cast（dtype 映射 + 失效判据修正）、DepthToSpace/SpaceToSpace（blocksize 校验失败返回 FAILED 且不写属性）、Clip（FLOAT 字符串承载 + 缺省键按 0）、Pad（pads 奇数长度 FAILED）、BitShift（属性白名单，非 `direction` 属性即 FAILED）、ArgMax（select_last_index==1 返回 FAILED）、Slice/ReduceMean（双 opset 注册分支各自语义） | 逐文件对照迁移前源码改写；失败分支的返回值、日志、属性写入顺序（先失败后写属性）逐一保持；Cast 按任务书 2.1 节第 7 条修正判据 |
| D. 含动态注册/拆图函数插件 | 37 个含 `DynamicInputRegister`/`DynamicOutputRegister`、33 个含 `ParseOpToGraphFn` | 动态注册语句与拆图函数**原样保留**，仅迁移解析函数；`PartitionedCall` 注册项的 `.ParseOpToGraphFn` 不动 |

同一文件内多算子（`similar_opp` 63 条目、`sequence` 6 算子、reduce_sum/accumulate_nv2 各 2 文件）按文件为单位整体迁移，保持注册顺序不变。

#### 失效判据修正（唯一语义例外）

`math/cast/framework/cast_onnx_plugin.cpp` 迁移前：

```cpp
int32_t om = GetOmDtypeFromOnnxDtype(onnx);   // 未命中返回 ge::DT_UNDEFINED（值非 -1）
if (om == -1) { ... return FAILED; }          // 永不触发：判据与工具函数返回约定不符
```

迁移后改为 `if (om == static_cast<int32_t>(ge::DT_UNDEFINED))` 命中即 `OP_LOGE` 并返回 FAILED。影响面：不支持的 dtype（如 `to=20`）从"引擎选择阶段失败"提前到"解析阶段失败"，非法模型两轮 om 转换都失败（两轮比对判据不受影响）、合法模型行为不变。若迁移中发现其他同类"判据与工具函数真实返回约定不符"的情况，按同一原则处理并在自查表中说明。

#### 公共工具复用与新增策略

优先复用 `math_onnx_plugin_util.h` 既有工具（`GetOpName`/`CreateScalar`/`Vec2Tensor`/`GetOmDtypeFromOnnxDtype`/`ChangeFormatFromOnnx`），不复制实现；确需新增公共函数时新增公共头使用仓前缀命名（`math_onnx_plugin_util.h` 同级、`math_` 前缀），保持该头文件零 protobuf 依赖的既有约束。

### UT 设计

1. **放置**：57 个 UT 文件按算子放入 `<prefix>/<op>/tests/ut/framework/test_*_onnx_plugin.cpp`（common 插件放 `common/tests/ut/framework/`）；目录下放置仓内标准 `CMakeLists.txt`（`add_modules_ut_sources` 模板，`cmake/ut.cmake` 按目录算子名自动过滤收集），无需改动任何构建脚本。
2. **写法**（与仓内 `test_reverse_sequence_onnx_plugin.cpp` 参考实现同构）：gtest 直接 `#include` 插件源文件（仓内相对路径，如 `../../../framework/flatten_onnx_plugin.cpp`）；`ge::Operator op_src` + `SetAttr("attribute", JSON 字符串)` 构造输入、空 `op_dest` 调用解析函数；断言返回值（SUCCESS/FAILED）与 `op_dest` 属性值。
3. **覆盖**：每算子四类场景——属性齐全正常解析、属性缺失用默认值返回 SUCCESS、非法 attribute JSON 返回 FAILED、边界/校验（不支持 dtype、type 不符按缺失、`i`/`f` 键缺省按 0、空列表/负值原样透传）。
4. **构建运行**：`bash build.sh` 开启仓内 `OP_HOST_UT` 开关构建，产物 `math_op_host_ut`，`--gtest_filter='Onnx*PluginTest.*'` 全部通过；配套用例包 57 文件 624 用例全量执行。

### 支持硬件

本任务为 Host 侧改造，插件代码与硬件无关，不改变算子计算逻辑与转换产物结构。om 转换精度自验在 Atlas A2（推荐，实测参考环境 CANN 9.1.0 + Ascend910B3）与 Ascend 950 中任一硬件完成即可；迁移前后基线在同一硬件获取。

## 算子约束限制

- 仅迁移 `framework/*_onnx_plugin.cpp` 的参数解析函数及 framework UT；不改 kernel、tiling、算子原型，不改 `ParseOpToGraphFn` 拆图函数，保持动态输入注册（`DynamicInputRegister`/`DynamicOutputRegister`）逻辑不变；
- 等价重构：除 Cast 失效判据修正外，各分支返回值（SUCCESS/FAILED）、写入 `op_dest` 的属性名/属性值/写入顺序、日志语义均与迁移前一致；禁止顺手重构与解耦无关的代码；
- `REGISTER_CUSTOM_OP` 的注册名、`OriginOpType`、`ImplyType` 不变，对 ATC 用户透明；
- 遵循仓内 C++ 编码规范；nlohmann/json 使用仓内 third_party 自带版本，不另引新版本。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| UT | 开启仓内 `OP_HOST_UT` 开关构建，`math_op_host_ut --gtest_filter='Onnx*PluginTest.*'` 全部通过（配套用例包 57 文件 624 用例）；UT 直接使用 `ge::Operator`，不构造 protobuf 对象 | 任务书 3.5 节 |
| 精度（主判据） | 同一 onnx 模型、同一输入，迁移前/迁移后插件分别经 ATC（`--framework=5`，经 `ASCEND_CUSTOM_OPP_PATH` 注入本轮插件并以 `[Register][Check] Plugin of <算子> already registered, skip` 日志留证）转换为 om 并推理，输出**逐数值相等**（等价重构，不设 rtol/atol）；用配套 `verify_om/two_round_compare.sh` 一键两轮比对覆盖全部迁移算子 | 任务书 3.2 节 |
| 精度（辅助判据） | om 推理输出与 golden（onnxruntime 或脚本内 numpy 参考实现）按混合容差标准比对（FP32 rtol=2^-10、atol=2^-16，逐元素硬上限 max(fixed_limit, 32*ULP)，匹配率 ≥0.99，整数/布尔逐位精确），并校验输出 dtype/shape；`golden:false` 用例退化为两轮行为一致 | 任务书 3.2 节；[混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 不涉及。Host 侧等价重构，不改变算子计算逻辑与模型转换产物结构 | 任务书 3.3 节 |
| 内存标准 | 不涉及 | 任务书 3.4 节 |

自验通过后按任务书第 4 节模板整理交付件（自验证步骤说明、精度自验证报告与日志、单算子迁移自查表）提交 `task_submission` 目录，随后在 ops-math 仓提交 PR：按插件文件/算子分组提交、目标目录为各插件文件所在目录及对应 UT 目录、不携带个人敏感信息、不含冗余文件；迁移前旧接口插件在部分 CANN 版本上的加载适配（`undefined symbol: ...ge::onnx::AttributeProto...`）按用例包 `verify_om/compat/README.md` 处理，属环境适配问题、迁移后插件无此依赖。

## 兼容性分析

等价重构，不改任何既有对外符号与行为：插件注册名、`OriginOpType`、`FrameworkType(ONNX)`、`ImplyType` 均不变，ATC 侧模型转换的使用方式与产物结构不变；解析结果的参数序列与迁移前一致，仅解析实现方式变化。迁移后插件不再依赖 `ge_onnx.pb.h` 与 protobuf 运行库，编译链路缩短，proto 定义变动不再波及插件；已合入的 9 个已迁移插件与本批迁移共存于同一注册体系，无冲突。CANN 版本要求大于 9.0.0（om 自验实测基线 CANN 9.1.0 + Atlas A2）。
