# 需求背景（required）

## 需求来源

10 月社区任务：ops-math 仓 onnx 算子插件 protobuf 解耦迁移适配（Host 侧框架适配，等价重构，共 124 个插件条目、57 个插件文件）。

- 任务书：`ops-math_onnx_task_doc.md`（本设计的唯一语义真值源；配套用例包 `test_cases/README.md` 与任务书冲突时以任务书为准）
- 代码合入仓：`https://gitcode.com/cann/ops-math`（迁移改造与 PR 提交均基于该仓）
- 迁移方式参考：ops-math 仓 issue `https://gitcode.com/cann/ops-math/issues/2892`（关联 PR 示例：迁移 onnx 插件 ParseParams API 并补充 framework 层 UT 用例、ReverseSequence 插件迁移、npu_indexing 插件迁移）
- 设计文档提交仓：`cann-ops-competitions`（`https://gitcode.com/cann/cann-ops-competitions`）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/10月社区任务-ops-math算子onnx适配/lunhao2023/docs/design.md`
- 验证配套：任务书随附用例包 `test_cases/`，含 57 个 framework UT 文件（624 个用例）、56 个 om 验证模型生成脚本、`verify_om/` 两轮比对工具，全量覆盖迁移范围

## 背景介绍

### ops-math onnx 插件的现状与问题

ATC 将 onnx 模型转换为 om 模型时，由 GE 框架按 `FrameworkType(ONNX)` 回调各算子插件的参数解析函数，完成"onnx 节点 → 内部算子"的映射。ops-math 仓现有 124 个 onnx 插件条目（文件 × 算子，57 个插件文件）通过 `ParseParamsFn` 注册该解析函数，其签名以 protobuf `Message` 基类为入参（实际传入 `ge::onnx::NodeProto`），插件内部将入参还原为 `NodeProto` 后遍历 protobuf attribute，按 `i()/f()/s()` 等生成代码字段取值。以 `math/arg_max_v2/framework/argmax_onnx_plugin.cpp` 为代表的迁移前形态：

```cpp
#include "onnx_common.h"   // 间接引入 "onnx/proto/ge_onnx.pb.h"

static Status ParseParamsArgMax(const Message* op_src, ge::Operator& op_dest)
{
    auto node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);   // protobuf 还原
    if (node == nullptr) {
        return FAILED;
    }
    int64_t axis = 0;                                               // 默认值
    bool keepdims = false;
    for (const auto& attr : node->attribute()) {                    // protobuf 反射遍历
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis = attr.i();
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keepdims = attr.i() != 0;
        }
    }
    op_dest.SetAttr("axis", axis);
    ...
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMax")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11", "ai.onnx::12", "ai.onnx::13"})
    .ParseParamsFn(ParseParamsArgMax)       // 旧注册接口，依赖 protobuf
    .ImplyType(ImplyType::TVM);
```

该形态带来三类问题（任务书第 1 节）：

1. 与 protobuf 生成代码强耦合：插件源码依赖 `ge_onnx.pb.h`，proto 定义变动波及全部插件；
2. 编译依赖重：需拉入 protobuf 头文件与生成代码，编译链路长；
3. UT 成本高：单测必须构造 protobuf 对象，用例编写与维护成本高。

### 任务目标

将上述 124 个条目的参数解析函数由 `ParseParamsFn` 替换为 `ParseParamsByOperatorFn`，属性解析由 protobuf 遍历改为 `op_src.GetAttr("attribute")` 获取 JSON 字符串后用 nlohmann/json 解析，使插件代码彻底摆脱对 protobuf 生成代码的依赖，同时保持插件原有解析语义与功能不变。本任务只涉及各算子目录 `framework/*_onnx_plugin.cpp` 及对应 framework UT，不涉及 kernel、tiling、算子原型的开发与修改。

### 仓内迁移先例

ops-math master 上已有一批完成同型迁移的插件（reverse_sequence、correlation、npu_rotary_mul、npu_format_cast、npu_indexing、elu、leaky_relu、bounding_box_decode、bernoulli 等），对应 issue 2892 的关联 PR。本批 124 个条目沿用同一模式，行为契约、UT 写法与这些在库先例保持一致。

# 需求分析（required）

## 需求描述

对 124 个 onnx 插件条目完成 protobuf 解耦等价重构，统一落实六项替换：

1. 解析函数签名：`domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest)`（函数名保持仓内现有命名，如 `ParseParamsCast`、`ParseParamsFlatten`）；
2. 注册方式：`.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`，`REGISTER_CUSTOM_OP` 的 `FrameworkType(ONNX)`、`OriginOpType`、`ImplyType` 均不变；
3. 属性解析：不再遍历 protobuf attribute，统一 `op_src.GetAttr("attribute")` 取 JSON 字符串后用 nlohmann/json 解析；
4. 依赖移除：删除对 `ge_onnx.pb.h` 及 protobuf 运行库的依赖，include 替换为 `operator.h`、`nlohmann/json.hpp`、仓内公共工具头文件（`math_onnx_plugin_util.h`）、`register/register.h`；
5. 节点名获取：原 protobuf `node->name()` 改为 `op_dest.SetAttr("name", GetOpName(op_dest))`；
6. 语义保持：等价重构，迁移前后各分支返回值（SUCCESS/FAILED）、写入 `op_dest` 的属性、缺失属性默认值行为完全一致；唯一例外是 Cast 插件失效判据的修正（见详细设计），其余禁止顺手重构与解耦无关的代码。

## 需求拆解

1. **签名与注册替换**：删除原函数开头的 `dynamic_cast`/`reinterpret_cast` 判空分支（`ge::Operator` 入参无此失败模式），注册宏逐条替换；`similar_opp_onnx_plugin.cpp` 等 63 条目的宏批量注册文件在宏定义一处完成改动。
2. **属性解析替换**：`GetAttr("attribute")` → `nlohmann::json::parse` → 按 `name` 分派逐属性读取；非法 JSON（解析异常或未知异常）记录日志并返回 FAILED；属性缺失保持原默认值并返回 SUCCESS；属性名匹配但 `type` 值不符按缺失处理。
3. **include 与公共工具**：删除 `onnx_common.h` 入口；优先复用 `common/inc/framework/math_onnx_plugin_util.h` 既有工具（`GetOpName`、`GetOmDtypeFromOnnxDtype`、`ChangeFormatFromOnnx`、`CreateScalar`、`Vec2Tensor` 等）；确需新增公共头时使用仓前缀命名，避免与其他算子仓重名。
4. **失效判据修正（唯一语义例外）**：`math/cast/framework/cast_onnx_plugin.cpp` 以 `if (om == -1)` 判断 dtype 映射失败，而 `GetOmDtypeFromOnnxDtype()` 未命中返回 `ge::DT_UNDEFINED` 而非 -1，该判据永不触发；迁移后改判 `om == static_cast<int32_t>(ge::DT_UNDEFINED)`，命中记录日志并返回 FAILED。
5. **framework UT**：配套用例包 57 个 UT 文件（624 个用例，gtest，命名 `Onnx*PluginTest.*`）放入各算子 `tests/ut/framework/` 目录；UT 直接用 `ge::Operator` 构造输入，不构造 protobuf 对象。
6. **om 两轮自验**：用配套 `verify_om/two_round_compare.sh` 对全部迁移算子执行迁移前/迁移后两轮 om 转换 + 推理 + 比对，作为 PR 前自验记录；通过后按任务书第 4 节整理交付件提交 `task_submission`，再向 ops-math 仓提交 PR。

# 详细设计（required）

## 算子分析

### 迁移本质

本任务不改变任何计算逻辑。onnx 模型 → om 图转换中，参数解析环节的输入（onnx 节点属性集合）与输出（写入 `op_dest` 的属性键值序列）都保持不变，变化的只是解析实现载体：protobuf 反射遍历 → JSON 解析。因此本设计无数学公式、无 tiling/kernel 内容，核心是**逐属性语义等价**与**返回值等价**。

### attribute JSON 输入约定

迁移后解析函数经 `op_src.GetAttr("attribute")` 获取的 JSON 顶层格式为 `{"attribute": [ <属性对象>, ... ]}`，属性对象键名与 ONNX AttributeProto 字段同名：

| 属性形态 | type 值 | JSON 示例 | 读取要点 |
| --- | --- | --- | --- |
| 标量整数 INT | 2 | `{"name":"axis","type":2,"i":2}` | `i` 为数字 |
| 标量浮点 FLOAT | 1 | `{"name":"alpha","type":1,"f":"0.25"}` | `f` 以**字符串**承载，读字符串后转 float |
| 字符串 STRING | 3 | `{"name":"mode","type":3,"s":"half_pixel"}` | `s` 为字符串 |
| 整数列表 INTS | 7 | `{"name":"axes","type":7,"ints":[2]}` | 数组，空列表原样透传 |
| 浮点列表 FLOATS | 6 | `{"name":"pads","type":6,"floats":[0.5,0.5]}` | 数组 |

两条关键解析约定（依据仓内已迁移参考实现 elu/leaky_relu/bounding_box_decode/bernoulli 的读法及配套 UT）：

1. 标量浮点 `f` 由 GE 序列化为字符串，须按字符串读取再转 float（`atof` 等价转换），转换失败返回 FAILED；标量整型 `i` 与 `ints`/`floats` 列表仍为数字。
2. 属性存在但 `f`/`i` 键缺省（proto3 省略零值标量）时取值为 0，**不得回落到算子默认值**（例：Flatten 的 `axis` 属性存在、无 `i` 键 → axis=0 而非默认 1）。

### 迁移语义保持对照（任务书 2.4 节）

| 算子 | 属性 | 迁移前后一致的行为 |
| --- | --- | --- |
| Cast | to | 缺失返回 FAILED；映射后写入 `dst_type`；不支持的 dtype 返回 FAILED（失效判据修正后的新行为，见下文） |
| Squeeze | axes | 缺失按空列表返回 SUCCESS；负 axis 原样透传 |
| Flatten | axis | 缺失默认 1 返回 SUCCESS；`i` 键缺省按 0；type 不符按缺失 |
| Concat | axis | 必需属性，缺失返回 FAILED 且不写任何属性；多个同名属性取首个 INT |
| 通用 | 任意 | 属性名匹配但 type 值不符按缺失处理；非法 JSON 返回 FAILED |

## 算子实现

### 实现方案

#### 通用迁移模式（以 Cast 为主例）

`math/cast/framework/cast_onnx_plugin.cpp` 迁移后（对照 `common/inc/framework/math_onnx_plugin_util.h` 工具签名）：

```cpp
#include <string>
#include "graph/operator.h"
#include "register/register.h"
#include "common/inc/framework/math_onnx_plugin_util.h"
#include "nlohmann/json.hpp"
#include "log/log.h"

namespace domi {
Status ParseParamsCast(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op_dest.GetName().c_str(), "get attribute failed.");
        return FAILED;                       // 缺 attribute 键按迁移前分支返回 FAILED
    }
    nlohmann::json attrs;
    try {
        attrs = nlohmann::json::parse(attrs_string.GetString());
    } catch (...) {                          // 非法 JSON：记日志、FAILED（与迁移前返回值一致）
        OP_LOGE(op_dest.GetName().c_str(), "parse attribute json failed.");
        return FAILED;
    }
    for (const auto& attr : attrs["attribute"]) {
        const std::string name = attr["name"].get<std::string>();
        if (name != "to" || attr["type"] != 2) {
            continue;                        // type 不符按缺失处理
        }
        int64_t to = attr.contains("i") ? attr["i"].get<int64_t>() : 0;  // i 缺省按 0，不落默认
        int32_t om = GetOmDtypeFromOnnxDtype(static_cast<uint32_t>(to));
        if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) {
            // 迁移前判据为 om == -1（永不触发）；按任务书 2.1 节第 7 条修正
            OP_LOGE(op_dest.GetName().c_str(), "unsupported onnx dtype:%lld.", to);
            return FAILED;
        }
        op_dest.SetAttr("dst_type", static_cast<ge::DataType>(om));
    }
    op_dest.SetAttr("name", GetOpName(op_dest));    // 原 node->name() 的等价获取
    return SUCCESS;
}
}  // namespace domi

REGISTER_CUSTOM_OP("Cast")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9", "ai.onnx::10", ...})
    .ParseParamsByOperatorFn(ParseParamsCast)       // 注册宏替换
    .ImplyType(ImplyType::TVM);
```

迁移要点逐项对应：

- 默认值在属性循环外初始化：缺失 → 默认值 → SUCCESS；
- `attr["type"] != 2` 判断等价于迁移前 `attr.type() == AttributeProto::INT`；
- `contains("i")` 等价于 proto3 零值标量省略语义；
- 非法 JSON 用 try/catch 包裹 parse 与取值全过程，catch 中记日志并返回 FAILED；
- 解析成功路径最后统一 `SetAttr("name", GetOpName(op_dest))`，属性写入内容与顺序与迁移前一致。

#### 分类迁移策略（57 个插件文件按特征分四类）

| 类别 | 识别特征 | 范围（代表） | 处理方式 |
| --- | --- | --- | --- |
| A. 无属性批量注册 | 解析函数体为空/仅判空，或经宏批量生成注册 | `common/src/framework/similar_opp_onnx_plugin.cpp`（63 条目） | 宏定义一处改签名（`ge::Operator` 入参、删 `dynamic_cast`）与注册宏（`ParseParamsByOperatorFn`），保持注册顺序与算子清单不变 |
| B. 简单标量/列表 | 1~3 个属性、线性默认值表 | Flatten、Squeeze、Transpose、Concat、Reshape、Reduce 系列、DepthToSpace/SpaceToDepth | 按通用迁移模式逐属性改写；失败分支（如 Pad pads 长度为奇数、DepthToSpace blocksize 非法返回 FAILED）逐条保持 |
| C. 枚举映射/复杂语义 | dtype 映射、字符串属性、双 opset 分支 | Cast（判据修正）、Clip（FLOAT 字符串承载 + 缺省键按 0）、Pad（INTS 长度校验）、BitShift（属性白名单，非 `direction` 即 FAILED）、ArgMax（`select_last_index==1` 返回 FAILED）、Slice/ReduceLogSum 等多 opset 注册 | 逐文件对照迁移前源码改写，返回值、日志、属性写入顺序逐一比对 |
| D. 含动态注册/拆图 | `DynamicInputRegister`/`DynamicOutputRegister`、`ParseOpToGraphFn` | 37 个含动态注册、33 个含拆图函数的文件 | 动态注册与拆图函数**原样保留**（任务书 7 节第 4 条），仅迁移参数解析函数；`PartitionedCall` 注册项的 `.ParseOpToGraphFn` 不动 |

同一文件多算子（similar_opp 63 条目、sequence_onnx_plugin 6 个 Sequence 算子、accumulate_nv2 下 Sum/Mean 两文件）以文件为单位整体迁移。`experimental/math/asinh` 与 `math/maximum` 两份 Max 插件同型同改。

#### 失效判据修正（唯一语义例外）

`GetOmDtypeFromOnnxDtype()`（`math_onnx_plugin_util.h`）对未命中 dtype 返回 `ge::DT_UNDEFINED`，迁移前 Cast 以 `if (om == -1)` 判失败，分支永不触发——不支持的 dtype（如 `to=20`）要到引擎选择阶段才报 `Failed to select engine`。迁移后改判 `om == static_cast<int32_t>(ge::DT_UNDEFINED)`：命中即记日志并返回 FAILED（UT `OnnxCastPluginTest.ReturnFailedWhenDtypeUnsupported`、`ReturnFailedWhenToIsZeroWithoutIField` 按此判定）。影响面分析：非法模型迁移前后两轮 om 转换均失败（两轮比对判据不受影响）、合法模型行为不变。迁移中若发现其他"判据与工具函数真实返回约定不符"的同类情况，按同一原则处理并在自查表中说明。

### UT 设计

1. **放置**：57 个 UT 文件放入 `<prefix>/<op>/tests/ut/framework/`（common 插件放 `common/tests/ut/framework/`），include 按仓内相对路径标注（如 `../../../framework/flatten_onnx_plugin.cpp`）；目录配套仓内标准 `CMakeLists.txt`（`add_modules_ut_sources` 模板），构建脚本无需改动。
2. **写法**（与在库先例 `test_reverse_sequence_onnx_plugin.cpp` 同构）：gtest 直接 `#include` 插件源文件；辅助函数 `SetAttr("attribute", ge::AscendString(JSON))` 构造 `op_src`，空 `op_dest` 调用解析函数；断言返回值与 `op_dest` 属性值。
3. **覆盖**：每算子四类场景——属性齐全正常解析、属性缺失用默认值返回 SUCCESS、非法 attribute JSON 返回 FAILED、边界/校验（`i`/`f` 键缺省按 0、type 不符按缺失、空列表/负值原样透传、不支持 dtype 失败）。
4. **执行**：开启仓内 `OP_HOST_UT` 开关构建 `math_op_host_ut`，`--gtest_filter='Onnx*PluginTest.*'` 全部通过（624 用例）。

### 支持硬件

本任务为 Host 侧改造，插件代码与硬件无关，不改变算子计算逻辑与转换产物结构。om 转换精度自验在 Atlas A2（推荐）与 Ascend 950 中任一硬件完成即可，迁移前后基线在同一硬件获取；CANN 版本要求大于 9.0.0。

## 算子约束限制

- 仅迁移 `framework/*_onnx_plugin.cpp` 的参数解析函数及 framework UT；不改 kernel、tiling、算子原型、`ParseOpToGraphFn` 拆图函数；`DynamicInputRegister`/`DynamicOutputRegister` 逻辑保持不变；
- 等价重构：除 Cast 失效判据修正外，各分支返回值、写入 `op_dest` 的属性名/值/顺序、日志语义均与迁移前一致；禁止顺手重构与解耦无关的代码；
- 插件注册名、`OriginOpType`、`FrameworkType(ONNX)`、`ImplyType` 不变，对 ATC 使用者透明；
- 遵循仓内 C++ 编码规范；nlohmann/json 使用仓内 third_party 自带版本，不另引新版本；新增公共头使用仓前缀命名并保持零 protobuf 依赖。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| framework UT | 开启 `OP_HOST_UT` 构建，`math_op_host_ut --gtest_filter='Onnx*PluginTest.*'` 全部通过（57 文件 624 用例）；UT 直接用 `ge::Operator`，不构造 protobuf 对象 | 任务书 3.5 节 |
| 精度（主判据） | 同一 onnx 模型、同一输入，迁移前/迁移后插件分别经 ATC（`--framework=5`，经 `ASCEND_CUSTOM_OPP_PATH` 注入本轮插件、日志留证生效版本）转 om 并推理，输出**逐数值相等**（等价重构，不设 rtol/atol）；用 `verify_om/two_round_compare.sh` 一键两轮比对覆盖全部迁移算子 | 任务书 3.2 节 |
| 精度（辅助判据） | om 推理输出与 golden（onnxruntime 或脚本内 numpy 参考实现）按混合容差标准比对（FP32 rtol=2^-10、atol=2^-16，逐元素硬上限 max(fixed_limit, 32*ULP)，匹配率 ≥0.99，整数/布尔逐位精确），同时校验输出 dtype/shape；`golden:false` 用例退化为两轮行为一致 | 任务书 3.2 节；[混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 平台容错 | 个别算子在部分 SOC 因 CANN op-info 无条目无法 ATC 转换时，记录平台、算子与 ATC 报错原文，该项以"迁移前后两轮行为一致"为准 | 任务书 3.2 节 |
| 性能标准 | 不涉及。Host 侧等价重构，不改变算子计算逻辑与转换产物结构 | 任务书 3.3 节 |
| 内存标准 | 不涉及 | 任务书 3.4 节 |

自验通过后按任务书第 4 节模板整理交付件（自验证步骤说明、精度自验证报告与日志、单算子迁移自查表逐项填写：include 替换、签名/注册宏/属性解析/name 获取迁移、缺失属性默认值 SUCCESS、非法 JSON FAILED、用例更新、`git diff --check` 与 `bash build.sh --onnxplugin -j8` 通过）提交 `task_submission` 目录，随后在 ops-math 仓提交 PR：按插件文件/算子分组提交、目标目录为各插件文件所在目录及对应 UT 目录、不携带个人敏感信息、不含冗余文件。迁移前旧接口插件在部分 CANN 版本上的加载适配问题（`undefined symbol: ...ge::onnx::AttributeProto...`）按用例包 `verify_om/compat/README.md` 处理，属环境适配问题；迁移后插件不挂该兼容库、亦无此依赖。

## 兼容性分析

等价重构，不改任何既有对外符号与行为：插件注册名、`OriginOpType`、`FrameworkType(ONNX)`、`ImplyType` 均不变，ATC 模型转换的使用方式与产物结构不变；同一 onnx 节点解析出的 `op_dest` 属性参数序列与迁移前一致，仅解析实现方式变化。迁移后插件不再依赖 `ge_onnx.pb.h` 与 protobuf 运行库，编译链路缩短，proto 定义变动不再波及插件。已合入仓的已迁移插件与本批 124 个条目共存于同一注册体系，无冲突；om 精度自验基线要求 CANN 大于 9.0.0。
