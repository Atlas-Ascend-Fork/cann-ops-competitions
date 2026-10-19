# ops-math 算子 ONNX 适配设计文档

# 需求背景（required）

## 需求来源

- 来源：《ops-math 算子 onnx 插件 protobuf 解耦任务书》（10 月社区任务"ops-math 算子 onnx 适配"）；
- 仓库：https://gitcode.com/cann/ops-math ；
- 开发者私仓（本次交付）：https://gitcode.com/Irene-/ops-math ；
- 参考实现 issue：https://gitcode.com/cann/ops-math/issues/2892 （关联 PR：`fix: 迁移onnx插件ParseParams API并补充framework层UT用例`、`feat(onnx): 迁移 ReverseSequence 插件至 ParseParamsByOperatorFn 并补单测`、`refactor: npu_indexing onnx 插件迁移至 ParseParamsByOperatorFn 接口`）；
- 目标分支：各迁移插件文件所在算子目录（如 `math/cast/framework/`、`common/src/framework/`）及对应 framework UT 目录。

## 背景介绍

### ONNX 插件框架的 protobuf 耦合现状

ops-math 仓的 ONNX 插件（`*_onnx_plugin.cpp`）通过 `REGISTER_CUSTOM_OP` 宏注册到 GE 框架（`FrameworkType(ONNX)`），ATC 模型转换时由 GE 回调插件的参数解析函数，把 ONNX 节点属性映射为内部算子的属性。

迁移前，ops-math 仓仍有 **124 个 ONNX 插件条目**（分布见下方"迁移范围表"，源码静态扫描结果）使用**依赖 protobuf 的旧接口**注册参数解析函数：

```cpp
// 迁移前（旧接口）
static Status ParseParamsXxx(const Message* op_src, ge::Operator& op_dest);
...
    .ParseParamsFn(ParseParamsXxx)
```

该接口的入参 `op_src` 实际是 `ge::onnx::NodeProto` 对象（protobuf 生成类），插件内做 `dynamic_cast` 后遍历 `node->attribute()`，按 `attr.name()` / `attr.type()` 匹配并读取 `attr.i()/f()/s()/ints()/floats()` 等 protobuf 字段。插件因此强依赖 `onnx/proto/ge_onnx.pb.h`（经 `common/inc/framework/onnx_common.h` 间接引入）及 protobuf 运行库。

### 三类问题

1. **与 protobuf 生成代码强耦合**：插件源码依赖 `ge_onnx.pb.h`，proto 定义变动会波及全部插件；
2. **编译依赖重**：插件编译需拉入 protobuf 头文件与生成代码，编译链路长；
3. **UT 成本高**：单测必须构造 protobuf 对象（`NodeProto`、`AttributeProto`），用例编写与维护成本高。

### 解耦方向（已定接口）

GE 框架已提供以 `ge::Operator` 为入参的新参数解析接口 `ParseParamsByOperatorFn`：节点属性由 GE 侧序列化为 JSON 字符串挂在 `op_src` 的 `"attribute"` 属性上，节点输入/输出名列表分别挂在 `"input"` / `"output"` 属性上；插件改用 `op_src.GetAttr("attribute", ...)` 取 JSON 后用 nlohmann/json 解析，可彻底摆脱对 protobuf 生成代码与运行库的依赖。

## 现状分析（迁移前基线）

对迁移前基线（worktree @ `HEAD^` = `b7a0fd17f`）静态扫描与动态库符号核对：

| 现状项 | 数据 |
| --- | --- |
| 任务书迁移范围表覆盖 | **124 个算子条目 / 57 个插件文件** |
| 迁移前仍使用 `ParseParamsFn` 的 `*_onnx_plugin.cpp` | 57 个（迁移后为 **0**） |
| 已采用新接口（`ParseParamsByOperatorFn`）的参考插件 | `math/shape`、`random/stateless_bernoulli` 等 |
| 公共工具 | `common/inc/framework/math_onnx_plugin_util.h`（迁移中新增，已存在 `GetOpName` 等）+ `common/inc/framework/onnx_common.h`（protobuf 相关） |
| 迁移前插件动态库的 protobuf 符号 | `liboponnx_plugin_math.so` 中 **5 个**未定义 `ge::onnx::*` 符号（详见"兼容性设计"） |
| 配套 UT | 57 个 framework UT 文件（用例包提供 **624 个用例**） |
| 配套 om 验证模型生成脚本 | 56 个 `gen_model_*.py`（两份 Max 插件共用一个脚本） |

---

# 需求分析（required）

## 需求描述

完成 **124 个算子 onnx 插件条目（57 个插件文件）** 的 protobuf 解耦迁移适配。本任务属于 **Host 侧等价重构**：迁移前后插件解析行为必须完全一致，不改变算子计算逻辑、不改变模型转换产物结构，仅替换参数解析函数的入参形态、注册方式、属性解析手段与编译依赖。

迁移后接口（保持各插件原有函数命名，如 `ParseParamsCast`、`parse_params_pad_v11`）：

```cpp
domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest);

REGISTER_CUSTOM_OP("Xxx")
    .FrameworkType(ONNX)
    .OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParamsXxx)
    .ImplyType(ImplyType::TVM);
```

## 需求拆解

1. **解析函数签名替换**：`const Message* op_src` → `const ge::Operator& op_src`；随之删除 `dynamic_cast<const ge::onnx::NodeProto*>` 及判空分支（迁移后入参类型即编译期保证，`op_src` 不可能为空）；
2. **注册方式替换**：`.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`（函数名、`OriginOpType`、`ImplyType`、`ParseOpToGraphFn` 等其他注册项一律不动）；
3. **属性解析方式替换**：`node->attribute()` 遍历 → `op_src.GetAttr("attribute", attrs_string)` 取 JSON，nlohmann/json 解析；解析时校验 `attr.value("type", 0)` 与算子属性定义一致，并按 `f`（字符串承载的浮点标量）/ `i` / `s` / `ints` / `floats` 取值；
4. **移除 protobuf 依赖**：迁移范围内 57 个插件删除对 `onnx_common.h`（及 `ge_onnx.pb.h`、protobuf 运行库）的依赖，include 替换为 `graph/operator.h`、`nlohmann/json.hpp`、仓内公共工具头 `common/inc/framework/math_onnx_plugin_util.h`、`register/register.h`；
5. **算子名称获取替换**：原 `node->name()` → `op_dest.SetAttr("name", GetOpName(op_src))`；
6. **公共工具函数**：优先复用仓内已有公共工具（`math_onnx_plugin_util.h` 中的 `GetOpName`、`GetOmDtypeFromOnnxDtype`、`ChangeFormatFromOnnx` 等）；本次迁移确需新增的公共函数（`GetNodeInputNum`、`GetNodeOutputNum`、`StrToFloat`）与下沉自 `onnx_common.h` 的 `Vec2Tensor`、`CreateScalar` 一并放入该头，命名沿用仓前缀 `math_` 避免与其他算子仓重名；
7. **语义保持**：属性缺失时保持原有默认值行为并返回 `SUCCESS`；属性非法（JSON 解析异常或未知异常）时记录日志并返回 `FAILED`；各分支返回值（SUCCESS/FAILED）与迁移前保持一致；除解耦改造外，不得修改插件其他逻辑，不得顺手重构与解耦无关的代码。不解析 attribute JSON 的注册路径保持原 SUCCESS 语义（实现与 UT 都不改）；迁移前永远不成立的失效判据须按工具函数的真实返回约定修正（Cast 的 dtype 映射失败判断，见 3.5 节）；
8. **验证**：按任务书 3.5 节完成 framework UT（`--gtest_filter='Onnx*PluginTest.*'` 全部通过）与迁移前后两轮 om 转换精度比对自验。

---

# 详细设计（required）

## 算子（插件）分析

### 接口对比

| 项 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 解析函数签名 | `Status ParseParamsXxx(const Message* op_src, ge::Operator& op_dest)` | `Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)` |
| 注册宏 | `.ParseParamsFn(ParseParamsXxx)` | `.ParseParamsByOperatorFn(ParseParamsXxx)` |
| 属性来源 | `ge::onnx::NodeProto::attribute()`（protobuf 对象） | `op_src.GetAttr("attribute", ge::AscendString&)`（JSON 字符串） |
| 属性取值 | `attr.i()/f()/s()/ints()/floats()` | nlohmann/json：`attr["i"]`(int64) / `attr["f"]`(**字符串转 float**) / `attr["s"]` / `attr["ints"]` / `attr["floats"]` |
| 节点输入/输出个数 | `node->input_size()` / `node->output_size()` | `GetNodeInputNum(op_src)` / `GetNodeOutputNum(op_src)`（读 `"input"` / `"output"` 属性） |
| 节点名 | `node->name()` | `GetOpName(op_src)` |
| 头文件依赖 | `onnx_common.h`（→ `ge_onnx.pb.h`） | `math_onnx_plugin_util.h`、`graph/operator.h`、`nlohmann/json.hpp`、`register/register.h` |
| 编译依赖 | 需 protobuf 生成代码与运行库 | 无 protobuf |
| 动态库 protobuf 符号 | 5 个未定义 `ge::onnx::*` | **0 个** |

### 属性 JSON 结构

GE 侧将 ONNX 节点序列化为 JSON，顶层键 `attribute` 为属性数组，每个元素为一个属性对象：

```json
{
  "attribute": [
    { "name": "alpha",  "type": 1, "f": "0.25" },
    { "name": "to",     "type": 2, "i": 10 },
    { "name": "auto_pad","type": 3, "s": "SAME_UPPER" },
    { "name": "axes",   "type": 7, "ints": [2] },
    { "name": "scales", "type": 6, "floats": [0.5, 0.25] }
  ]
}
```

约定（依据任务书 2.4 节，与仓内已迁移参考实现及 GE `Pb2Json` 实测核对）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | int | ONNX AttributeProto 类型枚举：FLOAT=1、INT=2、STRING=3、FLOATS=6、INTS=7 |
| `i` | 数字 | 标量整型属性值 |
| `f` | **字符串** | 标量浮点属性值（GE 序列化时以字符串承载，须按字符串读后再转 float） |
| `s` | 字符串 | 单值字符串属性 |
| `ints`/`floats` | 数组 | 列表属性 |

两条关键语义规则：

1. **`f` 键缺省取 0.0，不得回落默认值**：proto3 序列化会省略值为 0 的标量字段。若算子属性存在且 `type=FLOAT` 但 JSON 中无 `f` 键，表示该属性实际值为 0，直接取 `0.0f`，**不能**回落到算子的默认值（例：某算子 `alpha` 属性存在但无 `f` 键 → alpha=0，而非默认值）；
2. **type 不符按缺失处理**：属性名匹配但 `type` 与算子定义不符时，视为该属性缺失，保持默认值行为（UT 中 `Ignore*WithMismatchedType` 系列用例覆盖）。

## 插件实现（改造范式）

以 Cast 为例（真实迁移 diff 摘录）：

```cpp
// 迁移后
#include "math_onnx_plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;
const int32_t kCastTypeInt = 2;

static Status ParseParamsCast(const ge::Operator& op_src, ge::Operator& op_dst)
{
    bool bfind_to = false;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("type", 0) != kCastTypeInt) {
                        continue;                      // type 不符按缺失处理
                    }
                    if (attr.value("name", "") == "to") {
                        bfind_to = true;
                        int32_t onnx = attr.value("i", 0);
                        int32_t om = static_cast<int32_t>(GetOmDtypeFromOnnxDtype(onnx));
                        if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) {   // ← 失效判据修正点
                            OP_LOGE(GetOpName(op_dst).c_str(), "do not support onnx dtype %d", onnx);
                            return FAILED;
                        }
                        op_dst.SetAttr("dst_type", om);
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dst).c_str(), "JSON parse error: %s", e.what());
            return FAILED;                                 // 非法 JSON → FAILED
        } catch (...) {
            OP_LOGE(GetOpName(op_dst).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    if (!bfind_to) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Message op_src do not have attribute to.");
        return FAILED;                                     // 缺失 → FAILED（保持迁移前语义）
    }
    op_dst.SetAttr("name", GetOpName(op_src));             // node->name() 的等价替换
    return SUCCESS;
}
}
```

改造后的插件统一遵循如下结构：

1. **入参校验**：因入参类型即 `ge::Operator`，不再有 `dynamic_cast` 失败分支；
2. **取属性 JSON**：`GetAttr("attribute", ...)` 不成功时（属性不存在）保持各算子迁移前行为（默认值 + `SUCCESS`）；
3. **try/catch 包裹**：`nlohmann::json::exception` 与未知异常都返回 `FAILED` 并打日志；
4. **逐属性匹配**：先 `name`、再 `type`，再按类型取 `f`/`i`/`s`/`ints`/`floats`；
5. **失败分支语义逐条对齐迁移前**：默认值、`SUCCESS`/`FAILED` 返回时机与原实现一一对应。

### 失效判据修正（Cast）

迁移前 `cast_onnx_plugin.cpp` 的"不支持 dtype"判据写作 `if (om == -1)`（或等价的永假/永真形式），在迁移前的工具函数返回约定下**永远不成立**，属任务书 2.1 节第 7 条与 2.4 节表格明确要求修正的例外项。迁移后按本仓公共工具的真实返回约定修正为：

```cpp
if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) { ... return FAILED; }
```

`GetOmDtypeFromOnnxDtype(int onnx_type)` 在 `math_onnx_plugin_util.h` 中的约定是：命中 `onnx2om_dtype_map` 返回对应 `ge::DataType`，未命中返回 `ge::DT_UNDEFINED`。修正后该分支可达，UT 中已补 `to` 取不支持 dtype（如 `to=20`）的用例守护。

## 公共工具设计（math_onnx_plugin_util.h）

解耦后插件对公共工具的依赖集中到 `common/inc/framework/math_onnx_plugin_util.h`（本次迁移新增/扩充，全头**不含任何 protobuf 依赖**），其中：

| 函数 | 来源 | 作用 |
| --- | --- | --- |
| `GetOpName(op)` | 原有 | 取算子名（替代 `node->name()`），失败返回 `"None"` |
| `GetNodeInputNum(op_src)` | **新增** | ONNX 节点输入个数等价读取。新接口下 `ParseParams` 返回时才挂接输入，故解析期 `GetInputsSize()` 恒为 0；GE 把 `NodeProto::input`（**含 initializer 名**）以同名属性注入，函数读该属性取 size，属性不存在时回落 `GetInputsSize()`（兼容 UT 手工构造的 op_src） |
| `GetNodeOutputNum(op_src)` | **新增** | 上述的 `output` 对应物（替代 `NodeProto::output_size()`） |
| `StrToFloat(str, value)` | **新增** | 字符串转 float（空串/非数字/尾部有残余字符/`ERANGE` 越界均返回 false），用于读 `f` 字段；返回 false 时插件记日志并 `FAILED` |
| `Vec2Tensor` / `CreateScalar` | **下沉自 `onnx_common.h`** | 二者只依赖 GE `Shape/TensorDesc/Tensor`，与 protobuf 无关，随解耦下沉至本头，供插件继续使用（避免插件为用一个模板函数而 include `onnx_common.h` 拖回 protobuf） |
| `GetOmDtypeFromOnnxDtype` / `DataTypeOnnx` / `onnx2om_dtype_map` | 原有 | ONNX dtype 枚举 → `ge::DataType`；未命中返回 `DT_UNDEFINED`，供 Cast 等使用 |
| `ChangeFormatFromOnnx` | 原有 | 按 ONNX 语义设置输入/输出的 origin format 与 format |

**与 ops-nn 仓做法的差异**：ops-nn 在迁移中删除了 `onnx_common.h`；ops-math 仓因范围外仍有两个插件引用该头（见"迁移范围与文件清单"末段），故 `onnx_common.h` **保留**，仅在迁移范围内的 57 个文件中断开引用。这是本仓解耦的边界：**范围内 57 个插件已零 protobuf 依赖**，`onnx_common.h` 的最终删除留给后续把 `math/shape`、`random/stateless_bernoulli` 纳入范围的迁移。

## 特殊场景与语义保持清单

| 场景 | 处理 |
| --- | --- |
| `f` 键缺省（proto3 零值省略） | 取 0.0，**不回落**算子默认值 |
| 属性 `type` 与定义不符 | 按缺失处理（保持默认值） |
| 非法 attribute JSON | 日志 + 返回 `FAILED` |
| 不解析 attribute JSON 的插件 | 豁免：仍照写固定属性并返回 `SUCCESS`（实现与 UT 均不改） |
| 节点 input/output 个数 | `GetNodeInputNum/GetNodeOutputNum` 等价读取（GE 以属性形式注入，**含 initializer 名**） |
| 数值边界判据（如 Unsqueeze 空 axes 返回 `PARAM_INVALID`、TopK V9 的 `k == -1` 返回 `PARAM_INVALID`、Slice V8/V9 的 `starts`/`ends` 为空返回 `FAILED`、Clip V11 的 min/max 全空返回 `FAILED`） | 逐条与迁移前实现比对，行为保持一致 |
| 迁移前永远不成立的失效判据 | 按工具函数真实返回约定修正（Cast 的 dtype 映射失败判断），并新增 UT 用例守护 |
| `ParseOpToGraphFn`（子图拆图，如 Reduce 家族、Slice、Clip、Pad 注册 PartitionedCall） | **不在迁移范围**，保持原样 |
| `DynamicInputRegister` 等动态输入注册 | 保持原样 |

## 迁移范围与文件清单

按任务书范围表共 **57 个插件文件 / 124 个算子条目**，分布如下：

| 目录 | 文件数 | 插件文件（相对该目录） |
| --- | --- | --- |
| `common/src/framework/` | **14** | `concat_from_sequence`、`deformable_offsets`、`einsum`、`eyelike`、`phony_concat`、`phony_split`、`random_normal_like`、`random_uniform_like`、`reduceL1`、`reduceL2`、`repeat_with_pad`、`sequence`（6 条目）、`similar_opp`（63 条目）、`split_to_sequence` |
| `math/` | **25** | `accumulate_nv2/{mean,sum}`、`addcmul`、`arg_max_v2`、`arg_min`、`bitwise_xor`、`cast`、`cumsum`、`is_inf`、`maximum`、`minimum`、`mod`、`one_hot`、`reduce_log_sum_exp`、`reduce_max`、`reduce_mean`、`reduce_min`、`reduce_prod`、`reduce_sum/{reduce_sum,reduce_log_sum}`、`right_shift`、`sign_bits_pack`、`sign_bits_unpack`、`square_sum_v1`、`topk` |
| `conversion/` | **14** | `clip_by_value`、`concat`、`depth_to_space`、`flatten`、`pad_v3`、`reshape`、`slice`（NPUSlice）、`space_to_depth`、`split_d`、`squeeze`、`strided_slice_v2`（Slice）、`transpose`、`trilu`、`unsqueeze` |
| `random/` | **3** | `dsa_random_uniform`、`stateless_random_normal_v2`、`stateless_sample_multinomial` |
| `experimental/math/asinh/` | **1** | `max_onnx_plugin.cpp`（Max，默认构建**不含**，见下） |
| **合计** | **57** | 覆盖 124 个算子条目；其中 `common` 的 `sequence`（6）+ `similar_opp`（63）= 69 条目单独占 2 个文件 |

> 目录计数存在两处易错点：`math/reduce_sum/framework/` 一个目录内**两个插件文件**（`reduce_sum_onnx_plugin.cpp`、`reduce_log_sum_onnx_plugin.cpp`）；`conversion/slice/` 与 `conversion/strided_slice_v2/` 是**两个不同目录**下的两个 Slice 类插件（前者为 NPUSlice，条目 88；后者为 Slice，条目 92）。

配套改造：每个插件文件对应 framework UT 目录新增/更新 UT 文件（**57 个 UT 文件**），其中 5 个目录需同步更新 `CMakeLists.txt`（`conversion/flatten`、`conversion/squeeze`、`conversion/unsqueeze`、`math/is_inf`、`random/dsa_random_uniform`）。

不在范围内的遗留：全仓 `*_onnx_plugin.cpp` 中仍 `#include "onnx_common.h"` 的仅 **2 个**（`math/shape/framework/shape_onnx_plugin.cpp`、`random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp`）——二者**已使用新接口**（`ParseParamsByOperatorFn`）但拖带 protobuf 依赖，且**均不在任务书 124 条目范围表内**，属既有状态，本次不改动。迁移后全仓 `*_onnx_plugin.cpp` 使用 `ParseParamsFn` 的文件数为 **0**。

## 多分支（双注册）插件改造要点

57 个文件中**有 16 个文件包含 ≥2 处 `ParseParamsByOperatorFn` 注册**：其中 **15 个是"同一 ONNX 算子、不同 opset、不同 parse 函数"的真分支**（迁移中两个 parse 函数都要处理），另 1 个是 `sequence` 的 6 个**不同**算子共用一套范式（非分支）。真分支清单：

| # | 插件文件 | 算子 | 分支形态 | 前支行为 | 后支行为 |
| --- | --- | --- | --- | --- | --- |
| 1–8 | `math/reduce_sum/framework/reduce_sum_onnx_plugin.cpp`·`…/reduce_log_sum_onnx_plugin.cpp`·`math/reduce_prod`·`math/reduce_min`·`math/reduce_max`·`math/reduce_mean`·`math/reduce_log_sum_exp`·`math/square_sum_v1/framework/reduce_sum_square_onnx_plugin.cpp` | ReduceSum / ReduceLogSum / ReduceProd / ReduceMin / ReduceMax / ReduceMean / ReduceLogSumExp / ReduceSumSquare | 属性 axes 支 vs 输入 axes 支 | `ParseParamsReduceXxx`：读 `axes` 属性（type=7）+ `keepdims`（type=2）+ `noop_with_empty_axes`，写入节点属性 | `ParseParamsReduceXxx13`：读同名属性并置 `input_size = GetNodeInputNum(op_src)`；`ParseOpToGraphXxx13` 按 `input_num` 分派——1 路时由属性造 `Const` 挂 axes，2 路时用 `op::Data(_data1)` 接输入 |
| 9 | `conversion/clip_by_value/framework/clip_onnx_plugin.cpp` | Clip | min/max 属性支 vs 可选输入支 | `ParseParamsClipV9`：读 `min`/`max` FLOAT 属性（缺省 ±3.402823e+38） | `ParseParamsClipV11`：由第 2/3 路**输入名是否为空串**推断 `no_min`/`no_max`（`num<2` → FAILED；`num==2` → `no_max=true`；两路皆空 → FAILED）；`ParseOpToGraphClipV11` 分派 ClipByValue / Maximum / Minimum |
| 10 | `conversion/squeeze/framework/squeeze_onnx_plugin.cpp` | Squeeze | opset 8–12 vs 13–18 | 读 `axes`（type=7）写 `axis` 属性；缺省返回 SUCCESS | `ParseParamsSqueezeV3` 空实现（axes 由框架按输入连接），返回 SUCCESS |
| 11 | `conversion/unsqueeze/framework/unsequeeze_onnx_plugin.cpp` | Unsqueeze | opset 8–12 vs 13–18 | 读 `axes`，**空列表返回 PARAM_INVALID** | `ParseParamsUnsqueezeV3` 空实现 |
| 12 | `conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp` | Slice | opset 8/9 vs 10–18 | `ParseParamSliceCall`：读 `starts`/`ends`/`axes` 属性，**ends 或 starts 为空 → FAILED** | `ParseParamSlice` 空实现（三者均为输入，走 StridedSliceV2） |
| 13 | `conversion/pad_v3/framework/pad_onnx_plugin.cpp` | Pad | opset 9 vs 11–18 | `parse_params_pad_v9`：**`pads` 属性缺失 → FAILED**；读 `pads`/`value`/`mode`，把 pads 交错为成对形式、置 `constant_values` | `parse_params_pad_v11`：**只解析 `mode`**，置 `paddings_contiguous=false`（pads/value 为输入） |
| 14 | `math/topk/framework/topk_onnx_plugin.cpp` | TopK | opset 9 vs 10–18 | `ParseParamsTopKV9`：`k` 为 **INT 属性**，`k == -1` → PARAM_INVALID，建 `CreateScalar(k, DT_INT32)` | `ParseParamsTopK`：`k` 为第 2 路输入，读 `axis`/`largest`/`sorted` 属性 |
| 15 | `math/minimum/framework/min_onnx_plugin.cpp` | Min | 4 组注册（`{8,9,10}` / `11` / `12` / `13–18`） | — | 4 个 parse 函数**逻辑完全相同**，仅 `original_type` 字符串不同 |

第 16 个：`common/src/framework/sequence_onnx_plugin.cpp` —— SequenceAt / SequenceConstruct / SequenceEmpty / SequenceErase / SequenceInsert / SequenceLength **六个不同算子**各 1 处注册，共用同一套范式，不构成"同算子多版本分支"。

`common/src/framework/similar_opp_onnx_plugin.cpp` 用 `REGISTER_CUSTOM_ONNX_OP(ONNX_OPP, ATC_OPP, FUN)` 宏批量展开，共 **578 条注册**，覆盖 **63 个 ONNX 算子**（对应任务书第 18–80 条目；多数算子按 opset 版本注册 11 次），各注册的 parse 行为**完全一致**（均为空实现/固定映射），故不构成分支，但其**注册面**是范围内最大的一处，两轮比对需对其 63 个算子逐一取模。

## 实验性 Max 插件与构建开关

任务书第 96 条目 `Max` 位于 `experimental/math/asinh/framework/max_onnx_plugin.cpp`。仓内 CMake 以 `option(ENABLE_EXPERIMENTAL "Enable experimental module" OFF)` 控制，`experimental/` 与 `math`/`conversion`/`random` 是**互斥的 `add_subdirectory` 分支**（开启 `--experimental` 会**替换**而非**叠加**常规目录）：

| 构建配置 | 产物内含 | 结果 |
| --- | --- | --- |
| 默认（`--onnxplugin`） | `math` + `conversion` + `random` + `common` | 不含实验性 Max；`math/maximum` 的 `Max` 插件生效 |
| `--experimental` | `experimental/` + `common` | 含实验性 Max；常规目录不在产物内 |

因此实验性 Max 需要**独立一套 pre/post `.so`** 做专项两轮比对，不能与常规两轮共用一次构建。若该插件与 `math/maximum` 注册同名 op 导致抢占，则如实记录为"同名注册竞争"并说明其覆盖由 UT 承担。

## 兼容性设计

1. **模型/转换兼容**：改造仅替换属性解析路径，`OriginOpType` 列表、`ImplyType`、`ParseOpToGraphFn` 等注册项不变，ATC 对同一模型应产出等价的图与 om（由两轮比对验证）；
2. **旧插件自验兼容**：为完成"迁移前后"两轮比对，迁移前插件在本环境加载报 `undefined symbol: …ge::onnx::AttributeProto…` —— CANN 9.1.0 的 `libgraph_base.so` 未导出 `AttributeProto` 的拷贝构造重载，而迁移前插件用 protoc 25.1 生成的 inline 代码会引用它。按任务包 `test_cases/verify_om/compat/` 的方式以兼容库（`libge_onnx_compat.so`）重新链接（仅用于自验基线，**不进仓库**）；迁移后插件不需要该库。实测对照：

   ```
   迁移前 liboponnx_plugin_math.so：5 个未定义 ge::onnx::* 符号
     Arena::CreateMaybeMessage<AttributeProto>(Arena*) / AttributeProto::MergeImpl(...)
     AttributeProto::AttributeProto(AttributeProto const&) / AttributeProto::~AttributeProto()
     typeinfo for ge::onnx::NodeProto
   → DT_NEEDED 多出 libge_onnx_compat.so
   迁移后 liboponnx_plugin_math.so：0 个；DT_NEEDED 无 compat 库
   ```

3. **编译兼容**：不新增第三方依赖（nlohmann/json 用仓内 third_party 自带版本）；公共头以 `math_onnx_plugin_util.h` 命名，避免与其他算子仓重名；
4. **UT 兼容**：UT 直接以 `ge::Operator` 构造输入（`op_src.SetAttr("attribute", ge::AscendString(json.c_str()))`），不再构造任何 protobuf 对象——这正是解耦后可测性提升的体现。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 迁移前后两轮 om 转换 + 推理比对：① 随机算子固定 seed 逐位一致；② 确定性算子 NaN/正负 0 等逐字节一致；③ 非确定性算子满足生态混合容差标准（不可复现用例按用例包声明档位） | 任务书 3.2 节；生态算子开源精度标准混合容差标准（commit `a5e8e716743b20bfb2ffa16e0213c4aa73b823b3`） |
| 性能标准 | 不涉及。本任务为 Host 侧等价重构，不改变算子计算逻辑与模型转换产物结构，不引入新的算子实现，无新增算力开销，不设性能验收指标 | 任务书 3.3 节 |
| 内存标准 | 不涉及。同上，不新增显存/内存占用，无内存验收指标 | 任务书 3.4 节 |

## 测试设计

### framework UT（可测性设计）

- 构建：开启仓内 `OP_HOST_UT` 开关（`bash build.sh -u --ophost`）；
- 入口：`--gtest_filter='Onnx*PluginTest.*'`；
- 用例组织：按**插件文件**一一对应（57 个 UT 文件），套件名统一 `Onnx*PluginTest`；
- 用例直接 `#include "../../../framework/xxx_onnx_plugin.cpp"`（白盒直调静态解析函数），**不构造任何 protobuf 对象**——构造输入只需 `ge::Operator op_src; op_src.SetAttr("attribute", ge::AscendString(json.c_str()));`。这是解耦后 UT 成本下降的直接度量：用例不再依赖 `.proto` 生成代码，只依赖 JSON 字面量；
- 每个算子覆盖四类场景（对应任务书 3.5 节入参生成规则）：
  1. 属性齐全的正常解析；
  2. 属性缺失（使用默认值，返回 `SUCCESS`）；
  3. 非法 attribute JSON（返回 `FAILED`；无属性插件豁免）；
  4. 边界/校验场景（如 `f` 键缺省按 0.0、`type` 不符按缺失处理、空列表/负值原样透传、Cast 不支持 dtype 返回 `FAILED`、TopK `k=-1` 返回 `PARAM_INVALID`）；
- 另含 `Ignore*WithMismatchedType` 系列与"变异判错"自查：故意写反判据（如精确比较写成容差比较）应能被用例抓住。

### om 转换精度验证（两轮比对）

| 项 | 设计 |
| --- | --- |
| 用例来源 | 任务包 56 个 `gen_model_*.py` 生成脚本（覆盖 57 个插件文件，两份 Max 插件共用一个脚本）；另按任务书 3.5 节"开发者自行补充数据类型、可选输入输出、数据格式、属性取值的未覆盖场景用例"的要求，自编补充用例 |
| 模型输入 | numpy 随机（μ=0、σ=1），属性取可区分值（如 Cast 的 `to=10`、Squeeze 的 `axes=[2]`、Flatten 的 `axis=2`），保证解析错误在输出 dtype/shape/数值上可观测 |
| 规模控制 | 遵循任务书"每算子用例非必要小于 50 个"的减本意 |
| 流程 | 对每个用例：① 生成 model.onnx + input_*.npy；② `ASCEND_CUSTOM_OPP_PATH` 注入对应轮次插件后 ATC 转换（两轮**同模型、同输入、同 ATC 参数**）；③ pyACL 推理落盘 out_*.npy；④ 比对两轮输出 |
| 判据 | 默认逐位一致（md5）；非确定性/平台差异用例按 meta.json 声明的档位判定；pre 轮转换失败的用例，若两轮一致失败且报错摘要匹配声明，则按"两轮一致失败"通过（属任务书 3.2 节用例包修订许可范围，在验收说明中说明） |
| 精度模式 | 主模式 `--precision_mode=must_keep_origin_dtype`（FP32 用例保持 ONNX 精度语义）；对平台侧不支持该模式的算子回落默认模式重跑，两轮同参数 |
| 部署证据 | 每轮每用例保留 `plugin_evidence_*.txt`，含 `LoadCustomOpLib` 与 `Plugin of … already registered, skip` 证据行，证明生效的是注入插件而非内置插件 |
| 补充用例约定 | 自编补充用例不修改用例包任何文件，新建独立目录并以薄封装复用包内生成脚本或其约定（`--workdir` 下产出 `model.onnx`/`input_*.npy`/`meta.json`；`input_*.npy` 按文件名排序对应模型输入顺序；以 initializer 形式挂载的常量不产输入文件）；重点补包内未覆盖的 **opset 分支、可选输入输出、属性取值与 dtype 变体** |

### 编译与自查（可维性设计）

- `bash build.sh --onnxplugin -j<N>` 全量编译通过；
- 单算子迁移自查清单逐项填写：include 替换、签名/注册宏/属性解析/name 获取迁移、缺失属性默认值返回 `SUCCESS`、非法 JSON 返回 `FAILED`、新增/更新用例、`git diff --check` 通过；
- 静态自查：全仓 `*_onnx_plugin.cpp` 中 `ParseParamsFn` 出现次数为 0、`ge_onnx.pb.h` 引用次数为 0（范围内）；
- 动态自查：迁移后 `liboponnx_plugin_math.so` 的未定义 `ge::onnx::*` 符号数为 0、DT_NEEDED 中无 protobuf/compat 库。

## 兼容性分析

- 接口兼容：新接口由 GE 框架提供（`register/register.h`），插件产物（`liboponnx_plugin_math.so`）的注册 ABI 不变，ATC 使用方式不变；
- 行为兼容：等价重构通过两轮比对（同一 onnx 模型、同一输入、同一 ATC 参数、迁移前后插件）逐位验证，覆盖全部迁移算子；
- 存量兼容：`onnx_common.h` 保留，范围外已使用新接口的两个插件（`math/shape`、`random/stateless_bernoulli`）不受影响；
- 不引入新依赖：JSON 解析使用仓内 third_party 自带 nlohmann/json。

## 风险与规避

| 风险 | 规避措施 |
| --- | --- |
| 属性 JSON 与 protobuf 取值语义的细节差异（`f` 字符串承载、`f` 键缺省=0、type 不符按缺失） | 逐算子与迁移前源码比对 + 专项 UT 用例（见"测试设计·场景 4"）+ 两轮比对实测取证 |
| 双注册分支只改一支、另一支漏改 | 已静态清点 16 个多注册文件（15 真分支 + sequence），逐文件核对两处注册均已替换；om 侧对每个真分支各取模 |
| `GetNodeInputNum` 读的是**含 initializer 名**的 `input` 列表，与迁移前 `node->input_size()` 语义一致但易误解为"实际输入个数" | 头文件注释显式声明；Reduce 家族按 `input_num` 分派构图的分支以"axes 作为 initializer 输入"的用例实测验证（输出 shape 非全维归约即证明走了 2 路分支） |
| 自定义算子包未安装，NPU 自定义算子用例转换失败 | 按用例包既有设计走"两轮一致失败"判据，逐例在验收说明申报 |
| `precision_mode` 个别算子不支持 | 该用例回落默认模式重跑两轮（同参数），在验收说明列出 |
| 实验性 Max 不在默认构建内，且 `--experimental` 替换而非叠加常规目录 | 另建一套 `--experimental` 的 pre/post `.so` 做专项两轮比对；若同名 op 被抢占则如实记录并说明覆盖由 UT 承担 |
| 与同机其它任务争抢 NPU | 只用空闲设备分片并行，执行前后各查一次 `npu-smi` |
| 全量两轮比对耗时 | 按 NPU 分片并行；`similar_opp`（63 模型）单独分片 |

## 设计兑现情况（实测结论）

本节记录上述设计的实际兑现结果，明细与证据见《1 自验证步骤说明.md》《2.1 精度自验证报告.xlsx》
《2.2 精度自验证日志.log》与《验收说明附件-om用例与用例包修订.md》。

| 设计项 | 实测结果 |
| --- | --- |
| 接口替换覆盖 | 57 个插件文件 / 124 个算子条目全部由 `ParseParamsFn` 改为 `ParseParamsByOperatorFn`；插件相关文件中旧接口 **0** 次、新接口 **91** 次 |
| protobuf 解耦达成 | 迁移前 `liboponnx_plugin_math.so` 有 **5** 个未定义 `ge::onnx::*` 符号、`DT_NEEDED` 含兼容库 `libge_onnx_compat.so`；迁移后同项为 **0**、不再依赖兼容库（`nm -DC`，脚本 `tools/check_decoupling.sh`） |
| framework UT | 默认构建：`Onnx*PluginTest.*` **648/648** PASS，全量 **2871/2871** PASS；实验性构建：`Onnx*PluginTest.*` **188/188** PASS（含 `OnnxMaxExperimentalPluginTest` 9 例），全量 **908/908** PASS |
| om 两轮比对 | 十个分片共 **185** 个用例，**PASS 185 / FAIL 0**；195 个输出张量中 193 个逐字节一致，1 例按申报容差档、1 例按申报行为档判定 |
| 用例覆盖面 | 包内 56 个生成脚本全量（119 例：54 脚本各 1 例、`clip` 2 例、`similar_opp` 63 例）+ 自编补充 65 个生成脚本（64 例，瞄准 opset 双分支与属性/dtype/可选输入输出变体） |
| 一致失败用例 | 10 例申报"两轮一致失败"（自定义算子包未安装 / 平台 IR 前向兼容限制 / 内核源码缺陷），逐例在验收说明附件列出并附复现命令；非预期失败 **0** |
| 实验性 Max | 独立 `--experimental` 构建：插件两轮 om 输出逐字节一致（2 例）+ UT 9 例全过；构建中另发现并修复了该文件在 `-Werror=unused-variable` 下无法编译的缺陷（`(void)attrs;`） |
| 性能/内存 | 按任务书 3.3/3.4 节判定"不涉及"（Host 侧等价重构，不涉及 kernel/tiling/原型，无 Device 侧执行路径变化） |

> 一处与设计预设不同的地方如实记录：实验性 Max 的"同名注册竞争"风险**未发生**——该
> `--experimental` 插件 `.so` 内只编入 4 个 onnx 插件源（squeeze / unsqueeze / 实验性 max /
> stateless_bernoulli），不含 `math/maximum`，故其 UT 覆盖与 om 覆盖均已独立取得，无需回落到
> "覆盖由 UT 承担"的兜底表述。

---

**TeamName：Irene-**
