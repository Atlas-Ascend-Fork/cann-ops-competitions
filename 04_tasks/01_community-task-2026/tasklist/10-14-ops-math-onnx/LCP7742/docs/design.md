# ops-math 算子 ONNX 插件 protobuf 解耦迁移设计文档

> 任务：2026 年 10 月 CANN 社区任务 · 任务编号 14《10月社区任务-ops-math算子onnx适配》
> 目标仓库：https://gitcode.com/cann/ops-math
> 提交目录：04_tasks/01_community-task-2026/tasklist/10-14-ops-math-onnx/LCP7742/docs/design.md
> 设计基线：ops-math master（任务书静态扫描基线 commit 91c6c1715）
> 文档版本：v1.1.0（2026-10-23）

---

## 1. 需求背景（required）

### 1.1 需求来源

本任务来自 2026 年 10 月 CANN 社区任务的任务 #14《10月社区任务-ops-math算子onnx适配》，任务书为《ops-math 算子 onnx 插件 protobuf 解耦任务书》（`ops-math_onnx_task_doc.md`）。要求把 `ops-math` 仓仍在通过 `ParseParamsFn` 注册参数解析函数的 **124 个 ONNX 算子插件条目（57 个插件文件）** 迁移到不依赖 protobuf 的 `ParseParamsByOperatorFn` 接口：属性解析由“遍历 protobuf attribute 并读取 `i/f/s/ints` 字段”改为“`op_src.GetAttr("attribute")` 取 JSON 字符串后用 nlohmann/json 解析”，使插件代码彻底摆脱对 protobuf 生成代码（`ge_onnx.pb.h`）及 protobuf 运行库的依赖，同时**保持插件原有解析语义与功能不变**（等价重构）。

设计依据：

- **任务输入**：任务书（第 1 节迁移范围表、2.1 功能实现要求、2.3 接口定义、2.4 attribute JSON 字段说明、第 3 节验收标准、第 4/5 节交付件与 PR 要求）；
- **迁移契约**：随交付保存的团队内部《迁移规范》文档（v1.2.3，不在 ops-math 仓内；2026-10-15 一致性总检后修订），§2 统一迁移模式与 include 模板、§4 通用属性契约、§5 例外条款、§6 Cast 判据修正、§7 UT 放置规则；该文为本团队内部设计留档，其节号仅供溯源，本文档结论自洽；
- **行为契约**：任务配套 57 个 framework UT 文件（`test_cases/ops-math/ut/`）的 gtest 断言；
- **参考实现**：仓内 @91c6c1715 已合入的 9 个 `ParseParamsByOperatorFn` 参考实现（见 1.2.4），任务书点名的样板为 `reverse_sequence_onnx_plugin.cpp`、`conversion/strided_slice/framework/npu_indexing_onnx_plugin.cpp`、`random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp`；
- **公共工具**：`common/inc/framework/math_onnx_plugin_util.h`（protobuf-free，见 1.2.4）；
- **关联 issue**：[ops-math issue 2892](https://gitcode.com/cann/ops-math/issues/2892)（含关联 PR：迁移 onnx 插件 ParseParams API 并补 framework UT、迁移 ReverseSequence 插件、npu_indexing 插件迁移）；
- **验收参考**：混合容差标准（[opbase](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)）、CANN ONNX 算子开发说明（`ai_framework_adaptation/onnx_framework/call_examples.md`）、nlohmann/json 文档。

本设计文档以 PR 形式提交到 `cann/cann-ops-competitions` 仓的任务提交目录（路径见文档头「提交目录」）。

### 1.2 背景介绍

#### 1.2.1 现状：插件参数解析与 protobuf 强耦合

`ops-math` 的 ONNX 适配插件通过 `REGISTER_CUSTOM_OP(<OpName>).ParseParamsFn(ParseParams<Op>)` 注册参数解析函数。该函数入参类型是 `google::protobuf::Message`，框架实际传入 `ge::onnx::NodeProto`，插件先 `reinterpret_cast`/`dynamic_cast`，再遍历 protobuf attribute 按 `attr.i()`（INT）/`attr.f()`（FLOAT）/`attr.s()`（STRING）/`attr.ints()`（INTS）取值，最后写入目标算子属性。

**依赖链**：`*_onnx_plugin.cpp` → `#include "onnx_common.h"` → `#include "onnx/proto/ge_onnx.pb.h"` → protobuf 生成代码与运行库。`onnx_common.h` 在基线的唯一残留作用就是拉入 `ge_onnx.pb.h`（《迁移规范》（项目内部文档）§2.1/D8），因此每个插件的这一行 include 都会把整套 protobuf 生成代码带进编译单元，插件源码也随之直接使用 `ge::onnx::NodeProto`、`ge::onnx::AttributeProto` 等类型。

迁移前典型形态之一（`math/cast/framework/cast_onnx_plugin.cpp`，行号 11–45、47–61，注册段略）：

```cpp
#include "onnx_common.h"
#include "math/cast/op_graph/cast_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsCast(const Message* op_src, ge::Operator& op_dst)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool bfind_to = false;
    for (auto attr : node->attribute()) {
        if (attr.name() == "to") {
            bfind_to = true;
            int onnx = attr.i();
            int32_t om = GetOmDtypeFromOnnxDtype(onnx);
            if (om == -1) {   // 永不成立的失效判据，见 3.2.6
                OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
                return FAILED;
            }
            op_dst.SetAttr("dst_type", om);
            break;
        }
    }

    if (!bfind_to) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Message op_src do not have attribute to.");
        return FAILED;
    }
    return SUCCESS;
}
```

另两种典型形态（节选，完整代码见基线快照）：

- **循环 + type 校验 + 默认值**（`conversion/flatten/flatten_onnx_plugin.cpp:29-35`）：`int axis = 1; for (auto attr : node->attribute()) { if (attr.name()=="axis" && attr.type()==ge::onnx::AttributeProto::INT) { axis = attr.i(); } } op_dest.SetAttr("axis", axis);`
- **浮点属性 + 节点名 + 动态 IO + 拆图函数**（`common/src/framework/random_normal_like_onnx_plugin.cpp:29-56`）：先 `DynamicInputRegister("x", 1)`、`DynamicOutputRegister("y", 1)`、写 `original_type`，再在 `node->attribute()` 循环里 `mean = attr.f(); scale = attr.f(); seed = (int)attr.f();`，最后 `op_dest.SetAttr("name", node->name());`；同文件还注册了 `.ParseOpToGraphFn(ParseOpToGraphRandomNormalLike)`。

这三类形态覆盖了全部改动点：**入参类型、注册方式、属性来源与取值方式、节点名来源、输入/输出数量来源、include 依赖**。

#### 1.2.2 三类问题

与 protobuf 生成代码强耦合带来三类问题（任务书第 1 节原文口径）：

1. **与 protobuf 生成代码强耦合**：插件源码依赖 `ge_onnx.pb.h`，proto 定义一旦变动会波及全部插件；
2. **编译依赖重**：编译插件需拉入 protobuf 头文件与生成代码，编译链路长；
3. **UT 成本高**：单测必须构造 protobuf 对象（`NodeProto`、`AttributeProto`），用例编写与维护成本高。

工程副作用：插件“接口形状”由 protobuf 决定，无法按 ONNX 算子语义演进；属性取值分散在 `attr.i()/f()/s()/ints()` 各分支，缺乏可被 UT 直接构造的统一输入表示，非法输入与缺字段边界难以覆盖。

#### 1.2.3 迁移目标与范围

**迁移目标（5 项替换 + 1 项语义要求）**：

1. 签名替换：`ParseParamsXxx(const Message* op_src, ge::Operator& op_dst)` → `ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dst)`，删除 protobuf 转换与判空；
2. 注册替换：`.ParseParamsFn(F)` → `.ParseParamsByOperatorFn(F)`，其余注册要素（`REGISTER_CUSTOM_OP` 名称、`OriginOpType` 条目与顺序、`ParseOpToGraphFn` 有无与函数名、`ImplyType`）一字不动；
3. 属性解析替换：`node->attribute()` 遍历 → `GetAttr("attribute")` + nlohmann/json（契约见 3.1）；
4. 依赖清理：删除 `onnx_common.h`（及其拉入的 `ge_onnx.pb.h`），按 3.2.2 模板替换 include；
5. 节点名与计数替换：`node->name()` → `GetOpName(op_src)`；`node->input_size()/output_size()` → 源算子输入/输出数量（3.2.4）；
6. 语义要求：迁移前后对同一 ONNX 节点写入 `op_dest` 的属性序列逐项一致，各分支返回值（`SUCCESS`/`FAILED`/`PARAM_INVALID`）一致；**显式例外分三类（见 3.5：E-1 Cast 判据修正、E-2 B 类 4 文件非法 JSON 新增 `FAILED`、E-3 5 文件空名回落）**。

**范围**：对仓内源码静态扫描得到的、仍使用 `ParseParamsFn` 的全部插件，共 **57 个文件 / 124 个条目**（同一算子多个 opset 合并计一条）。以任务书第 1 节范围表为唯一基准；开发前用 `git grep -l '.ParseParamsFn(' 91c6c1715 -- '*_onnx_plugin.cpp'` 复核得 57 个文件，与范围表集合完全相等（《迁移规范》（项目内部文档）§0.1）；基线中 `ParseParamsByOperatorFn` 恰好命中 9 个已迁移文件，故待迁移数 = 57。按目录归集见 2.3。

**不在范围内**：算子 kernel/tiling/op_api 的修改；`ParseOpToGraphFn` 拆图函数体逻辑；TensorFlow 插件与已迁移插件；与解耦无关的重构/格式化/注释/license 改动；除 Cast 判据外的新增校验。

#### 1.2.4 参考实现与公共工具

**仓内已合入的参考实现（@91c6c1715 恰好 9 个）**：`common/src/framework/{correlation,npu_rotary_mul,reverse_sequence}_onnx_plugin.cpp`、`conversion/npu_format_cast/framework/npu_format_cast_onnx_plugin.cpp`、`conversion/strided_slice/framework/npu_indexing_onnx_plugin.cpp`、`math/cast/framework/npu_dtype_onnx_plugin.cpp`、`math/one_hot/framework/one_hot_d_onnx_plugin.cpp`、`math/shape/framework/shape_onnx_plugin.cpp`、`random/stateless_bernoulli/framework/bernoulli_onnx_plugin.cpp`。任务书点名的 3 个样板：

| 样板文件 | 示范写法 |
|---|---|
| `reverse_sequence_onnx_plugin.cpp` | 纯参数解析：先写默认值，再 `GetAttr("attribute")` + `contains/is_array` 守卫 + try/catch，不写 name |
| `npu_indexing_onnx_plugin.cpp` | 含 `ParseOpToGraphFn`：`stub_ops.h` + `graph/graph.h` + 具体 `*_proto.h` 的 include 组合；`GetName` 取源名、空则回落目标算子名 |
| `bernoulli_onnx_plugin.cpp` | FLOAT 读取：`std::string seed_string = attr["f"]; seed = atof(seed_string.c_str());`（浮点标量在 JSON 中以**字符串**承载） |

> 两条边界（《迁移规范》（项目内部文档）§2.4.0、§8-D4）：① `math/shape` 与 `bernoulli` 两个参考实现自身仍 include `onnx_common.h` 并使用 `ge::onnx::AttributeProto::*`，属“遗留参考”，**不得照抄其 include 与 protobuf 常量**；② bernoulli 的 `SetAttr("name", GetOpName(op_dest))` 是唯一一处写成目标算子名的先例，**不得照抄**为全局模板。任务书/用例包提到的 elu、leaky_relu、bounding_box_decode 在本仓 @91c6c1715 不存在（《迁移规范》§8-D2 登记为包内失效引用），不据此推断行为。

**protobuf-free 公共工具头 `math_onnx_plugin_util.h`**（基线已存在、无 protobuf 依赖）：`GetOpName`（`GetName` 失败返回 `"None"`）；`GetOmDtypeFromOnnxDtype`/`onnx2om_dtype_map`/`DataTypeOnnx`（未命中返回 **`ge::DT_UNDEFINED`**，见 3.2.6）；`Vec2Tensor`/`CreateScalar`（构造张量/标量属性）；`ChangeFormatFromOnnx`（改写输入/输出 format）。迁移中在该既有头内追加三项 protobuf-free 能力（**未新增公共头文件**）：`GetSourceInputNum`/`GetSourceOutputNum`（真实 ATC 下由 `input`/`output` 列表属性推导数量，UT 回退 `GetInputsSize()/GetOutputsSize()`）与 `enum OnnxAttrProtoType`（`ONNX_ATTR_FLOAT/INT/STRING/FLOATS/INTS = 1/2/3/6/7`，替代 protobuf 枚举的数值比对）。

### 1.3 接口与语义基线

迁移后签名（保持各文件原函数命名，如 `ParseParamsCast`、`ParseParamsFlatten`、similar_opp 的宏命名 `ParseParams<FUNC>`）：`static domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest);`

注册（仅替换参数解析注册行，其余逐字保留）：`REGISTER_CUSTOM_OP("${OpName}").FrameworkType(ONNX).OriginOpType({<条目>})` 与 `.ParseParamsByOperatorFn(ParseParams${OpName})`，`OriginOpType` 条目与顺序、`ParseOpToGraphFn`（有无与函数名）、`ImplyType` 原样。

**语义基线**：迁移前后对同一 ONNX 节点、同一输入，解析函数写入 `op_dest` 的属性序列必须**逐项一致**（键名、值、写入次序、动态 IO 注册、`original_type` 均一致），仅解析实现方式变化；各分支返回值一致。**例外集合见 3.5**（E-1/E-2/E-3）；其中 **E-1** 为 `math/cast/framework/cast_onnx_plugin.cpp` 的 dtype 失效判据修正（3.2.6）——不改变合法模型行为，只把“不支持的 dtype”的失败从引擎选择阶段前移到解析阶段；E-2 的新增失败面仅限非法 JSON 输入，E-3 只影响空名节点。

---

## 2. 需求分析（required）

### 2.1 需求描述

以**等价重构**完成 57 个文件的 protobuf 解耦，并补齐每个插件的 framework UT，用迁移前/后两轮 om 转换与推理证明行为不变：

1. **接口与注册**：按 1.3 完成 57 个文件（124 条目）替换，删除 protobuf 转换与判空；
2. **属性解析**：`GetAttr("attribute")` → JSON → 按原语义写 `op_dest`；FLOAT 按字符串承载、`f` 缺省 = 0.0；非法 JSON → `FAILED`（少数恒 SUCCESS 文件除外，见 3.2.3）；
3. **节点名与计数**：原本写 name 的文件改为源算子名（空名按该文件 UT 契约处理）；原本不写的**不新增**；输入/输出数量改为“列表属性优先、UT 回退”；
4. **依赖清理**：目标文件不 include `onnx_common.h`、不出现 `ge::onnx::`；`.ParseParamsFn(` 残留为 0；
5. **UT 补齐**：每文件配套 `test_<op>_onnx_plugin.cpp`，用 `ge::Operator` + `SetAttr("attribute", <JSON 串>)` 构造输入、**不构造 protobuf 对象**，覆盖正常/缺失默认值/非法 JSON/边界校验四类；
6. **验证**：framework UT 全通过；两轮 om 逐位比对一致（随机项先做旧/旧重复性）；混合容差辅助判据 + 输出 dtype/shape 校验；
7. **约束**：不碰 kernel/tiling/op_api，不改拆图逻辑，不做无关重构；性能/内存不涉及（Host 侧等价重构）；
8. **环境**：A2 与 Ascend 950 任一硬件通过即可（任务书口径；本设计实测环境为 Ascend910B3 ×4 + CANN 9.1.0）；CANN > 9.0.0。

### 2.2 外部组件依赖

| 组件 / 依赖 | 用途 | 版本与来源 | 迁移后状态 |
|---|---|---|---|
| nlohmann/json | attribute JSON 解析 | **仓内 third_party 自带版本**，不新增/不升级/不另引其他 JSON 库 | 新增头文件依赖（原由 protobuf 承担解析） |
| `math_onnx_plugin_util.h` | protobuf-free 公共工具（见 1.2.4；本次在既有头内扩展 3 项） | 仓内既有文件 | 复用/扩展，**不新增公共头** |
| `graph/operator.h`、`graph/graph.h` | `ge::Operator`/`ge::Graph`/`GetName/GetAttr/GetInputsSize` | CANN 图接口头 | 新增直接依赖 |
| `register/register.h`、`log/log.h` | 注册宏、`OP_LOGE/OP_LOGW` | 仓内/CANN 框架头 | 保持/显式补 include |
| `stub_ops.h`、`op_graph/*_proto.h`、`op_math_proto_extend.h` | 拆图函数构造子图 | 仓内（仅含 `ParseOpToGraphFn` 的文件） | 保持 |
| protobuf / `ge_onnx.pb.h` / `onnx_common.h` | 原参数解析、节点名、计数来源 | 原依赖 | **移除**：57 个目标文件残留 0（仓内 `math/shape`、`random/stateless_bernoulli` 两个非目标插件仍 include，不在本任务；`onnx_common.h` 保留给 TF 插件） |
| CANN Toolkit（ATC、pyACL） | om 转换（`--framework=5`）与推理 | > 9.0.0（实测 CANN 9.1.0） | 验证期依赖 |
| Python 3.8+ / onnx / onnxruntime / numpy | onnx 模型与 golden 生成 | 实测组合见 4.3；声明满足下限即可 | 验证期依赖，不进插件构建 |

约束：不引入新第三方库；nlohmann/json 只用仓内版本；不为解耦新增公共头文件。

### 2.3 迁移范围清单（57 文件 / 124 条目）

按任务书第 1 节范围表逐行归集（同一算子多个 ONNX opset 合并计 1 条）：

| 目录大类 | 文件数 | 条目数 | 代表算子 | 主要特征 |
|---|---:|---:|---|---|
| `common/src/framework/` | 14 | 81 | similar_opp（63 算子）、sequence（6 算子）、eyelike、einsum、reduceL1/L2、phony_concat/phony_split、repeat_with_pad、random_normal_like、random_uniform_like、concat_from_sequence、split_to_sequence、deformable_offsets | 宏批量注册（578 条注册语句）与“文件 × 多算子”聚合；多数含 `ParseOpToGraphFn` 与动态 IO |
| `conversion/*/framework/` | 14 | 14 | concat、flatten、squeeze、transpose、depth_to_space、space_to_depth、trilu、clip、reshape、pad、split、unsqueeze、slice、npu_slice | 布局/形状类属性为主；`clip`/`pad`/`squeeze`/`slice`/`unsqueeze` 双注册；多文件含拆图函数 |
| `math/*/framework/` | 25 | 25 | cast、addcmul、xor、cumsum、isinf、maximum/minimum、mod、onehot、bitshift、sign_bits_pack/unpack、topk、argmax/argmin、reduce 系列、mean/sum | 单/少量属性；reduce 家族双注册且写张量属性；`maximum`/`minimum`/`xor` 不读属性但需 JSON 合法性校验 |
| `experimental/math/asinh/framework/` | 1 | 1 | Max（实验路径） | 与 `math/maximum` 共用一份 om 脚本、UT 独立；**两个同名 `max_onnx_plugin.cpp` 必须按全路径区分** |
| `random/*/framework/` | 3 | 3 | dsa_random_uniform、stateless_random_normal_v2、stateless_sample_multinomial | 随机算子，含 seed/dtype/shape 属性，按 4.2 随机判据处理 |
| **合计** | **57** | **124** | — | — |

范围声明：

1. **以任务书第 1 节迁移范围表为唯一基准**；开发前 `git grep -l '.ParseParamsFn(' 91c6c1715 -- '*_onnx_plugin.cpp'` 复核得 57 个文件，与范围表集合完全相等（《迁移规范》（项目内部文档）§0.1）。
2. 上表 5 行 + 合计行为完整归集，开发与汇总按**全路径**逐一核对（不得只对组数）。`common` 81 = similar_opp 63 + sequence 6 + 12 个单条目文件；合计 14 + 14 + 25 + 1 + 3 = **57**、81 + 14 + 25 + 1 + 3 = **124**。
3. **两个同名 `max_onnx_plugin.cpp` 按全路径区分**：`math/maximum/framework/max_onnx_plugin.cpp`（Max 常规路径）与 `experimental/math/asinh/framework/max_onnx_plugin.cpp`（实验路径）是两个独立文件，共用 `gen_model_max.py`，但 UT/属性表/特殊语义不同，不得合并计数或互相替代。
4. 上表按**目录大类**归集，不是迁移批次；批次中 Flatten/Transpose/Addcmul/Cast 属 P1 试点，故“P4 math”批次为 24 文件（《迁移规范》（项目内部文档）§3.0），与 `math/*` 25 文件的差异仅来自试点与实验路径的归属口径。
5. 若开发期间仓内源码变动导致与本表不符，按任务书要求提交 issue，不在本设计内私改范围。

### 2.4 需求拆解

1. **接口迁移**：57 个文件签名替换为 `(const ge::Operator&, ge::Operator&)`；删除 `dynamic_cast/reinterpret_cast<NodeProto*>`、判空与 `Message*` 代码；`.ParseParamsFn(` → `.ParseParamsByOperatorFn(`。
2. **属性解析迁移**：按 3.1/3.2.1/3.2.3 逐文件改写；FLOAT 字符串承载、`f/i/ints/s` 缺省语义、type 校验、重复属性、失败分支逐文件保持。
3. **节点名迁移**：**35 个文件写 `SetAttr("name", <节点名>)`**（另 **22** 个不写，其中多为只写 `original_type`），迁移后写 name 的文件改为源算子名语义（5 个源名空回落目标名，`sequence` 写空串），不写的不得新增；写入位置与失败判据相对次序不变（源码口径 35/22，UT 口径见 3.2.4（1））。
4. **节点计数迁移**：动态 IO 与数量改用 `GetSourceInputNum/GetSourceOutputNum`；回调内目的算子的 `GetInputsSize()` 保持原样。
5. **依赖清理**：include 替换（3.2.2）；目标文件 `onnx_common.h`/`ge_onnx.pb.h`/`ge::onnx::` 残留 0；注册三要素逐字保留。
6. **例外修正**：Cast 判据改判 `ge::DT_UNDEFINED`（3.2.6）；排查其余“判据与工具函数真实返回不符”并在自查表说明。
7. **UT 补齐**：57 个 framework UT 按《迁移规范》（项目内部文档）§7 放置（`<prefix>/<op>/tests/ut/framework/`、`common/tests/ut/framework/`），include 迁移后源文件，`--gtest_filter='Onnx*PluginTest.*'` 全通过。**用例数口径必须在第 4 章显式分离、不得混用**：`624` = 57 个 UT 文件静态 `TEST/TEST_F/TEST_P` 宏总数（台账口径）；`612` = 任务书 §3.5 所记用例数（引用口径）；按 gtest 输出逐文件映射实得 `615`（差 3 例源于 2 处 suite 映射：onehot→OneHotD 8、sequence→ReverseSequence 8，真实 suite 为 9/10）；`648` = `Onnx*PluginTest.*` 实跑用例数（XML `tests=648 failures=0`）= 615 + 33（8 个非目标 suite）。不得把 648 写成“57 文件用例数”（详见 4.2）。
8. **验证与交付**：两轮 om 逐位比对（56 个模型脚本覆盖 57 文件，两份 Max 共用脚本）+ 混合容差辅助判据 + `git diff --check`（声明 CRLF 前提）+ `bash build.sh --onnxplugin -j8` + 单算子自查表归档。

---

## 3. 详细设计（required）

### 3.1 插件分析：attribute JSON 契约

迁移后通过 `op_src.GetAttr("attribute", attrs_string)` 拿到 JSON 字符串（GE 在 ONNX 解析阶段序列化 `AttributeProto` 得到）。顶层是对象，属性数组位于键 `attribute` 下，属性对象键名与 `AttributeProto` 字段同名，形如 `{"attribute":[{"name":"to","type":2,"i":10},{"name":"alpha","type":1,"f":"0.25"},{"name":"mode","type":3,"s":"constant"}]}`（各形态见下表）。

#### 3.1.1 分形态契约

| 属性形态 | `type` | JSON 示例 | 迁移后读取 | 键缺省取值 |
|---|---:|---|---|---|
| 标量整数 INT | 2 | `{"name":"to","type":2,"i":10}` | `attr.value("i", 0)`（或 `contains("i") ? get<int64_t>() : 0`） | **0**（不是算子默认值） |
| 标量浮点 FLOAT | 1 | `{"name":"alpha","type":1,"f":"0.25"}`（**`f` 以字符串承载**） | 先取 `std::string`，再 `std::stof`/`atof` 转 float（见 3.1.2） | **0.0**（不是算子默认值） |
| 字符串 STRING | 3 | `{"name":"mode","type":3,"s":"constant"}` | `attr.value("s", std::string())` | **空串** |
| 整数列表 INTS | 7 | `{"name":"axes","type":7,"ints":[-1,2]}` | `attr["ints"]` 数组逐元素取值（元素类型按文件原样） | **空列表** |
| 浮点列表 FLOATS | 6 | `{"name":"fa","type":6,"floats":[0.1,0.2]}` | `attr["floats"]` 逐元素取值 | 空列表（本 57 文件未使用 FLOATS） |

> 口径：任务书 2.4 节 + 《迁移规范》（项目内部文档）§4.2（D10 读 `ge_onnx.proto`：UNDEFINED=0、FLOAT=1、INT=2、STRING=3、TENSOR=4、GRAPH=5、FLOATS=6、INTS=7）。迁移后 type 比较使用 `OnnxAttrProtoType` 数值常量（`ONNX_ATTR_INT=2` 等），不引用 protobuf 枚举。

#### 3.1.2 浮点标量的两点约定

**约定 ①：FLOAT 标量值以字符串承载，须按字符串读取后再转 float。** 仓内参考 bernoulli 的写法是 `std::string seed_string = attr["f"]; seed = atof(seed_string.c_str());`，但 `atof("abc")` 会**静默返回 0**，而 UT（如 `test_addcmul` 的 `f` 非数字字符串场景）要求返回 `FAILED`，因此不能只写 `atof`，必须显式校验。交付实现用 `std::stof` + `try/catch`（`std::invalid_argument`/`std::out_of_range` → `FAILED`，`"1e999"` 等越界同样失败）；等价写法为 `strtof` + `errno/endptr`。

**约定 ②：属性存在但 `f` 键缺省 = 0.0，不得回落算子默认值。** proto3 省略零值标量字段，因此“属性对象存在、无 `f` 键”表示真实值 0.0，而非“属性缺失用默认值”。例：`alpha` 默认 1.0，若 ONNX 属性为 `{"name":"alpha","type":1}`，迁移后 `alpha` 必须为 **0.0**。UT 以无 `f` 键构造并断言 0.0（`test_addcmul`、`test_clip`、`test_random_normal_like`）。

**衍生**：`i` 缺省 = **0**（UT 大量使用 `{"name":"axis","type":2}` 无 `i` 并断言 0，如 `test_flatten`、`test_concat`）；`ints` 缺省 = 空列表；`s` 缺省 = 空串。以上“缺键值”均不回落算子默认值；算子默认值只在属性整体缺失（或 type 不符被当作缺失）时生效。

### 3.2 实现方案

#### 3.2.1 统一代码骨架

除“不读属性”的文件（3.2.3 第 8 条）外，所有解析函数采用如下骨架（依据 《迁移规范》（项目内部文档）§2.3.2，与 `math/cast`、`conversion/flatten` 实际改写一致）：

```cpp
static Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // ① 属性解析前的原有副作用逐行保留（默认值、动态 IO、original_type），次序不得重排
    int axis = 1;                                   // 该文件的迁移前默认值

    // ② 取 attribute JSON；取不到不报错，走“属性缺失”路径
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            const json attrs = json::parse(attrs_string.GetString());
            // ③ 守卫：无 attribute 键 / 空数组 / 非数组 → 跳过（按属性缺失处理）
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (const json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "axis" &&
                        attr.value("type", -1) == ONNX_ATTR_INT) {   // type 校验逐文件照抄
                        axis = attr.value("i", 0);                   // i 缺省 = 0
                    }
                }
            }
        } catch (const std::exception& e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "Parse source attribute JSON failed: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "Parse source attribute JSON failed: unknown exception.");
            return FAILED;
        }
    }

    // ④ 写出与收尾：键名、次序、失败判据位置与迁移前一致
    op_dest.SetAttr("axis", axis);
    return SUCCESS;                                 // “缺属性即失败”的文件（如 Cast 缺 to）在此返回 FAILED
}
```

要点：

1. **无 `attribute` 键按原语义**：`GetAttr` 非 `GRAPH_SUCCESS` 时不得失败，继续走“属性缺失”路径；最终返回值由原逻辑决定（多数 `SUCCESS`；Cast 缺 `to` 等必需属性返回 `FAILED`）。
2. **结构不合法（非数组）按缺失处理**：`{"attribute":[]}`、无 `attribute` 键、非数组都跳过循环；唯一例外是“必需属性缺失即失败”的文件（如 einsum 的 `equation`），其失败来自必需属性缺失而非 JSON 结构。
3. **try/catch 覆盖 `json::parse` 与所有 `get<>()/value<>()`**：`json::exception` 与兜底 `catch(...)` 均记日志返回 `FAILED`；57 个文件迁移前源码全部没有 try/catch（原由 protobuf 解析），“非法 JSON → FAILED”是本次统一新增行为，与 UT 对齐且**不改变合法输入语义**。
4. **“不读属性”两类处理**：A 类 4 文件（`similar_opp`、`sequence`、`mean`、`sum`）**不解析 JSON、恒 SUCCESS**，不得引入失败分支；B 类 4 文件（`experimental/math/asinh` 的 max、`math/maximum`、`math/minimum`、`math/bitwise_xor`）仅做 JSON 合法性校验（`json::accept`），非法即 `FAILED`，且**失败必须早于任何 `SetAttr`/`Dynamic*Register`**。
5. **direct-traverse 形态照抄**：部分文件迁移前直接遍历 `attrs["attribute"]`（无守卫），语义是“非数组时抛异常 → FAILED”；是否保留该形态由原实现决定，**不得统一化**。

#### 3.2.2 include 替换

按文件是否含拆图函数分两套清单（**删除** `onnx_common.h` 及其拉入的 `ge_onnx.pb.h`）：

- **A 纯解析插件**：`graph/operator.h`、`math_onnx_plugin_util.h`、`log/log.h`、`register/register.h`、`nlohmann/json.hpp`；按需补 `graph/graph.h`、`stub_ops.h`（原实现用到 `ge::Graph`/`op::Xxx` 时）。
- **B 含 `ParseOpToGraphFn`**：在 A 基础上必须显式补 `graph/graph.h` + `stub_ops.h`（`op::Data`/`op::Const` 等声明来源，原先由 `onnx_common.h` 传递引入），并保留具体 `op_graph/*_proto.h`、`op_math_proto_extend.h`。
- **改写对照**：`#include "onnx_common.h"` → 上述显式头；`#include "math/cast/op_graph/cast_proto.h"` 等组合算子头原样保留。

规则与验收：

1. **统一删除 `onnx_common.h`**（唯一作用是拉入 `ge_onnx.pb.h`）；57 个目标文件**禁止**出现 `onnx_common.h`、`ge_onnx.pb.h`、`ge::onnx::`。
2. 原经 `onnx_common.h` **间接获得**的头（`log/log.h`、`graph/operator.h`、`graph/graph.h` 等）按实际使用**显式补 include**，避免编译缺符号。
3. **保留**原有组合算子头（`op::Xxx`）、`stub_ops.h`、`op_math_proto_extend.h`、具体 `op_graph/*_proto.h`。
4. namespace 惯例按文件现状：`namespace domi { using json = nlohmann::json; }`（correlation）或 include 后 `using json = nlohmann::json;`（bernoulli、shape），不强制统一。
5. **残留计数只能声明“57 个目标文件 = 0”**：`math/shape`、`random/stateless_bernoulli` 两个**非目标**插件仍 include `onnx_common.h`，且该头保留给 TF 插件，不得声明“全仓 = 0”（《迁移规范》（项目内部文档）§5.6-R2、§5.9-J2）。
6. 不新增公共头文件；仅在既有 `math_onnx_plugin_util.h` 内追加 3 项能力（见 1.2.4、3.2.4），并随 PR 单独说明。

#### 3.2.3 属性读取与语义保持规则

判定标准是“**语义逐文件等价**”，不是“写法统一”。8 条规则（《迁移规范》（项目内部文档）§4.2/§4.3/§4.4）：

1. **整数标量**：`attr.value("i", 0)`（或 `contains("i") ? attr["i"].get<int64_t>() : 0`）；`i` 缺省 = **0**，不回落默认值；`int64_t → int/int32_t` 收窄按文件原样，必要时逐元素 `static_cast`。
2. **浮点标量**：`f` 是字符串，`std::string s = attr.value("f", "0");` 再转 float；必须显式校验（`std::stof` + `try/catch` 或 `strtof` + `errno/endptr`），非法/越界 → `FAILED`；`f` 缺省 = **0.0**。这是 5 个 P6 随机文件 + addcmul + clip + pad 的必查点。示例：`seed = static_cast<int>(std::stof(attr.value("f", "0")));`
3. **字符串**：`attr.value("s", std::string())`；`s` 缺省 = 空串（DepthToSpace 的 `mode` 由 "DCR" 默认值兜底，但“属性命中无 `s`”时为 ""，不回落默认值）。
4. **列表**：`ints`/`floats` 为数组，`get<std::vector<int64_t>>()` 或逐元素 `get<int64_t>()`；元素类型保持文件原样；写入 int32 容器/张量时**逐元素** `static_cast<int32_t>`（如 Pad 的 `v_pads.push_back(static_cast<int32_t>(...))`，同时保持“前 half/后 half 交错”顺序），不得整表转换、不得改变元素类型或顺序；`ints` 缺省 = 空列表。
5. **type 校验逐文件照抄**：迁移前是否校验 `attr.type()`、校验失败按“缺失”还是“失败”，必须逐文件保留，不得新增也不得删除。《迁移规范》表 A 有 40 个带 type 判断的“属性命中点”：`flatten` 的 `==INT` 不符则跳过（默认值 1 生效）；`concat` 的 type 非 INT 时**跳过并继续扫描后面的同名 INT 属性**（`test_concat:107` 要求首个 FLOAT axis 被跳过、后续 INT 生效，不能写成“命中 name 即 break”）；`bitshift` 严格白名单；`eyelike` 不校验。UT 中 `select_last_index`/`keepdims` 的 `type:1`(FLOAT) 构造是**有意**的“type 不符按缺失”覆盖点，实现不得放宽 type 判断。
6. **同名属性**：保持“首个命中即 break / 后出现覆盖 / 累加”三种语义，逐文件照抄：Cast、Concat、OneHot 首个即 break；Flatten、Transpose 后出现覆盖；Slice 的 `axes` 循环内累加。
7. **无关属性**：一律忽略，不得因未知属性失败（`bitshift` 的“任何不匹配属性立即失败”是其原语义，例外，保留）。
8. **缺失与非法**：属性整体缺失（或 type 不符按缺失）→ 保持原默认值与返回值（多数 `SUCCESS`）；非法 JSON/取值异常 → 记日志返回 `FAILED`；返回码与迁移前一致；写出次序与失败判据位置不得重排（ArgMax/ArgMin/TopKV9/Clip 的 name 在失败判据之后；reduce B 分支 `original_type` 在判空之后）。例外按 3.2.1 第 4 条：A 类恒 SUCCESS，B 类仅校验且失败先于写出。**不得新增**“列表不能为空”“type 必须匹配”“数值必须非负”等迁移前没有的失败条件。

#### 3.2.4 节点名与输入输出计数替换

**（1）节点名：`node->name()` → 源算子名。** 迁移前共 **35 个文件**写 `SetAttr("name", <节点名>)`（另 **22** 个不写，其中多为只写 `original_type`；迁移后也不得新增），按 UT 契约分三类（《迁移规范》（项目内部文档）§2.3.4/§5.7）：

> 口径说明：源码口径 **35/22**；UT 口径为 57 份冻结 UT 中 **42 个断言节点名、15 个不断言**（`GetAttr("name")`）。核验：`git grep -l 'SetAttr("name"' 91c6c1715 -- '*_onnx_plugin.cpp'`（在 57 集内命中 35）。

| 场景 | 迁移后写法 | 文件（UT 依据） |
|---|---|---|
| 源算子名非空 | 写**源算子名原值**（UT 中 op_src 名为 `"src"` 时断言 `name == "src"`） | 所有写 name 的文件；`test_npu_slice:172-183` 的 `EXPECT_NE(name,"npu_slice")` 仅在此场景适用 |
| 源名空 → **回落目标算子名** | `source_name.empty() ? GetOpName(op_dest) : source_name` | 5 文件：`math/maximum`、`math/minimum`、`experimental/math/asinh` 的 max、`math/bitwise_xor`、`conversion/transpose` |
| 源名空 → **写空串（不回落）** | `op_dest.SetAttr("name", GetOpName(op_src));` | `common/src/framework/sequence_onnx_plugin.cpp` 的 6 个算子 |

回落类参考写法（`correlation`、`npu_indexing`、`one_hot_d`、`shape` 四处先例）：先取目标名 `op_name = GetOpName(op_dest)`，再取源名 `source_name`（`op_src.GetName(...) == GRAPH_SUCCESS ? source_name_string.GetString() : ""`），最后 `source_name.empty() ? op_name : source_name` 写入 `name`。

注意：① “回落”写法**只对这 5 个文件成立**，不是全局模板；② name 写入位置与源码一致；③ name 会被拆图函数读走拼进子图节点名（`ori_name + "_data_0"`），“回落 vs 空串”会改变空名节点的图节点名，两轮 om 需单独观察。

**（2）输入/输出数量：`node->input_size()/output_size()` → 源算子数量。** 真实 ATC 下解析器把输入/输出名序列化为 `op_src` 的 `"input"`/`"output"` **列表属性**，此时 `op_src.GetInputsSize()` 恒为 0；UT 未序列化时回退 `GetInputsSize()/GetOutputsSize()`。迁移后用公共头辅助函数：`GetSourceInputNum` 先 `GetAttr("input", std::vector<std::string>)` 成功则返回其 `size()`，否则回退 `op_src.GetInputsSize()`；`GetSourceOutputNum` 同构取 `"output"`。

口径与边界：① 动态 IO 按**实测**为 **73 个动态注册引用点**（`DynamicInputRegister` 38 + `DynamicOutputRegister` 35）；含动态注册调用的文件 38 个，其中 26 个文件的动态 IO 需真实 ATC 模型二次实证（按动态 IO 注册点清单核对，含 3 项口径差；任务书旧记“20 文件 / 24 引用点”为旧口径，引用须标注，《迁移规范》§5.2/A-05）；② UT 裸 `op_src` 的 `GetInputsSize()` 期望 0（`test_concat:132-140`），是回退分支契约；③ **唯一合法例外**：回调内对**目的算子**的 `op_dest.GetInputsSize()` 读的是已注册完成的输入数（候选仅 `conversion/trilu/.../trilu_onnx_plugin.cpp:71` 一处），保持原样；④ 动态 IO 来源等价性需用至少 1 个真实 ATC 动态 IO 模型二次实证（归档 Operator 的 `input`/`output` 列表属性与数量、加载证据行），未完成前不得声称已核对。

#### 3.2.5 特殊机制保持

迁移只改“属性从哪来、怎么读”，以下机制逐条保持原逻辑与原时序：

1. **动态输入/输出注册**：`DynamicInputRegister/DynamicOutputRegister` 的端口名、数量来源（改用 `GetSourceInputNum/GetSourceOutputNum`）、调用位置（默认值之后、失败判据前后）与原实现一致。
2. **`ParseOpToGraphFn` 拆图函数**：函数体、函数名、注册关系、`ImplyType` 全部不变；仅 include 从 `onnx_common.h` 换为 `stub_ops.h` + `graph/graph.h`（3.2.2-B）；拆图函数读 `name`/`original_type` 的行为依赖迁移后的 name 语义（3.2.4）。
3. **双注册/多注册（opset 分支）**：14 个双注册文件（Clip 的 `8~10`/`11~18`、Pad 的 `8~10`/`11~18`、Reduce 家族的 `8~12`/`13~18` 等），`Min` 为四条注册。每个注册对应解析函数**逐条**替换；`OriginOpType` 条目与顺序、`ImplyType`、`ParseOpToGraphFn` 有无与名字逐字保留（多行重排用集合比对）。
4. **张量属性写入**：ArgMax/ArgMin 的 `dimension`、Reduce 系列/ReduceSumSquare 的 `axes`、TopK 的 `k`、RandomUniform 的 `shape/max/min`、Sum 的 `N` 继续用 `Vec2Tensor`/`CreateScalar` 构造，dims/dtype 与迁移前一致。
5. **`original_type`/`ImplyType`/注册三要素**：字符串值（如 `"ai.onnx::11::Transpose"`、`npu::1::NPUSlice`）、写入位置、条数原样；`similar_opp` 的 **578 条注册**（裸 `grep -c` 得 579，含宏定义行 1，实数 578）与 63 个 ONNX 算子名映射（11 个 ATC 名不同，如 Not→LogicalNot、Det→MatrixDeterminant、Where→SelectV2）逐条不变。
6. **失败先于写出的 B 类语义**：`math/maximum`、`math/minimum`、`experimental/math/asinh`、`math/bitwise_xor` 的非法 JSON 失败必须发生在任何 `SetAttr`/`Dynamic*Register` 之前（UT 断言失败路径上 `GetAttr("name")==GRAPH_FAILED`、`GetDynamicInputNum("x")==0`）。

#### 3.2.6 例外：Cast 失效判据修正

**这是 E-1，三类显式例外（见 3.5）中唯一改判据者**；E-2 的新增失败面仅限非法 JSON 输入，E-3 只影响空名节点；该修正必须做、不算违反语义保持。

事实（《迁移规范》（项目内部文档）§6.1）：迁移前判据为 `if (om == -1)`（`cast_onnx_plugin.cpp:31`），而 `GetOmDtypeFromOnnxDtype` 未命中时返回 **`ge::DT_UNDEFINED`**（`math_onnx_plugin_util.h:109-116`），其值为 `C_DT_UNDEFINED = 28`（CANN 9.1.0 `graph/types.h:109`、`graph/c_types.h:45`）；`-1` 与 28 永不相等，故旧判据**永不成立**（死代码）。下游影响：不支持的 dtype 迁移前要到引擎选择阶段才失败（`Failed to select engine`，`test_cases/README.md:43`）。

**改前（`cast_onnx_plugin.cpp:27-37`）：**

```cpp
        if (attr.name() == "to") {
            bfind_to = true;
            int onnx = attr.i();
            int32_t om = GetOmDtypeFromOnnxDtype(onnx);
            if (om == -1) {                    // 永不成立：函数返回 DT_UNDEFINED(28)
                OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
                return FAILED;
            }
            op_dst.SetAttr("dst_type", om);
            break;
        }
```

**改后（迁移后交付实现）：**

```cpp
                    if (attr.value("name", "") == "to") {
                        bfind_to = true;
                        const int onnx = attr.value("i", 0);
                        const int32_t om = static_cast<int32_t>(GetOmDtypeFromOnnxDtype(onnx));
                        // 修正：GetOmDtypeFromOnnxDtype 未命中返回 ge::DT_UNDEFINED 而非 -1，
                        // 原 om == -1 分支永不成立；按工具函数真实返回约定判定失败分支。
                        if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) {
                            OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
                            return FAILED;
                        }
                        op_dst.SetAttr("dst_type", om);
                        break;
                    }
```

**为什么不破坏等价性判据：** ① 合法模型：`to` 为受支持 dtype 时映射值与迁移前相同，走同一行 `SetAttr("dst_type", om)`，属性序列逐项一致；② 非法模型：迁移前解析不失败（判据失效）、在引擎选择阶段失败（`Failed to select engine`），迁移后解析阶段返回 `FAILED`——两轮均无法产出 om，符合“转换失败且无产物”的等价判据，差别仅是失败更早、报错更明确；③ 修正只把“永不触发的死分支”换成“工具函数真实失败约定下的失败分支”，不新增对合法 dtype 的拒绝；④ UT 一致：`OnnxCastPluginTest.ReturnFailedWhenDtypeUnsupported`（`to=20`）与 `ReturnFailedWhenToIsZeroWithoutIField`（`to` 无 `i` → 0 → 映射不到）均要求 `FAILED` 且不写 `dst_type`。

**同类问题处理原则**：其余若发现“判据与工具函数真实返回不符”，以工具函数真实返回为准，并在单算子自查表逐项说明；不得扩大修改面。

#### 3.2.7 文件分组与工程实施

按实施方式分 4 组（可复跑：A/B 组来自 《迁移规范》（项目内部文档）§4.4.1“不读属性 8 文件两分”，D 组来自《迁移规范》§3.2b 表 B“双注册/多注册”集合，C 组为剩余单注册）：

| 组 | 文件数 | 文件集合（特征） | 迁移要点 |
|---|---:|---|---|
| **A. 宏批量注册 / 不解析属性（恒 SUCCESS）** | 4 | `similar_opp`（63 算子/578 注册）、`sequence`（6 算子）、`accumulate_nv2/{mean,sum}` | 只改签名与注册行；**不解析 JSON、不新增失败分支**（非法 JSON 也 SUCCESS）；mean/sum 只写 `original_type`/`name`/`N`/动态 IO；`sequence` 的 name 写空串；注册表/函数后缀/ATC 名映射不变 |
| **B. 不读属性但需 JSON 合法性校验** | 4 | `experimental/math/asinh`（max）、`math/maximum`、`math/minimum`（四注册）、`math/bitwise_xor` | `json::accept` 仅校验并丢弃结果；非法 JSON → `FAILED` 且**先于任何 `SetAttr`/`Dynamic*Register`**；合法内容忽略、其余恒 SUCCESS；`min` 四条注册逐条替换 |
| **C. 常规单注册属性解析** | 35 | 其余 35 文件（含 `transpose`、`concat`、`split`、`trilu`、`eyelike` 等含拆图但单注册的文件） | 按 3.2.1~3.2.4 迁移；拆图函数体不动；动态 IO 用 `GetSourceInputNum/OutputNum` |
| **D. 双注册/多注册（opset 分支）** | 14 | `clip`、`pad`、`squeeze`、`slice`、`unsqueeze`、`reduce_log_sum_exp`、`reduce_max`、`reduce_mean`、`reduce_min`、`reduce_prod`、`reduce_log_sum`、`reduce_sum`、`reduce_sum_square`、`topk` | 每个注册对应函数**逐条**迁移；两套语义分别保持（Clip V9 浮点 vs V11 端口名、Pad v9 三属性 vs v11 仅 mode、Reduce B 分支的 `original_type`/`input_size`/`noop_with_empty_axes`）；`OriginOpType` 逐字保留 |
| **合计** | **57** | — | — |

> `min` 虽四条注册，因“不读属性 + 需 JSON 校验”归入 B 组，四条注册仍逐条替换；分组只用于排期与评审回溯。

**工程实施约束**：

1. **改动量最小**：单文件只做迁移必需改动（include、签名、属性读取、name/计数来源、注册行），不重排格式、不改注释与 license 头、不做无关重构。
2. **函数嵌套深度 ≤5**（C++ 编码规范/CodeCheck）：逻辑复杂时按 `reverse_sequence_onnx_plugin.cpp` 抽出 `ApplyXxxAttr` 等静态辅助函数。
3. **注册三要素逐字比对**：OP 名、`OriginOpType` 每条与顺序、`ParseOpToGraphFn` 有无与名字、`ImplyType`；多行重排用集合比对。
4. **提交粒度**：按插件文件/算子分组提交；独立缺陷单独说明并隔离，不混入解耦改动。
5. **静态检查**：`git diff --check` 须声明前提 `git config core.whitespace cr-at-eol`（52/57 基线文件为 CRLF，默认配置会把新增行的 CR 误报为 trailing whitespace）；目标文件 `.ParseParamsFn(`、`onnx_common.h`、`ge_onnx.pb.h`、`ge::onnx::` 均为 0。
6. **构建与测试**：`bash build.sh --onnxplugin -j8` 通过；UT 按 4.2 放置并补同构 `CMakeLists.txt`（被 `file(GLOB)` + `add_subdirectory` 链发现），`--gtest_filter='Onnx*PluginTest.*'` 通过。
7. **验证留证**：两轮 om、插件生效正控、UT 输出归档；报告显式声明数字口径（静态宏 vs 运行用例数、文件数 vs 条目数），不得混用。

---

### 3.3 分组特殊场景与语义保持清单

本节按 common / conversion / math+experimental+random 三组逐文件列出特殊语义与保持方式，**合计覆盖 57/57 文件、124/124 条目**，与任务书第 1 节范围表逐条一致；表内「保持方式」指迁移后等价实现手段，通用规则见 3.2，每行只列该文件最独特的 1–2 个点。

#### 3.3.1 common 组（14 文件 / 81 条目）

覆盖 `common/src/framework/` 14 个插件文件、81 个算子条目（similar_opp 63 + sequence 6 + 其余 12 文件各 1）。本组共性：include 去 `onnx_common.h`、注册三要素逐字保留、仅原本写 name 的文件写 name、动态 IO 用 `GetSourceInputNum/GetSourceOutputNum`、读取属性的文件新增非法 JSON→`FAILED`（A 类 `similar_opp`/`sequence` 恒 `SUCCESS`）。

| 插件文件 | 算子条目数 | 关键属性 | 特殊语义 | 保持方式 |
|---|---|---|---|---|
| `concat_from_sequence_onnx_plugin.cpp` | 1（ConcatFromSequence） | `axis`、`new_axis`（INT，缺省取 0），均无 type 校验 | 无动态 IO、无失败分支，恒 SUCCESS；声明 required 的 `axis` 缺省仍取 0 | 逐属性 JSON 提取；写出顺序与 `original_type` 不变 |
| `deformable_offsets_onnx_plugin.cpp` | 1（DeformableOffsets） | `strides`/`pads`/`dilations`/`ksize`（INTS）等 7 个属性 | 单元素 INTS 展开为 `[1,1,v,v]`（ksize 不展开）；`dilations` 先清空再累加；三次格式改写任一失败即 FAILED | 两个 helper 改为 `const json&`，展开与失败逻辑逐行照抄 |
| `einsum_onnx_plugin.cpp` | 1（Einsum） | `equation`（STRING） | 缺失、键缺省或空串即 FAILED，且失败先于任何写出与动态注册；动态输入数等于源输入数并以 `N` 保存 | 输入数改用 `GetSourceInputNum`；失败时序与写出顺序不变 |
| `eyelike_onnx_plugin.cpp` | 1（EyeLike） | `dtype`（INT，缺省 1）、`k`（INT，缺省 0），均无 type 校验 | 解析恒 SUCCESS；失败在拆图回调：取不到 name 或 dtype 映射为 `DT_UNDEFINED` 即 FAILED | 拆图回调逐行未改；include 换图接口并保留拆图 proto 头 |
| `phony_concat_onnx_plugin.cpp` | 1（PhonyConcat） | `concat_dim`（INTS）、`N`（INTS）、`keep_input_offset`（INT） | 默认 `concat_dim={0}`；INTS 非空才清空并累加；`keep_input_offset` 按 `i==0` 置 false，`i` 缺省也算命中 | 清空条件改为 JSON 数组非空；`i` 判据与写出顺序不变 |
| `phony_split_onnx_plugin.cpp` | 1（PhonySplit） | `split_dim`（INTS）、`num_split`（INTS）、`keep_output_offset`（INT） | 与 PhonyConcat 同构；默认 `num_split={源输出数}`；动态输出数等于源输出数 | 输出数改用 `GetSourceOutputNum`；注册位置与顺序不变 |
| `random_normal_like_onnx_plugin.cpp` | 1（RandomNormalLike） | `dtype`（INT）、`mean`/`scale`（FLOAT）、`seed`（FLOAT 按 int 用） | FLOAT 以字符串承载，`f` 键缺省为 0.0 而不回落声明默认；拆图仅支持 float/float16/double | JSON 解析段重写但 `has_dtype_attr` 语义不变；拆图回调未改 |
| `random_uniform_like_onnx_plugin.cpp` | 1（RandomUniformLike） | `dtype`（INT）、`low`/`high`（FLOAT）、`seed`（FLOAT 按 int 用） | 与 RandomNormalLike 同构；拆图用 `high-low` 作 Mul 系数、`low` 作 Add 偏置；`f` 缺省为 0.0 | 解析段 JSON 重写；拆图回调与注册不变 |
| `reduceL1_onnx_plugin.cpp` | 1（ReduceL1） | `axes`（INTS）、`keepdims`（INT）、`noop_with_empty_axes`（INT） | `axes` 逐元素累加不清空；`noop==1` 仅告警不失败；写 `keepdim`（无 s）与 `p=1.0`；OP 名 `LpNormV2`、opset 8–17 | type 判定与解析逐条照抄；键名与常量差异保留 |
| `reduceL2_onnx_plugin.cpp` | 1（ReduceL2） | 同 ReduceL1 | 与 ReduceL1 逐行同构，仅 `p=2.0f`、OriginOpType 为 ReduceL2，OP 名同为 `LpNormV2` | 与 ReduceL1 同一迁移方式，键名与常量逐字保留 |
| `repeat_with_pad_onnx_plugin.cpp` | 1（RepeatWithPad） | `axis`（INT，缺省 0）、`capacity`（INT，缺省 0） | 恒 SUCCESS；不写 name、无动态 IO、无 original_type；写出顺序 `capacity` 先于 `axis` | name 判断与 SetAttr 顺序照抄；注册与函数名不变 |
| `sequence_onnx_plugin.cpp` | 6（SequenceAt/Construct/Empty/Erase/Insert/Length） | 6 个算子均不读属性 | 仅 SequenceConstruct 注册动态输入，其余只写 original_type 与 name；ImplyType 全为 AI_CPU；name 空写空串不回落；非法 JSON 恒 SUCCESS | 无 try/catch、无失败分支；不引入 json 头；注册逐条保留 |
| `similar_opp_onnx_plugin.cpp` | 63（Abs 至 Where，含 11 个 ATC 名与 ONNX 名不同） | 无（63 个算子均不读属性） | 宏批量生成 63 个解析函数，宏体仅返回 SUCCESS；缺属性、非法 JSON、类型异常一律 SUCCESS | 宏签名改 Operator 形态并删不可达判空；578 条注册零改动 |
| `split_to_sequence_onnx_plugin.cpp` | 1（SplitToSequence） | `axis`（INT，缺省 0）、`keepdims`（INT，缺省 1） | 恒 SUCCESS；写出 name→axis→keepdims（键名无下划线）→original_type；无动态 IO | JSON 解析段迁移；节点名改用 `GetOpName`；写出顺序不变 |

#### 3.3.2 conversion 组（14 文件 / 14 条目）

覆盖 `conversion/*/framework/` 14 个插件文件、14 个条目（Clip/Pad/Squeeze/Slice/Unsqueeze 双注册不拆条目）。本组唯一语义特例为 Clip opset≥11 的 F-C2 修复（两侧口径见 5.5）；拆图回调函数体逐行未改，仅补 `stub_ops.h`+`graph/graph.h` include。

| 插件文件 | 算子条目数 | 关键属性 | 特殊语义 | 保持方式 |
|---|---|---|---|---|
| `conversion/clip_by_value/framework/clip_onnx_plugin.cpp`（Clip，双注册） | 1（#82） | V9 读 `max`/`min`（FLOAT）写标量张量；V11 不读属性，`min/max` 为可选输入 | A=8–10 属性形态、B=11–18 可选输入形态；B 按端口名判空走 ClipByValue/Maximum/Minimum；F-C2 修复即恢复逐端口判空 | 取值、默认值、覆盖顺序与张量布置一致；B 回调未改 |
| `conversion/concat/framework/concat_onnx_plugin.cpp`（Concat） | 1（#83） | `axis`（INT），首个命中即 break，未命中返回 `PARAM_INVALID` | 单注册 8–18 + 拆图回调按 `N` 建 ConcatD 子图；失败路径不写任何属性 | `axis` 取值与首命中语义一致；回调体未改 |
| `conversion/depth_to_space/framework/depth_to_space_onnx_plugin.cpp`（DepthToSpace） | 1（#84） | `blocksize`（INT）、`mode`（STRING） | 任何 SetAttr 之前先判 `block_size<2`；格式改写先输入后输出；最后写 `data_format="NCHW"` | 默认值、失败位置与改写顺序一致；仅新增非法 JSON 加固 |
| `conversion/flatten/framework/flatten_onnx_plugin.cpp`（Flatten） | 1（#85） | `axis`（INT），`i` 缺省取 0 且视为命中 | 无业务失败分支、解析恒 SUCCESS；唯一写入在属性循环之后 | `axis` 取值与 type 门槛保留；无新增失败分支 |
| `conversion/pad_v3/framework/pad_onnx_plugin.cpp`（Pad，双注册） | 1（#86） | v9 读 `mode`/`pads`/`value`；v11 仅读 `mode` | A=`PartitionedCall` 8–10 属性形态、B=`PadV3` 11–18 输入形态；`pads` 交错重排为 `[b1,e1,b2,e2]` | 重排公式与失败判据逐项一致；A 回调未改 |
| `conversion/reshape/framework/reshape_onnx_plugin.cpp`（Reshape） | 1（#87） | `allowzero`（INT），`i` 缺省取 0 | 无业务失败分支、无 name、无动态 IO、无回调；ImplyType 为 GELOCAL、opset 8–18 | 取值一致；注册与 ImplyType 不变 |
| `conversion/slice/framework/npu_slice_onnx_plugin.cpp`（NPUSlice） | 1（#88） | `offsetss`（INTS，双 s）、`sizes`（INTS） | 恒 SUCCESS；动态输入输出数取源算子；多次出现累加；`original_type="npu::1::NPUSlice"` | dtype、形状与元素顺序一致；回调逐行未改 |
| `conversion/space_to_depth/framework/space_to_depth_onnx_plugin.cpp`（SpaceToDepth） | 1（#89） | `blocksize`（INT，默认 -1），`i` 缺省取 0 | 任何 SetAttr 之前校验 `<=0`；格式改写先输出后输入，与 DepthToSpace 相反 | 默认值、失败位置与改写顺序一致 |
| `conversion/split_d/framework/split_onnx_plugin.cpp`（Split） | 1（#90） | `axis`（INT）、`split`（INTS）、`num_outputs`（INT） | 失败判据在写属性之前：`split` 非空且不等于输出数即 FAILED；写 `input_size` 与 `num_split` | 判据、顺序与 `num_split` 口径一致；回调未改 |
| `conversion/squeeze/framework/squeeze_onnx_plugin.cpp`（Squeeze，双注册） | 1（#91） | Squeeze 读 `axes`（INTS）写单数键 `axis`；SqueezeV3 不读属性 | `Squeeze` 8–12 属性形态 + `SqueezeV3` 13–18 恒 SUCCESS；均 GELOCAL、无回调 | `axis` 元素一致；删除的仅是不可达 protobuf 判据 |
| `conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp`（Slice，双注册） | 1（#92） | Call 读 `ends`/`starts`/`axes`（INTS） | A=`PartitionedCall` 8–9 属性形态、B=`StridedSliceV2` 10–18 恒 SUCCESS；B 注册顺序 11,10,12 起不得重排 | 属性读取与失败点一致；`MakeConstOp` 与回调未改 |
| `conversion/transpose/framework/transpose_onnx_plugin.cpp`（Transpose） | 1（#93） | `perm`（INTS），无 type 校验，多次出现累加 | 解析恒 SUCCESS；`perm` 缺失或为空由拆图回调判 FAILED；`original_type` 硬编码 11 | `perm` 序列语义与失败顺序不变；回调未改 |
| `conversion/trilu/framework/trilu_onnx_plugin.cpp`（Trilu） | 1（#94） | `upper`（INT），`i` 缺省取 0 且视为命中 | 输入个数非 1/2 只告警不失败；动态输入数取源算子；回调按 1/2 输入建 Trilu 图 | 告警语义、写入序列与回调分支一致 |
| `conversion/unsqueeze/framework/unsequeeze_onnx_plugin.cpp`（Unsqueeze，双注册） | 1（#95） | Unsqueeze 读 `axes`（INTS）写复数键 `axes`；V3 不读属性 | `Unsqueeze` 8–12 属性形态 + `UnsqueezeV3` 13–18 恒 SUCCESS；`axes` 为空返回 `PARAM_INVALID` | `PARAM_INVALID` 返回值与判据位置不变；V3 行为不变 |

#### 3.3.3 math + experimental + random 组（29 文件 / 29 条目）

覆盖 math 25 + experimental 1 + random 3 共 29 文件、29 条目（41 条注册）。reduce 家族 8 个双注册文件：A 分支 opset 8–12 / `PartitionedCall`、B 分支 13–18，A/B 的 axes 写入形态与 `noop_with_empty_axes` 语义逐文件保留；两个同名 `max_onnx_plugin.cpp` 除共用 `gen_model_max.py` 外，版权年份、日志文本、`OriginOpType` 换行各按本文件保留。random 组仅记录口径，随机项与动态 IO 的实测结论见第 5 章。

| 插件文件 | 算子条目数 | 关键属性 | 特殊语义 | 保持方式 |
|---|---|---|---|---|
| experimental/math/asinh/framework/max_onnx_plugin.cpp | 1（注册 1） | 无属性（不消费 attribute） | 与 math/maximum 同名不同文件、两份 UT 不同；JSON 仅语法校验，非法即 FAILED 且先于写出；name 空回落目标名；共用 `gen_model_max.py` | `json::accept` 仅校验；name 回落保留；回调体未改 |
| math/accumulate_nv2/framework/mean_onnx_plugin.cpp | 1（注册 1） | 无属性 | 注册 OP 名 `PartitionedCall`、opset 8–18；`original_type` 写在其余写出之前；回调在 N 缺失或 0 时 FAILED | 保留无属性形态；N 用 `GetSourceInputNum`；回调不改 |
| math/accumulate_nv2/framework/sum_onnx_plugin.cpp | 1（注册 1） | 无属性 | 注册 OP 名是 `AccumulateNV2` 而非 Sum；`input_num<1` 即 FAILED；只写动态输入与 N，不写 name | OP 名与失败判据原样；N 用 `GetSourceInputNum` |
| math/addcmul/framework/addcmul_onnx_plugin.cpp | 1（注册 1） | `value`（FLOAT，字符串承载） | 属性整体缺失即 FAILED；`f` 缺省为 0.0；非数值经 `stof` 抛异常即 FAILED；original_type 为 torch 符号名，ImplyType 为 GELOCAL | `stof` 承载保留；注册三要素与回调逐字保留 |
| math/arg_max_v2/framework/argmax_onnx_plugin.cpp | 1（注册 1） | `axis`/`keepdims`/`select_last_index`（INT） | 三个并列 if 非 else-if；`select_last_index==1` 即 FAILED 且先于写 name；dimension 用 `CreateScalar` 张量；keep_dims 原值写入 | 标量张量写法与失败时序原样；回调不改 |
| math/arg_min/framework/argmin_onnx_plugin.cpp | 1（注册 1） | 同 ArgMax | 与 ArgMax 同构，差异在 original_type 与日志文本；含 `rtGetSocSpec` 平台分支按平台选 INT64/INT32 | 平台分支与回调原样；标量张量写法不变 |
| math/bitwise_xor/framework/xor_onnx_plugin.cpp | 1（注册 1） | 无属性 | `Xor`、opset 8–18；JSON 仅语法校验；name 空回落；硬编码动态输入 2、输出 1 | `json::accept` 与回落保留；注册与回调不改 |
| math/cast/framework/cast_onnx_plugin.cpp | 1（注册 1） | `to`（INT，dtype 枚举） | 映射未命中返回 `DT_UNDEFINED`；**Cast 自身唯一行为变化（E-1）** 是把判据由 `om==-1` 改为 `DT_UNDEFINED`，失败不写 dst_type | 见 3.2.6 改前改后；opset 与 TVM 保留 |
| math/cumsum/framework/cumsum_onnx_plugin.cpp | 1（注册 1） | `exclusive`/`reverse`（INT） | 注册 OP 名 `Cumsum`；两属性按 `i==1` 取 bool；axis 是输入张量不是属性；不写 name | OP 名与 `i` 判据保留；不新增任何写出 |
| math/is_inf/framework/isinf_onnx_plugin.cpp | 1（注册 1） | `detect_negative`/`detect_positive`（INT，默认 1） | 任一属性值不等于 1 即 FAILED（缺 i 按 0 也算命中）；成功路径不写任何属性 | 判据原样；注册要素逐字保留 |
| math/maximum/framework/max_onnx_plugin.cpp | 1（注册 1） | 无属性 | 与 experimental 同名不同文件、禁止互抄；name 空回落；N 来自 input 列表；日志文本与版权按本文件 | `json::accept` 保留；回调未改；共用脚本口径不变 |
| math/minimum/framework/min_onnx_plugin.cpp | 1（注册 4） | 无属性（四条解析函数） | 注册 OP 名 `PartitionedCall` 四条（8–10/11/12/13–18），仅 original_type 不同；四条均带回调 | 四条注册逐条逐字保留，不得模板化；装填顺序不变 |
| math/mod/framework/mod_onnx_plugin.cpp | 1（注册 1） | `fmod`（只比 name，无 type 判断） | 命中 fmod 仅告警不失败、不取 i；成功路径不写属性；非法 JSON 即 FAILED | 保留 JSON 解析仅为告警；注册不变 |
| math/one_hot/framework/onehot_onnx_plugin.cpp | 1（注册 1） | `axis`（INT，默认 -1，无 type 判断） | name 命中即取 i，缺 i 视为 0 命中；首个命中即 break，与后者覆盖相反；动态输入输出硬编码 3/1 | 无 type 过滤与 break 语义保留；回调不改 |
| math/reduce_log_sum_exp/framework/reduce_log_sum_exp_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`；B：另加 `noop_with_empty_axes` | 双注册 A 8–12/B 13–18；axes 为张量属性；A 与 B 的 dims 空表均补 0；A 不取 noop，回调硬编码 false | A/B 两分支不合并；dims 补零与写出顺序保留 |
| math/reduce_max/framework/reduce_max_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`/`noop`（独立 if，仅告警） | 双注册；A 的 noop==1 仅告警且写出硬编码 0，B 取原值；axes 为张量属性 | 告警与硬编码 0 原样；回调不改 |
| math/reduce_mean/framework/reduce_mean_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`/`noop` | 双注册 A 8–12/B 13–18；A 的 noop 仅告警并硬编码 0；dims 补 0 | 同 reduce_max 六条；两分支解析函数均保留 |
| math/reduce_min/framework/reduce_min_onnx_plugin.cpp | 1（注册 2） | 同 reduce_mean | 与 reduce_max 逐行同构，仅 include、局部变量名与 4 处算子名不同；双注册 | 禁止与 reduce_max 合并生成；其余同 reduce_max |
| math/reduce_prod/framework/reduce_prod_onnx_plugin.cpp | 1（注册 2） | 同 reduce_mean | 与 reduce_max 同构；双注册；om 侧归约维为 64，其余 reduce 为 256 | 同 reduce_max；64 维口径不在迁移中改动 |
| math/reduce_sum/framework/reduce_log_sum_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`；B：另加 `noop` | 双注册 A 8–12/B 13–18；A 的 dims 空表不补 0（源码无 else）；A 注册动态 IO 端口 x/output，B 不注册；A 回调追加 `op::Log` | A 不得补 else 补零；端口名与 Log 追加保留 |
| math/reduce_sum/framework/reduce_sum_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`；B：另加 `noop` | 双注册 A 8–12/B 13–18；dims 均补 0；A 回调硬编码 noop=false，B 取属性 | 函数名大小写与硬编码 noop 分开保留；回调不改 |
| math/right_shift/framework/bitshift_onnx_plugin.cpp | 1（注册 1） | `direction`（STRING，默认空串，无 RIGHT 默认） | 严格白名单：任何非 direction 或非 STRING 属性即 FAILED；完全无属性时循环不执行、解析 SUCCESS | 逐元素白名单判断，不得只判 direction；回调不改 |
| math/sign_bits_pack/framework/npu_sign_bits_pack_onnx_plugin.cpp | 1（注册 1） | `size`（INT，REQ_ATTR_NUM=1） | SetAttr 在属性循环内执行；命中计数不等于 1 即 FAILED（缺失、非 INT、重复均失败）；不写 name | 循环内写出与计数判据原样；`npu::1::` 前缀保留 |
| math/sign_bits_unpack/framework/npu_sign_bits_unpack_onnx_plugin.cpp | 1（注册 1） | `size`/`dtype`（INT，REQ_ATTR_NUM=2） | 两个 SetAttr 均在循环内；命中计数不等于 2 即 FAILED；dtype 写 int 原值 | 计数判据与日志等价写法保留；`npu::1::` 前缀保留 |
| math/square_sum_v1/framework/reduce_sum_square_onnx_plugin.cpp | 1（注册 2） | A：`axes`/`keepdims`；B：另加 `noop` | 双注册且 A/B 的 OP 名不同：A=`PartitionedCall`（opset 1 加 8–12）、B=`SquareSumV1`（13–18）；A 写 vector 键 axis 且 dims 补 0，B 写 vector 无 dims | vector 形态与 axis 键名保留；两个 OP 名差异逐字保留 |
| math/topk/framework/topk_onnx_plugin.cpp | 1（注册 2） | A：`k`/`axis`；B：`sorted`/`largest`/`axis` | 双注册但只有 A 带回调；A 的 k 用 `CreateScalar` 张量；k 缺失返回 PARAM_INVALID；B 的 sorted/largest 缺省为 false | 双注册方向与 PARAM_INVALID 原样；B 解析函数在 A 之前 |
| random/dsa_random_uniform/framework/random_uniform_onnx_plugin.cpp | 1（注册 1） | `dtype`/`high`/`low`/`seed`（FLOAT 按 int 用）/`shape`（INTS） | 无 golden；5 属性均带 type 判断；`f` 缺省为 0.0；shape 为空时 dims 为 {0}；0 输入图 ATC 报 E16001 | `stof` 与 int 截断保留；判据退化为两轮同一 E16001 失败 |
| random/stateless_random_normal_v2/framework/random_normal_onnx_plugin.cpp | 1（注册 1） | `dtype`/`mean`/`scale`/`seed`/`shape`（必需） | 无 golden；`f` 缺省为 0.0（scale 会变 0）；shape 为空即 FAILED；shape 写 vector 而非 Tensor | 字符串承载与空 shape 失败保留；两轮同一 E16001 一致 |
| random/stateless_sample_multinomial/framework/multinomial_onnx_plugin.cpp | 1（注册 1） | `dtype`/`sample_size`（无 type 判断）、`seed`（FLOAT 写 float） | 无 golden；dtype 与 sample_size 无 type 校验；seed 原值写入；回调先用默认值覆盖写回再判 `DT_UNDEFINED` | 覆盖写回段不得重构；判据退化为两轮行为一致 |

---

### 3.4 支持硬件

本任务书未限定单一芯片，实测与声明口径如下（**本队实测环境只有 Ascend910B3 ×4，物理卡 2/3/4/5，没有 Ascend 950 环境**；凡未亲测的硬件一律写「未实测/按任务书声明」，不标注通过）：

| 支持的芯片版本 | 任务书要求 | 本次实测情况 |
|---|---|---|
| Atlas 800I/T A2（A2） | 任务书声明支持 | **实测通过（Ascend910B3 ×4）** |
| A3 | 任务书未单独声明 A3（任务书口径：Atlas A2 与 Ascend 950 任一硬件验证通过即可） | 本次未实测、不涉及 |
| Ascend 950 | 任务书声明「任一硬件通过即可」 | **未实测**（本队无 950 环境，不得标「√」） |

任务书口径：**A2 / 950 任一硬件通过即可**；且**迁移前后两轮必须同硬件**（同型号、同 CANN 版本、同 ATC 参数），否则两轮输出不具备可比性。本设计的两轮 om 实测均在 Ascend910B3 + CANN 9.1.0（aarch64）上完成，环境明细见 4.3。

### 3.5 插件约束限制

1. **仅限 Host 侧插件参数解析层**：不改算子 kernel/tiling/op_api，不改 `ParseOpToGraphFn` 拆图函数体，不改变算子计算逻辑与图结构。
2. **只做等价重构**：迁移前后写入 `op_dest` 的属性名/类型/值序列、各分支返回值保持一致；**功能语义保持，显式例外三类**：① **E-1** Cast 失效判据修正（`om == -1` → `om == DT_UNDEFINED`，见 3.2.6）；② **E-2** 规范要求的 B 类 4 文件（`math/maximum`、`math/minimum`、`experimental/math/asinh`、`math/bitwise_xor`）在**非法 JSON** 输入上新增 `FAILED`（基线不解析 attribute；GE 不会产生非法 JSON，差异仅在 UT 构造的非法 JSON 输入上可观察）；③ **E-3** 规范要求的 5 个文件在**节点名为空**时回落/写空串，仅影响空名节点。②③在常规合法输入上行为不变，且已分别由 UT 与两轮 om 观测。
3. **依赖边界**：只使用仓内既有 `math_onnx_plugin_util.h` 与 third_party 自带 nlohmann/json；不新增第三方库、不新增公共头文件；`onnx_common.h`/`ge_onnx.pb.h` 仅从 57 个目标文件移除（公共头保留给 TF 插件与范围外插件）。
4. **属性通道约束**：插件侧只依赖 `op_src.GetAttr("attribute")` 的 JSON 字符串契约（3.1），不读取 protobuf 类型；FLOAT 值以字符串承载，`f` 缺省 = 0.0、`i` 缺省 = 0，不回落算子默认值。
5. **动态 IO 约束**：真实 ATC 下输入/输出数量取自源算子的 `input`/`output` 列表属性，UT 未序列化时回退 `GetInputsSize()/GetOutputsSize()`；回调内对目的算子的 `GetInputsSize()` 保持原样（唯一合法例外 `trilu`）。
6. **构建与使用方式不变**：产物仍为 `liboponnx_plugin_math.so`，通过 `ASCEND_CUSTOM_OPP_PATH` 注入 ATC，`--framework=5` 转 om；不要求用户改调用方式或环境变量布局。
7. **注册不可变**：算子名、`OriginOpType` 条目与顺序、`ParseOpToGraphFn` 有无与函数名、`ImplyType` 逐字保留；不得为通过某用例改动注册键或放宽判据。
8. **提交序列纪律**：每个增量必须被验证覆盖；F-C5/F-C6 等非功能性清理不进入提交序列（见 5.5/R6）。

---

## 4. 可维可测分析

### 4.1 精度标准/性能标准

本任务为 Host 侧等价重构（插件参数解析环节），判定标准全部沿用任务书与生态开源精度标准，未新增也未放宽任何阈值。

| 验收项 | 判定口径 | 口径来源（可核验） |
|---|---|---|
| **精度·主判据（等价性）** | 同一 onnx 模型、同一输入，迁移前（基线插件）与迁移后（提交版插件）各转 om 并推理一轮；两轮 `out_*.npy` **逐数值相等**（逐元素比较 + md5 逐位一致，浮点含 `NaN==NaN`，同时校验 shape/dtype）；**不设 rtol/atol，不以 `allclose` 作判据**（`allclose` 仅作辅助信息） | 任务书 §3.2 第 2 条 |
| **精度·辅助判据（数值正确性）** | om 输出 vs golden：逐元素 `abs(a-g) <= atol + rtol*abs(g)`；硬上限 `abs(a-g) <= max(fixed_limit, 32*ULP(abs(g)))`；整体 `matched_ratio >= 0.99` 且无硬上限违规；`NaN==NaN`、`±INF==±INF` 计匹配；**FP32 `rtol=2^-10`、`atol=2^-16`；FP16 `rtol=atol=2^-9`；BF16 `rtol=atol=2^-6`；整数/布尔逐位精确** | 任务书 §3.2 第 3 条；`test_cases/README.md` §4.1；opbase 混合容差标准；实现 `precision_standard.py`（9619 B，未修改） |
| **性能/内存标准** | **不涉及**：Host 侧等价重构，不改变算子计算逻辑与模型转换产物结构，不引入新算子实现，无新增算力/内存开销，不设指标 | 任务书 §3.3/§3.4 |

补充判定规则（同源，仅适配用例形态，不改变阈值）：平台容错按「迁移前后两轮行为一致（均失败且报错一致，或均成功且输出一致）」；`golden=false` 用例辅助判据退化为两轮一致；`expect_convert_failure`/`expect_atc=fail` 两轮都须转换期失败且报错签名一致，一轮失败一轮成功即 FAIL；辅助判据双口径（任务包口径 + opbase 口径）并列落盘，不静默改阈值、不互相覆盖。

### 4.2 测试用例规划

#### 4.2.1 framework UT 规划（可测性设计）

覆盖对象：57 个 framework UT 文件，与迁移范围 57 个插件文件 / 124 个条目一一对应；按目录分布 `common/` 14/81、`conversion/` 14/14、`math/` 25/25、`experimental/` 1/1、`random/` 3/3，合计 **57 文件 / 124 条目**，与 2.3 逐条一致。

**用例数——四种口径必须分开读（评审高风险点）**：

| 口径标签 | 数值 | 计算方式 / 证据 | 声明 |
|---|---:|---|---|
| **A. 静态 TEST 宏（源文件口径）** | **624** | 台账 `Σtextual_test_macros=624`；对 `submit/ut` 57 个文件独立 `grep TEST/TEST_F/TEST_P` 复核亦为 **624** | **静态宏数 ≠ gtest 运行数**，不得当作通过数/覆盖数 |
| **B. 任务书 §3.5 记载（57 文件映射口径）** | **612** | 任务书 §3.5 原文「57 个 UT 用例文件（612 个用例）」；`ut_per_file_diff.tsv` 逐行 `run_tests` 合计亦为 612 | **引用值**，非实跑基调 |
| **C1. 57 文件映射实跑** | **615** | 624 − 9（`test_max_experimental` 的 9 个宏因 `ENABLE_EXPERIMENTAL=OFF` 未编译）；或 648 − 33（8 个非目标 suite 合计 33 例） | 57 个文件实际对应 **56 个 suite / 615 例** |
| **C2. 过滤器实跑** | **648** | `--gtest_filter='Onnx*PluginTest.*'`：XML `tests="648" failures="0"`；stdout `Running 648 tests from 64 test suites` → `[  PASSED  ] 648 tests` | **通过率基调 = 648/648**；含 8 个非目标 suite 的 33 例（Correlation 6、NpuDtype 1、NpuFormatCast 1、NpuIndexing 7、NpuRotaryMul 1、OneHotD 8、ReverseSequence 8、Shape 1），**不是「57 个文件的用例数」** |

> **复核差异说明**：B 的 612 与按 gtest 输出修正 2 处 suite 映射后实得的 615 相差 3 例，差异已定位到 `ut_per_file_diff.tsv` 的 2 处 suite 映射异常——`test_onehot_onnx_plugin.cpp` 被映射到 `OnnxOneHotDPluginTest`（8 例，实为另一非目标文件）、`test_sequence_onnx_plugin.cpp` 被映射到 `OnnxReverseSequencePluginTest`（8 例，同为非目标文件）；两文件真实 suite 分别为 `OnnxOneHotPluginTest`（9 例）与 `OnnxsequencePluginTest`（10 例）。故 **612 仅作任务书引用口径**，本文实跑依据为 **615 / 648**。

**四类场景规划**（任务书 §3.5 第 3 条「Attr 生成规则」）：

| 场景 | UT 构造方式 | 期望 | 代表用例 |
|---|---|---|---|
| ① 属性齐全的正常解析 | `ge::Operator op_src; op_src.SetAttr("attribute", AscendString(json))`，JSON 覆盖全部候选属性 | `SUCCESS`，`op_dest` 属性键/值/类型逐一符合预期 | `OnnxOneHotPluginTest.ParseAxisAttribute`、`OnnxCastPluginTest.ParseToFloat16` |
| ② 属性缺失 / 默认值 | 仅给部分属性或键缺省（`i` 缺省、`f` 缺省） | `SUCCESS` 且按原语义取缺省值（`f` 缺省 = 0.0、`i` 缺省 = 0，不回落算子默认值）；「缺属性即失败」算子返回 `FAILED` | `OnnxOneHotPluginTest.UseDefaultAxisWhenAttributeEmpty`、`OnnxsequencePluginTest.SucceedWhenNoAttributeKey` |
| ③ 非法 attribute JSON / 无属性形态 | 截断/非法 JSON；或无 attribute 键的算子不传该键 | 解析类插件 `FAILED`；无属性插件按原省略语义 `SUCCESS`（豁免） | `OnnxOneHotPluginTest.ReturnFailedWhenAttributeJsonInvalid`、`OnnxsequencePluginTest.SucceedWhenAttributeJsonInvalid` |
| ④ 边界 / 校验 | 属性 type 不符、同名重复、空列表/负值透传、异常 dtype/取值 | 按迁移前逐文件判据返回（type 不符按缺失、重复取最后、Cast 不支持 dtype 返回 `FAILED` 等，逐文件照抄） | `OnnxOneHotPluginTest.IgnoreAttributesNotNamedAxis`、`OnnxCastPluginTest.ReturnFailedWhenDtypeUnsupported` |

- **用例组织**：按插件文件一对一（57 个 UT 文件），suite 名统一 `Onnx*PluginTest`；UT 直接 `#include "../../../framework/xxx_onnx_plugin.cpp"` 白盒直调静态解析函数，输入只用 `ge::Operator` + JSON 字面量，**不构造任何 protobuf 对象**——这是解耦后可测性提升的直接体现。
- **规模与侧重（名称口径）**：57 个文件共 624 个静态宏，其中用例名含 `ReturnFailed*` 的失败分支宏 **140** 个、`Ignore*`（type 不符按缺失）**72** 个；两类与其他类不互斥，不做加和。

#### 4.2.2 om 两轮验证规划（等价性主判据）

| 项 | 规划 |
|---|---|
| 用例来源 | 任务包 **56 个 `gen_model_*.py`** 生成脚本（计数复核 = 56），覆盖 **57 个插件文件**（`math/maximum` 与 `experimental/math/asinh` 两份 Max 插件共用 `gen_model_max.py`）；随包用例 **61 例** = 56 个主模型 + 5 个脚本自建变体（`clip_default`、`sequence/variants/sequence_construct`、`similar_opp_{hardswish,neg,where}`） |
| 自建补充用例 | 按任务书 §3.2 第 6 条/用例包修订许可补 **10 个 opset 变体**（`omcases/<op>/<variant>/` 预生成目录，锚定双注册另一分支与可选输入形态：Clip 11 三种端口形态，Pad/Squeeze/Unsqueeze/Slice 的 v11/v13 输入形态，Reduce 家族 axes 属性/输入两形态）⇒ 全量 **71 例 = 61 + 10**；不改任务包文件、不改 golden、不改阈值 |
| 一键脚本 | 任务包 `verify_om/two_round_compare.sh`（13710 B）：`ASCEND_CUSTOM_OPP_PATH` 注入本轮 `.so` → `--framework=5` 转 om → pyACL 推理落盘 → `compare_rounds.py` 两轮比对；`precision_standard.py`、`run_om_and_compare_acl.py` 均按包内原件使用 |
| 输入与属性 | 模型输入 numpy 随机（μ=0、σ=1）；属性取可区分值（Cast `to=10`、Squeeze `axes=[2]`、Flatten `axis=2`），保证解析错误在输出 dtype/shape/数值上可观测 |
| 判定口径（三类用例） | ① 常规：两轮 ATC 都成功、`out_*.npy` 逐位一致；② 预期失败：两轮都在转换期失败、报错签名一致（声明 `expect_error_contains` 须命中）；③ 行为一致（`golden=false` 且未声明预期）：两轮结论一致。非确定性随机算子（`multinomial`）以「两轮 shape/dtype 一致 + 同呈非确定性」判定，不使用数值/统计容差 |
| random 3 文件 | `random_uniform`/`random_normal` 为 0 输入模型，两轮以同一报错失败（`E16001 The model has no input node.`）→ 预期一致失败；`multinomial` 非确定性（int32 输出，两轮 md5 必然不同）→ 退化为行为一致；`random_uniform_like`/`random_normal_like` 实测两轮逐位一致 |

#### 4.2.3 插件生效证据判据（强/中/弱三级）

> 结论先行：**功能性正对照是主证据；`LoadCustomOpLib` 绝对路径日志是中间证据；`already registered, skip` 仅为辅助弱证据，不得单独作为「本轮生效的是我方插件」的证明。**

| 强度 | 判据 | 本批实测 |
|---|---|---|
| **强（主证据）** | **功能性正对照**：同模型、同注入方式、只换 `.so`，用「功能结果差异」证明注入插件决定转换结果 | ① **Addcmul 探针**：仅修注册键的探针插件（2492576 B）`rc=0` 且产 om，而 baseline / candidate / 我方 P1 全部失败 ⇒ 注入的插件确实在起作用（证据见 5.4 #14/#15）。② **Clip 形态正对照**：内置插件（不设 `ASCEND_CUSTOM_OPP_PATH`）在 `clip_v11_max_only` 上 `rc=0` 且与 pre 轮逐位相同，而候选注入时 `rc=255` ⇒ 该失败只可能由候选插件引起 |
| 中 | `LoadCustomOpLib:plugin load <本轮 opp 绝对路径>` 命中，且该行**先于**内置插件加载出现 | 71 例 × 2 轮 × 2 口径的 `plugin_evidence_*.txt` 均命中该行；任务包权威脚本侧另有 122 个证据文件、732 条含绝对路径的 `plugin load` 到 `success` 命中行（引用口径） |
| **弱（仅辅助）** | 仅 `Plugin of X already registered, skip` | **已量化证伪其单独证据力**：每个 tag-round 的 skip 行**恒为 14 条**，算子名集合（12 个，含 `PartitionedCall`×3）**在所有用例中完全相同且与用例无关**——这是内置插件之间的互相抢占噪声，不能定位到本轮 `.so` |

---

### 4.3 环境与依赖

> 口径：本表“实测”项均于 **2026-10-19** 在交付主机（SSH 别名已脱敏）现场命令复核，并与环境实测记录（内部留档，冻结于 2026-10-16 20:34:39）及 `runtime.json` 交叉一致；“**仓内自带**”指随 ops-math 仓 `third_party/` 提供、随仓版本走，“**环境安装**”指主机/容器预装、换机必须重采。本次仅为只读采集，未写入交付树。

| 项目 | 版本或取值 | 来源与核验方式 | 说明 |
|---|---|---|---|
| NPU 型号与数量 | `Ascend910B3 ×4`；物理 `npu-smi` id **2/3/4/5**；每卡 HBM 65536 MiB；HCCS 全互联；Health OK | `npu-smi info`（需设备锁，环境实测记录（内部留档））；`runtime.json` → `hardware_observed` | 【环境安装】容器内 `ASCEND_VISIBLE_DEVICES=2,3,4,5`；运行时接受的是可见集合**下标**而非物理卡号（见坑 3）。 |
| 主机与算力 | Atlas 900 RCK A2 Compute Node（BIOS 7.36）；191 核 / 1509 GiB 内存（≈1.5 TiB） | `nproc`、`free -g`（现场实测）；`runtime.json` → `physical_host` | 【环境安装】构建与 NPU 同机；`-j8` 远低于机器并行上限，为任务书口径命令。 |
| OS 与架构 | Huawei Cloud EulerOS 2.0（`aarch64`）；内核 `5.10.0-60.18.0.50.r865_35.hce2.aarch64`；64 位 | `cat /etc/os-release`、`uname -srm`、`getconf LONG_BIT`（现场实测） | 【环境安装】aarch64（snt9b），不可用 x86 产物替代。 |
| 驱动 | `Version=24.1.0.3`；`Innerversion=V100R001C19SPC005B220`；`ascendhal_version=7.35.23` | `cat /usr/local/Ascend/driver/version.info`（现场实测）；`docs/ENV_REPORT.md §1` | 【环境安装】与 CANN 9.1.0 配套（`compatible_version` 含 `V100R001C19`）。 |
| 固件与 npu-smi | 板卡固件 `7.5.0.5.220`；`npu-smi 24.1.0.3` | `npu-smi -v`（版本，现场实测）；`npu-smi info`（固件字段，需设备锁）；`runtime.json`、环境实测记录（内部留档） | 【环境安装】按设备锁纪律本片段未复采固件字段，取台账冻结时采集值；`npu-smi -v` 现场复核为 24.1.0.3。 |
| CANN Toolkit | `Ascend-cann-toolkit 9.1.0`；`innerversion V100R001C11SPC001B243` | `cat /usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info`（现场实测，读 `version`/`innerversion`） | 【环境安装】`/usr/local/Ascend/ascend-toolkit/latest` → `cann` → `cann-9.1.0`；机上另有 `cann-8.5.2`/`cann-9.0.0` 目录但未生效，**一律以该 info 文件为准**；`atc --version` 不支持（rc=255），不得用其取版本。 |
| Python 解释器 | `/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python3.12`；Python **3.12.0** | `$PY -V`、`$PY -c "import sys;print(sys.version)"`（现场实测）；`runtime.json` → `interpreter` | 【环境安装】onnx/om 链路**只能**用此解释器；`/usr/bin/python3`=3.9.9 且无 onnx，禁止使用。 |
| onnx | **1.17.0** | `$PY -c "import onnx;print(onnx.__version__)"`（现场实测） | 【环境安装】满足任务书 §3.1.3 建议下限 `onnx ≥ 1.14`。 |
| onnxruntime | **1.19.2** | `$PY -c "import onnxruntime;print(onnxruntime.__version__)"`（现场实测） | 【环境安装】满足建议下限 `onnxruntime ≥ 1.16`。 |
| numpy | **1.26.4** | `$PY -c "import numpy;print(numpy.__version__)"`（现场实测） | 【环境安装】满足建议下限 `numpy ≥ 1.21`。 |
| nlohmann/json | **3.12.0**（`NLOHMANN_JSON_VERSION_MAJOR/MINOR/PATCH = 3/12/0`） | 仓内 `third_party/lib_cache/json/single_include/nlohmann/json.hpp` 版本宏；源包 `third_party_offline/json-3.12.0.tar.gz`；构建日志 `CMAKE_ARGS` 中 `CANN_3RD_LIB_PATH` | 【仓内自带】**不新增、不升级**；third_party 由 `bash tools/seed_third_party.sh <tree>` 从离线缓存播种（第三方站 OBS 不可达），每个工作树首次构建前必须播种。 |
| protobuf / `ge_onnx.pb.h` | 迁移后插件侧引用数 **0**（`onnx_common=0`、`ge_onnx.pb=0`）；仓内 `third_party/protobuf/`（25.1）原样保留 | 证据 `evidence/t37-candidate-*/candidate/target_check.tsv`；仓内 `third_party/protobuf/`、`third_party_offline/protobuf-25.1.tar.gz` | 【仓内自带】本任务目标即插件不再 include `onnx_common.h`/`ge_onnx.pb.h`（见 3.2.2）；未新增、未升级任何三方依赖。 |
| ATC 关键参数：`--framework` | `--framework=5`（原始框架 = ONNX） | 任务包 `test_cases/verify_om/verify_op.sh:53`、`two_round_compare.sh:129`；任务书 `ops-math_onnx_task_doc.md:249` | 【环境安装】调 CANN 9.1.0 自带 `atc`；`5` 即 ONNX（与 CANN 官方 onnx 适配调用样例一致）。 |
| ATC 关键参数：`--soc_version` | `--soc_version=Ascend910B3` | `verify_op.sh:24` → `SOC_VERSION=${3:-Ascend910B3}`（用例包 ATC 命令的默认卡型取值） | 【环境安装】与实测卡型一致；Ascend 950 场景为 `--soc_version=ascend950`（见 3.4，非本次实测机，未实测）。 |
| ATC 注入：`ASCEND_CUSTOM_OPP_PATH` | 指向目录含 `<opp>/framework/liboponnx_plugin_math.so`（另推荐 `<opp>/op_proto/`） | `tools/prepare_opp_round.sh:5-13,25-26`；`two_round_compare.sh:127-129`（`ASCEND_CUSTOM_OPP_PATH="$opp" atc` 命令）；环境实测记录（内部留档） | 【环境安装】以环境变量前置注入，使 ATC 使用被测插件 `.so`；`op_proto/` 非必需（功能性正对照两种布局 8/8 成功）但推荐；插件是否生效以“功能性正对照为主、plog 为辅”判定。 |
| 构建（插件 `.so`） | `bash build.sh --onnxplugin -j8` → `build/liboponnx_plugin_math.so` | `scripts/build_options.sh:182-193`（`--onnxplugin` 定义）；交付件「1 自验证步骤说明」步骤一；环境实测记录（内部留档） | 【环境安装】`-j` 仅并行度（实测用 `-j64`：首次 117 s / 增量 44 s）；**退出码恒 0**（`build.sh` 尾部管道），判成败看日志 `Build libs success` + 产物文件（见坑 4）。 |
| UT 运行 | `ASCEND_CUSTOM_OPP_PATH= ./build/tests/ut/op_host/math_op_host_ut --gtest_filter='Onnx*PluginTest.*'` | 交付件「1 自验证步骤说明」步骤三；`tools/fill_submission.py:648`；`docs/ENV_REPORT.md §4` | 【环境安装】命名契约 `Onnx<Op>PluginTest`（迁移后 57 文件）；前置 `ASCEND_CUSTOM_OPP_PATH=` 清空以避免宿主注入干扰；用例数与通过数口径见 4.2 与第 5 章。 |

**环境坑与规避（5 条）**

1. **长 `TMPDIR` 触发 atc `AF_UNIX path too long`（`EC0009`，rc=255）** —— `TMPDIR` 约 75 字符即可触发。规避：统一 `export TMPDIR=/cache/oam-tmp`（短路径且不落 `/`）。来源：环境实测记录（内部留档）。
2. **ATC 默认精度模式是 `force_fp16`，不是 fp32** —— `atc --help` 原文为 `force_fp16(default)`；直接与 fp32 golden 比会出现“精度失败”假象（实测 47 例辅助判据失败中 36 例仅加 `--precision_mode=force_fp32` 即在双口径下转 PASS）。规避：验收口径保持默认 fp16，fp32 复跑单列为**补充口径**（不得写成“放松阈值”，不改 golden/阈值）。来源：环境实测记录（内部留档）、`tools/verify_fp32_caliber.sh`。
3. **`ASCEND_RT_VISIBLE_DEVICES` 是“可见集合下标”，不是物理卡号** —— 物理 2/3/4/5 对应下标 0/1/2/3；写 `=2` 会落到物理卡 4 且无任何报错。症状：rc=207004 或 `acl.rt.get_device_count()` 返回 0。规避：用 `eval "$(tools/npu_lock.sh acquire <物理卡号> --eval)"` 由工具导出正确下标。来源：环境实测记录（内部留档）、`locks/dev<物理卡号>.lock`。
4. **`build.sh` 退出码恒 0，不能当成功判据** —— 尾部 `main "$@" 2>&1 | while IFS= read -r line` 使 `$?` 来自 `while` 而非构建。规避：判据 = 日志 `Build libs success` + 产物文件存在；UT 构建另看 `UT BUILD END` 的 rc 字段。来源：环境实测记录（内部留档）、`repo/build.sh` 末行。
5. **`ASCEND_SLOG_PRINT_TO_STDOUT=1` 会静默作废 plog 重定向** —— 设了它之后 `ASCEND_PROCESS_LOG_PATH` 不生效，run 目录 plog 文件数变 0，NPU 证据作废。规避：只设 `ASCEND_PROCESS_LOG_PATH=/cache/workspace/ops-math-ONNX/ascend_log`，**不设** `ASCEND_SLOG_PRINT_TO_STDOUT`。注：任务包 `two_round_compare.sh:127` 为捕获 stdout 仍显式设了该变量（仅该脚本的对比用途可接受）；我方流水线 `tools/verify_pipeline.sh:506` 按环境实测记录（内部留档）只设 `ASCEND_PROCESS_LOG_PATH`。来源：环境实测记录（内部留档）。


表中每个版本号均可由“现场命令”或“仓内/台账文件路径”二选一回溯（见各行“来源与核验方式”），无凭记忆填写项；复核入口示例：`cat /etc/os-release`、`cat /usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info`、`cat /usr/local/Ascend/driver/version.info`、`npu-smi -v`、`npu-smi info`（需设备锁）、`$PY -c "import onnx,onnxruntime,numpy;..."`。

三条口径：①“环境安装”行以本机实测为准，换机/换镜像必须重采；②“仓内自带”行随仓版本走（ops-math 基线 commit `91c6c1715`，见文档头），不随环境变化；③与任务书 §3.1.3 的差异——任务书文档“实测组合”为 Python 3.11.15 / onnx 1.23.2 / onnxruntime 1.31.0 / numpy 2.4.6，本机为 **Python 3.12.0 / onnx 1.17.0 / onnxruntime 1.19.2 / numpy 1.26.4**，满足建议下限但组合不同，已在交付件「1 自验证步骤说明」与「2.2 精度自验证日志」如实声明（`docs/DELIVERY_COMPLIANCE.md §2`），不影响任何判据。

> 与 2.2「外部组件依赖」表的分工：2.2 只列“用了什么组件、用途、来源（仓内/外部）”，本表列“实测环境的具体版本与核验入口”；两表同一事实引用同一来源，不各写一套数字。

---

### 4.4 兼容性分析

本节口径：迁移只替换插件内部的**参数解析通路**（旧 `ParseParamsFn(protobuf::Message&, ge::Operator&)` → 新 `ParseParamsByOperatorFn(const ge::Operator&, ge::Operator&)`）与**属性来源**（protobuf 属性 → `attribute` JSON），**注册要素、产物文件名与加载方式、om 转换与推理行为均保持不变**；旧 protobuf 回调签名在框架侧继续可用（范围外旧插件不受影响），迁移后插件对该签名的依赖降为 **0**。

#### 4.4.1 注册兼容

| 注册要素 | 迁移前 → 迁移后 | 校验结论 |
|---|---|---|
| `REGISTER_CUSTOM_OP("<OpName>")`（算子名/条目）、`.FrameworkType(ONNX)` | 不变 | 57 文件全部存在、名称集合不变：`SPEC_CONFORMANCE.md` §B「356 条断言 356 PASS / 0 FAIL」 |
| `.OriginOpType({...})`（opset 集合与条目顺序，含双注册多分支） | 不变 | 同上 §B/§C（`OriginOpType` 为注册集合、`original_type` 为写入值，两者分开比对） |
| `.ParseOpToGraphFn(...)`（有无与函数名） | 不变 | `conformance_facts.tsv` 回调列；`SPEC_CONFORMANCE.md` §C-C6（`trilu` 回调内读 `GetInputsSize`，非 protobuf 依赖） |
| `.ImplyType(...)` | TVM 52 / GELOCAL 4 / AI_CPU 1 → 同左 | `SPEC_CONFORMANCE.md` §B 计数分布（52+4+1 = 57 文件） |
| **解析回调注册点** | `.ParseParamsFn(F)` → `.ParseParamsByOperatorFn(F)` | 新注册点 **79/79** = 基线调用点数；`.ParseParamsFn(` 残留 **0**（57 个目标文件） |
| 动态输入/输出注册 | `node->input_size()`/`output_size()` → `DynamicInputRegister`/`DynamicOutputRegister` + `GetSourceInputNum/GetSourceOutputNum` | `dyn_in=38`/`dyn_out=35`；`input_size()=0`、`output_size()=0`（`SPEC_CONFORMANCE.md` §B） |
| 节点名语义 | `node->name()` → `GetOpName(op_src)`（空名按文件「回落目标算子名」或「写空串」） | `node_name=0`；`getopname=380`、`fallback_dest=188`；逐文件差异见《迁移规范》§5.7 |
| 注入与加载方式 | `ASCEND_CUSTOM_OPP_PATH` → `framework/liboponnx_plugin_math.so` → 同左 | 四段兼容实验每段均命中本地 opp 路径与 `LoadCustomOpLib:plugin load` 到 `success`（`COMPAT_COMPLIANCE.md` §3） |

**结论**：对 GE/ATC 而言插件注册表内容（算子名、opset 集合、`ImplyType`、回调类型、拆图回调、动态 IO）**逐点等价**，UT 与 ATC 注入链路均无需改动。**唯一已知且有意不修的注册缺陷**是 `OriginOpType` 裸键（F-C3，上游既有缺陷，迁移侧不做改动，见 5.5）。

#### 4.4.2 产物兼容

插件产物仍是**同名** `liboponnx_plugin_math.so`，`DT_NEEDED` 仅含系统运行库（实测提交系列 7 项：`libc_sec.so`、`libregister.so`、`libunified_dlog.so`、`libstdc++.so.6`、`libgcc_s.so.1`、`libc.so.6`、`ld-linux-aarch64.so.1`），**不含 protobuf 实现库、不含 compat 兼容库**；两代 `.so` 均带 `BIND_NOW`，差异只在**未定义符号**：

| 对象 | 大小 | sha256 | 未定义 `onnx`/`protobuf`/`absl` 符号 | `DT_NEEDED` 含 protobuf/compat |
|---|---|---|---|---|
| 迁移前 `baseline/pre/liboponnx_plugin_math.so` | 2492576 B | `ed2430c4af72b27061aea51e16fb7e15951ed612321cd2a5d28db76346876a52` | **15** | 否 |
| **迁移后·提交系列** `work/submit-decouple/build/liboponnx_plugin_math.so` | 2361504 B | `044a1d6c5b79db82b3afb78288f870e83e96a0878e6412df4064a1a8e5893263` | **0** | 否 |
| 上游候选原样 `candidate/pr6355/liboponnx_plugin_math.so`（F-C2 未修，`clip_v11_max_only` `rc=255`） | 2361504 B | `787e84bc9e989c722a9b43f1fecedd4b8f7c8126b72a5580e0a88532fab73fb7` | 0（解耦已完成） | 否 |

> 15 个符号 = 9 个 `ge::onnx`/protobuf 类型符号（`AttributeProto` 构造/析构/`MergeImpl`、`RepeatedPtrFieldBase`、`ThreadSafeArena`、`typeinfo` 等）+ 5 个 abseil log 符号（`lts_ascend_private`）+ 1 个旧回调符号 `domi::OpRegistrationData::ParseParamsFn(...)`。核验命令：`readelf -d <so>`（看 `DT_NEEDED`）、`nm -D --undefined-only <so>`（筛 `onnx/protobuf/absl`）、`ldd -r <so>`；两代 `.so` 均仅 1 个未解析符号 `rtGetSocSpec`（NPU 运行时符号，由 atc 进程自身提供，与 protobuf 无关）。

**转换产物与推理兼容**：

| 判据 | 结果 | 证据 |
|---|---|---|
| ATC 转换可产出 om | 迁移前后插件均 `rc=0` 且产出 `model.om`（含**不挂** compat 的边界证明） | `COMPAT_COMPLIANCE.md` §3 |
| 两轮输出逐值一致（主判据） | **71 例主判据 FAIL=0**：默认口径 58/71 两轮 `out_*.npy` 逐字节一致；2 例 host-CPU 放置经 fp32/定点补证后逐位一致；余 11 例「两轮行为一致」逐例归类（7 例一致转换失败 + 3 例一致运行时不可推理 + 1 例随机项 `multinomial`） | 5.2、5.4 |
| 包内权威脚本全量 | 56/56 脚本 → 61 case；硬结论 **`TWOROUND_FAIL_OTHER = 0`**（整体 FAIL 1 例 `multinomial` 属非确定性，非「两轮不一致」） | 5.2.3 |
| om 字节级等价 | **不主张**：四段实验（迁移前/后 × 挂/不挂 compat）产出的 om sha256 互不相同 ⇒ 等价性判据是「转换成功 + 推理输出逐位一致」，**不是** om 文件逐字节相等 | `COMPAT_COMPLIANCE.md` §0.4 |

**下游无感知**：插件文件名、注入目录布局（`opp/framework/*.so`）、ATC 调用方式与 pyACL 推理流程均不变；om 结构与推理输入输出 dtype/shape 由逐例 `cases/*.json` 与两轮比对作证。

#### 4.4.3 编译兼容

- **include/依赖**：57 个目标文件公共头统一换成 protobuf-free 的 `common/inc/framework/math_onnx_plugin_util.h`；`ge_onnx.pb.h`/`onnx_common.h` 引用残留 **0 / 0**（`SPEC_CONFORMANCE.md` §B，`inc_util=57`）；仍需要的组合算子头（`*_proto.h`、`stub_ops.h`、`op_math_proto_extend.h`）按原文件保留。
- **不新增三方依赖**：JSON 用仓内 `third_party` 自带 nlohmann/json，沿用既有公共头，不新增公共头文件。
- **构建命令**：插件 `bash build.sh --onnxplugin -j64`；UT `bash build.sh -u --ophost --noexec -j64`；从零构建 `error:` 计数 **0**、`[100%] Built target math_op_host_ut`。
- **UT 实跑**：`--gtest_filter='Onnx*PluginTest.*'` → `648 tests from 64 test suites`、**passed 648 / failed 0**、rc=0（XML `failures=0`）；静态宏 624 与实跑 648 为不同口径（`test_max_experimental` 9 条未编译，映射实跑 615，见 4.2.1）。
- **平台约束**：aarch64 + CANN 9.1.0；`build.sh` 退出码恒为 0，故构建成败只看日志 `Build libs success`/`Built target` 与产物文件，不看 rc。

#### 4.4.4 对旧接口无破坏

1. **旧接口仍然可用**：迁移前插件（旧 protobuf 回调）在不挂 compat 时 `atc rc=0` 且产出 om ⇒ 旧回调签名 `domi::OpRegistrationData::ParseParamsFn(...)` 仍由宿主运行时闭包解析（该符号未被 `libgraph_base.so` 导出，由闭包内其它库补齐）。
2. **新插件对旧接口依赖为 0**：上述旧回调符号与 `AttributeProto` 族符号在迁移后 `.so` 的未定义符号表中均为 0；`.ParseParamsFn(` 残留 0。
3. **范围外既有能力不受影响**：公共头 `common/inc/framework/onnx_common.h` **保留在仓内**（未被删除），仅不再被 57 个目标文件引用；范围外既有 onnx 插件与其 UT 在同一遍实跑中被覆盖（`Onnx*PluginTest.*` 中除 57 目标文件映射的 56 个 suite/615 例外，另有 8 个非目标 suite/33 例，XML `failures=0`）。
4. **多算子/双注册文件逐点替换**：`similar_opp` 单文件注册 63 个算子、双 opset 分支文件均按**每个注册点**替换，新注册点 79/79 对齐基线调用点数，无「只改一支」残留。
5. **红线**：不得为「让某模型转过去」而改动注册键或放宽判据；迁移后能转换成功而基线失败者一律判 **FAIL** 并升级。

### 4.5 风险与降级预案

风险 R1–R7 及降级预案如下（「降级」均指判据降级并显式声明，不是放宽阈值）。

| # | 风险 | 降级预案与判定条件 | 证据 |
|---|---|---|---|
| **R1** | 迁移前旧接口插件加载报 `undefined symbol: AttributeProto`（示例），任务包提供 `libge_onnx_compat.so` 兜底 | 若需兜底，仅用于**迁移前轮**自验，不进仓库/提交序列；验收要求：迁移后插件不挂任何 compat 库即 `rc=0` 且产 om、未定义 protobuf/onnx 符号 = 0；本环境实测不需要 compat | `COMPAT_COMPLIANCE.md` §0–§3 |
| **R2** | 仓内 abseil 内联命名空间与 CANN 运行库不一致 | 本环境已一致，无需改 `absl/base/options.h`；判据：`nm -D --undefined-only` 命中 `lts_ascend_private`；出现其它 `lts_*` 才改宏重编；禁止替换运行库/`patchelf` 绕过 | `ENV_REPORT.md` §2 |
| **R3** | 个别算子在当前 CANN/SOC 无法 ATC 转换，易被误猜为「op-info 缺条目」 | 逐字记录原文（算子/optype/错误码/行号）；该用例判据降级为「两轮行为一致」（`atc rc` 相同且报错逐字相同）；基线侧同失败者写为平台限制，不归因候选；迁移后成功而基线失败 ⇒ FAIL 升级 | `ADDCMUL_BASELINE.md`；5.2.5 |
| **R4** | `attribute` JSON 形态与契约不符（键缺省、`f` 字符串、`type` 缺省/不符、两种遍历混用） | 逐文件按迁移前形态保留、禁止统一化；FLOAT 必须显式校验（`atof("abc")` 静默返回 0 而 UT 要求 FAILED）；`f` 缺省 = 0.0、`i` 缺省 = 0；非法 JSON → FAILED 且先于任何写出；UT 期望值不变 | 3.1/3.2.1/3.2.3 |
| **R5** | 本地/评审环境无 CANN+NPU，无法做 om 两轮与推理精度 | 逐级降级并声明覆盖边界：①静态结构与注册一致性（356/356、79/79、残留 0）；②二进制扫描（`readelf -d`/`nm -D --undefined-only`/`ldd -r`：15→0）；③host 侧 ATC（四段 `rc=0`）；④host 侧 UT（648/648）；⑤无 NPU 模型预检（`onnx.checker` PASS 53/61，8 个 FAIL 均为 NPU 自定义算子无 ONNX schema）；⑥缺 NPU 时不得主张精度/两轮逐位结论 | `COMPAT_COMPLIANCE.md` §8；`ENV_REPORT.md` §0.3 |
| **R6** | 低危清理（F-C5 重复 include、F-C6 行尾）若并入提交序列，会改变 `.so` 的 `.text` 位移 | 移出提交序列，仅作独立清理建议；原则「提交序列内每个增量必须被验证覆盖」；`.so` 三态：候选 `787e84bc` → 序列状态 `044a1d6c` → 移出后 `bab4da8b` | `DELIVERY_FINAL.md` §3/§5 |
| **R7** | ATC 默认 `force_fp16` 使「与 golden 比对」的辅助判据大量失配，易被误读为迁移回归 | 主判据只看两轮输出逐位一致；辅助判据双口径并列，需严格比对时显式 `force_fp32` 并单独归档；禁止以「放宽阈值/改 golden」收尾 | 环境实测记录（内部留档） |

---

## 5. 实测验证结果（自验证据）

> **口径声明**：实测环境 Ascend910B3 + CANN 9.1.0（aarch64），环境明细见 4.3；本章数字均指向交付树内已归档证据（路径见 5.4），哈希为撰写时对盘上文件现算、非手抄（5.1 的 UT 二进制除外，采用运行时刻/盘上现值/同尺寸三态说明）；UT 与 om 两轮为本方执行并归档，撰写阶段未重跑实机实验，引用他人结果处标注「引用（未亲验）」。被测 `.so` 三方：pre = 冻结基线 `baseline/pre/liboponnx_plugin_math.so`（2492576 B / `ed2430c4af72b27061aea51e16fb7e15951ed612321cd2a5d28db76346876a52`）；post = 提交版本（F-C2 修复后）`build/liboponnx_plugin_math.so`（迁移后提交树）（2361504 B / **`044a1d6c5b79db82b3afb78288f870e83e96a0878e6412df4064a1a8e5893263`**）；候选原样 = `candidate/pr6355/liboponnx_plugin_math.so`（2361504 B / `787e84bc9e989c722a9b43f1fecedd4b8f7c8126b72a5580e0a88532fab73fb7`）。证据表内文件名中的 `T*` 前缀为内部归档批次编号，不影响结论；正文中形如 `787e84bc` 的 8 位十六进制为 sha256 前缀（完整值对表内文件现算）。

### 5.1 构建与 UT

| 项 | 实测 | 证据 |
|---|---|---|
| 构建树 | F-C2 修复树（分支 `fix/f-c2`，HEAD `7197348`，1 file changed/+29−7）+ 覆盖我方提交 UT 集（overlay 后 57/57 与 `submit/ut` 逐字节一致） | `evidence/t71-full-rerun-20261016T105109/full-rerun/ut/overlay_sha256_ut2.txt` |
| 构建命令 | `cd work/t71-ut2 && bash build.sh -u --ophost --noexec -j64` | `build_ut2.log` |
| 构建结果 | `error:` 计数 **0**；`[100%] Built target math_op_host_ut`；rc=0 | `build_ut2.log`、`ut/ut_run_ut2.txt`（903 B） |
| UT 二进制 | `build/tests/ut/op_host/math_op_host_ut`：**268,721,328 B**；sha256 三态：运行时刻（11:01）`9cb6275e`、盘上现值（11:19 重建后）`df969289`；UT 结论由 XML（`tests=648 failures=0`）与运行日志证明，不依赖二进制哈希 | `ut_run_ut2.txt`、`ut/run_onnx_plugin.xml`、`ut/run_onnx_plugin.log` |
| `--gtest_list_tests` | rc=0，**3276 行 / 405 suite**（全二进制口径，含非 ONNX suite） | `ut/list_tests.txt`（109230 B） |
| 运行 | `--gtest_filter='Onnx*PluginTest.*' --gtest_output=xml:ut/run_onnx_plugin.xml`；`Running 648 tests from 64 test suites` → **`[  PASSED  ] 648 tests`**，0 FAILED，rc=0 | `ut/run_onnx_plugin.log`、`ut/run_onnx_plugin.xml`（152943 B） |
| 通过率基调 | **648 / 648 = 100%**（XML `tests="648"`、`failures="0"`、`n_suites_failing=0`） | `ut/ut_summary.json`（348 B） |
| 逐文件对账 | `ut/ut_per_file_diff.tsv` 57 行：`Σstatic_macros=624`、`Σrun_tests=612`（含 2 处 suite 映射异常）；57 文件 → 56 suite / **615 例** | `ut_per_file_diff.tsv`（6708 B） |

**口径对账**：**624**（静态宏）/ **612**（任务书 §3.5 引用口径）/ **615**（57 文件→56 suite 实跑）/ **648**（`Onnx*PluginTest.*` 实跑 = 615 + 33）四口径的含义、出处与差异（含 2 处 suite 映射异常、8 个非目标 suite）统一见 4.2.1 的四口径表，本节不重复；57 文件实跑 615 例与 648 例的关系见 **4.2.1 四口径表**。

**实验性 Max（9 例）的独立补充口径**：默认构建不编译该文件；在 `ENABLE_EXPERIMENTAL=TRUE` 的独立实验口径下 `OnnxMaxExperimentalPluginTest` **9/9 PASS**（引用，未亲验）。该口径会把 ONNX 覆盖收窄为 188 例/20 suite，故**不并入默认 648 口径**。

**结论**：提交版本在默认 UT 构建口径下 **648/648 全过**（XML `failures=0`），无未解释失败；实验性 Max 的 9 条静态宏由独立实验口径补齐并通过。

### 5.2 两轮 om 逐位比对

#### 5.2.1 规模与身份

- 全量 **71 例 = 61 例（56 个随包主模型 + 5 个脚本自建变体）+ 10 个 opset 变体**；每个用例跑 **pre（基线 `.so`）/ post（提交版 `.so`）两轮**，同模型、同输入、同 ATC 参数。
- 两个独立 harness 交叉：**我方 harness**（`tools/verify_pipeline.sh om`，运行窗口 2026-10-16 10:51–12:03）与**任务包权威脚本**（`two_round_compare.sh` 原样使用，引用）；两者 pre/post 为同一对 `.so`，构成独立交叉验证。
- 设备与卫生：Ascend910B3 ×4（物理 2/3/4/5 → RT 0/1/2/3），`ASCEND_RT_VISIBLE_DEVICES` 用 RT 下标，pyACL 一律 `--device 0`；`TMPDIR=/cache/oam-tmp`；plog 只设 `ASCEND_PROCESS_LOG_PATH`（不设 `ASCEND_SLOG_PRINT_TO_STDOUT`）。

#### 5.2.2 我方 harness 全量结果（post = 提交版本）

| 口径 | 两轮逐位一致 | 说明 |
|---|---:|---|
| 默认精度模式（`force_fp16`） | **58 / 71** | 另 2 例（`eyelike`、`sequence_variants_sequence_construct`）因 ATC host-CPU 放置产出 `model_linux_aarch64.om`，默认 harness 按 `model.om` 推理而「无输出」；已由 fp32 口径 + host-CPU 定点补证解决 ⇒ 合计 **60 / 71** |
| fp32 全量口径（`force_fp32`） | **60 / 71** | 同一 11 例非逐位；上述 2 例在本口径下逐位一致（`eyelike` md5 `ab0b0e97c19b9302`、`sequence_construct` md5 `16a4d3a1dac051b4`） |
| **合并判定（主判据）** | **71 / 71 无「两轮行为不一致」** | ⇒ **主判据 FAIL = 0**；58 + 2 = 60 = fp32 口径的 60，口径自洽 |

**11 例非逐位一致用例的逐例归类（均为「两轮一致」的例外，非迁移差异）**：

| # | 用例 | 两轮行为 | 归类（原始报错摘要） |
|---:|---|---|---|
| 1 | `addcmul` | 两轮均 ATC 失败，同一报错 | 预期一致失败（`E13010 No operator plugin is registered for Op: addcmul_node`） |
| 2–4 | `sequence`、`split_to_sequence`、`concat_from_sequence` | 两轮均不能推理，同一错误 | 序列型输出的运行时限制（`[error] malloc(0) 失败 ret=100000`） |
| 5–6 | `random_uniform`、`random_normal` | 两轮均 ATC 失败，同一报错 | 0 输入模型特性（`E16001 The model has no input node.`） |
| 7–10 | `reduce_sum_square`、`reduce_log_sum_exp`、`deformable_offsets`、`repeat_with_pad` | 两轮均 ATC 失败，同一报错 | 平台/算子原型或引擎限制（基线侧同失败） |
| 11 | `multinomial` | 非确定性，逐位一致不可达 | 随机算子（两轮 shape/dtype 一致 + 同呈非确定性，不使用容差） |

> 以上 11 例 **pre/post 行为完全一致**，没有一例是「一轮成功/一轮失败」⇒ 主判据 FAIL=0。证据：`evidence/t71-full-rerun-20261016T105109/full-rerun/` 下 `T71_SUMMARY.json`（50055 B）、`T71_CASES.csv`（10520 B）及逐例产物。

#### 5.2.3 任务包权威脚本全量交叉（引用，未亲验）

| 项 | 实测 |
|---|---|
| 脚本覆盖 | **56 / 56** 个 `gen_model_*.py` 全量；对应 **61 个 case** |
| 逐例判定 | 两轮判定 PASS **46** / FAIL **1**（`multinomial`，非确定性 + 整数输出口径）/ 未判定 **14**（无输出 12 + 随机 `*_like` 2 例） |
| 分类 | `BIT_IDENTICAL` **48**（46 判 PASS、2 例随机 `*_like` md5 逐位一致但不判 PASS）、`NONDET_INT_FAIL` **1**、`NO_OUTPUT_NOT_EVAL` **12** |
| **硬结论** | **`TWOROUND_FAIL_OTHER = 0`**——除随机算子口径限制外，两轮无真不一致 |
| 与我方 harness 交叉 | **SAME 48 / DIFF 1**（`multinomial`）/ 无输出 12 ⇒ 除随机算子外逐例 md5 完全一致 |
| 证据 | `evidence/t74-two-round-full-20261016T105205/two-round-full/CASE_TABLE.csv`（17060 B）、`STATS.json`（236 B） |

#### 5.2.4 F-C2 修复的实测闭合（修复前 → 修复后）

| 用例 | 修复前（post = 候选原样 `787e84bc`） | 修复后（post = 提交版本 `044a1d6c`） |
|---|---|---|
| `clip_v11_max_only`（opset≥11，`input=["X","","MAX"]`） | post **rc=255 / 无 om**（`E19999 Assert ((src_out_anchor) != nullptr) failed[ReLinkInputDataEdge][graph_utils.cc:115]`） | post **rc=0 / 产 om**；默认口径两轮 md5 逐位相同；golden `min(x,max)` matched_ratio=**1.000000**；fp32 口径双口径两轮全 PASS |
| `clip_v11_min_only` / `clip_v11_both` / `clip_default` | 逐位一致 | 逐位一致（**无回归**） |

功能性正对照同时成立：内置插件在 `clip_v11_max_only` 上 `rc=0` 且与 pre 逐位相同 ⇒ 该失败只可能由候选实现引起、修复后消失。

#### 5.2.5 平台容错与一致失败项（逐例申报）

本批 11 例「非逐位」全部按任务书 §3.2 第 5 条以**两轮行为一致**判定，并在验收说明中逐例申报：**一致转换失败 7 例**（addcmul、random_uniform、random_normal、reduce_sum_square、reduce_log_sum_exp、deformable_offsets、repeat_with_pad）+ **一致运行时不可推理 3 例**（sequence、split_to_sequence、concat_from_sequence）+ **非确定性 1 例**（multinomial）。无一项因失败被归因平台而放宽判据；均保留 ATC/推理原始报错。

### 5.3 辅助精度判据

辅助判据以「om 输出 vs 用例 golden」按 4.1 混合容差标准执行，**双口径并列**（任务包 `precision_standard.py` + 生态 opbase 口径），不放宽任何阈值。

| 口径 | 双口径两轮全 PASS | 说明 |
|---|---:|---|
| 默认 fp16 | **15 / 71** | 其余 56 例绝大多数为 fp16 量化/65504 溢出造成的辅助失配，**pre 与 post 判定逐例完全一致**（61 例 pre/post 双口径判定不一致数 = 0）⇒ 与迁移无关 |
| **fp32 全量** | **56 / 71** | 其中 `random_uniform`/`random_normal`/`multinomial` 3 例为 `meta.golden=false`（无 golden ⇒ 辅助判据不适用）⇒ **真正数值双口径通过 = 53 例** |

**fp32 下仍不全 PASS 的 15 例归类**（引用，未亲验）：NPU 自定义算子 golden 约定缺陷 4 例（`npu_sign_bits_pack/unpack`、`phony_concat/split`）、既有基线侧差异 3 例（`mod` 的 C `fmod` vs ONNX `%`、`argmin` 输出 dtype `int32` vs ONNX/ORT `int64` 但数值全等、`einsum` fp32 matched_ratio=0.814491）、序列型不可推理 3 例、预期一致失败 1 例、平台/原型/引擎限制 4 例。

**残余 11 例的最终终态**（引用，未亲验）：`clip_v11_max_only` 修复后复验通过；`npu_sign_bits_pack/unpack` 以内核权威口径新 golden 后 1.0000 通过（64/64 字节、512/512 元素）；`argmin` fp32 轮整型归一化 2048/2048 全等；其余判为「已证明例外（平台/用例不可验证）」并给出替代判据。**「仍需处置 = 0」；候选实现缺陷 = 0；阈值/容差放宽提案 = 0；`precision_standard.py`（`5170a187`）一字未改**。

### 5.4 证据索引

以下为本团队内部证据归档路径（随交付保存，评审可按需索取）；表中 #1–#10、#12–#21 相对交付树根，#11 亦相对交付树根下的提交构建树 `work/submit-decouple/build/`。

> 下表只列**已落盘、可核验**证据；字节与 sha256 均为撰写时对交付树下文件现算（sha256 取前 16 位，完整值用 `sha256sum <path>` 现算）。

| # | 证据项 | 路径（相对交付树根） | 字节 | sha256（前 16） | 说明 |
|---:|---|---|---:|---|---|
| 1 | UT 运行 XML | `evidence/t71-full-rerun-20261016T105109/full-rerun/ut/run_onnx_plugin.xml` | 152943 | `eb5df342a699625a` | `tests=648`、`failures=0` |
| 2 | UT 汇总 | `evidence/t71-full-rerun-20261016T105109/full-rerun/ut/ut_summary.json` | 348 | `1bfa5472b031331b` | list 行数/套件数/通过数/静态宏口径 |
| 3 | `--gtest_list_tests` 输出 | `evidence/t71-full-rerun-20261016T105109/full-rerun/ut/list_tests.txt` | 109230 | `70e129fe8053b735` | 3276 行 / 405 suite |
| 4 | 逐文件 UT 对账 | `evidence/t71-full-rerun-20261016T105109/full-rerun/ut/ut_per_file_diff.tsv` | 6708 | `63cf01a0b78c9c12` | 57 行：Σ静态宏 624 / Σrun_tests 612 |
| 5 | om 两轮逐例汇总 | `evidence/t71-full-rerun-20261016T105109/full-rerun/T71_SUMMARY.json` | 50055 | `1b0af4d120f6e3f2` | 71 例 default/fp32 主判据+辅助 |
| 6 | om 两轮逐例表 | `evidence/t71-full-rerun-20261016T105109/full-rerun/T71_CASES.csv` | 10520 | `11e506c612b18d85` | 逐例 md5/判定/双口径 |
| 7 | 包内脚本交叉 | `evidence/t74-two-round-full-20261016T105205/two-round-full/CASE_TABLE.csv` | 17060 | `cd070b65c1a157cf` | 61 例逐例结论 |
| 8 | 包内脚本交叉统计 | `evidence/t74-two-round-full-20261016T105205/two-round-full/STATS.json` | 236 | `7a1dc806d1b9f472` | 分类/判定/交叉计数 |
| 9 | 基线插件 | `baseline/pre/liboponnx_plugin_math.so` | 2492576 | `ed2430c4af72b270` | pre 轮 `.so` |
| 10 | 候选原样插件 | `candidate/pr6355/liboponnx_plugin_math.so` | 2361504 | `787e84bc9e989c72` | 修复前（F-C2 现场） |
| 11 | **提交版插件** | `work/submit-decouple/build/liboponnx_plugin_math.so`（另有同哈希构建副本） | 2361504 | `044a1d6c5b79db82` | post 轮 `.so`（F-C2 修复后） |
| 12 | 两轮一键脚本 | `task-package/materials/task/test_cases/verify_om/two_round_compare.sh` | 13710 | `babbed0db6ea4da0` | 包内权威脚本，原样使用 |
| 13 | 混合容差实现 | `task-package/materials/task/test_cases/verify_om/precision_standard.py` | 9619 | `5170a1874c68c27e` | 辅助判据实现，未修改 |
| 14 | 功能性正对照 | `evidence/t4-addcmul-4round-20261015T141453/pilot/compare/functional_control_addcmul-probe.txt` | 750 | `dbbdbe811555ca7a` | addcmul 探针功能性证明 |
| 15 | 功能性正对照探针 `.so` | `scratch/t28-addcmul-probe/build/liboponnx_plugin_math.so` | 2492576 | `d8190a80f3946475` | 探针插件（仅修注册键） |
| 16 | 执行台账 | `docs/ledger/57文件执行台账.csv` | 69473 | `7888720f7bfc5296` | 57 行/124 条目/624 静态宏 |
| 17 | UT 全量验证报告 | `docs/FULL_VALIDATION_RERUN.md` | 19944 | `eb564c91980181a2` | UT 与 om 两轮结论索引 |
| 18 | 包内脚本全量报告 | `docs/TWO_ROUND_PACKAGE_TOOL_FULL.md` | 34129 | `642e85a75d5e878b` | 包内脚本结论与归因 |
| 19 | om 两轮报告 | `docs/OM_TWO_ROUND.md` / `docs/OM_TWO_ROUND_VARIANTS.md` | 23475 / 27953 | `257ea064cb7322db` / `8d4739dc7b36b7f3` | 61 例 / 10 变体（含 F-C2 原始证据） |
| 20 | F-C2 修复说明 | `docs/FIX_FC2.md` | 30369 | `7628022f744cd273` | 修复前/后代码与回归 |
| 21 | 辅助判据修订集 | `docs/AUX_CASE_REVISION_EXEC.md` | 32605 | `fc7780e73d66e805` | 残余 11 例终态（v1.3） |

---

### 5.5 已知缺陷与处置

**终态（一句话）**：候选侧唯一功能性缺陷为 **F-C2**；**我方提交版本** = 上游候选 + F-C2 修复 + 任务包指定落位的 framework UT 集（含 4 类已证明的必要修正），一遍通过全部验证；**上游候选原样** = `NOT RELEASED` / `FIXED_PENDING_UPSTREAM`（修复已提交、尚未被上游接受）。

| 项 | 上游候选原样（`c9eda5d2`，`.so` `787e84bc`） | 我方提交版本（候选 + F-C2 修复，`.so` `044a1d6c`） |
|---|---|---|
| `clip_v11_max_only`（opset≥11，输入 `["X","","MAX"]`） | `rc=255`、无 om：`E19999 Assert ((src_out_anchor) != nullptr) failed[ReLinkInputDataEdge][graph_utils.cc:115]`；正对照（内置插件）`rc=0` 且与 pre 逐位相同 ⇒ 只由候选引起 | `rc=0`、产 om；默认口径两轮 md5 逐位相同；golden `min(x,max)` matched_ratio=**1.000000**；fp32 双口径两轮全 PASS |
| `clip_v11_min_only` / `clip_v11_both` / `clip_default` | 逐位一致 | 逐位一致（**无回归**） |
| 证据指针 | F-C2 原始报错与分支反证见 `OM_TWO_ROUND_VARIANTS.md`；修复说明 `FIX_FC2.md` | 回归证据见 5.4 #5/#6；全量 71 例 FAIL=0 与 UT 648/648 见 5.1/5.2 |

其余缺陷：F-C1（候选自带 UT 编译缺陷，真实 2 处未限定名 + 两目标重复编译放大；我方不使用候选自带 UT，**不进入提交**）、F-C3（`OriginOpType` 裸键，上游既有缺陷，`addcmul` 1 处 + `similar_opp` 63 算子与基线逐字相同，迁移侧不改）、F-C4a/b（任务包侧 UT 契约缺陷 CASE-09/CASE-02：Mean/Sum 的 `N` 类型与 Clip 默认值，我方 UT 集含必要修正且不改期望值）。**F-C5 重复 include、F-C6 行尾风格属非功能性，清理不在提交序列内。**

## 6. 提交与 PR 信息

> 本章把「设计文档 / 代码 / 自验证材料」在社区平台上的对应关系集中列出，便于评审相互定位；技术设计见第 1–4 章，实测证据见第 5 章。

### 6.1 代码提交（cann/ops-math）

| 项 | 内容 |
|---|---|
| PR 链接 | <https://gitcode.com/cann/ops-math/pull/6545> |
| 源分支 → 目标分支 | `LCP7742/ops-math:submit/ops-math-onnx-decouple` → `cann/ops-math:master` |
| PR 标题 | `refactor(onnx): onnx 插件 protobuf 解耦——57 个插件改用 ParseParamsByOperatorFn 并补充 framework UT` |
| 基线 commit | `11aa1c753cc9deac8597a495b6a1b359d412c359`（2026-10-14） |
| 当前 HEAD | `04c1668b4f93a1019b348d46da30588acb99cf2e`（2026-10-27 第三次同步后） |
| 功能提交 tree | `ae73a005eaec52e0dd3d02afdccbdc2f4d9fffef` |

提交序列（4 个功能提交 + 3 个上游同步合并提交，共 7 个提交）：

| # | commit | 类型 | 说明 |
|---|---|---|---|
| 1 | `c9eda5d21` | refactor | 57 个 onnx 插件迁移至 `ParseParamsByOperatorFn` 并补 framework UT（上游候选实现，作者与日期原样保留） |
| 2 | `371436d5d` | style | 恢复 `sum_onnx_plugin.cpp` 为仓内 CRLF 约定，避免整文件行尾 diff |
| 3 | `ff4accd5c` | fix | Clip opset≥11 逐端口空名判据（F-C2，见 5.5） |
| 4 | `583dc4b1e` | test | 采用任务包提供并指定落位的 framework UT 集 |
| 5 | `aa61dfb07` | merge | 同步 upstream/master（137 个提交）：`addcmul` 冲突取本迁移实现（上游为纯格式调整），`argmax` / `argmin` 采纳上游 include 与 `NpuArch::DAV_9201` |
| 6 | `e9146d0b5` | merge | 再次同步 upstream/master（新增 51 个提交，合并点 `d1611333f`）：与迁移文件交集 0，无冲突；合并后仍为 178 文件 |
| 7 | `04c1668b4` | merge | 第三次同步 upstream/master（合并点 `c9b82b691`）：上游 `AccumulateNV2` P1 整改重排了 `math/accumulate_nv2/framework/sum_onnx_plugin.cpp` 的注册块，与本迁移同区域冲突；解决为保留 `ParseParamsByOperatorFn` 与去 `NodeProto` 依赖，并采用上游缩进风格，冲突文件 1 个；合并后仍为 178 文件（58 改 / 120 增，+17,164 / −1,459） |

### 6.2 变更规模

| 项 | 数值 |
|---|---|
| 改动文件 | **178**（修改 58 / 新增 120） |
| 插件文件 | 57（`*/framework/*_onnx_plugin.cpp`） |
| 共享工具头 | 1（`common/inc/framework/math_onnx_plugin_util.h`） |
| framework UT | 57（`test_*_onnx_plugin.cpp`） |
| CMakeLists | 63（含 `conversion/reshape/tests/CMakeLists.txt`） |
| 行数变化 | +17,173 / −1,473（相对合并后的上游 master） |

### 6.3 平台审批口径

代码 PR 按目录拆分为 **29 个审批模块**，每模块需 **2 个 lgtm + 1 个 approve**：

- `conversion`（7）：`concat`、`flatten`、`reshape`、`slice`、`squeeze`、`transpose`、`unsqueeze`
- `experimental`（1）
- `math`（20）：`accumulate_nv2`、`addcmul`、`arg_max_v2`、`arg_min`、`bitwise_xor`、`cast`、`cumsum`、`is_inf`、`maximum`、`minimum`、`mod`、`one_hot`、`reduce_log_sum_exp`、`reduce_max`、`reduce_mean`、`reduce_min`、`reduce_prod`、`right_shift`、`sign_bits_pack`、`sign_bits_unpack`
- `repo-cann/ops-math`（1，仓库级）

模块与 2.3 的 57 文件清单一一对应（`common/` 下 14 个文件归仓库级模块）。CI 在 PR 评论 `compile` 触发：编译 + 静态检查 + UT + 冒烟；本地等价验证见第 5 章。

### 6.4 交付材料对应

| 交付件 | 位置 | 说明 |
|---|---|---|
| 设计文档 | 本 PR（cann/cann-ops-competitions #2299） | `04_tasks/01_community-task-2026/tasklist/10-14-ops-math-onnx/LCP7742/docs/design.md` |
| 代码 | cann/ops-math PR #6545 | 178 个文件，见 6.1 / 6.2 |
| 自验证材料 | 任务平台「提交验收文件」 | 1 自验证步骤说明、2.1 / 2.2 精度、3.1 / 3.2 性能（不涉及）、4.1 / 4.2 内存（不涉及）、5 单算子迁移自查表 |

### 6.5 当前状态（2026-10-23）

| 项 | 状态 |
|---|---|
| 代码 PR #6545 | 已提交；CLA 通过；与 upstream/master 合并试算无冲突；待 29 个模块的 lgtm / approve 与 CI |
| 设计文档 PR #2299 | 已提交；CLA 通过；待 `repo-cann/cann-ops-competitions` 模块 2 个 lgtm + 1 个 approve |
| 后续维护 | 上游仍在并行迁移同类算子；若再次改动这 57 个文件，按「保留本迁移实现 + 吸收上游非格式性改动」的原则再合并并重跑验证 |

---

## 附录

### 附录 A 修订记录

| 日期 | 版本 | 修订说明 | 作者 |
|---|---|---|---|
| 2026-10-19 | v1.0.0 | 初稿：完成 57 文件/124 条目迁移设计、attribute JSON 契约、分组特殊语义清单、Cast 判据修正、UT/om 两轮验证与兼容性分析 | LCP7742 |
| 2026-10-23 | v1.1.0 | 新增第 6 章「提交与 PR 信息」：代码 PR 链接与提交序列、178 文件口径、29 模块审批口径、交付材料对应与当前状态 | LCP7742 |

### 附录 B 参考材料

- 任务书：《ops-math 算子 onnx 插件 protobuf 解耦任务书》（`ops-math_onnx_task_doc.md`）
- 目标仓库：<https://gitcode.com/cann/ops-math>（基线 commit `91c6c1715`）
- 关联 issue：<https://gitcode.com/cann/ops-math/issues/2892>
- 参考实现（仓内 @`91c6c1715` 已合入的 9 个 `ParseParamsByOperatorFn` 文件，见 1.2.4）
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
- CANN ONNX 算子开发说明：`ai_framework_adaptation/onnx_framework/call_examples.md`
- 社区贡献规范：`cann/cann-ops-competitions` 仓 `CONTRIBUTING.md`
- nlohmann/json 文档：<https://json.nlohmann.me/>
