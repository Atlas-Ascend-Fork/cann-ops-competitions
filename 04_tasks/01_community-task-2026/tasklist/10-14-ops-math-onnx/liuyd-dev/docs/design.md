# 10月社区任务-ops-math算子onnx适配 设计文档

| 项 | 内容 |
| --- | --- |
| 任务 | 10月社区任务-ops-math算子onnx适配（任务编号 14） |
| 目标仓 | [cann/ops-math](https://gitcode.com/cann/ops-math) |
| 改造范围 | 57 个 onnx 插件文件 / 124 个算子条目（任务书 §1 范围表） |
| 基线 commit | `8ade3796f8fb22de7679569337fcbe6618b2ce32`（master，2026-10-17 取） |
| 验证环境 | Ascend910B3（Atlas A2）/ aarch64 / CANN 9.1.0 / 驱动 npu-smi 26.1.1 |
| 验证依据 | 2026-10-17/18 于 Ascend910B3 实测，全部数据取自基线 `8ade3796` 这一轮：插件未定义 `ge::onnx` 动态符号 5 → 0；framework UT 常规轮 648/648、experimental 轮 188/188 通过；om 两轮比对 61 个用例全部一致（46 逐位一致 + 3 生态混合容差 + 5 om 图结构比对 + 7 个两轮同样转换失败）。逐条见 §5.3 |

---

# 1 需求背景（required）

## 1.1 需求来源

10月社区任务-ops-math算子onnx适配（任务编号 14）。ops-math 仓的 onnx 插件通过
`REGISTER_CUSTOM_OP(...).ParseParamsFn(...)` 注册参数解析函数，其签名为

```cpp
domi::Status ParseParamsXxx(const google::protobuf::Message* op_src, ge::Operator& op_dest);
```

入参是 protobuf 的 `ge::onnx::NodeProto`，插件因此必须 include `onnx/proto/ge_onnx.pb.h`
并链接 protobuf 运行库。GE 已提供不依赖 protobuf 的新接口 `ParseParamsByOperatorFn`，
入参改为 `const ge::Operator&`，onnx 节点属性以 JSON 字符串挂在 `op_src` 的
`attribute` 属性上。任务书要求把仓内剩余的 57 个 onnx 插件文件（124 个算子条目）
迁移到新接口，解除插件对 protobuf 的编译期与运行期依赖，并为每个插件文件补齐 framework 层 UT。

## 1.2 背景介绍

**接口定义**（`register/register.h:92-93`，任务书 §2.3）：

```cpp
using ParseParamFunc     = std::function<domi::Status(const google::protobuf::Message*, ge::Operator&)>;
using ParseParamByOpFunc = std::function<domi::Status(const ge::Operator&, ge::Operator&)>;
```

**解耦的收益是可量化的**。迁移前后对同一基线编译出的 `liboponnx_plugin_math.so`
做动态符号比对（`nm -D`，Ascend910B3 / CANN 9.1.0 / 基线 `8ade3796`）：

| 项 | 迁移前 | 迁移后 |
| --- | ---: | ---: |
| 未定义动态符号总数 | 255 | 239 |
| 其中 `ge::onnx::*` | 5 | **0** |
| 其中 `ascend_private::protobuf::*` | 7 | **0** |
| 其中 `absl::lts_ascend_private::*`（protobuf 的 CHECK 宏带入） | 5 | **0** |
| `domi::OpRegistrationData::ParseParamsFn(...)`（旧注册 API） | 1 | **0** |
| `__dynamic_cast@CXXABI_1.3` | 1 | **0** |
| `.so` 体积 | 2 488 200 B | 2 369 416 B |

（前五行按符号名所属类别分别计数，**类别之间有交叉，不可相加**。）按集合差看，
消失 20 个、新增 4 个（`ge::Operator::GetAttr(char const*, std::vector<ge::AscendString>&)`、
`ge::Operator::GetOutputsSize()`、`strtof`、`std::__throw_invalid_argument`），
这 4 个与本设计 §3.2.3 / §3.2.4 的助手实现一一对应，没有计划外的新依赖。

**仓内已有样板**。ops-math 仓已有 9 个 onnx 插件完成迁移，可直接作为写法基准：
`one_hot_d`、`reverse_sequence`、`correlation`、`npu_indexing`、`shape`、`bernoulli`、
`npu_rotary_mul`、`npu_format_cast`、`npu_dtype`。本设计的所有写法与这 9 个文件保持一致，
不引入新风格（任务书 §6 参考资料 1 给出的 issue #2892 及其关联 PR 即这批样板）。

## 1.3 迁移对象核验（任务书 §1 范围表）

任务书 §1 的范围表基于 commit `91c6c17` 静态扫描得到，并声明"如与仓内实际不符请提交 issue"。
本设计在基线 `8ade3796` 上逐条脚本核验（非人工抽样）：

| 核验项 | 任务书 | 基线实测 | 结论 |
| --- | ---: | ---: | --- |
| 带 `.ParseParamsFn(` 的非测试 `*_onnx_plugin.cpp` 文件数 | 57 | 57 | 一致 |
| 算子条目数 | 124 | 124 | 一致 |
| `common/src/framework/similar_opp_onnx_plugin.cpp` 内算子数 / 注册条数 | 63 / 578 | 63 / 578 | 一致 |
| OriginOpType 字面量 | 逐条 | 逐条 | 一致 |
| 用例包引用的 57 个解析函数名 | — | 全部真实存在 | 一致 |

两处名义差异，均为基线既有事实、不构成范围偏差，也无需提 issue：

1. `addcmul` 的 OriginOpType 是 `torch.onnx.symbolic_opset11.addcmul`，不是 `ai.onnx::N::` 形式；
2. 57 个解析函数名中存在不规则拼写（`ParseParamConcatFromSequence` 无 s、
   `parse_params_pad_v9` 下划线、`parseParamsRepeatWithPad` 小驼峰）。
   迁移**逐字保留原函数名**，不顺手改名——改名会使下发用例包失效，且超出任务书 §2.1 的要求。

另核对到任务书 §3.5 正文写"57 个 UT 用例文件（612 个用例）"，而用例包实际为 **624** 个
（逐文件 `grep -c '^TEST('` 核计，常规轮 615 + experimental 轮 9）。以用例包为准。

---

# 2 需求分析（required）

## 2.1 需求描述

在不改变任何算子可观测行为的前提下，把 57 个 onnx 插件文件的参数解析入口从
`ParseParamsFn(const google::protobuf::Message*, ge::Operator&)` 迁移到
`ParseParamsByOperatorFn(const ge::Operator&, ge::Operator&)`，使插件产物不再引用
protobuf 与 `ge::onnx` 符号；同时为每个插件文件补齐 framework 层 UT，并以
"迁移前 / 迁移后"两轮 om 转换与推理比对证明等价性。

本任务是 **Host 侧等价重构**：不新增算子、不改算子数学语义、不涉及 kernel 与 tiling。

## 2.2 需求拆解（任务书 §2.1 逐条落地）

| # | 任务书 §2.1 要求 | 本设计的落地方式 | 详见 |
| ---: | --- | --- | --- |
| 1 | 解析函数签名替换 | `(const Message*, ge::Operator&)` → `(const ge::Operator&, ge::Operator&)`，**函数名逐字不变** | §3.2.2 |
| 2 | 注册方式替换 | `.ParseParamsFn(f)` → `.ParseParamsByOperatorFn(f)`，注册链其余项（`FrameworkType` / `OriginOpType` / `ParseOpToGraphFn` / `ImplyType`）一字不改 | §3.2.2 |
| 3 | 属性解析方式替换 | 统一 `op_src.GetAttr("attribute", ge::AscendString&)` + nlohmann/json | §3.1.2、§3.2.3 |
| 4 | 移除 protobuf 依赖 | include 去掉 `onnx_common.h`，按文件实际需要改为 `register/register.h` 等 | §3.2.8 |
| 5 | 算子名获取方式替换 | `node->name()` → `GetOpName(op_src)`；带拆图的文件用 `GetOnnxNodeName(op_src, op_dest)` 兜底 | §3.2.5 |
| 6 | 公共工具函数 | 复用并扩充仓内既有 `common/inc/framework/math_onnx_plugin_util.h`，**不新增公共头文件** | §3.2.3 |
| 7 | 语义保持 + 失效判据修正 | 逐文件语义保持矩阵见 §3.2.6；Cast 的失效判据修正是唯一例外 | §3.2.6 |

## 2.3 算子工程模式（任务书 §2.2）

| 任务书 §2.2 条目 | 本设计 |
| --- | --- |
| 2.2.1 插件经 `REGISTER_CUSTOM_OP` 注册到 GE 框架（`FrameworkType(ONNX)`），ATC 解析期由框架回调参数解析函数 | 注册链只改 `.ParseParamsFn` → `.ParseParamsByOperatorFn`，`FrameworkType` / `OriginOpType` / `ImplyType` 一字不改；插件文件原地改写，不移动、不重命名、不拆分 |
| 2.2.2 只涉及各算子目录下 `framework/*_onnx_plugin.cpp` 及对应 framework UT，不涉及 kernel、tiling、算子原型 | 57 个 UT 中，43 个放在对应的 41 个算子目录下，`common/src` 下的 14 个插件对应放 `common/tests/ut/framework/`；`op_host/` / `op_kernel/` / `op_graph/` / `examples/` / `external/` 零改动（§2.4） |
| 2.2.3 不改 `ParseOpToGraphFn` | 50 个带拆图的注册项逐条核对，拆图函数体**一行未改**（§5.3.1 给出计数证据） |

## 2.4 改造边界

**改**：57 个 `*_onnx_plugin.cpp` 的解析函数体与注册语句；`math_onnx_plugin_util.h`
增加公共读取助手；新增 57 个 framework UT；为 5 个走 `add_subdirectory` 链的算子目录
补叶子 `CMakeLists.txt`；`cmake/ut.cmake` 补一条 include 路径；
`tests/ut/op_host/CMakeLists.txt` 补 `runtime` 链接（§3.2.9）。

**不改**：`ParseOpToGraphFn` 拆图函数体；算子原型 `op_graph/*_proto.h`；`op_host/`、
`op_kernel/`；`external/`；`examples/`；`onnx_common.h` 本身（仍为 TF 插件与未迁移场景服务）。

---

# 3 详细设计（required）

## 3.1 对象分析

### 3.1.1 ATC 侧处理链路

```
onnx 模型 ── ATC(--framework=5) ──> onnx 模型解析器
                                      │  ① 按 "ai.onnx::<opset>::<Op>" 查注册表
                                      │  ② 构造 op_src(ge::Operator)：
                                      │      attribute(VT_STRING, 属性 JSON)
                                      │      input / output(VT_LIST_STRING, 节点输入/输出名单)
                                      │      name / op_type(VT_STRING)
                                      ├─> ParseParamsByOperatorFn(op_src, op_dest)   ← 本次迁移对象
                                      ├─> (若注册为 PartitionedCall) ParseOpToGraphFn(op, graph)
                                      └─> GE 图编译 ──> om
```

### 3.1.2 `op_src` 的真实形态（任务书 §2.4 的真机核对）

任务书 §2.4 给出了 attribute JSON 的字段说明。本设计在 Ascend910B3 + CANN 9.1.0 上，
用一个注册为 `PartitionedCall` 的探针插件在 `ParseParamsByOperatorFn` 内枚举 `op_src`
（`GetAllAttrNamesAndTypes`）实测核对，结果与任务书一致，并补出任务书未写明的两项：

| 项 | 实测值 |
| --- | --- |
| `op_src` 属性清单 | 恰好 5 个：`attribute`(VT_STRING)、`input`(VT_LIST_STRING)、`output`(VT_LIST_STRING)、`name`(VT_STRING)、`op_type`(VT_STRING) |
| `attribute` 内容 | `{"attribute":[{...}]}`，pretty-print 多行，键按字母序 |
| 标量 FLOAT 的承载 | `"f"` 是**字符串**（`%g` 格式化）；`"i"` / `"ints"` / `"floats"` 是 JSON 数字 |
| `op_type` | `ai.onnx::13::Concat`（带 opset 的完整类型） |
| `GetOpName(op_src)` | onnx 节点名，等价于迁移前 `node->name()` |
| **`op_src.GetInputsSize()`** | **0**（任务书与用例包文件头未提及，见 §3.2.4） |
| **`op_src.GetOutputsSize()`** | **0** |
| `input` / `output` 列表 | 节点的输入/输出名单，长度即真实个数 |

### 3.1.3 属性缺省与零值省略

GE 的 pb2json 桥接对 singular 字段先判 `HasField`、对 repeated 先判 `FieldSize > 0`，
未设置的字段整个不出现。由此得到三条判定规则，它们是全部 57 个文件的共同契约：

1. **属性项存在但缺值字段** ⇒ 取 proto3 零值，**不得回落到算子默认值**：
   缺 `i` ⇒ 0；缺 `f` ⇒ 0.0；缺 `ints` ⇒ 空列表；缺 `s` ⇒ 空串。
2. **属性项整个不存在** / `attribute` 数组为空 / `attribute` 键缺失 / `attribute` 值不是数组
   ⇒ 按"节点无该属性"处理，走算子自身默认值。
3. **`type` 与声明不符**（如 `axis` 标成 FLOAT）⇒ 按"该属性不存在"处理，与迁移前
   `attr.type() == AttributeProto::INT` 的判据等价。**迁移前不校验 type 的属性，迁移后同样不校验**
   （逐文件对齐，不统一加严）。

### 3.1.4 迁移形态分三档

| 档 | 判据 | 文件数 | 条目数 | 迁移手法 |
| --- | --- | ---: | ---: | --- |
| **T1** | 不读属性、不读输入个数 | 2 | 64 | 函数体/宏体只剩 `return SUCCESS` |
| **T2** | 只读属性 | 31 | 36 | `VisitOnnxAttrs` + 属性读取助手 |
| **T3** | 依赖 onnx 节点输入/输出个数 | 24 | 24 | 再加 `GetOnnxInputCount/GetOnnxOutputCount`，见 §3.2.4 |

T1 中 `similar_opp_onnx_plugin.cpp` 是全任务性价比最高处：只动 include 与两个宏
（`REGISTER_PARSE_PARAMS_OP` 的宏体由 10 行塌成 1 行、`REGISTER_CUSTOM_ONNX_OP` 里
改 1 行注册调用，合计 +7/−12 行），就迁完 63 个算子 / 578 条注册（占 124 条目的一半）。其用例契约要求**连 JSON 都不解析**
——非法 JSON 也必须 SUCCESS 且不写任何属性，否则属于改变可观测语义。

## 3.2 实现方案

### 3.2.1 Host / Kernel 侧的划分

本任务只有 Host 侧改造，**不涉及 kernel 侧设计、tiling 与分核策略**：onnx 插件在 ATC 的
模型解析阶段运行于 Host，产物 `liboponnx_plugin_math.so` 不包含任何 device 代码；
算子的 kernel、tiling、原型均不在本次改动范围（§2.4）。

### 3.2.2 Host 侧设计：注册与函数签名改写

```cpp
// 迁移前
static Status ParseParamsFlatten(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) { OP_LOGE(...); return FAILED; }
    int axis = 1;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) { axis = attr.i(); }
    }
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}
REGISTER_CUSTOM_OP("Flatten").FrameworkType(ONNX).OriginOpType({...})
    .ParseParamsFn(ParseParamsFlatten).ImplyType(ImplyType::TVM);

// 迁移后
static Status ParseParamsFlatten(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int axis = 1;
    const Status parse_status = VisitOnnxAttrs(op_src, op_dest, [&](const json& attr) {
        if (IsOnnxAttr(attr, "axis", ONNX_ATTR_INT)) { axis = static_cast<int>(GetOnnxAttrInt(attr)); }
    });
    if (parse_status != SUCCESS) { return FAILED; }
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}
REGISTER_CUSTOM_OP("Flatten").FrameworkType(ONNX).OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParamsFlatten).ImplyType(ImplyType::TVM);
```

`dynamic_cast` 判空分支随 protobuf 入参消失（新接口入参是引用，不可能为空），
这一分支在迁移后**不可达**，用例包亦将其列为"不覆盖范围"。

### 3.2.3 公共读取助手（扩充既有公共头，不新增公共头）

全部助手以 `inline` 函数加入仓内既有的 `common/inc/framework/math_onnx_plugin_util.h`
（任务书 §2.1.6）。该头**不得 include `onnx_common.h`**，否则会把 `ge_onnx.pb.h` 再拉回来。

| 助手 | 作用 |
| --- | --- |
| `VisitOnnxAttrs(op_src, op_dest, visitor)` | 取 `attribute` 字符串 → `json::parse` → 遍历 `attribute` 数组逐项回调；**try/catch 收在助手内**，JSON 非法或属性值类型不符即记日志并返回 `FAILED` |
| `CheckOnnxAttrs(op_src, op_dest)` | 只校验 JSON 合法性、不读任何属性（给"无属性但用例要求非法 JSON → FAILED"的算子） |
| `IsOnnxAttr(attr, name[, type])` | 按名（可选按 type）匹配；对非对象数组元素免疫 |
| `GetOnnxAttrInt / GetOnnxAttrInts / GetOnnxAttrString / GetOnnxAttrFloat` | 读 `i` / `ints` / `s` / `f`，缺键按 §3.1.3 取零值；类型不符时抛异常，由 `VisitOnnxAttrs` 统一转 `FAILED` |
| `GetOnnxInputCount / GetOnnxOutputCount / IsOnnxInputOmitted` | 见 §3.2.4 |
| `GetOnnxNodeName(op_src, op_dest)` | 见 §3.2.5 |
| `OnnxAttrType` 枚举 | `ONNX_ATTR_FLOAT=1 / INT=2 / STRING=3 / INTS=7 …`，替代 `ge::onnx::AttributeProto::XXX` |

两点设计取舍：

- **把 try/catch 收进 `VisitOnnxAttrs`**：57 个文件各写一份 8 行 try/catch 会形成大面积
  重复代码块（codecheck 对 ≥10 行重复块告警），visitor 方案让每个文件的样板只剩 3 行。
- **只实现 INT / INTS / FLOAT / STRING 四种形态**：实测 57 个文件中 `attr.floats()`
  与 `attr.t()` 一次都没用到，实现它们会留下死代码。

### 3.2.4 【关键】节点输入/输出个数：必须读属性，不能用 `GetInputsSize()`

24 个 T3 文件依赖 `node->input_size()` / `node->output_size()`。任务书 §2.1 与下发用例包
文件头都写"迁移后取 `op_src.GetInputsSize()`"，但 §3.1.2 的实测表明该接口在 ParseParams
阶段**恒为 0**。

**取错的后果是静默的**（实测，非推演）。以 3 输入 Concat 为例，用 `GetInputsSize()` 时：

```
GE: After ParseParams, op[concat_node]: type[PartitionedCall] have input size: 0, output size: 1
GE: operator_impl.cc:68 SetInputImpl: [Check][Param] dst name is empty   × 3
→ ParseOpToGraphFn 根本不被调用
→ atc 返回码 0，产出 6811 字节 om（正常产物 32614 字节）
```

即 **ATC"转换成功"不构成任何证据**，错误只埋在 plog 里。改为读 `input` 属性名单后：

```
ParseOpToGraphConcat: N=3 op.GetInputsSize()=3
GE: After ParseParams, op[concat_node]: have input size: 3, output size: 1
GE: node_utils_ex.cc:128 Verify: concat_node(ConcatD) input size 3, Input names:[0,x0][1,x1][2,x2]
→ om 32614 字节，pyACL 推理 + golden 比对 matched_ratio=1.000000 PASS
```

仓内还有一个旁证：PR !5311 正是按 `GetInputsSize()` 写的，2026-09-11 合入、
09-14 被 !5642 原样回退。该问题只有真机 om 轮能暴露，CI 不覆盖 om 转换与精度比对。

**本设计采用的口径**：

| 迁移前 | 迁移后 |
| --- | --- |
| `node->input_size()` | `GetOnnxInputCount(op_src)` = `GetAttr("input", std::vector<ge::AscendString>&)` 的 `.size()`，取不到属性才回退 `GetInputsSize()` |
| `node->output_size()` | `GetOnnxOutputCount(op_src)`，同上取 `"output"` |
| `node->input(i) == ""` | `IsOnnxInputOmitted(op_src, i)`：名单第 i 项为空串（`GetString()` 可能是 `nullptr`，故判 `s == nullptr \|\| s[0] == '\0'`） |
| `node->name()` | `GetOpName(op_src)` |

"取不到属性才回退描述符个数"这一层兜底不是防御性冗余，而是**让同一份代码同时满足两套契约**：
真机 ATC 下走属性名单（真实语义）；UT 下用例以 `op_src.DynamicInputRegister("x", n)`
构造输入个数、不设 `input` 属性，走描述符个数——于是 624 个下发用例一条都不用为此改动。
`ge::Operator` 的外部头只有 `std::vector<ge::AscendString>` 一个列表字符串 `GetAttr` 重载
（`graph/operator.h:343`），不存在 `std::vector<std::string>` 版本。

### 3.2.5 属性转写与工程行为

**类型逐字保持**。`op_dest.SetAttr` 的键名、值类型（`int` / `uint32_t` / `bool` / `float` /
`std::vector<int64_t>` / `ge::Tensor`）、`Vec2Tensor` 的 dtype 与 format 全部与迁移前一致。
例：同为 Reduce 系列，`reduce_sum` 的 `axes` 张量是 `Vec2Tensor(v_axes, dims, DT_INT32)`
（FORMAT_ND），而 `reduce_max/min/prod/mean` 是 FORMAT_NCHW；`reduce_sum` 的动态输出名是
`"output"`，`reduce_max` 是 `"y"`——这些差异一律原样保留，不统一。

**动态输入/输出注册**。常量注册（如 Xor 的 `DynamicInputRegister("x", 2)`）保持常量，
**不改成读输入个数**；变长注册（如 Max 的 `DynamicInputRegister("x", n)`）按 §3.2.4 取 n。

**名称兜底**。写 `op_dest` 的 `"name"` 属性时：不带 `ParseOpToGraphFn` 的文件用
`GetOpName(op_src)`（与任务书 §2.1.5 字面一致，源名为空就写空串，与迁移前
`node->name()` 为空串等价）；带 `ParseOpToGraphFn` 的文件用 `GetOnnxNodeName(op_src, op_dest)`，
源名为空时回退 `op_dest` 名，避免拆图阶段用空名拼子图节点名。后者是仓内已迁移插件
`correlation_onnx_plugin.cpp` 的既有写法，且用例包对 Max / Min / Xor / experimental Max
四个文件明确断言这一兜底。

**失败副作用位置保持原样**。迁移前在校验前已写出的属性（ReduceSum13 的 `original_type`、
ReduceSum 的动态输入注册、Clip V11 的 `original_type` 与注册）迁移后同样先写；
迁移前"失败即什么都不写"的插件（Xor、ReduceMax 的 opset8~12 分支）迁移后也保持
"解析失败先于任何写出"。

### 3.2.6 重点算子设计

**（1）Cast：任务书 §2.1.7 指定的失效判据修正——等价重构的唯一例外**

```cpp
// 迁移前：GetOmDtypeFromOnnxDtype() 对未命中 dtype 返回 ge::DT_UNDEFINED，不是 -1
//         ⇒ 该分支永不成立，不支持的 dtype（如 to=20）要到引擎选择阶段才失败
if (om == -1) { return FAILED; }
// 迁移后
if (om == static_cast<int32_t>(ge::DT_UNDEFINED)) { OP_LOGE(...); return FAILED; }
```

修正后，`to` 属性缺 `i` 键（取 0 ⇒ 未命中映射表）与 `to=20` 两种情况都在解析阶段返回 FAILED，
对应用例 `ReturnFailedWhenToIsZeroWithoutIField` / `ReturnFailedWhenDtypeUnsupported`。
该修正不影响两轮比对判据：非法模型两轮都转换失败，合法模型行为不变。

**（2）Clip V11：可选输入的空名语义完整还原**

迁移前靠 `node->input(1)/input(2)` 是否为空串区分"min/max 未提供"。onnx 的可选输入在名单中
以空名占位，`IsOnnxInputOmitted` 正好还原这一语义；UT 构造的 `op_src` 没有名单，该函数返回
false，退化为"按输入个数判"，与用例 `V11MinOnlyWithTwoInputs` / `V11BothMinMaxWithThreeInputs`
的断言一致。同一份代码在真机上是真实语义、在 UT 下是用例语义，无需分叉。

**（3）"首个命中优先"与"独立 if"等控制流细节**

迁移前用 `break` 或 `std::find_if` 取首个命中的属性（Concat 的 `axis`、OneHot 的 `axis`、
Addcmul 的 `value`、Cast 的 `to`），visitor 不能 `break`，改用 `if (!found && ...)` 守卫，
保证**重复同名属性时仍取首个**（用例对 Addcmul 明确断言 0.5 而非 2.0）。反之，迁移前
后者覆盖前者的（Clip V9 的 max/min、Reduce 系列的 keepdims）保持覆盖语义；迁移前把
`num_outputs` 写成独立 `if`（不在 else-if 链里）的（Split）保持独立。

**（4）带副作用早返回的循环**

ArgMax / ArgMin 的 `select_last_index == 1`、Pad V9 的 `pads` 长度为奇数、BitShift 的
"出现非 direction 属性"三处是在循环体内 `return FAILED`。visitor 内不能直接返回，
改为置标志位、循环后统一判定。这些循环体除局部赋值外无副作用，两种写法可观测行为相同。

**（5）"无属性"插件的 JSON 校验分支逐文件对齐**

这类插件迁移前完全不读属性，迁移后"是否新增非法 JSON → FAILED 分支"**以用例包断言为准**，
不做统一：

| 要求非法 JSON → FAILED | 要求非法 JSON 仍 SUCCESS |
| --- | --- |
| Xor、Max（maximum / experimental）、Min 全版本 | similar_opp 的 63 个算子、sequence 的 6 个算子、SqueezeV3、UnsqueezeV3、Slice(opset10+)、Sum、Mean |

**（6）计数式必填校验**

SignBitsPack（`size`）/ SignBitsUnpack（`size` + `dtype`）迁移前用"命中次数 == 必填个数"
判必填，重复同名属性会使计数超标而判 FAILED。该行为原样保留（属迁移前既有行为，
不在任务书 §2.1.7 的修正范围）。

### 3.2.7 下发用例的 9 处修订与依据

57 个 UT 文件取自任务书配套用例包。其中 48 个在去掉空白后与下发件**逐字符相同**
（差异仅为仓内 pre-commit 的 clang-format 18.1.8 重排版）；9 个文件有实质改动，分两类：

| 类别 | 文件 | 原因 |
| --- | --- | --- |
| **A. 断言与真机行为不符，附探针实测依据** | `deformable_offsets`、`depth_to_space`、`space_to_depth` | 原件用裸 `ge::Operator(name,"TestOp")` 构造 `op_dest`，无输入/输出描述符；被测函数（迁移前后一致）调用 `ChangeFormatFromOnnx` 在裸算子上实测返回 `GRAPH_FAILED`，带 `DynamicInputRegister/DynamicOutputRegister` 时返回 `GRAPH_SUCCESS`。真实 ATC 下 `op_dest` 由算子原型构造、描述符齐备，故改用带描述符的 `CreateDestOperator()`，**其余断言一字未改** |
| | `clip` | 原件把 ClipV9 的 max/min 默认值写成 `std::numeric_limits<float>::max()`（3.40282347e+38），插件源码里是字面量 `3.402823e+38`（3.40282306e+38）。迁移是等价重构，不改该字面量，故按插件实际默认值断言 |
| | `mean`、`sum` | 原件断言"以其它整型宽度读 `N` 必须 `GRAPH_FAILED`"。实测 GE 的 `GetAttr` 在整型间隐式转换、并不报错——该断言测的是 GE 属性接口行为而非插件行为，故改为只断言"以声明类型读到正确取值" |
| **B. 适配仓内编码规范，断言逐条不变** | `similar_opp` | 一条超长 JSON 字面量拆成相邻字符串字面量拼接，满足 120 列行宽（G.FMT.05），内容逐字符相同 |
| | `random_normal`、`random_uniform` | 两处用例共用的同一组断言抽成辅助函数，满足重复代码块规范，断言内容逐条相同 |

每个修订点都在该文件头部以 `【用例修订（日期，附实测依据）】` 注释写明原因与探针输出，
便于评审核对。A 类 6 个文件若不改，UT 会失败在"下发用例自身的断言"而非插件行为上。

### 3.2.8 依赖与文件组织

```cpp
// 迁移前（间接拉入 ge_onnx.pb.h）
#include "onnx_common.h"

// 迁移后（按文件实际需要取）
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"           // 仅 ParseOpToGraphFn 的 ge::Graph 需要
#include "stub_ops.h"              // 仅带 ParseOpToGraphFn（用 op::）的文件需要
#include "math_onnx_plugin_util.h" // GetOpName / Vec2Tensor / CreateScalar / 本次新增助手
#include "log/log.h"
#include "nlohmann/json.hpp"       // 仅直接出现 json 类型的文件需要
```

`onnx_common.h` 保持存在（仍为 TF 插件与未迁移场景提供 `ge_onnx.pb.h`），但本次 57 个文件
不再 include 它。注意 `onnx_common.h` 原本顺带提供 `stub_ops.h`（`REG_OP` 宏），
换头后用到 `op::` 的文件必须显式补这一行。

### 3.2.9 构建与 UT 接线（实测结论）

- onnx 插件目标 `oponnx_plugin_math_obj` 已带 `${JSON_INCLUDE_DIR}` 并链接 `json`
  （`cmake/func.cmake:303-337`，其中 include 目录在 317 行、链接在 333-337 行），
  **插件侧无需改任何 CMake**；该目标显式设 **C++14**（`func.cmake:305-309`），
  故新增代码不使用 C++17 特性。全仓非 Debug/ASAN 构建带 `-Werror`
  （`cmake/dependencies.cmake:54-59`），该目标另加 `-Wno-shadow -Wno-unused-parameter`，
  因此未使用的**局部变量**仍会直接报错，助手与 visitor 中不留死变量。
- framework UT 的收集有**两套机制**，必须分辨后再决定是否加文件：
  - 多数算子目录走 `add_all_modules_sources` → `add_all_ut_sources` 的无条件 GLOB，
    **按算子目录放 UT 文件即可，不需要新增 CMakeLists.txt**；
  - 少数算子目录走 `add_subdirectory` 链，没有叶子 `CMakeLists.txt` 就收不到文件。
    本次有 **5 个**属于这一类，需补叶子 CMakeLists（内容为仓内既有模板的
    `add_modules_ut_sources(...)` 三行）：`conversion/flatten`、`conversion/squeeze`、
    `conversion/unsqueeze`、`math/is_inf`、`random/dsa_random_uniform`。
  - `common/tests/ut/framework/` 目录已存在且已有 CMakeLists，直接放文件。
- **下发用例以 `#include "..._onnx_plugin.cpp"` 把插件源码文本引入用例**，插件的编译
  依赖与链接依赖因此都变成 UT 目标的依赖。这带来两项必须补的构建改动：
  - `tests/ut/op_host/CMakeLists.txt` 补 `runtime` 链接——插件中对 runtime 的调用
    （如 `arg_min` 的 `rtGetSocSpec`）成为 UT 可执行文件的未定义符号；
  - `cmake/ut.cmake` 的 framework UT 目标补 `${OPS_MATH_DIR}/common/inc/op_host`
    这条 include 路径——带拆图的插件会包含算子原型头，而原型头（如
    `conversion/transpose/op_graph/transpose_proto.h`）包含该目录下的公共定义头。
    插件目标本身已有这条路径（`func.cmake:313`），UT 目标原先没有；这条在基线上不暴露，
    因为基线没有把插件源码编进 UT 的用例。
- `experimental/` 子树与常规算子目录**互斥**（根 `CMakeLists.txt:132-138` 是
  `if(ENABLE_EXPERIMENTAL) add_subdirectory(experimental) else() foreach(ops_category) endif()`），
  故 `experimental/math/asinh/framework/max_onnx_plugin.cpp` 及其 UT 需单独跑一轮
  `--experimental` 的 UT。注意 `build.sh --onnxplugin --experimental` 在仓内**不成立**
  （experimental 配置下没有 `oponnx_plugin_math` 目标，实测报
  `No rule to make target 'oponnx_plugin_math'`，基线与本分支一致），
  因此该插件文件的编译证据来自 **experimental 轮的 UT**，而不是独立 `.so`。

以上改动合计 7 个构建文件（5 新增 + 2 修改），都是"不加就收不到用例 / 链不过 / 编不过"的必需项，
符合任务书 §5"不得多出无用的编译文件"。

### 3.2.10 文件清单

| 类别 | 数量 | 位置 |
| --- | ---: | --- |
| 迁移的插件文件 | 57 | `{common/src,<category>/<op>}/framework/*_onnx_plugin.cpp` |
| 扩充的公共头 | 1 | `common/inc/framework/math_onnx_plugin_util.h` |
| 新增 framework UT | 57 | `common/tests/ut/framework/`（14）+ 41 个算子目录的 `tests/ut/framework/`（43） |
| 新增叶子 CMakeLists | 5 | 见 §3.2.9 |
| 修改的构建文件 | 2 | `tests/ut/op_host/CMakeLists.txt`（+3 行，补 `runtime`）、`cmake/ut.cmake`（+4 行，补 include 路径） |
| **合计** | **122** | — |

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 | 本工程的验证状态 |
| --- | --- | --- |
| Atlas A2 系列产品（含 Atlas 800I/T A2） | √ | **已实测**：Ascend910B3，两轮 om 比对与 UT 均在此完成 |
| Ascend 950 系列 | √ | 未实测；任务书 §3.1 允许"任一硬件验证通过即可，推荐 A2" |

本任务为 Host 侧 onnx 模型解析适配，产物 `liboponnx_plugin_math.so` 是 Host 侧动态库，
不含 device 代码，**与 NPU 型号无关**。任务书 §3.1 要求"Atlas A2 与 Ascend 950
任一硬件验证通过即可，推荐 A2"，且要求"om 精度比对的迁移前后基线需在同一硬件上获取"。
本设计的两轮比对全部在同一台 Ascend910B3（Atlas A2）上完成，满足该约束。

## 3.4 约束限制

| 约束项 | 内容 |
| --- | --- |
| 数值语义 | 零改动。迁移是等价重构，唯一例外是任务书 §2.1.7 指定的 Cast 失效判据修正（§3.2.6） |
| 标量 FLOAT 精度 | GE 的 JSON 桥接按 `%g`（6 位有效数字）格式化后以字符串承载，极端取值（格式化文本 ≥10 字符）退化为 JSON 数字。`GetOnnxAttrFloat` 两种形态都接受，但**桥接本身的精度损失不可逆**（cann/ge#656）。这是 GE 侧既有行为，与本次迁移无关；下发用例与 om 模型取的都是 0.25/0.5/2.5 这类短十进制，不触发 |
| 不可达分支 | 迁移前 `dynamic_cast` 判空 → FAILED 的分支在新接口下不可达，不在 UT 覆盖范围 |
| 空节点名 | onnx 节点名为空时，带拆图的插件写 `op_dest` 名而非空串（§3.2.5），这是对拆图阶段的保护，不影响数值语义 |
| 属性形态覆盖 | 助手只实现 INT / INTS / FLOAT / STRING 四种；`floats` / `t`（TENSOR）在本范围 57 个文件中一次都没用到，不实现 |
| 平台容错 | 任务书 §3.2.5：个别算子在部分 SOC 上无法完成 ATC 转换。本轮有 7 个用例两轮同样在转换期失败（§5.3.3），不影响其余算子的验收 |

---

# 4 特性交叉分析

| 特性 | 与本次迁移的交叉 |
| --- | --- |
| 多 opset 版本注册 | 同一文件内多个 `REGISTER_CUSTOM_OP` 共用或分用解析函数，注册链逐条改 `ParseParamsByOperatorFn`，`OriginOpType` 列表一字不动 |
| `ParseOpToGraphFn`（拆图） | 50 个注册项带拆图。拆图函数体不改；但它依赖 `op_dest` 上已注册的动态输入个数，因此 §3.2.4 的输入个数口径直接决定拆图是否被调用 |
| `ImplyType::AI_CPU` | 4 个算子（eyelike / sequence / concat_from_sequence / split_to_sequence，连同 sequence 的 sequence_construct 变体共 5 个用例目录）的 om 文件名带平台后缀 `model_linux_aarch64.om`；其中 sequence 三者的输入/输出为 sequence 张量、size 为 0，pyACL 不可推理，统一改以 om 图结构比对（§5.3.3） |
| 可选输入（空名占位） | Clip V11 的 min/max；`IsOnnxInputOmitted` 还原空名语义（§3.2.6） |
| 重复同名属性 | 分两族：取首个（Concat/OneHot/Addcmul/Cast）与后者覆盖（Clip V9/Reduce keepdims），逐文件对齐，不统一 |
| 非确定性算子 | multinomial / random_normal_like / random_uniform_like 三个带随机数，两轮**不保证**逐位一致，故按生态混合容差判定（§5.1）；本轮实测后两者恰好逐位一致，multinomial 不一致且已用对照组证明是平台不可复现 |
| 宏生成的批量注册 | `similar_opp_onnx_plugin.cpp` 以宏生成 578 条注册；改两个宏即覆盖 63 个算子，但也意味着**一处写错即波及一半条目**，故其 UT 对 63 个算子逐个展开断言 |
| `-Werror` 与 C++14 | 插件目标显式设 C++14，全仓非 Debug 构建带 `-Werror`：助手不得用 C++17 语法；该目标另加 `-Wno-unused-parameter`，但未使用的**局部变量**仍直接报错（§3.2.9） |
| codecheck 重复代码块 | 57 个文件若各写一份 try/catch 必然命中 ≥10 行重复块；`VisitOnnxAttrs` 把样板压到 3 行/文件（§3.2.3） |
| clang-format 18.1.8 | 仓内 pre-commit 固定该版本；下发用例按仓内风格重排版后有 39 个文件、803 行格式差异，内容未变（§3.2.7） |

---

# 5 可维可测分析

## 5.1 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 判据是**迁移前后一致**，不是与 GPU/CPU 标杆对齐。确定性算子要求两轮推理输出**逐位一致**（md5 相同）；非确定性算子（multinomial / random_normal_like / random_uniform_like）按生态混合容差判定；sequence 类（输入/输出 size 为 0）pyACL 不可推理，改为 om 图反解析后逐字段比对 | 任务书 §3.2；[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | **不涉及**。任务书 §3.3 标注不涉及：本任务为 Host 侧等价重构，不改 kernel 与 tiling，不影响算子执行性能；插件只在 ATC 模型解析期执行一次 | 任务书 §3.3 |
| 内存标准 | **不涉及**。任务书 §3.4 标注不涉及：改造不触及 device 内存分配与 workspace | 任务书 §3.4 |

## 5.2 验证方案（任务书 §3.5 自验要求）

按风险从高到低排四轮，**后一轮不替代前一轮**：

| 轮次 | 内容 | 判据 |
| --- | --- | --- |
| ① 编译 | `bash build.sh --onnxplugin -j32`（experimental 子树无该目标，见 §3.2.9） | rc=0；`-Werror` 下无告警；`nm -D` 未定义 `ge::onnx` 符号为 0 |
| ② framework UT | `bash build.sh -u --ophost --noexec -j64` 后执行 `math_op_host_ut --gtest_filter='Onnx*PluginTest.*'`（含 `--experimental` 一轮） | 624 个新增用例 + 仓内既有用例全部 PASS |
| ③ om 两轮比对 | 56 个生成脚本产出 61 个用例，分别用"迁移前"与"迁移后"两份 `.so` 经 ATC 转 om 并推理 | 见 §5.1 判据；**两轮必须同基线、同 `third_party`** |
| ④ 插件加载证据 | plog 中 `LoadCustomOpLib` + `After ParseParams … have input size: N` | 证明走的是新插件且输入个数正确 |

**验证流程的硬性规则**（由 §3.2.4 的实测推出，写进脚本而非靠人记）：

1. `atc` 返回码与 om 文件存在性**都不作为通过判据**，必须有推理输出比对结果；
2. 两轮出现不一致时，先用"同一份 `.so` 自比"做对照组排除平台抖动，再判断是否为迁移回归；
3. 整数/布尔 dtype 的 `matched_ratio` 是二值的（有一个元素不等就报 0.000000），
   看到 0 要先算真实逐元素一致率再下结论；
4. AI CPU 算子的 om 文件名是 `model_linux_aarch64.om`，按固定名 `model.om` 查找会把
   "转换成功"误判成"无产物"。

## 5.3 验证结果（Ascend910B3 / CANN 9.1.0 / 基线 `8ade3796`，2026-10-18 实测）

### 5.3.1 构建与符号级解耦

| 项 | 结果 |
| --- | --- |
| 插件编译（常规轮） | rc=0，无告警 |
| 未迁移 `.so` | 2 488 200 B，未定义 `ge::onnx` 符号 **5** |
| 已迁移 `.so` | 2 369 416 B，未定义 `ge::onnx` 符号 **0** |
| 符号级比对 | 未定义符号 255 → 239：消失 20 个、新增 4 个，明细见 §1.2 |
| 注册链未动的计数证据 | `DynamicInputRegister` 43 → 43、`DynamicOutputRegister` 40 → 40、`ParseOpToGraphFn` 50 → 50 |
| 拆图函数体未动的 diff 证据 | 57 个插件文件的 diff 中，与 `ParseOpToGraphFn` 相关的非注释改动只有 2 处——`slice` 与 `bitshift` 注册链的缩进（clang-format 对改过的注册链重排），拆图函数体零改动 |
| 构建改动不污染插件产物 | 补完 `cmake/ut.cmake` 的 include 路径后重建插件，产物与 om 两轮所用 `.so` 逐字节一致（md5 与体积均未变） → §5.3.3 的 om 证据不受该构建改动影响 |

### 5.3.2 framework UT

| 轮次 | 结果 |
| --- | --- |
| 常规轮 | 编译 rc=0；`648 tests from 64 test suites ran. [ PASSED ] 648` —— 本任务新增 615 条 + 仓内既有 8 个用例文件的 33 条 |
| experimental 轮 | 编译 rc=0；`188 tests from 20 test suites ran. [ PASSED ] 188` —— 含本任务 `OnnxMaxExperimentalPluginTest` 9 条 |

### 5.3.3 om 两轮比对

两轮在**同一基线 `8ade3796`、同一份 `third_party`** 上分别构建"未迁移"与"已迁移"两套插件，
差异只剩迁移本身。61 个用例（56 个 `gen_model_*.py` + 5 个变体子目录）的判定分布：

| 判定方式 | 用例数 | 结果 |
| --- | ---: | --- |
| 推理输出逐位一致（确定性算子） | 46 | 全部一致 |
| 生态混合容差（非确定性算子：multinomial / random_normal_like / random_uniform_like） | 3 | 全部达标 |
| om 图结构逐字段比对（AI CPU 产物，om 名带平台后缀 `model_linux_aarch64.om`；其中 sequence 三者因输入/输出 size 为 0 另不可推理） | 5 | 规范化后**差异字段 0**（分别 0/2397、0/2525、0/4396、0/4844、0/1020） |
| 两轮同样在转换期失败、报错逐字相同 | 7 | 行为一致，原因与迁移无关 |
| **合计** | **61** | **无一例可归因于本次迁移** |

7 个两轮同样转换失败的用例：`addcmul`、`deformable_offsets`、`random_normal`、
`random_uniform`、`reduce_log_sum_exp`、`reduce_sum_square`、`repeat_with_pad`
（例如 `repeat_with_pad` 是 `SelectEngine: Cannot find engine`，本 SOC 无引擎实现），
适用任务书 §3.2.5 平台容错条款。

om 文件内嵌 `ASCEND_CUSTOM_OPP_PATH` 路径串与 `dynamic_const_<pid>_N` 节点名，
**两轮 om 的 md5 必然不同**（AI CPU 那 5 个的 om 大小两轮差 3 字节，就是路径串
`pre`/`post` 的长度差）。因此 om 结构比对走 `atc --mode=1` 反解析成 json，
把路径与 PID 归一化、属性按 key 排序后再比——这是口径，不是放宽。

### 5.3.4 测试强度自证：变异判错与覆盖率

UT 全绿只证明"没踩到"，不证明"踩了能被发现"。为此另做两项：

**变异测试**：对 7 条最关键契约逐条注入反向改动，看 UT 是否变红。

| # | 注入的错误契约 | 结果 |
| ---: | --- | --- |
| M1 | 缺 `attribute` 键时返回 FAILED（应 SUCCESS） | 56 条用例失败 ✔ |
| M2 | 非法 attribute JSON 返回 SUCCESS（应 FAILED） | 66 条失败 ✔ |
| M3 | `type` 不符时仍读取（应按缺失处理） | 32 条失败 ✔ |
| M4 | 缺 `f` 键时回落算子默认值（应取 0.0） | 9 条失败 ✔ |
| M5 | 输入个数只用 `GetInputsSize()`（应读 `input` 属性） | **0 条失败 ✘** |
| M6 | Cast 失效判据改回 `== -1` | 编译即被拦 ✔ |
| M7 | 去掉 name 兜底 | 4 条失败 ✔ |

**6/7 被抓到。M5 结构上不可能被 UT 抓到**——UT 以 `DynamicInputRegister` 构造输入个数，
两种读法在 UT 下返回值相同（§3.2.4）。也就是说，本任务最危险的那条契约**只有 om 轮能把关**。
这正是 §5.2 把 om 两轮列为必跑轮次、而不是"UT 过了就行"的原因。

**覆盖率**：手写迁移的 165 个解析函数（含其 visitor lambda）行覆盖 **86.6%**（2010/2322），
**完全未被执行的 0 个**。未覆盖部分集中在 `OP_LOGE` 失败日志分支与 §3.4 列出的不可达分支。
任务书没有覆盖率门限，该数字不作为达标判据。

统计口径有两条，否则读数不可信：
① 下发 UT 以 `#include` 文本引入插件源码，gcda 挂在 `math_op_framework_ut_cases_obj`
目标下而非插件目标下，函数名需 `c++filt` 还原后匹配；
② 必须"全新构建 → 清空 gcda → 只跑一次 → 再 gcov"，否则 gcda 与 gcno 来自不同次构建，
`gcov` 只在 stderr 提示 `stamp mismatch with notes file`，而把整个翻译单元读成 0。
本次统计以 `stamp mismatch` 出现 0 次为前置条件。

## 5.4 测试用例规划

下发用例包 57 个文件 / 624 个用例，按四类场景组织：

1. 属性齐全的正常解析；
2. 属性缺失 / 值字段省略 / `attribute` 数组为空 → 走默认值或零值（§3.1.3）。
   **该类对 4 个文件天然不适用**：`concat`（`axis`）、`sign_bits_pack`（`size`）、
   `sign_bits_unpack`（`size`/`dtype`）的属性必填，缺失即 FAILED；`sum` 不读属性。
   这 4 个文件的"属性缺失"分支由第 3 类的 FAILED 用例覆盖；
3. 非法 attribute JSON → 按该算子契约返回 FAILED 或 SUCCESS（§3.2.6(5)）；
4. 边界与校验：不支持 dtype、`type` 与声明不符按缺失、空列表 / 负值透传、重复同名属性、
   输入个数边界 0/1/3/16。

用例以 `ge::Operator` 构造输入，**不构造任何 protobuf 对象**（任务书 §3.5.1）。
用例包已逐项核验与基线函数名、属性名、注册关系一致，可直接作为行为契约；
9 处必要修订及其实测依据见 §3.2.7。

## 5.5 兼容性分析

- **对 onnx 模型使用者**：无感知。opset 覆盖与算子映射关系逐条不变，`OriginOpType` 一字未改。
- **对仓内其它模块**：`math_onnx_plugin_util.h` 只新增 `inline` 函数与枚举，不改既有函数签名；
  `onnx_common.h` 不动，未迁移的 TF 插件与其它 onnx 场景不受影响。
- **对 GE 版本**：`ParseParamsByOperatorFn` 在 CANN 9.1.0 已可用；§3.1.3 的零值省略规则
  在 9.1.0 / 9.2.0 / develop 三个分支下写法一致（`contains("i") ? 读 : 0`），不需按版本分叉。
- **对已合入的同类改造**：本次写法与仓内 9 个已迁移插件一致，未引入第二套风格。

## 5.6 可测性

- 57 个插件文件**逐一**有对应 UT 文件，命名 `test_<原文件名>`，测试套件名
  `Onnx<Op>PluginTest`，与仓内既有 framework UT 的命名模式一致。
- 失败路径基本都可由 UT 触达，两处例外：一是 §3.4 列出的 `dynamic_cast` 判空分支，
  新接口下不可达；二是输入个数的取法（§5.3.4 的 M5），UT 以 `DynamicInputRegister`
  构造输入个数，两种取法在 UT 下返回值相同，只有 om 轮能区分。
- 两轮 om 比对脚本对每个用例落盘：`model.onnx` / `model.om` / `out_*.npy` /
  `plugin_evidence_<算子>.txt`（plog 中的插件加载与输入个数证据）/ `atc_*.log` / `infer_*.log`，
  可逐算子复查，不依赖汇总结论。

## 5.7 限定（本章数据不能用来做什么）

1. 全部数据取自 **Ascend910B3（Atlas A2）/ CANN 9.1.0 / aarch64 / 基线 `8ade3796`** 这一组合，
   不等同于其它 SOC 或其它 CANN 版本上的结果；任务书 §3.1 允许"任一硬件通过即可"，
   本工程未在 Ascend 950 上复验。
2. `.so` 的 md5 只在"同基线 + 同 `third_party` 快照"下可比。`third_party/` 由 `build.sh`
   拉取、仓内不做版本固定，**不同时间构建可能得到不同的 opbase 版本**，届时 md5 会变而不代表回归。
3. 两轮 om 的 md5 **必然不同**（§5.3.3），不可用 om 文件 md5 作为一致性判据。
4. "86.6% 行覆盖"是手写迁移函数（含其 visitor lambda，共 165 个）的行覆盖，**不是**
   整个插件库的覆盖率，也不构成任何达标阈值——任务书未规定覆盖率要求。
5. 7 个两轮同样转换失败的用例，本工程只证明了"两轮行为一致"，**未证明**这些算子在本 SOC 上可用。
6. §5.3.4 的变异测试证明的是"UT 能发现这 7 类错误"，不是"UT 能发现任何错误"。

---

# 6 交付与合入

## 6.1 交付件（任务书 §4）

| 序号 | 交付件 | 位置 / 说明 |
| ---: | --- | --- |
| 1 | 迁移后代码（PR） | ops-math 仓 PR，122 个文件，7 个提交：公共头(1) → common/src(14) → conversion(14) → math(25) → random+experimental(4) → UT(57) → 构建接线(7) |
| 2 | UT 用例及执行结果 | `task_submission/2.1 精度自验证报告.xlsx`（UT sheet）+ `2.2 精度自验证日志.log` |
| 3 | om 转换精度验证记录 | 同上（om sheet）：含验证模型、ATC 转换命令、两轮比对结果与一致性说明 |
| 4 | 单算子迁移自查表 | `task_submission/1 自验证步骤说明.md` 附录（57 行 × 8 列，按任务书 §3.5 自查清单逐项填写）+ `2.1 精度自验证报告.xlsx` 的自查表 sheet |
| 5 | 性能自验证报告及日志 | 不涉及（任务书 §3.3） |
| 6 | 内存自验证报告及日志 | 不涉及（任务书 §3.4） |

交付件 1 走 ops-math 仓的 PR；交付件 2~6 按任务书 §4 放开发者私仓的 `task_submission/` 目录，
文件名与任务书给出的模板一字一致，**不随 PR 合入上游**（`cann/ops-math` 全历史无此目录）。

## 6.2 PR 合入（任务书 §5）

| 任务书 §5 要求 | 本工程做法 |
| --- | --- |
| 目标目录为各插件文件所在目录及对应 UT 目录 | 改动全部落在这两类目录 + 1 个公共头 + 7 个构建文件 |
| 不得包含明显冗余内容（反例 1：文件数不正常） | 122 个文件 = 57 插件 + 57 UT + 1 公共头 + 7 构建文件，逐类可解释（§3.2.10） |
| 反例 2：多出无用的代码、测试和编译文件 | 新增的 5 个叶子 CMakeLists 是"不加就收不到用例"的必需项，2 个修改分别是"不加就链不过"与"不加就编不过"（§3.2.9） |
| 不得携带个人敏感信息 | 提交前以脚本扫全部交付件与 diff（含 xlsx 解包后内容），作为提交门禁的一环 |
| 建议按插件文件/算子分组 | 插件改造按目录分 4 个提交（14 / 14 / 25 / 4 个文件）；UT 单独成一个提交（57 个新增文件，与插件改动互不混杂，便于分别审阅）；公共头与构建接线各一个 |

**静态检查准备**：仓内 pre-commit 固定 clang-format 18.1.8，提交前以该版本格式化；
本地另跑 R5（圈复杂度）/ R6（嵌套深度）/ R7（函数 NBNC）/ G.FMT.04 / G.FMT.05 /
重复代码块 / license 头 / `#pragma once` 八项；G.EXP.22（除零）与 G.FMT.06（续行缩进）
本地脚本覆盖不到，已逐处人工核对。

---

# 7 参考资料

1. onnx 算子开发说明（调用示例）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/advanced_programming/ai_framework_adaptation/onnx_framework/call_examples.md>
2. 生态算子开源精度标准（混合容差）：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
3. 迁移适配 issue（含已合入的同类 PR 示例）：<https://gitcode.com/cann/ops-math/issues/2892>
4. 仓内已迁移样板插件：`one_hot_d` / `reverse_sequence` / `correlation` / `npu_indexing` /
   `shape` / `bernoulli` / `npu_rotary_mul` / `npu_format_cast` / `npu_dtype`
