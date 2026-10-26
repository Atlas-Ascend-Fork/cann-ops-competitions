# ops-nn算子onnx适配 设计文档

| 项 | 内容 |
| --- | --- |
| 任务 | 10月社区任务-ops-nn算子onnx适配（任务编号 13） |
| 目标仓 | https://gitcode.com/cann/ops-nn |
| 关联 issue | https://gitcode.com/cann/ops-nn/issues/5115 |
| 改造类型 | **Host 侧等价重构**（ONNX 插件框架适配），不涉及 kernel / tiling / 算子原型 |
| 迁移范围 | 任务书迁移范围表全部 **74 个插件文件 / 75 个算子条目 / 76 处注册点** |
| 对标基线 | 同仓迁移前版本（任务发布时的仓代码），由开发者自行构建获取 |
| 验证平台 | Atlas A2（任务书 3.1：A2 与 Ascend 950 任一验证通过即可，推荐 A2） |
| CANN 版本 | > 9.0.0（任务书 3.1 第 2 条） |
| 三方件 | nlohmann/json —— 用仓内 `third_party` 自带版本，不另引新版本（任务书 3.1 第 3 条） |

---

# 1 需求背景（required）

## 1.1 需求来源

ONNX 插件框架 protobuf 解耦需求，对应 ops-nn 仓 issue
[#5115](https://gitcode.com/cann/ops-nn/issues/5115)：
"把 onnx 插件中注册的函数由 `ParseParamsFn` 替换为 `ParseParamsByOperatorFn`，
实现对 protobuf 依赖的消除"。

任务书给出迁移范围表，列出仓内**仍通过 `ParseParamsFn` 注册参数解析函数**的全部 ONNX 插件。

## 1.2 背景介绍

### 1.2.1 现状：`ParseParamsFn` 与 protobuf 生成代码强耦合

迁移前的插件注册与解析形态：

```cpp
// 入参是 protobuf 基类指针，实际传入 ge::onnx::NodeProto
Status ParseParamsXxx(const google::protobuf::Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    for (const auto& attr : node->attribute()) {        // 遍历 protobuf attribute
        if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            op_dest.SetAttr("alpha", attr.f());          // 按 i/f/s 等字段取值
        }
    }
    op_dest.SetAttr("name", node->name());               // 节点名来自 protobuf
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Xxx").FrameworkType(ONNX).OriginOpType({...})
    .ParseParamsFn(ParseParamsXxx).ImplyType(ImplyType::TVM);
```

由此带来任务书 1 节列出的三类问题：

| # | 问题 | 具体表现 |
| --- | --- | --- |
| 1 | 与 protobuf 生成代码强耦合 | 插件源码 `#include "onnx/proto/ge_onnx.pb.h"`，proto 定义变动会波及全部插件 |
| 2 | 编译依赖重 | 需拉入 protobuf 头与生成代码，编译链路长 |
| 3 | UT 成本高 | 单测必须构造 `ge::onnx::NodeProto` 对象，用例编写与维护成本高 |

### 1.2.2 现状分析：仓内已迁移的参考实现

仓内已有少量插件完成迁移，是本次迁移的**行为基准**（而非凭文档推测）：
`ada_cast`、`elu`、`leaky_relu`、`bounding_box_decode`、`bernoulli`。

本设计的 attribute JSON 读法（特别是 3.2.2 标量浮点以字符串承载、3.2.3 proto3 零值键省略
两条）与这些已迁移实现的读法、以及它们**仓内配套 UT 的输入表示**交叉核对一致 ——
与任务书 2.4 节"数据来源与核对方式"同一口径。

### 1.2.3 迁移后的目标形态

```cpp
using json = nlohmann::json;

Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)
{
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) { /* 逐属性解析 */ }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception.");
            return FAILED;
        }
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Xxx").FrameworkType(ONNX).OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParamsXxx).ImplyType(ImplyType::TVM);
```

## 1.3 迁移范围

任务书范围表的三个计数彼此不等，核对结论如下（可由仓内源码复核）：

| 口径 | 数量 | 说明 |
| --- | --- | --- |
| 任务书范围表行数（算子条目） | **75** | 文件 × 算子一行 |
| 去重后的插件文件数 | **74** | `quant/dynamic_quant/framework/npu_dynamic_quant_onnx_plugin.cpp` 出现 2 行（`NPUDynamicQuant` / `NPUDynamicQuantV2`，同文件两个算子类型） |
| 仓内实际的 `.ParseParamsFn(` 注册点 | **76** | 比表行数多 1：`activation/softmax_v2/framework/softmax_onnx_plugin.cpp` 对 `SoftmaxV2` 注册**两次**（opset 8~12 → `ParseParamsSoftmaxV11`，opset 13+ → `ParseParamsSoftmaxV13`），被任务书"同一算子的多个 ONNX 版本合并为一行"的规则并成 1 行 |

**三者互不矛盾**，74 / 75 / 76 分别是"文件 / 算子条目 / 注册点"三个不同口径。
**迁移以注册点为单位执行**，因此完成口径是 76 处全部改完，而不是 74 或 75 ——
按文件或按条目计数会漏掉 softmax 的第二处注册。

---

# 2 需求分析（required）

## 2.1 需求描述

完成上述 76 处注册点的 protobuf 解耦迁移，**等价重构**：迁移前后插件解析行为必须完全一致
（任务书 2.1 开篇）。完成口径有两条，缺一不可：

1. **解耦成立**：插件 `.so` 对 `ge::onnx` / protobuf / protobuf 带入的 absl / 旧注册 API
   四类未定义符号全部归零（判据见 5.3.4）；
2. **行为不变**：framework UT 全通过，且同一 onnx 模型经迁移前/迁移后插件转 om 后推理输出
   逐数值相等（任务书 3.2 主判据，见 5.3.2）。

## 2.2 需求拆解（逐条对应任务书 2.1）

| 任务书 2.1 条号 | 要求 | 本设计的落地方式 | 详见 |
| --- | --- | --- | --- |
| 1 | 解析函数签名替换为以 `ge::Operator` 为入参 | `Status ParseParams${OpName}(const ge::Operator&, ge::Operator&)`，**保持各插件现有函数命名**（含 `parse_params_hard_sigmoid` 这类既有下划线命名，不统一改名） | 3.3.1 |
| 2 | 注册方式改为 `.ParseParamsByOperatorFn(...)` | 76 处注册点逐一替换，`FrameworkType` / `OriginOpType` / `ImplyType` 三要素不动 | 3.3.1 |
| 3 | 属性解析改为 `op_src.GetAttr("attribute")` + nlohmann/json | 统一取 JSON 字符串后 `json::parse`，解析异常返回 FAILED | 3.1.1、3.2 |
| 4 | 移除 protobuf 依赖，include 替换 | 去掉 `onnx_common.h`（它 include `ge_onnx.pb.h`），改 include 新公共头 + `register/register.h` | 3.3.2 |
| 5 | 算子名改为 `GetOpName(op_src)` | 日志与 `SetAttr("name", ...)` 统一改用 `GetOpName`，不再用 `node->name()` | 3.3.1 |
| 6 | 公共工具优先复用；确需新增用**仓前缀命名** | 新增 `common/inc/framework/nn_onnx_plugin_util.h`（`nn_` 前缀），参考 `plugin_util.h` 的做法 | 3.3.2 |
| 7 | 语义保持：缺失→默认值且 SUCCESS；非法→日志 + FAILED；返回值一致；不顺手重构 | 逐属性核对 `SetAttr` 键集合与基线一致；基线上的怪写法原样保留 | 3.1.1、3.2.5 |
| 7 附 | **注意失效判据**：不得照抄永不成立的失败分支 | 按工具函数的真实返回约定判定失败分支 | 3.2.5 |

## 2.3 改动边界（任务书 2.2）

**在范围内**：各算子目录下 `framework/*_onnx_plugin.cpp` 及对应 framework UT。

**明确不在范围内**（任务书 2.2 第 2、3 条与 7 节第 4 条）：

| 不改的东西 | 依据 |
| --- | --- |
| 算子 kernel / tiling / 算子原型 | 任务书 2.2 第 2 条 |
| `ParseOpToGraphFn` 子图拆图函数（如 Conv、Gemm 等注册为 PartitionedCall 的算子） | 任务书 2.2 第 3 条 |
| `DynamicInputRegister` / `DynamicOutputRegister` 注册语义 | 任务书 7 节第 4 条 |
| 与解耦无关的任何逻辑 | 任务书 2.1 第 7 条 |

---

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 等价重构的语义契约

本任务没有计算公式，待保持的"公式"是**同一个 onnx 节点 → 同一组 `op_dest` 属性**这个映射：

```
对任意 onnx 节点 N：
    SetAttrSet( ParseParams_后(N) )  ==  SetAttrSet( ParseParams_前(N) )
且  返回值_后(N)                      ==  返回值_前(N)          ∈ {SUCCESS, FAILED}
```

其中 `SetAttrSet` 指写入 `op_dest` 的属性键值集合。任务书 3.5 第 2 条的"与基线的参数对应关系"
即此式。三条可核的分解判据：

| 判据 | 口径 |
| --- | --- |
| 属性齐全 | `SetAttr` 键集合与基线逐文件一致 |
| 属性缺失 | 保持原有默认值行为并返回 **SUCCESS** |
| 属性非法 | JSON 解析异常或未知异常 → 记日志并返回 **FAILED** |

### 3.1.2 attribute JSON 的字段与数据类型

`op_src.GetAttr("attribute")` 返回的 JSON 文本结构（任务书 2.4 字段表）：

| 字段名 | 位置 | 描述 | 数据类型 | 值域 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| `attribute` | 顶层键 | onnx 节点属性数组 | JSON array | 元素为属性对象 | JSON 非法 → FAILED |
| `name` | 属性内 | onnx 属性名 | string | 与算子属性定义一致 | 未匹配到算子属性时忽略（保持迁移前行为） |
| `type` | 属性内 | `AttributeProto` 类型枚举 | int | FLOAT=1、INT=2、STRING=3、**TENSOR=4**、FLOATS=6、INTS=7、**STRINGS=8** | 类型与取值键不匹配 → 按缺失处理 |
| `i` | 属性内 | 标量整型载体 | int64（JSON 数字） | 各算子属性值域 | 见 3.2.3 |
| `f` | 属性内 | 标量浮点载体 | **string**（如 `"f":"0.25"`） | 各算子属性值域 | 见 3.2.2、3.2.3 |
| `s` | 属性内 | 字符串载体 | string（经 CEscape 转义） | — | 见 3.2.4 |
| `ints` / `floats` | 属性内 | 整型/浮点列表载体 | JSON array（数字） | 各算子属性值域 | 见 3.2.3 |
| `t` | 属性内 | 张量载体（`TENSOR` 类型属性） | object，payload 在 `t.raw_data` | — | 见 3.2.4 |

> **TENSOR=4 与 STRINGS=8 不在任务书 2.4 的类型表里**，但迁移范围内确有算子使用
> （`ConstantOfShape` 的 `value` 为 TENSOR；`BasicLSTMInplaceFillWindowCache` 的
> `activations` 为 STRINGS）。新公共头的 `OnnxAttrType` 枚举按 ONNX 标准补全
> FLOAT=1 … GRAPHS=10 全部十项，避免后续插件再踩。

### 3.1.3 不涉及项

| 项 | 本任务 | 原因 |
| --- | --- | --- |
| 支持形状 | **不涉及** | 插件只做 onnx 节点属性 → `op_dest` 属性的映射，不接触张量数据，无形状约束 |
| 支持数据类型（计算侧） | **不涉及** | 同上；3.1.2 给出的是 attribute JSON 的字段类型，不是算子计算的 dtype |
| 数学公式 | **不涉及** | 等价重构，不改变任何算子的计算逻辑（任务书 3.3 节） |

## 3.2 核心设计决策

本节是方案的地基。契约不是推测的，依据为 **GE 解析器公开源码多个 ref 交叉比对**：
`parser/parser/common/convert/pb2json.cc` 与 `message2operator.cc`。

### 3.2.1 头号设计点：`op_src` 不挂 TensorDesc，`GetInputsSize()` 解析期恒为 0

这是整个迁移**唯一不是机械替换**的地方。

解析期回调拿到的 `op_src` 由 parser 直接从 `NodeProto` 构造，身上**只有 5 个属性**：

| 属性 | 类型 | 内容 |
| --- | --- | --- |
| `name` | VT_STRING | 节点名 |
| `op_type` | VT_STRING | 节点类型 |
| `input` | VT_LIST_STRING | 输入名列表 |
| `output` | VT_LIST_STRING | 输出名列表 |
| `attribute` | VT_STRING | 整个属性数组的 JSON 文本 |

**不挂任何 TensorDesc**。因此 `op_src.GetInputsSize()` / `GetOutputsSize()` 在解析期**恒为 0**，
与迁移前 `node->input_size()` / `output_size()`（节点声明的输入/输出个数）**不等价**。
真正等价的是 `input` / `output` 两个 `VT_LIST_STRING` 属性的长度。

**方案**：公共头提供两个计数函数，优先读列表属性，取不到时回退描述符个数：

```cpp
inline size_t GetOnnxInputCount(const ge::Operator& op_src);
inline size_t GetOnnxOutputCount(const ge::Operator& op_src);
```

回退分支是为 framework UT 服务的：UT 用 `DynamicInputRegister` 挂载输入来构造
"节点有 n 个输入"的场景，此时没有 `input` 属性。

**为什么必须单独设计**：迁移范围内有 17 个文件用到输入/输出计数。其中
`Gemm` / `LayerNormalization` / `MaxPool` 若照搬 `GetInputsSize()`，会拆出错误形状的子图
或直接 ATC 失败 —— 而这是一个**静默**错误：编译通过、UT 也可能通过，只在 om 转换时暴露。
以 `Gemm` 为例，输入个数（2 或 3）直接决定 `ParseOpToGraphGemm` 拆出 `MatMulV2`
还是 `MatMulV2 + Add`。

### 3.2.2 标量浮点以字符串承载，须先取 string 再转 float

`pb2json.cc` 用 `%g` 把标量 float 格式化进一个长度 10 的缓冲区：结果 ≤9 字符时落为**字符串**
（`"f":"0.25"`），否则落为 JSON 数字。故读取必须先取 `std::string` 再转 float，
复用仓内 `common/inc/framework/plugin_util.h` 的 `StrToFloat`（空串/非法/越界返回 false），
失败返回 FAILED。`i` / `ints` / `floats` 仍为 JSON 数字。

**已知副作用**：`%g` 只保留 6 位有效数字，两个相邻 fp32 位模式（`3f9e0652` / `3f9e0653`）
都会变成 `"1.23457"`。**属性值需要 >6 位有效数字时，迁移前后不可能逐位一致** ——
这是 GE 桥接的上限，不是本次迁移引入的缺陷，已反馈
[cann/ge#656](https://gitcode.com/cann/ge/issues/656)。
同一处代码对 `double_data` 没有 TYPE_DOUBLE 分支，typed DOUBLE 列表会变成 `[null,...]`。

### 3.2.3 proto3 零值键省略：键缺省取 0，**不得回落算子默认值**

属性**存在**但 `f` / `i` / `s` 键缺省时（proto3 省略值为 0 的标量），取值为 `0.0 / 0 / ""`，
**不得回落到算子默认值**。例：`HardSigmoid` 的 `alpha` 属性存在、无 `f` 键 → `alpha = 0`，
而不是默认值 0.2。

这条与"属性整个缺失则用默认值"是两种不同情形，极易混为一谈：

| 情形 | JSON | 取值 |
| --- | --- | --- |
| 属性整个缺失 | `{"attribute":[]}` | 算子默认值（如 alpha=0.2），返回 SUCCESS |
| 属性存在但值键缺省 | `{"attribute":[{"name":"alpha","type":1}]}` | **0.0**，不是 0.2 |
| 属性名匹配但 `type` 不符 | `{"attribute":[{"name":"alpha","type":2,"i":3}]}` | 按缺失处理（走默认值） |

与仓内已迁移的 `ada_cast` / `elu` 处理一致，另见任务书 2.4 节"浮点标量 f 的两点约定"。

### 3.2.4 `bytes` 字段走 `protobuf::CEscape`，不是 base64

`AttributeProto.s` 与 `TensorProto.raw_data` 经 `CEscape` 转义（**不是 base64** ——
这是最容易想当然的一处）。`ConstantOfShape` 的 `value` 张量 payload 落在 `t.raw_data`，
需在插件内自行还原。

**方案**：在新公共头内自带一个不依赖 protobuf 的 `CUnescapeBytes`，处理
`\n \r \t \a \b \f \v`、1~3 位八进制转义与兜底。不能调 `protobuf::CUnescape` ——
那会把 protobuf 依赖重新引回来，与本任务目标相违。

### 3.2.5 失效判据：不照抄永不成立的失败分支（任务书 2.1 第 7 条附注）

任务书举的典型是：以 `if (om == -1)` 判断 dtype 映射失败，而工具函数
（如 `GetOmDtypeFromOnnxDtype`）对未命中实际返回 `ge::DT_UNDEFINED`，该分支永不触发。

**设计上的处理方式**（三条并用，而不是只靠人眼看）：

1. 逐条核对失败分支的判据与所调用工具函数的**真实返回约定**是否一致；
2. 构建开 `-Wall -Wextra`（含 `-Wtype-limits` / `-Wtautological-compare`），
   让恒真/恒假比较由编译器机械筛出；
3. 对修正过的判据**补 UT 用例覆盖**（"非法取值应返回 FAILED"一类），
   否则修正本身没有回归保护。

**同时要克制**：基线上存在的"怪写法"（如先 `SetAttr` 再校验、同一属性被两段代码分别处理）
予以**原样保留**并在代码 PR 的描述中逐条列出 —— 任务书 2.1 第 7 条明令"不得顺手重构与解耦无关的
代码"，等价重构的目标是行为一致，不是代码变好看。只有**确属失效判据**的才按真实返回约定修正。

## 3.3 算子实现

### 3.3.1 host 侧设计：接口定义与注册形态（任务书 2.3）

解析函数接口（保持各仓现有函数命名）：

```cpp
domi::Status ParseParams${OpName}(const ge::Operator& op_src, ge::Operator& op_dest);
```

注册宏五要素形态：

```cpp
REGISTER_CUSTOM_OP("${OpName}")
    .FrameworkType(ONNX)
    .OriginOpType({...})
    .ParseParamsByOperatorFn(ParseParams${OpName})
    .ImplyType(ImplyType::TVM);
```

单个注册点的迁移是固定的六步，76 处一致执行：

| 步 | 动作 |
| --- | --- |
| 1 | 签名改 `(const ge::Operator& op_src, ge::Operator& op_dest)` |
| 2 | 注册改 `.ParseParamsByOperatorFn(...)`，其余三要素不动 |
| 3 | 属性取值改 `op_src.GetAttr("attribute")` + `json::parse`，外层 `try/catch` 两级（`json::exception` 与 `...`）均记日志返回 FAILED |
| 4 | 节点名改 `GetOpName(op_src)`（日志与 `SetAttr("name", ...)`） |
| 5 | 输入/输出计数改 `GetOnnxInputCount` / `GetOnnxOutputCount`（仅 17 个文件涉及，见 3.2.1） |
| 6 | include 去掉 `onnx_common.h`，改 `nn_onnx_plugin_util.h`；`ParseOpToGraphFn` 函数体不动 |

### 3.3.2 公共头拆分方案（任务书 2.1 第 6 条）

**新增** `common/inc/framework/nn_onnx_plugin_util.h` —— `nn_` 仓前缀，避免与其他算子仓重名，
做法参考 `common/inc/framework/plugin_util.h`：

| 内容 | 来源 | 为什么必须在这里 |
| --- | --- | --- |
| `Vec2Tensor` / `CreateScalar` / `ChangeFormatFromOnnx` | 从 `onnx_common.h` **原样搬出** | 这三个 helper 本身不依赖 protobuf，但原先与 `ge_onnx.pb.h` 挤在同一个头里；带拆图函数的插件离不开它们 —— 不搬出来就无法"去掉 `onnx_common.h`"与"保留拆图函数"兼得 |
| `GetOnnxAttrFloat / Int / String / Type / Name` | 新增 | 属性读取的统一入口，封装 3.2.2、3.2.3 两条约定，避免 76 处各写一遍 |
| `OnnxAttrType` 枚举（FLOAT=1 … GRAPHS=10） | 新增 | 含任务书类型表未列的 TENSOR=4 / STRINGS=8，见 3.1.2 |
| `GetOnnxInputCount` / `GetOnnxOutputCount` | 新增 | 3.2.1 的头号设计点 |
| `CUnescapeBytes` | 新增 | 3.2.4，不依赖 protobuf |

**改造** `common/inc/framework/onnx_common.h`：include 新头并删去搬走的定义（避免仓内出现
第二份实现），只保留仍依赖 `NodeProto` 的 `GetEnableUncacheFromNode`，
供**尚未解耦**的插件继续使用。

新头的头注释显式写明约束：
`must not include onnx_common.h, because that header pulls in ge_onnx.pb.h`。
这条是防回归的 —— 一旦有人为了省事在新头里 include 旧头，整个解耦当场失效，而编译不会报错。

**健壮性约定**：`{"attribute":[5]}` 这类不规范输入不应让整个解析失败，
故 `GetOnnxAttrName` / `GetOnnxAttrType` 先判 `attr.is_object()`，按"没有这个字段"忽略。

### 3.3.3 kernel 侧设计

**不涉及。** 任务书 2.2 第 2 条明确："开发仅涉及各算子目录下 `framework/*_onnx_plugin.cpp`
及对应 framework UT，不涉及算子 kernel、tiling、算子原型的开发与修改。"

## 3.4 支持硬件

本任务为 Host 侧改造，插件代码与硬件无关；任务书 3.1 第 1 条规定"Atlas A2 与 Ascend 950，
**任一硬件验证通过即可，推荐 A2**"。

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √（本任务的验证平台） |
| Atlas A3 | √（同一份 Host 侧代码，未在其上验证） |
| Ascend 950PR / 950DT | √（同一份 Host 侧代码，未在其上验证） |

> om 精度比对的迁移前后基线**必须在同一硬件上获取**（任务书 3.1 第 1 条）。

## 3.5 算子约束限制

| # | 约束 | 说明 |
| --- | --- | --- |
| 1 | 标量浮点属性精度上限 | 属性值有效数字 >6 位时，GE 的 `%g` 桥接使迁移前后**不可能**逐位一致（3.2.2）。非本次改动引入，但构成"逐位一致"判据的上限 |
| 2 | typed DOUBLE 列表 | `pb2json.cc` 无 TYPE_DOUBLE 分支，`double_data` 会序列化为 `[null,...]`（3.2.2） |
| 3 | 解析期无 TensorDesc | `op_src` 上取不到任何输入/输出的 dtype / shape（3.2.1）。需要这些信息的逻辑只能放在 infershape 或拆图阶段，**不能放在 ParseParams** |
| 4 | `GetInputsSize()` 恒为 0 仅限解析期 | 其他阶段的 `ge::Operator` 不适用此结论 |
| 5 | 不解耦 `onnx_common.h` 本身 | 该头仍保留 `ge_onnx.pb.h` 与 `GetEnableUncacheFromNode`，供仓内尚未迁移的插件使用；彻底删除需等全仓迁完 |
| 6 | 范围外仍有文件 include `onnx_common.h` | 仓内另有 4 个插件 include 该头，但它们**在基线上就已迁完 API**（已是 `ParseParamsByOperatorFn`），对 protobuf 的使用仅为 `using NodeProto = ge::onnx::NodeProto;` 类型别名或残留 include，不产生动态符号引用，故不在任务书的 74 个文件内 |
| 7 | `experimental/` 子树的 UT 收集不到 | `cmake/variables.cmake` 的 `OP_CATEGORY_LIST` 不含 `experimental`、`ENABLE_EXPERIMENTAL` 默认 OFF，该子树不被 `add_subdirectory`，放在其下的 UT 收集不到，须落位到被收集的目录并相应调整 include |

---

# 4 特性交叉分析

| 交叉对象 | 是否受影响 | 结论依据 |
| --- | --- | --- |
| 算子计算精度 | **否** | 不改 kernel / tiling；以 om 两轮逐数值相等为判据（5.3.2） |
| 算子性能 | **否** | 不改变计算逻辑与模型转换产物结构（任务书 3.3） |
| Host 内存占用 | **否** | 解析期一次性构造 JSON 对象后即释放，无常驻结构（任务书 3.4） |
| ATC 模型转换产物（om）结构 | **否** | 以 om 图逐字段比对为兜底判据（5.3.2） |
| `ParseOpToGraphFn` 拆图行为 | **否** | 函数体未改；拆图依赖的输入计数由 3.2.1 保证等价 |
| 动态输入注册（`DynamicInputRegister`） | **否** | 注册语义未改；`GetOnnxInputCount` 的回退分支专为该场景保留 |
| 仓内**尚未迁移**的其他 ONNX 插件 | **否** | `onnx_common.h` 保留 `GetEnableUncacheFromNode` 与 protobuf include，旧插件继续可编 |
| 其他算子仓（ops-math 等）的同名公共头 | **否** | 新头用 `nn_` 仓前缀（任务书 2.1 第 6 条） |
| 编译依赖 | **改善** | 74 个插件不再拉 `ge_onnx.pb.h`；`.so` 的 protobuf 相关未定义符号归零（5.3.4） |
| framework UT 的编写方式 | **改善** | UT 直接用 `ge::Operator` 构造输入，不再构造 protobuf 对象（任务书 3.5 第 1 条） |

---

# 5 可维可测分析

## 5.1 精度标准 / 性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度 · 主判据 | 同一 onnx 模型、同一输入，分别用迁移前/迁移后插件转 om 并推理，输出**完全一致（逐数值相等，不设 rtol/atol 阈值）** | 任务书 3.2 第 2 条 |
| 精度 · 辅助判据 | om 推理输出与用例 golden 按**混合容差标准**比对（fp32 rtol=2⁻¹⁰≈9.77e-4、atol=2⁻¹⁶≈1.53e-5，逐元素硬上限 max(fixed_limit, 32·ULP)，整体匹配率 ≥0.99，整数/布尔逐位精确），并校验输出 dtype/shape | 任务书 3.2 第 3 条；[混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 精度 · 平台容错 | 个别算子在部分 SOC 上因 CANN op-info 无条目等原因无法完成 ATC 转换时，记录平台/算子/报错原文，该项以"迁移前后两轮行为一致"为准 | 任务书 3.2 第 5 条 |
| framework UT | 开启仓内 `OP_HOST_UT` 开关构建，`--gtest_filter='Onnx*PluginTest.*'` 全部通过；UT **不得构造 protobuf 对象** | 任务书 3.5 第 1 条 |
| 性能 | **不涉及** —— Host 侧等价重构，不改变算子计算逻辑与模型转换产物结构，不设性能验收指标 | 任务书 3.3 |
| 内存 | **不涉及** | 任务书 3.4 |

## 5.2 兼容性分析

| 维度 | 结论 |
| --- | --- |
| 对下游（GE / ATC） | 注册宏五要素形态不变，仅解析回调的注册入口由 `ParseParamsFn` 换为 `ParseParamsByOperatorFn`，两者都是 `OpRegistrationData` 的公开 API |
| 对仓内未迁移插件 | 无影响，`onnx_common.h` 保留 protobuf include 与 `GetEnableUncacheFromNode` |
| 对既有 om 模型 | 无影响，转换产物结构未变 |
| CANN 版本 | 要求 > 9.0.0（任务书 3.1 第 2 条） |
| 旧接口的版本适配问题 | 第 1 轮加载**迁移前**（旧接口）插件时若报 `undefined symbol: …ge::onnx::AttributeProto…`，属 CANN 版本与旧接口的适配问题，与迁移代码无关，按用例包 `verify_om/compat/README.md` 处理；**迁移后的插件不挂该兼容库** —— 迁移后已不依赖 protobuf 类型，这正是本任务的核心结论 |
| 三方件 | nlohmann/json 用仓内 `third_party` 自带版本，不引入新版本（任务书 3.1 第 3 条） |

## 5.3 测试设计

### 5.3.1 framework UT（任务书 3.5 第 1、3 条）

```bash
bash build.sh -u --ophost --noexec -j<N>
./build/tests/ut/op_host/nn_op_host_ut --gtest_filter='Onnx*PluginTest.*'
```

构建上有一处容易卡住：部分插件（如 `max_pool`）在迁移前就调用了 `rtGetSocVersion`
（`IsRegBasePlatform()`），官方用例把该插件带进 UT 可执行文件后会缺 `libruntime.so`
而链接失败。该符号只由 runtime 提供，需在 `tests/ut/op_host/CMakeLists.txt` 的链接库里
补一项 `runtime`。这不是迁移引入的，但不补就跑不起来 UT。

每个算子的用例覆盖任务书 3.5 第 3 条要求的四类场景：

| 场景 | 判据 |
| --- | --- |
| 属性齐全的正常解析 | `SetAttr` 结果与基线一致 |
| 属性缺失（使用默认值） | 返回 **SUCCESS** |
| 非法 attribute JSON | 返回 **FAILED** |
| 边界/校验场景 | 如 `fixed_shift_value` 越界回退默认值、`pads` 奇数长度返回 FAILED |

通用规则：属性名匹配但 `type` 值不符时按缺失处理（`Ignore*WithMismatchedType` 系列覆盖）。

**UT 收集路径是个陷阱**：cmake 有两条互不相同的 framework UT 收集路径 ——
`add_all_modules_sources` → `add_all_ut_sources` 的 GLOB 路径**无条件收集**；
而 `add_op_subdirectory` 的条件路径带 `if(EXISTS "${OP_DIR}/${SUB_DIR}/CMakeLists.txt")`，
要求 `<算子>/tests/CMakeLists.txt`、`tests/ut/CMakeLists.txt`、`tests/ut/framework/CMakeLists.txt`
**三层齐备**，缺中间层则叶子不可达、该算子的用例被**静默跳过**（不报错、不提示）。
走条件路径的算子必须补齐中间层样板。

### 5.3.2 om 转换精度验证（任务书 3.2 第 4 条）

用任务书提供的用例包，迁移前、迁移后各执行一轮：

```bash
bash test_cases/verify_om/two_round_compare.sh --pre <迁移前.so> --post <迁移后.so> --gen <生成脚本>
```

脚本自动将指定插件经 `ASCEND_CUSTOM_OPP_PATH` 注入 ATC，并在日志中留证"生效的是本轮插件"。
覆盖迁移范围全部 74 个插件文件（其中 39 个脚本的模型 onnxruntime 不可执行 —— 含全部 17 个
NPU 自定义算子，golden 由脚本内 numpy 参考实现产出）。

**无法做逐输出比对的用例**，判据分两层而不是略过：

| 层 | 判据 |
| --- | --- |
| 1 | 两轮用例集合逐项相同（成败行为一致） |
| 2 | 两轮**报错文本逐字相同** —— 归一化运行期标识符（PID / tid / 时间戳 / session_id / graph_id / TaskID / 地址 / 轮次路径）后比对，并逐条声明归一化了什么 |

第 2 层排除了"迁移后在同一个用例上以不同原因失败"这种会被第 1 层漏掉的情形。
对动态 shape AI CPU 算子（pyACL 对 size 0 输出报 `malloc(0)`、取不到输出）另备兜底判据：
`atc --mode=1 --om=<f> --json=<j>` 反解析 om 后做**图级逐字段比对**。

> **用例包有一处会导致用例被静默跳过，须先处理再采信结果**：比对脚本按固定名 `model.om`
> 加载 ATC 产物，而 ATC 对含 AI CPU 算子的模型产出的是带平台后缀的 `model_<平台>.om`。
> 两轮都取不到产物 → 都不落输出文件 → 比对阶段既不计 PASS/FAIL 也无任何提示。
> 须按 `model*.om` 的实际名解析后重跑这批用例。这类缺口只能靠 5.4 的分母对账发现 ——
> 日志本身看起来是闭合的。

### 5.3.3 平台不可复现项的判定方法

遇到两轮不一致时**不得直接判为迁移回归**，须补三组对照：

| 对照 | 作用 |
| --- | --- |
| `--pre A.so --post A.so`（迁移前自比） | 若同一个 `.so` 自比也不一致，则与迁移无关 |
| `--pre B.so --post B.so`（迁移后自比） | 同上，双向确认 |
| 两轮均加 `--deterministic=1` 重测 | 排除非确定性算子调度 |

### 5.3.4 解耦本身的判据（符号级）

用插件 `.so` 未定义符号的**集合差**（`nm -D -u`，两侧都列名），而不是总数相减 ——
净值会把"消失 N + 新增 M"压成一个数，既看不出消失的是什么，也看不出新增里有没有意外依赖。

四类必须归零：`ge::onnx::*`、protobuf（`ascend_private::protobuf::*`）、
absl（protobuf 带入的 `lts_ascend_private`）、旧注册 API
`OpRegistrationData::ParseParamsFn`。

> 其中**旧注册 API 本身消失**是最硬的一条：它证明二进制层面已不再引用那条以 protobuf
> 为签名的注册入口，比"体积变小"或"源码里 grep 不到"都更有力。
> 新增符号应逐条对应新实现的 API 表面，不应出现意料之外的依赖。

### 5.3.5 依赖闭合性

为回答"被验证的 `.so` 对应的就是提交的内容"：从插件 target 的 `.d` 依赖文件取出全部实际
编译输入，逐个归类为「已跟踪且未改动 / 本次提交 / **未跟踪且未提交**」，第三类必须为空。
否则 `.so` 里含有不会随 PR 进入仓库的代码，验证结论不成立。

## 5.4 可测性与定位手段

| 手段 | 用途 |
| --- | --- |
| `ASCEND_CUSTOM_OPP_PATH` 注入 + ATC 日志留证 | 证明"生效的是本轮插件"，而不是系统自带插件 |
| `atc --mode=1 --om=<f> --json=<j>` | om 反解析为 JSON，做图级逐字段比对 |
| `nm -D -u` 集合差 | 符号级解耦判据（5.3.4） |
| `-Wall -Wextra` 构建告警 | 失效判据的机械筛查（3.2.5） |

**验证方法上的两条纪律**，比具体数字更重要：

1. **分母要拿外部锚点对账。** 两轮比对与 UT 的日志都只报"做了多少"，不报"少做了多少" ——
   只看"全绿"会漏掉静默跳过（5.3.1 的 UT 收集路径就是一例）。UT 条数应与源码里全部
   `TEST/TEST_F` 的普查数对账，且**不按套名预筛**（按 `--gtest_filter` 预筛会让不匹配的用例
   同时从源码集合和运行集合消失，集合差恒为 0、检查"通过"）；om 用例总数应按用例包的实际
   目录结构枚举后逐项归属，合计必须等于总数。
2. **不会失败的检查没有价值。** 每个检查都应做反证（把数据改成错的，确认它确实报错）；
   做不到的要如实标注"未做专门反证"，不并入"全部做过"的说法。

---

# 6 交付与合入

## 6.1 交付件（任务书 4 节）

| 序号 | 交付件 | 说明 |
| --- | --- | --- |
| 1 | 迁移后代码（PR） | 覆盖 75 个算子条目 / 74 个文件 / 76 处注册点 |
| 2 | UT 用例及执行结果 | 每个迁移插件文件对应更新用例，附执行日志 |
| 3 | om 转换精度验证记录 | 验证模型、ATC 转换命令、精度比对结果（含迁移前后一致性说明） |
| 4 | 单算子迁移自查表 | 按任务书 3.5 节自查清单逐项填写 |
| 5 | 性能自验证报告及日志 | 不涉及（任务书 3.3） |
| 6 | 内存自验证报告及日志 | 不涉及（任务书 3.4） |

按模板形式提交，并同步提交到开发者私仓目录 `task_submission`（任务书 4 节）。

## 6.2 合入计划（任务书 5 节）

1. 目标目录为各迁移插件文件所在目录（如 `activation/hard_sigmoid/framework/`、
   `common/src/framework/`）及对应 UT 目录；
2. **提交按插件文件/算子分组**，便于评审与回溯（任务书 5 节第 4 条）；
3. 不包含明显冗余内容 —— 不出现"文件个数明显不符合正常数量"或"多出无用的代码、测试和
   编译文件"（任务书 5 节典型反例 1、2）；
4. PR 描述与提交信息**不携带任何个人敏感信息**（任务书 5 节第 3 条）；
5. 遵循仓内贡献指南与 C++ 编码规范（任务书 7 节第 1 条）。

## 6.3 仓内门禁的已知约束

提交前需注意：仓内 `Check_Example` 门禁（`scripts/ci/check_examples_ut.py`）要求
**PR 涉及的每个算子目录**下 `examples/` 存在且含源码、`tests/ut` 存在且含测试源码。
由于 onnx 插件就位于算子目录内，改插件会把该算子拉入门禁检查范围 ——
而仓内有部分算子在基线上即缺 `examples/`，需按门禁口径补齐
（def 仅声明 `ascend950`/`ascend350` 的新架构算子，示例放 `examples/arch35/` 即满足）。
此类改动与解耦无关，应在代码 PR 的描述中单独声明，便于评审判断是否要求拆分提交。

# 7 参考资料

1. 迁移适配 issue（含关联 PR 示例）：https://gitcode.com/cann/ops-nn/issues/5115
2. onnx 算子开发说明：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/advanced_programming/ai_framework_adaptation/onnx_framework/call_examples.md
3. 生态算子开源精度标准（混合容差标准）：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. GE 标量 float `%g` 精度损失（本设计约束 1 的依据）：https://gitcode.com/cann/ge/issues/656
