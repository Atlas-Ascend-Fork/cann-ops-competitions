# ops-nn 算子 ONNX 插件 protobuf 解耦设计

## 需求背景（required）

### 需求来源

本设计对应「10 月社区任务 — ops-nn 算子 ONNX 适配」，依据任务书的 protobuf 解耦要求，对 **74 个插件文件 / 75 个"文件×算子"条目**进行迁移。范围为任务书清单所列全部插件：各算子目录下 `framework/*_onnx_plugin.cpp`、`common/src/framework/*`，以及 `experimental/` 下的插件条目。

### 背景介绍

迁移前，ONNX 插件通过 `ParseParamsFn` 注册参数解析函数，入参为 protobuf `Message`（`ge::onnx::NodeProto`），从该对象读取节点属性、输入输出名称与数量、节点名称，因此插件直接依赖 `ge_onnx.pb.h` 与 protobuf 对象类型。

CANN 9.1 起插件接口不再向开发者暴露这些 protobuf 类型，迁移前的插件在 ATC 阶段无法完成注册/参数解析（表现为算子未注册被跳过或图解析失败）。本次改用 `ParseParamsByOperatorFn`，由 `ge::Operator` 承载源节点信息，通过 `GetAttr` 读取属性 JSON 与输入输出名称列表，完成到目标算子的参数映射，实现等价重构。

## 需求分析（required）

### 需求描述

解除插件参数解析对 protobuf 对象与生成头文件的直接依赖，**保持原有算子映射、属性默认值、输入输出处理与子图构建行为不变**。

本任务属 Host 侧 framework 适配：不涉及 kernel、tiling、aclnn 接口与算子原型开发；`ParseOpToGraphFn` 与动态输入/输出注册保持原实现，不做无关重构。

### 需求拆解

| 改造项 | 设计要求 |
| --- | --- |
| 解析接口 | 源参数改为 `const ge::Operator&`，目标参数继续 `ge::Operator&`，函数名保持不变。 |
| 注册接口 | 替换为 `ParseParamsByOperatorFn`，保留算子名、ONNX 版本范围与 `ImplyType`。 |
| 属性解析 | 从源 Operator 的 `attribute` 属性读取 JSON 字符串，使用仓内 `nlohmann/json` 解析。 |
| 输入输出数量 | **优先读取 `input` / `output` 名称列表的长度**（ONNX 节点声明个数），列表不存在时回退描述符计数。 |
| 节点名称 | 使用 `GetOpName(op_src)` 写入目标 `name` 属性。 |
| 公共能力 | JSON 解析、浮点转换、输入输出计数由公共工具提供（`common/inc/framework/nn_onnx_plugin_util.h`）。 |
| 异常处理 | 捕获 JSON 解析/取值转换异常，记录日志并返回 `FAILED`。 |
| 失效判据 | 按被调用工具函数的真实返回约定判断失败分支，并补充非法取值 UT。 |
| 单元测试 | 直接构造 `ge::Operator`（不构造 protobuf 对象），验证属性映射、默认值、异常与边界。 |
| 精度验证 | 迁移前后插件对相同模型/输入的两轮 om 输出完全一致；golden 按任务包混合容差独立校验。 |

## 详细设计（required）

### 1. 处理流程与接口

GE 将 ONNX 节点转换为源 Operator，调用插件的参数解析函数。插件读取源信息、设置目标属性与描述符，随后由原有算子映射或子图构建流程继续处理。

函数保留原名称，仅替换源参数类型：

```cpp
domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest);
```

| 信息 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 节点属性 | `node->attribute()` | `op_src.GetAttr("attribute")` 取 JSON，由公共工具解析 |
| 输入数量 | `node->input_size()` | **优先读 `input` 名称列表长度**，无该属性时回退描述符计数 |
| 输出数量 | `node->output_size()` | **优先读 `output` 名称列表长度**，无该属性时回退描述符计数 |
| 节点名称 | `node->name()` | `GetOpName(op_src)` |
| 参数注册 | `ParseParamsFn` | `ParseParamsByOperatorFn` |

### 2. JSON 属性解析

```json
{ "attribute": [ { "name": "alpha", "type": 1, "f": "0.25" } ] }
```

| 属性类型 | type 值 | 取值键 | 读取方式 |
| --- | --- | --- | --- |
| FLOAT | 1 | `f` | **字符串**转 float（仓内 `StrToFloat`）；键缺省时取 0.0 |
| INT | 2 | `i` | 按整数读取；键缺省时取 0 |
| STRING | 3 | `s` | 按字符串读取；键缺省时取空串 |
| TENSOR | 4 | `t` | 按原 dtype / dims / 数据载体构造 `ge::Tensor` |
| FLOATS | 6 | `floats` | 按浮点数组读取 |
| INTS | 7 | `ints` | 按整数数组读取 |

「属性整体缺失」与「属性存在但取值键缺省」分别处理：如 HardSigmoid 无 `alpha` 属性时用原默认值 0.2；存在 FLOAT 类型 `alpha` 但省略 `f` 键时取 0.0（**不回落算子默认值**）。类型不匹配、未知属性、重复同名属性的处理顺序保持迁移前分支语义（重复取后者）。

### 3. 浮点属性处理

标量浮点由 GE 序列化为字符串承载，统一经仓内 `StrToFloat` 转换：空串/非法/越界返回失败并按迁移前语义处理；`f` 键缺省保持 proto3 零值语义。业务侧约束（取值范围、枚举等）仍由各插件自行管理。

### 4. 默认值、异常与校验

保留原属性默认值、目标属性类型、列表长度、维数、pads 重排（Conv 类按 `[h_begin, h_end, w_begin, w_end]`）、枚举与范围检查。解析或取值转换异常记录日志并返回 `FAILED`，正常分支保持原返回状态。

**失效判据**按任务书 §2.1-7 处理：对「按工具函数真实返回约定判断」的分支（如 dtype 映射未命中实际返回 `ge::DT_UNDEFINED`），迁移时按真实约定修正失败分支，并补充"非法取值应返回 FAILED"一类 UT。

### 5. 输入输出与名称处理（本任务的关键点）

**问题**：迁移前用 protobuf `node->input_size()/output_size()` 取"ONNX 节点**声明的**输入/输出个数"；新接口下 `ge::Operator::GetInputsSize()/GetOutputsSize()` 在 `ParseParams` 阶段（描述符尚未挂接）**恒返回 0**，二者不等价。

**方案**：GE 会把 ONNX 节点的输入/输出**名称列表**挂在源 Operator 的 `input` / `output` 属性上，其**长度即声明个数**；仅当属性不存在时才回退描述符计数：

```cpp
inline int GetOnnxInputCount(const ge::Operator& source) {
    std::vector<ge::AscendString> names;
    if (source.GetAttr("input", names) == ge::GRAPH_SUCCESS) {
        return static_cast<int>(names.size());
    }
    return static_cast<int>(source.GetInputsSize());
}
// GetOnnxOutputCount 同法读 "output"
```

空字符串代表可选端口占位，**保留位置与列表长度**。动态端口名称、注册顺序、`original_type` 与描述符格式更新沿用原实现；涉及格式更新的场景需为目标 Operator 注册相应端口。

**落实范围**：全仓逐一核对后，共 **14 个文件 19 处**在 `ParseParams` 阶段把 `op_src.GetInputsSize()/GetOutputsSize()` 当作声明个数使用（用途为 `DynamicInputRegister`、`SetAttr("input_num"/"num_output")`、合法性校验），全部改用上述 helper。其中 `max_pool`、`layer_normalization` 两处若不修正会造成**"迁移前能转、迁移后 ATC 失败"**（`EZ9999: The output of Indices is not support ... [FUNC:ParseParamsMaxPool]`）。

> 备注：任务书用例包 §五-2 的表述（"`op_src.GetInputsSize()` 返回 0，与迁移前空 NodeProto 的 `input_size()=0` 行为一致"）**只在 UT 夹具下成立**，在真实 ATC 场景不成立，建议按本节口径同步修订。

### 6. Tensor 属性处理

ConstantOfShape 等插件保留原 dtype、dims、元素数量、字节长度与 `TensorDesc` 构造方式，分别覆盖标量、空载体与非空 payload。实际 GE JSON 中的 `raw_data` 按 **C 转义格式**还原字节（不能按 base64 处理）；typed 数据按对应 dtype 读取，不通过填零或改 dtype 改变原 Tensor 语义。

### 7. 文件组织与依赖

| 位置 | 职责 |
| --- | --- |
| 各算子 `framework/*_onnx_plugin.cpp`、`common/src/framework/*` | 参数接口、属性读取与注册方式迁移 |
| `common/inc/framework/nn_onnx_plugin_util.h`（新增） | 输入输出计数（`GetOnnxInputCount/GetOnnxOutputCount`） |
| 各算子及 common 的 `tests/ut/framework/` | 构造 Operator，验证对应插件的解析行为 |
| framework UT 构建规则 | 各算子目录与 common 目录的 `tests/ut/framework/CMakeLists.txt`（通用 GLOB 形式） |

使用 `graph/operator.h`、`register/register.h`、仓内公共工具头与 `nlohmann/json.hpp`，移除对 `ge_onnx.pb.h` 的直接依赖。保留原 `OriginOpType`、版本分支与 `ImplyType`，逐一覆盖同文件多处注册。

### 支持硬件与约束

按任务书要求选择 Atlas A2 或 Ascend 950（本设计以 **Ascend910B4 / A2** 实施自验），CANN 版本 > 9.0.0（实测 9.1.0），迁移前后使用同一环境。不扩展算子 dtype、shape、广播能力或 ONNX 版本支持范围；不新增分核、UB 分块与 tiling 设计。

## 可维可测分析

### 单元测试设计

直接构造 `ge::Operator`，用 `SetAttr` 设置属性 JSON 与输入输出名称列表，调用迁移后的解析函数，**不构造 protobuf 对象**。

| 场景 | 验证内容 |
| --- | --- |
| 正常解析 | 目标属性名称、类型、数值、节点名与端口数量 |
| 属性缺失 | 默认值、必填失败分支、proto3 零值省略 |
| 异常输入 | 非法 JSON、载体类型错误、浮点转换失败、业务校验 |
| 边界情况 | 空列表、负值、重复属性、可选输入占位、端口数量 |
| Tensor 载体 | dtype、shape、空数据、正负 payload 与字节还原 |
| 特殊分支 | 失效 dtype 判据、多版本注册、动态输入、格式更新 |

**实测**：开启 `OP_HOST_UT` 构建后运行 `--gtest_filter='Onnx*PluginTest.*'` → **760 / 760 通过**（87 suite）= 任务包 74 个 UT 文件 **731 条全部覆盖** + 仓内既有 29 条；`git diff --check` 干净。

### OM 验证方法

使用任务包生成模型、输入与 golden；分别加载迁移前/迁移后插件（`ASCEND_CUSTOM_OPP_PATH` 注入并留存"本轮插件生效"证据），用同一 `atc --framework=5 --soc_version=Ascend910B4` 转换并在设备推理，比较输出的 dtype、shape 与数值。

**实测**（74 个生成脚本）：**117 个用例 / 325 条逐位比对 → 286 条逐位一致**；另做了**产物存在性对称检查**（逐用例比较两侧是否都产出 `model.om`）→ `pre_om == post_om == 121`，**零不对称**，即不存在"一侧能转、一侧转不了"。

> 方法学备注：主判据的逐位比对表**只在两侧都有输出时记录**，"一侧能转一侧转不了"的用例不会进表。因此我们额外做了上述对称性检查——此前正是靠它发现并修掉了 `max_pool` / `layer_normalization` 两处真回归。

### 精度、性能与内存标准

| 验收项目 | 设计标准 | 实测 |
| --- | --- | --- |
| 参数解析 | 保持原默认值、属性映射、输入输出与组图语义 | 74/74 插件编译通过，UT 760/760 |
| 两轮输出 | 相同模型与输入，迁移前后输出完全一致（不设阈值） | 325 条比对中 286 条逐位一致，39 条为算子固有不可复现（见下） |
| golden 精度 | 任务包混合容差 + 硬上限 + 匹配率，校验 dtype/shape；整数/布尔精确 | 117 用例中 107 通过 |
| 性能 | 不涉及（任务书 §3.3） | — |
| 内存 | 不涉及（任务书 §3.4） | — |

### 例外分析与证据（两类，均与迁移无关）

| 类别 | 条数 | 算子 | 证据 |
| --- | --- | --- | --- |
| A：算子固有不可复现 | 38 | `gru`(8)、`npu_gru`(30) | **同一份 `.so` 连续两次运行约 90% 元素不同**（`max_abs` 达 1.77）；ATC `--deterministic=1` **无效**；`pre` 自比 = `post` 自比 = 迁移前后，条数完全相同（30/8） |
| B：可配置消除的非确定性 | 1 | `instance_norm_v3` | 差异仅 `2^-9` / 98 个元素（占 393216 的 0.025%）；**`--deterministic=1` 下自比 0 条**，且迁移前后逐位一致 |

A 类属 Device 侧算子执行问题（同类先例：`ops-nn` #3728 ScatterNd 两次结果不一致，已确认问题），不在本任务的 kernel 范围内，建议按平台项登记并另报算子缺陷；B 类在确定性配置下满足主判据。

### 兼容性与维护

公共工具仅提供解析与计数能力，各插件自行管理默认值与业务校验；保留原动态注册与子图构建，避免公共工具改动影响其他算子。等价迁移依赖 GE 桥接保留原属性信息；桥接或测试材料存在差异时单独留证并请求维护者确认，**不在插件中猜测原值、不放宽精度、不修改算子计算行为**。

### 自查与证据

按任务书 §3.5-1 逐项自查（include 替换、签名/注册宏/属性解析/name 获取、默认值返回 SUCCESS、非法 JSON 返回 FAILED、失效判据修正、新增/更新用例、`git diff --check`、`build.sh --onnxplugin` 通过、UT 全通过、om 两轮比对），**逐项完成**；自验证材料见验收交付件（`task_submission` 七件套）。

## 实施安排与交付计划

| 阶段 | 状态 |
| --- | --- |
| 1. 设计评审（本 PR） | 进行中 |
| 2. 基线准备（迁移前插件 `.so`、任务包用例/模型） | ✅ 已完成 |
| 3. 分组迁移（74 个插件文件） | ✅ 已完成（`--onnxplugin` 构建 0 error） |
| 4. 全量自验（UT 760/760、om 两轮、对称性检查、例外留证） | ✅ 已完成 |
| 5. 交付合入（代码 PR + `task_submission` 七件套） | 代码 PR：https://gitcode.com/cann/ops-nn/merge_requests/12042 |

**关联**：代码 PR [ops-nn !12042](https://gitcode.com/cann/ops-nn/merge_requests/12042)；测试材料与 SDK 语义差异确认 [#6686](https://gitcode.com/cann/ops-nn/issues/6686)；framework UT 构建两处缺口 [#6717](https://gitcode.com/cann/ops-nn/issues/6717)。

## 附录：迁移范围

覆盖任务书清单的 74 个插件文件（75 个算子条目）。完整清单以任务书表格为准，代表性条目：

| 类别 | 示例 |
| --- | --- |
| `common/src/framework/` | `gemm`、`if`、`mean_variance_normalization`、`max_pool`、`layer_normalization`、`conv`、`conv_transpose`、`int8_*`、`npu_gru`、`npu_lstm`、`lstm`、`gru`、`lrn`、`lppool`、`maxunpool`、`constantofshape`、`scatter_nd`、`sparse_to_dense`、`softmax_cross_entropy_loss`、`moving_sum_with_sigmoid`、`deformable_conv2d` 等 |
| `activation/` | `celu`、`celu_v2`、`hard_sigmoid`、`log_softmax_v2`、`selu`、`shrink`、`softmax_v2`、`softmax_cross_entropy_with_logits`、`threshold`、`ge_glu_v2` |
| `conv/` | `deformable_conv2d`（NPUDeformableConv2d） |
| `index/` | `gather_v2`（Gather/Compress）、`gather_elements`、`gather_nd`、`non_zero`、`scatter`、`scatter_elements`、`unique_with_counts_and_sorting` |
| `norm/` | `add_rms_norm`、`batch_norm`、`group_norm_silu`、`instance_norm_v3`、`lp_norm_v2`、`rms_norm` |
| `pooling/` | `adaptive_avg_pool2d`、`adaptive_max_pool2d` |
| `quant/` | `ascend_anti_quant`、`ascend_dequant`、`ascend_quant`（含 `int8_quantize`）、`ascend_quant_v2`、`dynamic_quant`、`group_quant`、`quantize` |
| `rnn/` | `dynamic_rnn`（NPULstm）、`dynamic_rnnv2`（NPULstmCell） |
| `experimental/` | `norm/fused_add_rms_norm`（NPUFusedAddRmsNorm） |
