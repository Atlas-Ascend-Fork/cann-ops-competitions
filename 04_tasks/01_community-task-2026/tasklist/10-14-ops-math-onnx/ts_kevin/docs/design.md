# ops-math ONNX 插件 protobuf 解耦设计

# 需求背景（required）

## 需求来源

本设计依据 ops-math ONNX 插件 protobuf 解耦任务书整理。目标是让 ops-math 仓的 ONNX 参数解析插件从 protobuf 节点对象迁移到 GE `Operator` 接口。本文只描述 ops-math 的迁移范围、技术约束和验收方法。

## 背景介绍

现有插件通过 `ParseParamsFn` 接收 `google::protobuf::Message`，再访问 ONNX `NodeProto` 的属性。这样会让 Host 侧插件依赖 `ge_onnx.pb.h` 和 protobuf 运行库，也让 UT 必须构造 protobuf 对象。GE 提供 `ParseParamsByOperatorFn` 后，插件可以从源 `ge::Operator` 读取序列化的 `attribute` JSON，使用仓内 nlohmann/json 完成相同的属性映射。

任务范围是 **57 个插件源文件、124 个算子注册条目**。其中 `similar_opp_onnx_plugin.cpp` 聚合 63 个条目，`sequence_onnx_plugin.cpp` 聚合 6 个条目；`math/maximum` 与 `experimental/math/asinh` 下各有一个独立的 Max 插件。文件数与注册条目数不能混用。

# 需求分析（required）

## 需求描述

本任务是 ATC 模型转换阶段的 Host 侧等价迁移。解析后的目标算子属性、原有返回码、动态端口注册及 ONNX opset 注册关系应保持原行为。设备 Kernel、Tiling、算子原型及运行时计算路径不在本设计的迁移范围内。

| 代码区域 | 插件文件 | 算子条目 | 主要特点 |
| --- | ---: | ---: | --- |
| `common/src/framework/` | 14 | 81 | 单算子属性解析、sequence 和 similar_opp 聚合注册 |
| `conversion/*/framework/` | 14 | 14 | shape、轴、列表属性与多 opset 分支 |
| `math/*/framework/` | 25 | 25 | dtype、归约、输入计数与必填属性 |
| `experimental/math/asinh/framework/` | 1 | 1 | 独立 Max 插件 |
| `random/*/framework/` | 3 | 3 | shape、浮点参数、随机种子 |
| **合计** | **57** | **124** | 与任务范围表对应 |

## 需求拆解

1. 将解析函数的源入参改为 `const ge::Operator&`，注册接口改为 `.ParseParamsByOperatorFn(...)`。
2. 读取 `op_src.GetAttr("attribute", ...)` 得到 JSON，按每个插件原有的属性名、类型、顺序和缺省规则处理。
3. 将节点名称读取改为 `GetOpName(op_src)`；需要输入或输出数量的插件，优先读取源算子的名称列表，并保留空占位。
4. 去除插件对 protobuf 生成头和运行库的直接依赖，优先复用 `math_onnx_plugin_util.h`。
5. 保留 `OriginOpType`、`DynamicInputRegister`、`DynamicOutputRegister` 和未获单独授权的 `ParseOpToGraphFn` 逻辑。
6. 逐文件配套 framework UT，并以迁移前、迁移后两轮模型转换和推理证明行为一致。

# 详细设计（required）

## 算子分析

### 解析接口和数据流

```mermaid
flowchart LR
    A[ONNX 节点] --> B[GE 源 Operator]
    B --> C[attribute JSON 与 input/output 名称列表]
    C --> D[ParseParamsByOperatorFn]
    D --> E[目标 Operator 属性及动态端口]
    E --> F[后续图编译]
```

| 项目 | 迁移后设计 |
| --- | --- |
| 回调签名 | `domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)` |
| 插件注册 | 保持原 `REGISTER_CUSTOM_OP` 和 opset 集合，仅替换解析回调注册入口 |
| 属性来源 | `op_src.GetAttr("attribute", ge::AscendString&)` |
| 属性解析 | 仓内 nlohmann/json；异常记录日志并返回 `FAILED` |
| 节点名 | 在原逻辑需要写入时调用 `GetOpName(op_src)` |
| 端口数量 | 优先取 `input`/`output` 名称列表长度；缺失时按原实现所需回退端口数 |
| 依赖 | `graph/operator.h`、`register/register.h`、`nlohmann/json.hpp` 和 ops-math 公共工具 |

属性数组位于 JSON 顶层的 `attribute` 键。按 ONNX 属性类型处理 `i`（INT=2）、`f`（FLOAT=1，GE JSON 中为字符串）、`s`（STRING=3）、`floats`（FLOATS=6）及 `ints`（INTS=7）。当属性本身存在但值键因 proto3 零值省略时，`i`、`f`、`s` 分别按 0、0.0、空串处理，不能误用“整个属性缺失”时的算子默认值。

### 语义保持规则

插件不能被强制套入统一的属性解析器。逐文件保留以下差异：

| 场景 | 设计要求 |
| --- | --- |
| 属性不存在 | 沿用该插件的默认值或必填失败分支；不能把所有插件统一判为成功 |
| `type` 不符 | 大多数插件跳过；Pad 的 pads、OneHot 的 axis 等按原路径处理 |
| 重复属性 | Cast、Concat 等取首个；多数标量取末个；部分列表累加或先清空再写 |
| JSON 非法 | 真正读取 JSON 的路径捕获异常并返回 `FAILED`；不读取属性的路径保持其原返回值 |
| 空列表与负值 | 原实现允许透传时继续透传；各插件原有校验继续生效 |
| 多版本注册 | Clip、Pad、Slice、Min、TopK、Squeeze/Unsqueeze 与归约系列保留独立回调和版本映射 |
| 返回码 | `SUCCESS`、`FAILED`、`PARAM_INVALID` 按原路径逐一核对，不能互相替换 |

Cast 有一处明确的失效判据：`GetOmDtypeFromOnnxDtype()` 的未命中值为 `ge::DT_UNDEFINED`，原先比较 `-1` 不会命中。迁移时改为比较 `static_cast<int32_t>(ge::DT_UNDEFINED)`，使不支持的 dtype 在解析阶段返回 `FAILED`；这项修正须单独记录，并由对应 UT 守护。

### 公共工具与输入输出计数

复用 `common/inc/framework/math_onnx_plugin_util.h` 中的名称、dtype、格式转换和张量构造能力。只有多个插件确实共享的 JSON 原子读取规则才考虑放入 ops-math 前缀的公共头，不能让公共工具隐式覆盖某个插件的首个、末个或累加策略。

ATC 提供的源 `Operator` 可能尚未连接端口描述，因此直接依赖 `GetInputsSize()` 会得到 0。需要模拟 ONNX 节点端口数时，优先读取 `input`/`output` 属性的名称数组并取其长度；名称为空的可选输入仍占位置。缺少该属性时再回退端口接口，以兼容 UT 中直接构造的 `Operator`。

## 算子实现

按目录和行为拆分实施与评审。下表列出所有 57 个文件的覆盖位置；同名 Max 是两个独立文件。

| 组别 | 文件范围 | 重点检查 |
| --- | --- | --- |
| common 单算子（12） | concat_from_sequence、deformable_offsets、einsum、eyelike、phony_concat、phony_split、random_normal_like、random_uniform_like、reduceL1、reduceL2、repeat_with_pad、split_to_sequence | 必填属性、列表重复、动态端口、浮点零值 |
| common 聚合（2） | sequence、similar_opp | 6/63 个注册条目和各自 opset；只写元信息的路径不额外解析 JSON |
| conversion（14） | clip、concat、depth_to_space、flatten、npu_slice、pad、reshape、slice、space_to_depth、split、squeeze、transpose、trilu、unsqueeze | 多版本回调、轴和 shape 列表、`PARAM_INVALID` 分支 |
| math（25） | addcmul、argmax、argmin、xor、cast、cumsum、isinf、max、mean、min、mod、onehot、topk、reduce_log_sum_exp、reduce_max、reduce_mean、reduce_min、reduce_prod、reduce_log_sum、reduce_sum、reduce_sum_square、bitshift、sign_bits_pack、sign_bits_unpack、sum | dtype 映射、归约版本、输入计数与校验边界 |
| random（3） | random_uniform、random_normal、multinomial | 必填 shape、浮点零值、种子和重复属性 |
| experimental（1） | `experimental/math/asinh/framework/max_onnx_plugin.cpp` | 与 math/maximum 下的 Max 分别迁移、分别验证 |

实施顺序为公共工具核对，随后 common、conversion、math、random 与 experimental。每批随源码同步放入对应的 `tests/ut/framework/` 用例，并检查 `OriginOpType`、动态端口与拆图注册没有意外变化。

### 独立缺陷修复的隔离

配套设计材料另行记录了获授权的 Multinomial 拆图属性保留修复：旧拆图会覆盖解析得到的 `sample_size`、`dtype`、`seed`。这项修复不属于 protobuf 解耦的等价变更。验证它时应保留原始迁移前基线，另建“仅应用同一缺陷修复”的迁移前基线，再与迁移后版本比较；不得把原错误输出与修复后输出称为等价。

## 支持硬件与约束

本设计为 Host 侧插件适配，不新增芯片专用计算路径。按任务书要求，在 Atlas A2 或 Ascend 950 中任选一个平台完成验证，迁移前后使用相同硬件、CANN 版本和 ATC 参数；CANN 版本需大于 9.0.0。JSON 库使用仓内依赖，不引入新的第三方版本。性能和内存无专项验收指标。

# 可维可测分析

## 单元测试规划

57 个插件文件分别对应 framework UT 文件，输入使用 `ge::Operator` 的 `SetAttr` 构造，不创建 protobuf 对象。每个用例至少核对返回码、目标属性是否存在以及值、dtype 或 shape。按插件的真实解析路径覆盖正常、缺省、非法 JSON、类型不符、值键省略、重复属性、边界和多 opset 分支。

| 组别 | UT 文件数 | 覆盖用例数 |
| --- | ---: | ---: |
| common 单算子 | 12 | 128 |
| common 聚合 | 2 | 15 |
| conversion | 14 | 165 |
| math | 25 | 276 |
| random | 3 | 31 |
| experimental Max | 1 | 9 |
| **合计** | **57** | **624** |

任务书第 3.5 节写的是 **612** 个用例，逐文件覆盖矩阵合计为 **624**。实施验收以实际提交的 57 个 UT 文件及执行日志为准，报告应注明所用用例包版本及具体用例数。构建插件使用 `bash build.sh --onnxplugin -j8`；开启 `OP_HOST_UT` 后执行 `--gtest_filter='Onnx*PluginTest.*'`，并检查 `git diff --check`。

## 模型验证规划

配套 **56 个模型生成脚本**覆盖 57 个插件文件，两份 Max 共享模型脚本。每个脚本固定 ONNX 模型、输入与 ATC 参数，分别加载迁移前和迁移后插件转换 om，再用相同输入推理。日志需保留插件实际加载路径，避免误把系统自带插件的结果当成目标版本。

1. 两轮都成功的确定性模型：核对输出数量、dtype、shape 和逐数值完全一致；主判据不设置浮点容差。
2. 有 golden 的模型：额外与 onnxruntime 或脚本内 numpy 参考比对。按任务书的混合容差标准：FP32 `rtol=2^-10`、`atol=2^-16`，匹配率至少 0.99，并检查逐元素硬上限；整数和布尔值精确相等。
3. `golden=false` 的随机模型：按用例元数据记录行为一致性并核对 dtype/shape；不能伪称已有 golden 数值对齐。
4. 预期转换失败或平台不支持的模型：保存两轮 ATC 原始错误，按同一平台、同一配置、同一失败类型判定，不计入成功推理样本。
5. 对 Multinomial 独立修复使用单独的同修复基线；在报告中同时保留原基线失败证据与额外变体测试。

配套设计材料记录了后续验证结果：全仓 framework UT 为 667/667，原任务 UT 为 624/624，原 56 个脚本展开的 61 个模型及变体为 61/61，另有 2 个 Multinomial 真机变体。这些是来源方案的既有记录；本次设计文档 PR 仅提交文档，不将其表述为由该 PR 新运行的测试，也不把授权修复与纯解耦结果混为一谈。

## 验收标准与风险

| 项目 | 验收口径 |
| --- | --- |
| 范围 | 57 个插件文件、124 个注册条目逐项映射，无漏迁或额外无关改动 |
| 依赖与构建 | 插件正式构建通过，迁移后插件不依赖 `ge::onnx` protobuf 类型 |
| UT | 当前用例包的 `Onnx*PluginTest.*` 全量通过，57 个文件均有对应结果 |
| om 比对 | 迁移前后同模型、同输入、同环境；逐例记录转换、加载、推理及 golden 判定 |
| 可追溯性 | 记录基线 commit、插件哈希、ATC 命令、设备与 CANN 版本；独立修复单列 |

主要风险是 JSON 表示与 protobuf 零值语义差异、重复属性策略被统一逻辑覆盖、ATC 插件加载错误、多 opset 回调遗漏，以及旧接口在新 CANN 下缺少 protobuf 符号。对应控制是逐文件 UT、注册映射核对、插件加载日志留证、两轮基线隔离，以及只在旧基线侧使用任务包说明的兼容处理。

# 交付与审查

设计文档放在社区任务目录下个人提交位置的 `docs/design.md`。源码迁移和验证证据应在后续 ops-math 代码 PR 与 `task_submission` 中分别交付。本次设计文档可先审查范围、行为契约、测试覆盖和独立修复边界；正式代码合入仍以对应源码、UT 与 om 证据为准。
