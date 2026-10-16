# ops-math 算子 ONNX 适配设计文档

# 需求背景（required）

## 需求来源

- 社区任务：[10月社区任务-ops-math算子onnx适配](https://www.hiascend.com/activities/task-center/details/18e1d3beae784812b2562460542ad6b3)。
- 设计依据：《ops-math算子onnx插件protobuf解耦任务书》及配套测试用例。
- 接口迁移需求：[ops-math Issue #2892](https://gitcode.com/cann/ops-math/issues/2892)。
- 开发者：`nudt_wsy`（小帅）。

## 背景介绍

ONNX 插件负责把模型节点的属性和端口信息转换为 GE 算子信息。现有部分插件通过 `ParseParamsFn` 接收 protobuf 节点，依赖 `ge_onnx.pb.h` 和生成代码，单元测试也需要构造 protobuf 对象。

本方案拟改用 `ParseParamsByOperatorFn`。插件从源 `ge::Operator` 获取节点信息，以 nlohmann/json 解析属性，在保持原有解析语义的前提下消除 protobuf 依赖。

例如，Flatten 的 `axis=2` 应继续写入目标算子的 `axis=2`；缺少该属性时仍取默认值 `1`。迁移仅改变属性读取方式。

# 需求分析（required）

## 需求描述

按任务书迁移 57 个插件文件、124 个算子条目。同一算子的多组 ONNX opset 注册合并统计为一个条目。

| 源码目录 | 插件文件数 | 算子条目数 |
| --- | ---: | ---: |
| `common/src/framework/` | 14 | 81 |
| `conversion/` | 14 | 14 |
| `experimental/math/asinh/` | 1 | 1 |
| `math/` | 25 | 25 |
| `random/` | 3 | 3 |
| 合计 | 57 | 124 |

逐文件范围以任务书迁移清单为准。若上游发生变化，先核对清单与注册项，再调整迁移范围。

## 需求拆解

1. 替换解析回调签名及注册入口，保留原有 ONNX 版本和 GE 算子映射。
2. 将 protobuf 属性遍历替换为 Operator 属性读取及 JSON 解析。
3. 保留默认值、返回码、动态端口和子图逻辑。Cast 的无效 dtype 判据按任务书单独修正。
4. 清理插件及 framework UT 的 protobuf 头文件、生成代码和链接依赖。
5. 使用 Operator 编写 framework UT，并规划迁移前后模型转换及推理比对。

# 详细设计（required）

## 算子分析

本任务是 Host 侧框架适配。各算子的数学公式、输入输出数据类型、形状及广播规则沿用原定义。设备侧 Kernel、Tiling 和计算接口不在改造范围内。

## 算子实现

### 总体流程

```mermaid
flowchart LR
    A[ONNX 节点] --> B[GE 前端生成源 Operator]
    B --> C[ParseParamsByOperatorFn]
    C --> D[读取属性 JSON 与端口信息]
    D --> E[写入目标 Operator]
    E --> F[原有子图回调或图编译流程]
```

### 解析接口与注册

解析函数保留原名称，统一采用以下签名：

```cpp
domi::Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest);
```

每组注册将 `.ParseParamsFn(原函数)` 替换为 `.ParseParamsByOperatorFn(原函数)`。保留 `OriginOpType`、opset 范围、目标类型、`FrameworkType`、`ImplyType` 和 `ParseOpToGraphFn` 绑定。

| 原信息来源 | 迁移方案 |
| --- | --- |
| `NodeProto::attribute()` | `op_src.GetAttr("attribute", ...)` 获取 JSON，再按原顺序遍历 |
| `NodeProto::name()` | 在原有写入位置使用 `GetOpName(op_src)` |
| 原始输入名称及数量 | 优先读取源 Operator 的 `input` 名称列表；缺失时使用端口信息 |
| 原始输出名称及数量 | 优先读取源 Operator 的 `output` 名称列表；缺失时使用端口信息 |
| protobuf 数据类型及张量辅助转换 | 优先复用 `math_onnx_plugin_util.h` 的对应能力 |

原始名称列表需保留空占位和顺序。Clip 等可选输入场景按输入位置判断 min、max 是否存在，避免仅凭有效端口数量改变含义。

### JSON 属性解析

源 Operator 的 `attribute` 字符串按以下结构解析：

```json
{"attribute":[{"name":"axis","type":2,"i":2}]}
```

| ONNX 属性类型 | `type` | 取值字段 | 字段省略时取值 |
| --- | ---: | --- | --- |
| FLOAT | 1 | `f`，字符串，如 `"0.25"` | `0.0` |
| INT | 2 | `i`，整数 | `0` |
| STRING | 3 | `s`，字符串 | 空字符串 |
| FLOATS | 6 | `floats`，数值数组 | 空数组 |
| INTS | 7 | `ints`，整数数组 | 空数组 |

属性未命中时保留算子默认值；属性已命中而值字段省略时使用表中的零值。例如浮点属性存在但没有 `f` 时取 `0.0`，与整个属性缺失时的算子默认值分开处理。

属性 `type` 不匹配时按任务书的缺失语义处理。保留原有属性遍历顺序、重复属性处理、整数宽度及列表顺序，不通过按名称去重改变解析结果。

`f` 先按字符串读取，再转换为 FP32。非法字符串和消费字段的类型错误应记录算子名及原因，并返回 `FAILED`。JSON 解析及字段转换置于异常处理范围内，避免异常越过插件回调。

不读取属性的分支保留原有处理流程。需要读取属性的分支在缺省场景保留原返回码；必需属性缺失时沿用原失败行为。

### 公共辅助能力

优先复用 `math_onnx_plugin_util.h`。多插件共用的 JSON 读取、浮点转换及原始端口读取能力，拟集中到 `common/inc/framework/math_onnx_json_util.h`，函数采用 `MathOnnx` 仓名前缀。

公共辅助函数只处理信息读取和异常转换。各插件继续决定属性默认值、取值校验、目标属性和返回码，避免通用工具掩盖算子差异。

### 特殊场景

| 场景 | 处理方案 |
| --- | --- |
| Cast 不支持的 dtype | 映射结果与 `static_cast<int32_t>(ge::DT_UNDEFINED)` 比较，返回 `FAILED` |
| Squeeze、Flatten 等缺省属性 | 保留各 opset 分支的默认值及目标属性名 |
| 动态输入、输出 | 保留 `DynamicInputRegister` 等注册方式、名称、数量和顺序 |
| Sequence 系列 | 保留端口及子图契约，分别核对六个注册条目 |
| 随机算子 | 保留 seed、dtype、shape 和子图参数的转换方式 |
| `ParseOpToGraphFn` | 保留子图主体及其消费的属性、Tensor 和端口契约 |
| 批量注册宏 | 修改宏内接口，逐项核对宏调用表，避免漏迁移注册项 |

### 构建依赖

目标插件移除 `ge_onnx.pb.h` 及间接 protobuf 引用。Shape、Bernoulli 等相邻插件如通过公共头引入同一依赖，应同步调整头文件依赖边界。

在 `cmake/func.cmake` 和 `cmake/ut.cmake` 中核对 ONNX 插件及 UT 目标，移除对应 protobuf 生成、头文件目录和链接项。JSON 使用仓内 `third_party` 版本。

通过源码扫描、插件构建和共享库依赖检查共同确认解耦，保留范围外目标仍需要的依赖。

### 修改位置规划

| 仓内位置 | 拟修改内容 |
| --- | --- |
| 上述 57 个 `*_onnx_plugin.cpp` | 回调、注册、属性解析及节点信息读取 |
| `common/inc/framework/` | 复用现有工具，补充最小 JSON 辅助能力 |
| Shape、Bernoulli 的 framework 插件 | 调整间接依赖头文件 |
| 对应 `tests/ut/framework/` 用例 | 使用 Operator 构造输入并验证迁移契约 |
| `cmake/func.cmake`、`cmake/ut.cmake` 及相关 UT CMake | 清理 protobuf 依赖，接入用例 |

## 支持硬件与约束

拟在 Atlas A2 上验证，CANN 版本大于 9.0.0。任务书也允许使用 Ascend 950；迁移前后必须在同一硬件、同一环境获取基线。

Host 侧适配不新增设备计算路径。分核、UB 分块、TilingKey 和性能优化设计不涉及。

# 可维可测分析

## 单元测试规划

开启 `OP_HOST_UT`，运行 `--gtest_filter='Onnx*PluginTest.*'`。每个目标插件使用 `ge::Operator` 构造测试输入，覆盖以下场景：

| 分类 | 主要用例 | 检查项 |
| --- | --- | --- |
| 正常解析 | 属性齐全，使用可区分值 | 返回码、目标属性值及类型 |
| 缺省 | 属性缺失、值字段省略 | 算子默认值与字段零值分别正确 |
| 非法 JSON | 语法错误、字段类型错误 | 返回 `FAILED`，保留错误日志 |
| 边界与校验 | 无效 dtype、type 不匹配、空列表、负值、空可选输入 | 校验与透传符合任务书及原解析语义 |

补充动态端口、多 opset 和子图参数检查。维护“插件文件—注册条目—UT—模型”的对应关系，核对 57 个文件及 124 个条目的覆盖。

## 模型验证规划

1. 固定迁移前源码基线、模型、输入和转换参数。两轮分别构建插件，通过 `ASCEND_CUSTOM_OPP_PATH` 加载，并保存实际加载路径。
2. 执行任务包模型生成脚本、ATC 转换和 OM 推理。OM 是 ATC 生成、供昇腾设备运行的离线模型。
3. 确定性用例的迁移前后输出逐数值相等，同时核对输出数量、dtype 和 shape。随机或非确定性用例增加迁移前重复运行参照，单列行为一致性证据，不混入逐值相等结论。
4. 有 golden 的用例另按任务包混合容差检查。FP32 使用 `rtol=2^-10`、`atol=2^-16`，匹配率至少 `0.99`，满足逐元素硬上限 `max(fixed_limit, 32*ULP)`；整数、布尔精确比较。golden 是任务包提供或生成的参考输出。
5. 平台不支持的用例保存两轮转换状态和原始错误，按任务书核对行为一致。Cast 的无效 dtype 按明确修正规则检查。

两轮保持相同精度配置。涉及原始 dtype 保留时统一设置 `must_keep_origin_dtype`，避免转换配置差异干扰比较。旧接口如需兼容库，仅用于迁移前基线轮。

## 验收标准

| 项目 | 计划验收条件 | 依据 |
| --- | --- | --- |
| 迁移范围 | 57 个插件文件、124 个条目及对应 UT 覆盖完整 | 任务书迁移清单 |
| 构建与依赖 | `bash build.sh --onnxplugin -j8` 成功；目标插件及 UT 无 protobuf 依赖 | 任务书 2.1、3.5 节 |
| 单元测试 | Operator 构造输入，framework UT 全部通过 | 任务书 3.5 节 |
| 模型精度 | 两轮主判据及 golden 辅助判据按用例记录；异常和随机行为单列说明 | 任务书 3.2 节 |
| 规范检查 | `git diff --check` 及仓库要求的格式、头文件和许可证检查通过 | 仓库贡献规范 |
| 性能、内存 | 不涉及，本任务不设对应验收指标 | 任务书 3.3、3.4 节 |

## 兼容性与风险

| 风险 | 控制方法 |
| --- | --- |
| 字段省略被误当作算子默认值 | 单独测试属性缺失与字段零值 |
| 空可选输入丢失位置 | 保留原始名称列表并测试仅有 min、仅有 max 的分支 |
| 多 opset 或宏注册漏迁移 | 对照完整注册清单检查接口和映射 |
| 测试误加载内置插件 | 两轮均记录实际插件加载路径 |
| 环境或随机性干扰比较 | 固定配置，重复基线运行，分开记录输出差异与历史行为 |

保留插件文件布局和原函数命名。将公共读取逻辑与算子属性规则分开维护，以配套 UT 约束后续修改。
