# ops-nn 算子 ONNX 适配设计文档

# 需求背景（required）

## 需求来源

- 社区任务：10月社区任务-ops-nn算子onnx适配，依据《ops-nn 算子 onnx 插件 protobuf 解耦任务书》及其迁移范围表。
- 目标代码仓库：[ops-nn](https://gitcode.com/cann/ops-nn)。
- 接口迁移参考实现：[ops-nn Issue #5115](https://gitcode.com/cann/ops-nn/issues/5115)（关联 PR：迁移 AdaCast onnx 插件解耦 protobuf）。
- 社区任务流程与注意事项：[cann discussions #39](https://gitcode.com/org/cann/discussions/39)。
- 开发者：`Irene-`。

## 背景介绍

ONNX 插件（`*_onnx_plugin.cpp`）位于各算子的 `framework/` 目录，在 ATC 模型转换阶段把 ONNX 节点的属性、输入输出名称与数量映射为目标 GE 算子的属性。

迁移前，本任务范围内的插件通过 `ParseParamsFn` 注册参数解析函数，源参数为 protobuf `Message`（实际是 `ge::onnx::NodeProto`），插件内 `dynamic_cast` 后遍历 `node->attribute()`，按 `attr.i()/f()/s()/ints()/floats()` 取值。由此带来三个问题：

1. 与 protobuf 生成代码强耦合：插件源码依赖 `onnx/proto/ge_onnx.pb.h`，proto 定义变动会波及全部插件；
2. 编译依赖重：插件编译需拉入 protobuf 头文件与生成代码，编译链路长；
3. UT 成本高：单元测试必须构造 protobuf 对象，用例编写与维护成本高。

GE 框架已提供以 `ge::Operator` 为源参数的新接口 `ParseParamsByOperatorFn`：节点属性由 GE 侧序列化为 JSON 字符串挂在源算子的 `"attribute"` 属性上，插件改用 `op_src.GetAttr("attribute", ...)` 取 JSON，用仓内 nlohmann/json 解析，从而彻底摆脱对 protobuf 生成代码与运行库的依赖。本任务即按此接口完成等价迁移。

### 迁移前基线（源码静态扫描）

| 现状项 | 数据 |
| --- | --- |
| 仓内 ONNX 插件文件 include `onnx_common.h`（该头内 include `ge_onnx.pb.h`） | 79 个 |
| 任务书迁移范围表覆盖 | 74 个插件文件 / 75 个算子条目 |
| 已采用新接口的仓内参考实现 | elu、leaky_relu、bernoulli、bounding_box_decode、AdaCast、matmul 等 |
| 公共工具 | `common/inc/framework/plugin_util.h`（已存在）+ `common/inc/framework/onnx_common.h`（protobuf 相关，91 行） |

`onnx_common.h` 中除 protobuf 依赖外，还承载三个与 protobuf 无关的工具函数（`Vec2Tensor`、`CreateScalar`、`ChangeFormatFromOnnx`）以及一个 protobuf 相关的 `GetEnableUncacheFromNode`，被大量插件使用——这是解耦时删除该头文件牵连面较大的原因。

# 需求分析（required）

## 需求描述

完成 75 个算子 onnx 插件条目（74 个插件文件）的 protobuf 解耦迁移。本任务属于 Host 侧框架等价重构：迁移前后插件的参数解析行为必须完全一致，不改变算子计算逻辑、不改变模型转换产物结构，仅替换解析函数入参形态、注册方式、属性取值手段与编译依赖。设备侧 Kernel、Tiling、aclnn 接口及算子原型定义不在改造范围内。

| 源码目录 | 插件文件数 | 说明 |
| --- | ---: | --- |
| `activation/` | 11 | 含连带处理 `gelu_v2` |
| `common/src/framework/` | 37 | 含连带处理 `lstmp` |
| `quant/` | 9 | 含连带处理 `ascend_anti_quant_v2` |
| `index/` | 9 | 含连带处理 `bucketize` |
| `norm/` | 6 | |
| `rnn/` | 2 | |
| `pooling/` | 2 | |
| `conv/` | 1 | |
| `experimental/norm/` | 1 | |
| 合计 | 78 | 任务书范围 74 个文件 + `onnx_common.h` 删除连带 4 个文件 |

`ParseOpToGraphFn`（子图拆图）与 `DynamicInputRegister`（动态输入注册）保持原实现，不进行与解耦无关的重构。

## 需求拆解

1. **解析函数签名替换**：`const Message* op_src` → `const ge::Operator& op_src`，随之删除 `dynamic_cast<const ge::onnx::NodeProto*>` 及判空分支（迁移后入参类型即编译期保证）；
2. **注册方式替换**：`.ParseParamsFn(...)` → `.ParseParamsByOperatorFn(...)`，函数名、`OriginOpType`、`ImplyType`、`ParseOpToGraphFn` 等其他注册项一律不动；
3. **属性解析方式替换**：`node->attribute()` 遍历 → `GetAttr("attribute", ...)` 取 JSON 后按 nlohmann/json 解析；解析时校验 `type` 与算子属性定义一致，并按 `f`（字符串承载的浮点标量）/ `i` / `s` / `ints` / `floats` 取值；
4. **移除 protobuf 依赖**：include 替换为 `plugin_util.h`、`graph/operator.h`、`nlohmann/json.hpp`、`register/register.h`，删除对 `onnx_common.h`（及 `ge_onnx.pb.h`、protobuf 运行库）的依赖；
5. **节点名称获取替换**：原 `node->name()` → `op_dest.SetAttr("name", GetOpName(op_src))`；
6. **语义保持**：属性缺失时保持原有默认值行为并返回 `SUCCESS`；非法 JSON 或取值异常时记录日志并返回 `FAILED`；不解析 attribute JSON 的插件豁免（实现与 UT 均不改）；迁移前永远不成立的失效判据按所调用工具函数的真实返回约定修正；不顺手重构无关代码；
7. **公共工具**：优先复用仓内已有公共工具；确需下沉公共能力时沿用仓内既定头文件与命名；
8. **验证**：按任务书完成 framework UT（`--gtest_filter='Onnx*PluginTest.*'` 全部通过）与迁移前后两轮 om 转换精度比对自验。

# 详细设计（required）

## 算子分析

### 解析接口对比

| 项 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 解析函数签名 | `Status ParseParamsXxx(const Message* op_src, ge::Operator& op_dest)` | `Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)` |
| 注册方式 | `.ParseParamsFn(...)` | `.ParseParamsByOperatorFn(...)` |
| 属性来源 | `ge::onnx::NodeProto::attribute()`（protobuf 对象） | `op_src.GetAttr("attribute", ge::AscendString&)`（JSON 字符串） |
| 属性取值 | `attr.i()/f()/s()/ints()/floats()` | `attr["i"]`(int64) / `attr["f"]`（字符串转 float）/ `attr["s"]` / `attr["ints"]` / `attr["floats"]` |
| 节点名称 | `node->name()` | `GetOpName(op_src)` |
| 输入/输出数量 | `node->input_size()` / `node->output_size()` | `GetNodeInputNum/GetNodeOutputNum(op_src)` |
| 头文件依赖 | `onnx_common.h`（→ `ge_onnx.pb.h`） | `plugin_util.h`、`graph/operator.h`、`nlohmann/json.hpp`、`register/register.h` |
| 编译依赖 | 需 protobuf 生成代码与运行库 | 无 protobuf |

### 属性 JSON 结构与语义

GE 侧将 ONNX 节点序列化为 JSON，顶层键 `attribute` 为属性数组：

```json
{
  "attribute": [
    { "name": "alpha",        "type": 1, "f": "0.25" },
    { "name": "transA",       "type": 2, "i": 1 },
    { "name": "auto_pad",     "type": 3, "s": "SAME_UPPER" },
    { "name": "kernel_shape", "type": 7, "ints": [3, 3] },
    { "name": "scales",       "type": 6, "floats": [0.5, 0.25] }
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | int | ONNX AttributeProto 类型：FLOAT=1、INT=2、STRING=3、FLOATS=6、INTS=7 |
| `i` | 整数 | 标量整型属性值 |
| `f` | **字符串** | 标量浮点属性值（GE 序列化时以字符串承载，须按字符串读取后再转 float） |
| `s` | 字符串 | 单值字符串属性（经 protobuf CEscape + JSON 双重转义） |
| `ints` / `floats` | 数组 | 列表属性（不经 CEscape 转义） |

两条关键语义规则：

1. **`f` 键缺省取 0.0，不得回落默认值**：proto3 序列化会省略值为 0 的标量字段。属性存在且 `type=FLOAT` 但 JSON 中无 `f` 键时，表示该属性实际值为 0，直接取 `0.0f`（例：HardSigmoid 存在 `alpha` 属性但无 `f` 键时取 0，而非默认 0.2）；
2. **type 不符按缺失处理**：属性名匹配但 `type` 与算子定义不符时，视为该属性缺失，保持默认值行为。

## 算子实现

### 总体流程

```mermaid
flowchart LR
    A["ONNX 节点"] --> B["GE 前端生成源 Operator<br/>属性序列化为 JSON"]
    B --> C["ParseParamsByOperatorFn 回调"]
    C --> D["读取 attribute JSON 与 input/output 名称列表"]
    D --> E["写入目标 Operator 属性"]
    E --> F["原有子图回调或图编译流程继续处理"]
```

### 改造范式

以 HardSigmoid 为例（迁移后实现摘录）：

```cpp
#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;
const int kHardSigmoidTypeFloat = 1;

static Status parse_params_hard_sigmoid(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float alpha = 0.2;   // 迁移前的算子默认值原样保留
    float beta = 0.5;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("type", 0) != kHardSigmoidTypeFloat) {
                        continue;                      // type 不符按缺失处理
                    }
                    if (attr.value("name", "") == "alpha") {
                        if (attr.contains("f")) {
                            std::string alpha_str = attr["f"];
                            if (!StrToFloat(alpha_str, alpha)) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "invalid alpha value: %s", alpha_str.c_str());
                                return FAILED;
                            }
                        } else {
                            alpha = 0.0f;              // f 键缺省：proto3 零值，不回落默认值
                        }
                    } else if (attr.value("name", "") == "beta") { /* 同 alpha，默认 0.5 */ }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
            return FAILED;                                 // 非法 JSON → FAILED
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    op_dest.SetAttr("alpha", alpha);
    op_dest.SetAttr("beta", beta);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("HardSigmoid")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::HardSigmoid"), /* ... */})
    .ParseParamsByOperatorFn(parse_params_hard_sigmoid)   // 仅注册入口改名
    .ImplyType(ImplyType::TVM);
}  // namespace domi
```

改造后的插件统一遵循如下结构：

1. **入参校验**：入参类型即 `ge::Operator`，不再有 `dynamic_cast` 失败分支；
2. **取属性 JSON**：`GetAttr("attribute", ...)` 不成功（属性不存在）时保持各算子迁移前行为（默认值 + `SUCCESS`）；
3. **try/catch 包裹**：`nlohmann::json::exception` 与未知异常均返回 `FAILED` 并记录日志；
4. **逐属性匹配**：先 `name`、再 `type`，再按类型取 `f`/`i`/`s`/`ints`/`floats`；
5. **失败分支语义逐条对齐迁移前**：默认值与返回码时机与原实现一一对应。

### 输入输出与节点名称

真实 ATC 源 Operator 可能尚未连接描述符，插件数量优先读取源 Operator 的 `input`/`output` 名称列表（GE 以同名属性注入，保留空字符串占位和顺序），列表属性不存在时回退描述符数量，兼容 framework UT 直接构造的 `op_src`。需要记录节点名称的插件使用 `GetOpName(op_src)` 写入目标 `name` 属性。

### 公共工具设计（`common/inc/framework/plugin_util.h`）

| 函数 | 来源 | 作用 |
| --- | --- | --- |
| `GetOpName(op)` | 原有 | 取算子名（替代 `node->name()`），失败返回 `None` |
| `StrToFloat(str, value)` | 原有 | 字符串转 float（空串/非法/越界返回 false），用于读 `f` 字段 |
| `GetNodeInputNum` / `GetNodeOutputNum` | 新增 | ONNX 节点输入/输出个数等价读取，含描述符回退 |
| `UnescapeOnnxString(escaped)` | 新增 | 还原 GE `Pb2Json` 对 **bytes 字段（张量 raw_data）与单值字符串属性 `s`** 的 CEscape + JSON 双重转义（如 `\000`、`\177`、`\n`、`\"`）；repeated 字段（`strings`、`float_data`、`int64_data`）不经此转义，不套用 |
| `CreateScalar` / `Vec2Tensor` / `ChangeFormatFromOnnx` | 下沉自 `onnx_common.h` | 三者只依赖 GE `graph/Operator` 接口，与 protobuf 无关，随 `onnx_common.h` 删除下沉至本头 |

`GetEnableUncacheFromNode`（原 `onnx_common.h` 中唯一的 protobuf 工具）不做公共化，其逻辑内联进 Gemm 插件（直接按 JSON 读取 `enable_uncache`）。解耦后全仓 `*_onnx_plugin.cpp` 已无任何文件引用 `ge_onnx.pb.h`。

### 特殊场景与语义保持清单

| 场景 | 处理方案 |
| --- | --- |
| `f` 键缺省（proto3 零值省略） | 取 0.0，不回落算子默认值 |
| 属性 `type` 与定义不符 | 按缺失处理（保持默认值） |
| 非法 attribute JSON | 记录日志 + 返回 `FAILED` |
| 不解析 attribute JSON 的插件 | 豁免：仍照写固定属性并返回 `SUCCESS`（实现与 UT 均不改） |
| 张量 raw_data / 字符串属性 `s` | 经 `UnescapeOnnxString` 还原后使用，与迁移前读取 protobuf 原始字节等价 |
| 节点输入/输出个数与空占位 | `GetNodeInputNum/GetNodeOutputNum` 等价读取，保留空名称占位与顺序 |
| 正则/数值边界判据（如 `fixed_shift_value` 越界回退默认值、`pads` 奇数长度返回 `FAILED`） | 逐条与迁移前实现比对，行为保持一致 |
| 迁移前永远不成立的失效判据 | 按工具函数真实返回约定修正，并新增 UT 用例守护 |
| `ParseOpToGraphFn`（如 Conv、Gemm 注册 PartitionedCall） | 不在迁移范围，保持原样 |
| `DynamicInputRegister` 等动态输入注册 | 保持原样 |

### 修改位置规划

| 仓内位置 | 拟修改内容 |
| --- | --- |
| 上述 78 个 `*_onnx_plugin.cpp` | 解析回调和注册方式、属性解析、节点名称与端口数量读取 |
| `common/inc/framework/` | 删除 `onnx_common.h`，公共能力集中到 `plugin_util.h` |
| 各算子及 `common` 的 `tests/ut/framework/` | 以 `ge::Operator` 构造输入，对应插件文件新增/更新 UT 与 `CMakeLists.txt` |
| framework UT 构建规则 | 收集 `experimental/` 等目录的用例，移除 protobuf 链接依赖 |

## 支持硬件与约束

- 迁移验证环境与任务书一致（CANN 版本大于 9.0.0），迁移前后两轮使用同一硬件、同一 CANN 与同一转换参数；
- Host 侧适配不新增设备计算路径，不涉及分核、UB 分块、TilingKey 与性能优化设计；
- 不扩展算子 dtype、shape、广播能力或 ONNX 版本支持范围，不新增第三方依赖（JSON 使用仓内 `third_party` 自带 nlohmann/json）。

# 可维可测分析

## 单元测试规划

开启仓内 `OP_HOST_UT`（`bash build.sh -u --ophost`），运行 `--gtest_filter='Onnx*PluginTest.*'`。用例直接 `#include "../../../framework/xxx_onnx_plugin.cpp"` 白盒直调静态解析函数，输入通过 `ge::Operator` 的 `SetAttr` 构造，**不构造任何 protobuf 对象**——这是解耦后可测性提升的直接体现：

```cpp
ge::Operator op_src;
op_src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"alpha","type":1,"f":"0.25"}]})"));
```

| 场景分类 | 主要用例 | 检查项 |
| --- | --- | --- |
| 正常解析 | 属性齐全，使用可区分值 | 返回码、目标属性名与数值 |
| 默认值 | 属性缺失、`f` 键省略、`type` 不符 | 算子默认值与 proto3 零值分别正确 |
| 异常输入 | 非法 JSON、字段类型错误、浮点转换失败 | 返回 `FAILED`，保留错误日志 |
| 边界与校验 | 越界回退、奇数长度 `pads`、空列表、可选端口空占位 | 与迁移前分支语义一致 |
| 特殊分支 | 失效 dtype 判据、多 opset 注册、动态输入 | 注册项与端口行为不变 |

用例组织按**插件文件与注册条目**一一对应，覆盖全部 78 个文件，并维护"插件文件—注册条目—UT—模型"的对应关系。

## 模型验证规划

1. 固定迁移前源码基线、模型、输入与转换参数；两轮分别构建插件，通过 `ASCEND_CUSTOM_OPP_PATH` 注入加载，并保留实际加载路径证据（转换日志中 "Plugin of … already registered, skip" 证明生效插件正确）；
2. 使用任务包模型生成脚本产出 `model.onnx` 与输入数据，经 ATC 转换生成 om（离线模型）并在昇腾设备上推理，取两轮输出比对；
3. **主判据**：同一模型、同一输入、同一 ATC 参数下，迁移前后两轮输出逐位一致（md5）；同时核对输出个数、dtype 与 shape；
4. 随机或非确定性用例先用任务包 `check_determinism.py` 取"同一 om 重复推理输出互不相同"的证据，再按用例包 `meta.json` 申报的档位（`tolerance` / `behavior`）判定，逐位一致结论不混入该类用例；
5. 平台侧不支持的用例保存两轮转换状态与原始错误，两轮同参数同失败且报错摘要匹配申报时按"两轮一致失败"通过；
6. 精度配置：主模式 `--precision_mode=must_keep_origin_dtype`（FP32 用例保持 ONNX 精度语义）；平台侧不支持该模式的算子（如 DynamicGRUV2Hidden 权重仅支持 FP16）回落默认精度模式两轮同参数复跑；旧接口插件如需兼容库，仅用于迁移前基线轮，不进仓库。

## 验收标准

| 验收项目 | 验收条件 | 依据 |
| --- | --- | --- |
| 迁移范围 | 任务书 74 个插件文件 / 75 个条目全部迁移，连带 4 个文件同步处理，UT 覆盖完整 | 任务书迁移范围表 |
| 构建与依赖 | `bash build.sh --onnxplugin -j8` 通过；插件 `.so` 无 `ge::onnx::*` / `protobuf::*` 未定义符号 | 任务书 2.1、3.5 节 |
| 单元测试 | `Onnx*PluginTest.*` 全部通过 | 任务书 3.5 节 |
| 模型精度 | 两轮 om 转换 + 推理输出按主判据及申报档位逐例判定，无未申报失败 | 任务书 3.2 节 |
| 规范检查 | `git diff --check` 及仓内格式要求通过，无与解耦无关的改动 | 仓库贡献规范 |
| 性能、内存 | 不涉及，不设对应指标 | 任务书 3.3、3.4 节 |

## 兼容性与风险

| 风险 | 控制方法 |
| --- | --- |
| 属性 JSON 与 protobuf 取值语义的细节差异（`f` 字符串、CEscape 转义、`f` 缺省=0） | 逐算子与迁移前源码比对 + 专项 UT + 两轮比对实测取证 |
| 属性整体缺失与值字段省略被混淆 | 两类场景分别设用例，默认值与零值分别校验 |
| 字段省略被误当作算子默认值 | 见上；`type` 不符同样按缺失处理并设用例 |
| 部分用例实测不可复现、无法逐位比对 | 先以 `check_determinism.py` 取证，再按 `meta.json` 申报档位判定并在验收说明中逐例说明 |
| 平台侧转换失败用例 | 两轮同参数同失败即判一致；在 `meta.json` 与验收说明中申报（本任务共申报 35 例预期转换失败、2 例预期推理失败） |
| 删除 `onnx_common.h` 的连带影响 | 全仓 grep 确认无残留引用；连带修改 4 个半迁移插件文件并在验收说明中列出 |
| 共享机器上两轮比对耗时受 NPU 争用影响 | 用例按设备分组并行；两轮使用同一设备、同一模型与 ATC 参数，逐位比较不受耗时影响 |
| 测试误加载内置插件 | 每轮转换日志保留插件加载证据 |

# 验证结果（实测）

本设计的落地验证已完成，实测记录如下（供评审参考；设计章节仍为本方案主体）。

| 项 | 结果 |
| --- | --- |
| 迁移实现 | 提交 `0d8e6dbb6`（父 `827cf0bf1`），198 个文件、+22193/−4147 |
| 插件编译 | `bash build.sh --onnxplugin -j8` 通过；产物 `liboponnx_plugin_nn.so` md5 `66244d167effae0dcec48942790715c7`（迁移前为 `84efe009a446e3d65a95db073938901f`，旧插件在部分 CANN 版本上需按任务包 `test_cases/verify_om/compat/` 的兼容库重新链接，仅用于自验基线） |
| 解耦证据 | 插件 `.so` 的 `ge::onnx::*` / `protobuf::*` / `ParseParamsFn` 未定义符号 6 → 0；全仓 0 处代码引用 `onnx/proto/ge_onnx.pb.h`；`onnx_common.h` 已删除 |
| framework UT | 验收口径 `--gtest_filter='Onnx*PluginTest.*'` 755 例 / 86 套全部通过；experimental 树 `OnnxNpuFusedAddRmsNormPluginTest` 10 例单独编译运行 10/10 通过；仓内全量 9082/9084（2 例失败为 `AddRmsNormTiling` 的 op_host tiling 用例，仓内既有，与本次迁移无关） |
| om 两轮比对 | 190 例：150 例按主判据逐位一致；2 例按 `meta.json` 申报档位判定（`tolerance` / `behavior`）；35 例申报预期转换失败；2 例申报预期推理失败；1 例平台容错（`ascend_anti_quant` dtype=0，另以 FP16 变体给出逐位一致证据）；FAIL 0、未申报失败 0 |
| 精度模式 | 主跑 `must_keep_origin_dtype`；GRU 家族 5 例按用例包说明改用默认精度模式复跑，两轮全部一致 |
| 单算子迁移自查 | 78 个插件文件逐项填表，`git diff --check` 提交级与工作区级均通过 |

自验证交付件（自验证步骤说明、精度/性能/内存自验证报告与日志、UT 与 OM 验证情况说明）已按任务书要求整理完成，随验收申请一并提交。
