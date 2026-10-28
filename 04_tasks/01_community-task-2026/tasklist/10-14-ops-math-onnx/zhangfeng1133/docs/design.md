# 需求背景（required）

## 需求来源

社区任务 10-14「10月社区任务-ops-math算子onnx适配」，在 A2/A3/950 上完成 [cann/ops-math](https://gitcode.com/cann/ops-math) 算子 ONNX 适配迁移重构，涉及 `ConcatFromSequence`、`Corr` 等 60+ 函数改造。验收以任务书及配套算子清单为准；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿。

## 背景介绍

### 适配对象与仓内现状

ops-math 的 ONNX 适配落在 `*_onnx_plugin.cpp`，现存约 75 个文件，呈**两种布局并存**：

| 布局 | 位置 | 现状 |
| --- | --- | --- |
| 集中式 | `common/src/framework/`（17 个，含 `concat_from_sequence_onnx_plugin.cpp`、`correlation_onnx_plugin.cpp`、einsum、eyelike、reduceL1/L2、random_uniform_like 等） | 已具备集中注册雏形，UT 在 `common/tests/ut/framework/` |
| 分散式 | 各算子目录 `<op>/framework/`（math/conversion/random/experimental 下散布） | 与 ops-nn 相同的逐算子插件模式 |

现状痛点：

| 痛点 | 表现 |
| --- | --- |
| 双布局并存 | 同类适配两种组织方式，构建注册路径不统一，新增算子无所适从 |
| 解析样板重复 | 手写 `node->attribute()` 遍历与类型分派，与 ops-nn 同源的复制式实现 |
| tf/onnx 插件混装 | `*_tf_plugin.cpp`（TensorFlow 域）与 `*_onnx_plugin.cpp` 共目录，FrameworkType/OriginOpType 语义易混 |
| opset 覆盖参差 | OriginOpType 手写 `ai.onnx::12::X…::18::X` 列表，上限不一、无统一扩展机制 |
| 端到端验证缺失 | 多数插件止步编译级验证，缺 ATC 加载 ONNX → 图编译 → 执行 → 精度对拍闭环 |

### 平台范围

A2/A3（Ascend 910B 系）与 Ascend 950（arch35）。适配层为 Host 侧图解析与注册；端到端执行以各算子 kernel 既有 dtype/shape 支持面为上限。

# 需求分析（required）

## 需求描述

对任务书 60+ 函数清单逐个完成 ONNX 解析对齐、opset 统一扩展、GE proto/infershape 一致性修复、公共构件下沉与布局归一（向集中式收敛），每算子补端到端验证用例；分类分批 PR 合入，A2/A3/950 三平台验证。

## 需求拆解

1. **清单冻结**：任务书 60+ 清单 → `op × onnx算子 × opset × 属性集 × 现布局(集中/分散) × 平台` 审计矩阵，逐项标注迁移动作。
2. **布局归一**：分散式插件迁移至 `common/src/framework/`（保留 `<op>/framework/` 目录仅放算子特有共享代码），构建注册统一入口。
3. **公共构件下沉**：`common/src/framework/onnx_parse_utils`（属性解析模板 + opset 注册宏 + 可选输入归一 + UT 构造器），存量插件迁移复用。
4. **逐算子改造**：解析对齐 ONNX 标准定义、opset 向上扩展至当前支持面（向下保 12 兼容）、`ConcatFromSequence`（sequence 输入 + axis/new_axis/pack_mode 属性）与 `Corr`（相关计算映射）等代表算子先行打样。
5. **端到端验证**：每算子 ≥1 条最小 ONNX 用例（三平台 ATC 编译 + 执行 + 对拍）入 `common/tests/ut/framework/` 或算子 `tests/`。
6. **分批交付**：4～6 批 PR，每批独立编译、评审、回退。

# 详细设计（required）

## 总体方案

### 分层与布局

```text
common/src/framework/                     # 唯一插件归属地（迁移终点）
├── onnx_parse_utils.h / .cpp             # 新增公共层
│   ├── AttrFetcher: FLOAT/INT/INTS/TENSOR/STRING 强类型取属性 + 默认值
│   ├── OpsetRegistrar: opset 范围 → OriginOpType 列表批量生成
│   ├── OptionalInputAdapter: "" 占位可选输入 → GE 归一
│   └── PluginUtBase: NodeProto 构造 + 断言
├── <op>_onnx_plugin.cpp                  # 只留算子特有映射逻辑
└── CMakeLists.txt                        # 统一注册进构建
```

`*_tf_plugin.cpp` 维持 FrameworkType(TF) 独立命名与注册，不与 ONNX 混编；目录内以 `_onnx_plugin.cpp` 后缀收敛 ONNX 适配物。

### 改造模板（以 ConcatFromSequence 为例）

```cpp
static Status ParseParamsConcatFromSequence(const Message* op_src, ge::Operator& op_dest) {
  RETURN_IF_NOT_NODE(op_src, op_dest);
  op_dest.SetAttr("axis", AttrFetcher::GetInt(node, "axis", 0));
  op_dest.SetAttr("new_axis", AttrFetcher::GetInt(node, "new_axis", 0));
  op_dest.SetAttr("pack_mode", AttrFetcher::GetString(node, "pack_mode", "CONCAT"));
  return SUCCESS;
}
REGISTER_ONNX_OP("ConcatFromSequence", /*opsets*/{13, 14, 15, 16, 17, 18});
```

`Corr` 类无 1:1 ONNX 标准对应的算子按任务书给定映射接入，映射约定写入算子 README 与 `docs/` 适配说明。

### 批次划分

| 批次 | 内容 | 特征 |
| --- | --- | --- |
| B0 | 公共构件 + 2 个打样算子（ConcatFromSequence、Corr） | 定型迁移范式 |
| B1 | math/reduce、conversion 类 | 数量最大，样板消重主力 |
| B2 | random 类（stateless 系、*_like 系） | 生成类属性/seed 语义对齐 |
| B3 | 其余清单 + tf 插件隔离整理 | 收尾 |
| B4 | opset 扩展收口 + 全量回归 + 文档 | 支持矩阵发布 |

### 验证设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| UT | PluginUtBase 解析断言（默认/全量/缺省属性 3 用例）+ opset 注册断言 | 每算子 ≥3 条 |
| 图编译 | `atc --framework=5 --model=<op>.onnx --soc=<A2/A3/950>` | 三平台零 ERROR |
| 端到端 | 最小 ONNX 执行 + NumPy/CPU golden 对拍 | 仓内单算子精度标准 |
| 一致性 | plugin↔op_graph proto/infershape 字段扫描脚本 | 零缺字段 |
| 回归 | 全仓 build + 存量 UT + 本批新增 | 零存量回归 |

迁移兼容性防线：迁移 PR 前对每个插件做「属性快照」（解析结果序列化对比），迁移前后逐字段一致方可合入。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 双布局归一动到构建系统，波及面大 | B0 先迁移 2 个算子验证 CMake/注册链，批量迁移每批独立 PR 可回退 |
| 任务书清单与仓内现状不一致（个别函数已有/无插件） | 审计矩阵开工周对齐任务方；已有插件走"验证+修补"，缺失走"新建" |
| sequence 类输入（ConcatFromSequence）在 GE 侧图表达受限 | 以 ONNX 标准 loop/sequence 语义 + 任务书约定为准；不可表达项在任务讨论区确认范围，不自行裁剪承诺 |
| opset 语义分叉 | 逐 opset 差异清单化，分叉算子分注册项，不合并 opset |
| 三平台资源与工具链差异 | ATC/执行用例 soc 参数化；950 资源按任务书验收安排申请 |
