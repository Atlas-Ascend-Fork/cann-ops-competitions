# 需求背景（required）

## 需求来源

社区任务 10-13「10月社区任务-ops-nn算子onnx适配」，在 A2/A3/950 上完成 [cann/ops-nn](https://gitcode.com/cann/ops-nn) 算子 ONNX 适配迁移重构，涉及 `Celu`、`GeGluV2` 等 60+ 函数改造。验收以任务书及配套算子清单为准；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿。

## 背景介绍

### 适配对象与仓内现状

ops-nn 每个算子目录的标准五段结构为 `op_host / op_kernel / op_graph / framework / tests`，其中 ONNX 适配落在 `framework/<op>_onnx_plugin.cpp`：`ParseParamsXxx` 将 ONNX NodeProto 属性映射为 GE 算子属性，`REGISTER_CUSTOM_OP(...).FrameworkType(ONNX).OriginOpType({ai.onnx::<opset>::<OpType>, ...})` 完成注册（当前普遍覆盖 opset 12～18）。仓内现存 onnx plugin 文件 ≈100 个（含 UT），分布在 activation/matmul/norm/loss/index 等目录。

现状痛点（迁移重构的动因）：

| 痛点 | 表现 |
| --- | --- |
| 解析代码重复 | 每个插件手写 `for (attr : node->attribute())` 遍历与类型分派，同型属性（FLOAT/INT/INTS/TENSOR/STRING）解析逻辑大量复制 |
| opset 注册碎片 | 各插件手写 `ai.onnx::12::X` 到 `::18::X` 列表，新 opset 扩展需逐文件改；个别插件 opset 上限不一致 |
| 属性/输入约定不统一 | 同为 ONNX 可选输入（如 alpha），有的走属性、有的走输入 tensor，graph infer 侧对应不一致 |
| 图推导覆盖不齐 | `op_graph/*_infershape.cpp` 对动态 shape/非默认 dtype 的支持程度参差，950（arch35）与 A2/A3 的 dtype 支持面差异未统一收敛 |
| 缺端到端验证 | 多数插件只有编译级验证，缺 ATC 加载 ONNX→图编译→单算子执行的闭环用例 |

### 平台范围

A2/A3（Ascend 910B 系）与 Ascend 950（arch35）。ONNX 适配层为 Host 侧图解析与算子注册，不含 kernel 改动；但验收要求端到端执行，故以各算子 kernel 已支持的 dtype/shape 面为适配上限，不扩大承诺。

# 需求分析（required）

## 需求描述

对任务书给定的 60+ 函数清单逐个完成：ONNX 属性/输入解析对齐 ONNX 标准算子定义、opset 版本覆盖扩展与统一、GE 属性/原型（proto/infershape）一致性修复、公共解析构件下沉、端到端验证用例补齐；以分类分批 PR 合入，A2/A3/950 三平台验证通过。

## 需求拆解

1. **清单冻结**：与任务书 60+ 清单对齐，建立 `op × onnx算子 × opset范围 × 属性集 × 平台支持` 审计矩阵（开发期文档），逐项标注新建/修复/仅验证。
2. **公共构件下沉**：新建 `common/framework/onnx_parse_utils`（属性解析模板 + opset 注册辅助宏），存量插件迁移到公共构件，消重 ≥70% 样板代码。
3. **逐算子改造**：按审计矩阵执行解析对齐、opset 统一（向上扩展至当前 ONNX Release 版本支持面，向下保持 12 兼容）、可选输入/属性双形态兼容。
4. **图推导一致性**：`op_graph` proto/infershape 与 plugin 解析结果对齐；动态 shape 用例覆盖。
5. **端到端验证**：每算子至少 1 条最小 ONNX 模型 ATC→执行→精度对拍用例（A2/A3/950），纳入 `tests/`。
6. **分批交付**：按算子类目分 4～6 批 PR，每批可独立编译、独立评审、独立回退。

# 详细设计（required）

## 总体方案

### 分层设计

```text
onnx_parse_utils（新增公共层）
├── AttrFetcher：按名取属性，FLOAT/INT/INTS/TENSOR/STRING/TYPE强类型模板 + 默认值语义
├── OpsetRegistrar：REGISTER_CUSTOM_OP 封装，opset 范围 → OriginOpType 列表批量生成
├── OptionalInputAdapter：ONNX 可选输入（"" 占位）→ GE 可选输入/属性 归一
└── PluginUtBase：NodeProto 构造器 + 解析断言（UT 复用）
```

存量 `framework/*_onnx_plugin.cpp` 收敛为：`ParseParamsXxx` 只保留算子特有的映射逻辑（如 Celu 的 alpha→alpha1/alpha2 双写），通用遍历全部走 `AttrFetcher`。

### 改造模板（以 Celu 为例）

```cpp
static Status ParseParamsCelu(const Message* op_src, ge::Operator& op_dest) {
  RETURN_IF_NOT_NODE(op_src, op_dest);            // 公共判空/转型
  float alpha = AttrFetcher::GetFloat(node, "alpha", 1.0f);
  op_dest.SetAttr("alpha1", alpha);
  op_dest.SetAttr("alpha2", alpha);
  op_dest.SetAttr("alpha3", AttrFetcher::GetFloat(node, "alpha3", 1.0f));
  return SUCCESS;
}
REGISTER_ONNX_OP("Celu", /*opsets*/{12, 13, 14, 15, 16, 17, 18});  // 宏内展开 OriginOpType
```

GeGluV2 类非标准算子（无 1:1 ONNX 对应）按任务书给定映射处理：ONNX 侧以组合/自定义域（`com.microsoft` 或任务书指定域）接入，映射关系写入各算子 README。

### 审计矩阵与批次划分

| 批次 | 类目（示例） | 预估数量 | 改造特征 |
| --- | --- | --- | --- |
| B1 | activation（Celu、GeGluV2、Elu、Mish、Selu…） | ~15 | 属性解析为主，样板消重大户 |
| B2 | norm/loss（Softmax 系、CrossEntropy…） | ~10 | 可选输入 + 维度属性 |
| B3 | matmul/index | ~12 | 输入张量形态与 transpose 属性 |
| B4 | 其余类目收尾 | 清单余量 | 按矩阵剩余项 |
| B5 | 回归 + opset 扩展收口 + 文档 | — | 全量端到端回归 |

### 验证设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| UT（host） | PluginUtBase：构造 NodeProto → ParseParams → 断言 GE 属性；opset 注册断言 | 每算子 ≥3 用例（默认属性/全属性/缺省可选输入） |
| 图编译 | `atc --framework=5 --model=<op>.onnx --soc=<A2/A3/950型号>` 编译通过 | 三平台编译零 ERROR |
| 端到端 | 最小 ONNX 模型推理执行 + CPU/NumPy golden 对拍 | 精度满足仓内单算子标准 |
| 一致性 | plugin 属性 ↔ op_graph proto/infershape 字段扫描脚本 | 零缺字段、零类型不匹配 |
| 回归 | 合入前 CI：全仓 build + 存量 UT + 本批新增 | 不引入存量回归 |

### 工程与交付

- 分支策略：每批次一个分支（`feat/onnx-adapt-b1`…），PR 描述附该批审计矩阵行；与任务书验收流程对齐。
- 文档：仓内 `docs/` 增补 ONNX 适配说明（支持的 ONNX 算子×opset 矩阵、映射约定、新增公共构件使用指南）。
- 风险控制：公共构件先落地 B1 并行小批验证，确认 CI 稳定后再放量；存量行为保持兼容（属性名、默认值逐字段对比，快照测试防漂移）。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 60+ 清单与任务书最终清单有出入 | 以任务书附件清单为唯一基线，审计矩阵在开工周内对齐并请任务方确认 |
| 公共构件迁移触发大范围 diff，评审困难 | 新构件与迁移分 PR：先合构件（零行为变更），再按类目迁移（diff 仅样板删除） |
| opset 向上扩展引入标准语义变化（同算子跨 opset 行为差异） | 逐 opset 差异清单化（ONNX 标准变更记录），有语义分叉的算子分注册项处理，不合并 opset |
| 非 1:1 映射算子（GeGluV2 等）映射无标准 | 以任务书映射为准，映射不明的在任务讨论区确认后再动工，不自行发明 |
| 三平台执行环境差异 | ATC/执行用例参数化 soc 型号，CI 矩阵覆盖；950 设备受限时按任务书验收安排申请资源 |
