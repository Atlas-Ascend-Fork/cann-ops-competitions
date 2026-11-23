# 需求背景（required）

> 任务编号：02-25　算子名称：SoftMarginLoss　提交团队：xoljj
> 本文所有源码结论均基于 cann/ops-nn `master@188be8327c92033239047673f3b20a84907105c3`（2026-11-16 10:53 +08）与 cann/cann-ops-competitions `master@da8fdd5221f326d632bffe9d1261de7535086b2c`。凡未能在源码或官方接口文档中直接核对的结论，正文以「【未验证】」显式标注，并在附录「证据清单与未验证项」集中列出。

## 需求来源

1. CANN 社区任务 2026（年度算子开发任务）。任务列表见 `04_tasks/01_community-task-2026/docs/README.md` 的「2月份发放的任务」表格，第 `02-25` 行为 `SoftMarginLoss`；同批次语义强相关的还有 `02-24 SoftMarginLossGrad`（反向）、`02-19 Softplus`、`02-20 Softshrink`、`02-23 MseLossGrad`。
2. 认领入口与讨论区：`https://gitcode.com/org/cann/discussions/22`（仓库 README 给出的官方入口）。
3. 交付要求：本任务目录 `04_tasks/01_community-task-2026/tasklist/02-25-SoftMarginLoss/` 下当前仅有模板占位目录 `TeamName/`，没有任何团队提交过设计文档；本次提交按「最小可接受交付」要求提供 `{TeamName}/docs/design.md`，即 `xoljj/docs/design.md`。
4. 说明：`docs/` 目录只为 202604/202605/202607/202608 四个批次提供了 `*_task_doc.md` 任务书，**2 月批次（含 02-25）没有单独任务书文件**。因此本文的精度/性能验收口径参照同类 loss 算子的通用要求给出建议值，并在「可维可测分析」中标注需与维护者确认的阈值来源。

## 背景介绍

### SoftMarginLoss 算子现状分析

SoftMarginLoss 是二分类逻辑损失（logistic loss / log-sigmoid loss）：对预测分数 `self` 与带符号标签 `target ∈ {-1, +1}` 逐元素求 `log(1 + exp(-target * self))`，再按 `reduction` 做 `none / sum / mean` 归约。它是 `SigmoidCrossEntropyWithLogits` 在 ±1 标签场景下的等价形式，也是 PyTorch `nn.SoftMarginLoss`、TensorFlow `tf.losses.log_loss`（符号标签变体）的公共底座，在分类训练、SVR/逻辑回归、离线评估与推荐 CTR 场景中使用频繁。

社区现状（已核对源码）：`cann/ops-nn` 中已存在该算子，目录结构为

```
loss/soft_margin_loss/
├── docs/aclnnSoftMarginLoss.md            # 已提供（中/英）
├── examples/test_aclnn_soft_margin_loss.cpp, examples/arch35/
├── op_api/aclnn_soft_margin_loss.{cpp,h}  # 两段式 aclnn 接口 + 参数校验
├── op_graph/soft_margin_loss_graph_infer.cpp, soft_margin_loss_proto.h
├── op_host/soft_margin_loss_def.cpp       # OpDef 注册（CANNBot 生成）
├── op_host/soft_margin_loss_infershape.cpp
├── op_host/arch35/soft_margin_loss_tiling{,_arch35}.{cpp,h}
├── op_kernel/soft_margin_loss.cpp         # __global__ 入口，按 Reduction 三分支
├── op_kernel/arch35/soft_margin_loss_{dag,reduce,tiling_def,tiling_key}.h
└── tests/{st,ut}
```

即：**该任务不是从零造算子，而是在既有 Ascend C 实现上做「读懂—验证—补齐—优化」**。设计文档因此把重点放在三件事上：(a) 现有实现的完整链路与数值实现细节；(b) 接口文档与 host 校验之间的口径差异；(c) 可维可测与可优化项。

### 算子实现路径（现状分析）

`op_host/soft_margin_loss_def.cpp` 中 `OpDef` 的 AICore 配置只注册了两个架构族：

```cpp
aicoreConfig.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .PrecisionReduceFlag(false);
this->AICore().AddConfig("ascend950", aicoreConfig);
this->AICore().AddConfig("ascend350", aicoreConfig);
```

由此可确定两条实现路径：

- **arch35（ascend950 / ascend350）新式 Ascend C 实现**：kernel 侧 `#include "arch35/soft_margin_loss_dag.h"` 与 `atvoss/{elewise,reduce,broadcast}` 调度器，host 侧 `op_host/arch35/soft_margin_loss_tiling_arch35.cpp` 用 `ElewiseBaseTiling` / `Tiling4ReduceOp` 生成 tiling。这是本文详细设计的主线。
- **其余架构（regbase 类）走存量实现**：`op_api/aclnn_soft_margin_loss.cpp` 中 `CheckFormat` 以 `Ops::NN::AclnnUtil::IsRegbase(curArch)` 分流，并用 `GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST)` 按架构选择 dtype 集合，说明接口层显式承认存在「非 arch35 的第二套实现」。
  【未验证】该第二套实现是否为 toolkit 内置 TBE 单算子（模板中提到的 `.../opp/built-in/op_impl/ai_core/tbe/impl` 目录形态）。本机未安装 Ascend 工具包，无法用 `ls` 核实；且 arch35 的 `op_kernel` 目录里没有旧式 `soft_margin_loss_tiling.h` 之外的 TBE 文件，故本文不将「TBE 实现」作为设计依据，只作为需要向维护者确认的兼容面。

`docs/aclnnSoftMarginLoss.md` 给出的产品支持矩阵（用于「支持硬件」章节）：Ascend 950PR&950DT、Atlas A3、Atlas A2、Atlas 推理系列（310p）、Atlas 训练系列（910）**支持**；Atlas 200I/500 A2 推理产品**不支持**；910/310p 上**不支持 BFLOAT16**。

### 算子功能分析

以 `op_def_registry` 的入参出参与接口文档为准确认功能边界：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| `input_x`（接口名 `self`） | 预测分数/逻辑值 | ND，FLOAT 系 | FLOAT、FLOAT16、BFLOAT16（910/310p 无 BF16） | 与 `input_y` **dtype 必须相同**；非连续支持；`FORMAT_FRACTAL_NZ` 仅告警不拒绝 | 0–8 维 |
| `input_y`（接口名 `target`） | 带符号真实标签，语义取值 ±1 | ND，FLOAT 系 | FLOAT、FLOAT16、BFLOAT16（同上） | host 侧要求与 `input_x` **rank 相同、已知维度逐维相等**（详见差异 D2） | 0–8 维 |
| `reduction`（attr） | 归约方式 | STRING（attr 默认 `"mean"`）；aclnn 接口为 `int64_t` | `none`/`mean`/`sum`；接口层 `0`/`1`/`2` | 接口层校验 `0 ≤ reduction ≤ 2`；非法值 `ACLNN_ERR_PARAM_INVALID(161002)` | - |
| `output_z`（接口名 `out`） | 损失结果 | ND，FLOAT 系 | 由 `InferDataType4SoftMarginLoss` 直接取 `Input(0)` 的 dtype | `reduction=none` 时 shape 须等于 broadcast 结果 shape；`mean/sum` 时须为 0 维标量 | 0–8 维 / 0 维 |

要点：

1. 输出 dtype 恒等于 `self` 的 dtype（`context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0))`），不做提升；中间计算提升到 `float`。
2. `target` 不做 ±1 合法性校验：传入 0、2 或 NaN 时算子照常计算。这是与 PyTorch（`SoftMarginLoss` 用 int64 类别索引，语义完全不同）之间最容易踩坑的差异，功能分析必须写明。
3. 空输入（任一维为 0）受支持：接口文档「约束说明」明确「支持空输入」。
4. 确定性：接口文档明确 `aclnnSoftMarginLoss` 默认确定性实现。

## 需求分析（required）

### 需求描述

在 arch35（Ascend 950PR/950DT、Ascend 350 系）与 regbase 存量路径两个层面，交付一个满足下列条件的 SoftMarginLoss 算子实现与设计说明：

- **功能**：逐元素 `L = log(1 + exp(-target * self))`，并按 `reduction ∈ {none, sum, mean}` 输出张量或 0 维标量；支持 FLOAT/FLOAT16/BFLOAT16、0–8 维、非连续 tensor、空 tensor。
- **数值**：全程 fp32 中间量；`exp` 上溢按 `+inf` 语义传播；`log1p` 在小 `exp` 值处不得因 `1 + e` 舍入到 1 而提前塌陷为 0。
- **性能**：单算子访存受限（`none` 分支读 2N 写 N；归约分支额外一轮 workspace 归约），要求在不增加额外全局内存往返的前提下完成三级链融合。
- **可维可测**：tiling key 组合可穷举、精度用例可自动生成、失败可定位到具体 reduction×dtype×pattern 组合。
- **接口稳定**：两段式 aclnn 接口，第一段完成全部参数校验；`reduction` 在接口层（int）与算子层（string→int）之间的映射不改变对外语义。

### 需求拆解

| 编号 | 子需求 | 落点 | 验收方式 |
|------|--------|------|----------|
| R1 | 入参校验与 dtype/broadcast/维度检查 | `op_api/aclnn_soft_margin_loss.cpp` 的 `CheckNotNull3Tensor / CheckDtypeValid / CheckShape / CheckFormat / CheckReduction / CheckParams` | UT：非法组合逐个断言错误码 |
| R2 | shape 与 rank 约束、输出 shape 推导 | `op_host/soft_margin_loss_infershape.cpp` + arch35 tiling 的 `CheckShape()` | UT：0/1/8 维、rank 不等、0 维标量 out |
| R3 | 输出 dtype 推导规则 | `op_graph/soft_margin_loss_graph_infer.cpp`（取 Input0） | ST：三种 dtype 输出类型一致 |
| R4 | `reduction=none` 元素级分支实现 | kernel `Reduction==0` → `ElewiseSch<0, SoftMarginLossOp<T>::OpDag>`；host `TilingEle()` | ST + 性能对比 CPU golden |
| R5 | `sum/mean` 归约分支实现 | kernel `Reduction==1/2` → `ReduceSch<..., SumDag/MeanDag>`；host `TilingReduce()` 全轴归约 + `meanVar` | ST：fp64 golden 与 fp32 累加误差 |
| R6 | log-sigmoid 数值稳定性 | `SoftMarginLog1pVF` 的补偿式 `log(u) * e / (u - 1)` 与 `u==1`/`u==inf` 双 select | UT：`e ∈ [1e-12, 1e-8]`、`-t*x > 88.72` |
| R7 | tiling key 与可枚举性 | `soft_margin_loss_tiling_key.h`：`REDUCTION_BIT_WIDTH=2`、`SML_SUM=1`、`SML_MEAN=2`、`SML_TPL_{FP16=10,FP32=20,BFP16=30}` + `REDUCE_TPL_*` | 编译期检查 key 空间不冲突 |
| R8 | 文档口径一致性（D1/D2/D3 差异，见详细设计） | `docs/aclnnSoftMarginLoss.md` 与 host `CheckShape` 描述 | 提出 issue/修订 PR |
| R9 | 性能与精度基线（可维可测） | `tests/{st,ut}` + 精度/性能标准表 | 基线报告 |

## 详细设计（required）

### 算子分析

#### 数学公式

记 `x = self[i]`、`t = target[i]`、`N = numel(broadcast(shape_x, shape_t))`：

```
L(x, t) = log(1 + exp(-t * x))                    # 逐元素
out = none : L                                       (shape 同 broadcast 结果)
    | sum  : Σ_i L(x_i, t_i)                          (0 维标量)
    | mean : (1/N) * Σ_i L(x_i, t_i)                  (0 维标量)
```

实现恒等变形（kernel 实际链路，逐字来自 `soft_margin_loss_dag.h` 注释 `neg_out().mul_().exp_().log1p_()`）：

```
tx  = t * x                    → Vec::Mul
ntx = -1 * tx                  → Vec::Muls(ConstNegOne = -1)
e   = exp(ntx)                 → Vec::Exp
L   = log1p(e)                 → SoftMarginLog1p（自研 VF，非直接 Log）
```

数值边界（与接口文档「约束说明」一致）：

- **上溢**：当 `-t*x > ln(FLT_MAX) ≈ 88.72` 时 `e = +inf`，`u = 1 + e = +inf`，`L = log(+inf) = +inf`；`mean/sum` 时任一元素为 `inf` 使标量输出为 `inf`。典型场景：预测方向严重错误（`t=1, x=-100`）。
- **下溢/塌陷**：当 `e` 小于 1.0 附近半个 ULP（fp32 约 `6e-8` 量级）时，`u = 1 + e` 舍入为 1，朴素 `log(u)` 直接给 0，丢失全部有效位。因此必须用补偿式或对数-1 专用近似，不能用朴素 `Log(1+e)`。
- **NaN**：`x` 或 `t` 含 NaN 时结果 NaN，算子不做屏蔽。
- **`N = 0`**（空输入且 `mean`）：0/0 语义。kernel 的 mean 分支写作 `op.Process(static_cast<DTYPE_INPUT_X>(NAN))`，即以 NaN 作为累加器初值/占位，从而在空输入时倾向输出 NaN 而不是 0。【未验证】`Process(fillValue)` 的精确语义来自 atvoss 通用 `ReduceSch`（本次未读取 `atvoss/reduce/reduce_sch.h` 源文件），需与维护者或 `tests/st` 用例确认空输入 `mean` 的期望值。

#### 支持数据类型

| 输入组合 | 是否支持 | 依据 |
|----------|----------|------|
| self/target/out 同为 FLOAT | ✔ | `def.cpp` 三处 `ge::DT_FLOAT`；`DTYPE_2_INT_KEY{DT_FLOAT,20}` |
| 同为 FLOAT16 | ✔ | `DT_FLOAT16` → key 10 |
| 同为 BFLOAT16 | ✔（arch35 / 910B；910 与 310p 不支持） | `DT_BF16` → key 30；文档 `<!-- npu="910,310p" -->` 注释块；`ASCEND910_DTYPE_SUPPORT_LIST` 只含 FLOAT/FLOAT16 |
| self 与 target dtype 不同 | ✘ | tiling `CheckShape()` 显式报错「The dtypes of input_x and input_y must be the same」 |
| INT/UINT 系列、INT64 标签 | ✘ | OpDef 未注册；接口层 `OP_CHECK_DTYPE_NOT_SUPPORT` |
| 输出 dtype 提升为 FP64 | ✘ | `InferDataType` 取 Input0；fp64 不在支持列表 |

中间累加类型：`using PromoteType = __reduceType::GetPromoteType<DTYPE_INPUT_X>::T;`，fp16/bf16 归约时提升到 `float`（host `Tiling4ReduceOp<..., half, float>` / `<..., float, float>` 模板实参可直接印证）。设计取向是「输入半精度、累加单精度、输出回写原精度」，与 fp16 训练损失的通行做法一致。

#### 支持形状

- rank：`0–8`（tiling `MAX_SUPPORTED_RANK = 8`，接口层 `OP_CHECK_MAX_DIM`）。
- `Ops::Base::EnsureNotScalar()` 在校验前把 0 维 storage shape 归一化，避免标量分支进入逐维比较。
- `IsUnknownRank` 时跳过静态校验，交给动态编译/运行时（`DynamicShapeSupportFlag/DynamicRankSupportFlag` 均为 `true`）。
- 维度语义：本算子对形状无「类别维」要求（与 PyTorch `(N, C)` + `int64` target 的形式不同），只要求两输入逐维可比。

### 算子实现

#### 3.2.1 host侧设计

链路（`SoftMarginLossTiling::RunTiling`，逐段对应源码）：

```
RunTiling
 ├─ CheckShape()                      # 空指针/未知 rank/rank 相等/已知维相等/dtype 一致/output dtype 合法性
 ├─ attrs->GetAttrPointer<char>(0)    # reduction 字符串（"none"|"sum"|"mean"）
 ├─ STR_2_INT.find()                  # none→0, sum→1, mean→2
 ├─ reduction == none  → TilingEle()      # ElewiseBaseTiling::DoTiling<SoftMarginLossOp<T>::OpDag>(tiling->baseTiling)
 ├─ reduction in (mean,sum) → TilingReduce(compileInfo)
 │     ├─ ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0)
 │     ├─ opInput.axes = {0..rank-1}      # 全轴归约，标量输出
 │     └─ Tiling4ReduceOp<SumDag<T,float> | MeanDag<T,float>>(..., key.ReduceTiling, &compileInfo->opInfo, &tiling->reduceTiling)
 └─ SetTilingData()                   # GEN_REDUCE_TILING_KEY(tilingKey, key.ReduceTiling, key.Reduction, key.Dtype)
                                       # if (reduction == REDUCTION_NONE) { workspace[0]=16MiB; SetBlockDim(tiling->baseTiling.blockNum); }
                                       # SetTilingKey(tilingKey)
```

##### 1. 分核策略

- **none 分支**：分核完全委托通用 `ElewiseBaseTiling`，host 只负责把结果写进 `tiling->baseTiling` 并 `SetBlockDim(tiling->baseTiling.blockNum)`。即「按 AI Core 数均分元素块 + UB 内再分块」，多 shape/非连续由 `atvoss/broadcast/broadcast_tiling.h` 参与决策。这与「手写按 `total / blockNum` 向上取整切块」的旧式 tiling 相比，收益是新增 dtype/shape 组合时无需改算子代码，代价是分核行为随公共组件版本演进，需在可维可测里固定回归用例。
- **sum/mean 分支**：不做手工分核，交给 `Tiling4ReduceOp`。它依据输入 shape 与 axes（本算子恒为全轴）选择 reduce pattern，并把 `patternID / loopARCount / loopInnerARCount` 三个字段填进 `key.ReduceTiling` 供 tilingkey 生成——`OP_LOGI` 同时打印这三项，是现场定位「为什么这个 shape 慢」的第一手日志。
- **`SetBlockDim` 只在 none 分支设置**（源码即如此）。归约分支的 blockDim 由 `ReduceOpTmpl` 内部设定，这是两条分支共用一个 `SetTilingData()` 时容易被忽略的不对称点。

##### 2. 数据分块和内存优化策略

- **UB 分块**：由 `ElewiseBaseTiling` / `Tiling4ReduceOp` 依据 arch35 的 UB 容量与 `VECTOR_REG_WIDTH` 计算每轮 tile 长度；kernel 侧 `SoftMarginLog1pVF` 以 `vlSize = VECTOR_REG_WIDTH / sizeof(T)`、`loopNum = CeilDivision(count, vlSize)`、`Reg::UpdateMask<T>(count)` 处理尾块，保证非整向量长度不越界。
- **DAG 级融合（关键优化）**：`OpDag = DAGSch<Outputs>` / `DAGSch<Outputs, void, MemOptCfg<MemLevel::LEVEL_2>>` 把 CopyIn → Cast → Mul → Muls → Exp → Log1p →（Cast / Reduce）→ CopyOut 表达为一张 DAG，由调度器统一分配 TB 与流水，避免「先落 GM 再读回」的额外全局内存往返。归约两个分支额外指定 `MemOptCfg<MemLevel::LEVEL_2>`，即在二级（L2）做中间数据优化。
- **workspace 用量**：`ASCEND_WORKSPACE = 16 * 1024 * 1024`（16 MiB）。这里存在一个值得核对的实现细节，记为 **差异 D3**：源码写的是 `if (this->reduction == REDUCTION_NONE) { currentWorkspace[0] = ASCEND_WORKSPACE; ... }`，即**只有 none 分支申请 16 MiB workspace，而需要跨核累加的 sum/mean 分支不申请**。常规实现通常相反（归约才需要 workspace）。可能的解释：(a) arch35 归约走 L2/单核 pattern 无需 GM 暂存；(b) 元素级分支的 broadcast 需要辅助缓冲；(c) 条件应为 `!=`，属笔误。**本次只读到源码文本，未在设备上观测 workspace 实际使用**，故按「需维护者确认」提出，而不是断言为 bug。
- **累加精度**：归约前统一 `Vec::Cast<PromoteT, T, 0>` 提升到 fp32，累加与 `meanVar` 乘法都在 fp32 完成，最后 `Vec::Cast<T, PromoteT, 1>` 回写，避免 fp16 直接累加导致的截断。

##### 3. tilingkey规划策略

`soft_margin_loss_tiling_key.h`：

```c
#define REDUCTION_BIT_WIDTH 2
#define SML_SUM 1
#define SML_MEAN 2
#define SML_TPL_FP16 10
#define SML_TPL_FP32 20
#define SML_TPL_BFP16 30

ASCENDC_TPL_ARGS_DECL(SOFTMARGINLOSS, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(Reduction, REDUCTION_BIT_WIDTH, ASCENDC_TPL_UI_LIST, 0, SML_SUM, SML_MEAN),
    ASCENDC_TPL_UINT_DECL(Dtype, BIT_WIDTH, ASCENDC_TPL_UI_LIST, SML_TPL_FP16, SML_TPL_FP32, SML_TPL_BFP16));
```

组合为 `GEN_REDUCE_TILING_KEY(tilingKey, key.ReduceTiling, key.Reduction, key.Dtype)`，即三维正交：

| 维度 | 取值 | 语义 |
|------|------|------|
| `ReduceTiling` | `patternID / loopARCount / loopInnerARCount`（atvoss 通用宏 `REDUCE_TPL_KEY_DECL/SEL`） | 归约遍历形态与循环展开档 |
| `Reduction` | 2 bit，`{0 = none, 1 = sum, 2 = mean}` | 选择 kernel 里 `if constexpr` 的三条编译分支 |
| `Dtype` | `{10 = fp16, 20 = fp32, 30 = bf16}` | 选择模板实参，与 `DTYPE_2_INT_KEY` 一一对应 |

规划要点：(a) `ASCENDC_TPL_ARGS_SEL` 为三个 dtype × 三个 reduction 分别列出合法组合（`none` 时 `ReduceTiling` 用 `REDUCE_TPL_KEY_SEL_EMPTY()`，因为元素级没有归约 pattern），非法组合不会编译出冗余二进制；(b) `Reduction` 只占 2 bit，与 `Dtype` 值域不重叠，`SML_*` 与 `SML_TPL_*` 不冲突——这是可维可测阶段定位「命中了哪条实现」的基础；(c) `REDUCTION_BIT_WIDTH` 若未来要支持按轴归约需要扩位，应同步 `docs` 与 tilingkey 声明。

**差异 D1（接口层与算子层的 reduction 枚举不同序）**：

| 层 | none | mean | sum |
|----|------|------|-----|
| aclnn 接口 `int64_t reduction` | 0 | **1** | **2** |
| 算子层 attr 字符串 → `STR_2_INT` / tilingkey / kernel `Reduction` | 0 | **2** | **1** |

`op_api/aclnn_soft_margin_loss.cpp` 定义 `REDUCTION_MEAN_NUM = 1`、`REDUCTION_SUM_NUM = 2`；`op_host/.../soft_margin_loss_tiling_arch35.cpp` 定义 `REDUCTION_SUM = 1`、`REDUCTION_MEAN = 2`。因此 kernel 里 `Reduction == 1` 走 `SoftMarginLossSumDag`、`Reduction == 2` 走 `SoftMarginLossMeanDag` 是**正确**的（与接口文档 `0=none|1=mean|2=sum` 相反序）。风险在于二次开发者容易误读为错位。建议：在 `tiling_key.h` 上方补一行注释说明映射关系，或在 `aclnn` 层把 int 转成字符串 attr 后统一只留一套枚举。

#### 3.2.2 kernel侧设计

入口（`op_kernel/soft_margin_loss.cpp`）：

```cpp
template <REDUCE_TPL_PARAM, uint32_t Reduction, uint32_t Dtype>
__global__ __aicore__ void soft_margin_loss(GM_ADDR self, GM_ADDR target, GM_ADDR loss,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftMarginLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftMarginLossTilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_INPUT_X>::T;
    if constexpr (Reduction == 0) {
        ElementwiseSch<0UL, SoftMarginLoss::SoftMarginLossOp<DTYPE_INPUT_X>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(self, target, loss); sch.Process();
    } else if constexpr (Reduction == 1) {          // sum
        using Op = ReduceSch<REDUCE_TPL_VALUE, SoftMarginLoss::SoftMarginLossSumDag<DTYPE_INPUT_X, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.Init(&pipe, self, target, loss, workspace); op.Process();
    } else if constexpr (Reduction == 2) {          // mean
        using Op = ReduceSch<REDUCE_TPL_VALUE, SoftMarginLoss::SoftMarginLossMeanDag<DTYPE_INPUT_X, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<PromoteType, 0>(tilingData.reduceTiling.meanVar);
        op.Init(&pipe, self, target, loss, workspace);
        op.Process(static_cast<DTYPE_INPUT_X>(NAN));
    }
}
```

**Init/Process 结构**：`Init()` 负责绑定 GM 地址、按 tiling 申请 LocalTensor/TBuf、构造调度上下文；`Process()` 驱动 DAG 逐 tile 完成 CopyIn → Compute → CopyOut，归约分支的 `Process()` 内再分「块内累加 → 跨迭代累加 →（可选跨核归约）→ 标量回写」阶段。`REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 是 arch35 新式写法，tiling 结构体由 `SoftMarginLossTilingData{ EleBaseTilingData baseTiling; ReduceOpTilingData reduceTiling; }` 复用公共布局，算子自身不新增字段——好处是公共组件升级即受益，约束是不能塞算子私有参数（若将来要加，应扩 `tiling_def.h` 并同步 host）。

**三个 DAG 的拓扑差异**：

| 分支 | DAG 尾部 | 说明 |
|------|----------|------|
| `SoftMarginLossOp`（none） | `OpResult` → `Cast<T,PromoteT,1>` → `CopyOut` | 纯元素级，fp16/bf16 结果回写原精度 |
| `SoftMarginLossSumDag` | `OpResult` → `PairwiseReduceSumOp<PromoteT>` → `Cast` → `CopyOut` | `MemOptCfg<LEVEL_2>` |
| `SoftMarginLossMeanDag` | `OpResult` → `Vec::ReduceSumOp<PromoteT>` → `Muls(Var<0> = meanVar)` → `Cast` → `CopyOut` | `SetVar<PromoteType,0>(tilingData.reduceTiling.meanVar)` 注入 1/N，用乘法替代除法（矢量单元无高效除法） |

`soft_margin_loss_reduce.h` 里的 `PairwiseReduceSumOp<T> : public Vec::ReduceSumOp<T>` **目前是空继承**（只留默认构造函数），即「pairwise/树形求和」尚未真正生效，只是给未来留出替换点。这是一个明确可交付的改进项，记为 **优化 O1**。

**log-sigmoid 核心：`SoftMarginLog1pVF` 的补偿式实现**（本任务最有含金量的部分）：

```cpp
constexpr uint32_t vlSize = VECTOR_REG_WIDTH / sizeof(T);
const uint16_t loopNum = CeilDivision(count, vlSize);
for (loopIdx ...) {
    mask = Reg::UpdateMask<T>(count);            // 尾块掩码
    Reg::LoadAlign(e, src + loopIdx * vlSize);
    Reg::Adds(u, e, one, mask);                   // u = 1 + e
    Reg::Compares<T, CMPMODE::EQ>(uIsOne, u, one, mask);   // 塌陷检测
    Reg::Adds(v, u, negOne, mask);                // v = u - 1（可精确还原小 e）
    Reg::Log(logU, u, mask);                      // log(u)
    Reg::Div(ratio, e, v, mask);                  // e / (u-1)
    Reg::Mul(result, logU, ratio, mask);          // log(u) * e/(u-1) == log(1+e)
    Reg::Select(result, e, result, uIsOne);       // u==1 时退化为 e
    Reg::Compare<T, CMPMODE::EQ>(uIsInf, u, infReg, mask);
    Reg::Select(result, infReg, result, uIsInf);  // exp 上溢时保持 inf
    Reg::StoreAlign(dst + loopIdx * vlSize, result);
}
```

原理：当 `e` 很小时 `u = 1 + e` 会舍入到 1，`log(u)` 丢精度甚至得 0。用 `log(1+e) = log(u) * e / (u-1)` 做一阶补偿——`log(u)` 的相对误差与 `(u-1)/e` 的舍入误差相乘后被抵消；当 `u` 已经严格等于 1（`u-1 = 0`，除法会得到 NaN）时，用 `uIsOne` 掩码直接取 `e`（因为 `log(1+e) ≈ e`，此时 `e` 就是正确结果）。当 `e = +inf` 时 `u - 1 = inf`、`e/(u-1)` 为 NaN，故再用 `uIsInf` 掩码强制输出 `+inf`。两个 `Select` 是这段实现能同时覆盖「下溢塌陷」与「上溢」两端的关键，也是精度用例必须专门打靶的路径。

指令构成上，单条向量 tile 需要 `Adds×2 + Log + Div + Mul + Compares×2 + Select×2`，比朴素 `Log(Adds(...))` 多约 5 条矢量指令。取舍判断：本算子是访存受限，`Exp`/`Log` 走特殊函数单元本就存在流水线气泡，补偿式把空闲的 VF 周期换成精度，成本接近零；若后续实测出现 VF 瓶颈，可考虑用多项式近似（`log1p` 的 3–5 阶泰勒 + 分段）替换 `Log`+`Div`+`Mul`。

**与 regbase 存量路径的关系**：kernel 入口整体在 arch35 头文件驱动下编译，非 arch35 架构由接口层分流到存量实现（见「实现路径」）。因此在补齐本任务时，任何 tilingkey 或语义变更都要同步验证两套路径，不能只跑 arch35。

## 支持硬件

| 产品形态 | 是否支持 | 备注（均引自 `docs/aclnnSoftMarginLoss.md`） |
|----------|----------|---------------------------------------------|
| Ascend 950PR & 950DT 系列 | 支持 | arch35 新式 Ascend C 实现（`AddConfig("ascend950")`） |
| Ascend 350 系列 | 支持 | `AddConfig("ascend350")`，与 ascend950 共用同一份 tiling/kernel |
| Atlas A3 系列 | 支持 | 接口文档标注支持；OpDef 未显式出现对应架构名【未验证】具体由哪个 arch 名承接 |
| Atlas A2 系列 | 支持 | 同上；BF16 可用 |
| Atlas 推理系列（310p） | 支持 | **不支持 BFLOAT16** |
| Atlas 训练系列（910） | 支持 | **不支持 BFLOAT16**（`ASCEND910_DTYPE_SUPPORT_LIST = {FLOAT, FLOAT16}`） |
| Atlas 200I / 500 A2 推理 | 不支持 | 接口文档明确标注 |

> 注：`910B` 出现在接口层支持列表（`ASCEND910B_DTYPE_SUPPORT_LIST` 含 BFLOAT16），与文档「Atlas A2 系列支持 BF16」表述相互印证。硬件矩阵的最终权威口径以合集文档构建后的 `aclnnSoftMarginLoss` 页面为准。

## 算子约束限制

1. **输入 dtype 必须一致**：`input_x` 与 `input_y` 数据类型不同直接 `GRAPH_FAILED`；`out` 只能是 FLOAT/FLOAT16/BFLOAT16。
2. **不支持真正的 broadcast（差异 D2）**：接口文档写「shape 需要与 target 满足 broadcast 关系」，`op_api::CheckShape` 也调用 `OP_CHECK_BROADCAST_AND_INFER_SHAPE`；但 arch35 的 `SoftMarginLossTiling::CheckShape()` 要求「rank 相同且所有已知维度逐维相等」，否则报「The known shapes of input_x and input_y must be the same」。于是形如 `self(4,10)` + `target(10,)` 的合法 broadcast 用例在接口层放行、在 tiling 层失败——错误信息层级不一致。建议：把 host 校验放宽为真 broadcast（`atvoss/broadcast` 本身具备 broadcast tiling 能力），或在 `op_api` 层提前拦截并给出与文档一致的提示，同时更新 `docs`。**这是本设计文档中优先级最高的待修项。**
3. **rank ≤ 8**，0 维标量输入允许（`EnsureNotScalar` 归一化）；`reduction != none` 时 `out` 必须是 0 维张量。
4. **仅支持 ND 格式**；`FORMAT_FRACTAL_NZ` 作为 `self` 的存储格式只打 WARNING（「may lead to precision failure」）而不拒绝；`IsRegbase(curArch)` 为真时私有格式（private format）直接报错。
5. **reduction 双枚举**（差异 D1）：接口层 `0/1/2 = none/mean/sum`，算子层 `0/1/2 = none/sum/mean`。任何新增分支必须以字符串 attr 为唯一真源，禁止在两处各自硬编码数字。
6. **`target` 不校验 ±1**：非 ±1 标签会静默计算出「有定义但不是你想要」的损失值；上层（如 torch 适配）必须自行保证语义或改用 `SoftmaxLoss`/`SparseSoftmaxCrossEntropy` 类算子。
7. **空输入行为需固定**：`N=0` 且 `mean` 时期望值为 NaN 还是 0，源码只给出 `Process(NAN)` 线索（见上文【未验证】），必须以 ST 用例锁定。
8. **确定性**：接口文档声明默认确定性实现。若引入原子跨核累加（`workspace` 方案）会破坏这一点，因此 O1 的 pairwise 求和应优先在块内/单核完成。
9. **`DynamicCompileStaticFlag(true)`**：静态与动态 shape 双编译，新增 tiling 分支要保证「shape 未知名」路径同样可编译，否则静态图（GE/图模式）会退化。
10. **workspace 申请只覆盖 none 分支**（差异 D3）：若确认归约分支需要 workspace，需同步 `SetTilingData()` 的条件与 `GetWorkspaceSize` 对外返回值，避免出现「文档说 0 但 kernel 越界写」的隐患。

# 可维可测分析

## 精度标准/性能标准

**精度**（建议值，来源：同类 loss 算子通用口径 + 本算子 fp32 中间量实现；02-25 无任务书，需与维护者最终确认）：

| 场景 | 计算域 | 建议阈值 |
|------|--------|----------|
| FLOAT 逐元素（none） | fp32 vs fp64 golden | `atol = 1e-5`, `rtol = 1e-5` |
| FLOAT16 / BFLOAT16 逐元素 | 输入提升 fp32 计算后回写 | `atol = 1e-3 / 1e-2`, `rtol = 1e-3 / 1e-2` |
| sum（FLOAT，N ≤ 1e6） | fp32 累加 | 相对误差 ≤ `1e-5`；随 N 增长按 `O(sqrt(N)·eps)` 放宽 |
| sum/mean（FP16/ BF16） | fp32 累加 | `rtol = 1e-3`（fp16）/ `1e-2`（bf16） |
| 特殊路径 | `e < 6e-8`（`u==1` select） | 与 `log1p` 参考值逐位一致或误差 ≤ 1 ULP |
| 特殊路径 | `-t*x > 88.72`（`uIsInf` select） | 输出 `+inf`，禁止出现 NaN |
| 特殊路径 | `x` 或 `t` 为 NaN | 输出 NaN |
| 特殊路径 | 空输入 + `mean` | 与 ST 约定一致（见约束 7） |

**必须覆盖的精度用例矩阵**（可直接由脚本按 tilingkey 生成，`3 reduction × 3 dtype × shape` 组合）：

1. shape：`(2,2)`（README 示例）、`(1,)` 与 0 维标量、`(8192,)`（单 tile 尾块）、`(4097,)` 与 `(vectorLen*k+1,)`（专打 `UpdateMask` 尾块）、`(4,10)`+`(10,)`（broadcast，验证差异 D2 的行为）、`(0,)`（空输入）、rank=8。
2. 数值：`t*x` 跨越 `-90 / -1 / 0 / 1 / 90`；全 `-1` 与全 `+1` 标签；含 `inf/NaN` 混合；fp16 接近最大值的累加。
3. 归约：`N` 从 1 到 1e6 扫，观察 fp32 累加误差曲线，作为 O1 的量化输入。
4. 负样本（应报错）：dtype 不一致、rank 不等（若维持当前 host 校验）、`reduction=3`、非 0 维 out、`out` dtype 为 INT32。

**性能**：本算子是访存受限（memory-bound）。以 `none` 分支为例，搬运量为读 `2·N·sizeof(T)` + 写 `N·sizeof(T)`；`sum/mean` 分支写量退化为标量，读量不变。因此

- 期望带宽利用率：`N ≥ 1e6` 时应达到设备理论带宽的 80% 以上；
- 关键指标：`CopyIn/CopyOut` 次数（DAG 融合后应为每 tile 一次）、特殊函数单元占用（`Exp` + `Log` + `Div` 三条指令/元素）、`sum/mean` 的跨核归约轮次（`patternID`）；
- 基线对比：与 `aclnnSigmoidCrossEntropyWithLogits`（同为「逐元素 log-sum-exp 型 loss」）在相同 `N` 下对齐，若明显劣化，优先怀疑 `MemOptCfg<LEVEL_2>` 未生效或 UB tile 长度过小；
- 回归门禁：本任务不引入新的 GM 往返；O1（pairwise 求和）落地后必须给出「精度提升幅度 vs 耗时增加」的成对数据。

> 本文不填写具体实测时延数字：本次交付在工作站上以源码走查方式完成，未在真机运行基准（无 arch35/A2 设备），伪造数字会误导评审。实测列由 `tests/st` 跑出后补充。

**可维护性专项**：

- `PairwiseReduceSumOp` 目前是 `Vec::ReduceSumOp` 的空别名。若确定 pairwise 求和为精度手段，应落地实现并用上面第 3 条误差曲线验证收益（**O1**）。
- `meanVar` 的正确性只依赖通用 `reduceTiling.meanVar`。需要一条 UT 断言「mean = sum / N」在 `N` 非整除（如 4097）时不出现截断（例如 `meanVar` 若按整型倒数计算会偏移）。
- 日志：`SetTilingData()` 的 `OP_LOGI` 已打印 `patternID/loopARCount/loopInnerARCount/tilingKey/reduction/dtype`，评审与现场定位可直接依赖，不建议删除；新增分支时应同步打印。

## 兼容性分析

1. **对外接口兼容**：两段式 `aclnnSoftMarginLossGetWorkspaceSize` + `aclnnSoftMarginLoss` 原型不变，参数顺序 `(self, target, reduction, out, workspaceSize, executor)` 固定；错误码只使用 `ACLNN_ERR_PARAM_NULLPTR(161001)`、`ACLNN_ERR_PARAM_INVALID(161002)`。任何校验放宽（如差异 D2）都属**兼容性增强**，不改变合法调用的结果，故可在小版本发布。
2. **框架适配**：torch_npu 侧 `torch.nn.SoftMarginLoss` 是「`(N,C)` 输入 + int64 类别索引」语义，与本算子（±1 浮点标签）不同。适配层需要 (a) 把索引转成 one-hot 符号标签后逐元素计算，或 (b) 改用类别索引版算子并新增实现。这是最容易造成「结果对不上 PyTorch」的兼容性陷阱，必须在适配文档中显式声明。同时 `reduction` 字符串 `'none'/'mean'/'sum'` → 接口 `int` 的映射要在适配层集中定义，避免各处硬编码。
3. **ONNX / TF**：ONNX 无标准 `SoftMarginLoss` 节点，导出通常落为 `Neg → Mul → Exp → Add → Log → Reduce*` 子图。若走子图分解，则本算子的数值稳定性（补偿式 log1p）优势会在子图里丢失（`Add(1)` 塌陷），需要提示用户保持单算子路径。
4. **前后向一致**：`02-24 SoftMarginLossGrad` 是同一套 log-sigmoid 的损失反向。`SoftMarginLossGrad` 的公式为 `-t / (1 + exp(t*x))`，其数值实现必须与前向共享同一套溢出/塌陷处理约定；两算子的 `reduction` 语义（前向的 `mean` 会引入 1/N 缩放）必须对齐，否则前反向梯度尺度不一致。
5. **多架构一致**：arch35 与 regbase 存量路径对同一输入应给出位级或阈值内一致的结果。差异 D1/D2/D3 都要在两套实现上分别验证；`docs` 中的产品矩阵（BF16 在 910/310p 不支持）需与 `op_api` 的 `GetDtypeSupportListV2` 保持同步，任何一方修改必须同时改另一方。
6. **版本与升级**：`SoftMarginLossTilingData` 完全复用公共结构（`EleBaseTilingData` + `ReduceOpTilingData`），算子私有字段为零。好处是升级 atvoss 公共组件即受益；风险是公共组件的 tiling 策略变化会直接改变本算子性能，因此回归用例要固定 shape 集合，并在 CI 中保留带宽/时延趋势记录。

## 结论与建议改动清单

| 编号 | 项 | 建议 | 优先级 |
|------|----|------|--------|
| D1 | reduction 接口层/算子层枚举反序 | 补注释或统一枚举来源 | 中（易误读，不影响正确性） |
| D2 | 接口文档允许 broadcast，arch35 host 要求 shape 全等 | 放宽 host 校验为真 broadcast 或在 op_api 提前拦截，同步文档 | **高**（用户可见报错行为不一致） |
| D3 | workspace 仅在 `reduction == NONE` 分支申请 | 与归约路径确认是否应为 `!=`；确认 `GetWorkspaceSize` 对外语义 | 高（潜在越界/性能问题） |
| O1 | `PairwiseReduceSumOp` 为空别名 | 落地 pairwise/树形求和或以 UT 证明不需要 | 中 |
| O2 | 空输入 + `mean` 的期望值未定义 | 用 ST 固定为 NaN 或 0 并写入文档 | 中 |
| O3 | `target` 不校验 ±1 | 文档明确；适配层负责语义转换 | 低 |
| O4 | 无 2 月批次任务书 | 补齐 `docs/202602/SoftMarginLoss_task_doc.md` 或在 README 标注口径来源 | 低 |

---

## 附录：证据清单与未验证项

**证据（均为 atomgit.com 上的 `cann/ops-nn@master` / `cann/cann-ops-competitions@master` 文件原文）**

1. `04_tasks/01_community-task-2026/docs/README.md`：「2月份发放的任务」表，`02-25 = SoftMarginLoss`、`02-24 = SoftMarginLossGrad`。
2. `04_tasks/01_community-task-2026/README.md`：目录结构与「`YourTeamName/docs/design.md` 为必选交付」的规定。
3. `loss/soft_margin_loss/op_host/soft_margin_loss_def.cpp`：`input_x/input_y/output_z` 的 dtype 集合与 `reduction` attr 默认 `"mean"`；`AICore().AddConfig("ascend950"/"ascend350")`；四个 Dynamic* flag。
4. `loss/soft_margin_loss/docs/aclnnSoftMarginLoss.md`：产品支持矩阵、两段式函数原型、参数表（0–8 维、非连续 √、BF16 在 910/310p 不支持）、约束说明（确定性、支持空输入、仅 ND、`ln(FLT_MAX)≈88.72` 溢出与 log1p 稳定实现描述）、调用示例（`self/target = (2,2)`，`target = {-1,1,1,-1}`，`reduction = 0`）。
5. `loss/soft_margin_loss/op_kernel/soft_margin_loss.cpp`：`if constexpr` 三分支、`REGISTER_TILING_DEFAULT`、`SetVar<PromoteType,0>(meanVar)`、`Process(NAN)`。
6. `loss/soft_margin_loss/op_kernel/arch35/soft_margin_loss_dag.h`：`SoftMarginLog1pVF` 全部指令序列与注释、三个 DAG 的算子链与 `MemCfg = MemOptCfg<MemLevel::LEVEL_2>`。
7. `loss/soft_margin_loss/op_kernel/arch35/soft_margin_loss_reduce.h`：`PairwiseReduceSumOp` 空继承。
8. `loss/soft_margin_loss/op_kernel/arch35/soft_margin_loss_tiling_key.h`：`SML_SUM/SML_MEAN/SML_TPL_*` 与 `ASCENDC_TPL_ARGS_DECL/SEL`。
9. `loss/soft_margin_loss/op_host/arch35/soft_margin_loss_tiling_arch35.cpp`：`ASCEND_WORKSPACE`、`MAX_SUPPORTED_RANK`、`STR_2_INT`、`CheckShape/SetTilingData/TilingEle/TilingReduce/RunTiling` 全部逻辑与错误文案。
10. `loss/soft_margin_loss/op_api/aclnn_soft_margin_loss.cpp`：`REDUCTION_MEAN_NUM=1/REDUCTION_SUM_NUM=2`、`ASCEND910/ASCEND910B_DTYPE_SUPPORT_LIST`、`CheckDtypeValid/CheckShape/CheckFormat/CheckReduction/CheckParams`。
11. `loss/soft_margin_loss/op_graph/soft_margin_loss_graph_infer.cpp`：`InferDataType4SoftMarginLoss` 输出 dtype 取 `Input(0)`。

**未验证项（诚实声明）**

- 【未验证】regbase/存量路径是否为 toolkit 内置 TBE 实现，以及 `.../opp/built-in/op_impl/ai_core/tbe/impl` 目录是否存在同名算子实现：本机无 Ascend 工具包，未做文件系统核对；结论仅来自 `op_api`/`op_def` 的分流线索。
- 【未验证】`ReduceSch::Process(fillValue)` 与 `reduceTiling.meanVar` 的精确语义（未读 `atvoss/reduce/*`）。
- 【未验证】arch35 设备的 UB/L2 实际容量与 `ElewiseBaseTiling` 选出的 tile 长度：未真机运行。
- 【未验证】差异 D3（workspace 条件）是否构成缺陷：需要设备侧 workspace 使用观测。
- 【未验证】文中精度阈值与带宽利用率 80% 为工程建议值，非任务书强制指标；性能实测列待 `tests/st` 补齐。
- 本次提交仅新增设计文档 `xoljj/docs/design.md`，不含算子代码变更；`docs/aclnn{OpName}.md`、`examples/`、`op_host/`、`op_kernel/`、`CMakeLists.txt`、`README.md` 等「任务完成」级交付物需在实现合入后另行提交。
