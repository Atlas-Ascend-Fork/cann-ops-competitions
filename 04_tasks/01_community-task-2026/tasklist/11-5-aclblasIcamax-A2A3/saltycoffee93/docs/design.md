# aclblasIcamax 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasIcamax 算子开发任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/iamax/`、`blas/iamax/arch22/isamax_*`（实数兄弟，arch22 归约骨架主复用源）、`blas/iamax/arch35/`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasIcamax 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/iamax/arch22/`（实现，**该目录已存在**——同目录已有实数 `aclblasIsamax`，并列加文件）
与 `test/iamax/icamax/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数向量最大模元素索引能力，
对标 cuBLAS `cublasIcamax`（cuBLAS Library 文档；语义参考 Netlib `icamax.f`）。

## 背景介绍

### 算子功能定位

BLAS Level-1 复数索引归约：在单精度复数向量中查找最大"模"（1-范数口径）元素的索引，返回 1-based 索引。

数学公式：

```
result = argmax_{i=1..n} ( |Re(x[k])| + |Im(x[k])| )，相同最大值取最小索引；k = 1 + (i−1)×incx
result ∈ [1, n]（正常计算）；n = 0 或 incx ≤ 0 时 result = 0
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasIcamax | cublasIcamax | 单精度复数 ICAMAX：最大模（\|Re\|+\|Im\|，SCABS1 口径）元素的 1-based 索引（INT32 输出） |

### 业界对标

- 对标接口：cuBLAS `cublasIcamax(handle, n, x, incx, result)`（"(smallest) index" 语义）。
- Golden 语义来源：Netlib BLAS `icamax.f`（https://www.netlib.org/blas/icamax.f，SCABS1 + 严格大于更新）。
- 语义差异点（相对仓内既有实现）：
  1. **复数"模"为 1-范数口径**：`|x[k]| := |Re(x[k])| + |Im(x[k])|`（Netlib `SCABS1` 语义），
     **不是欧氏模** `sqrt(Re²+Im²)`——实现与 golden 均须使用该口径（任务书 §2.1-2/§7.6 明令）；
  2. **tie-break：相同最大值取最小索引**——归约比较必须**严格大于才更新**
     （对齐 Netlib `icamax.f` 与 cuBLAS "(smallest) index"，任务书 §2.1-1）；
  3. **特殊值语义**：含 Inf 元素时首个 Inf 所在元素胜出；NaN 元素在严格大于比较下不会胜出
     （首元素含 NaN 时 Netlib 语义下结果停留为索引 1）；特殊值用例以 cblas golden 精确一致为判定（任务书 §3.2）；
  4. **quick return 口径**：n=0 或 incx ≤ 0（含 0 与负步长）时不触发归约，result 写 0 并返回成功；
     **不支持负步长反向遍历**（Netlib 1993 年起 incx ≤ 0 直接返回 0，任务书 §2.1-4）；
  5. **golden 索引基准转换**：cblas 返回 0-based 索引，golden 侧 +1 转换为 1-based 后与 NPU 输出精确比对
     （参照仓内 `test/isamax/isamax_golden.h` 的 aclblasIsamax golden 范式，任务书 §3.2）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/iamax/arch22/`（**已存在**，含实数 isamax 完整归约实现 → **并列加文件，不新建族目录**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| **同目录 arch22 实数实现（主复用源）** | `blas/iamax/arch22/isamax_{host,kernel,kernel_impl}.*` | arch22 两段式归约骨架（partial + finalize）、块归约与掩码定位、中间结果及时压缩规划（DEAL_TIMES_EACH_CORE_REDUCE=63）、host 校验序——实数骨架加复数解包 |
| 同算子 arch35 实现 | `blas/iamax/arch35/` | 复数路径语义（(value,index) 成对比较）与 tiling 规划参考；kernel 不直接平移（SIMT） |
| 交错复数处理范式 | `blas/asum/arch35/scasum_*`、ICAMIN 设计（同批次） | 交错存储免物理解包：2n float 逐分量 Abs + 成对 Add 得 1-范数模 |

### 接口规格

```cpp
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

- 唯一声明位置：`include/cann_ops_blas.h` **L242**（现场核验；任务书 §2.1-7/§7.3 写 L239，
  L241 为 aclblasIsamax——以仓内实际为准）。**本次任务不修改公共头**
  （声明已存在、与 950PR 共用，签名逐参数对齐同族 `aclblasIsamax` → 红线 3）。
- `aclblasComplex` 定义于 `include/cann_ops_blas_common.h`：实/虚部各 float32，交错存储，n 个复数元素对应 2n 个 float。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | x 的复数元素个数，Host 内存 | int | int32 | 任意 int32；n=0 走 quick return（result=0，成功）；n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE` | — |
| x | 复数向量，Device 内存，只读 | const aclblasComplex* | COMPLEX64 | n>0 且 incx≥1 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验（含 Inf/NaN，比较语义见上） | 一维 [n]，物理长度 `1+(n−1)*incx`（incx≥1 时） |
| incx | x 相邻元素步长（复数元素为单位），Host 内存 | int | int32 | 任意 int32；incx ≤ 0（含 0 与负步长）quick return（result=0，成功，非报错用例）；`(n−1)*incx` 超 int32 范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #745） | — |
| result | 最大模元素的 1-based 索引（相同最大值取最小索引），Device 内存，独立输出 | int* | INT32 | 任意 n（含快速返回路径）下为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；正常路径 ∈ [1,n]，quick return 为 0 | 单值标量 |

### 数学公式

```
result = argmax_i ( |Re(x[k])| + |Im(x[k])| )（首个最大者，1-based，严格大于更新）
```

归约运算：逐元素计算 1-范数模（SCABS1）后取最大值索引；并列取最小索引。输出为离散整数，
比较在 IEEE 语义下确定，与归约遍历/分块顺序无关（任务书 §3.2）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 复数 ICAMAX：COMPLEX64 一维向量、正步长（≤0 quick return）、
输出 INT32 1-based 索引；精度为整数索引精确比对（索引错一位即失败）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `result = argmax (|Re|+|Im|)`（SCABS1 口径，首个最大者）。golden 白嫖 cblas（Netlib `icamax`，0-based → +1 转 1-based，参照 `test/isamax/isamax_golden.h` 范式），整数索引精确比对 |
| 2 | 接口原型 | `aclblasStatus_t aclblasIcamax(aclblasHandle_t, int, const aclblasComplex*, int, int*)`，声明于 `include/cann_ops_blas.h` L242（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 输入 **COMPLEX64**（分量 FLOAT32），输出 **INT32**（任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 实现，不受影响 |
| 6 | 精度要求 | 整数索引精确比对：`actual == golden`（CSV 不写容差列，精确相等模式，对齐仓内 copy 族 EXACT 惯例）；任一用例索引 ≠ golden 即失败（任务书 §3.2） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定，COMPLEX64 输入场景。典型 case（实测 CSV 读数）：n=1048576 → ≤21.82us；n=2097152 → ≤24.55us；n=4194304 → ≤34.54us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase 两段式归约（复用同目录 isamax arch22 骨架 + 复数解包）**：交错数据按 2n float 读入 → 逐分量 Abs → 成对 Add 得 SCABS1 模 → 块内 (value,index) 归约（严格大于更新）→ partial → finalize。死路①：arch35 kernel 直接平移（SIMT）；死路②：欧氏模（口径错误）；死路③：纯 ReduceMax 无索引（丢失 tie-break/NaN 控制权——与同目录 isamax 实现同结论） |
| 9 | 评测来源 | 精度 golden：cblas（Netlib `icamax`，+1 转 1-based）；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V6，对齐任务书 §2.4/§2.5 与同族 isamax 校验序，顺序不可调换）**：

| 序 | 条件 | 返回码 / 行为 |
|----|------|--------------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `result == nullptr`（**任意 n，含快速返回路径**，CSV result 列写 NULLPTR 覆盖） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `n == 0 \|\| incx <= 0` | **快速返回**：result 写 0（aclrtMemsetAsync），返回 SUCCESS，不触发归约（注意：incx≤0 是**合法 quick return**，非负向报错用例——与 Icamin 的 incx<1 口径一致，与 rot 族"负步长报错"不同） |
| V5 | `n>0 && incx≥1 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `n>0 && incx≥1 && (n−1)*incx 超 int32 范围`（int64 计算，issue #745；仅正步长，无负下界需求，保留 int64 全程计算） | `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 详细设计（required）

## 算子分析

### 数学公式

`result = argmax_i (|Re(x[k])| + |Im(x[k])|)`（首个最大者，1-based，严格大于更新）。
确定性：SCABS1 模的比较在 IEEE 语义下确定，严格大于更新保证并列取最小索引。

### 支持数据类型

输入 COMPLEX64（实/虚部各 FLOAT32，交错存储），输出 INT32；中间计算保持 FLOAT32。
特殊值语义（任务书 §3.2，逐条对齐 Netlib icamax / cblas golden）：
- 严格大于更新：`modVal > bestMod` 才更新索引；相等不更新（取最小索引）；
- Inf：首个 Inf 模元素胜出（后续相等 Inf 不更新）；
- NaN：`modVal > bestMod` 对 NaN 恒 false → NaN 不会胜出；**首元素含 NaN 时结果停留为索引 1**（golden 同谓词）；
- ±0.0：零模并列（全零向量 → result=1）；
- 常量向量（全部元素同模）→ result=1（并列取最小索引，§3.5 覆盖项）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 [n] | 物理长度 `1+(n−1)*incx` 个复数元素（2n float 起算）；调用方保证存储有效 |
| n = 0 / incx ≤ 0 | quick return：result=0、成功、不启动 kernel（含 0 与负步长，**无负步长遍历**） |
| n < 0 | 非法：INVALID_VALUE |
| n = 1 | result = 1（含首元素 NaN 边界） |
| 广播 / 0 维 Tensor | 不涉及（BLAS-1） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx == 1` | 连续读入（2n float）→ 解包/逐分量 Abs → 成对 Add 得模 → 块内 (value,index) 归约 → partial → finalize | 是（性能基线 base case 均连续） |
| **V_STRIDED** | `incx ≥ 2` | src_gap 步长 DMA（8B/复数元素）读 → 同上归约（index 为逻辑序 1-based） | 否（精度覆盖） |

数据流（V_CONTIG，复用同目录 isamax arch22 两段式骨架）：

```
GM x(2n float) ──DataCopy──▶ UB inBuf
                                │ Abs（逐分量，交错数据免物理解包）
                                ▼
                            UB absBuf（2n float 绝对值）
                                │ 成对 Add：mod[i] = |Re_i| + |Im_i|（SCABS1）
                                ▼
                            UB modBuf（n float 模）
                                │ 块内归约：模最大值 + 首索引（严格大于更新；NaN 不胜出）
                                ▼
             workspace（每核 1 组 partial：maxMod + maxIndex）
                                ▼
        finalize（核 0）：跨核 (value,index) 归约 → GM result（INT32，1-based）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`；每核负责连续逻辑区间，kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4：tiling 不按核数组分配；
  同目录 isamax host 既有方案）；
- 1-based 索引 = 逻辑区间偏移 + 块内偏移 + 1；小 n 单核完成。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_CONTIG tileSize 按 2n float 计：inBuf 双缓冲 + absBuf + modBuf + 归约 workLocal，
  tile 取 2048~4096 复数元素量级（复数路径 UB 占用约为实数 isamax 2 倍，tile 相应减半）；
- 中间结果及时压缩：单核多轮循环时按固定轮次做一次块归约（isamax arch22 的
  DEAL_TIMES_EACH_CORE_REDUCE=63 规划直接复用），partial 恒定一个 block；
- 尾块处理：补零位置必须写负哨兵（如 -FLT_MAX）或按 calCount 有效归约——|0|=0 会被非零模
  元素覆盖，但全零输入尾块补零会与真实零并列（取最小索引语义下补零 index 更大不破坏结果；
  设计仍取"哨兵/有效计数"双保险，S4 实测定案首选）。

**tilingkey 规划策略**：tiling 字段分支（incx==1 / ≥2）；
Tiling 字段：n、incx、perCoreN、remainder、useCoreNum、tileSize（对齐 arch35 同族命名）。

**数据检测**：按 V1..V6 顺序执行（见需求拆解节）：
- V3（result 判空）先于 V4（quick return）；
- quick return 写 result 用 `aclrtMemsetAsync`（issue #738 惯例，Device 地址不得 host 解引用）；
- V6 用 int64 计算跨度。

**其他 host 侧约定**：
- workspace：跨核 partial 各核独立槽（float value + int32 index，与 isamax 布局一致）；
- 同步：异步下发，host 不同步；读回 result 前由调用方 `aclrtSynchronizeStream`；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策与 partial 汇总结果。

#### kernel 侧设计

**Init/Process 两阶段**（V_CONTIG）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(inBuf/absBuf/modBuf/reduceBuf)
Process（每核）:
   ┌── for each tile (双缓冲) ────────────────────────────────────┐
   │  MTE2: DataCopy inBuf ← GM x（2×cnt float，交错）              │
   │  MTE2→V: SetFlag/WaitFlag                                     │
   │  V:    Abs(absBuf, inBuf, 2*cnt)                              │
   │        成对 Add → modBuf（cnt 个 SCABS1 模）                   │
   │        块最大值 + 首索引（严格大于更新；NaN 不胜出）             │
   │        与本核累计 (maxMod, maxIndex) 成对比较更新               │
   │  V→MTE2: SetFlag/WaitFlag                                     │
   └───────────────────────────────────────────────────────────────┘
   └── 本核结束：写 workspace[coreIdx] = (maxMod, maxIndex+1)
finalize（核 0）:
   读 workspace 全部 partial → 跨核 (value,index) 归约（严格大于）→ GM result
```

**复数解包要点**：交错存储无需物理解包——对 2n float 逐分量 Abs 后，相邻成对（偶位=Re、奇位=Im）
Add 即得每复数元素的 SCABS1 模；成对 Add 用向量指令 stride-2 语义或先偶/奇拆分再 Add（S4 实测定案）。

**块内"首个最大索引"实现要点**：
- ReduceMax 只给值不给索引 → 值经 ReduceMax 后，用相等掩码定位块内**第一个**等于最大值的位置
  （同目录 `isamax_kernel_impl.h` 的 REDUCE_MAX_RST_MASK 掩码范式直接复用，本任务方向为 max、现成）；
- 比较谓词统一 `modVal > bestMod`（严格大于）；**不得**用 `>=`（破坏 tie-break）。

**步长路径（V_STRIDED）**：`asc_copy_gm2ub_align(src_gap=(incx−1)*8)`（复数元素 8B burst）；
块内索引按逻辑序累计，与物理 DMA 基址解耦；无负步长分支（incx≤0 已在 host quick return）。

**workspace 布局**：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | numBlocks × 8B | 每核 partial（maxMod float + maxIndex int32，8B 对齐） |

**同步事件配对**：MTE2→V、V→MTE2 显式 SetFlag/WaitFlag；finalize 与各核 partial 写入按单 kernel 内栅栏（S4 与两 kernel 方案实测取舍）。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 大 n 满核并行 + 单 kernel finalize，一次 launch（同目录 isamax 既有形态） |
| 双缓冲流水 | inBuf 2 槽，MTE2 与 V 重叠 |
| 免解包复数处理 | 交错数据逐分量 Abs + 成对 Add，无中间重排 GM 访问 |
| 掩码定位首索引 | ReduceMax + REDUCE_MAX_RST_MASK 掩码扫描（实数版现成范式），热路径全向量指令 |
| 中间结果及时压缩 | 每 63 轮压缩 partial，UB 占用恒定 |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc（workspace 框架管理） |

**未采用的路线及原因**：
- **arch35 kernel 直接平移**：SIMT 线程模型——死路；
- **欧氏模 sqrt(Re²+Im²)**：口径错误（任务书 §2.1-2/§7.6 明令禁止），golden 按 SCABS1——死路；
- **纯 ReduceMax 无索引方案**：丢失 tie-break 与 NaN 语义控制权，需二次扫描补索引——不采用；
- **负步长反向遍历**：incx≤0 即 quick return（Netlib 1993 起）——不实现；
- **FP64 比较基准**：改变并列/边界判定——不用。

---

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **支持**（本任务目标） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **支持**（本任务目标） |
| Ascend 950PR / Ascend 950DT | 支持（由 arch35 既有实现提供，本次不改） |

---

## 算子约束限制

逐条对齐任务书 §2.5：

| 约束项 | 结论 |
|--------|------|
| 参数合法性 | n≥0（n<0 → INVALID_VALUE）；n=0 或 incx≤0 → quick return（result=0，成功）；x/result/handle 不可为 nullptr（result 判空任意 n 生效）；跨度溢出（issue #745）→ INVALID_VALUE；元素值不校验 |
| 非连续 Tensor 支持 | 不要求额外 padding 场景：步长由 incx≥1 表达；incx≤0 不遍历数据 |
| broadcast 规则 | 不涉及（单向量归约到标量索引） |
| dynamic shape 要求 | 不要求：n 运行时入参 |
| 原地与视图语义 | 不涉及原地更新；result 为独立输出标量 |
| 确定性计算要求 | 不要求归约划分顺序固定，但结果必须满足"首个最大者（最小索引）"语义，与 golden 精确一致 |
| 空 Tensor 与 0 维处理 | n=0 合法 quick return，result 写 0；incx≤0 同理 |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream；读回 result 前须同步 stream |

**已知限制（如实标注）**：
1. 不支持负步长反向遍历（incx≤0 quick return result=0，对齐 cuBLAS/Netlib）；
2. NaN/并列语义按 Netlib 严格大于口径实现，与 std 库语义不同（注释显式标注防误改）；
3. 步长路径不参与性能判定；
4. 复数路径 UB 占用约为实数 isamax 2 倍，大 n 下 tile 轮次增加（中间压缩已规划，性能以 S4 实测为准）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | cblas（Netlib `icamax`）单标杆；cblas 返回 0-based，golden 侧 +1 转 1-based（参照 `test/isamax/isamax_golden.h` 范式） | 任务书 §3.2 |
| 比对方式 | 整数索引精确相等（bit-exact）：索引 ≠ golden 即失败；CSV 不写容差列，精确相等模式判定（对齐仓内 copy 族 EXACT 惯例） | 任务书 §3.2 |
| 浮点阈值 | 不适用于输出比对；输入分量按 FLOAT32 档生成域约束 | 任务书 §3.2 |
| 模口径 | \|Re\|+\|Im\|（1-范数 / SCABS1），实现与 golden 均不得使用欧氏模 | 任务书 §2.1-2/§3.2 |
| 特殊值语义 | 严格大于更新：首个 Inf 胜出；NaN 不胜出（首元素含 NaN → 索引 1）；以 cblas golden 精确一致为判定 | 任务书 §3.2 |
| quick return 判定 | incx≤0 用例不生成 cblas 归约 golden，直接按期望 result=0 比对 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 输入场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{icamax_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`，共 1200 条（CSV 实测 1201 行含表头）；
- CSV 列格式对齐仓内 `test/isamax/isamax_param.h`（n / incx / x 填充模式），参照同族归约惯例扩展 result 列标记 result 空指针负向用例（"NULLPTR"）；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/iamax/icamax/arch22/`，并在 `test/iamax/icamax/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：n ∈ {0, 1, 2, 3, 7, 2 的幂及 ±1, …, ≥2^20}；incx ∈ {1, 2, 3}（正常）与 {0, −1, −2, −3}（合法 quick return，期望 SUCCESS 且 result=0，**非负向报错用例**）；填充含全零（并列最小索引语义）、常量（全部同模 → 索引 1）、正负交替、极端值、Inf/NaN；负向含 n<0（INVALID_VALUE）、x nullptr、result nullptr；实部/虚部独立采样；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]，实/虚部独立采样）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|incx|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasIcamax` 声明已存在于 `include/cann_ops_blas.h`（L242），本次不改公共头；
- **对 arch35 影响**：无。`blas/iamax/arch35/` 保持不变，按 SOC 选目录编译互不覆盖；
- **对 arch22 同目录 isamax 的影响**：无。同目录并列新文件（aclblasIcamax vs aclblasIsamax），
  接口独立、无共享可变状态；复用其 kernel_impl 归约范式通过新增文件实现，不修改既有文件（若复用需提取公共
  头文件，须评估对 isamax 既有 GTest 的影响并在自测报告说明）；
- **测试构建影响**：`test/iamax/icamax/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支保持不变；
- **文档影响**：`blas/iamax/README.md` 产品支持表新增 aclblasIcamax「Atlas A2/A3：支持」条目，并同步约束说明
  （n<0 报错 / n=0 或 incx≤0 quick return result=0 / 1-范数 SCABS1 口径 / 严格大于 tie-break 语义）。
