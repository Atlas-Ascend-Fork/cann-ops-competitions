# aclblasIsamin 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasIsamin算子开发 任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/iamin/`（arch22 不存在，须新建）、`blas/iamin/arch35/isamin_host.cpp`、`blas/iamax/arch22/`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasIsamin 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/iamin/arch22/`（实现，**该目录需新建**）与 `test/isamin/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度实数向量最小绝对值元素索引能力，
对标 cuBLAS `cublasIsamin`（cuBLAS Library 文档 §2.5.2；Netlib 参考库无 isamin.f）。

## 背景介绍

### 算子功能定位

BLAS Level-1 索引归约：在单精度实数向量中查找绝对值最小元素的**最先出现位置**，返回 1-based 索引。

数学公式：

```
result = argmin_{i=1..n} |x[j]|，并列时取最先出现的 i；j = 1 + (i−1)×incx
result ∈ [1, n]（正常计算）；n = 0 或 incx < 1 时 result = 0
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasIsamin | cublasIsamin | 单精度实数 ISAMIN：最小绝对值元素的 1-based 索引（INT32 输出） |

### 业界对标

- 对标接口：cuBLAS `cublasIsamin(handle, n, x, incx, result)`（cuBLAS Library §2.5.2）。
- Golden 语义来源：**Netlib 无 isamin.f**，golden 由按 cuBLAS 文档语义编写的测试工程内 CPU 参考实现生成
  （任务书 §3.2），输出 1-based 整数索引，`actual == golden` 精确比对。
- 语义差异点（相对仓内既有实现）：
  1. **NaN 语义（cuBLAS 口径，与 std::min 不同）**：首元素为 NaN 时基准置 FLT_MAX，后续元素中的 NaN 跳过
     （比较谓词 `!(absVal != absVal) && absVal < minAbs`），因此 NaN 不会被选为最小元素；
     若全部元素均为 NaN，结果为 1（任务书 §2.1 明示，实现须逐条对齐）；
  2. **并列 tie-break**：并列最小值取最先出现索引（1-based）——归约比较必须**严格小于才更新**；
  3. **incx < 1 quick return**：n=0 或 incx<1 时 result 写 0 并返回成功（不报错）；n<0 才返回 INVALID_VALUE；
  4. **quick return 的 result 写入**：result 为 Device 地址，host 不得直接解引用，
     按 `aclrtMemsetAsync` 写回（仓内 arch35 `isamin_host.cpp` issue #738 注释确认的既有惯例）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/iamin/arch22/`（**新建目录**；`blas/iamin/` 当前仅 arch35）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同算子 arch35 实现 | `blas/iamin/arch35/isamin_{host,kernel,tiling_data}.*` | 校验序（L29-53）/溢出守卫（issue #745）/tiling 规划平移；kernel 从 SIMT 改写为 SIMD |
| **arch22 归约范式（主复用源）** | `blas/iamax/arch22/isamax_{host,kernel,kernel_impl}.*` | arch22 两段式归约骨架（partial + finalize）、块内 ReduceMin/mask 压缩、中间结果及时归约规划（DEAL_TIMES_EACH_CORE_REDUCE） |
| 同目录 arch35 复数兄弟 | `blas/iamin/arch35/icamin_*` | 与 Icamin 同族共享的 (value,index) 成对比较实现参考 |

### 接口规格

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

- 唯一声明位置：`include/cann_ops_blas.h` **L243**（现场核验；任务书 §2.3/§7.1 写 L240，以仓内实际为准）。
  **本次任务不修改公共头**（声明已存在、与 950PR 共用 → 红线 3）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | 向量元素个数，Host 内存 | int | int32 | 任意 int32；n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 或 incx<1 走快速返回（result 写 0，成功） | — |
| x | 输入向量，Device 内存，只读 | const float* | FLOAT32 | n>0 且 incx≥1 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验（含 Inf/NaN，比较语义见 §2.1） | 一维，物理长度 `1+(n−1)*|incx|` |
| incx | x 相邻元素步长，Host 内存 | int | int32 | 任意 int32；incx<1（含 0 与负步长）快速返回（result=0）；`(n−1)*incx` 超 int32 范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #745） | — |
| result | 最小绝对值元素的 1-based 索引，Device 内存，独立输出 | int* | INT32 | 任意 n（含快速返回路径）下为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；正常路径 ∈ [1,n]，快速返回为 0 | shape [1] |

### 数学公式

```
result = argmin_i |x[j]|（首个最小者，1-based）
```

归约运算：逐元素计算 |x| 后取最小值索引；并列取最小索引。输出为离散整数，
IEEE 浮点语义下结果确定，与归约遍历/分块顺序无关（任务书 §3.2）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 实数 ISAMIN：FLOAT32 一维向量、正步长（<1 quick return）、
输出 INT32 1-based 索引；精度为整数索引精确比对（索引错一位即失败）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `result = argmin |x|`（首个最小者）。golden：测试工程内 CPU 参考实现（按 cuBLAS §2.5.2 语义手写循环；Netlib 无 isamin.f），整数索引 `EXPECT_EQ` 精确比对 |
| 2 | 接口原型 | `aclblasStatus_t aclblasIsamin(aclblasHandle_t, int, const float*, int, int*)`，声明于 `include/cann_ops_blas.h` L243（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 输入 **FLOAT32**，输出 **INT32**（任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 isamin，不受影响 |
| 6 | 精度要求 | 整数索引精确比对：`actual == golden`（matched_ratio 1.0 / max_abs_error 0 为形式化表述）；不适用浮点容差（任务书 §3.2） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定。典型 case（实测 CSV 读数）：n=1048576 → ≤21.61us；n=2097152 → ≤22.04us；n=4194304 → ≤26.50us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase 两段式归约（复用 iamax/arch22 isamax 骨架）**：每核块内 (value,index) 成对归约（严格小于更新）→ partial 经 workspace → finalize 取全局最小索引。死路①：arch35 kernel 直接平移（SIMT）；死路②：Abs+ReduceMin 纯向量指令一步到位（ReduceMin 不带索引语义，须自管 index 配对，与 isamax arch22 既有实现同结论） |
| 9 | 评测来源 | 精度 golden：cuBLAS 文档语义 CPU 参考实现（含 NaN/并列/±0 边界）；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V6，对齐 arch35 `isamin_host.cpp` L29-53 实测 + 任务书 §2.4/§2.5，顺序不可调换）**：

| 序 | 条件 | 返回码 / 行为 |
|----|------|--------------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `result == nullptr`（**任意 n，含快速返回路径**） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `n == 0 \|\| incx < 1` | **快速返回**：result 写 0（aclrtMemsetAsync），返回 SUCCESS，不触发 kernel |
| V5 | `n>0 && incx≥1 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `n>0 && incx≥1 && (n−1)*incx 超 int32 范围`（int64 计算，issue #745 双边守卫） | `ACLBLAS_STATUS_INVALID_VALUE` |

> arch35 实测顺序为 handle → n<0 → result 判空 → (n==0‖incx<1) quick return → x 判空 → 跨度守卫，与 V1..V6 一致。

---

# 详细设计（required）

## 算子分析

### 数学公式

`result = argmin_{i} |x[j]|`（首个最小者，1-based）。
确定性：|x| 比较在 IEEE 语义下确定，归约顺序不影响最终索引（严格小于更新保证取最小索引）。

### 支持数据类型

输入 FLOAT32，输出 INT32；中间比较基于 FLOAT32 绝对值，不做精度收窄。
特殊值语义（任务书 §2.1，逐条对齐 cuBLAS）：
- NaN：`!(absVal != absVal)` 为 false → NaN 永不更新基准（不被选为最小）；
- 首元素为 NaN：基准置 FLT_MAX，首个非 NaN 的最小元素胜出；全 NaN 向量结果为 1；
- ±0.0：|−0.0| == |+0.0| == 0.0，并列 → 取最先出现索引；
- Inf：|Inf| = +Inf，只要有任何有限元素即不选 Inf；全 Inf 向量首个 Inf 胜出（index 1，若为首元素）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 | 物理长度 `1+(n−1)*|incx|`；调用方保证存储有效 |
| n = 0 / incx < 1 | 快速返回：result=0、成功、不启动 kernel（含 0 与负步长，**无负步长反向遍历**） |
| n < 0 | 非法：INVALID_VALUE |
| n = 1 | result = 1（除非 NaN 语义边界——首元素即结果） |
| 广播 / 0 维 Tensor | 不涉及（BLAS-1） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx == 1` | 连续读入 → 块内 (value,index) 归约 → partial → finalize | 是（性能基线 base case 均连续） |
| **V_STRIDED** | `incx ≥ 2` | src_gap 步长 DMA 读 → 同上归约（index 为逻辑序 1-based） | 否（精度覆盖） |

数据流（V_CONTIG，复用 `blas/iamax/arch22/isamax` 两段式骨架）：

```
GM x ──DataCopy──▶ UB inBuf
                      │  Abs → UB absBuf（float）
                      │  块内归约：ReduceMin 求块最小值 + mask/标量定位首个最小索引
                      │    （(value,index) 成对比较：严格小于更新，相等保留旧 index）
                      ▼
             workspace（每核 1 组 partial：minValue + minIndex）
                      ▼
        finalize（核 0）：跨核 (value,index) 归约 → GM result（INT32，1-based）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`；每核负责连续逻辑区间，kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4：tiling 不按核数组分配；
  arch35 `CalcIsaminTiling` 同款 perCoreN/remainder 方案）；
- 小 n 单核完成（无跨核同步开销）；归约 1-based 索引 = 逻辑区间偏移 + 块内偏移 + 1。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_CONTIG tileSize 取 4096~8192 float：inBuf/absBuf 双缓冲 + 块归约 workLocal；
- 中间结果及时压缩：单核多轮循环时按固定轮次做一次块归约，partial 恒定一个 block
  （参照 `isamax_kernel_impl.h` DEAL_TIMES_EACH_CORE_REDUCE=63 的规划）；
- 尾块 DataCopyPad 补零——**补零元素 |0|=0 会被误选为最小**：尾块归约必须按有效元素数
  calCount 计算，或给补零位置写 FLT_MAX 哨兵（终态以 S4 实测定案，设计取"哨兵填充"为首选——
  与 Abs/ReduceMin 流水兼容且无逐元素标量分支）。

**tilingkey 规划策略**：tiling 字段分支（incx==1 / ≥2），arch22 现行模式；
Tiling 字段：n、incx、perCoreN、remainder、useCoreNum、tileSize（对齐 arch35 `isamin_tiling_data.h` 命名）。

**数据检测**：按 V1..V6 顺序执行（见需求拆解节）：
- V3（result 判空）先于 V4（quick return）——"任意 n"口径；
- quick return 写 result 用 `aclrtMemsetAsync`（issue #738 惯例，Device 地址不得 host 解引用）；
- V6 用 int64 计算跨度、双边守卫（对齐 arch35 `lastElement < 0 || lastElement > INT32_MAX`）。

**其他 host 侧约定**：
- workspace：跨核 partial 需要各核写独立槽（写冲突规避），大小按 arch22 isamax 既有布局；
- 同步：异步下发，host 不同步；读回 result 前由调用方 `aclrtSynchronizeStream`；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策与 partial 汇总结果。

#### kernel 侧设计

**Init/Process 两阶段**（V_CONTIG）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(inBuf/absBuf/reduceBuf)
Process（每核）:
   ┌── for each tile (双缓冲) ───────────────────────────────────┐
   │  MTE2: DataCopy inBuf ← GM x[off..]                          │
   │  MTE2→V: SetFlag/WaitFlag                                    │
   │  V:    Abs(absBuf, inBuf, cnt)                               │
   │        块最小值 + 首索引（严格小于更新；NaN 跳过；哨兵护尾）    │
   │        与本核累计 (minValue,minIndex) 成对比较更新             │
   │  V→MTE2: SetFlag/WaitFlag                                    │
   └──────────────────────────────────────────────────────────────┘
   └── 本核结束：写 workspace[coreIdx] = (minValue, minIndex+1)
finalize（核 0）:
   读 workspace 全部 partial → 跨核 (value,index) 归约（严格小于）→ GM result
```

**块内"首个最小索引"实现要点**：
- ReduceMin 指令只给值不给索引 → 值经 ReduceMin 后，用相等掩码（mask 扫描）定位块内
  **第一个**等于最小值的位置（arch22 isamax 的 REDUCE_*_RST_MASK 掩码范式同源，min 方向自实现）；
- 比较谓词统一 `absVal < minAbs`（严格小于）+ `!(absVal != absVal)`（NaN 跳过）；
  **不得**用 `<=`（会破坏 tie-break）或 `std::isless` 类库函数（NaN 语义不同）。

**步长路径（V_STRIDED）**：`asc_copy_gm2ub_align(src_gap=(incx−1)*4)`；块内索引仍按逻辑序累计
（tile 起始逻辑号 + 块内偏移），与物理 DMA 基址解耦。

**workspace 布局**：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | numBlocks × 8B | 每核 partial（minValue float + minIndex int32，8B 对齐） |

**同步事件配对**：MTE2→V、V→MTE2 显式 SetFlag/WaitFlag；finalize 与各核 partial 写入按单 kernel 内栅栏（S4 与两 kernel 方案实测取舍）。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 大 n 满核并行，partial 槽位写无冲突，核内 finalize 一次 launch |
| 双缓冲流水 | inBuf 2 槽，MTE2 与 V 重叠 |
| Abs+归约融合 | Abs 结果直接进块归约，无额外 GM 访问 |
| 掩码定位首索引 | ReduceMin + 掩码扫描替代逐元素标量循环（热路径全向量指令） |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc（workspace 框架管理） |

**未采用的路线及原因**：
- **arch35 kernel 直接平移**：SIMT 线程模型 + arch35 tiling 传递方式——死路；
- **纯 ReduceMin 无索引方案**：丢失 tie-break 与 NaN 语义控制权，须二次扫描补索引（等价于掩码方案但多一遍），不采用；
- **负步长反向遍历**：incx<1 即 quick return（result=0）——不实现，注释显式说明；
- **FP64 比较基准**：golden 为 float32 比较，引入 FP64 改变并列/边界判定——不用。

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
| 参数合法性与校验层级 | handle nullptr → HANDLE_IS_NULLPTR；n<0、result nullptr（任意 n）、n>0 且 incx≥1 时 x nullptr、跨度溢出 → INVALID_VALUE；n=0 或 incx<1 → 快速返回（result=0，成功）；元素值不校验 |
| 非连续 Tensor 支持 | 支持：incx≥1 跳步访问；incx<1 不进入计算（result=0 快速返回），无负步长遍历 |
| broadcast 规则 | 不涉及（BLAS-1） |
| dynamic shape 要求 | 不要求：n/incx 运行时入参，host tiling 按运行时 n 计算 |
| 原地与视图语义 | result 为独立输出，不复用输入内存；x 只读 |
| 确定性计算要求 | 并列取最先出现索引；归约划分顺序不影响结果（严格小于更新语义保证） |
| 空 Tensor 处理 | 等价 n=0：不访问数据、不启动 kernel、result=0 返回成功 |
| 0 维 Tensor 处理 | 不涉及 |
| 异步执行 | 经 handle 绑定 stream 异步下发；读取输出前须 aclrtSynchronizeStream |

**已知限制（如实标注）**：
1. 不支持负步长反向遍历（incx<1 快速返回 result=0，对齐 cuBLAS）；
2. NaN/并列语义按 cuBLAS 口径实现，与 std 库语义不同（注释显式标注防误改）；
3. 步长路径不参与性能判定；
4. 尾块哨兵填充（FLT_MAX）设计下，全 FLT_MAX 输入的边界 case 由精度用例覆盖验证。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | 测试工程内 CPU 参考实现（按 cuBLAS §2.5.2 语义手写 cblas 风格循环，直接输出 1-based 索引；Netlib 无 isamin.f） | 任务书 §3.2 |
| 比对方式 | 整数索引精确比对：`actual == golden`，索引错位即失败 | 任务书 §3.2 |
| 浮点阈值 | 不适用于输出比对（matched_ratio 1.0 / max_abs_error 0 为形式化表述）；x 的 FLOAT32 值仅为生成域 | 任务书 §3.2 |
| NaN 语义 | 首元素 NaN 基准置 FLT_MAX、后续 NaN 跳过、全 NaN → result=1；golden 与实现同谓词 | 任务书 §2.1/§3.2 |
| 并列语义 | 严格小于更新，并列取最小索引（含 ±0） | 任务书 §2.1 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定） | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{isamin_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`，共 1200 条（CSV 实测 1201 行含表头）；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/isamin/arch22/`（CSV 列格式对齐 `test/isamin/isamin_param.h`：n / incx / x 填充模式，x 列写 NULLPTR 表示空指针负向用例），并在 `test/isamin/CMakeLists.txt` 补齐 arch22 GTest 注册（任务书 §2.2）；
- 用例分层（§3.5）：边界域——n=0、n=1、incx<1（result=0）、并列最小值（重复元素与 ±0）、NaN 样本；负向域——handle nullptr、result nullptr（含 n=0 路径）、n<0、x nullptr、步长溢出；
- incx 覆盖 {1, 2, 7, 16}（incx<1 走快速返回）；溢出负向含 n=4 且 incx=INT_MAX、n=2^30 且 incx=8；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|incx|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasIsamin` 声明已存在于 `include/cann_ops_blas.h`（L243），本次不改公共头；
- **新建目录 `blas/iamin/arch22/`**：不影响 arch35；CMake 构建 `SOC_ARCH_DIRS` 按 SOC 选择目录，ascend950 构建行为不变；
- **对 arch22 同族 isamax 的影响**：无。复用其归约范式但为独立文件与独立接口，无共享可变状态；
- **测试构建影响**：`test/isamin/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支保持不变；
- **文档影响**：`blas/iamin/README.md`（或族 README）产品支持表新增 aclblasIsamin「A2/A3：支持」条目，并同步约束说明（n<0 报错 / n=0 或 incx<1 快速返回 result=0 / NaN cuBLAS 语义）。
