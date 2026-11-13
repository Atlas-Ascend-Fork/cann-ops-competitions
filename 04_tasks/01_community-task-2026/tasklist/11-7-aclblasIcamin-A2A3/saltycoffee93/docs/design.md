# aclblasIcamin 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasIcamin 算子开发任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/iamin/`（arch22 不存在，须新建）、`blas/iamin/arch35/icamin_*`、`blas/iamax/arch22/`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasIcamin 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/iamin/arch22/`（实现，**该目录需新建**，与 aclblasIsamin 同目录并列）
与 `test/iamin/icamin/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数向量最小模元素索引能力，
对标 cuBLAS `cublasIcamin`（cuBLAS Library 文档；Netlib 无 icamin 例程）。

## 背景介绍

### 算子功能定位

BLAS Level-1 复数索引归约：在单精度复数向量中查找最小"模"（1-范数口径）元素的最先出现位置，返回 1-based 索引。

数学公式：

```
result = argmin_{i=1..n} ( |Re(x[k])| + |Im(x[k])| )，并列时取最小索引；k = 1 + (i−1)×incx
result ∈ [1, n]（正常计算）；n = 0 或 incx < 1 时 result = 0
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasIcamin | cublasIcamin | 单精度复数 ICAMIN：最小模（\|Re\|+\|Im\|）元素的 1-based 索引（INT32 输出） |

### 业界对标

- 对标接口：cuBLAS `cublasIcamin(handle, n, x, incx, result)`。
- Golden 语义来源：**Netlib 无 icamin 例程**——golden 由测试工程内 cblas 风格 CPU 参考循环生成
  （按 |Re|+|Im| 逐元素比较、严格小于更新索引，任务书 §3.2），整数索引 bit-exact 比对。
- 语义差异点（相对仓内既有实现）：
  1. **复数"模"为 1-范数口径**：`|x| := |Re| + |Im|`（BLAS icamin 惯例），**不是欧氏模**
     `sqrt(Re²+Im²)`——实现与 golden 均须使用该口径（任务书 §2.1-2）；
  2. **NaN 语义（本算子与 Isamin 不同，逐字对齐任务书 §3.2）**：golden 以严格小于
     （`absVal < bestVal`）更新索引，NaN 与任何值比较均为 false，故 NaN 不会被选为最小元素；
     **若首元素为 NaN，基准即为 NaN，后续任何元素都不小于它，结果为 1（不置 FLT_MAX）**——
     与 Isamin 的"首元素 NaN 基准置 FLT_MAX"处理不同，实现不得混淆；
  3. **incx < 1 quick return**：n=0 或 incx<1（含 0 与负步长）时不触发 kernel，result 写 0 并返回成功；
     n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`（对齐同族实数 aclblasIsamin 约束，任务书 §2.1-4）；
  4. **quick return 的 result 写入**：result 为 Device 地址，host 不得直接解引用，
     按 `aclrtMemsetAsync` 写回（仓内 arch35 isamin_host.cpp issue #738 注释确认的惯例）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/iamin/arch22/`（**新建目录**；若 Isamin 任务先行已建目录则并列加文件）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同目录 arch35 双实现 | `blas/iamin/arch35/{isamin,icamin}_{host,kernel,tiling_data}.*` | icamin 校验序/复数 (value,index) 成对比较/tiling 规划平移；kernel 从 SIMT 改写为 SIMD |
| **arch22 归约范式（主复用源）** | `blas/iamax/arch22/isamax_{host,kernel,kernel_impl}.*` | arch22 两段式归约骨架（partial + finalize）、块归约与掩码定位、中间结果及时压缩规划 |
| 同目录 arch22 实数兄弟 | `blas/iamin/arch22/isamin_*`（Isamin 任务交付） | 实数版骨架直接加复数解包：交错存储按 2n float 读入，逐分量 Abs 后成对求和得 1-范数模 |

### 接口规格

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

- 唯一声明位置：`include/cann_ops_blas.h` **L244**（现场核验；任务书 §2.1-6/§7.3 写 L241，以仓内实际为准）。
  **本次任务不修改公共头**（声明已存在、与 950PR 共用 → 红线 3）。
- `aclblasComplex` 定义于 `include/cann_ops_blas_common.h`：实/虚部各 float32，交错存储，n 个复数元素对应 2n 个 float。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | x 的复数元素个数，Host 内存 | int | int32 | 任意 int32；n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 走 quick return（result=0，成功） | — |
| x | 复数向量，Device 内存，只读 | const aclblasComplex* | COMPLEX64 | n>0 且 incx≥1 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验（含 Inf/NaN，比较语义见上） | 一维 [n]，物理长度 `1+(n−1)*|incx|` 个复数元素 |
| incx | x 相邻元素步长（复数元素为单位），Host 内存 | int | int32 | 任意 int32；incx<1（含 0 与负步长）quick return（result=0，成功）；`(n−1)*|incx|` 超 int32 范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #745） | — |
| result | 最小模元素的 1-based 索引，Device 内存，独立输出 | int* | INT32 | 任意 n（含快速返回路径）下为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；正常路径 ∈ [1,n]，quick return 为 0 | 单值标量 |

### 数学公式

```
result = argmin_i ( |Re(x[k])| + |Im(x[k])| )（首个最小者，1-based）
```

归约运算：逐元素计算 1-范数模后取最小值索引；并列取最小索引。输出为离散整数，
比较在 IEEE 语义下确定，与归约遍历/分块顺序无关（任务书 §3.2）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 复数 ICAMIN：COMPLEX64 一维向量、正步长（<1 quick return）、
输出 INT32 1-based 索引；精度为整数索引精确比对（索引错一位即失败）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `result = argmin (|Re|+|Im|)`（首个最小者）。golden：测试工程内 cblas 风格 CPU 参考循环（严格小于更新；无 Netlib icamin 可白嫖），整数索引 `EXPECT_EQ` 精确比对 |
| 2 | 接口原型 | `aclblasStatus_t aclblasIcamin(aclblasHandle_t, int, const aclblasComplex*, int, int*)`，声明于 `include/cann_ops_blas.h` L244（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 输入 **COMPLEX64**（分量 FLOAT32），输出 **INT32**（任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 icamin，不受影响 |
| 6 | 精度要求 | 整数索引精确比对：`actual == golden`（bit-exact，任务书 §3.2）；|Re|+\|Im\| 比较在 IEEE 语义下确定，归约顺序不影响结果 |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定，COMPLEX64 输入场景。典型 case（实测 CSV 读数）：n=1048576 → ≤22.70us；n=2097152 → ≤27.03us；n=4194304 → ≤45.76us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase 两段式归约（复用 iamax/arch22 骨架 + 实数 isamin arch22 骨架加复数解包）**：交错数据按 2n float 读入 → 逐分量 Abs → 成对 Add 得模 → 块内 (value,index) 归约（严格小于更新）→ partial → finalize。死路①：arch35 kernel 直接平移（SIMT）；死路②：欧氏模（口径错误，判据即失败）；死路③：纯 ReduceMin 无索引（丢失 tie-break/NaN 控制权） |
| 9 | 评测来源 | 精度 golden：cuBLAS 语义 CPU 参考实现（含 NaN/并列/±0/全零边界）；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V6，对齐任务书 §2.4/§2.5 与同族 arch35 校验序，顺序不可调换）**：

| 序 | 条件 | 返回码 / 行为 |
|----|------|--------------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `result == nullptr`（**任意 n，含快速返回路径**，CSV result 列写 NULLPTR 覆盖） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `n == 0 \|\| incx < 1` | **快速返回**：result 写 0（aclrtMemsetAsync），返回 SUCCESS，不触发 kernel |
| V5 | `n>0 && incx≥1 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `n>0 && incx≥1 && (n−1)*|incx| 超 int32 范围`（int64 计算，issue #745 双边守卫） | `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 详细设计（required）

## 算子分析

### 数学公式

`result = argmin_i (|Re(x[k])| + |Im(x[k])|)`（首个最小者，1-based）。
确定性：1-范数模的比较在 IEEE 语义下确定，严格小于更新保证取最小索引。

### 支持数据类型

输入 COMPLEX64（实/虚部各 FLOAT32，交错存储），输出 INT32；中间计算保持 FLOAT32。
特殊值语义（任务书 §3.2，与 Isamin 不同点加粗）：
- NaN：`absVal < bestVal` 对 NaN 恒 false → **NaN 永不被选为最小**；
- **首元素为 NaN：基准即为 NaN，后续任何元素都不小于它 → 结果为 1（不置 FLT_MAX）**；
- ±0.0：|−0|+|Im| 等价归约，零模并列 → 取最先出现索引（全零向量 result=1）；
- Inf：只要存在有限模元素即不选 Inf；全 Inf（模）向量首个 Inf 胜出。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 [n] | 物理长度 `1+(n−1)*|incx|` 个复数元素（2n float 起算）；调用方保证存储有效 |
| n = 0 / incx < 1 | 快速返回：result=0、成功、不启动 kernel（含 0 与负步长，无负步长遍历） |
| n < 0 | 非法：INVALID_VALUE |
| n = 1 | result = 1（含首元素 NaN 的边界——见 NaN 语义） |
| 广播 / 0 维 Tensor | 不涉及（BLAS-1） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx == 1` | 连续读入（2n float）→ 解包/逐分量 Abs → 成对 Add 得模 → 块内 (value,index) 归约 → partial → finalize | 是（性能基线 base case 均连续） |
| **V_STRIDED** | `incx ≥ 2` | src_gap 步长 DMA（8B/复数元素）读 → 同上归约（index 为逻辑序 1-based） | 否（精度覆盖） |

数据流（V_CONTIG，复用 iamax/arch22 两段式骨架）：

```
GM x(2n float) ──DataCopy──▶ UB inBuf
                                │ Abs（逐分量，交错数据免物理解包）
                                ▼
                            UB absBuf（2n float 绝对值）
                                │ 成对 Add：mod[i] = |Re_i| + |Im_i|
                                ▼
                            UB modBuf（n float 模）
                                │ 块内归约：模最小值 + 首索引（严格小于更新；首元素 NaN 即停）
                                ▼
             workspace（每核 1 组 partial：minMod + minIndex）
                                ▼
        finalize（核 0）：跨核 (value,index) 归约 → GM result（INT32，1-based）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`；每核负责连续逻辑区间，kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4：tiling 不按核数组分配）；
- 1-based 索引 = 逻辑区间偏移 + 块内偏移 + 1；小 n 单核完成。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_CONTIG tileSize 按 2n float 计：inBuf 双缓冲 + absBuf + modBuf + 归约 workLocal，
  tile 取 2048~4096 复数元素量级（复数路径 UB 占用约为实数版 2 倍，tile 相应减半）；
- 中间结果及时压缩（参照 isamax arch22 DEAL_TIMES_EACH_CORE_REDUCE 规划），partial 恒定一个 block；
- 尾块处理：补零位置必须写 FLT_MAX 哨兵（|0|=0 会被误选为最小）或按 calCount 有效归约——
  哨兵方案为首选（与向量流水兼容）；复数路径按 float 对补零，成对 Add 前保证虚部分量同被哨兵覆盖。

**tilingkey 规划策略**：tiling 字段分支（incx==1 / ≥2）；Tiling 字段：n、incx、perCoreN、remainder、useCoreNum、tileSize（对齐 arch35 `icamin_tiling_data.h` 命名）。

**数据检测**：按 V1..V6 顺序执行（见需求拆解节）：
- V3（result 判空）先于 V4（quick return）；
- quick return 写 result 用 `aclrtMemsetAsync`（issue #738 惯例）；
- V6 用 int64 计算跨度、双边守卫。

**其他 host 侧约定**：
- workspace：跨核 partial 各核独立槽，大小按 arch22 isamax 既有布局（复数版 partial 结构相同：float value + int32 index）；
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
   │        成对 Add → modBuf（cnt 个模）                           │
   │        块最小值 + 首索引（严格小于更新；NaN 语义按 §3.2）        │
   │        与本核累计 (minMod, minIndex) 成对比较更新               │
   │  V→MTE2: SetFlag/WaitFlag                                     │
   └───────────────────────────────────────────────────────────────┘
   └── 本核结束：写 workspace[coreIdx] = (minMod, minIndex+1)
finalize（核 0）:
   读 workspace 全部 partial → 跨核 (value,index) 归约 → GM result
```

**复数解包要点**：交错存储无需物理解包——对 2n float 逐分量 Abs 后，相邻成对（偶位=Re、奇位=Im）
Add 即得每复数元素的 1-范数模；成对 Add 用向量指令 stride-2 语义或先偶/奇拆分再 Add（S4 实测定案，
首选向量指令可表达的形态）。

**块内"首个最小索引"实现要点**：与实数 Isamin 同款——ReduceMin 求模最小值 + 相等掩码定位**第一个**
等于最小值的位置；比较谓词统一 `absVal < bestVal`（严格小于）；
**不得**用 `<=`（破坏 tie-break）。

**步长路径（V_STRIDED）**：`asc_copy_gm2ub_align(src_gap=(incx−1)*8)`（复数元素 8B burst）；
块内索引按逻辑序累计，与物理 DMA 基址解耦。

**workspace 布局**：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | numBlocks × 8B | 每核 partial（minMod float + minIndex int32，8B 对齐） |

**同步事件配对**：MTE2→V、V→MTE2 显式 SetFlag/WaitFlag；finalize 与各核 partial 写入按单 kernel 内栅栏（S4 与两 kernel 方案实测取舍）。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 大 n 满核并行 + 单 kernel finalize，一次 launch |
| 双缓冲流水 | inBuf 2 槽，MTE2 与 V 重叠 |
| 免解包复数处理 | 交错数据逐分量 Abs + 成对 Add，无中间重排 GM 访问 |
| 掩码定位首索引 | ReduceMin + 掩码扫描，热路径全向量指令 |
| 中间结果及时压缩 | 固定轮次压缩 partial，UB 占用恒定 |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc（workspace 框架管理） |

**未采用的路线及原因**：
- **arch35 kernel 直接平移**：SIMT 线程模型——死路；
- **欧氏模 sqrt(Re²+Im²)**：口径错误（任务书 §2.1-2/§7.6 明令禁止），golden 也按 1-范数——死路；
- **纯 ReduceMin 无索引方案**：丢失 tie-break 与 NaN 语义控制权——不采用；
- **负步长反向遍历**：incx<1 即 quick return——不实现；
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
| 参数合法性 | n≥0（n<0 → INVALID_VALUE）；n=0 或 incx<1 → quick return（result=0，成功）；x/result/handle 不可为 nullptr（result 判空任意 n 生效）；跨度溢出（issue #745）→ INVALID_VALUE；元素值不校验 |
| 非连续 Tensor 支持 | 不要求额外 padding 场景：步长由 incx≥1 表达；incx<1 为 quick return 不做反向遍历 |
| broadcast 规则 | 不涉及（单向量归约到标量索引） |
| dynamic shape 要求 | 不要求：n 运行时入参 |
| 原地与视图语义 | 不涉及原地更新；result 为独立输出标量 |
| 确定性计算要求 | 整数索引结果语义确定（严格小于更新保证最小索引），bit-exact 判定不依赖归约顺序 |
| 空 Tensor 与 0 维处理 | n=0 合法 quick return，result 写 0；n<0 返回 INVALID_VALUE |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream；读回 result 前须同步 stream |

**已知限制（如实标注）**：
1. 不支持负步长反向遍历（incx<1 quick return，对齐 cuBLAS）；
2. NaN 语义按 cuBLAS/golden 口径（首元素 NaN → result=1，不置 FLT_MAX），与同族实数 Isamin 的处理不同（注释显式标注防混淆）；
3. 步长路径不参与性能判定；
4. 复数路径 UB 占用约为实数版 2 倍，大 n 下 tile 轮次增加（中间压缩已规划，性能以 S4 实测为准）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | 测试工程内 cblas 风格 CPU 参考循环（\|Re\|+\|Im\| 逐元素比较、严格小于取最小索引；Netlib 无 icamin 例程） | 任务书 §3.2 |
| 比对方式 | 整数索引精确比对（bit-exact）：`actual == golden`，错位即失败 | 任务书 §3.2 |
| 浮点阈值 | 不适用于输出比对；输入分量按 FLOAT32 生成域约束 | 任务书 §3.2 |
| NaN 语义 | 严格小于更新，NaN 不被选中最小；首元素 NaN → result=1（不置 FLT_MAX）；实现须对齐该语义 | 任务书 §3.2 |
| 并列语义 | 严格小于更新，并列取最小索引（含全零/常量向量） | 任务书 §2.1/§3.2 |
| quick return 判定 | n=0 或 incx<1 用例判定 result == 0 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 输入场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{icamin_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`，共 1200 条（CSV 实测 1201 行含表头）；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/iamin/icamin/arch22/`（参照 `test/isamin/` 模式新建；CSV 列格式对齐 `test/isamin/isamin_param.h`，x 列写 NULLPTR 表示空指针负向用例），并在 `test/iamin/icamin/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：n ∈ {0, 1, 2, 3, 5, 7, 2 的幂及 ±1, …, ≥2^20}；incx ∈ {1, 2, 3}（正常）与 {0, −1, −2, −3}（quick return，期望成功且 result=0）；填充含全零、正负交替、极端值、Inf/NaN；负向含 n<0（INVALID_VALUE）、x nullptr、result nullptr（含 n=0 / incx<1 的 quick return 路径，均期望 INVALID_VALUE）、跨度溢出；实部/虚部独立采样；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]，实/虚部独立采样）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|incx|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasIcamin` 声明已存在于 `include/cann_ops_blas.h`（L244），本次不改公共头；
- **新建目录 `blas/iamin/arch22/`**（若 Isamin 已建则并列加文件）：不影响 arch35；按 SOC 选目录编译，ascend950 构建行为不变；
- **对 arch22 同族 isamin/isamax 的影响**：无。复用归约范式但为独立文件与独立接口，无共享可变状态；
- **测试构建影响**：`test/iamin/icamin/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支保持不变；
- **文档影响**：`blas/iamin/README.md`（或族 README）产品支持表新增 aclblasIcamin「Atlas A2/A3：支持」条目，并同步约束说明（n<0 报错 / n=0 或 incx<1 quick return result=0 / 1-范数模口径 / NaN 语义）。
