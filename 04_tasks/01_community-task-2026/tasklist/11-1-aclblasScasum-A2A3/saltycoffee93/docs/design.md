# aclblasScasum 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasScasum算子开发 任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/asum/`、`include/cann_ops_blas.h`、`blas/asum/arch22/sasum_host.cpp`、`blas/asum/arch35/scasum_*`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasScasum 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/asum/arch22/`（实现）与 `test/asum/scasum/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数向量绝对值分量之和（ASUM）能力，
对标 cuBLAS `cublasScasum`（cuBLAS Library 文档，语义参考 Netlib `scasum.f`）。

## 背景介绍

### 算子功能定位

BLAS Level-1 复数向量归约到实标量：对向量 x 每个复数元素取 `|Re| + |Im|` 后求和。

数学公式：

```
result = Σ_{i=1..n} ( |Re(x[k])| + |Im(x[k])| ),  k = 1 + (i−1)×incx
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasScasum | cublasScasum | 单精度复数 ASUM：Σ(|Re|+|Im|)，输出 FLOAT32 实标量，支持正步长（≤0 走 quick return） |

### 业界对标

- 对标接口：cuBLAS `cublasScasum(handle, n, x, incx, result)`。
- Golden 语义来源：Netlib BLAS `scasum.f`（https://www.netlib.org/blas/scasum.f）。
- 语义差异点（相对仓内既有实现）：
  1. **quick return 口径（本任务最易踩坑项）**：cuBLAS 文档与 Netlib `scasum.f` 一致——
     **n ≤ 0 或 incx ≤ 0（含 0 与负步长）都不触发 kernel，直接写 result = 0.0f 并返回成功**。
     这与同族实数 sasum / rot 族的"负步长反向遍历"语义**不同**（scasum 无负步长遍历能力，任务书 §2.1-3 明示）；
  2. **result 空指针守卫**：result 为 nullptr 时**任意 n（含 quick return 路径）**返回
     `ACLBLAS_STATUS_INVALID_VALUE`——quick return 前必须先过 result 判空；
  3. **归约累加顺序**：不要求与 golden 逐位一致（任务书 §2.1-4），按 §3.2 阈值判定。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/asum/arch22/`（该目录**已存在**，当前仅实数 sasum → **并列加文件**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同族 arch22 实数实现 | `blas/asum/arch22/sasum_{host,kernel}.cpp` | arch22 工程骨架、host 校验序、分核/tiling 骨架（注意其 AsumTilingData 含按核数组 startOffset/calNum[40]，arch22 现行风格——复数版遵循同目录形态，kernel 侧自算偏移） |
| 同算子 arch35 实现 | `blas/asum/arch35/scasum_{host,kernel,tiling_data}.*` | 复数路径语义、ReduceSum 用法（`scasum_kernel.cpp:203` `ReduceSum(outLocal, inLocal, workLocal, dataCount)`）、workspace 最小化计算（L87）平移；kernel 需按 arch22 SIMD 范式改写 |
| 步长访问范式 | arch22 同族（如 axpy/caxpy） | incx≥1 的 src_gap 步长读 |

### 接口规格

```cpp
aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);
```

- 唯一声明位置：`include/cann_ops_blas.h` **L239**（现场核验；任务书 §2.1-6 写 L236，以仓内实际为准）。
  **本次任务不修改公共头**（声明已存在、与 950PR 共用 → 红线 3）。
- `aclblasComplex` 定义于 `include/cann_ops_blas_common.h`：实部/虚部各 float32，交错存储，n 个复数元素对应 2n 个 float。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | x 的复数元素个数，Host 内存 | int | int32 | 任意整数；n ≤ 0 走 quick return（写 result=0.0f，返回成功，不报错） | — |
| x | 单精度复数向量，Device 内存，只读 | const aclblasComplex* | COMPLEX64 | n>0 且 incx>0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验 | 一维 [n]，物理长度 `1+(n−1)*|incx|` 个复数元素 |
| incx | x 相邻元素步长（复数元素为单位），Host 内存 | int | int32 | 任意整数；incx ≤ 0（含 0 与负步长）走 quick return（result=0.0f，返回成功）；`(n−1)*|incx|` 超 int32 范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #805） | — |
| result | （\|Re\|+\|Im\|）之和，实数标量，Device 内存，独立输出 | float* | FLOAT32 | 任意 n（含 quick return 路径）下为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；正常路径值非负，溢出场景可为 Inf | 单值标量 |

### 数学公式

```
result = Σ (|Re(x[k])| + |Im(x[k])|)
```

与 §3.2 golden（cblas_scasum，Netlib SCAS1 口径逐项非负累加）同一形态；
逐项 `|Re|+|Im|` 非负 → 部分和单调不减，大 n × 接近 FLT_MAX 元素时结果可溢出到 Inf（golden 亦溢出，按 Inf 一致性判定）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 单精度复数 ASUM：COMPLEX64 一维向量、正步长（≤0 quick return）、
输出 FLOAT32 实标量；精度按 FLOAT32 档（rtol=atol=2^-13）标量比对；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `result = Σ(|Re|+|Im|)`（归约到标量）。golden 白嫖 cblas（Netlib BLAS 3.10.0 复数实现），随测试工程提供；quick return 用例 golden 固定 0.0f，无需调 cblas（任务书 §3.2） |
| 2 | 接口原型 | `aclblasStatus_t aclblasScasum(aclblasHandle_t, int, const aclblasComplex*, int, float*)`，声明于 `include/cann_ops_blas.h` L239（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 输入 **COMPLEX64**（分量 FLOAT32），输出 **FLOAT32**（任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 scasum，不受影响 |
| 6 | 精度要求 | 标量比对（非逐元素）：`|actual − golden| ≤ atol + rtol*|golden|`，rtol=atol=2^-13，max_abs_error ≤ 1e-2 or 32×ULP；matched_ratio 退化为单值判定；Inf/NaN 按 §3.2 等效规则；归约顺序不要求 bit-exact（任务书 §3.2，mixed_tolerance_standard，opbase commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定。典型 case（实测 CSV 读数）：n=1048576 → ≤24.83us；n=2097152 → ≤27.69us；n=4194304 → ≤41.94us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase 两段归约**：①逐元素 `Abs+Add`（实部/虚部解包为 2n float 或交错处理）→ UB 内块归约；②跨核 partial 经 workspace 汇总（ReduceSum finalize）。死路①：arch35 kernel 直接平移（SIMD 指令集不同）；死路②：负步长反向遍历（scasum 语义为 quick return，不实现） |
| 9 | 评测来源 | 精度 golden：Netlib BLAS 3.10.0 `cblas_scasum`；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V5，由任务书 §2.4/§2.5 反推，顺序不可调换）**：

| 序 | 条件 | 返回码 / 行为 |
|----|------|--------------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `result == nullptr`（**任意 n、任意 incx，先于 quick return 判断**） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `n > 0 && incx > 0 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `n > 0 && incx > 0 && (n−1)*|incx| 超 int32 范围`（int64 计算，issue #805） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V5 | `n <= 0 \|\| incx <= 0` | **quick return**：写 `result = 0.0f`（aclrtMemsetAsync，issue #738 惯例——result 为 Device 地址，host 不得直接解引用）、返回 SUCCESS、不触发 kernel |

> 注意 V2 先于 V5：任务书 §2.4 result 行明确"nullptr 时（任意 n，含快速返回路径）返回 INVALID_VALUE"。

---

# 详细设计（required）

## 算子分析

### 数学公式

`result = Σ (|Re(x[k])| + |Im(x[k])|)`，k = 1+(i−1)×incx，i = 1..n。
归约运算，累加顺序不要求与 golden 一致（任务书 §2.1-4/§3.2），按阈值判定。

### 支持数据类型

输入 COMPLEX64（实/虚部各 FLOAT32，交错存储）；中间计算保持 FLOAT32（|Re|+|Im| 逐项非负累加）；
输出 FLOAT32。特殊值：Inf/NaN 按 IEEE 传播——含 NaN 元素时部分和为 NaN（golden 同为 cblas 浮点累加，等效判定）；
大 n × 极端值溢出到 Inf 时与 golden 按 Inf 一致性判定（§3.2，自测报告单独标注）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 [n] | 物理长度 `1+(n−1)*|incx|` 个复数元素（2n 个 float 起算）；调用方保证存储有效 |
| n ≤ 0 | quick return：result 写 0.0f、返回成功、不访问 x |
| incx ≤ 0 | quick return：同上（含 0 与负步长，**无负步长遍历**） |
| incx ≥ 1 | 正常计算路径（含 incx>1 跳步访问） |
| 广播 / 0 维 Tensor | 不涉及（单向量归约到标量） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx == 1` | 连续读入，Abs + 成对 Add + 块内 ReduceSum | 是（性能基线 3 条 base case 均 incx=1） |
| **V_STRIDED** | `incx ≥ 2` | src_gap 步长 DMA 读 → 同上归约 | 否（精度覆盖） |

数据流（V_CONTIG）：

```
GM x(2n float) ──DataCopy──▶ UB inBuf
                                │  Abs（复数解包后按 float 逐分量取绝对值）
                                ▼
                            UB absBuf ──ReduceSum（块内，复用 arch35 ReduceSum 范式）──▶ UB partialBuf
                                │ 每核循环内及时压缩（中间结果降为一个 block）
                                ▼
                    workspace（每核 1 个 float partial）
                                ▼
        finalize kernel / 尾核 ReduceSum ──▶ GM result（单 float）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`；每核负责连续逻辑区间，kernel 侧自算 offset/count（arch35 `scasum_host.cpp` 注释确认的 kernel 自算分配范式：`offset(i) = i*perCoreN + min(i, remainder)`）；
- 小 n（n ≤ 单核 tile 容量）单核完成，避免跨核同步开销。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_CONTIG tileSize 按 2n float 计：inBuf 双缓冲 + absBuf + ReduceSum workLocal，tile 取 4096 复数元素（32 KiB float×2）量级，留 workspace 余量；
- 中间结果及时压缩：单核多次 tile 循环时，累计超过 63 轮即做一次 ReduceSum 中间归约（参照 `blas/iamax/arch22/isamax_kernel_impl.h` 的 DEAL_TIMES_EACH_CORE_REDUCE 规划思想），partial 占用降为一个 block；
- 尾块 DataCopyPad 对齐补零（Abs 对补零的 0 值无影响，不污染结果）。

**tilingkey 规划策略**：tiling 字段分支（`incx==1` 连续 / ≥2 步长），同族 arch22 现行模式；
Tiling 字段：n、incx、perCoreN、remainder、useCoreNum、tileSize（命名对齐 arch35 `scasum_tiling_data.h`）。

**数据检测**：按 V1..V5 顺序执行（见需求拆解节）：
- V2（result 判空）必须先于 V5（quick return）——任务书 §2.4 result 行"任意 n"口径；
- quick return 的 result 写 0 用 `aclrtMemsetAsync`（Device 地址，不得 host 解引用，对齐仓内 issue #738 既有注释范式）；
- V4 用 int64 计算跨度，阈值 INT32_MAX（仅正步长，无负下界需求，但保留 int64 全程计算）。

**其他 host 侧约定**：
- workspace：跨核归约需要 —— 大小 = `numBlocks × sizeof(float)`（对齐 arch35 scasum workspace 最小化计算思路，`scasum_kernel.cpp:87`）；host 计算 `GetWorkspaceSize` 返回该值；
- 同步：异步下发，host 不同步（任务书 §2.5）；读回 result 前由调用方 `aclrtSynchronizeStream`；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策。

#### kernel 侧设计

**Init/Process 两阶段**（V_CONTIG，单核视角）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(inBuf/absBuf/workBuf)
Process:
   ┌── for each tile (双缓冲) ────────────────────────────────┐
   │  MTE2: DataCopy inBuf ← GM x[off..]（2×cnt float）         │
   │  MTE2→V: SetFlag/WaitFlag                                 │
   │  V:    Abs(absBuf, inBuf, 2*cnt)                          │
   │        ReduceSum(sumLocal, absBuf, workLocal, 2*cnt)      │
   │        Add(accumLocal, accumLocal, sumLocal)              │
   │  V→MTE2: SetFlag/WaitFlag（下一 tile 搬运重叠）            │
   └───────────────────────────────────────────────────────────┘
   └── 每 63 轮中间归约一次；结束后 partial 写 workspace[coreIdx]
finalize: 核 0（或独立小 kernel）ReduceSum workspace → GM result
```

复数解包说明：交错存储可直接对 2n 个 float 做 Abs（|Re| 与 |Im| 逐分量取绝对值等价于复数元素取 |·|₁），
无需物理解包——Abs 按 float 粒度作用于交错数据即得逐分量绝对值，ReduceSum 直接求和，天然实现 |Re|+|Im| 口径。

**步长路径（V_STRIDED）**：`asc_copy_gm2ub_align(src_gap=(incx−1)*8)`（复数元素 8 字节）；
DMA 粒度按复数元素（8B burst），其余流程同 V_CONTIG。基址公式 `base = t0*incx`（仅正步长，无负步长分支）。

**workspace 布局**：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | numBlocks × 4B | 每核 partial 结果（float） |

**同步事件配对**：MTE2→V、V→MTE3/MTE2 显式 SetFlag/WaitFlag；finalize 与各核 partial 写入间靠 kernel 内栅栏（单 kernel 方案）或事件同步（两 kernel 方案，S4 实测定案）。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 大 n 满核并行 + 单 kernel 内 finalize，避免两次 launch（终态以 S4 实测取舍：两 kernel 方案为降级备选） |
| 双缓冲流水 | inBuf 2 槽，MTE2 与 V 重叠 |
| Abs+ReduceSum 融合 | Abs 结果直接 ReduceSum，无额外 GM 访问；交错存储免解包 |
| 及时中间归约 | 每 63 轮压缩 partial，UB 占用恒定 |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc（workspace 除外，框架管理） |

**未采用的路线及原因**：
- **arch35 kernel 直接平移**：SIMD 指令集与 tiling 传递方式（GM 指针 vs const 引用）不同——死路；
- **负步长反向遍历**：cuBLAS/Netlib scasum 语义为 incx ≤ 0 quick return——不实现（与 rot 族不同，注释显式说明防误改）；
- **double/FP64 累加抗溢出**：任务书 golden 为 float32 累加且按等效性判定 Inf，引入 FP64 反而偏离 golden 口径——不用。

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
| 参数合法性 | result nullptr（任意 n）→ INVALID_VALUE；n>0 且 incx>0 时 x nullptr → INVALID_VALUE；跨度溢出（issue #805）→ INVALID_VALUE；n≤0 或 incx≤0 → quick return（result=0.0f，成功）；元素值不校验 |
| 非连续 Tensor 支持 | 不要求额外 padding 场景：步长由 incx 表达（incx≥1 跳步访问）；incx≤0 按 quick return 不遍历 |
| broadcast 规则 | 不涉及（单向量归约到标量） |
| dynamic shape 要求 | 不要求：n 运行时入参，host tiling 按运行时 n 计算 |
| 原地与视图语义 | 不涉及原地更新；result 为独立输出标量 |
| 确定性计算要求 | 不要求（归约累加顺序不保证 bit-exact，按阈值判定） |
| 空 Tensor 与 0 维处理 | n ≤ 0 合法 quick return，result 写 0.0f |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream；读回 Device 结果前须同步 stream |

**已知限制（如实标注）**：
1. 不支持负步长反向遍历（语义即 quick return，对齐 cuBLAS/Netlib）；
2. 步长路径（incx≥2）不参与性能判定（性能用例为连续路径）；
3. 大 n × 接近 FLT_MAX 元素的累加溢出场景结果可为 Inf，与 golden 按 Inf 一致性判定。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_scasum`（float32 累加）；quick return 用例 golden 固定 0.0f 不调 cblas | 任务书 §3.2 |
| 比对方式 | 标量比对（单值相对/绝对误差），非逐元素 | 任务书 §3.2 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4（FLOAT32 档） | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99（标量场景退化为单值判定） | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | `\|actual − golden\| ≤ atol + rtol*|golden|` 且 max_abs_error ≤ 限值 | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；溢出场景（大 n×极端值）在自测报告单独标注 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 输入场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{scasum_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`，共 1200 条（CSV 实测 1201 行含表头）；
- CSV 列格式对齐仓内 `test/asum/sasum/sasum_param.h` 参数定义（n / incx / x 填充模式 / result 空指针标记）；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/asum/scasum/arch22/`，并在 `test/asum/scasum/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：n ∈ {0, 1, 2, 3, 7, 2 的幂及 ±1, …, ≥2^20}；incx ∈ {1, 2, 3}（正常）与 {0, 负}（quick return）；填充含全零、正负交替、极端值、Inf/NaN、大 n×极端值溢出场景；负向含 result/x nullptr、跨度溢出；实部/虚部独立采样；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]，实/虚部独立采样）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|incx|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasScasum` 声明已存在于 `include/cann_ops_blas.h`（L239），本次不改公共头；
- **对 arch35 影响**：无。`blas/asum/arch35/scasum_*` 保持不变，按 SOC 选目录编译互不覆盖；
- **对 arch22 同族 sasum 影响**：无。同目录并列新文件（aclblasScasum vs aclblasSasum），接口独立、无共享可变状态；
- **测试构建影响**：`test/asum/scasum/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支保持不变；
- **文档影响**：`blas/asum/README.md` 产品支持表新增 aclblasScasum「Atlas A2/A3：支持」条目，并同步约束说明（n≤0 或 incx≤0 quick return、result 判空先于 quick return）。
