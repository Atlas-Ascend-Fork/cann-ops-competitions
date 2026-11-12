# 需求背景（required）

## 需求来源

- 任务来源：2026 年 11 月社区任务「aclblasIcamin 算子开发（A2/A3）」（昇腾任务广场）。
- 开源仓库：https://gitcode.com/cann/ops-blas
- 任务范围：在 `blas/iamin/arch22/` 新增 `aclblasIcamin` 的 Atlas A2/A3（dav-c220 / arch22）实现，与仓内已有 arch35 实现并存。
- 对标基线：cuBLAS `cublasIcamin`（A100 性能基线）、Netlib/cblas `icamin`（精度与语义基线，任务书指定）。

## 背景介绍

`Icamin` 属于 BLAS Level 1 的 index-returning reduction（`i?amin` 家族），返回单精度复数向量中
1-范数（|Re|+|Im|，Manhattan 模）最小元素的 **1-based** 索引，常用于稀疏预处理、数值优化中的
pivot 搜索等场景。

仓库现状与差距：

| 项目 | 现状 | 目标 |
| --- | --- | --- |
| 架构支持 | 仅 arch35 目录实现 | 新增 `blas/iamin/arch22/`，A2/A3（910B）支持 |
| 访存 | arch35 可用 SIMT | arch22 无 SIMT，需 DataCopyPad 批量搬运 + 标量访问实现跨步 |
| 归约 | 已有带索引归约范式 | 使用 `Abs + ReducePairElem(vcpadd) + ReduceMin(calIndex)` 向量流水线 |
| 并行 | - | AIV_ONLY 多核分片 + workspace 槽位两阶段归约 |

核心挑战是把 cblas 语义（严格 `<`、NaN 永不获胜、并列取最小索引、首元素 NaN 恒返回 1）在
arch22 的 Ascend C 指令约束下做到 bit-exact，同时性能达到 GPU 基线 0.8 倍以上。

# 需求分析（required）

## 需求描述

```
result = argmin_i ( |Re(x[k])| + |Im(x[k])| ),  k = 1+(i-1)*incx,  i = 1..n
```

输出为 **1-based** 索引；多个元素模相等时取**最小索引**；NaN 语义遵循 cblas：比较使用严格 `<`，
NaN 永不更新最优值，首元素 NaN 时结果恒为 1。

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n,
    const aclblasComplex* x, int incx, int* result);
```

| 参数 | 输入/输出 | 类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | ops-blas 上下文句柄（stream、workspace） |
| n | 输入 | int | 复数元素个数，n ≥ 0 |
| x | 输入 | COMPLEX64* | Device 内存，物理长度 1+(n-1)*\|incx\| |
| incx | 输入 | int | 步长，≥1 正常计算 |
| result | 输出 | INT32* | Device 内存，最小模元素 1-based 索引 |

错误与快速返回行为：

- handle == nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- n < 0 → `ACLBLAS_STATUS_INVALID_VALUE`
- n == 0 或 incx < 1 → result 写 0，返回 SUCCESS（quick return，不触发 kernel）
- x == nullptr（n>0 且 incx≥1）→ `ACLBLAS_STATUS_INVALID_VALUE`
- result == nullptr（任意 n）→ `ACLBLAS_STATUS_INVALID_VALUE`
- (n-1)*\|incx\| > INT32_MAX（host 侧 int64 计算）→ `ACLBLAS_STATUS_INVALID_VALUE`

## 需求拆解

| 编号 | 拆解项 | 说明 |
| --- | --- | --- |
| R1 | 模值计算 | 交错复数数据 Abs 后经 vcpadd 配对求和：[|re|,|im|,...] → [|re|+|im|,...] |
| R2 | 带索引最小值 | ReduceMin(calIndex=true) 输出 (minVal, minIdx)，跨块/跨核并列取最小索引 |
| R3 | NaN/Inf 语义 | 首元素 NaN 短路；vcmin 传播 NaN 时标量重扫恢复有限最小值 |
| R4 | 跨步访问 | incx≠1 时标量 GM 访问，IEEE 语义天然正确 |
| R5 | 分核调度 | n≤4096 单核单 kernel；否则多核分片 + reduce 合并 |
| R6 | 精度 | INT32 索引与 CPU golden bit-exact（EXPECT_EQ） |
| R7 | 性能 | 200 条 TC_PF 用例 GPU/NPU ≥ 0.8（msprof task duration 口径） |

# 详细设计（required）

## 算子分析

### 数学公式

result = argmin_i ( |Re(x[k])| + |Im(x[k])| )，k = 1+(i-1)*incx；并列取最小 i。

### 支持数据类型

输入 COMPLEX64（分量 FLOAT32 全集，含 INF/NAN），输出 INT32。

### 支持形状

逻辑一维 [n]，n ∈ [0, 2^31)，incx ≥ 1；incx < 1 为合法 quick return。

## 算子实现

### 实现方案

#### host 侧设计

1. **参数校验与 quick return**：按上表顺序校验；n==0 / incx<1 时 `aclrtMemsetAsync` 写 result=0。
2. **路径分派**：
   - incx==1 且 n ≤ 4096 → `icamin_arch22_small_kernel`（单核，1 次 launch，直写 result）
   - incx==1 且 n > 4096 → memset + `icamin_arch22_vec_kernel` + `icamin_arch22_reduce_kernel`
   - incx≠1 → memset + `icamin_arch22_strided_kernel` + reduce
3. **分核策略**（vec 路径）：`GetAivCoreCount()` 取 AIV 核数（40），每核分到
   `perCoreN = ceil(n/40)` 并向上对齐到 4 复数元素（保证每核 GM 起始 32B 对齐、切片长度 32B
   整倍数）；末核取余量。三 kernel 流水线的 launch 开销约 16us，n ≤ 4096 时已超过 GPU 阈值，
   故小规模走单核单 kernel（约 5us）。
4. **workspace**：每核 2 float 槽位（≤40 核 × 8B = 320B），复用 handle 内置 workspace，
   launch 前 `aclrtMemsetAsync` 预清零，保证未发布槽位读到 0（reduce 判定 v < bestVal 时
   0 永不敌 -FLT_MAX / 真实正值）。

#### kernel 侧设计

**向量流水线**（small / vec 共用，2048 复数元素分块）：

1. DataCopyPad GM→UB：burstLen 传**实际（非对齐）字节数** + 显式 rightPaddingNum，
   尾部以 +Inf 填充到 32B；块尾 Duplicate +Inf 到 64 float 对齐边界
2. `Abs(ub_x, ub_x, alignedFloats)`：交错数据逐分量取绝对值
3. `ReducePairElem<SUM>(ub_mag, ub_x, 64, alignedFloats/64, 1, 1, 8)`（vcpadd）：
   相邻分量配对求和得到模值数组
4. mag 尾部 Duplicate +Inf 补齐到 64 元素对齐
5. `ReduceMin(ub_slot, ub_mag, ub_work, 64, magAligned/64, 8, true)`（vcmin，规范形式）：
   输出 (minVal, minIdx)
6. 跨块累积：严格 `<` 更新，相等取较小全局索引

多核路径每核发布 (bestVal, bestIdx) 到 workspace 槽位；reduce kernel（单核）合并全部槽位，
写 1-based result。

**NaN / Inf 语义**：

- **首元素 NaN 短路**：cblas 参考实现 bestVal 以 x[0] 播种，NaN 时后续所有有限值均不满足
  `<`，结果恒为 1。vec/small kernel 每核直接读全局 x[0]，NaN 即整体退出（全 NaN 4M 输入
  从 1.9ms 降到 ~16us）
- **vcmin 传播 NaN**（实测非 minNum）：块内含 NaN 时 ReduceMin 返回 NaN，触发该块标量重扫
  恢复有限最小值（NaN 永不获胜），保证混合 NaN 正确性
- **Inf**：正常参与比较；全 Inf 数组并列取最小索引（与 golden 一致）

**跨步路径**（incx ≠ 1）：逐元素 GM 标量访问（无法批量搬运），IEEE 比较语义天然正确，
每核处理一段逻辑索引范围后经 workspace 汇聚。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制

dav-c220（910B3）实测约束（踩坑记录）：

1. `DataCopyPad` burstLen 传已对齐长度时 paddingValue 不生效（GM 越界读 + stale UB 残留）
2. `Compare(count)` 要求 count×4 为 256 倍数，否则不写掩码 → Select 选出垃圾
3. 元素级 `Min/Max`（vmin/vmax）各种调用形式（count-mode / bit-mask repeat / in-place /
   out-of-place / 切片）实测均不写回，本算子不可用
4. `ReduceMin` 基础 API 的 int-mask > 128 走未定义移位，必须用 mask=64 重复形式 + +Inf 补齐

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | INT32 索引与 CPU golden bit-exact；1252/1252 PASS（1200 官方 CSV + 51 条 GAUSS_NORM_5_2 正态补充 + NullHandle），0 FAIL/ERROR/SKIP | 任务书 §3.2 |
| 性能标准 | 200/200 条 TC_PF 达标：事件计时口径最低 GPU/NPU ratio 0.845；msprof 单算子 task duration 口径 12 条代表用例交叉验证最低 0.931 | 任务书 §3.3 |
| 内存标准 | 不涉及（任务书 §3.4 原文） | 任务书 §3.4 |

代表性性能数据（Atlas 800T A2 / 910B3，CANN 9.1.0，10 warmup + 100 有效样本）：

| n | incx | NPU avg_us（事件） | NPU mean_us（msprof task） | 阈值（GPU/0.8） |
| --- | --- | --- | --- | --- |
| 512 | 1 | 4.96 | 2.19 | 10.28 |
| 8192 | 1 | 16.22 | 5.14 | 17.13 |
| 1048576 | 1 | 16.10 | 12.58 | 22.70 |
| 4194304 | 1 | 36.59 | 34.00 | 45.80 |
| 4194304（全 NaN） | 1 | 16.62 | 4.62 | 45.80 |
| 3375004 | 1 | 30.61 | 28.55 | 33.49 |

测试与交付：`test/iamin/icamin/`（README 含复现步骤；官方 CSV 未改动，MD5 与任务包一致；
正态分布补充用例以独立 CSV 提供）；task_submission 七件套在私仓根目录；代码与全部测试证据
对应 Commit `0461f1bc8fa3586cf9701575e9be52699d4f30b4`（分支 icamin-arch22）。

## 兼容性分析

新增算子实现（arch22 目录新增文件），不改动既有算子与公共接口，不涉及兼容性分析。
