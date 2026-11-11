# aclblasIcamin 算子设计文档（Atlas A2 / A3）

| 项目 | 内容 |
|---|---|
| 任务名称 | 11月社区任务-aclblasIcamin算子开发（A2/A3） |
| 任务 ID | `aclblas-icamin-a2a3` |
| 目标硬件 | Atlas A2 训练/推理系列（910B，arch22 / dav-2201）、Atlas A3 训练/推理系列（910_93，arch22 / dav-2201） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/11-7-aclblasIcamin-A2A3/kekemashu/docs/design.md` |
| 实现落点 | `blas/iamin/arch22/`（新建） |
| 测试落点 | `test/iamin/icamin/arch22/`（新建） |
| 对标接口 | cuBLAS `cublasIcamin`（GPU 基线见任务包 `gpu_baseline.csv`，A100 实测） |
| 文档版本 | v1.0 |

---

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务 2026 年 11 月《11月社区任务-aclblasIcamin算子开发（A2/A3）》，任务截止 2026-11-22。任务要求在 Atlas A2/A3 上使用 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasIcamin`：在 COMPLEX64 向量中查找 **1-范数模** `|Re|+|Im|` 最小元素的 **1-based 索引**；模相同时返回最小索引。验收通过后合入 ops-blas 主仓。

设计文档按 [cann-competitions 官方模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 编写并以 PR 形式提交；实现代码进入 ops-blas，公开声明复用 `include/cann_ops_blas.h` 中已有声明，禁止定义私有平行接口。

## 背景介绍

### 算子要做什么

数学表达式：

```text
result = argmin_{i=1..n} ( |Re(x[k])| + |Im(x[k])| ),   k = 1 + (i-1)*incx
```

- 输入 `x`：COMPLEX64（`aclblasComplex`，实部/虚部各 float32，交错存储）；逻辑长度 `n`，物理长度 `1+(n-1)*|incx|`。
- 输出 `result`：单个 INT32，取值 `0`（quick return）或 `[1, n]`，**1-based**（Fortran/BLAS 惯例）。
- 复数“模”按 BLAS `icamin` 惯例定义为 **SCABS1 1-范数** `|Re|+|Im|`，**不是**欧几里得模 `sqrt(Re²+Im²)`。
- 并列语义：只有**严格小于**才更新，因此多个元素模相同时返回**最小索引**。
- NaN 语义：golden 以首元素 `abs1(x[0])` 为种子，之后仅 `absVal < bestVal` 更新。NaN 与任何值比较为 false，永不更新；若首元素为 NaN，则基准为 NaN，后续都不小于它，结果为 1（实现不得改置 FLT_MAX）。

### 与同族算子的关系

| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
|---|---|---|---|---|
| `aclblasIsamin` | FP32 | ReduceMin | `\|x\|` | `blas/iamin/arch35/`（仅 950） |
| `aclblasIcamax` | COMPLEX64 | ReduceMax | `\|Re\|+\|Im\|` | `blas/iamax/arch35/`（仅 950） |
| `aclblasIsamax` | FP32 | ReduceMax | `\|x\|` | `blas/iamax/arch22/`（A2/A3 已有） |
| **aclblasIcamin** | COMPLEX64 | **ReduceMin** | `\|Re\|+\|Im\|` | **本任务新增 arch22** |

核心映射：在 arch35 `icamin` 的「模计算 → 最小值归约 → 索引 1-based 写回」流水线上，按 arch22（dav-2201）可用 API 重写向量路径与跨核框架；host 的 handle workspace、异步 launch 复用 arch22 `sdot` 的现代模板，不沿用 `isamax` arch22 的 per-call `aclrtMalloc` + `aclrtSynchronizeStream`。

### 现状分析

- `include/cann_ops_blas.h` 已有 `aclblasIcamin` 声明（仓库实际 L244），无需新增声明。
- `blas/iamin/arch22/` 不存在；A2/A3 构建（`SOC_VERSION=ascend910b*` / `ascend910_93*`）会跳过该算子。
- `test/iamin/icamin/CMakeLists.txt` 只在 `arch35` 注册 `ops_blas_add_gtest_tests`，其他架构退回 `ops_blas_add_tests`（非 GTest runner，输出整数毫秒），本任务需补 arch22 GTest 注册。
- `blas/iamin/README.md` 产品支持表当前标注 A2/A3 不支持，交付时需更新为支持。

### 对标实现

| 项 | cuBLAS `cublasIcamin` | 本实现 |
|---|---|---|
| 数学语义 | `argmin(\|Re\|+\|Im\|)`，1-based | 同 |
| 并列 | 最小索引 | 同（严格 `<`） |
| `n` 范围 | `n ≤ 0` 置 0 返回成功 | `n = 0` quick return；`n < 0` 返回 `INVALID_VALUE`（对齐仓内 isamin 契约与任务书） |
| 步长 | `incx` 任意，负步长反向遍历 | `incx ≥ 1` 正常；`incx < 1`（含 0、负）合法 quick return，不反向遍历 |
| NaN | 未强制规定 | 首元素 NaN → 1；NaN 永不更新（任务书 §3.2 强制口径） |

Netlib 标准 BLAS 无 `icamin` 例程；golden 由测试工程内 cblas 风格 CPU 循环生成（`test/iamin/icamin/icamin_golden.h` 已存在且语义正确）。

---

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上以 Ascend C 直调方式完成：

```text
aclblasIcamin(handle, n, x, incx, result)
  → Host 参数校验 + tiling
  → 单次 kernel launch（quick return 除外）
  → Device 写回 1-based INT32 索引
```

精度为整数索引 **bit-exact**（`actual == golden`），无浮点容差；性能按任务包 `gpu_baseline.csv` 的 0.8× 判据。

## 输入输出规格

| 参数 | 方向 | 类型 | 内存 | shape / 值域 | 说明 |
|---|---|---|---|---|---|
| `handle` | 输入 | `aclblasHandle_t` | Host | 有效句柄 | 携带 stream；nullptr → `HANDLE_IS_NULLPTR` |
| `n` | 输入 | `int` | Host | `n ≥ 0` | 负值 → `INVALID_VALUE`；`n=0` quick return |
| `x` | 输入 | `const aclblasComplex*` | Device | 逻辑 `[n]`，物理 `1+(n-1)*incx` | 实/虚部 float32 交错；`n>0 && incx≥1` 时 nullptr → `INVALID_VALUE` |
| `incx` | 输入 | `int` | Host | `incx ≥ 1` 正常 | `incx<1` quick return；`(n-1)*incx` 超 int32 → `INVALID_VALUE` |
| `result` | 输出 | `int*` | Device | 标量 | `0` 或 `[1,n]`；nullptr（任意 n，含 quick return）→ `INVALID_VALUE` |

## 需求拆解

| 编号 | 子项 | 验收要点 |
|---|---|---|
| R1 | 公开 API | 复用 `include/cann_ops_blas.h` L244 声明；签名与 `cublasIcamin` 逐参数对齐 |
| R2 | 功能语义 | `|Re|+|Im|`、1-based、严格 `<`、并列最小索引；禁止欧氏模 |
| R3 | 边界与错误码 | `n=0`/`incx<1` quick return 写 0；`n<0`、`result` 空、`x` 空、int64 跨度溢出 → `INVALID_VALUE`；`handle` 空 → `HANDLE_IS_NULLPTR` |
| R4 | NaN 语义 | 首元素 NaN → 1；NaN 永不更新；全 NaN → 1；实现与 golden 一致 |
| R5 | Host 设计 | 校验顺序固定；quick return 不启 kernel；单次异步 launch；handle workspace 复用；无 per-call malloc/sync |
| R6 | Kernel 设计 | 单遍扫描、value-only 归约、每核局部最优 + 跨核小规模归约、尾块安全、incx>1 正确 |
| R7 | 测试工程 | `test/iamin/icamin/arch22/` CSV 驱动 GTest；1200 条任务用例 + `GAUSS_NORM_5_2` 补充；CMake arch22 GTest 注册 |
| R8 | 性能与文档 | 单 case ≥ 0.8× GPU；`blas/iamin/README.md` 标注 A2/A3 支持；测试 README 可复现 |

## 与标杆的功能差异

| 项 | 处理 | 依据 |
|---|---|---|
| `n < 0` | `INVALID_VALUE`（非 cuBLAS 的置 0） | 任务书 §2.4 / 仓内 isamin README |
| `incx < 1` | quick return（非反向遍历） | 任务书 §2.4 |
| 首元素 NaN | 结果 1（不置 `FLT_MAX`） | 任务书 §3.2 补充说明 |
| `(n-1)*incx` 溢出 | host 用 int64 计算，超 `INT32_MAX` 报 `INVALID_VALUE` | 任务书 §2.4（issue #745） |

---

# 详细设计（required）

## 算子分析

### 数学公式与 golden

```text
m_i = |Re(x[1+(i-1)*incx])| + |Im(x[1+(i-1)*incx])|,  i = 1..n
result = argmin_i m_i，并列取最小 i
```

golden（测试工程内 CPU 参考，已存在于 `test/iamin/icamin/icamin_golden.h`）：

```cpp
float abs1(float re, float im) {
    float ar = (re >= 0.0f) ? re : -re;
    float ai = (im >= 0.0f) ? im : -im;
    return ar + ai;
}
float best = abs1(x[0].real, x[0].imag);
int bestIdx = 0;
for (int i = 1; i < n; ++i) {
    float v = abs1(x[i*incx].real, x[i*incx].imag);
    if (v < best) { best = v; bestIdx = i; }   // 严格 <，NaN 不更新
}
*result = bestIdx + 1;                          // 1-based
```

关键点：

1. 首元素为种子。首元素 NaN 时 `best = NaN`，后续比较全 false → 结果 1。实现必须复现该语义，**不能**用 `FLT_MAX` 初始化后跳过 NaN。
2. 幅值只含 `abs + add`，无乘法、无 sqrt、无 FMA，CPU 与 device 的 IEEE 结果逐位一致（除 denormal FTZ 场景，见风险）。
3. 归约顺序不影响结果：规则等价于对 `(m_i, i)` 取字典序最小；在无 NaN 的归约域内该规则可结合。

### 支持数据类型

| 参数 | dtype | 说明 |
|---|---|---|
| `x` | COMPLEX64（`aclblasComplex`） | 2×FLOAT32 交错 |
| `result` | INT32 | 单值标量 |
| 计算中间值 | FLOAT32 | 幅值与比较 |

### 支持形状 / 布局

- 一维向量，逻辑长度 `n ∈ [0, INT32_MAX]`；测试预算 `n ≤ 2^24`（host 侧单用例 ≤4GB 的内存预算，非硬件限制）。
- `incx ≥ 1`：任意正步长；`incx = 1` 为连续快路径。
- `incx < 1`（含 0、负步长）：quick return，不启动 kernel。
- 无 broadcast、无 leading dimension、无原地更新。

## 语义与边界

### 校验顺序（host，固定）

| 顺序 | 条件 | 返回 |
|---|---|---|
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n == 0 \|\| incx < 1` | `SUCCESS`，异步写 `result = 0`，不启 kernel |
| 5 | `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `last = (int64)(n-1)*incx > INT32_MAX` | `ACLBLAS_STATUS_INVALID_VALUE` |

说明：

- 第 4 步先于第 5、6 步，因此 `n=0`/`incx<1` 搭配空 `x` 或极端 `incx`（含 `INT32_MIN`）仍为合法 quick return；这与任务包 CSV 的负向组合用例一致。
- 第 6 步用 int64 计算，避免 `(n-1)*incx` 在 int32 域溢出（issue #745）；同时保证 kernel 内 `2*last` 不超过 `UINT32_MAX`。
- quick return 的 `result=0` 用 `aclrtMemsetAsync` 按**设备地址**写回，不在 host 解引用。

### 错误码汇总

| 返回码 | 场景 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 正常计算；quick return |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACLBLAS_STATUS_INVALID_VALUE` | n<0、result 空、x 空、跨度溢出 |
| `ACLBLAS_STATUS_EXECUTION_FAILED` | quick return 写回失败、workspace 不足/内部写回失败 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | AIV 核数查询失败等内部错误 |

### NaN / Inf / ±0 / 并列

| 场景 | golden | 实现 |
|---|---|---|
| 任意位置 NaN | NaN 永不 `<`，不更新 | 归约前把 NaN 幅值 sanitize 为 `+Inf`（见下） |
| 首元素 NaN | 基准 NaN，结果 1 | 单独检测元素 0 的实/虚部，命中则直接写 1 |
| 全 NaN | 结果 1 | 首元素 NaN 路径 |
| `+Inf` | 正常参与比较，多个 +Inf 取最小索引 | sanitize 后按普通值归约 |
| `-Inf` | 幅值不可能为 -Inf（abs） | 同左 |
| `+0/-0` | `==` 相等，严格 `<` 不更新 | FP32 `+0 == -0` 为真，EQ 比较天然合并 |
| 并列 | 取最小索引 | 归约合并用严格 `<`；索引裁决用 `value == min` 后取最小索引 |

**NaN 为什么必须前置 sanitize（关键正确性论证）**

定义 `select(a,b) = a if !(b < a) else b`。含 NaN 时该规则**不可结合**。反例 `[5, NaN, 3]`：

- 从左到右：`best=5`；NaN 不更新；`3 < 5` → 结果索引 3。
- 按 `5 ⊕ (NaN ⊕ 3)` 分组：`NaN ⊕ 3` 因 `3 < NaN` 为 false 保留 NaN；`5 ⊕ NaN` 因 `NaN < 5` 为 false 保留 5 → 结果索引 1。

同一向量仅因切块方式不同得到不同结果。因此必须保证**参与任何 tile/核归约的值域不含 NaN**：对每个幅值做 `mask = (v == v)`，`Select(v, mask, v, +Inf)` 把 NaN 替换为 `+Inf`；再单独处理“元素 0 的 `abs1` 为 NaN”这一 golden 特例。sanitize 后归约域只含普通浮点值，`(value, index)` 字典序取最小可结合，切块/切核/归约顺序均不影响结果。

## Host 侧设计

### 参数校验与 quick return

见「语义与边界」校验顺序表。quick return 实现：

```cpp
aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)
// 失败 → ACLBLAS_STATUS_EXECUTION_FAILED
```

### 分核策略

1. `aivCoreNum = GetAivCoreCount()`；为 0 返回 `INTERNAL_ERROR`。
2. 期望核数：
   - `n ≤ ICAMIN_SINGLE_CORE_LIMIT`（实现取 8192，约 4 个 tile）→ `useCoreNum = 1`。
   - 否则 `cores = min(aivCoreNum, ceil(n / ICAMIN_MIN_PER_CORE), 64)`，`ICAMIN_MIN_PER_CORE = 8192`；
     多核时把 `cores` 取偶（A2/A3 的 AIV 以 2 个为一组），最小为 2。
3. 连续路径（`incx == 1`）：
   - `perCoreN = CeilAlign(ceil(n / cores), 4)`（4 个复数 = 32B，保证每核起始地址与整块 `DataCopy` 对齐）；
   - `useCoreNum = cores`（偶数）；`lastCoreN = n - (cores - 1) * perCoreN`，恒 `> 0` 且 `≤ perCoreN`。
4. 跨步路径（`incx > 1`）：
   - `perCoreN = ceil(n / cores)`（无对齐要求，逻辑下标）；
   - `useCoreNum = cores`（偶数）；`lastCoreN = n - (cores - 1) * perCoreN`。
5. 负载差：连续路径单核切片相差 < 4 个复数（+ 尾核余量），相对 1M 规模 < 0.05%；每核至少 8192 个复数，不会出现空转核。

### Tiling 数据

```cpp
struct IcaminTilingData {
    uint32_t n;          // 逻辑复数元素个数
    uint32_t incx;       // 步长（≥1）
    uint32_t useCoreNum; // 实际使用 AIV 数（连续路径为偶数；单核为 1）
    uint32_t perCoreN;   // 前 useCoreNum-1 核的复数元素数
    uint32_t lastCoreN;  // 最后一核的复数元素数（>0）
    uint32_t tileSize;   // 连续路径每 tile 复数元素数（默认 2048）
    uint32_t mode;       // 0=连续（incx==1），1=跨步（incx>1）
    uint32_t reserved;   // 对齐/扩展
};
```

- tiling 结构**按值**随 kernel 参数传递，不做 host→device `aclrtMemcpy`，压低小 n 固定开销。
- `tileSize = 4096` 复数（32KB 输入/tile）；`PairReduceSum` 每 tile 128 个 repeat，`ReduceMin` count 4096（≤ 单核 UB 预算）。
- 单核路径不使用 workspace、不做跨核同步。

### Workspace 布局

多核时每核写一个 `(minValue, candidateIndex)` 槽：

```text
workspace:
  [0, alignedCores)                float32  minValue[core]
  [alignedCores*4, alignedCores*8) int32    candidateIndex[core]   // 连续路径为获胜 tile 起始复索引；跨步路径为首个最小元素索引
  alignedCores = CeilAlign(useCoreNum, 64)   // 2 个 64-float 对齐数组
  requiredBytes = alignedCores * 8 + 64（保护）
```

- 使用 handle 自带 workspace（默认 32 MiB），不足返回 `EXECUTION_FAILED`。
- 每核先写自己的槽，全部写完后 `SyncAll()`；核 0 读取全部槽合并。无原子操作、无跨核锁。

### 启动与分支

| 场景 | 分支 | 启动 |
|---|---|---|
| `n ≤ 8192`，`incx==1` | 单核连续 | `<<<1>>>`，AIV1 立即返回，无同步 |
| `n > 8192`，`incx==1` | 多核连续 | `<<<useCoreNum>>>`（A2 910 实测 `__vector__` kernel 为 1 block = 1 AIV） |
| `incx > 1`，`useCoreNum==1` | 单核跨步 | `<<<1>>>`，标量扫描 |
| `incx > 1`，`useCoreNum>1` | 多核跨步 | `<<<useCoreNum>>>`，标量扫描 + `SyncAll` |

- 所有分支均为一次 launch；quick return 零 launch。
- 异步返回（不 `aclrtSynchronizeStream`），stream 由 `aclblasSetStream` 绑定。
- `blockDim = useCoreNum`：在 Atlas 800 A2（Ascend910，CANN 9.2.0-beta.1）实测 `__vector__` kernel 为 **1 block = 1 AIV**，`GetBlockIdx()` 即 AIV 序号；`numBlocks = useCoreNum` 已由 n=16384 多核用例验证（见风险 R3）。

## Kernel 侧设计

### 连续路径（incx == 1，主性能路径）

数据流：

```text
GM complex[n] (交错 re/im)
  → DataCopyPad  → UB 交错 float[2*count]（TQue depth=2）
  → 整 tile：Abs + PairReduceSum；尾块：Gather 偶/奇 + Abs + Add → mag = |Re|+|Im|
  → NaN sanitize：Compare(v,v,EQ) + Select(mag, mask, mag, +Inf)
  → 尾部 pad +Inf 到 64-lane 对齐
  → ReduceMin(value-only) → 每 tile 的 minValue（写入 32B slot 的 lane 0）
  ... 所有 tile 结束（每 64 个 tile 一批）
  → 核内：Gather 抽稀 lane0 → ReduceMin + CompareScalar(EQ) + Select + ReduceMin
         → 首批内首个获胜 tile；跨批严格 < 合并
  → workspace[core] = (coreMin, 获胜 tile 起始复索引)
  → SyncAll
  → 核 0：合并所有核槽（最小值 + 最小核序号）→ winCore/winStart
       → 对 winStart 起的 1 个 tile 二次读（≤32KB）
       → mag + sanitize + Compare(EQ, globalMin) → lane 序号 →
         result = winStart + firstLane + 1（int32 精确）
  → 若元素 0 的 abs1 为 NaN → 直接 result = 1
```

关键实现细节：

1. **幅值计算（混合策略）**：tile 长度为 64 的整数倍时用 `Abs(in, in, 2*count)` + `PairReduceSum(mag, in, repeats, 64, 1, 1, 8)`，每个 tile 只有一条 PairReduceSum，V pass 最少；非整 repeat 的尾块（`count % 64 != 0`）改用 `Gather` 分奇偶 + `Abs + Add`，避免 PairReduceSum 在部分尾对上的布局/填充问题。实测（A2，CANN 9.1.0）该混合路径把 1M/2M/4M 从 24/33/50 µs 降到 18.4/22.7/30.1 µs，200 条性能用例全部达标。
2. **NaN sanitize**：`Compare(mask, mag, mag, CMPMODE::EQ, lanes)` 标记非 NaN；`Select(mag, mask, mag, +Inf, VSEL_TENSOR_SCALAR_MODE, lanes)`。NaN lane 的 EQ 为 false，被替换为 +Inf。元素 0 的 NaN 信息在 sanitize 前单独记录（仅核 0 的 tile 0 读取 `inLocal[0..1]`）。
3. **尾部 pad**：`DataCopyPad` 只复制 `2*count` 个 float；`Gather`/`Abs`/`Add` 只写 `[0,count)`；`mag` 在计算前整体 `Duplicate(+Inf)`，因此 `[count, alignedLanes)` 天然为 `+Inf`（`alignedLanes = RoundUp(ceil(2*count/64)*32, 64)`）。pad lane 永不参与胜出。
4. **value-only 归约**：`ReduceMin(tileMin, mag, work, alignedLanes, false)`。不传 `calIndex=true`，规避技能库已记录的「`ReduceMin(calIndex)` NaN 行为不可依赖」问题，也规避平局索引是否 first-occurrence 的未验证行为。
5. **核内 tile 合并（分批，UB 有界）**：每 tile 的 minValue 写在 32B slot 的 lane 0；每 64 个 slot 为一批，用 `Gather`（byte offset 0,32,64,...）抽稀 lane 0 成 64 元素 `compact`，尾部不足 64 的 lane 保持 `+Inf`。`batchMin = ReduceMin(compact, 64)`；`CompareScalar(EQ)` + `Select(laneIdx, +Inf)` + `ReduceMin` 得**批内最小 tile 序号**；跨批用严格 `<` 合并（并列保留更早的批）。tile 序号与 `startOffset + t*tileSize` 相加得到全局复索引（int32/uint32 精确）。批大小固定，UB 占用与 n 无关。
6. **跨核合并**：核 0 读 `wsVal[]`（float）与 `wsIdx[]`（int32）；`wsVal` pad `+Inf` 到 64 lane；`coreMin = ReduceMin(wsVal, 64)`；`CompareScalar(EQ)` + `Select(laneIdx, +Inf)` + `ReduceMin` 得**最小核序号**（≤ 64，float 精确）；`winStart = wsIdx[winCore]`。这样只需一次标量读取，且与核数/索引大小无关（不受 float32 2^24 精确范围限制）。
7. **二次读求首索引**：只读获胜 tile（≤2048 complex，16KB）。`laneVec = CreateVecIndex(0..alignedLanes-1)`（lane ≤ 2047，float 精确）；`mask = (mag == coreMin)`；`Select(sel, mask, laneVec, +Inf)`；`firstLane = ReduceMin(sel)`；`result = (int32)(winStart + firstLane) + 1`。**索引算术全程整型**，避免 float32 对大索引的精度损失。
8. **尾块**：tile 读取长度 `count = min(tileSize, calNum - t*tileSize)`；`DataCopyPad` 支持非 32B 整数倍长度；pad lane 用 +Inf。
9. **元素 0 NaN**：核 0 在处理全局 tile 0 前读 `inLocal[0], inLocal[1]`，任一为 NaN 置 `firstNan_`；最终写结果前若 `firstNan_` 直接写 1，跳过合并结果。
10. **1-based**：仅最终写回时 `+1`；kernel 内部一律 0-based。

### 跨步路径（incx > 1）

- 正确性优先，性能非判据（任务包 TC_PF 200 条全部 `incx=1`）。
- 每核对其连续逻辑区间做标量扫描：`k = (uint64)(start+i)*incx`；读 `xGm[2k], xGm[2k+1]`；`mag = |re|+|im|`；NaN → `+Inf`；首元素作种子，严格 `<` 更新。
- 每核直接得到**实际最小元素索引**（不是 tile 起始），写入 workspace。
- 核 0 合并：最小 `minValue`；并列时取**最小核序号**（切片连续有序 ⇒ 其 `wsIdx` 即全局最小索引）；结果为 `idx+1`。与连续路径共用 `MergeCoreSlots`。
- 元素 0 NaN：核 0 扫描时记录，最终写 1。
- `(n-1)*incx` 已由 host 限制 ≤ `INT32_MAX`，kernel 用 `uint64` 计算 `2*k`，避免 32 位回绕。

### 小 n 路径

- `n ≤ 8192`：单核连续路径，扫描 1~4 个 tile，核内合并后直接二次读获胜 tile 求首索引并写回；无 workspace、无 `SyncAll`、无第二核参与。
- `n = 1`：tile count=1，`alignedLanes=64`，pad 63 个 +Inf；`firstLane=0` → 结果 1。首元素 NaN → 结果 1（同一路径）。

### 同步与流水

| 依赖 | 机制 |
|---|---|
| MTE2 → V（搬入后计算） | TQue depth=2 的 `EnQue/DeQue` 自动事件 |
| V → S（ReduceMin 结果标量读取） | `SetFlag/WaitFlag<HardEvent::V_S>` |
| S → MTE3（结果标量写 UB 后 DMA） | `PipeBarrier<PIPE_ALL>()`（对齐 arch35 做法） |
| 核间槽可见性 | 每核 `DataCopyPad` 写槽 → `PipeBarrier<PIPE_ALL>()` → `SyncAll()` → 核 0 读 |
| 核 0 二次读 | `DataCopyPad` + `EnQue/DeQue`（复用 TQue） |

- 双缓冲：`TQue<VECIN, 2>`，tile t+1 的 MTE2 与 tile t 的 V 计算重叠。
- 单 kernel 完成 scan + 跨核合并 + 二次读，避免第二次 launch（小 n 判据仅 ~10.3 µs，launch 开销敏感）。
- `SyncAll()` 为仓内 arch22 `isamax` 已验证的跨核栅栏；备选 `FftsCrossCoreSync + WaitFlagDev`（`sdot` 模式），作为性能优化项 A/B。

### UB 预算（tileSize=4096，单核，TQue depth=2）

| 缓冲 | 大小 |
|---|---|
| inQueue 2 × 2*4096 float | 64 KB |
| re（4096 float，含 mag） | 16 KB |
| im（4096 float，仅 Gather 备选） | 16 KB |
| byteIdx（4096 int32，仅 Gather 备选） | 16 KB |
| laneIdx（4096 float） | 16 KB |
| slot（64 × 8 float）+ compact/laneIdx（64） | 2.5 KB |
| ReduceMin work + mask + 标量输出 | < 1 KB |
| out（32B） | 32 B |
| **合计** | **≈ 132 KB**（UB 192 KB，余量充足；depth 4 的 200 KB 方案实测越界，已回退 depth 2） |

> 说明：A2 910B3 的 UB 按 192 KiB 规划（arch22 `caxpy` 的静态预算注释），实际以运行时平台查询为准；本设计用量 < 100 KB，不依赖具体 SKU。

## 支持硬件

| 芯片 | 支持 |
|---|---|
| Atlas A2 训练系列（910B / 910B3） | √ |
| Atlas A2 推理系列 | √ |
| Atlas A3 训练系列（910_93） | √ |
| Atlas A3 推理系列 | √ |
| Ascend 950PR / 950DT（arch35） | 由仓内既有 arch35 `icamin` 提供，本任务不改 |

## 算子约束限制

- 仅支持 COMPLEX64 输入、INT32 标量输出。
- 仅支持 `incx ≥ 1`；负步长不反向遍历（quick return）。
- `n ≤ INT32_MAX`；`(n-1)*incx` 必须 ≤ `INT32_MAX`，否则 `INVALID_VALUE`。
- 不保证 denormal（次正规数）输入与 CPU golden 逐位一致（NPU 向量运算可能 FTZ）；任务用例不含次正规数，见风险 R7。
- 单次调用不支持多 stream 并发写同一 `result`（调用方保证）。

---

# 特性交叉分析（required）

## 与 stream / 异步的依赖

- 计算与 quick return 均绑定 `handle->stream`；接口异步返回，调用方需在读取 `result` 前同步 stream。
- 不创建/销毁 stream，不做 host-device 同步；quick return 的 `aclrtMemsetAsync` 也落在同一 stream，保证顺序。

## 与 workspace 的依赖

- 多核路径使用 handle workspace（`aclblasSetWorkspace` 可注入用户 buffer），需求 < 1KB；单核路径不用 workspace。
- workspace 布局按 64-float 对齐，不复用其他算子的区域（每次调用覆盖自己的槽），并发调用同一 handle 的不同算子仍由 stream 串行化。

## 确定性与并发

- 结果只依赖输入与 `(n, incx)`，与核数/切分/归约顺序无关（无 NaN 归约域 + 字典序合并）。
- 同一 handle 上的多次调用按 stream 顺序执行；不同 handle 之间无共享状态。

## 功能失效模式

| 失效 | 表现 | 防护 |
|---|---|---|
| `(n-1)*incx` 溢出 | 地址回绕/越界 | host int64 计算 + `INT32_MAX` 上限 |
| `result` 空 | host 崩溃 | 校验先于 quick return |
| 首元素 NaN 未特判 | 结果变成后续有限值索引 | sanitize 前记录元素 0 并强制 1 |
| NaN 未 sanitize | 切分改变结果 | 每 tile 归约前 `Select` 替换 +Inf |
| pad lane 参与胜出 | 小 n 返回错误索引 | pad `+Inf`；lane 序号取最小 |
| 核数/block 映射错误 | 部分切片未扫描 | 首轮 probe 验证；tiling/启动常量集中可调 |
| quick return 写 host 指针 | runtime 错误 | 测试 wrapper 为 quick return 分配 device result（见测试方案） |

---

# 可维可测分析（required）

## 精度标准

| 项 | 标准 |
|---|---|
| 输出比对 | `EXPECT_EQ(npu_result, golden)`，整数索引 **bit-exact**，无容差 |
| golden | 测试工程内 cblas 风格 CPU 循环（`icamin_golden.h`），严格 `<`，首元素种子 |
| 输入 dtype 档 | COMPLEX64（分量按 FLOAT32 档生成：均匀/正态/全零/交替/极端/Inf/NaN） |
| NaN/Inf | 与 golden 完全一致：首元素 NaN → 1；NaN 不更新；全 NaN → 1 |
| quick return | `result == 0` 且返回 `SUCCESS` |
| 判定 | 单用例通过即 matched_ratio = 1；任一不等即失败 |

## 性能标准

| 项 | 内容 |
|---|---|
| 设备 | 任务书推荐 Atlas 800T A2 (910B3)；当前可用 A3 单卡，数据需注明设备 |
| 口径 | `msprof op` 单算子 task duration（us），warmup 5 次不计入，有效样本 ≥10 取算术平均；GTest `[PERF] avg_us=`（aclrtEvent，warmup 5~10、100 次）用于脚本判定 |
| 判据 | `NPU_avg_us ≤ gpu_ms * 1000 / 0.8`（单 case） |
| 基线 | 任务包 `gpu_baseline.csv`（A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4） |

判据折算（`gpu_ms → 阈值 us`）：

| n | incx | gpu_us | 阈值 us（/0.8） |
|---|---|---|---|
| 1 | 1 | 8.28 | 10.35 |
| 64 | 1 | 8.11 | 10.13 |
| 1024 | 1 | 13.40 | 16.75 |
| 65536 | 1 | 14.59 | 18.24 |
| 262144 | 1 | 15.55 | 19.43 |
| 1048576 | 1 | 18.16 | **22.70** |
| 2097152 | 1 | 21.62 | **27.03** |
| 4194304 | 1 | 36.57 | **45.71** |

实测结果（Atlas 800 A2，CANN 9.1.0 容器，单卡，`[PERF] avg_us`，warmup 10 + 100 次）：

| case | n | 阈值 us | 实测 avg_us | 余量 |
|---|---|---|---|---|
| TC_PF_1004 | 1 | 10.35 | 7.76 | 25% |
| TC_PF_1001 | 1048576 | 22.70 | 18.38 | 19% |
| TC_PF_1002 | 2097152 | 27.03 | 22.73 | 16% |
| TC_PF_1003 | 4194304 | 45.80 | 30.09 | 34% |

全量：精度 1041/1041 PASS，性能 **200/200 PASS**（`logs/perf_cann910.log`）。

## 测试方案

### 用例来源与补充

- 任务包 `work/input/test_cases/icamin_test.csv`：1200 条（1000 精度 + 200 性能），复制到
  `test/iamin/icamin/arch22/icamin_test.csv`。
- 正态分布补充：任务包 `gen_csv.py` 只生成均匀分布；按任务书 §3.5 补充 `GAUSS_NORM_5_2`
  （μ=5，σ=2，实/虚独立）用例，本实现补 40 条（`TC_GAUSS_*`），纳入精度自验证并在自测报告说明数量与覆盖。
- 负数/空指针/quick return 用例保持任务包口径；负向用例缓冲区用单元素 dummy 或 `NULLPTR`，
  不按 `(n-1)*|incx|` 预分配。
- `test/iamin/icamin/arch22/gpu_baseline.csv`：从任务包复制 200 行（仅 arch22 用例使用，避免影响 arch35）。

### GTest 设计

- `test/iamin/icamin/arch22/icamin_test.cpp`：仿 `arch35/icamin_test.cpp`，CSV 驱动，
  `TEST_P` 逐条 `EXPECT_EQ(ret, expect_result)` + `EXPECT_EQ(result, golden)`。
- 负向用例：
  - `x=NULLPTR` → 传 nullptr，期望 `INVALID_VALUE`；
  - `result=NULLPTR` → 直接调用，期望 `INVALID_VALUE`（含 `n=0`/`incx<1`）；
  - `handle=nullptr` → 单独 `TEST_F`，期望 `HANDLE_IS_NULLPTR`。
- **quick return 的 result 必须是 device 指针**：本任务在 arch22 wrapper 内为所有路径分配 device result
  （当前任务包 wrapper 对 `n<=0||incx<=0` 直接传 host 指针，与 `aclrtMemsetAsync` 的设备语义冲突），
  调用后同步并 D2H 读回；仅 `SUCCESS` 时比对。
- 性能用例 `TC_PF_*`：仅当环境变量 `ICAMIN_PERF_BENCH=1` 时 bench；打印
  `[PERF] case=<name> n=<n> incx=<incx> warmup=10 iters=100 avg_us=<v> threshold_us=<t> verdict=<PASS/FAIL>`。
- 精度用例排除 `TC_PF_*`（`verify_accuracy.py` 已排除）。

### 复现步骤（交付 README）

```bash
# 精度（A2: ascend910b；A3: ascend910_93）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b --csv ./icamin_test.csv

# 性能
python verify_performance.py --repo /path/to/ops-blas --soc ascend910b

# 或直接构建
bash build.sh --soc=ascend910b --ops=icamin
./build/test/iamin/icamin/icamin_test --gtest_filter=*TC_L0*
ICAMIN_PERF_BENCH=1 ./build/test/iamin/icamin/icamin_test --gtest_filter=*TC_PF*
```

- 测试工程需新增 `test/iamin/icamin/README.md`（步骤、CSV 列、golden 口径、设备要求、常见问题）。
- `msprof op` 采集命令写入 README 与自测报告。

## 兼容性分析

- 公开 API 无新增（声明已存在），不改变其他算子。
- arch22 新增源码由 `blas/CMakeLists.txt` 的 `ARCH_SRC_FILES` 自动收集，不影响 arch35/arch20 构建。
- 测试仅新增 `arch22/` 目录与 arch22 GTest 注册；arch35 用例与判据不变。
- `blas/iamin/README.md` 更新 A2/A3 支持表与约束说明（n<0、quick return、NaN、overflow）。

---

# 性能设计

## Bound 判断

- 访存：n=1M complex = 8 MB；A2/A3 单卡 HBM/L2 带宽下数据时间预计 2~5 µs（x 在 100 次连续下发中可驻留 L2）。
- 计算：`Gather + Abs + Add + Compare/Select + ReduceMin`，约 5~6 个 V pass；V 吞吐远高于访存，预计 non-bound。
- 固定开销：kernel launch（单次）+ `SyncAll` + 核 0 合并/二次读（≤32KB）。小 n（≤256）判据 ~10.3 µs，launch 与 host 固定开销是主要矛盾。
- 结论：**大 n 访存/带宽 bound，小 n launch bound**；优化方向分别是提高 MTE2 效率与压低 host/launch 开销。

## 候选优化清单（≥5，按类别）

| 编号 | 类别 | 候选 | 预期收益 | 优先级 |
|---|---|---|---|---|
| P1 | 算法 | 单遍 value-only 归约 + 仅获胜 tile 二次读求首索引 | 避免全量第二遍；避免 calIndex 语义风险 | P0 |
| P2 | 算法/精度 | NaN→+Inf 前置 sanitize + 元素 0 NaN 特判 | 归约可结合、可任意切分（正确性门禁） | P0 |
| P3 | 切分 | AIV 满核（实测 48）+ 4 complex(32B) 对齐 + 尾核吸收；tile=4096 | 负载差 <0.05%、DMA 对齐 | P0（已实现） |
| P4 | 搬运 | TQue depth-2 双缓冲；tile 内 MTE2/V 重叠 | 隐藏搬运延迟 | P0 |
| P5 | 计算 | `Gather` 分奇偶 + `Abs + Add` 出 1-范数；无 sqrt、无标量逐元素 | 每 tile 1 次归约 | P0 |
| P6 | 计算 | `Compare(EQ)+Select+CreateVecIndex+ReduceMin` 定 first-occurrence | 确定性、与切分无关 | P0 |
| P7 | 同步 | 单 kernel + `SyncAll` + 核 0 合并（备选 FFTS） | 省一次 kernel launch | P0 |
| P8 | Host | handle workspace、tiling 按值传参、无 malloc/sync | 压低小 n 固定开销 | P0 |
| P9 | 快路径 | n≤8192 单 block；小 n 单核 + 无同步 | 小 n ≤10.3 µs | P1 |
| P10 | 计算 | `PairReduceSum` 整 tile + `Gather` 尾块（已实现，200/200 PASS）；`ReduceMin(calIndex=true)` 单指令索引（未验证） | 少 2~3 个 V pass | P10 已实现 / 其余 P1 |
| P11 | 搬运 | incx>1 的 `asc_copy_gm2ub_align` 32B-slot gather / 预打包 | 跨步路径提速（非判据） | P1 |
| P12 | 同步 | FFTS `FftsCrossCoreSync + WaitFlagDev` 替代 `SyncAll` | 降跨核栅栏开销 | P1 |
| P13 | 算法 | 双 kernel（scan + reduce）对照 | 实现简单、便于定位问题 | P2 |

## 测量与迭代

1. 固定环境：同一 A3 单卡、CANN 9.1.0、同一编译选项、同一输入；记录 commit。
2. baseline：首版 arch22 实现 + 仓内 arch22 `isamax` 参考（同族框架对照）。
3. 每轮：baseline → profiling（msprof/Event）→ bound 判断 → 单变量 A/B → 全量精度回归（1200+ 用例）→ 记录 `work/PERF.md`。
4. 报告 Avg/Median/Min；与 `gpu_baseline.csv` 按 `(n,incx)` 关联；A3 数据注明设备，910B3 待补测。
5. 失败方案（含 calIndex 路径若失败）保留在 `work/` 并在报告说明原因。

---

# 风险与备选方案

| ID | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | `ReduceMin(calIndex=true)` NaN/平局行为不可依赖（技能库负面实证） | 索引错 | 主方案不依赖：value-only + EQ/Select/Min(index) |
| R2 | 幅值计算路径 | 幅值错 | 混合策略已在 CANN 9.1.0 容器与 9.2.0-beta.1 宿主机分别通过 1041/1041 精度；尾块 Gather 覆盖非整 repeat |
| R3 | arch22 `__vector__` 的 AIV/block 映射 | 部分切片未扫描 | **已实测**：Atlas 800 A2 / Ascend910 + CANN 9.2.0-beta.1 为 1 block = 1 AIV，`numBlocks = useCoreNum`；n=16384 多核用例由失败转为通过 |
| R4 | `SyncAll` 与空闲 AIV | 死等 | 单核模式无同步；多核模式 useCoreNum 取偶、所有 AIV 都参与；`coreIdx>=useCoreNum` 的分支先 `SyncAll` 再退出 |
| R5 | 任务包 wrapper 对 quick return 传 host result | `aclrtMemsetAsync` 失败/未定义 | arch22 wrapper 统一分配 device result（测试代码内修正，不改生产 API） |
| R6 | denormal/FTZ 与 CPU golden 不一致 | 极端输入索引差异 | 任务用例不含次正规数；README/报告标注；如验收要求可加 golden 侧 FTZ 说明 |
| R7 | A3 与 910B3 性能差异 | 判据设备不一致 | 报告中分别标注；910B3 数据待补测 |
| R8 | 小 n 判据 ~10.3 µs 对 launch 敏感 | 性能不达标 | 单 kernel、无 tiling memcpy、无 workspace 清零；必要时 probe `aclrtLaunchKernel` 固定开销并优化 host |
| R9 | 跨核槽可见性 | 核 0 读到旧值 | 槽写后 `PipeBarrier<PIPE_ALL>` + `SyncAll`；workspace 每核独立槽，无复用竞争 |
| R10 | 大 n 索引 float 精度 | 索引错 | 索引只在 64-lane 内用 float；全局索引用 int32 整型相加 |

## 备选实现

| 方案 | 描述 | 取舍 |
|---|---|---|
| A（主） | 单 kernel + `SyncAll` + 核 0 合并 + 获胜 tile 二次读 | 一次 launch；实现中等；已按 arch22 仓内模式验证 API |
| B | 双 kernel：scan kernel 写槽 + reduce kernel 合并/二次读 | 实现简单、调试方便；多一次 launch，小 n 可能不达标 |
| C | `ReduceMin(calIndex=true)` 每 tile 直接出 (value,index) | V 开销更低；NaN/平局语义需实测，暂列 P10 对照 |
| D | 每核完整扫描后写实际首索引（无核 0 二次读） | 逻辑最简单；每核多读一个 tile（40×32KB），大 n 多 ~16% 流量 |

---

# 交付清单

| 序号 | 交付件 | 路径 / 说明 |
|---|---|---|
| 1 | 设计文档 | 本文档 → cann-competitions PR（用户提交） |
| 2 | 算子实现 | `blas/iamin/arch22/icamin_{host,kernel,kernel.h,tiling_data.h}` |
| 3 | 测试工程 | `test/iamin/icamin/arch22/{icamin_test.cpp,icamin_test.csv,gpu_baseline.csv}` |
| 4 | 测试 README | `test/iamin/icamin/README.md`（步骤/CSV/口径/复现） |
| 5 | 构建接入 | `test/iamin/icamin/CMakeLists.txt` arch22 GTest 注册 |
| 6 | 仓内文档 | `blas/iamin/README.md` A2/A3 支持表与约束 |
| 7 | 自测报告 | 精度/性能/内存模板 + 日志（`work/`、`out/`） |
| 8 | 代码 PR | 个人 fork → ops-blas（用户授权后） |

## 参考资料

1. 任务书：`work/input/task_doc.md`
2. cuBLAS cublasIcamin：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin
3. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. 仓内参考：`blas/iamin/arch35/icamin_*`、`blas/iamax/arch22/isamax_*`、`blas/dot/arch22/sdot_*`、`blas/axpy/arch22/caxpy_*`
5. 社区任务流程：https://gitcode.com/org/cann/discussions/39
