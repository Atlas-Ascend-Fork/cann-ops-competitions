# aclblasSaxpy 算子设计文档（Atlas A2/A3 · arch22）

- 任务：2026 年 11 月社区任务 —— aclblasSaxpy 算子开发（A2/A3）
- 目标代码仓：`cann/ops-blas`，代码落位 `blas/axpy/arch22/`，测试落位 `test/saxpy/arch22/`
- 本文档提交路径：`04_tasks/01_community-task-2026/tasklist/11-aclblasSaxpy-A2A3/gcw_uZ9OegMF/docs/design.md`
- 提交人：gcw_uZ9OegMF

> 本文所有现状结论均来自对 `cann/ops-blas` 仓 master 分支源码的直接阅读，关键处标注文件与行号，便于评审复核。

---

# 需求背景（required）

## 需求来源

`aclblasSaxpy` 是 BLAS Level-1 的单精度实数向量组合算子。仓内 `include/cann_ops_blas.h:476` 已有对外声明：

```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy);
```

但 `blas/axpy/arch22/` 目录下**只有 caxpy 五个文件**（`caxpy_host.cpp` / `caxpy_kernel.cpp` / `caxpy_kernel.h` / `caxpy_tiling_data.h` / `caxpy_dispatch_policy.h`），saxpy 仅在 `blas/axpy/arch35/` 存在。即：该接口在 Atlas A2/A3（arch22）上尚无实现，本任务补齐 arch22 实现，与既有 arch35（Atlas 950PR）实现构成同族双平台覆盖。

## 背景介绍

### 算子功能定位

计算 `y ← α·x + y`，逐元素原地累加。按 BLAS 惯例以 1 基逻辑下标描述：第 `i`（`i = 1..n`）次更新使用 `x[(i-1)·incx]` 与 `y[(i-1)·incy]`；`incx`/`incy` 可为负，此时按 BLAS 语义从向量末端反向遍历。`n <= 0` 为合法 no-op。

### 现状分析

| 维度 | 仓内现状（已核对） | 本设计处理 |
| --- | --- | --- |
| arch35 实现 | `blas/axpy/arch35/saxpy_host.cpp`（5,743 B）+ `saxpy_kernel.cpp`（6,576 B），双路径：连续走 AIV 向量、跨步走 SIMT | 语义与校验对齐 arch35，工程形态对齐 arch22 |
| arch22 实现 | 缺失（目录内仅 caxpy） | 新增 saxpy 五件套 |
| 跨步路径编程模型 | arch35 用 `simt_api/asc_simt.h`、`__simt_vf__`、`asc_vf_call`、`threadIdx.x/blockDim.x`（`saxpy_kernel.cpp` 的 `saxpy_simt_compute`） | **arch22 不提供该 SIMT 向量函数模型**；仓内 arch22 兄弟算子 caxpy 全部以向量单元 + UB 直址 + host 偏移表实现，本设计沿用该路线 |
| UB 容量常量 | `blas/common/helper/kernel_constant.h` 定义 `UB_SIZE = 248 * 1024`，arch35 host 直接用它求 tile | **arch22 的 UB 预算是 192 KiB**（`blas/axpy/arch22/caxpy_kernel.cpp:30` `CAXPY_UB_BUDGET_BYTES = 192 * 1024`）。直接照搬 arch35 tiling 公式会按 248 KiB 分配缓冲，**必然溢出 UB**，这是本次移植的首要陷阱 |
| 设备代码链接 | `blas/CMakeLists.txt:65` 明确「arch22 设备代码不支持跨 TU 链接」 | kernel 以单 TU 组织，host 通过 `saxpy_kernel_do` 包装下发 |
| 测试 | `test/saxpy/` 下只有 `arch35/`，共享 `saxpy_golden.h` / `saxpy_param.h`；`test/saxpy/CMakeLists.txt` 仅在 `SOC_VERSION STREQUAL "ascend950"` 时 `ops_blas_add_gtest_tests(... saxpy_test)`，其它平台退回 `ops_blas_add_tests` | 新增 `test/saxpy/arch22/`，并在 CMake 的 arch22 分支补 GTest 注册 |
| CPU golden | `test/saxpy/saxpy_golden.h` 的 `aclblasSaxpy_cpu` 直接调用 `cblas_saxpy` | 精度与语义以 netlib/CBLAS 行为为唯一基准 |

### 功能分析

1. 连续（`incx == 1 && incy == 1`）高性能路径。
2. 跨步路径：正/负步长、x 与 y 步长不等、单侧跨步。
3. 参数校验与返回码语义与 arch35 / CBLAS 一致。
4. `n <= 0`、`alpha == 0` 等快速返回口径明确。
5. A2/A3 上可构建、可注册、可跑 GTest。

## 业界对标

- **Netlib BLAS `saxpy.f`**：`IF (N.LE.0) RETURN`、`IF (DA.EQ.ZERO) RETURN`（alpha 为 0 时直接返回，**不读取 x**）、`incx`/`incy` 为 0 属非法。
- **cuBLAS `cublasSaxpy`**：参数序列与返回码语义对标。
- 仓内 `cblas_saxpy`（golden 使用）即上述语义的可执行参照。

---

# 需求分析（required）

## 需求描述

在 `blas/axpy/arch22/` 实现 `aclblasSaxpy`，使其在 Atlas A2/A3 上：

- 接口签名复用 `include/cann_ops_blas.h:476` 既有声明，**不另立私有平行接口**。
- 支持 `n > 0`、`alpha` 为 host 侧 `const float*`、`x`/`y` 为 device 内存、`incx`/`incy` 为非零整数（含负值）。
- 返回码：`handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`alpha`/`x`/`y` 为空指针或 `incx`/`incy` 为 0 → `ACLBLAS_STATUS_INVALID_VALUE`；核数获取失败 → `ACLBLAS_STATUS_INTERNAL_ERROR`；`n <= 0` → `ACLBLAS_STATUS_SUCCESS`（no-op）。
- 数值结果与 `cblas_saxpy` 对齐（FP32，逐元素一次乘加，无归约）。
- 交付 host / kernel / tiling / dispatch 四件 + CMake 接入 + `test/saxpy/arch22/` 测试。

## 需求拆解

| 编号 | 子需求 | 设计落点 |
| --- | --- | --- |
| R1 | 接口与返回码语义 | 「接口规格」「数据检测」 |
| R2 | 连续路径高性能实现 | 「kernel 侧-Dense 路径」 |
| R3 | 跨步路径（无 SIMT） | 「kernel 侧-Strided 路径」「host 侧-偏移表」 |
| R4 | 负步长与地址溢出安全 | 「host 侧-索引映射与溢出校验」 |
| R5 | arch22 UB 预算与分块 | 「host 侧-tiling」 |
| R6 | 快速返回口径（n<=0 / alpha==0） | 「算子约束限制」 |
| R7 | 构建与测试接入 | 「工程落位」 |
| R8 | 精度与性能验收 | 「可维可测分析」 |

---

# 详细设计（required）

## 算子分析

### 数学公式

```
for i in 0 .. n-1:
    kx = (incx > 0) ? i * |incx| : (n - 1 - i) * |incx|
    ky = (incy > 0) ? i * |incy| : (n - 1 - i) * |incy|
    y[ky] = alpha * x[kx] + y[ky]
```

- 每个逻辑下标 `i` 唯一映射到一个物理 `x` 下标与一个物理 `y` 下标（`inc != 0` 保证单射）。
- 无归约、无跨元素依赖；**只要按逻辑下标划分核间任务，各核写回的物理 y 集合互不相交**，因此不需要原子操作，结果天然确定。
- 计算强度极低（每元素 1 乘 1 加、12 B 访存），属**访存受限**算子。

### 支持数据类型

| 项 | 支持 |
| --- | --- |
| x / y | FLOAT32（`float`） |
| alpha | host 侧 `const float*`，解引用后以 FP32 参与计算 |
| n / incx / incy | `int`（tiling 内部以 `int64_t` 承载步长，避免中间乘法溢出） |

不支持 half / bf16 / 复数（复数由同目录 caxpy 覆盖）。

### 支持形状

一维向量，`n` 上限受 `uint32_t` 与物理地址跨度约束（见溢出校验）。`x`/`y` 的物理跨度为 `(n-1)·|inc| + 1` 个元素，可以远大于 `n`。

## 算子实现

### 接口规格

| 参数 | 含义 | 位置 | 约束 |
| --- | --- | --- | --- |
| handle | 库上下文（内部携带 stream） | host | 非空，否则 `HANDLE_IS_NULLPTR` |
| n | 逻辑元素个数 | host | `n <= 0` 直接成功返回 |
| alpha | 标量系数指针 | host | 非空；`*alpha == 0.0f` 时成功返回且不读 x |
| x | 输入向量 | device | 非空；物理跨度须可寻址 |
| incx | x 步长 | host | 非 0，可负 |
| y | 输入/输出向量（原地） | device | 非空；物理跨度须可寻址 |
| incy | y 步长 | host | 非 0，可负 |

### 工程落位

```
blas/axpy/arch22/
  saxpy_host.cpp            # 入口 + 校验 + tiling + 偏移表生成 + 下发
  saxpy_kernel.cpp          # 单 TU：dense / strided 两个 __global__ kernel + saxpy_kernel_do
  saxpy_kernel.h            # saxpy_kernel_do 声明
  saxpy_tiling_data.h       # host/device 共享的唯一 tiling 定义
  saxpy_dispatch_policy.h   # 步长分类、变体选择、tile 容量公式（constexpr）
test/saxpy/arch22/          # GTest 用例
test/saxpy/CMakeLists.txt   # 补 arch22 分支的 ops_blas_add_gtest_tests 注册
```

- tiling 结构**只在 `saxpy_tiling_data.h` 定义一次**，host 与 kernel 同头文件按字段名访问，禁止按硬编码字节偏移读取（沿用 `caxpy_tiling_data.h` 头部注释所强调的防漂移约定）。
- arch22 设备代码不支持跨 TU 链接（`blas/CMakeLists.txt:65`），故 dense 与 strided 两个 kernel 与全部 device 辅助函数放在同一 `saxpy_kernel.cpp` 内。
- `blas/CMakeLists.txt` 中按 `SOC_VERSION` 过滤源文件的逻辑需把 `blas/axpy/arch22/saxpy_*.cpp` 纳入 arch22 构建；单算子构建场景保留 host 与 kernel 两个源文件。

### host 侧 tiling

#### 索引映射与溢出校验

移植 arch35 的校验（`ValidateSaxpyParams`：三个指针非空、`incx`/`incy` 非 0），并**补上 arch22 caxpy 已有而 arch35 saxpy 缺失的健壮性检查**（参照 `blas/axpy/arch22/caxpy_host.cpp` 的 `AbsIncrement` / `CheckedVectorSpan` / `PhysicalIndex`）：

- `AbsIncrement`：以 `-(inc+1)+1` 求绝对值，避免 `INT64_MIN` 取负溢出。
- `CheckedVectorSpan`：校验 `(n-1)·|inc| + 1` 不溢出 `uint64_t`，且 `span·sizeof(float)` 不超出 `size_t` 可寻址范围；失败返回 `INVALID_VALUE`。
- `PhysicalIndex(logical, n, inc)`：负步长时映射为 `(n-1-logical)·|inc|`，与 arch35 `saxpy_simt_compute` 中「issue #483：保留步长符号」的口径完全一致。

#### 分核策略

- 核数取 `GetAivCoreCount()`，为 0 时返回 `INTERNAL_ERROR`。
- 上限沿用 arch22 caxpy 的 `DEFAULT_VECTOR_NUM = 40`（A2/A3 向量核规模），并与 `n` 取小：`useCoreNum = min(coreNum, n, 40)`，至少 1。
- **优先满核、余块给前几核**：`base = n / useCoreNum`，`rem = n % useCoreNum`，前 `rem` 个核各 `base+1`，其余 `base`。
- 与 arch35 不同，**不在 tiling 里下发 `startOffset[64]` / `calCount[64]` 两个数组**（arch35 `SaxpyTilingData` 因此固定膨胀 512 B）；改为下发 `base`/`rem`，kernel 内由 `GetBlockIdx()` 反算：`start = idx*base + min(idx, rem)`，`count = base + (idx < rem ? 1 : 0)`。减少 host→device 常量搬移，且不受 64 核上限硬编码约束。

#### 数据分块与 UB 优化

**UB 预算以 arch22 为准：`SAXPY_UB_BUDGET_BYTES = 192 * 1024`**（与 `caxpy_kernel.cpp:30` 一致），不复用 `kernel_constant.h` 的 `UB_SIZE = 248 * 1024`。

- **Dense 路径**：saxpy 是实数原地累加，`y` 缓冲既可作输入也可作输出，**无需独立的 out 区**（这是相对 caxpy 的简化：caxpy 因复数交错→平面转换才需要 out0/out1）。因此只需 `x` 与 `y` 各一对 ping-pong：

  ```
  regions = 2 (x, y) × BUFFER_NUM(2) = 4
  tileFloats = floor((SAXPY_UB_BUDGET_BYTES - reserve) / (4 × sizeof(float)))
  tileFloats = alignDown(tileFloats, 8)        // 32 B / sizeof(float)
  ```

  取 `reserve = 0` 时 `tileFloats = 12288`（4 × 48 KiB = 192 KiB，恰好占满）；实现中保留少量 reserve 给尾块 padding 与临时量，建议 tile 取 **9728（38 KiB/区，与 caxpy 分区粒度一致，4×38 KiB=152 KiB，留 40 KiB 余量）**，两者均以公式求解、不写死魔数。
- **Strided 路径**：需要 `x` 紧凑区、`y` 紧凑区、span 原始读区（lowScratch）、偏移表区。span 读会覆盖 `(tile-1)·|incx| + 1` 个物理元素，故 tile 容量随步长收缩：

  ```
  maxSpanFloats = regionBytes / sizeof(float)
  tileCapacity(|incx|) = min(TILE_COUNT_CAP, (maxSpanFloats - 1) / |incx| + 1)
  ```

  与 `caxpy_dispatch_policy.h` 的 `CaxpySharedTileCapacity` 同构，只是元素字节数由 8 改为 4，因而同 UB 下可容纳的跨步 tile 是 caxpy 的两倍。`TILE_COUNT_CAP` 取 1024（caxpy 复数用 512）。
- **尾块**：每核最后一块 `count % tile`。对齐部分用 `DataCopy`，非 32 B 对齐尾部用 `DataCopyPad` + `DataCopyExtParams{1, tailBytes, 0,0,0}` + `DataCopyPadExtParams<float>{true, 0, padNum, 0.0f}`，与 arch35 `SingleIteration` 完全一致；写回同样用 `DataCopyPad` 只写 `tailBytes`，**不写 padding**，避免踩坏相邻元素。
- **burst 限制**：arch22 caxpy 的搬移辅助函数以 `maxBurstCount = 4088` 为单次上限（`caxpy_kernel.cpp:108/158`），saxpy 的搬移循环沿用同一上限分次下发。
- 双缓冲：`BUFFER_NUM = 2`（`kernel_constant.h` 已定义），CopyIn / Compute / CopyOut 三级重叠。arch35 的 `SaxpyAIV` 用单个 `TBuf<VECIN>` 且无 ping-pong，搬移与计算串行；本设计在 arch22 上改为 ping-pong，是**有依据的性能改进**，不改变数值语义。

#### 偏移表（跨步路径核心）

沿用 arch22 caxpy 的思路：**地址计算全部放在 host，device 热路径只查表**。

- host 为每个 tile 生成三张 `uint32_t` 字节偏移表：`xGatherOffset[i] = PhysicalIndex(start+i, n, incx) * 4`、`yGatherOffset[i]`、`yScatterOffset[i]`（后两者同值，分开命名以便 x/y 步长不同时独立生成）。
- 表存放在 GM workspace（`saxpy_kernel_do` 的 `workSpace` 形参，arch35 传 `nullptr`，arch22 跨步路径需要实际分配），kernel 启动后一次性搬入 UB 偏移表区。
- 实数 saxpy **不需要 caxpy 的交错→平面转换表**（`GenComputeOffsets` 那套 real/imag 分离），表规模与生成成本显著低于 caxpy。
- workspace 大小：`align32(tileCapacity × 3 × sizeof(uint32_t))`，由 host 计算并通过既有 workspace 机制申请；dense 路径 workspace 为 0。

#### tilingkey / 变体规划

`SaxpyStrideClass`（简化自 caxpy 的 6 类，实数无需区分复数交错）：

| 类 | 条件 | 变体 |
| --- | --- | --- |
| DENSE | `incx==1 && incy==1` | `DENSE_PIPELINED` |
| X_STRIDED | `incx!=1 && incy==1` | x 查表 gather，y 连续 `DataCopy`/`DataCopyPad` |
| Y_STRIDED | `incx==1 && incy!=1` | x 连续，y 查表 gather + scatter |
| DUAL_STRIDED | 两者均 `!=1` | 双侧查表；`|incx|` 小则走 span-read 共享流水，否则走 disjoint 稀疏读 |
| NEGATIVE_MIXED | 任一步长为负 | 同上，索引映射统一走 `PhysicalIndex` |

- 当某核 `count <= SAXPY_SCALAR_MAX_PER_BLOCK`（取 16，与 `CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK` 一致）时走标量循环，避免为极小批量建流水的固定开销。
- 变体选择全部在 host 完成并以 `tiling.variant` 下发，kernel 内按常量分支裁剪，热循环无运行时判断。

#### 数据检测

host 入口顺序（**顺序本身是语义的一部分**）：

1. `handle == nullptr` → `HANDLE_IS_NULLPTR`；
2. `n <= 0` → `SUCCESS`（no-op，先于其它校验，与 arch35 及 golden 一致）；
3. 指针非空 + `incx`/`incy` 非 0 → 否则 `INVALID_VALUE`；
4. 跨度溢出校验 → 否则 `INVALID_VALUE`；
5. **`*alpha == 0.0f` → `SUCCESS`，不读取 x、不修改 y**；
6. 求 tiling、生成偏移表、下发 kernel。

> 第 5 步是相对 arch35 的**有意行为修正**，理由见「算子约束限制」第 2 条。

### kernel 侧

#### Init

- 读取共享 `SaxpyTilingData`（`totalN`、`base`、`rem`、`useCoreNum`、`tile`、`alpha`、`incx`、`incy`、`variant`、`tileCapacity`、workspace 地址）。
- 由 `GetBlockIdx()` 反算本核 `[start, start+count)`；`count == 0` 直接返回。
- `GlobalTensor<float>` 绑定 x/y：dense 路径长度为 `n`（arch35 的 AIV kernel 仅用于 dense，故其以 `totalN` 绑定是正确的）；**跨步变体必须按 `CheckedVectorSpan` 求得的物理跨度绑定**，否则 gather/scatter 会越界。
- 按变体初始化 UB 分区（dense：x0/x1/y0/y1；strided：x 紧凑 / y 紧凑 / lowScratch / 偏移表），跨步变体先把偏移表从 GM 搬入 UB。

#### Process — Dense 路径

```
for each tile (ping-pong slot = i % 2):
    DataCopy(xLocal[slot], xGM[cur], len)          // 尾块用 DataCopyPad
    DataCopy(yLocal[slot], yGM[cur], len)
    Axpy(yLocal[slot], xLocal[slot], alpha, len)   // 原地：y = alpha*x + y
    DataCopy(yGM[cur], yLocal[slot], len)          // 尾块用 DataCopyPad，只写 len
    cur += len
```

- 直接复用 Ascend C `Axpy` 向量接口（arch35 已验证可用），无需手写 `Mul`+`Add`，减少指令数。
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

#### Process — Strided 路径

```
for each tile:
    if 使用 span-read:
        一次连续读入覆盖本 tile 的物理 span 到 lowScratch
        按 xGatherOffset 从 lowScratch 紧凑化出 xLocal
    else:
        按 xGatherOffset 稀疏 gather 出 xLocal
    按 yGatherOffset gather 出 yLocal（incy==1 时退化为连续 DataCopy）
    Axpy(yLocal, xLocal, alpha, len)
    按 yScatterOffset 写回 yGM（incy==1 时退化为连续 DataCopy/DataCopyPad）
```

- **不使用原子操作**：逻辑下标核间不相交 + `inc != 0` 保证物理下标单射 ⇒ 各核写回集合互不相交。这与 caxpy 的 `*_PIPELINED_ATOMIC` 变体不同——caxpy 需要原子是因为复数 span 写回会触及邻居元素；saxpy 只 scatter 本核拥有的偏移，故可做到**无原子、可复现**。若后续改用「整 span 写回」优化，则必须改为原子加，本设计不采用该路线。
- 负步长完全由 host 偏移表吸收，kernel 内无符号判断。
- 尾块：`len` 为实际元素数，gather/scatter 表只生成 `len` 项；连续退化分支用 `DataCopyPad` 处理非对齐。

## 支持硬件

| 平台 | 支持 | 说明 |
| --- | --- | --- |
| Atlas A2（arch22） | 支持 | 本任务验收平台 |
| Atlas A3（arch22） | 支持 | 同 arch22 实现，UB 预算与核数上限按仓内 arch22 常量 |
| Atlas 950PR（arch35） | 不在本次范围 | 已有独立实现，本任务不修改 arch35 代码 |

## 算子约束限制

1. `n <= 0`：no-op，返回 `SUCCESS`，不校验指针（与 arch35、golden 一致）。
2. **`*alpha == 0.0f`：直接返回 `SUCCESS`，不读 x、不改 y。** arch35 现有实现无此分支，会实际执行 `Axpy`；当 x 含 NaN/Inf 时 `0·NaN + y = NaN` 会**污染 y**，而 netlib `saxpy.f` 与 golden 使用的 `cblas_saxpy` 均在此情形下保持 y 不变。因此该早返回既是标准口径，也是一处正确性修正，需在 UT 中以「alpha=0 + x 含 NaN」用例固化。
3. `incx == 0` 或 `incy == 0`：`INVALID_VALUE`（不支持常数向量广播）。
4. `x` 与 `y` 内存重叠：行为未定义（与 BLAS 一致）。span-read 路径下重叠会读到本核已改写的值，明确列为不支持场景。
5. 物理跨度溢出可寻址范围：`INVALID_VALUE`，不下发 kernel。
6. 仅 FP32；不支持 half/bf16/复数。
7. 确定性：无原子、核间划分固定，同输入多次运行 bitwise 一致。
8. 不承诺跨 arch22/arch35 的 bitwise 一致（两者 FMA 使用与分块可能不同），只承诺各自与 golden 在阈值内一致。

---

# 可维可测分析

## 精度标准

- golden：`test/saxpy/saxpy_golden.h` 的 `aclblasSaxpy_cpu` → `cblas_saxpy`，**复用现有 golden，不新造基准**。
- 误差来源只有一处：`Axpy` 是否使用 FMA（`a*x+y` 融合）与 CPU 侧分离乘加之间的最多 1 ulp 差异；无归约、无类型转换、无累加顺序问题。
- 判据：优先要求 **bitwise 一致**；若平台 `Axpy` 走 FMA 导致末位差异，则按「相对误差 ≤ 1 ulp（FP32 即 2^-23 量级）」验收，并在测试报告中记录实际使用的阈值与原因。
- 用例矩阵（`test/saxpy/arch22/`，参数化复用 `saxpy_param.h`）：

| 维度 | 取值 |
| --- | --- |
| n | 0、-1、1、7、8、9、127、128、1023、1024、9728、9729、12288、12289、1e6、跨 32 核边界值 |
| incx / incy | (1,1)、(1,2)、(2,1)、(2,3)、(7,7)、(1,-1)、(-1,1)、(-2,-3)、(1024,1)、(1,1024) |
| alpha | 0.0、1.0、-1.0、0.5、1e-8、1e8、NaN（x 含 NaN 时）、Inf |
| 数据 | 全 0、随机均匀、含 NaN、含 ±Inf、次正规数、大数抵消 |
| 对齐 | 首地址 32 B 对齐 / 非对齐；n 非 8 整数倍 |
| 校验 | handle 空、alpha/x/y 空、incx=0、incy=0、跨度溢出 |
| 快速返回 | n<=0 不修改 y；alpha=0 且 x 含 NaN 时 y 必须逐位不变 |

- 每个用例同时校验**返回码**与**y 的物理布局**（按 `PhysicalIndex` 逐点比对，且校验未被本算子触碰的元素保持原值，防止 span 写回越界）。

## 性能标准

- 访存量（FP32，n 元素，dense）：读 x `4n` + 读 y `4n` + 写 y `4n` = **`12n` 字节**；计算量 `2n` FLOP，算术强度 ≈ 0.17 FLOP/B，**明确访存受限**。
- 理论时间 `T_theory = 12n / BW_peak`，`BW_peak` 取验收平台实测可达带宽（以 msprof 实测为准，本文不写死数值）。
- 验收判据：有效带宽 ≥ 任务书要求的 bound 占比（**11 月任务书未在仓内 `docs/` 下发布，目录仅有 202604/202605/202607/202608；具体百分比以社区任务发放页 gitcode.com/org/cann/discussions/22 的任务书为准**，编码阶段确认后填入验收报告）。
- 测量：msprof 取 kernel 耗时，排除 host 校验/偏移表生成/workspace 申请；同 shape 重复 100 次取中位数；dense 与 strided 分别测。
- 性能用例：`n = 1024 / 1e6 / 1e7`，dense 与 `(incx,incy) = (2,3)`、`(1024,1)` 各一组。
- 风险与对策：
  1. **跨步路径 host 生成偏移表的开销**随 n 线性增长，可能超过 kernel 本身 → 表按 tile 复用（同一 tile 内偏移规律一致时只生成一份基表 + 增量），并在 host 侧缓存；小 n 直接走标量变体。
  2. span-read 在 `|inc|` 大时读入大量无用数据 → 由 `tileCapacity(|inc|)` 收缩 tile，超阈值切 disjoint 稀疏读。
  3. 单缓冲导致搬移/计算串行 → ping-pong 双缓冲（相对 arch35 的改进点）。
  4. 极小 n 下 kernel launch 占比高 → `useCoreNum = min(coreNum, n)`，必要时单核。
- **本文不提供任何实测性能数字**，全部指标在 Atlas A2/A3 上实测后填入验收报告。

## 兼容性分析

- 纯新增 arch22 实现，**不修改 arch35 任何文件**，不影响 950PR 既有行为。
- 对外接口复用既有声明，上层调用无需改动；A2/A3 上从「无实现/链接失败」变为可用。
- `alpha == 0` 早返回是向 netlib/golden 口径收敛，属修正而非破坏性变更；若评审认为应与 arch35 保持完全一致，可将该分支置于 dispatch policy 开关后，但默认建议开启（理由见约束第 2 条）。
- 构建兼容：遵循 arch22 单 TU 约束与 `blas/CMakeLists.txt` 的 `SOC_VERSION` 源文件过滤；`test/saxpy/CMakeLists.txt` 补 arch22 GTest 注册后，A2/A3 不再退回非 GTest runner。
- 依赖接口均为仓内 arch22 已在用的能力（`Axpy`、`DataCopy`、`DataCopyPad`、`GetAivCoreCount`、UB 直址搬移辅助），不引入实验性接口、不依赖 SIMT。

---

# 交付件清单

1. 本设计文档（本次 PR 交付物）。
2. 评审通过后：`blas/axpy/arch22/saxpy_{host.cpp,kernel.cpp,kernel.h,tiling_data.h,dispatch_policy.h}`。
3. `blas/CMakeLists.txt` arch22 源文件接入。
4. `test/saxpy/arch22/` 用例 + `test/saxpy/CMakeLists.txt` arch22 GTest 注册。
5. `test/saxpy/README` 或 arch22 说明：构建命令、跑测命令、实测精度/性能报告。

## 待确认项（不阻塞设计评审）

1. 11 月任务书的性能 bound 百分比与精度阈值原文（仓内 `docs/` 无 202611 目录，需从任务发放页确认）。
2. arch22 上 `Axpy` 是否走 FMA（决定 bitwise 还是 1 ulp 判据）。
3. `alpha == 0` 早返回是否被评审接受为对 arch35 的有意差异。
4. 跨步路径 workspace 的申请方式是否需与 caxpy 的 `gmAug` 机制统一（本设计倾向统一，便于后续维护）。