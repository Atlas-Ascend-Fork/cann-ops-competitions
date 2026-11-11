# aclblasSaxpy 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（AGENTS.md §F0 模板纪律）。
> 数据来源：`spec/contract.yaml`（任务书逐条溯源）+ `spec/SAXPY_A2A3-深挖要点.md`（对 ops-blas master@5bd9514 现场核验）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasSaxpy 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/axpy/arch22/`（实现）与 `test/saxpy/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度实数 AXPY 能力，
对标 cuBLAS `cublasSaxpy`（cuBLAS Library 文档 §2.5.4）。

## 背景介绍

### 算子功能定位

BLAS Level-1 逐元素向量运算，实现 `y = alpha * x + y`（y 原地覆写）。

数学公式：

```
y[j] = alpha * x[k] + y[j],   k = 1 + (i-1)*incx,  j = 1 + (i-1)*incy,  i = 1..n
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasSaxpy | cublasSaxpy | 单精度实数 AXPY，y 原地更新，支持正负步长 |

### 业界对标

- 对标接口：cuBLAS `cublasSaxpy(handle, n, alpha, x, incx, y, incy)`（cuBLAS Library §2.5.4）。
- Golden 语义来源：Netlib BLAS `saxpy.f`（https://netlib.org/blas/saxpy.f）。
- 语义差异点（相对仓内 arch35 既有实现）：
  1. **步长溢出拒绝**：arch35 `saxpy_host.cpp` 未做 INT32_MIN 步长与 `(n-1)*|inc|` 溢出拒绝，
     本任务按任务书 §2.4/§2.5（issue #780/#866 口径）补充。
  2. **编程模型**：arch35 实现基于 SIMT（`simt_api/asc_simt.h`，arch35 专属），arch22 无 SIMT →
     必须改写为 SIMD（Membase/TQue/DMA + 向量指令）。
  3. **alpha == 0 早退**：Netlib `saxpy.f` 有 `IF (DA.EQ.ZERO) RETURN`；本实现须在 alpha==0 时
     y 不变且不读 x（与 golden 一致，已实测 openblas 行为确认，见深挖要点 §5.2）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**（非 aclnn 两段式）。host 入口做参数校验与 tiling 下发，
  kernel 在 handle 绑定的 stream 上异步执行。
- 代码落位：`blas/axpy/arch22/`（该族目录**已存在**，含复数实现 caxpy → **并入加文件，不新建族目录**）。
- 仓内近亲与复用方式：
  | 近亲 | 位置 | 复用方式 |
  |------|------|----------|
  | 同算子 arch35 实现 | `blas/axpy/arch35/saxpy_{host,kernel}.cpp` | 语义/校验序/tiling 形态平移；kernel 需从 SIMT 改写为 SIMD |
  | 同族 arch22 实现 | `blas/axpy/arch22/caxpy_*` | arch22 校验顺序、int64 跨度计算、负步长物理下标、步长 DMA 原语范式 |
  | arch22 实数步长范式 | `blas/spr/arch22/sspr_kernel.cpp` | 负步长逻辑序还原（ArithProgression + Gather）备用路径 |

### 接口规格

```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy);
```

- 唯一声明位置：`include/cann_ops_blas.h:472`（与 arch35 实现共用声明）。
  **本次任务不修改公共头**（任务书 §7.1 仅要求"复用既有声明、禁止另立私有平行接口"，
  未授权修改公共头 → AGENTS.md 红线 3）。
- 仓内公共头中 `x` 为**非 const** 指针（cuBLAS 为 `const float*`）；实现**不得写入 x**。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（内部携带 stream），Host 内存 | aclblasHandle_t | — | `aclblasCreate` 返回的非空句柄；为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | 向量元素个数，Host 内存 | int | int32 | 任意 int32；`n<=0` 为合法 no-op（不访问 alpha/x/y，返回成功，不报错） | — |
| alpha | 标量乘数指针，Host 内存；**值不校验**（含 0.0/1.0/Inf/NaN） | const float* | FLOAT32 | `n>0` 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` | 标量 |
| x | 输入向量，Device 内存，只读 | float* | FLOAT32 | `n>0` 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验，NaN/Inf 按 IEEE 规则自然传播 | 一维，物理长度 `1+(n-1)*|incx|` |
| incx | x 中相邻元素存储步长（元素为单位），Host 内存 | int | int32 | 非零且不为 INT32_MIN（正负步长均支持）；`n>0` 时为 0 或 INT32_MIN 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`(n-1)*|incx|` 超出 **int32** 可表示范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算） | — |
| y | 输入兼输出（原地覆写），Device 内存 | float* | FLOAT32 | `n>0` 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验 | 一维，物理长度 `1+(n-1)*|incy|` |
| incy | y 中相邻元素存储步长（元素为单位），Host 内存 | int | int32 | 同 incx | — |

### 数学公式

```
y[j] = alpha * x[k] + y[j],  k = 1+(i-1)*incx,  j = 1+(i-1)*incy,  i = 1..n
```

负步长按 Netlib 物理基址约定反向取元素：逻辑第 i 个元素（i 从 0 起）的物理下标为
`(n-1-i)*|inc|`（见 `blas/axpy/arch22/caxpy_host.cpp:101-105` `PhysicalIndex` 既有范式）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 单精度 AXPY：支持 FLOAT32 一维向量、正负步长、
原地输出；精度对齐 Netlib `cblas_saxpy`（rtol=atol=2^-13，matched_ratio≥0.99）；
性能在 Atlas 800T A2 (910B3) 上连续路径 Avg time ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `y = alpha*x + y`（原地）。golden 可白嫖 Netlib `cblas_saxpy`（仓库已有 `test/saxpy/saxpy_golden.h` + `cblas_compat.h`），无需自写参考实现 |
| 2 | 接口原型 | `aclblasStatus_t aclblasSaxpy(aclblasHandle_t, int, const float*, float*, int, float*, int)`，唯一声明于 `include/cann_ops_blas.h:472`，本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | **仅 FLOAT32**（任务书 §2.4 dtype 面个数=1；alpha/x/y 全 FLOAT32） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 实现，不受影响 |
| 6 | 精度要求 | rtol = atol = 2^-13 ≈ 1.2207e-4；required_matched_ratio = 0.99；max_abs_error ≤ 1e-2 or 32×ULP；Inf/NaN 按 §3.2 特殊值规则单独判定（任务书 §3.2，mixed_tolerance_standard，commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`；200 条性能用例**全部 incx=incy=1**（实测 CSV 全量），故性能只考核连续路径。典型 case：n=1048576 → ≤15.08us；n=2097152 → ≤20.50us；n=4194304 → ≤43.52us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase**（arch22 无 SIMT，arch35 的 `simt_api` 不可用 —— 死路①）。连续路径：DMA 搬 x/y → `Axpy` 向量指令 → DMA 写回 y；步长路径：`asc_copy_gm2ub_align`/`asc_copy_ub2gm_align` 的 src_gap/dst_gap 实现步长 DMA，负步长在 UB 内反转对齐逻辑序（死路②：不采用窗口+Gather 方案，因其需额外 UB 窗口且仅作降级备选） |
| 9 | 评测来源 | 精度 golden：Netlib BLAS 3.10.0 `cblas_saxpy`；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V7，由任务包 1200 条 CSV 全量反推验证，对 1200/1200 条 expect_result 一致）**：

| 序 | 条件 | 返回码 |
|----|------|--------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n <= 0` | `ACLBLAS_STATUS_SUCCESS`（合法 no-op，先于其余校验） |
| V3 | `n>0 && (alpha==nullptr \|\| x==nullptr \|\| y==nullptr)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `n>0 && (incx==0 \|\| incy==0)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V5 | `n>0 && (incx==INT32_MIN \|\| incy==INT32_MIN)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `n>0 && ((n-1)*\|incx\| > INT32_MAX \|\| (n-1)*\|incy\| > INT32_MAX)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V7 | 其余 | `ACLBLAS_STATUS_SUCCESS` |

> V6 阈值取 **INT32_MAX**（任务书 §2.4「超出 int32 可表示范围」口径）。注意仓内
> `blas/copy/arch35/scopy_host.cpp` 用的是 UINT32_MAX，本任务**不照抄**——CSV 用例
> TC_ED_121（n=3, incx=INT_MAX → 跨度 4294967294）期望 INVALID_VALUE，UINT32_MAX 阈值会误判为合法。

---

# 详细设计（required）

## 算子分析

### 数学公式

`y[j] = alpha * x[k] + y[j]`，其中 `k = 1+(i-1)*incx`、`j = 1+(i-1)*incy`，`i = 1..n`。
逐元素独立运算，无跨元素依赖 → 输出对同一输入确定性可复现（任务书 §2.5「确定性计算要求：不要求」）。

### 支持数据类型

alpha、x、y 统一 FLOAT32；中间乘加保持 FLOAT32，不做降精度。
特殊值（Inf/NaN）按 IEEE 754 规则自然传播：唯一例外是 alpha==0.0 的早退路径（不读 x，故不产生 0×Inf=NaN）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 | x 物理长度 `1+(n-1)*|incx|`，y 物理长度 `1+(n-1)*|incy|`；C ABI 无法推断真实分配长度，调用方保证存储有效 |
| n = 0 / n < 0 | 合法 no-op：不访问 alpha/x/y、不启动 kernel、y 保持不变、返回 SUCCESS |
| n = 1 | 单元素，步长无实际意义 |
| 步长语义 | BLAS 步长即非连续访问；正负步长均生效，负步长自物理 span 尾端起算 |
| 广播 | 不涉及（BLAS-1，参与运算的向量元素个数同为 n） |
| 0 维 Tensor | 不涉及（标量语义由 alpha 的 Host 入参承担） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_DENSE** | `incx==1 && incy==1` | 连续双缓冲流水：DataCopy x/y → `Axpy` → DataCopy y 回写 | **是**（200/200 性能用例） |
| **V_STRIDED** | 其余（正/负/混合步长） | 步长 DMA 读 x/y → 向量计算 → 步长 DMA 写回 y | 否（精度覆盖 881 条步长用例） |

数据流（V_DENSE）：

```
GM x ──DataCopy──▶ UB xBuf ─┐
                             ├─▶ Axpy(yBuf, xBuf, alpha, cnt) ─▶ UB yBuf ──DataCopy──▶ GM y
GM y ──DataCopy──▶ UB yBuf ─┘
        （双缓冲：xBuf/yBuf 各 2 槽，MTE2 与 V/MTE3 重叠）
```

数据流（V_STRIDED）：

```
GM x ──asc_copy_gm2ub_align(src_gap=(|incx|-1)*4)──▶ UB xTile ─┐
                                                               ├─▶ Axpy ─▶ UB yTile
GM y ──asc_copy_ub2gm_align(dst_gap=(|incy|-1)*4)──▶ UB yTile ─┘
        （负步长：x/y 同号 → 逻辑域一致，0 次反转；异号 → UB 内反转 x 1 次）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`（R2：动态取核数，禁止硬编码；R8：取到后先判零）。
- 每核负责**连续的逻辑区间** `[myOffset, myOffset+myCount)`，逻辑序而非物理序 → 负步长天然按 BLAS 语义反序取元素。
- 余数分配：前 `remainder` 个核各多 1 个元素。

**数据分块和内存优化策略**：
- UB 预算 `UB_SIZE = 248 KiB`（`blas/common/helper/kernel_constant.h:16`）。
- V_DENSE tileSize = 8192 float（32 KiB/缓冲）：xBuf 双缓冲 64 KiB + yBuf 双缓冲 64 KiB = 128 KiB ≤ 248 KiB。
- V_STRIDED tileSize = 1024 float：步长路径另需 UB 反转 scratch，tile 缩小以留余量。
- 尾块用 `DataCopyPad` + `DataCopyPadExtParams` 对齐补零（32 字节块 = 8 float）。

**tilingkey 规划策略**：本算子用 **tiling 字段分支**（非 tilingkey）——仓内
`agent/skills/repo-op-templates/references/simd-membase/blas/{op}/archxx/op_host.cpp`
模板注释已明确 saxpy 的"连续 + stride 双路径"用 `Cal{Op}TilingDataContiguous` /
`Cal{Op}TilingDataStrided` + `kernel_do` 内分支的合法模式。

Tiling 结构（`blas/axpy/arch22/saxpy_tiling_data.h`，host/device 共用唯一真源，按字段名访问）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `totalN` | uint32 | 逻辑元素总数 n |
| `perCoreN` | uint32 | 每核基础分配量 |
| `remainder` | uint32 | 余数（前 remainder 个核各多 1） |
| `tileSize` | uint32 | 每 tile 元素数 |
| `incx` / `incy` | int32 | 步长（带符号） |
| `alpha` | float | 标量乘数 |
| `useCoreNum` | uint32 | 参与计算的核数 |

> **R4 铁律**：TilingData **不得用数组按核分配地址**（数组长度依赖运行期核数）→ 采用
> `totalN / perCoreN / remainder`，kernel 侧自算
> `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（与 arch35 saxpy 同款，符合 R4）。

**数据检测（校验顺序与异常码，逐条对齐任务书 §2.4）**：
按 V1..V7 顺序执行，顺序不可调换——
- V2（n<=0 → SUCCESS）**必须**在 V3/V4 之前：CSV 用例 TC_ED_101/104/107/110 表明 n<=0 时
  即使 incx==0 或 x/y 为 NULLPTR 也期望 SUCCESS。
- V3 必须在 alpha==0 早退之前：alpha==nullptr 时须返回 INVALID_VALUE，不得被 0 值判断短路。
- V6 使用 int64 计算 `(n-1)*|inc|`，比较阈值 INT32_MAX。

**其他 host 侧约定**：
- workspace：本算子不需要额外 workspace → 传 nullptr（任务书 §3.4「不涉及」）。
- 同步：**不同步**（任务书 §2.5「异步执行」）。tiling 以 `const` 引用传入 `kernel_do`，
  **无 H2D 拷贝、无 aclrtMalloc、无 aclrtSynchronizeStream** → 小 n 场景下发开销最小化
  （小尺寸门槛约 8.25us，launch 开销敏感）。
- 日志：`OP_LOGE` 记录各校验失败分支，`OP_LOGD` 记录 tiling 决策。

#### kernel 侧设计

**Init/Process 两阶段**（V_DENSE）：

```
Init:   解析 tiling → 计算 myOffset/myCount → SetGlobalBuffer → InitBuffer(xBuf/yBuf)
Process:
   ┌── for each full tile ────────────────────────────────┐
   │  MTE2: DataCopy xBuf ← GM x[off..]                     │
   │  MTE2: DataCopy yBuf ← GM y[off..]                     │
   │  MTE2→V: SetFlag/WaitFlag (MTE2_V)                     │
   │  V:    Axpy(yBuf, xBuf, alpha, cnt)                    │
   │  V→MTE3: SetFlag/WaitFlag (V_MTE3)                     │
   │  MTE3: DataCopy GM y[off..] ← yBuf                     │
   │  MTE3→MTE2: SetFlag/WaitFlag (MTE3_MTE2)               │
   └────────────────────────────────────────────────────────┘
   └── 尾块：DataCopyPad 补零对齐 → Axpy(calCount) → DataCopyPad 精确写回
```

**步长路径（V_STRIDED）的符号统一处理**（关键设计）：

对步长 s、元素数 n，逻辑第 i 个元素的物理下标：
`L(i) = (s>0) ? i*|s| : (n-1-i)*|s|`

DMA 只能按物理递增方向搬运，而负步长的逻辑序是物理递减序。推导：
- x 与 y **同号** → 二者在"物理递增序"下的逻辑域一致 → 直接配对计算，**0 次反转**；
- x 与 y **异号** → 逻辑域相反 → 在 UB 内把 x **反转 1 次**即对齐（与
  `blas/axpy/arch22/caxpy_kernel.cpp:130-152` `normalize_loaded_complexes` 的 UB 内反转同范式）。

每 tile 的 DMA 基址（tile 覆盖逻辑区间 [t0, t0+cnt)）：
- x 基址 = `(incx>0) ? t0*|incx| : (n-1-t0-cnt+1)*|incx|`
- y 基址 = `(incy>0) ? t0*|incy| : (n-1-t0-cnt+1)*|incy|`

步长 DMA 参数：
- 读：`asc_copy_gm2ub_align(dst, src, n_burst=cnt, len_burst=4, 0, 0, src_gap=(|incx|-1)*4, dst_gap=0)`
- 写：`asc_copy_ub2gm_align(dst, src, n_burst=cnt, len_burst=4, 0, 0, src_gap=0, dst_gap=(|incy|-1)*4)`

> `src_gap`/`dst_gap` 步长 DMA 范式来源：`blas/axpy/arch22/caxpy_kernel.cpp:104-177`
> （arch22 同族实现，已合入）；多块步长 DMA 备选：`blas/hpr/arch22/chpr_kernel.cpp:150-167`。

**workspace 布局**：本算子无 workspace（不需要跨核 partial 或归约缓冲）。

**同步事件配对**：MTE2→V、V→MTE3、MTE3→MTE2 三段显式 SetFlag/WaitFlag，
尾块路径额外需要 UB 内标量写（反转）后的全流水栅栏。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 一次调用只发射一个 AIV kernel；核数 = min(AIV核数, n) |
| 双缓冲流水 | xBuf/yBuf 各 2 槽，MTE2 与 V/MTE3 重叠，掩盖搬运时延 |
| host 零同步、零分配 | tiling const 引用传参，无 H2D、无 aclrtMalloc、无 SynchronizeStream → 小 n 延迟最小 |
| 满 32B 对齐的 DMA | tile 对齐 8 float（32B），尾块用 DataCopyPad 精确写回，不写越界 |
| 单次 `Axpy` 向量指令 | `y = α·x + y` 融合为一条向量指令，避免 Muls+Add 两次访存 UB |

**未采用的路线及原因**：
- **SIMT**：arch22 不提供 SIMT（`asc_simd.h` 仅在 `__NPU_ARCH__==2201/3510` 分支，2201 无 simt）——死路。
- **窗口整段拷贝 + Gather 取逻辑序**（`blas/spr/arch22/sspr_kernel.cpp` 范式）：可正确实现负步长，
  但需按 |inc| 自适应缩小 tile 并额外占用 UB 窗口；仅保留为**降级方案**（当 4 字节突发 DMA 不可用时启用）。
- **arch35 版校验逻辑平移**：缺 INT32_MIN 与跨度溢出拒绝 → 不满足任务书 §2.4。

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
| 参数合法性与校验层级 | handle==nullptr → `HANDLE_IS_NULLPTR`；n<=0 → 合法 no-op（不访问 alpha/x/y）；n>0 时 alpha/x/y 为 nullptr、incx/incy 为 0 或 INT32_MIN、`(n-1)*|inc|` 超 int32 → `INVALID_VALUE`；alpha 标量值不校验 |
| 非连续 Tensor 支持 | **支持**：BLAS 步长语义即非连续访问；正负步长均生效，负步长自物理 span 尾端起算；incx==0 / incy==0 返回 `INVALID_VALUE` |
| broadcast 规则 | **不涉及**：BLAS-1 向量算子，无操作数间广播 |
| dynamic shape 要求 | **不要求**：n/incx/incy 为运行时入参，host tiling 按运行时 n 计算，任意合法规模一次适配 |
| 原地与视图语义 | **y 原地覆写**：输入旧值与输出新值复用同一入参 y 内存，不返回视图；x 只读（公共头中 x 非 const，但实现不写入） |
| 确定性计算要求 | **不要求**：逐元素运算，单次执行内确定；同输入、同步长跨次结果 bit 级一致 |
| 空 Tensor 处理 | 等价于 n=0：不访问向量数据、不启动 kernel，返回成功 |
| 0 维 Tensor 处理 | **不涉及** |
| 异步执行 | 计算经 handle 绑定的 stream 异步下发；返回成功仅表示 Host 已提交；alpha 为 Host 内存标量，host 侧同步解引用后随 tiling 下发 |

**已知限制（如实标注）**：
1. 步长路径的 UB 内反转采用标量 swap（沿用同族 caxpy arch22 既有范式），严格意义上属逐元素操作；
   该路径**不参与性能判定**（200/200 性能用例均为连续），代价可接受。
2. C ABI 无法推断 x/y 的真实分配长度，调用方须保证 `1+(n-1)*|inc|` 元素均有效。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_saxpy`（float32 直接计算），经 `test/saxpy/saxpy_golden.h` 的 `aclblasSaxpy_cpu` 调用 | 任务书 §3.1/§3.2/§7.4 |
| 普通精度（rtol） | 2^-13 ≈ 1.2207e-4 | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| 普通精度（atol） | 2^-13 ≈ 1.2207e-4 | 同上 |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | 逐元素 `\|actual - golden\| <= atol + rtol*\|golden\|` | 任务书 §3.2 |
| Inf/NaN 特殊值 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf → 视为等效判过；一方有限值另一方 Inf/NaN → 判不过；非有限值元素不计入 matched_ratio 与 max_abs_error，单独统计非有限值总数及其中通过数；用例通过 = 有限值满足 ratio/max_abs 且非有限值全部通过 | 任务书 §3.2（与仓内 `test/frame/verify.h` issue #818 同口径） |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（0.8× GPU） | 任务书 §3.3 |
| 性能采集口径 | msprof op 模式采集单算子 task duration (us)；msprof 自带 5 次 warmup 不计入样本；warmup 后连续执行 ≥10 次取有效样本（≥10 个），算术平均为 Avg time；仅返回 SUCCESS 的执行计入有效样本 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（warmup 5 后取有效采样均值），见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 + test_cases/README.md |
| 内存标准 | **不涉及**：BLAS-1 向量算子，无额外 workspace 判据 | 任务书 §3.4 |

## 测试概览

**用例分类统计**（任务包 `saxpy_test.csv` 共 1200 条，实测前缀分布）：

| 类别 | 前缀 | 条数 | 说明 |
|---|---|---|---|
| L0 基础 | TC_L0 | 8 | n ∈ {1,8} × 步长组合 |
| 尺寸扫描 | TC_SQ | 38 | 38 种尺寸（1→1048576，含质数/2的幂±1/非对齐） |
| 步长 | TC_INC | 17 | 正步长全组合、负步长、双负、正负混合 |
| alpha 取值 | TC_AB | 12 | alpha ∈ {0, ±1, ±2.5, 1e10, 1e-30, Inf, NaN} |
| 填充模式 | TC_FL | 24 | 6 种填充（零/交替/极端/Inf/NaN）× 2 尺寸 |
| 边界负向 | TC_ED | 19 | n=0/负 n quick return、零步长、空指针、INT32_MIN、跨度溢出 |
| 扩展 | TC_EX | 882 | 尺寸 × 步长 × alpha × 填充的确定性采样 |
| 性能 | TC_PF | 200 | 3 条任务书典型 case + 延迟区 + 规模扫描（**全部 incx=incy=1**） |

精度全量 1000 条 + 性能子集 200 条（任务书 §3.5 口径，dtype 面个数=1 → 全部计入 FLOAT32）。

**正态分布用例补齐（任务书 §3.5 补充要求）**：任务包 `gen_csv.py` 默认仅生成均匀分布
`RANDOM_NORM_5_5`；§3.5 要求正态分布（μ∈[-5,5]、σ∈[0.1,2]，填充 token `GAUSS_NORM_5_2`）占 50%。
实现方式：以 `gen_csv.py --dist mixed` 扩展生成器产出补充 CSV，合并入
`test/saxpy/arch22/saxpy_test.csv`，并纳入精度自验证；在自测报告中说明补齐方式、条数与覆盖情况。

**负向用例缓冲区分配规则**（任务书 §3.5 明示）：期望非 SUCCESS 的用例**不得**按
`(n-1)*|incx|` / `(n-1)*|incy|` 预分配（INT32_MIN 与溢出乘积在 int 域即溢出，元素数可达数十亿，
测试工程自身分配失败会误判接口行为）→ 以单元素 dummy 缓冲或 NULLPTR 占位；
`n=0` 或 `incx<=0`/`incy<=0` 的 quick return 用例同样不按步长分配。

**复现入口**：交付件 03「自验证步骤说明」给出 `verify_accuracy.py` / `verify_performance.py`
的完整命令行（A2 用 `--soc ascend910b`，A3 用 `--soc ascend910_93`）。

## 兼容性分析

- **新增算子，无既有接口的兼容性问题**：`aclblasSaxpy` 在 `include/cann_ops_blas.h:472` 已有声明
  （arch35 实现使用），本次**不修改公共头**（任务书 §7.1），不存在 ABI 变更。
- **对 arch35 的影响**：无。arch22 源文件位于 `blas/axpy/arch22/`，构建系统按
  `SOC_ARCH_DIRS` 选择目录（`ascend910b*`/`ascend910_93*` → arch22；`ascend950` → arch35），
  两套实现互不编译、互不覆盖。
- **对 arch22 同族 caxpy 的影响**：无。两者是同一头文件中的**不同接口**
  （`aclblasSaxpy` vs `aclblasCaxpy`），各自独立 host/kernel 文件，无共享可变状态。
- **对测试构建的影响**：`test/saxpy/CMakeLists.txt` 增加 arch22 的 GTest 注册分支；
  arch35 分支保持不变，ascend950 构建行为不变。
- **文档影响**：`blas/axpy/README.md` 中 `aclblasSaxpy` 的「产品支持情况」由
  「A2/A3：不支持」更新为「A2/A3：支持」，并同步约束说明（n<=0 no-op、incx/incy 非零且非 INT32_MIN、
  跨度溢出拒绝）。
