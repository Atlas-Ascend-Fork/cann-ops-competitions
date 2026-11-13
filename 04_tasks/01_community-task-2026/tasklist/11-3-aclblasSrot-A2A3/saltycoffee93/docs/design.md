# aclblasSrot 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《11月社区任务-aclblasSrot算子开发（A2A3）》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/rot/`、`include/cann_ops_blas.h`、`blas/rot/arch35/srot_host.cpp`、`test/rot/srot/CMakeLists.txt`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasSrot 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/rot/arch22/`（实现）与 `test/rot/srot/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度实数 Givens 平面旋转能力，
对标 cuBLAS `cublasSrot`（cuBLAS Library 文档 §2.5.8）。

## 背景介绍

### 算子功能定位

BLAS Level-1 Givens 平面旋转：旋转矩阵 G = [[c, s], [−s, c]] 同时原地作用于向量 x、y 的对应元素对。

数学公式：

```
x[k]' = c × x[k] + s × y[j]
y[j]' = −s × x[k]' + c × y[j]      （等式右端 x 取更新前值）
其中 k = 1 + (i−1)×incx, j = 1 + (i−1)×incy, i = 1..n
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasSrot | cublasSrot | 单精度实数 Givens 旋转，x/y 原地更新，支持正负步长 |

### 业界对标

- 对标接口：cuBLAS `cublasSrot(handle, n, x, incx, y, incy, c, s)`（cuBLAS Library §2.5.8）。
- Golden 语义来源：Netlib BLAS `srot.f`（https://netlib.org/blas/srot.f）。
- 语义差异点（相对仓内 arch35 既有实现）：
  1. **n<0 校验**：arch35 `srot_host.cpp` 入口对 `n <= 0` 一律成功返回（L249-251 实测）；
     本任务按任务书 §2.4/§2.5 口径区分——**n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`，仅 n=0 快速返回成功**。
  2. **编程模型**：arch35 实现基于 SIMT（tilingKey=0 连续路径为 SIMD、tilingKey=1 步长路径为 SIMT 线程循环），
     arch22 无 SIMT → 步长路径必须改写为 SIMD 步长 DMA（沿用同目录 csrot arch22 范式）。
  3. **c/s 位置自适应**：c、s 为 Host 或 Device 内存指针，运行时经 `aclrtPointerGetAttributes`
     自动判定（arch35 tiling 的 `cIsDevice/sIsDevice` 双态设计平移）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。host 入口做参数校验与 tiling 下发，kernel 在 handle 绑定 stream 上异步执行。
- 代码落位：`blas/rot/arch22/`（该目录**已存在**，当前仅 csrot 实现 → **并列加文件，不新建族目录**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同算子 arch35 实现 | `blas/rot/arch35/srot_{host,kernel,tiling_data}.*` | 校验序/溢出守卫（L116-129）/tiling 字段规划平移；SIMT 步长路径改写 |
| 同族 arch22 实现 | `blas/rot/arch22/csrot_{host,kernel}.cpp` | arch22 工程骨架、校验顺序、步长 DMA 原语范式 |
| arch22 步长 DMA 范式 | `blas/axpy/arch22/caxpy_kernel.cpp` | `asc_copy_gm2ub_align`/`ub2gm_align` src_gap/dst_gap 步长访问 |

### 接口规格

```cpp
aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
```

- 唯一声明位置：`include/cann_ops_blas.h` L459（与 arch35 实现共用声明，现场核验）。
  **本次任务不修改公共头**（任务书 §2.3/§7.1 仅要求复用既有声明、禁止另立私有平行接口 → 红线 3）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（内部携带 stream），Host 内存 | aclblasHandle_t | — | `aclblasCreate` 返回的非空句柄；为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | 向量元素个数，Host 内存 | int | int32 | 任意 int32；n=0 走 no-op 不报错；n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书 §2.4，比 arch35 更严） | — |
| x | 输入兼输出（原地更新），Device 内存 | float* | FLOAT32 | n>0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不校验，NaN/Inf 按 IEEE 规则自然传播 | 一维，物理长度 `1+(n−1)*|incx|` |
| incx | x 相邻元素存储步长（元素为单位），Host 内存 | int | int32 | 非零且非 INT32_MIN；为 0/INT32_MIN 或 `(n−1)*|incx|` 超 int32 范围返回 `ACLBLAS_STATUS_INVALID_VALUE` | — |
| y | 输入兼输出（原地更新），Device 内存 | float* | FLOAT32 | 同 x（步长为 incy） | 一维，物理长度 `1+(n−1)*|incy|` |
| incy | y 相邻元素存储步长，Host 内存 | int | int32 | 同 incx | — |
| c | Givens 余弦元素指针，Host 或 Device 内存（`aclrtPointerGetAttributes` 判定） | const float* | FLOAT32 | n>0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；标量值不校验（含 0/±Inf/NaN） | 标量 |
| s | Givens 正弦元素指针，Host 或 Device 内存（同 c） | const float* | FLOAT32 | 同 c | 标量 |

### 数学公式

```
x[k]' = c·x[k] + s·y[j]
y[j]' = −s·x[k](旧) + c·y[j]
```

负步长按 Netlib 物理基址约定反向取元素：逻辑第 i 个元素（0 起）物理下标 `(n−1−i)*|inc|`。
逐元素独立，元素对内先读后写（x 旧值用于两式），元素对之间无依赖。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 单精度 Givens 旋转：FLOAT32 一维向量、正负步长、
原地双输出；精度对齐 Netlib `cblas_srot`（rtol=atol=2^-13，matched_ratio≥0.99）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | Givens 旋转 `[[c,s],[−s,c]]·(x,y)`（原地，x 旧值语义）。golden 可白嫖仓内 `srot_golden.h` 的 `cblas_srot`（Netlib 3.10.0，任务书 §3.2/§7.4），无需自写参考实现 |
| 2 | 接口原型 | `aclblasStatus_t aclblasSrot(aclblasHandle_t, int, float*, int, float*, int, const float*, const float*)`，声明于 `include/cann_ops_blas.h` L459（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | **仅 FLOAT32**（任务书 §2.4 dtype 面个数=1；x/y/c/s 全 FLOAT32） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 实现，不受影响 |
| 6 | 精度要求 | rtol = atol = 2^-13 ≈ 1.2e-4；required_matched_ratio = 0.99；max_abs_error ≤ 1e-2 or 32×ULP；Inf/NaN 按 §3.2 等效规则单独判定（任务书 §3.2，mixed_tolerance_standard，opbase commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定、无均值兜底。典型 case（实测 CSV 读数）：n=1048576 → ≤20.71us；n=2097152 → ≤31.34us；n=4194304 → ≤61.03us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase**（arch22 无 SIMT，arch35 SIMT 步长路径不可平移——死路①）。连续路径：DMA 搬 x/y → 向量指令 `Muls+Add`（或 Mul/Add 融合序列）→ DMA 写回；步长路径：caxpy arch22 的 src_gap/dst_gap 步长 DMA（死路②：不采用 Gather 窗口方案，仅作降级备选） |
| 9 | 评测来源 | 精度 golden：Netlib BLAS 3.10.0 `cblas_srot`（`srot_golden.h`）；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V6，对齐任务书 §2.4/§2.5 与 arch35 守卫，顺序不可调换）**：

| 序 | 条件 | 返回码 |
|----|------|--------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE`（任务书 §2.4；arch35 为 n≤0 一并成功，arch22 收紧，勿照抄） |
| V3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（合法 no-op：不访问 x/y/c/s、不启动 kernel） |
| V4 | `n>0 && (x==nullptr \|\| y==nullptr \|\| c==nullptr \|\| s==nullptr)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V5 | `n>0 && (incx==0 \|\| incy==0 \|\| incx==INT32_MIN \|\| incy==INT32_MIN)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `n>0 && ((n−1)*|incx| 或 (n−1)*|incy| 超 int32 范围（int64 计算，含 <INT32_MIN 下界，对齐 arch35 L116-129 双边守卫）)` | `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 详细设计（required）

## 算子分析

### 数学公式

`[[c,s],[−s,c]]·(x[k], y[j])`，元素对内两式共享 x 旧值；逐元素对独立、无跨元素依赖。
确定性：单次执行内确定；同输入、同步长跨次执行 bit 级一致（任务书 §2.5"不要求"但天然满足）。

### 支持数据类型

x、y、c、s 统一 FLOAT32；中间乘加保持 FLOAT32，不做降精度、不使用 FMA 收窄。
特殊值（Inf/NaN）按 IEEE 754 传播：0×Inf=NaN 等自然出现，与 golden（cblas_srot 同为 float32 直接计算）一致。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 | x 物理长度 `1+(n−1)*|incx|`，y 同理 incy；调用方保证存储有效（C ABI 无法推断） |
| n = 0 | 合法 no-op：不访问任何指针、不启动 kernel、返回 SUCCESS |
| n < 0 | 非法：返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书 §2.4） |
| n = 1 | 单元素对，步长无实际意义 |
| 步长语义 | 正负步长均生效；负步长自物理 span 尾端起算；0 与 INT32_MIN 拒绝 |
| 广播 / 0 维 Tensor | 不涉及（BLAS-1） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（tilingKey 字段分支，平移 arch35 `SrotTilingData.tilingKey` 规划）：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_DENSE** | `incx==1 && incy==1` | 连续双缓冲流水：DataCopy x/y → 旋转计算 → DataCopy 写回 | 是（性能基线 3 条 base case 均连续） |
| **V_STRIDED** | 其余（正/负/混合步长） | 步长 DMA 读 x/y → 旋转计算 → 步长 DMA 写回 | 否（精度覆盖步长用例） |

数据流（V_DENSE，双缓冲）：

```
GM x ──DataCopy──▶ UB xBuf ─┐
                             ├─▶ tmpX = c·x + s·y;  y' = −s·x + c·y（UB 内，x 用旧值）
GM y ──DataCopy──▶ UB yBuf ─┘        │
                                     ▼
                        UB xBuf'/yBuf' ──DataCopy──▶ GM x / GM y
```

> 实现要点：先算 `newX = c·x + s·y` 存入临时缓冲（或先算 y'），保证第二式使用 x 旧值；
> 不得原地先写 x 再算 y。

数据流（V_STRIDED）：

```
GM x ──asc_copy_gm2ub_align(src_gap=(|incx|−1)*4)──▶ UB xTile ─┐
                                                                ├─▶ 旋转计算 ─▶ UB 写回
GM y ──asc_copy_ub2gm_align(dst_gap=(|incy|−1)*4)──▶ UB yTile ─┘
   （负步长：x/y 同号 → 逻辑域一致 0 次反转；异号 → UB 内反转其一）
```

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`，每核负责连续逻辑区间 `[myOffset, myOffset+myCount)`；
- 余数分配：前 `remainder` 个核各多 1 个元素；kernel 侧自算 `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4：tiling 不按核数组分配，平移 arch35 perCoreN/remainder 方案）。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_DENSE tileSize = 8192 float：xBuf/yBuf 双缓冲各 64 KiB + 临时缓冲 32 KiB ≤ 248 KiB；
- V_STRIDED tileSize = 1024 float，为 UB 内反转 scratch 留余量；
- 尾块 `DataCopyPad` + 对齐补零（32B 块 = 8 float）。

**tilingkey 规划策略**：tiling 字段 `tilingKey`：0 = 连续路径，1 = 步长路径（平移 arch35 `srot_tiling_data.h` L29-49 字段规划）；arch22 侧 TilingData 保留 tilingKey/totalN/cosValue/sinValue/cIsDevice/sIsDevice/perCoreN/remainder/tileSize/incx/incy 字段，删除 SIMT 专属 `nthreads`（死路字段，arch22 不用）。

**c/s 内存位置判定**：host 侧对 c、s 调 `aclrtPointerGetAttributes`：
- Host 内存 → 值直接写入 tiling `cosValue/sinValue`（`cIsDevice=0`），零额外 H2D；
- Device 内存 → 指针经 tiling 下发（`cIsDevice=1`），kernel 从 GM 读取；
- 判定失败按 Device 指针处理并在自测报告标注。

**数据检测**：按 V1..V6 顺序执行（见需求拆解节），顺序不可调换：
- V2（n<0 → INVALID_VALUE）在 V3（n=0 → SUCCESS）之前——任务书口径与 arch35 不同，注释中显式标注防止回归；
- V6 用 int64 计算跨度，双边守卫（>INT32_MAX 或 <INT32_MIN，负步长下 `(n−1)*inc` 为负）。

**其他 host 侧约定**：
- workspace：本算子不需要 → nullptr（任务书 §3.4 不涉及）；
- 同步：异步下发，host 不做 `aclrtSynchronizeStream`（任务书 §2.5）；tiling const 引用传参，无 H2D、无 aclrtMalloc；
- 日志：`OP_LOGE` 记录各校验失败分支，`OP_LOGD` 记录 tiling 决策与 c/s 位置判定结果。

#### kernel 侧设计

**Init/Process 两阶段**（V_DENSE）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(xBuf/yBuf/tmpBuf)
Process:
   ┌── for each tile ─────────────────────────────────────────┐
   │  MTE2: DataCopy xBuf←GM x, yBuf←GM y                      │
   │  MTE2→V: SetFlag/WaitFlag                                 │
   │  V:    tmp = c·x + s·y（Mul/Add ×2）                      │
   │        y'  = −s·x + c·y（x 为旧值）                        │
   │        x' = tmp                                           │
   │  V→MTE3: SetFlag/WaitFlag                                 │
   │  MTE3: DataCopy GM x←xBuf', GM y←yBuf'                    │
   │  MTE3→MTE2: SetFlag/WaitFlag                              │
   └───────────────────────────────────────────────────────────┘
   └── 尾块：DataCopyPad 补零对齐 → 计算 → 精确写回
```

**步长路径（V_STRIDED）符号统一**：与 SAXPY 设计同款推导——x/y 步长同号 → UB 内 0 次反转直接配对；
异号 → 反转其一。每 tile DMA 基址 `(inc>0) ? t0*|inc| : (n−1−t0−cnt+1)*|inc|`；
步长 DMA 参数：读 `src_gap=(|inc|−1)*4`，写 `dst_gap=(|inc|−1)*4`（范式来源
`blas/axpy/arch22/caxpy_kernel.cpp` src_gap/dst_gap 步长访问，已合入）。

**c/s 读取**：`cIsDevice/sIsDevice==0` 时从 tiling 标量取值；==1 时 tile 循环外一次性 `DataCopy` 读入 UB 标量缓冲（避免每 tile 重复 GM 读）。

**workspace 布局**：无 workspace（无跨核归约需求）。

**同步事件配对**：MTE2→V、V→MTE3、MTE3→MTE2 三段显式 SetFlag/WaitFlag；UB 内反转（步长路径）后加全流水栅栏。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 一次调用只发射一个 AIV kernel；核数 = min(AIV 核数, n) |
| 双缓冲流水 | xBuf/yBuf 各 2 槽，MTE2 与 V/MTE3 重叠 |
| host 零同步、零分配 | tiling const 引用传参；c/s 为 Host 内存时零额外 H2D（最常见路径） |
| 满 32B 对齐 DMA | tile 对齐 8 float，尾块 DataCopyPad 精确写回 |
| 计算融合 | 每元素对 4 Mul + 2 Add（或 2 Mul + 2 FMA 视指令时延实测选择），全部在 UB 内完成，无中间 GM 访问 |

**未采用的路线及原因**：
- **SIMT 步长线程循环**（arch35 tilingKey=1 方案）：arch22 无 SIMT——死路；
- **Gather 窗口方案**（`blas/spr/arch22/sspr_kernel.cpp` 范式）：可正确但多占 UB 窗口，仅作 4 字节突发 DMA 不可用时的降级备选；
- **arch35 校验逻辑整体平移**：n≤0 口径不一致（arch35 n<0 也成功），不满足任务书 §2.4。

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
| 参数合法性与校验层级 | handle nullptr → HANDLE_IS_NULLPTR；n<0 → INVALID_VALUE；n=0 → 成功 no-op；n>0 时 x/y/c/s nullptr、incx/incy 为 0/INT32_MIN、跨度溢出 → INVALID_VALUE；c/s 标量值不校验 |
| 非连续 Tensor 支持 | 支持：步长语义即非连续访问；正负步长均生效（负步长自尾端起算）；0/INT32_MIN 拒绝 |
| broadcast 规则 | 不涉及（BLAS-1） |
| dynamic shape 要求 | 不要求：n/incx/incy 运行时入参，host tiling 按运行时 n 计算 |
| 原地与视图语义 | x、y 原地更新（元素对先读后写，第二式 x 取旧值）；不返回视图 |
| 确定性计算要求 | 不要求；同输入同步长跨次 bit 级一致 |
| 空 Tensor 处理 | 等价 n=0：不访问数据、不启动 kernel、返回成功 |
| 0 维 Tensor 处理 | 不涉及 |
| 异步执行 | 经 handle 绑定 stream 异步下发；读取输出前须 aclrtSynchronizeStream；c/s 位置运行时自动判定 |

**已知限制（如实标注）**：
1. 步长路径 UB 内反转为标量 swap（同族 caxpy 范式），不参与性能判定；
2. C ABI 无法推断 x/y 真实分配长度，调用方保证 `1+(n−1)*|inc|` 元素有效；
3. arch35 与 arch22 的 n<0 行为不同（arch35 成功 / arch22 INVALID_VALUE），README per-arch 差异表须同步说明。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_srot`（float32 直接计算），经仓内 `srot_golden.h` 调用 | 任务书 §3.2/§7.4 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4 | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | 逐元素 `|actual − golden| ≤ atol + rtol*|golden|`；x 与 y 分别判定 | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；非有限值不计入 ratio/max_abs 统计，单独记录 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定） | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{srot_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`，共 1200 条（CSV 实测 1201 行含表头）；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/rot/srot/arch22/`（srot_test.cpp + CSV），并在 `test/rot/srot/CMakeLists.txt` 补齐 arch22 GTest 注册（任务书 §2.2）；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|inc|` 预分配（INT32_MIN/溢出乘积达数十亿元素），以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasSrot` 声明已存在于 `include/cann_ops_blas.h`，本次不改公共头；
- **对 arch35 影响**：无。源文件按 `SOC_ARCH_DIRS` 选目录编译，两套实现互不编译、互不覆盖；
- **对 arch22 同族 csrot 影响**：无。同目录并列新文件，接口独立（aclblasSrot vs aclblasCsrot），无共享可变状态；
- **测试构建影响**：`test/rot/srot/CMakeLists.txt` 确认 arch22 构建 GTest 注册生效（当前单行注册无分支，须在 A2/A3 构建下实测确认不退回非 GTest runner）；
- **文档影响**：`blas/rot/README.md` 产品支持表新增 aclblasSrot「A2/A3：支持」条目，并同步 per-arch 差异表（n<0 口径差异、步长拒绝口径）。
