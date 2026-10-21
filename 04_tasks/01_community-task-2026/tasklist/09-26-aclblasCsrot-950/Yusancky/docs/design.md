# aclblasCsrot 算子设计文档

> 社区任务：CANN训练营南科大-aclblasCsrot算子开发(950)
> 任务书：`aclblasCsrot_Atlas950PR_task_doc.md`
> 目标仓库：https://gitcode.com/cann/ops-blas

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 — 算子开发任务 `aclblasCsrot`（Ascend 950PR / CANN 训练营南方科技大学专场）。

- 任务书：`aclblasCsrot_Atlas950PR_task_doc.md`
- 上游仓库：https://gitcode.com/cann/ops-blas ，目标目录 `blas/rot/arch35/`
- 接口声明：`include/cann_ops_blas.h`（已存在，本任务不新增私有平行接口）

## 背景介绍

### aclblasCsrot 算子实现优化

ops-blas 仓的 `rot` 算子族提供 Givens 平面旋转（BLAS Level-1）。实数版本 `aclblasSrot` 已完成 arch35（Ascend 950PR）适配；复数版本 `aclblasCsrot` 此前仅有 arch22（Atlas A2/A3）实现。本任务基于 Ascend C 编程语言，为 arch35 实现完整、可合入的 `aclblasCsrot`，采用 Ascend C kernel 直调方式：基于 ops-blas 开源仓工程框架实现句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel。

### aclblasCsrot 算子实现现状分析

| 参数 | 参数含义 | 数据类型 | 当前支持（arch22） | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄 | scalar | 句柄式 API | 未校验空句柄 | — |
| n | 元素个数 | int | 任意整数 | 未按 Netlib 处理 n≤0 | — |
| x / y | 复数向量 | COMPLEX64 | Device，原地 | 未校验空指针 | 逻辑 [n] |
| incx / incy | 步长 | int | 仅隐式连续访存 | 未覆盖负/零步长与非 1 正步长 | — |
| c / s | 旋转因子 | FLOAT32 指针 | 仅 Host 指针直接解引用 | 不支持 Device 指针 | 标量 |

计算公式（y 的更新使用旋转前的原始 x[k]）：

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]
```

arch22 实现不能直接复用到 arch35，主要差距：

1. **入口校验**：arch22 仅判 c/s 空指针，未校验 handle、n≤0 no-op 语义、x/y 空指针、int32 偏移溢出。
2. **步长语义**：arch22 未覆盖 incx/incy 为正非 1、负、零及其混合组合，未实现 Netlib 的尾端起算与零步长串行累加语义。
3. **c/s 指针位置**：任务要求 Host/Device/mixed 三种位置自动判定，需 `aclrtPointerGetAttributes`。
4. **并行与流水**：arch22 为固定 8 核、手工 pipeline 风格；arch35 需对齐仓内 arch35 工程模式（平台 API 获取 AIV 核数、TQue 三级流水、SIMT grid-stride）。
5. **舍入语义**：需避免编译期 FMA 收缩导致相消场景（c≈s 且 x≈y）与参考实现的相对误差发散。**实测**：bisheng 设备端编译忽略 `-ffp-contract=off` 命令行参数与 `#pragma STDC FP_CONTRACT OFF`，须以 volatile 物化强制分步舍入（详见 3.2.2）。

本实现在 `blas/rot/arch35/` 新增 `csrot_{host,kernel,kernel.h,tiling_data}` 四个文件，沿用 `srot` 的双路径架构，将 complex64 视为交错 float lane（每元素 2 个 float）处理。

### aclblasCsrot 算子功能分析

对两个等长复数向量 `x`、`y` 做原地 Givens 旋转。`c`、`s` 为**实数**标量（与 `crot` 区分：`crot` 的 `s` 为复数）。实数标量乘以复数的语义为实部、虚部分别乘以该实数，因此旋转在实部/虚部两个 lane 上独立同形，无复数乘法交叉项。典型用途为 QR 分解等数值算法中的平面旋转。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCsrot`：

- 数据类型：complex64 向量（`aclblasComplex`，实部/虚部各 float32）+ float32 实数旋转因子 c/s
- 硬件：Ascend 950PR / Ascend 950DT（arch35），CANN 9.1.0
- 接口签名与 `include/cann_ops_blas.h` 既有声明一致，参数序列与 `cublasCsrot` 一一对应
- 边界与舍入语义对齐 Netlib `csrot.f`
- 精度、性能满足任务书 §3.2、§3.3

## 需求拆解

1. **双路径**：`incx==1 && incy==1` 走连续路径（SIMD membase）；其余 stride 组合（含正非 1、负、零及其混合）走 stride 路径（SIMT grid-stride）。
2. **原地安全**：先读齐原始 x/y 再写回；y 的更新使用原始 x[k]。
3. **边界对齐 Netlib**：`n≤0` 为合法 no-op 直接返回成功；零步长按参考语义对同一元素反复旋转（串行累加）；负步长从向量尾端起算沿负方向步进；`c==1 && s==0` 单位旋转不短路。
4. **c/s 位置自动判定**：host / device / mixed 三种指针位置，逐指针独立判定。
5. **分步舍入**：连续路径用 `Muls + Muls + Add` 向量指令（天然无 FMA）；stride 路径乘积经 **volatile 临时变量物化**。实测依据：bisheng（Clang 15）设备端编译忽略命令行 `-ffp-contract=off` 与 `#pragma STDC FP_CONTRACT OFF`，`c*x + s*y` 仍被融合为 FMA——首元素（输入完全相同）即与 netlib 分开舍入差 1-2 ULP，零步长串行累加场景下漂移复合、直至改变 Inf 出现步点。volatile 物化是 C++ 标准保证的防融合手段（volatile 访问为可观察副作用），修复后 NPU 与 netlib golden 在 26 万步串行累加上逐位一致。
6. **负向校验**：handle 空指针、n>0 时 x/y/c/s 空指针、`(n-1)*stride` 超出 int32 表示范围，分别返回对应状态码。
7. 精度 / 性能达到任务书 §3.2 / §3.3。

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]        // x[k] 为旋转前原始值

k = (i-1)*incx,  j = (i-1)*incy,  i = 1..n   （1-based 索引，兼容 Fortran）
```

实部 / 虚部独立旋转（实数标量，无交叉项）：

```
xr' = c*xr + s*yr;  xi' = c*xi + s*yi
yr' = c*yr - s*xr;  yi' = c*yi - s*xi
```

据此关键结论：`aclblasComplex { float real; float imag; }` 连续存放时，**n 个复数 ≡ 2n 个 float lane**（偶 lane 实部、奇 lane 虚部），实数标量的 `Muls` 可同时作用于实部与虚部两个 lane，kernel 以 float 视图操作，无需额外 layout 转换。

### 支持数据类型

| 参数 | 类型 |
|------|------|
| x, y | complex64（`aclblasComplex`，8 字节） |
| c, s | float32（Host 或 Device 指针） |
| 计算精度 | float32 |

### 支持形状

逻辑一维 `[n]`，物理跨度 `1+(n-1)*|inc|`。n 为任意整数（n≤0 为合法 no-op）；incx/incy 为任意整数（含 0 与负值，均不拦截）。不涉及广播；不要求非连续 Tensor（非连续访问由 incx/incy 表达）；不要求 dynamic shape（n 为运行时入参）。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

**内存布局与 kernel 视图**：complex64 数组即交错 float 数组（real/imag 相邻）。连续路径按 float lane 搬运与计算（lane 数 = 复数个数 × 2）；stride 路径按复数元素寻址。kernel 直接以 float 视图工作，Host 侧不做任何布局转换。

**TilingData（`csrot_tiling_data.h`，计数以 complex 元素为单位）**：

| 字段 | 含义 |
|------|------|
| `tilingKey` | 0=连续路径（SIMD），1=stride 路径（SIMT） |
| `totalN` | 参与旋转的复数元素总数 |
| `cosValue` / `sinValue` | c/s 为 host 指针时的标量值（device 指针时为 0.0f 占位） |
| `cIsDevice` / `sIsDevice` | c/s 指针位置标志（1=device，0=host），两者相互独立 |
| `perCoreN` / `remainder` | 连续路径分核：整除数与余数（余数分给前 remainder 个核） |
| `tileSize` | 连续路径 UB 单次搬运上限（complex 个数） |
| `incx` / `incy` | stride 路径步长（int32，承载负值） |
| `nthreads` | stride 路径每 block 线程数 |

##### 1. 分核策略

优先使用满核、核间均分的原则。

**连续路径（tilingKey=0）**：

- `numBlocks = min(n, aivCoreNum)`，`aivCoreNum` 由平台 API（`GetAivCoreCount`）获取；为 0 时返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。
- `perCoreN = n / numBlocks`，`remainder = n % numBlocks`，余数元素分给前 `remainder` 个核（每核多 1 个）。不做 block 对齐向下取整，保证 `n < aivCoreNum` 时负载依然均衡、无空转核。
- 均分后 per-core 起始偏移与元素数均非 block 对齐；连续路径 kernel 使用 `DataCopyPad`，GM 侧仅需 1 字节对齐（复数天然 8 字节对齐），非对齐偏移与尾块均安全。

**stride 路径（tilingKey=1）**：

- `numBlocks = min(ceilDiv(n, SIMT_MIN_THREAD_NUM), aivCoreNum)`，至少为 1。
- `nthreads = min(CeilAlign(CeilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`：按每核元素数确定线程数，向上对齐到硬件线程调度粒度（128）并封顶 2048。小 n（如 n=10、单 block）只起 128 线程而非 2048，消除空转线程的启动开销；大 n 仍可打满单 block 2048 线程。kernel 内以 grid-stride（跨度 = block_num × blockDim.x）遍历全部 n 个元素。

##### 2. 数据分块和内存优化策略

充分使用 UB 空间的原则，同时为 kernel 侧 API 的临时数据预留容量。

**连续路径 UB 预算**：TQue 三级流水（CopyIn → Compute → CopyOut），共 **5 个 UB tile**：

- `inQueueX` / `inQueueY`：VECIN，MTE2 输入（保存原始 x/y）
- `outNewX` / `outNewY`：VECOUT，Vector 输出（保存旋转结果）
- `workBuf`：VECCALC，纯 Vector 临时区（暂存 `round(s*y)` / `round(-s*x)`，无 MTE 流量）

`tileSize` 按 `UB_SIZE / (5 × sizeof(complex64))` 计算并向下对齐到 32 字节块单元（4 个 complex），保证 UB 分配与搬运均满足 32B 边界。arch35 UB_SIZE 为 248KB，据此 `tileSize ≈ 6348` 个 complex（每 tile 约 50KB）。

**原地安全（in-place）设计**：Compute 阶段原始 x/y 始终驻留 VECIN 队列，两个新值写入独立 VECOUT 队列，CopyOut 才写回原 GM 地址；任何暂存区都不会在其消费者完成前被覆写，因此 x、y 原地更新且允许 x/y 指向同一缓冲区（别名）时语义仍与 Netlib 一致。

**零步长串行语义**：Netlib 在 `inc==0` 时索引常量保持，N 次迭代复用同一地址、结果串行累加。多线程并发会读同一原始值并对同一地址竞争写，破坏语义。因此 stride 路径 kernel 在 `incx==0 || incy==0` 时，仅 `block0.thread0` 以严格参考顺序串行执行 n 次旋转，其余线程/核立即返回，保证该地址只被一条程序顺序流访问。

**无 workspace**：本算子为纯原地逐元素运算，不申请任何 device workspace，Host 侧无 `aclrtMalloc`。

##### 3. tilingkey 规划策略

| tilingKey | 条件 | kernel |
|---|---|---|
| 0 | `incx == 1 && incy == 1` | SIMD 连续路径（TQue 流水） |
| 1 | 其余全部 stride 组合 | SIMT stride 路径（grid-stride） |

c/s 指针位置**不进入** tilingKey，由 `cIsDevice`/`sIsDevice` 在 kernel 内分支选择取值来源，避免 tilingKey 组合膨胀。

**c/s 来源解析（host/device/mixed）**：host 侧通过 `aclrtPointerGetAttributes` **逐指针独立**判定 c、s 的内存位置（与 rotg 的全有或全无不同，srot 已验证独立判定模式）：

- host 指针：解引用一次，标量填入 `cosValue`/`sinValue`，kernel 侧标量广播；下传指针置空。
- device 指针：指针原样作为 GM_ADDR 下传，标量字段填 0.0f 占位，`cIsDevice/sIsDevice` 置 1；kernel 从 GM 读数。device 路径不解引用、不做 D2H 拷贝、不同步 stream。

**数据检测（入口校验顺序，与 srot 既有行为对齐）**：

| 序号 | 检测项 | 不通过时返回 |
|---|---|---|
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（优先于 n≤0 快速返回） |
| 2 | `n <= 0` | `ACLBLAS_STATUS_SUCCESS`（合法 no-op，不再校验其余参数，不修改 x/y） |
| 3 | `n > 0` 且 `x == nullptr` 或 `y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `c == nullptr` 或 `s == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `aclrtPointerGetAttributes` 查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | stride 路径下 `(n-1)*incx` 或 `(n-1)*incy` 超出 int32 表示范围 | `ACLBLAS_STATUS_INVALID_VALUE` |

第 6 项在 host 侧以 int64 计算乘积后再收窄检查，避免 kernel 侧 int32 地址偏移溢出；连续路径偏移恒 `< n ≤ INT32_MAX`，无需该项检查。incx/incy 取任意值（含 0、负值）均不拦截。

**下发**：校验与 tiling 计算完成后，在 `handle->stream` 上异步启动 kernel，不阻塞、不同步；读回 Device 结果前由调用方同步 stream（依赖 `aclblasSetStream` 绑定的 stream）。

#### 3.2.2 kernel 侧设计

kernel 入口分两个独立二进制（SIMD 的 TPipe/DataCopy 与 SIMT 的 `asc_vf_call` 代码互不混杂），由 host 侧 `csrot_kernel_do` 按 `tilingKey` 选择启动。

**连续路径（`csrot_aiv_kernel`，tilingKey=0）**：Init + Process，Process 内按 tile 循环 CopyIn / Compute / CopyOut 三阶段。

- CopyIn：`DataCopyPad` 将 x/y 的 `dataCount*2` 个 float lane 搬入 VECIN；EnQue 作为 MTE2→Vector 同步点。
- Compute（Vector 阶段，laneCount = dataCount × 2，实数标量对实/虚 lane 同时生效）：

```
newX = round(c*x);  newX = newX + round(s*y)     // Muls + Muls + Add
newY = round(c*y);  newY = newY + round(-s*x)    // Muls + Muls + Add
```

  两个乘积分别舍入后再相加（分步舍入），不用 `Axpy` 的 FMA 一步累加：当 c==s（45° 旋转）且 x≈y 时结果向 0 相消，FMA 的未舍入乘积腿会偏离参考实现，放大相对误差。原始 x/y 在整个 Compute 阶段保持驻留，两个新值读齐后才释放，保证原地安全。
- CopyOut：`DataCopyPad` 将结果写回原 GM 地址（UB→GM 自动剥掉 UB 内补齐尾块）。

**stride 路径（`csrot_simt_kernel`，tilingKey=1）**：grid-stride 循环，线程全局索引 `tid = blockIdx.x*blockDim.x + threadIdx.x`，步长 `block_num*blockDim.x`，每个线程处理互不相交的元素。

- 索引按 Netlib 语义计算：负步长时逻辑第 i 个元素映射到物理下标 `n-1-i`（尾端起算）；`|inc|` 经 int64 取负再收窄，规避 `INT32_MIN` 取负 UB；inc==0 时该侧下标强制为 0。
- 零步长（任一侧）时仅 `block0.thread0` 按参考顺序串行执行 n 次旋转（串行累加），其余线程/核返回。
- 每线程将 x[xIdx]、y[yIdx] 两个复数（各 2 个 float）读入寄存器，按四个分量分别计算新值后写回；寄存器天然保证“y 用原始 x”的原地语义。
- **分步舍入（volatile 物化）**：每个乘积先写入 volatile 临时变量再参与加减，强制单次舍入。背景：bisheng 设备端忽略 `-ffp-contract=off` 与 `#pragma STDC FP_CONTRACT OFF`（均有实测），FMA 融合会在相消场景（c≈s 且 x≈y）与零步长串行累加下偏离 netlib 参考实现的舍入序列；CMake 中 `csrot_kernel.cpp` 的 `-ffp-contract=off` 保留（对 host 侧编译仍有效，设备端由 volatile 保证）。修复后经 NPU 探针验证：n=262144、incy=0 的 26 万步串行累加，NPU 与 netlib golden 逐位一致（xDiff=0、yDiff=0）。

**kernel 源文件清单**：

```
blas/rot/arch35/csrot_host.cpp        // host API：校验 + tiling + 异步下发
blas/rot/arch35/csrot_kernel.cpp      // AIV 连续 kernel + SIMT stride kernel
blas/rot/arch35/csrot_kernel.h        // csrot_kernel_do 启动器声明
blas/rot/arch35/csrot_tiling_data.h   // CsrotTilingData 定义
```

arch22 既有实现（`blas/rot/arch22/csrot_*`）不改动；构建时按目标 SOC 选择 arch 目录（ascend950→arch35，ascend910b*/910_93*→arch22），两套实现按 SOC 分派、无符号冲突。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 既有 arch22 实现，本任务不改动 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 既有 arch22 实现，本任务不改动 |

合入后同步更新 `blas/rot/README.md`：`aclblasCsrot` 产品支持表将 Ascend 950PR / Ascend 950DT 标注为**支持**，并将函数原型修正为与 `include/cann_ops_blas.h` 一致的声明（`int n`、`const float* c/s`），禁止定义 950PR 私有平行接口。

## 算子约束限制

- 仅支持 complex64（`aclblasComplex`，实部/虚部各 float32）与 float32 实数旋转因子；复数 `s` 属 `crot`，不在本算子范围。
- x、y 位于 Device 内存，原地更新；y 的更新使用旋转前原始 x[k]；不返回视图。
- incx/incy 可为正/负/零均不拦截；stride 路径要求 `(n-1)*|inc|` 在 int32 表示范围内。
- n≤0（含 n=0 与负 n）为合法 no-op，返回成功且不修改 x、y；`c==1 && s==0` 单位旋转不短路。
- c、s 可为 Host 或 Device 指针（含 mixed），不可为 nullptr；算子运行时通过 `aclrtPointerGetAttributes` 自动判定。
- 不涉及广播、非连续 Tensor 额外 padding、dynamic shape、确定性计算与 workspace 内存要求。
- 异步执行依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol=2^-13（1.2e-4），atol=2^-13（1.2e-4），required_matched_ratio≥0.99，max_abs_error_limit=1e-2 或 32×ULP；逐元素通过条件 `|actual-golden| ≤ atol + rtol×|golden|` | 任务书 §3.2；生态算子开源精度标准 |
| 辅判据 | 测试工程 MERE/MARE 口径：MERE≤2^-13，MARE≤10×2^-13（复数按实部/虚部分别统计） | ops-blas ST 测试框架 |
| 性能标准 | Ascend 950PR，COMPLEX64，warmup 5 次后有效采样 >50 次取平均：n=1048576（inc=1）≤21.69us；n=2097152（inc=1）≤43.19us；n=4194304（inc=1）≤119.38us | 任务书 §3.3 |
| 内存标准 | 不涉及（无 workspace，纯原地运算） | 任务书 §3.4 |

golden 为 netlib `csrot.f` 公式的显式实现（实部/虚部逐通道独立计算、乘积 volatile 物化保证每步单次舍入；本机 libblas 无 `cblas_csrot` 符号且 gfortran 版 `csrot_` 存在 `0*Inf=NaN` 伪交叉项缺陷，详见下方说明），实部、虚部分别比对。本算子为逐元素线性运算，误差来源仅为每元素两次实数乘加的浮点舍入，正常实现应远优于上述阈值。

**自测方案**：使用 ops-blas 仓 test 目录 CSV 驱动 GTest 框架（参照 `test/rot/srot` 同族工程模式）完成自验，测试工程落在 `test/rot/csrot/arch35/`：

```
test/rot/csrot/csrot_param.h          // CSV 参数结构（n/x/incx/y/incy/c/s/cs_ptr_mode/expect_result/mere/mare）
test/rot/csrot/csrot_golden.h         // CPU golden：显式实现 netlib csrot.f 公式（见下方说明），校验逻辑与 Netlib 对齐
test/rot/csrot/arch35/csrot_npu_wrapper.h  // NPU 下发 + cs_ptr_mode（host/device/mixed）落地 + cIsNull/sIsNull 空指针负向透传
test/rot/csrot/arch35/csrot_test.cpp  // GTest：空句柄、快速返回、CSV 驱动用例
test/rot/csrot/arch35/csrot_benchmark.cpp  // kernel 级性能 benchmark（设备事件计时 + 典型 shape 硬门槛断言）
test/rot/csrot/arch35/csrot_test.csv  // 1200 条用例（1000 精度 + 200 性能）
test/rot/csrot/CMakeLists.txt         // 扩展 arch35 目标
test/rot/csrot/README.md              // 自测说明（环境/编译/精度/性能/内存，对齐 cswap 交付结构）
```

用例分类（随任务提供的 `gen_csv.py` 生成，固定随机种子、条数可扩展）：

| 类别 | 前缀 | 覆盖点 |
|------|------|--------|
| L0 基础 | TC_L0 | 小尺寸 × 基础步长（±1/±2） |
| L1 尺寸 | TC_SQ | 尺寸扫描：0/1/小质数/2 的幂及 ±1/非对齐值，直至 2^20 |
| L2 步长 | TC_INC | incx×incy 全组合（±1/±2/±3/0，含零步长） |
| L3 旋转因子 | TC_CS | (c,s) 特殊值：单位旋转/纯交换/零/负/大值 × 2 尺寸 |
| L5 填充 | TC_FL | x/y 分别覆盖均匀/全零/交替/极端/Inf/NaN |
| L5b 指针位置 | TC_PM | host/device/mixed × 连续/步长/负步长/单位旋转 |
| L6 边界负向 | TC_ED | n=0/负 n no-op（SUCCESS）、零步长（SUCCESS）、空指针 x/y/c/s（INVALID_VALUE）、空句柄 |
| EX 扩展 | TC_EX | 尺寸×步长×旋转因子×指针位置确定性采样 |
| PF 性能 | TC_PF | 3 条任务书 §3.3 典型 case + 规模扫描 + c/s device 指针 + 混合（均连续访存） |

精度判定走测试工程 MERE/MARE 口径（CSV 列驱动，实部/虚部分开统计，MERE<2^-13、MARE<10×2^-13；任务书混合容差口径 matched_ratio≥0.99 同样满足）；NaN 输入用例按 NaN 传播直接判定，Inf+单位旋转用例按逐位相等判定。性能用例先 warmup 10 次（任务书要求 ≥5）再有效采样 100 次（要求 >50）取平均；kernel 级耗时应采集 `OpBasicInfo.csv` 的 `Task Duration(us)` 求平均（GTest 端到端耗时含 host 准备与 wrapper 拷贝开销，不能反映 kernel 本身效率）。自验通过 `verify_accuracy.py` / `verify_performance.py` 驱动，输出自测报告。

**golden 实现说明（验收机环境适配）**：验收机（Ascend 950PR，CANN 9.1.0）的 libblas 为 netlib 参考实现（BLAS 3.10.0），其 cblas.h 未声明 `cblas_csrot`、libblas 亦不导出该符号；直接调用 Fortran `csrot_` 入口同样不等价——gfortran 将复数乘编译为 `(a*c - b*d, a*d + b*c)` 且 b≡0，某通道到达 ±Inf 后伪交叉项 `0*Inf = NaN` 会污染另一通道（实测：x.re=-Inf 会迫使 x.im=NaN），属编译器代码生成缺陷而非 netlib 语义（cublasCsrot / OpenBLAS 的 csrot 均逐通道独立计算）。因此 golden 显式实现 netlib `csrot.f` 公式：实部/虚部逐通道独立，乘积 volatile 物化保证每步单次舍入；负步长尾端起算、零步长串行累加、`c==1 && s==0` 不短路等 Netlib 语义不变。经探针对比，该 golden 与 netlib `csrot_` 在非溢出场景逐位一致，且与 NPU kernel 在全用例集（含 Inf/NaN 溢出场景）逐位一致。

**自验结果（2026-10-20，验收机实测）**：

| 验证项 | 结果 |
| --- | --- |
| 精度（`verify_accuracy.py`，1000 条 = 1200 − 200 TC_PF） | **PASS=1000 / FAIL=0，ALL PASS** |
| 性能 kernel 级（`csrot_benchmark`：设备事件计时、设备缓冲按用例一次性分配、warmup 20 + 100 次采样、计时区间只覆盖 kernel 下发，与 GPU 纯 kernel 标杆同口径） | n=1048576：**11.07us**（标杆 ≤21.69us，倍率 0.78）；n=2097152：**18.27us**（≤43.19us，倍率 0.95）；n=4194304：**38.53us**（≤119.38us，倍率 1.24）——**均满足 ≥0.4，benchmark 硬门槛断言 PASSED** |
| 负向用例 | handle 空指针、n≤0 快速返回、null c/s、null x/y 返回码均与预期核对表一致 |
| msprof | 验收机 FFTS profiling channel 启动失败（容器/驱动限制，两种调用语法均失败），kernel 级耗时以上述 ACL timeline 探针实测替代 |

> 备注：任务包 `verify_performance.py` 解析的是 GTest 整条用例墙时间（含 host 数据生成与 110 次 wrapper 调用的 aclrtMalloc/H2D/D2H），与其文档声称的"平均单次耗时"指标不符——即使 harness 零开销，110 次 × kernel 单次耗时（n=2²⁰ 时约 11us → 1.2ms）仍远超标杆/0.4 = 21.69us，其 FAIL 结论不反映算子性能；性能验收以 kernel 级数据为准。

## 兼容性分析

- 接口与 `cublasCsrot` 参数序列一一对应（handle 及参数顺序一致），无需额外映射说明。
- 接口声明沿用 `include/cann_ops_blas.h` 既有声明，不新增 950PR 私有平行接口，可与其他产品线共用。
- arch22 既有实现保留，构建时按目标 SOC 的 arch 目录分派，不影响既有 Atlas A2/A3 产品线行为。
- 合入后更新 `blas/rot/README.md` 产品支持表（Ascend 950PR / 950DT 标注支持）并修正接口原型与头文件一致。
