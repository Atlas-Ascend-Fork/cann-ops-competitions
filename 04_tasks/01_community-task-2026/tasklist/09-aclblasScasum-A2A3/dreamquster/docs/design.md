# aclblasScasum 算子设计文档（Atlas A2/A3，arch22）

> - 任务书：`aclblasScasum_AtlasA2_A3_task_doc.md`（11 月社区任务）
> - 对标接口：cuBLAS `cublasScasum` / Netlib `scasum`
> - 代码仓：https://gitcode.com/dreamquster/ops-blas （分支待验收代码，合入目标 `https://gitcode.com/cann/ops-blas` 目录 `blas/asum/arch22/`）
> - 设计文档提交仓：https://gitcode.com/dreamquster/cann-ops-competitions （PR 合入 `cann/cann-competitions` 任务目录）
> - 适配硬件：Atlas 800T A2（910B3，dav-2201，性能测试设备）/ Atlas 800T A3，CANN 9.1.0
> - 数据类型：COMPLEX64（单精度复数，`aclblasComplex { float real, imag; }`，交错 AoS 布局，8 字节/元素）→ FLOAT32 归约标量

---

# 需求背景（required）

## 需求来源

11 月社区任务：在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C 编程语言开发单精度复数向量绝对值分量之和算子 `aclblasScasum`，完成算子设计、开发、测试全流程工作。验收通过后合入昇腾算子开源仓 `https://gitcode.com/cann/ops-blas` 的 `blas/asum/arch22/` 目录。

## 背景介绍

### 算子功能

计算单精度复数向量 `x` 各元素实部与虚部绝对值之和，数学表达式（0-based 展开写法对齐 Netlib `scasum.f`）：

```
result = Σ_{i=0..n-1} ( |Re(x[1 + i*incx])| + |Im(x[1 + i*incx])| )
```

输入为交错存储的 COMPLEX64 向量，输出为单个 FLOAT32 实数标量。当 `n > 0 && incx > 0` 时，物理访问跨度为 `1 + (n-1)*incx` 个复数元素。

### 对标分析（cublasScasum / Netlib scasum）

参数序列与 cuBLAS `cublasScasum(handle, n, x, incx, result)` 完全一致（含 handle 位置），与 Netlib BLAS `scasum.f` 语义对齐：

| 语义项 | cuBLAS / Netlib | 本算子 |
| --- | --- | --- |
| `n <= 0` | 不触发 kernel，直接写 0.0 | 对齐（quick return） |
| `incx <= 0` | 不触发 kernel，直接写 0.0 | 对齐（quick return，本任务书 §2.1-3 明确 incx ≤ 0 不反向遍历） |
| 累加顺序 | 不规定 | 不规定；按精度阈值判定 |
| `|Re|+|Im|` 结果溢出 | 传播为 +Inf | 对齐（不做饱和） |
| Inf/NaN 传播 | IEEE 浮点语义 | 对齐 |

### 仓内现状分析

- **arch35 已有实现**：`blas/asum/arch35/scasum_*`（Atlas 950PR），host 侧完整覆盖 quick return / int64 溢出校验 / partial-sum workspace / final reduction kernel；kernel 侧两级归约（Vector 内归约 → Block partial sum → Final kernel 归约）。可直接借鉴**校验顺序 / tiling 结构 / 归约框架**，但其 UB 预算（256KB）、核数（arch35）、Vector 指令与 arch22 不同，**不可直接复用 kernel 代码**。
- **arch22 实数参考**：`blas/asum/arch22/sasum_host.cpp` + `sasum_kernel.cpp` 已实现单精度实数 asum（`aclblasSasum`），是本算子的最直接参考——**复数路径可视为实数路径的扩展：每元素从 1 个 float 读变 2 个 float（交错 real/imag），计算从 `|x[i]|` 变 `|Re|+|Im|`**。
- **arch22 平台参数**（来源 `blas/common/arch/hardware.h` 及仓内 CtrsmBatched 设计约定）：UB = 192KB / 核，L1 = 512KB，L2 = 192MB；Vector 核数运行时通过 `GetAivCoreCount()` 查询（910B3 为 40）。
- **仓内 `include/cann_ops_blas.h:236` 已预留声明**（任务书 §2.1-6 禁止定义私有平行 API）。
- **GTest 框架 gap**：`test/asum/scasum/CMakeLists.txt` 当前仅在 arch35 分支注册 GTest，arch22 构建会退回非 GTest runner（输出整数毫秒）。**本任务必须补齐 arch22 GTest 注册**，与 sasum 在 arch22 的 GTest 接入方式对齐。

# 需求分析（required）

## 需求描述

基于 ops-blas 开源仓工程框架，在 `blas/asum/arch22/` 下以 kernel 直调（Ascend C）方式实现 `aclblasScasum` 句柄式 BLAS 接口：完成参数校验（合法/边界/负向）、复数向量 `|Re|+|Im|` 归约 kernel（COMPLEX64 → FLOAT32）、quick return、int64 溢出校验、partial-sum 并行归约、`aclblasSetStream` 异步语义，满足 FLOAT32 混合容差精度标准与 GPU 标杆 0.8× 性能阈值，并通过仓内 CSV 驱动 GTest 框架完成自验（含自补正态分布用例）。

## 需求拆解

1. **功能 F1**：公共 API `aclblasScasum` 声明已预留，禁止平行接口。
2. **功能 F2**：复数路径扩展自 arch22 `sasum`（实数）——连续读 float2 对 → `abs(real)+abs(imag)` 累加；incx>0 跨步访问、incx≤0/n≤0 quick return 写 0.0。
3. **校验 V1**：handle 空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；result 空 → `ACLBLAS_STATUS_INVALID_VALUE`（**含 quick return 路径**，任务书 §2.5 要求）；x 空（仅正常计算路径）→ `ACLBLAS_STATUS_INVALID_VALUE`；`(n-1)*|incx|` 超出 int32 域 → `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #805）。
4. **校验 V2**：校验顺序对齐仓内 sasum / arch35 scasum 惯例——`handle` → `result` → quick return 判定 → `x`（仅正常路径）→ 溢出校验 → 计算。
5. **性能 P1**：`gpu_baseline.csv` 共 201 条（含 id=000/001/002 base case + 200 条 perf case），**全部 incx=1，n 分布 1..4,194,304**；910B3 上平均单次耗时 ≤ GPU ms / 0.8（即达到 0.8× GPU 性能）。判据：warmup 5 次 + 有效采样 ≥10 次取平均（任务书 §3.3）。
6. **精度 A1**：标量比对（非逐元素）；FLOAT32 混合容差 rtol=atol=2⁻¹³(1.22e-4)，matched_ratio ≥ 0.99（标量单值判定），max_abs_error ≤ 1e-2 或 32×ULP；golden 由 cblas Netlib `scasum` 生成。
7. **精度 A2**：Inf/NaN 特殊值规则（任务书 §3.2）——golden 与 actual 同为 NaN、同号 Inf 或一方为 NaN 另一方为 ±Inf，视为等效；一方有限值另一方 Inf/NaN 判定失败。
8. **测试 T1**：CSV 驱动 GTest；补齐 arch22 GTest 注册；自补 `GAUSS_NORM_5_2` 正态分布用例（随任务 gen_csv.py 默认仅生成均匀分布）。
9. **测试 T2**：负向用例缓冲区规则（任务书 §3.5）——期望错误码的用例 **不得按 `(n-1)*|incx|` 分配 x**，须以单元素 dummy/NULLPTR 占位，防 host 分配失败误判。
10. **工程 E1**：kernel 直调（`<<<numBlocks, nullptr, stream>>>`），tiling 结构体值传递，零 workspace 申请（partial sums 用 GM 临时数组，不通过 handle workspace 接口）。
11. **工程 E2**：`blas/CMakeLists.txt` 按 `blas/*/${SOC_ARCH_DIRS}/*.cpp` 自动 glob 收录，**CMake 零改动**。

# 详细设计（required）

## 算子分析

### 数学公式

对第 i 个逻辑复数元素（1-based 下标展开，0-based 转换后等价）：

```
result = Σ_{i=0..n-1} ( | Re( x[ 2 * (1 + i*incx) ] ) |
                       + | Im( x[ 2 * (1 + i*incx) ] ) | )
```

当 incx=1 时简化为连续访问的 float2 对序列；incx>1 时按 stride 跨步；incx≤0 或 n≤0 时不进入本公式，由 quick return 写 0.0。

### 支持数据类型

输入 COMPLEX64（`aclblasComplex { float real; float imag; }` AoS）；中间累加 FLOAT32；输出 FLOAT32 标量。不要求确定性（归约顺序不保证逐位一致）。

### 支持形状

- n：任意 int；n≤0 quick return；大规模 ≥2²⁰ 验收至 4M+
- incx：int；incx>0 正常计算；incx≤0 quick return（**不反向遍历**，任务书 §2.1-3 与 Netlib/cuBLAS 语义一致）
- 物理跨度：`1 + (n-1)*|incx|` 个复数元素（host 侧 int64 计算并做 int32 溢出校验）
- 值域：FLOAT32 全集，含 Inf/NaN/极端值；允许累加溢出到 +Inf

## 算子实现

### 实现方案

**工程组织**（与 arch22 `sasum` 平铺命名风格一致；`blas/CMakeLists.txt` glob 自动收录）：

```
ops-blas/
├── include/cann_ops_blas.h                     # 第 236 行已预留 aclblasScasum 声明
├── blas/asum/arch22/
│   ├── scasum_host.cpp                         # 接口入口、校验、quick return、tiling、kernel 启动
│   ├── scasum_tiling_data.h                    # tiling 结构体定义（值传递）
│   ├── scasum_kernel.h                         # kernel 启动包装函数声明（*_kernel_do）
│   └── scasum_kernel.cpp                       # Ascend C kernel（主 kernel + final reduction kernel）
└── test/asum/scasum/
    ├── CMakeLists.txt                          # + arch22 分支 ops_blas_add_gtest_tests 注册
    ├── scasum_param.h                          # CSV → 参数结构体 + 复数输入生成（扩展自 sasum）
    ├── scasum_golden.h                         # cblas scasum golden（仅 quick return 写 0，正常路径调 cblas）
    └── arch22/
        ├── scasum_test.cpp                     # GTest 入口 + CSV 加载 + 负向硬编码用例
        ├── scasum_npu_wrapper.h                # 设备内存管理 + 调用被测接口
        └── scasum_test.csv                     # 用例（随任务提供 + 自补 GAUSS_NORM_5_2）
```

**接口声明**（已在 `include/cann_ops_blas.h:236` 提供，禁止平行接口）：

```cpp
aclblasStatus_t aclblasScasum(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

**调用链总览**：

```
aclblasScasum
  └─ Host 校验（顺序固定，见下表）
  └─ Quick return：(n<=0 || incx<=0) → Device 异步写 result=0.0f
  └─ (n-1)*|incx| int64 溢出校验
  └─ 计算 ScasumTilingData（blockCount、tileElements、workspace 布局）
  └─ <<<blockCount, nullptr, h->stream>>> 发起 scasum_kernel（每 block 写 1 个 partial sum）
  └─ 若 blockCount > 1：
       <<<1, nullptr, h->stream>>> 发起 scasum_final_kernel（读 blockCount partial sum，写 result）
```

#### host 侧设计

**1. 参数校验**（`CHECK_RET` 宏，顺序与仓内 arch35 `scasum_host.cpp` / arch22 `sasum_host.cpp` 对齐）：

| 步骤 | 校验项 | 失败返回 | 备注 |
| --- | --- | --- | --- |
| 1 | handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 第一优先 |
| 2 | result == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` | **quick return 路径仍需写零，必须先判** |
| 3 | n <= 0 \|\| incx <= 0 | Device 写 result=0.0f，返回 `ACLBLAS_STATUS_SUCCESS` | quick return，不访问 x |
| 4 | x == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` | 仅在正常路径下执行 |
| 5 | `(n-1) * (int64_t)|incx|` 溢出 int32 域 | `ACLBLAS_STATUS_INVALID_VALUE` | issue #805；int64 中间变量，校验物理跨度是否 ≥ 0 且 ≤ INT32_MAX |

**2. Quick return 实现**：不发起 kernel；通过仓内已有 Device scalar 写零封装（或复用 arch22 sasum 的 zero-fill kernel 退化路径，launch 0 元素写 1 个标量），**在 `h->stream` 上异步写 `*result = 0.0f`**，保证与后续同 stream 操作顺序一致。

**3. Tiling 计算**（全部 host 侧完成，`ScasumTilingData` 值传递，无 tilingkey、无 GM 序列化——kernel 直调形态）：

```cpp
struct ScasumTilingData {
    uint32_t n;               // 逻辑复数元素数（已过 n>0 校验）
    uint32_t incx;            // 复数元素步长（已过 incx>0 校验）
    uint32_t blockCount;      // 参与计算的 block 数
    uint32_t tileElements;    // 每 block 负责的逻辑元素数（除尾块）
    uint32_t workspaceOffset; // partial sums 在 GM workspace 中的起始偏移（0 表示直接写 result）
    float*   resultPtr;       // 输出标量指针
    float*   partialPtr;      // partial sums 工作区指针（单 block 时为 nullptr）
    uint32_t flags;           // 标志位：FLAGS_INCX_CONTINUOUS（incx==1）等
};
```

**4. 分核策略**（参考 arch35 `scasum_tiling_data.h` + arch22 平台 40 核）：

```
AIV_NUM        = GetAivCoreCount()   // 运行时查询，910B3 为 40
MIN_PER_BLOCK  = 4096                // 单核至少处理 4096 个复数元素（≈ 32KB），掩盖 launch 开销
blockCount     = min( AIV_NUM, max(1u, n / MIN_PER_BLOCK) )
               // n ≤ 4096 → 1 block，直接写 result（免 workspace + final kernel）
               // 4096 < n ≤ 40*4096=163840 → floor(n/4096) blocks
               // n > 163840 → 40 blocks（每 block 处理 n/40 个元素）
tileElements   = ceil_div(n, blockCount)
```

每 block 逻辑区间：`[blockId * tileElements, min((blockId+1)*tileElements, n))`

**5. Kernel 启动**：`scasum_kernel_do(x, partial_or_result, tiling, blockCount, stream)` → `<<<blockCount, nullptr, stream>>>`；当 `blockCount > 1` 时，后续 `scasum_final_kernel_do(partial_or_result, resultPtr, blockCount, stream)` → `<<<1, nullptr, stream>>>`。

**6. Workspace**：partial sums 位于 GM，大小 `blockCount * sizeof(float)`。实现上在 `result` 之后追加一段连续内存——host 侧在 `blockCount > 1` 时分配一个 `(blockCount + 1) * sizeof(float)` 的 Device buffer，resultPtr 指向首元素、partialPtr 指向后续 blockCount 元素；接口返回时不持有 workspace，调用方不感知。

#### kernel 侧设计

**编程模型**：SIMT（`__simt_vf__` / `asc_vf_call`），与 arch22 `sasum` / CtrsmBatched 同款；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

##### 主 kernel：scasum_kernel

**每个 block 的工作**：

```
for chunk [chunkBegin, chunkEnd) in myLogicalRange step UB_CHUNK:
    # DataCopy GM → UB（double buffer）
    #   incx == 1：连续 Complex64 搬入，直接 float2 配对
    #   incx > 1：按 stride 地址计算，仍保证每个复数的 real/imag 配对
    # Vector 计算：对 UB 中每个 float2，abs(real) + abs(imag) → float term
    # 局部归约：Vector 组内 FP32 累加 → block-local partial
# 块内跨 Vector 组归约 → 单 FP32 partial
# 写 partial[blockId]（blockCount==1 时写 result）
```

**incx == 1 专用路径**（覆盖 GPU baseline 全部 201 个性能 case，热路径）：
- 连续访问：将输入视为 `float2` 数组（`Complex64` 在 arch22 上原生按 `float2` 对齐），每 block 读取 `2 * tileElements` 个 float；
- UB 分块：ARCH22_UB = 192KB，留出 Vector 计算缓冲 32KB 后，单 buffer 可放 160KB / 8B = 20480 个复数元素 → **double buffer 各 10240 元素 ≈ 80KB**；
- 每轮计算 10240 个复数 → Vector 展开，**约 20480 次 abs + add，完全掩盖 GM 搬运延迟**。

**incx > 1 路径**（仅覆盖功能用例，不追求最优）：
- 以逻辑复数元素下标 i 计算地址：`addr = x + 2 * (i * incx)`；
- UB 仍以 float2 对装载，保证每次搬运的两个 float 是同一复数的 real/imag；
- 允许 stride 访问，但每轮 tile 仍保持连续 float2 计算。

**UB 预算**（arch22，192KB/核，double buffer 配置）：

| 用途 | 大小 | 备注 |
| --- | --- | --- |
| double buffer × 2（Complex64 输入） | 2 × 80KB = 160KB | 各 10240 复数元素 |
| Vector 中间 term 缓冲 | 16KB | FP32 数组 |
| Block-local 累加标量 + 对齐余量 | ~16KB | 保守 |
| **合计** | **≤ 192KB** | |

**数据类型与精度处理**：
- 所有累加使用 FP32（与 arch35 scasum 一致），不做 higher-precision 扩展；
- `abs(-0.0f)` 保持 `+0.0f`；
- 所有项非负，累加结果自然非负；接近 `FLT_MAX` 的大 n 场景允许溢出到 +Inf（golden 同为 FP32 累加，行为一致）；
- Vector 指令：优先使用 Ascend C 内置 `AscendC::Abs` + `AscendC::Add` 组合，编译器自动融合为 SIMD FMA；热循环无分支（incx 路径在 host 已设 flags，kernel 按模板/分支分派）。

**跨 block 同步**：block 之间无数据依赖，无需 `asc_syncthreads()` 跨 block 操作。主 kernel 内仅需 Vector 组内同步（`asc_syncthreads()`）以确保 block-local partial sum 正确归约。

##### Final reduction kernel：scasum_final_kernel

- 仅 `blockCount > 1` 时由 host 发起，`<<<1, nullptr, stream>>>`；
- 读取 `blockCount` 个 partial FP32 值，在 UB 中累加（Vector 归约），写入 `result`；
- `blockCount == 1` 时主 kernel 直接写 result，**不发起 final kernel**。

#### 性能设计

**访存 / 计算特征**：本算子为**访存带宽受限**（memory-bound）算子——`|Re|+|Im|` 为轻量计算（2 次 abs + 1 次 add ≈ 3 FLOPs / 复数元素），每个元素读 8B、写 0（中间结果留在 UB），每元素 8B / 3 FLOPs 算术强度极低。以最大性能 case（n=4,194,304, incx=1, GPU 基线 ~33 us）估算：

```
GM 读流量  = 4,194,304 × 8B = 32MB
GPU 耗时   = 33 us
等效带宽    = 32MB / 33us = 970 GB/s
910B3 目标  = 970 GB/s / 0.8 = 1.21 TB/s
```

**910B3 A2 峰值带宽**典型值 1.2~1.6 TB/s，**margin 约 1.3×~1.6×，有充足余量**。

**并行充分性**：40 Vector 核 × 双 buffer 流水线，每核每轮 80KB 搬运 + 80KB 计算，完全掩盖 GM 延迟（910B3 L2 192MB 足够装下大尺寸数据）。

**Double buffer 必要性**：incx=1 连续访问场景，GM 搬运速率 ≈ 计算速率（同为 ~1 TB/s 量级），单 buffer 会使 Vector 单元等待搬运而闲置；double buffer 将搬运/计算完全隐藏，利用率接近 100%。

**分核效率分析**：

| n | blockCount | 每 block 元素 | 每 block UB 轮次 | 备注 |
| --- | --- | --- | --- | --- |
| ≤ 4,096 | 1 | n | 1 | 单 block 直写 result，免 final kernel |
| 10K | 2 | 5K | 1 | |
| 100K | 24 | 4167 | 1 | |
| 500K | 40 | 12500 | 2（double buffer） | |
| 4M | 40 | 104857 | 11 | GM 读约 32MB，每核约 800KB，L2 缓存全覆盖 |

**Small n 退化**：n ≤ 4096 时 1 block 单轮完成，**kernel launch 开销（~5-10us）主导**——此时 GPU 基线 ~10us（`gpu_baseline.csv` scasum-perf-003/004 等 case），910B3 kernel 空 launch 也需 ~5us，故 0.8× 阈值要求实际需要硬件 Vector 核的启动开销与 GPU 相当。这是本算子唯一可能**接近阈值**的区间，后续以 warmup 掩盖首次 launch，有效样本均值仍可达标。

**Quick return 无 kernel**：n≤0 或 incx≤0 路径不发起任何 kernel，仅 Device 异步写 0.0f → 该路径耗时亚微秒级，远低于任何基准。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3，dav-2201，arch22） | √（性能测试设备） |
| Atlas 800T A3（arch22） | √ |

CANN 版本：9.1.0。核数、UB 大小运行时经 `GetAivCoreCount()` / 平台信息获取，不写死。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 数据类型 | 仅 COMPLEX64 输入 → FLOAT32 输出 |
| 参数合法性 | handle 空 → `HANDLE_IS_NULLPTR`；result 空 → `INVALID_VALUE`（含 quick return）；正常路径 x 空 → `INVALID_VALUE`；`(n-1)*|incx|` int32 溢出 → `INVALID_VALUE` |
| incx 语义 | incx>0 正常跨步；incx≤0 quick return（**不反向遍历**） |
| 非连续 Tensor | 不支持 leading dimension；incx 表达步长，仅此一种非连续访问形式 |
| broadcast / batch | 不涉及，单向量归约 |
| dynamic shape | 不要求；n 为运行时入参，host tiling 支持 |
| 原地语义 | 不支持；result 独立于 x |
| 确定性 | 不要求（FP32 累加顺序不保证逐位一致，按标量阈值验收） |
| 空 Tensor / 0 维 | n≤0 或 incx≤0：SUCCESS + result=0.0f（quick return） |
| 异步执行 | 依赖 `aclblasSetStream`；设备侧 partial workspace 与 result 同 stream 写入，顺序有保障 |
| 性能规模 | n ≤ 4×10⁶ 验收范围；实现不限上限 |
| kernel 类型 | 仅 AIV（Vector），无 Cube / MTE3 |

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 标量比对：`|actual - golden| ≤ atol + rtol × |golden|`，rtol=atol=2⁻¹³(1.22e-4)；matched_ratio 退化为单值（通过即 1）；max_abs_error ≤ 1e-2 或 32×ULP；golden 由 cblas Netlib `scasum` 生成；Inf/NaN 按任务书 §3.2 收窄规则 | 任务书 §3.2，生态算子开源精度标准 mixed_tolerance_standard |
| 特殊值规则 | golden 与 actual 同为 NaN / 同号 Inf / 一方 NaN 一方 ±Inf → 通过；一方有限值另一方 Inf/NaN → 不通过 | 任务书 §3.2 |
| 性能标准（判据） | 910B3 上 **201 个 case**（`gpu_baseline.csv`），平均单次 kernel 耗时（msprof op 采集 `Task Duration(us)`）≤ `gpu_ms / 0.8`（即达到 GPU 标杆 0.8× 性能）；warmup 5 次 + 有效采样 ≥10 次取平均 | 任务书 §3.3，`test_cases/gpu_baseline.csv` |
| 正态分布补充 | 自补 `GAUSS_NORM_5_2` 用例（μ∈[-5,5]，σ∈[0.1,2]），纳入精度验证 | 任务书 §3.5-3 |

## 测试方案

- **框架**：ops-blas 仓 CSV 驱动 GTest；arch22 分支须在 `test/asum/scasum/CMakeLists.txt` 调用 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`（arch35 分支已有注册代码，复制到 arch22 SOC_ARCH_DIRS 分支）。
- **CSV 格式**：表头对齐仓内 `test/asum/sasum/sasum_param.h`（含 n / incx / x 填充模式 / result 空指针标记）；枚举列大小写不敏感（仓内 `csv_loader.h` 统一解析）；路径由 `__FILE__` 自动推导同名 CSV。
- **用例覆盖**（随任务提供 CSV + 自补 GAUSS_NORM_5_2）：
  - L0 基础：n=0/1/2/3/7，incx=1/2/3；
  - L1 尺寸扫描：2 的幂及 2 的幂±1，直至 2²⁰；大 n × 极端值（接近 FLT_MAX）覆盖累加溢出；
  - L2 填充模式：均匀、全零、正负交替、极端值、Inf、NaN；
  - L3 GAUSS_NORM_5_2：自补正态分布 100~200 条（`gen_csv.py` 默认仅均匀，扩展 token 或单独 CSV）；
  - L4 quick return：n=0 / 负 n / incx=0 / 负 incx，均要求 result=0.0f + SUCCESS；
  - L5 负向用例：x=nullptr（正常路径）、result=nullptr（任意 n/incx，含 quick return 路径）、handle=nullptr；按 §3.5-4 负向用例缓冲区规则——**不得按 (n-1)*|incx| 分配 x**，用 dummy/NULLPTR 占位，由库侧在读数据前返回错误码。
- **golden 与比对**：`scasum_golden.h` 正常路径调 `cblas_scasum`（`test/utils/cblas_compat.h`），quick return 直接写 0.0f；标量结果用 `Verifier::verifyVector` + `applyMixedTolerance(ACL_FLOAT)` 做单值判定。
- **执行与 profiling**：`python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv ./scasum_test.csv`；性能用 msprof op 采集，`OpBasicInfo.csv` 按 kernel 名取 `Task Duration(us)` 平均，逐 case 对照 `gpu_baseline.csv` 判据。

## 兼容性分析

新算子，不涉及兼容性分析。接口声明已在 `include/cann_ops_blas.h:236` 预留（与 950PR 共用），本任务仅实现 arch22 路径；`blas/CMakeLists.txt` glob 自动收录 arch22 新目录，**不改动构建脚本**；kernel 直调形态，零 handle workspace 申请。
