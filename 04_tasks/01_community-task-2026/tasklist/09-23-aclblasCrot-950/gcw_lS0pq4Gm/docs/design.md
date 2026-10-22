# 需求背景（required）

## 需求来源

本文档提交 CANN 训练营南方科技大学专场 `aclblasCrot`（Ascend 950PR）算子设计评审，参与账号与个人目录为 `gcw_lS0pq4Gm`。本文结合候选原型（V5，commit `c967820`）及 Ascend 950PR 自测结果，说明接口、核函数设计和参数选择依据，现提交设计评审。

参考资料：
1. [官方任务页与任务书附件](https://www.hiascend.com/activities/task-center/details/b6f4d038761348f5955c0abd6d7e5e85?menu=tasks)（原始任务压缩包 SHA256：`b8c03ea406eca1c875887a7dc4dd305bf9d3b52c8e392ed9d8cb1db0b99bff05`）
2. [社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
3. [南科大专场活动流程说明 (#318)](https://gitcode.com/org/cann/discussions/318)、[社区任务流程说明 (#39)](https://gitcode.com/org/cann/discussions/39) 及 [本任务讨论帖 (#48)](https://gitcode.com/cann/ops-blas/discussions/48)
4. `cann/ops-blas` 仓库 `blas/rot/arch35/` 目录下同族算子实现（基线 commit `3c1542897f738857e928bce1996f1512220fafbc`）

| 版本 | 日期 | 修改内容 |
|---|---|---|
| V5 设计评审稿 | 2026-10-21 | 说明 `aclblasCrot` Host 与 Kernel 实现方案；连续向量路径统一采用寄存器计算（`tilingKey = 2`）；补充 V5 精度测试与原配置 Event 性能自测数据 |

## 背景介绍

### aclblasCrot 算子新增

`aclblasCrot` 对两个单精度复数向量执行原地（in-place）Givens 平面旋转，其中旋转系数 $c$ 为单精度实数，$s$ 为单精度复数。算子采用 `ops-blas` 公共 C 接口，通过 `handle` 绑定的 `aclrtStream` 直接下发 Ascend C 核函数，不经过 ACLNN 算子注册。

### 同族实现与适用范围

`arch35` 目录下的 `Srot` 提供了 Host 参数传递、连续路径与跨步路径的工程组织参考；同族 `Csrot` 的旋转系数 $c$、$s$ 均为实数，不能替代含复数系数 $s$ 的 `Crot`。本设计针对复数 $s$ 的虚部及共轭乘法实现旋转计算。复数向量采用 `complex64` AoS（Array of Structures）内存布局，每个元素占 8 字节（`{float real; float imag;}`）。

# 需求分析（required）

## 需求描述

在公共头文件 `include/cann_ops_blas.h` 中新增如下函数声明：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n,
    aclblasComplex* x, int incx,
    aclblasComplex* y, int incy,
    const float* c, const aclblasComplex* s);
```

参数及返回值约定如下：

| 参数 / 项目 | 约定说明 |
|---|---|
| `handle` | `ops-blas` 句柄；为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | 旋转元素个数；当 `handle` 有效且 `n <= 0` 时作为合法 no-op 直接返回 `ACLBLAS_STATUS_SUCCESS`，不读写向量或标量内存 |
| `x`, `y` | Device 侧 `complex64` 向量指针，原地更新；`n > 0` 时不得为空 |
| `incx`, `incy` | 复数元素步长，支持正数、负数和零；单侧跨度需满足 `(n - 1) * |inc| <= INT32_MAX` |
| `c` | 指向单个 `float` 的旋转系数指针，可位于 Host 或 Device；`n > 0` 时不得为空 |
| `s` | 指向单个 `aclblasComplex` 的旋转系数指针，可位于 Host 或 Device；`n > 0` 时不得为空 |
| 数据范围 | 实部与虚部按任务书支持 `FLOAT32` 取值，不对 $c^2 + |s|^2 = 1$ 做归一化假设 |
| 错误返回 | `n > 0` 时若 `x/y/c/s` 任一为空、步长跨度超过 `INT32_MAX` 或指针属性查询失败，返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 执行方式 | 在 `handle->stream` 上异步下发核函数，调用方在读取结果前需同步流 |

## 需求拆解

1. 按旋转前原始 $x_k, y_j$ 计算复数系数乘法与共轭乘法，并原地写回 $x, y$。
2. 支持连续步长（`incx = incy = 1`）、非零正负步长以及零步长（顺序递推），使用 64 位整数计算跨度与地址偏移。
3. 独立识别 `c` 与 `s` 的 Host / Device 指针属性，支持 HH、HD、DH、DD 四种组合，不在 Host 侧解引用 Device 标量指针。
4. 在 Ascend 950PR（CANN 9.1.0）环境下按 `arch35` 直调目录结构构建算子与 GTest 测试工程 `crot_test`。
5. 保留官方 `crot_test.csv` 原始 1,200 条用例，实部与虚部分别按任务书混合容差判据校验，并针对特殊值与 `cublasCrot` 进行对照。
6. 按任务书要求预热 5 次、有效采样超过 50 次（本测试取 100 次）计算算术平均耗时，评估 200 个性能用例表现。

# 详细设计（required）

## 算子分析

### 数学公式

对第 $i$ 个逻辑步（$i = 0, \dots, n - 1$）读取的当前元素 $x = a + bi$、$y = d + ei$ 以及旋转系数 $c$、$s = u + vi$，变换公式为：

```text
x' = c * x + s * y
y' = c * y - conj(s) * x

xr = c * a + (u * d - v * e)     xi = c * b + (u * e + v * d)
yr = c * d - (u * a + v * b)     yi = c * e - (u * b - v * a)
```

每次迭代先读取旋转前的 4 个实虚分量 $(a, b, d, e)$，计算新分量后再写回。为对齐受测 `cublasCrot`（cuBLAS 12.8.4.1 / RTX 5060）在抵消与溢出边界上的舍入行为，本实现采用如下二次 FP32 融合乘加（FMA）求值图，其中 `fl` 表示单独一次 FP32 乘法舍入，`fma(p, q, r)` 表示单次舍入乘加 $p \cdot q + r$：

```text
xr = fma(c, a,  fma(u, d, -fl(v * e)))
xi = fma(c, b,  fma(v, d,  fl(u * e)))
yr = fma(c, d, -fma(u, a,  fl(v * b)))
yi = fma(c, e, -fma(u, b, -fl(v * a)))
```

- `tilingKey = 2`（连续向量寄存器路径）：在 `CrotRegisterCompute` 中通过 `Muls`、`Neg` 与 `MicroAPI::MulAddDst` 实现上述求值图；
- `tilingKey = 1`（跨步与零步长 SIMT 路径）：在 `crot_simt_compute` 中使用 `volatile float` 保存 4 个初始乘积以执行独立 FP32 舍入，再调用 `fmaf` 完成两次融合乘加；
- `tilingKey = 0`（保留的传统 SIMD 路径，V5 Host 不选用）：通过 `Muls` 初始化与两次 `Axpy` 累加并将负号折入标量；
- 测试代码中的 CPU 参考实现（`crot_golden.h`）同步采用 `std::fma` 按相同求值图计算。其外部对照依据为前期保存的真实 `cublasCrot` 输出样本与设备侧位型探针测试结果。

### 支持数据类型

输入输出向量 `x`、`y` 与系数 `s` 为 `aclblasComplex`（`complex64`），系数 `c` 为 `float`（`float32`）。`aclblasComplex` 由两个连续的 32 位浮点数组成，逻辑复数下标 $j$ 对应 `float` 数组下标 $2j$（实部）与 $2j + 1$（虚部），字节偏移为 $8j$。

### 支持形状

算子输入为一维逻辑长度 $n$。当 $n > 0$ 时，每侧物理缓冲区需容纳至少 $1 + (n - 1) \cdot |\text{inc}|$ 个复数元素：
- 正步长（$\text{inc} > 0$）：逻辑位置 $i$ 对应复数元素下标 $j = i \cdot \text{inc}$；
- 负步长（$\text{inc} < 0$）：逻辑位置 $i$ 锚定在缓冲区尾端，$j = (n - 1 - i) \cdot |\text{inc}|$；
- 零步长（$\text{inc} = 0$）：固定访问 $j = 0$，下一轮迭代读取上一轮写回后的复数值，按 $i = 0 \dots n - 1$ 顺序递推。

计算 $|\text{inc}|$ 时先将 `int` 提升至 `int64_t` 再取负，可正常处理 `INT_MIN`（例如 $n = 1, \text{inc} = \text{INT\_MIN}$ 时跨度为 0，合法；$n = 2, \text{inc} = \text{INT\_MIN}$ 时跨度超过 `INT32_MAX`，返回 `ACLBLAS_STATUS_INVALID_VALUE`）。

## 算子实现

### 实现方案

实现代码位于 `blas/rot/arch35/`，包含 `crot_host.cpp`、`crot_kernel.cpp`、`crot_kernel.h`、`crot_tiling_data.h`；测试工程位于 `test/rot/crot/`，构建目标为 `crot_test`。

#### 3.2.1 host侧设计

Host 入口 `aclblasCrot` 的检查顺序为：`handle == nullptr` $\to$ `n <= 0` $\to$ `x/y/c/s` 空指针检查 $\to$ `CrotSpanFits` 跨度溢出检查 $\to$ `c/s` 指针位置识别。设备可用 AIV 核数通过 `GetAivCoreCount()` 获取并保存在函数内 `static` 常量中，避免每次调用重复查询；若核数返回 0 则返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。

`c` 与 `s` 分别通过 `CrotCheckPtrLocation` 调用 `aclrtPointerGetAttributes` 识别内存属性：仅当 `ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE` 时标记为 Device 指针（`cIsDevice = 1` / `sIsDevice = 1`）并将地址传给核函数；若为 Host 或 Unregistered 内存，则在 Host 侧读取标量值填入 `CrotTilingData`，不引入额外的 Host-Device 同步拷贝。

##### 1. 分核策略

记向量长度 $N = n$，设备可用 AIV 核数为 $C$（在受测 Ascend 950PR 上 $C = 56$），启动核数为 $B$：

| 路径与条件 | 启动核数 $B$ 与工作量划分 |
|---|---|
| 连续向量 $N \ge 524288$ | $B = \min(N, C)$；设 $q = \lfloor N / B \rfloor, r = N \bmod B$，前 $r$ 个核处理 $q + 1$ 个元素，其余核处理 $q$ 个元素，核偏移为 $b \cdot q + \min(b, r)$ |
| 连续向量 $0 < N < 524288$ | $B = \min(N, C, \lceil N / 2048 \rceil)$（当 $N < 65536$ 时再取 $\min(B, 56)$）；同样按 $q, r$ 均分 |
| 非零跨步向量（SIMT） | $B = \min(C, \lceil N / 128 \rceil)$，每核线程数 `nthreads = min(2048, ceilAlign(ceil(N / B), 128))`，按 Grid-Stride 循环分配 |
| 含零步长向量（`incx == 0 \|\| incy == 0`） | $B = 1$，`nthreads = 128`（核内仅 `block 0, thread 0` 执行串行循环） |

分核数量完全由 $(N, \text{incx}, \text{incy})$ 和设备核数 $C$ 决定。在 $C = 56$ 的受测设备上，当 $N < 524288$ 时实际核数受 $\lceil N / 2048 \rceil$ 约束，例如 $n = 32768$、$43752$、$64504$ 时实际启动核数分别为 $B = 16$、$22$、$32$。V5 相对前一候选版本（V4）仅将 $N < 65536$ 分支中的核数上限从 28 调整为 56，受 $\lceil N / 2048 \rceil$ 限制，该改动仅在 $N \in [57345, 65535]$ 时使启动核数从 28 增加到 $29 \sim 32$（在官方 200 个性能用例中仅影响 `TC_PF_1169`）。

##### 2. 数据分块和内存优化策略

| 路径 | 单 Tile 复数元素数 $T$ | 缓冲结构与流水方式 | 单核显式 UB 占用 |
|---|---:|---|---:|
| `tilingKey = 2`（V5 连续向量路径） | 4096 | 4 个 `TQue`（`inQueueX_`、`inQueueY_`、`outQueueX_`、`outQueueY_`），depth = 1，不分配 `workBuf_`；单 Tile 依次执行 `CopyIn -> Compute -> CopyOut` 后复用缓冲 | $32 T\text{ B} = 128\text{ KiB}$ |
| `tilingKey = 0`（保留的传统 SIMD 路径） | 2048 | 4 个 `TQue`（depth = 2）加 `workBuf_`（10 个 `tileSize` 切片），软件流水预取下一 Tile | $104 T\text{ B} = 208\text{ KiB}$ |
| `tilingKey = 1`（跨步 SIMT 路径） | — | 线程按 64 位偏移直接读写 GM，不使用 `TQue` 搬运 | 不使用 SIMD 队列 UB |

`crot_tiling_data.h` 中对 `CROT_SMALL_TILE_SIZE * 32U <= 128U * 1024U` 设有编译期 `static_assert`。该数值为代码中显式分配的 `TQue` 缓冲大小，算子不额外申请 GM workspace。

##### 3. tilingkey规划策略

| `tilingKey` | 选择条件 | 核函数入口与实现 |
|---:|---|---|
| `2` | `incx == 1 && incy == 1`（V5 对所有合法正长度 $N$ 选用） | `crot_aiv_kernel` $\to$ `CrotAIV` + `CrotRegisterCompute` |
| `0` | 保留的连续双缓冲 SIMD 分支（V5 Host 不选用） | `crot_aiv_kernel` $\to$ `CrotAIV`（`workBuf_` + `DeInterleave`/`Interleave`） |
| `1` | 非单位步长（含正负跨步与零步长） | `crot_simt_kernel` $\to$ `crot_simt_compute` |

#### 3.2.2 kernel侧设计

1. **连续路径（`CrotAIV` 与 `CrotRegisterCompute`，`tilingKey = 2`）**：
   - `CrotAIV::Init` 根据核编号计算本核起始复数偏移 `myOffset_` 与元素数 `myCount_`，若 `cIsDevice` 或 `sIsDevice` 为真则从 GM 指针读取系数值，并为 4 个输入输出队列分配单缓冲（每个队列 `2 * 4096 * sizeof(float) = 32 KiB`）。
   - `CrotAIV::CopyIn` 与 `CopyOut` 均按当前分块实际有效字节数 `dataCount * 8U` 调用 `DataCopyPad`，避免尾块越界访问 GM。
   - `CrotAIV::Compute` 在 `tilingKey == 2` 时通过 `asc_vf_call<CrotRegisterCompute>` 进入寄存器计算函数。`CrotRegisterCompute` 对当前 Tile 计算子循环次数 `loops = (count + 63) / 64`，在进入 `i = 0 .. loops - 1` 寄存器子循环前调用 `Duplicate` 将 `u`、`v`、`c` 预广播至 `cu`、`cv`、`cc` 三个 `RegTensor<float>`（该广播随 `CrotRegisterCompute` 在每个 Tile 内执行一次）。子循环内使用 `LoadAlign<float, LoadDist::DIST_DINTLV_B32>` 将 UB 中的 AoS 复数解交错加载到实虚寄存器 `(a, b)` 与 `(d, e)`，通过 `Muls`、`Neg`、`MulAddDst` 完成四分量更新，再通过 `StoreAlign<float, StoreDist::DIST_INTLV_B32>` 交错写回输出队列（注：仅保留的 `tilingKey = 0` 分支在 UB 切片上直接调用 `DeInterleave` 与 `Interleave`）。
2. **跨步与零步长路径（`crot_simt_compute`，`tilingKey = 1`）**：
   - 当 `incx != 0 && incy != 0` 时，各线程通过 Grid-Stride 循环处理互不重叠的逻辑下标 $i$，以 `uint64_t` 计算 `float` 下标 `2ULL * j * absInc`，读取四个分量并完成 `fmaf` 计算后写回。
   - 当 `incx == 0 || incy == 0` 时，核函数开头过滤掉非 `(blockIdx.x == 0 && threadIdx.x == 0)` 的线程，由单线程使用 `volatile __gm__ float*` 在每次迭代重读并写回被复用的元素地址，保持 $n$ 次顺序旋转递推语义。
   - 各核处理独立的输出下标区间，无需跨核同步；同一 `handle->stream` 上的前后依赖由流顺序保证。

## 支持硬件

| 硬件环境 / 软件版本 | 状态 |
|---|---|
| Ascend 950PR / CANN 9.1.0 / `arch35` (`ascend950`) | 已在 `Ascend950PR_9579` 上完成构建与本文所列用例自测 |
| 其他芯片型号或 CANN 版本 | 未开展验证 |

## 算子约束限制

1. 算子支持向量 $x$ 自身原地更新和向量 $y$ 自身原地更新；本文按 $x$ 与 $y$ 互不重叠、Device 标量与输出向量互不重叠的使用场景设计，交叉别名场景未验证。
2. 当 $n > 0$ 时，调用方需保证传入的指针非空且物理容量满足 `1 + (n - 1) * |inc|`，并在异步流执行完成前保持 Device 内存有效。
3. 零步长分支因数学定义为顺序递推，采用单核单线程串行执行。官方 200 个性能用例均为连续步长（`incx = incy = 1`）。

# 可维可测分析

## 1. 正确性与特殊值验证

精度验证按任务书 §3.2 对实部和虚部分别检查 `FLOAT32` 指标：`rtol = atol = 2^-13`，匹配比例 $\ge 0.99$，最大绝对误差上限为 $\max(10^{-2}, 32\text{ ULP})$。

- **V5 正确性测试结果**：在 Ascend 950PR 上运行 `crot_test`，共 **2211 / 2211** 个独立测试用例通过（日志 `acc5.log`）。其中包含 2,196 条 CSV 驱动用例（完整保留官方 `crot_test.csv` 原始 1,200 条用例，并补充 996 条边界与步长组合用例）以及 15 个 `TEST_F` 专项用例（覆盖 HH/HD/DH/DD 指针模式、Device 标量非零偏移、缓冲区首尾及步长空洞 Canary 检查、同流异步生产者-消费者串联、未截断正态/均匀分布输入，以及与 LAPACK `crot_` 的交叉核对）。
- **特殊值对照说明**：当前 V5（commit `c967820`，动态库 SHA256 `49773e1ea2c7a6a02e6557087c8d91afd2c988e525579bc0252b6476c8887b70`）的核函数计算源码与前期独立 `cuBLAS` 对照版本（V2）相同；前期已完成 117 组特殊值组合（161,304 个分量）和 12 个极端微用例的位级对照。V5 本轮仅调整了 Host 侧核数/路径选择，没有在 V5 二进制上重复运行该独立位型探针。

## 2. 性能自测与计时口径说明

### 2.1 官方 `verify_performance.py` 的整数毫秒解析与亚毫秒阈值冲突

任务书附件中的 `gpu_baseline.csv`（SHA256：`e32c4bf01b5a2ac87af439ac87c39f1ac613d6ff21d6ab10d0383646282bc1bd`）共 200 行、196 个唯一 `(n, incx, incy)` 键，参考耗时 `gpu_ms` 全部在 `0.002541 ms` 至 `0.047807 ms` 之间；按倍率阈值 `0.4` 计算，允许的最大单次平均耗时为 `0.047807 / 0.4 = 0.1195175 ms`（`119.5175 μs`）。

官方脚本 `verify_performance.py`（SHA256：`67989a3c009698bcabe4e09d2404be63539fb4bcc6f60c63439391f8c7b14eff`）通过如下正则从 GTest 输出中提取整条用例的整数毫秒耗时：

```python
m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
if m: results.append({"case_name": m.group(1), "npu_ms": int(m.group(2)), "status": "ok"})
```

随后按 `if base is not None and npu > 0:` 计算 `ratio = base / npu`：
- 当 `npu == 0`（耗时不足 1 ms）时，进入 `else` 分支并标记为 `NO_REF`；
- 当 `npu >= 1`（正整数毫秒）时，`ratio <= 0.047807 / 1 < 0.4`，标记为 `FAIL`。

因此，整条 GTest 的整数毫秒计时与亚毫秒级的单次阈值不匹配，在不修改原脚本与基线文件的前提下不存在进入 `PASS` 分支的整数毫秒值。同时，官方 `test_cases/README.md` 第 95～105 行介绍了使用 `msprof` 从 `OpBasicInfo.csv` 提取微秒级 `Task Duration(us)` 的方式。

### 2.2 V5 原配置 Event 性能自测结果

我们按任务书 §3.3 的预热、采样和平均规则补充了微秒级 Event 自测（探针 `probe_v3`），请评审确认其验收适用性。测试读取官方 `crot_test.csv` 全部 200 个 `TC_PF_1001`～`TC_PF_1200` 的原始参数（含原系数与 198 个 Host、2 个 Device 指针模式），每次调用前在计时区间外恢复输入向量，每个用例预热 5 次、有效采样 100 次并取算术平均。本次使用 `aclrtEventElapsedTime` 记录同一 `stream` 上包围公共 API 调用的事件区间；该区间与 profiler 记录的 kernel 任务时间边界不同，可能包含提交空档。

1. **任务书 §3.3 三个典型用例实测结果**（表中阈值采用任务书直接给出的标杆耗时，展示值保留三位小数，计算与判定使用未舍入原精度）：

| 用例 | 形状 (`n, incx, incy`) | 任务书标杆阈值 (`μs`) | `r1-full` 均值 (`μs`) | `r1-full` 倍率 | `r2-full` 均值 (`μs`) | `r2-full` 倍率 | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| `TC_PF_1001` | `1048576, 1, 1` | 21.96 | 17.369 | 0.506 | 16.685 | 0.527 | 通过 |
| `TC_PF_1002` | `2097152, 1, 1` | 43.70 | 28.021 | 0.624 | 27.230 | 0.642 | 通过 |
| `TC_PF_1003` | `4194304, 1, 1` | 119.52 | 48.850 | 0.979 | 51.437 | 0.929 | 通过 |

> 注：若按 `gpu_baseline.csv` 除以 `0.4` 的未舍入阈值（分别为 `21.965`、`43.705`、`119.5175 μs`）核对，上述三个典型用例在两轮测试中同样全部满足要求。

2. **全量 200 个性能用例与历史批次汇总**：

| 测试轮次 / 批次 | 用例范围与采样规模 | 结果与最低倍率用例 | 备注 |
|---|---|---|---|
| `r1-full`（原配置完整轮 1） | 200 例，每例 5 warmup + 100 sample（共 21,000 条调用记录，20,000 条有效采样） | **200 / 200 通过**；最低倍率 `TC_PF_1117`：均值 `7.701 μs`，阈值 `8.220 μs`，`ratio = 0.427` | 探针二进制 `ad06afb0...`（归档中仅保留 `r2` 探针源码，故以 `r2-full` 为主、`r1-full` 为补充） |
| `r2-full`（原配置完整轮 2） | 200 例，每例 5 warmup + 100 sample（共 21,000 条调用记录，20,000 条有效采样） | **200 / 200 通过**；最低倍率 `TC_PF_1039`：均值 `5.692 μs`，阈值 `6.3525 μs`，`ratio = 0.446` | 当前探针二进制 `b89e61ce...` |
| `r1169-1..10`（定向复测 10 轮） | `TC_PF_1169` 单例连跑 10 轮（每轮 105 条调用记录，共 1,000 条有效采样） | **10 / 10 通过**；均值 `5.893 ~ 6.625 μs`，阈值 `6.935 μs`，`ratio = 0.419 ~ 0.471` | 针对 `n = 64504` 的重复核验 |
| 早期固定 Host 系数协议 100 轮 | 200 例 × 100 轮（固定 Host 系数，未逐次恢复输入） | 99 轮 200/200，其中 `B12/TC_PF_1169` 出现 1 次超阈值记录（均值 `7.294 μs`，阈值 `6.935 μs`，`ratio = 0.3803`） | 固定系数、未恢复输入的历史协议；结果单独列示 |

上述 `r1-full`、`r2-full` 及 `r1169-1..10` 合计包含 43,050 条调用记录，其中 warmup 记录 2,050 条、有效性能采样 **41,000** 条（归档包 `q08-verify3-evidence-20261021.tar.gz`，SHA256：`d5243ec5c3dc454133609b5167e0379740fcb19818f5bcb59fa06489d402f414`）。独立复算脚本对样本行数、精确 `sample_id` 集合（warmup `0..4`、sample `0..99`）、参数字段、6 个返回状态码及算术均值进行了复核，结果与汇总一致。

### 2.3 内存与后续验收准备

任务书 §3.4 对内存无硬性阈值要求，§4 自测报告需填写内存占用数据。V5 连续路径（`tilingKey = 2`）单核显式 UB 队列占用为 128 KiB；早期 V2 批次曾记录 Host `maxRSS` 为 `392,464 KiB` 与 Device 内存快照。待设计评审通过后，正式提交自测报告时将据实际采集范围填写 V5 内存占用数据；是否补充 `msprof` 性能采集则根据评审对计时口径的反馈决定。

## 兼容性分析

新增 `aclblasCrot` 接口保持 `cann/ops-blas` 现有 `Srot` / `Csrot` 的头文件声明与构建兼容性，仅面向 `arch35`（Ascend 950PR / CANN 9.1.0）实现。本 PR 仅申请设计文档评审；待设计评审通过并合入后，再按活动流程整理个人代码仓分支、邀请官方账号并提交正式验收申请；收到后台测试通过通知后再向 `cann/ops-blas` 提交代码 PR。
