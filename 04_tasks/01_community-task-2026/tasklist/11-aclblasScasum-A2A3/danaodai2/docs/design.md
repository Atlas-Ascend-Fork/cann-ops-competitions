# aclblasScasum 算子设计文档

## 1. 需求背景（required）

### 1.1 需求来源

本算子来自 CANN 社区任务 2026 `aclblasScasum` 算子开发任务，要求在 Atlas A2/A3（arch22）上使用 Ascend C 实现 complex64 向量绝对值分量之和，并对齐 cuBLAS `cublasScasum` / Netlib `scasum` 语义。任务书见 `task_aclblasScasum/aclblasScasum_AtlasA2_A3_task_doc.md`。

### 1.2 背景介绍

`aclblasScasum` 属于 BLAS Level-1 归约算子，数学上计算复数向量各元素实部与虚部绝对值之和。ops-blas 仓已有实数版本 `aclblasSasum` 的 arch22 实现，以及 950PR（arch35）的 `aclblasScasum` SIMT 实现；Atlas A2/A3 的 arch22 目录缺少 scasum。本设计在 arch22 上新增 SIMD 实现，复用仓内公共接口与 handle/workspace/stream 机制，不定义私有 API。

### 1.3 现状与基线

- 接口声明已在 `include/cann_ops_blas.h` 中提供，与 950PR 共用。
- 实数 sasum arch22 采用 `SetAtomicAdd` + 多核 partial 的方案；本算子不直接复制该方案，而是采用“每核独立 partial + SyncAll + 0 号核二级归约”，避免共享 result 的零初始化与原子加顺序依赖。
- 性能基线来自任务书 `test_cases/gpu_baseline.csv`（A100 / cuBLAS 实测），验收判据为 NPU 平均单次耗时 ≤ 1.25×GPU。

## 2. 需求分析（required）

### 2.1 需求描述

实现句柄式 BLAS 接口：

```cpp
aclblasStatus_t aclblasScasum(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

计算：

```text
result = Σ_{i=0}^{n-1} ( |real(x[i*incx])| + |imag(x[i*incx])| )
```

其中 `aclblasComplex` 为实部/虚部各 float32、交错存储，n 个复数对应 2n 个 float；物理长度为 `1+(n-1)*|incx|` 个复数。

### 2.2 参数与语义

| 参数 | 方向 | 类型 | 语义/约束 |
|---|---|---|---|
| handle | 输入 | `aclblasHandle_t` | 必须非空；空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | `int` | 任意整数；n≤0 为合法 quick return |
| x | 输入 | `const aclblasComplex*` | Device 只读；正常路径（n>0 且 incx>0）不得为空 |
| incx | 输入 | `int` | 任意整数；incx≤0 为合法 quick return |
| result | 输出 | `float*` | Device 标量；任意 n（含 quick return）不得为空 |

返回值：`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / `ACLBLAS_STATUS_EXECUTION_FAILED`。

### 2.3 异常与边界语义

1. `result == nullptr`：任意 n（含 quick return）返回 `INVALID_VALUE`。
2. `n > 0 && incx > 0 && x == nullptr`：返回 `INVALID_VALUE`。
3. `n <= 0 || incx <= 0`：不启动计算 kernel，向 Device result 写 `0.0f`，返回 `SUCCESS`；incx=INT32_MIN 也先走该分支，不能对 int32 取绝对值。
4. `(n-1)*incx > INT32_MAX`：返回 `INVALID_VALUE`（host 侧 int64 计算，任务书 §2.4）。
5. `n=1` 时跨度为 0，任意正 incx 只读 `x[0]`。
6. 输入 Inf/NaN 不替换为 0；结果可为 Inf/NaN，按任务书 §3.2 特殊值规则判定。

### 2.4 需求拆解

1. arch22 host 侧参数校验、quick return、跨度保护、tiling、workspace/stream 使用。
2. arch22 kernel 连续路径：complex64 按 float2 视图归约。
3. arch22 kernel 非连续路径：按逻辑下标安全读取 Re/Im。
4. 单 kernel 多核分片与二级归约。
5. 精度、边界、负向、特殊值、正态分布测试。
6. 性能满足 1.25×GPU 基线，并给出 msprof task duration。

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

```text
result = Σ (|Re(x[k])| + |Im(x[k])|),  k = i*incx, i=0..n-1
```

#### 3.1.2 支持数据类型

- 输入：COMPLEX64（`aclblasComplex`，实部/虚部各 FLOAT32，交错存储）。
- 输出：FLOAT32 标量。
- 中间计算：FLOAT32；归约顺序不要求逐位一致，按任务书标量精度阈值判定。

#### 3.1.3 支持形状与内存布局

- 逻辑一维 `[n]`；物理长度 `1+(n-1)*|incx|` 个复数。
- `incx>0` 时有效元素为 `x[i*incx]`；`incx<=0` 不遍历。
- 交错 float 下标：Re 为 `2*i*incx`，Im 为 `2*i*incx+1`；末端合法 float 下标为 `2*(n-1)*incx+1`。

#### 3.1.4 特殊值

- `Abs` 将 -0 归一为 +0；NaN 保持 NaN；±Inf 变为 +Inf。
- 逐项非负累加，大 n × 接近 FLT_MAX 时结果可溢出为 +Inf，属预期行为。

### 3.2 算子实现

#### 3.2.1 Host 侧设计

文件：`blas/asum/arch22/scasum_host.cpp`。

1. **参数校验**：handle → result → 正常路径 x，顺序保证 quick return 路径的 result 空指针仍返回 `INVALID_VALUE`。
2. **quick return**：`n<=0 || incx<=0` 时用 `aclrtMemcpy` 向 Device result 写 0，不启动计算 kernel。
3. **跨度保护**：`span = (int64)n-1 * (int64)incx`；`span > INT32_MAX` 返回 `INVALID_VALUE`，此时不形成任何指针运算。
4. **分核 tiling**：
   - 单核上限 `SCASUM_SINGLE_CORE_MAX = 4096` 复数；
   - 多核每核目标 `SCASUM_MIN_COMPLEX_PER_CORE = 32768` 复数；
   - 多核最少 4 核，并向上取整到 4 的倍数，避免 2/3 核 SyncAll 尾区；
   - 上限 `min(实际 AIV 核数, 64)`。
5. **对齐分片**：`base = (n/cores)/4*4`，余量按 4 个复数为单位分配给前若干核，末尾余数给最后一核。所有核起点为 4 复数倍数，即 32B 对齐，主 `DataCopy` 无需 pad。
6. **workspace**：每核 2 个 float 的 partial slot，共 `useCoreNum*8` 字节；检查不超过 handle workspace，使用 `GetEffectiveWorkspace(h)`，不额外 `aclrtMalloc`。
7. **下发**：直接调用 `scasum_kernel_do(...)`，绑定 `h->stream`；正常路径不同步 stream，由调用方在读取 result 前同步。

#### 3.2.2 Kernel 侧设计

文件：`blas/asum/arch22/scasum_kernel.cpp`。

1. **float2 视图**：complex64 的 Re/Im 交错存储，一个复数恰好是两个连续 float。连续 `incx=1` 时，一个复数区间就是 `2*count` 个连续 float，`Σ(|Re|+|Im|)` 等价于对该 float 区间做 `Abs` 后求和。
2. **双缓冲**：`TQue<VECIN, 2>`，单 tile `MAX_FLOATS_PER_TILE = 16384` float（64KB），两个 buffer 共 128KB，满足 A2 UB 预算。
3. **连续路径**：
   - 主 tile 用 `DataCopy`，起点 32B 对齐、长度为 8 float 的倍数；
   - 尾块用 `DataCopyPad`，零填充到 8 float 对齐，GM 实际读取严格限制在 `2*count` float；
   - `Abs(inLocal, inLocal, alignedCount)` → `PipeBarrier<PIPE_V>` → `ReduceSum(outLocal, inLocal, workBuf, alignedCount)`；
   - 每 tile 的标量和累加到 `localSum`（V_S 同步后 `GetValue`）。
4. **步长路径**：
   - 按逻辑 tile 逐复数读取 `inGM[2*(idx)*incx]` 与 `+1`，写入 UB 的 Re/Im 位置；
   - 尾 float 零填充后走同一 `Abs+ReduceSum`；
   - 保证跳过位置的 NaN/Inf 不进入累加。
5. **每核 partial**：`summaryBuf` 写 `localSum` 和 0 到两个 float，`S_MTE3` 同步后 `DataCopyPad` 到 `workGM[vecIdx*2]`，`PipeBarrier<PIPE_MTE3>`。
6. **二级归约**：`SyncAll()` 后，0 号核 `DataCopyPad` 读回 `useCoreNum*2` 个 float（含每核的 0 填充），补零到 8 对齐，`ReduceSum` 后写 result。
7. **同步**：`SyncAll` 只用于单次 kernel 内跨核 partial 汇合；所有 block 都到达同一 barrier，launch block 数 ≤ 物理 AIV 核数，满足 co-resident 要求。

#### 3.2.3 Tiling 与性能策略

实测调优（msprof，单位 us）：

| 配置 | n=32768 | n=65536 | n=98304 | n=1M | n=4M |
|---|---:|---:|---:|---:|---:|
| 32K/核，不取整 | 11.6~19.2 | 18.3~22.5 | 23.6~24.6 | 13.1~13.9 | 30.7~32.0 |
| 16K/核（更多核） | — | — | — | 15.4 | 32.3 |
| 32K/核 + 4 核取整 + 4K 单核上限 | 7.0~8.0 | 10.0~10.8 | 13.3~14.1 | 13.3~13.9 | 31.5~32.2 |

结论：小块 2/3 核的 barrier 代价高于并行收益；多核统一按 4 核分组后中尺寸稳定，1M 使用 32 核，2M/4M 使用 48 核。

#### 3.2.4 主要常量

| 常量 | 值 | 说明 |
|---|---:|---|
| `SCASUM_MAX_CORE_NUM` | 64 | block 数上限 |
| `SCASUM_COMPLEX_ALIGN` | 4 | 分片对齐复数数，保证 32B 对齐 |
| `SCASUM_MIN_COMPLEX_PER_CORE` | 32768 | 多核每核目标复数数 |
| `SCASUM_SINGLE_CORE_MAX` | 4096 | 单核最大复数数 |
| `MAX_FLOATS_PER_TILE` | 16384 | 单 tile float 数（64KB） |
| `PARTIAL_SLOT` | 2 | 每核 partial slot（sum + 0） |

### 3.3 支持硬件

| 芯片/产品 | 支持情况 | 说明 |
|---|---|---|
| Atlas A2 训练/推理系列 | 支持 | arch22 本实现，实测 Atlas 910（ascend910b，48 AIV） |
| Atlas A3 训练/推理系列 | 支持 | 共用 arch22 路径与接口，本机未实机验证 |
| Ascend 950PR/950DT | 支持 | 复用仓内 arch35 scasum，本任务不修改 |

### 3.4 算子约束限制

1. 不支持负步长遍历；`incx<=0` 按 quick return 处理。
2. `(n-1)*incx` 不得超过 INT32_MAX；超过返回 `INVALID_VALUE`。
3. 仅支持一维逻辑向量，不支持 leading dimension/padding。
4. 归约顺序不保证逐位确定性；按任务书标量精度阈值判定。
5. 不进行原地更新；result 为独立 Device 标量。
6. result 为 Device 指针，调用方读取前需同步 handle 绑定的 stream。

## 4. 可维可测分析（required）

### 4.1 精度标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32 标量：`|actual-golden| ≤ atol + rtol*|golden|`，atol=rtol=2^-13；max_abs_error ≤ 1e-2 或 32ULP；Inf/NaN 按任务书特例判定 | 任务书 §3.2、生态算子开源精度标准 |

测试结果：1024 条精度/边界用例全部通过；GTest 含 NullHandle 共 1025/1025；`verify_accuracy.py` 1024/1024 PASS。

### 4.2 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 性能标准 | 平均单次耗时 ≤ 1.25×GPU 基线；msprof op、5 次 warmup、≥10 次有效样本 | 任务书 §3.3 |

实测：

| case | n | GPU us | NPU 中位数 us | NPU/GPU | 结论 |
|---|---:|---:|---:|---:|---|
| TC_PF_1001 | 1,048,576 | 19.87 | 14.24 | 0.717 | PASS |
| TC_PF_1002 | 2,097,152 | 22.15 | 20.64 | 0.932 | PASS |
| TC_PF_1003 | 4,194,304 | 33.55 | 31.88 | 0.950 | PASS |

200 条 TC_PF 的 event 口径全部 PASS，最低 GPU/NPU 倍率 1.299。

### 4.3 测试设计

- 精度：基础小 shape、尺寸扫描（含 2^k±1）、incx=1/2/3、填充模式、极值/Inf/NaN、累加溢出、quick return、空指针、跨度溢出、正态分布。
- 性能：任务书 3 条典型 case、小尺寸延迟、规模扫描、特殊填充、随机混合，共 200 条。
- 负向用例：跨度越界只传单元素 dummy，不按 `(n-1)*|incx|` 预分配。
- 详细用例清单：`deliverables/case_list_accuracy.csv`、`deliverables/case_list_performance.csv`。

### 4.4 兼容性分析

- 新算子，不涉及历史版本兼容性。
- 接口与 950PR 共用 `include/cann_ops_blas.h` 声明，参数序列与 `cublasScasum` 一致。
- 不修改 `aclblasSasum` 行为；arch35 scasum 保持独立实现。

## 5. 附录：文件与证据

- Host：`blas/asum/arch22/scasum_host.cpp`
- Kernel：`blas/asum/arch22/scasum_kernel.cpp`
- 测试：`test/asum/scasum/arch22/{scasum_test.cpp,scasum_npu_wrapper.h,scasum_test.csv,README.md}`
- CMake：`test/asum/scasum/CMakeLists.txt`
- 补丁：`patches/scasum_arch22.patch`
- 精度日志：`results/accuracy_gtest.log`、`results/accuracy_verify.log`
- 性能日志：`results/performance_event_summary.log`、`results/msprof_summary.csv`
- 自测报告：`task_submission/2.1~4.2`
