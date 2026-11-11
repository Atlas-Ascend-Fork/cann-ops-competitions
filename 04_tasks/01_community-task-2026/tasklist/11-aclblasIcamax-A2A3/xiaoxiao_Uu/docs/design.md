# 需求背景

## 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务名称 | aclblasIcamax 算子开发（A2/A3） |
| 任务来源 | 202611 社区任务讨论（CANN ops-blas，A2/A3 方向） |
| 开源仓库 | https://gitcode.com/cann/ops-blas |
| BLAS 级别 | Level 1（向量-向量 / 向量归约） |
| 工程模式 | aclblas 句柄式（handle-based）kernel 直调，Ascend C 实现 |
| 交付目录 | `blas/iamax/arch22/icamax_*` |
| 参考资料 | 仓内 950 任务书；`blas/iamax/arch22/isamax_*`（同族 isamax 的 arch22 实现） |

## 背景介绍

`Ic amax`（Netlib 名 `icamax` / cblas 名 `cblas_icamax`）是 BLAS Level 1 中的"最大模分量索引"操作，用于复向量的最大模元素定位，是后续 `cblas_caxpy` pivot 选择、稀疏直接法选主元、以及各类 norm 估计（Hager / Higham 估计）的基础 kernel。

其数学语义与 `isamax` / `idamax` 的关键差异在于"模"的定义：复数域下采用 **1-范数（非欧氏模）**，即 `|Re| + |Im|`，而不是 `sqrt(Re² + Im²)`。这一点与 Netlib/BLAS 参考实现完全一致，实现时不得替换为 2-范数或 `hypot`。

仓库现状：`blas/iamax/` 家族当前仅存在 `arch35`（A5/Ascend 950 方向）实现。arch35 版本依赖 AscendC SIMT 能力（`asc_simt`）进行 stride 访问与归约，该能力在 A2/A3（910B，arch22）上不可用。因此需要在 `blas/iamax/arch22/` 下新建面向 Atlas A2/A3 的 `icamax_*` 实现，并与同目录下的 `isamax` 保持实现范式一致：

- 复数输入在 UB 内"拆实虚 → 求 `|Re| + |Im|`"，之后复用与 `isamax` 相同的 **reduce-max-with-index** 骨架；
- stride 访问使用 **membase gather / 标量访问** 实现，禁止任何 SIMT 接口；
- 保持 aclblas 句柄式 API 与 host/kernel 直调的工程结构，便于后续 cblas 包装层接入。

# 需求分析

## 需求描述

实现并开源 `aclblasIcamax` 算子，支持 Atlas A2/A3 硬件（arch22），输入为 COMPLEX64（`aclblasComplex`，两个 float32 交错），输出为 `int` 型索引，返回复向量 `x` 中 `|Re| + |Im|` 最大的元素的 **1-based** 下标；若存在并列，返回 **最小索引**。接口与 Netlib `icamax` / cblas `cblas_icamax` 对齐：

```c
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle,
                              int n,
                              const aclblasComplex* x,
                              int incx,
                              int* result);
```

功能与精度需对标 cblas/Netlib 参考实现（含 1-based 约定、并列取小、1-范数模），性能需在典型规模上满足「NPU Task Duration 中位数 ≤ A100 cuBLAS 中位数 / 0.8」的验收线。

## 需求拆解

| 编号 | 拆解项 | 说明 | 依赖/备注 |
| --- | --- | --- | --- |
| R1 | API 与工程骨架 | 在 `blas/iamax/arch22/` 下新建 `icamax_*`，含 host 侧 tiling/launch 与 kernel 侧 Ascend C 实现，遵循 aclblas 句柄式 kernel 直调 | 参考 `blas/iamax/arch22/isamax_*` |
| R2 | 复数模计算 | UB 内将 COMPLEX64 拆成 real/imag 两条 float32 流，逐元素计算 `abs(re) + abs(im)` | 禁止 `sqrt`/`hypot`，禁止 2-范数 |
| R3 | 带索引的 max 归约 | 在 float32 幅值流上做 reduce-max-with-index，维护 (value, index) 对，比较器保证"严格大于才更新、相等保留小索引" | 与 isamax 同骨架，需扩展到复数拆分 |
| R4 | stride 支持 | `incx = 1` 走连续向量化搬运；`incx > 1` 走 membase gather / 标量访问 | 禁止依赖 AscendC SIMT / `asc_simt` |
| R5 | 多核分核与 tiling | host 侧完成核数选择、chunk 划分、UB 双缓冲容量计算、tilingkey 选择 | 需覆盖 n 从 1 到 16M+ 的场景 |
| R6 | 语义细节 | 1-based 输出、并列取最小索引、`n <= 0` 时返回 0、`incx < 1` 报错 | 与 cblas 行为一致 |
| R7 | 精度验证 | 索引结果与 cblas/Netlib 参考 bit-exact；内部浮点按 FLOAT32 混合容差 | 归约索引类精度要求 |
| R8 | 性能验收 | 典型 case（n=1M/4M/16M）NPU Task Duration 对标 A100 cuBLAS / 0.8 | 性能测试设备：Atlas 800T A2 (910B3) |
| R9 | 文档与约束 | 输出 design.md、约束限制、可维可测分析 | 本文档 |

# 详细设计

## 算子分析

### 数学公式

给定复向量 `x ∈ C^n`，步长 `incx ≥ 1`，定义元素 `x_i = x[ (i-1) * incx ] = re_i + j * im_i`，`i = 1..n`：

```
m_i   = |Re(x_i)| + |Im(x_i)|          (复数 1-范数，非欧氏模)
result = min { i : m_i = max_k m_k }   (1-based，并列取最小索引)
```

等价伪式：

```
best_val = -0.0 ; best_idx = 0
for i in 1..n:
    t = |re_i| + |im_i|
    if t > best_val or (t == best_val and i < best_idx):
        best_val = t ; best_idx = i
```

约定与边界语义（对标 cblas `icamax`）：

| 输入 | 输出语义 |
| --- | --- |
| `n <= 0` | `*result = 0`，返回 `ACLBLAS_STATUS_SUCCESS` |
| `n >= 1` | `*result ∈ [1, n]`（1-based） |
| 全部 `m_i` 相等（含全 0） | `*result = 1` |
| 存在并列最大值 | 取最小索引 |
| 含 NaN 分量 | 约束限制中声明为未定义行为（与 Netlib 一致），不保证 bit-exact |
| ±Inf 分量 | 正常参与比较，`|+inf| + |im| = +inf`，可作为最大值 |

关键点：模运算使用 `|Re| + |Im|`，是 **1-范数**；与 `icamax` 在 BLAS 参考实现中的 `cabs1` 定义一致。实现中必须使用 `fabsf(re) + fabsf(im)`，不得使用 `sqrt(re*re + im*im)`、`hypotf` 或任何会产生不同舍入/不同比较结果的写法。

### 支持数据类型

| 类别 | 数据类型 | 说明 |
| --- | --- | --- |
| 输入向量 `x` | COMPLEX64（`aclblasComplex`，2 × float32 交错，8 字节/元素） | 单一数据类型，首版本不支持 COMPLEX32/COMPLEX128 |
| 输出索引 `result` | int（int32，4 字节） | 1-based 索引 |
| 内部计算精度 | float32（幅值流）、int32（索引流） | 幅值不累加、只做选择，无累积误差 |
| 计算精度口径 | FLOAT32 混合精度容差 | 仅用于内部幅值比较；最终索引要求 bit-exact |

说明：由于归约操作是 **max 选择**而非求和累加，逐元素幅值 `|re| + |im|` 在 float32 下逐元素精确（加法两输入为 exact 的绝对值结果，只有一次舍入风险，且与参考实现同序同法），因此最终索引可做到与参考实现 bit-exact，详见"可维可测分析"。

### 支持形状

| 参数 | 形状/约束 | 说明 |
| --- | --- | --- |
| `n` | 标量 int，`0 ≤ n ≤ INT_MAX` | 向量长度（元素个数） |
| `x` | 逻辑长度 `1 + (n-1) * incx` 个 COMPLEX64 | 以 `x` 为首地址的 GM 区间，需可读 |
| `incx` | int，`incx ≥ 1` | 元素步长（单位：元素，非字节） |
| `result` | 标量 1 × int | 设备侧 GM 地址，4 字节对齐 |
| 实际访存范围 | 字节数 = `(1 + (n-1) * incx) * 8` | 需在 tensor 越界检查范围内 |

典型场景规模（性能用例）：

| 场景 | n | incx | 特征 |
| --- | --- | --- | --- |
| 小向量 | 1 / 16 / 256 / 4096 | 1 | 单核单块，短路径 |
| 中向量 | 65536 / 262144 | 1、2 | 多核单趟 |
| 大向量 | 1,048,576（1M） | 1 | 多核、向量化 |
| 超大向量 | 4,194,304（4M） / 16,777,216（16M） | 1 | 多核、双缓冲、访存主导 |
| 稀疏步长 | 1,048,576 | 3 / 8 / 16 | gather 路径，访存不连续 |

### 接口定义

```c
typedef enum {
    ACLBLAS_STATUS_SUCCESS = 0,
    ACLBLAS_STATUS_INVALID_PARAM = 1,
    ACLBLAS_STATUS_NOT_SUPPORTED = 2,
    ACLBLAS_STATUS_RUNTIME_ERROR = 3
} aclblasStatus_t;

typedef struct { float real; float imag; } aclblasComplex;

aclblasStatus_t aclblasIcamax(aclblasHandle_t handle,
                              int n,
                              const aclblasComplex* x,
                              int incx,
                              int* result);
```

参数语义：

| 参数 | 方向 | 含义 | 约束 |
| --- | --- | --- | --- |
| `handle` | in | aclblas 句柄，持有 stream / 算力与 UB 资源信息 | 非空 |
| `n` | in | 向量长度 | `n ≥ 0`；`n = 0` 时写 `*result = 0` |
| `x` | in | 设备侧 COMPLEX64 向量首地址 | 非空（`n > 0` 时）；8 字节对齐 |
| `incx` | in | 元素步长 | `incx ≥ 1`；非法值返回 `ACLBLAS_STATUS_INVALID_PARAM` |
| `result` | out | 设备侧 int 地址，写入 1-based 索引 | 非空；4 字节对齐；不得与 `x` 重叠 |

返回值：成功 `ACLBLAS_STATUS_SUCCESS`；参数非法（空指针、`incx < 1`、地址不满足对齐）返回 `ACLBLAS_STATUS_INVALID_PARAM`；当前 arch 不支持的类型/规模返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；kernel/调度失败返回 `ACLBLAS_STATUS_RUNTIME_ERROR`。

目录与文件组织（与 isamax 对齐）：

```
blas/iamax/arch22/
├── icamax/                    # aclblas 句柄式 kernel 直调实现
│   ├── icamax_host.cpp        # 参数校验、tiling 计算、kernel 选择与 launch
│   ├── icamax_tiling.h        # tiling 结构体与 tilingkey 定义
│   ├── icamax_kernel.cpp      # Ascend C kernel（__aicore__ 入口）
│   └── icamax_kernel.h        # kernel 声明与核函数原型
├── icamax_broadcast / ...     # （预留，本期不交付）
└── CMakeLists.txt             # arch22 目标编译与用例挂接
```

## 算子实现

### 实现方案

总体思路：**"拆实虚 → 逐元素 1-范数幅值 → 带索引 max 归约 → 两级合并（核内 UB / 核间 scalar）"**，全链路只使用 arch22 可用的 AscendC 基础能力（`DataCopy`、向量算子、UB 标量读写、`membase` gather/标量访问、`SyncAll`），**不使用任何 AscendC SIMT / `asc_simt` 接口**。

计算流程（单核视角）：

1. **取数**：按 chunk 从 GM 搬 COMPLEX64 到 UB；`incx = 1` 走连续 `DataCopy`，`incx > 1` 走 membase gather（按元素间隔 `8 * incx` 字节取数）或标量循环填 UB。
2. **拆实虚**：将 UB 中交错的 `(re, im)` 拆成两条连续 float32 流 `re_buf` / `im_buf`。优先用向量 stride 搬运（VNCHW/stride copy 类能力）；若不便，则使用"按 8 字节步长 gather 成两条流"的等价写法。
3. **幅值计算**：对两条流分别 `vabs`，再 `vadd` 得到 `mag_buf`（float32，长度 = chunk 元素数）。
4. **块内 argmax**：在 `mag_buf` 上做 **reduce-max-with-index**：
   - 维护 `best_val`（float32 向量）与 `best_idx`（int32 向量，为全局 1-based 偏移 = `chunk_base * incx` 化元素序号 + i）；
   - 比较器为 `(val > best) || (val == best && idx < best_idx)`，用 `vcmp` 掩码 + `vsel` 逐轮合并（成对树形归约，log2 步内完成）；
   - 索引流随比较掩码同步 `vsel`，保证并列取小索引语义在向量化后仍然成立。
5. **双缓冲**：chunk `k` 计算时预取 chunk `k+1`，Ping-Pong 两个 UB 缓冲组，隐藏 GM↔UB 搬运延迟。
6. **核内收尾**：每核输出一对 `(local_best_val, local_best_idx)`，写入 GM 的中间结果 buffer（每核 8/16 字节，天然对齐）。
7. **核间合并**：采用两段式，避免每核返回 host：
   - 若 `n` 小（单核可容纳），直接由 0 号核写出最终结果；
   - 否则由 0 号核（或二次小 kernel）对 `blockNum` 个候选对做一次 **标量/小向量** reduce-max-with-index（比较器同上），写出 `*result = best_idx`，结果 +1 已在索引生成时固化为 1-based。

语义正确性要点：

- **1-based**：索引流初始化为 `chunk 元素起始序号（0-based）+ 行内偏移 + 1`。
- **并列取小**：比较器只在"严格大于"时更新；相等时 `vsel` 选择索引更小者。树形归约顺序不影响结果，因为比较器满足"最小索引优先"的全序。
- **`n <= 0`**：host 侧直接写 `*result = 0`，不 launch kernel。
- **数值稳定性**：只做 `abs` 与 `add`，无累加、无乘法，无溢出（float32 幅值最大为 `2 * FLT_MAX` 的加法结果，理论可到 inf，与参考实现行为一致）。

### 支持数据类型（实现角度的类型分支）

| 路径 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| 主路径 | COMPLEX64 (2×float32) | int32 | 唯一交付路径 |
| 预留 | COMPLEX32 / COMPLEX128 | int32 | 本期不实现，返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |

#### Host侧设计

##### 1. 分核策略

| 场景 | 核数（blockNum） | 每核工作量 | 说明 |
| --- | --- | --- | --- |
| `n <= singleCoreCap` | 1 | 全量 | 单核单趟，无核间归约；覆盖 n ≤ 约 4K~16K（视 UB 与 incx） |
| `n > singleCoreCap` | `min(maxCoreNum, ceil(n / elemsPerCore))` | 连续元素区间 | 元素级静态均匀切分，无负载不均（每元素代价相同） |
| 大 `incx`（gather 路径） | 同上，但 `elemsPerCore` 按 gather 有效带宽打折 | 连续元素区间 | 仍按元素序号切分，保证每核访存区间互不重叠 |

- 切分维度为 **元素序号（0-based i ∈ [0, n)）**，而不是字节/地址；每核负责 `[i0, i1)`，对应 GM 地址 `x + i * incx`。
- 每核元素数 `elemsPerCore = ceil(n / blockNum)`，最后一核收尾；`blockNum` 取 `min(物理核数, ceil(n / minElemsPerCore))`，`minElemsPerCore` 由 UB 容量与流水级数推出，避免小任务多核 launch 开销（n 很小时直接单核）。
- 0 号核额外承担"核间最终归约 + 写 `*result`"，其负担 O(blockNum)（典型 ≤ 48~64 项），不影响均衡。
- 无数据依赖，无核间同步；核间合并用一次性 GM 中间 buffer + 二次归约（或 0 号核在 `SyncAll` 后标量归约），避免全局同步多次。

##### 2. 数据分块和内存优化策略

| 策略 | 说明 |
| --- | --- |
| UB 分块 | 每核 chunk 元素数 `chunkElems = floor(ubCapacityBytes / (pingpong 组数 × 每元素 UB 开销))`；每元素 UB 开销 = 8B（原值）+ 4B(re) + 4B(im) + 4B(mag) + 4B(idx) ≈ 24B，实际以实虚拆分为两条流 + 幅值流 + 索引流共 4 条 float32/int32 流计 |
| Ping-Pong 双缓冲 | 两组输入缓冲交替：计算 chunk k 时预取 chunk k+1；幅值/索引流可复用同一对缓冲，进一步压缩 UB 占用 |
| 原值不驻留 | COMPLEX64 拆出 re/im 后原交错流可被覆盖复用，UB 峰值下降约 30% |
| 128B 对齐 | chunk 起始地址、UB buffer 均按 128B 对齐；`incx = 1` 时 `DataCopy` 走连续搬运补零对齐尾块 |
| gather 路径 | `incx > 1`：用 membase gather 按 `8*incx` 字节跨度取数；小 incx（2/3/4）可用"连续搬 + 向量 stride 抽取"；大 incx 直接标量/gather，避免搬入无用数据 |
| 带宽利用 | `incx = 1` 期望接近 GM 有效带宽上限；`incx > 1` 为访存稀疏场景，性能预期按有效带宽折算 |
| 索引流复用 | 索引为 chunk 内常量基址 + 行内偏移，可用 `vadds` 由序列生成，避免额外 GM 读 |
| 中间结果 buffer | `blockNum × 16B`（val + idx 各 8B 对齐），由 handle 侧临时 buffer 提供，生命周期与调用一致 |
| 溢出防护 | `n`、`incx`、`(n-1)*incx` 计算使用 int64 中间量做越界检查，避免 int32 溢出 |

##### 3. tilingkey规划策略

tilingkey 为 64-bit，按"高位类别 + 低位属性"分段，便于调度器与 kernel 选择一一对应，也便于扩展新类型：

| 位段 | 名称 | 取值 | 说明 |
| --- | --- | --- | --- |
| [63:56] | dataTypeKey | 0x01 = COMPLEX64 | 预留 COMPLEX32/128 |
| [55:48] | accessModeKey | 0x00 = contiguous（incx=1）<br>0x01 = stride-pow2（incx ∈ {2,4,8,...}）<br>0x02 = gather（通用 incx） | 决定取数 kernel 模板 |
| [47:40] | scaleKey | 0x00 = single-core<br>0x01 = multi-core-no-final<br>0x02 = multi-core-with-final-reduce | 决定是否需要核间合并 |
| [39:32] | pipelineKey | 0x00 = 单缓冲<br>0x01 = 双缓冲<br>0x02 = 四缓冲 | 按 n 与 UB 余量选择 |
| [31:0] | reserved | 0 | 预留（如 unroll、mask 对齐） |

本期实际下发的 key 组合（示例）：

| 场景 | tilingkey（十六进制示意） | kernel |
| --- | --- | --- |
| n 小、incx=1 | `0x01_00_00_01_00000000` | icamax_contig_single |
| n 大、incx=1 | `0x01_00_02_01_00000000` | icamax_contig_multi + final |
| n 大、incx=2/4/8 | `0x01_01_02_01_00000000` | icamax_stride_multi + final |
| n 大、通用 incx | `0x01_02_02_01_00000000` | icamax_gather_multi + final |

tiling 结构体（描述，非代码）：`IcamaxTiling { n; incx; blockNum; elemsPerCore; chunkElems; loopCount; tailElems; xDtype; outAddr; midAddr; blockIdxToElemBase; }`，由 host 计算后经 GM/`GetSysWorkSpace` 传给 kernel。

#### Kernel侧设计

kernel 入口与数据流（文字描述）：

1. **参数解析**：读取 tiling 结构体，计算本核元素区间 `[i0, i1) = [blockIdx * elemsPerCore, min(n, (blockIdx+1) * elemsPerCore))`，换算 GM 偏移 `byteOffset = i0 * incx * 8`。
2. **流水**：外层按 chunk 循环，内层 `pipe_barrier` 分级：
   - `copy_in`（GM→UB，独立 pipe）；
   - `split`（交错 → re/im 两条流，或 gather 直接生成两条流）；
   - `abs_add`（`vabs` + `vabs` + `vadd` → mag）；
   - `argmax`（`vcmp` 掩码 + `vsel` 树形归约，维护 (val, idx) 对）；
   - `write_back`（本核候选对写 GM 中间 buffer）。
3. **索引生成**：`idx = i0 + 1 + k`（1-based，k 为 chunk 内偏移），由向量序列 + `vadds` 生成，避免额外读；跨 chunk 时基址单调递增，保证全局序号正确。
4. **比较器实现**：`mask_gt = (mag > best_val)`；`mask_eq = (mag == best_val)`；`mask_idx = (idx < best_idx)`；更新掩码 `mask = mask_gt | (mask_eq & mask_idx)`；用 `vsel` 同步更新 `best_val` 与 `best_idx`。树形归约每轮将有效长度减半，直到标量。
5. **核间归约**：
   - 单核路径：本核候选对直接 `+1` 化后写出 `*result`；
   - 多核路径：各核写 `(val, idx)` 到 mid buffer，`SyncAll` 后由 0 号核以标量循环（或小向量 kernel）做同比较器归约，写出 `*result`。
6. **边界**：尾块不足 chunk 长度时用 mask 有效长度限制比较范围（或补 `-inf` 幅值 + 极大索引，使其永不胜出）；`n = 0` 由 host 直接处理。
7. **约束自检**：kernel 内不做 SIMT、不做 per-lane 任意地址写；所有 gather/标量访问走 membase，满足 arch22 编程模型。

## 支持硬件

| 硬件 | 架构 | 数据类型 COMPLEX64 → int | 状态 |
| --- | --- | --- | --- |
| Atlas 800I A2（910B） | arch22 | ✓ 支持 | 本期交付 |
| Atlas 800T A2（910B3） | arch22 | ✓ 支持 | 本期交付（性能测试设备） |
| Atlas A3 | arch22 | ✓ 支持 | 本期交付 |
| Atlas 800T A2 训练服务器整机 | arch22 | ✓ 支持 | 本期交付 |
| Atlas A5（950，arch35） | arch35 | 复用仓内既有 `arch35` 实现 | 不在本期范围 |
| CPU（x86/Arm） | - | 仅作为精度参考（cblas/Netlib） | 不作为交付目标 |

## 算子约束限制

| 编号 | 约束 | 说明 |
| --- | --- | --- |
| C1 | 数据类型仅支持 COMPLEX64（输入）/ int32（输出） | 其他类型返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| C2 | `n ≥ 0`，`n = 0` 时 `*result = 0` | 与 cblas 一致 |
| C3 | `incx ≥ 1` | 不支持 `incx = 0` 或负步长（Netlib 旧式负步长语义不实现），返回 `ACLBLAS_STATUS_INVALID_PARAM` |
| C4 | 索引为 1-based，范围 `[1, n]` | 与 Netlib `icamax` 一致，调用方若需 0-based 需自行减 1 |
| C5 | 并列最大值取最小索引 | 比较器严格大于才更新 |
| C6 | 模定义为 `|Re| + |Im|`（1-范数） | 不得替换为 2-范数/`hypot` |
| C7 | `x`、`result` 均为设备侧 GM 地址，`x` 8 字节对齐、`result` 4 字节对齐 | 不支持 host 指针直传 |
| C8 | `x` 与 `result` 不得地址重叠 | 未定义行为 |
| C9 | 输入含 NaN 时结果未定义，不做 NaN 传播语义保证 | 与 Netlib 参考实现一致 |
| C10 | 输入含 ±Inf 时正常参与比较 | `|+inf| + |im| = +inf` |
| C11 | 单次调用需满足 `(n-1) * incx * 8` 字节落在 `x` 可读范围内，内部用 int64 做越界校验 | 防止 int32 溢出 |
| C12 | 不依赖 AscendC SIMT / `asc_simt` | stride 用 membase gather / 标量访问实现 |
| C13 | kernel 直调模式下 stream 由 handle 提供，调用为异步 | `result` 读取前需同步（如 `aclrtSynchronizeStream`） |
| C14 | 单核 chunk 需 128B 对齐


<!-- design revision: CLA single-commit / design.md only -->
