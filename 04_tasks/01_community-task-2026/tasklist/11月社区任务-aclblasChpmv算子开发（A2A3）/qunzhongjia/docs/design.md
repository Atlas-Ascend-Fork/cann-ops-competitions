# aclblasChpmv 算子设计文档

## 一、需求背景

### 1.1 需求来源

本需求来源于“11月社区任务-aclblasChpmv算子开发（A2/A3）”。开发者为 qunzhongjia，目标是在 Atlas A2/A3 系列产品、CANN 9.1.0 环境下，基于 ops-blas 工程新增单精度复数 Hermitian 压缩存储矩阵与向量乘法接口 `aclblasChpmv` 的 `arch22` 实现。

### 1.2 背景介绍

Chpmv 属于 BLAS Level-2 算子，计算 $y \leftarrow \alpha A x + \beta y$。其中 $A$ 为 $n \times n$ 单精度复数 Hermitian（厄米特）矩阵，满足共轭对称关系 $A = A^H$ 且主对角线元素虚部为零；$x$、$y$ 为逻辑长度 $n$ 的复数向量；$\alpha$、$\beta$ 为复数标量。为降低设备显存占用，$A$ 按列主序仅将 `uplo` 指定的上三角或下三角紧密压缩存储于一维连续数组 `AP`（共 $n(n+1)/2$ 个复数元素），另一侧三角元素由共轭对称关系隐含恢复，计算结果原地写入 `y`。

仓库 `blas/spmv/arch22/` 已有实数对称压缩矩阵向量乘 `aclblasSspmv`，可沿用其句柄绑定 stream 下发、步长跨度检查及目录组织方式。本算子面向 Atlas A2/A3 纯矢量（AIV SIMD）硬件架构，处理复数实虚交错排布、Hermitian 共轭对称归约、实对角提取及三角变长列多核负载均衡，复用仓库公共头文件、构建脚本与 GTest 测试框架。

## 二、需求分析

### 2.1 接口定义

公共声明位于 `include/cann_ops_blas.h`，参数顺序及类型与 `cublasChpmv` 一致：

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* beta,
    aclblasComplex* y,
    int incy);
```

| 参数 | 位置与含义 | 约束 |
| --- | --- | --- |
| `handle` | Host，携带执行 stream 与工作区的上下文句柄 | 指向已创建的有效句柄，为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | Host，压缩矩阵存储三角模式 | 取值 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | Host，Hermitian 矩阵阶数及向量逻辑长度 | `n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，`n == 0` 直接返回 `ACLBLAS_STATUS_SUCCESS` |
| `alpha`、`beta` | Host 或 Device，复数标量乘数指针 | 非空；支持 Host 与 Device 两种指针位置，否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `AP` | Device，只读列主序压缩三角矩阵 | `complex64`，物理元素数至少为 $n(n+1)/2$，`n > 0` 时非空，对角虚部忽略 |
| `x` | Device，只读输入复数向量 | `complex64`，物理元素数至少为 $1 + (n-1)\lvert\text{incx}\rvert$，`n > 0` 时非空 |
| `incx` | Host，向量 `x` 的逻辑步长 | `incx != 0`，支持正负值；`n > 1` 时跨度不得溢出 |
| `y` | Device，原地更新的输出复数向量 | `complex64`，物理元素数至少为 $1 + (n-1)\lvert\text{incy}\rvert$，`n > 0` 时非空 |
| `incy` | Host，向量 `y` 的逻辑步长 | `incy != 0`，支持正负值；`n > 1` 时跨度不得溢出 |

复数类型使用仓库 `aclblasComplex`，实部 `real` 与虚部 `imag` 各占一个 `float32`（单元素 8 字节）。Device 输入与输出缓冲区在关联 `aclrtStream` 执行完成前保持有效，调用方同步 stream 后读取结果。

### 2.2 功能要求

1. 支持 `ACLBLAS_UPPER` 与 `ACLBLAS_LOWER` 两种列主序三角压缩存储，未存储侧按共轭对称 $A_{j,i} = \overline{A_{i,j}}$ 参与运算。
2. 主对角元素仅读取实部分量参与乘加，忽略对角虚部内存槽位，避免对角虚部含 `NaN` 或 `Inf` 时污染结果。
3. 支持正负非零步长 `incx`、`incy`，保证 `AP`、`x` 只读不变，且 `y` 在非单位步长下的间隙元素保持原值。
4. 支持 `n == 0` 及 `alpha == (0,0) && beta == (1,0)` 的快速返回；支持 `alpha == (0,0)` 时跳过 `AP`、`x` 读取仅缩放 `y`，以及 `beta == (0,0)` 时跳过旧 `y` 读取直接覆写输出。
5. 通过任务全部精度与性能用例，提供测试工程、复现步骤、自验证报告及原始日志。

### 2.3 支持范围

目标硬件为 Atlas A2 训练/推理系列产品（含 Atlas 800T A2）与 Atlas A3 训练/推理系列产品（含 Atlas 800T A3），对应架构目录 `arch22`，软件环境为 CANN 9.1.0。支持 `complex64` 数据类型、运行时动态阶数 $n \ge 0$ 及 `incx`、`incy` 描述的正负步长访问。算子不涉及张量广播与批处理；精度按 `FLOAT32` 混合容差标准独立校验实部与虚部。

## 三、详细设计

### 3.1 算子分析

记未乘 $\alpha$ 的中间向量为 $w = A x$。以下索引均从 0 开始，以复数元素为单位（实数分量下标乘以 2）：

| 存储模式 | 列 $j$ 起始偏移 $\text{base}(j)$ | 对角元素偏移 | 列内非对角区间与贡献 |
| --- | --- | --- | --- |
| `ACLBLAS_UPPER` | $j(j+1)/2$ | $\text{base}(j) + j$ | $i \in [0, j)$：$a_{i,j} = \text{AP}[\text{base}(j)+i]$，更新 $w_i \mathrel{+}= a_{i,j} x_j$ 与 $w_j \mathrel{+}= \overline{a_{i,j}} x_i$ |
| `ACLBLAS_LOWER` | $j(2n - j + 1)/2$ | $\text{base}(j)$ | $i \in (j, n)$：$a_{i,j} = \text{AP}[\text{base}(j)+i-j]$，更新 $w_i \mathrel{+}= a_{i,j} x_j$ 与 $w_j \mathrel{+}= \overline{a_{i,j}} x_i$ |

对角项仅读取实部 $d_j = \operatorname{Re}(A_{j,j})$，贡献 $w_j \mathrel{+}= (d_j x_{j,r},\; d_j x_{j,i})$。最终输出按 $\beta$ 是否为零分为 $y_i \leftarrow \alpha w_i$（$\beta = 0$）与 $y_i \leftarrow \alpha w_i + \beta y_i$（$\beta \neq 0$）。普通复乘展开为 $(a_r b_r - a_i b_i,\; a_r b_i + a_i b_r)$，共轭复乘展开为 $(a_r b_r + a_i b_i,\; a_r b_i - a_i b_r)$。

对于逻辑下标 $k \in [0, n)$ 与步长 $\text{inc} \neq 0$，物理复数偏移在 64 位整数下计算：当 $\text{inc} > 0$ 时为 $k \cdot \text{inc}$；当 $\text{inc} < 0$ 时逻辑起点位于 $(n - 1) \cdot (-\text{inc})$，第 $k$ 个元素偏移为 $(n - 1 - k) \cdot (-\text{inc})$。

压缩矩阵 `AP` 共含 $n(n+1)/2$ 个复数（约 $4n^2$ 字节），算术复杂度为 $O(n^2)$，算术强度约 1 FLOP/Byte，属于典型访存带宽受限算子。通过对 `AP` 每列非对角切片单次流式搬入、同时完成前向列乘加（AXPY）与共轭内积归约（DOT），可将 `AP` 的全局内存读取量压缩至理论下限 $4n(n+1)$ 字节。

### 3.2 Host 侧设计

#### 3.2.1 参数校验

Host 实现位于 `blas/spmv/arch22/chpmv_host.cpp`，校验与返回顺序如下：

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER` | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 4 | `incx == 0`、`incy == 0`，或 `n > 1` 时步长为 `INT32_MIN` 及 $(n-1)\lvert\text{inc}\rvert$ 跨度溢出 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `alpha`、`beta` 任一为空指针 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `n == 0` | 返回 `ACLBLAS_STATUS_SUCCESS`，不下发 Kernel |
| 7 | `AP`、`x`、`y` 任一为空指针 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 通过 `CheckPtrLocation` 探测 `alpha`/`beta` 内存位置并读取标量（位于 Device 时异步拷贝至 Host 并同步） | 查询或拷贝失败返回对应错误状态 |
| 9 | `alpha == (0,0)` 且 `beta == (1,0)` | 返回 `ACLBLAS_STATUS_SUCCESS`，不下发 Kernel |
| 10 | 平台 AIV 核数或 UB 容量查询为 0 | 返回 `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 11 | 其余合法输入 | 构造 Tiling，绑定 `handle` 的 stream 异步下发 Kernel 并检查发射错误 |

当 $n == 1$ 时 $(n - 1) = 0$，任意非零 `int` 步长（含 `INT32_MIN` 与 `INT32_MAX`）的物理偏移均为 0；当 $n > 1$ 时采用 `uint64_t` 校验物理字节跨度不越界。每次核函数下发前清空线程级残留错误，下发后调用 `aclrtGetLastError(ACL_RT_THREAD_LEVEL)` 检查发射状态。

#### 3.2.2 分核与模板选择

Host 通过 `PlatformAscendCManager` 动态查询当前设备的 AIV 核心数 `aivCoreNum` 与单核 UB 容量 `ubBytes`，按标量值、矩阵阶数和片上容量选择执行路径：

| 条件 | 分核与模式配置 | 处理方式 |
| --- | --- | --- |
| `alpha == (0,0)` 且 `beta != (1,0)` | `reduceBlocks = min(CeilDiv(n, 1024), aivCoreNum)` | 下发 `chpmv_scale_arch22_kernel`，仅计算 $y \leftarrow \beta y$ |
| `alpha != (0,0)`、`n <= 128` | `accumBlocks = min(n, aivCoreNum), directOut = 1` | 多核片上直出，单 Kernel 内各核独立加载并行计算所辖行并融合 $\alpha/\beta$ 写回，零 Workspace 开销 |
| `alpha != (0,0)`、`n > 128` 且 $4 \cdot \text{CeilAlign}(n,64) \cdot 4\text{B} + \text{固定缓冲} \le \text{UB}$ | `accumBlocks <= aivCoreNum, fullUb = 1, directOut = 0` | 全向量片上驻留两阶段路径：阶段 1 单遍读取 `AP` 写出各核部分和，阶段 2 多核规约写回 `y` |
| `alpha != (0,0)` 且超出全向量 UB 驻留预算 | `accumBlocks <= aivCoreNum, fullUb = 0, tileRows = 1024` | 行分块多遍路径：按 `1024` 行分块流式累加至 Workspace，再由阶段 2 规约写回 `y` |

在 192 KiB UB 容量下，全向量驻留模式覆盖单位步长 $n \le 8192$ 及非单位步长 $n \le 6144$。多核并行时，阶段 1 核数随规模自适应递增（$16 < n \le 64$ 时按 $\lceil n/8 \rceil$ 扩展，$64 < n \le 128$ 时上限 16 核，$n > 128$ 时扩展至满 AIV 核），并受可用 Workspace 容量约束；若使用库默认工作区且当前容量不足 `accumBlocks * 2 * CeilAlign(n, 64) * sizeof(float)`，Host 调用 `EnsureDefaultWorkspace` 自动扩容。

阶段 1 对 `accumBlocks` 个 AIV 核采用蛇形折返（Boustrophedon）列映射：第 $b$ 核在第 $\text{ord}$ 轮若 $\text{ord}$ 为偶数则处理列 $\text{ord} \cdot \text{accumBlocks} + b$，若 $\text{ord}$ 为奇数则处理列 $\text{ord} \cdot \text{accumBlocks} + (\text{accumBlocks} - 1 - b)$。相邻两轮列长之和恒定，消除三角矩阵长短列引起的核间负载倾斜。阶段 2 将 $n$ 行输出按商和余数均分至 `reduceBlocks` 个核独立写回。

Host 与 Kernel 之间通过 POD 结构体按值传递参数：

```cpp
struct ChpmvTilingData {
    uint32_t n;
    uint32_t nPad;
    uint32_t uplo;
    uint32_t accumBlocks;
    uint32_t reduceBlocks;
    uint32_t tileRows;
    uint32_t fullUb;
    uint32_t directOut;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    int64_t incx;
    int64_t incy;
};
```

### 3.3 Kernel 侧设计

Device 代码位于 `blas/spmv/arch22/chpmv_kernel.cpp`，设定 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，`TPipe` 在核函数入口局部变量定义并按指针传入算子类。

1. **硬件奇偶解交织与打包**：
   - 全局内存中的 `aclblasComplex` 为实虚交错排布（AoS）。搬入 UB 后，调用硬件 `GatherMask` 指令（内置模式 `1` 提取偶数位实部，模式 `2` 提取奇数位虚部，单周期吞吐 64 个 `float`）拆分为连续 `Re` 与 `Im` 平面（SoA），后续乘加全部采用满宽度连续 FP32 矢量指令。
   - 写回 `y` 前，利用预生成的奇偶交错索引向量通过单次 `Gather` 将 `Re`、`Im` 重新拼装为 AoS 复数序列，再经 `DataCopyPad` 写出。
2. **阶段 1 部分和累积（`ChpmvAccumAIV`）**：
   - **全向量驻留路径（`fullUb == 1`）**：`xRe[nCap]`、`xIm[nCap]`、`accRe[nCap]`、`accIm[nCap]` 常驻 UB（$n = 8192$ 时占 128 KiB，其余列块缓冲与归约临时缓冲共约 58.5 KiB，总计 186.5 KiB < 192 KiB）。核函数启动时整段加载 `x` 并清零片上累加器；遍历所属蛇形列 $j$ 时，先通过 MTE2 异步发起对角实部 `apGm_[2 * diagIdx]` 的 4 字节读取（跳过对角虚部），再以 `1024` 复数为块长（首块对齐至 8-float 边界使后续满块 32 字节对齐）流式搬入该列非对角元素，片上同时完成直侧矢量 AXPY 累加与共轭逐元素积矢量树累加，每列仅在末尾执行一次水平 `ReduceSum` 并合并对角实数乘积；全部列扫描完成后，若 `directOut == 1` 直接在片上乘 $\alpha$ 并融合 $\beta y$ 写回 `yGm_`，否则以两次连续突发 `DataCopyPad` 将 `n` 个实部和虚部写出至 `wsGm_[blockIdx * 2 * nPad]`。
   - **行分块多遍路径（`fullUb == 0`）**：针对 $n > 8192$ 场景，先矢量清零当前核 Workspace 切片，再按 `tileRows = 1024` 遍历行块，仅扫描与当前行块存在交集的列（`UPPER` 满足 $j \ge \text{tileStart}$，`LOWER` 满足 $j < \text{tileEnd}$），块内矢量累加后写回 Workspace。
3. **阶段 2 多核规约与缩放写回（`ChpmvReduceAIV`）**：
   - 各核对所辖行段按 `1024` 长度切块，循环读取 `accumBlocks` 个核的 Workspace 平面执行矢量 `Add` 得到 $w$，计算复数乘法 $\alpha w$。
   - 当 `beta == (0,0)` 时不读取 `yGm_`，直接打包写出 $\alpha w$；当 `beta != (0,0)` 时读取旧 `y` 计算 $\alpha w + \beta y$ 后写回。`alpha == (0,0)` 缩放入口复用同一写回管线。
   - 非单位步长读取 `x`/`y` 时，若片段物理跨度不超过 `4096` 复数则单次 `DataCopyPad` 搬入连续窗口后按步长抽取；非单位步长写回 `y` 时逐逻辑位置发起 8 字节 `DataCopyPad`，保持间隙内存不变。
4. **流水同步**：跨流水数据传输全部采用 `FetchEventID` 配对 `SetFlag`/`WaitFlag`（覆盖 `MTE2_V`、`V_MTE2`、`V_MTE3`、`MTE3_V`、`MTE2_S`、`S_V`、`V_S`、`S_MTE3`、`MTE3_MTE2`），同流水矢量寄存器依赖插入 `PipeBarrier<PIPE_V>()`。

### 3.4 性能设计

性能测试集共 200 条用例，阶数覆盖 $n \in [1, 8192]$，包含 `UPPER`/`LOWER`、单位步长及正负非单位步长场景。小规模场景耗时主要来自核函数启动与多核同步，中大规模场景受限于 HBM 访存带宽与片上解交织吞吐。

针对各规模瓶颈采取以下措施：

1. **小规模单核直出（`n <= 16`）**：省去 Workspace 写读及第二阶段 Kernel 启动开销，单次调用压缩为单 Kernel 执行。
2. **中大规模单遍全驻留（`n <= 8192`）**：向量 `x` 与累加器全量常驻 UB，`AP` 全程仅从 HBM 顺序读取一次，消除跨行块重复加载与全局内存读改写开销。
3. **指令级吞吐优化**：以单拍 64 floats 的硬件 `GatherMask` 替代索引 `Gather` 完成复数奇偶分离；跨块大列先做片上矢量加法树累加，每列仅执行一次 `ReduceSum`；对角实部采用异步 MTE2 读取与矢量计算重叠。
4. **分派边界验证**：在 Atlas A2 真机上对小规模阈值两侧（$n = 15/16/17/32$）、全驻留上限（$n = 8192$）及回退分块路径（$n > 8192$）分别验证功能与耗时，确保各分支衔接平滑。

### 3.5 工程集成

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | `aclblasChpmv` 公共接口声明（沿用仓内既有声明） |
| `blas/spmv/arch22/chpmv_tiling_data.h` | Tiling 结构体与分块常量定义 |
| `blas/spmv/arch22/chpmv_kernel.h` | 三个 Kernel 启动包装函数声明 |
| `blas/spmv/arch22/chpmv_host.cpp` | 参数校验、Workspace 管理、自适应分核与流式下发 |
| `blas/spmv/arch22/chpmv_kernel.cpp` | AIV 矢量计算、蛇形分列累加、多核规约与步长写回 |
| `blas/spmv/README.md` | 更新 `aclblasChpmv` 产品支持表及构建自测命令说明 |
| `test/spmv/chpmv/CMakeLists.txt` | 在 `arch22` 分支通过 `ops_blas_add_gtest_tests` 注册 `chpmv_test` |
| `test/spmv/chpmv/arch22/` | NPU 包装器、GTest 测试源码、用例 CSV 及性能基线比对 |

工程通过 `bash build.sh --soc=ascend910b2 --ops=chpmv`（或对应 A2/A3 SoC 型号）统一构建算子动态库 `libops_blas.so` 与独立/聚合测试二进制。全部新增源码遵循 CANN Open Software License Agreement Version 2.0 许可头、`#pragma once` 头文件规范与仓库 `.clang-format` 格式要求。

## 四、可维可测分析

### 4.1 精度验证

采用 `ops-blas` 仓 CSV 驱动的 C++ GTest 框架，通过公开接口 `aclblasChpmv` 执行；黄金标杆使用 Netlib BLAS 3.10.0 `cblas_chpmv`（`CblasColMajor`，调用前在副本上将对角元素虚部置 0 以对齐 Hermitian 语义）。覆盖全部官方精度与边界用例，并对性能形状同步校验数值结果。

输出向量 `y` 的实部与虚部分别按生态算子开源 `FLOAT32` 混合容差标准判定：

- `atol = 2^-13`、`rtol = 2^-13`，逐元素满足 $\lvert\text{actual} - \text{golden}\rvert \le \text{atol} + \text{rtol} \cdot \lvert\text{golden}\rvert$；
- 有限值元素匹配比例 `matched_ratio >= 0.99`，最大绝对误差 `max_abs_error <= max(1e-2, 32 * ULP)`；
- 非有限值元素单独统计：同为 `NaN`、同为同号 `Inf` 或 `NaN` 与 `±Inf` 按任务规则判定通过，一方有限而另一方非有限或异号 `Inf` 判定不通过。

测试数据使用固定随机种子独立采样实部与虚部，同时覆盖均匀分布 `RANDOM_NORM_5_5` 与正态分布 `RANDOM_GAUSS_0_1`（$N(0, 1)$），以及上下三角、零维与单元素、质数及非对齐尺寸、特殊标量、正负步长、对角虚部 `NaN`/`Inf` 注入、`beta = (0,0)` 且旧 `y` 含 `NaN`、次正规数（Subnormal）极值、参数校验优先级、`AP`/`x` 前后哨兵字（`0x4a16b33d`）只读检查与 `y` 步长间隙不变性检查。

### 4.2 性能验证

在 Atlas A2、CANN 9.1.0 环境下执行全部 200 条 `TC_PF` 性能用例。采用 `msprof op` 采集单算子 `Task Duration(us)`，预热 5 次不计入样本，其后连续执行 10 次有效采样取算术平均作为 NPU 平均单次耗时；同时使用 `aclrtEvent` 在设备缓冲就绪后连续下发计时作为交叉验证，排除 Host 造数、H2D/D2H 拷贝与 CPU Golden 计算开销，每次调用前恢复原始 `y`。

与 `gpu_baseline.csv` 中的 `cublasChpmv` 基线按参数维度一一关联（将 `gpu_ms` 换算为 `us`），逐例核对满足任务规定的性能倍率要求（平均单次耗时不高于 GPU 标杆耗时的 1.25 倍，即达到标杆 0.8 倍以上性能），完整保留原始日志与逐例对比表。

### 4.3 兼容性与维护

新增实现位于 `blas/spmv/arch22/` 与 `test/spmv/chpmv/arch22/` 架构独立目录，复用公共头文件既有声明、状态码及句柄机制，不影响 `arch35` 的 `chpmv` 实现与同目录的 `sspmv` 实现。后续若调整压缩偏移或步长寻址需回归上下三角及正负步长用例；调整分核或片上缓冲阈值需回归阈值邻域尺寸与大规模分块用例；调整浮点复乘顺序需执行全量精度回归。

### 4.4 交付内容

按社区任务规范提交算子设计文档 PR、完整算子实现与测试代码、`blas/spmv/README.md` 支持说明，以及 `task_submission/` 标准自验证材料（自验证步骤说明、精度自验证报告与原始日志、性能自验证报告与原始日志、内存自验证报告与说明日志），并在个人私有代码仓邀请官方账号开放验收访问。

## 五、参考资料

1. 任务来源：[ops-blas 社区任务讨论 #86](https://gitcode.com/cann/ops-blas/discussions/86) 及任务测试附件（`chpmv_test.csv`、`gpu_baseline.csv`）。
2. [CANN 社区任务算子设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
3. [ops-blas 开源算子仓](https://gitcode.com/cann/ops-blas)：`blas/spmv/`、公共接口与测试框架。
4. [Netlib BLAS chpmv 参考实现](https://www.netlib.org/blas/chpmv.f)：列主序三角压缩存储、共轭对称与正负步长语义。
5. [cuBLAS cublasChpmv 官方文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpmv)：接口定义与参数契约。
6. [CANN 生态算子开源混合容差精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
