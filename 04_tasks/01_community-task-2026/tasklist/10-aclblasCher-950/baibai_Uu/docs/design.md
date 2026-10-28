# aclblasCher 算子设计文档

时间：2026 年 10 月 28 日

作者：baibai_Uu

## 1 需求背景

### 1.1 需求来源

本项目通过昇腾 CANN 社区任务，为 ops-blas 新增单精度复数 Hermitian 秩-1 更新接口 `aclblasCher`。接口行为对齐 cuBLAS `cublasCher` / Netlib `cher`，工程采用 Ascend C SIMT kernel 直调模式，目标硬件为 Ascend 950PR（arch35），软件版本为 CANN 9.1.0。设计范围覆盖公共接口、Host 分发、Device kernel、测试与可维护性。

计算定义为：

```text
A := alpha * x * x^H + A
```

其中 `alpha` 为实数 `float`，`x` 与 `A` 为 `complex64`（列主序）。`x^H` 为共轭转置。仅 `uplo` 指定三角区域参与运算并原地更新；另一三角不引用也不更新；对角虚部更新后置 0。

### 1.2 背景介绍

#### 1.2.1 算子实现优化与参考路径

这是 BLAS 新增接口开发任务，非现有 TBE 算子的 aclnn 迁移。CheckList 中的 TBE 源码路径在本任务中不适用；审核目的由公共接口、同族工程与标杆语义分析承接。

| 参考对象 | 路径或文件 | 在本设计中的用途 |
| --- | --- | --- |
| 公共接口 | `ops-blas/include/cann_ops_blas.h` | 新增 `aclblasCher` 声明 |
| 公共类型 | `ops-blas/include/cann_ops_blas_common.h` | 复数类型、句柄、枚举和状态码 |
| 同族实现 | `ops-blas/blas/syr/arch35/` | 参考 `aclblasSsyr` / `aclblasCsyr` 工程接入与 SIMT 模式 |
| 同族测试 | `ops-blas/test/syr/ssyr/`、`test/syr/csyr/` | 参考 CSV + GTest 接入 |
| 本任务实现 | `ops-blas/blas/syr/arch35/cher_*` | Host / tiling / kernel |
| 本任务测试 | `ops-blas/test/syr/cher/` | CSV 1200、`cblas_cher` golden、arch35 ST |

#### 1.2.2 标杆算子现状分析

##### 数据类型与数据格式

`cublasCher` 的 `alpha` 为实数 float，`x` 与 `A` 为单精度复数；`A` 列主序，`uplo` 指定存储三角。`aclblasCher` 对应使用公共 `aclblasComplex`，实部/虚部各 `float32`。只对齐 Cher 所要求的类型与行为，不把 Ssyr/Csyr/Zher 等其它接口并入本设计。

##### 标杆语义实现描述

对合法非 no-op 输入，按行/列遍历有效三角元素，计算：

- 对角：`A_ii.re += alpha * (|x_i|^2)`，`A_ii.im = 0`
- 非对角：`A_ij += alpha * x_i * conj(x_j)`

参考实现以 Netlib/CBLAS `cblas_cher`（`CblasColMajor`）生成 golden；状态码、特殊值与 no-op 另行判定。

与 `aclblasCsyr` 的关键差异：Csyr 使用普通转置 `x^T` 且 `alpha` 为复数、对角虚部不置零；Cher 使用共轭转置 `x^H`、`alpha` 为实数、对角虚部强制为 0。

## 2 需求分析

### 2.1 外部组件依赖

| 组件 | 要求及职责 | 适配验证方式 |
| --- | --- | --- |
| CANN 与 Ascend C | CANN 9.1.0；编译与运行 950PR SIMT kernel | 记录安装包版本，编译并运行 ST |
| AscendCL Runtime | 设备、内存、stream、同步 | 非默认 stream 跑通 |
| ops-blas | 公共 BLAS 接口、handle、构建与导出 | 固定分支提交后核对头文件与同族模式 |
| C++ 测试与 GTest | CSV 参数化 + `cblas_cher` golden | `build.sh --ops=cher --soc=ascend950 --run` |
| cuBLAS / Netlib 文档 | `cublasCher` / `cher` 语义参考 | 文档依据；golden 用 CBLAS |

### 2.2 内部适配模块

Host 入口复用 ops-blas 句柄与 stream，禁止新建 950PR 私有平行 API。内部划分为：参数校验、tiling 选择、kernel 启动。Device 划分为：逻辑索引、GM/UB 路径选择、Hermitian 复数更新、对角虚部置零。测试模块包括 CSV 适配、独立 golden、实/虚分量 mixed tolerance、非 uplo 不变检查。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

| 参数 | 位置和类型 | 约束与语义 |
| --- | --- | --- |
| handle | Host 句柄 | 有效，携带调用 stream |
| uplo | Host 枚举 | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host int | `n >= 0` |
| alpha | Host `const float*` | 非空；`0.0` 合法 no-op |
| x | Device 复数向量 | `n>0` 且 alpha 非零时非空 |
| incx | Host int | 非零；支持负步长；禁止 `INT_MIN` |
| A | Device 复数矩阵 | 原地更新，列主序，`lda × n` |
| lda | Host int | `lda >= max(1, n)` |

#### 2.3.2 约束与差异

- 仅 COMPLEX64；不支持 batch / packed / complex128
- `n==0` 或 `alpha==0`：合法 quick return，不更新 A
- 非法 `uplo` → `ACLBLAS_STATUS_INVALID_ENUM`
- 不写回镜像三角以“补全 Hermitian”
- 负步长按 Netlib 语义反向索引；`incx==INT_MIN` 直接 `INVALID_VALUE`，避免 `abs` 溢出

## 3 详细设计

### 3.1 调用方式与参数处理

调用方创建公共 handle、绑定 stream，准备 `alpha/x/A` 后调用 `aclblasCher`；读取 Device 结果前同步同一 stream。

| 顺序 | 条件 | 处理结果 |
| --- | --- | --- |
| 1 | handle 为空 | `HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `INVALID_VALUE` |
| 3 | `n == 0` | `SUCCESS`（不读其余指针） |
| 4 | uplo 非法 | `INVALID_ENUM` |
| 5 | `lda` / `incx` 非法（含 0、`INT_MIN`） | `INVALID_VALUE` |
| 6 | `alpha == nullptr` | `INVALID_VALUE` |
| 7 | `*alpha == 0` | `SUCCESS`，不访问 x/A |
| 8 | x/A 为空 | `INVALID_VALUE` |
| 9 | 正常路径 | tiling + 启动 kernel |

本任务书约定 `alpha` 为 Host 指针；入口复制标量后入 tiling，不在 Device 侧再读 alpha。

### 3.2 Host 侧设计

#### 分核策略

- `aivCoreNum = GetAivCoreCount()`
- `numBlocks = min(ceil(n / SIMT_MIN_THREAD_NUM), aivCoreNum)`
- `numThreads = clamp(对齐后的每 block 行数需求, SIMT_MIN_THREAD_NUM, SIMT_MAX_THREAD_NUM)`
- `rowsPerBlock = ceil(n / numBlocks)`

小 `n` 自动减少 block，避免空核；大 `n` 吃满 AIV 核。

#### Tiling 字段

| 字段 | 含义 |
| --- | --- |
| `n` / `lda` / `uplo` / `alpha` / `incx` | 问题规模与参数 |
| `numThreads` | 每 block SIMT 线程数 |
| `rowsPerBlock` | 每 block 负责的行跨度 |

### 3.3 Kernel 侧设计

编程模型：Ascend C SIMT（`__simt_vf__` + `asc_vf_call`），`KERNEL_TYPE_AIV_ONLY`。

#### 路径选择

1. **GM 通用路径**（`incx != 1` 或 `n < 32`，或所需 x 窗口超出 UB）：每线程按行 `row` 扫描 uplo 内列，直接读写 GM。
2. **UB 缓存路径**（`incx == 1` 且本 block 所需 x 段长度 `<= UB_COMPLEX(4096)`）：先将相关 x 段装入 UB，再对负责行区间做更新，降低重复读 x。

UPPER：本 block 行 `[rowStart, rowEnd)` 需要的 x 为 `[rowStart, n)`。  
LOWER：需要的 x 为 `[0, rowEnd)`。

#### 计算内核

对每个 `(row, col)`（在 uplo 内）：

```text
xr, xi = x[row];  yr, yi = x[col]
if row == col:
    A.re += alpha * (xr*xr + xi*xi)
    A.im  = 0
else:
    A.re += alpha * (xr*yr + xi*yi)
    A.im += alpha * (xi*yr - xr*yi)   # x_row * conj(x_col)
```

负 `incx` 仅在 GM 路径通过索引映射支持。

### 3.4 精度与测试设计

- Golden：`cblas_cher`
- 比对：实部/虚部分别按 opbase **mixed tolerance**（matched_ratio + max_abs_error）
- Inf/NaN：按任务书视为等价（同为 NaN、同号 Inf、或 NaN vs Inf）
- 对角 imag ≈ 0；非 uplo / lda padding 不变（NaN-aware 比较）
- 覆盖：尺寸扫描、alpha 特殊值、步长、padding、fill、负向、Inf/NaN、性能 CSV（`TC_PF_*`）

自测命令：

```bash
bash build.sh --ops=cher --soc=ascend950 --run --device=0
```

预期：`[  PASSED  ] 1201 tests.`（1200 CSV + NullHandle）

### 3.5 目录与产品支持

| 项 | 内容 |
| --- | --- |
| 实现 | `blas/syr/arch35/cher_{host,kernel,tiling_data}.*` |
| 测试 | `test/syr/cher/`（含 README） |
| README | Ascend 950PR：**支持**；A2/A3：**不支持** |
| 私仓分支 | `baibai_Uu/ops-blas` / `feature/aclblas-cher-arch35` |

## 4 风险与后续

1. 正式 msprof op 性能表（warmup 5 + 有效 10 次）可在验收材料中补齐；功能 ST 中 `TC_PF_*` 已跑通。
2. 代码合入 PR 待设计评审通过后再提交主仓。
3. 设计文档合入路径：`04_tasks/01_community-task-2026/tasklist/10-aclblasCher-950/baibai_Uu/docs/design.md`。
