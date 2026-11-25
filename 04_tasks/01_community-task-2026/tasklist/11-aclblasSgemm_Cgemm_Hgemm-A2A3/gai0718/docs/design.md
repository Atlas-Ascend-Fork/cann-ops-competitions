# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 队名 | `gai0718` |
| 任务 | 11 月社区任务：gemm 类 3 个算子开发（A2/A3） |
| 目标仓库 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 设计基线 | `ops-blas@55ef791` |
| 目标架构 | arch22（Atlas A2/A3，910B3 性能设备） |
| 软件环境 | CANN 9.1.0，Ascend C |

本文按社区 `design_template.md` 组织，说明三个 GEMM 接口的功能、现有能力、Host/Kernel 方案和验收方法。

## 需求背景（required）

### 需求来源

任务书要求基于 `ops-blas` 的句柄式 BLAS 工程，在 `blas/gemm/arch22/` 使用 Ascend C kernel 直调实现：

- `aclblasSgemm`：FLOAT32 单精度实数矩阵乘；
- `aclblasCgemm`：COMPLEX64 单精度复数矩阵乘；
- `aclblasHgemm`：FLOAT16 半精度实数矩阵乘。

接口语义分别对齐 cuBLAS `cublasSgemm`、`cublasCgemm` 和 `cublasHgemm`。测试代码进入现有 `test/gemm/arch22/`，算法文档进入 `blas/gemm/README.md`。

### 背景介绍

三个接口统一计算列主序矩阵乘：

\[
C\leftarrow\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C
\]

其中 \(\operatorname{op}(A)\) 为 \(m\times k\)，\(\operatorname{op}(B)\) 为 \(k\times n\)，\(C\) 为 \(m\times n\)。`transa/transb` 支持 `ACLBLAS_OP_N/T/C`；对复数数据，`T` 为非共轭转置，`C` 为共轭转置；对 S/H 实数数据，`T` 与 `C` 等价。

### 现有能力分析

| 接口 | 当前 arch22 状态 | 本任务方案 |
| --- | --- | --- |
| `aclblasSgemm` | 已有 Host、Matmul Cube、pack、epilogue 和 quick-return 路径 | 保留并验证现有实现，补齐三个 dtype 的共用测试回归 |
| `aclblasCgemm` | 已有复数拆分、四个实数乘积和复数组合路径 | 保留现有实现，验证 `OP_T`/`OP_C` 差异、参数校验和全部边界行为 |
| `aclblasHgemm` | 公共声明和 arch22 实现均缺失 | 新增公共声明、Host 入口和 FP16 Cube Kernel |

公共头文件已有 `aclblasHalf` 的 16-bit 位模式类型和 S/C 接口声明。`aclblasHgemm` 需紧随同族 GEMM 接口新增，保持各产品线共用同一 API。

## 需求分析（required）

### 需求描述

在 arch22 上实现并交付三个异步句柄式 GEMM 接口，覆盖列主序前导维、九种转置组合、一般 `alpha/beta`、矩阵边界和错误参数。Kernel 使用 handle 绑定的 stream，接口返回不代表设备计算完成。

### 公共函数原型

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb,
    const aclblasHalf* beta, aclblasHalf* C, int ldc);
```

### 参数与布局

| 参数 | 类型与位置 | 约束 |
| --- | --- | --- |
| `handle` | Host 句柄 | 必须有效，携带计算 stream |
| `transa/transb` | 枚举属性 | N/T/C，其他值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m/n/k` | 运行时整数 | 非负 |
| `alpha/beta` | S/C/H 对应的 Host 标量指针 | 必须非空 |
| `A/B` | 对应 dtype 的 Device 只读矩阵 | 需要参与乘法时不可为空 |
| `C` | 对应 dtype 的 Device 原地输出 | `m>0` 且 `n>0` 时不可为空 |
| `lda/ldb/ldc` | 前导维 | 按转置模式和列主序物理形状校验 |

物理地址使用：

\[
A_{r,c}=A[r+c\cdot lda],\qquad
B_{r,c}=B[r+c\cdot ldb],\qquad
C_{r,c}=C[r+c\cdot ldc].
\]

`transa=N` 时 \(lda\geq\max(1,m)\)，否则 \(lda\geq\max(1,k)\)；`transb=N` 时 \(ldb\geq\max(1,k)\)，否则 \(ldb\geq\max(1,n)\)；\(ldc\geq\max(1,m)\)。Kernel 始终使用实际 `lda/ldb/ldc` 访问数据，仅写 C 的逻辑 \(m\times n\) 区域。

### 校验与边界行为

Host 参数校验顺序为：handle、转置枚举、维度、前导维、标量指针、矩阵指针。结构合法后执行 quick return：

| 条件 | 行为 |
| --- | --- |
| `m=0` 或 `n=0` | 返回 `SUCCESS`，不访问 A/B/C，不启动 Kernel |
| `k=0` 或 `alpha=0` | 不读取 A/B，仅计算 `C=beta*C` |
| `beta=0` | 不读取 C 旧值 |
| `beta=1` 且无乘加 | C 保持不变，不启动 Kernel |
| 非空输出且 C 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 需要乘加且 A/B 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |

### 数据类型与计算精度

| 接口 | A/B/C 与标量类型 | 累加与输出 |
| --- | --- | --- |
| Sgemm | FLOAT32 | FP32 乘加，FP32 输出 |
| Cgemm | COMPLEX64 | 实部/虚部各 FP32；直接计算复数乘加，不进入 `aclblasCgemm3m` 路径 |
| Hgemm | FLOAT16 | FP16 输入，FP32 累加，最终舍入到 FP16 |

复数主计算直接使用：

\[
(a_r+ia_i)(b_r+ib_i)=(a_rb_r-a_ib_i)+i(a_rb_i+a_ib_r).
\]

`alpha` 和 `beta` 也按复数公式参与组合。Kernel 分开计算实部、虚部，不混用 3M 接口。

## 详细设计（required）

### 算子实现

#### Host 侧设计

1. `gemm_host.cpp` 中的 `aclblasSgemm` 保留现有路径选择和 workspace 管理。当前实现按形状和转置选择直接 Matmul、原始 stride pack 或通用 pack，并在目标块上执行 alpha/beta epilogue。
2. `cgemm_host.cpp` 中的 `aclblasCgemm` 保留现有四实数乘积分层：

\[
P_{rr}=\operatorname{op}(A_r)\operatorname{op}(B_r),\quad
P_{ii}=\operatorname{op}(A_i)\operatorname{op}(B_i),
\]

\[
P_{ri}=\operatorname{op}(A_r)\operatorname{op}(B_i),\quad
P_{ir}=\operatorname{op}(A_i)\operatorname{op}(B_r).
\]

共轭符号由四个实数乘积的系数处理。四个实数后端共享已调优的 arch22 实数 GEMM，最后融合为：

\[
C_r'=\alpha_r(P_{rr}-P_{ii})-\alpha_i(P_{ri}+P_{ir})+\beta_rC_r-\beta_iC_i,
\]

\[
C_i'=\alpha_i(P_{rr}-P_{ii})+\alpha_r(P_{ri}+P_{ir})+\beta_iC_r+\beta_rC_i.
\]

该路径直接表达 COMPLEX64 乘加，不复用 `aclblasCgemm3m` 入口。

3. 新增 `hgemm_host.cpp`，完成 Hgemm 参数校验、FP16 Host 标量位模式转换、quick return、workspace 规划和 Kernel 分发。FP16 标量按 IEEE 位模式转换为 `float` 后供 Host 分支使用，不通过近似文本或整数直接解释。
4. 所有 workspace 请求经过溢出检查；Host 使用调用方 workspace 或 handle 默认 workspace，并在启动前检查实际容量。

#### Kernel 侧设计

**Sgemm**

沿用 `gemm_kernel.cpp/.h` 的 FP32 Matmul 与 vector tiling。对紧凑且对齐的输入使用直接 Matmul 视图；存在 padding、转置后的非连续逻辑访问或对齐不满足时，由 pack Kernel 按实际 `lda/ldb` 生成连续工作矩阵。输出按 `ldc` 写回，padding 保持原值。

**Cgemm**

`cgemm_kernel.cpp/.h` 执行三段流水：

1. AIV 将 COMPLEX64 AoS 拆分为 \(A_r/A_i\) 和 \(B_r/B_i\)，按物理 leading dimension 读取；
2. 四个 FP32 实数 GEMM 通过 handle stream 顺序提交；
3. AIV epilogue 组合 `alpha/beta`，只写 C 的逻辑区域。

长 K 按块提交并在同一 stream 上累加，避免额外 Host 同步。S/P 两种转置模式统一由实平面布局和共轭符号表达。

**Hgemm**

新增 `hgemm_kernel.cpp/.h`，使用 FP16 Cube 输入、FP32 L0C 累加和 FP16 输出。Host 将列主序矩阵映射为 Matmul 的 ND/DN 视图；直接访问要求物理地址满足 Matmul 对齐时使用直接视图，其余情况用 pack Kernel 按实际 `ldb` 生成对齐缓冲。尾块只在片上补零，写回时裁剪到逻辑 \(m\times n\)。

分核和片上资源规划采用与 arch22 实数 GEMM 一致的 tiling：

\[
\text{tiles}=\left\lceil\frac{m}{T_m}\right\rceil
\left\lceil\frac{n}{T_n}\right\rceil ,
\]

\[
\text{usedCoreNum}=\min(N_{AIC},\text{tiles}).
\]

当 \(k\) 超出单个片上归约容量时，Host 按 K 分块，并通过同一 stream 的有序 epilogue 累加；`beta` 只应用一次。小块使用较小 tile 提高 AIC 占用率，大块使用完整 Cube tile 提高数据复用。

#### 数据流

```text
A/B(Device, column-major)
        |
        +-- 直接对齐视图 ----------------------+
        |                                      |
        +-- pack(lda/ldb, transpose, pad) -----+
                                               v
                                    Ascend C Matmul / Cube
                                               |
                                               v
                                   alpha/beta epilogue(AIV)
                                               |
                                               v
                                  C logical m x n(ldc)
```

Cgemm 在中间增加 real/imag 拆分和组合二层；Sgemm/Hgemm 使用实数路径，其中 Hgemm 的 Cube 输入和 C 输出为 FP16。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3） | √ |
| Atlas A3 系列（arch22） | √ |

自验证至少覆盖一种 A2/A3 款型；性能数据在 910B3 采集。

### 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 存储布局 | A/B/C 为列主序，仅支持 leading dimension，不支持广播或任意 stride/view |
| 原地输出 | C 原地覆写；调用方保证 C 与 A/B 不重叠 |
| 动态形状 | `m/n/k`、leading dimension 和转置均为运行时参数 |
| 异步执行 | Kernel 在 handle 绑定的 stream 上异步执行 |
| 确定性 | 多核切分和 K 方向归约顺序可按形状变化，不要求逐位确定 |
| 精度模式 | S 使用 FP32；C 使用 FP32 实/虚分量；H 使用 FP16 输入和 FP32 累加 |

## 可维可测分析

### 精度标准

| 数据类型 | rtol | atol | matched ratio | max error |
| --- | ---: | ---: | ---: | --- |
| FLOAT32 | \(2^{-13}\) | \(2^{-13}\) | ≥0.99 | 1e-2 或 32 ULP |
| FLOAT16 | \(2^{-10}\) | \(2^{-10}\) | ≥0.99 | 1e-1 或 32 ULP |

逐元素满足

\[
|actual-golden|\leq atol+rtol\,|golden|
\]

记为 matched。Cgemm 的实部和虚部分别统计。S/C 的 golden 分别由 Netlib `cblas_sgemm` / `cblas_cgemm` 生成；H 的 golden 由 float64 累加后依次舍入到 FP32、FP16。

### 用例覆盖

测试代码复用 `test/gemm/arch22/gemm_test.cpp` 和 CSV 驱动框架，按 `func` 字段分派 `sgemm/cgemm/hgemm`，不新增 `test/gemm/sgemm/arch22/`：

| 数据 | 精度用例 | 性能用例 |
| --- | ---: | ---: |
| Sgemm | 334 | 67 |
| Cgemm | 333 | 67 |
| Hgemm | 333 | 66 |
| 合计 | 1000 | 200 |

覆盖九种 N/T/C 组合、尺寸池、非对齐与矩形形状、`lda/ldb/ldc` padding、均匀/正态输入、alpha/beta 特殊值、Inf/NaN、FP16 上限和次正规数、空矩阵、零 K、alpha=0、beta=0、非法枚举、负维度、非法 leading dimension 和空指针。Cgemm 另含非对称复数矩阵的 `OP_C` 共轭专项；S/H 验证 `OP_C` 与 `OP_T` 等价。

### 性能标准

性能设备为 Atlas 800T A2（910B3）。GTest 先 warmup 5 次，再用 `aclrtEvent` 对连续 10 次下发取平均，打印：

```text
[PERF] case=<case_name> ... avg_us=<value>
```

正式判据使用 `msprof op` 采集 kernel Task Duration：warmup 5 次不计入，后续 10 次有效率样本取平均，满足

\[
T_{NPU}\leq1.25\,T_{GPU}.
\]

`gpu_ms` 为对应算子相同参数下 GPU kernel-only 基线。ACL Event 的 `avg_us` 作为端到端辅助证据；两种口径分开记录，不用事件时间替代正式 kernel 判据。

### 兼容性分析

1. S/C 公共声明保持不变，仅新增 Hgemm 声明；
2. Cgemm 不改变 `aclblasCgemm3m` 的 API 和语义；
3. 新实现位于 arch22，arch35 及其他产品线路径保持不变；
4. 所有 Kernel 使用 handle stream，沿用 ops-blas 异步模型；
5. 新代码遵守仓内命名、日志、格式和 pre-commit 规范，并补充 `blas/gemm/README.md` 的 A2/A3 支持说明。

### 验证与交付

1. 使用 `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3` 完成编译；
2. 先执行边界、转置、padding 和 Cgemm 共轭定向用例，再执行 1000 条精度用例；
3. 精度全部通过后单独执行 200 条性能用例，保存 GTest 日志、`[PERF]` 数据、`msprof` 原始记录和逐 case 汇总；
4. 保存环境版本、设备、命令、退出码和日志路径；
5. 交付设计文档、测试用例包、自测报告、个人仓 commit 和 `task_submission`。

### 参考资料

1. `aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md` 及随附测试用例包；
2. [CANN 社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)；
3. [cann/ops-blas](https://gitcode.com/cann/ops-blas)；
4. [Netlib SGEMM](https://www.netlib.org/blas/sgemm.f)、[Netlib CGEMM](https://www.netlib.org/blas/cgemm.f)；
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)；
6. [cuBLAS GEMM API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)。
