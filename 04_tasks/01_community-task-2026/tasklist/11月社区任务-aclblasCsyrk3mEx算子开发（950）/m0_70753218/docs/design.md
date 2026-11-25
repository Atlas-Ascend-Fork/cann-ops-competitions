# aclblasCsyrk3mEx 算子设计文档

## 一、需求背景

### 1.1 需求来源

本文档对应 2026 年 11 月社区任务《11月社区任务-aclblasCsyrk3mEx算子开发（950）》，目标是在 Ascend 950PR（arch35）上，使用 Ascend C 和 `cann/ops-blas` 的句柄式 kernel 直调工程，新增单精度复数对称矩阵秩-k 更新接口 `aclblasCsyrk3mEx`。

需求依据：

1. 任务包中的 `aclblasCsyrk3mEx_Atlas950PR_task_doc.md`。
2. 任务包中的 `test_cases/README.md`、`csyrk3m_ex_test.csv`、`gpu_baseline.csv`、生成器和验证脚本。
3. `cann/ops-blas` 当前 `master`，源码基线提交 `55ef791b77f3677b45da3468b0a8b17941424f84`。
4. 仓内已有 `blas/syrk/arch35/csyrk_*`、`blas/gemm/arch35/cgemm3m_*` 及 `test/syrk/csyrk/`。

### 1.2 背景介绍

`aclblasCsyrk3mEx` 对标 cuBLAS `cublasCsyrk3mEx`，计算：

$$
C = \alpha\,\operatorname{op}(A)\operatorname{op}(A)^T + \beta C
$$

其中 \(C\) 为 \(n\times n\) 对称复数矩阵，\(\alpha,\beta\) 为 COMPLEX64 标量，\(A\) 为 COMPLEX64 或 complex-int8。

本任务采用非 Hermitian 对称语义：

- 只读写 `uplo` 指定的三角区域。
- `trans=ACLBLAS_OP_C` 与 `trans=ACLBLAS_OP_T` 等价，不做共轭。
- 对角元素的虚部是有效数据，计算后不置零。

仓内非 Ex 的 `aclblasCsyrk` 已具备参数校验、三角处理、direct 路径和 4M cube 路径，但其公共接口固定为 COMPLEX64，且 4M 分解不符合本任务指定的 3M 公式。`cgemm3m` 提供复数 GEMM 的 3M 分项和融合执行经验。本任务新增独立接口与实现，不依赖尚未合入的 `aclblasCsyrkEx`。

## 二、需求分析

### 2.1 接口原型

在 `include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

接口声明与 cuBLAS `cublasCsyrk3mEx` 的参数序列一一对应。`Atype=ACL_INT8` 表示 complex-int8，每个元素按 `(int8 real, int8 imag)` 交错 2 Byte 存储。

### 2.2 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 硬件 | Ascend 950PR，arch35 |
| CANN | 9.1.0 及以上 |
| `C`、`alpha`、`beta` | COMPLEX64 |
| `A` | COMPLEX64 或 ACL_INT8 |
| `uplo` | ACLBLAS_UPPER、ACLBLAS_LOWER |
| `trans` | ACLBLAS_OP_N、ACLBLAS_OP_T、ACLBLAS_OP_C |
| 存储 | 列主序，使用 `lda`/`ldc` 描述带 padding 布局 |
| 输入输出 | `C` 原地覆写，仅写 `uplo` 指定的三角 |
| 广播、动态 shape | 不涉及 |

### 2.3 数学与转置语义

令

$$
X=\operatorname{op}(A)=X_r+\mathrm{i}X_i
$$

当 `trans=ACLBLAS_OP_N` 时，\(A\) 按 \(n\times k\) 存储，\(X=A\)；当 `trans=ACLBLAS_OP_T` 或 `ACLBLAS_OP_C` 时，\(A\) 按 \(k\times n\) 存储，\(X=A^T\)。`OP_C` 不做共轭。

任务的 3M 分解为：

$$
P_0=X_rX_r^T,\qquad
P_1=X_iX_i^T,\qquad
P_2=(X_r+X_i)(X_r+X_i)^T
$$

$$
\operatorname{Re}(XX^T)=P_0-P_1,\qquad
\operatorname{Im}(XX^T)=P_2-P_0-P_1
$$

再与 \(\alpha\)、\(\beta\) 组合：

$$
\begin{aligned}
C' &=\alpha\left[(P_0-P_1)+\mathrm{i}(P_2-P_0-P_1)\right]+\beta C.
\end{aligned}
$$

三次实数矩阵乘 \(P_0,P_1,P_2\) 在一次 kernel launch 中完成。实现不会通过三次独立 GEMM API 调用拼接，也不使用 Hermitian 的 \(XX^H\) 分解。

### 2.4 参数校验与快速返回

校验顺序为：

1. `handle`。
2. `uplo`、`trans`。
3. `n`、`k`。
4. `lda`、`ldc`。
5. `alpha`、`beta`、`Atype`、`Ctype`、`A`、`C`。

返回码约定：

| 条件 | 返回 |
| --- | --- |
| `handle=nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 非法 `uplo` 或 `trans` | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n<0`、`k<0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda`、`ldc` 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 必需指针为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `Atype`/`Ctype` 或组合不支持 | `ACLBLAS_STATUS_NOT_SUPPORTED` |

前导维度约束：

$$
lda\ge
\begin{cases}
\max(1,n),&\text{trans}=N\\
\max(1,k),&\text{trans}=T\text{ 或 }C
\end{cases},
\qquad
ldc\ge\max(1,n).
$$

退化语义：

- \(n=0\)：返回 `SUCCESS`，不访问 `A/C`，不启动 kernel。
- \(k=0\) 或 \(\alpha=(0,0)\)：秩-k 贡献为零。
- \(k=0\) 或 \(\alpha=(0,0)\)，且 \(\beta=(1,0)\)：快速返回，`C` 不变。
- \(k=0\) 或 \(\alpha=(0,0)\)，且 \(\beta\ne(1,0)\)：只执行指定三角的 \(C=\beta C\)，不访问 `A`。
- \(\beta=(0,0)\)：不读取旧 `C`。

## 三、详细设计

### 3.1 文件组织

在 `blas/syrk/arch35/` 新增：

```text
csyrk3m_ex_host.cpp
csyrk3m_ex_kernel.cpp
csyrk3m_ex_kernel.h
csyrk3m_ex_tiling_data.h
csyrk3m_ex_gemm_fused.h
csyrk3m_ex_reference_device.h
```

同步修改：

```text
include/cann_ops_blas.h
blas/syrk/README.md
```

测试代码放入：

```text
test/syrk/csyrk3m_ex/
├── CMakeLists.txt
├── csyrk3m_ex_param.h
├── csyrk3m_ex_golden.h
└── arch35/
    ├── csyrk3m_ex_npu_wrapper.h
    ├── csyrk3m_ex_test.cpp
    └── csyrk3m_ex_test.csv
```

### 3.2 Host 侧设计

Host 入口完成参数校验、标量位置识别、Atype 分派、tiling 计算和异步 kernel 下发。`alpha/beta` 同时支持 Host 和 Device 指针；位于 Device 时，在绑定 stream 上完成一次 D2H 读取后计算快速返回标志和 tiling 标量。

`Atype=ACL_INT8` 与 `Atype=ACL_COMPLEX64` 使用独立 tiling key。INT8 路径在 kernel 的 A 分解阶段把每个交错字节分别精确转换为 FLOAT32，后续计算与 COMPLEX64 路径相同。`Ctype` 仅接受 `ACL_COMPLEX64`。

工作区按实际物理跨度 \(lda\)、\(ldc\) 计算，并完成 64 位乘法与加法溢出检查。主要工作区为：

1. \(X_r\)、\(X_i\)。
2. 3M 的三个结果平面 \(P_0\)、\(P_1\)、\(P_2\)。

`k=0` 或 `alpha=0` 且 `beta\ne1` 的 beta-only 路径只启动缩放 kernel，不申请 3M 工作区。

### 3.3 Tiling 与分核

| Tiling 类型 | 条件 | 处理 |
| --- | --- | --- |
| beta-only | \(k=0\) 或 \(\alpha=0\)，且不满足快速返回 | 按 `uplo` 三角分块执行复标量缩放 |
| INT8 3M | \(Atype=ACL_INT8\) | int8 解包后执行融合 3M |
| COMPLEX64 3M | \(Atype=ACL_COMPLEX64\) | 直接解交织后执行融合 3M |
| 特殊值诊断 | 小规模输入含 Inf/NaN 或极值 | 在正式非有限值语义路径中处理，不改变正常路径分派 |

分核使用 AIC 数量按 64x64 C tile 的三角集合均分。每个 tile 只由一个 AIC 核负责，避免同一 C 区域多核写冲突。`uplo` 通过 tile 选择跳过另一三角。

单 tile 使用 BM=64、BN=64。K 方向按片上容量切为多个 chunk，L1/L0 使用双缓冲；三个 \(64\times64\) 的 L0C 累加器分别保存 \(P_0\)、\(P_1\)、\(P_2\)。尾块按实际 `curM`、`curN`、`curK` 处理。

### 3.4 Kernel 侧设计

#### 3.4.1 A 分解

对当前 tile 所需的 \(X_r\)、\(X_i\) 面板：

- COMPLEX64：按列连续搬入交错复数，使用 `DeInterleave` 分离实部和虚部。
- INT8：按列搬入 `(real, imag)` 字节对，使用向量转换得到 FLOAT32 实部和虚部。
- 非转置：生成 M/N 侧共用矩阵视图 \(X\)。
- 转置：使用 \(X^T\) 的列主序视图，不执行共轭。

分解结果写入 GM 工作区，供融合 cube kernel 读取。

#### 3.4.2 融合 3M cube

每个拥有的 C tile 按 K chunk 搬入以下输入面板：

$$
X_{r,M},\quad X_{i,M},\quad
(X_r+X_i)_M,\quad
X_{r,N},\quad X_{i,N},\quad (X_r+X_i)_N.
$$

在同一个 kernel 内依次累积：

$$
P_0\leftarrow X_{r,M}X_{r,N}^T
$$

$$
P_1\leftarrow X_{i,M}X_{i,N}^T
$$

$$
P_2\leftarrow (X_r+X_i)_M(X_r+X_i)_N^T
$$

三个累加器使用各自的 L0C 地址。K chunk 首项负责清零累加器，末项负责将结果从 L0C 写回 GM。三个 REAL mmad 序列在同一个融合 kernel 的 tile 循环内完成。

#### 3.4.3 组合与三角写回

组合阶段对 `uplo` 三角内的每个元素执行：

$$
t_r=P_0-P_1,\qquad
t_i=P_2-P_0-P_1
$$

复数 \(\alpha\) 与 \(t\) 做复数乘：

$$
u_r=\alpha_r t_r-\alpha_i t_i,\qquad
u_i=\alpha_r t_i+\alpha_i t_r
$$

对旧 `C` 做同样的 \(\beta C\)，再相加并写回同一 C 地址。\(\beta=(0,0)\) 时跳过旧 C 载入。padding 和另一三角不写。

对角元素仍执行完整复数公式，保存 \(X_r^2-X_i^2\) 和 \(2X_rX_i\) 对应的实虚部，不对虚部置零。

#### 3.4.4 beta-only

\(k=0\) 或 \(\alpha=(0,0)\) 且 \(\beta\ne(1,0)\) 时，只对指定三角执行：

$$
C_r'=\beta_r C_r-\beta_i C_i,\qquad
C_i'=\beta_r C_i+\beta_i C_r.
$$

该路径不读取 A 或 3M 中间结果。

#### 3.4.5 非有限值

正常 3M 与 INT8 转换按 FLOAT32 执行。组合阶段使用任务规定的乘法顺序，保留 `0*Inf`、NaN 传播等 IEEE 行为。对测试包中的 Inf/NaN 专项用例按 CPU CBLAS 的实虚部结果单独核对；该检查用于验证公式分支，不扩大正式接口值域。

### 3.5 异步与内存

接口只在 handle 绑定的 stream 上下发 kernel，不隐式同步。调用方在读取 C 前同步同一 stream。工作区由 handle 管理，调用期间保持有效。接口不产生视图，不申请用户可见输出。

### 3.6 支持硬件

| 芯片与软件 | 状态 |
| --- | --- |
| Ascend 950PR、arch35、CANN 9.1.0 及以上 | 开发与验收目标 |
| 其他昇腾产品 | 本任务不纳入 |

## 四、特性交叉分析

| 交叉项 | 设计措施 |
| --- | --- |
| `uplo` × 写回 | tile 与元素级均检查，只写指定三角 |
| `trans` × `lda` | N 使用 n 下界，T/C 使用 k 下界 |
| `OP_T` × `OP_C` | 映射到同一转置视图，不共轭 |
| INT8 × padding | 按 `lda` 的 complex-int8 元素跨度寻址 |
| INT8 极值 | 解包后直接转换为可精确表示的 float |
| 对角虚部 × 3M | 使用对称公式的 \(P_2-P_0-P_1\)，对角虚部保留 |
| `k=0` / `alpha=0` × `A` | 非快速返回路径只缩放 C，不访问 A |
| `beta=0` × `C=NaN` | 不读取旧 C，不做无用乘法 |
| 原地 C × 分核 | 每个三角元素由唯一核完成读改写 |
| 3M × kernel launch | 三个实数乘积在一个融合 kernel 的 tile 循环内完成 |

## 五、可维可测分析

### 5.1 正式精度判据

golden 使用 Netlib BLAS 3.10.0/CBLAS `cblas_csyrk`；INT8 先精确提升为 float 后调用。结果仅比较指定 uplo 三角中的实部和虚部。

每个 FLOAT32 分量使用：

$$
|\text{actual}-\text{golden}|
\le 2^{-13}+2^{-13}|\text{golden}|
$$

用例同时要求：

- matched ratio 不低于 0.99。
- 最大绝对误差不超过 \(10^{-2}\) 或 32 ULP。

3M 与直接复数乘法只要求满足上述阈值，不要求逐位一致。另一三角、padding 和只读输入使用哨兵检查。

### 5.2 正式测试集合

任务包实有 1200 条 CSV：

- 1000 条非 `TC_PF` 精度用例。
- 200 条 `TC_PF` 性能用例。

精度覆盖：

- UPPER/LOWER、N/T/C 全组合。
- COMPLEX64/INT8。
- n/k 尺寸、宽窄矩阵、前导维 padding。
- alpha/beta 特殊值和随机值。
- int8 交错与边界值。
- k=0、alpha=0、beta=0、空 handle、非法枚举、非法前导维、不支持 dtype。
- 对角虚部、Inf/NaN、`TC_3M_*` 专项。

精度与性能分开执行、分开判定。正式精度集合排除全部 `TC_PF`；正式性能集合只包含 `TC_PF`。

### 5.3 正式性能判据

正式数据为 `msprof op` 采集的 kernel `Task Duration(us)`。每个 case：

1. warmup 5 次。
2. 采集 10 次成功执行的有效样本。
3. 算术平均后除以 1000 转为 ms。

与 `gpu_baseline.csv` 的 GPU kernel-only `gpu_ms` 比较：

$$
\frac{gpu\_ms}{npu\_ms}\ge0.4
$$

且 200 条可比 case 的加速比算术平均：

$$
\operatorname{mean}\left(\frac{gpu\_ms}{npu\_ms}\right)\ge0.5.
$$

单 case 与整体判据同时满足。`verify_performance.py` 的 `[PERF] avg_us` 仅用于快速回归，不作为正式性能结论。INT8 补充性能沿用 COMPLEX64 的 baseline，按任务书单独记录。

### 5.4 内存

任务书 §3.4 为“不涉及”，不设置验收门槛。自测报告记录工作区和单核片上缓冲规模，作为诊断信息。

### 5.5 测试工程

GTest 使用任务 CSV：

- 加载 1000 条精度用例并生成 CBLAS golden。
- 对上行映射 Device 输入，调用 `aclblasCsyrk3mEx`。
- stream 同步后按实部和虚部校验。
- 单独覆盖空 handle。
- `TC_PF` 用例使用驻留 Device 缓冲，warmup 5 次，再以 `aclrtEvent` 连续采样 10 次，打印 `[PERF] case=<name> ... avg_us=<value>`。

构建入口：

```bash
bash build.sh --ops=csyrk3m_ex --soc=ascend950
```

### 5.6 交付件

自测通过后生成：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

内存报告记录“任务书 §3.4 不涉及”及工作区诊断数据。交付 ZIP 不超过 50 MiB、文件数不超过 150。

## 六、兼容性与工程影响

1. 新接口使用独立名称，不修改 `aclblasCsyrk` 的 ABI。
2. arch35 的 `csyrk` 现有实现和测试保留。
3. 新文件使用 `csyrk3m_ex_` 前缀，现有单算子构建过滤规则可直接保留所需源文件。
4. `test/syrk/csyrk3m_ex/CMakeLists.txt` 注册 `csyrk3m_ex_test` 构建目标。
5. `blas/syrk/README.md` 增加接口说明、数据类型、算法语义、约束和 950PR 支持状态。

## 七、正式验收清单

| 验收项 | 原文依据 | 正式判据 | 证据 |
| --- | --- | --- | --- |
| 接口与功能 | 任务书 §2.1～§2.4 | 13 参数签名、对称复数语义、N/T/C、两种 A 类型 | 声明、实现、GTest |
| 参数校验 | 任务书 §2.4、§2.5 | 指定状态码和校验顺序 | 负向用例日志 |
| 三角与对角 | 任务书 §2.1、§3.2 | 只写 uplo 三角，对角虚部保留 | 哨兵与专项用例 |
| 3M 算法 | 任务书 §2.1 | \(P_0,P_1,P_2\) 公式，同一 kernel launch | 源码与 profiling |
| 精度 | 任务书 §3.2 | rtol=atol=2^-13，匹配率 0.99，最大误差 1e-2/32 ULP | 1000 条精度日志 |
| 性能 | 任务书 §3.3 | 单 case ≥0.4×，平均值 ≥0.5× | 200 条 msprof 日志 |
| 内存 | 任务书 §3.4 | 不涉及 | 交付模板说明 |
| 交付件 | 任务书 §4 | 模板目录、ZIP 大小/文件数、commit 链接 | `task_submission/` 与 IT 系统 |

本设计的正式门槛只来自任务书原文。额外复测、自增极端用例、其他硬件和更严格写保护属于辅助诊断，不替代或扩张上述门槛。
