# 需求背景（required）

## 需求来源

2026 年 11 月社区任务“aclblasCgemm3m 算子开发（A2A3）”。目标硬件为 Atlas A2/A3，
开发基线为 ops-blas，目标 CANN 版本为 9.1.0。任务在 arch22 上补齐
`aclblasCgemm3m`，公共接口声明已存在于 `include/cann_ops_blas.h`，本任务不新增平行 API。

## 背景介绍

`aclblasCgemm3m` 计算 complex64 通用矩阵乘法：

\[
C = \alpha \operatorname{op}(A)\operatorname{op}(B) + \beta C
\]

其中 \(A,B,C\) 按列主序存储，\(\alpha,\beta\) 为 complex64 标量，
\(\operatorname{op}\) 支持不转置、转置和共轭转置。ops-blas 的 arch35 已有对应实现，
arch22 已有 `aclblasCgemm` 及其四个实数乘积路径、反交错和复数 epilogue。本任务复用
arch22 已验证的实数 GEMM、K 分块和工作区管理，实现三次实数矩阵乘法的 Cgemm3m 路径。

# 需求分析（required）

## 需求描述

接口签名为：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数语义与 cuBLAS `cublasCgemm3m` 对齐。`op(A)` 为 \(m\times k\)，`op(B)` 为
\(k\times n\)，C 为 \(m\times n\)。`transa`、`transb` 覆盖 N/T/C 九种组合；
其中 C 表示共轭转置。前导维分别满足：

\[
\operatorname{lda}\ge
\begin{cases}
\max(1,m), & \operatorname{transa}=N\\
\max(1,k), & \operatorname{transa}\in\{T,C\}
\end{cases}
\]

\[
\operatorname{ldb}\ge
\begin{cases}
\max(1,k), & \operatorname{transb}=N\\
\max(1,n), & \operatorname{transb}\in\{T,C\}
\end{cases},
\qquad \operatorname{ldc}\ge\max(1,m)
\]

## 需求拆解

| 场景 | 行为 |
| --- | --- |
| handle 为空 | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 非法枚举、负维度、非法前导维、空标量指针 | 返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举按 ops-blas 约定返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| \(m=0\) 或 \(n=0\) | 参数校验后返回 `SUCCESS`，不引用 A/B/C，不启动 kernel |
| \(k=0\) 或 \(\alpha=0\) | 不引用 A/B，仅计算 \(C=\beta C\) |
| \(m,n>0\) | C 必须非空 |
| \(\beta=0\) | 不读取 C 的旧值 |
| 带填充的前导维 | 只写 C 的逻辑 \(m\times n\) 区域，padding 保持不变 |

# 详细设计（required）

## 算子分析

### 数学公式

记处理转置和共轭后的输入为

\[
\operatorname{op}(A)=A_r+iA_i,\qquad
\operatorname{op}(B)=B_r+iB_i
\]

标准三乘法分解为：

\[
P_0=A_rB_r,\qquad P_1=A_iB_i,\qquad
P_2=(A_r+A_i)(B_r+B_i)
\]

\[
\operatorname{Re}(\operatorname{op}(A)\operatorname{op}(B))=P_0-P_1
\]

\[
\operatorname{Im}(\operatorname{op}(A)\operatorname{op}(B))=P_2-P_0-P_1
\]

共轭转置通过符号折叠到辅助操作数：

\[
s_a=\begin{cases}-1,&\operatorname{transa}=C\\+1,&\text{其他}\end{cases},
\qquad
s_b=\begin{cases}-1,&\operatorname{transb}=C\\+1,&\text{其他}\end{cases}
\]

实际计算使用：

\[
P_2=(A_r+s_aA_i)(B_r+s_bB_i)
\]

展开后与上面的虚部公式一致。三个乘积均为 FP32 实数矩阵乘法，随后在复数 epilogue
中应用 complex64 的 \(\alpha,\beta\)。

### 支持数据类型

| 参数 | 类型 |
| --- | --- |
| A/B/C | complex64，实部和虚部各为 float32 |
| alpha/beta | complex64 标量 |
| m/n/k/lda/ldb/ldc | int |

### 支持形状

支持任意非负 \(m,n,k\)。正向计算的物理存储为按 lda/ldb/ldc 描述的列主序密集矩阵；
不要求支持超出前导维语义的任意非连续布局、广播或跨步视图。

## 算子实现

### Host 侧设计

在 `blas/gemm/arch22/cgemm_host.cpp` 增加公共入口 `aclblasCgemm3m`，与既有
`aclblasCgemm` 共用参数校验、标量读取、工作区管理、scratch 子句柄和复数 epilogue。
新增布尔型 `threeProducts` 只控制乘积数量和 epilogue 重组方式，旧接口继续走四个实数乘积。

Host 侧执行以下步骤：

1. 按固定顺序校验 handle、枚举、维度、前导维和标量指针。
2. 读取 Host 或 Device 上的 alpha/beta，保留 `ops-blas` 既有兼容行为。
3. 对 \(m=0\) 或 \(n=0\) 直接返回；对 \(\alpha=0\) 或 \(k=0\) 允许 A/B 为空。
4. 规划 A/B 实部、虚部和辅助和操作数，以及 \(P_0,P_1,P_2\) 和实数 GEMM scratch。
5. 在 handle 绑定的 stream 上依次提交反交错、三个实数 GEMM 和复数 epilogue。

所有 workspace 尺寸使用带溢出检查的乘加计算。三个乘积按固定 K 顺序计算并合并，
不使用浮点原子累加。\(k\) 超过已验证的实数 GEMM 归约长度时按固定块继续切分。

为避免 `RANDOM_EXTREME` 中的 \(\pm\mathrm{FLT\_MAX}\) 在辅助和或累加中提前溢出，
三个乘积路径先把 A/B 各缩放 \(2^{-8}\)，再在 epilogue 的 alpha 上精确乘回 \(2^{16}\)。
缩放因子是 2 的整数次幂，不对正常数引入额外舍入。

### Kernel 侧设计

`blas/gemm/arch22/cgemm_kernel.cpp` 新增 `cgemm_split_sum_kernel`：

- 每个 AIV 核按列轮转处理输入；
- 一次反交错同时写出 \(X_r\)、\(X_i\)、\(X_r+sX_i\)；
- 对三个输出执行显式 MTE3/MTE2 同步；
- 使用缩放后的实数平面进入既有 arch22 实数 GEMM。

三个乘积写入独立工作区平面：

\[
P_0=\operatorname{GEMM}(A_r,B_r)
\]

\[
P_1=\operatorname{GEMM}(A_i,B_i)
\]

\[
P_2=\operatorname{GEMM}(A_r+s_aA_i,B_r+s_bB_i)
\]

`cgemm_combine_kernel` 增加三乘积分支。对每个输出元素按固定顺序计算：

\[
\text{real}=P_0-P_1,\qquad
\text{imag}=P_2-P_0-P_1
\]

随后执行复数 alpha/beta：

\[
C_r=\alpha_r\text{real}-\alpha_i\text{imag}
+\beta_rC_{r,\mathrm{old}}-\beta_iC_{i,\mathrm{old}}
\]

\[
C_i=\alpha_r\text{imag}+\alpha_i\text{real}
+\beta_rC_{i,\mathrm{old}}+\beta_iC_{r,\mathrm{old}}
\]

alpha 或 beta 的分量为 0 时跳过对应乘法；beta 为零时不读取旧 C。C 按列只访问
逻辑 m 行，物理 padding 不写。

## 支持硬件

| 芯片版本 | 支持 |
| --- | --- |
| Atlas 800T A2 / Atlas A2（arch22） | 是 |
| Atlas 800T A3 / Atlas A3（arch22） | 是 |
| Atlas 950PR（arch35） | 由仓内既有 Cgemm3m 实现支持，本任务不修改 |

## 算子约束限制

- A/B/C 位于 Device 内存；alpha/beta 按公共约定指向 Host 标量，同时兼容仓内既有
  Host-or-Device 标量读取。
- C 与 A/B 存储不得重叠。
- 调用方保证 Device 缓冲区覆盖有效前导维；Host 侧检查结构参数和空指针。
- m/n/k/ld 和 trans 均为运行时参数，Host 侧完成真实形状的工作区规划和 launch。
- 同一 stream 上的执行顺序提供依赖关系；Host 读回结果前由调用方同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判据 | 来源 |
| --- | --- | --- |
| 精度 | 实部、虚部分别比对；`atol=rtol=2^-13`；matched ratio ≥99%；max abs error ≤0.01 或 32 ULP | 任务书 §3.2 |
| 非有限值 | 同为 NaN 或同号 Inf 匹配；有限与非有限、异号 Inf 不匹配 | 任务书 §3.2 |
| 正式性能 | `msprof op` kernel Task Duration；warmup 5 次后 10 个有效样本；NPU ≤1.25×GPU `gpu_ms` | 任务书 §3.3 |
| 辅助性能 | GTest `aclrtEvent` 连续 10 次下发取平均，打印 `[PERF] avg_us=` | 任务书 §3.3 |
| 测试规模 | 精度 1000 条，性能 200 条，使用任务包 CSV | 任务书 §3.5 |

## 兼容性分析

- 复用现有 `include/cann_ops_blas.h` 声明，不修改 ABI、枚举值或结构体布局。
- `aclblasCgemm` 保持原有四乘积路径、参数语义和返回值；新增分支仅在
  `aclblasCgemm3m` 入口启用。
- arch22 构建通过 `build.sh --ops=cgemm3m --soc=ascend910b3`；arch35 既有测试路径不变。
- arch22 测试位于 `test/gemm/cgemm3m/arch22/`，arch35 测试保持原位置。

## 测试方案

1. 用任务包 `cgemm3m_test.csv` 驱动 1200 条 GTest 用例，其中精度 1000 条、性能 200 条。
2. 精度测试调用 arch22 `aclblasCgemm3m`，CPU golden 使用 Netlib cblas `cblas_cgemm`。
3. 逐条验证状态码、实部、虚部、padding 不变、quick return、beta=0 和条件性空指针。
4. 性能测试先完成 GTest `avg_us` 交叉验证，再用 `msprof op` 对 TC_PF 用例采集
   kernel Task Duration，并以 `gpu_baseline.csv` 的 kernel-only 数据判定。
5. 自验证报告记录版本、目标 SOC、构建日志、精度日志、性能原始数据与结果。

## 资料来源

- 任务书：`aclblasCgemm3m_AtlasA2_A3_task_doc.md`
- 社区设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`
- ops-blas：<https://gitcode.com/cann/ops-blas>
- CANN 社区任务设计仓：<https://gitcode.com/cann/cann-ops-competitions>
- Netlib BLAS `cgemm`：<https://www.netlib.org/blas/cgemm.f>
