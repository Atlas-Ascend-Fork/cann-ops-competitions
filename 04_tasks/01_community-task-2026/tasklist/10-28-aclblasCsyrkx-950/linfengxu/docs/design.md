# aclblasCsyrkx（Atlas 950PR）算子设计

## 需求背景

### 需求来源

[10 月社区任务 aclblasCsyrkx 算子开发（950）](https://www.hiascend.com/activities/task-center/details/00e0335bc557460a9ebdce3ee4d60a9b)。
在 [ops-blas](https://gitcode.com/cann/ops-blas) 中使用 Ascend C kernel 直调方式补齐 complex64 对称 Rank-kx 更新，适配 Atlas 950PR、CANN 9.1.0，接口对标 cuBLAS `cublasCsyrkx`。

### 功能与现状

设 X=op(A)、Y=op(B)，在 uplo 指定的三角内计算：

```text
C(i,j) = alpha * sum[p=0..k-1](X(i,p) * Y(j,p)) + beta * C_old(i,j)
```

trans=N 时 X=A、Y=B，计算 A·Bᵀ；trans=T/C 时 X=Aᵀ、Y=Bᵀ，计算 Aᵀ·B。
按本任务书 §2.1，OP_C 与 OP_T 完全等价，不取共轭。只计算一个乘积，不增加 B·Aᵀ 项，也不将对角线虚部置零。
一般独立 A、B 的乘积不必对称；本接口按任务书对指定三角求值，不能依据“对称”假设镜像另一个三角或替换 A/B。
未选三角和前导维 padding 保持调用前的位模式。

仓内已有 arch35 实数 `aclblasSsyrkx`、复数 `aclblasCgemm` 及对称更新辅助组件，可复用句柄、工作空间、矩阵分块和直调机制；新增统一 `aclblasCsyrkx` 声明与 arch35 实现。

## 需求分析

### 接口定义

在 `include/cann_ops_blas.h` 中声明，不增设产品线专用 API：

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 参数 | 类型、形状与约束 |
| --- | --- |
| handle | 非空 aclblas 句柄，使用其绑定的 stream。 |
| uplo | ACLBLAS_UPPER 或 ACLBLAS_LOWER，均包含对角线。 |
| trans | ACLBLAS_OP_N、ACLBLAS_OP_T、ACLBLAS_OP_C；C 按 T 处理。 |
| n、k | 非负 int；分别是输出阶数、缩并维。 |
| alpha、beta | 指向 Host complex64 标量；两者必须非空。 |
| A、B | Device complex64 列主序矩阵；N 为 n×k，T/C 为 k×n；n>0 且 k>0 时指针必须非空。 |
| lda、ldb | 单位为复数元素，分别不小于 max(1, trans=N ? n : k)，允许合法 padding。 |
| C | Device complex64 列主序 n×n 输入/输出；n>0 时非空；仅引用和写回指定三角。 |
| ldc | 单位为复数元素，不小于 max(1,n)。 |

复数由两个 FLOAT32 分量构成。支持合法前导维，不支持额外任意 stride、广播或编译期未知 shape。

### 边界与错误处理

1. 先校验 handle、枚举、非负维度、前导维及条件指针，再判断 no-op。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、负维度、非法前导维、应非空的指针为空均返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. n=0 返回 SUCCESS，不下发计算、不访问 C；alpha/beta 和前导维仍须合法。
3. k=0 不引用 A/B，仅缩放指定三角：beta=0 直接置零，beta=1 保持原值，其余计算复数 beta·C。
4. alpha=0 时经过参数检查后采用同一三角缩放路径；beta=0 不读取 C 旧值，避免将未初始化值或 NaN 引入结果。
5. 固定输出分工及缩并次序，同输入重复运行逐位一致；不使用无序浮点原子累加。
6. 地址、工作空间字节数使用带溢出检查的宽整数计算；工作空间申请及下发失败沿用仓内状态码，不返回虚假的 SUCCESS。

### 需对齐的附件差异

本设计采用任务书 §2、§3 的明确条款。附件 README 中 OP_C 共轭旧描述与 §2.1 不一致，以不共轭实现；CSV `TC_ED_370/alpha0_a_null` 在 n=k=16、A=null 时期望 SUCCESS，与 §2.4 的指针约束不一致，本设计按 §2.4 返回 INVALID_VALUE。原始用例保留，验收前请确认该例预期。
性能以 §3.3 的 10 次有效采样、逐例 0.4× 与平均 0.5× 为设计标准；§7.4 与附件中不同采样数、倍率的文字请一并确认。

## 详细设计

### 数学分解

令 X=Xr+iXi、Y=Yr+iYi，计算：

```text
Pr = Xr * Yr^T - Xi * Yi^T
Pi = Xr * Yi^T + Xi * Yr^T
Cr = alpha_r*Pr - alpha_i*Pi + beta_r*C_old_r - beta_i*C_old_i
Ci = alpha_r*Pi + alpha_i*Pr + beta_r*C_old_i + beta_i*C_old_r
```

四个实矩阵乘复用同族 GEMM 分块及 Cube 计算模板，FLOAT32 累加。计算与暂存只覆盖有用三角块，三角内完整块和对角边界块分开处理。完整块批量搬运，对角块逐列限制有效写回区间，尾块不越界。

### Host 侧

完成参数验证，将 OP_C 规范化为 OP_T，读取 Host 标量并处理 n=0、k=0、alpha=0 快路径。
根据 n/k、前导维、平台核数、片上容量选择分块和核数；分派仅依赖合法输入属性，不依赖用例名称或预知输出。
通过 handle 复用所需工作空间，将打包、乘法和输出阶段顺序下发到同一 stream，不逐次申请和释放临时内存，不复制未选三角。

### Kernel 侧

| 阶段 | 计算单元与职责 |
| --- | --- |
| 输入组织 | AIV 对 A/B 交错实虚分量拆分并按 GEMM 需要打包；按各自 lda/ldb 寻址，转换融合于搬运，尾块补零。 |
| 乘积 | AIC/Cube 复用 GEMM 模板执行四个实乘积；沿 k 分块，复用 A/B panel，使用片上双缓冲覆盖搬运。固定分块归约顺序。 |
| 输出 | AIV 合成 Pr/Pi，融合复数 alpha/beta 运算与三角写回；beta=0 不读旧 C，保留对角虚部。 |
| 小规模与退化路径 | AIV 直接复数累加或缩放，减少打包、暂存和多次 kernel 启动；保持相同接口与精度规则。 |

优先采用 FP32 精度路径建立正确实现。若采用更快的 Cube 乘法模式，必须配套误差补偿并通过本任务混合容差和相消输入验证，不能直接将 complex64 输入降为低精度。
三实乘（3M）、融合打包及确定顺序的 K 分段属于可比较的后续实现选择，采用前验证精度、非有限值和完整调用成本。
核间不共享未同步的可变缓冲，MTE/Vector/Cube 各阶段按生产消费关系使用官方同步机制。

### 工程位置与兼容性

| 位置 | 内容 |
| --- | --- |
| include/cann_ops_blas.h | 统一接口声明。 |
| blas/syrkx/arch35/ | Csyrkx Host、tiling 与直调 kernel。 |
| blas/syrkx/README.md | 参数、约束、示例和支持硬件。 |
| test/syrkx/csyrkx/arch35/ | 沿仓内同族组织参数化 GTest、CSV 和 golden。 |

保持现有 Ssyrkx 和其他架构行为不变。新增代码复用公共组件，按仓库构建规则接入 `--ops=csyrkx --soc=ascend950`，不引入独立算子框架。支持硬件为 Atlas 950PR，软件基线为 CANN 9.1.0。

## 可维可测分析

### 精度与功能

采用 ops-blas CSV/GTest 框架和 Netlib BLAS 3.10.0 `cblas_cgemm` golden。N 使用 NoTrans/Trans，T、C 均使用 Trans/NoTrans，随后恢复非选三角原值。
uplo 三角实部、虚部分别统计：atol=rtol=2⁻¹³，匹配比例至少 0.99，最大绝对误差满足任务书的 1e-2 或 32 ULP 上限；MERE/MARE 不能替代该判据。
非选三角及 padding 按原始字节比较；相同输入独立执行两次核对确定性。

Inf/NaN 按 §3.2 单独分类：同为 NaN、同号 Inf、NaN 与任意 Inf 视为等效；有限与非有限不匹配、异号 Inf 不匹配。非有限分量全部通过，有限分量再按混合容差和最大误差判定。
覆盖任务附件 1000 条精度用例、UPPER/LOWER×N/T/C、非对齐与 padding、相消、复数标量、n/k=0、beta=0 配 NaN 旧 C、指针和非法参数；明确记录附件冲突项，不能静默修改预期制造通过。

### 性能

采用任务附件 200 条性能 case 及 `gpu_baseline.csv`，H100 基线环境为 CUDA 12.4、驱动 550.104.05。
在 Atlas 950PR 上用官方单算子 profiling 工具采集 task duration，warmup 5 次后有效采样 10 次取算术平均。一次接口调用包含多个 kernel 时计入全部阶段，不仅取 GEMM 主体；参数一致的 case 仍按各自 case_name 保留对应结果。
每例 GPU/NPU 耗时比至少 0.4，全部 200 例倍率的算术平均至少 0.5，两项同时满足。错误返回和无效采样不作为有效成绩，须说明原因并补齐。
使用官方 msopprof 的流水、访存和负载数据定位主要成本；阶段性方案形成后回归完整组合，不将单个 kernel 加速视作接口达标。
