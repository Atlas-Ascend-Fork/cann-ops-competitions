# Atlas A2/A3 上 aclblasSgemm、aclblasCgemm 与 aclblasHgemm 的设计

## 1 需求背景

### 需求来源

11 月社区任务「gemm 类 3 个算子开发（A2A3）」要求在 CANN 9.1.0 的 ops-blas 工程上，以 Ascend C 提供与 cuBLAS 同名操作参数顺序一致的单矩阵 GEMM。适配 Atlas A2、A3，正式性能对照设备为 Atlas 800T A2（910B3）。

### 现状与复用范围

arch22 的 `blas/gemm/arch22/gemm_host.cpp` 已提供 Sgemm，包含紧凑矩阵直算、带前导维输入打包以及工作空间规划。以该入口为起点，保留可用的 FP32 计算与分派路径，重点复核短路顺序、转置别名和尾块写回。Cgemm 参考 arch35 的 `cgemm_host.cpp` 中的复数表示和标量语义，Hgemm 参考 `hgemmStridedBatched_host.cpp` 及对应 kernel 的 FP16 搬运与乘加。两者在 arch22 上按本代 Matmul 的类型、搬运对齐和片上容量重新选择分块。Hgemm 处理一对输入矩阵，无批次步长。

## 2 需求分析

三个入口按列主序原地更新有效的 m×n 输出区域。沿用公开声明中的参数顺序：

```text
(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

| 入口 | alpha、beta、A、B、C 的元素类型 | 乘加与输出 |
| --- | --- | --- |
| aclblasSgemm | float | FP32 乘加，FP32 输出 |
| aclblasCgemm | aclblasComplex | 两个 FP32 分量进行复数乘加，交错写回复数 |
| aclblasHgemm | aclblasHalf | 按 IEEE FP16 位模式读入，FP32 累加及缩放，最终舍入到 FP16 |

alpha、beta 为 Host 标量指针。A、B 为只读 Device 指针，C 为可写 Device 指针。沿用 S、C 公开签名，在 `include/cann_ops_blas.h` 增加同序的非 batch Hgemm 声明，返回类型均为 `aclblasStatus_t`。调用方通过 handle 绑定 stream，在读取结果前同步该 stream。

## 3 详细设计

### 3.1 数学公式与访存

对 0 ≤ i < m、0 ≤ j < n，计算
`C[i+j*ldc] = alpha * sum(Aop(i,p) * Bop(p,j), p=0..k-1) + beta * C[i+j*ldc]`。

| 矩阵 | N 时的元素位置、前导维下界 | T/C 时的元素位置、前导维下界 |
| --- | --- | --- |
| Aop(i,p) | i+p×lda，lda ≥ max(1,m) | p+i×lda，lda ≥ max(1,k) |
| Bop(p,j) | p+j×ldb，ldb ≥ max(1,k) | j+p×ldb，ldb ≥ max(1,n) |

ldc ≥ max(1,m)。S/H 将 C 转置标志归一为 T。对复数，T 仅交换坐标，C 还对该操作数的虚部取反。矩阵仅支持由前导维描述的列填充布局，无广播；调用方保证 C 的写入区与 A/B 的读取区不重叠。

### 3.2 Host 校验与异步分派

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | handle 为空 | 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 2 | 任一操作枚举超出 N/T/C | 返回 ACLBLAS_STATUS_INVALID_ENUM |
| 3 | m、n、k 有负值，或任一前导维小于上述下界 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 4 | alpha 或 beta 指针为空 | 返回 ACLBLAS_STATUS_INVALID_VALUE，包括空输出场景 |
| 5 | m=0 或 n=0 | 返回 SUCCESS，允许 A/B/C 为空，无 kernel 下发 |
| 6 | 输出非空且 C 为空，或乘积有效时 A/B 任一为空 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 7 | k=0 或 alpha=0 | 允许 A/B 为空。beta=1 直接成功返回，beta=0 仅写零，其余只对有效 C 做缩放 |
| 8 | 常规乘加 | 在 handle 的 stream 上提交计算。beta=0 选择纯写回分支，其余分支读取旧 C 一次 |

复数的零和一分别按两个分量与 (0,0)、(1,0) 比较。Host 将标量按值传给异步任务。前导维乘积、工作空间偏移和对齐尺寸用宽整数计算，申请前检查溢出，工作空间沿用 handle 的分配及生命周期约定。分配、下发错误沿用仓内状态码返回。

### 3.3 分核、分块与数据流

1. Host 按输出块 BM×BN 将 m×n 划为网格，每块由一个核负责；工作核数取可用核数与块数的较小值。各核按网格编号跨步领块。核内沿 k 方向分 BK 深度遍历，同一输出块的归约有唯一写者。
2. 沿用 arch22 紧凑路径的布局判定。仅当底层 Matmul 能直接表达实际转置及物理步长时传入原指针；其余情况按 3.1 的地址将有效面板打包。尾 K 的乘数在本地补零，尾 M/N 只搬运有效元素。输出按列仅写有效行，保护每列末尾 padding。对于跨核边界的非对齐短写，使用有效长度搬运或合并相邻行块的写入所有权，避免读改写覆盖邻块。
3. 按 SoC 可用 L1、L0A/B、L0C 与 UB 容量选 BM、BN、BK。L0C 需容纳当前 FP32 累加块；UB 预留转换、复数拆合及尾块空间。仅在输入双缓冲和累加区同时可容纳时开启预取。缩小 BK 可压低输入缓冲占用，缩小 BM/BN 可压低累加区占用。以实际运行尺寸、布局和类型分派。
4. 小矩阵优先合并搬运、缩放与写回以控制下发开销。大矩阵复用面板并让预取与计算重叠。对于瘦长输出，沿较长轴增加块数。多阶段打包或计算须在同一 stream 排序；性能采集覆盖整次调用所需的全部 kernel。

### 3.4 三条计算路径

- **Sgemm** 沿用 arch22 的 FP32 乘加路径，将所有 K 块的部分和累加完成后再施加 alpha、beta。选用下级 Matmul 模式时保留 FP32 精度；未经逐点比对的降精度模式不作为默认配置。
- **Cgemm** 在载入面板时拆出 Ar、Ai、Br、Bi，并按各输入的共轭标志调整 Ai、Bi。实部分块累加 Ar·Br − Ai·Bi，虚部分块累加 Ar·Bi + Ai·Br，按四个实乘积实现常规 Cgemm。写回时用完整复数乘法计算 alpha·P 和 beta·C，再合并为交错分量。beta 为复零时跳过 C 的加载。以复数元素为单位计算 lda、ldb、ldc，拆分后再转换为字节偏移。
- **Hgemm** 将 FP16 面板送入矩阵乘单元，累加块保持 FP32。alpha、beta 和需要读取的旧 C 精确扩展到 FP32，在最后一个 K 块之后完成缩放、相加，按最近偶数规则收窄一次。以位模式转换代码处理 Host half 标量，避免把 16 位载体当成整数值。

### 3.5 工程接入与兼容性

实现放在 `blas/gemm/arch22/` 的 host、kernel 和配套头文件；在同一工程注册 S/C/H 的构建入口。扩展 `test/gemm/arch22/gemm_test.cpp` 的 dtype、复数标量及 half 转换分派，通过已有 CSV 测试入口加载三套用例。更新 `blas/gemm/README.md` 的产品支持表和参数限制。S/C 保留现有 ABI，H 新增入口复用公共 half 类型；arch35 仍按其原有工程路径构建。A2 以 `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3` 编译，A3 按对应实际 SoC 配置编译。

## 4 可维可测分析

### 4.1 用例范围与数值判定

| 附件用例 | Sgemm | Cgemm | Hgemm | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 精度及接口 | 334 | 333 | 333 | 1000 |
| 性能 | 67 | 67 | 66 | 200 |

使用仓内 ATK/GTest，按用例名关联输入、参考输出及执行记录。负向用例核对返回码；数值用例仅比较逻辑输出区，另逐列核对 padding 哨兵。常规输入覆盖均匀分布与正态分布，复数实虚部独立生成。九种转置组合均覆盖；以非对称且带虚部的矩阵区别 T/C。覆盖紧凑与加 1、8、64 的前导维、矩形、非对齐及 4095³/4096³。用空指针或 NaN 输入检查短路，用旧 C 含 NaN 检查 beta=0。H 另测次正规数、最大有限值与溢出。有限值和 NaN/Inf 分开判定，复数按分量记录。

S/C 对照 Netlib BLAS 3.10.0 的列主序 cblas_sgemm/cblas_cgemm；若使用高精度冻结参考，记录采用 FP64 分量累加并最终收窄至 FP32 的生成方式。H 参考以量化后的 FP16 输入及标量扩展到 FP64 计算，先转 FP32 再按最近偶数舍入到 FP16。逐元素比较绝对误差 ≤ atol + rtol × |golden|。逐例要求匹配率 ≥ 99%，且最大绝对误差满足对应上限或标准的 32 ULP 条件。S/C 的 atol=rtol=2⁻¹³，绝对上限 0.01；H 的 atol=rtol=2⁻¹⁰，绝对上限 0.1。C 的实部和虚部各自满足两项条件。

### 4.2 性能与内存

在 910B3 上按 dtype 和最大逻辑矩阵字节数分档（≤64 KiB、>64 KiB 至 2 MiB、>2 MiB）。每点预热 5 次，随后取连续 10 次成功执行的样本。用 msprof op 采集 Task Duration（μs）；多阶段路径按一次调用归集各阶段耗时，再取 10 次算术平均。与 A100 同名 cuBLAS 的 kernel 基线按类型、转置、尺寸、前导维及 alpha、beta 配对。逐点目标为平均耗时 ≤ 1250 × gpu_ms（μs）。记录无效样本原因，缺失有效样本或基线时保留该点状态。

另用 aclrtEvent 在预置 Device 缓冲上包围 10 次下发，记录每次平均值 `[PERF] avg_us`；计时范围排除造数、传输与 CPU 比对。正式耗时和事件计时分别落表。任务书未设置独立内存验收指标；对带 padding 的输出、尾块搬运以及最大尺寸工作空间做边界核查。

### 4.3 验证记录

自验记录需注明 SoC、CANN 版本、构建提交、用例名、输入种子、参考实现及 profiler 原始样本。A2、A3 分别记录编译或实机执行的覆盖范围。以上为设计与验证计划，精度、性能结果由后续实测填写。
