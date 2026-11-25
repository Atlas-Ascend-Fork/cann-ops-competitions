# aclblasCsyrk3mEx 算子设计文档

文档头信息表（任务、团队、仓库、目录与状态）：

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11月社区任务-aclblasCsyrk3mEx 算子开发（950） |
| 提交团队 | KQEA |
| 任务讨论 | [ops-blas discussions #88](https://gitcode.com/cann/ops-blas/discussions/88) |
| 设计文档提交目录 | ```04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasCsyrk3mEx算子开发（950）/KQEA/docs/design.md``` |
| 适配硬件 | Ascend 950PR，@arch35@（实测 28 AIC / 56 AIV） |
| CANN 版本 | CANN 9.1.0 及以上（asc-devkit 9.1 + Ascend-cann-950-ops 9.1.0） |
| 相关代码仓 | @cann/ops-blas@（开发分支 @perf/csyrk-3m@） |
| 代码目标目录 | @blas/syrk/arch35/@、@test/syrk/csyrk3m_ex/arch35/@ |
| 文档状态 | 设计评审阶段；本文只描述设计与自测结果，代码与测试随 @ops-blas@ 代码 PR 提交 |

# 需求背景（required）

## 需求来源

本算子来自 2026 年 11 月 CANN 社区任务「11月社区任务-aclblasCsyrk3mEx算子开发（950）」，目标是在 @cann/ops-blas@ 中
补齐 Atlas 950PR（@arch35@）上的复数对称秩-k 更新扩展接口 @aclblasCsyrk3mEx@（BLAS-3），
以 **3M 算法** 为唯一主路径，完成设计评审、算子开发、自验证与社区 PR 合入。

语义基准有三处，冲突时按下列顺序裁决：

1. cuBLAS @cublasCsyrk3mEx@（CUDA 文档 §2.8.16）：签名、参数语义、Atype/Ctype 组合与返回码口径；
2. Netlib @csyrk.f@：数学定义 @C = alpha*op(A)*op(A)^T + beta*C@；
3. 仓内 @blas/syrk/arch35/csyrk_host.cpp@（非 Ex @aclblasCsyrk@）：uplo 三角处理、对角处理与 Host 校验落地口径。

## 背景介绍

### aclblasCsyrk3mEx 算子实现

算子在 uplo 指定的三角上完成复数对称矩阵的秩-k 更新，C 原地覆写。接口沿用仓内公共声明
@include/cann_ops_blas.h@（本任务新增），13 个参数，列主序，句柄绑定 stream 直调 NPU kernel：

~~~cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha, const void* A, aclDataType Atype, int lda, const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
~~~

数学定义：

    C = alpha * op(A) * op(A)^T + beta * C

与 Hermitian 的 @cherk@ 不同，本算子**对称不共轭**：@trans = ACLBLAS_OP_C@ 按 @ACLBLAS_OP_T@ 等价处理，
不取共轭；对角元素 @C(i,i)@ 的虚部是有效数据，**计算后原样保留，不置 0**。

### 现状分析

| 审核对象 | 核查结论 | 依据 |
| --- | --- | --- |
| 原 TBE 算子源码 | 不适用 | 不迁移 TBE 算子；语义以 cuBLAS @cublasCsyrk3mEx@ 与 Netlib @csyrk.f@ 为准 |
| 算子信息库（ACLNN） | 不适用 | 接口为 @aclblas*@ 公共 BLAS 接口，不注册 ACLNN 算子 |
| 仓内可复用实现 | 部分适用 | @blas/syrk/arch35/@ 的句柄/stream/tiling/kernel 框架、@test/syrk/csyrk/@ 的 CSV+GTest 框架 |
| 3M 参考实现 | 适用 | @blas/gemm/arch35/cgemm3m_host.cpp@、@cgemm3m_rounding_kernel.cpp@ 的 3M 分解与舍入处理 |

仓内 @blas/syrk/arch35/@ 只有非 Ex 的 @aclblasCsyrk@（complex64 SYRK，**4M 融合**实现）与 @ssyrk@；
@test/syrk/@ 只有 @csyrk@、@ssyrk@；@include/cann_ops_blas.h@ 无 @aclblasCsyrk3mEx@ 声明。
@aclblasCsyrkEx@ 与 @aclblasCsyrk3mEx@ 是同批次新增的相互独立接口，前者未合入，本实现**不依赖**它，
也不允许用多次 GEMM 调用拼接 3M。

本任务新增文件：@csyrk3m_ex_host.cpp@（入口与校验，128 行）、@csyrk3m_ex_bridge.h@（入口与共享流水的内部桥）、
@csyrk_3m_cube_kernel.cpp@ / @csyrk_3m_gemm_fused.h@ / @csyrk_3m_m128.h@（3M cube）、
@csyrk_direct_kernel.cpp@（对角直算）、以及 @csyrk_host.cpp@ 中新增的 3M 共享流水与 workspace 规划；
测试新建 @test/syrk/csyrk3m_ex/@（@arch35/@ 下 CSV + GTest）；@blas/syrk/README.md@ 补接口章节。

### 算子功能分析

| 参数 | 含义 | 类型/位置 | 约束与异常行为 |
| --- | --- | --- | --- |
| handle | BLAS 句柄，携带 stream | Host | 为空返回 @HANDLE_IS_NULLPTR@ |
| uplo | C 的存储三角，另一半由对称性隐含 | Host | 非 @UPPER@/@LOWER@ 返回 @INVALID_ENUM@ |
| trans | op(A) 是否转置；@OP_C@ 与 @OP_T@ 等价 | Host | 非 @N@/@T@/@C@ 返回 @INVALID_ENUM@ |
| n | C 的阶数 | Host @int@ | @n<0@ 返回 @INVALID_VALUE@；@n=0@ 合法 no-op |
| k | 秩-k 的 k 维 | Host @int@ | @k<0@ 返回 @INVALID_VALUE@；@k=0@ 不发生秩-k 更新 |
| alpha | 复数标量乘数 | Host @const aclblasComplex*@ | @n*k>0@ 时不可为空；值不校验（含 0/Inf/NaN） |
| A | 输入矩阵，列主序 | Device @const void*@ | @trans=N@ 为 n×k，否则 k×n；@n*k>0@ 时不可为空 |
| Atype | A 的元素类型 | Host @aclDataType@ | @{COMPLEX64, INT8}@；INT8 为 (int8 re, int8 im) 交错 2 字节 |
| lda | A 的前导维度 | Host @int@ | @lda >= max(1, trans==N ? n : k)@ |
| beta | 复数标量乘数 | Host @const aclblasComplex*@ | 不可为空；值不校验 |
| C | 输入兼输出矩阵，原地覆写 | Device @aclblasComplex*@ | 仅 uplo 指定三角读写；@n>0@ 时不可为空 |
| Ctype | C 的元素类型 | Host @aclDataType@ | @{COMPLEX64}@ |
| ldc | C 的前导维度 | Host @int@ | @ldc >= max(1, n)@ |

## 口径裁决

1. **对称非 Hermitian**：只算 @X·X^T@（@X = op(A)@），@OP_C@ 不共轭且与 @OP_T@ 等价；对角虚部保留。
2. **INT8 语义**：@Atype=ACL_INT8@ 表示 complex-int8，每元素按 @(int8 real, int8 imag)@ 交错 2 字节
   （cuBLAS @CUDA_C_8I@），Host 侧**精确提升**为 complex64 副本后进入同一 3M 流水；不支持组合返回 @NOT_SUPPORTED@。
3. **3M 强制**：三次实数矩阵乘必须在**同一次 kernel launch** 内完成；@aclblasCsyrk3mEx@ 由内部 @force3m@
   强制 3M，不依赖环境变量，也不得回退为 4M。
4. **短路语义**：@n=0@ 合法 no-op（允许 A/C 为空）；@k=0@ 或 @alpha=(0,0)@ 时秩-k 贡献为 0 ——
   @beta=(1,0)@ quick return，@beta!=(1,0)@ 只做 @C = beta·C@（不引用 A）。
5. **返回值口径**：@uplo@/@trans@ 非法取 @INVALID_ENUM@（cuBLAS 为 @INVALID_VALUE@，此处遵 ops-blas 约定，
   属有意差异）；阶数/秩/前导维度类返回 @INVALID_VALUE@；类型组合不支持返回 @NOT_SUPPORTED@。
6. **3M 结果不与直算逐位一致**：3M 的中间 FP32 舍入与直接复数乘不同，精度按 §4.1 阈值判定，不要求 bit-exact。
7. **性能口径**：kernel-to-kernel —— NPU 侧 @msprof op@ 的 @Task Duration(us)@ 对比 GPU @cublasCsyrk3mEx@
   kernel-only 耗时；GTest @[PERF] avg_us@ 只作端到端参考，不与 @gpu_ms@ 混用。

## 设计阶段范围

本文覆盖算法分解（3M）、接口与异常码、Host/Kernel 实现方案、精度设计、性能实测与测试方案。
不含自测报告 xlsx、精度/性能原始日志与截图（随验收交付件单独提交）。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 @arch35@ 上实现 @aclblasCsyrk3mEx@：支持 @uplo@ 两侧三角、@trans@ 的 N/T/C、
@Atype ∈ {COMPLEX64, INT8}@、@Ctype = COMPLEX64@；主路径为同一次 launch 内的 3M 分解；
精度满足生态算子开源精度标准（FLOAT32 分量阈值）；性能满足单 case ≥ 0.4×、平均 ≥ 0.5× GPU kernel 标杆。

## 需求拆解

1. 新增公共接口声明与 Host 校验（handle/uplo/trans/n/k/lda/ldc/alpha/beta/Atype/Ctype/A/C，顺序见 §3.2.1）；
2. 3M 主路径 kernel 流水：反交织与 HF32 残差 → 对角直算 → 3M cube → combine；
3. INT8 输入的精确提升路径（复用 workspace，主路径不变）；
4. 抵消误差治理：HF32x3 补偿 + 对角直算（3M 的 @Im = P2-P0-P1@ 在对角完全相消）；
5. 精度验证：官方 1201 例 CSV（@mere@/@mare@ 口径）+ 自建 1202 例回归 + 100 例边界矩阵；
6. 性能验证：5 个参考 shape 的 kernel-only 实测与判据核算。

# 详细设计（required）

## 算子分析

### 数学公式

记 @X = op(A) = Xr + i·Xi@（@Atype=INT8@ 时先精确提升），则 @X·X^T@ 的实部/虚部分解为：

    3M：P0 = Xr·Xr^T，P1 = Xi·Xi^T，P2 = (Xr+Xi)·(Xr+Xi)^T
        Re(X·X^T) = P0 - P1
        Im(X·X^T) = P2 - P0 - P1
    直算：Re = Xr·Xr^T - Xi·Xi^T，Im = Xr·Xi^T + Xi·Xr^T

- 3M 只用 3 个实数乘积（4M 需 4 个），理论上省 1/4 立方运算；代价是第 3 个 quad 需额外的 @S = Xr+Xi@ 面板；
- **不得套用 Hermitian 分解**：若按 @X·X^H@ 计算，虚部会得到 @Xi·Xr^T - Xr·Xi^T@，而正确值是
  @Xi·Xr^T + Xr·Xi^T@，符号错误；
- 对角处 @Im(i,i) = 2·Σ(Xr·Xi)@，而 @P0@、@P1@、@P2@ 三项都是量级大得多的正定和，
  三者相减时发生强抵消（相对误差被 @|P2|/|Im|@ 放大），故对角单独直算（§3.2.2）。

### 支持数据类型和形状

| Atype | Ctype | 支持 | 说明 |
| --- | --- | --- | --- |
| @ACL_COMPLEX64@ | @ACL_COMPLEX64@ | 是 | 直接使用，8 字节/元素 |
| @ACL_INT8@ | @ACL_COMPLEX64@ | 是 | complex-int8，(int8 re, int8 im) 交错 2 字节/元素，Host 提升后计算 |
| 其它任意组合 | — | 否 | 返回 @NOT_SUPPORTED@（枚举本身非法则 @INVALID_ENUM@） |

形状：A 为 @n×k@（@trans=N@）或 @k×n@（@trans≠N@），C 为 @n×n@，均列主序；
@n@、@k@ 为运行时入参，Host 按实际取值 tiling，不要求 dynamic-shape 注册；@n=0@ 为合法空。

### 接口与参数

参数见 §1.2.3 表（13 参）。异常码与判定顺序（与 @csyrk3m_ex_host.cpp@ 一致，自 handle 起逐级短路）：

| 序号 | 条件 | 返回码 |
| --- | --- | --- |
| 1 | @handle == nullptr@ | @ACLBLAS_STATUS_HANDLE_IS_NULLPTR@ |
| 2 | @uplo@ 非 @UPPER@/@LOWER@ | @ACLBLAS_STATUS_INVALID_ENUM@ |
| 3 | @trans@ 非 @N@/@T@/@C@ | @ACLBLAS_STATUS_INVALID_ENUM@ |
| 4 | @n < 0@ 或 @k < 0@ | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 5 | @lda < max(1, trans==N ? n : k)@ | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 6 | @ldc < max(1, n)@ | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 7 | @alpha@ 或 @beta@ 为空 | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 8 | @Atype@/@Ctype@ 不在 @aclDataType@ 枚举内 | @ACLBLAS_STATUS_INVALID_ENUM@ |
| 9 | 类型组合非 (COMPLEX64,COMPLEX64) / (INT8,COMPLEX64) | @ACLBLAS_STATUS_NOT_SUPPORTED@ |
| 10 | @n>0 && k>0@ 且 @A@ 为空 | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 11 | @n>0@ 且 @C@ 为空 | @ACLBLAS_STATUS_INVALID_VALUE@ |
| 12 | @n == 0@ | @ACLBLAS_STATUS_SUCCESS@（no-op，不启动 kernel） |
| 13 | @k==0@ 或 @alpha==(0,0)@ | @ACLBLAS_STATUS_SUCCESS@；@beta!=(1,0)@ 时只缩放 C |

### 数据流

~~~text
A ──► deint_resid(AIV) ──► [pre-diag(AIV)] ──► 3m_cube(AIC, 单 launch) ──► combine(AIV) ──► C
      Ar|Ai|ArLow|AiLow|S|SLow    对对角直算(32 行批)    X|Y|Z 三 quad        Ci=Q2-Q0-Q1
~~~

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

1. **校验**：@csyrk3m_ex_host.cpp@ 按 §3.1.3 的 13 步顺序短路返回，结构合法后再做 quick return；
   @n=0@ 允许 A、C 为空，@k=0@ 或 @alpha=0@ 时 A 可为空但 @n>0@ 时 C 必须非空。
2. **INT8 提升**：@CsyrkPromoteInt8A@ 在 workspace 尾部申请 complex64 副本，按 (int8 re, int8 im) 交错解释后
   一次性精确提升（@aclrtMemcpyAsync@ + Device 侧展开），随后 @promotedA@ 交给共享 3M 流水，
   cube/combine 主路径对 Atype 无感知。
3. **tiling**：三份结构体 —— @CsyrkCubeTilingData@（tile 步长、@triangleMode@、@hf32@、@m128@、@isTransN@、@usedCoreNum@）、
   @CsyrkCombineTilingData@（@threeM@、@useDirectDiag@、@usePreDiag@、@isFastCfg@、@useSimt@、alpha/beta 系数）、
   对角直算 tiling；均由 Host 按 (n, k, uplo, trans, 核数) 生成并随下发传递。
4. **workspace 预算**：
   @ws = Nslab · arBytes + tempBytes + diagBytes + tail@，其中
   @arBytes = arLdc × arCols × 4@（@arLdc = align(arRows, 8)@）、
   @tempBytes = n × tempLdc × 4 × 4@（4 个 quad 槽，3M 实际用 X|Y|Z）、
   @Nslab = (hf32 ? 4 : 2) + (3M ? 2 : 0)@（3M 满配 6 个平面：@Ar|Ai|ArLow|AiLow|S|S_low@）。
   以 @n=k=2048@、@trans=N@ 为例：@arBytes@ = 16 MiB、6 平面 = 96 MiB、@tempBytes@ = 64 MiB、
   @diagBytes@ = 16 KiB，合计约 **160 MiB**；workspace 由 handle 持有并跨调用复用（只增不减），
   首次调用按实际需求扩容。
5. **流水**：同一 stream 顺序下发 3~4 个 kernel（@deint_resid@ → [@pre-diag@] → @3m_cube@ → @combine@），
   条件不满足时跳过 pre-diag，避免为不需要直算的形状付一次 launch。
6. **开关与回退**：@ACLBLAS_CSYRK_3M=1@ 让非 Ex @aclblasCsyrk@ 也走 3M（用于对照测试，Ex 接口无需该开关）；
   @ACLBLAS_CSYRK_M128=0/1@ 强制关闭/开启 128×64 大 tile 驱动；@ACLBLAS_CSYRK_PREDIAG@ 控制 pre-diag 路径。
   小规模（@n<=256 && k<=512@）走原生 fp32 cube（精确且更快）。

#### 3.2.2 kernel 侧设计

| kernel | 核心 | 职责 |
| --- | --- | --- |
| @csyrk_deint_resid_kernel@ | AIV | 一次读入复数 A，UB 内 @DeInterleave@，同批算出 @Ar@/@Ai@、@ArLow@/@AiLow@、@S@/@S_low@ 六个平面（每列单个大事务写回） |
| @csyrk_diag_direct_kernel@ | AIV | 按 32 行批直算对角：@Re = Σ(ar·ar - ai·ai)@、@Im = 2Σ(ar·ai)@，读 Ar/Ai 用 256B 事务而非 32B 短读 |
| @csyrk_3m_cube_kernel@ | AIC | 3 个 quad（X/Y/Z）沿 K 累加，**单次 launch**；L0C 3 累加器 × 2 组 ping-pong，drain 到 temp |
| @csyrk_combine_kernel@ | AIV | 内联 @Ci = Q2 - Q0 - Q1@（q2 原地复用，避免 scratch 竞态）、@Re = Q0 - Q1@ 缩放、对角覆盖、三角写回 |

3M cube 的关键设计：

- **两种 tile 驱动**：64×64（12 面板/48~64 K chunk，2 个 L1 buffer）与 128×64（24KB+12KB 面板，216KB×2 buffer，
  面板流量 −25%）；后者仅在 @trans=N@ 且三角形 tile 数在 28 个 AIC 核上均衡时启用
  （@tiles <= cores@，或 @ceil(tiles/cores) <= (tiles/cores)×1.10@），避免转置布局下 128×48 面板退化成
  128 次 192B 读的负优化；
- **HF32x3 补偿**：@x = xH + xL@，@xH = hf32(x)@（10 位尾数、fp32 指数域），每对面板做
  @xH·yH + xL·yH + xH·yL@ 三个 mmad，把有效尾数从 10 位恢复到约 30 位；首环 @initAcc@、末环 @unitFlag=3@；
- **S 面板融合**：resid pass 已读 Ar/Ai，顺带写 @S = Ar+Ai@、@S_low = S - hf32(S)@，省一次 GM 往返；
- **三角跳过**：按 tile 跳过非 uplo 侧，UPPER 跳过 @mStart > nStart + BM - 1@ 的 tile；combine 只处理有效区域，
  另一三角一个字节都不写。

3M cube 驱动伪代码（64×64 版）：

~~~cpp
for (tile : triangle_tiles_by_residue) {
  WaitFlag<FIX_M>(accParity);
  for (quad in {X, Y, Z}) {
    for (kChunk = 0; kChunk < K; kChunk += CK) {
      WaitFlag<MTE1_MTE2>(bufId);
      LoadL1(quad, kChunk, bufId);              // 4 面板：leftH/leftL/rightH/rightL
      SetFlag<MTE2_MTE1>(bufId); WaitFlag<MTE2_MTE1>(bufId);
      LoadL0(bufId);                            // L0A: H,L   L0B: H,L
      Mmad(xH·yH, xL·yH, xH·yL -> acc[quad]);   // 3 个 mmad
      SetFlag<MTE1_MTE2>(bufId);
    }
  }
  SetFlag<M_FIX>(parity); WaitFlag<M_FIX>(parity);
  Drain3Quads(acc, temp);                       // fixpipe -> temp[X|Y|Z]
  SetFlag<FIX_M>(parity);
}
~~~

实现要点（均为实测踩坑后固化）：

1. 向量化 HF32 残差：@hf32(x) = (bits + 0xfff + ((bits>>13)&1)) & 0xffffe000@；dav3510 的 @Ands@ 只支持 16 位，
   故用 @b - ((b>>1)<<1)@ 与 @(b>>13)<<13@ 表达，配 DataCopy 分块（4096 floats/tile），残差 pass 从 141us 降到 44us（n=2048）；
2. 对角覆盖屏障：标量 @SetValue@ 写对角与 PIPE_V 的 quad 公式之间必须加 @PipeBarrier<PIPE_ALL>@，
   否则偶发被向量结果覆盖（该 bug 曾导致全量回归每轮偶发 7~16 条失败，修复后稳定 1202/1202）；
3. pre-diag 门限：只在 @trans=N@、@n % 32 == 0@（内部行跨度无填充）且 combine 确实需要直算对角时启动，
   填充形状走 combine 内联的 8 行批路径，避免为省 99us 反而多付一次 launch。

### 设计取舍

| 取舍 | 结论 | 理由 |
| --- | --- | --- |
| 3M vs 4M | 取 3M | 乘积数 3 vs 4，@n=k=2048@ 时 cube 计算量降 25%；代价是 S 面板 + 抵消误差治理 |
| 对称 vs Hermitian 分解 | 对称（@X·X^T@） | 任务语义为对称；套 Hermitian 分解虚部符号错误 |
| 64×64 vs 128×64 tile | 两者并存，按核间均衡选择 | 128×64 流量 −25%，但转置布局面板退化，故仅 @trans=N@ 启用 |
| 对角：内联 vs 独立 kernel | 二者按形状择一 | 独立 32 行批 kernel 用 256B 事务，2048 上省约 99us；小 n/填充形状内联更划算 |
| 128×128 方形 tile | 否决 | 正确但慢 4.7~23%（L0C 单套 + 粒度粗） |
| 按列区间切两段做 AIV/AIC 重叠 | 否决 | 2048 只有 136 个 tile 单元，切两段后最慢核 6 轮 vs 5 轮，cube +20% |
| 2-pass cube（X/Y 成对 + Z） | 否决 | 精度错且慢 4%（L0C 占满 256KB） |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（@arch35@） | √ |

自验证覆盖一种款型（950PR）；@arch22@ 无 SIMT 且本算子依赖 @arch35@ 的 cube/hf32 能力，不在此列。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数校验层级 | Host 在入口按 §3.1.3 逐参数校验，不做 Device 元素级扫描；A/C 的数据内容是否满足对称约束由调用方保证 |
| 非连续 Tensor | 不涉及；仅通过 lda/ldc 描述列主序带填充布局，不使用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 不要求；n、k 为运行时入参，Host 按实际取值 tiling 与 launch |
| 原地语义 | C 原地覆写；仅 uplo 指定三角读写；调用方须保证 C 可写区与 A 可读区不重叠 |
| 确定性 | 不要求逐位确定；3M 的中间舍入与直算不同，同一输入两次执行不保证 bit-exact，按精度阈值判定 |
| 空 Tensor | @n=0@ 合法 no-op；@k=0@ 时 A 可无效；@beta!=(1,0)@ 仍需 @C = beta·C@ |
| 异步执行 | kernel 在 handle 绑定的 stream 上异步执行，Host 读回前须同步该 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 分量：@rtol = atol = 2^-13@（1.22e-4），@required_matched_ratio = 0.99@，@max_abs_error <= 1e-2@ 或 32 ULP | 生态算子开源精度标准（mixed_tolerance_standard） |
| 性能标准 | 单 case @<= gpu_ms / 0.4@（即倍率 ≥0.4×），全部性能 case 倍率算术平均 ≥0.5×；kernel-to-kernel 口径 | 任务书 §3.3 |

**精度设计与误差界**

- HF32 保留 10 位尾数、fp32 指数域；单次乘积相对误差约 @2^-11@，残差 pass 后 @xH + xL@ 有效尾数约 20 位，
  三段 mmad 补偿后有效尾数约 30 位，@k@ 段累加误差按 @sqrt(k)@ 增长；
- 3M 的 @Re = P0 - P1@ 与数学定义同构，无额外抵消；**@Im = P2 - P0 - P1@ 存在抵消**：
  对角处 @Im(i,i) = 2Σ(ar·ai)@ 而三个被减项各为 @Σ(ar±ai)^2@ 量级，误差被 @|P2|/|Im|@ 放大，
  故对角在 @k>256@ 时由 @csyrk_direct_kernel@ 直算覆盖；
- 实测（自建 CPU 对拍）：512²、1024²(@N@/@T@)、2048²、257×65 均 0 超差点，最大绝对误差 ≤ 3.6e-4；
- 官方口径：每例 CSV 带 @mere_threshold = 1.22e-4@ 与 @mare_multiplier = 10@，由 @verify_accuracy.py@
  驱动 GTest 逐例判定，实部/虚部分别比对。

## 性能标准/实测

环境：Ascend 950PR（A310-50-C00MM304A1，28 AIC / 56 AIV）、CANN 9.1.0、驱动 25.7.rc1.10。
数据口径：kernel-only（@msprof op@ @Task Duration@），warmup 5 + 有效采样 ≥10 次取平均，run-to-run 波动约 ±1%。

| n, k, uplo, trans | GPU 4M (us) | GPU 3M 基线 (us) | NPU 4M (us) | **NPU 3M (us)** | **r** |
| --- | --- | --- | --- | --- | --- |
| 512, 512, U, N | 50.509 | 53.5 | 61.7 | **49.9** | **1.072** |
| 1024, 1024, U, N | 153.065 | 162.1 | 227.1 | **221.4** | **0.732** |
| 2048, 2048, U, N | 788.030 | 834.8 | 1102.7 | **1100.4** | **0.759** |
| 2048, 2048, L, T | 850.791 | 901.3 | 1057.6 | **1108.2** | **0.813** |
| 1024, 256, U, C | 64.475 | 68.3 | 81.5 | **80.1** | **0.853** |
| 平均 | — | — | — | — | **0.846** |

- 判据：单 case ≥0.4（最小 **0.732** ✓）、平均 ≥0.5（**0.846** ✓）；
- GPU 3M 基线按 cuBLAS 3M/4M = 1/0.944 标定折算（@gpu3m = gpu4m / 0.944@）；
- 3M 在 512（49.9 vs 61.7）与 1024×256（80.1 vs 81.5）上**快于 4M**（128×64 tile 生效 + 9 vs 12 个 mmad），
  在 1024/2048 上慢 1%~5%（12 面板 L1 填充受限 + LOWER/T 的对角直算开销）。

**三段带宽核算（n=k=2048，说明为何已接近上限）**

| 阶段 | 流量 | 耗时 | 有效带宽 |
| --- | --- | --- | --- |
| 3m cube（12 面板/chunk，128×64） | 约 3.76 GB | 903 us | **4.16 TB/s** |
| deint_resid | 33.5 MB 读 + 100 MB 写 ≈ 134 MB | 73 us | **1.83 TB/s** |
| combine（含预计算对角） | temp 50 MB 读 + C 32 MB 写 ≈ 82 MB | 57 us | **1.44 TB/s** |

结论：**三段都已贴近显存/L2 带宽上限**（cube 占端到端 82%、deint 6.6%、combine 5.2%），
继续提速必须减少搬运字节数，而不是改流水或调参数。在不改 3M 算法、不降精度的前提下，
该 tile 形态已接近性能上限；已知剩余空间见 §3.4 的否决项与下方限制。

**已知限制**

1. cube 受 MTE2 带宽约束（@n=2048@ 时 mte2 占用约 97.5%、mac 约 53%），3M 必需的 S/S_low 面板使 L1
   放不下更大 tile 的双缓冲，故大 tile 只到 128×64；
2. combine 受事务率约束：列主序 temp 的 64 行读为 256B 事务（2048 时约 13.5 万事务），需改 temp 布局
   或更高 tile 才能再降；
3. 转置（@T@/@C@）形状在 M 侧为行主序，128×48 面板退化为 128 次 192B 读，故大 tile 仅用于 @trans=N@；
4. 固定开销约 20 us（4 次 kernel 启动 + tiling 组装），小 shape 上占比明显，已核查无法再压。

## 兼容性分析

- 新接口，不改动任何既有 API 的语义与 ABI；@include/cann_ops_blas.h@ 为**新增**声明，与既有声明无冲突；
- 非 Ex @aclblasCsyrk@ 保持 4M 行为不变（@ACLBLAS_CSYRK_3M=1@ 仅用于对照测试），两接口互不依赖；
- @aclblasComplex@ 以 @include/cann_ops_blas_common.h@ 定义为准（实部/虚部各 float32），与 cuBLAS @cuComplex@ 布局一致；
- 与 cuBLAS 的唯一有意差异：非法 @uplo@/@trans@ 返回 @INVALID_ENUM@ 而非 @INVALID_VALUE@（ops-blas 统一约定）。

## 测试设计与自测流程

| 用例集 | 规模 | 覆盖内容 | 结果 |
| --- | --- | --- | --- |
| 官方 CSV（@csyrk3m_ex_test.csv@） | **1201 例** | L0 基础/三角/转置、INT8（493 例）、负向异常码、no-op、规模档与边界 | @mere@/@mare@ 逐例判定 |
| 自建 GTest 回归 | **1202 例** | 3M 与 4M 双路径全量（@ACLBLAS_CSYRK_3M@ 对照）、CPU 对拍 | 3M **1202/1202**、4M **1202/1202** |
| 边界矩阵 | 100 例 | n/k ∈ {0,1,7,8,31,32,63,64,127,128,255,256,257}、填充 lda/ldc、非 8 倍数 n、极端 alpha/beta | PASS |
| 负向用例 | 随覆盖集 | 空 handle、非法枚举、非法 lda/ldc、空指针、类型组合不支持 | 返回码符合 §3.1.3 |

复现方式：

~~~bash
bash build.sh --soc=ascend950 --ops=csyrk                 # 构建
./build/test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test --gtest_brief=1   # 官方 CSV 驱动
./build/test/syrk/csyrk/csyrk_test --gtest_brief=1                     # 自建回归（4M）
ACLBLAS_CSYRK_3M=1 ./build/test/syrk/csyrk/csyrk_test --gtest_brief=1  # 3M 对照
msprof op --application="./build/test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test --gtest_filter=*TC_PF*" \
          --output=./prof_csyrk3m_ex                             # kernel-only 计时
~~~

## 设计评审与代码交付关系

本 PR 只提交设计文档。算子代码、测试代码、CSV 与自测报告在 @ops-blas@ 代码 PR 中提交：

| 序号 | 交付件 | 位置/说明 | 状态 |
| --- | --- | --- | --- |
| 1 | 算子设计文档 | 本文 | 本次提交 |
| 2 | 算子实现代码 | @blas/syrk/arch35/@：@csyrk3m_ex_host.cpp@、@csyrk3m_ex_bridge.h@、@csyrk_3m_cube_kernel.cpp@、@csyrk_3m_gemm_fused.h@、@csyrk_3m_m128.h@、@csyrk_direct_kernel.cpp@、@csyrk_host.cpp@（共享流水与 workspace） | 随代码 PR |
| 3 | 公开声明 | @include/cann_ops_blas.h@ 新增 @aclblasCsyrk3mEx@ | 随代码 PR |
| 4 | 算子说明 | @blas/syrk/README.md@ 的 @aclblasCsyrk3mEx@ 章节（含产品支持情况） | 随代码 PR |
| 5 | 测试代码与用例 | @test/syrk/csyrk3m_ex/arch35/@（CSV 1201 例 + GTest 驱动 + golden） | 随代码 PR |
| 6 | 自测报告 | 精度/性能/内存三份 xlsx + 日志，按模板提交 | 验收时提交 |
| 7 | 性能原始数据 | 5 参考 shape 的 @msprof@ 采集、三段带宽核算、优化历程记录 | 验收时提交 |
| 8 | 待验收代码地址 | 个人仓分支 + 算子目录，邀请 @Ascend-CANN@ 为开发者 | 验收时提交 |
| 9 | @task_submission@ 目录 | 按模板组织（自验证步骤说明、精度/性能/内存报告与日志） | 验收时提交 |

## 参考资料

1. cuBLAS 文档 @cublasCsyrk3mEx@：https://docs.nvidia.com/cuda/cublas/index.html
2. Netlib BLAS @csyrk.f@：https://www.netlib.org/blas/csyrk.f
3. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
