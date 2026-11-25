# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/gemm/arch22/`
> 任务书：`aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— gemm 类 3 个算子开发（A2A3）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/gemm/arch22/`；测试 `test/gemm/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

三个接口共用同一数学定义 `C = alpha * op(A) * op(B) + beta * C`（列主序，op(A) 为 m×k、op(B) 为 k×n、C 为 m×n），仅数据类型不同。

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasSgemm | cuBLAS `cublasSgemm`｜Netlib `sgemm.f` | FLOAT32 通用矩阵乘 |
| aclblasCgemm | cuBLAS `cublasCgemm`｜Netlib `cgemm.f` | COMPLEX64 通用矩阵乘（直接复数乘加，**不走 3M 分解**） |
| aclblasHgemm | cuBLAS `cublasHgemm`（Netlib 无 half 参考） | FLOAT16 存储的通用矩阵乘 |

与同族 `aclblasCgemm3m` 的区别：参数序列相同，但后者用 3 次实数乘分解复数乘、位于 `blas/gemm/arch35/cgemm3m_host.cpp`；本任务的 Cgemm 是 4 乘法的直接复数乘加，二者实现不混用（任务书 §2.2）。

### 业界对标

| 项 | cuBLAS | 本算子 |
|---|---|---|
| 参数序列 | handle/transa/transb/m/n/k/alpha/A/lda/B/ldb/beta/C/ldc | 逐参数一致（§2.3、§3.5） |
| trans 取值 | N / T / C | 同。Cgemm 中 C 为共轭转置、结果与 T 不同；Sgemm/Hgemm 为实数，C≡T 不取共轭 |
| alpha/beta 位置 | 随 pointer mode 可为 Host 或 Device | 约定指向 Host 内存（§2.4） |
| 非法 trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 执行模型 | stream 异步 | 同：kernel 在 handle 绑定的 stream 上异步执行，读回前须同步 |

### 工程形态

- 工程模式：Kernel 直调——Host C API 完成参数校验、分派与 tiling，经 handle 绑定的 stream 直调 NPU kernel（§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 接口 | arch22 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| aclblasSgemm | 已合入 | `blas/gemm/arch22/gemm_host.cpp`、`gemm_kernel.cpp`、`gemm_kernel.h`；提交 `b912973`（2026-10-21） | 按任务书口径对齐校验顺序并修复已知缺陷，测试扩展 |
| aclblasCgemm | 已合入 | `blas/gemm/arch22/cgemm_host.cpp`、`cgemm_kernel.cpp`、`cgemm_kernel.h`、`cgemm_tiling_data.h`；提交 `5ebfa3c`（2026-11-24） | 按任务书口径对齐，测试并入通用测试工程 |
| aclblasHgemm | 不存在 | `include/cann_ops_blas.h` 与 `blas/` 下均无该入口（grep 核验） | **新增声明与实现（主要工作量）** |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm/arch22/gemm_*`（Sgemm arch22） | ★ 主模板 | Cube（Matmul 高阶 API）矩阵乘、宿主分块与工作区规划、AIV 尾声缩放链；Cgemm 直接委托其入口 | 校验顺序对齐；修复 K>1024 缺陷（见 host 侧设计） |
| `blas/gemm/arch22/cgemm_*`（Cgemm arch22） | ★ 已合入 | 拆实虚平面 → 4 次实数 GEMM → 复数组装三阶段 | 仅做口径对齐 |
| `blas/gemm/arch35/hgemmStridedBatched_*` | ◇ 参考 | FP16 计算路径：half 面板 ND→NZ 搬入 L1、LoadData2D 进 L0、FP32 累加 | arch35 为指令级流水且带 batch 维，只参考 dtype 语义，不复用代码 |
| `blas/gemm/arch35/gemm_host.cpp`、`cgemm_host.cpp` | ◇ 参考 | Sgemm/Cgemm 的 arch35 实现说明与校验口径 | 不复制 arch35 代码 |

### 接口规格

`aclblasSgemm`（`include/cann_ops_blas.h` L575）与 `aclblasCgemm`（L579）声明已存在，无需修改；`aclblasHgemm` 需新增一条声明，紧随 `aclblasCgemm` 之后，参数序列与 `cublasHgemm` 逐参数对应（§2.3）。三者签名：

```cpp
aclblasStatus_t aclblasSgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc);

// 新增声明
aclblasStatus_t aclblasHgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha, const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb, const aclblasHalf* beta, aclblasHalf* C, int ldc);
```

`aclblasHalf` 为 16-bit FP16 位模式类型（`typedef uint16_t aclblasHalf`，`include/cann_ops_blas_common.h` L74，`include/cann_ops_blas.h` L28 为同一定义）；`aclblasComplex` 为实部/虚部各 float32 的结构体（`cann_ops_blas_common.h` L77）。

参数规格（逐行对齐 §2.4；S=FLOAT32、C=COMPLEX64、H=FLOAT16，alpha/beta 与同算子 A/B/C 同类型）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| transa | op(A) 运算类型 | attr | int | N / T / C；其它 → `ACLBLAS_STATUS_INVALID_ENUM` | — |
| transb | op(B) 运算类型 | attr | int | 同上 | — |
| m | op(A) 与 C 的行数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |
| n | op(B) 与 C 的列数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |
| k | 缩并维长度 | attr | int | ≥0；负 → `INVALID_VALUE`；0 只做 `C = beta*C` | — |
| alpha | 标量乘数（Host 内存） | scalar | S=float / C=aclblasComplex / H=aclblasHalf | 非空；空 → `INVALID_VALUE` | 标量 |
| A | 输入矩阵 A（列主序，Device） | tensor | 同 alpha | m、n、k>0 且 alpha≠0 时非空，空 → `INVALID_VALUE` | 物理 lda×k（N）或 lda×m（T/C） |
| lda | A 前导维 | attr | int | N：≥max(1,m)；T/C：≥max(1,k)；否则 `INVALID_VALUE` | 标量 |
| B | 输入矩阵 B（列主序，Device） | tensor | 同 alpha | 非空条件同 A | 物理 ldb×n（N）或 ldb×k（T/C） |
| ldb | B 前导维 | attr | int | N：≥max(1,k)；T/C：≥max(1,n)；否则 `INVALID_VALUE` | 标量 |
| beta | 标量乘数（Host 内存） | scalar | 同 alpha | 非空；空 → `INVALID_VALUE`；=0 时不读 C 旧值 | 标量 |
| C | 输出矩阵 C（列主序，Device，原地覆写） | tensor | 同 alpha | m>0 且 n>0 时非空，空 → `INVALID_VALUE` | 逻辑 m×n；物理 ldc×n |
| ldc | C 前导维 | attr | int | ≥max(1,m)；否则 `INVALID_VALUE`；padding 列不被修改 | 标量 |

### 数学公式

逐元素（i∈[0,m)、j∈[0,n)、p∈[0,k)）：

```text
C(i,j) = alpha · Σ_p op(A)(i,p) · op(B)(p,j) + beta · C(i,j)
op = X（N）/ X^T（T）/ X^H（C，仅 Cgemm 取共轭；Sgemm/Hgemm 的 C 等价 T）
```

Cgemm 的复数乘加直接计算：`(a+bi)(c+di) = (ac−bd) + (ad+bc)i`，alpha、beta 亦为复数（§2.1）。Hgemm 以 FP16 存储输入输出，乘加在 FP32 进行、仅最终输出收窄到 FP16（口径见 §3.2 第 1 条）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上提供三个句柄式 BLAS 接口：`aclblasSgemm`（FLOAT32）、`aclblasCgemm`（COMPLEX64）、`aclblasHgemm`（FLOAT16），支持 trans 3×3 全组合、任意合法前导维与运行时形状（用例覆盖 m/n/k ≤ 4096）；精度达生态算子开源混合容差标准（S/C 按 FLOAT32 档、H 按 FLOAT16 档），性能达 GPU 标杆的 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | `C = alpha·op(A)·op(B) + beta·C`。Cgemm 拆实虚平面后化为 4 个实数 GEMM，直接委托已合入的 arch22 `aclblasSgemm`；Hgemm 与 Sgemm 数学同构，仅 dtype 不同 |
| 2 | 接口原型 | §2.3 三条签名（见"接口规格"），返回 `aclblasStatus_t` |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | Sgemm：FLOAT32；Cgemm：COMPLEX64；Hgemm：FLOAT16。不做混合精度输入 |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；950（arch35）不在本任务范围 |
| 6 | 精度要求 | §3.2：S/C 的分量 rtol=atol=2^-13、H 为 2^-10；matched_ratio≥0.99 且 max_abs_error≤限值；golden S/C 用 cblas、H 用 FP32（float64 累加）收窄到 FP16 |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 次 + 10 次有效样本取均值 |
| 8 | 代码架构选型 + 可行性 | Cube（Matmul 高阶 API）+ AIV 尾声。Sgemm/Cgemm 复用已合入实现并按任务书对齐；Hgemm 新增 FP16 Cube 路径，**已排除**"上转 FP32 后委托 Sgemm"（见"性能优化要点"）。已知缺陷 ops-blas#1077 必须先修。NPU 侧能否达标均**待实测** |
| 9 | 评测来源 | 任务包 `test_cases/{sgemm,cgemm,hgemm}/`：精度 1000 条（334/333/333）+ 性能 200 条（67/67/66），`gpu_baseline.csv` 的 `gpu_ms` 已全部回填 |

# 详细设计

## 算子分析

### 数学公式

- Sgemm / Hgemm：实数乘加，`C(i,j) = alpha·Σ_p A'(i,p)·B'(p,j) + beta·C(i,j)`，A'、B' 为按 trans 取原矩阵或其转置。
- Cgemm：A = Ar + i·Ai，B = Br + i·Bi，记 s_a、s_b 为共轭操作数取 −1、否则 +1 的符号（已合入 `cgemm_host.cpp` L25-32 的形式）：

```text
Re(op(A)·op(B)) = Ar·Br − (s_a·s_b)·Ai·Bi
Im(op(A)·op(B)) = s_b·Ar·Bi + s_a·Ai·Br
Cr = αr·Re − αi·Im + βr·Cr_old − βi·Ci_old
Ci = αr·Im + αi·Re + βr·Ci_old + βi·Cr_old
```

共轭只体现在三个标量符号上，拆分阶段不区分 T 与 C。

### 支持数据类型

| 接口 | A/B/C | alpha/beta | 计算精度 | 元素大小 |
|---|---|---|---|---|
| aclblasSgemm | float | float | FP32（Cube 关闭 HF32 截断，`gemm_kernel.cpp` L113） | 4 B |
| aclblasCgemm | aclblasComplex（实部/虚部各 float） | aclblasComplex | FP32（4 个实数 GEMM + FP32 组装） | 8 B |
| aclblasHgemm | aclblasHalf | aclblasHalf | FP32 累加，输出收窄 FP16 | 2 B |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m、n | ≥0 | m=0 或 n=0：**结构校验通过后**直接返回 SUCCESS，不引用 A/B/C、不启动 kernel |
| k | ≥0 | k=0 或 alpha=0：只做 `C = beta*C`，不引用 A/B（允许为空）；beta=0 时 C 可为未初始化 |
| lda / ldb / ldc | 见参数表下界 | 低于下界 → INVALID_VALUE；允许 padding |
| OP_C | Sgemm/Hgemm 为实数 | 等价 OP_T，不做共轭 |

C 仅写逻辑 m×n 区域，`ldc` 的 padding 列保持不变；C 与 A/B 存储不得重叠（由调用方保证，§2.1）。

## 算子实现

### 实现方案

#### 总体架构与数据流

内部统一按 `C^T = op(B)^T · op(A)^T` 以行主序喂 Cube（已合入 `gemm_kernel.cpp` L17-18、L105-107 的做法），Sgemm 与 Hgemm 共用同一套宿主分派：

| 路径 | 触发条件（host 判定） | kernel 链 |
|---|---|---|
| 大方阵整问题 | 2048 ≤ m,n,k ≤ 8192，且 m%16==0、n%16==0、k%32==0 | 紧凑 NN 时输入直读（nosplit）；否则 AIV pack → Cube 方阵 kernel（nn/nt/tn/tt 四选一）→ AIV 尾声 |
| 整问题直算 | m,n,k ≤ 1024，m/n 按 16、k 按 8 对齐且 lda/ldb 紧凑 | Cube 行属主 kernel（按 trans 四选一）；alpha=1、beta=0、ldc=m 时乘积直写 C，否则加 AIV 尾声 |
| 整问题打包 | m,n,k ≤ 1024 的其余形状 | AIV pack → Cube → AIV 尾声 |
| 二维分块 | NN 且 n ≤ 128、m ≥ 512、k ≥ 512 | Cube 分块 kernel → AIV 散写 |
| 宿主分块 | 其余 | 以 1024×1024×1024 为宿主 tile：pack → K 向逐块 Cube → AIV 累加 → AIV 尾声 |
| 仅缩放 | k=0 或 alpha=0 | beta=0：清零逻辑区域；否则 `C = beta*C`。Sgemm 现为清零 kernel + 逐列 `aclblasSscal`；Cgemm 走组装 kernel 的无乘积分支；Hgemm 用自带的 half 清零/缩放 kernel（arch22 的 `blas/scal/` 仅有 sscal、cscal） |

数据流（Sgemm / Hgemm 主路径）：

```text
GM: A, B ──[AIV pack：仅当操作数不能被 Cube 直接消费]──> GM: packA / packB
              │
              └─[Cube：Matmul 高阶 API，FP32 乘积]──> GM: product ──[AIV accumulate：K 向多块时]──> GM: accum
                                                                                              │
GM: C(旧值) ────────────[AIV epilogue：alpha·P + beta·C，仅写逻辑 m×n]────────────────────────────┘──> GM: C
```

Cgemm（已合入）在其之上多一层编排：AIV 拆实虚平面 → 4 次实数 GEMM（子句柄委托 `aclblasSgemm`）→ AIV 复数组装，直接写 C。

Hgemm 的 FP16 处理（新增）：

```text
GM: A(half), B(half) ──[AIV pack，half]──> Cube：half×half，FP32 累加，乘积以 FP32 写 GM
GM: C(旧值, half) ─┐
GM: product(float) ┴> [AIV epilogue：C_old 转 float，按 FP32 计算 alpha·P + beta·C_old，收窄 half] ──> GM: C(half)
```

乘积以 FP32 落 GM 而不直接输出 FP16，原因有二：① 缩放前的部分和可能超出 FP16 范围而最终结果有限（如 alpha 很小）；② 避免缩放前的二次舍入，保证只做一次 FP16 舍入，与 golden 口径一致。

工作区布局（Hgemm，自低址向高址，各段起始 32B 对齐）：

| 区段 | 大小 | 说明 |
|---|---|---|
| packA | pk × pm × 2 B | 仅 pack 路径使用；pm 为 m 对齐 16、pk 为 k 对齐 8 后的值 |
| packB | pn × pk × 2 B | 同上；pn 为 n 对齐 16 后的值 |
| product | pn × pm × 4 B | 当前宿主 K 块的 FP32 乘积（内部 C^T 布局，行主序） |
| accum | 同 product | K 向多块时累加；单块时复用 product 地址，不额外占用 |

工作区经句柄默认工作区申请，超限返回 `ACLBLAS_STATUS_ALLOC_FAILED`。Cgemm 工作区约为 `2*(m*k + k*n + 2*m*n)` 个 float 加 4 MiB 余量（`blas/gemm/README.md` L354）。

#### host 侧设计

**分核策略**：
- Cube 侧核数取 AIC 核数。行属主 kernel 每核负责内部 C^T 的一段连续输出行：先按 64 行的单核包络求最少块数，再向上补齐到整波；块数与每核行数由 host 与 kernel 用同一公式推出（已合入 `gemm_kernel.cpp` L47-59）。
- 宿主 tile 固定 1024×1024×1024，n 向外、m 向内循环；每轮重算 pm/pn（对齐 16）与 pk（对齐 8）。
- AIV 侧 pack / accumulate / epilogue / 清零的块数取 AIV 核数与元素数的较小者。

**数据分块和内存优化策略**：
- Cube 静态 tiling：Sgemm 现状为单核 64 行包络、baseK=64（`gemm_kernel.cpp` L28 的 `kRowsShape`，`gemm_host.cpp` L30 注释）。Hgemm 的 L1/L0 按元素计容量是 FP32 的两倍，tiling 形状参数按 half 重新推导——**待上板核对** `GetMatmulApiTiling` 的生成结果。
- 能被 Cube 直接消费的操作数不打包：紧凑 NN 走 DMA 直读；转置且对齐的形状保留原始物理方向，由对应行属主 kernel 解释转置。
- 仅单个宿主 K 块时尾声直接读 product，省一次 accumulate。

**tilingkey 规划策略**：仓内 arch22 gemm 不使用 tilingkey 宏分支，分派在 host 侧以显式 kernel 选择完成，参数经 `GemmArch22Tiling` 结构体传入 kernel。Hgemm 沿用同一风格：

| 字段 / 条件 | 取值 | 分支含义 |
|---|---|---|
| `transA`、`transB` | 0 / 1（C 归并为 T） | 选择四个行属主 / 方阵 kernel 之一 |
| `rawPack` | 0 / 1 | 原始转置打包或通用地址映射打包 |
| `alpha`、`beta` | 标量值内联 | 尾声直接使用，不依赖 Device 标量 |
| 宿主 K 块数 | 1 / 多 | 单块直写 C，或多块经 accumulate |
| `k==0 ∨ alpha==0` | — | 进入仅缩放路径，不启动 Cube |

**数据检测**（校验顺序逐条对齐 §2.4/§2.5，三接口一致）：

1. handle 非空，否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. transa、transb ∈ {N, T, C}，否则 `ACLBLAS_STATUS_INVALID_ENUM`
3. m、n、k ≥ 0，否则 `ACLBLAS_STATUS_INVALID_VALUE`
4. lda ≥ max(1, transa=N ? m : k)、ldb ≥ max(1, transb=N ? k : n)、ldc ≥ max(1, m)，否则 `INVALID_VALUE`
5. alpha、beta 非空，否则 `INVALID_VALUE`
6. 结构全部合法后判 quick return：m=0 或 n=0 → SUCCESS（此时 A/B/C 可为空）
7. m>0 且 n>0 时 C 非空，否则 `INVALID_VALUE`
8. k>0 且 alpha≠0 时 A、B 非空，否则 `INVALID_VALUE`（alpha 的值在标量读回后判定）
9. k=0 或 alpha=0 → 仅 `C = beta*C`，不引用 A/B；beta=0 时不读 C 旧值

已合入实现与上述顺序的两处偏差（源码核验，本任务修正）：

| 位置 | 现状 | 偏离条款 |
|---|---|---|
| `gemm_host.cpp` L292（`ValidateSgemm`） | `m==0` 或 `n==0` 的 quick return 位于前导维校验（L293）之前 | §2.5"结构合法后再执行 quick return"：m=0 且 lda 低于下界时应返回 `INVALID_VALUE` |
| `cgemm_host.cpp` L185（`ValidateCgemmPointers`） | 只要 k>0 就要求 A/B 非空，未区分 alpha=0 | §2.5：alpha=0 时允许 A/B 为 nullptr |

**已知缺陷 ops-blas#1077（本任务必须修复）**：arch22 实数入口在 K>1024 且 M、N 均低于整问题门限时计算结果错误，910B3 实测 512×512×1536 有 262123/262144 个元素错误、maxAbsErr 273（`blas/gemm/README.md` L356、`cgemm_host.cpp` L62-69）。master `55ef791` 上未见修复提交（`git log --grep=1077` 仅命中 Cgemm 的绕行注释），Cgemm 现以 K 按 1024 分块绕行。本任务的精度与性能用例含大量此类形状（如 63×63×4095），Sgemm 与 Hgemm 都经过同一 K 向累加路径，须在精度阶段定位并修复根因，修复后再移除 Cgemm 的 K 分块绕行。具体受影响的分派子路径**待上板复现定位**。

#### kernel 侧设计

Hgemm 的 kernel 链按 Init / Process 两阶段组织，与 Sgemm 同构：

```text
[Cube] Init    ：解析 tiling → 求本核输出行窗口 → 绑定 left / right（half）与 output（float）
       Process ：Matmul 迭代，FP32 乘积写到 GM product
[AIV]  Init    ：解析 tiling → 求本核元素区间 → 队列初始化
       Process ：CopyIn（product：float；C_old：half）
                → Compute（half 转 float，alpha·P + beta·C_old）
                → CopyOut（收窄 half，掩码只写逻辑 m×n，ldc padding 列不写）
```

文件落位（与同目录 `cgemm_*` 四件套同构）：

| 文件 | 内容 |
|---|---|
| `blas/gemm/arch22/hgemm_host.cpp` | 公开入口 `aclblasHgemm`：校验、分派、工作区、发射 |
| `blas/gemm/arch22/hgemm_kernel.cpp`、`hgemm_kernel.h` | FP16 的 pack / Cube 行属主 / 累加 / 尾声 / 清零缩放 kernel 及其 `_do` 发射封装 |
| `blas/gemm/arch22/gemm_host_common.h`（新增，提取自 `gemm_host.cpp` 的匿名空间） | 三接口共用的校验、分派判定、工作区布局；Sgemm 行为不变，以既有 `gemm_test` 全量回归验证 |

Cgemm 与 Sgemm 不新增 kernel，只做上表两处校验对齐与 #1077 修复。

#### 性能优化要点

本文未上板，以下为设计依据与规划，**实测数据在性能阶段以 `msprof op` 回填**。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms` / 0.8，括号内为 `gpu_ms`）：

| 代表 case（NN，alpha=1，beta=0） | Sgemm | Cgemm | Hgemm |
|---|---|---|---|
| 4096³ | 11.03 ms（8.82） | 36.41 ms（29.13） | 0.77 ms（0.62） |
| 4095³ | 9.06 ms（7.25） | 35.89 ms（28.71） | 2.10 ms（1.68） |
| 4096×64×4096 | 0.225 ms（0.180） | 0.636 ms（0.509） | 0.043 ms（0.034） |
| 64×64×4096 | 27.0 µs（21.6） | 34.3 µs（27.4） | 20.6 µs（16.5） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| Hgemm 走 FP16 Cube，而非上转 FP32 后委托 Sgemm | Hgemm 4096³ 的允许耗时 0.77 ms 仅为 Sgemm 同形状允许耗时 11.03 ms 的 7%；委托路线耗时≈Sgemm 耗时，量级上不可达 | 设计定案（排除委托路线） |
| 大方阵整问题单次发射 | 避免宿主分块带来的多次发射与 GM 往返（Sgemm 现有设计，Hgemm 沿用） | 沿用，待实测确认门限 |
| 小形状缩短 kernel 链 | 64×64×4096 的允许耗时仅 21~34 µs，pack → Cube → 累加 → 尾声的多 kernel 链发射开销占比高 | 待 `msprof` 归因后决定是否合并 kernel |
| Cgemm 的 4 次实数 GEMM 压缩 | 4096³ 允许耗时 36.41 ms，摊到 4 次实数 GEMM 每次约 9.1 ms，而 GPU 的 Sgemm 4096³ 为 8.82 ms——要求 NPU 单次实数 GEMM 不慢于 A100 同形状 | 风险项，首轮性能实测裁决 |
| 移除 Cgemm 的 K 分块绕行 | #1077 修复后，K=4096 时实数 GEMM 调用数降为原来的 1/4 | 依赖 #1077 修复 |
| 未采用：Hgemm 乘积直接输出 FP16 | 缩放前舍入与溢出风险（见"总体架构与数据流"） | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| Ascend 950PR / 950DT（arch35） | 不在本任务范围（arch35 另有 Sgemm/Cgemm 实现） |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：全部在 host 入口按上表顺序校验并返回状态码，不对 Device 张量做元素级扫描；数据有效性由调用方保证。
- 非连续 Tensor：不支持任意非连续布局，仅支持 lda/ldb/ldc 描述的列主序带填充布局；不使用 AutoContiguous/ViewCopy，不产生相关临时空间。
- broadcast：不支持，A/B/C 为独立操作数。
- dynamic shape：不要求；m/n/k/lda/ldb/ldc 与 trans 均为运行时入参，host 按实际取值 tiling 与发射。
- 原地与视图语义：C 原地覆写，接口不返回视图；C 的可写区域与 A/B 的可读区域不得重叠。
- 确定性计算：不要求；多核切分与 K 向归约顺序由实现按形状选择，同一输入两次执行不保证逐位一致。
- 空 Tensor：m=0 或 n=0 为合法 no-op；k=0 或 alpha=0 只做 `C = beta*C`。
- 0 维 Tensor：不涉及；矩阵以指针配合 m/n/k 与前导维描述。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行，接口返回不代表计算完成。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。Sgemm 与 Cgemm（实部、虚部分别统计）取 FLOAT32 档：rtol=atol=2^-13（1.2e-4），max_abs_error 限值 1e-2 或 32 ULP；Hgemm 取 FLOAT16 档：rtol=atol=2^-10（1e-3），限值 1e-1 或 32 ULP | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | Sgemm：`cblas_sgemm`；Cgemm：`cblas_cgemm`；均按 float64 累加后收窄到 float32。Hgemm：不用 `cblas_hgemm`，由测试工程按 FP32（float64 累加）计算后收窄到 FP16 | 任务书 §3.1、§3.2 第 1、4 条 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8（即达到 GPU 的 0.8 倍） | 任务书 §3.3 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均。GTest 的 `[PERF] avg_us` 事件计时仅作交叉验证 | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

用例来自任务包 `test_cases/`（`gen_csv.py` 固定随机种子生成，逐条可复现），精度合计 1000 条、性能合计 200 条：

| 类别 | 前缀 | Sgemm | Cgemm | Hgemm | 覆盖 |
|---|---|---|---|---|---|
| L0 基础 | TC_L0 | 18 | 18 | 18 | trans 3×3 全组合 × 小尺寸 |
| 尺寸扫描 | TC_MN | 52 | 52 | 52 | 1→4096 方阵（含 4095³/4096³）与矩形 |
| 标量 | TC_AB | 24 | 23 | 23 | alpha/beta 特殊值与随机值 |
| 前导维 | TC_LD | 15 | 15 | 15 | 紧凑与 padding |
| 填充值域 | TC_FL | 21 | 21 | 21 | 均匀 / 正态 / 全零 / 交替 / 极端 / Inf / NaN |
| 中等覆盖 | TC_CV | 16 | 16 | 16 | 中等尺寸 |
| 边界负向 | TC_ED | 41 | 43 | 49 | m/n/k=0、alpha=0、beta=0、空指针、非法枚举、负维度、OP_C 对照 |
| 扩展精度 | TC_EX | 147 | 145 | 139 | 尺寸 × 转置 × 标量 × 前导维采样 |
| 精度小计 | | 334 | 333 | 333 | 合计 1000 |
| 性能 | TC_PF | 67 | 67 | 66 | 合计 200；含 9 种转置组合、4095³/4096³、非默认 alpha/beta |

- 测试工程：扩展仓内通用 `test/gemm/arch22/gemm_test.cpp`，按 CSV 的 `func` 列分派到 `sgemm` / `cgemm` / `hgemm`；列解析兼容任务包 CSV 的列名（`transA`/`transB` 写枚举全名、`alpha_null` 等空指针列）。任务包三个脚本的用例都安装到 `test/gemm/arch22/gemm_test.csv`、由同一个 `gemm_test` 加载；不新建 `test/gemm/sgemm/arch22/`。
- 性能采样：GTest 在预置 Device 缓冲上直接调用，warmup 5 次后用 `aclrtEvent` 夹 10 次取 `avg_us`，并按用例打印 `[PERF] case=<case_name> ... avg_us=<数值>`；正式数据另用 `msprof op` 采集。
- 复现入口：`bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`。按 `build.sh` 的族名解析，`sgemm`、`hgemm` 落到 `test/gemm`，`cgemm` 落到 `test/gemm/cgemm`（已合入的独立 `cgemm_test`）；任务包脚本 `verify_accuracy.py` / `verify_performance.py` 加载上述 CSV 运行。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h`：仅**新增** `aclblasHgemm` 一条声明（任务书 §2.3 明确要求），不改动既有声明与 ABI；声明对全产品线可见，属仓内既有惯例。
- 对已合入 Sgemm / Cgemm 的改动只涉及两处校验顺序与 #1077 修复：合法调用的数值结果不变；返回码变化的仅有"m=0 或 n=0 且前导维非法"（SUCCESS → `INVALID_VALUE`）与"alpha=0 且 A/B 为空"（`INVALID_VALUE` → SUCCESS）两类负向场景，以对齐任务书为目的，用 `TC_ED` 用例回归。
- `blas/gemm/README.md`：新增 `aclblasHgemm` 章节；`aclblasSgemm` 章节的产品支持情况现为"Atlas A2/A3 不支持"（L69-71），与 arch22 实现已合入的事实不符，随本任务更正；`aclblasCgemm` 章节现写"Atlas A3 不支持"（L335），以任务书 §3.1 的 A2/A3 为准一并更正。
- 与 `aclblasCgemm3m` 无代码交叉。#1077 的修复位于 `blas/gemm/arch22/` 内部的 K 向累加路径，`cgemm_host.cpp` 以子句柄委托 `aclblasSgemm`，修复后其 K 分块绕行方可移除。
