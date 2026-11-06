# aclblasGemmEx / aclblasSgemmEx / aclblasCgemmEx 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 队名 | `jyk2026` |
| 任务批次 | 2026 年 10 月社区任务 |
| 任务名称 | 10月社区任务-aclblasGemmEx_SgemmEx_CgemmEx算子开发（A2A3） |
| 任务书 | `aclblasGemmEx_SgemmEx_CgemmEx_A2_task_doc.md` |
| 目标仓库 | [ops-blas](https://gitcode.com/cann/ops-blas) |
| 设计基线 | `master` @ `4347ecb`（2026-11-03） |
| 目标硬件 | Atlas A2 / Atlas A3（arch22，SOC 例 `ascend910b3`） |
| 合入路径 | `blas/gemm_ex/arch22/`（实现）、`test/gemm_ex/arch22/`（测试）、`include/cann_ops_blas.h`（声明） |
| 日期 | 2026-11-03 |

---

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年 10 月算子开发任务，要求在昇腾 NPU（Atlas A2/A3 系列）上使用 Ascend C 编程语言，基于 ops-blas 开源仓工程框架，开发三个扩展版通用矩阵乘法句柄式接口：

- `aclblasGemmEx`（19 参，对标 `cublasGemmEx`）
- `aclblasSgemmEx`（17 参，对标 `cublasSgemmEx`）
- `aclblasCgemmEx`（17 参，对标 `cublasCgemmEx`）

参数口径一律以 cuBLAS 官方文档为准，仅做昇腾侧类型名映射（`cublasHandle_t → aclblasHandle_t`、`cudaDataType_t → aclDataType`、`cuComplex → aclblasComplex` 等）。三个接口共用同一数学定义与边界语义，差异集中在标量类型与是否存在 `computeType`/`algo` 形参。

## 背景介绍

### 仓内现状（源码实测）

对 `ops-blas` master `4347ecb` 逐文件核验，GemmEx 族在仓内的现状如下：

| 接口 | 声明（`include/cann_ops_blas.h`） | 实现 | 与任务书 §2.3 的差异 |
| --- | --- | --- | --- |
| `aclblasGemmEx` | 19 参，`const void* alpha/beta` + `Atype/Btype/Ctype` + `computeType`/`algo`（第 516–519 行） | 仅 `blas/gemm_ex/arch35/`（`gemm_ex_host.cpp` 500 行 + `gemm_ex_kernel.cpp` 671 行 + `gemm_ex_tiling_data.h`） | 声明完全同构，**无需改动**；实现缺 arch22 |
| `aclblasSgemmEx` | **15 参**：`const float* alpha/A/B/C` 强类型、缺 `Atype/Btype/Ctype`、多 `algo`（第 521–524 行） | 仅 `blas/gemm_ex/arch35/sgemm_ex_host.cpp`（427 行）+ `sgemm_ex_kernel.cpp`（505 行） | **不同构**，须按 §2.3 修正为 17 参并删除 `algo`；实现缺 arch22 |
| `aclblasCgemmEx` | 已存在 17 参声明，但 dtype 形参为 `int typeA/typeB/typeC`（第 472 行） | 仅 `blas/gemm/arch35/cgemm_ex_host.cpp`（950 线，内部使用私有枚举 `aclblasCgemmExType_t` + `cgemm_ex_type.h`），测试在 `test/gemm/cgemm_ex/arch35/` | 任务书称"全仓零命中"为**过期口径**；需按 §2.3 的 `aclDataType` 口径对齐，并补齐 arch22 实现 |

其余相关事实：

- `blas/gemm_ex/README.md` 的两个接口产品支持表均标注「Atlas A2/A3：不支持」，任务书 §2.2 明确要求更新为「支持」。
- `blas/gemm_ex/arch35/gemm_ex_tiling_data.h` 定义 `GemmExTilingData` 与 `GemmDTypeCase`（FP16/BF16/FP32/FP8 组合），arch35 使用 `BlockMmad` 低阶 API；`BlockMmad` 为 arch35 专属，arch22 不可直接复用。
- arch22 已有实数 GEMM 底座：`blas/gemm/arch22/{gemm_host.cpp(514), gemm_kernel.cpp(1269), gemm_kernel.h(46)}`，提供 `GemmArch22Tiling`、pack/matmul/accumulate/epilogue/whole-grid 多 kernel 流水与 workspace 三段布局，是本任务实数路径的主近亲。
- arch22 的复数 GEMM 通道已存在：`blas/gemm_strided_batched/arch22/cgemm_strided_batched_*`（`square`/`balanced`/`balanced_pair`/`padded`/`augmented`/`nz` 等变体，`CgemmSbTiling` 直接带 `alphaReal/alphaImag/betaReal/betaImag`）与 `blas/herk/arch22/cherk_compute_kernel.cpp`（`CherkArch22Mmad` 的 `Mmad` 用法），采用"成对向量核打包复数 tile + Cube 增广实 GEMM"形态，是复数主路径（V4）的直接近亲。
- 构建侧由 `blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动 `GLOB_RECURSE` 收集 `*/<arch>/*.cpp`，新增 `blas/gemm_ex/arch22/*.cpp` 无需改构建脚本；测试侧 `test/CMakeLists.txt` 会探测 `test/<name>/<arch>/<name>_test.cpp`，新增 `test/gemm_ex/arch22/gemm_ex_test.cpp` 即可被 `--ops gemm_ex` 拉起。

### 与任务书的差异处理

| 差异点 | 任务书口径 | 现场事实 | 处理方式 |
| --- | --- | --- | --- |
| `aclblasCgemmEx` 是否已存在 | 无声明、无实现 | 有声明（`int typeA`）+ arch35 实现 | 视作**跨产品线对齐**任务：按 §2.3 统一为 `aclDataType`，同步 `blas/gemm/arch35/cgemm_ex_host.cpp`、`blas/gemm/README.md` 与 `test/gemm/cgemm_ex`，保证 950 线不回归；arch22 为新增实现 |
| `aclblasSgemmEx` 形参 | 17 参，无 `algo` | 15 参强类型 + `algo` | 公共头手术（任务书 §2.3 明确要求，属红线 3 例外），同步 arch35 定义与调用点 |
| 用例条数 | §3.5 写"200 条" | 随任务 `test_cases/` 每接口 1200 条（1000 精度 + 200 性能 `TC_PF`），合计 3600 条 | 以 1200/接口 为验收范围（覆盖 §3.5 全部场景与更多补充），自测报告按 3600 条口径统计并说明差异 |
| 主路径 dtype | FP32 / COMPLEX64 | arch35 支持扩展混合精度；arch22 现有 sgemm 仅 FP32 | arch22 主路径实现 FP32（GemmEx/SgemmEx）与 COMPLEX64（CgemmEx），FP16/BF16 按 §2.4 支持集纳入（见"支持数据类型"），其余组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| CANN 版本 | 9.1.0（§3.1） | 现场工具链为 `cann-9.0.0` 与 `cann-9.2.0`（`ascend-toolkit/latest → cann-9.2.0`），无 9.1.0 | 经确认以 **9.2.0** 编译与上机；arch35 的 `ASC_DEVKIT_GE_9_1` 守卫分支（`blas/common/helper/devkit_version_compat.h`）在 9.2.0 下参与编译，利于公共头手术后 950 线回归 |
| 复数实现路线 | 未指定 | arch22 既有复数 GEMM 采用"成对向量核打包复数 tile + Cube 增广实 GEMM"形态（`blas/gemm_strided_batched/arch22/cgemm_strided_batched_*`，`CgemmSbTiling` 直接携带 `alphaReal/alphaImag/betaReal/betaImag`） | 经确认复数直上 **arch22 复数 Cube 通道**（V4 主路径），四实乘分解降为精度/性能兜底（V5） |

---

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上实现三个句柄式扩展 GEMM 接口，计算

`C = alpha * op(A) * op(B) + beta * C`

A、B、C 均按 BLAS 列主序（Column-Major）存储，op(A) 为 m×k、op(B) 为 k×n、C 为 m×n。C 原地覆写。接口通过 `aclblasSetStream` 绑定的 stream 异步下发 kernel，读回结果前由调用方同步。

## 需求拆解

1. **接口与公共声明**：`include/cann_ops_blas.h` 中三个接口统一为 §2.3 的 19/17/17 参口径，与 950PR 共用，不引入产品私有平行接口。
2. **host 侧**：参数合法性校验（严格按 §2.5 顺序）、no-op/缩放短路、tiling 计算、workspace 分配、按规模选择 kernel 变体并下发。
3. **kernel 侧**：实数 FP32 Cube 主路径、向量退化路径（小规模/非默认标量）、复数 COMPLEX64 路径、beta/alpha 融合 epilogue。
4. **测试**：`test/gemm_ex/arch22/` 下 CSV 驱动 GTest 工程，覆盖随任务提供的 3×1200 条用例、golden 由 cblas 生成、精度与性能分别验收。
5. **文档**：`blas/gemm_ex/README.md` 产品支持表与接口说明更新，新增接口说明与调用示例。
6. **不回归**：公共头改动后 950（arch35）线编译与语义不回退。

## 公共函数原型

```cpp
// 19 参，与 cublasGemmEx 同构（仓内已同构，无需改动）
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k,
    const void* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const void* beta,
    void* C, aclDataType Ctype, int ldc,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

// 17 参，与 cublasSgemmEx 同构（无 computeType、无 algo）
aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const float* beta,
    void* C, aclDataType Ctype, int ldc);

// 17 参，与 cublasCgemmEx 同构（复数 alpha/beta，无 computeType、无 algo）
aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const aclblasComplex* beta,
    void* C, aclDataType Ctype, int ldc);
```

dtype 枚举一律使用 `aclDataType`（`acl/acl_base_rt.h`）：实数 `ACL_FLOAT`、`ACL_FLOAT16`、`ACL_BF16`，复数 `ACL_COMPLEX64`。不使用任何不存在的枚举名（`ACL_DATA_FLOAT`、`ACL_DT_COMPLEX64`、`aclblasType_t` 均禁止）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据排布 | 维度 | 值域 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | aclblas 句柄，携带 stream，Host 内存 | scalar | - | - | 已创建的有效句柄 | nullptr → `HANDLE_IS_NULLPTR` |
| transA | 属性 | A 的运算方式（N/T/C） | int | - | - | {N,T,C} | 其他值 → `INVALID_ENUM` |
| transB | 属性 | B 的运算方式，同 transA | int | - | - | {N,T,C} | 其他值 → `INVALID_ENUM` |
| m | 属性 | op(A) 与 C 的行数 | int | - | 标量 | m ≥ 0 | m<0 → `INVALID_VALUE`；m=0 合法 no-op |
| n | 属性 | op(B) 与 C 的列数 | int | - | 标量 | n ≥ 0 | n<0 → `INVALID_VALUE`；n=0 合法 no-op |
| k | 属性 | op(A) 列数 / op(B) 行数 | int | - | 标量 | k ≥ 0 | k<0 → `INVALID_VALUE`；k=0 仅 beta 缩放 |
| alpha | 输入 | 标量乘数：GemmEx `const void*`（FP32）、SgemmEx `const float*`、CgemmEx `const aclblasComplex*` | FLOAT32 / COMPLEX64 | - | 标量 | 对应 dtype 全集 | nullptr → `INVALID_VALUE` |
| A | 输入 | 矩阵 A，Device 内存，只读，列主序 | 由 Atype 决定 | ND | N: m×k；T/C: k×m | 对应 dtype 全集 | k>0 且 alpha≠0 时 nullptr → `INVALID_VALUE` |
| Atype | 属性 | A 的 dtype 枚举 | int | - | - | 受支持集（见下表） | 不支持 → `NOT_SUPPORTED` |
| lda | 属性 | A 列主序前导维 | int | - | 标量 | transA=N: ≥ max(1,m)；否则 ≥ max(1,k) | 不满足 → `INVALID_VALUE` |
| B | 输入 | 矩阵 B，Device 内存，只读，列主序 | 由 Btype 决定 | ND | N: k×n；T/C: n×k | 对应 dtype 全集 | k>0 且 alpha≠0 时 nullptr → `INVALID_VALUE` |
| Btype | 属性 | B 的 dtype 枚举 | int | - | - | 同 Atype | 不支持 → `NOT_SUPPORTED` |
| ldb | 属性 | B 列主序前导维 | int | - | 标量 | transB=N: ≥ max(1,k)；否则 ≥ max(1,n) | 不满足 → `INVALID_VALUE` |
| beta | 输入 | 标量乘数，类型同 alpha；beta=0 时 C 可为 nullptr | FLOAT32 / COMPLEX64 | - | 标量 | 对应 dtype 全集 | nullptr → `INVALID_VALUE` |
| C | 输入输出 | 矩阵 C，Device 内存，列主序，原地覆写 | 由 Ctype 决定 | ND | m×n | 对应 dtype 全集 | beta≠0 时 nullptr → `INVALID_VALUE` |
| Ctype | 属性 | C 的 dtype 枚举 | int | - | - | 同 Atype | 不支持 → `NOT_SUPPORTED` |
| ldc | 属性 | C 列主序前导维 | int | - | 标量 | ldc ≥ max(1,m) | 不满足 → `INVALID_VALUE` |
| computeType | 属性 | 计算精度（仅 GemmEx） | int | - | - | `ACLBLAS_COMPUTE_32F` 等受支持值 | 不支持 → `NOT_SUPPORTED` |
| algo | 属性 | 算法选择（仅 GemmEx） | int | - | - | `ACLBLAS_GEMM_DEFAULT` | 不支持 → `NOT_SUPPORTED` |

## 校验顺序与边界行为

校验顺序严格按任务书 §2.5，实现上与 arch35 `gemm_ex_host.cpp` 同构，分三段函数：

| 序号 | 校验项 | 失败返回 |
| --- | --- | --- |
| 1 | handle 判空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | m/n/k 非负 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | lda/ldb/ldc 前导维 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | alpha/beta 判空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | Atype/Btype/Ctype 支持集 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 6 | transA/transB 枚举 | `ACLBLAS_STATUS_INVALID_ENUM` |
| 7 | A/B/C 指针（按是否会被读写） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | computeType/algo 支持集（仅 GemmEx） | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 9 | no-op 短路（m=0 或 n=0） | `ACLBLAS_STATUS_SUCCESS` |

边界矩阵：

| 场景 | 行为 |
| --- | --- |
| m=0 或 n=0 | 上述校验全部通过后返回 SUCCESS，不下发任何 kernel |
| k=0 | 只做 `C = beta*C`，不引用 A/B |
| alpha=0 | beta=1 → C 不变；beta=0 → C 置零；其他 → 逐元素缩放。不引用 A/B |
| beta=0 且 C=nullptr | 返回 SUCCESS，**全程不读写 C**（含 k=0 分支禁止 memset/置零） |
| beta=0 且 C 有效 | 旧 C 不参与计算（不得读入 alu 结果） |
| 仅缩放且 beta=1 | C 保持原值（位精确） |
| 右值全为完整矩阵 | 非连续 stride、广播、视图均不支持 |

---

# 详细设计（required）

## 算子分析

### 数学公式

$$
C \;=\; \alpha\cdot \operatorname{op}(A)\cdot \operatorname{op}(B) \;+\; \beta\cdot C
$$

其中

- `transA = ACLBLAS_OP_N`：`op(A) = A`（m×k）；`ACLBLAS_OP_T`：`op(A) = Aᵀ`；`ACLBLAS_OP_C`：`op(A) = Aᴴ`（实数路径与 T 等价，复数路径为共轭转置）。transB 同理。
- 元素定位按列主序：`A[r + c*lda]`、`B[r + c*ldb]`、`C[r + c*ldc]`，每个矩阵使用自己的前导维。
- 复数 `COMPLEX64` 按实部/虚部交错（interleaved）存储在 `aclblasComplex { float real; float imag; }` 中。复数乘法展开为

$$
(a_r + i a_i)(b_r + i b_i) = (a_rb_r - a_ib_i) + i\,(a_rb_i + a_ib_r)
$$

复数 epilogue 展开为

$$
C'_r = \alpha_r P_r - \alpha_i P_i + \beta_r C_r - \beta_i C_i,\qquad
C'_i = \alpha_i P_r + \alpha_r P_i + \beta_i C_r + \beta_r C_i
$$

其中 `P = op(A) · op(B)`，共轭转置通过翻转对应操作数虚部符号实现（`transA=C` 时对 A 虚部取负，`transB=C` 时对 B 虚部取负）。

### 支持数据类型

| 接口 | 主路径（Atype=Btype=Ctype） | 扩展组合 | 其余组合 |
| --- | --- | --- | --- |
| `aclblasGemmEx` | `ACL_FLOAT` ×3，`computeType=ACLBLAS_COMPUTE_32F` | `ACL_FLOAT16`×3、`ACL_BF16`×3（按各自生态精度档判定） | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `aclblasSgemmEx` | `ACL_FLOAT` ×3 | `ACL_FLOAT16`×3、`ACL_BF16`×3 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `aclblasCgemmEx` | `ACL_COMPLEX64` ×3 | — | `ACLBLAS_STATUS_NOT_SUPPORTED` |

A/B/C 三者 dtype 不一致的混合组合（如 A=FP16、C=FP32）在 arch22 首版返回 `NOT_SUPPORTED`，与 README 宣称的支持集保持一致；受支持的组合按 `(Atype,Btype,Ctype)` 三元组在 host 侧判定，kernel 侧以模板参数实例化。

### 支持形状与布局

- 维度：m、n、k 均为运行时入参（dynamic shape 由 host tiling 支持），无静态 shape 约束。
- 布局：ND / 列主序，A/B/C 各自带前导维 lda/ldb/ldc（可大于紧凑值，形成列间 padding）。
- 分块参数：Cube 基础块 `BASE_M=128`、`BASE_N=128`、`BASE_K` 按 dtype 取（FP32 为 8、FP16/BF16 为 16），列步长按 8 个 float 对齐；L1/L0 容量约束下 K 分段由 host 控制（`hostTileK`）。
- 不支持：非连续步长、broadcast、batch 维、视图返回。

## 算子实现

### 实现方案

新增目录与文件（全部落在任务书 §5 指定路径，不新建族目录）：

```
blas/gemm_ex/arch22/
├── gemm_ex_host.cpp          # 三入口 + 校验 + 短路 + 分派 + tiling + workspace
├── gemm_ex_tiling_data.h     # host/kernel 共享 tiling 结构（arch22 版）
├── gemm_ex_real_kernel.h     # 实数路径 kernel 入口声明
├── gemm_ex_real_kernel.cpp   # FP32/FP16/BF16 Cube 流水 + 向量退化路径 + epilogue
├── gemm_ex_complex_kernel.h  # 复数路径 kernel 入口声明
└── gemm_ex_complex_kernel.cpp# COMPLEX64 布局拆装 + 四实乘/复数 Cube + 复数 epilogue
test/gemm_ex/arch22/
├── gemm_ex_test.cpp          # CSV 驱动 GTest（复用 test/gemm_ex 共享 param/golden）
└── gemm_ex_test.csv          # 三接口 CSV 依次安装（各 1200 条）
```

命名沿用仓内 `gemm_ex_` 前缀，保证 `build.sh --ops gemm_ex` 的源过滤（`^gemm_ex_` / `/gemm_ex/`）与测试探测（`test/gemm_ex/<arch>/gemm_ex_test.cpp`）均能命中，无需改 `blas/CMakeLists.txt`、`test/CMakeLists.txt` 与 `test/gemm_ex/CMakeLists.txt`。

#### host 侧设计

**入口与分派。** 三个公共入口共用一段校验与短路逻辑，之后按接口分两条计算流水：

| 入口 | 计算流水 | 说明 |
| --- | --- | --- |
| `aclblasGemmEx` | `RunRealGemm` | 额外校验 `computeType ∈ {ACLBLAS_COMPUTE_32F …}`、`algo == ACLBLAS_GEMM_DEFAULT`；alpha/beta 为 `const void*`，host 侧按 `Atype` 解引用为 float |
| `aclblasSgemmEx` | `RunRealGemm` | alpha/beta 直接为 `const float*`；无 computeType/algo |
| `aclblasCgemmEx` | `RunComplexGemm` | alpha/beta 为 `const aclblasComplex*`；布局拆装 + 复数乘加 |

**tiling 决策表。** host 侧按"规模 + 标量默认性 + dtype"选择 kernel 变体，决策表如下（阈值随实测微调，不影响接口语义）：

| 变体 | 触发条件 | 计算通道 | workspace |
| --- | --- | --- | --- |
| V0 `ScaleOnly` | `k==0` 或 `alpha==0` | AIV 向量：beta=1 直返（不写 C）、beta=0 零填、其他逐元素乘 | 无 |
| V1 `TinyVector` | `m*n*k ≤ 4096`，或 `alpha != 1` / `beta ∉ {0,1}` 且规模较小 | AIV 向量 FMA（按 K 逐次累加，C 在 UB 中） | 无 |
| V2 `WholeGrid` | 方阵且单核可覆盖（`m,n ≤ BASE_M/BASE_N` 上限，K 较小） | AIC 单核 Cube，免 pack、免 K 分段 | 无 |
| V3 `PackedMultiCore` | 大规模（默认） | AIC 多核 Cube：pack → matmul（K 分段）→ accumulate → epilogue | 三段（pack A/B、product、accum） |
| V4 `ComplexCube`（主路径） | `aclblasCgemmEx` 默认 | arch22 复数 Cube 通道：AIV 成对打包复数 tile → Cube 以增广实 GEMM 一次完成复数乘加 → 复数 epilogue | 打包平面 + 单乘积平面 |
| V5 `ComplexFourReal`（兜底） | V4 精度或性能不达标的降级路线 | 四实乘分解复用 `RunRealGemm` + 复数 epilogue | 4 个乘积平面 + 输入平面 |

复数主路径选 V4 而非 V5 的依据：arch22 既有复数 GEMM（`cgemm_strided_batched`、`herk`）已实测采用"成对向量核打包 + Cube 增广实 GEMM"形态，FLOP 与 cuBLAS 同级，避免 V5 的 4× 乘法放大；V5 仅作为大 K 精度补偿或 V4 调不通时的兜底，切换不改变接口与精度判据。

**分核策略。** 取 `aicCores = GetAicCoreCount()`（910B3 为 24），按输出分块二维均分：

```
mBlocks = CeilDiv(m, singleCoreM),  nBlocks = CeilDiv(n, singleCoreN)
blockIdx -> (mBlockIdx = blockIdx % mBlocks, nBlockIdx = blockIdx / mBlocks)
usedCoreNum = min(aicCores, mBlocks * nBlocks)
```

尾块按实际有效行列数搬运（`min(singleCoreM, m - mBlockIdx*singleCoreM)`），不读越界 padding。AIV 侧的 pack/epilogue 采用 block-stride 任务分配（`for i = blockIdx; i < total; i += aivCores`），与仓内 AIV 惯例一致。

**K 分段。** `hostTileK` 由 L1/L0 容量与 dtype 决定（FP32 取 256，复数取 128），host 侧按 `kBase += hostTileK` 循环下发；首片 `firstK=1` 覆盖写，后续片累加。workspace 布局复用 `GemmArch22Tiling` 的三段偏移思路：

```
packAOffset = 0
packBOffset = packAOffset + align(lda_m) * kTile
productOffset = packBOffset + align(ldb_n) * kTile
accumOffset   = productOffset + align(m) * align(n)
totalBytes    = accumOffset + align(m) * align(n)
```

workspace 由 handle 的 `EnsureDefaultWorkspace` 增长，用户 workspace 走容量检查；不足返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

**tilingkey 规划。** arch22 不采用 tilingkey 下发，沿用仓内 arch22 的"host 选函数、kernel 为独立 `__global__` 入口"机制：dtype 与基础块由模板参数实例化，host 通过函数指针表选择具体 kernel（实数 1 组、复数 2 组），避免运行时分支。

#### kernel 侧设计

**实数 FP32 Cube 主路径（V2/V3）。** 列主序 GEMM 通过交换 A/B 映射为行主序 Cube 计算（`C = A·B` 列主序等价于 `Cᵀ = Bᵀ·Aᵀ` 行主序），流程与 `blas/gemm/arch22/gemm_kernel.cpp` 同构：

1. **pack**：当 `lda/ldb` 不是 8 个 float 的整数倍、或需要转置时，先按实际前导维把逻辑元素打包到 `lda/ldb = align8(逻辑行数)` 的 workspace 平面，尾部填零；已对齐且可原样跨列读取时走 raw-stride 快路径（不做打包）。
2. **matmul**：`LoadData`/`Nd2Nz` 加载 A/B 到 L1→L0A/L0B，逐 `kIdx` 调用 `Mmad` 累加到 L0C；`SetOrgShape` 显式传入实际物理步长。
3. **accumulate**：K 分段产物按片写入 workspace，首片覆盖、后续片逐元素累加（避免原子加的不确定性）。
4. **epilogue**：融合 `alpha`/`beta`，按 `ldc` 直写逻辑 C；`beta==0` 分支不读旧 C；`alpha==1 && beta==0 && ldc==m` 时跳过 workspace 直接写出。

FP16/BF16 扩展组合复用同一流水，仅改模板 dtype 与 `BASE_K`；L0C→GM 时按 Ctype 决定是否保持在 FP32 后转半精度输出。

**向量退化路径（V1）。** 规模小或标量非默认时，A/B/C tile 一次性载入 UB：

- A 非转置按列连续读，先以 beta 缩放 C，再按 k 逐项 `FusedMulAdd` 累加 `alpha*A[.,k]*B[k,.]`；
- A 转置时用 strided copy / Gather 取出元素，先累加乘积再组合 alpha/beta；
- 转置 A 的最终 beta 组合使用与 Netlib 一致的乘加表达式顺序，保证小规模用例的舍入方向贴近 golden。

**复数路径（V4 主路径：arch22 复数 Cube 通道）。** 不做四实乘放大，而是复用 arch22 既有复数 GEMM 形态——**AIV 成对打包复数 tile + Cube 执行增广实 GEMM**，一次 Mmad 完成复数乘加。增广形式为

$$
\begin{bmatrix} A_r & -A_i \\ A_i & A_r \end{bmatrix}
\begin{bmatrix} B_r \\ B_i \end{bmatrix}
=
\begin{bmatrix} A_rB_r - A_iB_i \\ A_iB_r + A_rB_i \end{bmatrix}
=
\begin{bmatrix} P_r \\ P_i \end{bmatrix}
$$

具体做法参考 `blas/gemm_strided_batched/arch22/`（`CgemmSbTiling` 直接携带 `alphaReal/alphaImag/betaReal/betaImag`，`paddedM/N/K`，并已有 `square`/`balanced`/`balanced_pair`/`padded` 等 kernel 变体）与 `blas/herk/arch22/cherk_compute_kernel.cpp`（`CherkArch22Mmad<128U>` 的 `MmadParams`/`Mmad` 用法）。实现要点：

1. **打包（AIV）**：按逻辑 `op(A)`/`op(B)` 读取交错 real/imag，AIV 成对生成增广实平面（A 的实块与负虚块、B 的实/虚块），列步长按 8 个 float 对齐、尾部填零；`lda/ldb` 非对齐或转置时由打包吸收，Cube 侧只见对齐步长。
2. **共轭**：`trans=C` 时在打包阶段翻转对应操作数虚部符号（相当于对 `A_i`/`B_i` 取负），Cube 侧路径不变——与 `cgemm_strided_batched` 的既有处理一致。
3. **计算（Cube）**：增广实 GEMM 以 `BASE_M/BASE_N=128`、`BASE_K` 按 FP32 取 8 的块配置走标准 Mmad 累加；K 分段与实数路径共用同一套 `hostTileK` 与 workspace 累加逻辑。
4. **epilogue（AIV）**：从增广结果的上下两半取 `P_r`、`P_i`，按 §详细设计 的复数公式融合 `alpha = (α_r, α_i)`、`beta = (β_r, β_i)`，按 `ldc` 重新交错写回；`beta=0` 分支不读旧 C。
5. **精度**：单次增广实 GEMM 的 K 累加次数与实数路径相同（无 4× 放大），误差量级与 `aclblasSgemm` 同档；大 K（≥2048）若 `max_abs_error` 超标，先加 K 分片补偿求和（Neumaier，参考 `sparse/spmm/arch22/kernels/compensated_math.h`），仍不达标再降级 V5。
6. **小规模退化**：`m*n*k ≤ 4096` 或非默认标量时走 AIV 单核向量复数路径（UB 内一次载入、逐 K FMA），避免小算子的多次 kernel 下发开销。
7. **极端值**：`0×Inf`、`Inf-Inf` 等不定式按任务书 §3.2.5 的等效规则判定，实现不额外屏蔽 NaN/Inf（保留 IEEE 传播），由测试侧按规则核对。

**复数兜底路径（V5，仅降级时启用）。** 将复数 GEMM 展开为四个实数 GEMM 复用 `RunRealGemm`：

$$
P_{rr}=\operatorname{op}(A_r)\operatorname{op}(B_r),\quad
P_{ii}=\operatorname{op}(A_i)\operatorname{op}(B_i),\quad
P_{ri}=\operatorname{op}(A_r)\operatorname{op}(B_i),\quad
P_{ir}=\operatorname{op}(A_i)\operatorname{op}(B_r)
$$

`P_r = P_{rr} - P_{ii}`、`P_i = P_{ri} + P_{ir}` 后再走同一复数 epilogue；实现简单但 FLOP 4×，仅当 V4 调不通或大 K 精度补偿后仍不达标时启用，切换不影响接口与判据。

**同步与异步生命周期。** 所有 kernel 使用 handle 绑定的同一 stream，顺序执行 pack→matmul→accumulate→epilogue；kernel 内 MTE2/MTE3 与向量/Cube 之间使用 `PipeBarrier<PIPE_ALL>`/`PIPE_V` 显式同步（沿用 `gemm/arch22` 现有写法）。主计算不做 host D2H；调用方同步 stream 后读回。

### 复用与新写清单

| 类别 | 内容 | 来源 |
| --- | --- | --- |
| 直接复用（不新写） | `_aclblas_handle`/stream/`EnsureDefaultWorkspace`、`GetAicCoreCount`/`GetAivCoreCount`、`dtype_cast.h` 半精度转换、`GemmArch22Tiling` 三段 workspace 思路、**复数增广打包 + Cube Mmad 形态**（`CgemmSbTiling`/`CherkArch22Mmad`）、共轭（虚部符号翻转） | `common/helper/*`、`blas/gemm/arch22/*`、`blas/gemm_strided_batched/arch22/*`、`blas/herk/arch22/*` |
| 语义复用（照抄结构，重写实现） | 校验三段式与校验顺序、no-op/缩放短路、`C==nullptr && beta==0` 直返 | `blas/gemm_ex/arch35/gemm_ex_host.cpp` |
| 新写 | arch22 host 入口与 tiling（`gemm_ex_host.cpp`）、实数 kernel 流水与 epilogue、复数增广打包/复数 epilogue、arch22 tiling 结构 | 本任务 |
| 修改（红线 3 例外，任务书明确要求） | `include/cann_ops_blas.h`（SgemmEx 15→17 参、CgemmEx dtype 形参改 `aclDataType`）、`blas/gemm_ex/README.md`（产品支持表 A2/A3 改"支持" + 接口说明）、arch35 连带修复（`sgemm_ex_host.cpp`、`cgemm_ex_host.cpp`、`blas/gemm/README.md`、`test/gemm/cgemm_ex` 适配） | 本任务 |
| 测试新写 | `test/gemm_ex/arch22/{gemm_ex_test.cpp, gemm_ex_test.csv}`，复用共享 `gemm_ex_param.h`/`gemm_ex_golden.h` | 本任务 |

## 支持硬件

| 项目 | 内容 |
| --- | --- |
| 架构 | arch22（dav-2201） |
| 产品 | Atlas A2 训练/推理系列（含 Atlas 800T A2）、Atlas A3 训练/推理系列（含 Atlas 800T A3） |
| 自验款型 | 精度与性能均在 Atlas 800T A2（910B3）上完成（任务书 §3.1 允许单款型自验，性能建议 910B3） |
| CANN | **9.2.0**（现场实际 + 用户确认；任务书 §3.1 要求 9.1.0，差异见"与任务书的差异处理"表） |
| golden | Netlib BLAS 3.10.0：实数 `cblas_sgemm`、复数 `cblas_cgemm` |
| 构建 | Release，`build.sh --soc ascend910b3 --ops gemm_ex` 单算子增量构建 |

## 算子约束限制

1. 仅支持列主序（Column-Major）与各自独立前导维，不支持行主序、非连续 stride、视图与广播。
2. A/B/C 的 dtype 组合必须在支持集内，混合组合返回 `NOT_SUPPORTED`。
3. `ACLBLAS_OP_C` 在实数路径下与 `T` 完全等价（不做共轭），在复数路径下为共轭转置。
4. `beta==0` 时完全不读 C，且 `C==nullptr` 合法；不得对 C 做任何写入（含 memset/置零）。
5. `m==0` 或 `n==0` 为合法 no-op；`k==0` 只做 beta 缩放；`alpha==0` 只做 beta 缩放。
6. 精度非 bit-exact（浮点乘加累加），但 `alpha==0` 的 beta 缩放路径为位精确。
7. 不承诺确定性累加顺序（K 分段与多核划分随规模变化），跨次运行结果可能末位不同。
8. 不涉及内存占用约束（§3.4 不涉及）；单用例 host 侧内存预算沿用测试工程 ≤512MB。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准（FLOAT32） | rtol = atol = 2⁻¹³（≈1.2207031e-4），matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP | 生态算子开源精度标准（opbase `mixed_tolerance_standard.md`） |
| 精度标准（COMPLEX64） | 实部、虚部分别按 FLOAT32 阈值判定（同一档 rtol/atol/matched_ratio） | 同上 + 任务书 §3.2.2 |
| 精度标准（FP16/BF16 扩展） | FP16：rtol = atol = 2⁻¹⁰；BF16：rtol = atol = 2⁻⁷ | 任务书 §3.2.3 |
| 特例 | `alpha==0` 且 beta 给定时 `C = beta*C` 为位精确（EXACT） | 任务书 §3.2.4 |
| 非有限值 | golden 与 actual 同为 NaN / 同为同号 Inf / 一方 NaN 一方 ±Inf 判等效；一方有限另一方非有限判不通过；非有限元素不计入 matched_ratio 与 max_abs_error，单独统计 | 任务书 §3.2.5 |
| 性能标准 | 910B3 上平均单次耗时（msprof op task duration，5 次 warmup 不计入，10 次有效采样取均值）≤ 1.25 × GPU 标杆（≥0.8× 标杆性能） | 任务书 §3.3 |
| 性能标杆 | `test_cases/aclblasGemmEx/gpu_baseline.csv`（每接口 200 行，字段 `id,transA,transB,m,n,k,gpu_ms`） | 任务书 §3.3 + 随任务基准文件 |

> 标杆口径说明：随任务 `gpu_baseline.csv` 的 `gpu_ms` 为 NVIDIA P102-100 实测回填（随测用例 README 明示，非 A100），自测报告将如实标注该口径，判定仍按任务书 1.25× 执行。

## 测试与验证方案

### 用例组织

`test/gemm_ex/arch22/gemm_ex_test.cpp` 采用 CSV 驱动 GTest（与 `test/gemm/arch22/gemm_test.cpp` 的 `TEST_P(...CsvDriven)` + `INSTANTIATE_TEST_SUITE_P` 同构），复用 `test/gemm_ex/gemm_ex_param.h`（参数解析、`parseAclDataType`、物理维度换算）与 `test/gemm_ex/gemm_ex_golden.h`（golden 生成）。三个接口的 CSV 依次安装到 `test/gemm_ex/arch22/gemm_ex_test.csv` 后重编运行：

| 接口 | 用例文件 | 条数 |
| --- | --- | --- |
| `aclblasGemmEx` | `test_cases/aclblasGemmEx/gemm_ex_test.csv` | 1200（1000 精度 + 200 性能） |
| `aclblasSgemmEx` | `test_cases/aclblasSgemmEx/sgemm_ex_test.csv` | 1200（1000 + 200） |
| `aclblasCgemmEx` | `test_cases/aclblasCgemmEx/cgemm_ex_test.csv` | 1200（1000 + 200） |

用例分类（每接口）与覆盖意图：

| 类别 | 条数 | 覆盖内容 |
| --- | --- | --- |
| TC_L0 | 9 | transA×transB 全组合（N/T/C 3×3），m=n=k=4 |
| TC_MN | 116 | 23 种尺寸（1 起至 2048，含奇数/非对齐/2 的幂±1）× 4 组转置 + 8 组矩形 × 3 组转置 |
| TC_AB | 24 | alpha ∈ {1, 0.5, 0, −1, 2.5, −0.5, 100} × beta ∈ {0, 1, 0.5} |
| TC_LD | 12 | lda/ldb/ldc padding（+4/+8）× 4 组转置 |
| TC_FL | 18 | 均匀/全零/交替/极端/Inf/NaN，A/B/C 分别填充 |
| TC_CV | 16 | 中等尺寸（10~400）× 2 组转置 |
| TC_ED | 37 | 零维 no-op、k=0 缩放、alpha=0、beta=0 不读 C、空指针（含 handle 空）、非法枚举/dtype/computeType/algo、负维度、非法前导维、校验顺序叠加 |
| TC_EX | 768 | 尺寸×转置×标量×padding 确定性采样（补足精度条数） |
| TC_PF | 200 | 4 条任务书典型 case（1024³、2048³、4096³、1024³ TN）+ 小尺寸 + 方阵对数扫描 + skinny + 混合 |

固定随机种子 `20261028`，`gen_csv.py` 可逐字节复现。CSV 中 `mere_threshold=0.00012207031`（2⁻¹³）、`mare_multiplier=10.0` 仅作为测试工程 MERE/MARE 配置，不替代生态标准比对。

### 判定与追溯

- golden：实数用 Netlib `cblas_sgemm`、复数用 `cblas_cgemm`，与用例同 lda/ldb/ldc 与转置口径；CgemmEx 实部/虚部分别全矩阵比对。
- 逐元素通过条件：`|actual − golden| ≤ atol + rtol × |golden|`；用例通过条件：matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32×ULP）；非有限值单独按 §3.2.5 判定。
- 负向用例只比对返回码，不参与数值比对；多参数非法叠加用例以任务书 §2.5 的校验顺序为判定依据（例如 `m0_null_A_check_first` 验证 A 指针校验先于 no-op 短路）。
- 每次运行落 JSON/XML：case_id、期望码、实际码、matched_ratio、max_abs_error、非有限计数、失败原因；不依赖人工读日志判读。
- padding 隔离：TC_LD 组额外把 padding 区域填 NaN，断言其未被写入（防越界）。

### 性能测量

- 采样：`msprof op --application="<test_bin> --gtest_filter=*TC_PF_1001*"`，取 `OPPROF_*/OpBasicInfo.csv` 的 `Task Duration(us)`；5 次 warmup 不计入，11 次有效采样（满足 §3.3 的 10 次与 §7.4 的 >10 次）取平均。
- 判据：`Avg_NPU ≤ 1.25 × gpu_ms × 1000`；ACL event 墙钟作为对照列单列，不作为判据。
- 按接口 200 条逐 case 判定，输出 `results/perf_gap.json`（case / 实测 / 标杆 / 超比），超标 case 用 msprof 归类（launch / 带宽 / K 归约 / 串行依赖）。

### 分阶段门禁

| 门禁 | 范围 | 通过条件 |
| --- | --- | --- |
| G2 冒烟 | L0 9 条 + TC_ED 37 条（三接口） | 全部符合期望返回码与精度 |
| G3 精度全量 | 3×1000 精度用例（TC_EX 分块跑） | matched_ratio 与 max_abs_error 全部达标 |
| G4 性能基线 | 3×200 性能用例 | 全部 ≤1.25× 标杆；超标项进冲刺清单 |
| G6 回归 | 干净 clone + 重新构建，全量精度 + 性能 | 与 G3/G4 结论一致，公共头改动后 950 线编译通过 |

## 兼容性分析

| 影响面 | 变更 | 兼容性处理 |
| --- | --- | --- |
| `include/cann_ops_blas.h` · `aclblasSgemmEx` | 15 参（强类型 + `algo`）→ 17 参（`void*` + `Atype/Btype/Ctype`，删 `algo`） | 任务书 §2.3 明确要求；仓内调用点为 arch35 定义、README 示例与测试，逐一同步；源码级不兼容属预期（接口修正），无二进制兼容承诺 |
| `include/cann_ops_blas.h` · `aclblasCgemmEx` | dtype 形参 `int typeA/typeB/typeC` → `aclDataType Atype/Btype/Ctype` | `extern "C"` 链接层 ABI 不变（整型宽度一致）；源码层枚举替换，arch35 内部 `aclblasCgemmExType_t` 保留为私有映射（`aclDataType → 内部枚举`），对外不再暴露 |
| 950（arch35）产品线 | 上述两接口的 arch35 实现与 README、`test/gemm/cgemm_ex` 依赖旧签名 | 同步改造并回归编译；与 950PR 共用同一份公共声明，禁止产品私有平行接口 |
| `blas/gemm_ex/README.md` | A2/A3 支持表由"不支持"改"支持"，补 arch22 说明与调用示例 | 任务书 §2.2 明确要求（红线 3 例外，登记例外清单） |
| 现有 arch22 `aclblasSgemm` | 不受影响 | GemmEx 族走新增 `blas/gemm_ex/arch22/`，不改动 `blas/gemm/arch22/` 的 sgemm 语义 |
| 构建/测试脚本 | 无改动 | arch 目录按 `SOC_ARCH_DIRS` 自动收集；测试按 `test/<name>/<arch>/<name>_test.cpp` 自动探测 |

## 交付与合入清单

- [ ] 三接口声明符合 §2.3 的 19/17/17 参数顺序与枚举，arch35 同步且编译通过
- [ ] `blas/gemm_ex/arch22/` 六文件实现 + `blas/gemm_ex/README.md` 支持表更新
- [ ] `test/gemm_ex/arch22/` CSV 驱动 GTest + 三份 CSV 安装说明，`test/` 下 README 给出可复现步骤
- [ ] 3×1200 用例精度判定全部达标，3×200 性能用例全部 ≤1.25× 标杆
- [ ] 自测报告含用例参数、精度对比结果与截图、性能数据（11 次有效采样均值）与截图
- [ ] 个人仓提供任务分支与算子目录，并邀请 `Ascend-CANN` 账号为开发者
- [ ] 合入 PR 路径：`blas/gemm_ex/arch22/` 与 `test/gemm_ex/`（含 csv），无冗余文件、无无用代码

