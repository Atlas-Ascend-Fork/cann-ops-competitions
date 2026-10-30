# 需求背景（required）

## 需求来源

- 任务书：《10 月社区任务-aclblasGemmEx_SgemmEx_CgemmEx 算子开发（A2A3）》（`aclblasGemmEx_SgemmEx_CgemmEx_A2_task_doc.md`）及其附件测试用例包（三个接口各 1200 条 CSV）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，设计依据的 `master` 提交为 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。
- 目标硬件：Atlas A2 系列（性能测试设备 Atlas 800T A2 / 910B3）、Atlas A3 系列；CANN 9.1.0；精度 golden 为 Netlib BLAS 3.10.0（`cblas_sgemm` / `cblas_cgemm`）。
- 本文为实现方案的设计说明，不含性能验收结论。

## 背景介绍

### 三个接口的功能与现状

三个句柄式 BLAS 接口计算列主序的扩展通用矩阵乘法，统一表达 `C = alpha * op(A) * op(B) + beta * C`，参数口径以 cuBLAS 为准，仅做昇腾侧类型映射。

| 接口 | 对标接口 | 参数个数 | 仓内现状（`05b96ea`） |
| --- | --- | --- | --- |
| `aclblasGemmEx` | `cublasGemmEx` | 19 | `include/cann_ops_blas.h:505-508` 已声明且与 cuBLAS 同构 |
| `aclblasSgemmEx` | `cublasSgemmEx` | 17 | 声明为 15 参强类型变体（缺 `Atype/Btype/Ctype`、多 `algo`），与 cuBLAS 不同构 |
| `aclblasCgemmEx` | `cublasCgemmEx` | 17 | `include/`、`blas/`、`docs/` 全仓零命中，需新增声明与实现 |

`blas/gemm_ex/` 目录已有 `arch35`（Ascend 950PR）实现，`blas/gemm_ex/README.md` 的产品支持表标注 Atlas A2/A3 为「不支持」；`blas/gemm/arch22/` 存在单精度实数 GEMM（`aclblasSgemm`/`aclblasCgemm`）的 arch22 工程与 `test/gemm/arch22` 用例，可作为 A2/A3 的工程与性能基础，但其参数校验和边界语义（`C==nullptr`、`beta==0` 等）不能直接沿用。

### 需求要点

- 补齐 A2/A3 实现：三个接口实现落在 `blas/gemm_ex/arch22/`，测试落在 `test/gemm_ex/`、`test/sgemm_ex/`。
- 接口声明统一放 `include/cann_ops_blas.h`，与 950PR 共用：`aclblasSgemmEx` 改为任务书 17 参，新增 `aclblasCgemmEx`，禁止产品私有平行接口。
- `README` 产品支持表将 A2/A3 更新为「支持」，并补齐 `aclblasCgemmEx` 章节。
- 主路径：`aclblasGemmEx`/`aclblasSgemmEx` 为 FP32 实数；`aclblasCgemmEx` 为交错存储的 COMPLEX64。

# 需求分析

## 需求描述

使用 Ascend C 编程语言、以 kernel 直调方式（handle 绑定 stream）实现三个句柄式 GEMM 扩展接口，满足任务书 §2.4/§2.5 的维度、前导维、枚举、指针与短路语义，精度满足生态算子开源精度标准，性能达到 GPU 标杆的 0.8 倍以上。

支持能力：

| 能力 | 目标 |
| --- | --- |
| 转置 | `ACLBLAS_OP_N` / `T` / `C` 全组合；实数路径 C 与 T 等价，复数路径 C 为共轭转置 |
| 数据类型 | A2/A3 主路径 FP32（GemmEx/SgemmEx）与 COMPLEX64（CgemmEx）；支持集外组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 形状 | 运行时 m/n/k ≥ 0，支持矩形、奇数、非对齐与大规模方阵（性能用例至 4096³） |
| 前导维 | `lda/ldb/ldc` 取任意合法值（含 padding），padding 不参与计算且不被覆盖 |
| 边界 | `m=0/n=0` 合法 no-op；`k=0`/`alpha=0` 仅做 `C=beta*C`；`beta=0` 时 C 允许为空且全程不读写 C |
| 特殊值 | 输入含 Inf/NaN 时按任务书 §3.2 结构等效规则判定 |

## 需求拆解

| 项目 | 方案与交付边界 |
| --- | --- |
| 公共接口 | `include/cann_ops_blas.h`：`GemmEx` 保持 19 参；`SgemmEx` 修正为 17 参（`const float* alpha/beta` + `void* A/B/C` + `Atype/Btype/Ctype`，无 `algo`）；新增 17 参 `CgemmEx`（`const aclblasComplex* alpha/beta`） |
| A2/A3 实现 | 新增 `blas/gemm_ex/arch22/`：host 校验与编排、复数 pack kernel、复数 epilogue kernel；实数路径复用仓内 `aclblasSgemm` |
| 单算子构建 | `blas/CMakeLists.txt` 登记 `gemm_ex`/`sgemm_ex` 对 `blas/gemm`、`blas/scal` 源目录的依赖，保证 `build.sh --ops=...` 可独立链接 |
| 950 兼容 | 同步把 `arch35` 的 `aclblasSgemmEx` 定义与 `test/sgemm_ex/arch35` 调用方改到 17 参，避免头文件变更造成 950 构建断裂 |
| 测试 | `test/gemm_ex/arch22/`（GemmEx + CgemmEx）、`test/sgemm_ex/arch22/`（SgemmEx），按 CSV 表头自动识别被测接口；覆盖精度、边界/负向、特殊值与性能 |
| 文档 | `blas/gemm_ex/README.md`（产品支持表、17 参 SgemmEx、新增 CgemmEx 章节）、`test/gemm_ex/README.md`（测试步骤） |

# 详细设计（required）

## 算子分析

### 数学公式与地址映射

`C = alpha * op(A) * op(B) + beta * C`，`op(A)` 为 m×k、`op(B)` 为 k×n、C 为 m×n，全部列主序，元素地址按元素计：

| 操作数 | trans = N | trans = T / C |
| --- | --- | --- |
| A 的 `(i,p)` | `A[i + p*lda]` | `A[p + i*lda]` |
| B 的 `(p,j)` | `B[p + j*ldb]` | `B[j + p*ldb]` |
| C 的 `(i,j)` | `C[i + j*ldc]` | 不适用（C 无转置） |

实数路径下 `C` 与 `T` 等价；复数路径下 `C` 对索引到的元素再取共轭。`lda ≥ max(1, m)`（A 为 N）或 `max(1, k)`（A 为 T/C）；`ldb ≥ max(1, k)`（B 为 N）或 `max(1, n)`；`ldc ≥ max(1, m)`。

### 计算量

一次实数 GEMM 为 `2mnk` FLOPs；复数乘加按 4 次实数乘 + 2 次实数加计，等效 `8mnk`，与 GPU 标杆中 CgemmEx 恰好为 SgemmEx 4 倍的实测关系一致，故复数路径采用 4 个实数 GEMM 的分解在算力口径上与标杆同阶。

### 支持数据类型

| 接口 | Atype | Btype | Ctype | alpha/beta | 其余参数 |
| --- | --- | --- | --- | --- | --- |
| `aclblasGemmEx`（A2/A3） | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `const void*`，指向 FP32 | `computeType=ACLBLAS_COMPUTE_32F`、`algo=ACLBLAS_GEMM_DEFAULT` |
| `aclblasSgemmEx`（A2/A3） | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `const float*` | 无 computeType/algo |
| `aclblasCgemmEx`（A2/A3） | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `const aclblasComplex*` | 无 computeType/algo |
| 扩展混合精度 | FP16/BF16/FP8 等组合 | — | — | — | 本任务不实现，返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |

dtype 判定分两级：取值不属于 `aclDataType` 枚举时返回 `ACLBLAS_STATUS_INVALID_VALUE`，属于枚举但不在支持集内返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（与附件 SgemmEx/CgemmEx 用例中 `Atype=999` 与 `ACL_DOUBLE` 的不同期望一致）。

### 支持形状与边界

- `m, n, k ≥ 0`；`m=0` 或 `n=0` 为合法 no-op。no-op 判定放在**全部参数校验之后**，因此 `m=0` 且 `A=nullptr`（`k>0`、`alpha≠0`）返回 `INVALID_VALUE`。
- `k=0` 或 `alpha=0`：不引用 A/B，执行 `C = beta*C`（`beta=0` 置零、`beta=1` 不变、其余逐元素缩放）。
- `beta=0` 且 `C=nullptr`：返回成功且不读写 C（含 `k=0` 分支）；`beta≠0` 且 C 为空返回 `INVALID_VALUE`。
- 参数校验顺序（三接口一致）：handle → m/n/k → lda/ldb/ldc → alpha/beta 判空 → dtype 支持集 → transA/transB 枚举 → A/B/C 指针 → `GemmEx` 的 computeType/algo → no-op 短路。
- 状态码：空 handle→`HANDLE_IS_NULLPTR`；非法枚举→`INVALID_ENUM`；非法值/空指针→`INVALID_VALUE`；不支持组合→`NOT_SUPPORTED`。

## 算子实现

### 实现方案

整体分两条路径，公共校验与编排放在 `blas/gemm_ex/arch22/gemm_ex_host.cpp`：

```
aclblasGemmEx / aclblasSgemmEx ─ 统一校验 ─ 实数路径：委派 aclblasSgemm（blas/gemm/arch22，含其 beta 缩放）
aclblasCgemmEx ─────────────── 统一校验 ─ 复数路径：pack kernel + 4 个 FP32 GEMM + epilogue kernel
```

#### 3.2.1 host 侧设计

##### 1. 参数校验与 quick return

按上文顺序实现统一校验函数 `ValidateGemmEx`，以参数结构体区分「复数标量」与「是否携带 computeType/algo」三种入口；校验通过后由 `RunRealGemm` / `RunComplexGemm` 分发。附件 3600 条用例的 `expect_result` 已逐行比对通过。

##### 2. 实数路径：复用仓内 SGEMM

`blas/gemm/arch22` 的 `aclblasSgemm` 已实现列主序、全部转置组合、任意合法前导维、`k=0`/`alpha=0` 的 beta 缩放与整问题多 Cube 发射，Ex 适配层只补 Ex 契约：

- `C == nullptr && beta == 0` 提前返回成功（`aclblasSgemm` 要求 C 非空）；
- `m == 0 || n == 0` 提前返回成功；
- dtype/computeType/algo 支持集在适配层拦截，不下传。

其余参数原样传递，不引入任何额外拷贝。

##### 3. 复数路径：四实数 GEMM 分解

记 `A = Ar + i·Ai`、`B = Br + i·Bi`，四个实数乘积 `P1 = Ar·Br`、`P2 = Ai·Bi`、`P3 = Ar·Bi`、`P4 = Ai·Br`，则

```
Re(C) = α_re(P1 − P2) − α_im(P3 + P4) + β_re·Re(C_old) − β_im·Im(C_old)
Im(C) = α_re(P3 + P4) + α_im(P1 − P2) + β_re·Im(C_old) + β_im·Re(C_old)
```

共轭（`trans = C`）与转置无关：`op(A) = A^T` 与 `A^H` 的差别仅在于虚部符号，因此在打包阶段翻转虚部平面即可，四个 GEMM 统一调用 `aclblasSgemm` 并把实/虚平面按原前导维送入，转置语义由实数平面继承。

alpha/beta 在 epilogue 中统一处理（GEMM 以 `alpha=1, beta=0` 调用），因此 4 个乘积平面可复用同一套 tiling；`alpha=0` 或 `k=0` 时 `hasProduct=0`，完全跳过 pack 与 4 次 GEMM，只做 `C = beta·C`。

##### 4. workspace 规划

workspace 按 `[A_re | A_im | B_re | B_im | P1 | P2 | P3 | P4]` 顺序排布，每段按 8 个 fp32（32 B）对齐；平面沿用复数矩阵的前导维（紧凑输入保持紧凑，可命中 SGEMM 的直通路径），乘积平面 `ldp = m`。

由于 `aclblasSgemm` 使用同一 handle workspace，复数路径在写入任何平面之前一次性 `EnsureDefaultWorkspace(自身平面 + SGEMM workspace 上界)`，避免 4 次 GEMM 内部扩容释放仍被引用的平面内存。上界按 SGEMM 的两条执行路径（整问题多 Cube：`(m+n)·align128(k)+mn`；1024 分块：`4·1024²+20·256·208`）取大值并保留 4096 元素余量；超过 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）时返回 `ALLOC_FAILED`。4096³ 复数用例峰值约 736 MB。

##### 5. tiling 数据设计

host 与 device 共享结构 `GemmExArch22ComplexTiling`（`blas/gemm_ex/arch22/gemm_ex_arch22_tiling.h`）：

| 字段 | 含义 |
| --- | --- |
| `m, n, k` | 问题规模 |
| `lda, ldb, ldc` | 复数操作数前导维（以复数元素计） |
| `rowsA/colsA/rowsB/colsB` | 物理行列数（由 trans 决定），决定打包任务数 |
| `ldp` | 乘积平面前导维（= m） |
| `transA, transB, conjA, conjB` | 转置与共轭标志 |
| `alphaReal/alphaImag/betaReal/betaImag` | 复数标量 |
| `hasProduct` | 0 表示 `k=0` 或 `alpha=0`，只做 beta 缩放 |
| `aReOffset … p4Offset` | 各平面在 workspace 中的 float 偏移 |

#### 3.2.2 kernel 侧设计

##### 1. Pack Kernel（AIV）

按列分核：任务数 `colsA + colsB`，`for (task = blockIdx; task < total; task += blockNum)`。每列按 `kTileComplex = 1024` 个复数切块，每块：

1. `DataCopyPad` 搬入 2·n 个 fp32（交错 r/i/r/i…）；
2. `CreateVecIndex + Muls(2·sizeof(float))` 生成字节偏移表，`Gather` 基址 0 取实部、基址 `sizeof(float)` 取虚部；
3. 共轭时对虚部平面 `Muls(-1)`；
4. `DataCopyPad` 写实部/虚部平面（平面列 = 连续 `rowsA` 个元素）。

搬运与计算之间用 `SetFlag/WaitFlag<V_MTE2|MTE2_V|V_MTE3|MTE3_V>` 显式同步。

##### 2. 四个 FP32 GEMM

`aclblasSgemm(handle, transA', transB', m, n, k, 1.0, A_plane, lda, B_plane, ldb, 0.0, P_i, m)`，其中 `transA' = (transA != N) ? T : N`（共轭已折入平面），四次调用与 pack、epilogue 同在 handle stream 上串行下发，异步执行。

##### 3. Epilogue Kernel（AIV）

按列分核，每列按 `kTileComplex` 切块：

1. 搬入 `P1..P4` 与（`beta≠0` 时）旧 C；
2. `Sub(P1,P2)` 得实部乘积、`Add(P3,P4)` 得虚部乘积，再乘复数 alpha；
3. 叠加 `beta·C_old`：`beta_im == 0` 时走「两次单乘 + 加」快路径，否则走完整 4 乘 2 加；
4. 用预先构造的交错索引表 `Gather` 把 `[Re | Im]` 两平面合并成交错复数结果并写回 C。

`hasProduct=0` 时第 1 步只搬旧 C，第 2 步以零平面参与运算，等价于 `C = beta·C`；`beta=0` 时不搬旧 C，只写结果。

##### 4. 片上资源规划

单块 UB 预算（A2 单核 192 KB）：pack kernel ≈ 20 KB（输入 2·1024 fp32 + 两平面 + 索引表）；epilogue kernel ≈ 104 KB（4 个乘积块 + 旧 C/实虚平面 + 3 个临时 + 交错结果 + 三张索引表）。

##### 5. 数据搬运

- pack：读 `2kn + 2mk` fp32，写 `2kn + 2mk` fp32；
- epilogue：读 `4mn`（+ `2mn` 旧 C），写 `2mn` fp32。

以 4096³ 为例，额外搬运合计约 0.9 GB，相对 4×137 GFLOP 的 Cube 计算量可忽略；小尺寸用例靠 1024 元素块并行摊薄。

### 性能优化方案

1. 实数接口不引入任何额外数据搬运（直接使用原始 A/B/C 指针），完全继承 `blas/gemm/arch22` 的整问题多 Cube / 2D tile / 行分块路径。
2. 复数接口优先保证 4 次 GEMM 走同一套 SGEMM 调优路径；后续若 profiling 显示 pack/epilogue 占比偏高，再评估块大小（当前 1024）与列分核粒度的调整，或把四次乘积合并为两次双 K 展开。
3. 性能以 `msprof op` 的 kernel task duration 为准；测试侧对 `TC_PF` 用例跳过 cblas golden，改用 double 精度固定子集抽查，避免 CPU 标杆时间污染计时。

### 支持硬件

| 产品 | 架构目录 | 状态 |
| --- | --- | --- |
| Atlas A2 系列（800T A2 / 910B3） | `arch22` | 本任务实现与验证目标 |
| Atlas A3 系列 | `arch22` | 同一份实现，功能/精度需在 A3 复测 |
| Ascend 950PR / 950DT | `arch35` | 保持既有实现，仅同步 `SgemmEx` 17 参签名；需回归构建 |

### 算子约束限制

- 不支持广播、batch 维度和任意非列主序步长；A/B/C 按列主序前导维连续存放。
- C 原地覆写，不返回视图；按前导维访问的 padding 不参与计算且不得被覆盖。
- 不要求确定性（浮点乘加累加非 bit-exact）；特例：`alpha=0` 的 `C=beta*C` 按位精确校验。
- 扩展混合精度（FP16/BF16/FP8 等）不在本任务实现范围，一律返回 `NOT_SUPPORTED`，README 以实际支持集声明。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FP32 与 COMPLEX64（实部/虚部分别）：`rtol = atol = 2^-13`，逐元素 `|actual−golden| ≤ atol + rtol·|golden|`，matched_ratio ≥ 0.99，单元素误差 ≤ `max(1e-2, 32·ULP)` | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md` |
| 位精确项 | `alpha=0`（或其等价分支）的 `C=beta*C` 按 EXACT 校验 | 任务书 §2.5/§3.2 |
| 特殊值 | NaN/Inf 按「同为 NaN、同为同号 Inf、一方 NaN 一方 Inf 视为等效」判定，非有限值不计入 matched_ratio 与 max_abs_error | 任务书 §3.2 第 5 条 |
| 性能标准 | 910B3 上 warmup 5 次后取 ≥10 个有效样本的平均 task duration，单个用例 ≤ GPU 标杆 / 1.25（≥0.8×） | 任务书 §3.3 |

### 测试覆盖

| 类别 | 覆盖内容 |
| --- | --- |
| 基础与形状 | N/T/C 全组合小尺寸；1 → 2048 的尺寸扫描（含奇数、非对齐、2 的幂 ±1）；矩形与大规模（性能用例至 4096³） |
| 标量与填充 | alpha/beta 取 0/1/±0.5/100 等；均匀/正态/全零/交替/极端/Inf/NaN 填充；复数标量含纯虚与共轭 |
| 前导维 | `lda/ldb/ldc` 紧凑最小值与 +4/+8 padding |
| 边界与负向 | 零维 no-op、`k=0`、`alpha=0`、`beta=0` 不读 C、空指针（alpha/beta/A/B/C）、空 handle、非法枚举、非法 dtype（枚举外/合法不支持）、非法前导维、负维度、校验顺序叠加 |
| 性能 | 任务书 §3.3 的典型 case（1024/2048/4096 方阵与 1024³ transA=T）及尺寸扫描 |

三个接口各 1200 条（1000 精度/边界 + 200 性能），由任务包 CSV 提供，测试工程按表头自动识别接口；`alpha_real` 列存在时走 CgemmEx，`computeType`/`algo` 列存在时走 GemmEx，否则走 SgemmEx。

### 测试方法

1. 精度：`build.sh --soc=ascend910b3 --ops=gemm_ex,sgemm_ex` 编译；`gemm_ex_test`/`sgemm_ex_test` 以 `--gtest_filter=-*TC_PF*` 跑 1000 条精度/边界用例，负向用例只比较返回码，正向用例与 cblas 单标杆比对；`CgemmEx` 的 CSV 安装为 `test/gemm_ex/arch22/gemm_ex_test.csv` 后由同一二进制执行。
2. 特殊值：`TC_FL` 用例的 NaN/Inf 元素单独计数并逐元素按等效规则判定。
3. 性能：`msprof op --application="... --gtest_filter=*TC_PF_1003*"` 采集，取 `OpBasicInfo.csv` 中 Task Duration 均值；同时用任务包 `verify_performance.py` 做相对观察（其 gtest 计时含 host 与 golden 开销，仅作诊断）。
4. 报告记录代码 SHA、SoC、CANN 版本、用例文件哈希、命令、退出码、逐用例结果与性能原始样本；未执行或未取到有效样本的用例不得记为 PASS。

### 可复现命令

```bash
# 编译
bash build.sh --soc=ascend910b3 --ops=gemm_ex,sgemm_ex

# 精度（GemmEx / CgemmEx / SgemmEx）
build/test/gemm_ex/gemm_ex_test   --gtest_filter=-*TC_PF*
build/test/sgemm_ex/sgemm_ex_test --gtest_filter=-*TC_PF*

# 性能（任务书 §3.3 典型 case）
msprof op --application="./build/test/gemm_ex/gemm_ex_test --gtest_filter=*TC_PF_1003*" --output=./prof
```

## 兼容性分析

- 公共 ABI 变更：`aclblasSgemmEx` 由 15 参改为任务书规定的 17 参，属已发布符号签名变更；仓内 `arch35` 的实现、测试与调用方同步修改，并由维护者确认对外 ABI 迁移口径。
- 新增符号 `aclblasCgemmEx` 为纯增量，不影响既有接口。
- 实数路径复用 `aclblasSgemm`，其语义（含 `k=0`/`alpha=0` 缩放）与本任务一致；Ex 层仅做前置校验与早退，不改变既有算子的行为。
- 单算子构建依赖：`gemm_ex`/`sgemm_ex` 的构建会一并保留 `blas/gemm`、`blas/scal` 源目录，不改变全量构建的结果。
