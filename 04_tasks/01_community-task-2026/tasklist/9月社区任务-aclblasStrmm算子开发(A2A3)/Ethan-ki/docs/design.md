# 需求背景（required）

## 需求来源

9 月社区任务：在昇腾 NPU（Atlas 800T A2 (910B3) / Atlas 800I A3，arch22）上基于 Ascend C 开发单精度实数三角矩阵乘算子 `aclblasStrmm`。

- 任务书：`aclblasStrmm_A2A3_task_doc.md`（本文唯一语义真值源；配套 `test_cases/README.md` 若与任务书冲突，以任务书为准）
- 设计文档提交仓：`cann-ops-competitions`（`https://gitcode.com/cann/cann-ops-competitions`，任务书中的 `cann-competitions` 为其别名）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/9月社区任务-aclblasStrmm算子开发(A2A3)/Ethan-ki/docs/design.md`
- 代码合入仓：`https://gitcode.com/cann/ops-blas`
  - 实现：`blas/trmm/arch22/`
  - 测试：`test/trmm/strmm/arch22/`
- 对标接口：cuBLAS `cublasStrmm`；精度 golden 为 Netlib `cblas_strmm`
- 工程模式：kernel 直调（句柄式 BLAS），不走 ACLNN / tilingKey 注册

## 背景介绍

### aclblasStrmm 算子实现

在 ops-blas 开源仓框架下，用 Ascend C kernel 直调在 Atlas A2/A3（架构目录 `arch22`）实现 FLOAT32 三角矩阵-矩阵乘，结果离席写入独立输出 C：

```
side=LEFT :  C = alpha * op(A) * B      （A 为 m×m）
side=RIGHT:  C = alpha * B * op(A)      （A 为 n×n）
输出：C（m×n 一般矩阵，非三角输出）
```

`op(A)` 由 `trans` 决定：`N` → `A`，`T` → `Aᵀ`。TRMM 不允许 `ACLBLAS_OP_C`（实数语义下 C 转置无定义），本接口收紧为非法枚举。`uplo` 指定 A 的引用三角，`diag=UNIT` 时对角视为 1 且不读存储。矩阵列主序；A、B 只读，C 独立输出（允许 `B≡C` 原地等价用法，除此之外 C 与 A/B 不允许内存重叠）。

### 同类算子现状分析

当前 `cann/ops-blas` master 相关基线：

| 算子 / 目录 | 数据类型 | 仓内现状 | A2/A3 |
| --- | --- | --- | --- |
| `aclblasStrmm`（`blas/trmm/arch35/`） | FLOAT32 | host + kernel + tiling（Mirror→GEMM→Scale 三段流水）；`include/cann_ops_blas.h` 已有公共声明 | 不支持 |
| `blas/trmm/` | — | 仅 `arch35/`，无 `arch22/` | — |
| `test/trmm/strmm/` | — | 公共 `strmm_param.h` / `strmm_golden.h`（cblas golden）+ `arch35/` 测试，无 `arch22/` | — |
| `aclblasCtrsmBatched`（`blas/trsmbatched/arch22/`） | COMPLEX64 | arch22 MIX 内核（AIV panel + AIC Matmul 高阶 API + workspace）已落地，是本仓 arch22 Matmul 用法的在库先例 | 支持 |
| `blas/common` 复数 BLAS3（`symm/cherk/...` arch22） | COMPLEX64 | 验证过 arch22 上「实数拆分 + FP32 Matmul + 合并」路径 | 支持 |

结论：

1. 公共接口 `aclblasStrmm` 已在 `include/cann_ops_blas.h` 声明（与 `aclblasStrsm` 等并列），本任务**无需改动公共头**，只需为该声明补 arch22 后端。
2. arch35 `strmm` 的 **Mirror（三角补全稠密化）→ 稠密 GEMM → alpha Scale** 三段流水是经仓内评审过的同语义参考实现，算法结构可直接继承；arch22 与 arch35 硬件参数不同（arch22：L1=512KB / L0C=128KB / UB=192KB / Cube fract=32，20 AIC / 40 AIV），tile 尺寸与 workspace 布局按 arch22 重算。
3. arch22 侧已有两个成熟可复用的工程件：Matmul 高阶 API + `TCubeTiling` 的接入方式（`ctrsm_batched` arch22 AIC 路径）、`EnsureDefaultWorkspace` 的 workspace 管理方式（arch35 `strmm_host.cpp`）。

### aclblasStrmm 算子功能分析

**功能**：单精度实数三角矩阵乘，`C = alpha * op(A) * B`（LEFT）或 `C = alpha * B * op(A)`（RIGHT），结果写入独立输出 C。

**输入**：handle、side、uplo、trans、diag、m、n、alpha（标量指针）、A（m×m 或 n×n 三角阵，只读）、lda、B（m×n，只读）、ldb、C（m×n，输出）、ldc。

**输出**：C 的逻辑 m×n 区域写为乘积结果；`ldc` padding 行不触碰。`alpha=0` 时 A、B 不被引用（可不合法），C 置零。`m=0` 或 `n=0` 为合法 no-op。

**支持数据类型**：仅 FLOAT32。

**支持广播**：不涉及。无 dynamic shape 要求（m/n 为运行时 Host 标量）。

**异步**：依赖 `aclblasSetStream` 绑定 stream；三段 kernel 同 stream 顺序发射，读回 C 前须同步。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 ops-blas 的 `arch22` 路径实现 `aclblasStrmm`，接口与仓内既有声明及 cuBLAS `cublasStrmm` 参数序列对齐，覆盖 LEFT/RIGHT × UPPER/LOWER × N/T × NON_UNIT/UNIT 共 16 种组合，精度达到生态 FLOAT32 混合容差，性能不超过任务书 §3.3 达标耗时。

函数原型（任务书 §2.3，公共头已有声明，保持不变）：

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc);
```

## 需求拆解

1. `include/cann_ops_blas.h` 不改（声明已存在）；`blas/trmm/README.md` 产品支持表将 Atlas A2/A3 标为支持。
2. 实现放 `blas/trmm/arch22/`，文件名前缀 `strmm_`，由顶层 CMake 按 `SOC_ARCH_DIRS=arch22` 自动编入（与 arch35 互斥编入，无符号冲突）。
3. Host 参数校验与错误码严格按任务书 §2.4：**非法枚举（含 trans=OP_C）返回 `ACLBLAS_STATUS_INVALID_VALUE`**（arch35 同接口当前对非法枚举返回 `INVALID_ENUM`、且接受 OP_C，arch22 按任务书收紧，见差异表）。
4. Kernel 直调：handle 取 stream；A/B/C 为 Device 指针（非指针数组）。
5. 覆盖 16 种枚举组合、lda/ldb/ldc padding、alpha=0 置零（A/B 不校验）、m=0/n=0 no-op、UNIT 跳过对角读取、`B≡C` 原地安全。
6. 精度：cblas `strmm` 作 golden，FLOAT32 混合容差（rtol=atol=2⁻¹³，matched_ratio≥0.99，max_abs≤1e-2 或 32ULP）。
7. 性能：Atlas 800T A2 (910B3) 上 5 条标杆 Avg time 不超过任务书表。
8. 测试：`test/trmm/strmm/arch22/`，CSV 驱动 GTest，复用公共 `strmm_param.h` / `strmm_golden.h`；以任务配套 `strmm_test.csv`（1200 条，21 列含 nullA/nullB/nullC 标志）为底。

# 详细设计（required）

## 算子分析

### 数学公式

记 `k = (side==LEFT) ? m : n`。A 为 `k×k` 三角阵（`lda×k` 存储），B/C 为 `m×n`（`ldb×n` / `ldc×n` 存储），列主序。

```
LEFT :  C = alpha · op(A) · B
RIGHT:  C = alpha · B · op(A)
```

| trans | op(A) |
| --- | --- |
| `ACLBLAS_OP_N` (111) | A |
| `ACLBLAS_OP_T` (112) | Aᵀ |
| `ACLBLAS_OP_C` (113) | 非法枚举，Host 返回 `ACLBLAS_STATUS_INVALID_VALUE`（TRMM 实数语义不接受） |

乘法语义与 TRSM 求解类的差异：无回代、无对角非零约束——`diag=NON_UNIT` 时对角元正常读取且**允许为零**（全零 A 为合法输入，结果为全零 C）；`diag=UNIT` 时对角视为 1 且不读取存储。

`alpha = 0`：A、B 不被引用，C 的逻辑 `m×n` 区域置零（保留 `ldc` padding 原值），返回成功。

确定性：不要求逐位一致（浮点累加顺序不保证）。

### 支持数据类型

仅 FLOAT32。按生态标准 FLOAT32 档混合容差判定。

### 支持形状

- `m ≥ 0`，`n ≥ 0`；LEFT：`lda ≥ max(1,m)`；RIGHT：`lda ≥ max(1,n)`；`ldb ≥ max(1,m)`；`ldc ≥ max(1,m)`
- `m=0` 或 `n=0`：合法 no-op，不启动任何 kernel
- 无广播、无超出 lda/ldb/ldc 语义的非连续访问
- `B≡C` 原地允许：三段流水在 Scale 段前不写 C，天然安全；其余 C 与 A/B 重叠未定义

## 算子实现

### 实现方案

工程文件（任务书 §5 + ops-blas 目录规范，与 arch35 同构的三文件形态）：

```
blas/trmm/arch22/
├── strmm_host.cpp          # 校验、alpha 解析、tiling 计算、workspace、三段 launch
├── strmm_kernel.cpp        # mirror(AIV) + gemm(AIC Matmul) + scale(AIV) 三个 kernel
└── strmm_tiling_data.h     # Host/Device 共享 POD（三段各一结构）

test/trmm/strmm/
├── CMakeLists.txt          # 已存在，不改
├── strmm_param.h           # 已存在，复用（21 列 CSV 解析，含 nullA/nullB/nullC）
├── strmm_golden.h          # 已存在，复用（cblas golden）
└── arch22/
    ├── strmm_test.cpp      # CSV 驱动 GTest（arch22 SOC 下自动替代 arch35 用例）
    └── strmm_test.csv      # 任务配套 1200 条
```

#### 三段式流水（与 arch35 同构，按 arch22 重排）

```
A(三角,GM) ─[K1 Mirror, AIV]→ workspaceA(稠密 k×k, GM)
workspaceA + B ─[K2 GEMM, AIC Matmul]→ workspaceT(n × tempRowStride, GM)
workspaceT ─[K3 Scale, AIV]→ alpha 缩放并按列写 C(m×n, 保留 padding)
```

三段同 stream 顺序发射，依赖由 stream 顺序保证，无需 device 侧跨核同步原语。

**K1 Mirror（AIV，VECTOR_ONLY）**：把三角 A 补全为稠密方阵 `Ã` 写入 workspace。
- 只读 `uplo` 指定三角；`diag=UNIT` 时对角写 1.0（不读存储），`NON_UNIT` 时原值拷贝；非引用三角写 0。
- **mirror 不感知 trans**：恒产出非转置 `Ã`，`op(A)` 语义全部交由 K2 的 Matmul 转置标志消化（消除 trans×uplo 4 种镜像分支，mirror 内部只按 uplo/diag 两种行为实现）。
- 按列分片并行：AIV 核 t 负责列区间 `[t·colPerCore, min(k, (t+1)·colPerCore))`；列内先向量化填零非引用段，再 `DataCopy` 引用段，对角单点写。输出步长 `ldw = CeilAlign(k, 32)`（对齐 K2 读带宽）。

**K2 GEMM（AIC，Matmul 高阶 API）**：稠密 FP32 GEMM，`Cᵀ = op1 × op2`。

列主序→行主序视角转换（与 arch35 `strmm_host.cpp` 同一性质：列主序存储即行主序转置），在行主序视角下统一为：

```
M' = n, N' = m, K = k
side=LEFT :  Cᵀ(n×m) = Bᵀ(n×k) · op(A)ᵀ(k×m)     op1 = B 直读(flag=false)，op2 = workspaceA
side=RIGHT:  Cᵀ(n×m) = op(A)ᵀ(n×k) · Bᵀ(k×m)     op1 = workspaceA，op2 = B 直读(flag=false)
op(A)ᵀ 的转置语义由 isTrans 标志消化：
  side=LEFT  且 trans=N → op2 isTrans=true；LEFT  且 trans=T → false
  side=RIGHT 且 trans=N → op1 isTrans=false；RIGHT 且 trans=T → true
  （合成规则：A 侧操作数 isTrans = (side==LEFT) == (trans==OP_N)）
```

- `Matmul<GM,ND,float ×3>` + `MultiCoreMatmulTiling`：`SetDim(usedAic)`、`SetOrgShape/SetShape(M', N', K)`、A/B/C 均ND-FP32；`REGIST_MATMUL_OBJ(GetSysWorkSpacePtr())`（接入方式与 `ctrsm_batched` arch22 AIC 路径一致）；`IterateAll` 输出到 workspaceT。
- tile 尺寸：`tileM/tileN ∈ {64,128}`、`tileK ∈ {64,128}`，Host 按 `L1 512KB 双缓冲 ≤ 0.9` 与 `L0A/L0B 64KB、L0C 128KB` 约束从大到小收敛（对齐 arch35 `CalGemmTilingData` 的收敛算法，常数换为 arch22 硬件参数）；小 shape（如 case1 256×256）按实际维度裁齐，保证 ≤48 核满载分片。
- `tempRowStride = CeilAlign(n, 32)`（float 数），K2 每行输出与 K3 读取共用该步长。

**K3 Scale（AIV，VECTOR_ONLY）**：`C[:, j] = alpha · T[j, 0:m]`。
- workspaceT 行 j 即 C 列 j（Cᵀ 行主序存储），行内连续 m 个 float；向量化 Muls 后按 `DataCopyPad` 写 `C + j·ldc`，只写 m 个激活行，padding 不触碰。
- 按行分片并行；`alpha=1` 时退化为纯拷贝（同一 kernel，标量分支）。

**B≡C 原地安全性**：K1/K2 不写 C，K3 读 workspaceT 写 C，原地调用无读写交叠。

#### host 侧设计

入口流程：

```
1. handle == nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. side/uplo/diag 枚举校验；trans 仅接受 N/T（OP_C 及其它值非法）
3. m < 0 || n < 0 → INVALID_VALUE
4. m == 0 || n == 0 → SUCCESS 快速返回（先于 ld/指针校验，BLAS 惯例）
5. lda/ldb/ldc 前导维约束 + alpha nullptr → INVALID_VALUE
6. C == nullptr → INVALID_VALUE
7. alpha == 0 → C 激活区 m×n 置零（MemsetAsync 按列/整块，保留 padding），A/B 不校验不读取 → SUCCESS
8. A/B == nullptr → INVALID_VALUE
9. 计算三段 tiling；EnsureDefaultWorkspace(workspaceA + workspaceT)；同 stream 顺序发射 K1/K2/K3，异步返回
```

**参数校验表（任务书 §2.4）**：

| 条件 | 返回码 |
| --- | --- |
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| side/uplo/diag 不在规定枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| trans 不为 N/T（含 OP_C） | `ACLBLAS_STATUS_INVALID_VALUE` |
| m<0 或 n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda/ldb/ldc 不满足前导维约束 | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| C 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha≠0 且 A/B 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |

**与 arch35 同接口实现的差异（任务书要求，评审关注点）**：

| 项 | arch35 现状 | arch22（本任务） | 依据 |
| --- | --- | --- | --- |
| trans=OP_C | 接受（按 T 处理） | 非法枚举，`INVALID_VALUE` | 任务书 §2.4 明确收紧 |
| 非法枚举返回码 | `ACLBLAS_STATUS_INVALID_ENUM` | `ACLBLAS_STATUS_INVALID_VALUE` | 任务书 §2.4 异常行为列 |
| alpha 位置 | 兼容 Host/Device（Device 时 D2H 同步读回） | 按 Host 指针直读（任务书 §2.4「Host 内存」；不再引入同步点） | 任务书 §2.4 |
| 零维快速返回位置 | 枚举后、ld 校验前 | 同（保持一致） | BLAS 惯例 / CSV TC_ED |

**workspace（EnsureDefaultWorkspace，handle 复用）**：

```
workspaceA：ldw × k × 4B，ldw = CeilAlign(k, 32)
workspaceT：n × tempRowStride × 4B，tempRowStride = CeilAlign(n, 32)
总量：k=4096 时约 67MB + 67MB，Host 侧一次申请、容量检查，失败返回分配错误码
```

#### tiling 结构（无 tilingKey，Host/Device POD 布局一致）

```cpp
struct StrmmMirrorTilingData {   // K1
    uint32_t side, uplo, diag;
    uint32_t dimA, lda, ldw;
    uint32_t usedAivCoreNum, colsPerCore;
};
struct StrmmGemmTilingData {     // K2
    uint32_t m, n, k;
    uint32_t isTransAop;         // A 侧操作数转置标志（含 side/trans 合成）
    uint32_t aIsWorkspace;       // A 侧操作数选 workspaceA(LEFT:op2 / RIGHT:op1)
    uint32_t tileM, tileN, tileK;
    uint32_t lda, ldw, ldb, tempRowStride;
    uint32_t usedAicCoreNum;
    TCubeTiling cubeTiling;      // 二进制嵌入或 GM 分开传递
};
struct StrmmScaleTilingData {    // K3
    uint32_t m, n, ldc, tempRowStride;
    float alphaVal;
    uint32_t usedAivCoreNum, rowsPerCore;
};
```

#### kernel 侧补充设计

- **同步**：三段 kernel 无跨核 flag 依赖；K2 的 Matmul 高阶 API 内部同步由 API 保证；`GetSysWorkSpacePtr` 系统workspace 按 CANN 9.1 框架要求传入。
- **Inf/NaN**：不做清洗，按 IEEE 754 传播；填充类用例（B 含 Inf/NaN）按生态标准 NaN / 同号 Inf 判等。
- **数值格式**：全程 FLOAT32 原生，无复数拆分、无格式转换层；K2 为纯实数单 GEMM（较复数 BLAS3 的 4 GEMM 负载轻）。
- **性能风险与备选**：稠密化使 GEMM 有效负载为三角感知实现的 2 倍。以最大 case5（4096³，6681us 达标线）计，dense 方案需约 20.6 TFLOPS 有效 FP32 算力，与仓内 arch22 已合入 Matmul 类算子的实测水平同量级、可达；若实测未达标，备选方案为复用 `ctrsm_batched` arch22 已验证的 MIX 分块框架（panel AIV 三角乘 + trailing Cube GEMM），按三角非零结构减半 GEMM FLOPs，接口与校验层不变。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B / arch22） | √ |
| Atlas A3 训练/推理系列（arch22） | √ |
| Ascend 950PR/DT（arch35） | 不涉及（arch35 已有实现） |

自验证覆盖一种款型即可；精度建议 910B3 或 A3，性能建议 910B3。

## 算子约束限制

- 仅 FLOAT32，列主序，单矩阵（无 batch、无指针数组）
- 不支持超出 lda/ldb/ldc 的非连续访问，不做广播
- `trans` 仅 OP_N/OP_T（OP_C 判非法，与 cuBLAS 字面行为的差异已在任务书声明）
- `diag=UNIT` 时不读对角存储；`diag=NON_UNIT` 允许对角为零（乘法语义，不做奇异性检测）
- `alpha=0` 时 A/B 不引用、不校验；C 仍置零
- C 为独立输出；仅允许 `B≡C` 原地等价用法；不返回视图
- alpha 按 Host 指针（较 arch35 行为收紧，任务书 §2.4）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = `cblas_strmm`（复用 `test/trmm/strmm/strmm_golden.h`）；比对 C 的逻辑 m×n。FLOAT32：rtol=atol=`2^-13`≈1.22e-4，`matched_ratio≥0.99`，`max_abs_error ≤ 1e-2 或 32*ULP`；逐元素 `|actual-golden| ≤ atol + rtol×|golden|`；含规约可放宽 max_abs 至 2ULP | 任务书 §3.2；[混合容差](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 设备 Atlas 800T A2 (910B3)。warmup 后有效采样 >10 次取 Avg time (us)，不超过下表 | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

性能标杆（达标线 = GPU 实测 ÷ 0.8，GPU 实测见配套 `gpu_baseline.csv`）：

| case | m | n | side | uplo | trans | diag | 达标耗时 (us) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 58.429 |
| 2 | 512 | 512 | LEFT | LOWER | T | NON_UNIT | 138.2 |
| 3 | 1024 | 1024 | LEFT | UPPER | N | UNIT | 342.2 |
| 4 | 2048 | 2048 | RIGHT | LOWER | N | NON_UNIT | 1182 |
| 5 | 4096 | 4096 | LEFT | UPPER | T | NON_UNIT | 6681 |

测试工程：任务配套 `strmm_test.csv`（1200 条 = 1000 精度 + 200 性能，21 列含 nullA/nullB/nullC/非法枚举/非法前导维/负维度/零维/alpha=0 边界），CSV 驱动 C++ GTest（`test/trmm/strmm/arch22/`，arch22 SOC 构建时自动替换 arch35 用例），`bash build.sh --ops=strmm --run --soc=ascend910b3`。golden 在 Host 调 cblas；负向用例只查返回码不比数值；`TC_PF_` 前缀性能用例经 msprof 采集 kernel 平均耗时后与 `gpu_baseline.csv` 逐条比对。

## 兼容性分析

公共头已有声明的补后端任务，不改任何既有符号与行为：arch35 路径（Ascend 950）继续使用原 `blas/trmm/arch35/` 实现，arch22 路径新增文件由 CMake `SOC_ARCH_DIRS` 机制互斥编入，两后端对同一声明的行为差异仅限任务书明确要求的两点（OP_C 判非法、非法枚举返回 `INVALID_VALUE`），不影响 arch35 既有验收。CANN 9.1.0。
