# aclblasCsyrk3mEx 算子设计文档

> 目标芯片：昇腾 950PR（arch35 / dav-c220，SOC=ascend950）｜CANN 9.1.0 及以上
> 数据类型：A ∈ {complex64, complex-int8((int8,int8) 交错 2B)}；C = complex64；alpha/beta = COMPLEX64 复标量
> 合入位置：`ops-blas/blas/syrk/arch35/`；测试：`ops-blas/test/syrk/csyrk3m_ex/arch35/`
> 任务书：`aclblasCsyrk3mEx_Atlas950PR_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasCsyrk3mEx 算子开发（昇腾 950PR，arch35）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/syrk/arch35/`（§1/§5）；测试代码 → `test/syrk/csyrk3m_ex/arch35/`（新建子目录并注册构建目标，§2.2）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 接口现状：`include/cann_ops_blas.h` **尚无 `aclblasCsyrk3mEx` 声明（grep 实证 0 命中），本任务新增该声明**——公共头新增（仅增不改），波及面见"兼容性分析"。

## 背景介绍

### 算子功能定位

复数**对称**矩阵秩-k 更新（3M 扩展版）：用对称专用 3M 分解把复数秩-k 乘积 X·X^T 化为 3 次实数矩阵乘，对 C 的 uplo 三角原地更新：

```
C = alpha * op(A) * op(A)^T + beta * C      （C 为 n×n 对称，仅 uplo 三角读写；对角虚部为有效数据不置 0）
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCsyrk3mEx | cuBLAS `cublasCsyrk3mEx`｜Netlib `csyrk`（数学语义） | complex64 / complex-int8 → complex64 的对称秩-k 更新（3M 加速） |

### 业界对标

| 项 | cuBLAS cublasCsyrk3mEx | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/trans/n/k/alpha/A/Atype/lda/beta/C/Ctype/ldc | 逐参数一致（§2.3） |
| trans 取值 | N/T/C | 同——**对称语义 OP_C ≡ OP_T 不共轭**（§2.1.2） |
| Atype 组合 | (CUDA_C_8I, CUDA_C_64F) 与 (CUDA_C_64F, CUDA_C_64F) | 同：ACL_INT8/ACL_COMPLEX64 → ACL_COMPLEX64；其它 → NOT_SUPPORTED（§2.1.5） |
| 标量 | alpha/beta 复数 | 同（COMPLEX64；Host/Device 指针位置均支持，§2.5） |
| 对称语义 | 仅 uplo 三角读写；对角虚部有效数据 | 同（计算后原样保留，不置 0，§2.1.3） |
| 舍入 | 3M 与直接复数乘舍入不同 | 同——与 aclblasCsyrkEx/cublasCsyrk 不保证逐位一致，按 §3.2 阈值判定 |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/syrk/arch35/` 下**新建 `csyrk3m_ex_host.cpp` / `csyrk3m_ex_kernel.cpp`**（或按仓内规范扩展现有 csyrk host/kernel，§2.2）；与同目录非 Ex `aclblasCsyrk` 多件套并存但**互不依赖**（aclblasCsyrkEx 亦为未合入新接口，禁止调用，§2.2 末段）。
- 公共头：`include/cann_ops_blas.h` **新增** `aclblasCsyrk3mEx` 声明（§2.3 签名，含 Atype/Ctype 两个 aclDataType 参数）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/syrk/arch35/csyrk_*`（非 Ex，已合入） | ★ 主模板 | host：ValidateCsyrkParams、ReadComplexScalarsFromDevice（复数标量双位置）、三段 tiling、CsyrkEnsureWorkspace 防溢出；kernel：deinterleave→cube→combine 三段、对称 uplo 三角处理、对角虚部保留的 combine 语义 | 其 GEMM 段多次 launch——3M 版改**单 launch 三积融合**；特值 Netlib 参考 fallback 不需要（host 短路替代） |
| `blas/gemm/arch35/cgemm3m_*` | ◆ 参考 | 3M 通用分解代数（Re=P0−P1、Im=P2−P0−P1 与本任务 X·X^T 令 B=X^T 完全同式）、融合纠正路径 | 直接套通用式；单 launch 化 |
| Cherk3mEx（同批姊妹任务设计） | ◆ 参考 | 单 launch 三累加器融合 kernel 骨架、tile 预算 | 分叉：P2 两边同和式、复数 alpha/beta、对角不置 0、OP_C≡T（两任务不建代码依赖） |
| `test/syrk/{csyrk,ssyrk}` 工程 | ◇ 参考 | ATK 测试组织 | 新建 csyrk3m_ex/ 子目录 |

### 接口规格

需在 `include/cann_ops_blas.h` 新增声明（§2.3，与 cuBLAS `cublasCsyrk3mEx` 逐参对应）：

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

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| uplo | C 的存储三角模式 | attr | int | {UPPER, LOWER}；其它 → `INVALID_ENUM` | 标量 |
| trans | op(A) 运算类型 | attr | int | {OP_N, OP_T, OP_C}，**OP_C≡OP_T 不共轭**；其它 → `INVALID_ENUM` | 标量 |
| n | 方阵 C 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |
| k | 秩-k 更新的 k 维 | attr | int | ≥0；负 → `INVALID_VALUE`；0 时秩-k 贡献为 0 | 标量 |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 实/虚部任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| A | 输入矩阵（列主序，Device） | tensor | COMPLEX64 / COMPLEX8I | N: n×k；T/C: k×n；n>0∧k>0 时 nullptr → `INVALID_VALUE` | N: lda×k；否则 lda×m 逻辑 |
| Atype | A 的 dtype 枚举 | attr | int | {ACL_COMPLEX64, ACL_INT8}；不在支持集或组合不支持 → `NOT_SUPPORTED` | 标量 |
| lda | A 前导维 | attr | int | N: ≥max(1,n)；非 N: ≥max(1,k)；低于下界 → `INVALID_VALUE` | 标量 |
| beta | 复数标量乘数 | scalar | COMPLEX64 | 同 alpha；nullptr → `INVALID_VALUE` | 标量 |
| C | n×n 输出（列主序，原地覆写，仅 uplo 三角读写，对角虚部保留） | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE` | ldc×n（逻辑 n×n） |
| Ctype | C 的 dtype 枚举 | attr | int | {ACL_COMPLEX64}；其它 → `NOT_SUPPORTED` | 标量 |
| ldc | C 前导维 | attr | int | ≥max(1,n)；低于 → `INVALID_VALUE` | 标量 |

> Atype=ACL_INT8 语义：每个复元素 (int8 real, int8 imag) 交错 2 字节（对应 cuBLAS CUDA_C_8I），参与计算前**精确提升**为 float（§2.4/§3.2）。

### 数学公式

```
C = alpha * op(A) * op(A)^T + beta * C
op(A) = A（trans=N，A 为 n×k）；op(A) = A^T（trans=T 或 C，A 以 k×n 存储，不共轭）
```

对称专用 3M 分解（§2.1.7，X = op(A) = Xr + i·Xi，int8 先精确提升）：

```
乘积为 X·X^T（Br = Xr^T，Bi = Xi^T）：
P0 = Xr·Xr^T，P1 = Xi·Xi^T，P2 = (Xr+Xi)·(Xr+Xi)^T     ← 两边同和式
Re(X·X^T) = P0 − P1
Im(X·X^T) = P2 − P0 − P1
```

⚠️ 与 Hermitian 的 X·X^H 分解不同（后者 P1 取正号、P2 用和差式并置对角虚部 0）；错误套用会把虚部算成 Xi·Xr^T − Xr·Xi^T（正确值 Xi·Xr^T + Xr·Xi^T），属实现错误（§2.1.7 点名）。对角元素 C(i,i) 的虚部为**有效数据**，计算后原样保留（§2.1.3）。三次实数矩阵乘**必须在同一次 kernel launch 内完成**。

# 需求分析

## 需求描述

用 Ascend C 在昇腾 950PR（arch35）上实现新增句柄式接口 `aclblasCsyrk3mEx`：单 launch 融合 kernel 完成对称专用 3M 三乘积（P0/P1/P2 同轮累加）与 uplo 三角 epilogue（复数 alpha/beta 缩放、对角虚部保留、交织写回），支持 Atype∈{complex64, complex-int8}、trans∈{N,T,C}（C≡T）、复数标量双指针位置；精度按 FP32 分量达生态混合容差（golden = 非 3M cblas_csyrk），性能达 GPU 标杆双条件（单 case ≥0.4× 且全 case 均值 ≥0.5×）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 对称 3M（§2.1.7 公式，与通用 cgemm3m 的 X·Y 分解令 B=X^T 完全同式）；uplo 三角/对角保留/复数 epilogue 可整体复用非 Ex csyrk 的 combine 段语义 |
| 2 | 接口原型 | §2.3 签名（13 参数，含 Atype/Ctype，alpha/beta 为 COMPLEX64）；**公共头新增声明**（仅增不改） |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4（含 NOT_SUPPORTED 类） |
| 4 | 支持数据类型 | A: COMPLEX64 或 INT8 复（2B 交错）；C: COMPLEX64；alpha/beta: COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | 950PR → arch35；构建 `--soc=ascend950` |
| 6 | 精度要求 | §3.2：FP32 分量 rtol=atol=2^-13、ratio ≥0.99、max_abs ≤ 1e-2 或 32 ULP；golden=cblas_csyrk（int8 精确提升后调用；FP64 用 cblas_zsyrk）；3M 舍入差异由阈值吸收 |
| 7 | 性能预期 | §3.3：**双条件**——单 case NPU ≤2.5×gpu_ms（≥0.4×）且全 200 case 加速比均值 ≥0.5×；msprof op Task Duration（warmup5+10）为正式判据；gpu_baseline 实测 max=8.6362ms / avg=1.7164ms |
| 8 | 代码架构选型 + 可行性 | deinterleave（含 int8 提升）+ **单 launch 融合 kernel（三累加器同轮乘加 + uplo epilogue）**两段；已排除死路：三次独立 GEMM 拼接（§2.1.7 明令禁止）、调用未合入 aclblasCsyrkEx（§2.2 禁止）、Hermitian X·X^H 分解（数学错误）；可行性旁证 = 非 Ex csyrk 的 deinterleave/combine 段与 cube 基建均已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/（1201 行 csv = 精度 1000 含 TC_3M_* 4 条 + 性能 200；gpu_baseline 201 行） |
| 10 | 特值与退化 | n=0 no-op；k=0 或 alpha=(0,0)：beta=(1,0) → quick return 不动 C；beta≠(1,0) → 仅 uplo 三角 C=beta·C（复数 beta，不引用 A，A 可 null）；beta=(0,0) 不读 C 旧值 |
| 11 | 错误码口径 | 非法 uplo/trans → INVALID_ENUM；维度/ld/指针 → INVALID_VALUE；Atype/Ctype 不支持 → NOT_SUPPORTED |
| 12 | workspace | Xr/Xi 实平面（deinterleave 暂存，int8 提升同址折入）；融合 kernel 累加器驻核内；沿用 csyrk workspace 防溢出口径（CsyrkMulU64/AddU64 模式） |

# 详细设计

## 算子分析

### 数学公式

逐元素（i, j 属于 uplo 三角，含对角）：

```
C[i,j] = alpha * Σ_{t=0..k-1} X[i,t]·X[j,t] + beta * C[i,j]
X = op(A)（trans=N: X=A 的 n×k；trans=T/C: X=A^T，A 以 k×n 存储，不共轭）
```

3M 展开（FP32 实平面，Xr/Xi 为 X 的实/虚部平面；int8 先精确提升）：

```
P0[i,j] = Σ Xr[i,t]·Xr[j,t]
P1[i,j] = Σ Xi[i,t]·Xi[j,t]
P2[i,j] = Σ (Xr+Xi)[i,t]·(Xr+Xi)[j,t]
Re(P)[i,j] = P0[i,j] − P1[i,j]
Im(P)[i,j] = P2[i,j] − P0[i,j] − P1[i,j]
C[i,j]   = alpha·P[i,j] + beta·C[i,j]（复数 epilogue：αr·Re−αi·Im / αr·Im+αi·Re）
对角 C[i,i] 虚部为有效数据，计算后原样保留（不置 0）
```

uplo 三角外元素不读不写（由 C=C^T 隐含）；ldc padding 列不动。特值：alpha=(0,0) 或 k=0 → P 项跳过，仅 `C = beta·C`（uplo 三角内，复数 beta）；beta=(0,0) → C 旧值不读取（含 NaN 安全）；beta=(1,0) 且（alpha=(0,0) 或 k=0）→ 完全 quick return 不动 C。

### 支持数据类型

| 对象 | 接口 dtype | kernel 内部 |
|---|---|---|
| A（Atype=ACL_COMPLEX64） | complex64（8B 交织） | 拆为 Xr/Xi FP32 平面 |
| A（Atype=ACL_INT8） | (int8,int8) 交错 2B（CUDA_C_8I 对应） | deinterleave 时**精确提升**为 FP32 平面 |
| C（Ctype=ACL_COMPLEX64） | complex64 | 交织读写（对角虚部保留） |
| alpha/beta | COMPLEX64（Host/Device 指针位置均可） | tiling 传 re/im 分量 |

支持组合（§2.1.5）：(INT8→C64) 与 (C64→C64)；其它 → NOT_SUPPORTED。

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥ 0 | 0 → SUCCESS no-op，不引用 A/C、不下发 kernel |
| k | ≥ 0 | 0 → 秩-k 贡献 0：beta=(1,0) quick return；beta≠(1,0) 仅 uplo 三角 beta 缩放 |
| lda | N: ≥max(1,n)；非 N: ≥max(1,k) | 低于下界 → INVALID_VALUE |
| ldc | ≥max(1,n) | 低于下界 → INVALID_VALUE |
| 覆盖档位（§3.5） | n ∈ {0..4096 全档}，k ∈ {0..2048 全档}；uplo × trans(N/T/C) 全组合 | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

两段式（3M 三乘积在**同一次 kernel launch** 内，§2.1.7 硬约束）：

```
aclblasCsyrk3mEx (csyrk3m_ex_host.cpp)
 └─ 校验（含 Atype/Ctype 支持集）/ quick return / 复数标量双位置读取
 └─ Phase 0  csyrk3m_ex_deinterleave_kernel（AIV）:
        complex64 或 int8 复 → Xr / Xi 实平面（列主 packed；int8 精确提升折入 cast；
        trans=T/C 时按 A^T 读——下标换位，无共轭符号）
 └─ Phase 1  csyrk3m_ex_kernel（单次 launch，融合三积 + epilogue）:
        逐 uplo 三角 C-tile × K 分块:  加载 Xr/Xi 面板 → 三积同轮累加
            acc0 += Xr·Xr^T ; acc1 += Xi·Xi^T ; acc2 += s·s^T（s = Xr+Xi，核内合成单和式）
        Re=acc0−acc1, Im=acc2−acc0−acc1 → C = alpha·P + beta·C（uplo 三角，复数 epilogue）
        对角虚部保留；ldc padding 哨兵；beta=0 不读 C
特值旁路: alpha=(0,0)/k=0 → 仅 uplo 三角 beta 缩放 kernel（beta=(1,0) 直接 SUCCESS 不发 kernel）
```

kernel 变体划分表：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| csyrk3m_ex_deinterleave_kernel | 常规路径（Atype=C64/INT8 两形态） | 交织复面（或 int8 复面）→ Xr/Xi FP32 平面（trans=T/C 换位读） |
| csyrk3m_ex_kernel | 常规路径 | **单 launch 内三次实乘同轮累加 + 3M combine + 复数 alpha/beta + uplo 交织写回（对角虚部保留）** |
| beta 缩放旁路 kernel | alpha=(0,0) 或 k=0 且 beta≠(1,0) | 仅 uplo 三角 C=beta·C（复数 beta；beta=(0,0) → 写 0） |

数据流：GM 交织 A →(deint)→ GM Xr/Xi 平面 →(融合 kernel：面板进核)→ 核内三累加器 → 3M/alpha/beta/uplo → GM C。

#### host 侧设计

- **分核策略**：融合 kernel 按 C 的 uplo 三角元素量切核（三角块行/列带划分，满核原则 + 尾核余块；参照非 Ex csyrk 的 CalCubeTilingData/CalCombineTilingData 口径）；deinterleave 按 A 平面元素均分。
- **数据分块和内存优化策略**：C-tile 尺寸按三累加器并存预算取保守档（arch35 分形基准下调）；和式 s=Xr+Xi 在核内由面板合成，不落 GM；K 分块循环累加；workspace 防溢出沿用 CsyrkMulU64/AddU64 模式。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| uplo | 0/1 | Upper/Lower 三角路径 |
| trans | 0/1 | N / T·C（换位读，无共轭） |
| atype | 0/1 | C64 / INT8（deinterleave cast 形态） |
| alpha/beta re/im | FP32 分量 | 复数 epilogue；alpha=(0,0)/beta=(1,0)/beta=(0,0) 特值分支 |

- **数据检测**（校验顺序逐条对齐 §2.5 注；结构合法后再 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | uplo ∉ {UPPER, LOWER} | INVALID_ENUM |
| 3 | trans ∉ {OP_N, OP_T, OP_C} | INVALID_ENUM |
| 4 | n<0 或 k<0 | INVALID_VALUE |
| 5 | lda 低于下界（按 trans） | INVALID_VALUE |
| 6 | ldc < max(1,n) | INVALID_VALUE |
| 7 | alpha/beta == nullptr | INVALID_VALUE |
| 8 | Ctype ≠ ACL_COMPLEX64 或 Atype ∉ {ACL_COMPLEX64, ACL_INT8} | NOT_SUPPORTED |
| 9 | n>0 时 C == nullptr | INVALID_VALUE |
| 10 | n>0∧k>0 时 A == nullptr | INVALID_VALUE |
| 11 | 全过后：n==0 → SUCCESS（不下发 kernel） | — |
| 12 | k==0 或 alpha=(0,0) → beta=(1,0) 直接 SUCCESS；否则仅 beta 缩放旁路（不引用 A） | — |

  标量读取：alpha/beta 指针位置自动识别（复用 csyrk 的 ReadComplexScalarsFromDevice 模式，Host/Device 双位置）。

#### kernel 侧设计

- **Init 阶段**：blockIdx → 本核 C-tile 集（uplo 三角映射）；绑定 Xr/Xi 平面与 C 的 GM tensor；UB 布局三累加器 + 面板 buffer + epilogue 暂存。
- **Process 阶段**（融合 kernel 主循环）：

```
for (jt, it) ∈ 本核三角 tile 集:
    acc0/acc1/acc2 := 0
    for t ∈ K 分块:
        载 Xr[it,t], Xr[jt,t], Xi[it,t], Xi[jt,t] 面板（trans=T/C 已在 deint 换位）
        acc0 += Xr[it]·Xr[jt]^T          ← 实乘 1
        acc1 += Xi[it]·Xi[jt]^T          ← 实乘 2
        s := Xr+Xi（核内合成）
        acc2 += s[it]·s[jt]^T            ← 实乘 3（同一 launch 内）
    Re := acc0−acc1 ; Im := acc2−acc0−acc1
    C[tile] = alpha·(Re+i·Im) + beta·C[tile]（beta≠(0,0) 时读旧值）
    对角元虚部保留（不做置 0）；uplo 三角外不写；ldc padding 列不触碰
```

- **workspace 布局表**（FP32 元素，对齐沿用 csyrk 口径）：

| 区 | 大小 | 用途 |
|---|---|---|
| Xr / Xi | ldX×k（ldX = n 对齐分形） | deinterleave 输出平面（int8 提升同址折入） |
| （无和式面） | 0 | s=Xr+Xi 核内面板合成，不占 GM |
| （无积平面） | 0 | 三积在核内累加，直接 epilogue |

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| 单 launch 融合 | 三积 + epilogue 一次发射 | 相比 deint+3×GEMM+combine 多段链省 2+ 次发射与中间积落盘（待 msprof 回填） |
| 面板一载三用 | Xr/Xi 面板一次加载完成三积；和式核内合成（单边） | GM 读量低于独立三积（待实测） |
| 三角裁剪 | 仅 uplo 三角 tile 参与计算与写回 | 计算量与带宽 ~0.5×（相对全阵） |
| 已排除路线 | 3×GEMM 拼接（§2.1.7 禁止）；Hermitian X·X^H 分解（数学错误）；依赖未合入 CsyrkEx（§2.2 禁止） | — |

判据双条件（§3.3）：单 case NPU ≤2.5×gpu_ms 且全 case 加速比均值 ≥0.5×；gpu_baseline 200 条 max=8.6362ms / avg=1.7164ms（cublasCsyrk3mEx kernel-only）。达标性以设备 msprof 实测为准（设计阶段不判决）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Ascend 950PR | ✔（本任务目标，arch35） |
| Ascend 950DT | （按仓 README 惯例随 950 系声明，不单独验证） |
| Atlas A2 / A3 系列 | （本任务不涉及 arch22） |

## 算子约束限制

| 约束项 | 内容（§2.5 逐条） |
|---|---|
| 校验层级 | uplo/trans/n/k/lda/ldc/Atype/Ctype host 侧校验；int8 按交错 (int8,int8) 解释；不扫描 Device 数据；对称性由调用方保证 |
| 非连续支持 | 仅 lda/ldc 带填充布局；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 天然支持（n/k 运行时入参驱动 tiling） |
| 原地语义 | C 原地覆写、仅 uplo 三角读写、不返回视图；C 可写区与 A 可读区不重叠由调用方保证 |
| 确定性 | 3M 路径与 K 归约顺序由形状/片上容量决定，不按用例编号或种子分派；不保证逐位一致 |
| 空 Tensor | n=0 no-op；k=0 秩-k 贡献 0（A 可不必有效） |
| 0 维 Tensor | 不涉及（标量 Host/Device 双位置支持） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = CBLAS **非 3M** cblas_csyrk（int8 先精确提升 float 后调用；FP64 域可用 cblas_zsyrk 收窄）；实/虚部分别逐元素 `|actual−golden| ≤ atol + rtol·|golden|`；FLOAT32 分量 rtol=atol=2^-13、matched_ratio ≥0.99 且 max_abs_error ≤ 1e-2 或 32 ULP 时用例通过；3M 舍入差异由阈值吸收并在自测报告说明 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel 双条件：msprof op Task Duration（warmup5+10 均值）单 case ≤2.5×gpu_ms（≥0.4×）**且**全部性能 case 加速比算术平均 ≥0.5×，两者同时满足；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py）；int8 路径性能自采、沿用 COMPLEX64 gpu_ms 基线 | §3.3 / §3.3.1 / §3.3.2 / §3.3.6 |

## 测试概览

- 用例规模：精度全量 1000（含 **TC_3M_* 4 条专项**：交替/极值填充实/虚部抵消、int8 极值、**对角虚部保留**）+ 性能子集 200（dtype × 规模档 small/medium/large 分层）；csv 实测 1201 行。
- 覆盖：uplo{U,L} × trans{N,T,C} 全组合（**OP_C 结果须与 OP_T 完全一致**——对称语义交叉验证点）；Atype{C64,INT8}（int8 用 RANDOM_INT8 + VALUE_INT8_EXTREME）；n 全档至 4096、k 至 2048；lda/ldc 紧凑+padding；负向（非法 uplo/trans/负值/ld 下界/不支持 Atype·Ctype/空指针）。
- 特殊场景：对角虚部非零输入（**有效数据原样保留**）；alpha=(0,0) 且 A 为 NaN（不引用 A）；beta=(0,0) 且 C 为 NaN（不读旧 C）；k=0 且 beta≠(1,0)；Inf/NaN 传播。
- 性能口径：`msprof op --gtest_filter=*TC_PF*` 采集 OpBasicInfo.csv Task Duration(us)（正式）；verify_performance.py 解析 `[PERF] avg_us`（交叉）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/syrk/csyrk3m_ex/arch35/` ATK 工程）。

## 兼容性分析

- **公共头新增声明**：`include/cann_ops_blas.h` 仅新增 `aclblasCsyrk3mEx` 一条声明，不改任何既有声明（§2.3）；新增符号对既有链接面无破坏（新增导出函数）。
- 新文件 `blas/syrk/arch35/csyrk3m_ex_{host,kernel}.cpp`（+kernel.h/tiling 按需），非 Ex csyrk 多件套零改动；与 aclblasCsyrkEx（另一未合入新接口）及姊妹任务 aclblasCherk3mEx **无代码依赖**（各自独立文件，不共享未合入符号，§2.2 末段）。
- `test/syrk/` 新建 csyrk3m_ex/ 子目录并注册 CMake 目标；既有 csyrk/ssyrk 测试不动。
- README：`blas/syrk/README.md` 补 aclblasCsyrk3mEx 章节（含 Atype/Ctype 组合、OP_C≡T 对称语义、对角虚部保留、950PR 支持情况）。
- 无既有调用方（新接口），无行为变更波及。

