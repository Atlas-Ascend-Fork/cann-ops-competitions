# aclblasCherk3mEx 算子设计文档

> 目标芯片：昇腾 950PR（arch35 / dav-c220，SOC=ascend950）｜CANN 9.1.0 及以上
> 数据类型：A ∈ {complex64, complex-int8((int8,int8) 交错 2B)}；C = complex64；alpha/beta = FLOAT32 实标量
> 合入位置：`ops-blas/blas/herk/arch35/`；测试：`ops-blas/test/herk/cherk3m_ex/arch35/`
> 任务书：`aclblasCherk3mEx_Atlas950PR_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasCherk3mEx 算子开发（昇腾 950PR，arch35）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/herk/arch35/`（§1/§5）；测试代码 → `test/herk/cherk3m_ex/arch35/`（新建子目录并注册构建目标，§2.2）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 接口现状：`include/cann_ops_blas.h` **尚无 `aclblasCherk3mEx` 声明（grep 实证 0 命中），本任务新增该声明**——公共头新增（仅增不改），波及面见"兼容性分析"。

## 背景介绍

### 算子功能定位

复数 Hermitian 矩阵秩-k 更新（3M 扩展版）：用 Hermitian 专用 3M 分解把复数秩-k 乘积 X·X^H 化为 3 次实数矩阵乘，对 C 的 uplo 三角原地更新：

```
C = alpha * op(A) * op(A)^H + beta * C      （C 为 n×n Hermitian，仅 uplo 三角读写）
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCherk3mEx | cuBLAS `cublasCherk3mEx`｜Netlib `cherk`（数学语义） | complex64 / complex-int8 → complex64 的 Hermitian 秩-k 更新（3M 加速） |

### 业界对标

| 项 | cuBLAS cublasCherk3mEx | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/trans/n/k/alpha/A/Atype/lda/beta/C/Ctype/ldc | 逐参数一致（§2.3） |
| trans 取值 | 文档列 N/T/C，但 OP_T 公式未定义 | **按定义域收窄：仅 N/C；OP_T → INVALID_ENUM**（有意差异，§2.1.2） |
| Atype 组合 | (CUDA_C_8I, CUDA_C_64F) 与 (CUDA_C_64F, CUDA_C_64F) | 同：ACL_INT8/ACL_COMPLEX64 → ACL_COMPLEX64；其它 → NOT_SUPPORTED（§2.1.6） |
| 标量 | alpha/beta 实数 | 同（FLOAT32；Host/Device 指针位置均支持，§2.5） |
| Hermitian 语义 | 仅 uplo 三角读写；对角虚部为 0 | 同（计算后强制置 0，§2.1.3） |
| 舍入 | 3M 与直接复数乘舍入不同 | 同——与 aclblasCherkEx/cublasCherk 不保证逐位一致，按 §3.2 阈值判定 |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/herk/arch35/` 下**新建 `cherk3m_ex_host.cpp` / `cherk3m_ex_kernel.cpp`**（或按仓内规范扩展现有 cherk host/kernel，§2.2）；与同目录非 Ex `aclblasCherk` 四件套并存但**互不依赖**（aclblasCherkEx 亦为未合入新接口，禁止调用，§2.2 末段）。
- 公共头：`include/cann_ops_blas.h` **新增** `aclblasCherk3mEx` 声明（§2.3 签名，含 Atype/Ctype 两个 aclDataType 参数）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/herk/arch35/cherk_{host,kernel}.cpp`（非 Ex，已合入） | ★ 主模板 | host 校验/标量双位置读取（ReadCherkScalarsFromDevice）/workspace 规划；kernel 的 uplo 三角全套（StoreUploColumn/StoreNonDiagBlock/RestoreLowerDiagSegment）、ZeroDiagonal、ApplyBeta、deinterleave（LoadComplexABlock→StoreRealBlocks） | GEMM 段改为**单 launch 三积融合**（其 RunCherkGemmPhase 是多次 launch，不满足 §2.1.7）；对角虚部/缩放复用 |
| `blas/gemm/arch35/cgemm3m_host.cpp` + `cgemm3m_rounding_kernel.cpp` | ◆ 参考 | 3M 分解代数、舍入处理、融合路径思路 | 其为 X·Y 非 Hermitian；本任务按 X·X^H 分解（P1 取正号、P2=(Xr+Xi)(Xr−Xi)^T），逐字对齐 §2.1.7 |
| `gemm/arch35` 共享 cube 层 + `common/helper/syrk_host_utils.h` | 共享基建 | GemmTilingData/多核切分口径、syrk 族 host 工具 | 融合 kernel tiling 沿用同口径并加三累加器预算 |
| `test/herk/cherk/` 工程 | ◇ 参考 | ATK 测试组织、golden/param 组织方式 | 新建 cherk3m_ex/ 子目录 |

### 接口规格

需在 `include/cann_ops_blas.h` 新增声明（§2.3，与 cuBLAS `cublasCherk3mEx` 逐参对应）：

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| uplo | C 的存储三角模式 | attr | int | {UPPER, LOWER}；其它 → `INVALID_ENUM` | 标量 |
| trans | op(A) 运算类型 | attr | int | {OP_N, OP_C}；**OP_T 或其它 → INVALID_ENUM** | 标量 |
| n | 方阵 C 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |
| k | 秩-k 更新的 k 维 | attr | int | ≥0；负 → `INVALID_VALUE`；0 时秩-k 贡献为 0 | 标量 |
| alpha | 实数标量乘数 | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| A | 输入矩阵（列主序，Device） | tensor | COMPLEX64 / COMPLEX8I | N: n×k；C: k×n；n>0∧k>0 时 nullptr → `INVALID_VALUE` | N: lda×k；非 N: lda×k（物理 k×n 逻辑） |
| Atype | A 的 dtype 枚举 | attr | int | {ACL_COMPLEX64, ACL_INT8}；不在支持集或组合不支持 → `NOT_SUPPORTED` | 标量 |
| lda | A 前导维 | attr | int | N: ≥max(1,n)；非 N: ≥max(1,k)；低于下界 → `INVALID_VALUE` | 标量 |
| beta | 实数标量乘数 | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| C | n×n 输出（列主序，原地覆写，仅 uplo 三角读写） | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE` | ldc×n（逻辑 n×n） |
| Ctype | C 的 dtype 枚举 | attr | int | {ACL_COMPLEX64}；其它 → `NOT_SUPPORTED` | 标量 |
| ldc | C 前导维 | attr | int | ≥max(1,n)；低于 → `INVALID_VALUE` | 标量 |

> Atype=ACL_INT8 语义：每个复元素 (int8 real, int8 imag) 交错 2 字节（对应 cuBLAS CUDA_C_8I），参与计算前**精确提升**为 float（§2.4/§3.2）。

### 数学公式

```
C = alpha * op(A) * op(A)^H + beta * C
op(A) = A（trans=N，A 为 n×k）；op(A) = A^H（trans=C，A 以 k×n 存储，共轭转置为 n×k）
```

Hermitian 专用 3M 分解（§2.1.7，X = op(A) = Xr + i·Xi，int8 先精确提升）：

```
乘积为 X·X^H（Br = Xr^T，Bi = −Xi^T）：
P0 = Xr·Xr^T，P1 = Xi·Xi^T，P2 = (Xr+Xi)·(Xr−Xi)^T
Re(X·X^H) = P0 + P1
Im(X·X^H) = P2 − P0 + P1
```

⚠️ 与通用 cgemm3m 的 X·Y 分解不同：P1 为**取正号形式**（等价于通用式 P1′=Xi·Bi=−Xi·Xi^T 的负号吸收）、P2 第二操作数为 (Xr−Xi)。错误套用 X·X^T 对称分解会把虚部算成 Xi·Xr^T+Xr·Xi^T（正确值 Xi·Xr^T−Xr·Xi^T），属实现错误（§2.1.7 点名）。三次实数矩阵乘**必须在同一次 kernel launch 内完成**。

对角元素：C(i,i) = alpha·Σ(|X|²) 为实数，计算后虚部强制置 0（§2.1.3）。

# 需求分析

## 需求描述

用 Ascend C 在昇腾 950PR（arch35）上实现新增句柄式接口 `aclblasCherk3mEx`：单 launch 融合 kernel 完成 Hermitian 专用 3M 三乘积（P0/P1/P2 同轮累加）与 uplo 三角 epilogue（beta 缩放 + 对角虚部置 0 + 交织写回），支持 Atype∈{complex64, complex-int8}、trans∈{N,C}、实数标量双指针位置；精度按 FP32 分量达生态混合容差（golden = 非 3M cblas_cherk），性能达 GPU 标杆双条件（单 case ≥0.4× 且全 case 均值 ≥0.5×）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | Hermitian 3M（§2.1.7 公式，P1 正号 + P2=(Xr+Xi)(Xr−Xi)^T）；uplo/对角虚部/beta 缩放处理可整体复用非 Ex cherk kernel 的 combine 段 |
| 2 | 接口原型 | §2.3 签名（13 参数，含 Atype/Ctype）；**公共头新增声明**（仅增不改） |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4（含 NOT_SUPPORTED 类） |
| 4 | 支持数据类型 | A: COMPLEX64 或 INT8 复（2B 交错）；C: COMPLEX64；alpha/beta: FLOAT32 |
| 5 | 目标芯片与 arch 映射 | 950PR → arch35；构建 `--soc=ascend950` |
| 6 | 精度要求 | §3.2：FP32 分量 rtol=atol=2^-13、ratio ≥0.99、max_abs ≤ 1e-2 或 32 ULP；golden=cblas_cherk（int8 精确提升后调用；FP64 用 cblas_zherk）；3M 舍入差异由阈值吸收 |
| 7 | 性能预期 | §3.3：**双条件**——单 case NPU ≤2.5×gpu_ms（≥0.4×）且全 200 case 加速比均值 ≥0.5×；msprof op Task Duration（warmup5+10）为正式判据；gpu_baseline 实测 max=9.93ms / avg=1.92ms |
| 8 | 代码架构选型 + 可行性 | deinterleave（含 int8 提升）+ **单 launch 融合 kernel（三累加器同轮乘加 + uplo epilogue）**两段；已排除死路：三次独立 GEMM 拼接（§2.1.7 明令禁止）、调用未合入 aclblasCherkEx（§2.2 禁止）、X·X^T 对称分解（数学错误）；可行性旁证 = 非 Ex cherk 的 deinterleave/combine 段与 gemm cube 共享层均已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/（1201 行 csv = 精度 1000 含 TC_3M_* 4 条 + 性能 200；gpu_baseline 201 行） |
| 10 | 特值与退化 | n=0 no-op；k=0 或 alpha=0：beta=1 → quick return 不动 C；beta≠1 → 仅 uplo 三角 beta 缩放（不引用 A，A 可 null）；beta=0 不读 C 旧值 |
| 11 | 错误码口径 | 非法 uplo/trans（含 OP_T）→ INVALID_ENUM；维度/ld/指针 → INVALID_VALUE；Atype/Ctype 不支持 → NOT_SUPPORTED |
| 12 | workspace | Xr/Xi 实平面（deinterleave 暂存，int8 提升同址折入）；融合 kernel 累加器驻核内；沿用 cherk workspace 规划口径 |

# 详细设计

## 算子分析

### 数学公式

逐元素（i, j 属于 uplo 三角，含对角）：

```
C[i,j] = alpha * Σ_{t=0..k-1} X[i,t]·conj(X[j,t]) + beta * C[i,j]
X = op(A)（trans=N: X=A 的 n×k；trans=C: X=A^H，A 以 k×n 存储，共轭折入平面加载）
```

3M 展开（FP32 实平面，Xr/Xi 为 X 的实/虚部平面；int8 先精确提升）：

```
P0[i,j] = Σ Xr[i,t]·Xr[j,t]          （对称积）
P1[i,j] = Σ Xi[i,t]·Xi[j,t]          （对称积，取正号——Im 中两处 +P1 的来源）
P2[i,j] = Σ (Xr+Xi)[i,t]·(Xr−Xi)[j,t]
Re(P)[i,j] = P0[i,j] + P1[i,j]
Im(P)[i,j] = P2[i,j] − P0[i,j] + P1[i,j]
C[i,j]   = (alpha·Re(P)[i,j]) + i·(alpha·Im(P)[i,j]) + beta·C[i,j]
C[i,i]   虚部强制置 0（数学上 Re(P)[i,i] = Σ|X[i,t]|² ≥ 0，虚部为 0）
```

uplo 三角外元素不读不写（由 C=C^H 隐含）；ldc padding 列不动。特值：alpha=0 或 k=0 → P 项跳过，仅 `C[i,j] = beta·C[i,j]`（uplo 三角内）；beta=0 → C 旧值不读取（含 NaN 安全）；beta=1 且（alpha=0 或 k=0）→ 完全 quick return 不动 C。

### 支持数据类型

| 对象 | 接口 dtype | kernel 内部 |
|---|---|---|
| A（Atype=ACL_COMPLEX64） | complex64（8B 交织） | 拆为 Xr/Xi FP32 平面 |
| A（Atype=ACL_INT8） | (int8,int8) 交错 2B（CUDA_C_8I 对应） | deinterleave 时**精确提升**为 FP32 平面 |
| C（Ctype=ACL_COMPLEX64） | complex64 | 交织读写 |
| alpha/beta | FLOAT32 实标量（Host/Device 指针位置均可） | tiling 传值 |

支持组合（§2.1.6）：(INT8→C64) 与 (C64→C64)；其它 → NOT_SUPPORTED。

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥ 0 | 0 → SUCCESS no-op，不引用 A/C、不下发 kernel |
| k | ≥ 0 | 0 → 秩-k 贡献 0：beta=1 quick return；beta≠1 仅 uplo 三角 beta 缩放 |
| lda | N: ≥max(1,n)；非 N: ≥max(1,k) | 低于下界 → INVALID_VALUE |
| ldc | ≥max(1,n) | 低于下界 → INVALID_VALUE |
| 覆盖档位（§3.5） | n ∈ {0..4096 全档}，k ∈ {0..2048 全档}；uplo × trans(N/C) 全组合 | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

两段式（3M 三乘积在**同一次 kernel launch** 内，§2.1.7 硬约束）：

```
aclblasCherk3mEx (cherk3m_ex_host.cpp)
 └─ 校验（含 Atype/Ctype 支持集）/ quick return / 标量双位置读取
 └─ Phase 0  cherk3m_ex_deinterleave_kernel（AIV）:
        complex64 或 int8 复 → Xr / Xi 实平面（列主 packed；int8 精确提升折入 cast）
 └─ Phase 1  cherk3m_ex_kernel（单次 launch，融合三积 + epilogue）:
        逐 C-tile × K 分块:  加载 Xr/Xi 面板 → 三积同轮累加
            acc0 += Xr·Xr^T ; acc1 += Xi·Xi^T ; acc2 += (Xr+Xi)·(Xr−Xi)^T
        Re=acc0+acc1, Im=acc2−acc0+acc1 → C = alpha·P + beta·C（uplo 三角）
        对角虚部置 0；ldc padding 哨兵；beta=0 不读 C
特值旁路: alpha=0/k=0 → 仅 uplo 三角 beta 缩放 kernel（beta=1 直接 SUCCESS 不发 kernel）
```

kernel 变体划分表：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| cherk3m_ex_deinterleave_kernel | 常规路径（Atype=C64/INT8 两形态） | 交织复面（或 int8 复面）→ Xr/Xi FP32 平面 |
| cherk3m_ex_kernel | 常规路径 | **单 launch 内三次实乘同轮累加 + 3M combine + beta + 置对角虚部 + uplo 交织写回** |
| beta 缩放旁路 kernel | alpha=0 或 k=0 且 beta≠1 | 仅 uplo 三角 C=beta·C（beta=0 → 写 0） |

数据流：GM 交织 A →(deint)→ GM Xr/Xi 平面 →(融合 kernel：面板进核)→ 核内三累加器 → 3M/alpha/beta/对角/uplo → GM C。trans=C 的共轭转置折入面板加载（Xi 取负行/列映射，与 §2.1.7 的 Bi=−Xi^T 一致）。

#### host 侧设计

- **分核策略**：融合 kernel 按 C 的 uplo 三角元素量切核（三角块行/列带划分，满核原则 + 尾核余块；对角带单独处理，参照 cherk combine 的 UploCount/对角段划分）；deinterleave 按 A 平面元素均分。
- **数据分块和内存优化策略**：C-tile 尺寸按三累加器并存预算取保守档（arch35 分形基准下调；面板 (Xr/Xi/和差) 复用一次加载完成三积——和差面 (Xr+Xi)/(Xr−Xi) 在核内由面板直接合成，不再单独落 GM，省 2 次 plane 读写）；K 分块循环累加。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| uplo | 0/1 | Upper/Lower 三角路径（写区/对角段方向） |
| trans | 0/1 | N / C（面板加载共轭映射） |
| atype | 0/1 | C64 / INT8（deinterleave cast 形态） |
| alphaZero / betaZero / betaOne | 0/1 | 特值旁路（不引用 A / 不读 C / quick return） |

- **数据检测**（校验顺序逐条对齐 §2.5 注；结构合法后再 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | uplo ∉ {UPPER, LOWER} | INVALID_ENUM |
| 3 | trans ∉ {OP_N, OP_C}（**含 OP_T**） | INVALID_ENUM |
| 4 | n<0 或 k<0 | INVALID_VALUE |
| 5 | lda 低于下界（按 trans） | INVALID_VALUE |
| 6 | ldc < max(1,n) | INVALID_VALUE |
| 7 | alpha/beta == nullptr | INVALID_VALUE |
| 8 | Ctype ≠ ACL_COMPLEX64 或 Atype ∉ {ACL_COMPLEX64, ACL_INT8} | NOT_SUPPORTED |
| 9 | n>0 时 C == nullptr | INVALID_VALUE |
| 10 | n>0∧k>0 时 A == nullptr | INVALID_VALUE |
| 11 | 全过后：n==0 → SUCCESS（不下发 kernel） | — |
| 12 | k==0 或 alpha==0 → beta==1 直接 SUCCESS；否则仅 beta 缩放旁路（不引用 A） | — |

  标量读取：alpha/beta 指针位置自动识别（复用 cherk 的 ReadCherkScalarsFromDevice 模式，Host/Device 双位置）。

#### kernel 侧设计

- **Init 阶段**：blockIdx → 本核 C-tile 集（uplo 三角映射）；绑定 Xr/Xi 平面与 C 的 GM tensor；UB 布局三累加器 + 面板 buffer + epilogue 暂存。
- **Process 阶段**（融合 kernel 主循环）：

```
for (jt, it) ∈ 本核三角 tile 集:
    acc0/acc1/acc2 := 0
    for t ∈ K 分块:
        载 Xr[it,t], Xr[jt,t], Xi[it,t], Xi[jt,t] 面板（trans=C 时 Xi 行/列共轭映射）
        acc0 += Xr[it]·Xr[jt]^T          ← 实乘 1
        acc1 += Xi[it]·Xi[jt]^T          ← 实乘 2
        s_i := Xr[it]+Xi[it]; d_j := Xr[jt]−Xi[jt]（核内合成，不落 GM）
        acc2 += s_i·d_j^T                ← 实乘 3（同一 launch 内）
    Re := acc0+acc1 ; Im := acc2−acc0+acc1
    C[tile] = alpha·(Re+i·Im) + beta·C[tile]（beta≠0 时读旧值；beta=0 直写）
    对角元 (it==it) 虚部置 0；uplo 三角外不写；ldc padding 列不触碰
```

- **workspace 布局表**（FP32 元素，对齐沿用 cherk 口径）：

| 区 | 大小 | 用途 |
|---|---|---|
| Xr / Xi | ldX×k（ldX = n 对齐分形） | deinterleave 输出平面（int8 提升同址折入） |
| （无和差面） | 0 | (Xr±Xi) 核内面板合成，不占 GM |
| （无积平面） | 0 | 三积在核内累加，直接 epilogue |

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| 单 launch 融合 | 三积 + epilogue 一次发射 | 相比 deint+3×GEMM+combine 多段链省 2+ 次发射与中间积落盘（待 msprof 回填） |
| 面板一载三用 | Xr/Xi 面板一次加载完成三积；和差核内合成 | GM 读量 ~0.5×（相对独立三积各读各的） |
| 三角裁剪 | 仅 uplo 三角 tile 参与计算与写回 | 计算量与带宽 ~0.5×（相对全阵） |
| 已排除路线 | 3×GEMM 拼接（§2.1.7 禁止）；X·X^T 对称分解（数学错误）；依赖未合入 CherkEx（§2.2 禁止） | — |

判据双条件（§3.3）：单 case NPU ≤2.5×gpu_ms 且全 case 加速比均值 ≥0.5×；gpu_baseline 200 条 max=9.93ms / avg=1.92ms（cublasCherk3mEx kernel-only）。达标性以设备 msprof 实测为准（设计阶段不判决）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Ascend 950PR | ✔（本任务目标，arch35） |
| Ascend 950DT | （按仓 README 惯例随 950 系声明，不单独验证） |
| Atlas A2 / A3 系列 | （本任务不涉及 arch22） |

## 算子约束限制

| 约束项 | 内容（§2.5 逐条） |
|---|---|
| 校验层级 | uplo/trans/n/k/lda/ldc/Atype/Ctype host 侧校验；int8 按交错 (int8,int8) 解释；不扫描 Device 数据；对称性/Hermitian 性由调用方保证 |
| 非连续支持 | 仅 lda/ldc 带填充布局；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 天然支持（n/k 运行时入参驱动 tiling） |
| 原地语义 | C 原地覆写、仅 uplo 三角读写、不返回视图；C 可写区与 A 可读区不重叠由调用方保证 |
| 确定性 | 3M 路径与 K 归约顺序由形状/片上容量决定，不按用例编号或种子分派；不保证逐位一致 |
| 空 Tensor | n=0 no-op；k=0 秩-k 贡献 0（A 可不必有效） |
| 0 维 Tensor | 不涉及（标量指针 Host/Device 双位置支持） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = CBLAS **非 3M** cblas_cherk（int8 先精确提升 float 后调用；FP64 域可用 cblas_zherk 收窄）；实/虚部分别逐元素 `|actual−golden| ≤ atol + rtol·|golden|`；FLOAT32 分量 rtol=atol=2^-13、matched_ratio ≥0.99 且 max_abs_error ≤ 1e-2 或 32 ULP 时用例通过；3M 舍入差异由阈值吸收并在自测报告说明 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel 双条件：msprof op Task Duration（warmup5+10 均值）单 case ≤2.5×gpu_ms（≥0.4×）**且**全部性能 case 加速比算术平均 ≥0.5×，两者同时满足；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py）；int8 路径性能自采、沿用 COMPLEX64 gpu_ms 基线 | §3.3 / §3.3.1 / §3.3.2 / §3.3.6 |

## 测试概览

- 用例规模：精度全量 1000（含 **TC_3M_* 4 条专项**：交替/极值填充虚部抵消、int8 极值、对角虚部置 0）+ 性能子集 200（dtype × 规模档 small/medium/large 分层）；csv 实测 1201 行。
- 覆盖：uplo{U,L} × trans{N,C} 全组合；Atype{C64,INT8}（int8 用 RANDOM_INT8 + VALUE_INT8_EXTREME）；n 全档至 4096、k 至 2048；lda/ldc 紧凑+padding；负向（非法 uplo/trans 含 OP_T/负值/ld 下界/不支持 Atype·Ctype/空指针）。
- 特殊场景：对角虚部非零输入（输出置 0）；alpha=0 且 A 为 NaN（不引用 A）；beta=0 且 C 为 NaN（不读旧 C）；k=0 且 beta≠1；Inf/NaN 传播。
- 性能口径：`msprof op --gtest_filter=*TC_PF*` 采集 OpBasicInfo.csv Task Duration(us)（正式）；verify_performance.py 解析 `[PERF] avg_us`（交叉）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/herk/cherk3m_ex/arch35/` ATK 工程）。

## 兼容性分析

- **公共头新增声明**：`include/cann_ops_blas.h` 仅新增 `aclblasCherk3mEx` 一条声明，不改任何既有声明（§2.3）；新增符号对既有链接面无破坏（新增导出函数）。
- 新文件 `blas/herk/arch35/cherk3m_ex_{host,kernel}.cpp`（+kernel.h/tiling 按需），非 Ex cherk 四件套零改动；与 aclblasCherkEx（另一未合入新接口）无依赖关系（§2.2 末段）。
- `test/herk/` 新建 cherk3m_ex/ 子目录并注册 CMake 目标；既有 cherk 测试不动。
- README：`blas/herk/README.md` 补 aclblasCherk3mEx 章节（含 Atype/Ctype 组合、OP_T 定义域收窄、Atlas 950PR 支持情况）。
- 无既有调用方（新接口），无行为变更波及。

