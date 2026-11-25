# aclblasCgemm3m 算子设计文档

> 算子名称：`aclblasCgemm3m`（Ascend A2/A3，arch22）
> 任务来源：cann-ops-competitions 社区任务 community-task-2026 · aclblasCgemm3m_AscendA2A3
> 对标基线：cuBLAS `cublasCgemm3m`；golden：Netlib `cblas_cgemm`

---

## 1 需求背景（required）

### 1.1 需求来源

社区任务 community-task-2026 要求在昇腾 NPU（Atlas A2/A3 系列）上使用 Ascend C 编程语言、基于 ops-blas 开源仓工程模式，开发单精度复数（complex64）三乘法通用矩阵乘算子 `aclblasCgemm3m`，补齐 arch22（Ascend 910B3/A3）实现与测试，验收通过后合入昇腾算子开源仓。

### 1.2 背景介绍

复数通用矩阵乘（CGEMM）`C = alpha·op(A)·op(B) + beta·C` 是复数 BLAS-3 的核心算子。直接复数乘法每个输出元素需要 4 次实数乘加（4M 算法）。Gauss 三乘法（3M）将其降为 3 次实数矩阵乘，复数乘加总量由 `4mnk` 降至 `3mnk`（省 25%），同时实数乘加粒度更大、更适合 Cube 单元的流水化执行，是 cuBLAS 提供 `cublasCgemm3m` 专用接口的动机。

ops-blas 仓现状：

- `blas/gemm/arch35/cgemm3m_host.cpp` 已有 arch35（Ascend 950PR）实现，其 3M 路径选择与缩放逻辑可参考，但 SDK/硬件不兼容代码不得直接复制；
- `blas/gemm/arch22/` 已有 `aclblasCgemm`（4M 直接复数乘）实现（`cgemm_host.cpp` / `cgemm_kernel.cpp` / `cgemm_tiling_data.h`），采用「Phase 0 拆分 → Phase 1 实数 GEMM → Phase 2 复数组装」三段式流水线；
- `include/cann_ops_blas.h`（L584）已声明 `aclblasCgemm3m`，本任务无需新增声明。

### 1.3 现状分析

对仓内已有复数 GEMM 基础设施的梳理（本设计复用的部分加粗标注）：

| 能力 | 位置 | 本任务复用方式 |
| --- | --- | --- |
| 复数→实部/虚部平面拆分 kernel（AIV，GatherMask 单遍列扫描，`negateImag` 折叠共轭符号） | `blas/common/helper/complex_blas3_arch22.h` + `cgemm_kernel.cpp` | **扩展为三/四输出 split3m**（增发组合平面） |
| 实数 FP32 GEMM 委托入口（arch22 tiling、K 拆分、多核切分） | `aclblasSgemmEx` + sub-handle scratch 机制 | **直接委托**，Phase 1 由 4 次调用改为 3 次 |
| 实数平面累加 kernel（`dst = scale·src` / `dst += scale·src`） | `cgemm_accum_kernel_do` / `CgemmAccumTilingData` | **直接复用**（K 分块累加） |
| 复数 epilogue 组装 kernel（`C = alpha·(re+i·im) + beta·C`） | `cgemm_combine_kernel_do` / `CgemmCombineTilingData` | **改造为系数化三乘积组装** |
| 参数校验、workspace 规划、quick return、标量读取 | `cgemm_host.cpp` | **沿用同族校验口径**，workspace 增加组合/乘积平面 |
| arch35 3M 路径分派（direct / panel / onchip / unit-batches） | `blas/gemm/arch35/cgemm3m_strided_batched_host.cpp` | 仅作路径选择思想参考 |

---

## 2 需求分析（required）

### 2.1 需求描述

实现 `aclblasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`，与 cuBLAS `cublasCgemm3m` 逐参数对应、数学语义完全对齐，计算：

```text
C = alpha * op(A) * op(B) + beta * C
```

其中 alpha/beta 为 complex64 标量（Host 内存），A/B/C 为 complex64 列主序矩阵（Device 内存，按 lda/ldb/ldc 带填充布局访问）；`op(X)` 取 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置）。

复数乘积按 Gauss 3M 算法分解为 **3 次实数矩阵乘**，与 4M 直接复数乘的 FP32 舍入过程不同，不保证与 `aclblasCgemm` 逐位一致，按 §3.5 阈值判定。

### 2.2 需求拆解

1. **接口与语义对齐**：与 `aclblasCgemm` 同参数序列；`m=0`/`n=0` 合法 quick return；`k=0` 或 `alpha=0` 时只做 `C = beta·C` 且不引用 A/B；`beta=0` 不读 C 旧值；仅写 C 的逻辑 m×n 区域，ldc padding 列不被修改。
2. **参数校验**：host 侧按任务书 §2.4 逐参数校验（handle → transa/transb → m/n/k → lda/ldb/ldc → alpha/beta → 指针），状态码与 `cann_ops_blas_common.h` 一致；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，与 cuBLAS 有意差异）。
3. **3M 算法落地**：两条代数等价的分解路径（非融合 / 融合）按形状与片上容量在 host 侧分派；共轭转置（`OP_C`）的虚部符号 s_a/s_b 在 Phase 0 拆分时折叠进虚部平面与组合平面（`Ai' = s_a·Ai`、`Bi' = s_b·Bi`），实数 GEMM 不感知共轭、组装系数与 trans 组合无关（仅随路径变化）。
4. **三段式流水线复用**：Phase 0 拆分扩展为「实部 + 虚部 + 组合平面」多输出；Phase 1 委托 `aclblasSgemmEx` 执行 3 次实数 GEMM（K 分块规避 ops-blas#1077）；Phase 2 系数化组装。
5. **workspace 规划**：实数平面 + 组合平面 + 乘积平面 + 委托 scratch 的统一布局，支持分区复用以控制峰值（单用例 host 侧 ≤ 512MB 预算，覆盖 4096³ 用例）。
6. **性能达标**：kernel 平均耗时 ≤ GPU（cuBLAS `cublasCgemm3m`）基线的 1/0.8 倍；复数乘减量 25% 是主要收益来源。
7. **精度达标**：按生态算子开源精度标准，COMPLEX64 按 FLOAT32 分量判定（atol=rtol=2^-13，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP）。
8. **测试工程**：`test/gemm/cgemm3m/arch22/` GTest + CSV 驱动（1000 精度 + 200 性能），golden 由 cblas 生成，在 `test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标。

---

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

**主公式**：

```text
C = alpha * op(A) * op(B) + beta * C      (列主序 complex64)
```

记 `A = Ar + i·Ai`、`B = Br + i·Bi`（Ar/Ai/Br/Bi 为实部/虚部矩阵），共轭转置带来的虚部符号：

```text
s_a = +1 (transa = N/T) , s_a = -1 (transa = C)
s_b = +1 (transb = N/T) , s_b = -1 (transb = C)

op(A) = Ar^T + i * s_a * Ai^T ,  op(B) = Br^T + i * s_b * Bi^T
```

直接复数乘（4M，`aclblasCgemm` 采用）展开为：

```text
op(A)*op(B) = [ Ar·Br − (s_a·s_b)·Ai·Bi ]  +  i·[ s_a·Ai·Br + s_b·Ar·Bi ]
               └──────── Re ────────┘         └──────── Im ────────┘
```

**符号折叠约定（本实现）**：3M 的两个组合平面 `Ar ± s_a·Ai`、`Br ± s_b·Bi` 本身就要用到带符号的虚部，因此本实现把 s_a/s_b 在 **Phase 0 拆分**时一次性折进虚部平面（复用 split 内核既有的 `negateImag` 开关），后续一律以折叠后的平面记号和运算：

```text
Ai' = s_a·Ai ,  Bi' = s_b·Bi ,  sumA = Ar + Ai' ,  sumB = Br + Bi'
```

此后 op(A)·op(B) 等价于 `(Ar + i·Ai')·(Br + i·Bi')`，共轭不再显式出现，公式与任务书 §2.1 给出的 3M 分解形式逐字一致。（4M 版选择在 Phase 1 的乘积标量里放符号，是因为它不需要组合平面；两条策略代数等价，差异只在舍入顺序。）

**3M 分解路径一（非融合，默认路径）**：

```text
P0 = Ar · Br
P1 = Ai' · Bi'
P2 = sumA · sumB = (Ar + Ai') · (Br + Bi')

Re = P0 − P1
Im = P2 − P0 − P1 = P2 − Re
```

**3M 分解路径二（融合路径，含纠正重算）**：

```text
P0' = sumA · Br = (Ar + Ai') · Br
P1' = Ar · (Bi' − Br)
P2' = Ai' · sumB = Ai' · (Br + Bi')

Re = P0' − P2'
Im = P0' + P1'
```

两条路径代数等价，但中间项的 FP32 舍入不同（非融合路径的 `P2` 与 `P0/P1` 独立舍入；融合路径的加减在组装前级联），**不能把一种路径的误差分析套用到另一种**，验收按 §3.5 阈值判定。

**epilogue**：

```text
Cr = alpha_r·Re − alpha_i·Im + beta_r·Cr_old − beta_i·Ci_old
Ci = alpha_i·Re + alpha_r·Im + beta_r·Ci_old + beta_i·Cr_old
```

#### 3.1.2 支持数据类型

| 类别 | 类型 | 说明 |
| --- | --- | --- |
| A / B / C | COMPLEX64（实部/虚部各 FLOAT32） | `aclblasComplex`，定义以 `include/cann_ops_blas_common.h` 为准 |
| alpha / beta | COMPLEX64 标量（Host 内存指针） | 与 cuBLAS 默认 host pointer mode 一致 |
| 内部计算 | FLOAT32 | 拆分平面、实数 GEMM、组装均按 FP32 执行 |

#### 3.1.3 支持形状

| 项 | 规则 |
| --- | --- |
| op(A) / op(B) / C 形状 | m×k / k×n / m×n |
| 物理存储 | A 为 lda×k（transa=N）或 lda×m（T/C）；B 为 ldb×n（transb=N）或 ldb×k（T/C）；C 为 ldc×n |
| 前导维度下界 | transa=N 时 lda ≥ max(1,m)，否则 lda ≥ max(1,k)；transb=N 时 ldb ≥ max(1,k)，否则 ldb ≥ max(1,n)；ldc ≥ max(1,m) |
| 维度取值 | m/n/k ≥ 0；m=n=0 合法 no-op；k=0 只做 `C=beta·C` |
| 数据排布 | 连续列主序 + 前导维度填充；不支持超出前导维度语义的任意非连续布局；C 与 A/B 存储不得重叠 |
| broadcast / 动态 shape | 不支持 broadcast；m/n/k/lda/ldb/ldc 为运行期入参，host 侧动态 tiling |

### 3.2 算子实现

#### 3.2.1 实现方案总览

复用 arch22 `aclblasCgemm` 的三段式流水线骨架，将 Phase 1 的「4 次实数 GEMM」改为「3 次实数 GEMM + 组合平面」，Phase 2 改为系数化组装。核心等价变换：**共轭符号 s_a/s_b 在 Phase 0 拆分时一次性折进虚部平面（`Ai' = s_a·Ai`、`Bi' = s_b·Bi`），实数 GEMM 不感知共轭、其 trans 标志与 4M 版完全一致**，因此 Phase 1 可直接委托 `aclblasSgemmEx`；组装系数只随路径变化（非融合/融合两套），与 trans 组合无关，kernel 侧无 trans 分支。

```text
                     ┌───────────────────────── Phase 0 (AIV) ─────────────────────────┐
  A (complex)  ──▶  split3m: Ar , Ai'=s_a·Ai , sumA = Ar + Ai'      （一遍 GatherMask 三输出）
  B (complex)  ──▶  split3m: Br , Bi'=s_b·Bi , sumB = Br + Bi' [, subB = Bi' − Br]

                     ┌───────────────────────── Phase 1 (AIC) ─────────────────────────┐
  3 次实数 GEMM（委托 aclblasSgemmEx，sub-handle scratch，K 分块 ≤1024 累加）：
    非融合: prod0 = Ar·Br , prod1 = Ai'·Bi' , prod2 = sumA·sumB
    融合  : prod0 = sumA·Br , prod1 = Ar·subB , prod2 = Ai'·sumB

                     ┌───────────────────────── Phase 2 (AIV) ─────────────────────────┐
  combine3m: 按路径系数 (c0r,c1r,c2r)/(c0i,c1i,c2i) 组装 Re/Im，再
             C = alpha·(Re + i·Im) + beta·C_old（beta=0 不读 C 旧值）
```

文件规划（遵循任务书 §2.2 目录约定）：

| 文件 | 内容 |
| --- | --- |
| `blas/gemm/arch22/cgemm3m_host.cpp` | `aclblasCgemm3m` 入口：参数校验、quick return、workspace 规划与分区复用、路径分派、三段下发 |
| `blas/gemm/arch22/cgemm3m_tiling_data.h` | `Cgemm3mSplitTilingData`（多输出 split）、`Cgemm3mCombineTilingData`（系数化组装） |
| `blas/gemm/arch22/cgemm3m_kernel.h` / `.cpp` | split3m kernel、combine3m kernel 的 launcher 声明与实现 |
| `blas/gemm/arch22/gemm_host.cpp` | 增加 `aclblasCgemm3m` 入口转调（或按仓内规范由 CMake 直接编入 `cgemm3m_host.cpp`） |
| `blas/gemm/arch22/cgemm_kernel.cpp` | 复用 split / accum；3M 组合平面输出在 split 内核内以开关扩展 |
| `test/gemm/cgemm3m/arch22/` | GTest + CSV 驱动测试；`test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标 |
| `blas/gemm/README.md` | 补充 `aclblasCgemm3m` 的 Atlas A2/A3 支持说明 |

> 不放入 `blas/gemm3m/`：该目录是实数 `aclblasSgemm3m`（3 个实数矩阵沿 K 拼接）的实现目录，与复数 `aclblasCgemm3m` 不是同一接口族。

#### 3.2.2 host 侧设计

**(1) 参数校验与 quick return**

校验顺序与 `aclblasCgemm` arch22 实现保持一致（先结构参数、后 quick return、再指针条件校验）：

```text
1. handle == nullptr                → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. transa/transb ∉ {N,T,C}          → ACLBLAS_STATUS_INVALID_ENUM
3. m < 0 || n < 0 || k < 0          → ACLBLAS_STATUS_INVALID_VALUE
4. lda/ldb/ldc 低于 §3.1.3 下界      → ACLBLAS_STATUS_INVALID_VALUE
5. alpha/beta == nullptr            → ACLBLAS_STATUS_INVALID_VALUE
6. m == 0 || n == 0                 → 直接返回 SUCCESS（不引用 A/B/C，不启动 kernel）
7. C == nullptr (m>0 && n>0)        → ACLBLAS_STATUS_INVALID_VALUE
   A/B == nullptr (m>0 && n>0 && k>0 && alpha≠0) → ACLBLAS_STATUS_INVALID_VALUE
```

alpha/beta 为 Host 指针，直接解引用得到 `alphaVal`/`betaVal`（无需 D2H 往返，与 arch35 口径的差异在文档与自测报告中说明）。

**(2) 退化分支**（与 4M 版同构）

| 条件 | 行为 |
| --- | --- |
| `k == 0` 或 `alpha == 0` | `hasTemp = false`：跳过 Phase 0/1，仅执行 Phase 2 的 `C = beta·C`（beta=1 时直接返回；beta=0 时按逻辑 m×n 区清零，不动 padding 行） |
| `hasTemp == false && beta == 1` | 直接返回 SUCCESS，不启动任何 kernel |
| `m == 0 || n == 0` | quick return，A/B/C 允许为 nullptr |

**(3) workspace 规划与分区复用**

布局（单位：float，行 stride 均按 64B 对齐，即 `Align64(rows)`）：

```text
[S1] Ar / prod0      [S2] Ai' / prod1     [S3] Br / prod2      [S4] Bi'
[S5] sumA            [S6] sumB            [S7] subB（仅融合路径）  [S8…] scratch（委托 aclblasSgemmEx）
```

分区复用策略（按 4096³ 峰值 384MB 设计，低于 512MB 预算）：

| 阶段 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 0 写 | Ar | Ai' | Br | Bi' | sumA | sumB | subB |
| Phase 1 后（非融合） | **prod0** | **prod1** | **prod2** | 空闲 | 空闲 | 空闲 | — |
| Phase 1 后（融合） | prod0(sumA 区) | prod1 | prod2 | 空闲 | 空闲 | 空闲 | 空闲 |

- 非融合路径 GEMM 顺序 `G0=Ar·Br → G1=Ai'·Bi' → G2=sumA·sumB`：`prod0` 覆盖已用完的 Ar 区、`prod1` 覆盖 Ai' 区、`prod2` 覆盖 Br 区，峰值 6 个平面；
- 融合路径 GEMM 顺序 `G0=sumA·Br → G1=Ar·subB → G2=Ai'·sumB`：`prod0` 覆盖 sumA 区（G0 后 sumA 空闲）、`prod1` 覆盖 subB 区、`prod2` 覆盖 Br 区，峰值 7 个平面（小形状专用）；
- 所有尺寸累加走 64 位 checked 运算（复用 `CBlas3TryMulU64`/`CBlas3TryAddU64`），溢出时返回 `ACLBLAS_STATUS_ALLOC_FAILED` 而非回绕成小请求；
- scratch 切片交给 sub-handle（`workspace_owner = User`），防止委托的实数 GEMM 通过 `EnsureDefaultWorkspace` 回收仍在使用的平面（沿用 4M 版的 sub-handle 机制）。

**(4) 路径分派（3M 路径选择）**

按形状与片上容量在 host 侧分派，判据为可调常量：

```text
默认：非融合路径（组合平面少 1 个，workspace 峰值低，K 分块累加机制与 4M 版完全一致）
融合路径条件（示例）：m ≤ 256 && n ≤ 256 && k ≤ 1024
    —— 三次乘积规模小、组合平面 subB 的额外开销占比低，且小形状下融合路径的
       中间舍入级联更短；具体阈值以实测标定，标定记录留在自测报告
```

两条路径的组装系数由 host 计算后随 tiling 下发，kernel 无路径分支（见 3.2.3）。

**(5) Phase 1 委托与 K 分块**

- 每个乘积一次 `aclblasSgemmEx(sub, ta, tb, m, n, kCount, 1.0, 左操作数, ldA, 右操作数, ldB, 0.0, prod, ldRe)`，`ta/tb` 将 `OP_C` 归一为 `OP_T`（共轭符号已在平面与系数中折叠）；
- K 按 `CGEMM3M_K_CHUNK = 1024` 分块（沿用 4M 版对 ops-blas#1077 的规避：M/N 均低于整网格阈值时 K>1024 会误算），首块直接写平面、后续块经 `cgemm_accum_kernel_do` 以 `dst += 1.0·src` 累加——FP32 加法精确，不损失精度；
- K 在 packed 平面上的步距按 trans 推导：A 侧 `ta=N` 时间距 `ldAr`、`ta=T` 时间距 1；B 侧 `tb=N` 时间距 1、`tb=T` 时间距 `ldBr`（与 4M 版一致）；
- 规避 `aclblasSgemm` 的 `RowsLeftTransMatmul` 路径（`m%16==0 && n%16==0 && k%8==0 && transB==T` 时跨 AIC 核传未对齐偏移）：当 `tb != N && m%16==0 && n%16==0` 时把 `kCount` 从 8 的倍数上挪开（沿用 4M 版策略）。

**(6) 分核策略**

| 阶段 | 分核方式 |
| --- | --- |
| Phase 0 split3m | `blocks = min(aivCoreNum, cols)`，按列 round-robin（一列一次 GatherMask 连续产出三/四个输出） |
| Phase 1 实数 GEMM | 委托 `aclblasSgemmEx` 内部 AIC 多核切分（m/n 方向 tile 均分，余块前核多分） |
| Phase 2 combine3m | `blocks = min(aivCoreNum, n)`，按输出列切分 |

#### 3.2.3 kernel 侧设计

**(1) Phase 0：split3m（AIV）**

在 4M 版 split 内核上扩展输出开关，一遍 GatherMask（列主序下逻辑列为连续的 `2·rows` 个 float）同时产出：

```text
Ar  = Xr = Re(A)                          # 实部
Ai' = s_a·Xi                              # negateImag 折叠共轭符号 s_a
sumA = Ar + Ai'                           # emitSum = 1
（B 侧同构：Br、Bi' = s_b·Xi、sumB = Br + Bi'；subB = Bi' − Br 仅融合路径下发）
```

- tiling（`Cgemm3mSplitTilingData`）：`rows, cols, lda, packedLd, negateImag, emitSum, emitSub, sumLd, subLd`；
- 列内 `SPLIT_CHUNK = 4096` 个复数一批（256B repeat 的整数倍，repeat 计数 < 255），尾批走 DataCopyPad；
- 组合输出与 Xr/Xi 同一遍产生，AIV 额外开销仅为每元素一次加/减，不增加 GM 读流量。

**(2) Phase 1：实数 GEMM（AIC，委托）**

复用 `aclblasSgemmEx` 的 arch22 tiling 与多核逻辑，本算子不新增 Cube kernel。三个乘积按路径下发：

| 路径 | G0 | G1 | G2 |
| --- | --- | --- | --- |
| 非融合 | `Ar · Br → prod0` | `Ai' · Bi' → prod1` | `sumA · sumB → prod2` |
| 融合 | `sumA · Br → prod0` | `Ar · subB → prod1` | `Ai' · sumB → prod2` |

K 分块累加沿用 `cgemm_accum_kernel_do`（`CgemmAccumTilingData{totalFloats, scale, overwrite}`）。

**(3) Phase 2：combine3m（AIV，系数化组装）**

tiling（`Cgemm3mCombineTilingData`）：

```cpp
struct Cgemm3mCombineTilingData {
    uint32_t m, n;          // 逻辑输出形状
    uint32_t ldc;           // C 前导维度（complex 元素）
    uint32_t ldProd;        // 乘积平面行 stride（float）
    float c0r, c1r, c2r;    // Re  = c0r·P0 + c1r·P1 + c2r·P2
    float c0i, c1i, c2i;    // Im  = c0i·P0 + c1i·P1 + c2i·P2
    float alphaRe, alphaIm, betaRe, betaIm;
    uint32_t hasTemp;       // alpha==0 或 k==0：无有效乘积平面，只做 C = beta·C
    uint32_t hasBeta;       // beta==0：不发起 C 旧值读回
};
```

host 侧按路径填系数（s_a/s_b 已在 Phase 0 平面中折叠，系数与 trans 组合无关）：

| 路径 | (c0r, c1r, c2r) | (c0i, c1i, c2i) | 验证 |
| --- | --- | --- | --- |
| 非融合 | (1, −1, 0) | (−1, 0, 1) | Re = P0−P1；Im = P2−P0−P1 |
| 融合 | (1, 0, −1) | (1, 1, 0) | Re = P0'−P2'；Im = P0'+P1' |

kernel 流程（按输出列分核，每列连续 `m` 个元素）：

```text
for j in [blockIdx, n) step blockNum:
    for i in [0, m):                       # 列主序连续
        Re = c0r·P0[i,j] + c1r·P1[i,j] + c2r·P2[i,j]
        Im = c0i·P0[i,j] + c1i·P1[i,j] + c2i·P2[i,j]
        if hasTemp:
            Tr = alphaRe·Re − alphaIm·Im
            Ti = alphaIm·Re + alphaRe·Im
        else: Tr = 0, Ti = 0               # alpha==0 或 k==0
        if hasBeta:
            Tr += betaRe·C[i,j].re − betaIm·C[i,j].im
            Ti += betaRe·C[i,j].im + betaIm·C[i,j].re
        C[i,j] = (Tr, Ti)
```

- `hasTemp == 0` 时三个乘积平面指针传 C 本体（不产生无效 GM 读），对应 `C = beta·C` 分支；
- `hasBeta == 0` 时不发起 C 旧值读回，满足「beta=0 不读 C 旧值」（测试以 NaN 旧值验证）；
- 仅写逻辑 m×n 区，`ldc` padding 行/列不被修改；
- UB 内一次处理整列（`m` 个复数），列内为连续 DataCopy，尾块按实际行数走 DataCopyPad。

**(4) 异步执行与错误处理**

kernel 在 handle 绑定的 stream 上异步下发；每次 launch 后经 `aclrtGetLastError(ACL_RT_THREAD_LEVEL)` 捕获异步错误，非 `ACL_SUCCESS` 返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3，arch22） | √ |
| Atlas A3 系列（arch22，对应 soc 取值） | √ |

- CANN 版本：9.1.0；
- 编译入口：`bash build.sh --ops=cgemm3m --soc=ascend910b3`（A3 使用对应 `--soc` 取值）；
- 自验证覆盖一种款型即可，精度建议 910B3 或 A3，性能建议 910B3。

### 3.4 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数校验层级 | host/API 入口按 §3.2.2 逐参数校验并返回状态码；不对 Device 张量做元素级扫描 |
| 非连续 Tensor | 不支持；按 lda/ldb/ldc 列主序带填充布局访问，不使用 AutoContiguous/ViewCopy，无相关临时空间 |
| broadcast | 不支持，A/B/C 为独立操作数 |
| dynamic shape | 不要求；m/n/k/lda/ldb/ldc 与 transa/transb 运行期入参，host 侧动态 tiling 与 launch |
| 原地语义 | C 原地覆写；调用方保证 C 可写区与 A/B 可读区不重叠 |
| 确定性 | 不要求；3M 路径与 K 归约顺序由形状与片上容量决定，同一输入两次执行不保证逐位一致 |
| 空 Tensor | `m=0`/`n=0` 为合法 no-op；`k=0`/`alpha=0` 只做 `C=beta·C`（beta=0 置零、beta=1 不变），允许 A/B 为 nullptr，m>0 且 n>0 时 C 必须非空 |
| 0 维 Tensor | 不涉及；矩阵经指针 + m/n/k + lda/ldb/ldc 描述，标量以标量指针传入 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；接口返回不代表 Device 计算完成，Host 读回前须同步 |
| 数值差异 | 3M 与 4M/直接复数乘舍入过程不同，与 `cblas_cgemm` golden 不保证逐位一致，按 §3.5 阈值判定 |

### 3.5 可维可测分析

#### 3.5.1 精度标准

按生态算子开源精度标准，COMPLEX64 按 FLOAT32 分量执行（实部、虚部分别比对）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT32（complex64 分量） | 2^-13 (1.2e-4) | 2^-13 (1.2e-4) | 0.99 | 1e-2 或 32×ULP |

- 逐元素判定：`|actual − golden| ≤ atol + rtol·|golden|` 计入 matched；用例同时满足 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`（或 32×ULP）判通过；
- golden：CPU 侧 `cblas_cgemm`（Netlib BLAS），float64 累加后收窄到 complex64 落盘（或直接用 float32 参考实现，差异在自测报告说明）；
- Inf/NaN 传播用例按「同为 NaN 即匹配」判定。

#### 3.5.2 性能标准

| 指标 | 口径 | 判据 |
| --- | --- | --- |
| kernel 耗时（正式判据） | `msprof op` 采集 NPU kernel task duration（us），warmup 5 次后连续 10 次取平均 | 单 case 及平均倍率 ≥ GPU 基线 `gpu_ms` 的 0.8 倍 |
| 端到端耗时（交叉验证） | GTest `aclrtEvent` 夹住连续 10 次下发取 `avg_us`，打印 `[PERF] case=... avg_us=...` | 仅交叉验证，不作为正式判据 |

- GPU 基线：cuBLAS `cublasCgemm3m` 相同参数采集（host pointer mode，warmup 5 + 有效 10 次平均）；
- 性能子集 200 条，按规模档分层（small ≤64KiB / medium ≤2MiB / large >2MiB），显式保留 alpha/beta 列（约 12% 非默认标量）。

#### 3.5.3 验收标准

| 维度 | 标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- | --- |
| 精度 | 与 `cblas_cgemm` 按 §3.5.1 阈值比对通过 | 实部/虚部分别比对，matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 | 任务书 §3.2 / 生态精度标准 |
| 性能 | kernel 耗时 ≤ GPU 基线 1.25 倍 | `msprof op` kernel-to-kernel 对比 ≥ 0.8 | 任务书 §3.3 |
| 内存 | 单用例 host 侧 ≤ 512MB（设计选择，非硬件限制） | 4096³ 峰值约 384MB | 任务书 §3.4 / 测试工程口径 |
| 功能 | quick return / beta=0 不读 C / k=0·alpha=0 缩放 / OP_C 共轭 / padding 写保护 / 负向状态码 | 1000 条精度 + 200 条性能 + 负向用例 | 任务书 §3.5 |

#### 3.5.4 自验设计

- 用例：CSV 驱动（`case_name,description,transa,transb,m,n,k,alpha_real,alpha_imag,a_fill,lda,b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,mere_threshold,mare_multiplier,random_seed`），GTest 加载后调 `aclblasCgemm3m`；
- 覆盖：trans 3×3 全组合；尺寸 0/1/2/3/5/7/16…4096（含 4095³/4096³）；alpha/beta 特殊值（0/±1/纯虚/±1e4）+ 随机值；lda/ldb/ldc 紧凑与 +1/+8/+64 padding；均匀/正态/全零/交替/Inf/NaN 填充；负向（非法枚举、负维度、非法前导维、条件空指针）；
- 特殊场景：`OP_C` 与 `OP_T` 同形状结果应不同；ldc padding 哨兵写保护；alpha=0 且 A/B 为 NaN 验证不引用 A/B；beta=0 且 C 为 NaN 验证不读旧值；
- 复现步骤由 `test/gemm/cgemm3m/arch22/` 的 readme 说明。

### 3.6 兼容性分析

- **新增能力，不涉及存量兼容**：arch22 新增 `aclblasCgemm3m` 入口，不改动 `aclblasCgemm`/`aclblasSgemm` 等既有接口的语义与签名；
- **与 arch35 实现的关系**：arch35 已有 `aclblasCgemm3m`（`blas/gemm/arch35/cgemm3m_host.cpp`，经 `ASC_DEVKIT_GE_9_1` 版本宏隔离）；arch22 版本独立实现，两条产品线通过编译宏/目录隔离，互不影响；
- **共享代码的影响面**：扩展 split 内核时以新增开关字段（默认值保持 4M 行为）方式演进，`chemm/csymm/cherk/csyrk/cher2k` 等既有复数 BLAS-3 算子行为不变；
- **接口稳定性**：声明已存在于 `include/cann_ops_blas.h`（L584），本任务不新增/不修改公开 API。

---

## 4 交付件与合入

| 序号 | 交付件 | 要求 |
| --- | --- | --- |
| 1 | 算子设计文档 | 按官方模板，PR 合入 cann-competitions 仓库 |
| 2 | 自测用例及测试代码 | 精度全量 1000 条 + 性能子集 200 条，含 readme 复现步骤 |
| 3 | 自测报告 | 用例参数、精度对比（实部/虚部）、性能数据（`avg_us` 与 `msprof op` 双口径）及截图 |
| 4 | 待验收代码地址 | 个人仓链接/分支/算子目录，邀请 Ascend-CANN 为开发者；提供算子 readme |

合入目录：实现 `blas/gemm/arch22/`，测试 `test/gemm/cgemm3m/arch22/`，遵循 ops-blas 贡献指南（`CONTRIBUTING.md`）。
