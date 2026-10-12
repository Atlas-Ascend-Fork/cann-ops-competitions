# aclblasCtrmm 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 参考样例：cann-ops-competitions PR #1318（aclsparseDenseToSparse 设计文档）。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasCtrmm 算子开发(A2A3)」。在 `ops-blas` 仓为 Atlas
A2/A3（`arch22`，性能设备 910B3，覆盖 A3 系列）基于 Ascend C/CATLASS 全新实现
单精度复数（COMPLEX64）三角矩阵乘算子 `aclblasCtrmm`：执行
`C = alpha · op(A) · B`（side=LEFT）或 `C = alpha · B · op(A)`（side=RIGHT），
对齐 cuBLAS `cublasCtrmm` 的功能与参数语义，离席计算（结果写独立输出 C）。

代码交付至 `https://gitcode.com/cann/ops-blas`：实现放 `blas/trmm/arch22/`，
测试放 `test/trmm/ctrmm/arch22/`；接口声明入 `include/cann_ops_blas.h`。
设计文档交付至 `https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### 算子定位：带三角结构约束的 Level-3 复数矩阵乘

`ctrmm` 是 BLAS Level-3 三角矩阵乘：A 为三角矩阵（仅引用 uplo 指定三角、
diag=UNIT 时对角不读），B/C 为一般矩阵。与求解类（trsm）不同，**无回代、无对角
非零要求**（NON_UNIT 下对角可为零，全零 A 合法），是纯乘加计算——计算特征更接近
GEMM，三角性体现在「一半的乘法可省略 + 非引用三角的访存裁剪」。

规模特征（性能 case）：256² → 4096²，case 5（4096×4096, LEFT, UPPER, C）有效
复数 MAC ≈ M³/2 = 34.4 G，折合实数 206–275 GFLOP，预算 19931 us，**要求有效
实数算力 ≈ 10–14 TFLOPS**——这是 Cube（矩阵计算单元）量级的需求，向量核无法
覆盖，设计必须以 Cube 为主力。

### 三个核心难点

1. **复数 × Cube**：arch22 Cube 原生面向实数（FP32）MAC。复数乘加须分解为实数
   GEMM 组合（4 实 GEMM 法或 3M 优化法），并解决交织存储（interleaved complex64）
   与 Cube 需要的规整操作数布局之间的转换——布局转换（pack/拆平面）的访存开销
   必须计入性能预算。
2. **24 组枚举组合的统一**：side(2) × uplo(2) × trans(3) × diag(2)。trans=T/C
   改变 A 的消费方向（C 还需共轭）；side=RIGHT 把三角矩阵换到右侧、A 阶数由 n
   决定；diag=UNIT 对角置 1 不读。组合爆炸若在各分支分别处理，kernel 数量与测试
   矩阵都会失控——需要规范化降维（§3.3.2）。
3. **原地语义 B≡C**：cuBLAS trmm 允许 B 与 C 同地址（等价原地用法）。设计必须
   保证该场景读写无竞争（§3.3.6）。

### 任务书关键约束的解读

1. **性能预算口径**：达标耗时 = GPU 标杆 ÷ 0.8。GPU 五个 case 实测
   83.7 us – 15.94 ms，预算 104.6 us – 19931 us；小 case（256²）预算被
   launch/pack 固定开销支配，大 case（4096²）被 Cube 有效算力支配——设计需要
   「小规模低开销路径 + 大规模 Cube 路径」双轨。
2. **alpha 为 Host 指针**：调用时同步读值固化进 TilingData；alpha=0 时 A/B 不被
   引用（可为 nullptr）、C 全置零——独立的 memset 快路径。
3. **零维 no-op**：m=0 或 n=0 返回 SUCCESS 不计算（A/B/C 允许空指针）。
4. **离席 + 受限原地**：结果写 C；仅允许 B≡C 原地，其余重叠不允许。
5. **确定性不要求**：浮点累加顺序不保证逐位一致，混合容差判定。

### 同族算子与工程模式参照

| 参照 | 关系 | 可借鉴点 |
| --- | --- | --- |
| ops-blas 仓既有 blas 算子（arch22） | 同仓工程规范 | kernel 直调模式、handle/stream、CSV/GTest 框架、README |
| CATLASS 线性代数模板库 | 任务书点名的编程基础 | arch22 分块 GEMM/Cube 调度模板，本设计在其 blocked GEMM 骨架上叠加 trmm 调度 |
| ops-sparse `sparse/spmm`（arch22 标准 Ascend C） | 同架构仓内先例 | TPipe/op.Init/Process 工程模式、AIV 分核 |

### 功能缺口

ops-blas 仓当前无任何 trmm 实现（实数/复数、任何 arch 均无）。缺口为整体性
缺失：接口、host、kernel、tiling、构建、测试、README 全套新建。社区同期存在
Ctrmm-950（A5）与 Strmm（A2/A3）任务，代码目录互不重叠，接口声明公共头按先
合入方 rebase 对齐。

---

# 需求分析（required）

## 需求描述

在 `blas/trmm/arch22/` 以 Ascend C/CATLASS 实现 `aclblasCtrmm` 句柄式 BLAS
接口，能力达到：COMPLEX64 × side{LEFT,RIGHT} × uplo{UPPER,LOWER} ×
trans{N,T,C} × diag{UNIT,NON_UNIT} 全 24 组组合 × m,n ∈ [0, 大规模] ×
lda/ldb/ldc 满足前导维约束（含 padding）× alpha 任意复值（含 0）× B≡C 原地；
精度满足生态算子开源精度标准 COMPLEX64 档（混合容差 rtol=atol=2⁻¹³，
matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP，对 C 全矩阵实/虚部分别
比对）；任务书 §3.3 五个性能 case 达标（NPU 平均单次 kernel 耗时 ≤ GPU/0.8）。

## 需求拆解

| # | 需求 | 来源 | 性质 |
| --- | --- | --- | --- |
| R1 | arch22 全新实现：Host 校验/分发 + Ascend C/CATLASS Kernel + tiling + 构建接入 + README | §1、§2.2、§5 | 新增（主体） |
| R2 | 24 组枚举组合语义正确：side 决定 A 阶数（LEFT 为 m×m、RIGHT 为 n×n）、uplo 裁剪读取范围、T/C 转置（C 共轭）、UNIT 对角不读 | §2.1、§7.3 | 新增 |
| R3 | 复数乘加的 Cube 分解：交织 complex64 → 实/虚平面（SoA）布局，4 实 GEMM 组合（3M 优化为可选项） | §2.1、§2.2 | 新增（技术主体） |
| R4 | alpha Host 指针读值；alpha=0 时 memset C、不引用 A/B（允许空指针） | §2.4、§2.5 | 新增 |
| R5 | 离席写 C；B≡C 原地正确；其余重叠不支持（调用方保证） | §2.5 | 需保证 |
| R6 | 参数校验全覆盖：handle 空、alpha 空、非法枚举、m/n<0、ld 非法、矩阵引用非空时 A/B/C 空指针，返回码对齐 §2.4 | §2.4 | 新增 |
| R7 | m=0/n=0 no-op；Inf/NaN 传播对齐 cuBLAS（乘加不检查） | §2.1、§3.5 | 需保证 |
| R8 | 五个性能 case 达标（预算 = GPU/0.8：104.6/235.2/803.8/3241/19931 us），大 case 需 Cube 有效算力 10–14 TFLOPS | §3.3 | 约束 |
| R9 | 小规模低开销路径：小 m·n 走向量核直算，规避 pack+Cube 固定开销 | §3.3 case1 | 设计选择 |
| R10 | 接入 ops-blas CSV/GTest 测试框架，覆盖随任务提供的 1200 条用例并按需补充 | §3.5 | 新增 |

---

# 详细设计（required）

## 算子分析

### 数学公式

```
side=LEFT :  C = alpha · op(A) · B      （A 为 m×m）
side=RIGHT:  C = alpha · B · op(A)      （A 为 n×n）
op(A) = A（N）/ Aᵀ（T）/ Aᴴ（C，复共轭转置）
diag=UNIT 时 op(A) 对角视为 1 且不读取；NON_UNIT 时正常读取（允许为零）
B、C 为 m×n 一般矩阵，列主序；仅引用 uplo 指定三角的 A
```

分量式（LEFT, op(A) 下三角为例）：`C(i,j) = alpha · Σ_{k≥i} op(A)(i,k) · B(k,j)`。
三角性使有效乘加量约为同规模 GEMM 的一半：`MAC ≈ M·m·n/2`（M 为 A 阶数）。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| A/B/C 元素 | COMPLEX64（8 字节交织） | 唯一支持档 |
| alpha | COMPLEX64，Host 指针 | 调用时读值固化进 TilingData |
| Cube 操作数 | FP32 实/虚平面（SoA） | pack/staging 阶段拆分，epilogue 重组 |
| 累加器 | FP32 | 实部/虚部独立平面累加，精度与 golden（Netlib FP32 复数）对齐 |

### 支持形状

| 参数 | 范围 | 说明 |
| --- | --- | --- |
| m, n | ≥ 0 | m=0 或 n=0 no-op；非方阵（fat/thin）全覆盖 |
| A | side=LEFT: m×m；RIGHT: n×n | lda ≥ max(1, m) 或 max(1, n)；仅引用三角 |
| B, C | m×n | ldb/ldc ≥ max(1, m)；支持 padding |
| 组合约束 | 24 组枚举全支持 | 无 NOT_SUPPORTED 分支 |

## 算子实现

### 总体结构

```text
include/cann_ops_blas.h              公开接口声明（随本算子新增）
blas/trmm/arch22/
  ctrmm_host.cpp                     参数校验、规范化、workspace、tiling、launch
  ctrmm_tiling_data.h                TilingData 结构（host/kernel 共享）
  ctrmm_kernel.h                     kernel 启动函数声明
  ctrmm_kernel.cpp                   Ascend C kernel：pack / GEMM 主循环 / 小矩阵向量核
  ctrmm_gemm.h(.cpp)                 分块复数 GEMM（CATLASS 模板封装：实平面 Cube MAC）
blas/trmm/README.md                  支持矩阵与接口说明（产品支持表标注 A2/A3）
test/trmm/ctrmm/arch22/              CSV 驱动 GTest
```

### Host 侧设计

#### Host 调用流程

```text
┌───────────────────────── Host: aclblasCtrmm ─────────────────────────┐
│                                                                       │
│ ① 参数校验（顺序对齐 §2.4 异常行为表）                                 │
│    handle == nullptr            → ACLBLAS_STATUS_HANDLE_IS_NULLPTR   │
│    side/uplo/trans/diag 非法枚举 → ACLBLAS_STATUS_INVALID_VALUE       │
│    m < 0 / n < 0                → ACLBLAS_STATUS_INVALID_VALUE       │
│    alpha == nullptr             → ACLBLAS_STATUS_INVALID_VALUE       │
│    lda/ldb/ldc 不满足前导维约束  → ACLBLAS_STATUS_INVALID_VALUE       │
│    m==0 或 n==0                 → SUCCESS（no-op，返回；允许 A/B/C 空）│
│    alpha == 0                   → memset C（stream 上 aclrtMemsetAsync│
│                                     或置零 kernel），SUCCESS 返回，    │
│                                     不引用 A/B（允许空）               │
│    矩阵引用非空（m,n>0）且       → ACLBLAS_STATUS_INVALID_VALUE       │
│    A/B/C == nullptr                                                │
│                                                                       │
│ ② 读 alpha 值（Host 指针，同步读）→ 固化进 TilingData                  │
│                                                                       │
│ ③ 规范化（§3.3.2）：side/uplo/trans → {A' 布局, 扫描方向, conj,        │
│    B/C 转置视图标志}；M = side==LEFT ? m : n                           │
│                                                                       │
│ ④ 路径选择                                                            │
│    m·n ≤ VEC_THRESH（初值 128×128，可调）→ 单 kernel 向量直算路径       │
│    否则 → pack + Cube 分块 GEMM 路径                                  │
│    side=RIGHT 且 B≡C → 追加 C 临时缓冲（§3.3.6）                      │
│                                                                       │
│ ⑤ workspace 规划（aclrtMallocAsync，stream 生命周期）                  │
│    A' 面板（SoA 实/虚平面）：≈ M²/2 complex × 2 平面                  │
│    RIGHT&B≡C 的 C 临时缓冲：m·n complex                               │
│                                                                       │
│ ⑥ tiling：NB（A 面板块宽，64–256，按 M 与 UB/TCM 定）、NC（B/C 列块）、│
│    Cube核/AIV核 分配、split-K 参数 → TilingData                        │
│                                                                       │
│ ⑦ launch（handle 绑定 stream）：pack kernel → GEMM kernel 群           │
│    （每块列一次 launch 或持久 kernel 单 launch，按规模选择）             │
└───────────────────────────────────────────────────────────────────────┘
```

#### 规范化：24 组合 → 4 个正交维度

**A 的规范化（pack 阶段完成）**：无论 side/uplo/trans，pack 统一产出「规范下三角
（或上三角）+ 列主序面板 + 实/虚分离平面」的 A'，使 GEMM 主循环只面对单一布局：

| 原始组合 | pack 动作 |
| --- | --- |
| 任意 side, uplo, trans=N | 读引用三角，拆实/虚平面，按块列面板写出 |
| trans=T | 读引用三角时行列互换（转置搬运），其余同上 |
| trans=C | T 的基础上虚部取负（共轭） |
| diag=UNIT | 对角位置写 1+0i，不读原对角 |
| diag=NON_UNIT | 对角正常拷贝（允许零，无检查） |

uplo 决定读哪个三角；side 决定 M（A 阶数）与扫描方向（§3.3.4）。

**side 的规范化（转置视图）**：

```
side=RIGHT:  C = alpha·B·op(A)   ⇒   Cᵀ = alpha·op(A)ᵀ·Bᵀ
```

即 RIGHT 归约为 LEFT 形式：A'' = op(A)ᵀ（阶数 n×n，三角性翻转/conj 组合由 pack
一并处理），B'' = Bᵀ（n×m 视图：Cube 以转置标志消费 B 的列主序布局，无需物理
转置），C'' = Cᵀ（m×n 视图：epilogue 以转置映射写回 C，2D DataCopy 交换
stride）。**计算主循环只实现 LEFT 一种**。

### Kernel 侧设计

#### 算法：右视（right-looking）分块三角乘

以规范形式 `C'' = alpha·A''·B''`（A'' 下三角，M 阶；B''/C'' 为 M×N，N 为另一
维）为例。按 NB 分块 A'' 的行列、B''/C'' 的行（P = ⌈M/NB⌉ 块）：

```text
for kb = 0 … P−1:                       # 下三角：升序（上三角：降序镜像）
  ① 大 GEMM（先做，读原始 B''[kb, :]）:
     C''[rows ≥ (kb+1)·NB, :] += Rect_kb · B''[kb, :]
        Rect_kb = A''[(kb+1)·NB:, kb]   # 对角块以下的稠密矩形 (M−(kb+1)·NB)×NB
  ② 小三角 GEMM（后做，读仍是原始的 B''[kb, :]）:
     C''[kb 块行, :] = T_kb · B''[kb, :]
        T_kb = A''[kb 对角块]（NB×NB 三角，非三角区 pack 时已置零）
```

- 总 GEMM 体量 ≈ M²·N/2 个复数 MAC，三角性收益自动兑现（对角块以上无存储、
  无计算）；
- **B≡C 原地安全性（LEFT/规范形式）**：步骤①只写块行 kb 以下的 C''，不触碰
  B''[kb,:]；步骤②读原始 B''[kb,:]、整块算完（TCM 累加）后一次性写块行 kb。
  kb 递增推进，后续步骤只读块行 > kb 的 B''（未修改）✓；
- 4096 规模 P = 16–64 块，每块一次大 GEMM + 一次小 GEMM，kernel 群规模可控。

#### 复数 → 实数 4-GEMM 分解（Cube 消费形态）

A''/B'' 拆为实/虚平面（SoA）：`A = Ar + i·Ai`，`B = Br + i·Bi`：

```
Cr += Ar·Br − Ai·Bi          （实部平面）
Ci += Ar·Bi + Ai·Br          （虚部平面）
```

- pack 产出 Ar/Ai 两个 FP32 平面；B'' 的 tile 在 staging 阶段（GM→UB→TCM 搬运
  时）以向量抽取（奇偶分离）拆为 Br/Bi 平面；
- 四个实 GEMM 共享同一累加器平面（Cr/Ci 各一），Cube 调度按
  `ArBr / AiBi → Cr` 与 `ArBi / AiBr → Ci` 两组交替发射，TCM 复用率最大化；
- 3M 法（Karatsuba，3 实 GEMM + 平面加减）作为可选优化开关：省 25% Cube MAC、
  增加平面加减与 staging 复杂度，默认关闭，实测再定；
- epilogue：`C = alpha·(Cr + i·Ci)` 按
  `(αr·cr − αi·ci) + i·(αr·ci + αi·cr)` 重组为交织 complex64 写 GM
  （ldc 步长；side=RIGHT 时按转置映射写）。

#### Kernel 流程图

```text
┌───────────────── Device ──────────────────────────────────────────┐
│                                                                    │
│  K1 pack（AIV 满核，仅引用三角）                                    │
│   读 A（三角裁剪/转置/共轭/UNIT 置 1）                              │
│   → SoA 面板 Ar|Ai（块列布局，GM workspace）                        │
│                                                                    │
│  K2 GEMM 群（Cube + AIV 搬运，每 kb 块列）                          │
│   ┌────────────┐   ┌────────────┐   ┌────────────┐                │
│   │ staging    │──→│ 4×实 GEMM  │──→│ epilogue   │                │
│   │ B tile 拆  │   │ Cube MAC   │   │ alpha 缩放 │                │
│   │ Br|Bi 平面 │   │ Cr/Ci 累加 │   │ 交织写 C   │                │
│   └────────────┘   └────────────┘   └────────────┘                │
│   循环体 = ①大 GEMM（Rect_kb）→ ②小 GEMM（T_kb，整块缓冲后写出）     │
│   上三角方向镜像（kb 降序，Rect 取对角块以上）                        │
│                                                                    │
│  K0 小矩阵向量核（m·n ≤ VEC_THRESH 时替代 K1+K2）                   │
│   行均分到 AIV：C(i,:) = alpha·Σ_k op(A)(i,k)·B(k,:)               │
│   三角掩码 + 复数 FMA 全在 UB，无 pack、无 Cube                     │
└────────────────────────────────────────────────────────────────────┘
```

#### 性能设计与预算核算

| case | m×n | 组合 | 有效复 MAC | 预算(us) | 需有效实算力* |
| --- | --- | --- | --- | --- | --- |
| 1 | 256² | LEFT,UPPER,N,NON_UNIT | 8.4 M | 104.6 | ~0.6 TF（含固定开销约束） |
| 2 | 512² | LEFT,LOWER,C,NON_UNIT | 67 M | 235.2 | ~2.3 TF |
| 3 | 1024² | LEFT,UPPER,T,UNIT | 537 M | 803.8 | ~5.3 TF |
| 4 | 2048² | RIGHT,LOWER,N,NON_UNIT | 4.3 G | 3241 | ~10.6 TF |
| 5 | 4096² | LEFT,UPPER,C,NON_UNIT | 34.4 G | 19931 | ~10.3–13.8 TF |

\* 按 6–8 实 flop/复 MAC 折算。910B3 FP32 Cube 理论算力数十 TFLOPS 量级，
需求 10–14 TF 有效利用率约 20–40%，处于工程可达区间，但 case 5 裕量有限，
列为首要性能风险（见风险表）。

配套措施：

1. **访存预算**（case 5）：pack 读 A 三角 67 MB + 写面板 67 MB；GEMM 读面板
   67 MB×2（实虚）已含、读 B（拆平面 staging）≈ 134 MB、写 C 268 MB + RIGHT
   原地场景 temp 268 MB——合计 < 1 GB，HBM 带宽下 ~2–3 ms，非瓶颈但需 L2 友好
   的面板调度（同 kb 的四个实 GEMM 共享 B tile）；
2. **panel 常驻与双缓冲**：A'' 面板按块列驻 TCM，B tile GM→UB→TCM 三级流水
   double buffer，掩盖搬运延迟；
3. **split-K**：大 M 小 N 的 fat 场景（精度用例 TC_WS）按 K 维分裂多 Cube 核
   并行，GM 规约合并；
4. **kernel 群组织**：M ≤ 1024 用持久 kernel 单 launch（块内循环），大 M 每块列
   一次 launch——小 case 固定开销最小化（case 1 预算 104.6 us 中 launch+pack
   占比高）；
5. **归因先行**：msprof `--task-time=on` 逐 kernel 归因后再优化（吸取参考 PR
   经验），首批观测项：Cube 占空比、staging 带宽、epilogue 写出效率。

### 支持硬件

| 芯片 | 支持情况 |
| --- | --- |
| Atlas 800I/T A2（910B3 / 910B4） | 支持（arch22），CANN 9.1.0+ |
| Atlas A3 系列（910_93 等） | 支持（arch22） |

自验证以 910B3 为主（性能）；验收覆盖 A2/A3 两个系列（任务书 §3.1、§7.4）。

### 算子约束限制

1. 仅支持 COMPLEX64；仅支持列主序（lda/ldb/ldc 语义内），不支持额外非连续
   Tensor 布局（§2.5）。
2. 原地仅允许 B≡C；C 与 A/B 的其他重叠形式不支持（调用方保证）。
3. side=RIGHT 且 B≡C 时内部使用 C 临时缓冲（m·n complex），峰值内存额外
   +128 MB（4096² 最坏），在自测报告内存章节说明（任务书 §3.4 不设门槛）。
4. alpha=0 路径不引用 A/B（允许空指针），但 C 必须可写且非空（m,n>0 时）。
5. m=0 或 n=0 时 A/B/C 均可为空指针。
6. 无广播、无 dynamic shape 要求（m/n 运行时入参，host tiling 支持）。
7. Inf/NaN 逐元素传播，无检查分支（对齐 cuBLAS）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差（COMPLEX64 档）：逐元素 `|actual−golden| ≤ atol + rtol·|golden|`，atol=rtol=2⁻¹³；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP；golden 由 Netlib BLAS ctrmm 生成，对 C 全矩阵（m×n，实/虚部分别）比对 | 任务书 §3.2、opbase 混合容差标准 |
| 性能标准 | 910B3 上 5 个性能 case 平均单次 kernel 耗时 ≤ GPU 标杆 ÷ 0.8（msprof `OpBasicInfo.csv` Task Duration 均值），预算：104.6 / 235.2 / 803.8 / 3241 / 19931 us | 任务书 §3.3、test_cases README |
| 内存标准 | 不涉及（任务书 §3.4）；workspace 峰值（A' 面板 ≈ M²/2 complex × 2 平面 + RIGHT 原地 temp）在自测报告中说明 | 任务书 §3.4 |

## 测试策略

对接随任务提供的测试资产（`test_cases/`：`trmm_test.csv` 1200 条 +
`gpu_baseline.csv` 200 条 + `gen_csv.py`/`verify_accuracy.py`）：

| 层次 | 内容 |
| --- | --- |
| 框架 | ops-blas 仓 CSV 驱动 C++ GTest；CSV 枚举列简写/全名大小写不敏感解析（README 约定）；`verify_accuracy.py` 一键编译-执行-解析 |
| 精度覆盖 | TC_L0（24 组枚举全组合 × 8×8）/ TC_SQ（尺寸扫描 1→2048，含奇数 3/5/7/65、2 幂±1、非对齐）/ TC_AB（alpha 特殊值：(0,0) 置零、(1,0)、(-1,0)、(0,1) 纯虚、(1,1)、(0.5,±0.5)、(100,0)）/ TC_WS+TC_TH（fat/thin 非方阵，A 阶数取较小维）/ TC_LD（lda/ldb/ldc +4 padding）/ TC_FL（A 均匀/交替/极端、NON_UNIT 全零（乘法语义合法）、UNIT 全零；B 全零/交替/极端/Inf/NaN）/ TC_CV / TC_ED（零维 no-op、alpha=0 含 A/B 空指针合法、空指针、非法枚举、非法前导维、负维度）/ TC_EX（833 条扩展采样） |
| 补充用例 | 按任务书 §3.5 第 4 条自查补齐：B≡C 原地专项（24 组合抽样 × 含 padding）、大尺寸（>2048）采样、C guard 区哨兵（越界写检测）、重复调用幂等性 |
| 性能 | GTest 跑 TC_PF（200 条，全连续访存紧凑 ld）；msprof op 单 case 收敛采集，`OpBasicInfo.csv` Task Duration 均值对比 `gpu_baseline.csv`（阈值 gpu_ms/0.8）；前 5 条 `pf_doc_*` 与任务书 §3.3 逐参数对齐 |
| 报告 | task_submission 目录八件套（步骤说明、精度/性能/内存报告与日志），含逐 case PASS 明细、matched_ratio/max_abs_error 统计、>10 次有效采样均值与截图 |

## 兼容性分析

- **新算子，无存量行为影响**：`blas/trmm/`、`test/trmm/` 为新目录；公共文件仅
  `include/cann_ops_blas.h` 新增声明（纯增量，供其他产品线共用，任务书 §4 交付件
  4 要求）。
- **构建**：arch22 目录经仓内 SOC→arch 映射自动纳入。
- **同期任务**：社区存在 Ctrmm-950（A5）与 Strmm-A2A3 任务，目录/arch 互不
  重叠；公共头接口声明按任务书 §2.3 签名逐字一致，先合入方为基准 rebase。
- **CANN 版本**：9.1.0 及后续配套；Cube 调度优先采用 CATLASS 稳定模板，不使用
  实验性 API。

---

# 风险点

| # | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| 1 | case 5（4096²）Cube 有效算力裕量有限（需 10–14 TF） | 性能不达标 | pack/epilogue 向量化 + 面板 L2 调度压缩非 Cube 时间；4-GEMM 改 3M（省 25% MAC）作为储备；split-K 提升 Cube 占空比；msprof 归因驱动迭代 |
| 2 | 复数拆平面的 staging 开销（B tile 奇偶抽取） | 大 case 带宽与延迟 | staging 与 Cube 双缓冲流水；抽取用向量指令（每 128B 一组奇偶分离）；实测超预算则 pack 阶段直接产出 B 平面（额外 m·n complex 读+2×FP32 写） |
| 3 | 24 组合正确性（side 阶数混淆、C 共轭遗漏、UNIT 不读断言） | 精度误报 | L0 24 组合小例手算对拍；UNIT 场景用「对角置 NaN」用例验证不读（对齐 TC_FL 设计）；规范化表单一出处，kernel 只消费规范形式 |
| 4 | B≡C 原地时序（RIGHT 场景） | 原地用例数据竞争 | LEFT 规范形式下 ①先大 GEMM ②后小 GEMM 的块序天然安全；RIGHT 归约后映射到「C 列块」→ 使用 temp 缓冲 + 结尾回拷，专项用例覆盖 |
| 5 | 小 case（256²，预算 104.6 us）固定开销 | case 1 性能 | 小规模向量核路径免 pack 免 Cube；持久 kernel 单 launch；VEC_THRESH 上板调优 |
| 6 | CATLASS 模板与 arch22 环境的适配差异 | 工程延期 | 上板首日跑模板冒烟（4-GEMM 骨架最小闭环）；不可用时回退为 Ascend C Matmul API 手写分块（设计已按此粒度拆分，回退面可控） |
| 7 | A3 环境到位晚 | 双系列验收进度 | arch22 同源，先 910B3 全量自验，A3 到位后补跑三件套 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions`
   `04_tasks/01_community-task-2026/tasklist/09-aclblasCtrmm-A2A3/<gitcode_id>/docs/design.md`
   （目录编号若与仓内 `09-*` 惯例冲突，按评审意见调整）。
2. **代码**（PR 至 `ops-blas`，任务书 §5 结构）：`include/cann_ops_blas.h` 新增
   `aclblasCtrmm` 声明；`blas/trmm/arch22/` host/tiling/kernel/gemm；
   `blas/trmm/README.md`（产品支持表标注 Atlas A2/A3 系列：支持）。
3. **自测用例与测试代码**：`test/trmm/ctrmm/arch22/`（CSV + GTest，结构对齐主仓
   blas 算子）；复现 README（环境/编译/步骤）。
4. **自测报告**：task_submission 目录八件套（步骤说明、精度/性能/内存报告与
   日志），精度含 golden 对比与截图，性能含 >10 次有效采样均值与 msprof 证据。
