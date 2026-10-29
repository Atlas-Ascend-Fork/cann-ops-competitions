# aclblasCsyrkx 算子设计文档（Atlas 950PR）

## 1. 需求背景（required）

### 1.1 需求来源

昇腾社区 2026 年 10 月社区任务《aclblasCsyrkx 算子开发（950）》。基于 ops-blas 开源仓
（https://gitcode.com/cann/ops-blas ）工程框架，使用 Ascend C 编程语言实现单精度复数
（complex64）扩展对称秩-k 更新算子 `aclblasCsyrkx`，适配 Atlas 950PR（arch35 / DAV_3510），
验收通过后合入 `blas/syrkx/arch35`，测试工程合入 `test/syrkx/csyrkx/arch35`。

### 1.2 背景介绍

#### 1.2.1 算子语义

`aclblasCsyrkx` 是 BLAS Level-3 对称秩-k 扩展更新算子，对标 cuBLAS legacy 接口
`cublasCsyrkx`：

```text
C = alpha * op(A) * op(B)^T + beta * C
```

- `trans = ACLBLAS_OP_N`：A、B 均为 n×k（列主序），`C = alpha*A*B^T + beta*C`；
- `trans = ACLBLAS_OP_T`：A、B 均为 k×n，`C = alpha*A^T*B + beta*C`；
- `trans = ACLBLAS_OP_C`：**与 OP_T 完全等价（不取共轭）**。syrkx 的输出是对称
  （非 Hermitian）矩阵，共轭转置会破坏对称性；cuBLAS `cublasCsyrkx` 对
  `CUBLAS_OP_C` 亦按 `CUBLAS_OP_T` 执行（P102 + CUDA 12.5 实测 trans=T 与
  trans=C 的输出逐位相同）；
- C 为 n×n 对称复数矩阵，仅引用并更新 `uplo` 指定三角（ACLBLAS_UPPER /
  ACLBLAS_LOWER），另一三角不引用、不写回；
- alpha、beta 为复数标量（Device 内存指针），矩阵按列主序前导维存储。

与 `aclblasCsyrk` 的本质区别是第二个因子独立为 B（`A*B^T` 而非 `A*A^T`），
因此 A、B 各有独立前导维 lda/ldb，乘积不再是 Gram 矩阵，对角元素的相消特性
也与 csyrk 不同（A==B 时退化为 csyrk 语义，精度路径必须覆盖该退化）。

#### 1.2.2 现状分析

ops-blas 仓已有同族实数算子 `aclblasSsyrkx`（arch35）与复数算子 `aclblasCsyrk`
（arch35，含 fused-4M cube 驱动与共享层），但无 `aclblasCsyrkx` 接口声明与实现，
`include/cann_ops_blas.h` 需按任务书 §2.3 新增声明（签名与 cuBLAS 参数序列一致，
禁止 950PR 私有平行接口）。

复数 BLAS3 族在 arch35 上的成熟解法是「AIV 拆复数 + AIC 实数 GEMM + AIV 合并」
三阶段流水（csyrk/csyr2k/cherk 等 6 个已合入 PR 互证）。csyrkx 的增量工作：
4M 分解的左右操作数分属 A 与 B、deinterleave 单次启动处理双矩阵、对角直算公式
四流读入、以及**双操作数使小 shape 下多 kernel 固定开销占比进一步放大**带来的
路径选择与性能工程（本设计 §3.6）。

### 1.3 参考实现分析

| 参照 | 用途 |
| --- | --- |
| cuBLAS `cublasCsyrkx`（CUDA 12.4 / H100） | 接口语义、性能标杆（gpu_baseline.csv） |
| Netlib BLAS `cblas_cgemm`（3.10.0 参考实现） | 精度 golden：trans=N 用 NoTrans+Trans、trans=T/C 用 Trans+NoTrans，随后把非 uplo 三角恢复为调用前原值 |
| 任务书 §3.2 生态算子开源精度标准 | COMPLEX64 按 FLOAT32 分量判定：rtol=atol=2⁻¹³、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32×ULP) |

## 2. 需求分析（required）

### 2.1 功能需求

1. `uplo`（UPPER/LOWER）× `trans`（N/T/C）全枚举，complex64 全参数；
2. 参数校验严格对齐任务书 §2.4：非法枚举 / 负 n/k / 前导维不足 / alpha、beta 空指针 /
   C 空指针（n>0 时）→ `ACLBLAS_STATUS_INVALID_VALUE`，handle 空 →
   `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；A、B 仅在「n>0 且 k>0」时报空指针；
3. BLAS quick-return 语义：`n=0` 直接 SUCCESS 不写 C；`k=0` 仅对 uplo 三角执行
   `C = beta*C`（beta=(0,0) 置零、beta=(1,0) 不变），不引用 A/B，另一三角不改写；
   `(alpha=(0,0) 或 k=0) 且 beta=(1,0)` 时 C 不变；参数校验先于 no-op 判定；
4. 确定性计算：同输入多次执行逐位一致（所有路径无原子操作、无跨核写重叠）。

### 2.2 性能需求（任务书 §3.3）

- 采样：msprof 采集单算子 task duration（us），warmup 5 次不计入，连续 10 次有效
  采样取算术平均，仅 SUCCESS 样本计入；
- 判据（两项须同时满足）：全部 200 条性能 case 单条倍率（GPU 标杆耗时 / NPU 耗时）
  ≥ 0.4（即 NPU ≤ 2.5× H100 标杆），且全部 case 倍率算术平均 ≥ 0.5。

### 2.3 精度需求

COMPLEX64 按实部/虚部 FLOAT32 分量判定（§3.2 口径）；非 uplo 三角须与调用前
**逐位一致**；Inf/NaN 按 §3.2-4 等效规则（同为 NaN / 同号 Inf / NaN↔Inf 等效；
非有限值元素不计入统计，须全部按规则判定通过）。

### 2.4 需求拆解

1. host 侧：句柄式 API、参数校验、Device 标量读回、路径选择、workspace 规划
   （含 64 位溢出防护）、三阶段 tiling 与 kernel 调度；
2. 小 shape：单 kernel SIMT 直算（warp-Kahan，精确路径）；
3. Phase 0（AIV）：A、B 双矩阵一次启动解交错为 Ar/Ai/Br/Bi；
4. Phase 0b（AIV SIMT，按需）：HF32x3 低残差平面；
5. Phase 1（AIC）：fused-4M 实数 cube（quad-split 小形状并行 / band-driver 大形状）；
6. Phase 2（AIV）：三角 combine（列突发对角 tile + 两遍式对角直算）；
7. 精度保障：useDirectDiag 对角重算、HF32x3 补偿、特殊值 host 兜底；
8. 测试工程：CSV 驱动 GTest 全量 1200 用例 + msprof/mstx 性能采集。

## 3. 详细设计（required）

### 3.1 数学分析：4M 实数分解

记 op(A) = Ar + i·Ai、op(B) = Br + i·Bi（trans 仅改变 A/B 的读取布局，不改变分解形式）：

```text
Q0 = Ar·Br^T    Q1 = Ai·Bi^T    Q2 = Ar·Bi^T    Q3 = Ai·Br^T
Cr = Q0 − Q1    Ci = Q2 + Q3    C = alpha·(Cr + i·Ci) + beta·C_old
```

四个乘积均为实数矩阵乘，等价计算量 4·n²k（朴素复数 GEMM 为 6·n²k，节省 1/3），
全部复用仓内 csyrk 的 fused-4M cube 驱动与共享层
（`common/helper/syrk_gemm_arch35.h`、`syrk_gemm_fused_arch35.h`）。

### 3.2 总体架构与路径选择

```text
                    ┌─ n<=16，或 n<=64 且 k>128 ──────────────► csyrkx_direct_kernel（单 kernel，warp-Kahan）
                    │
aclblasCsyrkx ──────┤  alpha=(0,0)|k=0 且 beta=(1,0) ────────► 直接返回（C 不变）
   （host 校验/     │
   标量读回后）     ├─ n<=640 且 k<=512（fp32 区）───────────► Phase0 解交错 ─► Phase1 fp32 cube ─► Phase2 combine
                    │                                              （n<=1024 时 quad-split 4 倍分核）
                    └─ 其余（大 n 或 k>512，HF32x3 区）────────► Phase0 ─► Phase0b 残差 ─► Phase1 HF32x3 cube ─► Phase2 combine
```

combine 子路径：fast-config（alpha=(1,0)、beta=(0,0)）且 n≤64 走 SIMT 逐元素 kernel；
n>64 走 tile kernel（FastDiagTile 列突发对角相位 + 两遍式 useDirectDiag，§3.5.3）。

**路径门限全部有实测依据**（950PR，msprof task duration 口径）：

| 决策点 | 门限 | 依据 |
| --- | --- | --- |
| direct kernel | n≤16 全 k；n≤64 且 k>128 | (n≤64, 128<k≤512) 是对角相消精度角落，直算 kernel 精确且最快：20×227 单 kernel ~6us vs 流水线+SIMT 对角重算 59us；但 k≤128 时 3-kernel 流水更快（32×32：6.7us vs 7.2us；64×64：11.9us vs 11.4us），故不整体下放 |
| fp32 vs HF32x3 | n≤640 且 k≤512 用 fp32 | fp32 mmad 速率约为 HF32 的 1/4：3664×115 实测 fp32 cube 277us vs HF32x3 197us，故大 n 必须留 HF32x3；n≤640 区 cube 时间由启动/L1 填充/fixpipe 主导，fp32（少一次 resid 启动 + quad-split 4 倍分核）占优 |
| quad-split | !hf32 且 n≤1024 | (n/64)²/2×4 个 quad 任务在数百内，拆分仍能买到并行度；超过后 band-driver 已饱和核数 |
| combine SIMT | n≤64（含 useDirectDiag） | 单 tile 尺寸下向量化 kernel 的 per-op 开销占主导；n≤64 时 SIMT 的逐元素对角直算（CsyrkxSimtDirectDiag）无宽形状串行化问题 |

### 3.3 host 侧设计

入口 `aclblasCsyrkx`（`blas/syrkx/arch35/csyrkx_host.cpp`）：

1. **校验**（`ValidateCsyrkxParams`）：uplo/trans 枚举 → 前导维（trans=N 按 max(1,n)、
   T/C 按 max(1,k)）→ alpha/beta 指针 → C 指针（n>0 时）；A/B 空指针在标量解析后
   按值相关条件判定（BLAS 语义：alpha=(0,0) 或 k=0 时 A/B 可空）；
2. **n=0 quick-return**：校验全部通过后判定；
3. **标量读回**（`ReadComplexScalarsFromDevice`）：alpha/beta 为 Device 指针，
   两次 `aclrtMemcpyAsync` + 一次 `aclrtSynchronizeStream` 读回，得到
   isAlphaZero / isKZero / isBetaZero / isFastCfg；
4. **C-不变短路** 与 **beta-only 分支**：后者仅启动 combine 的 beta 分支，
   不分配 workspace；
5. **小 shape 分流**（`CsyrkxTryLaunchDirect`）：direct kernel 读取 GM 原始复数矩阵，
   无 workspace、无 Phase0-2；
6. **特殊值兜底**（`CsyrkxTrySpecialValueFallback`，n·k ≤ 128×128 时）：D2H 扫描
   A/B 及标量，命中 Inf/NaN/|v|>1e19/亚正常值时走 host Netlib 参考循环回写，
   保证 4M 分解的 Inf/NaN 传播位置与 golden 一致（大小门槛保证热路径零同步）；
7. **workspace 规划**（`CsyrkxEnsureWorkspace`）：布局
   `[Ar | Ai | Br | Bi | (ArLow | AiLow | BrLow | BiLow) | temp(4n×tempLdc)]`，
   arLdc = 8 对齐的行数、tempLdc = 8 对齐的 n；全部字节数经 64 位乘加溢出检查
   （uint32 维数相乘回绕会绕过 2 GiB 上限），超限返回 ALLOC_FAILED 并打印需求量；
8. **tiling 与分核**：Phase0/2 用 AIV 核数（小形状限核避免每核工作过薄）、
   Phase1 用 `GetUsedAicCoreNum`；三阶段 tiling 结构见 `csyrkx_tiling_data.h`。

### 3.4 kernel 侧设计（一）：小 shape 直算

`csyrkx_direct_kernel`（AIV SIMT，`csyrkx_direct_kernel.cpp`）：

- 线程映射：uplo 三角逐元素，行方向为快轴（列主序下 A/B/C 访问合并）；
  trans=N 且 n≥64 时 rowLanes=4 做 k 维 warp 分片；
- 精度：**warp-Kahan 补偿求和**（k>512 时补偿量穿过 shuffle 归约树），
  每元素先乘后减再累加，32-lane 归约；求值顺序与容差特性与参考实现一致，
  覆盖 (n≤64, k>512) 的 cube HF32 精度盲区与对角相消角落；
- 特殊值：checked 模式逐项扫描，不安全元素回落 `CsyrkxReferenceElement`
  （C99 `__mulsc3` 复数乘法语义，Inf/NaN 位置与 golden 一致）；
- 实测：n≤32 单 kernel 2.3~2.4us，倍率 1.25~1.35×（GPU 标杆 3us 级）。

### 3.5 kernel 侧设计（二）：三阶段流水

#### 3.5.1 Phase 0：csyrkx_deinterleave_kernel（AIV）

A、B 在**同一 kernel、同一核段**内完成解交错（csyrk 需两次启动，csyrkx 合一）：
列段分核，列内按 2040 元素批量，2D `DataCopyPad`（Compact）整块读入 →
向量 `DeInterleave` 拆实/虚 → 2D 写回 Ar/Ai/Br/Bi 实平面。
非紧凑 lda/ldb 走通用按列路径。

#### 3.5.2 Phase 0b / Phase 1：HF32 残差与 fused-4M cube

- `csyrkx_hf32resid_kernel`（SIMT）：计算 `x − HF32(x)` 低残差平面（HF32 就近舍入）；
- `csyrkx_cube_kernel`（AIC）：quad q 的左右操作数
  `left = (q==0||q==2)?Ar:Ai`、`right = (q==0||q==3)?Br:Bi`；
  - trans=N：left 用 DNExt(aLdc,k)、right 用 NDExt(k,bLdc)；
    trans=T/C：left 用 NDExt(n,aLdc)、right 用 DNExt(bLdc,n)；
  - temp 布局 `[Q0|Q1|Q2|Q3]`，quad 间距 n·tempLdc；
  - 两条驱动：**quad-split**（fp32 小形状：一个 quad 一个任务，4 倍核并行，
    L1 双缓冲 + L0 ping-pong + FIX_M 握手）与 **band-driver**（大形状：
    行带主序三角 tile 遍历 + 蛇形 n 序，M 侧 panel 带内 L2 命中）；
  - HF32x3：三次 mmad（xh·yh + xl·yh + xh·yl）恢复 fp32 精度，k>512 或大 n 启用；
  - AIC 核 <4 时回退单核串行 4 quad（`SyrkGemmKernelImpl` 通用驱动）。

#### 3.5.3 Phase 2：csyrkx_combine_kernel（AIV）

`CsyrkxKernel` 类三部分：

- **非对角半块**（CombineHalfFast）：4 次 64×64 quad 突发读 → `Sub/Add` 合并
  Cr/Ci → `Interleave` 交错 → 单次 64 块突发写 C；软件流水（prefetch/combine
  双缓冲 + 同轮 Set/Wait MTE3_V drain——`PipeBarrier` 只栅栏发射顺序，
  不能保证 MTE3 完成，UB 源复用前必须事件 drain）；
- **对角 tile（FastDiagTile）**：整 tile 四象限列突发读（含 tempLdc padding
  过读，垃圾行不出向量域）→ 单次 Sub/Add + Interleave → **逐列部分突发写**
  （UPPER 列 j 写行 [0,j]、LOWER 写 [j+1,cols)，天然只覆盖 uplo 三角；
  UPPER 源偏移天然 32B 对齐，LOWER 以 4 行对齐分裂 head≤3 元素——head 暂存到
  32B 对齐 scratch 后用已验证的 4B 块形式发出，规避 DataCopyPad UB→GM 源
  必须对齐的约束）。该设计将对角相位的 MTE 块粒度从 4B/元素提升到整列
  8B×len 突发，n=569 实测对角相位 ~10us → ~2-3us；
- **useDirectDiag（k>512 对角精度兜底），两遍式地址不相交**：
  - A 遍：全部 AIV 核按 8 行组轮转重算对角线（`ComputeDiagBatch8`，
    8 行共享连续读 + 元素级累加 + 一次归约），直接写 C 的 n 个对角元素；
  - B 遍：FastDiagTile 的逐列写**排除对角元素**（UPPER 行 [0,j)、LOWER 行 [j+1,cols)）；
  - 两遍 C 元素地址不相交 → 无跨核写序依赖、无需核间同步，对角重算恢复
    全核分布（修复单核串行化回归：357×2338 实测 190us → 58us）。
- **beta-only / SIMT 分支**：alpha=(0,0) 或 k=0 时仅 `C = beta*C_old`；
  n≤64 fast-config 走 `csyrkx_combine_simt_kernel`（逐元素，内置
  `CsyrkxSimtDirectDiag` 对角重算，无 TPipe 开销）。

### 3.6 关键工程决策汇总（含实测证据）

| # | 决策 | 证据（950PR，msprof task duration） |
| --- | --- | --- |
| 1 | fp32/HF32x3 门限 n≤640 | 3664×115：fp32 cube 277us vs HF32x3 197us（fp32 MAC 速率 ~1/4）；256<n≤640 且 k≤512 少一次 resid 启动（5.5~7.8us）且 quad-split 占优 |
| 2 | 对角相位列突发化 | n=569 combine 16.6us → 8.8us；n=3664 combine 184us → 132us |
| 3 | directDiag 两遍式 | 357×2338 combine 190us（单核串行回归）→ 58us |
| 4 | direct kernel 覆盖 (n≤64, k>128) | 20×227：direct ~6us vs 流水线+SIMT 对角 59us；TC_EX_0508（17×200）精度由该路径保证 |
| 5 | MTE3 事件 drain | outBuf_ 复用竞态：PipeBarrier 不等 MTE3 完成，须 Set/Wait MTE3_V 同轮握手（联调实测否则对角重算暂存被覆写） |
| 6 | UB→GM 源 32B 对齐 | LOWER 列头非对齐源会使 kernel 异常终止，head 暂存对齐 scratch 解决 |
| 7 | mstx range 标注 | 测试程序内置 CSYRKX_MSTX_RANGE 开关：每次调用唯一命名 range，供 `msprof op --replay-mode=range --mstx=on` 按 case 采集（默认 no-op，不影响常规运行） |

### 3.7 支持硬件与约束

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35 / DAV_3510） | √ |

1. 仅 complex64；不支持非连续 Tensor 步长、广播、编译期未知 dynamic shape；
2. C 为对称（非 Hermitian）矩阵，对角虚部不置零；OP_C ≡ OP_T（不共轭）；
3. 仅更新 uplo 三角，另一三角不引用、不写回（golden 逐位校验）；
4. workspace ≤ 2 GiB（64 位溢出检查 + 显式报错）。

## 4. 可维可测分析

### 4.1 验收标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度 | FLOAT32 分量：rtol=atol=2⁻¹³、matched_ratio≥0.99、max_abs_error≤max(1e-2,32×ULP)；非 uplo 三角逐位一致；Inf/NaN 按 §3.2-4 等效规则 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能 | 单 case ≥0.4×（NPU ≤2.5× H100 标杆）且平均 ≥0.5×，两项同时满足 | 任务书 §3.3 |
| 内存 | 单用例 host 侧 ≤512MB | 测试包 README |

### 4.2 自测设计

- **用例**：任务包 `csyrkx_test.csv` 1200 条（1000 精度 + 200 性能，固定种子
  20261028 逐字节可复现），覆盖 L0 基础 / SQ 方阵 / MN 矩形 / AB 标量 /
  LD 前导维 padding / FL 填充（含 Inf/NaN/极端值）/ CV 覆盖 / ED 边界负向 /
  EX 扩展 / PF 性能；
- **精度框架**：ops-blas CSV 驱动 GTest（`test/syrkx/csyrkx/arch35/csyrkx_test.cpp`），
  golden 为 Netlib cblas_cgemm（`csyrkx_golden.h`：OP_C 按 OP_T 不共轭，
  合并后非 uplo 三角恢复原值）；比对 uplo 三角实/虚部 + 非 uplo 逐位一致 +
  Inf/NaN 等效规则；
- **性能框架**：每 case 5 次 warmup + 10 次有效采样（任务书 §3.3 协议），
  `[PERF]` 输出 kernel 路径 wall time 与采样窗口；msprof 采集 task duration 后
  按 kernel 调用序列归属（200×16 次调用恰好闭环、零解析误差的算术校验，
  免疫主机/设备时钟漂移）；测试程序同时内置 mstx range 标注（§3.6-7），
  在具备完整采集通道的环境可直接按 case 用 msprof op range 模式复核。

### 4.3 实测结果（Atlas 950PR / CANN 9.1.0）

| 项 | 结果 |
| --- | --- |
| 精度 | 1000/1000 全部通过（含 Inf/NaN、边界负向、A==B 退化、非紧凑前导维） |
| 性能·单 case | 200/200 全部 ≥0.4×，最差 0.45~0.454（357×2338 计算受限型），两轮独立复测稳定 |
| 性能·平均 | 0.706~0.709（门槛 0.5，余量 +41%） |
| 典型倍率 | n≤32 直算 1.25~1.35×；256~569 中形状 0.53~0.72×；4096 大形状 0.5~0.6× |

## 5. 交付与合入

- 代码：`blas/syrkx/arch35/`（host/kernel/tiling）+ `include/cann_ops_blas.h` 声明；
- 测试：`test/syrkx/csyrkx/arch35/`（CSV 驱动 GTest + golden + 参数解析 + mstx 标注）；
- 本设计文档随 PR 提交至 cann-competitions 任务目录，评审通过后合入。
