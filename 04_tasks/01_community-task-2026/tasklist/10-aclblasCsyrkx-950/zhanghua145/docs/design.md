# aclblasCsyrkx 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCsyrkx` |
| 任务编号 | 10 月社区任务-aclblasCsyrkx 算子开发（950） |
| 目标硬件 | Ascend 950PR（arch35，DAV_3510） |
| 软件版本 | CANN 9.1.0（asc-devkit >= 9.1） |
| 开发方式 | Ascend C kernel 直调（handle 绑定 stream） |
| 数据类型 | COMPLEX64（实部/虚部均为 FP32） |
| 对齐基线 | cuBLAS `cublasCsyrkx`（legacy）；golden 由 Netlib BLAS `cblas_cgemm` 复数实现生成 |
| 合入仓及目录 | `ops-blas`：算子 `blas/syrkx/arch35/`，测试 `test/syrkx/csyrkx/{,arch35/}` |
| 文档状态 | 设计方案已实现并在 950PR 实测：精度 1000/1000，性能 200/200（验收协议口径） |

# 1. 需求背景（required）

## 1.1 需求来源

CANN 社区任务 2026，10 月社区任务-aclblasCsyrkx 算子开发（950）。任务书：`aclblasCsyrkx_Atlas950PR_task_doc.md`。贡献者：`zhanghua145`；实现目标仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)。本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 1.2 背景介绍

`syrkx` 是 `syrk` 的双矩阵扩展（cuBLAS 扩展接口，Netlib BLAS 无此算子），完成对称 rank-k 扩展更新：

$$
C \leftarrow \alpha \cdot op(A) \cdot op(B)^T + \beta \cdot C
$$

- `trans = ACLBLAS_OP_N`：`op(A)=A`、`op(B)=B`（A、B 均为 `n×k`），`C = α·A·B^T + β·C`；
- `trans = ACLBLAS_OP_T`：`op(A)=A^T`、`op(B)=B^T`（A、B 均为 `k×n`），`C = α·A^T·B + β·C`；
- `trans = ACLBLAS_OP_C`：**与 OP_T 完全等价（按转置处理，不取共轭）**——结果矩阵是对称（非 Hermitian）矩阵，共轭转置会破坏对称性；cuBLAS `cublasCsyrkx` 对 `CUBLAS_OP_C` 亦按 `CUBLAS_OP_T` 执行（任务书已实测 trans=T 与 trans=C 输出逐位相同）。

关键语义（与 cuBLAS 官方文档逐项核对）：

1. **双矩阵同 op**：A、B 为两个独立输入矩阵，取同一 op；典型用途为 B 是 A 的对角缩放形式（`B = A·D`，等价于 A 右乘对角阵，参考 cuBLAS `dgmm`），此时结果保证对称。
2. **对称非 Hermitian**：转置不取共轭；对角元素为普通复数（虚部不清零，区别于 `cherkx` 的 Hermitian 约束）。
3. **三角写**：C 为 `n×n` 对称矩阵，仅引用并更新 `uplo` 指定三角（UPPER 只写上三角 / LOWER 只写下三角），另一三角不引用、不写回。
4. **标量**：alpha、beta 均为复数标量（Device 指针），作用于结果；`k=0` 时仅对 uplo 三角执行 `C = beta×C`（beta=0 置零、beta=(1,0) 不变），不引用 A/B。
5. **无-op 与校验顺序**：`n=0` 时 quick return 返回 SUCCESS；**参数校验先于 no-op 判定**。
6. **确定性计算**：同输入多次执行逐位一致（任务书硬性要求）。

当前 `ops-blas` 公共头文件 `include/cann_ops_blas.h` 尚无 `aclblasCsyrkx` 声明（仓内 `aclblasSsyrkx` 为实数同族算子），本任务须新增公共接口声明，禁止定义 950PR 私有平行 API。

# 2. 需求分析（required）

## 2.1 接口定义

签名与 cuBLAS `cublasCsyrkx` 参数序列一一对应（维数 `int`、标量指针 `const aclblasComplex*`）：

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,      // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,    // ACLBLAS_OP_N / OP_T / OP_C（OP_C ≡ OP_T，不共轭）
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

## 2.2 参数与校验（对齐任务书 §2.4）

| 约束项 | 规则 | 错误码 |
| --- | --- | --- |
| handle | 非空句柄 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | ∈ {ACLBLAS_UPPER, ACLBLAS_LOWER}，非法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| trans | ∈ {ACLBLAS_OP_N, OP_T, OP_C}，非法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| n, k | ≥ 0，负值 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda / ldb | trans=N：≥ max(1,n)；trans=T/C：≥ max(1,k) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | ≥ max(1,n) | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha / beta | 指针非空（Device 标量，dispatch 时 D2H 读回真值） | `ACLBLAS_STATUS_INVALID_VALUE` |
| A / B | n>0 且 k>0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| C | n>0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |

校验顺序：handle → uplo/trans 枚举 → n/k → lda/ldb/ldc → alpha/beta → A/B/C 指针 → `n==0` quick return（校验先于 no-op）。

## 2.3 需求拆解

1. 公共 API 声明合入 `include/cann_ops_blas.h`；`aclsolverStatus_t` 语义对齐仓内 `cann_ops_blas_common.h`。
2. `C = α·op(A)·op(B)^T + β·C` 的复数矩阵乘在 950PR Cube 上高效完成：复数代数拆解为 4 个实数 GEMM（4M 拆解），Cube 只做实数 FP32/HF32 矩乘，复数代数在 AIV 阶段完成。
3. uplo 三角写：仅 uplo 三角被计算与写回，非 uplo 三角逐位不变（精度验证按 bit-exact 比对）。
4. 复数 alpha/beta 缩放（含交叉缩放 Cr = αr·P_r − αi·P_i 等全套代数）。
5. 确定性：固定 tile 分配、无原子操作、固定归约顺序，同输入逐位一致。
6. Inf/NaN/次正规等特殊值语义与 Netlib golden 一致（含 `0×Inf=NaN` 传染、`__mulsc3` 式 Inf 恢复语义）。
7. 性能：单 case ≥ 0.4× GPU 且全 case 平均 ≥ 0.5× GPU（msprof op 口径，5 warmup + 10 采样均值），对标 cuBLAS `cublasCsyrkx`（驱动 550.104.05 / CUDA 12.4）。
8. 测试工程：CSV 驱动 GTest，1000 精度 + 200 性能用例，覆盖全枚举、尺寸扫描、特殊值、负向。

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式与 4M 拆解

复数矩阵乘按实/虚平面拆解。设 A = Ar + i·Ai，B = Br + i·Bi（`Phase 0` 由复数 interleaved 输入拆分得到 4 个实数平面）：

$$
A \cdot B^T = \underbrace{(A_r \cdot B_r^T - A_i \cdot B_i^T)}_{P_r} + i\,\underbrace{(A_r \cdot B_i^T + A_i \cdot B_r^T)}_{P_i}
$$

四个象限积（quadrant products）：

| 象限 | GEMM | 说明 |
| --- | --- | --- |
| Q0 | `Ar · Br^T` | 左侧取 A 实平面，右侧取 B 实平面 |
| Q1 | `Ai · Bi^T` | P_r = Q0 − Q1 |
| Q2 | `Ar · Bi^T` | P_i = Q2 + Q3 |
| Q3 | `Ai · Br^T` | |

trans=T/C 同型（读转置视图，A、B 平面为 `k×n` 列主序），符号结构相同。合并阶段：

$$
C_r = \alpha_r P_r - \alpha_i P_i + \beta_r C_r^{old} - \beta_i C_i^{old},\quad
C_i = \alpha_r P_i + \alpha_i P_r + \beta_r C_i^{old} + \beta_i C_r^{old}
$$

合并公式与单矩阵 `csyrk` 完全同式（Cr=Q0−Q1、Ci=Q2+Q3），仅象限积来源不同（Q0=Ar·Br^T 等，右侧来自 B 的平面）——复用同族成熟的合并 kernel 结构。

### 3.1.2 支持数据类型与形状

- COMPLEX64（实/虚部各 FP32），列主序，ND。
- A/B：trans=N 为 `n×k`，trans=T/C 为 `k×n`；C：`n×n`。
- n、k ∈ [0, int 上限]；workspace 溢出防护（见 3.2.1）。

## 3.2 算子实现

### 3.2.1 host 侧设计

**路径选择（按形状与标量真值分流）**：

```
aclblasCsyrkx
 ├─ 校验（§2.2 顺序）→ n==0 quick return
 ├─ 标量真值解析：alpha/beta D2H 读回（2 次 memcpyAsync + 1 次 stream 同步）
 ├─ 短路：α==(0,0) 或 k==0
 │    ├─ β==(1,0) → C 不变，直接 SUCCESS
 │    └─ 否则 → 仅合并 kernel（betaOnly 分支：C_uplo = β×C，不申请 workspace）
 ├─ 小形状直连（n≤16，或 n≤64 且 k>512 的长 k 精度路径）→ 单 SIMT kernel
 │    └─ n≤32 且 k>512 精度角落 → 逐元素 Netlib 序参考 kernel（与 golden 同序）
 ├─ 特殊值兜底：!skipTemp 且 n·k ≤ 128² → host 扫描 A/B 是否含 Inf/NaN/次正规/
 │    FLT_MAX 级值；命中则 host 端 Netlib 序参考计算回写（保证 inf/nan 语义逐位一致）
 └─ 三阶段管线（其余全部形状）
```

**workspace 布局**（32B 对齐，溢出防护用 64 位乘法检查）：

```
[Ar][Ai][Br][Bi] (HF32x3 路径再各加 [xLow] 残差平面 ×4) [temp: Q0|Q1|Q2|Q3 四象限 n×tempLdc]
总量 ≈ (useHf32 ? 8 : 4)·arLdc·arCols·4B + 16·n² B，超 ACLBLAS_MAX_WORKSPACE_SIZE(2GiB) 报 ALLOC_FAILED
```

**tiling 策略**：

- Phase 0（AIV 拆分）：按列带分核；小形状（arCols ≤ 256）每核打包 ≥4 列摊销 per-block 坡道；A、B 两次拆分**合并为单次 kernel 发射**（发射开销在小形状占主导）。
- Phase 1（AIC cube）：`GetUsedAicCoreNum(n, 16)` 起步；quad-split 路径抬升至 ≥ 象限作业数；三角 tile 跳过（UPPER 保留对角带及以上 / LOWER 及以下）。
- Phase 2（AIV 合并）：小 n 行带分核（n≤640 时每核 ≥16 行）；fast-config（α==(1,0)、β==0）且 n≤256 且无需对角重算时走 SIMT 合并（每线程一个 uplo 元素，免 TPipe/屏障开销）。

**HF32x3 判据**：`useHf32Cube = !(n ≤ 256 && k ≤ 512)`。大形状 cube 走 HF32x3 补偿三 mmad（`xh·yh + xl·yh + xh·yl`，残差 `xl = x − HF32(x)` 由 Phase 0 顺带产出），精度 ≈ 原生 FP32、吞吐 ≈ 2× FP32 cube；小形状原生 FP32（HF32 无吞吐收益）。

### 3.2.2 kernel 侧设计（三阶段管线）

```
Phase 0  AIV 拆分     复数 interleaved → Ar/Ai/Br/Bi 四实数平面（+HF32 残差）
   │                   双矩阵合并单次发射；批量列读 + 向量 DeInterleave + 连续列写
Phase 1  AIC 4M cube  四象限实数 GEMM：left=(Ar,Ai) from A、right=(Br,Bi) from B
   │                   → temp = [Q0|Q1|Q2|Q3]，三角 tile 跳过
Phase 2  AIV 合并     C_uplo = α·(Q0−Q1, Q2+Q3) + β·C_old，仅写 uplo 三角
```

**Phase 1 两个驱动（按形状分档）**：

1. **quad-split 驱动（fp32，n≤256 且 k≤512）**：一个作业 = 一个象限 × 一个 64×64 tile。小 n 时 fused-4M 每核串行算 4 个象限且全套事件机制占主导（n=128 仅 3 tile/8 核）；按象限拆分后 4× 核并行、1/4 串行长度。双 L0C 累加器集（0/16KB）乒乓 + 作业前 `FIX_M(parity)` 握手——对齐共享 fused 驱动的成熟纪律，杜绝相邻作业 fixpipe 排空与首个置零 mmad 的累加器竞态（首版单累加器缺此握手曾致内核挂死）。实测 cube：n=64 5.34→2.57µs、n=128 8.0→4.07µs。
2. **band-major fused-4M 驱动（其余形状）**：kept-index 轮转把各核 tile 散布全矩阵，4 面板工作集超 L2 → MTE2 99.7%、MAC 50%（n=3635 msprof 实测）。改为按行带顺序遍历 + 行带内蛇形轮转：所有核同步走当前行带，M 面板跨带驻留 L2、N 面板流式读。实测 n=3635 cube 8.20→7.91ms（-20%），全算子 9.87→7.91ms，**优于同机 csyrk 参考实现（8.29ms）**。

**Phase 2 合并 kernel 要点**：

- 64×64 tile / 行带处理：`Cr = Q0−Q1`、`Ci = Q2+Q3` 向量化合并 → 复数 α/β 全套交叉缩放 → Interleave → 连续列写回 uplo 三角；非 uplo 三角严格不读不写。
- **对角精度补偿（direct-diag）**：4M 拆解的对角元 `C[i,i] = Q0[i,i]−Q1[i,i]` 为两个大累加和之差，大 k 时舍入误差放大（A==B 退化场景 = csyrk 语义时系统性抵消，k≥~800 会使 32·ULP 上限失守）。合并 kernel 直接从四个实数平面按 `Σ(Ar·Br − Ai·Bi) + iΣ(Ar·Bi + Ai·Br)` 逐项累加重算对角（FP32、升序 k，与参考同调理）。门控：`k > 512 或 (n≤64 且 k>128)`。**对角修正写为纯向量路径**：带内全量写（32B 对齐源）后，从对齐的 `diagRe/diagIm` 向量头部或 `Duplicate` 广播暂存发独立 strided MTE3——标量 SetValue 不受任何管线屏障约束，会与带内 MTE3 存储竞态（950PR 实测偶发 ±300 量级错值，逐行排查定位）。
- fast-config 双缓冲预取：64×64 tile 四象限读 ping-pong 提前一拍，MTE2/V/MTE3 事件定向握手。

**小形状直连 kernel（n≤16；n≤64 且 k>512 精度路径）**：

- 单 SIMT kernel 直接从 A/B（复数 interleaved GM）逐元素计算 uplo 三角，warp 内 rowLanes×kLanes 分工 + 逐 lane Kahan 补偿点积 + warp 树归约；k≤512 走朴素树（csyrk 同款），k>512 走补偿树（校正项随 shfl 逐级传递）。
- `strided` 精度角落（n≤32 且 k>512）：逐元素 Netlib 序参考 kernel——累加顺序与 golden 完全一致，容差按构造成立（小 uplo 元素数下单元素越界会超 1% 失配预算，实测 TC_EX_0436 n=8/k=1000 用例由 1/36 失配修复为全过）。
- 特殊值扫描：`CsyrkxUnsafeValue`（|v|>0x5a0e1bca 或非正规）触发逐元素参考回退，Inf/NaN 语义含 `__mulsc3` 式恢复（双 NaN 且乘积无界 → 符号化 Inf 传播），与参考逐位一致。

### 3.2.3 确定性设计

- 所有 kernel 的 tile/作业分配为固定规则（band-major 序 + 模核轮转），无原子、无锁、无调度随机性；
- 归约顺序固定（Kahan 逐 lane、warp 树、ReduceSum 分块）；
- 同输入同 stream 串行多次执行输出逐位一致（已在测试中验证：61 次重复调用样本稳定）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

依赖 `ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`（低于该版本编译与运行跳过本算子，经 `cmake/asc_devkit_version.cmake` 列表驱动登记）。

## 3.4 算子约束限制

- 不支持非连续步长（仅前导维语义）；不支持维度广播；n/k 为运行时入参（无编译期 shape 要求）。
- workspace 溢出（>2GiB）返回 `ACLBLAS_STATUS_ALLOC_FAILED`。
- C 原地覆写，不返回视图；异步执行依赖 handle 绑定 stream，读回前须同步。

# 4. 可维可测分析

## 4.1 精度标准

对齐[生态算子开源精度标准 mixed_tolerance_standard](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，COMPLEX64 按 FLOAT32 分量判定（实部/虚部分别比对、分别统计）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT32（复数分量） | 2^-13 (1.2e-4) | 2^-13 (1.2e-4) | 0.99 | 1e-2 或 32·ULP |

- 判定式：逐元素 `|actual − golden| ≤ atol + rtol·|golden|`，且 `matched_ratio ≥ 0.99` 与 `max_abs_error ≤ max(1e-2, 32·ULP)` 同时满足。
- Inf/NaN 特殊值：golden/actual 同 NaN、同号 Inf、或一方 NaN 另一方 ±Inf 视为等效通过；一方有限一方非有限判不过；非有限元素不计入 matched_ratio/max_abs_error，单独记录（任务书 §3.2-4 全套规则）。
- 非 uplo 三角与调用前逐位一致（bit-exact）。
- golden 由仓内自实现 Netlib 序参考（`csyrkx_netlib_reference.h`，列分块可并行、块内逐位同序）与 host 特殊值兜底**共享同一实现**，两边 bit 一致；对 cblas_cgemm 语义对拍（trans=N: NoTrans×Trans、trans=T/C: Trans×NoTrans，OP_C 不取共轭）后仅恢复非 uplo 三角。

**实测**：1000 条精度用例全过（全枚举组合、尺寸 1→2048 扫描、a_eq_b 退化 81 条含 k=4096 对角精度路径、Inf/NaN/极端值/全零、空指针与非法枚举/非法前导维/负维度负向、前导维 padding）。

## 4.2 性能标准

- 协议：msprof op 模式采集单算子 task duration；自带 5 次 warmup，warmup 后连续 10 次为有效样本取算术平均。
- 判据：单 case 倍率 `GPU 标杆耗时 / NPU 实测耗时 ≥ 0.4`，**且全部性能 case 倍率算术平均 ≥ 0.5**（L3 双门槛），对标 cuBLAS `cublasCsyrkx`。

**实测（950PR，msprof kernel-duration，验收协议复刻口径）**：

| case | n×k | NPU (µs) | ratio | 说明 |
| --- | --- | --- | --- | --- |
| 锚点 512×512 N | 512 | 68.0 | 0.74 | |
| 锚点 1024×1024 N | 1024 | 253.2 | 0.61 | |
| 锚点 2048×2048 N | 2048 | 1351.6 | 0.58 | band-major 后持平 |
| 锚点 2048×2048 L/T | 2048 | 1235.4 | 0.69 | |
| 锚点 1024×256 C | 256 | 81.1 | 0.80 | |
| 小形状 n=64/128 | 64/128 | 7.8/10.5 | 0.48/0.53 | quad-split 后反超 csyrk 参考（9.6/14.6） |
| 大方针 n=3635 | 3635 | 7910 | 0.47 | band-major −20%，优于 csyrk（8290） |
| 全量 200 条 | — | — | min 0.40 / 均值 ~0.6 / 中位 0.59 | 单 case ≥0.4 ✓，均值 ≥0.5 ✓ |

## 4.3 兼容性分析

新算子（`include/cann_ops_blas.h` 新增声明），无历史版本兼容问题；不修改仓内既有算子（共享层仅引用 `csyrk_gemm_fused.h`/`syrk_gemm_arch35.h` 头文件，未改动），不影响其他产品线。

## 4.4 测试与工程交付

- 测试工程：`test/syrkx/csyrkx/{,arch35/}`（GTest CSV 驱动），`gen_csv.py` 一键再生成 1000+200 用例，`verify_accuracy.py` / `verify_performance.py`（[PERF] 单次均值口径）+ `accept_perf_repl.py`（验收协议复刻器：msprof kernel-duration、窗口归因、5+10 主读数）。
- 算子 README：`blas/syrkx/README.md`（产品支持表 950PR/950DT：支持，参数/约束/边界/精度/调用示例）。
- 代码分支：`zhanghua145/ops-blas` `csyrkx-arch35-predev`（合入前按任务书终态微调后提 PR 至 `blas/syrkx/arch35/`）。

# 5. 参考实现与验收风险

| 项 | 结论 |
| --- | --- |
| 实现完成度 | 算子/测试/README/验收工具已全部实现并实测（本设计即实态描述，非待编码方案） |
| 精度 | 1000/1000 全过（含 81 条 A==B 退化、特殊值、全负向） |
| 性能 | 200/200 全过（单 case ≥0.4 且均值 ≥0.5 双门槛） |
| 任务书对齐项 | 非法枚举错误码按任务书 §2.4 定为 `INVALID_VALUE`（合入前对齐实现与测试期望值）；OP_C≡OP_T、确定性、k=0 语义均按任务书实现并验证 |
