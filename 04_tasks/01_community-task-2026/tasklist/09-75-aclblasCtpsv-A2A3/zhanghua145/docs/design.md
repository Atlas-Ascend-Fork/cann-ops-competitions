# aclblasCtpsv 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasCtpsv_A2A3 任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-blas （`blas/tpsv/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 9 月社区任务「aclblasCtpsv 算子开发(A2A3)」。在 `ops-blas` 仓为 Atlas A2/A3（`arch22`，覆盖 910B3/910B4/910_93 系列，性能测试设备 Atlas 800T A2 (910B3)）实现单精度复数（COMPLEX64）三角打包求解算子 `aclblasCtpsv`：解 `op(A)·x = b`，右端 b 初始存放于 x，解原地写回 x。

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasCtpsv_A2A3_task_doc.md` | 功能、接口、硬件、精度/性能门槛与交付要求 |
| `test_cases/`（tpsv_test.csv、gpu_baseline.csv、gen_csv.py、verify_accuracy.py、README） | 用例运行方式、golden、性能统计口径 |
| ops-blas 仓 master | 工程框架、头文件、同族实现与测试组织 |

## 1.2 背景介绍

### 算子语义：串行依赖链上的连续访存

`ctpsv` 为 BLAS Level-2 三角求解（packed 存储）。A 为 n×n 三角矩阵，按列主序打包于一维数组 AP（共 `n(n+1)/2` 个 complex64，无 lda）：

- `UPPER`：`A(i,j)（j ≥ i）` 存于 `AP[i + j(j+1)/2]`（0 基，下同）；
- `LOWER`：`A(i,j)（j ≤ i）` 存于 `AP[i + (2n-j-1)j/2]`。

**关键结构性质：无论 UPPER/LOWER，A 的每一列在 AP 中物理连续**（UPPER 列 j 占 `AP[j(j+1)/2 .. j(j+1)/2+j]`；LOWER 列 j 占列首 `j(2n-j-1)/2` 起的 `n-j` 个元素）。这决定了 kernel 的两条等价求解形态都只做连续读：

- **右视（AXPY 型）**，适配 `OP_N`：按列扫过三角，先解出当前列对角元 x_k，再用该列剩余元素批量更新未解分量；
- **左视（DOT 型）**，适配 `OP_T/OP_C`：转置/共轭转置求解等价于"逐行对已解前缀做点积"，而该前缀恰为 AP 中某列的连续段。

求解存在逐列（或逐行）的循环携带依赖，单右端项场景下并行度天然受限；本算子为延迟/依赖链敏感型，而非带宽敏感型（详见 §3.2.5 性能反算）。

### ops-blas 现状分析

基于 ops-blas master 实测核对：

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasCtpsv` 声明**已存在**（与 950PR/arch35 实现共用；另有实数版 `aclblasStpsv`） | 不改头文件签名，禁止私有平行接口 |
| `blas/tpsv/arch35/` | 已有 ctpsv 全套实现（SIMT 编程模型：`ctpsv_kernel_simt.cpp` 等） | 语义与测试对齐参考；arch22 不照搬 |
| `blas/tpsv/arch22/` | 仅有实数版 `stpsv_*`（标准 Ascend C：`stpsv_host.cpp / stpsv_kernel.cpp / stpsv_kernel.h / stpsv_tiling_data.h`） | **直接工程模板**：其单核"x 常驻 UB + 列扫描 + AxpyWindow/DotWindow 双形态"骨架可整体沿用 |
| `test/tpsv/ctpsv/` | 家族级 `ctpsv_param.h / ctpsv_golden.h` 已存在，仅有 `arch35/` 测试 | 测试落点 `test/tpsv/ctpsv/arch22/`，复用家族参数映射与 golden 组织 |

arch22 工程先例要点（`stpsv_tiling_data.h` / `stpsv_kernel.cpp` 实测）：

- 整个解向量 x 常驻 UB，使单核依赖链最短，`ACLBLAS_STPSV_MAX_N = 32768`（FP32，x 占 `4n` 字节 + ~32KB 搬运/归约暂存）；
- 单核启动 `stpsv_kernel<<<1, nullptr, stream>>>`，按列 `ColumnGeom` 计算打包偏移，`AxpyWindow`（N 型）与 `DotWindow`（T/C 型）两条路径；
- 超上限 n 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，非法枚举/参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，handle 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

### 功能缺口

**arch22 复数版整体缺失**：无 host、无 kernel、无 tiling、无 arch22 测试。相对 stpsv 的差异点即本设计的全部工作量：

1. 数据宽度 4B → 8B（complex64 实虚交错），UB 容量约束下的 n 上限需重估（§3.2.2）；
2. 标量乘减 → 复数乘减（4 乘 2 加/元素）与复数除法（对角归一）；
3. `OP_C` 需对 A 元素取共轭（T/C 两型在复数下不再等价，必须分路径）；
4. `incx` 为负/非 1 时的 gather/scatter 语义。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/tpsv/arch22/` 以标准 Ascend C 实现 `aclblasCtpsv`（kernel 直调、handle 绑定 stream），能力达到：COMPLEX64 × {UPPER, LOWER} × {OP_N, OP_T, OP_C} × {NON_UNIT, UNIT} 全 12 组枚举，`incx` 非零（含负），`n = 0` 合法 no-op，`n` 覆盖精度段 [1, 2048] 与性能段 [1, 4096]；精度满足混合容差（COMPLEX64：rtol = atol = 2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP）；5 条锚点性能 case 达标（910B3，Avg time ≤ GPU 实测 / 0.8）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | COMPLEX64 输入输出，解 `op(A)·x = b`，原地写回 x | 任务书 §2.1 | 实虚分离的向量计算，x 常驻 UB |
| R02 | 12 组枚举全组合 | 任务书 §2.1 | Host 统一分发 + kernel 双形态（N 型 AXPY / T·C 型 DOT，C 型共轭） |
| R03 | packed 列主序下标映射（UPPER/LOWER 两公式） | 任务书 §2.1 | `ColumnGeom` 列几何（沿用 stpsv 骨架） |
| R04 | `incx ≠ 0`，含负步长 | 任务书 §2.1 | 装载/回写阶段按 `XPhys(i)` gather/scatter，主循环恒按连续 x 工作 |
| R05 | `n = 0` no-op；非法参数/枚举返回 INVALID_VALUE；handle 空 HANDLE_IS_NULLPTR | 任务书 §2.4/§2.5 | Host 校验层（§3.2.1） |
| R06 | UNIT 对角不读取对角元 | 任务书 §2.1 | 对角分支跳过除法且不触碰 AP 对角位 |
| R07 | 泛化能力（不限于锚点 shape） | 任务书 §2.1/§3.5 | 按列 tile 流式处理 + 尾块处理，任意合法 n |
| R08 | 精度：混合容差 | 任务书 §3.2 | §4.1 双口径验证，Netlib ctpsv golden |
| R09 | 性能：5 锚点 ≤ GPU/0.8 | 任务书 §3.3 | §3.2.5 反算：依赖链与带宽双维度均留余量 |
| R10 | 异步执行（handle 携带 stream） | 任务书 §2.5 | `stpsv_kernel_do(..., stream)` 直调模式，内部不做隐式同步 |
| R11 | 工程合规：文件结构对齐仓内同族 | 任务书 §5 | §3.2.0 文件布局 |
| R12 | 可测可复现 | 任务书 §3.5 | CSV 驱动 GTest + 固定种子 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `A = (aᵢⱼ)`，`x = (xᵢ)`，`op(A)` 由 trans 决定。求解等价于四种递推（0 基下标）：

**LOWER + OP_N（前代，右视列扫描）**：对 k = 0..n−1

```text
xk = UNIT ? xk : xk / akk
x[i] -= a[i,k] · xk    （i = k+1 .. n-1，列 k 的 LOWER 连续段）
```

**UPPER + OP_N（回代，右视列扫描）**：对 k = n−1..0，同上形式（更新 i = 0..k−1）。

**LOWER + OP_T / OP_C（点积型回代）**：对 i = n−1..0

```text
x[i] = ( b[i] − Σ_{j>i} ô[j,i] · x[j] ) / ô[i,i]，  ô = A（OP_T）或 conj(A)（OP_C）
```

求和项 `ô[j,i]（j > i）` 为 LOWER 打包中列 i 的连续段。UPPER + OP_T/OP_C 为前代方向的点积型（`j < i`，列 j 上段连续）。

**复数基础运算**（实虚分离，w = a·u）：

```text
Re(w) = ar·ur − ai·ui     Im(w) = ar·ui + ai·ur          （4 mul + 2 add）
复数除 x/a：d = ar² + ai²；Re(x/a) = (xr·ar + xi·ai)/d；Im(x/a) = (xi·ar − xr·ai)/d
```

OP_C 的共轭在 A 列 tile 装入 UB 后对虚部 lane 取负实现（交错布局下按奇偶 lane 选择性取反）。

### 3.1.2 支持数据类型

仅 COMPLEX64（实部/虚部各 FP32，`aclblasComplex` 定义见 `include/cann_ops_blas_common.h`）。

### 3.1.3 支持形状

`n ≥ 0`；AP 长度 `n(n+1)/2`；x 逻辑长度 n、物理长度 `1 + (n−1)·|incx|`。n 上限受 UB 约束见 §3.2.2（超出返回 `NOT_SUPPORTED`，任务用例最大 n = 4096，上限留 ≥ 4× 余量）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内同族）

```text
blas/tpsv/arch22/
├── ctpsv_host.cpp           # 接口校验 + tiling 组装 + kernel 启动
├── ctpsv_kernel.h           # ctpsv_kernel_do 声明
├── ctpsv_kernel.cpp         # Ascend C kernel（N 型 / T·C 型双路径）
└── ctpsv_tiling_data.h      # CtpsvTilingData（Host/Kernel 共享）
test/tpsv/ctpsv/arch22/
├── ctpsv_npu_wrapper.h      # stream 绑定与同步封装
├── ctpsv_test.cpp           # CSV 驱动 GTest
└── ctpsv_test.csv           # 任务包 1200 条用例
```

### 3.2.1 Host 侧设计

校验层（任何 kernel 启动前完成，顺序即返回优先级）：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | `HANDLE_IS_NULLPTR`（n=0 亦先判，对齐 stpsv arch22 口径） |
| 2 | n < 0 | `INVALID_VALUE` |
| 3 | uplo ∉ {121,122} / trans ∉ {111,112,113} / diag ∉ {131,132} | `INVALID_VALUE` |
| 4 | n > 0 且（AP == nullptr 或 x == nullptr） | `INVALID_VALUE` |
| 5 | incx == 0 | `INVALID_VALUE` |
| 6 | n > N_MAX | `NOT_SUPPORTED` |
| 7 | n == 0 | `SUCCESS`（不发射 kernel，AP/x 允许空） |

通过后组装 `CtpsvTilingData`（与 `StpsvTilingData` 同构，仅指针宽度含义为 complex64 元素）：

```cpp
struct CtpsvTilingData {
    uint64_t ap;    // packed 三角矩阵 Device 指针（n(n+1)/2 个 complex64）
    uint64_t x;     // 右端/解向量 Device 指针
    uint32_t n;     // 阶数
    uint32_t uplo;  // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint32_t trans; // ACLBLAS_OP_N / OP_T / OP_C
    uint32_t diag;  // ACLBLAS_NON_UNIT / ACLBLAS_UNIT
    int64_t  incx;  // x 步长（非零，可负）
};
```

启动：`ctpsv_kernel_do(tiling, handle->stream)`，单核 `<<<1, nullptr, stream>>>`。Host 不做任何 Device 回读，不引入隐式同步（R10）。

### 3.2.2 Kernel 侧设计（本任务技术主体）

**总体：单核、x 常驻 UB、按列 tile 流式扫描。** 沿用 stpsv arch22 的"常驻 + 列几何 + 双形态"骨架，向 complex64 扩展。

#### UB 预算与 n 上限

arch22 单 AIV 可用 UB 约 192KB。占用：x 工作区 `8n` B + AP 列 tile 双缓冲 `2 × tileB` + 复数乘减的实虚拆分临时 ≤ `2 × tileB` + 归约暂存 ~4KB。取 `tile = 1024` 个 complex64（8KB/tile，32 对齐）：

```text
8n + 4×8KB + 4KB ≤ 192KB   ⇒   N_MAX = 16384（x 占 128KB，留 56KB tile/暂存）
```

`ACLBLAS_CTPSV_MAX_N = 16384`（超限 `NOT_SUPPORTED`；任务最大 n = 4096，余量 4×）。

#### 主流程

```text
Init:    TPipe 装配 xRes(u8n) / aT0,aT1(u8·tile) / tmp(u2·tile) / red(4KB)
Phase1:  LoadXResident —— 按 XPhys(i) = (incx>0 ? i : n-1-i) · |incx| 从 GM gather x 到 UB 连续工作区
Phase2:  列扫描主循环（方向与形态由 uplo×trans 决定，见下）
Phase3:  StoreXResident —— 按 XPhys 散射回 GM（原地覆写）
```

#### 两条计算形态

**N 型（OP_N，右视 AXPY）**，k 沿求解方向步进：

```text
对每列 k：
  ① 列几何 g = ComputeColumnGeom(k)：本列在 AP 中的首偏移与有效长度
  ② 分段 DataCopy 列数据 → UB tile（双缓冲：本段计算与下段搬运重叠）
  ③ xk = g.diag 段中的对角元；NON_UNIT 时 x[k] = CDiv(x[k], akk)（§3.1.1 复数除）
  ④ 尾段更新 AxpyWindow：x[i] -= CMul(a[i], xk)，i 为本列未解连续段
     —— complex64 交错布局下：把 tile 视作 2N 个 fp32，实/虚 lane 分别算
        Re: xr[i] − (ar·wr − ai·wi)；Im: xi[i] − (ar·wi + ai·wr)（w = xk 拆出的两个标量）
```

**T·C 型（OP_T/OP_C，左视 DOT）**，i 沿求解方向步进：

```text
对每个 i：
  ① 取列 i 的已解前缀连续段（长度 L 随 i 增长/缩短），分段 DataCopy → UB tile
  ② OP_C 时对虚部 lane 取负（装入后一次向量 Sub(0, ai) 完成）
  ③ DotWindow：d = Σ CMul(a[j], x[j]) —— 向量内积分实虚两路：
        dRe = Σ (ar·xr − ai·xi)；dIm = Σ (ar·xi + ai·xr)
     段内用向量 Mul + ReduceSum 累加，段间标量累加，UB 内完成（无 GM 归约）
  ④ x[i] = (b[i] − d)；NON_UNIT 时再除 a[i,i]（列首元素）
```

**Tiny 路径**：n ≤ 32 时 AP 整体一次搬入（≤ `32×33/2×8B ≈ 4.2KB`），纯 UB 内双层标量循环完成，规避小 n 的搬运/tile 开销（对应锚点 case1 n=256 的启动开销敏感区下界）。

**尾块与对齐**：列 tile 按 32B 对齐取整搬运、以掩码/长度参数约束有效计算；`n(n+1)/2` 无需对齐假设，GM→UB 搬运按实际字节数。

### 3.2.3 调用与执行流程图

```text
Host                                          Device (单 AIV)
────                                          ─────────────
aclblasCtpsv(...)
  ├─ 校验表 §3.2.1 ─→ 非法即返
  ├─ n==0 ─→ SUCCESS（不发射）
  ├─ CtpsvTilingData 组装
  └─ ctpsv_kernel_do(t, stream) ──────────→  ctpsv_kernel
                                                ├─ Init(TPipe/Que)
                                                ├─ LoadXResident（gather by incx）
                                                ├─ up to 8 path dispatch:
                                                │    LOWER·N 前代 AXPY ┐
                                                │    UPPER·N 回代 AXPY │ 列扫描
                                                │    LOWER·T/C 点积回代 ├ 双缓冲 tile
                                                │    UPPER·T/C 点积前代 ┘
                                                │    （n≤32 走 Tiny）
                                                └─ StoreXResident（scatter by incx）
```

### 3.2.4 关键技术点与风险

| # | 点 | 处置 |
| --- | --- | --- |
| 1 | 复数除法（对角归一）的精度与除零 | `d = ar²+ai²` 为 0 时语义未定义（任务书不做奇异检测，与 cuBLAS 一致）；除法按 §3.1.1 公式 fp32 实现，误差 ≤ 2 ULP/分量 |
| 2 | OP_C 共轭的实现位置 | 装入 UB 后、计算前对虚部 lane 一次向量取反，避免在点积内逐元素分支 |
| 3 | 交错布局的向量计算效率 | 实虚分别形成连续 fp32 lane 后用标准向量 Mul/Sub/ReduceSum；不做逐元素标量循环 |
| 4 | 负步长 incx | 仅在 Phase1/Phase3 做 gather/scatter，主循环恒连续，T·C 点积与 N 型不受影响 |
| 5 | 大 n 的 UB 常驻 | N_MAX=16384 硬校验 + NOT_SUPPORTED；超限场景不在本任务用例域（≤4096） |
| 6 | 与 arch35 实现的行为对齐 | 以 test_cases 1200 条 CSV 为共同回归集（arch35 已通过同一 CSV 族） |

### 3.2.5 性能设计反算（R09）

锚点最大 case：n = 4096，AP 有效读 `4096×4097/2 × 8B ≈ 67.1MB`，达标线 7327µs ⇒ 有效带宽 ≥ **9.2GB/s**；GPU（A100 级）实测 5.86ms ⇒ 11.5GB/s，同为依赖链受限量级。arch22 单 AIV GM 顺序读带宽（数十 GB/s 量级）对 9.2GB/s 留有充分余量；瓶颈在列间串行 launch-free 依赖链长度（n 次"除法 + 段更新"），Tiny/小 n case（252µs@n=256）由固定开销主导，单 kernel 单发射结构使该开销最小化。达标结论：结构上无反例，最终以 msprof 实测为准（§4.2）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3/910B4） | √ |
| Atlas A3 系列（910_93 等，arch22 同目录） | √ |

## 3.4 算子约束限制

- n ∈ [0, 16384]，超限返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（UB 常驻约束，任务用例域 n ≤ 4096）；
- 不做奇异性/近奇异性检测（对齐 cuBLAS/Netlib），NON_UNIT 时调用方保证对角非零；
- AP 与 x 不允许内存重叠；不支持超出 incx 语义的非连续访问；无广播、无 dynamic shape 声明、无确定性计算承诺；
- UNIT 对角时不读取 AP 对角位置（golden 同口径）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差：逐元素 `|actual−golden| ≤ 2⁻¹³ + 2⁻¹³·|golden|`（COMPLEX64，实/虚分量），matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP（含规约，可放宽 2 ULP） | 任务书 §3.2；opbase mixed_tolerance_standard |
| 性能标准 | 5 锚点 Avg time ≤ 达标线（GPU/0.8）：252.2 / 573.9 / 1228 / 1211 / 7327 µs（n=256/512/1024/2048/4096）；msprof op 口径，warmup 后有效采样 ≥ 10 次均值 | 任务书 §3.3；gpu_baseline.csv |

## 4.2 测试方案

- **用例**：任务包 1200 条（1000 精度 + 200 TC_PF 性能；12 组枚举 × 尺寸扫描 1→2048/4096 × incx ±1/±2/±3 × 填充/Inf/NaN × 边界负向），固定种子可复现；
- **golden**：cblas（Netlib `ctpsv`）单标杆；测试工程填充 AP 后对被引用对角加 `boost = max(5, n)`（实部）保证非奇异，golden 用同一偏移后矩阵回代，两侧输入严格一致；
- **执行**：`verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv tpsv_test.csv` 编译并运行 GTest，逐条 PASS/FAIL；性能用例借 `msprof op` 采集 `OpBasicInfo.csv` 的 Task Duration 与 `gpu_baseline.csv` 逐条关联（NPU ≤ gpu_ms/0.8）；
- **覆盖核查**：L0 基础 12 组全枚举、L1 尺寸扫描（含奇数/边界/非对齐）、L2 步长（含负）、L5 填充/Inf/NaN、L6 边界负向（零维/空指针/非法枚举 999/负维度/零步长）全覆盖，缺失场景按任务书 §3.5 自行补充。

## 4.3 兼容性分析

新增 arch22 实现与测试，不改头文件（`aclblasCtpsv` 声明已存在并与 arch35 共用）、不改 arch35 既有文件、不动构建脚本（CMake 按 arch 目录 glob 自动纳入）。对上层无行为变更；A2 与 A3 同 arch22 目录，README 产品支持表统一标注。

# 5. 参考链接

1. 任务书与测试包：`aclblasCtpsv_A2A3_task_doc.md`、`test_cases/README.md`；
2. ops-blas 仓：https://gitcode.com/cann/ops-blas （`blas/tpsv/arch22/stpsv_*` 工程先例、`include/cann_ops_blas_common.h`）；
3. Netlib BLAS ctpsv：https://www.netlib.org/blas/ctpsv.f ；
4. cuBLAS cublasCtpsv：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tpsv ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md 。
