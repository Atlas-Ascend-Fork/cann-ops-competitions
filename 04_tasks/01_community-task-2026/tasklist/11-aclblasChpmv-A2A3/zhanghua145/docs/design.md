# aclblasChpmv 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasChpmv_AtlasA2_A3_task_doc.md》（2026-11-25 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/spmv/arch22/`，测试 `test/spmv/chpmv/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasChpmv算子开发（A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数 Hermitian 压缩存储矩阵-向量乘 `aclblasChpmv`
（对标 cuBLAS `cublasChpmv`，语义参考 Netlib `chpmv`）：

```text
y = alpha · A · x + beta · y        （complex64）
A：n×n Hermitian 复数矩阵以 packed 格式存放于 AP：
  uplo=UPPER：列 j 的 A(i,j)（i ≤ j）存于 AP[i + j*(j+1)/2]
  uplo=LOWER：列 j 的 A(i,j)（i ≥ j）存于 AP[i + j*(2*n-j-1)/2]
共 n(n+1)/2 个复数元素，无前导维度参数；对角虚部不被引用（实现与 golden 均只取对角实部）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasChpmv_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（chpmv_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasChpmv` 声明**已存在**（约 L161） | 无需新增声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/spmv/arch22/` | 实数参考算子（host/kernel/tiling 三件套组织） | 本任务实现落点（Chpmv 落位 spmv 目录，任务书 §1/§5）；host 校验框架与 tiling 传递模式沿用 |
| `blas/spmv/arch35/` | **已有 `aclblasChpmv` 950PR 实现** | packed 寻址（三角下标闭式）、uplo 分支、对角虚部忽略的实现参照（任务书 §2.2；不得直接复制 SDK 不兼容代码） |
| `test/spmv/` | 仅有既有目标（chpmv 子目录需新建 arch22 分支） | 测试工程新建 `test/spmv/chpmv/arch22/`（任务书 §1） |

### 功能缺口与本任务差异点

**arch22 Chpmv 缺失**。相对实数 spmv（arch22）与 arch35 Chpmv 的差异即主要工作量：

1. **packed 三角寻址**：AP 无前导维度——列起偏移 `j*(j+1)/2`（UPPER）/
   `j*(2n−j−1)/2`（LOWER）为非线性递推，多核切分需列段累积偏移表（避免逐列除法）；
2. **Hermitian 语义**：对角虚部不读（只取实部）；列点积带共轭（dot 项）、
   行散射无共轭（scatter 项，Netlib chpmv 精确语义）；
3. **复数 alpha/beta 交叉缩放** + interleaved complex64 deinterleave 平面化；
4. **n=0 不豁免结构校验**（非法 uplo/incx/incy/alpha/beta 仍按 §2.4 返回对应错误，
   与 cuBLAS 一致——任务书 §2.5 校验顺序明确 quick return 在校验之后）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/spmv/arch22/` 实现 `aclblasChpmv`：uplo ∈ {UPPER, LOWER} × n 覆盖
0/1/2/3/5/7/16…4095/4096/**8191/8192** × incx/incy（±1/±2/±3 + 大步长 {7,8,16,31,64,127} 抽样；
性能子集以单位步长为主）× alpha/beta 特殊值 (0,0)/(1,0)/(0,1) + 随机 × Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准 FLOAT32 档（实部/虚部分别判定：rtol = atol = 2⁻¹³，
matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP）；golden 为 Netlib CBLAS 3.10.0
`cblas_chpmv`（CblasColMajor）单标杆。性能 msprof op 模式 task duration，
5 次 warmup + 10 次有效采样均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `y = α·A·x + β·y`（packed Hermitian 三角引用） | §2.1 | 列段组批 kernel（§3.2.2） |
| R02 | uplo 非法 → `INVALID_ENUM`；校验顺序 handle→n→uplo→incx/incy→标量→**n=0 quick return**→指针 | §2.5 | Host 校验链（§3.2.1） |
| R03 | n=0 不豁免结构校验（与 cuBLAS 一致） | §2.5 | quick return 前置全量校验（§3.2.1） |
| R04 | α=(0,0) 且 β=(1,0) quick return；α=(0,0) 只做 β 缩放；β=(0,0) 不读 y | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R05 | packed 三角闭式寻址（UPPER/LOWER 两种列起偏移） | §2.1 | 列段累积偏移表（§3.2.2） |
| R06 | 对角虚部不读；未引用三角不读 | §2.1/§2.5 | 对角实部标量捕获 + 装载裁剪（§3.2.2） |
| R07 | 负步长语义（逻辑↔物理反转） | §2.1 | X/YPhysPos 映射（§3.2.2） |
| R08 | 精度 FLOAT32 档 + Inf/NaN 等效 | §3.2 | 测试工程（§4.2） |
| R09 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

0 基列主序 packed（Netlib chpmv 精确语义）。UPPER（存 A(i,j), i ≤ j）逐列 j：

```text
temp1 = alpha · x(j)
i < j:  y(i) += temp1 · A(i,j)          # scatter 项，无共轭
        temp2 += conj(A(i,j)) · x(i)    # dot 项，带共轭
y(j) += temp1 · real(A(j,j)) + alpha · temp2
```

LOWER（存 A(i,j), i ≥ j）对称地按 i > j 展开。等价的平面化形式（本实现）：

```text
dot(j)  = Σ_{i∈三角(j)} conj(A(i,j))·x(i)            # 列点积（带共轭，除对角）
t1(i)   = Σ_{j∈三角(i)} A(i,j)·x(j)                   # 行散射（无共轭，除对角）
y(j)    = alpha·(x(j)·real(A(j,j)) + dot(j)) + alpha·t1(j) + beta·y(j)
```

packed 列起偏移：UPPER 列 j 起点 `j*(j+1)/2`（列长 j+1）；LOWER 列 j 起点
`j*(2n−j−1)/2`（列长 n−j）。对角元素 UPPER 在列首（offset 起点）、LOWER 在列尾。

### 3.1.2 支持数据类型

COMPLEX64（interleaved float32 re/im，`aclblasComplex`），alpha/beta 同型；
输出原地覆写 y。AP 为 n(n+1)/2 元素压缩向量；x/y 物理元素数 ≥ 1+(n−1)·|inc|。
n ≤ 8192（任务书 §3.5 覆盖档）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/spmv/arch22/
├── chpmv_host.cpp           # aclblasChpmv 入口：校验链/标量读取/tiling/发射
├── chpmv_kernel.cpp         # Ascend C AIV kernel（列段组批引擎）
├── chpmv_kernel.h
└── chpmv_tiling_data.h      # tiling 结构 + 列段参数 + 偏移表闭式
test/spmv/chpmv/arch22/
├── chpmv_test.cpp           # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── chpmv_test.csv           # 官方 1000 精度 + 200 性能
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（约 L161），无需修改
test/spmv/chpmv/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序，n=0 不豁免结构校验）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ n<0（`INVALID_VALUE`）→ uplo 非法
   （`INVALID_ENUM`）→ incx=0 / incy=0（`INVALID_VALUE`）→ alpha/beta 指针判空
   （`INVALID_VALUE`）→ **n=0 quick return SUCCESS**（不引用 AP/x/y、不启动 kernel）→
   AP/x/y 指针判空（n>0 时）→ 读标量后 α=(0,0) 且 β=(1,0) quick return SUCCESS。
2. **标量读取**：alpha/beta 为 Host 内存 complex64 标量（任务书 §2.4），直读实/虚部；
   α=(0,0)/β=(0,0) 短路标志下传 tiling。
3. **快速分支**：α=(0,0) 时跳过 AP/x 装载，仅 β 重写 y；β=(1,0) 且 α=(0,0) 直接 SUCCESS。
4. **tiling 组装**：n/uplo/incx/incy/alpha/beta 及其 0/1 标志 + coreNum
   （ChooseCoreNum：strided 或 β 重写 → K=1 单核；列段数 ≤ 32 → K=段数；
   否则按 chunk 均衡 K ≤ 40）+ 列段偏移表参数（§3.2.2）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（列段组批引擎，本任务技术主体）

AIV 向量引擎（无 Cube 路径——Level-2 BLAS packed 访存受限，Cube 无收益）：

- **列段偏移表**：Host 侧按等差闭式预生成列段起偏移（UPPER：`seg(k)` 列段首
  `j0*(j0+1)/2`、段内列步进 1 递推；LOWER 对称），kernel 直接查表装载——
  规避 packed 非线性下标的逐列乘除；
- **组批参数**：B=8 列/组、R=512 行/窗、E=B×R=4096 元素平面；
  每组 chunk 的 AP 三角段按列连续读入 UB ping-pong 暂存（inter×2=64KB）；
- **deinterleave**：interleaved complex64 经 Gather 索引（byte offset 8m / 8m+4）
  一次拆为 re/im 平面，后续乘累加全为平坦 E 元素向量 pass；
- **列点积（带共轭）**：`d = conj(A)·x窗`（4 Mul + 2 Add/Sub 平面 pass）→
  ReduceSum 按列求和（紧凑 stage 槽布局规避 dst 槽间隙垃圾）→ 累加进 per-group
  stage 槽（跨窗口持久）；
- **行散射（无共轭）**：x 列广播 × A 平面 → 8 槽树状归约 → 累加进 per-window
  win 累加器；
- **对角**：每列在其边沿窗口内一次标量捕获 `real(A(j,j))`（UPPER 列首/LOWER 列尾）
  后置零平面，fold 阶段补 `x(j)·d(j)` 项；对角虚部不读（装载裁剪）；
- **回写**：每窗一次 α 折叠 + interleave + **MTE3 原子加** flush 到 y；
  每组一次 `alpha·(x·d + dot)` 折叠原子加到 y[c0..c0+8)。多核 β 重写走 MTE3
  向量路径（标量 GM 写对原子加不可见）；
- **chunk 均衡**：列段 chunk 闭式累积（UPPER/LOWER 各自）+ 二分，
  各核分得近似等量 (窗口,组) chunk，等时化墙钟；
- **strided/单核路径**（K=1 门控）：每 chunk 点积直接标量 RMW 折进 y，
  β 基底先铺；x/y 步长访问用标量 gather/scatter，AP 读取仍向量化；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
  Set/Wait 相邻配对、id 7 不可用、等待对不足补 PipeBarrier；
- **UB 预算**：~165KB / 174KB 实测上限（inter 64K + planes 32K + xrow 32K +
  win/stage/idx 44K）。

### 3.2.3 调用与执行流程

```text
aclblasChpmv ─校验链（n=0 前全量）→ 标量直读 → quick-return 判定
  → ChooseCoreNum → 列段偏移表 tiling 组装 → chpmv_kernel<<<K, stream>>>
       └─ AIV: Init(偏移表/清零) → [β 重写(MTE3)+SyncAll] → 窗口循环
            { x 窗装载驻留 → 组循环{ping-pong 载入→列点积/行散射} → win flush 原子加 }
          → FoldGroups(α·(x·d+dot) 逐组原子加) → 完成
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| n=8192 带宽天花板（packed 8192²/2×8B=268MB 流读，HBM 1.44TB/s → 底线 ~187us） | AP 每元素仅读一次的列段连续流式装载；原子回写/同步开销压到 ≤8us |
| packed 非线性下标多核寻址 | Host 侧列段偏移表（等差闭式预生成）+ kernel 查表 |
| 复数原子加多核可见性 | β 重写走 MTE3 + 全核屏障；win/stage 双累加器解耦跨窗/跨组序 |
| 对角捕获（UPPER 列首/LOWER 列尾位置差异） | 边沿窗口标量捕获单点维护 + LOWER straggler 清零 |
| strided/负步长性能 | 门控 K=1 保精度（性能用例全为 inc=1，无代价） |

### 3.2.5 性能设计反算（R09）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 锚点：

| case | uplo | n | incx/incy | H100 gpu_ms | 达标线(ms) |
| --- | --- | --- | --- | --- | --- |
| chpmv-base-004 | UPPER | 8192 | 1/1 | 1.527 | 1.909 |
| chpmv-perf-076 | LOWER | 8192 | 1/1 | 1.486 | 1.858 |

n=8192 大档：AP 268MB + x/y（128KB×2）→ HBM 1.44TB/s 流读底线 ~187us，
对 1909us 达标线余量 ~10×（H100 cuBLAS chpmv 本身非带宽最优——GPU 大档耗时
1.5ms 远超其 HBM 底线）；NPU 侧按「AP 单遍流读 + 组批向量乘累加」即可覆盖全档；
小 n 档以固定开销（init 链 + 发射 ≤ 8us）为下限约束。采集口径：msprof op task
duration，5 warmup + 10 采样均值；实测按全量 200 条回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立列段 slab（chunk 均衡）；UB 165KB 布局见 §3.2.2；
无 workspace/GM 中转（除原子回写）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0；incx/incy ≠ 0；COMPLEX64 专用；AP 列主序无间隙 packed（仅 uplo 三角被引用）；
对角虚部不读；y 原地覆写；异步执行依赖 handle 绑定 stream；
n=0 不豁免结构校验（与 cuBLAS 一致）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP），
  实部/虚部分别判定；Inf/NaN 等效规则；golden = Netlib CBLAS 3.10.0
  `cblas_chpmv`（CblasColMajor）单标杆（任务书 §3.2）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 chpmv_test.csv 全量接入（1000 精度 + 200 性能；uplo 全覆盖、
   n 全档含 8191/8192、incx/incy ±1/±2/±3 + 大步长抽样（性能以单位步长为主）、
   alpha/beta 特殊值 (0,0)/(1,0)/(0,1)、AP 按 Hermitian packed 填充（对角虚部置 0）、
   Inf/NaN 特殊值、负向用例（nullptr handle/非法 uplo/负 n/零步长/空标量/
   n=0 叠非法结构参数——期望返回对应结构错误不因 n=0 豁免/
   对角虚部非零输入（验证输入虚部被忽略）/α=(0,0) 叠 x=NaN/β=(0,0) 叠 y=NaN））；
2. **工程**：`test/spmv/chpmv/arch22/chpmv_test.cpp` CSV 驱动 GTest；
   `test/spmv/chpmv/CMakeLists.txt` 注册 arch22 目标
   （编译入口对齐仓内 `--ops=chpmv --soc=ascend910b3`，A3 对应 --soc）；
3. **精度回归**：1000 条全量 VerifyMixed（实/虚部分别统计）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建命令、gtest filter、msprof 命令、golden 构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 声明已存在不改（约 L161），与其他产品线共用（禁止私有平行 API）；
- arch22 与 spmv 族同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasChpmv_AtlasA2_A3_task_doc.md（2026-11-25）
2. Netlib chpmv：https://www.netlib.org/blas/chpmv.f
3. cuBLAS cublasChpmv：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
