# aclblasChpr2 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasChpr2_AtlasA2_A3_task_doc.md》（2026-11-25 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/spr2/arch22/`，测试 `test/spr2/chpr2/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasChpr2算子开发（A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数 Hermitian 压缩存储秩-2 更新 `aclblasChpr2`
（对标 cuBLAS `cublasChpr2`，语义参考 Netlib `chpr2`）：

```text
AP = alpha · x · y^H + conj(alpha) · y · x^H + AP        （complex64）
AP 按列主序逐列无间隙 packed，仅存 uplo 一半三角，共 n(n+1)/2 个复数元素，无前导维度：
  uplo=UPPER：列 j 的 A(i,j)（i ≤ j）存于 AP[i + j*(j+1)/2]
  uplo=LOWER：列 j 的 A(i,j)（i ≥ j）存于 AP[i + j*(2*n-j-1)/2]；对角 A(j,j) 存于 AP[j + j*(2*n-j-1)/2]
两项贡献互为共轭转置，求和后天然 Hermitian；对角虚部计算后强制置 0
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasChpr2_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（chpr2_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasChpr2` 声明**已存在**（约 L515） | 无需新增声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/spr2/arch22/` | 实数参考算子（sspr2：host/kernel/tiling 三件套组织） | 本任务实现落点（任务书 §1/§5）；host 校验框架与 tiling 传递模式沿用 |
| `blas/spr2/arch35/`、`blas/chpr2/arch35/`（仓内 Chpr2 950 实现所在族） | arch35 已有 `aclblasChpr2` 实现 | packed 三角寻址、uplo 分支、共轭秩-2 语义与对角置 0 的实现参照（任务书 §2.2；不得直接复制 SDK 不兼容代码） |
| `test/spr2/` | 既有 spr2 测试组织（chpr2 子目录需新建 arch22 分支） | 测试工程新建 `test/spr2/chpr2/arch22/`（任务书 §1） |

### 功能缺口与本任务差异点

**arch22 Chpr2 缺失**。相对实数 sspr2（arch22）的差异即主要工作量：

1. **复数秩-2 语义**：两项贡献 `α·x·y^H` 与 `conj(α)·y·x^H`（alpha 取共轭），
   off-diagonal 元素 `AP(i,j) += α·x(i)·conj(y(j)) + conj(α)·y(i)·conj(x(j))`；
   对角 `AP(j,j) += 2·Re(α·x(j)·conj(y(j)))`、虚部强制置 0（天然 Hermitian）；
2. **INT_MIN 步长守卫**：incx/incy = 0 **或 INT_MIN** 均返回 `INVALID_VALUE`
   （任务书 §2.4——`(n−1)×|inc|` 以 int64 计算的溢出防御口径，INT32_MIN 取 abs 会 UB）；
3. **packed 三角寻址**（UPPER/LOWER 两种列起偏移闭式）与多核列段偏移表；
4. **α=0 quick return 的指针校验前置**：α=0 时不读取 x/y 数值，但 x/y/AP 指针
   仍需非空（任务书 §2.1 第 6 条与 §2.5 校验顺序——指针判空先于 α=0 quick return）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/spr2/arch22/` 实现 `aclblasChpr2`：uplo ∈ {UPPER, LOWER} × n 覆盖
0/1/2/3/5/7/16…4095/4096 × incx/incy（±1/±2/±3 + 大步长 {7,8,16,31,64,127} 抽样；
性能子集以单位步长为主）× alpha 特殊值 (0,0)/(1,0)/(0,1) + 随机 ×
Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准 FLOAT32 档（实部/虚部分别判定：rtol = atol = 2⁻¹³，
matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP）；golden 为 Netlib CBLAS 3.10.0
`cblas_chpr2`（CblasColMajor）单标杆。性能 msprof op 模式 task duration，
5 次 warmup + 10 次有效采样均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `AP += α·x·y^H + conj(α)·y·x^H`（packed Hermitian） | §2.1 | 列段组批 kernel + 对角置 0（§3.2.2） |
| R02 | uplo 非法 → `INVALID_ENUM`；incx/incy = 0 或 INT_MIN → `INVALID_VALUE`；校验顺序 handle→n→alpha→uplo→inc（含 INT_MIN）→**n=0 quick return**→x/y/AP 判空→α=0 quick return | §2.5 | Host 校验链（§3.2.1） |
| R03 | α=0 不读 x/y 数值，但指针校验前置（x/y/AP 空指针仍 `INVALID_VALUE`） | §2.1/§2.4 | 指针判空先于 α quick return（§3.2.1） |
| R04 | n=0 不豁免结构校验（与 cuBLAS 一致） | §2.5 | quick return 前置全量校验（§3.2.1） |
| R05 | packed 三角闭式寻址（UPPER/LOWER 列起偏移） | §2.1 | 列段偏移表（§3.2.2） |
| R06 | 对角虚部计算后置 0；对角实部 `+= 2·Re(α·x·conj(y))` | §2.1 | 对角标量折叠（§3.2.2） |
| R07 | 负步长语义（逻辑↔物理反转） | §2.1 | X/YPhysPos 映射（§3.2.2） |
| R08 | 精度 FLOAT32 档 + Inf/NaN 等效 | §3.2 | 测试工程（§4.2） |
| R09 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

0 基列主序 packed。对 uplo 三角内的 (i,j)（UPPER：i ≤ j；LOWER：i ≥ j）：

```text
off-diagonal：AP(i,j) += αr·t − αi·u + i·(αr·u + αi·t)
  其中 t = x(i)·yr(j) + y(i)·xr(j)（实积和），u = x(i)·yi(j) + y(i)·xi(j)（交叉和）
  （即 α·[x(i)·conj(y(j))] + conj(α)·[y(i)·conj(x(j))] 展开的实/虚分组）
diagonal：   AP(j,j)_re += 2·(αr·(xr·yr + xi·yi) − αi·(xr·yi − xi·yr))
             AP(j,j)_im = 0（计算后强制置 0）
```

两项贡献互为共轭转置、求和天然 Hermitian；conjugate 作用于 y^H/x^H（外积的列侧共轭）。

### 3.1.2 支持数据类型

COMPLEX64（interleaved float32 re/im，`aclblasComplex`），alpha 同型；
AP 原地覆写（读旧值累加后写回）。AP 为 n(n+1)/2 元素压缩向量；
x/y 物理元素数 ≥ 1+(n−1)·|inc|。n ≤ 4096（任务书 §3.5 覆盖档）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/spr2/arch22/
├── chpr2_host.cpp           # aclblasChpr2 入口：校验链/标量读取/tiling/发射
├── chpr2_kernel.cpp         # Ascend C AIV kernel（列段组批引擎）
├── chpr2_kernel.h
└── chpr2_tiling_data.h      # tiling 结构 + 列段参数 + 偏移表闭式
test/spr2/chpr2/arch22/
├── chpr2_test.cpp           # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── chpr2_test.csv           # 官方 1000 精度 + 200 性能
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（约 L515），无需修改
test/spr2/chpr2/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：handle 判空（`HANDLE_IS_NULLPTR`）→
   n<0（`INVALID_VALUE`）→ alpha 指针判空（`INVALID_VALUE`）→ uplo 非法
   （`INVALID_ENUM`）→ incx=0 或 INT_MIN / incy=0 或 INT_MIN（`INVALID_VALUE`）→
   (n−1)×|inc| 以 host 侧 int64 计算并拒绝溢出 → **n=0 quick return SUCCESS**
   （不引用 AP/x/y、不启动 kernel）→ x/y/AP 指针判空（n>0 时）→
   读标量后 α=(0,0) quick return SUCCESS（**指针校验已前置**，R03）。
2. **标量读取**：alpha 为 Host 内存 complex64 标量，直读实/虚部（含 conj 预展开
   `conjαr=αr、conjαi=−αi` 下传 tiling）。
3. **快速分支**：α=(0,0) 时 AP 不变直接 SUCCESS（不读 x/y 数值，指针已校验）。
4. **tiling 组装**：n/uplo/incx/incy/α（实/虚 + conj 展开）/步长反转标志 +
   coreNum（ChooseCoreNum：strided → K=1 单核；列段数 ≤ 32 → K=段数；
   否则按 chunk 均衡 K ≤ 40）+ 列段偏移表参数（§3.2.2）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（列段组批引擎，本任务技术主体）

AIV 向量引擎（无 Cube 路径——packed Level-2 更新访存受限，Cube 无收益）：

- **列段偏移表**：Host 侧按等差闭式预生成列段起偏移（UPPER 列 j 起点
  `j*(j+1)/2`、列长 j+1；LOWER 列 j 起点 `j*(2n−j−1)/2`、列长 n−j，
  对角在列尾），kernel 查表装载——规避 packed 非线性下标逐列乘除；
- **组批参数**：B=8 列/组、R=512 行/窗、E=B×R=4096 元素平面；
  每组 chunk 的 AP 三角段按列连续读改写（UB ping-pong inter 暂存 64KB）；
- **deinterleave**：x/y 段与 AP 段 Gather 索引（byte offset 8m/8m+4）拆 re/im 平面，
  乘累加全为平坦向量 pass；
- **秩-2 更新主体**：x/y 两段窗装载驻留（各 32KB 平面）→
  外积平面 pass（t/u 分组实积：4 Mul + 2 Add/Sub）→ α/conjα 折叠
  （`αr·t−αi·u`、`αr·u+αi·t`）→ 与 AP 旧值累加 → interleave 写回同一段
  （原地：AP 段读-改-写一次完成，无原子——AP 三角段按列段划分天然不重叠）；
- **对角**：每列对角元素（UPPER 列首/LOWER 列尾）标量折叠
  `2·Re(α·x(j)·conj(y(j)))` 累加实部、虚部强制置 0（R06）；
- **多核划分**：AP 列段 chunk 闭式累积 + 二分均衡（列段间无写冲突——
  每元素恰属一个列段，天然免原子免锁）；
- **strided/单核路径**（K=1 门控）：x/y 步长访问用标量 gather（负步长反向映射
  X/YPhysPos），AP 读取仍向量化；性能用例以 inc=1 为主（任务书 §3.5），
  主档不受影响；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
  Set/Wait 相邻配对、id 7 不可用、等待对不足补 PipeBarrier；
- **UB 预算**：~170KB（AP inter ping-pong 64K + x/y 平面 4×16K + 索引/对角槽）。

### 3.2.3 调用与执行流程

```text
aclblasChpr2 ─校验链（n=0 前全量、指针先于 α=0）→ 标量直读（含 conj 展开）
  → α=0 quick return → ChooseCoreNum → 列段偏移表 tiling 组装
  → chpr2_kernel<<<K, stream>>>
       └─ AIV: Init(偏移表) → 窗口循环{ x/y 窗装载驻留
            → 组循环{AP 段 ping-pong 载入 → 外积平面 pass(t/u)
            → α/conjα 折叠 → +AP 旧值 → 对角折叠(置 0) → interleave 写回} }
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| packed 非线性下标多核寻址 | Host 侧列段偏移表（等差闭式预生成）+ kernel 查表 |
| INT_MIN 步长 abs UB | Host 侧显式等值拒绝（任务书 §2.4 口径）+ int64 跨度校验 |
| 复数秩-2 展开符号错误（conj 侧） | t/u 分组实积单点维护 + 非对称复数 x/y 专项对拍 golden |
| AP 原地读改写与多核写冲突 | 列段划分每元素恰属一段（免原子免锁）；ping-pong 段内一次完成 |
| 对角位置差异（UPPER 列首/LOWER 列尾） | 对角折叠单点维护（含置 0） |
| strided/负步长性能 | 门控 K=1 保精度（性能用例全为 inc=1，无代价） |

### 3.2.5 性能设计反算（R09）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 锚点：

| case | uplo | n | incx/incy | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- | --- |
| chpr2-base-003 | LOWER | 4096 | 1/1 | 0.8606 | 1076 |
| chpr2-perf-006 | UPPER | 4096 | 127/1 | 0.9267 | 1158 |

n=4096 大档：AP 4096²/2×8B=67MB 读 + 67MB 写 = 134MB@HBM 1.44TB/s → 流底线 ~93us，
加 x/y 装载与组批开销 ≤ 10us → 预算 ~105us，对 1076us 达标线余量 ~10×
（H100 cuBLAS chpr2 大档耗时 0.86ms 亦远超其带宽底线——packed 三角更新的 GPU 实现
非带宽最优）；NPU 侧按「AP 单遍流读改写 + 组批外积」即可覆盖全档；小 n 档以
固定开销（init 链 + 发射 ≤ 8us）为下限。采集口径：msprof op task duration，
5 warmup + 10 采样均值；实测按全量 200 条回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立列段 slab（chunk 均衡、免原子）；UB ~170KB 布局见 §3.2.2；
无 workspace/GM 中转。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0；incx/incy ≠ 0 且非 INT32_MIN（(n−1)×|inc| int64 不溢出）；COMPLEX64 专用；
AP 列主序无间隙 packed（仅 uplo 三角被读写）；对角虚部计算后置 0；
AP 原地覆写；α=0 时 AP 不变（指针校验前置）；异步执行依赖 handle 绑定 stream；
n=0 不豁免结构校验（与 cuBLAS 一致）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP），
  实部/虚部分别判定；Inf/NaN 等效规则；golden = Netlib CBLAS 3.10.0
  `cblas_chpr2`（CblasColMajor）单标杆（任务书 §3.2）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 chpr2_test.csv 全量接入（1000 精度 + 200 性能；uplo 全覆盖、
   n 全档、incx/incy ±1/±2/±3 + 大步长抽样（性能以单位步长为主）、
   alpha 特殊值 (0,0)/(1,0)/(0,1)、x/y/AP 均匀 [-5,5] 50% + 正态 N(0,1) 50%
   （AP 按 Hermitian packed 填充、对角虚部置 0）、Inf/NaN 特殊值、
   负向用例（nullptr handle/非法 uplo/负 n/**零步长/INT_MIN 步长**/空 alpha/
   n=0 叠非法结构参数——期望返回对应结构错误不因 n=0 豁免/
   α=0 且 x/y/AP 空指针——仍返回 `INVALID_VALUE`（指针校验先于 quick return）/
   对角虚部非零输入（验证输出置 0）））；
2. **工程**：`test/spr2/chpr2/arch22/chpr2_test.cpp` CSV 驱动 GTest；
   `test/spr2/chpr2/CMakeLists.txt` 注册 arch22 目标
   （编译入口对齐仓内 `--ops=chpr2 --soc=ascend910b3`，A3 对应 --soc）；
3. **精度回归**：1000 条全量 VerifyMixed（实/虚部分别统计）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建命令、gtest filter、msprof 命令、golden 构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 声明已存在不改（约 L515），与其他产品线共用（禁止私有平行 API）；
- arch22 与 sspr2 同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasChpr2_AtlasA2_A3_task_doc.md（2026-11-25）
2. Netlib chpr2：https://www.netlib.org/blas/chpr2.f
3. cuBLAS cublasChpr2：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
