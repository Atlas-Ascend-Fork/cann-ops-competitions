# aclblasChpr2 算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：complex64（x/y/AP/alpha；实部/虚部各 FP32）
> 合入位置：`ops-blas/blas/spr2/arch22/`；测试：`ops-blas/test/spr2/chpr2/arch22/`
> 任务书：`aclblasChpr2_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasChpr2 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/spr2/arch22/`（§1/§5）；测试代码 → `test/spr2/chpr2/arch22/`（§2.2）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 本设计与仓内现状的关系：上游已于 2026-11-23 合入 arch22 实现五件套（cann-robot），但其 host 校验与本任务书契约存在两处缺口（n=0 提前返回豁免结构校验、非法 uplo 错误码）；本设计 = **沿用已合入单 kernel 架构 + 契约对齐层**（见"总体架构与数据流"）。

## 背景介绍

### 算子功能定位

复数 Hermitian 压缩存储秩-2 更新：对 packed 存储的 n×n Hermitian 矩阵 AP，以两个向量 x、y 做互为共轭转置的双秩-1 更新（结果天然 Hermitian）：

```
AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasChpr2 | cuBLAS `cublasChpr2`｜Netlib `chpr2` | packed Hermitian 复数秩-2 更新 |

### 业界对标

| 项 | cuBLAS cublasChpr2 | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/n/alpha/x/incx/y/incy/AP | 逐参数一致（§2.3） |
| packed 布局 | Upper：AP[i + j(j+1)/2]（i≤j）；Lower：AP[i + j(2n−j−1)/2]（i≥j） | 同（§2.1.2/2.1.3） |
| 错误检查 | n=0 **不豁免**结构参数校验；alpha=0 quick return 前仍查指针 | 同（§2.5 校验顺序注——本设计对齐点） |
| 非法 uplo 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 对角虚部 | 计算后强制置 0 | 同（§2.1.4） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/spr2/arch22/chpr2_{host,kernel}.cpp + chpr2_kernel.h + chpr2_tiling_data.h`（复用 2026-11-23 已合入五件套，在其上做契约对齐，不新建文件）。
- 公共头：`include/cann_ops_blas.h` L515 声明**已存在，无需修改**（grep 核验）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/spr2/arch22/chpr2_*`（11-23 合入） | ★ 主基线 | 单 AIV kernel 架构（numBlocks=min(n,核数) 列带并行、核内秩-2 共轭对更新 + 对角虚部置 0）、tiling（n/uplo/numBlocks/alpha 分量/incx/incy int64）、size_t 溢出防护 | host 校验层按 §2.5 重排（n=0 后移）+ 非法 uplo 改 `INVALID_ENUM` |
| `blas/spr2/arch35/chpr2_*` | ◆ 参考 | 任务书指定参考（arch35 同名实现） | arch22 沿用现网方案 |
| `blas/spr2/arch22/sspr2_*`、`blas/spmv/arch22/chpmv_*` | ◇ 参考 | packed 下标映射、负步长起点折算、防溢出模式 | 实数版/读侧对照 |
| `test/spr2/chpr2/` 工程 | 既有 | ATK 测试组织、golden/param、arch22 目标已注册 | csv 负向预期按契约口径核对 |

### 接口规格

签名与任务书 §2.3 一致（= `include/cann_ops_blas.h` L515 既有声明）：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| uplo | AP 的存储三角模式 | attr | int | {UPPER, LOWER}；其它 → `INVALID_ENUM` | 标量 |
| n | 方阵 A 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op（**结构参数仍需校验**） | 标量 |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 实/虚部任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| x | 输入向量 | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE`（alpha=0 亦不豁免） | n 元素（物理 ≥1+(n-1)·abs(incx)） |
| incx | x 元素步长（可为负） | attr | int | 任意非零 int，INT_MIN 除外；0/INT_MIN → `INVALID_VALUE` | 标量 |
| y | 输入向量 | tensor | COMPLEX64 | 同 x | n 元素 |
| incy | y 元素步长（可为负） | attr | int | 同 incx | 标量 |
| AP | Hermitian 压缩矩阵（原地覆写，对角虚部置 0） | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE` | n(n+1)/2 元素 |

### 数学公式

```
AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
逐 packed 元素（uplo 三角内，i 与 j 同列）：
AP(i,j) += alpha·x[i]·conj(y[j]) + conj(alpha)·y[i]·conj(x[j])
对角（i==j）: AP(j,j) += 2·Re(alpha·x[j]·conj(y[j]))，虚部强制置 0
```

packed 下标映射（§2.1.2/2.1.3）：Upper `AP[i + j(j+1)/2]`（i≤j）、Lower `AP[i + j(2n−j−1)/2]`（i≥j）。负步长按 BLAS 反向遍历语义（起点折算 (1−n)·inc）。特值：n=0 或 alpha=(0,0) → quick return（AP 不更新；alpha=0 **仍要求 x/y/AP 指针非空**——指针校验先于零 alpha quick return，§2.5 顺序）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 complex64 句柄式 packed Hermitian 秩-2 更新接口 aclblasChpr2：单 AIV kernel 列带并行（核内完成双秩-1 共轭对更新、对角虚部置 0 与 packed 原地读改写）；支持 uplo 双向 packed 下标映射、任意非零（含负、INT_MIN 除外）步长、n 至 4096；精度按 FP32 分量达生态混合容差（golden = cblas_chpr2 单精度直接输出），性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 秩-2 = 两个互共轭秩-1 之和；alpha 与 conj(alpha) 的分量展开（αr±i·αi 符号对折）核内一次遍历完成；单 kernel 架构整体复用仓内已合入实现（11-23） |
| 2 | 接口原型 | §2.3 签名（= L515 既有声明），返回 `aclblasStatus_t`，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | x/y/AP/alpha 全 COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--soc=ascend910b3`；arch35 已有实现不动 |
| 6 | 精度要求 | §3.2：FP32 分量 rtol=atol=2^-13、ratio ≥0.99、max_abs ≤ 1e-2 或 32 ULP；golden = cblas_chpr2（CblasColMajor 单精度直接输出，实/虚部分别比对） |
| 7 | 性能预期 | §3.3：单 case NPU Task Duration ≤ 1.25×gpu_ms（≥0.8× GPU）；gpu_baseline 200 条实测 max=0.9267ms / avg=0.2847ms |
| 8 | 代码架构选型 + 可行性 | 单 AIV kernel（列带并行、原地读改写、无 workspace、无两段式——更新侧无跨核归约需求）；已排除死路：两段式部分和（写侧按元素更新无归约可拆）、cube 路径（秩-2 无矩阵级收益）；可行性旁证 = 同架构实现已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/（1201 行 csv = 精度 1000 + 性能 200；gpu_baseline 201 行） |
| 10 | 契约对齐层 | 相对 11-23 已合入实现需补 2 点：① **n=0 quick return 位置**（现网在 alpha/uplo/incx/incy 校验前提前返回，违反 §2.5 不豁免条款；移到 incx/incy 校验后、x/y/AP 判空前）；② **非法 uplo 错误码**（现网 INVALID_VALUE → 按 §2.4 改 INVALID_ENUM） |
| 11 | 特值与退化 | n=0 no-op（结构参数仍校验）；alpha=0 quick return（不读 x/y 数值，**指针仍需非空**）；对角虚部计算后置 0（输入对角虚部被忽略） |
| 12 | workspace | 无（AP 原地读改写；核间列带不相交） |

# 详细设计

## 算子分析

### 数学公式

逐 packed 元素（uplo 三角内，FP32 分量域，记 α = αr + i·αi）：

```
AP(i,j) += α·x[i]·conj(y[j]) + conj(α)·y[i]·conj(x[j])
分量展开（记 x[i]=xr+i·xi，y[j]=yr+i·yj）：
  第一项 α·x·conj(y) = (αr+i·αi)(xr+i·xi)(yr−i·yj)
  第二项为第一项的共轭转置配对（y[i]·conj(x[j])）
  Δre = 2·Re(第一项)（两项实部相等）；非对角 Δim = 2·Im(第一项)
对角（i==j）：Δ = 2·Re(α·x[j]·conj(y[j]))（纯实），AP(j,j) 虚部强制置 0
```

packed 下标（§2.1.2/2.1.3）：Upper `AP[i + j(j+1)/2]`（i≤j）、Lower `AP[i + j(2n−j−1)/2]`（i≥j）；负步长起点折算 `(1−n)·inc`。特值语义同"需求背景·数学公式"。

### 支持数据类型

| 对象 | 接口 dtype | kernel 内部 |
|---|---|---|
| x / y / AP | COMPLEX64 | 拆 re/im 两平面 FP32 分量计算，交织读写 |
| alpha | COMPLEX64（Host/Device 双指针位置） | tiling 传 re/im 分量（α 与 conj(α) 由符号对折） |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥ 0 | 0 → 结构参数校验通过后 SUCCESS no-op（不引用 AP/x/y、不下发 kernel）；非法结构参数仍报错（§2.5） |
| incx/incy | 任意非零 int（含负；INT_MIN 除外） | 0 / INT_MIN → INVALID_VALUE |
| 覆盖档位（§3.5） | n ∈ {0,1,2,3,5,7,16,31,32,33,63,64,65,127,128,129,255,256,511,512,1023,1024,2047,2048,4095,4096}；incx/incy ±1/±2/±3 + 大步长 {7,8,16,31,64,127} | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

沿用仓内 11-23 已合入的单 kernel 架构：

```
aclblasChpr2 (chpr2_host.cpp)
 └─ 校验（§2.5 顺序，见"数据检测"）→ n=0 quick return → alpha=(0,0) quick return
 └─ chpr2_kernel（单 AIV kernel，numBlocks = min(n, aivCoreNum)）:
        每核一个列带 [colBase, colBase+cols)；逐列 j：
          x[j]/y[j] strided 读 → 对带内行 i 逐元素：
            AP(i,j) += α·x[i]·conj(y[j]) + conj(α)·y[i]·conj(x[j])（FP32 分量展开）
          对角元 (i==j) 虚部置 0
          packed 原地写回（Upper/Lower 下标映射）
```

**契约对齐层**（相对 11-23 已合入实现，host 校验两处）：

| # | 对齐项 | 现状（arch22 host） | 契约目标（§2.5 注/§2.4） |
|---|---|---|---|
| 1 | n=0 quick return 位置 | n==0 在 alpha/uplo/incx/incy 校验**前**直接返回 SUCCESS | 移至 incx/incy 校验**后**、x/y/AP 判空**前**——n=0 不豁免结构参数校验，只豁免 tensor 判空与 kernel |
| 2 | 非法 uplo 错误码 | INVALID_VALUE（ValidateChpr2Body L37-38） | **INVALID_ENUM**（§2.4 表） |

数据流：GM x/y（strided）+ GM AP（packed 交织）→ 核内列带更新（读-改-写）→ GM AP 原地。无中间暂存。

#### host 侧设计

- **分核策略**：`numBlocks = min(n, aivCoreNum)`，按 packed 列带均分（列 j 的 packed 段长随 j 变化——Upper 段长 j+1、Lower 段长 n−j；按累计元素量均衡列带边界，避免长尾核）。
- **数据分块和内存优化策略**：核内逐列处理，x[j]/y[j] 每列各读一次供整列更新复用；无 workspace、无 device 临时分配；size_t 溢出防护（lenX/lenY/lenAP ≤ maxElements，复用现网 L45-55）。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| uplo | 0/1 | Upper/Lower packed 下标映射 |
| alphaReal/Imag | FP32 分量 | α 与 conj(α) 符号对折 |
| incx/incy | int64 | strided 访问（负步长起点折算） |
| numBlocks | 切核数 | 列带划分 |

- **数据检测**（校验顺序逐条对齐 §2.5 注——cuBLAS 口径）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | n < 0 | INVALID_VALUE |
| 3 | alpha == nullptr | INVALID_VALUE |
| 4 | uplo ∉ {UPPER, LOWER} | **INVALID_ENUM** |
| 5 | incx/incy == 0 或 INT_MIN | INVALID_VALUE |
| 6 | **n == 0 → SUCCESS（不下发 kernel）** | — |
| 7 | n>0 时 x/y/AP == nullptr | INVALID_VALUE |
| 8 | 溢出防护（lenX/lenY/lenAP ≤ size_t 域） | INVALID_VALUE |
| 9 | alpha=(0,0) → SUCCESS（不读 x/y 数值；指针已在第 7 步判过） | — |

  kernel 发射失败（aclError ≠ SUCCESS）→ `EXECUTION_FAILED` 记日志。

#### kernel 侧设计

- **Init 阶段**：blockIdx → 本核列带 [colBase, colBase+cols)；绑定 x/y/AP 的 GM tensor；UB 小 buffer 缓存列头元素。
- **Process 阶段**（主循环）：

```
for j ∈ 本核列带:
    xj = strided_read(x, j·incx) ; yj = strided_read(y, j·incy)     （列头，一次读）
    for i ∈ 该列 packed 行域（Upper: 0..j；Lower: j..n-1）:
        xi = strided_read(x, i·incx) ; yi = strided_read(y, i·incy)
        term = α·xi·conj(yj) + conj(α)·yi·conj(xj)                   （FP32 分量展开）
        AP[packIdx(i,j)] += term                                      （交织读-改-写）
    AP[packIdx(j,j)].imag = 0                                          （对角虚部强制置 0）
```

- **workspace 布局表**：无（原地读改写；核间列带互不相交，无竞争）。

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| 列带核间并行 | packed 列带按元素量均衡划分 | 满核利用（待 msprof 回填） |
| 列头复用 | x[j]/y[j] 每列一次读供整列 | x/y 读量 ~2n（相对逐元素 2n²） |
| 单 kernel 原地 | 无中间暂存、无二次发射 | 更新侧零额外带宽 |
| 已排除路线 | 两段式部分和（写侧无归约可拆）；cube 路径（秩-2 无矩阵级收益） | — |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline 200 条 max=0.9267ms / avg=0.2847ms（cublasChpr2 kernel-only）。达标以设备 msprof 实测为准（设计阶段不判决）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A2 推理系列产品 | ✔（同 arch22） |
| Atlas A3 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A3 推理系列产品 | ✔（同 arch22） |
| Ascend 950PR / 950DT | （arch35 既有实现，本任务不涉及） |

## 算子约束限制

| 约束项 | 内容（§2.5 逐条） |
|---|---|
| 校验层级 | n/uplo/incx/incy/alpha host 侧校验；不扫描 Device 数据；Hermitian 性由调用方保证 |
| 非连续支持 | x/y 按 incx/incy 跨步（含负）；AP 列主序无间隙 packed；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 天然支持（n/步长运行时入参驱动 tiling） |
| 原地语义 | AP 原地覆写、不返回视图；AP 可写区与 x/y 可读区不重叠由调用方保证 |
| 确定性 | 归约顺序由实现决定，不保证逐位一致；按 §3.2 阈值判定 |
| 空 Tensor | n=0 no-op（结构参数仍校验，仅豁免 tensor 判空与 kernel） |
| 0 维 Tensor | 不涉及（标量 Host/Device 双位置支持） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = Netlib CBLAS `cblas_chpr2`（CblasColMajor，单精度复数直接输出）；实/虚部分别逐元素 `|actual−golden| ≤ atol + rtol·|golden|`；FLOAT32 分量 rtol=atol=2^-13、matched_ratio ≥0.99 且 max_abs_error ≤ 1e-2 或 32 ULP 时用例通过 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration（warmup5+10 均值）单 case ≤1.25×gpu_ms（≥0.8× GPU）；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py） | §3.3；test_cases/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 1000 + 性能子集 200（csv 实测 1201 行）；性能子集按 small ≤64KiB / medium ≤2MiB / large >2MiB 分层。
- 覆盖：uplo{U,L}；n 全档至 4096；incx/incy ±1/±2/±3 + 大步长 {7,8,16,31,64,127}（性能以单位步长为主 + 少量大步长/负步长）；alpha 特殊值 (0,0)/(1,0)/(0,1)；负向（非法 uplo→INVALID_ENUM/负 n/零步长/INT_MIN/空指针）。
- 特殊场景：对角虚部非零输入（输出置 0）；alpha=0 且 x/y 为 NaN（quick return 不读数值）；**alpha=0 且 x/y/AP 为空指针仍返回 INVALID_VALUE**（指针校验先于零 alpha quick return）；n=0 叠加非法结构参数（不因 n=0 豁免）；Inf/NaN 传播。
- 性能口径：msprof op（正式，README"精确 kernel 耗时获取方式"节）+ verify_performance.py 交叉。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/spr2/chpr2/` ATK 工程）。

## 兼容性分析

- 公共头零改动（L515 既有声明），无接口面波及。
- 行为变更（契约对齐，arch22 范围内）：① n=0 叠加非法结构参数从 SUCCESS（现网提前返回）改为返回对应结构错误；② 非法 uplo 从 INVALID_VALUE 改为 INVALID_ENUM——均与任务书 §2.4/§2.5 及 cuBLAS 口径一致；`test/spr2/chpr2/arch22/` 既有 csv 的相关负向预期按此核对更新。
- 在 11-23 已合入五件套上原位修改（host 校验层），不新增文件；arch35 与 sspr2 路径零触碰。
- README：`blas/spr2/README.md` 的 aclblasChpr2 章节核对 A2/A3 支持情况与错误码口径说明。
- 无既有合法调用方行为变化（仅非法参数场景的错误码修正）。

