# aclblasChpmv 算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：complex64（AP/x/y/alpha/beta；实部/虚部各 FP32）
> 合入位置：`ops-blas/blas/spmv/arch22/`；测试：`ops-blas/test/spmv/chpmv/arch22/`
> 任务书：`aclblasChpmv_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasChpmv 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/spmv/arch22/`（§1/§5）；测试代码 → `test/spmv/chpmv/arch22/`（§2.2）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 本设计与仓内现状的关系：上游已于 2026-11-23 合入 arch22 实现五件套（cann-robot），但其 host 校验顺序与本任务书契约存在一处缺口（n=0 提前返回豁免了结构参数校验）；本设计 = **沿用已合入两段式架构 + 契约对齐层**（见"总体架构与数据流"）。

## 背景介绍

### 算子功能定位

复数 Hermitian 压缩存储矩阵-向量乘：A 为 n×n Hermitian，以 packed 格式（AP 列主序逐列无间隙，仅 uplo 三角，共 n(n+1)/2 个复元素，无前导维参数）存放。

```
y = alpha * A * x + beta * y
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasChpmv | cuBLAS `cublasChpmv`｜Netlib `chpmv` | packed Hermitian 复数 mv |

### 业界对标

| 项 | cuBLAS cublasChpmv | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/n/alpha/AP/x/incx/beta/y/incy | 逐参数一致（§2.3） |
| packed 布局 | Upper：AP[i + j(j+1)/2]（i≤j）；Lower：AP[i + j(2n−j−1)/2]（i≥j） | 同（§2.1.2/2.1.3） |
| 对角虚部 | 不引用（只取实部），非存储三角共轭隐含 | 同（§2.1.4） |
| 错误检查 | n=0 **不豁免**结构参数校验（非法 uplo/incx=0 仍报错） | 同（§2.5 校验顺序注——本设计对齐点） |
| 非法 uplo 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 标量 | alpha/beta 复数 | 同（Host/Device 指针位置均支持，§2.5） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/spmv/arch22/chpmv_{host,kernel}.cpp + chpmv_kernel.h + chpmv_tiling_data.h`（复用 2026-11-23 已合入五件套，在其上做契约对齐，不新建文件）。
- 公共头：`include/cann_ops_blas.h` L161 声明**已存在，无需修改**（grep 核验）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/spmv/arch22/chpmv_*`（11-23 合入） | ★ 主基线 | 两段式 kernel（accum 部分和 → reduce 归约+beta+写回）、scale-only 旁路、tiling（nPad/accumBlocks/reduceBlocks）、workspace 每核 nPad×2 float | host 校验层按 §2.5 重排（n=0 移到 alpha/beta 判空后、AP/x/y 判空前） |
| `blas/spmv/arch35/chpmv_*` | ◆ 参考 | 任务书指定参考（变体分派/tiling 对照） | arch22 沿用现网方案 |
| `blas/spmv/arch22/sspmv_*` | ◇ 参考 | packed 对称 mv 实数版工程组织 | 复数化差异 |
| `test/spmv/chpmv/` 工程 | 既有 | ATK 测试组织、golden/param/utils、run_self_test.py | csv 按自验包口径核对（n 到 8192、负向预期） |

### 接口规格

签名与任务书 §2.3 一致（= `include/cann_ops_blas.h` L161 既有声明）：

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| uplo | AP 的存储三角模式 | attr | int | {UPPER, LOWER}；其它 → `INVALID_ENUM` | 标量 |
| n | 方阵 A 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op（**结构参数仍需校验**） | 标量 |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 实/虚部任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| AP | Hermitian 压缩矩阵（仅 uplo 三角，对角虚部忽略） | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE` | n(n+1)/2 元素 |
| x | 输入向量 | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE`；物理 ≥1+(n-1)·abs(incx) | n 元素 |
| incx | x 元素步长（可为负） | attr | int | 任意非零 int（含 INT_MIN 防护）；0 → `INVALID_VALUE` | 标量 |
| beta | 复数标量乘数；beta=(0,0) 时 y 不必有效 | scalar | COMPLEX64 | 同 alpha；nullptr → `INVALID_VALUE` | 标量 |
| y | 输入/输出向量（原地覆写） | tensor | COMPLEX64 | n>0 时 nullptr → `INVALID_VALUE` | n 元素 |
| incy | y 元素步长（可为负） | attr | int | 同 incx；0 → `INVALID_VALUE` | 标量 |

### 数学公式

```
y = alpha * A * x + beta * y,   A = A^H（Hermitian，仅存 uplo 三角）
y[i] = alpha * ( Σ_{j} A[i,j]·x[j] ) + beta·y[i]
```

packed 下标映射（列主序逐列无间隙，§2.1.2/2.1.3）：

```
Upper: A(i,j)（i ≤ j）= AP[i + j*(j+1)/2]        非存储下三角按 conj(A(j,i)) 引用
Lower: A(i,j)（i ≥ j）= AP[i + j*(2*n-j-1)/2]    非存储上三角按 conj(A(j,i)) 引用
对角:  A(i,i) 实部参与计算，虚部不引用（实现与 golden 均按 0 处理）
```

负步长按 BLAS 反向遍历语义（x/y 起点折算 (1-n)·inc）。特值：alpha=(0,0)∧beta=(1,0) → quick return；beta=(0,0) → 不读 y 旧值；alpha=(0,0)∧beta≠(1,0) → 仅 `y = beta·y`。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 complex64 句柄式 packed Hermitian mv 接口 aclblasChpmv：两段式 AIV kernel（核间部分和 + 跨核归约）+ alpha=0 scale-only 旁路；支持 uplo 双向 packed 下标映射、对角虚部忽略、任意非零（含负）步长、n 至 8192；精度按 FP32 分量达生态混合容差（golden = cblas_chpmv 单精度直接输出），性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | packed Hermitian mv = 逐元素复乘加 + 三角共轭隐含；两段式（accum/reduce）架构整体复用仓内已合入实现（11-23） |
| 2 | 接口原型 | §2.3 签名（= L161 既有声明），返回 `aclblasStatus_t`，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | AP/x/y/alpha/beta 全 COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--soc=ascend910b3`；arch35 已有实现不动 |
| 6 | 精度要求 | §3.2：FP32 分量 rtol=atol=2^-13、ratio ≥0.99、max_abs ≤ 1e-2 或 32 ULP；golden = cblas_chpmv（CblasColMajor 单精度直接输出，实/虚部分别比对） |
| 7 | 性能预期 | §3.3：单 case NPU Task Duration ≤ 1.25×gpu_ms（≥0.8× GPU）；gpu_baseline 200 条实测 max=1.5271ms / avg=0.6306ms |
| 8 | 代码架构选型 + 可行性 | 两段式 AIV kernel + scale-only 旁路（已合入架构的延续）；已排除死路：单 kernel 全归约（跨核部分和无处暂存）、cube 路径（mv 无矩阵级收益）；可行性旁证 = 同架构实现已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/（1201 行 csv = 精度 1000 + 性能 200；gpu_baseline 201 行） |
| 10 | 契约对齐层 | 相对 11-23 已合入实现需补 1 点：**n=0 quick return 位置**——现网在 uplo/incx/incy/alpha/beta 校验前提前返回 SUCCESS，违反 §2.5「n=0 不豁免结构参数校验」；需移至 alpha/beta 判空后、AP/x/y 判空前 |
| 11 | 特值与退化 | n=0 no-op（结构参数仍校验）；alpha=0∧beta=1 quick return；beta=0 不读 y（NaN 安全）；alpha=0∧beta≠1 仅 beta 缩放 |
| 12 | workspace | 每核 nPad×2 float 部分和（accumBlocks × nPad × 2 × 4B，来自 handle 有效 workspace；nPad=CeilAlign(n,8)） |

# 详细设计

## 算子分析

### 数学公式

逐元素（i = 0..n-1，FP32 分量域）：

```
y[i] = alpha · ( Σ_{j≠i} A(i,j)·x[j] + Re(AP_diag(i))·x[i] ) + beta · y[i]
其中（uplo=LOWER，i ≥ j 时直读，i < j 时 A(i,j) = conj(A(j,i))）：
  A(i,j)·x[j] = (Ar − i·Ai)(xr + i·xi) → 分量：re += Ar·xr + Ai·xi；im += Ar·xi − Ai·xr（conj 展开）
对角项只取实部：re += Re(AP_diag)·xr；im += Re(AP_diag)·xi
```

packed 下标（§2.1.2/2.1.3）：Upper `AP[i + j(j+1)/2]`、Lower `AP[i + j(2n−j−1)/2]`；负步长起点折算 `(1−n)·inc`。特值语义同"需求背景·数学公式"。

### 支持数据类型

| 对象 | 接口 dtype | kernel 内部 |
|---|---|---|
| AP / x / y | COMPLEX64 | 拆 re/im 两平面 FP32 分量计算 |
| alpha / beta | COMPLEX64（Host/Device 双指针位置） | tiling 传 re/im 分量 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥ 0 | 0 → 结构参数校验通过后 SUCCESS no-op（不引用 AP/x/y、不下发 kernel）；非法结构参数仍报错（§2.5） |
| incx/incy | 任意非零 int（含负；INT_MIN 防护） | 0 → INVALID_VALUE |
| 覆盖档位（§3.5） | n ∈ {0,1,2,3,5,7,16,31,32,33,63,64,65,127,128,129,255,256,511,512,1023,1024,2047,2048,4095,4096,8191,8192}；incx/incy ±1/±2/±3 + 大步长 {7,8,16,31,64,127} | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

沿用仓内 11-23 已合入的两段式架构 + scale-only 旁路（单入口三分发）：

```
aclblasChpmv (chpmv_host.cpp)
 └─ 校验（§2.5 顺序，见"数据检测"）→ alpha=0∧beta=1 quick return
 ├─ alpha=(0,0)：  chpmv_scale_kernel            （y = beta·y，strided，beta=0 直写 0）
 └─ 常规：        chpmv_accum_kernel             （每核行带：AP packed 读 + x strided 读
                                                    → 复乘加部分和 re/im → workspace 每核 nPad×2 float）
                  chpmv_reduce_kernel            （跨核部分和归约 + alpha·Ax + beta·y
                                                    + 对角实部项 + strided 写回，beta=0 不读 y）
```

**契约对齐层**（相对 11-23 已合入实现，仅 host 校验一处）：

| # | 对齐项 | 现状（arch22 host） | 契约目标（§2.5 注） |
|---|---|---|---|
| 1 | n=0 quick return 位置 | n==0 在 uplo/incx/incy/alpha/beta 校验**前**直接返回 SUCCESS | 移至 alpha/beta 判空**后**、AP/x/y 判空**前**——n=0 不豁免结构参数校验，只豁免 AP/x/y 判空与 kernel |

数据流：GM AP（packed 交织）+ GM x（strided）→(accum)→ GM workspace 部分和（每核 re/im 平面）→(reduce)→ GM y（strided，原地覆写）。共轭与对角实部处理折入 accum 的列遍历。

#### host 侧设计

- **分核策略**：accum 按 n 均分行带（`accumBlocks = min(n, aivCoreNum, workspace 可容纳核数)`，满核原则 + ws 上限收缩）；reduce 按 `reduceBlocks = min(ceil(n/TILE_ROWS), aivCoreNum)` 列块归并。
- **数据分块和内存优化策略**：nPad = CeilAlign(n, 8)（复数 16B 对齐）；部分和复用 handle 有效 workspace（GetEffectiveWorkspace，不足时报 EXECUTION_FAILED）；无 device 侧临时分配。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| uplo | 0/1 | Upper/Lower packed 下标映射与共轭方向 |
| alphaReal/Imag、betaReal/Imag | FP32 分量 | 复数 epilogue；alpha=(0,0) 分发 scale-only；beta=(0,0) 不读 y |
| incx/incy | int64 | strided 访问（负步长起点折算） |
| accumBlocks/reduceBlocks/tileRows | 切核三元组 | 两段式核间划分 |

- **数据检测**（校验顺序逐条对齐 §2.5 注——cuBLAS 口径）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | n < 0 | INVALID_VALUE |
| 3 | uplo ∉ {UPPER, LOWER} | INVALID_ENUM |
| 4 | incx==0 或 incy==0（含 INT_MIN 防护） | INVALID_VALUE |
| 5 | alpha == nullptr 或 beta == nullptr | INVALID_VALUE |
| 6 | **n == 0 → SUCCESS（不下发 kernel）** | — |
| 7 | n>0 时 AP/x/y == nullptr | INVALID_VALUE |
| 8 | alpha=(0,0) 且 beta=(1,0) → SUCCESS（不读 x/AP） | — |
| 9 | alpha=(0,0) → scale-only 旁路 | — |

  kernel 发射失败（aclError ≠ SUCCESS）→ `EXECUTION_FAILED` 记日志。

#### kernel 侧设计

- **chpmv_accum_kernel**：本核行带 [rowBase, rowBase+rows)；逐行 i：遍历列 j（uplo 三角内直读 AP、三角外按 conj(A(j,i)) 引用），x[j] strided 读入；FP32 分量乘加累积 re[i]/im[i]；对角项按实部处理；行带结束写 workspace 本核平面。行带内 x 元素复用（按列外积式遍历时 x[j] 每列一次读、对带内多行复用）。
- **chpmv_reduce_kernel**：对本核列块：跨 accumBlocks 个部分和平面求和 → `t = alpha·Σ + beta·y`（beta=0 时 y 项以 0 代入不读 GM）→ 按输出 dtype 交织写回 y（incy strided，负步长折算起点）。
- **chpmv_scale_kernel**：`y[i] = beta·y[i]`（beta=(0,0) 时直写 0，不读 y）。

```
流程（常规路径）：
GM AP ─┐
GM x ──┴→ accum（行带×列遍历，conj/对角实部处理）→ ws 部分和（每核 re/im）
              → reduce（跨核求和 + α/β epilogue）→ GM y（strided 原地写）
```

- **workspace 布局表**（FP32 元素）：

| 区 | 大小 | 用途 |
|---|---|---|
| partialRe/partialIm | accumBlocks × nPad（交织平面） | 每核部分和（复用 handle workspace） |

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| 两段式核间并行 | accum 行带并行 + reduce 归并 | packed 三角读均分到满核（待 msprof 回填） |
| x 列复用 | 列遍历中 x[j] 一读多行复用 | x 读量 ~1×（相对逐行重读 n×） |
| scale-only 旁路 | alpha=0 只发单 kernel | 特值用例省 AP/x 读带宽 |
| 三角裁剪 | 仅 uplo 三角 packed 元素参与 | AP 读量 = n(n+1)/2（全阵的一半） |
| 已排除路线 | 单 kernel 全归约（部分和无处暂存）；cube 路径（mv 无矩阵级收益） | — |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline 200 条 max=1.5271ms / avg=0.6306ms（cublasChpmv kernel-only）。达标以设备 msprof 实测为准（设计阶段不判决）。

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
| 校验层级 | n/uplo/incx/incy/alpha/beta host 侧校验；不扫描 Device 数据；Hermitian 性由调用方保证 |
| 非连续支持 | x/y 按 incx/incy 跨步（含负）；AP 列主序无间隙 packed；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 天然支持（n/步长运行时入参驱动 tiling） |
| 原地语义 | y 原地覆写、不返回视图；y 可写区与 x/AP 可读区不重叠由调用方保证 |
| 确定性 | 归约顺序由实现决定，不保证逐位一致；按 §3.2 阈值判定 |
| 空 Tensor | n=0 no-op（结构参数仍校验，仅豁免 AP/x/y 判空与 kernel） |
| 0 维 Tensor | 不涉及（标量 Host/Device 双位置支持） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = Netlib CBLAS `cblas_chpmv`（CblasColMajor，单精度复数直接输出）；实/虚部分别逐元素 `|actual−golden| ≤ atol + rtol·|golden|`；FLOAT32 分量 rtol=atol=2^-13、matched_ratio ≥0.99 且 max_abs_error ≤ 1e-2 或 32 ULP 时用例通过 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration（warmup5+10 均值）单 case ≤1.25×gpu_ms（≥0.8× GPU）；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py） | §3.3；test_cases/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 1000 + 性能子集 200（csv 实测 1201 行）；性能子集按 small ≤64KiB / medium ≤2MiB / large >2MiB 分层。
- 覆盖：uplo{U,L}；n 全档至 8192；incx/incy ±1/±2/±3 + 大步长 {7,8,16,31,64,127}（性能以单位步长为主 + 少量大步长/负步长）；负向（非法 uplo/负 n/零步长/空指针/**n=0 叠加非法结构参数**）。
- 特殊场景：对角虚部非零输入（虚部被忽略）；alpha=0∧beta=1（不读 x）；beta=0 且 y 为 NaN（不读旧值）；alpha=0∧beta≠1（只做缩放）；Inf/NaN 传播。
- 性能口径：msprof op（正式，README"精确 kernel 耗时获取方式"节）+ verify_performance.py 交叉。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/spmv/chpmv/` ATK 工程：run_self_test.py / run_memcheck.py）。

## 兼容性分析

- 公共头零改动（L161 既有声明），无接口面波及。
- 行为变更（契约对齐，arch22 范围内）：n=0 叠加非法结构参数（非法 uplo/incx=0/空 alpha/beta）从 SUCCESS（现网提前返回）改为返回对应结构错误——与任务书 §2.5 及 cuBLAS 一致；`test/spmv/chpmv/arch22/` 既有 csv 的 n=0 负向预期按此口径核对更新。
- 在 11-23 已合入五件套上原位修改（host 校验顺序一处），不新增文件；arch35 与 sspmv 路径零触碰。
- README：`blas/spmv/README.md` 的 aclblasChpmv 章节核对 A2/A3 支持情况与 n=0 校验口径说明。
- 无既有合法调用方行为变化（仅非法参数场景的错误码修正）。

