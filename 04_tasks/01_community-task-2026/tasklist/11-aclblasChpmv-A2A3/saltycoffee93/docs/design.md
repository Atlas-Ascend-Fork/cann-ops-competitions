# aclblasChpmv 算子设计文档

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/spmv/arch22/`
> 任务书：`aclblasChpmv_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasChpmv 算子开发（A2A3）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/spmv/arch22/`；测试 `test/spmv/chpmv/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

`aclblasChpmv` 实现单精度复数（complex64）Hermitian 压缩存储矩阵-向量乘：`y = alpha·A·x + beta·y`。n×n Hermitian 矩阵以 packed 格式存放于 AP（仅 uplo 指定的一半三角，n(n+1)/2 个复数元素，无前导维），支持任意非零步长（含负步长反向遍历）。

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasChpmv | cuBLAS `cublasChpmv`｜Netlib `chpmv.f` | 复数 Hermitian packed 矩阵-向量乘 |

### 业界对标

| 项 | cuBLAS cublasChpmv | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/n/alpha/AP/x/incx/beta/y/incy | 逐参数一致（§2.3、§3.5） |
| packed 布局 | 列主序逐列无间隙打包 | 同：UPPER `AP[i + j(j+1)/2]`；LOWER `AP[i + j(2n−j−1)/2]`（§2.1） |
| 对角虚部 | 不引用 | 同：实现与 golden 均只取对角实部 |
| 步长 | incx/incy 任意非零（含负） | 同；incx=0 或 incy=0 → `INVALID_VALUE` |
| n=0 | 结构参数非法仍报错（不豁免） | 同（§2.5 校验顺序说明，与 cuBLAS 一致） |
| 非法 uplo 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |

### 工程形态

- 工程模式：Kernel 直调（Host C API 校验/分派/tiling，经 handle 绑定的 stream 直调 NPU kernel，§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| arch22 实现 | **已合入**（两段式 AIV 实现） | `blas/spmv/arch22/chpmv_host.cpp`(156 行)、`chpmv_kernel.cpp`(695 行)、`chpmv_kernel.h`、`chpmv_tiling_data.h`；提交 `f600423`（2026-11-23）、修复 `5a01644`（2026-11-24，kernel_do 返回发射状态） | 按任务书口径核对并补齐差异（见 host 侧设计），测试用例对齐 |
| arch35 参考 | 已有 | `blas/spmv/arch35/chpmv_*`（613 行） | 参考口径 |
| 公开声明 | 已存在 | `include/cann_ops_blas.h` 约 L161 | 零公共头改动 |
| 测试工程 | arch22/arch35 均已有 | `test/spmv/chpmv/arch22/chpmv_test.cpp` 等 | 用例按任务包口径对齐扩展 |
| README | 章节已有 | `blas/spmv/README.md` L207 `### aclblasChpmv` | 核对补充 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/spmv/arch22/chpmv_*`（已合入） | ★ 主对象 | 两段式 AIV 流水（见详细设计）、packed 下标映射、负步长语义、scale-only 快路 | 仅做任务书口径对齐 |
| `blas/spmv/arch35/chpmv_*` | ◇ 参考 | 同算子 arch35 实现口径 | 不复制代码 |

### 接口规格

声明已存在于 `include/cann_ops_blas.h`（约 L161），**无需新增或修改**（§2.3）。签名：

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

参数规格（逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| uplo | AP 的存储三角 | attr | int | UPPER / LOWER；其它 → `INVALID_ENUM` | — |
| n | 方阵阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op（结构参数仍校验） | — |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE`；Host/Device 指针位置均支持 | 标量 |
| AP | Hermitian 压缩矩阵 | tensor | COMPLEX64 | n>0 时非空；对角虚部忽略 | n(n+1)/2 个元素 |
| x | 输入向量 | tensor | COMPLEX64 | n>0 时非空；物理元素 ≥1+(n−1)·\|incx\| | n 个元素 |
| incx | x 步长 | attr | int | 非零；0 → `INVALID_VALUE`；可为负 | 标量 |
| beta | 复数标量乘数 | scalar | COMPLEX64 | 非空；=0 时 y 不必有效 | 标量 |
| y | 输出向量（原地覆写） | tensor | COMPLEX64 | n>0 时非空；物理元素 ≥1+(n−1)·\|incy\| | n 个元素 |
| incy | y 步长 | attr | int | 非零；0 → `INVALID_VALUE`；可为负 | 标量 |

### 数学公式

```text
y(i) = alpha · Σ_j AP(i,j)·x(j) + beta·y(i)      （i = 0..n−1，按 incy 步长写入）
packed 下标（列主序）：
    UPPER：A(i,j)（i ≤ j）存于 AP[i + j(j+1)/2]；列 j 元素 A(i,j)（i ≥ j）由 A^H 隐含（取 AP[j + j(j+1)/2] 的共轭转置关系）
    LOWER：A(i,j)（i ≥ j）存于 AP[i + j(2n−j−1)/2]
对角元素只取实部（AP 对角项虚部不引用）。
```

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上提供句柄式接口 `aclblasChpmv`：complex64、UPPER/LOWER packed、任意非零步长（含负）、n ≤ 8192；精度达生态算子开源混合容差标准 FLOAT32 分量档，性能达 GPU 标杆 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | `y = alpha·A·x + beta·y`，A 为 packed Hermitian。已合入的 arch22 实现为"两段式"：先按分核把 packed 行乘累加为每核部分和，再归约并按 alpha/beta 组装写回——带宽型 BLAS-2，无 Cube 依赖 |
| 2 | 接口原型 | §2.3 签名（10 参数），返回 `aclblasStatus_t`；声明已存在，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | 全 COMPLEX64（AP/x/y/alpha/beta） |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；arch35 已有实现，不在本任务范围 |
| 6 | 精度要求 | §3.2：FLOAT32 分量档 rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32ULP；golden=`cblas_chpmv`（CblasColMajor） |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 + 10 样本 |
| 8 | 代码架构选型 + 可行性 | arch22 已合入两段式 AIV 实现（accum → reduce，带 scale-only 快路），本任务以口径对齐为主：**已核验一处校验顺序差异需修正**（n=0 提前返回先于 uplo/步长/标量校验，违反 §2.5"n=0 不豁免结构校验"）。NPU 侧达标性**待实测** |
| 9 | 评测来源 | 任务包 `test_cases/`：精度 1000 条 + 性能 200 条（`gpu_baseline.csv` 200 条全回填），CSV 安装到 `test/spmv/chpmv/<arch>/` 由 `chpmv_test` 加载 |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。计算按"逐输出元素对 packed 列做点积"组织：输出 y(i) 的被引用元素为 AP 中列 i 的 uplo 三角段与来自 A^H 隐含的行段（读时按共轭对称取对端元素的共轭），对角项只取实部。

### 支持数据类型

| 项 | 类型 | 说明 |
|---|---|---|
| AP/x/y | COMPLEX64 | packed 列主序 / 步长向量 |
| alpha/beta | COMPLEX64 | Host/Device 指针位置均支持（§2.5） |
| 计算精度 | FP32 | AIV 向量计算 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥0 | n=0：**结构参数（uplo/incx/incy/alpha/beta）先校验**，通过后 SUCCESS，不引用 AP/x/y、不启动 kernel |
| incx/incy | 非零（可负） | 0 → INVALID_VALUE；负步长按反向遍历访问 |
| AP/x/y 指针 | n>0 时非空 | 否则 INVALID_VALUE |

beta=(0,0) 时不读 y 旧值；y 可写区域与 x/AP 可读区域不得重叠。

## 算子实现

### 实现方案

#### 总体架构与数据流

已合入的 arch22 实现为**两段式 AIV 流水**（`chpmv_host.cpp` 文件头与 `LaunchChpmvTwoStage` L81-130）：

| 变体 | 触发条件 | 职责 | kernel |
|---|---|---|---|
| quick return | n=0（结构校验后）；或 alpha=0 且 beta=1 | host 直接返回 | 0 |
| scale-only | alpha=0 且 beta≠1 | 仅按 incy 步长做 `y = beta*y` | `chpmv_scale_kernel_do` |
| 两段式主路径 | 其余 | accum：按分核计算 packed×x 的每核部分和入 workspace；reduce：归约部分和并按 alpha/beta 组装写回 y | `chpmv_accum_kernel_do` → `chpmv_reduce_kernel_do` |

数据流（主路径）：

```text
GM: AP(packed), x(步长 incx) ─[AIV accum：每核负责若干输出行的部分和，行分块 1024]─> GM workspace（每核 nPad×2 floats）
                                                                                              │
GM: y_old(步长 incy) ────────────[AIV reduce：归约部分和，y = alpha·(·) + beta·y_old，按 incy 写回]────┴──> GM: y
```

workspace：每核 `nPad×2×sizeof(float)`（nPad = CeilAlign(n,8)），块数取 min(n, AIV 核数, workspace 容量)（`chpmv_host.cpp` L87-95）；不足返回 `EXECUTION_FAILED`。

#### host 侧设计

**分核策略**：accum 按输出行分块（每块 ≤ `CHPMV_ARCH22_TILE_ROWS`=1024 行），块数与 AIV 核数、workspace 容量取小；reduce 按 CeilDiv(n, 1024) 分块。

**数据分块和内存优化策略**：packed 列下标按 uplo 闭式计算（`AP[i + j(j+1)/2]` / `AP[i + j(2n−j−1)/2]`），无查找表；x 按步长分块搬运进 UB（`CHPMV_BUF_FLOATS`=2048），负步长在地址映射中取反向；对角项读入时虚部置 0。

**tilingkey 规划策略**：无 tilingkey 宏分支，host 经 `ChpmvTilingData`（n/nPad/uplo/accumBlocks/reduceBlocks/tileRows/alphaReal…/incx/incy，`chpmv_tiling_data.h`）传参，kernel 变体由 host 显式选择（quick return / scale-only / two-stage）。

**数据检测**（校验顺序逐条对齐 §2.5 校验顺序说明）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. n ≥ 0 → 否则 `INVALID_VALUE`
3. uplo ∈ {UPPER, LOWER} → 否则 `INVALID_ENUM`
4. incx ≠ 0 且 incy ≠ 0 → 否则 `INVALID_VALUE`
5. alpha、beta 非空 → 否则 `INVALID_VALUE`
6. **n=0 quick return**（此时 AP/x/y 可空）
7. AP/x/y 非空（n>0）→ 否则 `INVALID_VALUE`
8. alpha=(0,0) 且 beta=(1,0) → SUCCESS；beta=(0,0) 不读 y 旧值；alpha=0 且 beta≠1 → 仅 beta 缩放

**已合入实现与任务书口径的差异（本任务修正）**：

| 位置 | 现状 | 偏离条款 |
|---|---|---|
| `chpmv_host.cpp` L42-44（`ValidateChpmvInput`） | n==0 的 quick return 位于 uplo、incx/incy、alpha/beta 校验**之前** | §2.5"结构参数在 n=0 时同样校验，非法取值仍返回对应错误"；任务包负向用例含"n=0 叠加非法 uplo / incx=0 / alpha-beta 为空"（期望对应结构错误） |

#### kernel 侧设计

```text
[AIV accum] Init   ：解析 tiling → 求本核输出行区间与 workspace �位
            Process：packed 列段与 x 段按 2048 floats 分块搬入 UB → 复数乘累加 → 部分和写 workspace
[AIV reduce] Init   ：解析 tiling → 求本核输出区间
            Process：归约各核部分和 → y = alpha·(·) + beta·y_old → 按 incy 步长写回
[AIV scale ] Process：uplo 无关，按 incy 步长 y = beta*y
```

文件落位（已合入，本任务仅按口径微调）：

| 文件 | 内容 |
|---|---|
| `blas/spmv/arch22/chpmv_host.cpp` | 入口 `aclblasChpmv`：校验、分派、workspace、发射 |
| `blas/spmv/arch22/chpmv_kernel.cpp/.h` | accum / reduce / scale 三个 AIV kernel |
| `blas/spmv/arch22/chpmv_tiling_data.h` | `ChpmvTilingData` 与常量 |
| `test/spmv/chpmv/arch22/` | `chpmv_test.cpp` + CSV（复用顶层 `chpmv_golden.h`/`chpmv_param.h`） |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms`/0.8，括号内为 `gpu_ms`）：

| 代表 case | uplo | n | 允许耗时 |
|---|---|---|---|
| chpmv-perf-195 | UPPER | 4173 | 0.94 ms（0.75） |
| chpmv-perf-196 | LOWER | 2950 | 0.59 ms（0.47） |
| chpmv-perf-199 | LOWER | 1647 | 0.16 ms（0.12） |
| chpmv-perf-198 | LOWER | 923 | 0.09 ms（0.07） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| 两段式（accum 部分和 + reduce 归约） | 带宽型 BLAS-2：packed 读量 n(n+1)/2 ×8 B 为主要开销，两段式让 packed 每元素只读一次 | 已合入实现沿用 |
| 每核 workspace 部分和（免原子竞争） | accumBlocks 与 workspace 容量取小，reduce 单读多源 | 沿用 |
| 大 n（4173）的单 case 预算 0.94 ms | packed 读约 4173×4175/2×8 B ≈ 69.6 MB → 需 ≥74 GB/s 有效带宽 | 待 `msprof` 实测裁决 |
| 未采用：Cube 通道 | BLAS-2 带宽型，Cube 无收益 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练/推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练/推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| 昇腾 950PR / 950DT（arch35） | 已有实现（`blas/spmv/arch35/chpmv_*`），不在本任务范围 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验；不对 Device 张量做元素级扫描；Hermitian 约束由调用方保证（对角虚部非零输入的虚部被忽略）。
- 非连续 Tensor：x/y 仅支持 incx/incy 步长语义（可负），AP 为 packed 无间隙布局；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，n 与步长为运行时入参。
- 原地与视图语义：y 原地覆写，不返回视图；y 可写区域与 x/AP 可读区域不得重叠。
- 确定性计算：不要求；归约顺序由实现按形状选择。
- 空 Tensor：n=0 合法 no-op（结构参数先校验，不豁免）。
- 0 维 Tensor：不涉及；向量经指针配合 n、incx/incy 描述。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 的实部/虚部分量记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。按 FLOAT32 分量档：rtol=atol=2^-13（1.2e-4），max_abs_error 限值 1e-2 或 32 ULP | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | `cblas_chpmv`（Netlib `chpmv`，CblasColMajor）单精度复数输出直接作为比对基准；实部、虚部分别比对 | 任务书 §3.1、§3.2 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8 | 任务书 §3.3.2 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均；GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3.1、§3.3.4 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

用例来自任务包 `test_cases/chpmv_test.csv`（固定随机种子，逐条可复现），精度 1000 条 + 性能 200 条：

| 类别 | 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础 | TC_L0 | 4 | uplo × 小尺寸 |
| 方阵尺寸 | TC_SQ | 72 | 尺寸池（0→8192，含 4095/4096/8191/8192） |
| 中等覆盖 | TC_CV | 16 | 中等 n |
| 标量 | TC_AB | 16 | 复数 alpha/beta 特殊值 (0,0)/(1,0)/(0,1)/±1e4 与随机值 |
| 填充值域 | TC_FL | 18 | 均匀/正态/全零/交替/极端/Inf/NaN |
| 步长 | TC_INC | 72 | ±1/±2/±3 与大步长 {7,8,16,31,64,127}，含负步长 |
| 边界负向 | TC_ED | 28 | n=0 叠加非法结构参数、空指针、零步长、非法枚举、负维度、beta=0+C NaN |
| 特殊 packed | TC_ST | 8 | 对角虚部非零输入、Hermitian 隐含三角场景 |
| 扩展精度 | TC_EX | 766 | n × uplo × 步长 × 标量采样 |
| 性能 | TC_PF | 200 | 单位步长为主，含少量正大步长与负步长；与基线逐条对应 |

- 测试工程：`test/spmv/chpmv/arch22/` 已有 `chpmv_test.cpp` + `chpmv_test.csv`（复用顶层 `chpmv_golden.h`/`chpmv_param.h`/`chpmv_test_utils.h`）；任务包 CSV 安装到 `test/spmv/chpmv/<arch>/chpmv_test.csv` 后由 `chpmv_test` 加载。
- 复现入口：`bash build.sh --ops=chpmv --soc=ascend910b3` → `build/test/spmv/chpmv/arch22/chpmv_test`；任务包 `verify_accuracy.py` / `verify_performance.py` 按 README 步骤执行。
- 负向用例重点：`n=0` 叠加非法结构参数（非法 uplo、incx=0、alpha/beta 为空）期望返回对应结构错误——与本设计"校验顺序对齐项"直接对应。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **零改动**：`aclblasChpmv` 声明已存在（约 L161）。
- 对已合入 arch22 实现的改动仅一处校验顺序（n=0 提前返回移到结构校验之后）：合法调用的数值结果不变；返回码变化的仅有"n=0 且 uplo/步长/标量指针非法"（原 SUCCESS → 对应结构错误码）一类负向场景，以对齐任务书 §2.5 与 cuBLAS 为目的，用 `TC_ED` 用例回归。
- `test/spmv/chpmv/arch35/` 与其他算子零影响；`blas/spmv/README.md` 的 `aclblasChpmv` 章节（L207）按核对结果补充 arch22 说明。


