# aclblasChpr2 算子设计文档

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/spr2/arch22/`
> 任务书：`aclblasChpr2_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasChpr2 算子开发（A2A3）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/spr2/arch22/`；测试 `test/spr2/chpr2/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

`aclblasChpr2` 实现单精度复数（complex64）Hermitian 压缩存储秩-2 更新：`AP = alpha·x·y^H + conj(alpha)·y·x^H + AP`。n×n Hermitian 矩阵以 packed 格式存放于 AP（仅 uplo 指定的一半三角，n(n+1)/2 个复数元素，无前导维），两项贡献互为共轭转置、结果天然 Hermitian，对角虚部计算后强制置 0。

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasChpr2 | cuBLAS `cublasChpr2`｜Netlib `chpr2.f` | 复数 Hermitian packed 秩-2 更新 |

### 业界对标

| 项 | cuBLAS cublasChpr2 | 本算子 |
|---|---|---|
| 参数序列 | handle/uplo/n/alpha/x/incx/y/incy/AP | 逐参数一致（§2.3、§3.5） |
| packed 布局 | 列主序逐列无间隙打包 | 同：UPPER `AP[i + j(j+1)/2]`；LOWER `AP[i + j(2n−j−1)/2]`（§2.1） |
| Hermitian 语义 | 两项互为共轭转置，对角虚部置 0 | 同 |
| 步长 | incx/incy 任意非零（含负；`INT_MIN` 除外） | 同；0 或 `INT_MIN` → `INVALID_VALUE` |
| n=0 / alpha=0 | quick return，不豁免结构校验 | 同（§2.5 校验顺序说明） |
| 非法 uplo 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.4） |

### 工程形态

- 工程模式：Kernel 直调（Host C API 校验/分派/tiling，经 handle 绑定的 stream 直调 NPU kernel，§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| arch22 实现 | **已合入**（单 kernel AIV 实现） | `blas/spr2/arch22/chpr2_host.cpp`(109 行)、`chpr2_kernel.cpp`(388 行)、`chpr2_kernel.h`、`chpr2_tiling_data.h`；提交 `87a95e9`（2026-11-23）、修复 `55ef791`（2026-11-25，kernel_do 返回发射状态） | 按任务书口径核对并补齐差异（见 host 侧设计），测试用例对齐 |
| arch35 参考 | 已有 | `blas/spr2/arch35/chpr2_*` | 参考口径 |
| 公开声明 | 已存在 | `include/cann_ops_blas.h` 约 L515 | 零公共头改动 |
| 测试工程 | arch22 已有 | `test/spr2/chpr2/arch22/chpr2_test.cpp` 等 | 用例按任务包口径对齐扩展 |
| README | 章节已有 | `blas/spr2/README.md` L123 `### aclblasChpr2` | 核对补充 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/spr2/arch22/chpr2_*`（已合入） | ★ 主对象 | 单 kernel 分核流水（见详细设计）、packed 下标映射、负步长语义、alpha=0 快路 | 仅做任务书口径对齐 |
| `blas/spr2/arch35/chpr2_*` | ◇ 参考 | 同算子 arch35 实现口径 | 不复制代码 |
| `blas/spmv/arch22/chpmv_*` | ◇ 参考 | 同为 packed Hermitian 族：packed 布局与负步长处理口径一致 | 不复制代码 |

### 接口规格

声明已存在于 `include/cann_ops_blas.h`（约 L515），**无需新增或修改**（§2.3）。签名：

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

参数规格（逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| uplo | AP 的存储三角 | attr | int | UPPER / LOWER；其它 → `INVALID_ENUM` | — |
| n | 方阵阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op（结构参数仍校验） | — |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE`；Host/Device 指针位置均支持 | 标量 |
| x | 输入向量 | tensor | COMPLEX64 | n>0 时非空；物理元素 ≥1+(n−1)·\|incx\| | n 个元素 |
| incx | x 步长 | attr | int | 非零且 ≠`INT_MIN`；否则 `INVALID_VALUE`；可为负 | 标量 |
| y | 输入向量 | tensor | COMPLEX64 | n>0 时非空；物理元素 ≥1+(n−1)·\|incy\| | n 个元素 |
| incy | y 步长 | attr | int | 非零且 ≠`INT_MIN`；否则 `INVALID_VALUE`；可为负 | 标量 |
| AP | 输出 Hermitian 压缩矩阵（原地覆写） | tensor | COMPLEX64 | n>0 时非空；对角虚部置 0 | n(n+1)/2 个元素 |

### 数学公式

```text
AP = alpha·x·y^H + conj(alpha)·y·x^H + AP
两项贡献互为共轭转置，求和结果天然 Hermitian；对角元素虚部计算后强制置 0。
packed 下标（列主序，同 chpmv）：
    UPPER：A(i,j)（i ≤ j）存于 AP[i + j(j+1)/2]
    LOWER：A(i,j)（i ≥ j）存于 AP[i + j(2n−j−1)/2]
```

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上提供句柄式接口 `aclblasChpr2`：complex64、UPPER/LOWER packed、任意非零步长（含负，`INT_MIN` 除外）、n ≤ 8192；精度达生态算子开源混合容差标准 FLOAT32 分量档，性能达 GPU 标杆 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | `AP = alpha·x·y^H + conj(alpha)·y·x^H + AP`。秩-2 更新每元素两次复数乘加；第二项与第一项互为共轭转置，可由同一分核同时写 uplo 三角两端（或按块只写 uplo 三角），对角虚部置 0 由 Hermitian 语义内联完成 |
| 2 | 接口原型 | §2.3 签名（9 参数），返回 `aclblasStatus_t`；声明已存在，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | 全 COMPLEX64（AP/x/y/alpha） |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；arch35 已有实现，不在本任务范围 |
| 6 | 精度要求 | §3.2：FLOAT32 分量档 rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32ULP；golden=`cblas_chpr2`（CblasColMajor） |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 + 10 样本 |
| 8 | 代码架构选型 + 可行性 | arch22 已合入单 kernel 分核实现，本任务以口径对齐为主：**已核验两处差异需修正**（① n=0 提前返回先于结构校验，违反 §2.5"n=0 不豁免"；② 非法 uplo 返回 INVALID_VALUE 而非 §2.4 要求的 INVALID_ENUM）。NPU 侧达标性**待实测** |
| 9 | 评测来源 | 任务包 `test_cases/`：精度 1000 条 + 性能 200 条（`gpu_baseline.csv` 200 条全回填），CSV 安装到 `test/spr2/chpr2/<arch>/` 由 `chpr2_test` 加载 |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。实现要点：每输出元素 AP(i,j) 的增量为两次复数乘加——`alpha·x(i)·conj(y(j)) + conj(alpha)·y(i)·conj(x(j))`（对角项额外取实部置虚 0），两个共轭项在同一次遍历中同时累加，避免重复读 x/y。

### 支持数据类型

| 项 | 类型 | 说明 |
|---|---|---|
| AP/x/y | COMPLEX64 | packed 列主序 / 步长向量 |
| alpha | COMPLEX64 | Host/Device 指针位置均支持（§2.5） |
| 计算精度 | FP32 | AIV 向量计算 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥0 | n=0：**结构参数（uplo/incx/incy/alpha）先校验**，通过后 SUCCESS，不引用 AP/x/y、不启动 kernel |
| incx/incy | 非零且 ≠`INT_MIN`（可负） | 0 或 `INT_MIN` → INVALID_VALUE；负步长按反向遍历访问 |
| AP/x/y 指针 | n>0 时非空 | 否则 INVALID_VALUE |

alpha=0 时 AP 不变（quick return，但 x/y/AP 指针仍需非空，§2.1 第 6 条）；AP 可写区域与 x/y 可读区域不得重叠。

## 算子实现

### 实现方案

#### 总体架构与数据流

已合入的 arch22 实现为**单 kernel 分核流水**（`chpr2_host.cpp` `LaunchChpr2Kernel` L59-87）：

| 变体 | 触发条件 | 职责 | kernel |
|---|---|---|---|
| quick return | n=0（结构校验后）；或 alpha=0 | host 直接返回 | 0 |
| 主路径 | 其余 | 单 kernel：按分核计算两次复数乘加并原地累加进 AP 的 uplo 三角，对角虚部置 0 | `chpr2_kernel_do` |

数据流：

```text
GM: x(步长 incx), y(步长 incy) ─[AIV 单 kernel：numBlocks = min(n, AIV 核数) 分核]─>
      每核负责若干输出列/行块：两次复数乘加（alpha·x·ȳ + conj(alpha)·y·x̄）→ 原地累加进 AP uplo 三角
GM: AP(读旧值 + 写回，packed；对角虚部置 0)
```

workspace：无（AP 原地读写，x/y 直读）。

#### host 侧设计

**分核策略**：`numBlocks = min(n, GetAivCoreCount())`，每核负责 n 的连续输出块；块内按 uplo 三角遍历。

**数据分块和内存优化策略**：packed 下标按 uplo 闭式计算（同 chpmv 口径）；x/y 按步长搬运进 UB，负步长在地址映射中取反向；写回仅 uplo 三角，对角项虚部置 0。

**tilingkey 规划策略**：无 tilingkey 宏分支，host 经 `Chpr2TilingData`（n/uplo/numBlocks/alphaReal…/incx/incy，`chpr2_tiling_data.h`）传参；分支仅有 quick return 与主路径。

**数据检测**（校验顺序逐条对齐 §2.5 校验顺序说明）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. n ≥ 0 → 否则 `INVALID_VALUE`
3. alpha 非空 → 否则 `INVALID_VALUE`
4. uplo ∈ {UPPER, LOWER} → 否则 `INVALID_ENUM`
5. incx/incy ≠ 0 且 ≠ `INT_MIN` → 否则 `INVALID_VALUE`
6. **n=0 quick return**（此时 x/y/AP 可空）
7. x/y/AP 非空（n>0）→ 否则 `INVALID_VALUE`
8. alpha=(0,0) → SUCCESS（AP 不变）

**已合入实现与任务书口径的差异（本任务修正）**：

| 位置 | 现状 | 偏离条款 |
|---|---|---|
| `chpr2_host.cpp` L97-99（入口） | n==0 的 quick return 位于 alpha/uplo/步长校验**之前** | §2.5"结构参数在 n=0 时同样校验，非法取值仍返回对应错误"；任务包负向用例含 n=0 叠加非法结构参数 |
| `chpr2_host.cpp` L37-39（`ValidateChpr2Body`） | 非法 uplo 返回 `INVALID_VALUE` | §2.4 要求非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM` |

#### kernel 侧设计

```text
[AIV 主 kernel]
 Init   ：解析 tiling → 求本核输出块（uplo 三角内）
 Process：x/y 按步长分块搬入 UB → 逐元素两次复数乘加（alpha·x·ȳ + conj(alpha)·y·x̄）
          → 读 AP 旧值累加 → 写回 uplo 三角（对角虚部 SetValue 置 0）
```

文件落位（已合入，本任务仅按口径微调）：

| 文件 | 内容 |
|---|---|
| `blas/spr2/arch22/chpr2_host.cpp` | 入口 `aclblasChpr2`：校验、分派、发射 |
| `blas/spr2/arch22/chpr2_kernel.cpp/.h` | 单 kernel AIV 实现 |
| `blas/spr2/arch22/chpr2_tiling_data.h` | `Chpr2TilingData` 与常量 |
| `test/spr2/chpr2/arch22/` | `chpr2_test.cpp` + CSV（复用顶层 golden/param） |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms`/0.8，括号内为 `gpu_ms`）：

| 代表 case | uplo | n（incx） | 允许耗时 |
|---|---|---|---|
| chpr2-perf-199 | LOWER | 2566 (1) | 0.38 ms（0.31） |
| chpr2-perf-197 | LOWER | 2355 (1) | 0.27 ms（0.21） |
| chpr2-perf-198 | LOWER | 714 (1) | 0.04 ms（0.03） |
| chpr2-perf-196 | LOWER | 933 (31) | 0.06 ms（0.05） |
| chpr2-perf-195 | UPPER | 308 (1) | 0.02 ms（0.02） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| 单 kernel 原地累加（无 workspace） | 秩-2 更新写量与读量同阶（AP 读+写 n(n+1)/2×8 B×2），单 pass 最优 | 已合入实现沿用 |
| 两个共轭项同遍历累加 | x/y 每元素只读一次 | 沿用 |
| 大 n（2566）单 case 预算 0.38 ms | AP 流量 ≈ 2566²/2×8 B×2 ≈ 52.7 MB → 需 ≥138 GB/s 有效带宽 | 待 `msprof` 实测裁决 |
| 未采用：两段式（如 chpmv） | 秩-2 输出无归约维度，直接原地累加即可 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练/推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练/推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| 昇腾 950PR / 950DT（arch35） | 已有实现（`blas/spr2/arch35/chpr2_*`），不在本任务范围 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验；不对 Device 张量做元素级扫描；Hermitian 约束由调用方保证。
- 非连续 Tensor：x/y 仅支持 incx/incy 步长语义（可负，`INT_MIN` 除外），AP 为 packed 无间隙布局；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，n 与步长为运行时入参。
- 原地与视图语义：AP 原地覆写，不返回视图；AP 可写区域与 x/y 可读区域不得重叠。
- 确定性计算：不要求；归约顺序由实现按形状选择。
- 空 Tensor：n=0 合法 no-op（结构参数先校验，不豁免）；alpha=0 时 AP 不变。
- 0 维 Tensor：不涉及；向量经指针配合 n、incx/incy 描述。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 的实部/虚部分量记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。按 FLOAT32 分量档：rtol=atol=2^-13（1.2e-4），max_abs_error 限值 1e-2 或 32 ULP | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | `cblas_chpr2`（Netlib `chpr2`，CblasColMajor）单精度复数输出直接作为比对基准；实部、虚部分别比对 | 任务书 §3.1、§3.2 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8 | 任务书 §3.3 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均；GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

用例来自任务包 `test_cases/chpr2_test.csv`（固定随机种子，逐条可复现），精度 1000 条 + 性能 200 条：

| 类别 | 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础 | TC_L0 | 4 | uplo × 小尺寸 |
| 方阵尺寸 | TC_SQ | 68 | 尺寸池（0→8192） |
| 中等覆盖 | TC_CV | 16 | 中等 n |
| 标量 | TC_AB | 16 | 复数 alpha 特殊值 (0,0)/(1,0)/(0,1)/±1e4 与随机值 |
| 填充值域 | TC_FL | 18 | 均匀/正态/全零/交替/极端/Inf/NaN |
| 步长 | TC_INC | 72 | ±1/±2/±3 与大步长 {7,8,16,31,64,127}，含负步长与 `INT_MIN` 负向 |
| 边界负向 | TC_ED | 29 | n=0 叠加非法结构参数、空指针、零步长/`INT_MIN`、非法枚举、负维度 |
| 特殊 packed | TC_ST | 8 | 对角虚部置 0、Hermitian 隐含三角场景 |
| 扩展精度 | TC_EX | 769 | n × uplo × 步长 × 标量采样 |
| 性能 | TC_PF | 200 | 单位步长为主，含少量正大步长与负步长；与基线逐条对应 |

- 测试工程：`test/spr2/chpr2/arch22/` 已有 `chpr2_test.cpp` + `chpr2_test.csv`；任务包 CSV 安装到 `test/spr2/chpr2/<arch>/chpr2_test.csv` 后由 `chpr2_test` 加载。
- 复现入口：`bash build.sh --ops=chpr2 --soc=ascend910b3` → `build/test/spr2/chpr2/arch22/chpr2_test`；任务包 `verify_accuracy.py` / `verify_performance.py` 按 README 步骤执行。
- 负向用例重点：`n=0` 叠加非法结构参数期望返回对应结构错误（对应本设计两处对齐项）；非法 uplo 期望 `INVALID_ENUM`。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **零改动**：`aclblasChpr2` 声明已存在（约 L515）。
- 对已合入 arch22 实现的改动仅两处校验口径（n=0 校验顺序、uplo 返回码）：合法调用的数值结果不变；返回码变化仅负向场景（n=0 叠加非法结构参数、非法 uplo），以对齐任务书 §2.4/§2.5 与 cuBLAS 为目的，用 `TC_ED` 用例回归。
- `test/spr2/chpr2/arch35/` 与其他算子零影响；`blas/spr2/README.md` 的 `aclblasChpr2` 章节（L123）按核对结果补充 arch22 说明。

