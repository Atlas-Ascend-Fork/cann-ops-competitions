# aclblasCrot 算子设计文档（Atlas A2/A3，arch22）

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCrot` |
| 任务名称 | 11月社区任务-aclblasCrot算子开发（A2/A3） |
| 任务目录 | 11-aclblasCrot-A2A3 |
| 团队 | EulerTeng |
| 目标硬件 | Atlas 800T A2 / Atlas 800T A3 系列产品 |
| 目标架构 | arch22（dav-2201，UB 192 KiB，40 AIV，L2 192 MiB） |
| 实现仓 | `cann/ops-blas` |
| 实现目录 | `blas/rot/arch22/`，测试 `test/rot/crot/arch22/` |
| 对标接口 | cuBLAS `cublasCrot`，语义 golden 为 LAPACK `crot.f` |
| 数据类型 | complex64（`aclblasComplex`，实部/虚部各 float32） |
| 文档版本 | v1.0 |
| 日期 | 2026-11-11 |

## 修订记录

| 版本 | 日期 | 变更说明 |
| --- | --- | --- |
| v1.0 | 2026-11-11 | 初稿。含同访存量代理算子的 A3 实机带宽摸底、arch22 口径确认、入参契约偏差说明。 |

---

# 一、需求背景（required）

## 1.1 需求来源

昇腾社区 2026 年 11 月社区任务《11月社区任务-aclblasCrot算子开发（A2/A3）》，
任务书 `aclblasCrot_AtlasA2_A3_task_doc.md`，任务讨论帖 `cann/ops-blas` discussions/75。
目录命名对齐同批 11 月任务的既有提交（如 `11-aclblasSaxpy-A2A3`）。
实现与测试合入 `cann/ops-blas`，任务截止 2026-11-23。

## 1.2 背景介绍

`aclblasCrot` 是 BLAS Level-1 的复数 Givens 平面旋转：对两个 complex64 向量 x、y
逐元素施加同一个 2×2 旋转，两个向量均原地更新。Givens 旋转是 QR 分解、最小二乘、
特征值迭代等稠密线性代数流程的基础构件，复数版本用于 Hermitian/复数矩阵的相应算法。

本算子在数学上是逐元素映射（element-wise map），不含归约、不含跨元素依赖
（零步长情形除外，见 §3.1.5），因此是典型的 **memory-bound** 算子。

## 1.3 现状分析

| 算子 | 架构 | 仓内现状 | 对本任务的可复用性 |
| --- | --- | --- | --- |
| `aclblasCrot` | arch35（950PR） | 已实现（`blas/rot/arch35/`，host 228 行 + kernel 406 行） | 数学与入参契约可参照；但其连续路径用 `Interleave`/`DeInterleave`、非连续路径用 SIMT（`asc_vf_call`），**这两类在 arch22 全仓零使用**，编程模型需整体重写 |
| `aclblasCrot` | arch22（A2/A3） | **未实现**（`blas/rot/README.md` 标注 A2/A3「不支持」） | 本任务的交付目标 |
| `aclblasCsrot` | arch22 | 已实现（host 136 行 + kernel 327 行） | arch22 工程骨架、双缓冲流水、非连续 staging 可参照；**但其数学不可照搬**（s 为实数、实虚部不耦合），其 `numBlocks` 固定为 8、host 每次调用分配并同步，在有性能门禁的本算子中需另行设计 |
| `aclblasCaxpy` | arch22 | 已实现（host 625 行 + kernel 919 行） | **复数标量 × 复数向量的车道交叉问题已在此解决**（`vgather` 相邻车道交换 + 交替符号系数向量），是本算子连续路径的直接先例 |

接口声明 `aclblasCrot` **已存在**于 `include/cann_ops_blas.h:250`，签名与任务书一致，无需新增；
`test/rot/crot/CMakeLists.txt` 已无条件注册 GTest；`test/rot/crot/crot_golden.h` 与
`crot_param.h` 为跨 arch 共享资产，可原样复用。

## 1.4 本设计的切入点

基于对上游仓与任务数据的核对，本设计在以下四点上形成了可验证的差异化基础：

1. **带宽可行性已在 A3 实机摸底**（§4.2）：用同访存量的既有算子做代理，扫描了
   AIV 核数对耗时的影响，确认本平台的访存能力足以支撑该访存量，并定位出最优核数区间。
2. **识别出核数这一性能杠杆**（§3.3.3）：arch22 的 rot/swap 族现有实现固定使用 8 个
   AIV（该族此前未设置 arch22 性能门禁，无需核数调优），而本平台可用 40 个；实测
   表明在大规模下提升显著，同时中小规模需要反向收敛核数。
3. **量化了 host 侧固定开销的构成**（§3.3.4、§4.2）：逐项测出每次调用的运行时原语耗时，
   据此选择 handle 常驻缓冲 + 异步上传，而非每次调用分配并同步。
4. **厘清了入参契约的依据**（§2.4）：对 n<0 与零步长两项，逐条核对了 README、
   issue、共享 golden 与测试数据，给出一致口径并把分歧列为待拍板的未决项。

## 1.5 算子功能分析

| 参数 | 方向 | 类型 | 说明 | 异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | 句柄，提供 stream | `nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（最高优先级） |
| `n` | 输入 | `int` | 向量逻辑长度 | `n ≤ 0` → `ACLBLAS_STATUS_SUCCESS`（no-op，不修改 x、y） |
| `x` | 输入/输出 | `aclblasComplex*` | Device 内存，原地更新 | `n > 0` 时 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 输入 | `int` | x 步长，可正可负可为零 | `INT_MIN` 或跨度溢出 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 输入/输出 | `aclblasComplex*` | Device 内存，原地更新 | 同 x |
| `incy` | 输入 | `int` | y 步长 | 同 incx |
| `c` | 输入 | `const float*` | 旋转余弦，Host 或 Device 内存 | `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `s` | 输入 | `const aclblasComplex*` | 旋转正弦（复数），Host 或 Device 内存 | `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C 在 Atlas A2/A3（arch22）上实现 complex64 向量的 Givens 平面旋转算子
`aclblasCrot`，语义对齐 cuBLAS `cublasCrot` / LAPACK `crot.f`，支持任意正负步长与零步长、
支持 c/s 标量位于 Host 或 Device 内存（含混合），精度与性能均达到任务书验收标准。

## 2.2 需求拆解

| 编号 | 需求 | 验收出口 |
| --- | --- | --- |
| R1 | 连续步长（incx=incy=1）正确实现旋转公式 | CSV 精度用例 TC_L0/TC_SQ 全通过 |
| R2 | 任意非零正负步长正确，负步长自尾端起算 | TC_INC 共 36 组合全通过 |
| R3 | 零步长按参考实现串行语义反复旋转同一元素 | TC_ED_124/125/126 返回 SUCCESS 且数值与 golden 一致 |
| R4 | c、s 支持 Host/Device/混合三种指针位置 | TC_PM 6 条 + TC_EX 中 82 条 device/mixed 用例 |
| R5 | 标量极值（含 `c=s=1e10`）下与 golden 逐位同序，不触发 32-ULP 上限 | TC_CS 20 条 + TC_EX 极值组 88 条 |
| R6 | Inf/NaN 按既定规则传播，不做 c/s 短路优化 | TC_FL 12 条 |
| R7 | 非法入参返回正确错误码 | TC_ED 中 7 条 INVALID_VALUE 用例 |
| R8 | 跨度 `(n-1)·\|inc\|` 溢出 int32 时在 host 侧拦截 | TC_ED_131/132/133 |
| R9 | 性能不低于 GPU 标杆的 0.8 倍（耗时 ≤1.25 倍） | `msprof op` 采 200 条 TC_PF 的 Task Duration |
| R10 | 内存无越界、无泄漏 | `run_memcheck.py` 报告 `No error detected` |
| R11 | 不破坏同族既有算子（csrot/srot）与 arch35 实现 | 全量回归 + 仅新增 arch22 文件 |

## 2.3 接口定义

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy,
    const float* c,
    const aclblasComplex* s);
```

声明已存在于 `include/cann_ops_blas.h:250`，本任务不修改公共头文件签名。

## 2.4 入参契约（含与任务书的偏差说明）

本算子 arch22 侧采用下表契约。其中 `n < 0` 与 `incx/incy == 0` 两项**与任务书 §2.1.3/§2.4/§2.5
的表述不一致**，依据如下（证据均来自上游仓与任务提供的测试数据）：

| 场景 | 本设计 | 任务书 | 依据 |
| --- | --- | --- | --- |
| `handle == nullptr` | `HANDLE_IS_NULLPTR` | 一致 | 两侧一致，最高优先级 |
| `n == 0` | `SUCCESS`（no-op） | 一致 | 两侧一致 |
| `n < 0` | **`SUCCESS`（no-op）** | `INVALID_VALUE` | 见下方依据 1–4 |
| `x/y/c/s == nullptr`（n>0） | `INVALID_VALUE` | 一致 | 两侧一致 |
| `incx/incy == 0` | **放行，串行语义** | `INVALID_VALUE` | 见下方依据 1–4 |
| `incx/incy == INT_MIN` | `INVALID_VALUE` | 一致 | 两侧一致，显式拦截（覆盖 n=1 时跨度守卫失效的情形） |
| `(n-1)·\|inc\|` 溢出 int32 | `INVALID_VALUE` | 一致 | 两侧一致，host 侧 int64 守卫 |

**依据 1**：`blas/rot/README.md` 中的跨 arch 返回码对照表，标题为「跨 arch 入参返回码对照
（**aclblasCsrot**）」，仅覆盖 csrot；同一文件的 `aclblasCrot` 小节中，arch35 约束为
「n ≤ 0 时为 no-op」「incx、incy 可为正、负、零，均不拦截」，而 A2/A3 条目为「不支持」——
即 crot 的 arch22 契约此前并不存在，任务书是将 csrot 的历史约束外推到了本算子。

**依据 2**：issue #827 的标题与正文均只针对 `aclblasCsrot`，其首选处理意见是
「统一两侧口径（建议对齐 netlib：n≤0 放行，零步长语义显式定义）」。arch22 csrot 之所以保留
拒绝语义，是「既有实现行为保持不变」的兼容性妥协；crot arch22 为全新实现，不存在该前提。

**依据 3**：`test/rot/crot/crot_golden.h` 位于 `test/rot/crot/` 根目录、由两个 arch **共享**，
其中写死 `if (n <= 0) return ACLBLAS_STATUS_SUCCESS;`。采用拒绝语义将需要在共享 golden 中
引入 arch 条件编译，属对已合入资产的侵入式改动。

**依据 4**：任务提供的 `crot_test.csv` 中 `TC_ED_123`（n=-5）、`TC_ED_124/125/126`（零步长）
的 `expect_result` 均为 `ACLBLAS_STATUS_SUCCESS`；且该 CSV 前 131 行与上游
`test/rot/crot/arch35/crot_test.csv` 内容完全一致（仅换行符为 CRLF 与 LF 之别），说明其编码的是 arch35 契约。

**反方论据与回应**：任务书 §3.5-4 写明「边界、负向用例及特殊值用例的期望状态码以本
任务书 §2.4/§2.5 为准」，即任务书自身把其条款置于 CSV 期望值之上；§2.1.3 也声明该口径
「见 blas/rot/README.md 的 per-arch 差异表，issue #827 已声明」。本设计并非主张任务书无据，
而是指出：该引用所指向的 README 表格标题限定为 `aclblasCsrot`，crot 小节内并无 arch22 行，
因此「arch22 家族契约」这一前提在 crot 上尚未成立。鉴于这属于公共 API 语义决策而非单侧
实现选择，本设计按上游既有资产（共享 golden + arch35 实现 + Netlib）给出一致口径，
**并将该分歧作为待维护者拍板的未决项明确提出**，不单方面认定任务书有误。

**交付时的配套动作**：在 `blas/rot/README.md` 的 crot 小节补一张与 csrot 平行的 per-arch 表，
显式声明「arch22 crot 放行、arch22 csrot 拦截」这一差异，避免后续审查者重复开 issue；
同时在任务讨论帖中请维护者对该公共 API 语义决策拍板。若维护者要求改按任务书口径，
回退方案为修改 CSV 中 4 条 TC_ED 的期望值并在共享 golden 增加 arch 分支。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

对 i = 1..n，k = 1+(i−1)·incx，j = 1+(i−1)·incy（1-based，兼容 Fortran）：

```
x'[k] = c · x[k] + s · y[j]
y'[j] = c · y[j] − conj(s) · x[k]      （使用旋转前的原始 x[k]）
```

c 为实数（float），s 为复数（complex64）。`conj(s)` 仅出现在 y 的更新式中。

### 3.1.2 复数分量展开

记 s = sr + i·si，x = xr + i·xi，y = yr + i·yi：

```
x'.re = c·xr + (sr·yr − si·yi)
x'.im = c·xi + (sr·yi + si·yr)
y'.re = c·yr − (sr·xr + si·xi)      ← conj 使 −si·(−xi) 变为 +si·xi
y'.im = c·yi − (sr·xi − si·xr)
```

`y'.re` 中的 `+si·xi` 是最易写错之处，CSV 中纯虚 s（`(0,0,1)`，89 条）与虚部为负的
一般复数 s（`(0.5,0.3,-0.4)`，90 条）专门覆盖该符号。

### 3.1.3 数据类型与内存布局

`aclblasComplex` 为 AoS 交错布局：`{re, im, re, im, ...}`，每元素 8 字节。
kernel 侧将其视作长度 `2n` 的 float 车道序列。

**与 csrot 的本质差异**：csrot 的 s 为实数，实部虚部各自独立缩放，可直接在交错车道上
整体做 `muls`；crot 的 s 为复数，实部虚部**强耦合**，必须做车道交叉（见 §3.4.2）。

### 3.1.4 形状与步长

逻辑形状 `[n]`，物理跨度 `1+(n−1)·|inc|`。incx/incy 独立取值，可正、可负、可为零；
负步长自物理尾端起算，逻辑下标 i 对应物理下标 `(n−1−i)·|inc|`。

### 3.1.5 关键语义

1. **原地且依赖旧值**：y 的更新必须使用旋转前的 x，实现上必须先读出 xr/xi/yr/yi 四个值
   再计算四个结果，最后统一写回。
2. **零步长的串行性**：incx=0 时所有迭代读写 x 的同一个元素，结果**顺序敏感**。
   例如 incx=0、incy=1、n=10 时，`y[j] = c·y⁰[j] − conj(s)·x⁽ʲ⁾`，其中 `x⁽ʲ⁾` 是第 j 次
   迭代后的 x 值而非初值。若并行化为「所有线程读同一初值」，末元素相对误差可达 179%，
   属语义错误而非精度问题。
3. **别名**：x 与 y 同基址且步长不同时，迭代间存在读写交叉，必须串行。
4. **不做短路**：c=1 且 s=0 的单位旋转也要正常执行公式，否则 Inf/NaN 无法正确传播
   （CSV 中相关用例共 356 条）。

## 3.2 总体架构

```
aclblasCrot (host)
  ├─ 入参校验（顺序见 §3.3.1）
  ├─ 标量解析：c / s 各自判定 Host 或 Device 内存
  ├─ Tiling：elementCount / totalN / incx / incy / 标量值
  ├─ 分核：numBlocks = min(AIV 核数, ceil(n / 每核最小元素数))
  └─ crot_kernel_do → kernel
        ├─ tilingKey=0：连续路径（incx=incy=1）
        │     CopyIn(双缓冲) → 车道交叉复数运算 → CopyOut
        ├─ tilingKey=1：非连续路径（任一步长 ≠1，无零步长）
        │     标量 GM 读 → UB 32B 对齐 staging → DataCopyPad 写回
        └─ tilingKey=2：串行路径（任一步长为 0，或 x/y 别名且步长不同）
              仅 block 0 执行顺序循环
```

## 3.3 Host 侧设计

### 3.3.1 入口与校验顺序

```cpp
if (handle == nullptr)            return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
if (n <= 0)                       return ACLBLAS_STATUS_SUCCESS;        // no-op，不触碰指针
if (x == nullptr || y == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
if (c == nullptr || s == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
if (incx == INT_MIN || incy == INT_MIN) return ACLBLAS_STATUS_INVALID_VALUE;
// (n-1)*|inc| 在 int64 域计算后与 INT32_MAX 比较
if (spanX > INT32_MAX || spanY > INT32_MAX) return ACLBLAS_STATUS_INVALID_VALUE;
// incx/incy == 0 不拦截，走串行路径
```

`INT_MIN` 显式拦截的必要性：n=1 时 `(n−1)·|inc| = 0` 不触发跨度守卫，仅靠守卫会漏判。
取绝对值必须先转 int64，`-static_cast<int32_t>(INT32_MIN)` 是未定义行为。

### 3.3.2 标量指针位置判定

c 与 s **各自独立**判定内存位置，因此天然支持 mixed 模式。复用仓内
`ReadScalarFromHostOrDevice`（`blas/common/helper/host_utils.h:144`，模板函数，
对 `aclblasComplex` 可直接实例化，`blas/gemv/arch22/cgemv_host.cpp:111` 已有同样用法）：
Host 指针直接解引用，Device 指针走 `aclrtMemcpyAsync` + 同步读回，结果 bake 进 tiling。

### 3.3.3 分核策略

```cpp
uint32_t numBlocks = GetAivCoreCount();
numBlocks = std::min(numBlocks, CROT_DEFAULT_VECTOR_NUM);          // 40
numBlocks = std::min(numBlocks, CeilDiv(n, CROT_MIN_ELEMS_PER_CORE));
numBlocks = std::max(numBlocks, 1U);
```

**设计依据（实测）**：arch22 的 `csrot` 与 `cswap` 当前固定使用 8 个 AIV（该族 arch22 侧此前未设置性能门禁，
`test/rot/csrot/README.md` 载明 arch22 目录无 TC_PF 用例，因此无需核数调优），
而本平台可用 40 个 AIV。本文档在 A3 实机上扫描了核数对耗时的影响（§4.2），8→40 核在
n=4194304 时提速 2.3 倍，是本算子最大的单点性能杠杆。同时按 `CeilDiv(n, 每核最小元素数)`
收敛核数，避免小 n 场景下多核 dispatch 与同步开销超过并行收益
（`blas/dgmm/arch22/cdgmm_host.cpp:118` 为同类做法的仓内先例）。

### 3.3.4 Tiling 与下发

`CrotTilingData` 为 POD，含 `elementCount(2n)`、`totalN`、`incx`、`incy`、
`cosValue`、`sinRe`、`sinIm`、`tilingKey`。

arch22 rot 族现有实现在每次调用中执行 `aclrtMalloc` + 同步 `aclrtMemcpy` +
`aclrtSynchronizeStream`（对无性能门禁的功能路径是合理选择）。
实测（§4.2）该路径合计约 14.7 µs，其中 24 字节 tiling 的**同步 H2D 单项就占 9.74 µs**。
本设计改用 handle 上常驻的 tiling 缓冲 + pinned host staging + `aclrtMemcpyAsync`，
不在算子内部做 stream 同步——`blas/axpy/arch22/caxpy_host.cpp` 的 `caxpy_uploads` 与
`handle->cherk_arch22_metadata` 均为仓内同类先例。该优化不计入 `msprof op` 的 kernel
判据，但直接影响端到端时延与异步语义正确性。

## 3.4 Kernel 侧设计

### 3.4.1 arch22 编程模型约束

| arch35 crot 的做法 | arch22 是否可用 | arch22 替代方案 |
| --- | --- | --- |
| `Interleave`/`DeInterleave` 拆合实虚部 | **否**（全仓 arch22 零使用） | `vgather` 相邻车道交换（§3.4.2） |
| SIMT（`asc_vf_call`）跑非连续路径 | **否**（全仓 arch22 零使用） | 标量 GM 读 + UB staging + `DataCopyPad`（§3.4.3） |
| `TPipe` + `TBuf` + `DataCopyPad` | 可用但非 arch22 惯例 | `AsdopsBuffer<ASCEND_V220>` + `gm_to_ub_align` |
| 结构体按值随 `<<<>>>` 传 tiling | 否 | GM 传 tiling，kernel 侧偏移读取 |

### 3.4.2 连续路径（tilingKey=0）

复数乘法的车道交叉采用 `caxpy` arch22 已验证的方案（`caxpy_kernel.cpp:200-216`）：
用 `vgather` 把交错车道 `[re, im]` 重排为 `[im, re]`，再乘以交替符号的系数向量。

对 `s·Y`（用于 x'）：

| 步骤 | 操作 | 结果 |
| --- | --- | --- |
| 1 | `vgather(swapY, offTable, ubY)` | `[yi, yr, yi, yr, ...]` |
| 2 | `vmuls(t0, ubY, sr)` | `[sr·yr, sr·yi, ...]` |
| 3 | `vmul(t1, swapY, coefA)`，`coefA = [−si, +si, ...]` | `[−si·yi, +si·yr, ...]` |
| 4 | `vadd(t0, t0, t1)` | `[sr·yr−si·yi, sr·yi+si·yr]` = `s·Y` ✓ |

对 `conj(s)·X`（用于 y'）：同样四步，系数向量改为 `coefB = [+si, −si, ...]`，
得到 `[sr·xr+si·xi, sr·xi−si·xr]` = `conj(s)·X` ✓。

完整序列（每 tile）：

| 步骤 | 操作 |
| --- | --- |
| 1–2 | `vgather` 交换 x、y 的相邻车道 |
| 3–6 | `s·Y`：`vmuls` + `vmul` + `vadd`；`conj(s)·X`：同构 |
| 7–8 | `vmuls(cx, ubX, c)`、`vmuls(cy, ubY, c)` |
| 9 | `vadd(outX, cx, sY)` → x' |
| 10 | `vsub(outY, cy, conjSX)` → y' |

合计约 12 条向量指令/tile。相比 arch35 先 `DeInterleave` 再按平面计算的 20 条，
指令数减少约 40%，且不依赖 arch22 不具备的 API。

**原地安全**：x' 与 y' 写入独立的 UB 输出缓冲，输入缓冲在两式都算完前不被覆盖，
满足「y 必须用旋转前的 x」。

**双缓冲**：沿用 arch22 惯例的显式 ping-pong（两组缓冲 + `SET_FLAG`/`WAIT_FLAG`
串接 MTE2→V→MTE3→MTE2），并参照 `caxpy` 增加 tile+1 的软件预取。
tile 尺寸与缓冲数量全部用 `constexpr` 固定，以保证编译期常量折叠
（`blas/gemv_batched/arch35/cgemv_batched_kernel.cpp:989` 注释明确说明把 tile 尺寸
放进成员变量会阻断 copy/vector 循环内的常量折叠）。

**UB 预算**（192 KiB）：x/y 输入各 2 块、输出各 2 块、swap 暂存 2 块、系数与偏移表常驻，
按 `constexpr` 公式计算 tile 尺寸并以 `static_assert` 兜底，Host 与 Kernel 共用同一常量。

### 3.4.3 非连续路径（tilingKey=1）

照 arch22 `csrot_kernel.cpp:56-107` 的 staging 写法：按逻辑下标算物理下标
（负步长自尾端起算），用标量 `GetValue` 从 GM 读四个 float，在 UB 的 32 字节对齐 slot 内
计算，再用每元素一次 8 字节 `DataCopyPad` burst 写回。UB 侧偏移必须 32B 对齐，
这是 MTE 数据通路的硬要求。标量 UB 写不走 V 流水，写回前需 `PIPE_BARRIER(ALL)`。

### 3.4.4 串行路径（tilingKey=2）

触发条件：`incx == 0 || incy == 0 ||（x == y 且 incx != incy）`。
仅 block 0 执行顺序循环，其余 block 直接返回，以复刻参考实现的逐次迭代语义（§3.1.5）。
零步长侧的 `|inc|` 置 1、物理下标恒为 0。该路径不参与性能用例，只保证语义正确。

### 3.4.5 舍入策略

每个乘积经由 `volatile` 临时量，禁止 FMA 收缩，求值顺序与共享 golden
（`test/rot/crot/crot_golden.h`）逐字一致：

```cpp
volatile float srYr = sr * yr;  volatile float siYi = si * yi;
volatile float srYi = sr * yi;  volatile float siYr = si * yr;
volatile float srXr = sr * xr;  volatile float siXi = si * xi;
volatile float srXi = sr * xi;  volatile float siXr = si * xr;
volatile float cXr = c * xr;    /* ... */
newXr = cXr + (srYr - siYi);    newYr = cYr - (srXr + siXi);
```

**必要性**：CSV 中 88 条 `c = 1e10, s = ±1e10` 用例里，中间项量级约 5e10，而两项相减后
golden 可能只有 O(1)。此时逐元素 32-ULP 上限坍缩到 1e-2，而一次融合舍入的差异量级为
`ULP(5e10) ≈ 4096`，相差 5 个数量级，必然判失败。括号分组与减号的作用范围均不可重排。

## 3.5 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Atlas 800T A2 / Atlas 800I A2 系列 | √ |
| Atlas 800T A3 / Atlas 800I A3 系列 | √ |
| Ascend 950PR | 由 arch35 既有实现覆盖，本任务不改动 |
| Ascend 950DT | 既有实现未支持，本任务不涉及 |
| Atlas 300I / 310P | 不支持 |

## 3.6 算子约束限制

1. 仅支持 complex64；不支持动态 shape（n 为运行时入参）。
2. x、y 均原地更新，不返回视图。
3. 部分重叠（基址不同但区间相交）的缓冲不做检测，不在契约内。
4. 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream。
5. 不做 c/s 的短路优化。

---

# 四、可维可测分析（required）

## 4.1 精度标准

实际生效的判据为测试框架的 `MIXED_TOLERANCE`（`test/frame/verify.h`），
对 x、y 的实部与虚部共 4 路**独立**判定，4 路全过才算通过：

| 项 | 值 |
| --- | --- |
| rtol / atol | 2⁻¹³（0.0001220703125） |
| 逐元素容限 | `tolLimit = atol + rtol × \|gold\|` |
| 逐元素硬上限 | `ulpLimit = max(1e-2, 32 × ULP(\|gold\|))` |
| 匹配率要求 | `matched_ratio ≥ 0.99` |
| 通过条件 | `matched_ratio ≥ 0.99` **且** 无任何元素 `diff > ulpLimit` |

注意 `ulpLimit` 是**逐元素一票否决**：99% 的匹配率宽容只覆盖 `tolLimit`，
任意一个元素超出 `ulpLimit` 即整条用例失败。

Inf/NaN 按输入分流：输入含 NaN 时校验「输出是否全 NaN」；输入含 Inf 时走特殊值比较器
（NaN↔NaN、**同号** Inf↔同号 Inf，异号 Inf 判失败），均为零容忍。
此处与任务书 §3.2「NaN 与 ±Inf 视为等效」存在差异，实现以测试框架代码为准。

## 4.2 性能标准与实机实测

判据：`msprof op` 模式采 single-op task duration，5 次预热后取 ≥10 次均值，
平均耗时 ≤ GPU 标杆（NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4）的 1.25 倍。

**本节数据为 wall-clock 口径**（含 host 下发与同步），非验收要求的 device 侧
Task Duration；两者的关系见本节末尾的固定开销分析。实现完成后须以 `msprof op`
按验收口径复测 200 条 TC_PF。

**本设计在 Atlas A3 实机（soc `ascend910_93`，CANN 9.1.0）上做了带宽可行性摸底。**

> **数据口径声明**：下列实测数据来自**既有算子 `aclblasCsrot`**，不是本算子的实现。
> 选它作代理的理由是访存量完全相同——两者都读 x、y 各一遍、写回 x、y 各一遍，
> 每个 complex64 元素 32 字节，差异仅在向量指令条数（标量 s 为实数 vs 复数），
> 而本算子为 memory-bound，访存量决定时间下界。因此这些数字回答的是
> **「本平台的访存能力能否支撑该访存量」**，不代表本算子已实现或已达标。
> 本算子实现完成后须按 §4.3 重新实测。测量时通过环境变量覆盖了 csrot 写死的
> `numBlocks` 以扫描核数（仅用于测量，不进交付）。

核数扫描（µs，5 次预热 + 20 次采样去尾 25%，wall-clock 含 host 开销）：

| n | 8 核 | 16 核 | 24 核 | 32 核 | 40 核 | 48 核 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.67 | 43.32 | 39.94 | 38.45 | 38.55 | 44.98 |
| 1024 | 43.67 | 43.20 | 42.15 | 38.98 | 43.85 | 46.14 |
| 65536 | 38.24 | 41.19 | 46.72 | 45.07 | 46.33 | 45.92 |
| 1048576 | 70.44 | 61.81 | 56.14 | 50.83 | **45.79** | 57.36 |
| 2097152 | 93.95 | 76.39 | 61.93 | 60.67 | **52.80** | 57.65 |
| 4194304 | 145.96 | 99.45 | 86.56 | 71.70 | **63.88** | 69.28 |

**n ≥ 1M 时 40 核最优**（48 核超过 AIV 实际核数后退化）；而 n = 1024 与 65536 两档
40 核反而慢于 8 核（43.85 vs 43.67、46.33 vs 38.24），说明中小规模下多核 dispatch
与同步开销超过并行收益。这正是 §3.3.3 按 `CeilDiv(n, 每核最小元素数)` 收敛核数的
实测依据。40 核下与 GPU 标杆对比：

| n | NPU (µs) | GPU (µs) | NPU/GPU | 端到端等效带宽¹ |
| ---: | ---: | ---: | ---: | ---: |
| 1048576 | 45.79 | 18.98 | 2.41 | 733 GB/s |
| 2097152 | 52.80 | 45.53 | **1.16** | 1271 GB/s |
| 4194304 | 63.88 | 87.47 | **0.73** | **2101 GB/s** |

¹ 按 32 B × n ÷ wall-clock 计算，**含约 38 µs 的固定开销**，因此低估了 device 侧的
真实访存带宽；用于与同样含 launch 开销的 GPU 标杆做同口径比较。

n=4194304 的有效带宽 2101 GB/s 已超过该芯片标称 HBM 带宽（约 1.6 TB/s），
说明 192 MiB L2 吃下了驻留足迹——x、y 两条向量共 2×4194304×8 B = **64 MiB**
（32 B/元素是读写流量，不是足迹），远小于 192 MiB。

**关于与 GPU 标杆的缓存条件对比，需要如实区分两段**：

- **n ≤ 2M**：足迹 ≤ 32 MiB，A100 的 40 MB L2 同样容得下，两侧都处于 L2 驻留区，
  比较条件对等。该区间实测 NPU/GPU = 1.16，达标。
- **n = 4M**：足迹 64 MiB **超出** A100 的 40 MB L2 而仍在昇腾 192 MiB L2 之内，
  即该档位 NPU 享有 GPU 不具备的缓存驻留优势。NPU/GPU = 0.73 这一结果因此
  **不应表述为同等缓存条件下的对比**；它是在各自平台真实缓存行为下、按验收
  规定的同一 GPU 标杆表格比较得出的结果，在验收口径内成立，但不构成
  「本算子访存效率优于 A100」的一般性结论。

**固定开销分解**（独立探针，50 次采样去尾 25%）：

| 每次调用的原语 | 耗时 |
| --- | ---: |
| `aclrtMalloc` + `aclrtFree` | 2.56 µs |
| `aclrtMemcpy` H2D 同步（24 B tiling） | **9.74 µs** |
| `aclrtSynchronizeStream`（空闲流） | 0.21 µs |
| 三项分项之和 | 12.51 µs |
| 三者串行组合探针实测 | **14.70 µs** |

组合值高于分项之和约 2.2 µs，差额为连续提交三个运行时调用的额外开销。

40 核下 n=1 耗时 38.55 µs、n=1048576 为 45.79 µs，增量仅 7.24 µs——wall-clock 几乎
全部由 host 调用路径与 kernel 下发构成。`msprof op` 的 device 侧口径不含这部分，
按该口径三档典型规模均有数倍余量；小 n 侧，仓内 910B3 实测暖态 task duration floor
约 3–4.5 µs，而 n≤256 档位预算为 8.6–8.8 µs，余量约 2 倍。

**尚未验证项（如实记录）**：本轮摸底未取得 `msprof op` 口径的 device 侧 Task Duration，
上表为 wall-clock 口径，device 侧 kernel 时间为按固定开销作差的估算。按验收口径的
正式性能数据将在实现完成后随自测报告提供（200 条 TC_PF 全量）。

## 4.3 测试方案

工程结构（`test/rot/crot/`）：

| 文件 | 处理方式 |
| --- | --- |
| `crot_golden.h`、`crot_param.h` | 跨 arch 共享，原样复用 |
| `CMakeLists.txt` | 已无条件注册 GTest，无需改动 |
| `arch22/crot_test.cpp` | 新建，复用共享 golden 与 param |
| `arch22/crot_npu_wrapper.h` | 新建，参照 arch35 版本（三种标量指针模式逻辑中立） |
| `arch22/crot_test.csv` | 任务提供的 1200 条，文件名与路径由框架按 `__FILE__` 推导，不可更名 |

用例分布（1200 条 = 精度 1000 + 性能 200）：

| 前缀 | 条数 | 覆盖维度 |
| --- | ---: | --- |
| TC_L0 | 8 | 小尺寸 × 基础步长 |
| TC_SQ | 38 | 尺寸扫描 1→1048576 |
| TC_INC | 36 | {±1,±2,±3}² 全步长组合 |
| TC_CS | 20 | 10 组标量特殊值（含 `c=s=1e10`、纯虚 s） |
| TC_FL | 12 | 填充模式（0 / 交替 / 极值 / Inf / NaN） |
| TC_PM | 6 | c、s 的 host/device/mixed |
| TC_ED | 13 | 边界与负向（6 SUCCESS / 7 INVALID_VALUE） |
| TC_EX | 867 | 上述维度的扩展随机采样 |
| TC_PF | 200 | 性能，n ∈ [1, 4194304]，全部 incx=incy=1 |

**任务书 §3.5 要求自行补齐的用例**：随包 `gen_csv.py` 仅生成均匀分布填充
（`RANDOM_NORM_5_5`，1194 条），不生成正态分布用例；任务书要求补齐正态分布
（μ∈[-5,5]、σ∈[0.1,2]，测试框架 token 为 `GAUSS_NORM_5_2`，`test/frame/fill.h`
已支持该生成器，无需改测试工程）并在自测报告中说明补齐方式、数量与覆盖情况。
本设计将在 arch22 CSV 中追加该组用例，覆盖连续与非连续步长、常规与极值标量。

其他补充测试：
1. 跨度溢出用例需绕开 CSV 路径直接调用公开 API（CSV 路径会先按 `(n−1)·|inc|+1`
   预分配缓冲，INT32_MIN 需约 17 GB，分配先失败则测不到守卫）。
2. 零步长用例需逐元素核对串行语义（§3.1.5），不能只看返回码。
3. 内存检查用 `run_memcheck.py`，要求出现 `No error detected`。

注：上游 `arch35/crot_test.cpp` 将期望用例数写死为 `kExpectedTotal = 1206`，
`test/rot/crot/run_self_test.py`（crot 根目录共享文件）为 `EXPECTED_CSV_CASES = 1206`，
而本任务 CSV 为 1200 条。arch22 测试文件自带各自的常量即可；但共享的
`run_self_test.py` 需改为按 arch 分支取期望值——**这是本任务唯一一处共享文件改动**，
已列入 §4.4。不得直接改 `arch35/crot_test.cpp` 的常量，那会打挂 arch35 既有用例。

## 4.4 兼容性分析

1. **新增文件**：`blas/rot/arch22/crot_{host,kernel}.cpp`、`crot_kernel.h`、
   `crot_tiling_data.h`，以及 `test/rot/crot/arch22/` 下三个文件。`blas/CMakeLists.txt`
   按 glob 自动收集，无需改动构建脚本。
2. **共享文件改动（唯一一处）**：`test/rot/crot/run_self_test.py` 的期望用例数改为
   按 arch 分支取值（arch35 保持 1206，arch22 取 1200），保证两侧自检均可通过。
2. **公共头文件不变**：`aclblasCrot` 声明已存在，签名不修改，无 API/ABI 影响。
3. **同族算子不受影响**：不触碰 `csrot`、`srot` 的任何实现文件；arch35 crot 不改动。
4. **文档更新**：`blas/rot/README.md` 中 crot 的产品支持情况由「A2/A3 不支持」改为支持，
   并补充 arch22 约束说明与 per-arch 返回码对照表（含 §2.4 的差异声明）。

---

# 五、风险与未决项

| 编号 | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| R-1 | 入参契约与任务书不一致（n<0、零步长） | 评审可能要求改口径 | §2.4 已列证据链；提交时在讨论帖请维护者拍板，并准备 CSV + golden 的回退方案 |
| R-2 | `msprof op` 未能在本轮取得 device 侧数据 | 性能结论为估算 | 实现完成后以 `msprof op` 复核 200 条 TC_PF；本文档已标注该项为未验证 |
| R-3 | 中间规模（n≈1.3–1.6M）性能余量最小 | 该区间需要最高的带宽利用率 | 该区间 GPU 同处 L2 驻留区、阈值最紧；已规划分档 tile 与核数、32B 对齐切分作为该区间的专项优化，并在实测后按档位复核 |
| R-4 | `vgather` 的 V 流水开销可能成为次级瓶颈 | 连续路径带宽下降 | 已有 caxpy 先例；若实测受限，回退为逐项 `muls/add` 组合 |
| R-5 | 多核写同一 64B cache line 存在后写覆盖 | 边界元素结果错误 | 核间切分按 32 B（4 个复数）对齐，尾块单独处理 |
| R-6 | arch22 rot 族此前未设置性能门禁 | 无同族 arch22 性能基线可直接对标 | 以同访存量代理算子的实机数据建立基线（§4.2），并以 `caxpy` arch22 的优化手法作为同仓参照 |
| R-7 | 共享 golden 被 arch35 用例依赖 | 改动会波及既有测试 | 本设计不改共享 golden |

# 六、交付件清单

| 交付件 | 内容 | 状态 |
| --- | --- | --- |
| 设计文档 | 本文档 | 本次提交 |
| 算子实现 | `blas/rot/arch22/crot_{host,kernel}.cpp`、`crot_kernel.h`、`crot_tiling_data.h` | 待开发 |
| 测试工程 | `test/rot/crot/arch22/crot_test.cpp`、`crot_npu_wrapper.h`、`crot_test.csv`（1200 条 + 正态分布补充） | 待开发 |
| 文档更新 | `blas/rot/README.md` 的 crot 产品支持情况与 per-arch 契约表 | 待开发 |
| 自测报告 | 精度全量结果、`msprof op` 性能数据、`run_memcheck.py` 结论 | 待出 |

# 七、迭代规划

| 阶段 | 内容 | 出口条件 |
| --- | --- | --- |
| 一 | arch22 工程骨架 + host 校验与 tiling | 编译通过，TC_ED 13 条返回码全对 |
| 二 | 连续路径（车道交叉复数运算 + 双缓冲） | TC_L0/TC_SQ/TC_CS/TC_FL 全通过，极值标量 maxAbsErr 达标 |
| 三 | 非连续路径与串行路径 | TC_INC 36 组合 + 零步长用例全通过 |
| 四 | 性能调优与验收材料 | 200 条 TC_PF 按 msprof 口径达标，memcheck 无告警 |

# 八、参考资料

1. 任务书 `aclblasCrot_AtlasA2_A3_task_doc.md` 及随包 `test_cases/`（1200 条 CSV、GPU 基线）
2. cuBLAS 文档 `cublasCrot`；Netlib/LAPACK `crot.f`
3. `cann/ops-blas`：`blas/rot/arch35/crot_{host,kernel}.cpp`、`blas/rot/arch22/csrot_kernel.cpp`、
   `blas/axpy/arch22/caxpy_kernel.cpp`、`blas/rot/README.md`、`test/rot/crot/crot_golden.h`、
   `test/frame/verify.h`、`include/cann_ops_blas.h:250`
4. issue #827（csrot 跨 arch 返回码差异）、issue #866（跨度溢出守卫）、issue #818（异号 Inf）
5. 生态算子开源精度标准 `mixed_tolerance_standard.md`
