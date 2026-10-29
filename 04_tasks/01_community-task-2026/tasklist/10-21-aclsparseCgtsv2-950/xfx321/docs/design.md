# aclsparseCgtsv2（Ascend 950）算子设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| v1.0 | 2026-10-28 | xfx321 | 按 10 月社区任务 aclsparseCgtsv2 任务书（950）编写设计文档 |

# 需求背景（required）

## 需求来源

本设计对应"10 月社区任务-aclsparseCgtsv2 算子开发（950）"任务书。任务要求面向 Ascend 950（DAV_3510，`arch35`，A5），使用 C++ 与 Ascend C 在 `ops-sparse` 算子工程中开发 `aclsparseCgtsv2` 复数三对角线性方程组求解接口，功能、接口语义与 NVIDIA cuSPARSE `cusparseCgtsv2`（CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1）对齐，实现 complex64、带部分主元选取、B 原地覆盖、workspace 查询和异步 stream 执行能力。

cuSPARSE 仅作为接口、能力、精度与性能标杆；NPU 实现使用 aclsparse 和 Ascend C。当前 `ops-sparse` 仓库尚不存在 Cgtsv2 公开接口与实现；已有 `aclsparseSgtsv2`（FP32）仅作调用流程与算法结构参考，本任务不得修改 `Sgtsv2(float*)` ABI。

## 背景介绍

### aclsparseCgtsv2 算子功能

给定 m×m 三对角矩阵 A（由下对角 `dl`、主对角 `d`、上对角 `du` 三条复数对角线表示）与 m×n 复数右端矩阵 B（列主序，leading dimension `ldb`），求解

$$
A \cdot X = B
$$

解 X 原地覆盖 B 的有效区 `[m, n]`，无独立 X 输出、无 batchCount。以零基下标表示：$A_{i,i}=d_i$、$A_{i,i-1}=dl_i\ (1\le i<m)$、$A_{i,i+1}=du_i\ (0\le i<m-1)$，边界 `dl[0]=du[m-1]=0`。

接口采用 `bufferSizeExt → 分配 128 字节对齐 workspace → 执行` 三段式流程，全部核心计算通过 NPU Dispatch 在 A5 上异步执行（沿 handle stream）。函数名中的 `C` 表示单精度复数（complex64）；新增独立 `aclsparseCgtsv2*` ABI 与公开类型 `aclsparseComplex_t`（`float real` 与 `float imag` 连续布局），与已有 `Sgtsv2(float*)` ABI 完全独立。

三对角系统的串行消元存在长度为 m 的数据依赖链，NPU 上高吞吐求解的关键在于：在满足部分选主元数值稳定性要求的前提下，充分利用 RHS 列维度的天然并行性。本设计沿用仓内 `gtsv2`（`aclsparseSgtsv2`）已验证的 SIMT"每线程处理一个 RHS 列"框架，将其扩展到 complex64：选主元决策只依赖矩阵系数 `dl/d/du`，对所有 RHS 列完全一致，因此 SIMT 线程分支一致（无 warp 分歧）、复用同一主元序列，同时天然满足确定性要求。

本设计仅描述实现方案，不包含实测结论；实现与验收使用任务包提供的 ATK 精度用例（随包 200 条 + 验收 800 条）、203 条性能用例、内存口径与 C++ UT/ST 逐项验证。

# 需求分析（required）

## 需求描述

在 Ascend 950（arch35）路径上实现 `aclsparseCgtsv2` / `aclsparseCgtsv2_bufferSizeExt` 完整接口语义，支持：

1. 新增公开类型 `aclsparseComplex_t`：由连续 `float real` 与 `float imag` 分量组成，编译期断言保证 `sizeof == 8`、`alignof == alignof(float)`。
2. 带部分选主元（按复数模长比较）的复数三对角方程组求解，B 原地覆盖，`dl/d/du` 只读。
3. `bufferSizeExt → 128 字节对齐 workspace → 执行` 流程：正确处理 stream、workspace 生命周期与错误码；`n=0` 时查询返回 0、执行跳过计算。
4. 覆盖 `m ≥ 3`、`n ≥ 0`、`ldb ≥ max(1, m)` 的全部合法动态组合：单/多 RHS、对角占优、需要 pivot、零主对角但非奇异、病态/奇异系统与边界输入；padding 区（`m ≤ i < ldb`）保持不变。
5. 确定性：相同设备、软件版本、输入及配置下复跑结果一致。
6. 真正奇异系统不崩溃、不越界，输出（Inf/NaN）与返回码如实记录，不以非有限输出作为精度通过依据。
7. 公共算法与 A2/A3 解耦，完成交叉回归；实现代码提交至 `ops-sparse` master。

## 需求拆解

| 编号 | 子需求 | 设计承接 |
|---|---|---|
| R1 | 新增独立 complex64 ABI | `include/cann_ops_sparse.h` 新增 `aclsparseComplex_t` 与两个 Cgtsv2 声明；不修改 `aclsparseSgtsv2` 声明 |
| R2 | 带部分选主元求解 | LAPACK `zgtsv` 同构的逐步消元：按复数模长平方比较当前主元与相邻候选，交换相邻方程行并同步更新全部 RHS；维护行交换产生的第一/第二上对角填充，回代求解。不允许无主元 Thomas 实现 |
| R3 | workspace 三段式流程 | Host 侧 `bufferSizeExt` 查询（128 字节对齐、溢出检查、n=0 返回 0），执行 ABI 只收地址、按查询值足量使用 |
| R4 | Host 参数校验 | handle、`m/n/ldb` 整数约束、空指针、workspace 对齐、乘法溢出；对应 `ACL_SPARSE_STATUS_*` 错误码 |
| R5 | NPU 核心计算 | arch35 SIMT Kernel（`KERNEL_TYPE_AIV_ONLY`），每线程处理一个 RHS 列，直接读写 GM；禁止 CPU fallback |
| R6 | 确定性 | 固定算法路径、固定主元序列、无原子操作与跨线程归约；复跑记录 |
| R7 | 精度验收 | 实部/虚部分别按 `rtol=atol=2^-13` 混合容差（匹配率 ≥ 0.99）+ 最大有限绝对误差 ULP 上限；CPU Golden 为同一 complex64 输入升 complex128 带主元带状求解 |
| R8 | 性能验收 | 全部 203 条性能用例 complex64 性能倍率 ≥ 0.3（GPU/NPU Kernel 总耗时中位数之比），Event 独立列示；内存满足 L2 约束 |
| R9 | 可验证性 | C++ UT/ST（类型布局、Host、Kernel、端到端）+ ATK 精度/特殊值/合同用例 + 性能/内存采集，全链路证据留痕 |

# 详细设计（required）

## 算子分析

### 数学公式

带部分选主元的 Gauss 消元（LAPACK `zgtsv` 同构）。记第 i 步消元（i 从 1 到 m−1）前的中间量为 $d'_i,\ du'_i,\ du2'_i,\ b'_i$（$b'$ 按 RHS 列独立），主元比较采用复数模长平方（与模长比较同序，避免开方）：

$$
|z|^2 = \mathrm{Re}(z)^2 + \mathrm{Im}(z)^2
$$

**前向消元（step i = 1..m−1，消去 $dl_i$）**：比较 $|d'_{i-1}|^2$ 与 $|dl_i|^2$：

- 若 $|d'_{i-1}|^2 \ge |dl_i|^2$（不交换行）：
  - $mult_i = dl_i\,/\,d'_{i-1}$（复数除法）
  - $d'_i = d_i - mult_i \cdot du'_{i-1}$，$du'_i = du_i$，$du2'_{i-1} = 0$
  - $b'_i = b_i - mult_i \cdot b'_{i-1}$（对每一 RHS 列）
- 否则（交换行 i−1 与 i）：
  - $mult_i = d'_{i-1}\,/\,dl_i$
  - 第 i−1 行终态：$d'_{i-1} \leftarrow dl_i$，$du'_{i-1} \leftarrow d_i$，$du2'_{i-1} \leftarrow du_i$（**第二上对角填充**），$b'_{i-1} \leftarrow b_i$
  - 第 i 行新值：$d'_i = du'_{i-1,old} - mult_i \cdot d_i$，$du'_i = -mult_i \cdot du_i$（**第一上对角填充**），$b'_i = b'_{i-1,old} - mult_i \cdot b_i$

由于主元按模长取较大者，恒有 $|mult_i| \le 1$，消元乘子有界，数值稳定性与 LAPACK `zgtsv` 一致。选主元只依赖矩阵系数，不依赖 RHS，因此主元序列对所有 RHS 列唯一且确定。

**回代（i 从 m−2 降至 0）**：

$$
x_{m-1,j} = b'_{m-1,j}\,/\,d'_{m-1}
$$
$$
x_{i,j} = \big(b'_{i,j} - du'_i \cdot x_{i+1,j} - du2'_i \cdot x_{i+2,j}\big)\,/\,d'_i \quad (du2'\ 项仅当\ i+2 < m)
$$

奇异系统（出现零/近零主元）按 IEEE-754 除零语义传播 Inf/NaN：不崩溃、不越界、返回 `ACL_SPARSE_STATUS_SUCCESS`，输出如实记录（对齐 LAPACK `zgtsv`、cuSPARSE `gtsv2` 与仓内 `Sgtsv2` 既有语义）。

### 公开接口原型

在 `include/cann_ops_sparse.h` 新增（不修改任何既有声明）：

```c
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;

// 编译期布局保证（头文件内 static_assert）
// sizeof(aclsparseComplex_t) == 8, alignof(aclsparseComplex_t) == alignof(float)

aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

新增类型与仓内既有 `aclsparseComplex`（字段 `x/y`）相互独立：`aclsparseComplex_t` 按任务书字段命名 `real/imag`，二者二进制布局兼容（8 字节连续双 float），互不影响对方 ABI。

### 支持数据类型与形状

| 参数 | 类型 | 形状/值域 | 说明 |
|---|---|---|---|
| `dl` | `const aclsparseComplex_t*`（complex64） | `[m]` 连续一维，`dl[0]=0+0i` | 只读 |
| `d` | `const aclsparseComplex_t*`（complex64） | `[m]` 连续一维 | 只读 |
| `du` | `const aclsparseComplex_t*`（complex64） | `[m]` 连续一维，`du[m-1]=0+0i` | 只读 |
| `B` | `aclsparseComplex_t*`（complex64） | `[ldb, n]` 列主序连续，`ldb >= max(1, m)` | 输入输出原地；元素地址 `B[i + j*ldb]` |
| `m` | `int` 标量 | `m >= 3` | 系统阶数 |
| `n` | `int` 标量 | `n >= 0` | RHS 列数（非 batchCount）；`n=0` 跳过求解 |
| `ldb` | `int` 标量 | `ldb >= max(1, m)` | B 的 leading dimension；`m <= i < ldb` 的 padding 必须保持不变 |

不广播、不接受转置视图、任意 stride 或非约定布局。raw C ABI 只传地址和整数，不能推断实际 dtype/分配长度/stride，也不承诺检测这些不可见元数据错误；dtype/shape/布局转换由调用方或适配层负责。Inf/NaN 允许性及奇异输出行为在异常边界测试中记录，未明确支持的非有限输入不纳入常规数值精度验收。

### 参数与属性

| 参数名 | 输入/输出/属性 | 校验项（Host） | 失败错误码 |
|---|---|---|---|
| `handle` | 输入 | 空句柄 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| `m` | 属性 | `m < 3` | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `n` | 属性 | `n < 0`；`n=0` 早退成功 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `dl/d/du/B` | 输入/输出 | `n>0` 时空指针 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `ldb` | 属性 | `ldb < max(1, m)` | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `pBufferSizeInBytes` | 独立标量输出 | 空指针 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `pBuffer` | 输入 workspace | `n>0` 时空指针或非 128 字节对齐 | `ACL_SPARSE_STATUS_INVALID_VALUE` |

`bufferSizeExt` 允许 `B == NULL`（查询不使用 B 的值）；`n=0` 在矩阵形状校验前短路返回 `*pBufferSizeInBytes = 0`。workspace 字节数计算全程使用 `size_t` 并做乘法溢出检查，溢出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

## 算子实现

### 总体方案

采用 aclsparse Legacy C++ API 的 Kernel 直调工程模式（Host 参数校验 + workspace 查询 + NPU 调度 + Ascend C `arch35` Kernel）。计算路径与仓内 `sparse/gtsv2/arch35`（`aclsparseSgtsv2`）同构，将 FP32 标量算术替换为 complex64 算术：

```text
aclsparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    ├─ handle 判空 → ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
    ├─ n == 0 → 早退 SUCCESS（不启动 Kernel，不触碰 B）
    ├─ ValidateCgtsv2Params（m/n/ldb/指针/128B 对齐）
    ├─ aclsparseGetStream(handle, &stream)          # 异步：Kernel 沿 handle stream
    ├─ ComputeWorkspaceBytes(m, n)（128B 对齐、溢出检查）
    └─ LaunchCgtsv2Kernel → cgtsv2_kernel_do(...)   # NPU Dispatch，立即返回
```

`aclsparseCgtsv2_bufferSizeExt` 复用同一 `ComputeWorkspaceBytes`，不触碰设备、不依赖 B 的值，纯 Host 计算。

### Host 侧设计

Host 实现位于 `sparse/cgtsv2/arch35/cgtsv2_host.cpp`，按"Validate + ComputeWorkspaceBytes + Launch"拆分，`dlog`（`OP_LOGE/OP_LOGI`）集成，风格对齐仓内 `gtsv2_host.cpp`：

1. **参数校验**：两入口各自校验（handle 判空在各入口内先行）；`n=0` 在执行入口早退、在查询入口短路返回 0；其余校验见上文"参数与属性"表。
2. **workspace 计算**：

   $$bytes = \mathrm{align}_{128}\big(\underbrace{3 \times m \times sizeof(aclsparseComplex_t)}_{24m} \times n\big) = \mathrm{align}_{128}(24mn)$$

   即每 RHS 列三组 m 个复数中间量：修改后主对角 $d'$、含填充的修改后上对角 $du'$、第二上对角填充 $du2'$。计算过程检查 `(24mn) / n != 24m` 乘法溢出。上限校验：203 条性能用例最大 `m·n = 7168×128` → 约 21.0 MB，远小于 A5 L2 Cache（128 MB），满足内存验收条件"方案固有 workspace 申请绝对值不超过目标硬件 L2 Cache 容量"。
3. **调度提交**：`numBlocks = min(ceil(n / kCgtsv2ThreadsPerBlock), GetAivCoreCount())`，grid-stride 覆盖全部 RHS 列；TilingData 记录 `m/n/ldb/numBlocks`；`cgtsv2_kernel_do(...)` 以 `<<<numBlocks, nullptr, stream>>>` 异步下发后立即返回 `ACL_SPARSE_STATUS_SUCCESS`。执行 ABI 不使用 `pBufferSizeInBytes`，按调用方按查询值足量分配的约定直接使用 `pBuffer`。
4. **生命周期**：Kernel 为异步执行，host 不做隐式同步；handle、`dl/d/du/B`、workspace 由调用方保证在 Kernel 完成前有效；库内不分配、不释放设备内存（workspace 由调用方管理），无泄漏路径。

### Kernel 侧设计

Kernel 实现位于 `sparse/cgtsv2/arch35/cgtsv2_kernel.cpp`（声明与 TilingData 在 `cgtsv2_kernel.h` / `cgtsv2_tiling_data.h`），采用 SIMT VF 编程模型（与 `gtsv2_nopivot` / `gtsv2` 一致）：

```text
Launcher 层: cgtsv2_kernel(AIV_ONLY) → Cgtsv2Launcher.Init → Process → asc_vf_call<Cgtsv2SimtCompute>
SIMT 层:     Cgtsv2SimtCompute（grid-stride 领取 RHS 列）→ Cgtsv2ProcessOneRhs
             ├─ Cgtsv2ForwardSweep   # 前向消元（带部分选主元），b' 原地写回 B[0..m-1]
             └─ Cgtsv2BackwardSubst  # 回代，x 原地写回 B[0..m-1]
```

#### 前向消元（带部分选主元）

每线程对所辖 RHS 列执行上节公式的逐步消元。寄存器/预读不变式与仓内 `Gtsv2ForwardSweep` 一致：

- `pivot_d / pivot_du / pivot_b`（行 i−1 演进状态）与 `cur_d / cur_du / cur_b`（行 i 预读）驻留寄存器；每次迭代预读行 i+1，隐藏 GM 读延迟。
- `dl[i]`、`d[i]`、`du[i]` 永远从 GM 读原始值：step j 仅写 `wsD[j-1]/wsDu[j-1]/wsDu2[j-1]` 与 `B[j-1]`，不影响 i > j 的原始系数。
- 每步终态写入 workspace（`d'/du'/du2'` 三组）与 B（`b'`），末行（row m−1）同样落盘供回代读取；两子阶段之间不传递寄存器状态，仅经 workspace/GM 通信。

#### 回代

按上节回代公式从 m−1 递降；`xNext/xNext2` 寄存器持有 $x_{i+1}, x_{i+2}$，逐行读回 `b'[i]` 与 `d'/du'/du2'`，结果 x 原地写回 B。无奇异性护栏——零主元按 IEEE-754 产生 Inf/NaN。

#### complex64 实现

- **存储**：`aclsparseComplex_t` 即两个连续 float；Kernel 内以 `Cgtsv2Complex`（`float real, float imag`）承载，与公开类型二进制兼容，`reinterpret_cast` 于 GM 指针之上。
- **主元比较**：模长平方 `real*real + imag*imag`，无开方、无额外精度损失；比较只涉及矩阵系数，分支对所有线程一致。
- **复数乘/乘加**：`(a+bi)(c+di) = (ac−bd) + (ad+bc)i`，固定求值次序（先实部后虚部，乘法用 FMA 合并），保证逐指令确定性。
- **复数除法**：采用 Smith 比例算法，避免朴素 `(ac+bd)/(c²+d²)` 在极端量级下的中间溢出/下溢：

  ```text
  若 |c| >= |d|: r = d/c;        den = c + d*r; 结果 = ((a + b*r)/den, (b − a*r)/den)
  否则:          r = c/d;        den = d + c*r; 结果 = ((a*r + b)/den, (b*r − a)/den)
  ```

  其中 a+bi 为分子、c+di 为分母。除法路径与 LAPACK 内部缩放除法同构，float32 工作精度下的舍入上界受控。
- **工作精度**：消元因子与中间量保持 complex64（与 cuSPARSE 单精度求解的工作精度一致）；`|mult_i| ≤ 1` 的部分选主元性质抑制误差增长，满足实部/虚部 `2^-13` 混合容差与 ULP 上限验收（由 CPU complex128 Golden 比对）。

#### SIMT 并行策略与访存

- 每 SIMT 线程独占一个 RHS 列：B 列内存连续（`B[j*ldb .. j*ldb+m-1]`），线程内读/写均为顺序访问，预读流水化。
- 选主元分支只依赖 `dl/d/du`（全体线程共享输入），同一迭代内全体线程走同一分支：**无 warp 分歧**，消元/交换逻辑天然 SIMD 一致。
- `dl/d/du` 读取为全线程同地址广播访问（L2 合并）；B 访问线程间按 `ldb` 跨距独立（各列连续段，L2 无伪共享）。`dl/d/du` 全程只读，无写冲突；每线程仅写自己的 B 列与自己的 workspace 段，无任何跨线程数据竞争与原子操作。
- Padding 区 `B[m .. ldb)` 不被任何读写触碰，保持不变。

#### 本地内存预算

Kernel 为直接读写 GM 的 SIMT 实现，不依赖 UB 分块；UB 仅承载标量级暂存，远小于 AIV UB 容量（约 248 KB）。workspace 为 GM 段（24mn 字节），无 UB 容量约束风险。

### Tiling 与分核策略

TilingData 仅含 `{m, n, ldb, numBlocks, reserved}`（int32×5）。分核沿 RHS 列维：

- `numBlocks = min(ceil(n / kCgtsv2ThreadsPerBlock), aivCoreNum)`，`kCgtsv2ThreadsPerBlock = 32`（与仓内 gtsv2 系列一致）；grid-stride 循环保证任意 n 覆盖。
- n ≤ 32 时单 block 单 AIV 承载（串行链长 2m，见性能预算）；n 更大时按 32 列/block 线性扩展并行度。
- 无需 tiling key 区分分支（单一 Kernel 路径）；`m/n/ldb` 为运行时动态输入，Host 每次执行重新计算 tiling。

### 确定性设计

1. 主元序列仅由 `dl/d/du` 决定，与 RHS、线程调度顺序无关。
2. 每线程独立处理固定 RHS 列，无原子、无跨线程归约、无动态调度；复数运算求值次序固定。
3. 分核数仅影响"哪些线程处理哪些列"，每列计算路径逐指令相同；相同输入复跑输出位级一致（复跑验证入 UT）。

### 异常处理

| 场景 | 行为 |
|---|---|
| 非法 `m/n/ldb`、空指针、非对齐 workspace | Host 校验拒绝，返回 `ACL_SPARSE_STATUS_INVALID_VALUE`，不启动 Kernel |
| 空句柄 | 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| `n=0` | 查询返回 0；执行早退 SUCCESS，B 不被触碰（含 padding） |
| workspace 乘法溢出 | `ComputeWorkspaceBytes` 返回 0，两入口分别返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 真正奇异/近奇异系统 | IEEE-754 Inf/NaN 传播，不崩溃、不越界，返回 SUCCESS；输出与返回码记录于自测报告 |
| Kernel 内部错误（理论不可达） | 返回 `ACL_SPARSE_STATUS_INTERNAL_ERROR` |

不出现 CPU fallback、非法同步或越界访问。

### 性能预算与优化方案

性能倍率 = GPU cuSPARSE Kernel 总耗时中位数 / NPU 求解范围内全部 Kernel 总耗时中位数，逐用例 ≥ 0.3。依据随包 `gpu_baseline.csv`（203 条，`m ∈ [512, 8192]`、`n ∈ {1, 8, 16, 32, 64, 128}`）与单线程串行链模型（前向+回代共 2m 步、每步复数除法+乘加估计 ~40 周期）做设计估算（以实测为准）：

| 场景类（条数） | GPU Kernel 中位数（μs） | 预算 = GPU/0.3（μs） | NPU 设计估算（μs） | 判断 |
|---|---|---|---|---|
| n=1，m=512~4480（27 条） | 84.9~190.9 | 283~636 | 60~240 | 最紧约束类（m=4096 预算 308）；串行链 2m 步 ≈ 180~240 μs，留有 ≥ 25% 裕量 |
| n=8~16（66 条） | 131.4~359.7 | 438~1199 | 100~450 | n 路线程并行，裕量充足 |
| n=32~128（110 条） | 334.8~1451.2 | 1116~4837 | 150~700 | 多 block 并行，GM 带宽主导（最大 B 流量 ~29 MB），裕量充足 |
| P-01 m=8192,n=64 / P-02 m=4096,n=64 / P-03 m=7168,n=128 | 640.9 / 628.9 / 1220.0 | 2136 / 2097 / 4067 | 500~700 / 250~400 / 450~700 | 裕量充足 |

若小 n 用例实测裕量不足，按下列优先级启用优化（均为数学等价变换，不牺牲选主元能力与确定性）：

1. **因子共享两 Kernel 拆分**：Kernel-1（单线程）串行计算 `d'/du'/du2'` 写入共享 workspace（24m 字节，与 n 无关）；Kernel-2 每线程读共享因子完成 B 的前向/回代。省去每列因子存储与流量（P-03 约 22 MB → 172 KB），代价是额外一次 kernel 启动与单线程因子链（m 步）显式化。Profiler 口径仍为"求解范围内全部 Kernel 求和"，不改变计时约定。
2. **小 n 的 m 维分区并行（SPIKE 型，主元限制于块内）**：行方向分 P 块，每块独立做含部分选主元的局部消元（块边界以增强列约束主元不跨块），块间以约 `2(P−1)` 阶小型 reduced system（单线程串行求解）耦合，再各块并行回代。串行链深度从 2m 降为 O(m/P + P)。
3. **常数级优化**：复数除法改"乘倒数"；前向/回代预读深度加大；`du2'[i]=0` 连续段跳过第二填充项乘加。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR / Ascend 950DT（DAV_3510，arch35，A5） | √ |
| Atlas A2/A3 训练/推理系列（arch22） | 不涉及（公共算法与 A2/A3 解耦） |

### 算子约束限制

1. `m >= 3`、`n >= 0`、`ldb >= max(1, m)`；`n=0` 为合法空 RHS（查询 0 / 执行跳过）。
2. `dl[0] = du[m-1] = 0`（用户责任，Kernel 不改写不校验数值内容）；`dl/d/du` 只读。
3. B 列主序连续、原地覆盖有效区 `[m, n]`，padding 区不变；不接受非约定布局。
4. 非奇异输入结果确定；奇异输入按 IEEE-754 传播 Inf/NaN（非精度通过依据）。
5. workspace 按 `bufferSizeExt` 查询值足量分配且 128 字节地址对齐；执行 ABI 不传实际大小。
6. raw C ABI 不校验调用方未声明的 dtype/长度/stride 元数据。

### 代码落点与构建接入

| 目录或文件 | 提交内容 |
|---|---|
| `include/cann_ops_sparse.h` | 新增 `aclsparseComplex_t`（含布局编译期断言）与 `aclsparseCgtsv2` / `aclsparseCgtsv2_bufferSizeExt` 声明及 Doxygen 注释；不修改 `Sgtsv2` 声明 |
| `sparse/cgtsv2/README.md` | 算子说明（功能、接口表、算法简述），对齐 `sparse/gtsv2/README.md` 风格 |
| `sparse/cgtsv2/arch35/cgtsv2_host.cpp` | 参数校验、workspace 查询、Kernel 调度（extern "C" 两入口） |
| `sparse/cgtsv2/arch35/cgtsv2_kernel.h` / `cgtsv2_tiling_data.h` | kernel_do 声明、线程常量与 TilingData |
| `sparse/cgtsv2/arch35/cgtsv2_kernel.cpp` | SIMT Kernel（前向消元 + 回代 + Launcher + `__global__` 入口） |
| `test/cgtsv2/` | 公共 Golden（CPU 复数带主元带状求解）、测试参数 CSV、README 与 ATK 测试附件引用 |
| `test/cgtsv2/arch35/cgtsv2_test.cpp` | 本平台 C++ UT/ST（GTest，参数化成功/奇异/异常用例 + 独立异常 TEST_F），入口复用 `test/frame/test_main.cpp` |
| 根目录及相关 `CMakeLists.txt` | 沿用仓内 `file(GLOB_RECURSE)` + arch 目录过滤机制，新增目录自动接入，不新增重复 build 目录 |

A2/A3 后续合入 Cgtsv2 时，按任务书约定基于已合入版本处理公共 Host 代码冲突并完成交叉回归；本次 arch35 目录内代码自包含，不预设跨架构共享文件。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | complex64 输出实部/虚部分别按混合容差 `\|actual − golden\| ≤ atol + rtol × \|golden\|`（`rtol = atol = 2^-13`）验收，两部分匹配率均 ≥ 0.99；每一分量最大有限绝对误差 ≤ `max(1e-2, 32 × ULP(g_low))`（`g_low` 为最大误差点 Golden 值收窄 float32）；±Inf 特例按 RNE 收窄同号判定。CPU Golden：同一 complex64 输入升 complex128 带主元带状求解。特殊值套件（NaN/Inf 分类与符号一致）独立记录 | 任务书 3.2 节及 opbase 混合容差标准 |
| 性能标准 | 全部 203 条性能用例 complex64 倍率 ≥ 0.3（两端 Kernel 总耗时中位数之比，NPU 侧求和覆盖求解范围内全部 Kernel）；预热 ≥ 10 次、采样 ≥ 30 次、逐次恢复 B 后计时、workspace 复用；Event 口径独立列示不参与倍率 | 任务书 3.3 节 |
| 内存标准 | 固有 workspace（最大用例约 21 MB）≤ 目标硬件 L2 Cache（128 MB）；逐用例报告 io/workspace/恢复副本/allocator 基线与峰值/设备快照；无 CPU fallback、泄漏、越界、非法同步 | 任务书 3.4 节 |

### 测试设计

**C++ UT/ST**（`test/cgtsv2/arch35/cgtsv2_test.cpp`，GTest + CSV 参数化，核心计算经 NPU Dispatch）：

| 组 | 覆盖项 |
|---|---|
| 类型布局 | `sizeof(aclsparseComplex_t) == 8`、`alignof == alignof(float)`、`real/imag` 偏移与连续性（static_assert + 运行期断言） |
| Host 校验 | 空句柄、`m<3`、`n<0`、`ldb < max(1,m)`、空指针（dl/d/du/B/pBuffer/pBufferSizeInBytes）、非 128B 对齐 pBuffer、workspace 溢出边界 |
| 接口行为 | `n=0`：查询 0、执行 SUCCESS 且 B 与 padding 全程不变；`bufferSizeExt` 允许 B=NULL；`bufferSizeExt → 分配 → 执行 → 释放` 全流程 |
| Kernel/端到端 | 对角占优、需 pivot（含逐行交换）、零主对角但非奇异、病态、奇异（Inf/NaN 输出与 SUCCESS 返回码记录）、m=3 边界、大 m、单/多 RHS、随机 CSV 用例（Golden 比对） |
| 语义不变式 | `dl/d/du` 执行前后位级只读；B padding 区位级不变；B 原地（无独立 X）；相同输入复跑 N 次位级一致（恢复原始 B 后重复执行并留复跑记录） |

**ATK 测试工程**（任务包随包，执行入口沿用包内 README 原生命令）：随包精度 200 条（`atk case -f case_designs/accuracy.yaml -p atk_plugins.py -s 1028`）、验收额外 800 条（新种子独立运行）、完整因子网格覆盖审计（66,432 组合元数据审计）、特殊值 8 条（NaN/Inf 分类与符号）、适配层拒绝合同 12 条（`CGTSV2_INVALID_PARAMETER:` 前缀约定）、性能 203 条（Kernel/Event/内存三口径采集，`collect_performance.py`）。适配层拒绝只证明合同；真实 C++ 错误码、非法指针、workspace 对齐与 NPU 执行以本节 C++ UT/ST 与上板证据为准。

### 性能优化方案

见"性能预算与优化方案"：主路径（SIMT 每线程一 RHS 列）覆盖 203 条预算分析；备选优化按"因子共享拆分 → 小 n 分区 SPIKE → 常数级"顺序启用，启用前后用同一 ATK 性能清单对比，两端重采保证口径一致。

## 兼容性分析

1. **新增 ABI，零存量影响**：`aclsparseComplex_t` 与 `aclsparseCgtsv2*` 为全新符号；不修改 `aclsparseSgtsv2(float*)` 等既有声明、类型与行为，已合入算子交叉回归不受影响。
2. **与既有 complex 类型关系**：新增 `aclsparseComplex_t`（real/imag 命名）与既有 `aclsparseComplex`（x/y 命名）布局兼容、定义独立，避免触碰既有用户编译单元。
3. **平台解耦**：本任务仅落 `arch35` 目录；A2/A3 与 A5 PR 先后合入时按任务书处理公共代码冲突，不假设对方先合入。
4. **版本配套**：CANN 9.0.0 及后续配套版本；ATK 26.9.8（源码提交 `9504bee6076820ca74259fb919a790a12ef88cf8`）；如提供 Python/ATen 适配，PyTorch 2.7+、torch_npu 26.0.0+。报告记录实际 CANN/驱动/固件/硬件型号与代码提交版本。

## 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| n=1、大 m 用例串行链接近预算（如 m=4096 预算 308 μs） | 个别用例倍率 < 0.3 | 上板实测优先验证该类用例；不达标即启用优化项 1/2（因子共享拆分、分区 SPIKE），两端同清单重采 |
| float32 中间精度与 complex128 Golden 的偏差 | 匹配率或 ULP 上限边缘失败 | Smith 除法 + `\|mult\| ≤ 1` 选主元控制误差增长；必要时消元因子以双 float（两分量分别补偿）承载——不改公开 ABI，仅内部 workspace 精度 |
| SIMT 单 block 场景（n ≤ 32）单核带宽利用不足 | 小 n 用例耗时上升 | 预算分析已计入；不足时按优化项 2 引入 m 维并行 |
| 奇异/病态用例的 Inf/NaN 语义争议 | 特殊值套件失败 | 严格按任务书 3.2 特殊语义表实现与记录；常规精度比对不含 NaN 通过 |
| 上游 A2/A3 Cgtsv2 或 Sgtsv2 公共代码冲突 | PR 合入冲突 | arch35 目录自包含；按任务书基于已合入版本处理并交叉回归 |
