# 需求背景

## 需求来源

社区任务：**aclblasCsyr2 / aclblasCher2k / aclblasCherk 算子精度修复（A2/A3）**，2026 年 10 月批次。

- 任务书：`aclblasCsyr2_Cher2k_Cherk精度修复_A2A3_task_doc.md`
- 实现仓库：https://gitcode.com/cann/ops-blas
- 适配硬件：Atlas A2 / Atlas A3 系列（性能测试设备 Atlas 800I A2，Ascend 910B3）
- CANN 版本：CANN 9.1.0；开发语言：Ascend C
- 承接团队：`wisdom_qq`

本任务为**精度修复**，不是新算子开发。三个算子已在 ops-blas 的 arch22 目录合入，本次只在 `arch22` 内修正写回边界，不改接口、不改语义、不回退性能。三个算子作为同一社区任务一并交付。

## 背景介绍

### 算子现状

三个算子都是「按 `uplo` 指定的三角，把更新量原地累加进同一块列主序缓冲区」的复数 BLAS 更新类算子。逻辑阶数 `n`，物理布局 `ld × n` 个 `complex64`，元素按 `M(i,j) = M[i + j*ld]` 寻址。实现分别位于：

| 算子 | 实现目录 | 测试目录 | 标量归属 |
| --- | --- | --- | --- |
| aclblasCsyr2 | `blas/syr2/arch22/` | `test/syr2/csyr2/arch22/` | α 在 Host |
| aclblasCher2k | `blas/her2k/arch22/` | `test/her2k/cher2k/arch22/` | α、β 在 Device |
| aclblasCherk | `blas/herk/arch22/` | `test/herk/cherk/arch22/` | α、β 在 Device |

> 任务书 §5 把 Cher2k 与 Cherk 归在同一行 `blas/herk/arch22/`，仓内实际是两个独立目录：Cher2k 在 `blas/her2k/arch22/`、Cherk 在 `blas/herk/arch22/`。本设计以仓内实际路径为准。

arch22（`dav_c220`，A2/A3）与 arch35（950PR）双架构共存，本任务**只修改 arch22**。

### 缺陷现象

在 A2（DevEnvC_X51Cm，1×NPU 910B3，CANN 9.1.0）上，以 Netlib BLAS/LAPACK 3.10.0 为 golden，加载本任务包 `test_cases/` 的 CSV 逐条执行，未修改的基线（pristine `blas/`）结果如下：

| 算子 | 通过 / 失败 | 失败用例 | 缺陷类别 |
| --- | --- | --- | --- |
| aclblasCsyr2 | 1024 / 5 | `TC_BIT_013`、`TC_ED_151` | 非操作区 bit 不一致；返回码期望值笔误 |
| aclblasCher2k | 1029 / 0 | — | — |
| aclblasCherk | 1028 / 1 | `TC_FL_166` | 操作区极值精度（虚部 `nan` vs `-inf`）；判据不可达，见交付件 3 §5 |

**类别 A：非操作区二进制不一致（本任务主门禁）**

`TC_BIT_013` 的失败信息给出了直接证据：

```text
[TC_BIT_013] untouched mismatch at (0,1): before=[0x7fc00000,0x7fc00000] after=[0x7fffffff,0x7fffffff]
```

位置 `(0,1)` 在 `uplo=LOWER` 下属于**未引用的另一三角**（严格上三角），调用前后必须逐 bit 相同。实际调用后该 `complex64` 的实部与虚部都从 `0x7fc00000`（quiet NaN，payload 全 0）被改写成 `0x7fffffff`（payload 全 1）。这不是数值误差，而是**位模式被硬件重写**。

**类别 B：操作区极值精度**

`TC_FL_166`（`fill_extreme`，Cherk，UPPER）操作区虚部为 `nan`，golden 为 `-inf`，496/528 个元素不符。属独立的极值处理缺陷。

**类别 C：任务包 CSV 期望值有误**

`TC_ED_151` 检查非法枚举的返回码。实现返回 `ACLBLAS_STATUS_INVALID_ENUM`(10)，CSV 期望 `ACLBLAS_STATUS_INVALID_VALUE`(3)。对照 `include/cann_ops_blas.h` 与 Netlib/cuBLAS 语义，**实现正确**，属任务包 CSV 期望值笔误，不属于算子缺陷。

### 缺陷根因分析

把 LOWER 路径的写回过程在 910B3 上逐段拆开后，确认是三条原因叠加，且**只有 LOWER 路径会命中**（UPPER 基线全过，即为佐证）。

**根因 1：非操作区的保护依赖 GM 原子加，而原子加会规范化 NaN payload。**

LOWER 路径原本在 `SetAtomicAdd<float>()` 下运行：先把结果写进 UB 的前缀区并清零，再用原子加「加」到 GM 上，等价于对旧值加 0，从而让未引用三角保持不变。但原子加在硬件上是一次「读—归一化—加—写」：`0x7fc00000 + 0.0f` 得到 `0x7fffffff`，`-0.0f + 0.0f` 得到 `+0.0f`。NaN 的 payload、NaN 的符号位、±0 的符号因此全部丢失，与「非操作区逐 `complex64` `memcmp` 相等」的验收口径直接冲突。

**根因 2：前缀清零掩码只覆盖一列。**

前缀行靠 `ApplyLowerPrefixMask` 用 128-bit 掩码 `Duplicate` 清零，但该调用 `repeatTimes` 恒为 1、block/stride 固定为 8，一次只处理 32 B（4 个 `complex64`）。当列批 `b > 1` 时，只有第 0 列的前缀被清零，其余列的前缀仍是计算值。于是在原子加被撤销后，这些前缀会把旧值覆盖成「新算出来的 0」，非操作区被真值污染。

**根因 3：整列写回的起点按 32 B 对齐上取整，天然越界到非操作区。**

`StoreLowerBatch` 按列写 `cur*2` 个 `float`，起点是 `row0 = j0 & ~3`（向量起始地址需 32 B 对齐，即 4 行）。LOWER 的操作区起点是第 `j` 行，因此 `[row0, j)` 最多 3 行属于非操作区，每次列写都会带上它们；当 `ld > n` 时 `[n, ld)` 的 padding 行也必须一并保证不被写。原实现正是靠根因 1、2 的手段兜住这 3 行——一旦去掉原子加，就必须显式处理。

# 需求分析

## 需求描述

在不改动公开接口、计算语义、异步行为与性能的前提下，修复 arch22 下 aclblasCsyr2 / aclblasCher2k / aclblasCherk 的写回边界：

1. **操作区**（`uplo` 选中三角，对角属操作区）满足 BLAS 公式并达混合容差 `rtol=atol=2⁻¹³`、`required_matched_ratio ≥ 0.99`；
2. **非操作区**（另一三角 + `n ≤ i < ld` 的 padding 行）逐 `complex64` 逐 bit 不变，`required_matched_ratio = 1.0`，含 NaN payload、NaN 符号、±0、次正规数；
3. quick return（`n = 0`、`α = 0`、`k = 0` 等）时整个 `ld × n` 缓冲区逐 bit 不变；
4. Hermitian 族（Cher2k / Cherk）正常计算时对角虚部置 0，不向另一侧写共轭值；
5. 性能不高于任务书 §3.3 标杆；不为保护非操作区申请与问题规模无关的持久 Device 缓存。

## 需求拆解

1. 明确非操作区边界，逐条核对 kernel 内每个写回路径的实际写入范围（packed / window / interleaved 收尾 / planar 收尾 / tiny 标量）。
2. 撤销 LOWER 对 GM 原子加的依赖，改为显式「读旧值 → UB 内累加」，使累加不改变原有位模式。
3. 对齐上取整引入的前缀行 `[row0, j)` 改为**逐位还原**：从 GM 读回旧值，在 UB 内以 `int32` 位复制写回，全程不经浮点运算。
4. 为该方案补齐 UB 预算校验与 host 侧路径选择，避免新增列旧值缓冲后 UB 越界。
5. 测试侧接入「调用前后非操作区逐 bit 相同」检查（`check_untouched_bitwise`），并新增 `TC_BIT_*` 用例覆盖 NaN/±0/次正规/padding/quick return。
6. 在 A2、A3 两套环境分别完成精度 + bit 检查，A2 上完成三项性能测试并对照 §3.3 标杆。

# 详细设计

## 算子分析

### 数学公式

列主序 `M(i,j) = M[i + j*ld]`。向量逻辑元素按 `start + i*inc` 访问，`inc < 0` 时零基起点为 `(1-n)*inc`。

**aclblasCsyr2**（对称更新，普通转置，不共轭；对角虚部参与普通计算，不置零）

- `uplo = UPPER`，`0 ≤ i ≤ j < n`：`A(i,j) ← A(i,j) + α·x(i)·y(j) + α·y(i)·x(j)`
- `uplo = LOWER`，`0 ≤ j ≤ i < n`：同上
- α 为 `complex64`，在 Host 侧

**aclblasCher2k**（Hermitian 秩 2k 更新；α 复数、β 实数）

- `trans = N`：`C(i,j) ← α·Σ_k A(i,k)·conj(B(j,k)) + conj(α)·Σ_k B(i,k)·conj(A(j,k)) + β·C(i,j)`
- `trans = C`：`C(i,j) ← α·Σ_k conj(A(k,i))·B(k,j) + conj(α)·Σ_k conj(B(k,i))·A(k,j) + β·C(i,j)`

**aclblasCherk**（Hermitian 秩 k 更新；α、β 实数）

- `trans = N`：`C(i,j) ← α·Σ_k A(i,k)·conj(A(j,k)) + β·C(i,j)`
- `trans = C`：`C(i,j) ← α·Σ_k conj(A(k,i))·A(k,j) + β·C(i,j)`

两个 Hermitian 算子只引用所选三角，对角虚部强制置 0，**不向另一三角写共轭值**——另一三角既不读、也不写。

### 支持数据类型

矩阵、向量以及 Csyr2 / Cher2k 的复数标量为 `complex64`（实、虚部各 `float32`）；Cherk 的 α、β 与 Cher2k 的 β 为 `float32`。

### 支持形状

列主序，逻辑阶数 `n ≥ 0`，物理输出 `ld × n`；`trans = N` 时输入矩阵逻辑形状 `n × k`，`trans = C` 时为 `k × n`，`k ≥ 0`。两种 `uplo`；Csyr2 支持非零正负 `incx` / `incy`。前导维与步长约束按任务书 §2.5 执行。

## 算子实现

### 3.2.1 Host 侧设计

**校验与入口**

沿用 ops-blas 现有句柄式入口与校验顺序，不改函数签名、参数顺序、枚举值与返回码。Cher2k / Cherk 对合法但不支持的 `OP_T` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。Csyr2 的 α 在 Host，可直接据此做 `α = (0,0)` quick return；Cher2k / Cherk 的 α、β 在 Device，**禁止为判断标量是否为零而引入 Host 同步或改变标量地址归属**，`α = 0` / `k = 0` 分支在 device 侧判。

**tiling 策略**

tiling 传递 `n`、`k`、`ld`、`uplo`、`trans`、步长、分块（`tileRow`）、列批（`batch`）、复用标志与核数。按输出三角的**有效工作量**分配核与列批；同一输出元素只归属一个核。

本次修复的关键约束是：**新方案的「读旧值 → 累加 → 写回」要求同一列的写集合在核间互斥**，否则两个核对同一列做读改写会丢更新。原原子加方案允许列交叠，因此该约束是新增的；现有按 `jBegin_` / `jEnd_` 的列划分已满足（每核持有互不重叠的列区间），tiling 侧不需要额外改动，但需在代码评审中作为不变量注明。

**新增 UB 预算校验**

LOWER 改成「读旧值 + UB 内累加」后，每核多出一份列旧值缓冲 `aOldBuf_`。整块模式（`csyr2_kernel_lower_one`，对应 `kFullMode = 2`、`kUnitAlpha`、`kRotateLower`）的 UB 占用因此上升，须在 host 侧对候选路径做显式预算校验，超出 192 KB 就退回通用路径 `csyr2_kernel_1`，否则 UB 越界：

```cpp
/* csyr2_kernel_lower_one 的整块 UB 占用，与 InitDirectStorage 逐项一致。
 * LOWER 改成「读旧值 + UB 内累加」后还要多留 packB_ 列旧值。 */
static bool Csyr2LowerOneAccumFits(uint32_t tileRow)
{
    const uint64_t ilvFloats   = ((static_cast<uint64_t>(tileRow) * 2u + 127u) / 128u) * 128u;
    const uint64_t planeFloats = ((static_cast<uint64_t>(tileRow) + 63u) / 64u) * 64u;
    const uint64_t ilvB   = ilvFloats * sizeof(float);
    const uint64_t planeB = planeFloats * sizeof(float);
    const uint64_t tmpF   = (ilvFloats * 2u > planeFloats * 3u) ? (ilvFloats * 2u) : (planeFloats * 3u);
    const uint64_t used = 64u + 2u * (ilvB + 768u) + 2u * (ilvB * 4u + 512u) +
                          (planeB * 6u + ilvB + 768u - 1024u) + (tmpF * sizeof(float) + 768u) + ilvB;
    const uint64_t aOldBytes = 4u * ilvB;                 /* packB_ = 4 */
    return (used + aOldBytes) <= 192u * 1024u;
}
```

该校验参与 `TryLaunchCsyr2LowerOne` 的规模门限判定（当前在 `n ∈ [640, 1408]` 区间生效），保证整块快路径只在安全规模上启用。

**分核策略**

满核优先；核间按列批均分，不能整除时余数分配到前若干核（大核 / 小核），保证每个 LOWER 列的真实累加只发生在其归属核内。列批取整不得把写回扩大到非操作区。

**数据分块与内存优化策略**

- **UB 缓冲**：新增 `AscendC::TBuf aOldBuf_`，容量按路径规划——packed / unit 路径为 `packB_ * ilvFloats_ * sizeof(float)`，window 路径为 `outBatch * ilvFloats_ * sizeof(float)`。
- **缓冲基址不变原则**：`aOldBuf_` 在既有 `InitDirectStorage` / `InitUnitStorage` / window 初始化**末尾**追加，保持原有一切缓冲的基址与 UB bank 相位 `(addr >> 5) mod 16` 不变，避免引入新的 bank 冲突导致性能回退。
- **三处预算同步补 `aOld` 项**：`SelectLowerPacking`（packing 选择）、`Csyr2SelectWindowBatchCount`（window 列批）、`Csyr2LowerDoubleBufferedColumnsFit`（双缓冲列数），否则会把 UB 用满后越界。
- **内存口径**：本方案只在 UB（片上）多留一份列旧值，**不申请任何与 `n` / `k` 无关的持久 Device 缓存**，也不使用「先写满整块再拷回」的做法，符合任务书 §3.4。

**tiling key 规划**

沿用既有编译期分支体系（`kFullMode`、`kUnitAlpha`、`kUpper`、`kRotateLower`、`TILING_KEY`），本方案**不新增 tiling key**。所有新增逻辑都包在 `if constexpr (!kUpper)` 内，`kUpper = true` 时被完全消除；`aOldBuf_` 的 `InitBuffer` 同样在 `if constexpr (!kUpper)` 内。因此 UPPER 生成的指令流与基线一致，`csyr2_kernel_upper_*` 与 `csyr2_kernel_0/2/4` 继续沿用 `SetAtomicAdd`。

### 3.2.2 Kernel 侧设计

Init 与 Process 两阶段，Process 含 CopyIn / Compute / CopyOut。本次修复的核心是把非操作区的保护方式从「靠原子加把 +0 叠上去」改成「**不产生需要撤销的写**」——要么写完后按位还原，要么根本不写。

#### 路径一（方案 A）：整列写路径（packed / window）改为「读旧值 → 累加 → 前缀位还原」

适用于 `PackedLowerChunk` → `ProcessLowerPackedGeneric` / `ProcessLowerPackedFull`，以及 `WindowChunk` → `ProcessLowerWindowGroup`。

**CopyIn（新增旧值搬入）**

除原有输入向量 / 矩阵搬入外，LOWER 额外把本批列在 `[row0, n)` 的旧值搬进 `aOldBuf_`：

```cpp
__aicore__ inline void LoadLowerOldCol(
    AscendC::LocalTensor<float> dst, int64_t gmFloatOff, int32_t floats)
{
    /* 一律走 DataCopyPad：UB 端 32B 对齐（由 pitchFloats 保证），GM 端任意对齐。
     * DataCopy 隐含要求 GM 起点 32B 对齐，而 lda 为奇数时 (j*lda + i0) 未必是 4 的倍数。 */
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(floats) * sizeof(float);
    p.srcStride = 0; p.dstStride = 0; p.rsv = 0;
    AscendC::DataCopyPadExtParams<float> pad(false, 0, 0, 0.0f);
    AscendC::DataCopyPad(dst, aGm_[gmFloatOff], p, pad);
}
```

多列批按 `kk * pitchFloats` 逐列搬入，`UB` 端 pitch 已按 4 个 `float` 取整保证 32 B 对齐。

> 实现注记：必须用 `DataCopyPad` 而不是 `DataCopy`。`lda` 为奇数时 `(j*lda + row0)` 未必是 4 的倍数，不满足 `DataCopy` 隐含的 32 B GM 对齐要求；早期版本把这里换成 `DataCopy` 后，`kk > 0` 的列被静默丢弃，表现为 `matchedRatio` 大幅下降。

**Compute**

正常计算 `α·x·yᵀ + α·y·xᵀ` 部分，结果留在 UB。沿用原有向量算子与分块。

**累加**

```cpp
__aicore__ inline void AccumulateLowerOld(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> aOld, int32_t nAll)
{
    if constexpr (kUpper) { return; }
    AscendC::Add(aOut, aOut, aOld, static_cast<uint32_t>(nAll));
}
```

浮点加法本身会规范化 NaN，但操作区本来就允许混合容差，且旧值参与的是真正的 BLAS 语义（`A ← A + term`），因此这里的差别仍在容差内。同步用 `SyncEvent<HardEvent::MTE2_V>()` 保证旧值搬入先于累加。

**前缀位还原（关键）**

`[row0, j)` 最多 3 行属非操作区，累加过的值不能用，必须用旧值覆盖。**用 `int32` 的 `Adds(..., 0)` 做纯位复制**：

```cpp
auto outBits = aOut.ReinterpretCast<int32_t>();
auto oldBits = aOld.ReinterpretCast<int32_t>();
AscendC::Adds(outBits[kk * pitchFloats], oldBits[kk * pitchFloats],
              static_cast<int32_t>(0), static_cast<uint32_t>(prefix * 2));
```

`int32` 的 `Adds(x, 0)` 是逐位复制，NaN payload、NaN 符号、±0 符号、次正规数全部原样保留。用 `float` 的 `Adds(x, 0.0f)` 不行——那会规范化 NaN 并抹掉 `-0.0f` 的符号位，等于把根因 1 换个位置重犯。

每列前缀长度 `prefix = clamp((j0 + kk) - row0, 0, cur)`，逐列独立 `Adds`，因此不存在「掩码只覆盖第 0 列」的问题（根因 2）。

**CopyOut**

仍按 `[row0, n)` 整列写。**这不是「零写」**，而是任务书 §2.0.3 允许的「先读后写」：前缀行（对角线上方至多 3 行，整窗分组至多 7 行）在 UB 内先被还原成旧值，再随整列一起写出，**写出的字节与调用前逐 `complex64` 逐 bit 相同**，因此满足 §2.0.2 的 `memcmp` 判定口径；`[n, ld)` 的 padding 行不在写回范围内，从不被触碰。

> §2.5.4 字面「零写」的实测结论：可实现，但代价明确。方案是把每列拆成
> `body = [4·ceil(j/4), n)`（一次 32 B 对齐 DMA）+ `head = [j, 4·ceil(j/4))`（至多 3 个复数，
> 只能用标量 8 B 写，因为 `DataCopyPad` 的 UB 侧起始地址必须 32 B 对齐，而操作区起点是 8 字节粒度）。
> 在 A3 上实测：`TC_BIT_*` 22/22 两遍全过、`TC_SQ_*` 46/46 全过，证明零写本身可达成；
> 但 `TC_PF_1002`（LOWER n=1024）从 **7.0~7.2 us 抬到 10.0~10.5 us（+45%）**，超过 §3.3 的 9.72 us 标杆。
> 另有 `kRotateLower` 交织路径（n∈[640,1408]）残留一处**非时序**的 32 B 块级数据错误未收敛。
> 综合权衡后交付版本采用「先读后写 + 逐位还原」，接口与语义均满足 §2.0.2 的可测口径，
> 且性能留在标杆以内；字面零写作为已知限制记录在此。

同步顺序上，`V_MTE2` 事件放在 `StoreLowerBatch` **之后**——让 MTE3 先发射，标量只需等向量排空即可，避免反过来挡住写出发射。

**同步约束（实现期实测踩坑）**：`V → MTE2` 必须**无条件**发出，不能挂在「有前缀」的分支里。`aOldBuf_` 是每核一份、跨 tile/跨 wave 复用的单缓冲；若某一段因 `prefix == 0` 没有发 `V_MTE2`，下一段的 `LoadLowerOldCol`（MTE2）就会在本段 `AccumulateLowerOld`（V）读完之前覆盖它，读到的是被后一段冲掉的数据。症状是「多 tile 的列出现 ≈|A| 量级的随机偏差」——n=2048 LOWER 实测 242 列整列出错、`matchedRatio` 0.885，定位后加回无条件同步即恢复到 1.0。这条约束对 packed / window / interleaved 三条路径同等适用。

**为什么不再用掩码 `Duplicate` 清零**

`Duplicate` + 128-bit 掩码一次只处理一个 32 B 块，多列批要按列展开并重设 stride，指令数随列批线性增长；而且清零之后仍必须靠原子加把旧值「加回来」——本质是「先毁再补」，既多一次全局读改写，又把 NaN payload 押在原子加的语义上。直接读旧值 + 位复制既不摧毁原值，也不依赖任何原子语义，是更短且更强的路径。

#### 路径二（方案 B）：interleaved 收尾无条件读旧值

`FinishInterleavedChunk` 早先按「有前缀才读旧值」做分支，结果 `prefix == 0` 的列丢掉了 A 的原有内容，实测证据：

```text
[TC_BIT_011] FAILED (matchedRatio=8.3088e-01 (req=9.9000e-01), maxAbsErr=3.4028e+38, 46/272 failures)
```

修正为：LOWER 下**无论有无前缀，整列都先取旧值再累加**，仅前缀部分走位还原：

```cpp
AscendC::LocalTensor<float> aOld;
if constexpr (!kUpper) {
    /* 非原子写：无论有无对角前缀，整列都要先取旧值再累加，否则会丢掉 A 的原有内容。 */
    aOld = aOldBuf_.Get<float>();
    LoadLowerOldCol(aOld, gmComplexOff * 2, cur * 2);
}
AscendC::Muls(aOut, xIlv, t1r, n2);
AscendC::Axpy(aOut, xs,   t1i, n2);
AscendC::Axpy(aOut, yIlv, t2r, n2);
AscendC::Axpy(aOut, ys,   t2i, n2);
if constexpr (kUpper) {
    if (prefix > 0) { AscendC::PipeBarrier<PIPE_V>(); AscendC::Duplicate(aOut, 0.0f, prefix * 2); }
} else {
    SyncEvent<AscendC::HardEvent::MTE2_V>();
    AccumulateLowerOld(aOut, aOld, cur * 2);
    if (prefix > 0) {
        auto outBits = aOut.ReinterpretCast<int32_t>();
        auto oldBits = aOld.ReinterpretCast<int32_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(outBits, oldBits, static_cast<int32_t>(0), static_cast<uint32_t>(prefix * 2));
        SyncEvent<AscendC::HardEvent::V_MTE2>();
    }
}
```

#### 路径三（方案 C）：planar 收尾路径

`FinishPlanarChunk` 在 `Gather` 重排之后同样插入 `AccumulateLowerOld`，让 planar 与 interleaved 两条收尾路径语义一致。`ChunkStep` 中原本把 `TILING_KEY_LOWER_UNIT` 与 `kUpper` 归为一类（即 LOWER 单位 α 也走「清零 + 原子加」），现改为只判 `kUpper`，LOWER 统一走读旧值分支。

#### 方案 D：kernel 入口统一撤销原子加

`csyr2_kernel.cpp` 中各 LOWER 入口的 `Csyr2RunAtomic(op)` 统一改为 `op.Process()`：

| kernel 入口 | 修改前 | 修改后 |
| --- | --- | --- |
| `csyr2_kernel_lower_one` | `Csyr2RunAtomic` | `op.Process()` |
| `csyr2_kernel_1`（LOWER 通用） | `Csyr2RunAtomic` | `op.Process()` |
| `csyr2_kernel_3`（LOWER strided） | `Csyr2RunAtomic` | `op.Process()` |
| `csyr2_kernel_5`（LOWER windowed） | `Csyr2RunAtomic` | `op.Process()` |
| `csyr2_kernel_lower_full_window_2 / 4` | 内联 `SetAtomicAdd`/`PipeBarrier`/`SetAtomicNone` | 删除，仅保留 `op.Process()` |
| `csyr2_kernel_upper_full_unit / full_window / 0 / 2 / 4` | `Csyr2RunAtomic` | **不变** |

UPPER 保留原子加：其写回范围严格落在操作区内（基线 cher2k 与 csyr2 UPPER 全过即为佐证），不产生非操作区写，原子加只承担「把更新量加到旧值上」的计算职责，不承担保护职责。

`csyr2_kernel_tiny` 也保留 `SetAtomicAdd`：该标量路径按 `[col, n)`（LOWER）/`[0, col+1]`（UPPER）严格裁剪写回，**写集合恰等于操作区**，不产生非操作区写；但它靠原子加完成「A + term」，若要改非原子需额外读旧值，属本任务范围外的改动。

#### 代码修改范围

只改 `arch22`，不动 arch35，不改公开头文件签名。

| 文件 | 类型 | 改动要点 |
| --- | --- | --- |
| `blas/syr2/arch22/csyr2_device_lower_guard.h` | 新增（100 行） | `LoadLowerOldCol` / `LoadLowerOldBatch` / `AccumulateLowerOld` / `RestoreLowerPrefix` |
| `blas/syr2/arch22/csyr2_device_lower_packed.h` | +28 | packed LOWER：读旧值 → 累加 → 位还原，`V_MTE2` 移到写出之后 |
| `blas/syr2/arch22/csyr2_device_lower_window.h` | +7 | window LOWER：同上，旧值按 `j0*lda + row0` 取回 |
| `blas/syr2/arch22/csyr2_device_chunks.h` | +27 / −4 | `PackedLowerChunk` 改 int32 位复制；`FinishInterleavedChunk` 无条件读旧值；`ChunkStep` 只判 `kUpper`；`FinishPlanarChunk` 补累加 |
| `blas/syr2/arch22/csyr2_kernel.cpp` | +30 / −12 | LOWER 入口去原子加；新增 `Csyr2LowerOneAccumFits` UB 预算守卫 |
| `blas/syr2/arch22/csyr2_ascend_kernel.h` | +18 | 新增 4 个成员函数声明、`aOldBuf_`、`aOldFloats_` |
| `blas/syr2/arch22/csyr2_device_init.h` | +19 / −1 | 三处 Init*Storage 末尾追加 `aOldBuf_`；`SelectLowerPacking` 预算补 `aOld` |
| `blas/syr2/arch22/csyr2_tiling.h` | +6 / −2 | `Csyr2SelectWindowBatchCount` 预算补 `aOld` |
| `blas/syr2/arch22/csyr2_host.cpp` | +4 / −1 | `Csyr2LowerDoubleBufferedColumnsFit` 预算补 `aOld` |
| `blas/syr2/arch22/csyr2_device.h` | +1 | 包含新增头文件 |
| `test/frame/check_untouched_bitwise.h` | 新增（82 行） | 非操作区逐 `complex64` `memcmp` 检查，接入三个 GTest |
| `test/syr2/csyr2/arch22/` | 用例 + 框架 | 新增 `TC_BIT_*`；`TC_ED_151` 期望值更正为 `INVALID_ENUM` |
| `test/her2k/cher2k/arch22/` | 用例 + 框架 | 接入 bit 检查，补 `TC_BIT_*` |
| `test/herk/cherk/arch22/`、`arch35/` | 用例 + 框架 | 接入 bit 检查（arch22 用宏 `CHERK_ARCH22_BIT_CHECK` 隔离，arch35 不参与）；`TC_FL_166` 定性为判据不可达 |

合计 `blas/syr2/arch22/` 内 9 个文件修改 + 1 个新增，`+122 / −18` 行。

#### Cher2k / Cherk 的对应处理

两个 Hermitian 算子的非操作区写回路径与 Csyr2 结构不同，基线表现也不同，因此改动范围按实测结论分档：

- **aclblasCher2k**（`blas/her2k/arch22/`）：基线 1029 全过，`cher2k_kernel_postprocess.cc` 的 `Cher2kSg3StoreChunk` 已用 `fullySelected` 判据把非操作区排除，`StorePartial` 还会用标量回填对齐空洞。因此**不改 kernel 写回**，只把 `check_untouched_bitwise` 接入 arch22 GTest 并补 `TC_BIT_*` 用例，把「已经正确」这件事变成可回归的断言。
- **aclblasCherk**（`blas/herk/arch22/`）：基线 1028/3，未通过项全部是 `TC_FL_166`（`fill_extreme`，UPPER）的**操作区**虚部——我们给出 `nan`，golden 给出 `-inf`，496/528 个元素不符。非操作区本身没有失败项，同样只接入 bit 检查。

  `TC_FL_166` **不修**，定性为「参考实现自身溢出导致判据不可达」：该用例 `a_fill=RANDOM_EXTREME` 且 `n=k=32`，操作区结果已越出 fp32 值域，golden 实部即为 `±inf`；用 Netlib 3.10.0 直接调 `cherk_` 复现同一输入，可确认 golden 虚部本身就是 `nan`/`-inf` 交替。要拿到逐元素一致，必须在 device 侧复刻 gfortran `__mulsc3` 的 C99 附录 G 无穷恢复规则与 Netlib 的累加/跳过顺序——这属于复刻编译器实现细节，不是算子正确性要求，且会引入显著的性能与回归风险。实测证据与替代判据建议见交付件 3 §5。

三个算子的修复互不共享 kernel 代码，因此可以放在同一 PR 中一起合入，也可拆成一组关联 PR（任务书 §5 允许）。

#### 性能影响分析

新增开销只有一项：每列多一次 `[row0, n)` 的 GM 读（`DataCopyPad`）与一次 UB 内 `Add`，被撤销的是原子加路径上**同量级**的一次「读—加—写」。也就是说，非操作区保护从「写后靠原子语义修正」变成「写前先取旧值」，全局访存量级不变；差别在于原子加本来由硬件在写通路里完成，现在显式成一条独立 DMA。前缀位还原只作用于每列最多 3 行（`prefix * 2 ≤ 6` 个 `float`），指令数可忽略。

实测口径以任务书 §3.3 三项标杆为门禁：Csyr2 `7.54 / 9.72 / 19.05 us`，Cher2k `1458.91 / 5240.68 / 1081.90 us`，Cherk `738.09 / 2616.28 / 485.94 us`。基线（pristine）实测为 `6.6935 / 7.1886 / 14.5172 us`，余量约 11% / 26% / 24%；
交付版本（含本设计全部改动）实测为 `7.14 ~ 7.23 us`（LOWER n=1024，A3 三轮），仍低于 9.72 us 标杆。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B3，`dav_c220`） | √ |
| Atlas 800I/T A3（`dav_c220`） | √ |
| Atlas 950PR（arch35） | 不涉及（本任务只改 arch22） |

## 算子约束限制

1. 仅列主序；矩阵、向量与复数标量为 `complex64`，Cherk 的 α、β 及 Cher2k 的 β 为 `float32`。
2. `ld ≥ max(1, n)`；`trans = N` 时输入矩阵为 `n × k`，`trans = C` 时为 `k × n`。
3. 三个 Hermitian 算子（Cher2k / Cherk）只计算 `uplo` 指定三角，另一三角既不读也不写；对角虚部置 0。
4. `n = 0`、`α = 0`、`k = 0`（配合 `β = 1`）等 quick return 条件下输出缓冲区逐 bit 不变。
5. 标量归属不变：Csyr2 的 α 在 Host，Cher2k / Cherk 的 α、β 在 Device；不得为判断标量而引入 Host 同步。
6. 不改 `include/cann_ops_blas.h` 的任何声明，不新增平行 API。

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 操作区精度 | 混合容差 `rtol = atol = 2⁻¹³`，`required_matched_ratio ≥ 0.99` | 任务书 §3.2.1 / CANN 混合容差标准 |
| 非操作区精度 | 逐 `complex64` `memcmp` 相等（`rtol = atol = 0`，ratio = 1.0），含 NaN payload、±0、次正规数 | 任务书 §2.0.2 |
| quick return | 整个 `ld × n` 缓冲区逐 bit 不变 | 任务书 §2.0.1 |
| 性能 | Csyr2 `7.54 / 9.72 / 19.05 us`；Cher2k `1458.91 / 5240.68 / 1081.90 us`；Cherk `738.09 / 2616.28 / 485.94 us` | 任务书 §3.3（Atlas 800I A2，warmup 后采样 > 50 次取平均） |
| 内存 | 不申请与 `n` / `k` 无关的持久 Device 缓存；不得「先写满整块再拷回」 | 任务书 §3.4，代码检视判定 |

测试工具链：复用任务包 `test_cases/{csyr2,cher2k,cherk}/` 的 CSV 与 `verify_accuracy.py`、`verify_performance.py`、`gen_csv.py`、`check_untouched_bitwise.py`；非操作区检查以 `test/frame/check_untouched_bitwise.h` 接入三个 arch22 GTest。A2 与 A3 均须给出精度与 bit 检查结果。

## 兼容性分析

1. **接口兼容**：`include/cann_ops_blas.h` 的签名、参数顺序、枚举、返回码全部不变，属纯实现层修复，调用方无需改动。
2. **行为兼容**：公式与 quick return 语义不变；唯一可见差异是非操作区不再被改写——这正是本次修复目标。
3. **架构兼容**：只修改 `arch22`（A2/A3）。`arch35`（950PR）保持原样，不受影响。
4. **数值兼容**：操作区结果仍在混合容差内，但 LOWER 路径的中间累加顺序由「GM 原子加」变为「UB 内 `Add`」，末位可能有差异，属容差允许范围；非操作区位模式由「不可预测」变为「与输入完全一致」。

# 已知问题与后续计划

## 容差口径：交付分支默认值与 §3.2.1 标准存在差异

本交付分支基于上游基线 `05b96ea`，其 `test/frame/verify.h` 中 `ACL_FLOAT` 的混合容差默认为
`rtol = 2⁻¹⁰ / atol = 2⁻¹⁶`；上游在该基线**之后**才把默认值对齐到任务书 §3.2.1 要求的 `2⁻¹³`。
因此自测主表（Cher2k `1043 / 1043`）是在交付分支自带框架口径下得到的，为使结论在标准口径下同样可查，
另按 `rtol = atol = 2⁻¹³` 复核了一轮（A2，同一提交 `4b1a633`）：

| 算子 | `2⁻¹³` 口径 | 说明 |
| --- | --- | --- |
| aclblasCsyr2 | **1226 / 1226** | 与主表一致 |
| aclblasCher2k | **1040 / 1043** | 3 条各超限 **1 个元素** |
| aclblasCherk | 1028 / 1029 | 仅 `TC_FL_166`（见下节） |
| `TC_BIT_*`（各两遍） | 22/22、24/24、24/24 | 容差收紧不影响 bit 校验 |

三条边界用例：

| 用例 | 形状 | matchedRatio | 相对限值 |
| --- | --- | --- | --- |
| `TC_EX_0356` | n=9, k=2048, UPPER, trans=N | 0.98889 | 1.04× |
| `TC_EX_0949` | n=4, k=1024, LOWER, trans=N | 0.95000 | 1.47× |
| `TC_EX_0950` | n=5, k=1024, UPPER, trans=C | 0.96667 | 1.23× |

**性质判定：判据可达，差距在实现侧。** 用 fp32 Netlib（`cblas_cher2k`）对同一输入、同一判据计算，
上述三条 `matchedRatio` 均为 **1.00000**，最坏元素只到限值的 0.23 / 0.58 / 0.44 倍。

**根因**：`arch22` 的 cube 路径将复数乘拆为四条实矩阵乘链（4M），四个实积分别累加到 L0C 之后，
才在 epilogue 合成 `S1 − S2`；对相消型元素（中间量 ~1e3、结果 ~3），比逐 `k` 合成多吃一次舍入。

**已验证的缓解与待办**：把 `n ≤ 32` 改道到带 Kahan 补偿的标量路径后，`TC_EX_0949` 由 0.95 变为
**1.00000**（`TC_EX_0950` 亦由 0.96667 变为 1.00000），说明精度余量存在；但该标量路径对
`LOWER + trans = N`、`ldc` 非 4 对齐等布局并不适用（整体改道会使 Cher2k 全量挂 73 条），
需要精确化改道条件。计划在提 PR 合入 `ops-blas`（任务书 §5）前完成并复跑双机全量。

## `TC_FL_166`（Cherk）：参考实现自身溢出，判据不可达

该用例 `a_fill = RANDOM_EXTREME`、`n = k = 32`，操作区结果已越出 fp32 值域，golden 实部即为 `±inf`，
虚部为 `nan` / `-inf` 交替（由 gfortran 复数乘法的 C99 附录 G 无穷恢复规则产生）。
即使虚部恒为 0 的完全正确实现，也只能得到 `324 / 528 = 61.4%` 的匹配率，低于 0.99 的口径，
属判据不可达而非实现缺陷；实测证据与替代判据建议见自测报告 §5。本次不修。

## 非操作区：字面「零写」的已知限制

任务书 §2.5.4 要求对非操作区与 `ld` padding 行「零写」，§2.0.3 同时把「先读后写且只提交操作区」列为合法解法。
当前实现属于前者允许的「先读后写 + 逐位还原」：LOWER 每列写回起点为 4 复数对齐，会读取对角线上方
至多 3 个（整窗分组至多 7 个）复数用于原位保留，写出的字节与调用前逐 `complex64` 逐 bit 相同，
因此满足 §2.0.2 的 `memcmp` 判定口径。字面零写的方案实测可达（`TC_BIT_*` 22/22、`TC_SQ_*` 46/46 全过），
但 `TC_PF_1002`（LOWER n=1024）由 `7.0~7.2 us` 抬到 `10.0~10.5 us`（+45%），超出 §3.3 的 `9.72 us` 标杆，
故交付版本采用前者；详见 §3.2.2。
