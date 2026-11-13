# aclblasCrot 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-ops-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasCrot 算子开发任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/rot/arch22/`（csrot）、`blas/rot/arch35/srot_host.cpp`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasCrot 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/rot/arch22/`（实现）与 `test/rot/crot/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数 Givens 平面旋转能力（BLAS-1），
对标 cuBLAS `cublasCrot`（语义参考 Netlib/LAPACK `crot.f`）。

## 背景介绍

### 算子功能定位

BLAS-1 复数 Givens 平面旋转：c（实数）与 s（复数）对复向量 x、y 施加旋转，x、y 原地更新。

数学公式：

```
x'[k] = c × x[k] + s × y[j]
y'[j] = c × y[j] − conjg(s) × x[k]      （x[k] 取旋转前原始值）
k = 1 + (i−1)×incx, j = 1 + (i−1)×incy, i = 1..n
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasCrot | cublasCrot | 单精度复数 CROT：复数向量旋转，c 实数/s 复数，conjg(s) 参与 y 更新，正负步长 |

### 业界对标

- 对标接口：cuBLAS `cublasCrot(handle, n, x, incx, y, incy, c, s)`；cuBLAS 文档 rot 条目末尾明确引用 netlib crot()。
- Golden 语义来源：**Netlib 参考库无 crot**——参考实现来自 LAPACK 辅助例程
  `crot.f`（L75-77：y' = c·y − conjg(s)·x）；golden 在测试工程内按该语义实现（任务书 §3.2）。
- 语义差异点（相对仓内既有实现）：
  1. **与 csrot 的参数差异**：csrot（rot/arch22 已有）为"实参数复向量"（c、s 均 float）；本算子
     c 为 float、**s 为复数**，且 y 更新含 `conjg(s)`——复数 Givens 参数数学，复乘/共轭逐字对齐 Netlib crot.f；
  2. **与 Srot（同批任务）的口径一致性**：n<0 → INVALID_VALUE、n=0 快速返回（arch22 家族契约，
     rot/README.md per-arch 差异表，issue #827 已声明；与 950PR/arch35 的 n≤0 放行口径互为已声明差异）；
  3. **负步长支持**：从向量尾端起算沿负方向步进（Netlib crot `IX = (−N+1)*INCX + 1` 语义）——
     与同批次 Scasum/ICAMIN 的"负步长 quick return"不同，Crot **实现负步长遍历**；
  4. **c/s 位置自适应**：Host 或 Device 内存指针，`aclrtPointerGetAttributes` 自动判定
     （依据 rot/README.md 同族 aclblasSrot 约束说明）；
  5. **quick return 不得先于判空**（#765/#861 维护者拍板口径——校验顺序约束）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/rot/arch22/`（该目录**已存在**，当前仅 csrot → **并列加文件**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同目录 arch22 csrot（主复用源） | `blas/rot/arch22/csrot_{host,kernel}.*` | arch22 工程骨架、校验序、元素对原地更新（x 旧值暂存）流水范式；复数化改造：c/s 双类型、conjg(s) 乘、c/s 双位置探测 |
| 同目录 arch22 同批兄弟 | `blas/rot/arch22/`（本批 Srot 任务交付的 srot arch22 文件） | 步长 DMA 符号统一（同/异号反转判定）与 c/s 位置探测范式直接共享 |
| arch35 参考 | `blas/rot/arch35/srot_host.cpp`（strided 溢出守卫 L116-129）、`blas/rot/arch35/crot_*`（若有） | 校验序与 int64 双边守卫参考；kernel 不平移（SIMT） |

### 接口规格

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

- **现场核验：`include/cann_ops_blas.h` L250 已存在该声明**（签名与任务书 §2.3 完全一致；
  任务书 §2.3 写"尚无声明、需新增"与仓内实际不符——以仓内为准，**本次不修改公共头**，
  与 950PR 共用声明，禁止另立私有平行接口 → 红线 3 不触发）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| n | x、y 元素个数，Host 内存 | int | int32 | n≥0；n<0 → INVALID_VALUE；n=0 合法快速返回（成功，不修改 x/y） | — |
| x | 复数向量，Device 内存，原地更新为 c·x + s·y | aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；元素值不校验 | 一维 [n]，物理长度 `1+(n−1)*|incx|` |
| incx | x 步长（可正可负），Host 内存 | int | int32 | ≠0 且非 INT32_MIN；0/INT32_MIN 或 `(n−1)*\|incx\|` 溢出（int64，issue #866）→ INVALID_VALUE | — |
| y | 复数向量，Device 内存，原地更新为 c·y − conjg(s)·x（x 取原始值） | aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；元素值不校验 | 一维 [n]，物理长度 `1+(n−1)*|incy|` |
| incy | y 步长（可正可负），Host 内存 | int | int32 | 同 incx | — |
| c | 旋转余弦（实数）指针，Host 或 Device 内存（运行时探测） | const float* | FLOAT32 | n>0 时 nullptr → INVALID_VALUE；值不校验 | [1] |
| s | 旋转正弦（复数）指针，Host 或 Device 内存（运行时探测） | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；值不校验 | [1] |

### 数学公式

```
x' = c·x + s·y
y' = c·y − conjg(s)·x（原始 x）
```

复数运算：复乘 `(a+bi)(c+di) = (ac−bd) + (ad+bc)i`；`conjg(a+bi) = a−bi`；
实数 c 乘复数为实/虚分别缩放。逐元素对独立、先读后写（x 旧值用于两式）。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-1 单精度复数 CROT：COMPLEX64 向量、c(FLOAT32)/s(COMPLEX64)、
正负步长（负步长真实遍历）、原地双输出；实部/虚部分别按 FLOAT32 档判定（rtol=atol=2^-13，ratio≥0.99）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | 复数 Givens 旋转（c 实、s 复、y 更新含 conjg(s)，x 旧值语义）。golden：测试工程内按 Netlib/LAPACK crot 语义（L75-77）实现（Netlib 参考库无 crot），实/虚分别比对 |
| 2 | 接口原型 | 8 参签名（上节）；声明已存在于 `include/cann_ops_blas.h` **L250**（现场核验，任务书"需新增"与实际不符），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | x/y/s **COMPLEX64**，c **FLOAT32**（任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 侧 rot 族 arch35 已有实现，不受影响 |
| 6 | 精度要求 | 实/虚分别 FLOAT32 档：rtol=atol=2^-13、ratio 0.99、max_abs_error ≤ 1e-2 or 32×ULP；Inf/NaN 按 §3.2 等效规则（mixed_tolerance_standard，opbase commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定。典型 case（实测 CSV 读数，全连续 incx=incy=1）：n=1048576 → ≤23.72us；n=2097152 → ≤56.92us；n=4194304 → ≤109.33us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase：csrot arch22 骨架 + 复数参数化**（连续双缓冲流水 + 步长 DMA 符号统一）。死路①：arch35 kernel 平移（SIMT）；死路②：把 s 当实数处理（丢 conjg 语义，判据直接失败）；死路③：Gather 窗口方案（降级备选，非首选） |
| 9 | 评测来源 | 精度 golden：LAPACK crot 语义 CPU 实现；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V7，由任务书 §2.5 校验顺序条款逐字反推，顺序不可调换）**：

| 序 | 条件 | 返回码 |
|----|------|--------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（快速返回：不修改 x/y） |
| V4 | `x == nullptr \|\| y == nullptr \|\| c == nullptr \|\| s == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V5 | `incx == 0 \|\| incy == 0 \|\| incx == INT32_MIN \|\| incy == INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `(n−1)*\|incx\| 或 (n−1)*\|incy\| 超 int32 范围（int64，issue #866 口径）` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V7 | 其余 | `ACLBLAS_STATUS_SUCCESS` |

> V1 判空最优先；V3（n=0 快速返回）按任务书 §2.5 顺序列在 inc/指针检查之前
> （quick return 不得先于 handle 判空——#765/#861 口径已由 V1 先行保证）。

---

# 详细设计（required）

## 算子分析

### 数学公式

`[[c, s], [−conjg(s), c]]·(x[k], y[j])`，元素对内两式共享 x 旧值；逐元素对独立、无跨元素依赖。
确定性：单次执行内确定；同输入同步长跨次 bit 级一致（任务书 §2.5"不要求"但天然满足）。

### 支持数据类型

x/y/s 为 COMPLEX64（分量 FLOAT32，交错存储），c 为 FLOAT32；中间计算保持 FLOAT32。
复数乘分解：`c·(a+bi) = (c·a) + (c·b)i`（实数缩放，2 Mul）；
`s·y = (sr·yr − si·yi) + (sr·yi + si·yr)i`；`conjg(s)·x = (sr·xr + si·xi) + (sr·xi − si·xr)i`。
特殊值按 IEEE 传播（含 0×Inf 不定式），按 §3.2 等效规则判定。

### 支持形状

| 场景 | 约定 |
|------|------|
| 一维向量 | x/y 物理长度按步长展开；调用方保证存储有效 |
| n = 0 | 合法快速返回：成功、不修改 x/y、不启动 kernel |
| n < 0 | 非法：INVALID_VALUE（arch22 家族契约，issue #827） |
| n = 1 | 单元素对，步长无实际意义 |
| 步长语义 | 正负步长**均真实遍历**（负步长自物理 span 尾端起算，Netlib crot IX 语义）；0/INT32_MIN 拒绝 |
| 广播 / 0 维 | 不涉及（BLAS-1） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_DENSE** | `incx==1 && incy==1` | 连续双缓冲流水：DataCopy x/y → 复数旋转计算 → DataCopy 写回 | 是（基线 3 条均连续） |
| **V_STRIDED** | 其余（正/负/混合步长） | 步长 DMA 读 x/y → 旋转计算 → 步长 DMA 写回 | 否（精度覆盖） |

数据流（V_DENSE，双缓冲）：

```
GM x(2n float) ──DataCopy──▶ UB xBuf ─┐
                                       ├─▶ newX = c·x + s·y      （UB 内向量指令）
GM y(2n float) ──DataCopy──▶ UB yBuf ─┘   newY = c·y − conjg(s)·x（x 用旧值）
                                            │
                                UB xBuf'/yBuf' ──DataCopy──▶ GM x / GM y
```

> x 旧值语义：先算 newY 存入独立缓冲（或先算 newY 再覆写 x），保证第二式读取原始 x——
> 与同目录 csrot arch22 的元素对先读后写流水同范式。

#### host 侧设计

**分核策略**：
- 核数 `numBlocks = min(GetAivCoreCount(), n)`；每核负责连续逻辑区间，kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4）。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- V_DENSE tileSize 按 2n float 计：xBuf/yBuf 双缓冲各 2 槽 + newBuf，tile 取 4096 复数元素量级；
- c/s 位置探测：host 侧 `aclrtPointerGetAttributes`——Host → 值直填 tiling（零 H2D）；
  Device → 指针经 tiling 下发，kernel 开头一次性读入 UB 标量缓冲；
- 尾块 DataCopyPad 补零对齐（32B 块 = 4 复数），计算按有效 calCount。

**tilingkey 规划策略**：tiling 字段分支（incx/incy 连续/步长）；Tiling 字段：n、incx、incy、
perCoreN、remainder、tileSize、c（float）、s（2×float，Host 态直填）、cIsDevice、sIsDevice
（对齐同目录 Srot arch22 tiling 规划）。

**数据检测**：按 V1..V7 顺序执行（见需求拆解节）；V6 用 int64 双边守卫（>INT32_MAX 或 <INT32_MIN，
负步长下乘积为负——对齐 arch35 strided 守卫 L116-129）。

**其他 host 侧约定**：
- workspace：无（无跨核归约）；
- 同步：异步下发，host 不同步（任务书 §2.5）；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策与 c/s 位置探测结果。

#### kernel 侧设计

**Init/Process 两阶段**（V_DENSE）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(xBuf/yBuf/newBuf)
Process:
   ┌── for each tile (双缓冲) ─────────────────────────────────────┐
   │  MTE2: DataCopy xBuf ← GM x（2×cnt float）；yBuf ← GM y        │
   │  MTE2→V: SetFlag/WaitFlag                                     │
   │  V:    newY = c·y − conjg(s)·x  （x 用旧值）                    │
   │        newX = c·x + s·y                                        │
   │        （c 缩放：实/虚通道 Muls；复乘：4 Mul+2 Add；conj：虚部取反）│
   │  V→MTE3: SetFlag/WaitFlag                                     │
   │  MTE3: DataCopy GM x ← newX；GM y ← newY                       │
   │  MTE3→MTE2: SetFlag/WaitFlag                                   │
   └────────────────────────────────────────────────────────────────┘
   └── 尾块：DataCopyPad 补零对齐 → 计算 → 精确写回
```

**步长路径（V_STRIDED）符号统一**（同批 Srot 设计同款推导）：
- x/y 步长**同号** → 物理递增序下逻辑域一致 → 0 次反转直接配对；
- **异号** → UB 内反转其一（同目录 caxpy arch22 normalize 范式）；
- 每 tile DMA 基址 `(inc>0) ? t0*|inc| : (n−1−t0−cnt+1)*|inc|`；
- `asc_copy_gm2ub_align(src_gap=(|inc|−1)*8)` / `ub2gm_align(dst_gap=(|inc|−1)*8)`（复数 8B burst）。

**workspace 布局**：无 workspace。

**同步事件配对**：MTE2→V、V→MTE3、MTE3→MTE2 三段显式 SetFlag/WaitFlag；UB 内反转（步长路径）后加全流水栅栏。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | 一次调用只发射一个 AIV kernel |
| 双缓冲流水 | xBuf/yBuf 各 2 槽，MTE2 与 V/MTE3 重叠 |
| 复数运算通道化 | c 缩放（2 Mul）+ 2 次复乘（各 4 Mul+2 Add）+ conj（虚部 Muls −1）全 UB 向量指令，无逐元素标量 |
| host 零同步 | tiling const 引用；c/s Host 态零额外 H2D（最常见路径） |
| 满 32B 对齐 DMA | tile 对齐 4 复数（32B），尾块 DataCopyPad 精确写回 |

**未采用的路线及原因**：
- **SIMT**：arch22 不提供——死路；
- **s 按实数简化**：丢 conjg(s) 语义，与 golden（crot.f L75-77）不符——死路；
- **Gather 窗口方案**（`blas/spr/arch22/sspr_kernel.cpp` 范式）：仅作 4 字节突发 DMA 不可用时的降级备选。

---

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **支持**（本任务目标） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **支持**（本任务目标） |
| Ascend 950PR / Ascend 950DT | 不涉及（rot 族 arch35 既有实现提供，本次不改） |

---

## 算子约束限制

逐条对齐任务书 §2.5：

| 约束项 | 结论 |
|--------|------|
| 参数合法性 | V1..V7 全序：handle 判空最优先 → n<0 INVALID_VALUE → n=0 快速返回成功 → x/y/c/s 判空 → inc 0/INT32_MIN → int64 跨度守卫；c/s 标量值不校验 |
| 非连续 Tensor 支持 | 不要求额外 leading dimension padding：incx/incy 语义内支持，正负步长均真实遍历（负步长尾端起算）；零步长与 INT32_MIN 拦截 |
| broadcast 规则 | 不涉及（两一维向量逐元素旋转） |
| dynamic shape 要求 | 不要求：n 运行时入参 |
| 原地与视图语义 | x、y 原地更新（y 的更新必须使用旋转前原始 x——实现先暂存再覆写）；不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | n=0 合法快速返回（成功、不修改 x/y）；n<0 INVALID_VALUE |
| 异步执行 | 依赖 aclblasSetStream；读回 Device 结果前须同步 stream；c/s 位置运行时自动判定 |

**已知限制（如实标注）**：
1. arch35（n≤0 放行）与 arch22（n<0 报错）的 per-arch 差异已在 rot/README.md 声明（issue #827），README 同步维护；
2. 步长路径 UB 内反转为标量 swap（同族范式），不参与性能判定；
3. 现场核验头文件 L250 已有声明——任务书"需新增声明"表述与仓内实际不符，以仓内为准（不修改公共头）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | 测试工程内按 Netlib/LAPACK `crot` 语义实现的 CPU 参考单元（含 conjg(s)、x 旧值；cblas 复数实现口径），实部/虚部分别落盘 | 任务书 §3.2 |
| 比对方式 | x、y 两向量原地更新后分别比对，实部/虚部分别按 FLOAT32 档 | 任务书 §3.2 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4 | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | 逐元素 `\|actual − golden\| ≤ atol + rtol*|golden|` | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；非有限值不计入统计、单独记录 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{crot_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/rot/crot/arch22/`（仓内 test/rot/crot/ 当前只有 arch35，arch22 参照 arch35 模式新建），并在 `test/rot/crot/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：n ∈ {0, 1, 2, 3, 7, 2 的幂及 ±1, …, ≥2^20}（n=0 no-op、n<0 负向）；incx/incy ∈ {1,2,3,−1,−2,−3}（正常，含负步长真实遍历）；0/INT32_MIN 负向；c ∈ {1, 0, −1} 及分布值；s ∈ {(0,0),(1,0),(0,1),(−1,0)} 及分布值（实/虚独立采样）；x/y 含 Inf/NaN；跨度溢出负向；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|inc|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **无 ABI 变更**：`aclblasCrot` 声明已存在于 `include/cann_ops_blas.h`（L250，现场核验），本次不改公共头；
- **对 arch35 影响**：无。按 SOC 选目录编译互不覆盖；
- **对 arch22 同目录 csrot / 同批 srot 影响**：无。并列新文件（aclblasCrot vs aclblasCsrot vs aclblasSrot），
  接口独立、无共享可变状态；步长符号统一与 c/s 位置探测范式通过新文件内实现；
- **测试构建影响**：`test/rot/crot/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支不变；
- **文档影响**：`blas/rot/README.md` 产品支持表新增 aclblasCrot「Atlas A2/A3：支持」，并同步 per-arch 差异表
  （n<0 口径、conjg(s) 语义、负步长真实遍历）。
