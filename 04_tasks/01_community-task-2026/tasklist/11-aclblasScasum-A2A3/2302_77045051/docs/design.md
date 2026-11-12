# aclblasScasum 算子设计文档（Atlas A2/A3，arch22）

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclblasScasum |
| 算子功能 | 单精度复数（complex64）向量绝对值分量之和：result = Σ (\|Re(x_k)\| + \|Im(x_k)\|) |
| 对标接口 | cuBLAS `cublasScasum`（语义参考 Netlib BLAS `scasum.f`） |
| 目标硬件 | Atlas A2 系列产品（性能设备 Atlas 800T A2 / 910B3）、Atlas A3 系列产品（arch22） |
| 工程模式 | ops-blas 开源仓 Ascend C kernel 直调（句柄式 BLAS 接口，handle 绑定 stream） |
| CANN 版本 | CANN 9.1.0 |
| 代码合入目录 | 实现 `blas/asum/arch22/`，测试 `test/asum/scasum/arch22/` |

---

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务《11月社区任务-aclblasScasum算子开发（A2A3）》（任务书：[aclblasScasum_AtlasA2_A3_task_doc.md](./aclblasScasum_AtlasA2_A3_task_doc.md)）。要求在 Atlas A2/A3 系列 NPU（arch22）上使用 Ascend C 开发 `aclblasScasum` 算子，完成设计、开发、测试全流程，验收后以 PR 形式合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas)（实现目录 `blas/asum/arch22/`，测试目录 `test/asum/scasum/arch22/`）。

## 背景介绍

### 算子功能定位

`aclblasScasum` 是 BLAS Level-1 复数归约算子：对单精度复数向量 x 各元素的实部与虚部分别取绝对值后求和，返回一个 float 实数标量。数学表达式为（i = 1..n，k = 1+(i-1)\*incx，1-based 兼容 Fortran 约定）：

```
result = Σ ( |real(x[k])| + |imag(x[k])| )
```

该算子与实数版 `aclblasSasum`（L1 范数）同族，常用于复数向量的稀疏度度量与误差估计。

### 对标基线语义

- cuBLAS `cublasScasum`：参数序列 (handle, n, x, incx, result)，x 为 complex64 交错存储（实部/虚部各 float32），物理长度 1+(n-1)\*|incx| 个复数元素；输出为 float 实数标量。
- Netlib 参考实现 `scasum.f`：`IF (N.LE.0 .OR. INCX.LE.0) RETURN`，即 **n <= 0 或 incx <= 0 时不遍历向量、结果为 0**。
- 任务书 §2.1-3 明确 quick return 语义：n <= 0 或 incx <= 0（含 0 与负步长）时不触发 kernel，直接写 `result = 0.0f` 并返回成功。**注意与批次 1 Scnrm2 的"incx=0 报错、负步长正常计算"口径不同，本算子以 cuBLAS 文档 / Netlib scasum.f / 任务书 / 仓内 Sasum README 四方一致口径为准。**

### ops-blas 仓现状分析

| 项目 | 现状 | 本任务动作 |
| --- | --- | --- |
| 接口声明 `include/cann_ops_blas.h:244` | `aclblasScasum` 已声明（与 950PR 共用），禁止定义私有平行接口 | 无需改动，直接实现 |
| 复数类型 `include/cann_ops_blas_common.h:68` | `aclblasComplex {float real; float imag;}`，交错存储 | 直接使用 |
| `blas/asum/arch35/`（950PR） | scasum 已有实现（SIMT kernel + tiling 按值传参，含参数校验 / quick return / 跨度溢出守卫 issue #805） | 作为语义与 host 范式参考 |
| `blas/asum/arch22/`（A2/A3） | 仅有实数 `sasum`（高层 Ascend C AIV 向量范式：TPipe/TQue + Abs + ReduceSum + SetAtomicAdd），**scasum 缺失** | 新增 `scasum_host.cpp / scasum_kernel.cpp / scasum_tiling_data.h` |
| `blas/asum/README.md` | 产品支持表中 aclblasScasum 标注 A2/A3"不支持" | 更新为"支持" |
| `test/asum/scasum/CMakeLists.txt` | 仅 arch35 分支注册 GTest，A2/A3 构建退回非 GTest runner（输出整数毫秒） | 补齐 arch22 分支 `ops_blas_add_gtest_tests` 注册 |
| `test/asum/scasum/` | `scasum_param.h`（CSV 参数）、`scasum_golden.h`（双精度 CPU golden）已存在且 arch 无关，可直接复用 | 新增 `arch22/scasum_test.cpp + scasum_test.csv` |

arch22 关键现状补充：

1. **arch22 全仓无 SIMT 用例**（SIMT 为 arch35 能力），A2/A3 上必须采用 AIV 向量范式开发 kernel。
2. arch22 `sasum_host.cpp` 硬编码 8 核下发、host 每调用 `aclrtSynchronizeStream`，且缺少 result/x 空指针校验与 quick return（旧范式）；arch22 `snrm2_host.cpp` 已具备本任务要求的完整校验 + quick return 范式，arch35 `scasum_host.cpp` 具备跨度溢出守卫（issue #805）。host 设计取三者之长。
3. 仓内源码按 `blas/<op>/<arch>/` 目录自动发现编译，新增文件即被纳入构建，无需改动中央构建脚本。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 arch22（Atlas A2/A3）上实现 `aclblasScasum` 句柄式 BLAS 接口，与 cuBLAS `cublasScasum` 核心功能、参数语义完全对齐；通过 ops-blas 仓工程框架直调 NPU kernel；配套 CSV 驱动的 GTest 测试工程，精度满足生态算子开源精度标准（FLOAT32 标量判定），性能达到 GPU（A100 / cuBLAS）标杆的 0.8 倍以上（平均单次耗时 ≤ 1.25× 标杆）。

## 需求拆解

1. **功能对齐**：计算 Σ(|Re|+|Im|)，x 为 complex64（2n 个 float 交错），输出 float 标量；incx 步长语义对齐（k = 1+(i-1)\*incx）。
2. **边界与异常语义**：
   - handle == nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - result == nullptr（任意 n，含 quick return 路径）→ `ACLBLAS_STATUS_INVALID_VALUE`；
   - n > 0 且 incx > 0 时 x == nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；
   - n <= 0 或 incx <= 0 → quick return：不触发 kernel，写 result = 0.0f，返回 `ACLBLAS_STATUS_SUCCESS`；
   - (n-1)\*|incx| 超出 int32/uint32 索引域 → `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，对齐 issue #805）。
3. **精度要求**：标量比对，FLOAT32 标准：atol = rtol = 2⁻¹³（≈1.22e-4），通过条件 `|actual − golden| ≤ atol + rtol×|golden|`，且 max_abs_error ≤ 1e-2 或 32×ULP；Inf/NaN 场景按等效性规则判定（同为 NaN / 同号 Inf / NaN↔±Inf 视为通过）；golden 由 CPU 双精度参考实现生成。
4. **性能要求**：Atlas 800T A2 (910B3) 上，FLOAT32 输入场景平均单次耗时（msprof op 采集单算子 task duration，warmup 5 次 + 有效采样 ≥10 次取均值）不高于 A100 标杆的 1.25 倍。
5. **工程要求**：实现进 `blas/asum/arch22/`；测试进 `test/asum/scasum/arch22/` 并在 CMake arch22 分支注册 GTest；README 产品支持表更新；代码遵循仓内贡献规范，无冗余文件。

---

# 详细设计（required）

## 算子分析

### 数学公式

```
result = Σ_{i=1}^{n} ( |Re(x_k)| + |Im(x_k)| ),   k = 1 + (i-1) * incx
```

- 输入：x ∈ complex64^（物理长度 1+(n-1)\*|incx| 个复数元素，交错存储），Device 只读；
- 输出：result ∈ float 实数标量，Device 内存；
- 非负累加：每项 |Re|+|Im| ≥ 0，n 大且元素接近 FLT_MAX 时结果可自然溢出至 Inf（golden 同为浮点累加，按 Inf 一致性判定）。

**关键恒等式（本设计的技术核心）**：复数元素交错存储（Re₀ Im₀ Re₁ Im₁ …），对每个元素取 |Re|+|Im| 再求和，等价于把 x 视作长度 2n 的 float 向量后逐元素取绝对值求和：

```
Σ_k (|Re(x_k)| + |Im(x_k)|)  ≡  Σ_{j=0}^{2n-1} |float_view(x)[j]|      （incx = 1 时）
```

即 **incx = 1 时 scasum 在 float 视图上严格等于 2n 元素的 sasum**。复数归约退化为实数归约，无需实虚分离（GatherMask/解交错），可直接复用 arch22 sasum 已验证的向量化数据通路（DataCopy + Abs + ReduceSum），这是性能与可维护性的最优路径。incx > 1 时元素的 float 下标为 2\*(k\*incx) 与 2\*(k\*incx)+1，仅 CopyIn 寻址方式不同（见 §kernel 侧设计）。

### 支持数据类型

| 输入/输出 | 数据类型 | 说明 |
| --- | --- | --- |
| x | COMPLEX64（`aclblasComplex`，实/虚部各 float32 交错） | 值域覆盖 FLOAT32 全集（含 Inf/NaN/±0/次正规数），元素值不校验 |
| result | FLOAT32 实数标量 | 非负实数；溢出场景可为 +Inf；含 NaN 输入时可为 NaN |
| n / incx | int（Host 内存） | 任意整数；n ≤ 0 或 incx ≤ 0 走 quick return |

### 支持形状

- x：逻辑一维 [n]，物理长度 1+(n-1)\*|incx| 个复数元素（ND 排布，无 leading dimension padding）；
- result：单值标量；
- 归约到标量的单向量算子：不涉及 broadcast、不涉及 dynamic shape（n 为运行时标量入参）、不涉及原地/视图语义。

## 算子实现

### 实现方案

**工程模式**：基于 ops-blas 仓 Ascend C kernel 直调框架。`aclblasScasum`（host）完成参数校验、quick return、tiling 计算与 kernel 下发；kernel 采用 **AIV 向量范式**（高层 Ascend C API：TPipe/TQue/GlobalTensor/DataCopy/ReduceSum），整体数据通路复用 arch22 sasum 的成熟结构，扩展复数 float 视图路径。

**技术路线选型**（依据仓内调研结论）：

| 决策点 | 选择 | 依据 |
| --- | --- | --- |
| kernel 范式 | AIV 向量范式（高层 API） | arch22 全仓无 SIMT 能力；arch22 sasum 同范式已验证；任务书 §2.2 指定参考 sasum 扩展复数路径 |
| 复数处理 | float 视图恒等式（不解交错） | Σ(\|Re\|+\|Im\|) ≡ Σ\|float\|，免去 GatherMask 实虚分离的额外 UB 开销与指令数 |
| 跨核归约 | 主选 GM 标量 SetAtomicAdd；备选 workspace 部分和 + 0 号核终归约 | 主选方案为 arch22 sasum 已验证路径（免 workspace、免 SyncAll 终归约、时延最低）；备选为 arch22 snrm2/cdot 范式，若原子加路径精度或争用不达阈值再切换（详见 §3.2.4） |
| 分核数 | 运行时 `GetAivCoreCount()` 满核（上限裁剪），不沿用 sasum 硬编码 8 | 本算子带宽受限，8 核无法打满 HBM 带宽，难以达到 0.8×A100 性能判据 |
| tiling 传参 | 结构体按值传 kernel 入参（对齐 arch35 scasum）；回退 GM 指针 | 按值传参免去每调用 aclrtMalloc/Memcpy/SynchronizeStream/Free，保住异步语义与小 n 时延；若 arch22 工具链对结构体按值入参有约束，回退为 sasum 的 GM 指针方式（kernel 解析逻辑不变） |

#### host 侧设计

`blas/asum/arch22/scasum_host.cpp`，入口 `aclblasScasum(handle, n, x, incx, result)`，流程：

```
aclblasScasum(handle, n, x, incx, result)
 ├─ 1. 参数校验（ValidateScasumParams，对齐 §2.4 异常行为表与仓内范式）
 │     ├─ handle == nullptr            → return ACLBLAS_STATUS_HANDLE_IS_NULLPTR
 │     ├─ result == nullptr（任意 n）   → return ACLBLAS_STATUS_INVALID_VALUE
 │     └─ n > 0 且 incx > 0 且 x == nullptr → return ACLBLAS_STATUS_INVALID_VALUE
 ├─ 2. quick return（校验通过后判定，对齐 cuBLAS/Netlib：N.LE.0 .OR. INCX.LE.0）
 │     └─ n <= 0 或 incx <= 0：aclrtMemcpy(result, 4B, &0.0f, H2D)（stream 序，不 launch kernel）
 │           → memcpy 失败返回 ACLBLAS_STATUS_EXECUTION_FAILED，成功返回 ACLBLAS_STATUS_SUCCESS
 ├─ 3. 跨度溢出守卫（issue #805 口径，kernel 侧索引为 uint32）
 │     └─ span = (int64)(n-1) * (int64)incx > UINT32_MAX → return ACLBLAS_STATUS_INVALID_VALUE
 ├─ 4. tiling 计算（CalcScasumTilingData，复数元素粒度，紧凑结构）
 │     ├─ aivCoreNum = GetAivCoreCount()；为 0 视为异常返回 ACLBLAS_STATUS_EXECUTION_FAILED
 │     ├─ useCoreNum = min(aivCoreNum, SCASUM_ARCH22_MAX_CORE_NUM, (uint32)n)，下限 1
 │     ├─ perCoreN   = n / useCoreNum
 │     └─ remainder  = n % useCoreNum   // 前 remainder 个核多处理 1 个复数元素
 └─ 5. kernel 下发（不做每调用 aclrtSynchronizeStream，保持异步语义，§2.5）
       └─ scasum_kernel<<<useCoreNum, tiling(stream)>>>  // kernel 失败状态透传（#764 范式）
```

**tiling 结构**（`scasum_tiling_data.h`，参考 arch35 `scasum_tiling_data.h` 紧凑式）：

```cpp
constexpr uint32_t SCASUM_ARCH22_MAX_CORE_NUM = 64;  // arch22 AIV 核数上限保护（910B3 实际由运行时查询）

struct ScasumTilingData {
    uint32_t n;          // 复数元素个数
    int32_t  incx;       // 复数元素步长（下发路径上恒 > 0）
    uint32_t useCoreNum; // 实际启用核数
    uint32_t perCoreN;   // 每核基础复数元素数
    uint32_t remainder;  // 前 remainder 个核各多 1 个元素
};
```

核间任务由 kernel 侧按公式派生（与 host 序列化数组等价分布，且 tiling 结构体不随核数膨胀）：

```
offset(i) = i * perCoreN + min(i, remainder)
count(i)  = perCoreN + (i < remainder ? 1 : 0)
```

**设计要点说明**：

1. **校验顺序**：result 空指针判定先于 quick return 判定——任务书 §2.4 要求"result 为 nullptr 时（任意 n，含快速返回路径）返回 INVALID_VALUE"，测试用例 `n0_null_result / incx0_null_result / neg_n_null_result` 验证该顺序。
2. **quick return 写 0 的方式**：result 是 Device 内存，host 侧不直接解引用，采用 4B `aclrtMemcpy` H2D（与 arch22 snrm2、arch35 scasum 一致）。memcpy 走默认同步语义，结果即刻可见，调用方随后读回无需额外同步（保险起见测试侧仍同步 stream）。
3. **跨度守卫在 kernel 下发前**：kernel 侧 (offset+i)\*incx 的 float 下标 2\*((k)\*incx) 以 uint32 计算须保证不回绕；host 以 int64 预判 `(n-1)*incx > UINT32_MAX` 即拒绝，覆盖 INT32_MAX 步长等极端组合（用例 `incx_span_overflow`：n=4, incx=INT32_MAX）。
4. **异步语义**：正常路径只下发 kernel 即返回 SUCCESS，不同步流；由调用方在读回 result 前 `aclrtSynchronizeStream`（README 调用示例与测试工程均遵循）。tiling 按值传参天然异步安全；若回退 GM 指针方案，则须维持 sasum 的"launch → SynchronizeStream → Free"顺序。

#### kernel 侧设计

`blas/asum/arch22/scasum_kernel.cpp`，模板类 `ScasumAIV<T>`（T = float，统一以 float 视图处理），三阶段流水（CopyIn → Compute → CopyOut），double buffer 搬数计算重叠。

**UB 资源规划**（arch22 UB = 248KB，对齐 sasum 实测参数）：

| 缓冲 | 大小 | 用途 |
| --- | --- | --- |
| inQueue（VECIN，BUFFER_NUM=2） | 2 × 80KB（每 tile 20480 个 float） | GM→UB 输入数据，ping-pong 重叠 |
| workBuf（VECCALC） | ~1.5KB（ReduceSum 最小临时空间，按 376 float 对齐） | ReduceSum 归约中间量 |
| outQueue（VECOUT，BUFFER_NUM=2） | 2 × 32B | 每 tile 的部分和标量 |

**tile 粒度**：连续路径（incx=1）每 tile 处理 20480 个 float（= 10240 个复数元素）；步长路径（incx>1）每 tile 处理 10240 个复数元素（产生 20480 个 float 槽位，含成对实虚）。32B 块对齐（8 float）由 DataCopyPad / 标量补零保证。

**Init 阶段**：

```
Init(inGM, outGM, tiling):
  blockIdx = GetBlockIdx()
  解析 tiling → n / incx / 派生本核 offset(复数粒度) 与 calNum
  inGM 以 float 视图绑定: 逻辑长度 = incx==1 ? 2*n : 2*(1+(n-1)*incx)（仅用于越界保护）
  outGM 绑定 1 个 float
  InitBuffer(inQueue / workBuf / outQueue)
  清零输出: workLocal Duplicate(0.0) → DataCopyPad(outGM, 4B)
  SyncAll()          // 保证清零先于所有核的原子加（各核写同值 0，良性竞争）
```

**Process 阶段**（每核遍历自己的 [offset, offset+calNum) 复数元素区间）：

```
Process():
  SetAtomicAdd<float>()
  将本核 calNum 个复数元素按 tile 迭代:
     repeat = calNum / tileComplex(10240)，remain = calNum % tileComplex
     ├─ 完整 tile:  SingleIteration(offsetCur, 10240)
     └─ 尾块:       SingleIterationAligned(offsetCur, remain)   // 补零至 8 float 对齐
  SetAtomicNone()
```

**SingleIteration（CopyIn → Compute → CopyOut）**：

```
CopyIn(offset, count):   // count 为本 tile 复数元素数
  if incx == 1:          // 连续快路径：float 视图整体搬运
      DataCopy(inLocal, inGM[2*offset], 2*count)        // 高层 API 自动按 32B burst 拆分
  else:                  // 步长路径：逐元素标量收集，实虚成对写入相邻 float 槽位
      for i in [0, count):
          base = 2 * (offset + i) * incx                 // uint32 乘法，host 侧守卫保证不回绕
          inLocal[2i]   = inGM.GetValue(base)            // Re
          inLocal[2i+1] = inGM.GetValue(base + 1)        // Im

CopyInPad(offset, count):  // 尾块版本
  if incx == 1: DataCopyPad 搬 2*count 个 float，右侧补零至 8 float 对齐
  else:         标量收集后，剩余槽位 [2*count, 2*countAligned) 补 0.0（不影响求和值）

Compute(count):            // countAligned 为对齐后 float 数
  Abs(inLocal, inLocal, 2*count或countAligned)   // float 视图逐元素绝对值，实虚统一处理
  pipe_barrier(PIPE_V)
  ReduceSum(outLocal, inLocal, workLocal, countAligned)  // tile 内树状归约 → 1 个 float 部分和

CopyOut():
  DataCopyPad(outGM, outLocal, 4B)   // 经 SetAtomicAdd 累加到 GM 标量（与清零/其他核原子加全局有序）
```

**kernel 流程图**：

```
                 ┌─────────────── 每核（blockIdx = 0..useCoreNum-1）───────────────┐
                 │ Init: 解析 tiling → 派生 (offset, calNum)；清零 outGM；SyncAll    │
                 └────────────────────────────┬───────────────────────────────────┘
                                              ▼
        ┌──────── Process: SetAtomicAdd<float> ────────────────────────────────┐
        │  ┌── tile 循环（10240 复数/tile，double buffer ping-pong）──────────┐ │
        │  │  CopyIn   ── incx=1: DataCopy(float 视图连续 2*count)            │ │
        │  │           └─ incx>1: GetValue 成对标量收集(Re/Im 相邻槽位)        │ │
        │  │  Compute  ── Abs(float 视图) → ReduceSum → tile 部分和标量        │ │
        │  │  CopyOut  ── 4B 原子加(DataCopyPad 至 outGM)                      │ │
        │  └───────────────────────────────────────────────────────────────────┘ │
        │  尾块：DataCopyPad / 标量补零至 8 float 对齐后同上                       │
        └──────── SetAtomicNone ─────────────────────────────────────────────────┘
```

**精度路径分析**：tile 内 ReduceSum 为向量树状归约；跨 tile、跨核经 float 原子加按不确定顺序累加。n=4M（8M float）时全算子约数百次原子加 + 每 tile 内部树状归约，相对舍入误差远小于 rtol=2⁻¹³ 阈值（golden 为双精度 CPU 参考）；任务书明确不要求归约顺序逐位一致（bit-exact），§3.2 容差按标量比对判定。若实测大 n 用例逼近阈值，切换备选方案（见 §3.2.4）。

**Inf/NaN 行为**：Abs 与 ReduceSum 对 Inf/NaN 依 IEEE 语义传播（|±Inf|=+Inf，|NaN|=NaN，含 NaN 的和为 NaN），与 golden（CPU fabs 双精度）一致，按 §3.2 等效性规则判定；全零/正负交替等填充对绝对值求和均为平凡路径。

#### 关键设计决策与备选方案

| 决策 | 主方案 | 备选方案（触发条件） |
| --- | --- | --- |
| 跨核归约 | GM 标量 SetAtomicAdd（sasum arch22 已验证；无 workspace、无终归约 barrier，小 n 时延最优） | workspace 存每核部分和 → SyncAll → 0 号核 DataCopyPad 取回 + ReduceSum 终归约出结果（snrm2/cdot arch22 范式）。触发条件：原子加浮点顺序不确定导致大 n 精度逼近 §3.2 阈值，或多核原子争用实测影响性能 |
| tiling 传参 | 结构体按值入参（arch35 scasum 范式；异步安全、零额外拷贝） | GM 指针 + aclrtMalloc/Memcpy（sasum arch22 范式）。触发条件：arch22 工具链对结构体按值入参支持有问题（编译验证阶段确认） |
| incx>1 数据通路 | 标量收集（正确性路径；性能用例全部 incx=1，不影响验收性能） | GatherMask 按 stride 解交错（可提升大步长用例吞吐）。触发条件：仅当步长用例出现性能或正确性问题时评估 |
| 分核策略 | 满核 GetAivCoreCount()，核间 perCoreN/remainder 均衡（大块优先） | 小 n（n < 核数）时自然退化为 useCoreNum = n 核各处理 1 个元素 |

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品（Atlas 800T A2，Ascend 910B3） | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 | √ |
| Atlas A3 推理系列产品 | √ |

- 映射关系：`ascend910b*` / `ascend910_93` → arch22（build.sh SOC_ARCH_DIRS 映射）；
- 自验覆盖：精度在 910B3（或 A3）验证，性能在 Atlas 800T A2 (910B3) 验证（任务书 §3.1：自验证只需覆盖一种款型）。

### 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 数据类型 | 仅 complex64 输入、float32 标量输出，不支持其他 dtype |
| 维度/步长 | x 逻辑一维 [n]；incx ≤ 0 一律 quick return（不遍历负步长），非连续仅由 incx 表达，无 leading dimension padding |
| 跨度上限 | (n-1)\*incx 须在 uint32 索引域内，超出返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| broadcast / dynamic shape / 原地 | 均不涉及（单向量归约到独立标量） |
| 确定性 | 归约累加顺序不保证逐位一致，精度按容差阈值判定 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；调用返回 SUCCESS 不代表结果就绪，读回前须同步 stream |
| 元素值 | 不校验（Inf/NaN/极端值按 IEEE 语义传播，溢出至 Inf 与 golden 一致） |

### 工程文件清单

| 文件 | 新增/修改 | 内容 |
| --- | --- | --- |
| `blas/asum/arch22/scasum_host.cpp` | 新增 | 校验 / quick return / 跨度守卫 / tiling / kernel 下发 |
| `blas/asum/arch22/scasum_kernel.cpp` | 新增 | ScasumAIV kernel（float 视图 + Abs + ReduceSum + 原子加） |
| `blas/asum/arch22/scasum_tiling_data.h` | 新增 | 紧凑 tiling 结构（参考 arch35） |
| `test/asum/scasum/arch22/scasum_test.cpp` | 新增 | CSV 驱动 GTest（精度/边界/负向/性能路径） |
| `test/asum/scasum/arch22/scasum_test.csv` | 新增 | 随任务提供的 1200 条用例 + 正态分布补充用例 |
| `test/asum/scasum/CMakeLists.txt` | 修改 | arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)` |
| `blas/asum/README.md` | 修改 | 产品支持表 aclblasScasum 增加 Atlas A2/A3：支持 |
| `include/cann_ops_blas.h` | 不改 | 接口声明已存在（与 950PR 共用） |
| `test/asum/scasum/scasum_param.h` / `scasum_golden.h` | 复用 | arch 无关，已存在 |

文件数量与仓内同族算子一致（host/kernel/tiling 三件套 + 测试），无冗余测试与编译文件，符合 ops-blas 贡献指南与任务书 §5 反例要求。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 FLOAT32 标量按标量比对：atol = rtol = 2⁻¹³（≈1.2207e-4），通过条件 `\|actual−golden\| ≤ atol + rtol×\|golden\|`；max_abs_error ≤ 1e-2 或 32×ULP；matched_ratio 单值退化为 1.0；Inf/NaN 按等效性规则（同 NaN / 同号 Inf / NaN↔±Inf 通过，有限值↔Inf/NaN 不通过）；golden = CPU 双精度参考（仓内 `scasum_golden.h`），quick return golden 固定 0.0f；溢出用例按 Inf 一致性判定并在自测报告单独标注 | 生态算子开源精度标准（opbase mixed_tolerance_standard），任务书 §3.2 |
| 性能标准 | Atlas 800T A2 (910B3)、FLOAT32 输入：平均单次耗时 ≤ A100 标杆 × 1.25（即倍率 ≥ 0.8）。口径：msprof op 采集单算子 task duration（us），自带 5 次 warmup，连续 ≥10 次有效采样取算术平均；GTest 侧以 aclrtEvent/chrono 统计 warmup 后连续下发均值（打印 `[PERF] ... avg_us=`）作辅助交叉验证 | 任务书 §3.3，test_cases/gpu_baseline.csv（A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4 实测回填） |
| 内存标准 | 任务书 §3.4 不涉及专项要求；按交付件模板采集内存占用数据（输入 n×8B + 输出 4B + tiling/workspace 固定小开销） | 任务书 §3.4、§4 交付件 4.1/4.2 |

**性能预算分析**（典型 case，A100 基线来自 gpu_baseline.csv）：

| n（复数） | 输入数据量 | A100 标杆 | NPU 判据（≤1.25×） | 需有效读带宽 |
| --- | --- | --- | --- | --- |
| 1048576 (2²⁰) | 8MB | 19.87 us | ≤ 24.83 us | ≥ 0.32 TB/s |
| 2097152 (2²¹) | 16MB | 22.15 us | ≤ 27.69 us | ≥ 0.58 TB/s |
| 4194304 (2²²) | 32MB | 33.56 us | ≤ 41.94 us | ≥ 0.76 TB/s |

本算子为纯带宽受限归约：设计上通过满核下发 + 连续 32B 对齐大块 DataCopy + double buffer 重叠逼近 HBM 峰值带宽；A100 基线隐含有效带宽约 0.9~1.0 TB/s，910B3 满核 AIV 连续读在该数据量级可满足 ≥0.76 TB/s 判据。小 n（≤1K）为时延敏感区，kernel 固定开销（tiling 解析、清零+SyncAll、单 tile 流水）决定水平，按值 tiling 传参与免同步下发将 host 开销排除在 kernel 时延之外。

## 测试方案

### 测试工程与执行链路

- 目录：`test/asum/scasum/arch22/`（`scasum_test.cpp` + `scasum_test.csv`），GTest 结构镜像仓内 arch35 版本：`NullHandle` 独立用例 + CSV 参数化套件（`GetCasesFromCsv<ScasumParam>`）按 `expect_result` / quick return / 正常路径三分派，复用 `scasum_param.h`（列：case_name, description, n, incx, x 填充, result 空指针标记, expect_result, mere_threshold, mare_multiplier, random_seed）与 `scasum_golden.h`。
- CMake 注册（随本任务补齐 arch22 分支）：

```cmake
if("arch35" IN_LIST SOC_ARCH_DIRS OR "arch22" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
```

- 构建执行：`bash build.sh --soc=ascend910b3 --ops=scasum`（A2；A3 用 `--soc=ascend910_93`）；
- 验收脚本：`verify_accuracy.py`（编译 → 安装 CSV → 运行 GTest → 解析逐条 PASS/FAIL，自动排除 TC_PF）、`verify_performance.py`（运行 TC_PF → 解析 `[PERF] avg_us` → 与 gpu_baseline.csv 比对，ratio ≥ 0.8 判 PASS）。

### 用例设计（1200 条 CSV，固定种子可复现）

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 4 | 小尺寸（n=1/8）× incx=1/2 |
| L1 尺寸扫描 | TC_SQ | 38 | n ∈ {0,1,2,3,4,5,7,8,15,16,…,2²⁰}：质数、2 的幂及 ±1、非对齐值 |
| L2 步长 | TC_INC | 9 | incx ∈ {1,2,3} × n ∈ {16,64,256}（正常计算路径） |
| L5 填充 | TC_FL | 12 | 均匀随机 / 全零 / 正负交替 / 极端值 / Inf / NaN × 2 尺寸 |
| 溢出 | TC_OV | 3 | 大 n × RANDOM_EXTREME（近 FLT_MAX），结果可至 Inf，按 Inf 一致性判定 |
| L6 边界/负向 | TC_ED | 13 | n=0 / 负 n quick return（SUCCESS，result=0.0f）；incx=0/-1/-2/-3、INT32_MIN quick return；x 空指针、result 空指针（INVALID_VALUE）；quick return 路径空指针组合（n0_null_x 仍 SUCCESS、n0/incx0/neg_n_null_result 仍 INVALID_VALUE）；跨度溢出（n=4, incx=INT32_MAX → INVALID_VALUE） |
| EX 扩展 | TC_EX | 921 | 尺寸 × 步长 × 填充确定性采样（约 1/8 采样 quick return 步长） |
| PF 性能 | TC_PF | 200 | incx=1 纯连续：3 条任务书典型 case（n=2²⁰/2²¹/2²²）+ 小尺寸时延区 + 规模扫描 + 特殊填充 + 混合 |

handle 空指针（`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）由 GTest 工程层面 `NullHandle` 用例覆盖，不在 CSV 表达。

**负向用例缓冲区规则**：期望非 SUCCESS 的用例不按 (n-1)×|incx| 预分配输入（INT32_MIN 步长乘积在 int 域溢出），以单元素 dummy 缓冲或 NULLPTR 占位，接口在读数据前返回预期错误码即判通过（用例 `incx_intmin_quick_return`、`incx_span_overflow` 已按此生成）。

**正态分布补充（任务书 §3.5-3 / gen_csv.py 说明）**：随任务生成器默认仅均匀分布 `RANDOM_NORM_5_5`（[-5,5]）；正态分布（μ∈[-5,5]，σ∈[0.1,2]，token `GAUSS_NORM_5_2`）需自行补齐——扩展测试框架 `csv_loader/makeBlasArray` 填充模式支持正态采样（实部/虚部独立），并在 CSV 中补充正态用例纳入精度自验证；自测报告说明补齐方式、用例数量与覆盖情况。

### 采集与判定方法

- **精度**：`python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv ./scasum_test.csv`；标量比对按 mixed tolerance（FLOAT32），Inf/NaN 用例按等效规则，全部 PASS 方通过。
- **性能**：`python verify_performance.py --repo <ops-blas> --soc ascend910b3`（GTest `[PERF] avg_us` 与基线 ratio ≥ 0.8）；精确 kernel 口径另以 `msprof op --application=".../scasum_test --gtest_filter=*TC_PF*" --output=./prof_scasum` 采集 `OpBasicInfo.csv` 的 Task Duration(us) 统计均值交叉验证，与任务书 §3.3 主口径一致。
- **内存**：随 msprof 采集输出内存占用，回填自测报告 4.1/4.2。
- **自测报告**：按模板输出用例参数、标量比对结果（含截图）、性能数据（含截图）、内存数据；溢出至 Inf 的用例单独标注；异常样本（非 SUCCESS）不计入性能样本并说明。

## 兼容性分析

- **新路径扩展，零既有行为变更**：`blas/asum/arch22/` 仅新增文件，不触碰 arch22 sasum 与 arch35 scasum 的任何既有代码路径；源码按目录自动发现，无中央构建脚本改动。
- **接口共用**：`include/cann_ops_blas.h` 声明已存在（与 950PR 共用），本任务不新增/修改任何对外 API，禁止私有平行接口（任务书 §7-3）。
- **测试框架**：`scasum_param.h / scasum_golden.h` 为 arch 无关共享文件，arch22 测试直接复用；CMake 仅在 scasum 测试目录增加 arch22 GTest 分支，不影响其他算子构建（同构家族 sasum/isamax 等目录不受影响）。
- **文档**：`blas/asum/README.md` 产品支持表将 aclblasScasum 的 A2/A3 由"不支持"更新为"支持"，属事实性更新。

---

# 交付与验收对照

| 交付件 | 本设计的落实 |
| --- | --- |
| 1. 设计文档 | 本文档，按官方模板提交 cann-competitions 仓 PR 评审 |
| 2. 自测用例及测试代码 | §测试方案：1200 条 CSV（+正态补充）+ `test/asum/scasum/arch22/` GTest 工程 + README 测试步骤说明 |
| 3. 自测报告 | 精度（标量比对+截图）/ 性能（msprof 与 GTest 双口径+截图）/ 内存数据，溢出用例单独标注 |
| 4. 待验收代码 | 个人仓 fork（邀请 Ascend-CANN 账号），实现 `blas/asum/arch22/`、测试 `test/asum/scasum/arch22/`、README 支持表更新；task_submission 目录按模板同步自验证材料 |

---

## 参考资料

1. 任务书：[aclblasScasum_AtlasA2_A3_task_doc.md](./aclblasScasum_AtlasA2_A3_task_doc.md) 与 [test_cases/README.md](./test_cases/README.md)
2. Netlib BLAS scasum.f 参考实现：https://www.netlib.org/blas/scasum.f
3. cuBLAS cublasScasum 文档：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum
4. ops-blas 开源仓：https://gitcode.com/cann/ops-blas （arch22 sasum/snrm2/cdot、arch35 scasum 参考实现）
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
6. Ascend C 算子开发文档 / 接口文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
