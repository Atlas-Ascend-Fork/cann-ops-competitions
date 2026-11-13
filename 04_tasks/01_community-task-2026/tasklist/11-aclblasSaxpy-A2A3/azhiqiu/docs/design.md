# aclblasSaxpy 算子设计文档（Atlas A2/A3）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务 — aclblasSaxpy 算子开发（A2A3）。
- 合入目标仓：https://gitcode.com/cann/ops-blas ，算子代码 `blas/axpy/arch22/`，测试代码 `test/saxpy/arch22/`。
- 交付分支（私仓）：https://gitcode.com/azhiqiu/ops-blas （master）。
- 代码 PR（待提交）：向 `cann/ops-blas` master 提交实现与测试 PR。
- 验收流程：cann-ops-competitions 社区任务流程（任务书 §4 交付件 / §5 PR 申请合入）。

## 背景介绍

### 算子功能定位

aclblasSaxpy 完成单精度实数向量组合 `y ← α·x + y`（逐元素累加，y 原地覆写），是 BLAS Level-1 最基础的向量组合算子。

数学公式：`y[j] = α · x[k] + y[j]`，其中 `k = (i-1)·incx`、`j = (i-1)·incy`（实现内转 0 基下标），`i = 1..n`；`n ≤ 0` 时 y 不修改（合法 no-op）。

该算子补齐 Blas 仓 Atlas A2/A3 系列产品的 BLAS-1 能力缺口；同一接口在 950PR（arch35）已有实现，本任务为 arch22 构建路径补齐。

### 现状分析

- 接口声明已存在于 `include/cann_ops_blas.h`（与 950PR 共用），**不另立私有平行接口**。
- 仓内 `blas/axpy/arch35/` 已有 950PR 实现，可作为语义与返回码的对齐基线；其非连续路径依赖 SIMT（`asc_vf_call` + `__simt_vf__`），经全仓检索确认 **arch22 零使用 SIMT**，A2 上不可用。
- 仓内 `blas/scal`、`asum`、`copy`、`rot` 等 BLAS-1 兄弟算子提供了 arch22 工程骨架参考。

### 业界对标

- 对标 cuBLAS `cublasSaxpy`（CUBLAS_Library.pdf §2.5.4）：参数序列、返回码语义、快速返回口径以该文档为唯一依据。
- 语义参考 Netlib BLAS `saxpy.f`：`IF (N.LE.0) RETURN`（n≤0 合法 no-op）、`IF (DA.EQ.ZERO) RETURN`（alpha=0.0 直接返回，y 不变且不读取 x）。
- 行为对齐仓内 950PR 既有实现 `blas/axpy/arch35/saxpy_host.cpp` 的 `ValidateSaxpyParams` 与 `aclblasSaxpy` 入口。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 `blas/axpy/arch22/` 实现 `aclblasSaxpy` 算子，支持：

1. float32 单精度（算子唯一 dtype 面）；
2. 任意合法非零 `incx`/`incy`（含负步长），不支持 `INT32_MIN`；
3. `n ≤ 0` 合法 no-op；`alpha == 0` 合法 no-op（不读取 x）；
4. `y` 原地覆写、`x` 只读，不产生中间张量、无额外 workspace；
5. 异步执行：经 handle 绑定的 stream 下发，不在 Host 侧插入同步。

## 需求拆解

1. Host 侧参数校验与快速返回，返回码对齐 cuBLAS 与仓内 arch35 口径；
2. Block 级 tiling 分核，核间负载基本均衡且满足 32 B 对齐；
3. UB 级 tiling：连续路径双缓冲流水，非连续路径 span 读 + 内重排；
4. 负步长物理下标映射，保证跨核物理区间不重叠；
5. 测试工程：CSV 驱动 GTest + 性能用例，覆盖任务书 §3.5 强制步长集合。

## 接口与约束

```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx, float* y, int incy);
```

| 参数 | 类型 | 内存位置 | 描述 | 约束与异常行为 |
|---|---|---|---|---|
| handle | `aclblasHandle_t` | Host | 库上下文，携带 stream | `nullptr` → `HANDLE_IS_NULLPTR` |
| n | `int` | Host | 元素个数 | `n <= 0` 为合法 no-op，返回 SUCCESS，不访问其他参数 |
| alpha | `const float*` | Host | 标量乘数 α | `n > 0` 且 `nullptr` → `INVALID_VALUE`；数值不校验（含 0/Inf/NaN） |
| x | `float*`（只读） | Device | 输入向量 | `n > 0` 且 `nullptr` → `INVALID_VALUE`；物理长度 `1+(n-1)*|incx|` |
| incx | `int` | Host | x 步长 | `0`/`INT32_MIN` → `INVALID_VALUE`；`(n-1)*|incx|` 超 int32 域 → `INVALID_VALUE`（int64 计算） |
| y | `float*` | Device | 输出向量（原地覆写） | 同 x 的空指针规则；物理长度 `1+(n-1)*|incy|` |
| incy | `int` | Host | y 步长 | 同 incx 规则 |

**Host 校验顺序**（第 1~4 步对齐 arch35 `ValidateSaxpyParams`，第 5~6 步为 arch22 新增，第 7 步为 arch35 所无的早退）：

```
1. handle == nullptr                      → HANDLE_IS_NULLPTR
2. n <= 0                                 → SUCCESS（立即返回，后续一律不访问）
3. alpha / x / y == nullptr               → INVALID_VALUE
4. incx == 0 || incy == 0                 → INVALID_VALUE
5. incx == INT32_MIN || incy == INT32_MIN → INVALID_VALUE
6. (n-1)*|incx| 或 (n-1)*|incy| > INT32_MAX → INVALID_VALUE（int64 计算）
7. *alpha == 0.0f                         → SUCCESS（Netlib IF (DA.EQ.ZERO) RETURN）
8. 下发 kernel                            → SUCCESS
```

第 7 步置于第 3~6 步之后，保证 `alpha == 0` 且指针非空时仍能捕获非法步长，符合任务书 §2.5「n>0 时 alpha、x、y 为 nullptr 或 incx==0… 返回 INVALID_VALUE」的强制口径。

接口保持异步语义：成功返回仅表示任务已提交到 handle stream，不表示 Kernel 已完成。

# 详细设计（required）

## 算子分析

### 数学公式

```
y[j] = α · x[k] + y[j]
k = (i-1) · incx,  j = (i-1) · incy,  i = 1..n
```

逐元素执行，元素间无数据依赖，天然可多核并行；`y` 原地覆写，`x` 只读。

### 支持数据类型

- [x] float32（算子唯一 dtype 面）
- [ ] float16 / bfloat16（不支持，不做升精度路径）

### 支持形状

一维向量（BLAS-1），无广播、无 0 维张量语义；`n`/`incx`/`incy` 均为运行时入参（dynamic shape）。

### 负步长语义

物理下标按 `|inc|` 递增，逻辑第 i 个元素取物理 `(n-1-i)*|inc|`，即从物理 span 尾端起算（Netlib 语义，与 arch35 issue #483 修正一致）。

## 算子实现

### 实现方案

#### Host 侧设计

**分核策略**（Block 级 tiling）：

```
ALIGN_ELEMS   = 32 / sizeof(float) = 8
coreNum       = GetAivCoreCount()                 // 910B3: 40
perCoreCount  = CeilAlign(CeilDiv(n, coreNum), ALIGN_ELEMS)
usedCoreNum   = CeilDiv(n, perCoreCount)          // <= coreNum
blockIdx      = GetBlockIdx()
myOffset      = blockIdx * perCoreCount
myCount       = min(perCoreCount, n - myOffset)
```

对每个核的份额**先向上取整再 8 元素对齐**，把余数摊入最后一核自身份额。以 n=1048576、coreNum=40 为例：`perCoreCount = CeilAlign(CeilDiv(1048576,40)=26215, 8) = 26216`、`usedCoreNum = 40`、尾核 `= 1048576 - 26216×39 = 26152`，不均衡度 **1.002×**；验证式 `26216 × 39 + 26152 = 1048576` ✓。

对比 arch35 的向下取整 + remainder 压尾核方案（按 `blas/axpy/arch35/saxpy_host.cpp:57-59` 复算：`rawPerCore=26214`、`perCoreN=26208`、`remainder=256`、尾核 `26464`，不均衡 **1.01×**），本设计避免了 `remainder` 随 `n/coreNum` 取值扩大导致的长尾，同时保证每核起点 32 B 对齐。

**TilingData 结构**（host 与 device 共享同一头文件）：

```cpp
struct SaxpyTilingData {
    uint32_t totalN;        // 逻辑元素个数 n
    uint32_t usedCoreNum;   // 实际使用 AIV 核数
    uint32_t perCoreCount;  // 整核处理的逻辑元素数（8 元素 / 32B 对齐）
    uint32_t tileCount;     // 核内单次处理元素数
    int64_t  incx;
    int64_t  incy;
    float    alpha;
};
```

传参方式采用 **struct by value**（`const SaxpyTilingData tiling`），与仓内 arch22 主流写法一致，免去 pinned 上传 slot 与事件回查，host 侧无额外同步。

**Workspace**：不需要额外 workspace——输入/输出均为调用方 Device 指针并原地覆写；tiling 经 by-value 传参；kernel 入参保留 `GM_ADDR workSpace` 仅为与仓内 `*_kernel_do` 签名一致，恒传 `nullptr`。

#### Kernel 侧设计

**连续路径（incx == 1 && incy == 1）**：

```
Init:
    pipe->InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(float))
    pipe->InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(float))
Process:
    for cur = myOffset; cur < myOffset + myCount; cur += TILE:
        len = min(TILE, myOffset + myCount - cur)
        xLocal <- DataCopy(xGM[cur], len)      // MTE2
        yLocal <- DataCopy(yGM[cur], len)      // MTE2
        Axpy(yLocal, xLocal, alpha, len)       // V: y = y + alpha*x
        DataCopy(yGM[cur], yLocal, len)        // MTE3
        SetFlag/WaitFlag(MTE3_MTE2)            // 写回完成后方可复用 UB
```

- 双缓冲（`BUFFER_NUM = 2`）流水：MTE2 与 V 重叠，隐藏 GM 延迟；
- **MTE3→MTE2 事件同步**：写回完成后显式置位，防止 UB 缓冲复用竞态（该竞态曾在自验证中出现 14 条非确定性失败，修复后消除）；
- 尾块不满足 8 元素对齐时改用 `DataCopyPad`（`DataCopyExtParams{1, len*4, 0,0,0}`）读写。

**UB 分配表（连续路径，float32）**：

| Buffer | 单元大小 | 数量 | 合计 |
|---|---|---|---|
| inQueueX | TILE × 4 B | 2 | 2 × TILE × 4 |
| inQueueY | TILE × 4 B | 2 | 2 × TILE × 4 |
| **总计** | — | — | **16 × TILE B** |

910B3 单 AIV UB = 192 KiB，留 64 KiB 余量后可用 128 KiB：`maxTileElements = 131072/16 = 8192`，32B 对齐后 **TILE = 8192**，实际占用 128 KiB（66.7%）。

**非连续路径（strided）**：A2 无 SIMT，三条路径对比后选择「span 读 + UB 内重排」：

| 方案 | 正确性 | \|inc\|≤3 性能 | 代码量 | 结论 |
|---|---|---|---|---|
| 逐元素 GM 搬运 | 稳 | 极差（1 次 MTE2/元素） | 小 | 备选 |
| **span 读 + UB 内重排** | 稳 | 好（1/\|inc\| 有效带宽） | 中 | **采用** |
| GatherMask / index 表 | 稳 | 最好 | 大 | 不采用 |

实现：把 `(len-1)*|inc|+1` 个连续元素整段读到 UB（`DataCopyPad`，GM 地址无需 32 B 对齐），在 UB 内按步长做标量 gather/scatter，再整段写回 y。未选中元素值不变、原样写回，正确性无损。UB 分配：xSpanBuf 48 KiB + ySpanBuf 48 KiB = 96 KiB（50%）。

`tileCount = min(8192, (MAX_SPAN-1)/max(|incx|,|incy|) + 1)`，下限为 1（`|inc| > MAX_SPAN` 时退化为逐元素），保证任意合法步长均可执行。

**物理下标映射**：

```
absInc = |inc|
pos(i) = (inc > 0) ? i * absInc : (n - 1 - i) * absInc
```

负步长时各核的物理区间互不重叠（逻辑→物理为单调映射），故 span 整段写回不会破坏其他核的数据。

### 性能优化方案

1. **Block 级均衡切分**：`CeilAlign(CeilDiv(n, core))`，尾核不均衡度 1.002×（§分核策略）；
2. **双缓冲流水**：`BUFFER_NUM=2` 的 TQue，MTE2 与 V 流水重叠；
3. **整块向量化**：主循环全 tile 批量 `DataCopy` + 向量算，仅尾块走 `DataCopyPad`；
4. **原地 UB 复用**：y 读出后直接作为输出 tensor，省出 2×TILE×4 B UB；
5. **alpha 走 Host 标量**：避免额外 GM 读。

算子特性：memory-bound（每元素 1 次乘加，读 x/y 各 4 B、写 y 4 B）；并行性高（元素级无依赖）。

### 约束核对（任务书 §2.5）

| §2.5 约束项 | 本设计落实 | 结论 |
|---|---|---|
| 参数合法性与校验层级 | 八步校验序列；返回码与 cuBLAS 一致，较 arch35 增加 INT32_MIN/span 溢出校验与 alpha==0 早退 | ✅ |
| 非连续 Tensor 支持 | span 读 + UB 重排，正负步长均支持；incx/incy==0 返回 INVALID_VALUE | ✅ |
| broadcast 规则 | 无广播：参与运算元素数同为 n | 不涉及 |
| dynamic shape 要求 | n/incx/incy 运行时入参，host 按运行值算 tiling | ✅ |
| 原地与视图语义 | y 原地覆写，x 只读，不返回视图 | ✅ |
| 确定性计算要求 | 逐元素单次执行确定；同输入同步长结果 bit 级一致（无原子累加、无跨核归约） | ✅ |
| 空 Tensor 处理 | n=0 快速返回，不下发 kernel | ✅ |
| 0 维 Tensor 处理 | 不涉及；标量语义由 n=0 快速返回与 Host 标量入参承担 | 不涉及 |
| 异步执行 | 经 handle stream 异步下发，不插入任何同步；alpha 在 Host 侧同步解引用 | ✅ |

**特别核对：异步性**。arch22 兄弟算子 `sscal` 的实现中每 tile 调用 `aclrtMalloc/Free` 并有 `aclrtSynchronizeStream`，不满足该条；本设计参考 `caxpy`，全程无 host 同步、无动态分配。

## 测试与验收

### 测试工程

- 位置：`test/saxpy/arch22/`（`saxpy_test.cpp`、`saxpy_npu_wrapper.h`、`saxpy_test.csv`），并在 `test/saxpy/CMakeLists.txt` 注册 arch22 的 `ops_blas_add_gtest_tests`；
- 数据驱动：CSV 驱动 GTest，golden 由仓内 `test/saxpy/saxpy_golden.h`（cblas_saxpy float32 参考实现）生成，precision 基准为 Netlib BLAS；
- 用例构成：**共 1210 条** = 1006 精度 + 3 手工负向（随代码）+ 200 性能 + 补充用例；
- **强制步长覆盖**：任务书 §3.5 要求 `incx` 覆盖 `{1, 2, 7, -1, -3}`，随任务包提供的 CSV 原始版本仅含 `{1,2,3,-1,-2,-3}`，本设计补充 `incx=7` / `incy=7` / `±7` 混合步长用例（`TC_INC_064~069`）及 `alpha=0.1/10.0`、`n=2^30 & incx=8` span 溢出负向（`TC_INC_070~072`、`TC_ED_122`）；
- **负向用例缓冲区规则**：`expectResult != SUCCESS`、`n <= 0`、`|inc|==INT32_MIN`、span 溢出用例一律用单元素 dummy 缓冲（1 float），不按 `(n-1)*|inc|` 预分配，避免测试工程自身分配数十 GB 而崩溃误判；
- 空指针用例（`handle == nullptr`、`alpha == nullptr`）由独立 TEST 覆盖（CSV 无法表达）。

### 验收口径

- 精度：`atol = rtol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 32 ULP；Inf/NaN 按任务书等效规则比对；
- 性能：`avg_us ≤ gpu_ms × 1000 / 0.8`（GPU 标杆耗时由 ms 换算为 us 后套用「NPU ≥ 0.8× A100」判据），warmup 50 次 + 有效采样 100 次；
- 内存：BLAS-1 向量算子，无额外 workspace 判据。

### 自验证结果（Atlas 800T A2 / 910B3 / CANN 9.1.0 aarch64）

| 项 | 结果 |
|---|---|
| 编译 | PASS（`bash build.sh --ops=saxpy`） |
| 精度 | **PASS=1010，FAIL=0** |
| 性能 | **PASS=200，FAIL=0**（相对 cublasSaxpy 提速 min 1.06×、median ≈1.9×） |
| 内存 | 达标（无额外 workspace） |
| 可复现性 | 三轮独立复跑结果一致 |

交付件（七件套）已同步至私仓 `task_submission/` 目录。

## 已知限制与规避

| # | 限制 | 影响 | 规避/说明 |
|---|---|---|---|
| 1 | `alpha == 0` 快速返回不读取 x | 若 x 含 Inf/NaN，IEEE 语义应得 NaN，本实现保留 y 原值 | 任务书 §2.1 强制对齐 Netlib `IF (DA.EQ.ZERO) RETURN`；精度用例集无「alpha=0 且 x 非有限」组合 |
| 2 | 非连续路径带宽放大 `max(|incx|,|incy|)` 倍 | 大 \|inc\| 时吞吐下降 | 性能子集 200 条全为连续访存（`gen_csv.py` 末尾有断言自检），不影响判据；后续可用 GatherMask 优化，记为技术债 |
| 3 | 未使用 SIMT（arch22 不支持） | 非连续路径无法像 arch35 那样线程级并行 | 由 span 读 + UB 内向量化补偿 |
| 4 | `x` 声明为非 `const` | 易误写 | 实现内按只读使用，code review 检查点 |

## 文件清单

新增（均在 arch22 路径生效）：

- `blas/axpy/arch22/saxpy_tiling_data.h` — host/device 共享 tiling 布局
- `blas/axpy/arch22/saxpy_kernel.h` / `saxpy_kernel.cpp` — kernel 入口（dense + strided）
- `blas/axpy/arch22/saxpy_host.cpp` — 校验 + tiling + 下发
- `test/saxpy/arch22/saxpy_test.cpp` / `saxpy_npu_wrapper.h` / `saxpy_test.csv` — CSV 驱动 GTest 与 NPU 调用封装
- `test/saxpy/CMakeLists.txt` — 补 arch22 的 `ops_blas_add_gtest_tests` 注册
- `blas/axpy/README.md` — 产品支持情况补 A2/A3，补约束说明

## 关联链接

- 社区任务：https://www.hiascend.com/activities/task-center/details/6ffdb3f781a044588935d6c7d33c3b96
- 交付私仓：https://gitcode.com/azhiqiu/ops-blas
- 设计文档 PR（ops-blas）：https://gitcode.com/cann/ops-blas/merge_requests/951
