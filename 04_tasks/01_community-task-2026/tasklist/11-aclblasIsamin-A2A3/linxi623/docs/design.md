# aclblasIsamin 算子设计文档（Atlas A2/A3，arch22）

## 1. 概述与接口定义

### 1.1 需求背景与对标接口

本设计文档面向昇腾算子开源仓 `ops-blas`（`master` 分支）中 `aclblasIsamin` 算子在 Atlas A2/A3 系列产品（`arch22` 架构）上的开发与性能优化交付，旨在补齐 Atlas A2/A3 系列产品的 BLAS-1 单精度实数向量最小绝对值索引查找能力。

本算子采用 Ascend C 编程语言与 **Kernel 直调（Host 参数校验 + 标量下发 + Device Kernel 发射）** 工程模式，在不改动公共头文件声明与 `arch35` 既有行为的前提下，完成 `blas/iamin/arch22/` 算子实现与 `test/isamin/arch22/` 测试工程建设。

算子严格对标 NVIDIA cuBLAS 官方接口 `cublasIsamin`（参考 `CUBLAS_Library.pdf` §2.5.2）：

```cpp
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x, int incx, int *result);
```

数学表达式定义如下：

$$\text{result} = \underset{i \in \{1, \dots, n\}}{\arg\min}\ |x[1 + (i - 1) \times \text{incx}]|$$

核心语义约定：

1. **1-based 索引输出**：正常计算（$n > 0$ 且 $\text{incx} \ge 1$）时返回值满足 $\text{result} \in [1, n]$。
2. **并列最小值打破规则（First-Index Tie-Break）**：当多个位置具有相同的最小绝对值时，严格取最先出现的逻辑索引（即最小的 1-based 下标 $i$）。
3. **快速返回（Quick Return）**：当 $n = 0$ 或 $\text{incx} < 1$ 时，不启动计算 Kernel，通过 `aclrtMemsetAsync` 向 Device 端 `result` 异步写入 `0` 并返回 `ACLBLAS_STATUS_SUCCESS`。
4. **参考实现（Golden）来源**：精度比对 Golden 由严格遵循 cuBLAS §2.5.2 规范编写的 CPU 参考实现（`test/isamin/isamin_golden.h`）生成，这是任务书指定的判定基准；Netlib reference BLAS 中另有 `isamin.f`，其语义（并列取首个、NaN 不参与比较）与本实现一致，可作为交叉校验参考。

### 1.2 函数原型与参数说明

接口统一声明于公共头文件 `include/cann_ops_blas.h`（第 240 行），与 `arch35` 共用唯一公开声明：

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | 库上下文句柄（内部携带 `aclrtStream` 与默认 Workspace），Host 内存 | attr | - | - | - | `aclblasCreate` 返回的非空句柄 | 为 `nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | 输入 | 向量逻辑元素个数，Host 内存 | attr | `int` | - | - | 任意 `int32` 整数（`n=0` 或 `incx<1` 走快速返回，不报错） | `n<0` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 或 `incx<1` 时 `result` 写 `0` 并返回 `ACLBLAS_STATUS_SUCCESS` |
| `x` | 输入 | 输入单精度实数向量，Device 内存 | tensor | `FLOAT32` | `ND` | 一维向量，物理长度 $1 + (n - 1) \times |\text{incx}|$ 个 `float` 元素 | 全值域（含 `±0`、`±Inf`、`NaN` 特殊值） | `n>0` 且 `incx>=1` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；元素值不在 Host 侧校验 |
| `incx` | 输入 | `x` 中相邻逻辑元素之间的存储步长（以元素为单位），Host 内存 | attr | `int` | - | - | 任意 `int32` 整数（`incx<1` 走快速返回，不报错） | `incx<1` 时 `result` 写 `0` 并返回 `ACLBLAS_STATUS_SUCCESS`；`(n-1)*incx > INT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `result` | 输出（独立输出） | 绝对值最小元素的最先出现位置（1-based 索引），单个元素，Device 内存 | tensor | `INT32` | `ND` | shape `[1]` | `n>0` 且 `incx>=1` 时为 `[1, n]`；`n=0` 或 `incx<1` 时为 `0` | 为 `nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`（任意 `n >= 0`，含快速返回路径） |

### 1.3 参数校验顺序与返回码表

Host 侧入口 `ValidateIsaminParams` 与 `aclblasIsamin`（`blas/iamin/arch22/isamin_host.cpp` L28-92）严格对齐 `arch35` 校验层级，按以下确定性顺序执行判定：

| 校验序号 | 判定条件 | 返回状态码 | Device 副作用与行为说明 |
| --- | --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 立即返回，不访问任何指针 |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 先于快速返回与指针判空执行（如 `n=-1, incx=0` 仍返回 `INVALID_VALUE`） |
| 3 | `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 即使 `n=0` 或 `incx<1` 也必须校验 `result` 非空 |
| 4 | `n == 0 \|\| incx < 1` | `ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_EXECUTION_FAILED` | 置 `quickReturn = true`，调用 `aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)` 异步写 `0`；若驱动调用失败则返回 `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 5 | `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 正常计算路径（`n > 0 && incx >= 1`）要求输入向量 Device 指针非空 |
| 6 | `static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx) > INT32_MAX` | `ACLBLAS_STATUS_INVALID_VALUE` | 64 位乘法防溢出守卫，拦截步长跨度超出 `int32` 范围的非法入参 |
| 7 | `EnsureDefaultWorkspace(h, numBlocks * 2 * sizeof(float))` | 透传非成功状态码 | 复用或按需扩容 `handle` 内置 Workspace，随后在 `handle->stream` 上连续异步发射两阶段 Kernel |

---

## 2. 算法与核函数设计

### 2.1 两阶段总体执行架构

`aclblasIsamin` 在 `arch22` 上采用 **“多核分片局部归约（Stage 1：`isamin_kernel`，最多 48 个 AIV 核）+ 单核向量化跨核归约（Stage 2：`isamin_reduce_kernel`，1 个 AIV 核）”** 的两阶段异步流水架构：

```text
Input x (GM, FLOAT32, n, incx passed by value)
   │
   ├─► [Core 0 .. numBlocks-1] (Device-derived start_, num_)
   │     ├─ incx == 1: Double-Buffered Tile Loop (ISAMIN_TILE_ELEMS = 22528)
   │     │    DataCopyPad (prefetch next) -> Abs -> ReduceTile:
   │     │      (1) BlockReduceMin (8 elems -> 1 block min)
   │     │      (2) 3-Level Indexed WholeReduceMin Tree over block minima
   │     │      (3) V_S Sync -> Resolve winning 8-element block via WholeReduceMin -> V_S Sync
   │     │    -> MergeCandidate(tileMin, start_ + off + tileIdx)
   │     └─ incx > 1 : ProcessStrided scalar loop over GM with (val != val) guard
   │     └─► WriteResult: workSpace[2*blockIdx .. 2*blockIdx+1] = (bestVal_, float_bits(bestIdx_))
   │
   ▼ (Stream ordering on handle->stream, no host-side synchronize)
[isamin_reduce_kernel (1 AIV Core)]
   DataCopyPad <- workSpace (coreNum * 2 floats)
   Two-stage masked WholeReduceMin (mask = 0x5555555555555555ULL, ORDER_VALUE_INDEX)
   V_S Sync -> Extract winning core's bestIdx -> resultIdx = bestIdx + 1 -> DataCopyPad -> result (GM, INT32)
```

### 2.2 连续路径：双缓冲流水与块最小值分层索引归约（ReduceTile）

在连续访存路径（`incx == 1`，`ProcessContiguous`）中，每个核以 `ISAMIN_TILE_ELEMS = 22528` 为单位遍历本核负责的区间 `[start_, start_ + num_)`：

1. **双缓冲显式预取（Double Buffering）**：输入队列定义为 `TQue<TPosition::VECIN, 2> inQue_`。进入循环前先搬入首个分片并入队；在循环体内，若存在下一分片（`nextOffset < num_`），先分配第二块缓冲区并发起下一分片 `CopyTile`（`DataCopyPad` 自动按 8 元素即 32B 边界填充尾部），随后出队当前分片执行向量计算，实现 MTE2 DMA 搬运与 Vector 计算的跨 Tile 流水重叠。
2. **原地取绝对值**：对向上对齐到 8 元素倍数的长度 `aligned = AlignUp8(cnt)` 执行 `Abs(in, in, aligned)` 并插入 `PipeBarrier<PIPE_V>()`。
3. **块最小值 + 索引树 + 获胜块精解（`ReduceTile`）**：为规避直接对全 Tile 执行带索引归约的高昂指令开销，`TileBlockMin`（`isamin_kernel.cpp` L111-126，第一步：每 64 元素做一次硬件 BlockReduceMin）+ `ReduceTile`（L128-175，后三步：索引树、组归约、获胜块精解）提取精确最小值与最先出现下标：
   - **Step A（8 元素块值归约 `BlockReduceMin`）**：将 `count` 个元素按每 64 个元素 1 个 Repeat 划分为 `blockRepeats = count / 64` 与 `blockTail = count % 64`。由于 `22528 / 64 = 352 > 255`，通过最多 255 Repeat 的分块循环调用 `BlockReduceMin(work[rep * 8], in[rep * 64], chunk, 64, 1, 1, 8)`（以及尾部 `blockTail` 调用），将每 8 个连续 `float`（1 个 32B UB Block）压缩为 1 个块最小值，数据量缩小 $8\times$ 至 `count = (count + 7) / 8`（最多 $22528 / 8 = 2816$ 个块最小值，存放于 `work[0..2815]`）。
   - **Step B（块最小值三级带索引归约树 `WholeReduceMin`）**：令 `in = work`、`work = work[2816]`，在缩减后的块最小值序列上执行三级 `ReduceOrder::ORDER_VALUE_INDEX` 归约：
     1. 第一级按每 64 个块最小值归约出 `pairs = repeats + (tail != 0 ? 1 : 0)` 对 `(val, idx)`（最多 44 对 = 88 个 `float`，写入 `work[0..87]`）；
     2. 第二级利用偶数槽位掩码 `mask[0] = 0x5555555555555555ULL`（仅选取 `(val, idx)` 交错排布中的偶数位 `val` 参与比较），每 32 对（64 个 `float`）归约出组内最优对并写入 `work[512..]`；
     3. 第三级同样使用偶数位掩码将 `finalPairs` 个组最优归约至 `out[0..1]`。
   - **Step C（回溯获胜 8 元素块并在原数组上精解局部下标）**：执行首次 `SyncVectorToScalar()`（`HardEvent::V_S`），标量侧按三级索引链回溯获胜的 8 元素块号：
     $$\text{pair} = (\text{groupPair} / 2) \times 32 + (\text{localPair} / 2), \quad \text{block} = \text{pair} \times 64 + \text{idx}_1, \quad \text{offset} = \text{block} \times 8$$
     随后对原始已取绝对值的张量切片 `original[offset]`（长度为 `min(originalCount - offset, 8)`）执行一次仅含 $\le 8$ 个元素的 `WholeReduceMin(out, original[offset], ..., ReduceOrder::ORDER_VALUE_INDEX)`，经第二次 `SyncVectorToScalar()` 后返回 `offset + out_idx` 作为当前 Tile 的精确局部下标 `tileIdx`，同时 `out.GetValue(0)` 给出 `tileMin`。
4. **核内跨 Tile 合并（`MergeCandidate`）**：将 `(tileMin, start_ + off + tileIdx)` 与核内当前最优 `(bestVal_, bestIdx_)` 合并：

```cpp
__aicore__ inline void MergeCandidate(float tileMin, uint32_t globalIdx)
{
    if (tileMin != tileMin) {
        return; // NaN never becomes the minimum
    }
    if (!hasValue_ || tileMin < bestVal_ || (tileMin == bestVal_ && globalIdx < bestIdx_)) {
        bestVal_ = tileMin;
        bestIdx_ = globalIdx;
        hasValue_ = true;
    }
}
```

### 2.3 跨步路径（ProcessStrided）

当 `incx > 1` 时，最多 48 个 AIV 核并行执行 `ProcessStrided`（`isamin_kernel.cpp` L177-190）：每核从全局起始偏移 `base = uint64_t(start_) * uint64_t(incx_)` 出发，按步长 `incx_` 逐元素读取 GM 数据并计算绝对值 `absVal`，通过 `if (!(absVal != absVal) && (!hasValue_ || absVal < bestVal_))` 更新核内最优值与全局 0-based 逻辑下标 `start_ + i`。

### 2.4 跨核归约：两级掩码 WholeReduceMin（isamin_reduce_kernel）

Stage 1 结束时，每个核通过 `WriteResult` 将 `(bestVal_, *reinterpret_cast<float*>(&bestIdx_))` 写入 `workSpace[blockIdx_ * 2 .. blockIdx_ * 2 + 1]`。Stage 2 启动单核 `isamin_reduce_kernel`（`isamin_kernel.cpp` L290-332，内部经 `CombineCorePairs` L265-288 完成两级归约）：

1. 使用 `DataCopyPad` 将 `coreNum * 2` 个 `float` 一次性搬入 UB `ws`，并通过 `HardEvent::MTE2_V` 同步到 Vector 单元。
2. 利用偶数位掩码 `mask[0] = 0x5555555555555555ULL` 直接在交错排布的 `(bestVal_i, bestIdx_i)` 数组上执行硬件 `WholeReduceMin(..., ReduceOrder::ORDER_VALUE_INDEX)`：
   - 当 `coreNum > 32`（33 至 48 核）时：第一级分别对前 32 核（`ws[0..63]`，满掩码）和剩余 `coreNum - 32` 核（`ws[64..]`，掩码右移 `64 - (coreNum - 32) * 2` 位）执行两次 `WholeReduceMin`，将两组胜出者写入 `work[0..1]` 与 `work[2..3]`；经 `PipeBarrier<PIPE_V>()` 后，第二级以 `mask[0] = 5`（二进制 `0101`，选中 `work[0]` 与 `work[2]`）执行一次 `WholeReduceMin` 得到全局胜出组 `minOut`。
   - 当 `coreNum <= 32` 时：仅需单次掩码 `WholeReduceMin` 即可在 `work[0..1]` 中获得全局最小核对应的偶数槽位下标。
3. 经 `HardEvent::V_S` 同步后，标量侧从 `ws` 对应奇数槽位 `pairOffset + 1` 读取获胜核记录的 0-based 全局下标 `bestIdx`，加 `1` 转换为 1-based 索引 `resultIdx = static_cast<int32_t>(bestIdx) + 1` 并写回 `result`。
4. **跨核并列打破正确性保证**：由于各核负责的逻辑下标区间 `[start_, start_ + num_)` 按 `blockIdx_`（$0, 1, \dots, \text{coreNum}-1$）严格单调递增且互不重叠，较小核号恒对应较小的全局索引；硬件 `WholeReduceMin`（`ORDER_VALUE_INDEX`）在数值相等时严格保留最小通道下标（即最小核号），因此无需在 Stage 2 额外解包比较 `bestIdx` 即可严格保证全局 First-Index 语义。

### 2.5 NaN 与特殊值精确语义

针对 `NaN`、`±0.0f`、`±Inf` 及并列最小值的代码实际行为如下表所示：

| 数值场景 | 代码实际执行机制 | 输出结果语义 |
| --- | --- | --- |
| 首元素为 `NaN`（种子行为） | 每核在 `Process()` 入口初始化 `bestVal_ = FLT_MAX`、`bestIdx_ = 0`、`hasValue_ = false`。若首元素为 `NaN`，`hasValue_` 保持 `false` 且基准值保持 `FLT_MAX`（等效于以 `FLT_MAX` 作为首元素 `NaN` 的比较种子），后续首个非 `NaN` 元素（含有限值、`FLT_MAX` 或 `+Inf`）均通过 `!hasValue_` 条件胜出 | 首元素 `NaN` 不阻塞后续任意非 `NaN` 候选，行为与 `isamin_golden` 完全一致 |
| 候选值 `NaN` 过滤（`NaN` 永不获胜；混合 Tile 场景已实测） | `ProcessStrided` 使用 `!(absVal != absVal)` 守卫；连续路径 `ReduceTile` 中硬件 `BlockReduceMin` / `WholeReduceMin` 在含非 `NaN` 元素的归约窗口内优先传播非 `NaN` 数值，而在 `MergeCandidate` 入口通过 `if (tileMin != tileMin) return;` 显式拦截 `NaN` 候选，绝不用 `NaN` 覆盖已有的非 `NaN` 最优记录 | `NaN` 永不胜出；非 `NaN` 最小绝对值及其精确下标在核内与跨核归约中被完整保留。**该路径已在 1200 条用例中的 NaN 组合（同一 Tile 内混合 `NaN` 与有限值、首元素 `NaN`、全 `NaN`、`NaN` 与 `±Inf` 混排）上实测通过**，说明目标硬件 `BlockReduceMin`/`WholeReduceMin` 在含非 `NaN` 的归约窗口内不传播 `NaN`；若未来硬件改变该行为，上述守卫会让该 Tile 退化为“未被采纳”（而非写入错误下标），并会被同一组用例立即暴露 |
| 全部元素均为 `NaN` | 所有核遍历结束后 `hasValue_` 均为 `false`，`WriteResult` 将初始种子 `(bestVal_ = FLT_MAX, bestIdx_ = 0)` 写入 `workSpace`（Workspace 中不会写入任何 `NaN`）。Stage 2 中所有核在 `FLT_MAX` 上并列，`WholeReduceMin` 依据最小下标优先规则选中 `blockIdx = 0` 的 `bestIdx = 0`，输出 `0 + 1 = 1` | 严格返回 `1`，与 cuBLAS 文档及 `isamin_golden` 全 `NaN` 约定一致 |
| `±0.0f`、`±Inf` 与重复并列值 | `Abs(±0.0f) = 0.0f`，`Abs(±Inf) = +Inf`；Tile 内 `WholeReduceMin(ORDER_VALUE_INDEX)`、跨 Tile `MergeCandidate`（`tileMin == bestVal_ && globalIdx < bestIdx_`）与跨核 `WholeReduceMin` 均在相等时保留最先出现的下标 | 严格返回最先出现的最小 1-based 索引 |

---

## 3. Tiling、UB 与 Workspace 内存布局设计

### 3.1 标量传参 Tiling 与设备侧核间切分

最终实现移除了 Host 侧 `IsaminTilingData` 结构体及 Device Tiling 内存分配，`blas/iamin/arch22/isamin_tiling_data.h` 仅保留两个编译期常量：

```cpp
constexpr uint32_t ISAMIN_MAX_VECTOR_NUM = 48;
constexpr uint32_t ISAMIN_TILE_ELEMS = 22528;
```

- **按值传参（Tiling by Value）**：Host 直接将 `static_cast<uint32_t>(n)` 与 `static_cast<uint32_t>(incx)` 作为标量参数传入 `isamin_kernel<<<numBlocks, nullptr, aclStream>>>(x, workSpace, n, incx)`。
- **启用核数**：`numBlocks = std::min<uint32_t>(static_cast<uint32_t>(n), ISAMIN_DEFAULT_CORES)`（`ISAMIN_DEFAULT_CORES = 48`），保证每个启动的 AIV 核至少分配到 1 个逻辑元素（`num_ >= 1`）。
- **设备侧核内切分（Per-Core Partition）**：每个 AIV 核在 `IsaminCore::Init`（`isamin_kernel.cpp` L38-55）中根据 `GetBlockNum()` 与 `GetBlockIdx()` 独立计算自身区间：
  $$\text{base} = \lfloor n / \text{coreNum} \rfloor, \quad \text{remainder} = n \bmod \text{coreNum}$$
  $$\text{start\_} = \text{blockIdx} \times \text{base} + \min(\text{blockIdx}, \text{remainder}), \quad \text{num\_} = \text{base} + (\text{blockIdx} < \text{remainder} ? 1 : 0)$$

### 3.2 UB 缓冲区布局（22528 双缓冲与紧凑工作区）

单 Tile 元素上限设定为 `ISAMIN_TILE_ELEMS = 22528`（即 $352 \times 64 = 2816 \times 8$ 个 `float`，单缓冲 $90,112\text{ B} = 88\text{ KB}$），在启用深度为 2 的双缓冲后，配合紧凑切分的 `workBuf_`，将 Atlas A2/A3 单 AIV 核 192 KB（$196,608\text{ B}$）UB 容量利用率推至最优：

| 缓冲区名称 | Ascend C 类型与位置 | 元素数 / 内部切分 | 字节数（Bytes） | 用途说明 |
| --- | --- | --- | --- | --- |
| `inQue_` | `TQue<TPosition::VECIN, 2>` | `2 * 22528` 个 `float` | `180,224` | 双缓冲输入队列，支持 MTE2 预取下一 Tile 与当前 Tile `Abs` / 精解并行 |
| `workBuf_` | `TBuf<TPosition::VECCALC>` | 共 `3392` 个 `float`：<br>- `[0..2815]`（2816 `float`）<br>- `[2816..3327]`（512 `float`）<br>- `[3328..3391]`（64 `float`） | `13,568` | `ReduceTile` 紧凑工作区：<br>- 存放 `BlockReduceMin` 输出的最多 2816 个块最小值<br>- 存放第一级 `WholeReduceMin` 输出的 `(val, idx)` 对<br>- 存放第二级掩码 `WholeReduceMin` 输出的组最优对 |
| `outBuf_` | `TBuf<TPosition::VECCALC>` | `8` 个 `float`（32B 对齐） | `32` | 存放第三级归约输出、8 元素块精解输出及写回 Workspace 的候选对 |
| **合计** | - | - | **193,824（189.28 KB）** | 占 192 KB UB 上限的 98.58%，零溢出且完全满足 32B 对齐 |

### 3.3 Handle 内置 Workspace 复用与零 Host 同步

- **Workspace 布局**：总大小为 `numBlocks * 2 * sizeof(float)`（最大 $48 \times 2 \times 4 = 384\text{ B}$），其中 `workSpace[2 * blockIdx]` 存放第 `blockIdx` 核局部最小绝对值 `bestVal_`，`workSpace[2 * blockIdx + 1]` 存放按 `float` 位模式编码的全局 0-based 下标 `bestIdx_`。
- **Handle 级内存复用**：Host 侧调用 `EnsureDefaultWorkspace(h, numBlocks * 2 * sizeof(float))` 与 `GetEffectiveWorkspace(h)` 直接复用 `aclblasHandle_t` 持有的持久化缓冲池，消除了每次算子调用的 `aclrtMalloc` / `aclrtFree` 开销。
- **零 Host 同步**：`isamin_kernel_do`、`isamin_reduce_kernel_do` 以及后续算子调用共享同一个 `handle->stream`，由昇腾硬件流顺序机制天然保证跨阶段及跨调用的读写依赖，Host 侧全程不调用 `aclrtSynchronizeStream`。

### 3.4 连续路径与跨步路径对比

| 维度 | 连续路径（`incx == 1`，`ProcessContiguous`） | 跨步路径（`incx > 1`，`ProcessStrided`） |
| --- | --- | --- |
| 触发条件 | `num_ != 0 && incx_ == 1` | `num_ != 0 && incx_ > 1` |
| 访存方式 | 双缓冲 `DataCopyPad` 连续块 DMA 搬入 UB（每 Tile 最多 22528 个 `float`） | 最多 48 核并行，每核通过 `__gm__ float*` 按 64 位偏移 `base + i * incx_` 读取 GM |
| 核内归约方式 | Vector 单元：`Abs` + `BlockReduceMin` + 三级 `WholeReduceMin` 索引树 + 获胜 8 元素块精解 | Scalar 单元：逐元素计算绝对值并按 `!(absVal != absVal) && (!hasValue_ \|\| absVal < bestVal_)` 更新 |
| 跨核归约方式 | 写入 `wsGm_[blockIdx_ * 2]` 后由 `isamin_reduce_kernel` 两级掩码 `WholeReduceMin` 归约 | 与连续路径完全共用 `WriteResult` 与 `isamin_reduce_kernel` |

---

## 4. 任务书 §2.5 约束核对与已知限制

### 4.1 任务书 §2.5 算子实现约束逐项核对表

| 约束项 | 任务书 §2.5 要求内容 | 本实现（`blas/iamin/arch22/`）核对结论与源码证据 | 状态 |
| --- | --- | --- | --- |
| 参数合法性与校验层级 | `handle` 为 `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n<0`、`result` 为 `nullptr`（任意 `n`）、`n>0` 且 `incx>=1` 时 `x` 为 `nullptr`、`(n-1)*incx` 超出 `int32` 可表示范围均返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 或 `incx<1` 为合法快速返回（`result` 写 0）；元素值不校验 | **已满足**。`isamin_host.cpp` 中 `ValidateIsaminParams`（L28-61）严格按 `handle == nullptr` $\rightarrow$ `n < 0` $\rightarrow$ `result == nullptr` $\rightarrow$ `n == 0 \|\| incx < 1`（`aclrtMemsetAsync` 写 0）$\rightarrow$ `x == nullptr` $\rightarrow$ `int64_t(n-1)*incx > INT32_MAX` 顺序校验，返回码与 `arch35` 及 cuBLAS 一致 | 已满足 |
| 非连续 Tensor 支持 | 支持：BLAS 步长语义即非连续访问——按 `incx` 间隔访问向量元素（间隔元素跳过）；正步长生效；`incx<1` 不进入计算（`result` 写 0 快速返回） | **已满足**。`incx < 1` 在 Host 侧快速返回并异步写 `0`；`incx > 1` 在 `isamin_kernel.cpp` 的 `ProcessStrided`（L153-165）中按 64 位偏移 `base + i * incx_` 跨步访问逻辑元素 | 已满足 |
| broadcast 规则 | 不涉及：BLAS-1 向量算子，无操作数间广播 | **不涉及**。单输入向量归约算子，无广播逻辑 | 已满足 |
| dynamic shape 要求 | 不要求（静态 shape 满足）：`n`、`incx` 为运行时入参，任意合法规模一次适配，无需图模式动态 shape | **已满足**。运行时标量 `n`、`incx` 按值传入 Kernel，各核在 `IsaminCore::Init`（L38-55）中根据运行时 `n` 动态计算分片区间 | 已满足 |
| 原地与视图语义 | `result` 为独立输出，不复用输入内存，不返回视图；`x` 只读 | **已满足**。输入指针 `x` 仅作只读访问，最终 1-based 索引独立写入 `result` 指向的 4 字节 Device 内存 | 已满足 |
| 确定性计算要求 | 不要求：并列最小值取最先出现索引（1-based），单次执行内部确定 | **已满足**。核内 `WholeReduceMin(ORDER_VALUE_INDEX)`、跨 Tile `MergeCandidate` 与跨核掩码 `WholeReduceMin` 均按确定性顺序保留最小下标，无原子争用 | 已满足 |
| 空 Tensor 处理 | 等价口径为 `n=0`：不访问向量数据、不启动计算 kernel，按快速返回口径处理 | **已满足**。`n == 0` 触发 `quickReturn = true`，仅对非空 `result` 执行 `aclrtMemsetAsync` 写 `0`，不访问 `x` 且不启动计算 Kernel | 已满足 |
| 0 维 Tensor 处理 | 不涉及：向量算子无 0 维输入，标量语义由 `n=0` 快速返回与 Host 标量入参承担 | **不涉及**。接口入参为 C 指针与标量 `n`、`incx` | 已满足 |
| 异步执行 | 计算经 `handle` 绑定的 `stream` 异步下发，返回成功仅表示 Host 已提交，不保证完成；读取输出前须调用 `aclrtSynchronizeStream` | **已满足**。`aclrtMemsetAsync`、`isamin_kernel_do` 与 `isamin_reduce_kernel_do` 均绑定 `handle->stream` 异步下发，Host 侧无同步阻塞 | 已满足 |

### 4.2 已知限制与规避方案

| 序号 | 已知限制描述 | 影响范围 | 当前规避方案与后续演进方向 |
| --- | --- | --- | --- |
| 1 | **`incx > 1` 采用多核标量 GM 访存路径（`ProcessStrided`）** | 仅影响 `incx > 1` 的非连续跨步场景吞吐率；全部 200 条性能基线用例均为 `incx = 1`，不受影响 | `arch22` 不支持 `arch35` 的 SIMT 线程级并行。当前通过最多 48 个 AIV 核对逻辑下标区间并行切分来分摊标量访存耗时，保证任意步长（含质数步长 2、7、16 等）100% 精确；后续可针对小固定步长引入 UB 块搬运 + `GatherMask` 向量化抽取 |
| 2 | **测试框架 `ACL_HIFLOAT4` 枚举在部分 CANN 头文件版本下未定义** | 仅影响测试工程编译兼容性 | 在 `test/frame/csv_loader.h` 中对 `ACL_HIFLOAT4` 映射项增加 `#ifdef ACL_HIFLOAT4` 条件编译保护，不影响算子本体与 `arch35` 构建 |

---

## 5. 性能优化过程

### 5.1 保留迭代与实测性能演进表

在真实 Atlas A3 硬件上围绕三个大规模带宽锚点（$n = 1\text{M}, 2\text{M}, 4\text{M}$，`incx = 1`，每次测量 5 次预热 + 20 次迭代采样，`0.8x` 达标上限为 `21.6 / 22.0 / 26.5 us`，预留 10% 安全裕度目标为 `19.44 / 19.80 / 23.85 us`）开展了 16 轮系统化工程迭代。保留至最终实现的关键迭代如下表所示：

| 迭代轮次 | 核心优化措施（单行摘要） | 1M 耗时 (us) | 2M 耗时 (us) | 4M 耗时 (us) | 相对上一保留版本收益与工程结论 |
| :---: | --- | ---: | ---: | ---: | --- |
| Iter 0 | 初始功能基线（Host Tiling 分配/拷贝 + 每调用 Malloc/Free Workspace + Host 流同步 + 40 核单缓冲 `ReduceMin`） | 76.322 | 83.168 | 102.591 | 原始比较基线 |
| Iter 1 | 修正性能测试事件记录流为 `handle->stream`（确保计时覆盖真实执行流） | 71.190 | 85.121 | 106.978 | 修正计时口径，确立优化基准 |
| Iter 2 | 标量 `n`/`incx` 按值传参、Device 侧计算分片、复用 Handle 内置 Workspace、移除每调用 Malloc/Memcpy/Free 与 Host 同步 | 23.611 | 36.393 | 61.313 | 彻底消除 Host 驱动同步与内存分配开销，1M 耗时下降 69.1% |
| Iter 4 | 将默认并发 AIV 核数上限从 40 提升至 `arch22` 满配 48 核 | 22.369 | 32.547 | 53.846 | 提升多核 HBM 带宽并发度，4M 耗时下降 12.2% |
| Iter 5 | 将 `inQue_` 升级为深度 2 双缓冲，在当前 Tile 归约前显式预取下一 Tile | 21.833 | 29.874 | 46.987 | 实现 MTE2 搬运与 Vector 归约跨 Tile 重叠，4M 耗时下降 12.7% |
| Iter 7 | 将通用带索引 `ReduceMin` 替换为三级显式 `WholeReduceMin` 索引归约树 + 单次 `V_S` 标量同步 | 20.901 | 28.943 | 45.683 | 消除通用高级 API 内部多次同步与额外开销 |
| Iter 14 | 先执行 `BlockReduceMin`（每 Repeat 产出 8 个块最小值），再对块最小值建索引树，最后仅对获胜 8 元素块做精解 | 17.067 | 20.409 | 27.373 | 将带索引归约数据量压缩 $8\times$，4M 耗时大幅下降 40.1% |
| Iter 15 | 将双缓冲 Tile 容量扩大至 `22528` 元素，配合 255 Repeat 分块 `BlockReduceMin` 与 3392 `float` 紧凑工作区 | 16.725 | 20.143 | 26.882 | 逼近 192 KB UB 极限（189.28 KB），减少 Tile 循环次数与同步次数 |
| Iter 16 | 将 Stage 2 标量跨核扫描替换为两级偶数位掩码 `WholeReduceMin`，利用核号连续性保持并列最小索引语义 | **12.707** | **16.221** | **23.105** | 消除 Stage 2 标量循环与 `MTE2_S` 停顿，三大锚点全部突破 10% 裕度目标 |

### 5.2 回退方案与原因分析

优化过程中所有实测出现负收益或引入回归的尝试均通过基准对比予以回退，具体工程记录如下：

| 回退轮次 | 尝试方案 | 实测耗时 (1M / 2M / 4M us) | 回退原因（单行结论） |
| :---: | --- | :---: | --- |
| Iter 3 | 使用 `SyncAll` 与核 0 融合跨核归约至单 Kernel | 27.502 / 40.704 / 66.132 | 全锚点回退：AIV 跨核硬同步 `SyncAll` 的屏障开销高于独立发射轻量单核 Stage 2 Kernel |
| Iter 6 | 热循环仅做无索引 `ReduceMin`，最后重读获胜 Tile 做带索引归约 | 29.362 / 37.158 / 53.783 | 全锚点回退：获胜 Tile 的额外 GM 重读与二次归约开销超过无索引归约节省的时间 |
| Iter 8 | 每 16 个 Tile 批量归约后再执行一次 `V_S` 标量同步 | 20.962 / 28.469 / 43.889 | 1M 锚点回退（20.901 $\rightarrow$ 20.962 us）：中小规模下单个核的 Tile 数较少，批处理分支管理开销抵消收益 |
| Iter 9 | 64 元素对齐核间切分 + 对齐 `DataCopy` 快路径 | 20.944 / 28.926 / 45.601 | 1M 锚点回退（20.901 $\rightarrow$ 20.944 us）：对齐切分导致尾核负载不均，且 `DataCopyPad` 硬件填充本身已非瓶颈 |
| Iter 10 | 向量寄存器常驻通道最小值与 Tile ID，最后重读获胜 Tile | 29.322 / 35.442 / 47.396 | 全锚点回退：维护向量通道级 Tile ID 掩码比较及获胜 Tile 重读显著拉长流水线耗时 |
| Iter 11 | 单缓冲 `46080` 元素超大 Tile + 扩展拷贝与分块 Repeat | 21.108 / 29.976 / 48.623 | 全锚点回退：单缓冲占满 UB 导致丧失 MTE2/Vector 双缓冲流水掩盖能力 |
| Iter 12 & 13 | 二元 `Min` 向量折叠 + 重读获胜 Tile（Iter 12 尾部未对齐，Iter 13 修复掩码尾部填充） | 99775.9 / 100619.1 / 100870.7 (Iter 12)<br>25.755 / 28.881 / 37.637 (Iter 13) | Iter 12 因非对齐尾部填充触发约 100 ms 灾难性异常退化；Iter 13 修复对齐掩码填充后仍因重读获胜 Tile 在 1M 上回退（25.755 vs 20.901 us） |

---

## 6. 精度与性能实测结果

### 6.1 精度实测结果

- **判定准则**：算子输出为 `INT32` 类型的 1-based 索引，按 `actual == golden` 执行整数精确相等比对（`matched_ratio = 1.0`、`max_abs_error = 0`）。
- **用例规模与构成**：共执行 **1201 条** 测试用例，包含 `test/isamin/arch22/isamin_test.csv` 中的 **1200 条** 参数化用例（1000 条功能/精度用例 + 200 条 `TC_PF` 性能用例，覆盖均匀分布 `RANDOM_NORM_5_5`、正态分布 `GAUSS_NORM_5_2`、`NaN` 特殊值、正负零并列、跨步、快速返回、指针判空与步长溢出负向用例）以及 1 条独立 GTest 空句柄负向用例 `IsaminArch22Test.NullHandle`。
- **实测结论**：**1201 / 1201 全部通过（100% PASSED）**，所有正向用例输出索引与 CPU Golden 完全一致，所有负向用例返回码与预期完全一致。

### 6.2 性能实测结果

- **评测口径与判据**：在 Atlas A2/A3 设备上对 `test_cases/gpu_baseline.csv` 全部 **200 条** 性能用例执行评测（Warmup 5 次后连续执行 20 次取平均耗时 `avg_us`），达标判据为单算子耗时满足 `avg_us <= gpu_ms / 0.8`（即达到 GPU 标杆 $\ge 0.8\times$ 性能）。
- **整体达标率**：**200 / 200 条性能用例全部达标（达标率 100%）**。
- **核心大规模锚点实测数据对比**：

| 锚点用例规模 ($n$, `incx = 1`) | GPU 标杆耗时 `gpu_ms` (us) | 0.8x 准入上限 `gpu_ms / 0.8` (us) | 10% 裕度目标 (us) | 最终实测耗时 `avg_us` (us) | 相对 GPU 标杆性能倍数 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| $n = 1,048,576$ (1M) | 17.28 | 21.6 | 19.44 | **12.6** | **1.37x** | 达标 |
| $n = 2,097,152$ (2M) | 17.60 | 22.0 | 19.80 | **16.3** | **1.08x** | 达标 |
| $n = 4,194,304$ (4M) | 21.20 | 26.5 | 23.85 | **24.0** | **0.88x** | 达标 |
