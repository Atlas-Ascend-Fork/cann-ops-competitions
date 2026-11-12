# aclblasIcamin 算子设计文档（Atlas A2/A3）

| 项 | 内容 |
| --- | --- |
| 算子名 | aclblasIcamin |
| 对标接口 | cuBLAS `cublasIcamin` |
| 目标硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品（arch22） |
| 验证环境 | Atlas 800T A2（Ascend 910B3），CANN 9.0.0 |
| 代码仓 | https://gitcode.com/qq_46370136/ops-blas ，分支 `feat/aclblas-icamin-arch22` |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务「aclblasIcamin 算子开发（A2/A3）」。ops-blas 仓
`include/cann_ops_blas.h` 已声明 `aclblasIcamin`（L244），`blas/iamin/arch35/` 已有
Ascend 950PR 实现，Atlas A2/A3 对应的 `blas/iamin/arch22/` 缺失，README 产品支持表中
A2/A3 两行标注为「不支持」。本任务补齐 arch22 实现。

## 背景介绍

Icamin 属 BLAS Level-1 索引归约族，查找单精度复数向量中模最小元素的下标，常用于稀疏
矩阵预处理与数值优化中的主元选择。

需要强调的是，BLAS `icamin` 族的「模」并非欧几里得模，而是 1-范数：

```
abs1(x[i]) = |Re(x[i])| + |Im(x[i])|
```

标准参考 BLAS（Netlib）不提供 `icamin` 例程（仅有 `isamin`），因此语义细节以 cuBLAS
官方文档为最高优先级来源，golden 由测试工程内 cblas 风格的 CPU 参考循环生成。

仓内同族现状：

| 实现 | 架构 | 说明 |
| --- | --- | --- |
| `blas/iamin/arch35/icamin_kernel.cpp` | arch35 | 复数版本，使用 SIMT（`asc_vf_call`）与 arch35 专用 helper |
| `blas/iamax/arch22/isamax_kernel_impl.h` | arch22 | 实数最大值版本，A2 vector/reduction 风格 |

arch35 实现依赖 SIMT 与 arch35 专用接口，不能直接移植到 arch22；arch22 需采用 A2 的
vector 归约风格重新实现。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 arch22 上实现 `aclblasIcamin`，与 cuBLAS `cublasIcamin` 逐参数对齐：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

数学表达式：

```
result = argmin_i (|Re(x[k])| + |Im(x[k])|),  i = 1..n,  k = 1+(i-1)*incx
```

结果为 1-based 索引（兼容 Fortran 惯例），写入 Device 内存的单个 INT32。

## 需求拆解

1. 复数模按 BLAS 惯例取 `|Re|+|Im|`，非欧氏模；
2. 1-based 索引输出，多个元素模相同时返回**最小**索引；
3. NaN 语义对齐 cblas 风格循环（见「算子分析/语义」）；
4. `n = 0` 或 `incx < 1`（含 0 与负步长）为合法 quick return：`result = 0` 且返回成功，
   **不触发 kernel**，不做反向遍历；
5. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 支持 `incx > 1` 的非连续访存（功能维度）；
7. 性能：连续访存场景不低于 GPU 标杆的 0.8 倍；
8. 精度：输出为整数索引，按 bit-exact 判定；
9. 补齐 arch22 的 GTest 注册，使 A2/A3 能跑 CSV 驱动测试工程。

# 详细设计（required）

## 算子分析

### 语义

参考实现（cblas 风格，也是 golden 的生成逻辑）：

```cpp
float bestVal = abs1(x[0]);
int bestIdx = 0;
for (int i = 1; i < n; ++i) {
    float v = abs1(x[(int64_t)i * incx]);
    if (v < bestVal) { bestVal = v; bestIdx = i; }   // 严格小于
}
*result = bestIdx + 1;
```

两处细节直接决定实现正确性：

**并列（tie）**：更新条件必须是严格小于 `<`，不能用 `<=`。否则后出现的等值元素会覆盖
先出现的，返回的不再是最小索引。

**NaN**：IEEE 语义下任何涉及 NaN 的 `<` 比较均为 false，因此由上述循环可推出：

| 情况 | 期望结果 | 原因 |
| --- | ---: | --- |
| 首元素模为 NaN | 1 | 基准即 NaN，后续任何值都不满足 `< NaN` |
| 非首元素模为 NaN | 不更新 | NaN 不满足 `<` |
| 全部为 NaN | 1 | 同首元素情形 |
| 全部为 Inf | 1 | 后续 Inf 不小于首个 Inf |
| 首个为 Inf，后有有限值 | 该有限值下标 | 有限值严格小于 Inf |

注意：**不能**把归约初值设为 `FLT_MAX`——那样首元素为 NaN 时会被后续有限值取代，
与 golden 不一致。

### 边界与异常行为

| 条件 | 行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE`（**含 quick return 路径**） |
| `n == 0` 或 `incx < 1` | 写 `result = 0`，返回 `ACLBLAS_STATUS_SUCCESS` |
| `x == nullptr`（`n>0 && incx>=1`） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `(n-1)*incx > INT32_MAX` | `ACLBLAS_STATUS_INVALID_VALUE` |

`incx < 1` 统一走 quick return 而非反向遍历，这一点与 axpy/nrm2 族不同，是 iamin 族的
既有约定（对齐仓内 `blas/iamin/README.md` 与 isamin arch35 实现）。

### 存储、workspace、异步

- `x` 为交错存储的 complex64，n 个复数元素对应 2n 个 float；
- `result` 为 Device 内存的单个 INT32；
- workspace 取自 handle（`GetEffectiveWorkspace`），仅需
  `ICAMIN_MAX_CORES * ICAMIN_SLOT_FLOATS * sizeof(float)` = 320 B，接口内**不做**
  `aclrtMalloc`/`aclrtFree`；
- 执行异步，依赖 `aclblasSetStream` 绑定的 stream，读回结果前需同步。

## 算子实现

### 总体架构

```
GM（交错 complex64）
   │
   ├─ 每核一段连续切片（host 按 4 复数对齐切分）
   │     │
   │     └─ 按 tile 遍历：
   │          CopyIn（incx=1 用 DataCopy/DataCopyPad；incx>1 逐元素 gather）
   │          Abs → GatherMask 去交错 → Add            得到 |Re|+|Im|
   │          Compare(EQ 自比) + Select                NaN 模改写为 +Inf
   │          WholeReduceMin + GetReduceMaxMinCount    折叠出 (value, index)
   │          与本核当前候选按严格小于合并
   │     │
   │     └─ 写 workspace[blockIdx] = {value, index}
   │
   ├─ SyncAll()
   └─ 0 号核按核序合并所有 slot → result = globalIndex + 1
```

### Host 侧设计

**参数校验顺序**（顺序本身是语义的一部分）：
`handle → n → result → quick return → x → stride 溢出`。
`result` 的检查必须早于 quick return，因为任务书要求 `n=0` / `incx<1` 且
`result == nullptr` 时仍返回 `INVALID_VALUE`。

**quick return 的写法**：`result` 位于 Device 内存，直接 `*result = 0` 是 host 对
device 指针解引用。正确做法是走 stream：

```cpp
aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
```

**溢出检查用 int64**：`(n-1)*incx` 在 int 域会先溢出，检查就永远不会触发：

```cpp
const int64_t lastComplex = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
if (lastComplex > static_cast<int64_t>(INT32_MAX)) return ACLBLAS_STATUS_INVALID_VALUE;
```

**分核策略**：`numBlocks = min(GetAivCoreCount(), 40, totalN / 1024)`，至少 1。
按每核至少约 1024 个复数元素投放，避免小向量被拆到多核后 slot 开销反而占主导。

`incx == 1` 时每核切片按 4 个复数（= 8 float = 32 B）上对齐，保证整 tile 的
`DataCopy` 在地址和长度上都是 32 B 对齐；`incx > 1` 逐元素 gather，无对齐要求。

**tiling 按值传递**：`IcaminTilingData` 是 24 B 的 POD，作为 kernel 实参随
`<<<>>>` 传入，而不是先 `aclrtMemcpy` 到 device 再由 kernel 读取。这一点对小 shape
性能影响显著（见「性能优化过程」）。

### Kernel 侧设计

**求模**：先对整个交错 tile 做一次 `Abs`（2×lanes 个 float），再用 `GatherMask`
按 pattern 1/2 分离实部与虚部，最后 `Add`。对交错数据整体取绝对值等价于分别取，
这样比去交错后做两次 `Abs` 少一趟向量操作。

**NaN 处理**：`Compare(mask, mag, mag, CMPMODE::EQ)` —— 一个值与自身做有序比较，
只有 NaN 会得到 false，因此 mask 恰好标出非 NaN 的 lane；再用
`Select(..., VSEL_TENSOR_SCALAR_MODE)` 把 NaN lane 换成 `+Inf`。采用标量立即数形式，
免去每个 tile 额外 `Duplicate` 一个 +Inf 向量。

全局首元素是否为 NaN 由其所属核（`startOffset == 0`）单独判定，若是则直接发布
`(NaN, 0)`，使跨核合并结果与顺序扫描一致。

**尾块**：tile 按 64 lane 对齐，不足部分填 `+Inf`，使每个 tile 都能按统一宽度归约，
且填充 lane 永远不会赢得最小值。

**归约**：`WholeReduceMin` 配合 `SetMaskCount` / `SetVectorMask<float, COUNTER>`
一次折叠整个 tile，随后 `GetReduceMaxMinCount` 取出 (value, index)，索引以 float 位
模式返回，需 `reinterpret_cast<uint32_t*>` 还原后加上 tile 起始偏移。

**跨核合并**：每核写一个 `{value, index}` slot（均以 float 位模式存放，使一次浮点
比较即可完成合并），`SyncAll()` 后由 0 号核按**核序**（即向量的自然顺序）以严格小于
合并，等价于整段从左到右扫描，因此并列取最小索引、NaN 语义都与 golden 一致。

### 软件流水

朴素写法「拷贝 tile t → 折叠 tile t」会让 MTE2 与 V 串行，深度 2 的队列形同虚设。
实现中改为先预取 tile 0，循环内在折叠 tile t 之前先发起 tile t+1 的拷贝：

```cpp
CopyInTile(tile 0);
for (t = 0; t < tiles; ++t) {
    if (t + 1 < tiles) CopyInTile(tile t+1);   // MTE2 与下面的 V 重叠
    FoldTile(tile t);
}
```

这是本算子最大的一处性能收益（n=4194304：44.3 us → 36.4 us）。

### UB 预算

tile = 6144 复数元素时：

| 用途 | 大小 |
| --- | ---: |
| 交错输入（深度 2 队列，2×2×6144×4 B） | 98304 B |
| re / im / magnitude 暂存（3×6144×4 B） | 73728 B |
| Compare mask（1 bit/lane） | 800 B |
| slot 暂存 | 352 B |
| **合计** | **≈169 KB** |

低于 192 KB 预算。实测 tile = 8192 会导致 UB 溢出（接口返回码 6），故取 6144。

### 目录与构建

| 路径 | 说明 |
| --- | --- |
| `blas/iamin/arch22/icamin_host.cpp` | 参数校验、tiling、kernel 下发 |
| `blas/iamin/arch22/icamin_kernel.cpp` | AIV kernel |
| `blas/iamin/arch22/icamin_kernel.h` | `icamin_kernel_do` 声明 |
| `blas/iamin/arch22/icamin_tiling_data.h` | `IcaminTilingData` 与常量 |
| `test/iamin/icamin/arch22/icamin_test.cpp` | CSV 驱动 GTest |
| `test/iamin/icamin/arch22/icamin_test.csv` | 1200 条用例 |
| `test/iamin/icamin/gpu_baseline.csv` | 性能标杆 |
| `test/iamin/icamin/CMakeLists.txt` | **补齐 arch22 的 GTest 注册** |

接口声明复用 `include/cann_ops_blas.h`，不新增私有平行 API。

原仓 `test/iamin/icamin/CMakeLists.txt` 只在 arch35 分支调用
`ops_blas_add_gtest_tests`，A2/A3 构建会退回非 GTest runner（只打印整数毫秒，
且不产生验收脚本所解析的 `[PERF] avg_us` 行），本次一并补齐。

编译：

```bash
bash build.sh --soc=ascend910b --ops=icamin          # A2
bash build.sh --soc=ascend910_93 --ops=icamin        # A3
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |

## 算子约束限制

- `n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- `incx < 1`（含 0 与负步长）为 quick return，**不做反向遍历**；
- `(n-1)*|incx|` 超出 int32 可表示范围时返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- `handle` / `x` / `result` 不可为 nullptr（`x` 在 quick return 路径不检查，
  `result` 在所有路径都检查）；
- 不涉及广播、dynamic shape、原地更新；
- 不要求非连续 tensor 的 leading dimension padding 场景。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出为 INT32 索引，与 golden **精确一致**（bit-exact，`EXPECT_EQ`），非浮点容差。归约顺序不影响离散的最小值索引，正常实现应完全一致 | 任务书 §3.2；生态算子开源精度标准（整数输出时阈值退化为精确判定） |
| 性能标准 | 单 case 平均耗时 ≤ GPU 标杆 / 0.8（即达到标杆 0.8 倍以上性能） | 任务书 §3.3，标杆见 `gpu_baseline.csv` |

## 测试设计

CSV 驱动的 GTest 工程，1200 条用例（1000 精度 + 200 性能）：

| 类别 | 前缀 | 覆盖 |
| --- | --- | --- |
| 基础 | TC_L0 | 小尺寸 × 基础步长，含全零填充的 tie 语义 |
| 尺寸扫描 | TC_SQ | 1 → 2^20，含质数、2 的幂及 ±1、非对齐值 |
| 步长 | TC_INC | 正步长 1/2/3；`incx = 0` 与负步长的 quick return |
| 填充 | TC_FL | 均匀/全零/正负交替/极端值/Inf/NaN，以及首元素 NaN、中段 NaN 的混合场景 |
| 边界负向 | TC_ED | `n=0`、负 n、空 x、空 result、`incx=INT32_MAX` 跨步溢出 |
| 扩展 | TC_EX | 尺寸 × 步长 × 填充的确定性采样 |
| 性能 | TC_PF | 连续访存，含任务书 §3.3 的三个典型 case |

两点测试工程层面的设计：

**负向用例不按步长分配缓冲区**。`TC_ED_092`（`n=4, incx=INT32_MAX`）若按
`(n-1)*|incx|` 预分配，需要约 51 GB，测试会因自身分配失败而误判接口行为。这类用例改为
用单元素 dummy device 缓冲区直接调用公共 API——接口必须在读取数据前返回错误码。

**绕过 wrapper 的直调用例**。`icamin_npu_wrapper.h` 会在进入公共 API 前把空 handle
转成 `ACLBLAS_STATUS_NOT_INITIALIZED`，因此经 wrapper 的用例覆盖的是 wrapper 行为。
文档补充了两个直调公共 API 的用例：`NullHandle`（期望 `HANDLE_IS_NULLPTR`）与
`NullResultOnQuickReturn`（`n=0` / `incx=0` 且 result 为空，期望 `INVALID_VALUE`）。

执行方式：

```bash
python3 verify_accuracy.py --repo <ops-blas> --soc ascend910b --csv ./icamin_test.csv --device 0
ICAMIN_PERF_BENCH=1 python3 verify_performance.py --repo <ops-blas> --soc ascend910b --device 0
```

## 自验证结果

环境：Atlas 800T A2（Ascend 910B3），CANN 9.0.0。

| 项 | 结果 |
| --- | --- |
| 精度 | **1000 / 1000 PASS**，全部 bit-exact |
| 性能 | **200 / 200 PASS**，无 NO_REF，最低 baseline/NPU = 0.833 |
| 内存 | host 峰值 64.000 MiB（预算 4096 MiB），device 峰值 32.000 MiB，UB 单核约 169 KB |

任务书 §3.3 指定的三个典型 case：

| n | NPU (us) | GPU 标杆 (us) | baseline/NPU |
| ---: | ---: | ---: | ---: |
| 1048576 | 16.90 | 18.16 | 1.075 |
| 2097152 | 23.64 | 21.62 | 0.915 |
| 4194304 | 36.47 | 36.64 | 1.005 |

性能采样：warmup 10 次后连续下发 100 次，`aclrtEvent` 夹取设备耗时取均值，含下发开销，
不含 H2D/D2H、host 造数与 golden 比对。

### 性能优化过程

| 优化项 | 效果 |
| --- | --- |
| tiling 由 device 拷贝改为按值传参 | 固定开销 17.2 us → 7.0 us，所有小 shape 用例转为 PASS |
| MTE2/V 软件流水（预取下一 tile） | n=4194304：44.3 us → 36.4 us |
| tile 4096 → 6144 | n=4194304：38.3 us → 36.4 us |
| `Select` 改标量立即数形式，省去逐 tile `Duplicate` | n=4194304：约 2 us |

其中「tiling 按值传参」是小 shape 的决定性因素：原先每次调用都有一次阻塞式
`aclrtMemcpy`，使 n=1 与 n=1048576 的耗时同为约 17 us，固定开销完全掩盖了计算本身。

曾尝试但回退的方案：将 tiling 改为 `aclrtMemcpyAsync` 上传——性能劣化 2~3 倍，且
tiling 是栈对象，异步拷贝可能在其析构后才真正读取，存在正确性隐患。

## 兼容性分析

新增 arch22 实现，复用已有公共接口声明，不改变 arch35 行为，不涉及接口兼容性问题。
`blas/iamin/README.md` 中 `aclblasIcamin` 的 A2/A3 支持状态由「不支持」改为「支持」；
同族实数算子 `aclblasIsamin` 本次未实现 arch22，其支持状态保持不变。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 并列时返回非最小索引 | 全链路（tile 内、tile 间、核间）统一使用严格小于；核间按核序合并 |
| NaN 语义与 golden 不一致 | 首元素 NaN 由所属核强制发布 `(NaN, 0)`；归约初值不使用 `FLT_MAX`；TC_FL 覆盖首/中段 NaN |
| 尾块读到越界或残留数据 | tile 按 64 lane 对齐，填充 lane 一律 `+Inf` |
| UB 溢出 | tile 固定 6144，预算约 169 KB / 192 KB；8192 实测溢出 |
| 大步长下 host 侧整数溢出 | `(n-1)*incx` 以 int64 计算后再比较 |
| 连续执行时性能互扰 | 判定以单用例采样为准，`verify_performance.py` 逐条解析 `[PERF] avg_us` |
