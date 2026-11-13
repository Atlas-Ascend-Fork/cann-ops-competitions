# aclblasIcamin 算子开发（Atlas A2/A3）设计文档

> 任务 ID：`aclblas-icamin-a2a3`；任务名称：11月社区任务-aclblasIcamin算子开发（A2/A3）
> 仓库：ops-blas（分支 `icamin-arch22-a2a3`）；核心交付 commit：`7fabc54`（框架）+ `c197f8e`（真机收尾）
> 说明：本文档基于真机（2× Ascend910，CANN 8.5.1）联调实证撰写，所有数字均可复现（见"实现状态与真机验证"一章）。

## 需求背景

### 需求来源

11 月社区任务 `aclblas-icamin-a2a3`（"11月社区任务-aclblasIcamin算子开发（A2/A3）"）要求为 ops-blas 仓补齐 `aclblasIcamin`（complex64，Atlas A2/A3 arch22 路径）的实现与 CSV 驱动测试，落点 `blas/iamin/arch22/` 与 `test/iamin/icamin/arch22/`。本设计文档描述该实现的语义契约、架构与真机验证结论。

### 算子语义

`aclblasIcamin(handle, n, x, incx, result)`：在复向量 `x` 中按步长 `incx` 取 `n` 个元素，返回 **`|Re| + |Im|`（abs1）最小元素的全局索引（1-based）**。

关键语义（与 arch35 golden `test/iamin/icamin/icamin_golden.h` 完全对齐）：

- **并列取首**：abs1 相同时取最小索引（golden 仅在 `absVal < bestVal` 严格小于时更新）；
- **全零**：全零复向量返回索引 1（首个 0 的位置）；
- **NaN**：golden 以 `abs1(x[0])` 为种子、只在严格小于时更新——首元素为 NaN 时种子永不被超越（NaN 参与的 `<` 恒假），**结果钉死为索引 1**，后续有限值无法取代；中位 NaN 因不满足严格小于永不参选；
- **边界**：`n < 0` 返回 `INVALID_VALUE`；`n == 0 || incx < 1` 快速返回 `SUCCESS` 且 `*result = 0`；`result == nullptr` 返回 `INVALID_VALUE`（该校验先于快速返回，任何 n 下都生效）；`x == nullptr` 返回 `INVALID_VALUE`；跨步溢出防护：末复元素索引 `(n-1)*incx` 超过 `INT32_MAX` 或其 float 偏移超 `INT64_MAX/2` 返回 `INVALID_VALUE`。

### 与 isamax/isamin 同族关系

iamin/iamax 族共享"幅值极值 + 全局索引"的归约骨架：

| 接口 | 输入 | 幅值定义 | 极值方向 | NaN 首元素 |
| --- | --- | --- | --- | --- |
| `aclblasIsamax` | fp32 | `|x|` | 最大 | — |
| `aclblasIsamin` | fp32 | `|x|` | 最小 | 不钉死（种子-续扫） |
| `aclblasIcamax` | complex64 | `|Re|+|Im|` | 最大 | — |
| `aclblasIcamin`（本任务） | complex64 | `|Re|+|Im|` | 最小 | 钉死索引 1 |

icamin 的两个特征点：幅值是复数成对折叠（`|Re|+|Im|`，A2 无 DeInterleave，采用 isamax 遗产的 in-place PairReduceSum 管线）；NaN 首元素语义与 isamin 相反——icamin 用 **-Inf 哨兵**（幅值非负，-Inf 胜出一切实幅值）在跨核归约中把结果钉死到索引 1。

### 现状分析

任务基线时 arch35 已有 icamin 实现，arch22（A2/A3）路径缺失——A2/A3 用户无法调用。本任务在 `blas/iamin/arch22/` 新建 kernel（`icamin_kernel.cpp`/`icamin_kernel_impl.h`）、host（`icamin_host.cpp`）、tiling 与 CSV 驱动测试三件（`test/iamin/icamin/arch22/`，1200 用例），并完成真机验证闭环。

## 需求分析

### 需求描述

按 arch35 既有语义，在 Atlas A2/A3（arch22，dav-2201）上实现 `aclblasIcamin`：

1. 函数签名（`include/cann_ops_blas.h:244`）：
   `aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);`
2. 输入输出规格：
   - `x`：complex64（交叠 float Re/Im）向量首地址，`incx` 为复元素步长（≥1 生效；incx>1 提供逐元素 int64 寻址跨步路径，只读恰好 n 个复元素）；
   - `result`：设备侧 int32 标量输出（1-based 全局索引），host 不解引用该指针，快速返回路径通过流上 `aclrtMemsetAsync` 写 0；
   - 同族 `aclblasIsamin`（fp32）为同批交付的同族扩展（见文末）；int64 索引变体不在本任务范围。
3. 语义全部以 arch35 golden 为准（含 NaN 首元素钉死、并列取首、边界码表）。
4. 全部用例在真机通过 CSV 驱动 GTest 验证；性能门（任务 §3.3：NPU 均值 ≤ GPU 基线 1.25 倍）由 `gpu_baseline.csv` + `TC_PF_*` 用例承载，默认开关 `ICAMIN_PERF_BENCH` 控制，msprof 为验收口径。

### 需求拆解

| 子项 | 内容 | 落点 |
| --- | --- | --- |
| D1 kernel 骨架 | 单 kernel + 跨核同步两阶段归约框架（isamax/icamin 同族遗产） | `blas/iamin/arch22/icamin_kernel_impl.h` |
| D2 分核与 tiling | host 计算分核数、每核偏移/长度，经 tiling 结构下发 | `icamin_host.cpp` + `icamin_tiling_data.h` |
| D3 三条扫描路径 | incx>1 跨步标量、短向量标量、长向量 4096-complex 分 tile 管线 | `icamin_kernel_impl.h` |
| D4 幅值折叠与 NaN 语义 | `Abs → PairReduceSum → mag`；向量路径 NaN 幅值→+Inf 消毒；-Inf 哨钉复现首 NaN 钉死 | kernel 各阶段 |
| D5 跨核归约与输出 | 每核发布 (value, index) 槽位 → 软件 barrier → 核 0 合并 → 写 1-based int32 | `PublishSlot`/`SoftBarrier`/`MergeCores` |
| D6 host 校验链 | arch35 同款校验顺序 + 复数跨步溢出防护 + 快速返回 | `icamin_host.cpp` |
| D7 测试 | CSV 驱动 GTest 1200 用例（负例/浮点特殊值/边界/大 n/跨步/快速返回）+ 性能门 + 真机验证 | `test/iamin/icamin/arch22/` |

## 详细设计

### 数学公式与 golden

设 `k(i) = 1 + (i-1)*incx`（1-based 全局复索引），`abs1(z) = |Re(z)| + |Im(z)|`，目标：

```
result = argmin_{i ∈ [1, n]} abs1(x[k(i)])     （并列取最小 i）
```

golden（CPU 参照，`aclblasIcamin_cpu`）：

```c
bestVal = abs1(x[k(1)]);  bestIdx = 0;         // 种子可为 NaN：钉死语义由此产生
for (i = 2 .. n):
    absVal = abs1(x[k(i)]);
    if (absVal < bestVal)                      // 严格小于；NaN 参与恒假
        { bestVal = absVal; bestIdx = i-1; }
return bestIdx + 1;
```

kernel 对该模型的两处等价复现：

- **NaN 消毒**：向量路径中每个 NaN 幅值被重写为 `+Inf`（`Compare(CMPMODE::EQ)` + `Select`），使归约不依赖裸 NaN 行为——NaN 从不成为候选；
- **首 NaN 钉死**：持有全局元素 0 的核在 `PublishSlot` 直接检查 `x[0]`，若 Re/Im 任一为 NaN，则发布 **-Inf 哨兵槽位 `(value=-Inf, index=0)`**——幅值非负，-Inf 胜出跨核最小值，索引 0 输出为 1，等价于 golden 的钉死语义；全 NaN 向量退化为 +Inf 并列，按并列取首回落核 0 → 同样 result 1。
- **并列取首**：所有比较均为显式 `(value, 0-based 全局 index)` 二元组比较，不信任归约原语的并列行为。最终输出 `bestIdx + 1`。

### 支持的 dtype 与形状

- 输入 dtype：complex64（交叠 fp32 Re/Im）；输出：int32 标量（设备内存）。
- 形状：一维复向量，`n ≥ 0`；`n == 0` 快速返回 0。`n` 上限受校验链约束：`(n-1)*|incx| ≤ INT32_MAX` 且 float 偏移 ≤ `INT64_MAX/2`。
- `incx`：≥1；`incx == 1` 走对齐向量管线（每核切片起点 4-complex / 32B 对齐），`incx > 1` 走逐元素跨步路径。
- 负例：`n < 0`、`result == nullptr`、`x == nullptr`、跨步溢出 → `INVALID_VALUE`（返回码在读 x 之前产生，不依赖输入缓冲）。

### 分核策略：blockDim = needVecCoreNum

host 侧 `CalcIcaminTiling`：

```
aivCoreNum = GetAivCoreCount()                 // 运行时查询，不硬编码
cores     = ceil(n / ICAMIN_TILE_COMPLEX)      // 每 4096 复元素一片
cores     = min(cores, aivCoreNum, ICAMIN_MAX_VECTOR_NUM=40)
perCoreN  = ceil(n / cores)；incx==1 时向上取整到 4 复元素（8 float = 32B 对齐）
useCoreNum(=needVecCoreNum) = ceil(n / perCoreN)
startOffset[i] = i * perCoreN
eleTotalEachCore[i] = 末核取余量，其余为 perCoreN
numBlocks = needVecCoreNum                     // 一个 block 覆盖一个向量核分片
```

**为什么 blockDim 不能取 `ceil(needVecCoreNum/2)`**（isamax 遗产惯例）：该惯例假设 mix-mode 启动下每个 block 提供 2 个 AIV。本机 on-device probe 实测否定该假设——`[PROBE] numBlocks=1 needVecCoreNum=1 aivs: [aiv=0 blknum=1]`，`GetBlockNum()==numBlocks`，即本平台**每 block 恰好 1 个 AIV**，`GetBlockIdx()` 直达 `0..numBlocks-1`。按 `/2` 启动会让一半分片（高半区 `startOffset`）永远无人处理，大 n 用例直接错值。故 host 按 `numBlocks = needVecCoreNum` 启动；kernel 内 `blockIdx + 1 > needVecCoreNum` 的兜底分支保留但在此约定下永不触发。

### 跨核归约 MergeCores（两阶段）

**阶段一（发布）**：每个活跃核把本核扫描结果折叠为单个 `(minVal, 0-based index)` 槽位，`DataCopyPad` 精确写入 workspace 槽区 `workspace[blockIdx * 2]`（2 float/槽，总长 `needVecCoreNum * 8` 字节）；无候选的核发布 `+Inf`（与一切实幅值竞争必败）。

**阶段二（合并）**：barrier 之后由核 0 将全部槽位拷入 UB（只拷精确字节数，避免未初始化尾字进入极小值——幅值非负，0 会假胜），做 (value, index) 标量折叠，胜者索引 +1 经 `outTensorsGM` 写回 `result`（int32，1-based）。

**为什么用 ≤2×40 float 的标量折叠替代 mask-form ReduceMin**：初版沿 isamax 思路用偶数 lane 掩码 + 2-repeat mask-form `ReduceMin` + `GetTPipePtr()->FetchEventID` 事件对完成合并。真机逐段 bisect 定位（同族 isamin 上完成四步实验：scans → +PublishSlot → +SoftBarrier → +MergeCores）：仅当该合并路径执行时 `aclrtSynchronizeStream` 返回设备错误 **507235**，即掩码归约 + 高级事件 API 在内核中触发设备异常。槽位规模上限 `2 * ICAMIN_MAX_VECTOR_NUM = 80` float，标量 `GetValue` 折叠毫秒级完成（功能路径、非性能路径），语义与掩码归约严格等价（并列取最小全局下标；全 +Inf 并列回落核 0 与 golden 全 NaN → 1 一致；-Inf 哨兵天然胜出）。

### 【重点】软件 barrier 设计：为什么不用无参 SyncAll()

**问题**：初版框架沿 isamax 惯例调用无参 `AscendC::SyncAll()`。真机首个用例（n=1，单 block）即挂死 >12 分钟、AICore 100% 自旋。定位实验序列（isamin 上完成，icamin 复用结论）：

1. 将 `Process()` 置为 no-op → 全部用例秒级跑完（启动/host/流同步全链正常，仅结果错）；
2. 仅保留 `SyncAll()` → 首例即挂死（timeout 杀掉）。

结论：`SyncAll()` 的软同步实现要在 GM 同步工作区上自旋，而该隐藏同步 workspace 在**自定义算子的 `<<<N>>>` 启动路径**下未被 provision（本机 CANN 8.5.1；仓内 lt_matmul 新代码调用 `SetSyncBaseAddr` 配置同步区，但该 API 仅存在于更新版本 CANN）。参与集本身正确（probe 证明 `GetBlockNum()==numBlocks`），问题纯在同步存储不可用。

**方案：workspace 内自管理软件 barrier**（`SoftBarrier()`，模式对齐 tikcfw `IBSet/IBWait`）：

- **存储**：op workspace 布局为 `[槽区 | 标志区]`，标志区紧随槽区、32B 对齐，每 block 一个 32 B（8×int32）标志字，容量 `ICAMIN_BARRIER_BYTES = 4 * 8 * 40 = 1280 B`；host 在 kernel 启动前对整个 workspace `aclrtMemsetAsync` 清零（与 kernel 同流，天然先序）。
- **发布**：每个 AIV 将自己的标志字置 1，经 MTE3 `DataCopy` 写入 `barrierFlagGM[blockIdx * 8]`。同核 MTE3 管线先进先出，标志字写入晚于该核槽位写入完成——**观察到一个核的标志位为 1，即可安全读取它的槽位**，barrier 同时承担槽位发布的可见性语义（-Inf 哨兵/槽位因此对合并核可见）。
- **轮询**：每核循环 `DataCopy`（队路由 MTE2→V→S，走 `TQue<VECIN,1>` 的 EnQue/DeQue 同步）读取 `bnum * 8` float 的标志区，逐字检查全 1 即出环；出环后 `PipeBarrier<PIPE_ALL>` 收尾。自旋只经过 L2（GM 写读同域），与框架 IBSet/IBWait 轮询模型一致。
- **参与集**：全部 `numBlocks = GetBlockNum()` 个 AIV（含"仅入 barrier 即退"的兜底分支），集合均匀，无死锁条件。

该 barrier 与 `SyncAll` 语义等价、零额外依赖（只用自有 workspace），真机 1200 用例无一次挂死。

### Tiling 数据结构

`icamin_tiling_data.h`（host/kernel 共享，整份作为 tiling 镜像 `aclrtMemcpy` 下发）：

```c
constexpr uint32_t ICAMIN_MAX_VECTOR_NUM    = 40;    // 并发向量核上限（A2 40 AIV；host 运行时查询后截断）
constexpr uint32_t ICAMIN_TILE_COMPLEX      = 4096;  // 每 tile 4096 复元素 = 8192 float = 128 个整 64-lane repeat
constexpr uint32_t ICAMIN_SCALAR_MAX_COMPLEX = 512;  // 短向量走标量扫描的阈值
constexpr uint32_t ICAMIN_BARRIER_SLOT_FLOATS = 8;   // 每 block 32B 标志字
constexpr uint32_t ICAMIN_BARRIER_BYTES     = 1280;  // 标志区总容量

struct IcaminTilingData {
    uint32_t incx;                                    // 步长（复元素），≥1
    uint32_t needVecCoreNum;                          // 实际用核数
    uint32_t rstLenAllCoreBytes;                      // 槽区精确字节数 = useCoreNum * 8
    uint32_t startOffset[ICAMIN_MAX_VECTOR_NUM];      // 每核首逻辑复索引
    uint32_t eleTotalEachCore[ICAMIN_MAX_VECTOR_NUM]; // 每核复元素数
};
```

kernel 侧按 `tilingBuf[0..2]` 读标量、`[3+i]` / `[3+40+i]` 读每核偏移/长度（`ParseTilingData`），据此判定本核是否活跃。

### UB 缓冲

| 缓冲 | 类型 | 容量 | 用途 |
| --- | --- | --- | --- |
| `inDataQueue` | TQue<VECIN,2> | 2 × 32 KB | incx==1 分 tile 双缓冲流水（CopyIn → 幅值折叠 → 归约） |
| `infSrcBuf` | TBuf<VECCALC> | 16 KB | +Inf 源（NaN 消毒 Select 替补张量 / tile 尾部填充） |
| `cmpMaskBuf` | TBuf<VECCALC> | 4 KB | Compare 掩码（NaN 检测） |
| `reduceWorkBuf` | TBuf<VECCALC> | (4096/64)×16+64 float ≈ 4.5 KB | tile 内 indexed ReduceMin 共享临时区 |
| `slotBuf` | TBuf<VECCALC> | 63 槽 × 8 float ≈ 2 KB | 每 63 个 tile 批量暂存 (value,index) 槽位 |
| `outDataQueue` | TQue<VECOUT,2> | 2 × 32 B | 槽位发布（PublishSlot DataCopyPad 源） |
| `coresRstInDataQueue` | TQue<VECIN,1> | 512 B | 核 0 合并读入全部槽位 |
| `coresRstOutDataQueue` | TQue<VECOUT,1> | 32 B | 合并结果 int32 出队下发 MTE3 |
| `barrierPollQueue` | TQue<VECIN,1> | 1280 B | barrier 标志区轮询读 |
| `barrierFlagBuf` | TBuf<VECCALC> | 32 B | 本核标志字发布源 |

合计约 95 KB，在 A2 192 KB UB 预算内（8192 float tile 保证每轮 Abs / PairReduceSum / ReduceMin ≤255 repeat，无需子批处理）。

### Host 流程与校验链（arch35 同款顺序）

```
handle == nullptr            → HANDLE_IS_NULLPTR (9)
n < 0                        → INVALID_VALUE (3)
result == nullptr            → INVALID_VALUE (3)   // 先于快速返回，任何 n 生效
n == 0 || incx < 1           → 快速返回：流上 memsetAsync(*result = 0)，SUCCESS
x == nullptr                 → INVALID_VALUE (3)
(n-1)*incx 溢出 int32 / float 偏移溢出 int64 → INVALID_VALUE (3)
```

通过校验后：`CalcIcaminTiling` → 每次调用独立分配 `[槽区+标志区]` workspace 与 tiling 镜像（memsetAsync 清零、流上先序）→ `icamin_kernel_do<<<numBlocks, nullptr, stream>>>` → `aclrtSynchronizeStream` → 释放（同步后释放，规避同 handle 多流 workspace 竞态）。

### 性能门（任务 §3.3）

`TC_PF_*` 用例 + `gpu_baseline.csv`（GPU 基线毫秒）承载：NPU 事件计时均值 `avg_us ≤ gpu_ms * 1000 / 0.8`（即 ≤1.25× GPU 基线），由环境开关 `ICAMIN_PERF_BENCH` 启用；msprof op 任务时长为验收口径，本表内 Event 计时仅为便捷门。功能验证默认关闭该开关（TC_PF 行按普通功能用例执行）。

## 实现状态与真机验证（as-built）

### 验证结论

**真机（device 1，2× Ascend910，CANN 8.5.1）全量 1201/1201 通过**：1200 条 CSV 用例 + 1 条 NullHandle 直调用例，0 失败、0 跳过。

证据链：

- 构建：`flock -w 600 <build.lock> cmake --build build --target icamin_test -j1`（nproc=1 全局串行），TARGET_RC=0；
- 运行：`BLAS_DEVICE_ID=1 timeout 600 build/test/iamin/icamin/icamin_test build/test/iamin/icamin`，RUN_RC=0；
- 摘录：`[==========] 1201 tests from 2 test suites ran. (76875 ms total)` / `[  PASSED  ] 1201 tests.`（1200 CSV 用例 73528 ms）；逐用例 0 条 return-code / index-mismatch / golden 失败；
- 报告文件：`test/iamin/icamin/arch22/icamin_arch22_test_report.md`（关键字段 1201/1201 passed, 0 failed, 0 skipped）；
- 交付 commit：`7fabc54`（框架与测试三件）+ `c197f8e`（真机收尾，本地提交，未 push）。

### 真机收尾修复（7fabc54 → c197f8e）

任务框架代码提交时未经过真机验证，真机联调暴露并修复 5 处阻断缺陷（方案均已在本仓同族 isamin 上先行验证，commit f166bb1）：

1. 两处 `reinterpret_cast<__gm__ T*)` 缺括号语法错误（代码此前从未编译通过）；
2. `VectorScan` 内 lambda 换私有成员 `TileLen()`（ccec 不支持 device 侧 lambda）；
3. 无参 `SyncAll()` 挂死 → workspace 内软件 barrier（见上节）；
4. host `ceil(needVecCoreNum/2)` 分核丢半 → `numBlocks = needVecCoreNum`；
5. `ReduceCoresRst` 掩码归约 507235 设备错误 → ≤80 float 标量折叠（见上节）。

### 用例覆盖（1200 CSV 行）

| 组 | 数量 | 覆盖内容 |
| --- | --- | --- |
| TC_L0 | 6 | n=1/小规模基线 |
| TC_SQ | 38 | incx==1 平方规模（短标量/向量管线/跨核分片） |
| TC_INC | 17 | incx>1 跨步路径 |
| TC_ED | 17 | 边界：n=0/负 n、result/x NULLPTR、跨步溢出、NULL handle |
| TC_FL | 14 | 浮点特殊值：NaN 首/中位（-Inf 哨兵钉死）、±Inf、全零并列 |
| TC_EX | 908 | 极端规模随机组合（大 n 跨核） |
| TC_PF | 200 | 性能门用例（默认按功能执行，`ICAMIN_PERF_BENCH=1` 启用计时门） |

### 平台注意事项（复现环境需知）

1. **ACL_HIFLOAT4 缺符号**：测试框架 `test/frame/csv_loader.h:491` 引用 `ACL_HIFLOAT4`，CANN 8.5.1/9.1.0 头文件均未定义（框架面向更新版本 CANN）。未改任何框架文件，以构建配置兜底：`-DCMAKE_CXX_FLAGS="-DACL_HIFLOAT4=42"`（42 为 aclDataType 现有枚举 41 之后的空位）。
2. cmake 需显式 `-DTEST_NAMES=icamin\;isamin`（未列入 TEST_NAMES 的算子无测试目标）。
3. **全仓 all 目标被 pre-existing 错误阻断**：`blas/lt_matmul/arch22/lt_matmul_fused_short_kernel.cpp`（commit 20551cf，调用 `SetSyncBaseAddr` 等仅存在于更新 CANN 的 API）编译失败，与本算子无关；测试按 per-op 目标构建验证。
4. 生命周期：所有 >1 分钟的构建/测试命令均在启动者单回合内阻塞守护至结束（后台子进程回合结束即被回收）。
5. 同族 `isamax` 的 arch22 host 仍按 `ceil(needVecCoreNum/2)` 启动且调用无参 `SyncAll()`，在本平台存在同样的丢半/挂死隐患，如需启用应参照本设计同步修复。

### 同族扩展：aclblasIsamin（fp32）

同一真机联调周期内，本仓完成同族 `aclblasIsamin`（fp32 输入、`|x|` 最小、NaN 首元素不钉死的种子-续扫语义）的 arch22 实现、测试三件与真机验证：**113/113 全过**（commit f166bb1，报告 `test/isamin/arch22/isamin_arch22_test_report.md`）。isamin 复用本设计的软件 barrier / 分核策略 / 标量核间合并三项平台适配，其与 icamin 的语义差异（幅值单元折叠、无 -Inf 哨兵、`(FLT_MAX, 0)` 种子）见需求背景一章对照表。

### 改动清单（c197f8e）

- `blas/iamin/arch22/icamin_kernel_impl.h`：语法修复、lambda 移除、软件 barrier（SoftBarrier）、每 block 1 AIV 约定、标量核间合并
- `blas/iamin/arch22/icamin_host.cpp`：numBlocks = needVecCoreNum、workspace 布局 [槽区|标志区] + memsetAsync
- `blas/iamin/arch22/icamin_kernel.cpp`：入口注释与启动约定同步
- `blas/iamin/arch22/icamin_tiling_data.h`：barrier 常量入共享 tiling 头
- `test/iamin/icamin/arch22/icamin_test.cpp`：quick-return 行空 span 传 1 元素缓冲（对齐 golden 非 null x 契约）
- `test/iamin/icamin/arch22/icamin_arch22_test_report.md`：证据链报告
