# 需求背景（required）

## 需求来源

2026 年 11 月社区任务：11月社区任务-aclblasIcamax算子开发（A2/A3）（任务书
`aclblasIcamax_AtlasA2_A3_task_doc.md`，昇腾社区任务中心）。目标代码仓为
[cann/ops-blas](https://gitcode.com/cann/ops-blas)，实现目录 `blas/iamax/arch22/`，
测试目录 `test/iamax/icamax/arch22/`。

## 背景介绍

### aclblasIcamax 算子开发

`aclblasIcamax` 是 BLAS Level-1 复数索引归约算子，与 cuBLAS `cublasIcamax` /
Netlib `icamax` 对标：在单精度复数（COMPLEX64）向量 x 中查找 1-范数口径"模"最大
元素的 1-based 索引，多个元素并列最大时取最小索引。

仓内现状：

- `include/cann_ops_blas.h` 已提供接口声明（与 950PR 共用，禁止定义私有平行接口）；
- `blas/iamax/arch22/` 已有同族实数接口 `aclblasIsamax` 的 Ascend C 实现，可复用
  框架；`blas/iamax/arch35/` 已有 950PR 版 `aclblasIcamax`，复数归约逻辑可参考；
- A2/A3（910B 系列，arch22）此前无 `aclblasIcamax` 实现：`test/iamax/icamax/CMakeLists.txt`
  仅在 arch35 分支注册 GTest，A2/A3 构建退回非 GTest runner。

### 需求描述

在 Atlas A2/A3（Ascend 910B 系列）上以 Ascend C kernel 直调方式实现
`aclblasIcamax` 句柄式 BLAS 接口，精度与 cblas（Netlib `icamax`）golden 精确相等
（bit-exact，输出为 1-based 整数索引），性能不高于 A100 cuBLAS 标杆耗时的 1.25 倍
（倍率 ≥ 0.8，测试设备 Atlas 800T A2 / 910B3），并补齐 A2/A3 的 GTest 测试工程。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 dav-2201（910B）上实现 `aclblasIcamax`：

```
result = 1 + argmax_i ( |Re(x[k])| + |Im(x[k])| )，k = 1 + (i-1)*incx，i = 1..n
```

- 复数"模"为 1-范数口径 `|Re| + |Im|`（Netlib SCABS1 语义，非欧氏模）；
- 输出 1-based 索引 int 标量；并列最大取最小索引（Netlib"严格大于才更新"语义）；
- `n = 0` 或 `incx ≤ 0`：quick return，result 写 0 且返回 SUCCESS（不读取 x）；
- `n < 0`、`result == nullptr`、正常路径 `x == nullptr`、`(n-1)*incx` 超出 int32
  可表示范围：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

## 需求拆解

1. 精度：1000 条任务用例（含 130 条全 NaN、Inf/并列/极端值/边界/负向）与 golden 精确相等；
2. 性能：200 条 `TC_PF_` 用例平均单次耗时 ≤ A100 cuBLAS 基线 × 1.25（倍率 ≥ 0.8），
   n=4M 输入（32 MB）基线 27.5 us，需 ≤ 34.4 us；
3. 工程代码：实现位于 `blas/iamax/arch22/`，测试位于 `test/iamax/icamax/arch22/`，
   并在 `test/iamax/icamax/CMakeLists.txt` arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`；
4. NaN 语义与 Netlib 严格一致：首元素 score 为 NaN 时结果停留为 1，其余 NaN 不胜出。

# 详细设计（required）

## 算子分析

### 数学公式

```
result = 1 + argmax_i score(i)，score(i) = |Re(x[k])| + |Im(x[k])|，k = 1+(i-1)*incx
```

归约到标量索引，无 broadcast、无动态 shape、无原地语义。

### 支持数据类型

输入 x：COMPLEX64（交错存储，n 个复数对应 2n 个 float32）；输出 result：INT32 标量。

### 支持形状

逻辑一维向量 [n]，物理长度 1+(n-1)*incx；incx ≥ 1 时正常计算，incx ≤ 0 走 quick return。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

`blas/iamax/arch22/icamax_host.cpp`：

1. 参数校验顺序对齐 arch35 通路与任务书 §2.4：handle 空指针 → `HANDLE_IS_NULLPTR`；
   `n < 0` → `INVALID_VALUE`；`result` 空指针 → `INVALID_VALUE`；`n == 0 || incx < 1`
   → `aclrtMemsetAsync` 向 device 侧 result 写 0（quick return，不触发 kernel、不读 x）；
   `x` 空指针 → `INVALID_VALUE`；int64 计算 `(n-1)*incx` 超 int32 范围 → `INVALID_VALUE`；
2. 核数规划：n ≤ 8192 走单核路径（无核间同步/汇总开销，覆盖全部小尺寸高频用例）；
   n > 8192 时 `useCoreNum = min(ceil(n/4096), min(GetAivCoreCount(), 40))`，保证每核
   至少一个完整 tile；`perCoreN/remainder` 均分，前 remainder 个核各多 1 个元素；
3. 任务下发：tiling 参数（n/incx/useCoreNum/perCoreN/remainder）以 kernel 入参按值
   传递，无 GM tiling 缓冲区，host 侧无 `aclrtMalloc`/`aclrtMemcpy`，单次调用零额外
   Device 内存分配；核间中间结果复用 `aclblasCreate` 预分配的 handle 工作区
   （固定 1280 字节槽位：40 核 × 32 B 对齐 × [value, index]）；
4. 块数：实测 910B3 上 `__vector__` 内核的 `GetBlockIdx()` 即 block 序号
   （1 block = 1 AIV，GetAivCoreCount()=40），故 `numBlocks = useCoreNum`。

#### 3.2.2 kernel 侧设计

`blas/iamax/arch22/icamax_kernel.cpp`（`__global__ __aicore__ __vector__`，单 kernel）：

1. 分核与快路径：核 c 处理逻辑区间 `[start, start+count)`；`useCoreNum == 1` 时
   直接写结果，不参与核间同步与汇总（小尺寸高频繁路径）；
2. incx == 1 向量快路径：按 tile（10240 复数 = 80 KB）双缓冲预取搬运——
   先下发 tile0 的 `DataCopy`，循环内先预取下一 tile 再 `DeQue`/计算当前 tile，
   使 MTE2 搬运与 Vector 计算充分重叠（n=4M 实测 33.5 us → 28.3 us）；每 tile 执行
   `Abs`（2n floats）→ `PairReduceSum`（交错 [re,im] 相邻配对求和，直接得到
   1-范数 score 序列；repeatTime 超 255 时按 isamax 同款分批）→ 计数掩码
   `WholeReduceMax`（VALUE_INDEX 序，硬件返回 tile 内最大值与首个最大值下标）→
   `GetReduceMaxMinCount` 读出 (val, idx)，与当前最优做"严格大于 + 并列取更小全局索引"
   归并；不足 64 复数的尾部与对齐余量按标量精确扫描；
3. incx > 1：标量精确扫描（`GlobalTensor.GetValue` 按 `i*incx` 步长取数，uint64 地址
   防溢出），正确性优先（性能用例全为 incx=1）；
4. NaN 语义：核 0 先以标量读 x[0] 判定首元素 score 是否为 NaN（FIRST_NAN，结果强制为 1）；
   向量路径中任一 tile 的 WholeReduceMax 结果为 NaN 时，从该 tile 起对本核剩余区间
   改为"跳过 NaN + 严格大于"的精确标量重扫（本地回退，硬件 NaN 行为不可移植场景不进入结果），
   多余启动的核同样参与唯一一次 `SyncAll` 防止 barrier 悬挂；
5. 核间汇总：各核将 (value, index) 写入 handle 工作区自有槽位（`S_MTE3` 事件同步 +
   `DataCopyPad`），`SyncAll` 后核 0 一次性 `DataCopy` 将 40 核槽位搬入 UB 再扫描
   （逐元素标量读 GM 过慢，实测仅读 80 个字即增 ~16 us）——严格大于，并列时保留更小
   全局索引；全部无效（全 NaN 经回退）按 Netlib 语义输出 1；
6. 输出：核 0 直接向 result GM 写 1-based 索引（首元素 NaN 时恒为 1）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3，AIV 40） | √ |
| Atlas A2 训练系列 / Atlas A3 训练系列（arch22 通路） | √ |

## 算子约束限制

- incx ≤ 0 与 n = 0 为合法 quick return（不反向遍历，对齐 Netlib 1993+ 语义）；
- 复数"模"为 1-范数 |Re| + |Im|，禁止欧氏模口径；
- (n-1)×incx 超出 int32 可表示范围返回 `ACLBLAS_STATUS_INVALID_VALUE`（issue #745）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | NPU 输出 1-based 整数索引与 cblas（Netlib `icamax`，0-based + 1）golden 精确相等（bit-exact）；1000 条精度/边界用例 + 2 条补充测试（空 handle/空输出地址）全部通过 | 任务书 §3.2 |
| 性能标准 | 平均单次耗时 ≤ A100 cuBLAS 标杆 × 1.25（倍率 ≥ 0.8）；msprof 同口径下 warmup 5 + aclrtEvent 连续 15 次取均值；200 条 TC_PF 用例全部通过，实测最低倍率 ≈0.87 | 任务书 §3.3 |
| 内存标准 | 不涉及（任务书 §3.4）；实测单次调用零额外 Device 分配，单 AIV UB 峰值 ≈161.3 KiB < 192 KiB | 任务书 §3.4 |

## 兼容性分析

新增算子通路，不改动静载接口签名（声明沿用 `include/cann_ops_blas.h` 既有声明）；
A3/A2 通过 SOC→arch 目录映射（`arch22`）自动启用，950PR（arch35）与 310P 通路不受影响；
测试 CSV 列格式对齐仓内 `test/isamax/isamax_param.h` 惯例并扩展 result 列 NULLPTR 标记。
