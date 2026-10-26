# 需求背景（required）

## 需求来源

- 任务：2026 年 9 月 aclblasCrot 算子开发（Ascend 950PR），竞赛仓任务编号 09-23。
- 团队及提交账号：HuangRZzzz。
- 依据：任务方《aclblasCrot 950 算子开发任务书》（aclblasCrot_Atlas950PR_task_doc.md）、`test_cases/`（CSV、GPU 基线、测试指导）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，硬件 Ascend 950PR，CANN 9.1.0。
- 文档版本：2026-10-19。本文档为**已完成真机实现与自测**的设计说明：架构、关键决策与全部性能/精度数据均来自实测，自测结论不代表官方评审或验收状态。

## 背景介绍

### aclblasCrot 功能与工程现状

Crot 对两个复数（COMPLEX64）向量施加 Givens 平面旋转：c 为实数（旋转余弦）、s 为复数（旋转正弦），x、y 原地更新，y 的更新使用 `conjg(s)`（对齐 netlib/LAPACK `crot`）。属于 BLAS Level 1 逐元素运算，无矩阵、无 batch 参数。

工程参考提交为 ops-blas 的 rot 家族：公共头文件已有 `aclblasSrot`（实数，arch35）与 `aclblasCsrot`（复数向量 + 实 c/实 s，arch22），**均无 aclblasCrot**；`blas/rot/arch35/` 已有实数 Srot 的 Host/Tiling/Kernel 组织。本次新增复数版本 `aclblasCrot`，复用句柄/stream/构建注册/测试组织，独立实现复数乘加、共轭与步长语义；Csrot 的实正弦语义不能覆盖 crot 的复正弦 + 共轭。

实现方式为 Ascend C kernel 直调：`<<<>>>` 启动 + 连续路径 regbase SIMD（`__simd_vf__` 寄存器级计算）、跨步路径 SIMT VF，不涉及 TBE/ACLNN/算子注册 JSON。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增公共声明（整数维参数保持 `int`，类型复用 `cann_ops_blas_common.h`）：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy,
    const float* c, const aclblasComplex* s);
```

| 参数 | 位置与方向 | 类型/布局 | 约束与含义 |
| --- | --- | --- | --- |
| handle | Host 输入 | aclblasHandle_t | 有效句柄，使用绑定 stream；异步语义 |
| n | Host 输入 | int | n ≤ 0 为合法空操作（SUCCESS，不改 x/y） |
| x | Device 输入/输出 | COMPLEX64 步长向量 | 原地更新为 c·x + s·y；物理跨度 1+(n−1)·\|incx\| |
| incx | Host 输入 | int | 可正/负/零；负步长自尾端起算；零步长按参考语义复用同一元素 |
| y | Device 输入/输出 | COMPLEX64 步长向量 | 原地更新为 c·y − conjg(s)·x（使用旋转前原始 x） |
| incy | Host 输入 | int | 同 incx |
| c | Host 输入 | const float* | 旋转余弦标量指针；host/device 内存均可（运行时自动判定）；n>0 时不可为 nullptr |
| s | Host 输入 | const aclblasComplex* | 旋转正弦标量指针；host/device 内存均可（运行时自动判定）；n>0 时不可为 nullptr |

## 需求拆解

1. 语义对齐 netlib/LAPACK `crot`：x 与 y 的更新公式、y 更新的 conjg(s) 与"原始 x"要求。
2. 支持 incx/incy 任意正/负/零步长；负步长尾端起算、零步长串行语义；非连续访存正确。
3. 边界与负向：n≤0 no-op（优先级低于空句柄检查）；n>0 时 x/y/c/s 空指针返回 INVALID_VALUE；(n−1)·\|inc\| 超出 int32 返回 INVALID_VALUE。
4. c/s 为标量指针参数，可为 Host 或 Device 内存，由算子运行时通过 `aclrtPointerGetAttributes` 自动判定且相互独立。
5. 以任务书口径实现 golden（参考 BLAS 无 crot 例程，按任务书 §3.2 注 1 在测试工程内实现），实部/虚部分别核对精度标准。
6. 覆盖附带 1000 条精度、200 条性能用例；性能按"warmup 5 次 + >50 次有效采样取平均"采集并全量达标。
7. 按目标仓规范交付：公共头文件声明、README 接口章节、测试注册（CSV 驱动）、精度/性能/复现证据。

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[i] = c · x[i] + s · y[i]
y'[i] = c · y[i] − conjg(s) · x[i]        （使用旋转前的原始 x[i]）
```

令 s = sr + i·si，按实部/虚部展开（kernel 以 2×FP32 视角计算）：

```
x'.re = (c·x.re + sr·y.re) − si·y.im
x'.im = (c·x.im + sr·y.im) + si·y.re
y'.re = (c·y.re − sr·x.re) − si·x.im
y'.im = (c·y.im − sr·x.im) + si·x.re
```

### 支持数据类型与形状

仅 COMPLEX64（内部按 2×FP32）。附带用例覆盖 n≤2^20（精度）与 n≤2^22（性能）；更大合法 n 由多核均分 + UB 分片处理，受设备内存与地址可表示范围约束。无广播；n 为运行时标量。

### 地址计算

以复数元素为单位、64 位中间量：

```text
x 物理起点 = incx < 0 ? int64(n−1)·(−int64(incx)) : 0        // 负步长由 kernel 内部按 (n−1−i)·|inc| 处理
第 i 逻辑元素物理下标 = incx ≥ 0 ? i·incx : (n−1−i)·|incx|
连续路径 GM/UB 以 float 视角寻址：复数下标 × 2（元素 8B = 2×FP32）
```

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

- **双路径分派**：`incx == 1 && incy == 1` 走连续路径（regbase SIMD 内核 `crot_aiv_kernel`）；其余任意组合（含正非 1、负、零及混合）走跨步路径（SIMT 内核 `crot_simt_kernel`）。
- **分核与并行策略**：连续路径 `numBlocks = min(n, AIV 核数)`，元素按 `perCoreN = n/numBlocks` 均分、前 `remainder` 核各多 1 个（无对齐取整，DataCopyPad 仅需 1 字节对齐）；跨步路径 `numBlocks = min(ceil(n/128), AIV 核数)`，`nthreads = clamp(ceilDiv(n, numBlocks), 128, 2048)`（小 n 不再空跑满线程）。
- **tiling 与 UB 预算**：连续路径 tile 取 3840 个复数（寄存器块 64 的整数倍，尾块零填充对齐），4 个队列 × BUFFER_NUM=2 深度共 8 个 UB 缓冲，占用 245,760B ≤ 248KiB。
- **c/s 标量来源**：host 指针 → host 解引用一次、经 tiling 常量广播；device 指针 → 直接作为 GM_ADDR 传入内核读取（无 D2H、无 stream 同步）；二者独立判定。
- **溢出防护**：(n−1)·\|inc\| 在 host 以 int64 计算并按**幅度**检查（> INT32_MAX 即拒绝，含 −2^31 负步长），避免内核地址偏移溢出；kernel 内实际寻址使用 uint64。

#### 3.2.2 Kernel 侧设计

- **连续路径（regbase SIMD，带宽导向）**：复数按 float 视角处理，`LoadDist::DIST_DINTLV_B32` 装载时去交错、`StoreDist::DIST_INTLV_B32` 回写时重组，复数四则分解为 4 组实部/虚部乘加。全部使用 Muls/Add/Sub **每步舍入**（不使用 FMA 融合）——与 CPU golden 的两次舍入序列逐位对齐。每个 UB tile 一次 `asc_vf_call` 处理 64 复数的整寄存器块序列；副本尾块的计算垃圾经"精确字节数回写"丢弃（DataCopyPad 只做 32B 对齐补零，规避超限 padding 的 MTE 约束）。
- **原地安全**：Compute 阶段全程持有输入队列（新值先写独立 VECOUT 缓冲），x/y 的原始值在两者更新完成前不被覆盖，确保 y' 使用旋转前的 x。
- **跨步路径（SIMT）**：grid-stride 循环，正/负步长分别按 `i·inc` 与 `(n−1−i)·|inc|` 寻址（uint64 偏移）；**零步长**按 netlib 串行累积语义回退为 `block0.thread0` 单线程串行执行（多线程会产生竞争），与参考逐位一致。
- **位级一致策略**：SIMT 路径每个乘法经 `volatile` 临时量强制单步舍入（禁用编译器 FMA 收缩与重结合）——在 c=1e10 级极端标量且结果相消的场景下，1 ULP 的中间量差异会超出 32×ULP 逐元限，位级一致是唯一稳健解。

### 性能优化方案

| 项 | 做法 | 实测 |
| --- | --- | --- |
| 寄存器级去交错（连续路径） | DINTLV/INTLV 载荷变换，复数计算零额外 UB 搬运 | n=4M kernel 46.10us（门限 119.52us） |
| 双缓冲流水 | 4 队列 × 2 深度，MTE2/Vector/MTE3 跨 tile 重叠 | 稳定带宽达成，全 200 用例达标 |
| tile 规模选择 | 3840 复数/块（UB 约束内最大化），64 复数整寄存器块 | 见 3.1 性能表 |
| 每步舍入（非 FMA） | 精度优先的算术序列，与 golden 位级对齐 | 1000 用例 ×4 分量 max_err=0 |
| SIMT 线程数自适应 | 小 n 仅启动所需线程（128~2048 可调） | 小规格用例无满线程空转 |
| 已记录决策 | 尾块整块计算 + 精确字节回写（避免超限 padding 的 MTE 异常）；步长溢出按幅度拒绝 | 见自测报告归档 |

### 工程组织与构建安装

新增/修改（目标仓）：`blas/rot/arch35/crot_{host.cpp,kernel.cpp,kernel.h,tiling_data.h}`、`blas/rot/README.md`（接口章节，含支持表与调用示例）、`include/cann_ops_blas.h`（声明）、`test/rot/crot/**`（CSV 驱动测试 + golden + NPU wrapper + CMakeLists）。

```bash
bash build.sh --soc=ascend950 --ops=crot
./build/test/rot/crot/crot_test --gtest_filter="-*TC_PF*"      # 全量精度
./build/test/rot/crot/crot_test --gtest_filter="*TC_PF*"       # 全量性能（打印单次平均）
msprof op --application="./build/test/rot/crot/crot_test --gtest_filter=*TC_PF_1001*" --output=./prof_1001
```

## 支持硬件

- Ascend 950PR / Ascend 950DT（arch35 / DAV_3510），AIV-only（`KERNEL_TYPE_AIV_ONLY`），CANN 9.1.0。

## 算子约束限制

- n ≤ 0 为合法 no-op；n > 0 时 x/y/c/s 不可为 nullptr；handle 为 nullptr 返回 HANDLE_IS_NULLPTR（优先于 no-op）。
- (n−1)·\|incx\|（或 incy）超出 int32 表示范围返回 INVALID_VALUE。
- 零步长为单线程串行回退（语义正确性优先，性能显著低于连续路径）；恒等旋转（c=1 且 s=0）不做短路，正常执行旋转公式。
- 仅原地更新 x/y，不触碰步长间隙，无 workspace、无额外 Device 内存分配。

# 可维可测分析

## 精度标准/性能标准

- 精度：任务书 §3.2（complex64 实/虚部按 FLOAT32：rtol=atol=2⁻¹³、matched≥0.99、maxabs≤max(1e-2, 32×ULP)），双口径另录 MERE≤2⁻¹³、MARE≤10×阈值；golden 按 netlib/LAPACK crot 语义在测试工程内实现。
- 性能：三条任务书 §3.3 上限与实测（msprof kernel Task Duration 平均，110 实例 warmup 5，单位 us）：

| 用例 | 形状 | 上限 | 实测 | 余量 |
| --- | --- | --- | --- | --- |
| TC_PF_1001 | n=1048576 | 21.96 | 13.41 | 39% |
| TC_PF_1002 | n=2097152 | 43.70 | 23.41 | 46% |
| TC_PF_1003 | n=4194304 | 119.52 | 46.10 | 61% |

## 测试输入与覆盖

- 官方精度用例 **1000/1000 通过（exit=0）**，4×1000 个分量与 golden **全部逐位一致**（max_abs_error=0、MERE=0、MARE=0）；覆盖尺寸池（0/1/小质数/2 的幂±1/非对齐/2^20）、步长全组合（±1/±2/±3/0）、标量特殊值（单位/90°/45°/复数 s/纯虚 s/负 c/1e10）、填充（均匀/全零/交替/极端/Inf/NaN）、c/s 指针位置（host/device/mixed）、边界与负向。
- 官方性能用例 **200/200 达标**（对照官方 gpu_baseline：41 条临界用例附 msprof kernel 精确证据，其余以测试工程保守上界即达标）。
- 另新增 4 组非 CSV 负向用例（NullHandle、快速返回优先级、步长溢出边界），随测试工程提交。

## 配套用例核对及修正计划

- 官方 `verify_performance.py` 的 verdict 列以「GTest 用例整数毫秒墙钟」对比微秒级单次基线，口径错配（README 亦注明该值为保守上界）；自测中提供修正版脚本（优先按 msprof kernel 单次平均判定）与 diff 留档，判定以任务书 §3.3 单次平均指标为准。

## 计时与复现

- 环境：容器内 CANN 9.1.0，Ascend 950PR；`bash build.sh --soc=ascend950 --ops=crot` 后直接运行 gtest 过滤集。
- 性能口径：测试工程 10 次预热 + 100 次有效采样取平均（保守上界）；精确 kernel 耗时用 `msprof op` 采集 `OpBasicInfo.csv` 的 Task Duration 平均（warmup 5 + 3~110 实例），同一窗口内完成对比。

## 交付与可维护性

- 与 rot 家族同构：Host/Tiling/双路径内核分层（连续 regbase SIMD + 跨步 SIMT），tiling 结构与 srot 一致（tilingKey 分派、perCoreN/remainder/nthreads），便于维护者对照阅读；测试工程与 srot 同构（CMakeLists/param/golden/wrapper/CSV）。

## 兼容性分析

- 仅新增接口与文件，不改动既有算子行为；`include/cann_ops_blas.h` 为增量声明；测试目录独立注册，不影响其它用例。

## 参考资料

- 《aclblasCrot 950 算子开发任务书》及 `test_cases/`（CSV、GPU 基线、测试指导）
- netlib/LAPACK `crot`（参考实现，含 conjg(s) 语义）
- ops-blas 仓库 rot 家族（Srot/Csrot）实现与贡献指南
