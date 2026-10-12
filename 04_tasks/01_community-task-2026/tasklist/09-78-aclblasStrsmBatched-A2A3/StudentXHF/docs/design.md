# aclblasStrsmBatched 算子设计文档

# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务 aclblasStrsmBatched 算子开发（A2/A3）。目标是在 Ascend C 上提供与任务书约定一致的单精度实数批量三角矩阵求解接口。

## 背景介绍

`aclblasStrsmBatched` 在 Atlas A2/A3 的 `arch22` 架构上实现单精度实数批量三角求解。每个 batch 独立求解：

- `side=LEFT`：`op(A[i]) * C[i] = alpha * B[i]`；
- `side=RIGHT`：`C[i] * op(A[i]) = alpha * B[i]`。

矩阵使用列主序，`Aarray/Barray/Carray` 是 Device 侧指针表，`alpha` 是 Host 标量。接口采用 B 只读、C 独立输出的离席语义。实现覆盖 LEFT/RIGHT、UPPER/LOWER、N/T、UNIT/NON_UNIT 全组合，不支持实数 `OP_C`。

最终实现源码位于 `ops-blas`：

- `blas/trsmbatched/arch22/`：Host 校验、AIV 通用路径、AIC+AIV MIX Cube 路径及 launcher；
- `test/trsmbatched/strsmbatched/arch22/`：CSV 驱动的真机测试、Netlib CBLAS golden、官方及补充用例；
- `include/cann_ops_blas.h`：公共接口声明。

设计对应[代码 PR #529](https://gitcode.com/cann/ops-blas/pull/529) 的 `feature/aclblasStrsmBatched-a2a3-studentxhf` 分支，目标环境为 CANN 9.1.0、Atlas 800I A2（910B3）；A3 复用同一 `arch22` 实现。最终代码 HEAD 及其与真机测试证据的对应关系以验收记录为准。

# 需求分析（required）

## 需求描述

实现列主序、单精度实数、离席输出的 batched TRSM，在 A2/A3 上支持左右乘、上下三角、转置和单位对角组合，并按任务书覆盖参数校验、数值精度和性能验证。

## 需求拆解

1. 为所有合法枚举和尺寸提供确定的计算结果，非法参数返回约定状态码。
2. 保持 A/B 只读、C 独立输出、padding 和 guard 不变，并处理 `alpha==0` 与 no-op。
3. 用 AIV 通用路径覆盖合法形状，用 AIC+AIV MIX Cube 路径加速适用形状。
4. 通过官方精度、补充精度、功能性能、profiling 与内存证据核验实现。

## 接口与功能约束

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* const* Aarray,
    int lda,
    const float* const* Barray,
    int ldb,
    float* const* Carray,
    int ldc,
    int batchCount);
```

### 2.1 参数与形状

| 参数 | 约束 |
| --- | --- |
| `side` | `LEFT` 或 `RIGHT` |
| `uplo` | `UPPER` 或 `LOWER` |
| `trans` | `OP_N` 或 `OP_T`；`OP_C` 返回非法参数 |
| `diag` | `UNIT` 或 `NON_UNIT` |
| `m,n,batchCount` | 非负；任一为 0 时合法 no-op |
| `lda` | LEFT 时至少为 `m`，RIGHT 时至少为 `n` |
| `ldb,ldc` | 至少为 `m` |
| A | LEFT 时每批 `m*m`，RIGHT 时每批 `n*n` |
| B、C | 每批 `m*n`，B 只读，C 独立输出 |

所有地址跨度使用 64 位计算并检查溢出。实现只承诺 `lda/ldb/ldc` 表达的非紧凑列主序布局，不支持额外任意 stride、broadcast 或 view。

### 2.2 特殊语义

- no-op 在读取 `alpha` 或 Device 指针表之前返回成功；
- `alpha==0` 不读取 A/B，专用 copy kernel 只把 C 的活跃区写为 `+0`；
- `diag=UNIT` 不读取 A 的存储对角；
- B 和 A 始终只读，只写 C 活跃区，C padding 与 guard 不变；
- C 与 A/B 或其他 C batch 的地址跨度不能重叠。

# 详细设计（required）

## 算子分析

### 数学公式与数据类型

对每批矩阵，LEFT 求解 `op(A[i]) * C[i] = alpha * B[i]`，RIGHT 求解 `C[i] * op(A[i]) = alpha * B[i]`。输入输出元素均为 `float32`，矩阵为列主序。

### 支持形状

`m,n,batchCount` 为非负整数，A 为 `order × order`，B/C 为 `m × n`；leading dimension 可以大于逻辑行数。

## 算子实现

### 总体架构

```text
公开 API
  |
  +-- Host 参数/跨度/指针表/重叠校验
  |
  +-- alpha==0 ----------------------> copy-to-C 清零 kernel
  |
  +-- NON_UNIT && order>=256 && rhs>=8
  |       |
  |       +-- Cube plan 可建立 ------> copy-to-C + MIX_1_2 blocked TRSM
  |       |
  |       +-- plan 不可建立 --------> AIV 通用路径
  |
  +-- 其余合法形状 -----------------> AIV resident RHS panel
                                          |
                                          +-- order>4096：GM scalar fallback
```

公开默认分派只依赖参数和设备能力。`STRSM_KERNEL=reference|vector|cube` 与 `STRSM_BLOCKS` 仅用于开发诊断和专项复验，不属于公开 API 语义。

### Host/Tiling 设计

### 4.1 校验顺序

Host 固定按以下顺序执行：

1. 校验 handle；
2. 校验枚举、负尺寸；
3. 处理 no-op；
4. 校验 leading dimension、`alpha`、`Carray`；
5. 读取 Host `alpha`；非零时再校验 `Aarray/Barray`；
6. 计算 A/B/C 保守地址跨度并检查整数溢出；
7. 从 handle 获取 stream；
8. 将 Device 指针表异步拷到 pinned Host buffer，并在同一 stream 同步；
9. 校验元素地址、输出间重叠和输入输出重叠；
10. 选择 kernel、规划 workspace 并 launch。

指针表 metadata 必须读回 Host 才能发现数组元素中的空指针和非法重叠，因此接口在 metadata 阶段同步 stream；kernel 计算本身仍按 handle 所绑定 stream 提交。

### 4.2 地址跨度和别名

矩阵跨度按下式计算：

```text
bytes = sizeof(float) * ((cols - 1) * ld + rows)
range = [device_pointer, device_pointer + bytes)
```

输出区间排序后必须两两不交；A/B 输入允许只读复用，但所有输入区间均不得与任何 C 区间相交。若 scalar fallback 的相邻输出落在同一 64-byte cache line，Host 将其降为单 block，避免不同核 DCache 写回互相覆盖。

### 4.3 Workspace 规划

Cube 路径使用 handle 管理的默认 workspace。Host 对所有加法、乘法和 512-byte 对齐操作做溢出检查，并限制总量不超过 `ACLBLAS_MAX_WORKSPACE_SIZE`。规划失败不改变功能语义，直接回退 AIV 路径。

### Kernel 设计：AIV 通用路径

### 5.1 Resident RHS panel

当三角阶数 `order<=4096` 时，AIV 将一组 RHS 常驻 UB：

- `order = m`（LEFT）或 `n`（RIGHT）；
- 每个 panel 的 RHS 数按 `ChooseTile(order)` 选择，最大 64，并向 8 对齐；
- panel 上限 32768 个 float（128 KiB）；
- 另有 A 系数、搬运 staging、列状态、广播和临时 buffer。

每个 AIV task 独占一个 `(batch, rhs-panel)`，block 以 grid-stride 方式处理后续 task，不存在跨核写同一 C 元素。

RIGHT 的 RHS lane 在列主序下可直接按连续行段搬运；LEFT 需要把列主序 RHS pack 为 `[order,pitch]` 的 panel，求解后再 unpack，只写 C 活跃区。`DataCopyPad` 的块数上限通过分段搬运处理，覆盖 4096 阶边界。

### 5.2 解算顺序

四类组合遵循三角依赖顺序：

| 分支 | pivot 顺序 | 主要操作 |
| --- | --- | --- |
| LEFT/N/LOWER | 升序 | pivot 除法后更新后续行 |
| LEFT/N/UPPER | 降序 | pivot 除法后更新前序行 |
| LEFT/T/UPPER | 升序 | 先累减已解依赖，再除对角 |
| LEFT/T/LOWER | 降序 | 先累减已解依赖，再除对角 |
| RIGHT/N/UPPER | 升序 | 当前列减去先前列贡献 |
| RIGHT/N/LOWER | 降序 | 当前列减去后续列贡献 |
| RIGHT/T/UPPER | 降序 | pivot 后更新前序列 |
| RIGHT/T/LOWER | 升序 | pivot 后更新后续列 |

UNIT 路径跳过对角加载。对可能出现 NaN/Inf 的系数，vector 优化先以列状态证明整列满足安全条件；只有证明成立的连续区间才做批量向量更新，其他数据保持与标量参考一致的逐项传播语义。

### 5.3 GM scalar fallback

当 `order>4096` 或强制 reference 时，kernel 直接从 GM 顺序求解。该路径无大块 Device workspace，覆盖超过 UB resident 上限的合法 shape，并作为问题定位基准。它保证功能泛化，不承担大 shape 性能目标。

### Kernel 设计：AIC+AIV MIX Cube 路径

### 6.1 进入条件与规范化

默认 Cube 条件为 `diag=NON_UNIT && order>=256 && rhs>=8`。Host 还会检查：

- 对齐后 `order/rhs` 不超过 4096；
- 静态 Matmul 的 grouped K 不超过 1024；
- vector/Cube core 数可以从 Runtime 正确读取；
- workspace 可安全计算并分配。

列主序矩阵的同一字节可解释为转置后的行主序视图。Host 将方程规范为 MIX kernel 的 canonical 形式：交换 LEFT/RIGHT、翻转 UPPER/LOWER，并保持 N/T 的代数含义。这样无需先生成完整转置副本。

### 6.2 Blocked TRSM

阶数选择 panel 宽度 `nb`：

- `order>=4096`：64；
- `order>=256`：32；
- 其余：16。

每个 panel 分两步：

1. AIV 将对角块载入 UB，执行 panel TRSV，结果写回 C，并将 `-X` 写入 Xneg workspace；
2. AIC 使用 Ascend C Matmul 完成剩余块更新 `B_next += A_offdiag * (-X)`。

AIV 与 AIC 通过 cross-core flag 协调 TRSV 与 GEMM 的生产/消费。panel 数超过 16 时按组复用双缓冲 slot；Host 按 `min(numPanels,32)` 规划 Xneg 上界，与 device 的 slot 计算保持一致。

### 6.3 batch 和列拆分

- batch 数充足时，一个物理 `MIX_1_2` block 处理两批，并以 grid-stride 继续处理后续 batch pair；
- batch 小而 `order>=768` 时，按可用 Cube core 将 RHS 列拆为 1～8 份，多个 MIX block 协作处理同一 batch 的互不相交列区间；
- 非 8 对齐 shape 在 GEMM workspace 使用前由 `aclrtMemsetAsync` 清零 padding；对齐 shape 不执行该 memset。

正式 profiler 预期 kernel 序列为：

- vector：每次 invocation 一个 `strsm_batched_vector`；
- 对齐 Cube：`strsm_batched_copy_to_c` 后接 `strsm_batched_cube_mix12`；
- 非对齐 Cube：先有可归属的 memset，再执行 copy 和 mix12。

### 数值精度设计

测试端以 Netlib CBLAS `strsm` 逐 batch 生成 golden。有限元素采用混合容差：

```text
abs(actual-golden) <= 2^-13 + 2^-13 * abs(golden)
matched_ratio >= 0.99
max_abs_error <= 1e-2
```

同时要求 `nonfinite_mismatch=0`。同位置 NaN 与 NaN、同号 Inf 与 Inf 只作为非有限掩码匹配；不能用全非有限输出掩盖有限数值错误。

测试还独立检查：

- API 状态码；
- A/B 输入未修改；
- C padding 与 guard 未修改；
- UNIT 对角及未使用三角区即使含 NaN 也不被读取；
- `alpha==0` 时毒化 A/B 仍能正确清零 C；
- 地址别名、非法枚举、负维度、非法 LD 和空指针按契约返回。

### 性能设计与验收方法

性能用例固定为 `TC_PF_1001..TC_PF_1200` 共 200 条，原始 GPU baseline 文件受 SHA-256 锁定。每个 case 独立执行：

1. 1 次 sanity；
2. 5 次 warmup；
3. 20 次有效测量；
4. `msprof op` 采集所有相关 kernel 的 `Task Duration(us)`；
5. 多 kernel invocation 先求和，再对 20 个 invocation 求平均。

验收阈值为：

```text
NPU mean kernel time <= min(GPU baseline / 0.8, task limit if defined)
```

先选择三条代表用例做失败封闭预检：

- `TC_PF_1003`：vector；
- `TC_PF_1001`：8 对齐 Cube；
- `TC_PF_1061`：需要 memset 的非 8 对齐 Cube。

只有三条都能可靠解析、kernel 序列匹配且性能通过，才启动完整 200 条。

## 支持硬件

| 支持的芯片版本 | 实现路径 |
| --- | --- |
| Atlas 800I A2（910B3） | `arch22`，已有真机功能与精度证据 |
| Atlas A3 | 共用 `arch22`，产品级回归证据待补 |

## 算子约束限制

仅支持 `float32`、列主序和独立 C 输出；不支持 `OP_C`、任意 stride、广播或输入输出地址重叠。超出 Cube 静态规划范围的合法输入回退 AIV 路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 官方精度 1000 条、补充精度 286 条及非有限掩码核验 | 社区任务与本设计测试方案 |
| 功能性能标准 | PF1001～PF1200 的 NPU 功能运行全部 PASS | 社区任务用例 |
| 计时性能标准 | 逐例 profiler 平均 kernel 时间满足下文阈值 | 社区任务性能要求 |

## 兼容性分析

该算子为新增接口；A2 与 A3 共用 `arch22` 实现。已有 A2 测试不能替代 A3 产品回归。

## 内存与资源

任务未定义 HBM 容量阈值，因此不生成虚假的“容量 PASS”。验收使用 `aclrtGetMemInfo(ACL_HBM_MEM)` 进行真机全局采样：

- 先采至少 20 个基线样本；
- 运行预测 footprint 最大的 `TC_PF_1081`；
- 记录 HBM 绝对峰值、基线 min/median/max 和相对基线中位数增量；
- 采样期间拒绝并发 profiler 或其他测试进程。

结果状态为 `MEASURED`，并同时要求 runner 为 NPU PASS、子进程返回 0、输入/padding 检查通过。

## 测试覆盖与复现

| 类别 | 数量/方法 | 门禁 |
| --- | --- | --- |
| 官方精度 | 原始 CSV 中 1000 条 | 1000/1000 NPU PASS |
| 补充精度 | 独立 286 条 | 286/286 NPU PASS |
| 标准入口 | `build.sh --ops=strsmbatched --run --soc=ascend910b3 --device=0` | 返回 0 |
| 性能预检 | vector、对齐 Cube、非对齐 Cube 各 1 条 | 三条均通过 |
| 完整性能 | 200 条逐 case msprof | `PASS_200` |
| HBM | `TC_PF_1081` 独立采样 | `MEASURED` 且证据完整 |

精度、性能和 HBM 均绑定同一源码提交、动态库 SHA-256 与测试二进制 SHA-256。任何缺失、CPU backend、SKIP、重复 case、未知时间单位、kernel 数量/顺序不符或证据摘要不一致均按失败处理。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `strsmbatched_host.cpp` | API 校验、指针表快照、别名检查和分派 |
| `strsmbatched_kernel.cpp` | AIV resident panel 与 scalar fallback |
| `strsmbatched_cube_plan.cpp` | Cube tiling、core 分配和 workspace 规划 |
| `strsmbatched_copy_kernel.cpp` | B→C/alpha0、tiling data 序列化 |
| `strsmbatched_cube_kernel.cpp` | MIX kernel 入口 |
| `strsmbatched_cube_aiv*.h` | AIV panel TRSV、规范化和搬运 |
| `strsmbatched_cube_aic.h` | AIC Matmul trailing update |
| `strsmbatched_launch.*` | Ascend C kernel launch 封装 |
| `test_support.*` | CSV 解析、fixture、CBLAS golden 和严格校验 |

## 已知边界

- Host 为检查 Device 指针表内容会在 kernel 提交前同步一次 stream；
- Cube 静态 Matmul 只覆盖规划范围，超出范围的合法输入回退 AIV；
- GM scalar 路径用于大阶数泛化，性能不作为其目标；
- A2 与 A3 共用 `arch22`，正式合入仍应在平台允许时补充对应产品回归证据。
