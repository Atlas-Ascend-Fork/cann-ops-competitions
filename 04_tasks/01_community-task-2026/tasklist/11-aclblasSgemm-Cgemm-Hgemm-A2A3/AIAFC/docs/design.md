# aclblasSgemm、aclblasCgemm、aclblasHgemm 算子设计（Atlas A2/A3）

## 1. 需求背景

本设计对应 11 月 CANN 社区 GEMM 三算子任务 `4b45f7b048224317a2ec62eab1e8df52`。在 Atlas A2/A3 上，以 Ascend C 实现句柄式接口 `aclblasSgemm`（FP32）、`aclblasCgemm`（COMPLEX64）和 `aclblasHgemm`（FP16），对齐 cuBLAS 对应接口的列主序矩阵乘语义。实现使用 `ops-blas` 的 `arch22` 构建与直调链，正式性能测试环境为 Atlas 800T A2（910B3）和 CANN 9.1.0。

[官方任务页](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52) · [任务书及自验附件](https://www.hiascend.com/p/resource/202611/65f1c7ed628f416ebbb68e9dc9595ce0.zip) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/92)

任务书以 `arch22` 仅有 Sgemm 的版本为起点。当前上游已有 Sgemm 和 Cgemm 的 `arch22` 入口。Sgemm 可复用 FP32 Cube 和列主序 Tiling 框架，Cgemm 可复用四实积及交错/分量布局的处理逻辑，但现有 Cgemm 从 Device 读标量，且在判断 `alpha=0` 之前拒绝空 A/B。新方案复用既有符号并校正这些行为。Hgemm 单矩阵入口新增声明和实现，参考同仓 `arch35` 半精度路径的类型、缩放和写回规则。

[ops-blas 固定源码 `55ef791b`](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm) · [Cgemm 当前入口](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch22/cgemm_host.cpp)

## 2. 需求分析

三个入口沿用同一参数次序 `handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc`，返回 `aclblasStatus_t`，无批量参数。标量由 Host 以对应 dtype 传入。矩阵为 Device ND 连续列主序存储，支持前导维度填充，C 原地更新，调用方保证其可写区域与 A/B 可读区域不重叠。

| 接口 | 标量及存储类型 | `op(X)` | 计算精度 |
| --- | --- | --- | --- |
| `aclblasSgemm` | `float` | `N` 原样，`T/C` 转置 | FP32 乘加和 epilogue |
| `aclblasCgemm` | `aclblasComplex`，相邻 FP32 实、虚分量 | `N` 原样，`T` 转置，`C` 共轭转置 | FP32 分量乘加和复数 epilogue |
| `aclblasHgemm` | `aclblasHalf`，IEEE 754 binary16 位模式 | `N` 原样，`T/C` 转置 | FP16 输入、FP32 累加与缩放，写回一次 RNE FP16 |

`op(A)` 为 `m×k`，`op(B)` 为 `k×n`，`C` 为 `m×n`。A 的物理列数为 `transa=N ? k : m`，`lda≥max(1, transa=N ? m : k)`。B 的物理列数为 `transb=N ? n : k`，`ldb≥max(1, transb=N ? k : n)`。C 的 `ldc≥max(1,m)`，只更新 `C[i+j*ldc]`（`i<m,j<n`），列间 padding 保留原字节。接口按运行时尺寸和转置分派，不涉及广播、任意张量视图或跨调用计算缓存。

### 2.1 参数校验与退化语义

Host 按以下顺序处理，维度、前导维度和标量空指针在 quick return 前校验。

1. 空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。负维度、非法前导维度、空 `alpha/beta` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. `m=0` 或 `n=0` 时 A/B/C 均允许为空，直接返回成功，无矩阵访问或 kernel 下发。非空输出范围要求 C 非空。
3. Host 值快照参与 Tiling。`k=0` 或 `alpha=0` 时 A/B 可为空且完全不访问。`beta=1` 原样返回，`beta=0` 直接写零，其余仅缩放逻辑 C。一般乘加仅在乘积有效时要求 A/B 非空，并在 `beta=0` 时完全省去 C 旧值的加载。复数零和一按实、虚两个分量识别。H 标量先按二进制位解释成 FP16 值再转 FP32。
4. 对本调用实际访问的矩阵，用检查过的 64 位乘加计算最后逻辑元素偏移、字节跨度与工作区大小。所有前置校验完成后按 `handle` 绑定的 stream 异步下发，返回提交状态，调用者同步后读取结果。Host 只处理参数和调度。

## 3. 详细设计

### 3.1 计算与数据流

每个输出执行 `C=alpha·op(A)·op(B)+beta·C`。S/H 的主项为 `2mnk` FLOP，C 的四实积为 `8mnk` FLOP，另有分量合成和 epilogue。忽略设备实际重复搬运，若逻辑元素宽度为 `s`，矩阵最低流量为 `s·(mk+kn+mn+[beta≠0]mn)` 字节。小尺寸、短 K 和窄 M/N 主要受下发、打包和尾块成本影响。大 K 的方阵重用度高，以 Cube 吞吐和流水重叠为主。比较路径时加上布局转换、所有 K 分段、累加和写回。

| 路径 | 适用条件 | 设计 |
| --- | --- | --- |
| Vector | 退化缩放或输出块很少且 K 短 | 一个 kernel 分配逻辑 C 区域。缩放或完成该输出的整个 K 归约，在 UB 完成 epilogue，避免为小矩阵启动独立打包核。 |
| S 的 Cube | 对齐、布局和 stride 满足原生载入与写回 | 复用 FP32 `arch22` Cube。紧凑 `alpha=1,beta=0` 且能一次完整归约时尝试直接写 C。其他情况用片上或有界暂存，并用 AIV 写回。HF32 降精度路径保持关闭。 |
| H 的 Cube | 有足够计算量摊销搬运 | FP16 Tile 按 K 分段搬入 L1/L0A/L0B，L0C 保留 FP32 累加。一般 alpha/beta 在 FP32 epilogue 处理，末次写回才转换为 FP16。紧凑标准标量仅在确认 Cube 写回满足单次 RNE 后直出。 |
| C 的 Cube + Vector | 大中型复数问题 | AIV 按列主序拆分 Tile 的实、虚分量并处理尾部。令 `sA,sB` 对共轭输入为 −1，其余为 +1，计算 `Pr=Ar·Br−sA·sB·Ai·Bi`、`Pi=sB·Ar·Bi+sA·Ai·Br`，四个 FP32 实积按输出块复用已装载的输入 Tile。完成 K 归约后，AIV 对 `Pr/Pi` 施加复数 alpha、beta 并交错写回。 |

C 的第一版以已有四实积路线作正确性对照。将共享的 FP32 Tile 调度下沉到 `arch22` 内部，避免 Cgemm 通过另一个公开 GEMM 接口重复执行参数校验、整矩阵打包与临时空间分配。较大分块可使用同一 stream 上分段打包、计算和写回，分段只保留复用的 A/B Tile 和本批 C 子块；只有完整设备调用收益支持时才合并为 AIC/AIV 流水。保持四实积公式，3M 属于另一接口。对小尺寸和 `M/N=1` 采用独立 Vector 路径或直接用已核对的 Tile 归约实现。已报告的 `K>1024` 分块及窄形状回归纳入首轮验证。

[arch22 Sgemm 缺陷记录](https://gitcode.com/cann/ops-blas/issues/1077)

### 3.2 Host Tiling 与工作区

- 通过 `PlatformAscendC` 获取 `GetSocVersion()`、`GetCoreNumAic()`、`GetCoreNumAiv()` 和参与流水的 UB、L1、L0A/B/C、L2 容量。各分派以 C 的逻辑输出块为工作单元，实际核数不超过可用核数和有效块数。按 `总块数/使用核数` 分配，余块依次补给前面的核。矩阵大于可用 L2 时按输出条带复用 A/B Tile。
- Tiling 记录逻辑 `m/n/k`、物理 `lda/ldb/ldc`、转置/共轭标志、dtype、Tile 尺寸与末块真长、块分配和各工作区字节偏移。缓冲基址及间隔遵守所选搬运接口的对齐要求。检查每级缓冲峰值，包括双缓冲、暂存、累加器和 epilogue 临时区。容量不足时缩小 Tile、减少双缓冲或改用分段，在最小有效 Tile 仍无法分配时报告分配错误。
- 首选一个核或一个混合核组拥有同一输出 Tile 的完整 K。仅在输出 Tile 数不足且长 K 并行有收益时启用 split-K。每个分片写独立、互不重叠的 FP32 局部和，单一归约核完成累加和最终写回。`beta*C` 对每个输出只应用一次。A/B 的 GM 尾部按真实长度读取，对齐补位在片上置零。对非对齐实数或交错复数写出用 `DataCopyPad` 按逻辑有效字节写回，保证相邻 Tile 和列填充不受影响。
- 使用 `ops-blas` 句柄工作区。相同 stream 利用队列顺序复用。跨 stream、工作区扩容或句柄销毁时，保留在途缓冲直到依赖完成。共享同一用户工作区的并发调用遵循库已有的所有权和同步约定。按每个缓冲核对 MTE/Cube/Vector 的 RAW、WAR 与 WAW。混合路径使用受支持的跨核同步及 GM 可见性，跨 kernel 路径由同 stream 前后顺序建立依赖。

### 3.3 工程与兼容性

- 实现放在 `blas/gemm/arch22/`。Sgemm/Cgemm 复用 `include/cann_ops_blas.h` 既有 ABI。新增非批量 `aclblasHgemm` 声明及 `arch22` 构建注册，按现有宏和产品分派一次定义符号。扩充 `blas/gemm/README.md` 的产品支持、参数、样例与异步约束。测试统一在 `test/gemm/arch22/` 注册，沿用 GTest/ATK 工程。
- Cgemm 的新任务接口覆盖普通 Host 栈、堆和钉页标量。保留上游 Cgemm 已有 Device 标量调用的兼容路径，评估使用 CANN Runtime 内存属性区分两类指针。先在 CANN 9.1.0 验证普通 Host 指针、已有 Device 指针及失败分支的识别和性能，再确定分派细节。识别失败时禁止将未知地址直接解引用。Host 路径不执行 Device 同步读标量。Device 兼容行为和出错处理纳入评审及回归，映射地址受 Runtime 文档约束。

  [CANN 9.1 Runtime 内存属性](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1803.html)
- `arch35` 保持独立编译，现有接口与行为做构建或同族回归。A2/A3 共用 `arch22` 实现，基于实际 SoC 与硬件查询分派。A2 构建入口为 `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`，A3 取对应 `--soc`。

## 4. 可维可测分析

### 4.1 精度与边界

| 接口 | 参考链 | `atol=rtol` | 绝对误差兜底 | 官方附件精度行 |
| --- | --- | --- | --- | ---: |
| Sgemm | 任务书高精度参考，以 float64 累加后落 FP32，独立对照 Netlib 3.10.0 `cblas_sgemm` | `2^-13` | `1e-2` | 334 |
| Cgemm | 两分量 float64 参考，落为 complex64，独立对照 Netlib 3.10.0 `cblas_cgemm` | `2^-13` | `1e-2` | 333 |
| Hgemm | FP16 输入先解码，以 float64 累加，先收窄 FP32 再按 IEEE RNE 落 FP16 | `2^-10` | `1e-1` | 333 |

按任务书 §3.2 和固定版本精度标准，以 `|actual−golden|≤atol+rtol·|golden|` 计算匹配率，要求至少 99%。有限点最大绝对误差同时受 `max(兜底, 32×ULP(g_low))` 约束，`g_low` 取最大绝对误差索引处按输出类型收窄的 golden。Cgemm 实部、虚部分别计算并同时达标。非有限值照固定标准 §2.3 比较，先处理收窄、溢出和符号，再算有限点统计。CSV 的 MERE/MARE 列供测试框架兼容使用，报告另按上述公式逐例验收。

[固定混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)

保留 1,000 条精度行以及 200 条性能行及独立 ID。测试记录包含随机种子、实际 A/B/C 字节、标量位模式、参考版本、输出与逐分量指标。覆盖 N/T/C 九种组合、紧凑和填充前导维度、零/负维度、非法枚举和空指针、对齐与 4095/4096 边界、均匀/正态及各 dtype 特殊值。以非对称复数矩阵区分 T 与 C，以 `alpha=0` 搭配空或 NaN A/B、`beta=0` 搭配 NaN 旧 C 验证实际访问，并检查 C padding 哨兵。补测大 K 和窄 M/N 的组合；若高精度参考与 Netlib 对照有差异，报告舍入链、影响用例和各自结果。精度 PASS 按任务书容差，不将位级复跑诊断设为验收条件。

### 4.2 性能与复现

正式测试在 910B3/CANN 9.1.0 运行。对 S/C/H 分别保留 67/67/66 条 GPU kernel 基线，按算子、`transa/transb/m/n/k/lda/ldb/ldc` 及完整 alpha/beta（复数四分量）对应唯一行 ID。预热 5 次，取随后 10 次有效调用的算术均值。异常样本保留原始记录并说明，缺失或无效基线保持缺证。

- **正式 kernel 口径**。`msprof op` 保存 `Task Duration(us)` 和逐例原始 CSV。单次调用若含多个必要设备任务，分别记录打包、计算、归约和写回，设计验证先按所有必需设备任务之和与 GPU kernel 数据对比。正式阈值为逐例 NPU 均值 `≤ 1.25 × gpu_ms × 1000`，同时保留整体平均倍率。任何单段耗时均另列为优化诊断。
- **辅助调用口径**。GTest 在预置 Device 缓冲上，以 `aclrtEvent` 夹住连续 10 次完整算子调用，输出 `[PERF] case=… avg_us=…`。该口径包括下发和完整设备路径，排除造数、H2D/D2H 和 golden。丢失事件样本为辅助验证缺项，无 GTest 整例毫秒替代。
- **首轮验证**。先验证可复用的全量正确性参考与测试链，再用同一实现覆盖 4095³/4096³、窄矩阵长 K、非对齐 TN/CT、填充 `ldc` 和非默认标量靶点。Hgemm 的 4096³ 案例 `hgemm-perf-023` 对应的 NPU 耗时上限约为 771 μs。小矩阵重点核对较低毫秒基线对应的微秒上限。记录改动前后的完整调用耗时，收益覆盖转换及同步成本后再保留优化。

验收前以同一源码与构建产物归集精度、两套性能数据、原始日志及复现步骤。按任务书 §4 生成自测用例交付件，纳入 `cases.json`、`perf/`、冻结输入、golden 与 `facts.json`。§3.4 内存验收不涉及，随模板的内存项注明该条款。本文给出设计及验证计划，测试结果由后续自验提供。
