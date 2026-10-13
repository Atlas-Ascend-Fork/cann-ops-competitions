# aclblasCtrsmBatched 算子设计文档（Atlas A2/A3，arch22）

| 项 | 内容 |
|---|---|
| 算子名称 | ctrsmBatched（批量复数三角方程组求解） |
| 目标架构 | arch22（Atlas A2 训练/推理系列，Ascend 910B 系列） |
| 数据类型 | complex64（ACL_FLOAT 复数） |
| 参考接口 | cuBLAS `cublasCtrsmBatched` |
| 交付路径 | `blas/ctrsmBatched/arch22/` + `test/ctrsmBatched/ctrsmbatched/` |

## 1 需求背景

BLAS Level 3 的 TRSM（Triangular Solve）是稠密线性代数求解链路（如 LU/CholesLin 分解后的回代求解、迭代精化 preconditioner）的核心原语。科学计算与 AI 框架中大量场景以批量独立小矩阵形式出现（边界元、量子线路仿真、控制论 Riccati 方程求解等），逐次调用 `ctrsm` 的 launch 与同步开销成为瓶颈，需要 batched 接口一次性求解整批方程组。

ops-blas 仓库现状：

- `blas/trsm`、`blas/trsmbatched` 仅有 arch35（950PR/950DT）实数单精度实现；
- arch22 平台（Atlas A2/A3）上**无任何 TRSM 类算子**（实数/复数均无）；
- 仓内 `experimental/aclblasCtrsmBatched2` 提供了经 320 case 验证的复数 batched 参考实现（对标 cuBLAS 平均 GPU/NPU=0.800），但为独立样例工程，未接入正式构建、接口体系与测试框架。

本任务将其产品化：交付 `aclblasCtrsmBatched` 正式接口，补齐 arch22 平台复数 TRSM 能力。

## 2 需求分析

### 2.1 功能定义

求解批量复数三角线性方程组，解矩阵原地覆盖 B：

```
side='L': op(A[i])   * X[i] = alpha * B[i]
side='R': X[i] * op(A[i])   = alpha * B[i],  i = 0 .. batchCount-1
```

- `op(A)`：`trans='N'` 时 A；`trans='T'` 时 A^T；`trans='C'` 时 A^H（共轭转置）；
- A[i] 为 kDim×kDim 三角矩阵（side=L 时 kDim=m，side=R 时 kDim=n），仅引用 uplo 指定三角部分，另一三角不访问；
- diag='U'（unit）时对角元按 1 处理（不访问）；diag='N'（non-unit）时除以实际对角元；
- alpha 为复数标量（host 指针）；alpha=0 时 B 置零（不访问 A）；
- B[i] 为 m×n 复数矩阵，输出时原地覆盖为 X[i]。

### 2.2 接口定义

与仓内 batched 系算子（`aclblasStrsmBatched`/`aclblasCgemmBatched`）签名风格一致：

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t transa,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const A[],
    int lda,
    aclblasComplex* const B[],
    int ldb,
    int batchCount);
```

参数约束（校验顺序遵循 cherk 规范：枚举/维度 → quick-return → 指针 → ld 下界）：

| 参数 | 约束 |
|---|---|
| handle | 非空，否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| side/uplo/transa/diag | 枚举合法值，否则 `ACLBLAS_STATUS_INVALID_VALUE` |
| m, n | ≥ 0；m==0 或 n==0 时 quick-return SUCCESS |
| alpha | host 指针，非空 |
| A | device 指针数组（batchCount 个）；alpha≠0 时非空 |
| B | device 指针数组，非空；元素可为空仅当对应 batch 满足 m==0\|\|n==0 |
| lda | side=L 时 ≥ max(1,m)；side=R 时 ≥ max(1,n) |
| ldb | ≥ max(1,m) |
| batchCount | ≥ 1 |

### 2.3 性能需求

对标 cuBLAS `cublasCtrsmBatched`（V100 级 GPU，320 case 实测数据见任务包 `docs/GPU_TEST`）：

- 验收口径：NPU 耗时 ≤ GPU 标杆（样例实测平均 GPU/NPU=0.800，即整体已快 25%）；
- 计时方法：TC_PF 路径，warmup 5 + 55 次同步计时取平均（与仓内 cher2k_perf 同口径）；
- 覆盖：16 种 side×uplo×trans×diag 组合 × 尺寸 16~8192 × batch 1~248 × 对齐/非对齐。

### 2.4 精度需求

- 单元素相对误差 MERE < 2^-13 × 10 = 1.22e-3（任务包口径）；
- 仓内测试框架容差：`MERE_MARE` 模式（mere=1/8192、mare multiplier=10）+ `applyMixedTolerance` 混合容差兜底（rtol=2^-10、atol=2^-16、99% 匹配率）；
- TRSM 为逐 panel 前代/回代，误差随 panel 数线性增长，kDim=8192/nb=32 时 256 个 panel，fp32 累加误差量级 ~1e-4，满足上述容差。

## 3 详细设计

### 3.1 总体架构：MIX_AIC_1_2 混合核

采用 1 Cube 核 + 2 Vector 核混合调度的**单 kernel** 设计（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`），每个 block 内：

- **AIV（Vector）×2**：三角 panel 的前代/回代求解（TRSV）、复数 AoS↔SoA 格式转换、结果回写；
- **AIC（Cube）×1**：trail 区域的复数 GEMM 更新（Matmul API，4M 实平面分解）；
- **跨核同步**：`CrossCoreSetFlag/WaitFlag`（FLAG_TRSV=6：AIV→AIC panel 完成；FLAG_GEMM=7：AIC→AIV GEMM 完成；FLAG_DUAL_AIV=8：双 AIV 工作区屏障）。

选择混合核而非 chemm/cherk 式多 kernel 顺序流水的原因：TRSM 的 panel 求解（AIV）与 trail 更新（AIC）存在逐 panel 交替依赖，单 kernel 内跨核 flag 往返可避免每 panel 一次 kernel launch；样例 320 case 实测该设计已达性能标杆。

### 3.2 算法流程（blocked TRSM）

复数消元按实平面 4M 分解组织。以 side=L、forward（下三角不转置）为例，kDim 按 nb（16/32）分 panel：

```
规范化阶段（AIV）:
  A' = pad/canon(A)     # 补零对齐 8、必要时转置/共轭 → AoS 稠密 kDim×2kDim
  Bc = deinterleave(B)  # AoS(m×n) → SoA 实/虚平面(kDim×nColsAligned)，含补零

求解阶段（逐 panel p = 0..numPanels-1，forward 从前往后）:
  [AIV] LoadPanelA(p)      # 对角块 A[p*nb:(p+1)*nb, p*nb:...] → UB
  [AIV] PanelTrsv(p)       # 解 Xp = A_p^{-1} * B_p，写 Xneg 工作区（双缓冲槽位）
  [AIV] SetFlag(TRSV)      # 通知 AIC
  [AIV] （与 AIC 并行）WriteBackPanelRows / 预加载下一 panel A
  [AIC] WaitFlag(TRSV) → GEMM: B_trail -= A_trail * Xp（实/虚平面各一次实 GEMM）
  [AIC] SetFlag(GEMM)
  [AIV] WaitFlag(GEMM) → 下一 panel

回写阶段（AIV）:
  side=L: 逐 panel 求解后即写回 B 原始布局
  side=R: 求解完成后整矩阵 SoA → AoS 转置回写
```

Upper/backward 路径仅 panel 遍历方向相反（p 从 numPanels-1 到 0），模板参数 `CtrsmMixAivImpl<FORWARD, RIGHT>` 编译期消除分支，共 4 条实例化路径。

### 3.3 AIC 组调度与远端更新

AIC 侧按 LIM_GROUP=16 个 panel 一组组织 GEMM：

- **DirectRankK**：panel 解出后立即更新紧邻 trail 的 nb 行（与 AIV 求解下一 panel 重叠）；
- **PreUpdateRankKWithinGroup**：等待期提前更新组内其余行；
- **BigGroupGemm / FarGemm**：组末对远端大块（组边界之外）做一次 K=groupSize×nb 的大 GEMM 累加（FAR_STEP=2 分段，与 AIV 并行）；
- Xneg 双缓冲（组奇偶交替槽位）避免 AIV 写下一组 X 与 AIC 读当前组 X 的竞争。

### 3.4 复数 GEMM 更新（AIC，Matmul API）

trail 更新 `B_trail -= A_trail * Xp` 展开为两个实 GEMM（Xneg 以实/虚平面分别存储）：

```
Bc_real -= AaR * Xr - AaI * Xi     # mm.SetTensorB(xR); IterateAll(→eBR, enAtomic=1)
Bc_imag -= AaR * Xi + AaI * Xr     # mm.SetTensorB(xI); IterateAll(→eBI, enAtomic=1)
```

A 以 AoS 交织存储（kDim×2kDim，AaR/AaI 由 GEMM 的 K 维 2 倍技巧在一次 GEMM 内同时取实虚列：`SetSingleShape(rowCount, nCols, 2*actualK)`，K 沿实虚交织展开，A_trail 列即 [Ar | Ai]），X 为解出的实平面。tiling 由 host 侧 `MultiCoreMatmulTiling`（ND/ND/ND float，dim=1）生成，经 `REGIST_MATMUL_OBJ` 注册。

### 3.5 多核与 batch 并行策略

block 划分（numBlocks 由 host 计算）：

| 场景 | 条件 | 划分方式 |
|---|---|---|
| 多核列拆分 | dualAiv 且 batch ≤ AI核数/2 | 每 batch 的列分 numSplits 个 split（≥64 列/split），每 split 占 1 block（2 AIV） |
| 双 AIV 分列 | kDim≥128 且 nColsAligned≥128 | 每 batch 1 block，2 AIV 各处理一半列 |
| 原始模式 | 其余 | 每 block 处理 1 个 batch（奇数 batch 时尾 block 走 DummyPanelSync 维持 flag 对齐） |

- 双 AIV 分列/分行协作经 FLAG_DUAL_AIV（CrossCore mode1）互等；
- AI Core 总数运行时经 `platform_ascendc::PlatformAscendCManager::GetCoreNumAiv()` 获取（产品化改造点：样例硬编码 20）。

### 3.6 数据流与 workspace 布局

每 batch（或 split）的 GM workspace：

```
+-------------------+---------------------------+------------------------+
| A' AoS 工作区      | Bc 实平面 | Bc 虚平面       | Xneg/GEMM 缓冲          |
| kDim × 2kDim      | kDim × nColsAligned (×2)  | 2×LIM_GROUP×2nb×nCols  |
+-------------------+---------------------------+------------------------+
（useOrigA 时跳过 A' 直接用原矩阵；split 模式 B 平面 × numSplits）
```

- workspace 总量 = `batchCount × workspaceOffset`（split 模式含 numSplits 份 split 缓冲）；
- 产品化改造：样例每次调用 `aclrtMalloc/Free` 6 块缓冲 → 改用 `_aclblas_handle` 的 `EnsureDefaultWorkspace(h, need)` 统一分配（tiling/cubeTiling/指针数组等小缓冲并入同一块，按 32B 对齐切分）；Matmul 系统工作区（`GetLibApiWorkSpaceSize()`，下限 16MiB）与 tiling 缓冲按需追加；
- A[]/B[] 为 device 指针数组，直接传给 kernel（`__gm__ uint64_t*` 数组寻址），**无 D2H 回读**（区别于 arch35 逐 batch launch 模式，保持单 kernel 全 batch 设计）。

### 3.7 UB 缓冲（192KB 预算）

编排类统一分配，子类持指针（别名布局）：

```
bufPanelA_real/imag (nbAligned² ×2) | bufPanelB_real (max(TS², 2nb·colTile))
| bufPanelB_imag (nb·colTile) | bufRankK (colTile) | bufNeg_real/imag (nb·colTile ×2)
| bufNeg_imag_neg | bufRow (rowMax) | bufGatherEven/Odd (gatherChunk int ×2)
```

colTile 自适应收缩直至满足预算（README：大矩阵 240~280 列/tile）。

### 3.8 kernel 代码结构（产品化文件组织）

仓内目标结构（glob 自动进构建，零手工注册）：

```
blas/ctrsmBatched/
├── README.md                        # 产品支持情况/接口/约束（对齐 strsmbatched README 体例）
└── arch22/
    ├── ctrsmbatched_host.cpp        # 校验/tiling/workspace/launch（重写）
    ├── ctrsmbatched_kernel.h        # kernel_do 声明（GM_ADDR 退化技巧）
    ├── ctrsmbatched_kernel.cpp      # MIX 入口 + kernel_do
    ├── ctrsmbatched_tiling_data.h   # tiling struct + 枚举镜像（host/device 共用，布局字节一致）
    └── ctrsmbatched_kernel_parts.h  # AIV/AIC 实现单元（cfg/convert/canon_a/canon_b/deinterleave/solver/aic 按节组织）
```

样例 11 个文件的职责拆分保留（CodeCheck 函数≤50 行友好），合并为少量仓内文件以符合四件套+辅助头模式；所有头文件 `#pragma once`；include 路径改用 `kernel_operator.h` + `lib/matmul_intf.h`（CANN 安装路径，不再依赖 `../../` 相对路径与 `aclblas_minimal.h`）。

### 3.9 host 侧流程

```
aclblasCtrsmBatched
  ├─ 校验（枚举→quick-return→指针→ld）
  ├─ alpha==0 快路径：逐 batch aclrtMemset B 置零（复用样例逻辑）
  ├─ FillTiling：kDim/nCols 映射、useOrigA/nb(16|32) 判定、dualAiv/numSplits 计算
  │   （AI 核数运行时获取）、workspaceOffset/splitBGemmSize 计算
  ├─ GenerateCubeTiling：MultiCoreMatmulTiling（ND float, dim=1）
  ├─ EnsureDefaultWorkspace(h, total)  # gemmWs + sysWs + tiling/cubeTiling/ptrArrays
  ├─ H2D：tiling struct、cubeTiling（A[]/B[] 已在 device，直传）
  └─ ctrsmbatched_mix12_kernel_do(..., numBlocks, h->stream)
```

### 3.10 精度保障

- 规范化阶段补零对齐（kDim/nCols 对齐 8 float=32B），GEMM/搬运全程对齐路径；
- Xneg 双缓冲 + CrossCore flag 保证 AIV/AIC 数据依赖；
- GM 工作区跨核传递点按仓内规范补 `DataCacheCleanAndInvalid<CacheLine::ENTIRE_DATA_CACHE>`（chemm 同款三保险，样例已含 PipeBarrier，产品化时统一核对）；
- 非对齐尺寸（17/20/31/33/227 等）走 Gather 偏移表解交织 + DataCopyPad。

## 4 可维可测分析

### 4.1 测试工程

```
test/ctrsmBatched/ctrsmbatched/
├── CMakeLists.txt                  # ops_blas_add_gtest_tests(${OPS_BLAS} ctrsmbatched_test)
├── ctrsmbatched_param.h            # CSV 解析（复数标量 alpha_real/alpha_imag 列）
├── ctrsmbatched_golden.h           # CPU 双精度参考 + FNV-1a golden 缓存
└── arch22/
    ├── ctrsmbatched_npu_wrapper.h  # H2D/D2H/workspace/超时保护封装
    ├── ctrsmbatched_test.cpp       # BlasTest<> + TC_PF 性能分支
    └── ctrsmbatched_test.csv       # 用例表
```

### 4.2 用例设计（CSV）

| 类别 | 覆盖 |
|---|---|
| TC_L0 负例 | handle/side/uplo/trans/diag/alpha/A/B 非法、lda/ldb 越界、m/n<0、batchCount<1 |
| TC_L0 边界 | m==0、n==0、alpha==(0,0)、kDim=1、batch=1 |
| TC_L1 功能 | 16 种 side×uplo×trans×diag 全排列；对齐（64/128/256/1024/8192）与非对齐（17/31/100/227）；batch 1/2/3/8/32/118/248；复数 alpha（含纯虚、负实） |
| TC_PF 性能 | 对标 GPU_TEST 的 320 case（同形状同 batch 采样） |

golden：CPU 双精度逐 batch 前代/回代（op(A) 展开含转置/共轭），对角 unit 跳过除法。

### 4.3 判定口径

- 精度：`MERE_MARE`（1/8192×10）为主，mixed tolerance 兜底；负例仅断言返回码；
- 性能：TC_PF 分支 aclrtEvent 计时（warmup 5 + 55 iters），输出 `[case] Avg time = N.NN us`，验收 ≤ GPU_TEST 标杆值。

### 4.4 维护性

- 算法单元按职责类拆分（convert/canonA/canonB/solver/aic），单元内函数 ≤50 行（个别热路径大函数按仓内惯例走 CodeCheck 豁免）；
- tiling struct host/device 字节布局一致约束以注释固化；
- 路径选择（split/dual/original）、workspace 公式在 host 单点实现，README 记录关键参数（nb/LIM_GROUP/FLOAT_ALIGN/dualAivMode）。

### 4.5 交付物清单

1. `blas/ctrsmBatched/arch22/` 实现 + `include/cann_ops_blas.h` 声明；
2. `test/ctrsmBatched/ctrsmbatched/` 测试工程（CSV ≥ 60 精度用例 + 320 性能采样）；
3. `blas/ctrsmBatched/README.md`（产品支持情况：Atlas A2 训练/推理系列支持）；
4. 自测报告（精度截图 + 性能对标数据 + 内存数据）。

## 5 与参考实现（experimental）的差异清单

| # | 样例 | 产品化后 |
|---|---|---|
| 1 | `int64_t` + `std::complex<float>` | `int` + `aclblasComplex`（仓内风格） |
| 2 | 每调用 `aclrtMalloc/Free` ×6 | `EnsureDefaultWorkspace` handle 工作区 |
| 3 | `TOTAL_AICORES=20` 硬编码 | `PlatformAscendCManager` 运行时获取 |
| 4 | 独立 CMake + `aclblas_minimal.h` | 仓内 glob 构建 + `cann_ops_blas_common.h` |
| 5 | `../../trsm_mix_common.h` 相对路径 | 并入 `ctrsmbatched_kernel_parts.h`/tiling 头 |
| 6 | host 指针数组 H2D | device 指针数组直传 kernel |
| 7 | python verify 独立校验 | GTest + CSV + golden + mixed tolerance |
| 8 | deinterleave 头文件 doxygen 注释未闭合（编译错误） | 修复并全量 clang-format |
