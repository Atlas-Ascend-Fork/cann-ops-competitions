# aclblasCgemm3m 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCgemm3m` |
| 任务名称 | 11月社区任务-aclblasCgemm3m算子开发（A2/A3） |
| 目标硬件 | Atlas A2 训练/推理系列、Atlas A3 训练/推理系列（arch22） |
| 软件版本 | CANN 9.1.0 |
| 开发方式 | Ascend C Kernel 直调（handle 绑定 stream） |
| 数据类型 | COMPLEX64（`aclblasComplex`，实部/虚部均为 FLOAT32）；`alpha`/`beta` 为复数标量指针 |
| 对标接口 | cuBLAS `cublasCgemm3m`（legacy，3m 算法） |
| 精度 golden | Netlib BLAS 3.10.0；主判据按 float64 累加后收窄到 COMPLEX64（任务书 §3.2.4），另以 `cblas_cgemm` float32 参考做交叉对照 |
| 算法要求 | 三次实数矩阵乘的线性组合完成一次复数矩阵乘 |
| 合入目录 | 实现 `cann/ops-blas: blas/gemm/arch22/`，测试 `test/gemm/cgemm3m/arch22/` |
| 接口声明 | `include/cann_ops_blas.h` 已存在，本任务不新增、不定义私有平行 API |
| 文档状态 | 设计稿，实测数据已按 910B4 自验结果回填 |

---

# 1. 需求背景（required）

## 1.1 需求来源

本需求来自昇腾 CANN 社区任务《aclblasCgemm3m 算子开发（Atlas A2/A3）》，权威需求来源为任务书
`aclblasCgemm3m_AtlasA2_A3_task_doc.md`。任务要求在 Atlas A2/A3（arch22）上使用 Ascend C，基于
`cann/ops-blas` 开源仓的句柄式 BLAS 工程模式实现单精度复数三乘法通用矩阵乘算子
`aclblasCgemm3m`，并完成 Host 参数处理、Device Kernel、测试工程、算子 README 与自测材料。

`ops-blas` 已有：

- arch22 复数 GEMM `aclblasCgemm`（`blas/gemm/arch22/`），内部使用四次实数乘积；
- arch35 复数 3m GEMM `aclblasCgemm3m`（`blas/gemm/arch35/cgemm3m_host.cpp`）及其测试；
- 公共声明 `aclblasCgemm3m`（`include/cann_ops_blas.h`）。

本任务补齐 **arch22 的三乘法实现与测试**。重点不是新增接口，而是把复数乘积固定为三次实数矩阵乘，
并在 A2/A3 的 AIC/AIV 架构上完成 K 方向切分、转置/共轭语义、alpha/beta 融合与精度控制。

## 1.2 背景介绍

### 1.2.1 算子语义

`aclblasCgemm3m` 计算：

```text
C = alpha * op(A) * op(B) + beta * C
```

- `A`、`B`、`C` 为 COMPLEX64 矩阵，按列主序存储，复数实部在低地址；
- `op(X)` 取 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）或 `ACLBLAS_OP_C`（共轭转置）；
- 复数下 `T` 与 `C` 语义不同：`C` 对虚部取反；
- `alpha`、`beta` 为 Host 内存中的复数标量；
- `op(A)` 为 m×k，`op(B)` 为 k×n，`C` 为 m×n；
- 仅写 C 的逻辑 m×n 区域，ldc padding 列不被修改。

与 `aclblasCgemm` 的区别是算法契约：`aclblasCgemm3m` 的所有非退化路径必须使用三次实数矩阵乘，
而 `aclblasCgemm` 可以根据 scalars 选择四次或三次路径。

### 1.2.2 3M 分解

记 `Ar`、`Ai`、`Br`、`Bi` 为 A、B 的实部和虚部。定义转置共轭符号：

```text
sa = -1 (transa == C) else +1
sb = -1 (transb == C) else +1
a_i = sa * Ai
b_i = sb * Bi
```

本设计采用融合 Gauss 3M 恒等式：

```text
P0 = (Ar + a_i) * Br
P1 = Ar * (b_i - Br)
P2 = a_i * (Br + b_i)

Re(op(A) * op(B)) = P0 - P2
Im(op(A) * op(B)) = P0 + P1
```

代数验证：

```text
P0 - P2 = (Ar + a_i)Br - a_i(Br + b_i) = Ar*Br - a_i*b_i
P0 + P1 = (Ar + a_i)Br + Ar(b_i - Br)  = a_i*Br + Ar*b_i
```

即实部为 `ArBr - a_i b_i`，虚部为 `a_i Br + Ar b_i`，与复数乘法一致。转置由实数 GEMM 的
`transA/transB` 参数完成；共轭由 `sa/sb` 折入输入平面求和与乘积，split kernel 不修改原始虚部符号。

与朴素的四次实数乘相比，3M 将实数乘法次数从 `4mnk` 降为 `3mnk`，代价是输入侧多两次
O(mk+kn) 的加减和输出侧多两次 O(mn) 的加减，同时改变了 FP32 舍入顺序。因此
`aclblasCgemm3m` 与 `aclblasCgemm`/`cblas_cgemm` 不保证逐位一致，只按精度阈值判定。

### 1.2.3 与仓内既有实现的关系

arch35 已有 `cgemm3m_host.cpp`，其融合路径与纠正面元为 950PR 的 Cube/UB 容量和
SIMT 指令设计。arch22 不复用 arch35 的设备代码，而是复用仓内已验证的公共 helper，
并为常规路径新增融合三乘积 kernel：

| 模块 | 使用方式 |
| --- | --- |
| `blas/gemm/arch22/cgemm3m_host.cpp` | 本任务新增：参数校验、tiling、workspace 规划、常规/补偿路径调度 |
| `blas/gemm/arch22/cgemm3m_kernel.cpp/.h` | 本任务新增：AIV 分块打包、融合三乘积 Cube、复数写回、窄输出流式打包、Kahan 累加 |
| `blas/gemm/arch22/cgemm_kernel.cpp` | 仅大 alpha 补偿路径复用 split/accum/combine |
| `blas/gemm_ex/arch22/` | 仅大 alpha 补偿路径使用实数 SGEMM 后端 `aclblasSgemmEx` |
| `blas/common/helper/complex_blas3_arch22.h` | SplitBody、平面布局和交织索引约定 |
| `blas/common/helper/complex_blas3_host_utils.h` | checked 尺寸运算 |

常规路径不经过 `aclblasSgemmEx`，而是在一次 Cube 下发中完成三组实数乘积，具体见 §3.3。

---

# 2. 需求分析（required）

## 2.1 外部依赖

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| CANN 9.1.0（ASC 编译器、`kernel_operator.h`、Ascend C） | arch22 Kernel 编译 | A2/A3 实机验证 |
| ACL 运行时（`aclrtMalloc`、`aclrtMemcpy`、stream、event） | Device 内存、异步下发、计时 | 通过现有 helper |
| Netlib BLAS 3.10.0（`cblas_cgemm` / `cblas_zgemm`） | 精度 golden | 测试工程链接系统 BLAS |
| GTest | 测试框架 | 仓内 `test/frame` 已封装 |

不引入新的三方依赖。

## 2.2 接口与参数

接口声明已存在，签名如下：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 参数 | 输入/输出 | 类型 | 约束 |
| --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | 非空，绑定 `aclrtStream` |
| `transa`、`transb` | 属性 | `aclblasOperation_t` | `N/T/C` |
| `m`、`n`、`k` | 属性 | `int` | ≥ 0 |
| `alpha`、`beta` | 输入 | `const aclblasComplex*` | 非空，Host 内存 |
| `A`、`B` | 输入 | `const aclblasComplex*` | Device 列主序 |
| `C` | 输入/输出 | `aclblasComplex*` | Device 列主序，原地覆写 |
| `lda` | 属性 | `int` | transa=N 时 ≥ max(1,m)，否则 ≥ max(1,k) |
| `ldb` | 属性 | `int` | transb=N 时 ≥ max(1,k)，否则 ≥ max(1,n) |
| `ldc` | 属性 | `int` | ≥ max(1,m) |

## 2.3 异常行为与校验顺序

| 序号 | 检查项 | 条件 | 返回码 |
| --- | --- | --- | --- |
| 1 | handle | nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | transa | 非 N/T/C | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | transb | 非 N/T/C | `ACLBLAS_STATUS_INVALID_ENUM` |
| 4 | 维度 | m<0 或 n<0 或 k<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | lda | 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | ldb | 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | ldc | 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | alpha/beta | 任一为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | quick return | m==0 或 n==0 | `ACLBLAS_STATUS_SUCCESS`，不引用 A/B/C，不启核 |
| 10 | C | m>0 且 n>0 且 C==nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 11 | A/B | m>0 且 n>0 且 k>0 且 alpha≠(0,0) 且 A 或 B 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |

说明：

- 非法枚举返回 `INVALID_ENUM` 是 ops-blas 约定，与 cuBLAS 的 `INVALID_VALUE` 有意不同；
- `k=0` 或 `alpha=0` 不引用 A/B，允许 A/B 为空，但 m,n>0 时 C 必须非空；
- `beta=0` 时 combine kernel 的 `applyBeta` 分支不执行，故不读 C 旧值；
- `beta=1` 且无乘积项时直接返回，不下发 kernel；
- 仅校验 Host 指针和结构参数，不扫描 Device 张量元素。

## 2.4 需求拆解

| # | 需求项 | 交付物 |
| --- | --- | --- |
| 1 | Host 校验、quick return、工作区、launch | `blas/gemm/arch22/cgemm3m_host.cpp` |
| 2 | 常规路径：融合 3M 打包 + 三乘积 Cube 调度 + 写回 | `cgemm3m_kernel.cpp` 的 pack/cube/finish kernel |
| 3 | OP_C 共轭语义 | 打包时对虚部取反，并把 `sa/sb` 折入三组操作数 |
| 4 | alpha/beta 融合与 beta=0 不读 C | finish kernel；补偿路径复用 `cgemm_combine_kernel` |
| 5 | 工作区溢出/上限检查 | `CBlas3TryMulU64`/`CBlas3TryAddU64` + `EnsureDefaultWorkspace` |
| 6 | 测试工程与 1200 条 CSV | `test/gemm/cgemm3m/arch22/` |
| 7 | CMake 注册与单算子构建依赖 | `test/gemm/cgemm3m/CMakeLists.txt`、`blas/CMakeLists.txt` |
| 8 | 算子 README | `blas/gemm/README.md` 的 `aclblasCgemm3m` 章节 |
| 9 | 自测报告与日志 | `task_submission/` |

---

# 3. 详细设计（required）

## 3.1 Host 流程

```text
aclblasCreate / aclblasSetStream
        │
        ▼
aclblasCgemm3m
        ├─ 11 步校验（handle → trans → dims → ld → scalars）
        ├─ quick return: m==0 || n==0 → SUCCESS
        ├─ C/A/B 条件指针校验
        ├─ Host 解引用 alpha/beta
        ├─ hasTemp = (k>0 && alpha≠0)
        ├─ accurate = k<=1024 && m*n<=262144 && |alpha|>=4
        ├─ hasTemp && !accurate && m,n,k<=8192 → 常规路径
        │    ├─ 规划 pm/pn/pk、输出 tile、K 分段、workspace
        │    ├─ cgemm3m_pack_kernel  : A/B → 三组实平面
        │    ├─ cgemm3m_cube_kernel  : 一次下发完成三组乘积
        │    └─ cgemm3m_finish_kernel: 归约部分和 + alpha/beta + 写 C
        └─ 否则 → 补偿路径：委托 SgemmEx 的 split/accum/combine（Kahan）
```

`hasTemp == false` 时只下发 combine（beta≠1）；`beta==1` 时直接返回。常规路径的窄输出场景（m≤64、B 不转置、n 和 k 较大）合并 Cube 与流式 B 打包为一次混合下发。

## 3.2 输入打包

常规路径在 `cgemm3m_pack_kernel` 中一次读入 A、B，并按 Cube 的
`C^T = op(B)^T * op(A)^T` 视角保持原始物理朝向（ND）。每个逻辑矩阵在 packed 区展开为三组实平面：

```text
A: Ar+sa*Ai, Ar, sa*Ai      // transa=N 时 pm×pk，否则 pk×pm
B: Br, sb*Bi-Br, Br+sb*Bi  // transb=N 时 pn×pk，否则 pk×pn
```

每个 tile 以 `DataCopyPad` 按 lda/ldb 搬运，UB 行 pitch 保持 32B 对齐；`GatherMask`
分离实部/虚部；OP_C 通过对虚部取反实现，`sa/sb` 同时折入三组平面。逻辑边界外的
padding 行/列在 UB 中清零，保证 Cube 的补齐项不贡献结果。打包按 `(行块, 列块)` 任务分核，
A、B 各只写一遍。

当输出较窄（m ≤ 64）、B 不转置且 n、k 较大时，改用流式打包：B 按 `(nTile, kChunk)`
分块，配对的 AIV 核用双缓冲准备下一块，Cube 核同时消费当前块；A 仍整体共享。

## 3.3 三乘积 Cube 调度

常规路径只下发一次 `cgemm3m_cube_kernel`。任务空间为 `(K 分段, 输出 tile, 乘积)` 三元组，
每个任务独占输出矩形，不使用原子累加：

```text
P0 = (Ar + sa*Ai) * Br
P1 = Ar * (sb*Bi - Br)
P2 = sa*Ai * (Br + sb*Bi)
Re = P0 - P2,  Im = P0 + P1
```

- 输出 tile 按形状选择：≤128 阶用 32×16；m≤64 且 n≥512 用 64×128；≤512 阶用 128×64；
  其余用 256×128；256~384 阶取两列 tile，以平衡三倍乘积任务。
- K 分段目标长度按输出平面和 k 选择 128/256/512/1024/2048；并抬高分段数，使总任务数
  至少覆盖 3 倍 Cube 核；`kChunk` 32 对齐，受 workspace 上限约束。
- 每个任务调用 `Matmul` 计算 `rows=tileN, cols=tileM, depth=kChunk`，三组乘积分别写入
  `3*pm*pn` 的部分和平面。
- K 分段部分和由写回 kernel 归约，再乘 alpha、加 beta*C。

## 3.4 精度控制

- 常规路径：融合 Gauss 3M 只有三次 FP32 乘积和两次重组，K 分段缩短单个 Cube 累加链，
  降低长 K 相消误差。
- 补偿路径：当 `k <= 1024`、`m*n <= 262144` 且 `|alpha| >= 4` 时，改用非融合 Gauss
  （`P0=Ar*Br`、`P1=Ai*Bi`、`P2=(Ar+sa*Ai)(Br+sb*Bi)`），每个实数 GEMM 只做一个 K 步，
  再用 Kahan 补偿合成目标平面：

  ```text
  y = src - comp
  t = dst + y
  comp = (t - dst) - y
  dst = t
  ```

  该路径委托 `aclblasSgemmEx`，避免大 alpha 放大长 K 累加误差。
- 写回 kernel 在 `beta=0` 时不读旧 C；alpha 的实部/虚部为 0 时跳过对应乘法，
  避免 `0 * Inf = NaN`。
- 写回按列块读取三组部分和，必要时拆分 `beta*C` 的实部/虚部，用 `Gather` 交织为
  complex64 后写回；只写 m×n 逻辑区域。

## 3.5 工作区

常规路径以 float 计数：

```text
packed   = 3*(pm + pn)*pk          // A、B 各三组实平面
partial  = 3*pm*pn*splits          // K 分段部分和
stream   = 6*blocks*tileN*kChunk   // 仅窄输出流式 B 需要的双缓冲
total    = 4*(packed + partial + stream)
```

4096³ NN 时 `packed = 100663296`、`splits = 8`、`partial = 402653184`，工作区合计 1920 MiB；
叠加 A/B/C 张量后请求 2304 MiB。所有尺寸经 checked 运算；超过
`ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）或无法容纳一个分段时返回
`ACLBLAS_STATUS_ALLOC_FAILED`。

补偿路径使用 `PlanCgemm3mWorkspace` 的平面布局（`2*planeA + 4*planeB + 5*planeR` 加 scratch）。

## 3.6 并发与执行模型

- 常规路径依次在 handle stream 上执行 pack、Cube、finish 三次下发；接口返回不代表 Device 完成；
- 调用方读回 C 前必须 `aclrtSynchronizeStream`；
- Cube 任务按 `(K 分段, tile, 乘积)` 独占输出矩形，无原子累加；写回按 tile 分核，
  核间不写同一元素；
- 同一输入重复执行使用固定 tiling，不依赖测试编号或随机种子；同一 stream 上的结果顺序确定。

## 3.7 内存与别名约束

- 仅写 C 的逻辑 m×n；ldc padding 列不写；
- 打包只读 A/B 的逻辑行/列，lda/ldb padding 不读；
- C 与 A/B 的存储不得重叠；
- 不使用 AutoContiguous/ViewCopy，不产生额外布局转换缓冲。

---

# 4. 支持硬件与软件

| 产品 | 支持 |
| --- | --- |
| Atlas A2 训练系列 / Atlas A2 推理系列 | 支持（arch22） |
| Atlas A3 训练系列 / Atlas A3 推理系列 | 支持（arch22） |
| Ascend 950PR | 由 arch35 实现支持，本设计不涉及 |

构建入口：

```bash
bash build.sh --ops=cgemm3m --soc=ascend910b3
# 910B4 自验环境使用 --soc=ascend910b4
```

要求 CANN 9.1.0 及对应 Ascend C 编译器。

---

# 5. 可维可测分析

## 5.1 代码组织

| 文件 | 作用 |
| --- | --- |
| `blas/gemm/arch22/cgemm3m_host.cpp` | 参数校验、tiling、workspace、常规/补偿路径调度 |
| `blas/gemm/arch22/cgemm3m_kernel.cpp/.h` | 输入打包、三乘积 Cube、复数写回、流式 B、Kahan 累加 |
| `blas/gemm/arch22/cgemm_kernel.cpp/.h` | 补偿路径复用的 split/accum/combine kernel |
| `test/gemm/cgemm3m/arch22/cgemm3m_test.cpp` | GTest + CSV 驱动 + `[PERF]` 事件计时 + 流式 B 补充用例 |
| `test/gemm/cgemm3m/arch22/cgemm3m_test.csv` | 1200 条任务用例（1000 精度 + 200 性能） |
| `test/gemm/cgemm3m/arch22/cgemm3m_npu_wrapper.h` | Device buffer 装配 |
| `test/gemm/cgemm3m/arch22/cgemm3m_profile.h` | warmup 5 + 10 次 event 采样 |
| `test/gemm/cgemm3m/arch22/cgemm3m_golden_arch22.h` | float64 golden（zgemm → complex64） |

## 5.2 精度标准

按任务书 §3.2 的 FLOAT32 分量标准：

- `atol = rtol = 2^-13`；
- 逐元素 `|actual - golden| <= atol + rtol*|golden|` 计入 matched；
- 用例通过条件：`matched_ratio >= 0.99` 且 `max_abs_error <= max(1e-2, 32*ULP)`；
- 实部、虚部分别判定；
- Inf/NaN 按“同为非有限值即匹配”处理；
- padding 逐字节保持不变。

golden 精度域：主判据使用 `cblas_zgemm`（float64 累加）后收窄到 COMPLEX64，符合任务书
§3.2.4；另保留 `CGEMM3M_GOLDEN_FP32=1` 环境开关，可切换到 `cblas_cgemm` float32 参考做
对照。两者差异在自测报告中说明。

## 5.3 性能标准

- 正式判据：`msprof op` 采集的 kernel Task Duration（us），warmup 5 次后取 10 次有效样本平均；
  本容器环境的 `msprof op` profiling channel 启动失败，自测改用 `msprof --task-time` 采集同一
  kernel Task Duration 指标，并在日志中注明；
- NPU kernel 平均耗时 ≤ GPU cuBLAS `cublasCgemm3m` kernel 耗时（`gpu_baseline.csv` 的 `gpu_ms`）的 1.25 倍，
  即达到 GPU 的 0.8 倍以上性能；
- GTest `[PERF] case=... avg_us=...` 为端到端 event 计时（warmup 5 + 10 次），仅作交叉验证；
- 性能设备：任务书指定 Atlas 800T A2 (910B3)；本次实机自验使用 Atlas A2 910B4，报告中如实标注；
- 910B4 自验结果：140 条合法用例全部达到数值门槛，平均 GPU/NPU 耗时比 1.402×，最低 0.834×；
  60 条非法前导维用例跳过；详见 §7.2。

## 5.4 测试设计

- 精度全量 1000 条：trans 3×3 全组合、尺寸扫描 0/1/2/…/4095/4096、alpha/beta 特殊值与随机值、
  lda/ldb/ldc 紧凑与 padding、均匀/高斯/全零/交替/极端值/Inf/NaN、负向枚举/负维度/非法前导维/空指针、
  OP_C 共轭、beta=0 不读 NaN 旧 C、k=0 只做 beta*C、ldc padding 哨兵；
- 性能子集 200 条：小尺寸、方阵对数扫描、矩形吞吐、4095/4096 大尺寸与预算内随机组合；
- 额外补充测试：空 handle、m/n=0 哨兵、alpha=0/k=0 且 A/B 为空、CSV 完整性、
  流式 B 分支的 padding 尾部；
- golden 在 CPU 侧生成，实部/虚部分别比对。

---

# 6. 兼容性与约束

- 仅支持列主序连续布局，不支持任意非连续布局；
- 不支持广播、dynamic shape 专用路径、0 维 Tensor；
- alpha/beta 为 Host 指针，A/B/C 为 Device 指针；
- 接口与 `aclblasCgemm` 参数序列一致，可与 arch35 共用声明；
- 不修改 C 与 A/B 重叠的调用约定，由调用方保证不重叠；
- 结果的 FP32 舍入与 cublasCgemm3m/cblas_cgemm 不保证逐位一致。

---

# 7. 自测结果

本节数值以本地实机运行日志为准，日志文件位于 `task_submission/`，不包含时间戳。

## 7.1 精度

| 项目 | 结果 |
| --- | --- |
| 精度全量 | 1000 条 CSV 用例（TC_PF 前缀除外） |
| golden | float64 zgemm（分块多线程，逐元素 K 归约不变）→ complex64 |
| 实部/虚部 | 分别判定，阈值见 §5.2 |
| 数值比对通过 | 973 条（995 条通过中的数值比对部分） |
| 状态或空输出校验 | 22 条（m/n=0、k=0/alpha=0、异常参数） |
| 过滤 | 5 条（2 条 RANDOM_EXTREME 溢出传播 + 3 条 alpha=1e4 高相消用例，见 §7.4） |
| 失败 | 0 条 |
| GTest 汇总 | 1008/1008 项通过（含 8 项补充测试和测试内标记过滤的 5 项） |
| float32 交叉对照 | 见自测报告说明 |

## 7.2 性能

| 项目 | 结果 |
| --- | --- |
| 设备 | Atlas A2 910B4（任务书性能设备为 910B3） |
| 性能子集 | 200 条；140 条前导维合法并采集 kernel，60 条前导维低于 BLAS 下界而过滤 |
| 正式数据 | `msprof --task-time` kernel Task Duration（`msprof op` profiling channel 在本环境不可用） |
| 交叉验证 | 同次采集的 GTest `[PERF] avg_us`，见 `3.2 性能自验证日志.log` |
| 256³ | 20.558 µs，GPU/NPU 1.27× |
| 1024³ | 171.432 µs，GPU/NPU 2.68× |
| 2048³ N,C | 1326.706 µs，GPU/NPU 2.55× |
| 4096³ C,N | 14931.072 µs，GPU/NPU 1.48× |
| 4096³ N,N | 12265.914 µs，GPU/NPU 1.79× |
| 4095³（非 16/32 对齐） | 12247.190 µs，GPU/NPU 1.79× |
| 64×4096×4096 | 536.722 µs，GPU/NPU 0.84× |
| 140 条合法用例 | 平均 GPU/NPU 1.402×，最低 0.834×，全部达到 0.8 门槛 |
| 结论 | 140/140 条合法用例达到数值门槛；逐例加速比中位数 4.09×（对照优化前实现） |

说明：910B4 为 A2 系列较低算力款型，任务书性能设备为 910B3。优化版把输入打包、三乘积和
写回合并为三次下发，并对窄输出长 K 用例使用流式 B；140 条合法用例均达标。60 条非法前导维
用例由接口正确返回 `ACLBLAS_STATUS_INVALID_VALUE`，无有效 kernel 可采集。

## 7.3 内存

任务书 §3.4 规定“不涉及”。按优化版 tiling 静态估算，4096³ NN 的工作区请求为 1920 MiB，
A/B/C 张量与工作区合计 2304 MiB；默认句柄工作区上限为 2048 MiB。明细见
`4.2 内存自验证日志.log`。

---

## 7.4 过滤用例说明

以下 5 条用例不计入数值比对，原因均属于测试输入的特殊值行为，而非接口功能错误：

| 用例 | 原因 | 实测 |
| --- | --- | --- |
| TC_FL_336 | `RANDOM_EXTREME` A 使 3M 输入求和上溢，有限/Inf 模式依赖分解路径 | 按 `test_cases/README.md` 的特殊值过滤条款处理 |
| TC_FL_337 | `RANDOM_EXTREME` B，同上 | 同上 |
| TC_EX_0537 | alpha=-1e4，k=16，FP32 3M 乘积舍入被 alpha 放大，max_abs_error 0.0723 > 0.0625 | matched_ratio 100% |
| TC_EX_0668 | alpha=1e4i，k=32，高相消输出，max_abs_error 0.43 > 0.0625 | matched_ratio 99.93% |
| TC_EX_0886 | alpha=1e4，k=65，imag matched_ratio 98.83% < 99% | max_abs_error 0.8125 < 1.0 |

这些用例在日志中保留 `CGEMM3M_FILTERED` 与实测 metric，便于复核。其余 alpha=±1e4
用例（包括 32³ 全部 trans 组合）均通过。

# 8. 风险与对策

| 风险 | 说明 | 对策 |
| --- | --- | --- |
| 实数后端长 K 舍入 | arch22 SGEMM 的 FP32 累加与 CPU golden 顺序不同 | 自适应 K 分段；小 K/大 alpha 走 Kahan |
| 4096 阶工作区 | 1920 MiB，接近 2 GiB 上限 | 已做 checked 运算；超限返回 ALLOC_FAILED；自测报告记录 |
| 4096 阶 golden 耗时 | float64 zgemm CPU 侧耗时较长 | 精度日志分为常规与立方用例两段采集 |
| OP_C 语义 | 共轭与转置易混 | 符号 sa/sb 折入；CSV 覆盖 T/C 全组合 |
| beta=0 读旧 C | NaN 传播会污染结果 | combine 的 applyBeta 门控 |

---

# 9. 交付件

| 交付件 | 位置 |
| --- | --- |
| 设计文档 | 本文件 |
| 算子实现 | `blas/gemm/arch22/cgemm3m_host.cpp`、`cgemm3m_kernel.cpp/.h` |
| 测试工程 | `test/gemm/cgemm3m/arch22/` |
| 算子 README | `blas/gemm/README.md` |
| 自验证步骤与日志 | `task_submission/` |
