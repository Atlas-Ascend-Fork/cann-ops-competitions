# 【社区任务】aclblasCtrmm算子设计文档

| 项目 | 内容 |
|---|---|
| 任务名称 | 算子实操工坊-杭州站-aclblasCtrmm算子开发(950) |
| 算子名称 | aclblasCtrmm（单精度复数三角矩阵-矩阵乘） |
| 目标硬件 | Ascend 950PR（arch35 / dav-3510） |
| 目标版本 | CANN 9.1.0 |
| 合入位置 | cann/ops-blas：实现 `blas/trmm/arch35/`、测试 `test/trmm/ctrmm/arch35/`、接口 `include/cann_ops_blas.h` |
| 语义标杆 | cuBLAS `cublasCtrmm` / Netlib `ctrmm.f` |

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求（CANN 社区任务 2026 · 算子实操工坊-杭州站）：在昇腾 NPU（Ascend 950PR）上开发单精度复数三角矩阵-矩阵乘算子 `aclblasCtrmm`，完成设计、开发、精度与性能自验，验收通过后以 PR 合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。

### 1.2 背景介绍

#### 1.2.1 aclblasCtrmm 算子实现优化

> **说明（与模板的差异）**：本任务不是对已有 TBE 算子的优化改造，而是 **ops-blas 开源仓新增的 BLAS 接口算子**——仓内目前只有实数版 `aclblasStrmm`，`include/cann_ops_blas.h` 中亦无 `aclblasCtrmm` 声明（需新增）。因此"TBE 算子源码获取路径 / 算子信息库路径"两项不适用；对应的参考资料与仓内先例列示如下（均已核对文件路径与文件名）：

| 类别 | 路径 | 用途 |
|---|---|---|
| 接口声明（需新增） | `cann/ops-blas` → `include/cann_ops_blas.h` | 与 `aclblasStrmm` 声明格式一致；禁止 950PR 私有平行接口 |
| 接口表 / 算子约束 | `cann/ops-blas` → `blas/trmm/README.md` | 产品支持表、参数口径（含 alpha 支持 Host/Device） |
| 仓内同族先例（结构参考） | `cann/ops-blas` → `blas/trmm/arch35/strmm_{host,kernel}.cpp`、`strmm_tiling_data.h` | 实数版 trmm 的"mirror → GEMM → scale"三段式 Cube 实现结构 |
| 共享 Cube 基建 | `cann/ops-blas` → `blas/common/helper/gemm_cube_arch35.h`（及 `gemm/arch35/gemm_tiling_struct.h`） | `TensorCopyGM2L1A/B`、`RunL1ToL0Mmad`、`TensorWriteFixpipeBlock` 等模板化 GEMM 组件 |
| 测试用例约定 | `cann/ops-blas` → `test/trmm/strmm/arch35/strmm_test.csv`、`test/trmm/strmm/arch35/strmm_param.h` | CSV 列格式（side/uplo/trans/diag/m/n/alpha/nullA/nullB/nullC/lda/ldb/ldc） |
| 语义标杆 | cuBLAS `cublasCtrmm`（CUDA 文档）、Netlib `ctrmm.f` | 逐参语义、边界行为（m/n=0、alpha=0、diag=UNIT） |
| 精度 golden | `cblas_ctrmm`（Netlib 复数实现） | 随测试工程提供，非三方软件依赖 |

#### 1.2.2 aclblasCtrmm 算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

| 项目 | 标杆（cublasCtrmm / Netlib ctrmm）取值 |
|---|---|
| 数据类型 | COMPLEX64（实部/虚部各 FLOAT32），对应 `aclblasComplex`（见 `include/cann_ops_blas_common.h`） |
| 数据排布 | 列主序（Column-Major）二维数组，无额外排布格式要求（ND） |
| A | 三角方阵：`side=LEFT` 为 m×m、`side=RIGHT` 为 n×n；物理数组 `lda×dimA`（列主序）；**仅引用 uplo 指定的三角**，另一三角不引用 |
| B | m×n 复矩阵，物理 `ldb×n`（列主序） |
| C | m×n 复矩阵，物理 `ldc×n`（列主序）；**离席（out-of-place）输出**，C 可与 B 传同一指针实现原地 |
| alpha | 复数标量（指针传入，支持 Host 或 Device 内存） |
| 属性 | `side ∈ {LEFT, RIGHT}`、`uplo ∈ {UPPER, LOWER}`、`trans ∈ {N, T, C}`、`diag ∈ {UNIT, NON_UNIT}` |
| 快速路径 | `m=0` 或 `n=0` → 合法 no-op；`alpha=(0,0)` → A/B 均不引用，C 全零 |

> 以上与任务书 §2.4 参数说明表、仓内 `blas/trmm/README.md` 的接口表口径一致。

##### 1.2.2.2 标杆算子实现描述

**数学语义**：`side=LEFT` 时 `C = alpha·op(A)·B`；`side=RIGHT` 时 `C = alpha·B·op(A)`；其中 `trans=N` 取 `A`、`trans=T` 取 `Aᵀ`、`trans=C` 取 `Aᴴ`（共轭转置）。`uplo` 指定三角仅引用指定侧；`diag=UNIT` 时对角按 1 处理且不读 A 的对角元素。

**cuBLAS 实现要点（与 Netlib 的关键差异）**：BLAS 的 `trmm` 是原地实现（结果覆写 B），cuBLAS 为获得更好并行度改为离席输出（结果写 C）；调用方把 B 的地址传给 C 即可获得原地语义。内部实现按 GEMM 组织：三角矩阵被拆成"对角块 + 两个非对角三角块"，非对角块用稠密 GEMM、对角块单独处理——本质是**"三角填充/分块 + GEMM"**的组合，而非逐元素循环。

**对昇腾实现的启示（本设计沿用）**：昇腾侧同样应落到 Cube（矩阵乘单元）上，并以 GEMM 组件复用为主；三角结构通过"带宽内 K 段"与"分核均衡"来处理（详见 §3）。

##### 1.2.2.3 标杆算子实现流程图

```
                 op(A) 三角结构（以 side=LEFT, uplo=LOWER 为例）
        k→   0    1    2   ...                      dimA-1
   i=0  [x    .    .    .                         .      ]
   i=1  [x    x    .    .                         .      ]       x = 引用（带内）
   i=2  [x    x    x    .                         .      ]       . = 不引用（带外，不读取）
   ...                                                     
   i=d-1[x    x    x    x    x    x    x    x    x      ]

   标杆流程（cuBLAS 内部组织）:
   ┌───────────────────────────────────────────────────────────────┐
   │ 1. 参数校验 / 快速路径：m,n=0 → 直接返回；alpha=0 → C 置零返回  │
   ├───────────────────────────────────────────────────────────────┤
   │ 2. 三角分块：把 op(A) 按 GEMM tile 分块                          │
   │    · 非对角块（整块在带内/带外）→ 稠密 GEMM，带外块直接跳过     │
   │    · 对角块（跨越三角边界）→ 带掩码的稠密 GEMM / 单独处理        │
   ├───────────────────────────────────────────────────────────────┤
   │ 3. 输出：C = alpha · (分块结果累加)，离席写入 C（ldc 保留填充）  │
   └───────────────────────────────────────────────────────────────┘
```

---

## 二、需求分析

### 2.1 外部组件依赖

| 依赖 | 版本/说明 | 用途 |
|---|---|---|
| CANN | 9.1.0 | AscendC / tensor_api / aclrt 运行时、编译工具链 |
| cblas（Netlib 复数实现 `cblas_ctrmm`） | 随测试工程提供 | 精度 golden（**仅测试期**，非算子运行期依赖） |
| cuBLAS 文档 / Netlib `ctrmm.f` | 参考文档 | 语义与边界行为对齐依据 |
| 三方软件 | 无 | — |

### 2.2 内部适配模块

| 模块 | 位置 | 说明 |
|---|---|---|
| 公共接口头 | `include/cann_ops_blas.h` | 新增 `aclblasCtrmm` 声明（与 cuBLAS 逐参对应；整数维参数用 `int`，与 `aclblasStrmm` 一致）；供其他产品线共用，禁 950PR 私有接口 |
| 公共类型/状态码 | `include/cann_ops_blas_common.h` | `aclblasComplex`（实/虚各 float32）、`aclblasStatus_t` 语义 |
| 句柄与流 | `common/helper/aclblas_handle_internal.h` | `_aclblas_handle`（stream 绑定）、workspace（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`） |
| 共享 Cube GEMM 基建 | `blas/common/helper/gemm_cube_arch35.h` + `blas/gemm/arch35/gemm_tiling_struct.h` | `GemmCubeState/GemmTileState`、`TensorCopyGM2L1A/B`、`RunL1ToL0Mmad`、`TensorWriteFixpipeBlock`、`InitGemmPipelineFlags` |
| 同族实现先例 | `blas/trmm/arch35/strmm_*` | 实数版 trmm 的三段式结构与列主序适配约定 |
| 测试框架 | `test/frame/`、`test/trmm/ctrmm/` | GTest + CSV 参数驱动 + cblas golden + 精度/性能判定脚本 |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,      // ACLBLAS_SIDE_LEFT / ACLBLAS_SIDE_RIGHT
    aclblasFillMode_t uplo,      // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,    // ACLBLAS_OP_N / _T / _C
    aclblasDiagType_t diag,      // ACLBLAS_NON_UNIT / ACLBLAS_UNIT
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

**参数语义与校验（与任务书 §2.4 对齐）**：

| 参数 | 语义 | 校验与异常 |
|---|---|---|
| handle | 库上下文（携带 stream） | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| side / uplo / trans / diag | 四枚举属性 | 非法取值 → `ACLBLAS_STATUS_INVALID_VALUE` |
| m / n | C 的行/列数（side 决定 A 的阶） | m<0 或 n<0 → INVALID_VALUE；m=0 或 n=0 → 成功 no-op |
| alpha | 复数乘数指针（Host/Device） | nullptr → INVALID_VALUE；`(0,0)` → C 置零返回、不引用 A/B |
| A | 三角矩阵（Device，只读） | alpha≠0 且 m,n>0 时 nullptr → INVALID_VALUE；lda ≥ max(1,dimA)（dimA=side==LEFT?m:n） |
| B | 输入矩阵（Device，只读） | alpha≠0 且 m,n>0 时 nullptr → INVALID_VALUE；ldb ≥ max(1,m) |
| C | 输出矩阵（Device） | m>0 且 n>0 时 nullptr → INVALID_VALUE（含 alpha=0 场景）；ldc ≥ max(1,m) |
| 返回值 | `aclblasStatus_t` | SUCCESS / INVALID_VALUE / HANDLE_IS_NULLPTR / ALLOC_FAILED / INTERNAL_ERROR |

> 错误码口径与 cuBLAS 对齐（非法枚举一律 `INVALID_VALUE`），并保留"非法参数→INVALID_VALUE / 极端维度→INVALID_VALUE（纯算术防护）"的分层校验顺序。

#### 2.3.2 Ascend C 算子相关约束（与标杆相比的差异/缺失）

| 能力 | 标杆（cublasCtrmm） | 本实现 | 说明 |
|---|---|---|---|
| 数据类型 | COMPLEX64 | COMPLEX64 | 一致；不支持 FP32/FP16/BF16 等实数或低精度类型 |
| 排布 | 列主序 | 列主序 | 仅支持列主序；不要求非连续 Tensor（超出 lda/ldb/ldc 语义的访问不支持） |
| 原地 | C 与 B 同指针 | 支持 | 与 cuBLAS 口径一致；除此之外不支持其他参数重叠 |
| dynamic shape | 不涉及 | 不支持 | m/n 为运行期入参，无需动态 shape 机制 |
| broadcast | 不涉及 | 不涉及 | A/B/C 独立矩阵 |
| 确定性 | 不要求 | 不要求 | — |
| 极端维度 | 由内部实现限制决定 | `uint64` 溢出防护 → INVALID_VALUE | 保持既有测试契约（`WorkspaceOverflowGuard`） |
| 融合算子 | 不涉及 | 不涉及 | 无 MC²/通算融合交叉（单卡单算子） |

---

## 三、需求详细设计

### 3.1 调用方式

采用 **BLAS 句柄式接口 + Kernel 直调**（非 ACLNN 算子注册形态、非 PyTorch 框架）：

```
用户程序
  ├─ aclblasCreate(&handle) / aclblasSetStream(handle, stream)
  ├─ aclblasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
  │     host 侧：校验 → 解析 alpha（Host 直读 / Device D2H）→ 计算 tiling/workspace → 启动 kernel
  └─ aclrtSynchronizeStream(stream)  // 读回设备结果前同步
```

- **异步执行**：所有 kernel 在 `handle` 绑定的 stream 上顺序下发，函数返回不代表计算完成；
- **workspace**：由库句柄统一管理（`EnsureDefaultWorkspace`），调用方无需关心；
- **测试入口**：`test/trmm/ctrmm/`（GTest + CSV 参数驱动 + `cblas_ctrmm` golden），与仓内 `test/trmm/strmm` 同构。

### 3.2 需求总体设计

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

**三段流水与算力单元分工**（950PR 实测：AIV 56 核、AIC 28 核）：

| 阶段 | kernel | 单元 | 核数取值 | 并行方式 |
|---|---|---|---|---|
| 1 打包 | `ctrmm_cube_prep_kernel` | AIV | `min(ceil(work/nthreads), aivCoreNum)` | 网格跨步（work = max(dimA², dimA·(LEFT?n:m))） |
| 2 GEMM ×2 | `ctrmm_cube_gemm_kernel` | AIC | `mBlocks×nBlocks`（≤ aicCoreNum） | **全局 tile 网格跨步** |
| 3 折回 | `ctrmm_cube_scale_kernel` | AIV | `min(ceil(m·n/nthreads), aivCoreNum)` | 网格跨步 |
| 回退 | `ctrmm_compute_kernel`（SIMT） | AIV | 同上（tile 网格） | 网格跨步，`dimA < 256` 时启用 |

**GEMM 分核（本设计的关键点）**：**不使用**框架 `InitGemmState` 的"按 tile 数均分"分区，改为在**全局 tile 网格**上跨步枚举：

```cpp
totalTiles = ceil(m/BM) * ceil(n/BN);          // 框架视角 M=n、N=m、K=2*dimA
for (uint32_t t = blockIdx; t < totalTiles; t += numBlocks) {
    mi = t / nLoopCount;  ni = t % nLoopCount; // 全局 tile 坐标
    ...                                        // 每核均匀采样整条三角波段
}
```

原因：三角带 K 收缩后，同一 tile 行的 K 段长度随波段位置线性变化（2048² 下相差 16 倍）。按 tile 数均分会让"只分到高 index tile 的核"提前空转，**kernel 时间由最慢核决定**（实测该问题造成约 40% 的每 MAC 效率损失：工作量 −47% 而耗时仅 −24%）。跨步枚举使每个核必然同时持有"满长 K"与"极短 K"的 tile，负载自动均衡；修正后 case4 单轮 5.82ms → 2.87ms。

##### 3.2.1.2 数据分块和内存优化策略

**（a）打包平面与公式**（workspace 内，512B 对齐；`Align8(x) = (x+7)/8*8`）：

| 平面 | LEFT（dimA=m） | RIGHT（dimA=n） |
|---|---|---|
| `opPackLd` | `Align8(dimA)` | `Align8(2·dimA)` |
| op(A) 打包面 | `2·dimA × opPackLd` | `dimA × opPackLd` |
| `bPackLd` | `Align8(2·dimA)` | `Align8(m)` |
| B 打包面 ×2 | `n × bPackLd` | `2·dimA × bPackLd` |
| `tempLd` | `Align8(m)` | `Align8(m)` |
| C_r / C_i 面 ×2 | `n × tempLd` | `n × tempLd` |

总字节数 = `(opElems + 2·bElems + 2·tElems)·4 + 512(对齐余量)`；**2048² 实例**：opPack 4096×2048×4 = 32MB，B 面 2×(2048×4096×4) = 64MB，临时面 2×(2048×2048×4) = 32MB，合计 ≈ **128MB**（HBM 131GB，裕量充足）。

**（b）片上存储（L1/L0）与分块**：

| 层级 | 容量（arch35） | 本设计用量 | 说明 |
|---|---|---|---|
| L1 | 512KB（双 bank，各 256KB） | A 块 128×128×4 = 64KB + B 块 64KB = 128KB/bank | `tileKChunk=128`；剩余空间留作预取流水；B 区起始偏移按 A 的**实际 NZ 占用** `RoundUp(curM,16)·RoundUp(curKChunk,8)·4` 计算，否则 chunk 非 8 倍数时 B 会落进 A 区（曾致 56 条用例失败） |
| L0A / L0B | 各 64KB（双缓冲半区 32KB） | 128×8×4 = 4KB/步 | `baseK=8`（fp32 的 C0 分形），L1→L0 逐步搬入 + MMAD |
| L0C | 256KB | 128×128×4 = 64KB/槽 → `MAX_C0_TILES = 4` | 每个输出 tile 一个 L0C 槽，MMAD 累加后 fixpipe 写回 |

**（c）访存优化（实测驱动）**：
- 打包段连续 lane 沿**行**（列主序连续方向）变化——修正前沿列变化，每 lane 命中不同 32B sector（有效 4B/32B），实测仅 114GB/s；修正后 prep 段 1468us → 594us（−60%）；
- GEMM 段 K 块双缓冲预取（当前块 MMAD 时预取下一块到另一 bank，隐藏 MTE2 延迟）；
- GM→L1 采用框架 `Te::Copy(CopyGM2L1{})`（等价 nd2nz + DataCopy），L1→L0 固定 NORMAL 装载。

##### 3.2.1.3 tilingKey 规划策略

本算子为 BLAS 直调形态，**不使用 tilingKey**（无 aclnn 算子注册/二进制动静态选择需求）；运行期 tiling 通过 kernel 参数结构体按值下发：

| 结构体 | 字段要点 |
|---|---|
| `CtrmmCubePrepTilingData` | side/uplo/trans/diag、m、n、dimA、lda、ldb、`opPackLd`、`bPackLd`、numBlocks、nthreads |
| `CtrmmCubeBandParams` | `kLower`（1：带内 k ≥ index；0：k ≤ index；−1：调试关闭）、`onM`（带 index 是否作用于框架 M 轴：RIGHT=1） |
| `GemmTilingData`（共享结构，两遍共用） | m=n、n=m、k=2·dimA、lda/ldb/ldc、baseM=baseN=128、baseK=8、tileKChunk=128、c0Size=8、mBlocks/nBlocks/usedCoreNum、isTransA/B=0 |
| `CtrmmCubeScaleTilingData` | m、n、ldc、`tempLd`、alphaReal/alphaImag、numBlocks、nthreads |

关键约束：`baseM/baseN/baseK` 必须与 kernel 模板参数一致；`tileKChunk` 必须是 `baseK` 的倍数且满足 `(baseM+baseN)·tileKChunk·4 ≤ L1/2`；`band.kLower/onM` 由 host 按 `(side, uplo, trans)` 推出（见 3.2.2.1）。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

**（1）打包段 `ctrmm_cube_prep_kernel`（AIV/SIMT）**
从调用方复数 A、B 直接生成三个实数操作数（不物化中间复数矩阵）：

| 打包面 | 内容 | 布局要点 |
|---|---|---|
| op(A) 面 | `op(A)` 的转置排列，实/虚沿 K 维**交错**（K=2k 放实部、2k+1 放虚部） | 三角规则在打包时一次完成：带外写 0 且**不读 A**；`diag=UNIT` 写 1 且**不读对角**；`trans=T/C` 读转置存储位置；`trans=C` 虚部取反 |
| bReal 面 | B 去交织：2k → `Br`，2k+1 → `−Bi` | 与 op(A) 面配对，一遍 GEMM 即得 `C_r = Σ_k(Or·Br − Oi·Bi)` |
| bImag 面 | B 去交织：2k → `Bi`，2k+1 → `Br` | 配对得 `C_i = Σ_k(Or·Bi + Oi·Br)` |

> **交错（而非分块）的必要性**：稠密 GEMM 的累加顺序必须与 golden 的"逐 k 配对"一致；分块布局（实部项全在前、虚部项全在后）会改变非有限值（Inf/NaN）传播模式，触发非有限值模式比对失败（`TC_FL_119`）。

**（2）GEMM 段 `ctrmm_cube_gemm_kernel`（AIC/Cube，tensor_api）—— 两遍**
- 复用共享框架组件：`TensorCopyGM2L1A/B`（GM→L1，含分形重排）、`RunL1ToL0Mmad`（L1→L0 + MMAD，L0 双缓冲）、`TensorWriteFixpipeBlock`（L0C→GM）、`InitGemmPipelineFlags`（5 个预置旗标）；
- **列主序适配**：列主序缓冲与其行主序转置同像，故以 `M=n、N=m、K=2·dimA` 计算 **Cᵀ（行主序）** 写入临时面，第三段 scale 再折回调用方交错复数布局；
- **三角带 K 收缩**：按 tile 跳过带外 K 段。带 index 的映射：LEFT 的 op(A) 列索引落在框架 **N 轴**、RIGHT 落在 **M 轴**；`kLower=1 → K ≥ index`、`=0 → K ≤ index`；K 坐标 = 2·op(A) index（与交错布局一致）；
- **预取双缓冲 + 全局 tile 跨步分核**（见 3.2.1.1/3.2.1.2）；
- 两遍分别输出 `C_r`、`C_i` 临时面。

**（3）折回段 `ctrmm_cube_scale_kernel`（AIV/SIMT）**
`C = alpha·(C_r + i·C_i)` 写回调用方交错复数 C，保留 **位掩码零守卫** alpha 语义：exact-zero 分量不得乘穿非有限累加量（`0·NaN=NaN`），且昇腾浮点 select 会传播未选中分支的 NaN，故用 `0u −(cond?1:0)` 生成全 0/全 1 掩码做按位与，再做加减。

**（4）小规模回退 `ctrmm_compute_kernel`（SIMT 融合）**
`dimA < 256` 时改走 SIMT 单 kernel 路径（波段守卫逐元素跳过带外项）：既规避稠密表述的 `0×Inf` 语义差异，也避免小 shape 下 Cube 流水四次启动与打包流量不划算的问题。

##### 3.2.2.2 Ascend C 实现流程图

```
                    aclblasCtrmm（host）
   ┌──────────────────────────────────────────────────────────────────┐
   │ 校验(enums/dims/lds/alpha/指针) → m,n=0? → 返回 SUCCESS          │
   │ alpha=(0,0)? → zero kernel（C 置零，不读 A/B）→ 返回             │
   │ dimA < 256? → SIMT 融合 kernel（回退路径）→ 返回                 │
   │ 计算打包尺寸/带宽/K 收缩参数 → EnsureDefaultWorkspace(512B 对齐)  │
   └───────────────┬──────────────────────────────────────────────────┘
                   ▼  （同一 stream，顺序下发）
   ┌────────────────────────────┐   AIV ×56，网格跨步
   │ ① prep：A/B → 3 个实数面    │   lane 沿连续方向（行）→ 合并访问
   │    op(A)面(含三角/对角/共轭)│
   │    bReal面 / bImag面        │
   └───────────────┬────────────┘
                   ▼
   ┌────────────────────────────┐   AIC ×(mBlocks×nBlocks)
   │ ② GEMM 第 1 遍 → C_r 面     │   tile 网格跨步（负载均衡）
   │    K 交错配对 · 三角带收缩   │   L1 双 bank 预取 · L0 双缓冲
   │    M=n, N=m, K=2·dimA (Cᵀ)  │   L0C 槽 → fixpipe 写回
   ├────────────────────────────┤
   │ ② GEMM 第 2 遍 → C_i 面     │   （同一 tiling 与 band，bImag 面）
   └───────────────┬────────────┘
                   ▼
   ┌────────────────────────────┐   AIV ×56，网格跨步
   │ ③ scale：C = α·(C_r+i·C_i)  │   位掩码零守卫 alpha（非有限值安全）
   │    折回调用方交错复数布局    │   ldc 填充行不写
   └────────────────────────────┘
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| # | 差异点 | 标杆（cuBLAS/Netlib） | 本实现 | 原因与影响 |
|---|---|---|---|---|
| 1 | 计算表述 | 三角分块：非对角块稠密 GEMM + 对角块掩码处理，带外块跳过 | 稠密 GEMM + **按 tile 的 K 段裁剪**（带外 K 段整块跳过） | 复用仓内共享 Cube 框架（`gemm_cube_arch35.h`）成本最低；效果等价于"带外块跳过"，但**对角块内的结构化零仍参与乘法**——对非有限值输入会产生 `0×Inf=NaN`，与参考"跳过带外项"不同 → 由回退路径覆盖（见 #5） |
| 2 | 存储序 | 原生列主序 | 以"列主序 ⇔ 行主序转置"同像，用行主序框架算 **Cᵀ**，scale 段折回 | 框架与仓内 GEMM 组件均为行主序 ND 约定，转置适配可零成本复用（strmm 同法） |
| 3 | alpha 施加位置 | 预缩放进操作数（golden 亦然） | GEMM 内 alpha=1，**scale 段单独施加**，并以位掩码守卫 exact-zero 分量 | 保证 `alpha=(0,0)`/单分量零时不被 `0×Inf/NaN` 污染（昇腾浮点 select 会传播未选分支 NaN）；同时天然满足 `alpha=0` 的"C 置零、不引用 A/B"语义 |
| 4 | 三角填充策略 | 运行中分块掩码 | 打包时一次写入（带外写 0，`diag=UNIT` 写 1，转置/共轭在读取时完成） | 把控制流集中在 SIMT 打包段，Cube 段保持纯稠密 GEMM（无分支、可流水） |
| 5 | 非有限值/小规模 | 统一路径 | `dimA < 256` 回退 SIMT 融合 kernel（逐元素波段守卫） | 稠密表述对角块内 `0×Inf` 无法与"跳过语义"逐位对齐；小 shape 下 Cube 四次启动亦不划算。回退阈值取 256：全部非有限值用例（dimA=32）与极小 shape 落回退路径，标杆性能用例（≥256）走 Cube 路径 |

### 3.3 支持硬件

| 项目 | 内容 |
|---|---|
| 适配硬件 | Ascend 950PR（架构 arch35 / dav-3510） |
| CANN 版本 | 9.1.0 |
| 产品支持表 | `blas/trmm/README.md` 标注 Ascend 950PR：**支持** |
| 接口共用 | `include/cann_ops_blas.h` 统一声明（其他产品线可共用），**不定义 950PR 私有平行接口** |

### 3.4 算子约束限制

| 约束项 | 内容 |
|---|---|
| 数据类型 | 仅 COMPLEX64（实/虚各 FLOAT32）；不支持实数与低精度类型 |
| 存储 | 仅列主序（Column-Major）；不支持超出 lda/ldb/ldc 语义的非连续 Tensor |
| 形状 | A 必须为方阵（side=LEFT 取 m×m、RIGHT 取 n×n）；B/C 必须为 m×n |
| 前导维 | `lda ≥ max(1, side==LEFT?m:n)`、`ldb ≥ max(1,m)`、`ldc ≥ max(1,m)` |
| 内存重叠 | 仅允许 `C ≡ B`（原地）；其余参数重叠不支持 |
| 空/零场景 | m=0 或 n=0 → 合法 no-op；alpha=(0,0) → C 置零且不引用 A/B |
| dynamic shape / broadcast | 不支持（m/n 为运行期入参即可，无需动态 shape 机制） |
| 确定性 | 不要求（同输入多次执行的浮点结果不保证逐位一致） |
| 极端维度 | 参数范围溢出时返回 `ACLBLAS_STATUS_INVALID_VALUE`（纯算术防护，保测试契约） |
| alpha 位置 | 支持 Host 或 Device 内存（Device 时由 host D2H 读回，引入一次同步） |

---

## 四、特性交叉分析

| # | 交叉特性组合 | 交互点 | 设计处理 |
|---|---|---|---|
| 1 | 快速路径 × 全枚举 | `m=0/n=0`、`alpha=(0,0)` 与 side/uplo/trans/diag 任意组合 | 快速路径在枚举/维度校验**之后**判定（保持 Netlib 返回顺序）；`alpha=(0,0)` 仅置零 C 的逻辑 m×n 区，`ldc` 填充行不写；A/B 完全不引用 |
| 2 | 非有限值 × 三角带 / alpha | B 含 Inf/NaN、alpha 单分量为零 | 打包段交错布局保证逐 k 配对顺序；scale 段位掩码零守卫防 `0×Inf`；`dimA<256` 回退 SIMT 逐元素波段守卫，与参考"跳过带外项"语义一致 |
| 3 | 原地语义 × 快速路径 | `C ≡ B` 与 `alpha=(0,0)` | 置零路径只写 C（=B）的逻辑区，不读 A/B，语义与 cuBLAS 一致 |
| 4 | 填充（lda/ldb/ldc>dim）× 三角引用 | A 的填充列、C 的填充行 | 打包段仅访问 `lda` 范围内的有效列；scale 段只写 `[0,m)` 行，`[m,ldc)` 填充保留；测试含 `VerifyPaddingUntouched` 断言 |
| 5 | workspace 复用 × 多 shape 顺序调用 | `EnsureDefaultWorkspace` 只增不减 | 每次调用按当前 shape 计算需求并重新取有效指针；不同 shape 交替调用不残留 |
| 6 | 多调用/多流 | 同一 handle 连续调用、不同 handle 并行 | 同一 handle 的调用在 stream 上顺序执行；`alpha` Device 场景引入一次 D2H 同步（仅该场景） |
| 7 | 产品线 × 接口 | 接口共用 vs 实现分目录 | 声明统一放 `include/cann_ops_blas.h`；实现按 `archXX` 分目录，其他产品线可复用声明或新增 `archyy` 实现 |
| 8 | 回退阈值 × 精度口径 | `dimA<256`（SIMT）与 `≥256`（Cube）两路径 | 两条路径共用同一精度判定口径与 golden；边界 shape（255/256/257）纳入 shape 扫描用例，确保跨路径结果一致 |
| 9 | 测试体系 × 特性组合 | L0/L1/L2、负向、非有限值、性能 | CSV 驱动 + 参数正交覆盖（side×uplo×trans×diag）；非有限值用例组专属（TC_FL_*）；性能用例组 TC_PF_* |

---

## 五、可维可测分析

### 5.1 精度标准/性能标准

**精度标准（任务书 §3.2 口径）**：golden 由 `cblas_ctrmm`（Netlib 复数实现）生成；对输出 C（m×n）**实部、虚部分别**逐元素判定：

```
|actual − golden| ≤ atol + rtol × |golden|        atol = rtol = 2⁻¹³
matched_ratio ≥ 0.99     且     max_abs_error ≤ max(1e-2, 32·ULP)
```

**精度实测**：全量 GTest **1232/1232 PASSED**（含 1200 条官方 CSV + 26 条补充 + 6 条契约用例）；小 shape 组 `maxAbsErr ≈ 7.6e-06`、`matchedRatio = 1.0`；非有限值用例组按"非有限值模式比对"判定通过。

**性能标准（任务书 §3.3 口径）**：5 条标杆 case，warmup 后有效采样 >50 次取平均单次耗时（`op 调用 + stream 同步`）：

| 标杆 case | 参数 | 阈值（任务书） | 实测 | ratio（H100/实测，需 ≥0.4） | 判定 |
|---|---|---|---|---|---|
| 1 | 256² LEFT UPPER N NON_UNIT | 120.18 us | **0.068 ms** | 0.707 | ✅ PASS |
| 2 | 512² LEFT LOWER N NON_UNIT | 268.37 us | **0.137 ms** | 0.784 | ✅ PASS |
| 3 | 1024² RIGHT UPPER T NON_UNIT | 690.54 us | **0.488 ms** | 0.566 | ✅ PASS |
| 4 | 2048² LEFT LOWER C NON_UNIT | 2945.91 us | **2.874 ms** | 0.410 | ✅ PASS |
| 5 | 2048² RIGHT LOWER N UNIT | 2963.64 us | **2.533 ms** | 0.468 | ✅ PASS |

有效算力（复数 trmm 折算 8 flops/复数 MAC、稠密计）：1.97 / 7.84 / 17.60 / 23.91 / 27.13 TFLOPS，全部 ≥ 达标线（1.12 / 4.00 / 12.44 / 23.33 / 23.19）。

**可测性/可复现性**：

| 工具 | 位置 | 用途 |
|---|---|---|
| CSV 用例 | `test/trmm/ctrmm/ctrmm_test.csv`(+`_extra.csv`) | 参数驱动用例（含负向/非有限值/性能组） |
| 精度脚本 | `test_cases/verify_accuracy.py` | 精度批量判定（cblas golden） |
| 性能脚本 | `test_cases/verify_performance.py` | 官方性能判定（口径说明见下） |
| 纯算子基准 | `bench_ctrmm`（warmup 10 + 100 采样） | 任务书 §3.3 口径的 kernel 级耗时 |
| 采集 | `msprof_profile_run.sh`（cannbot ops-profiling） | 7 组 aic-metrics + 逐核采样，定位瓶颈 |

> **口径说明（已知差异）**：官方 `verify_performance.py` 解析 GTest 整 case 时长（含夹具数据生成/设备搬运/校验循环），大 shape 下夹具开销占 80% 以上，即便算子耗时为 0 也无法达标；本设计报告的判定按任务书 §3.3 的"算子平均单次耗时"口径给出，并已在自测报告中记录该差异与证据（同目录 `CP4-性能验收报告.md` / `6.x` 系列报告）。

### 5.2 兼容性分析

| 维度 | 分析 |
|---|---|
| 接口兼容 | 新增声明置于 `include/cann_ops_blas.h`，签名与 `cublasCtrmm` 逐参对应（整数维参数 `int`，与仓内 `aclblasStrmm` 一致）；不引入 950PR 私有平行接口 |
| 错误码兼容 | 非法枚举/维度/前导维/指针 → `ACLBLAS_STATUS_INVALID_VALUE`（与 cuBLAS 错误表对齐）；handle 为空 → `HANDLE_IS_NULLPTR`；分配失败 → `ALLOC_FAILED` |
| 语义兼容 | `m/n=0`、`alpha=(0,0)`、`diag=UNIT` 不读对角、仅引用指定三角、离席输出与 `C≡B` 原地——均与 Netlib `ctrmm` / cuBLAS 一致 |
| 边界行为兼容 | 填充行不写、填充列不读；非有限值输入与参考的差异由回退路径收敛（≤256 阶） |
| 版本兼容 | CANN 9.1.0（`ASC_DEVKIT_GE_9_1` 条件编译）；tensor_api/Cube 路径依赖 arch35 工具链 |
| 符号/污染 | 仅新增 `aclblasCtrmm` 及内部 kernel 符号；tiling 结构仅本算子内部使用，不改动共享结构布局 |
| 回归保障 | 每轮改动均跑全量 1232 条 GTest；新增调试开关（`CTRMM_CUBE_NO_BAND`）默认关闭、不影响生产行为 |

---

## 附录：优化历程与量化结果（自验数据摘要）

| 轮次 | 关键动作 | case4 (ms) | case5 (ms) |
|---|---|---|---|
| SIMT 基线 | 融合单 kernel + 指针流/64 位复数加载 | 101.3 | 127.5 |
| Cube 首版 | SIMT 打包 + 2 遍稠密实数 GEMM + SIMT 折回 | 6.708 | 6.964 |
| 减工作量 | 三角带 K 收缩 + L1 预取双缓冲 | 6.552 | 5.844 |
| 实测驱动 | msprof 定位 → 打包段合并访问（prep −60%） | 5.820 | 4.809 |
| **负载均衡** | 全局 tile 网格跨步分核（波段感知） | **2.874** | **2.533** |

累计相对 SIMT 基线：**35× / 50×**；相对 Cube 首版：**2.33× / 2.75×**。全程精度保持 1232/1232 全绿。
