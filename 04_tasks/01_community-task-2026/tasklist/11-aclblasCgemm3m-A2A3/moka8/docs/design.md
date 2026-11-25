# aclblasCgemm3m 算子设计（Atlas A2/A3）

## 1 需求背景

### 1.1 需求来源

11 月 CANN 社区任务「aclblasCgemm3m 算子开发（A2/A3）」要求在 `ops-blas` 的 `arch22` 上补齐单精度复数三乘法 GEMM。沿用现有公开接口和 BLAS handle，通过绑定的 stream 直接下发 Ascend C 内核。数学接口对齐 cuBLAS `cublasCgemm3m`，软件基线和验收判据见第 4 节。

### 1.2 上游现状

调研基线为 `ops-blas` 的 `55ef791b77f3677b45da3468b0a8b17941424f84`。

| 文件 | 现状及设计引用 |
| --- | --- |
| `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` | 已声明 `aclblasCgemm3m` 及 `aclblasComplex`，沿用现有 ABI、枚举和状态码。 |
| `blas/gemm/arch35/cgemm3m_host.cpp` | 已导出 `aclblasCgemm3m`，参考其复数缩放、三乘法路径和 workspace 组织。 |
| `blas/gemm/arch22/gemm_host.cpp`、`gemm_kernel.cpp` | 已有 `aclblasSgemm` 及 `sgemm_arch22_matmul_tiles_kernel`，可参考 arch22 的 FP32 Cube、AIV 布局打包和输出回写。Cube 路径显式关闭 HF32。 |
| `test/gemm/cgemm3m/CMakeLists.txt` | 现有目标按 `arch35` 注册。增加 `arch22` 注册和复数 GTest。 |

[ops-blas 调研基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84)

## 2 需求分析

### 2.1 需求描述

计算 `C = alpha * op(A) * op(B) + beta * C`。逻辑形状分别为 `m×k`、`k×n`、`m×n`。矩阵是 Device 上实部、虚部交错的 `complex64`，按 BLAS 列主序存储。两个复数标量由 Host 指针传入，入口校验后复制标量值到内核参数。

| 访问 | `N` | `T` / `C` | 前导维下界 |
| --- | --- | --- | --- |
| `op(A)[i,p]` | `A[i+p*lda]` | `A[p+i*lda]` | `max(1,m)` 或 `max(1,k)` |
| `op(B)[p,j]` | `B[p+j*ldb]` | `B[j+p*ldb]` | `max(1,k)` 或 `max(1,n)` |
| `C[i,j]` | `C[i+j*ldc]` | — | `max(1,m)` |

`OP_C` 还需将读取到的虚部取负，`OP_T` 保留虚部。前导维可大于下界。每列仅写 `C` 的前 `m` 个元素。调用方提供足够长的缓冲并保证 `C` 与输入矩阵的读区互不重叠。

### 2.2 需求拆解

| 任务书 | 设计要求 | 验证 |
| --- | --- | --- |
| §2.2–§2.5 | arch22 公共入口、列主序、九种转置组合、按 stream 异步运行 | 链接公开符号，验证转置、padding 及同一 stream 连续调用 |
| §2.4–§2.5 | 逐项校验参数、空指针和退化场景 | 独立 GTest 校验状态码、原地输出及读写边界 |
| §3.1–§3.2 | 复数实部、虚部分别满足 FLOAT32 混合容差 | 1000 条精度用例和分量级比对日志 |
| §3.3、§3.5 | 采集 kernel 耗时与端到端耗时 | 200 条性能用例，保留 `msprof op` 与 GTest 两套原始数据 |
| §3.4、§4 | workspace 有界，记录峰值和内存测试适用性 | 覆盖外部 workspace 边界和 4095/4096 大尺寸，按验收模板填写 |

## 3 详细设计

### 3.1 算子分析

令 `Ar/Ai` 和 `Br/Bi` 为按上述索引及共轭规则得到的实数矩阵。分阶段路径计算

```text
P0 = Ar · Br
P1 = Ai · Bi
P2 = (Ar + Ai) · (Br + Bi)
real = P0 - P1
imag = (P2 - P0) - P1
```

融合路径计算

```text
P0 = (Ar + Ai) · Br
P1 = Ar · (Bi - Br)
P2 = Ai · (Br + Bi)
real = P0 - P2
imag = P0 + P1
```

其中 `·` 表示实数矩阵乘法。得到乘积后，在 FP32 中完成复数 `alpha/beta` 缩放并交错写回。两种分解各自验证舍入和抵消误差。常规路径使用 FP32 输入、累加和中间值，Cube 显式关闭 HF32。输出 dtype 为 `complex64`。测试包含 4095/4096，实际分块根据运行时维度和片上容量确定。

### 3.2 算子实现

#### 3.2.1 Host 侧设计

入口先判空 handle，再校验转置枚举、维度和前导维，最后验证 `alpha/beta` 和按路径必需的矩阵指针。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。按 `size_t` 或 64 位整数检验物理跨度、字节数和 workspace 偏移的溢出风险。所有 kernel 在 handle 的当前 stream 上异步下发。

| 路径 | 条件 | 操作 |
| --- | --- | --- |
| 空输出 | `m=0` 或 `n=0` | 允许 `A/B/C` 为空，校验结构参数后返回成功，无下发。 |
| 仅缩放 | `k=0` 或 `alpha=(0,0)` | 允许 `A/B` 为空，`C` 必须有效。若 `beta=(1,0)`，直接返回。若 `beta=(0,0)`，由 AIV 写零。其余情况执行复数乘法并原地写回。 |
| 常规乘积 | 其余合法输入 | 由 tiling 选定 AIV 小矩阵、分阶段 Cube 或融合 Cube 路径。 `beta=(0,0)` 时不读 `C` 旧值。 |

`Tiling` 记录原始 `m/n/k`、前导维、转置标志、块形状、尾块边界、参与核数、归约分片数及 workspace 偏移。分派按输出块并行度、K 长度、访存对齐和可用空间确定。阈值通过代表性方阵与长宽矩阵测量确定。首选沿 `m/n` 切块，使每块独立完成 K 归约。对输出块不足且 K 较长的形状，可将 K 分片，按固定顺序归约部分和。

TilingKey 区分缩放、小矩阵 AIV、分阶段 Cube 和融合 Cube。转置方向、是否读取 C 旧值和尾块大小由路径参数携带。每次按运行时核数限制 AIC 和 AIV 的任务并行度。

复用仓内有效 workspace 机制。用分别对齐的输入面板、乘积部分和与归约区构成单次调度的布局。每次提交前检查需要的字节数；空间不足时缩小同批处理的 tile 数或使用小空间路径。仅引用能覆盖此次调用的缓冲区，内核参数和临时区在 stream 消费完成前保持有效。同一 stream 的前后调用可依赖顺序复用；不同 stream 的并发调用必须使用彼此隔离且生命周期匹配的缓冲。

#### 3.2.2 Kernel 侧设计

- **小矩阵 AIV 路径**：按 C 的列与连续行分配 tile，载入 A/B 的复数片段，在 UB 内拆分、共轭并计算三组 FP32 乘加，随后就地合成和写回。首次选择由主要维度和预计的访存、下发成本决定，重点覆盖短 K 和小输出。
- **分阶段 Cube 路径**：AIV 将复杂索引整理为 Cube 可读取的实数面板，归一化转置和共轭，在预处理时形成和面板。Cube 按 `(乘积编号, C tile)` 分工，计算三组 FP32 乘积。AIV 读取同一逻辑块的三个结果，执行复数合成及 `alpha/beta` 尾部运算。各阶段由同一 stream 中的下发顺序建立依赖。
- **融合 Cube 路径**：为可容纳的 tile 复用解交错、符号调整、面板搬入和三组累加结果，减少中间结果落到 GM 的次数。在 arch22 上验证 Vector/Cube 的同步和 L1、L0、UB 容量后启用。每个 C tile 由一个写回任务拥有。采用 split-K 时，每个分片写独立暂存，最终任务归约后再应用 `alpha/beta`。不并发原子写同一个复数结果。
- **尾块与对齐**：打包仅在逻辑范围内读取 A/B，面板中补齐的 K 元素置零。连续块用批量搬运，非对齐头尾用平台支持的精确长度搬运或逐元素处理。检查相邻任务写边界，保证 `C` 的 padding 和相邻列不被覆盖。
- **数值保护**：重点检查接近抵消、长 K、极端幅值和大复数标量的分量误差。对中间和可能溢出的 tile 验证 2 的幂缩放策略。对检测到的高抵消或非有限中间量，按数值条件进入局部纠正重算，分别验证实虚部及 NaN/Inf 传播。重算的触发条件通过合成矩阵与官方用例校验，保持正常有限输入的三乘法主路径。

按 `4096³` 估算，若同时常驻 A/B 各三个 FP32 平面与三个输出平面，仅临时区就约 576 MiB。因此大矩阵按 tile 批次限制常驻的面板和中间结果，优先复用输入 tile。使用 profiling 观察三组乘积的 Cube 利用率、打包/尾处理耗时和 workspace 搬运，再调整 tile 与分核策略。上述容量和分派均为设计估算，性能结论留待上机验证。

### 3.3 支持硬件

| 产品 | 目标与验证 |
| --- | --- |
| Atlas A2（arch22） | 性能自验使用 Atlas 800T A2（910B3），同时运行精度与接口回归。 |
| Atlas A3（arch22） | 完成同一入口的构建适配。按任务书允许的自验覆盖范围记录实测产品和版本。 |

任务书环境为 CANN 9.1.0。开发机计划使用 CANN 9.2.1，并在自验记录中注明实际工具链和运行库版本。

### 3.4 算子约束限制

支持单矩阵 GEMM、Host 复数标量和 BLAS 前导维 padding。仅处理约定的列主序物理布局，无广播或 batch 参数。输入大小均为运行时参数，调用方在读取结果前同步绑定的 stream。浮点归约按路径和分片执行，验收采用混合容差。

## 4 可维可测分析

### 4.1 精度和性能标准

| 项目 | 判定 |
| --- | --- |
| 精度 | 以 Netlib BLAS 3.10.0 的参考语义生成 golden，可用 float64 累加后收窄，或用任务书允许的 float32 `cblas_cgemm`，在报告写明实际口径。实部和虚部分别使用 `atol=rtol=2^-13`，匹配率 ≥99%，最大误差按任务书 §3.2 的 `1e-2` 或 `32 ULP` 档位判定。 |
| 性能 | 每例 warmup 5 次，取后续至少 10 次有效执行的均值。910B3 以 `msprof op` 的 kernel Task Duration（µs）和同参 cuBLAS A100 `gpu_ms` 比较，`gpu_ms/(npu_us/1000) ≥ 0.8`。同时保存 `aclrtEvent` 上报的 `avg_us`。一次调用若下发多个参与计算的 kernel，日志关联全部任务并统计每次调用的 kernel 时长。 |
| 内存 | 任务书 §3.4 未设性能验收阈值。验证 workspace 上界、对齐、地址范围和异步复用，内存报告口径按任务确认。 |

### 4.2 用例与验收材料

在 `test/gemm/cgemm3m/arch22/` 接入 CSV 驱动的 ATK/C++ GTest，使用 `bash build.sh --ops=cgemm3m --soc=ascend910b3` 构建。覆盖 9 种转置组合、大小和长宽矩阵、带 padding 的物理列、纯虚和一般复数、极端 `alpha/beta`、Inf/NaN、零维、非法参数及空 handle。用哨兵检测 padding 写保护，用 `alpha=0` 配合空或 NaN 的 A/B、`beta=0` 配合 NaN 的 C 验证旁路，使用按列可辨识的数据检查 `OP_C` 和 `OP_T`。

随包 CSV 包含 1000 条精度和 200 条性能配置。性能行中的短枚举 `N` 与生成器识别的 `ACLBLAS_OP_N` 不一致，按 §2.4 复核发现 60 条性能用例的 `lda` 或 `ldb` 低于下界。原始 CSV 与基线保留，需由任务方确认合法参数，并对参数有变更的行重采同参 GPU kernel 基线。验收继续使用合法输入和原规定阈值。冻结输入、golden、用例清单及运行环境应随最终自验结果绑定。

### 4.3 兼容性与改动范围

沿用已有 `aclblasCgemm3m` 声明、类型和句柄接口。实现位于 `blas/gemm/arch22/`，按仓内规范注册 kernel 源，扩展 `test/gemm/cgemm3m/CMakeLists.txt` 的 arch22 目标并补充 `blas/gemm/README.md` 的产品支持表。编译选择对应架构的导出定义，保持 arch35 的行为并回归已有测试。
