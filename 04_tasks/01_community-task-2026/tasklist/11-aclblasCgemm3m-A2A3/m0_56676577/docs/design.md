# aclblasCgemm3m（Atlas A2 / A3）算子设计文档

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-25 | m0_56676577 | Atlas A2/A3（`arch22` / `dav-2201`）四级自适应 Gauss 3M 复数矩阵乘设计方案：退化标量独立 AIV 缩放、极小尺寸零 Workspace 单核/多核 AIV 直算、中小及瘦高/扁宽尺寸单次下发 Mix-Core（AIC+AIV 1:2）融合 3M、大尺寸单趟融合拆包加和 + 全 IEEE-754 FP32 Cube 3M 乘加 + 融合重组 Epilogue |

> **实现与测试位置**：个人 `ops-blas` 仓 `blas/gemm/arch22/`（独立新增 `cgemm3m_*` 源文件，不放入实数 3M 目录 `blas/gemm3m/`）、`blas/CMakeLists.txt`、`blas/gemm/README.md` 及测试目录 `test/gemm/cgemm3m/arch22/` 与 `test/gemm/cgemm3m/CMakeLists.txt`。公共声明已存在于 `include/cann_ops_blas.h`（L584），本任务不新增重复声明。本 PR 仅提交设计文档，不包含算子源代码、测试代码或实测采样日志。

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026——「11月社区任务-aclblasCgemm3m算子开发（A2/A3）」。任务要求基于昇腾算子开源仓 [ops-blas](https://gitcode.com/cann/ops-blas) 工程框架，使用 Ascend C 算子直调模式开发单精度复数（`COMPLEX64`）三乘法通用矩阵乘算子 `aclblasCgemm3m`，适配 Atlas A2 系列（性能标定设备 Atlas 800T A2 / Ascend 910B3）与 Atlas A3 系列产品（`arch22` / `dav-2201`），接口签名与语义对齐 cuBLAS `cublasCgemm3m` 及 Netlib BLAS `cgemm` 参考实现，验收通过后合入 `ops-blas` 开源仓。

## 背景介绍

### aclblasCgemm3m算子实现优化

`cgemm3m`（Complex single-precision General Matrix-Matrix Multiply via Gauss 3M Algorithm）是 BLAS Level 3 核心高性能复数矩阵乘接口，广泛应用于雷达/声呐阵列信号处理、无线通信大规模 MIMO 预编码、电磁仿真、量子力学计算及复数神经网络等计算密集型场景。

标准复数矩阵乘（4M 算法）将 $(A_r + iA_i)(B_r + iB_i)$ 展开为 4 次实数矩阵乘法，浮点计算量为 $8MNK$。Gauss 3M 算法利用实虚部线性组合将其等价转化为 **3 次实数矩阵乘法**，理论 Cube 乘加计算量下降 **25%**（降至 $6MNK$），在计算受限的中大矩阵上具备显著的吞吐优势。但在 Atlas A2/A3（`arch22` / `dav-2201`，910B3 含 20 个 AIC Cube 核与 40 个 AIV Vector 核）上落地高性能、高精度的 `aclblasCgemm3m` 面临四类关键挑战：

1. **全尺度性能包络跨度极大（$9.86\,\mu\text{s} \sim 27.55\,\text{ms}$）**：
   分析任务配套的 `gpu_baseline.csv`（200 条 A100 kernel-only 性能基线，覆盖 Small $\le 64\,\text{KiB}$、Medium $\le 2\,\text{MiB}$、Large $> 2\,\text{MiB}$ 三档及 25 条非默认 $\alpha/\beta$ 用例）：
   - 小尺寸（$1^3 \sim 64^3$）及瘦高/扁宽矩阵（如 $4095\times 63\times 63$、$63\times 4095\times 63$、$63\times 63\times 4095$）的达标上限仅 **$9.86 \sim 25.47\,\mu\text{s}$**。若拆分为多次独立的 Host 侧 AIV/AIC Kernel 下发（拆包 A、拆包 B、3 次实数 GEMM、重组写回），仅多次 Kernel 发射与流间同步开销就会超限，必须为小/中尺寸设计**单次下发（Single-Launch）**直算或 Mix-Core（AIC+AIV 1:2）片上融合路径；
   - 大尺寸立方矩阵（如 $4096^3$ `CN`/`NN`/`NT` 与 $4095^3$ `NN`）的达标上限为 **$27.34 \sim 27.55\,\text{ms}$**，对应 3M 有效 FP32 算力需求 $\ge 15.0\,\text{TFLOPS}$。必须在保证全 IEEE-754 FP32 精度的前提下充分榨取 20 个 AIC Cube 核的 L1/L0/L0C 双缓冲流水与 L2 缓存局部性，同时将 AIV 预处理/重组带宽开销压缩到最低。
2. **Cube 精度与长 $K$ 归约舍入控制**：
   `arch22` 上 Cube 单元的 FP32 矩阵乘在部分 CANN/SOC 组合下默认开启 HF32（10-bit 尾数截断），在 $K = 1024 \sim 4096$ 归约时会引发显著精度漂移，违反生态精度标准 `rtol = atol = 2^-13`、`matched_ratio >= 0.99` 与 `max_abs_error <= max(1e-2, 32 * ULP)`。此外，3M 算法中间项（如 $A_r + A_i$、$B_r + B_i$）与直接 4M 复数乘法的 FP32 舍入路径不同，需要显式关闭 HF32 并结合合理 K 分块归约控制长 $K$ 误差与大数抵消误差。
3. **带宽与 Workspace 内存预算平衡**：
   $4096^3$ 复数矩阵单平面（`float32`）占 $64\,\text{MiB}$。若朴素实现「拆包写 GM $\to$ 再次读 GM 做加和 $\to$ 3 次 GEMM 各自写完整中间平面 $\to$ 再读 3 个平面做重组」，额外 HBM 流量高达十余倍矩阵体积。通过 **AIV 单趟融合拆包加和**（读一次复数输入同时输出 $A_r, A_i, A_r + s_a A_i$）以及**两缓冲原地轮转/片上融合重组**，可大幅削减 HBM 访问量并将设备侧 workspace 控制在合理范围内。
4. **严格的 BLAS 语义与异常边界守卫**：
   需严格满足列主序 9 种 `(transa, transb)` 组合（`OP_C` 共轭与 `OP_T` 转置语义区分）、前导维 padding 写保护、条件空指针校验、`m=0`/`n=0` quick return、`k=0`/`alpha=0` 不访问 `A/B`、`beta=0` 不读取旧 `C`（含 `C=NaN` 场景）以及标量零分量免污染（避免 `0 * Inf = NaN`）。

### aclblasCgemm3m算子实现路径和相关API路径

| 内容 | 仓库路径 | 说明 |
| --- | --- | --- |
| 公共接口声明 | `include/cann_ops_blas.h`（L584） | 已存在，全产品线共用，本任务不新增重复声明 |
| 3M Host 入口与 Tiling | `blas/gemm/arch22/cgemm3m_host.cpp`、`blas/gemm/arch22/cgemm3m_tiling_data.h` | 校验、路径分派、Workspace 规划与 Kernel 启动（禁止放入 `blas/gemm3m/`） |
| 3M Kernel 实现 | `blas/gemm/arch22/cgemm3m_kernel.cpp`、`blas/gemm/arch22/cgemm3m_kernel.h` | AIV 退化缩放、小尺寸 AIV 直算、Mix-Core 融合 3M、大尺寸融合拆包/3M Cube/重组 Epilogue |
| 构建注册 | `blas/CMakeLists.txt` | 补齐 `--ops=cgemm3m` 单算子构建时的依赖文件收集规则 |
| 算子文档 | `blas/gemm/README.md` | 更新 `aclblasCgemm3m` 章节，补充 Atlas A2/A3 产品支持与实现说明 |
| 测试工程 | `test/gemm/cgemm3m/arch22/`、`test/gemm/cgemm3m/CMakeLists.txt` | 复用公共 `cgemm3m_{param,golden,verify,profile}.h`，注册 `arch22` GTest 目标并接入 1200 条用例与 `[PERF] avg_us` 输出 |

### aclblasCgemm3m算子现状分析

对上游 `ops-blas` 仓现状分析如下：
- `include/cann_ops_blas.h`（L584）已提供 `aclblasCgemm3m` 公开声明，签名与 `aclblasCgemm` 及 cuBLAS `cublasCgemm3m` 逐参数一致；
- `blas/gemm/arch35/cgemm3m_host.cpp` 仅面向 Ascend 950PR（`arch35`）实现了 3M 路径（依赖 `arch35` 专用 `cgemm_host.cpp` 与片上指令），`blas/gemm/arch22/` 当前仅有 `aclblasSgemm`（`gemm_host.cpp`/`gemm_kernel.cpp`）与 4M 版本 `aclblasCgemm`（`cgemm_host.cpp`/`cgemm_kernel.cpp`），尚无 `aclblasCgemm3m` 的 `arch22` 实现；
- 既有 `blas/gemm/arch22/cgemm_host.cpp` 采用 4 次调用 `aclblasSgemmEx` + 独立 `cgemm_accum_kernel` + 独立 `cgemm_split_kernel`/`cgemm_combine_kernel` 的 4M 管线，存在四次实数 GEMM 计算量、小尺寸多次 Kernel 启动延迟高及 `CGEMM_K_CHUNK = 1024` 额外累加开销等问题，不满足 3M 语义与本任务性能要求；
- `test/gemm/cgemm3m/CMakeLists.txt` 当前仅在 `"arch35" IN_LIST SOC_ARCH_DIRS` 时注册 `cgemm3m_test`，`arch22` 分支会直接跳过，且 `test/gemm/cgemm3m/arch22/` 目录尚未建立。

对标接口参数能力如下表所示：

| 参数 | 输入/输出/属性 | 参数含义 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | ops-blas 库上下文句柄（绑定 `aclrtStream`） | `aclblasHandle_t` | Host 标量 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transa` | 属性 | `op(A)` 运算类型：`N` 不转置、`T` 转置、`C` 共轭转置 | `aclblasOperation_t` | Host 枚举 | 仅 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`，否则返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `transb` | 属性 | `op(B)` 运算类型（同 `transa`） | `aclblasOperation_t` | Host 枚举 | 仅 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`，否则返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m` | 属性 | `op(A)` 与 `C` 的行数 | `int` | Host 标量 | `m >= 0`；`m < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`m == 0` 合法 quick return |
| `n` | 属性 | `op(B)` 与 `C` 的列数 | `int` | Host 标量 | `n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 合法 quick return |
| `k` | 属性 | `op(A)` 的列数与 `op(B)` 的行数 | `int` | Host 标量 | `k >= 0`；`k < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`k == 0` 仅做 `C = beta * C` |
| `alpha` | 输入 | 复数标量乘数指针 | `const aclblasComplex*` | Host 内存，`COMPLEX64` | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`alpha == (0,0)` 时不引用 `A/B` |
| `A` | 输入 | 输入复数矩阵 `A`，列主序 | `const aclblasComplex*` | Device ND，`lda×k`(N) 或 `lda×m`(T/C) | `m>0 && n>0 && k>0 && alpha!=0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 属性 | 矩阵 `A` 的前导维度 | `int` | Host 标量 | `transa==N` 时 `lda >= max(1, m)`，否则 `lda >= max(1, k)`；否则 `INVALID_VALUE` |
| `B` | 输入 | 输入复数矩阵 `B`，列主序 | `const aclblasComplex*` | Device ND，`ldb×n`(N) 或 `ldb×k`(T/C) | `m>0 && n>0 && k>0 && alpha!=0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` | 属性 | 矩阵 `B` 的前导维度 | `int` | Host 标量 | `transb==N` 时 `ldb >= max(1, k)`，否则 `ldb >= max(1, n)`；否则 `INVALID_VALUE` |
| `beta` | 输入 | 复数标量乘数指针 | `const aclblasComplex*` | Host 内存，`COMPLEX64` | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`beta == (0,0)` 时不读 `C` 旧值 |
| `C` | 输出（原地） | 输出复数矩阵 `C`，列主序，原地覆写 | `aclblasComplex*` | Device ND，`ldc×n` | `m>0 && n>0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；不改写 `row >= m` padding |
| `ldc` | 属性 | 矩阵 `C` 的前导维度 | `int` | Host 标量 | `ldc >= max(1, m)`；否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |

### aclblasCgemm3m算子功能分析

- 计算公式：$C = \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(B) + \beta \cdot C$，其中 $\operatorname{op} \in \{N, T, C\}$；
- 输入：`handle`、`transa`、`transb`、`m`、`n`、`k`、`alpha`（Host）、`A`（Device）、`lda`、`B`（Device）、`ldb`、`beta`（Host）、`ldc`；
- 输出：`C`（Device 原地更新）；
- 支持数据类型：`COMPLEX64`（`aclblasComplex`，实部/虚部各为 IEEE-754 `float32`）；
- 核心算法：Gauss 3M 分解（3 次实数矩阵乘法替代 4 次）；
- 支持广播：不支持（独立矩阵操作数，仅按列主序前导维布局访问）。

# 需求分析（required）

## 需求描述

在 `ops-blas` 仓 `blas/gemm/arch22/` 路径下使用 Ascend C kernel 直调模式实现 `aclblasCgemm3m`，在 `test/gemm/cgemm3m/arch22/` 下补齐 `arch22` GTest 测试工程与构建注册，并同步更新 `blas/CMakeLists.txt` 及 `blas/gemm/README.md`。实现需严格对齐任务书参数校验顺序、Gauss 3M 数值语义、边界/特殊值处理与列主序 padding 保护；精度满足生态算子开源精度标准（实部/虚部独立按 FLOAT32 判定），正式性能（`msprof op` kernel Task Duration）在 Atlas 800T A2（910B3）上达到全部 200 条 A100 `gpu_baseline.csv` 基线的 $0.8\times$ 以上（单条与平均倍率均 $\ge 0.8$）。

## 需求拆解

1. **严格参数校验与退化语义**：按「`handle` 判空 $\to$ `transa/transb` 枚举校验（`INVALID_ENUM`）$\to$ `m/n/k < 0` $\to$ `lda/ldb/ldc` 下界校验 $\to$ `alpha/beta` 判空 $\to$ `m==0 || n==0` quick return（`SUCCESS`）$\to$ `C` 判空及 `k>0 && alpha!=0` 条件下的 `A/B` 判空」顺序执行校验；`k==0 || alpha==0` 时绝不访问 `A/B`；`beta==0` 时绝不读取 `C` 旧值；严格保护 `ldc` padding 区域不被修改。
2. **真 3M 复数乘法与全 IEEE-754 FP32 精度保障**：所有非退化路径均采用 3 次实数矩阵乘（非 4M），将 `OP_C` 共轭符号在拆包阶段折叠进虚部平面以统一 9 种转置组合；所有 AIC Cube 路径显式调用 `SetHF32Mode(HF32Mode::DISABLE)` 关闭 HF32 截断，结合 K 分块归约确保长 $K$（至 4096）与大数抵消场景下满足 `rtol = atol = 2^-13`、`matched_ratio >= 0.99` 与 `max_abs_error <= max(1e-2, 32 * ULP)`。
3. **全尺度四级自适应高性能架构**：
   - 退化标量路径零 workspace 单 Kernel 处理；
   - 极小矩阵（$1^3 \sim 32^3$）采用零 workspace 单次下发 AIV 直算，确保核耗时压入 $10\,\mu\text{s}$ 级；
   - 中小与瘦高/扁宽矩阵（$64^3 \sim 256^3$ 及极端长宽比）采用单次下发 Mix-Core（AIC+AIV 1:2）片上/共享 scratch 融合 3M，消除多 Kernel 发射墙；
   - 大矩阵（最高 $4096^3$）采用「单趟融合拆包加和 + 3 次高吞吐 FP32 Cube GEMM（含 L2 Tile Swizzle 与双缓冲）+ 融合重组 Epilogue」，实现 $\ge 15.0\,\text{TFLOPS}$ 有效 3M 算力并控制 workspace 开销。
4. **双口径测试与自验闭环**：在 `test/gemm/cgemm3m/arch22/` 下集成 Netlib BLAS 3.10.0 `cblas_cgemm` 标杆，支持全量 1,000 条精度用例与白盒边界单测（空 handle、padding 哨兵、零积空指针、大数抵消等），同时支持 `msprof op` 正式核耗时采集与 GTest `[PERF] avg_us` 交叉验证（含 25 条非默认 `alpha/beta` 性能用例）。

# 详细设计（required）

## 算子分析

### 数学公式

设列主序复数矩阵 $A = A_r + i A_i$、$B = B_r + i B_i$、$C = C_r + i C_i$，复数标量 $\alpha = \alpha_r + i \alpha_i$、$\beta = \beta_r + i \beta_i$。定义共轭符号：

$$
s_a = \begin{cases} -1, & \text{transa} = \text{ACLBLAS\_OP\_C} \\ +1, & \text{transa} \in \{\text{ACLBLAS\_OP\_N}, \text{ACLBLAS\_OP\_T}\} \end{cases}, \qquad
s_b = \begin{cases} -1, & \text{transb} = \text{ACLBLAS\_OP\_C} \\ +1, & \text{transb} \in \{\text{ACLBLAS\_OP\_N}, \text{ACLBLAS\_OP\_T}\} \end{cases}
$$

令符号折叠后的实虚平面为 $\tilde{A}_r = A_r$、$\tilde{A}_i = s_a A_i$、$\tilde{B}_r = B_r$、$\tilde{B}_i = s_b B_i$，并令无共轭转置算子 $\overline{\operatorname{op}}(X) = X$（若 `trans == N`）或 $X^T$（若 `trans == T/C`）。由此，全部 9 种 `(transa, transb)` 组合均统一为无共轭的实矩阵乘积：

$$
\operatorname{op}(A)\operatorname{op}(B) = \bigl(\overline{\operatorname{op}}(\tilde{A}_r) + i\,\overline{\operatorname{op}}(\tilde{A}_i)\bigr)\bigl(\overline{\operatorname{op}}(\tilde{B}_r) + i\,\overline{\operatorname{op}}(\tilde{B}_i)\bigr)
$$

算子按形状与片上容量在两种代数等价的 Gauss 3M 分解中选择执行路径：

1. **融合路径（2 项重组 3M，用于中小/片上融合与两缓冲轮转路径）**：
   $$
   \begin{aligned}
   P_0 &= \overline{\operatorname{op}}(\tilde{A}_r + \tilde{A}_i)\cdot \overline{\operatorname{op}}(\tilde{B}_r) \\
   P_1 &= \overline{\operatorname{op}}(\tilde{A}_r)\cdot \overline{\operatorname{op}}(\tilde{B}_i - \tilde{B}_r) \\
   P_2 &= \overline{\operatorname{op}}(\tilde{A}_i)\cdot \overline{\operatorname{op}}(\tilde{B}_r + \tilde{B}_i) \\
   R &= P_0 - P_2, \qquad I = P_0 + P_1
   \end{aligned}
   $$
   **优势**：每个输出分量仅需 2 项加减；在 Mix-Core 融合或两缓冲路径中，$P_0$ 计算完成后可仅用 1 份临时缓冲轮转 $P_1$ 与 $P_2$，相比保存 3 份完整乘积平面减少 $33\%$ 中间结果写回流量与空间占用。

2. **非融合路径（对称 3M，用于大尺寸独立 3×Cube GEMM 与 Split-K 路径）**：
   $$
   \begin{aligned}
   P_0 &= \overline{\operatorname{op}}(\tilde{A}_r)\cdot \overline{\operatorname{op}}(\tilde{B}_r) \\
   P_1 &= \overline{\operatorname{op}}(\tilde{A}_i)\cdot \overline{\operatorname{op}}(\tilde{B}_i) \\
   P_2 &= \overline{\operatorname{op}}(\tilde{A}_r + \tilde{A}_i)\cdot \overline{\operatorname{op}}(\tilde{B}_r + \tilde{B}_i) \\
   R &= P_0 - P_1, \qquad I = P_2 - P_0 - P_1
   \end{aligned}
   $$
   **优势**：前两项 $P_0, P_1$ 的操作数均直接为原始实虚平面（仅 $P_2$ 引入和矩阵），便于在 AIV 单趟预处理中对称生成 $(A_r, A_i, A_s)$ 与 $(B_r, B_i, B_s)$，并与大尺寸 2D 分块 Cube GEMM 复用同一套实数矩阵乘微核；当 `A` 或 `B` 为纯实/纯虚矩阵时，误差结构对称清晰。

3. **复数标量 Epilogue（零分量免污染）**：
   $$
   C_r \leftarrow (\alpha_r R - \alpha_i I) + (\beta_r C_{r,\text{old}} - \beta_i C_{i,\text{old}}), \qquad
   C_i \leftarrow (\alpha_r I + \alpha_i R) + (\beta_r C_{i,\text{old}} + \beta_i C_{r,\text{old}})
   $$
   对 $\alpha_r, \alpha_i, \beta_r, \beta_i$ 中为 `0.0f` 的分量跳过乘法项（不计算 `0.0f * x`），避免另一分量出现 `Inf` 时产生 `0 * Inf = NaN` 跨分量污染；当 $\beta = (0, 0)$ 时完全不读取 $C_{\text{old}}$。

### 支持数据类型

- 矩阵 `A`、`B`、`C` 及标量 `alpha`、`beta`：`COMPLEX64`（`aclblasComplex`，含连续两个 IEEE-754 `float32` 实部与虚部）；
- 内部计算与累加精度：全路径严格采用 IEEE-754 `float32`（所有 AIC Cube 核显式关闭 `HF32` 模式）。

### 支持形状

- 矩阵维度 `m`、`n`、`k`：运行时 `int` 入参，支持 $m, n, k \ge 0$ 全泛化规模（覆盖 0、1、小质数、2 的幂及 $\pm 1$、非对齐尾块、极端瘦高/扁宽矩形及 $4095^3 / 4096^3$ 以上大尺寸）；
- 前导维 `lda`、`ldb`、`ldc`：支持紧凑布局及任意奇偶 padding（满足 BLAS 前导维下界约束），写出严格限定于逻辑区域 $0 \le r < m, 0 \le c < n$。

## 算子实现

### 实现方案

#### 3.2.0 整体执行流程图

```text
aclblasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  │
  ├─► 1. 分级参数校验（Host）
  │     handle==nullptr -> HANDLE_IS_NULLPTR
  │     transa/transb ∉ {N,T,C} -> INVALID_ENUM
  │     m<0 || n<0 || k<0 -> INVALID_VALUE
  │     lda/ldb/ldc < minLd -> INVALID_VALUE
  │     alpha==nullptr || beta==nullptr -> INVALID_VALUE
  │     m==0 || n==0 -> Quick Return SUCCESS (不访问 A/B/C，不启动 Kernel)
  │     C==nullptr -> INVALID_VALUE
  │     k>0 && (alpha!=(0,0)) && (A==nullptr || B==nullptr) -> INVALID_VALUE
  │
  └─► 2. 运行时四级自适应路径分派（绑定 handle->stream）
        ├── Path 0 [DEGENERATE_SCALE] : k == 0 || alpha == (0,0)
        │     ├─► beta == (1,0) : Quick Return SUCCESS (零 Kernel)
        │     └─► 否则 : cgemm3m_scale_kernel (AIV-only，零 Workspace，不读 A/B，beta==0 不读旧 C)
        │
        ├── Path 1 [TINY_AIV_DIRECT]  : M*N*K ≤ 32768 (覆盖 1³~32³ 等极小矩阵)
        │     └─► cgemm3m_tiny_direct_kernel (单次下发 AIV-only，零 Workspace，UB 内直接完成 3M + Epilogue)
        │
        ├── Path 2 [SMALL_MED_MIX_3M] : 中小及瘦高/扁宽尺寸 (如 64³~256³、4095×63×63、63×4095×63、63×63×4095)
        │     └─► cgemm3m_fused_mix_kernel (单次下发 KERNEL_TYPE_MIX_AIC_1_2，消除多 Kernel 启动墙)
        │           ├─► Stage A (AIV) : 并行拆包 A->(Ar,Ai,Asum), B->(Br,Bi,Bdiff/Bsum) 并折叠 OP_C 符号
        │           ├─► SyncAll<false>()
        │           ├─► Stage B (AIC) : 20 核按 (product, mTile, nTile, kSplit) 联合网格计算 P0/P1/P2 (关闭 HF32)
        │           ├─► SyncAll<false>()
        │           └─► Stage C (AIV) : 按 2 项/3 项 3M 规则重组 (Re,Im) + alpha/beta Epilogue 原地写回 C
        │
        └── Path 3 [LARGE_HIGH_PERF]  : 大尺寸计算密集型矩阵 (最高 4096³+)
              ├─► Stage 1 (AIV) : cgemm3m_prep_kernel — 单趟读 A/B 融合输出对齐平面 (Ar,Ai,As) 与 (Br,Bi,Bs)
              ├─► Stage 2 (AIC) : 3× 高吞吐 FP32 Cube GEMM (SetHF32Mode(DISABLE)，2D L2 Swizzle + L1/L0/L0C 双缓冲)
              │     ├─► 常规高并行度大矩阵 : 对称 3M / 两缓冲轮转 3M 依次或联合调度 P0, P1, P2
              │     └─► 深 K 窄输出形状   : Split-K 分区归约，保持 20 AIC 满载并缩短单条 FP32 累加链
              └─► Stage 3 (AIV) : cgemm3m_epilogue_kernel — 向量化 3M 线性重组 + alpha/beta 缩放 + 交错写回 C
```

#### 3.2.1 host侧设计：

**tiling策略**：
`cgemm3m_host.cpp` 负责完成严格的参数校验、路径决策、Workspace 规划与核函数 Tiling 参数构造：

##### 1. 校验与标量读取策略
- 严格遵循任务书 §2.4/§2.5 校验顺序：先检查结构参数（`handle` $\to$ `transa/transb` $\to$ `m/n/k` $\to$ `lda/ldb/ldc` $\to$ `alpha/beta` 指针），合法后再执行 `m==0 || n==0` quick return；随后再对 `C` 判空，并仅在 `k > 0 && !(alpha.real == 0.0f && alpha.imag == 0.0f)` 时对 `A/B` 判空；
- `alpha/beta` 按 ops-blas 约定直接作为 Host 指针解引用读取；同时支持与 `handle` 的异步 stream 绑定，Host 侧不插入冗余 `aclrtSynchronizeStream`。

##### 2. 分核与四级路径分派策略
在运行时查询硬件核心数 `aicCoreNum = GetAicCoreCount()`（910B3 为 20）与 `aivCoreNum = GetAivCoreCount()`（910B3 为 40），依据问题体积、输出 Tile 并行度与 K 深度选择最优路径：
- **Path 0（退化缩放）**：`k == 0 || alpha == (0,0)`；若 `beta == (1,0)` 直接返回 `ACLBLAS_STATUS_SUCCESS`，否则以 `min(aivCoreNum, totalChunks)` 启动 `cgemm3m_scale_kernel`。
- **Path 1（极小尺寸直算）**：当 `static_cast<uint64_t>(m) * n * k <= 32768` 且 `max(m, n, k) <= 64`（如 $1^3 \sim 32^3$ 及微型矩形）时，Cube 搬运与核间同步成本高于计算本身，直接启动单次 AIV Kernel，按 `(mTile, nTile)` 或总输出元素划分至 `min(aivCoreNum, tasks)` 个核，零 workspace 完成计算。
- **Path 2（中小及瘦高/扁宽单次 Mix-Core 融合）**：当 `Path 1` 不适用，且问题处于中小规模或小发射预算区间（例如 $64^3 \sim 512^3$、以及单维 $\le 128$ 的瘦高/扁宽/长 K 矩阵如 $4095 \times 63 \times 63$、$63 \times 63 \times 4095$），且所需临时平面可装入 handle 默认 workspace 时，采用**单次下发 Mix-Core（`KERNEL_TYPE_MIX_AIC_1_2`，`blockDim = aicCoreNum`）**：
  - 将 3 个实数乘积 `(p = 0, 1, 2)` 与 `(mTiles, nTiles, kSplits)` 展平为统一任务网格 `totalTasks = 3 * mTiles * nTiles * kSplits`，由 20 个 AIC 核在一次 Kernel 启动内协同吞掉全部 3 次乘积，避免单次乘积 tile 过少导致的 AIC 核空转（例如 $256^3$ 用 $128\times 128$ tile 单独做 1 次 GEMM 仅 4 个任务、20 核利用率 20%，而将 3 个乘积联合调度并适度切分 $64\times 64$ 或 Split-K 后可生成 48 个独立任务，满载全部 20 个 AIC 核！）；
  - 前后 AIV 拆包与重组在同一 Kernel 中通过 `SyncAll<false>()` 衔接，彻底消除多次 Host 发射延迟。
- **Path 3（大尺寸高吞吐 3M 管线）**：对于计算密集型大矩阵（如 $512^3 \sim 4096^3+$，以及超出 Mix-Core 区间的大矩形），启动「AIV 融合预处理 $\to$ 3× 高吞吐 FP32 Cube GEMM $\to$ AIV 融合 Epilogue」管线：
  - Cube Tile 在候选集合 `(BM, BN, BK) ∈ {(128, 128, 64), (128, 256, 32), (256, 128, 32), (64, 128, 64)}` 中按负载均衡代价模型（最小化最大核波次 `ceil(mTiles * nTiles / aicCoreNum)` 与 L1/GM 带宽比）动态选取；
  - 引入 **2D Swizzle 块调度**（按 $S \times S$ 超级块遍历 `(mTile, nTile)`，典型 swizzle 宽度 $S = 4 \sim 8$），最大化 L2 Cache 中 `A`/`B` 面板的跨核复用率，降低 $2048^3 \sim 4096^3$ 场景下的 HBM 带宽压力；
  - 当单乘积输出 Tile 总数不足 `aicCoreNum` 且 `k` 较大（如 $64 \times 64 \times 4096$）时启用 **Split-K**，并在 Epilogue 阶段由 AIV 一次性完成 Split-K 规约与 3M 线性重组，避免额外规约 Kernel。

##### 3. Workspace 内存规划与安全溢出检查
所有中间平面均复用 `aclblasHandle_t` 内部默认 workspace（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`），平面前导维按 16 个 `float`（64 字节）对齐以保证 32B/64B DMA 整突发访问。记对齐后单平面元素数为 $S_A = \text{Align16}(\text{rows}_A)\cdot\text{cols}_A$、$S_B = \text{Align16}(\text{rows}_B)\cdot\text{cols}_B$、$S_C = \text{Align16}(m)\cdot n$：
- **对称 3M 路径**：存放 `Ar, Ai, As`（$3 S_A$）、`Br, Bi, Bs`（$3 S_B$）与 `P0, P1, P2`（$3 S_C$，或通过 2-term 融合路径压缩为 2 份输出缓冲 `buf0, buf1`，即 $2 S_C$）；以 $4096^3$ 为例，2 缓冲方案设备侧 workspace 仅需 $(3\times 64 + 3\times 64 + 2\times 64) = 512\,\text{MiB}$（3 缓冲方案 $576\,\text{MiB}$），远低于 910B3 的 $64\,\text{GiB}$ HBM 及 handle 默认上限；
- **64 位安全乘法检查**：所有字节数累加均通过带溢出检测的 64 位算术辅助函数计算，极端维数溢出时安全返回 `ACLBLAS_STATUS_ALLOC_FAILED`；
- **大矩阵降级保底**：若用户通过 `aclblasSetWorkspace` 绑定了较小的外部 workspace 且不足以容纳完整 3 平面，则在合法范围内优先切换至 2 缓冲方案或逐块分片计算，仍不足时按规范返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

##### 4. TilingKey / 模板特化规划
- 将转置组合 `(taNonN, tbNonN)`（共 4 种布局，共轭符号 $s_a, s_b$ 已在 AIV 拆包阶段折叠为 $\pm 1$，不膨胀 Cube 模板数）与路径模式作为编译期模板参数/轻量分发键；
- `cgemm3m_tiling_data.h` 定义紧凑、按 8 字节对齐的 POD Tiling 结构体（`Cgemm3mPrepTiling`、`Cgemm3mCubeTiling`、`Cgemm3mEpilogueTiling`、`Cgemm3mMixTiling`），通过值传参直接下发至设备端，避免额外 Tiling H2D 拷贝开销。

#### 3.2.2 kernel侧设计：

所有 Kernel 严格遵守 `arch22` 硬件内存层级与同步规范：显式管理 `MTE2`/`MTE1`/`M`/`FIX`/`V`/`MTE3` 硬件事件，确保无跨管道写后读/读后写冒险；所有 Cube 核入口第一条指令显式执行 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)`。

##### 1. 退化标量缩放核（`cgemm3m_scale_kernel`，Path 0）
- **职责**：处理 `k == 0 || alpha == (0,0)` 且 `beta != (1,0)` 场景，绝不引用 `A/B` 指针；
- **实现**：AIV-only，按列循环 + 行分块（`CHUNK = 1024` 复数元素，UB 占用 $\sim 48\,\text{KiB}$，开启 `MTE2/V/MTE3` 双缓冲）：
  - 若 `beta == (0,0)`：跳过 `DataCopyPad` 读旧 `C`，直接 `Duplicate(0.0f)` 后以 `DataCopyPad` 精确写回 `len * 8` 字节（保护 `C=NaN` 且不越界触碰 `ldc` padding）；
  - 若 `beta != (0,0)`：读入交错复数 `C_old` $\to$ `GatherMask` 拆分为 `Cr, Ci` $\to$ 按零分量守卫计算 `Cr_new = br*Cr - bi*Ci`、`Ci_new = br*Ci + bi*Cr` $\to$ `Gather` 交错回写 `C`。

##### 2. 极小尺寸 AIV 直算核（`cgemm3m_tiny_direct_kernel`，Path 1）
- **职责**：覆盖 $M\cdot N\cdot K \le 32768$（含 $1^3 \sim 32^3$），消除 Cube 搬运、Workspace 分配与核间同步延迟；
- **实现**：单次启动 AIV-only Kernel，将逻辑矩阵 `A` 与 `B` 直接通过 `DataCopyPad` 载入 UB 并拆分实虚部（同时应用共轭符号 $s_a, s_b$），在 UB 内直接以 Gauss 3M 公式构造 $A_r + \tilde{A}_i$、$B_r + \tilde{B}_i$，向量化累加 $P_0, P_1, P_2$，合并 $R, I$ 并叠加 $\alpha / \beta$ 后精确写回 `C` 的逻辑 $m \times n$ 区域。单次核耗时预期控制在 $5 \sim 12\,\mu\text{s}$ 区间，满足 $1^3$（上限 $9.86\,\mu\text{s}$）至 $32^3$（上限 $17.02\,\mu\text{s}$）全部小尺寸性能达标线。

##### 3. 单趟融合拆包与和/差矩阵生成核（`cgemm3m_prep_kernel` 及 Mix-Core Stage A）
- **职责**：将传统「先拆包写 GM、再读 GM 做加减」的 2 趟 AIV 合并为 **1 趟读取、3 路输出**，节省 50% 输入带宽；
- **实现**：
  - 按列分块分配至 40 个 AIV 核，开启双缓冲（单缓冲块长 `CHUNK = 1024` 复数，UB 总占用 $< 112\,\text{KiB} < 192\,\text{KiB}$）；
  - `CopyIn`：`DataCopyPad` 搬入列主序交错复数块（长度 `len * 8` 字节）；
  - `Compute`：`GatherMask`（pattern 1/2）一次抽取 `Re` 与 `Im`；若 `op == OP_C` 对 `Im` 执行 `Muls(-1.0f)` 折叠共轭符号；随后直接在 UB 内计算：
    - 对称 3M 模式：输出 `Ar`、`Ai`、`Asum = Ar + Ai`；对 `B` 输出 `Br`、`Bi`、`Bsum = Br + Bi`；
    - 2 项融合 3M 模式：对 `A` 输出 `Ar`、`Ai`、`Asum = Ar + Ai`；对 `B` 输出 `Br`、`Bdiff = Bi - Br`、`Bsum = Br + Bi`；
  - `CopyOut`：对于行尾未对齐 16-float（64B）的部分在 UB 末尾补零至 `packedLd`，再以连续整突发 `DataCopy` 写入三个对齐实数平面（保证后续 Cube `Nd2Nz` 加载不越界、无脏尾污染）。

##### 4. 高吞吐全 IEEE-754 FP32 Cube 3M 核心（`cgemm3m_cube_kernel` 及 Mix-Core Stage B）
- **硬件映射（列主序零转置同构）**：
  对于列主序矩阵乘 $P = \overline{\operatorname{op}}(X)\cdot \overline{\operatorname{op}}(Y)$（形状 $M \times N$），其在内存中的字节排列与行主序的 $P^T = \overline{\operatorname{op}}(Y)^T \cdot \overline{\operatorname{op}}(X)^T$（形状 $N \times M$）完全等价。由此：
  - `transB == N` 时，物理存储的 `B`（列主序 $K \times N$）直接为行主序 $[N][K]$ 左操作数，无需转置；
  - `transA == N` 时，物理存储的 `A`（列主序 $M \times K$）直接为行主序 $[K][M]$ 右操作数，无需转置；
  - `transA == T/C` 或 `transB == T/C` 时，分别对应内部右/左操作数转置标志，由 `Nd2Nz` / `LoadData` 在 L1 $\to$ L0A/L0B 硬件路径中零额外搬运完成转置。
- **片上流水与内存预算（`dav-2201` 硬件容量约束）**：
  - 典型主 Tile `(BM, BN, BK) = (128, 128, 64)`（或依据长宽比选择 `(64, 128, 64)` / `(128, 64, 64)`）：
    - **L1（512 KiB）**：`A1`（$128 \times 64 \times 4\text{B} = 32\,\text{KiB}$）与 `B1`（$64 \times 128 \times 4\text{B} = 32\,\text{KiB}$）均开启 2 级乒乓（总计 $128\,\text{KiB}$），或预取更大 K 块（$BK_{\text{L1}} = 128$ 时双缓冲共 $256\,\text{KiB} \le 512\,\text{KiB}$）；
    - **L0A / L0B（各 64 KiB）**：按子块 $BK_{\text{L0}} = 32$ 双缓冲装载（单块 $128 \times 32 \times 4\text{B} = 16\,\text{KiB}$，双缓冲 $32\,\text{KiB} \le 64\,\text{KiB}$）；
    - **L0C（128 KiB）**：单输出累加块 $128 \times 128 \times 4\text{B} = 64\,\text{KiB}$，**开启 2 级双缓冲（$128\,\text{KiB}$）**，使上一 Tile 的 `Fixpipe`（`L0C -> GM`）与下一 Tile 的 `Mmad` 完全流水重叠！
  - **严格关闭 HF32**：核内首行调用 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)`，确保所有 Cube MAC 以完整 IEEE-754 FP32 执行；
  - **单次下发 Mix-Core 模式（Path 2）**：在 `KERNEL_TYPE_MIX_AIC_1_2` 单一核函数中，AIV 完成 Stage A 后执行 `SyncAll<false>()`，AIC 按联合任务网格 `3 * mTiles * nTiles * kSplits` 一次性完成 $P_0, P_1, P_2$ 三个乘积的全部 Tile，再 `SyncAll<false>()` 交由 AIV 执行 Stage C 合并写回。

##### 5. 融合 3M 线性重组与复数 Epilogue 核（`cgemm3m_epilogue_kernel` 及 Mix-Core Stage C）
- **职责**：单趟读取中间乘积平面，完成 3M 线性重组（及可选 Split-K 规约）、复数 $\alpha/\beta$ 缩放、实虚交错并原地写回 `C`；
- **实现**：
  - 40 个 AIV 核按列/分块均分逻辑输出 $m \times n$，块长 `CHUNK = 1024` 复数元素，开启双缓冲；
  - 3M 重组：
    - 2 项融合模式：$R = P_0 - P_2$，$I = P_0 + P_1$；
    - 对称 3M 模式：$R = P_0 - P_1$，$I = P_2 - P_0 - P_1$（先算 $T = P_0 + P_1$ 再 $I = P_2 - T$，节省一次访存与指令）；
    - 若启用 Split-K，则按 `kSplit` 顺序流式累加至局部 `P0, P1, P2` 后再执行上述重组；
  - $\alpha$ 与 $\beta$ 零分量守卫缩放：与 `cgemm_kernel.cpp` 一致，对 $\alpha_r, \alpha_i, \beta_r, \beta_i$ 中非零项才执行 `Muls`/`Add`/`Sub`；$\beta == (0,0)$ 时完全不读取 `C_old`；
  - `CopyOut`：预构造 `Gather` 交错索引表，将 `[Cr | Ci]` 交错为 `[Cr0, Ci0, Cr1, Ci1, ...]`，以 `DataCopyPad` 精确写回 `len * 8` 字节至 `C[col * ldc + row]`，保证 `row >= m` 的 `ldc` padding 区域逐比特不被修改。

##### 6. 关键边界与数值风险防护机制
- **极值与非有限数处理**：
  - `alpha == (0,0)` 或 `k == 0` 走 Path 0，不触碰 `A/B` 中的 `NaN`/`Inf` 或 `nullptr`；
  - `beta == (0,0)` 不读取 `C_old`，确保 `C` 预填 `NaN` 时不会污染输出；
  - 针对 `RANDOM_EXTREME` 等极端量级或潜在和矩阵溢出场景，在 `cgemm3m_prep_kernel` 预留轻量安全检测位或依据 3M 误差特性保持全路径无 HF32 截断，并对照任务书 §3.2 非有限值等效判定规则完成全量验证。
- **跨步与尾块越界防护**：
  - 所有输入/输出 `DataCopyPad` 均按实际有效长度 `len` 精确设定字节数；
  - 对齐平面尾部补零（`m <= row < packedLd`）确保 Cube 分形块计算不会引入未初始化内存中的 `NaN` 反向污染有效输出行。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B3，`arch22` / `dav-2201`） | √ |
| Atlas A3 系列（`arch22`） | √ |

## 算子约束限制

1. 仅支持单精度复数 `COMPLEX64`（`aclblasComplex`）与列主序带前导维（`lda/ldb/ldc`）存储，不支持广播及超出前导维语义的任意非连续视图；
2. 标量 `alpha`、`beta` 指向 Host 内存；矩阵 `A`、`B`、`C` 指向 Device 内存，`C` 为原地更新输出，调用方须保证 `C` 的可写区域与 `A/B` 的可读区域不重叠；
3. 维度满足 `m >= 0`、`n >= 0`、`k >= 0`；前导维满足 `lda >= (transa==N ? max(1,m) : max(1,k))`、`ldb >= (transb==N ? max(1,k) : max(1,n))`、`ldc >= max(1,m)`；`transa/transb` 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，与 cuBLAS 返回 `INVALID_VALUE` 属有意差异）；
4. 算子为异步执行，绑定于 `handle` 关联的 `aclrtStream`，Host 侧读取结果前须同步该 stream；不保证不同形状路径间逐比特确定性，以生态精度标准判定。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 单标杆对比 Netlib BLAS 3.10.0 `cblas_cgemm`（`CblasColMajor`，严禁被 OpenBLAS 劫持符号；测试工程同时支持 float64 累加对照说明）。输出矩阵 `C`（$m \times n$）实部与虚部分别独立按 FLOAT32 混合容差判定：逐元素满足 `\|actual - golden\| <= 2^-13 + 2^-13 * \|golden\|` 的比例 `matched_ratio >= 0.99`，且所有有限值元素满足 `max_abs_error <= max(1e-2, 32 * ULP(golden))`；非有限值按任务书等效规则判定，同时逐字节校验 `ldc` padding 哨兵不被改写。验证覆盖任务书提供全部 1,000 条精度 CSV 用例（含均匀 `RANDOM_NORM_5_5` 496 条、正态 `GAUSS_NORM_0_1` 492 条、极值/Inf/NaN/空指针/非法枚举/负维度/padding 边界及 $4095^3 / 4096^3$ 立方用例）及 GTest 专项白盒测试（空 handle、零积空指针、大数抵消等） | 任务书 §3.2 & [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 在 Atlas 800T A2（Ascend 910B3）上对 `test_cases/gpu_baseline.csv` 全部 200 条 `TC_PF` 性能用例（含 7 条 Small、26 条 Medium、167 条 Large 及 25 条非默认 `alpha/beta` 用例）采集双口径性能：<br>1. **正式判据（kernel-to-kernel）**：使用 `msprof op` 采集单算子 kernel `Task Duration(us)`，warmup 5 次后连续 10 次有效样本取算术平均，要求单 case 与全部可比 case 平均倍率均 $\ge 0.8$（即 `npu_kernel_ms <= 1.25 * gpu_ms`）；典型达标上限包括：$1^3 \le 9.86\,\mu\text{s}$、$64^3 \le 18.69\,\mu\text{s}$、$256^3 \le 32.51\,\mu\text{s}$、$1024^3 \le 573.44\,\mu\text{s}$、$2048^3\text{ (NC)} \le 4.236\,\text{ms}$、$4096^3\text{ (CN)} \le 27.549\,\text{ms}$；<br>2. **辅助交叉验证**：GTest 通过 `aclrtEvent` 打印 `[PERF] case=<name> ... avg_us=<val>` 并由 `verify_performance.py` 交叉比对（不替代正式判据） | 任务书 §3.3 & `test_cases/gpu_baseline.csv` |
| 内存标准 | 不涉及（任务书 §3.4 明确内存要求不涉及；算子内部 workspace 复用 `aclblasHandle_t` 默认池，不新增对外内存接口；4096³ 场景严格控制设备侧 workspace 占用并在溢出时安全返回 `ACLBLAS_STATUS_ALLOC_FAILED`） | 任务书 §3.4 |

## 兼容性分析

- **接口兼容性**：`aclblasCgemm3m` 公开原型已存在于 `include/cann_ops_blas.h`（L584），本任务严格按该既有原型实现 `arch22` 后端，不修改公开头文件签名，不引入私有平行 API，与 `arch35`（Ascend 950PR）产品线完全共用同一头文件契约；
- **目录与代码隔离**：所有实现文件置于 `blas/gemm/arch22/` 下独立 `cgemm3m_*` 文件中，不侵入 `blas/gemm3m/`（实数 `aclblasSgemm3m` 目录），不影响同目录既有 `aclblasSgemm`、`aclblasCgemm` 及 `arch35` 算子的编译、符号导出与运行行为；
- **构建与测试兼容性**：在 `blas/CMakeLists.txt` 中补齐 `--ops=cgemm3m` 在 `arch22` 下的源文件收集规则，在 `test/gemm/cgemm3m/CMakeLists.txt` 中为 `"arch22" IN_LIST SOC_ARCH_DIRS` 注册 `cgemm3m_test` 目标，保持 `arch35` 原有构建与测试逻辑零回归。
