# aclblasGemmEx / aclblasSgemmEx / aclblasCgemmEx A2/A3 设计文档

日期：2026-10-28。设计基线：ops-blas `05b96ea22cea926ecfd43fc7ac691a795522bd61`。
本文按社区 `design_template.md` 的四个主要章节组织，描述接口、计算路径与验证方案。完整精度和性能验收仍在进行。

| 项目 | 内容 |
| --- | --- |
| 队名 | `gai0718` |
| 任务批次 | 10 月社区任务 |
| 任务名称 | aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 算子开发（A2/A3） |
| 设计 PR | [!2407](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2407) |
| 实现仓库 | [gai0718/ops-blas](https://gitcode.com/gai0718/ops-blas)，分支 `community/gemmex-a2-a3` |

## 需求背景（required）

### 需求来源

任务包 `aclblasGemmEx_SgemmEx_CgemmEx_A2_task_doc.md` 要求在 Atlas A2/A3 上用 Ascend C kernel 直调完成三个扩展矩阵乘法接口。实现交付到 `blas/gemm_ex/arch22/`，测试交付到 `test/gemm_ex/`，公共声明放在 `include/cann_ops_blas.h`。

### 现状分析

基线仓库 GemmEx 只有 arch35。GemmEx 已有 19 参声明；SgemmEx 是带 algo 的 15 参强类型接口；CgemmEx 尚无声明或实现。此次在 arch22 新增独立实数计算后端，直接表达物理 lda/ldb 与转置，普通 `aclblasSgemm` 保持原实现。

公共 SgemmEx 调整为 17 参时，同步更新 arch35 定义和相关调用。950 GemmEx 的既有类型分派继续由 arch35 维护。

## 需求分析（required）

计算 \(C\leftarrow\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C\)。\(\operatorname{op}(A)\) 为 \(m\times k\)，\(\operatorname{op}(B)\) 为 \(k\times n\)，C 为 \(m\times n\)。矩阵采用列主序，C 原地更新。

| 接口 | 参数数 | Host 标量 | 当前 arch22 支持集 |
| --- | --- | --- | --- |
| GemmEx | 19 | `const void*` 指向 FP32 | A/B/C 全 `ACL_FLOAT`、`ACLBLAS_COMPUTE_32F`、`ACLBLAS_GEMM_DEFAULT` |
| SgemmEx | 17 | `const float*` | A/B/C 全 `ACL_FLOAT`，无 computeType/algo |
| CgemmEx | 17 | `const aclblasComplex*` | A/B/C 全 `ACL_COMPLEX64`，无 computeType/algo |

FP16/BF16/FP8 等其他组合在当前 arch22 实现中返回 NOT_SUPPORTED。配套 CSV 的成功用例均采用上表主路径，扩展实现与验收范围后续单列。

### 公共函数原型

```cpp
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const void* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const float* beta, void* C, aclDataType Ctype, int ldc);

aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const aclblasComplex* beta, void* C, aclDataType Ctype, int ldc);
```

### 参数说明

| 参数 | 方向/位置 | 类型与含义 |
| --- | --- | --- |
| handle | 输入/Host | `aclblasHandle_t`，有效句柄，携带 stream/workspace |
| transA | 输入/属性 | `aclblasOperation_t`，A 的 N/T/C 操作 |
| transB | 输入/属性 | `aclblasOperation_t`，B 的 N/T/C 操作 |
| m | 输入/属性 | `int`，op(A)、C 行数，非负 |
| n | 输入/属性 | `int`，op(B)、C 列数，非负 |
| k | 输入/属性 | `int`，收缩维，非负 |
| alpha | 输入/Host | GemmEx 的 `const void*` 指向 FP32；SgemmEx 为 `const float*`；CgemmEx 为 `const aclblasComplex*` |
| A | 输入/Device | `const void*`，列主序只读矩阵 |
| Atype | 输入/属性 | `aclDataType`，A 的 dtype |
| lda | 输入/属性 | `int`，A 的物理列前导维 |
| B | 输入/Device | `const void*`，列主序只读矩阵 |
| Btype | 输入/属性 | `aclDataType`，B 的 dtype |
| ldb | 输入/属性 | `int`，B 的物理列前导维 |
| beta | 输入/Host | 指针类型与该接口 alpha 相同 |
| C | 输入输出/Device | `void*`，列主序原地结果 |
| Ctype | 输入/属性 | `aclDataType`，C 的 dtype |
| ldc | 输入/属性 | `int`，C 的物理列前导维 |
| computeType | 输入/属性 | 仅 GemmEx，`aclblasComputeType_t`，当前 `ACLBLAS_COMPUTE_32F` |
| algo | 输入/属性 | 仅 GemmEx，`aclblasGemmAlgo_t`，当前 `ACLBLAS_GEMM_DEFAULT` |

### 布局与操作

元素按 \(A[r+c\cdot lda]\)、\(B[r+c\cdot ldb]\)、\(C[r+c\cdot ldc]\) 定位，每矩阵使用自己的前导维。N 直接读原矩阵；T 交换行列；复数 C 在交换行列时取共轭，实数 C 与 T 等价。COMPLEX64 为交错 real/imag 存储。

\(m,n,k\geq0\)。A 为 N 时 \(lda\geq\max(1,m)\)，否则 \(lda\geq\max(1,k)\)；B 为 N 时 \(ldb\geq\max(1,k)\)，否则 \(ldb\geq\max(1,n)\)；\(ldc\geq\max(1,m)\)。不涉及广播、视图或额外 stride。

### 校验与边界

顺序：handle → 非负维度 → leading dimensions → alpha/beta → dtype → operation enum → A/B/C 指针 → GemmEx computeType/algo → no-op。

| 条件 | 行为 |
| --- | --- |
| handle 为空 | HANDLE_IS_NULLPTR |
| 非法维度、前导维、必需指针 | INVALID_VALUE |
| 非 N/T/C 枚举 | INVALID_ENUM |
| 不支持 dtype/computeType/algo | NOT_SUPPORTED |
| 全部校验通过，m=0 或 n=0 | SUCCESS，不下发计算 |
| k=0 或 alpha=0 | 仅 beta 缩放，不引用 A/B |
| beta=0 且 C 为空 | 其他校验通过后 SUCCESS，不读写 C |
| beta=0 且 C 有效 | 旧 C 不参与计算 |
| 仅缩放且 beta=1 | C 保持原值 |

alpha/beta 始终非空；k>0 且 alpha 非零时 A/B 必需，beta 非零时 C 必需。

## 详细设计（required）

### Host 与工程组织

`blas/gemm_ex/arch22/gemm_ex_host.cpp` 负责三个公共入口、校验、短路、workspace 分配及 `RunRealGemm` 分派。`gemm_ex_real_kernel.h/.cpp` 实现实数 Cube/vector/epilogue；`gemm_ex_complex_kernel.h/.cpp` 实现复数布局、scale 和 epilogue。源由 `blas/CMakeLists.txt` 按 SOC 自动收集。

两个实数 Ex 入口共用当前目录的 `RunRealGemm`；复数通用路径将四实乘展开到同一 Cube 网格，小工作量路径复用 `RunRealGemm`。全部 kernel 使用 handle stream。极值 FMA helper 通过已有 `gemm_strided_batched_fp32.h` 头内实现复用。

### FP32 计算

Cube 路径使用 `MatmulImpl`，将列主序乘法映射为交换 A/B 后的行主序计算。输入基址或列步长需要对齐时，先按实际 lda/ldb 读取逻辑元素，打包到列步长为 8 个 float 倍数的 workspace，新增尾部填零。`SetOrgShape` 显式传入实际计算缓冲区的物理步长；`SetTensorA/B` 按转置和 tile 起点定位。输出按 32、64、128 轴长分块；短 K 可由一个 256×128 单核任务复用两个 128×128 基础块，窄矩阵的静态 K 上界按 L1 容量选取。运行时始终传入实际 K，实数归约分片上界随配置选择，HF32 关闭。

Cube 路径在 alpha=1、beta=0 时按实际 ldc 直接写逻辑 C。仅缩放分支的 epilogue 按输出规模选择 2048 或 12288 元素 tile；短列在 UB 按对齐步长分组，通过二维 DMA 每列只写 m 个元素。大行数按列分段，GM 列间跨度超出二维 DMA 字段范围时也使用逐列路径；beta 缩放结果通过位拷贝保持。

当 \(mnk\leq4096\)、alpha 非 1 或 beta 非 0 时，采用按 K 逐次 FMA 的 vector 路径。Host 用除法形式比较工作量以控制整数范围。非转置 A 按列连续读，先以 beta 缩放 C，再逐项累加 A 与 alpha 缩放后的 B；转置 A 用 strided copy 和 Gather 取出元素，先累加乘积，再组合 alpha/beta。

输出元素数 \(mn\leq4096\)、参与乘加的输入幅值超过 \(10^{30}\) 或 \(|\alpha|>32\) 时，调用 `GemmSbFp32::Fma` 软件实现，保留完整乘积后做 FP32 舍入。其他 vector 乘加使用 `FusedMulAdd`。转置 A 的最终 beta 组合使用软件 FMA，与 Netlib 的融合表达式保持相同舍入顺序。特殊值、极值和抵消场景由 Netlib 对照验证。

### COMPLEX64 计算

小矩阵采用单 AIV kernel：一次将输入载入 UB，通过 Gather 与向量乘加计算输出。各维不超过 32 且物理跨度满足 UB 容量时，复数路径使用此实现；实数路径还要求默认标量和小工作量。复数的非转置 A、对齐列和默认标量路径在 UB 预生成旋转向量，直接在交错 real/imag 输出上执行每 K 两次 Axpy。其他转置与标量使用分离实虚部的通用向量路径。输入极值在同一调用内转为有序软件浮点计算。复数输出不超过 128 个元素且 K 大于 64 时采用有序复数归约，保留复杂乘积与逐项相加顺序。

vector kernel 的 `ComplexLayout` 使用 GatherMask 拆分实虚部，Gather 重交错，位级 Or 拷贝平面。源 A/B 按实际 lda/ldb 读取，拆分后的 \(A_r,A_i,B_r,B_i\) 保留列主序，列步长按对应逻辑行数向 8 个 float 对齐，新增尾部填零。C 操作在拆分时对对应虚平面取负，再由实数 Matmul 的转置参数处理维度交换。向量依赖处设置 `PIPE_V` 屏障，MTE3 写回前后使用 `PIPE_ALL` 同步。

四个 FP32 GEMM 使用 alpha=1、beta=0，产生列主序、leading dimension 为 \(\operatorname{align}_8(m)\) 的平面。\(mnk>4096\) 时，将四平面的输出 tile 展开到同一个 Cube 网格；总 K 不超过 512 时整段计算，其余按 256 分片，以缩短长归约的平面内累加链。复数 Matmul 的静态 K 上界设为 512，每次 `SetSingleShape` 传入真实片长。每个分片通过 MatmulImpl 写出，首片覆盖、后续片原子累加。该路径依次下发拆分、四乘积和复数 epilogue 三个 kernel。其他形状调用四次 `RunRealGemm`：

$$
P_{rr}=\operatorname{op}(A_r)\operatorname{op}(B_r),\quad
P_{ii}=\operatorname{op}(A_i)\operatorname{op}(B_i),\quad
P_{ri}=\operatorname{op}(A_r)\operatorname{op}(B_i),\quad
P_{ir}=\operatorname{op}(A_i)\operatorname{op}(B_r).
$$

这里虚平面已包含共轭符号。vector epilogue 组合 \(P_r=P_{rr}-P_{ii}\)、\(P_i=P_{ri}+P_{ir}\)，再执行：

$$
C'_r=\alpha_rP_r-\alpha_iP_i+\beta_rC_r-\beta_iC_i,\qquad
C'_i=\alpha_iP_r+\alpha_rP_i+\beta_iC_r+\beta_rC_i.
$$

beta=0 分支直接生成输出；其余分支读取旧 C 的逻辑元素。按 ldc 重交错写回。拆分 tile 按输入规模选择 128、512 或 4096；单位 alpha、零 beta 的 epilogue 按输出规模选择 128、512、1024 或 4096，其余 epilogue 和缩放采用 1024。工作按 block-stride 分配，尾块按实际有效数量搬运。短列在 UB 按对齐步长分组，通过二维 DMA 保留源和目标的列间 padding。单位 alpha 路径保留正负零虚部及零乘 Inf 的传播。

### Workspace 与异步生命周期

采用 handle workspace，库 workspace 由 `EnsureDefaultWorkspace` 增长；用户 workspace 遵循容量检查。设 A 逻辑行列数 \((p_A,q_A)=(m,k)\)（N）或 \((k,m)\)（T/C），B 为 \((p_B,q_B)=(k,n)\)（N）或 \((n,k)\)（T/C）。输入平面长度为 \(S_A=\operatorname{align}_8(p_A)q_A\)、\(S_B=\operatorname{align}_8(p_B)q_B\)；乘积平面按列对齐，\(S_C=\operatorname{align}_8(m)n\)。输入四平面与乘积四平面需 \(4(2S_A+2S_B+4S_C)\) 字节。

后端临时区大小为 \(W_B=\max(64\text{ MiB},4\operatorname{align}_8(m)n)\)，复杂平面从 workspace 的 \(W_B\) 字节后开始。每次追加平面前按剩余容量检查，控制累计偏移在 workspace 上限内，再核对字节转换和后端区相加范围。

先扩容整个区域，再计算平面指针。四次 `RunRealGemm` 使用复制的临时 handle view；stream、workspace 基址与原 handle 一致，但 `workspace_size` 限制为 \(W_B\)、`workspace_owner` 为 `User`。若后端请求超过前段容量，返回 ALLOC_FAILED；view 不重新分配原 workspace，平面地址保持有效。

此分区将后端 scratch 与复杂平面隔开。最终验证包含大型转置/padding、输入/输出平面的 32 字节起址，以及容量不足时的返回值和数据保护。

多次 kernel 依靠同一 handle stream 顺序执行，结果读回前同步该 stream。主计算不进行 Host D2H 求值。

### 支持硬件与环境

| 项目 | 目标 |
| --- | --- |
| 架构 | arch22 / dav-2201 |
| 产品 | Atlas A2、Atlas A3 |
| 自验 | 覆盖一种款型，性能建议 910B3 |
| CANN | 9.1.0 |
| golden | Netlib BLAS 3.10.0：cblas_sgemm / cblas_cgemm |
| 构建 | Release；公共声明同步 arch35 |

## 可维可测分析

### 用例覆盖与追溯

配套三个 CSV 各 1200 项，共 3600 项，其中每接口 200 条 TC_PF。全部保留并迁移为统一 GTest 输入，另补充 70 条用例，共 3670 条：原有 43 条地址偏移和校验顺序用例，加上三个接口各九种转置组合的 Cube/generic pack 偏移用例。新增组使用非对齐 ldc、元素对齐的基地址偏移及 NaN padding，检查逻辑输入与 padding 的隔离。源文件、case、原期望值可追溯。六个非法 dtype 原期望与任务书冲突按 `requirements.md` 显式记录。

覆盖九种 N/T/C 组合、复数共轭差异、尺寸扫描、padding、均匀/正态分布、Inf/NaN、零维/零 K/alpha=0、空指针及非法参数。按全逻辑矩阵比对，padding 独立检查。

### 精度

FP32 与 COMPLEX64 的实部/虚部分别采用 \(rtol=atol=2^{-13}\)，有限值匹配比例至少 0.99。匹配条件为 \(|a-g|\leq atol+rtol|g|\)，每个有限元素还需满足 \(|a-g|\leq\max(0.01,32\operatorname{ULP}(g))\)。alpha=0 的 beta 缩放按 EXACT。

同为 NaN、同号 Inf、NaN 与 Inf 互配按任务书判为等效；有限/非有限以及异号 Inf 不匹配。非有限值独立计数，全部满足规则后通过该部分，不计入有限误差统计。

Golden 主路径分别调用 Netlib `cblas_sgemm` / `cblas_cgemm`，相同输入及 leading dimensions。保存退出码、GTest XML、执行数量、逐 case 指标与缺失/跳过项。

### 性能

任务正文为 10 次，交付及 §7 要求 >10 次；采用 5 次 warmup、11 次有效采样。以 `msprof op` task duration 为验收依据，一次 API 内全部 kernel 求和，再对 11 次调用取平均。GPU `gpu_ms` 转 us 后比较 \(T_{NPU}\leq1.25T_{GPU}\)。共 600 条逐 case 判定，保留 profiler 原始数据和聚合映射。ACL event 作为单独诊断列。

### 审阅与提交清单

- [ ] 19/17/17 参数数、顺序、枚举符合 §2.3，arch35 SgemmEx 同步。
- [ ] CANN 9.1.0 arch22 单算子构建成功，公共头的 arch35 编译兼容性已检查。
- [ ] 动态 workspace 前段和对齐后的复杂平面互不重叠，临时 handle view 容量检查通过，大型转置/padding 已验证。
- [ ] 3600 项原始用例及 70 项补充用例具备数量、状态、精度、padding 和完整日志。
- [ ] 600 条性能具备 5 次预热、11 次有效完整调用、kernel 聚合证据及 GPU 判据。
- [ ] 设计、README、自验步骤、报告、日志与最终源码版本一致。
- [ ] 设计评审、任务平台验收、代码 PR 和合入分别记录外部状态。

设计 PR [!2407](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2407) 已提交，当前状态为开放，评审结论尚未确认。代码 fork `gai0718/ops-blas` 和分支 `community/gemmex-a2-a3` 已创建；实现与验证继续推进，最终精度、性能和验收结果记录在自测报告中。
