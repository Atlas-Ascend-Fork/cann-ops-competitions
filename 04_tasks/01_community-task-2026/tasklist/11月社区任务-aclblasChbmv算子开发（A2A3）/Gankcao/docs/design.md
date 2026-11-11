# aclblasChbmv（Atlas A2/A3）设计方案

## 需求背景

### 需求来源

11 月社区任务《aclblasChbmv 算子开发任务书（Atlas A2/A3）》及随包 `test_cases/chbmv_test.csv`、`gpu_baseline.csv`。实现提交至 `cann/ops-blas`。本设计属于 A2/A3 arch22 任务，与已有 950 训练营 Chbmv 任务区分。

官方任务目录索引目前只列至 10 月，尚未列出本任务的编号；本次使用完整任务名称建目录，不推断任务编号。本文是开发计划，实测结果与评审结论后续补充，不表示算子已达标。

### 背景介绍

Chbmv 计算单精度复数 Hermitian 带状矩阵与向量的乘加。矩阵仅存储指定三角的带内元素，另一侧通过共轭对称关系恢复；对角元素虚部不参与计算。需要在有限带宽下减少无效访存，在大带宽下提高向量计算和数据搬运利用率，同时保持 BLAS 的步长、带状排布和异步 stream 语义。

## 需求分析

### 需求描述

公开接口严格采用任务书定义：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

A、x、y 位于 Device；alpha、beta 是 Host 复数标量。y 原地更新，计算提交到 handle 绑定的 stream，调用后由调用方同步并读取结果。实现放在 `blas/sbmv/arch22/`，测试放在 `test/sbmv/chbmv/arch22/`，按主仓结构复用测试公共部分；不改动普通 Ssbmv 语义，不添加旁路公共接口。

### 需求拆解

1. 实现公共声明、参数校验、arch22 dispatch 和 Ascend C kernel。
2. 覆盖 UPPER/LOWER、正负非零步长、带宽超过矩阵阶数、lda padding、标量及特殊值语义。
3. 使用指定 Netlib BLAS 3.10.0 生成高精度 golden，运行原包 1000 条精度和 200 条性能用例，并补齐正态分布及未被 CSV 表达的 handle 空指针测试。
4. 在 910B3 上以 msprof op kernel replay 测量全部性能用例，逐项对比原包 GPU 表，保留完整采样而非择优取值。
5. 输出可复现构建/运行说明、自验报告、真实终端截图和模板材料；代码 PR 与自验材料分开管理。

## 详细设计

### 算子分析

#### 数学公式

`y[i] = alpha * sum(A[i,j] * x[j]) + beta * y[i]`，有效 j 范围为 `[max(0,i-k), min(n-1,i+k)]`。

令 `r = min(i,j)`、`c = max(i,j)`：

- UPPER 存储 `A(r,c)` 的地址是 `k + r - c + c*lda`；当 `i>j` 时取其共轭。
- LOWER 存储 `A(c,r)` 的地址是 `c - r + r*lda`；当 `i<j` 时取其共轭。
- `i==j` 时只取实部，对角虚部即使是 NaN/Inf 也不能进入计算。
- 当 `k>n-1` 时只截断遍历范围，UPPER 的存储偏移仍使用原始 k。

复数乘法按实虚部 FP32 展开，累加采用向量分段归约。是否使用乘加融合以实际编译链和精度结果记录，不用降低精度或修改 golden 掩盖差异。

#### 支持数据类型与形状

仅支持接口规定的 `aclblasComplex`（complex64，实虚部 FLOAT32），无额外混合精度和广播。A 的物理形状为 `lda*n`；x/y 的物理长度至少是 `1+(n-1)*abs(inc)`。负步长的逻辑起点位于 `(n-1)*abs(inc)`，设备访问使用足够宽的地址计算。n、k 是运行时参数。

### 算子实现

#### Host 侧设计

校验顺序固定为：handle 非空；n 非负；n==0 直接成功；其余参数检查。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，其余非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`，特别是非法 uplo 不沿用同族算子的 INVALID_ENUM。

- k 非负；在 int64 域检查 `lda >= k+1`。
- incx/incy 不为 0 或 INT32_MIN，在 int64 域检查 `(n-1)*abs(inc) <= INT32_MAX`。
- n>0 时 alpha、beta、A、x、y 均非空，即使 alpha 或 beta 为零也不豁免指针校验。
- n==0 时不访问其他输入，不启动 kernel。

从平台获取 AIV 核数和 UB 容量，根据 n、有效带宽及步长划分输出行与 tile。小规模优先少核、单次启动，避免固定开销主导；中大规模保持足够并行度。tiling 和标量按仓内惯例传递，常规路径不引入 Host 同步和重复 Device 分配。

#### Kernel 侧设计

初版按输出行归属划分核，每行由唯一核心负责。分离读取存储三角与共轭镜像段，避免跨核原子累加和 y 写竞争。A 的列内连续段使用批量 DMA；镜像方向结合带状布局分块搬运，再在 UB 内抽取。x 窗口复用，复杂步长通过分块读取/索引处理，避免逐元素 GM 往返。不能读取带外无效区、未引用三角或 lda padding 参与算术。

UB 缓冲包含 A tile、x 窗口、实虚部累加和输出暂存，按实际设备预算确定 tile；搬入、向量计算、搬出之间设置明确的依赖与缓冲复用同步。非对齐尾块采用合法的 DataCopyPad 或仓内等效方式，禁止跨输出范围写回。

- k==0：只处理实对角与复数 x，作为低开销路径候选。
- 窄带：以连续输出行块摊薄搬运及启动开销，减少活跃核数量。
- 宽带：增大连续段搬运，使用分段向量归约；仅在 profiler 证明必要时增加分段部分和路径，其额外 kernel 与临时缓冲计入完整成本。
- beta==(0,0)：不读取旧 y，防止未初始化值或 NaN 污染。
- alpha==0：允许合法短路，但必须先完成参数校验，并遵守任务特殊值用例约定。

不会把带状 A 展开为完整 n×n 矩阵作为常规计算路径。优化分支由尺寸、布局和平台资源决定，禁止按 case 名称或期望结果特判。

### 支持硬件

| 硬件 | 适配范围 | 本次自验证计划 |
| --- | --- | --- |
| Atlas A2 / 910B3 | arch22 | 精度与性能实测 |
| Atlas A3 | arch22 | 保持接口/架构兼容；未实测不得声称通过 |

任务书允许自验覆盖一种款型。软件固定 CANN 9.1.0，记录实际 driver、toolkit/ops、编译器和库版本、编译参数及二进制 SHA256。

### 算子约束限制

无超出 lda/incx/incy 的视图或广播承诺；不要求额外确定性模式。保留原始带宽与有效带宽的区别。中间地址和 Host 校验避免整数溢出。任务书 §3.4 不设内存专项验收标准，模板相关项目标注不涉及，不虚报完成。

## 可维可测分析

### 精度标准与 golden

按任务书 §3.2、§3.5 和测试说明，将原始 complex64 输入及标量提升为 complex128，调用 **Netlib BLAS 3.10.0 的 cblas_zhbmv，CblasColMajor**，最后将输出实虚部收窄为 FLOAT32。测试说明已明确这个高精度实现，不能实际使用 complex64 cblas_chbmv 后声称 FP64 真值。记录 Netlib 源码版本、编译方式及实际链接库哈希。

实部、虚部分别判定：`rtol=atol=2^-13`，有限值 matched_ratio≥0.99，最大绝对误差≤`max(0.01,32*ULP(golden_at_max_error))`。保留原包 legacy MERE/MARE 字段，但不将其作为本任务判据。

非有限值独立统计：双 NaN、同号 Inf、NaN 与任意符号 Inf 等效；有限值与非有限值、异号 Inf 不等效。所有非有限值位置均需通过。对角虚部、带外及 padding 另设污染验证，检查未引用区确实未参与计算。

原包 1000 条精度用例保持不变，另补 Gaussian 数据（mu∈[-5,5]、sigma∈[0.1,2]），按对应合法随机输入几何补齐，使均匀/正态覆盖各 50%；报告分别列原包、补充、非法参数及特殊值数量。固定种子和输入生成规则，必要时保存失败输入便于复现。

### 性能标准

| 项目 | 计划与判据 |
| --- | --- |
| GPU 标杆 | 随包 gpu_baseline.csv，200 条，A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4；保留原表及哈希 |
| NPU 采样 | msprof op，显式 kernel replay，task duration，单位 us |
| 次数 | 区分工具自带 5 次 warmup 与应用预热；计划 5 次应用预热、64 次有效采样，满足正文≥10次要求 |
| 统计 | 完整有效样本算术平均，不取最小值或筛选轮次；记录样本数及波动 |
| 门槛 | 每 case `GPU时间 / NPU平均时间 >= 0.8`；GPU ms 换算为 us |
| 完整成本 | 若存在多个必要 kernel，逐次汇总其 task duration，不能仅展示主 kernel |
| 状态恢复 | y 原地修改，每次独立调用恢复同一 y 输入；核查 replay 工具恢复行为，避免连续 beta 累加改变测试对象 |

附件 verify_performance.py/README 将 event 时间作为判据，与任务书 §3.3 不同。本次依任务书正文及用户指定采用 msprof op kernel replay；event 仅作辅助观测，不混用两种结论。每批保存命令、replay 配置、缓存配置、采样表、工具版本、被测 commit 和库 SHA256；有效采样要求接口成功，失败或无有效 kernel 不得按零耗时计通过。

先验证小 n、k==0、窄带、宽带、UPPER/LOWER 等代表性用例，再结合流水和瓶颈优化未通过项。全量覆盖完成后，只对变更影响范围进行必要回归；修改数值路径须重测相应精度。

### 兼容性与交付

公共接口、返回码及文件布局与当前 ops-blas 对齐；新增 Chbmv 不改变 Ssbmv 或其他算子行为。提交设计 PR 后开始迭代，评审意见持续回填，未经评审不标记设计通过。

阶段交付为：接口/构建及 golden 核验 → 代表性功能与 kernel replay → 原包全量及补充精度 → 全量性能与证据 → 代码 PR 与模板验收包。每阶段只报告实际结果，不承诺未测性能。开发完成提交验收时同步提出官方代码 PR；不等待验收通过再启动合入，也不提前进行与任务无关的重构。
