# aclblasGemmGroupedBatchedEx（Atlas A2/A3）设计文档

## 需求背景

### 需求来源

本设计依据[10月社区任务](https://www.hiascend.com/activities/task-center/details/36dbb2d3e38d4b4c96fc71b4869cd3e7)及其任务书、`test_cases/README.md`、用例 CSV 和 GPU 基线。任务网页发布的 ZIP 原文未提供可关联的 Git commit，版本依据为该网页附件。设计结构参考[社区设计模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

### 背景介绍

分组批量 GEMM 在一次句柄式调用中表达多个异构矩阵乘法分组。组内矩阵共享形状、转置、前导维和标量，组间可不同；实际矩阵通过设备指针数组传递。Atlas A2/A3 实现需要兼顾小矩阵的批处理开销、大矩阵的 Cube 吞吐以及零维和零标量的 BLAS 语义。

实现放在 `cann/ops-blas` 的 `blas/gemm_grouped_batched_ex/arch22/`，测试放在 `test/gemm_grouped_batched_ex/arch22/`。复用公共头 `include/cann_ops_blas.h` 中现有声明，与其他架构共用 API。

## 需求分析

### 接口与存储语义

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray,
    void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType);
```

第 g 组第 i 个矩阵的全局下标为 `p = sum(groupSize[h], h < g) + i`，计算：

`C[p] = alpha[g] * op(A[p]) * op(B[p]) + beta[g] * C[p]`。

| 项目 | 约定 |
| --- | --- |
| 主路径 | A/B/C 均为 `ACL_FLOAT`，计算类型 `ACLBLAS_COMPUTE_32F`；其他组合返回 `NOT_SUPPORTED` |
| 布局 | 列主序；物理元素地址 `row + column * ld`；C 原地覆写，padding 不修改 |
| 转置 | N/T/C 全覆盖，实数下 C 与 T 等价 |
| Host 数组 | transa/transb、m/n/k、lda/ldb/ldc、groupSize；每列长度为 groupCount |
| 设备数组 | Aarray/Barray/Carray 本身和各矩阵均在 Device，按组连续编号 |
| 标量 | alphaArray/betaArray 为连续 FP32 标量数组，支持 Host 或 Device 存储；不将其解释为标量指针数组 |
| 执行 | 使用 handle 绑定的 stream 下发计算，调用方在读回前同步 stream |
| 范围 | 不额外支持广播、任意非连续视图、复数或未经实现验证的混合类型 |

任务书同时使用“标量数组”和“指针数组”描述标量；按公共 `const void*` 接口及测试 CSV 的逐组连续标量语义实现，在 README 明确内存布局。

### 合法性和边界

1. handle 为空返回 `HANDLE_IS_NULLPTR`；随后检查 dtype/computeType 组合，不支持返回 `NOT_SUPPORTED`。
2. groupCount<0 返回 `INVALID_VALUE`；groupCount=0 直接 SUCCESS，不检查后续数组。
3. groupCount>0 时检查所有 Host 数组及 alpha/beta 非空；逐组检查转置枚举、非负维度和 groupSize。非法枚举返回 `INVALID_ENUM`，非法数值返回 `INVALID_VALUE`。
4. 前导维满足 `lda >= max(1, transa==N ? m : k)`、`ldb >= max(1, transb==N ? k : n)`、`ldc >= max(1,m)`。先验证全部组，再执行任何写操作。
5. 以宽整数累计 problemCount；当其大于0时检查三个设备矩阵指针数组非空，包括零 m/n 的分组。各组 m=0、n=0 或 groupSize=0 时不计算。
6. k=0 或 alpha=0 时不读取 A/B 矩阵，也不读取其单个指针；beta=1 保持 C 位模式，beta=0 直接写零且不读旧 C，其余 beta 缩放。beta=0 的正常乘法也不读取旧 C。
7. 裸 C 数组接口没有长度参数，无法识别“多出的数组元素”；测试 CSV 在创建指针前严格校验各列长度，拒绝不一致输入，避免把测试加载器校验当作接口返回值。

资源尺寸、任务前缀、工作区偏移及字节乘法使用64位并检查溢出；分配失败返回 `ALLOC_FAILED`，运行时错误按库状态码上报。不得截断任务或以 no-op 掩盖错误。

## 详细设计

### 算子分析

每个矩阵计算复杂度为 O(mnk)，整体为各组 `groupSize[g] * m[g] * n[g] * k[g]` 的和。矩阵之间没有数据依赖，每个输出 tile 只归一个核所有，避免跨核规约和原子累加。

列主序映射为行主序的 `C^T = op(B)^T * op(A)^T`，在 Cube 输入顺序中交换 B/A，同时保持前导维、转置和偏移映射一致。FP32 输入与累加保留 FP32，关闭 HF32；不以 FP16/BF16 截断输入来提高主路径吞吐。

小尺寸、退化缩放使用 AIV，较大尺寸使用 AIC 的 Ascend C Matmul。对适合直接写出的稠密小矩阵，可使用 Cube 并随后在 AIV 扫描输入指数位，遇到极大值、Inf/NaN 时按 Netlib 逐 K 顺序重算；普通输入保留 Cube 输出。特殊值比较遵循任务书规则。

### Host/Tiling 设计

将分组映射为描述符，记录 m/n/k、lda/ldb/ldc、转置、矩阵指针数组起始下标、实例数、Cube/AIV 任务区间、工作区偏移和标量。设备标量保留地址，由 kernel 在 handle stream 的正确执行时刻读取，避免 Host 在异步前序标量更新完成前读到旧值。

优先使用按值传递的小批分组描述符，限制单次 kernel 参数体积；超过批容量的组顺序拆为多个 kernel 调用，语义上不限制总 groupCount。该方式避免为短暂的 Host 元数据建立同步释放的设备分配。块数从平台实际 AIC/AIV 数查询，不写死910B3核心数。

Cube 初始 tile 采用128×128，K按 Matmul 支持范围分段并以 FP32 累加；尾块以实际有效 m/n/k 设置，跨段只在首段清零。任务区间按“分组前缀—实例—二维输出 tile”定位，核以 block-stride 方式遍历，避免对每个矩阵逐次 Host 下发。

小规模 AIV 路径以“实例—输出列—行段”分配任务；行段长度按 UB 预算选择，空分组不分配任务。全部逻辑偏移与物理 ld 分离，padding 不参与数值比较但另检查未写坏。

需要 epilogue 的 Cube 输出暂存于 handle 工作区，按32字节对齐排列。Host 预先计算工作区上界并检查容量；仅在库管理模式允许增长，用户提供的工作区不足时报错。分组批次复用同一工作区，由同 stream 上 Cube→epilogue→下一批的顺序保证生命周期；函数返回不提前释放仍在使用的临时空间。

对 Host alpha=1、beta=0 且输出布局满足直接写出约束的路径，可直接写 C，减少中间结果与额外 kernel。对 Device 标量优先保留通用 epilogue，确保每次调用读取实际值。

### Kernel 设计

Cube kernel 从分组区间和设备指针数组获取对应矩阵，按转置计算输入偏移，设置原始 leading dimension 和实际单核 shape；保持 FP32 Matmul，覆盖尾 M/N 和分段 K。alpha=0 或 k=0 的组跳过矩阵乘和 A/B 指针读取。

AIV kernel 同时承担小规模 GEMM、beta 缩放和 Cube epilogue。读取 C 前先分支 beta；beta=0 时直接产生结果。小尺寸输入通过 DMA/局部缓冲加载连续行段，转置路径按正确物理下标访问，累加使用 FP32。k=0/alpha=0 的缩放分支独立于矩阵指针路径；beta=1 不写回，保证 NaN payload、符号零等原位值保持。

epilogue 从对齐的 FP32 中间矩阵读取有效行段，执行 `alpha*product + beta*C`，仅写 `m*n` 逻辑元素。搬运长度按有效元素数量设置，不用越界对齐写覆盖相邻列或相邻矩阵。局部缓冲复用前执行相应流水同步；Cube 与 AIV 通过 stream 上独立 kernel 顺序建立依赖。

### 硬件与约束

目标为 Atlas A2/A3 的 arch22，CANN9.1.0；性能采集选 Atlas800T A2（910B3）。AIC处理矩阵乘，AIV处理小矩阵和缩放；GM存储设备指针数组和矩阵，UB存放行段、输入和累加缓冲，Matmul 管理 L1/L0 及搬运流水。

初始 tile 和缓冲量需通过编译器资源报告及目标设备核验。AIV 常驻缓冲按128元素行段规划；小矩阵指数检测覆盖 A/B 合计最多1024个 FP32 元素，输入、掩码、归约缓冲及行段结果合计约13KB UB，避免将一般尺寸整矩阵放入 UB；DMA 的32字节对齐与有效尾长分别处理。不同产品的平台核数和支持能力通过平台查询确认。自验只需一种款型；未实际测到的产品不编造测试结果。

## 可维可测分析

### 工程与兼容性

arch22 使用本架构的 Host、kernel 和 tiling 头，自动纳入 ops-blas 现有编译收集。公共 API 原型复用，不新增产品私有入口。测试沿用 GTest/CSV 注册，新增 arch22 的严格字段加载和 Netlib golden，保留 arch35 的既有测试及类型能力。

README 分别写明产品支持和各架构 dtype 支持范围。记录编译命令、完整代码版本、Netlib版本和动态库来源，不依赖开发机临时补丁。

### 精度验证

覆盖附件全量1000精度和200性能用例；补充未覆盖的 host ldb/ldc 空指针、Device标量及其 stream更新、混合零组、alpha=0 时 A/B单指针不可读、beta=0 且旧C为NaN、非对齐地址、padding保护和空运行防护。

Golden 固定为 Netlib BLAS3.10.0 的 `cblas_sgemm`，逐组逐矩阵生成。保留原始 C，逻辑输出全矩阵比较：有限值 atol=rtol=2^-13、matched_ratio>=0.99，并报告绝对误差及32ULP口径。同时检查逐元素绝对误差不超过 max(1e-2, 32×ULP(golden))，并给出可核对ULP统计；不以 CSV 的 MERE/MARE 替代任务书规则。

非有限值单独计数：同NaN、同号Inf、NaN与Inf等效通过；有限与非有限不匹配、反号Inf失败。有限值统计不纳入非有限元素，无有限元素时只判断特殊值。alpha=0 缩放按位校验；no-op 检查 C 不变。非法参数验证真实返回码；CSV结构错误单独记录输入校验结果。

### 性能与空间验证

性能仅使用 `msprof op` 的设备 task duration（us），不使用 GTest墙钟时间。为覆盖资料中不同采样要求，保留5次预热后60次有效调用，另独立采集10次均值。一次接口如产生多个kernel，按调用归属相加后统计，不只选最快或单个Cube。

将原基线 `gpu_ms` 转为 us，逐case报告 GPU/NPU 比例、>=0.8和>=1.0两种口径，以>=1.0作为保守开发目标；保留基线GPU型号描述不一致的事实并向验收方核实，不自行重写来源。记录设备负载、每次调用的status、有效采样数和异常样本。

记录实际工作区需求与分配值。任务书内存条款“不涉及”与交付表的workspace/GPU<=1.2要求不一致，缺少GPU workspace基线时注明待确认，不填测量常量或声称空间通过。

### 复现与交付

所有用例、输入生成器、GTest、Netlib部署、msprof解析及材料生成入口纳入同一验收完整commit或随附件提供。保存原始CSV、GPU基线、固定种子、运行参数、逐case结果、日志和要求的截图；报告从本次输出生成，清空模板示例数据。

交付《自验证步骤说明》和《复现附件说明》，说明从完整commit获取代码、部署环境、准备固定输入、执行精度/性能、解析结果并生成报告的顺序命令。最终在独立检出目录完整运行，刷新任务网页和附件后核对要求，再生成不加密的DEFLATE/STORE ZIP，检查100000000字节上限、全包算法、CRC和完整解压逐文件一致性。
