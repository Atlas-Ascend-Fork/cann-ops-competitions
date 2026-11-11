# aclblasSgemv（Atlas A2/A3）设计文档

## 需求背景（required）

### 需求来源

[11月社区任务-aclblasSgemv算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/a8c0b7d0985349c59503ec0c34ca7344)，任务讨论入口为 [ops-blas discussion 79](https://gitcode.com/cann/ops-blas/discussions/79)。依据官网任务包 `8caf097923a04a20bfaeee0cce0ede06.zip` 内 `aclblasSgemv_AtlasA2_A3_task_doc.md` 及 `test_cases/`。任务包未发布对应 Git commit，不用下载时间或哈希替代任务书 commit；提交验收前重新下载核对变化。任务归属 2026 年 11 月，官方资料未给出编号，目录采用已确认的无编号 `11-aclblasSgemv-A2A3`。

### 背景介绍

在 [ops-blas master](https://gitcode.com/cann/ops-blas/tree/master) 补齐 Atlas A2/A3 的 FLOAT32 通用矩阵向量乘法。沿用公共 `include/cann_ops_blas.h` 中的 `aclblasSgemv` 声明，采用 Host 参数校验、值传递 tiling 和 Ascend C AIV kernel 直调。实现放置 `blas/gemv/arch22/`，测试放置 `test/gemv/sgemv/arch22/`，与既有 arch35 实现分架构构建，不添加平行私有接口。

## 需求分析（required）

### 接口与数据约定

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, const float* x, int incx,
    const float* beta, float* y, int incy);
```

| 参数 | 位置、类型与约束 |
| --- | --- |
| handle | Host 库句柄，使用其绑定 stream，不能为空 |
| trans | Host 枚举 N/T/C；实数 C 等价于 T |
| m、n | 非负 int32；零维度在校验通过后快速返回 |
| alpha、beta | 非空 Host FLOAT32 标量指针，允许零、负值、Inf、NaN |
| A | Device FLOAT32，只读列主序，物理长度 lda×n，A(i,j)=A[i+j×lda]，0≤i<m、0≤j<n |
| lda | int32，至少 max(1,m)，大于 m 时列间存在填充 |
| x | Device FLOAT32，只读，N 的逻辑长度 n，T/C 的逻辑长度 m |
| y | Device FLOAT32，原地更新，N 的逻辑长度 m，T/C 的逻辑长度 n |
| incx、incy | 非零 int32 且不为 INT32_MIN；正负步长均支持；向量最大元素偏移不得超过 INT32_MAX |

令向量长度 L、步长 s，则物理长度为 `1+(L-1)*abs(s)`；s<0 时逻辑首元素从 `(L-1)*abs(s)` 起，后续依次加 s。所有跨度、矩阵偏移及字节数以 int64/uint64 计算。逻辑长度为零允许相应向量指针为空；A 仅在 m>0 且 n>0 时必须非空。标量不支持 Device 指针，这是本任务的 Host 标量接口范围。

### 参数校验与返回码

首先 handle 判空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；随后验证 trans、非负 m/n、lda、步长、标量指针、按逻辑长度决定的数据指针和 int32 跨度。以上参数错误均返回 `ACLBLAS_STATUS_INVALID_VALUE`。特别是 `m=0` 不掩盖非法 lda/步长或空标量；全部校验通过后才判断 `m==0 || n==0`，返回 SUCCESS，保持 y 不变且不发射 kernel。

数值本身不做合法性检查；`beta==0` 不读取旧 y；`alpha==0` 不跳过 A/x 计算，不引入 `alpha==0 && beta==1` 快速返回。成功下发后返回 SUCCESS，读取输出前由调用方同步 stream。平台资源查询失败或运行时下发失败按仓库可获取的错误返回机制处理，不能把异步返回成功当成设备计算完成。

### 任务书 §2.5 约束核对

| 约束 | 设计处理 |
| --- | --- |
| 参数合法性和校验层级 | 使用上述顺序，补齐 arch35 参考中未提供的跨度溢出守卫 |
| 非连续 Tensor | 按 lda 跨列、按带符号步长访问向量，不把 lda 误当行间距；负步长从物理尾端起算 |
| broadcast | 不涉及操作数广播；kernel 内对同一 x 元素的复用不改变接口形状 |
| dynamic shape | m/n 等为运行时参数，Host 当次生成 tiling，无需图模式动态 shape |
| 原地与视图 | y 原地址回写，A/x 只读；不承诺 A/x 与 y 交叠的额外别名语义 |
| 确定性 | 单次计算固定块内归约顺序；不承诺跨架构或不同 tiling 的逐位一致 |
| 空 Tensor | 校验后零维度无 kernel、无数据读写 |
| 0 维 Tensor | 不涉及；alpha/beta 是 Host 标量 |
| 异步 | 普通计算不做 Host 同步或临时设备缓冲的立即释放；多个阶段在同一 stream 排序 |

## 详细设计（required）

### 算子分析

令 K 为归约长度、O 为输出长度。N 时 `(K,O)=(n,m)`，T/C 时 `(K,O)=(m,n)`：

```text
y[j] = alpha * sum(i=0..K-1, op(A)[j,i] * x[i]) + beta * old_y[j]
```

计算量约 2mn FLOPs，至少读取 4mn 字节矩阵，主要瓶颈为矩阵访存、短向量发射开销、极端矩形并行度及长归约误差。N 沿矩阵行方向连续搬运，T/C 沿归约方向连续搬运，两种路径分别组织数据，避免把所有矩阵元素转置到 GM。全程 FLOAT32，不将输入降为 FP16 或 TF32 计算。采用分块乘积与树形/分层归约；长归约必要时使用补偿累加，特殊值避免补偿中的 Inf-Inf 引入无关 NaN。

### Host/Tiling 设计

tiling 结构只保存 m、n、lda、归一化转置标记、int64 步长与向量起点、alpha/beta、输出 tile 大小及可选归约分片数。小结构按值传入 launch stub，避免每次 malloc 和同步上传 tiling。查询实际 AIV 核数，输出 tile 循环按 blockIdx/blockNum 分配，核数上限由设备决定，小规模减少空核。

N 路径初始考虑每 tile 32/64/128 个输出行，归约块 64/128 列；每列有效行按列主序连续搬运。T/C 路径初始考虑每 tile 1/4/8 个输出列，归约块 512/1024/2048 个连续行，复用对应 x 分段。具体 tile 根据合法对齐、UB 容量、有效 DMA 块数和设备测试确定，调参不能识别测试编号或缓存输入输出。

输出极少且 K 很长时，可将 K 分片给多个 AIV，各片写独立 workspace，再在第二个 kernel 合并并且仅一次应用 alpha/beta。workspace 按 `splitCount * alignedOutputCount * sizeof(float)` 规划，使用句柄已有工作区并检查容量，不将局部内存地址传给异步 kernel；工作区不足则退回不分片路径。不得使用缺少跨核同步的单 kernel 自旋或无初始化的原子累加。正常输出并行路径无需 workspace。

### Kernel 设计

#### N：列主序连续行 tile

按输出行段取得若干列的连续数据；x 按逻辑归约索引读取并广播到各行，执行向量乘法。将相邻列的乘积以树形 Add 合并，跨归约块再累加，降低逐元素串行累加误差。x 紧密且正步长时批量 DMA 搬运；一般步长以受边界限制的 gather 或标量加载构成局部连续向量。输出向量先乘 alpha，再在 beta 非零时读取旧 y、执行乘加。

#### T/C：连续归约 tile

对每个输出列分段读取 A 的有效 m 行，加载相同逻辑区间的 x，向量 Mul 后 ReduceSum，分层合并部分和，再应用 alpha/beta。C 路径与 T 共用代码，不进行无意义的共轭操作。小 O、大 K 可启用上述归约分片策略；各分片只读 A/x，最终 kernel 独占输出。

#### 对齐、尾块与同步

使用 Ascend C `DataCopyPad`/扩展 DMA 参数搬运非对齐有效字节；GM 访问长度必须以实际有效元素限制，不能向上取整后越界读取。UB 中填零的尾部参与归约前必须确认不会与 Inf/NaN 相乘而改变有效结果；不将矩阵列填充纳入计算。局部行跨度按 8 个 float（32 字节）对齐，向量计算按 64 个 float repeat 分解，超过 repeat/count 限制时分段。

负步长输入以 int64 地址映射到物理 span，不生成负 DMA stride；必要时先读正向物理段再局部反排。非单位步长输出仅回写逻辑元素并保留间隔值，采用精确字节写入以防相邻核 32 字节块重叠。输入搬入、向量计算、标量局部访问、输出搬出之间显式建立 MTE2/V/S/MTE3 依赖；复用缓冲前等待对应消费者完成，不只依赖代码顺序。

#### 资源预算

初始 N tile 取 64×64，单个矩阵缓冲 16 KiB，乘积/广播缓冲各 16 KiB，索引缓冲 16 KiB，x/y 与累加约 4 KiB；即使矩阵双缓冲，计划总量不超过 128 KiB。T tile 取 8×2048，矩阵缓冲 64 KiB，乘积缓冲 64 KiB，x、归约临时区与输出不超过 24 KiB，计划总量不超过 160 KiB。最终按 SDK 分配和编译资源核实，保守控制在 A2/A3 的 192 KiB 可用 UB 预算内；不能仅按理论容量配置所有缓冲。选择 AIV SIMD，无需 Cube、FP16 转换或 arch35 SIMT 专用指令。

### 硬件与约束

适配 Atlas A2/A3（arch22），CANN 9.1.0；正式自验和性能采集使用任务要求的 910B3。硬件核数用运行时平台信息确定。A2/A3 共用 arch22 代码，A3 未实际测试时须明确标记；支持列表不代替测试结论。构建沿用仓内自动收集算子源文件及 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`，核查当前版本是否已统一注册，若已经支持 arch22 则复用而不重复注册。

## 可维可测分析

### 模块与维护

实现控制为 Host、kernel 与共用 tiling 头三部分；测试复用公共 CSV 参数结构和 CPU cblas golden，arch22 专用测试负责安全分配、精度统计、性能采样和结果落盘。共享接口、arch35 源文件及既有算子行为保持兼容；仅在确有调用依赖时修改公共构建文件。

### 精度与边界验证计划

完整执行官方 1000 条精度 CSV，并补充正态分布及任务参数域缺口。输入保留随机种子、分布、逻辑/物理长度；负向和 quick return 在造数/设备分配之前调用真实 API，以单元素 dummy 或空指针占位，避免 INT32_MIN、INT_MAX 跨度在测试侧触发溢出或巨量分配。单独覆盖真实空 handle、非法枚举、非对齐 lda、正负 inc、溢出、标量/矩阵/向量空指针、零维度优先级、beta=0 忽略非有限旧 y、同 stream 连续异步调用及间隔元素保持不变。

golden 按任务书 §3.5 与 §7 最终说明采用 `cblas_sgemv` FLOAT32；另以 float64 计算辅助分析长归约误差，并分别标明来源，不将辅助结果冒充 cblas 输出。有限元素逐元素阈值为 `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)`，匹配比例至少 0.99，且最大绝对误差不超过 `max(1e-2, 32*ULP)` 的任务允许范围，报告实际采用的 ULP 定义及最大误差。非有限值单独统计：同 NaN、同号 Inf、NaN 与任意 Inf 等效；有限/非有限或相反号 Inf 失败；非有限元素全部通过才接受该例。全非有限输出不以空有限集合误判失败。

官方附件中若存在任务书明确排除的 `alpha=0` 且 A/x 含 Inf/NaN 组合，保留原始 CSV并记录具体用例、冲突依据与修正映射，以有限输入替换冲突条件或另外给出公式 oracle 检查，不静默丢例、不放宽精度标准。正态分布补充明确 μ∈[-5,5]、σ∈[0.1,2]，不因 token 名称正确就认为覆盖充分；补充用例不改变 200 条官方性能参数与 GPU 基线关联。

### 性能验证计划

200 条性能用例与原始 `gpu_baseline.csv` 按 `(trans,m,n,alpha,beta,lda,incx,incy)` 全键关联。判据为每例 `NPU_avg_us <= GPU_ms*1000/0.8`，缺样本、下发失败、缺基线不能判通过。设备负载、预热次数、有效样本数和计时边界随报告记录。

任务书 §3.3 指定 msprof op task duration，附件脚本/README 又将 aclrtEvent avg_us 作为判据，两者有口径冲突。因此计划同时保留：msprof 每例 5 次预热、至少 10 个有效样本的逐 kernel 原始明细及平均值；GTest 在预置设备缓冲上 5 次预热后连续执行至少 10 次的 aclrtEvent 平均耗时。前者按正文判据报告，后者通过随附脚本报告，不混用两者或使用 GTest 整数毫秒耗时替代。若 split-K 使用两次 kernel，每次算子调用必须包含两阶段时间并关联到同一用例，不能只统计其中较快的一段。

计时窗口不包含输入生成、H2D/D2H、golden 与报告生成；计时期间暂停后台结果传输并等待在途传输结束。性能采样反复调用原地 y 时需明确 beta 参数，本任务官方性能集 beta=0，避免因输出累积改变负载含义。

### 复现与交付计划

提交实现到个人私有 Fork 的指定分支，以完整 commit 交付；设计 PR 独立评审，不把设计 CI 成功当成设备自验通过。提供从完整 commit 获取代码、环境部署、生成输入、构建、全部自验、解析原始输出和生成官方表格的顺序命令。脚本随 commit 或验收包提供，原始任务包、GPU 基线、修正 CSV/配置、实际结果与必要输入输出建立对应关系。

`task_submission/` 按官方模板同步保存自验证步骤、精度/性能/内存报告和日志；内存条目依任务书“不涉及”填写原因，不沿用模板示例结论。另提供复现附件说明。提交验收前联网刷新任务书，独立检出验证完整复现步骤，清理模板旧数据与旧图片并核对每个测量值来源；最终 ZIP 使用 STORE/DEFLATE，主包小于 100,000,000 字节，实际执行 CRC、完整解压和文件一致性检查。

### 风险与处置

主要风险包括极端矩形并行度不足、长累加误差、非单位步长低带宽、tiny case Host/发射开销及任务文档采样口径冲突。通过分规模 tiling、分层归约、真实边界用例和双口径性能报告逐项验证。若最终存在未过项，保留真实失败结果、未测范围与原因，不用换用计时口径或删减用例宣称通过。设计中 tile 数值是实现起点，实测后的实现调整记录在代码和自测材料中。
