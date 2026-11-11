# aclblasChbmv 算子设计文档（Atlas A2/A3）

团队及开发者：Qianqiuer。设计日期：2026-11-11。

# 需求背景（required）

## 需求来源

依据11月社区任务随包 `aclblasChbmv_AtlasA2_A3_task_doc.md`、`test_cases/chbmv_test.csv`、`gpu_baseline.csv`及测试指导。设计文档提交竞赛仓，代码在通过设计评审、完成开发与验收后贡献至 [ops-blas](https://gitcode.com/cann/ops-blas)。本文为拟实施方案，不表示已完成设备验证。

## 背景介绍

Chbmv计算单精度复数Hermitian带状矩阵向量乘 `y=alpha*A*x+beta*y`，用于只需保存主对角附近元素的复数线性系统计算。A满足 `A(i,j)=conj(A(j,i))`，对角线仅实部有效。按带存储可将矩阵存储从O(n²)降低到O(n·(k+1))，计算仅遍历实际带内元素。

本任务是新增句柄式BLAS接口，不涉及历史TBE迁移。核对ops-blas基线 `5bd95145a9c499e43ebad44a313a1029c63d54bc`：公共头已有Ssbmv，未声明Chbmv；`blas/sbmv/arch22`已有实数Ssbmv的host、kernel与tiling文件。沿用同族工程结构，但不能直接将实数乘法改名：镜像侧需要共轭，对角虚部必须忽略。

# 需求分析（required）

## 需求描述

使用CANN 9.1.0及Ascend C kernel直调模式，支持Atlas A2/A3，性能验证设备为Atlas 800T A2（910B3）。通过handle绑定stream，不新建私有平行API，不依赖950的SIMT或HF32实现。

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

`aclblasComplex`使用仓库公共定义，不重复定义布局。alpha、beta位于Host；A、x、y位于Device；y原地更新。无batch、broadcast或trans参数，无超出lda/inc语义的视图支持。

## 需求拆解

| 项目 | 设计约束 |
| --- | --- |
| 数据类型 | A/x/y/alpha/beta均为complex64 |
| 形状 | A物理大小lda×n，x/y逻辑长度n；运行时n/k，不按测试编号分派 |
| 带宽 | k≥0；有效计算带宽min(k,n−1)，但存储对角位置及lda校验仍使用原始k |
| leading dimension | int64域检查lda≥k+1 |
| 向量跨度 | inc非零且非INT32_MIN；int64计算(n−1)×abs(inc)，不得超过INT32_MAX |
| 空输入 | handle判空最优先；n<0拒绝；n=0直接成功，不校验剩余参数且不启动kernel |
| 标量分支 | beta=0禁止读y旧值；alpha=0可直接执行缩放，但n>0仍校验A/x等指针 |
| 错误 | 空handle返回HANDLE_IS_NULLPTR；其他规定非法参数包括非法uplo均返回INVALID_VALUE |
| 硬件 | A2/A3支持；只需一种款型自验，计划910B3完成精度和性能 |

# 详细设计（required）

## 算子分析

### 数学与零基地址公式

令物理带数组为B，行列下标从0开始。每个输出行i仅访问 `j∈[max(0,i−keff),min(n−1,i+keff)]`，其中keff=min(k,n−1)。

| 模式 | 直接存储侧 | 镜像侧 |
| --- | --- | --- |
| UPPER | i≤j：B[k+i−j+j*lda] | i>j：conj(B[k+j−i+i*lda]) |
| LOWER | i≥j：B[i−j+j*lda] | i<j：conj(B[j−i+i*lda]) |

对角元素读取后只使用实部，以实数乘复数x的方式计算，避免把虚部0与Inf相乘而引入额外NaN。UPPER左上、LOWER右下无效角区以及lda padding不参与计算。

负步长起点为 `start=(1−n)*inc`，逻辑元素t的物理位置为 `start+t*inc`；正步长起点为0。地址与字节数计算使用64位并检查溢出，不能对INT32_MIN先做32位abs。

复数非对角乘法按 `(ar*xr−ai*xi, ar*xi+ai*xr)` 计算。镜像侧先共轭。累加后进行alpha缩放，beta非零才加载并加上beta*y。不提前读入y以便事后丢弃。

## 算子实现

### Host侧设计

严格按handle→n合法性→n=0→其余参数的顺序校验。Host只读取Host标量，不读取Device矩阵做数值扫描，不引入额外全流同步。使用现有handle获取stream，按仓库错误码规范处理stream及launch错误，结果读取由调用方同步。

tiling保存n、原始k、有效带宽、lda、带符号步长、向量起点、复数标量、行tile和实际核数。利用平台信息查询可用核数与UB容量；UB预算覆盖A/x的实虚拆分、临时乘积、累加器及双缓冲空间，不硬编码其他架构容量。

初始分核采用输出行分块，每行由唯一核写入，不采用跨核浮点原子加。边界行的带内长度更短，按有效工作量或循环分配行tile改善负载均衡，小规模减少核数避免启动/搬运固定成本。

分派仅依赖n/k、uplo、inc、对齐及标量类别：n=0无kernel；alpha=0缩放路径；k=0对角路径；窄带小矩阵缓存路径；其余通用带状归约路径。初版优先确保通用路径正确，优化分支逐步通过独立验证后启用。

### Kernel侧设计

采用A2/A3支持的Ascend C向量流水。按有效带段读取A与x，将交错复数拆分为实/虚向量，分段乘加、归约并写回y。行拥有者方案可能重复读取镜像元素，但避免跨核同步、额外输出workspace和原子归约，是正确性与性能基线。

对每一行把直接侧、镜像侧及对角分开处理：连续的存储带段批量DMA，非连续段采用受限分块搬运或索引加载；不得为改善DMA吞吐无条件读取越界角区。连续inc=1优先复用相邻行共享x窗口，通用步长路径在有效位置Gather/Scatter；负步长仍按逻辑索引读取，不改调用者指针。

UB不能容纳整个带时按带长分块，分块结果在本核内归约。double buffer只在明确生产、消费及复用依赖后启用。尾块使用合法长度、掩码和本地补零；写GM仅覆盖y有效元素，保留stride空洞。DMA对齐单位按CANN9.1.0实际重载确认，不把元素数、字节数和32B块数混用。

### 精度与性能优化

基线采用FP32复数乘累加和分层归约；对大带宽、抵消明显的输入评估补偿累加，以官方fp64 golden验证，不通过调大阈值掩盖误差。不能默认将数据转FP16或迁移950HF32策略到A2/A3。

重点优化方向为窄带启动成本、x窗口复用、带区访问合并、尾块开销及多核负载。禁止将全矩阵解压为n×n稠密矩阵作为默认方案。若宽带路径需要增加kernel，正式性能必须核算一次API的全部计算任务，不能只挑其中最快的kernel。

### 工程组织

实现放 `blas/sbmv/arch22/`，计划最小集合为chbmv_host.cpp、chbmv_kernel.cpp、必要的launch头与tiling头；共享定义按实际使用者决定，不新建无关utils层。公共声明新增到include/cann_ops_blas.h，与其他架构共用，若开发期间上游已增加声明则复用。

测试放 `test/sbmv/chbmv/arch22/`，复用仓库CSV/GTest设施；公开接口说明并入blas/sbmv/README.md。构建注册采用现有sbmv选择机制，同时核查单算子构建与完整构建，不为本算子修改全仓编译策略。

## 支持硬件及限制

| 产品 | 支持设计 |
| --- | --- |
| Atlas A2（含800T A2/910B3） | 支持；计划主验证平台 |
| Atlas A3（含800T A3） | 支持；使用arch22兼容实现 |
| 其他产品 | 不在本任务适配范围 |

未要求确定性计算；不以bit-exact替代浮点标准。输入输出跨数组别名未由任务书明确保证，本设计不扩展为任意A/x/y重叠安全。内存专项验收不涉及，但越界、padding和指针合法性仍需测试。

# 可维可测分析

## 精度标准与覆盖

保留发布的1200条CSV原件与200条GPU基线，不重新生成覆盖它们。完整执行1000条非性能用例和200条性能用例；性能用例同时检查输出，负向用例不按非法尺寸分配巨量缓冲，采用最小dummy占位验证提前拒绝。

golden按Netlib BLAS3.10.0列主序Hermitian语义实现：complex64输入提升为复数float64，以cblas_zhbmv计算后收窄float32。任务书称cblas_chbmv的fp64真值口径，随包README已说明用zhbmv承担双精度真值，不能用单精度结果冒充fp64。

实虚分别采用rtol=atol=2^-13、matched_ratio≥0.99及最大绝对误差硬限（1e−2或最大误差点golden的32ULP，按官方标准实现）。非有限值单独计数：NaN/NaN、同号Inf、NaN/Inf按任务规则等效；有限/非有限不等效；所有非有限比较均需通过。MERE/MARE列不取代此判据。

另附独立补充CSV与接口单测：

- 均匀与正态各50%的配对随机子矩阵，正态μ∈[-5,5]、σ∈[0.1,2]，A/x/y及标量实虚独立采样；使用GAUSS_NORM_5_2并记录实际参数、数量及分类覆盖，官方固定用例和特殊值集合单列报告。
- handle=null叠加n=0；n=0叠加其余非法参数；n<0；非法uplo；k/lda极值，INT32_MIN步长和跨度溢出。
- UPPER/LOWER、k=0/1/n−1/>n−1、非对齐规模、lda padding、incx/incy的±1/±2/±3全组合。
- 对角虚部毒化、无效带角区及padding毒化、beta=0的旧y毒化、stride空洞哨兵；确认无效区域不影响结果且未被覆写。
- Inf/NaN的x不与alpha=0组合；标量特殊值及大幅抵消场景；原件不变、补充测试不替代原件。

## 性能标准与采集

在910B3/CANN9.1.0上使用未经本地修改的配套官方msprof op采集Task Duration(us)，工具5次warmup后连续有效采样至少10次并取算术平均。只计返回SUCCESS的调用，异常样本记录原因及数量，不静默删去慢样本。保存逐case原始CSV、有效样本数和汇总。

逐条满足 `NPU_avg_us ≤ GPU_ms*1000/0.8`，即GPU/NPU≥0.8；200条全部达标，不能用平均加速比代替。标杆来自包内A100-SXM4-80GB / CUDA12.8 / cuBLAS12.8.4数据，关联键和lda等参数一起核对，避免ms/us单位错误。

随包README末段称Event耗时参与判定、msprof仅交叉分析，与任务书§3.3/§7冲突。本设计以任务书的msprof op为正式口径；原Event脚本结果独立保留为辅助，不改官方脚本或伪装采样来源。实现测试工程时明确工具warmup与程序调用的窗口关系，避免双重预热或把预热算入有效样本；多kernel路径须逐调用关联全部kernel并向验收方明确统计方式。

## 可维护性、兼容性与交付

保持既有Ssbmv及其他架构接口不变，使用公共复数类型与枚举；非法uplo按本任务返回INVALID_VALUE，不照搬Csymv的INVALID_ENUM。实现中不增加官方未规定的矩阵阶数限制，资源不足或尺寸计算溢出按仓库错误约定处理。

先小规模手算和边界验证，再全量精度，最后完整200性能；优化候选先覆盖被改路径和失败case，通过后才全量回归。记录源码commit、库哈希、官方材料哈希、设备/工具链及调用命令，不将旧库测试结果继承到新二进制。

私仓根目录task_submission按模板提供：1 自验证步骤说明.md、2.1 精度自验证报告.xlsx、2.2 精度自验证日志.log、3.1 性能自验证报告.xlsx、3.2 性能自验证日志.log、4.1 内存自验证报告.xlsx、4.2 内存自验证日志.log；内存两项明确“不涉及”，不伪造数值。精度实虚分别有结果及真实截图，性能提供原始数据与截图。验收说明填写完整commit链接、分支和算子/测试目录；邀请Ascend-CANN按平台流程处理。正式代码PR只纳入仓库要求的源码、测试、CSV和长期说明，一次性报告不混入源码提交。

## 风险及控制

主要风险为大带宽误差累积、镜像侧访存不连续、窄带启动开销及测试包口径歧义。分别通过补偿/分层归约、分块与x复用、小规模核数选择、严格区分正式与辅助计时应对。所有性能策略以真机结果决定；当前阶段不承诺未经测试的性能数字。
