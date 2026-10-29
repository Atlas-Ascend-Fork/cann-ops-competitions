# 需求背景（required）

## 需求来源

任务：[10月社区任务-aclblasChpr算子开发（950）](https://www.hiascend.com/activities/task-center/details/f783b3800ad744ef8af7f50ba237de04?menu=guide)，发布于2026/10/28，截止2026/11/15。依据任务书 `aclblasChpr_Atlas950PR_task_doc.md` 及原始 test_cases。开发账号/团队为 wblxr408。

本文依照 cann-competitions 的 `04_tasks/01_community-task-2026/resources/design_template.md`（基线 b11e879347ec583b6f0a309aeade8050c5d541ee）组织全部适用章节。代码基于 ops-blas 上游 e56c4cd6f41ea63712f1c5c123a596004ce2e574，工作分支 codex/aclblas-chpr-950pr。本次设计提交至既有正式目录 `04_tasks/01_community-task-2026/tasklist/10-28-aclblasChpr-950/wblxr408/docs/design.md`；该目录已在上游 a18791828054cc11674e1f64943e9e1beeb4a4d5 中确认。对应代码提交为 [62fc37e3](https://gitcode.com/wblxr408/ops-blas/commit/62fc37e3fe1350f46ac10249ee730a96ff2b8553)，设备验证按当前计划后补。

## 背景介绍

### aclblasChpr算子实现背景

Chpr 对列主序压缩存储的单精度复数 Hermitian 矩阵执行实数系数的秩一更新，省去完整 `n*n` 矩阵及未存储三角的读写。共享头文件已有 aclblasChpr 声明，原有 arch22 实现继续保留；本次补充 Ascend 950PR 的 arch35 Host 与 Ascend C SIMT kernel。

### aclblasChpr标杆现状分析

功能参考 cuBLAS cublasChpr；精度 golden 使用 Netlib BLAS3.10.0 的 cblas_chpr(CblasColMajor, ...)。任务附件给出 H100 的200行 gpu_baseline.csv。对标使用原始 uplo/n/incx、实数 alpha=1，GPU ms 转换为 us 后与 NPU kernel 时间比较。当前未取得任何 950PR 实测结果，不能声明性能达标。

参考：[Netlib Chpr源码](https://www.netlib.org/blas/chpr.f)、[ops-blas仓库](https://gitcode.com/cann/ops-blas)、[社区任务仓库](https://gitcode.com/cann/cann-competitions)。Netlib 的零 x[j] 分支与乘法次序用于特殊值语义的设计。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 携带执行stream的Host句柄 | aclblasHandle_t | 不涉及数值类型 | n>0时为有效句柄 | 标量句柄 |
| uplo | 选择列主序压缩存储的三角 | aclblasFillMode_t | Upper、Lower | n>0时枚举值有效 | 标量 |
| n | Hermitian矩阵阶数 | int | int32 | n≥0 | 标量 |
| alpha | Host实数更新系数 | const float* | float32 | n>0时非空 | 单元素 |
| x | Device复数输入向量 | const aclblasComplex* | complex64 | n>0时非空，存储长度覆盖步长 | 物理长度≥1+(n-1)*abs(incx) |
| incx | 向量逻辑元素的物理步长 | int | int32 | n>0时不为0或INT_MIN；支持负步长 | 标量 |
| ap | Device原地更新的压缩矩阵 | aclblasComplex* | complex64 | n>0时非空，与x不重叠 | n*(n+1)/2 |

计算公式：`A := A + alpha * x * xᴴ`，仅更新uplo指定的压缩三角。

### aclblasChpr功能分析

接口为 `aclblasStatus_t aclblasChpr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx, aclblasComplex* ap)`。alpha 为 Host 实数标量，x/AP 为 Device complex64；AP 原地更新，只写存储三角，支持正负非零步长。正常更新将对角虚部写零；合法 alpha=0 的 no-op 保留 AP 全部原值。

输入：handle、uplo、n、alpha、x、incx，以及ap中原有的压缩矩阵。

输出：原地更新后的ap。

支持数据类型：alpha为float32；x和ap为complex64。

支持广播：不支持。

本任务使用 ops-blas 原生公共 API，不涉及 ACLNN 接口、TBE 实现、图算子注册、广播或独立 workspace 查询接口。不能把模板 Addcdiv 的接口和广播说明套入本算子。

# 需求分析（required）

## 需求描述

| 需求 | 实现约束与输出 |
|---|---|
| 精度与类型 | alpha FP32，x/AP COMPLEX64，实虚部分量分别验收 |
| 存储与步长 | Upper/Lower 列主序 packed；incx 可负，排除0和INT_MIN |
| 参数边界 | n<0拒绝，n=0直接成功；正n完整校验后才alpha=0成功返回 |
| 对角 | 非零alpha忽略旧对角虚部并写0；alpha=0保持原值 |
| 异步与资源 | 使用handle stream，API不分配workspace、不内部同步 |
| 性能 | 对每个PF用例要求GPU/NPU≥0.4，时间来自msprof kernel记录 |
| 交付格式 | 原生目录、完整版权头、RAII README示例、官方设计章节及报告附件 |

## 需求拆解

1. 新增 `blas/hpr/arch35/chpr_host.cpp`、`chpr_kernel.cpp`、`chpr_tiling_data.h`；共享声明和arch22源码保留。
2. 在 `test/hpr/chpr/arch35/` 完成参数、Netlib golden、Device wrapper、CSV GTest、边界补充、精度/采样驱动；复用父级 CMake 自动选择arch35。
3. 保留原始1200行CSV和200行GPU基线，补齐均匀/正态输入与随机alpha对比。提供可独立编译的RAII调用示例并嵌入算子README。
4. 完成950PR构建、精度、性能、示例及真实报告后进行正式验收。设备验证尚未执行，相关结果将在验证后补充。

# 详细设计（required）

## 算子分析

### 数学公式

```text
A(i,j) := A(i,j) + alpha * x(i) * conj(x(j))
Upper: 0 <= i <= j < n, packed index = i + j*(j+1)/2
Lower: 0 <= j <= i < n, packed index = i + (2*n-j-1)*j/2
incx > 0: offset(k) = k*incx
incx < 0: offset(k) = (n-1-k)*(-incx)
```

以`temp=alpha*conj(xj)`计算复数乘积。若xj为复零，按Netlib跳过该列对应更新，仍在非零alpha调用中清零对角虚部。对角实部按`real(xj*temp)`加到原AP实部，旧对角虚部从不读取；非对角读写两个分量。alpha=0在Host直接返回，不运行上述对角清零。

### 支持数据类型

| 参数 | 类型/位置 |
|---|---|
| alpha | float32 / Host |
| x、ap | aclblasComplex（real、imag各float32）/ Device |
| n、incx | int / Host值；内部索引和步长扩展为64位 |
| uplo | aclblasFillMode_t / Host值 |
| handle | aclblasHandle_t / Host句柄，携带stream |

### 支持形状

x的物理长度≥`1+(n-1)*abs(incx)`，AP长度=`n*(n+1)/2`。测试精度尺寸覆盖至2048、性能至4096；这些是附件用例范围，非API硬编码上限。一般n受实际分配容量和设备寻址范围约束，不支持动态广播、批量或转置参数。

## 算子实现

### 实现方案

packed数组均匀分段，warp协作访问连续AP元素。无需展开Hermitian矩阵、全局归约、原子操作或中间workspace。每个AP元素恰有一个线程写入；x只读。incx=1的中等尺寸额外使用UB共享缓存，其他输入走GM通用路径。

#### 3.2.1 host侧设计：

tiling策略：将`n*(n+1)/2`个压缩矩阵元素视为一维连续空间，按packed元素数分核、按连续段分块，并根据uplo及x缓存条件选择编译期分支。分核、数据分块和分支规划如下。

##### 1. 分核策略：

每block 256线程，warp宽32，每warp一段256个packed复数；block覆盖8段、2048个元素。block数为min(ceil(packed元素数/2048), AIV核数)，再按grid-stride遍历剩余段。Upper/Lower列长不同，但分配单位为packed元素数，减少按列分核的负载差异。零规模和零alpha均不发射kernel。

##### 2. 数据分块和内存优化策略：

AP以连续packed段访问，实虚部直接按交错float地址读写，避免临时矩阵。incx=1且128≤n≤4096时每block协作将`2*n`个float装入UB，容量最多32KiB，屏障后只读复用。其他步长和尺寸直接读GM。4096是缓存容量阈值；不拒绝更大n。AP无需UB缓存或搬运队列，写入后不再在本kernel读取其他线程的AP，因而不需要跨block同步。

32位Host n扩展后计算packed总长，负incx先扩展为int64再取负，避免INT_MIN取绝对值溢出。内部复数到float偏移使用64位运算。Host未额外限制512MiB；该预算仅属于测试程序。

##### 3. tilingkey规划策略：

本实现不使用框架TilingKey注册。ChprTilingData按值传递n、elements、incx、uplo、cacheX和alpha。四条编译期模板路径为Upper/Lower×GM/UB，在kernel入口派发。256线程、256元素段、128缓存下界是待实测调节的初始值，尚无最优性结论。

数据检测：Host参数处理为n<0拒绝→n=0成功→正n检查handle→uplo→incx→alpha/x/AP非空→读取alpha并处理零值→查询核数及发射。任务书明确n及有效handle下的其他参数顺序，对空handle叠加其他错误未逐项举例；本实现将正n的空handle作为前置条件，返回HANDLE_IS_NULLPTR，并在补充测试和README中记录这一约定。

#### 3.2.2 kernel侧设计：

每段首通过整数二分求packed列号，无浮点sqrt舍入风险；段内随索引推进递推列边界。线程从begin+lane开始以32跨步覆盖该段，段尾用总元素数截断。Upper行号=index-base；Lower行号=col+index-base。

UB分支先协作加载x，再执行asc_syncthreads；分支选择对整个block一致，不存在部分线程跳过屏障。GM分支按正负incx计算物理位置。计算`temp.real=alpha*xj.real`、`temp.imag=-(alpha*xj.imag)`，更新AP实部及非对角虚部；对角单独写零。独立元素写入消除归约误差与写冲突。

不宣称采用双缓冲、DMA流水、Cube/Tensor Core或多kernel融合；当前优化是packed均衡分段、连续访问、编译期派发和x的UB复用。是否调整缓存阈值/线程数，须以实际950PR延迟、访存及资源占用为依据。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

本次新增arch35实现，950PR上的构建、精度和性能验证待执行。950DT未纳入支持声明；Atlas A2/A3的既有arch22实现由原工程保留。

## 算子约束限制

alpha为Host指针，x/AP为有效且不重叠的Device存储；负步长传分配首地址，不能重复偏移。正n不接受nullptr、非法uplo、0或INT_MIN步长。调用者负责stream同步及存储生命周期。API返回SUCCESS仅表明Host校验/提交完成，异步错误由调用者同步获得。n=0不读取指针；alpha=0仅在全部参数合法时保持AP，含脏对角虚部。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS3.10.0 cblas_chpr，CblasColMajor；实际链接库及版本需保留 | 任务书第3节 |
| 有限精度 | 实部/虚部分别rtol=atol=2^-13，matched_ratio≥0.99；逐元素误差≤max(1e-2,32*ULP(golden)) | 任务书精度表及特殊值说明 |
| 非有限精度 | 同NaN、同号Inf、NaN/±Inf等效；有限/非有限、异号Inf失败；单独统计 | 任务书Inf/NaN规则 |
| 边界 | 原始1000条精度/边界+5项独立GTest；两种uplo、正负步长、INT_MIN、nullptr、no-op、脏对角 | 原始CSV与任务书第2、3.5节 |
| 输入分布 | 原始seed奇偶切换均匀/正态；实虚部分别采样；补充alpha两种分布各12次 | 任务书第3.5节 |
| 性能 | 200个PF，NPU均值≤2.5*GPU；GPU原始ms转us，记录逐采样值 | 任务书及gpu_baseline.csv |
| 采样协议 | 默认工具warmup5+有效51条，另报前10均值；10/>10/>50的正式口径待确认 | 任务书第3.3、7节及附件README |
| 采样真实性 | 每次恢复AP，application replay，公共API成功及最终结果核对；只取msprof kernel Task Duration(us) | 原地更新语义及性能标准 |
| 内存 | 正式内存自验证不涉及；测试额外检查x只读、AP前后哨兵，不能替代完整内存检测 | 任务书内存项 |
| 资源预算 | 单用例Host预算512MiB，精度2048、PF4096用例不扩大API限制 | 任务书/测试工程约束 |

所有测试当前未执行。测试驱动对完整计划集合、XML通过状态、返回码、统计字段、profiling记录及标杆匹配进行核对，缺失或失败均不计通过。驱动指定BasicInfo采集；依据[Ascend官方msOpProf指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)（2026-10-28读取），application会重放应用且不主动清除L2，后续须记录工具版本及缓存口径并核对GPU比较条件。实际msprof版本及application replay暖机/导出行为须在950环境确认。

## 兼容性分析

保留共享API的既有声明，不重复接口或改动ABI。生产源码由仓库架构扫描自动收集；测试父级CMake按SOC选择arch35，原arch22头文件及用例不替换。示例放在test目录的独立example工程，避免被生产库递归编入main。

README按仓库指南给出单行无分号原型、Host/Device参数说明、非空约束及完整AclContext+unique_ptr示例；设计按官方模板保留全部必需章节。自验证报告采用官方模板文件名，并使用腾讯文档的shape/dtype/精度/方法/截图/GPU/AscendC/说明列；实际链接为其他算子填写示例，不能复用其结果。未测试的Excel、日志、截图、commit链接和评审结论均待真实执行后补齐。
