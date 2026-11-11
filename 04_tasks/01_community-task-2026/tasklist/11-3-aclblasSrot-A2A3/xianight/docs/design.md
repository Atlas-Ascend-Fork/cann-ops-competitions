# aclblasSrot（Atlas A2/A3）设计文档

## 需求背景

### 需求来源

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/a42dd9c650154f4f9acdfb3d2bea468b)要求补齐 `cann/ops-blas` master 的 FP32 实数 Givens 旋转。任务目录依据维护者[任务清单 PR 2846](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2846)的 `11-3-aclblasSrot-A2A3`。任务书来自网页 ZIP，未公布可核对的 Git commit，故不将清单提交当作任务书版本。

设计依据为任务书 §§2–4、[Netlib srot.f](https://netlib.org/blas/srot.f)、[cuBLAS §2.5.8](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)，及开发仓基线 `5bd95145a9c499e43ebad44a313a1029c63d54bc` 的 `blas/rot/arch22/csrot_*`、`arch35/srot_*`、`blas/rot/README.md`。接口复用 `include/cann_ops_blas.h` 的公共声明，采用 Ascend C kernel 直调，不新增平行 API。

### 场景与目标

Givens 旋转用于 QR 分解等线性代数流程。目标是为 Atlas A2/A3 提供相同向量长度的 FP32 原地双输出旋转，在 910B3 / CANN 9.1.0 上满足规定精度与 200 条性能基线。

## 需求分析

```cpp
aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n,
    float* x, int incx, float* y, int incy,
    const float* c, const float* s);
```

x/y 位于 Device，物理长度分别为 `1+(n-1)*abs(incx)` 和 `1+(n-1)*abs(incy)`，ND 一维 FP32。c/s 为 FP32 指针，各自独立允许 Host 或 Device。n、步长为 int32，handle 携带运行 stream。x/y 输出覆盖各自输入，未访问的间隔元素保持不变；不提供广播或返回视图。

### 返回码与校验优先级

1. 空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 直接成功，不读取数据和标量，不启动 kernel。
3. n>0 时 x/y/c/s 任一空指针、步长为 0 或 INT_MIN 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 使用 int64 计算 `(n-1)*abs(stride)`，任一大于 INT32_MAX 返回 `ACLBLAS_STATUS_INVALID_VALUE`。在分配、指针属性查询和 kernel 下发前完成上述校验。
5. 指针属性查询失败拒绝调用；不对 c/s 数值及向量元素作有限性检查。

### 任务书与附件的歧义处理

任务书 §§2.4–2.5 明确上述 A2/A3 返回码；附带测试 README/生成器/CSV 却按 arch35/Netlib 契约把负 n 和零步长视为成功。采用正文及仓内已声明 A2/A3 口径，保留原 CSV，生成逐例差异清单与 arch22 适配 CSV，不静默删去冲突用例。cuBLAS §2.5.8 不逐一声明这些非法参数返回码，故不声称其独立证明任务正文全部校验。

正文 §3.3 要求 msprof op kernel task duration，附件脚本则使用 event 时间；验收以正文为准，event 时间仅作诊断。负步长是有效输入，不能照抄正文“inc≤0 quick return 不分配”一句对其使用 dummy 缓冲。仅 n=0 和预期报错用例使用 dummy。

### 任务书 §2.5 约束核对

| 约束 | 设计处理 |
| --- | --- |
| 参数合法性 | 按上节顺序在 Host 校验，无效参数不触发计算 |
| 非连续输入 | 非零正负步长；负步长从完整物理 span 尾端起算 |
| 广播 | 不涉及，x/y 逻辑长度均为 n |
| 动态 shape | n/stride 运行时 tiling，无图模式动态 shape 要求 |
| 原地与视图 | 同一对元素的旧 x/y 先读后写，间隔位置不变 |
| 确定性 | 无归约、原子操作和跨核数据依赖，同输入执行顺序固定 |
| 空 Tensor | n=0 成功且不访问指针 |
| 0 维 Tensor | 不涉及；c/s 使用标量指针 |
| 异步执行 | 仅在 handle stream 下发；函数不主动同步，调用方同步后读输出 |

## 详细设计

### 算子分析

令 `ix(i)=incx>0 ? i*incx : (n-1-i)*abs(incx)`，y 同理，i∈[0,n)。

```text
oldX = x[ix(i)]; oldY = y[iy(i)]
newX = c*oldX + s*oldY
newY = c*oldY - s*oldX
```

每逻辑元素四次乘法、一次加法、一次减法，O(n) 时间；连续路径读写共 16n 字节，主要受带宽及小规模启动开销限制。计算全部使用 FP32，不降低精度。不以 c=1/s=0 等数值跳过计算，避免改变 Inf/NaN 的 IEEE 传播。不使用近似数学或 fast-math；需要时禁止融合乘加以匹配 Netlib float32 中间舍入。

### Host / Tiling 设计

在 `blas/rot/arch22/` 新增 srot host、kernel 和轻量 tiling 头。复用公共 handle/平台信息。c/s 分别查询 `aclrtPointerGetAttributes`：Host 指针读取标量随 tiling 按值传入，Device 指针直接传 kernel，执行时读取，不做 D2H 或同步。

Tiling 携带 n、incx/incy、Host c/s 值、Device 来源标志、每核范围/块长。无 GM 工作区，无逐调用 malloc/free。kernel 参数按值传递，确保调用后 Host 栈生命周期不影响异步执行。

核数取平台 AIV 数与有效工作块数的较小值，最少 1 核。连续路径按 32B（8 个 float）或更大块粒度划分互不重叠的区间，小规模降低核数；仅最后区间含不足一个块的尾部。块长初值 4096 个元素，按 UB 预算及实测调优。非连续路径按逻辑元素段划分，所有索引先在 Host 经过范围校验；设备端地址计算保留足够位宽。

### Kernel 设计

连续路径使用 AIV SIMD，执行 CopyIn → Compute → CopyOut：

- 将旧 x/y 通过精确字节长度 DataCopyPad 搬入 UB；不越界读写尾部。
- 保留旧值，分别算 c*x、s*y、c*y、s*x，再 Add/Sub。双输出落入独立 UB 区，不提前覆盖仍需参与第二条公式的输入。
- TQue/TBuf 管理 MTE2/V/MTE3 依赖；双缓冲隐藏搬运延迟，各阶段复用前确保上次 DMA 完成。
- 每核拥有独立输出区，连续区间按块对齐；不使用跨核归约或原子更新。

非连续路径优先复用分块搬运与 SIMD 算术。允许按正物理顺序以多段 DataCopyPad 搬入每个逻辑元素、UB 中 Gather 还原逻辑顺序，再计算并按相应方向搬回；每段只写 4 字节，不覆盖 stride 空洞。负步长块的起始物理下标为 `(n-1-(logicalStart+count-1))*abs(stride)`。块数、blockLen、stride 字段遵守 DMA 限制，过大的 stride 或极小规模使用逐元素标量访问兜底。兜底同样分核且先读旧值，不借用错误的连续地址。

非连续路径初版固定单核执行，消除混合正负步长和非 8 倍数 n 下 x/y 无法同时按同一逻辑边界隔离 32B 物理块的跨核写风险。只有建立两个向量联合物理块隔离证明并通过哨兵测试后才启用多核优化；性能验收 200 条均为连续路径，不依赖此优化。单核内部相邻块也通过流水同步确保写回完成后再复用相邻物理范围。

### 硬件与资源约束

目标 Atlas A2/A3（arch22）；规定自验为 910B3 / CANN 9.1.0，不把 A2 结果声称为 A3 实测。使用 AIV Vector/MTE，不依赖 arch35 SIMT。UB 预算以平台查询为准，按 A2 192KiB 上限保守规划：连续路径双缓冲 x/y 与输出共 8 个 4096-float 缓冲约 128KiB，加两个临时块约 32KiB，总计不超过 160KiB；非连续 packed padding/index 开销独立计算并缩小 tile，不能复用连续路径容量假设。32B UB 对齐，Vector 64-float repeat，尾部采用 count mask/精确拷贝。实现须编译核对具体 API 和限制。

### 已知限制与规避

x/y 必须是有效 Device 存储，长度满足完整物理 span；未声明跨向量重叠内存的顺序语义，调用方不得让两个输出区域互相覆盖。Device c/s 在执行完成前有效且不与可写输出重叠。对不同 arch 的零步长/负 n 行为保留差异，不改 arch35 既有语义。

## 可维可测分析

### 精度与功能

保留官方 1000 条精度 + 200 条性能 CSV 和原始基线；对冲突返回码作可追溯 arch22 适配。采用 Netlib BLAS 3.10.0 的 `cblas_srot` FP32 参考，保存准确版本及构建选项，避免用待测 kernel 自身生成 golden。补充约 50/50 均匀[-5,5] / 正态（μ∈[-5,5]、σ∈[0.1,2]）输入，并记录种子、μ/σ及数量。补充 stride 7、c/s=-0.5、空 handle/标量指针、n=0 叠加非法指针、INT_MIN、溢出、块边界、两种 mixed 标量方向、重复确定性与间隔哨兵测试。预期报错用例不按巨大 span 分配。

x/y 分别按 `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)` 统计有限值，matched ratio≥0.99 且最大绝对误差≤max(1e-2, 32×ULP(g_max))，g_max 为最大绝对误差点的 FP32 golden，按[指定精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)计算。非有限元素按任务规则独立比较：同 NaN、同号 Inf、NaN/Inf 等效，有限/非有限不等效，异号 Inf 不等效；非有限项不进入有限统计。没有有限元素时显式记录 N/A，不能让空统计掩盖非有限失败。

在 `test/rot/srot/arch22/` 提供 CSV 驱动 GTest，确认 CMake 注册路径实际覆盖 arch22。每个有效精度用例的 FP32 golden 落盘，并保存输入生成参数与本次输出索引；实际压缩评估附件大小，不通过删去必交数据规避 100M 上限。保存逐向量误差、元素数、特殊值计数、预期/实际状态和进程退出码，防止仅解析到少数 PASS 即认为全量通过。

### 性能与内存

200 条性能用例不改变参数和 GPU 基线，以 `(n,incx,incy,c,s)` 核对关联，保留重复参数各用例身份。使用 msprof op 5 次 warmup 后至少 10 个 SUCCESS 的 kernel task duration，逐例保留明细、算术平均、GPU 毫秒转微秒后的数值与比率；每例平均≤GPU×1.25。记录设备型号/软件、设备负载和计时范围，不混入 H2D、golden、报告生成和回传。持续旋转的性能循环与一次调用精度分开；异常样本明确报告。

正文 §3.4 无内存指标；模板中内存表/日志写“不涉及”，不填虚构通过。必要的越界检查用于实现质量验证，与任务内存指标分开。

### 复现与交付

源码、输入生成/适配、数据下载或固定附件、全量测试、msprof 解析和模板报告生成脚本归入验收 commit 或附件。个人仓根 `task_submission/` 提供规定七项命名文件，清空模板示例与旧日志。提供《复现附件说明》关联输入配置、实际输出、日志和用例，原始输出由本次执行生成。独立检出按完整步骤验证部署、自验、解析和材料生成；最终 ZIP 使用标准 DEFLATE/STORE，校验大小、CRC、所有内容解压和路径。设计评审与代码验证结果分别记录，不用自验结果替代人工合入。
