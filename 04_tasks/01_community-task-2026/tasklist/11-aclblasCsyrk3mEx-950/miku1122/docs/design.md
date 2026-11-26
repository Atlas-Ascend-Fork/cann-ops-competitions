# aclblasCsyrk3mEx（Ascend 950PR）算子设计文档

版本：v0.2，2026-11-25。贡献者：miku1122。用途：社区任务设计方案评审。

本文为拟实施方案，记录接口、数学分解、分核分块及验证设计。代码提交与真机结果另行补充，不申报实现完成、性能达标或验收通过。

# 需求背景（required）

## 需求来源

- 官方任务：https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6
- 官方附件：https://www.hiascend.com/p/resource/202611/a34a01ae360a451bb9abfcbc4421dcb5.zip
- 任务讨论：https://gitcode.com/cann/ops-blas/discussions/88
- 开源工程：https://gitcode.com/cann/ops-blas
- 设计模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md

## 背景介绍与现状

基于 ops-blas 句柄式 BLAS 工程开发复数对称矩阵秩-k 更新扩展接口，支持 COMPLEX64 与 complex-int8 输入。通过 handle 绑定 stream，在 NPU 上执行 Ascend C kernel。新增接口独立实现，不依赖尚未合入的 Ex 接口。

拟复用非 Ex syrk 的参数校验与三角存储处理，以及 arch35 cgemm3m 的三实数乘积组织方式；复用前核对最新主线和接口语义，不能直接把四乘积路径当成 3M。本接口不走 TBE 或 ACLNN。

# 需求分析（required）

## 功能与接口

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 项目 | 设计约束 |
| --- | --- |
| 数学定义 | X=op(A)，C=alpha·X·X^T+beta·C |
| 转置 | N / T / C；其中 C 与 T 等价，均不共轭 |
| 数据类型 | Atype 为 ACL_COMPLEX64 或 ACL_INT8；Ctype 仅 ACL_COMPLEX64 |
| INT8 存储 | 每复数两个交错 signed int8：real、imag，不是单通道 int8 |
| 标量 | alpha/beta 为复数 FLOAT32 分量，支持 Host/Device pointer mode |
| 布局 | 列主序，lda/ldc 支持 padding，仅更新 uplo 指定三角 |
| 前导维度 | N 时 lda≥max(1,n)，否则 lda≥max(1,k)；ldc≥max(1,n) |
| 硬件与环境 | 950PR / arch35；CANN 9.1.0 及以上 |

## 边界与异常行为

校验顺序：handle → uplo/trans → n/k、lda/ldc → 标量、dtype 与结构指针 → quick return。n=0 不访问 A/C、不发 kernel；alpha=0 或 k=0 不读取 A，beta=1 时保持 C 不变，否则只缩放指定三角；beta=0 不读取旧 C，避免旧 C 的 NaN 污染输出。Device 标量不在 Host 直接解引用，退化路径在设备读取标量后判断。

对角虚部是有效结果，保留按公式计算出的值，不置零。非指定三角和 padding 保持原样。非法枚举、负维度、非法前导维、空标量、非支持 dtype 和必要矩阵指针按任务书返回对应状态码。不对 Device 张量逐元素扫描，不采用 Host 主计算，不按 case_name、seed 或 golden 分派。

# 详细设计（required）

## 数学公式

令 X=R+iI，采用标准三乘积分解：

```text
P0 = R·R^T
P1 = I·I^T
P2 = (R+I)·(R+I)^T
Re(X·X^T) = P0-P1
Im(X·X^T) = P2-P0-P1
```

手算检查：X=[1+2i, 3+4i]^T 时，X·X^T=[[-3+4i,-5+10i],[-5+10i,-7+24i]]。

三次逻辑实数矩阵乘在同一次乘积 kernel launch 中完成，不使用三次独立 GEMM 下发拼接。预处理和 epilogue 如需独立 launch，应在 profiling 中完整列出。复数 alpha/beta 在最终重组后按复乘公式应用。

## Host 侧设计

1. 新增公开接口及 csyrk3m_ex_host.cpp，统一参数校验、标量位置和零贡献分派。
2. 根据 n/k、dtype、uplo、trans、可用核数及片上容量选择小规模或三角 tiled 路径。
3. 使用宽整数计算解包平面、乘积临时区、对齐元数据的工作区，检查溢出及 handle 容量限制。
4. 明确逻辑 X 到列主序 A 的映射，向设备传递三角 tile、K-panel 和尾块信息。内部若交换行列，配套翻转 uplo。
5. 保持 stream 异步语义，避免为读取 Device 标量引入全流同步。

## Kernel 侧设计

### 输入准备

AIV 解交错并准备 R、I、R+I 三个 FP32 平面。N/T/C 通过逻辑坐标映射处理，OP_C 不引入共轭负号。INT8 各分量先精确提升；R+I 的范围为 [-256,254]，不能存回 int8。

### 三角乘积与分核

小规模候选为单次 AIV 计算 kernel，每个输出归约内保持三个逻辑实乘积，避免解包和全矩阵临时区开销。中大规模候选为 AIC 三角 tile 调度，仅枚举 q(q+1)/2 个有效 tile；每 tile 内执行三乘积并复用输入，K 分段后重组和归约。对角 tile 逐元素掩码，禁止写到未引用区域。

优先评估 64×64 输出 tile、K-panel 64/128/256，依据实际片上容量、编译资源和 profiler 决定分派阈值。数据搬运与计算之间显式处理 MTE2/Vector/MTE3 依赖；局部缓冲复用前完成相应事件同步。跨核 split-K 不得无保护地并发改写同一 C 元素。

### INT8 与浮点精度

INT8 候选使用 K-panel≤256 的精确 FP32 partial，转 int32 后在片上跨块归约并完成实虚重组，最后转换和应用 alpha/beta。因为 |R+I|≤256，每项乘积绝对值≤65536，分块任意部分和绝对值≤2^24，在不降低输入精度的 FP32 运算下可精确表示。int32 归约保守要求 65536·K≤INT32_MAX；更大合法 K 使用更宽归约，不截断输入。该路线仍需真实工具链验证。

COMPLEX64 路径保留 FP32 累加，重点检查长 K、近抵消和大幅 alpha/beta 的舍入误差。必要时评估分段、补偿求和及 R+I 输入变换残差；记录补偿成本，不以放宽阈值、替换 golden 或四乘积回退掩盖失败。

### Epilogue 与工作区

在有效三角上重组实虚部，再应用复数 alpha/beta，beta=0 分支不加载旧 C。优先 tile-owner 局部归约，避免为全部 split-K 分块写出完整 n×n 三乘积矩阵。

三个 FP32 输入平面约占 12·n·k 字节；若保存三个完整乘积，每分区另需约 12·nPad² 字节。所有对齐、分区和元数据均计入工作区，不以设备显存容量代替库工作区限制。

## 工程集成与兼容性

- 公开声明：include/cann_ops_blas.h。
- 实现：blas/syrk/arch35/csyrk3m_ex_host.cpp、csyrk3m_ex_kernel.cpp 及必要 tiling 头。
- 测试：test/syrk/csyrk3m_ex/arch35/，复用仓内 GTest/CSV 框架。
- 构建：bash build.sh --ops=csyrk3m_ex --soc=ascend950；单算子构建显式保留跨算子复用依赖。
- 新增独立接口，不改变现有非 Ex API 签名；文档注明 950PR 和 dtype 支持。

# 可维可测分析

## 精度与边界测试

官方附件静态统计：1000 精度、200 性能；精度中 COMPLEX64 505、INT8 493、FLOAT 负向 2。此处为用例数量，不是测试通过数。

使用任务要求的 Netlib BLAS 3.10.0 非 3M cblas_csyrk 参考，INT8 分量精确提升后调用；如使用 cblas_zsyrk，明确精度域并收窄到 complex64。实部与虚部分开判定：rtol=atol=2^-13，匹配率≥0.99，执行任务书要求的最大误差限制（1e-2 或逐元素 32 ULP）。不取整矩阵最大元素的 ULP 放宽小值误差，参考库差异单独记录。

覆盖 N/T/C、上下三角、31/32/33 与 63/64/65 尾块、lda/ldc padding、INT8 极值、长 K、复数标量、NaN/Inf、零贡献和非法参数；未引用三角及 padding 使用哨兵检查。分别记录精度、性能、失败和跳过数，确认二进制生成且用例真实执行，不把空测试或全 skip 当通过。

## 性能标准与采集

正式要求为每 case GPU/NPU kernel 耗时倍率≥0.4，200 项倍率的算术平均≥0.5，二者同时满足。使用 msprof op 的 Task Duration，warmup 5 后至少 10 次有效采样；必要的预处理、乘积和 epilogue kernel 全部留证。事件计时只用于快速趋势，不替代正式 profiler。INT8 性能另行补充，重复 baseline key 按原始行身份分别记录。

## 风险与实施顺序

主要风险为 FP32 3M 抵消误差、INT8 归约资源、小尺寸 launch 开销、三角转置映射和 profiler 口径。实施顺序为最小真机构建与语义小集、官方全量精度、代表尺寸 profiling、逐项性能优化和原始结果归档。

本设计提交评审，按评审意见持续修订。代码 commit、正式真机精度及性能数据在后续提交中补充；设计合入、代码合入和平台验收分别记录。
