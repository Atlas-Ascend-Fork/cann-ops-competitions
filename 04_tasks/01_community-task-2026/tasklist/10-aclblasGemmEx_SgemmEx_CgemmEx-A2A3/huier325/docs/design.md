# 需求背景（required）

## 需求来源

昇腾社区“10月社区任务-3个GemmEx类算子开发（A2/A3）”，任务编号
`2b4e298cbf6e4ed89e7fbcfaea1a5b02`。开发账号为 huier325。
任务书 §3.1 允许自验证覆盖一种款型，本次使用 Atlas A2 910B3、CANN 9.1.0。

## 背景介绍

基于 ops-blas `8081a87d2a6f926c85705b9a2a90bd22b9edc9e8` 开发。
该基线已有 arch35 GemmEx 实现，缺少本任务的 arch22 路径；共享 SgemmEx
声明仍是旧 15 参数版本。CgemmEx 在当前基线已经存在 17 参数声明及 arch35
实现，不能按旧任务描述再增加同名平行接口。

# 需求分析（required）

实现 `C = alpha * op(A) * op(B) + beta * C`，保留列主序和前导维语义。
实数的 C 转置等价于 T，复数 C 转置必须取共轭。

| 接口 | 公开签名 | 首版主路径 |
| --- | --- | --- |
| aclblasGemmEx | 19 参数，含 computeType/algo | A/B/C 为 FLOAT32 |
| aclblasSgemmEx | 17 参数，不含 computeType/algo | A/B/C 为 FLOAT32 |
| aclblasCgemmEx | 17 参数，不含 computeType/algo | A/B/C 为 COMPLEX64 |

三个接口作为同一任务交付。首版支持表不代表任务全部要求已经完成；扩展
FP16/BF16/FP8 的必选组合需在评审中确认，并对最终声明支持的组合补齐实现和
测试，不能用现有主路径 CSV 的通过代替扩展类型验证。

# 详细设计（required）

## Host 设计与校验

新增 arch22 入口，使用任务专用共享校验函数，按照任务规定的顺序检查
handle、尺寸、前导维、标量指针、dtype 支持集、转置枚举、矩阵指针，最后
检查 GemmEx 的 computeType/algo；全部检查通过后才能执行零维短路。
不根据测试用例名称、随机种子或 dtype 999 等特定错误值分派。

`m==0` 或 `n==0` 是校验后的 no-op。`beta==0 && C==nullptr` 返回成功且
不读写 C。`alpha==0` 或 `k==0` 进入仅缩放路径，不访问 A/B。
`beta==1` 时保持 C 的原始位模式；`beta==0` 时直接写零，不读取旧 C。

## 正常矩阵乘路径

实数复用已有 A2 `aclblasSgemmStridedBatched` 的 batchCount=1 路径；复数复用
已有 `aclblasCgemmStridedBatched`。传递真实物理矩阵 span，而非零 stride，以
满足上游小形状优化的合法性判定。保留 N/T/C 和 padding 的完整语义。
上游复数 Cube 分解所需的所有计算和搬运内核均计入性能评估，不只测主 Cube。

全量 FP32 回归发现部分高幅值标量、长 K 小输出及大型 NN 的误差不达标。
当前 08 候选补充按通用形状和标量条件选择的 Device 严格运算顺序路径，
并为地址对齐的规则大型 NN 复用现有 Sgemm 路径。运算顺序原型完成 CPU
参考核验，08 候选尚未完成 CANN 编译和 Device 回归；不按用例名称分派，
也不在 Host 回退计算矩阵结果。

## 仅缩放 Device 路径

新增 Ascend C AIV 内核，按列主序逻辑行划分 32 元素 tile，只写逻辑 C。
对齐列按可用 AIV 核分配；可能共享 DMA 块的未对齐列使用单一写入者，避免
相邻列覆盖 padding 或发生写竞争。Host 只传递尺寸及 beta，不回读矩阵计算。

复数 beta 缩放对齐固定 Netlib 3.10.0 构建的实际乘加顺序。A2 向量 Axpy 和
FusedMulAdd 的极值探针未实现所需的单次舍入抵消，因此复用库内已有
`GemmSbFp32` 整数 FP32 FMA，保留 48 位乘积后统一舍入，仍在 Device 上执行。
Vector/Scalar 读取和写入之间使用 V_S、S_V 同步。有限输入的独立 CPU 精确
整数参考审计和 Device 探针仅证明其各自覆盖范围；一般 NaN payload 与 CPU
参考的逐位兼容性尚未完整证明，不据此宣布全部特殊值满足 EXACT。

## 兼容性与资源

共享头文件中的 SgemmEx 改为任务规定的 17 参数签名，同步 arch35 定义及
现有测试调用者；CgemmEx 的类型声明统一为 aclDataType 并同步 arch35。
不保留公开旧签名或产品私有平行 API。arch35 的计算算法不因本任务替换，
相关构建和回归仍是最终兼容性门禁。

新增 Host 分派不主动同步 stream。复用后端的工作区生命周期、异步行为及
跨调用资源安全仍需验证。测试中的流同步由调用方执行，不放到公开 API 中。

# 可维可测分析

## 精度标准/性能标准

原始三个 CSV 各 1,200 条，GPU baseline 各 200 条，均保留原始字节并记录 SHA。
单一 arch22 测试二进制按接口分别运行，不用同名文件互相覆盖，也不把单次
1,200 条报告当成三个接口全部通过。负向值和 null 指针按原值传入公开 API。

参考直接调用固定 Netlib 3.10.0 的 `cblas_sgemm`、`cblas_cgemm`。复数实部和
虚部分别判定，有限点通过率排除非有限点；非有限值按本任务明示的 NaN/Inf
等效规则逐项检查。绝对误差上限采用最大绝对误差点的收窄 golden 计算 ULP。
`alpha==0` 的结果逐字节检查，零维 no-op 和 C padding 对比原始输入位模式。

性能采用任务要求的 msprof op 模式，先 warmup，再获取超过 10 个有效调用
样本并计入每次调用的全部计算任务；每条 GPU/NPU 比值需达到 0.8。
task-time 筛查和 API/GTest 总耗时不替代正式 op 模式的完整 600 条证据。
性能采样测试明确标记为 performance_sampling_only，不能当精度通过证据。

XML 核验要求三个接口全部 3,600 条唯一用例，无遗漏、重复、跳过、失败或
错误；特殊用例不能因已知材料冲突而被静默排除。

## 待评审确认的材料冲突

SgemmEx `TC_ED_218/219/220` 与 CgemmEx `TC_ED_213/214/215` 对 dtype=999
期望 INVALID_VALUE；任务正文 §2.4、§3.5 对不支持/非法 dtype 规定
NOT_SUPPORTED。原 CSV 不修改，首版按正文统一返回 NOT_SUPPORTED。这六条
差异将如实报告，请评审确认正式验收口径；确认前不宣称 3,600 条全通过。

## 当前验证状态与 AI 辅助说明

A2 06 版 GemmEx 全量 1,200 条有 7 条数值失败，SgemmEx 全量 1,200 条有
9 条数值失败和 3 条材料错误码冲突；CgemmEx 全量尚未完成。08 修复候选
尚未完成 CANN 编译和 Device 回归。九条 task-time 性能筛查达标，但正式 op 采集
失败，不能替代全部 600 条正式性能证据。完整精度、性能、异步资源行为和
arch35 兼容性尚未完成。本设计 PR 用于方案评审，不是正式验收申请，
不声称已经通过测试或获得奖金。

AI 在方案分析、代码实现、测试设计、结果分析和文档整理中提供辅助。
验证结论以实际运行的源码、库哈希、原始日志和正式评审记录为依据。
