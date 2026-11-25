# 需求背景

任务 55123ab3caa64d089f525f646155094c 要求在 Atlas A2/A3 增加 aclblasSgemmBatched 和 aclblasHgemmBatched。ops-blas 已有两者的公共声明及 arch35 实现。arch22 现有复数批量乘法不能处理这两个实数接口。实现放在 blas/gemm_batched/arch22，沿用现有 handle、stream 和状态码。

## 需求分析

每个 batch 独立计算列主序 C[i] = alpha × op(A[i]) × op(B[i]) + beta × C[i]。S 的输入、标量、输出为 FP32；H 为 IEEE FP16。C 转置枚举在实数上等价于 T。输入地址表及表中的矩阵均在设备上，批次之间可不连续，标量从主机传入。对所有 batch，仅修改 C 的逻辑行，保留 ldc 填充区。

## 详细设计

Host 依次校验 handle、转置枚举、m/n/k、各 leading dimension、batchCount 与必需指针。空 handle、非法枚举和其他非法值分别返回 HANDLE_IS_NULLPTR、INVALID_ENUM、INVALID_VALUE。m、n 或 batchCount 为零时跳过发射。k 或 alpha 为零时走 C 缩放分支，允许 A/B 地址表为空。beta 为零直接写结果，以免从未初始化的 C 引入 NaN。调用方负责设备地址表元素有效及输出间无重叠。

核函数在绑定流中读取设备地址表。将 C 的行列分块与 batch 编号组合分配给核；地址偏移使用 64 位。对列主序输入，Cube 计算转写为 C 的行主序转置视图，交换 A/B 与相应转置标识。常规对齐矩阵采用 Cube 分块乘加，累加保留 FP32。长 K 和非对齐尾块需按真实 K、逻辑行列截断。非单位标量和 H 的输出收窄在乘积累加结束后处理。小矩阵及纯缩放独立调度，避免为每个 batch 在主机发起一次调用。按形状和步幅选路径，所有设备工作均使用 handle 的 stream。

## 可维可测分析

S/H 共用 test/gemm_batched/sgemm_batched/arch22 下的 CSV 驱动 GTest，用输入文件区分两个接口。每个接口运行 500 条精度、100 条性能。检查 N/T/C、非法参数、空矩阵、独立设备指针、padding 哨兵、零标量不读输入及 H 的特殊浮点值。S 逐 batch 使用 Netlib BLAS 3.10.0 cblas_sgemm 对照；H 用量化后的 FP16 操作数做 FP64 累加，结果经 FP32 再舍入到 FP16。S/H 的混合容差分别为 2^-13、2^-10，匹配率须达到 99%，同时检查任务规定的最大误差。

性能按 910B3、CANN 9.1.0 核验。正式数据由 msprof op 获取，每例预热 5 次、采集 10 次，单次调用如包含多个 kernel，应汇总该次调用的所有 kernel 后取 10 次均值。NPU 均值不超过对应附件 gpu_ms 的 1.25 倍。辅助计时在同一 stream 上使用 aclrtEvent 包围预置 device 缓冲的连续 10 次调用。两种计时均保留逐例记录。任务书 §3.4 未设内存达标项，报告模板相应栏标明不适用。arch35 和公共 ABI 沿用现有实现。
