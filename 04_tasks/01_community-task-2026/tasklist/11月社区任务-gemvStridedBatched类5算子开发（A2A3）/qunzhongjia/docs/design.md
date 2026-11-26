# aclblasSgemvStridedBatched 设计文档

## 需求背景

五个实数 GEMV 入口通过相邻批次的元素偏移取地址，S/HSH/HSS/TST/TSS 以 FP32 累加，输出分别为 FP32/FP16/FP32/BF16/FP32。矩阵按列主序存储，转置只改变运算方向。

目标平台为 Atlas A2/A3（arch22），性能验收使用 Atlas 800T A2（910B3）。本方案采用按输出窗口跨批次调度。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

用 m≠n 且 lda>m 的转置矩阵验证最小 strideA；覆盖等于最小跨度、少一个元素、额外间隙和非正 stride，检查错误码以及相邻输出保护区。

## 详细设计

### 按输出窗口跨批次调度

以 (batch, 输出窗口) 分核，每个核从三个基址加 batch×stride 推出物理地址。N 路径按列广播 x，T/C 路径归约物理列；沿有符号 incx/incy 访问逻辑向量。

Host 以 64 位检查 strideA≥lda×n，向量跨度至少为 1+(长度−1)×abs(inc)。矩阵在 N/T/C 下始终占用 lda×n；只有结构参数合法才进入 quick return。

实现落在 `blas/gemv_strided_batched/arch22/`，GTest 接入 `test/gemv_strided_batched/arch22/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 逐 batch 的 Netlib SGEMV，按 y 类型收窄 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `hsh_gemv_strided_batched_test.csv` | 1000 | 200 |
| `hss_gemv_strided_batched_test.csv` | 1000 | 200 |
| `sgemv_strided_batched_test.csv` | 1000 | 200 |
| `tss_gemv_strided_batched_test.csv` | 1000 | 200 |
| `tst_gemv_strided_batched_test.csv` | 1000 | 200 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 NPU 平均内核时间 ≤ 对应 GPU 基线的 1.25 倍。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
