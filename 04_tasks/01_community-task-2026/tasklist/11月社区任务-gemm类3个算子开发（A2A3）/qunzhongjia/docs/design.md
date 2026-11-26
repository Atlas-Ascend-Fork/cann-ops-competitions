# aclblasSgemm / Cgemm / Hgemm 设计文档

## 需求背景

对 FLOAT32、COMPLEX64、FLOAT16 三种公共入口分别计算 C=alpha·op(A)·op(B)+beta·C。实数 OP_C 与转置一致，复数 OP_C 还改变虚部符号。

目标平台为 Atlas A2/A3（arch22），性能验收使用 Atlas 800T A2（910B3）。本方案采用统一输出块驱动。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

在相同维度上分别调用三个入口，使用非整块 K、带填充的 lda/ldb/ldc；复数补充非实 alpha 和 OP_C；FP16 独立验证最终舍入。

## 详细设计

### 统一输出块驱动

按输出矩形分核，沿 K 双缓冲 A/B 面板。S 保留 FP32 累加；H 以 FP32 累加后收窄。C 的面板分成实虚两路，用四个实数乘积恢复复积，实虚分别执行缩放。

每个输出块直接写回一次。M/N/K 的非整块部分分别屏蔽访存，保留 ldc 的列间填充。

实现落在 `blas/gemm/arch22/`，GTest 接入 `test/gemm/arch22/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 Netlib SGEMM、CGEMM 及半精度结果收窄参考 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `cgemm_test.csv` | 333 | 67 |
| `hgemm_test.csv` | 333 | 66 |
| `sgemm_test.csv` | 334 | 67 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 NPU 平均内核时间 ≤ 对应 GPU 基线的 1.25 倍。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
