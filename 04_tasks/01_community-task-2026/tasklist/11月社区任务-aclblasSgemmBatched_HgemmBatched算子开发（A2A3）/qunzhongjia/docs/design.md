# aclblasSgemmBatched / HgemmBatched 设计文档

## 需求背景

S/H 两个指针数组接口共用矩阵形状和前导维；三个数组的第 b 个地址定义独立矩阵，列主序下分别执行 Cb=alpha·op(Ab)·op(Bb)+beta·Cb。

目标平台为 Atlas A2/A3（arch22），性能验收使用 Atlas 800T A2（910B3）。本方案采用以 batch 内矩形输出块分核。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

将指针表顺序打乱，batch 使用不同填充值，验证未把地址表当作连续张量。覆盖 batchCount=0、各前导维尾部及 beta=0 时旧 C 含 NaN。

## 详细设计

### 以 batch 内矩形输出块分核

把 (batch, 输出列块, 输出行块) 展开为线性任务。核从指针表取得三个基址，遍历 K 面板，在片上完成乘加；写回按 S/H 路径转换，FP16 只在输出端收窄。

小矩阵由一个核连续领取多个 batch，较大矩阵按输出矩形并行。地址表只读，避免对矩阵间距作等距假设。

实现落在 `blas/gemm_batched/arch22/`，GTest 接入 `test/gemm_batched/sgemm_batched/arch22/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 逐 batch 的 SGEMM、高精度计算后收窄的 H 参考 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `hgemm_batched_test.csv` | 500 | 100 |
| `sgemm_batched_test.csv` | 500 | 100 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 NPU 平均内核时间 ≤ 对应 GPU 基线的 1.25 倍。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
