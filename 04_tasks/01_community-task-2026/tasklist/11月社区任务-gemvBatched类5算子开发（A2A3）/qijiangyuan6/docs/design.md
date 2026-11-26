# aclblasSgemvBatched 设计文档

## 需求背景

S/HSH/HSS/TST/TSS 五个公共入口通过设备指针数组逐批计算 y=alpha·op(A)x+beta·y。S 为 FP32；H、T 输入分别为 FP16、BF16。所有路径使用 FP32 累加；HSH/TST 分别写 FP16/BF16，其他写 FP32。

目标平台为 Atlas A2/A3（arch22），性能验收使用 Atlas 800T A2（910B3）。本方案采用长内积按归约维度分段。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

指针表使用乱序与不同分配间距；实数 OP_C 输出与 OP_T 对照。保持 A 为 lda×n 且 lda≥max(1,m)，用 m≠n、负步长和跨 batch 哨兵捕捉错址。

## 详细设计

### 长内积按归约维度分段

对极短向量让一个核处理多个 batch。长内积拆成互不重叠的连续 K 片，先形成 FP32 部分和，再按 batch、输出坐标合并。beta 在最终写者处应用一次。

将不连续的 x 访问先聚到本地连续片。按输入/输出宽度选择窗口，记录 split 前后的启动开销与部分和流量，再决定阈值。

实现落在 `blas/gemv_batched/arch22/`，GTest 接入 `test/gemv_batched/arch22/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 逐 batch 的 Netlib SGEMV，按 y 类型收窄 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `hsh_gemv_batched_test.csv` | 1000 | 200 |
| `hss_gemv_batched_test.csv` | 1000 | 200 |
| `sgemv_batched_test.csv` | 1000 | 200 |
| `tss_gemv_batched_test.csv` | 1000 | 200 |
| `tst_gemv_batched_test.csv` | 1000 | 200 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 NPU 平均内核时间 ≤ 对应 GPU 基线的 1.25 倍。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
