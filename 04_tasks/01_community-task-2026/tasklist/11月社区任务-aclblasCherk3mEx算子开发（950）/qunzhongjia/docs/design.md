# aclblasCherk3mEx 设计文档

## 需求背景

仅更新 C 所选三角，计算 alpha·X·Xᴴ+beta·C。alpha、beta 为 FP32 实数，A 是 COMPLEX64 或实虚交错的 INT8，C 是 COMPLEX64。OP_N、OP_C 有定义，OP_T 返回 INVALID_ENUM。

目标平台为 Ascend 950PR（arch35）。本方案采用三角面板预打包。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/5f0045789d554794aca26d146e93d6b7)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

构造两行 X=(1+i,2) 与 (2i,1)，手算非对角元素并验证交换行后为共轭。测试 int8 极值、对角虚部污染、OP_T 拒绝、未选三角和 padding 保持原值。

## 详细设计

### 三角面板预打包

将 X 的行面板拆分为 U、V。右端共轭后复乘，计算 P=UUᵀ、Q=V(−V)ᵀ、R=(U+V)(U−V)ᵀ，在同一次计算核启动内得到实部 P−Q、虚部 R−P−Q。用 64 位前缀给上或下三角块编号。

对角块屏蔽半块外地址，对角单独计算实数模方并写零虚部。INT8 逐分量带符号提升后才求和。

实现落在 `blas/herk/arch35/`，GTest 接入 `test/herk/cherk3m_ex/arch35/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 Netlib CHERK 或 ZHERK 收窄 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `cherk3m_ex_test.csv` | 1000 | 200 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 GPU/NPU ≥ 0.4，200 个官方编号的加速比算术平均 ≥ 0.5。另按任务书覆盖两种 A 编码。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
