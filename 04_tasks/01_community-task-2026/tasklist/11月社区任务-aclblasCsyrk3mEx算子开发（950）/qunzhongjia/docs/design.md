# aclblasCsyrk3mEx 设计文档

## 需求背景

列主序 X 的对称秩 k 更新 C=alpha·XXᵀ+beta·C。OP_C 与 OP_T 都按普通转置读取。alpha、beta 以及 C 用复数单精度，A 兼容 COMPLEX64 与交错有符号 INT8。

目标平台为 Ascend 950PR（arch35）。本方案采用按输出块片上构造三乘积。[官方任务及附件](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)

## 需求分析

沿用 ops-blas 公共头文件的入口、枚举和状态码。Host 按任务书 §2.4–2.5 校验结构参数，区分空规模、标量短路和实际读写张量的情形。设备侧从 handle 绑定的 stream 执行，所有地址计算使用能容纳物理跨度的整数。

取单元素 X=1+i、alpha=1、beta=0，期望对角为 2i。覆盖 OP_C/OP_T 等价、非实 beta、int8 极值和非指定三角及列间哨兵。

## 详细设计

### 按输出块片上构造三乘积

将指定三角划为方块，按行区间分配核心，每个块拥有最终写权限。每次读入两组 X 的 K 切片，构造 P=UrUcᵀ、Q=VrVcᵀ、R=(Ur+Vr)(Uc+Vc)ᵀ，单个计算核内融合完成，保留 P−Q 与 R−P−Q。

用同一微核覆盖上下三角，将边界裁剪放在写回阶段。对角虚部由 2·Σ(UV) 决定；对角不能按 Hermitian 语义清零。

实现落在 `blas/syrk/arch35/`，GTest 接入 `test/syrk/csyrk3m_ex/arch35/`。同一 API 在不同 tiling 下采用相同的参数校验与数据类型约定。对齐搬运的最后一段以逻辑末元素为界，必要时用带掩码的存取处理尾部。

## 可维可测分析

在设备侧通过公共 API 发起计算，用 Netlib CSYRK 或 ZSYRK 收窄 生成独立参考值。每个用例保留随机种子、输入类型、期望状态与实际状态；正常输出按用例的 `mere_threshold` 和 `mare_multiplier` 核对，复数输出分别统计实部和虚部。计数由本期随包 CSV 得出。

| 官方用例表 | 精度及契约用例 | 性能用例 |
| --- | ---: | ---: |
| `csyrk3m_ex_test.csv` | 1000 | 200 |

正式性能采集先预热 5 次，后取 10 次有效调用。用 msprof op 读取 Task Duration，每次先合计该调用的计算内核再求均值，换算为 ms 后与对应变体的 gpu_baseline.csv 对齐。每项 GPU/NPU ≥ 0.4，200 个官方编号的加速比算术平均 ≥ 0.5。另按任务书覆盖两种 A 编码。记录实际加载的库、设备和原始计时。任务书 §3.4 将内存专项列为不涉及。
