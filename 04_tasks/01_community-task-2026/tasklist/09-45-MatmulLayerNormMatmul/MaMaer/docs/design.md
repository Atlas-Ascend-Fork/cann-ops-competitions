# MatmulLayerNormMatmul 算子设计文档

## 需求背景（required）

### 需求来源

本设计对应 [9 月社区任务 MatmulLayerNormMatmul 算子开发](https://gitcode.com/cann/catlass/discussions/22)，参与账号为 **MaMaer**。目标代码仓为 [CATLASS](https://gitcode.com/cann/catlass)，开发语言为 Ascend C，目标硬件为 Ascend 950PR。

### 背景介绍

标杆方案依次执行 `torch.mm + F.layer_norm + torch.mm`，需要多次算子启动和中间张量传递。本任务将三段计算融合到一次 kernel launch，在保证精度的前提下，减少启动、同步和中间数据搬运开销。

## 需求分析（required）

### 需求描述

算子先计算 `C0 = A0 @ B0`，沿 C0 的每一行执行 LayerNorm，再计算 `C1 = LayerNorm(C0) @ B1`。`epsilon` 固定为 `1e-6`，唯一对外输出为 C1。

| 参数 | 输入/输出 | 数据类型 | 逻辑形状 | 布局 |
| --- | --- | --- | --- | --- |
| A0 | 输入 | FP16 | `(M0, K0)` | RowMajor |
| B0 | 输入 | FP16 | `(K0, N0)` | ColumnMajor |
| B1 | 输入 | FP16 | `(N0, M1)` | ColumnMajor |
| gamma | 输入 | FP32 | `(N0,)` | ND 连续向量 |
| beta | 输入 | FP32 | `(N0,)` | ND 连续向量 |
| C1 | 输出 | FP16 | `(M0, M1)` | RowMajor |

### 需求拆解

1. 单次混合核启动完成两次矩阵乘和行级归一化。
2. 均值、方差及中间分块由算子内部管理，不增加业务输出。
3. 按任务书提供 Optest 测试件、ATK 泛化测试件和使用文档。
4. 在 Ascend 950PR 上验证精度，并以 `msprof op` 采集完整算子时延。

## 详细设计（required）

### 算子分析

对每一行 m，使用总体方差，计算如下：

```text
C0[m, :] = A0[m, :] @ B0
mean[m] = sum(C0[m, :]) / N0
var[m] = sum((C0[m, :] - mean[m])²) / N0
Z[m, n] = (C0[m, n] - mean[m]) / sqrt(var[m] + 1e-6) * gamma[n] + beta[n]
C1 = Z @ B1
```

两次矩阵乘采用 FP32 累加，LayerNorm 的统计与仿射计算使用 FP32。中间格式转换需通过生态精度标准验证，最终输出转换为 FP16。

### 算子实现

#### Host 侧设计

Host 校验输入类型、布局和矩阵维度关系，确定输出形状，并规划分块、核数和内部 workspace。分派依据为矩阵形状及片上存储容量，复用 CATLASS 的矩乘模板和调度组件。通过 CATLASS 样例及 Optest 接入调用与测试。

#### Kernel 侧设计

Cube 执行两次矩阵乘，Vector 执行均值、方差、归一化和仿射变换。按行块组织三阶段的数据依赖；LayerNorm 等待完整 N0 维的结果就绪后执行，第二次矩阵乘等待对应归一化结果就绪后执行。

优先在片上保留可复用的中间分块。超出片上容量时，以有界 workspace 衔接分块，避免按小算子拼接方式完整保存两份中间矩阵。缓冲复用前等待最后一次读取和异步写回完成，尾块仅处理有效元素。

#### 性能优化思路

- 小 M0 场景关注并行度和调度开销，评估细粒度分块及 Split-K。
- 大矩阵场景关注 Cube 吞吐、数据复用和流水重叠，分别选择两次矩阵乘的 TileShape。
- 归一化按行并行，复用 gamma/beta，减少重复搬运和标量操作。
- 根据完整融合时延选择配置，兼顾中间存储、同步和精度成本。

### 支持硬件与约束

目标平台为 Ascend 950PR，CANN 版本遵循 CATLASS 仓库要求。输入为任务规定的二维矩阵和一维仿射参数，维度须满足两次矩阵乘关系。任务测试集中的形状均为覆盖目标，扩展形状通过泛化测试验证。

## 可维可测分析

### 精度标准/性能标准

| 验收项 | 验证计划与目标 |
| --- | --- |
| 功能与精度 | 使用 Optest 覆盖任务测试集的 116 条用例，按生态算子开源精度标准判定 |
| 泛化 | 使用 ATK 验证不少于 200 条用例，覆盖形状边界、尾块、常量行及不同数值分布 |
| 内存与同步 | 检查 workspace 容量、缓冲复用和跨核依赖，结合硬件插桩检查 |
| 性能 | 使用 `msprof op --application`，读取 `OpBasicInfo*.csv` 的 `Task Duration(us)` |

性能指标为任务测试集中逐例 `标杆时延 / 融合时延` 的算术平均值，要求大于 **1.1**。记录设备、CANN 版本、源码版本、实现配置及有效采样数；异常采样不计入性能结论。本文为设计说明，精度和性能结果由后续自验证报告给出。

### 兼容性分析

采用独立算子目录及接口接入 CATLASS，保持既有算子的行为。分块、流水或中间精度调整后，对受影响的形状及边界重新验证。

### 参考资料

- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [CATLASS Optest](https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md)
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
