# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-aclblasCherk3mEx |
| 提交团队 | Eason_Cat |
| 任务讨论 | https://gitcode.com/cann/ops-blas/discussions |
| 适配硬件 | Atlas A5 / 950PR，arch35 |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/syrk/arch35/` |
| 测试目标目录 | `test/syrk/cherk3mex/arch35/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务。目标：Hermitian rank-k 3mEx；950 arch35 直调。

## 背景介绍

Hermitian rank-k 3mEx；950 arch35 直调。

### 约束

1. 按官方任务书接口与精度/性能口径验收。
2. A2/A3（arch22）禁止照搬 arch35 的 SIMT / Te。
3. Host 校验与 quick-return 对齐 cuBLAS / Netlib / 任务书。

# 技术方案（required）

## 总体方案

- Host：参数校验、tiling、kernel launch。
- Device：Ascend C 直调。
- 测试：CSV + GTest；精度全量 + 性能子集。

## 关键模块

1. tiling
2. kernel
3. ST golden

# 验收与风险

- 精度 / 性能按任务书；风险含非单位 stride 与复数口径。

# 任务书摘录

```text
(按官方讨论帖补齐)
```
