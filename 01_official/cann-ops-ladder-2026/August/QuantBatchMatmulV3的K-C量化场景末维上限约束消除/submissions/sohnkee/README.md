# QuantBatchMatmulV3的K-C量化场景末维上限约束消除

2026 年 CANN 算子天梯赛 August 参赛工程。GitCode 提交账号：`sohnkee`。

## 来源

取自 孑木 对应月份的最后一次评测提交 [187592](https://cannjudge.cn/public/ladder_august/cube1/submission/6a9522ed9856135024d67135)，15/15 个测试点通过。Host 与 Kernel 源码保持原评测版本。

## 结构

`op_host/` 包含注册、shape 与 tiling；`op_kernel/` 包含 Device 实现。工程附带构建配置、调用示例和平台测试文件，按仓库投稿规范保存。
