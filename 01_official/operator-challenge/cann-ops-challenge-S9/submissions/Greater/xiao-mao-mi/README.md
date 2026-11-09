# Greater — 小茂密

中山大学，小茂密团队，算子挑战赛 S9 Greater 源码提交。

## 实现

Host 根据数据类型、广播关系、连续维度和片上空间选择 tiling；设备侧使用 Ascend C 完成严格大于比较并输出 bool。包含连续输入、广播和尾块处理路径。int32 保留精确整数比较；宽面板路径采用 epoch mask 暂存与预取，FP32 使用对应的分阶段路径。

## 构建

环境：Linux、CANN 8.5、Ascend 910B。加载 CANN 环境，使 ASCEND_HOME_PATH 指向 Toolkit 安装目录后，在本目录执行：

```bash
bash build.sh
```

提交保留完整 op_host、op_kernel、cmake 和 scripts 构建依赖。CMakePresets.json 使用环境中的 Toolkit 路径，默认本机编译；交叉编译需按实际工具链调整配置。

## 版本与验证

算法源码来自最后提交包 Greater_R1_epoch4_prefetch_EXPERIMENTAL_cann85_20260915.zip 对应工程；本次合仓未修改算法，仅整理目录及构建路径。

该包历史比赛结果五项均 Pass：4.728、385.312、44.392、77.692、200.492 μs，合计 712.616 μs。该记录不代表重新测量结果，也不将榜单 508.0205 μs 归属于本源码。原工程已完成 CANN 8.5 构建；本次便携构建配置等待仓库 CI 验证。

AI 辅助：OpenAI Codex 参与源码迭代、资料整理与合仓。

## 合仓记录

官方合仓申请：<https://gitcode.com/cann/cann-ops-competitions/pull/2690>。仓库流程检查已通过；该流程未执行设备侧编译或性能测试，不替代上板验证。
