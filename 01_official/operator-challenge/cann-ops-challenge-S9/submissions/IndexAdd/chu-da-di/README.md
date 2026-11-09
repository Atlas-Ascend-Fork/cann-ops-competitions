# IndexAdd 算子

- 团队：锄大地
- GitCode：[MuGuC](https://gitcode.com/MuGuC)
- 赛事：算子挑战赛 S9 赛季

## 功能与接口

沿指定维度按 INT32 索引把 source 累加到 self，生成 out。

输入 self、index、source，属性 dim 为 int64，输出 out。值类型支持 FP16、BF16、FP32、INT32、INT8；index 为 INT32。原提交协议中的 alpha 由调用方先乘入 source，本算子 schema 不含 alpha 参数。

## 核心实现

按类型选择整数驻留或流式累加、浮点有序分组及波次处理路径；处理重复索引并由 Host 控制容量、分核与回退条件。

## 代码版本

源码取自最终线上提交包 `IndexAdd_general_dma_20260913.zip`，版本为 **general DMA 20260913**。`op_host` 和 `op_kernel` 内的 C++ 与头文件仅统一文本换行为 LF，其余源码字节与该包一致；构建文件及本说明用于恢复独立编译工程。原始成员和合仓副本的 SHA256 均记录在 `SOURCE_MANIFEST.json`。

## Host 形状与类型推导

新增的 `op_host/infer_shape.cpp` 为图模式注册形状与输出类型推导，校验广播、操作轴、索引维度或排列约束；原有 Tiling 与 Device 计算源码保持线上版本。`tests/meta_shape_test.cpp` 覆盖有效输入、未知维度和无效参数，构建脚本会运行这些 CPU 元数据检查。

## 构建

环境为 Linux AArch64、Ascend 910B 系列设备、CANN 社区版 8.5.0，以及 CANN 对应的 C/C++ 工具链和 CMake。先指定实际 SDK 路径，再编译：

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0
BUILD_JOBS=4 bash build.sh
```

默认构建目录为 `build_out`，可通过 `BUILD_DIR` 指定独立目录；打包产物由 CANN 的 `package` 目标生成。也可先设置上述环境变量，再使用 CMake 的 `default` 配置预设。切换 SDK 时使用新的构建目录。

## 验证与性能记录

历史最终线上包的总耗时为 **589.2425 μs**，来源为团队比赛反馈归档；该数值对应历史完整评测协议，不代表当前设备上的单次调用耗时。新的构建产物需要重新进行编译、运行和精度验证。

涉及工作空间和跨核同步的路径需要使用生成的完整 ACLNN 接口；各算子在独立安装目录中验证，按生成的 `aclnn_*.h` 调用接口。

## 许可证

遵循仓库根目录的 MIT 许可证。
