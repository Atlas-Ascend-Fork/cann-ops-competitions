# Greater 算子

- 团队：锄大地
- GitCode：[MuGuC](https://gitcode.com/MuGuC)
- 赛事：算子挑战赛 S9 赛季

## 功能与接口

按广播规则逐元素计算 x1 > x2，输出布尔张量。

输入 x1、x2，输出 y 为 BOOL。输入支持 BF16、FP16、FP32、INT32、INT8。

## 核心实现

Host 合并连续广播维度并按资源条件选择线性、周期广播及行批处理；Kernel 将比较位图驻留在 UB 中，再展开写出布尔结果。

## 代码版本

源码取自最终线上提交包 `Greater.zip`，版本为 **generalized 20260912 215822**。`op_host` 和 `op_kernel` 内的 C++ 与头文件仅统一文本换行为 LF，其余源码字节与该包一致；构建文件及本说明用于恢复独立编译工程。原始成员和合仓副本的 SHA256 均记录在 `SOURCE_MANIFEST.json`。

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

历史最终线上包的总耗时为 **467.419 μs**，来源为团队比赛反馈归档；该数值对应历史完整评测协议，不代表当前设备上的单次调用耗时。新的构建产物需要重新进行编译、运行和精度验证。

涉及工作空间和跨核同步的路径需要使用生成的完整 ACLNN 接口；各算子在独立安装目录中验证，按生成的 `aclnn_*.h` 调用接口。

## 许可证

遵循仓库根目录的 MIT 许可证。
