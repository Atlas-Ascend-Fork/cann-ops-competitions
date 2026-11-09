# SquareSumV1 — nju

团队：`nju`。本目录提交 CANN 算子挑战赛 S9 的 SquareSumV1 实现。

## 版本与验证

Host、Kernel、TilingData 及两侧 CMake 文件与 S9 最后一次正式提交的源码逐字节一致：

- 提交次数：449；提交时间：2026-09-15 01:00:54（UTC+8）。
- 源码版本：`4f63c394db2c981c4ce6eb73e947ccc63a2695b0`。
- 评测包 SHA-256：`91a78b7365daa7eca2b7ec02008b53c743e760a370700366fa744658edabc3b2`。
- 官方结果：5 个用例全部 Pass，耗时合计 1386.668 μs。

| 用例 | 结果 | 耗时（μs） |
| --- | --- | ---: |
| Case1 | Pass | 3.080 |
| Case2 | Pass | 144.413 |
| Case3 | Pass | 47.231 |
| Case4 | Pass | 455.939 |
| Case5 | Pass | 736.005 |

以上是该历史提交的评测结果。本次合仓补齐完整构建框架和说明文档；合仓流水线结果以当前 PR 为准。

## 实现与优化

- Host 归一化归约轴，处理负轴与 `keep_dims`，完成输出形状和类型推导；根据数据类型、归约布局、UB 容量及工作量选择 Tiling 和多核划分。
- Kernel 使用 Ascend C 实现平方与归约，包含连续尾轴、首轴、中间轴及分离多轴归约路径。
- 通过分组加载、双缓冲、流水调度、对齐搬运和多组局部累加降低数据搬运与归约开销；跨核归约通过 workspace 保存局部结果并同步。
- 提供 FP16、BF16、FP32 的算子注册与对应设备实现。历史五个用例通过不代表对所有输入的穷尽验证。

## 目录

`op_host/` 包含算子注册、InferShape、InferDataType、Tiling 和 TilingData 头文件；`op_kernel/` 包含设备入口与各归约路径。`cmake/`、`scripts/`、顶层 CMake 文件和 `build.sh` 提供完整构建支持；`framework/` 保留原工程的 TensorFlow 适配代码。

## 构建

目标环境：Linux、Ascend 910B、CANN 社区版 8.5.0、CMake、Python 3 和 C++ 编译器。先加载本机 CANN 环境脚本。

在本目录运行以下命令，并将工具包路径替换为本机的 CANN 8.5.0 路径：

```bash
cmake -S . -B build_out \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DENABLE_SOURCE_PACKAGE=True \
  -DENABLE_BINARY_PACKAGE=True \
  -DENABLE_TEST=False \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build_out"
cmake --build build_out --target binary -j2
cmake --build build_out --target package -j2
```

安装包在 `build_out/` 中生成。若使用保留的 `bash build.sh`，应先将 `CMakePresets.json` 中的 `ASCEND_CANN_PACKAGE_PATH` 改为本机工具包路径，并加载对应环境。
