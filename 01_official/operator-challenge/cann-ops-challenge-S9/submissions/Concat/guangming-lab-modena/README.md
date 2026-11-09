# Concat — 光明实验室-MODENA

昇腾 AI 创新大赛算子挑战赛 S9 赛季 Concat 算子提交。

## 实现说明

沿 `dim` 维拼接动态输入张量列表，输出为 ND 格式；注册的数据类型为 `float32`、`float16`、`int32`、`int8`。Host 端完成形状推导、Tiling 数据生成和多核调度，Kernel 根据 TilingKey 选择通用拷贝、行批量拷贝、输入分片并行、线性拷贝或特定形状的快速路径。

ACLNN 入口保留原实现的分发逻辑：根据张量形状、步长及布局选择自定义 Kernel 或 CANN 标准算子路径。`op_host/CMakeLists.txt` 中的符号重命名和 `S9_BUILD_ACLNN_API` 编译定义用于生成并链接该封装。

## 构建

环境：Linux、CANN 社区版 8.5.0、Ascend 910B、CMake ≥ 3.19、GCC/G++、Make、Python 3。先加载实际安装位置下的 CANN `set_env.sh`，确认 `ASCEND_HOME_PATH` 指向工具包根目录，再执行：

```bash
bash build.sh
```

安装包生成在 `build_out/custom_opp_*.run`。脚本每次重新生成本目录的 `build_out`，可用 `JOBS=8 bash build.sh` 指定编译并行度。

## 目录

- `op_host/`：Tiling、InferShape、算子注册和 ACLNN 分发封装。
- `op_kernel/`：Ascend C Kernel。
- `cmake/`、`scripts/`：构建与安装包工具。
- `CMakeLists.txt`、`CMakePresets.json`、`build.sh`：构建入口。

## 构建工具来源

通用构建与打包工具复用本仓库 `01_official/operator-challenge/cann-ops-challenge-S8/submissions/Fills/终不似，少年游/` 中的构建脚手架（基准提交 `8b7d83054724cf8726d1f36beee4585e402bb7fb`），保留原版权及 makeself 许可文件；未使用该目录的算子实现。顶层构建增加环境检查和动态链接库依赖，并让 opbuild 失败及时终止。
