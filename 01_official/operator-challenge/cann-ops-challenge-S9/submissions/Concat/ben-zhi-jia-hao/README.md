# Concat

本目录提交本质嘉豪团队的 S9 Concat 算子工程。将输入张量列表沿指定维度拼接，输出拼接结果。

## 团队信息

- 团队名称：本质嘉豪
- 所属单位：华中科技大学
- 联系方式：通过本 PR 提交者联系团队

## 接口

| 项目 | 说明 |
| --- | --- |
| 输入 | 动态输入列表 `x` |
| 输出 | `y`，与输入 dtype 一致 |
| 数据类型 | float16、float32、int32、int8 |
| 数据格式 | ND |
| 属性 | `dim`：拼接维度，默认 0。 |

支持负轴；非拼接维度需一致。

## 工程结构

- `op_host/`：算子注册、类型与维度推导、Tiling 及其依赖文件。
- `op_kernel/`：Ascend C Kernel 及其依赖文件。
- `CMakeLists.txt`、`CMakePresets.json`、`build.sh`：工程配置和构建入口。
- `cmake/`、`scripts/`：代码生成、编译和 OPP 安装包制作所需的工具。

## 编译

使用 Linux、CANN 社区版 8.5.0、CMake 3.19 或更新版本，以及 CANN 配套的
Python 与 C++ 编译环境。目标计算单元为 `ascend910b`；本工程默认进行本机
OPP 安装包构建，执行精度与性能验证需要对应的 Ascend 910B 设备。

在 CANN 环境中进入本目录，执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
```

如安装位置不同，请使用实际的 `set_env.sh` 路径。也可通过
`ASCEND_CANN_PACKAGE_PATH` 指定 toolkit 目录。构建时使用不含空格的工程路径。
每次构建会重新创建 `build_out/`，安装包输出到 `build_out/custom_opp_*.run`。

需要降低并发时执行：

```bash
S9_BUILD_JOBS=4 bash build.sh
```

仅生成编译产物时可使用 `bash build.sh binary`。构建脚本在任一步骤失败时停止，
不将失败过程报告为成功。安装及设备验证应使用本次重新构建的安装包。

## 验证说明

本次已完成源码文件完整性、项目内引用依赖、构建配置与脚本语法的静态检查。
CANN 环境中的完整重编译以及设备侧精度、性能测试尚未在本次整理环境执行。
本说明不包含未经本次验证的性能结果。

## 参考

- [S9 官方提交规范](https://gitcode.com/cann/cann-ops-competitions/tree/master/01_official/operator-challenge/cann-ops-challenge-S9)
- 构建工具来源和许可证见 `THIRD_PARTY_NOTICES.md`。
