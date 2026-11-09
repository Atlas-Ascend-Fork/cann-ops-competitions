# SquareSumV1

团队：uuku。实现来源：算子挑战赛 S9 赛季提交版本 v284。

## 功能和实现

沿指定轴计算张量元素平方和，支持多轴与保留维度选项。`op_host/` 包含算子注册、输出形状推导和 Tiling；`op_kernel/` 包含平方、分块归约及结果写回。数据分块和核间划分由 Host 侧布局与规模参数决定，具体路径与边界处理见源码注释。

## 构建

需要 Linux、CANN 社区版 8.5.0、CMake 3.19 或以上版本，以及对应的编译工具链。目标计算单元为 `ascend910b`。

先加载 CANN 安装环境提供的 `set_env.sh`，确认 `ASCEND_HOME_PATH` 指向当前工具包，再在本目录执行：

```bash
S9_BUILD_JOBS=5 bash build.sh
```

生成的安装包位于 `build_out/`。`CMakePresets.json` 从 `ASCEND_HOME_PATH` 读取 CANN 路径，`cmake/`、`scripts/` 和 `framework/` 保留了原工程构建依赖。编译不需要 NPU，设备正确性和性能验证需要相应运行环境。

## 比赛结果

v284 的最后一次确认提交为五例全部 Pass、总用时 1398.72 µs。该数值来自比赛平台同一次完整评测，不保证其他环境具有相同表现。开源副本的 host/kernel 算法文件（`.cpp`、`.h`）与该版本源码逐字节一致；构建入口增加失败即退出，CANN 路径改为环境变量，辅助文件整理了换行和空白格式。

保留原工程模板文件的版权及许可声明。
