# IndexAdd

团队：VibeThinking。目标平台：Ascend 910B，CANN 社区版 8.5.0。

- 队伍成员姓名：杜延彤，联系邮箱：3115858508@qq.com
- 队伍成员姓名：唐俊，联系邮箱：tangjun_nj@163.com
- 队伍成员姓名：赵路遥，联系邮箱：1518779148@qq.com

## 功能和接口

将 source 沿 dim 轴按 index 累加到 self，输出形状与 self 相同。

输入依次为 self、index、source，属性为 dim，输出为 out；index 为一维且取值落在目标轴范围内。

数据类型：float32、bfloat16、float16、int32、int8；index 为 int32。张量格式为 ND。

## 实现说明

Host 将输入整理为 outer、轴长度和 inner，按数据布局及 UB 容量选择执行路径。Kernel 对独立目标区域分核，结合行缓存、分组累加和通用顺序累加，处理重复索引。

op_host/index_add_tiling.h 定义并注册 TilingData，由 CANN 构建流程生成 Kernel 使用的数据定义。

## 构建

在已加载 CANN 8.5.0 开发环境的 Linux 主机上，准备 CMake 3.19 或更新版本及配套的 C++ 编译器，然后在本目录执行：

```bash
bash build.sh
```

开发套件不在默认路径时，使用 `ASCEND_CANN_PACKAGE_PATH` 指定安装目录。脚本也识别 `ASCEND_HOME_PATH`。并行数由 `JOBS` 指定，默认为 8。

```bash
ASCEND_CANN_PACKAGE_PATH=/path/to/cann JOBS=8 bash build.sh
```

构建脚本先生成算子二进制，再生成 `build_out/custom_opp_*.run`，不会自动安装。预设的厂商目录为 `customize`，与赛题调用样例一致。需要调整 CMake 变量时，可在脚本后追加 `-D变量=值`。

## 验证

安装新生成的算子包并加载配套环境后，可使用官方赛题资料中 `case_910b/IndexAdd` 的调用样例验证。执行样例前请确认当前加载的是本次构建的算子库。

本目录只提供源码与构建配置；性能数据以实际测试结果为准。
