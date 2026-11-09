# Greater

团队：VibeThinking。目标平台：Ascend 910B，CANN 社区版 8.5.0。

- 队伍成员姓名：杜延彤，联系邮箱：3115858508@qq.com
- 队伍成员姓名：唐俊，联系邮箱：tangjun_nj@163.com
- 队伍成员姓名：赵路遥，联系邮箱：1518779148@qq.com

## 功能和接口

执行逐元素严格大于比较，支持广播，输出 bool 张量。

输入为同类型张量 x1 和 x2，输出 y 的形状按广播规则确定。

数据类型：float32、bfloat16、float16、int32、int8。张量格式为 ND。

## 实现说明

Host 合并访问模式一致的广播维度，按数据类型与缓存需求规划分块。Kernel 分别处理连续输入、标量广播和行广播，采用流水搬运并将比较掩码转换为字节输出。int32 比较保留整数精度。

op_kernel/greater_tiling.h 保存共用布局，op_host/greater_tiling.h 转接该定义。

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

安装新生成的算子包并加载配套环境后，可使用官方赛题资料中 `case_910b/Greater` 的调用样例验证。执行样例前请确认当前加载的是本次构建的算子库。

本目录只提供源码与构建配置；性能数据以实际测试结果为准。
