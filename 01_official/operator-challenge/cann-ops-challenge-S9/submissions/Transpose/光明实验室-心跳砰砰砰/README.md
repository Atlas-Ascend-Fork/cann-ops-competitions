# Transpose

团队：光明实验室-心跳砰砰砰。提交账号及联系方式：[Qianqiuer](https://gitcode.com/Qianqiuer)。

按 `perm` 重排ND张量，支持float16、float32、int32、int8和1至8维输入；支持合法负轴索引。Host完成参数校验、输出形状推导与分片，最多使用40个AIV核。

根据连续维、对齐关系和UB容量选择分块路径，以DataCopy/DataCopyPad、Gather及TransDataTo5HD完成搬运和重排；使用流水缓冲、任务均衡和缓存策略减少访存开销。每次调用均处理当前输入，workspace仅用于本次调用的中间数据。

## 构建

环境为Linux、Ascend 910B、CANN社区版8.5.0、CMake 3.19及以上，并安装CANN要求的C++编译器和Python依赖。在已加载CANN `set_env.sh` 的终端执行：

```bash
bash build.sh
```

也可显式指定CANN路径：

```bash
ASCEND_CANN_PACKAGE_PATH=/path/to/cann-8.5.0 bash build.sh
```

生成的 `.run` 安装包位于 `build_out/` 下。
