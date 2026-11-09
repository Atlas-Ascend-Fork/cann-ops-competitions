# IndexAdd

ventus 队伍的 S9 赛季 IndexAdd 算子实现。沿指定轴将 source 累加到 index 指定的输出位置。

环境：Linux、CANN 社区版 8.5.0、CMake 3.19 或更新版本及匹配的编译工具。默认目标为 ascend910b。

加载 CANN 的 `set_env.sh` 后，在本目录执行：

```bash
bash build.sh
```

输入各维度须为正数。
