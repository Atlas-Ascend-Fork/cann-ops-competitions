# SquareSumV1

团队：光明实验室-心跳砰砰砰。

基于 Ascend C 实现指定轴上的平方和，支持 float16、bfloat16、float32、多轴、负轴及 keep_dims。

环境：Ascend 910B、CANN 社区版 8.5.0、CMake 3.19 及以上。加载 CANN 环境后，在本目录执行：

```bash
bash build.sh
```

通过 `ASCEND_HOME_PATH` 指定 CANN 安装目录，`BUILD_JOBS` 设置编译并行数。安装包生成于 `build_out/`。
