# Concat

团队：光明实验室-心跳砰砰砰。

沿指定维度拼接输入张量，支持 float16、float32、int32、int8 及负轴，采用多核分块搬运与 UB 内重排。

环境：Ascend 910B、CANN 8.5.0。加载 CANN 环境后执行：

```bash
bash build.sh
```

安装包生成于 `build_out/`。
