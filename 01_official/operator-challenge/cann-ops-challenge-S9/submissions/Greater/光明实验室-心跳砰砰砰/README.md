# Greater

团队：光明实验室-心跳砰砰砰。

实现逐元素严格大于比较，输入类型为 FP32、BF16、FP16、INT32、INT8，输出 BOOL。

## 构建

使用 Ascend 910B、CANN 社区版 8.5.0。将本目录文件覆盖到 CANN 8.5 生成的 Greater 自定义算子工程中，保留模板的 `cmake/` 和 `scripts/` 目录。加载 CANN 的 `set_env.sh` 后执行：

```bash
bash build.sh
```
