# Alpamayo-R1 图模式与预取优化设计

提交账号：[zhouhongzhe](https://gitcode.com/zhouhongzhe)。本文提交设计评审，**绝对精度尚未达标，不构成最终验收通过声明**。实现代码仅保存在个人 fork，尚未提交实现合入 PR。2026-10-21 已向任务广场提交含精度差异说明的审阅 ZIP，尚未验收通过；本设计仍待人工评审与合入。

## 需求背景

对应“10月社区任务-Alpamayo-R1图模式与预取优化”。在固定权重、BF16、输入和采样配置下，将确定性计算接入 capture/replay，并对实际 profiling 确认的搬运热点使能权重预取，降低 A2/A3 端到端耗时。

方案范围为 `navigation/alpamayo-r1/` 和 `docs/navigation/alpamayo-r1/`，不修改公共 `module/`、系统框架或模型权重。模型私有适配位于样例 `patches/` 中，通过实例级 monkey patch 安装和卸载。

- [个人 fork 方案 Issue](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/issues/1)
- [R37固定代码](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/tree/ce5a67436acee8ac5477d9c7fb54c56dfc791b13)
- [当前评测协议与RFC](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/blob/ef592f8e9f4028fdfdc94740b793acf5576d2921/docs/navigation/alpamayo-r1/graph-prefetch-rfc.md)
- [R37完整验证说明](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/blob/ef592f8e9f4028fdfdc94740b793acf5576d2921/docs/navigation/alpamayo-r1/validation-r37.md)与[逐次测量JSON](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/blob/ef592f8e9f4028fdfdc94740b793acf5576d2921/docs/navigation/alpamayo-r1/validation-r37.json)已发布；原始Memory CSV附件见文末。

## 需求分析

`enable_graph` 与 `enable_prefetch` 独立、默认关闭。保持原始输入、轨迹数、扩散步数、权重和数值类型；关闭开关恢复相应 eager 路径。

同平台固定输入和 seed，比对原始 xyz/rot Tensor、序列化轨迹及 CoC。完整 `chunk_0000` 的 100 clips 分别运行 1/3/5/10 轨迹四档矩阵；10 轨迹同时检查不劣于 eager 和 minADE≤0.5676。失败保留非零退出码。

性能每档至少一次预热、三次测量，组合模式改善必须超过 eager 波动。稳定性每档至少20次连续推理，启用 `profile_memory`，检查输出、显存、实际 capture/replay、CMO 和同步事件。

### 冻结配置

| 项目 | 配置 |
|---|---|
| Recipes 基线 | `db9b2453ad70ac919f8c42e02a61d16db63f4e66` |
| NVIDIA 源码 | `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073`，应用样例原适配 patch |
| 本轮执行提交 | 干净执行 `58a4f8c01222a68033d0ca652d6c6e9ee0473098`，个人fork `ce5a67436acee8ac5477d9c7fb54c56dfc791b13`；两者Git tree均为`578e053c149f03b87f70995510d80e95b52c4471` |
| 权重 | `nvidia/Alpamayo-R1-10B`，revision `dd4a24cacefc9a6477a6dfc7354de2443401409d`；五个分片与官方 SHA256 一致 |
| A2 / A3 | Ascend910B2 / Ascend910_9382，单逻辑设备；driver 25.5.2 / 25.3.rc1.2 |
| 两端私有环境 | CANN8.5.0、Python3.11.17、torch2.8.0+cpu、torch-npu2.8.0.post2、transformers4.57.1、PyAV18.1.0；完整依赖见 `requirements-evaluation.txt` |
| CPU | OMP/MKL/intra-op/inter-op 均为32线程，不显式绑定 CPU/NUMA；四模式配置一致 |
| 性能输入 | 默认 clip `030c760c-ae38-49aa-9ad8-f5650a545d26`，位于 `chunk_3119`，与精度集合分开记录 |
| 精度输入 | PhysicalAI-Autonomous-Vehicles 25.10，revision `2ae73f49ffd2b5db43b404201beb7b92889f7afc`；实际 `chunk_0000` 全部100 clips，按冻结Parquet原始行顺序遍历 |
| 输入索引 SHA256 | `9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c` |
| 每请求参数 | `t0_us=5100000`、整轮seed42一次、BF16、top_p0.98、temperature0.6、max_generation_length256；扩散默认步数不变 |

按[任务维护者澄清](https://gitcode.com/cann/cann-recipes-embodied-ai/discussions/8?parent_id=97ce23396f784b949eb31cf8ce53ca14#tid-301bd8c036a34c4387ee4ffd224b3328)，模型加载后整轮seed一次，无需逐clip重置。R37已在结果产生前固定Parquet原始行顺序；每clip预热及重复测量重放同一起点RNG，下一clip只推进一次推理的随机流，协议为`seed-once-replay-per-clip-v1`。历史0.5676表格顺序仍未知，不搜索排列或seed；R13原协议和失败记录保留。模型、图/预取实现及采样算法未改。

两平台分别新建环境和干净 checkout，45项实际依赖及21个模型 Python 文件已核对。原系统环境、toolkit 和 driver 均保留。CPU线程参数为可选项，入口默认不改变线程数；实际线程和亲和性写入 `run.json`，矩阵检查拒绝缺失或不同的 CPU 配置。

## 详细设计

### 图边界与资源生命周期

采用 `torch.npu.NPUGraph` / `torch.npu.graph`，包含两类稳定子图：

1. VLM 的36个单 token decode MLP：保持 gate→SiLU→up→multiply→down 的原计算顺序。prefill、attention、采样和 KV 更新在这些图之外。
2. 原始去噪循环内的 diffusion expert forward：固定 hidden、position IDs 与私有 BSND KV 存储。每个请求复制 prompt KV，后续步骤更新未来64个 token 区域，避免反复全量拼接与布局转换。随机生成、Euler 更新及轨迹后处理在图外；外部直接 expert 调用保持原路径。

输入签名包括 shape、stride、dtype、device、autocast、prompt 长度及调用 stream；支持1/3/5/10轨迹。每个 wrapper 最多保留一个签名，签名或 stream 改变时回退，避免无界图缓存与持续重捕获。图外分配 buffer、执行预热并建立 stream 依赖；replay 前更新输入，返回输出 clone，保留旧结果的所有权。

expert 捕获仅在 dropout 关闭时为原融合 Attention 设置 `sync=True`，处理辅助 stream 事件；图内私有路径已计算 RoPE，且未来64个位置连续，因此不再向 Attention 转发冗余位置参数。修改限于该实例及原推理线程，外部调用保持原参数；实际设备 trace 检查是否新增主机同步。

训练、autograd、不支持的输入、外部捕获、并发与跨 stream 重入不共享静态 buffer。权重或配置变化使旧图失效。设备、OOM 和捕获执行异常向上传播，不吞掉错误继续运行。卸载恢复原 forward，并释放图、位置及 KV buffer；日志分别记录 warmup/capture 时间、capture/replay 和回退原因。

### 预取目标与依赖窗口

真实模型 PipeUtilization 将候选锁定为 decode MLP 的只读 `down_proj.weight`。其 MatMul 为 `10×12288 @ 12288×4096`，默认请求756次；A2/A3 的 MAC 比例约0.148/0.151，MTE2比例约0.827/0.829。该证据来自候选选择阶段的完整模型采集，不冒充最终版本的新 PMU 测量。

在 gate 激活之后，以激活 Tensor 建立依赖，调用 `torch_npu.npu_prefetch` 预取 down 权重的连续16MiB前缀；独立 up MatMul 位于预取与 down 消费之间。对象和范围由模块配置决定，不依赖输入值。预取单独运行和图内调用使用同一固定位置，组合模式的 replay 必须对应真实设备 CMO。

已检查的窗口为 Swish→CMO→up→Mul→down，CMO 在 up 开始前完成，未观察到异步重叠。预取降低部分 MatMul 时间和，但尚未证明独立端到端或 MLP 设备区间净收益；组合性能单独按端到端计时验收。

### 复现入口

按样例 README 完成固定代码、权重、数据及环境准备，在 `navigation/alpamayo-r1/` 运行：

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 python evaluate.py \
  --checkpoint model_ckpt --cache-dir dataset_cache --cpu-threads 32 \
  --output results/eager --warmup 1 --runs 3
```

其余三档分别增加 `--enable-graph`、`--enable-prefetch` 或两个开关，并通过 `--reference results/eager` 比较输出。精度入口增加 `--chunk0 --trajectories 10 --seed 42`，保留1/3/5入口。`check_matrix.py` 检查提交、输入、环境、CPU配置与四档门禁；具体性能括号、Memory及归档命令见运行说明。

## 可维可测分析

### 精度与输出一致性

每平台100 clips × 1/3/5/10轨迹 × eager/graph/prefetch/combined，每clip一次预热、三次正式测量，共9600份正式输出。同平台同轨迹数的四档及重复运行均精确匹配原始Tensor、序列化轨迹和CoC。独立oracle从原推理函数只移除请求级播种，两平台共200份n10预测均用原始张量和独立真值重新计算minADE；产品n10的2400份输出全部精确匹配oracle。

| 平台 | 1轨迹 | 3轨迹 | 5轨迹 | 10轨迹 | 10轨迹≤0.5676 |
|---|---:|---:|---:|---:|---|
| A2 | 1.8009917754 | 1.0428165472 | 0.8435808738 | 0.6040339522 | 失败 |
| A3 | 2.0288625801 | 1.0704024467 | 0.8558570467 | 0.6040339150 | 失败 |

每个数值在同平台四档完全相同；n10八个模式作业均保留原exit1。固定预测和真值后minADE也固定，图/预取边界调整不能在保持输出精确一致的同时降低该指标。维护者认为另一开发者约0.633的结果正常，但没有据此认定本项目获得数字门槛豁免。

八份矩阵归档共45111文件、两份oracle归档共822文件，均在本地逐文件验哈希。JSON重新核对9600条原始metrics，再去重保存800个片段/轨迹数指标；不以去重记录数冒充实际测量次数。

### 无profile性能复测

每平台预先固定两轮串行顺序：eager→combined→prefetch→graph→eager_after；每档一次预热、三次正式测量。端到端包含本地解码、预处理、H2D及同步后的模型执行，排除模型加载、哈希、D2H和归档。精度矩阵的并发耗时不参与性能判定。

| 平台/轮次 | eager前置中位数(s) | eager后置中位数(s) | combined中位数(s) | 合并eager收益 / 极差(s) |
|---|---:|---:|---:|---:|
| A2/1 | 10.583778 | 10.580305 | 10.320363 | 0.261678 / 0.207396 |
| A2/2 | 10.533629 | 10.575831 | 10.295755 | 0.258975 / 0.105930 |
| A3/1 | 8.609086 | 8.494075 | 8.175990 | 0.349619 / 0.133293 |
| A3/2 | 8.549320 | 8.552846 | 8.162609 | 0.388474 / 0.147156 |

四轮分别对前置、后置及合并六次eager计算收益与最大值减最小值，十二道门禁全部通过。四份归档508文件及180个原始输出文件已独立核验；每平台第二轮输出也匹配第一轮。

JSON包含四档全部逐次时间、模型/预处理时间、显存峰值、首次预热及wrapper capture时间之和。wrapper capture之和不等于整个模型编译耗时。prefetch-only未获得稳定独立收益，组合收益不能全部归因于预取。

### Memory、生命周期与泛化

A2/A3四档各20次连续推理及本地独立复核均完成，每平台80份输出、240个原始文件逐字节一致。两平台eager/prefetch的allocated/reserved均恒为22173104128/35848716288字节，graph/combined为26747814912/40217083904。图模式各37次capture、16086次replay（含一次预热），无持续重捕获；预期非decode输入回退原路径。

两平台预取两档每个正式步骤756次CMO，关闭时为0；四档每步主机同步计数相同。每平台八个窗口均为Swish→CMO→up→Mul→down，未观察到异步重叠。每平台完整6531200行operator_memory CSV中，每档最后一步各有一条0.5 KiB分配缺少释放字段；A2/A3解析日志分别提示2/4个不完整事件，不等于缺失分配行数。20次显存计数稳定，但不能证明采集结束后的释放。

| 平台/模式 | 每步kernel数 | kernel时长之和中位数(ms) | MLP设备区间之和中位数(ms) |
|---|---:|---:|---:|
| A2/eager | 86262 | 6306.209 | 306.873 |
| A2/graph | 85474 | 5915.876 | 311.909 |
| A2/prefetch | 86262 | 6284.961 | 310.752 |
| A2/combined | 85474 | 5905.263 | 316.591 |
| A3/eager | 86262 | 6186.299 | 303.247 |
| A3/graph | 85474 | 5805.274 | 305.771 |
| A3/prefetch | 86262 | 6175.189 | 307.309 |
| A3/combined | 85474 | 5852.627 | 310.065 |

MLP区间从gate MatMul开始至down结束，包含中间CMO及设备间隙，不含wrapper复制或此前主机开销。该区间没有改善，不能用部分MatMul时长下降宣称整个模块加速。R13的PipeUtilization只用于同一模型、优化模块、环境、权重、输入及输出均已核对一致的热点说明；没有新采R37 PMU。

两平台默认clip上的六组轨迹数/seed配置×四档各24场景均已通过，每平台273个归档文件及72个原始输出文件核验一致。A3先前设备占用和预热显存不足的两次退出1单独保留；恢复使用同一设备及配置，只继续未完成场景。A2按原队列一次完成。泛化运行不作为耗时证据，也不替代固定100 clips精度。

每平台四档各20次Memory的393个归档文件及生命周期、kernel、60480个MLP区间均已复核。

A2实测采用post2的进程内离线解析顺序修正，先完成CANN导出并等待timeline读取；样例README已给出相同脚本、固定版本检查及解析错误非零退出检查。A3使用标准`analyse()`。这只影响离线证据解析，不属于模型计时；完整20次采集的单档解析约需一小时。

### 工程检查与交付状态

两平台各33个CPU测试及改动文件pre-commit通过；全仓pre-commit仍存在固定基线同样的四类失败：尾部空白、文件末尾换行、shebang执行权限及既有debug import。检查在独立checkout运行，没有带入无关自动修改。实现CI尚未通过。

[validation-r37.json](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/blob/ef592f8e9f4028fdfdc94740b793acf5576d2921/docs/navigation/alpamayo-r1/validation-r37.json)仅覆盖已完成的精度与性能，SHA256为`ede7526cfe13338d6bd4a0b7a0789fceb293e752ef9dc9e76662c2bf83fca3d1`。两平台各16份完整trace/CSV已下载并校验压缩件及原始内容，A2/A3原文分别为41581505169/37444608808字节。

两平台各12份完整Memory CSV、复核报告和脱敏运行元数据已发布为[审阅附件](https://gitcode.com/zhouhongzhe/cann-recipes-embodied-ai/releases/alpamayo-r37-evidence-20261020)。两个完整ZIP均经公开接口无认证下载回读，大小及SHA256匹配冻结包：

| 平台 | ZIP字节数 | SHA256 |
|---|---:|---|
| A2 | 950896809 | `99dafba5935a1970f69170040b965f5bca50b8d8478ddc3cd35e66f59a74f85c` |
| A3 | 955083473 | `f5cbe3076858b772a9b0c7e31cc919b03e51063ac78348add1e8a0ce3992d217` |

每包33个条目均已核验，包含标准库校验命令；完整trace和原始轨迹未包含在附件中，仍不能宣称全部原始制品已公开交付。

[设计PR #2265](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2265)仍待人工评审与合入。绝对精度仍失败，实现CI及任务验收尚未完成；没有提交验收ZIP或上游实现PR。
