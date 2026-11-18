# TokenSlotAddressTranslation 设计文档

## 需求背景

分页 KV Cache 在写入 K/V 前，需要把逻辑 token slot 翻译为每层的物理 slot。
例如 `S=128`、`slot=130`、`M[1]=5`，输出为 `5*128+2=642`。
本算子只生成地址；调用方执行实际 K/V 写入。

## 需求分析

采用 Triton Ascend 实现。输入是同一 NPU 上连续的 INT32 `slots[N]` 和 `M[B]` 或 `M[L,B]`。
属性为 `{"block_size": S}`，`1 <= S <= INT32_MAX`，不接受 bool。
输出是独立、连续的 INT32 `[N]` 或 `[L,N]`。N 可以为零，B、L 至少为一。

```python
def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared: ...
def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor: ...
# prepared.run() -> torch.Tensor
```

prepare 一次性检查类型、设备、维度、连续性、元素数和实际值域。
非负 slot 满足 `slot//S < B`。所有 M 非负，且每一整页满足 `M*S+S-1 <= INT32_MAX`。
非法类型抛出 TypeError；其他非法输入抛出 ValueError。

## 详细设计

### 地址计算

对每层、每个 slot 执行：

```text
slot < 0:  output = -1
slot >= 0: page = slot // S; offset = slot - page*S
           output = M[layer,page]*S + offset
```

除余及地址合成都使用整数。负值先用零生成安全查表地址，最终恢复为 -1。
非负 slot 与合法物理页均受 INT32_MAX 约束，最终 INT32 地址乘加无溢出。
幂二 S 使用位移与整数余数。非幂二 S 取 `magic=ceil(2^32/S)`，
计算 `q=(INT64(slot)*magic)>>32`，若 `q*S>slot` 则减一。
因 slot 小于 2^31，乘积小于 2^63，商只可能高估零或一；一次纠正得到精确除余。
该路径避开当前 910B 后端不支持的 UINT32 除法转换，不使用浮点近似。
GM 扁平索引先提升为 INT64。

### 并行与搬运

主路径按连续的扁平输出分块，tile 至少为 8 个 INT32，防止相邻程序共享 32 字节输出块。
tile 不超过 N，因此每块最多跨两个层。slots 在层边界拆成两次连续读取。
程序连续载入最多两行 M，并用 `tl.gather` 在片上查表，融合地址计算与输出。
表和活跃向量须满足显式片上预算；短输入或超出预算的映射表走直接查表核。
路径选择只依赖输入几何与片上预算，不依赖用例编号、数据内容或环境开关。
空输入仅分配空输出，不启动 kernel。

### 流、动态值与生命周期

prepare 保留输入 Tensor 对象和固定元数据，并完成首次编译。
run 读取调用时的 Tensor 存储和当前 NPU stream，每次重新分配输出。
合法内容更新及同形状 `set_` 重绑定立即生效；不缓存输入指针、值或 stream。
输入允许非零 storage_offset，编译禁止对输入对齐特化。
快速启动保留 Triton hooks，并把 Tensor 对象传给 Ascend runtime 以跟踪流生命周期。
改变 shape、dtype、device 或 S 后须重新 prepare；值更新后仍须满足已校验的契约。

### 内存与依赖

主路径只分配输出，无逐次索引表或全局 workspace。映射表和中间向量位于片上。
大小主性能用例的输出有效字节数分别为 16,384 和 8,388,608。
依赖 Python、PyTorch、torch_npu 和 triton-ascend；无需 ACLNN 包或模型权重。
快速启动使用 Triton Ascend 已编译 runtime stub，实际验证版本随自测报告记录。

## 可维可测分析

独立工程提供任务原始 CPU 标杆、冻结的 NPU 基线、固定种子输入与验证驱动。
覆盖 14 个规定正向输入、10 个非法输入、INT32 边界、非幂二页长、动态更新、
存储重绑定、两条 stream 各保留 3 个输出，以及 20 次调用后的释放回收。
另外覆盖相邻尺寸和超片上预算映射路径。

性能计时使用完整 prepared.run，包含 Python 提交、每次输出分配、设备执行和最终同步。
每侧预热 50 次，交替测量 6 轮、每轮 200 次。以逐轮墙钟均值比较。
配对 bootstrap 采用 10,000 次共同重采样、seed=20261031 和线性分位数。
两个主性能用例同时检查均值差、中位数差、置信区间下界及 85/155 μs 上限。
S=64/96 两个诊断用例只提交配对结果，不套用主性能门槛。

目标验收环境为 Atlas A3、CANN 9.1.0。当前 Ascend910B3 的开发证据单独标注，A3 验收待补。
代码、复现命令、七件自验证材料及固定提交在个人 xLLM 分支提供。
