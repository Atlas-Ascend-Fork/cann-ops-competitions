# ragged page indices 打包（PackedPageIndices）算子设计文档

作者：TreamTik

# 需求背景（required）

## 需求来源

对应《题目6：ragged page indices 打包算子开发任务书》。代码目标为 [xLLM NPU 算子目录](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops)，个人代码仓库为 [TreamTik/xllm](https://github.com/TreamTik/xllm)。

## 背景介绍

变长 Attention 需要将每个请求的有效页表前缀压紧，再按层翻译为物理页号。本算子融合页号整理步骤，保留请求顺序和负页号的 padding 语义，不搬运 K/V 内容。

# 需求分析（required）

## 需求描述

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| X | [R,C] | INT32 | 每个请求的逻辑页表 |
| M | [B] 或 [L,B] | INT32 | 单层或逐层物理页号映射，值非负 |
| cu | [R+1] | INT32 | 从 0 开始的非递减前缀和 |
| output | [P] 或 [L,P] | INT32 | 独立连续输出，P=cu[R] |

输入位于同一 NPU，布局连续，支持非零 storage offset。R、C、P 可为零，L、B 至少为 1。单个输入和输出的元素总数均不超过 INT32_MAX。仅检查和读取每行有效前缀，尾部可以包含任意值。

## 需求拆解

提供 Python `prepare(inputs, attributes)`、`prepared.run()` 和 `run(inputs, attributes)`。attributes 为空字典。prepare 校验输入并准备固定形状的编译参数；run 使用当前输入和当前 stream，每次新分配输出。prepare 后总 P 固定，支持同总量 counts 更新和 X/M/cu 同形状 set_ 重绑定。

# 详细设计（required）

## 算子分析

令 `count[r]=cu[r+1]-cu[r]`。对 `0<=j<count[r]`：

```text
p = cu[r] + j
logical = X[r,j]
output[l,p] = -1 if logical < 0 else M[l,logical]
```

例如 counts 为 `[2,0,1]` 时，输出依次保存第一个请求的两页和第三个请求的一页。X 的元素地址为 `r*C+j`，M 为 `l*B+logical`，输出为 `l*P+p`。映射值保持 INT32，不经过浮点转换，保留 INT32_MAX。

## 算子实现

### host 侧设计

Python 在 prepare 中检查输入个数、属性、dtype、设备、连续性、尺寸、cu 合法性和有效页号值域。一次性校验允许读取 CPU 数据，执行阶段在 NPU 上完成。持有 Tensor 对象和形状，不持有冻结的裸输入指针。

按输入几何尺寸、片上表和中间向量的内存预算选择融合路径、打包后片上映射路径或原生 Gather 路径，选择只依赖输入几何尺寸。Triton 编译和主机 C++ 扩展构建在计时前完成。C++ 扩展通过 PyTorch dispatcher 调用 NPU 原生 index_select，不实现 CPU 页号翻译。

### kernel 侧设计

小表路径在 Triton 核中读取当前 cu，只加载 X 的有效前缀，并将映射表放入片上内存。按输出连续分块，通过 cu 查找所属请求，再使用片上 gather 翻译页号。一次核调用完成打包与映射。

较长页表先用 Triton 核生成长度 P 的 INT32 紧凑逻辑页号。后续 Triton 核按连续输出分块，将当前层及下一层的映射表加载到片上，再翻译页号并处理负值。每个块至多跨两层，跨层读取拆成两个连续区间，避免全局内存中的逐元素 gather。

单层映射表超过片上预算，或紧凑页表很短时，使用原生 Gather 路径。Triton 生成长度 P 的安全索引和负页号标记，C++ 调用 NPU index_select 得到 `[L,P]` 输出，再用 Triton 核将负页号位置写为 -1。单层入口返回 `[P]` 视图。P=0 时直接分配空输出。

C++ 负责非空调用的输出及临时张量分配和调度。Triton 编译参数在 prepare 中固定，执行时向已编译启动器传入当前 Tensor 对象，保留分配器的流跟踪及 Triton 启动钩子。

### 流、生命周期与内存

每次调用取得当前 NPU stream，Tensor 参数使用当次存储。prepare 不固定 stream。输出和临时张量由 PyTorch 分配器管理，不在 prepared 对象中缓存输出或每次调用的工作区。

融合路径没有显式 GM 临时张量。打包后片上映射路径保留 `4*P` 字节的 INT32 紧凑页号。原生 Gather 路径的显式临时张量为 INT64 索引 P 项和 INT32 标记 P 项，共 `12*P` 字节，另加输出 `4*L*P` 字节及原生算子运行时开销。内存报告按预热后分配器可见的活跃分配增量实测，不将上述公式当作总 HBM 用量。

# 可维可测分析

## 精度标准/性能标准

使用任务包独立 CPU 标杆做 INT32 逐元素精确比较。覆盖 14 个正向输入、额外单层大输入、8 项非法输入、动态 X/M/cu、INT32 上界、offset=1 重绑定、双流各保留 3 输出和 20 次释放回收。

主性能输入为 `[8,64]/[8,1024]/P=287` 和 `[64,256]/[32,4096]/P=7898`，目标分别为 40 μs 和 75 μs。两侧各预热 50 次，交替测量 6 轮，每轮 200 次完整 prepared.run 并在轮末同步。提交全部轮次，检查均值、中位数收益和 10000 次配对 bootstrap 的 95% 置信区间下界。seed 为 20261031，分位数采用线性插值。输出总数 32736、32768 的补充输入另做相同配对计时。

## 兼容性分析

依赖 PyTorch、torch_npu、triton-ascend、C++ 编译器和 Ninja，使用 CANN 9.1.0。工程沿用 xLLM 的 Python Triton、NPU C++ 和 tests/python 目录。独立测试不加载模型权重，不生成 ACLNN 自定义安装包。

正式目标为 Atlas A3。当前按用户指定在 Ascend 910B 上开发和调优；910B 的实测只作为开发验证，Atlas A3 的正式精度、性能和内存验收待取得机器后补测。
