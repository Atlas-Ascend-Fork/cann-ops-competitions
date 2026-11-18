# token-offset tile 标志（TokenOffsetTileFlags）算子设计文档

作者：TreamTik

# 需求背景（required）

## 需求来源

对应《题目10：token-offset tile标志算子开发任务书》。代码目标为 [xLLM NPU 算子目录](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops)，个人仓库为 [TreamTik/xllm](https://github.com/TreamTik/xllm)。

## 背景介绍

变长请求访问 KV Cache 前，需要检查 token offset tile 是否完整、对齐且连续。本算子融合收集、逐元素比较和归约，输出布局标志。后续 Attention 快路径接入不在本题范围。

# 需求分析（required）

## 需求描述

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| offsets | [N] | INT32 | 非负 token offset |
| indptr | [R+1] | INT32 | 非递减分段边界，从 0 到 N |
| lengths | [R] | INT32 | 请求有效长度，不超过段长 |
| tileptr | [R+1] | INT32 | 固定为 [0,K,...,R*K] |
| flags | [R*K] | UINT8 | 独立连续输出 |

输入在同一 NPU 上连续，支持非零 storage offset。T 为 1..4096 的整数，K 为正整数，均拒绝 bool。R、N 可为零；单个输入及输出元素数不超过 INT32_MAX。每个请求满足 ceil(length/T)<=K。

## 需求拆解

提供 Python prepare(inputs, attributes)、prepared.run() 和 run(inputs, attributes)。prepare 一次性检查接口与值域，固定形状和编译参数。run 使用当前输入及当前 stream，每次新分配输出。

# 详细设计（required）

## 算子分析

对请求 r 和输出 tile t，令 begin=indptr[r]+t*T。仅当 (t+1)*T<=lengths[r] 时检查完整 tile：

```text
first = offsets[begin]
flags[r*K+t] = (first mod T == 0) and all(offsets[begin+i] == first+i)
```

其余输出为 0。例如 T=4 时，[8,9,10,11] 输出 1，[9,10,11,12] 输出 0。地址使用 INT64。比较采用 INT32 的 data-first==lane；两个非负 INT32 值的差在 INT32 范围内，避免 first+i 溢出及 INT64 向量计算。非幂二 tile 的补齐 lane 不参与归约，段内 padding 和逻辑尾块均不读取。

## 算子实现

### host 侧设计

Python 校验四个连续 NPU INT32 向量、形状、同设备、属性、分段、有效长度、非负 offsets 和稠密 tileptr。一次性 CPU 回读仅用于校验，标志在设备计算。prepared 持有 Tensor 对象，支持合法的同形状值更新与存储重绑定；不缓存裸指针、输出或 stream。prepare 预编译启动器，run 传递当前 Tensor 对象并保留 Triton launch hooks 和分配器流跟踪，减少 Python 提交开销。

### kernel 侧设计

每个 Triton program 独占全局输出中一个 32 字节对齐区间，避免 K=17、65 等情况下不同请求写同一存储块。每个输出位置由全局序号推导请求号和 tile 号，收集当前分段元数据及 tile 地址。完整 tile 连续搬入二维 lane 矩阵，再执行本地比较和归约。

flag、row 和 tile 使用 INT32。请求号、tile 号和首值对齐均采用精确常量除法：幂二使用位移；非幂二先计算 q=(INT64(x)*ceil(2^32/d))>>32，再按 q*d>x 修正一次。x 非负且不超过 INT32_MAX，乘积在 INT64 范围内。该实现避免当前 910B Triton 后端 signed INT32 除法转浮点造成的高位舍入。

完整性在幂二 T 时以 tile<(length>>log2(T)) 判断；非幂二以 (INT64(tile)+1)*T<=INT64(length) 判断。GM 地址中的 tile*T 先转 INT64。数据矩阵的连续性比较仍使用 INT32。

lane 宽度为 next_power_of_2(T)，每组 tile 数为 min(32,4096/lane宽度)。每组最多 4096 个元素，控制片上数据和比较临时量。核内收集首值、检查对齐、归约连续性并写出 UINT8。空 offsets 直接分配设备零输出，空请求返回空输出。

### 流、生命周期与内存

每次在调用线程当前 NPU stream 启动 Triton。输出由 PyTorch 分配器管理，每次独立分配；计算没有显式 GM 临时张量。报告按预热后活跃分配增量测量，包含分配器对齐，不等同于总 HBM 用量。大小两组都检查双流各保留 3 输出以及候选和基线各 20 次释放回收。

# 可维可测分析

## 精度标准/性能标准

使用任务包独立 CPU 标杆逐元素精确比较，覆盖 14 个正向、9 个非法输入、INT32 上界、非幂二补齐 lane、空输入、padding、动态值、storage offset 和独立输出。另有高位 offset 对齐和 14 组常量的高位精确除法正式回归。

主性能 R=8/T=128/K=17、R=32/T=128/K=65，均值上限均为 85 微秒。候选和冻结 PyTorch 基线各预热 50 次，交替计时 6 轮，每轮 200 次完整 prepared.run，轮末同步。报告全部轮次、均值和中位数收益、10000 次配对 bootstrap 的 95% 区间，seed=20261031，分位数线性插值。T=255、257 补充输入只提交同协议配对结果。

## 兼容性分析

依赖 PyTorch、torch_npu、triton-ascend 和 CANN 9.1.0，无 ACLNN 安装包或服务器构建。代码沿用 xLLM 的 Python Triton、tests/python 与 examples/kernels 目录。

正式目标为 Atlas A3。当前按用户指定在 Ascend 910B3 上开发验证；910B 结果仅作为开发证据，Atlas A3 正式精度、性能和内存验收待取得机器后补测。
