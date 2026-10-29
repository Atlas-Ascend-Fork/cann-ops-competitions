# aclsparseCgtsv2 算子设计（Ascend 950）

贡献者：`gcw_17nqekTF`。本文按[社区任务规范](../../../../README.md)及
[设计模板](../../../../resources/design_template.md)组织，描述正式实现
`8d0b82bedb3dd38a2de8d794613d838a1ea42221`。实现和详细测试证据供获授权评审者在
[私仓开发分支](https://gitcode.com/gcw_17nqekTF/ops-sparse-cgtsv2-a5-private/tree/codex%2Fcgtsv2-a5)
查看，七件材料位于其 `task_submission/`。

# 需求背景（required）

## 需求来源与背景

10 月社区任务 `aclsparseCgtsv2` 算子开发（Ascend 950 / A5），目标仓为
[cann/ops-sparse](https://gitcode.com/cann/ops-sparse) 的 `master`。接口语义参考
CUDA Toolkit 13.3 Update 1 的 cuSPARSE `cusparseCgtsv2`，新增 complex64 三对角
求解能力，保留已有 `aclsparseSgtsv2` 的 float ABI。核心计算由 Ascend C 在 NPU 完成。

# 需求分析（required）

## 接口与数据合同

```cpp
typedef struct { float real; float imag; } aclsparseComplex_t;
aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, const aclsparseComplex_t *dl,
    const aclsparseComplex_t *d, const aclsparseComplex_t *du,
    const aclsparseComplex_t *B, int ldb, size_t *pBufferSizeInBytes);
aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n, const aclsparseComplex_t *dl,
    const aclsparseComplex_t *d, const aclsparseComplex_t *du,
    aclsparseComplex_t *B, int ldb, void *pBuffer);
```

| 项目 | 约定 |
| --- | --- |
| 类型与形状 | complex64 为连续两个 float，大小 8 字节、对齐等于 float；`m>=3,n>=0,ldb>=m` |
| 系数 | `dl/d/du` 各为连续 `[m]` 设备数组，只读，`dl[0]=du[m-1]=0` |
| RHS | 列主序 `[ldb,n]`，地址 `B[i+j*ldb]`；仅前 m 行原地写回，padding 不变 |
| 空 RHS | `n=0` 仍检查 handle、m/ldb 和查询输出指针；查询返回 0，不发射 Kernel |
| workspace | 非空求解按查询值足量分配，设备地址 128 字节对齐 |
| 生命周期 | 在句柄 stream 异步提交，输入、B、workspace 保持有效直至设备工作完成 |

## 需求拆解

实现独立 C ABI、Host 参数与溢出校验、workspace 查询、带部分主元的复数求解和
多 RHS 调度；配套 C++ UT/ST、官方 ATK、性能/内存采集及 A2/A3 兼容回归。
raw 指针接口无法判断真实 dtype、容量或 stride，调用方负责这些条件；不支持
batchCount、独立 X、广播和任意步长。重叠调用不得共用可写 B 或 workspace。

# 详细设计（required）

## 算子分析

三对角矩阵满足 `A[i,i]=d[i]`、`A[i,i-1]=dl[i]`、`A[i,i+1]=du[i]`，对所有
RHS 列求解 `A*X=B`。通用路径共享一次矩阵分解，随后按列前代和回代。

## Host 侧设计

先校验 handle、m/n/ldb，再处理空 RHS；非空求解检查四个数据地址和 workspace。
字节数、偏移、乘法及对齐加法在计算前检查整数上界。空 handle 返回
`HANDLE_IS_NULLPTR`，无效尺寸/必需指针/对齐/范围返回 `INVALID_VALUE`；stream
获取错误原样返回，Kernel 下发错误返回 `EXECUTION_FAILED`。成功返回不代表
设备已完成或矩阵非奇异，异步错误仍需在同步点检查。

同一 stream 依次下发三个 Kernel，Host 不读回输入或路径标志，也不插入同步：

| Kernel | 职责与并行方式 |
| --- | --- |
| prepare | 最多 `min(n,AIV核数,32)` 个 block，每块 256 线程检查实际系数/RHS，准备 Jacobi 系数并保存局部条件 |
| factor | 单块归并条件；通用路径由一个 lane 执行共享 LU，保存 L/D/DU/DU2 与 pivot |
| solve | 通用 worker 使用独立 scratch 逐列求解；Jacobi 每块 512 线程处理一列，列间 grid-stride 分配 |

实际通用 worker 数同时受 workspace 预算与下发 block 数限制。路径只依赖
设备输入数值，测试编号、种子和测试模式不参与选择。

## Kernel 侧设计

### 扩展精度部分主元路径

每个内部实分量表示为 `(hi+lo+tail)*2^exponent`，三项尾数为 FP32、指数为
int64；内部 Real/Complex 分别占 24/48 字节，公开 ABI 仍为 8 字节。乘积用
显式舍入和 FMA 残差拆分，加减保持舍入顺序，输出按完整三词值一次 RNE 收窄，
含次正规边界处理；不使用 FP64 或 CPU fallback。

分解比较当前主元与下一行下对角的欧氏复数模长，相等时不交换。交换相邻行时
保存产生的第二上对角 DU2，不能退化为无主元 Thomas 算法。每个 RHS 按相同
pivot/L 前代，再按 `x[i]=(b[i]-DU[i]*x[i+1]-DU2[i]*x[i+2])/D[i]` 回代。
系数只读，每个 worker 只写自身 scratch 和当前 B 列，尾部访问按有效行界判断。

原系数、RHS 和当前解有限时，以原始 B 计算扩展精度残差，复用 LU 最多修正
两轮；非有限结果停止后续修正。非有限 RHS 保留参考求解所需的 IEEE 传播，
通过独立特殊值套件判定。扩展精度不保证任意病态矩阵的前向误差，真实失败
原样保留。逐行辅助函数通过 GM 地址传参，Kernel 源文件单独使用 Release O2。

### 输入条件限定的 Jacobi 路径

仅当 `m<=8192`，所有系数分量有限且绝对值不超过 64，逐行 `|d[i]|>=1`、
`(|dl[i]|+|du[i]|)/|d[i]|<=0.75`，所有有效 B 分量有限且绝对值不超过 16 时使用。
任一条件失败即采用通用主元路径。

预存 `alpha=dl/d`、`beta=du/d`、`c=B/d`，从 `x0=c` 开始固定 80 次更新：
`x_next[i]=c[i]-alpha[i]*x_prev[i-1]-beta[i]*x_prev[i+1]`，端点缺项取零。
每轮以 block 内同步保证读写次序，全部迭代位于一个 solve Kernel。UB 保存
c 与两份迭代值，最大 `24*8192=196,608` 字节，小于实测 253,952 字节；
超出该 m 范围不申请这份 UB。收敛界不替代 FP32 实际精度测试。

## workspace 与工程组织

令 `align128(x)` 表示向上对齐至 128 字节，查询与执行共用以下布局：

```text
F = align128(48*m)                 # 扩展复数因子或 RHS
P = align128(4*m)                  # pivot
C = align128(8*m)                  # 一列原始 B
S = F + C
workers = min(n,32,max(1,floor(8*1024*1024/S)))
W = 128 + 4*F + P + workers*S      # n>0；n=0 时 W=0
```

头部 128 字节保存路径标志，随后依次为 L/D/DU/DU2、pivot、各 worker scratch。
scratch 按列复用，不为所有 RHS 分配副本。至少保留一个 worker，因此大 m 时
单列 scratch 可超过 8 MiB 预算。Jacobi 复用同一查询空间。

公共类型位于 `include/cann_ops_sparse.h`，参数与布局位于 `sparse/cgtsv2/`，
Host/Kernel 位于 `sparse/cgtsv2/arch35/`，测试位于 `test/cgtsv2/`，由现有
CMake 按 SOC 收集。构建运行命令为 `bash build.sh --ops=cgtsv2 --soc=ascend950 --run`。

## 支持硬件与限制

本次数值实现及实测目标为 Ascend950PR_9579（DAV-3510、arch35）。A2/A3 验证
范围仅为旧 float32 Sgtsv2 与公共 C ABI，不包含本次 complex64 Kernel。
奇异系统不承诺返回特定数值错误码或有限解；独立记录返回状态和同步后输出。

# 可维可测分析

## 判定方法

| 项目 | 方法 |
| --- | --- |
| 精度 | 同一 complex64 输入升为 complex128，以 SciPy/LAPACK 为 Golden；实/虚部分别 `atol=rtol=2^-13`、匹配率≥0.99，并检查最大误差点 `max(1e-2,32*ULP(g_low))` 上限 |
| 特殊与资源语义 | 独立检查 NaN/Inf 分类与符号、空 RHS、参数拒绝、只读系数、B padding、guard、非默认 stream 和确定性 |
| 性能 | 固定 203 条，每例预热 10、采样 30；求和 prepare/factor/solve 耗时后取中位数，逐例 GPU/NPU≥0.3；恢复 B 排除计时，Event 单列 |
| 内存 | 分列 IO、查询/实际对齐 workspace、恢复副本、allocator 和设备快照，按适用条件检查；不混加不同统计范围 |

GPU 基线采用任务附件汇总 CSV，不重采。附件缺少原始 NCU、逐次采样和完整版本
证据，该范围说明保留。实装 CANN 9.1.0、torch_npu 2.7.1.post11、ATK 26.9.8；
torch_npu 与任务书“26.0.0 及之后”的编号缺少映射证明，不直接认定满足该下限。

## 当前实测结果与范围

| 项目 | 实测结果 |
| --- | --- |
| 五组 ATK | 发布 200/200、自测 800/800、特殊 8/8、合同 12/12、固定输入精度 203/203 |
| 原生 small / medium | small 10,752/10,752 通过；medium 5,376 中 4,918 通过、458 失败 |
| 其他功能 | 补充 21/21、bridge 1/1；A5 旧 float32 67/67，A2 两次 71/71，A3 71/71 及公共 C ABI |
| 固定 203 性能 | 203/203 达到 0.3，最低 0.3571667773263209 |
| 固定 203 workspace | 203/203 不超过 128 MiB L2，最大 9,863,296 字节 |
| 最大行数动态边界 | `(m,n,ldb)=(1048576,1,1048576)` 功能通过；workspace 264,241,280 字节，超过 L2 134,217,728 字节 |

medium 的 458 条原始失败中，222 条日志明确 CPU singular matrix 后缺 Golden，
236 条为 NaN 比较拒绝且无法判定 NaN 来自哪一侧，不将这些记录改为通过。
原生域仅执行 16,128/66,432 个 ID，50,304 未执行，不能声明全域精度通过。
固定 203 条无 IO 超过 500 MB 的适用例，该相对额外内存条件未经验证；
固定清单的 workspace 结论不能外推到全部动态形状。

逐例结果、失败与未执行清单、原始日志和复现步骤随私仓 `task_submission/`
的七件自验证材料交付。报告区分当前实现与历史记录，不合并为唯一 shape 数。

## 兼容性与交付状态

新增类型和符号独立于 Sgtsv2，保留旧 float ABI，arch35 实现不编入 A2/A3。
若其他平台先合入公共类型/接口，按上游实际版本协调声明并重新验证兼容性。
自测 800 条不替代社区独立验收。本 PR 仅提交设计，正式验收与后续代码 PR
按社区流程推进；上述自测结论不代表社区验收通过。
