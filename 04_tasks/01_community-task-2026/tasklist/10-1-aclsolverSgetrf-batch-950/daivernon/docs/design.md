# 需求背景（required）

## 需求来源

[单精度实数 LU 分解、求逆和批量求逆（950）社区任务](https://www.hiascend.com/activities/task-center/details/cd66edbecb8840b3bbcd39035146a970)，代码仓为 [ops-solver](https://gitcode.com/cann/ops-solver)。交付四个计算接口和两个 workspace 查询接口。

## 背景介绍

### LU 分解与求逆实现优化

采用仓库 Host C API + AscendC Kernel 直调方式，在 Ascend 950PR 上实现 FP32 稠密矩阵分解与求逆。

### 现有实现状况分析

原 `Sgetrf`、`Sgetri` 简化原型存在 Host 指针、`lda == n`、未写入 `info`、缺少 workspace 查询等限制，须按任务书升级。新增两个批量接口，复用既有 handle 和 stream 管理。

### LU 分解与求逆功能分析

| 接口（均带 `aclsolver` 前缀） | 输入与输出 |
| --- | --- |
| `Sgetrf_bufferSize` | 查询 LU 分解 workspace 元素数 |
| `Sgetrf` | 原始矩阵原地变为 LU 因子，输出主元与 `info` |
| `Sgetri_bufferSize` | 查询单矩阵求逆 workspace 元素数 |
| `Sgetri` | LU 因子及主元原地变为逆矩阵 |
| `SgetriBatched` | 各矩阵 LU 因子及主元，输出独立逆矩阵 |
| `SmatinvBatched` | 各原始小矩阵，输出独立逆矩阵 |

# 需求分析（required）

## 需求描述

接口签名、参数顺序与错误语义遵循任务书。矩阵为 FP32 列主序；维数、主元和 `info` 为 INT32。矩阵、主元、`info`、workspace、批量指针数组均在 Device；Host 仅传标量、handle，以及查询结果 `Lwork` 指针。`Lwork` 单位为 FP32 元素数。

## 需求拆解

1. 支持矩形 LU、可选主元、合法 leading dimension padding、空问题和逐矩阵错误报告。
2. 使用确定性部分主元；相同输入在同一 stream 串行执行时，结果及主元、`info` 按位一致。
3. 核心计算在 NPU 完成，走调用方 stream；完成全量正确性与逐点性能验收后统一交付。

# 详细设计（required）

## 算子分析

### 数学公式

LU 分解为 $PA=LU$，$L$ 为单位下三角，$U$ 为上三角；矩形情形采用相应梯形。原地保存两者，主元为 1-based 行交换序列。主元指针为空时不选主元，满足 $A=LU$。

LU 求逆解 $LY=P$、$UC=Y$，得到 $C=A^{-1}$。批量接口逐矩阵执行同一语义；`SmatinvBatched` 直接从原始矩阵求逆，奇异位置按 LU 主元次序报告。

### 支持数据类型

计算、存储均为 FP32，不以降精度换取吞吐。地址偏移使用足够宽的整数，检查 workspace 大小溢出。

### 支持形状

验收范围：LU 的 $m,n\in[1,4096]$；LU 求逆的 $n\in[1,4096]$；批量数覆盖 $[1,10^6]$；直接批量求逆限 $n\in[1,32]$。另支持维数或 batch 为零的空问题。列间距不小于有效行数，不支持广播。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

各接口保留 `src/sgetrf/`、`src/sgetri/`、`src/sgetri_batched/`、`src/smatinv_batched/` 下的 Host 文件；共享参数、workspace 与调度放在 `src/utils/real_lu.h`、`real_lu_host.cpp`。公开声明位于 `include/cann_ops_solver*.h`。

Host 校验 handle、维数、leading dimension、指针与 workspace；计算返回 `aclsolverStatus_t`。`info` 可写时，非法参数写 `-i`（不计 handle），首个零主元写正下标，成功写零。批量设备指针项及重叠关系由设备校验。

##### 1. 分核策略：

按独立矩阵或矩阵 tile 分配工作，核数取可用核心与有效工作量的较小值。批量采用步进分配；同一矩阵的校验、求逆、特殊数值回退保持同一 block 所有权，避免跨核隐含依赖。

##### 2. 数据分块和内存优化策略：

LU 使用 64 列面板和递归叶块；依据面板驻留空间终止递归。小矩阵复用线程私有寄存器；中等矩阵一次加载 LU、主元并在 UB 复用；大矩阵分块利用 FP32 Cube GEMM。动态 UB 仅申请当前算法所需空间，保留 SIMT DCache 容量。分块、线程与尾部处理由维数和资源约束确定，不依赖测试编号或具体样本。

##### 3. tilingkey规划策略：

不使用 TILING_KEY。复用三个真实 kernel 入口，少量参数分支选择计算阶段或资源域，不为局部测试点新增 kernel。

#### 3.2.2 kernel侧设计：

1. `src/utils/arch35/real_lu_kernel.cpp` 合并 `LuVectorKernel` 的面板分解、换行、三角求解、校验、变换和写回阶段，并提供一个 `RealGemmKernel`。LU 面板确定性选择绝对值最大、下标最小的主元，随后换行、解块行并由 Cube 更新尾矩阵。
2. `src/sgetri_batched/arch35/inverse.cpp` 只提供一个 `BatchedInverseKernel`：小 LU 使用 SIMT 协作求逆，中等 LU 在 UB 驻留，通过共享双三角求解得到逆矩阵。超过 128 阶由 Host 调度共享 Vector/Cube kernel，完成 64 列分块上三角求逆、下三角右解和主元置换。单矩阵与批量共用计算实现。
3. 直接批量求逆的微小矩阵复用寄存器 LU；其余小矩阵采用单寄存器行表示的 sweep Gauss–Jordan：虚拟行位置与输出列置换替代物理换行和增广单位阵。次正规数、异常除法等不安全数值域在同一 kernel 内回退 LU 路径；短子组采用补偿残差修正。共享算术与三角求解集中在 `src/utils/kernel/real/arithmetic.hpp`。
4. 最终设备调用链直接使用当前编译器 intrinsic，不使用 TPipe、TQue、TBuf、LocalTensor/GlobalTensor 或 SDK 计算 wrapper。同步按实际读写依赖设置，复用 buffer 前确认最后读取完成；不添加无关全流水屏障。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR，`dav-3510` | √ |

950 的 AIC/Cube 与 AIV/Vector 分离：矩阵更新走 GM→L1→L0A/B→L0C→GM，向量与 SIMT 工作使用 UB、寄存器及 SIMT DCache。线程数结合每线程寄存器预算选择；不能跨代套用 910 的通路或参数。编译与运行统一使用仓库 README 已验证的 CANN 9.0.0 发行包（构建编号 `20260422000325096`），已在 950 真机核验配套环境。

## 算子约束限制

批量输入输出为 Device 指针数组，各矩阵独立寻址，输出与对应输入不得重叠；批量输入及所有 padding 保持不变。`Sgetrf`、`Sgetri` 原地输出，后者输入必须是 LU 因子。`SmatinvBatched` 的 $n>32$ 返回非法参数，不转调其他接口。非法主元、奇异矩阵和空问题均明确报告；计算不得以 CPU fallback 替代，不作无必要的 Host 同步。

# 可维可测分析

## 精度标准/性能标准

令 $\epsilon=2^{-24}$，$\|M\|_1=\max_j\sum_i|M_{ij}|$。LU 先逆序回放主元将 $LU$ 还原为 $\widetilde A$；LU 求逆的参考 $A$ 从实际输入因子和主元还原。批量逐矩阵判定，任一失败即该用例失败；`ratio_cpu` 为同输入 FP32 CPU 参考链路按同式实测结果。

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| LU 精度 | $\|\widetilde A-A\|_1/(n\|A\|_1\epsilon)\le\max(5\,ratio_{cpu},0.3)$，矩形分母 $n$ 取列数 | 任务书 3.2.1.1 |
| LU 求逆精度 | $\|I-CA\|_1/(n\|A\|_1\|C\|_1\epsilon)\le\max(5\,ratio_{cpu},0.05)$ | 任务书 3.2.1.2–3 |
| 直接求逆精度 | 同一逆矩阵残差公式，阈值为 $\max(5\,ratio_{cpu},0.5)$，$A$ 为原始输入 | 任务书 3.2.1.4 |
| 性能 | 每个有 GPU 金标的用例满足 $T_{GPU}/T_{NPU}\ge0.35$，按任务规定的平均设备耗时及 msprof 流程关联比较 | 任务书 3.3 |

验证计划覆盖全部附件及任务书 3.5：精度主路径均匀/可变均值方差正态各 50%，对角加 $n$，另加主路径数量 10% 的奇异/近奇异用例；覆盖 $2^k$、$2^k-1$、最小 leading dimension 及 +8/+32、规定 batch 边界、无主元、非法参数、非默认 stream 和按位确定性。

附件共 712 个正确性点、710 个有效 GPU 性能点；两个缺失 GPU 数据的点仅跳过性能。最终结果须关联同一源码和库 hash；当前开发验证尚未全部完成，本设计不声明已全面达标。

## 兼容性分析

原 `Sgetrf`、`Sgetri` 简化原型按任务书升级，旧 ABI 调用方须按新头文件重新编译，并准备 Device 缓冲区和查询所得 workspace。复用现有 handle/stream；其他任务算子的接口与实现保持不变。四个计算接口与两个查询接口作为同一任务交付。
