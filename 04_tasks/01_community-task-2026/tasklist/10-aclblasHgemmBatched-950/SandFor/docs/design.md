# aclblasHgemmBatched 算子设计

## 1 需求背景

### 1.1 需求来源

根据社区任务书，在 ops-blas 公共 BLAS 接口中增加 FLOAT16
指针数组批量矩阵乘。实现目录为 `blas/gemm/arch35/`，使用 Ascend C kernel 直调与 handle 绑定的 stream。

设计结构参考[社区设计模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
代码基线为公开 ops-blas `3e138a2fb58cefa78720587391253171d53431d7`。

### 1.2 现状

该基线没有 `aclblasHgemmBatched` 或 `aclblasHalf` 声明。已有批量 GEMM 模板支持 FP16 输入和 FP32 累加，
其任务划分、GM/L1/L0 搬运、流水同步和输出可复用。新增实现复用该模板，使用同一个公共 API。

## 2 需求分析

对每个 batch `b`：

`C_b[i,j] = alpha * sum_p(op(A_b)[i,p] * op(B_b)[p,j]) + beta*C_b[i,j]`。

| 项目 | 定义 |
|---|---|
| 输入输出类型 | FLOAT16，标量与矩阵类型一致 |
| 矩阵存储 | 列主序，支持 lda/ldb/ldc padding |
| 指针数组 | Device 内存，各元素指向对应的 Device 矩阵 |
| 转置 | N/T/C 九种组合，实数 C 等价于 T |
| 标量 | alpha/beta 为 Host 指针，同一调用的所有 batch 共用 |
| 输出 | Carray 所指矩阵原地更新，有效区域外 padding 不变 |
| 广播与额外 stride | 不支持 |
| 运行时维度 | m/n/k/batchCount 在调用时传入 |

## 3 详细设计

### 3.1 公共接口

在 `include/cann_ops_blas.h` 中声明任务规定的参数序列；`aclblasHalf` 使用 uint16_t 保存 IEEE 754 binary16 位模式。
实现内部通过位复制与 `__fp16` 转换读取 Host 标量，避免把位模式整数当作浮点数值使用。

### 3.2 Host 侧设计

`hgemm_batched_host.cpp` 按以下顺序处理：

1. 校验 handle；校验标量地址、非负维度与 batchCount。
2. 校验转置枚举与前导维。
3. `batchCount>0` 时校验三组数组地址。
4. 对 m/n/batchCount 为零的情况返回成功，不访问设备内存。
5. 读取标量，确定直接输出、FP32 临时结果或缩放路径。
6. 查询 Cube/Vector 核数，集中生成 shape、布局、任务划分和 workspace 计划。
7. 在 handle 的 stream 上启动所需 kernel，保持异步调用语义。设备运行错误由调用方同步检查。

错误码沿用 `cann_ops_blas_common.h`。非法枚举返回 INVALID_ENUM；非法参数返回 INVALID_VALUE。
空数组校验先于零维短路；batchCount=0 时不校验数组地址。

### 3.3 列主序与任务划分

使用 `C^T=op(B)^T*op(A)^T`，交换输入数组、m/n、lda/ldb 和对应转置标记，以共享 row-major 模板计算。
一个任务对应一个 batch 的输出子矩阵；不同任务输出区域独立。任务总数使用 64 位乘法计算，超出 32 位任务索引范围时返回错误。
默认输出 tile 为 128×256。m<=128 时 K 搬运块为 512，其余为 256。
对 512<m=n=k<1024 的中等方阵，batchCount<32 时使用 128×128 tile、128 的 K 搬运块并按输出 tile 划分任务；
batchCount>=32 时按 128 行划分任务，每个任务连续处理该行块的输出列。任务边界按 tile 对齐，减少尾部任务的不均衡。
其余形状的空间任务数根据 batch 并行度确定，单任务可连续处理多个输出 tile，复用指针及布局计算。
执行任务数受设备核数限制，剩余任务由设备循环处理。

### 3.4 Kernel 侧设计

`hgemm_batched_kernel.cpp` 调用共享 `GbProcessMNTiles<FP16>`：

- FP16 数据从 GM 搬入 L1，转换为 NZ/ZN 布局。
- L1 和 L0A/L0B 双缓冲；FP32 L0C 累加。
- 通过现有事件同步复用各缓冲区；输出使用双 L0C 区域。
- N/T/C 归一化为四种实际布局组合，在 Host 确定对应入口。

| 路径 | 条件 | 行为 |
|---|---|---|
| 直接输出 | alpha=1、beta=0、k>0 | Fixpipe 将 FP32 累加结果转换为 FP16，写入 C |
| 一般标量 | k>0、alpha≠0，其余标量组合 | Cube 输出 FP32 临时结果，SIMT 执行 alpha/beta 运算并转为 FP16 |
| 缩放 | k=0 或 alpha=0 | 只执行 C=beta*C；beta=0 时写零，跳过旧 C 读取 |
| 保持 | 缩放路径且 beta=1 | Host 返回成功 |

紧凑 NN 方阵在边长为 8、16、32、64 且 alpha=1、beta=0 时使用固定形状实例化。
32×32 以 4 个 batch 为一组，分别纵向拼接第一个输入、横向拼接第二个输入，在 L1 中形成 128×32 和 32×128 矩阵。
乘积的第 i 个 32×32 对角块为第 i 个 batch 的结果。其余乘积不写回。
末组输入用最后一个有效 batch 填充内部剩余块，仅写实际 batch 的结果。
一次 Mmad 对应多次 Fixpipe，L0C 的复用统一使用显式事件同步，Mmad 与 Fixpipe 的 unitFlag 设为 0。

一般标量路径在 FP32 中完成运算，最后转换一次。beta=0 分支不读取原 C，因此原 C 可为 NaN 或未初始化内容。
k=0 路径不读取 A/B 的设备数组内容；数组地址仍遵守 Host 端非空约束。

### 3.5 Workspace 与并发

临时结果按 `batchCount*n*align_up(m,16)*sizeof(float)` 分配，由 handle 现有 workspace 管理入口负责。
本实现不缓存用户设备指针，不进行 Host 指针数组回拷，不新增全局可变状态。
同一 handle 的 workspace 按绑定 stream 顺序复用；不同 stream 并发调用使用不同 handle。
用户 workspace 不足时沿用现有错误码，库管理 workspace 遵循既有分配上限。

### 3.6 产品与兼容性

实现面向任务要求的 Ascend 950PR，代码使用 arch35。公共接口与其他产品线共用，不定义私有平行 API。
实现和验证结果记录在交付的自测报告、日志和 CSV 中。

## 4 可维护性与可测试性

新增生产文件为一个 Host 文件、一个 kernel 文件和一个共享计划头文件。Cube 计算复用既有实现；
测试侧独立生成输入和 Netlib golden，不读取生产 tiling 逻辑。
构建门控登记到已有 tensor_api 算子清单，不新增产品配置或运行时开关。

测试使用仓库 GTest/CSV 框架，逐例验证错误码，逐 batch 检查精度、特殊值和 padding。
结果检查拒绝零用例、缺失 batch、重复样本及缺失 GPU 基线。

## 5 精度与性能标准

精度采用任务书 FLOAT16 档：rtol=atol=2^-10、有限值 matched_ratio>=0.99，以及最大绝对误差限制。
CPU golden 为 Netlib 3.10.0 `cblas_sgemm`，结果按 round-to-nearest-even 转换为 FLOAT16。
NaN/Inf 单独计数并按任务书第 3.2 节判定；不通过放宽阈值处理精度差异。

性能使用 msprof op 的 task duration，5 次 warmup、每例 10 个有效样本；对 200 条任务性能用例计算
`GPU耗时/NPU耗时`，单例要求>=0.4，算术平均要求>=0.5。
任务书末尾另要求>=1.0；报告单独给出该更严格判据的结果，不将两种标准合并或省略。

## 6 验证与结果

已有设备记录包含 1142 条精度用例和 200 条性能用例。精度结果为 1142/1142；性能每例10个样本，
GPU/NPU 最低0.403898、算术平均1.172708，满足 §3.3 的两项阈值。按末尾 §7 的 >=1.0 表述，120/200 达到，
另外80例未达到。本设计不把两种口径混写为全部通过。

记录声明设备为 Ascend 950、golden为 Netlib BLAS 3.10.0；原材料未附精确CANN版本、driver或芯片型号命令输出。
任务书要求 Ascend 950PR/CANN 9.1.0，该精确环境证据须在正式验收材料中补齐。

完整复现方法见[测试说明](https://gitcode.com/SandFor/ops-blas/blob/6500563d68c55b866e38f7b84285a48814084063/test/gemm/hgemmBatched/README.md)，[待验收源码](https://gitcode.com/SandFor/ops-blas/commit/6500563d68c55b866e38f7b84285a48814084063)及逐例原始记录随验收材料提供。
配套图根据真实日志和CSV排版生成，未作为原始设备终端截图使用。
