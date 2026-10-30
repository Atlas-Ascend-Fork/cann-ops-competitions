# aclblasHgemmStridedBatched（Ascend 950PR）设计

## 需求背景（required）

### 需求来源

2026 年 10 月社区任务《aclblasHgemmStridedBatched_Atlas950PR_task_doc.md》。
代码目标仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，实现目录为
`blas/gemm/arch35/`，测试目录为 `test/gemm/hgemmStridedBatched/arch35/`。
开发团队为 Ask710。本文按社区 `resources/design_template.md` 组织。

### 背景介绍

补齐 Ascend 950PR 的半精度、固定步长批量 GEMM 能力。上游已有单精度 strided-batch 接口、
GEMM 的公共 handle/stream/workspace 管理及 FP16 batched GEMM Cube 模板，
可复用其硬件数据搬运、MMAD 和流水同步，新增公共半精度接口。
开发基线为 ops-blas `05b96ea22cea926ecfd43fc7ac691a795522bd61`，
交付前已同步上游 `e56c4cd6f41ea63712f1c5c123a596004ce2e574`。

## 需求分析（required）

### 需求描述

对每个 batch b，计算 `C_b = alpha * op(A_b) * op(B_b) + beta * C_b`。
A/B/C 和 alpha/beta 均为 FLOAT16；alpha/beta 为 Host 标量指针，矩阵为 Device 缓冲。
矩阵列主序，相邻 batch 通过非负 `long long` 元素步长定位。
N 为原矩阵，T 为转置，实数 C 与 T 等价。m/n/k/batchCount 均为运行时参数。

### 需求拆解

1. 在 `include/cann_ops_blas.h` 声明共用 `aclblasHgemmStridedBatched`，参数顺序与 cuBLAS 对齐；
   不增加 950PR 私有 API。`aclblasHalf` 复用 `aclFloat16` 存储类型。
2. 完整校验 handle、枚举、非负维度/批数、前导维、非负步长及标量指针。
3. m/n/batchCount 为 0 时合法 no-op；k=0 时只做 C 的 beta 缩放；beta=0 不读旧 C。
4. 支持紧凑和 padding 的列主序布局、输入步长重叠及零步长，C 原地更新。
5. Ascend C kernel 直调、使用 handle 绑定的 stream，复用同族 GEMM 模板。
6. 在 Ascend 950PR、CANN 9.1.0 上完成 CSV + GTest、Netlib BLAS 3.10.0 精度验证、
   全部 200 个性能 case 的 msprof op 测量及交付报告。

## 详细设计（required）

### 算子分析

#### 数学公式与形状

```text
C_b[r,c] = alpha * sum(p=0..k-1, op(A_b)[r,p] * op(B_b)[p,c]) + beta*C_b[r,c]
A_b = A + b*strideA; B_b = B + b*strideB; C_b = C + b*strideC
C_b[r,c] 位于 C[b*strideC + c*ldc + r]
```

| 缓冲 | N 时物理形状 | T/C 时物理形状 | 前导维最小值 |
| --- | --- | --- | --- |
| A | m×k | k×m | max(1, 物理行数) |
| B | k×n | n×k | max(1, 物理行数) |
| C | m×n | 不适用 | max(1, m) |

不提供维度广播、额外视图或任意非连续 Tensor；零输入步长仅表示各 batch 显式复用同一地址。

### 算子实现

#### Host 侧设计

校验顺序为：handle → transa/transb → m/n/k/batchCount → lda/ldb/ldc → strideA/B/C、
alpha/beta 判空 → no-op → 读取 Host 标量 → A/B/C 判空 → 地址范围检查 → tiling/下发。
非法枚举返回 INVALID_ENUM，空 handle 返回 HANDLE_IS_NULLPTR，其他非法参数返回 INVALID_VALUE。
例如 batchCount=0 且 strideA<0 仍报错；合法 no-op 允许 A/B/C 为空，不读取标量值。

非空输出且 k>0 时，不论 alpha 是否为零，A/B/C 均须非空。
k=0 时不引用 A/B，beta≠0 要求 C 非空；按任务 §2.4 的明确约定，k=0、beta=0、C 为空可返回成功。
alpha=0 通过指针校验后只执行 beta 缩放，beta=0 的所有分支均不读旧 C。

矩阵最大访问元素为 `(batchCount-1)*stride + (columns-1)*ld + rows-1`。
使用 64 位无符号算术并在乘加前检查可寻址范围，避免 long long 截断和字节范围溢出。

通用分块先按每维最多 256 确定块数，再将该维均分并向上对齐到 16，保持块数不变；
分块上限仍为 256×256。这样可避免少数核长期处理大块而其他核只处理很小的尾块。
例如 336 维度分成 176+160，而不是 256+80。K 的 L1 分块为 256，
L0 分块复用模板的 64。总任务数为 `batchCount*ceil(n/tileM)*ceil(m/tileN)`，
按核编号循环分配。每个 batch 内使用蛇形 N 遍历，尾块取真实尺寸。

若 C 的物理访问范围重叠（含 strideC=0），按 batch 递增在相同 stream 上串行提交，
前序 batch 的 FLOAT16 更新成为后序 batch 的输入。普通不重叠 batch 共用一次下发。

#### Kernel 侧设计

1. **通用 Cube**：利用 `C^T = op(B)^T * op(A)^T` 将列主序输出变为行主序运算，
   交换 A/B 的计算角色与 m/n，不交换公共 API 的入参含义。
   复用 `gemm_batched/arch35/gemm_batched_cube_impl.h` 的 FP16 搬运及 MMAD 模板。
2. **流水**：GM→L1、L1→L0A/L0B、MMAD→L0C、Fixpipe→GM。
   L1/L0A/L0B 双缓冲，使用 MTE1_MTE2、MTE2_MTE1、MTE1_M、M_MTE1 和 FIX_M/M_FIX 事件约束读写。
   L0C 为 FP32。tile 占用不超过 128 KiB 时在两个输出半区间交替；较大 tile 使用完整 bank 0，
   每块写回后等待 FIX_M 再复用，最大 256×256×4=256 KiB。
   每个 L1 bank 最大占用 (256+256)×256×2=256 KiB；L0A/B 每个半区各为 256×64×2=32 KiB。
3. **直接输出**：alpha=1、beta=0 时由 950PR Fixpipe 将 FP32 累加值直接转换为 FLOAT16 写 C，
   包含非 16 倍的 ldc/strideC；保留真实尾块尺寸，避免 GM 中间结果及第二次下发。
4. **通用标量输出**：Cube 写 FP32 workspace；AIV 逐元素计算 alpha*workspace + beta*C，
   最后一次 FP16 RNE 转换。仅写 m×n 逻辑元素，保留所有 padding 与间隙。
5. **小矩阵**：m=n=k=8 使用 UB 共享输入的 SIMT 路径；m*n<256 且 k<=32 的其余场景采用 FP32 SIMT 累加。
   紧凑 NN 的 16×16/32×32、直接输出场景将最多 8 个 batch 一次搬入 L1，
   每批仍独立完成 FP32 MMAD 和输出，减少小搬运的启动成本。
6. **退化缩放**：k=0 或 alpha=0 使用 AIV beta 缩放；beta=0 置零，beta=1 直接返回。
   非零 beta 直接保存乘积，不额外加正零，保留乘法产生的负零符号。

#### 内存、stream 和错误处理

所有矩阵首地址偏移均以元素为单位，不创建 Device 指针数组。
FP32 workspace 大小为 `align_up(m,16)*n*batchCount*sizeof(float)`，来自 handle。
沿用库的 2 GiB 上限和 `EnsureDefaultWorkspace`；先比较每批元素数与上限除以 batchCount，
再进行批数乘法，避免三因子乘法回绕；用户借用的 workspace 不足则返回 ALLOC_FAILED，
不覆盖其边界。必要时库 workspace 扩容按照已有句柄语义同步并重新分配。
kernel 在 handle 的 stream 上异步执行；调用者须在读回、释放或复用缓冲前同步。
不保证确定性计算或跨不同 stream 对同一 C 的并发写入；A/B 与 C 不得别名。

### 支持硬件

| 产品 | 支持情况 | 软件要求 |
| --- | --- | --- |
| Ascend 950PR | 支持 | CANN 9.1.0，arch35 / DAV_3510 |
| Atlas A2 / A3 | 本实现不支持 | 公共声明可供后续产品线实现复用 |

### 算子约束限制

不要求编译期 shape、动态 shape 框架注册、随机数生成或确定性计算。
运行时 m/n/k/batchCount 和所有 stride 均由参数指定，不使用 case 名或测试环境变量选择 kernel。
未满足快路径前提时使用通用模板，不改变接口语义。

## 可维可测分析

### 精度标准 / 性能标准

| 标准 | 判据 | 来源 |
| --- | --- | --- |
| 精度 | Netlib cblas_sgemm FP32 → FP16 RNE；每 batch 全矩阵检查；rtol=atol=2^-10，匹配率 ≥0.99，误差 ≤max(0.1,32×ULP) | 任务 §3.1、§3.2、§3.5 |
| 非有限值 | NaN/NaN、同号 Inf/Inf、NaN/Inf 等效；有限/非有限和异号 Inf 失败，非有限值全部通过 | 任务 §3.2.5 |
| 性能 | msprof op BasicInfo task duration，工具 warmup=5，10 个 SUCCESS 有效调用取算术平均；每 case GPU/NPU≥0.4，全部倍率算术平均≥0.5 | 任务 §3.3 |

浮点计算不要求 bit-exact；FP32 累加后统一输出 FP16。若大 k 未达精度标准，保留 CPU golden、
逐 batch 数值和随 k 增长的误差证据，不放宽阈值。性能不包含 Host 数据生成及 golden 时间。
多个 kernel 的调用按相同样本将 task duration 相加。

### 自验与证据

保留任务原始 1000 精度和 200 性能 CSV 及 GPU 基线。扩展覆盖转置全组合、
正态/均匀标量与矩阵、尺寸 16384、batchCount 65536、前导维/步长 padding、零/重叠步长、
所有 no-op 的标量判空顺序、k=0、非法枚举/负参数、64 位地址范围、Inf/NaN、workspace 不足及恢复。
性能 case 同样执行独立的精度测试，并单独进行 profiler 采集。

测试提供固定种子、独立 FP16 位转换、输入保持、C padding/间隙与缓冲前后保护区检查。
GTest JSON 的实际 case 集合必须等于请求 CSV 集合；缺样本、禁用/跳过、失败或无数值记录均失败。
性能分段记录全部 10 个原始样本、profiler 原始 CSV 路径和 SHA256，并核对全部 200 个基线条目。
不同采集进程的 capture index 独立编号，汇总使用 segment 身份一起定位。
同时记录可执行文件和实际加载的 libops_blas.so 哈希，防止混用不同编译版本。
自测报告包含参数、逐 batch 精度、逐 case 性能、真实日志渲染截图及可复现命令。

### 兼容性与维护

不修改现有 Sgemm/Cgemm 等 API 行为；新增公共声明及 FLOAT16 存储别名。
Host/kernel 实现保持在 GEMM 同族目录，无独立私有工程和额外运行时依赖。
CSV 与 GTest 按仓库现有 CMake 注册规则编译。
提交设计 PR 至 cann-competitions；测试通过后提交实现 PR 至 ops-blas，
按 CLA、`/compile`、代码审查流程完成合入。原始采样和 XLSX 报告作为任务验收附件交付。
