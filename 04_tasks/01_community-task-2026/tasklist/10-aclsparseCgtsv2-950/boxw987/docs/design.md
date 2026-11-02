# aclsparseCgtsv2 算子设计文档（Ascend 950）

> 设计文档提交位置：
> `04_tasks/01_community-task-2026/tasklist/10-aclsparseCgtsv2-950/boxw987/docs/design.md`。
> 任务书和测试包以工作区 `official/` 中保存的原文件为准。

# 需求背景（required）

## 需求来源

本设计对应 CANN 2026 年 10 月社区任务「aclsparseCgtsv2 算子开发（950）」：

- 任务 ID：`3c8d06bce4f54557b895de27e3737e0f`。
- 任务书：`official/task-release-20261028/10月社区任务-aclsparseCgtsv2算子开发 (950)/aclsparseCgtsv2_A5_task_doc.md`。
- 任务包 SHA-256：`a57f0cae816c11f241607a73b1a59f3fd67520c82847529ccf4f6e539834a618`。
- 任务书 SHA-256：`27dd819b9d24e660dd2aea915643e2e1222ce041f2a317b2f2ff79803052fe56`。
- 目标仓库：<https://gitcode.com/cann/ops-sparse>，目标分支 `master`。

任务要求在 Ascend 950（DAV_3510，`arch35`，A5）上采用 C++、Ascend C
和 ops-sparse Legacy C++ API 实现 complex64 三对角方程组求解。仓库中已有的
`aclsparseSgtsv2(float*)` 只作为句柄、stream、构建和部分主元结构的参考；本任务
新增独立复数 ABI，不修改既有 Sgtsv2 声明或实现。

## 背景介绍

`aclsparseCgtsv2` 求解由 `dl/d/du` 表示的三对角矩阵 `A` 与多个右端项 `B`：

```text
A[i,i]   = d[i]
A[i,i-1] = dl[i]       (1 <= i < m)
A[i,i+1] = du[i]       (0 <= i < m-1)
```

`B` 按列主序存储，地址为 `B[i + j * ldb]`，求解结果原地覆盖 `B` 的有效
`[m,n]` 区域，`m <= i < ldb` 的 padding 保持不变。`n` 是 RHS 列数而不是
batchCount；`n=0` 是合法空操作，不启动求解 Kernel。

本任务不能使用无主元 Thomas 算法代替要求的算法。前向消元必须按复数模长比较
当前主元与相邻下对角候选值，必要时交换相邻行并同步更新所有 RHS；行交换产生的
第二上对角填充项必须保留到回代阶段。

# 需求分析（required）

## 需求描述

新增以下公开接口和实现：

1. `aclsparseComplex_t`：连续的 `float real` 与 `float imag`，大小 8 字节且与
   `float` 具有相同对齐要求。
2. `aclsparseCgtsv2_bufferSizeExt`：查询执行所需 workspace 字节数。
3. `aclsparseCgtsv2`：按查询值使用 128 字节对齐 workspace，异步在当前 handle
   的 stream 上求解并将结果写回 `B`。
4. `include/cann_ops_sparse.h`、`sparse/cgtsv2/`、`sparse/cgtsv2/arch35/`、
   `test/cgtsv2/` 和 `test/cgtsv2/arch35/` 的工程接入。

## 需求拆解

1. **ABI 和参数合同**：复数类型、函数签名、handle/stream、整数范围、空指针和
   workspace 对齐错误码明确且可通过 C++ UT 检查。
2. **数值算法**：复数部分主元消元、行交换、`du2` 第二上对角填充和上三角回代；
   系数只读，每个 RHS 独立更新。
3. **布局与边界**：列主序 `[ldb,n]`、原地 `B`、padding 不变、`n=0` 返回零
   workspace、奇异和非有限输入不崩溃且单独报告。
4. **精度**：CPU Golden 使用同一份已量化 complex64 输入升为 complex128 后的带主元
   带状求解；实部和虚部按任务书的 `rtol=atol=2^-13`、匹配率和 ULP 规则分别比较。
5. **性能与内存**：官方 203 条性能用例全部运行；每条使用不少于 10 次 warmup 和
   30 次正式采样，以求解范围内全部 Kernel 总耗时中位数计算倍率；同时报告查询
   workspace、128 字节对齐实际分配、输入输出和恢复副本内存。
6. **交付**：随包 200 条精度、203 条性能和原生命令保留在 review 交付面；额外
   800 条精度、完整组合和隐藏验收输入只按其真实来源声明，不能用本地生成结果冒充。

# 详细设计（required）

## 算子分析

### 数学公式和接口

```cpp
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;

aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

公共头文件加入：

```cpp
static_assert(sizeof(aclsparseComplex_t) == 8);
static_assert(alignof(aclsparseComplex_t) == alignof(float));
```

该类型与仓库已有的 `aclsparseComplex {float x, float y}` 保持独立；Cgtsv2
只接受 `aclsparseComplex_t*`，不能通过类型转换复用旧 ABI。`dl/d/du` 的长度均为
`m`，首个 `dl[0]` 和末个 `du[m-1]` 的边界零值由调用方按任务合同提供。

### 支持类型、布局和异常合同

| 参数 | 合法合同 | Host 可检查的异常 |
| --- | --- | --- |
| `handle` | 已创建的 aclsparse handle | 空句柄返回 handle 空指针错误 |
| `m` | `m >= 3`，运行时 int32 | 小于 3 或尺寸乘法溢出返回参数错误 |
| `n` | `n >= 0`，是 RHS 列数 | 小于 0 返回参数错误；`n=0` 返回成功且 workspace 为 0 |
| `dl/d/du` | complex64、长度 `m`、连续只读 | `n>0` 时空指针返回参数错误 |
| `B` | complex64、列主序 `[ldb,n]`、原地输出 | `n>0` 时空指针返回参数错误；padding 不写入 |
| `ldb` | `ldb >= max(1,m)` | 不满足返回参数错误 |
| `pBufferSizeInBytes` | 非空输出指针 | 空指针返回参数错误 |
| `pBuffer` | 查询值足量且地址 128 字节对齐 | `n>0` 且空指针或未对齐返回参数错误 |

raw C ABI 无法读取调用方实际分配长度、dtype、stride 或 shape 元数据；这些属于
调用方和 ATK 适配层的责任。Host 只校验公开可见的整数、指针、对齐和可计算的
size_t 溢出边界。Inf、NaN 和真正奇异系统不进入常规有限精度判定，按独立特殊值
和异常合同记录返回码及输出分类。

## 实现方案

### Host 侧设计

Host 文件按仓库约定拆成 `ValidateCgtsv2Params`、
`ComputeCgtsv2WorkspaceBytes` 和 `LaunchCgtsv2Kernel` 三类职责，API 入口只做
handle 检查、`n=0` 快速路径和调用转发。调用顺序为：

1. `bufferSizeExt` 先检查输出指针、`n`、`m`、`ldb` 和 `dl/d/du` 可见指针，计算
   workspace；`n=0` 在形状和系数检查前返回 0 workspace，符合任务书空 RHS 合同。
2. 对 `n>0`，按 `3 * m * n * sizeof(aclsparseComplex_t)` 计算三个复数工作数组，
   再向上取整到 128 字节。所有乘法使用 `size_t` 分步检查，不能溢出后再对齐。
3. 执行接口重复检查可见参数和 workspace 地址，获取 handle 的 stream，构造只含
   `m/n/ldb`、核数、每列 workspace stride 的 tiling data，然后调用 `cgtsv2_kernel_do`。
   Host 不同步 stream、不申请独立 X、不修改输入系数。
4. 由于每个 RHS 只写自己的 `B[j*ldb : j*ldb+m]`，列间可按 Vector Core 数做
   grid-stride 分配。不能整除时，前面的 block 多处理一列；`B` 的 padding 不参加
   读写。具体 block 上限和 32B 访问策略在 SDK 探针后以实测编译约束为准，并绑定
   到候选 commit 的 tiling 记录。

### Workspace 布局

```text
per_rhs = [d_prime(m), du_prime(m), du2_prime(m)]  // complex64
rhs_stride = align_up(3 * m * sizeof(aclsparseComplex_t), 128)
workspace[j * rhs_stride + ...]                  // j = 0..n-1
```

`d_prime`、`du_prime` 和 `du2_prime` 保存前向消元后的主对角、第一上对角和第二
上对角；`B` 保存消元后的 RHS 及最终解。workspace 在多次性能采样中复用，报告
查询字节数和实际对齐分配字节数。若后续 SDK 探针证明整列因子不能满足 950 的
设备限制，再以独立 commit 设计分块 GM 路径，并重新绑定所有受影响证据；不在
没有实测依据时宣称隐含的硬件容量。

### Kernel 侧设计

Kernel 采用 arch35 SIMT 结构，沿用仓内 Sgtsv2 的入口和线程调度习惯，但所有
算术和 workspace 元素改为 `aclsparseComplex_t`。每个 SIMT thread 处理一个 RHS
列，系数从 GM 只读，`B` 只写本列。

复数运算使用显式内联辅助函数：

```text
mul(a,b) = (a.real*b.real - a.imag*b.imag,
            a.real*b.imag + a.imag*b.real)
div(a,b) = a * conjugate(b) / (b.real*b.real + b.imag*b.imag)
abs2(a)  = a.real*a.real + a.imag*a.imag
```

除法和比较保持 float32 设备语义；`abs2` 只用于比较，不调用开方。前向第 `i`
步（`i=1..m-1`）维护上一行的 `pivot_d/pivot_du/pivot_b`、上一行交换留下的
`du2_prev` 和当前行的原始 `cur_d/cur_du/cur_b`：

```text
if abs2(pivot_d) >= abs2(dl[i]):
    mult      = dl[i] / pivot_d
    d'[i]      = cur_d - mult * pivot_du
    du'[i]     = cur_du - mult * du2_prev
    du2'[i-1]  = 0
    b'[i]      = cur_b - mult * pivot_b
else:
    mult       = pivot_d / dl[i]
    d'[i-1]    = dl[i]
    du'[i-1]   = cur_d
    du2'[i-1]  = cur_du
    d'[i]      = pivot_du - mult * cur_d
    du'[i]     = du2_prev - mult * cur_du
    b'[i-1]    = cur_b
    b'[i]      = pivot_b - mult * cur_b
```

`du2_prev` 初始为零；每一步结束后，它取当前行产生的第二上对角值，供下一步
更新 `du'[i+1]`。这条 carry 不能省略：连续发生相邻行交换时，上一轮的第二上
对角会成为当前行的第一上对角输入。CPU Golden 已用连续 pivot 用例覆盖该递推。

交换分支把原上对角保存为 `du2'[i-1]`，因此回代按下式处理：

```text
x[m-1] = b'[m-1] / d'[m-1]
x[i]   = (b'[i] - du'[i]*x[i+1] - du2'[i]*x[i+2]) / d'[i]
```

其中越界的 `x[i+2]` 项按 0 处理。实现必须保持 `dl/d/du` 的 GM 内容不变，
只把修改后的因子写入 workspace；回代结果覆盖 `B[i+j*ldb]` 的有效行。对真正
零主元，Kernel 保持有界的固定循环和合法地址访问，输出/状态交由异常套件记录，
不能以崩溃或越界作为“奇异系统行为”。

### 工程接入

新增文件按仓库命名规范放置：

```text
include/cann_ops_sparse.h
sparse/cgtsv2/README.md
sparse/cgtsv2/arch35/cgtsv2_host.cpp
sparse/cgtsv2/arch35/cgtsv2_kernel.cpp
sparse/cgtsv2/arch35/cgtsv2_kernel.h
sparse/cgtsv2/arch35/cgtsv2_tiling_data.h
test/cgtsv2/CMakeLists.txt
test/cgtsv2/cgtsv2_golden.h
test/cgtsv2/cgtsv2_param.h
test/cgtsv2/arch35/cgtsv2_npu_wrapper.h
test/cgtsv2/arch35/cgtsv2_test.cpp
test/cgtsv2/arch35/cgtsv2_test.csv
```

`sparse/CMakeLists.txt` 通过现有 arch35 GLOB 规则接入；测试使用现有 test/frame
和 GTest/CSV 驱动，不新增重复 build 目录。源码不引入 TBE、ops_transformer 私有
实现或 CPU fallback。

## 支持硬件

| 芯片 | 目录 | 本设计 |
| --- | --- | --- |
| Ascend 950（DAV_3510，A5） | `sparse/cgtsv2/arch35/` | 支持 |
| A2/A3 | 不在本任务新增路径 | 不实现，不修改既有 Sgtsv2 |

## 算子约束限制

1. 仅支持本任务新增的 complex64 ABI；不接受旧 `aclsparseComplex` 类型混用。
2. `m>=3`、`n>=0`、`ldb>=max(1,m)`；`n=0` 无计算和 workspace。
3. 系数和 B 按约定连续列主序存储；不支持广播、转置、任意 stride 或独立 X。
4. workspace 由调用方按查询值分配，地址必须 128 字节对齐；算子不负责释放。
5. API 异步返回，调用方在读取 B 前同步所属 stream。
6. 奇异和非有限输入只按独立异常语义报告，不能作为常规有限精度通过。

# 可维可测分析

## 精度标准/性能标准

| 类别 | 验收标准 | 证据边界 |
| --- | --- | --- |
| 普通精度 | 实部、虚部分别按 `|actual-golden| <= 2^-13 + 2^-13*|golden|`，匹配率至少 0.99；最大有限误差满足任务书 ULP 约束 | NPU 完整精度运行，不能用 CPU 模型替代 |
| 特殊值 | NaN/Inf 分类 mask 和 Inf 符号一致；其余有限分量按任务书容差 | 独立 8 条特殊值套件 |
| 合同/边界 | 非法 m/n/ldb、空指针、workspace 对齐、n=0、只读系数、padding、确定性、奇异不崩溃 | C++ UT/ST 与 NPU 运行记录 |
| 性能 | 203 条倍率全部不低于 0.3；10 次 warmup、30 次正式采样；主口径是 Kernel 总耗时中位数 | Profiler 原始日志、逐例 CSV 和解析器 |
| 内存 | 记录 IO、查询 workspace、对齐分配、B 恢复副本、allocator 与 free/total 快照 | 203 条性能内存字段，不能把 smoke 当完整结果 |

## 测试设计

### CPU 参考

测试侧实现独立 complex128 带状部分主元求解器。它从同一份已量化 complex64
输入读取 `dl/d/du/B`，只在参考层升为 complex128；不调用待测实现、不读取 NPU
输出生成 Golden。参考实现记录主元交换路径、奇异分类和确定性复跑结果，便于定位
Kernel 中的 `du2` 或复数除法问题。

### C++ UT/ST

使用仓库 `test/frame` 的 GTest/CSV、ACL 环境 RAII 和现有 Legacy 测试约定，至少
覆盖：

- `aclsparseComplex_t` 大小、对齐和 ABI 声明；
- `bufferSizeExt -> 128B 对齐分配 -> 执行 -> stream 同步 -> 释放` 完整流程；
- `m/n/ldb` 边界、空 handle、空指针、未对齐 workspace 和 `n=0`；
- 单 RHS、多 RHS、`ldb>m` padding、系数只读、需要 pivot、零主对角非奇异、
  病态和真正奇异系统；
- 同一输入恢复原始 B 后重复执行，比较有限输出确定性；
- 既有 Sgtsv2 的回归入口和 CMake 架构选择。

### 官方 ATK 和性能

官方任务包中的 200 条精度、12 条合同、8 条特殊值和 203 条性能清单作为
organizer-provided 输入原样保留。测试 README 提供官方 ATK 的生成、执行、覆盖审计
和结果解析命令；不以自写 JSON/Tensor 生成器替代 ATK。额外 800 条精度输入标记为
organizer-owned/reviewer-injected，交付 review 分支的复现命令和结果位置，但不在
本地生成结果中冒充官方验收。

性能采集分别保存 Kernel 总耗时和同 stream Event 耗时，Kernel 倍率只使用前者；
每条 case 恢复原始 B 后再开始计时，workspace 在 warmup 和正式采样中复用。所有
203 条必须有独立 case ID、30 个正式样本、内存字段和 PASS/FAIL 状态，部分运行
只能标为 smoke 或 partial。

## 兼容性与风险

- 新增符号与旧 `aclsparseComplex` 类型分离；公共头文件改动需检查所有既有 C++
  调用方和 A2/A3 后续合入冲突。
- 复数部分主元和 `du2` 是主要正确性风险；先用 CPU 参考和小 m 的 pivot 用例锁定，
  再扩展到完整 200 条和 203 条。
- 真实 SDK/A5 编译、设备可用性、性能倍率和 L2 约束在当前 intake 阶段均未验证；
  这些只能由绑定具体 source commit 的 SDK/NPU 证据结论化。
- 若编译器或大尺寸设备路径要求分块 workspace，必须在 develop 上建立单独实验、
  重新运行受影响的精度/性能套件，再晋级 submission；不能用 smoke 结果替换完整结果。

## 迭代计划和门禁

1. 完成 intake preflight、需求矩阵和候选 scope prepare 审计（当前已通过）。
2. 设计文档按已核对的 `10-aclsparseCgtsv2-950` 目录提交；PR diff 只含
   `docs/design.md`，提交后在评论区一次性艾特规定 reviewer，修订 commit 再次艾特。
3. 设计合入后，在 `develop/aclsparse-cgtsv2-950-20261030` 实现公共 ABI、Host、
   arch35 Kernel 和 C++ UT；每个改动绑定矩阵需求和 focused evidence。
4. 通过 SDK 编译、CPU/合同/精度和性能完整套件后，冻结
   `submission/aclsparse-cgtsv2-950-20261030`，从同一提交派生
   `review/aclsparse-cgtsv2-950-20261030`，加入验收脚本和完整交付材料。
5. 官方测试审核通过后才打开代码 PR；代码 PR 反馈继续走 develop → submission →
   review，交付包中的完整结果始终保留原始 source/environment/command 身份。
