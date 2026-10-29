# 需求背景（required）

## 需求来源

2026 年 10 月社区任务：[aclblasChpr 算子开发（950）](https://www.hiascend.com/activities/task-center/details/f783b3800ad744ef8af7f50ba237de04)，任务编号 `f783b3800ad744ef8af7f50ba237de04`。目标仓库为 CANN ops-blas，基准提交为 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。

## 背景介绍

主仓已有 arch22 的 Chpr 和 arch35 的实数 Sspr。本实现补齐 `blas/hpr/arch35/`，复用公开 `aclblasChpr` 声明，不新增产品专用平行接口。

# 需求分析（required）

## 需求描述

在 Ascend 950PR、CANN 9.1.0 环境实现 complex64 Hermitian packed 秩 1 更新：`A ← A + alpha*x*xᴴ`。alpha 为 Host float32 实数，x 与 ap 位于 Device。支持上、下三角，正负非零步长；拒绝 INT_MIN 步长。按 handle 绑定的 stream 异步执行。

## 需求拆解

1. 公开 ABI 保持不变，补齐参数校验、quick return 和异步下发。
2. 实现 packed 索引、复数共轭乘加及对角虚部清零。
3. 连续输入优先使用 SIMD；非连续输入采用支持 64 位索引的并行通用路径。
4. 适配任务 CSV，按指定精度阈值执行逐用例验证并采集设备性能。

# 详细设计（required）

## 算子分析

### 数学公式

令 `x_i = r_i + i*s_i`，列 j 的系数为 `t_r=alpha*r_j`、`t_i=-alpha*s_j`。

- 非对角实部更新：`A_re(i,j) += r_i*t_r - s_i*t_i`。
- 非对角虚部更新：`A_im(i,j) += r_i*t_i + s_i*t_r`。
- 对角只保留更新后的实部，虚部显式写为正零。
- 按 Netlib 语义，x_j 为复数零时跳过该列数值更新，但非 no-op 调用仍将对角虚部置零。

乘积与加减分开求值，保留 FP32 舍入边界。非有限值按任务规则比对，不使用改宽的误差阈值。

### 数据与索引

- 上三角列首：`j*(j+1)/2`，列内行 i 的偏移为 i。
- 下三角列首：`j*(2*n-j+1)/2`，列内行 i 的偏移为 i-j。
- n(n+1)/2 个 complex64 元素构成 ap，全程不展开稠密矩阵。
- 负步长的逻辑元素 i 对应物理元素 `(n-1-i)*abs(incx)`。
- packed 偏移和向量物理偏移使用 64 位运算。

## 算子实现

### Host 侧设计

`ValidateChprParams` 负责枚举、步长和指针校验，`LaunchChprKernel` 负责分核和下发。

校验顺序为 handle → n<0 → n=0 直接成功 → uplo → incx → alpha/x/ap 非空 → alpha=0 直接成功 → kernel。n=0 不访问 alpha/x/ap；n>0 的 alpha=0 仍须先验证指针。

通过仓内 `GetAivCoreCount()` 动态获取 AIV 核数，核数为零返回错误；实际核数为 `min(n, coreNum)`。Tiling 仅包含 n、实际核数、uplo、incx 和 alpha，不使用核间分配数组，不申请 GM workspace，不同步调用方 stream。

### Kernel 侧设计

连续路径使用 RegBase SIMD。每个 AIV core 循环处理 `coreId + k*usedCoreNum` 列，分散上下三角长短列的负载差异。单次最多处理 4096 个复数元素；x 和 ap 两个 UB buffer 共 64 KiB。TPipe 在 kernel 入口创建，再传给初始化函数。

每个 tile 通过精确字节数的 DataCopyPad 搬入，使用成对反交织加载取得实部、虚部，分别执行乘加和对角掩码，再交织存储并写回。尾部仅写实际元素，MTE3 完成后才复用 ap buffer，避免 DMA 流水竞争。

任意带符号步长可能超出无符号 DMA stride 字段且无法在固定 UB 内整体暂存。通用路径使用 SIMT，按 core 循环分列、按线程并行处理列内行，避免逐元素 DMA 串行搬运。上、下三角在编译期分派，索引采用 64 位。

### 工程变更

- 新增 `blas/hpr/arch35/chpr_{host,kernel}.cpp`、`chpr_kernel.h`、`chpr_tiling_data.h`。
- 新增 `test/hpr/chpr/arch35/` 测试、NPU wrapper 和原始 1200 条 CSV。
- 共享参数解析支持 x_fill/ap_fill 和旧 x/ap 列名，识别 null/NULLPTR/nullptr，不删除 arch22 支持。
- 更新 Hpr README 的 950PR 支持状态、约束和可运行 RAII 样例。

## 支持硬件

| 硬件 | 本次验证 |
| --- | --- |
| Ascend 950PR | 已完成 |
| Atlas A2/A3 | 保留已有 arch22 源码，本次未上板回归 |
| Ascend 950DT | 未验证 |

## 算子约束限制

调用方须提供满足物理长度要求的合法 Device 内存。alpha 为 Host 指针；不涉及框架 Tensor 广播、反向传播或额外非连续视图。

# 可维可测分析

## 精度与性能标准

实部、虚部分别按 rtol=atol=2^-13 比对，有限值匹配比例至少 0.99；最大绝对误差不超过 `max(1e-2, 32*ULP)`。非有限值单独按任务规则比较，不计入有限值比例分母。alpha=0 采用逐字节不变检查。

测试 wrapper 在 x/ap 前后放置 canary，并检查 x 的全部字节不变。无效参数在分配长度计算前处理，避免 abs(INT_MIN) 或负尺寸触发分配错误。

CSV 字段和用例与任务包一致，仅规范化 CRLF 为 LF，共 1000 条精度用例和 200 条性能用例。另有空 handle、平衡输入分布、对角虚部忽略和 4097 阶多 tile 边界检查，共 1204 个 GTest 项。平衡分布组覆盖 CSV 中全部 439 个有效且 n>0 的不同 (uplo,n,incx) 组合，每组分别使用均匀和正态分布独立生成 alpha、x 和 AP，共 878 个场景、各占 50%；AP 对角虚部为零。均匀范围 [-5,5]，正态均值在 [-5,5]、标准差在 [0.1,2] 抽取。另保留 18 个旧对角虚部 NaN 场景，均匀/正态各 9 个。每组记录唯一标识、输入参数、分布参数、种子和实部/虚部结果，详细记录见自验证日志。

性能每条进行 1 次正确性调用、5 次预热及 12 次有效采样。仅对 ChprMeasure 范围使用 msprof op BasicInfo / kernel replay，8 批共 2400 个有效设备 Task Duration 样本，逐用例要求 `T_GPU/T_NPU ≥ 0.4`。GPU 基线来自任务包。首次正确性结果在重放前同步拷回并按原阈值校验。

**采集器兼容修复：** 当前旧 msopscommon 将容器逻辑设备0错误转换为宿主物理设备7，导致采集接口报逻辑编号非法。采用与官方较新源码一致的逻辑编号转换，通过采集子进程专用兼容库修复。保留修复源码、官方差异和原始日志。分批采样控制旧工具附带插桩转储的磁盘占用，未替换算子或计时 API。

## 兼容性分析

公开接口不变；arch22 内核未修改。参数解析增加新列名和 null 拼写兼容，保留旧格式。950 实现不影响其他算子家族。
