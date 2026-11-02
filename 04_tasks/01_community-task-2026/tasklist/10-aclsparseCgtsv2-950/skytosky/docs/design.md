# 【CANN社区任务】aclsparseCgtsv2 算子设计文档（950）

目标仓库：<https://gitcode.com/cann/ops-sparse>
目标实现目录：`sparse/cgtsv2/arch35/`、`test/cgtsv2/arch35/`
目标硬件：Ascend 950（DAV_3510，arch35）
贡献者：`skytosky`

# 需求背景

## 1.1 需求来源

本设计对应 2026 年 10 月社区任务“aclsparseCgtsv2 算子开发（950）”。任务要求在 ops-sparse 中新增复数三对角线性方程组求解接口，语义对标 `cusparseCgtsv2`，支持 complex64、部分主元、B 原地覆盖、workspace 查询和异步 stream 执行。

## 1.2 算子与现状

`Cgtsv2` 求解 `A·X=B`。`A` 由 `dl/d/du` 三条复数对角线表示，`B` 为列主序 RHS 矩阵，求解后原地保存 `X`。当前仓库只有实数 `aclsparseSgtsv2`，其 arch35 实现采用 AIV 入口、`asc_vf_call` 和每线程一个 RHS 的 SIMT 结构，可作为工程与算法参考。

本任务新增独立 `aclsparseComplex_t` ABI 和 arch35 Host/Kernel，不修改 Sgtsv2 声明或实现。公开类型必须为 `{float real; float imag;}`，不能用仓内已有的 `aclsparseComplex{x,y}` 替代。

# 需求分析

## 2.1 接口与参数

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

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| `handle` | aclsparse 上下文和 stream | handle | 不涉及 | 非空有效句柄 | 标量 |
| `m` | 三对角系统阶数 | int32 | int32 | `m>=3` | 标量 |
| `n` | RHS 列数，不是 batchCount | int32 | int32 | `n>=0`，`n=0` 跳过 | 标量 |
| `dl` | 下对角，只读 | Tensor | complex64 | `dl[0]=0` | `[m]` |
| `d` | 主对角，只读 | Tensor | complex64 | 有限值及规格允许的 Inf/NaN | `[m]` |
| `du` | 上对角，只读 | Tensor | complex64 | `du[m-1]=0` | `[m]` |
| `B` | RHS，原地输出 X | Tensor | complex64 | 列主序，padding 不变 | `[ldb,n]` |
| `ldb` | B 的 leading dimension | int32 | int32 | `ldb>=max(1,m)` | 标量 |
| `pBufferSizeInBytes` | workspace 查询输出 | size_t 指针 | size_t | 非空 | 标量 |
| `pBuffer` | 执行临时工作区 | 地址 | uint8 | 足量且 128B 对齐 | 连续字节 |

编译期断言保证 `sizeof(aclsparseComplex_t)==8`、`alignof(aclsparseComplex_t)==alignof(float)`。`n=0` 时查询返回 0、执行成功且不要求 workspace。Host 校验可见的整数约束、空指针、workspace 对齐、B span 及 workspace 乘法和对齐加法溢出；raw ABI 无法检测的实际分配长度、dtype 或布局错误由调用方负责。

## 2.2 功能与异常语义

算法必须支持部分主元，不能只实现无主元 Thomas。每一步按复数模长比较当前主元与相邻候选，必要时交换相邻方程及对应 RHS，并维护第一、第二上对角 fill-in。`dl/d/du` 只读，B 仅写有效区 `[0,m)`，`m<=i<ldb` 的 padding 保持不变。

非奇异输入在相同设备、软件版本和配置下确定性复跑。奇异系统不崩溃、不越界，记录实际返回码和 Inf/NaN 输出，不把非有限输出记作常规精度通过。Inf/NaN 使用独立特殊语义检查。

# 详细设计

## 3.1 数学模型与共享分解依据

对每个 RHS 列 `b`：

```text
dl[i]·x[i-1] + d[i]·x[i] + du[i]·x[i+1] = b[i]
```

无交换时：

```text
mult = dl[i] / d'[i-1]
d'[i] = d[i] - mult·du'[i-1]
b'[i] = b[i] - mult·b'[i-1]
```

交换时保存 `du2'[i-1]=du[i]`，并按交换后的两行更新主元、上对角和 RHS。回代：

```text
x[i] = (b'[i] - du'[i]·x[i+1] - du2'[i]·x[i+2]) / d'[i]
```

主元序列只依赖 `dl/d/du`，与 B 无关，因此可以分解一次供所有 RHS 复用。该性质用于后续优化候选，不影响 direct 正确基线的语义。

## 3.2 Host 与 Tiling 设计

Host 分为参数校验、workspace 计算和 launch 三部分：

1. `bufferSizeExt` 与执行共享同一 workspace 公式，查询和执行不会分叉。
2. 使用 `GetAivCoreCount()` 获取运行时 AIV 数，不写死不同 950 子型号的核数。
3. direct 路径按 `ceil(n/32)` 计算 block 数并 capped 到 AIV 数，每 block 32 线程，grid-stride 遍历 RHS。
4. direct workspace 为 `align128(3·m·n·sizeof(aclsparseComplex_t))`，保存每列的 `d'/du'/du2'`。
5. 共享分解候选 workspace 为 `align128((4·sizeof(aclsparseComplex_t)+1)·m)`，即 `align128(33·m)`，保存公共 `d/du/du2/mult` 和 pivot 标记，与 n 无关。
6. `n=0` 直接 quick return；其余路径的乘法、地址偏移均使用 64 位中间量并检查溢出。

direct 的大 workspace 仅用于建立正确性基线。最终是否切换到共享分解由功能、内存和性能实验决定。

## 3.3 Kernel 侧设计

### direct 正确基线

采用 AIV 入口和 `asc_vf_call` 的混合 SIMT 模式，不添加 `--enable-simt`。每个线程独立完成一个 RHS 的前向消元与回代，避免跨线程写冲突：

```text
for rhs = global_thread_id; rhs < n; rhs += grid_stride:
    forward_partial_pivot(dl, d, du, B[:, rhs], workspace[rhs])
    back_substitute(B[:, rhs], workspace[rhs])
```

模长计算使用最大分量归一：

```text
big = max(abs(real), abs(imag))
small = min(abs(real), abs(imag))
abs(z) = big·sqrtf(1 + (small/big)^2)
```

`big==0` 直接返回 0，Inf 显式保留，避免平方模溢出和同 Inf 产生 NaN。复数除法使用 Smith 算法，降低直接通分导致的中间溢出/下溢；零除数按 IEEE 语义产生 Inf/NaN，不增加奇异性错误码。

### 共享分解候选

当 direct 的重复分解或 workspace 成为瓶颈时，拆成两个 Kernel：

1. Factor Kernel：单核计算公共 LU 因子和 pivot 序列，写入四组复数数组及 pivot 标记。
2. Solve Kernel：多核沿 RHS 分布，复用公共因子完成前向 RHS 消元和回代。

两个 Kernel 在同一 stream 顺序执行，性能统计必须累计二者。n=1 时额外 launch 可能抵消收益，因此 direct 与共享路径按实测选择，不预设统一阈值。

## 3.4 内存与同步

direct 无需 UB 驻留整个矩阵，SIMT 线程直接访问 GM 和各自 workspace。共享分解只由 Factor Kernel 写公共因子，后续 Solve Kernel 只读，stream 顺序提供跨 Kernel 可见性，无需 GM 自旋或原子操作。

最终内存验收按任务书二选一：IO 超过 500 MB 时额外 NPU 内存不超过 GPU 总量 5%，或固有 workspace 不超过目标 L2。分别报告 IO、workspace 查询值、实际对齐分配、allocator 峰值和设备快照，不重复相加。

## 四、支持硬件与约束

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 / DAV_3510 / arch35 | √ |

约束：仅 complex64；`m>=3`、`n>=0`、`ldb>=max(1,m)`；B 为连续列主序；无独立 X、batchCount 或 CPU fallback；workspace 足量且 128B 对齐。公共算法与 A2/A3 解耦，公共头变化需完成交叉回归。

# 可维可测分析

## 5.1 精度与性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度 | 实部、虚部分别 `atol=rtol=2^-13`，匹配率≥0.99；最大有限误差满足 `max(1e-2,32·ULP(g_low))`；200+800 均通过 | 任务书 §3.2 |
| 特殊值 | NaN/Inf 分类 mask 与 Inf 符号一致，有限分量逐点满足混合容差 | 任务书 §3.2 |
| 性能 | 203 条逐条 GPU/NPU Kernel 总耗时中位数倍率≥0.3 | 任务书 §3.3 |
| 采样 | 每例≥10 warmup、≥30 samples；Event 独立列示，不替代 Kernel | 任务书 §3.3 |
| 内存 | 满足任务书二选一条件，记录 workspace、allocator 和设备快照 | 任务书 §3.4 |

CPU Golden 从同一 complex64 输入升为 complex128，使用带主元求解。离线 golden、仿真和单元测试不能替代真实 NPU 与官方 ATK 证据。

## 5.2 测试设计

1. C++ UT/ST：ABI 布局、非法参数、未对齐 workspace、溢出、n=0、单/多 RHS、ldb padding、纯虚数、首/末/连续 pivot、极端量级、奇异、确定性和系数只读。
2. 官方 ATK：随包 200、额外 800、特殊值、拒绝合同及可执行覆盖集合；注册真实的 `cgtsv2_buffer_size_ext_npu` 和 `cgtsv2_npu`。
3. 性能与内存：先测 n=1、最大 m、最大 workspace 和三条主场景，最终覆盖完整 203 条。每次调用内全部 Kernel 求和后再取中位数。
4. 证据：源码 commit/工作区快照、实际二进制和加载库 hash、原始日志、Profiler、NPU Dispatch 和复跑记录。

## 5.3 兼容性与风险

不修改 Sgtsv2 ABI 或实现，新增 `sparse/cgtsv2/` 与 `test/cgtsv2/`。主要风险包括复数主元边界、极端量级除法、n=1 串行链、共享分解额外 launch 和大 n workspace。分别通过特殊值/极端量级用例、direct 与共享 A/B、全量性能和内存采集控制。950 环境未恢复前只报告离线状态，不虚构构建、精度或性能结果。

## 六、实施与交付计划

1. 基线评估：恢复 950 环境，构建未修改 Sgtsv2/arch35，冻结执行身份与逐 case 目标。
2. direct 验证：构建 direct 候选，运行 C++ UT/ST、最小端到端和官方 ATK。
3. 诊断与优化：采集高风险 case，按 direct 的实测瓶颈决定是否采用共享分解，保留失败实验。
4. 全量验收：精度、203 条性能、内存、确定性、Profiler 与交叉回归全部完成后生成交付报告。
5. 交付内容：`include/cann_ops_sparse.h`、`sparse/cgtsv2/`、`test/cgtsv2/`、官方 ATK 适配、精度/性能/内存证据和私仓 `task_submission/` 七件套。
