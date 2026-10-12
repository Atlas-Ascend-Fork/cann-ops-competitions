# MatmulLayerNormMatmul 算子设计

## 需求背景（required）

### 需求来源

本设计对应 CANN 9 月社区任务 `09-45-MatmulLayerNormMatmul`，实现仓库为 [cann/catlass](https://gitcode.com/cann/catlass)，目标硬件为 Ascend 950PR。设计遵循社区任务书、[设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)及[本题讨论](https://gitcode.com/cann/catlass/discussions/22)中关于单算子完成全流程的说明。

### 背景介绍

算子完成两次矩阵乘及中间的行级 LayerNorm，仅输出最终矩阵 C1。设计通过一次 MIX kernel launch 组织 Cube 与 Vector 协作，以分组流水复用内部工作空间，减少独立算子调度与中间张量管理开销。均值、方差及其他中间量不作为接口输出。

## 需求分析（required）

### 需求描述

输入 A0、B0、B1 为 FP16，gamma、beta 为 FP32。先计算 A0 与 B0 的矩阵乘，再沿结果的列维度进行 LayerNorm，最后与 B1 相乘并输出 FP16 C1。epsilon 固定为 `1e-6`，方差采用总体方差。

### 需求拆解

1. 单次 kernel launch 完成 Matmul → LayerNorm → Matmul。
2. 按真实布局访问输入，仅分配最终输出及算子内部 workspace。
3. 保持 FP32 累加和统计精度，在第二次 Matmul 前保留归一化结果的高、低 FP16 分量，输出合并后再收窄。
4. 根据矩阵形状、基本块数量与尾波利用率选择 Matmul 分块及 Split-K 路径，不依赖测试编号或参考答案。
5. 按任务要求组织官方测试集、ATK 泛化精度、性能与内存检测，并为最终实现提供可复现的测试步骤。

## 详细设计（required）

### 算子分析

记 `M=M0, K=K0, N=N0, P=M1`，计算过程为：

```text
C = A0 @ B0
mean[m] = sum(C[m, :]) / N
variance[m] = sum((C[m, :] - mean[m]) ** 2) / N
Z[m, n] = (C[m, n] - mean[m]) / sqrt(variance[m] + 1e-6)
          * gamma[n] + beta[n]
C1 = cast_fp16(Z @ B1)
```

| 参数 | 方向 | 形状 | dtype | 布局 |
| --- | --- | --- | --- | --- |
| A0 | 输入 | (M,K) | FP16 | RowMajor |
| B0 | 输入 | (K,N) | FP16 | ColumnMajor |
| B1 | 输入 | (N,P) | FP16 | ColumnMajor |
| gamma | 输入 | (N,) | FP32 | 连续一维 |
| beta | 输入 | (N,) | FP32 | 连续一维 |
| C1 | 输出 | (M,P) | FP16 | RowMajor |

主接口按任务要求支持 ColumnMajor B0/B1，同时提供二者均为稠密 RowMajor 的兼容路径；不同布局使用实际 stride 寻址。不支持任意带洞 stride 或 B0/B1 混合布局。任务书未另设 shape 白名单，维度关系、整数溢出与资源分配检查在 Host 侧完成。

### 算子实现

#### Host 侧设计

测试接入接口为：

```python
torch_catlass.matmul_layer_norm_matmul(
    a0, b0, gamma, beta, b1, eps=1e-6, numChunks=2
)
```

Host 检查 rank、dtype、形状关系、布局、同一 NPU 设备、gamma/beta 长度及 epsilon；按输入设备使用当前 stream。`numChunks` 保留为兼容性调度提示，实际分组策略由形状确定，不用于关闭计算阶段。

输入指针保留 storage offset，不隐式复制为 contiguous。输出分配为 `(M,P)` FP16 连续张量。字节数和 workspace 大小采用宽整数计算，在转换到 kernel 参数类型前检查范围。空维情形按下文规则处理。

workspace 经 CATLASS/Torch 分配桥管理，其生命周期覆盖设备侧全部使用。调度端持有资源直到对应 stream 工作完成，不能以 Python 调用返回作为设备完成依据。JIT 缓存键包含 dtype、布局及实现 revision；影响生成代码的修改同步更新 revision。

#### 分核、分块与工作空间

Cube 使用 CATLASS BlockMmad，Vector 完成统计、归一化、高低分量生成和输出合并。基本配置为：

| 路径 | L1 tile (M,N,K) | L0 tile (M,N,K) |
| --- | --- | --- |
| 通用 | 128×128×256 | 128×128×64 |
| 大矩阵 | 256×256×256 | 256×256×64 |

大矩阵路径按基本块覆盖率、整波/尾波占用和 padding 比例选择。Split-K 仅用于归约维较长且基本块数或尾波不足的形状；结合核心数计算切分因子，避免过度切 K。

尾波 Split-K 使用 FP32 原子累加：Vector 先清零目标累加区，Cube 在全体同步后写入部分和，消费端等待全部部分和完成。该路径要求 FP32 输出行跨度满足块对齐；不满足条件时使用普通 Matmul 路径。原子累加次序可能变化，因此重复运行检查数值判据和稳定性，不把原子路径描述为保证逐位确定。

沿 M 维采用最大 1024 行的分组。M≤1024 时使用一个槽；M>1024 时使用两个槽交错复用。每个槽保存当前行组的 FP32 C、FP16 高低 Z 分量和 FP32 高低次乘结果。令：

```text
R = min(M, 1024)
align512(x) = 512 * ceil(x / 512)
slot_bytes = align512(4*R*N) + align512(4*R*N) + align512(8*R*P)
workspace_bytes = slot_bytes                if M <= 1024
                  2 * slot_bytes           otherwise
```

公式适用于非空输出且 N>0 的情形；M、N 或 P 为零时 workspace 为零。容量在两个行组后不再随 M 增长，但仍随 N、P 线性增长。例如 `(M,N,P)=(2048,8192,4096)` 对应 192 MiB 内部 workspace，不含输入和最终输出。

工作空间是内部 GM 暂存，不能等同于全片上驻留。大 M 复用有限行组，不完整保存三算子拼接的全部中间流水；小 M 的单组覆盖全部行。实现优化需同时减少重复读写并维持跨核可见性，不以省略必要同步换取计时结果。

#### Kernel 侧流程

1. **首乘**：读取 FP16 A0/B0，Cube 以 FP32 累加，写入当前槽的 C。
2. **统计与归一化**：Vector 先减去行内参考值，再计算残差均值和中心化方差，以降低大偏置、低方差数据的消减误差。对中心化结果再次去除 FP32 残余均值，再进行缩放、gamma 和 beta 运算。N≤8192 使用行驻留路径，更长行采用分段规约。
3. **高低分量**：计算 `hi=round_fp16(Z)`、`lo=round_fp16(Z-float(hi))`，按相邻逻辑行组织。第二次 Matmul 处理 `2R` 行，分别获得 `hi@B1` 和 `lo@B1` 的 FP32 部分和。
4. **输出**：Vector 在 FP32 中合并两个部分和，再收窄为 FP16 C1。对因中间收窄产生非有限值的输出点，使用同 kernel 的原值点积路径处理，保持特殊值语义。
5. **槽复用**：后一组首乘与前一组 LN 重叠；一组次乘与另一组 LN 重叠；输出合并与后续计算重叠。只有所有消费者完成后才覆盖槽内容。

#### 同步与生命周期

MTE 搬运、Vector 运算和 Cube 输出在交接前完成相应流水依赖。跨所有核生产/消费的数据使用全体同步；局部 AIC/AIV 配对信号不替代全局生产完成条件。所有核以一致次数进入共享屏障，尾块和空工作核也遵守协议。

两槽的状态依次为：可写 → 首乘完成 → LN 完成 → 次乘完成 → 输出消费完成 → 可复用。Split-K 另有清零完成与部分和合并完成条件。Host 不提前释放异步设备操作仍引用的输入、输出或 workspace。

#### 边界与特殊值

- M=0 或 P=0：返回相应形状的空输出。
- N=0：按空规约输出零矩阵。
- K=0：首乘结果为零，继续执行 LayerNorm 和次乘。
- 非对齐维度：按有效长度搬运并处理尾块，padding 不参与统计。
- 常量行或零方差：使用固定 epsilon；gamma=0、随机 gamma/beta、较大直流量与小扰动分别测试。
- NaN/Inf：测试位置、符号与传播语义，不能因普通随机精度通过而省略。

#### 代码组织

实现按 CATLASS 样例约定组织：

```text
experimental/matmul/matmul_layer_norm_matmul/
    CMakeLists.txt
    matmul_layer_norm_matmul.cpp
    README.md
    test_matmul_layer_norm_matmul.py
include/catlass/gemm/kernel/
    matmul_layer_norm_matmul_lookahead.hpp
tests/optest/
    kernels/、src/、模板与 Python 注册
    ATK 测试接入
```

公共组件保持 Block/Tile/Kernel 职责分离；诊断与临时调参脚本不作为算子交付源码。目录迁移后检查 JIT include 路径、注册入口和干净构建。

### 支持硬件

面向 Ascend 950PR，使用 CANN 9.1.0 和对应 CATLASS Ascend950 后端。核心数从平台接口获取，调度逻辑不将测试设备的核心数硬编码为接口约束。其他型号不在本题支持声明范围内。

### 算子约束限制

输入类型、布局和 epsilon 如参数表所示。资源容量受设备可分配空间限制；维度、stride 与字节数检查失败时返回明确错误。不通过按测试编号分派或静默截断维度规避边界。

## 可维可测分析

### 精度标准/性能标准

精度遵循[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。自验使用独立 FP64 公式计算参考，两个 Matmul 及 LayerNorm 中间结果不收窄。FP16 输出检查 `rtol=atol=2^-10`、匹配率不低于 0.99，并执行标准要求的逐元素绝对误差、ULP 和非有限值检查。golden 不调用被测 kernel，失败样本、seed 与异常均保留。

性能目标按任务定义为全测试集 `平均标杆时延 / 平均测试时延 > 1.1`，在同一集合中等价于 `sum(baseline_us) / sum(kernel_us) > 1.1`。逐例倍率用于定位，不替代整体指标。使用官方 CSV 的对应 shape 和 950PR 标杆，不混入其他设备结果。

性能通过 `msprof op` 采集设备侧 kernel 时延；记录预热、重复次数、实现分支、TileShape、软件版本及实际产物。Host 完整调用时延另记，不与 kernel 时延混用。采集缺项、异常或空结果不能按零时延计入。

### 测试设计

| 类别 | 覆盖与方法 |
| --- | --- |
| 官方测试集 | 全 116 个 shape，独立 golden、完整精度与性能记录 |
| ATK 泛化 | 至少 200 条，覆盖不同尺寸、尾块、数值范围与布局 |
| 行组与分支 | 1024 行前后、两槽及多次复用、大小 tile、Split-K 与普通路径 |
| 数值边界 | 零方差、大偏置小扰动、近零输出、NaN/Inf、空维度 |
| 接口负向 | dtype、rank、维度关系、stride、设备、epsilon 不匹配 |
| 资源与稳定性 | 重复初始化/调用/释放、输入保护区、异步资源寿命、分配峰值 |
| 内存与同步 | 设备内存工具检查越界、未初始化读取及竞争，保留原始告警 |
| 兼容性 | 两种稠密 B 布局、storage offset、当前 stream、JIT revision 更新 |

先按失败共同根因修复，再运行受影响分支与边界回归；最终在同一源码与产物上覆盖全部必过项。性能比较保持设备、输入集合、计时范围和预热条件可比；保存源码版本、产物哈希、原始日志与运行命令。

### 性能优化方案

围绕两次 GEMM、行统计和阶段交接分别测量，结合全流程时延选择优化方向。Matmul 使用大小 tile 与几何条件 Split-K；Vector 采用行驻留/长行分段；两槽流水减少阶段空泡。进一步变更必须计入量化、数据变换、额外矩阵乘、同步与输出合并的完整成本，同时保持独立精度判据。

### 兼容性分析

新增独立融合接口，不改变既有 Matmul、LayerNorm 或其他样例的语义。Torch/JIT 注册保持现有 ABI；新增 kernel revision 避免命中旧产物。公共接口输出只包含 C1，内部工作空间与调度策略不成为调用者依赖。

### 设计风险与控制

| 风险 | 控制措施 |
| --- | --- |
| FP16 中间收窄放大误差 | FP32 首乘/统计，高低分量补偿，独立高精度参考 |
| 跨核交接或槽复用竞态 | 明确生产/消费边界，全体同步与设备工具验证 |
| Split-K 过度切分 | 按归约长度、块数与尾波选择，完整计入清零和同步成本 |
| 大矩阵工作空间增长 | 分组复用、显式容量公式与分配检查 |
| 小矩阵调度开销占比高 | 减少冗余初始化、选择适当基本块，并用完整调用对照 |
| 缓存或产物身份漂移 | JIT revision、源码和二进制哈希绑定及干净构建复现 |
