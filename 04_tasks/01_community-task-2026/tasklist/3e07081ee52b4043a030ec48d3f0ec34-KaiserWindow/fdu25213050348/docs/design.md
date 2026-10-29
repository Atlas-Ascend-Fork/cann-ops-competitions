# KaiserWindow 算子设计文档

贡献账号：`fdu25213050348`。实现仓：`https://gitcode.com/fdu25213050348/sip`，开发分支：`feat/kaiser-window`。

文档状态：设计已与实现对齐。Ascend910B3（A2，`b5bfbe0`）、Ascend910_9382（A3，`a95d417`）与 Ascend950PR_9579（A5，`a95d417`）均已上板通过精度 1000/1000、接口 23/23、内存 66/66、Device 性能 200/200。三档硬件实测已完成，报告按 A2、A3、A5 独立归档。

任务依据：`KaiserWindow_task_doc(A2A3A5).md`，2026 年 10 月社区任务。

# 需求背景（required）

## 需求来源

为 SiP 增加 Ascend C KaiserWindow 算子，并在同仓 `sip_pta` 提供 `torch_sip.asd_kaiser_window` 接口。目标仓为 `cann/sip`，对标 PyTorch 2.7.1 的 `torch.kaiser_window`。

## 背景介绍

Kaiser 窗用于有限冲激响应滤波器设计、频谱分析和信号分帧。窗序列由长度、周期性和形状参数唯一确定。将索引生成、坐标归一化、平方根、零阶修正贝塞尔函数及归一化融合到一个 Ascend C kernel，减少逐元素算子下发及中间 GM 张量。

### 现状分析

任务书要求新增 SiP 原生实现，采用 SiP kernel 直调工程和 `sip_pta` 扩展。实现包含 ACL 入口、Operation、Kernel、tiling、A2/A3 与 A5 设备代码、Torch 注册和 Python 封装。

### 功能分析

| 参数 | 含义 | 类型 | 支持范围与约束 |
| --- | --- | --- | --- |
| `outTensor` | 独立输出 | `aclTensor*` | 一维连续 ND，shape 为 `[windowLength]`；FLOAT16、BFLOAT16、FLOAT32 |
| `windowLength` | 窗长度 | host `int64_t` | 0 至 \(2^{31}-1\) |
| `periodic` | 周期窗开关 | host `bool` | 默认 true；false 生成对称窗 |
| `beta` | 窗形状参数 | host `double` | 默认 12.0；常规验收域为 [0, 88]，另覆盖负值、NaN、正负无穷 |
| `stream` | 当前 NPU stream | `void*` | 由调用者传入 |
| `workspace` | 工作区 | `void*` | 工作区大小恒为 0 |

输出张量没有输入张量依赖；本算子没有广播、原地修改或输入别名。

# 需求分析（required）

## 需求描述

使用 Ascend C 开发 SiP `asdKaiserWindow`，支持 Atlas A2、Atlas A3、Ascend 950（A5），并交付 `torch_sip.asd_kaiser_window`。工程遵循 SiP 的 Operation/Kernel 注册和直接调用约定。

## 需求拆解

1. 实现 FLOAT16、BFLOAT16、FLOAT32 输出，核内计算域为 float32。
2. 实现周期窗、对称窗以及空窗、单元素窗、负 beta 和非有限 beta 边界。
3. 新增 C 接口，校验输出 dtype、形状、连续性和长度，返回 SiP 状态码。
4. 新增 Torch C++ 注册与 Python 接口，支持默认 dtype/device、layout 和 requires_grad。
5. 在 A2/A3 的 arch22 路径和 A5 的 arch35 RegBase/MicroAPI 路径分别实现和编译。
6. 完成全量 1000 条精度、200 条性能、Torch 接口、确定性和内存自验证，保留原始日志。

# 详细设计（required）

## 算子分析

### 数学公式

长度记为 \(L\)，索引为 \(i=0,\ldots,L-1\)。当 \(L\ge2\) 时，周期窗取 \(N=L\)，对称窗取 \(N=L-1\)。

$$
r_i=\frac{2i-N}{N},\qquad t_i=\sqrt{|1-r_i^2|},\qquad
w_i=\frac{I_0(\beta t_i)}{I_0(\beta)}.
$$

采用偶函数性质，有限 beta 的计算使用 \(b=|\beta|\)。令 \(I_{0e}(x)=e^{-|x|}I_0(x)\)，计算形式为：

$$
w_i=\frac{I_{0e}(bt_i)}{I_{0e}(b)}\exp\bigl(b(t_i-1)\bigr).
$$

`I0e` 使用与 ATen `calc_i0` 同源的 Chebyshev 系数。小值段使用变换 \(x/2-2\)，大值段使用变换 \(32/x-2\)，并乘以 \(1/\sqrt{x}\)。Cephes 归属和许可证随设备源码保留。

910B 在 \(x\le5\) 使用 \(I_0(x)=\sum_{k\ge0}(x^2/4)^k/(k!)^2\) 的 9 次 Horner 多项式，再乘以 \(e^{-x}\)；\(\beta\le1\) 时使用 4 次多项式。9 次展开在 [0,5] 的余项相对误差小于 \(3\times10^{-7}\)。在 \(5<x\le8\) 使用 13 项低段 Chebyshev，在 \(x>8\) 使用 5 项高段 Chebyshev。相对完整系数，低段省略项绝对值之和小于 \(5.9\times10^{-6}\)，高段小于 \(2.7\times10^{-8}\)。

910B 将保留的 Chebyshev 多项式转换成相同的幂基多项式，以 Horner 形式求值，减少每项向量指令数。转换脚本随源码提供。根据 tile 两端坐标计算自变量上下界，整块落入同一分段时只求该段；跨界 tile 按每个元素的 x 在 5 和 8 处选择，保持同一元素与 tile 划分无关。分母每 core 计算一次。

A5 保留全部 18 项低段、8 项高段 Cephes 系数，将同一 Chebyshev 多项式转换成幂基 Horner 形式；生成器随交付材料提供于 `task_submission/scripts/generate_a5_horner.py`。低段在 \(\min(x,8)\) 求值，高段在 \(\max(x,8)\) 求值。两段乘积中的公共 \(I0e(8)\) 因子在分子、分母之间约去。`denLow`、`denHigh` 在每 core 的 tile 循环前计算一次，保存在两个 64 元素 UB 槽中；装配 VF 每 tile 求一次分母倒数，随后各向量组复用该倒数进行乘法归一化。

### 边界处理

| 场景 | 行为 |
| --- | --- |
| `L == 0` | 校验完成后返回成功；输出为空，不启动计算 kernel |
| `L == 1` | 输出 1，与 periodic、beta 无关，包括 NaN 和无穷 |
| `L >= 2, beta == 0` | 输出全 1 |
| `L >= 2, beta < 0` | 与绝对值 beta 相同 |
| `L >= 2, beta == NaN/±inf` | 输出全 NaN |
| 任意端点 | 先取 `abs(1-r*r)` 再开方 |

### 支持数据类型与形状

输出为 `[L]` 的连续 FLOAT16、BFLOAT16 或 FLOAT32 张量。FLOAT16 和 BFLOAT16 在 float32 域完成计算后，按目标格式收窄。FLOAT32 直接写出。相同参数重复执行应逐位一致。

## 算子实现

### 实现方案

#### Host 侧设计

**ACL 接口：**

```cpp
extern "C" {
AsdSip::AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);
AsdSip::AspbStatus asdKaiserWindow(aclTensor *outTensor, int64_t windowLength,
                                bool periodic, double beta,
                                void *stream, void *workspace);
}
```

`GetWorkspaceSize` 检查输出指针并写入 0。执行入口检查 `outTensor`、长度范围、dtype、维度、shape 与连续性，随后组装 `OpParam::KaiserWindow` 和 `OpDesc`，由 SiP 运行时下发。形状与参数错误使用仓内对应无效参数返回码，不支持的 dtype 使用对应不支持数据类型返回码。beta 保留 host double 输入语义，不以取值限制拒绝调用。

`windowLength == 0` 在校验后直接成功返回。其余长度将输出地址、长度、periodic、beta 和切分参数交给 Kernel。`InferShapeImpl` 的输出 shape 为 `[windowLength]`；`GetBestKernel` 根据平台选择 `KaiserWindow910BKernel` 或 `KaiserWindowArch35Kernel`。接口声明并导出到 `base_api.h`，在 `configs/op_list.yaml` 分别开启 ascend910b 与 ascend950。

**分核与 tiling：**

- 从平台取得 AIV 核数，以每 core 至少 512 元素选择实际核数。
- 每 core 的范围按 64 元素对齐，尾部仅写有效元素。
- 910B tile 为 5120 元素，使用 7 个 float32 计算 tile、一个 int32 基础索引 tile、640 字节选择掩码和一个输出 tile；FLOAT32 输出时总 UB 为 184960 字节，FLOAT16/BFLOAT16 为 174720 字节。基础索引每 core 初始化一次，后续 tile 仅执行向量整数偏移计算。
- A5 tile 为 8192 元素，使用 4 个 float32 tile（t、x、低段、高段）、两个 64 元素分母槽及一个输出 tile；scratch 为 131584 字节。FLOAT32 输出 tile 为 32768 字节，总 UB 为 164352 字节；FLOAT16/BFLOAT16 输出 tile 为 16384 字节，总 UB 为 147968 字节。
- 索引分子先在安全整型域精确计算，再转换为 float32。可采用 int64 的 `2*i-N`，或数值范围已证明安全的 `i-(N-i)`；覆盖最大合法长度。
- tiling 保存长度、实际核数、每核跨度、输出 dtype、特殊值分支、beta 和 periodic 产生的 N，共 28 字节。
- 工作区大小恒为 0；中间量只保存在 UB/寄存器。

#### Kernel 侧设计

核函数流程为 `Init → Process → CopyOut`，无 GM 输入搬入。

1. 根据 core id 计算本核索引区间，用不可变基础索引生成每个 tile 的全局整数索引。
2. 计算安全整数分子，转换 float32 后除以 N，得到归一化坐标。
3. 计算 \(t_i=\sqrt{|1-r_i^2|}\) 及 \(x_i=bt_i\)。
4. 910B 按自变量范围选择幂级数或 Chebyshev 分段，A5 使用端点饱和的两段乘积；随后计算缩放指数比值。
5. 将结果按目标 dtype 收窄，使用有效元素数处理尾块后写回 GM。

短路值分支在通用数值流程之前执行：L=1 写 1，beta=0 写 1，非有限 beta 写 NaN。最后一个 tile 的计算空间按向量对齐分配，GM 写出长度严格使用有效元素数。

**A2/A3 路径：** Ascend910B/arch22 AIV 路径，使用 Ascend C 向量 API、LocalTensor 和输出队列/缓冲组织计算。中间量在 UB 中复用。

A3 实测设备为 Ascend910_9382，SiP CMake 内核目标保持 `ascend910b`；CANN 原生基础算子的运行期安装 `Ascend-cann-A3-ops` 配套包。A3 验证环境为 CANN `9.3.0~weekly.20261021.01`、Python `3.12.13`、torch `2.7.1+cpu` 与 torch_npu `2.7.1.post6`。

**A5 路径：** Ascend950/arch35 路径，使用 `AscendC::Reg` 寄存器向量计算和 RNE 转换。设备入口为 `kaiser_window_arch35.asc`，由仓内 ASC 驱动针对 dav-3510 编译。坐标、低段、高段和装配分别组织 VF，阶段之间经 UB 和 `PipeBarrier<PIPE_V>()` 连接。最终实现 `a95d417a141695e5ad369cb81bc2ec8615f1ac5a` 已在 Ascend950PR_9579、CANN `9.3.0~weekly.20261021.01` 上完成构建和全量板上验证。

两条路径采用同一数学公式、同源系数与边界契约，各自编译和上板采集数据。

#### Torch 封装

Python 接口为：

```python
def asd_kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    ...
```

Python 层规范化默认值与 device。dtype=None 固定取 torch.float32；device=None 取当前 NPU；layout=None 规范化为 torch.strided。C++ 侧通过 NPUGuard 绑定设备，`at::empty` 分配输出，创建 ACL tensor，查询 0 字节 workspace 后调用 SiP 接口。输出的 requires_grad 在计算完成后的叶子张量上设置。

通过 `TORCH_LIBRARY_FRAGMENT(torch_sip)` 声明 schema，在 `PrivateUse1` 与 `BackendSelect` 注册同一 NPU 工厂实现，使没有 Tensor 入参的 `torch.ops.torch_sip.asd_kaiser_window` 裸调可用。`OpCommand` 回调捕获输出张量以保持异步下发生命周期，回调内创建 ACL 描述符、调用 SiP 并通过 RAII 释放描述符。

#### 工程文件

```text
core/base/kaiser_window.cpp
include/base_api.h
ops/base/CMakeLists.txt
ops/base/kaiser_window/CMakeLists.txt
ops/base/kaiser_window/kaiser_window_operation.cpp
ops/base/kaiser_window/kaiser_window_kernel.cpp
ops/base/kaiser_window/tiling/
ops/base/kaiser_window/op_kernel/
ops/include/params/kaiser_window.h
configs/op_list.yaml
sip_pta/csrc/base/kaiser_window.cpp
sip_pta/torch_sip/__init__.py
sip_pta/test/base/kaiser_window.py
```

以上为当前实现路径。

## 支持硬件

| 芯片/产品 | 编译路径 | 计划支持 | 自验证记录 |
| --- | --- | --- | --- |
| Atlas A2 | ascend910b / arch22 | 是 | 910B3，`b5bfbe0`：精度 1000/1000、接口 23/23、内存 66/66、Device 性能 200/200 |
| Atlas A3 | ascend910b / arch22 | 是 | Ascend910_9382，`a95d417`：精度 1000/1000、接口 23/23、内存 66/66、Device 性能 200/200 |
| Ascend 950（A5） | ascend950 / arch35 | 是 | Ascend950PR_9579，`a95d417`：精度 1000/1000、接口 23/23、内存 66/66、Device 性能 200/200 |

A2 实测代码保持为 `b5bfbe0b2accd9d2f326502b7b2accb61828aa33`，最小 A100/NPU 倍率为 1.15334501143068，中位数为 5.77385985671099。A5 实测代码为 `a95d417a141695e5ad369cb81bc2ec8615f1ac5a`，最小 H100/NPU 倍率为 0.7130361246861131，中位数为 12.251917871836044；200 条均满足 A5 的 ratio≥0.4。原始 Event 结果分别为 A2 13/200、A5 188/200 通过，按各自计时口径保留。

A3 使用 `a95d417a141695e5ad369cb81bc2ec8615f1ac5a` 完成独立实测，最小 A100/NPU 倍率为 1.3068183620829141，中位数为 5.7897560897840945，Device 200/200 均满足 ratio≥0.8。原始 Event 结果为 105/200 通过。A3 的报告与原始证据归档于 `task_submission/A3/`。

## 算子约束限制

`windowLength` 范围为 0 至 \(2^{31}-1\)；输出是一维连续 ND 张量，支持 FLOAT16、BFLOAT16、FLOAT32。Python 支持 NPU 设备及 strided layout。kernel 不申请 workspace，输出以外的临时数据保存在片上。常规数值验收 beta 范围为 [0,88]，边界测试覆盖负值及 NaN/±inf。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判据 | 来源 |
| --- | --- | --- |
| FLOAT16 精度 | atol=rtol=\(2^{-10}\)，matched_ratio≥0.99，最大绝对误差≤max(0.1,32 ULP) | 任务书 §3.2 |
| BFLOAT16 精度 | atol=rtol=\(2^{-7}\)，matched_ratio≥0.99，最大绝对误差≤max(0.1,32 ULP) | 任务书 §3.2 |
| FLOAT32 精度 | atol=rtol=\(2^{-13}\)，matched_ratio≥0.99，最大绝对误差≤max(0.01,32 ULP) | 任务书 §3.2 |
| A2 性能 | 200 条分别满足 \(T_{A100}/T_{NPU}\ge0.8\) | 任务书 §3.3 |
| A3 性能 | 200 条分别满足 \(T_{A100}/T_{NPU}\ge0.8\) | 任务书 §3.3 |
| A5 性能 | 200 条分别满足 \(T_{H100}/T_{NPU}\ge0.4\) | 任务书 §3.3 |
| 内存 | workspace=0，无中间 GM 张量；输出字节数为 L×sizeof(dtype) | 任务书 §3.4 |
| 确定性 | 同参数两次输出逐位相等，NaN 按对应位置处理 | 任务书 §2.5 |

golden 使用 CPU `torch.kaiser_window(dtype=torch.float64)` 在线计算，并按目标 dtype 收窄。空窗、单元素窗和全 NaN 场景按独立语义判定。全 NaN 边界不计入常规精度通过率分母。

### 测试覆盖

原始 CSV 共 1200 条：1000 条精度和 200 条性能。精度按 dtype 分为 FLOAT16 334、BFLOAT16 333、FLOAT32 333；按 section 分为 TC_L0 3、TC_L1 12、TC_L2 630、TC_L3 83、TC_L4 118、TC_L5 154。覆盖空窗、单元素、周期/对称、端点、非对齐尾块、大规模、零/负/非有限 beta。

验证顺序为：编译 → C 接口和 Python 导出检查 → 小用例 → 全量精度 → Torch 契约 → 确定性/内存 → 原始性能脚本 → Device kernel 性能采集 → 报告归档。A2、A3、A5 按硬件分别记录，不把其他型号的结果填写为本型号结果。

### 性能采集

保留任务提供 `verify_performance.py` 的原始结果；另外使用 torch_npu profiler 的 `kernel_details.csv` 记录 Device kernel 的持续时间。每条 case 在同一 profiler 会话中先运行 10 次 warmup，再运行 60 次正式采集；原始 CSV 保留全部 70 次。manifest 固定单 stream 的用例顺序，并核对总数为 \(200\times70=14000\) 条，再逐组取后 60 次的算术平均值。

任务提供脚本的 Event 包围 Python 调用循环，区间包含 Host 下发间隙。报告分别列出原始 Event 结果与 Device kernel 结果，不混用两种耗时。Device 结果保留每条的 60 个原始耗时及 profiler 来源。

随包 A100 基线 200 条 note 均为 warmup=10 collect=60。H100 基线 note 从 collect=60 逐行增加到 259，与任务书固定采集 60 次不一致；保留原始数值与文件，将此元数据差异写入报告来源说明。

### 交付件

按照官方 `task_submission.zip` 模板生成自验证步骤、三类 XLSX 报告及原始日志。报告中记录具体硬件型号、CANN、Torch、torch_npu、torch_sip、SiP commit、测试命令及执行结果。模板的示例日志和示例用例应由本算子实测内容替换。

已采集真实 Windows Terminal 截图 27 张，A2、A3、A5 每档各 9 张，覆盖精度 2 页、Torch 接口 3 页、Device 性能 2 页及内存 2 页。截图在本地可见终端中展示已经保存的板上记录，分别标注 original-log 与 derived-summary，并保留源码版本、输入来源和页码。源 PNG 和来源清单位于交付树公共 `task_submission/screenshots/`；真实 PNG 已嵌入三档 9 份 XLSX 的“终端截图”工作表，截图栏已填写。源图、工作簿和来源清单已随完整交付材料打包，并推送个人分支 `feat/kaiser-window`。

## 兼容性分析

本算子为新增 SiP C 接口和新增 torch_sip 接口。现有接口行为不变。新增代码依赖 CANN 9.3.0，Torch 与 torch_npu 均为 2.7.1，asdsip 与 torch_sip 从同一 SiP 源码构建。默认 dtype 固定为任务要求的 float32，默认 device 为当前 NPU。

## 参考资料

- 任务：[10月社区任务-torch_npu支持kaiser_window接口](https://www.hiascend.com/activities/task-center/details/3e07081ee52b4043a030ec48d3f0ec34)。
- 任务测试指导：任务附件 `test_cases/readme.md`。
- SiP 算子接入：`https://gitcode.com/cann/sip/blob/master/docs/developing_a_simple_operator.md`。
- SiP PTA 扩展：`https://gitcode.com/cann/sip/blob/master/sip_pta/README.md`。
- 设计模板：`https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md`。
- 社区流程：`https://gitcode.com/org/cann/discussions/39`。
