# KaiserWindow 算子设计（A2/A3/950）

## 需求背景（required）

### 需求来源

[10 月社区任务：torch_npu 支持 kaiser_window 接口](https://www.hiascend.com/activities/task-center/details/3e07081ee52b4043a030ec48d3f0ec34)。按照任务书，在 [SiP](https://gitcode.com/cann/sip) 中实现 Ascend C kernel 和 sip_pta 封装，对外提供 `torch_sip.asd_kaiser_window`。

### 背景介绍

Kaiser 窗用于信号分析与滤波，由长度、周期属性和形状参数确定。融合生成索引、窗函数计算和类型转换，可以减少逐元素算子组合的下发次数及中间张量。实现以 PyTorch 2.7.1 的接口语义和任务书规定的支持范围为准。

## 需求分析（required）

### 接口定义

SiP 侧在 `AsdSip` 命名空间内提供 C 链接符号，保持显式 stream、workspace 的调用形式：

```cpp
namespace AsdSip {
extern "C" {
AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);
AspbStatus asdKaiserWindow(aclTensor *outTensor,
                          int64_t windowLength,
                          bool periodic,
                          double beta,
                          void *stream,
                          void *workspace);
}
}
```

`asdKaiserWindowGetWorkspaceSize` 成功时写入 0。调用方创建输出并传入当前 NPU stream；workspace 可以为空。SiP 接口只写入输出，不分配输出张量。

Python 接口及 `torch.ops.torch_sip.asd_kaiser_window` 注册入口采用以下参数语义：

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

| 参数 | 约束与行为 |
| --- | --- |
| `outTensor` | 一维连续 ND 张量，shape 为 `[windowLength]`，支持 FLOAT16、BFLOAT16、FLOAT32。 |
| `windowLength` / `window_length` | 非负整数，范围 `[0, 2^31-1]`。长度为 0 时返回空张量且不启动计算 kernel。 |
| `periodic` | 默认 True；True 为周期窗，False 为对称窗，可按位置或关键字传入。 |
| `beta` | host double，默认 12.0；正常精度域为 `[0,88]`，另覆盖负值、NaN、±inf；不作取值拒绝。 |
| `dtype` | None 固定按 torch.float32 处理，仅支持上述三种浮点类型。 |
| `layout` | torch.strided；None 按默认值处理。 |
| `device` | None 使用当前 NPU；显式设备仅接受 npu，尊重设备序号。 |
| `requires_grad` | 默认 False；True 时返回要求梯度的叶子张量，不定义输入反向公式。 |

SiP 入口检查输出指针、维度、长度、连续性与 dtype；非法长度、shape 或连续性返回 `ACL_ERROR_INVALID_PARAM`，不支持的 dtype 返回 `ACL_ERROR_UNSUPPORTED_DATA_TYPE`。Torch 封装用 `TORCH_CHECK` 报告非法参数，不静默转到 CPU。FLOAT64、整数、布尔、复数、稀疏布局不在支持范围。算子不涉及广播、输入视图、输入原地更新或随机性，同参数重复执行要求逐位一致。

### 目标环境与交付范围

依据[任务讨论帖的环境更正](https://gitcode.com/cann/sip/discussions/4)，使用 CANN 9.2.0、torch 2.7.1、torch_npu 2.7.1，以及与 asdsip 同源构建的 torch_sip。Atlas A2、Atlas A3、Ascend 950 分别测试；A2/A3 使用 SiP 的 ascend910b/arch22 AIV 路径，950 使用 ascend950/arch35 RegBase 路径。kernel 与 Torch 封装均提交至 SiP。

## 详细设计（required）

### 数学公式与边界

长度记为 L。当 L≥2 时：

```text
N = L                  if periodic else L - 1
b = abs(beta)
q[i] = 2*i - N         （先在整型域内精确计算）
r[i] = float32(q[i]) / float32(N)
t[i] = sqrt(abs(1 - r[i]*r[i]))
x[i] = b * t[i]
y[i] = i0e(x[i]) / i0e(b) * exp(b * (t[i] - 1))
i0e(x) = exp(-abs(x)) * I0(x)
```

`I0` 是第一类零阶修正贝塞尔函数。使用 `b=abs(beta)` 使缩放形式保持 I0 的偶对称性；分子、分母均不恢复为可能溢出的 I0。`q[i]` 的计算采用足够宽的整型中间值或无溢出的等价整数式，不先把全局索引转成低精度浮点再相减。开方前取绝对值，不将舍入造成的微小负值送入 Sqrt。

| 情形 | 输出 |
| --- | --- |
| L=0 | 空张量；workspaceSize=0，不下发计算 kernel。 |
| L=1 | `[1]`，优先于 beta 的任何特殊值处理。 |
| L≥2，beta=0 | 全 1。 |
| L≥2，beta<0 | 与相应正 beta 相同。 |
| L≥2，beta 为 NaN 或 ±inf | 全 NaN。 |
| L=2，对称窗 / 周期窗 | `[1/I0(b),1/I0(b)]` / `[1/I0(b),1]`。 |
| L=3，对称窗 | `[1/I0(b),1,1/I0(b)]`。 |

周期窗以 N=L 计算，只写 L 个元素，不分配 L+1 的中间张量。偶数长度对称窗不强制存在值为 1 的采样点。

### Host 侧设计

SiP 入口校验后构造 `KaiserWindowOperation` 与参数，通过 `RunAsdOpsV2` 下发。参数包含长度、periodic、beta 与输出类型；Operation/Kernel 按平台选择实现。空输出在入口返回，其余情况区分常量填充与一般计算。

Tiling 根据设备向量核数、UB 容量、输出 dtype 和长度，把一维输出划成互不重叠的连续区间；短窗减少参与核数，长窗多核均分。每核循环处理固定上限的 tile，末尾按有效元素数写回。核边界满足搬运对齐，尾块使用精确长度搬运，禁止跨输出边界写入。长度和地址偏移计算避免整型乘法溢出。

所有算子中间量位于 UB 或寄存器，workspace 恒为 0。tiling 参数随框架下发，不使用输出等长的索引或临时 GM 张量。

### Kernel 侧设计

1. **生成坐标。** 在片上构造 tile 的整数等差索引，计算精确分子 q，再转换为 FLOAT32；计算 r、t、x。FLOAT16/BFLOAT16 只在最终写出前收窄。
2. **计算缩放贝塞尔函数。** 参考 PyTorch/Cephes 的分段 Chebyshev 形式，在 `[0,8]` 和 `(8,+∞)` 分段求值；高段包含 `1/sqrt(x)` 因子。用递推求值减少中间存储。按实际区间选择计算路径，跨区间 tile 用掩码合并，未选分支也保持安全的除数与开方参数。
3. **复用标量。** 同一 tile 的 `i0e(b)` 只计算一次，复用倒数；不逐元素重复分母计算。一般路径用缩放公式组合输出，避免两个大数的直接比值。
4. **写回。** 按目标 dtype 转换并写出有效区域。无输入搬入、核间归约或原子操作；需要流水时将当前 tile 的计算与前一 tile 的输出搬运重叠，并显式管理缓冲寿命。

A2/A3 使用 Ascend C Tensor API 的整数生成、Cast、Mul/Add、Abs、Sqrt、Exp 和向量多项式递推。950 以独立 RegBase/MicroAPI 实现，将坐标和递推状态保留在寄存器，用掩码控制尾部与分段，按其流水规则搬出。算法属于逐元素向量计算，无矩阵乘法或归约，主要使用 AIV 与存储引擎，不引入 Cube 转换。

### Torch 工厂接口

在 `sip_pta/csrc/base/kaiser_window.cpp` 中定义完整 schema。该算子没有张量入参，仅注册 PrivateUse1 不足以自动确定后端，因此增加工厂入口分派：解析并校验 device 后进入 NPU 实现，同时保留 PrivateUse1 注册。无设备时明确选择当前 NPU。

C++ 实现以 NPUGuard 绑定设备，`at::empty` 分配一维连续输出，获取当前 stream，查询零 workspace，再按仓内 `CreateAclTensorFromAtTensor` / `EXEC_FUNC` 方式调用 SiP。输出完成构造后设置 `requires_grad`，保证叶子语义。Python 层提供同签名封装，不以临时 Tensor 作为分派锚点，也不拼接 torch 小算子代替 kernel。直接 `torch.ops` 调用与 Python 封装分别验证。

### 工程组织

| 位置 | 变更内容 |
| --- | --- |
| `core/base/kaiser_window.cpp`、`include/base_api.h` | SiP 接口实现与 C 链接声明。 |
| `ops/base/kaiser_window/` | Operation、Kernel 注册、tiling、arch22 和 arch35 Ascend C 实现。 |
| `ops/include/params/kaiser_window.h` | 算子参数与框架缓存比较所需字段。 |
| `ops/base/CMakeLists.txt`、`configs/op_list.yaml` | 构建接入和两类 SoC 注册。 |
| `sip_pta/csrc/base/kaiser_window.cpp`、`sip_pta/torch_sip/__init__.py` | C++ 注册、NPU 工厂封装和 Python 入口。 |
| `sip_pta/test/base/kaiser_window.py`、算子接口文档 | 单算子测试与使用说明。 |

## 可维可测分析

### 精度标准与测试

使用原始测试包的 1000 条精度用例，以 CPU `torch.kaiser_window(dtype=torch.float64)` 为真值，按输出 dtype 收窄后同 dtype 比较。有限元素满足 `abs(actual-golden) <= atol + rtol*abs(golden)`，每例 matched_ratio≥0.99，同时最大绝对误差满足下表限制。

| dtype | atol=rtol | 最大绝对误差上限 |
| --- | --- | --- |
| FLOAT16 | 2^-10 | max(0.1, 32×ULP(g_low)) |
| BFLOAT16 | 2^-7 | max(0.1, 32×ULP(g_low)) |
| FLOAT32 | 2^-13 | max(0.01, 32×ULP(g_low)) |

`g_low` 为最大误差点已按输出 dtype 收窄的 golden；最大绝对误差只统计双方有限的点。NaN、inf 和空张量按任务原测试脚本判定，非有限 beta 的全 NaN 用例不计入混合容差通过率分母。

执行 `verify_accuracy.py --device npu` 和 `verify_torch_api.py --device npu`，覆盖正常参数、三种 dtype、周期/对称窗、空窗、单点窗、负 beta、零 beta、非有限 beta、分段点附近以及非对齐尾部。检查默认 float32、缺省 periodic 时仅给 beta、layout=None、显式设备和 requires_grad 叶子行为。SiP 入口另核对参数错误码与零 workspace，重复调用核对确定性。CPU 模式只支持测试包自检结论，不能代替 NPU 数值验证。

### 性能与内存标准

沿用原始 200 条性能用例及随包提供的 GPU CSV，不修改参数、基线和逐例判据。依据[任务讨论帖的计时说明](https://gitcode.com/cann/sip/discussions/4)，验收使用 Event 接口平均耗时，也可用 msprof 的 Device 耗时举证。Event 回放沿用原脚本，通过 `torch_sip.asd_kaiser_window` 预热 10 次、采集 60 次，取平均单次耗时；msprof op 采用工具自带的 5 次预热、10 次有效采样。报告注明所用计时口径、采样配置和原始结果，区分 Event 调用区间与 Device 执行时间，不将两者混合求平均。

| 硬件 | 基线 | 每例要求 |
| --- | --- | --- |
| Atlas A2、Atlas A3，分别测试 | `gpu_baseline_a100.csv` | T_A100/T_NPU≥0.8。 |
| Ascend 950 | `gpu_baseline_h100.csv` | T_H100/T_NPU≥0.4。 |

运行 `verify_performance.py --device npu --tier a2a3` 或 `--tier a5`，逐例判定，不能用平均加速比掩盖单例退化。用官方 msopprof 分析需要优化的 kernel，区分向量计算、标量计算、存储和下发空隙对完整调用的影响。

内存要求为仅占用输出 `L*sizeof(dtype)`，workspaceSize=0，运行期没有中间 GM 张量。结合源码、workspace 查询和内存观测核实。精度、接口、性能和内存分别记录，三类硬件均需完成验证。

### 兼容性分析

这是新增接口，不修改已有算子的语义。依托 SiP 当前 Operation/Kernel 注册、构建与 sip_pta 安装流程；保留各架构独立实现，公共接口一致。按 `sip_pta/README.md` 构建安装，测试所用 torch_sip 与 libasdsip.so 必须来自同一源码版本。

## 参考资料

- [SiP 算子接入指导](https://gitcode.com/cann/sip/blob/master/docs/developing_a_simple_operator.md)与 [sip_pta 封装说明](https://gitcode.com/cann/sip/blob/master/sip_pta/README.md)。
- [PyTorch 2.7.1 TensorFactories.cpp](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/TensorFactories.cpp)与 [Math.h](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/Math.h)。
- [SciPy i0e 定义与分段求值说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html)。
- [生态算子混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
