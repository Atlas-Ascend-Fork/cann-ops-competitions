# KaiserWindow 算子设计文档

## 1. 需求背景

本任务为 SiP 增加 Ascend C KaiserWindow 直调算子，并通过 sip_pta 提供 `torch_sip.asd_kaiser_window`，对齐 PyTorch 2.7.1 `torch.kaiser_window` 的参数与窗函数语义。当前 SiP 缺少原生实现；逐元素算子拼接会引入多次下发和与输出等长的中间张量。设计目标是一次核函数生成输出，不申请 workspace 或中间 GM 张量，并适配 Atlas A2、A3 与 Ascend 950（A5）。

## 2. 需求分析

### 2.1 接口与支持范围

SiP 直调接口为：

```cpp
extern "C" {
AsdSip::AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);
AsdSip::AspbStatus asdKaiserWindow(aclTensor *outTensor, int64_t windowLength,
                                   bool periodic, double beta, void *stream, void *workspace);
}
```

`outTensor` 为调用方创建的一维连续 ND 张量，shape 为 `[windowLength]`，支持 FLOAT16、BFLOAT16、FLOAT32；长度范围 `[0, 2^31-1]`。输出是独立张量，无输入张量、广播、原地或视图语义。`workspaceSize` 恒为 0，stream 为当前 NPU stream。SiP 入口校验空指针、长度、shape、连续性和 dtype，错误按仓内约定返回 `AspbStatus`。不对 beta、periodic 做取值校验。

Python 接口为：

```python
torch_sip.asd_kaiser_window(window_length, periodic=True, beta=12.0, *,
                            dtype=None, layout=torch.strided, device=None,
                            requires_grad=False) -> torch.Tensor
```

`dtype=None` 取 float32；`device=None` 使用当前 NPU，显式 device 只接受 NPU；layout 只支持 strided（None 按默认），返回一维连续张量。`requires_grad=True` 时返回可累积梯度的叶子张量，无反向公式。非法长度、dtype、layout 或 device 在封装层报告错误，不静默回退。periodic 和 beta 可位置或关键字传入。

### 2.2 数学与边界定义

设 `L=windowLength`。当 `L>=2`，`N = L if periodic else L-1`，对 `0<=i<L`：

```text
r = (2*i - N) / N
t = sqrt(abs(1 - r*r))
out[i] = I0(beta*t) / I0(beta)
```

其中 I0 为第一类零阶修正贝塞尔函数。周期窗按 N=L 计算，仅输出前 L 个元素；非周期窗 N=L-1。分子先在整型域计算再转 float32，避免中心附近相消；中间使用至少 int64，避免长度上限时 `2*i` 溢出。端点开方前取绝对值。

边界分支按以下顺序执行：L=0 返回空张量且不发核；L=1 写 1（包括 beta 为 NaN 或 ±inf）；L>=2 且 beta 非有限值写全 NaN；beta=0 写全 1；其他情况按公式计算。负 beta 与其绝对值等价。L=2 非周期输出两个端点，周期输出端点与中心；L=3 非周期输出端点、中心、端点。相同参数的结果应逐位确定。

## 3. 详细设计

### 3.1 方案分析

采用生成式单次核函数下发方案：每个输出元素独立计算，只在片上缓冲区保存当前 tile 的索引和中间值，最终直接写 out。相较于多个逐元素算子的组合，该方案减少 kernel 下发和中间 GM 流量。直接计算 `I0(beta*t)/I0(beta)` 在较大 beta 时可能溢出，因此使用缩放函数 `i0e(x)=exp(-abs(x))*I0(x)`：

```text
b = abs(beta)
out = i0e(b*t) / i0e(b) * exp(b*(t-1))
```

指数必须使用 `abs(beta)`，以满足负 beta 的偶函数语义；任务书中使用 `beta*(t-1)` 的写法对负 beta 不满足该语义，本设计以边界契约为准。对于有限 beta，采用与 PyTorch `calc_i0` 相同的分段 Chebyshev 展开思路计算 i0e，按区间选择系数和变量变换，使用 Clenshaw 递推或等价实现；避免先求可能溢出的 I0 再乘衰减因子。`i0e(b)` 对各元素相同，每个核或 tile 只计算一次并复用。计算主路径使用 float32，最终一次转换为输出 dtype；系数、分段阈值、取整和指令精度通过精度自验确定，特别验证端点、中心、区间边界和大 beta。对于超出主要测试范围的有限 double，转换及数值行为须在实现中与 CPU 参考额外对照，不能将有限值误判为非有限输入。

### 3.2 Host 侧设计

1. `sip_pta` Python 层规范化参数、处理 layout/device/requires_grad，按 SiP 扩展习惯注册 `TORCH_LIBRARY_FRAGMENT(torch_sip)` 并注册 `BackendSelect` 和 `PrivateUse1` 实现。无 tensor 入参，应保证工厂函数能够正确分派，并测试裸 `torch.ops.torch_sip.asd_kaiser_window`。
2. C++ 层以 NPUGuard 选择目标设备，以 `at::empty` 创建对应 dtype 的输出，转为 aclTensor，获取当前 stream。先查询 workspace（0），再通过 `EXEC_FUNC` 调用 SiP；L=0 不启动核。设置 `requires_grad` 时保证输出为叶子张量。
3. SiP 入口检查参数并构造 OpDesc，调用 `RunAsdOpsV2`。Operation 实现 shape 推断及 kernel 选择；Kernel 实现支持性检查、tiling 大小和初始化。将长度、periodic、beta、分块信息传入 tiling data，非法参数及时返回，workspace 不使用。
4. Tiling 按输出长度和硬件核数计算有效 block 数，将连续索引区间分配给各核，每核再按 UB 可用容量切 tile。尾块按真实元素数写回，避免越界；对写回对齐要求使用安全的尾部处理，不能覆盖相邻核区域。tile 大小及核数基于各档硬件测量调整，小长度控制启动开销，大长度覆盖计算与写回带宽。

### 3.3 Device 侧设计

每个核按 tiling 获取起始索引与长度，构造全局 int64 索引后计算 `2*i-N`，转 float32 并计算 r、t；按 beta 状态选快速分支或缩放贝塞尔主路径。中间计算在 UB/寄存器中进行，结果按输出类型转换后写入 GM。多核分段和 tile 尾处理不得造成重复写或空洞。A2/A3 使用 ascend910b（arch22）AIV 向量实现，A5 使用 ascend950（arch35）RegBase/MicroAPI 实现，分别组织硬件相关代码，保持参数和数值语义一致。需要根据各平台指令支持情况选择 int64 索引构造方式及缓冲布局；在全范围保持整数精确性，不以 float 相减替代。

### 3.4 代码组织与支持硬件

- `core/base/kaiser_window.cpp`、`include/base_api.h`：SiP 参数检查及导出接口。
- `ops/include/params/kaiser_window.h`、`ops/base/kaiser_window/`：参数、Operation、Kernel、tiling 与 Ascend C 实现；`ops/base/CMakeLists.txt` 接入构建。
- `configs/op_list.yaml`：注册 Operation/Kernel，支持 ascend910b 和 ascend950。
- `sip_pta/csrc/base/kaiser_window.cpp`、`sip_pta/torch_sip/__init__.py`、`sip_pta/test/base/kaiser_window.py`：注册、Python 封装及自测。

具体文件与注册方式以 SiP 当前开发规范为准。运行环境为 CANN 9.3.0、torch/torch_npu 2.7.1 和同源 asdsip/torch_sip。A2、A3、A5 分别验收；不支持 FLOAT64、整型、BOOL、复数、非连续或稀疏输出。动态 shape 和广播均不涉及。运行期 GM 占用为输出 `L * sizeof(dtype)`，workspace 为 0。

## 4. 可维可测分析

### 4.1 精度与功能

使用任务包 `verify_accuracy.py --device npu`，对 1000 条精度用例逐条比较 CPU float64 `torch.kaiser_window` 真值收窄到输出 dtype 后的结果。FLOAT16、BFLOAT16、FLOAT32 的 rtol 和 atol 分别为 `2^-10`、`2^-7`、`2^-13`；逐元素满足 `abs(actual-golden) <= atol + rtol*abs(golden)` 的比例至少 0.99，且有限值最大绝对误差不超过 `max(兜底值, 32*ULP(g_low))`，兜底值前两类为 0.1，FLOAT32 为 0.01。NaN 和无穷的比较遵循任务包规则，非有限 beta 的全 NaN 单独判定。覆盖 L=0、1、2、3、奇偶长度、tile 边界、两种 periodic、beta=0、负 beta、NaN、±inf、[0,88]、三种 dtype，以及中心、端点和分段近似边界。验证对称性、周期语义、确定性和长度上限附近的索引计算。

运行 `verify_torch_api.py --device npu` 检查默认参数、关键字调用、shape/dtype/device、layout、requires_grad 叶子语义和空张量等正常接口行为。可先用两个脚本的 `--device cpu` 做参考语义自检；CPU 自检不代替 NPU 数值验收。针对 SiP 接口补充空指针、shape、dtype、非连续、长度越界等错误返回测试。

### 4.2 性能与内存

用 `verify_performance.py --device npu --tier a2a3` 在 A2、A3 分别采集，`--tier a5` 在 A5 采集；200 条性能用例逐项与随包 A100/H100 基线关联。预热 10 次、采样 60 次，比较平均单次 Device 耗时：A2/A3 均要求 `T_A100/T_NPU >= 0.8`，A5 要求 `T_H100/T_NPU >= 0.4`。按 dtype、small/medium/large 分析并调整 block 数、tile 和双缓冲等策略，只有验证正确且实际收益明确时引入优化。

采集 out 大小及 `GetWorkspaceSize`，检查 workspace 恒为 0、无中间 GM 张量；关注不同长度、dtype 与架构的尾块、对齐和 UB 使用。所有测试结果、环境和异常原因记录于自测报告；本设计为计划，不表示上述测试已经通过。

### 4.3 兼容性与风险

与 SiP 的 stream、返回码、CMake、注册和 sip_pta ABI 约定保持一致，并分别在 A2/A3/A5 验证。重点风险为低精度输出的逼近误差、beta 的非有限/负值处理、超大索引整数溢出、写回尾部对齐及无 tensor 参数的框架分派；通过上述边界和跨平台测试定位，未达指标时先修正算法或 tiling 再重新验收。
