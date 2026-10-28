# KaiserWindow 算子设计文档

## 1. 需求背景（required）

### 1.1 需求来源

- 社区任务：10月社区任务-torch_npu支持kaiser_window接口开发（A2/A3/950）。
- 任务编号：`3e07081ee52b4043a030ec48d3f0ec34`；贡献帐号：`AIAFKK`。
- [任务详情](https://www.hiascend.com/activities/task-center/details/3e07081ee52b4043a030ec48d3f0ec34)。设计依据为任务所附 `KaiserWindow_task_doc(A2A3A5).md` 及 `test_cases`，任务书 SHA256：`0a2f7a6981c2203c029f0107bfeb29f74bede748474d256f571df783297a70b5`。
- 设计提交至 [CANN 社区任务设计仓](https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，实现及 PTA 封装提交目标为 [cann/sip](https://gitcode.com/cann/sip)。本文按活动设计模板组织。

### 1.2 背景介绍

Kaiser 窗用于信号处理和频谱分析。当前任务要求在 NPU 上直接生成窗口，替代 CPU 执行或多次逐元素算子拼接，减少 kernel 下发和与输出等长的中间 GM 张量。输出只由长度、周期标志和 beta 决定，无张量输入。

对标接口为 PyTorch 2.7.1 `torch.kaiser_window`。本实现采用 SiP Ascend C kernel 直调接口及仓内 `sip_pta` 扩展，提供 `torch_sip.asd_kaiser_window`，不修改 PyTorch 自身接口，也不注册 CPU 回退。

## 2. 需求分析（required）

### 2.1 需求描述

| 参数 | 类型 / 默认值 | 语义与约束 |
| --- | --- | --- |
| window_length | host int64，必填 | 0 至 2^31-1，决定一维输出长度 |
| periodic | bool，True | True：N=L；False：N=L-1 |
| beta | host double，12.0 | 任务常规范围 [0,88]，含负值、NaN、±inf 边界；不做 beta 取值拒绝 |
| dtype | torch.dtype，None | None 固定按 FP32；支持 FP16/BF16/FP32 |
| layout | torch.layout，strided | 仅连续 strided；None 按默认处理 |
| device | torch.device，None | NPU；None 使用当前 NPU 设备 |
| requires_grad | bool，False | True 时返回可累积梯度的叶子，无反向公式 |
| outTensor | aclTensor 输出 | ND、连续、shape=[L]；调用方负责分配与生命周期 |

### 2.2 需求拆解

1. 新增 SiP C ABI、Operation、Tiling、Kernel 注册及 Torch factory 封装，完整支持关键字参数和裸调入口。
2. A2/A3 使用 ascend910b（arch22）AIV 路径；950 使用 ascend950（arch35）RegBase/MicroAPI 路径，分开编译和选择。
3. 所有中间量在寄存器或 UB 内计算，workspace=0；非空普通路径一次 kernel 完成。
4. 按原任务工程逐档运行 1000 条精度与 200 条性能用例，A2/A3 分别达到 A100 的 0.8 倍，950 达到 H100 的 0.4 倍。
5. 验证空张量、标量窗、周期/对称、非有限 beta、确定性、内存及完整 Torch 参数面。

## 3. 详细设计（required）

### 3.1 算子分析

#### 数学公式

令 L=window_length，N=periodic ? L : L-1。对于 L>=2：

```text
j[i] = i - (N - i)                    # 精确整型计算，等价于 2*i-N
r[i] = float32(j[i]) / float32(N)
t[i] = sqrt(abs(1 - r[i]*r[i]))
b = abs(beta)
i0e(x) = exp(-abs(x)) * I0(x)
out[i] = i0e(b*t[i]) / i0e(b) * exp(b*(t[i]-1))
```

`i-(N-i)` 避免先算 `2*i` 在 INT32 上界溢出；整型中心化避免大索引转 FP32 后相减的抵消误差。`abs` 后开方保持任务要求的端点行为。稳定的 i0e 比值形式避免分别求 I0 时的大值溢出。所有普通数值计算使用 FP32，最终转换到输出类型。

#### 边界分支与顺序

| 条件 | 行为 |
| --- | --- |
| L=0 | 返回空输出，查询 workspace 得到 0，不启动计算 kernel |
| L=1，包括任意非有限 beta | 直接填 1，优先于 beta 分类 |
| L>=2 且 beta=0 | 全 1，不计算分母 |
| L>=2 且 beta 为 NaN 或 ±inf | 全 NaN |
| 其余有限 beta | 取绝对值后计算稳定公式 |

L=2 的对称窗为 `[1/I0(beta),1/I0(beta)]`，周期窗为 `[1/I0(beta),1]`；L=3 的对称窗中心为 1。输出不做额外裁剪，以独立 golden 和任务判据验证误差。

### 3.2 接口与 Host 设计

#### SiP C ABI

```cpp
extern "C" {
AsdSip::AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);
AsdSip::AspbStatus asdKaiserWindow(aclTensor *outTensor,
                                int64_t windowLength,
                                bool periodic,
                                double beta,
                                void *stream,
                                void *workspace);
}
```

查询接口写入 0。执行接口检查长度、输出 dtype、维数、shape 和连续性，按 SiP 错误码返回，成功为 ACL_SUCCESS。非空输出的内存由调用方持有，stream 显式传入；不在接口内分配输出或等待整个设备。

公共入口将 aclTensor 的形状、dtype、storage address 与 view offset 转为 Mki::Tensor，地址使用 `storage address + view offset * element_size`，再通过 `RunAsdOps` 调用零输入 Operation。此处沿用 SiP 已有下发路径，并保留偏移；不直接使用当前 `RunAsdOpsV2` 中丢失 offset 的转换，避免连续子视图写入 allocation 起点。Torch 正常工厂输出本身不是视图。

#### Torch factory

```python
def asd_kaiser_window(window_length: int,
                      periodic: bool = True,
                      beta: float = 12.0,
                      *, dtype=None, layout=torch.strided,
                      device=None, requires_grad: bool = False) -> torch.Tensor:
    ...
```

Python 层提供完整参数与类型标注，调用 `torch.ops.torch_sip.asd_kaiser_window`。C++ 层使用 `TORCH_LIBRARY_FRAGMENT` 定义 schema，注册 PrivateUse1，并为无张量输入的 factory 注册 BackendSelect，使裸调也能选择 NPU 实现。

C++ 处理顺序为：校验 dtype/layout/device 与长度 → NPUGuard 选择目标 NPU → `at::empty` 创建连续输出 → 构造 aclTensor → 查询 workspace → 使用当前 NPU stream 调用 SiP → 销毁 aclTensor 描述符 → 按 requires_grad 设置叶子属性。描述符生命周期与异步输出存储生命周期分开，输出 Tensor 保有数据所有权；默认 dtype 不跟随全局 `torch.set_default_dtype` 改变。

#### 分核和 Tiling

Tiling 记录 L、N、abs(beta) 的核内计算值、dtype、分支模式与填充值。初始 tile 长度为 4096；block 数为 `min(实际 Vector 核数, ceil(L/4096))`。第 k 个核处理起点 k*4096，后续按 block数*4096 跨步，保证不重叠且覆盖所有元素。核数从平台获取，不硬编码具体机器核心数。

尾 tile 只输出有效 count 个元素；向量计算长度按指令要求对齐，GM 写回使用有效字节数的 DataCopyPad，禁止按对齐长度越界写。dtype 分支选择三种独立输出类型，架构分支选择 arch22 或 arch35，普通/填一/填 NaN 模式独立。

### 3.3 Kernel 设计

#### i0e 近似

采用 PyTorch v2.7.1 `aten/src/ATen/native/Math.h` 中 Cephes 的分段 Chebyshev 系数：小参数区间 [0,8]，大参数区间 (8,+∞)。arch35 保留小段 18 项、大段 10 项，省略项的绝对值之和分别小于 3.30e-9 和 1.62e-11；arch22 为降低完整调用耗时，保留小段 13 项、大段 5 项，相对完整 Cephes 的省略绝对值之和分别小于 5.878e-6 和 2.662e-8。大参数段还需除以 sqrt(x)。在各自定义区间内 Chebyshev 基函数绝对值不超过 1，该界只界定截断误差，舍入和组合公式误差仍由完整 golden 校验。

负参数先取绝对值。beta<=8 时仅需小参数分支；beta>8 时对每个位置按 b*t 是否小于等于 8 选择对应结果。对大参数支路输入先取 max(x,8)，避免未选中支路在 x=0 时除零。最终分母为同一 beta 的 i0e 值，不随元素变化；每 tile 至多计算一次，并允许在同核多 tile 间复用，不在 Host 上计算窗口。

实现保留 Cephes/PyTorch 来源与 BSD 许可说明，新增源文件遵循 SiP 仓许可。

#### A2/A3：arch22

使用 Ascend C 高层向量 API 在 UB 中完成索引、FP32 转换、sqrt、exp 和分段多项式。基线采用 Clenshaw 求值；优化方案采用上述 13/5 项保留序列，转换为幂基并用 Horner 求值，减少向量操作和同步次数。已在 A3 原生入口通过 1000/1000 精度。分母 i0e(beta) 每核只计算一次；每次 CreateVecIndex 前显式配对 V_S set/wait，保护其 Scalar SetValue 播种与前一 tile Vector 消费之间的缓冲复用依赖。

4096 元素方案预留 10 个 FP32 scratch tile（160 KiB）、比较掩码 512 bytes、单输出队列最多 16 KiB，合计 176.5 KiB；在目标平台 UB 容量内运行。若减少活跃中间量，可进一步复用 scratch，不将临时张量放入 GM。

流水依赖用 PIPE_V barrier、向量到标量事件及输出队列同步表达；从分母向量读标量前等待 V_S 完成。队列写回后释放，下一 tile 复用前遵循队列生命周期，避免异步搬运与计算覆写冲突。

#### 950：arch35

使用 RegBase/MicroAPI 向量寄存器完成整型中心索引、FP32 函数与 Clenshaw 多项式；以 mask 控制向量尾部。UB FP32 结果缓冲为 16 KiB，单输出队列最多 16 KiB；完成转换后按有效长度写回。架构专用代码单独放在 arch35 目录，不把 arch35 指令编译到 A2/A3 路径。

#### 性能优化计划

小尺寸优先降低完整 Torch/SiP 调度成本与分母固定成本；大尺寸优先减少多项式指令、barrier 和重复分母求值，结合 UB 容量评估 tile 与并行度。使用同一份官方 200 条清单逐条比较，不按特定用例返回预计算结果。kernel-only 计时仅用于定位，最终性能判定保持完整规定接口范围。

### 3.4 文件布局与兼容性

| 模块 | SiP 仓位置 |
| --- | --- |
| C ABI 声明与校验入口 | include/base_api.h；core/base/kaiser_window.cpp |
| 参数和 Operation | ops/include/params/kaiser_window.h；ops/base/kaiser_window/kaiser_window_operation.cpp |
| Kernel 注册与 Tiling | ops/base/kaiser_window/kaiser_window/kaiser_window_kernel.cpp；同级 tiling/ |
| 架构内核 | ops/base/kaiser_window/kaiser_window/op_kernel/arch22/ 与 arch35/ |
| 构建注册 | ops/base/CMakeLists.txt；ops/base/kaiser_window/CMakeLists.txt；configs/op_list.yaml |
| Torch 封装 | sip_pta/csrc/base/kaiser_window.cpp；sip_pta/torch_sip/__init__.py |
| 独立回归 | sip_pta/test/base/kaiser_window.py；kaiser_window_sip_abi.py |

基于 SiP `c81024ba795a6cf5dfb55edd79721f04f0ede05e` 接入，asdsip 与 sip_pta 保持同源和一致 C++ ABI。新增算子不改变已有接口和其他算子调度。支持 A2、A3、950；不支持 FLOAT64、整型、BOOL、复数、稀疏 layout、非连续输出或广播。输出与其他张量重叠不在契约内。

## 4. 可维可测分析

### 4.1 精度标准与独立参考

按任务原始 `verify_accuracy.py` 和 `kaiser_window_test.csv` 完整运行 1000 条。CPU `torch.kaiser_window(dtype=torch.float64)` 在线生成真值，再收窄到实际输出 dtype 后比较，不使用被测 i0e 代码生成 golden。

| dtype | atol / rtol | matched_ratio | max_abs_error 上限 |
| --- | --- | --- | --- |
| FLOAT16 | 2^-10 | >=0.99 | max(0.1,32*ULP(g_low)) |
| BFLOAT16 | 2^-7 | >=0.99 | max(0.1,32*ULP(g_low)) |
| FLOAT32 | 2^-13 | >=0.99 | max(0.01,32*ULP(g_low)) |

逐元素条件为 `abs(actual-golden)<=atol+rtol*abs(golden)`，同时要求整条用例满足匹配率和最大误差条件。g_low 是最大有限绝对误差位置、已按输出 dtype 收窄的 golden；非有限点按任务规则匹配同号 inf，非有限 beta 全 NaN 用例单独判定。保留原始失败，不改阈值或删 case。

### 4.2 性能标准

| 硬件档 | 标杆文件 | 每条要求 |
| --- | --- | --- |
| A2 | gpu_baseline_a100.csv | T_A100 / T_NPU >=0.8 |
| A3 | gpu_baseline_a100.csv | T_A100 / T_NPU >=0.8 |
| 950（A5档） | gpu_baseline_h100.csv | T_H100 / T_NPU >=0.4 |

每档 200 条，三档各自采集、各自判断；参数、dtype、periodic 与对应基线逐行对齐。原始 `verify_performance.py --tier a2a3` / `--tier a5` 使用 10 次预热、60 次采样平均，保持其 Device Event 计时边界；不以单独 kernel 时间替换完整 `torch_sip.asd_kaiser_window` 结果。golden 比对不进入计时，Host 构造输出按原测试工程口径处理。

### 4.3 功能、内存与回归

- 运行原 `verify_torch_api.py` 检查默认参数、位置/关键字参数、dtype/layout/device、requires_grad 叶子行为，以及裸调接口。
- 覆盖 L=0/1/2/3、非对齐尾块、跨 tile、4M 大规模、正负/零/非有限 beta 和三种 dtype。
- 对同参数连续调用检查逐位一致；按任务规则单独处理 NaN。
- 对代表规模与三 dtype 查询 workspaceSize，记录 out bytes+workspaceSize；要求 workspaceSize=0，源码和运行路径无中间 GM 张量。
- 定向回归覆盖非默认 stream、连续 offset 输出视图前后哨兵及非法参数拒绝；这些补充回归不替代任务全量正常用例。
- 每次结果绑定源码、实际加载的 asdsip/torch_sip、CANN/torch/torch_npu、设备、命令和原始结果。相同最终候选覆盖全部必过项。

### 4.4 环境与当前验证范围

任务书列出 CANN 9.3.0、torch/torch_npu 2.7.1。当前实际环境为 CANN 9.1.0：A3 使用 torch2.7.1+cpu/torch_npu2.7.1，950 使用 torch2.7.1+cpu/torch_npu2.7.1.post8（官方26.1.0配套版本）。实际版本在自测报告与复现说明中披露。

同一 R4 候选已完成 A3/950 原生 SiP/Torch 验证：两档各 1000/1000 精度、23/23 接口、12/12 输出内存/workspace、12/12 连续视图 offset 哨兵；A3 两设备/stream/guard 恢复 12/12。A3 性能首轮 199/200，唯一 L2 BF16 为 23.869us、倍率0.7665；保留该次全量原始结果后，相同候选、相同协议完整复测 200/200，最低倍率约0.8270。950 200/200，最低约0.4697。A2 尚待真机补验。

另按用户要求记录内部 2% 余量门，定义为官方性能倍率门槛乘1.02，即 A2/A3 >=0.816、950 >=0.408；它不改变官方阈值。现有 A3/950 每档200/200均满足，保守最低倍率约0.82698/0.46974。完整结果与首次失败在自测报告中同时保留。

设计 PR 提交、设计评审、合入、自验及官方验收分别记录。最终交付提供同源源码、固定依赖、完整构建与测试步骤、原始结果和自测报告。

## 5. 参考资料

- [本任务页面与附件](https://www.hiascend.com/activities/task-center/details/3e07081ee52b4043a030ec48d3f0ec34)
- [设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [社区任务流程](https://gitcode.com/org/cann/discussions/39)
- [SiP 算子接入](https://gitcode.com/cann/sip/blob/master/docs/developing_a_simple_operator.md)
- [SiP PTA 封装](https://gitcode.com/cann/sip/blob/master/sip_pta/README.md)
- [PyTorch 2.7.1 Math.h](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/Math.h)
