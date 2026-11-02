# 需求背景（required）

## 需求来源

社区任务《aclblasCgemvBatched 算子开发任务书》，目标 Atlas 950PR、CANN 9.1.0。
本文遵循 [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 的章节结构。

## 背景介绍

开发基线为官方 ops-blas master 的 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。
此基线已包含 arch35 单矩阵 `aclblasCgemv` 的 Ascend C 内核，尚无 arch35 复数批量入口。
公共头与 arch22 批量入口使用单指针，不能表达彼此独立、非相邻的 batch 地址。
本次使用已合入的单 GEMV 实现，不引入未合入社区 PR。

# 需求分析（required）

## 需求描述

对每个 batch 计算 `y[b] = alpha * op(A[b]) * x[b] + beta * y[b]`。
输入输出为 complex64；A 始终为 m 行 n 列、列主序；N/T/C、lda padding、正负向量步长均需支持。
alpha/beta 位于 Host；三个指针数组及其指向的数据均位于 Device。

## 需求拆解

1. 公共 API 修正为指针数组，同步 arch22，移除原单指针语义。
2. 按任务书固定顺序校验；所有数组元素在首个 batch 下发前检查完毕。
3. arch35 逐 batch 调用现有 `aclblasCgemv`，通过 handle 绑定的 stream 下发 Ascend C kernel。
4. 接入 CSV/GTest、Netlib CBLAS golden、逐位确定性检查及独立性能判定。
5. 上机依次完成构建、小用例、完整精度、重复输出稳定性和性能验收。

# 详细设计（required）

## 算子分析

### 数学公式

`A(i,j) = A[i + j*lda]`。N 的 x/y 长度为 n/m；T、C 为 m/n。
复数乘法为 `(ar*xr-ai*xi, ar*xi+ai*xr)`；C 在乘法前令 `ai=-ai`。
负步长 inc 的逻辑元素 k 位于 `(length-1-k)*(-int64_t(inc))`，传入地址始终是物理分配起点。
输入含 Inf/NaN 时遵循任务书 §3.2 的专用判定；不通过修改 golden 来迎合设备结果。

### 支持数据类型与形状

complex64（两个 FP32 分量），m、n、batchCount 均大于零，`lda>=m`。
incx/incy 非零且不等于 INT_MIN；不做广播、通用非连续 tensor 或图模式动态 shape 处理。
不同 batch 的 y 不得重叠，调用方应保证所有存储与输出生命周期覆盖 stream 完成时刻。

## 算子实现

### 公共接口

```cpp
aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const xarray[], int incx, const aclblasComplex* beta,
    aclblasComplex* const yarray[], int incy, int batchCount);
```

三个数组参数均为 Device 指针表；alpha/beta 是 Host 复数标量指针。
`const` 限定输入矩阵、向量及表项，不改变 y 指向的数据可写的语义。

### Host 侧设计

公共实现位于 `blas/gemv_batched/cgemv_batched_host_impl.h`，两个架构入口各自调用它。
依次检查：handle、m/n、lda、步长/batchCount、三个数组指针、alpha/beta、trans。
普通参数失败时不分配 pointer snapshot、不访问 ACL runtime、不启动任何 kernel。

随后取得绑定的 stream，等待其先前操作完成，用三次 D2H 拷贝读取设备指针表。
主机只接收 `3*batchCount*sizeof(void*)` 元数据，不搬运矩阵、向量内容。
全部检查成功后才遍历 batch 调用单 GEMV；任何位置的空元素都返回 INVALID_VALUE，且没有本次输出写入。
分配失败返回 ALLOC_FAILED，指针表拷贝失败返回 MAPPING_ERROR，前置 stream 同步失败返回 EXECUTION_FAILED。
下层 GEMV 错误原样传播并停止后续下发；运行时错误不承诺回滚已成功下发的 batch。

**同步边界**：设备指针表内容需要同步返回空元素错误，因此入口的元数据预检会阻塞。
arch35 最后一次下发后不追加同步，读取结果仍由调用方同步绑定 stream。
这满足“调用前逐元素检查”和计算输出异步完成，但不提供完全非阻塞的 Host 入口或图捕获保证。
arch22 的单 GEMV 本身含同步，保留这一既有执行方式。

### Kernel 侧设计

复用 `blas/gemv/arch35/` 的 `cgemv_host.cpp` 及 `cgemv_*kernel.cpp`，不复制一套批量私有 kernel。

- 普通 N/T/C 和非单位步长复用 ordered 路径，地址计算采用单 GEMV 的宽索引。
- alpha=0 复用 scale 路径；alpha=0、beta=1 不写输出；beta=0 不读取旧 y。
- 满足单 GEMV 优化条件的输入复用 N/T slab 路径。N 可能有 compute/finalize 两次 kernel 下发。
- 单 GEMV 决定分核、UB 分块与 workspace 大小；不足时保留其已有回退，不另申请批量 device workspace。
- 同一个 stream 顺序使用 handle workspace，使前一 batch 的消费先于后一 batch 的覆盖。
- 批量层不引入原子规约、跨 batch 规约或自定义 tiling key；确定性依赖既有固定规约拓扑，并由重复测试验证。

`blas/CMakeLists.txt` 明确保留 cgemv 依赖，保证仅编译 `cgemv_batched` 时也包含所需 host/kernel。
arch22 原单指针批量 kernel 及专用 mask/plan 已无调用者，随接口迁移删除。

## 支持硬件

| 芯片 | 本次变更 | 验证状态 |
| --- | --- | --- |
| Atlas 950PR / arch35 | 新增接口并复用现有 Ascend C GEMV | 待 CANN 9.1.0 上机 |
| Atlas A2/A3 / arch22 | 公共签名迁移、逐 batch 复用 GEMV | 仅主机契约通过；旧 GEMV 存在下述已知缺陷，不能声明完整数值支持 |

## 算子约束限制

调用方负责设备地址有效性、容量与生命周期；预检仅检查空元素，不能证明非空设备地址有效。
不支持同时在另一 stream 修改同一指针表且不建立依赖。
单 GEMV 的数值行为、编译器兼容性和性能仍需在任务要求环境验证，Host 测试不能替代。

### arch22 已发现的依赖缺陷

本次自检对官方基线 `05b96ea` 的 arch22 单 GEMV 发现以下问题，不能以主机替身测试通过掩盖：

- 负步长：Host 未调整尾部起点，`ascblasCgemv_utils.h` 的 stride 拷贝还会对负步长形成负长度；仅移动传入指针不能修复。
- 零标量：旧 kernel 无条件读取参与 `alpha`/`beta` 乘法的数据。例如 T、m=n=1、A=x=1、alpha=1、beta=0、旧 y=NaN，应得有限值，但旧路径会传播 NaN。
- N 的多 section 路径：部分 section 的 Y 累加器未初始化，并通过原子加累加到同一输出。数值影响需设备复现，逐位确定性目前没有保证。

arch22 CSV 已增加 17 条负步长、零标量/NaN 和多 section 回归用例，保留正确期望、不跳过；这些设备测试尚未执行。
修复这些问题涉及既有单 GEMV 的 Host、N/T/C kernel 和 gather/scatter 助手，需独立修复及 A2/A3 设备验证；本次 950PR 设计不把接口迁移等同于 arch22 全功能修复。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 实现/判据 | 证据位置 |
| --- | --- | --- |
| 精度 | Netlib BLAS 3.10.0 `cblas_cgemv`，实部/虚部分别全量统计；atol=rtol=2^-13、ratio>=0.99 | GTest 日志、XML |
| 最大误差 | max(1e-2,32*ULP)，锚定最大误差位置的 golden；相同最大误差取更严格上限 | `cgemv_batched_precision.h` |
| 非有限值 | NaN/NaN、同号 Inf、NaN/Inf 等效；有限/非有限及异号 Inf 失败 | 专用计数及 Host 单元测试 |
| 确定性 | 恢复完全相同 y 初值后再执行，对全部输出字节 memcmp | 每个成功精度用例 |
| 内存边界 | y 前后 guard 和 stride gap 保持原字节 | GTest |
| 性能 | 全部 batch、全部必要 kernel 的 task duration 求和，再除有效调用次数 | 原始 msprof CSV |
| 核心门槛 | §3.3：10 次有效样本，NPU Avg<=GPU/0.4 | 200 个 GPU 标杆逐行匹配 |

性能基线保留原始 200 条、按行 ID 及全部形状字段核对，禁止只以形状去重。
小矩阵大 batch 的逐次下发开销是本方案的明确性能风险。未上机前不承诺达到阈值；若证据不达标，再定位优化，保留此受控基线。
正式验收只使用指定环境的 Netlib；CMake 自动找到的其他 BLAS 不能直接冒充 Netlib 3.10.0。

## 兼容性分析

公共 C 符号保持 `aclblasCgemvBatched`，签名修正为指针数组。旧单指针源码必须迁移并重新编译，不能依靠同名符号继续调用旧 ABI。
不保留单指针重载，不增加 950PR 私有 API。arch22 测试、README 样例同步迁移。

## 任务材料冲突与处理

| 材料问题 | 本次处理 |
| --- | --- |
| 测试 README 写 T/C 的 y 长 m | 按任务书公式及接口：y 长 n |
| README 中一处写空元素 SUCCESS | 按任务书 §2.4：全表预检，返回 INVALID_VALUE |
| §3.3 为 0.4 倍、10 次；§7 为至少 1 倍、超过 10 次；README 超过 50 次 | 同时输出三个判定，保留冲突，未获得统一口径前不声称全部验收达标 |
| 附件性能脚本读取 GTest 墙钟耗时，允许 NO_REF 成功 | 替换为 raw msprof task duration 验证；缺失、部分记录、重复记录均失败 |
| 附件脚本使用已不支持的 `--test` 构建参数 | 使用当前官方 `build.sh --soc=ascend950 --ops=cgemv_batched` |
| 原始 CSV 未提供正态填充，未提供空数组元素及纯实/纯虚数组用例 | 原 1200 条不改参数；增加 1175 条合法用例的正态镜像、216 条配对分布 CSV 及 4 项 GTest 专项检查 |

## 测试与交付状态

本地 Host 合约回归为 arch35/arch22 各 160 项，通过 ACL/GEMV 测试替身检查实际批量入口，格式整理前均通过。
最终格式整理后 arch35 再次通过 160 项；arch22 编译成功，但执行被 Windows 应用控制策略拒绝（WinError 4551），本轮不能记为运行通过。
两套完整 GTest 源码均通过本地对象编译，使用真实 GoogleTest 1.14.0/Netlib 3.10.0 CBLAS 头文件及 ACL 声明替身；这不是 CANN 编译证据。
自检修复后的 21 项合成证据/数据覆盖回归、两架构单算子依赖清单检查、clang-format 18.1.8、头文件规范与 OAT 扫描通过。
补充的正态镜像与纯实/纯虚数组测试已纳入源码；默认精度 2195 项、完整精度 2595 项，尚待设备执行。
性能解析器兼容原生时间戳文件名，拒绝拼接采集及修改标量后套用原 GPU 基线。`TC_PF_1016` 的双 kernel 路径需要 5120 条记录，超过文档中的 5000 采集上限，完整采集方案仍需上机确定。
设备精度、CANN 内核编译、性能、截图和验收提交均待后续上机完成。
测试复现说明位于 `test/gemv_batched/cgemv_batched/README.md`。
