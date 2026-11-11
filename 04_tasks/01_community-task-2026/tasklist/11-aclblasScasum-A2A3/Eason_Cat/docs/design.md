# 需求背景（required）

## 需求来源

- 来源渠道：202611 社区任务讨论（开源仓 `https://gitcode.com/cann/ops-blas`，任务方向 A2/A3 算子补齐）。
- 任务名称：`aclblasScasum` 算子开发（A2/A3）。
- 关联任务：BLAS L1 系列在 arch22（Atlas A2/A3，910B / arch22）上的复数算子补全；同批次任务包含 `sasum/scasum/scnrm2` 等归约类算子的 arch22 对齐。
- 工程模式：aclblas 句柄式 kernel 直调（Ascend C 实现，Host 侧 tiling + Kernel 侧算子函数），不走 ACLNN 图编译链路，不依赖 AscendC SIMT / `asc_simt`（该能力 A5 才具备）。

## 背景介绍

1. `scasum` 是 BLAS Level 1 的复数绝对值求和算子，广泛用于数值算法中的范数估计、条件数估计、迭代收敛判据等场景，是 `cblas` 兼容层的基础接口之一。
2. 仓内现状：
   - `blas/l1/scasum/arch35/` 已存在 `scasum_*` 实现，可作为语义与接口对齐参考；
   - `blas/l1/sasum/arch22/` 已存在实数绝对值求和实现，具备 arch22（A2/A3）上的分核、UB 分块、多核 partial sum 合并的成熟范式，可直接复用；
   - `blas/l1/scasum/arch22/` **缺失**，导致 arch22 平台上 complex64 的 `scasum` 无法通过 aclblas 句柄式接口下发，属于功能缺口。
3. arch22（910B）平台的 VectorCore 提供 FP32 向量算力与 `Abs / Add / ReduceSum` 等归约指令，但**不具备** SIMT 编程模型，stride 访问需通过 membase gather 或标量地址计算实现；因此 arch22 实现必须与 arch35 的实现方案解耦，重新按 arch22 指令与内存层级设计。
4. 精度与性能对标要求与仓内其它 BLAS 算子保持一致：精度对标 Netlib / cblas（`cblas_scasum`），性能对标 A100 上 cuBLAS `cublasScasum`（验收阈值见"可维可测分析"）。

# 需求分析（required）

## 需求描述

需要在 arch22 上新增 `aclblasScasum` 算子，提供 aclblas 句柄式 kernel 直调接口：

`aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);`

语义：对复数向量 `x` 的 `n` 个逻辑元素，逐元素计算实部绝对值与虚部绝对值之和并累加，结果写入 FP32 标量 `result`。

核心语义点：

| 语义项 | 约定 |
| --- | --- |
| 归约公式 | `result = Σ ( |Re(x[i*incx])| + |Im(x[i*incx])| )` |
| 输入类型 | COMPLEX64（FP32 real + FP32 im，交错排布） |
| 输出类型 | FP32 标量 |
| `n <= 0` | `*result = 0.0f`，正常返回 SUCCESS |
| `incx <= 0` | `*result = 0.0f`，正常返回 SUCCESS（与仓内 `sasum` arch22 语义一致；Netlib 对负步长语义未定义，此处统一收敛为置 0，避免越界） |
| 执行语义 | 在 `handle` 绑定的 stream 上异步下发，结果可见性由 handle 的同步机制保证 |
| 精度语义 | 浮点归约类，按 FLOAT32 混合容差验收（非 bit-exact） |

## 需求拆解

1. **接口与框架接入**：定义 `aclblasScasum` 对外签名、参数校验、handle/stream 绑定、错误码；接入仓内 tiling 框架（Host 侧 tiling 函数 + Kernel 直调）。
2. **复数拆分计算**：COMPLEX64 → 分离 real / im → 分别 `Abs` → 相加 → 归约求和。
3. **arch22 归约实现**：多核分块 partial sum + 二级归约合并；小规模退化单核，避免 partial sum 额外开销。
4. **stride 访问支持**：`incx == 1` 连续路径（高吞吐），`incx > 1` strided 路径（membase gather / 标量地址计算，禁止 SIMT）。
5. **边界与异常**：`n` 尾块、`incx` 非 1、非对齐地址、`incx <= 0` / `n <= 0` 置 0、空指针校验。
6. **精度保障**：分段子和 + 树形合并，控制长归约下的浮点漂移；给出量化容差标准。
7. **性能达标**：典型规模下 NPU Task Duration 满足 A100 cuBLAS median / 0.8 的验收线。
8. **可测性**：UT / 精度用例（对标 `cblas_scasum`）、性能用例（典型 case 表）、边界用例。
9. **兼容性**：语义对齐 arch35 `scasum_*` 与 CBLAS，不影响 `sasum` 等既有算子与 tilingkey 兼容性。

# 详细设计（required）

## 算子分析

### 数学公式

$$
\text{result} = \sum_{i=0}^{n-1} \Big( \big| \mathrm{Re}(x[i\cdot \mathrm{incx}]) \big| + \big| \mathrm{Im}(x[i\cdot \mathrm{incx}]) \big| \Big)
$$

其中 `x` 为 COMPLEX64 向量，`x[k] = (re_k, im_k)`，`n` 为逻辑元素个数，`incx` 为元素步长（单位：复数元素）。特殊情形：

- `n <= 0` 或 `incx <= 0`：`result = 0`；
- 单元素（`n == 1`）：`result = |re_0| + |im_0|`；
- 全零输入：`result = 0`。

误差传播特性（用于精度容差推导）：每个元素贡献非负，累加结果满足 `0 <= result`，且 `result <= n * max_k(|re_k| + |im_k|)`；累加误差量级约 `O(n·ε_fp32)`，因此容差需随 `n` 适度放缩。

### 支持数据类型

| 项目 | 类型 | 说明 |
| --- | --- | --- |
| 输入 `x` | COMPLEX64 | FP32 real + FP32 im 交错，单元素 8 字节 |
| 输出 `result` | FP32（`float*`） | 标量 |
| 累加器 | FP32 | 向量寄存器与 partial sum 均为 FP32 |
| 计数/索引 | INT32 | `n` / `incx` 为 `int`，内部索引用 INT32/INT64 混合防溢出 |

不支持：COMPLEX32（FP16 复数）、FP16/BF16 输入、FP64 输出、混合精度输入输出。

### 支持形状

| 维度 | 取值范围 | 说明 |
| --- | --- | --- |
| `n` | `[0, 2^31-1]` | 逻辑元素个数；`n == 0` 合法，输出 0 |
| `incx` | `[1, 2^31-1]`；`<= 0` 触发置 0 语义 | 步长单位为复数元素 |
| 实际访存跨度 | `(n-1) * incx * 8` 字节 | 需落在 `x` 的合法 buffer 内 |
| 向量形态 | 一维 | 无 batch / 无矩阵维度 |

典型规模分类（用于 tiling 分档与性能用例）：

| 档位 | 逻辑规模 | 特点 |
| --- | --- | --- |
| S | `n <= 4K` | 延迟敏感，单核或少核 |
| M | `4K < n <= 1M` | 常规多核 |
| L | `n > 1M`（至 16M 及以上） | 吞吐敏感，双缓冲 + 二级归约 |

### 接口定义

**函数签名**

`aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);`

**参数表**

| 参数 | 方向 | 类型 | 含义 |
| --- | --- | --- | --- |
| `handle` | in | `aclblasHandle_t` | 句柄，绑定目标 stream 与设备上下文 |
| `n` | in | `int` | 逻辑元素个数 |
| `x` | in | `const aclblasComplex*` | 输入复数向量首地址（device global） |
| `incx` | in | `int` | 元素步长；`<= 0` 时输出置 0 |
| `result` | out | `float*` | 输出标量地址（device global，由调用方保证可写、非空） |

**返回值 `aclblasStatus_t`**

| 返回值 | 触发条件 |
| --- | --- |
| `ACLBLAS_STATUS_SUCCESS` | 正常（含 `n <= 0` / `incx <= 0` 置 0 路径） |
| `ACLBLAS_STATUS_INVALID_PARAM` | `handle` 无效、`x` 或 `result` 为空指针、`result` 与 `x` 区域重叠 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 数据类型/平台不匹配（非 COMPLEX64、非 arch22 目标设备） |
| `ACLBLAS_STATUS_RUNTIME_ERROR` | kernel 下发或 stream 执行失败 |

**语义约定**：接口为异步执行语义，函数返回不代表结果已写入；结果可见性由 handle 对应 stream 的同步（`aclblasSynchronize` / `aclrtSynchronizeStream`）保证。同一 handle 上的多次调用按 stream 顺序执行。

## 算子实现

### 实现方案

候选方案对比：

| 方案 | 描述 | 评估 |
| --- | --- | --- |
| A. 单核顺序累加 | 单 VectorCore 全量串行 `Abs + Add + ReduceSum` | 实现最简，但大 `n` 下吞吐受限，不满足性能线 |
| B. 多核分块 + 二级归约（**采用**） | 各核独立计算局部和 partial sum，再由收尾归约核做树形合并 | 吞吐高、精度可控、与 `sasum` arch22 范式一致 |
| C. 全局原子累加 | 各核 `atomicAdd` 到同一标量 | 存在竞争与非确定性累加顺序，精度与性能均不稳定 |
| D. 专用归约硬件单元 | 依赖 SIMT / `asc_simt` | arch22 不具备，禁止使用 |

采用方案 B，整体流水：Host 校验与 tiling → 第一级 kernel（多核计算 partial sum）→ 第二级归约（同 kernel 内单核收尾 或 独立 reduce kernel）→ 写 `result`。

目录组织（与仓内一致）：

| 路径 | 内容 |
| --- | --- |
| `blas/l1/scasum/arch22/` | arch22 实现主目录 |
| `blas/l1/scasum/arch22/tiling/` | Host 侧 tiling、分核、tilingkey 生成 |
| `blas/l1/scasum/arch22/kernel/` | Ascend C kernel 实现 |
| `blas/l1/scasum/arch22/test/` | UT / 精度 / 性能用例 |
| 参考：`blas/l1/sasum/arch22/` | 实数版本范式复用源 |
| 参考：`blas/l1/scasum/arch35/` | 语义与接口对齐参考 |

#### Host侧设计

##### 1. 分核策略

- **核数来源**：从设备 platform 信息读取可用 VectorCore 数量（如 910B3 上的 `ubNumCores`），不硬编码常量，保证 A2/A3 各型号适配。
- **负载均分**：以"访问的物理元素数" `span = n * incx` 为基准做分块；每核负责一段**逻辑元素区间** `[i0, i1)`，段长度向上取整到对齐粒度（32B / 64B），保证各核访存块连续、避免核间边界错位。
- **小规模退化**：当 `n` 落在 S 档（如 `n <= 4K`）或 `span` 小于阈值时，核数退化为 1（或 2），省去 partial sum 的额外 kernel/访存开销，优先降低延迟。
- **归约合并**：第一级各核写出 partial sum（每核一个 FP32，可打包连续写入 GM 的 partial 缓冲区）；第二级由单个核（或单独的 reduce kernel）做树形/分段合并后写 `result`。对 M/L 档采用"每核 partial + 单核合并"两级结构，避免全局同步点。
- **奇偶/对齐**：分块边界对齐到向量宽度整倍数，尾块由 mask 或标量收尾，保证无重复、无遗漏。

##### 2. 数据分块和内存优化策略

- **UB 分块**：每核按 UB 容量切分为 `tileSize` 个复数元素的块，块大小取 `min(UB可用容量 / 双缓冲系数, 对齐上限)`，典型按 2 的幂选取以简化地址计算。
- **双缓冲（double buffer）**：L 档启用，`DataCopy` 与计算流水重叠，隐藏 GM→UB 延迟；S 档关闭以降低启动开销。
- **复数拆分策略**：
  - `incx == 1`：连续装载 COMPLEX64 块，按交错布局拆出 real / im 两个 FP32 序列（视布局与指令能力选择"整体装载后交错抽取"或"按 half 偏移分别取实/虚部"），随后 `Abs` + `Add` 得到元素贡献序列，再做归约；
  - `incx > 1`：使用 membase gather 或标量地址计算按步长取数，**不使用 SIMT / `asc_simt`**；gather 时按 stride 生成地址序列，装载到 UB 后走与连续路径相同的 `Abs/Add/ReduceSum` 流程。
- **分段子和（partial within core）**：向量寄存器内不做一次性长归约，而是每 128/256 元素求一次子和再统一合并（pairwise / 分段 ReduceSum），既提升指令级并行度，也把累加误差从 `O(nε)` 压到近似 pairwise 的 `O(log n · ε)` 量级。
- **内存与带宽**：输入只读一遍（bandwidth bound），不做输入缓存复制；partial sum 缓冲区大小为 `核数 * 8B` 级别，开销可忽略；`result` 单写，无竞争。
- **访存对齐**：首地址按 32B/64B 对齐时走对齐拷贝路径；不对齐时退化为安全装载路径（按尾块 mask 处理），保证正确性优先。

##### 3. tilingkey规划策略

tilingkey 以位段编码"路径选择 + 规模档位 + 优化开关"，保证前向兼容（新增位段向高位扩展）：

| 位段 | 含义 | 取值 |
| --- | --- | --- |
| `[1:0]` | 访存路径 | `00` 连续（`incx==1`）、`01` strided（`incx>1`）、`10` 置 0 路径（`n<=0` 或 `incx<=0`）、`11` 保留 |
| `[3:2]` | 规模档位 | `00` S、`01` M、`10` L、`11` 保留 |
| `[4]` | 双缓冲 | 0 关闭 / 1 开启 |
| `[5]` | 归约层级 | 0 单级（单核直出）/ 1 两级（partial + merge） |
| `[6]` | 对齐模式 | 0 非对齐路径 / 1 对齐路径 |
| `[7]` | 归约精度策略 | 0 常规分段子和 / 1 增强（子和密度更高，用于超大 `n`） |
| `[8:]` | 保留扩展 | 置 0 |

关键组合预期：`incx==1 + L + 双缓冲 + 两级归约 + 对齐` 为主力高性能路径；`strided` 与 `非对齐` 为兼容路径；置 0 路径不下发 kernel 或下发极轻量 kernel 写 0，保证统一的异步语义。

#### Kernel侧设计

- **kernel 组成**（Ascend C，kernel 直调）：
  1. **主计算 kernel**：按 tiling 参数划分的逻辑区间装载输入 → 拆 real/im → `Abs` → `Add` → 分段子和 → 核内 `ReduceSum` → 将 partial sum 写入 GM 临时缓冲（或按配置直接由收尾核读取）。
  2. **归约收尾**：单核读取 partial 缓冲 → 树形/分段合并 → 写 `result`。S 档可在主 kernel 内由 0 号核直接完成，减少一次 kernel 启动。
- **指令与 API 使用**：使用 Ascend C 的搬运（`DataCopy`/gather 模式）、二元/一元向量计算（Add、Abs 或 `Sub`/比较选通实现的绝对值）、归约（`ReduceSum`）、同步（`pipe_barrier`、`SyncAll`）等接口；stride 访问使用 membase gather 或标量地址计算，**不引入 AscendC SIMT / `asc_simt`**。
- **Abs 实现**：对 FP32 直接使用向量 Abs；若目标指令缺失，退化为与 0 比较取反/选择的等价向量序列，结果与 IEEE `fabs` 一致（符号位清零）。
- **边界处理**：尾块不足一个向量宽度时用 mask/标量路径处理；strided 路径按元素循环或按 gather 块处理，确保访问跨度 `(n-1)*incx*8` 不越界。
- **精度控制**：核内采用分段子和 + 树形合并；不使用全局原子累加，保证同输入同配置下结果可复现（确定性执行）。
- **同步与写回**：`result` 写入完成后 kernel 结束，可见性由 handle 的 stream 同步语义保证；置 0 路径同样通过 stream 上的写 0 操作保持异步一致性。

## 支持硬件

| 硬件 | 架构 | 支持状态 | 备注 |
| --- | --- | --- | --- |
| Atlas 800I A2 | arch22（910B） | ✔ 支持 | 推理场景部署 |
| Atlas 800T A2 | arch22（910B） | ✔ 支持 | **性能测试设备：Atlas 800T A2 (910B3)** |
| Atlas A3 | arch22 | ✔ 支持 | 复用 arch22 kernel 与 tiling |
| 其它


<!-- design revision: CLA single-commit / design.md only -->
