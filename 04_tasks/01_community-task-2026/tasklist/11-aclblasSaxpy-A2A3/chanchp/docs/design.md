# 需求背景

## 需求来源

本需求来自 2026 年 11 月 ops-blas 社区任务讨论（仓库：https://gitcode.com/cann/ops-blas），任务名称为「aclblasSaxpy 算子开发（A2/A3）」。该任务属于 BLAS L1（向量-向量）系列在昇腾 Atlas A2 / A3 系列硬件（架构代号 arch22 / 910B）上的算子补齐工作，工程模式为 aclblas 句柄式 kernel 直调（基于 Ascend C 编程范式），目标是固化连续（unit stride）与 stride 两种访问模式下的 Saxpy 实现方案，并形成社区验收口径。

## 背景介绍

Saxpy（Single-precision A·X Plus Y）是 BLAS Level 1 中最基础、调用频度最高的算子之一，广泛用于 Krylov 子空间迭代（CG / GMRES / BiCGStab）、稀疏矩阵求解器、神经网络优化器中的动量更新与权重衰减等场景。其访存特征为纯带宽受限（memory-bound）：3 次向量访存（读 x、读 y、写 y）对应 2 flops，因此实现质量主要由向量化访存、流水编排与核间负载均衡决定。

仓库 arch22 目录下已存在 saxpy_* 系列实现（含 stride 变体），本设计文档按社区验收口径对连续与 stride 两类方案进行设计固化：

- **连续模式（incx == incy == 1）**：使用 Ascend C 的 MulAdd / 向量化接口做 chunk 化流水，追求带宽打满。
- **stride 模式（incx > 1 或 incy > 1）**：使用 membase gather / window merge（拼窗）方式组织非连续访存，**明确禁止使用 Ascend C SIMT / asc_simt 接口**（该能力在 A5 架构才引入，A2/A3 不可用），跨步地址计算一律通过标量地址推导 + membase gather 完成。
- **退化情形**：n <= 0 时为 no-op，直接返回成功，不下发 kernel。

性能验收口径为：常用规模下 NPU Task Duration 中位数 ≤ A100（cuBLAS）对应 case 耗时中位数 / 0.8，即相对 A100 达到 0.8 倍的性能基线要求。

# 需求分析

## 需求描述

实现符合 aclblas 接口规范的单精度 Saxpy 算子，计算定义为：

```
y[i] = alpha * x[i] + y[i],  i = 0, 1, ..., n-1
```

对外接口签名：

```
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle,
                             int n,
                             const float* alpha,
                             float* x, int incx,
                             float* y, int incy);
```

接口语义与 cblas_saxpy（CblasColMajor / CblasRowMajor 无差别，纯向量）对齐：alpha 为标量指针（允许指向 device 或 host 内存，按 handle 约定解析，本实现按 device 指针 + 单次 scalar load 处理，并对 host 侧传值场景做兼容拷贝）；x、y 为 device 侧数据指针；incx / incy 为元素步长（>0）。需求同时覆盖连续访问与非连续（stride）访问两种数据布局。

## 需求拆解

| 编号 | 拆解项 | 说明 | 优先级 |
| --- | --- | --- | --- |
| R1 | API 语义对齐 | 返回值、参数含义与 cblas/Netlib saxpy 一致；n <= 0 时 no-op 返回 ACLBLAS_STATUS_SUCCESS | P0 |
| R2 | FP32 计算正确性 | y = alpha*x + y，精度对标 cblas/Netlib，按 FLOAT32 混合容差验收 | P0 |
| R3 | 连续路径 | incx == 1 且 incy == 1，向量化 MulAdd 流水实现 | P0 |
| R4 | stride 路径 | incx > 1 或 incy > 1，membase gather / window merge 实现，禁止 SIMT | P0 |
| R5 | 大规模分核与 tiling | 支持 n 从 1 到 1e8 量级，多核负载均衡，tilingkey 区分连续/stride 与规模档位 | P0 |
| R6 | 精度口径固化 | 归约索引类 bit-exact；浮点类按 FLOAT32 混合容差 | P1 |
| R7 | 性能达标 | 常用 case NPU Task Duration ≤ A100 cuBLAS median / 0.8 | P1 |
| R8 | 可维可测 | 用例覆盖边界 n、stride、alpha 取值；日志/错误码可定位；文档与代码同目录归档 | P1 |

# 详细设计

## 算子分析

### 数学公式

$$
y_i \leftarrow \alpha \cdot x_i + y_i, \quad i = 0, 1, \dots, n-1
$$

其中向量下标映射到内存地址为：

$$
x_i \rightarrow x + i \cdot incx \cdot sizeof(float), \qquad
y_i \rightarrow y + i \cdot incy \cdot sizeof(float)
$$

性质分析：

- 逐元素独立，无跨元素依赖，天然可按块并行；无归约（reduction）操作，因此**不存在浮点归约次序问题**，元素级结果的误差来源仅为 alpha*x 与 y 相加的一次 FMA 舍入。
- 计算强度为 2 flops / 12 Bytes（连续）或更差（stride），属纯带宽受限算子，性能目标等价于逼近 HBM 有效带宽。
- 数学恒等式：当 alpha == 0 时理论上 y 不变，但仍按标准实现执行乘加（不做特判跳过），以保证与 cblas 行为一致且便于性能对比；该行为写入算子约束。

### 支持数据类型

| 数据类型 | x | y | alpha | 计算精度 | 状态 |
| --- | --- | --- | --- | --- | --- |
| FLOAT32（FP32） | float | float | float | FP32（可用 FP32 FMA） | 支持 |
| FLOAT16 / BFLOAT16 | - | - | - | - | 暂不支持（后续任务） |
| FP64 | - | - | - | - | 暂不支持 |

说明：x 与 y 的数据类型必须一致，均为 FP32；alpha 为 FP32 标量。接口层对类型做校验，非法类型返回 ACLBLAS_STATUS_NOT_SUPPORTED。

### 支持形状

| 维度 | 约束 | 说明 |
| --- | --- | --- |
| n | int，n <= 0 为 no-op | 正常规模覆盖 n ∈ [1, 2^31-1]；实际实现按 size_t 做地址偏移计算防溢出 |
| incx | int，incx >= 1 | incx == 1 视为连续；incx > 1 走 stride 路径 |
| incy | int，incy >= 1 | incy == 1 视为连续；incy > 1 走 stride 路径 |
| 路径判定 | incx == 1 && incy == 1 → 连续路径；否则 → stride 路径 | 两者之一为 1 时仍走 stride 路径，但可启用单侧连续的优化（见实现方案） |

元素逻辑序号 i 的有效访问范围：x 最大地址偏移为 (n-1)*incx*4 Bytes，y 为 (n-1)*incy*4 Bytes，host 侧需据此做地址越界校验（可选，debug 版开启）。

### 接口定义

```
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle,
                             int n,
                             const float* alpha,
                             float* x, int incx,
                             float* y, int incy);
```

参数说明：

| 参数 | 输入/输出 | 类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | aclblas 句柄，携带 stream / device 信息与 kernel 直调上下文 |
| n | 输入 | int | 向量长度，n <= 0 为 no-op |
| alpha | 输入 | const float* | 标量系数指针，单次读取后在 kernel 内广播 |
| x | 输入 | float* | 源向量 device 指针，本算子不修改 x（接口签名沿用 aclblas 历史形式） |
| incx | 输入 | int | x 的元素步长，>= 1 |
| y | 输入/输出 | float* | 目标向量 device 指针，就地更新 |
| incy | 输入 | int | y 的元素步长，>= 1 |

返回值：

| 返回值 | 触发条件 |
| --- | --- |
| ACLBLAS_STATUS_SUCCESS | 正常完成，或 n <= 0 的 no-op |
| ACLBLAS_STATUS_INVALID_PARAM | handle / alpha / x / y 为空（n>0 时），incx < 1 或 incy < 1 |
| ACLBLAS_STATUS_NOT_SUPPORTED | 数据类型或架构不支持 |
| ACLBLAS_STATUS_RUNTIME_ERROR | kernel 下发 / 同步失败 |

调用约束：x 与 y 不允许重叠（non-overlapping），重叠行为未定义，与 cblas 一致。

## 算子实现

整体代码归档路径：`blas/l1/saxpy/arch22/`（family=l1，算子目录 saxpy，硬件目录 arch22）。目录内包含 host 侧 tiling 与 kernel 直调入口、kernel 侧 Ascend C 实现、tiling key 定义头文件、utest 与性能用例脚本。

### 实现方案

#### Host侧设计

##### 1. 分核策略

- **核数选择**：以"每核处理的 FP32 元素数"为基准做分档，目标是让单核工作量落在可流水、可向量化的区间（每核元素数目标区间 [8K, 64K]，具体按实测校准）。实际启用核数 = min(aivNum, ceil(n / chunkElem))，其中 chunkElem 为单核基础块元素数。
- **负载均衡**：将逻辑元素序号 [0, n) 均匀切分为 aivNum 段（blockNum 个），每段长度为 ceil(n / blockNum)，最后一核接收尾差。由于 Saxpy 每元素代价恒定（无稀疏/分支），均匀切分即最优，无需动态负载均衡。
- **地址计算**：核内起始元素偏移 s = blockId * segLen，实际起点地址为 x + s*incx、y + s*incy；由于 stride 路径下相邻核的地址跨度不连续，host 仅传递基址、incx/incy、n、alpha 与本核起始序号，由 kernel 标量推导地址，避免传大数组。
- **流水层级**：连续路径按 chunk（如 512~2048 个 FP32）分为 multi-buffer 流水段，取数（Gm→UB）/计算/回写（UB→Gm）重叠；stride 路径按"逻辑连续窗口"组织 gather，见下节。

##### 2. 数据分块和内存优化策略

- **连续路径（incx == incy == 1）**：
  - 以 UB 为缓冲，采用 double buffer / multi-buffer：一块 UB 承载 x 块与 y 块（x 块与 y 块可共用一段 UB，读 y 后与 x 做 MulAdd 再回写），块大小按 UB 容量的 1/2~1/4 取整到 32B 对齐。
  - 计算使用 Ascend C 向量化 MulAdd（或等价的 Mul + Add，FMA 语义），对尾部不足一个向量宽度（128B / 32 个 FP32）的零头，按掩码式拷贝补齐（拷贝长度上取整到对齐粒度后再回写有效长度），保证不越界访存。
  - 访存对齐：主循环块起始地址按 32B 对齐处理；若基址不对齐，先用标量/小块处理头部不对齐部分，主循环进入对齐快路径。
- **stride 路径（incx > 1 或 incy > 1）**：
  - **禁止使用 SIMT/asc_simt**，采用 **membase gather / window merge** 方案：将逻辑连续的一段元素（window，例如 w 个元素）在 Gm 上的离散地址，通过 membase 基址 + 标量地址序列做 gather 拉入 UB，拼成 UB 内连续窗口（window merge）；计算后按 y 的 stride 用 scatter/拼窗逆向回写。
  - 窗口长度 w 的选择准则：使一次 gather 的地址数落在硬件 gather 原语的最优批量区间，同时保证 UB 内 x_win、y_win、out 三段 buffer 放得下（w*4B*3 <= UB 可用容量）；w 取 2 的幂并对齐到 32B。
  - **单侧连续优化**：若 incx == 1 且 incy > 1，x 侧用直接连续拷贝，仅 y 侧走 scatter；若 incy == 1 且 incx > 1，y 侧连续回写，仅 x 侧走 gather。该分支在 host 侧以子 tilingkey 区分。
  - **大 inc 退化**：当 incx 或 incy 很大（元素间距超过一定阈值，如 > 64B）时，gather 的空间局部性退化为近似随机访问，此时缩小 window 并提高并发窗口数，以隐藏访存延迟；同时按行块切分保证 TLB/页局部性。
  - **地址计算**：全部用标量（index → offset）方式推导每个窗口内元素的 Gm 偏移，写入 UB 的地址序列区（或按 membase gather 接口要求的描述符形式），不使用任何依赖 SIMT 的 per-lane 地址生成。
- **alpha 处理**：alpha 单值在 host 侧确认可取后，kernel 首条指令以标量方式读入并 broadcast 到向量寄存器，全程复用，不重复访存。
- **无归约**：本算子不涉及 reduce，无需 pairwise/树形求和，也不需要额外的精度补偿 buffer。

##### 3. tilingkey规划策略

tilingkey 用位段编码：`[路径类型][单侧连续][规模档][对齐档]`，共 64bit 中取低位若干段。规划如下：

| tilingkey | 路径 | 适用条件 | 说明 |
| --- | --- | --- | --- |
| K0 | 连续 · 标准 | incx==1 && incy==1，n <= 64K | 单核或少核，不做复杂分块 |
| K1 | 连续 · 大规模 | incx==1 && incy==1，n > 64K | 多核 + multi-buffer 流水 |
| K2 | 连续 · 非对齐 | incx==1 && incy==1，基址非 32B 对齐 | 头部标量对齐处理 + 主循环 |
| K3 | stride · 双侧 | incx>1 && incy>1 | gather + scatter 双侧拼窗 |
| K4 | stride · x连续 | incx==1 && incy>1 | 直接拷贝 x，scatter y |
| K5 | stride · y连续 | incx>1 && incy==1 | gather x，直接拷贝 y |
| K6 | stride · 大inc | incx>阈值 \|\| incy>阈值 | 小窗口多并发，延迟隐藏 |
| K7 | no-op | n <= 0 | host 直接返回，不下发 kernel |

规模档阈值（64K 等）与窗口大小、块大小在 tiling 参数中以常量表形式给出，便于按 arch22 实测微调，不改变 tilingkey 语义。tiling 信息通过 tiling 结构体（含 n、incx、incy、alpha、基址、核内起始序号、chunk 数、window 长度、对齐偏移）随 kernel 参数下发。

#### Kernel侧设计

- **入口与数据流**：每个 AIV（AI Vector Core）执行同一 kernel，按 blockId 计算本核逻辑区间；使用 Ascend C 的流水基本单位（DataCopy / 拼窗 gather 接口 / MulAdd）组织取数—计算—回写三级流水，UB 采用 double buffer。
- **连续路径 kernel 流程**：
  1. 标量读取 alpha 并 broadcast；
  2. 计算本核起止元素序号与 Gm 偏移，处理不对齐头部；
  3. 主循环：Gm→UB 载入 x 块与 y 块（多 buffer 交替），向量 MulAdd，UB→Gm 回写 y 块；
  4. 尾块按有效长度补齐处理。
- **stride 路径 kernel 流程**：
  1. 标量读取 alpha 并 broadcast；
  2. 按 window 划分本核逻辑区间，标量生成本窗口内 x / y 的 Gm 偏移序列；
  3. 用 membase gather 将 x_win、y_win 拉入 UB 并 merge 成连续窗口（单侧连续时跳过对应侧）；
  4. UB 内向量 MulAdd；
  5. 按 y 的 stride scatter/拼窗逆向回写 y_win；
  6. 推进窗口直至本核区间结束。
- **禁止事项**：不使用 AscendC SIMT / asc_simt 任何接口；跨步地址一律标量推导 + membase gather；不做依赖 per-lane 标量分支的实现。
- **边界与安全**：所有 UB 拷贝长度上取整到 32B 对齐但严格受本核区间约束，回写仅覆盖 [s, e) 内的 y 元素；n<=0 情形由 host 拦截，kernel 不处理；尾块通过长度参数而非分支掩码控制（A2 上掩码写由长度截断实现）。
- **精度处理**：单次 FMA 完成 alpha*x+y，与 cblas 的 FP32 逐元素运算链一致；不引入中间精度放大（不使用 FP32 累加器阵列或多步补偿），保证与参考实现误差在容差内且无可累积误差源。

## 支持硬件

| 硬型 | 架构 | 状态 |
| --- | --- | --- |
| Atlas 800I A2（910B） | arch22 | ✔ 支持 |
| Atlas 800T A2（910B3） | arch22 | ✔ 支持（性能测试设备） |
| Atlas A3 | arch22 | ✔ 支持 |
| Atlas A5 及以后 | arch 系列 | ✘ 暂不支持（SIMT 能力不在本设计范围，另立任务） |

实现按 arch22 编译目标组织于 `blas/l1/saxpy/arch22/`，A3 与 A2 同属 arch22 编码路径，共用同一 kernel 与 tiling 逻辑，仅在核数/UB 容量等硬件参数上通过 capability 查询自适应。

## 算子约束限制

1. 仅支持 FP32（x、y、alpha 均为 float），不支持 FP16 / BF16 / FP64 混合类型。
2. incx、incy 必须 >= 1；incx < 1 或 incy < 1 返回 ACLBLAS_STATUS_INVALID_PARAM（本设计不含 BLAS 标准中 inc <= 0 的历史扩展语义）。
3. n <= 0 为 no-op，直接返回成功，不下发 kernel；n 的合法上限为 2^31-1，实际地址偏移按 64 位计算。
4. x 与 y 所指内存不允许重叠，重叠行为未定义。
5. alpha == 0 时仍执行乘加，不做跳过特判；NaN / Inf 按 IEEE754 默认传播，不做特殊处理。
6. x、y 的实际分配长度必须分别不小于 (n-1)*incx+1 与 (n-1)*incy+1 个元素，宿主需保证不越界，算子不做运行期全量校验（debug 版可开启）。
7. 硬件仅支持 Atlas A2 / A3（arch22）；禁止依赖 Ascend C SIMT / asc_simt 能力。
8. handle 与 stream 的绑定关系由调用方维护，算子内不创建/销毁 stream；并发调用同一 handle 的同步语义由 handle 层保证。
9. 本算子为 in-place 更新 y，x 只读；调用方需保证 kernel 执行期间 x、y 不被其他流并发写。

# 可维可测分析

## 精度标准/性能标准

**精度标准**（对标 cblas_saxpy / Netlib saxpy）：

| 类别 | 标准 | 说明 |
| --- | --- | --- |
| 归约/索引类 | bit-exact | 本算子无归约；索引、地址、长度、tiling 切分结果必须与参考完全一致 |
| 浮点类 | FLOAT32 混合容差 | 逐元素与 cblas 结果比较，采用 abs_err <= 1e-6 或 rel_err <= 1e-6（混合判据 abs(a-b) <= atol + rtol*|b|，atol=rtol=1e-6）；因单次 FMA，实测通常可达 bit-exact 或 1 ulp 级 |
| 特殊值 | 与 cblas 行为一致 | NaN / Inf / alpha=0 / 超大 inc 场景用例逐一比对 |

**性能标准**：性能测试设备为 **Atlas 800T A2（910B3）**。验收方式为：对每个典型 case，采集 NPU Task Duration（取 >=50 次迭代的中位数），对比 A100 上 cuBLAS 对应调用的耗时中位数（gpu_ms），要求：

```
npu_ms <= gpu_ms / 0.8
```

即达到 A100 cuBLAS 0.8 倍性能基线。典型 case 列表：

| Case | 规模 | incx / incy | 路径 | 备注 |
| --- | --- | --- | --- | --- |
| P1 | n = 1,048,576（1M） | 1 / 1 | 连续 | 中等规模带宽基准 |
| P2 | n = 4,194,304（4M） | 1 / 1 | 连续 | 常用规模主验收 |
| P3 | n = 16,777,216（16M） | 1 / 1 | 连续 | 大规模带宽上限 |
| P4 | n = 16M | 1 / 4 | stride（x连续） | 单侧连续 K4 |
| P5 | n = 16M | 4 / 1 | stride（y连续） | 单侧连续 K5 |
| P6 | n = 16M | 4 / 4 | stride 双侧 | gather + scatter K3 |
| P7 | n = 4M | 16 / 16 | 大 inc | K6 延迟隐藏路径 |
| P8 | n = 256 | 1 / 1 | 连续 · 小规模 | 时延敏感，关注 launch 开销占比 |

小规模 case（P8）对 kernel launch 开销敏感，单独统计并允许单独给出结论；主验收为 P1~P7。性能脚本与基线数据存放于 `blas/l1/saxpy/arch22/` 的 perf 用例目录，输出格式包含 npu_ms_median、gpu_ms_median、ratio（= npu_ms / gpu_ms）与是否通过（ratio <= 1/0.8）。

## 兼容性分析

- **接口兼容性**：`aclblasSaxpy` 签名与语义完全对齐既有 aclblas 规范与 cblas_saxpy，存量调用方无需改造；x 参数保持 const 语义（只读），与历史签名一致，不做收紧。
- **实现兼容性**：arch22 已有 saxpy_*（含 stride）实现，本设计对其方案（连续 MulAdd、stride gather/window merge）进行口径固化，代码目录 `blas/l1/saxpy/arch22/` 内保持命名与 tilingkey 语义向后兼容；新增 tilingkey 档位不改变既有档位含义，旧档位取值保留。
- **硬件兼容性**：同一 arch22 编码路径覆盖 910B（Atlas 800I/T A2）与 Atlas A3，通过运行期 capability 查询适配核数、UB 容量差异，不做硬编码；不依赖 A5 才具备的 SIMT 能力，保证 A2/A3 上可编译、可运行。
- **依赖兼容性**：仅依赖 Ascend C 基础接口（DataCopy、向量 MulAdd、membase gather / 拼窗类接口）与 aclblas handle 层，不引入额外第三方库；精度参考实现依赖 cblas/Netlib 仅用于测试，不进入产品依赖链。
- **升级影响**：未来扩展 FP16/BF16/FP64 或 A5 SIMT 优化时，新增独立 family/硬件目录（如 `blas/l1/saxpy/arch_next/`），不影响 arch22 现有行为与验收口径。


<!-- design revision: CLA single-commit / design.md only -->
