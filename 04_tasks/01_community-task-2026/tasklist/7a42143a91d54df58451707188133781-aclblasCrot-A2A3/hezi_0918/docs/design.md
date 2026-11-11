# aclblasCrot 算子设计文档（arch22，Atlas A2/A3）

> 11月社区任务-aclblasCrot算子开发（A2/A3）设计文档。本 PR 仅含本文件一个新增 md；
> 算子实现与测试代码后续按任务书合入 [ops-blas](https://gitcode.com/cann/ops-blas)
> （`blas/rot/arch22/`、`test/rot/crot/arch22/`）。性能数字以上机实测为准【待填】，不预填。

# 需求背景（required）

## 需求来源

- 任务：[11月社区任务-aclblasCrot算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/7a42143a91d54df58451707188133781)，任务 ID `7a42143a91d54df58451707188133781`
- 任务书：`aclblasCrot_AtlasA2_A3_task_doc.md`（接口契约、边界语义、精度/性能验收口径均以任务书为准）
- 对标接口：cuBLAS `cublasCrot`（语义参考 Netlib/LAPACK `crot`；参考 BLAS 库无 crot 例程，该参考实现来自 LAPACK 辅助例程）
- 目标仓：https://gitcode.com/cann/ops-blas ，实现目录 `blas/rot/arch22/`，测试目录 `test/rot/crot/arch22/`

## 背景介绍

Givens（平面）旋转是 QR 分解、Krylov 子空间迭代（GMRES 等）与 Jacobi 型消元的基本计算单元：按序作用于向量对 (x, y)，在选定平面内消去一个分量。复数版本中旋转参数为实余弦 c（float）与复正弦 s（complex64），y 的更新以 s 的共轭 conjg(s) 参与，对齐 Netlib/LAPACK crot 语义；x、y 原地更新。cuBLAS 通过 `cublasCrot` 提供该能力，ops-blas 仓 A2/A3（arch22）产品线当前空缺，本任务补齐。

**仓内现状（基于 ops-blas master 实测核实）**：

| 项 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` 尚无 `aclblasCrot` 声明（同族已有 `aclblasCsrot`、`aclblasSrot`） | 本任务在该公共头新增声明，签名对齐 cublasCrot，供其他产品线共用，禁止私有平行 API |
| arch22 现状 | `blas/rot/arch22/` 仅有 csrot 实现（复 x、实 s） | 本任务开发主体；csrot 已验证的 AIV 双路径范式是 arch22 侧直接参照 |
| arch35 参考 | `blas/rot/arch35/` 已有完整 crot 四件套（host/kernel/kernel.h/tiling_data.h） | 语义与工程结构首选参考；其 SIMD+SIMT 双 kernel 架构与 950 边界契约需按 arch22 适配 |
| 测试现状 | `test/rot/crot/` 仅有 arch35 工程，`test/rot/crot/CMakeLists.txt` 仅在 arch35 分支注册 GTest | 需新建 `test/rot/crot/arch22/` 工程并补齐 arch22 的 GTest 注册 |

**arch22 与 arch35（950）边界契约差异（双方已声明的 per-arch 差异，见 ops-blas 仓 issue #827/#780/#866）**：A2/A3 口径下 n<0、incx/incy=0、INT32_MIN 步长、(n-1)×|inc| 跨度溢出均返回 `ACLBLAS_STATUS_INVALID_VALUE`；950 口径放行 n≤0 与零步长。本设计按 A2/A3（arch22 家族）契约实现。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22，CANN 9.1.0）上基于 ops-blas 工程框架，使用 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasCrot`，对单精度复数向量 x、y 原地施加 Givens 平面旋转：

```cpp
aclblasStatus_t aclblasCrot(aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
                            aclblasComplex* y, int incy, const float* c, const aclblasComplex* s);
```

支持正/负步长、Host/Device 双位置标量（c、s）、n=0 快速返回，按 arch22 家族契约校验非法参数；精度满足 FLOAT32 混合容差标准，性能达到 GPU 标杆 0.8× 以上（Atlas 800T A2 / 910B3 实测）。

## 需求拆解

1. **参数校验（顺序固定，任务书 §2.5）**：handle 判空（`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）→ n<0（`ACLBLAS_STATUS_INVALID_VALUE`）→ n=0 快速返回 SUCCESS（不修改 x、y）→ incx/incy ∈ {0, INT32_MIN} 与 (n-1)×|inc| 跨度溢出（int64 计算，`ACLBLAS_STATUS_INVALID_VALUE`）→ n>0 时 x、y、c、s 判空（`ACLBLAS_STATUS_INVALID_VALUE`）；quick return 不得先于判空（#765/#861 维护者拍板口径）；
2. **复数旋转数学**：x'[k] = c·x[k] + s·y[j]，y'[j] = c·y[j] − conjg(s)·x[k]（y 使用旋转前的原始 x，实现侧先暂存）；1-based 兼容索引 k=1+(i-1)·incx、j=1+(i-1)·incy；
3. **标量双位置**：c、s 独立支持 Host 或 Device 指针，host 侧经 `aclrtPointerGetAttributes` 判定（对齐仓内 aclblasSrot 约束说明）；Host 指针解引用随 tiling 下发，Device 指针原样转发由 kernel 从 GM 读取，不做 D2H 同步拷贝；
4. **步长语义**：incx/incy 可正可负，负步长从向量尾端起算沿负方向步进（Netlib `IX = (-N+1)*INCX + 1` 语义）；
5. **精度**：FLOAT32 混合容差（rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP），Inf/NaN 按等效规则单独判定；golden 在测试工程内按 Netlib crot 语义实现；
6. **性能**：Atlas 800T A2（910B3），msprof op 采集 task duration，5 次 warmup + 有效采样 ≥10 次取均值，单 case 平均耗时 ≤1.25×GPU（≥0.8× 标杆）；
7. **工程接入**：`include/cann_ops_blas.h` 新增声明；`blas/rot/arch22/` 新增实现；`test/rot/crot/arch22/` 新建 CSV 驱动 GTest 工程并在 `test/rot/crot/CMakeLists.txt` 注册 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`；`blas/rot/README.md` 产品支持表标注 Atlas A2/A3：支持。

# 详细设计（required）

## 算子分析

### 数学公式

对 i = 1..n（k = 1+(i-1)·incx，j = 1+(i-1)·incy，1-based 索引兼容 Fortran）：

```text
x'[k] = c · x[k] + s · y[j]
y'[j] = c · y[j] − conjg(s) · x[k]        （使用旋转前的原始 x[k]）
```

记 c ∈ ℝ，s = sr + si·i，按实部/虚部展开（乘加序列固定，共 12 次实数乘法）：

```text
x'r = c·xr + (sr·yr − si·yi)          y'r = c·yr − (sr·xr + si·xi)
x'i = c·xi + (sr·yi + si·yr)          y'i = c·yi − (sr·xi − si·xr)
```

复数乘法 (a+bi)(c+di) = (ac−bd) + (ad+bc)i；conjg(a+bi) = a−bi；实数 c 与复数相乘为实部/虚部分别缩放。本算子为逐元素旋转，误差来源仅为两次复数乘加的浮点舍入。

### 支持数据类型

| 项 | 取值 |
| --- | --- |
| x、y | COMPLEX64（`aclblasComplex`，实部/虚部各 float32，以 `include/cann_ops_blas_common.h` 定义为准），Device 内存，原地更新 |
| c | FLOAT32 标量指针（旋转余弦），Host 或 Device 内存 |
| s | COMPLEX64 标量指针（旋转正弦），Host 或 Device 内存 |
| n、incx、incy | int32，Host 内存标量 |

### 支持形状

x、y 为逻辑一维向量 [n]（ND），物理长度 1+(n-1)×|incx|、1+(n-1)×|incy|；n 为运行时入参，host 侧 tiling 自适应，无编译期 shape 假设；向量非连续完全由 incx/incy 表达（含正、负步长），不涉及 leading dimension padding、broadcast 与动态 shape。

## 算子实现

### 实现方案

采用 Ascend C kernel 直调方式：host 侧校验参数并计算 tiling，经 handle 绑定 stream 异步下发单个 AIV kernel；`aclblasComplex` 在 GM 中为两个交错 float，kernel 侧以 float 视角按 2·idx / 2·idx+1 寻址。

#### Host 侧设计

入口 `blas/rot/arch22/crot_host.cpp`，声明新增入 `include/cann_ops_blas.h`：

1. **参数校验（顺序即任务书 §2.4/§2.5）**：handle 判空 → n<0 → n=0 快速返回 SUCCESS → incx/incy 为 0 或 INT32_MIN 拒绝 → (n-1)×|incx|、(n-1)×|incy| 以 int64 计算并比对 INT32_MAX（跨度溢出拒绝）→ n>0 时 x、y、c、s 判空。全部通过后再进入 launch；
2. **标量解析（c、s 独立处理）**：`aclrtPointerGetAttributes` 探测 c 与 s 的内存位置；Host 指针解引用一次，值随 tiling 下发（cosValue/sinRe/sinIm），转发 nullptr；Device 指针置 0 占位、指针原样转发（s 转发 &s->real 即两个交错 float），kernel 从 GM 读取，无 host 解引用、无 D2H 拷贝、无流同步；探测失败按非法输入拒绝（对齐 aclblasSrot 口径）；
3. **tiling 结构**：单控制结构体（tilingKey / totalN / incx / incy / cosValue / sinRe / sinIm / cIsDevice / sIsDevice / perCoreN / remainder 等字段），经 kernel 直调参数通道下发，host 不做额外流同步，保持接口异步语义；原地操作、无中间缓冲，workspace 需求为 0；
4. **路径分派**：incx==incy==1 走连续路径（tilingKey=0，满核均分 perCoreN/remainder）；其余任意合法步长组合（含负步长）走跨步路径（tilingKey=1）。x、y 同指针不同步长的别名场景无法安全多线程切分（迭代间读写依赖调度顺序），路由到单线程串行路径（serialOnly 标志）；部分重叠缓冲区不在契约内（与仓内同族实现口径一致）。

#### Kernel 侧设计

`blas/rot/arch22/crot_kernel.cpp`（附 `crot_kernel.h`、`crot_tiling_data.h`），单个 `__global__ __aicore__` AIV kernel 内按 tilingKey 分双路径，UB 经 `AsdopsBuffer<ArchType::ASCEND_V220>` 管理：

1. **连续路径（tilingKey=0，incx==incy==1）**：perCoreN/remainder 满核均分，每核 CopyIn → Compute → CopyOut 三段流水，UB ping-pong 双缓冲掩盖搬运延迟；复数流视为交错 float 流，x/y 整 tile 搬入 UB 后以向量级 Mul/Add/Mixup 组合完成 §算子分析 的 12 次实数乘加，newX/newY 同 tile 内基于原值计算后统一写回；
2. **跨步路径（tilingKey=1，其余步长含负步长）**：负步长按物理尾端起算，logical i → 物理索引 inc>0 时 i·inc、inc<0 时 (n-1-i)·|inc|（对齐 Netlib 语义）；每 pass 处理 1024 个复数元素：标量 GetValue 读入原 x/y（寄存器暂存保证 y 使用原始 x），旋转结果写入 32B 对齐 UB 暂存槽（每元素 8-float 槽，2 数据 + padding，满足 MTE 32B 块粒度），再以每元素 8 字节 DataCopyPad 突发回写 GM；标量 UB 写与 MTE3 读之间以全量 PIPE_BARRIER 隔离；
3. **原地安全（不变量）**：两条路径均先把原始 x[k]、y[j] 暂存（连续路径在 UB tile、跨步路径在标量寄存器）再计算两个输出，y 的更新始终消费旋转前的 x，与 Netlib crot 的 XTEMP/YTEMP 暂存语义一致；
4. **数值一致性**：乘加以固定序列执行、禁止编译器把 mul+add 收缩为 FMA（volatile 中间量或等价措施，对齐 arch35 crot 已验证做法），使 NPU 与 golden 的舍入序列一致，极端 c/s 幅值下避免灾难性消去放大单次融合舍入差；
5. **Device 标量**：cIsDevice/sIsDevice 置位时 kernel 于入口从 GM 读一次 c/s 广播使用，host 不感知取值。

### 精度设计

golden 在测试工程内按 Netlib crot 语义（XTEMP/YTEMP 暂存、conjg(s)）以 float32 实现，实部/虚部分别比对；FLOAT32 按 rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP 判定；Inf/NaN 按任务书等效规则（同为 NaN / 同号 Inf / NaN↔±Inf 等效）单独统计。逐元素旋转仅两次乘加舍入，正常输入远优于阈值，乘加序列对齐为主要数值控制手段。

### 性能设计

测试设备 Atlas 800T A2（910B3），msprof op 模式采集单算子 task duration（5 warmup 不计入 + ≥10 有效样本均值），逐 case 对照 `test_cases/gpu_baseline.csv`（≤1.25×GPU）。性能关键点：连续路径 ping-pong 双缓冲 + 满核均分 + DataCopyPad 整 tile 搬运逼近带宽上限；跨步路径 8B 突发回写与 1024 元素/pass 批处理摊薄标量访存开销；无 workspace、无 host 侧同步。分 case 实测数据【待填】（上机 msprof 采集后回填自测报告）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（含 Atlas 800T A2 / 910B3） | √ |
| Atlas A3 系列（arch22 同源实现） | √ |

软件环境：CANN 9.1.0；golden 由 cblas（Netlib BLAS 3.10.0 复数实现语义）生成口径，随测试工程提供；`blas/rot/README.md` 产品支持表标注 Atlas A2/A3：支持。

## 算子约束限制

- 校验顺序与错误码严格按任务书 §2.4/§2.5（arch22 家族契约）：n<0、incx/incy=0 或 INT32_MIN、跨度溢出、n>0 时空指针均返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n=0 合法快速返回不修改 x、y；
- 负步长从向量尾端起算；零步长与 INT32_MIN 步长不支持（与 950 口径的 per-arch 已声明差异，issue #827/#780）；
- c、s 标量指针支持 Host/Device 双位置；x、y 必须 Device 内存；
- 原地更新，不返回视图；y 依赖旋转前 x；x、y 部分重叠别名不在契约内；
- 异步执行：依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前须同步 stream；
- 不涉及 broadcast、动态 shape、随机数与确定性计算要求。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP；实部/虚部分别比对；Inf/NaN 等效判定 | 任务书 §3.2 / [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 910B3 单 case 平均耗时 ≤1.25×GPU（≥0.8× 标杆），msprof op 口径 5 warmup + ≥10 有效样本均值 | 任务书 §3.3 / test_cases/gpu_baseline.csv |

测试方案：

1. **测试工程**：`test/rot/crot/arch22/` 新建 CSV 驱动 GTest 工程（参照仓内 arch35 模式：python 脚本加载 CSV → 调用 GTest runner → golden 按 Netlib crot 语义生成、实部/虚部分别比对），并在 `test/rot/crot/CMakeLists.txt` 的 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`，补齐 A2/A3 构建的 GTest 路径（当前仅 arch35 注册）；README 说明构建与复现步骤，保证验收人可复跑；
2. **用例覆盖**：任务书自带 `crot_test.csv` 1200 条（1193 正向 + 7 负向）全跑——n 覆盖 0、1、小质数、2 的幂及 ±1 直至 ≥2^20；incx/incy 覆盖 1/2/3/-1/-2/-3；c、s 标量位置覆盖 host/device/mixed 三模式；Inf/NaN 特殊值与 c∈{1,0,-1}、s∈{(0,0),(1,0),(0,1),(-1,0)} 特殊点；负向（n<0、零步长、INT32_MIN、跨度溢出、空指针、handle 空）期望状态码按任务书 §2.4/§2.5，负向用例以单元素 dummy 缓冲占位、不按跨度预分配；自补正态分布用例（GAUSS_NORM_5_2：μ∈[-5,5]、σ∈[0.1,2]，gen_csv.py 默认仅均匀分布）并在自测报告说明补齐方式与数量；
3. **性能自测**：`gpu_baseline.csv` 200 条全跑 + 自补规模扫描，msprof op 采集，异常样本（非 SUCCESS）剔除并说明；输出自测报告（用例参数、精度对比截图、性能数据截图），内存要求按任务书 §3.4 不涉及。

## 兼容性分析

- `include/cann_ops_blas.h` 仅新增 `aclblasCrot` 声明，风格与同族 `aclblasCsrot`/`aclblasSrot` 一致，不修改既有声明与实现，供其他产品线共用；状态码、handle/stream 约定全部复用 ops-blas 公共层（`cann_ops_blas_common.h`、公共 handle 内部设施），不影响其他算子路径；
- arch22 边界契约（n<0 / 零步长 / INT32_MIN 拒绝）与 950（arch35）口径的差异为仓内已声明的 per-arch 差异（issue #827、#780），`blas/rot/README.md` 的 per-arch 差异表随本任务更新标注；
- 本实现为独立撰写：`blas/rot/arch35/` 既有 crot 与 `blas/rot/arch22/` 既有 csrot 仅作语义/结构参照（双路径分派、原地安全不变量、乘加序列对齐、32B 对齐暂存），不复制代码；arch35 的 SIMT 架构不适用于本路径，kernel 采用 arch22 AIV SIMD 范式重写；
- 新算子、无存量调用方，不涉及向后兼容。
