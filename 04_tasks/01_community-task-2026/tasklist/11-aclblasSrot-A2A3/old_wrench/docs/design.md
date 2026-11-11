# aclblasSrot 算子设计文档

# 需求背景（required）

## 需求来源

- 任务：2026 年 11 月社区任务「aclblasSrot 算子开发（Atlas A2/A3）」
- 合入仓与目录：`https://gitcode.com/cann/ops-blas`（开发分支 master），实现代码合入 `blas/rot/arch22/`，测试代码合入 `test/rot/srot/arch22/`（任务书 §5）
- 验收流程：cann 社区任务（hiascend.com 社区任务开发-验收流程），设计文档先于代码提交

## 背景介绍

### 算子功能定位

aclblasSrot 是 BLAS Level-1 Givens 平面旋转算子：对两个等长实数向量 x、y 施加平面旋转矩阵 G = [[c, s], [−s, c]]，**原地更新**（对标 cuBLAS `cublasSrot`，cuBLAS Library §2.5.8）。常用于 QR 分解、线性方程组求解与特征值计算等数值算法。本任务补齐 ops-blas 仓 Atlas A2/A3 系列（arch22）的 BLAS-1 能力缺口。

### 业界对标

| 对标对象 | 语义依据 | 差异点 |
|---|---|---|
| cuBLAS `cublasSrot(handle, n, x, incx, y, incy, c, s)`（§2.5.8） | 参数序列、返回码与快速返回口径的唯一依据 | A2/A3 侧返回码与 cuBLAS 口径一致：n<0、incx/incy==0 → INVALID_VALUE |
| Netlib BLAS `srot.f` | golden 参考语义（精度真值） | netlib 对 `N.LE.0` 直接 RETURN；仓内 950PR（arch35）采此口径；A2/A3 按 per-arch 差异表取 INVALID_VALUE（见下） |
| 仓内 950PR 既有实现 `blas/rot/arch35/srot_*` | 行为参照（入口校验 L242-251、strided 溢出守卫 L116-129） | arch35 放行 n<0 与零步长；**arch22 与之相反**，属仓内已声明的 per-arch 差异（`blas/rot/README.md`「跨 arch 入参返回码对照」，同族 aclblasCsrot 先例） |

### 工程形态

- 工程模式：**kernel 直调**（Host C API 入口 + host 侧参数校验与 tiling 下发 + device kernel），与 `blas/rot/` 既有 arch22/arch35 实现同一工程模式（任务书 §2.2）
- 代码落位：`blas/rot/arch22/`（srot_host.cpp / srot_kernel.cpp / srot_kernel.h / srot_tiling_data.h 四件，对齐同目录 arch22 兄弟 caxpy 的文件布局）
- 仓内近亲：主骨架 = `blas/rot/arch22/csrot_{host,kernel}.cpp`（同族同 arch，工程结构与校验链直接对齐）；行为模板 = `blas/rot/arch35/srot_*`（同算子同 dtype，c/s 指针处理与溢出守卫参照）

### 接口规格

接口声明已存在于 `include/cann_ops_blas.h` L459（与 950PR arch35 实现共用声明），本任务**零公共头改动**、不另立私有平行接口（任务书 §2.3、§7.1）：

```cpp
aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n, float* x, int incx,
                            float* y, int incy, const float* c, const float* s);
```

参数规格（固定表）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（内部携带 stream），Host 内存 | aclblasHandle_t | - | aclblasCreate 返回的非空句柄；为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 向量元素个数，Host 内存 | int | INT32 | n=0 快速返回 SUCCESS；n<0 返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| x | 输入兼输出（原地更新）向量，Device 内存 | float* | FLOAT32 | n>0 时 nullptr → ACLBLAS_STATUS_INVALID_VALUE；元素值不校验（NaN/Inf 按 IEEE 传播） | 一维，物理长度 1+(n−1)×\|incx\| |
| incx | x 相邻元素存储步长（元素单位），Host 内存 | int | INT32 | 非零且非 INT32_MIN，正负均支持；违例返回 ACLBLAS_STATUS_INVALID_VALUE；(n−1)×\|incx\| 超 int32 返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| y | 输入兼输出（原地更新）向量，Device 内存 | float* | FLOAT32 | n>0 时 nullptr → ACLBLAS_STATUS_INVALID_VALUE；元素值不校验 | 一维，物理长度 1+(n−1)×\|incy\| |
| incy | y 相邻元素存储步长，Host 内存 | int | INT32 | 同 incx | - |
| c | 旋转矩阵余弦标量指针，Host 或 Device 内存 | const float* | FLOAT32 | n>0 时 nullptr → ACLBLAS_STATUS_INVALID_VALUE；值不校验；内存位置由运行时 aclrtPointerGetAttributes 自动判定 | 标量 |
| s | 旋转矩阵正弦标量指针，Host 或 Device 内存 | const float* | FLOAT32 | 同 c（与 c 位置相互独立，可混合） | 标量 |

### 数学公式

（与精度判定所用 golden `cblas_srot` 的公式同一形态，任务书 §2.1）

```
x[k] = c × x[k] + s × y[j]
y[j] = −s × x[k] + c × y[j]        （两式右端 x[k]、y[j] 均取旋转前旧值）
其中 k = (i−1)×incx，j = (i−1)×incy，i = 1..n；G = [[c, s], [−s, c]]
n=0 时 x、y 不修改（no-op 快速返回）
```

# 需求分析（required）

## 需求描述

用 Ascend C kernel 直调方式，在 Atlas A2/A3（arch22）实现 FLOAT32 双向量 Givens 原地旋转：host 侧完成 cuBLAS 口径参数校验（n=0 快速返回；n<0、空指针、0/INT32_MIN 步长、(n−1)×|stride| int32 溢出均返回 ACLBLAS_STATUS_INVALID_VALUE）与 tiling 计算下发；device 侧单 kernel 双路径（连续 ping-pong 流水 / 跨步标量路径）完成原地旋转；精度按 mixed_tolerance（FLOAT32：rtol=atol=2⁻¹³，ratio≥0.99）全量 1000 条通过；性能按 msprof op task duration 口径达到 A100 标杆 0.8 倍以上（n=1M/2M/4M 阈值 20.7/31.3/61.0 µs）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 纯逐元素线性组合（c·x+s·y 与 −s·x+c·y），无归约、无跨元素依赖；x 旧值参与第二式，需先读后写（kernel 内天然满足：Compute 在 UB 备份后输出） |
| 2 | 接口原型 | `include/cann_ops_blas.h` L459 既有声明，签名与 cublasSrot 参数逐位一致（任务书 §3.5），零头文件改动 |
| 3 | 参数规格 | 见上文固定表；c/s 为 Host/Device 双位置标量指针（各自独立判定） |
| 4 | 支持数据类型 | 仅 FLOAT32（x/y 向量与 c/s 标量） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 训练/推理系列 → arch22（SOC=ascend910b\*/ascend910_93\*；自验覆盖 910B3） |
| 6 | 精度要求（任务书 §3.2） | mixed_tolerance：rtol=atol=2⁻¹³、required_matched_ratio=0.99、max_abs_error≤1e-2 or 32 ULP；x/y 分别判定；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN↔±Inf 等效，非有限值单独记录须全等效） |
| 7 | 性能预期（任务书 §3.3） | msprof op task duration（warmup 5 + ≥10 采样均值）≤ gpu_baseline.csv 的 gpu_ms/0.8；典型 case 阈值：n=1M≤20.71µs、n=2M≤31.34µs、n=4M≤61.03µs（A100-SXM4-80GB 实测标杆） |
| 8 | 代码架构选型 + 可行性 | kernel 直调 + 单 kernel 双路径：连续路径（incx=incy=1）满核 AIV ping-pong 双缓冲 MTE2→V→MTE3 流水（带宽型，n=4M 需 ~1.05TB/s 有效带宽，910B3 满核可达，S2 编译后探针实证）；跨步路径标量 GM 读写 + UB staging（仅精度覆盖，性能 case 全部 incx=incy=1）。已排除死路：①SIMT kernel（arch22 无 SIMT 指令，arch35 srot_simt_kernel 不可移植）；②csrot 的固定 8 核 host 模式（8/40 核，带宽天花板不足标杆）；③host 内部 aclrtSynchronizeStream（破坏异步语义且加大时延） |
| 9 | 评测来源 | golden=cblas_srot（Netlib 3.10.0，仓内 srot_golden.h 一致）；标杆=A100-SXM4-80GB/CUDA 12.8/cuBLAS 12.8.4 实测（gpu_baseline.csv） |

# 详细设计（required）

## 算子分析

### 数学公式

同「需求背景-数学公式」。逐元素操作，输出确定（同输入同步长跨次执行 bit 级一致，任务书 §2.5 确定性条款）。

### 支持数据类型

| 操作数 | 类型 |
|---|---|
| x / y | FLOAT32（ND 一维向量，步长访问） |
| c / s | FLOAT32 标量（Host 或 Device 指针） |
| n / incx / incy | int（Host 属性入参） |

### 支持形状

| 维度 | 约束 | 退化行为 |
|---|---|---|
| n | 任意 int32；物理长度 1+(n−1)×\|incx\|（y 同理） | n=0：快速返回 SUCCESS，不下发 kernel；n<0：INVALID_VALUE |
| incx/incy | 非零非 INT32_MIN，正负均支持（负步长自物理 span 尾端起算） | 0 或 INT32_MIN：INVALID_VALUE；(n−1)×\|stride\| 超 int32：INVALID_VALUE（host int64 预算后判定） |
| c/s | 任意 FLOAT32 值（含 Inf/NaN、c=1&&s=0 单位旋转） | 不短路、不校验值，按公式自然传播 |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（单 kernel 内按 tilingKey 分支，两路径）：

| 变体 | tilingKey | 触发条件 | 职责 |
|---|---|---|---|
| 连续路径（流水） | 0 | incx==1 && incy==1 | 满核 AIV 均分区间，ping-pong 双缓冲 MTE2→V→MTE3 流水搬运与旋转 |
| 跨步路径（标量+staging） | 1 | 其余合法步长组合（正非 1、负、混合） | 各核按逻辑下标标量读 GM 旧值 → UB 32B 对齐槽位计算 → DataCopyPad 逐元素回写 |

数据流：

```
GM(x,y) --MTE2(gm_to_ub_align)--> UB(双缓冲 x/y tile) --V(muls_v+Axpy)--> UB(temp1/temp2)
UB(temp1/temp2) --MTE3(ub_to_gm_align)--> GM(x,y)          [连续路径]
GM(x,y) --标量GetValue--> 寄存器 --V标量算--> UB staging槽 --DataCopyPad--> GM(x,y)  [跨步路径]
```

#### host 侧设计

**数据检测（校验顺序，逐条对齐任务书 §2.4/§2.5 与仓内 arch22 csrot 既有口径）**：

| 序 | 条件 | 返回 |
|---|---|---|
| 1 | handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 2 | n == 0 | ACLBLAS_STATUS_SUCCESS（快速返回，不访问 x/y/c/s，不下发 kernel） |
| 3 | n != 0 且 (x==nullptr \|\| y==nullptr) | ACLBLAS_STATUS_INVALID_VALUE |
| 4 | n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| 5 | c==nullptr \|\| s==nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| 6 | incx==0 \|\| incx==INT32_MIN \|\| incy==0 \|\| incy==INT32_MIN | ACLBLAS_STATUS_INVALID_VALUE |
| 7 | int64 预算 (n−1)×\|incx\| 或 (n−1)×\|incy\| 超 int32 范围 | ACLBLAS_STATUS_INVALID_VALUE |

**c/s 标量来源判定**（arch35 srot 模式）：对 c、s 各自独立调用 `aclrtPointerGetAttributes`；host 指针 → host 解引用一次写入 tiling 的 cosValue/sinValue；device 指针 → tiling 填 0.0f 占位、指针原样转发给 kernel（GM 直读，无 D2H 拷贝、无隐式同步）；cIsDevice/sIsDevice 标志随 tiling 下发。

**分核策略**：`GetAivCoreCount()` 动态获取 AIV 核数，cap 至 DEFAULT_VECTOR_NUM=40，再 min(核数, n)（连续路径按元素均分；跨步路径按逻辑元素均分）——满核原则，均分余块分配给前 r 个核（core i 处理 ⌈n/B⌉ 或 ⌊n/B⌋+1 个元素），与仓内 arch22 兄弟（caxpy BuildCaxpyExecution）同款。

**数据分块和内存优化策略**：连续路径 UB 双缓冲（ping-pong 2×4 buffer，每 buffer 24KB ≈ 6144 float，对齐 csrot 连续路径布局）；MTE2 搬入与 V 计算、MTE3 写回三级流水重叠，事件（EVENT_ID0/1）轮转同步；跨步路径 UB staging 槽 32B 对齐（每元素 8 槽 float，MTE 数据通路对齐要求）。

**tilingkey 规划策略**：

| tilingKey | 含义 | 生效字段 |
|---|---|---|
| 0 | 连续路径 | totalN/perCoreN/remainder/tileSize |
| 1 | 跨步路径 | totalN/incx/incy |

**tiling 下发**：tiling 结构经 handle 绑定的 workspace 缓冲区异步 H2D 写入后随 kernel 下发（kernel 从 GM 读 tiling，csrot 同款）；host 全程零 malloc、零 aclrtSynchronizeStream（异步语义，任务书 §2.5）。

#### kernel 侧设计

连续路径（tilingKey=0，性能主路径）流程：

```
每核: startIndex/elementPerCore ← host tiling 均分
loop over 每 24KB tile (ping-pong 事件轮转):
  CopyIn : WAIT(MTE3→MTE2) → gm_to_ub_align(x tile) → SET(MTE2→V,ev)
           gm_to_ub_align(y tile) → SET(MTE2→V,ev')
  Compute: WAIT(MTE2→V) → temp1 = c·x;  temp2 = −s·x
           → PIPE_BARRIER(V) → Axpy(temp1, y, s) ; Axpy(temp2, y, c)
  CopyOut: SET/WAIT(V→MTE3) → ub_to_gm_align(x←temp1) ; ub_to_gm(y←temp2)
           → SET(MTE3→MTE2)（归还 buffer）
```

- 旋转公式分解：x' = c·x + s·y = muls_v(x,c) ⊕ Axpy(y,s)；y' = −s·x + c·y = muls_v(x,−s) ⊕ Axpy(y,c)——与 csrot 连续路径（L148-181）同构，去复数分量（CSROT_COMPONENTS 2→1）
- 跨步路径（tilingKey=1）：物理下标映射 phys = (inc>0) ? logical×inc : (total−1−logical)×|inc|（负步长自尾端起算，netlib 语义，csrot L26-29 同式）；标量 GetValue 读旧值 → UB 槽位 SetValue 计算 → PIPE_BARRIER(ALL) → DataCopyPad 8B burst 回写（x、y 各一）
- 尾块处理：连续路径末 tile 按 loop_remain 截断 lenBurst；跨步路径按 passElems 截断
- c/s 读取：cIsDevice/sIsDevice=1 时 kernel 从转发指针 GM 直读；否则用 tiling 标量

**tiling/workspace 布局**：

| 区域 | 内容 | 大小 |
|---|---|---|
| tiling GM（handle workspace 内） | SrotTilingData{tilingKey,totalN,cosValue,sinValue,cIsDevice,sIsDevice,perCoreN,remainder,tileSize,incx,incy} | ≤64B |
| kernel workspace | 不需要（原地算子，无中间数据） | 0 |

#### 性能优化要点（终态）

| 优化项 | 设计依据/目标 |
|---|---|
| 满核 AIV（vs csrot 固定 8 核） | 带宽型算子带宽≈核数线性；n=4M 需 ~1.05TB/s（64MB/61µs 阈值），8 核模式天花板不足 |
| ping-pong 双缓冲三级流水 | 搬运与计算重叠，MTE2/V/MTE3 不空转（csrot 连续路径已验证同构） |
| host 零 malloc / 零同步 | tiling 走 handle workspace 异步下发；c/s device 指针 kernel 直读——消除 csrot 每 call 2×malloc+同步 memcpy+stream sync 的固定开销（小 n case 关键，n=1 阈值 8.8µs） |
| 24KB tile 单 burst 搬运 | 每 tile 一次 gm_to_ub_align/ub_to_gm_align 长突发，带宽利用率最优 |
| 实测回填 | S2 编译后探针（n=4M 连续路径 msprof task duration vs 61.03µs 阈值）；全量 200 条 S4 基线，实测数据见自测报告 |

未采用路线及原因：SIMT grid-stride（arch22 无 SIMT 指令）；packing fallback 跨步转连续（跨步 case 不担性能指标，标量路径正确性优先、代码量最小）；L2 驻留优化（采样口径与标杆一致，无需特殊处理，S4 实测再定）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品（ascend910b\*，自验设备 Atlas 800T A2 / 910B3） | ✔ 本任务目标 |
| Atlas A3 训练/推理系列产品（ascend910_93\*，同 arch22） | ✔ 随 arch22 构建 |
| Ascend 950PR / 950DT（arch35） | 已有独立实现，本任务不涉及 |

## 算子约束限制（任务书 §2.5 逐条核对结论）

| 约束项 | 结论 |
|---|---|
| 参数合法性与校验层级 | 按 host 数据检测表顺序校验，返回码与 cuBLAS/任务书 §2.4 一致 |
| 非连续 Tensor 支持 | 支持：BLAS 步长语义即非连续访问（incx/incy 间隔访问、负步长尾端起算）；0/INT32_MIN 拦截 |
| broadcast 规则 | 不涉及：BLAS-1 双向量元素个数同为 n，无操作数间广播 |
| dynamic shape 要求 | 不要求：n/incx/incy 为运行时入参，host tiling 按运行时 n 计算，任意合法规模一次适配 |
| 原地与视图语义 | x、y 原地更新（先读后写，第二等式用旧 x 值），不返回视图 |
| 确定性计算要求 | 满足：逐元素单次执行确定，跨次 bit 级一致 |
| 空 Tensor 处理 | 等价 n=0：快速返回 SUCCESS，无空 Device 指针参与计算路径 |
| 0 维 Tensor 处理 | 不涉及：向量算子无 0 维输入 |
| 异步执行 | 经 handle stream 异步下发；返回成功仅表示已提交；读取输出前需调用方同步；c/s 内存位置运行时自动判定 |

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32：rtol=atol=2⁻¹³（1.2207e-4），required_matched_ratio=0.99，max_abs_error ≤ 1e-2 or 32×ULP；x/y 分别判定；Inf/NaN 按等效规则（同 NaN/同号 Inf/NaN↔±Inf 等效，非有限值不计入 ratio 须单独全等效） | 生态算子开源精度标准 mixed_tolerance（gitcode.com/cann/opbase@28a1ce5）；任务书 §3.2 |
| 性能标准 | 单 case：NPU 平均单次耗时（msprof op task duration，warmup 5 + ≥10 采样均值）≤ gpu_baseline.csv gpu_ms / 0.8（≥0.8× A100）；200 条独立判定 | 任务书 §3.3；标杆 A100-SXM4-80GB/CUDA 12.8/cuBLAS 12.8.4 实测 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

## 测试概览

- 测试工程：`test/rot/srot/arch22/`（CSV 用例描述 + C++ GTest 加载执行），`test/rot/srot/CMakeLists.txt` arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} srot_test)`（补齐当前仓内仅 arch35 注册的缺口）；golden 用仓内共用 `srot_golden.h`（cblas_srot float32）
- 用例分类（任务包 1000 精度 + 200 性能，按任务书 §3.5 口径修正后使用）：L0 基础 12 / 尺寸扫描 38 / 步长 21（含负步长）/ c-s 取值 22（含 Inf/NaN、单位旋转）/ c-s 指针位置 11（host/device/mixed）/ 填充 24 / 边界负向 17（n=0、n<0、空指针、溢出 3 条）/ 扩展 855 / 性能 200；正态分布（GAUSS_NORM_5_2，μ∈[−5,5]、σ∈[0.1,2]）按任务书 §3.5 补充要求纳入精度全量
- 边界负向口径说明：任务包 CSV 原为 950PR（arch35）口径（n<0/零步长期望 SUCCESS）；本任务按任务书 §2.4 arch22 口径修正为 ACLBLAS_STATUS_INVALID_VALUE 并在自测报告说明；负向用例按单元素 dummy 缓冲占位（不按 (n−1)×|inc| 分配，INT32_MIN 行先短路防 std::abs UB）
- handle/c-s 空指针等 CSV 不可表达项由 TEST_F 固化用例覆盖
- 复现入口：交付件自测报告（用例参数、精度对比、性能数据、步骤说明；test_cases/README.md 的 verify_accuracy.py / verify_performance.py 流程）

## 兼容性分析

- 新增 arch22 目录四件 + arch22 测试目录，与 arch35 既有实现互不干扰（blas/CMakeLists.txt 按 SOC 自动拾取 arch 目录）；`include/cann_ops_blas.h` 零改动，无 API 兼容性影响
- `blas/rot/README.md` 随 PR 更新 aclblasSrot 的 A2/A3 支持情况与 arch22 约束节（并入既有 per-arch 差异表，不改 arch35 语义）
- 与 arch35 的 per-arch 行为差异（n<0、零步长、INT32_MIN）沿 `blas/rot/README.md`「跨 arch 入参返回码对照」既定声明方式固化为两侧各自测试契约，不属缺陷
