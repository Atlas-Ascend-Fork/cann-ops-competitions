# aclblasCrot 算子设计文档

# 需求背景（required）

## 需求来源

- 任务：2026 年 11 月社区任务「aclblasCrot 算子开发（Atlas A2/A3）」
- 合入仓与目录：`https://gitcode.com/cann/ops-blas`（开发分支 master），实现代码合入 `blas/rot/arch22/`，测试代码合入 `test/rot/crot/arch22/`（任务书 §4/§5）
- 验收流程：cann 社区任务（hiascend.com 社区任务开发-验收流程），设计文档先于代码提交（cann-ops-competitions 设计文档 MR）

## 背景介绍

### 算子功能定位

aclblasCrot 是 BLAS Level-1 复数 Givens 平面旋转算子：对两个等长单精度复数向量 x、y，以实数余弦 c（float 标量）与**复数正弦 s**（complex64 标量）施加平面旋转并**原地更新**（对标 cuBLAS `cublasCrot`，cuBLAS Library §2.5.8 rot 条目）。y 的更新使用 s 的共轭 conjg(s)，参考语义为 Netlib/LAPACK 辅助例程 `crot.f`（参考 BLAS 无 crot 例程，cuBLAS 文档 rot 条目末尾明确引用 netlib crot()；任务书 §2.1）。常用于复数 QR 分解、线性方程组求解与特征值计算。本任务补齐 ops-blas 仓 Atlas A2/A3 系列（arch22）的能力缺口——950PR（arch35）实现已合入（f37a077），本任务为其跨 arch 移植。

### 业界对标

| 对标对象 | 语义依据 | 差异点 |
|---|---|---|
| cuBLAS `cublasCrot(handle, n, x, incx, y, incy, c, s)` | 参数序列与接口形态的唯一依据 | 返回码按仓内 arch22 家族契约：n<0、incx/incy==0、INT32_MIN → INVALID_VALUE（issue #827 已声明的 per-arch 差异） |
| Netlib/LAPACK `crot.f`（L75-77） | golden 参考语义（y'=c·y−conjg(s)·x，负步长自物理尾端起算） | netlib 对 `N.LE.0` 直接 RETURN；仓内 arch35 采 n≤0 SUCCESS 放行口径；arch22 对 n<0 取 INVALID_VALUE（`blas/rot/README.md`「跨 arch 入参返回码对照」，同族 csrot 先例） |
| 仓内 950PR 既有实现 `blas/rot/arch35/crot_*` | 行为模板（volatile 数学展开序 L49-73、c/s 独立指针判定、serialOnly 别名处理、跨度溢出守卫） | arch35 放行 n<0 与零步长（SIMT 路径含零步长串行累加）；arch22 与之相反，且 SIMT 路径不可移植（arch22 无 SIMT 指令） |
| 仓内 arch22 兄弟 `blas/rot/arch22/csrot_*` | arch22 行为契约（校验链）与复数向量 kernel 工程先例 | csrot 的 s 为实数、y 更新无共轭（分量级标量乘即可）；crot 需 re/im 平面分离计算 |

### 工程形态

- 工程模式：**kernel 直调**（Host C API 入口 + host 侧参数校验与 tiling 按值随 kernel 参数下发 + device 单 kernel 双路径），与 `blas/rot/` 既有 arch22/arch35 实现同一工程模式（任务书 §2.2）
- 代码落位：`blas/rot/arch22/`（crot_host.cpp / crot_kernel.cpp / crot_kernel.h / crot_tiling_data.h 四件，对齐 arch35 crot 同名布局）
- 仓内近亲：主模板 = `blas/rot/arch35/crot_*`（同算子跨 arch）；工程底座 = 同族 arch22 通行模式（满核 AIV cap 40、tiling 按值下发零 malloc 零同步、ping-pong 双缓冲、跨步 UB staging——同仓 arch22 caxpy 系模式，经同族前序任务实测验收）；契约参照 = `blas/rot/arch22/csrot_*` + `blas/rot/README.md` per-arch 差异表

### 接口规格

接口声明**已存在**于 `include/cann_ops_blas.h` L250-252（950 任务 f37a077 合入，签名与任务书 §2.3 逐字一致），本任务**零公共头改动**、不定义私有平行接口（任务书 §2.3、§7.3）：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const aclblasComplex* s);
```

参数规格（固定表）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（内部携带 stream），Host 内存 | aclblasHandle_t | - | aclblasCreate 返回的非空句柄；为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 向量元素个数，Host 内存 | int | INT32 | n=0 快速返回 SUCCESS；n<0 返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| x | 输入兼输出（原地更新）复数向量，Device 内存 | aclblasComplex* | COMPLEX64 | n>0 时 nullptr → ACLBLAS_STATUS_INVALID_VALUE；元素值不校验（NaN/Inf 按 IEEE 传播） | 一维，物理长度 1+(n−1)×\|incx\| |
| incx | x 相邻元素存储步长（元素单位），Host 内存 | int | INT32 | 非零且非 INT32_MIN，正负均支持（负步长自物理尾端起算）；违例返回 ACLBLAS_STATUS_INVALID_VALUE；(n−1)×\|incx\| 超 int32 返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| y | 输入兼输出（原地更新）复数向量，Device 内存 | aclblasComplex* | COMPLEX64 | 同 x | 一维，物理长度 1+(n−1)×\|incy\| |
| incy | y 相邻元素存储步长，Host 内存 | int | INT32 | 同 incx | - |
| c | 旋转余弦标量指针，Host 或 Device 内存 | const float* | FLOAT32 | n>0 时 nullptr → ACLBLAS_STATUS_INVALID_VALUE；值不校验；内存位置由运行时 aclrtPointerGetAttributes 自动判定 | 标量 |
| s | 旋转正弦复数标量指针，Host 或 Device 内存 | const aclblasComplex* | COMPLEX64 | 同 c（与 c 位置相互独立，可混合；device 时转发 &s->real 双 float） | 标量 |

### 数学公式

（与精度判定所用 golden `test/rot/crot/crot_golden.h` 的公式同一形态，任务书 §2.1）

```
x[k] = c × x[k] + s × y[j]
y[j] = c × y[j] − conjg(s) × x[k]     （两式右端 x[k]、y[j] 均取旋转前旧值）
k = (i−1)×incx，j = (i−1)×incy，i = 1..n；n=0 时 x、y 不修改（no-op 快速返回）
```

实数展开（c 为实数，s=(s_r, s_i)；golden 与 kernel 同序，无 FMA 融合）：

```
x'_r = c·x_r + (s_r·y_r − s_i·y_i)      x'_i = c·x_i + (s_r·y_i + s_i·y_r)
y'_r = c·y_r − (s_r·x_r + s_i·x_i)      y'_i = c·y_i − (s_r·x_i − s_i·x_r)
```

# 需求分析（required）

## 需求描述

用 Ascend C kernel 直调方式，在 Atlas A2/A3（arch22）实现 COMPLEX64 双向量 Givens 原地旋转（复数 s、y 更新含共轭）：host 侧完成 arch22 契约参数校验（n=0 快速返回；n<0、空指针、0/INT32_MIN 步长、(n−1)×|stride| int32 溢出均返回 ACLBLAS_STATUS_INVALID_VALUE）与 tiling 按值异步下发（零 malloc、零隐式同步）；device 侧单 kernel 双路径（连续路径 DeInterleave 平面化向量计算 / 跨步路径标量+UB staging；x==y 异步长别名输入走单块串行链式）；精度按 mixed_tolerance（COMPLEX64 实部/虚部分别按 FLOAT32：rtol=atol=2⁻¹³，ratio≥0.99）全量通过；性能按 GTest 事件口径 avg_us（warmup 5 + ≥10 采样均值）达到 A100 标杆 0.8 倍以上（n=1M/2M/4M 阈值 23.72/56.92/109.33 µs）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 纯逐元素 4 分量线性组合（两次复数乘加），无归约、无跨元素依赖；y' 用旧 x（kernel 先读后写天然满足）；arch35 crot 已给出完整 volatile 展开序（CrotRotate L49-73），golden `crot_golden.h` 逐字同序——计算序直接镜像，无舍入歧义 |
| 2 | 接口原型 | `include/cann_ops_blas.h` L250-252 既有声明，签名与 cublasCrot 参数逐位一致（任务书 §3.5-2），零头文件改动 |
| 3 | 参数规格 | 见上文固定表；c/s 为 Host/Device 双位置标量指针（各自独立判定） |
| 4 | 支持数据类型 | x/y=COMPLEX64（交织 float 对）；c=FLOAT32；s=COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 训练/推理系列 → arch22（SOC=ascend910b\*/ascend910_93\*；自验覆盖 910B3） |
| 6 | 精度要求（任务书 §3.2） | mixed_tolerance：实部/虚部分别按 FLOAT32 判定 rtol=atol=2⁻¹³、required_matched_ratio=0.99、max_abs_error≤1e-2 or 32 ULP；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN↔±Inf 等效，非有限值单独记录须全等效） |
| 7 | 性能预期（任务书 §3.3） | avg_us（warmup 5 + ≥10 采样）≤ gpu_baseline.csv 的 gpu_ms/0.8；典型 case 阈值：n=1M≤23.72µs、n=2M≤56.92µs、n=4M≤109.33µs、n=1≤8.81µs（A100-SXM4-80GB 实测标杆 18.98/45.53/87.47/7.05µs） |
| 8 | 代码架构选型 + 可行性 | kernel 直调 + 单 kernel 双路径：连续路径（incx=incy=1）满核 AIV ping-pong 双缓冲 + DeInterleave 拆 re/im 平面 + 无融合向量算术（V220 有 DeInterleave/Interleave 先例：cgemm_ex/cgbmv arch22；V220 无向量 FMA 融合原语 → 天然逐条舍入，与 golden volatile 序一致）+ Interleave 回交织；跨步路径标量 GM 读 + 32B 对齐 UB 槽 + 8B DataCopyPad 回写；serialOnly 别名单块链式。已排除死路：①arch35 SIMT kernel（arch22 无 SIMT 指令）；②csrot 固定 8 核 + host 内部 ReadScalarFromHostOrDevice/stream sync 模式（带宽与时延双输：D2H 同步读标量 + 每 call 2×malloc）；③交织域直接向量计算（复数乘需 re/im 混合，V220 无 lane 级混洗向量原语，平面化为仓内既证模式） |
| 9 | 评测来源 | golden=仓内共用 `crot_golden.h`（Netlib/LAPACK crot 语义，volatile 无融合）；标杆=A100-SXM4-80GB/CUDA 12.8/cuBLAS 12.8.4 实测（gpu_baseline.csv 200 条） |

# 详细设计（required）

## 算子分析

### 数学公式

同「需求背景-数学公式」。逐元素操作，输出确定（同输入同步长跨次执行 bit 级一致，任务书 §2.5 确定性条款）。别名输入：x==y 且 incx==incy 时按元素独立（x' 先写、y' 后写，同址定序即 golden 串行末值）；x==y 且 incx!=incy 时按逻辑下标顺序链式求值（serialOnly）。

### 支持数据类型

| 操作数 | 类型 |
|---|---|
| x / y | COMPLEX64（ND 一维向量，交织 float 对，步长访问） |
| c | FLOAT32 标量（Host 或 Device 指针） |
| s | COMPLEX64 标量（Host 或 Device 指针，转发 &s->real） |
| n / incx / incy | int（Host 属性入参） |

### 支持形状

| 维度 | 约束 | 退化行为 |
|---|---|---|
| n | 任意 int32；物理长度 1+(n−1)×\|incx\|（y 同理） | n=0：快速返回 SUCCESS，不下发 kernel；n<0：INVALID_VALUE |
| incx/incy | 非零非 INT32_MIN，正负均支持（负步长自物理 span 尾端起算） | 0 或 INT32_MIN：INVALID_VALUE；(n−1)×\|stride\| 超 int32：INVALID_VALUE（host int64 预算后判定） |
| c/s | 任意 FLOAT32 值（含 Inf/NaN、c=1&&s=(0,0) 单位旋转） | 不短路、不校验值，按公式自然传播 |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（单 kernel 内按 tilingKey + serialOnly 分支）：

| 变体 | tilingKey | 触发条件 | 职责 |
|---|---|---|---|
| 连续路径（平面化流水） | 0 | incx==1 && incy==1 | 满核 AIV 均分区间，ping-pong 双缓冲：DataCopyPad 搬 AoS tile → DeInterleave 拆 re/im 平面 → 无融合向量旋转 → Interleave → DataCopyPad 回写 |
| 跨步路径（标量+staging） | 1 | 其余合法步长组合（正非 1、负、混合） | 各核按逻辑下标标量读复数旧值 → volatile 标量旋转 → 32B 对齐 UB 槽 → DataCopyPad 8B burst 逐元素回写 |
| 串行子态（serialOnly） | 1 + serialOnly=1 | x==y 且 incx!=incy | 单块按逻辑序逐元素 load→rotate→store 链式求值（与 golden 串行一致） |

数据流：

```
[连续路径]
GM(x,y) --MTE2(DataCopyPad)--> UB(aosX/aosY 双缓冲 2f/elem)
UB(aos) --V(DeInterleave)--> UB(xr/xi/yr/yi 平面 4f/elem)
UB(平面) --V(Muls/Add/Sub ×16, 无融合)--> UB(nxr/nxi/nyr/nyi 平面 4f/elem, 共享 t0/t1)
UB(n*) --V(Interleave, 复用 aos)--> UB(outX/outY) --MTE3(DataCopyPad)--> GM(x,y)
[跨步路径]
GM(x,y) --标量GetValue(2f/元素)--> 寄存器 --volatile 标量旋转--> UB 32B槽(8f/元素)
UB 槽 --PIPE_BARRIER(ALL)--> DataCopyPad(8B burst) --> GM(x,y)
```

#### host 侧设计

**数据检测（校验顺序，逐条对齐任务书 §2.4/§2.5）**：

| 序 | 条件 | 返回 |
|---|---|---|
| 1 | handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR（判空最优先，quick return 不得先于判空，#765/#861） |
| 2 | n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| 3 | n == 0 | ACLBLAS_STATUS_SUCCESS（快速返回，不访问 x/y/c/s，不下发 kernel） |
| 4 | incx==0 \|\| incx==INT32_MIN \|\| incy==0 \|\| incy==INT32_MIN | ACLBLAS_STATUS_INVALID_VALUE |
| 5 | int64 预算 (n−1)×\|incx\| 或 (n−1)×\|incy\| 超 int32 范围 | ACLBLAS_STATUS_INVALID_VALUE |
| 6 | x==nullptr \|\| y==nullptr \|\| c==nullptr \|\| s==nullptr（此时必有 n>0） | ACLBLAS_STATUS_INVALID_VALUE |

（与仓内 arch22 csrot/srot 校验链在全部输入上可观察等价：竞争检查返回同码、n=0 与 n<0 互斥。）

**c/s 标量来源判定**（arch35 crot 模式）：对 c、s 各自独立调用 `aclrtPointerGetAttributes`；host 指针 → host 解引用一次写入 tiling 的 cosValue/sinRe/sinIm；device 指针 → tiling 填 0.0f 占位、指针原样转发给 kernel（c 转发 float*，s 转发 reinterpret_cast<const float*>(s) 即 &s->real 双 float），kernel GM 直读——无 D2H 拷贝、无隐式同步；cIsDevice/sIsDevice 标志随 tiling 下发。

**分核策略**：`GetAivCoreCount()` 动态获取 AIV 核数，cap 至 DEFAULT_VECTOR_NUM=40，再 min(核数, n)；核内元素均分、余块给前 r 个核（core i 处理 ⌈n/B⌉ 或 ⌊n/B⌋+1），kernel 侧以 GetBlockIdx/GetBlockNum 复算（与同族 arch22 通行模式一致）；serialOnly 时强制 numBlocks=1。

**tilingkey 规划策略**：

| tilingKey | 含义 | 生效字段 |
|---|---|---|
| 0 | 连续路径 | totalN / cosValue / sinRe / sinIm / cIsDevice / sIsDevice |
| 1 | 跨步路径 | totalN / incx / incy / serialOnly（+上述标量字段） |

**tiling 下发**：tiling 结构**按值随 kernel 启动参数下发**（无每 call GM 暂存拷贝——同族实测该 pageable H2D 拷贝曾把小 n 提交地板抬到 ~31µs，按值后 5µs）；host 全程零 malloc、零 aclrtSynchronizeStream（异步语义，任务书 §2.5）。

#### kernel 侧设计

连续路径（tilingKey=0，性能主路径）流程：

```
每核: startElem/elemsPerCore ← GetBlockIdx/GetBlockNum 均分
loop over 每 tile (ping-pong 事件轮转, tile=1536 complex 起步, UB 预算内 S2 实测定):
  CopyIn : WAIT(MTE3→MTE2,ev) → DataCopyPad(aosX←GM x 段, 2N float)
           → DataCopyPad(aosY←GM y 段) → SET(MTE2→V,ev)
  Compute: WAIT(MTE2→V,ev) → DeInterleave(xr,xi←aosX) ; DeInterleave(yr,yi←aosY)
           → PIPE_BARRIER(V) → 平面旋转 16 条向量指令（见下表）
           → PIPE_BARRIER(V) → Interleave(aosX←nxr,nxi) ; Interleave(aosY←nyr,nyi)
           → SET(V→MTE3,ev)
  CopyOut: WAIT(V→MTE3,ev) → DataCopyPad(GM x 段←aosX) → DataCopyPad(GM y 段←aosY)
           → SET(MTE3→MTE2,ev)（归还 buffer）
```

平面旋转算术表（逐条对应 golden 展开，全部独立 Muls/Add/Sub，无融合；count=32B 对齐后的平面元素数）：

| 目标 | 指令序列（t0/t1 为共享 scratch 平面） |
|---|---|
| nxr = c·xr + (s_r·yr − s_i·yi) | Muls(t0,yr,s_r); Muls(t1,yi,s_i); Sub(t0,t0,t1); Muls(nxr,xr,c); Add(nxr,nxr,t0) |
| nxi = c·xi + (s_r·yi + s_i·yr) | Muls(t1,yi,s_r); Muls(t0,yr,s_i); Add(t1,t1,t0); Muls(nxi,xi,c); Add(nxi,nxi,t1) |
| nyr = c·yr − (s_r·xr + s_i·xi) | Muls(t0,xr,s_r); Muls(t1,xi,s_i); Add(t0,t0,t1); Muls(nyr,yr,c); Sub(nyr,nyr,t0) |
| nyi = c·yi − (s_r·xi − s_i·xr) | Muls(t1,xi,s_r); Muls(t0,xr,s_i); Sub(t1,t1,t0); Muls(nyi,yi,c); Sub(nyi,nyi,t1) |

- 向量原语调用形态对齐仓内 arch22 复数 kernel 先例（cgemm_ex/cgbmv arch22 的 DeInterleave/Interleave/Muls/Add/Sub V220 形态，S2 编码时逐参对齐）
- 尾块处理：平面化按 32B 对齐的 opCount 计算（垃圾尾部由 DataCopyPad 的 floatCount 截断，不写出）；跨步路径按 passElems 截断
- c/s 读取：cIsDevice/sIsDevice=1 时 kernel 从转发指针 GM 直读（c 读 1 float，s 读 2 float）；否则用 tiling 标量
- DeInterleave/Interleave 的 32B 对齐与偶数约束按仓内先例处理（opCount 8 对齐）

跨步路径（tilingKey=1）流程：

```
每核逻辑段 [startElem, startElem+elemsPerCore):
  分 pass（passElems=1024）:
    for i in pass: phys = (inc>0) ? logical*inc : (total-1-logical)*|inc|   （负步长尾端起算）
      xr,xi ← gm_x[2p], gm_x[2p+1] GetValue; yr,yi ← gm_y 同理
      volatile 旋转（CrotRotate 标量版，8 乘积 volatile 防 FMA 收缩）
      buf_temp1[i*8+0/1] = nxr,nxi; buf_temp2[i*8+0/1] = nyr,nyi   （32B 对齐槽）
    PIPE_BARRIER(ALL)   （标量 UB 写不骑 V 管线，MTE3 读槽前需全栅栏）
    for i in pass: DataCopyPad(gm_x[2p]←buf_temp1[i*8], lenBurst=8); DataCopyPad(gm_y[2p]←buf_temp2[i*8])
    PIPE_BARRIER(ALL)
```

- 回写次序 x' 先于 y'（同址别名 x==y&&incx==incy 时 golden 串行末值 = y'，天然一致）
- serialOnly 子态：单块、逐元素 load→rotate→store 后再进下一逻辑元素（passElems=1 退化，链式语义与 golden 一致）

**tiling/workspace 布局**：

| 区域 | 内容 | 大小 |
|---|---|---|
| kernel 启动参数（按值） | CrotTilingData{tilingKey,totalN,incx,incy,serialOnly,cosValue,sinRe,sinIm,cIsDevice,sIsDevice} | 40B |
| kernel workspace | 不需要（原地算子，无中间数据） | 0 |
| UB（连续路径） | 双集合（aosX/aosY 各 2f/elem + 4 输入平面 + 4 输出平面）+ 共享 t0/t1 ≈ 26f/elem × tile 1536 = ~160KB ≤ 192KB（V220 硬上限） | tile 定量 S2 探针实测定 |
| UB（跨步路径） | buf_temp1/buf_temp2 各 32KB（1024 槽 × 8f） | 64KB |

#### 性能优化要点（终态）

| 优化项 | 设计依据/目标 |
|---|---|
| 满核 AIV（vs csrot 固定 8 核） | 带宽型算子带宽≈核数线性；n=4M 需 ~1.17TB/s（128MB/109µs 阈值） |
| ping-pong 双缓冲三级流水 | MTE2/V/MTE3 重叠不空转（同族 arch22 连续路径已验证同构） |
| host 零 malloc / 零同步 + tiling 按值下发 | 消除每 call pageable H2D 暂存拷贝与 stream sync（小 n case 关键，n=1 阈值 8.81µs；同族实测提交地板 31µs→5µs） |
| 平面化无融合计算 | 16 条独立向量指令镜像 golden volatile 序，精度与性能兼得（无逐元素分支） |
| DeInterleave/Interleave 复用 AoS 缓冲 | 平面拆分不额外增加 GM 流量，UB 内完成 |
| 实测回填 | S2 编译后探针（n=4M 连续路径 avg_us vs 109.33µs 阈值）；全量 200 条 S4 基线，实测数据见自测报告 |

未采用路线及原因：SIMT grid-stride（arch22 无 SIMT 指令）；跨步转连续 packing（跨步 case 不担性能指标，标量路径正确性优先、代码量最小）；L2 驻留专项优化（采样口径与标杆一致，S4 实测再定）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品（ascend910b\*，自验设备 Atlas 800T A2 / 910B3） | ✔ 本任务目标 |
| Atlas A3 训练/推理系列产品（ascend910_93\*，同 arch22） | ✔ 随 arch22 构建 |
| Ascend 950PR（arch35） | 已有独立实现（f37a077），本任务不涉及 |

## 算子约束限制（任务书 §2.5 逐条核对结论）

| 约束项 | 结论 |
|---|---|
| 参数合法性与校验层级 | 按 host 数据检测表顺序校验（handle 最优先、quick return 不先于判空），返回码与任务书 §2.4 一致 |
| 非连续 Tensor 支持 | 支持：BLAS 步长语义即非连续访问（incx/incy 间隔访问、负步长尾端起算）；0/INT32_MIN 拦截（arch22 契约） |
| broadcast 规则 | 不涉及：BLAS-1 双向量元素个数同为 n，无操作数间广播 |
| dynamic shape 要求 | 不要求：n/incx/incy 为运行时入参，host tiling 按运行时 n 计算，任意合法规模一次适配 |
| 原地与视图语义 | x、y 原地更新（先读后写，y' 用旧 x），不返回视图；别名输入（x==y）分同步长（逐元素定序）/异步长（serialOnly 链式）两态处理 |
| 确定性计算要求 | 满足：逐元素单次执行确定，跨次 bit 级一致 |
| 空 Tensor 与 0 维处理 | n=0：快速返回 SUCCESS，不修改 x、y；向量算子无 0 维输入 |
| 异步执行 | 经 handle stream 异步下发；返回成功仅表示已提交；读取输出前需调用方同步；c/s 内存位置运行时自动判定（相互独立可混合） |

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
|---|---|---|
| 精度标准 | COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol=atol=2⁻¹³（1.2207e-4），required_matched_ratio=0.99，max_abs_error ≤ 1e-2 or 32×ULP；x/y 分别判定；Inf/NaN 按等效规则（同 NaN/同号 Inf/NaN↔±Inf 等效，非有限值不计入 ratio 须单独全等效） | 生态算子开源精度标准 mixed_tolerance（gitcode.com/cann/opbase@28a1ce5）；任务书 §3.2 |
| 性能标准 | 单 case：NPU 平均单次耗时（GTest 事件口径 avg_us，warmup 5 + ≥10 采样均值）≤ gpu_baseline.csv gpu_ms/0.8（≥0.8× A100）；200 条独立判定；msprof op 仅供交叉验证 | 任务书 §3.3；标杆 A100-SXM4-80GB/CUDA 12.8/cuBLAS 12.8.4 实测 |
| 内存标准 | 不涉及（原地算子，无 workspace） | 任务书 §3.4 |

## 测试概览

- 测试工程：`test/rot/crot/arch22/`（crot_test.cpp + crot_npu_wrapper.h + crot_test.csv 三件）；共用层直接复用仓内 `test/rot/crot/{crot_golden.h, crot_param.h, run_self_test.py, test_runner_utils.py, run_memcheck.py}`；`test/rot/crot/CMakeLists.txt` 现为无条件单行 GTest 注册（f37a077 已重构），arch22 构建随 arch22/ 目录自动生效（同构注册在同族 arch22 已实测），S2 构建确认，若退回非 GTest runner 则补 arch22 分支
- 用例分类（任务包 1200 条按任务书 §3.5 口径修正后使用 + 正态补充）：L0 基础 8 / 尺寸扫描 38 / 步长组合 36（±1/±2/±3 全组合）/ c-s 特殊值 20（单位旋转/纯虚 s/复数 s/大值/极端值）/ 填充 12（含 Inf/NaN/全零/交替）/ 指针位置 6（host/device/mixed）/ 边界负向 13（n=0×2、n<0×1、零步长×3、INT32_MIN×2、跨度溢出×1、空指针×4）/ 扩展 867 / 性能 200（均连续访存）
- CSV 修正说明：任务包原为 950PR（arch35）口径，4 行按任务书 §2.4 arch22 契约翻转为 ACLBLAS_STATUS_INVALID_VALUE（TC_ED_123 n=−5、TC_ED_124/125/126 零步长），自测报告留痕；负向用例按单元素 dummy 缓冲占位（不按 (n−1)×|inc| 分配，§3.5 负向缓冲规则）
- 正态补充：GAUSS_NORM_5_2（μ∈[−5,5]、σ∈[0.1,2]，实虚部独立采样）按任务书 §3.5-3 补充要求纳入精度全量，补齐方式与条数在自测报告说明
- TEST_F 固化用例：NullHandle（判空最优先）、NegativeNRejected、ZeroStrideRejected、IntMinStrideRejected、StridedOffsetOverflowRejected、QuickReturnSkipsDataPointerValidation（n=0）、GapElementsUntouched（步长间隔元素不被污染）、AliasedBuffersMatchGolden（x==y 同步长/异步长两态对 golden）
- 复现入口：交付件自测报告（用例参数、精度对比（实部/虚部分别）、性能数据、步骤说明；test_cases/README.md 的 verify_accuracy.py / verify_performance.py 流程）

## 兼容性分析

- 新增 arch22 目录四件 + arch22 测试目录三件，与 arch35 既有实现互不干扰（blas/CMakeLists.txt 按 SOC 自动拾取 arch 目录）；`include/cann_ops_blas.h` 零改动，无 API 兼容性影响
- `blas/rot/README.md` 随 PR 更新 aclblasCrot 产品支持表（A2/A3 → 支持）并并入「跨 arch 入参返回码对照」既有声明方式（不改 arch35 语义）
- 与 arch35 的 per-arch 行为差异（n<0、零步长、INT32_MIN）沿仓内 issue #827 既定声明固化为两侧各自测试契约，不属缺陷；仓内共用 golden/param 头零改动（arch22 负向断言在测试侧按 arch22 契约表达）
