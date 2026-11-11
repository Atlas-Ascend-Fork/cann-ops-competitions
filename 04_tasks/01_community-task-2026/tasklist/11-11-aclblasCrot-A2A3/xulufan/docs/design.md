# aclblasCrot（A2/A3）算子设计文档

# 需求背景（required）

## 需求来源

2026 年 11 月社区任务《aclblasCrot 算子开发（A2A3）》任务书。目标在 Atlas A2/A3 系列
（Atlas 800T A2 / Atlas 800T A3，arch22 家族，dav_c220 核心）上，使用 Ascend C 语言开发
单精度复数向量 Givens 平面旋转算子 `aclblasCrot`，交付至 ops-blas 开源仓 `blas/rot/arch22/`。

## 背景介绍

### aclblasCrot 功能与工程现状

- 数学语义（对齐 cuBLAS `cublasCrot`，参考 Netlib/LAPACK `crot`）：

  ```
  x'[k] = c * x[k] + s * y[j]
  y'[j] = c * y[j] - conjg(s) * x[k]    （使用旋转前的原始 x[k]）
  ```

  其中 c 为实数（float），s 为复数（complex64），x、y 为复数向量并原地更新，k、j 为按
  incx/incy 步长（可正可负，负步长自尾端起算）展开的 1-based 物理索引。
- ops-blas 仓 `blas/rot/arch22/` 此前仅有 `csrot`（实数版旋转）实现；`aclblasCrot` 的接口
  声明（`include/cann_ops_blas.h`）、arch22 内核、arch22 GTest 测试工程均为本任务新增。
- arch35（950PR）已有同族 `crot` 参考实现，但其依赖的 DeInterleave/Interleave 向量原语
  在 arch22 上不存在，方案不可直接移植，需针对 dav_c220 的原语面独立设计。

# 需求分析（required）

## 需求描述

在 ops-blas 工程框架内实现 `aclblasCrot` 句柄式 BLAS 接口：handle 绑定 stream 直调
Ascend C kernel，覆盖 n = 0 快速返回、n < 0 / 零步长 / INT32_MIN 步长 / 跨度溢出等负向
契约，正负步长全支持，c/s 标量支持 Host 或 Device 指针；精度满足生态算子开源精度标准
（FLOAT32：rtol = atol = 2^-13，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP），
性能满足"msprof op 模式 Task Duration 平均值 ≤ GPU 标杆 1.25 倍"。

## 需求拆解

1. host 侧参数校验与路由：handle 判空 → n < 0 → n = 0 快速返回 → incx/incy
   （0 / INT32_MIN / int64 跨度溢出）与 n > 0 时 x/y/c/s 判空；
2. 连续路径（incx = incy = 1）：多核分片 + ping-pong 双缓冲 MTE 流水线；
3. 步长路径（非单位步长）：逻辑索引按物理地址无竞态划分到多核；
4. 别名（x == y 且非单位步长）等无法安全并行的场景回退单核串行走查；
5. s_imag = 0 的实数旋转快速路径（全部性能用例形态）+ 非有限值守卫回退；
6. arch22 GTest CSV 驱动测试工程（1200 用例）+ 性能门禁 + msprof 采集与判据。

# 详细设计（required）

## 算子分析

### 数学公式

按复数乘法 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`、共轭 `conjg(s) = sr - s_i·i` 展开
（记 c 实数，s = sr + s_i·i，x = xr + x_i·i，y = yr + y_i·i）：

```
x'r =  c·xr + sr·yr − s_i·yi        x'i =  c·xi + sr·yi + s_i·yr
y'r =  c·yr − sr·xr − s_i·yi        y'i =  c·yi − sr·xi + s_i·xr
```

y' 使用旋转前的原始 x（实现中 y' 与 x' 在同一 tile 内先算 x' 后算 y'，x 原值仍驻留
UB 输入缓冲，天然满足）。

### 支持数据类型与形状

| 项 | 内容 |
| --- | --- |
| x / y | COMPLEX64（实部/虚部各 float32），ND 一维，原地更新 |
| c | FLOAT32 标量指针（Host 或 Device） |
| s | COMPLEX64 标量指针（Host 或 Device） |
| n / incx / incy | int 标量；步长可正可负，物理长度 1+(n-1)*\|inc\| |

### 地址计算

负步长对齐 Netlib `IX = (-N+1)*INCX + 1`：逻辑 i 的物理索引为
`i*inc`（inc > 0）或 `(n-1-i)*(-inc)`（inc < 0），int64 中间量防溢出。

## 算子实现

### 实现方案

#### Host 侧设计（crot_host.cpp）

1. 校验顺序（任务书 §2.5 契约）：handle 非空 → n < 0 返回 INVALID_VALUE → n = 0 直接
   SUCCESS（不触碰 x/y）→ incx/incy 为 0 或 INT32_MIN 返回 INVALID_VALUE →
   (n-1)*\|inc\| 以 int64 计算超 int32 返回 INVALID_VALUE → n > 0 时 x/y/c/s 判空；
2. c/s 指针经 `aclrtPointerGetAttributes` 自动判别 Host/Device：Host 标量在 host 侧解引用
   装入 tiling，Device 标量原样透传、kernel 内从 GM 读取；
3. 路由：incx == incy == 1 且无别名 → 连续内核（40 AIV 核分片）；非单位步长 → 步长内核
   （逻辑索引均分，|inc| ≥ 1 时物理元素两两不相交，分片无竞态）；x == y 且非单位步长的
   别名场景 → 单核串行走查（保持 Netlib 顺序语义）。

#### Kernel 侧设计（crot_kernel.cpp）

**连续路径**（性能主路径）：

- 分片：`perCoreN = totalN / 40`，前 `remainder` 核多 1 元素；每核按 2048 复数元素
  （16 KiB x/y 数据）成 tile 流水；
- UB 布局（192 KiB 预算内）：ping/pong 双 slot ×（bufX/bufY/t01/t23 四区）+ 共享
  swapA/swapB + vgather 偏移表 + 8-float 符号模板；
- 流水：`copy_in(k+1)` 的 MTE2 描述符先于 tile k 的计算入队（硬件按 slot 排空标志
   gating），V 计算与 MTE3 存储、MTE2 装载三路重叠，稳态以内存带宽为瓶颈；
- **交织域 Axpy 链**（通用路径）：dav_c220 无 DeInterleave，且 transto5hd/vreducev2 实测
  仅 10.2 / ~9 G-float/s，平面化重构不敌交织域直接计算。通用路径每 1024-float 段用
  vgather 做 pair-swap（`swapA[j] = y[j^1]`），配合 ±s_i 符号模板 vmla 与 vmuls/vaxpy，
  每 1024-float 6 条向量指令完成 x'、y' 全部交叉项；
- **实数旋转快速路径**（s_imag = 0，覆盖全部性能用例）：交叉项消失，`x' = c·x + sr·y`、
  `y' = c·y − sr·x` 退化为 vmuls×2 + vaxpy×2（4 指令/1024-float），完全免去 gather；
  舍入序列与通用路径 s_i = 0 时逐位一致（IEEE 加法交换律），同一 golden 判定；
- **非有限值守卫**：Netlib 复数语义下 `0×(±Inf) = NaN` 经 swap 交叉项落在特定分量
  （如 y'.imag 因 x.real 为 Inf 而为 NaN）。快速路径跳过的零项会使 ±Inf 输入得到 ±Inf
  而非 NaN。守卫以 `v−v`（有限值为 ±0，Inf/NaN 为 NaN）构造指示向量，逐半 vadd 折叠后
  单次 V→S 读回判定；脏 tile 惰性装载 swap 表并回退通用 gather 链，精确复现金标 NaN
  语义。指示向量按 ceil(count/64) 块覆盖，尾部陈旧 lane 只会保守误判为脏、不会漏判；
- 标量管线不直接读 MTE2 写入的数据（MTE2 栅栏仅对 V 管线有序）：守卫的读回只扫描
  V 管线写就的折叠结果。

**步长路径**：按 950/arch35 同族模式，逻辑索引分段落核，32B 对齐 UB 槽逐元素组合读写。

**串行路径**：单核按 Netlib 顺序逐元素走查，覆盖别名等无法并行划分的场景。

### 性能优化方案

1. 快速路径免去 gather 后，kernel 稳态收敛到冷态 HBM 混合读写带宽（910B3 实测
   ~1.9-2.0 TB/s 聚合）；
2. 守卫成本控制：V 指示/折叠全部藏于 MTE 流水之下，标量读回一次 V→S 往返/tile；
3. s_imag = 0 时跳过 16 KiB swap 表的 MTE2 装载与符号模板同步（小 n 用例收益显著）；
4. msprof op 模式实测（Task Duration，5 warmup + 10 样本，Atlas 800T A2 / 910B3）：
   n = 2^20 / 2^21 / 2^22 = 22.7 / 41.0 / 77.1 us，对应任务书 gpu_baseline 门限
   23.7 / 56.9 / 109.3 us（1.25×标杆），三档均达标。

### 工程组织与构建安装

```
blas/rot/arch22/crot_host.cpp        # host 校验/路由/tiling/launch
blas/rot/arch22/crot_kernel.cpp      # 连续/步长/串行三路径内核
blas/rot/arch22/crot_tiling_data.h   # tiling 结构与 tile 常量
include/cann_ops_blas.h              # aclblasCrot 声明（新增，供各产品线共用）
test/rot/crot/arch22/crot_test.cpp   # CSV 驱动 GTest + 性能门禁
test/rot/crot/arch22/crot_test.csv   # 1200 用例（精度 + 性能 + 负向 + 特殊值）
test/rot/crot/gpu_baseline.csv       # GPU 标杆（性能判据源）
test/rot/crot/CMakeLists.txt         # arch22 分支注册 ops_blas_add_gtest_tests
```

## 支持硬件

| 产品 | 支持状态 |
| --- | --- |
| Atlas A2（含 Atlas 800T A2 / 910B3） | 支持（自验证设备） |
| Atlas A3 系列 | 支持（arch22 家族同构） |

## 算子约束限制

- n < 0、incx/incy 为 0 或 INT32_MIN、(n-1)×|inc| 超 int32：返回 INVALID_VALUE
  （arch22 家族契约，与 950PR/arch35 的放行口径互为已声明 per-arch 差异）；
- n = 0 快速返回 SUCCESS，不修改 x/y；
- x、y 原地更新；y 的更新使用旋转前原始 x；
- c/s 标量指针可为 Host 或 Device 内存，运行时自动判别。

# 可维可测分析

## 精度标准/性能标准

- 精度：实部/虚部分别按 FLOAT32 标准判定（rtol = atol = 2^-13，matched_ratio ≥ 0.99，
  max_abs_error ≤ max(1e-2, 32×ULP)）；Inf/NaN 特殊值按任务书等效规则单独统计；
  golden 由 Netlib/LAPACK crot 语义（含 conjg(s)）在测试工程内生成；
- 性能：msprof op 模式 Task Duration，5 warmup + ≥10 有效样本取算术平均，
  平均值 ≤ gpu_baseline.csv 对应 case 的 1.25 倍。

## 测试输入与覆盖

- 1200 CSV 用例：小 shape 基础、shape 扫描（0/1/小质数/2 的幂±1 直至 ≥2^20）、
  步长覆盖（±1/±2/±3）、c/s 特殊值（1/0/−1、(0,0)/(1,0)/(0,1)/(−1,0)）、Inf/NaN
  特殊值、对齐偏移、负向（n<0、零步长、INT32_MIN、跨度溢出、空指针）与 200 性能 case；
- 负向用例按单元素 dummy 缓冲占位，避免测试工程自身按跨度预分配导致误判；
- 正态分布（GAUSS_NORM_5_2）用例在生成器基础上补齐并纳入精度回归。

## 计时与复现

- 精度/性能回归入口：`python3 test/rot/crot/run_self_test.py --repo <repo> --device 0`
  （汇总 GTest 全量结果与三档性能采样）；
- msprof 采集：`msprof op --application="<driver> <n>" --output=<dir>`，取
  OpBasicInfo.csv 的 Task Duration(us)；配套驱动程序内置 5 warmup + 10 样本。
