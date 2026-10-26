# 需求背景(required)

## 需求来源

通过10月社区任务,向昇腾算子开源仓 ops-solver 贡献单精度复数 QR 分解与最小二乘算子族
(Cgeqrf/Cungqr/Cunmqr/CCgels),在 Ascend 950PR(arch35)上实现完整 LAPACK 级求解功能。
目标合入:https://gitcode.com/cann/ops-solver

## 背景介绍

### 单精度复数 QR 分解与最小二乘

ops-solver 仓为 LAPACK 级求解器库,对标 cuSolver DN。本任务为首批 QR 族接口,包含:

| NPU 接口 | 对标 cuSolver | 数学含义 |
|---|---|---|
| aclsolverCgeqrf | cusolverDnCgeqrf | A = Q·R(Householder QR 分解) |
| aclsolverCungqr | cusolverDnCungqr | 由 Householder 向量生成正交阵 Q |
| aclsolverCunmqr | cusolverDnCunmqr | C := op(Q)·C 或 C·op(Q) |
| aclsolverCCgels | cusolverDnCCgels | min‖AX-B‖_F(最小二乘) |

四个接口为**同一算子族**,核心是 Householder QR;ungqr/unmqr 为 geqrf 的后处理,gels
为 geqrf + unmqr + 三角求解的组合。

### 参考实现

- 算法:Netlib LAPACK cgeqrf2/zgeqrf(Householder,列主序)
- 对标:cuSolver DN legacy API(接口签名/参数/行为逐项对齐)
- 仓内模板:blas/geqrf_batched/arch35/(sgeqrf_batched 三件套,实数 batched QR 分解)

# 需求分析(required)

## 需求描述

在 Ascend 950PR(arch35/DAV_3510)上使用 Ascend C/CATLASS 实现单精度复数(COMPLEX64)
QR 分解、Q 生成、Q 乘法与最小二乘求解,包含 4 个主接口及各自的 _bufferSize 变体(共 8 个
API 函数)。精度满足 LAPACK 测试标准(CQRT01/02/03/16),性能 ≤ 0.35× GPU 基线,
确定性计算(bit-wise 可复现),无 CPU fallback。

## 需求拆解

1. aclsolverCgeqrf:Householder QR 分解,A 原地覆写为 R + Householder 向量,输出 TAU
2. aclsolverCungqr:由 geqrf 结果(v, tau)前 k 列 Householder 向量生成 m×n 正交阵 Q
3. aclsolverCunmqr:将 op(Q)(N 或 C,不支持 T)左乘或右乘到 C,原地覆写
4. aclsolverCCgels:geqrf + Cunmqr + 三角求解,out-of-place 写入 dX
5. 各接口配套 _bufferSize 查询 workspace 大小
6. devInfo/dinfo 写 Device 内存(0=成功,-i=第 i 个参数非法)
7. handle 式 API(aclsolverCreate/SetStream 等),支持异步执行
8. 确定性:同一输入/同一 stream 重复执行结果 bit-wise 一致

# 详细设计(required)

## 算子分析

### 数学公式

**Householder QR 分解(Cgeqrf)**:
```
对 j = 0..min(m,n)-1:
  x = A[j:m, j]  (当前列)
  α = -sign(x₀)·‖x‖₂  (复数:sign(z) = z/|z| 当 z≠0,否则 1)
  v = x - α·e₁
  v = v / v₀  (归一化,v₀=1)
  τ[j] = (Re(α) - 1) / α  (复数 Householder 系数)
  A[j:m, j+1:n] := (I - τ·v·v^H) · A[j:m, j+1:n]
  A[j:m, j] := R 部分(上三角) + v(Householder 向量存下三角)
```

**Blocked QR(分块)**:
```
for ib = 0..n step nb:
  1. Panel factorization: A[ib:m, ib:ib+nb] 的未分块 QR(逐列 Householder)
  2. Trailing update: A[ib:m, ib+nb:n] := (I - V·T·V^H) · A[ib:m, ib+nb:n]
     (V = panel Householder 向量,T = 上三角反射系数阵)
```

**Q 生成(Cungqr)**:
```
Q = I; for j = k-1 downto 0: Q := (I - τⱼ·vⱼ·vⱼ^H)·Q
或 blocked: Q = V·T^H·V^H + (I - V·V^H) 的分块形式
```

**Q 乘法(Cunmqr)**:
```
side=LEFT:  C := op(Q)·C = (I - V·T·V^H)^op · C
side=RIGHT: C := C·op(Q)
```

**最小二乘(CCgels)**:
```
1. A = Q·R (geqrf)
2. Q^H·B (unmqr, side=LEFT, trans=C)
3. R·X = Q^H·B (三角求解,取前 n 行)
```

### 支持数据类型

COMPLEX64(aclFloatComplex,float 实部 + float 虚部)

### 支持形状

m,n ∈ [1, 4096];CCgels 要求 m ≥ n;nrhs ∈ [0, 128]

## 算子实现

### 实现方案

**分块 Householder QR + 后端 GEMM 更新**:

```
[K1] Panel kernel (AIV·SIMT)
     逐列 Householder:计算 v、τ,更新 panel 内后续列
     输出:V(panel Householder 向量)、T(上三角系数阵)

[K2] Trailing update kernel (AIC·tensor_api)
     A_trail := (I - V·T·V^H) · A_trail
     分解为:W = V^H·A_trail → A_trail -= V·(T·W)
     即两次复数 GEMM + 一次减法(复数平面分解)

[K3] Scale/Finalize kernel (AIV·SIMT)
     各接口的后处理:
     - geqrf: 写 TAU, 清零下三角(保留 Householder 向量)
     - ungqr: backward accumulation 生成 Q
     - unmqr: 将 Householder 变换应用到 C
     - gels: 三角求解
```

### 确定性计算保证

- Householder 向量计算:同一输入产生确定性的 v 和 τ(无 atomics/无浮点和 reduce 顺序变化)
- Panel 内逐列串行处理(依赖前列结果,天然确定性)
- Trailing update:GEMM 采用固定 tiling(同一 shape 固定 split,不受运行时影响)
- 禁止使用 atomicAdd 等非确定性原语

### 复数处理

采用与 ctrmm 相同的**实平面分解**:
- A 拆为 Ar/Ai 两个 FP32 平面
- 复数 GEMM 分解为实数 GEMM 组合
- 复数内积(v^H·x):Σ conj(vᵢ)·xᵢ = Σ(vRe·xRe + vIm·xIm) + i·Σ(vRe·xIm - vIm·xRe)

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR(arch35 / DAV_3510) | √ |
| Atlas A2/A3(arch22) | 不支持 |

## 算子约束限制

- 仅 COMPLEX64,列主序
- Cunmqr 不支持 trans=T(返回 INVALID_VALUE)
- CCgels 要求 m ≥ n
- 确定性计算:同一输入/同一 stream 重复执行 bit-wise 一致
- 无 CPU fallback

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度-geqrf | CQRT01:‖R-Q^H·A‖/(m·ulp·‖A‖) ≤ max(5·cpu_ratio, 0.1) | LAPACK 测试标准 |
| 精度-ungqr | CQRT02:‖I-Q^H·Q‖/(m·ulp) ≤ max(5·cpu_ratio, 1.5) | LAPACK 测试标准 |
| 精度-unmqr | CQRT03:依赖 ungqr 通过(先测 ungqr) | LAPACK 测试标准 |
| 精度-gels | CQRT16/17:残差范数比 | LAPACK 测试标准 |
| 性能 | 12 case(P-01~P-12),NPU ≤ 0.35× GPU | 任务书 §3.3 |

### 测试工程

参照 ops-solver 仓 test 框架,各接口独立测试目录(cgeqrf/cungqr/cunmqr/cgels),
含精度测试(LAPACK 标准)与性能测试(msprof 采集 kernel 时间)。
golden 由 NumPy/SciPy LAPACK(complex128)生成。

## 兼容性分析

新算子族,不涉及兼容性;接口声明放入 ops-solver 公共头文件。

## 风险与应对

| 风险 | 分析 | 应对 |
|---|---|---|
| 确定性计算约束 | Householder 本身确定性,但 GEMM trailing update 的浮点求和顺序需固定 | 固定 tiling(不动态切换),禁止 atomics |
| 复数 QR 精度 | 复数 Householder 涉及共轭运算,误差传播 | LAPACK 标准比率判定,CPU 同精度参考 |
| 4096² 性能(0.35× GPU) | 4096² QR 计算量大,FLOPs ≈ (4/3)·n³·8 | 分块 QR + 复数 GEMM 优化(参照 ctrmm 经验) |
| 多接口耦合 | unmqr 依赖 ungqr 通过,gels 依赖全部 | 按依赖顺序测试:geqrf → ungqr → unmqr → gels |
