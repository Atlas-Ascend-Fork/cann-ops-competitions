# aclblasCgeru（A2/A3）算子设计文档

- 任务：CANN 训练营山东大学-aclblasCgeru 算子开发（A2/A3）
- 提交账号：gyrjgs
- 代码 Commit：9d3d979（experiment/cgeru-r35-halfwrite-6-10）
- 动态库 SHA256：078ad36ac824f836ea0c14ac29dceb09d514685a78bfa2a7dad948c42d4648f8
- 目标代码仓：https://gitcode.com/cann/ops-blas（blas/ger/ + test/ger/cgeru/arch22/）

## 需求背景

aclblasCgeru 完成 BLAS Level 2 单精度复数（complex64）**无共轭**秩 1 更新：
A := alpha * x * y^T + A（y 不取共轭，区别于 gerc 的 y^H）。列主序，支持正/负非零步长，lda ≥ max(1,m)。

调研 ops-blas master 现状：`blas/ger/arch22/` 仅有 sger，无 cgeru。本任务新增完整实现。

## 需求分析

- 参数校验：handle/alpha 非空，m≥0, n≥0, incx≠0, incy≠0, lda≥max(1,m)
- no-op：m=0/n=0/alpha=(0,0) 直接返回
- 精度：atol=rtol=2⁻¹³, ratio≥0.99, max_abs≤1e-2或32ULP（实/虚部分别）
- 性能：512≤5.7, 1024≤10.1, 2048≤55.71, 4096≤219.35 μs

## 详细设计

### Host 侧
参数校验 → 地址溢出检查 → tiling 计算 → kernel 分派。fastPath = (incx==1 && incy==1 && lda==m)。

### 通用 Kernel（cgeru_kernel.cpp）
覆盖全部形状（1024/2048/4096、非单位步长、lda padding）：
- **ProcessFastPipelinedSingle**（m≤1024 fastPath）：单行块 + 深度2 TQue 双缓冲 + 双列 Axpy 融合（Muls→Axpy→Add 替代 Muls→Muls→Add→Add，省 1 条指令/列/批）
- **ProcessFastPipelined**（m>1024）：多行块深度 2 双缓冲
- **ProcessScalar**：通用标量回退（FmaExactK 精确舍入）
- **InputsNeedFma**：2^40 极值安全网

### 512 专用 Kernel（cgeru_kernel_512.cpp 独立编译单元）
32 核 × 16 列单 A 原位 Resident：

1. **共享 Mask 固定 repeat**：SetVectorMask(64) + isSetMask=false × 48 调用，循环外一次性配置
2. **Axpy 融合**：Muls(tmpP,xInt,pRe) → Axpy(tmpP,swapSign,pIm) → Add(aResC,aResC,tmpP)，3 步替代 4 步
3. **单 A 原位**：aResidentInBuf 兼作输入/输出/MTE3 源，无独立输出矩阵（省 64KB）
4. **符号表消除**：Gather 直接到 swapSign + 偶数位原位 Muls(-1)，省 signVec 表构建
5. **分段写回重叠**：前 6 列 MTE3 写回与后 10 列 V 计算重叠
6. **x/y 极值扫描**：Abs+ReduceMax+GetValue，2^40 阈值，任何回退发生在 A 写回前

### 分核及分块
32 核 × 16 列/核，A = 64KB/核驻留 UB。colStart = blockIdx × 16。

### UB 预算
186,368B / 196,608B（182/192 KiB），余量 10,240B。

### 事件同步
evX/evYV/evYS/evA (MTE2→V/S) + evV3 (V→MTE3) + evW (MTE3→V)，全部 Alloc/Fetch+Set+Wait+Release 配对。

## 可维可测分析

### 精度
严格任务书验证器（独立于 GTest）：1200/1200 PASS（atol=rtol=2⁻¹³）。

### 性能（Atlas 800T A2 / 910B3, CANN 9.1.0）

| 尺寸 | 门槛(μs) | msprof TD(μs) | ACL Event(μs) | 状态 |
|---|---|---|---|---|
| 512 | ≤5.699 | 5.70-6.00 | 5.42-5.78 | 临界 |
| 1024 | ≤10.098 | 9.48 | 9.28-9.76 | ✅ |
| 2048 | ≤55.71 | 25.15 | 24.3-25.5 | ✅ |
| 4096 | ≤219.35 | 77.94 | 77.2-80.5 | ✅ |

### 已知限制
- 512 性能受 AICore DVFS 影响（额定 1800MHz，实际运行 800MHz），导致跨会话 0.3μs 级波动
- 1024 的 m=513-1024 fastPath 形状已全部回归通过

## 提交范围
1. blas/ger/arch22/: cgeru_host.cpp, cgeru_kernel.cpp, cgeru_kernel_512.cpp, cgeru_tiling_data.h
2. test/ger/cgeru/arch22/: cgeru_test.cpp, cgeru_perf.cpp, cgeru_test.csv, gpu_baseline.csv, README.md
3. blas/ger/README.md（产品支持表更新 A2/A3 支持）
4. include/cann_ops_blas.h（不修改，声明已存在）
