# aclblasCgeru（A2/A3）算子设计文档

- 任务：CANN训练营山东大学-aclblasCgeru算子开发(A2/A3)
- 提交账号：hesc
- 代码仓库：https://gitcode.com/hesc/ops-blas （分支 `feat/cgeru-arch22`）
- 代码 Commit：b7445502d50e81d862deff50889ab7517b189788
- 目标代码仓：https://gitcode.com/cann/ops-blas （`blas/ger/` + `test/ger/cgeru/arch22/`）

## 需求背景（required）

### 需求来源

- 任务页面：CANN 训练营山东大学专场（2026-10-22 ~ 2026-11-15），任务书 `aclblasCgeru_Atlas800IA3_task_doc.md`
- 适配硬件：Atlas 800I/T A2、Atlas 800I A3（arch22），性能测机 Atlas 800T A2（910B3）
- CANN 版本：9.1.0
- 目标代码仓：https://gitcode.com/cann/ops-blas

### 背景介绍

`aclblasCgeru` 完成 BLAS Level 2 单精度复数（complex64）**无共轭**秩 1 更新：

```text
A := alpha * x * y^T + A        （y 不取共轭，区别于 gerc 的 y^H）
```

其中 A 为 m×n 列主序矩阵（lda×n 存储，仅更新前 m×n），x 逻辑长度 m（物理 1+(m-1)·|incx|），
y 逻辑长度 n（物理 1+(n-1)·|incy|），alpha 为 Host 侧复数标量。参数语义与 cuBLAS
`cublasCgeru` 逐参数一致；接口声明已存在于 `include/cann_ops_blas.h`（与 950PR 同名接口共用），
本任务交付 arch22 实现与测试。

调研基线（ops-blas master `e56c4cd`）：`include/cann_ops_blas.h` 已有声明；`blas/ger/arch22/`
已有 arch22 cgeru 首版实现；本交付在其基础上重构，重点解决三类问题：

1. **极端值语义**：退化 alpha（纯实/纯虚）乘 Inf/NaN 的 y 时，参考实现按普通复数乘积传播
   NaN；首版实现的零分量跳过分支会输出 ±Inf，导致 `TC_FL_103` 类用例失配（本机实测
   matchedRatio=0.5）。另外首版快路径的首块长度存在无符号下溢（`colCount*m - colStart*m`），
   使部分核整列不写、另一些核越界写坏邻列，需按本核 slab 剩余长度计算。
2. **golden 一致性**：golden 若直接依赖本机 BLAS 库的 `cblas_cgeru`，其融合乘加/重结合方式随
   库版本与编译选项漂移，Inf/NaN 填充用例（TC_FL_094/099/100）会出现 matchedRatio 0.91~0.98
   的不可复现失配。本交付将 golden 改为与 Netlib `cgeru.f` 逐式同源的本地参考实现。
3. **性能**：小中尺寸（512²/1024²）受固定流水开销影响，需压低每 chunk 的同步与搬运次数。

## 需求分析（required）

### 需求拆解

1. host 侧：参数校验（m/n≥0、alpha 非空、incx/incy≠0、lda≥max(1,m)）、no-op（m/n=0、
   alpha=(0,0) 直接返回且不触碰设备指针）、int64 域偏移计算（负步长折叠为 xStart/yStart）、
   均衡列拆分 tiling、kernel 直调。
2. kernel 侧（AIV only，列主序读-改-写，任一 A 元素仅属于一个核，无原子/跨核归约）：
   - **快路径**（incx==1 && lda==m && m%4==0 && m≤4096 且每核列数≤256）：x 一次搬运并构造
     成对交换向量 S=(-xi, xr, ...)，连续 slab 按 16 KiB chunk 流水；每 chunk 先
     `ChunkHasWork` 扫描系数平面，全零列整块跳过（保证 y=(0,0) 列的 A 原值——含 Inf/NaN——
     不被覆写）。累加采用普通 mul/sub/add 四步：`out = A + cr*x + ci*S`，与参考实现舍入
     次序一致，不引入 FMA 融合差异。
   - **通用路径**（strided x / lda>m / 超大形状）：行块（1024 行）× 逐列平面流水；x 按行
     标量聚集，y 逐列标量读取并门控。
3. 均衡列拆分：`colStart = n·blockIdx/useCoreNum`（u64 域），避免除零与尾核不均。
4. 测试：任务书 1200 条 CSV 全量（与仓内表逐行一致，无过滤）+ 13 个专项 GTest；性能 warmup 5 + 事件采样 100。

## 详细设计（required）

### host 侧设计

处理次序（非法/空计算场景不访问 Device 数据）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `m < 0 || n < 0`、`alpha == nullptr`、`incx == 0 || incy == 0`、`lda < max(1,m)` →
   `ACLBLAS_STATUS_INVALID_VALUE`（INT_MIN 步长不在此拒绝，交由寻址范围校验兜底）
3. `m == 0 || n == 0` → `SUCCESS`（不下发）
4. `alpha == (0,0)` → `SUCCESS`（Netlib quick return，不写 A）
5. `x/y/A == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
6. 寻址范围（u64 域防溢出）校验失败 → `ACLBLAS_STATUS_INVALID_VALUE`
7. stream 未绑定 → `ACLBLAS_STATUS_NOT_INITIALIZED`
8. 计算 tiling，`cgeru_kernel<<<numBlocks, nullptr, stream>>>` 直调；
   `numBlocks = min(n, GetAivCoreCount())`

tiling 字段：`m/n/lda/useCoreNum/incx/incy/xStart/yStart/alphaReal/alphaImag/useFast`
（按值传入，与仓库 arch22 惯例一致）。

### 分核及分块设计

- 均衡列拆分：核 c 负责列区间 `[n·c/C, n·(c+1)/C)`，尾核最多比其他核多 1 列。
- 快路径：A 的每列为连续 32B 对齐交错行段，核 slab 为一段连续 GM 区间，按 chunk 流水
  （VECIN/VECOUT 双缓冲）。
- 通用路径：行块 × 逐列。每列 piece 通过 aInQueue 搬入、aOutQueue 搬出，V/MTE2/MTE3
  之间以仓库惯用的 FetchEventID+Set/Wait 事件对衔接。

### kernel 侧设计

**快路径数据布局**：x 一次搬入为交错向量 X=(xr0,xi0,xr1,xi1,...)，配套
S = (-xi0,xr0,xi1,-xr1,...)（成对交换 + 偶通道取反，由 64 项种子表倍增生成的字节偏移表
经 Gather 构造，偶通道取反用向量掩码单条 Muls 完成）。每列系数 (cr, ci) 由 y slab 一次
向量化求出（incy==1 时 GatherMask 解交错 + 6 条向量指令；否则标量逐列），存入系数平面；同时生成每列
活动标志 `|yReal| + |yImag|`（仅 y_j==(0,0) 时精确为 0，Inf/NaN 分量保持非零）；每 chunk 用
`ChunkHasWork` 读活动标志标量，零 y 列整块跳过。门控按 y 本身判零（参考 `IF (Y(JY).NE.ZERO)`，
与 alpha 无关）：系数不参与跳列决策，非有限 alpha 下零 y 列的 NaN 系数不会把该列误判为活跃。

**每列段累加**（交错域 3 步，普通舍入）：

```text
out = A + cr*X + ci*S        （S 偶通道已含负号）
即   re: (A + cr*xr) - ci*xi ；im: (A + cr*xi) + ci*xr
```

与 Netlib `TEMP = ALPHA*Y(JY); A(I,J) += X(I)*TEMP` 的舍入次序一致；不使用 FMA 融合，
避免与参考实现的单/双级舍入差在极端值上翻转 Inf/NaN。

**y=(0,0) 列门控**：Netlib 参考对 `Y(JY)==0` 的列整体跳过（A 保持原值）。门控按 y 本身
判零（每列活动标志 `|yReal|+|yImag|`，与 alpha 无关，非有限 alpha 下零 y 列的 NaN 系数
不会误判为活跃）；快路径按活动标志跳过整列/整 chunk；若整核 slab 无工作则不发起搬运
（TC_FL_094 全零 y 用例；专项回归 FastPathNonFiniteAlphaZeroYColumnStaysOriginal 覆盖
alpha=(Inf,0) × 零 y 列逐位保持）。

**通用路径**：x 行块解交错为 xR/xI 平面（GatherMask），逐列标量系数后
`calcR = cr*xR - ci*xI; calcI = cr*xI + ci*xR`，Gather 交错写回 out，与 A piece 相加后
存储。列门控同快路径。

**UB 预算**（192 KiB）：快路径 m=4096 时 xKeep/xSwap/idx 共 96 KiB + chunk 队列
2×2×(16 KiB+guard) + 系数平面 ≈ 180 KiB 内；通用路径 6×4 KiB 平面 + 2 KiB 交错表 +
4 KiB 构建临时 + 3×2×8 KiB 队列 ≈ 96 KiB。

### 支持硬件

| 芯片版本 | 是否支持 |
| --- | --- |
| Atlas 800I/T A2（910B 系列，arch22） | √ |
| Atlas 800I A3（arch22） | √ |
| Ascend 950PR/DT（arch35） | 已有独立实现，本交付不涉及 |

### 算子约束限制

- 不支持超出 incx/incy/lda 语义的非连续视图（任务书本批次不要求）。
- 无 broadcast；仅 complex64。
- A 与 x/y 内存重叠未定义，不承诺。
- 异步语义依赖 `aclblasSetStream`，调用方读回前须自行同步。

## 可维可测分析（required）

### 精度标准 / 性能标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度 | golden 与 Netlib `cgeru.f` 逐式同源（列门控 + 普通复数乘积），实/虚部分别按 FLOAT32：`|a-g| ≤ 2^-13(1+|g|)`，matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32*ULP | 任务书 §3.2 + 生态算子精度标准 |
| 性能 | 910B3；warmup 5 + 采样 >100；msprof `Task Duration` 口径；512/1024/2048/4096 ≤ 5.7/10.1/55.71/219.35 µs；其余 196 条 ≤ gpu_baseline×1.25 | 任务书 §3.3 |
| 内存 | 无 GM workspace，仅 UB 内 scratch | 任务书 §3.4 |

### 兼容性分析

- 仅修改 `blas/ger/arch22/` 与 `test/ger/cgeru/arch22/`，不动 arch35、不动公共头文件
  （`aclblasCgeru` 声明已存在）、不改既有算子语义；`test/ger/cgeru/README.md` 补充 arch22。
- `cgeru_perf.cpp` 支持 `BLAS_DEVICE_ID` 环境变量选择设备，与仓库测试框架多卡分片惯例一致。

### 自测与证据

- 精度：任务书 CSV 1200 条全量（TC_FL_099/100 保持仓内 normal 填充版本，无过滤）
  + 13 条专项 GTest（含上游 2 条回归）= 1213 用例，910B3 实测 **1213/1213 PASS**
  （含 TC_FL 全零 y、Inf/NaN、退化 alpha NaN 传播、TC_ED 全部负向用例、混合列 chunk 零列守卫、
  非有限 alpha × 零 y 列逐位保持）。
- 性能（910B3 @1800MHz，msprof Task Duration 口径，事件口径为 ACL Event 100 次均值）：

同机同时段 A/B（msprof Task Duration，910B3 空闲卡，合并版与本实现背靠背采集）：

| case | 门槛(µs) | 本实现 TD(µs) | 合并版 TD(µs) | 相对合并版 |
|---|---|---|---|---|
| 512² | ≤5.70 | 8.56 | 8.38 | -2% |
| 1024² | ≤10.10 | **13.02** | 14.88 | **快 12%** |
| 2048² | ≤55.71 | **30.78 ✓** | 35.22 | **快 13%** |
| 4096² | ≤219.35 | **100.78 ✓** | 110.80 | **快 9%** |

注：本机为共享服务器，单次采集的绝对值随其他租户负载波动（512² 同一二进制曾观测到 6.8~8.9µs），
上表用同机同时段背靠背数据以保证可比性。列门控改为按 y 判零（评审修复，commit b744550）后
复测同四 case：6.82 / 11.42 / 29.54 / 100.22 µs，与上表同量级（门控改动仅为每核 ≤256 元素上
的 3 条向量指令），2048²/4096² 仍达标。

  2048²/4096² 两个口径均达标，且同机 A/B 比合并版快 13%/9%；512²/1024² 的缺口来自单次调用的
  固定流水开销（launch ≈3µs + 8 次 16KiB MTE 往返 + 事件同步，与数据量无关），平台侧同场景
  公开数据也都在 5.7~6.3µs 临界。已实测排除的加速方案：>16KiB 的单次 DataCopyPad（被截断）、多块描述符
  （blockCount/dstStride 两种语义均不产生连续布局）、切片拼接（正确但更慢）、手动事件
  流水/预取（arch22 上触发 TQue 同步死锁）。全表 200 条参考用例事件口径 105 条达标，
  未达标集中在 gpu 基线仅 3µs 的小尺寸与通用路径形状。
- 内存：算子零 GM workspace，仅 UB 内 scratch（快路径 ≈180 KiB / 通用路径 ≈96 KiB）。

## 提交范围

1. `blas/ger/arch22/`：cgeru_host.cpp、cgeru_kernel.cpp、cgeru_tiling_data.h（重构）
2. `test/ger/cgeru/arch22/`：cgeru_test.cpp（专项用例扩充 + naive golden）、
   cgeru_perf.cpp（BLAS_DEVICE_ID + 输出格式）、cgeru_npu_wrapper.h（精简）、
   cgeru_test.csv / gpu_baseline.csv（任务包官方 1200 条）
3. `test/ger/cgeru/README.md`：覆盖 arch22/arch35 用法、用例构成与验收口径
4. `include/cann_ops_blas.h` 不修改（声明已存在）；`blas/ger/README.md` 支持表已含
   Atlas A2/A3：支持
