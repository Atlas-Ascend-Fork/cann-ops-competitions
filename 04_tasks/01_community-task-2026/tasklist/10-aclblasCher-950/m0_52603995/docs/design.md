# aclblasCher（Ascend 950PR）算子设计

## 需求背景（required）

### 需求来源

10月社区任务-aclblasCher算子开发（950），报名账号／团队：`@m0_52603995`。
任务书为 `aclblasCher_Atlas950PR_task_doc.md`，随包包含 1000 条精度、200 条性能用例。
设计依据社区任务官方 design_template.md；实现集成至 `cann/ops-blas`。
源码基线：`05b96ea22cea926ecfd43fc7ac691a795522bd61`（master）。

### 背景与当前实现

新增单精度复数 Hermitian 秩一更新 `A := alpha * x * x^H + A`，补齐 BLAS L2 能力。
仓内已有 `aclblasSsyr`、`aclblasCsyr`，其转置不共轭；`aclblasCherk` 为不同的 L3 接口。
复用 `blas/syr/arch35` 的句柄、stream、AIV 查询、SIMT 直调及测试构建机制；不增加 ACLNN/PyTorch 层。

## 需求分析（required）

### 接口与数据

在共享头文件 `include/cann_ops_blas.h` 新增声明：

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

| 参数 | 类型／位置 | 约束及含义 |
| --- | --- | --- |
| handle | Host 句柄 | 有效句柄，使用 handle->stream 异步入队 |
| uplo | Host 枚举 | UPPER 或 LOWER |
| n | Host int | n >= 0 |
| alpha | Host const float* | 实数 float32，非复数；本任务不要求 Device alpha |
| x | Device complex64* | n 个逻辑元素，物理跨度 1+(n-1)*abs(incx)，只读 |
| incx | Host int | 非零，负数采用 Netlib 反向索引 |
| A | Device complex64* | 列主序，索引 col*lda+row；仅指定三角原地更新 |
| lda | Host int | >= max(1,n)，padding 保持原样 |

`aclblasComplex` 沿用仓内 ABI，实／虚部各一个 FP32，不引入替代公开类型。
只支持 complex64，不涉及广播、额外非连续 Tensor、动态 shape 图接口或确定性专项。

### 参数校验顺序

1. handle 为空：HANDLE_IS_NULLPTR。
2. n < 0：INVALID_VALUE；n == 0：立即 SUCCESS，不检查其他参数或访问指针。
3. uplo 非法：INVALID_ENUM。
4. lda < n 或 incx == 0：INVALID_VALUE。
5. alpha 为空：INVALID_VALUE。
6. *alpha == 0（含 -0）：SUCCESS，不读写 x/A，不清零对角虚部。
7. x/A 为空：INVALID_VALUE。
8. 查询向量核数，传值打包 tiling，启动 kernel；查询失败返回 EXECUTION_FAILED。

任务包 TC_ED_150 将 INT_MIN 额外判为非法，与正文“任意非零 incx”存在冲突。
实现使用 int64_t 负步长与 uint64_t 矩阵索引避免 int abs 溢出，保留正文语义；
该例不使用不足长度的缓冲冒险启动，不改原始期望值冒充通过，单独提交问题说明并等待评审裁决。

## 详细设计（required）

### 数学分解与特殊值

设 x_i = r_i + i*s_i，t_r = alpha*r_j，t_i = -alpha*s_j。
非对角元素：

```text
A.real(i,j) += r_i*t_r - s_i*t_i
A.imag(i,j) += r_i*t_i + s_i*t_r
```

乘积先按 FP32 舍入，再相加减，防止融合乘加改变有限值溢出时的 Inf/NaN 分类。
沿用 Netlib CHER 的列零值分支：x_j 实部和虚部均为零时，非对角元素不变；
对角元素仍只保留实部，虚部强制写 +0。非零列的对角更新使用
`real(x_j * (alpha*conj(x_j)))`，不读取旧对角虚部。
alpha == 0 在 Host 返回，因此不会破坏 no-op 时的原始矩阵。

### Host 侧设计

新增 `cher_host.cpp`、`cher_kernel.cpp`、`cher_kernel.h`（紧凑 tiling 与启动声明放在同一头文件）。
Host 校验后将 n、lda、incx、alpha、uplo 传入 kernel，无临时 Device 分配，无跨流状态。
使用 `GetAivCoreCount()`，启动核数按可用核数和 tile 数取最小值；不写死设备核数。
偏移乘法先扩展为 64 位。最大 int 入参的元素索引可以用 uint64_t 表示；缓冲实际可用性由调用者保证。

### Kernel 侧设计与分块

采用 Ascend C SIMT，AIV-only kernel 中 `asc_vf_call` 启动线程函数。
矩阵划分为 64×64 tile，块间循环分配 tile；1024 线程对应 64 个行 lane ×16 个列 lane，
每个线程最多处理 4 个列元素。仅枚举指定三角的 tile；对角 tile 逐元素判断 row/col。
每个矩阵元素只有一个线程负责，不使用原子操作或核间同步。
尺寸较小时减少启动 tile／线程，避免大网格开销。

负步长：`xIndex(i) = incx > 0 ? int64_t(i)*incx : int64_t(n-1-i)*(-int64_t(incx))`。
通用路径使用 float 标量 GM load/store，兼容 complex 的 4 字节自然对齐；
仅在地址满足 8 字节对齐时考虑 float2 搬运优化，不额外要求调用者 8 字节对齐。
uint32_t 地址快路径必须先验证完整索引范围，超出后回退 64 位路径。
不为了对齐读取、回写未引用三角或 lda padding；尾块用 n 和三角条件保护所有访问。

```text
host: validate -> alpha quick return -> query AIV -> launch on handle stream
kernel: tile = block index; tile += launched blocks
  map triangular tile to row/column origin
  each SIMT lane computes row and up to four columns
  if out-of-range or outside selected triangle: continue
  load x_j; if zero: only clear diagonal imaginary
  load x_i; compute separately rounded products
  load old A.real; load A.imag only off diagonal
  store updated real and imag (diagonal imag = 0)
```

### UB、workspace 与同步

| 资源 | 大小 | 说明 |
| --- | --- | --- |
| 显式输入／输出 UB 缓冲 | 0 | GM 直读直写 |
| 显式中间 UB 缓冲 | 0 | 中间值位于 SIMT 寄存器 |
| SIMT DCache 预算 | 40 KiB | 950 工具链占用；不与自定义 UB 分配重叠 |
| Device workspace | 0 | 不需要 staging、归约或跨核交换 |
| tiling | 小型按值结构 | 不申请 16 MiB 框架 workspace |

函数正常路径只入队；结果读回前由调用者同步绑定 stream。输入 x 只读、A 原地更新。
指针有效性与 x/A 不重叠遵循 BLAS 调用约束。

### 性能方案

本算子 O(n²) 访存、每元素少量 FP32 运算，主要关注读写带宽及小尺寸 launch 开销。
连续行线程合并访存；三角 tile 均衡分配；在局部复用 x_i/x_j。
先取得可追溯基线，再根据 msprof op 数据决定是否引入 float2、较小线程数或分组列优化。
调优常量不是公开 API 的 n 上限。每次影响核心路径的修改复验受影响精度与性能用例。

### 支持硬件与集成

| 产品 | 支持 |
| --- | --- |
| Ascend 950PR / CANN 9.1.0 | 是，本任务验证目标 |
| 其他产品线 | 共享 API 声明，暂不声明本实现支持 |

源码 `blas/syr/arch35/cher_*` 自动参与仓内构建；测试目录 `test/syr/cher/arch35/`，
接口目录的 CMakeLists 注册 `ops_blas_add_gtest_tests`。补充 `blas/syr/README.md` 与测试 README。

## 可维可测分析

### 验证与判据

| 项目 | 计划与通过标准 |
| --- | --- |
| 构建 | `bash build.sh --ops=cher --soc=ascend950`，确认目标符号和候选库 |
| 精度 | 随包 1000 条用例完整保留，使用 Netlib BLAS 3.10.0 cblas_cher(CblasColMajor) |
| 分量统计 | 实／虚部分开，atol=rtol=2^-13，matched_ratio>=0.99，最大绝对误差<=1e-2 或 32 ULP |
| 非有限值 | 同 NaN、同号 Inf、NaN/Inf 等效；有限/非有限及异号 Inf 不等效；非有限独立计数且全部通过 |
| 保护区域 | 未引用三角、padding、输入 x 逐位不变；对角虚部写 0；quick return 整个 A 不变 |
| 补充边界 | null handle、校验优先级、4 字节对齐、非零旧对角虚部、零列与 Inf、随机 alpha 50% 均匀/50% 正态 |
| 性能 | 全部 200 条 TC_PF，msprof op task duration，5 次 warmup 后 10 个有效样本算术平均，逐 case <= GPU H100 的 2.5 倍 |
| 内存专项 | 不涉及，任务书 §3.4；功能测试仍检查边界写保护 |

性能样本数存在材料冲突：正文 §3.3 为 10 次，§7 为 >10，附件 README 为 >50。
执行时保留 §3.3 主报告，并可用同一采集补充 60 个有效样本统计，全部保留原始数据，向评审确认最终口径。
附件 verify_performance.py 使用 GTest 毫秒墙钟时间，不能作 kernel 耗时验收；工作副本将改为解析 msprof op。
INT_MIN 冲突、样本数冲突和任务目录将关联开发 RFC／问题 Issue，不私自降低验收标准。

### 兼容性与交付

新增共享 API，不修改已有 Ssyr/Csyr/Cherk 行为；Host alpha 的生命周期在函数返回后无设备依赖。
使用现有许可、编码规范和测试框架，禁止引入无关生成骨架。
交付设计 PR（须评审合入）、完整源码/CSV/测试脚本、复现 README、精度和性能 XLSX/日志/真实截图、
验收说明及 ZIP。验收代码位于私仓，邀请 Ascend-CANN 为开发者，提供分支和精确 commit 链接。
本设计中的测试均为计划，实际结果另附自验报告；评审和开发并行，正式上游代码 PR 在社区验收通过后另行提交。
