# 需求背景

## 需求来源

[10月社区任务：aclblasCsyrkx 算子开发（950）](https://www.hiascend.com/activities/task-center/details/00e0335bc557460a9ebdce3ee4d60a9b)。接口、数值语义和测试要求依据[任务书及测试附件](https://www.hiascend.com/p/resource/202610/9fcef4761b284022831aff31edca954e.zip)，目标环境为 Atlas 950PR、CANN 9.1.0。

## 背景介绍

`aclblasCsyrkx` 对两个独立的单精度复数矩阵进行乘法，仅更新结果矩阵的指定三角。输入和输出按列主序存放，复数的实部、虚部均为 FLOAT32。

以 [ops-blas 的 b82a22c6747a3cae8f6483b3cb6b36596f8e8b86](https://gitcode.com/cann/ops-blas/tree/b82a22c6747a3cae8f6483b3cb6b36596f8e8b86) 为参考，现有公共头文件提供 `aclblasCsyrk`、`aclblasSsyrkx`，尚无 `aclblasCsyrkx`。本算子沿用句柄、stream、工作区管理和 Ascend C kernel 直调方式，参考复数 GEMM 的实数分解计算和 Csyrk 的三角分块方式。

# 需求分析

## 计算语义

对选中的元素 `(i,j)` 计算：

```text
P(i,j) = sum(Aop(i,p) * Bop(j,p), p=0..k-1)
C(i,j) = alpha * P(i,j) + beta * C(i,j)
```

| 参数 | 计算与存储约定 |
| --- | --- |
| `trans=N` | A、B 均为 n×k，计算 `alpha*A*B^T + beta*C` |
| `trans=T` 或 `trans=C` | A、B 均为 k×n，计算 `alpha*A^T*B + beta*C`；C 与 T 完全等价，不取共轭 |
| `uplo=UPPER` | 更新 `i<=j` 的元素，包含对角线 |
| `uplo=LOWER` | 更新 `i>=j` 的元素，包含对角线 |
| A、B | 相互独立的只读 complex64 矩阵，各自使用 lda、ldb |
| C | n×n complex64 矩阵，原地更新选中三角；其余三角、ldc padding 和缓冲区边界保持原有位模式 |
| alpha、beta | Host 侧复数标量指针，使用仓内公共类型 `aclblasComplex` |

任务书 §2.1 明确了 `trans=C` 的特殊约定。计算只有一个 `Aop*Bop^T` 乘积，不增加 `Bop*Aop^T`，不对乘积做平均，也不向另一三角镜像写回。A、B 不相等时，两侧乘积元素通常不同，必须按指定位置计算。

附件 README 后段仍出现 OP_C 取共轭的表述，与任务书 §2.1、§3.2 及 README 前段冲突。本方案统一采用任务书明确的 T/C 等价语义，测试参数和原始用例 ID 保持不变。

## 接口与退化输入

在 `include/cann_ops_blas.h` 增加统一接口，参数顺序与任务书 §2.3 一致：

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 输入条件 | 行为 |
| --- | --- |
| 合法参数且 n=0 | 返回 SUCCESS，不启动 kernel，不访问 A/B/C |
| n>0、k=0 | 不访问 A/B，仅对选中三角执行 `beta*C` |
| n>0、k>0、alpha=(0,0) | 完成指针校验后，仅执行三角缩放，不读取 A/B 数据 |
| 仅缩放且 beta=(0,0) | 将选中三角置零，不读取原 C |
| 仅缩放且 beta=(1,0) | 返回 SUCCESS，C 的全部位模式不变 |
| 一般计算且 beta=(0,0) | 写入 `alpha*P`，不读取原 C，避免旧 C 中的 NaN 影响结果 |
| 一般计算且 beta 非零 | 仅读取选中三角的原 C，执行复数乘加 |

`k=0` 分支与 alpha 的数值无关。普通计算中的对角元素仍按复数公式计算，保留其虚部；不沿用 Hermitian 算子的实对角处理。

alpha=0 的空指针约定需评审确认：任务书 §2.4 要求 n>0、k>0 时 A/B 非空，附件 `TC_ED_370 / alpha0_a_null` 却期望 SUCCESS。本方案按任务书返回 INVALID_VALUE，需确认该用例期望或参数表是否修订。

# 详细设计

## Host 侧处理

### 参数检查

按以下次序完成检查，随后选择执行分支：

1. handle 为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 uplo、trans、n、k。非法枚举或负维度返回 `ACLBLAS_STATUS_INVALID_VALUE`，与任务书 §2.4 一致。
3. 检查 `lda, ldb >= max(1, trans==N ? n : k)`、`ldc >= max(1,n)`，以及 alpha、beta 指针非空；不满足时返回 INVALID_VALUE。
4. n=0 时返回 SUCCESS。n>0 时要求 C 非空；n>0 且 k>0 时要求 A、B 非空，失败返回 INVALID_VALUE。
5. 使用检查溢出的 64 位运算计算访存跨度、工作区大小和任务数，超出地址或长度类型可表示范围时返回 INVALID_VALUE。

各矩阵最大元素偏移为：N 模式下 A/B 为 `(k-1)*ld+(n-1)`，T/C 模式下为 `(n-1)*ld+(k-1)`，C 为 `(n-1)*ldc+(n-1)`。乘以 complex64 元素字节数前再次检查溢出；空矩阵不代入带 `-1` 的跨度公式。

参数检查完成后，将 `trans=C` 归一化为 T。Host 将所需的标量值复制到启动参数，kernel 不解引用 Host 标量地址。计算、准备和写回 kernel 均提交到 handle 绑定的同一 stream。

### 分派与分块

| 分支 | 适用条件 | 执行方式 |
| --- | --- | --- |
| 三角缩放 | k=0 或 alpha=0 | 单个向量/SIMT kernel，按列处理合法三角区间 |
| 直接计算 | 小 n、小 k，启动及搬运开销占主要部分 | 单个 FP32 kernel 完成复数点积、alpha/beta 运算与三角写回 |
| Cube 计算 | 矩阵规模足以摊薄输入准备开销 | 复数输入去交错、四个实数乘积计算、三角复数写回 |

分派使用 n、k、转置模式、工作区容量和设备资源。直接计算与 Cube 分支的切换点通过相同版本的性能测试确定，并覆盖切换点两侧的精度与耗时。

输出按正方形 tile 划分，初始采用 64×64，与同族三角选择器的块尺寸一致。令 `q=ceil(n/64)`，只调度选中三角的 `q*(q+1)/2` 个 tile。UPPER 使用 tile 行号不大于列号的任务，LOWER 使用相反条件；对角 tile 内再检查每个元素的 i、j。任务以固定编号分配到核心，同一个输出位置只由一个任务写回。若增加 128×128 配置，须同时将三角选择器的块尺寸参数化，重新验证对角与尾块。

## Kernel 计算

### 列主序寻址

以复数元素为偏移单位，收缩下标为 p：

| 模式 | Aop(i,p) | Bop(j,p) | C(i,j) |
| --- | --- | --- | --- |
| N | `A[i+p*lda]` | `B[j+p*ldb]` | `C[i+j*ldc]` |
| T/C | `A[p+i*lda]` | `B[p+j*ldb]` | `C[i+j*ldc]` |

所有分支使用同一映射。lda、ldb、ldc 分别参与寻址，不能用 n 或 k 代替带 padding 的前导维。

### 直接计算与缩放

小矩阵分支为选中三角内的输出元素分配计算任务。每个任务按 p 递增的顺序累加实部和虚部；需要线程内分组时采用固定的归并树，归并结束后只执行一次 alpha/beta 运算。各核心之间不对同一元素做原子累加。

对列 j，UPPER 的合法行区间为 `[0,j]`，LOWER 为 `[j,n-1]`。缩放分支和直接计算分支仅在该区间加载、写回 C。尾部按实际复数元素数处理；不足一个搬运块的边界使用定长元素访问，避免整块写回触及另一三角或 padding。

### Cube 主路径

输入去交错后得到 Ar、Ai、Br、Bi 四个 FP32 平面。复数乘积分解为：

```text
Prr = Ar * Br^T    Pii = Ai * Bi^T
Pri = Ar * Bi^T    Pir = Ai * Br^T
Pr  = Prr - Pii    Pi  = Pri + Pir
```

这里的 A、B 平面已按前述 Aop、Bop 的逻辑行组织，T/C 模式不会再引入共轭。复用 [Cgemm 的 FP32 乘积计算实现](https://gitcode.com/cann/ops-blas/blob/b82a22c6747a3cae8f6483b3cb6b36596f8e8b86/blas/gemm/arch35/cgemm_compute_kernel.cpp)组织四路乘积；输入、乘积和累加均采用 FP32，关闭 HF32。四路乘积在同一个计算阶段调度，减少独立 kernel 启动。

使用 [Csyrk 的三角任务划分](https://gitcode.com/cann/ops-blas/tree/b82a22c6747a3cae8f6483b3cb6b36596f8e8b86/blas/syrk/arch35)时，将左右输入改为独立 A、B。Csyrk 针对 A=A 的平方和、对角修正不适用于本接口。首版使用四实乘积的 FP32 方案，避免三乘积拆分和输入降精度引入额外抵消误差。

每个 tile 沿 K 维使用固定顺序完成累加；首版 splitK=1，覆盖通用 Cgemm 在长 K 下自动启用两分区的策略。K 尾块只补齐局部缓冲的无效元素，不越界读取输入。输入搬运使用双缓冲，缓冲复用前等待对应计算或搬运事件完成。底层原生 FP32 模板按 `K_L1=32768/B`、`K_L0=8192/B` 组织 K 分块；选择 B=64 时仍须按实际缓冲数核对 L1、L0A/B、L0C 占用。

### 复数写回

令 `alpha=ar+i*ai`、`beta=br+i*bi`，原 C 为 `cr+i*ci`，最终写回：

```text
out_r = ar*Pr - ai*Pi + br*cr - bi*ci
out_i = ar*Pi + ai*Pr + br*ci + bi*cr
```

beta=0 使用单独分支，省略原 C 的加载和 beta 项运算。每个 tile 先根据全局 i、j 判断 uplo，再读取原 C、写回实虚部。对角 tile 的行列边界、矩阵尾 tile 和 ldc padding 均使用实际有效范围。未选中的元素不写回，保留包括 NaN payload、正负零在内的原始位模式。

通用 Cgemm 的 epilogue 会覆盖整个输出矩阵，因此本算子使用独立的三角 epilogue。对角线、尾部和非选中区域的处理不修改公共 GEMM 的行为。

## 工作区与工程接入

工作区沿用 [handle 内部管理接口](https://gitcode.com/cann/ops-blas/blob/b82a22c6747a3cae8f6483b3cb6b36596f8e8b86/blas/common/helper/aclblas_handle_internal.h)的 `GetEffectiveWorkspace`、`GetEffectiveWorkspaceSize` 和 `EnsureDefaultWorkspace`。默认工作区按需求申请，遵守句柄现有的 2 GiB 上限；用户提供的工作区按实际容量使用。

令 `a512(x)=512*ceil(x/512)`、`a16(x)=16*ceil(x/16)`。一个输出分组覆盖 `Mg×Ng`，输入准备覆盖完整 k，splitK=1 时所需空间为 `2*a512(4*Mg*k) + 2*a512(4*k*Ng) + 16*a16(Mg)*a16(Ng)` 字节，对应四个输入平面和四个 FP32 乘积平面。该公式不包含 A/B/C 本身的存储；用户工作区起始地址的对齐损耗另行扣除。

完整问题超出可用容量或句柄上限时，按输出行列分组缩小 Mg、Ng，复用同一工作区。每组依次执行准备、乘积、写回，同一 stream 的前一组写回完成后才覆盖其临时数据。每组沿完整 K 维计算，alpha 和 beta 各应用一次；最小处理块仍放不下时返回 `ACLBLAS_STATUS_ALLOC_FAILED`。准备阶段在工作区补零，不读写 C。

实现放入 `blas/syrkx/arch35/`，包含 Host、kernel 和必要的 tiling 数据；测试按仓内 family/operator/arch 结构接入 `test/syrkx/csyrkx/arch35/`，归属任务书 §5 指定的 syrkx 测试目录。公共声明、算子构建注册和测试构建注册按仓库已有方式增加。共享复数乘积计算只提取实际需要的通用部分，保留既有 Cgemm、Csyrk 入口和数值配置。

# 可维可测分析

## 功能与精度验证

完整执行附件 1000 条精度用例及 200 条性能用例的计算正确性检查。输入先按 complex64 量化，golden 使用 Netlib BLAS 3.10.0 的 `cblas_cgemm`：N 对应 NoTrans/Trans，T/C 均对应 Trans/NoTrans；不使用 ConjTrans。golden 对未选中三角恢复调用前原值，仅对选中三角的实、虚分量进行数值比较。

按任务书 §3.2 和其引用的[生态精度标准 §2.1.2](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，分别对选中三角的实部、虚部计算以下指标：

```text
e(i) = abs(actual(i) - golden(i))
分量匹配条件：e(i) <= 2^-13 + 2^-13 * abs(golden(i))
matched_ratio = 匹配的有限值比对点数 / 全部有限值比对点数
i_max = 最大绝对误差所在的有限值比对点
max_abs_error = e(i_max)
max_abs_error_limit = max(1e-2, 32 * ULP(float32(golden(i_max))))
```

实部、虚部均须满足 `matched_ratio>=0.99` 且 `max_abs_error<=max_abs_error_limit`。误差以 FP64 计算，最大绝对误差在全部有限值比对点中选取，包含未满足混合容差的那部分点；`1e-2` 是固定绝对误差限值，标准表中的 `or` 表示与 32 ULP 取较大值。ULP 使用最大误差点的 FLOAT32 值域，零值和次正规数间隔为 `2^-149`。CSV 中的 MERE/MARE 配置不替代上述判定。

actual、golden 均有限时才计入有限值分母。其余比对点按任务书 §3.2 的非有限规则逐点判定，要求全部通过；有限值与非有限值不匹配、异号 Inf 等情况判失败。没有有限值比对点时，有限值两项不作分母为零的计算，以非有限值判定和位模式保护检查决定结果。

补充验证覆盖以下内容：

- A、B 不相等且含非零虚部；同一输入下 T 与 C 输出逐位一致，区分普通转置与共轭转置。
- UPPER/LOWER、全部对角元素、tile 边界两侧、不同 lda/ldb/ldc、三角边界及 padding。未选中三角、padding 和前后保护区按原始字节比较。
- n=0、k=0、alpha=0、beta=0/1、纯虚标量、非有限输入；beta=0 配合 NaN 旧 C，验证旧值未参与计算。
- 非法枚举、负维度、非法前导维、空标量指针、不同 n/k 条件下的空矩阵指针，以及 n=0 与非法参数叠加；核对返回值优先级。
- 固定输入、固定分派重复执行两次，每次恢复原 C；对全部输出按位比较。单独覆盖直接计算、Cube、缩放和工作区分组分支。
- 所需工作区的临界容量、地址对齐偏移、整数溢出检查及 stream 顺序。共享计算部分有改动时，回归受影响的 Cgemm/Csyrk 用例。

测试结果按原始 case ID 核对，要求无漏项、无跳过。子进程失败、未产生结果或仅部分用例执行完，不计为完整通过。

## 性能验证

正式计时使用 msprof op 的设备侧 Task Duration，保留清 L2 的测量条件。[任务讨论帖中的官方回复](https://gitcode.com/cann/ops-blas/discussions/56)明确采用该方式。一次 API 调用若包含输入准备、乘积、写回或多个工作区分组，则求和该次调用的全部 kernel 耗时。

先执行 5 次完整 API 预热，再固定采集 51 次有效调用，记录每次调用对应的 kernel 列表和总耗时。报告分别列出预定的前 10 次均值和全部 51 次均值，以覆盖附件现存的采样次数差异；两列分别与原始 GPU 基线关联，不混合统计。所有计入样本的 API 调用和 stream 同步均须成功。

按任务书 §3.3，200 个 PF ID 逐条计算 `GPU耗时/NPU耗时`，报告单条不少于 0.4、200 条倍率算术平均不少于 0.5 的结果；同时列出逐条是否达到 1.0。§7 则要求超过 10 次采样且达到 GPU 数据，附件 README 又要求超过 50 次。正式验收采用的性能门槛和样本数需任务方在设计评审中确认。

性能评估覆盖小规模、窄 K、大规模、两种三角和转置模式，并检查分派边界。冻结版本须在完整正确性通过后采集全部 200 条性能，不以局部测试代替全量结论。记录源码、二进制、CANN、设备、工具版本和原始采样；降频或采样异常单独标记，不用于宣称稳定达标。
