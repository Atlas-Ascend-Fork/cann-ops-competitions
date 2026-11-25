# aclblasCsyrk3mEx 算子设计

## 需求背景

11 月社区任务面向 Ascend 950PR，在 ops-blas 中提供 cuBLAS Csyrk3mEx 对应的句柄式接口。仓内 csyrk 支持复数对称更新，其 Cube 主路径分别计算四个实矩阵积。Ex 接口需要在一个计算 kernel 内完成三个实矩阵积，同时接受交错复 int8 输入。

## 需求分析

给定列主序矩阵 A，令 X 为 n×k 的逻辑操作数。N 模式下 X(i,t) 取 A[i+t·lda]，T 和 C 模式均取 A[t+i·lda]。更新目标为 C(i,j) = α∑X(i,t)X(j,t)+βC(i,j)。UPPER 选择 i≤j，LOWER 选择 i≥j。保留对角虚部，未选三角与 ldc 填充保持原字节。A 的复数元素由两个 float 或两个有符号 int8 构成，输出及两个标量均为 complex64。

## 详细设计

### 接口与校验

```cpp
aclblasStatus_t aclblasCsyrk3mEx(aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda, const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

在 include/cann_ops_blas.h 增加声明。arch35 host 入口依次检查句柄、枚举、n/k、lda/ldc、标量和类型、矩阵指针。空句柄返回 HANDLE_IS_NULLPTR，非法 uplo/trans 返回 INVALID_ENUM，负尺寸、非法前导维度及必要地址为空返回 INVALID_VALUE，不支持的类型组合返回 NOT_SUPPORTED。lda 至少为 max(1,N 模式下的 n 或转置模式下的 k)，ldc 至少为 max(1,n)。

结构合法且 n=0 时直接返回。对 n>0 要求 C 有效；分别识别标量的指针位置，按绑定 stream 的顺序取得其值。k=0 或 α=0 时允许空 A，β=1 直接返回，其他情况仅缩放目标三角。β=0 直接写结果，避免输入 C 的 NaN 传播。一般路径要求 A 有效。

### 数据流与分核

一般形状先拆取 A 的实部 R、虚部 I，同时生成 S=R+I。复 int8 的两个分量分别有符号扩展到 float，所有地址偏移在乘法前扩展为 64 位。主计算按输出块分配核心，跳过整块位于未选三角的任务。每个块沿 K 依次搬入两侧面板，累加 P=RRᵀ、Q=IIᵀ、H=SSᵀ。三个乘积在同一个 kernel 中下发计算。复数积的实部为 P−Q，虚部为 (H−P)−Q。块内元素独占输出位置，无跨核原子累加。尾块按有效行列和剩余 K 长度搬运，填充位置参与片上计算时补零。小形状可在一个直接计算 kernel 内完成拆分、归约和写回，避免中间张量开销。

合成阶段对实部与虚部实施复数 α 乘法和 β 更新，对角块逐列限制写回区间。面板和中间积通过 handle workspace 管理，按对齐后的尺寸检查大小与溢出。分块根据可用 AIC/AIV 数、片上缓冲容量、n/k 与输入类型确定。对长 K 和消去误差使用分段累加并检验端点和对角线的分量误差。所有计算提交到 handle 绑定的 stream。

### 支持环境与约束

目标为 Ascend 950PR、arch35、CANN 9.1.0 及以上。A 和 C 的有效区域不得重叠。支持 lda/ldc 描述的填充，接口不提供广播。公开接口与已有非 Ex 接口并存。

## 可维可测分析

通过 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 构建。CSV 驱动的 GTest 对照 Netlib CBLAS 3.10.0 的 csyrk 或双精度 zsyrk 参考，int8 先精确提升。实虚分量采用 atol=rtol=2⁻¹³，匹配比例至少 99%，并满足最大绝对误差 0.01 或 32 ULP。未选三角与 padding 逐字节检查。覆盖原始 1000 条精度用例、非法参数优先级、设备标量、NaN/Inf、对角虚部以及不读 A/C 的退化分支。

逐行保留 200 个原始性能用例和相应 GPU 标杆，同参基线按原始序号区分，另补 int8 路径与同形状标杆比较。msprof op 记录每次调用的所有 kernel 的 Task Duration，剔除前 5 次预热，后 10 次逐调用求和再平均。每条 gpu_ms/npu_ms 至少 0.4，算术平均至少 0.5。输出原始采样、用例顺序和源码版本供复核。任务书 §3.4 无独立内存考核。
