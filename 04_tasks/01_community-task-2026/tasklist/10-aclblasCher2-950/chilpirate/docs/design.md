# 需求背景（required）

## 需求来源

10月社区任务：aclblasCher2 算子开发（950）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，开发者为 `chilpirate`。在 Ascend 950PR、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数 Hermitian 秩2更新能力。

## 背景介绍

接口对齐 cuBLAS `cublasCher2`，数值 Golden 使用 Netlib BLAS 3.10.0 的 `cblas_cher2(CblasColMajor, ...)`。同族实数 Ssyr2、复数对称 Csyr2 已位于 `blas/syr2`；Cher2 新增共轭与实对角语义，复用仓内句柄、stream、复数类型和构建框架。

# 需求分析（required）

## 需求描述

计算 `A := A + alpha*x*y^H + conj(alpha)*y*x^H`，A 为列主序 complex64 Hermitian 矩阵。仅访问 uplo 指定三角，保留另一三角及 lda padding，更新后对角虚部为正零。

公开接口唯一声明于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha, const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy, aclblasComplex* A, const int lda);
```

`aclblasComplex` 沿用公共头文件的 `{float real; float imag;}`。alpha 位于 Host，x/y/A 位于 Device。原地更新 A，不返回视图，不引入平行的平台私有接口。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 结构参数 | 有效 handle；uplo 为 UPPER/LOWER；n>=0；incx/incy 非零；lda>=max(1,n)；alpha 非空 |
| quick return | 结构校验完成后，n=0 或 alpha=(0,0) 返回 SUCCESS，不访问 x/y/A，不启动 Kernel |
| 向量步长 | 非零正负步长；负步长按 Netlib 物理基址约定反向取元素 |
| 非空数据 | 正常求解要求 x/y/A 非空 |
| 三角与 padding | 仅指定三角参与读写；其余存储逐字节保持 |
| 对角 | 不以旧虚部参与运算；更新后虚部为0；quick return 不修改对角 |
| 异步 | 使用 handle 当前 stream，API 不同步设备；调用方保留存储至 stream 完成 |
| 自验 | 1000条原始精度、200条性能，补齐正态分布和随机alpha覆盖，真实C++接口执行 |

随包 `neg_n_invalid_uplo` 同时含多项错误，预期 INVALID_VALUE，故 n<0 先于枚举校验；uplo 非法且 alpha 为空时返回 INVALID_ENUM。随包 INT_MIN 步长用例按其期望返回 INVALID_VALUE，避免 abs(INT_MIN) 溢出；其他正负步长使用64位地址计算。

# 详细设计（required）

## 算子分析

### 数学公式

对列 j 先计算 `t1 = alpha*conj(y[j])`、`t2 = conj(alpha*x[j])`。

非对角指定元素为 `(A[i,j] + x[i]*t1) + y[i]*t2`；对角实部为 `real(A[j,j]) + real(x[j]*t1 + y[j]*t2)`，虚部直接写0。x[j]、y[j]同时为复数零时，按 Netlib 语义保留该列的非对角，只清零对角虚部。

### 支持数据类型

alpha、x、y、A 统一 complex64。中间乘加使用 FLOAT32，禁止半精度降精度；SIMT 使用显式 FLOAT32 中间值舍入，AIV 使用分离的 MicroAPI 乘法与加减，保持参考表达式的舍入及特殊值传播顺序。

### 支持形状

A 物理跨度为 `lda*n`，元素地址 `A[i+j*lda]`；x/y 至少具有 `1+(n-1)*abs(inc)` 个复数元素。原始 C ABI 无法推断真实分配长度，调用方保证存储有效。Host 检查矩阵字节跨度溢出。运行时 n 支持0、1及一般尺寸。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

1. 校验 handle、n、uplo、incx/incy、lda、alpha；返回仓内既有状态码。
2. 检查合法 quick return；非空计算再校验 x/y/A、跨度。
3. 查询平台 AIV 核数，取得 handle 的 stream。
4. `incx=incy=1` 且 `256<=n<=4096` 时走 AIV 向量路径；其余走通用 SIMT 路径。选择依据仅为布局、规模、平台信息。
5. 一次非空调用发射一个计算 Kernel。无 CPU fallback、无 Device 结果读回，无额外算子 workspace。

Tiling 传递 n、lda、uplo、alpha实虚部、64位incx/incy和SIMT线程数。SIMT每block使用32至256线程，按n对32取整；核数不超过实际AIV核数和列数。AIV核数取实际核数向下对齐到8，匹配下三角UB对齐策略。

#### 3.2.2 kernel侧设计

**通用路径：** 每个block按 `j=blockIdx+k*gridDim` 处理若干列，线程分配该列指定三角内的行。正步长偏移为 `i*inc`；负步长偏移为 `(n-1-i)*(-inc)`。A使用64位列主序地址。每个输出仅有一个写者，未指定三角不进入访存分支。

**连续向量路径：** 基于同族AIV实现，将x/y以精确长度DMA搬入UB并解交织为实虚平面，预计算每列t1/t2。各核处理互不重叠的列，A仅搬入该列选中连续区间，使用MicroAPI寄存器执行复数乘加，再按实际字节长度DMA写回。尾部补齐仅发生在UB。

下三角按核号mod8平移向量起点，列步长为8的倍数，使后续SoA行子视图保持32字节对齐。每平面容量4160个float，10平面共166400字节。尺寸上限4096之外直接使用通用路径，不依赖固定UB容量处理任意大向量。

两条路径均显式处理对角和零列分支。对角只使用旧实部，强制写回零虚部。AIV通过DMA/向量/标量同步保证UB复用和写回次序；跨核按列划分，无原子累加。

### 工程组织

| 位置 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | 新增唯一公共声明 |
| `blas/syr2/arch35/cher2_host.cpp` | 参数合同、路径选择、异步调度 |
| `blas/syr2/arch35/cher2_tiling_data.h` | Tiling及launch约定 |
| `blas/syr2/arch35/cher2_kernel.cpp` | 通用SIMT实现 |
| `blas/syr2/arch35/cher2_aiv_kernel.cpp` | 连续输入向量实现 |
| `test/syr2/cher2/arch35/` | CSV、GPU基线、GTest |
| `test/syr2/cher2/` | 参数读取、CBLAS对照、README、测试CMake接入、精度验证与msprof采集脚本 |

构建复用仓内按文件名前缀与架构筛选机制，测试通过 `ops_blas_add_gtest_tests` 注册。正式自验日志、报告和截图交付至个人私仓，不放入上游源码目录。

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Ascend 950PR | 支持 |

## 算子约束限制

仅支持任务约定的complex64、列主序和incx/incy/lda布局；无广播。alpha为Host指针，不进行设备指针探测或D2H标量读取。调用方保证操作数存储及异步生命周期。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 判定 | 来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_cher2`，CblasColMajor | 任务书§3.1/§3.2 |
| 普通精度 | 实/虚部分别atol=rtol=2^-13、matched_ratio>=0.99；max_abs<=max(0.01,32*ULP) | 任务书§3.2及指定opbase标准 |
| 非有限值 | 非有限值单列计数：NaN/NaN、同号Inf及NaN/Inf等效；有限/非有限不等效 | 任务书§3.2 |
| 完整性 | 选中三角对照；未引用三角/padding的NaN载荷哨兵逐字节检查；x/y只读检查 | 任务书§3.2/§3.5 |
| 性能 | 200条逐例GPU平均耗时/NPU Kernel平均耗时>=0.4；使用msprof op task duration | 任务书§3.3 |
| 采样 | 每例5次预热后采集60个有效Kernel样本取算术平均；msprof op的每个20-Kernel范围另执行5次工具预热，满足随包指导>50次口径 | 任务书及随包测试指导 |
| 内存 | 本任务不涉及内存验收，不新增内存报告要求 | 任务书§3.4 |

原始1000条精度CSV保持其参数与均匀分布语义，配对增加正态分布执行，以覆盖x/y/A随机输入的50%均匀、50%正态。另以32条随机alpha用例覆盖实虚部独立采样，均匀/正态各16条。固定alpha边界场景继续使用原值。性能清单及GPU基线保持原值，使用连续布局按原始输入口径配对。

精度通过数以GTest实际结果及逐例实/虚指标为准。完整测试含2233项，其中200项为性能用例的单次更新数值对照。性能由msprof原生数据解析，排除GTest整例墙钟耗时、输入恢复和其他计算；核对采样数量、顺序和失败状态。性能采用MSTX范围采集，每范围20个Kernel、每例3范围；仅将msprof op原生Task Duration纳入均值。容器设备映射适配只使profiler通过公开aclrtGetDevice取得设备号，不改动算子、频率或时长。多次profiling采用新的输出目录，最终报告保留原始CSV、命令及stdout/stderr，结果与提交版本对应。

## 兼容性分析

新增函数声明与arch35实现，复用既有公共复数类型，不修改Ssyr2/Csyr2接口及实现。其他产品线共用本次公开声明；其实现按各自任务独立接入。本任务仅声明Ascend 950PR支持。

## 交付与复现

按官方模板提供设计文档、测试代码与README、自验证步骤、精度/性能日志及报告材料、原始msprof证据和截图。开发侧根据真机结果填写精度、性能XLSX报告；用户填写平台验收页面。个人私仓邀请 `Ascend-CANN` 为开发者，并提供个人代码仓、分支、算子目录及完整Commit链接。

参考：[Netlib Cher2](https://www.netlib.org/blas/cher2.f)、[ops-blas贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)、[社区任务流程](https://gitcode.com/org/cann/discussions/39)。本设计中的性能数值为验收目标，实际结果另以完成后的真机报告提交。
