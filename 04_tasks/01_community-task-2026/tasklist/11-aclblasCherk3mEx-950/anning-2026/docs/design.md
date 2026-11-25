# aclblasCherk3mEx 算子设计

## 需求背景（required）

11月社区任务：在950PR（arch35）上补充 `aclblasCherk3mEx`，使用Ascend C和ops-blas句柄式接口，实现复数Hermitian矩阵秩-k更新。接口对标cuBLAS `cublasCherk3mEx`，数学语义参考Netlib `cherk`。

## 需求分析（required）

计算 `C = alpha·X·X^H + beta·C`。`trans=N` 时 `X=A`，`trans=C` 时 `X=A^H`。A、C按列主序及lda/ldc访问；只读写uplo指定三角，对角虚部置零，另一三角及padding保持不变。

A支持complex64和交错存储的complex-int8，后者每个复元素为两个int8，分别表示实部与虚部；C为complex64，alpha/beta为实数float32。支持Host和Device标量指针。`OP_T`及非法uplo/trans返回INVALID_ENUM，不支持的数据类型组合返回NOT_SUPPORTED。

`n=0`不访问矩阵、不启动kernel。`k=0`或`alpha=0`时不读取A；此时`beta=1`保持C不变，否则只缩放指定三角。`beta=0`不读取旧C。负维度、非法前导维度和必要指针为空按任务书返回对应状态码。

## 详细设计（required）

### 公开接口与工程

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

在`include/cann_ops_blas.h`增加公开声明；实现置于`blas/herk/arch35/`，测试置于`test/herk/cherk3m_ex/arch35/`，更新`blas/herk/README.md`。注册`cherk3m_ex`构建入口，复用仓内句柄、stream、标量地址判定和测试框架，不依赖其他尚未合入的Ex算子。

### Host与tiling

按handle、uplo/trans、n/k与lda/ldc、标量/类型/矩阵指针的顺序校验，再处理quick return。标量通过仓内公共辅助函数按其内存位置获取，Device标量读取与handle绑定stream保持顺序。

按C的输出tile划分任务，仅调度目标三角；对角tile按元素掩码写回。根据n、k、转置方式和片上容量选择tile及K分块，输入类型只影响解包和提升路径。跨度及字节数使用64位计算。分派不使用case编号、随机种子或参考结果。

### Hermitian 3M计算

令`X=Xr+iXi`，采用：

```text
P0 = Xr·Xr^T
P1 = Xi·Xi^T
P2 = (Xr+Xi)·(Xr-Xi)^T
Re(X·X^H) = P0+P1
Im(X·X^H) = P2-P0+P1
```

第二操作数按共轭转置处理，不能套用`X·X^T`的对称分解。三次实数矩阵乘在同一次kernel launch中完成，不调用三次独立GEMM拼接。

普通规模采用AIC/AIV协作：AIV按trans及lda读取A，complex-int8精确提升为float，构造三个实数乘积所需面板；面板经片上buffer送入Cube，三个乘积沿K分块累加；AIV合并实虚部并执行alpha/beta尾处理。面板使用流水缓冲和事件同步，确保跨核读写顺序及尾块边界。

小规模可采用同一3M公式的AIV路径，降低调度开销。两条路径均在单次launch内完成所需计算，依据形状与硬件容量选择。精度优化围绕分块累加和3M重组误差进行，保持验收阈值不变。

### 三角写回与退化路径

只写uplo指定区域；对角虚部显式为0。beta为0时不搬入旧C，避免旧C中的NaN参与计算；beta非0时忽略旧对角虚部。缩放路径不访问A。quick return按任务书保持C原样，包括不额外清理对角虚部。

## 可维可测分析

保留原始1000条精度用例和200条性能用例，增加Host/Device标量、空指针、三角及padding哨兵检查，并补充int8性能用例。支持OP_N/OP_C、上下三角、两种A类型、padding、非对齐规模和3M专项抵消场景。

CPU参考采用独立Netlib BLAS 3.10.0的非3M `cblas_cherk`，需要float64参考时使用`cblas_zherk`；int8输入先精确提升。实部、虚部分别检查mixed tolerance：atol=rtol=2^-13、匹配率不低于0.99，最大绝对误差满足1e-2或32 ULP；不要求3M与直接复数乘逐位一致。

正式性能使用`msprof op`的kernel Task Duration；预热5次，保留后续至少10次有效样本取算术平均。逐例GPU/NPU倍率不低于0.4，全部性能用例倍率算术平均不低于0.5；事件计时只用于开发诊断，不替代正式数据。重复参数的GPU基线按用例对应关系保留，不能用字典覆盖。

内存验收按任务书为“不涉及”。原始任务包、补充用例和工具适配分别记录，最终报告、日志及七文件按官方模板整理。

## 参考

- [Netlib CHERK](https://www.netlib.org/blas/cherk.f)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
- ops-blas：`blas/herk/arch35/cherk_host.cpp`及`blas/gemm/arch35/`内3M实现。
