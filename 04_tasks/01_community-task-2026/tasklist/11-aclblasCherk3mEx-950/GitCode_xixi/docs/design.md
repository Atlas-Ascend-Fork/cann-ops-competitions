# aclblasCherk3mEx（Ascend 950PR）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/5f0045789d554794aca26d146e93d6b7)要求新增 arch35 Hermitian Rank-k 3M 接口。工程结构沿用 ops-blas 的 HERK 家族。

## 需求分析

以 X=op(A)（n×k）计算 C ← αXXᴴ + βC；支持 N/C，其中 C 入口将逻辑装载阶段共轭转置 A。A 支持交错 complex64 或 int8 实虚对，C 是 complex64，α/β 是实 FP32 标量。仅更新 uplo 选定三角；计算路径的 C 对角以实数处理，写回虚部 +0，对侧与 ldc 填充保持位级相同。新增 `aclblasCherk3mEx` 公共声明，形参顺序为 handle、uplo、trans、n/k、alpha、A/Atype/lda、beta、C/Ctype/ldc。

## 详细设计

### Host 与边界路径

在 `blas/herk/arch35/` 实现 Host/Kernel，通过 handle 在同一 stream 异步运行。Host 检查句柄、枚举、n/k、前导维度、标量指针和 dtype，随后根据空维、β 缩放、产品计算路径检查 A/C。n=0 不引用 A/C；k=0 或 α=0 走 β 缩放并允许 A 为空，β=1 快速返回。Host 标量按值传递，Device 标量在 kernel 内读取；β=0 不读旧 C。

### 三角分块与融合 3M

把有效 C 预先划分成完整三角内 tile 和相交对角 tile，每个 tile 只由一个核写回。令 U、V 为 X 左右行块：

```text
P = real(U) · real(V)^T
Q = imag(U) · imag(V)^T
S = (real(U)+imag(U)) · (real(V)-imag(V))^T
real(XX^H) = P+Q
imag(XX^H) = S-P+Q
```

装载器按 N/C 统一映射 X，复数对精确提升成 FP32；上述公式中的右虚部负号负责最后一次共轭，避免与输入 C 分支混淆。每个 K 面板用 UB 形成加减组合，通过同一个 kernel 累积三组实矩阵积。优先测试 FP32 高吞吐路径；使用真实 FP32 精度的计算单元，片上余额不足时缩小输出 tile。融合 epilogue 用实 α/β 更新有效 C；跨对角 tile 逐列限制写入范围，计算对角可直接用平方和以减少虚部抵消。调度根据形状和设备片上容量选择，保持一次产品发射。

## 可维可测分析

`test/herk/cherk3m_ex/arch35/` 使用 CBLAS cherk 作参照；按任务书完成 1000 条精度与 200 条性能输入。验证 N/C、T 返回 INVALID_ENUM、上下三角、两种 Atype、异常 dtype、k=0、β=0，并在对侧与 ldc 间隙插入哨兵。分别统计实虚分量，FP32 rtol=atol=2^-13、匹配率≥0.99，同时检查 §3.2 最大误差与对角虚部规则。正式性能用 `msprof op` 采集 5 次热身后 10 次有效 kernel 时长均值，要求逐例 NPU≤2.5×GPU，全部用例 GPU/NPU 比值的算术平均≥0.5。当前为设计文档，无设备验证结果。
