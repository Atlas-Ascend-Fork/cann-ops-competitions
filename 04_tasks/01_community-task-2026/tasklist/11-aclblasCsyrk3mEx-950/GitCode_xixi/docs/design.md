# aclblasCsyrk3mEx（Ascend 950PR）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)要求在 arch35 新增复数对称 Rank-k 3M 接口。仓库已有非 Ex 的 SYRK，可复用接口组织和三角写回思路。

## 需求分析

以 X=op(A) 表示 n×k 的列主序逻辑矩阵，计算 C ← αXXᵀ + βC，原地修改 uplo 指定三角。trans 支持 N/T/C，其中 C 与 T 等价，只转置。α/β、C 为 complex64，A 支持 complex64 及交错 int8 实虚对。对角是复数，对侧三角和 ldc 填充字节保持原样。新签名按任务书 §2.3 加入 `include/cann_ops_blas.h`，参数顺序为 handle、uplo、trans、n/k、alpha、A/Atype/lda、beta、C/Ctype/ldc。

## 详细设计

### Host 与派发

校验 handle、uplo/trans、n/k、lda/ldc、标量指针及 Atype/Ctype，再根据空矩阵与标量分支校验矩阵指针；非法枚举、维数、类型分别返回仓库对应状态。n=0 无输出；k=0 或 α=0 只做 β 缩放，β=1 原位返回，β=0 直接置零所选三角。Host 标量可在发射前判值，Device 标量由 kernel 读入，产品指针仅在更新分支使用。新增 arch35 Host 与 kernel 至 `blas/syrk/arch35/`，沿用 handle 的 stream。

### 三角 tile 与 3M

按上/下三角生成外部 tile 编号。每个核独占一个或多个 C tile，完整 tile 连续写回，对角 tile 按全局 i/j 掩码进行精确长度写回。令 R、I 为 X 的实虚部，对应行 tile L、右 tile R′，每个 K 切片在同一次 kernel 内计算：

```text
P = real(XL) · real(XR′)^T
Q = imag(XL) · imag(XR′)^T
S = (real(XL)+imag(XL)) · (real(XR′)+imag(XR′))^T
real(XX^T) = P-Q
imag(XX^T) = S-P-Q
```

交错 int8 先精确提升到 FP32；complex64 输入不经 FP16/BF16 截断。由片上程序确认 arch35 FP32 矩阵通路的精度，保留 FP32 向量计算实数积的兼容路径，两路径均用 3M。A 面板向量拆分、形成和面板并按 K 分块双缓冲；三路 FP32 累加在 UB/累加缓冲中保留，epilogue 完成复数 α/β 更新。若三组 tile 超过预算，优先减小输出 tile，保持一次 kernel 发射；避免复制整个 A 或把三路乘积写入全量 GM 矩阵。

## 可维可测分析

`test/syrk/csyrk3m_ex/arch35/` 对照 CBLAS csyrk，int8 先提升后比较。附件 1000 条精度输入含 3M 专项，分别统计复数分量，rtol=atol=2^-13、匹配率≥0.99、最大绝对误差按 §3.2。设置非实对角、纯虚交叉项、T/C 同值、对侧与填充哨兵，检出误共轭与越界写入。200 条分层性能用例用 `msprof op` 采集 5 次热身后 10 次有效 kernel 时长均值，要求逐例 NPU≤2.5×GPU，全部用例 GPU/NPU 比值的算术平均≥0.5。当前为设计评审，尚无设备实测值。
