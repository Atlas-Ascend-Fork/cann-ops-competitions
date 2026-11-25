# aclblasCgemm3m（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3)要求补齐 `aclblasCgemm3m` 的 arch22 实现，采用三实乘复数 GEMM。arch35 实现用作调度和接口对照。

## 需求分析

C ← α·op(A)op(B)+βC，A/B/C 及 α/β 都为 complex64，矩阵列主序，支持 N/T/C（C 表示共轭转置）。保持 `aclblasCgemm3m(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)` 公共 ABI，写回仅覆盖 C 的 m×n 有效元素。

## 详细设计

### Host 调度

放入 `blas/gemm/arch22/`。校验 handle、两个枚举、m/n/k、按转置计算的物理 leading dimension、Host 标量指针和实际引用的矩阵；m/n=0 无访问，k=0 或 α=0 只对 C 缩放并豁免 A/B。当 β=0 时不读取旧 C。输入共轭通过装载器符号控制，地址跨度用 64 位检查，发射沿用 handle 的 stream。

### Kernel 三实乘

令 X=op(A)、Y=op(B)，在同一 kernel 内对每个输出 tile、每个 K 面板计算 P=XrYr、Q=XiYi、S=(Xr+Xi)(Yr+Yi)。FP32 累加后得到实部 P−Q、虚部 S−P−Q，再应用复数 α/β。三路都在片上累加，避免三次 API 启动或三份全量 GM 中间矩阵。内存受限时缩小 K 面板或输出 tile，累加器按三组 FP32 预算；采用块内拆交错与加法成形，MTE 搬运与计算按面板双缓冲。对 arch22 保留 FP32 向量算术实现 3M 的通用路径，高吞吐矩阵微内核以不损害精度的微基准结果作为启用依据；避免在输入和面板中直接截断复数分量。每个输出 tile 单写者，按逻辑 C 行数掩码回写。

## 可维可测分析

`test/gemm/cgemm3m/arch22/` 用附件 1000 条精度和 200 条性能输入。黄金参考为独立 CBLAS cgemm，对实虚各自执行 §3.2 混合容差及最大绝对误差。专项测试纯实/纯虚、共轭、接近相消的交叉项、α/β 特值、非对齐尾块和 padding 保持；检查路径不随用例标号分派。正式比较 5 次热身后的 10 次有效 kernel `Task Duration` 均值（NPU≤1.25×GPU），同时收集 GTest 端到端 `avg_us`。实测数据留待实现阶段。
