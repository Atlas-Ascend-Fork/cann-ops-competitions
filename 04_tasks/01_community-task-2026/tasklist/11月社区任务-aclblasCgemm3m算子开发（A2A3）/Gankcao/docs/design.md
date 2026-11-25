# Atlas A2/A3 aclblasCgemm3m 设计

## 需求背景

Gauss 3M 通过三次实数矩阵乘完成一次复数矩阵乘。ops-blas 已公开 aclblasCgemm3m，arch35 已有实现。本任务在 arch22 支持 cuBLAS cublasCgemm3m 的列主序复数运算和主机标量语义。

## 需求分析

计算 C = alpha × op(A) × op(B) + beta × C。输入及输出均为 complex64，A、B、C 位于 device，alpha 和 beta 位于 host。两个操作符分别接受 N、T 和 C，C 表示共轭转置。支持 lda、ldb、ldc 带填充的矩阵，输出仅覆盖 C 的 m×n 逻辑区域，调用方保证输出与输入存储不重叠。

## 详细设计

### 接口与调度

在 blas/gemm/arch22 下实现公共头文件声明的句柄式入口，复用 handle 的 stream。先检查句柄、枚举、非负尺寸、前导维度和必需指针。空句柄返回 HANDLE_IS_NULLPTR，非法枚举返回 INVALID_ENUM，其余非法结构参数返回 INVALID_VALUE。按转置状态核验 lda、ldb 下限，ldc 下限为 max(1,m)。元素偏移与工作区尺寸用宽整数并检查溢出。

结构合法后 m 或 n 为零时允许矩阵指针为空并直接返回。非空输出要求 C 有效。k 为零或 alpha 为零时允许 A、B 为空，仅按 beta 缩放 C；beta 为一可直接返回，beta 为零写零。通用路径在 beta 为零时也跳过旧 C 的读取。

### 分块与复数乘法

按 C 的行块、列块分核，每个任务独占一个输出块。先映射 N/T/C 到实际行列和虚部符号，再沿 K 分块读取实部、虚部。普通路径计算 P0=Ar×Br、P1=Ai×Bi、P2=(Ar+Ai)×(Br+Bi)，实部取 P0−P1，虚部取 P2−P0−P1。三个实数乘积采用 FP32 累加。最后执行复数 alpha/beta 合并与带边界保护的写回。

融合备选路径计算 Q0=(Ar+Ai)×Br、Q1=Ar×(Bi−Br)、Q2=Ai×(Br+Bi)，实部取 Q0−Q2，虚部取 Q0+Q1。对强抵消或大标量放大误差分别检查两种展开，评估带残差的纠正重算。按公开尺寸、布局、片上容量和数值条件选择路径。短 K、小矩阵限制分核与启动开销，长 K 检查归约后的累加误差。工作区按 tile 或有效平面申请，在绑定的 stream 上依次提交打包、乘法、归并，最后一次读取完成后才允许复用。工作区不足返回库状态。

## 可维可测分析

目标硬件为 Atlas A2/A3，计划使用 910B3、CANN 9.1.0 自验。在 test/gemm/cgemm3m/arch22 注册 CSV 驱动的 GTest，覆盖九种转置组合、零尺寸、非法参数、纯实纯虚、NaN/Inf、未读输入和 ldc 哨兵。记录所用 Netlib BLAS 3.10.0 版本及编译配置。以 CPU cblas_cgemm 对照，另说明 FP64 累加后收窄为 complex64 的冻结 golden。实部和虚部分别记录混合容差匹配率及最大误差，atol、rtol 均为 2⁻¹³，匹配率至少 0.99，最大绝对误差按 0.01 或 32 ULP 判定。

附件包含 1000 条精度与异常用例和 200 条性能用例。按完整参数键匹配 A100 表，键包含复数 alpha、beta 及三个前导维度。使用 msprof op kernel replay，预热 5 次后采集 10 次成功调用，按一次接口包含的全部必要 kernel 计算算术平均，要求 GPU 时间除以 NPU 时间至少 0.8。另在预置缓冲区使用 aclrtEvent 记录 5+10 的辅助均值。非零 beta 的重复采样须保证相同初始 C，恢复数据的耗时单独处理。保留逐次值、失败记录与真实终端截图。

更新 blas/gemm/README.md 的硬件支持说明，精度与性能报告、日志、冻结用例及复现说明随验收材料提交。代码 PR 按任务书在测试通过后申请。
