# Atlas A2/A3 单矩阵 Sgemm、Cgemm、Hgemm 设计方案

## 需求背景
三个入口使用列主序，按句柄绑定的 stream 计算 C = alpha × op(A) × op(B) + beta × C。标量位于 Host，矩阵位于 Device。目标为 CANN 9.1.0 和 arch22。基于 ops-blas master 55ef791b77f3677b45da3468b0a8b17941424f84 设计，该版本已包含 Sgemm 和 Cgemm 的 arch22 入口。

## 需求分析
Sgemm 保留现有 FP32 计算路径。检查零尺寸操作前的前导维度校验。Cgemm 保留四个实数乘积及共轭符号处理；现有入口从 Device 读回复数标量，且在 k>0 时一律要求 A/B 非空。应先核查已有调用方的标量约定，兼顾已有设备指针调用与任务规定的 Host 标量，按实际是否计算乘积决定 A/B 校验。两项都需要用本期 CSV 建立基线。

## 详细设计
Hgemm 在公开头文件声明非 batch 接口。alpha、beta、A、B、C 均为 aclblasHalf，解读为 IEEE FP16。复用仓内 tiling 与数据搬运约定，设计 AIC 以 FP16 乘、FP32 累加，最后应用 alpha、beta 再收窄到 FP16。按输出矩阵分块，多核各写独立的逻辑输出，利用 lda/ldb/ldc 处理带填充的视图。首轮先验证紧凑与带 padding 的 NN、NT、TN、TT，再验证 S/H 的共轭转置等价性。小矩阵是否需要短路径按包含完整下发链的实测结果决定。

Host 校验句柄、枚举、尺寸、前导维度和指针。m/n 为零允许 A/B/C 为空且不启动 kernel。k 或 alpha 为零时允许 A/B 为空，只缩放 C；beta 为零的写回路径不能读取旧 C。以 NaN 污染被跳过的操作数，检查结果及 padding 哨兵，避免数值上相乘为零却仍读取输入。

## 可维可测分析
扩展 test/gemm/arch22 的统一 GTest，三类官方 CSV 按算子分别消费，相同 case_name 以算子区分。精度数 S/C/H 为 334/333/333；性能为 67/67/66。S/C 对照 Netlib BLAS 3.10.0，C 实虚部分别比对。H 对照 FP64 累加、先收窄为 FP32、再舍入为 FP16 的参考输出。按任务书 §3.2 同时核验匹配率和最大误差。

910B3 上使用 msprof op 的 kernel Task Duration，5 次预热后 10 次有效采样取算术平均，与同参数 gpu_ms 换算单位后逐例比较，要求不超过 1.25 倍。使用 kernel replay，列出实际 profiler 配置。多阶段算子按一次调用的必要 kernel 合计。另在预置 Device 缓冲区上用 aclrtEvent 记录 5+10 的 avg_us，输出 [PERF] 供附件脚本解析。两套记录均随报告提交。内存专项按 §3.4 标为不适用。本文为设计，未记录实机通过结论。
