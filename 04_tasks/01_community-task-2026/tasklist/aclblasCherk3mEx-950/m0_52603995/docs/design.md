# 需求背景（required）

## 需求来源

11 月社区任务 `aclblasCherk3mEx`（Atlas 950PR），任务书 `aclblasCherk3mEx_Atlas950PR_task_doc.md`。团队：m0_52603995。基于 `cann/ops-blas` 的 master，在 `blas/herk/arch35/` 新增独立 Ex 接口；不依赖尚未合入的 CherkEx。

## 背景介绍

计算复数 Hermitian 秩 k 更新 `C = alpha X X^H + beta C`。N 模式 X=A；C 模式 X=A^H。A 按列主序及 lda 寻址，C 按列主序及 ldc 寻址，仅更新 uplo 指定三角。现有 cherk 的四次独立 GEMM 调用不满足本任务的 3M 单次乘积 kernel 要求。复用仓内 cgemm 的三乘积 Cube 内核及公共句柄设施，新增类型转换、共轭映射和 Hermitian 后处理。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const void* A, aclDataType Atype,
    int lda, const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

- alpha/beta 为实数 float，支持 Host/Device 标量；使用 handle 绑定 stream。
- Atype 为 ACL_COMPLEX64 或 ACL_INT8；后者每个复数由相邻 int8 real/imag 两字节组成，先精确提升至 float。Ctype 仅 ACL_COMPLEX64。
- trans 仅 N/C；T 及非法 uplo/trans 返回 INVALID_ENUM；不支持 dtype 返回 NOT_SUPPORTED。
- 校验顺序：handle → uplo/trans → n/k/lda/ldc → 标量/dtype/必需指针 → quick return。
- n=0 不启动 kernel；k=0 或 alpha=0 不读取 A，beta=1 原样返回，否则仅缩放 C 三角。beta=0 不读取旧 C。实际更新的对角虚部强制为零，quick return 保持 C。
- lda(N)≥max(1,n)，lda(C)≥max(1,k)，ldc≥max(1,n)。A/C 不重叠；不支持广播或任意 stride 视图。

## 需求拆解

新增公共 API 声明、Host 参数/tiling/工作区管理、arch35 kernel 路径、CSV 驱动 GTest、算子 README 章节与完整验收材料。保持非 Ex cherk 行为与其他架构接口兼容。

# 详细设计（required）

## 算子分析

令 X=Xr+iXi，采用 Hermitian 3M：

`P0=Xr Xr^T; P1=Xi Xi^T; P2=(Xr+Xi)(Xr-Xi)^T`。

`Re(XX^H)=P0+P1; Im(XX^H)=P2-P0+P1`。

三次实数乘在同一次 kernel launch 内完成，禁止三次独立 GEMM 调用。非融合路径允许独立的输入打包与输出后处理 kernel；性能统计计入本接口所有必要 kernel，不遗漏辅助 kernel。输入有限值和 Inf/NaN 的差异单独验证；不以参考值、case ID 或随机种子分派路径。

## 算子实现

### Host 设计

1. 按规定顺序校验结构；借助公共指针位置查询读取标量，Device 标量遵守 stream 顺序。
2. 处理 n=0、零贡献及 quick return。对实际计算路径，C 必须有效，非零贡献时 A 必须有效。
3. 以平台查询得到 AIC/AIV 数。小规模采用单核/多核 SIMT 三乘积累加与三角写回，减少 launch 固定开销；大规模优先为complex64、alpha=1、beta=0、n≥64、k≤64复用片上3M内核并启用Hermitian三角模式；其余大矩阵复用 `gemm_cgemm_triangular_products_do` 的三乘积调度，在一次 launch 内按三角 tile、乘积编号与 K partition 分发工作。
4. 以 uint64_t/size_t 计算矩阵跨度与工作区，检查分配上限与溢出。工作区归 handle 所有，遵守用户 workspace 容量限制，不缓存输入/输出值。

### Kernel 与 Tiling 设计

大矩阵路径先把 X 按 row-major `[n,kAligned]` 打包为 Xr、Xi、Xr+Xi、Xr-Xi 四个 FP32 plane，C 模式在读取时取共轭。INT8 每个分量单独精确提升；尾 K 补零，禁止越界读。共轭在这里完成一次。

Cube 入口使用已有 ND→NZ、LoadData、Mmad、Fixpipe 设施并关闭 HF32。通过行列交换适配列主序，P0/P1/P2 正确映射到 C(i,j)。初始 tile 选 32/64/128，根据 n 与平台核数调度；K 分块限制单段累加误差，后处理分段合并并按需采用补偿累加。三角 tile 只分派有效半区；对角 tile 中仅写属于 uplo 的元素。

输入打包与后处理均按32×32块通过DataCopyPad搬入UB，SIMT在UB内进行转排/合并，保证GM连续访问。后处理按 `(row,column)` 映射，每个输出只有一个 writer：合并三乘积、alpha/beta 缩放、对角虚部置零，只写选中三角，不读写 padding 和未选三角。beta=0 跳过旧 C 读取。缩放路径不读取 A。

小规模 SIMT 路径每32线程协作一个输出的K归约，短K大输出路径按输出元素分工，三个实数乘累积在单个 kernel 内完成。使用 512 线程上限，按工作量裁剪核数；stride 索引用 64 位，无全网同步。最终阈值依据真实 profiler 结果调整并记录。

### 存储与同步

| 存储 | 大小/生命周期 | 约束 |
| --- | --- | --- |
| 输入 FP32 plane ×4 | `4*n*kAligned*4` 字节 | 512B 对齐，打包结束后同 stream 消费 |
| 三乘积中间结果 | `3*parts*nAligned*nAligned*4` 字节 | K 分段；超过工作区上限时调整分块，不能溢出 |
| SIMT 用户 UB | 直接路径0；打包24576B；后处理28672B；片上混合路径196608B | 默认DCache保留，32×32 tile与尾块独立检查 |
| Cube L1/L0A/L0B/L0C | 复用 cgemm 三乘积模板静态容量断言 | 按 BS/K 阶段双缓冲，保持原有 event 生命周期 |
| 后处理累加 | 每线程寄存器 | 不覆盖仍存活输入，不用全局原子累加 |

同 stream 的 kernel 先后顺序保障打包→乘积→合并依赖。Cube 内 MTE2/MTE1/M/FIX 双缓冲 event 成对消费；每个 tile 独占输出，分段写不同 workspace 区域。SIMT store 不跨线程写同一地址。

### 性能规划

三角 3M 理论乘加工作量约 `3*n*(n+1)*k` FLOP，输入/输出最低流量由 A 类型与 C 三角确定；实际还包含打包和分段工作区流量。小形状关注 launch 开销，大形状关注 Cube 利用率与分段合并流量。所读 Roofline 技能的纯 Vector 标定不能代表混合 Cube 路径，故不填未经实测的预计 us 或达标结论。

## 支持硬件

Atlas 950PR / DAV_3510 / arch35，CANN≥9.1.0。本机第一步已确认 950PR、CANN 9.1.0；正式构建和运行另行记录。

## 算子约束限制

只新增本接口及其 arch35 实现；n/k 为非负 int，地址/工作区运算使用宽类型。异步 kernel 绑定 handle stream，调用方同步后读回。只在 Host 必需读取 Device 标量及工作区扩容时遵循仓内同步管理，不把 Host golden 混入计算路径。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判据与覆盖 | 来源 |
| --- | --- | --- |
| 精度 | 原始 1000 条精度 CSV；CBLAS Netlib 3.10.0 cherk 或 zherk golden，int8 精确提升。实虚分量 atol=rtol=2^-13，matched_ratio≥0.99，最大绝对误差≤0.01 或按 32 ULP 判据；非有限值显式匹配 | 任务书 §3.2/3.5 |
| 边界/保护 | n/k零、alpha零、beta零/一、Host/Device标量、null/非法枚举/类型/维度、两种 uplo/trans、padding、另一三角逐位不变、对角虚部、四条 TC_3M | 任务书 §2/3.5 |
| 性能 | 保留原始 200 条 case 身份；msprof op 采集所有必要 kernel，warmup5+有效10取算术均值；每例 gpu/device≥0.4，全部算术平均≥0.5；另外补测 int8 | 任务书 §3.3 |
| 内存专项 | 当前不涉及；仍检查三角及 padding 哨兵 | 任务书 §3.4 |

材料差异：README 的 MERE 说明不覆盖任务书混合容差；实际性能 CSV 200 条均为 COMPLEX64，额外补 int8 而不替换原始清单。任务包未附所谓冻结输入/golden 自验包，测试工程保留随机种子、生成逻辑与可重建数据，并在交付时记录实际保存的 frozen inputs/golden 清单。

构建：`bash build.sh --ops=cherk3m_ex --soc=ascend950`。测试位于 `test/herk/cherk3m_ex/arch35/`，使用仓内 GTest/CBLAS 框架。按 gtest_list_tests 与原始 CSV 交叉生成完整 expected.json，运行 check_validation.py 校验退出码、完整身份与零失败/跳过；性能保留 msprof 原始记录与归一化采样表。

同一实现完成一轮完整精度与性能后才冻结验收版本；修改实现/用例/判据后重新全量，禁止拼接不同轮次。当前冻结实现已完成原始1000条精度及11项补充测试，合计1011项GTest全通过；另3组默认GEMM模式回归通过。正式202项性能采集进行中，尚未宣称完整性能验收通过。

## 兼容性分析与交付

公开头文件新增接口；沿用 aclblasComplex/状态码/handle，不添加平行私有 API，不调用 CherkEx。源码验收分支与七项 task_submission 材料同步推送到已确认私有的个人 ops-blas Fork，Ascend-CANN 为开发者。基于同份目录打包并逐字节核验 task_submission.zip；正式代码 PR 在官方验收后的第三步提交。

AI 使用说明：本设计与实现工作使用 AI 辅助；所有完成状态以真实构建、测试及平台回读为准。

开发期数值细化：普通路径完成3M后，仅对非有限中间结果触发Device输出级重算。该异常路径用独立指数的双float数值表示和分量乘积，避免3M变换引入的溢出、下溢与非有限抵消；alpha/beta合并后才转换为FP32。它不启动独立GEMM，也不从Host导入结果。K分段初始256，超过2GiB工作区时扩大段长；所有验收仍使用同一任务阈值。

片上短K实现补充：OP_C且K≤8时使用16float raw行距，三个NZ平面按实际paddedRows×8紧密存放并合并搬运；完整K8/16行对齐面板不执行冗余清零。三角tile对角优先，按形状分配核数；非对角使用批量MTE3，对角按掩码写C，保证不改另一三角。正常计算不逐tile读取异常标志；每AIV在独立UB末端槽累计状态，完成写回后只查询一次，必要时对本核输出修复。默认GEMM模式保持原流程，并用三组全矩阵CBLAS zgemm回归检查。

## 当前验证状态

已完成：仓内构建、1011项完整精度及接口测试、默认GEMM mode0三组回归、原始1000条输入与Netlib golden冻结（独立测试包）。性能按原始200例加2例int8完整采集，不复用诊断结果。内存专项依任务书§3.4不涉及。正式验收结论以最终自测报告为准。
