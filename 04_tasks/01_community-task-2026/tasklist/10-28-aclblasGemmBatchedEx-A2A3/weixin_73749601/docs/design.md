# 需求背景（required）

## 需求来源

昇腾算子开源仓（https://gitcode.com/cann/ops-blas）算子补齐：基于 ops-blas 工程框架与 Ascend C 编程语言，在 `blas/gemm_batched_ex/arch22/`（Atlas A2/A3，dav-2201）实现句柄式批量混合精度通用矩阵乘法扩展接口 `aclblasGemmBatchedEx`，对标 cuBLAS `cublasGemmBatchedEx`。任务书：`aclblasGemmBatchedEx_A2_task_doc.md`（社区任务 2026-10）。

## 背景介绍

### aclblasGemmBatchedEx 算子实现优化

基于 ops-blas 工程框架使用 Ascend C 编程语言（AIC Cube 核 + AIV 向量核 kernel 直调）实现批量 GEMM 扩展算子补齐。

aclblasGemmBatchedEx 算子实现路径为：`blas/gemm_batched_ex/arch22/`（host 编排、tiling、kernel、kernel 声明共 3 文件上库）。

算子接口声明路径：`include/cann_ops_blas.h:471`（全产品线共用头，`blas/CMakeLists.txt` 自动按 SOC_ARCH_DIRS 收集 arch22）；测试路径：`test/gemm_batched_ex/arch22/`。

### aclblasGemmBatchedEx 算子实现现状分析

通过对 ops-blas 仓 GEMM 类算子现状分析，当前能力缺口如下：

- **既有实现**：`blas/gemm_batched_ex/arch35/`（Ascend 950，CATLASS/tensor_api 编程模型）支持 FP16/BF16/FP8 共 7 种 dtype 组合；arch22（Atlas A2/A3）此前无实现，A2/A3 上调用返回 NOT_SUPPORTED。
- **架构差异**：arch35 kernel 为 SIMD membase 方案（128×128×16 baseM/N/K）；arch22 需走 Ascend C Matmul 接口（`MatmulImpl` + GM 直挂输入），FP32 数据通路依赖 `SetHF32(false)` 的全精度 f32 Cube 乘。两架构不可移植，仅接口语义层参考。
- **性能基线**：任务书提供 GPU 标杆实测（gpu_baseline.csv，200 条），达标要求 = NPU 平均单次耗时 ≤ GPU 标杆 / 0.8（倍率 ≥ 0.8）；采样口径 msprof op 单算子 task duration，5 次 warmup + 10 次有效样本。
- **golden**：cblas（Netlib BLAS 3.10.0 实数实现 `cblas_sgemm`）逐 batch 生成，混合容差判定（rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤max(1e-2, 32ULP)；Inf/NaN 按 NaN/Inf 等效规则）。

### aclblasGemmBatchedEx 算子功能分析

批量矩阵乘法扩展：`C[i] = alpha·op(A[i])·op(B[i]) + beta·C[i]`，i ∈ [0, batchCount)，各批次共享 m/n/k/alpha/beta/leading dimension/转置标志，矩阵指针经设备侧指针数组独立指定。列主序（column-major）存储。

| 参数 | 参数含义 | 数据类型 | 支持取值 | 约束 |
| --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream 与 workspace） | aclblasHandle_t | — | 非空，否则 HANDLE_IS_NULLPTR |
| transa/transb | A/B 转置操作 | aclblasOperation_t | N / T / C | 实数路径 C 与 T 等价；非法枚举 INVALID_ENUM |
| m/n/k | 矩阵维度 | int | ≥ 0 | m=0/n=0/batchCount=0 合法 no-op；k=0 仅 beta 缩放 |
| alpha/beta | 标量乘数（float） | const void* | FP32 全集 | 非空；alpha=0 不引用 A/B；beta=0 时 C 不必为有效输入 |
| Aarray/Barray | 批次矩阵指针数组 | Device `void* const[]` | — | batchCount>0 时非空 |
| Atype/Btype/Ctype | 矩阵 dtype | aclDataType | ACL_FLOAT | 其他组合 NOT_SUPPORTED |
| lda/ldb/ldc | 前导维 | int | — | transa=N 时 lda≥max(1,m) 否则 ≥max(1,k)；transb=N 时 ldb≥max(1,k) 否则 ≥max(1,n)；ldc≥max(1,m) |
| Carray | 批次输出矩阵指针数组（原地覆写） | Device `void* const[]` | — | batchCount>0 时非空 |
| batchCount | 批次数 | int | ≥ 0 | =0 合法 no-op |
| computeType | 计算精度 | aclblasComputeType_t | ACLBLAS_COMPUTE_32F | 其他 NOT_SUPPORTED |
| algo | 算法选择 | aclblasGemmAlgo_t | ACLBLAS_GEMM_DEFAULT | 其他 NOT_SUPPORTED |

接口签名（`include/cann_ops_blas.h`，禁止修改、禁止平行 API）：

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

# 需求分析（required）

## 需求描述

用 Ascend C（AIC Cube 核 + AIV 向量核，kernel 直调）实现 `aclblasGemmBatchedEx` Atlas A2/A3（arch22）路径：FP32 主路径（ACL_FLOAT × ACLBLAS_COMPUTE_32F），支持 N/T/C 全部转置组合 × 任意合法 m/n/k/ld/batchCount/alpha/beta/特殊值/负向语义，精度满足生态算子开源精度标准（混合容差 FLOAT32 档），性能 200 条用例每条 NPU 平均单次耗时 ≤ GPU 标杆/0.8。

## 需求拆解

1. 语义对齐 cuBLAS cublasGemmBatchedEx（列主序、参数校验顺序、quick return、alpha=0 不引用 A/B、k=0 仅 beta 缩放、batchCount=0 no-op）。
2. 指针数组语义：Aarray/Barray/Carray 为 Device 内存指针数组，kernel 侧解引用取各批次矩阵基地址，Host 不做 D2H 读取，保持 stream 异步。
3. 精度：1200 条 CSV（1000 精度 + 200 性能，TC_PF 排除于精度统计）golden=cblas_sgemm 逐 batch，混合容差判定；alpha=0 且 beta=1 时 D=beta*C 位精确（EXACT）。
4. 性能：200 条 TC_PF 用例（全 alpha=1/beta=0）NPU msprof 单算子 task duration ≤ GPU/0.8。
5. 工程规范：3 文件上库（host/tiling/kernel），测试随 `test/gemm_batched_ex/arch22/` 合入；接口声明不动，README 产品支持表 A2/A3 标注支持。

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 i ∈ [0, batchCount)：

```
C[i] = alpha · op(A[i]) · op(B[i]) + beta · C[i]
op(X) = X        (trans = N)
      = Xᵀ       (trans = T)
      = Xᴴ       (trans = C，实数路径与 T 等价)
```

边界语义（对齐 cuBLAS/Netlib quick return）：

- m=0 或 n=0 或 batchCount=0：参数校验通过后直接返回成功，不 launch（合法 no-op）。
- k=0：不引用 A/B，C[i] = beta·C[i]；beta=1 时 C 逐位不变。
- alpha=0：不引用 A/B，C[i] = beta·C[i]（位精确单乘）；beta=1 时 C 逐位不变。
- beta=0：C 不必为有效输入，输出即 alpha·op(A)·op(B)。

### 算法与路径选择

Host 侧按任务特征分派四条路径（沿用 ops-blas arch22 既有 strided-batched GEMM 的验证架构，将批次寻址从 host 计算 stride 改为 kernel 解引用设备指针数组）：

| 路径 | 条件 | 实现 |
| --- | --- | --- |
| early-exit scale | k=0 或 alpha=0（且 beta≠0、beta≠1） | AIV `gemm_bex22_scale`：C = beta·C 逐元素单乘（位精确） |
| vector direct | batchCount<32 且 m·n ≤ 4096/max(1,k) | AIV `gemm_bex22_vector`：逐列 accumulate（FusedMulAdd 防溢出缩放 + 大 alpha Netlib 顺序路径） |
| cube direct | alpha=1 且 beta=0 | AIC `gemm_bex22_cube`：Cᵀ tile 直写 C |
| cube+vector 组合 | 其余（一般 alpha/beta） | AIC cube 先把每批次结果写为 workspace 内 8 行对齐的 Cᵀ 平面（tempLd=⌈m/8⌉·8），AIV vector 再做 C = alpha·temp + beta·C 融合 |

### 指针数组处理

- Aarray/Barray/Carray 在 Device：kernel 以 `__gm__ uint64_t*` 解引用第 batch 项得到该批次矩阵基地址，再构造 `GlobalTensor`；批次间地址无连续性假设。
- 分组分批（workspace 平面容量不足容纳全部批次时）：每次 cube launch 处理 `batchStep = floor(workspace/plane)` 个批次，tiling 携带 `batchOffset` 供 A/B 指针索引换算；组内 temp 平面复用，组内 vector epilogue 消费后进入下一组。

### Cube 内核（Matmul f32 全精度）

- `MatmulImpl<GM ND float>` + `SetHF32(false)`：不做 HF32 降精度，f32 输入全精度进 Cube。
- 列主序换算为行主 Cᵀ：Cᵀ = op(B)ᵀ·op(A)ᵀ，matmul 的 A 位输入为 op(B)、B 位输入为 op(A)，输出直写或经 temp 平面。
- 128×128 tile，K 按 4096 分片累加（`IterateAll(..., startK!=0)`），`SetOrgShape` 传物理 ld 与输出 pitch（直写 tempLd=ldc、组合 tempLd=paddedM）。
- 16/32 阶单位布局（m=n=k、lda=ldb=m）特化 `ComputeSmallCube`，单核多次 `SetSingleShape` 连发。

### Vector 内核

- `VECTOR_TILE=256` 行粒度逐列处理；transA 用 Gather（列主序 k×m 物理 A 的 op(A) 行=物理列），transB 列用 UB staging（n<8 退化为 GM 标量读）。
- 防溢出：B 元素 |bv|>1 时按 scale 缩放累加（`Muls/FusedMulAdd/Muls` 拆分，避免 finite×finite+Inf 提前 NaN）；非转置大 alpha（|alpha|>65504）走 `GemmBexFp32::Fma` 软件 FMA 保 Netlib 舍入序。
- extreme 值（|a|,|b|>1e30）触发软件 FMA 精确重建，保证混合容差下 matched_ratio。

### 校验顺序（任务书 §2.5）

handle → alpha/beta 非空 → 转置枚举合法（INVALID_ENUM）→ m/n/k/batchCount ≥0（INVALID_VALUE）→ leading dimension 约束（INVALID_VALUE）→ algo=DEFAULT（NOT_SUPPORTED）→ dtype 组合=FLOAT32×32F（NOT_SUPPORTED）→ batchCount>0 时指针数组非空（INVALID_VALUE）→ no-op 判定 → early-exit → 路径分派。参数校验先于 no-op 判定，与任务书口径一致。

##昇腾特性利用

- AIC Cube：f32 Matmul（MDL 配置，A1/B1/L0A/L0B/L0C 流水复用，CONFIG_MDL + enUnitFlag 小方阵优化）。
- AIV：256-bit 向量 FusedMulAdd/Axpy/Muls；DataCopyPad 2D staging 支持带 padding 的 ld。
- 多核：cube 任务 (n/128)·(m/128)·batch 按 block 循环均分；vector 任务 (m/256)·n·batch 均分。

# 方案实现

## 实现文件

| 文件 | 职责 |
| --- | --- |
| `blas/gemm_batched_ex/arch22/gemm_batched_ex_tiling.h` | GemmBex22Tiling 结构 + 3 个 kernel 入口声明 |
| `blas/gemm_batched_ex/arch22/gemm_batched_ex_kernel.cpp` | cube（含 16/32 特化）/vector/scale kernel 与设备指针数组解引用 |
| `blas/gemm_batched_ex/arch22/gemm_batched_ex_host.cpp` | 校验链、quick return、路径分派、workspace 分组 |
| `test/gemm_batched_ex/arch22/gemm_batched_ex_test.cpp` | GTest CSV 驱动（1200 条） |
| `test/gemm_batched_ex/arch22/gemm_batched_ex_test.csv` | 任务包官方用例 |
| `test/gemm_batched_ex/arch22/gemm_batched_ex_npu_wrapper.h` | 设备指针数组搬运与 D2H 回收 |

## 关键决策

1. **不引入跨批次连续性假设**：指针数组模式下各批次矩阵无连续性，弃用 strided 版依赖 plane 连续的 Tiny/BatchCube 多批次合并 DMA 特化，通用 cube/vector 路径兜底全部 shape；性能 case（alpha=1/beta=0）全部走 cube 直写单 kernel。
2. **workspace 组内消费**：单批次 Cᵀ 平面超 workspace 时回退 vector direct（任意 shape 正确），正常情况按容量分批，temp 平面组内复用，避免向 workspace 要求与 batchCount 成比例的容量。
3. **位精确 early-exit**：beta=1 直接返回（0 开销位精确）；beta≠0,1 用逐元素单乘 Muls kernel（与 golden 单乘同舍入），beta=0 写 +0（Netlib C=zero 分支）。
4. **dtype 白名单**：arch22 仅接入 FLOAT32×32F 主路径，其余组合维持 NOT_SUPPORTED（任务书 §2.1.3：扩展组合按 README 宣称范围内覆盖，README 已注明 FP16/BF16/FP8 为 950 路径）。

# 测试分析与设计（required）

## 精度测试

- 用例：任务包 `gemm_batched_ex_test.csv` 1200 条（TC_L0/TC_BA/TC_TR/TC_MN/TC_LD/TC_AL/TC_EX 等段），golden = CPU 参考实现逐批次计算，混合容差（rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤max(1e-2, 32·ULP)），Inf/NaN 等效判定；alpha=0&&beta=1 的 D=beta·C 位精确（EXACT 校验）。
- 命令：`python verify_accuracy.py --repo <repo> --soc ascend910b3`（或 `build/test/gemm_batched_ex/gemm_batched_ex_test --gtest_filter=-*TC_PF*`）。
- 结果：1000 条精度用例（1200 条 CSV 中排除 200 条 TC_PF 性能用例）全部通过，0 失败；覆盖转置全组合（N/T/C×N/T/C）、非对齐维度、ld padding、alpha/beta 特殊值（0/1/负值/±100）、Inf/NaN 特殊值（VALUE_NORM_INF/NAN，按任务书 NaN/Inf 等效规则分类判定）、负向用例（空指针/非法枚举/非法维度，返回码逐条核对）。测试设备 Atlas 800I A2（Ascend910B3），CANN 9.1.0，多进程并行分四段执行（log 归档 f9_a/b/c/d）。

## 性能测试

- 用例：`TC_PF_1001..1200` 200 条（全 alpha=1/beta=0），基线 gpu_baseline.csv（GPU A100 实测）。
- 采样：msprof op 模式 task duration，5 warmup + 10 有效样本取平均；判据每条 NPU ≤ GPU/0.8。
- 结果：200 条性能用例中 196 条达标（NPU ≤ GPU/0.8，msprof op task duration 中 5 warmup + 10 采样均值）；ratio 中位数 2.01、最大 11.19。4 条未达标 case（TC_PF_1009/1111/1115/1129，ratio 0.44~0.74）根因分析：TC_PF_1111/1115/1129 为 K=51~66 的宽 m×n 形状且 ldc 非 32 字节对齐（1919/870/1938×4B），每 128×128 tile 的 Matmul 标量 setup 占 aicore 时间 34~40%，同时 C 写回量（4~10MB）受访存带宽限制；256 宽 tile 方案经实测因 f32 下 L0C 布局限制产生写回错位不可用，OUTER_PRODUCT 调度在该 API 调用模式下数值无效。TC_PF_1009 为 16³×2 小 batch（kernel launch+Matmul 初始化 ~12us 主导，GPU cuBLAS 单 kernel 5.2us）。已实施的有效优化：混合 task 遍历顺序（T/T 列优先、其余行优先，TC_PF_1075 达标）、direct 路径 K-512 分片 tiling。逐 case 数据见 test/gemm_batched_ex/perf_report_910b3.csv。

## 测试结论

- 精度：1000/1000 全部通过（混合容差 + Inf/NaN 等效 + alpha=0&&beta=1 位精确），满足任务书 §3.2。
- 性能：196/200 达标（≥0.8× GPU）；4 条未达标（1 条 16³ 小 batch launch 开销、3 条短 K 宽形状非对齐 ldc），根因与优化方向已在性能结果中说明，继续列入后续调优。
- 内存：不设独立门禁（任务书 §3.4）；实现无与问题规模无关的持久 Device 缓存（workspace 仅按批分组复用 handle 的 32MB 默认分配，组内消费）。

# 资料附录（required）

- 任务书：aclblasGemmBatchedEx_A2_task_doc.md（社区任务 2026-10）
- 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
- cuBLAS GemmBatchedEx：https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
- ops-blas 仓：https://gitcode.com/cann/ops-blas
