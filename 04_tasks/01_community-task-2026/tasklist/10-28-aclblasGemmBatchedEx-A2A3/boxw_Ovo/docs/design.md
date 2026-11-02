# aclblasGemmBatchedEx 算子设计文档

## 需求背景（required）

### 需求来源

- 任务：10月社区任务-aclblasGemmBatchedEx算子开发（A2/A3）
- 任务 ID：`d945b8eabdc44fa98939d23f5cef0517`
- 任务书：官方任务包中的 `aclblasGemmBatchedEx_A2_task_doc.md`
- 参考接口：cuBLAS `cublasGemmBatchedEx`
- 目标仓：`https://gitcode.com/cann/ops-blas`

本设计以任务书的完整接口和边界契约为准。任务包中的 CSV、生成器、精度脚本和
性能基线作为自验证入口；它们的用例数量、采样次数和性能阈值与任务书存在差异，
在验收前按组织方确认的口径冻结，不能用较窄的 helper 资产删除任务书中的合法
输入或边界。

### 背景介绍

`aclblasGemmBatchedEx` 是句柄式批量通用矩阵乘法接口。每个 batch 使用独立的
A、B、C 指针，所有 batch 共享矩阵维度、leading dimension、alpha 和 beta。算子
需要在 Atlas A2/A3 的 Ascend C 路径上补齐 ops-blas 的扩展批量 GEMM 能力，并与
现有公共头文件和句柄/stream 约定保持一致。

对第 `i` 个 batch，列主序矩阵满足：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

其中 `op` 由 `transa`、`transb` 决定；实数路径下 `ACLBLAS_OP_C` 与
`ACLBLAS_OP_T` 等价。

## 需求分析（required）

### 接口与数据类型

接口声明沿用 `include/cann_ops_blas.h` 中的公共 API，不定义产品私有平行接口：

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

主路径为 `ACL_FLOAT` 的 A、B、C 与 FLOAT32 计算类型。任务书还要求扩展接口
声明的 FP16、BF16、FP8、INT8 等组合按实现和 README 的支持范围处理；由于任务书
没有给出完整合法 tuple 表，设计将把实际支持集合单独固化在 README、OpDef/Host
校验和测试矩阵中，未经确认的组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，不把
未声明的组合默认为合法输入。

### 参数、布局和约束

- A、B、C 的每个 batch 都是列主序矩阵，指针数组按 batch 顺序索引；按任务书，Aarray、Barray、Carray 的矩阵指针数组位于 Device 内存，Host 侧只负责传递合法的数组契约。
- `transa=N` 时 A 的有效行数为 `m`、列数为 `k`，`lda >= max(1,m)`；否则
  `lda >= max(1,k)`。B 按 `transb` 对称处理，C 满足 `ldc >= max(1,m)`。
- alpha 和 beta 是对应 computeType/Ctype 的标量，可位于 Host 或 Device；Host 校验
  只检查指针契约，实际读取位置遵循现有 ops-blas handle/stream 约定。
- `m`、`n`、`k`、`batchCount` 必须非负；负维度或负批数返回
  `ACLBLAS_STATUS_INVALID_VALUE`。
- `transa`、`transb` 支持 N/T/C，非法枚举返回
  `ACLBLAS_STATUS_INVALID_ENUM`。
- `Atype`、`Btype`、`Ctype`、`computeType` 和 `algo` 在当前实现支持集合之外时
  返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
- handle、alpha、beta 及在 `batchCount > 0` 时的 Aarray/Barray/Carray 必须满足
  任务书的非空约束；空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
- 参数检查先于 quick return。`m=0`、`n=0` 或 `batchCount=0` 是成功 no-op；
  `k=0` 只执行 beta 缩放；alpha 为零时不读取 A/B，仅按 beta 更新 C。
- C 原地更新，不创建额外视图；调用通过 handle 绑定的 stream 异步提交，读回
  结果前由测试同步 stream。

### 结果和错误语义

Host 侧先完成 handle、枚举、维度、leading dimension、指针和 dtype/compute/algo
组合检查，再决定 quick return 或建立 tiling。所有状态码来自
`cann_ops_blas_common.h` 的公共定义。Kernel 执行失败通过公共错误路径返回
`ACLBLAS_STATUS_EXECUTION_FAILED`，不以未定义的指针访问替代参数错误。

## 详细设计（required）

### 工程目录和公共层

- 公共声明：`include/cann_ops_blas.h`
- 公共状态码：`include/cann_ops_blas_common.h`
- A2/A3 Ascend C 实现：`blas/gemm_batched_ex/arch22/`
- 测试工程：`test/gemm_batched_ex/`

实现沿用 ops-blas 现有 handle、stream、CMake 和公共构建约定。生产候选只包含
算子实现、必要的公共声明、构建配置和测试代码；验收 runner、任务附件、原始日志
和设备材料只进入后续 `review` 分支或交付包。

### Host 侧设计

1. **参数归一化**：读取 operation、维度、leading dimension、batchCount、dtype、
   computeType 和 algo，执行统一合法性校验，计算 `op(A)`/`op(B)` 的物理行列数。
2. **快速路径**：对合法的零维、零批、`k=0` 和 alpha=0 场景生成对应 tiling，
   保证不读取任务书规定可以忽略的 A/B 数据，并保持 beta 缩放和 no-op 结果语义。
3. **批指针准备**：将 Aarray、Barray、Carray 作为 batch 指针序列传给 kernel，
   由统一的 batch index 选择每组矩阵；不把 batch 指针误当成连续大矩阵，也不引入
   未声明的 broadcast 语义。
4. **tiling 参数**：根据 `m/n/k`、转置组合、leading dimension、batchCount、
   dtype 和可用核数选择 tile 形状、batch 分片、workspace 大小和尾块信息。tiling
   数据使用固定布局和编译期大小检查，Host/Kernel 两侧字段顺序保持一致。
5. **资源与 stream**：通过现有 handle 获取 stream，workspace 由公共分配/复用约定
   管理；算子不创建跨 handle 或跨 stream 的隐式全局状态。

### Kernel 侧设计

采用 Ascend C 直调 kernel，按 batch、输出 tile 和归约 K 维组织计算：

1. 每个任务先确定 batch index 和 C tile 的列主序坐标，按 transa/transb 计算
   A/B 的物理偏移；padding leading dimension 只影响地址步长，不改变逻辑 shape。
2. 通过 DMA 将 A/B 的 tile 搬入 UB，必要时使用双缓冲覆盖搬运与 Cube/Vector
   计算延迟；K 维分块后在累加寄存器或 UB 中保持 FLOAT32 累加。
3. 对 `N/T/C` 选择同一套地址映射，实数 C 路径复用 T 的寻址分支，避免为共轭
   语义引入无意义的数据转换。
4. 使用 Cube 矩阵乘主路径处理足量的 FLOAT32 工作量，Vector 路径处理 beta 缩放、
   alpha=0、k=0 和尾块；小矩阵或不满足 Cube 对齐的 tile 使用受控尾块路径。
5. 将结果按 C 的列主序和 `ldc` 写回，严格按实际输出字节数处理尾块，不写越界
   padding。batch 之间保持独立，任何一个 batch 的指针和 tile 状态都在本 batch
   的索引范围内。
6. 所有 dispatch 分支只由已经校验的 dtype、转置、shape、batch 和算法字段选择；
   不依赖 TBE、ops_transformer 私有实现或产品私有平行 API。

### 精度设计

精度 golden 使用 Netlib BLAS 3.10.0 的 `cblas_sgemm`，逐 batch、逐元素比较完整
C 矩阵。FLOAT32 采用任务书的 `rtol=atol=2^-13`、匹配比例至少 0.99 以及
`max_abs_error <= 1e-2 或 32*ULP` 规则；Inf/NaN 按任务书的同类/等价判定单独
统计。alpha=0 的 beta 缩放路径保留任务书要求的 EXACT 特例。

测试矩阵分别记录合法 dtype/属性组合、shape、转置、padding、batch、边界和负向
状态码。任务书明确的每一项都保留，即使随包 CSV 只展示其中一部分；实际输入来源
记录为 organizer-provided test contract 或 reviewer-injected input，不能把本地
生成的张量结果写成组织方隐藏输入的验收结果。

### 性能设计

性能目标为 Atlas 800T A2（910B3）上的 FLOAT32 场景。测试采用 msprof op 的
Task Duration(us)，先执行任务书规定的 warmup，再按最终确认的有效样本数取算术
平均，逐 case 与 GPU baseline 比较。设计阶段不固化任务书中互相矛盾的 10、>10、
>50 样本口径或 1.25x、1.2x 阈值；这些字段在验收前以组织方确认结果更新矩阵和
报告。

### 可测试性和交付

测试代码读取任务包 CSV，覆盖基础、shape、填充、转置、批数、padding、边界/负向、
Inf/NaN 以及性能 case，并输出逐例状态、精度统计、Task Duration、环境和提交
身份。测试 README 同时给出普通 build/test 命令、分步复现入口和完整串联入口。

设计文档单独通过 `cann-competitions` 的设计 PR 评审，目标目录使用当前官方树中
已有的 `10-28-aclblasGemmBatchedEx-A2A3`，PR 差异仅包含本文件。代码开发在设计
合入后按 `develop/<task>` -> `submission/<task>` -> `review/<task>` 进行；
`submission` 只含生产代码，`review` 再增加验收材料和复现脚本。

## 可维可测分析（required）

| 验收项 | 设计保证 | 证据入口 |
| --- | --- | --- |
| 接口和状态码 | 公共头文件、公共错误码、Host 先校验 | API/负向用例矩阵 |
| dtype/转置 | 主路径 FLOAT32，N/T/C 全覆盖，未声明组合显式拒绝 | dtype/transpose case 表 |
| shape/leading dimension | 逻辑 shape 与物理 stride 分离，尾块按字节数写回 | shape/LD case 表 |
| 边界语义 | zero/no-op、k=0、alpha=0、空指针按任务书执行 | boundary/negative case 表 |
| 精度 | cblas_sgemm 逐 batch 全矩阵比较和非有限值统计 | accuracy report |
| 性能 | A2/910B3、msprof Task Duration、任务书确认的采样与阈值 | performance report |
| 交付复现 | 普通命令、分步入口、一键串联入口与固定 review 提交 | review README/runner |

## 兼容性分析（required）

- 目标产品为 Atlas A2/A3；实现目录使用 arch22，其他产品路径不在本任务范围内。
- CANN 目标版本为 9.1.0，接口和状态码依赖 ops-blas 公共头文件。
- 句柄、stream、异步提交和 C 原地更新语义与现有 ops-blas BLAS 接口保持一致。
- 不改变其他算子或公共算法路径；不引入 TBE、ops_transformer 私有依赖、额外
  产品接口或未映射的测试/构建文件。
