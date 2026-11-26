# Atlas A2/A3 aclblasSgemmBatched 与 aclblasHgemmBatched 设计

## 需求背景与来源说明

任务 `55123ab3caa64d089f525f646155094c` 要求在 arch22 补齐 FP32/FP16 实数批量 GEMM，保持 `include/cann_ops_blas.h` 中现有 ABI，通过 handle 绑定的 stream 直调 kernel。

本方案基于公开设计 PR [#3232](https://gitcode.com/cann/cann-ops-competitions/merge_requests/3232) 和公开实现 PR [#1409](https://gitcode.com/cann/ops-blas/merge_requests/1409) 进行独立复核与交付补充，保留原实现的 Git 作者记录，不将其声明为本账号独立原创。本文重点记录接口契约、调度路径、精度语义和可复现验收方法；新增 S 16/32 紧凑方形 Mmad 路径、测试 README 与冻结数据回放/独立审计步骤。

## 接口与数据契约

每个 batch 独立计算列主序：

```text
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
```

S 的 A/B/C 与 alpha/beta 均为 FP32；H 均为 IEEE binary16。两个算子都是实数接口，`OP_C` 等价于 `OP_T`。Aarray/Barray/Carray 本体和元素均位于 Device，矩阵地址可以独立且不连续；alpha/beta 位于 Host。仅修改 C 的逻辑 m×n 区域，保留 ldc padding。调用方保证 Device 地址有效、C 之间及 C 与 A/B 之间不重叠。

Host 按 handle、transa/transb、m/n/k、lda/ldb/ldc、batchCount、标量与所需数组本体的顺序校验。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他结构参数错误返回 `ACLBLAS_STATUS_INVALID_VALUE`。m、n 或 batchCount 为零时不启动 kernel，三个数组允许为空；k 或 alpha 为零时只做 beta×C，A/B 数组允许为空而 C 必须有效；beta 为零时不读取旧 C。

## 总体实现

S/H 共用 `RealBatchArgs` 和 Host 调度实现，按实际运行时形状、转置、前导维、标量与 batch 数选择 kernel，不读取或回传 Device 指针数组到 Host，也不按测试编号或随机种子分派。

主要计算路径如下：

1. **小矩阵与缩放路径**：在单个 kernel 内处理多个 batch，减少逐 batch Host 发射。极小矩阵使用分组或打包布局，保持独立 Device 地址访问；纯缩放不读取 A/B。对 FP32、`alpha=1,beta=0`、16/32 紧凑方形且 batch>4 的场景，增加直接 Mmad 路径：逐 batch 从 Device 地址表取指针，ND→NZ 装载，FP32 累加后按真实 ldc 写回。小 batch、24³ 与极值输入仍走公开通用路径，不启用未通过全量数值复核的分派。
2. **通用 Cube 路径**：将列主序问题转写为输出转置视图，交换 A/B 与对应转置标识，把输出块、batch 和 split-K 编号映射到 AIC。Matmul 累加器为 FP32；真实 K 尾块和逻辑 M/N 边界按运行时尺寸裁剪。
3. **对齐与打包路径**：对需要 16 对齐或规则面板的形状，在 handle workspace 中生成 A/B 对齐副本或紧凑面板，再执行批量 Cube 计算。工作区按 batch 容量复用，Host 只保存地址与尺寸，不读取矩阵内容。
4. **FP16 输出与非默认标量**：Cube 内部始终以 FP32 累加。可直接写 half 的路径在累加完成后做一次 binary16 舍入；需要 FP32 临时输出、split-K 或 alpha/beta 后处理的路径，由 AIV 完成归约、缩放和最后一次 half 转换，避免中途 binary16 累加。
5. **专用面板流水线**：短 K 大面板、长 K 小 batch、方形与转置面板等形状使用打包复用、MTE/Cube 重叠和独立 AIV epilogue，减少重复搬运。通用路径保持为这些形状的语义回退，不改变公开 API。

所有 kernel 均在 handle 绑定 stream 中异步发射。Host 返回成功不代表 Device 已完成，调用方读回结果前必须同步该 stream。workspace 不足或用户提供的 workspace 不满足专用路径需求时，按已有容量选择通用路径或返回对应错误，不分配私有平行句柄/API。

## 精度设计

S/H 共用 `test/gemm_batched/sgemm_batched/arch22/` 的 CSV 驱动 GTest，通过两个官方包装入口区分算子。每个接口执行 500 条精度与 100 条性能用例。

- S 参考使用 Netlib BLAS 3.10.0 `cblas_sgemm`，容差 `atol=rtol=2^-13`。
- H 将实际 binary16 输入和标量提升至 FP64，以 `cblas_dgemm` 计算，先收窄到 FP32，再按 IEEE round-to-nearest 收窄到 binary16；容差 `atol=rtol=2^-10`。
- 每个用例匹配率至少 99%，并满足固定绝对误差或 32 ULP 硬上限；特殊值不匹配和 padding/guard 污染均为失败。
- Device 指针数组使用独立缓冲和乱序偏移，避免实现意外依赖连续 batch。
- 首次全量运行通过 `SHGB_FREEZE_ROOT` 保存真实输入与 golden，第二次通过 `SHGB_REPLAY_ROOT` 逐字节回放；独立审计脚本核对 CSV 期望集、GTest 执行集和逐元素统计，任何 MISSING 均失败。

## 性能与可维护性验证

性能设备为 Atlas 800T A2（910B3），算子按 CANN 9.1.0 编译。GTest 先 warmup 5 次，再以 `aclrtEvent` 包围连续 10 次公开接口调用，所得 `avg_us` 仅作辅助数据。

正式性能使用 `msprof op` 的 kernel `Task Duration(us)`：每例独立采集 15 次调用，剔除前 5 次并平均后 10 次；一次调用中的多个 kernel 按调用序号完整相加。逐例要求：

```text
NPU kernel 平均耗时 <= 对应 gpu_ms / 0.8
```

采集缺失、重复、样本不足、非有限值、异常进程、NO_REF 或 ratio<0.8 均不放行。测试 README 给出编译、精度、冻结回放、事件和 msprof 的完整复现步骤。任务书 §3.4 的内存要求为“不涉及”，交付报告对应标记为不适用，不构造虚假内存测试。

## 兼容性边界

不修改现有公开声明、handle 或 stream 行为，不增加私有平行公共 API，不改变 arch35 与其他架构实现。CSV 与官方附件逐字节保持一致。所有形状专用优化必须由运行时参数 predicate 决定，并接受同一通用精度、padding 与 Device 指针数组合同约束。
