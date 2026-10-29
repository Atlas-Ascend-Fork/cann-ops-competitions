# 需求背景（required）

## 需求来源

10月社区任务《aclblasCgemm3m 算子开发（950）》任务书：在昇腾 Atlas 950PR 上以 Ascend C 开发单精度复数（COMPLEX64）3m 算法通用矩阵乘算子 `aclblasCgemm3m`，对标 cuBLAS legacy 接口 `cublasCgemm3m`，补齐 ops-blas 仓 Blas 能力，验收后合入 `blas/gemm/arch35`。

## 背景介绍

### 目标接口

`cublasCgemm3m` 计算 `C = alpha * op(A) * op(B) + beta * C`（列主序，op 取 N/T/C），以 3m 分解（三次实数乘）求值复数乘。任务要求在 ops-blas 工程框架内以 kernel 直调句柄式 BLAS 接口实现同名同参 API（签名与 cublasCgemm3m 参数序列一一对应），并满足：

- 精度：COMPLEX64 按 FLOAT32 分量比对，混合容差（rtol/atol 动态 2^-10/2^-16 量级、matched_ratio ≥ 0.99、逐元素 max_abs_error ≤ max(1e-2, 32·ULP)），golden 由 Netlib BLAS cgemm 生成；Inf/NaN 按任务 §3.2.4 等效规则判定
- 性能：单 case 耗时 ≤ 2.5×GPU（≥0.4×），且全部性能 case（GPU/NPU）倍率算术平均 ≥ 0.5×；口径为 warmup 5 + 10 次采样的算子耗时均值，标杆 cublasCgemm3m@H100（CUDA 12.4）

### 仓内现状分析

| 资产 | 位置 | 现状 |
| --- | --- | --- |
| sgemm3m 实数 3m 引擎 | `blas/gemm3m/arch35/` | tensor_api 自研引擎，A/B 沿 K 合并 3 实平面；实测 4096³ 3-pair 吞吐 13.49 TF/s（30.56ms），低于 3m 复数 4096³ 所需 ≥15 TF/s（27.48ms）门槛 |
| cgemm 复数管线 | `blas/gemm/arch35/` | deinterleave → products/fused/onchip → epilogue 全链路；α=1,β=0 时已走 Gauss 三乘路径，其余走四乘；全链路实测 4096³ 复数 18.25ms ≈ 22.6 TF/s |
| 复数 4M Cube 先例 | `blas/syrk/arch35/`、`blas/gemm_strided_batched/arch35/` | 复数在 arch35 Cube 的工程做法参考 |

S1 实测结论（4096³ 全链路，warmup5+10 均值）：sgemm3m 引擎 30.56ms（不达标）、cgemm 机制 18.25ms（达标且余量 33%）。故本设计选择复用 cgemm kernel 家族实现 3m，而非自拼 sgemm3m 引擎三次调用。

# 需求分析（required）

## 需求描述

在 `blas/gemm/arch35/` 实现 `aclblasCgemm3m`：COMPLEX64 列主序通用矩阵乘，transa/transb 支持 N/T/C 全组合与非法枚举报错，复数 alpha/beta 指针语义，任意 α/β 下恒以 3m 分解（三次实数乘）求值；m=0/n=0 快速返回、k=0/α=0 仅做 C=β×C、β=0 不读 C；同输入逐位一致（确定性计算）。

## 需求拆解

1. 公共头 `include/cann_ops_blas.h` 新增 `aclblasCgemm3m` 声明（任务 §2.3/§7.3 明示头文件新增，签名与其他产品线共用）
2. host 侧：参数校验（任务 §2.4 异常表逐条）、路径分发、tiling 组装，恒置 `cgemmThreeProducts=1`
3. kernel 侧：三乘平面生产（deinterleave/onchip/packed 三处）与三乘组合（fused MergeProducts/BuildSinglePartOutput、epilogue Vf/Planar）——复用 cgemm kernel 家族，无新增设备 kernel
4. 测试工程：`test/gemm3m/cgemm3m/`（1201 条任务用例 CSV 驱动 GTest + cblas golden + Inf/NaN 等效判定）
5. 精度全绿 + 性能双判据达标

# 详细设计（required）

## 算子分析

### 数学公式

`C = alpha * op(A) * op(B) + beta * C`，其中复数乘以 Karatsuba 3m 分解求值：

```
P1 = Ar·Br,  P2 = Ai·Bi,  P3 = (Ar+Ai)·(Br+Bi)     （三次实数 FP32 乘）
Cr = P1 - P2,  Ci = P3 - P1 - P2
```

该形式与任务书 §2.1 的 3m 描述逐字对应（三个实数乘 A1·B1、A2·B2、A3·B3 的线性组合得到实部 ac−bd 与虚部 ad+bc）。实虚平面由 deinterleave 阶段一次读 GM 产出（A/B 同布局 [Ar, Ai, Ar+Ai]），共轭（op=C）在平面拆分时翻虚部符号，Cube 引擎只接触实数 FP32 且 HF32 显式 DISABLE。

分解形式选择依据：初版采用仓内既有 Gauss 恒等式（P0=(Ar+Ai)Br, P1=Ar(Bi−Br), P2=Ai(Br+Bi)），在 RANDOM_ALTER 压力填充（±(i+1) 交替）下 Ci 部分和幅度约为四乘 2 倍，逐位复刻数值仿真与 NPU 4P 对照实验定位到案级 32·ULP 超线（26/64 ULP@中量级元素）；Karatsuba 的 Cr=P1−P2 与四乘 Cr 同构、配合 K 二分后两项压力用例修复，且 cgemm 全量回归 1001/1001 不受影响。

### 支持数据类型

COMPLEX64（实部/虚部均为 FLOAT32 分量）；alpha/beta 为复数标量指针（Host 内存）。

### 支持形状

m/n/k ∈ [0, int 上限]；列主序，lda/ldb/ldc 按任务 §2.4 约束（transA=N 时 lda≥max(1,m)，T/C 时 lda≥max(1,k)；transB=N 时 ldb≥max(1,k)，T/C 时 ldb≥max(1,n)；ldc≥max(1,m)）；不支持非连续步长/广播/动态 shape/视图。

## 算子实现

### 实现方案

#### host 侧设计（`blas/gemm/arch35/cgemm3m_host.cpp`）

路径分发（按任务 §2.5 边界语义）：

| 条件 | 路径 | 说明 |
| --- | --- | --- |
| m=0 ∥ n=0 | 快速返回 | SUCCESS，不执行计算、不写 C |
| k=0 ∥ α=0 | epilogue-only | C=β×C（β=0 置零/β=1 不变/其他复数缩放），不引用 A/B |
| α=1,β=0 且 k≤128 | onchip 单 kernel | 小 K 全片上 3m（TransformNz 产平面） |
| α=1,β=0 且输出 tile 覆盖满核×3 | fused 单 kernel | 3 乘片上合并直写 C |
| 其余（任意 α/β） | deinterleave → products(3P) → epilogue | 主路径；epilogue 做 Karatsuba 组合 + 复数 α/β + βC 累加 |

参数校验按任务 §2.4 异常表逐条实现（非法 trans 返回 INVALID_VALUE——注意与仓内 cgemm 的 INVALID_ENUM 不同，以任务书为准；A/B 空指针判罚带 α≠0 前置条件；handle 空返回 HANDLE_IS_NULLPTR）。

tiling 策略（沿用 cgemm 已实测参数）：

- 基础 tile 128×128（L0C 256KB / 2 输出级推导），占用不足时按 16 倍数降档（SelectAccumulationTile）
- K 分区：k>256（L1 panel 深度）按流水线需要 2 分区；**3m 精度增强：16<k≤64 强制 2 分区**——压力填充的消纳误差集中在小 K 区，二分使部分和幅度减半；中 K（64,256] 保持单分区避免双份乘积写回流量
- 任务步进：3 乘积 × tiles × partitions 跨核轮转，负载均衡；K_L1=256（512KB L1 双缓冲）、K_L0=64（L0A/B 乒乓）

#### kernel 侧设计（复用 `blas/gemm/arch35/` cgemm kernel 家族）

- deinterleave（AIV/SIMT）：interleaved COMPLEX64 → [Ar, Ai, Ar+Ai] 三平面，共轭翻符号；packed 快速路径（ld=rows）与通用路径（SIMT）双形态
- products（AIC/Cube）：128×128×64 Mmad 逐乘积累积到 L0C，Nd2Nz/LoadData2D 搬运，HF32 DISABLE，kPartitions 分段写回临时平面
- fused/onchip（MIX_AIC_1_2）：小 K/满核场景单 kernel 完成 3m 全流程（平面变换 → 3×Mmad → MergeProducts 合并 → 直写 C），含非有限值检测与直接复算修复
- epilogue（AIV）：SIMT 逐元素 Karatsuba 组合（productReal=P1−P2 与四乘同式、productImag=P3−P1−P2）+ 复数 α/β 缩放 + βC 累加（β=0 不读 C）；α=1,β=0 走 SIMD Planar 快速路径

数值路径：块内 K 顺序累加 + 分区合并，随机符号下误差 ~√K·eps（K=4096 约 4e-6，远小于容差）；β×C 与 α×P 组合按 IEEE 顺序执行，Inf/NaN 形态由测试侧按任务 §3.2.4 等效规则判定（golden cblas 将 0×Inf 折叠为 0，NPU IEEE 路径产 NaN，均判等效）。

### 工程接线

- `blas/CMakeLists.txt`：新增 `OP_SOURCE_DEPS_cgemm3m=cgemm` 源码依赖映射，`--ops=cgemm3m` 单算子构建可链接（仅 blas 目录作用域）
- `test/gemm3m/cgemm3m/`：GTest 套件（NullHandle + CSV 驱动 1201 条），golden 为 Netlib cblas cgemm（α=0 分支手工 βC 精确复现），参数解析兼容任务 CSV 负向约定（标量值列 "null"、tensor "NULLPTR"、非法枚举 999）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（ascend950 / arch35） | √ |
| Atlas A2/A3（arch22） | × |

## 算子约束限制

- 不支持非连续 Tensor、广播、动态 shape、原地/视图语义（C 为 in/out，β≠0 时读原值累加后写回）
- alpha/beta 不可为空指针；A/B 仅在 m>0,n>0,k>0,α≠0 时不可为空；m>0,n>0 时 C 不可为空
- 确定性：无随机归约路径（固定 tiling + 顺序累加），同输入多次执行逐位一致

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 非 PF 精度用例 1001/1001 全绿（含负向/边界/Inf-NaN/ld-padding/ALTER 压力）；cgemm 共享内核回归 1001/1001 | 任务书 §3.2 混合容差 + 生态算子开源精度标准 |
| 性能标准 | 200 条 PF 全表：单 case ≥0.4× 达 200/200；平均倍率 0.64（长稳×3：0.640/0.651/0.664）；4096³=18.24ms（0.596×/0.616×，msprof 抽查 17.81ms 互证） | 任务书 §3.3 双判据 |

## 兼容性分析

- 共享内核三乘平面/组合由 Gauss 切换为 Karatsuba，影响 cgemm 自身 α=1,β=0 快速路径数值——cgemm 全量回归 1001/1001 验证无回退，且 Cr 结构与四乘同构（预期更优）
- 新增 `--ops=cgemm3m` 单算子构建映射，不改公共 API 头既有声明、不影响其他算子构建
