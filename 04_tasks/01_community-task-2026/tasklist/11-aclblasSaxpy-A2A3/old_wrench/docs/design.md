# aclblasSaxpy 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasSaxpy算子开发（A2A3）；算子 aclblasSaxpy（对标 cuBLAS cublasSaxpy，CUBLAS_Library.pdf §2.5.4；Netlib `saxpy.f`）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/axpy/arch22/`（实现，与 950PR arch35 兄弟实现同目录分层）+ `test/saxpy/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；接口声明已在 `include/cann_ops_blas.h`（L461，与 950PR 共用），不改公共头。

## 背景介绍
### 算子功能定位
单精度实数向量-向量加法：`y[j] = α × x[k] + y[j]`，k = 1+(i−1)×incx，j = 1+(i−1)×incy，i = 1..n；y 原地覆写。补齐 Blas 仓 Atlas A2/A3 系列 BLAS-1 能力缺口（arch22 目前仅有复数版 caxpy，实数版 saxpy 缺位）。

### 业界对标
- cublasSaxpy（§2.5.4）：参数语义、返回码与快速返回口径**唯一依据**。
- Netlib `saxpy.f`：n≤0 直接 RETURN（合法 no-op，不访问 α/x/y）；**α=0.0 时 `IF (DA.EQ.ZERO) RETURN` 直接返回，y 不变且不读取 x**——本实现对齐该语义（与 Sgemv 批次"α=0 不短路"口径相反，逐任务书执行）。
- 行为对齐锚点：仓内 `blas/axpy/arch35/saxpy_host.cpp` 的 ValidateSaxpyParams 与 aclblasSaxpy 入口（L20-160，现场核验）。

### 工程形态
- Kernel 直调（与仓内 blas/axpy/ 既有 arch22/arch35 实现同一工程模式：host 入口 + tiling + device kernel）。
- 代码落位：`blas/axpy/arch22/`（并入既有目录，新增 saxpy_* 文件，不动 caxpy 复数实现）。
- 仓内近亲：`blas/axpy/arch35/saxpy_host.cpp`（实数版 host 全套：校验 + contiguous/strided 双 tiling 路径 + α 并入 tiling 下发）、`blas/axpy/arch22/caxpy_*`（arch22 复数版：AIV 分核 + UB 双缓冲流水骨架，实数版同构去复数机制）。
- 测试工程 `test/saxpy/arch22/` 新建，并在 `test/saxpy/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} saxpy_test)` 注册（任务书 §2.2 明确要求：当前仅 arch35 注册，A2/A3 构建会退回非 GTest runner）。

### 接口规格
```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy);
```
（声明见公共头 L461，与 950PR 共用声明，禁止另立私有平行接口）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 向量元素个数 | int | attr | 任意 int32；n≤0 合法 no-op（不访问 α/x/y，直接 SUCCESS，不报错） | - |
| alpha | 标量乘数 α 指针（Host） | const float* | FLOAT32 | 任意 FLOAT32 值（含 0.0/1.0/Inf/NaN，值不校验）；n>0 时 nullptr → INVALID_VALUE；Host 侧同步解引用后并入 tiling 下发 | - |
| x | 输入向量（Device，只读） | float* | FLOAT32 | 一维，物理长度 1+(n−1)×\|incx\|；全值域含 Inf/NaN（按 IEEE 自然传播）；n>0 时 nullptr → INVALID_VALUE | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | 非零且 ≠INT32_MIN，正负均支持（负步长自物理 span 尾端起算）；n>0 时为 0/INT32_MIN → INVALID_VALUE；(n−1)×\|incx\| 超 int32 → INVALID_VALUE（host 侧 int64，issue #780/#866） | - |
| y | 输入兼输出（原地覆写，Device） | float* | FLOAT32 | 物理长度 1+(n−1)×\|incy\|；n>0 时 nullptr → INVALID_VALUE；跨度守卫同 x | [1+(n−1)·\|incy\|] |
| incy | y 步长 | int | attr | 同 incx | - |

### 数学公式
y[j] = α × x[k] + y[j]，k = 1+(i−1)×incx，j = 1+(i−1)×incy，i = 1..n；
n≤0 时 y 不修改（no-op）；α=0.0 时 y 不变且不读取 x（Netlib 语义）；y 原地覆写。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调在 Atlas A2/A3（arch22/ascend910b）实现 aclblasSaxpy 全语义（正负步长、跨度守卫、α=0 短路、n≤0 no-op），精度满足 mixed_tolerance FLOAT32 口径全绿，性能单 case 达 0.8×A100 cublasSaxpy。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | y += α·x 逐元素；arch35 saxpy（实数版全套：校验 + 双 tiling 路径 + α 入 tiling）平移 arch22，arch22 caxpy（AIV 分核 + UB 双缓冲）提供 device 侧工程先例——实数版机制更简，无新机制 |
| 2 | 接口原型 | 见上节；声明已在公共头 L461，不改头文件 |
| 3 | 参数规格 | 见参数规格表；校验分支逐条对齐 arch35 saxpy_host.cpp ValidateSaxpyParams（α/x/y 判空 → incx/incy 非零）；n≤0 no-op 先于一切指针访问（任务书 §2.5） |
| 4 | 支持数据类型 | 输入/输出全 FLOAT32（单 dtype 单接口） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验仅需覆盖 910B3（§3.1） |
| 6 | 精度要求 | mixed_tolerance FLOAT32：rtol=atol=2⁻¹³(1.2e-4)，ratio≥0.99，max_abs≤1e-2 或 32ULP；golden=cblas_saxpy（fp64 真值收窄 FLOAT32）；Inf/NaN 等效规则（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3）；内存 case 不涉及（§3.4） |
| 8 | 代码架构选型+可行性 | host（校验+tiling）+ SIMD 逐元素 axpy 内核（contiguous 快路 + strided 慢路）；已排除死路：SIMT（simt_api 仅 arch35）、复用 caxpy 原样（复数交织机制是负担而非助力，删繁就简） |
| 9 | 评测来源 | 精度 golden=cblas_saxpy（与仓内 saxpy_golden.h 一致）；性能标杆=gpu_baseline.csv（cublasSaxpy 同规模同 dtype 同步长） |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景；逐元素运算，无归约、无共轭；α 为 host 标量同步读回后经 tiling 下发（kernel 不做二次探测）。

### 支持数据类型
FLOAT32（α/x/y 全单精度实数）。

### 支持形状
n 任意 int32（n≤0 no-op）；n 覆盖 {0,1,2,3,1024,16384,262144,1048576}；incx/incy ∈ {±1,±2,±7} 及非零任意值；负步长自物理 span 尾端起算；|inc|>1 间隔访问（间隔元素跳过）；溢出组合（n=3 且 incx=INT_MAX、n=2³⁰ 且 inc=8）为负向校验用例。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| saxpy_contiguous | incx==1 且 incy==1 | 主性能路径：x 段/y 段整段连续 MTE2 搬 UB → V 乘加（Muls+Add 或 vmla 融合）→ MTE3 写回 y；双缓冲流水；每核块长按对齐单元整除（arch35 CalSaxpyTilingDataContiguous 同式，tileSize 上限 32768 元素） |
| saxpy_strided | incx≠1 或 incy≠1 | 慢路：按 incx/incy 步进寻址分段搬运（负步长按 span 尾端反向），标量/小段 MTE2 → V 乘加 → 写回；分支隔离不拖累主路 |

数据流：GM(x,y) → MTE2 → UB（x 段/y 段 ping-pong 双缓冲，事件 ID 交替）→ V（y += α×x，α 由 tiling 寄存器提供）→ MTE3 → GM(y 原地)。

#### host 侧设计
- 分核策略：numBlocks=min(GetAivCoreCount(), 上限)；n 均分，contiguous 路径按对齐单元取整 + 余块归尾核（perCoreN/remainder，arch35 同式）。
- 数据分块与内存优化：tileSize = UB 容量/(2×4B) 对齐整除、上限 32768（arch35 常量沿袭）；α 经 host 读回后写入 tiling（float 标量字段），消除 kernel 侧 host 指针依赖。
- tilingkey 规划：无独立 tilingkey，contiguous/strided 由 TilingData 字段（incx/incy）分发；字段：totalN/incx/incy/perCoreN/remainder/tileSize/alpha（arch35 SaxpyTilingData 同构）。
- 数据检测（校验顺序，逐条对齐 §2.5 与 arch35 ValidateSaxpyParams L20-160）：handle 判空（HANDLE_IS_NULLPTR）→ n≤0 no-op（SUCCESS，不访问 α/x/y，任务书 §2.5 明确 no-op 不报错）→ α/x/y 判空（INVALID_VALUE）→ incx==0/incy==0（INVALID_VALUE）→ INT32_MIN 步长（INVALID_VALUE）→ 跨度溢出 (n−1)×\|inc\|>INT32_MAX（host 侧 int64 计算，issue #780/#866）。α=0.0 短路在 host 读回 α 值后判定：直接 SUCCESS 返回，不启动 kernel（Netlib 语义）。异步下发。

#### kernel 侧设计
```
saxpy_contiguous:
  for 本核分块（tileSize 粒度）按序:
    [MTE2] x 段 + y 段 搬 UB（双缓冲）
    [V] y_buf = alpha × x_buf + y_buf（Muls + Add / vmla 融合）
    [MTE3] y_buf 写回 y（原地）
saxpy_strided:
  for 元素 i（按核交错）:
    [MTE2] x[1+(i-1)·incx] / y[1+(i-1)·incy] 小段搬运（负步长反向寻址）
    [V] 乘加 → 写回
```
无 workspace（无归约）；无原子操作；α=0 或 n≤0 不进 kernel（host 短路）。

#### 性能优化要点（终态，S5 回填实测）
- contiguous 主路 UB 双缓冲 + vmla 融合，大 n 多核满载逼近带宽；tileSize 联调（沿 arch35 32768 上限）。
- 候选：Muls+Add 与 vmla 融合对比、尾核余块处理策略（对齐 caxpy arch22 经验）、numBlocks 上限放开实验（Crot P30 先例）。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（已有 saxpy arch35 实现） |

## 算子约束限制
- n≤0 合法 no-op（不访问 α/x/y，§2.5）；α=0.0 短路（y 不变不读 x）。
- 步长语义即非连续访问；无广播；y 原地覆写不返回视图；确定性不要求（逐元素，同输入跨次执行 bit 级一致）。
- 不支持超出 incx/incy 语义的非连续访问；dynamic shape 不要求。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | rtol=atol=2⁻¹³，ratio≥0.99，max_abs≤1e-2 或 32ULP（golden 动态）；golden=cblas_saxpy fp64 真值收窄 FLOAT32；Inf/NaN 等效规则（NaN≡NaN、同号 Inf≡、NaN↔±Inf 等效） | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv；内存 case 不涉及 | 任务书 §3.3/§3.4 |

## 测试概览
- 用例分层（§3.5）：边界域（n=0/1、全零向量、alpha=0.0/1.0）、负向域（handle/α/x/y 空指针、incx/incy=0、INT32_MIN 步长、跨度溢出）、精度全量 1000 条（均匀 [-5,5] 50% + 正态 GAUSS_NORM_5_2 50%）、性能子集 200 条（单 dtype FLOAT32）。
- n 覆盖 {0,1,2,3,1024,16384,262144,1048576}；步长覆盖 {1,2,7,−1,−3}；α 覆盖 {0.0,1.0,2.5,−1.5,0.1,10.0}。
- 期望码：空指针/零步长/INT32_MIN/跨度溢出 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；n≤0 与 α=0.0 → SUCCESS。
- 负向用例缓冲区：不按 (n−1)×|inc| 预分配（dummy/NULLPTR 占位，§3.5 规则）；no-op 用例不分配。
- 性能口径：msprof op，warmup5+≥10 均值；复现步骤见 test_cases/README.md。

## 兼容性分析
- 不改公共头（声明已在 L461）；`blas/axpy/arch22/` 并入既有目录新增 saxpy_* 文件（不动 caxpy）；测试 CMakeLists 在 arch22 分支补 GTest 注册（任务书明确要求，波及面仅测试构建，不影响 arch35 既有注册）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
