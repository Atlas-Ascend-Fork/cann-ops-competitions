# vision_pos_embed_interp 算子 Ascend 适配设计文档

> 任务编号：10-29-vision_pos_embed_interp（2026 年 10 月社区任务）
> 提交团队：zhangfeng1133
> 目标平台：Ascend A2 / A3 / 950（triton-ascend ≥ 3.6.0，torch-npu ≥ 2.10.0，CANN ≥ 9.2.0）

# 需求背景（required）

## 需求来源

上游 [vllm-project/vllm](https://github.com/vllm-project/vllm)（main@aa31fa0）在 Qwen3-VL 视觉编码器中提供 Triton 融合算子 `triton_pos_embed_interpolate`（`vllm/model_executor/models/qwen3_vl.py:186-298`），将 G×G（默认 48×48）可学习位置表双线性插值到 (h,w) 特征网格，4 邻域 gather 加权、spatial-merge(2×2) 重排编进地址算术、时间维 t 以重计算代替物化，单次 launch 完成。

当前 NPU 上的问题：

1. vllm-ascend 在 `vllm_ascend/patch/worker/patch_qwen3vl.py:72-89` 以 monkey-patch 强制 Qwen3-VL 的 `fast_pos_embed_interpolate` 走 native 路径（Triton kernel 在 NPU 不可用，另见 `patch_idex_310.py:38` 注释）；
2. 上游测试 `tests/kernels/core/test_vit_bilinear_pos_embed.py` 为 CUDA-only，NPU 无实机用例；
3. 同款 kernel 拷贝存在于 `qwen3_omni_moe_thinker.py:891` 附近；
4. `fast_pos_embed_interpolate`（qwen3_vl.py:744-763）对 grid_thw 列表逐图 launch 后 `torch.cat`，多图场景产生 N 次 launch + 一次全量拷贝。

## 背景介绍

### 算子功能

`vision_pos_embed_interp`（上游名 `triton_pos_embed_interpolate`）：对 G×G 平铺的可学习位置表 `embed_weight [G*G, hidden_dim]` 按特征网格 (h,w) 做双线性插值，输出 spatial-merge 顺序的 `[t*h*w, hidden_dim]` 位置编码，时间维逐帧重复同一空间模式。

计算语义（与 native 标杆 `pos_embed_interpolate_native` 一致）：

```
h_idx(i) = i * (G-1)/(h-1)   （h>1；h=1 时 h_scale=0）
w_idx(j) = j * (G-1)/(w-1)   （w>1；w=1 时 w_scale=0）

hf=floor(h_idx(i)), wf=floor(w_idx(j))
hc=min(hf+1, G-1),  wc=min(wf+1, G-1)
dh=h_idx(i)-hf, dw=w_idx(j)-wf
w11=dh*dw; w10=dh-w11; w01=dw-w11; w00=1-dh-w01

out(i,j,:) = w00*E[hf,wf,:] + w01*E[hf,wc,:] + w10*E[hc,wf,:] + w11*E[hc,wc,:]
```

其中权重与邻域值均在**输出 dtype** 下做乘累加（上游 :236-251，禁止改为 FP32 累加）。spatial-merge 重排将 (h,m_size)×(w,m_size) 网格重排为 (h/m_size, w/m_size, m_size, m_size) 顺序，即输出行序为「merge 块 → 块内行优先」，与 native 的 permute(0,2,1,3,4) 语义一致。时间维 t>1 时输出为 t=1 结果的逐位 repeat（atol=0）。

### 接口参数（与上游公共 API 一致，签名不得变更）

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 维度 |
| --- | --- | --- | --- | --- |
| embed_weight | 输入 | G×G 可学习位置表（平铺） | FP32/BF16 | [G*G, hidden_dim] |
| t | 属性 | 时间维帧数 | int | - |
| h, w | 属性 | 特征网格高宽（须被 m_size 整除） | int | - |
| num_grid_per_side | 属性 | 位置表边长 G（Qwen3-VL=48） | int | - |
| m_size | 属性 | spatial-merge 边长（Qwen3-VL=2） | int | - |
| dtype | 属性 | 输出 dtype | FP32/BF16 | - |
| out | 输出 | 插值+merge 重排后的位置编码 | 与 dtype 一致 | [t*h*w, hidden_dim] |

### NPU 现状分析

| 路径 | kernel 数/次 | 问题 |
| --- | --- | --- |
| native 逐算子插值（NPU 现役，monkey-patch 强制） | 38~39 | 约 20 次 dispatch、5 次全量访存 |
| 多图 native + cat | 116（3 图） | 逐图 launch + 全量拷贝 |
| 上游 Triton kernel（被测实现） | 1 | CUDA-only，需 Ascend 亲和适配 |

# 需求分析（required）

## 需求描述

基于 triton-ascend 3.6.0 完成 `_bilinear_pos_embed_kernel` 的昇腾亲和支持，保证算子功能与上游实现一致，并：

1. kernel 本体 NPU 适配（`tl.math.floor` 等内建函数的昇腾亲和替换，如有必要）；
2. 摘除 vllm-ascend 的 monkey-patch，按平台能力分派 Triton/native；
3. 多图（grid_thw 列表）场景将逐图 launch + `torch.cat` 合并为单 kernel；
4. 提交至 mojo_opset（kernel 置于 `kernels/npu_a2_triton/` 与 `kernels/npu_a5_triton/`，接口置于 `functions/`，封装置于 `modules/`，精度/性能套件置于 `tests/`）。

## 需求拆解

1. **功能一致性**：输出为 spatial-merge 顺序的 `[t*h*w, hidden_dim]`；权重以输出 dtype 乘累加；`h_scale/w_scale` 由 wrapper 预计算传入；h=1 或 w=1 时 scale=0 边界保持；h、w 必须被 m_size 整除的断言保持。
2. **泛化能力**：G、m_size、hidden_dim 不针对 Qwen3-VL 单一配置硬编码（兼容 qwen3-omni thinker 同款调用）；G=14 非 2 的幂须正确。
3. **多图合并**：合并单 launch 版本输出与逐图版逐位一致。
4. **精度**：随附精度套件全量通过（0 failed / 0 error），标杆为 `pos_embed_interpolate_native`，不允许修改断言与容差。
5. **稳定性**：不出现 aicore / vector core 超时（507014 / 507034）及编译器崩溃；视频大图 (16,100,168) 不超时。
6. **性能**：MBU ≥ 60%（常规单图/多图），尾块 ≥ 50%，(1,8,16) 端到端不劣于 native。

# 详细设计（required）

## 算子分析

### 数学公式

见「需求背景-算子功能」。输出按 spatial-merge 顺序展开：`out_row(pid)`，`pid ∈ [0, t*h*w)`：

```
total_spatial = h * w
spatial_idx   = pid % total_spatial        # 时间维帧内空间索引（时间维广播=重计算）
block_idx     = spatial_idx // (m_size*m_size)
local_idx     = spatial_idx %  (m_size*m_size)
num_blocks_w  = w // m_size
br = block_idx // num_blocks_w;  bc = block_idx % num_blocks_w
lr = local_idx  // m_size;       lc = local_idx  % m_size
row = br*m_size + lr;  col = bc*m_size + lc
```

### 支持数据类型

FP32、BF16（权重计算精度与上游一致：权重 cast 到输出 dtype 后乘累加，不引入或消除 ULP 差异来"凑容差"）。

### 支持形状

- embed_weight: [G*G, hidden_dim]，G 任意 ≥1（主测 48，泛化抽测 14），hidden_dim 任意（主测 1152，尾块测 1000）；
- h、w 任意正整数且被 m_size 整除（含 h=m_size 或 w=m_size 退化）；
- t ≥ 1；多图为 grid_thw 列表（元素数任意 ≥1）。

## 算子实现

### 实现方案

#### 1. host 侧设计

**（a）单图 wrapper（接口与上游一致）**

```
assert h % m_size == 0 and w % m_size == 0
h_scale = float(G-1)/float(h-1) if h > 1 else 0.0
w_scale = float(G-1)/float(w-1) if w > 1 else 0.0
BLOCK_D = triton.next_power_of_2(hidden_dim)
grid = (t*h*w,)
```

- `h_scale/w_scale` 在 host 以 float32 预计算传入（上游 :281-282 语义保持），避免 kernel 内除法与 torch.linspace 的步进语义差异；
- `h=1 或 w=1 → scale=0` 边界保持（等效单行/单列 gather，NUM_GRID-1 clamp 生效防越界）。

**（b）多图合并单 launch**

对 grid_thw 列表 `[(t_k,h_k,w_k)]`，host 侧预计算并构造设备张量：

- `cu_rows[N+1]`（int32）：各图输出行数 `t_k*h_k*w_k` 的前缀和，`cu_rows[0]=0`；
- `img_hw[N]`（int32×2）：每图 (h_k, w_k)；
- `img_scale[N]`（float32×2）：每图 (h_scale_k, w_scale_k)。

kernel grid 取 `(cu_rows[N],)`，每个 program 用行号 `pid` 在 `cu_rows` 上做**顺序线性定位**得到图号 k 与图内行偏移 `local_row = pid - cu_rows[k]`（图数 N 通常 ≤ 8，线性扫描仅 N 次标量 load，开销可忽略；若 N 较大可切换二分，接口不变）。图内 `spatial_idx = local_row % (h_k*w_k)`，后续地址算术与单图版完全一致，仅输出偏移改为 `pid*HIDDEN_DIM`（天然承担行偏移），因此**合并版与逐图版逐位一致**，无需 `torch.cat`。

wrapper 返回 `torch.empty(sum(t_k*h_k*w_k), hidden_dim)`。

**（c）vllm-ascend 分派改造（摘除 monkey-patch）**

替换 `patch_qwen3vl.py:72-89` 的 native 强制 patch 为能力分派：

```
if HAS_TRITON and _triton_ascend_available():   # 平台能力检测
    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate_triton_merged
else:
    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate_native  # 兜底
```

- `_triton_ascend_available()`：检测 `triton_ascend` 可导入且 NPU 设备可用（懒检测 + lru_cache，避免每 step 重复检测）；
- Triton 路径直接使用本算子的**多图合并版**（对上游逐图版也是增强）；
- 接入前后对比验证：native 路径输出 vs 分派后 Triton 路径输出，按精度套件容差对拍 + 性能对拍（native 38~39 kernel vs 单 launch）。

#### 2. kernel 侧设计

**（a）核心 kernel（昇腾亲和适配点）**

沿用上游整体结构（pid→merge 地址算术→双线性权重→BLOCK_D 循环 gather/store），针对 triton-ascend 做以下亲和处理：

1. **`tl.math.floor` 昇腾亲和替换**：`h_frac = row*h_scale ≥ 0` 恒成立（row ≥ 0 且 scale ≥ 0），floor 与 int 截断在非负域等价，故 `hf = h_frac.to(tl.int32)`（trunc）替代 `tl.math.floor(...).to(tl.int32)`，消除 vector core 上 floor 数学库调用；负数域不等价但输入域保证非负（设计约束见「算子约束限制」）。
2. **块内循环 `tl.range(0, HIDDEN_DIM, BLOCK_D)`**：保持特征维分块循环；BLOCK_D = next_power_of_2(hidden_dim)，`mask = cols < HIDDEN_DIM` 保证尾块（hidden=1000）最后一段不满时越界列不写入、不越界读。
3. **4 邻域 gather**：`tl.load(embed_ptr + offXY + cols, mask=mask)`，offXY 为标量行基址 + 向量列偏移的连续访存（每邻域行内连续 BLOCK_D），符合 NPU vector 单元连续 load 模式；邻域间为 4 路独立行基址，不强制 reshuffle。
4. **权重精度路径**：`w00_c..w11_c = w.to(out_dtype)` 在 BLOCK_D 循环外计算一次，循环内乘累加，与上游 :236-251 舍入路径一致（FP32/BF16 均不变）。
5. **时间维广播**：`spatial_idx = pid % total_spatial` 重计算代替物化，t>1 与 t=1 repeat 逐位一致（同一算式、同一输入）。
6. **dtype 处理**：BF16 输入按上游语义直接以 out_dtype 乘累加，不做额外 FP32 转换。

**（b）多图 kernel**

在 (a) 基础上前置「图定位」小节（线性扫描 cu_rows），其余逐位一致。constexpr 参数（NUM_GRID/M_SIZE/HIDDEN_DIM/BLOCK_D）取多图公共值（同一模型内 G、m_size、hidden_dim 相同），逐图变化的 (h,w,scale) 走 device 张量。

#### 3. tiling/分块策略

- **并行度**：以输出行为粒度，grid = 总输出行数（单图 t*h*w / 多图 Σt*h*w），NPU 多核天然按 pid 均分，无核间不均分问题（每 program 计算量恒定：4×BLOCK_D load + 加权求和 + BLOCK_D store）；
- **UB 用量**：每 program 常驻 4×BLOCK_D 邻域向量 + 输出向量，BLOCK_D 最大 2048（hidden≤2048 场景），数据量小，无需 double buffer；hidden > BLOCK_D 上限或超大 hidden 时由 tl.range 循环分块；
- **tilingkey 规划**：dtype（FP32/BF16）与 BLOCK_D 为 constexpr 特化，无需额外 tilingkey。

#### 4. mojo_opset 交付布局

```
mojo_opset/
├── kernels/npu_a2_triton/vision_pos_embed_interp/   # kernel + wrapper（A2）
├── kernels/npu_a5_triton/vision_pos_embed_interp/   # 同源登记（A5/950）
├── functions/vision_pos_embed_interp.py             # 函数接口（单图 + 多图合并）
├── modules/vision_pos_embed/                        # 带参数封装（fast_pos_embed_interpolate 接入形态）
├── config/npu_a2.yaml, npu_a5.yaml                  # OPS 登记
└── tests/functions|modules/..., tests/performance/...  # 精度/性能套件
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B 系列） | √ |
| Atlas A3 | √ |
| Ascend 950 | √ |

## 算子约束限制

1. 输入/输出 shape、布局、dtype、参数含义与上游 `triton_pos_embed_interpolate` 一致，公共 API 签名不得变更；
2. spatial-merge 重排顺序、时间维语义与 native 一致；t>1 必须与 t=1 的 repeat 逐位一致（atol=0）；
3. h、w 被 m_size 整除断言保持；G、m_size、hidden_dim 不得硬编码为 Qwen3-VL 配置；
4. 权重以输出 dtype 乘累加，不得改 FP32 累加；
5. kernel 依赖 `h_frac/w_frac ≥ 0`（floor→trunc 等价前提），由 h_scale/w_scale ≥ 0 与 row/col ≥ 0 保证；
6. 多图合并要求各图 G/m_size/hidden_dim 一致（同一模型实例约束，多图定位只扫行前缀和）；
7. 不出现 NPU AI Core / Vector Core 超时（507014/507034）及编译器崩溃。

# 测试方案（可维可测分析）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 随附精度套件全量通过；标杆 = `pos_embed_interpolate_native`（torch.linspace 采样 + 4 邻域 gather 加权 + permute 重排）；容差 FP32 atol=5e-5/rtol=1e-5，BF16 atol=1e-2/rtol=1e-2；时间维 atol=0/rtol=0；多图合并与逐图逐位一致；不得修改断言与容差 | 任务书精度套件 |
| 性能 | MBU = 总读写字节 ÷ NPU 总耗时 ÷ fill_ 标定带宽；常规配置 ≥60%，尾块配置 ≥50%，(1,8,16) 端到端不劣于 native；读写字节按 `2×(G²×hidden + t·h·w×hidden)` 固定算式（多图按图累加） | 任务书性能套件（bench_vit_pos_embed.py 口径） |

## 测试设计（配套测试用例文档）

配套测试用例按 cannbot 测试设计方法论（黑盒因子矩阵 / 白盒路径覆盖 / 干扰与鲁棒 / 单元断言 / 精度对拍）分五类设计：

1. **黑盒功能用例（40 条普通）**：以 dtype×grid×G×hidden×t 为因子矩阵，覆盖任务书精度套件全部 shape 档位；
2. **边缘用例（20 条）**：h=m_size/w=m_size 退化、G=14 非 2 幂、hidden=1000 尾块、t 大值、非方形、多图混合、(16,100,168) 大网格等；
3. **精度用例**：Triton vs native 对拍 + 时间维位精确 + 多图逐位一致，容差按 dtype；
4. **干扰/鲁棒用例**：非法 h/w（不被 m_size 整除）断言、超界 G、全零权重、极端 scale 等输入扰动；
5. **单元用例**：host 侧 scale 计算、BLOCK_D/mask 计算、cu_rows 前缀和构造、多图定位映射的正确性断言（可 CPU 侧先行验证）。

用例明细见提交仓 `tests/` 下测试用例文档与 pytest 实现。

## 兼容性分析

- 上游公共 API 签名不变，qwen3-omni thinker 同款调用自动受益；
- vllm-ascend 分派改造保留 native 兜底，旧环境（无 triton-ascend）行为不变；
- `+ patch` 加法（qwen3_vl.py:881-882）不在本任务范围。
