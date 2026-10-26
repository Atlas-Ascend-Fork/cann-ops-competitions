# 需求背景（required）

## 需求来源

来源：9月社区任务「kv_cache_turbo_quant算子开发」任务书及随任务提供的 `op.json`、`case.json`、`golden.py`。目标仓库：ops-transformer，算子目录 `experimental/attention/kv_cache_turbo_quant`。

## 背景介绍

### KvCacheTurboQuant 算子实现

当前 vllm-ascend 中 KV cache 压缩方案有 INT8 静态量化（per-channel scale+offset，压缩率仅 2x）与 LSH 哈希（KVComp，仅用于稀疏选择不压缩存储）。参考 TurboQuant 论文，本算子面向标准 MHA/GQA 的 KV cache 提供在线向量量化压缩 Encode：对每个 K/V 向量做「MSE 主量化 + QJL 残差修正」两阶段编码。

首版采用统一 3-bit MSE 主编码（支持 2/3/4 bit）+ 1-bit QJL 残差编码。head_dim=128 时物理存储 4.25 bit/channel（68 字节/head，含范数与残差范数标量），理论显存压缩比约 3.76x（vs BF16）。论文中的 3.5 bit/channel 混合模式（outlier/non-outlier 分组）不在首版范围。本算子不覆盖 MLA（DeepSeek MLA 缓存低秩 latent + RoPE key，接口与融合路径需单独设计）。

### 现状分析

ops-transformer 已有 TurboQuant 生态算子：ops-nn `TurboQuantCompressLatent`（MLA 512 维 latent 的 4-bit 压缩）与 attention 下游 `TurboQuantSparseAttnSharedkv`（258B compact slot 解码）。本算子面向 MHA/GQA 的 128 维普通 KV head，编码协议（2/3/4-bit 主码 + QJL + 双范数标量）与上述 MLA 变体不同，是补齐生态的独立 Encode 算子。题包 golden 为 PyTorch eager 实现（固定 Lloyd-Max 型质心表、正交旋转、残差高斯投影符号量化、LSB-first 比特流打包），本设计以与其四输出逐位一致为精度目标。

# 需求分析（required）

## 需求描述

| 项目 | 契约 |
| --- | --- |
| 输入 kv_vectors | BF16，ND 连续，[T, H, 128]，T≥0、4≤H≤32（MHA/GQA） |
| 输入 rotation_matrix / qjl_matrix | FP32，ND 连续，均 [128,128]；H 为 Hadamard 或预生成随机正交阵，S 为固定高斯随机投影阵（推理期间不变） |
| 属性 | mse_bits ∈ {2,3,4}，默认 3 |
| 输出 quant_idx | UINT8 [T, H, 16·mse_bits]，主量化编码 bit-packed（连续比特流，LSB-first 小端） |
| 输出 quant_qjl | UINT8 [T, H, 16]，残差 QJL 符号编码 bit-packed |
| 输出 quant_norm / quant_gamma | BF16（或 FP16）[T, H]，向量范数 ‖x‖₂ 与原尺度残差范数 ‖x‖₂·‖r‖₂ |
| 接口 | aclnnKvCacheTurboQuantGetWorkspaceSize / aclnnKvCacheTurboQuant 两段式 |
| 硬件 | Atlas 800T A2（ascend910b），CANN 9.1.0+ |

## 需求拆解

1. 与 golden 逐位对齐：矩阵方向（B 侧不转置）、码本常量、fp32 中点边界、严格大于索引、QJL `>=0` 记 1、Div 归一化（非乘倒数）、`clamp_min(1e-30)`、零向量/零残差语义、LSB-first 连续比特流打包、范数 fp32 单次舍入。
2. 工程：OpDef 注册（ascend910b）、推形、tiling（分核 + Matmul tiling）、kernel（MIX_AIC_1_2）、aclnn 两段式接入。
3. 精度验证：题包 5 用例 + 零向量/零残差/边界两侧/尾 tile/空输入扩展用例，四输出逐位一致。
4. 性能验证：msprof Task Duration 口径对比任务书 baseline ≥10×；另附同机 eager event 口径对照。
5. 内存：workspace 仅含 matmul 库工作区与每核中间量槽位，无动态分配。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 (token, head) 行向量 x∈R^128：

```text
n      = ‖float32(x)‖₂                      # bf16→fp32 精确升位后求范数
u      = x / max(n, 1e-30)                  # Div（非乘倒数）；n=0 时 u=0
y      = u @ H                              # 等价 golden 的 unit @ rotation.T（B 侧不转置）
idx[d] = Σ_k ( y[d] > bnd_k )               # bnd_k = (c_k + c_{k+1}) × 0.5（fp32），严格大于
r      = y - c[idx]                         # 单位向量量化残差（旋转空间）
rho    = ‖r‖₂
gamma  = n × rho                            # fp32 标量乘，一次舍入到 BF16
v      = r / max(rho, 1e-30)                # rho=0 时 v=0
p      = v @ S                              # 高斯投影
qjl[d] = ( p[d] >= 0 ) ? 1 : 0              # 含 +0.0
norm   = n（一次舍入到 BF16/FP16）
```

打包协议（与 golden `_pack_bits` 一致）：连续比特流，值 d 的 bit0 位于字节 ⌊d·b/8⌋ 的位 (d·b) mod 8，小端。即 2-bit 每 4 维 1 字节；3-bit 每 8 维 3 字节（跨字节连续）；4-bit 每维 1 nibble（低 nibble=偶数维）；1-bit 每 8 符号 1 字节。

### 支持数据类型

| 参数 | dtype | 计算/存储 |
| --- | --- | --- |
| kv_vectors | BF16 | 读入升 fp32 计算 |
| rotation/qjl_matrix | FP32 | 两次 fp32 Matmul |
| quant_idx / quant_qjl | UINT8 | 低位优先 bit-packed |
| quant_norm / quant_gamma | BF16（FP16 预留） | fp32 计算后单次舍入 |

### 支持形状

T·H ≤ INT32_MAX；T=0 为合法空操作（不 launch 计算）。矩阵维度固定 128。

## 算子实现

### 实现方案

单次 MIX_AIC_1_2 launch：AIC 执行两次 fp32 Matmul；两个 AIV 按「职能所有者 + 相位移动」流水化分工——AIV1 负责 phase-0（kv→norm 树→u），AIV0 负责 phase-1（y→idx/r/rnorm/gamma→v）与打包写出；中间量（u/y/v/p/norm32，各 [32,128] fp32）经 GM workspace 每 tile 5 槽交换（地址全不复用），CrossCore flag 按 C1→V1→C2→V2 每 tile 握手一次，三方屏障（AIC↔AIV0↔AIV1）对称参与。相邻 tile 的 phase0(t+1) 计算与 phase1(t)/pack(t) 及 GEMM 并行，消除逐 tile 串行等待。

#### 3.2.1 host 侧设计

##### 1. 分核策略

N = T·H 为总行数，tile 行数 B=32（MaMaer 实测验证的平衡点：cube 单瓦 M=32 利用率与 AIV 向量工作量的折中）。tile 数 = ceil(N/32)，核 c 依次处理 tileId = c, c+coreNum, c+2·coreNum…（跨核步进）。coreNum = min(AIC 核数, tile 数)；N=0 时 coreNum=1 空转。尾 tile 在片上清零补齐，仅写回有效行。

##### 2. 数据分块和内存优化策略

- Matmul：M=32, N=128, K=128 fp32 单核（MatmulApiTiling 固定形状，A/B/C 均 GM）；TilingKey=mse_bits（2/3/4），模板期选定码本与打包路径。
- AIV UB 预算：AIV0 ≈112KB（y/p/idx32/workA/workB/uv 各 16KB + 相位 4KB + half 8KB + packU8 2.5KB + zip 2KB + 范数槽 2KB + normIn/广播/掩码 ≈1.3KB）；AIV1 ≈79KB（inQ 8KB + x32/workA/workB/uv 各 16KB + 相位 4KB + 范数槽 2KB + 广播 ≈1KB）。
- workspace = [matmul 库工作区（GetLibApiWorkSpaceSize，32B 对齐）] + [每 tile 5 槽 u/v/y/p/norm32（5×16KB×tile 数），地址全不复用——根除 Cube L1 旧数据命中]。
- 打包（全部经本地 numpy 仿真与 golden._pack_bits 逐位互验，含 2^24-1 极值与常值扫描）：
  - 4-bit：idx≥8 先减 16（两位补码位型不变，防 int4b_t Cast 饱和）→ fp32→half→int4b_t 硬件打包（低 nibble=偶数维）；
  - 2-bit：GatherMask 内置 pattern 3..6（Normal 模式，src0RepeatStride=8）提取 4 相位 dst_p[k]=idx[4k+p]，加权折叠 1/4/16/64 后两级 Cast（fp32→half→uint8，整数值精确）；
  - 3-bit：GatherMask 自定义掩码（M(p)=Σ2^{p+8k}，双 u32 字/repeat）提取 8 相位 dst_p[k]=idx[8k+p] → word24=Σ2^{3p}·phase（整数 <2^24，fp32 精确）→ 精确 digit-split（×2^-8 幂 + Cast FLOOR + mul-sub）得 b0/b1/b2 → zip=b0+256b1+65536b2 → Cast TRUNC 至 int32 → 按 u8 重解释得 [b0,b1,b2,0]×组 → DataCopyPad{blockLen=3, srcStride=1} 跳过填充字节连续写出；
  - QJL：signs 同法 8 相位提取 → byte=Σ2^l·s_l → 两级 Cast；
  - 全程整数域 fp32 精确算术（权重均为 2 的幂，值域 <2^24），无舍入误差引入。
- 范数（归约树与 torch_npu 逐位一致，M9 定案）：**归约树 T = halve-L1（w[j]=v[j]+v[j+64]）+ 64 宽相邻对树**——内核实现为每行 `Add(半加,64) + ReduceSum(64)`（实测 aclnn sum = 向量半加 + 64 宽归约，该树在 208 组结构化探针、20k 随机行、64 组对抗分布行上逐位命中）→ 整体向量 Sqrt → 双 pattern3 GatherMask 提取每行 lane0 → Maxs(EPS) 得分母 → 逐行 Duplicate 广播 + Div（IEEE 除法与 torch 逐位一致）；gamma=norm·rnorm 向量乘。

##### 3. tilingkey 规划策略

| tiling key | 含义 |
| ---: | --- |
| 2 | 2-bit 主编码 + 1-bit QJL |
| 3 | 3-bit 主编码 + 1-bit QJL |
| 4 | 4-bit 主编码 + 1-bit QJL |

#### 3.2.2 kernel 侧设计（M63g 流水化分工）

1. 【AIV1·phase0(t)】搬入 bf16 行块（DataCopy）→ 升 fp32（尾行清零）→ norm 归约树（Mul+半加+ReduceSum×32+V_S）→ `u = Div(x, max(n,1e-30))`（零向量自然得 0）→ 结果留 UB；在 V2(t-1) 镜像释放后 MTE3 写 u→槽0、norm32→槽4，dsb release 后 **SetFlag C1（真实写跟随）**。
2. 【AIC】wait C1 → GEMM1 y = u @ H（A/B/C 全 GM fp32，B 侧不转置）→ dsb → SetFlag V1。
3. 【AIV0·phase1(t)】wait V1（acquire dsb）→ 读 y + norm32（槽4）→ 边界比较链（严格 `>`）→ 质心查表 → r = y - c → 残差范数（同款归约树）→ gamma = norm32×rnorm32（fp32）→ `v = Div(r, max(rho,1e-30))` → MTE3 写槽 v → SetFlag C2（真实写跟随）。
4. 【AIC】wait C2 → GEMM2 p = v @ S → SetFlag V2。
5. 【AIV0·pack(t)】wait V2（acquire）→ 读 p → 符号位（`>=0`）→ 四种位宽打包 → norm/gamma/idx/qjl DataCopyPad 写 GM。
6. **流水化**：AIV1 在 V1(t) 镜像释放后立即计算 phase0(t+1)（与 AIV0 的 phase1(t) 并行，AIV 间零互等），u(t+1) 的 GM 写推迟到 V2(t) 之后、C1(t+1) Set 之前；AIV0 的 C1(t+1) 镜像早到置于 pack(t) 前——GEMM1(t+1) 仅等待 phase0(t+1)，pack(t) 与 GEMM1/GEMM2(t+1) 完全重叠。

同步纪律：AIV/AIC 间用 CrossCoreSetFlag/WaitFlag（SYNC_MODE2，AIC↔AIV 同 AI Core），flagId 8..11（避开 Matmul 高阶 API 内部占用的 [0, 2N-1]）；SYNC_MODE2 语义为 AIC↔AIV0↔AIV1 三方屏障（OL-190），每旗三方对称参与、每 AIV 内旗序严格递增；真实 Set 跟随本子块同管道 GM 写（前置 dsb release），消费侧 Wait 后 acquire dsb。UB 复用边界（MTE2→V、MTE3→V、V→MTE3、V→S）显式事件同步。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 (ascend910b) | √ |

注册 ascend910b；实测以 CANN 9.1.0 真机为准。310P/A3 未验证。

## 算子约束限制

- 仅覆盖标准 MHA/GQA KV cache Encode；不含分页写回、attention 解码、MLA latent、3.5-bit 混合模式。
- 调用方保证 rotation_matrix 正交、qjl_matrix 高斯分布且推理期间固定；host 只校验形状/dtype 元数据。
- head_dim 固定 128；qjl_dim 固定 128（= head_dim）。
- 量化为确定性计算（无原子/无随机）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 四输出与题包 golden 逐位一致（idx/qjl uint8 exact；norm/gamma bf16 位型一致）；统计指标另报：重构相对 MSE（MSE/‖x‖²）与内积相对误差 p95 | 任务书 3.2 |
| 性能 | 相对功能等价 PyTorch eager ≥10×（任务书 baseline 口径），另附同机 eager/ACLNN event 对照与 msprof Task Duration | 任务书 3.3 |

统计口径说明（诚实标注）：2-bit 主编码的理论相对 MSE ≈ 0.117，高于任务书 0.05 门槛（独立于实现，golden 同样如此）；内积误差分母 |⟨q,x⟩| 在近零处天然发散。两项统计指标取决于算法与数据分布，编码对拍通过不代表统计达标——报告中将分别列示实测值并与验收方确认口径。

测试覆盖：题包 5 用例（AscendOpTest 数据生成/golden/比较全流程）+ 扩展 case（H=4/5/31/32、尾 tile、跨多 tile、空输入、零向量、零残差、边界两侧定向样本、bitwise 二次运行确定性）。每轮 kernel 改动后全量复验。

## 兼容性分析

新增目录接入 experimental/attention 构建体系，生成标准 aclnn 两段式接口；不改变既有算子 ABI。下游消费本编码时须使用相同矩阵、码本、位序与 gamma 定义（K/V 分别调用，H/S 推理期间固定）。

风险登记：
1. fp32 累加顺序与 torch GEMM 的差异可能导致码本边界翻转（实测翻转率量级 ~5e-7/坐标）——采用与 golden 同构的 fp32 Cube 路径并以逐位对拍验证，若翻转按 GEMM 块形/累加序校准。**已闭环：官方 5 用例四输出全 BITWISE PASS。**
2. norm/gamma 的 bf16 err_threshold=0 语义（位型精确）**已闭环**：定位 torch_npu 归约树（halve-L1 + 64 宽相邻对树）并在内核复现，mse2/mse4 曾有的 1/16384 gamma 刀口已消除（M9，208 组探针 + 20k 随机行 + 64 组对抗行仲裁实锤）。
3. 题包未提供官方 rotation/qjl 二进制：自测用 seed=20260923 固定种子合成（QR+符号规范化正交阵、iid 高斯阵），报告中明确标注与官方矩阵身份不同。
4. prefill_t2048 实测 543.7μs（任务书 10× 口径需 ≤385.5μs）：剩余瓶颈为 AIC 每 tile matmul 簿记标量（0.22μs/tile）与跨核屏障延迟；后续方向=屏障精简、簿记内联。
