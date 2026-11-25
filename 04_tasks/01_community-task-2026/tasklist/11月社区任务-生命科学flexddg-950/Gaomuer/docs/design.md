# 需求背景（required）

## 需求来源

本设计文档对应 2026 年 11 月 CANN 社区任务《Rosetta Flex ddG 端到端加速开发（Ascend 950PR）任务书》，面向生命科学蛋白质-蛋白质结合界面突变自由能变化（$\Delta\Delta G$）预测场景，基于 **24 核 CPU + 单卡昇腾 Ascend 950PR** 设计整模块端到端异构加速方案。

对标开源软件与协议基线：

- **Rosetta 主体**：GitHub 开源仓 `RosettaCommons/rosetta`（参考基线 `2026.29` / commit `80a0635615`）；
- **Flex ddG 官方协议**：GitHub 开源仓 `Kortemme-Lab/flex_ddG_tutorial`（随包只读协议原件 `test_cases/ddG-backrub.xml` 与基线脚本 `test_cases/baseline/`）；
- **运行软件栈**：CANN 9.0.0 及以上版本，采用 Host CPU（C++/Python 调度）+ NPU（Ascend C / CANN Kernel 驻留流水线）混合异构工程模式。

> 说明：本文档为开发期架构与算子设计方案，不代表已完成全部源码实现或真机验收；文中性能拆解与时间预算为基于基线数据与硬件特性的设计期分析估算，最终结论以 24 核 CPU + Ascend 950PR 实测与 `verify_bd623.py` 自动化验收输出为准。

## 背景介绍

### Rosetta Flex ddG 协议与计算特征

Rosetta Flex ddG（Barlow et al., *J. Phys. Chem. B* 2018）通过蒙特卡洛主链柔性采样（Backrub）与侧链重排/能量最小化系综平均，预测蛋白质界面单点突变引发的结合自由能变化 $\Delta\Delta G$。相较刚性骨架方法，Flex ddG 引入局部主链柔性松弛，对抗体-抗原界面中存在空间位阻或氢键网络重排的突变具有显著更高的预测精度，但纯 CPU 计算代价极高。

### BD-623 评测集结构与输入特征剖析

对随包评测输入 `test_cases/PDB/BD-623.pdb`、`test_cases/bd623_mutations.csv` 与 48 核参考基线 `test_cases/bd623_reference.csv` 进行统计剖析，核心结构与工作负载特征如下：

| 维度 | 统计特征 | 工程与数值设计含义 |
| --- | --- | --- |
| 复合物规模 | 3 条链共 **420 残基**（无 `HETATM`）：链 A（抗原，189 残基，333–526）、链 H（重链，124 残基，2–125）、链 L（轻链，107 残基，4–110）；PDB 重原子 3,217 个，全原子补氢后约 **6,307 原子** | 单构象全原子坐标仅约 75 KB（FP32）/ 151 KB（FP64）；全部 1,750 条轨迹坐标可全量常驻单卡 Ascend 950PR HBM |
| 链断裂与插入码 | 链 A 在 **516–522 之间存在 5 残基缺失链断裂**（编号 517–521 缺失）；BD-623 中无插入码，但通用接口须完整支持 `(chain, pdb_position, icode)` 定位 | `PrimarySequenceNeighborhood` 与 Backrub 支点段严禁跨越链断裂/FoldTree Jump（尤其影响邻近突变 `A:S514Q / S514Y`） |
| 二硫键拓扑 | 含 **6 对二硫键**（`SG–SG < 3.0 Å`）：`H:C22–C95`、`L:C22–C90`、`A:C336–C361`、`A:C379–C432`、`A:C391–C525` 及 **`A:C480–A:C488`** | `A:C480–A:C488` 直接位于界面核心突变环内，必须按 `CYD` 类型与 `dslf_fa13` 势函数精确建模，禁止破坏共价二硫桥 |
| CA 距离约束 | `max_distance=9.0 Å`、`coord_dev=0.5`、`CA_only=1` 在 `BD-623.pdb` 上生成 **2,787 对 CA–CA 谐波约束** | 约束对数量有限且拓扑固定，极适合紧凑静态索引表在 NPU 上向量化求值与求导 |
| 突变与邻域分布 | 50 个抗原侧单点突变分布于 **21 个唯一位点**（全在链 A）；`cb8`（8 Å）`bubble` 含 **5–14 残基**（中位 11.5），`neighbor_shell`（`bubble ± 1`）含 **9–29 残基**（中位 19.0） | 每次 Backrub/Sidechain MC 与 Pack 仅修改局部 9–29 残基（约 150–450 原子），高度契合片上 UB/L1 局部复用 |
| 效应与几何敏感样本 | 大效应组（$\lvert \overline{\Delta\Delta G}_{\text{ref}} \rvert > 1.0\ \text{REU}$）**15 例**，集中于 `G476 / G485 / F486 / N487`（峰值 `A:F486P = +8.892 ± 0.843 REU`，最大方差 `A:G476R = +4.336 ± 4.476 REU`）；小效应组 **35 例**；含 4 例突变为脯氨酸（`N440P / N481P / F486P / N501P`）、3 例野生型脯氨酸（`P479D/L/V`）及 8 例涉甘氨酸突变 | 必须完整精确实现 `pro_close` 脯氨酸环闭合、甘氨酸以 `CA` 为 `nbr_atom` 的邻域判定、以及全量 `talaris2014` 氢键与排斥项 |

### 48 核 CPU 开源基线性能剖析与热点分析

48 核 CPU（2×Xeon Platinum 8360Y，开源 Rosetta `2026.29`，48 进程池）运行 BD-623 全部 50 突变（1,750 条轨迹）实测统计：

- **累计核时**：2,671,363.3 s（742.05 CPU·h）；单突变均值 53,427 s（14.84 CPU·h），范围 50,617.5–60,126.7 s（最慢 `Y369C`，位点差异约 ±10%）；
- **48 核满载端到端墙钟基线**：**55,653.4 s（15.46 h / 927.6 min）**，均摊单轨迹 CPU 耗时约 **1,526.5 s**；
- **24 核 CPU + 单卡 Ascend 950PR 性能门槛**：端到端墙钟 **`wall_time_s <= 5,565.3 s`（≤ 92.8 min / ≤ 1.55 h）**，即相对 48 核基线加速比 **`>= 10.0×`**，折合全系统均摊单轨迹墙钟须 **`<= 3.18 s`**。

按 `ddG-backrub.xml` 协议阶段拆解纯 CPU 耗时热点与根因：

| 协议阶段 | 单轨迹典型耗时占比（估计） | 计算特征与纯 CPU 瓶颈根因 | 异构加速策略 |
| --- | --- | --- | --- |
| 1. 结构准备与初始 WT 最小化 | ~1%–2% | 对 `BD-623.pdb` 执行加氢、`addcst`、`neighbor_shell` 选择与全构象 L-BFGS（`max_iter=5000`）；对同一输入 PDB 完全确定性 | Host 一次性导出全原子拓扑与参数；同一 PDB 确定性初始最小化结果精确复用（支持关闭共享以逐轨迹校验） |
| 2. Backrub + Sidechain MC（35,000 步） | **~55%–70%** | 6,125 万次串行 MC 试探；Rosetta C++ 虚函数派发、`EnergyGraph` 动态指针跳转、标量非键对计算严重受限于分支与访存延迟 | **下沉 NPU 驻留批处理算子（`BackrubMCKernel`）**：跨轨迹大规模并发 + 核内 SIMT/SIMD 局部增量能量评估 + 分段常驻减少下发开销 |
| 3. 后处理 WT `repack` + MUT `mutate` | **~5%–15%** | 3,500 次 `UseMultiCoolAnnealer(states=6)`，开启 `-ex1 -ex2` 与 `extrachi_cutoff=0` 导致旋转异构体组合数庞大 | **下沉 NPU 批处理算子（`PackRotamersKernel`）**：向量化并行构建 1-body/2-body 交互图（IG）+ 片上驻留 MultiCool 退火 |
| 4. 后处理 WT/MUT 约束 L-BFGS 最小化 | **~20%–35%** | 3,500 次 420 残基全构象 `bb=1, chi=1` 最小化（约 1,800–2,000 DOF），`fa_max_dis=9.0` 截断与 2,787 CA 约束，反复评估全原子能量与 AtomTree 解析梯度 | **下沉 NPU 批处理算子（`LbfgsMinKernel`）**：融合能量-力计算 + AtomTree 反向 F1/F2 梯度归约 + 批处理非单调 Armijo L-BFGS |
| 5. 四态 `InterfaceDdGMover` 与 GAM 汇总 | < 1% | 每轨迹 4 次静态单点 `fa_talaris2014` 打分 + 双差 + ZEMu GAM 重加权与 CSV 落盘 | 24 核 CPU 异步流水线重叠执行官方精确四态打分与残差核验（耗时完全隐藏于 NPU 计算周期内） |

可见，仅加速单一阶段（例如仅加速 Backrub 而不加速后处理 L-BFGS 与 Pack）按 Amdahl 定律难以稳定突破 10× 端到端门槛；本方案将 **Backrub/Sidechain MC、PackRotamers（IG + MultiCool）、Constrained L-BFGS** 三大热点全部下沉至 Ascend 950PR，并与 24 核 CPU 异步流水线协同，实现端到端高倍加速。

---

# 需求分析（required）

## 需求描述

在 **24 核 CPU（`taskset -c 0-23` 严格限核）+ 单卡 Ascend 950PR（CANN ≥ 9.0.0）** 上构建 Rosetta Flex ddG 端到端异构加速模块，完整等价实现官方 `test_cases/ddG-backrub.xml` 协议，在 `BD-623` 全部 50 个突变（1,750 条独立轨迹）上同时达成：

1. **性能目标**：完整端到端墙钟耗时 `wall_time_s <= 5,565.3 s`（加速比 ≥ 10.0×）；
2. **精度目标**：大效应组（15 例）中位相对误差 `< 5%`，小效应组（35 例）全部满足偏差 `< 1.0 · std_ref`，且 50/50 突变完整输出、无 NaN/Inf、`abs(ddG_residual) <= 0.01`；
3. **三条不可修改红线**：
   - 力场严格保持 `talaris2014`（等价开启 `-corrections::restore_talaris_behavior true`），不得替换为 REF2015 或删减能量项；
   - 采样与收敛预算不得调低：必须保持 `nstruct = 35`、`backrub_trials = 35000`、`Neighborhood distance="8.0"`（`cb8`）、`max_minimization_iter = 5000`、`abs_score_convergence_thresh = 1.0`；
   - 不得删减 `ddG-backrub.xml` 核心阶段：不允许跳过初始最小化、缩短采样步数或删减 WT/MUT 任一侧重排与后处理最小化。

## 需求拆解

1. **协议语义与接口对齐**：提供统一命令行入口 `run_flexddg_npu.py`，完整实现任务书 §2.3/§2.4 规定的 12 个参数、22 列 `results.csv` 与必填字段完整的 `run_summary.json`，所有异常输入与超核请求立即拒绝并返回非零退出码。
2. **全原子拓扑与 `talaris2014` 参数无损桥接**：从开源 Rosetta `2026.29` 精确导出全原子结构、二硫键、AtomTree 运动学拓扑、CA 约束、Dunbrack 2010 `-ex1 -ex2` 旋转异构体库及 `talaris2014` 全部力场参数表至 NPU 紧凑 SoA 缓冲。
3. **NPU 驻留式 Backrub + Sidechain MC 引擎**：在 Ascend 950PR 上并发执行多条独立轨迹的 35,000 步 Metropolis-Hastings 采样，保证跨突变完全独立采样、独立随机流与局部增量能量零漂移。
4. **NPU 批处理 PackRotamers 与约束 L-BFGS 引擎**：实现 `neighbor_shell` 的 WT `repack` 与 MUT `mutate`（`UseMultiCoolAnnealer states=6`）以及全构象 `bb=1, chi=1` 的 `lbfgs_armijo_nonmonotone` 最小化。
5. **四态打分与 `talaris2014` 数值一致性闭环**：通过 5 级校核体系与异步 CPU 四态精确打分核验，确保 7 个分子间项及 ZEMu GAM 结果严格自洽、`abs(ddG_residual) <= 0.01`。
6. **24 核严格限核与动态负载均衡**：全程严格遵循 `taskset -c 0-23` 约束，按轨迹粒度 LPT 动态分批与多流异步重叠，消除慢位点尾延迟并隐藏 Host 后处理开销。
7. **可复现工程组织与全量自验**：按任务书 §4/§5 规范交付 `rosetta_flex_ddg/` 源码布局、自动化负向/正向测试及私仓 `task_submission/` 验收交付件（不入库 `.db3` 大文件）。

---

# 详细设计（required）

## 算子与协议分析

### 核心数学公式与 `talaris2014` 力场规范

#### 1. 四态结合自由能双差与 ZEMu GAM 重加权

按 `chainstomove`（BD-623 为链 `A`）将复合物分离为解离态（沿界面方向刚体平移 $1000\ \text{Å}$ 且不做二次重排，完全对齐 `InterfaceDdGMover` 语义），单条轨迹的原始双差与逐项双差定义为：

$$\Delta\Delta G = (E_{\text{bound,mut}} + E_{\text{unbound,wt}}) - (E_{\text{bound,wt}} + E_{\text{unbound,mut}})$$

$$\Delta\Delta G_t = (E^{(t)}_{\text{bound,mut}} + E^{(t)}_{\text{unbound,wt}}) - (E^{(t)}_{\text{bound,wt}} + E^{(t)}_{\text{unbound,mut}})$$

所有仅依赖单链内部构象的能量项（`fa_dun`、`rama`、`p_aa_pp`、`omega`、`pro_close`、`fa_intra_rep`、`dslf_fa13`、`yhh_planarity`、`ref`、`hbond_sr_bb`）在四态双差中数学上精确抵消，剩余 7 个分子间项集合为：

$$\mathcal{T}_{\text{inter}} = \{\text{fa\_atr},\ \text{fa\_rep},\ \text{fa\_sol},\ \text{fa\_elec},\ \text{hbond\_sc},\ \text{hbond\_bb\_sc},\ \text{hbond\_lr\_bb}\}$$

自洽性残差定义为 $\text{ddG\_residual} = \Delta\Delta G - \sum_{t \in \mathcal{T}_{\text{inter}}} \Delta\Delta G_t$，验收要求 $|\text{ddG\_residual}| \le 0.01\ \text{REU}$。

ZEMu GAM 重加权公式（单位 `kcal/mol`，逐轨迹计算后跨 35 轨迹求均值与 `ddof=1` 样本标准差）：

$$\text{gam}(x, t) = -e^{a_t} + \frac{2 e^{a_t}}{1 + \exp\!\left(-x \cdot e^{b_t}\right)}, \qquad \Delta\Delta G_{\text{gam}} = \sum_{t \in \mathcal{T}_{\text{inter}}} \text{gam}(\Delta\Delta G_t, t)$$

7 项官方拟合参数严格固定（对齐 `test_cases/baseline/flexddg_lib.py`）：

| 能量项 $t$ | `fa_atr` | `fa_rep` | `fa_sol` | `fa_elec` | `hbond_sc` | `hbond_bb_sc` | `hbond_lr_bb` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $a_t$ | 2.313 | 1.659 | 6.940 | 0.697 | 1.902 | 0.063 | 2.738 |
| $b_t$ | -1.649 | -0.836 | -6.722 | -0.122 | -1.999 | 0.452 | -1.179 |

#### 2. `talaris2014` 完整能量项建模与截断口径

NPU 力场评估完整实现 `talaris2014`（`-corrections::restore_talaris_behavior true`）全部激活加权项，不删减任何能量项：

| 类别 | 能量项 | 物理形式与 `talaris2014` 关键实现细节 | 作用截断与阶段差异 |
| --- | --- | --- | --- |
| 2-body 非键 | `fa_atr`、`fa_rep` | Lennard-Jones 6-12 势 + 低距离线性排斥切换（$r < 0.6\sigma_{ij}$ 防 $r^{-12}$ 奇异）+ 末端三次样条平滑衰减；遵循 Rosetta 1-2/1-3/1-4 键距 count-pair 掩码（1-4 缩放因子 0.20） | Backrub/Pack/Score 默认 `fa_max_dis = 6.0 Å`；`fa_talaris2014_cst` 最小化显式设为 **`9.0 Å`** |
| 2-body 溶剂化 | `fa_sol` | Lazaridis-Karplus 隐式溶剂高斯排斥积分，含原子溶剂化体积 $V_j$、自由能 $\Delta G^{\text{free}}_i$ 与相关长度 $\lambda_i$，采用与 LJ 相同的 count-pair 与三次样条开关 | 同 `fa_max_dis`（6.0 Å / 9.0 Å） |
| 2-body 静电 | `fa_elec` | 距离相关介电常数 $\varepsilon(r) = 10 r$ 的库仑势（$E \propto q_i q_j / r^2$）叠加 Rosetta 偏移与边界平滑样条，含 count-pair 缩放 | 固定截断 **`5.5 Å`**（最小化截断变化不影响 5.5 Å 截断定义，以参数导出为准） |
| 2-body 氢键 | `hbond_sr_bb`、`hbond_lr_bb`、`hbond_bb_sc`、`hbond_sc` | `talaris2014` sp2/sp3 几何多项式势：耦合供体-受体距离 $r_{AH}$、角度 $\cos\theta_{AHD}$、$\cos\theta_{BAH}$ 及 sp2 二面角 $\chi_{\text{BAH}}$ 权重，按主链短程/长程、主-侧链、侧-侧链分类加权 | 供-受体距离截断 3.0 Å（重原子对约 4.2 Å） |
| 2-body 二硫键 | `dslf_fa13` | 6 自由度二硫键几何势（$d_{SS}$、双键角、三二面角），对 `BD-623` 全部 6 对 `CYD`（含 `A:C480–A:C488`）精确求值与求导 | 仅作用于拓扑登记的二硫键对 |
| 1-body 扭转与内部 | `fa_dun`、`rama`、`p_aa_pp`、`omega`、`pro_close`、`fa_intra_rep`、`yhh_planarity`、`ref` | Dunbrack 2010 主链依赖旋转异构体库（三次插值）、Ramachandran 与氨基酸倾向双三次样条、肽键 $\omega$ 偏离势、脯氨酸环闭合势（`pro_close` 对 `F486P/N481P` 至关重要）、残基内 `fa_intra_rep`、酪氨酸羟基平面性 `yhh_planarity` 与参考态能量 `ref` | 残基局部 1-body，单残基独立并行 |
| 约束项（仅最小化） | `atom_pair_constraint` | 谐波距离约束 $E_{\text{cst}} = \sum_{(i,j)\in\mathcal{C}} \left(\frac{\|\mathbf{r}_{CA,i}-\mathbf{r}_{CA,j}\| - d^0_{ij}}{0.5}\right)^2$，`weight = 1.0` | 仅在 `fa_talaris2014_cst` 阶段启用（BD-623 共 2,787 对，$d^0_{ij} \le 9.0\ \text{Å}$） |

### 接口与参数规格

#### 1. 统一命令行入口

```bash
taskset -c 0-23 python3 pipeline/run_flexddg_npu.py \
  --selected   <mutations_csv> \
  --pdb-dir    <pdb_dir> \
  --xml        <protocol_xml> \
  --out        <results_csv> \
  --summary    <run_summary_json> \
  --work       <work_dir> \
  --move-chain <chain_id> \
  --nstruct    35 \
  --trials     35000 \
  --cpu-cores  24 \
  --device-id  0 \
  --seed       111111
```

与标杆脚本 `test_cases/baseline/06b_ddG_flex_antigen.py` 的对应关系：保留全部核心协议参数（`--selected`、`--pdb-dir`、`--xml`、`--out`、`--work`、`--move-chain`、`--nstruct`、`--trials`），将 `--nproc` 统一为 `--cpu-cores`（正式验收固定 24），新增 `--device-id`（显式指定 Ascend 950PR 逻辑卡号，在分配任何设备内存前调用 `aclrtSetDevice`）、`--seed` 与 `--summary`。

#### 2. 参数合法域与前置校验顺序

Host 入口在加载模型或申请设备内存前，按以下严格顺序校验全部参数，任一失败即打印明确诊断信息并返回非零退出码：

| 校验序号 | 参数 | 类型与合法域 | 异常拒绝条件 |
| --- | --- | --- | --- |
| 1 | `--cpu-cores`、`--nstruct`、`--trials`、`--device-id`、`--seed` | 标量 int；`cpu-cores ∈ [1, 24]`（验收固定 24）、`nstruct > 0`（验收固定 35）、`trials > 0`（验收固定 35000）、`device-id >= 0`、合法 32/64 位整数 seed | 任一越界、非正数、超 24 核或格式错误立即报错退出；调用 `aclrtGetDeviceCount` 校验 `device-id` 未越界 |
| 2 | `--selected`、`--pdb-dir`、`--xml` | UTF-8 CSV、目录、XML 文件 | 路径不存在、不可读或 XML 解析失败/与 `ddG-backrub.xml` 核心协议节点缺失时退出 |
| 3 | `--work`、`--out`、`--summary` | 可写输出路径 | 父目录无法创建或无写权限时退出 |
| 4 | CSV 表头与内容 | 必含 6 列 `pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`，至少 1 行；`wt_aa`/`mut_aa` 为合法 20 种标准氨基酸单字母码 | 缺列、空表、重复突变键、非法氨基酸码时退出 |
| 5 | PDB 匹配与 `--move-chain` | 每个 `pdb_id` 在 `--pdb-dir` 中存在；`--move-chain` 存在于 PDB 链集合；按 `(chain, pdb_position, icode)` 精确定位残基且其实际单字母码严格等于 `wt_aa` | PDB 缺失、链不存在、位点不存在或 `wt_aa` 与结构不一致时立即拒绝计算 |

#### 3. 输出契约（`results.csv` 与 `run_summary.json`）

- **`results.csv`（`[N, 22]`，UTF-8 带表头）**：与输入 `bd623_mutations.csv` 逐行精确匹配，严格包含全部 22 列：
  - 列 1–6：`pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`；
  - 列 7–9：`ddG_mean(REU)`、`ddG_std(REU)`（`ddof=1`）、`ddG_all(REU)`（35 个有限浮点数，分号 `;` 分隔）；
  - 列 10–16：`ddG_fa_atr(REU)`、`ddG_fa_rep(REU)`、`ddG_fa_sol(REU)`、`ddG_fa_elec(REU)`、`ddG_hbond_sc(REU)`、`ddG_hbond_bb_sc(REU)`、`ddG_hbond_lr_bb(REU)`；
  - 列 17：`ddG_residual(REU)`（验收要求 `abs(ddG_residual) <= 0.01`）；
  - 列 18–19：`ddG_gam_mean(kcal/mol)`、`ddG_gam_std(kcal/mol)`（`ddof=1`）；
  - 列 20–22：`nstruct_done`（严格 `35`）、`backrub_trials`（严格 `35000`）、`cpu_time(s)`（该突变累计耗时，非负有限值）。
  - 写入前校验：若任意突变未完成、缺列、存在 NaN/Inf 或 `abs(ddG_residual) > 0.01`，入口返回非零退出码。
- **`run_summary.json`**：UTF-8 JSON 对象，必含 `wall_time_s`（完整端到端墙钟秒数，含参数导出、算子加载、采样、后处理与落盘）、`mutations_completed`（`50`）、`cpu_cores_limit`（`24`）、`npu_device`、`cann_version`、`driver_version`、`seed`、`git_commit`、`command` 及各阶段耗时分解字段 `stage_breakdown_s`。

---

## 整模块异构架构与算子实现

### 端到端异构流水线总览

```text
+======================================================================================+
|                        Host CPU (严格绑定 24 核: taskset -c 0-23)                     |
|                                                                                      |
| [Stage 0: 前置准备 (一次性/唯一起始 PDB)]                                             |
|   CLI/CSV/PDB 严格校验 -> Rosetta 2026.29 (talaris2014) 全原子拓扑与力场参数表导出      |
|   -> 21 位点 cb8 neighbor_shell 与 2,787 CA 约束构建 -> 确定性初始 WT 约束 L-BFGS 最小化 |
|                                      |                                               |
|                                      v (静态力场表 + 初始 WT 构象一次性 H2D 常驻 HBM)   |
| [异步微批调度器: 50 突变 × 35 轨迹 = 1,750 独立轨迹, LPT 按 shell 规模降序动态分批]     |
|   Stream A (H2D): 下发 Batch k+1 轨迹种子/位点索引/初始坐标切片                         |
|   Stream C (D2H): 回传 Batch k-1 终态 WT & MUT 构象                                    |
|   CPU 工作池 (<=20 核, 与 NPU 完全重叠):                                                |
|     对 Batch k-1 执行官方精确 fa_talaris2014 四态打分 (bound/unbound × wt/mut)         |
|     -> 校验 |ddG_residual| <= 0.01 -> ZEMu GAM 计算 -> 按突变聚合落盘 results.csv      |
+======================================|========================^======================+
                                       | ACL Async Stream B     | D2H 终态构象
                                       v                        |
+======================================================================================+
|                     Ascend 950PR 单卡 (arch35 / Da Vinci v35)                         |
|  [HBM 常驻区]                                                                         |
|    - 静态只读表 (~20-40 MB): LJ/LK/电荷/HBond/Dunbrack-ex1-ex2/Rama/P_aa_pp/CA 约束    |
|    - 批构象缓冲 (Batch B=128~256, ~10-40 MB): SoA 全原子坐标、扭转角、局部近邻表、RNG状态 |
|                                                                                      |
|  [Kernel Group 1: BackrubMCKernel (Ascend C SIMT + Vector, 分段常驻)]                 |
|    B 条独立轨迹并发执行 35,000 步 Backrub + Sidechain MC (mc_kt=1.2, ex1/ex2)          |
|    -> 增量局部 Verlet 近邻表 + 融合非键/氢键/1-body 能量 + FP64 防漂移累加 + 周期重同步 |
|                                      |                                               |
|                                      v (复制 Backrub 终点构象至 WT 支路与 MUT 支路)    |
|  [Kernel Group 2: PackRotamersKernel (Ascend C Vector + SIMT)]                        |
|    WT repack (repackonly) & MUT mutate (res_mutate) 并发执行:                         |
|    (2a) 包围球裁剪 + 向量化 1-body/2-body Rotamer IG 矩阵构建                         |
|    (2b) 片上查表驻留 UseMultiCoolAnnealer (states=6) 多轮降温退火                      |
|                                      |                                               |
|                                      v (2B 构象: B 条 WT + B 条 MUT 合并批处理)        |
|  [Kernel Group 3: LbfgsMinKernel (Ascend C Vector + SIMT)]                            |
|    fa_talaris2014_cst (fa_max_dis=9.0, 2,787 CA 约束, bb=1/chi=1, max_iter=5000):     |
|    正向 AtomTree 坐标递推 -> 融合能量与 F1/F2 受力 -> 反向树梯度扫描 -> 批 L-BFGS 迭代  |
|    (支持活跃掩码提前退出已收敛轨迹, 非单调 Armijo 线搜索 tol=1e-6, abs_conv=1.0)        |
+======================================================================================+
```

### 3.2.1 Host 侧设计（24 核 CPU 调度、参数导出与负载均衡）

#### 1. 一次性全原子拓扑与 `talaris2014` 参数导出

为从源头上消除人工抄写力场常数导致的精度漂移，Host 侧在 Stage 0 调用本地开源 Rosetta `2026.29`（开启 `-corrections::restore_talaris_behavior true -in:file:fullatom -ignore_unrecognized_res -ignore_zero_occupancy false -ex1 -ex2`）对输入 PDB 执行一次解析并导出标准化二进制参数包：

- **全原子拓扑**：每个残基的原子列表、原子类型编号、`talaris2014` 原子部分电荷、键连邻接与 1-2/1-3/1-4 count-pair 掩码、二硫键配对表（`BD-623` 全部 6 对 `CYD`）、链断裂与 FoldTree Jump 标记（精确标记 `A:516–522` 缺失环，禁止跨越 Jump 形成主链连续段）；
- **AtomTree 运动学表**：每个残基主链 $(\phi, \psi, \omega)$ 与侧链 $\chi_1 \dots \chi_4$ 的旋转轴锚点原子、旋转子树原子范围与逆序梯度汇聚拓扑（含脯氨酸环闭合约束及甘氨酸无侧链标记）；
- **`neighbor_shell` 与 CA 约束表**：直接调用官方 `ddG-backrub.xml` 中的 `Neighborhood(distance=8.0)`（以 `nbr_atom` 计算：非甘氨酸为 `CB`，甘氨酸为 `CA`）+ `PrimarySequenceNeighborhood(lower=1, upper=1)` 生成每个突变位点的精确 `neighbor_shell` 位图；由 `AddConstraintsToCurrentConformationMover(coord_dev=0.5, max_distance=9, CA_only=1)` 导出 2,787 对 CA 索引与锚定参考距离 $d^0_{ij}$；
- **力场查找表与旋转异构体库**：导出 `talaris2014` LJ/LK 原子对参数表、HBond sp2/sp3 多项式系数、Rama/P_aa_pp/Omega 样条系数、以及针对 `neighbor_shell` 残基按 `-ex1 -ex2` / `extrachi_cutoff=0` 展开的 Dunbrack 2010 旋转异构体模板与概率/标准差表。参数导出耗时全程计入 `wall_time_s`。

#### 2. 初始 WT 约束最小化与轨迹严格独立原则

- **初始 WT 最小化（Stage 0）**：`ddG-backrub.xml` 中 `addcst -> apply_score -> neighbor_shell_storer -> minimize -> clearcst` 位于 `backrub` 之前，此时构象为未突变的野生型复合物，且 `lbfgs_armijo_nonmonotone` 不含任何随机性。因此同一输入 PDB 的全部 1,750 条轨迹对应的初始最小化输入与输出完全一致。本方案完整执行全预算初始最小化（`max_iter=5000`, `tol=1e-6`, `abs_score_convergence_threshold=1.0`，绝不跳过或降预算），将最小化后的野生型构象作为全部 1,750 条轨迹的统一初始态，同时提供 `--no-share-init-min` 调试选项用于逐轨迹复核等价性。
- **Backrub 及后续阶段（100% 独立）**：即便多个突变共享同一抗原位点（50 突变分布于 21 位点），**所有 50 × 35 = 1,750 条轨迹均使用各自独立随机流独立执行完整 35,000 步 Backrub/Sidechain MC、WT/MUT Pack 与 WT/MUT 约束 L-BFGS 最小化**，严禁跨突变共享 Backrub 轨迹，严格遵守任务书统计独立性要求与三条红线。

#### 3. 24 核资源约束下的负载均衡与异步流水线

- **严格限核治理**：进程启动时检查实际 CPU 亲和性集合并强制 `worker_threads + scheduler_threads <= min(cpu_cores, available_affinity_cores)`（验收固定 `<= 24`），显式设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 防止底层库隐式拉起超核线程。
- **跨突变微批聚合与 LPT 调度**：
  - 将 1,750 条独立轨迹按 `neighbor_shell` 残基数及突变侧链旋转异构体规模从大到小排序（Longest Processing Time first），优先派发高复杂度位点（如 `neighbor_shell` 达 25–29 残基或涉及 `Arg/Tyr/Trp` 的突变），将高复杂度小批次安排在前期消化，避免最后阶段出现单批尾延迟。
  - 将相同或相近 `neighbor_shell` 规模的轨迹打包至同一微批 $B$（默认 $B = 128$ 或 $256$，按运行时查询的 NPU AI Core 数量取整数倍，如 `blockDim` 的 4–8 倍），最大化同批核间负载均衡并降低 SIMT/循环迭代次数差异。
- **三阶段双缓冲异步重叠**：
  - 主调度线程维护 2 组 ping-pong 锁页内存（pinned host buffer）与 3 条 ACL Stream（`H2D`、`Compute`、`D2H`）；
  - 当 NPU 在 `Compute Stream` 上处理 Batch $k$ 时，`H2D Stream` 异步准备 Batch $k+1$，CPU 侧 20 个工作进程/线程并行对 Batch $k-1$ 回传的 WT/MUT 终态坐标执行官方 Rosetta `fa_talaris2014` 四态打分（`bound_wt / unbound_wt / bound_mut / unbound_mut`）及与 NPU 能量值的交叉一致性检查，每完成一个突变的全部 35 条轨迹即汇总写入 `results.csv` 并 `flush`。
- **无状态独立伪随机流**：采用 `Philox-4x32-10` 计数器型 RNG，以 `(seed, mut_idx, struct_idx, stage_id, step_id)` 为密钥与计数器，无需在片上维护大体积 MT19937 状态数组（单轨迹 RNG 状态仅 16 字节），且在不同批大小与调度顺序下均可复现一致的统计分布。

### 3.2.2 Kernel 侧设计（Ascend 950PR / arch35 三大算子组与 Tiling/内存策略）

本方案面向 **Ascend 950PR（`arch35` / Da Vinci v35）** 设计原生 Ascend C 算子（内核直调），运行时通过 CANN 平台接口动态查询实际可用 AI Core（AIV）核数与 UB 容量完成分核与 Tiling 计算，**不在源码中硬编码设备核数或固定 UB 大小**。

#### 1. 设备内存布局与片上存储预算

- **SoA + 32B 对齐数据布局**：
  - 批构象坐标 `coords[B, 3, N_atom_pad]`（`N_atom_pad` 向上 32B/64B 对齐，BD-623 为 6,336），使同批/同轨迹连续原子的 $x, y, z$ 坐标满足 32B 连续合并搬运（DMA 友好）；
  - 每条轨迹维护紧凑活动子集映射表：`active_res_ids[N_shell_max]`（$\le 32$ 残基）、`env_res_ids[N_env_max]`（9 Å 扩展环境壳内约 80–150 残基）与紧致原子偏移数组；
  - 静态力场表（LJ $\sigma, \varepsilon$ 按原子类型矩阵 `[N_atype, N_atype]`、LK 参数、HBond 系数、Rama/P_aa_pp 样条表）只读常驻 HBM，核内按需预取至 UB/L1。
- **HBM 峰值容量估算**（以单微批 $B = 256$ 轨迹估算）：
  - 批坐标与备份缓冲（Backrub 当前/提议/最优、WT/MUT 分叉构象，FP64/FP32 双视图）：约 120–240 MB；
  - 批空间近邻表（每原子最大 64–96 邻居，按 `fa_max_dis=9.0` 预留）：约 400–650 MB；
  - 批 PackRotamers 1-body/2-body 交互图（仅存储 `neighbor_shell` 有效对，带包围球稀疏压缩）：约 300–800 MB；
  - 批 L-BFGS 历史缓冲（$2B$ 个 WT/MUT 构象 × 约 2,000 DOF × 历史窗口 $m=12$ × FP64）：约 100 MB；
  - 静态只读参数表与中间 workspace：约 100 MB；
  - **总 HBM 占用约 1.2–2.0 GB**，远低于 Ascend 950PR 单卡 HBM 容量，具备充裕空间支持更大批大小调优与多流 ping-pong 缓冲。

#### 2. 算子组 1：`BackrubMCKernel`（驻留式批处理 Backrub + Sidechain MC）

- **分核与持久化策略**：
  - `blockDim = min(available_aiv_cores, B)`，当 $B > \text{available\_aiv\_cores}$ 时采用核内多轨迹轮转或按 `available_aiv_cores` 的整数倍设微批；核内结合 `arch35` SIMD/SIMT 向量宽度并行计算单次提议的原子对能量。
  - 将 35,000 步拆分为 $N_{\text{chunk}}$ 段（默认每段 $K_{\text{chunk}} = 1,000 \sim 2,500$ 步，即 14–35 次内核启动，具体由真机剖析微调），每段内完全驻留 NPU 自主完成提议、几何更新、增量打分、Metropolis 判据与接受/回滚，彻底消除 35,000 次 Host-Device 启动开销。
- **提议动作与几何更新**：
  - 严格对齐 `BackrubProtocol` 动作集：
    - **Backrub 主链旋转**：在 `neighbor_shell` 内选择同一非断裂连续链段内的支点区间 $[r_a, r_b]$（长度 2–12 残基，跳过 `A:516–522` 链断裂与链末端不合法支点），以 $\hat{\mathbf{u}} = \frac{\mathbf{r}_{CA,b} - \mathbf{r}_{CA,a}}{\|\mathbf{r}_{CA,b} - \mathbf{r}_{CA,a}\|}$ 为轴抽取旋转角 $\theta$ 执行 Rodrigues 刚体旋转，并对起止肽平面执行耦合局部旋转 $\tau_1, \tau_2$ 以维持 $C_\alpha$ 键角几何，同步评估 `BackrubMover` 主链键角弯曲约束项（`mm_bend` / Rosetta Backrub 键角势）；
    - **Sidechain 旋转异构体与 $\chi$ 微扰**：在 `neighbor_shell` 中随机选取可动残基（`CYD` 二硫键半胱氨酸保持二硫键拓扑受保护），按 `-ex1 -ex2`、`extrachi_cutoff=0` 采样 Dunbrack 2010 旋转异构体或高斯 $\chi$ 角微扰，经侧链内坐标重建原子位置（脯氨酸保持 5 元环闭合及 `pro_close` 几何）。
- **增量局部能量评估与防漂移**：
  - 维护带 $0.5\ \text{Å}$ 缓冲壳的局部 Verlet 近邻表（截断 $6.0 + 0.5 = 6.5\ \text{Å}$），仅当某残基移动原子相对建表坐标最大位移 $> 0.25\ \text{Å}$ 时触发该残基局部近邻刷新。
  - 单次提议仅评估“变动残基集 $S_{\text{moved}}$（1–12 残基）内部 + $S_{\text{moved}}$ 与其近邻残基集 $N(S_{\text{moved}})$”的旧/新能量差 $\Delta E = E_{\text{new}} - E_{\text{old}}$；UB 内按原子分块（如 `TILE_A = 64`）双缓冲搬入原子坐标与参数，使用向量指令单遍融合计算 `fa_atr`、`fa_rep`、`fa_sol`、`fa_elec` 与 `hbond_*`，叠加 1-body 能量项差值。
  - **数值稳定机制**：能量差 $\Delta E$ 与轨迹累计总能 $E_{\text{cur}}$ 使用 **FP64 累加**，Metropolis 判据 $u < \exp(-\Delta E / k_B T)$（$k_B T = 1.2$）在 FP64 下判定；每 1,000 步执行一次局部活动壳能量全量重算回校准，彻底消除 35,000 步增量累积舍入漂移。
- **UB 内存预算核算**：单核 UB 同时容纳当前变动原子块坐标/参数（双缓冲）、近邻原子块坐标/参数（双缓冲）、距离矩阵与掩码临时向量、以及局部 1-body 查表切片，总用量按平台查询 UB 容量留出 ≥ 20% 余量动态确定原子分块宽度 `TILE_A`，避免 UB 溢出或频繁换出。

#### 3. 算子组 2：`PackRotamersKernel`（IG 并行构建 + `UseMultiCoolAnnealer states=6`）

Backrub 终点构象在设备端直接复制为 WT 与 MUT 两份状态，分别执行 `repack`（仅 `neighbor_shell` 保持原生氨基酸重排）与 `mutate`（突变位点替换为 `mut_aa`，其余 `neighbor_shell` 重排，`everythingelse` 固定不动）：

- **阶段 2a — 批处理 1-body / 2-body Interaction Graph（IG）构建**：
  - 1-body 能量 $E_1(p, r)$：对每个可动残基位置 $p \in \text{neighbor\_shell}$ 及其全部 `-ex1 -ex2` rotamer $r$，并行计算 rotamer 内部能量（`fa_dun`、`fa_intra_rep`、`pro_close`、`ref`）+ 与所有固定背景残基的非键/氢键能量。由于各 $(p, r)$ 完全独立，可直接按 `(traj_id, pos_id, rot_id)` 三维扁平化铺满全卡 AI Core 向量计算单元。
  - 2-body 能量 $E_2(p_1, r_1; p_2, r_2)$：先用残基包围球与 rotamer 外接球两级裁剪过滤不可能发生 $< 6.0\ \text{Å}$ 接触的残基对与 rotamer 对，仅对有效活跃对按 Tile 分块并行计算融合非键+氢键矩阵，输出紧凑分块 IG 存回 HBM。
- **阶段 2b — NPU 驻留 `UseMultiCoolAnnealer (states=6)`**：
  - 完整对齐 Rosetta `MultiCoolAnnealer`（`states=6`）算法流程：第一阶段在高温至低温区间执行多轮快速退火采样，动态维护每条轨迹能量最低且构象互异的 **6 个候选低能状态**；第二阶段从这 6 个状态出发执行低温精细淬火与贪心局部收敛，选出全局最低能旋转异构体组合。
  - 每步状态替换仅需对当前残基的邻居列表做 $O(\text{deg})$ 次 IG 查表加减（每条轨迹仅需数十次标量累加），因此单核可在 UB/L1 缓存下以极高吞吐并发推进多条轨迹，并将最优 rotamer 坐标写回 WT/MUT 构象缓冲。

#### 4. 算子组 3：`LbfgsMinKernel`（批处理全构象约束扭转空间 L-BFGS）

对每条轨迹的 WT 与 MUT 构象（合批规模 $2B$）分别执行 `addcst -> minimize -> clearcst`，严格遵守 `fa_talaris2014_cst`（`fa_max_dis = 9.0`、`atom_pair_constraint weight = 1.0`）、`bb=1, chi=1`、`type = lbfgs_armijo_nonmonotone`、`tolerance = 1e-6`、`max_iter = 5000`、`abs_score_convergence_threshold = 1.0`：

- **步骤 A — 正向运动学重建（`ForwardKinematicsKernel`）**：由当前扭转角向量 $\mathbf{x} \in \mathbb{R}^{D}$（$D \approx 1800\text{–}2000$）沿 3 条链主链（及 `A:516–522` 链断裂后的独立子段）与 420 个侧链分支并行递推全原子三维坐标（FP64 旋转矩阵递推，避免长链杠杆误差累积）。
- **步骤 B — 融合能量与原子力矩内核（`EvalEnergyGradKernel`）**：
  - 按 `fa_max_dis = 9.0 Å` 刷新/复用 `MinimizationGraph` 空间近邻表（Armijo 线搜索小步试探时复用到位，仅当最大原子位移超壳厚时重建）；
  - 单遍遍历原子对、二硫键对与 2,787 对 CA 距离约束，同时计算总能量 $E$（FP64 归约）及 Rosetta 标准力的双向量表示 $\mathbf{f}_1 = \mathbf{r}_i \times \mathbf{F}_i$ 与 $\mathbf{f}_2 = \mathbf{F}_i$（同时包含 `fa_atr / fa_rep / fa_sol / fa_elec / hbond_* / dslf_fa13 / atom_pair_constraint` 及 1-body `fa_dun / rama / p_aa_pp / omega / pro_close / fa_intra_rep / yhh_planarity` 的精确解析导数）。
- **步骤 C — AtomTree 反向树扫描（`BackwardTorsionGradKernel`）**：
  - 从侧链末端向 $C_\alpha$、再沿各连续链段从 C 端向 N 端执行逆序前缀和汇总下游总力矩与总力 $(\mathbf{F}_{1,\text{sub}}, \mathbf{F}_{2,\text{sub}})$，按旋转轴单位向量 $\hat{\mathbf{e}}_k$ 与轴上锚点 $\mathbf{p}_k$ 计算每个扭转角精确解析梯度：
    $$\frac{\partial E}{\partial \theta_k} = \mathbf{F}_{1,\text{sub}} \cdot \hat{\mathbf{e}}_k + \mathbf{F}_{2,\text{sub}} \cdot (\hat{\mathbf{e}}_k \times \mathbf{p}_k) + \frac{\partial E_{\text{1-body}}}{\partial \theta_k}$$
- **步骤 D — 批处理非单调 Armijo L-BFGS 迭代与活跃掩码**：
  - 在 NPU 内完成 L-BFGS 两循环递推（$s_k, y_k$ 点积与缩放全 FP64）、非单调 Armijo 条件检验（比较历史窗口最大参考能量 $E_{\max}$）及双收敛判据检查（`tolerance = 1e-6` 与 `abs_score_convergence_threshold = 1.0`）；
  - 维护 `active_mask[2B]`：已收敛轨迹立刻移出后续迭代，将其对应的计算资源重新压缩分配给尚未收敛轨迹（Stream Compaction），避免个别难收敛突变（如大空间位阻 `G476R`、`F486P`）拖慢整批执行；
  - 若线搜索遇到极端碰撞导致的非有限值（NaN/Inf）或曲率条件 $s_k^T y_k \le 0$，安全缩减步长并重置 L-BFGS 历史（回退最速下降步），保证数值鲁棒性。

#### 5. TilingKey 规划与工程编译规范

| TilingKey 编码范围 | 对应核函数模块 | 核心分派维度 | 编译边界与同步要求 |
| --- | --- | --- | --- |
| `0x1000–0x10FF` | `BackrubMCKernel` | `neighbor_shell` 规模分档（小 ≤16 / 中 17–24 / 大 >24）、chunk 步长 | 独立 `arch35` 编译单元；精确流水同步（`MTE2->V->MTE3`），禁止滥用 `PIPE_ALL` |
| `0x2000–0x20FF` | `PackIGKernel` / `MultiCoolKernel` | IG 稀疏分块尺寸、rotamer 密度分档 | 1-body 与 2-body 核分离；UB 双缓冲与显式尾块填充对齐 |
| `0x3000–0x30FF` | `LbfgsKinematics` / `LbfgsEvalGrad` / `LbfgsUpdate` | 活跃轨迹压缩规模、`fa_max_dis = 9.0` 约束开关 | FP64 累加与树扫描保持独立精度边界；Host/Kernel 共享声明头文件，严禁声明漂移 |

所有核函数遵循：派生偏移与字节尺寸统一先提升为 `uint64_t` / `int64_t` 防 32 位乘法溢出；尾块搬运严格匹配有效长度、padding 与 32B 行距，绝不越界写 GM；非显然常量具名并注释物理含义。

### 3.2.3 `talaris2014` 数值一致性保障体系

由于 Flex ddG 是高敏热力学系综采样算法，微小能量项系统偏差在 35,000 步 MC 与 5,000 步 L-BFGS 中可能被放大。本方案建立 **五级递进校核机制**：

1. **L1 逐项单点能量精度校验**：抽取 BD-623 初始构象及 50 突变随机扰动构象（重点包含 `A:C480–A:C488` 二硫键、`F486P / N481P` 脯氨酸、`G476R` 甘氨酸与 `S514Q` 链断裂邻域），逐一比对 NPU 全部激活 `talaris2014` 能量项（17 项核心力场项及最小化阶段 `atom_pair_constraint`）与 Rosetta `2026.29` 原生打分，目标单能量项绝对误差 `< 1e-4 REU`、总分误差 `< 1e-3 REU`。
2. **L2 解析梯度与 L-BFGS 收敛一致性校验**：比对 `LbfgsMinKernel` 输出的约 1,800–2,000 维 AtomTree 扭转角梯度与 Rosetta `eval_atom_derivative` 及中心有限差分，目标最大相对误差 `< 1e-5`；从同一构象出发验证 NPU 与 CPU L-BFGS 终态总能一致（`< 1e-2 REU`）。
3. **L3 轨迹重放与 MC 接受率统计校验**：使用 Rosetta 产生的提议序列在 NPU 上回放 $\Delta E$，核验 Metropolis 接受概率分布、平均接受率及 Backrub 骨架 RMSD 分布无系统性偏移。
4. **L4 四态精确打分与残差零漂移保障**：利用异步流水线中完全隐藏的 24 核 CPU 算力，对 NPU 输出的每条轨迹 WT/MUT 终态执行官方 Rosetta `fa_talaris2014` 四态打分（`bound_wt / unbound_wt / bound_mut / unbound_mut`）并与 NPU 侧打分做在线交叉核验，确保 7 项分子间能量与原始双差残差严格满足 `abs(ddG_residual) <= 0.01`（预期 `<= 0.001`）。
5. **L5 全量 50 突变系综统计验收**：使用官方 `verify_bd623.py` 评估大效应中位相对误差、小效应全量 $\sigma_i$、全 50 突变 Pearson $r$ 以及平均偏差（排查正负系统偏置）。

### 3.2.4 端到端性能预算分析（设计期估算）

> 以下时间预算为基于 BD-623 工作量、48 核基线统计与 Ascend 950PR 并行度建立的**设计期分析估算模型**，用于指导开发优先级与瓶颈定位，**不代表真机实测结果**。

| 阶段 | 48 核 CPU 基准均摊墙钟 | 24 核 CPU + 950PR 设计目标估算 | 核心加速杠杆 |
| --- | ---: | ---: | --- |
| Stage 0：输入校验、参数导出与初始 WT 最小化 | ~750 s | **~15–30 s** | 一次性参数表导出 + 唯一起始 PDB 确定性初始最小化复用（计入 `wall_time_s`） |
| Stage 1：1,750 条 × 35,000 步 Backrub/Sidechain MC | ~33,000–39,000 s | **~1,100–2,200 s** | NPU 批并发 + 分段常驻消除下发延迟 + Verlet 局部增量融合评估（消除虚函数与指针跳转） |
| Stage 2：3,500 次 PackRotamers (`MultiCool states=6`) | ~3,000–8,000 s | **~150–400 s** | 向量化 IG 并行构建 + 包围球裁剪 + 片上查表驻留退火 |
| Stage 3：3,500 次全构象约束 L-BFGS 最小化 | ~11,000–19,000 s | **~500–1,200 s** | 融合能量-F1/F2 受力 + 反向树扫描 + 批 L-BFGS 活跃掩码早停 |
| Stage 4：四态打分、GAM 计算与 CSV/JSON 落盘 | ~100 s | **~10–30 s**（绝大部分与 NPU 重叠隐藏） | 24 核 CPU 双缓冲异步后处理重叠 |
| **端到端总墙钟 `wall_time_s`** | **55,653.4 s**（15.46 h） | **~1,775–3,860 s（29.6–64.3 min）** | 相对 48 核基线预期 **14.4×–31.4×**，相对门槛 **5,565.3 s** 预留 ≥ 30% 安全余量 |

开发期将通过 `msprof` 算子耗时剖析与阶段墙钟打点持续迭代：优先消除 Host-Device 隐式同步、动态调优微批大小 $B$ 与 Backrub chunk 步数 $K_{\text{chunk}}$、以及优化 L-BFGS 高频热点（9.0 Å 近邻遍历与非单调线搜索步数）。

---

## 工程组织与交付目录结构

严格对齐任务书 §5 源码目录规范，保证结构清晰、无冗余文件、无硬编码机器路径、不入库大体积 `.db3` 轨迹文件：

```text
rosetta_flex_ddg/
├── README.md                        # 环境依赖、编译构建、24 核+950PR 一键复现与自验说明
├── build.sh                         # 一键构建脚本（调用 CMake 编译 Ascend C kernels 与 Host 扩展）
├── CMakeLists.txt                   # 顶层构建配置（按 CANN 9.x / arch35 工具链规范注册源码）
├── env/
│   └── setup_env.sh                 # CANN 9.x、Rosetta 2026.29 及严格限核线程环境变量配置
├── pipeline/
│   ├── run_flexddg_npu.py           # 统一端到端 CLI 入口（对齐 §2.3/§2.4，含前置校验与计时）
│   ├── param_exporter.py            # Rosetta talaris2014 全原子拓扑、CA 约束与 rotamer 导出器
│   ├── scheduler.py                 # 24 核 LPT 微批调度与双缓冲异步流水线
│   └── scoring_and_report.py        # 四态双差校验、ZEMu GAM 重加权、results.csv/run_summary.json 输出
├── kernels/
│   ├── include/                     # Host/Kernel 共享 Tiling、SoA 数据结构、talaris2014 常量与 launch 声明
│   ├── host/                        # ACL 设备管理、Tiling 计算、多流编排与 PyBind/C 扩展绑定
│   └── arch35/                      # Ascend 950PR 原生 Ascend C 算子（BackrubMC / PackRotamers / LbfgsMin）
└── tests/
    ├── test_cli_validation.py       # 负向与边界参数校验回归
    ├── test_energy_and_grad.py      # L1 能量项与 L2 AtomTree 解析梯度逐项校验
    └── run_e2e_bd623.sh             # taskset -c 0-23 一键端到端运行 + verify_bd623.py 验收脚本
```

私仓根目录按任务书 §4 同步维护 `task_submission/` 验收目录（不含 `.db3` 中间文件）：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
└── 3.2 性能自验证日志.log
```

## 支持硬件

| 硬件/产品型号 | 设计支持 | 本任务实测要求与状态说明 |
| --- | :---: | --- |
| **Ascend 950PR（单卡）+ 24 核 CPU** | √（`arch35` 原生适配） | 本任务唯一正式目标环境；将完成完整 50 突变精度、端到端性能与全量边界自验 |
| Ascend 950DT | 不涉及（未实测） | 同属 `arch35` 架构但非本任务验收环境，不宣称已完成 DT 真机实测 |
| Atlas 800I/T A2 / Atlas A3 | ×（不涉及） | 任务书仅适配 Ascend 950PR，本设计面向 `arch35` 特性，不覆盖 `arch22` |
| Atlas 200I/500 A2 / 历史训练及推理系列 | ×（不涉及） | 非本任务支持范围 |

## 约束与限制

1. **协议与三条红线不可违背**：严禁切换 `talaris2014` 力场、调低 `nstruct=35` / `backrub_trials=35000` / `cb8` 8 Å / `max_minimization_iter=5000` / `abs_score_convergence_thresh=1.0` 或删减 `ddG-backrub.xml` 任一核心阶段；`test_cases/ddG-backrub.xml` 与 `test_cases/verify_bd623.py` 保持只读原件。
2. **24 核严格限核约束**：验收须在 `taskset -c 0-23`（或容器 `--cpuset-cpus=0-23`）下执行；进程内所有线程池/进程池并发数严格不超过 `--cpu-cores`（≤ 24），禁止通过超核或跨机并行规避。
3. **输入结构与接口约束**：输入 PDB 须为无 `HETATM` 的标准全原子蛋白结构（含 20 种标准氨基酸，支持二硫键、插入码与链断裂）；非标准配体/辅因子暂不在本任务验收范围内。
4. **统计确定性契约**：Flex ddG 为蒙特卡洛随机系综算法，不承诺不同运行/不同硬件间逐比特一致，以第 3.2 节统计精度判据验收；支持 `--seed` 控制无状态 Philox RNG 以实现同配置可复现排查。
5. **计时与入库约束**：`wall_time_s` 必须覆盖入口启动、参数导出、算子加载、全部 1,750 条轨迹采样与后处理直至 `results.csv` 落盘的完整墙钟时间（不含一次性环境安装与源码编译）；大体积 `.db3` 临时目录与编译产物严禁入库。

---

# 可维可测分析

## 精度标准/性能标准

严格对齐任务书第 3 章与自动化验收脚本 `test_cases/verify_bd623.py`：

| 验收维度 | 判据指标与阈值 | 标准来源 |
| --- | --- | --- |
| **完整性与自洽性** | 50/50 突变全部完成，22 列格式完整，无 NaN/Inf；每突变 `nstruct_done == 35`、`backrub_trials == 35000`、`ddG_all(REU)` 含 35 个有效有限值、`abs(ddG_residual(REU)) <= 0.01` | 任务书 §2.4、§3.2，`verify_bd623.py` |
| **大效应组精度** | $\lvert \overline{\Delta\Delta G}_{\text{ref},i} \rvert > 1.0\ \text{REU}$（共 15 例）：相对误差 $\text{rel\_err}_i = \frac{\lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref},i} \rvert}{\lvert \overline{\Delta\Delta G}_{\text{ref},i} \rvert}$ 的**中位数 `< 5%`（`< 0.05`）** | 任务书 §3.2，`verify_bd623.py` |
| **小效应组精度** | $\lvert \overline{\Delta\Delta G}_{\text{ref},i} \rvert \le 1.0\ \text{REU}$（共 35 例）：**全部 35 例**均满足 $\sigma_i = \frac{\lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref},i} \rvert}{\text{std}_{\text{ref},i}} < 1.0$（即 $\max(\sigma) < 1.0$） | 任务书 §3.2，`verify_bd623.py` |
| **辅助系统偏移诊断** | 同步输出全 50 突变 Pearson 相关系数 $r$、中位 $\sigma$、最大 $\sigma$ 及大/小效应逐突变误差明细，确认无系统性漂移 | 任务书 §3.2，`verify_bd623.py` |
| **端到端性能** | 24 核 CPU（`taskset -c 0-23`）+ 单卡 Ascend 950PR 运行 BD-623 全量 50 突变，端到端墙钟 **`wall_time_s <= 5,565.3 s`**（相对 48 核基线 `55,653.4 s` 加速比 **`>= 10.0×`**） | 任务书 §3.3，`verify_bd623.py` |
| **内存要求** | 任务书 §3.4 明确标注“不涉及”；开发期仍记录 NPU HBM 峰值与 Host RSS 峰值，防止 OOM 风险 | 任务书 §3.4 |

## 测试与自验方案

1. **模块级单元测试（不依赖外部私密数据，干净检出可运行）**：
   - `test_energy_and_grad.py`：在 `BD-623.pdb` 及代表性突变位点上验证 L1 全部激活 `talaris2014` 能量项（17 项核心力场项及 `atom_pair_constraint`）误差（`< 1e-4 REU`）、L2 AtomTree 解析梯度误差（`< 1e-5`）、ZEMu GAM 公式数值与 `flexddg_lib.py` 逐位一致；
   - `test_cli_validation.py`：覆盖任务书 §3.5 要求的全部负向场景（缺失 PDB 文件/目录、CSV 缺失必选列、空 CSV、`wt_aa` 与 PDB 实际残基不匹配、非法 `--move-chain`、`--cpu-cores <= 0` 或 `> 24`、`--nstruct <= 0`、`--trials <= 0`、无效 `--device-id`、非法 `--seed`、不可写 `--work`），逐项确认报错清晰且退出码非零。
2. **几何敏感位点专项冒烟测试**：
   - 在全量 50 突变运行前，优先对高风险子集执行小批量真机校验：脯氨酸突变 `A:F486P` / `A:N481P`、二硫键 `A:C480–A:C488` 邻域、高方差位点 `A:G476R`、甘氨酸位点 `A:G485T` 及链断裂邻域 `A:S514Q`，核验无 NaN/Inf、二硫键键长保持合理范围、残差 `<= 0.01`。
3. **全量端到端正式自验流程**：
   - 严格通过 `taskset -c 0-23` 启动 `run_flexddg_npu.py`，显式传入 `--device-id` 与全部官方固定输入（`bd623_mutations.csv`、`PDB`、`ddG-backrub.xml`、`--move-chain A`、`--nstruct 35`、`--trials 35000`、`--cpu-cores 24`、`--seed 111111`），外部计时与内部 `run_summary.json` 双口径核对 `wall_time_s`；
   - 执行官方未修改验收脚本：
     ```bash
     python3 test_cases/verify_bd623.py \
       --pred     ./out_bd623/results.csv \
       --ref      test_cases/bd623_reference.csv \
       --summary  ./out_bd623/run_summary.json \
       --json-out ./out_bd623/verify_report.json
     ```
   - 固化版本与证据：在 `run_summary.json` 与自测报告中完整记录 CPU 型号与限核方式、Ascend 950PR 卡号、CANN 版本、驱动与固件版本、代码最终 commit SHA、完整执行命令、阶段耗时剖析与 `verify_report.json`。

## 兼容性分析与风险应对

- **兼容性分析**：本任务为独立端到端生命科学加速模块（非替换现有 `aclnn` / BLAS 公共 ABI），不破坏任何既有 CANN 算子库接口；输出 CSV 与基线脚本 `06b_ddG_flex_antigen.py` 的 22 列定义严格兼容，可直接接入官方 `verify_bd623.py` 验收流水线；CANN 接口兼容 9.0.0 及以上版本，设备核数与 UB 容量均通过运行时平台接口查询。
- **关键技术风险与应对预案**：

| 潜在风险 | 影响说明 | 缓解与降级预案 |
| --- | --- | --- |
| `fa_dun` / `hbond` 细微数值偏差导致大效应组相对误差逼近 5% | 35,000 步 MC 与 L-BFGS 对力场势面斜率敏感 | 以 Rosetta `2026.29` 原生参数导出为唯一真源；建立 L1/L2 自动逐项对账门槛（`< 1e-4 REU` / `< 1e-5` 梯度）；终态四态打分由 CPU 官方 Rosetta 精确复核 |
| `G476R` / `F486P` 等位点在 L-BFGS 中收敛步数偏长造成批内拖尾 | 同批中其余轨迹空等导致 NPU 利用率下降 | 引入 LPT 跨突变调度 + L-BFGS 活跃轨迹流压缩（Stream Compaction），已收敛轨迹立即退出计算 |
| 全量 1,750 轨迹在单一固定微批下 HBM 或近邻表溢出 | 大 `neighbor_shell` 位点近邻对超预期 | 邻域表与 IG 缓冲区按 `BD-623` 最大 shell 留足安全倍数并支持 Host 动态自适应缩小微批 $B$（如 $256 \to 128 \to 64$） |
| 24 核限核下 CPU 四态打分或参数导出耗时超出预期 | 削弱端到端加速比 | 将参数导出二进制持久化缓存于 `--work` 同次运行内共享；若四态 CPU 复核成为瓶颈，在 L1 单点误差验证 `< 1e-4 REU` 前提下切换至 NPU 四态打分并仅对残差异常样本触发 CPU 复核 |

---

# 参考资料

1. Rosetta 开源代码仓：https://github.com/RosettaCommons/rosetta
2. Flex ddG 官方协议教程：https://github.com/Kortemme-Lab/flex_ddG_tutorial
3. Flex ddG 原始论文（Barlow et al., *J. Phys. Chem. B* 2018）：https://pubs.acs.org/doi/10.1021/acs.jpcb.7b11367
4. CANN 社区任务设计文档模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
5. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
6. CANN 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html
7. 【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39
