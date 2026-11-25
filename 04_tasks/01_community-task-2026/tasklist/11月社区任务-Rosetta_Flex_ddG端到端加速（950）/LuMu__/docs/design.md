# Rosetta Flex ddG 端到端异构加速模块设计文档（Ascend 950PR + 24 核 CPU）

| 项 | 内容 |
| --- | --- |
| 任务名称 | 【11月社区任务】Rosetta Flex ddG 端到端加速开发（950） |
| 任务书 | `Rosetta_Flex_ddG_Atlas950_task_doc.md`（随包） |
| 目标硬件 | Ascend 950PR（单卡，HBM 128 GB）+ 24 核 CPU（`taskset -c 0-23` 严格限核） |
| 软件栈 | CANN 9.1.0（≥9.0.0 满足）、Triton-Ascend 3.2.2、torch_npu 2.7.1.post8、Python 3.11 |
| 对标基线 | Rosetta 2026.29（commit `80a0635615`）+ Kortemme-Lab `flex_ddG_tutorial` 官方 `ddG-backrub.xml` |
| 性能目标 | BD-623 全部 50 突变端到端墙钟 ≤ **5,565.3 s**（48 核基线 55,653.4 s 的 1/10，加速比 ≥ 10×） |
| 精度目标 | §3.2 分层统计验收：15 个大效应突变中位相对误差 < 5%；35 个小效应突变全部 \|Δ\| < std_ref；完整性/自洽性 50/50 通过 |

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026（生命科学方向）：基于 24 核 CPU + 昇腾 Ascend 950PR 单卡，使用 Ascend C / Triton Ascend / CANN 软件栈结合 C++/Python 异构调度，重实现 Rosetta Flex ddG 蛋白质-蛋白质结合界面突变自由能变化（ΔΔG）预测协议的端到端计算，完成设计、开发、测试全流程，并按社区流程提交设计文档与 PR。

## 背景介绍

### 协议语义（不可改变的官方口径）

Flex ddG（Barlow et al., *J. Phys. Chem. B* 2018）是主流的界面突变 ΔΔG 系综预测协议。对 BD-623（PDB 7E7Y，420 残基，链 A=抗原 / H、L=抗体）的每个抗原侧单点突变，官方 `ddG-backrub.xml` 定义的计算流程为：

1. **结构准备与初始最小化**：加载全原子 PDB（含极性氢重建），校验 `wt_aa` 与结构一致；以突变位点为种子按 `cb8`（CB 原子 8 Å）构造 `bubble`，并向前后各扩 1 个残基得 `neighbor_shell`；施加 CA 距离约束（`coord_dev=0.5`、`max_distance=9`、flat-bottom）后执行 L-BFGS 约束最小化（`lbfgs_armijo_nonmonotone`，`tolerance=1e-6`，`max_iter=5000`，`abs_score_convergence_thresh=1.0`）。
2. **Backrub 蒙特卡洛采样**：每突变 `nstruct=35` 条独立轨迹，每条 `backrub_trials=35000` 步（`mc_kt=1.2`，支点限定在 `neighbor_shell`，`-ex1 -ex2`，`extrachi_cutoff=0`，`recover_low=0`）。
3. **WT/MUT 后处理**（每条轨迹终点执行 `finish_ddg_post_backrub`）：
   - WT 侧：`repack`（`neighbor_shell` 内重排，`UseMultiCoolAnnealer states=6`）→ 加约束 → L-BFGS 最小化 → 存为 bound WT 终态；
   - MUT 侧：回溯 backrub 终态 → `mutate`（按 resfile 定点突变并打包）→ 加约束 → L-BFGS 最小化；
   - `trajectory_stride = 35000`，即该后处理在轨迹结束时仅执行一次。
4. **四态接口双差与重加权**（`InterfaceDdGMover`，`chainstomove=A`）：
   ```text
   ΔΔG = (E_bound_mut + E_unbound_wt) − (E_bound_wt + E_unbound_mut)
   ```
   unbound 态将链 A 刚体移开至相互作用截断外并重排界面；输出原始双差 `ddG_mean(REU)`、7 个分子间能量项（`fa_atr / fa_rep / fa_sol / fa_elec / hbond_sc / hbond_bb_sc / hbond_lr_bb`）及 ZEMu GAM 重加权 `ddG_gam_mean(kcal/mol)`（参数取自官方 `analyze_flex_d01_dG.py`，随包 `flexddg_lib.py` 已内置）。

**三条不可触碰红线**：力场保持 `talaris2014`（等价 `-corrections::restore_talaris_behavior true`）；采样与收敛预算不得调低（`nstruct=35`、`trials=35000`、8 Å 邻域、`max_iter=5000`、`conv_thresh=1.0`）；不得删减任何核心阶段。

### 48 核 CPU 基线剖析（热点在哪里）

实测基线（2×Xeon 8360Y、48 进程、Rosetta 2026.29，随包 `bd623_reference.csv`）：

| 统计项 | 数值 |
| --- | ---: |
| 50 突变累计核时 | 2,671,363.3 s = 742.05 CPU·h |
| 单轨迹平均 CPU 耗时 | ≈ 1,526 s（0.42 CPU·h） |
| 48 核整批墙钟 | 55,653.4 s（15.46 h） |
| 总轨迹数 | 50 × 35 = 1,750 条，每条 35,000 backrub 步 |

按 Rosetta 实现的时间结构拆分单轨迹（以 1,526 s 计）：

| 阶段 | 占比（估） | 计算特征 |
| --- | ---: | --- |
| 35,000 步 backrub MC（段旋转 + 支点键几何优化 + 移动段支链 RotamerTrials + 增量/全量打分 + Metropolis 判定） | ≈ 85–90% | **标量串行循环 + 小规模分支计算**，单步 40 ms 级；批间无任何数据依赖 |
| 轨迹终点 2×（repack + 约束最小化） | ≈ 8–12% | 中等规模组合优化（bubble 12–15 残基 × 含 ex1/ex2 的 rotamer 组合）+ 全 pose 自由度 L-BFGS |
| 初始最小化、四态打分与 IO | < 2% | 一次性开销 |

**关键观察（本方案成立的基础）**：

1. **系综级数据并行度极高**：1,750 条轨迹在 MC 阶段完全相互独立，且每条步数固定为 35,000——所有轨迹共享同一“步数轴”，可按 (trial 索引 × 批内轨迹) 二维网格整批下发 NPU；
2. **能量计算是规则的近邻对计算**：`talaris2014` 主项（fa_atr/fa_rep/fa_sol/fa_elec/hbond/fa_pair）都是“原子对距离 → 查表/闭式公式 → 加权求和”，天然映射为距离掩码 + LUT + 分块归约的向量化内核；查表类项（fa_dun/rama/p_aa_pp/ref）按残基独立，可批量查表；
3. **单步触碰的原子集合很小**：backrub 单步只移动 2–4 个连续残基的骨架段及其支链，能量重算范围 ≈ 移动原子 × 截断内环境原子（数百 × ~10³ 量级），增量评分是确定性正确的优化而非近似；
4. **串行瓶颈在于单指令流的逐对计算与分支**，而非算法本身——这正是 NPU 向量/矩阵部件 + 批量化的收益空间；48 核基线的极限 ≈ 48 路轨迹级并行，而本方案在 NPU 上同时推进 B 路轨迹 × 批内原子对级并行，等效并行度高出 2–3 个数量级。

---

# 需求分析（required）

## 需求描述

交付一个端到端异构加速模块（PR 目录结构 `rosetta_flex_ddg/`），提供统一入口 `pipeline/run_flexddg_npu.py`，参数序列与 `test_cases/baseline/06b_ddG_flex_antigen.py` 对齐（`--selected/--pdb-dir/--xml/--out/--summary/--work/--move-chain/--nstruct/--trials/--cpu-cores/--device-id/--seed`），在 24 核 + 950PR 上完成 BD-623 全部 50 个突变，输出符合 §2.4.2 规范的 22 列 `results.csv` 与含 `wall_time_s/mutations_completed/cpu_cores_limit/npu_device/cann_version/seed/git_commit/command` 的 `run_summary.json`，并通过 `verify_bd623.py` 的精度与性能验收。

## 需求拆解

| # | 子需求 | 落点 |
| --- | --- | --- |
| R1 | 输入解析与校验（CSV 6 列、`wt_aa` 一致性、PDB/XML 存在性、参数合法性），非法即拒绝并非零退出 | `pipeline/input.py`（CPU） |
| R2 | `talaris2014` 力场参数导出为 NPU 常量（LJ、LK 溶剂化、电荷、氢键参数、Dunbrack rotamer 库、rama/p_aa_pp/ref 查表、GAM 参数），版本 pin 到 Rosetta `80a0635615` | `pipeline/params_export.py`（CPU，一次性） |
| R3 | 全原子构象张量化（极性氢重建、内部坐标/χ 角、批次内存布局、邻居表） | `kernels/conformation.py` |
| R4 | `talaris2014` 全能量项 NPU 内核（含 atom_pair_constraint） | `kernels/energy/` |
| R5 | Backrub MC 批量采样内核（段选取、CA-CA 轴旋转、支点键几何优化、移动段 RotamerTrials、Metropolis） | `kernels/backrub/` |
| R6 | 批量 repack（MultiCoolAnnealer×6 states 语义）与批量约束 L-BFGS 最小化 | `kernels/pack/`、`kernels/minimize/` |
| R7 | 四态计算（bound/unbound × WT/MUT）、7 项分解、GAM 重加权、汇总统计 | `pipeline/four_state.py` |
| R8 | Host 侧编排、批调度、异步流水线、24 核绑核与负载均衡 | `pipeline/runner.py` |
| R9 | 可复现随机数（`--seed` 派生每轨迹独立子流） | `pipeline/rng.py` |
| R10 | 单元对拍 / 集成冒烟 / 端到端验收三级测试 | `tests/` |

## 验收约束（红线，开发全程自检）

1. 力场 `talaris2014`，不换 REF2015、不删项；权重与 Rosetta database 完全一致；
2. `nstruct=35`、`trials=35000`、`bubble=cb8(8.0 Å)`、`max_iter=5000`、`conv_thresh=1.0` 原值直传，任何捷径（缩短步数、跳过最小化、缩小邻域）都判废；
3. 协议阶段全覆盖：初始约束最小化 → backrub MC → WT repack+min → MUT mutate+pack+min → 四态双差；
4. 全程 `taskset -c 0-23`，不得私启超核进程池；
5. 端到端计时含冷编译与参数导出等前置阶段；
6. 中间 `.db3`/轨迹目录不入库，仅提交 `results.csv / run_summary.json / verify_report.json / 日志`。

---

# 详细设计（required）

## 3.1 总体架构与 CPU/NPU 分工

```text
┌─────────────────────────── Host CPU（24 核，taskset 0-23）───────────────────────────┐
│  run_flexddg_npu.py (pipeline/runner.py)                                              │
│  ├─ 输入校验/参数导出/构象准备（一次性，~8–16 线程）                                    │
│  ├─ 批调度器：50 突变 → 批（每批 B 条轨迹，拓扑同构优先）                               │
│  ├─ 异步流水线：CPU 准备批 k+1（邻居表/rotamer 展开） ‖ NPU 计算批 k                  │
│  └─ 归并与统计：四态 → 22 列 CSV / run_summary.json / 断点续跑                         │
└──────────────┬───────────────────────────────────────────────────────────────────────┘
               │  HBM 常驻：批量构象张量 / 参数表 / 邻居表 / RNG 状态
┌──────────────▼─────────────────── Ascend 950PR（Triton-Ascend 内核）──────────────────┐
│  ① 能量内核：fa_atr/fa_rep(LJ-LUT) fa_sol(LK) fa_elec fa_pair hbond×4(几何+fade)      │
│              fa_dun/rama/p_aa_pp/ref(查表) atom_pair_constraint(flat-bottom)          │
│  ② Backrub MC 常驻循环内核：B 路轨迹 × 35,000 步（trial 步内 fused，状态不出 HBM）    │
│  ③ 批量 repack：rotamer-背景/rotamer-rotamer 能量矩阵 + 批量 MultiCool 退火(6 states) │
│  ④ 批量 L-BFGS(Armijo nonmonotone)：约束最小化，解析梯度向量化                        │
│  ⑤ 四态：刚体移链 + 界面 repack + 打分 → ΔΔG 双差                                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**分工原则**：CPU 只做低频、不规则、串行敏感的工作（解析、校验、参数导出、批次打包、统计与 IO）；一切高频、规则、可批量的数值工作（能量、MC、pack、minimize）全部下沉 NPU。CPU 侧 worker 数量可配（默认 12 线程做批准备），全部约束在 24 核亲和性内。

## 3.2 批量构象数据模型（内存布局与批策略）

- **构象张量**：`coords[B, N_max, 3]`（fp32，N 按批内拓扑 pad + mask）、`atom_type[B, N_max]`（int16 → 参数表行索引）、`res_type[B, R_max]`、`chi[B, R_max, 4]`；同一突变的 35 条轨迹拓扑完全一致，跨突变拼批时按 `N_max` pad。
- **批策略**：`B = 35 × k`（k=5→B=175 起步，内存与占用实测后上调，目标 B=350–700）。1,750 条轨迹分 5–10 批；批内所有轨迹共享 trial 步数轴（35,000），MC 内核按 `[B]` 维独立推进。
- **内存预算**（B=350）：坐标 ~350×9k×3×4B ≈ 38 MB；邻居表（9 Å+skin，~每原子 60 邻居）≈ 350×9k×60×8B ≈ 1.5 GB；rotamer 展开按批惰性分配。HBM 128 GB 充裕，瓶颈在算力不在容量。
- **邻居表**：Verlet 式（cutoff 9.0 Å + skin 1.0 Å），批内重建采用“位移阈值触发”（单轨迹位移 > skin/2 才重建该轨迹），重建本身也是 NPU 批量 kernel（cell list + 半径筛选）。
- **极性氢重建**：`talaris2014` 为全原子力场（重原子 + 极性氢），输入 PDB 无氢；按残基拓扑模板在加载期补齐（CPU，一次性），氢键项与 fa_sol 均以该原子集为准——与 Rosetta `-in:file:fullatom` 行为对齐。

## 3.3 talaris2014 参数导出与数值一致性保障（交付重点）

### 参数导出管线

从 Rosetta `80a0635615` 源码树（GitHub，无需编译，只读 database/源码常量）导出为带 SHA256 快照的 `.npz` 常量包：

| 能量项 | 参数来源（以 2026.29 树内实际路径为准，导出脚本带路径探测） |
| --- | --- |
| 权重 | `database/scoring/weights/talaris2014.wts`（含 restore_talaris_behavior 等价修正） |
| fa_atr/fa_rep | LJ 半径/ε：fa_standard 原子类型属性表；截断/线性化常数取 Rosetta 源码（`fa_max_dis=9` 于 `fa_talaris2014_cst`） |
| fa_sol | Lazaridis–Karpus 逐原子 λ/ΔG（原子属性表） |
| fa_elec | 逐原子电荷 + 介电常数设定（talaris 口径，源码核对） |
| fa_pair | 残基对统计势矩阵（`database/scoring/score_functions/PairEPotential/`） |
| hbond×4 | `database/scoring/score_functions/hbonds/` 参数（AHdist/平面性权重、fade 函数形状，talaris 不用新 fade 的口径以源码为准） |
| fa_dun | Dunbrack bb-dependent rotamer 库（含 ex1/ex2 额外 χ 细分与 `extrachi_cutoff=0` 语义） |
| rama/p_aa_pp/ref | φ/ψ 查表与残基参考能量（talaris2014 对应版本） |
| GAM | ZEMU 七项 (a,b) 参数（随包 `flexddg_lib.py` 原值） |

### 数值一致性策略（三层）

1. **单点对拍（golden.npz）**：用官方 Rosetta 2026.29 生成 20–50 个构象的逐项能量 golden 数据（每项 + total，double 精度文本导出，规避浮点写读差异），本模块对同一构象逐项对拍，容差 `|ΔE_term| ≤ 0.01 REU`（BD-623 规模）。golden 生成在自测环境完成一次并入库（小文件）。退路：若 golden 环境受限，以 `bd623_reference.csv` 的 35×50 轨迹级分布作统计基准（均值/方差逐突变对齐），并以手算物理公式单测兜底。
2. **计算精度**：pair 级 fp32 计算 + **分块树形归约**（块内顺序确定，块间树形），最终项级累加在 fp64（torch_npu fp64 支持性于 M1 实测；不支持则块级 fp32 归约 + 末级 CPU fp64 累加）。查表项（rama/dun/p_aa_pp/hbond fade）保持与 Rosetta 相同表分辨率与插值方式（线性插值），避免系统性插值偏差。
3. **统计一致性（验收口径）**：MC 系综不要求 bit 复刻，但要求分布一致——自测指标：批内 35 条轨迹 `ddG_all` 的均值/样本标准差与参考逐突变对齐（大效应中位 rel err <5%，小效应 |Δ|<std_ref），并以接受率、能量轨迹平稳性作过程监控。若出现系统偏移，按“参数 → 单项 → 协议顺序”三级定位（`ddG_residual` 列 + 分项能量列天然构成诊断面）。

## 3.4 NPU 能量内核设计（逐项）

统一模式：`邻居表 → gather 坐标 → 距离 → 掩码/LUT → 加权累积 → 树形归约`，单 pass fused kernel，中间量不出 UB。

| 项 | 内核要点 |
| --- | --- |
| fa_atr/fa_rep | LJ 12-6 拆分：r ≥ 截断置零；atr 侧截断附近线性化处理与 Rosetta 逐字节对拍；LUT（r² → 能量）预生成常量表，避免大量超越函数 |
| fa_sol | LK 遮蔽体积形式：pair 距离函数查表（`lk_...` 系数按原子类型），同样 LUT 化 |
| fa_elec | q_i·q_j/(ε·r)，短截断 + 距离依赖介电口径以源码导出常量为准 |
| fa_pair | 残基对二维表查表，与 packing 联动（仅 repack 侧更新） |
| hbond_sc / hbond_bb_sc / hbond_lr_bb / hbond_bb_bb | 先几何配对（D-H···A 距离/角度）再 fade 函数打分；bb 与 sc 按 donor/acceptor原子类别分项归集；`hbond_lr_bb` 按 seq_sep 阈值划入 lr；逐对几何在 kernel 内闭式计算，fade 参数 LUT |
| fa_dun | rotamer 概率 + 半经验偏差项（χ 偏离 well 的高斯惩罚），批量查 Dunbrack 表（双线性插值） |
| rama / p_aa_pp / ref | φ/ψ 网格表批量双线性插值；ref 逐残基常量 |
| pro_close | Pro 环闭合残差项，仅含 Pro 残基的构象激活 |
| atom_pair_constraint | CA-CA pairwise flat-bottom 二次势（`coord_dev=0.5`、`max_distance=9`、bound_width=0），只在最小化阶段启用（与 XML 的加/清约束节奏一致：MC 阶段无约束） |

**增量评分**：MC 循环内只重算“移动原子集合 × 其邻居原子集合”的交互（含移动集合内部），增量结果与静态部分分别累积；批内各轨迹的移动集合不同 → 用变长压缩索引（CSR 式 `[B]` 分段），kernel 以原子对总数 tile 切分保证负载均衡。

## 3.5 Backrub MC 批量采样内核（最大热点）

单个 trial 的语义（以 Rosetta `protocols/backrub.cc` 为核对基准，M1 阶段冻结行为差异）：

1. **段选取**：从 `neighbor_shell` 内按官方分布采连续支点段（典型 2–4 残基）；
2. **刚体旋转**：绕支点段两端 CA–CA 轴整段旋转，角度从官方分布采样；
3. **支点键几何优化**：对支点处两个 pseudo-键角/键长做小规模局部优化（闭式/牛顿，每 trial 只涉及 3 原子 × 2 支点）；
4. **移动段支链寻优**：对段内残基执行 RotamerTrials 式贪心重排（逐残基在其 rotamer 集合中取当前环境最优；含 ex1/ex2、`extrachi_cutoff=0`）；
5. **能量与 Metropolis**：增量重算（§3.4），`ΔE` 与 `exp(-ΔE/kt), kt=1.2` 比较，`recover_low=0` 保持链式推进。

**批量化实现**：

- 一个“trial 步”kernel 组处理 `[B]` 路轨迹的同一步（35,000 步共享步数轴）；
- **常驻循环优化**：优先实现 device-side 常驻循环（单次 launch 执行全部/大段 trial 循环，RNG 状态、坐标、邻居表、MC 状态全驻 HBM，逐步原位更新），消除 35,000 × 每 trial ~5–10 次 launch 的开销；Triton 内以 `for` 循环 + 编译期展开块实现，不满足时退化为“每 64/256 步一次 launch”的分段常驻；
- RNG：每轨迹独立 Philox 子流（seed ⊕ hash(mut_idx, traj_idx)），保证 `--seed` 可复现、批划分无关；
- 数据依赖控制流（接受/拒绝、段长采样）在批维上是独立分支，kernel 内以掩码/选择实现，无跨轨迹同步。

## 3.6 批量 repack（MultiCoolAnnealer 语义）与批量 L-BFGS

### repack（WT 侧 `repack` 与 MUT 侧 `mutate`，及 unbound 界面 repack）

- rotamer 集合：bubble（12–15 残基）+ `neighbor_shell` 打包范围，`-ex1 -ex2`、`extrachi_cutoff=0`（展开 χ1/χ2 半步细分），单残基 rotamer 数 4–80+；
- **两步法**：① 能量矩阵 kernel——rotamer-背景（`[B, Σrot, 邻居原子]` pairwise）与 rotamer-rotamer（`[B, Σrot, Σrot]` 压缩存储）；② 批量退火 kernel——每轨迹独立模拟退火（初始温度序列 × `states=6` 次 quench，保留历史最优），矢量化为 `[B, Σrot]` 状态翻转 + 能量差查表（矩阵已预计算，退火阶段零 pair 计算）；
- 与 Rosetta 的 annealer 调度不做 bit 级复刻（原版为标量随机序退火），以“收敛到同等能量水平的 packing + 端到端统计验收”为一致性判据（与任务书 §2.1.4 确定性口径一致）；偏差监控项：packing 终态能量分布 vs golden 参考分布。

### 约束 L-BFGS（`lbfgs_armijo_nonmonotone`）

- 自由度：全 pose bb+chi（`chi=1 bb=1`，与 MinMover 一致），批内 pad 对齐；
- 梯度：解析梯度向量化（LJ/LK/elec/constraint 的 pair 力 + 查表项插值梯度），hbbond 项几何梯度按参数化闭式导数实现；
- 线搜索：Armijo 非单调（`abs_score_convergence_thresh=1.0`、`tolerance=1e-6`、`max_iter=5000`），批内各轨迹独立判停（掩码收敛后该轨迹不再迭代，不拖累批）；
- 批内所有轨迹共享迭代轴直至全部收敛或达 5,000 上限——预算上界即协议上界，无捷径。

## 3.7 四态 ΔΔG 与 ZEMu GAM 重加权

- 四态生成完全对齐 `InterfaceDdGMover`：bound WT = 后处理后的 WT 终态；bound MUT = 后处理后的 MUT 终态；unbound = 将链 A 沿界面法向刚体平移至全部相互作用截断外（平移量与方式以源码核对为准），随后对界面残基 repack（界面定义按 `chainstomove=A` 的官方口径）；
- 每态以 `fa_talaris2014`（无约束版）终态打分，`ΔΔG = (bound_mut + unbound_wt) − (bound_wt + unbound_mut)`；
- 7 个分子间项 = 双差后非零项集合（分子内项四态精确抵消，输出 `ddG_residual` 必须在 0.01 内自检）；
- GAM：`gam(x) = -exp(a) + 2exp(a)/(1+exp(-x·exp(b)))`，七项 (a,b) 用 `flexddg_lib.py` 内置值，fp64 计算（量级小、无性能压力）；
- 输出与断点：逐轨迹结果落盘（轻量 `.json` 而非 `.db3`，不入库），支持按突变粒度续跑；`results.csv` 22 列、`run_summary.json` 字段严格对齐 §2.4。

## 3.8 Host 侧异步流水线与 24 核负载均衡（交付重点）

**流水线（批粒度双缓冲）**：

```text
批0: [CPU准备] → [NPU: MC 35000步 + 四态] → [CPU: 归并]
批1:      [CPU准备] ‖ [NPU计算批0] …
```

- CPU 侧准备（邻居表初建/rotamer 展开/批打包）与 NPU 计算以 `torch_npu` 事件重叠；批间共享 trial 轴，NPU 侧无空闲；
- 统计与 CSV 写出在收尾线程增量进行，不阻塞下一批。

**24 核分配（`taskset -c 0-23` 下）**：

| 角色 | 核数 | 说明 |
| --- | --- | --- |
| 主进程编排 + kernel launch | 2 | 单驱动线程 + 事件监控 |
| 批准备 worker 池 | 8–14 | 邻居表/rotamer 展开等 CPU 段（NumPy/OpenMP 受亲和性约束） |
| 统计/IO | 1–2 | 增量写 CSV/JSON |
| 兜底裕量 | 6+ | 避免与 NPU 驱动线程争核（实测调整） |

**负载均衡**：50 突变按 bubble 规模（12–15 残基差异 ±10% 对应基线 CPU 耗时差 ±10%）排序，大者先行（LPT 调度），消除长尾；批内 pad 开销用掩码抵消。冷启动编译（Triton JIT 缓存预热）与参数导出纳入计时并在 summary 中单列（§特别注意事项 5）。

## 3.9 性能预算与关键路径估算

总量预算 ≤ 5,565.3 s，分解（保守口径，M1 实测后回填）：

| 阶段 | 预算 | 依据 |
| --- | ---: | --- |
| 冷启动（JIT 预热 + 参数导出 + 构象准备） | ≤ 300 s | 一次性；Triton 缓存落盘后二次运行显著缩短 |
| MC 主循环（1,750 × 35,000 trials） | ≤ 3,600 s | 合计 6.125e7 trials；按增量 ~1.5e5 pair-eval/trial 计 ≈ 9.2e12 pair-eval，要求有效吞吐 ≥ 2.6 G pair-eval/s（fp32、fused、LUT 化；相当于数十 TFLOP/s，950PR 向量算力可达，M1 微基准门限 ≥ 3 G/s） |
| 四态后处理（repack×4 + min×2 + 打分） | ≤ 1,100 s | 能量矩阵 ~20M pair-eval/轨迹 × 矩阵复用 + L-BFGS 批内掩码收敛 |
| 归并/IO/自检 | ≤ 100 s | 增量写出 |
| 裕量 | ≥ 465 s | 长尾与重建邻居表开销 |

**里程碑实测门限（不达标即回退优化项）**：M1 批量能量内核 ≥3 G pair-eval/s；M2 常驻 MC 循环 ≥17k trials/s（B=350）；M3 单突变端到端 ≤110 s（35 轨迹）；M4 全量 ≤5,565 s。优化储备（按需启用）：fp16 距离 + fp32 累积、LUT 精度分级、邻居表压缩（uint32 索引）、rotamer 矩阵分块驻留 UB。

## 3.10 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（单卡） | √ |
| Atlas 800I/T A2 | 未适配（本任务口径外） |

## 3.11 约束限制

- 仅支持 CPU 24 核 + 单卡 950PR 配置；不支持多卡、跨机；
- 输入限定标准全原子 PDB（无 HETATM）+ 官方 `ddG-backrub.xml` 语义的参数面；非官方 XML 变体（如 atom4 邻域）不在验收范围；
- MC 为随机系综算法，不提供跨运行 bit 级确定性（提供 `--seed` 复现）；
- golden 单点对拍依赖一次性 Rosetta 2026.29 参考数据导出（小体积，随 tests 入库）。

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度：大效应 | 15 个 \|ddG_ref\|>1 REU 突变，`median(rel_err) < 5%` | 任务书 §3.2 |
| 精度：小效应 | 35 个小效应突变全部 \|Δ\| < 1.0·std_ref（max σ < 1.0） | 任务书 §3.2 |
| 精度：完整性/自洽 | 50/50 完成；nstruct_done=35；trials=35000；ddG_all 35 个有限值；\|residual\|≤0.01；无 NaN/Inf | 任务书 §3.2 |
| 一致性 | 逐项能量单点对拍 \|ΔE_term\| ≤ 0.01 REU（golden.npz，内部标准，防系统偏移）；GAM 路径 fp64 | 本设计 §3.3 |
| 性能 | `wall_time_s ≤ 5565.3`（含冷启动，summary 单列），24 核绑定真实生效 | 任务书 §3.3 |
| 验收工具 | `test_cases/verify_bd623.py` 一键判定（退出码即结论） | 随包 |

## 测试计划（三级）

1. **单元级（tests/unit）**：每能量项公式 vs 手算/物理极值；LUT 与闭式一致性；邻居表正确性（随机构象 vs 暴力全对）；增量评分 vs 全量重算（差 ≤ 1e-3 REU）；RNG 子流独立性；χ/骨架几何闭合精度。
2. **集成级（tests/integration）**：单突变（Y369C，参考中耗时最长）`nstruct=35 × trials=35000` 全流程，对照参考行（mean/std/七项分解）——最小可用的“分布一致性”证据；golden 逐项对拍脚本；断点续跑与非法输入退出码测试。
3. **端到端（tests/e2e）**：§README 的 24 核 + 950PR 一键脚本，产出 `results.csv / run_summary.json / verify_report.json`；性能 profiling 拆分（各阶段耗时、NPU 利用率，用 `npu-compute`/`msopprof` 采集算子级指标，PerformanceHints（950PR&950DT 系列）用于定位 Vector/Scalar/MTE 占比问题——工具契约引自 cannbot-knowledge）。

## 风险与对策

| 风险 | 等级 | 对策 |
| --- | --- | --- |
| 能量项实现细节与 Rosetta 有系统差（hbond fade、LK 截断、fa_elec 介电口径） | 高 | golden 逐项对拍前置（M1 门禁）；分项列诊断；参数快照版本 pin |
| MC 统计漂移（接受率/步长分布不匹配 → 均值偏移） | 高 | 步长/段长分布逐项核对 Rosetta 源码；接受率监控；先单突变对齐分布再放量 |
| 常驻循环 kernel 在 Triton-Ascend 上的控制流/UB 压力 | 中 | 分段常驻退路（每 64–256 步一次 launch）；关键 kernel 备选 Ascend C 直写 |
| repack 退火与原版 MultiCoolAnnealer 能量面不一致 | 中 | 6-state 语义保留 + 终态能量分布对齐 + 端到端统计兜底 |
| L-BFGS 全 pose 自由度批量化开销 | 中 | 掩码收敛（已收敛轨迹退出迭代）；梯度 fused kernel；上界即 5,000 迭代不破红线 |
| fp32 归约累加误差 | 低 | 树形归约 + fp64 末级；对拍容差 0.01 REU 验证 |
| 24 核竞争 NPU 驱动线程 | 低 | 亲和性实测调优，预留裕量核 |
| 性能不达 10× | 中 | §3.9 优化储备池（fp16 距离、LUT 分级、更大 B、批间流水加深）；里程碑门限提前暴露 |

## 兼容性分析

新交付模块（独立 `rosetta_flex_ddg/` 目录），无历史兼容负担。环境依赖：CANN ≥9.0.0（实测 9.1.0）、Triton-Ascend 3.2.x、torch_npu 2.7.1.post8、Python 3.11、numpy/scipy/pandas（自测环境已核齐）。不依赖 PyRosetta 运行时（参数经一次性导出快照固化）；Rosetta 源仅用于参数与 golden 生成阶段。

## 里程碑

| 里程碑 | 内容 | 出口判据 |
| --- | --- | --- |
| M0 | 环境确认、参数导出快照、构象张量化 + 邻居表 | golden 参数完整性校验通过 |
| M1 | 全能量项内核 + 单点对拍 | 逐项 \|ΔE\|≤0.01 REU；批量内核 ≥3 G pair-eval/s |
| M2 | Backrub MC 常驻批量内核 | ≥17k trials/s；接受率/能量轨迹分布合理 |
| M3 | repack + L-BFGS 批量内核 | 单轨迹后处理与参考能量水平一致 |
| M4 | 单突变端到端 | Y369C 35 轨迹 ≤110 s 且对齐参考分布 |
| M5 | 50 突变全量 + 验收产物 | `verify_bd623.py` 全绿 + wall_time ≤5,565.3 s |

## 交付件映射

| 任务书交付件 | 对应物 |
| --- | --- |
| 1 方案设计文档 | 本文档 |
| 2 端到端代码与复现脚本 | `rosetta_flex_ddg/{README.md, build.sh, env/, pipeline/, kernels/, tests/}`，一键脚本按 §README 自验流程 |
| 3 自测报告与验收产物 | `task_submission/{1 自验证步骤说明.md, 2.1/2.2 精度×2, 3.1/3.2 性能×2}` + `results.csv / run_summary.json / verify_report.json` + profiling 拆分 |
| 4 待验收代码地址 | 个人私仓（邀请 `Ascend-CANN`），commit 与 summary 中 `git_commit` 一致 |
