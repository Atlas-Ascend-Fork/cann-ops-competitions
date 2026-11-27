# Rosetta Flex ddG 端到端异构加速设计文档（24 核 CPU + Ascend 950PR）

| 项目 | 内容 |
| --- | --- |
| 任务编号 | 11 月社区任务-生命科学 flexddg ΔΔG 计算加速（24核CPU+950) |
| 目标硬件 | 24 核 CPU（`taskset -c 0-23` 严格限核）+ 昇腾 950PR 单卡（arch35） |
| 软件版本 | CANN 9.0.0 及以上（Ascend C / CANN Kernel 直调 + C++/Python 异构调度） |
| 对标基线 | 开源 Rosetta `2026.29`/`80a0635615` + `flex_ddG_tutorial`（`ddG-backrub.xml` 原件）；48 核 CPU 实测参考输出 `bd623_reference.csv` |
| 性能目标 | BD-623 全部 50 突变端到端墙钟 ≤ 5,565.3 s（≥ 10.0× vs 48 核基线 55,653.4 s） |
| 精度目标 | 大效应样本（15 个）相对误差中位数 < 5%；小效应样本（35 个）全部 \|ΔΔG 偏差\| < 1.0·std_ref；50/50 完整性自洽 |
| 工程模式 | 整模块端到端异构工程：Host CPU（C++/Python 调度）+ NPU（算子下沉 + 批量系综 + 异步流水线） |
| 文档状态 | 设计定稿；性能/精度按 §3.2/§3.3 在 950PR + 24 核实测后回填自测报告 |

# 1. 需求背景（required）

## 1.1 需求来源

CANN 社区任务 2026，11 月社区任务-生命科学 flexddg ΔΔG 计算加速（24核CPU+950)。任务书：
`Rosetta_Flex_ddG_Atlas950_task_doc.md`。贡献者：`zhanghua145`。

Rosetta Flex ddG 是蛋白质-蛋白质结合界面突变自由能变化（ΔΔG）预测的主流蒙特卡洛系综协议：
在 48 核 CPU 上运行 BD-623（PDB `7E7Y`，420 残基）50 个抗原侧单点突变
（每突变 35 轨迹 × 35,000 步 backrub，共 1,750 条轨迹）累计 **742.05 CPU·h**、
端到端墙钟 **55,653.4 s（15.46 h）**。本任务在 24 核 + 950PR 单卡上达到其 **1/10 耗时**。

| 材料 | 用途 |
| --- | --- |
| 任务书 §2.1–§2.4 | 协议阶段、三条红线、CLI 接口与 22 列输出规范 |
| `test_cases/ddG-backrub.xml` 原件 + `baseline/` 脚本 | 功能与口径基线（`06b_ddG_flex_antigen.py` 参数对应） |
| `test_cases/bd623_mutations.csv`（50 突变）+ `bd623_reference.csv`（48 核参考） | 正向全量输入与精度真值 |
| `test_cases/verify_bd623.py` + README | 完整性/分层精度/端到端性能一键验收 |
| Rosetta `2026.29` / `80a0635615` | 力场（talaris2014）与协议语义参照 |

## 1.2 背景介绍

### 热点结构分析（48 核基线实测口径）

| 协议阶段 | 占比（核时） | 计算特征 | 异构机会 |
| --- | --- | --- | --- |
| Backrub/Sidechain MC 采样（35×35,000 步） | ~75% | 每步：小段刚体旋转提案 + 局域能量增量重估 + Metropolis 判决；轨迹间完全独立 | **批量系综下沉**：多轨迹 bubble 并行填满向量/Cube 阵列 |
| WT/MUT repack + 约束 L-BFGS 最小化（后处理，每轨迹 2×2 态） | ~20% | 侧链组合优化（MultiCoolAnnealer）+ 迭代梯度最小化 | 批量 rotamer 打分矩阵 + 迭代最小化 kernel 化 |
| 结构准备/初始最小化/四态打分汇总 | ~5% | 一次性 + 逐轨迹打分求和 | 打分核（7 分子间项）批量化 |

关键结构性事实：**1,750 条轨迹相互独立**（每轨迹独立 seed 的 MC 链），
50 个突变共享同一 PDB 骨架与邻域图——轨迹级并行度 1,750 × 步级细粒度，
天然适配 NPU 大规模数据并行；每轨迹活跃集（8 Å bubble + neighbor_shell）仅
数十残基（~10²–10³ 原子对量级），单轨迹单步计算量小——**必须跨轨迹组批**
才能填满 950PR AIV/AIC 阵列，这是本设计的核心出发点。

# 2. 需求分析（required）

## 2.1 接口定义（任务书 §2.3）

```bash
taskset -c 0-23 python3 run_flexddg_npu.py \
  --selected <mutations_csv> --pdb-dir <pdb_dir> --xml <protocol_xml> \
  --out <results_csv> --summary <run_summary_json> --work <work_dir> \
  --move-chain A --nstruct 35 --trials 35000 \
  --cpu-cores 24 --device-id 0 --seed <int>
```

与基线脚本 `06b_ddG_flex_antigen.py` 参数一一对应（`--nproc`→`--cpu-cores`，
新增 `--device-id`/`--seed`/`--summary`）；输入校验（文件缺失/缺列/空表/
`wt_aa` 与 PDB 残基不一致/`--nstruct<=0`/`--cpu-cores>24` 等负向场景）均
返回非零退出码。

主输出 `results.csv` 22 列（§2.4.2）：6 输入回显 + `ddG_mean/std/all(REU)` +
7 分子间能量项双差均值 + `ddG_residual`（\|residual\| ≤ 0.01）+
`ddG_gam_mean/std(kcal/mol)`（ZEMu GAM 重加权）+ `nstruct_done==35` +
`backrub_trials==35000` + `cpu_time(s)`；
`run_summary.json` 必含 `wall_time_s/mutations_completed/cpu_cores_limit/
npu_device/cann_version/seed/git_commit/command`。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 完整等价覆盖 `ddG-backrub.xml` 四阶段（准备/初始最小化 → MC 采样 → WT/MUT repack+最小化 → 四态打分重加权） | §2.1 | 流水线阶段映射（§3.2.1） |
| R02 | 三条红线：`talaris2014` 力场不变；`nstruct=35`/`trials=35000`/8 Å 邻域/最小化收敛阈值不调低；不删减核心阶段 | §2.1 | 数值一致性保障（§3.2.4） |
| R03 | ΔΔG 四态公式 + 7 项能量 + GAM 重加权 + residual ≤ 0.01 | §2.1/§2.4.2 | 打分汇总模块（§3.2.1 S5） |
| R04 | 大效应中位 rel_err < 5%、小效应全部 < 1σ_ref、50/50 完整性 | §3.2 | 统计精度策略（§3.2.4） |
| R05 | 端到端墙钟 ≤ 5,565.3 s（含输入解析/算子加载/落盘，不含一次安装编译） | §3.3 | 性能预算反算（§3.2.5） |
| R06 | 24 核严格限核（taskset/cgroup），不得超核 | §2.2/§7 | 调度器核绑定（§3.2.1） |
| R07 | `--seed` 可复现；MC 随机系综不要求逐位确定 | §2.1 | 种子谱系（§3.2.4） |
| R08 | 交付目录结构 `rosetta_flex_ddg/{README,build.sh,env,pipeline,kernels,tests}` | §5 | 工程布局（§3.2.0） |

# 3. 详细设计（required）

## 3.1 算子分析（协议 → 异构计算映射）

### 3.1.1 数学核心

- **MC 步（内层 35,000 ×）**：backrub 提案（邻段刚体旋转，pivot 在 neighbor_shell）→
  局域能量增量 ΔE（talaris2014 全项：fa_atr/fa_rep（LJ 12-6）、fa_sol（EFA）、
  fa_elec（距离依赖介电 Coulomb）、hbond_*（取向依赖氢键）等）→
  Metropolis 判决 `min(1, exp(−ΔE/kT))`，kT = 1.2；
- **repack（MultiCoolAnnealer, states=6）**：bubble 内侧链组合优化（rotamer 打分矩阵）；
- **L-BFGS（armijo nonmonotone, tol=1e-6, max_iter=5000）**：CA 距离约束下最小化
  （`coord_dev=0.5`, `max_distance=9`）；
- **四态 ΔΔG**：`ΔΔG = (E_bound_mut + E_unbound_wt) − (E_bound_wt + E_unbound_mut)`，
  7 分子间项双差 + GAM：`gam(x)=−exp(a)+2exp(a)/(1+exp(−x·exp(b)))` 逐项重加权求和。

### 3.1.2 计算粒度与批量形态

每轨迹活跃集为 8 Å bubble（数十残基/数百原子，随位点差异 ±10%）；
系统总量 = 50 突变 × 35 轨迹 = 1,750 个独立 MC 链 × 35,000 步 ≈ 6.125×10⁷ 步。
单步 ΔE 仅涉及移动段邻域原子对（~10²–10³ 对）——单轨迹喂不饱 NPU 阵列，
**跨轨迹组批（同 batch 内 N 条轨迹同步推进，kernel 一次发射处理 N 个 bubble 的
同类型计算）** 是达到 10× 的必要条件。

## 3.2 算子实现

### 3.2.0 工程布局（任务书 §5）

```text
rosetta_flex_ddg/
├── README.md            # 环境依赖、编译构建与一键复现说明
├── build.sh + CMakeLists.txt
├── env/                 # CANN/依赖激活脚本
├── pipeline/            # run_flexddg_npu.py 入口、输入校验、CPU/NPU 调度、结果汇总
├── kernels/             # NPU 算子（Ascend C）与原生扩展
│   ├── mc_backrub_batch_kernel     # 批量刚体旋转提案 + 原子坐标更新
│   ├── energy_pairwise_batch_kernel# 批量 LJ/ fa_sol/ fa_elec 原子对能量（AIC+AIV）
│   ├── energy_hbond_batch_kernel   # 批量取向依赖氢键项
│   ├── metropolis_batch_kernel     # 批量 Metropolis 判决 + 状态回滚
│   ├── rotamer_score_batch_kernel  # repack 侧链组合打分矩阵
│   └── lbfgs_batch_kernel          # 批量迭代最小化步
└── tests/               # 单元测试与端到端自验脚本（verify_bd623.py 接入）
```

### 3.2.2 Host/NPU 分工与批量系综执行模型

- **Host（24 核，C++ 调度核 + Python 入口）**：
  1 核主调度（轨迹分派/GT 表维护/断点续跑）、~8 核协议逻辑（邻域图构建、
  rotamer 库展开、结果汇总/GAM 重加权/CSV 落盘）、~15 核喂料/回写
  （bubble 打包 buffer 与 NPU 结果 D2H 双缓冲）；
- **NPU（950PR 单卡）批量系综**：调度器把 1,750 条轨迹切成 **batch = 64~128 条**
  的批次（同批共享 kernel 发射），每批一条「MC 主循环驻留 kernel」驱动：
  ```
  for step in 1..35000:                     # 单次 launch 内驻留循环（避免逐步发射）
      mc_backrub_batch    (N bubble 坐标更新)
      energy_pairwise/hbond_batch (N bubble ΔE 重估，增量式：只算移动段邻域对)
      metropolis_batch    (N 判决 + 坐标回滚)
      （每 ~1024 步轮换一次 batch，长短期轨迹错峰填谷）
  ```
  增量 ΔE 的原子对列表由 Host 预生成邻域图（8 Å）+ kernel 内段索引计算；
  不规则原子对用压缩平面（CSR 式 offset + 平面化对表）映射到 AIV 向量 pass；
- **异步流水线**：CPU 打包下一批 bubble 与 NPU 计算当前批重叠（双缓冲）；
  post-processing（repack + L-BFGS）与后续批次 MC 并行错峰——把 ~20% 的
  后处理核时藏进 MC 流水线；
- **repack/L-BFGS 批量化**：35 轨迹 × 4 态 repack 与最小化同为跨轨迹组批
  kernel（rotamer 打分矩阵批量构造、MultiCoolAnnealer 退火序 CPU 生成 +
  打分 kernel 批量评估、L-BFGS 迭代步批量前向/回退判定）；
- **四态打分**：每轨迹终点构象 4 态打分（bound/unbound × wt/mut）复用
  energy_batch kernel（unbound 态按 `chainstomove=A` 平移后重打分子间项）。

### 3.2.3 调用与执行流程

```text
run_flexddg_npu.py ─输入校验(非零退出)→ 环境探测(CANN/设备) → 邻域图/rotamer 库预构(CPU)
  → 批量系综调度: [MC 35,000 步驻留 kernel × 1,750 轨迹(64~128/批)]
                  ‖ 异步: bubble 打包/回写双缓冲 + repack/L-BFGS 错峰
  → 每轨迹 4 态打分 → ΔΔG/7 项/GAM 汇总 → results.csv(22 列) + run_summary.json
  → verify_bd623.py 一键校验
```

### 3.2.4 talaris2014 数值一致性与统计精度（R02/R04）

1. **力场常数与公式逐项对齐**：fa_atr/fa_rep（LJ 12-6 + 平滑截断）、fa_sol（EFA）、
   fa_elec（距离依赖介电 4r）、hbond（取向依赖 SSRDE）等系数与函数形式以 Rosetta
   `2026.29` 源码为唯一参照，构建期从 Rosetta 参数文件**读取并冻结**（不手抄常量表）；
2. **精度策略**：能量内核 FP32 计算、**按原子对 FP64 累加**收窄 FP32（对齐 golden
   统计口径的浮点域），MC 累计 score 与四态求和在 FP64 域完成——保证
   `ddG_residual = ddG_mean − Σ7项` ≤ 0.01 的自洽性（求和顺序固定）；
3. **统计精度而非逐位一致**：MC 为随机系综（任务书 §2.1 确定性条款）——以
   `--seed` 谱系（任务 seed → 突变号 → 轨迹号 → 步号，可复现）保证单次运行可复跑；
   验收以分层统计判据：大效应组 rel_err 中位数 < 5%（15 样本）、小效应组全部
   < 1σ_ref（35 样本）、50/50 完整性（nstruct_done=35、trials=35000、35 有限值、
   无 NaN/Inf）；
4. **红线守护**：nstruct/trials/8 Å/收敛阈值作为运行期断言（低于即拒绝启动），
   协议 XML 原件解析校验（`ddG-backrub.xml` 哈希比对）。

### 3.2.5 性能预算反算（R05）

目标：50 突变端到端 ≤ 5,565.3 s。预算分解（设计目标，实测回填）：

| 阶段 | 预算(s) | 依据 |
| --- | --- | --- |
| 输入解析/邻域图/rotamer 预构 + 算子加载 | ≤ 180 | 一次性（CPU 24 核并行 + 图捕获） |
| MC 采样（6.125×10⁷ 步） | ≤ 4,300 | 系统级吞吐 ≥ 1.43×10⁴ 步/s：batch=96 × 驻留 kernel 单步 ~6.7ms（96 bubble 增量 ΔE ~10³ 对/轨迹 → ~10⁵ 对/步，AIV 向量 pass + AIC 阵列利用率 ≥ 60%） |
| repack/L-BFGS 后处理（1,750×4 态） | ≤ 800（与 MC 重叠后显式 ≤ 500） | 批量 rotamer 矩阵 + 迭代最小化 kernel；流水线错峰 |
| 四态打分/汇总/落盘 | ≤ 120 | 打分核批量化 + 汇总 CPU |

关键杠杆：① 轨迹级组批（填满阵列）② 增量 ΔE（只算移动段邻域对）③ 驻留循环
（免逐步发射 ~30–50us/步的 launch 开销）④ MC 与后处理错峰。
基线参照：48 核 CPU 单步均摊 ~1.53s/步·轨迹（2,671,363 s ÷ 1.75×10⁷ 步·核时折算），
NPU batch 96 单步 6.7ms ≈ 等效 2.2×10⁴ 步/s（> 10× 基线的必要斜率）。

### 3.2.6 内存与 workspace

- bubble 批量 buffer：96 × (坐标 3F + 邻域对表 + 状态) ≈ 96×~1MB 级，GM 驻留 +
  UB 分窗 ping-pong；rotamer 打分矩阵按轨迹分窗复用（GM ~百 MB 级内）；
- `.db3` 轨迹中间文件不入库（任务书 §7）；work 目录仅保留汇总与日志。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 昇腾 950PR 单卡 + 24 核 CPU（taskset -c 0-23） | √ |

CANN 9.0.0 及以上；版本与驱动、代码 commit、完整命令记录进 `run_summary.json` 与自测报告。

## 3.4 约束限制

- 不得超核/跨机（24 核绑定真实生效）；三条红线（力场/预算/协议阶段）运行期守护；
- 输出 22 列规范与 residual/有限值校验（违例非零退出）；
- 计时口径：从命令启动读输入至 `results.csv` 落盘的完整墙钟（含前置阶段，
  不含一次性安装编译，任务书 §3.3 第 3 条）。

# 4. 可维可测分析

## 4.1 精度/性能标准

- 精度（任务书 §3.2 分层统计）：大效应（\|ΔΔG_ref\| > 1.0 REU，15 个）
  rel_err 中位数 < 0.05；小效应（≤ 1.0 REU，35 个）全部偏差 < 1.0·std_ref；
  完整性 50/50（nstruct_done=35、backrub_trials=35000、ddG_all 35 有限值、
  \|residual\| ≤ 0.01、无 NaN/Inf）；verify_bd623.py 一键校验并输出逐突变
  误差/σ 与 Pearson r；
- 性能（任务书 §3.3）：`wall_time_s ≤ 5,565.3 s`（≥ 10.0×），完整墙钟口径。

## 4.2 测试方案

1. **正向全量**：`bd623_mutations.csv` 50 真实突变（含 15 大效应/35 小效应/
   4 个脯氨酸几何敏感位点 N440P/N481P/N501P/F486P），不随机造数；
2. **负向边界**：缺失 PDB/缺失 CSV 必选列/`wt_aa` 不匹配/`--cpu-cores>24`/
   `--nstruct<=0`/设备越界等——均明确报错非零退出（任务书 §3.5）；
3. **单元级**：各能量项 kernel 对拍 Rosetta 单步打分（同构象同参数，
   逐项误差 < 1e-3 相对量级）；MC/repack/L-BFGS 模块级对拍（同 seed 短程轨迹
   统计量）；增量 ΔE 对拍全量重估；
4. **端到端**：verify_bd623.py 验收报告 + 阶段 profiling 拆分（MC/后处理/汇总
   各自墙钟）随自测报告交付。

## 4.3 兼容性分析

- 与开源 Rosetta 共存（力场参数只读引用，不修改 Rosetta 源）；
- 模块独立工程（PR 目标仓由社区维护者指定，按 §5 目录结构）；无其他产品线耦合；
- `--seed` 支持复现与排查（任务书 §2.1 第 4 条建议项）。

# 5. 参考实现与验收风险

| 项 | 结论 |
| --- | --- |
| 实现完成度 | 设计定稿；批量系综驻留 kernel/增量 ΔE/异步流水线为技术主体，按 §3.2.5 预算分阶段落地（先单突变 35 轨迹竖切跑通，再全量 50 突变） |
| 精度 | 分层统计判据以 `bd623_reference.csv` 为真值；能量项逐项单元对拍先行，防止系统性偏移（Pearson r 监控） |
| 性能 | 10× 目标依赖三个杠杆全生效（组批/增量/驻留）；风险集中在不规则邻域对的阵列利用率，以 CSR 平面化 + batch 自适应（64~128）兜底 |
| 任务书对齐项 | ①CLI 与参数序列对齐 §2.3；②22 列输出与 residual 自洽；③三红线运行期守护；④24 核绑定真实生效；⑤墙钟口径含前置阶段；⑥大文件不入库 |
| 风险 | 脯氨酸突变等几何敏感位点的统计涨落（σ 口径吸收）；repack 退火序批量化后的系综等价性（同 seed 对拍退火轨迹统计量）；NPU 驻留循环的栈/状态管理（固定槽 + 双缓冲） |
