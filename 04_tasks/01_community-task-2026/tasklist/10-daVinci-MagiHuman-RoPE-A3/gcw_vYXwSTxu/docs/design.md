# 【CANN社区任务】daVinci-MagiHuman RoPE 优化（A3）最终实现设计

| 项目 | 内容 |
|---|---|
| 任务 | 10月社区任务—daVinci-MagiHuman RoPE 优化（A3） |
| 开发者 | gcw_vYXwSTxu |
| 文档版本 | v2.0，2026-10-15，最终封存实现的补充设计评审 |
| RFC | https://gitcode.com/cann/cann-recipes-infer/issues/429 |
| 实现 PR | https://gitcode.com/cann/cann-recipes-infer/pull/1059 |
| 最终实现 Commit | e07b8adaeda7524065709a1ad230f5ad0be78c19 |
| 原设计 PR | https://gitcode.com/cann/cann-ops-competitions/pull/2150 |
| 设计 PR | https://gitcode.com/cann/cann-ops-competitions/pull/2220 |

## 1. 需求背景

daVinci-MagiHuman 将文本、视频和音频 token 放入联合序列。RoPE 位于 Q/K 归一化之后、Attention 之前。本任务在 Atlas A3 的实际 base 推理链上复用 RoPE 系数，减少冗余准备，同时保留融合算子调用、partial rotary 尾部、数据所有权和显式回退语义。

本设计对应用户最终交付 ZIP 中的 owned-expanded 源码。原包 SHA256 为 `ddb5cd407023faf9f35a533afcb59ed3493666c6521b1a256461525a730f04ad`，其中 86 个文件已按原清单核对。首次发布整理格式；当前 Commit 还包括官方 CI 的格式、命名、导入顺序及许可证文件头修订，231 项现有 CPU 测试通过，逐文件来源映射随实现 PR 材料提供。

本设计对应最终实现；设计 PR 为 #2220，原设计 #2150 作为历史来源索引。

## 2. 需求分析与范围

### 2.1 功能范围

- 修改限定于 `models/davinci-magihuman/`。
- 复用既有 `torch_npu.npu_rotary_mul`，不新增独立 kernel、ACLNN 接口或 tiling 工程。
- 优化模型内部 RoPE 数据准备和请求生命周期，不重写 FlashAttention、线性层、并行策略、权重格式或公共模块。
- 支持路径直接使能，不依赖新增用户配置开关。
- 不支持的合法输入保留显式数学回退；非法输入和执行失败给出异常，不以宽泛捕获掩盖设备错误。

### 2.2 基线和源码身份

上游集成基线为 `f949a68f183439b1d63d740c378622d7eaadfd25`，实现分支 `davinci-rope-a3-final-20261015`，个人仓 `gcw_vYXwSTxu/cann-recipes-infer`。上述集成基线与原 A3 测量基线不同：原测试比较的是完整原模型目录快照与 owned-expanded 快照，不能仅凭早期顶层 recipe `bbd278fe...` 重建原实测模型树。

原模型上游固定提交为 `209209b7086eba2020c5439265221495a8357322`。发布源码包含最终 ZIP 已准备的模型依赖；导入依赖的文件数量和行数不作为 RoPE 优化收益。验收材料分别保留上游累计补丁、原封存快照补丁、原始基线源码包、原始候选源码包、发布源码包与逐文件哈希映射。

### 2.3 输入与数值语义

正式主链记录为 BF16、B=1、Q heads=40、K heads=8、D=128、R=96，S 取 3753 和 4102。half rotary 对前 R 维作旋转，R 以外尾部保持不变。

令 `u=x[..., :R]`，`rotate_half(u)=concat(-u[..., R/2:],u[..., :R/2])`，则旋转前缀为 `u*cos + rotate_half(u)*sin`，partial rotary 输出拼接原尾部。half 与 interleaved 的展开规则不同，不混用。形状、布局、设备、dtype 和旋转宽度由模型调用契约检查，不根据文件名推断融合命中。

## 3. 详细设计

### 3.1 调用链

```text
pipeline 请求作用域
  → 模型 Fourier 坐标与系数准备
  → 请求拥有的有界系数缓存
  → 每个 forward 获取 fresh contiguous (cos, sin)
  → Q/K 成对语义与所有权检查
  → 支持路径 npu_rotary_mul / 显式数学回退
  → 原 Attention 与后续生成链
  → finally 清理请求缓存
```

当前使用展开的普通 `(cos, sin)`，没有启用历史 compact/native 候选。当前请求缓存按精确内容和元数据匹配，最多两项；跨请求不共享进程级无界缓存。请求结束和异常退出均走清理路径。

### 3.2 所有权与缓存

缓存用于复用准备结果，不允许把缓存内部张量暴露为可修改的公共返回值。每次原生 preparation 都返回两份 fresh contiguous clone，包括命中、未命中和无请求缓存路径。这样下游原地修改某次返回系数时，不会污染缓存、输入或后续调用。

原完整 profile 记录为 2 miss / 62 hit；64 次 forward 的 128 次 clone 属于所有权契约，不能为了减少调用数而删除。请求 finally 后记录 entries=0、resident=0。缓存峰值与生成峰值分别记录，不把清理完成等同于整个进程长期零增长。

### 3.3 准备优化与派发

固定 shape 的 where 替代准备流程中的冗余动态路径，保持原数学顺序和 reference size 语义，处理非有限输入，异常或特殊模式保留原回退。paired Q/K 共享契约仅校验一次，但首个融合调用之前必须检查双方，不因先执行 Q 而遗漏 K 的错误。

`rotary.py` 保留 full/partial rotary、half/interleaved、非连续布局、梯度和张量子类相关的明确处理。支持边界由实际检查决定；不以 `except Exception: fallback` 掩盖 OOM、设备错误或未知执行失败。partial rotary 尾部和输入不可被静默改写。

### 3.4 文件与影响

| 文件 | 职责 |
|---|---|
| inference/model/dit/dit_module.py | 模型系数准备和 Q/K 调用集成 |
| inference/model/dit/rotary.py | RoPE 语义检查、融合派发、参考回退和所有权契约 |
| inference/model/dit/rotary_cache.py | 请求拥有的有界缓存、命中与清理 |
| inference/pipeline/pipeline.py | 请求作用域与异常清理边界 |
| tests/ | 原封存的数值、所有权、异常、兼容性与完整模型验证工具 |

配置仅保留可移植权重占位路径，不把私人服务器目录写成用户必须使用的路径。大型 profiling、完整报告 JSON、视频和权重不进入 Git；其获取渠道、字节数和 SHA256 作为 PR 附件材料提供。

## 4. 可维可测分析

### 4.1 原封存协议

两棵独立源码树使用相同环境、权重、prompt、seed、448×256、5 秒、32 steps、CFG=2 与 NoCache。每侧新进程执行 1 warmup + 5 formal + 2 quality；完整 profile 独立采集，不将 profile wall 时间当作正式性能。保留所有失败和补充口径，不通过挑选成功样本替换原记录。

### 4.2 原测试记录与发布核验

| 项目 | 原封存结果或当前提交核验 |
|---|---|
| 数值与所有权 | 两种实际 S，系数 bitexact、旋转前缀、尾部、poisoning 等记录保留 |
| 完整请求，不含媒体保存 | 中位节省 76.403930 ms；原基线极差 52.487780 ms |
| RoPE 累计设备耗时 | 下降 4.278592%；两侧各 5120 MAIN 融合 kernel |
| 含媒体保存的补充口径 | 原 FAIL，保留，不改为 PASS |
| 其他模式 | 七种非 base 模式完成生成，逐模式证据保留 |
| NPU 内存 | candidate 每请求峰值增加 3,017,728 bytes，如实记录 |
| Host 内存 | 两侧均有增长，不宣称长期零增长或已排除泄漏 |
| 三组质量 | A/B 视频字节相同，仅为技术证据；人工双盲仍待指定人员完成 |
| 本次发布核验 | 86 原始文件哈希匹配；当前发布 87 文件远端回读匹配；231 项 CPU 测试通过；发布补丁 Git tree 一致 |
| 本次改动范围 pre-commit | 73 变更路径全部 hooks 通过 |
| 全仓 pre-commit | 已执行；无关上游目录既有空白/脚本权限问题保留，未扩大本次修改范围 |

上述 A3 数据来自既有封存材料。本次补充设计与发布没有重新运行硬件测试，也不重新作算子达标判定。当前官方 CI 的链接和结果在实现 PR 中持续回填，不用局部检查代替官方 CI。

### 4.3 交付索引

- 实现 PR 关联 RFC，包含可访问的原始证据附件、匿名盲测样本、还原脚本与校验表。
- 个人仓材料分支 `davinci-rope-a3-materials-20261015` 的 `task_submission/` 放七个指定文件；不把这些报告塞入上游代码 PR。
- 最终验收 ZIP 提供四类交付材料、根目录七件套、源码与累计补丁、明确目录项、哈希校验和真实状态说明。

## 5. 当前待维护者与验收人确认的事项

1. 实测环境为 CANN 9.1，任务书参考环境为 CANN 9.0，需要维护者明确是否接受；未伪造已接受记录。
2. 两名指定验收人尚未确定，匿名样本和空白评价表已准备；真实双盲结论由指定人员评价后填写。
3. 官方 CI 与维护者 lgtm/approve 以实现 PR 实时状态为准；本设计提交不代表自动验收或自动合入。
4. 设计 PR：https://gitcode.com/cann/cann-ops-competitions/pull/2220。

## 6. 参考与 AI 辅助披露

- 任务书及 test_cases/README.md：任务包 SHA256 `81ae18903dee6f6166c2d36ac9e9c117b2cedf72462480b7ca23072857a5d260`。
- 目标仓：https://gitcode.com/cann/cann-recipes-infer
- RFC：https://gitcode.com/cann/cann-recipes-infer/issues/429
- 实现 PR：https://gitcode.com/cann/cann-recipes-infer/pull/1059
- 原设计历史：https://gitcode.com/cann/cann-ops-competitions/pull/2150

本次设计整理和提交由 OpenAI Codex（GPT-6）辅助完成。所有数值来自封存记录，未生成或虚构人工验收、环境接受或社区批准结论。
