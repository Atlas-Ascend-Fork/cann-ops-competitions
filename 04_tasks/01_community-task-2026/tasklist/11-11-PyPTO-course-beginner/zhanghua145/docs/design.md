# 面向 0 基础人群的 PyPTO 算子开发课程 设计文档

| 项目 | 内容 |
| --- | --- |
| 任务编号 | 11 月社区任务-开发一套面向 0 基础人群的 PyPTO 算子开发课程 |
| 课程载体 | Jupyter Notebook（ipynb），每大章含章节概述、教程内容与 README |
| 课程定位 | 0 基础开发者（学生/软件开发/软件测试/其他行业从业者）的 PyPTO 入门课程：从 AI 基础到理解神经网络并习惯使用 PyPTO 编写算子 |
| 运行环境 | CANNLab（推荐 Atlas A2/A3 系列产品，CANN 9.1.0）开箱即用、运行无报错 |
| 算子框架 | [PyPTO 算子编程框架](https://pypto.gitcode.com/index.html)——课程涉及算子均以 PyPTO 实现 |
| 交付仓 | https://gitcode.com/cann/cann-learning-hub （`contrib/tutorials/` 目录，PR 合入） |
| 文档状态 | 课程设计定稿（章节结构/知识点覆盖/格式与实操规范）；随 PR 交付全部 ipynb 与答案 |

# 1. 需求背景（required）

## 1.1 需求来源

CANN 社区任务 2026，11 月社区任务-开发一套面向 0 基础人群的 PyPTO 算子开发课程。
任务书：`开发一套面向0基础人群的PyPTO算子开发课程.md`。贡献者：`zhanghua145`。

PyPTO 是一个算子编程框架，算子开发是 AI 领域的基础设施。本课程默认开发者**没有
AI 基础**（仅需基础 Python 能力，不要求编程很强），从 AI 基础开始学习，学习过程中的
算子均以 PyPTO 实现——最终让 0 基础开发者**理解神经网络，并熟悉、习惯使用 PyPTO
框架编写算子**（任务书 §2.1/§2.4）。

| 材料 | 用途 |
| --- | --- |
| 任务书 §2.1–§2.4 | 课程定位、受众、内容域与目标 |
| 任务书 §3.2–§3.4 | 知识点覆盖清单（验收口径）、格式与实操要求 |
| 任务书 §4 | 交付件结构（ipynb 命名、章节实践、答案要求、参考 cann-learning-hub 课程提交标准） |
| [PyPTO 算子开发文档](https://pypto.gitcode.com/index.html) | PyPTO 核心 API 与算子实现范式 |
| cann-learning-hub `docs/course_submission_criteria.md` | 课程提交标准参照 |

## 1.2 背景介绍

### 现状与差距

- **受众断层**：现有昇腾算子开发资料面向有框架/编译基础的工程师，0 基础人群
  缺少一条「数学 → 神经网络 → 算子实现」的完整学习路径；
- **框架独特性**：PyPTO 作为算子编程框架，缺少配套的、由浅入深的入门课程——
  本课程即补齐该缺口：知识点讲解与算子实操一体化（每个数学/网络组件都有
  PyPTO 实现与验证）；
- **验收口径明确**（任务书 §3）：课程须在至少一个 CANNLab 环境开箱即用、
  运行无报错；知识点覆盖以 §3.2 清单逐项验收；练习题须精确有效且全部提供答案。

# 2. 需求分析（required）

## 2.1 课程目标（任务书 §2.4）

学完本课程，开发者可以：

1. 理解向量、矩阵、张量、概率、梯度、损失函数等概念；
2. 理解 CNN 与 RNN 等神经网络的结构，理解注意力机制与 Transformer 结构；
3. 掌握 PyPTO 核心 API，独立完成算子的 PyPTO 实现与验证；
4. 学会使用在线环境（CANNLab）或本地环境运行 PyPTO 代码并验证正确性。

## 2.2 知识点覆盖矩阵（对齐任务书 §3.2 逐项）

| 章 | 任务书要求（§3.2） | 覆盖小节 |
| --- | --- | --- |
| 01 数学基础 | 线性代数：标量/向量/矩阵/张量基本运算、降维、点积、矩阵乘法；微积分：导数、偏导数、梯度与链式法则 | 01.02–01.07 |
| 02 线性神经网络 | 线性回归到多层感知机：架构定义、数据处理、损失函数、前向/反向传播、欠拟合与过拟合、数值稳定性与参数初始化、激活函数性质与选用、层数/隐藏单元等超参数影响、权重衰减、暂退法（Dropout） | 02.02–02.08 |
| 03 卷积神经网络 | 局部连接、权重共享、池化理论；卷积层、填充与步幅、多输入/多输出通道、汇聚层、多通道使用；现代 CNN：AlexNet、VGG、ResNet、DenseNet | 03.02–03.08 |
| 04 循环神经网络 | 序列模型基本概念、文本预处理、语言模型、RNN 简洁实现；长期依赖梯度消失/爆炸；GRU、LSTM；深度 RNN、双向 RNN；编码器—解码器架构与 seq2seq | 04.02–04.08 |
| 05 注意力机制与大模型 | 注意力机制、Transformer 架构全部组成（多头注意力、位置编码、FFN、LayerNorm、残差连接等）；当代大语言模型基础概览 | 05.02–05.07 |

# 3. 详细设计（required）

## 3.1 课程结构（5 大章 × ≥5 小节）

```text
pypto-course-beginner/
├── README.md                       # 课程总览：受众/前置/环境/章节链接/学习路径
├── 01-math-fundamentals/           # 数学基础（线性代数 + 微积分）
│   ├── README.md                   # 章节导航（各小节链接）
│   ├── 01.01_chapter_intro.ipynb   # 本章介绍：主要内容与章节目标
│   ├── 01.02_scalars_vectors.ipynb     # 标量、向量与张量（PyPTO 向量算子）
│   ├── 01.03_matrices.ipynb            # 矩阵与基本运算（PyPTO 矩阵加/数乘）
│   ├── 01.04_dot_matmul.ipynb          # 降维、点积与矩阵乘法（PyPTO dot/matmul）
│   ├── 01.05_derivatives_gradient.ipynb# 导数、偏导数与梯度（数值微分算子）
│   ├── 01.06_chain_rule.ipynb          # 链式法则（复合函数梯度传递）
│   └── 01.07_chapter_test.ipynb        # 章节实践（练习 + 参考答案）
├── 02-linear-neural-networks/      # 线性神经网络与多层感知机
│   ├── README.md
│   ├── 02.01_chapter_intro.ipynb
│   ├── 02.02_linear_regression.ipynb   # 线性回归（PyPTO 实现 y=Xw+b）
│   ├── 02.03_loss_data_pipeline.ipynb  # 数据处理与损失函数（MSE 算子）
│   ├── 02.04_forward_backward.ipynb    # 前向传播与反向传播（梯度算子）
│   ├── 02.05_mlp_activation.ipynb      # 多层感知机与激活函数（ReLU/Sigmoid/tanh）
│   ├── 02.06_overfit_regularization.ipynb # 欠/过拟合、权重衰减、Dropout
│   ├── 02.07_init_stability_hparams.ipynb # 数值稳定性、参数初始化与超参数
│   └── 02.08_chapter_test.ipynb
├── 03-cnn/                         # 卷积神经网络
│   ├── README.md
│   ├── 03.01_chapter_intro.ipynb
│   ├── 03.02_conv_principle.ipynb      # 局部连接、权重共享、卷积层（PyPTO conv2d）
│   ├── 03.03_padding_stride.ipynb      # 填充与步幅
│   ├── 03.04_channels.ipynb            # 多输入/多输出通道与多通道使用
│   ├── 03.05_pooling.ipynb             # 汇聚层（池化：max/avg，PyPTO pooling）
│   ├── 03.06_modern_cnn.ipynb          # 现代卷积神经网络（AlexNet/VGG）
│   ├── 03.07_resnet_densenet.ipynb     # ResNet 与 DenseNet（残差连接）
│   └── 03.08_chapter_test.ipynb
├── 04-rnn/                         # 循环神经网络
│   ├── README.md
│   ├── 04.01_chapter_intro.ipynb
│   ├── 04.02_sequence_text.ipynb       # 序列模型、文本预处理与语言模型
│   ├── 04.03_rnn_impl.ipynb            # RNN 简洁实现（PyPTO rnn cell）
│   ├── 04.04_vanishing_exploding.ipynb # 长期依赖：梯度消失/爆炸
│   ├── 04.05_gru_lstm.ipynb            # GRU 与 LSTM（PyPTO 门控 cell）
│   ├── 04.06_deep_bidirectional.ipynb  # 深度 RNN 与双向 RNN
│   ├── 04.07_seq2seq.ipynb             # 编码器—解码器与 seq2seq
│   └── 04.08_chapter_test.ipynb
└── 05-attention-transformer/       # 注意力机制与大模型
    ├── README.md
    ├── 05.01_chapter_intro.ipynb
    ├── 05.02_attention.ipynb           # 注意力机制（PyPTO softmax/加权求和）
    ├── 05.03_self_attention.ipynb      # 自注意力与缩放点积（QKV 算子）
    ├── 05.04_multihead_pos.ipynb       # 多头注意力与位置编码
    ├── 05.05_transformer_block.ipynb   # Transformer 块：FFN、LayerNorm、残差
    ├── 05.06_llm_overview.ipynb        # 大语言模型概览（GPT 结构串联）
    └── 05.07_chapter_test.ipynb
```

## 3.2 每小节内容规范（任务书 §1/§3.3/§3.4）

1. **结构**：小节概述 → 教程内容 → 课后练习/实践（含答案）；每大章首节为
   `NN.01_chapter_intro.ipynb`（本章主要内容、章节目标），尾节为
   `NN.0X_chapter_test.ipynb`（章节实践，所有练习与实践均提供答案或参考实现，
   任务书 §4 第 2 条）；
2. **首次出现的新概念/新工具/新命令/新参数必须解释清楚，不默认学习者已知**
   （任务书 §3.3 第 5 条）——每节开头设「本节新概念」框；
3. **理论讲解不堆复杂数学公式**（§3.3 第 2 条）——公式配直觉解释与图示；
4. **复杂内容优先用图片说明**（§3.3 第 4 条）——架构/流程/数据流/模型结构
   （如 Transformer 块、RNN 展开、卷积滑窗）配示意图（课程 `images/` 资源，
   图注说明）；
5. **正文完整可运行**（§3.4 第 1 条）——工程创建、依赖安装、代码开发、
   脚本运行均在 code cell 中执行，学习者无需手动切换文件夹；
6. **教程正文代码与实践题代码相互解耦**（§3.4 第 3 条）——纯知识点小节用
   选择题/填空题，代码实操小节用编程实践题（全部提供答案）；
7. **行文无基础文字错误、无过时 API、无失效链接**（§3.3 第 1 条）——
   PyPTO API 以官方文档当前版本核对，链接以 PyPTO 文档站为准。

## 3.3 PyPTO 算子实现贯穿设计（本课程技术主体）

每个数学/网络组件讲解后紧跟 **PyPTO 实现 + 验证**（对拍 NumPy 参考值），
形成「概念 → 算子 → 验证」闭环，覆盖 PyPTO 核心 API：

| 章 | PyPTO 实现的算子（示例） | 验证方式 |
| --- | --- | --- |
| 01 | 向量加/数乘、dot、matmul、 reductions（sum/max）、数值微分 | NumPy 对拍（rtol 口径打印） |
| 02 | linear（y=Xw+b）、MSE 损失、梯度（数值/解析）、ReLU/Sigmoid/tanh、Dropout（训练/推理态） | 小规模手工值 + NumPy 对拍 |
| 03 | conv2d（含 padding/stride/多通道）、max/avg pooling、残差加 | 3×3 手工卷积示例 + NumPy 对拍 |
| 04 | rnn cell、GRU/LSTM 门控 cell、双向拼接、seq2seq 编解码步进 | 逐步手工状态对拍 |
| 05 | softmax、缩放点积注意力、多头拆拼、LayerNorm、FFN、位置编码 | NumPy/参考实现对拍 |

验证统一模式：`pypto 实现 → 与参考（NumPy）逐元素对拍 → 通过断言`
（学习者从第 1 章就习惯「写算子必验证」的工程纪律）。

## 3.4 环境与可运行性设计（任务书 §3.1/§3.4）

1. **环境**：CANNLab（推荐 Atlas A2/A3、CANN 9.1.0）——课程总 README 提供
   环境入口/申请说明与依赖清单；第 1 章首节含环境自检 cell（设备探测、
   PyPTO 导入、示例算子运行）；
2. **开箱即用**：所有 ipynb 在至少一个 CANNLab 环境完整跑通无报错
   （交付前逐节全量执行自验并记录，§4.1）；
3. **依赖安装在 code cell 内完成**（pip install 等），学习者零手工操作。

## 3.5 交付与合入（任务书 §5/§4）

- 课程 ipynb 与 README 按上述目录结构组织，PR 提交至
  `https://gitcode.com/cann/cann-learning-hub` 的 `contrib/tutorials/` 目录；
- 命名遵循 `NN.0X_topic.ipynb`（首节 intro、尾节 chapter_test）；
- 其他交付要求对齐 `cann-learning-hub` 的 `docs/course_submission_criteria.md`
  （课程提交标准）。

# 4. 可维可测分析

## 4.1 验收对齐（任务书 §3 逐项自验）

| 验收项 | 自验方式 |
| --- | --- |
| 至少 5 大章、内容域覆盖 §3.2 清单 | §2.2 覆盖矩阵逐项勾稽（章节-知识点映射表随 PR 附） |
| 每大章 ≥5 小节、首节介绍/尾节实践、练习全带答案 | 结构清单核查 + 答案完整性检查 |
| CANNLab 开箱即用、运行无报错 | 交付前在 CANNLab（A2/A3）逐节全量执行，记录运行日志与截图 |
| 内容准确、无过时 API、无失效链接 | PyPTO 文档版本核对 + 链接逐一检查 |
| 概念首现解释、图片优先、不堆公式 | 章节评审自查清单 |
| 正文/实践题代码解耦、依赖安装 cell 化 | 结构核查 |

## 4.2 课程质量保障

- 每节练习题「精确有效」：题目与该节知识点一一对应，答案可运行复现；
- 小节间难度单调递增，每章 intro 给出前置依赖（跨章引用明示）；
- 课程总 README 提供学习路径图与章节链接导航（各章 README 提供小节链接，
  任务书 §3.3 第 7 条）。

# 5. 参考链接

1. 任务书：开发一套面向0基础人群的PyPTO算子开发课程.md（2026-11-11）
2. PyPTO 算子开发文档：https://pypto.gitcode.com/index.html
3. cann-learning-hub 课程提交标准：https://gitcode.com/cann/cann-learning-hub/blob/master/docs/course_submission_criteria.md
4. 合入目录：https://gitcode.com/cann/cann-learning-hub/tree/master/contrib/tutorials
