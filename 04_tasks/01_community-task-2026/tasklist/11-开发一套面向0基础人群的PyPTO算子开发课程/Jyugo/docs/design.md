# 面向零基础学习者的 PyPTO 算子开发课程设计文档

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11 月社区任务-开发一套面向 0 基础人群的 PyPTO 算子开发课程 |
| 贡献者 | Jyugo |
| 目标仓库 | `cann/cann-learning-hub` |
| 课程提交目录 | `contrib/tutorials/pypto_operator_course/` |
| 目标产品 | Atlas A2 / Atlas A3 系列产品 |
| 软件版本 | CANN 9.1.0、PyPTO 0.2.1、Python 3.12 |
| 运行模式 | 默认 SIM（CPU 仿真），可选 NPU |
| 文档版本 | v1.0，2026-11-11 |
| 文档状态 | 设计、实现与全量运行自验证已完成，待社区评审 |

## 一、需求背景（required）

### 1.1 需求来源

本任务来自 CANN 社区 2026 年 11 月社区任务：**开发一套面向 0 基础人群的 PyPTO 算子开发课程**。

- 任务页面：<https://www.hiascend.com/activities/task-center/details/1073bd69ad9f48f7848f2d940a5b2918?menu=guide>
- 课程上库仓库：<https://gitcode.com/cann/cann-learning-hub>
- 课程上库验收标准：<https://gitcode.com/cann/cann-learning-hub/blob/master/docs/course_submission_criteria.md>
- PyPTO 文档：<https://pypto.gitcode.com/index.html>
- PyPTO 仓库：<https://gitcode.com/cann/pypto>

任务要求以 Jupyter Notebook 为载体，开发一套从数学基础到 Transformer 与大语言模型的入门课程。课程中的核心算子全部使用 PyPTO 实现，学习者只需具备基础 Python 能力，不需要预先学习线性代数、微积分或深度学习。

### 1.2 课程定位

PyPTO 是昇腾 CANN 生态中的算子编程框架，算子开发是 AI 基础设施的核心工作。本课程面向学生、软件开发工程师、软件测试工程师以及其他行业从业者，目标是把“完全不了解 AI 的开发者”带到以下水平：

1. 理解向量、矩阵、张量、概率、梯度、损失函数等基础概念；
2. 理解线性回归、多层感知机、CNN、RNN、GRU、LSTM、注意力机制与 Transformer 的结构；
3. 掌握 PyPTO 的核心 API，能够独立编写、运行并验证算子；
4. 学会在 CANNLab 在线环境或本地环境中复现课程代码。

### 1.3 与课程规范的对应

课程目录、文件命名、章节组织、练习与答案、环境验证均按 `cann-learning-hub` 的课程提交规范执行。课程最终以 `contrib/tutorials/pypto_operator_course/` 目录整体提交，配套 PR 描述与验证报告随课程一并提供。

## 二、需求分析（required）

### 2.1 目标受众与前置知识

| 项目 | 要求 |
| --- | --- |
| 目标人群 | 学生、软件开发工程师、软件测试工程师、其他行业从业者 |
| 前置知识 | 基础 Python（变量、循环、函数、列表）；不需要 AI 基础 |
| 前置课程 | 无；第 1 章补齐课程所需的线性代数、微积分与概率知识 |
| 学习方式 | 按章节顺序运行 Notebook，完成每节练习与章节实践 |
| 环境要求 | 至少一个可通过 CANNLab 创建的环境，开箱即用、运行无报错 |

### 2.2 学习目标

| 能力层次 | 对应章节 | 学完能够完成的任务 |
| --- | --- | --- |
| 数学与 PyPTO 入门 | 第 1 章 | 辨认张量形状与轴；解释降维、点积、矩阵乘法、梯度、链式法则与 Softmax；写出第一个 PyPTO 算子并独立验证 |
| 神经网络基础 | 第 2 章 | 解释线性回归、损失函数、反向传播与梯度下降；实现 Softmax 回归与 MLP；判断欠拟合/过拟合并应用正则化 |
| 视觉与序列模型 | 第 3、4 章 | 实现卷积、填充、步幅、多通道与池化；解释 AlexNet/VGG/ResNet/DenseNet；实现 RNN、BPTT、GRU、LSTM 与 Seq2Seq |
| 注意力与大模型 | 第 5、6 章 | 实现 QKV 注意力、缩放点积注意力、自注意力、多头注意力与掩码；组装 Transformer 编码器/解码器并解释自回归生成与 KV Cache |

### 2.3 需求拆解与设计响应

| 任务书要求 | 设计响应 |
| --- | --- |
| 至少 5 个大章节，覆盖数学基础、线性神经网络、CNN、RNN、注意力与大模型 | 共 6 章、38 个 Notebook：数学基础、线性神经网络、卷积神经网络、循环神经网络、注意力机制、Transformer 与大语言模型 |
| 数学基础包含线性代数与微积分 | 第 1 章覆盖标量/向量/矩阵/张量、降维、点积、矩阵乘法、导数、偏导数、梯度与链式法则，并补充概率与 Softmax |
| 线性神经网络到多层感知机，包含损失函数、前向/反向传播、激活函数、欠拟合/过拟合、初始化、权重衰减、Dropout | 第 2 章逐项覆盖，并给出完整训练循环、数值梯度对照与正则化实验 |
| CNN 包含局部连接、权重共享、池化、卷积层、填充、步幅、多输入/多输出通道、现代 CNN | 第 3 章逐项覆盖，并用 PyPTO 实现卷积、池化与残差/拼接模块 |
| RNN 包含序列模型、文本预处理、语言模型、BPTT、梯度消失/爆炸、GRU、LSTM、深度/双向 RNN、编码器—解码器与 Seq2Seq | 第 4 章逐项覆盖，并用 PyPTO 实现各门控单元与状态更新 |
| 注意力机制与 Transformer 所有组成部分 | 第 5 章覆盖 QKV、缩放点积、自注意力、多头、掩码与交叉注意力；第 6 章覆盖嵌入、位置编码、编码器层、解码器层、GPT、KV Cache 与采样 |
| 所有课程文件为 Jupyter Notebook 格式 | 全部 38 个教学文件均为 `.ipynb`，答案使用 Markdown 与可执行代码片段 |
| 每章包含章节概述、教程内容与 README，README 链接各小节 | 每章首节为 `0X.01_chapter_intro.ipynb`，末节为 `0X.0Y_chapter_test.ipynb`，每章 `README.md` 提供小节导航 |
| 每节包含小节概述、教程内容、课后练习，练习提供答案 | 每个教学小节包含“本节概述 + 教程内容 + 课后练习”；每章实践包含客观题与简单/中等/困难编程题；共 38 份答案 |
| 代码完整可运行，无需手动切换目录 | 每个 Notebook 自带 import、数据生成、内核定义与验证；不使用相对文件依赖，可在任意工作目录启动 |
| 复杂内容优先使用图片说明 | 共 49 张原创配图，覆盖张量轴、计算图、卷积窗口、池化、循环状态、注意力与 Transformer 数据流 |
| 课程在 CANNLab 环境开箱即用 | 默认 SIM（CPU 仿真）模式，不依赖 NPU；已在 Atlas A2（910B4）+ CANN 9.1.0 环境完成全量验证 |

### 2.4 验收边界

1. 课程必须覆盖任务书明确列出的全部知识点，不能只罗列模型名称；
2. 正文代码与练习代码解耦，所有练习必须提供答案或参考实现；
3. 所有 Notebook 必须能从头到尾独立执行，提交前清除输出；
4. 至少在一个 CANNLab 环境完成全量运行验证，并保留验证记录；
5. 课程目录、文件名与章节结构符合 `cann-learning-hub` 的课程规范。

## 三、详细设计（required）

### 3.1 总体设计原则

1. **零基础优先**：从具体数字、形状和小例子出发，首次出现的新概念、新工具、新命令与新参数均给出解释，不默认学习者已知。
2. **理论—代码—练习闭环**：每节先说明“学什么、为什么”，再给出可运行代码，最后用练习检验；每章以综合实践收尾。
3. **PyPTO 为主线**：课程中涉及的张量运算、矩阵乘法、归约、激活、Softmax、卷积、池化、RNN/GRU/LSTM、注意力与 Transformer 子层均使用 PyPTO 实现；PyTorch 仅用于生成数据、承载主机张量与提供独立参考结果。
4. **Notebook 自包含**：每个 Notebook 都能从新内核开始执行，不依赖上一节的变量、文件或工作目录。
5. **结果可判定**：所有数值示例都打印形状、数值和“与 PyTorch 一致”等显式检查，学习者能立即判断对错。
6. **图文结合**：架构、流程、数据流与模型结构使用原创配图说明，避免堆砌公式。

### 3.2 课程结构总览

课程共 6 章、38 个 Notebook，其中章节概述 6 个、教程小节 26 个、章节实践 6 个。

| 章节 | Notebook 数 | 主要内容 | 配图 |
| --- | ---: | --- | ---: |
| 01_math_foundations | 6 | 标量/向量/矩阵/张量、降维、点积、矩阵乘法、导数/梯度/链式法则、概率/Softmax、PyPTO 入门 | 8 |
| 02_linear_neural_networks | 7 | 线性回归、损失函数、反向传播、Softmax 回归、MLP、激活函数、欠拟合/过拟合与正则化 | 10 |
| 03_convolutional_neural_networks | 6 | 卷积、填充、步幅、多输入/多输出通道、池化、完整 CNN 数据流、AlexNet/VGG/ResNet/DenseNet | 10 |
| 04_recurrent_neural_networks | 6 | 序列模型、文本预处理、语言模型、RNN/BPTT、梯度消失/爆炸、GRU、LSTM、深度/双向 RNN、Seq2Seq | 8 |
| 05_attention_mechanism | 6 | 查询—键—值、缩放点积注意力、自注意力、多头注意力、掩码、交叉注意力 | 6 |
| 06_transformer_and_llms | 7 | Transformer 架构、词嵌入/位置编码、编码器层、解码器层、GPT、KV Cache、采样 | 7 |

### 3.3 各章设计

#### 第 1 章 数学基础与 PyPTO 入门

| Notebook | 内容 |
| --- | --- |
| `01.01_chapter_intro.ipynb` | 章节概述：AI、模型与算子的关系，PyPTO 的定位，课程地图，环境说明与自检 |
| `01.02_scalars_vectors_matrices_tensors.ipynb` | 标量、向量、矩阵、张量；shape、axis、逐元素运算与广播；用 PyPTO 完成张量创建与逐元素计算 |
| `01.03_reduction_dot_matmul.ipynb` | 求和/均值/最大值归约、`keepdim`、点积、矩阵乘法；用 PyPTO 的 `sum`、`amax`、`matmul` 实现并与 PyTorch 对照 |
| `01.04_derivatives_gradients_chain_rule.ipynb` | 导数、偏导数、梯度、链式法则；数值梯度与解析梯度对照，为后续反向传播打基础 |
| `01.05_probability_softmax.ipynb` | 概率、期望、方差、Softmax；稳定 Softmax（先减最大值）与概率和检查 |
| `01.06_chapter_test.ipynb` | 章节实践：客观题 + 张量统计、线性打分 + Softmax、数值梯度 + 梯度下降三道编程题 |

#### 第 2 章 线性神经网络

| Notebook | 内容 |
| --- | --- |
| `02.01_chapter_intro.ipynb` | 章节概述：从线性回归到多层感知机，学习目标与知识地图 |
| `02.02_linear_regression_model.ipynb` | 线性回归模型、合成数据、均方误差损失；定义神经网络架构与数据处理流程 |
| `02.03_backprop_and_training.ipynb` | 前向传播、反向传播、链式求导、SGD 更新与完整训练循环 |
| `02.04_softmax_regression.ipynb` | 从回归到分类：Softmax 回归、交叉熵损失与分类评估 |
| `02.05_multilayer_perceptron.ipynb` | 多层感知机、隐藏层、激活函数（ReLU/Sigmoid/Tanh）的性质与选用场景 |
| `02.06_generalization_regularization.ipynb` | 欠拟合与过拟合、数值稳定性、参数初始化、权重衰减与 Dropout |
| `02.07_chapter_test.ipynb` | 章节实践：客观题 + 线性回归训练、补全 MLP 反向传播、诊断并改进过拟合 |

#### 第 3 章 卷积神经网络

| Notebook | 内容 |
| --- | --- |
| `03.01_chapter_intro.ipynb` | 章节概述：为什么需要卷积，局部连接、权重共享与感受野 |
| `03.02_convolution_basics.ipynb` | 二维互相关/卷积的直觉与逐窗口计算，使用 PyPTO 实现基础卷积 |
| `03.03_padding_stride_channels.ipynb` | 填充、步幅、输出尺寸推导、多输入/多输出通道与权重形状 |
| `03.04_pooling_and_cnn_flow.ipynb` | 最大池化、平均池化、池化窗口与边界处理，完整 CNN 数据流 |
| `03.05_modern_cnn.ipynb` | AlexNet、VGG、ResNet、DenseNet 的结构差异；用 PyPTO 实现残差相加与特征拼接模块 |
| `03.06_chapter_test.ipynb` | 章节实践：客观题 + 边缘提取、多通道卷积、组合小型 CNN 并验证 |

#### 第 4 章 循环神经网络

| Notebook | 内容 |
| --- | --- |
| `04.01_chapter_intro.ipynb` | 章节概述：序列顺序与隐藏状态的作用 |
| `04.02_sequence_models_and_text.ipynb` | 序列模型、文本预处理、词表、语言模型与下一词预测 |
| `04.03_rnn_and_bptt.ipynb` | 基本 RNN、随时间反向传播（BPTT）、长期依赖下的梯度消失/爆炸 |
| `04.04_gru_and_lstm.ipynb` | GRU 的更新门/重置门/候选状态，LSTM 的输入门/遗忘门/输出门与记忆状态；逐门 PyPTO 实现 |
| `04.05_deep_bidirectional_seq2seq.ipynb` | 深度 RNN、双向 RNN、编码器—解码器架构与 Seq2Seq |
| `04.06_chapter_test.ipynb` | 章节实践：客观题 + RNN 序列预测、门控状态更新、短序列映射 |

#### 第 5 章 注意力机制

| Notebook | 内容 |
| --- | --- |
| `05.01_chapter_intro.ipynb` | 章节概述：从固定状态到按内容检索信息 |
| `05.02_attention_intuition_qkv.ipynb` | 查询（Query）、键（Key）、值（Value）的直觉与注意力权重计算 |
| `05.03_scaled_dot_product_attention.ipynb` | 缩放点积注意力：QK 转置乘积、缩放、Softmax 加权求和 |
| `05.04_self_and_multihead_attention.ipynb` | 自注意力、多头注意力的投影/分头/合并与输出投影 |
| `05.05_masks_and_attention_variants.ipynb` | Padding Mask、Causal Mask、掩码位置与交叉注意力 |
| `05.06_chapter_test.ipynb` | 章节实践：客观题 + 验证注意力权重、实现多头注意力、检查因果性 |

#### 第 6 章 Transformer 与大语言模型

| Notebook | 内容 |
| --- | --- |
| `06.01_chapter_intro.ipynb` | 章节概述：从 Transformer 到大语言模型 |
| `06.02_transformer_architecture.ipynb` | Transformer 整体架构与数据流，编码器—解码器分工 |
| `06.03_embeddings_and_positional_encoding.ipynb` | 词嵌入、位置编码与位置顺序的显式表示 |
| `06.04_encoder_layer.ipynb` | 编码器层：多头自注意力、前馈网络、残差连接与 LayerNorm |
| `06.05_decoder_and_generation.ipynb` | 解码器层：掩码自注意力、交叉注意力、输出投影与自回归生成 |
| `06.06_from_transformer_to_llm.ipynb` | Decoder-only 大语言模型、训练标签移位、KV Cache 与采样策略 |
| `06.07_chapter_test.ipynb` | 章节实践：客观题 + 组装 Transformer 子层、短序列生成、因果性检查 |

### 3.4 练习与章节实践设计

1. **每节练习**：每个教程小节末尾设置“课后练习”，题型包括选择题、填空题和编程题；编程题要求使用 PyPTO 完成一个可运行的小算子，并打印与 PyTorch 参考结果的对照。
2. **章节实践**：每章最后一节包含客观题与三道编程题，难度分别为简单、中等、困难；题目覆盖本章核心知识点，并综合运用前序章节内容。
3. **答案独立**：每章 `answer/` 目录下按 Notebook 一一对应保存参考答案（共 38 份）。答案使用 Markdown 说明 + 可执行代码片段，独立于正文的中间状态。
4. **正文与练习解耦**：正文代码与练习代码使用独立函数和输入；练习答案不进入正文执行路径，避免未完成练习影响课程运行。

### 3.5 Notebook 工程组织

```text
contrib/tutorials/pypto_operator_course/
├── README.md                       # 课程总览、适用人群、环境、章节目录与运行约定
├── VALIDATION.md                   # 环境、验证方法与全量运行结论
├── 01_math_foundations/
│   ├── README.md                   # 本章导览与各小节链接
│   ├── 01.01_chapter_intro.ipynb
│   ├── 01.02_scalars_vectors_matrices_tensors.ipynb
│   ├── ...
│   ├── 01.06_chapter_test.ipynb
│   ├── answer/                     # 与 Notebook 一一对应的参考答案
│   ├── images/                     # 本章原创配图
│   └── src/README.md               # 源码目录说明（所有示例均在 Notebook 中）
├── 02_linear_neural_networks/
├── 03_convolutional_neural_networks/
├── 04_recurrent_neural_networks/
├── 05_attention_mechanism/
└── 06_transformer_and_llms/
```

命名约定：

- 章节目录：`0n_english_name`；
- 小节文件：`0n.0m_english_name.ipynb`；
- 章节概述固定为 `0n.01_chapter_intro.ipynb`，章节实践固定为最后一节 `0n.0m_chapter_test.ipynb`；
- 答案文件名与 Notebook 同名，追加 `_answers.md`；
- 图片使用英文文件名，存放于本章 `images/`，通过相对路径引用。

Notebook 运行约定：

1. 每个 Notebook 的第一个代码单元完成环境准备（import、日志级别、`RUN_MODE`、`DEVICE`、随机种子与版本打印）；
2. 不在 Notebook 中执行 `cd`，不依赖其他 Notebook 生成的文件；
3. 正文代码单元按顺序可直接执行，不保留执行输出；
4. 首次运行需要 JIT 编译，耗时较长属于正常现象；
5. 所有 PyPTO 内核通过 `out[:] = ...` 写回输出，不在内核中写 `return`。

### 3.6 PyPTO 教学与算子实现设计

#### 3.6.1 编程模型

课程统一采用 PyPTO Tensor 编程模型：

```python
@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def kernel(x: pypto.Tensor([], pypto.DT_FP32),
           out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(4, 8)
    out[:] = pypto.sum(x, dim=-1, keepdim=True)
```

- 主机侧使用 PyTorch 准备输入/输出张量、设置随机种子并读取结果；
- `@pypto.frontend.jit` 描述设备端计算，参数使用 `pypto.Tensor` 类型注解；
- 使用 `pypto.set_vec_tile_shapes` / `pypto.set_cube_tile_shapes` 设置与张量维数一致的 Tile 形状；
- `pypto.RunMode.SIM` 为默认运行模式，安装 `torch_npu` 后可切换 `pypto.RunMode.NPU`；
- 每个内核只演示一个概念，形状固定且较小，降低编译与理解成本。

#### 3.6.2 算子覆盖

| 教学模块 | 典型 PyPTO API | 验证方式 |
| --- | --- | --- |
| 张量与基础运算 | `pypto.Tensor`、`pypto.DT_FP32`、逐元素加减乘除、广播 | 形状推导 + 与 PyTorch 逐元素对照 |
| 归约与矩阵运算 | `pypto.sum`、`pypto.amax`、`pypto.matmul` | `torch.allclose`、手算小例子 |
| 激活与 Softmax | `pypto.relu`、`pypto.sigmoid`、`pypto.tanh`、`pypto.softmax`、`pypto.exp` | 饱和区、概率和、稳定 Softmax 对照 |
| 形状变换 | `pypto.view`、`pypto.reshape`、`pypto.permute`、`pypto.transpose`、`pypto.assemble` | 形状、轴顺序与参考实现对照 |
| 卷积与池化 | 窗口切片、乘加、`pypto.sum`/`pypto.amax` 归约、`pypto.pad` | 手算窗口、PyTorch `conv2d`/`pool2d` 对照 |
| 循环单元 | 线性变换、激活、门控状态更新、`pypto.sigmoid`/`pypto.tanh` | 逐时间步状态对照、门控数值检查 |
| 注意力 | `pypto.matmul`、`pypto.softmax`、`pypto.tril`、`pypto.full` | 掩码位置权重、有效长度、因果性检查 |
| Transformer | 嵌入查表、位置编码、多头重排、残差、LayerNorm、FFN、交叉注意力 | 形状流、归一化统计量、生成结果检查 |

#### 3.6.3 数值与正确性设计

- 基础示例统一使用 FP32；
- 每个数值示例都提供独立参考结果（PyTorch 或手算），并打印 `torch.allclose` / 形状检查 / 概率和检查的布尔结果；
- Softmax 先减最大值再取指数，避免上溢；
- 注意力有效行至少保留一个可见位置，Padding 查询位置按掩码处理；
- 梯度示例使用中心差分与高精度参考对照，避免在 ReLU 零点等不可导位置做相等判定；
- Dropout 使用固定种子，明确训练/推理分支与缩放约定；
- PyPTO 对中间张量存在缓冲区复用优化，课程对需要多次使用的张量使用 `pypto.clone`，或将中间结果拆到多个内核，并在相关小节解释原因。

### 3.7 环境适配与 SIM/NPU 双模式

课程默认使用 SIM（CPU 仿真）模式，原因如下：

1. 不要求学习者必须拥有昇腾 NPU；
2. 在 CANNLab CPU 仿真环境、云开发环境中均可直接运行；
3. 计算结果与真实硬件一致，适合学习、调试与正确性验证。

每个 Notebook 的环境准备单元给出 NPU 模式切换代码：

```python
# import torch_npu
# torch.npu.set_device(0)
# RUN_MODE = pypto.RunMode.NPU
# DEVICE = "npu:0"
```

课程明确说明：NPU 模式需要按具体硬件确认 Tile 配置；本次全量验证在 SIM 模式下完成。

## 四、可维可测分析（required）

### 4.1 内容与结构检查

| 检查项 | 结论 |
| --- | --- |
| 章节数量与知识点覆盖 | 6 章覆盖数学基础、线性神经网络、CNN、RNN、注意力与 Transformer/LLM |
| 章节概述与实践 | 每章首节为章节概述，末节为章节实践 |
| 练习与答案 | 26 个教程小节均含课后练习；6 个章节实践含客观题与三道编程题；38 份答案一一对应 |
| 目录与命名 | 章节目录、小节文件名、答案与图片命名符合课程规范 |
| 图文说明 | 49 张原创配图，覆盖架构、流程、数据流与模型结构 |
| 代码解耦 | 正文代码与练习代码独立；答案不依赖正文中间状态 |
| 输出清理 | 提交前已清除所有 Notebook 的 code cell 输出 |

### 4.2 全量运行验证

**验证环境**

| 项目 | 值 |
| --- | --- |
| 服务器 | Atlas A2 训练/推理系列（910B4） |
| CANN | 9.1.0 |
| PyPTO | 0.2.1 |
| PyTorch | 2.15.0+cpu |
| Python | 3.12.13 |
| 运行模式 | SIM（CPU 仿真） |
| 验证日期 | 2026-11-11 |

**验证方法**

按章节顺序对每个 Notebook 执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=pypto312 \
  --ExecutePreprocessor.timeout=2000 \
  --output-dir <executed_dir> <notebook>.ipynb
```

执行后检查：

1. 是否存在 `output_type == "error"` 的单元格输出；
2. 所有显式正确性检查是否输出 `True`；
3. Notebook 是否能从第一个代码单元执行到最后一个代码单元。

**验证结果**

| 章节 | Notebook 数 | 单元格报错 | 正确性检查 |
| --- | ---: | ---: | --- |
| 第 1 章 数学基础与 PyPTO 入门 | 6 | 0 | 全部通过 |
| 第 2 章 线性神经网络 | 7 | 0 | 全部通过 |
| 第 3 章 卷积神经网络 | 6 | 0 | 全部通过 |
| 第 4 章 循环神经网络 | 6 | 0 | 全部通过 |
| 第 5 章 注意力机制 | 6 | 0 | 全部通过 |
| 第 6 章 Transformer 与大语言模型 | 7 | 0 | 全部通过 |
| **合计** | **38** | **0** | **全部通过** |

补充说明：

- 每个 Notebook 首次运行需要 JIT 编译 PyPTO 内核，耗时从几秒到一两分钟不等；
- SIM 模式会输出性能仿真日志（如 `AICore Prof Summary`），属于 PyPTO 正常行为，不影响计算结果；
- 详细记录见课程目录下 `VALIDATION.md`。

### 4.3 可维护性设计

1. **章节独立**：新增或替换某一章不会影响其他章节的 Notebook；
2. **答案独立**：修改正文不会破坏答案的可执行性，答案可单独运行；
3. **风格统一**：所有 Notebook 使用相同的环境准备、内核定义、验证与练习结构；
4. **版本可追溯**：Notebook 打印 PyTorch/PyPTO 版本；`VALIDATION.md` 记录验证环境与结果；
5. **后续扩展**：可在现有章节内追加小节，或新增“推理优化”“算子性能分析”等章节，不影响既有目录约定。

## 五、交付与实施情况

### 5.1 交付物清单

| 交付物 | 数量/说明 |
| --- | --- |
| Jupyter Notebook | 38 个（6 个章节概述 + 26 个教程小节 + 6 个章节实践） |
| 参考答案 | 38 份（位于各章 `answer/`） |
| 配图 | 49 张（位于各章 `images/`） |
| README | 课程总 README + 6 个章节 README + 6 个 `src/README.md` |
| 验证报告 | `VALIDATION.md` |
| 提交 PR | 课程目录 + `contrib/tutorials/README.md` 的课程/贡献者列表更新 |

### 5.2 实施与自验证情况

- 课程已按 6 章、38 个 Notebook 完成开发；
- 所有核心算子均使用 PyPTO 实现，PyTorch 仅用于数据生成与独立参考；
- 已在 Atlas A2（910B4）+ CANN 9.1.0 + PyPTO 0.2.1 环境完成全部 38 个 Notebook 的 `nbconvert --execute` 验证；
- 验证结果：0 个单元格报错，所有正确性检查均为 `True`；
- 提交前已清除 Notebook 输出，符合课程上库规范。

### 5.3 后续计划

1. 根据社区评审意见调整章节划分、文字表述与练习难度；
2. 在 `cann-learning-hub` 在线体验环境由 Committer 配置后复验；
3. 如需在真实 NPU 上运行，补充各硬件型号的 Tile 配置说明与验证记录；
4. 根据 PyPTO 版本演进更新 API 用法，保持课程无过时接口。

## 六、风险与应对

| 风险 | 应对措施 |
| --- | --- |
| PyPTO 版本差异导致 API 或缓冲区复用行为变化 | Notebook 打印实际版本；课程按“中间张量尽量只使用一次”编写；升级后重新执行全量验证 |
| 不同硬件 UB/Tile 配置不同 | 课程默认 SIM 模式；NPU 模式明确要求按硬件确认 Tile 配置，不宣称所有硬件开箱即用 |
| 在线环境依赖缺失（如 matplotlib） | 在 README 与相关小节给出安装命令；核心课程不依赖绘图库 |
| 课程内容与任务书知识点偏差 | 按章节建立知识点覆盖表，评审时逐项核对 |
| Notebook 运行时间过长 | 使用小张量、小模型与固定随机种子；首次 JIT 编译时间在文档中说明 |
| 练习答案与正文不一致 | 答案独立保存并单独执行验证；修改正文时同步回归答案 |

## 七、参考链接

- 任务页面：<https://www.hiascend.com/activities/task-center/details/1073bd69ad9f48f7848f2d940a5b2918?menu=guide>
- 课程上库仓库：<https://gitcode.com/cann/cann-learning-hub>
- 课程上库验收标准：<https://gitcode.com/cann/cann-learning-hub/blob/master/docs/course_submission_criteria.md>
- CANNLab 课程开发指南：<https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_course_development_guide.md>
- PyPTO 文档：<https://pypto.gitcode.com/index.html>
- PyPTO 仓库：<https://gitcode.com/cann/pypto>
