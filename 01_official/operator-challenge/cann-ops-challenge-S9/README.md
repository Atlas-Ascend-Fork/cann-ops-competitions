# CANN算子挑战赛（S9赛季）

## 🏆 赛事介绍

昇腾AI创新大赛-算子挑战赛（S9赛季）是面向AI开发者打造的顶级赛事，旨在培养一批精通Ascend C算子开发的开发者，鼓励开发者基于昇腾AI云上算力、CANN的基础能力进行深度创新与实践，加速AI与行业融合，促进开发者能力提升。

本次赛事使用Ascend C编程语言开发大模型常用算子，挑战算子极致性能。

**赛事时间**：2026年6月15日 — 2026年9月15日 12:00

**CANN版本要求**：社区版8.5.0

## 🎯 赛题列表

| 序号 | 算子名称 | 赛题说明 | 难度 |
|------|----------|----------|------|
| 1 | **Concat** | 沿指定维度拼接多个输入张量，等价于 `torch.cat` | ⭐ |
| 2 | **Greater** | 逐元素比较 `x > y`，输出布尔结果，等价于 `torch.gt` | ⭐ |
| 3 | **IndexAdd** | 沿指定维度按索引累加，`out[index] += alpha * source`，等价于 `torch.index_add` | ⭐⭐ |
| 4 | **Transpose** | 按指定维度顺序重排张量，等价于 `torch.permute` / `torch.transpose` | ⭐⭐ |
| 5 | **SquareSumV1** | 沿指定轴计算元素平方和，支持多轴与 keepdims | ⭐⭐ |

> 注：具体赛题详情请下载官方赛题文档查看。

## 📁 提交规范

### 目录结构

在 `submissions/` 目录下按赛题和团队名称建立目录结构，每个赛题可包含多个团队的提交：

```
submissions/
└── {OperatorName}/
    ├── {TeamName}/
    │   ├── build.sh                       # 编译脚本
    │   ├── CMakeLists.txt                 # 顶层CMake
    │   ├── CMakePresets.json              # CMake预设配置
    │   ├── op_host/                       # Host侧代码
    │   │   ├── CMakeLists.txt
    │   │   └── {op_name}.cpp              # Tiling & 算子注册
    │   ├── op_kernel/                     # Kernel侧代码
    │   │   ├── CMakeLists.txt
    │   │   ├── {op_name}.cpp              # Kernel入口
    │   │   └── {op_name}_tiling.h         # TilingData结构
    │   └── framework/                     # 框架适配代码（可选）
    │       └── tf_plugin/                 # TensorFlow插件（可选）
    │           ├── CMakeLists.txt
    │           └── tensorflow_{op_name}_plugin.cc
    └── {AnotherTeam}/
        └── ...
```

### 命名规则

- `{OperatorName}`：赛题算子名称，与赛题列表保持一致
- `{TeamName}`：团队名称，建议使用英文短横线分隔格式（如 `team-alphafold`）

### 提交要求

1. 确保代码可编译、可运行，包含完整的 CMakeLists.txt
2. Host 端代码需正确处理 Tiling 分片、InferShape、多核调度
3. Kernel 端代码需通过 Ascend C 编译器检查
4. 提供必要的注释说明核心算法和优化策略
5. 鼓励提交性能优化对比数据
