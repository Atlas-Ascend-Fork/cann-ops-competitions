# CANN 开源社区竞赛仓

## 🎯 仓库简介
本仓库用于 CANN 开源社区各类竞赛、开源课题、社区任务等课题发布、开发者作品提交和展示。

## 📁 目录结构

### 顶层目录
- `official/` - 官方赛事
- `university/` - 高校赛事
- `enterprise/` - 行业/区域赛事
- `tasks/` - 开源课题/社区任务

### 赛事类型说明
- **官方赛事**：如CANN 全国挑战赛、算子天梯赛、区域赛等
- **高校赛事**：各高校基于 CANN 平台举办的赛事
- **行业/区域赛事**：各行业/区域基于CANN平台举办的赛事
- **开源任务**：CANN社区发起的开源课题/社区任务

## 🚀 如何参与

### 1. 选择赛事目录
根据您参与的赛事类型，选择对应的目录：
- 官方赛事：`official/{赛事名称}-{年份}/`
- 高校赛事：`university/{学校代码}/{赛事名称}-{年份}/`
- 行业/区域赛事：`enterprise/industry/{赛事名称}-{年份}/`、`enterprise/regional/{赛事名称}-{年份}/`
- 开源任务：`tasks/{任务名称}-{年份}/`

### 2. 作品提交规范
在对应赛事的 `submissions/` 目录下创建您的团队目录：
```
{team-id}_{name}_{work-title}/
├── README.md          # 作品说明文档
├── code/              # 源代码
├── model/             # 模型文件
├── results/           # 实验结果
└── docs/              # 相关文档
```

**命名规则：**
- `team-id`: 团队ID，支持多种格式：
  - 数字编号：team01, team02（推荐用于团队赛）
  - 自定义名称：awesome-team, ai-experts（适合个人赛或已有团队名称）
  - 个人ID：user123, developer-name（个人赛使用）
- `name`: 参赛选手姓名或者队伍名
- `work-title`: 作品标题（使用短横线分隔，如 image-classification）

### 3. 提交要求
- 确保代码可运行、文档完整
- 提供清晰的 README.md 说明
- 遵循赛事特定的提交要求

## 📝 提交指南
详细的作品提交规范请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证
本仓库采用MIT开源协议。

## 🤝 联系我们
- 竞赛论坛：https://gitcode.com/cann/cann-ops-competitions/discussions
- Issues：https://gitcode.com/cann/cann-ops-competitions/issues

## 🙏 致谢
感谢所有参与 CANN 开源社区竞赛的开发者和贡献者！
