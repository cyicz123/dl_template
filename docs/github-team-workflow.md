# 如何贡献代码

## 目录
- [简介](#简介)
- [前置准备](#前置准备)
- [贡献流程](#贡献流程)
- [分支管理策略](#分支管理策略)
- [代码提交规范](#代码提交规范)
- [Code Review流程](#code-review流程)
- [冲突解决](#冲突解决)
- [常见问题与解决方案](#常见问题与解决方案)

## 简介

本文档旨在指导贡献者如何正确地为本仓库贡献代码。遵循本指南可以帮助我们更高效地协作，减少冲突，确保代码质量。无论你是Git初学者还是有经验的开发者，都应按照本文档的指引参与项目开发。

## 前置准备

### 1. 安装Git

- **Windows**: 
  - 下载地址：[https://git-scm.com/download/win](https://git-scm.com/download/win)
  - 按照安装向导完成安装

- **Mac**:
  - 终端执行：`brew install git`
  - 或下载安装包：[https://git-scm.com/download/mac](https://git-scm.com/download/mac)

- **Linux**:
  - Ubuntu/Debian: `sudo apt-get install git`
  - CentOS/Fedora: `sudo yum install git`

### 2. 配置Git

打开终端或命令提示符，设置你的用户名和邮箱：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 3. 注册Github账号

访问 [Github官网](https://github.com/) 注册账号。

### 4. Fork本仓库

1. 访问本项目的GitHub页面
2. 点击右上角的"Fork"按钮，创建一个自己的仓库副本

### 5. 克隆仓库

```bash
git clone https://github.com/你的用户名/项目名.git
cd 项目名
```

### 6. 设置upstream

```bash
git remote add upstream https://github.com/原始组织名/项目名.git
```

## 贡献流程

### 1. 创建Issue

**在开始任何代码更改前，必须先创建一个Issue**：

1. 访问项目的Issues页面
2. 点击"New issue"按钮
3. 选择适当的Issue模板（如果有）
4. 填写Issue标题和描述，清晰说明你要解决的问题或实现的功能
5. 标记相关标签(Labels)
6. 提交Issue等待管理员确认

### 2. 获取最新代码

在开始工作前，确保你的本地仓库是最新的：

```bash
git checkout main
git pull upstream main
```

### 3. 创建功能分支

根据你的Issue编号和要开发的功能，创建并切换到新的功能分支：

```bash
git checkout -b feature/issue-123-功能描述 main
```

### 4. 进行开发工作

在你的功能分支上进行开发，并定期提交代码：

```bash
# 修改代码后...
git add .
git commit -m "feat: 添加了xxx功能 #123"
```

注意：在提交信息中引用Issue编号（如`#123`），这样GitHub会自动关联提交与Issue。

### 5. 保持分支同步

定期从upstream仓库的main分支获取最新代码，并通过rebase保持分支历史线性：

```bash
git checkout main
git pull upstream main
git checkout feature/issue-123-功能描述
git rebase main
```

如果出现冲突，请参考[冲突解决](#冲突解决)部分。

### 6. 推送分支到远程

当功能开发完成后，将你的分支推送到你的Fork仓库：

```bash
git push origin feature/issue-123-功能描述
```

如果你之前已经推送过分支并且使用了rebase，你需要强制推送：

```bash
git push origin feature/issue-123-功能描述 --force
```

注意：使用`--force`时要谨慎，确保没有其他人在同一分支上工作。

### 7. 创建Pull Request

1. 访问你Fork的仓库页面
2. 点击"Pull requests"标签
3. 点击"New pull request"按钮
4. 选择base分支为原始仓库的main分支，compare分支为你的功能分支
5. 填写PR标题和描述，说明你的变更内容
6. 在描述中使用关键词关联Issue，如"Closes #123"或"Fixes #123"
7. 点击"Create pull request"

PR描述应包含：
- 变更内容概述
- 关联的Issue编号
- 测试方法或结果
- 其他需要审阅者注意的事项

### 8. Code Review

等待项目维护者进行代码审查。如有修改意见，请在你的分支上继续提交修改，PR会自动更新。

### 9. PR被合并

当PR获得批准并被合并后，你可以：

```bash
# 切换回main分支
git checkout main

# 更新本地main
git pull upstream main

# 删除本地功能分支
git branch -d feature/issue-123-功能描述

# 更新你的fork仓库
git push origin main

# 删除远程功能分支
git push origin --delete feature/issue-123-功能描述
```

## 分支管理策略

我们采用基于功能分支(Feature Branch)的Git工作流：

### 主要分支类型

- **main**: 主分支，保存正式发布的历史
- **feature/issue-xxx-描述**: 功能分支，用于开发新功能
- **bugfix/issue-xxx-描述**: 修复bug的分支
- **docs/issue-xxx-描述**: 文档相关的修改
- **hotfix/issue-xxx-描述**: 紧急修复生产环境问题的分支

### 最小化原则

我们遵循**最小化原则**进行开发：

- 每个分支只做**一件事**，即解决一个具体的Issue
- 避免在一个分支中混合多种不相关的修改
- 分支内容应该尽可能小，便于审查和测试
- 如果一个功能较大，应在Issues中拆分为多个小功能点，分别创建分支开发

通过遵循最小化原则，可以：
- 减少代码冲突的可能性
- 简化代码审查过程
- 提高问题定位的效率
- 保持提交历史的清晰

## 代码提交规范

我们使用[约定式提交](https://www.conventionalcommits.org/zh-hans/v1.0.0/)规范来确保提交信息的一致性。

### 提交信息格式

```
<类型>[可选的作用域]: <描述> #Issue编号

[可选的正文]

[可选的脚注]
```

### 类型说明

- **feat**: 新功能
- **fix**: 修复Bug
- **docs**: 文档变更
- **style**: 代码格式调整，不影响代码含义的变更
- **refactor**: 代码重构，既不是新增功能，也不是修补bug的代码变动
- **perf**: 性能优化
- **test**: 添加或修改测试代码
- **chore**: 构建过程或辅助工具的变动

### 示例

```
feat(user): 添加用户登录功能 #123

实现了基于JWT的认证系统和用户登录表单。

Closes #123
```

## Code Review流程

### 提交PR后

1. 在PR中@提及相关人员或等待维护者指派Reviewer
2. 等待反馈
3. CI/CD检查需要通过

### Reviewer职责

1. 检查代码逻辑是否正确
2. 确认代码风格是否符合项目规范
3. 测试功能是否正常工作
4. 提出建设性的改进意见

### 处理反馈

1. 针对Review意见在PR评论中回应
2. 进行必要的代码修改
3. 推送新的修改到你的功能分支
4. PR会自动更新

### 合并标准

当满足以下条件时，PR可以被合并：

1. 至少一名项目维护者批准
2. 所有讨论都已解决
3. CI/CD检查通过
4. 代码符合项目质量标准

## 冲突解决

当你的分支与main分支有冲突时，需要手动解决：

### 使用rebase

```bash
git checkout main
git pull upstream main
git checkout feature/issue-123-功能描述
git rebase main
```

### 解决冲突步骤

1. 当rebase过程中出现冲突时，Git会暂停并提示你解决冲突
2. 打开有冲突的文件，找到被标记的冲突区域：
   ```
   <<<<<<< HEAD (当前更改)
   main分支上的代码
   =======
   你的代码
   >>>>>>> feature/issue-123-功能描述 (传入的更改)
   ```

3. 编辑文件解决冲突，删除冲突标记
4. 保存文件
5. 标记为已解决
   ```bash
   git add 已解决冲突的文件
   ```
6. 继续rebase过程
   ```bash
   git rebase --continue
   ```

7. 完成后推送到远程
   ```bash
   git push origin feature/issue-123-功能描述 --force
   ```
   
   注意：使用`--force`要谨慎，只在你确定没有其他人在同一分支工作时使用。

## 常见问题与解决方案

### 1. 撤销尚未推送的提交

如果你想撤销最后一次提交：

```bash
git reset --soft HEAD~1
```

### 2. 暂存当前工作

当你需要切换分支但不想提交当前工作时：

```bash
git stash save "工作描述"
# 切换分支后，恢复工作
git stash pop
```

### 3. 查看提交历史

```bash
git log --graph --oneline --all
```

### 4. 找出引入Bug的提交

```bash
git bisect start
git bisect bad  # 当前版本有bug
git bisect good 版本号  # 指定一个没有bug的旧版本
# Git会自动检出中间版本，你测试后标记：
git bisect good  # 或 git bisect bad
# 重复直到找出第一个有bug的提交
git bisect reset  # 完成后重置
```

### 5. 强制覆盖本地分支

⚠️ 危险操作，会丢失本地更改：

```bash
git fetch origin
git reset --hard origin/branch名称
```

### 6. 清理本地已删除的远程分支

```bash
git fetch --prune
```

### 7. 使用交互式rebase压缩提交记录

当你在功能分支上有多个小提交，提交PR前想将它们合并为一个或几个有意义的提交时，可以使用交互式rebase：

#### 基本用法

```bash
# n是你要合并的提交数量
git rebase -i HEAD~n
```

例如，压缩最近的5个提交：

```bash
git rebase -i HEAD~5
```

执行后会打开一个编辑器，显示类似下面的内容：

```
pick 1a2b3c4 第一个提交信息
pick 5d6e7f8 第二个提交信息
pick 9g0h1i2 第三个提交信息
pick 3j4k5l6 第四个提交信息
pick 7m8n9o0 第五个提交信息

# 命令:
# p, pick <commit> = 使用提交
# r, reword <commit> = 使用提交，但修改提交信息
# e, edit <commit> = 使用提交，但停下来修改
# s, squash <commit> = 使用提交，但融合到前一个提交
# f, fixup <commit> = 类似squash，但丢弃提交信息
# x, exec <command> = 使用shell运行命令
# b, break = 在这里停下来
# d, drop <commit> = 删除提交
```

#### 压缩提交步骤

1. 保留第一个提交为`pick`
2. 对于想要合并的后续提交，将`pick`改为`squash`（或`s`）或`fixup`（或`f`）
   - `squash`: 合并提交并保留提交信息
   - `fixup`: 合并提交但丢弃提交信息

例如，将5个提交压缩成一个：

```
pick 1a2b3c4 第一个提交信息
s 5d6e7f8 第二个提交信息
s 9g0h1i2 第三个提交信息
s 3j4k5l6 第四个提交信息
s 7m8n9o0 第五个提交信息
```

3. 保存并关闭编辑器
4. 如果使用了`squash`，Git会打开另一个编辑器让你编辑合并后的提交信息
5. 编辑、保存并关闭提交信息编辑器

#### 实际案例

假设你在开发新功能时有以下提交：

```
aac23f4 feat: 添加用户登录表单
b56d2e7 feat: 添加表单验证
c789f01 fix: 修复表单样式问题
d890e23 chore: 调整代码格式
ef01g23 docs: 添加注释
```

你希望将它们合并为一个完整的功能提交：

```bash
git rebase -i HEAD~5
```

在编辑器中修改为：

```
pick aac23f4 feat: 添加用户登录表单
s b56d2e7 feat: 添加表单验证
s c789f01 fix: 修复表单样式问题
s d890e23 chore: 调整代码格式
s ef01g23 docs: 添加注释
```

保存后，在新的编辑器中编写合并后的提交信息：

```
feat: 实现完整的用户登录功能

- 添加用户登录表单
- 实现表单验证
- 修复表单样式问题
- 优化代码格式和文档
```

#### 注意事项

1. **仅对尚未推送到远程的提交执行此操作**。如果已推送，需要使用`git push --force`，这可能影响其他人的工作。
2. 交互式rebase会改写提交历史，确保你了解其影响。
3. 如果rebase过程中出现冲突，解决冲突后使用`git add .`标记已解决，然后执行`git rebase --continue`继续。
4. 如果想要取消rebase操作，可以执行`git rebase --abort`。

通过适当地压缩提交，你可以在保持功能完整性的同时，使提交历史更加清晰和有条理，便于团队成员理解你的开发过程。

## 总结

遵循本文档中的贡献流程和规范，特别是"先提Issue，再提PR"的原则，可以帮助我们高效协作，减少冲突，保持代码库的整洁和可维护性。如果你对贡献流程有任何疑问或建议，请在Issues中提出。

感谢你对本项目的贡献！ 