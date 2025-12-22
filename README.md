<div align=center>
  <h1>昇思+昇腾开发板学习资源库</h1>
  <p><a href="./README.md">View English</a></p>
</div>

本项目是昇思+昇腾开发板的学习资源库，涵盖了从最基础的如何搭建环境，到如何基于昇思套件快速上手大模型的开发、微调、推理全流程，如何基于昇思框架接口从零实现一个简易版大模型的开发、训练、推理全流程的完整学习路径，开源包含课件、代码、实验指导手册、能力认证等丰富资源供开发者进行学习。同时，项目仓汇总了基于 昇思MindSpore 可复现、可扩展的昇腾开发板应用案例，覆盖图像、文本、表格、视频等多模态场景，为各类代表性任务提供可复用的实践方案。

## 📢 最新消息

- 2025-12-17 [功能优化]：重构仓库结构以优化应用导航体验；新增Issue与PR模板，让贡献流程更标准化。

## 前置知识

在正式开始学习、实践之前，您需要掌握：

- Python基础
- Linux命令基础
- Jupyter基础

您可以通过前置学习考试（*待上线*）进行自检。

## 环境准备

在开发前，请确保环境中的各软件包版本已完成配套，详见[环境搭建指南](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)

## 资源清单

应用案例（通常以 Notebooks 形式呈现）按技术领域分类，各领域下再按模型进一步细分，为开发者提供清晰的索引导航。

| 分类     | 简介                           |
| :------ | :----------------------------- |
| [课程(courses)](./courses/)                   | 《昇思+昇腾开发板：软硬结合玩转大模型实战》课程资源汇总，基于昇思套件及框架接口，手把手指导大模型开发、训练、推理全流程，详解混合精度训练等实用技术，以及开发板场景下的常见问题排查、性能优化思路。|
| [应用案例(applications)](./applications/)     | 基于 昇思MindSpore 的香橙派开发板应用案例，包含图像、文本、表格、视频等多模态任务场景。 |
| [算子开发(operators_development)](./operators_development/) | 昇腾开发板上的算子开发与自定义算子接入昇思MindSpore框架教程。|
| [测试工程(test)](./test/) | 昇腾开发板上的算子支持度测试脚本，一键获取当前CANN版本下对应昇思API的算子支持情况。|


## 常见问题（FAQ）

详见Wiki中[FAQ](https://github.com/mindspore-courses/applications/wiki/Developer-FAQ)。

## 贡献与反馈

1. **Issue**：欢迎各位开发者通过 [Issue](https://github.com/mindspore-lab/orange-pi-mindspore/issues) 提交建议或 bug 反馈

2. **Pull Request**: 开发者可发起 [PR](https://github.com/mindspore-courses/applications/pulls) 进行Bug修复或代码贡献（提交前请参考[提交规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)，由Committer @xing-yiren 及另一位Committer 完成评审合入），你的每一份参与都能让本项目更加完善。

3. **开源项目**：若开发者有符合条件的开源项目推荐/自荐，欢迎按照[项目仓规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)完善项目内容后，邮件至contact@public.mindspore.cn进行投稿，邮件标题请参考：`【昇思+香橙派项目投稿】项目名称`格式，并在正文中对项目进行简单介绍，附上代码仓链接。

### 提交规范

详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)

### 贡献者展示

向本项目的贡献者们致以最诚挚的感谢！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-lab/orange-pi-mindspore/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-lab/orange-pi-mindspore" />
  </a>
</div>
