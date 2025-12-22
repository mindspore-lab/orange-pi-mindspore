<div align=center>
  <h1>基于昇思+香橙派开发板的离线推理案例</h1>
</div>

本路径下包含基于昇思MindSpore的香橙派开发板离线推理应用案例。

## 模型案例清单和版本兼容

### 推理案例(inference)

官方开发的离线推理案例，覆盖OCR、图像分类、图像增强、风格迁移、语义分割经典任务，及如机械臂分拣的趣味应用。

| 案例 | CANN | Mindspore | 香橙派型号 |
|  ----  | ---- | ---- | ---- |
|[CNNCTC文字识别](./inference/01_cnnctc/) | 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[ResNet50图像分类](./inference/02_resnet50/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[HDR图像增强](./inference/03_hdr/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[CycleGAN风格迁移](./inference/04_cyclegan/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Shufflenet图像分类](./inference/05_shufflenet/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[FCN图像语义](./inference/06_fcn/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Pix2Pix风格迁移](./inference/07_pix2pix/)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[机械臂分拣](./inference/08_ros2_robot_arm/)| 8.0.0beta1 | 2.6.0 | 20T24G |

### 第三方应用案例(community)

社区开发者贡献的离线推理，包含RingMoE遥感图像分类。

| 案例名称 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|
|[RingMoE遥感图像分类](./community/01_ringmoe_classification/)| 8.0.0.beta1 | 	2.6.0 | 20T24G |

## 贡献与反馈

1. **Issue**：欢迎各位开发者通过 [Issue](https://github.com/mindspore-lab/orange-pi-mindspore/issues) 提交建议或 bug 反馈

2. **Pull Request**: 开发者可发起 [PR](https://github.com/mindspore-courses/applications/pulls) 进行Bug修复或代码贡献（提交前请参考[提交规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)，由Committer @xing-yiren 及另一位Committer 完成评审合入），你的每一份参与都能让本项目更加完善。

3. **开源项目**：若开发者有符合条件的开源项目推荐/自荐，欢迎按照[项目仓规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)完善项目内容后，邮件至contact@public.mindspore.cn进行投稿，邮件标题请参考：`【昇思+香橙派项目投稿】项目名称`格式，并在正文中对项目进行简单介绍，附上代码仓链接。