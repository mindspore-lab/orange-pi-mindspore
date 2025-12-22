<div align=center>
  <h1>基于昇思+香橙派开发板的带框架应用实践案例</h1>
</div>

本路径下包含基于昇思MindSpore的香橙派开发板带框架应用案例。

## 模型案例清单和版本兼容

### 推理案例（inference）

官方开发的推理案例，涵盖CV、NLP、GAN、大模型等领域的经典模型。

| 模型/案例 | CANN | MindSpore | 开发板型号 |
| :----- |:----- |:----- |:-----|
|[手写数字识别](./inference/01_quick_start/) | 8.1.RC1  | 2.6.0| 8T8G |
|[ResNet50](./inference/02_resnet50/) | 8.1.RC1  | 2.6.0| 8T8G |
|[ViT](./inference/03_vit/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[FCN](./inference/04_fcn/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ShuffleNet](./inference/05_shufflenet/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[SSD](./inference/06_ssd/)| 8.1.RC1  | 2.6.0| 8T8G |
|[RNN](./inference/07_rnn/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[LSTM+CRF](./inference/08_lstm_crf/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[GAN](./inference/09_gan/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DCGAN](./inference/10_dcgan/)|  8.1.RC1  | 2.6.0| 8T8G |
|[Pix2Pix](./inference/11_pix2pix/)|  8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Diffusion](./inference/12_diffusion/)|  8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ResNet50迁移学习](./inference/13_resnet50_transfer/)|  8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Qwen1.5-0.5b](./inference/14_qwen1_5_0_5b/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[TinyLlama-1.1B](./inference/15_tinyllama/)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DctNet](./inference/16_dctnet/)  | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DeepSeek-R1-Distill-Qwen-1.5B](./inference/17_deepseek_r1_distill_qwen_1_5b/)  | 8.1.RC1.beta1  | 2.6.0| 20T24G |
|[DeepSeek-Janus-Pro-1B](./inference/18_deepseek_janus_pro_1b/)  | 8.0.0beta1 | 2.5.0| 20T24G |
|[MiniCPM3-4B](./inference/19_minicpm3/)  | 8.0.0beta1 | 2.5.0| 20T24G |

### 训练案例（training）

官方开发的训练案例，包含BERT预训练任务。

| 模型/案例 | CANN | MindSpore | 开发板型号 |
| :----- |:----- |:----- |:-----|
| [BERT](./training/01_bert/) | 8.1.RC1  | 2.6.0| 20T24G |

### 第三方应用案例(community)

社区开发者贡献的应用案例，涵盖文档问答、图像识别、文本分类、翻译、视频分类等多模态任务。

| 模型/案例 | 训练/推理 | CANN | MindSpore | 开发板型号 |
| :----- |:----- |:----- |:-----|:-----|
| [Token Classification](./community/01_token_classification/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [Sentence Similarity](./community/02_sentence_similarity/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [Image To Text](./community/03_image_to_text/) | 推理 | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [Text Ranking](./community/04_text_ranking/) | 推理 | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [Feature Extraction](./community/05_feature_extraction/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [Table Question Answering](./community/06_table_question_answering/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [Image Classification](./community/07_image_classification/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [Text Classification](./community/08_text_classification/) | 推理 | 8.0.0.beta1  |2.6.0  |20T24G  |
| [Summarization](./community/09_summarization/) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [Translation](./community/10_translation/) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [Object Detection](./community/11_object_detection/) | 推理 | 8.0.0.beta1   |2.6.0  |8T16G  |
| [Video Classification](./community/12_video_classification/) | 推理 | 8.0.0.beta1  | 2.6.0 |8T16G |
| [Mask Generation](./community/13_mask_generation/) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [Document Question Answering](./community/14_document_question_answering/) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |

## 贡献与反馈

1. **Issue**：欢迎各位开发者通过 [Issue](https://github.com/mindspore-lab/orange-pi-mindspore/issues) 提交建议或 bug 反馈

2. **Pull Request**: 开发者可发起 [PR](https://github.com/mindspore-courses/applications/pulls) 进行Bug修复或代码贡献（提交前请参考[提交规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)，由Committer @xing-yiren 及另一位Committer 完成评审合入），你的每一份参与都能让本项目更加完善。

3. **开源项目**：若开发者有符合条件的开源项目推荐/自荐，欢迎按照[项目仓规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)完善项目内容后，邮件至contact@public.mindspore.cn进行投稿，邮件标题请参考：`【昇思+香橙派项目投稿】项目名称`格式，并在正文中对项目进行简单介绍，附上代码仓链接。