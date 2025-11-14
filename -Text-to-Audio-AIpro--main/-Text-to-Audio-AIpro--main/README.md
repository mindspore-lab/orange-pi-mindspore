# text-to-audio 文字生成音频模型

## 项目介绍

本项目基于香橙派AIPro边缘计算硬件的低功耗、高算力特性，结合MindSpore开源框架的灵活部署能力，搭载先进的文字生成音频模型，实现从文本描述到自然流畅音频内容的精准生成，可快速将文字语义转化为清晰、有感染力的语音或场景音频片段。

## 环境准备

开发者拿到香橙派开发板后，需先完成硬件资源确认、镜像烧录及CANN和MindSpore版本升级，确保基础环境满足模型运行需求，具体配置如下：

### 基础环境配置

- 开发板：香橙派AIPro 或其他同等硬件规格的边缘计算开发板

- 开发板镜像：Ubuntu 系统镜像

- CANN Toolkit/Kernels：8.0.0.beta1

- MindSpore：2.6.0

- MindSpore NLP：0.4.1

- Python：3.9

### 环境搭建步骤

1. **镜像烧录**：需烧录香橙派官网提供的Ubuntu镜像，详细烧录流程参考昇思MindSpore官网「香橙派开发专区 - 环境搭建指南 - 镜像烧录」章节。
      

2. **CANN升级**：完成镜像烧录后，按照昇思MindSpore官网「香橙派开发专区 - 环境搭建指南 - CANN升级」章节的操作步骤完成CANN版本升级。
      

3. **MindSpore升级**：遵循昇思MindSpore官网「香橙派开发专区 - 环境搭建指南 - MindSpore升级」章节的说明，将MindSpore及相关组件升级至指定版本。
      

## 依赖包配置

创建requirements.txt文件，复制以下内容后，通过命令`pip install -r requirements.txt`安装所需依赖：

- Python == 3.9

- MindSpore == 2.6.0

- mindnlp == 0.4.1

- pillow == 11.3.0

- sympy == 1.14.0

#### 核心TTS模型依赖
- tokenizers==0.19.1

- sentencepiece

#### 音频处理依赖
- librosa==0.11.0

- soundfile==0.12.1

- pydub==0.25.1

- ffmpy==0.6.4

- audioread==3.1.0

#### 数据处理依赖
- numpy==1.26.4

- scipy==1.15.3

- pandas==2.3.3

#### 可选（中文TTS/可视化）
- pypinyin==0.55.0

- matplotlib==3.10.7


## 快速使用

1. 确保上述环境及依赖包配置完成后，修改如下代码块中中的数据
~~~ python
infer_input = {
    "text": "李白，字太白，号青莲居士，唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。他的诗歌以豪放飘逸、想象丰富著称，代表作有《将进酒》《静夜思》《早发白帝城》等，深受人们喜爱。",
    "speaker_id": 0,
    "max_src_len": 1000
}
~~~


2. 修改后，直接运行项目根目录下的`synthesize_ST_reconstruct.ipynb`文件。

3. 程序运行后会生成对应的音频文件

原始文本: 李白，字太白，号青莲居士，唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。他的诗歌以豪放飘逸、想象丰富著称，代表作有《将进酒》《静夜思》《早发白帝城》等，深受人们喜爱。


音素序列: {l i3 b ai2 sp z ii4 t ai4 b ai2 sp h ao4 q ing1 l ian2 j v1 sh iii4 sp t ang2 d ai4 w uei3 d a4 d e5 l ang4 m an4 zh u3 y i4 sh iii1 r en2 sp b ei4 h ou4 r en2 y v4 w uei2 sp sh iii1 x ian1 sp t a1 d e5 sh iii1 g e1 y i3 h ao2 f ang4 p iao1 y i4 sp x iang3 x iang4 f eng1 f u4 zh u4 ch eng1 sp d ai4 b iao3 z uo4 y iou3 sp q iang1 j in4 j iou3 sp j ing4 y ie4 s ii1 sp z ao3 f a1 b ai2 d i4 ch eng2 sp d eng3 sp sh en1 sh ou4 r en2 m en5 x i3 ai4 sp}
语音合成完成！结果保存至：./output/result/AISHELL3

4. 输出路径：指定生成音频的保存路径（./output/result/AISHELL3）

5. 随后查看输出即可

## 注意事项

- 该模型需要提前准备两个参数已经转换好的文件，分别是 1. FastSpeechTTS_SF\output\ckpt\AISHELL3\600000.ckpt  2.FastSpeechTTS_SF\hifigan\generator_universal.ckpt

- 生成音频的时长、采样率、音量可在`synthesize_ST_reconstruct.ipynb`文件的参数配置区进行调整，人声朗读还可设置语速、音色，修改后需重新运行程序。

- 若运行过程中出现内存不足提示，可关闭开发板上其他占用资源的程序，或降低音频采样率、缩短生成时长。