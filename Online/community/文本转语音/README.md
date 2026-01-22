# text-to-audio 文字生成音频模型

## 项目介绍

本项目基于香橙派AIPro边缘计算硬件的低功耗、高算力特性，结合MindSpore开源框架的灵活部署能力，搭载先进的文字生成音频模型，实现从文本描述到自然流畅音频内容的精准生成，可快速将文字语义转化为清晰、有感染力的语音或场景音频片段。

## 环境准备

开发者拿到香橙派开发板后，需先完成硬件资源确认、镜像烧录及CANN和MindSpore版本升级，确保基础环境满足模型运行需求，具体配置如下：

### 基础环境配置

- 开发板：香橙派AIPro 或其他同等硬件规格的边缘计算开发板

- 开发板镜像：Ubuntu 系统镜像

- CANN Toolkit/Kernels：8.1.rc1

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

- soundfile

#### 核心TTS模型依赖
- tokenizers==0.19.1

#### 音频处理依赖

- soundfile==0.12.1

#### 数据处理依赖
- numpy==1.26.4

- pandas==2.3.3


## 快速使用

1. 确保上述环境及依赖包配置完成后，修改如下代码块中中的数据
~~~ python
samplerate = 24000
text = ["欢快地奔跑的小溪，像清澈的画卷"]
voice_preset = None
~~~
可修改要生成的文本内容，以及所使用的音色（遵循Bark模型仓库所支持的音色列表）

2. 修改后，继续运行后续代码块。

3. 程序运行后会生成对应的音频文件，并播放笔记本中

4. 输出路径：指定生成音频的保存路径（./output.wav）

5. 随后查看输出即可
![alt text](image.png)
![alt text](image-1.png)

## 注意事项

- 若运行过程中出现内存不足提示，可关闭开发板上其他占用资源的程序，或降低音频采样率、缩短生成时长。