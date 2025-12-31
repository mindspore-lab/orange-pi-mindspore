# Object Detection

[【开源实习】针对任务类型Object Detection，开发可在香橙派AIpro开发板运行的应用](https://gitee.com/mindspore/community/issues/ICJ5UE)
任务编号：#ICJ5UE  

基于`MindSpore`框架和`google/owlvit-base-patch32`模型实现的Object Detection应用  

### 介绍
目标检测（Object Detection） 是计算机视觉中的核心任务之一，旨在在图像或视频中同时 定位和识别目标物体。与图像分类不同，它不仅输出类别标签，还需要给出目标在图像中的边界框（bounding box）。  
OwlViT（Open-vocabulary Vision Transformer）是 Google Research 提出的一种 **开放词汇目标检测模型**。  
与传统检测模型（如 Faster R-CNN、DETR 等）仅能检测预定义类别不同，OwlViT 能够通过 **自然语言描述（Text Prompt）** 进行任意目标检测。
换言之，用户无需在训练阶段指定固定的类别标签，只需在推理时输入提示词（如 *"a cat"*, *"a red car"*, *"a person wearing glasses"*），模型即可检测出符合描述的目标区域。
OwlViT 的核心思想是将 **视觉表示（Vision Embedding）** 与 **文本表示（Text Embedding）** 映射到同一语义空间，通过 **跨模态相似度匹配** 来完成开放类别检测任务。



### 环境准备

开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

开发板：香橙派Aipro或其他同硬件开发板  
开发板镜像: Ubuntu镜像  
`CANN Toolkit/Kernels：8.0.0.beta1`  
`MindSpore: 2.6.0`  
`MindSpore NLP: 0.4.1`  
`Python: 3.9`

#### 镜像烧录

运行该案例需要烧录香橙派官网ubuntu镜像，烧录流程参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--镜像烧录](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html) 章节。

#### CANN升级

CANN升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--CANN升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

#### MindSpore升级

MindSpore升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--MindSpore升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### requirements
```
Python == 3.9

MindSpore == 2.6.0

mindnlp == 0.4.1

pillow == 11.3.0

sympy  == 1.14.0

matplotlib == 3.9.4
```
## 快速使用

用户在准备好上述环境之后，逐步运行object_detection.ipynb文件即可，代码中模型加载部分会自动从huggingface镜像中下载模型。
使用时需将图片路径替换为要识别的图片路径，逐步运行后模型会返回检测结果。

## 预期输出
展示带有绘制检测结果的图像



