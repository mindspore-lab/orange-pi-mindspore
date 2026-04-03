# Arabic Multimodal Inference

基于`MindSpore`框架和`Qwen2-0.5B-Instruct`模型实现的阿拉伯语多模态数据处理与推理验证应用。该项目主要面向 **WanJuanSiLu2O** 阿拉伯语多模态数据集（包含音频与图像数据），在端侧设备上完成数据抽取、特征清单（Manifest）生成以及基于文本描述的模型推理验证。

### 环境准备

开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

开发板：香橙派AIpro或其他同硬件开发板\
开发板镜像: Ubuntu镜像\
`CANN Toolkit/Kernels：8.1RC1` （或更高版本）\
`MindSpore: 2.6.0`\
`MindSpore NLP: 0.4.1`\
`Python: 3.9`

#### 镜像烧录

运行该案例需要烧录香橙派官网ubuntu镜像，烧录流程参考 `https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html`  章节。

#### CANN升级

CANN升级参考 `https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html` 章节。

#### MindSpore升级

MindSpore升级参考 `https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html` 章节。

### requirements

```text
Python == 3.9 

MindSpore == 2.6.0 

mindnlp == 0.4.1 

openxlab 

modelscope 

huggingface_hub 

fastapi 

uvicorn 
```

## 快速使用

用户在准备好上述环境之后，逐步运行根目录下的 `mindspore_arabic_qwen2.ipynb` 文件即可。本案例代码中已经内置了一体化的处理流程控制：

1. **按需配置**：在 Notebook 顶部的“运行开关”模块中，用户可将 `INSTALL_MISSING`、`DOWNLOAD_MODEL`、`DOWNLOAD_DATA` 等参数设为 `True`，以自动安装缺失依赖和下载数据集。
2. **数据与模型下载**：代码支持自动从 OpenDataLab 下载 WanJuanSiLu2O 阿语子集，并从 ModelScope 拉取 `Qwen2-0.5B-Instruct` 模型。
3. **数据清洗与处理**：逐步运行后，脚本会自动解压音频文件，提取图像 URL 及阿语描述文本，并生成专门用于推理验证的 Manifest 文件（`.jsonl` 格式）。
4. **模型加载与推理**：自动初始化 MindSpore 运行环境（设为 PyNative 同步模式），并加载模型对提取的多模态阿语文本数据进行推理。

## 预期输出

1. **数据处理环节**：在 `outputs/` 目录下成功生成 `audio_demo/extracted_manifest_multimodal.jsonl` 和 `image_demo/image_manifest_url.jsonl`，并在控制台打印出提取的数据样本。
2. **推理验证环节**：控制台输出模型成功加载的日志，展示针对抽取的阿拉伯语数据的推理结果。
3. **Web 服务展示**：若运行了最后的 Web 服务单元格，系统将在本地基于 FastAPI 启动测试接口供进一步调用与验证。

