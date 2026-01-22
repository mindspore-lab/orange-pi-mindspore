# Orange-Pi AI Pro Multimodal Korean  


## 基于昇腾 AI 的多模态韩语应用

利用 **“万卷·丝路”** 开源图文语料库，对 **Qwen2-VL-2B** 模型在服务器端进行 LoRA 微调图文对话模型，并基于mindspore、mindnlp、CANN和gradio部署于 **OrangePi AIpro**（20 TOPS Ascend SoC）。项目提供：

1. 🖼️+📝**图文联合问答** 微调基于LLamafactory微调
2. 微调后模型在**orangepi aipro**gradio界面

> 适用于端侧低资源小语种的昇腾算力 AI 场景。

---

## 主要特性

| 模块 | 说明 |
| ---- | ---- |
| **底座模型** | `Qwen2-VL-2B-Instruct`|
| **数据集** | [万卷·丝路](https://opendatalab.com/OpenDataLab/WanJuanSiLu2O)（韩语） |
| **训练框架** | LLaMA-Factory 0.9.4.dev0 /  | 
| **部署平台** | OrangePi AIpro（Ascend 20 TOPS，24 GB RAM） |
| **微调方法** | LoRA + SFT |

---

## 环境准备

### 服务器端（训练）

| 硬件 | 规格 |
| ---- | ---- |
| GPU  | NVIDIA A100 80 GB × 1 |
| CPU  | 32 cores |
| RAM  | 224 GB |

| 软件 | 版本 |
| ---- | ---- |
| OS   | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.7.2 + CUDA 12.2 |
| Deepspeed | 可选（多卡） |

#### llamafactory安装流程（参考官方github，这里加了国内源）  
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    conda create -n llama_factory python=3.10
    conda activate llama_factory
    cd LLaMA-Factory
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[metrics]

---

### Edge 端（OrangePi AIpro）

#### 环境准备

    MindSpore       2.6.0
    MindNLP         0.4.1
    CANN Toolkit    8.1.RC1
    Python          3.9
    gradio          4.4.0

---

## 🔧 微调流程（如不需要微调可直接调到部署步骤）

具体流程代码和步骤参考`ko_fintune.ipynb`模块

#### 1. 下载底座模型  

       git lfs install
       git clone https://www.modelscope.cn/Qwen/Qwen2-VL-2B-Instruct.git models/Qwen2-VL-2B-Instruct

#### 2. 添加数据集描述（编辑 `LLaMA-Factory/data/dataset_info.json`）  

       "ko_train": {
         "path": "data/ko_train.json",
         "type": "sharegpt_multi_modal"
       },
       "ko_val": {
         "path": "data/ko_val.json",
         "type": "sharegpt_multi_modal"
       }

#### 3. 启动 WebUI  

       llamafactory-cli webui

#### 4. 关键参数示例 

   | 选项 | 值 |
   | ---- | -- |
   | Model name  | Qwen2-VL-2B-Instruct |
   | Model path  | models/Qwen2-VL-2B-Instruct |
   | Finetune    | LoRA |
   | Stage       | Supervised Fine-Tuning |
   | Dataset     | ko_train |
   | Max epochs  | 3 |
   | Batch size  | 16 |
   | Save steps  | 200 |
   | lora_rank   | 64 |
   | lora_alpha  | 128（一般是rank的两倍） |
   | lora_dropout | 0.05（防止过拟合） |
   | Output dir  | saves/Qwen2-VL/lora/Qwen2-VL-sft-ko |

### 5. 监控显存  
       watch -n 1 nvidia-smi

### 6. 训练结果 
   30000张数据集使用单张 A100 微调约 **10 h**。
   主要通过**Bleu与Rouge**系数评估，训练前后Bleu与Bouge均提升**30%以上**：
   #### 训练前：
   ![alt text](pictures/orin.png)
   #### 训练后：
   ![alt text](pictures/lora.png)
   #### loss曲线
   ![qwen](pictures/qwenvl2-2B-loss.png)

---

### 7.合并 LoRA & 导出

在 WebUI **Expert** 标签执行  

    Model path      = models/Qwen2-VL-2B-Instruct
    Checkpoint path = saves/Qwen2-VL/lora/Qwen2-VL-sft-ko
    Export path     = models/qwen2ko_final

点击“开始导出”，得到合并权重。

---

## 📦 边缘端部署（如不需要微调可以直接部署原始的qwen2-vl-2B）
### orangepi aipro环境准备
开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

开发板：香橙派Aipro 20T 24G 
开发板镜像: Ubuntu镜像  
CANN Toolkit/Kernels：8.1.RC1  
MindSpore: 2.6.0  
MindSpore NLP: 0.4.1  
Python: 3.9

### 镜像烧录
运行该案例需要烧录香橙派官网ubuntu镜像，烧录流程参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--镜像烧录](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html) 章节。

### CANN升级
CANN升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--CANN升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### MindSpore升级
MindSpore升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--MindSpore升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### 应用代码

1. 将 `models/qwen2ko_final` 拷贝至 OrangePi AIpro  
2. 参考 `ko_infer.ipynb` 进行前端展示与推理测试  
   
### 预期结果
   多模态韩语图文问答，gradio前端交互
   结果示例：
   ![alt text](pictures/result.png)





