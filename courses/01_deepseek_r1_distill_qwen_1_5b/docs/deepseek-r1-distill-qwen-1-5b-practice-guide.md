# <center>昇思+昇腾开发板：</center>

# <center>软硬结合玩转DeepSeek蒸馏模型开发实战</center>

本教程将介绍如何在以香橙派为代表的昇腾开发板上，使用DeepSeek-R1-Distill-Qwen-1.5B模型，基于昇思MindSpore进行动态图开发，包括环境准备、模型开发、模型微调、模型推理、推理性能优化全流程。

**代码目录:**

```
01_deepseek_r1_distill_qwen_1_5b/
├── assessments/
│    ├── beginner_assessment.ipynb     # 开发板初级能力认证知识点考察代码（填空）
│    └── intermediate_assessment.ipynb # 开发板中级能力认证知识点考察代码（填空）
├── code
│    ├── deepseek-r1-distill-qwen-1.5b-gradio.py # 基于Gradio的模型推理界面
│    ├── deepseek-r1-distill-qwen-1.5b-jit.py    # 基于Gradio的模型推理界面（jit优化）
│    ├── deepseek-r1-distill-qwen-1.5b-lora.py   # 基于LoRA的模型微调
├── docs
│    ├── images/
│    ├── 昇思+昇腾开发板：软硬结合玩转DeepSeek蒸馏模型开发实战.pdf  # 本章内容课件
└──  └── DeepSeek-R1-Distill-Qwen-1.5b全流程手册.md           # 本章内容手册
```
**手册目录:**
- [环境准备](#一-环境准备)
  - [镜像烧录及CANN和MindSpore的升级](#1-镜像烧录及cann和mindspore的升级)
  - [Gradio安装](#2-gradio安装)
- [模型开发](#二-模型开发)
    - [环境准备](#1-环境准备)
    - [执行ut进行验证](#2-执行ut进行验证)
    - [报错分析](#3-报错分析)
- [模型微调](#三-模型微调)
  - [数据集介绍](#1-数据集介绍)
  - [环境准备](#2-环境准备)
  - [模型微调](#3-模型微调)
- [模型推理](#四-模型推理)
  - [实验环境](#1-实验环境)
  - [加载预训练权重推理](#2-加载预训练权重推理)
  - [禁用多线程](#3-禁用多线程)
  - [加载LoRA权重推理](#4-加载LoRA权重推理)
- [推理jit优化](#五-推理jit优化)
  - [实验环境](#1-实验环境-1)
  - [执行推理测试](#2-执行推理测试)
- [附录](#附录)
  - [报错信息汇总以及修改方案](#1-报错信息汇总以及修改方案)

<div style="page-break-after: always;"></div>

## 一. 环境准备

首先是环境准备，本章节将介绍如何在OrangePi AIpro(20T)开发板上进行环境准备，涵盖镜像烧录、CANN与MindSpore版本核验及安装、虚拟内存（Swap）分区配置、Gradio及相关依赖安装。

- 开发板：OrangePi AIpro(20T)
- 操作系统镜像：opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img.xz
- Python：3.9
- CANN：8.1RC1
-  MindSpore：2.5.0
-  MindSpore NLP：0.4分支（源码安装）

本章节所需的软/硬件如下：

- 硬件：昇腾开发板、PC（个人笔记本电脑）、电源线、HDMI线、显示器、鼠标、键盘、读卡器、USB Type-C 数据线（可选）
- 软件：balenaEtcher制卡工具、Vscode、MobaXterm（可选）

OrangePi AIpro(20T)规格昇腾开发板参考图：
<div align="center">
	<img src="./images/image_1_1.png" width="500" />
</div>

### 1. 镜像烧录及CANN和MindSpore的升级

请参考[昇思官网香橙派环境搭建指南](https://www.mindspore.cn/tutorials/zh-CN/r2.5.0/orange_pi/environment_setup.html)

OrangePi AIpro(20T)规格昇腾开发板镜像下载请看[此链接](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro-20T.html)，使用镜像：`opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img.xz`



### 2. Gradio安装

打开终端，输入如下命令，安装Gradio 4.44.0
```bash
pip uninstall gradio -y
pip install gradio==4.44.0
```

<div style="page-break-after: always;"></div>

## 二. 模型开发

DeepSeek-R1-Distill-Qwen系列模型由Qwen2蒸馏裁剪生成，在云侧适配完成的基础上，本章节介绍模型在开发板上适配流程、常见问题及解决方案，为开发者适配其他模型提供借鉴。

**当前MindSpore NLP 0.4分支已完成了Qwen2模型在昇腾开发板的适配，模型源码位于`mindnlp/transformers/models/qwen2`目录。**

1. 如开发者希望直接体验模型在开发板上的微调、推理与性能优化，可直接跳转至[**模型微调**](#三-模型微调)章节，直接从源码安装0.4分支的MindSpore NLP；
2. 如开发者希望学习模型在开发板的适配流程，掌握常见问题的解决方案，可阅读本节剩余内容。

### 1. 环境准备

克隆MindSpore NLP并安装相关依赖:

```bash
git clone https://github.com/mindspore-lab/mindnlp.git
# 如克隆遇到网络问题，可更改为以下命令：
# git clone https://gitee.com/mindspore-lab/mindnlp.git
cd mindnlp
pip install -r requirements/requirements.txt
```

### 2. 执行ut进行验证

为方便问题定位, 在文件tests/transformers/models/qwen2/test_modeling_qwen2.py的`import mindspore`之后的位置，加入如下代码

```bash
mindspore.set_context(pynative_synchronize=True)
```

设置环境变量：

```bash
export RUN_SLOW=True
```

执行用例（以qwen2为例）：

```bash
pytest -v -s tests/transformers/models/qwen2/test_modeling_qwen2.py
```

### 3. 常见报错分析

在执行ut后可能会遇到报错，下面针对在Qwen2模型适配中三个典型的报错进行分析并给出修改步骤，其他具体报错汇总详见[附录](#附录)。

#### 3.1 针对算子缺失报错进行分析处理

> 测试用例tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_new_cache_format_0的报错信息如下：

![image_2_1.png](./images/image_2_1.png)

**分析：** OrangePi AIpro开发板当前CANN版本不支持aclnn的cumsum，切换成aclop算子进行执行，Tensor.cumsum => ops.cumsum(input, dim, dtype=None) , 且注意其中的input不支持int64，要通过int()转换为int32

- 修改前代码：mindnlp\transformers\models\qwen2\modeling_qwen2.py

![image_2_2.png](./images/image_2_2.png)

- 修改后代码：

![image_2_3.png](./images/image_2_3.png)


#### 3.2 针对损失函数报错进行分析处理

> 测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_Qwen2_sequence_classification_model的报错信息如下：

![image_2_4.png](./images/image_2_4.png)

**分析：** OrangePi AIpro开发板当前CANN版本不支持CrossEntropyLoss，需要修改成支持的mindspore.ops.SoftmaxCrossEntropyWithLogits接口。mindspore.ops.SoftmaxCrossEntropyWithLogits使用one-hot编码获取预测值和真实之间的softmax交叉熵，并且要求输入的预测值和真实值的数据类型相同。同时因为CrossEntropyLoss默认计算输出元素的加权平均值，所以最后计算输出loss的加权平均值。

- 修改前代码：mindnlp\transformers\models\qwen2\modeling_qwen2.py

<div align="center">
    <img src="./images/image_2_5.png" width=450/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_2_6.png" width=450/>
</div>

#### 3.3 针对香橙派上Tensor索引/切片报错进行分析处理

> 测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_beam_search_generate_dict_outputs_use_cache的报错信息如下：

<div align="center">
    <img src="./images/image_2_7.png" width=550/>
</div>

**分析：** OrangePi AIpro开发板上Tensor的切片赋值目前只支持直接通过mindspore.Tensor的方式

- 修改前代码： mindnlp\transformers\generation\beam_search.py

<div align="center">
    <img src="./images/image_2_8.png" width=400/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_2_9.png" width=400/>
</div>

<div style="page-break-after: always;"></div>

## 三. 模型微调

目前[MindSpore NLP仓0.4分支](https://github.com/mindspore-lab/mindnlp/tree/0.4)已在昇腾开发板上适配了Qwen2模型，本章节介绍如何在昇腾开发板上，基于MindSpore对DeepSeek-R1-Distill-Qwen-1.5B模型进行LoRA微调，使得模型可以模仿《甄嬛传》中甄嬛的口吻进行对话。微调示例代码参考[此处](../code/deepseek-r1-distill-qwen-1.5b-lora.py)


### 1. 数据集介绍

本次实践使用了huanhuan数据集，该数据集从《甄嬛传》的剧本进行整理，从原始文本中提取出将我们关注的角色的对话，并形成 QA 问答对，最终整理为json格式的数据，数据样本示例如下：

```text
[
    {
        "instruction": "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——",
        "input": "",
        "output": "嘘——都说许愿说破是不灵的。"
    },
    {
        "instruction": "这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。",
        "input": "",
        "output": "你们俩话太多了，我该和温太医要一剂药，好好治治你们。"
    },
    {
        "instruction": "嬛妹妹，刚刚我去府上请脉，听甄伯母说你来这里进香了。",
        "input": "",
        "output": "出来走走，也是散心。"
    }
]
```



### 2 环境准备

#### 2.1 MindSpore NLP安装

打开终端，输入如下命令，从Github对MindSpore NLP 0.4分支进行**源码安装**：

```bash
pip uninstall mindnlp -y
pip install git+https://github.com/mindspore-lab/mindnlp.git@0.4
# 检查是否安装成功
pip show mindnlp
```

如克隆遇到网络问题，可更改为以下命令：

```bash
pip uninstall mindnlp -y
pip install git+https://gitee.com/mindspore-lab/mindnlp.git@0.4
# 检查是否安装成功
pip show mindnlp
```


#### 2.2 openMind Hub Client安装

打开终端，输入如下命令，安装openMind Hub Client:
```bash
pip install openmind_hub
```

#### 2.3 限制python进程数

因OrangePi AIpro昇腾开发板内存与显存共享，且在执行时会拉起多python进程，导致额外的内存占用，从而影响到显存。故通过配置环境变量的方式，限制python进程数，从而减少对显存的影响。

打开终端，配置如下环境变量，约束限制python进程数。

```bash
export MAX_COMPILE_CORE_NUMBER=1
export TE_PARALLEL_COMPILER=1
```

配置后可在终端输入如下命令，检查环境变量是否生效：

```bash
# 如打印结果为1，则证明环境变量生效
echo $MAX_COMPILE_CORE_NUMBER
# 如打印结果为1，则证明环境变量生效
echo $TE_PARALLEL_COMPILER
```

### 3 模型微调

#### 3.1 下载数据集

在示例代码`deepseek-r1-distill-qwen-1.5b-lora.py`中，我们通过openmind_hub提供的接口下载huanhuan.json数据集：

```python
# 从魔乐社区下载数据集
om_hub_download(
    repo_id="MindSpore-Lab/huanhuan",
    repo_type="dataset",
    filename="huanhuan.json",
    local_dir="./",
)
```


#### 3.2 执行微调

微调的超参可在代码中的TrainingArguments进行配置，示例代码的超参介绍如下：

```python
args = TrainingArguments(
    output_dir="./output/DeepSeek-R1-Distill-Qwen-1.5B",  # 输出保存路径
    per_device_train_batch_size=1,  # batch size
    logging_steps=1,  # 每多少步记录一次训练日志
    num_train_epochs=1,  # epoch数
    save_steps=3,  # 每多少步保存一次权重
    learning_rate=1e-4,  # 学习率
)
```

在终端输入如下命令，启动微调：

```bash
python deepseek-r1-distill-qwen-1.5b-lora.py
```

第一次执行时，会涉及到模型预训练权重等相关文件的下载，故需要等待5-10分钟时间，下载后的文件可在同路径下的`.mindnlp/model/MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16`找到，后续执行无需再进行模型权重下载。

微调结果输出如下图所示：

![image_3_2_finetune](./images/image_3_2_finetune.png)


#### 3.3 查看保存权重

在[执行微调](#32-执行微调)章节中，我们配置了每3步保存一次权重，在执行完微调后，可在`./output/DeepSeek-R1-Distill-Qwen-1.5B`中找到`checkpoint-3`的文件夹，内有保存微调后的LoRA adapter权重。

<div align="center">
    <img src="./images/image_3_3_adapter_model.png" width=300/>
</div>


#### 3.4 清理缓存

建议在每次执行完毕后，在终端输入如下命令，清除缓存：

```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

通过`npu-smi info`命令可发现，执行前后，显存的占用从4393下降到了1622。

- 清理缓存前：
![image_3_4_before_drop_caches](./images/image_3_4_before_drop_caches.png)
- 清理缓存后：
  ![image_3_5_after_drop_caches](./images/image_3_5_after_drop_caches.png)

<div style="page-break-after: always;"></div>

## 四. 模型推理

本章节将对DeepSeek-R1-Distill-Qwen-1.5B模型进行推理，推理过程将转化为一个可交互的对话机器人，以增强用户体验和实用性。推理示例代码参考[此处](../code/deepseek-r1-distill-qwen-1.5b-gradio.py)

### 1. 实验环境

- CANN版本: 8.1.RC1
- MindSpore版本: 2.5.0
- MindSpore NLP版本：[MindSpore NLP仓0.4分支](https://github.com/mindspore-lab/mindnlp/tree/0.4)
- Gradio版本：4.44.0

### 2. 加载预训练权重推理

为记录推理时间，需要在代码运行前设置以下环境变量：
```sh
export INFERENCE_TIME_RECORD=True
```
启动推理
```sh
python deepseek-r1-distill-qwen-1.5b-gradio.py
```

代码正常运行的日志如下：

![image_4_1](./images/image_4_1.png)

运行代码后，在浏览器中打开127.0.0.1:7860开启对话，如下图所示：

<div align="center">
    <img src="./images/image_4_2.png" width=600/>
</div>

此时，从终端的运行日志可以看到，平均推理时间为0.727秒。
<div align="center">
    <img src="./images/image_4_3.png" width=550/>
</div>


### 3. 禁用多线程

MindSpore动态图下框架存在多线程行为，而OrangePi AIpro昇腾开发板由于host侧开销导致单token推理时间较长，所以在OrangePi AIpro昇腾开发板的场景中反而使用单线程能提升性能，在代码中加入禁用多线程的指令：
```python
from mindspore._c_expression import disable_multi_thread
disable_multi_thread()
```
再次运行代码，在浏览器中打开127.0.0.1:7860开启对话，如下图所示：

<div align="center">
    <img src="./images/image_4_4.png" width=500/>
</div>

应用上述改动后，平均推理时间减少至0.674秒
<div align="center">
    <img src="./images/image_4_5.png" width=470/>
</div>


### 4. 加载LoRA权重推理

因示例实验只微调了3个iteration，无法直观看出微调后效果，本环节旨在说明如何更换微调后的LoRA权重进行推理:  将`deepseek-r1-distill-qwen-1.5b-gradio.py`文件中第16行的注释取消（使用PeftModel），并修改加载adapter model的路径为微调后的`adapter_model`文件所在路径:
![image_4_6](./images/image_4_6.png)

再次启动推理
```sh
python deepseek-r1-distill-qwen-1.5b-gradio.py
```
打开对话界面，即可体验Lora微调后的对话效果。

## 五. 推理JIT优化

本章节将对DeepSeek-R1-Distill-Qwen-1.5B模型推理进一步优化，主要通过MindSpore JIT（Just-In-Time）编译技术优化DeepSeek-R1-Distill-Qwen-1.5B模型的推理性能，降低单次推理耗时，提升对话的响应速度与用户体验。JIT优化示例代码参考[此处](../code/deepseek-r1-distill-qwen-1.5b-jit.py)


### 1. 实验环境

- CANN版本: 8.1.RC1
- MindSpore版本: 2.5.0
- MindSpore NLP版本：[MindSpore NLP仓0.4分支](https://github.com/mindspore-lab/mindnlp/tree/0.4)
- Gradio版本：4.44.0

### 2. 执行推理测试

#### 2.1 推理脚本介绍

本次实验使用逐token推理的方式，使用DeepSeek-R1-Distill-Qwen-1.5B模型进行文本推理。常见的`model.generate()`推理方式是在此基础上进行了多种封装和优化，此处仅包含基础的推理与Top_P、温度的参数调节，故文本生成结果可能会与使用`model.generate()`推理存在一定出入。

##### 2.1.1 Top_p函数的实现

```python
def sample_top_p(probs, p=0.9):
    """
    Top-p采样函数，用于生成文本时选择下一个token。
    此处优先采用基于numpy而不是原生MindSpore的实现方式，因为在昇腾开发板上运行效率更高
    """
    probs_np = probs.asnumpy()
    # 按概率降序排序
    sorted_indices = np.argsort(-probs_np, axis=-1)
    sorted_probs = np.take_along_axis(probs_np, sorted_indices, axis=-1)
    # 计算累积概率并创建掩码
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / np.sum(sorted_probs, axis=-1, keepdims=True)
    # 转换回MindSpore Tensor
    sorted_probs_tensor = mindspore.Tensor(sorted_probs, dtype=mindspore.float32)
    sorted_indices_tensor = mindspore.Tensor(sorted_indices, dtype=mindspore.int32)
    next_token_idx = ops.multinomial(sorted_probs_tensor, 1)
    batch_size = probs.shape[0]
    batch_indices = ops.arange(0, batch_size, dtype=mindspore.int32).reshape(-1, 1)
    # 此处采用基于mindspore.ops的实现方式，在昇腾开发板上兼容性最好（基于mindspore.mint的实现方式当前版本CANN暂不支持）
    next_token = mindspore.ops.gather(sorted_indices_tensor, next_token_idx, axis=1, batch_dims=1)
    # next_token = mindspore.mint.gather(sorted_indices_tensor, dim=1, index=next_token_idx)
    return next_token
```


##### 2.1.2 MindSpore NLP库的修改

该实验对MindSpore NLP库中的`transformers/models/qwen2/modeling_qwen2.py`进行了修改以支持静态图的运行，**[相关PR](https://github.com/mindspore-lab/mindnlp/pull/2028)已经合入MindSpore NLP的0.4分支，进行实验时无需再进行以下修改。**

- modeling_qwen2.py的decoder_layer中，需添加_modules.values()
<div align="center">
    <img src="./images/image_5_1_modeling.png" width=270/>
</div>


- 修改`RotaryEmbedding`类以支持静态图
<div align="center">
    <img src="./images/image_5_2_modeling.png" width=450/>
</div>
<div align="center">
    <img src="./images/image_5_3_modeling.png" width=450/>
</div>



#### 2.2 运行JIT加速的推理脚本

执行启用了MindSpore JIT编译的脚本，在浏览器中打开127.0.0.1:7860开启对话，同时查看终端，对比性能提升：
```bash
python deepseek-r1-distill-qwen-1.5b-jit.py
```
脚本关键改动：
- 设置`jit_level`优化等级为`O2`。
- 使用`model.jit()`将全图静态图化。
- 使用`@mindspore.jit`装饰器封装模型推理函数`get_decode_one_tokens_logits`，设置PSJit选项解析python的ast以构建静态图。该函数用于逐个token进行推理。


输出示例：
<div align="center">
    <img src="./images/image_5_6_jit.png"/>
</div>
首token推理时间约为140秒，随后每个token推理时间约为0.27秒。
可以注意到，使用JIT加速后，单token推理速度有显著提升，但是在推理首个token前需要对全图进行编译，故首token推理时间较长。在推理token数量较多时，使用JIT优化对效率提升效果更明显。

<div style="page-break-after: always;"></div>


## 附录

### 1. 报错信息汇总以及修改方案

在进行[模型开发](#二-模型开发)的验证时，执行ut后可能还会遇到如下报错，下面针对每个报错进行分析修改。

#### 1.1 用例test_generate_from_inputs_embeds_decoder_only报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_generate_from_inputs_embeds_decoder_only，报错信息如下：

![image_6_1.png](./images/image_6_1.png)

**分析：** 排除input_ids.shape包含0的情况

- 修改前代码：mindnlp\transformers\models\qwen2\modeling_qwen2.py

<div align="center">
    <img src="./images/image_6_2.png" width=500/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_6_3.png" width=500/>
</div>

#### 1.2 用例test_Qwen2_token_classification_model报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_Qwen2_token_classification_model,报错信息如下：

<div align="center">
    <img src="./images/image_6_4.png" width=500/>
</div>

**分析：** Tensor.masked_fill(mask, value)中的value (Union[Number, Tensor]) - 用来填充的值，只支持零维Tensor或者Number。所以可以直接通过python中的float()将value的类型直接转换为number。

- 修改前代码：mindnlp\transformers\models\qwen2\modeling_qwen2.py

<div align="center">
    <img src="./images/image_6_5.png" width=370/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_6_6.png" width=370/>
</div>

#### 1.3 用例test_batching_equivalence等报错

> 执行用例以及对应报错信息如下：

![image_6_7.png](./images/image_6_7.png)

**分析：** 需要重新改写方法_prepare_4d_causal_attention_mask_with_cache_position

- 修改前代码：mindnlp\transformers\models\qwen2\modeling_qwen2.py

<div align="center">
    <img src="./images/image_6_8.png" width=530/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_6_9.png" width=530/>
</div>

#### 1.4 用例test_batching_equivalence报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_batching_equivalence，报错信息如下：

![image_6_10.png](./images/image_6_10.png)
![image_6_11.png](./images/image_6_11.png)
![image_6_12.png](./images/image_6_12.png)

**分析：** 针对算子mindspore.mint.max()需要区分入参dim是否为None的不同情况

- 修改前代码： \mindnlp\core\ops\reduction.py

<div align="center">
<img src='./images/image_6_13.png' width=370>
</div>


- 修改后代码：

<div align="center">
<img src='./images/image_6_14.png' width=370>
</div>


#### 1.5 用例test_constrained_beam_search_generate_dict_output报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_constrained_beam_search_generate_dict_output，报错信息如下:

![image_6_15.png](./images/image_6_15.png)

**分析：** 昇腾开发板上Tensor的切片赋值目前只支持直接通过mindspore.Tensor的方式

- 修改前代码：mindnlp\transformers\generation\beam_search.py

<div align="center">
    <img src="./images/image_6_16.png" width=500/>
</div>

- 修改后代码：

<div align="center">
    <img src="./images/image_6_17.png" width=500/>
</div>

#### 1.6 用例test_left_padding_compatibilit报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_left_padding_compatibilit，报错信息如下：

![image_6_18.png](./images/image_6_18.png)

**分析：** Tensor.cumsum => ops.cumsum(input, dim, dtype=None) , 且注意其中的input不支持int64，要通过int()转换为int32

- 修改前代码：\tests\transformers\generation\test_utils.py

![image_6_19.png](./images/image_6_19.png)

- 修改后代码：

![image_6_20.png](./images/image_6_20.png)

#### 1.7 用例test_sample_generate报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_sample_generate，报错信息如下：

![image_6_21.png](./images/image_6_21.png)

**分析：** 对于算子ops.stack()的入参要求具有相同数据类型。

- 修改前代码：\mindnlp\transformers\generation\logits_process.py

![image_6_22.png](./images/image_6_22.png)

- 修改后代码：

![image_6_23.png](./images/image_6_23.png)


#### 1.8 用例test_group_beam_search_generate报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2ModelTest::test_group_beam_search_generate，报错信息如下：

![image_6_24.png](./images/image_6_24.png)

**分析：** 对于算子ops.tensor_scatter_update()的入参要求具有相同数据类型。

- 修改前代码： \mindnlp\core\ops\array.py

![image_6_25.png](./images/image_6_25.png)

- 修改后代码：

![image_6_26.png](./images/image_6_26.png)

注意同时修改 \mindnlp\core\ops\other.py

- 修改前代码：

<div align="center">
<img src='./images/image_6_27.png' width=500>
</div>

- 修改后代码：

<div align="center">
<img src='./images/image_6_28.png' width=500>
</div>

#### 1.9 用例test_model_450m_logits等报错

> 执行测试用例pytest -s -v tests\transformers\models\qwen2\test_modeling_qwen2.py::Qwen2IntegrationTest::test_model_450m_logits以及Qwen2IntegrationTest类中所有打标签@slow的用例时，报错信息如下：

![image_6_29.png](./images/image_6_29.png)

**分析：** huggingface镜像站中已经下线用例中对应的模型Qwen/Qwen2-450m-beta，参照huggingface transformers仓库中tag为v4.51.3版本中的对应用例修改为模型Qwen/Qwen2-0.5B以及相应的数据即可。

- 修改前代码：

<div align="center">
    <img src="./images/image_6_30.png" width=1000/>
</div>

- 修改后代码：

![image_6_31.png](./images/image_6_31.png)
