<div align=center>
  <h1>昇思+昇腾开发板：软硬结合玩转minGPT实战<br>实验指导手册</h1>
</div>

## 项目简介

![minGPT](./images/4.png)

本项目是将Karpathy的[minGPT项目](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/training/02-minGPT)迁移至MindSpore框架的实现，包括训练和推理全流程。由于目前大多数可用的GPT模型实现都略显杂乱，minGPT力求小巧、简洁、易懂且具有教育意义。GPT并不是一个复杂的模型，此实现大约有 300 行代码（见`mingpt/model.py`）。

## 实验环境准备

本章节将介绍如何在OrangePi AIpro-20t上烧录镜像，通过PC远程连接昇腾开发板配置运行环境，并自定义安装CANN和MindSpore。
- 开发板：OrangePi AI pro-20T
- 操作系统镜像：opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img.xz
- Python：3.9
- CANN：8.3RC1
-  MindSpore：2.7.1

本章节所需的软/硬件如下：

- 硬件：昇腾开发板、PC（个人笔记本电脑）、电源线、HDMI线、显示器、鼠标、键盘、读卡器、USB Type-C 数据线（可选）
- 软件：balenaEtcher制卡工具、Vscode、MobaXterm（可选）

### 镜像烧录

镜像烧录步骤请参考[香橙派环境设置](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)。

OrangePi AIpro 20T开发板镜像下载：[官方链接]((http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html))，推荐使用镜像：`opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img.xz`。

### 版本检测

请按照以下说明分别检测CANN以及MindSpore版本，若不满足要求，请按照[香橙派环境设置](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)对应内容进行升级。

#### CANN版本查询

1. 打开终端，确认为`HwHiAiUser`用户身份（可通过提示符“@”前的用户名验证）

    ![香橙派终端](./images/1.png)

2. 执行以下命令查询CANN版本信息：

    ```bash
    # 查询CANN版本信息
    cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
    ```

3. 版本不满足需求时，参考[香橙派环境设置](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)的**CANN升级**章节完成升级。


#### MindSpore版本查询

1. 确认终端为`HwHiAiUser`用户身份

    ![香橙派终端](./images/1.png)

2. 执行以下命令查询MindSpore版本信息：

    ```bash
    # 查询MindSpore版本信息
    pip show mindspore
    ```

    ![MindSpore](./images/3.png)

3. 若当前MindSpore版本不满足开发需求，可按照[香橙派环境设置](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)的**MindSpore升级**章节完成升级。

## Quick Start

### 项目结构介绍

```text
02_mingpt/
├── assessments/
│    ├── model-question.py    # GPT网络架构知识点考察代码（填空）
│    └── trainer-question.py  # 混合精度下的训练器知识点考察代码（填空）
├── code
│    ├── mingpt/
│    │    ├── __init__.py     # 包标识文件，定义mingpt模块
│    │    ├── bpe.py          # 简单的BPE编码器实现
│    │    ├── model.py        # 模型文件：包含Attention、Decoder Block及GPT实现
│    │    ├── trainer.py      # 训练器文件：配置O2模式混合精度训练
│    │    └── utils.py        # 工具类实现
│    ├── demo.py              # GPT-nano数字排序训练与推理全流程
│    ├── generate.py          # 加载预训练模型权重执行推理
│    └── run_demo.sh          # 自动化脚本：拉起demo.py完成训推流程
└── docs
    ├── images/               # 实验手册配图目录
    ├── 昇思+昇腾开发板：软硬结合玩转minGPT开发实战.pdf  # 课程PPT
    └── mingpt-practice-guide.md                      # 实验指导手册

```

### 任务介绍（`demo.py`）

示例中的任务是一个简单的根据输入的乱序数字进行从小到大排序的任务，依次执行：

1. [数据集构建](#数据集构建)
2. [模型与Trainer实例化](#模型与trainer实例化)
3. [模型训练](#模型训练)
4. [模型评估](#模型评估)
5. [模型推理](#模型推理)

#### 数据集构建

排序数列的长度与数值范围通过length和num_digits参数定义，默认配置下，模型将对长度为6、数值范围[0,2]的乱序数列进行排序。

需遵循“右移（shift right）”原则构建输入输出：输入由 **乱序数列 + 排序数列[:-1]** 组成，目标输出由 **掩码 + 排序数列** 组成。

对应`demo.py`中代码片段如下：

```python
# 自定义排序数据集类，用于生成乱序数字序列与对应排序标签，满足GPT模型输入格式要求
class SortDataset():
    """
    排序问题”的数据集。例如，对于问题长度为 6 的情况：
    输入：0 0 2 1 0 1 -> 输出：0 0 0 1 1 2
    这将被作为输入传递给transformer，并以以下形式进行拼接：
    输入： 0 0 2 1 0 1 0 0 0 1 1
    输出：I I I I I 0 0 0 1 1 2
    其中 I 表示“掩码”，因为转换器正在读取输入序列。
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    ...
```

#### 模型与Trainer实例化

代码将根据配置，调用`model.py`与`trainer.py`中定义的类，完成模型与训练器的实例化。

对应`demo.py`中代码片段如下：

```python
# 实例化GPT模型
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# 实例化trainer并开始训练
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4
train_config.max_iters = 2000
train_config.num_workers = 1
trainer = Trainer(train_config, model, train_dataset)
```

#### 模型训练

定义批次结束回调函数（callback），用于打印训练日志，随后启动模型训练。

对应`demo.py`中代码片段如下：

```python
def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)
```

#### 模型评估

将训练完成的模型在测试数据集上进行评估，统计排序任务的正确率。

对应`demo.py`中代码片段如下：

```python
# 开始验证
model.set_train(False)

def eval_split(split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length
    results = []
    ...
    rt = mindspore.tensor(results, dtype=mindspore.float32)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )

    return rt.sum()

# 对模型进行大量来自训练集和测试集的数据处理，并验证输出结果的正确性。
train_score = eval_split('train', max_batches=50)
test_score  = eval_split('test',  max_batches=50)
```

#### 模型推理

随机生成一条乱序数字序列，输入训练后的模型，验证其排序效果。

对应`demo.py`中代码片段如下：

```python
# 随机生成一条序列让模型进行推理
n = train_dataset.length
inp = mindspore.tensor([[0, 0, 2, 1, 0, 1]], dtype=mindspore.int32)
assert inp[0].nelement() == n
cat = model.generate(inp, n, do_sample=False)
sol = ops.sort(inp[0])[0]
sol_candidate = cat[:, n:]
print('input sequence  :', inp.tolist())
print('predicted sorted:', sol_candidate.tolist())
print('gt sort         :', sol.tolist())
print('matches         :', bool((sol == sol_candidate).all()))
```

### 运行demo脚本

请按照以下步骤执行demo脚本（假设已进入orange-pi-mindspore项目根目录）：

1. 打开终端进入minGTP文件夹中。

    ```bash
    cd courses/02_mingpt/code
    ```

2. 执行run_demo.sh脚本。

    ```bash
    bash run_demo.sh
    ```

    该脚本将自动配置实验所需环境变量，并启动`demo.py`执行训练与推理全流程。

3. 若实验验证成功，则会输出以下信息：

    ![输出](./images/5.png)

## 知识点考察

### 题目介绍

我们在[02_mingpt/assessments/](../assessments/)路径下设计了两个知识点考察脚本，采用**关键代码挖空**形式，需补齐代码确保流程完整运行，通过实操深化核心知识点理解：

- `model-question.py`：GPT网络架构知识点考察（聚焦Attention、GPT核心结构、解码机制等核心模块）
- `trainer-question.py`：混合精度训练器知识点考察（聚焦MindSpore函数式编程、loss缩放、精度溢出检测等关键实现）

挖空示例如下：

```python
# 因果自注意力；自注意力：(B, nh, T, hs) 与 (B, nh, hs,T) 相乘 -> (B, nh, T, T)
# >>>>>>> 填空1 完成attention计算 <<<<<<<
att = _____
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
att = manual_softmax(att, dim=-1)
att = self.attn_dropout(att)
```

### 能力自检步骤

1. 完成两个考察脚本的代码填空；
2. 删除脚本文件名中的`-question`后缀，生成`model.py`和`trainer.py`；
3. 将新生成的两个文件，替换`02_mingpt/code/mingpt/`路径下的原始文件；
4. 参考[运行demo脚本](#运行demo脚本)章节的步骤，启动训练与推理；
5. 若运行结果与原始脚本一致（排序正确率正常），则说明填空内容正确。

## 拓展实验

在掌握基础流程后，开发者可参考如下指南，更换不同模型规格，或自己的数据集进行拓展实验。

### 自定义模型初始化

`demo.py`默认实例化gpt-nano小型模型，可通过修改模型配置参数，实例化其他规格的GPT模型（如GPT-2 124M参数版本）。

这里以实例化一个GPT-2模型（124M参数版本）为例：

```python
from mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai的模型词表长度
model_config.block_size = 1024  # openai的模型输入长度
model = GPT(model_config)
```

注意事项：

- 模型参数越大，对开发板的内存和算力要求越高，建议从中小规格模型开始尝试；
- `vocab_size`（词表长度）和`block_size`（序列长度）需与目标任务匹配，不可随意设置。

## 自定义数据集

`demo.py`默认使用随机生成的排序数据集，可替换为自定义数据集（如文本生成、序列预测等任务），核心要求是遵循MindSpore的数据加载范式。

```python
# 数据集应使用MindSpore的范式处理，详细信息请参考
# https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/beginner/dataset.html
train_dataset = YourDataset()

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # 学习率
train_config.max_iters = 1000     # 最大迭代次数
train_config.batch_size = 32      # 数据集batch大小
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
```

注意事项：
- 自定义数据集需遵循MindSpore数据加载范式，建议参考[MindSpore数据集开发指南](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/beginner/dataset.html)；
- 文本类任务需先通过tokenization处理数据（可复用`mingpt/bpe.py`中的实现）；
- 需根据数据集规模和开发板性能，调整batch_size、max_iters等训练参数。
