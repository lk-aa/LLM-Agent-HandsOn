# 从0到1训练大模型

在人工智能领域，大模型的崛起正掀起一股新的浪潮。随着技术的不断突破，大模型已逐渐成为AI发展不可或缺的重要基石。无论是在自然语言处理、计算机视觉，还是其他领域，大模型展现了前所未有的潜力。而随着计算能力的提升与行业需求的细化，**越来越多的岗位需要针对特定应用场景开发专业的大模型，比如编程助手、医疗诊断工具、法律专业大模型，甚至是用于保密环境的专属模型**。

我们训练的大模型名为 **MateConv**，以下是我们的训练概况：

| 模型名称     | 模型参数量级      | 数据量级                | 硬件资源               | 训练时间              | 训练成本                |
| ------------ | ----------------- | ----------------------- | ---------------------- | --------------------- | ----------------------- |
| **MateConv mini** | 0.02B<br>（两千万参数） | 约3.6G（分词器约1G、预训练约1.5G、微调约1.1G） | RTX 3090 x2           | 全流程约1天          | AutoDL租赁，约100元     |

## Part 1 数据收集与数据预处理

如何让传统文本转变成大模型能够识别的数据？

文字/段落  ==> chunkize ==> tokenizer（token）==> encoder[0,1,2,3,4]  ==> 词向量构成

### 1.1 大模型训练流程概述

从 **0到1** 训练大模型是一个复杂而系统的工程，需要涵盖从数据准备到模型部署的多个环节。以下是一个完整的流程框架：

| 流程            | 说明                                                                                           |
|:---------------:|--|
| **数据准备**        | 收集高质量、覆盖面广的训练数据，对其进行清洗、去噪和格式化处理。划分<br>训练集、验证集，并存储为高效读取的格式。这一步为模型提供了扎实的输入基础。 |
| **硬件与环境配置**  | 为模型训练准备高性能硬件（如 A800、A100 GPU），搭建分布式训练环境，<br>并优化深度学习框架的配置。这一步确保训练效率和稳定性。 |
| **分词器训练**      | 根据训练数据量和模型任务需求，选择适合的分词算法（如 BPE 或 <br>SentencePiece）。分词器决定了模型如何理解数据，是数据与模型的桥梁。 |
| **设计模型架构**    | 选择适合的模型结构（如 GPT、BERT），并配置参数量、层数、激活函数<br>等细节。对于大规模任务，可以结合领域特点定制模型。 |
| **预训练**        | 使用无监督任务从海量数据中提取通用知识，比如语言模型的自回归建模或<br>掩码建模。预训练的效果直接影响模型后续的微调能力。 |
| **意图对齐微调**        | 通过监督微调（SFT）或强化学习对齐（RLHF），让模型学习人类偏好，<br>避免输出无意义或有害内容。对齐步骤是模型实用化的关键。 |
| **特定优化微调**  | 在特定任务（如文本分类、问答）上微调模型，结合冻结与解冻层的策略进一步<br>优化性能，满足应用需求。                          |
| **模型量化**       | 通过剪枝、量化和知识蒸馏等技术优化模型，提高推理效率，降低计算与存储<br>成本，使模型更适合部署环境。                            |
| **部署与监控**      | 将模型部署到生产环境中，使用推理优化工具提升服务效率，同时通过实时<br>监控与用户反馈不断改进模型性能和可靠性。                     |

### 1.2 MateConv Mini 的数据收集与数据预处理

数据是大模型训练的核心，质量和数量直接影响模型效果。在我们进行数据准备时，我们要考虑到模型的具体用途、模型的规模以及模型的精度要求等等、这些都会影响我们对数据的收集和参考。在大模型进行训练时，数据来源主要有两种——
- 一种是公开的、已经收集整理好的数据集、主要来自于Huggingface上已开源的各种中英文/代码数据集
- 另一种则是存储于数据库或者干脆就是以文件形式存储的原始数据（raw data）, 例如：PDF行业报告等。

#### 1.2.1 MateConv Mini所使用的数据集

MateConv Mini所使用的语料都是开源数据集、以中文为主、并且以轻量、高质量、少流程为主要追求、希望能够在最小的数据量级内获得较好的成果。分词器数据虽小，但能满足小模型实验需求；预训练数据覆盖广泛且经过清洗和去重，确保了多样性和高质量，对小规模模型的泛化能力有良好支持；意图对齐数据集经过精细设计，能帮助模型对齐人类意图，提升微调阶段的表现。然而，分词器数据词表覆盖范围有限，可能在特定领域的适应性上存在不足，同时预训练数据和微调数据在深度领域信息上略显欠缺。整体而言，这些数据集为 MateConv Mini 提供了扎实的训练基础，但在扩展到更大模型或特定领域时，还需补充规模更大、领域更专的高质量数据。

| 流程       | 数据集名称 | 数据量级 | 数据概况 |
|:---------------:|:---------------------:|:-------------:|:-:|
| Tokenizer训练流程 | Huggingface开源<br>tokenizer_train.jsonl | 1G | **jsonl超短文本数据集**<br><br>开源的超小型词表数据集，总词表量只有6400，几乎是所有开源Tokenizer数据集中最小的|
| 预训练流程   | huggingface开源<br>pretrain_hq.jsonl | 1.5G | **jsonl短文本数据集**<br><br>数据集由来自**网页、百科、博客、问答、开源代码、书籍、报刊、专利、教材、考题等多种公开可获取的数据进行汇总清洗之后而形成的大语言模型预训练语料**。它将不同来源的HTML、TEXT、PDF、EPUB等各类格式的数据统一整理为JSONL格式，并进行了仔细的筛选、去重、清洗和**价值对齐**，能够作为我们的Mini版本模型的预训练语料。|
| 意图对齐微调   | ModelScope开源<br>匠数科技大模型SFT数据集<br> sft_mini_512.jsonl | 1.1G  | **jsonl对话（问答对数据）**<br><br>匠数大模型SFT数据集是从**网络上的公开数据源收集并整理得来**，经过细致的数据清洗、格式统一，最终获得了用于大模型SFT的包含10M条数据的中文数据集和包含2M条数据的英文数据集。(注：Mini 模型仅使用其中部分数据微调) |

#### 1.2.2 数据集展示与分析


```python
import json

# 文件路径
# NOTE r前缀表示原始字符串，这样可以避免反斜杠\被当作转义字符
file_paths = [
    r"..\..\LLM_By_Hand\dataset\tokenizer_train.jsonl",
    r"..\..\LLM_By_Hand\dataset\pretrain_hq.jsonl",
    r"..\..\LLM_By_Hand\dataset\sft_mini_512.jsonl"
]

# 定义读取并展示样本的函数
def read_and_display_samples(file_path, num_samples=2):
    """定义读取并展示样本的函数

    Args:
        file_path (str): 读取的文件路径
        num_samples (int, optional): 展示的样本数量. Defaults to 2.
    """
    print(f"Reading from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                # 将每行的JSON字符串解析为Python对象
                # NOTE line.strip() 是字符串对象的一个方法，它的主要作用是移除字符串首尾的空白字符（包括空格、制表符 \t、换行符 \n 等）
                # NOTE line.strip() 确保 line 是一个纯粹的JSON字符串，没有多余的空白字符干扰 json.loads() 方法对JSON字符串的解析。要是字符串里存在多余的空白，可能会引发解析错误
                # print("line type: ", type(line))   # str
                """
                   line type:  <class 'str'>
                   json.loads(line.strip()) type:  <class 'dict'>
                """
                data = json.loads(line.strip())
                # 使用json.dumps函数将Python对象转换为JSON字符串，并设置indent=4以美化输出，ensure_ascii=False以支持非ASCII字符
                print(f"Sample {i+1}: {json.dumps(data, indent=4, ensure_ascii=False)}")
                # json dump type:  <class 'str'>
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n ============================= \n")

# 读取多个JSONL（JSON Lines）文件，并展示每个文件中的前两个样本数据
for path in file_paths:
    read_and_display_samples(path)
```

    Reading from: ..\..\LLM_By_Hand\dataset\tokenizer_train.jsonl
    Sample 1: {
        "text": "好的。现在请你将这个文本中的所有的逗号都替换成空格。 好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为：\"这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。\"。处理结果如何？"
    }
    Sample 2: {
        "text": "帮我回答一道历史题目。清朝时期的八旗共有多少旗人？ 清朝时期八旗旗人总数约为200万人左右，其中正黄旗、正蓝旗、正白旗、正红旗的人数较多，其他旗的人数较少。"
    }
    
     ============================= 
    
    Reading from: ..\..\LLM_By_Hand\dataset\pretrain_hq.jsonl
    Sample 1: {
        "text": "<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s> <s>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。</s> <s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s> <s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。</s>"
    }
    Sample 2: {
        "text": "<s>根据输入的内容，编写一个类别标签。\n这是一篇介绍如何阅读心电图的文章类别标签: 医学/心电图阅读指南</s> <s>帮我搜索一下最近的天气情况。当然，我可以帮您搜索最新的天气情况。请问您需要查询哪个城市的天气情况呢？</s> <s>帮我讲一个令人开心的笑话。好的，我帮您讲一个关于细菌的笑话。为什么细菌不会上网？因为连接总是断开了！</s> <s>现在给我生成一首关于大海的五言诗。碧波万顷月满天，海天相接处天地间。波涛滚滚江山美，海鸟翱翔日月闲。</s> <s>谢谢你，这篇文章很有用。不客气，我很高兴能够为您提供帮助。如果您还有其他问题或需求，随时可以对我说。</s> <s>你好，我想下载一个视频编辑软件，你有什么推荐吗？您好！当然，有很多选择。您想要免费软件还是愿意付费？</s> <s>为什么我的程序不输出正确结果？可能是代码逻辑有误，或者输入数据有误，需要仔细调试代码逻辑和输入数据。</s> <s>谢谢你的回答。现在我想知道这场比赛的具体时间和地点。这场比赛的时间是北京时间10月4日，地点是上海。</s>"
    }
    
     ============================= 
    
    Reading from: ..\..\LLM_By_Hand\dataset\sft_mini_512.jsonl
    Sample 1: {
        "conversations": [
            {
                "role": "user",
                "content": "请告诉我在中国古代的“四大发明”是什么？"
            },
            {
                "role": "assistant",
                "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n\n1. **造纸术**：据史书记载，东汉时期的蔡伦改进了造纸工艺，使得纸张的生产更加便捷、成本更低，质量也更加优良。这一发明极大地促进了文化的传播和保存。\n\n2. **印刷术**：中国古代的印刷术最早可以追溯到唐代的雕版印刷，到了宋代发展出了活字印刷技术。印刷术的发明极大地促进了知识的传播，降低了书籍的成本，对教育和文化的发展起到了重要的推动作用。\n\n3. **火药**：火药最初是在唐代被发现的，最初可能用于医疗或炼金术。到了宋代，人们开始将火药用于军事目的，发明了各种火器。火药的发明改变了战争的面貌，对世界军事技术的发展产生了重大影响。\n\n4. **指南针**：指南针最初在中国被用于风水测量，后来逐渐发展成为航海导航的重要工具。这一发明极大地促进了海上航行技术的发展，对于新航路的开辟和世界地理大发现起到了关键作用。\n\n这四项发明不仅在中国历史上占有重要地位，而且对全世界的科技进步和文明发展都产生了深远的影响。"
            }
        ]
    }
    Sample 2: {
        "conversations": [
            {
                "role": "user",
                "content": "请用一段话描述阿里巴巴集团的企业文化。"
            },
            {
                "role": "assistant",
                "content": "阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。公司倡导开放、透明、分享、责任的团队合作精神，鼓励员工创新、追求卓越，同时注重员工的个人成长和幸福感。阿里巴巴的企业文化还体现在其独特的“六脉神剑”价值观体系中，包括客户第一、拥抱变化、团队合作、诚信、激情、专业等六个方面，这些价值观不仅指导着公司的日常运营，也深深影响着每一位阿里人的行为准则。"
            }
        ]
    }    
    

<font color ="red">**在不同的训练流程中、我们需要不同类型的数据——**</font>

**1. 微调数据集：问答对**
- **如何实现预训练**：
  - **任务类型**：监督微调（SFT, Supervised Fine-Tuning）。
  - **训练方法**：基于人类标注的问答对，通过优化模型在给定输入（问题）下生成期望输出（回答）的能力。
    - 输入：问题文本（或带上下文的对话历史）。
    - 输出：目标回答文本。
  - **损失函数**：通常使用**交叉熵损失（Cross-Entropy Loss）**，评估模型生成回答与参考答案的相似程度。交叉熵损失会在每个时间步计算预测的分布与目标token的真实分布之间的差异，对于一段话、交叉熵损失通过将每个token的损失相加、得到整段话的损失。

**2. 预训练数据集：短文本**
- **如何实现预训练**：
  - **任务类型**：无监督学习（例如自回归语言建模或掩码语言建模）。
    - **自回归建模**（如 GPT）：模型按顺序预测每个词的下一个词。
    - **掩码语言建模**（如 BERT）：模型根据上下文预测被掩盖的词。
  - **训练方法**：利用短文本数据片段，构建语言建模任务。
    - 自回归任务：输入前半部分文本，预测后续部分。此时使用的是**自回归损失CLM**。
    - 掩码任务：随机掩盖部分单词，让模型填空。此时使用的是**掩码损失MLM**。
    - 多模态任务：通常对图文进行匹配。此时使用的是**对比损失**。

**3. 分词器训练数据集：超短文本**
- **如何实现预训练**：
  - **任务类型**：词汇构建与分词规则学习。
  - **训练方法**：
    1. 对超短文本进行统计分析，确定出现频率最高的字符或词。
    2. 使用分词算法（如 BPE 或 SentencePiece）将常见的字符组合压缩为词表项。
    3. 生成分词规则和词表，供后续模型训练使用。
    4. 由于分词器的训练本质上是一个基于规则或统计算法的过程，因此不是神经网络的优化过程，因此分词器的训练过程**并不需要损失函数**。

#### 1.2.3 Jsonl 数据格式说明

JSONL（JSON Lines）：JSONL（JSON Lines）是一种特别适合处理大规模数据的格式，尤其在机器学习和大数据领域得到了广泛应用，它是一种逐行存储 JSON 对象的文件格式，每行是一个独立的 JSON 对象，行与行之间并没有特定的结构。每行的 JSON 对象独立存在，不属于同一个数组或对象。例如：


```python
{"name": "Alice", "age": 25}
{"name": "Bob", "age": 30}
```




    {'name': 'Bob', 'age': 30}



这种数据类型展示的信息其实就是——

|name|age|
|:-:|:-:|
|Alice|25|
|Bob|30|

但由于JSONL天生的字典格式、它展示表单信息的效率远远高于Dataframe这些结构，因此许多大型数据都呈现jsonl格式，你可能看到超大数据集是这样的结构 ↓

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/13.png)

但是，JSONL有它的劣势，其中最核心的一条就是文件体积稍大、它还不是最有效存储超大数据的格式——

| 格式          | 优势                                                                 | 劣势                                                          |
|---------------|----------------------------------------------------------------------|---------------------------------------------------------------|
| **JSONL**     | 支持流式处理、高容错性、易扩展、通用性强                              | 文件体积稍大（每条记录有元数据开销）。                         |
| **CSV**       | 紧凑、高效，占用空间小                                               | 不适合嵌套或复杂结构数据，不支持多种数据类型。                 |
| **Parquet**   | 高压缩比、支持列式存储、查询速度快                                    | 不易阅读，不适合直接调试或小规模任务。                         |
| **Protobuf**  | 高效、压缩率高、适合大规模数据传输                                   | 二进制文件不可读，需要额外的工具和定义文件解析。               |
| **二进制文件** | 存储高效，占用空间小，适合高密度数据（如矩阵或图像）                  | 不可读、不易调试，对文件结构高度依赖。      

针对这样的数据集，我们可以完成一个极为特殊的预处理手段，那就是将JSONL格式转化为二进制Bin文件，不过这个操作需要在我们完成Tokenzier训练后才可以实现。

### 1.3 MateConv Mini 的 tokenizer 训练

为了加快训练流程, 可以选择使用云服务器进行模型训练。

- step 0. 服务器配置

```bash
#激活已建好的虚拟环境
conda activate MateConv

#进入MateConv文件夹
cd ~/autodl-tmp/MateConv/

#在MateConv文件夹中建立dataset目录
mkdir dataset

#进入dataset目录
cd ./dataset
```

接着，确保数据文件`tokenizer_train.jsonl`已经上传到dataset文件夹内。

上传完毕后即可使用！整个分词器的训练流程（下面的训练代码）也已打包好一个单独的ipy文件（Step1: tokenizer_training.ipynb）。这段代码同样上传至线上的`~/autodl-tmp/MateConv/`文件夹、然后在这个文件夹下启动jupyter，即可运行这段代码。

```bash

cd ~/autodl-tmp/MateConv/

jupyter lab --allow-root
```

启动jupyter之后，需要使用autodl的ssh隧道工具进行代理、才可以访问线上jupyter。然后就可以在线上运行这段文件、来实现Tokenizer的训练了。

- Step 1. 导入必要的库


```python
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os
```


```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

    Using GPU: NVIDIA GeForce RTX 3050 Laptop GPU
    

- Step 2. 读取 tokenizer_train.jsonl 文件


```python
def read_texts_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

# 测试读取数据，可以自定义目录
# data_path = '/root/autodl-tmp/MateConv/dataset/tokenizer_train.jsonl'
data_path = r"..\..\LLM_By_Hand\dataset\tokenizer_train.jsonl"


texts = read_texts_from_jsonl(data_path)
# 统计文本长度
lengths = [len(text) for text in texts]
print('文本长度：', lengths[:5])    # 打印前5条文本的长度
print('文本数量统计：', len(lengths))    # 统计文本数量


texts = read_texts_from_jsonl(data_path)
# 打印前几行文本
for i, text in enumerate(texts):
    if i < 3:
        print(text)
    else:
        break
```

    文本长度： [103, 78, 353, 417, 715]
    文本数量统计： 600000
    好的。现在请你将这个文本中的所有的逗号都替换成空格。 好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为："这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。"。处理结果如何？
    帮我回答一道历史题目。清朝时期的八旗共有多少旗人？ 清朝时期八旗旗人总数约为200万人左右，其中正黄旗、正蓝旗、正白旗、正红旗的人数较多，其他旗的人数较少。
    嗯，谢谢你介绍的做法很详细，但我不喜欢吃鸡蛋，有没有其他菜做法能介绍一下？ 当然，你可以试试酸辣土豆丝这道菜。
    材料：
    土豆2个、红椒1个、青椒1个、大葱1根、醋、生抽、盐、鸡精、料酒
    做法：
    1.土豆去皮，切成丝；红椒和青椒切成细丝；大葱切段备用。
    2.热锅凉油，油热后放入土豆丝，煸炒至变软。
    3.倒入红椒、青椒和大葱段，继续煸炒至熟。
    4.加入适量的盐、鸡精、料酒和生抽，翻炒均匀。
    5.最后，加入适量的醋，翻炒均匀即可。
    小贴士：
    1. 土豆切丝时，可以放入淡盐水中泡一下，这样可以去除多余的淀粉。
    2. 煮土豆丝时，不要煮得太久，以免烂糊。
    3. 加入醋的时候，根据自己的口味多少来进行调节，一般来说，盐与醋的比例为1:1。
    4. 如果喜欢辣味可以加入一些干辣椒丝。
    希望你会喜欢这道酸辣土豆丝！
    

- Step 3. 初始化分词器

首先，通过 `models.BPE()` 创建了一个基于 Byte-Pair Encoding (BPE) 模型的分词器。BPE 是一种常用于文本分词的子词分解算法，特别在自然语言处理任务中被广泛使用，如机器翻译和语言模型训练。BPE 的主要思想是通过将频繁出现的字符或字符对合并成一个新的子词单元，逐步构建一个子词级别的词汇表，从而处理词汇表稀疏性和未登录词问题。


```python
# 初始化tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 定义特殊token
special_tokens = ["<unk>", "<s>", "</s>"]

# 设置训练器并添加特殊token
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,  # 确保这三个token被包含
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

print("分词器初始化成功，准备训练。")
```

    分词器初始化成功，准备训练。
    

- Step 4. 训练分词器


```python
# 读取文本数据
texts = read_texts_from_jsonl(data_path)

# 训练tokenizer
tokenizer.train_from_iterator(texts, trainer=trainer)

print("分词器训练完成！")
```

- Step 5. 保存分词器

在训练完毕之后，还需要设置解码器 (`tokenizer.decoder = decoders.ByteLevel()`) ，这是为了在生成文本时正确地将分词器产生的 token 序列还原回原始文本。

同时，在保存tokenizer之前，你需要建立用于存放模型的目录 ↓

```bash
#激活已建好的虚拟环境
conda activate MateConv

#进入MateConv文件夹
cd ~/autodl-tmp/MateConv/

#在MateConv文件夹中建立dataset目录
mkdir -p ~/autodl-tmp/MateConv/model/mateconv_tokenizer

#进入dataset目录
cd ./model/mateconv_tokenizer
```


```python
# 设置解码器
tokenizer.decoder = decoders.ByteLevel()

# 保存tokenizer
tokenizer_dir = "/root/autodl-tmp/MateConv/model/mateconv_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save("/root/autodl-tmp/MateConv/model/mateconv_tokenizer")

# 手动创建配置文件
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": True,
    "added_tokens_decoder": {
        "0": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "1": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "2": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            }
    },
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "legacy": True,
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": None,
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<unk>",
    "use_default_system_prompt": False,
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
}

# 保存配置文件
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)

print("Tokenizer 保存成功！")
```

- Step 6. 评估分词器


```python
from transformers import AutoTokenizer

# 加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/mateconv_tokenizer")

# 测试一段对话
messages = [
    {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
    {"role": "user", "content": '是椭圆形的'},
    {"role": "assistant", "content": '456'},
    {"role": "user", "content": '456'},
    {"role": "assistant", "content": '789'}
]

# 使用模板进行文本处理
new_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(new_prompt)
```
