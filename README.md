# LLM-Agent-HandsOn

**探索大语言模型 (LLM) 与智能体 (Agent) 的世界，以项目驱动学习，构建未来人工智能应用！**

本仓库旨在记录和分享我在 LLM 和 Agent 领域的学习历程，并通过实践项目深入理解相关技术。

**你将找到：**

* **学习笔记:**  详细的学习笔记，涵盖 LLM 和 Agent 的核心概念、算法原理、应用场景等。
* **模型复现:**  从零开始实现的 LLM 和 Agent 模型，以加深对技术的理解。
* **学习资源:**  精选的教程、论文、博客文章和代码库，涵盖 LLM 和 Agent 的基础知识、最新进展和应用案例。
* **项目实践:**  从零开始构建基于 LLM 和 Agent 的应用，例如：
  * **DS-Agent**: 基于大语言模型构建的智能问答系统。
  * **Bili-Agent**: 基于大语言模型构建的BiliBili智能搜索Agent。
  * **Arxiv-Agent**: 基于大语言模型构建的Arxiv论文智能搜索Agent。 
* **代码分享:**  项目相关的代码、脚本和工具，方便学习和复现。
* **心得体会:**  学习过程中遇到的问题、解决方案和经验总结。

## 文档目录

- **Projects**
  
  项目实践。
  
  - [**README.md**](./LLM_Project_OnHand/README.md)
    - 目录索引。
  - [**DS-Agent**](./LLM_Project_OnHand/DS-Agent/README.md)
    - 基于大语言模型构建的智能问答系统。
  - [**Bili-Agent**](./LLM_Project_OnHand/Bili-Agent/README.md)
    - 基于大语言模型构建的BiliBili智能搜索Agent。

- **PaperNotes(LLM)**
  
  论文解读 & 随笔。
  
  - [**README.md**](./PaperNotes/Large_Language_Model/README.md)
    - 目录索引。
  - [Transformer 论文精读](./PaperNotes/Large_Language_Model/Transformer%20论文精读.md)
    - 从零开始复现 Transformer（PyTorch），并对各组件进行解读。
    - [Code](./PaperNotes/Large_Language_Model/Demos/动手实现%20Transformer.ipynb) 
  - [BERT 论文精读](./PaperNotes/Large_Language_Model/BERT%20论文精读.md)
    - 预训练任务 MLM 和 NSP
    - BERT 模型的输入和输出，以及一些与 Transformer 不同的地方
    - 以 $\text{BERT}_\text{BASE}$ 为例，计算模型的总参数量
  - [GPT 论文精读](./PaperNotes/Large_Language_Model/GPT%20论文精读.md)
    - GPT 数字系列论文：[GPT-1](./PaperNotes/Large_Language_Model/GPT%20论文精读.md#gpt-1) / [GPT-2](./PaperNotes/Large_Language_Model/GPT%20论文精读.md#gpt-2) / [GPT-3](./PaperNotes/Large_Language_Model/GPT%20论文精读.md#gpt-3) / [GPT-4](./PaperNotes/Large_Language_Model/GPT%20论文精读.md#gpt-4)

- **PaperNotes(Action Recognition)**
  
  论文解读(PDF 格式)。
  
  - [**README.md**](./PaperNotes/Action_Recognition/README.md)
    - 目录索引。
  - [Knowledge-Assisted Representation Learning for Skeleton-Based Action Recognition](./PaperNotes/Action_Recognition/Language_Knowledge-Assisted_Representation.pdf)
    - 日期：2024.04.10 / 2024.04.20
    - 标题：基于骨架的动作识别中的语言知识辅助表示学习Language
    - 从零解读论文，并对部分代码原理进行介绍。
    - 论文解读: 
      - v1: [解读 v1](./PaperNotes/Action_Recognition/Language_Knowledge-Assisted_Representation.pdf)
      - v2: [解读 v2](./PaperNotes/Action_Recognition/Language_Knowledge-Assisted_Representation_v2.pdf)
    - [Paper Address](./PaperNotes/Action_Recognition/Paper/Language_Knowledge-Assisted_Representation.pdf) 

  - [Learning Video Representations from Large Language Models](./PaperNotes/Action_Recognition/Learning_Video_Representations_from_Large_Language_Models.pdf)
    - 日期：2024.04.30
    - 标题：从大型语言模型中学习视频表示
    - 从零解读论文，并对部分代码原理进行介绍。
    - [Paper Address](./PaperNotes/Action_Recognition/Paper/Language_Knowledge-Assisted_Representation.pdf)

  - [TIMEMIXER: DECOMPOSABLE MULTISCALE MIXING FOR TIME SERIES FORECASTING](./PaperNotes/Action_Recognition/TimeMixer.pdf)
    - 日期：2024.07.03
    - 标题：TimeMixer 可分解多尺度混合用于时间序列预测
    - 从零解读论文，并对部分代码原理进行介绍。
    - [Paper Address](./PaperNotes/Action_Recognition/Paper/TimeMixer.pdf)  

  - [KAN: Kolmogorov–Arnold_Networks](./PaperNotes/Action_Recognition/Kolmogorov–Arnold_Networks.pdf)
    - 日期：2024.07.16
    - 标题：KAN: Kolmogorov–Arnold_Networks
    - 从零解读论文，并对部分代码原理进行介绍。
    - [Paper Address](./PaperNotes/Action_Recognition/Paper/Kolmogorov–Arnold_Networks.pdf)  


> [!NOTE]
> 上述论文解读有关 Action Recognition Based Skeleton 任务，部分论文结合了LLM辅助建模。

- **LearningNotes**
  
  学习笔记 & 资料分享。
  
  - [**README.md**](./LearningNotes/README.md)
    - 目录索引。
  - [**Attention Mechanism**](./llm_learning/Attention_Mechanism/Attention_by_hand.md)
    - 大语言模型 (LLM) 中注意力机制的原理详解，公式推导和代码实现。(包含多种注意力机制的详细图解)
  - [**LLama Model Reproduction**](./llm_learning/Llama_Structural_Reproduction/model.md)
    - 对 LLaMa 模型结构进行复现。
  - [**PyCharm-Install**](./llm_learning/PyCharm-Install.ipynb)
    - PyCharm 安装教程，包括安装步骤、配置和常见问题解决。
  - [**SQLALchemy_Learning**](./llm_learning/SQLALchemy_Learning.ipynb)
    - SQLAlchemy 学习笔记。
  - [**Transformer_Learning**](./llm_learning/transformer_pipeline/README.md)
    - Transformer 等模型手绘结构图。
  - [**Ollama Reference**](./llm_learning/ollama_reference/README.md)
    - Ollama 的配置教程，包括`Windows` 和 `Linux` 系统两种版本。
  - [**Mamba Reference**](./llm_learning/Mamba/)
    - [Mamba Notes](./llm_learning/Mamba/mamba_Notes)
    - [Mamba Codes](./llm_learning/Mamba/mamba_Code)

## License

MIT 

<!-- > [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions. -->
<!-- CLIP，LLM Pretrain SFT，ViT， -->