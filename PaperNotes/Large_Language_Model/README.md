# 论文解读 & 模型算法

> 在这个模块中，主要存放相关Paper（Transformer/Bert/GPT架构系列/多模态）的论文解读。



| Notes                                                        | Tag      | Describe                                                     | File                                                         | 
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| [Transformer 论文精读](./Transformer%20论文精读.md)          | NLP      | Attention Is All You Need<br />NeurIPS 2017<br /><br />从零开始复现 Transformer（PyTorch），具体路径如下：<br />1. 缩放点积注意力->单头->掩码->自注意力->交叉注意力->多头->对齐论文<br/>2. 位置前馈网络（Position-wise Feed-Forward Networks）<br/>3. 残差连接（Residual Connection）和层归一化（Layer Normalization, LayerNorm），对应于 Add & Norm<br/>4. 输入嵌入（Embeddings）<br/>5. Softmax<br/>6. 位置编码（Positional Encoding）<br/>7. 编码器输入处理和解码器输入处理<br/>8. 掩码实现（填充掩码和未来掩码）<br/>9. 编码器层（Encoder Layer）和解码器层（Decoder Layer）<br/>10. 编码器（Encoder）和解码器（Decoder）<br/>11. 完整模型（Transformer）<br />将介绍模型架构中的所有组件，并解答可能的困惑（访问[速览疑问](./Transformer%20论文精读.md#速览疑问)进行快速跳转） | [Code](./Demos/动手实现%20Transformer.ipynb)                 |
| [BERT 论文精读](./BERT%20论文精读.md)                        | NLP      | Pre-training of Deep Bidirectional Transformers for Language Understanding<br />NAACL 2019<br /><br />基于 Transformer 架构｜Encoder-Only<br />文章概览：<br />1. 预训练任务 MLM 和 NSP<br />2. BERT 模型的输入和输出，以及一些与 Transformer 不同的地方<br />3. 以 $\text{BERT}_\text{BASE}$ 为例，计算模型的总参数量<br /> | [BERT 论文解读](./BERT%20论文精读.md) |
| [GPT 论文精读](./GPT%20论文精读.md)                          | NLP      | GPT 数字系列论文：<br />- [GPT-1](./GPT%20论文精读.md#gpt-1)<br />- [GPT-2](./GPT%20论文精读.md#gpt-2)<br />- [GPT-3](./GPT%20论文精读.md#gpt-3)<br />- [GPT-4](./GPT%20论文精读.md#gpt-4) | [GPT 论文精读](./GPT%20论文精读.md) |