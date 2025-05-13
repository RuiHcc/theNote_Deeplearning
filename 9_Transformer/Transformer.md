前面通过与RNN，CNN对比，**自注意力**同时具有并行计算和最短的最大路径长度这两个优势。因此，使用自注意力来设计深度架构是很有吸引力的。

**Transformer模型**完全基于注意力机制，没有任何卷积层或循环神经网络层 ([Vaswani _et al._, 2017](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id174 "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008)."))。尽管Transformer最初是应用于在文本数据上的序列到序列学习，但现在已经推广到各种现代的深度学习中，例如语言、视觉、语音和强化学习领域。

### Transformer模型架构：
Transformer整体采用**编码器－解码器架构**：Transformer的编码器和解码器是基于自注意力的模块叠加而成的，源（输入）序列和目标（输出）序列的 _嵌入_（embedding）表示将加上 _位置编码_（positional encoding），再分别输入到编码器和解码器中。
![[transformer.svg|650]]
**Transformer的编码器**：由多个相同的层叠加而成的，每个层都有两个子层：
- 1、_多头自注意力_（multi-head self-attention）汇聚
- 2、_基于位置的前馈网络_ 
- 3、子层之间采用残差连接，紧接着应用层规范化
**Transformer解码器**：结构与编码器类似
- 1、多头自注意力层：_掩蔽_（masked）注意力保留了 _自回归_（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。
- 2、_编码器－解码器注意力_ 层：**查询来自前一个解码器层的输出，而键和值来自整个编码器的输出。**
- 3、_基于位置的前馈网络_
- 4、残差连接，紧接着应用层规范化

### 基于位置的前馈网络
对序列中的所有**位置表示**进行变换时使用的是同一个多层感知机（MLP）
	输入X（批量大小，时间步数或序列长度，隐单元数或特征维度）
	输出张量：（批量大小，时间步数，`ffn_num_outputs`）

### 残差连接和层规范化（add&norm）
 层规范化是基于**样本**维度进行规范化 与批量规范化区别

### 论文：
- 1、layernorm 和 batchnorm的区别，以及为什么要用layernorm
- 2、什么是自回归？（对解码器而言）
- 3、注意力机制为什么要除以$\sqrt{d_k}$
- 4、为什么采用多头注意力机制？
- 5、在transformer中是如何使用注意力机制的？
- 6、Transformer和RNN在处理序列数据的区别？