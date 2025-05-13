[【官方双语】GPT是什么？直观解释Transformer | 深度学习第5章_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV13z421U7cs/?spm_id_from=333.1387.collection.video_card.click&vd_source=e67c07b57bd6208ae6cf25baa99d3bcb)

GPT全称为Generative Pre-trained Transformer，这类大语言模型实际上是不断在对初始文本进行抽样和预测下一个词语的过程。
Transformer是一种特殊的神经网络模型

我们的目标是接受一段文本，属于一个预测的token
数据在Transformer中大致的流动：
	首先输入内容会被切分为许多小片段，称为`tokens`：对文本而言，`token`会是一个个单次，而图片，则会是若干小块的图像；每个`tokens`对应一个向量，用来表征这个词语的含义，那么相近含义的词语所对应的向量方向也应该会相似
	![[Pasted image 20250418200750.png | 400]]
	这些向量经过注意力机制处理，使得向量能够相互交流，并改变自身的值；比如检查上下文来找到model这个词的含义到底是模型还是模特
	之后这些向量会进入MLPs多层感知器/前馈层；此阶段向量不再相互交流，而是并行经历某种处理，类似于（通过向向量提出一些列问题，然后根据这些问题的答案来更新向量）；这两层的处理，本质都是大量的矩阵乘法
	然后不断重复...
	![[Pasted image 20250418201621.png | 500]]
	最后目标是，将整段文字的所有关键含义，融入到序列的最后一个向量，然后对该向量经过某种处理，得到下一个词所有可能token的概率分布，如此反复--> GPT3早期演示

GPT3 有1750亿个参数，上下文长度为2048个词
嵌入矩阵：
Attention:结合上下文将某一token移动到其对应的含义 向量上
解嵌入矩阵：
有温度的softmax：输入logits，输出其概率


### 注意力机制：
[【官方双语】直观解释注意力机制，Transformer的核心 | 【深度学习第6章】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TZ421j7Ke/?spm_id_from=333.1387.collection.video_card.click&vd_source=e67c07b57bd6208ae6cf25baa99d3bcb)

《Attention is all you need》
细化了token的含义，还允许模型互相传递这些嵌入向量所蕴含的信息

