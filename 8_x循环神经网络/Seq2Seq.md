Seq2Seq 从一个句子生成另一个句子  -> 机器翻译是其中一个应用
我们将使用两个循环神经网络的**编码器和解码器**， 并将其应用于 _序列到序列_（sequence to sequence，seq2seq）类的学习任务 ([Cho _et al._, 2014](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id24 "Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078."), [Sutskever _et al._, 2014](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id160 "Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems (pp. 3104–3112)."))。

遵循**编码器－解码器架构**的设计原则， 循环神经网络编码器使用长度可变的序列作为输入， 将其转换为固定形状的隐状态。输入序列的信息被 **_编码_** 到循环神经网络编码器的隐状态中。

解码器是基于输入序列的编码信息 和输出序列已经看见的或者生成的词元来预测下一个词元。下图演示了如何在机器翻译中使用两个循环神经网络进行序列到序列学习。
![[seq2seq.svg#pic_center| 650]]
`“<eos>”`表示序列结束词元。 一旦输出序列生成此词元，模型就会停止预测。
`“<bos>”`表示序列开始词元，它是解码器的输入序列的第一个词元。
**整体过程：** 在循环神经网络解码器的初始化时间步 -> `“<bos>”`  ->  使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态  ->  编码器最终的隐状态在每一个时间步都作为解码器的输入序列的一部分，允许标签成为解码器的观测/输入