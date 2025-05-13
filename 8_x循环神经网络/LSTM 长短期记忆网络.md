长期以来，隐变量模型存在着长期信息保存和短期输入缺失的问题。 解决这一问题的最早方法之一是长短期存储器（long short-term memory，LSTM）
它有许多与门控循环单元（ [9.1节](https://zh-v2.d2l.ai/chapter_recurrent-modern/gru.html#sec-gru)）一样的属性。 有趣的是，长短期记忆网络的设计比门控循环单元稍微复杂一些， 却比门控循环单元早诞生了近20年。

![[lstm-3.svg]]
**总结** ：
$$
\begin{matrix}
I_t = \sigma(X_tW_{xi} + H_{t-1}W_{hi} + b_i) \\ 
F_t = \sigma(X_tW_{xf} + H_{t-1}W_{hf} + b_f) \\ 
O_t = \sigma(X_tW_{xo} + H_{t-1}W_{ho} + b_o) \\ 
C_t' = tanh(X_tW_{xc} + H_{t-1}W_{hc} + b_c) \\ 
\\

C_t = F_t*C_{t-1}+I_t*C_t' \\ 
H_t = O_t*tanh(C_t)
\end{matrix}
$$
### 门控记忆元：
LSTM引入了 _记忆元_（memory cell），或简称为 _单元_（cell）。
有些文献认为记忆元$Ct$是隐状态的一种特殊类型， 它们与隐状态$Ht$具有相同的形状，其设计目的是用于记录附加的信息。

此外LSTM中还引入了：
- _输出门_（output gate）：$Ot$
- _输入门_（input gate）：$It$
- _遗忘门_（forget gate）：$Ft$ $$
\begin{matrix}
I_t = \sigma(X_tW_{xi} + H_{t-1}W_{hi} + b_i) \\ 
F_t = \sigma(X_tW_{xf} + H_{t-1}W_{hf} + b_f) \\ 
O_t = \sigma(X_tW_{xo} + H_{t-1}W_{ho} + b_o) \\ 
\end{matrix}
$$

### 候选记忆元
由于还没有指定各种门的操作，所以先介绍 _候选记忆元_（candidate memory cell） 𝐶𝑡'∈𝑅𝑛×ℎ。
$$C_t' = tanh(X_tW_{xc} + H_{t-1}W_{hc} + b_c) \\ $$

### 记忆元
输入门𝐼𝑡控制采用多少来自𝐶𝑡'的新数据， 而遗忘门𝐹𝑡控制保留多少过去的 记忆元𝐶𝑡−1∈𝑅𝑛×ℎ的内容。 使用按元素乘法。
举例来说，如果遗忘门始终为1且输入门始终为0， 则过去的记忆元𝐶𝑡−1 将随时间被保存并传递到当前时间步。
$$C_t = F_t*C_{t-1}+I_t*C_t' $$

### 隐状态
最后，我们需要定义如何计算隐状态 𝐻𝑡∈𝑅𝑛×ℎ， 这就是输出门发挥作用的地方。
$$H_t = O_t*tanh(C_t)$$
只要输出门接近1，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近0，我们只保留记忆元内的所有信息，而不需要更新隐状态。