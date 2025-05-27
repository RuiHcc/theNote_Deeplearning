## 1 background
[Lenet](https://so.csdn.net/so/search?q=Lenet&spm=1001.2101.3001.7020) 是一系列网络的合称，包括 Lenet1 - Lenet5，由 Yann LeCun 等人 在1990年《Handwritten Digit Recognition with a Back-Propagation Network》中提出，是卷积神经网络的开山之作。

>用途：接收**灰度**图像，实现手写数字识别

## 2 structure
![[Pasted image 20250327113924.png]]
Lenet是一个 7 层的神经网络，包含 3 个卷积层，2 个池化层，1 个全连接层，1个输出层。其中所有卷积层的卷积核都为 5x5，步长=1，池化方法都为平均池化，激活函数为 Sigmoid（目前使用的Lenet已改为ReLu）

下面把整个网络划分成三部分来进行解析：
	net，定义LeNet网络模型
	train，加载数据集并训练，计算loss和accuracy，保存训练好的网络参数
	test，用自己的数据集进行分类测试

## 3 code
### 2.1 net.py：构建一个网络必要的两部分：初始化和前向传播
```
class MyLeNet5(nn.Module):  
    # 初始化  
    def __init__(self):  
        super(MyLeNet5, self).__init__()  
  
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,  
                            padding=2, stride=1)  
        self.Sigmoid = nn.Sigmoid()  
  
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  
  
        self.flatten = nn.Flatten()  
        self.f6 = nn.Linear(120, 84)  
        self.output = nn.Linear(84, 10)  
  
    def forward(self, x):  
        x = self.Sigmoid(self.c1(x))  
        x = self.s2(x)  
        x = self.Sigmoid(self.c3(x))  
        x = self.s4(x)  
        x = self.c5(x)  
        x = self.flatten(x)  
        x = self.f6(x)  
        x = self.output(x)  
        return x
```

首先来讲讲继承 nn.Module 我的模型获得了什么？
注：torch要求必须重写foward函数，即foward不能通过继承获得
主要的：
	1、parameters()
		作用：返回模型所有可训练参数（nn.Parameter 对象）的生成器。
		用途：优化器（如 optim.SGD）通过此方法获取需要更新的参数。
	2、to(device)
		作用：将模型的所有参数和缓冲区移动到指定设备（如 CPU 或 GPU）
	3、state_dict()
		用途：保存和加载模型

下面讲一下每一层输出：
![[Pasted image 20250327121704.png || 600]]
input：输入层，**传统上，不将输入层视为网络层次结构之一。** 实际的数据训练格式为：`torch.Size([16, 1, 28, 28])` 批量大小16，输入通道1, 28×28

c1：卷积层，由于有6个输出通道，所以有6个对应卷积核
	先把输入图像填充为了32×32；对输入图像进行第一次卷积运算（使用 6 个大小为 5×5 的卷积核），得到6个C1特征图（6个大小为28×28的 feature maps, 32-5+1=28）；
	可训练参数：（5×5+1) ×6

激活函数不改变输出形状

s2：池化层
	2×2pooling，stride为2
	输出：(28-2+2)/2= 14×14

c3：输出：使用16个大小为 5×5 的卷积核，得到16个c3特征图10×10

s4：输出:5×5

c5：输出1×1,120个输出通道

f6全连接层：输入：c5 120维向量，展开

### 2.2 train.py
```
# 数据转化为tensor格式  
data_transforms = transforms.Compose([  
    transforms.ToTensor()  
])  
  
# 加载训练的数据集  
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transforms, download=True)  
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,  
                                               batch_size= 16, shuffle=True)  
  
# 加载测试数据集  
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transforms, download=True)  
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,  
                                               batch_size= 16, shuffle=True)  
```
`transforms.Compose`：把多种变换组合起来，虽然这里只有转换为`tensor`格式一种

`train_dataset`：加载训练数据集，参数说明：
	`root='./data'`：数据集存储在./data目录下。
	`train=True`：加载训练集（共60,000张图像）。**train=False** 保证了两个数据集的来源是不一样的
	`transform=data_transforms`：应用之前定义的预处理流程（转换为Tensor）。
	`download=True`：如果本地不存在数据，自动从网络下载。

`train_dataloader`：创建训练数据加载器
作用：将数据集转换为可迭代的DataLoader，支持按批次加载数据，适用于模型训练
也就是说**空有一堆数据其实是不能够直接用于训练的**，我们要将其处理成适合进行训练的格式
	`batch_size=16`：每个批次加载16张图像。
	`shuffle=True`：每个训练周期（epoch）开始时打乱数据顺序，防止模型学习到数据的顺序性偏差。

```
# 损失函数  
loss_fn = nn.CrossEntropyLoss()  
  
# 优化器  
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)  
  
# 学习率每隔10轮变为原来的0.1  
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

`nn.CrossEntropyLoss()`：交叉损失熵函数
	**交叉熵定义**：$H(P,Q)=-\sum Plog(Q)$ ，P表示真实概率分布；Q表示模型预测概率分布
	**计算流程**：PyTorch的 nn.CrossEntropyLoss() 实际上做了以下两步：1、Softmax归一化：将模型的原始输出（logits）转换为概率分布。2、计算交叉熵：根据真实标签索引计算损失。
	1. 模型输出（Logits）
	假设模型对一个样本的输出（未归一化的logits）为：
	$$logits=[z1,z2,...,zk]$$
	2. Softmax计算概率
	通过Softmax函数将logits转换为概率：
		$$y_{hat}=e^{zi} / \sum e^{z_j}$$
	3. 计算交叉熵
	对于真实标签 y，损失值为:
	$$Loss = -log(y_{hat})$$
	具体到LeNet，net(x)输出是一个1×10的tensor，对应10个类别
		一个样本的输出logits可能是：
		$$tensor([2.1, -0.5, 1.3, 3.2, 0.7, -1.2, 0.4, 2.9, -0.1, 1.5])$$
		步骤1：首先对其进行softmax归一化：
$$prob = [0.047, 0.004, 0.030, 0.465, 0.013, 0.002, 0.009, 0.325, 0.005, 0.033]$$
		步骤2：提取真实类别对应的概率
		假设真实标签为类别3（索引为3，对应第4个位置，因为PyTorch从0开始计数），则：$prob_{true}=prob[3]=0.465$
		步骤3：计算负对数损失$loss=-log(prob_{true}=-log(0.465)=0.766$
	而实际训练是以**批量**计算：假设一个训练批次（batch）中有2个样本：
		模型输出（logits）
		$$
		\begin{matrix}
		logits = torch.tensor([\\
		[2.1, -0.5, 1.3, 3.2, 0.7, -1.2, 0.4, 2.9, -0.1, 1.5]样本1
		\\
		[1.2, 0.8, -0.3, 2.5, -1.0, 0.5, 2.0, 3.1, 1.5, 0.2]
		])样本2
		\end{matrix}
		$$
		真实标签为：
		$y_{true} = torch.tensor([3, 7])   样本1的真实类别是3，样本2是7$
		1. 对每个样本应用Softmax归一化:
			样本1的概率分布：$[0.047, 0.004, 0.030, 0.465, 0.013, 0.002, 0.009, 0.325, 0.005, 0.033]$
			样本2的概率分布：$[0.032, 0.023, 0.006, 0.158, 0.002, 0.008, 0.070, 0.582, 0.103, 0.016]$
		2. 提取真实类别对应的概率:
			$样本1：prob_{true1} = 0.465$
			$样本2：prob_{true2} = 0.582$
		3. 计算每个样本的损失：
			$样本1：loss_1 = -log(0.465) ≈ 0.766$
			$样本2：loss_2 = -log(0.582) ≈ 0.541$
		4. 计算批次平均损失：
			$batch_{loss} = (0.766 + 0.541)/2 = 0.653$

`torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)`：SGD随机梯度下降
原理：$\theta_{t+1}=\theta_{t}-\eta*\nabla L(\theta_t​)$
	$\theta_t$：第t步的参数值，$\eta$：学习率，$\nabla L()$：损失函数对参数的梯度
	`momentum`：表示动量，可以显著加速模型训练并减少震荡
计算步骤：
	步骤1：初始化动量缓冲区
		在第一次参数更新前，为每个参数初始化动量变量 v（初始值为0）。
	步骤2：前向传播
		输入数据通过模型（如LeNet）计算预测值，得到损失值（如交叉熵损失）。
	步骤3：反向传播计算梯度
		调用 loss.backward()，计算损失对参数的梯度$\nabla L()$
	步骤4：更新动量
	步骤5：参数更新$\theta_{t+1}$
	步骤6：清空梯度
		调用 optimizer.zero_grad()，清空梯度缓冲区，避免梯度累积
		
`lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)`：学习率调度器
	`optimizer`：要调整学习率的优化器
	每隔`step_size`个epoch将学习率按固定比例缩小(每次调整时，学习率乘以 `gamma:0.1`)，

```
# 训练函数  
def train(dataloader, model, loss_fn, optimizer):  
    loss, current, n = 0.0, 0.0, 0  
    for batch, (X,y) in enumerate(dataloader):  
        # 前向传播  
        X, y = X.to(device), y.to(device)  
        output = model(X)  
        cur_loss = loss_fn(output, y)  
        _, pred = torch.max(output, axis=1)  
  
        cur_acc = torch.sum(y == pred)/output.shape[0]  
  
        optimizer.zero_grad()  
        cur_loss.backward()  
        optimizer.step()  
  
        loss += cur_loss.item()  
        current += cur_acc.item()  
        n += 1  
    print("training loss" + str(loss/n))   # 平均损失
    print("training accuracy" + str(current/n)) # 平均预测率
```
`loss, current, n`：累计批次损失，累计n次正确预测率，已处理批次数量

逐批次加载数据，每个批次包含 batch_size 16个样本
	`X`：输入图像张量，形状为 (16, 1, 28, 28)（MNIST图像）。
	`y`：标签张量，形状为 (16)。

`X.to(device), y.to(device)`：数据移动到设备（GPU）

`output`：每个样本对应10个类别的原始logits（未归一化的预测值）

`cur_loss = loss_fn(output, y)`：计算当前批次的损失（交叉熵损失）

`_, pred = torch.max(output, axis=1)`：沿维度1（类别维度）取最大值，并得到预测类别索引
	输出：
	`_`：最大值（不需要使用）。
	`pred`：形状为 (16,) 的预测类别索引，例如 `tensor([3, 7, 2, ..., 9])`。

`cur_acc = torch.sum(y == pred) / output.shape[0]`：y和pred每个元素做`==`验证，并求和统计准确率

优化器更新：
    `optimizer.zero_grad()`  :清空优化器中所有参数的梯度，避免梯度累积。
    `cur_loss.backward() `  :计算损失对模型参数的梯度，反向传播至每一层
    通过链式法则计算梯度/梯度存储在参数的 .grad 属性中（如 model.conv1.weight.grad
    `optimizer.step()`  :根据优化器（如SGD）的规则更新模型参数

`.item()`：作用是取出单元素张量的元素值并返回该值



```
def val(dataloader, model, loss_fn):  
    model.eval()  
    loss, current, n = 0.0, 0.0, 0  
    # 测试时模型不参与更新  
    with torch.no_grad():  
        for batch, (X, y) in enumerate(dataloader):  
            # 钱箱传播  
            X, y = X.to(device), y.to(device)  
            output = model(X)  
            cur_loss = loss_fn(output, y)  
            _, pred = torch.max(output, axis=1)  
  
            cur_acc = torch.sum(y == pred) / output.shape[0]  
  
            loss += cur_loss.item()  
            current += cur_acc.item()  
            n += 1  
        print("val loss" + str(loss/n))  
        print("val accuracy" + str(current/n))  
  
        return current/n
```
与train函数类似，不展开

```
for i in range(epoch):  
    print(f'epoch {i+1}\n---------------')  
    train(train_dataloader, model, loss_fn, optimizer)  
    # val(test_dataloader, model, loss_fn)  
    a = val(test_dataloader, model, loss_fn)  
    #保存最好的模型参数(准确率)  
    if a > min_acc:  
        folder = 'saved_models'  
        if not os.path.exists(folder):  
            os.makedirs(folder)  
        min_acc = a  
        print('save best model')  
        torch.save(model.state_dict(), 'saved_models/best_model2.0.pth')  
print("Done!")
```
`a`：表示测试集准确率

创建模型保存目录: 这里默认是在脚本的运行目录下创建folder
`folder = 'saved_models' ` 
`if not os.path.exists(folder): ` 
    `os.makedirs(folder)`
    
`torch.save(model.state_dict(), 'saved_models/best_model2.0.pth')`:
	`model.state_dict()` 保存的是模型参数，后面的是保存路径及名称


### 2.3 test.py
```
model.load_state_dict(torch.load("C:/Users/59564/PyCharmMiscProject/LeNet-5/saved_models/best_model2.0.pth"))  
  
# 获取结果  
classes = [  
    "0","1","2","3","4","5","6","7","8","9"]  
  
# tensor转化为图片 方便可视化  
show = ToPILImage()  
  
# 进入验证  
for i in range(20):  
    x, y = test_dataset[i][0], test_dataset[i][1]  
    show(x).show()  
  
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)  
    with torch.no_grad():  
        pred = model(x)  
  
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]  
  
        print(f'predicted: "{predicted}", actual: "{actual}"')
```
前面类似的，不做展开！！

`model.load_state_dict(torch.load("C:/xx")) `：加载训练参数，值得注意的是这里用的是反斜杠 `/`

`classes`: 定义类别标签，将MNIST数据集的标签索引（0-9）映射为字符串类别名称，比如模型预测结果为3，那么对应类别就是3

`show = ToPILImage()`：将张量（Tensor）转换为PIL图像对象，用于显示图像
	吐槽：这里目前感觉很鸡肋，并不能显示完整的测试20张图片

`x = Variable()`：
	`torch.unsqueeze(x, dim=0)`：在维度0（批次维度）上插入一个大小为1的新维度，将形状从 (1,28,28) 变为 (1,1,28,28)（即 batch_size=1）
	`Variable ` 是过去用于封装张量并启用自动梯度计算的类

`predicted, actual = classes[torch.argmax(pred[0])], classes[y]`：
	`torch.argmax(pred[0])`：找到 logits（原始输出，概率） 中最大值的索引
	值得注意的是`y`和`pred`其实都是类别的索引，`class`可以立即为一种映射