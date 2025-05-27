关于Python语法部分不过多赘述，可见：[RuiHcc/theNote_Python: 基础部分至此结束 后续随缘更新...](https://github.com/RuiHcc/theNote_Python)
这里补充一下numpy这个库

### 前言
本部分内容是对李沐深度学习课程的补充，算是对深度学习基础内容的复习~

# Numpy
NumPy 的数组类  (numpy.array)中提供了很多便捷的方法,用于处理数组和矩阵的计算。
作为外部库当然需要导入`import numpy as np`

**生成 NumPy 数组：** `np.array()` 接收 Python  列表作为参数,生成 NumPy 数组`(numpy.ndarray)`
```
x = np.array([1.0, 2.0, 3.0])
print(x)
>>> [ 1., 2., 3.]
type(x)
>>><class 'numpy.ndarray'>
```

**NumPy 的算术运算：** NumPy 数组不仅可以进行 element-wise `对应元素的`运算,也可以和单一的数值(标量)组合起来进行运算。
值得注意的是后者的这种功能通过**广播机制**来实现，这点很重要！
```
x = np.array([1.0, 2.0, 3.0]) 
y = np.array([2.0, 4.0, 6.0]) 
x + y # 对应元素的加法 
>>> array([ 3., 6., 9.]) 
x - y 
>>> array([ -1., -2., -3.]) 
x * y # element-wise product 
>>> array([ 2., 8., 18.])
x / y 
>>> array([ 0.5, 0.5, 0.5])

x = np.array([1.0, 2.0, 3.0]) 
>>> x / 2.0  
>>> array([ 0.5, 1. , 1.5])
```
当然，numpy也可以生成N维数组
```
A = np.array([[1, 2], [3, 4]])
print(A)
>>> [[1 2] [3 4]]
print(A.shape)
>>> (2, 2)
print(A.dtype)
>>> dtype('int64')
```
**注：** 
- 数学上将一维数组称为向量,  将二维数组称为矩阵。另外,可以将一般化之后的向量或矩阵等统称为张量(tensor)。本书基本上将二维数组称为“矩阵”,将三维数组及三维以上的数组称为“张量”或“多维数组”。
- 后面torch的基本处理数据也为`tensor`

**广播机制** ：NumPy 中,形状不同的数组之间也可以进行运算。
之前的例子中,在  2×2 的矩阵 A 和标量 10 之间进行了乘法运算。这个过程中，标量 10 被扩展成了 2 × 2 的形状,然后再与矩阵 A 进行乘法运算。这个巧妙的操作称为广播(broadcast)。
![[Pasted image 20250527132715.png | 650]]
再举个例子：B被扩展成了`np.array([10, 20], [10, 20]) `
```
A = np.array([[1, 2], [3, 4]]) 
B = np.array([10, 20]) 
print(A * B) 
array([[ 10, 40], [ 30, 80]])
```

# Matplotlib
**基本语法**：
```
import numpy as np 
import matplotlib.pyplot as plt  
# 生成数据  
x = np.arange(0, 6, 0.1) # 以 0.1 为单位,生成 0 到 6 的数据 
y1 = np.sin(x)

# 绘制图形  
plt.plot(x, y1, label="sin") 
plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制 
plt.xlabel("x") # x 轴标签 
plt.ylabel("y") # y 轴标签 
plt.title('sin & cos') # 标题 
plt.legend() 
plt.show()
```