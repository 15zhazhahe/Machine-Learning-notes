# 引言(Introduction)

## 机器学习是什么？

+ 1959年Arthur Samuel定义的机器学习为:**在没有明确设置的情况下，使计算机具有学习能力的研究领域.**("the field of study that gives computers the ability to learn without being explicitly programmed.")
+ 1998年Tom Mitchell给了一个更加正式的定义：**计算机程序从经验E中学习，解决某一问题E，进行某一性能度量P，通过P测定在T上的表现因经验E而提高.**(A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.)

## 监督学习

在监督需学习中，再给出的数据集中，我们已经知道了正确的输出，也就是输入与输出的关系是已知的。

监督学习问题一般被分类为回归和分类问题。在回归问题中，我们尝试用连续的函数来预测结果。在分类问题中，我们通常预测的是离散值。

+ 回归问题 - 基于给出的照片，预测照片中人的年龄
+ 分类问题 - 基于肿瘤的大小，预测肿瘤是良性还是恶性

## 无监督学习

在无监督学习中，就是给你一个数据集，你并不知道数据集里面有哪些类型，你需要自己从数据中发现其中的结构。而且监督学习是无法从预测的结果中获得任何准确性的反馈的。

+ 聚类问题
+ “鸡尾酒会”问题

# 单变量线性回归(Linear Regression with One Variable)

## 模型表示

+ $m$表示样本数量
+ $x^{(i)}$表示输入变量/特征
+ $y^{(i)}$表示输出变量/预测的变量
+ $(x,y)$表示训练样本
+ $(x^{(i)},y^{(i)})$表示第$i^{th}$个训练样本
+ $h$称为假设函数

### 监督学习思路

我们向算法提供训练集，学习算法的任务是输出一个函数，通常用$h$表示，假设函数的作用是根据输入变量来预测相应的输出$y$值

<div align=center>
![](./img/week-01.png)
</div>

当预测值是一个“连续”的输出时，我们称这个学习问题为回归问题，例如房价的预测。如果预测值为离散值时，这个学习问题就是一个分类问题，例如给你这个房子的面积，让你判断这个房子的户型

### 如何表示$h$

在这个单变量回归问题中，我们用$h_\theta(x) = \theta_0 + \theta_1 x$来表示，所以接下来的工作大致是，根据在给定的数据集$(x,y)$来确定$\theta_0和\theta_1$的值，在根据得到的$\theta_0和\theta_1$来预测新的$x$(即不在训练集当中的数据)

## 代价函数

代价函数(cost function)，用于表示$h_\theta$假设函数预测的准确性，在这里用对于输入$x$的预测结果与实际输出$y$之间的平均误差来表示代价函数。

$$J(\theta) = \frac{1}{2m}\sum_{1=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$

这里之所以有个$\frac{1}{2}$是为了便于后面的梯度计算。

### 代价函数的直观理解

由于代价函数表示的是预测输出与实际值的平均误差，那么在理想情况下，也就是预测的十分好的情况下，代价函数的值将等于$0$，所以为了让我们的假设函数$h_\theta$能更好的拟合训练集，我们找到一组$(\theta_0,\theta_1)$使得$J(\theta)$尽可能的小(即最小化代价函数)

## 梯度下降

梯度下降(Gradient Descent)，在这个视频中介绍的第一个算法，用于来寻找在代价函数中提到的$(\theta_0,\theta_1)$。

这个算法类似于下坡，即你要尽可能走到更低的地方，首先你被刷新在一个随机地点，然后你每继续走的一步，都会四处观望，走到一个比现在更低的地方，然后重复下去，直至走到不能走为止。

<div align=center>
![](./img/G-D.png)
</div>

### 梯度下降的直观理解

对于下坡来说，我们选择的方向就是代价函数在这一点的导数，而每一步的步长则是由学习率$\alpha$来决定的，具体表示为公式则为：

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) $$

j = 0,1表示特征的下标，梯度下降就是一直重复上述操作，直至$\theta_j$不再变化为止。

### 梯度下降的线性回归

将上述偏导写开来，就有如下：
$$
\begin{align}
\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) &= \frac{1}{2m}\sum_{1=1}^m\frac{\partial}{\partial \theta_j}  (h_{\theta}(x^{(i)})-y^{(i)})^2 \\

&= \frac{1}{m}\sum_{1=1}^m(h_{\theta}(x^{(i)})-y^{(i)})\frac{\partial}{\partial \theta_j}  (h_{\theta}(x^{(i)})-y^{(i)}) \\

&= \frac{1}{m}\sum_{1=1}^m(h_{\theta}(x^{(i)})-y^{(i)})\frac{\partial}{\partial \theta_j}  (\theta_0x_0+\theta_1x_1 - y) \\

&= \frac{1}{m}\sum_{1=1}^m((h_{\theta}(x^{(i)})-y^{(i)})x_0)
\end{align}
$$

### 提醒

+ 对于参数$\theta$需要同步更新，否则会出错。
+ 梯度下降算法可能会面临局部最小值的情况
+ 对于线性回归问题，梯度下降总能收缩到全局最小值
