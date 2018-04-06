# 逻辑回归(Logistic Regression)

对于分类问题，顾名思义就是根据一系列的特征值，来对数据集分类，这也就意味着，分类的结果是离散的值（即每个值代表一种类型），例如：

+ 垃圾邮件判断
+ 金融欺诈判断
+ 肿瘤诊断

像上面的这些例子属于二分类问题，则定义输出的结果$y \in \{0, 1\}$，分别表示正类(Positive Class)和负类(Negative Class)。

如果用线性回归的方法来拟合分类问题，由于线性回归的输出是一个连续的值，所以要想能很好的表述分类，可能只能选定一个阈值(threshold)，来界定其分类。但是这样的表现的结果往往不好，在这个视频中介绍的方法是逻辑回归(Logistic Regression)

## 假设函数

视频中给出的逻辑回归模型为一个名为Sigmoid Function的模型，又称Logistic Function。这是一个S型函数，函数的值域为$h_\theta \in \{0, 1\}$。

逻辑回归问题的假设函数表示如下：

$$\begin{align*} 
	&h_\theta (x) = g ( \theta^T x ) \\
	&z = \theta^T x \\ 
	&g(z) = \dfrac{1}{1 + e^{-z}}
\end{align*}$$

![](http://scruel.gitee.io/ml-andrewng-notes/image/2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png)

其表现意义为，对于一个输入$x$，其输出$y=1$的概率是多少。例如$h_\theta=0.7$，表示有$70\%$的可能性认为输出是$1$。数学表现形式如下：

$$\begin{align*}
	& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \\
    & P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1 \\
\end{align*}$$


## 决策边界

决策边界就是分类的分界线。

对于一个二分类问题，我们预测的是一个离散的值，但上述的假设函数模型表现来看，其输出是一个连续的值，所以需要确定一个决策边界来解决这个问题，这个和用线性回归来拟合设置阀值是类似的解决方案。

设置0.5为阀值，如下

$$\begin{align*}
	& z = \theta^Tx	\\
	& h_\theta(z) \geq 0.5 \rightarrow y = 1 \\
    & h_\theta(z) < 0.5 \rightarrow y = 0 \\
\end{align*}$$

观察假设函数模型的图像，会发现，当$z \gt 0$时，$g(z) \geq 0.5$，综合起来就有：

$$\begin{align*}
	& \theta^T x \geq 0 \Rightarrow y = 1 \\
    & \theta^T x < 0 \Rightarrow y = 0 \\
\end{align*}$$

举个例子如下：

$$\begin{align*}
	& \theta = \begin{bmatrix}5 \newline -1 \newline 0\end{bmatrix} \\
    & y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0 \\
    & 5 - x_1 \geq 0 \\
    & - x_1 \geq -5 \\
    & x_1 \leq 5 \\
\end{align*}$$

所以得出的决策边界为一条$x_1 = 5$的直线，而$x_1 = 5$的左侧则表示$y=1$的情况

## 代价函数

同样的，也有一个代价函数来表示拟合的好坏程度，然后去维护$\theta$，使得代价函数达到最小值。若用线性回归的代价函数$(J(\theta)=\dfrac{1}{2m}\sum_{i=1}{m}(h_\theta(x^{(i)})-y^{(i)})^2)$来表示，会发现，代价函数是一个非凸函数，这样在进行梯度下降的时候会卡在局部最小值上，这样就无法判断是否收敛至全局最优。

<div align="center">
	![](http://scruel.gitee.io/ml-andrewng-notes/image/20180111_080314.png)
</div>

在这个视频中，对于逻辑回归函数使用的是对数损失函数，这是由统计学中的最大似然估计方法推出的代价函数$J(\theta)$

$$\begin{align*}
	& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\
    & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \\
    & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}
\end{align*}$$

代价函数的图片如下：

<div align="center">
	![](http://scruel.gitee.io/ml-andrewng-notes/image/20180111_080614.png)
</div>

就像图片上看到的，当$y=1$的时候，若预测到$h_\theta(x)=0$，就会得到一个特别大的惩罚，而当$h_\theta(x)=1$时，代价函数的值就为$0$，将相当于预测的很好。当$y=0$时也是一样的。

## 简化代价函数和梯度下降

将上面的代价函数可以合并为一个式子，相当于将分段函数合并了：

$$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

那么代价函数的表现形式为：

$$J(\theta) = - \dfrac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$$

向量化后的表现形式为：

$$\begin{align*}
	& h = g(X\theta)\\
    & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \\
\end{align*}$$

为了优化参数$\theta$，这里也采用梯度下降的算法，梯度下降算法的一般表现形式如下：

$$\begin{align*}
& Repeat \; \lbrace \\
	& \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \\
    & \rbrace
\end{align*}$$

求偏导后的表现形式其实和线性回归的梯度下降的表现形式是一样的，只是假设函数的形式不同而已：

$$\begin{align*} 
& Repeat \; \lbrace \\ 
	& \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \\ 
    & \rbrace 
\end{align*}$$

向量化后为：
$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$

## 梯度下降的推到过程

$$\begin{align*}
& J(\theta) = - \dfrac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))] \\
& \text{现设} f(\theta) = y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))	\\
& \because h_\theta(x^{(i)}) = g(z) = \dfrac{1}{1+e^{(-z)}} = \dfrac{1}{1+e^{-\theta^Tx^{(i)}}}	\\
& \therefore f(\theta) =  y^{(i)}\log (\dfrac{1}{1+e^{(-z)}}) + (1 - y^{(i)})\log (1 - \dfrac{1}{1+e^{(-z)}})	\\
& \therefore f(\theta) =  y^{(i)}\log (\dfrac{1}{1+e^{(-z)}}) + (1 - y^{(i)})\log (\dfrac{e^{(-z)}}{1+e^{(-z)}})	\\
& \therefore f(\theta) = \log (\dfrac{e^{(-z)}}{1+e^{(-z)}}) +y^{(i)} \log(\dfrac{1}{e^{-z}}) \\
& \because \dfrac{\partial z}{\partial \theta_j} = \dfrac{\partial \theta^Tx^{(i)}}{\partial \theta_j} = x^{(i)}_j	\\
& \therefore \dfrac{\partial}{\partial \theta_j}f(\theta) =- \dfrac{x^{(i)}_j}{1+e^{-z}} + y^{(i)}x^{(i)}_j	\\
& \therefore \dfrac{\partial}{\partial \theta_j}f(\theta) = -(h(\theta)-y^{(i)})x^{(i)}_j\\
& \therefore \dfrac{\partial}{\partial \theta_j}J(\theta) = -\dfrac{1}{m}\sum_{i=1}^{m}\dfrac{\partial}{\partial \theta_j}f(\theta) = \dfrac{1}{m}\sum_{i=1}^{m}(h(\theta)-y^{(i)})x^{(i)}_j
\end{align*}$$

## 高级优化(Advanced Optimization)

高级优化算法相比梯度下降能大大提高对数几率回归的运行的速率。对于梯度下降来说，其实就是对于给定的参数$\theta$计算**代价函数和代价函数的偏导**来更新参数$\theta$，一直重复直至最小化代价函数。但梯度下降并不是唯一的最小化算法，下面有一些最小化算法：

+ 梯度下降法(Gradient Descent)
+ 共轭梯度算法(Conjugate gradient)
+ BFGS
+ L-BFGS

相比于梯度下降算法，算法会更加复杂，很难进行调试，但不需要手动选择学习率，比梯度下降算法效率更高。

在Octave/Matlab对这类算法做了封装，调用方式如下：

```matlab
% 创建一个函数用于返回其代价函数和代价函数的偏导
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

```matlab
% 将上面创建的那个函数即所需要参数传入fminunc中，来解决最小化问题
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
%	'GradObj', 'on': 启用梯度目标参数
%	'MaxIter', '100'：最大迭代次数为100次
% 	@xxx: Octave/Matlab 中的函数指针
% 	optTheta: 最优化后得到的参数向量
% 	functionVal: 引用函数最后一次的返回值
% 	exitFlag: 标记代价函数是否收敛
```

## 多分类问题(Multiclass Classification: One-vs-all)

对数几率回归用于多分类问题中，采用的方法是将多分类问题转化为二分类问题，对于一个输入$x$，然后求出$h^{(i)}_\theta(x)$（表示属于第$i$个分类的可能性）的值，判断哪一个分类的值大，这个$x$就属于哪一类：

$$\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}$$

# 正则化(Regularization)
## 过拟合问题(The Problem of Overfitting

对于数据拟合的表现来说，分为三种情况：

+ **欠拟合(Underfitting)**，就是在训练集上拟合的情况就十分糟糕了，跟别说到测试集上的表现了
+ **优良的拟合(Just right)**，在训练集的表现很好，也有很好的泛化能力，在测试集表现的也不错
+ **过拟合(Overfitting)**，在训练集的表现终极好，甚至可以认为$J(\theta) \rightarrow0$，这样的话泛化能力就很差了，在做预测的时候表现就会很糟糕。

避免过拟合的方法有：

+ 减少特征的数量
	+ 手动选取需要保留的特征
	+ 使用模型选择算法来自动选取合适的特征
+ 正则化 
	+ 保留所有的特征，但是减少参数$\theta_j$的量级(magnitude)或大小
	+ 当有很多参数对于模型只有轻微影响时，正则化的表现很好

## 代价函数(Cost Function)

在很多时候由于特征数量过多，过拟合时无法选择要保留的特征，那么可以采用正则化方法来解决过拟合的问题。

在之前的最小化方法中，因为没有对$\theta$做限制，为了在保留所有的特征值且避免过拟合，于是加入一些惩罚项来减少$\theta$的取值。但是由于我们不知道要减少哪个参数，所以统一的惩罚所有的除$\theta_0$外的所有参数，所以代价函数表现形式如下：

$$J(\theta) =  \dfrac{1}{2m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2$$

$\lambda$表示正则化参数，类似于学习数率，也是需要自己选择的

+ **过大**会导致模型欠拟合，甚至梯度下降可能无法收敛
+ **过小**等于没有进行正则化，就不会避免过拟合

## 线性回归正则化(Regularized Linear Regression)

应用正则化后的线性回归梯度下降算法：

$$\begin{align*} 
	& \text{Repeat}\ \lbrace \\
    & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\
    & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline 
    & \rbrace \\
\end{align*}$$

这里$\dfrac{\lambda}{m}\theta_j$表示正则化项，对于$j=1$的时候通过移项可以合并成如下形式：

$$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

正则化后的正规方程表现形式如下：

$$
\begin{align*}
	& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \\
	& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \\ 
    									& 1 & & & \\
                                        & & 1 & & \\
                                        & & & \ddots & \\
                                        & & & & 1 \\
                          \end{bmatrix}
\end{align*}$$

这里引入了正则化项后会发现，解决了矩阵是否可逆的问题，因为加入了正则化项后，保证了$(X^TX + \lambda \cdot L)^{-1}$是可逆的

## 逻辑回归正则化(Regularized Logistic Regression)

同理，为逻辑回归添加正则化项后，代价函数表现如下：

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

梯度下降算法为：

$$\begin{align*} 
	& \text{Repeat}\ \lbrace \\
    & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\
    & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline 
    & \rbrace \\
\end{align*}$$