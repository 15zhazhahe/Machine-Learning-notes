# 多变量线性回归

当线性回归问题考虑到需要受多个变量影响时，该问题就为多变量线性回归问题

+ $x_j^{(i)}$第$j$个特征在第$i$个训练样本的取值
+ $x^{(i)}$第$i$个训练样本的取值
+ $m$表示训练样本的个数
+ $n$表示特征的数量

在多变量线性回归中，多考虑一个$x_0=1$假设函数为：

$$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n $$

向量化后，多变量的假设函数为：

$$
\begin{align*}
	h_\theta(x) =
	\begin{bmatrix}
		\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n
	\end{bmatrix}
	\begin{bmatrix}
		x_0 \newline x_1 \newline \vdots \newline x_n
    \end{bmatrix}
    = \theta^T x
\end{align*}
$$

## 梯度下降

在上一节课总结的时候，已经有意识的引入$x_0$这个变量为了让梯度下降的表示更加简洁，且用一个式子就能同时表示单变量和多变量线性回归

$$
\begin{align*}
&\text{for j := 0...n} \\
& \hspace{2em} 
\theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; 
\end{align*}
$$

## 特征缩放

由于特征各自的特性问题，所以他们的取值范围就不不一样，那么梯度下降算法可能就不能很好的工作(在下降的时候会不断震荡，从而影响算法的性能)。为了加速(speed up)梯度下降算法，我们采用了特征缩放(标准化特征数据的范围)来解决这个问题

+ feature scaling(视频的用词，感觉怪怪的)
	+ $x_i := \dfrac{x_i}{max-min}$
+ 均值归一化(mean normalization)
	+ $x_i := \dfrac{x_i - \mu_i}{s_i}$
	+ $\mu_i$表示$x_i$的平均值
	+ $s_i$可以用方差，也可以用极差($max-min$)

总而言之，特征缩放的目的是为了减少梯度下降算的迭代次数，所以对于具体缩放后的范围没有很精确的要求，只要大致标准化了特征数据的范围就可以了了。

### 学习率

+ 学习率过小，收敛速度过慢
+ 学习率过大，每次迭代不一定会下降，甚至不会收敛

## 多项式回归

有时候为了改善我们的假设函数，我们可能会考虑高次的问题，比如说原本所有特征都是用一次项来表示，但是有可能用高次项来拟合，效果可能会更好

例如，假设函数为$h_\theta(x) = \theta_0 + \theta_1 x_1$，我们基于$x_1$扩展，如得到一个二次函数$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$，或得到一个三次函数$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$。

那么对于扩展后的函数，如何求解呢，不如上面的三次函数，我们可以假设两个新的变量$x_2 = x_1^2$和$x_3 = x_1^3$，然后带入上述式子，接下来应该和线性回归一样吧(个人理解)

# 正规方程(Normal Equation)

梯度下降是通过不断的迭代来最小化代价函数，而正规方程这种方式其实是通过直接对$\theta_j$求导，通过这个求导直接使得代价函数取得最小值，得到正规方程的结果如下：

$$\theta = (X^T X)^{-1}X^T y$$

## 正规方程推到过程

+ $\theta$是一个$(n+1)*1$的向量
+ $X$是一个$m*(n+1)$的矩阵
+ $y$是一个$m*1$的矩阵

正规方程的推导，其实就是现将代价函数向量化后，然后对矩阵求导，就可以了得到结果了，由于不会对矩阵求导，所以这里只是依葫芦画瓢推的。

$$
\begin{align*}
    J(\theta) =& \frac{1}{2m}\sum_{1=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2 \\\\
    向量化后为:&\\\\
    J(\theta) =& \dfrac{1}{2m} (X \theta -y)^T(X \theta -y)\\\\
    求导得：&\\\\
    \frac{\partial}{\partial \theta_j}J(\theta)=&\dfrac{1}{2m}(\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^Ty)\\
    =&\dfrac{1}{2m}(2X^TX\theta-X^Ty-X^Ty)\\
    =&\dfrac{1}{2m}(2X^TX\theta-2X^Ty)\\
    导数等于零可得：&\\
    &2X^TX\theta-2X^Ty=0\\
    &(X^TX)^{-1}X^Ty=\theta\\
\end{align*}
$$

# Octave

本人电脑操作系统为Ubuntu16.04-gnome，故采用Octave作为开发工具

## 安装Octave

下面的命令是安装Octave的4.2.1版本，其他版本都可以自行选择，但千万别选4.0.0就对了，一开始装了提交不了作业
```matlab
sudo apt-add-repository ppa:octave/stable
sudo apt-get update
sudo apt-get install octave
```

## 基础操作

```matlab
PS1('>> ');			% 改变提示符，把一大串直接变为'>> '
a = 3;				% 赋值，加分号就不打印结果
% disp用于打印字符，sprintf类似C语言用于生成字符
disp(sprinf('2 decimals: %0.2f', a)) 
format long 		% 执行后，默认输出小数点后更长的值
A = [1 2; 3 4; 5 6]	% 生成一个矩阵，这将产生一个3x2的矩阵
v = 1:0.1:2			% 这将生成一个行向量[1 1.1 ... 1.9 2]
ones(2, 3)			% 将会生成一个2x3的全为1的矩阵
C = 2*ones(2, 3)	% 将会生成一个2x3的全为2的矩阵
zeros(1, 3)			% 会生成一个全为零的行向量
% 会生成一个3x3的矩阵，矩阵里的每一只值，都是0到1的随机值
rand(3, 3)			
% 得到的随机值降服从高斯分布，均值为0，标准差或方差为1
w = randn(1, 3)			
hist(w)				% 将根据w绘制一个直方图
eyes(4)				% 将得到一个4x4的一个单位矩阵
help rand			% 查看相关的相关文档
```

## 移动数据

```matlab
%% 移动数据
size(A)				% 可以返回一个1x2的矩阵，表示所查询矩阵的大小
size(A, 1)			% 将返回第一维的大小，即矩阵的行数
length(v)			% 将返回最大的维数
pwd					% 表示当前的路径
cd 'string'			% 打开到一个路径
load('filename')	% 打开一个文件
who					% 显示在内存中的所有变量
whos				% 显示所有变量的详细信息
clear name			% 将删除变量
a = A(1:10)			% 将A的前十个数据赋给了a
save filename		% 存成一个文件
A(3,:)				% 将返回A第三行的所有数据
A([1 3], :)			% 将得到第一行和第三行的所有数据
A(:,2) = [10; 11; 12]	% 将A的第二列用新的数据替代
A = [A, [100; 101; 102]]	% 在A的旁边多加一列
```

## 计算数据

```matlab
%% 点表示是针对元素的训练
A .* B				% 表示A和B中每个元素对应相乘
A .^ 2				% 对A的每个元素取平方
A + 1				% A中对应元素都加1
A'					% 表示A的转置
max(A)				% 求得A中每列的最大值，和最大值的位置
find(A < 3)			% 找到A中全部小于3的元素
A = magic(3)		% 返回一个3x3的幻方
sum(A)				% 对A求和
max(A, [], 1)		% 求得A每行的最大值
max(A, [], 2)		% 求得A每列的最大值
pinv(A)				% 表示求A的伪逆矩阵
```

## 数据可视化

```matlab
PS1('>> ');
t=[0:0.01:0.98];
y1=sin(2*pi*4*t);
plot(t,y1); 				% 绘制正弦函数
y2=cos(2*pi*4*t);
plot(t,y2); 				% 绘制余弦图，在此之前消除原来的正弦图
hold on;					% 将图像绘制在旧的上
plot(t,y1); 
plot(t,y1,'r'); 			% 绘制正弦图,颜色为红色
xlabel('time'); 			% 标识横坐标
ylabel('value'); 			% 标识纵坐标
legend('sin','cos'); 		% 标识曲线
title('myplot'); 			% 显示图的名字
print -dpng 'myplot.png' 	% 保存图像，png格式
help plot; 					% 查看plot帮助
close; 						% 关闭图像
figure(1); 	plot(t,y1);		% 新的图像
figure(2); plot(t,y2); 		% 又一个新的图像
close;
close;
figure(1);
subplot(1,2,1); 			% 将图像分成1*2的格子，使用第1个格子
plot(t,y1); 				% y1显示在左边
subplot(1,2,2); 			% 使用第2个格子
plot(t,y2); 				% y2显示在右边
axis([0.5 1 -1 1]); 		% 此时作用在右图上，修改了右图的横纵坐标
clf; 						% 清除一幅图像
A=magic(5);
imagesc(A);
colorbar; 					% 添加颜色条 颜色条表示不同深浅的颜色所对应的值
colormap gray; 				% 变成灰度图像
%% 上数三个命令可以用一行命令来代替:imagesc(A),colorbar,colormap gray（命令依次执行）
imagesc(magic(15)),colorbar,colormap gray % 生成15*15的灰度图像
```

## 控制语句

```matlab
%% for循环
fo i=1:10,
	....
end;

%% while循环
i = 1;
while i<=5,
	...
end;

%% if-else
if ... ,
	...
elseif ...,
	...
else
	...
end;

%% 函数
function 返回值一个或多个 = squareThisNumber(参数)
	.....
end

```