# 神经网络

## 模型表示

神经网络被分为三部分，分别是输入层(input layer)，隐藏层(hidden layers)和输出层(output layer)

![](http://scruel.gitee.io/ml-andrewng-notes/image/20180116_001543.png)

+ 第一层为输入层，表示训练样本中特征的输入
 + $x_0$表示偏置单元(bias unit)，偏置单元恒为$1$
+ 最后一层为输出层，表示对于给定的输入得到的假设函数的输出。
+ 中间层被称为隐藏层，其包含的节点称之为激活单元(activation units)：
 + $a_i^{(j)}$表示第$j$层的第$i$个激活单元
 + $\theta^{(j)}$表示一个从第$j$层映射到第$j+1$层的权重矩阵
 + $\theta^{(j)}_{u,v}$表示从第$j$层的第$u$个单元映射到第$j+1$层的第$v$个单元的权重
 + 激活单元应用激活函数处理数据

---

+ 激活函数是$sigmoid$函数
+ $s_j$ 表示第$j$层激活单元数目
+ 每个单元都会作用于下一层的所有单元(向量化后做矩阵乘法即可)
+ 假设第$j$层有$s_j$个单元，第$j+1$层有$s_{j+1}$个单元，那么$\theta^{(j)}$就是一个维度为$s_{j+1} \times (s_j+1)$的权重矩阵
+ $+1$是偏置单元，除了输出层外，其它层可以添加偏置单元

## 前向传播

假设现在有一个三层的神经网络：

$$
\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ x_3\end{bmatrix}
\rightarrow
\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

那么各个激活单元的值以输出层的值计算过程为：

$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

求假设函数的过程就是一个前向传播的过程，即从输入层开始，一层一层的向下计算，并吧结果传递下去。

## 向量化实现

上面前向传播的计算过程为

$$\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline  \newline \end{align*}$$

这里定一个新的变量$z_k^{(j)}$，其表示该层激活函数的参数，同时也表示前一层传递来的结果,所以上述例子能表示为：

$$
\begin{align*}
	a_1^{(2)} = g(z_1^{(2)}) \\
    a_2^{(2)} = g(z_2^{(2)}) \\
    a_3^{(2)} = g(z_3^{(2)}) \\
\end{align*}
$$

$$
z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
$$

那么将其向量化后为：

$$
\begin{align*}a^{(1)} = x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*}
$$

所以有$z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$，$a^{(j)}=g(z^{(j)})$

由于权重矩阵的大小为$s_{j+1} \times (s_j+1)$，而第$j$层的激活单元矩阵大小为$m \times s_j$，所以$z^{(j+1)} = \theta^{(j)}a^{(j)T}$

---

+ m表示训练样本个数


