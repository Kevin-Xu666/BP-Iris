# 误差反向传播算法（BP）实现 **Iris** **数据分类** 

# 一、任务概述

通过反向传播算法实现鸢尾花数据的分类。

鸢尾花数据集是一个经典数据集，包含 3 类共 150 条记录，每类各 50 个数据，每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，可以通过这 4 个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。

分类任务是搭建一个神经网络、利用反向传播算法更新网络参数，在一定的迭代训练次数后，使网络能够较为准确地推断给定的花特征数据所代表的品种。

我仿照了一些机器学习框架的代码功能分割方式，将神经网络的代码拆分为数个部分、分工完成。最终完成了网络搭建、数据读取和反向传播训练，实现了鸢尾花的分类，达到了接近 100%的分类准确率。

# 二、反向传播算法分析

本次实现的神经网络分类任务中，最关键的是反向传播算法的实现。反向传播算法实质上是利用了链式法则的梯度下降法。

首先给出接下来的算法分析需要用到的符号与定义：

| $L$                                      | 神经网络层数                                           |
| ---------------------------------------- | ------------------------------------------------------ |
| $x^{(l)},z^{(l)},x_i^{(l)},z_j^{(l)}$    | 第 $l$ 层的输入与线性层直接输出                        |
| $y^{(l)},y_j^{(l)}$                      | 第 $l$ 层的激活                                        |
| $t,t_i$                                  | 正确标签为 1、其余位置为 0 的与输出等大的向量          |
| $W^{(l)},b^{(l)},W_{ij}^{(l)},W_j^{(l)}$ | $l-1$ 层与 $l$ 层间的权重矩阵/偏置向量及其对应位置元素 |
| $J(W,b;x,y),J$                           | 损失函数                                               |
| $f(z),f^{'}(z)$                          | 激活函数与导数                                         |
| $\delta ^{(l)},\delta_i^{(l)}$           | 残差及其元素                                           |
| $\eta$                                   | 学习率                                                 |

表 1：一些有关神经网络、机器学习的符号与定义

有了这些定义，我们开始进行下面的推导。

## （一）前向传播

在前向传播过程中，对于网络的任意一层，记为第 $l$ 层，有如下运算

$z^{(l)}=W^{(l)}x^{(l)}+b^{(l)},$  $y^{(l)}=f(z^{(l)})$  

而对于非输入层，其输入就是前一层的输出，即 

$x^{(L)}=y^{(l-1)},if$  $l > 1$ 

将一个数据点作为 $X^{(1)}$ 输入神经网络，迭代地在每一层进行上述运算，在最后一层得到输出 $y^{(L)}$ 就是网络对该数据点的推断结果。选取输出向量的最大元素，其位置就代表了对应的种类。这样，也就完成了神经网络对一个数据点的前向传播过程。

## （二）反向传播

根据梯度下降算法，对某一参数的优化，需要使其沿着损失函数梯度的反方向以一定的学习率进行下降，直到收敛。网络需要优化的参数为各层的权重与偏置。

依旧是对任意选取的第 $l$ 层，其权重矩阵梯度下降形式为

$W^{(l)}=W^{(l)}-\eta {\partial J\over \partial{W^{(l)}}}$ 

接下来需要求解第二项中的 Jacobian 矩阵。根据链式法则，有

${\partial J\over \partial{W_{ij}^{(l)}}}={\partial J\over \partial{z_i^{(l)}}} \cdot {\partial z_i^{(l)}\over \partial{W_{ij}^{(l)}}}={\partial J\over \partial{z_i^{(l)}}} \cdot {\partial \sum_{j=1}^{S_{l-1}}W_{ij}^{(l)}x_j^{(l)}+b_i^{(l)}\over \partial{W_{ij}^{(l)}}}={\partial J\over \partial{z_i^{(l)}}} \cdot x_j^{(l)} = \delta_i^l \cdot x_j^{(l)}$ 

将 ${\partial J\over \partial{z_i^{(l)}}}$ 记为第 $l$ 层的残差 $\delta_i^l$。

观察此式，可以得到矩阵形式的表示方法：

${\partial J\over \partial W^{(l)}}=\delta ^{(l)} \cdot x^{(l)} = \delta ^{(l)} \cdot (x^{(l)})^T$ 

偏置向量计算也类似地有：

${\partial J\over \partial b_i^{(l)}}={\partial J\over \partial z_i^{(l)}} \cdot {\partial z_i^{(l)}\over \partial b_i^{(l)}} = {\partial J\over \partial z_i^{(l)}} \cdot 1 = \delta_i^{(l)},{\partial J\over \partial b^{(l)}}=\delta^{(l)}$   

上式中未知的就是各层残差 $\delta$。只要求出了每一层的 $\delta$，也就可以进行相应的梯度下降。与神经网络进行推断的方向相反，残差的计算是由 ground truth 与结果的计算开始，也就是从最后一层开始。

最后一层残差的计算与具体的损失函数相关。这里以欧几里得距离为例

$J(y^L,t)={1\over 2}\sum_{j=1}^{S_L}(y_j^{(L)}-t_j)^2={1\over 2}\sum_{j=1}^{S_L}(f(z_j^{(L)})-t_j)^2$ 

$\delta_i^{(L)}={\partial J\over \partial z_i^{(L)}}={\partial J\over \partial y_i^{(L)}} \cdot {\partial y_i^{(L)}\over \partial z_i^{(L)}}=(f(z_i^{(L)})-t_i) \cdot f^{'}(z_i^{(L)})$ 

而非最后一层的残差则有递归关系： 

$\delta_i^{(l)}={\partial J\over \partial z_j^{(l)}}=\sum_{i=1}^m {\partial J\over \partial z_i^{(l+1)}} \cdot {\partial z_i^{(l+1)}\over \partial y_j^{(l)}} \cdot {\partial y_j^{(l)}\over \partial z_j^{(l)}}=(\sum_{i=1}^m {\partial J\over \partial z_i^{(l+1)}} \cdot W_{ij}^{(l+1)}) \cdot f^{'}(z_j^{(l)})$ 

​       $=f^{'}(z_j^{(l)}) \cdot \sum_{i=1}^m \delta_i^{(l+1)}W_{ij}^{(l+1)}$  

观察后也可以得到矩阵形式： 

$\delta^{(l)}=f^{'}(z^{(l)}) \cdot [(W^{(l+1)})^T \cdot \delta^{(l+1)}]$ 

至此，我们就推导完毕反向传播的数学过程，并得到了矩阵形式的表示与计算方法： 

$$\begin{cases}
{\partial J\over \partial W^{(l)}}=\delta ^{(l)} \cdot (x^{(l)})^T,{\partial J \over \partial b^{(l)}}=\delta^{(l)} \\
\delta^{(L)}={\partial J(y^{(L)},t)\over \partial y^{(L)}} \cdot f^{'}(z^{(L)}) \\
\delta^{(l)}=f^{'}(z^{(l)}) \cdot [(W^{(l+1)})^T \cdot \delta^{(l+1)}]
\end{cases}$$

按照上式的计算方法就可以得到每层的残差以及参数的 Jacobian 矩阵，也就可以进行梯度下降了。 

# 三、程序设计——神经网络

## （一）组织结构

在代码实现功能的划分上，我们小组参考了主流机器学习框架的结构。包含数据存储的矩阵类、层与网络类、损失函数类、激活函数类，以及将反向传播过程抽象出来的优化器类。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/%E4%BB%A3%E7%A0%81%E7%BB%84%E6%88%90%E7%BB%93%E6%9E%84.jpg)

                                                 图 1：代码组成结构

出于多态性考虑，优化器、损失函数与激活函数都定义了基类，供起具体效用的派生类公开继承。在需要使用相关类的函数形参都接受基类引用，传入派生类可以实现不同功能，以实现多态性。

在网络结构上，以 Linear 层为基本单元，将一层内各节点的输入、权重、偏置和输出都合并为矩阵，进行批量运算。然后以 Net 类作为整个网络的容器，其成员包含数个 Linear 对象。

## （二）基础数据结构

我们确定以矩阵运算作为基本运算操作，自然要以矩阵类作为基本的数据存 储方式，并需要定义一系列矩阵计算。我们在Matrix.h 文件中定义了模板类Matrix。类型参数决定了矩阵元素的数据类型。Matrix 类的成员包括行、列数目和 2 层vector 嵌套的数据单元。

矩阵类定义了常规的四则运算、求和运算、Hadamard 积、向量范数计算、初始化、输出等操作，虽然不够全面，但基本满足了全连接层与反向传播的计算和调试需要。

另外，为了提高效率，同时考虑到运算都比较简单，将绝大多数矩阵运算都声明为内联，因而仅在 Matrix.h 一个文件中进行了类的声明与成员函数的定义。

后面的网络搭建都是以具体类 Matrix<*double*> 类为基本数据单元。

## （三）全连接层

在Linear.h 和Linear.cpp 文件定义了 Linear 类及其方法。类声明如下。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/Linear%E7%B1%BB%E7%9A%84%E5%A3%B0%E6%98%8E.png)

                                         代码 1：Linear 类的声明

成员包括：权重矩阵、偏置向量、激活向量。激活的存储是为了方便反向传播的计算。

forward() 函数主要执行当前层的前向传播，起到 $z=Wx+b$ 的功能， 计算、存储并传播激活向量。

函数调用运算符的重载调用了 forward() 函数，使前向传播更加方便，也更贴近机器学习框架的简洁性。

initialization() 函数在构造函数中被调用，使用矩阵的初始化方法， 执行权重与偏置的初始化。可以执行给定范围的随机初始化，也有更加友好的 Xavier 初始化。这里实现的基本 Xavier 初始化功能是将偏置全部置 0，权重符号均匀分布。

$W$ ~ $U(-\sqrt{6\over S_l + S_{l-1}}, \sqrt{6\over S_l + S_{l-1}})$ 

## （四）损失函数

在LossBase/Euclidean/CrossEntropy.h/cpp 文件定义了损失函数类及其方法，相关代码都定义在 Loss 命名空间内。基类 LossBase 的声明如下。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/LossBase%E7%B1%BB%E7%9A%84%E5%A3%B0%E6%98%8E.png)

                                           代码 2：LossBase 类的声明

损失函数为函数类，重载了函数调用运算符，没有成员。

函数调用运算符的重载，接受网络输出的向量和输入数据的标签，返回损失值。

diff() 函数为损失函数导数的计算，接受最后一层的 $z$ 向量、激活函数和标签，返回向量 $\partial J/\partial z^{(L)}$，也即 $\delta^{(L)}$。 

在本任务中，我实现了 2 种激活函数：欧氏距离和交叉熵。

1、欧氏距离

首先，我们采用了基础的欧氏距离作为损失函数，即 $J(x,y)={1\over 2}||x-y||_2^2$ 

其中的 1/2 是为了简化求导的结果。而其具体的计算在第二节中已经推导，此处不再赘述。

2、交叉熵

传统的信息论交叉熵的定义为：对于 2 个序列 $p,q$，$H_p(q)=-\sum_i q_i log p_i$。 

而我们在这里采用的是 Pytorch 中使用的另一种交叉熵定义：对于预测结果 *x* 与真实结果 *y*，交叉熵为

$J(x,y)={1\over \sum_{n=1}^N \omega_{y_n}} \sum_{n=1}^N(-\omega_{y_n} log {e^{x_n}\over \sum_{i=1}^N e^{x_i}})$

这里的 $\omega_{y_n}$ 是真实结果 $y_n$ 所对应的权值。由于我们进行的是分类任务，不妨将正确类别的权值置 1，其他权值置 0。这样，我们可以继续化简：

$J(x,y)={1\over \sum_{n=1}^N \omega_{y_n}} \sum_{n=1}^N[-\omega_{y_n} (log {e^{x_n}- log \sum_{i=1}^N e^{x_i}})]$

​              $=log \sum_{i=1}^N e^{x_i} - {\sum_{n=1}^N \omega_{y_n} x_n\over \sum_{n=1}^N \omega_{y_n}}=log \sum_{i=1}^N e^{x_i} - x_t$ 

式中的 $x_t$ 指对应正确标签的预测输出值。这样，我们就得到了损失函数计算方法，

$J(y^L,t) = log \sum_{i=1}^N exp(y_i^{(L)})-y_t^{(L)}$ 

以及相应的残差计算方法

$\delta_i^{(L)}={\partial J\over \partial z_i^{(L)}} = {\partial J\over \partial y_i^{(L)}} \cdot {\partial y_i^{(L)}\over \partial z_i^{(L)}} = {\partial \over \partial y_i^{(L)}}(log \sum_{i=1}^Nexp(y_i^{(L)})-y_t^{(L)}) \cdot f^{'}(z_i^{(L)})$ 

​        $={exp(y_i^{(L)})\over \sum_{n=1}^N exp(y_n^{(L)})} - \delta_{in}$ 

式中 $\delta_{in}$ 为克罗内克 $\delta$ 符号，当 $i=n$ 时为 1，否则为 0。

采用了交叉熵损失函数后，网络准确率表现方面略优于欧氏距离。

## （五）优化器

优化器类是将反向传播过程抽绎出来的作为独立过程的一个类。我们也定义了优化器基类 OptimBase，和一个实际使用的 BGD（Batch Gradient Descent）类。相关代码都包含在 Optim 命名空间内。

BGD主要实现了误差反向传播和各层参数梯度下降的 2 个过程。梯度下降方法包含一般的梯度下降以及带动量的梯度下降。

记 $\theta$ 为待优化参数，$\alpha$ 为学习率，$i$ 为优化轮数。一般的梯度下降迭代为

$\theta_{i+1}=\theta_i - \alpha {\partial J\over \partial \theta_i}$ 

而加入动量项以后，更新规则则变为：

$v_{i+1}=\lambda v_i + \alpha {\partial J\over \partial \theta_i}$ 

$\theta_{i+1}=\theta_i - v_{i+1}$ 

式中 $\lambda$ 为动量衰减的系数，$v$ 为每一轮中的“动量”。可以看出，动量不仅包括了当前应该下降的梯度，也包括前面梯度的加权和。这使得梯度的下降能够保存其运动趋势，相对不容易陷入局部最优解。

BGD 类的声明如下。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/BGD%E7%B1%BB%E7%9A%84%E5%A3%B0%E6%98%8E.png)

                                           代码 3：BGD 类的声明

成员有网络的引用、损失函数引用、学习率、动量衰减率，以及 old_grad， 也就是各个参数梯度下降的动量。这些参数在构造函数中获得。

step() 函数执行反向传播与梯度下降操作，接受 1 个输入与标签，然后“前进一步”。

zero_grad() 函数顾名思义为清除梯度，负责清理动量 old_grad 和每一层中为反向传播保存的激活。

## （六）激活函数

 与前 2 项类似地，激活函数类也定义了基类 ActBase，以及 2 个具体执行功能 Sigmoid 和 ReLU 类。ActBase 类声明如下。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/ActBase%E7%B1%BB%E7%9A%84%E5%A3%B0%E6%98%8E.png)

                                           代码 4：ActBase 类的声明

激活函数也为函数类，重载了函数调用运算符，没有成员。函数调用运算符的重载，接受 z 向量，返回一个激活的向量。

① Sigmoid 函数为 $f(z_i)=1/(1+exp(-z_i)),$ 

② ReLU 函数为 $f(z_i)=max\{z_i,0\}$。

diff() 函数为激活函数导数的计算。

① Sigmoid 函数为 $f^{'}(z_i)=f(z_i)(1-f(z_i)),$ 

② ReLU 函数为 $f^{'}(z_i)=(sgn(z_i)+1)/2$。

## （七）网络整体

Net 类作为网络整体的容器，包含了多个全连接层和激活函数。类的声明如下。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/Net%E7%B1%BB%E7%9A%84%E5%A3%B0%E6%98%8E.png)

                                          代码 5：Net 类的声明

成员包含 Linear 层的 vector 容器和激活函数的引用。

在构造函数中，参数列表接收到输入输出维数与网络层数，并定义全网络共用的激活函数。

forward() 函数顺序调用容器中 Linear 层的 forward() 函数，将前一层的输出作为后一层的输入，最后返回网络对于输入的推测结果向量。

函数调用运算符的重载与 Linear 层一样，也是直接调用 forward()函数。

# 四、程序设计——数据载入与模型训练 

## （一）数据读取与划分

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/Load_data%20%E5%87%BD%E6%95%B0%E5%A3%B0%E6%98%8E.png)

                                        代码 6：load_data 函数声明

将每条数据的 4 个特征值放到 4×1 的向量中，然后构造std::vector<std::pair<Martrix<double>,int> 类型的训练集与测试集。构造一个函数load_data()，其声明如上。从字符串指定的文件中（即 iris.data）读取数据，同时记录 4 项特征值各自的最值，读取完毕后进行归一化，将数值范围限制在[0,1]。按照 4:1的比例划分训练集与测试集，存到由形参传入的训练集与测试集的引用中。

## （二）模型训练

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/main.cpp%E7%9A%84%E5%85%A8%E5%B1%80%E5%8F%98%E9%87%8F.png)

                                         代码 7：main.cpp 的全局变量

在全局变量中，我们以常量定义了网络的输入输出位数、隐藏层宽度、网络层数、和迭代轮数。然后定义了训练集与数据集数组。与 acc 和 loss 有关的是结果数据的记录。最后定义了神经网络的激活函数、网络结构、损失函数和优化器。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/main.cpp%E7%9A%84%E4%B8%BB%E5%87%BD%E6%95%B0%E9%83%A8%E5%88%86%E4%BB%A3%E7%A0%81.png)

                                           代码 8：main.cpp 的主函数部分代码

在主函数中，我们采用如上所示的简单的双循环，外层是训练轮数，内层则顺序输入打乱的训练集数据。对于每个数据点，调用一次 Net 类的前向传播过程、预测结果并保存相关信息，然后调用优化器的 step() 函数进行反向传播、更新网络参数。

在每一轮结束时，会输出当前一轮的训练集与测试集准确率。在训练全过程结束后，会在文件中保存每一轮的数据，展示训练过程的最佳准确度。并选取测试集的一个数据点，对比其预测结果与真实种类。

# 五、实验结果

在一定的尝试后，我选取了一个表现较好的网络结构：3 层全连接层、隐藏层宽度为4。如下图所示。这样的网络并不深，容易收敛，也不容易出现过拟合以及梯度消失等现象。重要的是，在该网络结构下取得了比较好的准确率。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/%E9%80%89%E5%AE%9A%E7%9A%84%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.jpg)

                                                图 2：选定的网络结构

在确定网络结构后，我们进行了激活函数、损失函数、学习率等超参数的修改调试，确定了一些选择。激活函数使用 Sigmoid，权重初始化使用对 Sigmoid 友好的 Xavier 初始化。训练轮数选取多数可以实现收敛的 500 轮。下面给出了2 种收敛比较快的结果。

① 当选择欧氏距离损失函数，学习率 0.02 时，在 300 多轮基本收敛。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res1.jpg)

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res2.jpg) 

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res3.jpg)

图 3~5：选择欧氏距离损失函数的收敛情况，从上到下依次为最佳结果、准确率和损失。图像由 MATLAB绘制，下同

数据点检测结果符合实际。最终达到 96.67%训练集准确率，100%测试集准确率。

② 当选择交叉熵损失函数，学习率 0.005 时，在 100 多轮就基本收敛。

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res4.jpg) 

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res5.jpg) 

![img](https://github.com/Kevin-Xu666/BP-Iris-/blob/main/IMG/res6.jpg) 

图 6~8：选择交叉熵损失函数的收敛情况，从上到下依次为最佳结果、准确率和损失

数据点检测结果符合实际。最终达到 99.17%训练集准确率，100%测试集准确率。

# 六、讨论分析 


## （一）参数调优

最初在使用交叉熵函数的时候，测试集准确率偶尔会收敛在 66.67%左右， 疑似陷入局部最优解。在进一步测试调优之后，通过使用 Xavier 初始化，使得神经网络准确率能够稳定收敛在 99.1667%，跳出局部最优解。

## （二）创新提高

1. 结构工整合理，参考了机器学习代码框架。

2. 采用 Optimizer 优化器 BGD 的动量表示，使神经网络参数能够更好的收敛。

3. 采用交叉熵损失函数，进一步提高收敛速度和准确率。 

## （三）不足之处

基础数据结构为矩阵，维数受到限制，因而数据只能通过循环逐个传入网络。而且缺乏张量那样的对齐、广播等功能。

使用的机器学习各个组件稍显单调。尤其是优化器方面由于某些错误，原定的 Adam 方法没能实现。实现的普通动量也有时会陷入局部最优点，这种情况有望在引入 Nesterov 动量后得到改善。

# 参考文献

[1] Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[C]. Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010: 249-256.

[2] CrossEntropyLoss — PyTorch 1.11.0 documentation[EB/OL].

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

 

[3] Pytorch 中交叉熵损失 nn.CrossEntropyLoss()的真正计算过程[EB/OL]. https://blog.csdn.net/qq_44523137/article/details/120557043



 
