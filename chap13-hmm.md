13.2.1 Maximum likelihood for the HMM

如果已经观测到一个数据集$\mathbf{x}_1, \cdots, \mathbf{x}_N$，我们可以使用MLE确定HMM的参数。似然函数可以通过对联合分布求相对隐变量的边缘分布而获得
$$
p(\mathbf{X}\vert\boldsymbol{\theta}) = \sum_{\mathbf{z}}p(\mathbf{X,Z}\vert \boldsymbol{\theta}) \tag{13.11}
$$
因为联合分布$p(\mathbf{X,Z}\vert \boldsymbol{\theta})$没有分解为$n$(与第9章中考虑的混合分布相反)，我们不能单独处理$\mathbf{z}_n$上的每个求和。我们也不能显式地执行求和，因为有$N$个变量要求和，每个变量都有$K$个状态，结果总共有$KN$个项。因此，求和中的项数随着链的长度呈指数增长。事实上，(13.11)中的求和对应于图13.7中通过晶格图的指数多条路径的求和。

当我们考虑图8.32中简单变量链的推理问题时，我们已经遇到了类似的困难。在那里，我们能够利用**图的条件独立性质**对求和进行重新排序，从而得到一个算法，其代价与链的长度成线性而不是指数关系。我们将在隐马尔可夫模型中应用类似的技术。

似然函数表达式(13.11)的另一个困难在于，因为它对应于混合分布的推广，所以它代表了排放模型中潜在变量不同设置的总和。因此，似然函数的直接最大化将导致没有封闭形式解的复杂表达式，就像简单混合模型的情况一样（回想一下，i.i.d.数据的混合模型是HMM的特例）。

等式13.11的一个困难是，因为其对应于混合分布的一个概括，似然函数代表了不同隐变量背景下发射模型的加和。对似然函数的直接最大化将导致没有闭型解的复杂表达式，就像简单混合模型的情况一样(回想一下，i.i.d.数据的混合模型是HMM的特例)。

我们因此转向期望最大化算法(EM算法)寻找一个有效的框架来最大化hmm中的似然函数。EM算法以模型参数的初始化选择为起始点，我们记为$\boldsymbol{\theta}^{\text{old}}$。在E步，我们使用这些参数值，并寻找隐变量的后验分布$p(\mathbf{Z}\vert \mathbf{X}, \boldsymbol{\theta}^{\text{old}})$。我们然后使用这个后验分布来计算完整数据对数似然的期望$Q(\boldsymbol{\theta,\theta}^{\text{old}})$，这是$\boldsymbol{\theta}$的一个函数，定义为
$$
Q(\boldsymbol{\theta,\theta}^{\text{old}}) = \sum_{\mathbf{z}} p(\mathbf{Z\vert X},\boldsymbol{\theta}^{\text{old}})\ln  p(\mathbf{X,Z}\vert \boldsymbol{\theta}) \tag{13.12}
$$
我们令$\gamma(\mathbf{z}_n)$代表隐变量$\mathbf{z}_n$的边缘后验分布，令$\xi(\mathbf{z}_{n-1}, \mathbf{z}_{n-1})$代表两个连续隐变量的联合分布，因此有
$$
\begin{aligned}
    \gamma(\mathbf{z}_n) &= p(\mathbf{z}_n \vert \mathbf{X}, \boldsymbol{\theta}^{\text{old}}) \\
    \xi(\mathbf{z}_{n-1}, \mathbf{z}_{n-1}) &=  p(\mathbf{z}_n,  \mathbf{z}_{n-1} \vert \mathbf{X}, \boldsymbol{\theta}^{\text{old}}) 
\end{aligned}
$$
对于$n$的每个值，我们使用$\gamma(\mathbf{z}_n)$存储$K$个非负值，其加和为1，类似的，我们使用$K\times K$维矩阵存储$\xi(\mathbf{z}_{n-1}, \mathbf{z}_{n-1})$，其加和仍为1。我们使用$\gamma(z_{nk})$代表$z_{nk}=1$的条件概率，类似的，使用。因为一个二元随机变量的期望是其取值为1的期望，我们有
$$
\begin{aligned}
    \gamma(z_{nk}) &= \mathbb{E}[z_{nk}] = \sum_{\mathbf{z}}\gamma(\mathbf{z})z_{nk}        \\
    \xi(z_{n-1,j}, z_{nk}) &= \mathbb{E}[z_{n-1,j}, z_{nk}]  =  \sum_{\mathbf{z}}  \gamma(\mathbf{z}) z_{n-1,j} z_{nk}
\end{aligned}   \tag{13.15-13.16}
$$

如果我们给定13.10与13.12来替代联合分布$p(\mathbf{X,Z}\vert \boldsymbol{\theta})$，并使用$\gamma$与$\xi$的定义，我们得到
$$
\begin{aligned}
    Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}} ) &= \sum_{k=1}^K \gamma(z_{1k})\ln \pi_k   + \sum_{n=2}^N \sum_{j=1}^K \sum_{k=1}^K  \xi(z_{n-1,j}, z_{nk}) \ln A_{jk}  + \sum_{n=1}^N\sum_{k=1}^K \gamma(z_{nk})\ln p(\mathbf{x}_n\vert \boldsymbol{\phi}_k)
\end{aligned}       \tag{13.17}
$$
E步的目标是计算量$\gamma(\mathbf{z}_n)$与$\xi(\mathbf{z}_{n-1},\mathbf{z}_n)$。

在M步，我们的目标是相对参数$\boldsymbol{\theta}=\{\boldsymbol{\pi},\mathbf{A},\boldsymbol{\phi}\}$最大化$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}})$，其中我们将$\gamma(\mathbf{z}_n)$与$\xi(\mathbf{z}_{n-1},\mathbf{z}_n)$看作常数。通过使用拉格朗日乘自，相对$\boldsymbol{\pi},\mathbf{A}$的最大化是很容易实现的
$$
\begin{aligned}
    \pi_k &= \frac{}{}
\end{aligned}
$$
EM算法必须通过选择$\boldsymbol{\pi}$与$\mathbf{A}$的起始值来初始化，当然其必须满足相关概率解释的加和约束。注意到$\boldsymbol{\pi}$与$\mathbf{A}$的任意元素初始化为0，会导致在后续EM更新中保持为0。一个典型的初始化过程涉及到为这些参数随机选择一个初始值，这些初始值受到加和与非负约束。注意，对于左至右模型，除了为适当元素设置为零的元素$A_{jk}$选择初始值之外，不需要对EM结果进行任何特殊修改，因为这些结果将始终保持零。

为了相对$\boldsymbol{\phi}_k$最大化，我们需要注意到只有13.17中的最后一项依赖于$\boldsymbol{\phi}_k$，此外，该项与i.i.d.数据的标准混合分布的相应函数中的数据相关项具有完全相同的形式，通过与高斯混合情况下的(9.40)进行比较可以看出。这里数量$\gamma(z_{nk})$扮演者责任的角色。
例如在高斯发射密度的情况下$p(\mathbf{x}\vert \boldsymbol{\phi}_k) = \mathcal{N}(\mathbf{x}\vert \boldsymbol{\mu}_k, \mathcal{\Sigma}_k)$，

EM算法需要发射密度的参数的初始值。设置这些参数的一种方式是将数据看作iid，并使用最大似然拟合发射密度，然后使用得到的值来初始化EM的参数。

13.2.2 The forward-backward algorithm

