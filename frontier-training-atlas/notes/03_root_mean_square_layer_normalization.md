<!-- 来源：https://arxiv.org/pdf/1910.07467；翻译范围：摘要至参考文献/附录前 -->

# 均方根层归一化

###### 摘要

层归一化（LayerNorm）已成功应用于各种深度神经网络，以帮助稳定训练并加速模型收敛，因为它能够处理输入和权重矩阵的重新中心化（re-centering）和重新缩放（re-scaling）。
然而，LayerNorm引入的计算开销使得这些改进代价高昂，并显著减慢了底层网络的速度，例如RNN尤为明显。
在本文中，我们假设LayerNorm中的重新中心化不变性是可有可无的，并提出了均方根层归一化（RMSNorm）。RMSNorm根据均方根（RMS）对一层中神经元的求和输入进行正则化，赋予模型重新缩放不变性和隐式学习率自适应能力。
RMSNorm在计算上更简单，因此比LayerNorm更高效。
我们还提出了部分RMSNorm，即$p$RMSNorm，其中RMS是根据$p$%的求和输入估计的，而不会破坏上述性质。
在多个任务上使用不同网络架构进行的大量实验表明，RMSNorm取得了与LayerNorm相当的性能，但在不同模型上将运行时间减少了7%$\sim$64%。源代码可在 https://github.com/bzhangGo/rmsnorm 获取。

## 1 引言

如何高效地训练深度神经网络是一个长期存在的挑战。为了加速模型收敛，Ba等人（2016）提出了层归一化（LayerNorm），通过均值和方差统计量对同一层内的神经元动态进行正则化，从而稳定深度神经网络的训练。由于其简单性且不需要训练样本之间的依赖关系，LayerNorm已被广泛应用于不同的神经架构，在从计算机视觉[19, 26]、语音识别[37]到自然语言处理[31, 35]的各种任务中取得了显著的成功。在某些情况下，LayerNorm被发现对于成功训练模型至关重要[6]。此外，与基于批次的样本解耦使得LayerNorm在使用RNN处理变长序列时，具有优于批归一化（BatchNorm）[12]的优势。

**(a) 训练损失与训练步数的关系。**

**(b) 训练损失与训练时间的关系。**

**图1：基于GRU的RNNSearch [4] 在前1万个训练步的训练过程。Baseline（基线）指没有任何归一化的原始模型。当Baseline的训练损失达到7.0时，LayerNorm的损失在相同的训练步数后达到5.4（图1(a)），但在相同的训练时间后仅达到5.9（图1(b)）。**

不幸的是，引入LayerNorm增加了计算开销。尽管对于只有少数归一化层的小型和浅层神经模型来说，这可以忽略不计，但当底层网络变得更大更深时，这个问题就变得严重了。结果，由更快、更稳定的训练（就训练步数而言）带来的效率提升，被每个训练步增加的计算成本所抵消，从而降低了净效率，如图1所示。LayerNorm的一个被广泛认为有助于稳定性的主要特征是其重新中心化不变性：当输入或权重矩阵发生一定量的噪声偏移时，经过LayerNorm后的求和输入保持不变。我们认为，这种均值归一化并不能减少隐藏状态或模型梯度的方差，并假设它对LayerNorm的成功影响甚微。

在本文中，我们提出了均方根层归一化（RMSNorm），它仅使用均方根（RMS）统计量对一层中神经元的求和输入进行正则化。与LayerNorm相比，RMSNorm减少了计算量并提高了效率。尽管公式更简单，但RMS归一化器有助于稳定层激活的幅度，确保对权重和数据集重新缩放的不变性。
我们还展示了在求和输入的子集上估计RMS的可能性，同时保持这种不变性。假设求和输入具有独立同分布的结构，我们提出了部分RMSNorm，其中仅利用前$p$%的求和输入进行RMS估计。

我们在各种任务上全面检验了我们的模型，包括机器翻译、图像分类、图像-描述检索和问答。实验结果表明，在不同的模型中，RMSNorm取得了与LayerNorm相当的性能，但在运行速度方面表现出优越性，加速比达到7%$\sim$64%。当使用部分（6.25%）求和输入估计RMS时，$p$RMSNorm取得了与RMSNorm相媲美的性能。

## 2 相关工作

深度神经网络被假设面临的一个瓶颈是内部协变量偏移（internal covariate shift）问题[27]，即一层的输入分布随着前一层的更新而发生变化，这显著减慢了训练速度。（脚注：请注意，内部协变量偏移被[12, 3]作为动机提出。最近的研究
提出了归一化成功的其他解释，例如未归一化深度网络中层激活的不可控增长[5]。）解决这个问题的一个有希望的方向是归一化。Ioffe和Szegedy（2015）引入了批归一化（BatchNorm），基于从每个训练小批量（mini-batch）中估计的均值和方差统计量来稳定激活。不幸的是，对训练样本之间的依赖使得BatchNorm失去了处理变长序列的能力，尽管一些研究人员开发了不同的策略以使其能够在RNN中使用[16, 8]。相反，Salimans和Kingma（2016）提出了权重归一化（WeightNorm）来重新参数化权重矩阵，从而将权重向量的长度与其方向解耦。Ba等人（2016）提出了层归一化，它与BatchNorm的不同之处在于，统计量直接从同一层估计，而无需访问其他训练样本。由于其简单性和有效性，LayerNorm已成功应用于各种深度神经模型，并在不同任务上取得了最先进的性能[19, 37, 31, 6]。

这些研究开创了将归一化作为模型架构一部分的研究方向。这种范式通过缩短模型收敛时间确保了令人鼓舞的性能，但代价是每个运行步消耗更多的时间。为了提高效率，Arpit等人（2016）采用了一种与数据无关的方法来近似估计均值和方差统计量，从而避免了计算批次统计量。Ioffe（2017）提出了批重归一化（batch renormalization），以减少BatchNorm中对小批量的依赖。Ulyanov等人（2016）在图像生成中用实例归一化（instance normalization）取代了批归一化。Hoffer等人（2018）和Wu等人（2018）观察到，$l1$-范数可以作为BatchNorm中方差的替代方案，其优点是具有更少的非线性操作和更高的计算效率。然而，所有这些工作仍然遵循原始的归一化结构，并利用从整个求和输入中估计的均值统计量来处理重新中心化不变性。

与这些相关工作不同，所提出的RMSNorm通过移除重新中心化操作并仅使用RMS对求和输入进行正则化，修改了归一化结构。我们的模型仅保留了重新缩放不变性，部分受组归一化（group normalization）[34]的启发，我们发现当仅从求和输入的子集估计RMS时，这种性质可以被继承。作为附带效果，我们的模型减少了计算开销并提高了效率。最近，Zhang等人（2019）表明，通过仔细的初始化，残差网络的训练可以像带有归一化的网络一样稳定。然而，该方法主要旨在改进残差网络，并且在不修改所有初始化层的情况下无法自由切换。此外，将其调整到其他通用神经网络（例如模型深度随可变序列长度扩展的RNN）并非易事。相比之下，我们的模型简单、有效，并且可以作为LayerNorm的直接替代品（drop-in replacement）使用。

## 3 背景

在本节中，我们基于标准的前馈神经网络简要回顾LayerNorm。给定一个输入向量$\mathbf{x}\in\mathbb{R}^{m}$，前馈网络通过线性变换和随后的非线性激活将其投影为输出向量$\mathbf{y}\in\mathbb{R}^{n}$，如下所示：

$$
a_{i}=\sum_{j=1}^{m}{w_{ij}x_{j}},\quad y_{i}=f\left(a_{i}+b_{i}\right),
$$
(1)

其中$\mathbf{w}_{i}$是第$i$个输出神经元的权重向量，$b_{i}$是通常初始化为0的偏置标量，$f(\cdot)$是逐元素的非线性函数。$\mathbf{a}\in\mathbb{R}^{n}$表示神经元的权重求和输入，这也是归一化的目标。

这种普通的网络可能会受到内部协变量偏移问题[12]的影响，即一层的输入分布随着前一层的更新而发生变化。这可能会对参数梯度的稳定性产生负面影响，从而延迟模型收敛。为了减少这种偏移，LayerNorm对求和输入进行归一化，以固定其均值和方差，如下所示：

$$
\bar{a}_{i}=\frac{a_{i}-\mu}{\sigma}g_{i},\quad y_{i}=f\left(\bar{a}_{i}+b_{i}\right),
$$
(2)

其中$\bar{a}_{i}$是向量$\mathbf{\bar{a}}\in\mathbb{R}^{n}$的第$i$个值，它作为$a_{i}$的归一化替代项用于层激活。$\mathbf{g}\in\mathbb{R}^{n}$是用于重新缩放标准化求和输入的增益参数，初始设置为1。$\mu$和$\sigma^{2}$分别是根据原始求和输入$\mathbf{a}$估计的均值和方差统计量：

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}a_{i},\quad\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(a_{i}-\mu)^{2}}.
$$
(3)

因此，LayerNorm迫使神经元的范数与输入和权重矩阵解耦。

## 4 RMSNorm

关于LayerNorm成功的一个著名解释是其重新中心化和重新缩放不变性。前者使模型对输入和权重上的偏移噪声不敏感，而后者在输入和权重被随机缩放时保持输出表示完好无损。
在本文中，我们假设重新缩放不变性是LayerNorm成功的原因，而不是重新中心化不变性。

我们提出了RMSNorm，它仅关注重新缩放不变性，并简单地根据均方根（RMS）统计量对求和输入进行正则化：

$$
\displaystyle\begin{split}&\bar{a}_{i}=\frac{a_{i}}{\text{RMS}(\mathbf{a})}g_{i},\quad\text{where}~~\text{RMS}(\mathbf{a})=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_{i}^{2}}.\end{split}
$$
(4)

直观地说，RMSNorm通过完全移除等式（3）中的均值统计量来简化LayerNorm，代价是牺牲了均值归一化所提供的不变性。当求和输入的均值为零时，RMSNorm完全等同于LayerNorm。尽管RMSNorm没有像LayerNorm那样对求和输入进行重新中心化，但我们通过实验证明，这一性质对于LayerNorm的成功并非根本性的，并且RMSNorm具有相似或更高的有效性。

| | 权重矩阵 | 权重矩阵 | 权重向量 | 数据集 | 数据集 | 单个训练样本 |
| --- | --- | --- | --- | --- | --- | --- |
| | 重新缩放 | 重新中心化 | 重新缩放 | 重新缩放 | 重新中心化 | 重新缩放 |
| BatchNorm | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| WeightNorm | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| LayerNorm | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| RMSNorm | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| $p$RMSNorm | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |

**表1：不同归一化方法的不变性。“✓”表示具有不变性，而“✗”表示相反。**

RMS测量输入的平方平均值，在RMSNorm中，这迫使求和输入进入一个按$\sqrt{n}$缩放的单位球面。通过这样做，无论输入和权重分布如何缩放，输出分布都保持不变，这有利于层激活的稳定性。尽管与RMS仅相差一个因子$\sqrt{n}$的欧几里得范数已被成功探索[22]，但我们凭经验发现它不适用于层归一化。我们假设根据输入向量的大小来缩放球面是很重要的，因为这使得归一化在不同大小的向量上更加鲁棒。据我们所知，将RMS用于神经网络归一化的想法以前尚未被研究过。

### 4.1 不变性分析

不变性衡量归一化后的模型输出是否随其输入和权重矩阵发生高度一致的变化。Ba等人（2016）表明，不同的归一化方法展现出不同的不变性，这对模型的鲁棒性有很大贡献。在本节中，我们从理论上检验RMSNorm的不变性。

我们考虑RMSNorm的以下一般形式：

$$
\mathbf{y}=f\left(\frac{\mathbf{Wx}}{\text{RMS}(\mathbf{a})}\odot\mathbf{g}+\mathbf{b}\right),
$$
(5)

其中$\odot$表示逐元素乘法。我们的主要结果总结在表1中。由于RMS具有以下线性性质，RMSNorm对权重矩阵和输入的重新缩放均具有不变性：

$$
\text{RMS}(\alpha\mathbf{x})=\alpha\text{RMS}(\mathbf{x}),
$$
(6)

其中$\alpha$是一个缩放值。
假设权重矩阵按因子$\delta$进行缩放，即$\mathbf{W}^{\prime}=\delta\mathbf{W}$，那么这种变化不会影响最终的层输出：

$$
\small\begin{split}\mathbf{y}^{\prime}=f\left(\frac{\mathbf{W^{\prime}x}}{\text{RMS}(\mathbf{a}^{\prime})}\odot\mathbf{g}+\mathbf{b}\right)=f\left(\frac{\delta\mathbf{Wx}}{\delta\text{RMS}(\mathbf{a})}\odot\mathbf{g}+\mathbf{b}\right)=\mathbf{y}.\end{split}
$$
(7)

相比之下，如果缩放仅在单个权重向量上执行，则该性质不再成立，因为不同的缩放因子破坏了RMS的线性性质。
类似地，如果我们对输入施加一个因子为$\delta$的缩放，即$\mathbf{x}^{\prime}=\delta\mathbf{x}$，通过类似于等式7的分析，RMSNorm的输出保持不变。我们可以轻松地将该等式扩展到基于批次的输入以及整个数据集。因此，RMSNorm对其输入的缩放具有不变性。

与LayerNorm的主要区别在于，RMSNorm没有重新中心化，因此对于变量偏移不表现出类似的线性性质。它对所有重新中心化操作都不具有不变性。

### 4.2 梯度分析

上述分析仅考虑了缩放输入和权重矩阵对层输出的影响。然而，在一般设置中，增强了RMSNorm的神经网络是通过标准的随机梯度下降方法进行训练的，其中模型梯度的鲁棒性对于参数更新和模型收敛至关重要（另见Santurkar等人（2018），他们认为归一化方法的成功并非来自于层输入稳定性的增加，而是由于优化地形平滑度的提高）。在本节中，我们研究RMSNorm中模型梯度的性质。

给定损失函数 $\mathcal{L}$，我们通过公式 (4) 进行反向传播，以获得关于参数 $\mathbf{g},\mathbf{b}$ 的梯度，如下所示：

$$
\displaystyle\frac{\partial\mathcal{L}}{\partial\mathbf{b}}=\frac{\partial\mathcal{L}}{\partial\mathbf{v}},\quad\frac{\partial\mathcal{L}}{\partial\mathbf{g}}=\frac{\partial\mathcal{L}}{\partial\mathbf{v}}\odot\frac{\mathbf{Wx}}{\text{RMS}(\mathbf{a})},
$$
(8)

其中 $\mathbf{v}$ 是公式 (4) 中 $f(\cdot)$ 内整个表达式的简写，$\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{v}}}$ 是从 $\mathcal{L}$ 反向传播到 $\mathbf{v}$ 的梯度。梯度 $\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{b}}}$ 和 $\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{g}}}$ 都对输入 $\mathbf{x}$ 和权重矩阵 $\mathbf{W}$ 的缩放具有不变性（在 $\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{g}}}$ 的情况下，这是因为公式 (6) 中的线性性质）。此外，$\mathbf{g}$ 的梯度与归一化后的求和输入成正比，而不是原始输入。这增强了 $\mathbf{g}$ 幅度的稳定性。

与这些向量参数不同，由于 RMS 中的二次计算，权重矩阵 $\mathbf{W}$ 的梯度更为复杂。形式上，

$$
\displaystyle\small\begin{split}&\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=\sum_{i=1}^{n}\left[\mathbf{x}^{T}\otimes\left(\text{diag}\left(\mathbf{g}\odot\frac{\partial\mathcal{L}}{\partial\mathbf{v}}\right)\times\mathbf{R}\right)\right]_{i},\text{where}~~\mathbf{R}=\frac{1}{{\text{RMS}(\mathbf{a})}}\left(\mathbf{I}-\frac{\left(\mathbf{Wx}\right)\left(\mathbf{Wx}\right)^{T}}{n\text{RMS}(\mathbf{a})^{2}}\right),\end{split}
$$
(9)

$\text{diag}(\cdot)$ 表示输入的对角矩阵，$\otimes$ 表示克罗内克积，而“$\mathbf{I}$” 表示单位矩阵。为了清晰起见，我们明确使用 “$\times$” 来表示矩阵乘法。矩阵项 $\mathbf{R}$ 将 $\mathbf{W}$ 的梯度与输入 $\mathbf{x}$ 和权重矩阵 $\mathbf{W}$ 联系起来。通过深入分析，我们可以证明该项与输入和权重矩阵的缩放呈负相关。在为输入 $\mathbf{x}$ ($\mathbf{x}^{\prime}=\delta\mathbf{x}$) 或权重矩阵 ($\mathbf{W}^{\prime}=\delta\mathbf{W}$) 分配缩放比例 $\delta$ 后，我们得到

$$
\small\begin{split}\mathbf{R}^{\prime}&=\frac{1}{{\delta\text{RMS}(\mathbf{a})}}\left(\mathbf{I}-\frac{\left(\mathbf{\delta Wx}\right)\left(\mathbf{\delta Wx}\right)^{T}}{n\delta^{2}\text{RMS}(\mathbf{a})^{2}}\right)=\frac{1}{\delta}\mathbf{R}.\\
\end{split}
$$
(10)

如果我们将缩放后的项 $\mathbf{R}^{\prime}$ 代回公式 (9)，我们可以很容易地证明梯度 $\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{W}}}$ 对输入缩放具有不变性，但保持与权重矩阵缩放的负相关性。降低梯度 $\nicefrac{{\partial\mathcal{L}}}{{\partial\mathbf{W}}}$ 对输入缩放的敏感性确保了其平滑性并提高了学习的稳定性。另一方面，这种负相关性充当了隐式的学习率自适应器，并动态控制梯度的范数，从而避免了大范数的权重矩阵并改善了模型收敛性。

## 5 $p$RMSNorm

RMSNorm 的重缩放不变性归因于 RMS 的线性性质。考虑到同一层中的神经元通常具有独立同分布的结构，我们认为可以在这些神经元的子集上而不是全部神经元上估计 RMS。我们提出了部分 RMSNorm ($p$RMSNorm)。给定未归一化的输入 $\mathbf{a}$，$p$RMSNorm 从 $\mathbf{a}$ 的前 $p$% 元素中推断 RMS 统计量：
$\overline{\text{RMS}}(\mathbf{a})=\sqrt{\frac{1}{k}\sum_{i=1}^{k}a_{i}^{2}},$
其中 $k=\lceil n\cdot p\rceil$ 表示用于 RMS 估计的元素数量。如公式 (6) 所示，线性性质对于 $\overline{\text{RMS}}$ 仍然成立，这表明 $p$RMSNorm 具有与 RMSNorm 相同的不变性，如表 1 所示。

$\overline{\text{RMS}}$ 是 RMS 的有偏估计，通常是不准确的。尽管在理论上 $p$RMSNorm 近似于 RMSNorm，但我们观察到了梯度不稳定性，即在 $m$ 较小时梯度容易爆炸。然而在实践中，使用 $p$RMSNorm 的模型在部分比例为 6.25% 时能够成功达到令人满意的收敛。

## 6 实验

为了测试层归一化在不同实现中的效率，我们在 Tensorflow [1]、PyTorch [20] 和 Theano [29] 上进行了实验。
我们将 RMSNorm 添加到不同的模型中，并与未归一化的基线和 LayerNorm 进行比较。这些模型基于多种架构，涵盖了不同的 RNN 变体、卷积和自注意力模型，以及各种激活函数（如 sigmoid、tanh 和 softmax），初始化方式包括均匀分布、正态分布、正交分布，并具有不同的初始化范围或方差。除非另有说明，所有与速度相关的统计数据均在一块 TITAN X (Pascal) 上测量。报告的时间是 3 次运行的平均值。我们还列出了这三次运行的标准差。

### 6.1 机器翻译

机器翻译旨在将句子从一种（源）语言转换为另一种（目标）语言。我们专注于基于注意力增强的编码器-解码器框架的神经机器翻译。我们在 WMT14 英德翻译任务上训练了两种不同的模型：基于 GRU 的 RNNSearch [4] 和基于自注意力的神经 Transformer [31]。关于实验设置的更多细节以及与 WeightNorm 的比较列在附录 A.1 中。

**图 2：RNNSearch 在 newstest2013 上的 SacreBLEU 分数。模型根据 Nematus [25] 在 Tensorflow 中实现。**

| 模型 | Test14 | Test17 | 时间 |
| --- | --- | --- | --- |
| 基线 | 21.7 | 23.4 | 399$\pm$3.40s |
| LayerNorm | 22.6 | 23.6 | 665$\pm$32.5s |
| L2-Norm | 20.7 | 22.0 | 482$\pm$19.7s |
| RMSNorm | 22.4 | 23.7 | 501$\pm$11.8s (24.7%) |
| $p$RMSNorm | 22.6 | 23.1 | 493$\pm$10.7s (25.9%) |

**表 2：使用 Tensorflow 版本 Nematus 的 RNNSearch 在 newstest2014 (Test14) 和 newstest2017 (Test17) 上的 SacreBLEU 分数。“时间”：每 1k 个训练步的时间（秒）。我们将 $p$ 设置为 6.25%。我们用粗体突出显示最佳结果，并在括号中显示 RMSNorm 相对于 LayerNorm 的加速比。**

我们首先在 RNNSearch 上进行实验。归一化被添加到循环连接和前馈层中。除了没有任何归一化的 RNNSearch（基线）和带有 LayerNorm 的 RNNSearch 之外，我们还与配备 L2-Norm 的相同模型（即用 L2-Norm 替换 RMS）进行了比较，据观察这可以改善词汇选择 [18]。

图 2 展示了在我们的开发集上每 30k 个训练步后 BLEU 分数的演变，表 2 总结了测试结果。简而言之，LayerNorm 和 RMSNorm 都通过加速模型收敛优于基线：它们将收敛前的训练步数减少了约 50%，并提高了测试准确率，其中 RMSNorm 与 LayerNorm 表现相当。
这支持了我们的假设，即重缩放不变性是 LayerNorm 的核心属性，而 RMSNorm 是一个有效的替代方案。
我们使用 L2-Norm 的结果表明它未能改善模型。（脚注：我们注意到 Nguyen 和 Chiang 2017 仅将 L2-Norm 应用于最后一层，并将缩放因子视为超参数。虽然这不是对他们实验的复制，但我们仍然认为值得测试 L2-Norm 作为 LayerNorm 的替代方案。）表 2 中的结果突显了一个挑战，即 Tensorflow 中带有 LayerNorm 的 RNN 存在严重的计算效率低下问题，其中 LayerNorm 比基线慢约 67%。在这方面，RMSNorm 表现得明显更好，比 LayerNorm 提高了 $\sim$25%。

表 3 进一步列出了在 Theano 和 Pytorch 中实现的不同模型的翻译结果。总体而言，与 LayerNorm 相比，RMSNorm 产生了相当的翻译质量，但产生的计算开销更少，优于 LayerNorm，加速比在 11%$\sim$34% 之间。此外，我们观察到，尽管在理论上 $p$RMSNorm 的计算量少于 RMSNorm，但 $p$RMSNorm ($p=6.25\%$) 有时往往更慢。我们将此归因于这些计算框架中张量切片操作的非最优实现，这可以通过特定的底层编码来改进。

在 $p$RMSNorm 中，部分比例 $p$ 直接控制估计 RMS 的准确性，从而影响模型训练的稳定性。图 7 显示了 $p$ 对模型性能的影响。令人惊讶的是，我们发现 $p$ 的大小对 RNNSearch 中的最终翻译质量几乎没有影响：使用较小的比例不会显著降低 BLEU 分数。在所有后续实验中，我们将 $p$ 设置为 6.25%。

**图 3：带有 $p$RMSNorm 的 RNNSearch 在 newstest2013（开发集）上的 SacreBLEU 分数。我们使用 Tensorflow 版本的 Nematus，并以 10% 的步长更改 $p$。**

| | 模型 | Test14 | Test17 | 时间 |
| --- | --- | --- | --- | --- |
| Th | 基线 | 21.8 | 22.9 | 596$\pm$20.8s |
| LayerNorm | 22.3 | 23.8 | 988$\pm$1.10s | |
| RMSNorm | 22.5 | 23.2 | 652$\pm$24.1s (34.0%) | |
| $p$RMSNorm | 22.7 | 24.0 | 658$\pm$17.9s (33.4%) | |
| Py | 基线 | 22.7 | 24.7 | 427$\pm$6.50s |
| LayerNorm | 23.2 | 24.3 | 857$\pm$17.2s | |
| RMSNorm | 22.9 | 24.5 | 763$\pm$16.2s (11.0%) | |
| $p$RMSNorm | 23.2 | 24.6 | 754$\pm$36.1s (12.0%) | |

**表 3：RNNSearch 在 newstest2014 (Test14) 和 newstest2017 (Test17) 上的 SacreBLEU 分数。“Th”：Theano 版本的 Nematus，“Py”：内部基于 PyTorch 的 RNNSearch。**

我们还对 Transformer 进行了实验，它基于自注意力，避免了循环连接并允许更高程度的并行化。
尽管如此，层归一化仍然是该架构的重要组成部分。我们使用内部的 Tensorflow 版本 Transformer 实现，并采用 [31] 中的基础设置，所有模型训练 300K 步。我们将没有归一化的 Transformer 作为我们的基线，并将增强了 RMSNorm 的 Transformer 与配备了 LayerNorm 的 Transformer 进行比较。表 5 显示了结果，从中我们观察到归一化对于 Transformer 的重要性，如果没有归一化，训练将会失败。RMSNorm 实现了与 LayerNorm 相当的 BLEU 分数，并产生了 7%$\sim$9% 的加速。与 RNNSearch 相比，归一化的相对成本较低，因为 Transformer 中的顺序归一化操作明显更少。

| 模型 | Test14 | Test17 | 时间 |
| --- | --- | --- | --- |
| 基线 | - | - | 210$\pm$0.23s |
| LayerNorm | 26.6 | 27.7 | 248$\pm$1.31s |
| RMSNorm | 26.8 | 27.7 | 231$\pm$0.04s (6.9%) |
| $p$RMSNorm | 26.5 | 27.8 | 225$\pm$1.63s (9.3%) |

**表 4：Transformer 在 newstest2014 (Test14) 和 newstest2017 (Test17) 上的 SacreBLEU 分数。“时间”：每 1k 个训练步的时间（秒），使用 Tesla V100 测量。“-” 表示我们未能训练该模型，且 BLEU 分数为 0。**

| 模型 | | 1 | 2 | 3 | 4 | ALL |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 | M | -2.60 | -1.19 | -1.43 | -1.53 | -1.60 |
| S | 7.35 | 2.33 | 2.61 | 2.73 | 3.04 | |
| LayerNorm | M | -0.43 | -0.48 | -0.50 | -0.50 | -0.51 |
| S | 1.19 | 1.51 | 1.51 | 1.51 | 1.51 | |
| RMSNorm | M | -0.40 | -0.60 | -0.69 | -0.74 | -0.73 |
| S | 1.27 | 1.51 | 1.50 | 1.49 | 1.50 | |

**表 5：在 RNNSearch 模型中解码器部分 GRU 单元的隐藏层到隐藏层映射上估计的均值 (M) 和标准差 (S) 统计量。我们使用 newstest2013 数据集。ALL：在所有词元位置上平均的统计量。数字 1,2,3,4 表示针对特定词元位置估计的统计量。**

归一化对均值和标准差的影响 表 5 显示了 RNNSearch 模型中隐藏表示的均值和标准差在各个词元位置上的分布。正如 Ba 等人 2016 年所观察到的，基线中的均值和标准差是不稳定的。
由于它们的归一化特性，RMSNorm 和 LayerNorm 都稳定了标准差。尽管 RMSNorm 中的均值未被归一化，但在实践中它比基线的均值更稳定。这支持了我们的假设，即 RMSNorm 能够稳定循环激活，而无需显式地对均值进行归一化。

**图 4：当初始化中心为 0.2 时，LayerNorm 和 RMSNorm 在 newstest2013（开发集）上的 SacreBLEU 分数曲线。**

关于 RMSNorm 的鲁棒性 剩下的一个问题是，LayerNorm 中的重新居中操作（RMSNorm 放弃了该操作）是否使模型对任意权重/偏置初始化更具鲁棒性。我们在 Tensorflow 中使用 Nematus 对 RNNSearch 进行了一项实验，并将权重初始化的中心更改为 0.2。图 4 中的结果表明，在异常初始化的情况下，LayerNorm 变得非常不稳定，但 RMSNorm 更具鲁棒性（两者的表现均不如原始初始化）。我们迄今为止的经验证据表明，RMSNorm 与 LayerNorm 具有相似的鲁棒性，甚至更强。

### 6.2 CNN/Daily Mail 阅读理解

这项阅读理解任务是一个完形填空式的问答任务，要求模型回答关于段落的问题，答案是段落中的一个匿名实体 [9]。我们在 CNN 语料库上训练了 Hermann 等人 2015 年提出的双向注意力阅读器（attentive reader）模型。关于实验设置的更多细节在附录 A.2 中给出。我们将 RMSNorm 与 LayerNorm 和 BatchNorm 进行了比较。

**图 5：注意力阅读器模型在验证集上的错误率。**

| 模型 | 时间 |
| --- | --- |
| 基线 | 315$\pm$6.30s |
| BatchNorm-Everywhere | 348$\pm$10.5s |
| BatchNorm-LSTM | 345$\pm$11.2s |
| LayerNorm | 392$\pm$5.70s |
| RMSNorm | 333$\pm$5.20s (15.1%) |
| $p$RMSNorm | 330$\pm$5.50s (15.8%) |

**表 6：注意力阅读器模型每 0.1k 个训练步的时间（秒）。**

**(a) Recall@1**

**(b) Recall@5**

**(c) Recall@10**

**图 6：顺序嵌入（order-embedding）模型在验证集上的 Recall@K 值。**

| | 模型 | 标题检索 | | | | 图像检索 | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | R@1 | R@5 | R@10 | Mean r | R@1 | R@5 | R@10 | Mean r | |
| | Sym [32] | 45.4 | | 88.7 | 5.8 | 36.3 | | 85.8 | 9.0 |
| 现有 | OE + Baseline [32]† | 46.7 | | 88.9 | 5.7 | 37.9 | | 85.9 | 8.1 |
| 工作 | OE + Baseline [3]‡ | 46.6 | 79.3 | 89.1 | 5.2 | 37.8 | 73.6 | 85.7 | 7.9 |
| | OE + LayerNorm [3] | 48.5 | 80.6 | 89.8 | 5.1 | 38.9 | 74.3 | 86.3 | 7.6 |
| | OE + Baseline | 45.8 | 79.7 | 88.8 | 5.4 | 37.6 | 73.6 | 85.8 | 7.7 |
| 本 | OE + LayerNorm | 47.9 | 79.5 | 89.2 | 5.3 | 38.4 | 74.6 | 86.7 | 7.5 |
| 工作 | OE + RMSNorm | 48.7 | 79.7 | 89.5 | 5.3 | 39.0 | 74.8 | 86.3 | 7.5 |
| | OE + $p$RMSNorm | 46.8 | 79.8 | 90.3 | 5.2 | 39.0 | 74.5 | 86.3 | 7.4 |

**表 7：Microsoft COCO 的 5 个测试集上的平均 R@K 值。R@K：Recall @ K，越高越好。Mean r：平均排名，越低越好。粗体数字突出显示了最佳结果。‡ 表示 † 的复现结果。**

图 5 和表 6 展示了结果。在通过为序列中每个时间步使用单独统计量的 BatchNorm 对 RNN 进行归一化后，BatchNorm-LSTM 和 BatchNorm-Everywhere 都有助于加快训练过程的收敛速度。相比之下，LayerNorm 和 RMSNorm 不仅比 BatchNorm 收敛得更快，而且达到了更低的验证错误率，尽管 $p$RMSNorm 的表现略差于 RMSNorm。虽然在图 5 中 RMSNorm 和 LayerNorm 的性能相当，但如表 6 所示，RMSNorm 比 LayerNorm 快约 15%。（脚注：请注意，BatchNorm 的实现是基于 cuDNN 的，因此表 6 中 BatchNorm 的时间成本不能直接与其他方法进行比较。）

### 6.3 图像-标题检索

图像-标题检索是一项跨模态任务，旨在学习图像和句子的联合嵌入空间，它包含两个子任务：图像检索和标题检索。前者根据查询标题对一组图像进行排名，后者根据查询图像对一组标题进行排名。我们在 Microsoft COCO 数据集 [17] 上训练了 Vendrov 等人 2015 年提出的顺序嵌入模型（OE），使用的是他们在 Theano 中的公开源代码。
关于实验设置的模型细节在附录 A.3 中提供。我们将 RMSNorm 与两个模型进行比较：一个没有任何归一化（Baseline），另一个使用 LayerNorm。

| 模型 | 时间 |
| --- | --- |
| Baseline | 2.11$\pm$0.047s |
| LayerNorm | 12.02$\pm$0.191s |
| RMSNorm | 7.12$\pm$0.207s (40.8%) |
| $p$RMSNorm | 4.34$\pm$0.168s (63.9%) |

**表 8：顺序嵌入模型每 0.1k 个训练步的时间（秒）。**

图 6 显示了每 300 个训练步后验证集上的 R@K 曲线，表 7 列出了最终的测试结果。在所有这些指标中，如图 6 所示，RMSNorm 和 LayerNorm 在模型收敛方面始终优于 Baseline。我们观察到，在验证集上，RMSNorm 在召回率值方面略微超过 LayerNorm。对于表 7 所示的最终测试结果，RMSNorm 和 LayerNorm 都提高了模型性能，达到了更高的召回率值（除了 LayerNorm 在 R@5 上）和更低的平均排名，尽管 RMSNorm 展现出比 LayerNorm 更好的泛化能力。此外，表 8 中的结果表明，与 LayerNorm 相比，RMSNorm 将训练速度提高了 40%$\sim$64%，突显了 $p$RMSNorm 更好的效率。

**表 9：ConvPool-CNN-C 模型的训练错误率。**

| 模型 | 测试错误率 | 时间 |
| --- | --- | --- |
| Baseline | 08.96% | 21$\pm$0.0s |
| BatchNorm | 08.25% | 38$\pm$0.0s |
| WeightNorm | 08.28% | 23$\pm$0.0s |
| LayerNorm | 10.49% | 39$\pm$0.4s |
| RMSNorm | 08.83% | 31$\pm$0.5s (20.5%) |
| $p$RMSNorm | 10.37% | 30$\pm$0.4s (23.1%) |

**表 10：ConvPool-CNN-C 模型的测试错误率和每个训练轮次的时间（秒）。时间是使用 GeForce RTX 2080 Ti 测量的。**

### 6.4 CIFAR-10 分类

CIFAR-10 是一个监督图像分类任务，包含 10 个不同的类别。我们训练了 ConvPool-CNN-C 架构 [15] 的修改版本，并遵循与 Salimans 和 Kingma 2016 相同的实验协议。BatchNorm、LayerNorm 和 WeightNorm 被包含在内以进行比较。训练细节在附录 A.4 中给出。

图 10 和表 10 展示了结果。使用归一化技术增强的模型比 Baseline 收敛得更快，其中 BatchNorm 表现最好。与之前的观察结果 [3] 类似，我们还发现层归一化在图像处理方面的效果不如 BatchNorm 和 WeightNorm。虽然 LayerNorm 通过缩短模型收敛时间优于 Baseline，但它未能泛化到测试集，导致测试错误率恶化了 1.53%。相比之下，RMSNorm 表现出更好的泛化能力，超过 Baseline 0.013%，并且与 LayerNorm 相比节省了约 20.5% 的训练时间。$p$RMSNorm 进一步获得了 2.6% 的加速，尽管代价是牺牲了 1.54% 的测试准确率。

## 7 结论与未来工作

本文提出了 RMSNorm，这是一种新颖的归一化方法，它根据 RMS 对求和输入进行归一化。RMSNorm 保留了 LayerNorm 的重缩放不变性，但摒弃了对模型训练贡献较小的重中心化不变性。与 LayerNorm 相比，使用 RMSNorm 的模型计算开销更小。RMSNorm 可以作为 LayerNorm 的直接替代方案，轻松应用于不同的模型架构。在几个 NLP 任务上的实验表明，RMSNorm 在质量上与 LayerNorm 相当，但加快了运行速度。
实际的速度提升取决于框架、硬件、神经网络架构以及其他组件的相对计算成本，我们在不同的模型和实现中经验性地观察到了 7%$\sim$64% 的加速。
我们的效率提升来自于简化计算，因此我们期望它们与提高训练速度的其他方法（如低精度算术和 GPU 内核融合）是正交的。
我们还对 $p$RMSNorm 进行了实验，它在求和输入的一个子集上估计 RMS。虽然理论上更快，但我们并没有一致地观察到 $p$RMSNorm 在经验上的速度提升。我们将通过代码优化是否能提高性能留给未来的工作去研究。

在未来，我们希望对 RMSNorm 成功背后的原因进行更多分析。受最近 BatchNorm 中 $l1$-范数成功的启发，我们将探索 RMSNorm 的不同范数，并简化其他归一化技术，如 BatchNorm。

## 致谢

我们感谢审稿人提出的深刻意见，以及 Antonio Valerio Miceli Barone 在机器翻译权重归一化方面提供的支持。
本项目获得了欧盟 H2020-ICT-2018-2-825460 (ELITR) 拨款的资助。
Biao Zhang 还要感谢百度奖学金的支持。这项工作使用了由剑桥大学研究计算服务中心 (http://www.hpc.cam.ac.uk) 运营的剑桥 Tier-2 系统提供的资源，该系统由 EPSRC Tier-2 资本拨款 EP/P020259/1 资助。
