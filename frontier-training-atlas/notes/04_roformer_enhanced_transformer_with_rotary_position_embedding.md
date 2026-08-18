<!-- 来源：https://arxiv.org/pdf/2104.09864；翻译范围：摘要至参考文献/附录前 -->

# RoFormer：带有旋转位置嵌入的增强型Transformer

###### 摘要

位置编码最近在Transformer架构中显示出了其有效性。它为序列中不同位置元素之间的依赖关系建模提供了有价值的监督。在本文中，我们首先研究了将位置信息整合到基于Transformer的语言模型学习过程中的各种方法。然后，我们提出了一种名为旋转位置嵌入（Rotary Position Embedding, RoPE）的新方法，以有效地利用位置信息。具体而言，所提出的RoPE使用旋转矩阵对绝对位置进行编码，同时在自注意力公式中结合了显式的相对位置依赖。值得注意的是，RoPE具有一些有价值的特性，包括序列长度的灵活性、随着相对距离增加而衰减的词元间依赖性，以及为线性自注意力配备相对位置编码的能力。
最后，我们在各种长文本分类基准数据集上评估了带有旋转位置嵌入的增强型Transformer（也称为RoFormer）。我们的实验表明，它始终优于其他替代方案。此外，我们提供了理论分析来解释一些实验结果。RoFormer已经集成到Huggingface中：https://huggingface.co/docs/transformers/model_doc/roformer。

关键词 预训练语言模型 $\cdot$
位置信息编码 $\cdot$
预训练 $\cdot$
自然语言处理。

## 1 引言

单词的顺序对自然语言理解具有巨大价值。基于循环神经网络（RNNs）的模型通过沿时间维度递归计算隐藏状态来编码词元的顺序。基于卷积神经网络（CNNs）的模型（CNNs）Gehring et al. 2017 通常被认为是位置无关的，但最近的研究 Islam et al. 2020 表明，常用的填充（padding）操作可以隐式地学习位置信息。
最近，基于Transformer Vaswani et al. 2017 构建的预训练语言模型（PLMs）在各种自然语言处理（NLP）任务中取得了最先进的性能，包括上下文表示学习 Devlin et al. 2019、机器翻译 Vaswani et al. 2017 和语言建模 Radford et al. 2019 等等。与基于RNN和CNN的模型不同，PLM利用自注意力机制在语义上捕获给定语料库的上下文表示。因此，与RNN相比，PLM在并行化方面取得了显著的改进，并且与CNN相比，提高了更长词元间关系的建模能力（脚注：堆叠多个CNN层也可以捕获更长的词元间关系，这里我们仅考虑单层设置。）。

值得注意的是，当前PLM的自注意力架构已被证明是位置无关的 Yun et al. 2020。基于这一观点，人们提出了各种方法将位置信息编码到学习过程中。一方面，通过预定义函数生成的绝对位置编码 Vaswani et al. 2017 被添加到上下文表示中，同时也有可训练的绝对位置编码 Gehring et al. 2017; Devlin et al. 2019; Lan et al. 2020; Clark et al. 2020; Radford et al. 2019; Radford and Narasimhan 2018。另一方面，先前的工作 Parikh et al. 2016; Shaw et al. 2018; Huang et al. 2018; Dai et al. 2019; Yang et al. 2019; Raffel et al. 2020; Ke et al. 2020; He et al. 2020; Huang et al. 2020 侧重于相对位置编码，这通常将相对位置信息编码到注意力机制中。除了这些方法之外，Liu et al. 2020 的作者提出从神经ODE Chen et al. 2018a 的角度对位置编码的依赖关系进行建模，而 Wang et al. 2020 的作者提出在复数空间中对位置信息进行建模。尽管这些方法很有效，但它们通常将位置信息添加到上下文表示中，从而使它们不适用于线性自注意力架构。

在本文中，我们引入了一种新颖的方法，即旋转位置嵌入（Rotary Position Embedding, RoPE），以将位置信息利用到PLM的学习过程中。具体而言，RoPE使用旋转矩阵对绝对位置进行编码，同时在自注意力公式中结合了显式的相对位置依赖。请注意，所提出的RoPE通过其有价值的特性优于现有方法，这些特性包括序列长度的灵活性、随着相对距离增加而衰减的词元间依赖性，以及为线性自注意力配备相对位置编码的能力。
在各种长文本分类基准数据集上的实验结果表明，带有旋转位置嵌入的增强型Transformer（即RoFormer）与基线替代方案相比能够提供更好的性能，从而证明了所提出的RoPE的有效性。

简而言之，我们的贡献主要有以下三个方面：

- 我们研究了现有的相对位置编码方法，发现它们大多基于将位置编码添加到上下文表示中的分解思想构建。我们引入了一种新颖的方法，即旋转位置嵌入（Rotary Position Embedding, RoPE），以将位置信息利用到PLM的学习过程中。其核心思想是通过将上下文表示与具有清晰理论解释的旋转矩阵相乘来编码相对位置。
- 我们研究了RoPE的特性，并表明它随着相对距离的增加而衰减，这正是自然语言编码所期望的。我们在此指出，以前基于相对位置编码的方法与线性自注意力不兼容。
- 我们在各种长文本基准数据集上评估了所提出的RoFormer。我们的实验表明，与其替代方案相比，它始终取得更好的性能。一些使用预训练语言模型的实验可在GitHub上获取：https://github.com/ZhuiyiTechnology/roformer。

本文的其余部分组织如下。我们在第2节中建立了自注意力架构中位置编码问题的正式描述，并回顾了先前的工作。然后，我们在第3节中描述了旋转位置编码（RoPE）并研究了其特性。我们在第4节中报告了实验。最后，我们在第5节中对本文进行总结。

## 2 背景与相关工作

### 2.1 预备知识

设 $\mathbb{S}_{N}=\{w_{i}\}_{i=1}^{N}$ 为包含 $N$ 个输入词元的序列，其中 $w_{i}$ 是第 $i^{th}$ 个元素。$\mathbb{S}_{N}$ 对应的词嵌入表示为 $\mathbb{E}_{N}=\{{\boldsymbol{x}}_{i}\}_{i=1}^{N}$，其中 ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$ 是词元 $w_{i}$ 不带位置信息的d维词嵌入向量。自注意力首先将位置信息结合到词嵌入中，并将它们转换为查询（queries）、键（keys）和值（value）表示。

$$
\displaystyle{\boldsymbol{q}}_{m}
$$
$$
\displaystyle=f_{q}({\boldsymbol{x}}_{m},m)
$$
$$
\displaystyle{\boldsymbol{k}}_{n}
$$
$$
\displaystyle=f_{k}({\boldsymbol{x}}_{n},n)
$$
$$
\displaystyle{\boldsymbol{v}}_{n}
$$
$$
\displaystyle=f_{v}({\boldsymbol{x}}_{n},n),
$$
(1)

其中 ${\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}$ 和 ${\boldsymbol{v}}_{n}$ 分别通过 $f_{q},f_{k}$ 和 $f_{v}$ 结合了第 $m^{th}$ 和第 $n^{th}$ 个位置。然后使用查询和键的值来计算注意力权重，而输出则计算为值表示的加权和。

$$
\displaystyle a_{m,n}
$$
$$
\displaystyle=\frac{\exp(\frac{{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}}{\sqrt{d}})}{\sum_{j=1}^{N}\exp(\frac{{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{j}}{\sqrt{d}})}
$$
$$
\displaystyle\mathbf{o}_{m}
$$
$$
\displaystyle=\sum_{n=1}^{N}a_{m,n}{\boldsymbol{v}}_{n}
$$
(2)

现有的基于Transformer的位置编码方法主要集中在选择合适的函数来构成等式1。

### 2.2 绝对位置嵌入

等式1的一个典型选择是

$$
f_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i},i):={\boldsymbol{W}}_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i}+{\boldsymbol{p}}_{i}),
$$
(3)

其中 ${\boldsymbol{p}}_{i}\in\mathbb{R}^{d}$ 是一个依赖于词元 ${\boldsymbol{x}}_{i}$ 位置的d维向量。先前的工作 Devlin et al. 2019; Lan et al. 2020; Clark et al. 2020; Radford et al. 2019; Radford and Narasimhan 2018 引入了一组可训练向量 ${\boldsymbol{p}}_{i}\in\{{\boldsymbol{p}}_{t}\}_{t=1}^{L}$ 的使用，其中 $L$ 是最大序列长度。Vaswani et al. 2017 的作者提出使用正弦函数来生成 ${\boldsymbol{p}}_{i}$。

$$
\begin{cases}{\boldsymbol{p}}_{i,2t}&=\sin(k/10000^{2t/d})\\
{\boldsymbol{p}}_{i,2t+1}&=\cos(k/10000^{2t/d})\end{cases}
$$
(4)

其中 ${\boldsymbol{p}}_{i,2t}$ 是d维向量 ${\boldsymbol{p}}_{i}$ 的第 $2t^{th}$ 个元素。在下一节中，我们将展示从正弦函数的角度来看，我们提出的RoPE与这种直觉相关。然而，RoPE并没有直接将位置添加到上下文表示中，而是提出通过与正弦函数相乘来结合相对位置信息。

### 2.3 相对位置嵌入

Shaw et al. 2018 的作者应用了等式1的不同设置，如下所示：

$$
\displaystyle f_{q}({\boldsymbol{x}}_{m}):={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m}
$$
$$
\displaystyle f_{k}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{k}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{k}_{r})
$$
$$
\displaystyle f_{v}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{v}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{v}_{r})
$$
(5)

其中 $\tilde{{\boldsymbol{p}}}^{k}_{r},\tilde{{\boldsymbol{p}}}^{v}_{r}\in\mathbb{R}^{d}$ 是可训练的相对位置嵌入。请注意，$r=\operatorname{clip}(m-n,r_{\text{min}},r_{\text{max}})$ 表示位置 $m$ 和 $n$ 之间的相对距离。他们对相对距离进行了截断，其假设是精确的相对位置信息在超过一定距离后就不再有用。
保持等式3的形式，作者 Dai et al. 2019 提出将等式2的 ${\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}$ 分解为

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n}+{\boldsymbol{p}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n},
$$
(6)

其核心思想是将绝对位置嵌入 ${\boldsymbol{p}}_{n}$ 替换为其正弦编码的相对对应项 $\tilde{{\boldsymbol{p}}}_{m-n}$，同时将第三项和第四项中的绝对位置 ${\boldsymbol{p}}_{m}$ 替换为两个独立于查询位置的可训练向量 $\mathbf{u}$ 和 $\mathbf{v}$。此外，${\boldsymbol{W}}_{k}$ 被区分为基于内容和基于位置的键向量 ${\boldsymbol{x}}_{n}$ 和 ${\boldsymbol{p}}_{n}$，分别表示为 ${\boldsymbol{W}}_{k}$ 和 $\widetilde{{\boldsymbol{W}}}_{k}$，从而得到：

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\mathbf{u}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+\mathbf{v}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}
$$
(7)

值得注意的是，通过设置 $f_{v}({\boldsymbol{x}}_{j}):={\boldsymbol{W}}_{v}{\boldsymbol{x}}_{j}$ 移除了值项中的位置信息。后来的工作 Raffel et al. 2020; He et al. 2020; Ke et al. 2020; Huang et al. 2020 遵循了这些设置，仅将相对位置信息编码到注意力权重中。然而，Raffel et al. 2020 的作者将等式6重构为：

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+b_{i,j}
$$
(8)

其中 $b_{i,j}$ 是一个可训练的偏置。Ke et al. 2020 的作者研究了等式6的中间两项，发现绝对位置和单词之间的相关性很小。Raffel et al. 2020 的作者提出使用不同的投影矩阵来对一对单词或位置进行建模。

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^{\intercal}\mathbf{U}_{q}^{\intercal}\mathbf{U}_{k}{\boldsymbol{p}}_{n}+b_{i,j}
$$
(9)

He et al. 2020 的作者认为，两个词元的相对位置只能使用等式6的中间两项来完全建模。因此，绝对位置嵌入 ${\boldsymbol{p}}_{m}$ 和 ${\boldsymbol{p}}_{n}$ 被简单地替换为相对位置嵌入 $\tilde{{\boldsymbol{p}}}_{m-n}$：

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\tilde{{\boldsymbol{p}}}_{m-n}^{\intercal}{\boldsymbol{W}}_{q}^{\intercal}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$
(10)

对相对位置嵌入的四种变体 Radford and Narasimhan 2018 的比较表明，类似于等式10的变体是其他三种中最有效的。一般来说，所有这些方法都试图在等式2的自注意力设置下，基于等式3的分解来修改等式6，这最初是在 Vaswani et al. 2017 中提出的。它们通常引入将位置信息直接添加到上下文表示中。与此不同，我们的方法旨在在某些约束条件下从等式1推导出相对位置编码。接下来，我们将展示通过将相对位置信息与上下文表示的旋转相结合，推导出的方法更具可解释性。

## 3 提出的方法

在本节中，我们讨论提出的旋转位置嵌入（RoPE）。我们首先在第3.1节中对相对位置编码问题进行公式化，然后在第3.2节中推导RoPE，并在第3.3节中研究其特性。

### 3.1 形式化

基于Transformer的语言建模通常通过自注意力机制利用单个词元的位置信息。正如在等式2中可以观察到的，${\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}$ 通常能够实现不同位置词元之间的知识传递。为了结合相对位置信息，我们要求查询 ${\boldsymbol{q}}_{m}$ 和键 ${\boldsymbol{k}}_{n}$ 的内积由一个函数 $g$ 来表示，该函数仅将词嵌入 ${\boldsymbol{x}}_{m}$、${\boldsymbol{x}}_{n}$ 及其相对位置 $m-n$ 作为输入变量。换句话说，我们希望内积仅以相对形式编码位置信息：

$$
\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n).
$$
(11)

最终目标是找到一种等效的编码机制来求解函数 $f_{q}({\boldsymbol{x}}_{m},m)$ 和 $f_{k}({\boldsymbol{x}}_{n},n)$，以符合上述关系。

### 3.2 旋转位置嵌入

#### 3.2.1 2D情况

我们从维度 $d=2$ 的简单情况开始。在这些设置下，我们利用2D平面上向量的几何性质及其复数形式来证明（更多细节请参见第3.4.1节），我们的形式化等式11的一个解为：

$$
\displaystyle f_{q}({\boldsymbol{x}}_{m},m)
$$
$$
\displaystyle=({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{im\theta}
$$
$$
\displaystyle f_{k}({\boldsymbol{x}}_{n},n)
$$
$$
\displaystyle=({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{in\theta}
$$
$$
\displaystyle g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n)
$$
$$
\displaystyle=\operatorname{Re}[({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}e^{i(m-n)\theta}]
$$
(12)

其中 $\operatorname{Re}[\cdot]$ 是复数的实部，$({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}$ 表示 $({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})$ 的共轭复数。$\theta\in\mathbb{R}$ 是一个预设的非零常数。我们可以进一步将 $f_{\{q,k\}}$ 写成乘法矩阵的形式：

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)=\left(\begin{array}[]{cc}\cos{m\theta}&-\sin{m\theta}\\
\sin{m\theta}&\cos{m\theta}\end{array}\right)\left(\begin{array}[]{cc}W^{(11)}_{\{q,k\}}&W^{(12)}_{\{q,k\}}\\
W^{(21)}_{\{q,k\}}&W^{(22)}_{\{q,k\}}\end{array}\right)\left(\begin{array}[]{cc}x^{(1)}_{m}\\
x^{(2)}_{m}\end{array}\right)
$$
(13)

其中 $(x^{(1)}_{m},x^{(2)}_{m})$ 是在2D坐标中表示的 ${\boldsymbol{x}}_{m}$。类似地，$g$ 可以被视为一个矩阵，从而能够在2D情况下求解第3.1节中的形式化问题。具体而言，结合相对位置嵌入非常直观：只需将仿射变换后的词嵌入向量旋转其位置索引倍数的角度，从而解释了旋转位置嵌入背后的直觉。

#### 3.2.2 一般形式

为了将我们在2D中的结果推广到任何 $d$ 为偶数的 ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$，我们将d维空间划分为 $d/2$ 个子空间，并利用内积的线性性质将它们组合起来，将 $f_{\{q,k\}}$ 转化为：

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)={\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{\{q,k\}}{\boldsymbol{x}}_{m}
$$
(14)

其中

$$
{\boldsymbol{R}}^{d}_{\Theta,m}=\begin{pmatrix}\cos{m\theta_{1}}&-\sin{m\theta_{1}}&0&0&\cdots&0&0\\
\sin{m\theta_{1}}&\cos{m\theta_{1}}&0&0&\cdots&0&0\\
0&0&\cos{m\theta_{2}}&-\sin{m\theta_{2}}&\cdots&0&0\\
0&0&\sin{m\theta_{2}}&\cos{m\theta_{2}}&\cdots&0&0\\
\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&0&\cdots&\cos{m\theta_{d/2}}&-\sin{m\theta_{d/2}}\\
0&0&0&0&\cdots&\sin{m\theta_{d/2}}&\cos{m\theta_{d/2}}\end{pmatrix}
$$
(15)

是具有预定义参数 $\Theta=\{\theta_{i}=10000^{-2(i-1)/d},i\in[1,2,...,d/2]\}$ 的旋转矩阵。RoPE 的图解如图 1 所示。将我们的 RoPE 应用于等式 2 中的自注意力机制，我们得到：

$$
{\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}=({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^{\intercal}({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})={\boldsymbol{x}}^{\intercal}{\boldsymbol{W}}_{q}R^{d}_{\Theta,n-m}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$
(16)

其中 ${\boldsymbol{R}}^{d}_{\Theta,n-m}=({\boldsymbol{R}}^{d}_{\Theta,m})^{\intercal}{\boldsymbol{R}}^{d}_{\Theta,n}$。请注意，${\boldsymbol{R}}^{d}_{\Theta}$ 是一个正交矩阵，这确保了在编码位置信息过程中的稳定性。此外，由于 $R^{d}_{\Theta}$ 的稀疏性，像等式 16 那样直接应用矩阵乘法在计算上并不高效；我们在理论解释中提供了另一种实现方式。

与先前工作中采用的位置嵌入方法的加性性质（即等式 3、4、5、6、7、8、9 和 10）相反，我们的方法是乘性的。此外，当应用于自注意力机制时，RoPE 通过旋转矩阵乘积自然地结合了相对位置信息，而不是改变加性位置编码展开公式中的项。

**图 1：旋转位置嵌入（RoPE）的实现。**

### 3.3 RoPE 的性质

##### 长期衰减：

遵循 Vaswani et al. 2017，我们设置 $\theta_{i}=10000^{-2i/d}$。可以证明，这种设置提供了一种长期衰减性质（更多细节请参考第 3.4.3 节），这意味着当相对位置增加时，内积将会衰减。这一性质符合直觉，即具有较长相对距离的一对 token 应该具有较少的联系。

##### 带有线性注意力的 RoPE：

自注意力机制可以重写为更一般的形式。

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})}.
$$
(17)

原始的自注意力机制选择 $\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})=\exp({\boldsymbol{q}}_{m}^{\intercal}{\boldsymbol{k}}_{n}/\sqrt{d})$。请注意，原始的自注意力机制需要计算每一对 token 的 query 和 key 的内积，这具有二次复杂度 $\mathbb{O}(N^{2})$。遵循 Katharopoulos et al. 2020，线性注意力机制将等式 17 重新表述为

$$
\operatorname{Attention}({\boldsymbol{Q}},{\boldsymbol{K}},{\boldsymbol{V}})_{m}=\frac{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^{\intercal}\varphi({\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^{\intercal}\varphi({\boldsymbol{k}}_{n})},
$$
(18)

其中 $\phi(\cdot),\varphi(\cdot)$ 通常是非负函数。Katharopoulos et al. 2020 的作者提出了 $\phi(x)=\varphi(x)=\operatorname{elu}(x)+1$，并首先利用矩阵乘法的结合律计算了 key 和 value 之间的乘法。Shen et al. 2021 中使用 softmax 函数在内积之前分别对 query 和 key 进行归一化，这等价于 $\phi({\boldsymbol{q}}_{i})=\operatorname{softmax}({\boldsymbol{q}}_{i})$ 和 $\phi({\boldsymbol{k}}_{j})=\exp({\boldsymbol{k}}_{j})$。有关线性注意力的更多细节，我们鼓励读者参考原始论文。在本节中，我们重点讨论将 RoPE 与等式 18 结合。由于 RoPE 通过旋转注入位置信息，这保持了隐藏表示的范数不变，我们可以通过将旋转矩阵与非负函数的输出相乘，从而将 RoPE 与线性注意力结合起来。

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\big({\boldsymbol{R}}^{d}_{\Theta,m}\phi({\boldsymbol{q}}_{m})\big)^{\intercal}\big({\boldsymbol{R}}^{d}_{\Theta,n}\varphi({\boldsymbol{k}}_{n})\big){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^{\intercal}\varphi({\boldsymbol{k}}_{n})}.
$$
(19)

值得注意的是，我们保持分母不变以避免除以零的风险，并且分子中的求和可能包含负项。尽管等式 19 中每个 value ${\boldsymbol{v}}_{i}$ 的权重没有经过严格的概率归一化，但我们认为该计算仍然可以对 value 的重要性进行建模。

### 3.4 理论解释

#### 3.4.1 2D 下 RoPE 的推导

在 $d=2$ 的情况下，我们考虑两个词嵌入向量 ${\boldsymbol{x}}_{q}$，${\boldsymbol{x}}_{k}$ 分别对应于 query 和 key 及其位置 $m$ 和 $n$。根据等式 1，它们经过位置编码后的对应项为：

$$
\displaystyle{\boldsymbol{q}}_{m}
$$
$$
\displaystyle=f_{q}({\boldsymbol{x}}_{q},m),
$$
$$
\displaystyle{\boldsymbol{k}}_{n}
$$
$$
\displaystyle=f_{k}({\boldsymbol{x}}_{k},n),
$$
(20)

其中 ${\boldsymbol{q}}_{m}$ 和 ${\boldsymbol{k}}_{n}$ 的下标表示编码的位置信息。假设存在一个函数 $g$，它定义了由 $f_{\{q,k\}}$ 产生的向量之间的内积：

$$
{\boldsymbol{q}}^{\intercal}_{m}{\boldsymbol{k}}_{n}=\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},n-m),
$$
(21)

我们进一步要求满足以下初始条件：

$$
\displaystyle{\boldsymbol{q}}
$$
$$
\displaystyle=f_{q}({\boldsymbol{x}}_{q},0),
$$
$$
\displaystyle{\boldsymbol{k}}
$$
$$
\displaystyle=f_{k}({\boldsymbol{x}}_{k},0),
$$
(22)

这可以被理解为编码了空位置信息的向量。给定这些设置，我们试图找到 $f_{q}$，$f_{k}$ 的解。首先，我们利用 2D 中向量的几何意义及其复数对应形式，将等式 20 和 21 中的函数分解为：

$$
\displaystyle f_{q}({\boldsymbol{x}}_{q},m)
$$
$$
\displaystyle=R_{q}({\boldsymbol{x}}_{q},m)e^{i\Theta_{q}({\boldsymbol{x}}_{q},m)},
$$
$$
\displaystyle f_{k}({\boldsymbol{x}}_{k},n)
$$
$$
\displaystyle=R_{k}({\boldsymbol{x}}_{k},n)e^{i\Theta_{k}({\boldsymbol{x}}_{k},n)},
$$
$$
\displaystyle g({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)
$$
$$
\displaystyle=R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)e^{i\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)},
$$
(23)

其中 $R_{f}$、$R_{g}$ 和 $\Theta_{f}$、$\Theta_{g}$ 分别是 $f_{\{q,k\}}$ 和 $g$ 的径向和角向分量。将它们代入等式 21，我们得到关系式：

$$
\displaystyle R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},n)
$$
$$
\displaystyle=R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m),
$$
$$
\displaystyle\Theta_{k}({\boldsymbol{x}}_{k},n)-\Theta_{q}({\boldsymbol{x}}_{q},m)
$$
$$
\displaystyle=\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m),
$$
(24)

相应的初始条件为：

$$
\displaystyle{\boldsymbol{q}}
$$
$$
\displaystyle=\|{\boldsymbol{q}}\|e^{i\theta_{q}}=R_{q}({\boldsymbol{x}}_{q},0)e^{i\Theta_{q}({\boldsymbol{x}}_{q},0)},
$$
$$
\displaystyle{\boldsymbol{k}}
$$
$$
\displaystyle=\|{\boldsymbol{k}}\|e^{i\theta_{k}}=R_{k}({\boldsymbol{x}}_{k},0)e^{i\Theta_{k}({\boldsymbol{x}}_{k},0)},
$$
(25)

其中 $\|{\boldsymbol{q}}\|$、$\|{\boldsymbol{k}}\|$ 和 $\theta_{q}$、$\theta_{k}$ 是 ${\boldsymbol{q}}$ 和 ${\boldsymbol{k}}$ 在 2D 平面上的径向和角向部分。

接下来，我们在等式 24 中设置 $m=n$，并考虑等式 25 中的初始条件：

$$
\displaystyle R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},m)
$$
$$
\displaystyle=R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=R_{q}({\boldsymbol{x}}_{q},0)R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|,
$$
$$
\displaystyle\Theta_{k}({\boldsymbol{x}}_{k},m)-\Theta_{q}({\boldsymbol{x}}_{q},m)
$$
$$
\displaystyle=\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\Theta_{k}({\boldsymbol{x}}_{k},0)-\Theta_{q}({\boldsymbol{x}}_{q},0)=\theta_{k}-\theta_{q}.
$$
(26a)

一方面，可以从等式 26a 得到 $R_{f}$ 的一个直接解：

$$
\displaystyle R_{q}({\boldsymbol{x}}_{q},m)
$$
$$
\displaystyle=R_{q}({\boldsymbol{x}}_{q},0)=\|{\boldsymbol{q}}\|
$$
$$
\displaystyle R_{k}({\boldsymbol{x}}_{k},n)
$$
$$
\displaystyle=R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{k}}\|
$$
$$
\displaystyle R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)
$$
$$
\displaystyle=R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|
$$
(27)

这解释了径向函数 $R_{q}$、$R_{k}$ 和 $R_{g}$ 独立于位置信息。另一方面，正如在等式 26b 中可以注意到的，$\Theta_{q}({\boldsymbol{x}}_{q},m)-\theta_{q}=\Theta_{k}({\boldsymbol{x}}_{k},m)-\theta_{k}$ 表明角向函数不依赖于 query 和 key，我们将它们设置为 $\Theta_{f}:=\Theta_{q}=\Theta_{k}$，并且项 $\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)-\theta_{\{q,k\}}$ 是位置 $m$ 的函数且独立于词嵌入 ${\boldsymbol{x}}_{\{q,k\}}$，我们将其记为 $\phi(m)$，从而得到：

$$
\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)=\phi(m)+\theta_{\{q,k\}},
$$
(28)

进一步，通过将 $n=m+1$ 代入等式 24 并考虑上述等式，我们可以得到：

$$
\phi(m+1)-\phi(m)=\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},1)+\theta_{q}-\theta_{k},
$$
(29)

由于等式右边（RHS）是一个与 $m$ 无关的常数，具有连续整数输入的 $\phi(m)$ 产生一个等差数列：

$$
\phi(m)=m\theta+\gamma,
$$
(30)

其中 $\theta,\gamma\in\mathbb{R}$ 是常数且 $\theta$ 非零。总结我们从等式 27、28、29 和 30 得到的解：

$$
\displaystyle f_{q}({\boldsymbol{x}}_{q},m)
$$
$$
\displaystyle=\|{\boldsymbol{q}}\|e^{i\theta_{q}+m\theta+\gamma}={\boldsymbol{q}}e^{i(m\theta+\gamma)},
$$
$$
\displaystyle f_{k}({\boldsymbol{x}}_{k},n)
$$
$$
\displaystyle=\|{\boldsymbol{k}}\|e^{i\theta_{k}+n\theta+\gamma}={\boldsymbol{k}}e^{i(n\theta+\gamma)}.
$$
(31)

请注意，我们没有对等式 22 的 $f_{q}$ 和 $f_{k}$ 施加任何约束，因此 $f_{q}({\boldsymbol{x}}_{m},0)$ 和 $f_{k}({\boldsymbol{x}}_{n},0)$ 可以自由选择。为了使我们的结果与等式 3 具有可比性，我们定义：

$$
\displaystyle{\boldsymbol{q}}=f_{q}({\boldsymbol{x}}_{m},0)
$$
$$
\displaystyle={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{n},
$$
$$
\displaystyle{\boldsymbol{k}}=f_{k}({\boldsymbol{x}}_{n},0)
$$
$$
\displaystyle={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}.
$$
(32)

然后，我们只需在最终解的等式 31 中设置 $\gamma=0$：

$$
\displaystyle f_{q}({\boldsymbol{x}}_{m},m)
$$
$$
\displaystyle=({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{im\theta},
$$
$$
\displaystyle f_{k}({\boldsymbol{x}}_{n},n)
$$
$$
\displaystyle=({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{in\theta}.
$$
(33)

#### 3.4.2 旋转矩阵乘法的计算高效实现

利用等式 15 中 ${\boldsymbol{R}}^{d}_{\Theta,m}$ 的稀疏性，$R^{d}_{\Theta}$ 和 ${\boldsymbol{x}}\in\mathbb{R}^{d}$ 乘法的一种计算上更高效的实现是：

$$
{\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{x}}=\begin{pmatrix}x_{1}\\
x_{2}\\
x_{3}\\
x_{4}\\
\vdots\\
x_{d-1}\\
x_{d}\end{pmatrix}\otimes\begin{pmatrix}\cos{m\theta_{1}}\\
\cos{m\theta_{1}}\\
\cos{m\theta_{2}}\\
\cos{m\theta_{2}}\\
\vdots\\
\cos{m\theta_{d/2}}\\
\cos{m\theta_{d/2}}\end{pmatrix}+\begin{pmatrix}-x_{2}\\
x_{1}\\
-x_{4}\\
x_{3}\\
\vdots\\
-x_{d}\\
x_{d-1}\end{pmatrix}\otimes\begin{pmatrix}\sin{m\theta_{1}}\\
\sin{m\theta_{1}}\\
\sin{m\theta_{2}}\\
\sin{m\theta_{2}}\\
\vdots\\
\sin{m\theta_{d/2}}\\
\sin{m\theta_{d/2}}\end{pmatrix}
$$
(34)

#### 3.4.3 RoPE 的长期衰减

我们可以将向量 ${\boldsymbol{q}}={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m}$ 和 ${\boldsymbol{k}}={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}$ 的元素成对分组，并且等式 16 中 RoPE 的内积可以写成复数乘法。

$$
({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^{\intercal}({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})=\operatorname{Re}\bigg[\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg]
$$
(35)

其中 ${\boldsymbol{q}}_{[2i:2i+1]}$ 表示 ${\boldsymbol{q}}$ 的第 $2i^{th}$ 到 $(2i+1)^{th}$ 个元素。记 $h_{i}={\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}$ 和 $S_{j}=\sum_{i=0}^{j-1}e^{i(m-n)\theta_{i}}$，并令 $h_{d/2}=0$ 和 $S_{0}=0$，我们可以使用阿贝尔变换（Abel transformation）重写该求和

$$
\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}=\sum_{i=0}^{d/2-1}h_{i}(S_{i+1}-S_{i})=-\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i}).
$$
(36)

因此，

$$
\displaystyle\bigg|\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg|
$$
$$
\displaystyle=\bigg|\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i})\bigg|
$$
$$
\displaystyle\leq\sum_{i=0}^{d/2-1}|S_{i+1}||(h_{i+1}-{h_{i}})|
$$
$$
\displaystyle\leq\big(\max_{i}|h_{i+1}-h_{i}|\big)\sum_{i=0}^{d/2-1}|S_{i+1}|
$$
(37)

请注意，通过设置 $\theta_{i}=10000^{-2i/d}$，$\frac{1}{d/2}\sum_{i=1}^{d/2}|S_{i}|$ 的值随着相对距离 $m-n$ 的增加而衰减，如图 2 所示。

**图 2：RoPE 的长期衰减。**

## 4 实验与评估

我们在各种 NLP 任务上评估所提出的 RoFormer，如下所示。我们在第 4.1 节的机器翻译任务上验证了所提出解决方案的性能。然后，在第 4.2 节中，我们在预训练阶段将我们的 RoPE 实现与 BERTDevlin et al. 2019 进行了比较。基于预训练模型，在第 4.3 节中，我们进一步在 GLUE 基准Singh et al. 2018 的不同下游任务中进行了评估。此外，在第 4.4 节中，我们使用所提出的 RoPE 与 PerFormer Choromanski et al. 2020 的线性注意力进行了实验。最后，在第 4.5 节中包含了对中文数据的额外测试。所有实验均在两台配备 4 x V100 GPU 的云服务器上运行。

### 4.1 机器翻译

我们首先展示 RoFormer 在序列到序列语言翻译任务上的性能。

#### 4.1.1 实验设置

我们选择标准的 WMT 2014 英德数据集Bojar et al. 2014，该数据集包含约 450 万个句子对。我们与基于 Transformer 的基线替代方案 Vaswani et al. 2017 进行了比较。

#### 4.1.2 实现细节

我们对基线模型 Vaswani et al. 2017 的自注意力层进行了一些修改，以使 RoPE 能够参与其学习过程。我们复制了英德翻译的设置，基于联合源和目标字节对编码（BPE）Sennrich et al. 2015，词表大小为 37k。在评估期间，通过平均最后 5 个检查点获得单个模型。结果使用束搜索（beam search），束大小（beam size）为 4，长度惩罚为 0.6。我们在 fairseq 工具包（MIT 许可证）Ott et al. 2019 中使用 PyTorch 实现该实验。我们的模型使用 Adam 优化器进行优化，其中 $\beta_{1}=0.9$，$\beta_{2}=0.98$，学习率从 $1e-7$ 线性增加到 $5e-4$，然后按步数的平方根倒数成比例衰减。还采用了 0.1 的标签平滑（label smoothing）。我们报告测试集上的 BLEUPapineni et al. 2002 分数作为最终指标。

#### 4.1.3 结果

我们在相同的设置下训练基线模型和我们的 RoFormer，并在表 1 中报告结果。可以看出，与基线 Transformer 相比，我们的模型给出了更好的 BLEU 分数。

**表 1：在 WMT 2014 英德翻译任务Bojar et al. 2014 上，与基线替代方案 Vaswani et al. 2017 相比，所提出的 RoFormer 给出了更好的 BLEU 分数。**

| 模型 | BLEU |
| --- | --- |
| Transformer-baseVaswani et al. 2017 | 27.3 |
| RoFormer | 27.5 |

### 4.2 预训练语言建模

第二个实验旨在验证我们提出的方法在学习上下文表示方面的性能。为了实现这一点，我们在预训练阶段将 BERT 原始的正弦位置编码替换为我们的 RoPE。

#### 4.2.1 实验设置

我们使用来自 Huggingface Datasets 库（Apache 许可证 2.0）的 BookCorpus Zhu et al. 2015 和 Wikipedia Corpus Foundation 2021 进行预训练。语料库进一步以 8:2 的比例划分为训练集和验证集。我们使用训练过程中的掩码语言建模（MLM）损失值作为评估指标。
著名的 BERT Devlin et al. 2019 被采用作为我们的基线模型。请注意，我们在实验中使用 bert-base-uncased。

#### 4.2.2 实现细节

对于 RoFormer，我们将基线模型自注意力块中的正弦位置编码替换为我们提出的 RoPE，并根据等式 16 实现自注意力。我们以 64 的批大小（batch size）和 512 的最大序列长度训练 BERT 和 RoFormer 100k 步。使用 AdamW Loshchilov and Hutter 2017 作为优化器，学习率为 1e-5。

#### 4.2.3 结果

预训练期间的 MLM 损失如图 3 左图所示。与原始的 BERT 相比，RoFormer 经历了更快的收敛。

**图 3：RoPE 在语言建模预训练中的评估。左图：BERT 和 RoFormer 的训练损失。右图：带有和不带有 RoPE 的 PerFormer 的训练损失。**

### 4.3 在 GLUE 任务上微调

与之前的实验一致，我们在各种 GLUE 任务上微调预训练 RoFormer 的权重，以评估其在下游 NLP 任务上的泛化能力。

#### 4.3.1 实验设置

我们考察了来自 GLUE 的几个数据集，即 MRPC Dolan and Brockett 2005、SST-2 Socher et al. 2013、QNLI Rajpurkar et al. 2016、STS-B Al-Natsheh 2017、QQP Chen et al. 2018b 和 MNLI Williams et al. 2018。我们使用 F1 分数作为 MRPC 和 QQP 数据集的评估指标，使用斯皮尔曼相关系数（spearman correlation）作为 STS-B 的评估指标，并使用准确率作为其余数据集的评估指标。

#### 4.3.2 实现细节

我们使用 Huggingface Transformers 库（Apache 许可证 2.0）Wolf et al. 2020 对上述每个下游任务进行 3 个 epoch 的微调，最大序列长度为 512，批大小为 32，学习率为 2,3,4,5e-5。遵循 Devlin et al. 2019，我们报告验证集上的最佳平均结果。

**表2：通过在下游GLEU任务上微调来比较RoFormer和BERT。**

| 模型 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI(m/mm) |
| --- | --- | --- | --- | --- | --- | --- |
| BERTDevlin et al. 2019 | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6/83.4 |
| RoFormer | 89.5 | 90.7 | 88.0 | 87.0 | 86.4 | 80.2/79.8 |

#### 4.3.3 结果

微调任务的评估结果报告在表2中。可以看出，RoFormer在六个数据集中的三个上显著优于BERT，并且提升相当可观。

### 4.4 带有RoPE的Performer

Performer Choromanski et al. 2020 引入了一种替代的注意力机制，即线性注意力，旨在避免随输入序列长度呈二次方增长的计算成本。如第3.3节所述，所提出的RoPE可以轻松地在PerFormer模型中实现，以实现相对位置编码，同时保持其在自注意力中线性扩展的复杂度。我们通过语言建模的预训练任务展示了其性能。

#### 4.4.1 实现细节

我们在Enwik8数据集 Mahoney 2006 上进行了测试，该数据集来自英文维基百科，除了英文文本外，还包含标记、特殊字符和其他语言的文本。我们将RoPE整合到具有768个维度和12个头部的12层基于字符的PerFormer中（脚注：对于本实验，我们采用了来自 https://github.com/lucidrains/performer-pytorch 的代码（MIT许可证））。
为了更好地说明RoPE的功效，我们报告了在相同设置下（即学习率1e-4、批量大小128和固定的最大序列长度1024等）使用和不使用RoPE的预训练过程的损失曲线。

#### 4.4.2 结果

如图3右图所示，将RoPE替换到Performer中，在相同训练步数下能够实现快速收敛和更低的损失。这些改进加上线性复杂度，使得Performer更具吸引力。

### 4.5 在中文数据上的评估

除了在英文数据上的实验外，我们还展示了在中文数据上的额外结果。为了验证RoFormer在长文本上的性能，我们在长度超过512个字符的长文档上进行了实验。

#### 4.5.1 实现

在这些实验中，我们对WoBERT Su 2020 进行了一些修改，将其绝对位置嵌入替换为我们提出的RoPE。作为与其他基于Transformer的中文预训练模型（即BERT Devlin et al. 2019、WoBERT Su 2020 和 NEZHA Wei et al. 2019）的交叉比较，我们在表3中列出了它们的分词级别和位置嵌入信息。

**表3：我们的RoFormer与其他中文数据预训练模型之间的交叉比较。'abs'和'rel'分别表示绝对位置嵌入和相对位置嵌入。**

| 模型 | BERTDevlin et al. 2019 | WoBERTSu 2020 | NEZHAWei et al. 2019 | RoFormer |
| --- | --- | --- | --- | --- |
| 分词级别 | char | word | char | word |
| 位置嵌入 | abs. | abs. | rel. | RoPE |

#### 4.5.2 预训练

我们在从中文维基百科、新闻和论坛收集的约34GB数据上对RoFormer进行预训练。预训练分多个阶段进行，通过改变批量大小和最大输入序列长度，以使模型适应各种场景。如表4所示，RoFormer的准确率随着序列长度上限的增加而提高，这证明了RoFormer处理长文本的能力。我们认为这归功于所提出的RoPE出色的泛化能力。

**表4：RoFormer在中文数据集上的预训练策略。训练过程分为多个连续的阶段。在每个阶段中，我们使用最大序列长度和批量大小的特定组合来训练模型。**

| 阶段 | 最大序列长度 | 批量大小 | 训练步数 | 损失 | 准确率 |
| --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

#### 4.5.3 下游任务与数据集

我们选择2019年中国法研杯相似案例匹配（CAIL2019-SCM）Xiao et al. 2019 数据集来说明RoFormer处理长文本（即语义文本匹配）的能力。CAIL2019-SCM包含由中国最高人民法院公布的8964个案例三元组。输入的三元组记为（A、B和C），是三个案例的事实描述。任务是预测在预定义的相似度度量下，（A, B）对是否比（A, C）更接近。请注意，由于文档的长度（即大多数超过512个字符），现有方法大多无法在CAIL2019-SCM数据集上取得显著表现。我们基于众所周知的6:2:2比例划分了训练集、验证集和测试集。

#### 4.5.4 结果

我们将预训练的RoFormer模型应用于具有不同输入长度的CAIL2019-SCM。如表5所示，该模型与在相同预训练数据上预训练的BERT和WoBERT模型进行了比较。在短文本截断（即512）的情况下，RoFormer的结果与WoBERT相当，并且略好于BERT的实现。然而，当将最大输入文本长度增加到1024时，RoFormer以1.5%的绝对提升优于WoBERT。

**表5：CAIL2019-SCM任务的实验结果。第一列中的数字表示最大截断序列长度。结果以百分比准确率的形式呈现。**

| 模型 | 验证集 | 测试集 |
| --- | --- | --- |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | 66.07% | 69.79% |

#### 4.5.5 本工作的局限性

尽管我们提供了理论基础以及有前景的实验证明，但我们的方法受到以下事实的限制：

- 尽管我们在数学上将相对位置关系格式化为2D子空间下的旋转，但缺乏关于为什么它比结合了其他位置编码策略的基线模型收敛更快的透彻解释。
- 尽管我们已经证明了我们的模型对于词元间乘积具有良好的长期衰减特性（第3.3节），这与现有的位置编码机制相似，并且我们的模型在长文本上表现出优于同类模型的性能，但我们尚未提出一个确切的解释。

我们提出的RoFormer建立在基于Transformer的基础架构之上，这需要用于预训练目的的硬件资源。

## 5 结论

在这项工作中，我们提出了一种新的位置嵌入方法，该方法在自注意力中结合了显式的相对位置依赖，以增强Transformer架构的性能。我们的理论分析表明，相对位置可以自然地使用自注意力中的向量乘积来公式化，而绝对位置信息则通过旋转矩阵进行编码。此外，我们在数学上说明了所提出方法应用于Transformer时的优势特性。最后，在英文和中文基准数据集上的实验表明，我们的方法促进了预训练中更快的收敛。实验结果还表明，我们提出的RoFormer在长文本任务上能够取得更好的性能。
