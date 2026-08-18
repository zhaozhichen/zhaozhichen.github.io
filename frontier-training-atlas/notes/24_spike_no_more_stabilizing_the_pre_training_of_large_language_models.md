<!-- 来源：https://arxiv.org/pdf/2312.16903；翻译范围：摘要至参考文献/附录前 -->

# 告别尖峰：

稳定大型语言模型的预训练

###### 摘要

在大型语言模型的预训练过程中，经常会出现损失尖峰（loss spikes）。
这些尖峰会降低大型语言模型的性能，有时甚至会毁掉预训练。
由于预训练需要庞大的计算预算，我们应该避免此类尖峰。
基于损失尖峰是由梯度范数突然增长引起的这一假设，我们通过分析子层雅可比矩阵的谱范数，探索了保持梯度范数较小的因素。
我们的研究结果表明，稳定预训练过程需要两个条件：较小的子层（small sub-layers）和较大的捷径连接（large shortcut）。
我们进行了各种实验，以经验性地验证我们的理论分析。
实验结果表明，满足这些条件的方法能有效防止预训练期间出现损失尖峰。

## 1 引言

**图1：Transformer的训练损失值，其维度和层数与Narayanan等人（2021）中17亿参数的配置相同。在Vanilla（原生模型）中，训练初期出现了一些尖峰，并且其损失值在约13000步时发生了爆炸。**

大型语言模型（LLM）已成为各种应用的基础资产（Brown et al. 2020; Chowdhery et al. 2022; Touvron et al. 2023）。
增加（神经）语言模型中的参数数量和训练数据量通常会带来更好的LLM（Kaplan et al. 2020）。
因此，预训练需要庞大的预算，从而将预训练失败的风险降至最低成为首要关注的问题。

尽管Transformer（Vaswani et al. 2017）被广泛用作LLM的基础架构，但尚未实现对其全面的理论理解。
其中一个关键的未解问题是，基于Transformer的LLM中频繁出现预训练失败的原因，这是由于损失值尖峰（损失尖峰）可能导致灾难性发散（Chowdhery et al. 2022），如图1中的Vanilla所示。
虽然已经提出了几种经验策略来缓解这个问题（Chowdhery et al. 2022; Le Scao et al. 2022; Zeng et al. 2023），但这些方法缺乏理论依据，使得它们在其他情况下的泛化能力（例如不同大小的模型参数）变得不明确。

在本研究中，我们针对LLM预训练期间的损失尖峰问题提供了理论分析。
我们通过分析子层雅可比矩阵的谱范数，确定了基于Transformer的LLM梯度范数的上界。
如果上界很大，梯度可能会突然出现尖峰，我们假设这种现象导致了损失尖峰。
然后，我们指出在典型设置中（例如广泛使用的实现Megatron-LM（Shoeybi et al. 2020）），该上界很大，因此很可能发生损失尖峰。
此外，为了使上界足够小，我们引入了两个条件：（1）用较小的值初始化子层的参数；（2）使每个嵌入的标准差接近$1$。
前者可以通过广泛用于LLM的初始化方法来满足（Shoeybi et al. 2020; Le Scao et al. 2022; Biderman et al. 2023）。
另一方面，后者在原始Transformer中通过缩放嵌入（scaling embeddings）得到了满足（Vaswani et al. 2017），但最近的实现中缺少了这种缩放。
综上所述，通过理论分析，我们在稳定LLM预训练方面重新评估了几种先前的技术。

在我们的理论分析基础上，我们通过一系列实证实验进一步证实了我们的主张，这些实验在不同的训练场景下明确区分了有效和无效的方法。
我们的结果表明，满足这些条件的方法避免了损失和梯度尖峰的发生。
相反，未能满足这些条件的方法仍然容易受到梯度尖峰的影响，即使它们以前被推荐作为损失尖峰问题的经验解决方案。
此外，我们证明了满足这些条件的方法使得LLM能够以相对较大的学习率进行预训练，从而带来更优越的性能结果。

## 2 预备知识

### 2.1 Pre-LN Transformer

本文主要关注GPT系列中使用的神经架构（Radford et al. 2018; Radford et al. 2019; Brown et al. 2020）。
它们使用了Pre-LN Transformer（Xiong et al. 2020），这是最近Transformer实现中事实上的标准架构，因为当我们堆叠许多层时，使用该架构的训练比原始Transformer架构更稳定（Xiong et al. 2020; Liu et al. 2020; Takase et al. 2023）。
设$x\in\mathbb{R}^{d}$为Transformer某层的输入，其中$d$表示该层的维度。
该层通过以下等式输出$y$：

$$
\displaystyle y
$$
$$
\displaystyle=x^{\prime}+\mathrm{FFN}(\mathrm{LN}(x^{\prime})),
$$
$$
\displaystyle x^{\prime}
$$
$$
\displaystyle=x+\mathrm{Attn}(\mathrm{LN}(x)),
$$
(1)

其中$\mathrm{LN}$是层归一化函数（脚注：我们在附录B.4中讨论了与使用均方根层归一化（RMSNorm）（Zhang & Sennrich 2019）代替LN的架构的区别，并在附录H中讨论了原始Transformer架构，即Post-LN Transformer。）。
我们将等式（1）和（2）中的第一项，即$x$和$x^{\prime}$，称为捷径连接（shortcut）。
此外，前馈网络（$\mathrm{FFN}$）和多头自注意力（$\mathrm{Attn}$）定义如下（脚注：为了简化等式，我们省略了偏置项。）：

$$
\displaystyle\mathrm{FFN}(x)
$$
$$
\displaystyle=W_{2}(\mathcal{F}(W_{1}\ x)),
$$
$$
\displaystyle\mathrm{Attn}(x)
$$
$$
\displaystyle=W_{O}(\mathrm{concat}(\mathrm{head}_{1}(x),...,\mathrm{head}_{h}(x))),
$$
$$
\displaystyle\mathrm{head}_{i}(x)
$$
$$
\displaystyle=\mathrm{softmax}\left(\frac{(W_{Qi}\ x)^{\mathrm{T}}(W_{Ki}\ X)}{\sqrt{d_{\mathrm{head}}}}\right)(W_{Vi}\ X)^{\mathrm{T}},
$$
(3)

其中$\mathcal{F}$是激活函数，$\mathrm{concat}$拼接输入向量，$\mathrm{softmax}$对输入向量应用softmax函数，$W_{1}\in\mathbb{R}^{d_{\mathrm{ffn}}\times d}$、$W_{2}\in\mathbb{R}^{d\times d_{\mathrm{ffn}}}$、$W_{Qi}\in\mathbb{R}^{d_{\mathrm{head}}\times d}$、$W_{Ki}\in\mathbb{R}^{d_{\mathrm{head}}\times d}$、$W_{Vi}\in\mathbb{R}^{d_{\mathrm{head}}\times d}$和$W_{O}\in\mathbb{R}^{d\times d}$是参数矩阵，$d_{\mathrm{ffn}}$和$d_{\mathrm{head}}$分别是FFN和多头自注意力子层的内部维度。
此外，我们将输入向量序列打包成矩阵$X\in\mathbb{R}^{d\times L}$，其中$L$是输入序列长度，以计算自注意力。

### 2.2 Pre-LN Transformer的梯度

设$\mathcal{L}$为$N$层Pre-LN Transformer的损失函数，$J_{n}$为第$n$层的雅可比矩阵。
我们可以使用等式（1）和（2）中的关系计算$\mathcal{L}$的梯度如下：

$$
\displaystyle\frac{\partial\mathcal{L}}{\partial x_{1}}=\frac{\partial\mathcal{L}}{\partial y_{N}}\prod_{n=1}^{N-1}J_{n}=\frac{\partial\mathcal{L}}{\partial y_{N}}\prod_{n=1}^{N-1}\left(\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\frac{\partial x^{\prime}_{n}}{\partial x_{n}}\right),\quad\mbox{where}\quad J_{n}=\frac{\partial y_{n}}{\partial x_{n}}=\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\frac{\partial x^{\prime}_{n}}{\partial x_{n}}.
$$
(6)

利用谱范数的次乘积性（submultiplicativity），即$\|AB\|_{2}\leq\|A\|_{2}\|B\|_{2}$，以及等式（6），我们可以推导出$\mathcal{L}$梯度范数的上界如下：

$$
\displaystyle\bigg\|\frac{\partial\mathcal{L}}{\partial x_{1}}\bigg\|_{2}=\bigg\|\frac{\partial\mathcal{L}}{\partial y_{N}}\prod_{n=1}^{N-1}\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\frac{\partial x^{\prime}_{n}}{\partial x_{n}}\bigg\|_{2}\leq\bigg\|\frac{\partial\mathcal{L}}{\partial y_{N}}\bigg\|_{2}\prod_{n=1}^{N-1}\bigg\|\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\bigg\|_{2}\bigg\|\frac{\partial x^{\prime}_{n}}{\partial x_{n}}\bigg\|_{2}.
$$
(7)

因此，我们可以通过分析FFN层和自注意力层的雅可比矩阵（即$\|\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\|_{2}$和$\|\frac{\partial x^{\prime}_{n}}{\partial x_{n}}\|_{2}$）的谱范数，来估计$\mathcal{L}$梯度范数的上界。

### 2.3 抑制上界的动机

在我们的初步实验中，当LLM预训练期间梯度范数突然增长时，我们观察到很可能发生损失尖峰问题。
因此，我们假设可以通过保持梯度范数较小来防止损失尖峰问题。
为了防止梯度范数增长，我们探索了抑制等式（7）所描述的上界的方法。
为了抑制该上界，我们在接下来的小节中分析了雅可比矩阵以寻找控制上界的因素，然后提供了两个条件：较小的子层和较大的捷径连接。
我们通过LLM预训练的实验验证了我们的假设和理论分析。

## 3 子层梯度的分析

对于本节的理论分析，我们采用以下假设：

#### 假设1。

设$x$和$x^{\prime}$为每一层的输入和中间向量。此外，设$W_{*}$表示每一层中的模型参数矩阵。
我们假设所有层的$x$、$x^{\prime}$和$W_{*}$均服从均值为0的正态分布，即$\mu=0$。

当我们使用正态分布初始化参数、等式（4）中的头数为$1$且$\mathcal{F}$为恒等函数时，此假设是成立的。根据经验，每个子层的输出都接近正态分布，如附录D所示。

### 3.1 FFN的雅可比矩阵

基于等式（1），等式（7）中的$\|\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\|_{2}$可以重写为：

$$
\displaystyle\bigg\|\frac{\partial y}{\partial x^{\prime}}\bigg\|_{2}
$$
$$
\displaystyle=\bigg\|\frac{\partial(x^{\prime}+\mathrm{FFN}(\mathrm{LN}(x^{\prime})))}{\partial x^{\prime}}\bigg\|_{2}=\bigg\|I+\frac{\partial(\mathrm{FFN}(\mathrm{LN}(x^{\prime})))}{\partial x^{\prime}}\bigg\|_{2}
$$
(8)

然后，我们可以通过应用谱范数的次可加性（subadditivity），即$\|A+B\|_{2}\leq\|A\|_{2}+\|B\|_{2}$，以及次乘积性，推导出$\|\frac{\partial y_{n}}{\partial x^{\prime}_{n}}\|_{2}$的上界如下：

$$
\displaystyle\bigg\|\frac{\partial y}{\partial x^{\prime}}\bigg\|_{2}
$$
$$
\displaystyle\leq 1+\bigg\|\frac{\partial\mathrm{FFN}(\mathrm{LN}(x^{\prime}))}{\partial\mathrm{LN}(x^{\prime})}\bigg\|_{2}\bigg\|\frac{\partial\mathrm{LN}(x^{\prime})}{\partial x^{\prime}}\bigg\|_{2}.
$$
(9)

该不等式的右侧表明，我们可以通过分别计算$\mathrm{FFN}$和$\mathrm{LN}$的雅可比矩阵的谱范数来估计$\|\frac{\partial y}{\partial x^{\prime}}\|_{2}$的上界。

关于$\mathrm{FFN}$部分，
我们假设激活函数$\mathcal{F}$是恒等函数（脚注：附录E讨论了我们使用$\mathrm{ReLU}$、$\mathrm{SiLU}$和$\mathrm{SwiGLU}$作为激活函数的情况，这得出了相同的结论。）
以简化讨论。
在此假设下，以下等式成立：

$$
\displaystyle\bigg\|\frac{\partial\mathrm{FFN}(\mathrm{LN}(x^{\prime}))}{\partial\mathrm{LN}(x^{\prime})}\bigg\|_{2}=\|W_{2}W_{1}\|_{2}.
$$
(10)

因此，我们可以从谱范数的次乘积性直接推导出关系$\|W_{2}W_{1}\|_{2}\leq\|W_{1}\|_{2}\|W_{2}\|_{2}$。
此外，设$\sigma_{1}$和$\sigma_{2}$分别为$W_{1}$和$W_{2}$的标准差。
根据假设1，$W_{1}$和$W_{2}$的谱范数由它们的标准差和维度得出（Vershynin 2018），即$\|W_{1}\|_{2}\approx\sigma_{1}(\sqrt{d}+\sqrt{d_{\mathrm{ffn}}})$和$\|W_{2}\|_{2}\approx\sigma_{2}(\sqrt{d}+\sqrt{d_{\mathrm{ffn}}})$。
最后，我们可以将$\mathrm{FFN}$的雅可比矩阵谱范数的上界表示为以下不等式：

$$
\displaystyle\bigg\|\frac{\partial\mathrm{FFN}(\mathrm{LN}(x^{\prime}))}{\partial\mathrm{LN}(x^{\prime})}\bigg\|_{2}\leq\sigma_{1}\sigma_{2}(\sqrt{d}+\sqrt{d_{\mathrm{ffn}}})^{2},
$$
(11)

其中右侧具有关系$\sigma_{1}\sigma_{2}(\sqrt{d}+\sqrt{d_{\mathrm{ffn}}})^{2}\approx\|W_{1}\|_{2}\|W_{2}\|_{2}$。

接下来，关于$\mathrm{LN}$部分，LN的雅可比矩阵可以写为：

$$
\displaystyle\frac{\partial\mathrm{LN}(x^{\prime})}{\partial x^{\prime}}=\frac{\sqrt{d}}{\|x^{\prime}\|_{2}}\bigg(I-\frac{x^{\prime}x^{\prime\top}}{\|x^{\prime}\|_{2}^{2}}\bigg)=\frac{\sqrt{d}}{\sigma_{x^{\prime}}\sqrt{d}}\bigg(I-\frac{x^{\prime}x^{\prime\top}}{\sigma_{x^{\prime}}^{2}d}\bigg)=\frac{1}{\sigma_{x^{\prime}}}\bigg(I-\frac{zz^{\top}}{d}\bigg).
$$
(12)

最左边的等式出现在Xiong等人（2020）的证明中。
第二个等式使用了$\|x^{\prime}\|_{2}=\sigma_{x^{\prime}}\sqrt{d}$，这可以基于假设1得出。
最后一个等式源自著名的$z=(x^{\prime}-\mu_{x^{\prime}})/\sigma_{x^{\prime}}$公式，该公式将正态分布$x^{\prime}$转换为标准正态分布$z$，其中假设1中的$\mu_{x^{\prime}}=0$。

我们考虑矩阵$zz^{\top}$中每个元素的方差（$\mathrm{var}$）。
由于$z_{i}z_{i}$服从自由度为$1$的$\mathcal{X}^{2}$，并且$z_{i}z_{j}(i\neq j)$是两个服从标准正态分布的独立值的乘积，因此方差如下：

$$
\displaystyle\mathrm{var}(z_{i}z_{j})=\begin{cases}1\ \ \text{if $i\neq j$}\\
2\ \ \text{otherwise}\end{cases}.
$$
(13)

等式（13）表明，在LLM中由于$d\gg 1$，$\frac{zz^{\top}}{d}\approx 0$。
因此，LN雅可比矩阵的谱范数可以写为：

$$
\displaystyle\bigg\|\frac{\partial\mathrm{LN}(x^{\prime})}{\partial x^{\prime}}\bigg\|=\frac{1}{\sigma_{x^{\prime}}},\quad\mbox{where}\quad\frac{\partial\mathrm{LN}(x^{\prime})}{\partial x^{\prime}}=\frac{1}{\sigma_{x^{\prime}}}I.
$$
(14)

最后，通过代入等式（11）和（14），等式（9）可以重写为：

$$
\displaystyle\bigg\|\frac{\partial y}{\partial x^{\prime}}\bigg\|_{2}
$$
$$
\displaystyle\leq 1+\frac{\sigma_{1}\sigma_{2}}{\sigma_{x^{\prime}}}C_{\mathrm{ffn}},
$$
(15)

其中为了简化，$C_{\mathrm{ffn}}=(\sqrt{d}+\sqrt{d_{\mathrm{ffn}}})^{2}$。

根据第2.3节的讨论和等式（15），
$W_{1}$和$W_{2}$的标准差（分别为$\sigma_{1}$和$\sigma_{2}$）应该足够小，并且捷径连接$x^{\prime}$的标准差$\sigma_{x^{\prime}}$应该满足$\sigma_{1}\sigma_{2}\ll\sigma_{x^{\prime}}$，以保持上界较小。

### 3.2 自注意力的雅可比矩阵

与$\mathrm{FFN}$类似，我们可以使用等式（2）将等式（7）中的$\|\frac{\partial x^{\prime}}{\partial x}\|_{2}$重写为：

$$
\displaystyle\bigg\|\frac{\partial x^{\prime}}{\partial x}\bigg\|_{2}
$$
$$
\displaystyle=\bigg\|\frac{\partial(x+\mathrm{Attn}(\mathrm{LN}(x)))}{\partial x}\bigg\|_{2}=\bigg\|I+\frac{\partial(\mathrm{Attn}(\mathrm{LN}(x)))}{\partial x}\bigg\|_{2}.
$$
(16)

然后，我们可以通过应用谱范数的次可加性和次乘积性推导出$\|\frac{\partial x^{\prime}}{\partial x}\|_{2}$的上界，即：

$$
\displaystyle\bigg\|\frac{\partial x^{\prime}}{\partial x}\bigg\|_{2}
$$
$$
\displaystyle\leq 1+\bigg\|\frac{\partial\mathrm{Attn}(\mathrm{LN}(x))}{\partial\mathrm{LN}(x)}\bigg\|_{2}\bigg\|\frac{\partial\mathrm{LN}(x)}{\partial x}\bigg\|_{2}.
$$
(17)

因此，为了估计$\|\frac{\partial x^{\prime}}{\partial x}\|_{2}$的上界，我们计算$\mathrm{Attn}$和$\mathrm{LN}$的雅可比矩阵的谱范数。

设$Z(\cdot)=\mathrm{concat}(\mathrm{head}_{1}(\cdot),...,\mathrm{head}_{h}(\cdot)))$，并设$J^{Z}$为$Z(\cdot)$的雅可比矩阵（脚注：我们在附录G中讨论了$J^{Z}$的细节。），我们可以将Attn雅可比矩阵的谱范数重写为：

$$
\displaystyle\bigg\|\frac{\partial\mathrm{Attn}(\mathrm{LN}(x))}{\partial\mathrm{LN}(x)}\bigg\|_{2}=\bigg\|\frac{\partial W_{O}Z(\mathrm{LN}(x))}{\partial Z(\mathrm{LN}(x))}\frac{\partial Z(\mathrm{LN}(x))}{\partial\mathrm{LN}(x)}\bigg\|_{2}=\|W_{O}J^{Z}\|_{2}.
$$
(18)

因此，我们可以从谱范数的次乘积性直接推导出关系$\|W_{O}J^{Z}\|_{2}\leq\|W_{O}\|_{2}\|J^{Z}\|_{2}$。

设$\sigma_{O}$为$W_{O}$的标准差。
关系$\|W_{O}\|_{2}\approx\sigma_{O}(2\sqrt{d})$由假设1推导得出。
我们将此值代入等式（18）并得到以下不等式：

$$
\displaystyle\bigg\|\frac{\partial\mathrm{Attn}(\mathrm{LN}(x))}{\partial\mathrm{LN}(x)}\bigg\|_{2}\leq\sigma_{O}(2\sqrt{d})\|J^{Z}\|_{2}.
$$
(19)

因此，我们可以通过代入等式（14）和（19）将等式（17）重写如下：

$$
\displaystyle\bigg\|\frac{\partial x^{\prime}}{\partial x}\bigg\|_{2}
$$
$$
\displaystyle\leq 1+\frac{\sigma_{O}}{\sigma_{x}}C_{\mathrm{Attn}},
$$
(20)

其中为了简化，$C_{\mathrm{Attn}}=(2\sqrt{d})\|J^{Z}\|_{2}$。

因此，与第3.1节末尾的讨论类似，$W_{O}$的标准差$\sigma_{O}$应该较小，并且捷径连接$x$的标准差$\sigma_{x}$应该满足$\sigma_{O}\ll\sigma_{x}$，以保持上界较小。

## 4 避免尖峰的条件

基于第3节的讨论，为了稳定大语言模型（LLM）的预训练，我们必须关注 $\sigma_{1}$、$\sigma_{2}$、$\sigma_{O}$ 的值以及捷径（shortcut）的标准差。
为了使 $\sigma_{1}$、$\sigma_{2}$ 和 $\sigma_{O}$ 变小，我们必须用较小的值初始化相应的参数。
让我们详细考虑实际的设置。
广泛用于LLM的初始化方法（Shoeybi et al. 2020; Le Scao et al. 2022; Biderman et al. 2023）使用正态分布 $\mathcal{N}(0,\sigma^{2})$ 初始化所有参数，其中 $\sigma=\sqrt{\frac{2}{5d}}$（Nguyen & Salazar 2019），然后根据层数将 $W_{2}$ 和 $W_{O}$ 缩放为较小的值：$\sqrt{\frac{1}{2N}}$，其中 $N$ 是层数（脚注：Biderman et al. 2023 在初始化中也将 $W_{2}$ 和 $W_{O}$ 缩放为较小的值，但他们使用了 Wang & Komatsuzaki 2021 引入的策略，而不是使用 $\sqrt{\frac{1}{2N}}$ 进行缩放。然而，其性质本质上是相同的，因为他们使用 $\sigma=\frac{2}{N\sqrt{d}}$ 初始化 $W_{2}$ 和 $W_{O}$，该值会根据层数变小。）。
在这种情况下，$\sigma_{1}$、$\sigma_{2}$ 和 $\sigma_{O}$ 是足够小的值。

然而，在这种情况下，捷径的标准差也太小了。
例如，在浅层，标准差接近 $\sqrt{\frac{2}{5d}}$，因为嵌入矩阵也是由 $\mathcal{N}(0,\sigma^{2})$ 初始化的，其中 $\sigma=\sqrt{\frac{2}{5d}}$。
因此，为了增加捷径的标准差，我们使每个嵌入的标准差接近 $1$（脚注：根据公式 (15) 和 (20)，上限随着捷径标准差的增加而变小。然而，如附录 C 所述，过大的值在经验上会降低性能。）。
为了实现这一点，我们引入了两种修改：“Scaled Embed”和“Embed LN”（脚注：我们可以通过使用正态分布 $\mathcal{N}(0,\sigma^{2})$ 初始化嵌入来满足该条件，其中 $\sigma=1$，但我们在本研究中没有采用这种策略，因为我们在实验中使用了相同的初始化方法。）。
Scaled Embed 使用适当的值对嵌入进行缩放。
例如，我们将嵌入乘以 $\sqrt{d}$，这在原始的 Transformer 论文（Vaswani et al. 2017）中被使用过（脚注：尽管原始的 Transformer 论文引入了此操作，但最近的实现忽略了这一点。因此，使用最近的实现训练的 LLM 不满足大捷径的条件。），然后嵌入的标准差变为 $\sqrt{\frac{2}{5}}$。
Embed LN 将层归一化（LN）应用于嵌入。
事实上，Le Scao et al. 2022 报告称，Embed LN 策略在经验上可以防止损失尖峰。
这两种方法作为验证示例提出，而不是作为提出的方法，如果满足条件，也可以采用其他替代方法。

**图 2：在 LLM 预训练开始时，每个 Transformer 层的公式 (15) 中描述的实际上限。由于很难估计所有层中 $\sigma_{x}$ 的严格值，我们通过使用一些输入来获得经验值，并将它们代入公式 (15)。**

为了展示公式 (15) 中描述的上限的实际值，我们以具有 17 亿参数的模型为例。
除了广泛用于 LLM 的初始化（Vanilla）和上述两种修改（Scaled Embed 和 Embed LN）之外，我们还比较了 Xavier Init，它使用 Xavier 初始化（Glorot & Bengio 2010）初始化所有参数，作为我们不根据层数缩放 $W_{2}$ 和 $W_{O}$ 的情况。
图 2 显示了预训练开始时每一层的公式 (15) 的值。
该图表明，没有抑制上限的方法（即 Xavier Init 和 Vanilla）会迅速增加该值，尤其是在浅层。
相比之下，Scaled Embed 和 Embed LN 保持了较小的值。
总之，为了使梯度范数的上限变小以稳定 LLM 预训练，我们必须满足两个条件：（1）小规模子层；用较小的值初始化子层的参数，以及（2）大捷径；使每个嵌入的标准差接近 $1$。

## 5 实验

我们验证了理论分析的经验有效性。
具体而言，我们证明了控制梯度范数的上限也能防止损失和梯度尖峰。
为了评估实际情况下的功效，我们在主要实验中重点关注使用广泛使用的方法（Shoeybi et al. 2020; Le Scao et al. 2022）初始化的方法（脚注：不满足小规模子层条件的 Xavier 初始化近年来并未广泛用于 LLM。附录 B.1 表明 Xavier 初始化的性能较差，并且未能避免尖峰。）。

### 5.1 数据集

我们使用 C4（Raffel et al. 2020）作为我们的 LLM 预训练语料库，该语料库由从 Common Crawl 提取的干净英文文本组成。
我们还使用 C4 的分离部分作为我们的验证数据。
我们使用包含字节对编码（BPE）子词单元（Sennrich et al. 2016）的 GPT-2 词表（Radford et al. 2019）作为我们的词表。
为了评估每种方法，我们在 WikiText（Merity et al. 2017）和 LAMBADA（Paperno et al. 2016）数据集上计算了困惑度。

### 5.2 模型配置

如第 2 节所述，我们使用了 Pre-LN Transformer 架构。
我们使用两种参数规模进行实验：3.5 亿（350M）和 17 亿（1.7B）。
我们将学习率（$\mathrm{lr}$）设置为 $5.0\times 10^{-4}$。
附录 A 描述了有关实验配置的更多细节。
我们比较了以下方法。
如果该方法满足抑制上限的两个条件，我们在方法名称前加上 $\checkmark$。

#### Vanilla

这是 LLM 预训练的标准配置。
由于此配置不抑制梯度范数的上限，因此很可能发生损失尖峰。

#### Embed Detach

Zeng et al. 2023 使用了收缩嵌入梯度技术（Ding et al. 2021）来稳定他们的 LLM 预训练。
该方法通过从计算图中分离一部分嵌入来收缩嵌入层上的梯度，如下所示：

$$
\displaystyle\mathrm{Embed}\leftarrow\gamma\mathrm{Embed}+(1-\gamma)\mathrm{Detach}(\mathrm{Embed}),
$$
(21)

其中 $\gamma$ 是一个超参数，$\mathrm{Detach}$ 将输入从计算图中分离。
按照 Zeng et al. 2023 的做法，我们将 $0.1$ 赋值给 $\gamma$。
Zeng et al. 2023 指出，该方法在经验上可以防止损失尖峰。
然而，该方法不满足大捷径的条件，因此，我们表明该方法不能完全解决损失尖峰问题。

#### $\checkmark$Embed LN

Dettmers et al. 2022 和 Le Scao et al. 2022 报告称，将 LN 应用于嵌入层可以稳定他们的 LLM 预训练。
如第 4 节所述，该方法满足控制梯度范数上限的条件。

#### $\checkmark$Scaled Embed

该方法将嵌入乘以 $\sqrt{d}$。
如第 4 节所述，该方法满足控制梯度范数上限的要求。

### 5.3 主要结果

图 3 显示了每种方法在验证数据中的损失值。
图 4 显示了每种方法的梯度范数。
这些图表明 Vanilla 和 Embed Detach 面临损失和梯度尖峰。
相比之下，Embed LN 和 Scaled Embed 没有面临尖峰。
这些结果与我们在第 3 节和第 4 节中描述的理论分析相符。
因此，只有使梯度范数上限变小的方法才能成功避免 LLM 预训练中的尖峰。

| 模型 | WikiText $\downarrow$ | LAMBADA $\downarrow$ |
| --- | --- | --- |
| 350M 参数 | | |
| Vanilla | 30.03 | 24.73 |
| Embed Detach | 30.69 | 26.93 |
| Embed LN | 29.85 | 25.03 |
| Scaled Embed | 29.86 | 24.37 |
| 1.7B 参数 | | |
| Vanilla | 22.58 | 15.22 |
| Embed Detach | 22.00 | 13.88 |
| Embed LN | 21.29 | 13.00 |
| Scaled Embed | 21.29 | 12.53 |

**表 1：每种方法的困惑度。**

在 350M 和 1.7B 参数的比较中，尖峰在 1.7B 参数中发生得更频繁。
因为我们使用 $\mathcal{N}(0,\sigma^{2})$ 初始化嵌入，其中 $\sigma=\sqrt{\frac{2}{5d}}$，所以在 Vanilla 和 Embed Detach 中，随着 $d$ 变大，嵌入的标准差变小。
这意味着公式 (15) 和 (20) 描述的上限随着 $d$ 的增大而变大，因为在浅层中 $\sigma_{x}$ 和 $\sigma_{x}^{\prime}$ 几乎等于输入嵌入的标准差。
因此，如果我们在没有任何技术来控制梯度范数上限的情况下增加 $d$，模型将变得更加不稳定。
这一结果与先前的研究报告（Le Scao et al. 2022; Chowdhery et al. 2022; Zeng et al. 2023）相符，即随着参数数量的增加，他们的模型变得更加不稳定。

表 1 显示了每种方法在 WikiText 和 LAMBADA 上的困惑度。
该表显示 Embed LN 和 Scaled Embed 实现了相当的性能。
这一结果意味着，如果每种方法都能防止损失和梯度尖峰，那么它们在性能上没有显著差异。
相比之下，除了在 LAMBADA 中具有 350M 参数的 Vanilla 之外，Vanilla 和 Embed Detach 的困惑度更差，并且在大量参数下性能差异更大。
这一结果意味着，随着参数规模的增大，解决尖峰问题对性能的影响更为严重。
我们在下一小节中对更大的模型进行实验。
此外，我们在附录 B 中讨论了其他设置。

**(a) 350M 参数。**

**(b) 1.7B 参数。**

**图 3：每种方法在验证数据中的损失曲线。**

**(a) 350M 参数。**

**(b) 1.7B 参数。**

**图 4：训练期间每种方法的梯度范数。**

### 5.4 13B 参数模型的结果

我们对 13B 参数模型的预训练进行了实验，以表明我们提供的条件可以稳定比第 5.3 节中讨论的参数多得多的模型。
由于计算预算的限制，我们重点比较了 Vanilla 和 Scaled Embed。
我们尝试了两种学习率：$3.0\times 10^{-4}$ 和 $1.0\times 10^{-4}$。
附录 A 描述了有关超参数的更多细节。

图 5 显示了验证数据中每种配置的损失值。
如图中 (a) 所示，在 $\mathrm{lr}=3.0\times 10^{-4}$ 中，Vanilla 的损失值从大约 10000 步开始上升。
然后，该模型的梯度变得太大而无法继续其预训练。
相比之下，Scaled Embed 的损失值持续下降。
这一结果表明，Scaled Embed 稳定了具有大量参数的模型的预训练。

**(a) $\mathrm{lr}=3.0\times 10^{-4}$。**

**(b) $\mathrm{lr}=1.0\times 10^{-4}$。**

**图 5：当我们使用两种学习率 $\mathrm{lr}=3.0\times 10^{-4}$ 和 $1.0\times 10^{-4}$ 时，具有 13B 参数的每种方法的损失值**

| | WikiText $\downarrow$ | | LAMBADA $\downarrow$ | |
| --- | --- | --- | --- | --- |
| 模型 | $\mathrm{lr}=3.0\times 10^{-4}$ | $\mathrm{lr}=1.0\times 10^{-4}$ | $\mathrm{lr}=3.0\times 10^{-4}$ | $\mathrm{lr}=1.0\times 10^{-4}$ |
| Vanilla | N/A | 15.12 | N/A | 6.50 |
| Scaled Embed | 14.47 | 15.25 | 5.97 | 6.53 |

**表 2：当我们使用两种学习率 $\mathrm{lr}=3.0\times 10^{-4}$ 和 $1.0\times 10^{-4}$ 时，具有 13B 参数的每种方法的困惑度。**

| 模型 | PIQA $\uparrow$ | OpenBookQA $\uparrow$ | HellaSwag $\uparrow$ | WinoGrande $\uparrow$ | ARC-easy $\uparrow$ | ARC-challenge $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| | $\mathrm{lr}=3.0\times 10^{-4}$ | | | | | |
| Vanilla | N/A | | | | | |
| Scaled Embed | 78.94 | 39.20 | 71.03 | 63.77 | 60.31 | 35.49 |
| | $\mathrm{lr}=1.0\times 10^{-4}$ | | | | | |
| Vanilla | 77.80 | 38.40 | 69.10 | 61.64 | 57.95 | 33.53 |
| Scaled Embed | 78.07 | 39.20 | 68.92 | 62.59 | 58.92 | 33.19 |

**表 3：具有 13B 参数的每种方法在标准任务上的性能。**

表 2 显示了评估数据中每种配置的困惑度。
该表表明，当我们使用更大的学习率时，我们可以获得更好的性能。
此外，当我们使用较小的学习率 $\mathrm{lr}=1.0\times 10^{-4}$ 时，Scaled Embed 的困惑度与 Vanilla 相当。
表 3 显示了每种配置在标准基准数据集上的性能：PIQA（Bisk et al. 2020）、OpenBookQA（Mihaylov et al. 2018）、HellaSwag（Zellers et al. 2019）、WinoGrande（Sakaguchi et al. 2021）以及 ARC easy 和 challenge（Clark et al. 2018）。
该表显示了与困惑度评估一致的结果。
这些结果意味着我们提供的条件在预训练中没有显著风险。
因此，除了小规模子层之外，我们还必须满足大捷径的条件，以稳定 LLM 的预训练。

## 6 相关工作

#### 稳定性

为了稳定基于 Transformer 的神经语言模型的训练，人们在架构（Xiong et al. 2020; Liu et al. 2020; Takase et al. 2023; Zeng et al. 2023; Zhai et al. 2023）、初始化方法（Nguyen & Salazar 2019; Zhang et al. 2019; Huang et al. 2020; Wang et al. 2022）、训练策略（Zhang et al. 2022; Li et al. 2022）和损失函数（Chowdhery et al. 2022; Wortsman et al. 2023）方面进行了各种讨论。

Xiong et al. 2020 从理论上分析了 Transformer 各部分的梯度尺度，并指出 Pre-LN Transformer 比 Post-LN Transformer（即原始的 Transformer 架构，Vaswani et al. 2017）更稳定。由于 Pre-LN Transformer 在理论和经验上都比 Post-LN Transformer 更稳定，最近的研究主要使用 Pre-LN Transformer 来构建 LLM。在本文对训练动态的分析中，我们也假设使用 Pre-LN Transformer。

为了稳定 LLM 预训练，Le Scao et al. 2022 将层归一化（layer normalization）应用于嵌入层（embedding layer）。Zeng et al. 2023 使用了收缩嵌入梯度（shrink embedding gradient）技术 (Ding et al. 2021)。Nishida et al. 2024 提出了作为重参数化的权重缩放（weight scaling as reparameterization, WeSaR），该方法使用额外的参数来缩放内部层的参数。在本研究中，我们从理论上证明了，当我们使用广泛用于 LLM 的初始化方法 (Nguyen & Salazar 2019; Shoeybi et al. 2020) 时，对嵌入层进行层归一化可以控制子层梯度范数的上界，从而稳定预训练。

对于初始化方法，最近的研究提出了最大更新参数化（maximal update parameterization, $\mu$P）及其变体，作为将超参数从较小模型迁移到较大模型而无需承担超参数搜索成本的方法 Yang & Hu 2021; Yang et al. 2024a; Yang et al. 2024b。然而，正如 Yang et al. 2024b 所述，当应用于 Transformer 时，此类方法可能无效，因为它们的潜在假设未能准确反映实际的 Transformer 架构。在专注于 Transformer 的研究中，Nguyen & Salazar 2019 提出了一种用较小值初始化 Transformer 参数的策略，以稳定其训练。Zhang et al. 2019 和 Huang et al. 2020 指出，如果我们使用他们提出的初始化方法，就可以移除 Transformer 中的层归一化。Wang et al. 2022 根据层数调整了初始参数尺度，以稳定 Post-LN Transformer。在本研究中，我们指出，广泛使用的使参数变小的初始化方法 (Shoeybi et al. 2020) 对于稳定 LLM 预训练是必要的。此外，我们证明了通过使嵌入的标准差接近 $1$，我们可以防止损失尖峰（loss spike）问题。

#### 效率

如第 5.4 节和附录 B.2 所示，我们的修改使得能够使用相对较大的学习率进行预训练，并能实现更好的性能。因此，这项研究可以被视为关于 LLM 预训练效率的研究，因为我们的修改可以在给定预算下构建更好的 LLM。Strubell et al. 2019 和 Schwartz et al. 2019 报告称，最近的神经方法需要大量的计算成本，因此，他们认为我们必须探索一种具有成本效益的方法。Rajbhandari et al. 2020 提出了 ZeRO，它在不增加通信量的情况下减少了多 GPU 训练期间的内存冗余。Dao et al. 2022 专注于 GPU 内存读写，并提出了 FlashAttention，它加速了 Transformer 中注意力机制的速度。为了减少注意力机制中的计算量，Shazeer 2019 提出了多查询注意力（multi-query attention），在每一层的所有注意力头之间共享一个键（key）和值（value）。Takase & Kiyono 2023 探索了几种参数共享策略，并指出在某些层之间共享参数可以用较少的参数实现与原始模型相当的性能。此外，一些研究探索了在有限预算下更好的构建方法 (Izsak et al. 2021; Takase & Kiyono 2021)。我们相信，我们可以利用他们的发现使我们的 LLM 更加高效。

## 7 结论

本文探讨了为什么大型语言模型（LLM）在预训练期间有时会经历损失尖峰。为了提供证据，我们特别关注了子层的梯度。通过分析子层雅可比矩阵的谱范数，我们引入了梯度范数的上界。然后，我们从理论上确定了避免损失尖峰的两个条件：较小的子层（small sub-layers）和较大的捷径（large shortcut）。为了满足这些条件，我们表明，使用广泛采用的 LLM 初始化方法可以使子层参数变小，并且嵌入缩放或将层归一化结合到嵌入层中可以使每个嵌入的标准差接近 $1$，从而产生较大的捷径。实验结果表明，满足这些条件的方法避免了损失尖峰。此外，这些方法允许使用相对较大的学习率进行训练，从而提高性能。我们希望我们的理论分析和经验发现将有助于避免在 LLM 构建过程中浪费宝贵的时间和计算预算。

## 伦理声明

为了稳定 LLM 预训练，本文对子层雅可比矩阵的谱范数进行了理论分析，以估计 $\mathcal{L}$ 梯度范数的上界。本文仅关注 LLM 预训练的稳定性，因此，为了在实际应用中使用 LLM，我们必须解决 LLM 的其他问题，例如幻觉（hallucinations）。

## 可复现性声明

我们在本文中并不旨在提出一种新颖的方法，而是主要关注对雅可比矩阵谱范数的理论分析，以寻找稳定 LLM 预训练的因素。我们通过在各种情况下的实验来证明我们的理论分析。为了激活我们的修改，我们仅向广泛使用的实现（即 Megatron-LM（脚注：https://github.com/NVIDIA/Megatron-LM））中添加了几行代码。因此，我们相信很容易复现我们的实验结果。此外，我们没有修改 Transformer 的内部架构。特别是，Scaled Embed 仅通过一个常数值来缩放嵌入，这使得它在跨各种实现（例如 HuggingFace 中的 Transformers（脚注：https://huggingface.co/docs/transformers/main/index）和 vLLM（脚注：https://github.com/vllm-project/vllm））的移植性方面非常方便。

然而，由于很难断言我们提供的条件完全解决了 LLM 预训练期间的不稳定性，因此最好结合其他稳定预训练的技术，例如 Chowdhery et al. 2022 描述的辅助损失（auxiliary loss），以在实际预训练情况中使预训练更加稳定。
