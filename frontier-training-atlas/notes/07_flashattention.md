<!-- 来源：https://arxiv.org/pdf/2205.14135；翻译范围：摘要至参考文献/附录前 -->

# FlashAttention：具有IO感知的快速且内存高效的精确注意力机制

###### 摘要

在长序列上，Transformer速度缓慢且极其消耗内存，因为自注意力机制的时间和内存复杂度与序列长度呈二次方关系。
近似注意力方法试图通过牺牲模型质量来降低计算复杂度以解决此问题，但通常无法实现挂钟时间（wall-clock）的加速。
我们认为缺失的一个原则是使注意力算法具备IO感知能力——即考虑GPU不同层级内存之间的读写操作。
我们提出了FlashAttention，这是一种具有IO感知的精确注意力算法，它使用分块（tiling）技术来减少GPU高带宽内存（HBM）和GPU片上SRAM之间的内存读写次数。
我们分析了FlashAttention的IO复杂度，表明它比标准注意力机制需要更少的HBM访问，并且在一定范围的SRAM大小下是最优的。
我们还将FlashAttention扩展到块稀疏（block-sparse）注意力，产生了一种比任何现有近似注意力方法都快的近似注意力算法。
FlashAttention训练Transformer的速度比现有基线更快：与MLPerf 1.1训练速度记录相比，在BERT-large（序列长度512）上实现了15%的端到端挂钟时间加速；在GPT-2（序列长度1K）上实现了3$\times$的加速；在Long-Range Arena（序列长度1K-4K）上实现了2.4$\times$的加速。
FlashAttention和块稀疏FlashAttention使Transformer能够处理更长的上下文，从而产生更高质量的模型（在GPT-2上困惑度改善了0.7，在长文档分类上提升了6.4个百分点）以及全新的能力：首批在Path-X挑战（序列长度16K，准确率61.4%）和Path-256（序列长度64K，准确率63.1%）上实现优于随机猜测性能的Transformer模型。

## 1 引言

Transformer模型[82]已成为自然语言处理和图像分类等应用中最广泛使用的架构。
Transformer变得越来越大[5]且越来越深[83]，但为其配备更长的上下文仍然很困难[80]，因为其核心的自注意力模块的时间和内存复杂度与序列长度呈二次方关系。
一个重要的问题是，使注意力机制变得更快且更节省内存，是否能帮助Transformer模型解决其在长序列上的运行时间和内存挑战。

许多近似注意力方法旨在减少注意力机制的计算和内存需求。
这些方法涵盖了从稀疏近似[51, 74]到低秩近似[84, 50, 12]及其组合[3, 92, 9]。
尽管这些方法将计算需求降低到与序列长度呈线性或近线性关系，但其中许多方法与标准注意力机制相比并未表现出挂钟时间的加速，且尚未获得广泛采用。一个主要原因是它们专注于减少FLOP（这可能与挂钟速度不相关），而往往忽略了内存访问（IO）带来的开销。

**图1：
左图：FlashAttention使用分块（tiling）技术来防止在（相对）缓慢的GPU HBM上实例化庞大的$N\times N$注意力矩阵（虚线框）。在外层循环（红色箭头）中，FlashAttention遍历$\mathbf{K}$和$\mathbf{V}$矩阵的块，并将它们加载到快速的片上SRAM中。
在每个块中，FlashAttention遍历$\mathbf{Q}$矩阵的块（蓝色箭头），将它们加载到SRAM中，并将注意力计算的输出写回HBM。
右图：在GPT-2上相对于PyTorch注意力实现的加速比。
FlashAttention不向HBM读写庞大的$N\times N$注意力矩阵，从而在注意力计算上实现了7.6$\times$的加速。**

在本文中，我们认为缺失的一个原则是使注意力算法具备IO感知能力[1]——即仔细考虑对不同层级的快速和慢速内存的读写操作（例如，在快速的GPU片上SRAM和相对缓慢的GPU高带宽内存或HBM之间[45]，图1左）。
在现代GPU上，计算速度已经超过了内存速度[61, 62, 63]，并且Transformer中的大多数操作都受到内存访问的瓶颈限制[43]。
当读写数据占据运行时间的很大一部分时，IO感知算法对于类似的内存受限（memory-bound）操作至关重要——例如数据库连接[71]、图像处理[70]、数值线性代数[4]以及其他领域[85, 40]。
然而，PyTorch和Tensorflow等常用的深度学习Python接口并不允许对内存访问进行细粒度控制。

我们提出了FlashAttention，这是一种新的注意力算法，它以少得多的内存访问次数计算精确注意力。
我们的主要目标是避免在HBM中读写注意力矩阵。
这要求（i）在不访问整个输入的情况下计算softmax归约（reduction）；（ii）不在反向传播中存储庞大的中间注意力矩阵。
我们应用了两种成熟的技术来应对这些挑战。
（i）我们重构了注意力计算，将输入分割成块，并对输入块进行多次遍历，从而增量地执行softmax归约（也称为分块/tiling）。（ii）我们存储前向传播中的softmax归一化因子，以便在反向传播中在片上快速重新计算注意力，这比从HBM读取中间注意力矩阵的标准方法更快。
我们在CUDA中实现了FlashAttention，以实现对内存访问的细粒度控制，并将所有注意力操作融合（fuse）到一个GPU内核（kernel）中。
尽管由于重新计算导致FLOP增加，但得益于HBM访问量的大幅减少，我们的算法不仅运行速度更快（在GPT-2上最高可达7.6倍[67]，图1右），而且比标准注意力机制使用的内存更少——与序列长度呈线性关系。

我们分析了FlashAttention的IO复杂度[1]，证明它需要$O(N^{2}d^{2}M^{-1})$次HBM访问，其中$d$是注意力头维度（head dimension），$M$是SRAM的大小，而标准注意力机制需要$\Omega(Nd+N^{2})$次。
对于$d$和$M$的典型值，与标准注意力机制相比，FlashAttention需要的HBM访问次数要少许多倍（最多减少9$\times$，如图2所示）。
此外，我们提供了一个下界，表明没有任何精确注意力算法能够在所有SRAM大小上渐进地改善HBM访问次数。

我们还表明，FlashAttention可以作为一个有用的原语（primitive），通过克服近似注意力算法在内存访问开销方面的问题，来发挥它们的潜力。
作为概念验证，我们实现了块稀疏（block-sparse）FlashAttention，这是一种稀疏注意力算法，甚至比FlashAttention快2-4$\times$，可扩展到64k的序列长度。
我们证明了块稀疏FlashAttention的IO复杂度优于FlashAttention，其优化系数与稀疏率成正比。
我们在第5节讨论了对其他操作的进一步扩展（多GPU上的注意力、核回归、块稀疏矩阵乘法）。
我们将FlashAttention开源，以便更容易地在此原语之上进行构建。（脚注：FlashAttention代码可在 https://github.com/HazyResearch/flash-attention 获取）

我们通过实证验证了FlashAttention通过对更长的上下文进行建模，加快了模型训练速度并提高了模型质量。我们还对FlashAttention和块稀疏FlashAttention与先前的注意力实现进行了运行时间和内存占用的基准测试。

- 更快的模型训练。FlashAttention在挂钟时间上更快地训练Transformer模型。我们训练BERT-large（序列长度512）的速度比MLPerf 1.1[58]中的训练速度记录快15%，训练GPT2（序列长度1K）的速度比HuggingFace[87]和Megatron-LM[77]的基线实现快3$\times$，在Long-Range Arena（序列长度1K-4K）上比基线快2.4$\times$。
- 更高质量的模型。FlashAttention将Transformer扩展到更长的序列，这提高了它们的质量并赋予了新的能力。
我们观察到在GPT-2上困惑度改善了0.7，并且在长文档分类[13]上通过对更长序列进行建模提升了6.4个百分点。
FlashAttention使得首个Transformer能够仅通过使用更长的序列长度（16K），在Path-X[80]挑战中实现优于随机猜测的性能。
块稀疏FlashAttention使Transformer能够扩展到更长的序列（64K），从而产生了首个在Path-256上实现优于随机猜测性能的模型。
- 注意力机制基准测试。在128到2K的常见序列长度范围内，FlashAttention比标准注意力实现快达3$\times$，并可扩展至64K。
在序列长度达到512之前，FlashAttention比任何现有的注意力方法都更快且更节省内存，而对于超过1K的序列长度，一些近似注意力方法（例如Linformer）开始变得更快。
另一方面，块稀疏FlashAttention比我们所知的所有现有近似注意力方法都要快。

## 2 背景

我们提供了关于现代硬件（GPU）上常见深度学习操作性能特征的一些背景知识。
我们还描述了注意力机制的标准实现。

### 2.1 硬件性能

我们在此重点关注GPU。
其他硬件加速器上的性能是相似的[48, 46]。

GPU内存层级。
GPU内存层级（图1左）包含多种不同大小和速度的内存形式，内存越小速度越快。
例如，A100 GPU拥有40-80GB的高带宽内存（HBM），带宽为1.5-2.0TB/s，其108个流式多处理器（streaming multiprocessor）每个拥有192KB的片上SRAM，估计带宽约为19TB/s[45, 44]。
片上SRAM比HBM快一个数量级，但尺寸小许多个数量级。
随着计算速度相对于内存速度变得更快[61, 62, 63]，操作越来越受到内存（HBM）访问的瓶颈限制。
因此，利用快速的SRAM变得更加重要。

执行模型。
GPU拥有海量的线程来执行一个操作（称为内核/kernel）。
每个内核将输入从HBM加载到寄存器和SRAM中，进行计算，然后将输出写回HBM。

性能特征。根据计算和内存访问的平衡，操作可以分为计算受限（compute-bound）或内存受限（memory-bound）。
这通常通过算术强度（arithmetic intensity）[85]来衡量，即每次内存访问（以字节为单位）所执行的算术操作数量。

1. 计算受限：操作所花费的时间由算术操作的数量决定，而访问HBM的时间要小得多。典型的例子是具有大内部维度的矩阵乘法，以及具有大量通道的卷积。
2. 内存受限：操作所花费的时间由内存访问的次数决定，而在计算上花费的时间要小得多。
例子包括大多数其他操作：
逐元素操作（例如，激活、dropout）和归约操作（例如，求和、softmax、批量归一化、层归一化）。

内核融合。
加速内存受限操作最常见的方法是内核融合：如果有多个操作应用于同一个输入，则可以从HBM加载一次输入，而不是为每个操作加载多次。
编译器可以自动融合许多逐元素操作[53, 65, 75]。
然而，在模型训练的背景下，中间值仍然需要写入HBM以保存供反向传播使用，这降低了朴素内核融合的有效性。

### 2.2 标准注意力实现

给定输入序列$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$，其中$N$是序列长度，
$d$是注意力头维度，我们希望计算注意力输出$\mathbf{O}\in\mathbb{R}^{N\times d}$：

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

其中$\mathrm{softmax}$按行应用。

标准注意力实现将矩阵$\mathbf{S}$和$\mathbf{P}$实例化到HBM中，这需要$O(N^{2})$的内存。
通常$N\gg d$（例如，对于GPT2，$N=1024$且$d=64$）。
我们在算法0中描述了标准注意力实现。
由于部分或大部分操作是内存受限的（例如，softmax），大量的内存访问转化为缓慢的挂钟时间。

这个问题由于应用于注意力矩阵的其他逐元素操作而加剧，例如应用于$\mathbf{S}$的掩码（masking）或应用于$\mathbf{P}$的dropout。
因此，有许多尝试融合几个逐元素操作，例如将掩码与softmax融合[77]。

在第3.2节中，我们将展示标准注意力实现执行的HBM访问次数与序列长度$N$呈二次方关系。
我们还比较了标准注意力和我们的方法（FlashAttention）的FLOP数量和HBM访问次数。

**算法0 标准注意力实现**

## 3 FlashAttention：算法、分析与扩展

我们展示了如何以更少的HBM读写次数计算精确注意力，并且无需为反向传播存储庞大的中间矩阵。
这产生了一种在内存上更高效且在挂钟时间上更快的注意力算法。
我们分析了它的IO复杂度，表明与标准注意力相比，我们的方法需要的HBM访问次数要少得多。
我们进一步表明，通过将FlashAttention扩展以处理块稀疏注意力，它可以作为一个有用的原语。

为了便于说明，我们在此重点关注前向传播；附录B包含了反向传播的细节。

### 3.1 结合分块与重计算的高效注意力算法

给定HBM中的输入$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$，我们旨在计算注意力输出$\mathbf{O}\in\mathbb{R}^{N\times d}$并将其写入HBM。
我们的目标是减少HBM访问量（降至关于$N$的次二次方）。

我们应用了两种成熟的技术（分块、重计算）来克服在次二次方HBM访问下计算精确注意力的技术挑战。
我们在算法1中对此进行了描述。
主要思想是我们将输入$\mathbf{Q},\mathbf{K},\mathbf{V}$分割成块，
将它们从缓慢的HBM加载到快速的SRAM中，然后计算关于这些块的注意力输出。
通过在将每个块的输出相加之前按正确的归一化因子对其进行缩放，我们最终得到了正确的结果。

分块（Tiling）。
我们按块计算注意力。
Softmax耦合了$\mathbf{K}$的列，
因此我们使用缩放来分解大型softmax[60, 51, 66]。

为了数值稳定性，向量$x\in\mathbb{R}^{B}$的softmax计算如下：

$$
m(x):=\max_{i}\ \ x_{i},\quad f(x):=\begin{bmatrix}e^{x_{1}-m(x)}&\ldots&e^{x_{B}-m(x)}\end{bmatrix},\quad\ell(x):=\sum_{i}f(x)_{i},\quad\mathrm{softmax}(x):=\frac{f(x)}{\ell(x)}.
$$

对于向量 $x^{(1)},x^{(2)}\in\mathbb{R}^{B}$，我们可以将拼接后的 $x=\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix}\in\mathbb{R}^{2B}$ 的 softmax 分解为：

$$
\displaystyle m(x)=m(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=\max(m(x^{(1)}),m(x^{(2)})),\quad f(x)=\begin{bmatrix}e^{m(x^{(1)})-m(x)}f(x^{(1)})&e^{m(x^{(2)})-m(x)}f(x^{(2)})\end{bmatrix},
$$
$$
\displaystyle\ell(x)=\ell(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\quad\mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}.
$$

因此，如果我们记录一些额外的统计量（$m(x),\ell(x)$），我们就可以每次计算一个块的 softmax。（脚注：这种聚合方式被称为代数聚合（algebraic aggregation）[33]。）

因此，我们将输入 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 划分为块（算法 1 第 3 行），计算 softmax 值以及额外的统计量（算法 1 第 10 行），并合并结果（算法 1 第 12 行）。

重计算（Recomputation）。
我们的目标之一是不为反向传播存储 $O(N^{2})$ 的中间值。
反向传播通常需要矩阵 $\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$ 来计算关于 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的梯度。
然而，通过存储输出 $\mathbf{O}$ 和 softmax 归一化统计量 $(m,\ell)$，我们可以在反向传播中利用 SRAM 中的 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 块轻松地重新计算注意力矩阵 $\mathbf{S}$ 和 $\mathbf{P}$。
这可以被视为一种选择性梯度检查点（selective gradient checkpointing）形式 [34, 10]。
虽然梯度检查点已被建议用于减少所需的最大内存量 [66]，但（据我们所知）所有实现都必须以速度换取内存。
相比之下，尽管 FLOPs 更多，但由于减少了 HBM 访问，我们的重计算加速了反向传播（图 2）。
完整的反向传播描述见附录 B。

实现细节：算子融合（Kernel fusion）。
分块（Tiling）使我们能够在一个 CUDA 算子（kernel）中实现我们的算法，从 HBM 加载输入，执行所有计算步骤（矩阵乘法、softmax、可选的掩码和 dropout、矩阵乘法），然后将结果写回 HBM（掩码和 dropout 见附录 B）。
这避免了在 HBM 中反复读写输入和输出。

**算法 1 FlashAttention**

我们展示了 FlashAttention 的正确性、运行时间和内存需求（证明见附录 C）。

###### 定理 1。

算法 1 返回 $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$，需要 $O(N^{2}d)$ FLOPs，并且在输入和输出之外需要 $O(N)$ 的额外内存。

### 3.2 分析：FlashAttention 的 IO 复杂度

我们分析了 FlashAttention 的 IO 复杂度，表明与标准注意力相比，HBM 访问量显著减少。
我们还提供了一个下界，证明没有任何精确注意力算法能在所有 SRAM 大小上渐进地改善 HBM 访问量。
证明见附录 C。

###### 定理 2。

设 $N$ 为序列长度，$d$ 为注意力头维度，$M$ 为 SRAM 大小，且 $d\leq M\leq Nd$。
标准注意力（算法 0）需要 $\Theta(Nd+N^{2})$ 次 HBM 访问，而 FlashAttention（算法 1）需要 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问。

对于典型的 $d$（64-128）和 $M$（约 100KB）的值，$d^{2}$ 比 $M$ 小很多倍，因此 FlashAttention 需要的 HBM 访问次数比标准实现少很多倍。
这带来了更快的执行速度和更低的内存占用，我们将在第 4.3 节中对此进行验证。

证明的主要思想是，给定大小为 $M$ 的 SRAM，我们可以加载大小各为 $\Theta(M)$ 的 $\mathbf{K},\mathbf{V}$ 块（算法 1 第 6 行）。
对于 $\mathbf{K}$ 和 $\mathbf{V}$ 的每个块，我们遍历 $\mathbf{Q}$ 的所有块（算法 1 第 8 行）来计算中间值，从而导致对 $\mathbf{Q}$ 进行 $\Theta(NdM^{-1})$ 次遍历。
每次遍历加载 $\Theta(Nd)$ 个元素，这相当于 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问。
我们同样证明了标准注意力的反向传播需要 $\Theta(Nd+N^{2})$ 次 HBM 访问，而 FlashAttention 的反向传播需要 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问（附录 B）。

我们证明了一个下界：在计算精确注意力时，对于所有的 $M$（SRAM 大小）值，无法在 HBM 访问次数上取得渐进改进。

###### 命题 3。

设 $N$ 为序列长度，$d$ 为注意力头维度，$M$ 为 SRAM 大小，且 $d\leq M\leq Nd$。
不存在一种算法能在范围 $[d,Nd]$ 内的所有 $M$ 上，以 $o(N^{2}d^{2}M^{-1})$ 次 HBM 访问来计算精确注意力。

该证明依赖于这样一个事实：对于 $M=\Theta(Nd)$，任何算法都必须执行 $\Omega(N^{2}d^{2}M^{-1})=\Omega(Nd)$ 次 HBM 访问。
这种在 $M$ 的子区间上的下界在流算法文献中很常见 [88]。
我们将证明关于 $M$ 的参数化复杂度 [27] 下界作为令人兴奋的未来工作。

我们验证了 HBM 访问次数是决定注意力运行时间的主要因素。
在图 2（左）中，我们看到尽管与标准注意力相比，FlashAttention 具有更高的 FLOP 计数（由于反向传播中的重计算），但它的 HBM 访问次数要少得多，从而导致运行时间快得多。
在图 2（中）中，我们改变了 FlashAttention 的块大小 $B_{c}$，这会导致不同数量的 HBM 访问，并测量了前向传播的运行时间。
随着块大小的增加，HBM 访问次数减少（因为我们对输入的遍历次数减少），运行时间也随之减少。
对于足够大的块大小（超过 256），运行时间随后会受到其他因素（例如算术运算）的瓶颈限制。
此外，更大的块大小将无法放入较小的 SRAM 中。

| 注意力 | 标准 | FlashAttention |
| --- | --- | --- |
| GFLOPs | 66.6 | 75.2 |
| HBM 读/写 (GB) | 40.3 | 4.4 |
| 运行时间 (ms) | 41.7 | 7.3 |

**图 2：
左图：在 A100 GPU 上，GPT-2 medium（序列长度 1024，注意力头维度 64，16 个注意力头，批大小 64）的标准注意力和 FlashAttention 的前向 + 反向运行时间。
HBM 访问是影响运行时间的主要因素。
中图：在 A100 GPU 上，FlashAttention（序列长度 1024，注意力头维度 64，16 个注意力头，批大小 64）的前向运行时间。在一定程度上，较少的 HBM 访问会导致更快的运行时间。
右图：块稀疏（block-sparse）FlashAttention 的运行时间（对于序列长度 4K）比 FlashAttention 更快，其加速比与稀疏度成正比。**

### 3.3 扩展：块稀疏 FlashAttention

我们将 FlashAttention 扩展到近似注意力：
我们提出了块稀疏（block-sparse）FlashAttention，其 IO 复杂度比 FlashAttention 小，减小的比例与稀疏度成正比。

给定输入 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 和掩码矩阵 $\tilde{\mathbf{M}}\in\{0,1\}^{N\times N}$，我们希望计算：

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}\odot\mathbb{1}_{\tilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

其中如果 $\tilde{\mathbf{M}}_{kl}=1$ 则 $(\mathbf{S}\odot\mathbb{1}_{\tilde{\mathbf{M}}})_{kl}=\mathbf{S}_{kl}$，如果 $\mathbf{M}_{kl}=0$ 则 $-\infty$。
我们要求 $\tilde{\mathbf{M}}$ 具有块状形式：对于某些块大小 $B_{r},B_{c}$，对于所有 $k,l$，$\tilde{\mathbf{M}}_{k,l}=\mathbf{M}_{ij}$ 且 $i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor$ 对应于某些 $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$。

给定预定义的块稀疏掩码 $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$，我们可以轻松地调整算法 1，使其仅计算注意力矩阵的非零块。
该算法与算法 1 相同，只是我们跳过了零块。
我们在附录 B 的算法 5 中重现了该算法描述。

我们还分析了块稀疏 FlashAttention 的 IO 复杂度。

###### 命题 4。

设 $N$ 为序列长度，$d$ 为注意力头维度，$M$ 为 SRAM 大小，且 $d\leq M\leq Nd$。
块稀疏 FlashAttention（算法 5）需要 $\Theta(Nd+N^{2}d^{2}M^{-1}s)$ 次 HBM 访问，其中 $s$ 是块稀疏掩码中非零块的比例。

我们看到，应用块稀疏性会根据稀疏度对 IO 复杂度中较大的项产生直接的改进。
对于较大的序列长度 $N$，$s$ 通常设置为 $N^{-1/2}$ [11] 或 $N^{-1}\log N$ [92, 3, 17]，从而导致 $\Theta(N\sqrt{N})$ 或 $\Theta(N\log N)$ 的 IO 复杂度。
对于下游实验，我们使用固定的蝶形稀疏模式（butterfly sparsity pattern）[17]，该模式已被证明能够近似任意稀疏性 [16]。

在图 2（右）中，我们验证了随着稀疏度的增加，块稀疏 FlashAttention 的运行时间成比例地改善。
在 LRA 基准测试中，块稀疏 FlashAttention 实现了 2.8$\times$ 的加速，同时性能与标准注意力相当（第 4 节）。

## 4 实验

我们评估了使用 FlashAttention 训练 Transformer 模型的影响。
我们验证了关于训练时间和模型准确率的两个主张，并报告了注意力的运行时间和内存基准测试。

- 训练速度。FlashAttention 比 MLPerf 1.1 [58] 的 BERT 速度记录高出 15%，并且在标准 Transformer 上，将 GPT-2 的速度比 HuggingFace [87] 提高了多达 3$\times$，比 Megatron [77] 提高了 $1.8\times$。
FlashAttention 将长程竞技场（LRA）基准测试加速了 2.4$\times$。
- 质量。FlashAttention 将 Transformer 扩展到更长的序列，从而产生更高的质量。FlashAttention 训练上下文长度为 4K 的 GPT-2 的速度比 Megatron 训练上下文长度为 1K 的 GPT-2 更快，同时困惑度（perplexity）提高了 0.7。
对更长序列进行建模在两个长文档分类任务上带来了 6.4 分的提升。
最后，FlashAttention 产生了第一个能在具有挑战性的 Path-X 任务（序列长度 16K）上实现优于随机性能的 Transformer，而块稀疏 FlashAttention 产生了据我们所知第一个能在 Path-256（序列长度 64K）上实现优于随机性能的序列模型。
- 注意力基准测试。我们基于序列长度测量了 FlashAttention 和块稀疏 FlashAttention 的运行时间和内存性能。
我们证实了 FlashAttention 的内存占用随序列长度线性扩展，并且对于常见的序列长度（高达 2K），比标准注意力快多达 3$\times$。
我们证实了块稀疏 FlashAttention 的运行时间随序列长度线性扩展，并且比所有现有的近似注意力基线都要快。

额外的实验细节见附录 E。

### 4.1 使用 FlashAttention 实现更快的模型

##### BERT。

FlashAttention 实现了据我们所知最快的单节点 BERT 训练速度。
我们在 Wikipedia 上使用 FlashAttention 训练了一个 BERT-large [22] 模型。
表 1 将我们的训练时间与 Nvidia 创下 MLPerf 1.1 [58] 训练速度记录的实现进行了比较。
我们的实现快了 15%。

**表 1：BERT-large 的训练时间，从 MLPerf 基准测试提供的相同初始化开始，在掩码语言建模（masked language modeling）上达到 72.0% 的目标准确率。
在 8$\times$A100 GPU 上 10 次运行的平均值。**

| BERT 实现 | 训练时间 (分钟) |
| --- | --- |
| Nvidia MLPerf 1.1 [58] | 20.0 $\pm$ 1.5 |
| FlashAttention (我们的) | 17.4 $\pm$ 1.4 |

##### GPT-2。

在大型 OpenWebtext 数据集 [32] 上，FlashAttention 为 GPT-2 [67] 带来了比广泛使用的 HuggingFace [87] 和 Megatron-LM [77] 实现更快的训练时间。
表 2 显示，与 Huggingface 相比，端到端加速高达 3$\times$，与 Megatron-LM 相比，加速达 1.7$\times$。
FlashAttention 实现了与其他两个实现相同的困惑度，因为我们没有改变模型定义。
附录 E 包含了整个训练过程中的验证困惑度图表，证实了 FlashAttention 与基线一样具有数值稳定性，并产生相同的训练/验证曲线。

**表 2：使用 FlashAttention 的 GPT-2 small 和 medium 与 Huggingface 实现相比实现了高达 3$\times$ 的加速，与 Megatron-LM 相比实现了高达 1.7$\times$ 的加速。
报告的训练时间是在 8$\times$A100 GPU 上测得的。**

| 模型实现 | OpenWebText (ppl) | 训练时间 (加速比) |
| --- | --- | --- |
| GPT-2 small - Huggingface [87] | 18.2 | 9.5 天 (1.0$\times$) |
| GPT-2 small - Megatron-LM [77] | 18.2 | 4.7 天 (2.0$\times$) |
| GPT-2 small - FlashAttention | 18.2 | 2.7 天 (3.5$\times$) |
| GPT-2 medium - Huggingface [87] | 14.2 | 21.0 天 (1.0$\times$) |
| GPT-2 medium - Megatron-LM [77] | 14.3 | 11.5 天 (1.8$\times$) |
| GPT-2 medium - FlashAttention | 14.3 | 6.9 天 (3.0$\times$) |

##### 长程竞技场（Long-range Arena）。

我们在长程竞技场（LRA [80]）基准测试上比较了原生 Transformer（使用标准实现或 FlashAttention）。
我们测量了所有模型的准确率、吞吐量和训练时间。
每个任务具有不同的序列长度，在 1024 到 4096 之间变化。
我们遵循 Tay et al. 2020a 和 Xiong et al. 2021 中的实现和实验设置。（脚注：众所周知，LRA 的准确率结果高度依赖于调参过程 [90]。
我们复现的基线性能优于原始比较 [80] 中报告的结果。）
表 3 显示，与标准注意力相比，FlashAttention 实现了高达 2.4$\times$ 的加速。
块稀疏 FlashAttention 比我们测试过的所有近似注意力方法都要快。

**表 3：标准注意力、FlashAttention、块稀疏 FlashAttention 以及近似注意力基线在长程竞技场（Long-Range-Arena）基准测试上的性能。**

| 模型 | ListOps | 文本 | 检索 | 图像 | Pathfinder | 平均 | 加速比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 36.0 | 63.6 | 81.6 | 42.3 | 72.7 | 59.3 | - |
| FlashAttention | 37.6 | 63.9 | 81.4 | 43.5 | 72.7 | 59.8 | 2.4$\times$ |
| Block-sparse FlashAttention | 37.0 | 63.0 | 81.3 | 43.6 | 73.3 | 59.6 | 2.8$\times$ |
| Linformer [84] | 35.6 | 55.9 | 77.7 | 37.8 | 67.6 | 54.9 | 2.5$\times$ |
| Linear Attention [50] | 38.8 | 63.2 | 80.7 | 42.6 | 72.5 | 59.6 | 2.3$\times$ |
| Performer [12] | 36.8 | 63.6 | 82.2 | 42.1 | 69.9 | 58.9 | 1.8$\times$ |
| Local Attention [80] | 36.1 | 60.2 | 76.7 | 40.6 | 66.6 | 56.0 | 1.7$\times$ |
| Reformer [51] | 36.5 | 63.8 | 78.5 | 39.6 | 69.4 | 57.6 | 1.3$\times$ |
| Smyrf [19] | 36.1 | 64.1 | 79.0 | 39.6 | 70.5 | 57.9 | 1.7$\times$ |

### 4.2 更长序列带来更好的模型

##### 长上下文语言建模。

FlashAttention的运行时间和内存效率使我们能够将GPT-2的上下文长度增加4$\times$，同时运行速度仍然快于Megatron-LM的优化实现。
表4显示，使用FlashAttention且上下文长度为4K的GPT-2，仍然比上下文长度为1K的Megatron GPT-2快30%，同时困惑度改善了0.7。

**表4：使用FlashAttention的GPT-2 small，与Megatron-LM相比上下文长度增加了4$\times$，但速度仍然快30%，同时困惑度改善了0.7。报告了在8$\times$A100 GPU上的训练时间。**

| 模型实现 | 上下文长度 | OpenWebText (ppl) | 训练时间 (加速比) |
| --- | --- | --- | --- |
| GPT-2 small - Megatron-LM | 1k | 18.2 | 4.7 天 (1.0$\times$) |
| GPT-2 small - FlashAttention | 1k | 18.2 | 2.7 天 (1.7$\times$) |
| GPT-2 small - FlashAttention | 2k | 17.6 | 3.0 天 (1.6$\times$) |
| GPT-2 small - FlashAttention | 4k | 17.5 | 3.6 天 (1.3$\times$) |

##### 长文档分类。

使用FlashAttention以更长的序列训练Transformer，提高了在MIMIC-III [47]和ECtHR [6, 7]数据集上的性能。
MIMIC-III包含重症监护室患者的出院小结，每份小结都标注了多个标签。
ECtHR包含欧洲人权法院的法律案件，每个案件都映射到据称被违反的《人权公约》条款。
这两个数据集都包含非常长的文本文档；MIMIC中的平均token数为2,395个，最长的文档包含14,562个token，而ECtHR中的平均和最长token数分别为2,197和49,392。
我们评估了增加预训练RoBERTa模型 [56] 序列长度所带来的提升（我们像Beltagy等人2020年的工作一样重复了位置嵌入）。

表6显示，在MIMIC上，16K的序列长度比512的长度性能高出4.3分，而在ECtHR上，8K的长度比512的长度高出8.5分。
这些差异可能是由于微妙的分布偏移造成的：MIMIC-III包含专业的医学文本，因此可能更容易受到文档长度分布偏移的影响，而ECtHR包含的是通用语言。

**表5：使用FlashAttention在不同序列长度下的长文档性能（micro $F_{1}$）。**

| | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-III [47] | 52.8 | 50.7 | 51.7 | 54.6 | 56.4 | 57.1 |
| ECtHR [6] | 72.2 | 74.3 | 77.1 | 78.6 | 80.7 | 79.2 |

**表6：我们报告了第一个能够在Path-X和Path-256上实现非随机性能的Transformer模型。**

| 模型 | Path-X | Path-256 |
| --- | --- | --- |
| Transformer | ✗ | ✗ |
| Linformer [84] | ✗ | ✗ |
| Linear Attention [50] | ✗ | ✗ |
| Performer [12] | ✗ | ✗ |
| Local Attention [80] | ✗ | ✗ |
| Reformer [51] | ✗ | ✗ |
| SMYRF [19] | ✗ | ✗ |
| FlashAttention | 61.4 | ✗ |
| Block-sparse FlashAttention | 56.0 | 63.1 |

##### Path-X和Path-256。

Path-X和Path-256基准测试是来自Long-Range Arena基准测试中旨在测试长上下文的挑战性任务。
该任务是分类一张128$\times$128（或256$\times$256）的黑白图像中的两点之间是否有一条路径相连，并且图像是一次一个像素地输入到Transformer中的。
在之前的工作中，所有的Transformer模型要么内存溢出，要么只达到了随机性能 [80]。
人们一直在寻找能够对这种长上下文进行建模的替代架构 [37]。
我们在此展示了Transformer模型能够解决Path-X和Path-256的第一个结果（表6）。
我们在Path-64上预训练了一个Transformer，然后通过对位置嵌入进行空间插值将其迁移到Path-X。
FlashAttention在Path-X上达到了61.4的准确率。
此外，块稀疏（block-sparse）FlashAttention使Transformer能够扩展到64K的序列长度，在Path-256上达到了63.1的准确率（脚注：Path-256需要更长的序列，但其路径相对Path-X较短，因此更容易获得更高的准确率。）。

### 4.3 注意力机制基准测试

**图3：左图：前向传播 + 反向传播的运行时间。右图：注意力内存使用情况。**

我们改变序列长度，并在配备40 GB HBM的一块A100 GPU上，在带有dropout和填充掩码（padding mask）的情况下，测量FlashAttention和块稀疏FlashAttention与各种注意力基线相比的运行时间和内存使用情况。
我们与精确注意力（exact attention）、近似注意力（approximate attention）和稀疏注意力（sparse attention）的参考实现进行了比较。
我们在正文中报告了部分基线；附录E包含更多基线和完整细节。

##### 运行时间。

图3（左）报告了FlashAttention和块稀疏FlashAttention的前向 + 反向传播运行时间（以毫秒为单位），并与精确、近似和稀疏注意力的基线进行了比较（具体数值见附录E）。
运行时间随序列长度呈二次方增长，但FlashAttention的运行速度明显快于精确注意力基线，比PyTorch实现快达3$\times$。
许多近似/稀疏注意力机制的运行时间随序列长度呈线性增长，但由于内存访问次数较少，FlashAttention在短序列上的运行速度仍然快于近似和稀疏注意力。
近似注意力的运行时间在序列长度介于512和1024之间时开始与FlashAttention产生交叉。
另一方面，在所有序列长度下，块稀疏FlashAttention都比我们所知的精确、稀疏和近似注意力的所有实现都要快。

##### 内存占用。

图3（右）显示了FlashAttention和块稀疏FlashAttention与各种精确、近似和稀疏注意力基线相比的内存占用情况。
FlashAttention和块稀疏FlashAttention具有相同的内存占用，且随序列长度呈线性增长。
FlashAttention的内存效率比精确注意力基线高出达20$\times$，并且比近似注意力基线更具内存效率。
除Linformer外，所有其他算法在A100 GPU上达到64K之前都会出现内存溢出，而FlashAttention的效率仍然比Linformer高2$\times$。

## 5 局限性与未来方向

我们讨论了我们方法的局限性和未来的方向。相关工作在附录A中给出。

编译为CUDA。我们目前构建IO感知（IO-aware）注意力实现的方法需要为每种新的注意力实现编写一个新的CUDA内核。
这要求使用比PyTorch底层得多的语言来编写注意力算法，并且需要大量的工程投入。
这些实现也可能无法在不同的GPU架构之间迁移。
这些局限性表明，需要一种支持在高级语言（例如PyTorch）中编写注意力算法，并将其编译为CUDA中IO感知实现的方法——类似于图像处理中的Halide等工作 [70]。

IO感知深度学习。
我们相信IO感知方法可以扩展到注意力机制之外。
注意力是Transformer中内存密集度最高的计算，但深度网络中的每一层都会访问GPU HBM。
我们希望我们的工作能启发更多模块的IO感知实现。
我们在附录D中讨论了这些潜在的扩展。

多GPU IO感知方法。
我们的IO感知注意力实现在单GPU上计算注意力时在常数范围内是最优的。
然而，注意力计算可以在多个GPU上并行化 [72]。
使用多个GPU为IO分析增加了一个额外的层面——需要考虑GPU之间的数据传输。
我们希望我们的工作能启发该方向的未来研究。

#### 致谢

我们的实现以Apex的FMHA代码（https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha）为起点。
我们感谢Young-Jun Ko对其FMHA实现的深入解释，以及对我们关于CUDA问题的周到解答。
我们感谢Sabri Eyuboglu、Megan Leszczynski、Laurel Orr、Yuhuai Wu、Beidi Chen和Xun Huang对论文初稿提供的建设性反馈和建议。
我们感谢Markus Rabe和Charles Staats就他们的注意力算法进行的有益讨论。

我们衷心感谢以下机构的支持：NIH（编号U54EB020405，Mobilize），NSF（编号CCF1763315，Beyond Sparsity；CCF1563078，Volume to Velocity；以及1937301，RTML）；ARL（编号W911NF-21-2-0251，Interactive Human-AI Teaming）；ONR（编号N000141712266，Unifying Weak Supervision）；ONR N00014-20-1-2480：Understanding and Applying Non-Euclidean Geometry in Machine Learning；N000142012275（NEPTUNE）；NXP、Xilinx、LETI-CEA、Intel、IBM、Microsoft、NEC、Toshiba、TSMC、ARM、Hitachi、BASF、Accenture、Ericsson、Qualcomm、Analog Devices、Google Cloud、Salesforce、Total、HAI-GCP与HAI-Azure云研究积分计划、斯坦福数据科学计划（SDSI）、国防部（DoD）通过国防科学与工程研究生奖学金（NDSEG）计划，以及斯坦福DAWN项目的成员：Facebook、Google和VMWare。尽管本文带有任何版权声明，美国政府仍有权为政府目的复制和分发重印本。本材料中表达的任何意见、发现、结论或建议均为作者个人观点，不一定反映NIH、ONR或美国政府的明示或暗示的观点、政策或认可。
Atri Rudra的研究由NSF资助（编号CCF-1763481）。
