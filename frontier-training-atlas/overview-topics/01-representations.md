## T01 · Encoder、Decoder 与统一序列建模

### 先把对象分清楚

Transformer 是一类计算模块，不等于某一种预训练目标。2017 年的原始模型是 encoder–decoder：编码器读取源句，解码器读取已知的目标前缀，并通过 cross-attention 查询源句。BERT 是 encoder-only；GPT 是 decoder-only，不是相反。原论文两端各 6 层只是当时的配置，不是架构定义。

以“苹果推出新手机”翻译为 “Apple launched a new phone”为例，编码器计算“苹果”的表示时可以同时利用后面的“手机”，消解水果与公司的歧义。解码器预测 “launched” 时，目标侧只能看到起始符和 “Apple”，但源侧整句已经可见。这里的“因果”是预测条件中的先后约束，不是对自然界因果关系的证明。

| 架构 | 可见性 | 典型任务 | 关键区别 |
| --- | --- | --- | --- |
| Encoder-only | 非 padding 位置通常双向可见 | 遮盖预测、分类、表征学习 | 不直接规定自回归生成接口 |
| Encoder–decoder | 源侧双向；目标侧因果并读取源侧 K/V | 翻译、摘要、去噪重建 | 源与目标使用不同堆叠 |
| Decoder-only | 通常使用因果 self-attention | 序列延续、条件文本生成 | 指令、示例和答案放在同一序列 |

### 从概率分解理解架构

对于源句 $x$、目标句 $y$，两种生成架构都可以表达

$$
p_\theta(y\mid x)=\prod_{t=1}^{|y|}p_\theta(y_t\mid x,y_{<t}).
$$

Encoder–decoder 的 cross-attention 使用目标隐状态产生 query，源表示产生 key/value：

$$
\operatorname{CrossAttn}(H_y,H_x)
=\operatorname{softmax}\!\left(
\frac{(H_yW_Q)(H_xW_K)^\top}{\sqrt{d_h}}
\right)H_xW_V.
$$

Decoder-only 则把条件 $x$ 放在目标前缀里，通过同一个注意力模块读取。训练时目标已在数据中，因此所有目标位置可以并行计算；只有生成未知目标时才必须逐步采样。“encoder 并行、decoder 串行”不能用来描述 teacher-forced 训练。

### 为什么生成式 LLM 常用 decoder-only

纯文本可以直接转成 next-token 样本，不必先构造源—目标对；同一接口能容纳 few-shot 示例、聊天、代码和工具轨迹。历史前缀不改变时，多轮生成也容易复用 KV cache。统一堆叠减少了源/目标两端参数容量分配的设计选择，并形成了成熟的训练与 serving 生态。

这些是经验与工程理由，不是“encoder–decoder 在数学上不能推理”。去噪训练的监督 token 数取决于破坏方式，不能一概说只有 15%–25%；没有直接 target loss 的 encoder 也会通过 cross-attention 收到梯度。完整 encoder–decoder 仍可适合有明确输入输出边界的任务。比较必须固定总计算、数据和推理约束，不能从监督位置多就直接推出能力更强。

参考：[原始 Transformer](https://arxiv.org/abs/1706.03762)、[BERT](https://arxiv.org/abs/1810.04805)、[T5](https://arxiv.org/abs/1910.10683)。

## T02 · Mini-batch、Teacher Forcing、Packing 与两类 Mask

### 一次更新到底发生了什么

一个样本可以包含许多预测位置，而不是一个 token。一次 micro-batch forward 并行计算这些位置的 logits；将有效位置的 loss 聚合后调用一次 backward。梯度累积时重复若干次 forward/backward，最后才做一次 optimizer step。因此“每个数据点都单独反传并更新模型”不是通常的大模型训练方式。

设输入为 [BOS, A, B, C]，目标为 [A, B, C, EOS]。A 所在位置的隐状态看见 BOS、A，用来预测 B。不能允许它看到 B，否则可以抄答案。Teacher forcing 指使用真实历史 A，而不是先由模型生成 A；真实历史是条件，不意味着允许看见待预测目标。

### Attention mask 与 loss mask 控制不同的东西

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad
M_{ij}=
\begin{cases}
0,&j\le i\ \text{且该连接被允许},\\
-\infty,&\text{否则}.
\end{cases}
$$

Attention mask 决定信息沿哪些边流动；loss mask $m_t$ 决定哪些预测位置受监督：

$$
\mathcal L=-\frac{\sum_t m_t\log p_\theta(x_t\mid x_{<t})}{\sum_t m_t}.
$$

SFT 常把提示位置设为 0、assistant 回答位置设为 1。**提示不计 loss，不等于提示隐状态不参与反传**：回答读取提示的 K/V，梯度仍可流过它们。冻结参数、detach 激活和屏蔽 target 是三件不同的事。

Packing 把短样本拼入固定长度块以减少 padding。有两种不同目标：允许跨文档关注的连续文本流；或用 block-diagonal causal mask 隔离独立样本。插入 EOS 不会自动隔离；重置 position ID 也不等于屏蔽 attention。必须明确边界处“预测下一篇第一 token”是否被训练，以及 position/mask/label shift 是否采用一致约定。

### 变长样本下的归一化

各 micro-batch 有效 target 数不同，不能简单平均各自的 mean loss。token-weighted 梯度是

$$
g=\frac{1}{M}\sum_{r=1}^{W}\sum_j\nabla_\theta S_{rj},
\quad M=\sum_{r,j}m_{rj},
$$

其中 $S_{rj}$ 是 loss sum，$m_{rj}$ 是 target 数，$W$ 是数据并行副本数。若 DDP 默认对梯度取平均，每个 rank 反传 $WS_{rj}/M$，平均后恰好得到上式。这假定已知完整 accumulation window 的全局 $M$；实际代码需先计数或在累积后正确缩放。

例如两段有 2、6 个 target，loss sum 为 4、24，整体均值是 $28/8=3.5$，平均局部均值却是 $(2+4)/2=3$。序列等权不是一定错误，但必须是有意选择。

检查实现时做微型测试：改变未来 token 不影响此前 logits；改变被隔离文档不影响另一文档；单卡大 batch 与多卡加累积的梯度在容差内一致。还要测试全零 mask、padding 与 EOS 共用 ID、多轮 assistant target 和最后一个有效停止标记。

## T03 · 多头输出、O Projection 与 GQA 的代价

### “拼接”和“相加”为什么都可能看到

$$
H_i=\operatorname{softmax}(Q_iK_i^\top/\sqrt{d_h})V_i,
\qquad H=\operatorname{Concat}(H_1,\ldots,H_h)W_O.
$$

若把 $W_O$ 按行分块，同一个计算也能写成

$$
H=\sum_{i=1}^{h}H_iW_{O,i}.
$$

因此，未经投影的 head 直接相加，与各 head 分别投影到 residual space 后相加不能混淆。后者与 concat 后投影完全等价。$W_O$ 学习将读取结果写回残差通道，不是再次计算 attention。

后面的 MLP 能混合通道，但中间有残差相加、归一化和非线性，不能直接删除 $W_O$ 再说“MLP 会代替它”。一个 head 的读取—写回映射包含 $W_{V,i}W_{O,i}$；这种回路分解比人为给每个 head 贴“语法专家”标签更可靠。

### GQA 共享的是 K/V，不是查询方式

取 $H_Q=32$、$H_{KV}=8$、$d_h=128$。每四个 query head 共用一组 K/V，但 $Q_i$ 不同，所以 attention 权重仍可不同。MHA 是 $H_{KV}=H_Q$；MQA 是 $H_{KV}=1$。

若 batch 为 $B$、缓存长度为 $L$、层数为 $n_\ell$、每元素 $s$ 字节，不计额外布局开销：

$$
\operatorname{KVBytes}=2BLn_\ell H_{KV}d_hs.
$$

该 GQA 的 KV cache 是 MHA 的四分之一。$W_K,W_V$ 输出宽度也从 4096 降为 1024。但 QK/AV 仍为每个 query head 做注意力，整个 attention FLOPs 或端到端延迟不会必然降低四倍；decode 带宽瓶颈、batch、kernel 和 TP 布局都影响收益。

### 有没有量化的能力损失证据

原始 GQA 论文把 T5-XXL 的 MHA checkpoint 转成 GQA-8，并追加原预训练步数的 5% 做 uptraining。表 1 的多任务平均分为 MHA 47.2、GQA 47.1、MQA 46.6；推理时间分别为 1.51、0.28、0.24。它支持“该实验里质量接近 MHA、速度接近 MQA”，不是所有任务恒定损失 0.1 分。作者使用 TPUv4，各配置单独优化并行与可容纳 batch，不能把这些时间当作通用硬件加速比。[GQA 表 1](https://aclanthology.org/2023.emnlp-main.298.pdf)

这不是严格同训练预算的从零架构比较。自己的消融应同时报告参数/FLOPs匹配训练、真实 serving，以及长文多实体检索、位置外推、变量绑定等分项。共享减少自由度，但泛化也受正则化和优化影响，不能推出每项指标必然下降。4:1 或 8:1 “安全比例”只能作为实验起点。

## T04 · Token ID、Embedding、词表与 Softmax Bottleneck

### 三个数量不是一回事

序列长度 $L$ 是此次输入的位置数；词表大小 $V$ 是可选符号数；隐层宽度 $d$ 是每个位置的实数维度。查表矩阵 $E\in\mathbb R^{V\times d}$，batch 激活 $X\in\mathbb R^{B\times L\times d}$。例如 $V=77{,}850$、$d=4096$，不意味着输入有 77,850 个 token，也不意味着只能表示 4,096 个词。

静态 embedding 是 $e_t=E[x_t]$。Transformer 输出依赖上下文的 $h_t$，最后

$$
z_t=W_Uh_t+b,\quad W_U\in\mathbb R^{V\times d},
\qquad p_t(i)=\frac{\exp z_{t,i}}{\sum_j\exp z_{t,j}}.
$$

再由解码规则选 token。输出层使用最终 hidden state，不是初始 embedding。交叉熵通常从 logits 做稳定的 log-softmax，无须先采样离散 token。

权重绑定令 $W_U=E$；不绑定则两者分开学习。上述一个矩阵有 $318{,}873{,}600$ 参数，BF16 权重约 638 MB（十进制），不含梯度和优化器状态。绑定节省参数，却也意味着输入与输出梯度共同更新一张表。

### 高维空间不是一格一个词的停车场

大量向量可以处于同一有限维空间，词表增长不自动让所有向量彼此接近。随机单位向量的成对余弦典型尺度约 $1/\sqrt d$；候选变多时，最近邻最大相似度的常见近似是 $\sqrt{2\log V/d}$，依赖随机、各向同性等假设，不是训练后 embedding 的定律。

长尾 token 在 $T$ 个训练 token 中出现约 $Tp_i$ 次，输入表示与正类输出证据可能不足。完整 softmax 的非目标词也会收到梯度，所以“没当目标就完全没梯度”不准确。大词表可能把可共享词根拆成许多低频整体词；小词表则增加序列长度。这是计算与统计共享的联合选择。

### 真正的秩瓶颈在哪里

在许多上下文上堆叠 $H\in\mathbb R^{m\times d}$，无偏置 logits $Z=HW_U^\top$ 满足 $\operatorname{rank}(Z)\le d$。但 softmax 非线性，不能推断概率矩阵也有相同秩上限。

$$
\log P=Z-a\mathbf1^\top,\qquad
a_i=\log\sum_j e^{Z_{ij}},
\qquad \operatorname{rank}(\log P)\le d+1.
$$

要在每行可加常数的等价类里，检查目标 log-probability 矩阵能否由低秩 logits 表示。瓶颈约束的是跨上下文分布族，不是单个向量能否有许多非零概率。增大宽度、非线性输出头或 mixture of softmax 是不同解决方向。[Softmax Bottleneck](https://arxiv.org/abs/1711.03953)

## T05 · 句向量、对比学习与 Gemini Embedding

### 从位置表示到整句表示

句向量不是为每句话新增词表 ID，而是对 $H=[h_1,\ldots,h_L]$ 池化。常用 masked mean、双向模型 CLS、因果模型最后一个有效位置：

$$
v_{\rm mean}=\frac{\sum_t m_th_t}{\sum_t m_t},
\qquad v_{\rm last}=h_{L_{\rm valid}}.
$$

因果模型只有最后位置有机会读到全句；“看得到”不保证保留全部语义。平均池化也不保证否定词、数字不被稀释。Pooling 必须与模型的训练协议一致，不能任意替换。

生成 hidden state 服务于 next-token prediction，不自动具有良好的检索几何。对比学习用 query、正例和负例训练：

$$
\mathcal L_q=-\log
\frac{\exp(s(v_q,v_+)/\tau)}
{\exp(s(v_q,v_+)/\tau)+\sum_j\exp(s(v_q,v_j^-)/\tau)}.
$$

梯度提高正例相似度并压低负例。$\tau$ 控制锐度；hard negatives 提供细粒度区分，但把相关文档误标为负例会伤害召回。使用 in-batch negatives 尤其要处理重复和多正例。

### 相似度由任务定义

“如何退货”和“退货规则全文”不是同义句，却是好的检索对；“天气很好”和“天气不好”词汇相近而情感相反。任务提示让同一骨干区分检索 query、document、聚类、分类。归一化后的点积等于余弦，方便 ANN 索引；质量仍要测 recall@k、排序和领域泛化，而不是只看二维可视化。

MRL 在不同前缀维度上同时训练：

$$
\mathcal L_{\rm MRL}=\sum_{m\in\mathcal M}w_m
\mathcal L_{\rm contrastive}(v_{q,1:m},v_{d,1:m}).
$$

因此截短维度有训练支持。任意向量截前 256 维没有同样保证；“节约三分之二内存保留 98% 准确率”只能是具体实验。

### 对 Gemini 的公开实现能确定什么

2025 年 Gemini Embedding 报告明确描述：从 Gemini 初始化、双向 attention、mean pooling、任务提示、对比训练与 MRL，并结合数据过滤、合成数据和 hard-negative mining。它不是“可能 EOS，也可能 mean”的未知二选一。Gecko、text-embedding-004 和后续 Gemini Embedding 不是可互换的模型名；也不能把 embedding 模型的双向 mask 反推成生成式 Gemini 的训练结构。[Gemini Embedding §3](https://arxiv.org/html/2503.07891v1#S3)

## T06 · 多模态 Token：离散码本与连续 Patch

### Token 不一定是整数 ID

Transformer 接收向量序列。文本常切成离散 ID 再查表；图像可以切 patch 后直接投影为连续向量；音频可以由频谱编码器输出连续帧表示，也可用神经 codec 产生离散码。广义 token 是序列单元，不等于共享文本 tokenizer 或统一 softmax。

$224\times224$ RGB 图像用 $14\times14$ patch，得到 256 个位置，每块原始维度为 588：

$$
X_{\rm patch}\in\mathbb R^{256\times588},
\qquad W_P\in\mathbb R^{588\times d}.
$$

14、256、588、$d$ 与“8192 个视觉码本条目”是不同概念。VQ 模型先编码为 latent grid，再对每个 $z$ 找最近 code：

$$
k^*=\arg\min_{k\in\{1,\ldots,V_{\rm img}\}}\|z-e_k\|_2^2.
$$

该 latent 单元的感受野由编码器决定，不一定是独立的 $14\times14$ 像素块。8192 是可选 code 数，不是 patch 数或向量宽度。

### 要在词表为图像、音频预留位置吗

可以使用互不重叠的 ID 区间：

$$
V_{\rm total}=V_{\rm text}+V_{\rm image}+V_{\rm audio}+V_{\rm control}.
$$

图像 code 加 offset 后查统一表，生成后去掉 offset 交给图像 decoder。也可以保留模态独立 embedding/head，再投影到共同宽度。音频 RVQ 每帧可有多个码本索引，通过求和、交错或多流预测组织。没有所有多模态模型通用的 ID 分区。

统一词表可能使概率归一化跨模态竞争；独立 head 或运行时限制允许 token 集合，则是不同目标与解码设计。模态样本数、时长、分辨率和 loss 权重共同决定梯度贡献。“图像熵一定高于文本”没有脱离 tokenizer、码率和条件的意义。

### 如何衔接训练

可在文本骨干上加媒体编码器，先对齐投影再部分/全部解冻；也可从较早阶段联合多模态训练。它们对遗忘、样本和计算要求不同。离散生成可用 next-token CE；连续表征理解把媒体作为条件；媒体生成还可使用扩散或 flow matching。**使用 Transformer 不等于使用 next-token prediction。** 未公开商业模型的 codec、码本和生成器细节不能据此推定。

参考：[ViT](https://arxiv.org/abs/2010.11929)、[VQ-VAE](https://arxiv.org/abs/1711.00937)、[Chameleon](https://arxiv.org/abs/2405.09818)、[AudioLM](https://arxiv.org/abs/2209.03143)。
