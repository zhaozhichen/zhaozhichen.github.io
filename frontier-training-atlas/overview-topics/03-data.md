## T13 · 去重的粒度、真值与因果评估

### 去重究竟想消除什么

去重不是删除所有表达相同事实的句子。应优先减少抓取副本、模板和大量重复片段对预算的占用，以及由此放大的记忆与污染。不同视角的解释、推导步骤和自然重复可能是有价值的训练信号。

文档精确 hash 便宜而严格；规范化 hash 忽略格式差异；MinHash/LSH 对 n-gram 集合找近重复；段落/长子串方法处理局部复制；embedding 相似度更接近语义，却可能把相同主题的不同内容误删。这些不是简单从“粗”到“细”就同时提高 precision、recall 的序列，比较前先声明什么算重复。

### 真值从哪里来

构造分层审计集：不同来源、语言、长度、相似度区间和候选方法。人工标签至少区分完整副本、局部复制、同一事实的独立表述、仅主题相似；记录删除哪一个及保留依据。加入已知扰动的合成副本检验格式鲁棒性，并用出处与时间戳辅助判断。

合并多个检索器的候选池可以发现各自漏检，但在候选池内计算的 recall 不是整个语料的 recall。必须再抽取未入池文档/片段审计，或明确报告局部覆盖。precision/recall 的金标准依赖任务定义，不能把某个相似度阈值自己生成的标签当独立真值。

### 模型层面的三条证据链

第一，干净 held-out 上的 token NLL 与下游能力：固定训练 FLOPs 或 processed token，另报告 unique token 和数据补齐方式。去重后不补数据与去重后补独立数据回答不同问题。第二，记忆：给相同长度前缀，测逐字续写、最长匹配片段和提取率，按重复次数、长度和罕见性分桶。第三，污染：审计训练—评测的题面、答案、解析和改写，而不是只检查题目标题。

$$
\operatorname{ExtractionRate}
=\frac{\text{达到预设精确匹配阈值的续写数}}{\text{测试前缀数}}.
$$

该指标必须报告解码预算、前缀来源与阈值。Membership inference 还需匹配成员/非成员的时间、来源、长度和难度，并报告低 FPR 区域；单凭成员 loss 较低无法区分记忆与分布差异。

新数据加入后 benchmark 提升，至少要做等预算替换对照、重复密度匹配、污染清理前后分层、多个随机种子，以及新构造/时间后移评测。干净题子集更难时，不能把子集分数下降全归于污染移除。真正的因果结论来自受控差异，而不是某个漂亮总分。[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

## T14 · 多语言 Fertility、Token Premium 与词表预算

### 分词效率不是每个 token 的“语义单位数”

Fertility 通常指平均每词 token 数。英文可按空白词近似，中文、日文的“词”依赖分词标准，所以跨语言 fertility 不能无条件横比。可同时记录 bytes/token、characters/token，以及平行译文的 token premium：

$$
\operatorname{Premium}_{\ell}
=\frac{\sum_i T(s_i^\ell)}{\sum_i T(s_i^{\rm base})}.
$$

分母分子使用语义对齐的句对，并报告语域、译文长度和置信区间。译文不是完全等价的信息单位；Shannon 信息也不是语义含量。不同脚本、形态变化、预切分正则、byte fallback 和训练语料占比都会影响 token 数。

### 为什么大词表可能更省计算

大词表能把高频片段编码得更短。在相同原始文本上，原先需 $T_0$ 个 token，改后需 $T_1$，同样 token context 可覆盖的原始文本比例近似提高到 $T_0/T_1$。这叫编码覆盖扩大，不代表模型获得了相同比例的长程推理能力。

若用简化成本模型

$$
C(V)=N_{\rm tok}(V)(A+BV),
$$

其中 $A$ 近似为骨干每 token 成本、$BV$ 为输出词表相关成本，最优点满足

$$
-\frac{N'_{\rm tok}(V)}{N_{\rm tok}(V)}
=\frac{B}{A+BV}.
$$

左边是再增词表带来的 token 相对节省，右边是每 token 成本的相对增加。这不是完整 scaling law，只是说明边际权衡；真实目标还需计入质量、长尾样本效率、memory bandwidth 和架构变化。[Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623)

### 为什么不是越大越好

增加 $V$ 扩大 embedding/head 参数，完整 softmax 的分类成本上升；许多新词条可能很少见，削弱词根共享。有些语言的高 fertility 来自训练分布和预切分规则，单纯多加词表未必解决。固定 $V$ 时不同语言争用容量，但不等于每次一种语言改善都会使另一种语言能力下降；旧分配可能本来就不优。

如果固定原始字符串长度，更少 token 可显著节约全局 attention；但训练常把所有样本重新 pack 到固定长度，此时每块 attention 的长度不变，收益主要体现在单位原始文本所需块数。用 GQA 或 FlashAttention 也没有消除密集 attention 的二次算术项。

### 怎样做可信的 tokenizer 实验

固定数据清洗、模型非词表参数、训练预算和评测文本；明确是固定 FLOPs、固定 token 还是固定字符。跨 tokenizer 比较 BPB/相同文本 NLL 与下游任务，不直接比较 per-token PPL。分语言记录压缩率、长尾、代码/数字的可组合性、编码吞吐、实际 context 覆盖。对不同词表使用适当调优而非让其中一方继承不合适超参数。原对话里的跨语言“最新数值表”不能脱离具体 tokenizer 版本复用。

## T15 · Domain Mixture、Temperature 与 Loss Reweighting

### Domain 是可操作的数据分区，不是固定的自然类别

领域可按来源、语言、主题、格式、质量、模态划分。代码中 Python、HTML 模板与测试代码未必应被视为同一分布；网页也不是低质量的同义词。实践可以保留丰富元数据，再用较少、稳定、样本足够的采样桶控制训练。没有“所有前沿模型都用 15–30 个领域”的通用标准。

设 $K$ 个域，token 级目标为

$$
\mathcal L(\theta;\alpha)=\sum_{k=1}^{K}\alpha_k
\mathbb E_{x\sim D_k}\ell(\theta;x),\qquad
\alpha_k\ge0,\quad\sum_k\alpha_k=1.
$$

必须声明 $D_k$ 是文档分布还是 token 分布。先等概率采文档再取不同长度序列，实际 token 配比不一定等于文档抽样概率。

### Temperature sampling 是受限的一条曲线

自然比例 $p_k>0$，一种惯例为

$$
\alpha_k(T)=\frac{p_k^{1/T}}{\sum_jp_j^{1/T}},\quad T>0.
$$

$T=1$ 保持原比例，$T\to\infty$ 趋向均匀，$T\to0^+$ 集中到最大域。有些论文把指数直接叫 temperature/alpha，必须看公式。对固定正 $T$，大小顺序保持不变，所以 99:1 可以平滑到 50:50，却不能让小域变成 70%。完整混合有 $K-1$ 个自由度，温度只有一个；它也不能给原先概率为零的数据源创造支持。

### 改采样和改 loss 哪里等价

若按分布 $q_k$ 抽域，再用 importance weight $\alpha_k/q_k$，则

$$
\mathbb E_{k\sim q,x\sim D_k}
\left[\frac{\alpha_k}{q_k}\nabla\ell(\theta;x)\right]
=\sum_k\alpha_k\mathbb E_{D_k}\nabla\ell.
$$

这是同一 $\theta$ 下的**期望梯度**等价，要求目标有支持的位置 $q_k>0$，并且归一化一致。两种 estimator 方差、计算成本、重复次数不同；Adam 的二阶矩、clipping 和有限步非线性使最终轨迹不等价。不能说实际训练一定、百分之百只用采样而不用 loss 权重，混合模态与多目标训练经常需要两者结合。

如果直接把某域 loss 乘 2，却不说明原始采样比例与整体归一化，也不等价于把其概率“直接乘 2”后什么都不改。采样改变还会影响 optimizer steps 所见数据以及小域 epoch 数。有效上采样次数约 $D_{\rm total}\alpha_k/D_{k,\rm unique}$，是设计重复预算时必看的量。

## T16 · DoReMi：Excess Loss 不是边际收益

### 从最坏域目标到参考模型

直接最小化最大域 loss 可能追逐本来就难预测的噪声域。Excess loss 改看当前模型相对参考模型的差距：

$$
\min_\theta\max_{\alpha\in\Delta^{K-1}}
\sum_k\alpha_k\big[L_k(\theta)-L_k(\theta_{\rm ref})\big].
$$

参考模型是按基准混合训练的有限模型，不是熵率 oracle，也不是每域单独训练得到的理论上限。差值仍是 nats/token 或 bits/token；减去基线可以抵消部分固有难度，但不自动实现跨 tokenizer、噪声方差和业务价值的完全校准。

DoReMi 使用小参考模型、小代理模型与大模型三个角色：先训练参考，冻结；代理上优化域权重；最后用平均权重采样训练大模型。算法 1 在训练 minibatch 上计算逐 token 的正部分 excess loss，而不是每步必须跑一套独立验证集：

$$
\widehat\Delta_k=
\frac{\sum_{x\in B\cap D_k}\sum_t
\max\{\ell_{\theta,t}(x)-\ell_{{\rm ref},t}(x),0\}}
{\sum_{x\in B\cap D_k}|x|}.
$$

随后做乘性更新、归一化与平滑：

$$
\widetilde\alpha_k=\alpha_k e^{\eta_\alpha\widehat\Delta_k},
\qquad
\alpha'_k=(1-c)\frac{\widetilde\alpha_k}{\sum_j\widetilde\alpha_j}
+\frac cK.
$$

代理模型向减小加权 loss 的方向更新，权重向增大当前困难域贡献的方向更新。最终输出训练轨迹的平均权重，不是只取最后一步。[DoReMi 算法 1](https://arxiv.org/html/2305.10429v3)

### 为什么“数学代码一定被上调”不是推论

若数学 excess 为 0.50、网页为 0.02，未平滑更新的权重比会乘上 $e^{0.48\eta_\alpha}$。这说明算法响应当前相对差距；不能反过来规定所有语料里数学必为 0.50。参考模型能力、已有暴露、数据噪声、域划分与学习阶段都会改变排序。

Regret 大也不代表增加一单位数据的边际收益大。例如域 A 与参考相差 1，但当前模型容量不足，短期再给数据也不改善；域 B 只相差 0.2，却很容易学。最坏差距与收益导数是不同统计量。原讨论中把二者称为“同一现象”过强。

### 正迁移如何证明

降低一个域的采样权重，同时它的 held-out loss 下降，在 DoReMi 的实验中确实可能出现；合理解释之一是混合训练改善共享表示。但这不是代码对所有其他域都“无损向下兼容”的定律。

做单域权重扰动必须同时说明从哪个域减去概率；测所有域的变化与独立任务，而不只看被上调域。保持处理 token、重复度、模型大小和训练 schedule；用多个规模检验迁移方向。需要显式业务偏好时，可以修改最坏域目标或增加约束，但那是新的问题定义，不是原算法已自动优化你的业务效用。

## T17 · 数据配比的双层优化与 Proxy 外推

### 自由度少，为什么搜索仍然贵

配比只有 $K-1$ 个自由度，但每次评估一个点都可能要训练模型：

$$
\min_{\alpha\in\Delta^{K-1}}
L_{\rm eval}(\theta_S(\alpha)).
$$

$\theta_S$ 是经过有限 $S$ 步训练的结果，不一定是训练目标的全局最优。真正贵的是这个内部映射，而不是存储几十个 alpha。若用理想化最优点 $\theta^*(\alpha)$，在可微、局部最优且 Hessian 可逆等条件下：

$$
\frac{\partial\theta^*}{\partial\alpha}
=-\left(\nabla_{\theta\theta}^2L_{\rm train}\right)^{-1}
\nabla_{\alpha\theta}^2L_{\rm train}.
$$

实际可以用 Hessian-vector product 与线性求解近似，不必显式存储整个逆矩阵；但长训练、非凸性和高维仍然很贵。单步 lookahead 只估局部影响，无法保证预测长期表示学习。

### 三类实用路线及其边界

一是 DoReMi 一类在线代理博弈，避免完整反传训练轨迹，但优化的是代理目标。二是随机/贝叶斯搜索，把完整训练当黑盒，用历史试验选择下一个配比；样本效率受维度与噪声限制。三是拟合 mixture/scaling surrogate，在拟合曲面上优化，再用真实训练验证。SLSQP、L-BFGS 是数值求解器，不会把非凸曲面自动变成凸问题，更不保证全局最低点。

如果用 softmax 参数化 alpha，会自动满足单纯形约束，但接近零权重的域可能难以恢复；直接投影或镜像更新具有不同边界行为。需要为重要域设下界时，应显式加入约束。

### Proxy 不是“小模型排个名直接照搬”

至少在多个规模、token 数和配比上布点，检查排序是否稳定。某策略可能小模型无效、大模型有效；长上下文数据在短上下文 proxy 上甚至测不到价值。分别区分模型规模 proxy、短训练 proxy 和廉价评估 proxy，它们有不同偏差。

可拟合相对 baseline 的 $\Delta L(N,D,\alpha)$，比拟合两个大数再相减更直接；保留中间规模作为外推检查，报告置信区间。适当复用随机种子和数据前缀降低比较噪声，但不能只用单 seed 证明稳定。

最终目标可写 $U(\alpha)=\sum_jw_jG_j(\alpha)$，但在单纯形上应考虑把质量从域 $j$ 转到域 $i$ 的方向导数 $\partial_iU-\partial_jU$。内点最优时这些边际量相等，不是每一个偏导都为零；边界上需要 KKT 不等式。最大化均分、最小化最坏域和满足安全约束也不是同一个目标。

## T18 · 计算最优、Overtraining 与部署生命周期

### 固定计算怎样分给模型和 token

简化 dense 模型近似：

$$
C\simeq6ND,\qquad
L(N,D)=L_\infty+AN^{-a}+BD^{-b},
$$

其中 $N$ 为参数量，$D$ 为 processed token。代入 $D=C/(6N)$，对 $N$ 求导：

$$
aAN^{-a}=bBD^{-b},
\qquad
N^*\propto C^{b/(a+b)},\quad
D^*\propto C^{a/(a+b)}.
$$

它说明最优点平衡的是两类边际 loss 收益，不是参数与 token 的数值相等。Chinchilla 给出在其实验范围内更接近模型和数据同步扩展的结论；常见“约 20 token/parameter”是经验位置，不能跨 tokenizer、数据质量、MoE、蒸馏和目标直接当常数。[Chinchilla](https://arxiv.org/abs/2203.15556)

早期 Kaplan 与后续 Chinchilla 的指数差异涉及拟合方式、训练范围与优化设置。真正的规划应以文献作先验，再用自己的架构、数据与实现拟合，尤其不要把较小模型的短训练曲线无误差条外推多个数量级。

### 为什么小模型常训练远超计算最优 token 数

“Overtrained” 常表示相对于单次训练 compute-optimal 位置用了更多 token，不等于统计过拟合。固定小模型继续看新数据，held-out loss 仍可下降，只是新增训练 FLOPs 的收益变小。

$$
C_{\rm life}=C_{\rm train}(N,D)
+Q\,C_{\rm infer}(N,\text{context},\text{output},\text{hardware}).
$$

若未来请求量 $Q$ 很大，更小但训练更久的模型可以节省总成本；内存、延迟与部署设备也可能硬性限制 $N$。应比较相同最终质量下的生命周期成本，而不只看训练那一次的最优点。

### 重复、数据池和过拟合

Processed token、unique token、可用数据池大小分别记录。某模型家族声明一个大数据池，不代表每个尺寸都遍历相同 token 数；蒸馏数据还带入 teacher 的额外信息，不能与普通网页同质计数。

重复有限语料时，训练 loss 继续下降而独立分布 loss 上升，才是典型过拟合证据。重复多少 epoch 可接受取决于容量、噪声、数据增强和目标，不存在“第四轮前一定安全”。CPT 在小专业数据上尤其需要 held-out、replay 和早停；大规模训练时没过拟合，不保证后续小数据适配不会过拟合。[Data-constrained scaling](https://arxiv.org/abs/2305.16264)
