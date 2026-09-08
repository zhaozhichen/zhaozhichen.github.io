## T43 · Thinking Effort：控制信号、条件策略与硬预算

### 不是 hardcoded rule 与 system prompt 二选一

可以分三层理解：输入里表达期望计算强度的控制信号；post-training 学到的条件化策略；runtime 对 token、时间、工具次数等设置的硬限制。概念模型是

$$
\pi_\theta(z,y\mid x,c),
\qquad c\in\{\text{low},\text{medium},\text{high}\},
$$

其中 $z$ 表示求解轨迹，$y$ 为最终答案。Effort 改变求解策略分布；temperature 通过 $p_i\propto e^{l_i/T}$ 改变采样随机性，二者不是同一旋钮。

公开例子是 gpt-oss：官方 Harmony 文档在 system message 中使用普通文本 “Reasoning: high”，支持 low/medium/high。字符串能够生效依赖模型学过该协议，不是这些词有特殊魔法；也不能据此断言其他闭源模型内部一定用相同字符串。[OpenAI Harmony](https://developers.openai.com/cookbook/articles/openai-harmony)

### 三档不意味着分别训练三个模型

同一参数集可用不同 $c$ 联合训练，一个 batch 混合不同 effort。SFT 数据可包含 $(x,c,z,y)$；RL 的一种概念目标是

$$
J(\theta)=\mathbb E_{x,c,z,y}
[R(x,z,y)-\lambda_c C(z,y)],
\quad \lambda_{\rm low}>\lambda_{\rm high}.
$$

Low 的计算价格高，只有值得时才继续；high 允许更多检查、搜索和尝试。这是可行的数学建模，不是已公开的逐 batch 商业配方。同一问题配多个预算轨迹能提供条件对照，但不能假定每家实际如此构造。

如果先只训 low，再只训 medium，最后只训 high，后阶段可能干扰前面的模式；replay、多条件混合或阶段化训练可以缓解。重点是学到一族共享参数策略，不是产品三个档位必对应三套 checkpoint。

### 硬预算仍然必要

模型可以自己决定结束，但生产系统还需最大输出、工具循环、wall-clock 和成本上限。硬 cap 不是要求模型一定用满；effort 也不是保证精确产生某个 token 数。简单问题各档可能差异很小，困难问题可能差异大。

Tool agent 的计算包括规划、搜索、执行和检查，不只是 reasoning 文本长度。要测试 effort 是否真正可控，按难度分桶测质量—成本曲线、停止/截断比例、工具次数与尾延迟。较长轨迹不必更深，较高 effort 也不保证更准确；应找自己的质量—成本前沿，而不是把最高档作为无条件最优。

## T44 · Linear Attention、Delta Rule 与 Mamba

### 从显式历史读取到压缩状态

全局 softmax attention 保存历史 K/V，让每个 query 读取过去位置。其 prefill 交互计算随长度近似二次增长，decode 每新 token 读取的历史随长度增长。Linear attention 用特征映射后的核替代 softmax 核，将求和结合起来：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),
\qquad
o_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
$$

$S_t$ 是历史键值压缩，$z_t$ 是归一化状态，不能在声称 normalized attention 时漏掉分母。为避免分母问题通常对特征映射与数值稳定性有要求。状态大小由特征维度决定，在固定维度下不随已读长度增长。

### 压缩为什么会有干扰

外积求和把许多键值叠加，相似 key 可能相互干扰。以 $M_t\in\mathbb R^{d_v\times d_k}$ 存储关联，delta rule 可以写成

$$
M_t=M_{t-1}+\beta_t(v_t-M_{t-1}k_t)k_t^\top.
$$

它是对局部平方误差 $\tfrac12\|M k_t-v_t\|^2$ 的一步梯度更新：先看已有记忆预测什么，再写入残差，而不是反复累加同样信息。若有遗忘因子或归一化，稳定性条件也随之改变。这里更新的是状态矩阵，不意味着每个 token 都在重训整个基础模型。

### SSM 与 Mamba 的选择性

简单状态空间模型写为 $h_t=\bar A h_{t-1}+\bar Bx_t$、$y_t=Ch_t$。Mamba 引入输入相关的选择机制，使状态的写入、传播和读取随内容变化；Mamba-2 研究结构化状态空间与 attention 的联系及高效算法。它们不是把 Transformer 的 softmax 随便删掉。[Mamba](https://arxiv.org/abs/2312.00752)、[Mamba-2](https://arxiv.org/abs/2405.21060)

递推形式适合流式推理，训练可以用 scan/chunk 等并行方法，但不能声称没有顺序依赖。与 dense attention 比较，需要同时看吞吐、精确检索、长依赖、状态精度和训练规模。混合架构保留少量全局 attention 可以改善某些精确读取能力；只要仍有随全长计算的全局层，其总体渐近项并不会自动变成纯线性。

## T45 · TTT、Titans 与 Nested Learning

### 把 hidden state 变成一个能更新的小模型

普通 RNN 用固定形式更新向量状态；TTT 用小学习器的参数作状态。令 fast model 为 $f_{W_t}$，当前 token 提供学习信号：

$$
\ell_t(W)=\frac12\|f_W(k_t)-v_t\|^2,\qquad
W_t=W_{t-1}-\eta_t\nabla_W\ell_t(W_{t-1}),
\qquad o_t=f_{W_t}(q_t).
$$

这是解释性形式，不替代具体论文的多视图设计与更新时序。外层训练学习如何构造 key/value/query 及如何利用内层更新；测试时变化的是 fast state。线性 fast learner 与非线性 MLP fast learner 的表达能力和成本不同。[Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620)

### Titans 为什么强调 surprise、momentum 与 forgetting

只持续写入会让记忆饱和；新信息如果已经能预测，也不必同等强度写入。Titans 用记忆损失梯度作为 surprise 的一种可计算信号，并考虑动量和遗忘。示意：

$$
u_t=\eta_tu_{t-1}-\theta_t\nabla_M\ell_t(M_{t-1}),
\qquad M_t=(1-\alpha_t)M_{t-1}+u_t.
$$

局部 attention 保留高精度短期读取，神经记忆压缩更远历史。Surprise 不等于“语义上重要”：随机噪声也可能很意外，重要但常见的规则反而不意外，因此写入机制必须通过外层任务训练与测试验证。[Titans](https://arxiv.org/abs/2501.00663)

### Nested Learning 提供的视角

它把架构与优化器都看作不同频率、不同上下文流上的学习过程：激活、记忆、momentum、参数都可以看成状态更新。Hope 是结合 self-modifying memory 与多频率 memory blocks 的研究原型。这个视角有助于设计 fast adaptation 与 slow consolidation，但公开研究结果不等于已经解决任意长持续学习或完成所有前沿模型部署。[Nested Learning / Hope](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)

### 无限流处理不等于无限无损记忆

有限精度、固定大小状态可以一直更新，却不能无损存下任意长历史的所有独立信息。压缩越强，越需选择忘什么；即使 full attention 保留 token，也不保证模型可靠取出每个事实。测试要分辨精确 associative recall、语义摘要、分布漂移和累积干扰。

Fast state 还需要生命周期定义：单样本何时 reset、跨用户是否隔离、何时持久化、是否允许回滚；outer gradient 是否穿过内循环、截断多长决定训练成本与偏差。不能把“测试时权重更新”直接等同于“把用户所有消息永久写入 foundation weights”。

## T46 · Continual Learning：更新对象与时间尺度

### 先问到底什么在学习

持续学习可以发生在基础参数、adapter、fast neural memory、外部知识库、上下文 playbook、agent code、数据采样或 evaluator 上。它们都会改变系统行为，却有不同的泛化、遗忘、隐私与验证边界：

$$
\mathcal S_t=(\theta_t,M_t,C_t,A_t,J_t,D_t),
\qquad
\mathcal S_{t+1}=F(\mathcal S_t,\text{experience}_t).
$$

$\theta$ 为慢参数，$M$ 为记忆，$C$ 为上下文，$A$ 为 agent 程序，$J$ 为评价器，$D$ 为经验数据。描述系统时应指出哪些分量被更新，不能只说“模型变聪明了”。

### Fast adaptation 与 slow consolidation

在线先把经验存入可审计、可删除的 memory，立即影响检索与行动；随后筛选可靠轨迹，周期性 SFT/RL/蒸馏到较慢的参数更新。这不是唯一架构，但比每次对话都无条件改全模型更易隔离、回滚与评估。

Context adaptation 不需要更新基础权重，成本低、可按用户/企业隔离，却受检索、上下文长度和总结失真影响。参数学习能更广泛改变能力，但昂贵且容易干扰旧知识。外部记忆不是没有遗忘：错误总结、检索失败和规则覆盖也会导致系统级遗忘。

### Continual learning 与 recursive self-improvement

持续适应环境不必改变学习算法。Recursive self-improvement 更强：系统的改进还能提升下一轮产生改进的机制，例如修改搜索策略、实验规划或自我修改程序。反复在同一数据上训练几轮、自动运行 pipeline 或保存对话摘要，都不自动达到这个定义。

需要同时测 forward transfer、backward transfer、retention、adaptation speed、资源增长和隐私隔离。每轮只看新任务成绩，会遗漏旧任务退化；每轮换评测器，又可能把评价漂移误认为能力增长。

## T47 · 前沿工作的七条路线：按机制读，不按公司口号读

### 路线与一手公开实例

下表是截至 2026-09-08 核对的代表性公开工作，不是公司完整内部路线或已部署能力清单。一个机构可以同时探索多条路线，论文原型、开源框架与生产系统证据不能互换。

| 路线 | 更新对象与核心主张 | 代表工作及要追问的边界 |
| --- | --- | --- |
| 内部多时间尺度学习 | 让记忆状态与慢参数按不同频率更新 | [Titans](https://arxiv.org/abs/2501.00663)、[Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)；追问固定状态容量、遗忘与大规模验证 |
| 外部 context/memory | 稳定骨干，快速更新可审计经验 | [ACE](https://arxiv.org/abs/2510.04618)；追问 playbook 冗余、错误积累与跨任务迁移 |
| 经验驱动的 agent RL | 将执行轨迹变成训练数据，尽量贴近部署 harness | [Agent Lightning](https://www.microsoft.com/en-us/research/project/agent-lightning/)、[Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/)；追问环境保真度与反馈成本 |
| 自生成课程与世界模型 | 不仅优化 learner，也扩大可交互任务分布 | [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)；追问模拟漏洞、课程难度和真实迁移 |
| 自生成监督 | 扩大可验证任务，训练更好的 evaluator | [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666)、[Nemotron-CrossThink](https://arxiv.org/abs/2504.13941)；追问循环偏差与独立真值 |
| 自动研究 | 让 agent 提假设、做训练实验并依据评测迭代 | [Automated Alignment Researchers](https://alignment.anthropic.com/2026/automated-alignment-researchers/)；追问研究任务是否预定义、预算与评价是否独立 |
| 自修改程序与改进器 | 修改 task agent，进一步允许修改 meta-agent | [Darwin Gödel Machine](https://sakana.ai/dgm/)、[Hyperagents](https://arxiv.org/abs/2603.19461)；追问搜索预算、可行空间和测试过拟合 |

### 每条路线真正补的是哪个瓶颈

内部记忆补长期历史的存储与访问；外部 memory 补快速知识更新；agent RL 补“知道怎么说”与“能在环境里做”的差距；课程/世界模型补新任务供给；self-generated supervision 补人类标注无法扩展的反馈；自动研究补实验与研究人力；recursive self-modification 则尝试改善改进过程本身。

例如 Qwen3-Coder 公开了可并行运行 20,000 个独立环境的训练基础设施。这说明环境规模是 agent RL 的关键变量，不证明每个环境都提供独立、无偏、高质量经验，也不代表所有模型都应复制该数字。Agent Lightning 的重点是衔接 agent 工作流与优化系统，降低训练—部署逻辑脱节，而不是发明一种适用于所有任务的新 reward。

ACE 将经验提炼进 evolving context，可在不更新主模型时改变行为；这支持系统层面的学习，但不能因此说基础权重获得了新知识。Self-Taught Evaluators 研究合成对比与评审改进，也不意味着真实人类价值可以被完全省略。CrossThink 用任务结构化和可验证答案扩大 RL 范围，不等于开放领域的所有事实判断都已自动可验证。

### 共通哲学与现实边界

共同趋势是从单个 checkpoint 转向“数据、经验、学习器、记忆、环境和评价器”的闭环。可验证反馈比逐条示范更易扩展，但可验证世界只覆盖真实世界的一部分；环境可以生成很多，却可能共同带偏；记忆可以持续更新，却需要删除与隔离。

因此，最可靠的结论不是某一公司“已经实现开放式自我进化”，而是不同工作在拆解同一组瓶颈。这里保留原对话的机制分类，去掉未经独立核对的公司星级、宣传式排名和把研究计划当成既成能力的表述。

## T48 · 自我改进闭环的研究判断与控制边界

### 六种闭环不要混成一种

数据闭环：模型生成候选，过滤后再训练；经验闭环：agent 执行任务，环境反馈产生轨迹；评价闭环：用新的比较与审计改进 judge；记忆闭环：总结成功失败并更新 context；程序闭环：提出 patch，运行评测，保留变体；研究闭环：提出训练方法，分配实验，检验假设再迭代。

每一类都可以写成“提议—验证—保留”，但保留的对象不同。生成更多题不等于优化器学会了新算法；修改 agent 的文件浏览工具不等于 foundation model 改了权重；搜索到更快 kernel 可能节省下一轮训练成本，却不是模型能力指标本身。

### DGM 与 Hyperagents 的关键区别

DGM 用可执行 agent 程序作为改进对象，维护候选 archive，通过经验评测保留有用修改。失败变体也可能作为后续搜索的中间节点；这不同于只保留单一当前最好版本的贪心 hill-climbing。它是经验验证的程序进化，不是 Gödel 式证明器保证每次修改有益。

Hyperagents 更进一步把解决任务的 agent 与提出修改的 meta-agent 放入可编辑系统，允许改进“如何产生修改”。其关键研究问题是改进机制是否对新任务迁移，而不是某个固定 benchmark 反复多搜几次后分数提高。[DGM](https://arxiv.org/abs/2505.22954)、[Hyperagents](https://arxiv.org/abs/2603.19461)

自动 alignment research 则在明确的训练与评测框架里提出方法、生成数据、运行实验。公开实验能支持特定已定义问题上的研究自动化，不能直接外推成任意新科学问题或所有未知 alignment failure 都能解决。[一手研究报告](https://alignment.anthropic.com/2026/automated-alignment-researchers/)

### 怎样证明下一轮真的更强

同时记录候选生成数、训练/推理 FLOPs、环境调用和人工干预；保留同预算的固定改进器、随机搜索、人工设计与不带 memory 的对照。最终测试不能反复参与候选筛选。每轮冻结一个独立评价锚，使用新任务、新环境和回归集，区分搜索 overfitting 与可迁移进步。

若改进器与 evaluator 一起变化，至少保留跨版本交叉评价矩阵：旧 judge 评新 policy、新 judge 评旧 policy，以及外部真值/人工评估。否则两者一起改变口味就会制造“持续进步”。

### 哪些控制不可交给闭环自行删除

实验应有独立权限边界、预算、日志、可回滚快照和数据隔离。Agent 提交的“更高分”不能替代对测试修改、数据污染和越权操作的检查；不能允许它直接改变用于证明自己成功的评测规则。涉及真实用户和敏感数据时，还需明确记忆的授权、保留与删除机制。

归根结底，持续学习的瓶颈不仅是梯度下降能否运行，而是能否不断获得有信息量的任务、可信反馈与独立证据。把所有对象放回同一学习系统视角，才能理解为什么 data prep、pre-training、post-training、test-time compute 与 agent environment 会相互影响，而不是彼此独立的技术清单。
