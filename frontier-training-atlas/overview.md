如果从 GPT-3.5/ChatGPT 开始往回看，我认为最重要的不是“模型一年比一年大”，而是 **LLM 的 scaling 对象连续发生了变化**：

$$
\boxed{
\text{参数规模}
\rightarrow
\text{训练 token / 数据质量}
\rightarrow
\text{post-training}
\rightarrow
\text{RL compute}
\rightarrow
\text{test-time compute}
\rightarrow
\text{工具与环境}
}
$$

今天的 frontier model 已经不是“一个训练得更大的 Transformer”，而更像一个由 **base model + synthetic-data engine + judge/verifier + RL loop + tools + inference-time search** 组成的系统。

而一个容易被忽视的事实是：**核心神经网络架构其实没有发生与能力增长同等幅度的革命。Decoder-only Transformer 至今仍是主干。**大量进步发生在数据、训练 recipe、稀疏计算、post-training 和 inference strategy 上。

---

## 一、先看整个历史的六个阶段

| 时期      | 代表                                | 最主要的范式变化                                                                     |
| ------- | --------------------------------- | ---------------------------------------------------------------------------- |
| 2022–23 | GPT-3.5 / ChatGPT                 | **pretrained LM → instruction-following assistant**；SFT + RLHF               |
| 2023–24 | GPT-4 / Llama 2→3                 | **参数 scaling → compute/data-optimal scaling**；数据质量、代码、长训练                    |
| 2023–24 | Mixtral / Gemini 1.5 / GPT-4o     | **dense → conditional compute；text → multimodal；短 context → long context**   |
| 2023–24 | DPO / RLAIF / synthetic data      | **人工标注 → scalable AI feedback；一次性 post-training → iterative post-training**  |
| 2024–25 | o1 / DeepSeek-R1 / Qwen3          | **alignment RL → capability RL；train-time + test-time reasoning scaling**    |
| 2025–26 | Llama 4 / o3-o4 / Gemini thinking | **offline training → continuous online RL；LM → reasoning/tool/agent policy** |

下面逐步展开。

---

## 二、第一阶段：GPT-3.5 / ChatGPT——最大的突破其实是 post-training

GPT-3 时代的基本想法仍然是：

$$
\boxed{\text{规模更大的 next-token predictor}}
$$

pretraining 获得：

* 知识；
* 语言；
* coding；
* latent reasoning；
* few-shot learning。

但用户需要大量 prompt engineering 才能把能力 elicitate 出来。

InstructGPT / ChatGPT 带来的关键改变是：

$$
\text{Pretraining}
\rightarrow
\text{SFT}
\rightarrow
\text{Reward Model}
\rightarrow
\text{PPO RLHF}
$$

人工写 demonstrations：

$$
(x,y^*)
$$

做 SFT，然后人工比较：

$$
y_A\succ y_B
$$

训练 RM，再 PPO。

OpenAI 当时一个非常有历史意义的结果是：**1.3B InstructGPT 在人类偏好上可以胜过 175B 的原始 GPT-3。**这说明“模型是否知道东西”和“模型是否能够按照用户需要把能力表现出来”是两回事。([OpenAI][1])

所以第一次大范式变化是：

$$
\boxed{
\text{Scaling parameters}
\quad\longrightarrow\quad
\text{Scaling usable behavior}
}
$$

这也是我们之前讨论的：

$$
\text{capability}\neq\text{elicitation}
$$

第一次成为产品级核心问题。

---

## 三、第二阶段：Chinchilla 思想真正影响 frontier training——“不要只加参数，加 token”

GPT-3 时代很容易形成一个直觉：

> 175B → 500B → 1T 参数，自然越来越好。

Chinchilla scaling law 改变了这个想法：给定训练 compute，**模型参数 \(N\) 和训练 token \(D\) 要共同扩大**；很多早期大模型实际上是“大而没训够”。([Google DeepMind][2])

于是之后非常明显的变化是：

$$
\boxed{
\text{parameter scaling}
\rightarrow
\text{parameter + data + compute scaling}
}
$$

Llama 3 是很好的公开例子。它仍然采用非常标准的 decoder-only Transformer，但 pretraining data 扩大到 **15T+ tokens**，是 Llama 2 的约 7 倍，并增加了约 4 倍代码数据。Meta 还指出 8B/70B 模型训练到 15T token 时仍继续改善。([Meta AI][3])

Llama 3.1 的最大 dense model 是 405B，同样训练在 15T+ tokens 上。([Meta AI][4])

所以 2023–24 一个非常大的认知变化是：

> **参数量已经不是模型“吃了多少 compute”的充分指标。**

---

## 四、数据的范式也变了：从“抓更多互联网”到“工程化构造训练分布”

GPT-3 时代可以粗略想成：

$$
\text{web/books/code}
\rightarrow
\text{filter}
\rightarrow
\text{pretrain}
$$

今天已经复杂得多：

$$
\text{raw web/docs/PDF/code}
$$

↓

quality scoring / dedup / language filtering

↓

知识密度控制

↓

STEM / code / reasoning oversampling

↓

synthetic textbooks / QA / code

↓

long-context curriculum

↓

pretraining

Qwen3 是特别透明的例子：约 **36T tokens、119 种语言/方言**；不仅使用 web，还用视觉语言模型从 PDF 类文档提取内容，并使用 Math/Coder 模型生成 synthetic textbooks、QA pairs 和 code snippets。训练又分成 30T+ general stage、约 5T knowledge-intensive stage 和 long-context stage。([Qwen][5])

所以：

$$
\boxed{
\text{data quantity}
\rightarrow
\text{data distribution engineering}
}
$$

越来越重要。

甚至可以说，今天 pretraining dataset 已经越来越像一种**人工设计的 curriculum**，而不是互联网的自然分布。

---

## 五、模型尺寸本身的含义也变了：MoE 让“总参数”和“每 token 计算量”分离

这可能是架构层面最重要的变化之一。

Dense model：

$$
\text{每个 token 使用几乎所有参数}
$$

MoE：

$$
\text{每个 token 只激活少量 experts}
$$

例如 Mixtral 8×7B 有约 47B 可访问参数，但每 token 只激活约 13B。([arXiv][6])

DeepSeek-V3：

$$
671B\text{ total}
$$

但：

$$
37B\text{ active/token}
$$

并训练了 14.8T tokens；同时用了 DeepSeekMoE、MLA、FP8 training、multi-token prediction 等训练/架构优化。([DeepSeek][7])

Gemini 1.5 也采用 MoE 架构。([blog.google][8])

所以今天讨论“模型大小”至少要分：

$$
\boxed{
N_{\rm total}
}
$$

和

$$
\boxed{
N_{\rm active/token}
}
$$

再加上：

$$
D_{\rm training}
,\quad
C_{\rm training}
$$

单说“这是 600B 模型”已经没什么信息量。

---

## 六、但 Transformer 本身并没有被推翻

这点值得单独强调。

从 GPT-3.5 到今天，基础模式仍然非常像：

$$
\text{embedding}
\rightarrow
\underbrace{
[\text{attention}+\text{MLP}]
\times L
}_{Transformer}
\rightarrow
\text{LM head}
$$

变化更多是：

* GQA/MQA；
* RoPE / context extension；
* MoE；
* MLA；
* tokenizer；
* normalization；
* positional encoding；
* FP8/BF16；
* FlashAttention；
* parallelism；
* MTP；
* speculative decoding。

Llama 3 甚至明确说自己故意保留 relatively standard decoder-only Transformer，只做 tokenizer、GQA 等改进。([Meta AI][3])

所以过去三年的能力提升不能主要归因于：

> “发现了 Transformer 之后的全新 architecture。”

没有。

更像：

$$
\boxed{
\text{同一种基础机器，被训练和使用得越来越聪明}
}
$$

---

## 七、第三次大变化：Multimodal 从“外挂 encoder”走向 native / end-to-end

早期 multimodal 很多是：

$$
\text{image encoder}
\rightarrow
\text{projection}
\rightarrow
\text{LLM}
$$

后来开始越来越 native。

GPT-4o 的公开系统卡明确说，它是一个 end-to-end omni model，同一个神经网络处理 text、audio、image、video 输入，并可生成多种模态输出。([OpenAI][9])

Gemini 1.5 则把：

$$
\boxed{\text{multimodality + long context}}
$$

结合起来，可以在数百万 token context 中处理 documents、video、audio；研究中甚至测试了 10M-token 量级 retrieval。([arXiv][10])

因此：

$$
\boxed{
\text{language model}
\rightarrow
\text{general sequence / multimodal model}
}
$$

是另一个重要范式变化。

---

## 八、第四次变化：post-training 从“一次 SFT + RLHF”变成迭代式数据飞轮

早期：

$$
\text{Base}
\rightarrow
\text{SFT}
\rightarrow
\text{RLHF}
\rightarrow
\text{done}
$$

后来逐渐变成：

$$
M_0
\rightarrow
\text{generate data}
\rightarrow
\text{filter/judge}
\rightarrow
\text{train}
\rightarrow
M_1
$$

然后：

$$
M_1
\rightarrow
\text{generate better data}
\rightarrow
\cdots
$$

这是一个非常重要但不像“MoE”那么醒目的范式变化。

Llama 3.1 明确采用 iterative post-training，每一轮 SFT + DPO，并利用当代最强的 405B 模型提升较小模型 post-training data 的质量。([Meta AI][11])

所以开始形成：

$$
\boxed{
\text{model}
\rightarrow
\text{data generator}
\rightarrow
\text{next model}
}
$$

的 feedback loop。

---

## 九、Synthetic data 的地位发生根本变化

早期 synthetic data 是补充。

今天它逐渐成为核心训练资产。

强 teacher 可以：

* 写 textbook；
* 生成 QA；
* 写代码；
* 产生 reasoning trajectories；
* critique；
* revise；
* 给 preference labels；
* 做 judge；
* 生成 hard prompts。

例如 Qwen3 pretraining 已经明确使用 Qwen2.5-Math/Coder 产生 synthetic math/code data。([Qwen][5])

所以数据来源开始从：

$$
\boxed{\text{human civilization 已经写出来的东西}}
$$

扩展成：

$$
\boxed{\text{更强模型主动生成出来的训练环境}}
$$

这是非常大的变化。

---

## 十、标注也从 Human Feedback → AI Feedback → Verifiable Feedback

最早：

$$
\text{human writes demonstration}
$$

$$
\text{human ranks A vs B}
$$

成本很高。

Constitutional AI / RLAIF 开始让 AI：

$$
\text{critique}
\rightarrow
\text{revise}
$$

以及：

$$
A,B
\rightarrow
\text{AI preference}
$$

再训练 RM/RL。([Anthropic][12])

随后越来越进一步：

### 主观问题

LLM judge / RM：

$$
\text{helpfulness, style, safety}
$$

### 客观问题

直接 verifier：

$$
\text{math answer correct?}
$$

$$
\text{unit tests pass?}
$$

$$
\text{tool executed successfully?}
$$

所以 human 的角色逐渐从：

> **逐条生产 label**

变成：

> **定义 rubric、标定 judge、审计 failure cases。**

这是 scalable oversight 的范式。

---

## 十一、DPO：post-training 从“训练 RM + 跑 PPO”出现了离线捷径

经典 RLHF：

$$
\text{preference pairs}
\rightarrow
RM
\rightarrow
PPO
$$

需要：

* policy；
* reference；
* reward model；
* critic；
* rollout infrastructure。

2023 DPO 表明，在特定 KL-regularized preference model 假设下，可以直接：

$$
(x,y^+,y^-)
\rightarrow
\boxed{\text{DPO loss}}
\rightarrow
\pi_\theta
$$

把 RM 和 PPO 都消掉。([arXiv][13])

这推动了 2023–24 大量：

* DPO；
* IPO；
* ORPO；
* KTO；
* preference optimization variants。

这是一个工程范式：

$$
\boxed{
\text{complex online RLHF}
\rightarrow
\text{cheap offline preference optimization}
}
$$

但之后 reasoning model 又让 pendulum 摆回真正的 online RL。

---

## 十二、最大的近期范式转变：post-training 不再只是 alignment，而开始创造 capability

GPT-3.5 时代，post-training 最自然的理解是：

> pretrained model 已经有能力，RLHF 主要让它更听话。

但 o1 / R1 之后这已经不够准确。

OpenAI 2024 对 o1 的公开描述非常明确：

$$
\boxed{
\text{大规模 RL}
\rightarrow
\text{学习 productive chain-of-thought}
}
$$

而且性能同时随：

$$
\text{train-time RL compute}
$$

和：

$$
\text{test-time reasoning compute}
$$

提高。([OpenAI][14])

这意味着出现了一条新的 scaling axis：

$$
\boxed{
C_{\rm posttrain}
}
$$

pretraining 不再是唯一真正创造能力的大训练阶段。

---

## 十三、DeepSeek-R1 把这个 recipe 公开得最清楚

R1-Zero 做了一个极其有影响力的实验：

$$
\text{Base model}
\rightarrow
\boxed{\text{RL directly}}
$$

没有先用大量 human CoT SFT。

结果模型自己出现：

* verification；
* reflection；
* backtracking；
* alternative strategies。

这说明 reasoning strategy 可以通过 outcome reward 产生，而不一定要逐步 imitation 人类 reasoning。([arXiv][15])

但真正的 R1 并不是“pure RL”。

实际 pipeline：

$$
\boxed{
\text{Cold-start SFT}
}
$$

↓

$$
\boxed{\text{Reasoning RL}}
$$

↓

$$
\boxed{\text{Rejection sampling + SFT}}
$$

↓

$$
\boxed{\text{General RL}}
$$

这样既保留 reasoning，又解决 readability、language mixing、general assistant behavior。([Nature][16])

这基本成为公开 reasoning model recipe 的原型之一。

---

## 十四、Qwen3 把它进一步系统化

Qwen3 的公开 post-training 是四阶段：

$$
\boxed{
1.\ Long-CoT\ SFT
}
$$

$$
\boxed{
2.\ Reasoning\ RL
}
$$

$$
\boxed{
3.\ Thinking/non-thinking\ fusion
}
$$

$$
\boxed{
4.\ General\ RL
}
$$

而第三阶段的数据本身，就是第二阶段增强后的模型生成的。([Qwen][5])

这已经构成一个很明显的：

$$
\boxed{
\text{RL}
\rightarrow
\text{generate better data}
\rightarrow
\text{SFT/distill}
\rightarrow
\text{RL}
}
$$

闭环。

---

## 十五、所以 SFT 的角色正在缩小、变精

这并不是说 SFT 不重要，而是角色改变了。

早期：

$$
\boxed{\text{SFT 是主要的 behavior learning stage}}
$$

今天越来越像：

$$
\boxed{
\text{SFT = bootstrap / protocol / cold start / distillation}
}
$$

然后真正困难的 reasoning、coding、agent capability：

$$
\boxed{\text{交给 online RL 去探索}}
$$

Meta 在 Llama 4 上讲得非常直白：

$$
\boxed{
\text{lightweight SFT}
\rightarrow
\text{online RL}
\rightarrow
\text{lightweight DPO}
}
$$

并表示过强的 SFT/DPO 会限制 exploration；他们甚至删掉大量 easy SFT data，再让 RL 聚焦 medium/hard prompts。([Meta AI][17])

这是一个非常大的观念逆转：

> 以前“更多高质量 demonstration”几乎总被认为更好；现在 frontier reasoning training 会担心它**过度约束 policy**。

---

## 十六、RL 的 reward 也从“人喜欢不喜欢”变成“任务到底成功没有”

传统 RLHF：

$$
R=\text{human preference}
$$

reasoning RL 越来越依赖：

$$
\boxed{\text{verifiable reward}}
$$

例如：

$$
R_{\rm math}
=
1[\text{answer correct}]
$$

coding：

$$
R_{\rm code}
=
f(\text{unit tests})
$$

tool agent：

$$
R_{\rm agent}
=
1[\text{task success}]
$$

这非常关键，因为 verifier：

* 成本低；
* scalable；
* 不容易产生主观 disagreement；
* 可以生成海量 rollout。

所以：

$$
\boxed{
\text{RLHF}
\rightarrow
\text{RLVR + RLAIF + multi-objective reward}
}
$$

是现在的大趋势。

---

## 十七、自我迭代也从 Rejection Sampling 走向 continuous online RL

早期 self-improvement：

$$
\text{sample N}
\rightarrow
\text{挑最好的}
\rightarrow
\text{SFT}
$$

也就是 rejection sampling。

现在进一步：

$$
\pi_t
$$

↓

大量 rollout

↓

verifier / judge

↓

筛选当前模型“有时会、有时不会”的问题

↓

RL

↓

$$
\pi_{t+1}
$$

↓

重新生成下一轮训练分布。

Llama 4 公开描述了这种 continuous online RL：训练和数据过滤交替进行，持续留下 medium-to-hard prompts，并过滤 zero-advantage prompts。([Meta AI][17])

所以 curriculum 不再是人一次性准备好的：

$$
D_{\rm train}
$$

而变成：

$$
\boxed{
D_t=f(\pi_t)
}
$$

训练数据分布随模型一起成长。

这是非常重要的“自我迭代”范式。

---

## 十八、Distillation 的含义也变了

过去 distillation 更多是：

> 大模型 → 小模型，降低成本。

现在：

$$
\boxed{\text{distillation = capability transfer}}
$$

特别是：

$$
\text{strong reasoning teacher}
\rightarrow
\text{long reasoning trajectories}
\rightarrow
\text{student SFT}
$$

DeepSeek 把 R1 reasoning distill 到 Qwen/Llama 系列的小模型，并报告这些 distilled models 显著超过原来的 instruct checkpoints。([arXiv][15])

DeepSeek-V3 也公开描述了把 R1 的 verification / reflection patterns distill 进普通模型。([GitHub][18])

所以现在可以有一种非常重要的策略：

$$
\boxed{
\text{昂贵 RL 只在 teacher 上做一次}
}
$$

然后：

$$
\boxed{
\text{通过 synthetic trajectories 大规模蒸馏}
}
$$

给很多学生模型。

---

## 十九、另一条全新的 Scaling Law：Test-Time Compute

GPT-3.5 的典型 inference：

$$
x\rightarrow y
$$

一次 forward/autoregressive generation。

reasoning model：

$$
x
\rightarrow
\text{think}
\rightarrow
\text{verify}
\rightarrow
\text{backtrack}
\rightarrow
y
$$

甚至：

$$
y_1,\ldots,y_N
\rightarrow
\text{compare/search}
\rightarrow
y^*
$$

OpenAI o1 明确展示性能随着 test-time compute 增长。([OpenAI][14])

Gemini 2.5 和 Claude 3.7 等随后都把 **thinking budget** 变成了显式产品参数：模型根据问题难度花不同数量的推理 token。([Google Developers Blog][19])

Gemini Deep Think 更进一步公开描述 parallel thinking——同时探索多个 hypotheses，再修正/组合。([blog.google][20])

所以性能现在变成：

$$
\boxed{
Q=
f(
C_{\rm pretrain},
C_{\rm posttrain},
C_{\rm test}
)
}
$$

这是 GPT-3.5 时代没有真正规模化利用的第三条 compute axis。

---

## 二十、Tool use 进一步把“模型”变成了 agent

早期 function calling：

$$
x
\rightarrow
\text{structured API call}
$$

主要靠 SFT。

今天 reasoning model 开始通过 RL 学：

> **什么时候需要工具，以及调用哪个工具才能完成最终任务。**

OpenAI 对 o3/o4-mini 的公开描述就明确说，这些 reasoning models 经过 RL 训练，能在推理中决定如何组合 web search、Python、file analysis、vision 等工具。([OpenAI][21])

于是 action 已经不仅是：

$$
a_t=\text{token}
$$

还可以是：

$$
a_t=
\begin{cases}
\text{text token}\\
\text{search}\\
\text{Python}\\
\text{browser action}\\
\text{API call}
\end{cases}
$$

因此 LLM 从：

$$
\boxed{\text{text generator}}
$$

逐渐变成：

$$
\boxed{\text{policy interacting with an environment}}
$$

这可能是下一阶段最重要的范式。

---

## 二十一、把所有变化按你问的维度重新整理

| 维度                      | GPT-3.5 左右                   | 今天的趋势                                                      |
| ----------------------- | ---------------------------- | ---------------------------------------------------------- |
| **Architecture**        | dense decoder Transformer    | Transformer 仍主导；MoE/GQA/MLA/native multimodal              |
| **模型尺寸**                | 参数数是核心指标                     | total vs active params；参数量不再足够描述能力                         |
| **Pretrain 数据量**        | 百亿～万亿级思路                     | 十万亿乃至数十万亿 token                                            |
| **数据来源**                | web/books/code               | curated web + PDF + code + STEM + multilingual + synthetic |
| **数据策略**                | 抓取+过滤                        | quality scoring、dedup、curriculum、difficulty selection      |
| **Pretraining**         | 基本 next-token                | next-token 仍核心，但有 staged training、long-context、MTP 等       |
| **SFT**                 | assistant 转换的核心              | lightweight cold-start / protocol / distillation           |
| **Preference training** | RM + PPO                     | DPO、RLAIF、online preference、multi-objective                |
| **RL**                  | alignment/helpfulness        | reasoning/coding/tool/agent capability acquisition         |
| **Reward**              | human preference RM          | verifier + AI judge + RM + rules                           |
| **标注**                  | human demonstrations/ranking | AI labels/rubrics + verifier，human audit                   |
| **Synthetic data**      | 辅助                           | pretraining/post-training 的核心来源之一                          |
| **Self iteration**      | 少量                           | generate→verify→filter→train→regenerate                    |
| **Distillation**        | 模型压缩                         | reasoning/capability transfer                              |
| **Inference**           | single generation            | thinking budget、best-of-N、parallel search、tools            |
| **Context**             | 几千 tokens                    | 128K–1M+ 成为实际能力轴                                           |
| **Modality**            | text                         | native text/image/audio/video                              |
| **Agent**               | 外部 orchestration             | tool use 本身进入 post-training/RL                             |

---

## 二十二、如果一定要给这些推动因素排重要性

不能严格定量归因，但我会把 GPT-3.5 以来的进步粗略归结为下面这几个层次：

**第一层仍然是 pretraining scaling。** 更多 compute、更高质量且更广的训练数据仍然是所有后续能力的地基。Llama 3、Qwen3、DeepSeek-V3 都说明这一点。([Meta AI][3])

**第二层是 post-training 从 alignment 变成 capability training。** 这是 2024 后最大的变化之一：RL 不再只是“让人更喜欢”，而开始大规模训练 reasoning、coding、tool use。([OpenAI][14])

**第三层是 synthetic data + scalable feedback。** 强模型同时充当 teacher、data generator、judge、critic，verifier 提供海量可靠反馈，使训练不再受 human-label throughput 限制。

**第四层是 inference-time compute。** 过去模型参数冻结后，性能基本固定；现在同一个 checkpoint 可以通过多想 10 倍、100 倍，得到明显更强的表现。([OpenAI][14])

**第五层才是架构创新。** MoE、MLA、GQA、multimodal architecture 非常重要，但最近能力增长并不是来自“Transformer 被另一个架构取代”。更多是在让同样的 Transformer compute **更便宜、更稀疏、更长上下文、更容易扩展**。

---

## 二十三、我认为最核心的历史脉络可以压缩成这张图

GPT-3 时代：

$$
\boxed{
\text{Internet}
\rightarrow
\text{Pretraining}
\rightarrow
\text{Big LM}
}
$$

GPT-3.5 / ChatGPT：

$$
\boxed{
\text{Pretraining}
\rightarrow
\text{SFT}
\rightarrow
\text{Human preference RLHF}
\rightarrow
\text{Assistant}
}
$$

GPT-4 / Llama 3 时代：

$$
\boxed{
\text{Better curated massive data}
+
\text{larger compute}
+
\text{stronger post-training}
}
$$

2024–25 reasoning 时代：

$$
\boxed{
\text{Base}
\rightarrow
\text{cold-start}
\rightarrow
\text{online RL}
\rightarrow
\text{synthetic/rejection sampling}
\rightarrow
\text{SFT/distill}
\rightarrow
\text{general RL}
}
$$

现在进一步变成：

$$
\boxed{
\begin{array}{c}
\text{strong pretrained model}\\
+\text{teacher models}\\
+\text{AI judges}\\
+\text{verifiers}\\
+\text{online RL}\\
+\text{self-generated curriculum}\\
+\text{tools/environment}\\
+\text{test-time search}
\end{array}
}
$$

也就是说，**“LLM 模型”这个研究对象本身正在改变。**

GPT-3.5 时代研究的问题主要是：

> 怎样训练一个更好的 neural network？

现在越来越像：

> **怎样构造一个能够自己生成经验、获得反馈、探索策略、调用工具、分配推理计算，并把所得能力再蒸馏回下一代模型的 learning system？**

我认为这是从 GPT-3.5 到今天最根本的范式变化。

[1]: https://openai.com/index/instruction-following/ "Aligning language models to follow instructions | OpenAI"
[2]: https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/ "An empirical analysis of compute-optimal large language model training — Google DeepMind"
[3]: https://ai.meta.com/blog/meta-llama-3/ "Introducing Meta Llama 3: The most capable openly available LLM to date"
[4]: https://ai.meta.com/research/publications/the-llama-3-herd-of-models/ "The Llama 3 Herd of Models | Research - AI at Meta"
[5]: https://qwenlm.github.io/blog/qwen3/ "Qwen3: Think Deeper, Act Faster | Qwen"
[6]: https://arxiv.org/abs/2401.04088 "Mixtral of Experts"
[7]: https://www.deepseek.com/news/deepseek-v3/ "DeepSeek | DeepSeek-V3 正式发布"
[8]: https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/ "Introducing Gemini 1.5, Google's next-generation AI model"
[9]: https://cdn.openai.com/gpt-4o-system-card.pdf "GPT-4o System Card"
[10]: https://arxiv.org/abs/2403.05530 "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context"
[11]: https://ai.meta.com/blog/meta-llama-3-1/ "Introducing Llama 3.1: Our most capable models to date"
[12]: https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback "Constitutional AI: Harmlessness from AI feedback"
[13]: https://arxiv.org/abs/2305.18290 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[14]: https://openai.com/index/learning-to-reason-with-llms/ "Learning to reason with LLMs | OpenAI"
[15]: https://arxiv.org/abs/2501.12948 "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
[16]: https://www.nature.com/articles/s41586-025-09422-z "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | Nature"
[17]: https://ai.meta.com/blog/llama-4-multimodal-intelligence/ "The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation"
[18]: https://github.com/deepseek-ai/deepseek-v3 "GitHub - deepseek-ai/DeepSeek-V3 · GitHub"
[19]: https://developers.googleblog.com/en/start-building-with-gemini-25-flash/ "Start building with Gemini 2.5 Flash - Google Developers Blog"
[20]: https://blog.google/products-and-platforms/products/gemini/gemini-2-5-deep-think/ "Gemini 2.5: Deep Think is now rolling out"
[21]: https://openai.com/index/introducing-o3-and-o4-mini/ "Introducing OpenAI o3 and o4-mini | OpenAI"
