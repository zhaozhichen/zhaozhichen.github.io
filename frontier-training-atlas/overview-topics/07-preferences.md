## T37 · Offline、Online DPO 与 Rejection Sampling

### 数据来源、学习目标和反馈来源是三个独立轴

Offline/online 描述数据是否随当前策略刷新；SFT、DPO、policy gradient 描述更新目标；human/AI/verifier 描述反馈来源。标准 DPO 常用于离线偏好集，但“DPO 天生不能 online”是错误的：可以用当前 policy 采样新回答，重新获得偏好，再做同一个 pairwise loss。

$$
\pi_t\longrightarrow\{y_i\sim\pi_t\}
\longrightarrow\text{比较与筛选}
\longrightarrow D_t
\longrightarrow\pi_{t+1}.
$$

这个闭环改善候选覆盖，却不把 DPO 自动变成 PPO/GRPO。是否探索充分取决于采样温度、prompt 分布、采样次数和保留策略，不由算法名称单独决定。固定 reference 与每轮移动 reference 也改变锚定目标，应明确区分。

### LLM 中的 rejection sampling 通常不是经典精确采样

经典 rejection sampling 从 proposal $q$ 采样，以 $p(y)/(Mq(y))$ 接受，在满足上界等条件时得到目标 $p$。LLM 流程里“生成多个候选，按 verifier 筛选，再 SFT”通常没有这样的密度比保证。

若 proposal 为 $q(y\mid x)$，acceptance 为 $a(x,y)\in[0,1]$，接受后的分布是

$$
q_{\rm acc}(y\mid x)=
\frac{q(y\mid x)a(x,y)}{Z(x)}.
$$

Best-of-n 选择一组中最高分者，是与 order statistics 有关的另一种分布；它既不等于固定阈值筛选，也不必等于指数倾斜策略。训练 student 模仿这些成功轨迹，本质上仍是加权/筛选后的行为克隆，不需要对 reward 做 policy gradient。

### 保留失败样本有什么价值

只模仿 winner 简单稳定，但丢掉了“为什么其他回答更差”的相对信息。DPO 使用正负对；RL 可以利用负 advantage；失败轨迹还可训练恢复策略与 judge。把所有低分样本扔掉可能缩窄覆盖，尤其是困难 prompt 根本没有 winner 时。

比较三种方法应固定候选生成成本、反馈成本、更新次数和最终推理预算。若一个方法多用了十倍采样再筛选，不能把所有收益归因于 loss 形式。显式 RM 可复用到 best-of-n、主动标注和新 rollout；DPO 的隐式 scorer 也可计算，但代价、校准与跨策略泛化需另测。

## T38 · RLAIF、Constitutional AI 与可扩展反馈

### AI feedback 不指定唯一优化器

RLAIF 指反馈由 AI 产生。AI 可以给 critique，再生成 revised answer 用于 SFT；可以给 pairwise preference，训练 RM 或直接 DPO；也可以在线给 reward，供 policy gradient 使用。因此“人类 RLHF、AI DPO”不是正确对应关系。

Constitutional AI 用明确原则引导批评、修订与偏好评价，使人类监督从逐条示范转向原则与审计。它仍依赖人的价值选择、模型能力与独立检验，不是完全无人工假设的自我对齐。[Constitutional AI](https://arxiv.org/abs/2212.08073)

### 先写 rubric，再选 judge

把正确性、帮助性、无害性、忠实性、格式与效率分别定义。对于可验证的数学或代码，规则/执行器常更一致，但测试不完备、答案提取错误、超时和环境漏洞仍会让 verifier 失真。自然语言 judge 擅长覆盖开放目标，却容易受长短、文风、自信表达和位置影响。

应随机交换 A/B 顺序、隐藏候选模型身份、控制长度与提示泄漏，用独立人类标注做 calibration。不要把 judge 自报“95% 自信”当作校准概率；多个 judge 共享模型家族或训练数据时，也不是独立投票。

### 反馈闭环里的最危险偏差

若 policy 学会迎合训练 judge，而评测仍用同一个 judge，同源错误可能被不断放大。把训练 reward、验证集、最终测试、人工审计和真实环境结果分层，并保留最容易让 judge 分歧的 hard cases。

分歧有时是标注噪声，有时反映真实价值冲突。可以降低不确定样本权重或请求复核，但不能把所有异议删除来制造“一致性”。安全高分也可能来自过度拒绝，因此同时测可正常回答任务的帮助性。

可扩展反馈的目标不是消灭人工，而是把有限人工预算用在校准、盲点发现与高风险边界上。只有 judge 的独立有效性跟得上 policy 的优化强度，更多 RL compute 才可能带来可相信的收益。

## T39 · 偏好不传递、多目标与约束优化

### 一个标量能表示什么

固定 prompt、固定 rubric 的 BT 模型给每个答案一个标量 $r(y)$。如果稳定偏好概率同时满足 $P(A>B)>1/2$、$P(B>C)>1/2$、$P(C>A)>1/2$，就不能由这个标量排序完全表示。单次标注出现循环却不一定违背模型，因为 BT 本来允许随机误判。

真实群体也能产生 Condorcet 循环：三等份人分别排序 A>B>C、B>C>A、C>A>B，则三组两两多数偏好都形成环。还需区分不同 prompt、不同人群与 rubric；把不同问题的比较拼成环不构成同一个偏好系统不传递的证据。

### 多维 reward 不会自动解决循环

可以保留 $r(y)=(r_1,\ldots,r_K)$，用 $w^\top r$ 决策。只要固定 $w$，它仍是一个传递的标量排序；保留向量的意义是让取舍显式，而不是获得表达任意循环的魔法。若必须建模非传递 pairwise 偏好，需要直接的成对比较模型或博弈式策略。

Raw reward 的数值尺度重要：把一个维度乘 100，固定权重的意义完全改变。应以校准、阈值、分布分位或任务指标解释尺度，而不把 0.7 帮助性 + 0.3 安全性当成人自然理解的稳定配比。

### 从加权和变为“先达标，再优化”

$$
\max_\pi R_{\rm help}(\pi)
\quad\text{s.t.}\quad R_{\rm safe}(\pi)\ge c.
$$

拉格朗日函数可写

$$
\mathcal L(\pi,\lambda)=R_{\rm help}(\pi)
+\lambda(R_{\rm safe}(\pi)-c),\quad\lambda\ge0.
$$

在 max-policy/min-dual 约定下，约束违反时

$$
\lambda\leftarrow[\lambda+\eta(c-\widehat R_{\rm safe})]_+,
$$

下一次 policy 更新更重视安全。约束不可行时，乘子持续变大不意味着算法最终能创造可行策略，应检查阈值、模型能力与估计偏差。

Pareto 最优意为不存在另一方案在所有维度不差、至少一维更好。最大化一个主目标、约束其他目标，可能只保证弱 Pareto 性或需要 tie-breaking；加入 KL 后优化的是正则化目标，不能直接把它等同于原始 reward 空间的无条件 Pareto 最优。期望安全下界也不保证每个请求安全。

## T40 · MOPO 深读：偏好约束、鲁棒下界与策略提取

### 问题不是“再挑一组 reward 权重”

MOPO（Multi-Objective Preference Optimization）希望直接使用多维成对偏好，最大化一个主维度，并让其他维度超过可调阈值。本文对应 [2505.10892v2](https://arxiv.org/html/2505.10892v2)，其版本日期为 2026-06-05。下面把论文方案与独立数学核对分开；不是声称其所有理论结论都已被严格证明。

令 $p_k(y\succ y'\mid x)$ 为第 $k$ 个维度的偏好概率。两个候选来自 behavior policy $\mu$；prompt 来自 $\nu$。定义对固定对手分布的平均胜率

$$
\bar p_k(x,y)=\mathbb E_{y'\sim\mu(\cdot\mid x)}
p_k(y\succ y'\mid x),\qquad
R_k(\pi)=\mathbb E_{x\sim\nu,y\sim\pi}\bar p_k(x,y).
$$

这是“对 $\mu$ 选出的对手，按该维度能赢多少”的评价，不是无条件的客观质量分。$\mu$ 改变，评价尺度也会改变；原始 pairwise 关系可能不传递，但对固定 $\mu$ 积分后得到一个可排序的平均胜率。

以 $p$ 表示主维度，$q_j$ 表示次维度：

$$
\max_\pi R_p(\pi)-\tau
\mathbb E_{x\sim\nu}D_{\rm KL}(\pi(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x))
\quad\text{s.t.}\quad R_{q_j}(\pi)\ge b_j.
$$

阈值 $b_j$ 是政策取舍，不是算法从无偏好数据自动得到的价值真理。

### 用 importance ratio 改写时，必须保留归一化

固定一个 $x$，令 $\rho(y)=\pi(y)/\pi_{\rm ref}(y)$。真正的概率比不仅非负，还满足

$$
\mathbb E_{\pi_{\rm ref}}\rho=1.
$$

主目标为 $\mathbb E_{\rm ref}[\rho\bar p-\tau\rho\log\rho]$，约束为 $\mathbb E_{\rm ref}[\rho\bar q_j]\ge b_j$。引入 $\lambda_j\ge0$ 和归一化乘子 $\zeta$，一阶条件为

$$
\bar p(y)+\lambda^\top\bar q(y)
-\tau(\log\rho(y)+1)+\zeta=0.
$$

因此，在完整归一化问题中：

$$
\rho^*_\lambda(y)=
\frac{\exp\{[\bar p(y)+\lambda^\top\bar q(y)]/\tau\}}
{\mathbb E_{\pi_{\rm ref}}
\exp\{[\bar p+\lambda^\top\bar q]/\tau\}}.
$$

论文式 (3) 写出带 $-1$ 的未归一化指数权重，后面的 tabular policy extraction 又显式归一化。学习时不能将那一个未归一化表达式独立当作满足所有概率约束的 $\pi/\pi_{\rm ref}$。上式是对规范化问题的独立推导，用来校验实现，而不是对原式逐字照抄。

### Dual variable 如何表达约束压力

固定 $\lambda$ 求 policy 后，dual objective 在归一化版本中可写为

$$
g(\lambda)=\tau\log
\mathbb E_{\pi_{\rm ref}}e^{(\bar p+\lambda^\top\bar q)/\tau}
-\lambda^\top b.
$$

于是 $\nabla_{\lambda_j}g=R_{q_j}(\pi_\lambda)-b_j$，对 dual 做下降：

$$
\lambda_j\leftarrow
[\lambda_j+\eta(b_j-\widehat R_{q_j})]_+.
$$

违反约束时乘子增加，policy 更偏向该维度。对偶最优、概率空间的凹性和真实 LLM 训练后的可行性应区分：函数逼近、采样误差与不精确内循环会破坏理想条件。

### 为什么需要鲁棒 lower bound

约束要求数值达标，不只需要排序正确。小数据上估计“安全胜率 0.8”，实际可能只有 0.65，因此论文考虑在 reference 周围的 KL ball 内，让一个对抗分布寻找更低的约束值。

为清楚展示数学，固定 $x$，把 $\bar q_j(y)$ 视作已经对对手平均的函数，设 $f_j(y)=\rho(y)\bar q_j(y)$：

$$
\underline G_j(\rho)=
\min_{Q:\,D_{\rm KL}(Q\|\pi_{\rm ref})\le\epsilon}
\mathbb E_Q f_j(y).
$$

在适当有限支持与可行性条件下，它的熵风险对偶为

$$
\underline G_j(\rho)=
\sup_{\chi>0}
\left[-\chi\log\mathbb E_{\pi_{\rm ref}}e^{-f_j/\chi}
-\chi\epsilon\right].
$$

对抗分布满足 $Q^*(y)\propto\pi_{\rm ref}(y)e^{-f_j(y)/\chi}$，把质量更多放到低值区域。$\epsilon$ 大，保护范围更宽、下界更保守；$\epsilon=0$ 回到 reference 下的平均。这里 lower bound 是对**指定分布邻域**的最坏值，不自动等于带任意置信水平的统计置信下界。

必须在 log-sum-exp 中保留负号才能偏向低值。还要明确不确定性施加在 $y$、$(y,y')$ 还是 prompt 分布上；$\exp(\mathbb E f)$ 与 $\mathbb E\exp f$ 不可交换。论文正文不同位置的归一化和符号需要结合公式定义核对，不能拿某一行直接拼成训练代码。

### 从权重到可生成的模型

求出期望的 normalized weight 后，可训练生成策略做 importance-weighted behavior cloning：

$$
\mathcal L_{\rm BC}(\psi)
=-\mathbb E_{x,y\sim\pi_{\rm ref}}
\left[\operatorname{stopgrad}(\rho^*(x,y))
\log\pi_\psi(y\mid x)\right].
$$

如果离线数据实际来自 $\mu\ne\pi_{\rm ref}$，还需相应密度比或明确近似；不能把数据来源标签改名就获得无偏期望。Policy loss 的梯度经 $\pi_\psi$，目标权重在该子步冻结；$\lambda,\chi$ 用各自目标更新。所有变量“联合优化”不意味着每个 loss 都对所有分支反传。

实践要避免指数溢出，使用 log-space；限制极大 importance weight、记录 effective sample size：

$$
\operatorname{ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

ESS 很低表示少数离线样本支配学习。裁剪/自归一化会引入偏差，应作为实际算法选择记录，而不是仍宣称精确等价。

### Lagged reference、LB 与实验到底证明了什么

论文用滞后 policy 刷新 reference，并根据前期表现调整次目标阈值。这样改变了锚与约束，需检查累计漂移和权重稳定性。**MOPO-Lag 的 Lag 指 Lagrangian 版本；MOPO-LB 的 LB 指 log-barrier 版本**，不是把二者误记成“有无 lagged reference”或“有无 lower bound”。

Log barrier 可在可行侧使用 $-\sigma\log(G_j-b_j)$ 作为惩罚，靠近边界惩罚增大；不可行侧需可微延拓，不能对负数硬取对数。Lagrangian 则通过 multiplier 自适应调节约束压力。

实验包括已知偏好结构的合成 sanity checks，以及 Helpful Assistant、Reddit Summarization 上的多目标评估，比较不同权重/阈值形成的经验前沿。它支持该设置下的有效性，不证明所有现实偏好集都达到全局 Pareto 最优。审阅时需看评价 RM 是否独立、阈值是否在测试集上调、各方法数据/算力是否匹配、约束违反率及置信区间，而不只挑二维图里最好看的点。

应用前的关键判断是：**MOPO 把多目标约束显式化，这是贡献；有限数据、归一化、参考移动与函数逼近后的保证，仍要单独验证。** 原对话里无条件的 Pareto 证明和直接复制未归一化概率的步骤不予沿用。

## T41 · 奖励涨、独立评测跌：四种原因的鉴别

### 先冻结现场，而不是继续追更高 reward

保留恶化前后 checkpoint、rollout、reward 分解、长度、KL、数据版本与 judge 配置。重新在同一固定评测集和同一推理预算上评估，确认下降超出采样误差；如果评估配置本身变化，先恢复可比性。

| 假设 | 支持证据 | 关键反证/实验 |
| --- | --- | --- |
| Reward hacking | 高 reward 样本利用文风、长度、格式或 verifier 漏洞 | 盲评、换独立 judge、修 verifier 后收益消失 |
| 过拟合/覆盖收缩 | 训练 prompt 改善，未见域退化、多样性下降 | 扩大干净覆盖、早停、replay 后恢复 |
| 评估偏差 | 特定评审/顺序/长度决定胜负 | A/B 交换、长度匹配、人工复核和真实任务结果 |
| 实现错误 | reward 与轨迹错配、mask/终止错误、KL 符号异常 | 固定微型样本手算、单步梯度与序列对齐测试 |

这些原因可以同时存在，不是互斥四选一。RM 在普通 hold-out 上不错，却在被 policy 优化出的新样本上失败，仍可属于 reward exploitation。

### 修复必须对应原因

对漏洞奖励先修验证器并重新标注，不只加 KL；对覆盖收缩用 replay、数据扩展与保守更新；对 judge 偏差做独立校准；对实现错误回到最小复现。降低 LR/KL 调整可以止损，但不能证明根因已经解决。

监控 reward–quality 曲线而非单点，按难度、语言、长度、拒绝和工具任务分组。最好的停止点可能早于 reward 最大点。最终选 checkpoint 的依据是预先定义的独立目标与约束，不是训练指标更好看。

## T42 · Test-time Compute：采样、验证、搜索与预算分配

### 推理时计算不只有“多写一些思考”

固定参数下可以增加串行推理、并行候选、自一致性投票、best-of-n、critic revision、树搜索、工具执行与 verification。它们消耗不同的成本：生成 token、前缀复用、环境延迟、judge 调用与失败重试都要计入。TTC 通常不更新权重，区别于 test-time training。

同题独立采样、单次成功率 $p$ 时，至少一次成功概率是

$$
1-(1-p)^k.
$$

但部署成功还需正确选择成功项。对不同 prompt，正确平均是 $\mathbb E_x[1-(1-p_x)^k]$，不能把平均成功率先代进去；难度异质性和相关失误会改变收益。多采样能看到正确答案，不等于实际答案更好。

### Process reward 与搜索的角色

Outcome reward 看最终结果，process reward 评价中间步骤；tree search 用局部评价分配探索预算。PRM 不必等于 $V^\pi$：它可以测步骤正确性，而 value 测后续期望回报。一个局部正确但走向死路的步骤可能有高 PRM、低成功 value。

Self-consistency 依赖答案聚合，best-of-n 依赖 selector，MCTS 依赖扩展与 value 的可靠性。把更高分轨迹蒸馏回模型会转成训练时收益，但比较时要计入产生轨迹的 teacher/search 成本。

### 为什么要自适应分配

对 prompt $x$，概念上选择

$$
C^*(x)=\arg\max_C[Q(x,C)-\lambda C].
$$

简单题多算可能几乎无收益，难题也可能由于能力/验证瓶颈而不再改善。应估计边际质量收益、停止条件和不确定性，而不是所有题统一最长输出。Anytime 是有用类比，但现实 LLM 不保证给更多时间质量单调不降；反复改写可能把原本正确答案改错。

比较方法时固定总预算，同时报告准确率、真实选择后的成功率、尾延迟、总 token、tool/judge 成本和失败恢复率。否则“更强 reasoning”可能只是用更多 compute，或换了更强 verifier。[Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314)
