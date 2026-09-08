## T25 · 6ND、MFU 与真正的系统效率

### 6 从哪里来

对一个 dense 线性层 $Y=XW$，乘加按 2 FLOPs 计，forward 约 $2n_{\rm tok}d_{\rm in}d_{\rm out}$；计算输入梯度和权重梯度各约相同成本，因此训练约为 forward 的三倍。对以矩阵权重为主的模型求和得到

$$
C_{\rm train}\simeq6ND.
$$

它是数量级预算模型：$N$ 为每 token 参与相应矩阵计算的参数量，$D$ 是实际处理 token。不是“每个参数总共只执行六次运算”。Embedding 输入查表不是完整词表乘法，但 LM head 通常是 dense 投影；绑定权重也不意味着两种运算成本相同。

注意力的 token-to-token 计算、softmax、norm、router、激活重计算和通信并未都精确包含。长上下文会让 attention 二次项更显著；MoE 用 active 参数估 FFN 计算，但 total 权重的加载/通信成本依然存在。需要精细预算时应按算子和序列长度核算，而不是拿 6ND 作为精确测量。

### MFU 不等于 GPU 看起来很忙

若吞吐为全系统每秒 $R$ token，GPU 数为 $G$，单 GPU 与实际 dtype 对应的峰值为 $P_{\rm peak}$：

$$
\operatorname{MFU}\simeq\frac{6NR}{GP_{\rm peak}}.
$$

MFU 的分子估算有用模型计算；HFU 的定义通常还计重计算等硬件实际执行的 FLOPs。Activation checkpointing 增加硬件工作，却未增加训练 token 的数学目标计算，可能提高 HFU 而不提高 MFU。GPU utilization 高也可能只是忙于低效率算子。

必须统一峰值口径：不能拿 FP8/sparsity 峰值作分母，却在另一系统用 BF16 dense 峰值。还要报告是否含 data loading、checkpoint、评测、故障恢复和编译预热。训练循环内速度与整个作业 wall-clock 是不同指标。

### 如何从指标回到工程决策

相同质量与工作量下，MFU 从 40% 到 50%，理想吞吐增加 25%，耗时下降 20%，两者不能混说。若为提高吞吐把 batch 或 packing 目标改了，还必须重新核对样本效率与质量。

把 step time 分解为 GEMM、attention、通信、pipeline bubble、数据等待和 checkpoint。改变序列长度、TP/PP/EP 或 checkpoint 策略后，一次只检验主要假设。不能用“MFU 低”独立判断网络差或 kernel 差；瓶颈证据来自 trace、算子吞吐与通信重叠。[PaLM 的效率口径](https://arxiv.org/abs/2204.02311)

## T26 · μP 与 μTransfer：可迁移的参数化，不是万能学习率

### 为什么宽度变大后同一超参数会失效

网络宽度变化会改变激活、梯度和函数更新的尺度。仅保持初始化时每层方差稳定，并不保证训练一步后的特征变化仍可比。μP（maximal update parameterization）联合设计初始化、学习率缩放和输出参数化，使宽度极限下各层保持合适尺度，并尽可能保留特征学习。

这里 maximal 不是“把每个权重更新成 $O(1)$”。单个权重更新可以随宽度缩小，但许多坐标汇总后给 hidden state 带来非退化改变。需要同时区分参数坐标尺度与函数变化尺度。

### 从一层的求和看问题

对宽度为 $n$ 的 hidden-to-hidden 层：

$$
h'_i=\sum_{j=1}^{n}W_{ij}h_j.
$$

初始化时若 $W_{ij}$ 独立、均值为零，选 $\operatorname{Var}(W_{ij})\propto1/n$ 可以控制 $h'_i$ 方差。但训练更新与 $h_j$ 相关，不能仍按独立随机求和计算：

$$
\Delta h'_i=\sum_j\Delta W_{ij}h_j.
$$

这正是只看 fan-in 初始化不足以决定 width-transfer 的原因。输入投影只有一条宽度轴；内部矩阵有两条；输出读出又把宽表示映到固定维度。它们应按参数类型处理，不能对所有层机械乘同一个 width multiplier。

### 与 NTK/lazy training 的区别

Lazy 极限里模型主要在初始特征附近调整，函数行为可由核近似。μP 旨在保留非平凡特征学习并使超参数随宽度更可预测。有限宽度实现仍需要具体约定：优化器、矩阵转置方向、embedding tying、输出缩放都会影响处方。不能脱离实现给一个“所有模型通用”的 LR 表。

### 如何验证自己真的用了 μP

选同一模型家族几个宽度，保持深度、数据和训练协议一致；设置 base shapes，核对输入/隐藏/输出参数分类。做 coordinate check：初始化与若干步后各层激活 RMS、更新引起的 $\Delta h$、logits 和 loss 是否随宽度稳定。

随后在小模型搜索 LR 等超参数，在保留的较大宽度验证最优区间是否迁移。只证明初始 loss 相同不够；只看权重 norm 也不够。Depth、context、batch、MoE 专家数或数据阶段变化不在“仅宽度迁移”的自动保证里。μTransfer 省的是特定家族的调参成本，不代替 scaling-law 验证。[Tensor Programs V](https://arxiv.org/abs/2203.03466)

## T27 · Pre-training、CPT 与 SFT：目标相似，数据角色不同

### 同一个交叉熵，训练出不同的条件行为

Pre-training 在广泛文本上学习 $p(x_t\mid x_{<t})$，建立语言、知识与模式基础。SFT 在指定任务输入与高质量输出上学习 $p(y\mid x)$，常只监督 assistant target：

$$
\mathcal L_{\rm SFT}
=-\mathbb E_{(x,y)\sim D}
\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

是否除长度、每轮是否等权会改变目标。技术区别不仅是“数据比较好”，还包括条件组织、目标位置、样本分布、停止符与评价方式。SFT 不只训练聊天，也可训练代码编辑、结构化输出、工具轨迹、拒绝边界、推理蒸馏和特定工作流。

### CPT 与 SFT 各补什么

CPT 常继续在领域文本上做语言建模，让模型熟悉术语、文体、符号和知识共现；SFT 直接规定在某输入下应执行什么任务。只有领域文档、没有可靠示范时，CPT 更自然；有优质输入—输出示范时，SFT 更直接。实际可先 CPT 再 SFT，也可通过混合减少遗忘，没有固定必须顺序。

SFT 可以获得新能力，不仅是“把已有知识叫出来”；但好的格式遵循不证明底层技能已提高。RL 也不是凭空创造知识的魔法，成败取决于探索、反馈与可学习结构。

### 遗忘是干扰，不是存储空间被字面占满

新域梯度可能改变旧任务依赖的共享表示与输出偏好。一个常见约束是 replay：

$$
\mathcal L=(1-\lambda)L_{\rm new}+\lambda L_{\rm general}.
$$

调低 LR、限制训练步数、保留多样旧任务、使用 adapter 或蒸馏约束各有取舍；不能保证完全不忘。专业域改善同时测通用语言、事实、校准、指令、安全和原有工具协议。

### 什么时候先 SFT，什么时候偏好优化/RL

若目标是明确格式或稳定工作流，已有高质量示范，先用 SFT 建立可执行 baseline。若难以写唯一标准答案，却能可靠比较两个结果，可考虑偏好优化。若有可扩展环境反馈，策略还需探索新的多步行为，且能承担 rollout、验证器与数据闭环成本，在线 RL 更有价值。

先问数据是什么、反馈是否可信、当前模型能否偶尔成功、独立评估是否存在。没有这些条件，换更复杂算法往往只是扩大不确定性。公开模型的多阶段 SFT/RL/rejection sampling 流程可以参考，但不能推成所有实验室相同的内部配置。[InstructGPT](https://arxiv.org/abs/2203.02155)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## T28 · LoRA 的梯度、表达能力与实际成本

### 把大矩阵更新约束成低秩

对 $W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$：

$$
W=W_0+sBA,\quad
B\in\mathbb R^{d_{\rm out}\times r},\quad
A\in\mathbb R^{r\times d_{\rm in}},\quad
s=\alpha/r.
$$

训练冻结 $W_0$，更新 A、B；新增参数为 $r(d_{\rm in}+d_{\rm out})$。每一层都可有自己的 adapter。它不是每 batch 新建一套矩阵，也不是推理时只运行一个 rank-$r$ 小模型：基座仍参与 forward。

### 从计算图推梯度

令输出 $y=W_0x+sBAx$，上游梯度 $g=\partial L/\partial y$：

$$
\nabla_BL=s\,g(Ax)^\top,\qquad
\nabla_AL=s\,B^\top gx^\top,\qquad
\nabla_xL=W_0^\top g+sA^\top B^\top g.
$$

常见初始化 A 随机、B 为零，初始 $\Delta W=0$；第一步 A 的梯度为零，B 通常非零，之后 A 才开始变化。把 A、B 都初始化为零会让二者任务梯度都为零。冻结 $W_0$ 只省掉 $\nabla_{W_0}L$，并不允许在有更早 adapter 的位置把 $\nabla_xL$ 也截断。

### “低秩”约束了什么

单个矩阵的更新 rank 不超过 $r$，不是整个模型的功能变化 rank 都不超过 $r$。多层低秩修改经非线性组合可以产生复杂行为；A、B 都训练时，其子空间也在变化，不是固定随机子空间。小 rank 不保证对所有任务足够，全量微调也不保证数据有限时一定更好。

常见目标包括 Q/K/V/O、MLP projections，而不是只改最后 unembedding。多层都加、只加 attention、只加部分矩阵，参数量与质量不同。新增 special token 时若 embedding 行被冻结，LoRA 不会自动把这些新行训练好。

### 显存与 FLOPs 不同比例下降

理想 dense 线性层 full tuning 约 forward 2、input backward 2、weight backward 2；冻结基座但保留输入反传后约剩 4，再加 adapter 运算：

$$
C_{\rm LoRA}\approx4ND+6N_{\rm adapter}D.
$$

这是线性层主导的粗略估算，不是固定实测比例。最早一段冻结网络若无需向任何更早可训练模块传梯度，可省更多；attention、checkpointing、反量化与 kernel 会改变整体成本。最大节省往往来自基座梯度和 optimizer states，而非训练计算按参数比例缩小。

普通 LoRA 可在推理前合并 $W_0+sBA$；量化基座的合并涉及反量化/重新量化和误差，不应声称完全免费。比较 full tuning、LoRA、QLoRA 要同时对齐质量、总计算、数据量和调参预算。[LoRA](https://arxiv.org/abs/2106.09685)

## T29 · Role Token、停止协议与 Tool-use 训练

### 特殊 token 的特殊之处在哪里

Special token 通常有保留 ID，tokenizer 保证它不被切成普通碎片；“Assistant:” 可能只是若干普通 token。两者都依赖训练建立语义，保留 ID 本身不保证服从角色边界。聊天协议还可能用“特殊分隔符 + 普通角色名”的组合，不能假定每个角色都独占一个 token。

BOS、EOS、end-of-turn、tool handoff、padding 各自语义不同。生成完一次 assistant 消息不一定意味着整个会话结束；tool call 后应把控制权交给 runtime，而不是让模型继续伪造工具输出。添加 token 需要更新 tokenizer、embedding 和输出头（若该 token 可生成），以及训练 target；只改模板不完成学习。

### 工具调用是一条交互轨迹

$$
s_t=(\text{指令、历史、工具 schema、观测}),\quad
a_t=(\text{工具名、参数或最终回答}),\quad
s_{t+1}=\operatorname{Env}(s_t,a_t).
$$

SFT 可以模仿正确工具选择、参数和读回结果后的行动；RL 则用最终任务结果或中间可验证反馈优化策略。ReAct 提供 reasoning/action/observation 交替的组织方式；现代训练可以使用结构化协议，不必照抄论文中的文字标签。[ReAct](https://arxiv.org/abs/2210.03629)

通常监督 assistant 生成的调用与答案，不监督工具观测文本本身；但观测作为后续条件仍参与计算。训练必须包含失败、空结果、格式错误、超时、重试和停止，而不只收集成功短轨迹。

### Schema 合法、工具正确、任务成功是三层评估

合法 JSON 只证明语法；工具名与参数对不对是语义；执行后是否实现用户目标才是任务成功。约束解码能保证某些格式，却不能保证一个日期、文件路径或参数值正确。未见过的工具可借 schema 泛化，但应通过新工具、新参数组合和多步环境验证。

工具观测包含外部数据，不因为来自工具就变成可信指令。模型应区分用户授权、系统协议和网页/文件内容；训练与执行层都要维护权限边界、敏感操作确认、预算和错误恢复。把所有 tool output 称为“可信执行真相”会混淆数据可信度与指令权限。

## T30 · 能力激发还是能力获得：怎样设计证据

### 先给可测定义，不争论不可见的“脑中本来有”

若 base model 在合理预算的提示、few-shot、采样和工具辅助下能完成任务，post-training 可能主要提高 elicitation、选择或执行可靠性。若新训练后在结构外推任务持续改善，且强 base elicitation 基线仍明显落后，则支持新的技能获得，但有限测试不能严格证明 base 绝对没有潜在能力。

$$
\mathcal E(\pi,\mathcal B)
=\max_{b\in\mathcal B}
\operatorname{Score}(\pi,\text{prompt/scaffold}_b).
$$

这是预算和允许策略集合 $\mathcal B$ 下的 operational envelope。改变预算就改变“已具备”的判定，所以必须声明比较条件。

### 设计一个二维对照

一轴是接口：格式、指令表述、示例、长度；另一轴是技能：新的算法组合、难度、领域和任务结构。若只换接口就恢复 base 表现，较像激发问题；若在格式统一后 post-trained 模型仍对新结构泛化，才更有理由讨论技能提升。

Base 的 pass@k 可以显示偶尔成功，却不意味着部署时能可靠选中成功项。Oracle selector 是上界；真实 verifier 的准确率、成本和偏差也要计入。Probe 能解码某信息不证明模型在任务中因果使用它，需要干预、ablation 或反事实测试。

### 哪些辅助分析有用

比较正确/错误轨迹的概率变化、关键决策 token 的 log-prob、采样多样性、长度匹配后的正确率和独立选择器效果。SFT 可能只把正确轨迹变常见，也可能学会新的中间计算；RL 可能改善探索或改变停止策略。单个成功例子、较长推理和 benchmark 提升都不足以独立回答机制问题。

最终应把结论写成“在这些接口、预算与外推分布上，观察到何种改进”，而不是“已经证明智能被创造”或“RL 只是把已会的东西喊出来”。
