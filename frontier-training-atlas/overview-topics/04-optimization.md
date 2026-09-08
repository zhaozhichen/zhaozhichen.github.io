## T19 · MoE 的容量、稀疏性与“去掉 Top-k”

### 专家通常不是 attention head

常见 MoE 保留共享 attention，把某些层的 FFN 替换为多个专家。每个 token 根据当前 hidden state 选择少量专家，输出加权合并。专家可以是 SwiGLU：

$$
E_i(x)=W_{{\rm down},i}
\left[\operatorname{SiLU}(W_{{\rm gate},i}x)
\odot W_{{\rm up},i}x\right],
\quad
y(x)=\sum_{i\in S(x)}g_i(x)E_i(x).
$$

不同层通常有各自 router；一个 token 到下一层表示变了，会重新路由。第 1 层的专家 3 与第 20 层的专家 3 没有天然语义对应。是否每层都放 MoE、是否共享 router 是架构选择，不是 MoE 定义规定的必然。

### Total parameters 与 active parameters

若共享部分有 $N_s$ 参数，每个 routed expert 有 $N_e$ 参数，共 $E$ 个，每 token 选 $k$ 个，则概略

$$
N_{\rm total}\simeq N_s+EN_e,\qquad
N_{\rm active}\simeq N_s+kN_e.
$$

Shared experts 每个 token 都经过，应计入共享成本。还需计 router、实际层比例和输出层。扩大 $E$ 可增加容量而不同比例增加每 token 的 FFN FLOPs，但全体权重仍需可访问；expert parallel 的 all-to-all、负载偏斜和小矩阵效率会影响吞吐。Active 参数相同，不保证与 dense 模型相同质量或延迟。

MoE 不是简单把一个已经训练好的 dense FFN 切成互不相干的小专家就能无损变强。Dense-to-MoE 可以是初始化策略，但要重新训练路由和专家分工。自然形成的专业化也不必与人定义的领域一一对应。

### 不做 Top-k 是否退回普通 Transformer

若激活全部专家：

$$
y(x)=\sum_{i=1}^{E}g_i(x)E_i(x),
$$

得到 dense mixture。动态输入相关的 $g_i(x)$ 仍存在，通常不等于原宽度的单个 FFN，并失去稀疏计算优势。但原讨论说“只有去除所有非线性才可合并”也过强：若 $g_i=c_i$ 固定，可以把各专家第一层竖向拼接、第二层横向拼接，构成更宽的普通非线性 FFN：

$$
W_1=\begin{bmatrix}W_{1,1}\\ \vdots\\W_{1,E}\end{bmatrix},
\qquad
W_2=\begin{bmatrix}c_1W_{2,1}&\cdots&c_EW_{2,E}\end{bmatrix}.
$$

于是 $W_2\sigma(W_1x)=\sum_i c_iE_i(x)$。需要去掉的是动态混合这一条件，而不必去掉逐元素激活；代价是更大的宽度。

Attention 的 gate 在 token 轴上选择上下文；SwiGLU 在特征轴上调制通道；MoE 在专家轴上选择子网络；LM head 的 softmax 给符号分类。都涉及加权，却不在同一个轴上，也不承担相同角色。[Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)、[Switch Transformer](https://arxiv.org/abs/2101.03961)

## T20 · Top-k 不可导，Router 怎么学

### 离散索引固定，连续权重仍可求导

令 router logits 为 $z=W_rx$，选中集合 $S=\operatorname{TopK}(z)$。常见实现把排序索引当作当前前向决定的路径，反向不对索引求导。若选中 logits 重新 softmax：

$$
g_i=\frac{e^{z_i}}{\sum_{j\in S}e^{z_j}},\qquad
a_i=\left(\frac{\partial\mathcal L}{\partial y}\right)^\top E_i(x),
\qquad
\frac{\partial\mathcal L}{\partial z_i}
=g_i\left(a_i-\sum_{j\in S}g_ja_j\right),\ i\in S.
$$

直觉上，router 比较被选专家输出对当前 loss 的局部贡献，调整相对权重。该梯度在选中集合不变的区域内有效；跨越排序边界时函数可能不光滑。不能把它理解为训练时知道所有未计算专家的反事实表现。

### Top-1 的重要特例

如果只选一个专家，并在选中集合内归一化，$g=1$ 恒定，上面的主任务 router 梯度为零。因此并非“Top-1 自然照着同样 softmax 公式也能学”。Switch 类方法可保留**完整 softmax 中被选专家的概率**作为乘子，辅以负载均衡目标；不同路由实现须分别追踪计算图。

在选中集合重新归一化时，未选专家通常没有该 token 的任务梯度；在保留完整 softmax 权重时，归一化分母还可让未选 logits 收到梯度，虽然未选专家网络本身没有执行。Router 梯度与 expert 参数梯度不是同一件事。

### 为什么会塌缩，如何诊断

初期几个专家稍好，就吸引更多 token、获得更多训练，其他专家缺少机会。常用辅助项

$$
\mathcal L_{\rm bal}=\alpha E\sum_i f_iP_i,
\qquad P_i=\frac1B\sum_{x\in B}p_i(x),
$$

$f_i$ 是实际分配频率，常作 stop-gradient；$P_i$ 由完整 softmax 可导。它鼓励均衡，但不保证每列每步都有非零梯度，也不保证专家语义多样性。噪声探索、router z-loss、容量管理、无辅助损失的偏置校正等是不同手段，不能混为一个配方。

每专家容量常设为 $\lceil cBk/E\rceil$。超过容量可丢弃、重路由或使用 dropless 实现，各自影响 objective 与通信。监控 token load、gate mass、drop rate、专家梯度范数、路由熵、每域路由和 all-to-all 时间；总体 loss 正常也可能掩盖某语言持续被 drop。

Soft MoE、STE、Gumbel 松弛另有取舍，不是标准 hard Top-k 的必要组成。最小验证可固定一个两专家样本手算 gate 梯度，再分别测 top-1 重新归一化与保留原概率，确认零梯度特例。

## T21 · AdamW：状态、裁剪与可恢复的更新

### AdamW 不只是一句“自适应学习率”

对当前完整更新的梯度 $g_t$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

平方与除法逐元素进行。一阶矩平滑方向，二阶矩调整坐标尺度，bias correction 处理零初始化造成的偏小。$\epsilon$ 不只是可忽略的小数：在梯度很小时它会改变有效更新。具体实现对 epsilon 放置、低精度存储和 fused kernel 的约定应核对。

Weight decay 与梯度项分开，区别于把 $\lambda\theta$ 塞进 Adam 的梯度再做自适应缩放。对 norm 的可学习 scale/bias、普通 bias 是否 decay 是配方选择；这里“norm 参数”指 LayerNorm/RMSNorm 的参数，不是某个向量范数。

### 一次可靠更新的顺序

累积 loss/梯度并按目标归一化；若用了 loss scaling，先 unscale；在正确的并行分片范围内得到 global grad norm；检查有限性并裁剪；再更新 moments 与权重；最后推进对应成功更新的 scheduler。若 overflow 跳过 optimizer step，要确认 scheduler、计步和日志也一致。

$$
g\leftarrow g\min\left(1,\frac{c}{\|g\|_2+\varepsilon}\right).
$$

这是全参数梯度范数裁剪，不能每个 micro-batch 分别裁完再说等价；也不能每张 TP/ZeRO 分片只按局部范数裁剪。裁剪频率和裁剪前范数比裁剪后恒定的范数更能诊断问题。

### Loss spike 的证据顺序

先定位异常 batch、域、长度、token/label/mask 与数据变更，再检查 LR、grad norm、update-to-weight ratio 和 optimizer 状态恢复；随后检查 logits/激活是否溢出、低精度算子及分布式一致性。固定 checkpoint 和 batch 复现比盲调 LR 更有信息。

若同一 batch 在 FP32 单卡正确、低精度多卡异常，才支持数值/通信方向；若各种实现都在同一文档 spike，则优先查数据，但不能据此完全排除模型难例。保存数据游标、RNG、optimizer/scheduler/scaler 状态与并行配置，才能可复现恢复；只加载模型权重不等于接续训练。

AdamW 是重要基线，不是所有实验室唯一选择。Adafactor 压缩二阶矩，矩阵型优化器改变预条件方式；更换优化器应在相同预算和调参强度下比较，而不是只用一次默认配置判断。[AdamW](https://arxiv.org/abs/1711.05101)

## T22 · Warmup、Cosine、WSD 与数据退火

### 学习率 schedule 控制更新幅度

常见线性 warmup 是 $\eta_t=\eta_{\max}t/T_w$。Warmup 后的 cosine：

$$
\eta(t)=\eta_{\min}
+\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right].
$$

它的下降速度在中段最大，两端导数为零，不能描述成“越到末尾下降越陡”。WSD 将 warmup、stable、decay 分开，便于在未知总预算时从稳定阶段分叉出不同 cooldown；仍需验证稳定段是否适合当前 batch 和数据。

Warmup 可缓解初始激活尺度、optimizer moments 和梯度分布尚未稳定时的大更新。早期 batch 的直接步长较小，却影响表示和 moment，不能说“几乎白训练”。也不应专门用垃圾数据 warmup；广覆盖、干净的数据与低价值数据不是同义词。

### 数据退火改变目标，不是另一种 LR

若 $\alpha_k(t)$ 随训练后期调整，目标变为

$$
\mathcal L_t(\theta)=\sum_k\alpha_k(t)L_k(\theta).
$$

LR 主要缩放更新；mixture 改变期望梯度方向和目标分布。更多高质量数据可能有帮助，但“高质量一定梯度方差更小”不成立：难题、长上下文与稀有领域可能更不稳定。

一个简单四格消融分别使用原/新 LR 和原/新 mixture，区分优化收尾与数据选择的贡献。固定 token、初始化与总预算，并报告一般能力、目标领域和遗忘。只把“新数据 + 更低 LR”与原配方比较，不能归因于任一单变量。

### 多语言、多模态的阶段不是通用时间表

多语言通常需要足够早、足够持续的覆盖，避免末期才加入时 tokenizer/表示与知识基础均不足。多模态可以从文本基座做对齐，也可以早期联合训练。扩大 context 会改变每步成本、位置分布与 batch；把它与混合比例和 LR 同时改动，必须保留可诊断的中间检查点。

CPT、长上下文扩展与末期 cooldown 可能相邻，却解决不同问题。日志至少分别记录 processed tokens、learning rate、数据比例、长度分布和有效 target 数，避免“同一步号”掩盖真正训练量差别。

## T23 · Global Batch、Gradient Noise 与时间最优

### 大 batch 降噪，但减少固定 token 下的更新次数

数据并行副本数 $W$、每卡 micro-batch $b$、累积次数 $a$ 时，样本级 global batch 为 $B=Wba$。TP/PP 使用的设备不应再乘一次；变长语言训练还要以有效 token 数衡量 batch。

若样本梯度独立同分布、协方差为 $\Sigma$：

$$
\mathbb E[g_B]=g,\qquad
\operatorname{Cov}(g_B)=\frac{\Sigma}{B}.
$$

同文档 token 和重复样本相关，不能把每个 token 都当独立观测。增加 accumulation 提高 global batch，不会让单次 forward 的小矩阵自动拥有更高硬件利用率。

固定 $D$ token 时，更新次数大致 $D/B$。大 batch 的梯度更精确，但参数调整机会更少；小 batch 噪声大，却可能样本效率更好。调整 LR 可以缓解但不是万能补偿，线性或平方根 scaling rule 只在特定区间近似有效。

### Critical batch 与 gradient noise scale

常用直觉量 $\operatorname{tr}\Sigma/\|g\|^2$ 衡量噪声相对信号，但临界 batch 还受曲率、训练阶段、优化器与目标 loss 影响。它不是任何任务都存在的一个固定 token 数。应在多个训练位置测量，而不是仅凭开头几百步推断全程。

观察达到同一质量所需的 token 与 optimizer steps。若 batch 增大 4 倍，steps 只减少 2 倍，则所需 token 翻倍，已经付出样本效率代价。对未达到相同 loss 的两个终点比较吞吐，会混淆优化效果与系统效率。

### 什么情况下仍值得增大

令大 batch 的吞吐加速为 $S$，达到同质量所需 token 增加倍率为 $\rho$：

$$
\frac{T_{\rm small}}{T_{\rm large}}\simeq\frac{S}{\rho}.
$$

只有 $S>\rho$，端到端时间才缩短。更一般地

$$
T(B)=\frac{D_{\rm target}(B)}{R(B)};
$$

内点时间最优要求两者对 $B$ 的对数斜率相等。吞吐已经饱和时，继续增大 batch 通常没有系统收益，却仍可能增加数据需求。

这个分析不意味着小 batch 永远更便宜：多机通信、pipeline bubble、矩阵尺寸和显存约束也重要。应报告质量—tokens—时间三条曲线，并对新 batch 重新调 LR/warmup；“当前行业最优 batch 是几百万 token”不是可移植答案。[Gradient Noise Scale](https://arxiv.org/abs/1812.06162)

## T24 · FP16、BF16、量化与 QLoRA 的数值边界

### 动态范围与精度是两个轴

| 格式 | 指数位 | 尾数显式位 | 主要取舍 |
| --- | --- | --- | --- |
| FP32 | 8 | 23 | 范围宽、局部精度高 |
| FP16 | 5 | 10 | 局部精度较好，但范围窄 |
| BF16 | 8 | 7 | 接近 FP32 范围，但舍入更粗 |

FP16 最大有限值约 65504，最小正常数约 $6.10\times10^{-5}$；BF16 最小正常数约 $1.18\times10^{-38}$。Subnormal 又是另一段范围，具体硬件还可能 flush-to-zero。BF16 比 FP16 尾数少，不代表它更容易因指数范围不足而 underflow。

在 1 附近，BF16 间距约 $2^{-7}$，很小的更新加到权重上可能被舍入掉；这是吸收/舍入，不是把这个小数本身 underflow 到零。混合精度因此可能让矩阵乘法用低精度，归约、master weight 和 optimizer states 用较高精度；实际 dtype 必须逐张量检查。

### Loss scaling 在哪里起作用

将 loss 乘 $s$，反传得到 $sg$，更新前除以 $s$。这帮助 FP16 小梯度保持在可表示范围，但也可能使大值溢出。动态 scaler 检测非有限梯度后跳过更新并减小尺度。裁剪必须在 unscale 后；BF16 通常不需 FP16 式 scaler，但仍需监控 overflow、舍入和不稳定算子。

### 量化不是“把所有浮点改成四位整数”

均匀 affine quantization 的一个形式是

$$
q=\operatorname{clip}\!\left(\operatorname{round}(w/s)+z,q_{\min},q_{\max}\right),
\qquad \widehat w=s(q-z).
$$

$s$ 是 scale，$z$ 是 zero point。PTQ 在训练后校准；QAT 在训练中模拟量化，round 的梯度常用 STE 近似；QLoRA 冻结量化基座，训练高精度低秩适配器。NF4 是非均匀码本，不应硬套成上面的均匀整数间隔。Weight-only 四位不等于激活、KV、optimizer 或计算全部四位。

按 group 量化需计元数据：

$$
b_{\rm eff}=b+\frac{b_s+b_z}{G}.
$$

例如每 128 个四位权重共用一个 16-bit scale、没有 zero point，理想平均是 4.125 bits/weight，尚未计 padding、索引和其他结构。更小 group 通常减小局部动态范围误差，但增元数据与 kernel 成本；double quantization 进一步压 scale，不是免费无误差。

评估应分别看校准域外、长上下文、离群通道、生成质量、实际峰值内存和吞吐。有的压缩节省显存却增加反量化开销。冻结基座减少其梯度与 optimizer 存储，但反传仍可能经过基座运算，为前面 adapter 计算梯度。[Mixed Precision Training](https://arxiv.org/abs/1710.03740)、[QLoRA](https://arxiv.org/abs/2305.14314)
