# 面向流式部署的轻量无参考语音质量评估

作者：　　　　　　　单位：　　　　　　　通信作者：　　　　　　　基金：

中图分类号：TN912.3；TP18　文献标志码：A

## 摘要

针对通信语音质量监测的计算与状态内存约束，构建单向双层门控循环单元（gated recurrent unit，GRU）与增量注意力池化组成的轻量模型，同时预测平均主观意见分（mean opinion score，MOS）及噪声、音色、不连续性和响度。模型采用训练侧固定增益保留输入电平，以因果谱前端、跨块循环状态和数值稳定的增量池化持续接收语音，独立输出五维评分。在NISQA验证集上，模型的原始Pearson为{{paper.v2.val.r}}，高于PLCMOS的{{paper.val.PLCMOS.r}}、UTMOSv2的{{paper.val.UTMOSv2.r}}、DNSMOS的{{paper.val.DNSMOS.r}}及SQUIM的{{paper.val.SQUIM.r}}；在FOR与TCD-VoIP中，对PLCMOS的训练映射RMSE配对差区间跨零，训练侧选择与跨来源评价分离，相关性使用原始预测，误差指标采用冻结的训练校准。模型含前端原生计算量为{{D.cost.mac.v2_96_native}} MMACeq/音频秒，不足官方NISQA MOS单头的{{paper.mac.native.NISQA.conservative_fraction}}、约为PLCMOS的{{paper.mac.native.PLCMOS.conservative_fraction}}；部署计算量为{{D.cost.mac.v2_96_deploy}} MMACeq/音频秒，精简流式进程常驻内存为{{paper.stream.rss}} MB，16/48 kHz输入的状态上界为{{D.cost.state_bound.v2_pure.state_kB_upper}} kB，10 ms分块计算中位数和P99为{{paper.stream.median}} ms及{{paper.stream.p99}} ms。三种子逐文件预测平均的前端比较显示，仅以训练侧固定增益替代整段峰值归一化，模拟到真实的MOS MAE由{{front96.A.SIM_TO_LIVE.mos.MAE}}降至{{front96.C.SIM_TO_LIVE.mos.MAE}}，响度MAE由{{front96.A.SIM_TO_LIVE.loud.MAE}}降至{{front96.C.SIM_TO_LIVE.loud.MAE}}；保留峰值处理并另行提供原始电平统计也降低误差，两条路径支持绝对电平信息可得性的作用，其中固定增益路径适合流式实现。已知网络参数的合成探针显示，丢包线索可观测性随补偿方式变化，学习检测特征在不连续性预测中具有增量。真实通话来源仍呈现较大的泛化误差，误差分解及丢包响应显示线性相关信息和单调性仍有提升空间。

关键词：无参考语音质量评估；轻量模型；流式推理；跨域泛化；丢包可观测性

## Lightweight Non-Intrusive Speech Quality Assessment for Streaming Deployment

Authors:                         Affiliations:

Abstract: The estimator uses a unidirectional two-layer gated recurrent unit (GRU) and incremental attention pooling to predict mean opinion score (MOS), noisiness, coloration, discontinuity, and loudness under computation and state-memory constraints. The model retains input levels with a training-fitted fixed gain and continuously processes speech through a causal spectral frontend, cross-block recurrent states, and numerically stable incremental pooling. On NISQA VAL, the model achieves a raw Pearson correlation of {{paper.v2.val.r}}, exceeding PLCMOS ({{paper.val.PLCMOS.r}}), UTMOSv2 ({{paper.val.UTMOSv2.r}}), DNSMOS ({{paper.val.DNSMOS.r}}), and SQUIM ({{paper.val.SQUIM.r}}); its paired TRAIN-mapped RMSE difference intervals against PLCMOS cross zero on FOR and TCD-VoIP; training-side selection remains separate from cross-source evaluation, correlations use raw predictions, and error metrics use frozen training-side calibration. Including the frontend, the native path requires {{D.cost.mac.v2_96_native}} MMACeq per audio second, less than {{paper.mac.native.NISQA.conservative_fraction}} of official NISQA MOS and approximately {{paper.mac.native.PLCMOS.conservative_fraction}} of PLCMOS; the deployment path requires {{D.cost.mac.v2_96_deploy}} MMACeq per audio second, uses {{paper.stream.rss}} MB of resident memory in a compact streaming process, retains at most {{D.cost.state_bound.v2_pure.state_kB_upper}} kB for 16/48 kHz inputs, and computes 10 ms blocks with median and P99 runtimes of {{paper.stream.median}} and {{paper.stream.p99}} ms. The frontend comparison first averages file predictions across three seeds and shows that replacing whole-utterance peak normalization with a training-fitted fixed gain alone reduces simulation-to-real MOS MAE from {{front96.A.SIM_TO_LIVE.mos.MAE}} to {{front96.C.SIM_TO_LIVE.mos.MAE}} and loudness MAE from {{front96.A.SIM_TO_LIVE.loud.MAE}} to {{front96.C.SIM_TO_LIVE.loud.MAE}}; supplying raw level statistics alongside peak normalization also reduces error, supporting availability of absolute-level information, while the fixed-gain path permits streaming implementation. Synthetic probes with known network parameters show that packet-loss cue observability varies with concealment method and that learned detection features improve discontinuity prediction. Real-call sources still exhibit substantial generalization error, and error decomposition and packet-loss responses indicate room for improvement in linear correlation and monotonicity.

Key words: non-intrusive speech quality assessment; lightweight model; streaming inference; cross-domain generalization; packet-loss observability

# 0 引言

编解码、丢包补偿、背景噪声和电平变化以不同方式影响通信语音听感。ITU-T P.800、P.808和P.835规范传统听评、众包听评及信号、噪声和总体质量的分维评价<sup>[1–3]</sup>。听评成本限制了持续监测，无参考模型因此需要从接收语音估计质量，并满足终端预算。NISQA提供总体质量及噪声、音色、不连续性、响度五维预测<sup>[4]</sup>；PLCMOS面向丢包补偿质量<sup>[5]</sup>，DNSMOS面向降噪质量<sup>[6–7]</sup>，SQUIM提供感知质量代理<sup>[8]</sup>。不同训练目标构成互补参照，部署成本则需覆盖重采样、谱前端和推理。

整段峰值归一化改变绝对电平信息，也引入未来输入依赖；质量变化还与缺损比例及连续组织有关。Gilbert–Elliott模型和E-model从状态转移与传输损伤描述错误聚集<sup>[9–11]</sup>。盲估计受到自然停顿和补偿算法影响，声学低能量段与真实丢包的关系需要受控检验。

本文有三项贡献：①比较整段峰值归一化、训练侧固定增益与原始电平统计，识别出跨域误差与绝对电平信息可得性的关系，并说明两种补偿路径中只有一种与流式推理相容；②构建轻量流式五维模型，联合跨集误差分解和含前端实际成本呈现部署权衡；③利用已知网络参数探针研究补偿方式对丢包线索可观测性的影响及其不连续性预测增量。

# 1 相关工作

P.563提供窄带无参考评价，PESQ采用参考信号<sup>[12–13]</sup>；MOSNet学习合成语音平均主观意见分（mean opinion score，MOS），NISQA学习通信五维质量<sup>[4,14]</sup>。UTMOS<sup>[15]</sup>和MOS-Bench关注跨域泛化<sup>[16]</sup>，FUSE-MOS研究融合及预测不确定性<sup>[17]</sup>；发音和说话人任务与通信质量目标不同<sup>[18–19]</sup>。DNSMOS OVRL/BAK及SQUIM PESQ保持总体、噪声和感知代理身份。

G.107/G.113/G.107.1的传输损伤估计依赖可确认参数<sup>[11,20,21]</sup>。本文以梯度提升树、可分离卷积和门控循环单元（gated recurrent unit，GRU）控制容量<sup>[22–24]</sup>，用AdamW及知识蒸馏训练学生<sup>[25–26]</sup>；轻量增强提供其他预算路线<sup>[27–28]</sup>。

# 2 方法

## 2.1 循环学生与五维监督

图1中GRU输入为16 kHz、48维对数梅尔谱，窗长20 ms、帧移10 ms。每折TRAIN拟合固定增益及标准化，四项电平统计取处理前波形；首版采用整段峰值归一化。实际带宽按来源记录，升采样不增加信息。

【FIG1】

循环输出h_t经注意力a_t及有效帧掩码池化，输出MOS、噪声（noisiness，NOI）、音色（coloration，COL）、不连续性（discontinuity，DIS）及响度（loudness，LOUD）。式（1）中yhat_d/y_d为预测/标签，MOS权重w_d=1，其余0.5；padding和缺失标签不计入。

$$ L_sup = sum_d w_d (yhat_d-y_d)^2. (1) $$

AdamW初始学习率10⁻³、余弦退火、最多60轮、训练内早停8轮。首版五种子、容量种子0/1/2，留出标签不参与早停；含前端乘加运算（multiply–accumulate，MAC）预算≤NISQA的1/10。

## 2.2 固定软目标与模型选择

式（2）以α=0.7混合真实标签y与官方NISQA教师y_teacher的平方损失L；部署由学生独立运行，候选使用相同前端。

$$ L_distill = alpha L(yhat,y) + (1-alpha)L(yhat,y_teacher). (2) $$

最终候选为双层GRU96、双层GRU64和卷积k5+GRU64，各完成三种子五折及来源留出。主模型以种子20261011、15次CV最佳轮数中位数拟合全TRAIN，种子0折外预测拟合映射；原值Pearson积矩相关与Spearman秩相关为主要相关指标。

AISHELL-1仿真及THCHS-30留出源用于受控丢包响应，与质量选模分别报告。

## 2.3 时间描述符及其顺序分解

时间支路在16 kHz峰值归一化波形上用20/40/80 ms帧、半帧移，活动判定含hangover及≤150 ms间隙连接；异常序列b_t的转移仅在活动段内累计。

式（3）中N为活动帧数、L_k为异常游程长度，μ/σ为游程起点间隔的均值/标准差；ρ、r、B表示异常占比、最长游程比例及间隔突发度。

$$ rho = sum_t b_t/N, r = max_k L_k/N, B = (sigma-mu)/(sigma+mu). (3) $$

N_ij为相邻状态i→j计数，式（4）的p/q采用Jeffreys伪计数；式（5）派生声学链BurstR/Ppl，与真实包链的对应由探针检验。

$$ p=(N01+0.5)/(N00+N01+1), q=(N10+0.5)/(N10+N11+1). (4) $$

$$ BurstR=1/(p+q), Ppl=p/(p+q). (5) $$

原T有12维，新T单尺度19维，含近零、谱距离及pYIN基频跳变统计<sup>[29]</sup>。

自然/乱序特征相关>0.99归为近似不变组，其余为顺序组。原T打乱通量、能量及平坦度序列，新T打乱活动范围内能量及对应分块；保留边际分布，以乱序TRAIN重拟合检验顺序贡献。

## 2.4 增量池化与流式前端

GRU跨块传递状态；式（6）维护注意力最大值m、分子u、分母z，以新块注意力a_j、输出h_j更新，池化为u/z，与同设备完整池化比较。

$$ m'=max(m,max_j a_j), u'=exp(m-m')u+sum_j exp(a_j-m')h_j, z'=exp(m-m')z+sum_j exp(a_j-m'). (6) $$

前端只保留下一窗样本及重采样状态。首版分块参照依赖已知整段峰值，本文去除该依赖，测量10/20/100 ms块；支持16/48 kHz，其他采样率需额外状态。

单层GRU的主要矩阵乘加量按式（7）计数，I与H分别为输入和隐藏维数，T为有效帧数。注意力、输出头、STFT、梅尔及重采样另计；非线性运算和FFT等价运算估计单独标注，总成本覆盖上述各部分。

$$ MAC_GRU=3T H(I+H). (7) $$

# 3 实验设置

## 3.1 数据与训练侧选择

误差采用平均绝对误差（mean absolute error，MAE）和均方根误差（root mean square error，RMSE）。表1列样本与分组：TRAIN/VAL条件逐文件唯一，TEST另有条件聚合；LIVETALK参考字段缺失，按文件回退。

【TABLE1】

五折按源参考或源语句分组，同一参考的降质版本保持在同折。四SIM来源留出分别留出AusTalk、DNS、TSP和UKIRE，训练保留LIVE；DNS来源留出仍可见DNS的LIVE源语句。SIM→LIVE仅用模拟域训练、真实域评价。结构选择采用0.4×CV等域MOS MAE+0.4×四SIM来源留出MAE+0.2×SIM→LIVE MAE，分差小于0.005选含前端MAC较低者；CV与来源留出改善均小于0.01时保留首版。三个候选在TRAIN选定后各评价一次VAL，冻结主模型各评价一次测试集，成绩不改变选择。

TCD-VoIP含{{data.external.n}}条和{{data.external.conditions}}个失真条件，其中chop类{{data.external.chop_n}}条、{{data.external.chop_conditions}}条件。其TSP2002四说话人录音池与NISQA部分TRAIN具有来源池及说话人重叠，提供跨条件、独立听评评价。官方NISQA权重开发涉及公开VAL的早停与选择，教师及蒸馏学生的VAL属于开发曝光范围。历史schema核对曾输出部分测试评分文本而未读对应音频；早期首版及其他模型随后实际完成测试评价。本轮选择仅用TRAIN，VAL/测试按已有曝光后的冻结模型作描述性评价；冻结后的GRU96前端四臂分析使用TRAIN，评价后收到的聚合与表述规则不参与选模。

## 3.2 评价与不确定性

MAE和RMSE衡量误差，Pearson与Spearman使用原始预测。域内分别计算后等权平均，三个NISQA测试集宏平均再按数据集等权。条件层先平均同条件标签与预测。学生的三阶校准用TRAIN折外预测拟合，PLCMOS等比较法用既定TRAIN预测拟合，非单调三阶退回一阶；官方NISQA保持原值。逐目标集线性映射只作目标标签尺度校正的次要分析，部署使用固定TRAIN映射<sup>[30]</sup>。

TRAIN/VAL按参考组、TEST按条件组作2000次同步配对bootstrap，报告95%百分位区间。每个数据集×文件/条件层面×MOS指标对首版、PLCMOS、NISQA三比较者共同作Holm校正，合计24族72项。前端四臂先平均种子0/1/2逐文件预测，再计算域/来源等权指标并作参考组分层bootstrap；该聚合规则在原拟合结果后收到，在聚合前固定，结果后B−D比较仅用于机制解释。20次乱序重拟合的经验概率为(k+1)/(n+1)，k为不劣于自然值的乱序次数，n=20。

## 3.3 对比模型与测量平台

首版（无蒸馏）指仅真实标签监督的8070参数GRU；首版指相同容量的蒸馏GRU；本文模型指冻结双层GRU96。NISQA采用作者官方五维权重，其MOS单头仅末层偏置与五维MOS头相差{{C.nisqa.bias_shift}}，单头只作成本实现参照。PLCMOS使用官方权重及作者重采样和15次调用；DNSMOS沿用官方16 kHz处理及OVRL输出；SQUIM采用官方16 kHz输入及PESQ代理；P.563在8 kHz输入上报告有效样本。UTMOSv2沿用作者五折权重、每折五次随机裁剪，提供VAL原值指标，无TRAIN映射。

质量训练使用RTX4090，成本平台为i7-13700KF。S1为不绑核的库默认线程，S2/S3分别为单性能核/单能效核单线程；固定600条VAL、热缓存、20文件预热和五个独立进程重复，分离读取、预处理、前端与推理，报告常驻内存（resident set size，RSS）及实时因子（real-time factor，RTF，处理耗时/音频时长）。基础应用背景受控，核心温度不可用；16设置中8项通过稳定性标准，失败项保留实测标记。完整2700条VAL核对代表性。流式在200个随机TRAIN文件上比较同设备完整与分块输出，最大绝对差门限10⁻⁵；CPU/CUDA差异另记。动态INT8在TRAIN折外检验，局部线性量化与GRU及线性量化分别报告。

# 4 结果与分析

## 4.1 电平信息、容量与跨集精度

绝对电平可以由训练侧固定增益保留，或由峰值归一化前的原始电平统计补入；第二条路径保留整段峰值依赖。早期约8千参数的前端选择中，固定增益与四统计组合将SIM→LIVE MOS MAE由{{early.frontend.mos.peak_simlive}}降至{{early.frontend.mos.gain_simlive}}，LOUD由{{early.frontend.loud.peak_simlive}}降至{{early.frontend.loud.gain_simlive}}，按3.1综合分确定前端。表2列出五个代表容量档。可比GRU系列的CV与来源留出误差随容量降低，双层64单元之后SIM→LIVE点值接近饱和；全部架构并不呈单调规律。双层GRU96的CV与来源留出MOS MAE为{{capacity.gru96_2.cv_mae_mean}}和{{capacity.gru96_2.loso_sim_mae_mean}}，相对首版改善{{capacity.gru96_2.gain_cv_mae_vs_v1}}和{{capacity.gru96_2.gain_loso_sim_mae_vs_v1}}，通过0.01止损规则。

【TABLE2】

冻结后在GRU96上比较四配置：A为峰值归一化且无四统计，B为固定增益加四统计，C为固定增益且无四统计，D为峰值归一化加原始四统计。表3给出A/B。SIM→LIVE MOS和LOUD的A−B改善分别为{{front96.A-B.SIM_TO_LIVE.mos.MAE.interval}}、{{front96.A-B.SIM_TO_LIVE.loud.MAE.interval}}；CV MOS区间跨零，四来源留出MOS区间为正。在约8千与9.9万参数两种容量下，以固定增益加运行电平统计替代整段峰值归一化均显著降低模拟到真实迁移误差，结论一致。

【TABLE3】

仅固定增益C−A在SIM→LIVE的MOS和LOUD MAE差为{{front96.C-A.SIM_TO_LIVE.mos.MAE.interval}}、{{front96.C-A.SIM_TO_LIVE.loud.MAE.interval}}；保留峰值的D−A MOS MAE差为{{front96.D-A.SIM_TO_LIVE.mos.MAE.interval}}、LOUD为{{front96.D-A.SIM_TO_LIVE.loud.MAE.interval}}，也降低两维误差，支持绝对电平信息保留的解释。固定增益下B−C的MOS区间在三设计均跨零，SIM→LIVE LOUD MAE差为{{front96.B-C.SIM_TO_LIVE.loud.MAE.interval}}，DIS为{{front96.B-C.SIM_TO_LIVE.dis.MAE.interval}}、Pearson增量{{front96.B-C.SIM_TO_LIVE.dis.Pearson.interval}}，呈现维度权衡。C的SIM→LIVE MOS点值最低，B并非最优；B−D差{{front96.B-D.SIM_TO_LIVE.mos.MAE.interval}}跨零。B来自先前小容量综合分选择，C/D不在当时菜单，后见分析沿用冻结模型；D的整段峰值路径无法直接因果流式运行。

独立48 kHz高频屏蔽诊断中，屏蔽使CV MOS MAE增大{{paper.highfreq.cv_degradation}}，SIM→LIVE原始减屏蔽的MAE差为{{paper.highfreq.simlive.interval}}、区间跨零，带宽信息收益因域而异。

本文模型VAL原始Pearson为{{paper.v2.val.r}}，训练映射RMSE为{{paper.v2.val.rmse}}；表4及图2比较各来源精度。FOR与TCD对PLCMOS的训练映射RMSE差分别为{{paper.paired.FOR.PLCMOS.RMSE.interval}}、{{paper.paired.TCD_VOIP.PLCMOS.RMSE.interval}}，均跨零；FOR Pearson差{{paper.paired.FOR.PLCMOS.Pearson.interval}}虽不跨零，Holm为{{paper.paired.FOR.PLCMOS.Pearson.holm}}。P501对PLCMOS的Pearson差{{paper.paired.P501.PLCMOS.Pearson.interval}}，Holm为{{paper.paired.P501.PLCMOS.Pearson.holm}}，保留显著劣势。

【TABLE4】

【FIG2】

FOR训练映射RMSE由首版{{paper.v1.for.rmse}}降至{{paper.v2.for.rmse}}，P501由{{paper.v1.p501.rmse}}降至{{paper.v2.p501.rmse}}，LIVETALK由{{paper.v1.live.rmse}}增至{{paper.v2.live.rmse}}。三集等权RMSE为{{paper.v2.macro.rmse}}，原始Pearson为{{paper.v2.macro.r}}；TCD对应{{paper.v2.tcd.rmse}}、{{paper.v2.tcd.r}}。表5限定MOS文件层，条件层使用同一分组协议。

【TABLE5】

图3用原值MOS预测分解MSE：均值偏移(μp−μy)²、相关调整尺度失配(σp−rσy)²与线性相关残差(1−r²)σy²。VAL按SIM/LIVE等权的总MSE为{{paper.mse.val.MSE}}，三分量为{{paper.mse.val.mean_bias_squared}}、{{paper.mse.val.scale_mismatch_squared}}及{{paper.mse.val.linear_correlation_residual}}；残差反映当前预测与标签的线性相关缺口，并非纯排序误差。LIVETALK相关残差为{{paper.mse.live.linear_correlation_residual}}，与SIM→LIVE容量收益饱和共同提示真实来源差距。UTMOSv2的图2 RMSE点因无TRAIN映射省略，表4另列其原值RMSE。

【FIG_MSE】

## 4.2 时间顺序与检测特征增量

原T的20次乱序重拟合均劣于自然值（经验概率{{B.shuffle.oldT.primary.p}}）。原T+新T等域MOS MAE为{{B.combo.mae}}，对原T下降{{B.combo.gain_vs_oldT_ci}}；按source/codec/工作模式/FER的0.05分箱的247格、2123文件调和加权，随机丢包收益更集中。44维=14维G0+12维原T+18维新T，45维仅增加相邻有声帧>4半音的pYIN跳变率；历史种子20261011。检测比较用seed0/1/2同容量HGB，44维加入8维成为52维，同参考抽样并平均三种子绝对误差。DIS等域MAE下降{{B.learned.dis.EQUAL_DOMAIN.gain_ci}}，两域均为正（表6）；MOS等域下降{{B.learned.gain_equal_ci}}跨零，SIM改善、LIVE下降量{{B.learned.gain_live_ci}}为负。可定义覆盖SIM/LIVE为{{B0.SIM.defined_pct}}%/{{B0.LIVE.defined_pct}}%；移除与缺失处理仍保留LIVE负增量，固定置信加权缩至{{B0.confidence.LIVE_MOS_ci}}、跨零。

【TABLE6】

## 4.3 丢包补偿与可观测性

表7展示补偿方式对线索的影响。20 ms对齐DSP在零填充中利用明确缺损，在原生Opus中帧F1为{{B.dsp.opus.F1}}；突发度相关{{B.dsp.opus.rho}}仅对应{{B.dsp.opus.coverage}}可定义预测。固定CNN在两来源共20,000条仿真中，零填充log BurstR的Spearman由AISHELL的{{B2.heldout.test.zero.logBurstR_Spearman}}降至THCHS的{{B2.heldout.thchs.zero.logBurstR_Spearman}}，联合分母{{B2.heldout.test.zero.joint}}与{{B2.heldout.thchs.zero.joint}}；后者低于预定0.7线。独立因果波形CNN-GRU在原生Opus的帧F1为{{paper.cues.test.f1}}与{{paper.cues.thchs.f1}}，THCHS低于0.3线；阈值由TRAIN、结构由DEV确定。

【TABLE7】

100参考、29说话人的七条件中，五点丢包曲线四相邻差均≤0的完整非增比例为{{paper.mono.min}}–{{paper.mono.max}}，各条件低于NISQA五维及PLCMOS。10%丢包率、同参考同丢包数的BurstR 8−1响应采用说话人簇bootstrap 2000次；五固定模型多数预测突发MOS较高，方向随补偿变化。E-model仅匹配{{paper.emodel.n}}条G.711 20 ms、Repeat1/silence、目标BurstR 4/Ppl 1%的单网络条件，公式用BurstR=1，差异来自时长/丢包数舍入；其余{{paper.emodel.unavailable}}条缺少共同系数。

## 4.4 计算成本与流式运行

表8与图4联合精度和成本。本文模型含{{paper.main.parameters}}参数，原生与部署路径为{{D.cost.mac.v2_96_native}}、{{D.cost.mac.v2_96_deploy}} MMACeq/音频秒；官方NISQA MOS为{{D.cost.mac.NISQA_MOS_official}}，PLCMOS为{{D.cost.mac.PLCMOS_official}}。S2的首版、本文模型原生/部署与NISQA MOS四项测量通过标准，NISQA MOS/部署耗时比为{{D.cost.s2.nisqa_ratio}}，漂移扰动敏感性范围2.16–2.29；范围并非置信区间。S3部署RTF为{{D.cost.v2_96.deploy.S3.rtf_median}}且通过标准，NISQA与PLCMOS在允许复测后仍未通过、保留†；其IQR/中位数及最大校准漂移分别为{{paper.cost.NISQA_MOS.S3.iqr_pct}}%/{{paper.cost.NISQA_MOS.S3.drift_pct}}%和{{paper.cost.PLCMOS.S3.iqr_pct}}%/{{paper.cost.PLCMOS.S3.drift_pct}}%，阈值为离散度5%、校准漂移3%。PLCMOS实测耗时约慢一个数量级。MAC比约12而S2时间比约2，GRU逐帧小矩阵、前端占比和算子并行造成不同缩放；S1默认多线程部署RTF {{D.cost.v2_96.deploy.S1.rtf_median}}高于S2 {{D.cost.v2_96.deploy.S2.rtf_median}}，符合线程调度开销。DNSMOS按作者9.01 s短音重复、1 s离散滑窗并计固定STFT卷积，其旧TRAIN200时长加权估计与本次VAL代表长度口径不同。

【TABLE8】

整句S3峰值RSS为{{D.cost.v2_96.deploy.S3.peak_rss_mb}} MB，包含框架和音频缓冲；独立精简流式进程10 ms块RSS为{{paper.stream.rss}} MB，中位与P99耗时{{paper.stream.median}}、{{paper.stream.p99}} ms。10/20/100 ms块P99均低于块周期，20 ms块算法时延{{D.cost.stream20.algorithm_delay_ms}} ms、未来前瞻0。支持16/48 kHz的本文模型与GRU64状态上界为{{D.cost.state_bound.v2_pure.state_bytes_upper}}、{{D.cost.state_bound.v2_64.state_bytes_upper}} B，其他采样率需额外重采样状态。部署与同设备冻结输出最大差{{paper.deploy.max_error}}；动态INT8总体指标变化较小而个别预测误差较大，独立流式最大差{{D.int8.stream.error}}超出10⁻⁵门限，采用浮点部署。完整VAL代表性核对相对差3.94%/1.02%低于5%；这些桌面测量尚待嵌入式硬件验证。

【FIG4】

# 5 结论

固定增益和原始电平统计提供两条补偿绝对电平信息的路径，在两种容量中组合前端降低模拟到真实误差，固定增益路径适合因果流式实现。冻结双层GRU96以有限含前端运算和状态内存提供五维评估，跨来源精度与成本共同界定部署权衡。已知参数探针表明补偿方式改变丢包线索可观测性，学习检测在DIS有稳定增量，在MOS呈现域间差异。

真实通话来源的泛化、丢包响应单调性和盲突发度跨源误差仍有限；教师与历史测试曝光、TCD录音池重叠、仿真16 kHz原生带宽及E-model参数范围限定了评价解释。现有成本来自桌面平台，支持16/48 kHz输入。后续研究重点是缩小合成与无标签真实语音的分布差距。

数据与代码可用性：本文使用NISQA、TCD-VoIP、AISHELL-1和THCHS-30的公开研究数据。版本化评价协议、预测、完整结果表、成本分解和复现代码的存档地址为已建立的公开GitHub仓库https://github.com/chenyikai1207/nisqa-streaming-quality，本稿关联版本标签为nisqa-paper1-ecice-v32-20261014；原始音频按各数据集许可从作者来源获取。

# 参考文献

{{REFERENCES}}
