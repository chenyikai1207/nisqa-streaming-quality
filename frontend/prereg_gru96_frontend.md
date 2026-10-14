# 双层 GRU96 前端比较预登记

本分析为**主模型冻结后追加的稳健性分析**。目的为检验整段峰值归一化与运行电平统计在较大容量下的 TRAIN 跨域误差关系，不用于选模型，不改变第一篇的冻结模型、VAL/TEST/TCD 成绩或登记。

2026-10-14 人类已明确配置编号：A=整段 peak、无四统计；B=TRAIN fixed gain、四统计；C=TRAIN fixed gain、无四统计；D=整段 peak、四统计。A/B 必须，C/D 建议；默认完整120次新拟合，先 A/B 60次，再 C/D 60次。所有配置使用同一新 runner；B 必须新拟合，绝不复用冻结模型旧预测。其他配置也默认全部 fresh，不自动接入旧 D。

结构保持 GRU96 双层、48 mel 输入、单向 GRU、注意力池化、五维输出、四维侧输入，98522 参数。A/C 的侧输入恒零，训练标准化后仍零；未使用的四个 head 输入列不传递任何信息，保留参数身份。它们数学上对输出的贡献恒零，不通过删除 head 列改变模型或随机初始化。B/D 四统计取归一化前原始波形的 RMS dBFS、active-frame level、clipping、nearzero；在池化后、head 前进入，均值/标准差仅以当前训练折拟合。D 的四统计不得取 peak 归一化后波形。

仅复用已存在、固定文件顺序的 packed causal cache：`peak16000_mel48` 与 `power16000_mel48`。实际两者 x 均11020×1199×48 float32、mask11020×1199 bool、side11020×8 float32；采用 power cache 的前四个原波形统计作为 B/D 相同侧信息。其他采样率/梅尔数/归一化前端不在本实验中虚构。整段 peak 是离线整段信息前端，不作为严格因果部署路径。

固定种子0/1/2；原 ref_group 分组五折；四个 SIM 来源留出（AusTalk、DNS、TSP、UKIRE），加 SIM→LIVE，共10划分单元/配置/种子。CV 最大60 epoch、patience8，以验证折 SIM/LIVE 等域 MOS MAE 早停；每配置每种子的四SIM留出与SIM→LIVE训练轮数取该配置该种子五CV最佳 epoch 的中位数。AdamW lr=.001、weight_decay=.01、cosine T_max60、batch128、private CUDA order RNG，五维损失权重1/.5/.5/.5/.5；目标沿既定 .7人工五维标签+.3固定官方NI5D TRAIN 输出，教师非cross-fit。不是纯人工标签训练。两CPU线程、CUDA训练、cuDNN benchmark及TF32沿容量搜索原政策。

全部报告折外五维 MAE 和 Pearson，重点 MOS/LOUD。CV 先分别计算 SIM/LIVE，再等域；SIM LOSO 各来源等权；SIM→LIVE 为全部 LIVE。各指标先按每种子计算再三种子等权。主比较 B−A，次比较 C−A、B−C、D−A，MAE 负值为改善、Pearson 正值为改善。ref_group 为重抽样单元，同一次抽样同步用于两配置和三种子，2000次；区间取2.5/97.5百分位，不将三个种子当作三个独立语音样本。各评价域/来源内重抽样，保留既定域/来源等权。bootstrap RNG20261014。失败/常量 Pearson、真实缺失与复现失败照实报告。

B 的三种子均值 CV 等域 MOS MAE 应复现0.4176，绝对偏差≤.002为管线检查。先完成 A/B，再运行后处理；失败则标注，保留全部预测并停止后续 C/D，不靠调超参数/种子或测试指标修复。该检查不是选择 A–D 的性能门。runner 的 C/D 入口必须读到实际 B 检查 PASS 方可启动。

`power` packed cache保存的是线性mel power，不是logmel；必须调用容量搜索原 `feature_store_v3.FeatureStore`，按当前训练折 raw_energy 拟合到−26dBFS的固定gain，先乘 gain²、floor1e−10、再log，然后进入同折特征标准化。peak cache已是逐音频peak后的logmel。两者不得混接为相同数值域。原缓存gain无需新增音频提取。

解释先报告数值、配对区间与方向；不在结果前新设“重要/主要来源”的效应阈值。正文归因措辞由root依既有三档定义与实际证据决定。

现有 `supplement_GRU96_normalization_20261014` 的 peak+同四原统计30新fit流程于本次指令前已启动，对应 D 的前端定义；它的旧协议还复用了30个旧 fixed-gain fit。该已有运行保持原身份，不冒称在本次预登记后启动，不自动复用为新 A/B/C/D 的任一 fresh fit。root将在完整fit边界处理排队，当前 runner 不停止它。

所有新输出仅在 O 独立 postfreeze 目录：runs、predictions、normalizers、STATE、LOG、metric与bootstrap表。旧第一篇目录仅作为输入，不写入其模型、质量表、ledger、LOG/status。未启动新fit时不写任何完成门。根三记录由root追加。成本测量已收口，本实验不启动计时/部署优化矩阵。

The later mean-prediction tier rule and B-minus-D request were received after results; see protocols/ANALYSIS_CHRONOLOGY.md. Original and revised statistical results are separately retained.
