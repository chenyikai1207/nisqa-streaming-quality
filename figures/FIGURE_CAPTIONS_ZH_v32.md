v32四主图图注与排版接口

fig1 本文模型的因果流式质量预测框架

Causal streaming architecture of the proposed quality model

训练侧固定增益和对数梅尔谱前端连接双层单向GRU96与增量注意力池化，预测总体质量及噪声、音色、不连续性、响度。归一化前4项原始电平统计独立累积，在池化后、输出头前拼接。各模块保留有界状态；虚线为独立时间解释支路。

fig2 不同方法的质量预测性能

Quality prediction performance of different methods

(a) 原始Pearson相关；(b) 训练侧映射的总体质量RMSE，官方NISQA五维按既定协议保留原值RMSE。首版为8070参数蒸馏GRU，本文模型为冻结双层GRU96。点形区分数据集，线段为已有95%区间，模拟域与真实域等权。UTMOSv2没有TRAIN映射，因此(b)无点；其已有原始相关仅覆盖VAL。

fig3 七种方法的原值总体质量均方误差分解

Raw overall-quality MSE decomposition for seven methods

(a) VAL；(b) FOR；(c) P501；(d) LIVETALK。七方法依次为首版、本文模型、NISQA五维、PLCMOS、DNSMOS、SQUIM与P.563。按逐文件、模拟域与真实域等权显示均值偏移平方、相关调整尺度失配平方和线性相关残差；误差线为已有总MSE的95%区间。新增本文模型VAL柱为冻结后对既有预测所做的一次授权描述性分解：2700文件、2602参照组，2000次ref_group重抽样；测试集沿用原condition_id区间。

fig4 质量预测与部署成本的权衡

Trade-off between quality prediction and deployment cost

左列为含前端计算量估计，右列为Windows CPU单能效核实测RTF，均为对数横轴；三行依次为VAL、三测试集等权宏平均及TCD-VoIP。纵轴为原始Pearson相关。右列实心点为已验收环境，空心†为保留的未验收测量，横线为五轮RTF的IQR。质量来自GPU推理，成本来自Windows CPU测量；不同模型的质量与成本保持同模型对应。双层GRU64只显示已完成的VAL评价。

图1几何与v31相同，可沿用已完成原生VSDX；图2–4按174mm双栏通栏宽度绘出，坐标文字8.5pt、标题9.5pt；图1保留180mm几何与至少8pt文字。Word按原宽插入，避免缩成半宽单栏。PNG600dpi，SVG文字可编辑，PDF嵌入TrueType。

VAL源为postfreeze/firstpaper_VAL_MSE_20261014/VAL_main_raw_MOS_MSE_decomposition.csv，模型v2_pure/rawMOS/file/EQUAL_DOMAIN，MSE0.3114245893，三分量0.00518020034/0.00487377063/0.30137061833。该柱替代v31缺项标记，v31保留历史。未由本绘图任务新算任何科学统计。


2026-10-14T16:10:19.811029 紧凑版布局接管前文尺寸说明：图1 172×58mm、图2 174×92mm、图3 174×106mm、图4 174×138mm。仅版式调整，不改数据。图内重复页眉/页脚说明已移除，Word沿用本文件图注。图1同节点/同接线但几何压缩，旧原生VSDX保留为可编辑源；最终版先使用当前SVG，或按新JSON重建native，不再将旧94mm高几何作为最终图。四图按上述物理尺寸插入，无需缩小文字。
