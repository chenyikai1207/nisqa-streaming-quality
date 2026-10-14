质量预测与部署成本的权衡

Trade-off between quality prediction and deployment cost

左列为含前端计算量估计，右列为Windows CPU单能效核实测实时因子（RTF），横轴为对数尺度；三行依次为VAL、三测试集等权宏平均及TCD-VoIP。纵轴为原始Pearson相关。右列点均通过验收，横线为五轮RTF的IQR。验收标准：离散度（IQR/中位数）<10%、校准漂移<11%。质量来自GPU推理，成本来自Windows CPU测量。双层GRU64仅显示已完成的VAL评价。MAC列的NISQA五维与仅MOS位置接近，分别以黑色方块和灰色叉号显示。TCD官方MOS使用与成本测量相同的权重补齐384条预测；NISQA五维无同身份RTF，未绘对应点。虚线连接成本递增、质量严格改善的非支配点。
