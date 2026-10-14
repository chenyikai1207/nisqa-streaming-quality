from pathlib import Path
import os
import pandas as pd,json
HERE=Path(__file__).resolve().parent;STAGE=Path(__file__).resolve().parent/'scientific_inputs/R4'
q=pd.read_csv(STAGE/'exp_C2/results/C_primary_accuracy_v3.csv');tables=[]
names={'A_G0':'G0 HGB','A_newT':'G0+新T','A_tiny':'GRU真实标签','A_distill':'GRU蒸馏v1','NISQA5D':'NISQA五维','DNSMOS':'DNSMOS OVRL','PLCMOS':'PLCMOS','SQUIM':'SQUIM PESQ代理','P563':'P.563'}
for model in q.model.unique():
    if str(model).lower().startswith('utmos'):names[model]='UTMOSv2'
completion=STAGE/'exp_C2/C3_C5_v3_COMPLETE.json'
diagnostic_model=None
if completion.exists():
    diagnostic_model=json.loads(completion.read_text(encoding='utf-8-sig'))['selected_v2']
    final_model=json.loads(completion.read_text(encoding='utf-8-sig')).get('final_main_model')
    selection=json.loads((STAGE/'exp_A/FROZEN_SELECTION_v3.json').read_text(encoding='utf-8-sig'))
    labels={'v2_pure':'双层GRU96','v2_64':'双层GRU64','v2_conv64':'卷积k5+GRU64'}
    for entry in selection['models']:
        names[entry['name']]=labels.get(entry['name'],entry['name'])
    names[diagnostic_model]='最终模型 v2' if final_model==diagnostic_model else '修订版（诊断）'
    if final_model=='v1':names['A_distill']='最终主模型 v1'
    elif final_model and final_model not in names:names[final_model]='最终主模型'
def value(model,ds,metric,mapping,domain='EQUAL_DOMAIN',protocol=None):
    x=q[q.model.eq(model)&q.dataset.eq(ds)&q.dim.eq('mos')&q.level.eq('file')&q.domain.eq(domain)&q.metric.eq(metric)&q.mapping.eq(mapping)]
    if protocol is not None:x=x[x.protocol.eq(protocol)]
    elif len(x)>1:
        preferred=x[x.protocol.eq('C_V2_RAW_OR_TRAIN_OOF_CALIBRATION')]
        if len(preferred)==1:x=preferred
        else:
            preferred=x[x.protocol.astype(str).str.contains('SECONDARY_TARGET_DOMAIN_LINEAR')]
            if len(preferred)==1:x=preferred
    return f'{x.iloc[0].value:.4f}' if len(x)==1 and pd.notna(x.iloc[0].value) else '—'
for ds,num,cn in [('VAL',4,'表4 VAL逐文件MOS结果'),('NISQA_TEST_FOR',5,'表5(a) FOR逐文件MOS结果'),('NISQA_TEST_P501',5,'表5(b) P501逐文件MOS结果'),('NISQA_TEST_LIVETALK',5,'表5(c) LIVETALK逐文件MOS结果'),('TCD_VOIP',5,'表5(d) TCD跨条件逐文件MOS结果')]:
    rows=[]
    for model,label in names.items():
        if ds!='VAL' and model=='A_distill' and completion.exists():model='v1_A_distill'
        x=q[q.model.eq(model)&q.dataset.eq(ds)]
        if x.empty:continue
        version='v2' if model.startswith('v2_') else 'v1' if model.startswith(('A_','v1_')) else '固定对比'
        opens=str(int(x.inference_openings_this_stage.max())) if 'inference_openings_this_stage' in x else '1'
        round_label='开发曝光' if ds=='VAL' else '2' if model==diagnostic_model else '1'
        calibrated='none' if model.startswith('NISQA') else 'cubic'
        rows.append([label,version,value(model,ds,'Pearson','none'),value(model,ds,'Spearman','none'),value(model,ds,'RMSE','none'),value(model,ds,'RMSE',calibrated),value(model,ds,'RMSE','SECONDARY_TARGET_DOMAIN_LINEAR')])
    if rows:tables.append(dict(number=num,cn=cn,en=cn.replace('表','Table ').replace('逐文件MOS结果',' file-level MOS results'),headers=['方法','版本','Pearson','Spearman','原值RMSE','训练映射','目标线性'],rows=rows,note='相关系数用原值；训练映射由TRAIN折外或TRAIN预测拟合。目标线性使用该评价集标签，仅作次要尺度校正口径；未发布的项保留空缺。'))
index=pd.read_csv(STAGE/'shared/data_index.csv');rows=[]
for db,g in index.groupby('db',sort=False):
    rows.append([db.replace('NISQA_',''),str(len(g)),str(g.ref_group.nunique()),str(g.condition_id.nunique()),'文件回退，非真实源参考' if db=='NISQA_TEST_LIVETALK' else '元数据源参考/源语句'])
tables.insert(0,dict(number=2,cn='表2 数据与分组',en='Table 2 Data and grouping',headers=['数据子集','文件','参考分组','条件','分组依据'],rows=rows,note='TRAIN/VAL条件逐文件唯一；LIVETALK的参考组数是回退文件数。'))
cost_tables=STAGE/'exp_D2/D_WORD_TABLE6_v3.json'
if cost_tables.exists():
    raw=cost_tables.read_bytes()
    try: decoded=raw.decode('utf-8-sig')
    except UnicodeDecodeError: decoded=raw.decode('gb18030')
    supplied=json.loads(decoded)
    if isinstance(supplied,dict):supplied=[supplied]
    if any(t['number']!=6 for t in supplied):raise ValueError('Cost owner table must be Table6')
    for t in supplied:
        t['cn']='表6 完成的整集CPU测量';t['note']='VAL2700，热缓存，3次整集预热、10次正式重复；有同机后台工作。MB为十进制。端到端中位数独立统计。'
        for r in t['rows']:
            if r[0]=='A_distill':r[0]='蒸馏GRU v1'
            if r[0]=='OfficialNISQA5D':r[0]='NISQA五维'
            r[0]={'v2_pure':'双层GRU96 v2','v2_64':'双层GRU64 v2','v2_conv64':'卷积k5+GRU64 v2'}.get(r[0],r[0])
            if str(r[1]).startswith('S2'):r[1]='性能核单线程'
            elif str(r[1]).startswith('S3'):r[1]='能效核单线程'
            elif str(r[1]).startswith('S1'):r[1]='全核'
    tables.extend(supplied)
# C publishes all frontend numbers; one selected development frontend and fixed v1.
front_csv=STAGE/'exp_C2/results/C_A1_frontend_word_v3.csv'
if front_csv.exists() and (STAGE/'exp_C2/C_A1_FRONTEND_COMPLETE_v3.json').exists():
    fronts=pd.read_csv(front_csv);chosen=fronts[fronts.selected_frontend_by_A.eq(True)]
    if len(chosen)!=1:raise ValueError('C selected frontend must be unique')
    r=chosen.iloc[0];rows=[]
    for dim,label in [('mos','MOS'),('loud','LOUD'),('col','COL')]:
        for setting in ['v1固定种子','所选前端']:
            vals=[]
            for metric in ['cv_mae','loso_sim_mae','sim_to_live_mae','score']:
                if setting=='所选前端':vals.append(f'{r[dim+"_"+metric+"_mean"]:.4f} ± {r[dim+"_"+metric+"_sd"]:.4f}')
                else:vals.append(f'{r[dim+"_baseline_"+metric]:.4f}')
            rows.append([label,setting,*vals])
    tables.insert(1,dict(number=3,cn='表3(a) 前端选择的训练侧比较',en='Table 3(a) Training-side comparison for frontend selection',headers=['维度','设置','CV MAE','四来源留出MAE','模拟→真实MAE','综合分'],rows=rows,note='所选前端为三种子均值±SD；v1为固定种子。SD表示训练变异；综合分权重见4.1。'))


# C matched three-seed learned detector table, separate from historical single-seed results.
if (STAGE/'exp_C2/C_B4_HGB_INCREMENT_COMPLETE_v3.json').exists():
    p=STAGE/'exp_C2/results/C_B4_HGB_increment_main_v3.csv';b4=pd.read_csv(p);rows=[]
    for model,label in [('G0_oldT_newT_no_pitch','原T+新T'),('G0_oldT_newT_B2_no_pitch','原T+新T+学习估计'),('G0_oldT_newT_B2_noise_no_pitch','原T+新T+高斯噪声')]:
        g=b4[b4.model.eq(model)&b4.domain.eq('EQUAL_DOMAIN')&b4.metric.eq('MAE')&b4.mapping.eq('none')];v=[]
        for dim in ['mos','noi','col','dis','loud']:
            x=g[g.dim.eq(dim)]
            if len(x)!=1:raise ValueError('Completed C B4 table requires exactly one dimension row')
            v.append(f'{x.iloc[0].value:.4f}')
        rows.append([label,str(int(g.iloc[0].total_feature_columns)),*v])
    pos=next(i for i,t in enumerate(tables) if t['number']==3)+1
    tables.insert(pos,dict(number=3,cn='表3(b) 学习式丢包估计的同容量增量比较',en='Table 3(b) Matched-capacity comparison of learned packet-loss estimates',headers=['特征组合','维数','MOS','NOI','COL','DIS','LOUD'],rows=rows,note='同容量HGB，TRAIN分组五折，种子0/1/2。先平均每文件的三种子绝对误差，再计算两域等权MAE；不含基频特征。G0的14维包含在各行中。'))


# v13 manuscript numbering follows first citation, owner source files retain original numbers.
for tab in tables:
    old=tab['number'];new={2:1,3:2 if '(a)' in tab['cn'] else 5,4:3,5:4,6:8}[old]
    tab['number']=new
    tab['cn']=tab['cn'].replace('表'+str(old),'表'+str(new),1)
    tab['en']=tab['en'].replace('Table '+str(old),'Table '+str(new),1).replace('Completed full-validation computational cost','Table 8 Completed full-validation computational cost')
    if new==5:
        tab['cn']='表5(a) 学习检测特征的同容量五维比较'
        tab['en']='Table 5(a) Five-dimensional matched-capacity comparison of learned detection features'
# Complete v4 active capacity grid, C-only accuracy and original D-derived cost.
p=STAGE/'exp_C2/results/C_v1p5_complete_method_increments_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_CAPACITY_ALL_660_COMPLETE_v4.json').exists():
    z=pd.read_csv(p)
    cost=pd.read_csv(STAGE/'exp_A/results_v3/A2_budget_screen_v3_full_precision.csv').set_index('candidate')
    active=pd.read_csv(STAGE/'exp_A/results_v3/A2_pareto_full_precision.csv')
    selected=json.loads((STAGE/'exp_A/A2_SELECTED_CAPACITY_v3.json').read_text(encoding='utf-8-sig'))['selected_capacity']['candidate']
    configs=[('A1V3_sr16000_mel48_global_gain_band0','v1.5：单层32','A2V3_gru32_layers1')]
    configs += [(f'A2V3_gru{w}_layers1',f'单层GRU {w}',f'A2V3_gru{w}_layers1') for w in [48,64,96,128]]
    configs += [(f'A2V3_gru{w}_layers2',f'双层GRU {w}',f'A2V3_gru{w}_layers2') for w in [48,64,96]]
    configs += [(f'A2V3_dsconvK{k}_gru{w}',f'卷积k{k}+GRU {w}',f'A2V3_dsconvK{k}_gru{w}') for k in [3,5] for w in [64,96]]
    configs += [(f'A2V3_tcn{w}',f'TCN {w}',f'A2V3_tcn{w}') for w in [32,48,64,96,128]]
    if {r[2] for r in configs} != set(active.candidate):raise ValueError('Full table must match actual active17 grid')
    rows=[]
    for candidate,label,costkey in configs:
        g=z[z.candidate.eq(candidate)&z.dim.eq('mos')]
        if len(g)!=1:raise ValueError('Complete capacity C selector not unique')
        r=g.iloc[0]
        if costkey==selected:label+='＊'
        rows.append([label,str(int(r.parameters)),f'{cost.loc[costkey,"macs_per_audio_second"]/1e6:.4f}',
                     *[f'{r[k+"_mean"]:.4f} ± {r[k+"_sd"]:.4f}' for k in ['cv_mae','loso_sim_mae','sim_to_live_mae']]])
    tables.append(dict(number=2,cn='表2(b) 预注册结构与容量比较',
        en='Table 2(b) Preregistered architecture and capacity comparisons',
        headers=['结构','参数','总MACeq/s\n百万','CV MAE','四来源留出MAE','模拟→真实MAE'],rows=rows,
        note='三种子均值±SD；同16k/48梅尔前端，训练侧固定增益及四项运行电平统计。＊为TRAIN综合分和成本规则选出的容量结构。五个早期配置及全部五维结果见补充材料。含前端MAC来自固定TRAIN200时长加权估计。'))
# Allhead C paired evidence.
p=STAGE/'exp_C2/results/C_B4_HGB_allhead_paired_v4.csv'
if p.exists():
    z=pd.read_csv(p);rows=[]
    for dim in ['mos','dis']:
        for dom,label in [('EQUAL_DOMAIN','两域等权'),('SIM','模拟域'),('LIVE','真实域')]:
            r=z[z.candidate.eq('G0_oldT_newT_B2_no_pitch')&z.comparator.eq('G0_oldT_newT_no_pitch')&z.dim.eq(dim)&z.domain.eq(dom)].iloc[0]
            rows.append([dim.upper(),label,f'{r.comparator_MAE:.4f}→{r.candidate_MAE:.4f}',f'{r.gain_comparator_minus_candidate:+.4f}',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]'])
    tables.append(dict(number=5,cn='表5(b) 检测特征的MOS与DIS配对增量',en='Table 5(b) Paired MOS and DIS increments from detection features',headers=['维度','域','MAE前→后','下降量','95%区间'],rows=rows,note='相对44维原T+新T，三种子逐文件绝对误差平均；参考组配对bootstrap 2000次。下降量为正表示误差降低。'))
# Existing temporal combination overall and harmonic support; C only.
p=STAGE/'exp_C2/results/C_B4_existing_T_paired_v3.csv';allp=pd.read_csv(p)
p=STAGE/'exp_C2/results/C_B4_existing_T_harmonic_paired_v3.csv';harm=pd.read_csv(p);rows=[]
for base,label in [('G0','相对G0'),('G0_oldT','相对原T'),('combo_noise31','相对同维噪声')]:
    r=allp[allp.candidate.eq('G0_oldT_newT')&allp.comparator.eq(base)&allp.domain.eq('EQUAL_DOMAIN')&allp.dim.eq('mos')].iloc[0]
    vals=[f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}]']
    for loss in ['random','burst']:
        r=harm[harm.candidate.eq('G0_oldT_newT')&harm.comparator.eq(base)&harm.loss_type.eq(loss)&harm.dim.eq('mos')].iloc[0];vals.append(f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}]')
    rows.append([label,*vals])
tables.append(dict(number=5,cn='表5(c) 既有时间特征组合的MOS增量',en='Table 5(c) MOS increments of the existing temporal-feature combination',headers=['组合的比较对象','两域等权','共同格随机','共同格突发'],rows=rows,note='45维组合，历史固定种子20261011；下降量及95%参考组区间。共同支持247格/2123文件，调和权重不变；与表5(a)/(b)三种子结果分别报告。'))
# All registered transfer repair results, no best-only reporting.
p=STAGE/'exp_C2/results/C_B0_transfer_paired_v4.csv'
if p.exists():
    z=pd.read_csv(p);labels={'drop_probability_mean':'去概率均值','drop_probability_sd':'去概率标准差','drop_probability_q95':'去概率95分位','drop_mask20_Ppl':'去掩码丢包率','drop_mask20_logBurstR':'去掩码log BurstR','drop_mask20_BurstR_valid':'去可定义标志','drop_direct_Ppl':'去直接丢包率','drop_direct_logBurstR':'去直接log BurstR','missing_BurstR':'无效BurstR设为缺失','missing_all_invalid':'无效连续项设为缺失','confidence_weighted':'固定置信加权'};rows=[]
    for candidate,label in labels.items():
        vals=[]
        for dim in ['mos','dis']:
            r=z[z.candidate.eq(candidate)&z.comparator.eq('G0_oldT_newT_no_pitch')&z.domain.eq('LIVE')&z.dim.eq(dim)].iloc[0];vals.append(f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]')
        rows.append([label,*vals])
    tables.append(dict(number=6,cn='表6 检测特征移除与固定修复的真实域增量',en='Table 6 Live-domain increments after feature removal and fixed repairs',headers=['预注册诊断方案','MOS下降量 [95%区间]','DIS下降量 [95%区间]'],rows=rows,note='相对44维无检测特征基线；三种子、原参考组五折、2000次配对重抽样。全部11方案为TRAIN开发诊断；负值表示MAE增加。'))
# Old controlled probe and its repaired DSP grid, exact denominators retained.
old=pd.read_csv(STAGE/'exp_B2/results/B4_blind.csv');dsp=pd.read_csv(STAGE/'exp_B2/results/B2_dsp_legacy_summary_v3.csv');rows=[]
for codec,plc,label in [('G711','zero','G.711／零填充'),('Opus16k','native_PLC','Opus／原生PLC'),('Opus16k','zero','Opus／零填充'),('PCM','zero','PCM／零填充')]:
    r=dsp[dsp.codec.eq(codec)&dsp.plc.eq(plc)&dsp.grid_ms.eq(20)].iloc[0]
    g=old[old.codec.eq(codec)&old.statistic.eq('Spearman')&old.truth.eq('actual_BurstR')]
    g=g[g.concealment.eq('zero')] if plc=='zero' else g[~g.concealment.eq('zero')]
    a=g.iloc[0];rows.append([label,f'{a.value:.4f} [{a.ci_low:.4f},{a.ci_high:.4f}]',f'{r.frame_F1_mean:.4f}',f'{r.Ppl_MAE:.4f}',f'{r.BurstR_Spearman:.4f}',f'{int(r.burst_n)}/{int(r.n)}'])
tables.append(dict(number=7,cn='表7 补偿方式与DSP检测线索的可观测性',en='Table 7 Compensation methods and observability of DSP detection cues',headers=['编解码／补偿','旧BurstR ρ [95%区间]','新网格F1','Ppl MAE','新BurstR ρ','可定义数量'],rows=rows,note='同一既有无附加噪声探针：旧80ms低能量特征；新20ms对齐DSP网格。新相关只在可定义样本计算；Opus原生PLC为950/8000的条件相关。学习检测器另按新AISHELL/THCHS留出评价。'))
# Completed GPU comparison settings, independent source owner medians.
rows=[]
for name,rel,model in [('五维NISQA','NATIVE_CUDA_VAL2700_OfficialNISQA5D_20261012_051048','OfficialNISQA5D'),('单MOS实现','NATIVE_CUDA_VAL2700_OfficialNISQA_MOS_20261012_042138','OfficialNISQA_MOS')]:
    p=STAGE/'exp_D2/measurements'/rel/'summary.csv'
    if p.exists() and (p.parent/'COMPLETE.json').exists():
        r=pd.read_csv(p).iloc[0];rows.append([name,f'{r.end_to_end_s_median:.4f} ({r.end_to_end_s_IQR:.4f})',f'{r.RTF_median:.4f}',f'{r.frontend_s_median:.4f}',f'{r.infer_s_median:.4f}'])
if rows:tables.append(dict(number=8,cn='表8(b) 完成的官方模型GPU对照',en='Table 8(b) Completed GPU comparisons for the official models',headers=['实现','端到端秒 (IQR)','RTF','前端秒','推理秒'],rows=rows,note='完整VAL2700、3预热/10正式；同机背景负载。阶段中位数分别统计，不相加作为端到端中位数；单MOS仅作实现成本补充。'))

# Uniform TRAIN200 costs, separate from actual timed settings.
p=STAGE/'exp_D2/D0_all_core_methods_TRAIN200_v4.csv'
if p.exists() and (STAGE/'exp_D2/D0_TRAIN200_uniform_v4_RECEIPT.json').exists():
    z=pd.read_csv(p);rows=[]
    for model,label in [('A_distill','蒸馏GRU v1'),('OfficialNISQA5D','官方NISQA五维'),('PLCMOS','PLCMOS'),('DNSMOS','DNSMOS'),('SQUIM','SQUIM'),('P563','P.563')]:
        r=z[z.model.eq(model)].iloc[0]
        rows.append([label,str(int(r.parameters)),f'{r.MMACeq_per_audio_s:.4f}' if pd.notna(r.MMACeq_per_audio_s) else 'NA', '15次评分调用' if model=='PLCMOS' else '9.01秒复制／全部滑窗' if model=='DNSMOS' else '非学习式算法' if model=='P563' else '完整前端及网络'])
    for t in tables:
        if t['number']==8:
            t['cn']=t['cn'].replace('表8(b)','表8(c)') if '(b)' in t['cn'] else '表8(b) 完成的整集CPU测量'
            t['en']='Table 8(c) Completed GPU comparisons for the official models' if '(c)' in t['cn'] else 'Table 8(b) Completed full-validation CPU measurements'
    pos=next((i for i,t in enumerate(tables) if t['number']==8),len(tables))
    tables.insert(pos,dict(number=8,cn='表8(a) 统一TRAIN200的参数与含前端运算量',en='Table 8(a) Parameters and frontend-inclusive operations on the same TRAIN200 sample',headers=['方法','可学习参数','百万MACeq/音频秒','计算范围'],rows=rows,note='同200条TRAIN，原音频总时长加权。FFT采用1.25Nlog₂N等价估计，FIR为支持上界；解析运算量不代表硬件耗时。PLCMOS完整15次调用；P.563无已验证运算计数，标NA。'))
p=STAGE/'exp_D2/memory_v4/RSS_layers_v4.csv'
if p.exists() and (STAGE/'exp_D2/memory_v4/LITE_V1_COMPLETE.json').exists():
    r=pd.read_csv(p).iloc[0];rows=[]
    for label,col,inc in [('解释器基线','interpreter_RSS_MB',None),('解释器与库','interpreter_plus_libraries_RSS_MB',r.interpreter_plus_libraries_RSS_MB-r.interpreter_RSS_MB),('加载模型后','model_loaded_RSS_MB',r.model_loading_increment_RSS_MB),('处理峰值','process_peak_RSS_MB',r.audio_inference_peak_increment_RSS_MB),('处理常驻中位数','process_steady_RSS_median_MB',None)]:
        rows.append([label,f'{r[col]:.4f}',f'{inc:.4f}' if inc is not None else '—'])
    tables.append(dict(number=8,cn='表8(d) 精简首版进程的RSS分层',en='Table 8(d) Layered RSS measurements of the compact v1 process',headers=['阶段','RSS／MB','相对上一加载阶段增量／MB'],rows=rows,note=f'独立Windows进程；NumPy循环推理，固定TRAIN200×三分块。常驻IQR={r.process_steady_RSS_IQR_MB:.4f} MB；权重数组{int(r.weight_resident_payload_bytes)} B，前端常量{int(r.frontend_immutable_payload_bytes)} B，最大可变状态{int(r.mutable_state_bytes_max)} B。数组载荷与进程RSS分别统计。整段峰值归一化和重采样在上游准备，属于输入准备后的流式内核。'))


# Completed learned-detector source holdouts; no fresh computation.
p=STAGE/'exp_C2/results/C_B2_heldout_report_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B2_HELDOUT_REPORT_COMPLETE_v4.json').exists():
    z=pd.read_csv(p);rows=[]
    labels={'zero':'零填充','repeat':'重复','native_opus':'Opus原生','g711_appendixI':'G.711补偿'}
    for source,label in [('test','AISHELL'),('thchs','THCHS')]:
        for plc in ['zero','repeat','native_opus','g711_appendixI']:
            nn=z[z.source.eq(source)&z.stratum.eq(plc)&z.method.eq('CNN_mask20')].iloc[0]
            dsp=z[z.source.eq(source)&z.stratum.eq(plc)&z.method.eq('DSP_zero20')].iloc[0]
            rows.append([label,labels[plc],str(int(nn.n)),f'{nn.frame_F1_pooled:.4f}',f'{dsp.frame_F1_pooled:.4f}',f'{nn.Ppl_MAE:.4f}',f'{nn.logBurstR_Spearman:.4f}',str(int(nn.joint_BurstR_n))])
    pos=next((i for i,t in enumerate(tables) if t['number']==8),len(tables))
    tables.insert(pos,dict(number=7,cn='表7(b) 固定学习检测器的两来源留出表现',en='Table 7(b) Two-source holdout performance of the fixed learned detector',headers=['留出源','补偿方式','文件数','CNN帧F1','零检测帧F1','CNN丢包率MAE','CNN突发度ρ','联合可定义数'],rows=rows,note='新仿真含预设编解码、噪声和电平变化，与表7(a)无附加噪声探针分别报告。CNN为固定CNN_mask20，零检测为DSP_zero20；帧F1按有效帧汇总，丢包率按比例计。ρ为log BurstR的Spearman，仅在真值与预测均可定义的文件计算，最后列给出分母。零填充ρ预定参考线为0.7，THCHS未达；不按留出结果重选模型或阈值。'))
    for t in tables:
        if t['number']==7 and '(b)' not in t['cn']:
            t['cn']=t['cn'].replace('表7 ','表7(a) ');t['en']=t['en'].replace('Table 7 ','Table 7(a) ')


# Actual paired response statistics, all seven compensation conditions.
p=STAGE/'exp_C2/results/C_B5_response_report_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B5_RESPONSE_REPORT_READY_v4.json').exists():
    z=pd.read_csv(p)
    model_columns=[('NISQA5D','NISQA五维'),('PLCMOS','PLCMOS'),('v1_GRU','GRU v1'),('G0_oldT','HGB G0+T')]
    if z.model.eq('selected_v2').any():model_columns.append(('selected_v2','GRU v2'))
    conditions=[('G711','zero','G.711／零填充'),('G711','repeat','G.711／重复'),('G711','g711_appendixI','G.711／Appendix I'),('G711','repeat1_silence','G.711／重复1包后静音'),('Opus','zero','Opus／零填充'),('Opus','repeat','Opus／重复'),('Opus','native_opus','Opus／原生PLC')]
    for table,title,en,signed in [('monotonicity','表7(c) 丢包率增加时完整预测曲线非增的比例','Table 7(c) Fraction of complete prediction curves nonincreasing with packet loss',False),('burst_minus_random','表7(d) 同丢包计数下目标突发度8与1的预测差','Table 7(d) Prediction difference for target burstiness 8 versus 1 at equal packet counts',True)]:
        rows=[]
        for codec,plc,label in conditions:
            values=[]
            for model,_ in model_columns:
                g=z[z.table.eq(table)&z.model.eq(model)&z.codec.eq(codec)&z.plc.eq(plc)]
                if len(g)!=1:raise ValueError('Actual response selector is not unique')
                r=g.iloc[0];fmt='+.4f' if signed else '.4f'
                values.append(format(r.value,fmt)+' ['+format(r.ci_low,fmt)+', '+format(r.ci_high,fmt)+']')
            rows.append([label,*values])
        tables.append(dict(number=7,cn=title,en=en,headers=['补偿条件',*[label for _,label in model_columns]],rows=rows,note='100个参考、29名说话人；按说话人簇bootstrap 2000次。表7(c)每条件400条五点曲线，四个相邻丢包率步均非增才计为1。表7(d)目标丢包率10%，同一参考和实际丢包计数配对，正值表示突发分布的预测MOS较高；不是主观质量准确度。五模型均使用固定权重，完整Ppl/BurstR网格见补充表。'))


# Actual fixed original detector: all seven matched-count diagnostic conditions.
p=STAGE/'exp_C2/results/C_B2_B5_diagnostic_report_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B2_B5_DIAGNOSTIC_REPORT_READY_v4.json').exists():
    q=pd.read_csv(p)
    conditions=[('G711','zero','G.711／零填充'),('G711','repeat','G.711／重复'),('G711','g711_appendixI','G.711／Appendix I'),('G711','repeat1_silence','G.711／重复1包后静音'),('Opus','zero','Opus／零填充'),('Opus','repeat','Opus／重复'),('Opus','native_opus','Opus／原生PLC')]
    rows=[]
    for codec,plc,label in conditions:
        z=q[q.codec.eq(codec)&q.plc.eq(plc)]
        if len(z)!=1:raise ValueError('Matched-count diagnostic selector is not unique')
        r=z.iloc[0]
        rows.append([label,int(r.total_n),int(r.random_defined_n),int(r.burst_defined_n),
                     '—' if pd.isna(r.random_vs_burst_AUC) else f'{r.random_vs_burst_AUC:.4f}'])
    tables.append(dict(number=7,cn='表7(e) 同参考同丢包计数的突发与随机条件区分',
                       en='Table 7(e) Burst versus random discrimination at matched references and lost-packet counts',
                       headers=['补偿条件','总数','随机可定义','突发可定义','条件性AUC'],rows=rows,
                       note='固定原检测器的20 ms掩码估计，目标Ppl=10%；每条件100参考、随机100条、目标BurstR 2/4/8的突发300条。AUC仅在估计logBurstR可定义的行中计算，随机与突发分母并列；可定义覆盖不足或类别缺失不填造分数。该配对诊断复用AISHELL开发来源，不用于模型或阈值选择。'))


# Fixed TRAIN frequency intervention: quality values are C-owned.
p=STAGE/'exp_C2/results/C_B48k_mask_main_full_precision_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B48K_MASK_COMPLETE_v4.json').exists():
    z=pd.read_csv(p);rows=[]
    for protocol,label in [('CV','五折等域'),('LOSO_SIM','四SIM来源等权'),('SIM_TO_LIVE','SIM→LIVE')]:
        for dim in ['mos','dis']:
            r=z[z.protocol.eq(protocol)&z.dim.eq(dim)].iloc[0]
            rows.append([label,dim.upper(),f'{r.natural_MAE:.4f}',f'{r.masked_MAE:.4f}',
                         f'{r.gain_natural_minus_masked:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',
                         f'{r.natural_Pearson:.4f} → {r.masked_Pearson:.4f}'])
    tables.append(dict(number=2,cn='表2(c) 固定48 kHz模型的高频屏蔽诊断',
                       en='Table 2(c) High-frequency masking of the fixed 48 kHz model',
                       headers=['训练评价','维度','原输入MAE','屏蔽MAE','MAE下降量及95%区间','Pearson 原→屏蔽'],
                       rows=rows,note='已完成48 kHz/48梅尔/固定增益/四电平统计模型；将FFT功率中>8 kHz频带置零，侧信息与各折增益/标准化不变。三种子预测先求均值后评价，区别于表2(a)三次MAE的平均。配对区间按参考组重抽样2000次；下降量=原输入−屏蔽，正值为改善。全部五维与来源分层见补充表。'))

# Same-sample component comparison, separate from the frozen backend MAC.
p=STAGE/'exp_D2/v1_v15_sameTRAIN200_totals_v5.csv'
if p.exists():
    z=pd.read_csv(p);rows=[]
    for model,label in [('v1','GRU v1'),('v1.5','GRU v1.5'),('v2_conv64','卷积k5+GRU64'),('v2_64','双层GRU64'),('v2_pure','双层GRU96')]:
        r=z[z.model.eq(model)&z.TRAIN_n.eq(200)].iloc[0]
        rows.append([label,int(r.parameters),f'{r.model_affine_pool_MACeq_per_audio_s/1e6:.4f}',f'{r.native_full_matched_scope_MACeq_per_audio_s/1e6:.4f}'])
    tables.append(dict(number=8,cn='表8(e) 同TRAIN200的模型与完整前端组件运算量',en='Table 8(e) Model and complete frontend operations on the same TRAIN200 sample',headers=['模型','参数数','网络及池化／MMACeq每秒','完整路径／MMACeq每秒'],rows=rows,note='同200条TRAIN，按原音频总时长加权；v1.5为单层GRU32、固定训练增益与四项运行电平统计。实际执行的频带计算纳入完整路径；FFT为等价估计，重采样为支持上界，标量项按所列代理计数。此组件表与冻结后端的正式运算量分别列明。'))
p=STAGE/'exp_D2/D_lean_raw_RSS_v5.csv'
if p.exists() and (STAGE/'exp_D2/D_LEAN_RAW_RSS_COMPLETE_v5.json').exists():
    z=pd.read_csv(p);rows=[]
    for model,label in [('v2_pure','双层GRU96'),('v2_64','双层GRU64')]:
        r=z[z.model.eq(model)&z.implementation.eq('rawcausal_numpy_minimal')].iloc[0]
        rows.append([label,f'{r.max_abs_error:.4e}',f'{r.steady_RSS_MB:.4f} ({r.steady_RSS_IQR_MB:.4f})',f'{r.peak_RSS_MB:.4f}'])
    tables.append(dict(number=8,cn='表8(f) 原始音频分包输入的精简进程与流式等价性',en='Table 8(f) Compact processes and streaming equivalence with raw audio packet inputs',headers=['模型','最大预测差','常驻RSS／MB (IQR)','采样峰值RSS／MB'],rows=rows,note='同200条TRAIN×10/20/100 ms；进程RSS包含解释器、NumPy、SoundFile、前端、模型和工作区。保留状态上界统一见表8(j)。与CPU原生整序列参考比较；精简实现的FFT、GRU、注意力及头使用的算术精度见正文。采样周期50 ms，平台为桌面性能核，测量期间存在背景工作。'))
p=STAGE/'exp_C2/results/C_INT8_engineering_report_v5.csv'
if p.exists() and (STAGE/'exp_C2/C_INT8_ENGINEERING_REPORT_READY_v5.json').exists():
    z=pd.read_csv(p);rows=[]
    for model,label in [('v2_pure','双层GRU96'),('v2_64','双层GRU64')]:
        r=z[z.model.eq(model)&z.domain.eq('pooled')&z.dimension.eq('mos')].iloc[0]
        rows.append([label,int(r.n),f'{r.FP32_RMSE:.4f}',f'{r.INT8_RMSE:.4f}',f'{r.FP32_Pearson:.4f}',f'{r.INT8_Pearson:.4f}',f'{r.INT8_vs_FP32_max_abs_prediction_difference:.4f}'])
    tables.append(dict(number=8,cn='表8(g) 修订版动态INT8的TRAIN折外工程比较',en='Table 8(g) Training out-of-fold engineering comparison of dynamic INT8 variants',headers=['模型','文件数','FP32 RMSE','INT8 RMSE','FP32 r','INT8 r','逐文件最大差'],rows=rows,note='原种子0的五个外层折，按文件池化的MOS原值；FP32为CPU整序列参考。量化覆盖GRU与线性层，未重拟合。两模型四条TRAIN×三块长的独立流式比较24行均未通过10⁻⁵，保留浮点部署版本。该工程结果与两域等权主精度分别报告。'))

# The finalized model decomposition is separate from the six-method historic figure.
p=STAGE/'exp_C2/results/C3_MSE_decomposition_new_v2_v3.csv'
if p.exists() and completion.exists():
    z=pd.read_csv(p);rows=[]
    for ds,label in [('NISQA_TEST_FOR','FOR'),('NISQA_TEST_P501','P501'),('NISQA_TEST_LIVETALK','LIVETALK'),('TCD_VOIP','TCD')]:
        r=z[z.model.eq(diagnostic_model)&z.dataset.eq(ds)&z.domain.eq('EQUAL_DOMAIN')&z.level.eq('file')&z.dim.eq('mos')].iloc[0]
        rows.append([label,f'{r.MSE:.4f}',f'{r.mean_bias_squared:.4f}',f'{r.scale_mismatch_squared:.4f}',f'{r.linear_correlation_residual:.4f}'])
    tables.append(dict(number=4,cn='表4(e) 冻结模型的原值MOS均方误差分解',en='Table 4(e) Raw MOS mean-square error decomposition of the frozen model',headers=['评价集','MSE','均值偏移','尺度失配','线性相关残差'],rows=rows,note='逐文件原始预测，域内先计算再等权。该线性分解与训练映射RMSE分别报告；线性相关残差不称纯排序误差。'))
p=STAGE/'exp_C2/results/C5_Holm72_v3.csv'
if p.exists() and completion.exists():
    z=pd.read_csv(p);rows=[]
    for ds,label in [('NISQA_TEST_FOR','FOR'),('NISQA_TEST_P501','P501'),('NISQA_TEST_LIVETALK','LIVETALK'),('TCD_VOIP','TCD')]:
        for model,pretty in [('v1_A_distill','GRU v1'),('PLCMOS','PLCMOS'),('NISQA5D','NISQA5D')]:
            a=z[z.dataset.eq(ds)&z.level.eq('file')&z.comparator.eq(model)&z.metric.eq('Pearson')].iloc[0]
            b=z[z.dataset.eq(ds)&z.level.eq('file')&z.comparator.eq(model)&z.metric.eq('RMSE')].iloc[0]
            rows.append([label,pretty,f'{a.difference_a_minus_b:+.4f} [{a.ci_low:+.4f}, {a.ci_high:+.4f}]',f'{a.p_Holm:.4f}',f'{b.difference_a_minus_b:+.4f} [{b.ci_low:+.4f}, {b.ci_high:+.4f}]',f'{b.p_Holm:.4f}'])
    tables.append(dict(number=4,cn='表4(f) 冻结模型相对三种比较法的逐文件配对结果',en='Table 4(f) Paired file-level results of the frozen model against three comparators',headers=['评价集','比较法','Pearson差 [95%区间]','Holm p','RMSE差 [95%区间]','Holm p'],rows=rows,note='差值=GRU v2−比较法；Pearson为原始预测，RMSE采用各自固定TRAIN映射，官方NISQA保持原值。每一评价集/层面/指标内三个比较共同Holm；2000次条件组配对bootstrap，完整72行含Spearman及条件层面结果见补充。'))
p=STAGE/'exp_C2/results/C_PLC_cues_report_v5.csv'
if p.exists() and (STAGE/'exp_C2/C_PLC_CUES_REPORT_READY_v5.json').exists():
    z=pd.read_csv(p);rows=[]
    for model,label in [('PLC_cue_CNN_mask20','CNN-GRU波形线索'),('PLC_cue_DSP_repeat20','重复线索'),('PLC_cue_DSP_spectral_second20','谱二阶变化'),('PLC_cue_DSP_phase_delay20','相位时延'),('PLC_cue_DSP_envelope_smooth20','包络平滑'),('PLC_cue_DSP_lpc_residual20','LPC残差')]:
        a=z[z.source.eq('test')&z.group.eq('codec_plc')&z.stratum.eq('Opus/native_opus')&z.method.eq(model)].iloc[0]
        b=z[z.source.eq('thchs')&z.group.eq('codec_plc')&z.stratum.eq('Opus/native_opus')&z.method.eq(model)].iloc[0]
        rows.append([label,f'{a.frame_F1_pooled:.4f}',f'{b.frame_F1_pooled:.4f}',f'{a.Ppl_MAE:.4f}',f'{b.Ppl_MAE:.4f}',f'{b.logBurstR_Spearman:.4f}',f'{int(b.joint_BurstR_n)}/{int(b.n)}'])
    tables.append(dict(number=7,cn='表7(f) 原生Opus补偿下新波形线索与DSP的固定比较',en='Table 7(f) Fixed comparison of waveform cues and DSP methods under native Opus concealment',headers=['方法','AISHELL F1','THCHS F1','AISHELL丢包率MAE','THCHS丢包率MAE','THCHS突发度ρ','可定义数'],rows=rows,note='20 ms帧，AISHELL原生Opus861条、THCHS889条，帧F1按有效帧汇总。五DSP阈值由TRAIN确定，CNN-GRU由DEV三种子选择；复用此前评价语音，来源曝光单列。ρ仅在真值与预测均可定义的子集计算。'))
p=STAGE/'exp_C2/results/C_PLC_packet_lengths_report_v5.csv'
if p.exists() and (STAGE/'exp_C2/C_PLC_CUES_REPORT_READY_v5.json').exists():
    z=pd.read_csv(p);rows=[]
    for source,label in [('AISHELL1','AISHELL'),('THCHS30','THCHS')]:
        for packet in [10,20,30,40]:
            r=z[z.source.eq(source)&z.packet_ms.eq(packet)&z.plc.eq('native_opus')&z.method.eq('PLC_cue_CNN_physical')].iloc[0]
            rows.append([label,packet,int(r.n),f'{r.frame_F1_pooled:.4f}',f'{r.Ppl_MAE:.4f}',f'{r.logBurstR_Spearman:.4f}',int(r.joint_BurstR_n)])
    tables.append(dict(number=7,cn='表7(g) 固定波形线索检测器的传输单元长度探针',en='Table 7(g) Transport-unit-length probes for the fixed waveform-cue detector',headers=['来源','单元/ms','文件数','物理单元F1','丢包率MAE','突发度ρ','可定义数'],rows=rows,note='每来源30参考，全部探针共1920条，表列原生Opus960条；无附加噪声，按实际传输单元聚合概率后以固定0.5阈值判定。30 ms为20+10 ms复合传输单元。使用原冻结权重，语音来源复用、不回选。'))

from cost_v3_publication import ready as cost_v3_ready, cost_tables
if cost_v3_ready():
    for table in tables:
        if table['cn'].startswith('表2(b)'):
            # Preserve historical selection data in its source; final costs use Table8.
            table['headers'].pop(2)
            for row in table['rows']:row.pop(2)
            table['note']='三种子均值±SD；同16k/48梅尔前端、训练侧固定增益及四项运行电平统计。*为TRAIN综合分及筛选期同TRAIN200预算内选出的容量结构。五个早期配置及全部五维结果列补充材料。最终模型的完整前端成本统一见表8，筛选期运算估计保留在原始训练记录。'
    tables = [t for t in tables if not (t['number'] == 8 and
              any(t['cn'].startswith('表8(' + part + ')') for part in 'abcdefhi'))]
    tables.extend(cost_tables())
# Submission-only table text and column selection; final owner CSV values are unchanged.
english_titles = {
    '表4(d)': 'Table 4(d) File-level MOS results for cross-condition evaluation on TCD-VoIP',
    '表8(a)': 'Table 8(a) Parameters, weight-file sizes, and frontend-inclusive operation estimates',
    '表8(b)': 'Table 8(b) Hot-cache end-to-end real-time factors on the same validation sample',
    '表8(c)': 'Table 8(c) Phase runtimes for the complete validation sample on a single efficiency core',
    '表8(d)': 'Table 8(d) Memory measurements of independent full-utterance processes',
    '表8(e)': 'Table 8(e) Complete operation breakdown of the main deployment path',
    '表8(f)': 'Table 8(f) Chunk runtimes and resident memory of compact F32 streaming kernels',
    '表8(h)': 'Table 8(h) Supplementary runtime measurements on a single performance core',
    '表8(i)': 'Table 8(i) Single-pass representativeness checks on the complete validation set',
    '表8(j)': 'Table 8(j) Upper bounds on retained streaming state at supported input sample rates',
}
feature_note = ('44维由14维G0基础统计、12维原T及18维新T组成；45维在相同输入上仅增加'
                'pYIN估计的相邻活动帧基频跳变超过4半音的比例（t2_pitch_jump4）。'
                '历史44/45维组合采用固定种子20261011；检测增量比较采用seed0/1/2的44维基线'
                '及其52维扩展，历史单种子与当前三种子结果分别报告。')
dnsmos_mac_note = ('DNSMOS旧运算估计按TRAIN200原音频总时长加权，当前估计使用VAL代表长度，'
                   '两者长度口径不同；短音频按作者9.01 s输入要求重复，滑窗按1 s步长离散计数，'
                   '并计入固定STFT与卷积开销，因此不能按音频时长连续缩放。')
for table in tables:
    for prefix, title in english_titles.items():
        if table['cn'].startswith(prefix):
            table['en'] = title
            break
    if table['number'] == 5:
        table['note'] = table.get('note', '') + ' ' + feature_note
    if table['cn'].startswith('表8(a)'):
        table['note'] = table.get('note', '') + ' ' + dnsmos_mac_note
    if table['cn'].startswith('表8(f)'):
        observed_columns = [i for i, h in enumerate(table['headers'])
                            if '观察状态' in h or h in ('状态／字节', '状态/字节')]
        table['headers'] = [h for i, h in enumerate(table['headers']) if i not in observed_columns]
        table['rows'] = [[v for i, v in enumerate(row) if i not in observed_columns]
                         for row in table['rows']]
        table['cn'] = table['cn'].replace('块耗时、载荷和常驻内存', '块耗时与常驻内存')
        table['note'] = table.get('note', '').replace(
            '观察状态为本输入的保留数组载荷及64位逻辑寄存器；各实际输入采样率的解析上界另见表8(j)。',
            '支持16 kHz和48 kHz输入的保留状态上界统一列于表8(j)。')

# Completed GRU96 frontend main contrast: exact postfreeze TRAIN CSV rows.
front96 = HERE.parent / 'frontend/rule_meanprediction_20261014'
fm = pd.read_csv(front96 / 'TRAIN_frontend_metrics_meanprediction.csv')
fp = pd.read_csv(front96 / 'TRAIN_frontend_paired_meanprediction.csv')
for table in tables:
    if table['cn'].startswith('表2(c)'):
        table['cn'] = table['cn'].replace('表2(c)', '表2(e)')
        table['en'] = table['en'].replace('Table 2(c)', 'Table 2(e)')
for metric,part,label in [('MAE','c','MAE'),('Pearson','d','原始Pearson')]:
    rows=[]
    for protocol,design in [('CV','等域CV'),('LOSO_SIM','四SIM来源留出'),('SIM_TO_LIVE','SIM→LIVE')]:
        for dim in ['mos','loud']:
            a=fm[fm.arm.eq('A')&fm.protocol.eq(protocol)&fm.dim.eq(dim)&fm.metric.eq(metric)]
            b=fm[fm.arm.eq('B')&fm.protocol.eq(protocol)&fm.dim.eq(dim)&fm.metric.eq(metric)]
            r=fp[fp.comparison.eq('B-A')&fp.protocol.eq(protocol)&fp.dim.eq(dim)&fp.metric.eq(metric)]
            if len(a)!=1 or len(b)!=1 or len(r)!=1:raise ValueError('Exact GRU96 main table selector not unique')
            r=r.iloc[0]
            if metric=='MAE':
                difference,lo,hi=-r.difference,-r.ci_high,-r.ci_low
                if protocol=='SIM_TO_LIVE':
                    tr=pd.read_csv(front96/'tier_improvements_A_minus_B.csv')
                    tr=tr[tr.dim.eq(dim)].iloc[0];difference,lo,hi=tr.improvement,tr.ci_low,tr.ci_high
            else:difference,lo,hi=r.difference,r.ci_low,r.ci_high
            rows.append([design,dim.upper(),f'{a.iloc[0].value:.4f}',f'{b.iloc[0].value:.4f}',f'{difference:+.4f} [{lo:+.4f}, {hi:+.4f}]'])
    tables.append(dict(number=2,cn=f'表2({part}) 双层GRU96前端A/B的{label}比较',en=f'Table 2({part}) Two-layer GRU96 frontend A/B comparison: {metric}',headers=['训练设计','维度','A','B',('A−B MAE下降量' if metric=='MAE' else 'B−A Pearson增量')+' [95%区间]'],rows=rows,note='A：整段峰值归一化、无四统计；B：TRAIN固定增益、原始运行电平四统计。先平均种子0/1/2的逐文件预测，再计算指标并按域/来源等权；参考组同步bootstrap 2000次，随机种子20261014。MAE下降量为A−B、Pearson增量为B−A，均以正值表示改善。结果后补充的聚合口径，TRAIN稳健性分析；三档判断仅使用SIM→LIVE的MOS与LOUD MAE。'))

"""Reader-facing eight-table projection of completed owner outputs."""
from copy import deepcopy
import json

LABELS={'GRU真实标签':'首版（无蒸馏）','GRU蒸馏v1':'首版','GRU v1':'首版','最终模型 v2':'本文模型','GRU v2':'本文模型','首版 v1':'首版','双层GRU96 原生':'本文模型 原生','双层GRU96 部署F32':'本文模型 整句F32','双层GRU64 部署F32':'GRU64 整句F32'}
def lab(x):return LABELS.get(str(x),str(x))

def compose(tables,here):
    old=deepcopy(tables)
    def get(prefix):
        matches=[t for t in old if t['cn'].startswith(prefix)]
        if len(matches)!=1:raise ValueError('Eight-table source selector '+prefix)
        return deepcopy(matches[0])
    def item(number,cn,en,headers,rows,note):
        return dict(number=number,cn=cn,en=en,headers=headers,rows=rows,note=note)
    main=[]
    t=get('表1 ');t.update(number=1,cn='表1 数据与分组',en='Table 1. Data and grouping');main.append(t)
    t=get('表2(b)');t.update(number=2,cn='表2 结构与容量的MOS训练结果',en='Table 2. MOS training results across architectures and capacities',note='三个训练种子均值±SD；相同16 kHz、48梅尔前端。按4.1的综合分和成本规则选出双层GRU96，*表示该结构。SD为训练初始化变化。');main.append(t)
    ma=get('表2(c)');pr=get('表2(d)');rows=[]
    for a,b in zip(ma['rows'],pr['rows']):
        if a[:2]!=b[:2]:raise ValueError('Frontend panel row identity differs')
        rows.append([*a[:2],a[2]+'→'+a[3],a[4],b[2]+'→'+b[3],b[4]])
    main.append(item(3,'表3 双层GRU96前端A/B的MAE与相关性','Table 3. MAE and correlation in the two-layer GRU96 frontend A/B comparison',['训练设计','维度','MAE A→B','A−B下降量 [95%CI]','Pearson A→B','B−A增量 [95%CI]'],rows,'A：整段峰值归一化、无原始电平四统计；B：训练侧固定增益、原始电平四统计。先平均三个种子的逐文件预测，再计算指标；参考组分层同步bootstrap 2000次。MAE下降量和Pearson增量均以正值为改善；SIM→LIVE MOS/LOUD用于表述判断。'))
    quality=[get(x) for x in ['表3 ','表4(a)','表4(b)','表4(c)','表4(d)']]
    labels=[]
    for t in quality:
        for r in t['rows']:
            if r[0] not in labels and r[0] not in ['双层GRU64','卷积k5+GRU64']:labels.append(r[0])
    rows=[]
    for label in labels:
        rr=[lab(label)]
        for t in quality:
            found=[r for r in t['rows'] if r[0]==label]
            if not found:rr.extend(['—','—']);continue
            r=found[0];rmse=r[5]
            if label=='UTMOSv2':rmse=r[4]+'ᵇ'
            rr.extend([r[2],rmse])
        rows.append(rr)
    main.append(item(4,'表4 各方法在VAL及跨集评价中的MOS精度','Table 4. MOS accuracy on VAL and cross-dataset evaluations',['方法','VAL\nr','VAL\nRMSE','FOR\nr','FOR\nRMSE','P501\nr','P501\nRMSE','LIVE\nr','LIVE\nRMSE','TCD\nr','TCD\nRMSE'],rows,'r为逐文件原始Pearson；RMSE采用既定TRAIN映射，官方NISQA保持原值。ᵇ UTMOSv2未拟合TRAIN映射，列原值RMSE。—表示该评价范围无预测；P.563使用实际有效样本。DNSMOS OVRL和SQUIM PESQ为不同目标的对比输出。'))
    t=get('表4(f)');t.update(number=5,cn='表5 本文模型对三个比较者的MOS文件层配对结果',en='Table 5. File-level paired MOS comparisons against three comparators');t['rows']=[[r[0],lab(r[1]),*r[2:]] for r in t['rows']];main.append(t)
    t=get('表5(b)');t.update(number=6,cn='表6 学习检测特征对MOS和DIS的增量',en='Table 6. MOS and DIS increments from learned detection features',note='44维基础组合加入8维检测特征，总52维。同容量HGB、三个训练种子逐文件绝对误差平均；参考组配对bootstrap 2000次。下降量为正表示MAE降低。');main.append(t)
    dsp=get('表7(a)');cnn=get('表7(b)');rows=[]
    for r in dsp['rows']:rows.append(['无噪声探针',r[0],r[1],r[2],'—',r[3],r[4],r[5]])
    for r in cnn['rows']:rows.append([r[0]+'留出',r[1],'—',r[4],r[3],r[5],r[6],str(r[7])+'/'+str(r[2])])
    main.append(item(7,'表7 补偿线索可观测性与两来源留出表现','Table 7. Concealment-cue observability and two-source held-out performance',['来源/探针','补偿','旧80ms ρ [CI]','DSP帧F1','CNN帧F1','Ppl MAE','BurstR ρ','联合n/全部n'],rows,'无噪声探针行的Ppl和ρ来自20 ms对齐DSP；来源留出行来自固定CNN，DSP列为零检测对照。ρ在联合可定义子集内计算，分母同时给出；20,000条留出仿真按来源及补偿分层。旧80 ms声学链与真实包链的响应区分报告。'))
    mac=get('表8(a)');s3={r[0]:r for r in get('表8(b)')['rows']};s2={r[0]:r for r in get('表8(h)')['rows']}
    stream={r[0]:r for r in get('表8(f)')['rows'] if str(r[1])=='10'}
    bounds={}
    for r in get('表8(j)')['rows']:
        if str(r[1])=='48000':bounds[r[0]]=r[2]
    rows=[]
    for r in mac['rows']:
        name=r[0];st=stream.get('GRU96' if name=='双层GRU96 部署F32' else 'GRU64' if name=='双层GRU64 部署F32' else '')
        state=bounds.get('GRU96' if name=='双层GRU96 部署F32' else 'GRU64' if name=='双层GRU64 部署F32' else '', '—')
        rows.append([lab(name),r[1],r[3],s2[name][1] if name in s2 else '—',s3[name][1] if name in s3 else '—',st[3] if st else '—',st[4] if st else '—',state])
    main.append(item(8,'表8 参数、完整运算与整句及流式成本','Table 8. Parameters, full operation estimates, and utterance/streaming costs',['模型/路径','参数','MMACeq/秒','S2 RTF','S3 RTF','10ms P99/ms','流式RSS/MB','状态上界/B'],rows,'S2为单性能核，S3为单能效核；固定600条VAL、五轮热缓存端到端中位数。†表示环境标准未通过，测量值保留。10 ms块的P99、RSS来自独立精简流式内核；状态上界覆盖支持的16/48 kHz输入。其他采样率需额外重采样状态。模型路径、阶段耗时、IQR/漂移、权重文件、各块长及逐率状态详见补充表。DNSMOS原TRAIN200时长加权与VAL代表长度估计口径不同，作者9.01 s短音重复和1 s离散窗口、固定STFT卷积均计入。'))
    # Eight-table journal projection; expanded outputs are repository artifacts.
    representative={'v1.5：单层32','单层GRU 64','双层GRU 64','双层GRU 96＊','卷积k5+GRU 64'}
    main[1]['rows']=[r for r in main[1]['rows'] if r[0] in representative]
    main[1]['rows'][0][0]='单层GRU 32'
    main[1]['cn']='表2 代表结构与容量的MOS训练结果'
    main[1]['en']='Table 2. MOS training results for representative architectures and capacities'
    main[1]['note']='三个训练种子指标均值±SD；相同16 kHz、48梅尔前端，真实标签/官方五维教师权重0.7/0.3。按3.1综合分及成本规则选择＊结构。SD表示初始化变化；本表保持原容量选择口径，与表3先平均预测的后见口径区分。'
    main[2]['note']+='聚合及表述规则在原拟合结果后收到、聚合前固定；这项TRAIN稳健性分析沿用已冻结模型。'
    main[4]['note']='差值=本文模型−比较法；Pearson取原值，RMSE使用固定TRAIN映射，官方NISQA保持原值。每个评价集/层面/指标的三个比较共同Holm；2000次条件组同步bootstrap。'
    main[7]['note']='S2为单性能核，S3为单能效核；固定600条VAL、五轮热缓存端到端中位数。表内已测S2/S3项均通过当前标准。10 ms P99和RSS来自独立精简流式内核，状态上界覆盖16/48 kHz；其他采样率需额外重采样状态。NISQA MOS单头用于成本，NISQA五维用于质量。DNSMOS按VAL代表长度、9.01 s短音重复和1 s离散窗口估计，含固定STFT卷积；与旧TRAIN200时长加权口径不同。'
    from cost_v3_publication import cost_source
    failures=pd.read_csv(cost_source())
    pieces=[]
    for method in ['NISQA_MOS','PLCMOS']:
        r=failures[failures.method.eq(method)&failures.path.eq('official')&failures.setting.eq('S3')].iloc[0]
        pieces.append(f'{method} S3（'+('通过' if r.accepted else '†')+f'）：IQR/中位数{r.iqr_ratio*100:.6f}%、最大校准漂移{r.calib_drift*100:.6f}%')
    main[7]['note']+=' S3实际值：'+'；'.join(pieces)+'；当前IQR/中位数<10%、校准漂移<11%，这些测量量及敏感性范围均非置信区间。'
    return main

tables=compose(tables,HERE)

tables.sort(key=lambda table: (table['number'], table['cn']))
(HERE/'owner_tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
print('C-only accuracy tables',len(tables),'rows',sum(len(t['rows']) for t in tables))
