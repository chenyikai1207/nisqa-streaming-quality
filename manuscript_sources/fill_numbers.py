"""Fill only published owner CSVs; C is the sole source of accuracy values."""
from pathlib import Path
import os
import pandas as pd,json,re
HERE=Path(__file__).resolve().parent;STAGE=Path(__file__).resolve().parent/'scientific_inputs/R4'
values={};mapping={}
def put(key,value,path,selector,column='value',digits=4):
    if value is None or pd.isna(value):return
    values[key]=str(value) if isinstance(value,str) else str(int(value)) if digits==0 else f'{float(value):.{digits}f}'
    mapping[key]=dict(csv=str(path.relative_to(STAGE)) if path.is_relative_to(STAGE) else str(path.relative_to(HERE.parents[1])),selector=selector,column=column)
def row(frame,**filters):
    q=frame
    for k,v in filters.items():q=q[q[k].eq(v)]
    return q.iloc[0] if len(q)==1 else None
accuracy_path=STAGE/'exp_C2/results/C_primary_accuracy_v3.csv';accuracy=pd.read_csv(accuracy_path)
for ds in ['FOR','P501','LIVETALK']:
    for metric,label in [('RMSE','rmse'),('Pearson','pearson')]:
        sel=dict(model='A_distill',dataset='NISQA_TEST_'+ds,domain='EQUAL_DOMAIN',level='file',dim='mos',metric=metric,mapping='none' if metric=='Pearson' else 'cubic',protocol='V1_ORIGINAL_UNCHANGED')
        q=row(accuracy,**sel)
        if q is not None:put(f'test.{ds}.v1.{label}',q.value,accuracy_path,sel)
p=STAGE/'exp_A/A3_tiny.csv';q=row(pd.read_csv(p),candidate='tiny_gru_b0')
put('v1.parameters',q.parameters,p,dict(candidate='tiny_gru_b0'),'parameters',0)
p=STAGE/'shared/data_index.csv';index=pd.read_csv(p)
for name,prefix in [('TRAIN','NISQA_TRAIN_'),('VAL','NISQA_VAL_'),('FOR','NISQA_TEST_FOR'),('P501','NISQA_TEST_P501'),('LIVETALK','NISQA_TEST_LIVETALK')]:
    q=index[index.db.str.startswith(prefix)];put(f'data.{name}.n',len(q),p,dict(db_prefix=prefix),'row_count',0)
    if name not in ['TRAIN','VAL']:put(f'data.{name}.conditions',q.condition_id.nunique(),p,dict(db_prefix=prefix),'condition_id_nunique',0)
p=STAGE/'exp_C2/results/label_exposure_exact_counts.csv';q=pd.read_csv(p)
for key,col in [('complete_rows','complete_csv_rows'),('partial_rows','partial_csv_rows'),('rating_cells','complete_visible_rating_cells')]:put('exposure.'+key,q[col].sum(),p,dict(rows='all'),col+'_sum',0)
p=STAGE/'exp_B2/results/B1_models.csv';q=pd.read_csv(p)
for group,key in [('T_inv','inv'),('T_ord','ord')]:
    r=row(q,family='new',group=group,dim='mos');put('B.'+key+'.mae',r.mae,p,dict(family='new',group=group,dim='mos'),'mae')
p=STAGE/'exp_B2/results/B2_combo.csv';q=pd.read_csv(p)
for model,key in [('G0_oldT_newT','combo'),('G0_oldT_newT_no_pitch','combo_no_pitch')]:
    r=row(q,model=model,dim='mos',loss_type='all');put('B.'+key+'.mae',r.mae,p,dict(model=model,dim='mos',loss_type='all'),'mae')
p=STAGE/'exp_B2/results/B2_harmonic.csv';q=pd.read_csv(p)
for model,base,loss,key in [('newT','G0','burst','new_vs_g0_burst'),('newT','oldT','burst','new_vs_old_burst'),('newT','oldT','random','new_vs_old_random')]:
    r=row(q,model=model,baseline=base,loss_type=loss);text=f'{r.gain:+.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]'
    put('B.harmonic.'+key,text,p,dict(model=model,baseline=base,loss_type=loss),'gain|ci_low|ci_high')
for key in ['n_cells','n_files']:put('B.harmonic.'+key,q.iloc[0][key],p,dict(rows='fixed_support'),key,0)
p=STAGE/'exp_B2/TRAIN_ONLY/B6_compression_train.csv';q=pd.read_csv(p)
for dim in ['mos','dis']:
    r=row(q,model='v1_GRU',dim=dim)
    for key,col in [('slope','prediction_on_truth_slope'),('std_ratio','std_ratio')]:put(f'B.train.{dim}.{key}',r[col],p,dict(model='v1_GRU',dim=dim),col)
p=STAGE/'exp_B2/results/B3_shuffle_p.csv'
if p.exists():
    q=pd.read_csv(p)
    for role,key in [('primary','primary'),('sensitivity','sensitivity')]:
        r=q[q.role.eq(role)]
        if len(r):put('B.shuffle.'+key,'；'.join(f"{'原T' if s.family=='old' else '新T'}{'顺序组' if s.group=='T_ord' else '完整组'}的经验概率{s.p:.4f}" for s in r.itertuples() if not(s.family=='old' and s.group=='T_ord')),p,dict(role=role),'p')
p=STAGE/'exp_D2/d4_oof/TRAIN_OOF_quant_metrics.csv'
if p.exists():
    q=pd.read_csv(p);q=q[q.dim.eq('mos')]
    labels={'torch_cpu_float':'浮点','ort_cpu_float':'ONNX浮点','ort_cpu_INT8_linear_only':'ONNX线性层INT8','torch_cpu_INT8_GRU_linear':'GRU与线性层INT8'}
    text='；'.join(f'{labels.get(r.version,r.version)}的MOS等域MAE为{r.equal_domain_MAE:.4f}' for r in q.itertuples())
    put('D.quant.summary',text,p,dict(dim='mos'),'equal_domain_MAE')
p=STAGE/'exp_B2/results/B5_emodel_summary.csv'
if p.exists():
    q=pd.read_csv(p).iloc[0]
    put('B.emodel.summary',f"在{int(q.TRAIN_SIM_files)}条TRAIN_SIM中，排除额外处理后保留{int(q.after_nine_flags)}条，单一编解码条件有{int(q.after_single_codec)}条，其中G.711为{int(q.G711_after_filters)}条；因实际包长、PLC实现或数值突发参数不明，符合已知参数与盲估计成对比较条件的文件为{int(q.eligible_known_blind_pairs)}条",p,dict(row='scope_counts'),'TRAIN_SIM_files|after_nine_flags|after_single_codec|G711_after_filters|eligible_known_blind_pairs')
p=STAGE/'exp_B2/results/B4_blind.csv'
if p.exists():
    q=pd.read_csv(p);q=q[q.statistic.eq('Spearman')&q.truth.eq('actual_BurstR')]
    text='；'.join(f"{r.codec}{'原生补偿' if r.concealment!='zero' else '零填充'}为{r.value:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]" for r in q.itertuples())
    put('B.probe.blind.summary',text,p,dict(statistic='Spearman',truth='actual_BurstR'),'value|ci_low|ci_high')
p=STAGE/'exp_D2/D2_MAC_summary_with_attention_pool.csv'
if p.exists():
    q=pd.read_csv(p)
    r=row(q,target_sr=16000)
    put('D.v1.gru_macs',r.model_affine_and_attention_pool_MAC_equivalent_per_audio_s/1e6,p,dict(target_sr=16000),'model_affine_and_attention_pool_MAC_equivalent_per_audio_s / 1e6')
    for sr in [16000,48000]:
        r=row(q,target_sr=sr)
        put(f'D.v1.front{sr}',r.total_dense_with_pool_MAC_equivalent_estimate_per_audio_s/1e6,p,dict(target_sr=sr),'total_dense_with_pool_MAC_equivalent_estimate_per_audio_s / 1e6')
p=STAGE/'exp_D2/D3_stream_equivalence_completed_summary.csv'
if p.exists():
    q=pd.read_csv(p);q=q[q.platform.eq('Windows P core (S2)')]
    put('D.v1.stream.max_error',f'{q.max_prediction_difference.max():.4e}',p,dict(platform='Windows P core (S2)'), 'max_prediction_difference_max')
    put('D.v1.state_bytes',q.max_mutable_kernel_state_bytes.max(),p,dict(platform='Windows P core (S2)'), 'max_mutable_kernel_state_bytes_max',0)
    put('D.v1.stream.n',q.TRAIN_n.min(),p,dict(platform='Windows P core (S2)'), 'TRAIN_n',0)
p=STAGE/'exp_C2/external_index_TCD_VOIP_v3.csv';q=pd.read_csv(p)
put('data.external.n',len(q),p,dict(rows='all'),'row_count',0)
put('data.external.conditions',q.condition_id.nunique(),p,dict(rows='all'),'condition_id_nunique',0)
z=q[q.degradation.eq('CHOP')]
put('data.external.chop_n',len(z),p,dict(degradation='CHOP'),'row_count',0)
put('data.external.chop_conditions',z.condition_id.nunique(),p,dict(degradation='CHOP'),'condition_id_nunique',0)
# New owner results and fixed DSP diagnostics.
p=STAGE/'exp_C2/results/C0_weight_tensor_comparison_v3.csv'
q=pd.read_csv(p);r=row(q,MOS_key='pool.model.linear3.bias',fiveD_key='pool_layers.0.model.linear3.bias')
put('C.nisqa.bias_shift',abs(r.delta),p,dict(MOS_key='pool.model.linear3.bias',fiveD_key='pool_layers.0.model.linear3.bias'),'delta')
p=STAGE/'exp_C2/results/C3_MSE_decomposition_v1_v3.csv';q=pd.read_csv(p);parts=[]
for ds in ['FOR','P501','LIVETALK']:
    r=row(q,model='A_distill',dataset='NISQA_TEST_'+ds,domain='EQUAL_DOMAIN',level='file',dim='mos',mapping='none')
    if r is not None:parts.append(f'{ds}的均值偏移、尺度失配、相关残差分别为{r.mean_bias_squared:.4f}、{r.scale_mismatch_squared:.4f}、{r.linear_correlation_residual:.4f}')
if parts:put('C.mse.summary','；'.join(parts),p,dict(model='A_distill',domain='EQUAL_DOMAIN',level='file',dim='mos'),'mean_bias_squared|scale_mismatch_squared|linear_correlation_residual')
p=STAGE/'exp_B2/results/B2_dsp_legacy_summary_v3.csv'
if p.exists():
    q=pd.read_csv(p);q=q[q.grid_ms.eq(20)]
    put('B.dsp.summary','；'.join(f'{r.codec}{"零填充" if r.plc=="zero" else "原生补偿"}的帧F1均值为{r.frame_F1_mean:.4f}、丢包率MAE为{r.Ppl_MAE:.4f}；突发度在{r.burst_n}/{r.n}条可定义预测上Spearman为{r.BurstR_Spearman:.4f}' for r in q.itertuples()),p,dict(grid_ms=20),'BurstR_Spearman|Ppl_MAE')
p=STAGE/'exp_D2/measurements/FORMAL_VAL2700_A_distill_S2_20261012_030941/summary.csv'
if p.exists():
    q=pd.read_csv(p)
    r=q.iloc[0]
    put('D.v1.s2.rtf',r.RTF_median,p,dict(model='A_distill',setting='S2'),'RTF_median')
    put('D.v1.s2.seconds',r.end_to_end_s_median,p,dict(model='A_distill',setting='S2'),'end_to_end_s_median')
# Completed official Windows CUDA settings; exact cost owner summaries.
for tag,relative,model in [
('full','exp_D2/measurements/NATIVE_CUDA_VAL2700_OfficialNISQA5D_20261012_051048/summary.csv','OfficialNISQA5D'),
('mos','exp_D2/measurements/NATIVE_CUDA_VAL2700_OfficialNISQA_MOS_20261012_042138/summary.csv','OfficialNISQA_MOS')]:
    p=STAGE/relative
    if p.exists() and (p.parent/'COMPLETE.json').exists():
        q=pd.read_csv(p)
        r=row(q,model=model,setting='CUDA_WINDOWS',n_VAL=2700,warmup_repeats=3,measured_repeats=10)
        if r is not None:
            for label,column in [('seconds','end_to_end_s_median'),('seconds_iqr','end_to_end_s_IQR'),('rtf','RTF_median'),('front_seconds','frontend_s_median'),('infer_seconds','infer_s_median')]:
                put('D.gpu.'+tag+'.'+label,r[column],p,dict(model=model,setting='CUDA_WINDOWS',n_VAL=2700,warmup_repeats=3,measured_repeats=10),column)

# C-owner complete frontend development summaries, not final-model/test results.
p=STAGE/'exp_C2/results/C_A1_frontend_word_v3.csv'
if p.exists() and (STAGE/'exp_C2/C_A1_FRONTEND_COMPLETE_v3.json').exists():
    q=pd.read_csv(p);r=row(q,selected_frontend_by_A=True)
    if r is not None:
        put('A1.frontend.label',f"{int(r.sr)//1000} kHz、{int(r.n_mels)}梅尔通道、训练侧固定增益与四个运行电平统计",p,dict(selected_frontend_by_A=True),'sr|n_mels|normalization')
p=STAGE/'exp_C2/results/C_A1_frontend_factor_word_ranges_v3.csv'
if p.exists():
    q=pd.read_csv(p)
    for key,sel in [
        ('gain.loud.cv',dict(factor='gain_and_running_level_bundle',dim='loud',metric='cv_mae')),
        ('gain.loud.simlive',dict(factor='gain_and_running_level_bundle',dim='loud',metric='sim_to_live_mae')),
        ('rate.mos.cv',dict(factor='sample_rate48k_vs16k',dim='mos',metric='cv_mae')),
        ('rate.mos.loso',dict(factor='sample_rate48k_vs16k',dim='mos',metric='loso_sim_mae'))]:
        r=row(q,**sel)
        if r is not None:
            put('A1.factor.'+key+'.range',f'[{r.delta_min:+.4f}, {r.delta_max:+.4f}]',p,sel,'delta_min|delta_max')
            if key=='gain.loud.cv':put('A1.factor.gain.cells',r.matched_factor_cells,p,sel,'matched_factor_cells',0)

# C recomputation of existing single-seed temporal combination OOFs.
if (STAGE/'exp_C2/C_B4_existing_T_COMPLETE_v3.json').exists():
    p=STAGE/'exp_C2/results/C_B4_existing_T_main_v3.csv';q=pd.read_csv(p)
    for model,key in [('G0_oldT_newT','combo'),('G0_oldT_newT_no_pitch','combo_no_pitch')]:
        sel=dict(model=model,domain='EQUAL_DOMAIN',dim='mos',metric='MAE',mapping='none')
        r=row(q,**sel)
        if r is not None:put('B.'+key+'.mae',r.value,p,sel)
    p=STAGE/'exp_C2/results/C_B4_existing_T_paired_v3.csv';q=pd.read_csv(p)
    for comparator,key in [('G0','gain_vs_G0_ci'),('G0_oldT','gain_vs_oldT_ci'),('combo_noise31','gain_vs_noise_ci')]:
        sel=dict(candidate='G0_oldT_newT',comparator=comparator,domain='EQUAL_DOMAIN',dim='mos',metric='MAE',mapping='none')
        r=row(q,**sel)
        if r is not None:put('B.combo.'+key,f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'gain_comparator_minus_candidate|ci_low|ci_high')
    p=STAGE/'exp_C2/results/C_B4_existing_T_harmonic_paired_v3.csv';q=pd.read_csv(p)
    for comparator,loss,key in [('G0','random','harmonic_vs_g0_random_ci'),('G0','burst','harmonic_vs_g0_burst_ci'),('G0','burst_minus_random','harmonic_interaction_ci'),('G0_oldT','burst','harmonic_vs_old_burst_ci')]:
        sel=dict(candidate='G0_oldT_newT',comparator=comparator,loss_type=loss,domain='SIM',dim='mos',metric='MAE',mapping='none')
        r=row(q,**sel)
        if r is not None:put('B.combo.'+key,f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'gain_comparator_minus_candidate|ci_low|ci_high')


# Matched three-seed learned detector increment; all accuracy numbers from C.
if (STAGE/'exp_C2/C_B4_HGB_INCREMENT_COMPLETE_v3.json').exists():
    p=STAGE/'exp_C2/results/C_B4_HGB_increment_main_v3.csv';q=pd.read_csv(p)
    for model,key in [('G0_oldT_newT_no_pitch','base'),('G0_oldT_newT_B2_no_pitch','learned'),('G0_oldT_newT_B2_noise_no_pitch','noise')]:
        for dim in ['mos','dis']:
            sel=dict(model=model,domain='EQUAL_DOMAIN',dim=dim,metric='MAE',mapping='none');r=row(q,**sel)
            if r is not None:put(f'B.learned.{key}.{dim}.mae',r.value,p,sel)
    p=STAGE/'exp_C2/results/C_B4_HGB_increment_paired_v3.csv';q=pd.read_csv(p)
    for comparator,domain,key in [('G0_oldT_newT_no_pitch','EQUAL_DOMAIN','gain_equal_ci'),('G0_oldT_newT_B2_noise_no_pitch','EQUAL_DOMAIN','gain_noise_ci'),('G0_oldT_newT_no_pitch','SIM','gain_sim_ci'),('G0_oldT_newT_no_pitch','LIVE','gain_live_ci')]:
        sel=dict(candidate='G0_oldT_newT_B2_no_pitch',comparator=comparator,domain=domain,dim='mos',metric='MAE',mapping='none');r=row(q,**sel)
        if r is not None:put('B.learned.'+key,f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'gain_comparator_minus_candidate|ci_low|ci_high')
    p=STAGE/'exp_C2/results/C_B4_HGB_increment_harmonic_paired_v3.csv';q=pd.read_csv(p)
    for loss,key in [('random','gain_random_ci'),('burst','gain_burst_ci'),('burst_minus_random','interaction_ci')]:
        sel=dict(candidate='G0_oldT_newT_B2_no_pitch',comparator='G0_oldT_newT_no_pitch',domain='SIM',loss_type=loss,dim='mos',metric='MAE',mapping='none');r=row(q,**sel)
        if r is not None:put('B.learned.'+key,f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'gain_comparator_minus_candidate|ci_low|ci_high')


# v4 C completed dimension-specific increments and fixed representative capacity.
p=STAGE/'exp_C2/results/C_B4_HGB_allhead_paired_v4.csv'
if p.exists():
    q=pd.read_csv(p)
    for dom in ['SIM','LIVE','EQUAL_DOMAIN']:
        sel=dict(candidate='G0_oldT_newT_B2_no_pitch',comparator='G0_oldT_newT_no_pitch',domain=dom,dim='dis')
        r=row(q,**sel)
        if r is not None:put('B.learned.dis.'+dom+'.gain_ci',f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'gain_comparator_minus_candidate|ci_low|ci_high')
p=STAGE/'exp_C2/results/C_v1p5_complete_method_increments_v4.csv'
if p.exists():
    q=pd.read_csv(p);sel=dict(candidate='A2V3_gru96_layers2',dim='mos');r=row(q,**sel)
    if r is not None:
        for k in ['cv_mae_mean','loso_sim_mae_mean','sim_to_live_mae_mean','gain_cv_mae_vs_v1','gain_loso_sim_mae_vs_v1']:
            put('capacity.gru96_2.'+k,r[k],p,sel,k)
p=STAGE/'exp_B2/results/B3_shuffle_p.csv';q=pd.read_csv(p);sel=dict(family='old',group='T',role='primary');r=row(q,**sel)
if r is not None:put('B.shuffle.oldT.primary.p',r.p,p,sel,'p')
p=STAGE/'exp_C2/results/C_v1p5_baseline_v4.csv';q=pd.read_csv(p)
for dim in ['mos','loud']:
    r=row(q,model='v1.5',dim=dim)
    if r is not None:
        put('v1p5.'+dim+'.simlive',r.sim_to_live_mae_mean,p,dict(model='v1.5',dim=dim),'sim_to_live_mae_mean')
# Historical v1 SIMLIVE point values are C baseline fields, same selector as front table.
p=STAGE/'exp_C2/results/C_v1p5_complete_method_increments_v4.csv';q=pd.read_csv(p)
for dim in ['mos','loud']:
    r=row(q,candidate='A1V3_sr16000_mel48_global_gain_band0',dim=dim)
    if r is not None:put('v1.'+dim+'.simlive',r.v1_sim_to_live_mae,p,dict(candidate='A1V3_sr16000_mel48_global_gain_band0',dim=dim),'v1_sim_to_live_mae')
p=STAGE/'exp_B2/results/B2_dsp_legacy_summary_v3.csv';q=pd.read_csv(p);r=row(q,codec='Opus16k',plc='native_PLC',grid_ms=20)
for key,col in [('F1','frame_F1_mean'),('Ppl_MAE','Ppl_MAE'),('rho','BurstR_Spearman')]:put('B.dsp.opus.'+key,r[col],p,dict(codec='Opus16k',plc='native_PLC',grid_ms=20),col)
put('B.dsp.opus.coverage',f'{int(r.burst_n)}/{int(r.n)}',p,dict(codec='Opus16k',plc='native_PLC',grid_ms=20),'burst_n|n')
p=STAGE/'exp_B2/results/B0_defined_coverage_v4.csv'
if p.exists():
    q=pd.read_csv(p)
    for dom in ['SIM','LIVE']:
        r=row(q,domain=dom);put('B0.'+dom+'.defined_pct',r.defined_fraction*100,p,dict(domain=dom),'defined_fraction*100')
p=STAGE/'exp_C2/results/C_B0_transfer_paired_v4.csv'
if p.exists():
    q=pd.read_csv(p);r=row(q,candidate='confidence_weighted',comparator='G0_oldT_newT_no_pitch',domain='LIVE',dim='mos')
    put('B0.confidence.LIVE_MOS_ci',f'{r.gain_comparator_minus_candidate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,dict(candidate='confidence_weighted',comparator='G0_oldT_newT_no_pitch',domain='LIVE',dim='mos'),'gain_comparator_minus_candidate|ci_low|ci_high')

# Actual fresh-process RSS, table values and prose share the D CSV selector.
p=STAGE/'exp_D2/memory_v4/RSS_layers_v4.csv'
if p.exists() and (STAGE/'exp_D2/memory_v4/LITE_V1_COMPLETE.json').exists():
    q=pd.read_csv(p);r=q.iloc[0]
    for key,col in [('steady_RSS_MB','process_steady_RSS_median_MB'),('steady_IQR_MB','process_steady_RSS_IQR_MB'),('peak_RSS_MB','process_peak_RSS_MB')]:
        put('D.lite.'+key,r[col],p,dict(model=r.model),col)
p=STAGE/'exp_D2/D0_all_core_methods_TRAIN200_v4.csv'
if p.exists() and (STAGE/'exp_D2/D0_TRAIN200_uniform_v4_RECEIPT.json').exists():
    q=pd.read_csv(p)
    for model,key in [('A_distill','v1'),('OfficialNISQA5D','nisqa'),('PLCMOS','plc')]:
        r=row(q,model=model)
        put('D.uniform.'+key+'.MMAC',r.MMACeq_per_audio_s,p,dict(model=model),'MMACeq_per_audio_s')

# Historical tool-output exposure counts, CSV row and cell units only.
p=STAGE/'exp_C2/results/label_exposure_exact_counts.csv';q=pd.read_csv(p)
for col in ['complete_csv_rows','partial_csv_rows','complete_visible_rating_cells']:
    put('exposure.'+col,int(q[col].sum()),p,dict(scope='three TEST file/condition tables and LIVETALK listening-test table'),col+'_sum',0)

# Formal owners publish these summaries only after their real completion barriers.
for relative in ['exp_C2/results/C_WORD_SUMMARY_v3.csv','exp_D2/D_WORD_SUMMARY_v3.csv','exp_B2/results/B4_WORD_SUMMARY_v2.csv','exp_B2/results/B6_WORD_SUMMARY_v2.csv','exp_C2/results/C_WORD_PAPER_VALUES_v5.csv']:
    p=STAGE/relative
    if p.exists():
        q=pd.read_csv(p,keep_default_na=False)
        if q.key.duplicated().any():raise ValueError('Owner summary contains duplicate keys: '+relative)
        for r in q.itertuples():
            if r.value and str(r.value).strip().lower() not in ['nan','none','null','pending','pending_not_complete']:
                put(r.key,r.value,p,dict(key=r.key,owner_source=r.source_csv,owner_selector=r.selector),'value')
if 'v2.model.stop_loss_return_v1' in values:
    s=values['v2.model.stop_loss_return_v1'].lower()
    if s in ['true','1','false','0']:
        values['v2.decision.summary']='按预定止损规则保留首版作为最终主模型' if s in ['true','1'] else '按预定训练侧选择规则采用修订版作为最终主模型'
        mapping['v2.decision.summary']=mapping['v2.model.stop_loss_return_v1']
refs=(HERE/'reference_text.md').read_text(encoding='utf-8-sig').strip()

# Actual completed heldout detector rows, exact C extract only.
p=STAGE/'exp_C2/results/C_B2_heldout_report_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B2_HELDOUT_REPORT_COMPLETE_v4.json').exists():
    q=pd.read_csv(p)
    for source in ['test','thchs']:
        for plc in ['zero','repeat','native_opus','g711_appendixI']:
            r=row(q,source=source,stratum=plc,method='CNN_mask20')
            if r is not None:
                for col in ['frame_F1_pooled','Ppl_MAE','logBurstR_Spearman']:
                    put('B2.heldout.'+source+'.'+plc+'.'+col,r[col],p,dict(source=source,stratum=plc,method='CNN_mask20'),col)
                put('B2.heldout.'+source+'.'+plc+'.joint',f'{int(r.joint_BurstR_n)}/{int(r.n)}',p,dict(source=source,stratum=plc,method='CNN_mask20'),'joint_BurstR_n|n')


# Actual fixed B5 response rows; these are model responses without subjective labels.
p=STAGE/'exp_C2/results/C_B5_response_report_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B5_RESPONSE_REPORT_READY_v4.json').exists():
    q=pd.read_csv(p);r=row(q,table='monotonicity',model='v1_GRU',codec='Opus',plc='native_opus',metric='nonincreasing_all')
    for col,key in [('n_references','references'),('n_speakers','speakers')]:
        put('B5.'+key,int(r[col]),p,dict(table='monotonicity',model='v1_GRU',codec='Opus',plc='native_opus',metric='nonincreasing_all'),col,0)
    put('B.probe.summary','表7(c)比较丢包率上升时完整曲线非增的比例，表7(d)在10%丢包率和相同丢包计数下比较目标BurstR 8减1的原始MOS预测；四个固定模型已完成',p,dict(table='monotonicity|burst_minus_random',scope='seven registered PLC conditions'),'actual_completed_response_scope')
    e=row(q,table='emodel_rank',model='NISQA5D',metric='planning_Spearman')
    put('B.emodel.v3.summary',f'只有G.711 20 ms、Repeat1/silence、目标BurstR 4和目标丢包率1%的{int(e.n)}条规划参照匹配，共{int(e.network_conditions)}个网络条件；其余{int(e.unavailable_joint_coefficient_rows)}条缺少匹配系数。参照内相关来自源时长与丢包数离散舍入，不构成跨失真验证',p,dict(table='emodel_rank',model='NISQA5D',metric='planning_Spearman'),'n|network_conditions|unavailable_joint_coefficient_rows')
    if (STAGE/'exp_B2/B5_RESPONSE_ANALYSIS_COMPLETE_v3.json').exists():
        put('B.probe.v2.summary','修订候选已按同一配对探针完成冻结后响应，主模型与诊断候选身份分别保留',p,dict(table='monotonicity|burst_minus_random',model='selected_v2'),'actual_completed_response_scope')
p=STAGE/'exp_D2/D1_cost.csv'
if p.exists() and (STAGE/'exp_D2/PLCMOS_S3_ACTUAL_COMPLETE_RECEIPT_v4.json').exists():
    q=pd.read_csv(p);sel=dict(model='PLCMOS',setting='S3',measurement_id='FORMAL_VAL2700_PLCMOS_S3_20261012_042646');r=row(q,**sel)
    for key,col in [('rtf','RTF_median'),('rtf_IQR','RTF_IQR'),('seconds','end_to_end_s_median'),('seconds_IQR','end_to_end_s_IQR')]:
        put('D.plc.s3.'+key,r[col],p,sel,col)


# Actual completed C frequency intervention; no fresh quality computation here.
p=STAGE/'exp_C2/results/C_B48k_mask_main_full_precision_v4.csv'
if p.exists() and (STAGE/'exp_C2/C_B48K_MASK_COMPLETE_v4.json').exists():
    q=pd.read_csv(p)
    for protocol in ['CV','LOSO_SIM','SIM_TO_LIVE']:
        sel=dict(protocol=protocol,dim='mos');r=row(q,**sel)
        for column in ['natural_MAE','masked_MAE','gain_natural_minus_masked','ci_low','ci_high']:
            put('B48.'+protocol+'.'+column,r[column],p,sel,column)

# Completed submission deployment evidence; use the exact owner selectors.
p=STAGE/'exp_D2/D_WORD_TABLE6_v5.csv'
if p.exists():
    q=pd.read_csv(p)
    for model,key in [('A_distill','v1'),('OfficialNISQA5D','nisqa'),('v2_pure','main'),('v2_64','v2_64')]:
        for setting in ['S3','S1']:
            sel=dict(model=model,setting=setting,warmup_repeats=3,measured_repeats=10,n_VAL=2700)
            r=row(q,**sel)
            if r is not None:
                for suffix,col in [('rtf','RTF_median'),('rtf_IQR','RTF_IQR'),('seconds','end_to_end_s_median'),('seconds_IQR','end_to_end_s_IQR')]:
                    put('D.'+key+'.'+setting.lower()+'.'+suffix,r[col],p,sel,col)
                    if key=='main' and setting=='S3' and suffix in ['rtf','rtf_IQR']:
                        put('D.main.s3_'+suffix,r[col],p,sel,col)
p=STAGE/'exp_D2/v1_v15_sameTRAIN200_totals_v5.csv'
if p.exists():
    q=pd.read_csv(p)
    for model,key in [('v1','v1'),('v1.5','v1p5')]:
        r=row(q,model=model,TRAIN_n=200)
        if r is not None:
            for suffix,col in [('affine','model_affine_pool_MACeq_per_audio_s'),('total','native_full_matched_scope_MACeq_per_audio_s')]:
                put('D.component.'+key+'.'+suffix,r[col]/1e6,p,dict(model=model,TRAIN_n=200),col+' / 1e6')
p=STAGE/'exp_D2/D_lean_raw_RSS_v5.csv'
if p.exists() and (STAGE/'exp_D2/D_LEAN_RAW_RSS_COMPLETE_v5.json').exists():
    q=pd.read_csv(p)
    for model in ['v2_pure','v2_64']:
        sel=dict(model=model,implementation='rawcausal_numpy_minimal',TRAIN_n=200)
        r=row(q,**sel)
        if r is not None:
            put('D.lean.'+model+'.rss',r.steady_RSS_MB,p,sel,'steady_RSS_MB')
            put('D.lean.'+model+'.state',r.state_bytes_max,p,sel,'state_bytes_max',0)
            put('D.lean.'+model+'.error',f'{r.max_abs_error:.4e}',p,sel,'max_abs_error')
p=STAGE/'exp_C2/results/C_INT8_engineering_report_v5.csv'
if p.exists() and (STAGE/'exp_C2/C_INT8_ENGINEERING_REPORT_READY_v5.json').exists():
    q=pd.read_csv(p)
    for model in ['v2_pure','v2_64']:
        sel=dict(model=model,domain='pooled',dimension='mos')
        r=row(q,**sel)
        if r is not None:put('D.int8.'+model+'.file_error',r.INT8_vs_FP32_max_abs_prediction_difference,p,sel,'INT8_vs_FP32_max_abs_prediction_difference')
p=STAGE/'exp_D2/D_WORD_SUMMARY_v5.csv'
if p.exists():
    q=pd.read_csv(p,keep_default_na=False)
    for key in ['D.main.macs','D.main.total_mac_estimate','D.main.state_bytes','D.stream.max_error','D.stream.chunk10','D.stream.chunk20','D.stream.chunk100']:
        r=row(q,key=key)
        if r is not None:
            x=float(r.value)
            value=f'{x/1e6:.4f}百万MAC等价量/音频秒' if key in ['D.main.macs','D.main.total_mac_estimate'] else str(int(x)) if key=='D.main.state_bytes' else f'{x:.4e}' if key=='D.stream.max_error' else f'{x:.4f} ms'
            put(key,value,p,dict(key=key,owner_source=r.source_csv,owner_selector=r.selector),'value')
            if key=='D.main.total_mac_estimate':
                put('D.main.mac_million',x/1e6,p,dict(key=key,owner_source=r.source_csv,owner_selector=r.selector),'value / 1e6')
    r=row(q,key='D.stream.delay')
    if r is not None:put('D.stream.delay',f'FIR最多{float(r.value):.4f} ms，另加20 ms谱窗与块调度',p,dict(key='D.stream.delay'),'value')
    put('D.stream.limitation','原始音频因果前端与固定TRAIN增益；原生CUDA资格、精简CPU资格及正式CPU计时分别报告',p,dict(key='D.stream.limitation'),'value')
p=STAGE/'exp_D2/INT8_stream_small_v5/COMPLETE.json'
if p.exists():
    r=json.loads(p.read_text(encoding='utf-8-sig'));put('D.int8.stream.error',r['max_abs_error'],p,dict(rows=24,passed_all=False),'max_abs_error')

from cost_v3_publication import ready as cost_v3_ready, fill_cost_numbers
if cost_v3_ready():
    fill_cost_numbers(put)
# Existing early frontend comparison: exact registered candidate, not new GRU96 outcomes.
p=STAGE/'exp_C2/results/C_A1_frontend_summary_v3.csv'
q=pd.read_csv(p)
for dim in ['mos','loud']:
    r=row(q,candidate='A1V3_sr16000_mel48_global_gain_band0',dim=dim)
    put('early.frontend.'+dim+'.peak_simlive',r.baseline_sim_to_live_mae,p,dict(candidate=r.candidate,dim=dim),'baseline_sim_to_live_mae')
    put('early.frontend.'+dim+'.gain_simlive',r.sim_to_live_mae_mean,p,dict(candidate=r.candidate,dim=dim),'sim_to_live_mae_mean')

# Completed postfreeze GRU96 four-arm TRAIN-only comparison; no new statistics.
front96 = HERE.parent / 'frontend/rule_meanprediction_20261014'
p = front96 / 'TRAIN_frontend_metrics_meanprediction.csv'
front96_metrics = pd.read_csv(p)
for rec in front96_metrics.itertuples():
    sel = dict(arm=rec.arm, protocol=rec.protocol, dim=rec.dim, metric=rec.metric)
    put(f'front96.{rec.arm}.{rec.protocol}.{rec.dim}.{rec.metric}', rec.value, p, sel, 'value')
p = front96 / 'TRAIN_frontend_paired_meanprediction.csv'
front96_paired = pd.read_csv(p)
for rec in front96_paired.itertuples():
    sel = dict(comparison=rec.comparison, protocol=rec.protocol, dim=rec.dim, metric=rec.metric)
    key = f'front96.{rec.comparison}.{rec.protocol}.{rec.dim}.{rec.metric}'
    for col in ['difference', 'ci_low', 'ci_high']:
        put(key+'.'+col, getattr(rec,col), p, sel, col, 4)
    put(key+'.interval', f'{rec.difference:+.4f} [{rec.ci_low:+.4f}, {rec.ci_high:+.4f}]', p, sel, 'difference|ci_low|ci_high')

# Display primary MAE improvement as A-B. This reverses an existing contrast,
# not a recomputation of metrics or bootstrap draws. SIM_TO_LIVE uses D's tier CSV.
for rec in front96_paired.itertuples():
    if rec.comparison!='B-A' or rec.metric!='MAE':continue
    sel=dict(comparison='B-A',protocol=rec.protocol,dim=rec.dim,metric='MAE',display_orientation='A-B')
    key=f'front96.A-B.{rec.protocol}.{rec.dim}.MAE'
    put(key+'.interval',f'{-rec.difference:+.4f} [{-rec.ci_high:+.4f}, {-rec.ci_low:+.4f}]',p,sel,'-difference|-ci_high|-ci_low')
p=front96/'tier_improvements_A_minus_B.csv'
for rec in pd.read_csv(p).itertuples():
    sel=dict(dim=rec.dim,comparison='A-B',metric='MAE')
    key=f'front96.A-B.SIM_TO_LIVE.{rec.dim}.MAE'
    put(key+'.interval',f'{rec.improvement:+.4f} [{rec.ci_low:+.4f}, {rec.ci_high:+.4f}]',p,sel,'improvement|ci_low|ci_high')

# Reader-facing VAL comparisons and post-result B-D auxiliary contrast.
for model in ['PLCMOS','UTMOSv2','DNSMOS','SQUIM']:
    sel=dict(model=model,dataset='VAL',domain='EQUAL_DOMAIN',level='file',dim='mos',metric='Pearson',mapping='none')
    r=row(accuracy,**sel)
    if r is not None:put('paper.val.'+model+'.r',r.value,accuracy_path,sel)
p=front96/'aux_B_D_20261014/B_minus_D_SIMLIVE_MOS_MAE_meanprediction.csv'
for rec in pd.read_csv(p).itertuples():
    sel=dict(comparison='B-D',protocol='SIM_TO_LIVE',dim='mos',metric='MAE')
    put('front96.B-D.SIM_TO_LIVE.mos.MAE.interval',f'{rec.difference:+.4f} [{rec.ci_low:+.4f}, {rec.ci_high:+.4f}]',p,sel,'difference|ci_low|ci_high')

# Self-contained journal values from already completed owner outputs only.
p=HERE.parent/'firstpaper_VAL_MSE_20261014/VAL_main_raw_MOS_MSE_decomposition.csv'
q=pd.read_csv(p);sel=dict(model='v2_pure',dataset='VAL',domain='EQUAL_DOMAIN',level='file',dim='mos',mapping='none');r=row(q,**sel)
for col in ['MSE','mean_bias_squared','scale_mismatch_squared','linear_correlation_residual']:
    put('paper.mse.val.'+col,r[col],p,sel,col)
p=STAGE/'exp_C2/results/C3_MSE_decomposition_new_v2_v3.csv'
q=pd.read_csv(p);sel=dict(model='v2_pure',dataset='NISQA_TEST_LIVETALK',domain='EQUAL_DOMAIN',level='file',dim='mos',mapping='none');r=row(q,**sel)
put('paper.mse.live.linear_correlation_residual',r.linear_correlation_residual,p,sel,'linear_correlation_residual')
p=STAGE/'exp_C2/results/C5_Holm72_v3.csv';q=pd.read_csv(p)
for dataset in ['FOR','P501','TCD_VOIP']:
    for metric in ['Pearson','RMSE']:
        sel=dict(dataset=('NISQA_TEST_'+dataset if dataset in ['FOR','P501'] else dataset),candidate='v2_pure',comparator='PLCMOS',domain='EQUAL_DOMAIN',level='file',dim='mos',metric=metric)
        r=row(q,**sel)
        if r is not None:
            key='paper.paired.'+dataset+'.PLCMOS.'+metric
            put(key+'.interval',f'{r.difference_a_minus_b:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,sel,'difference_a_minus_b|ci_low|ci_high')
            put(key+'.holm',r.p_Holm,p,sel,'p_Holm')
p=STAGE/'exp_C2/results/C_B48k_mask_main_full_precision_v4.csv';q=pd.read_csv(p)
r=row(q,protocol='CV',dim='mos');put('paper.highfreq.cv_degradation',r.masked_MAE-r.natural_MAE,p,dict(protocol='CV',dim='mos'),'masked_MAE-natural_MAE')
r=row(q,protocol='SIM_TO_LIVE',dim='mos');put('paper.highfreq.simlive.interval',f'{r.gain_natural_minus_masked:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',p,dict(protocol='SIM_TO_LIVE',dim='mos'),'gain_natural_minus_masked|ci_low|ci_high')
p=HERE.parent/'firstpaper_VAL_MSE_20261014/existing_monotonicity_seven_conditions_selected_rows.csv';q=pd.read_csv(p)
r=q[q.model.eq('selected_v2')&q.metric.eq('nonincreasing_all')]
for aggregate in ['min','max']:
    put('paper.mono.'+aggregate,getattr(r.value,aggregate)(),p,dict(model='selected_v2',metric='nonincreasing_all',conditions='seven'), 'value_'+aggregate)
p=STAGE/'exp_C2/results/C_B5_response_report_v4.csv';q=pd.read_csv(p);sel=dict(table='emodel_rank',model='NISQA5D',metric='planning_Spearman');r=row(q,**sel)
for key,col in [('n','n'),('conditions','network_conditions'),('unavailable','unavailable_joint_coefficient_rows')]:
    put('paper.emodel.'+key,r[col],p,sel,col,0)
p=STAGE/'exp_D/cost_v3/results/cost_final.csv';q=pd.read_csv(p)
r=q[q.method.eq('v2_96')&q.path.eq('deploy')&q.setting.isin(['S1','S2','S3'])]
put('paper.deploy.max_error',f'{r.prediction_max_abs.max():.2e}',p,dict(method='v2_96',path='deploy',setting=['S1','S2','S3']),'prediction_max_abs_max')

# Journal display precision for measured RSS and chunk runtimes.
p=STAGE/'exp_D/cost_v3/results/streaming_kernel.csv';q=pd.read_csv(p)
# Reuse the exact selector already registered by the cost publication helper.
for key,original,digits in [('rss','D.cost.stream.v2_pure.10.resident_rss_mb',1),('p99','D.cost.stream.v2_pure.10.block_ms_p99',2),('median','D.cost.stream.v2_pure.10.block_ms_median',2)]:
    values['paper.stream.'+key]=f'{float(values[original]):.{digits}f}'
    mapping['paper.stream.'+key]={**mapping[original],'display_digits':digits}

# Conservative native arithmetic ratios from completed owner contrasts.
p=STAGE/'exp_D/cost_v3/results/MAC_contrasts.csv';q=pd.read_csv(p)
for label,reference,rounding in [('NISQA','NISQA_MOS_official','lower integer denominator'),('PLCMOS','PLCMOS_official','nearest ten denominator')]:
    sel=dict(reference=reference,target='v2_96_native');r=row(q,**sel)
    denominator=int(r.estimated_MAC_ratio) if label=='NISQA' else int(round(r.estimated_MAC_ratio/10)*10)
    put('paper.mac.native.'+label+'.conservative_fraction','1/'+str(denominator),p,sel,'estimated_MAC_ratio; display '+rounding)

# Existing S3 failure values are percentages, not confidence intervals.
p=STAGE/'exp_D/cost_v3/results/cost_final.csv';q=pd.read_csv(p)
for method in ['NISQA_MOS','PLCMOS']:
    sel=dict(method=method,path='official',setting='S3');r=row(q,**sel)
    for label,col in [('iqr_pct','iqr_ratio'),('drift_pct','calib_drift')]:
        put('paper.cost.'+method+'.S3.'+label,r[col]*100,p,sel,col+'*100',6)

text=(HERE/'manuscript.md').read_text(encoding='utf-8-sig').replace('{{REFERENCES}}',refs)
for key,value in values.items():text=text.replace('{{'+key+'}}',value)
# Retain the continuous verified citation ranges requested by the author.
(HERE/'manuscript_filled.md').write_text(text,encoding='utf-8')
(HERE/'number_map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
pending=sorted(set(re.findall(r'\{\{([^{}]+)\}\}',text)))
(HERE/'number_receipt.json').write_text(json.dumps(dict(filled=len(values),pending=pending,accuracy_source=str(accuracy_path),cost_source=str(STAGE/'exp_D/cost_v3/results/cost_final.csv') if cost_v3_ready() else 'historical_cost_v5',status='ecice_selfcontained_v32_source_complete_doi_and_author_deferred' if cost_v3_ready() else 'working_draft_remaining_cost_matrix_and_author_metadata',frontend_robustness_complete=True,author_metadata_deferred=True,submission_final=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(filled=len(values),pending=pending),ensure_ascii=False))

