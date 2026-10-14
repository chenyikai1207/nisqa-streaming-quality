"""Four Chinese grayscale main figures from existing owner CSVs; no inference or statistics."""
from pathlib import Path
import json,shutil,sys,datetime,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch

H=Path(__file__).resolve().parents[1]
S=H/'code/R4'
OUT=H/'figures'
# Resolve installed fonts; no private font directory or bundled proprietary font.
families={f.name for f in fm.fontManager.ttflist}
chinese=next((n for n in ['STZhongsong','Noto Serif CJK SC','Source Han Serif SC','Microsoft YaHei','SimSun'] if n in families),'DejaVu Sans')
english='Times New Roman' if 'Times New Roman' in families else 'DejaVu Serif'
CN=Path(fm.findfont(fm.FontProperties(family=chinese)))
EN=Path(fm.findfont(fm.FontProperties(family=english)))
plt.rcParams.update({'font.family':[english,chinese],
 'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'text.color':'#111111','axes.labelcolor':'#111111','axes.edgecolor':'#111111',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white',
 'hatch.linewidth':.6,'axes.unicode_minus':False})
SETS=['VAL','NISQA_TEST_FOR','NISQA_TEST_P501','NISQA_TEST_LIVETALK']
SETLABEL=['验证集（VAL）','FOR测试集','P501测试集','真实通话（LIVETALK）']

def save(fig,name,multi=True):
 fig.canvas.draw();prefix=OUT/name
 for ext in ['svg','pdf','png']:fig.savefig(str(prefix)+'.'+ext,dpi=600)
 plt.close(fig)

def panel(ax,letter,title):
 ax.set_title('('+letter+') '+title,loc='left',pad=8)

def framework():
 fig,ax=plt.subplots(figsize=(172/25.4,58/25.4));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
 ax.set_xlim(0,172);ax.set_ylim(58,0);ax.axis('off')
 nodes=[('wave',1,2,30.5,22,'输入波形','16/48 kHz\n因果分块'),('front',35,2,30.5,22,'频谱前端','训练固定增益\n48通道对数梅尔谱'),('gru',69,2,30.5,22,'递归编码','双层单向GRU96'),('pool',103,2,30.5,22,'注意力池化','逐块递增更新'),('output',137,2,33,22,'质量输出','拼接原始电平统计\n总体质量、噪声程度\n音色、不连续性、响度'),('time',1,34,30.5,17,'时间分析','顺序及突发特征'),('front_state',35,34,30.5,17,'前端状态','滤波与窗历史\n重采样历史'),('gru_state',69,34,30.5,17,'递归状态','两层隐状态'),('pool_state',103,34,30.5,17,'池化状态','累计量递增更新'),('raw4',137,34,33,17,'原始电平统计','归一化前4项\n独立有界累计')]
 edges=[]
 for id,x,y,w,h,title,body in nodes:
  ax.add_patch(Rectangle((x,y),w,h,fc='white',ec='black',lw=.85))
  ax.text(x+w/2,y+4,title,ha='center',va='center',fontsize=8.5)
  ax.text(x+w/2,y+h*.66,body,ha='center',va='center',fontsize=8.1,linespacing=1.18)
 def arrow(a,b,dashed=False):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.85,color='black',linestyle='--' if dashed else '-'));edges.append(dict(start_mm=a,end_mm=b,dashed=dashed))
 for x in [31.5,65.5,99.5,133.5]:arrow((x+.25,13),(x+3.2,13))
 for x in [50.25,84.25,118.25,153.5]:arrow((x,33.5),(x,24.5))
 arrow((16.25,24.5),(16.25,33.5),True)
 points=[(1,13),(.2,13),(.2,56),(153.5,56),(153.5,51.5)]
 ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color='black',lw=.85)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,lw=.85,color='black'))
 edges.append(dict(start_mm=points[0],end_mm=points[-1],bend_points_mm=points[1:-1],points_mm=points,dashed=False,role='raw4_before_normalization_bypass_to_output_head'))
 save(fig,'fig1_framework_zh_v32',False)
 (OUT/'fig1_nodes_edges_zh_v32.json').write_text(json.dumps(dict(canvas_mm=[172,58],origin='top_left',font_chinese='STZhongsong',font_english='Times New Roman',font_min_pt=8.1,nodes=[dict(id=i,x_mm=x,y_mm=y,width_mm=w,height_mm=h,title=t,body=b) for i,x,y,w,h,t,b in nodes],edges=edges,raw4_scientific_connection='raw amplitude stats before normalization concatenate with pooled GRU features at output head; not frequency input',free_text=[]),ensure_ascii=False,indent=2),encoding='utf-8')

def quality():
 data=pd.read_csv(S/'exp_C2/results/C_primary_accuracy_v3.csv')
 models=['A_distill','v2_pure','A_G0','A_newT','NISQA5D','PLCMOS','SQUIM','DNSMOS','P563','UTMOSv2']
 labels=['首版','本文模型','HGB（G0）','HGB（新T）','NISQA五维','PLCMOS','SQUIM','DNSMOS','P.563','UTMOSv2']
 record=[];missing=[];markers=['o','s','^','D']
 fig,axes=plt.subplots(1,2,figsize=(174/25.4,92/25.4),sharey=True)
 fig.subplots_adjust(left=.20,right=.985,bottom=.16,top=.82,wspace=.22)
 for k,(metric,mapping,title) in enumerate([('Pearson','none','原始相关'),('RMSE','cubic','训练侧映射误差')]):
  ax=axes[k];own=[]
  for j,(ds,marker) in enumerate(zip(SETS,markers)):
   for i,model in enumerate(models):
    m='none' if metric=='RMSE' and model.startswith('NISQA') else mapping
    q=data[data.model.eq(model)&data.dataset.eq(ds)&data.domain.eq('EQUAL_DOMAIN')&data.level.eq('file')&data.dim.eq('mos')&data.metric.eq(metric)&data.mapping.eq(m)]
    if len(q)>1:
     v=q[q.protocol.eq('C_V2_RAW_OR_TRAIN_OOF_CALIBRATION')]
     q=v if len(v)==1 else q[q.protocol.eq('V1_ORIGINAL_VAL_UNCHANGED' if ds=='VAL' else 'V1_ORIGINAL_UNCHANGED')]
    if len(q)>1:raise RuntimeError(f'nonunique published row {model}/{ds}/{metric}')
    if len(q)==0:missing.append(dict(model=model,dataset=ds,metric=metric,mapping=m));continue
    row=q.iloc[0];v=float(row.value);lo=float(row.ci_low);hi=float(row.ci_high);y=i+(j-1.5)*.15
    color='black' if model=='v2_pure' else '#555555'
    if np.isfinite(lo) and np.isfinite(hi) and lo<=v<=hi:ax.plot([lo,hi],[y,y],color=color,lw=.65,zorder=2)
    ax.plot(v,y,marker=marker,ms=5.1 if model=='v2_pure' else 4.3,mfc='white' if j%2==0 else color,mec=color,mew=.9,ls='none',zorder=3)
    record.append(row.to_dict());own.append(row)
  ax.set_yticks(range(len(models)),labels);ax.set_ylim(len(models)-.6,-.6);ax.tick_params(axis='y',length=0)
  ax.grid(axis='x',color='#DDDDDD',lw=.45);ax.set_axisbelow(True);panel(ax,'ab'[k],title)
  ax.set_xlabel('Pearson相关系数' if metric=='Pearson' else '总体质量均方根误差（RMSE）')
  values=[float(r.value) for r in own]+[float(r.ci_low) for r in own if np.isfinite(r.ci_low)]+[float(r.ci_high) for r in own if np.isfinite(r.ci_high)]
  ax.set_xlim(min(.1,min(values)-.02) if k==0 else min(.33,min(values)-.02),max(1.02,max(values)+.02) if k==0 else max(1.34,max(values)+.02));ax.xaxis.set_major_locator(plt.MaxNLocator(4))
 for tick in axes[0].get_yticklabels():
  if tick.get_text()=='本文模型':tick.set_fontweight('bold')
 handles=[Line2D([0],[0],marker=m,ls='none',mec='black',mfc='white' if i%2==0 else 'black',ms=5) for i,m in enumerate(markers)]
 fig.legend(handles,['验证集（VAL）','FOR测试集','P501测试集','真实通话（LIVETALK）'],loc='upper center',bbox_to_anchor=(.54,.985),ncol=4,fontsize=8.5,columnspacing=.8,handletextpad=.4)
 save(fig,'fig2_quality_zh_v32');pd.DataFrame(record).to_csv(OUT/'fig2_source_rows_zh_v32.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
 return dict(source=str(S/'exp_C2/results/C_primary_accuracy_v3.csv'),rows=len(record),missing=missing,UTMOS_no_TRAIN_mapping=True)

def mse():
 published=pd.read_csv(OUT/'fig3_source_rows_zh_v32.csv')
 old=published[published.model.ne('v2_pure')].copy()
 new=published[published.model.eq('v2_pure')].copy()
 val_source=H/'VAL_MSE/VAL_main_raw_MOS_MSE_decomposition.csv'
 models=['A_distill','v2_pure','NISQA5D','PLCMOS','DNSMOS','SQUIM','P563']
 labels=['首版','本文\n模型','NISQA\n五维','PLC\nMOS','DNS\nMOS','SQUIM','P.563']
 cols=['mean_bias_squared','scale_mismatch_squared','linear_correlation_residual'];gray=['white','#BBBBBB','#777777'];hatches=['///','..',r'\\']
 records=[];missing=[]
 fig,axes=plt.subplots(2,2,figsize=(174/25.4,106/25.4),sharey=True)
 fig.subplots_adjust(left=.105,right=.985,bottom=.16,top=.93,wspace=.23,hspace=.55)
 upper=0
 for ax,ds,title,letter in zip(axes.flat,SETS,SETLABEL,'abcd'):
  for i,model in enumerate(models):
   d=new if model=='v2_pure' else old
   q=d[d.model.eq(model)&d.dataset.eq(ds)&d.domain.eq('EQUAL_DOMAIN')&d.level.eq('file')&d.dim.eq('mos')&d.mapping.eq('none')]
   if len(q)>1:raise RuntimeError(f'nonunique decomposition {model}/{ds}')
   if q.empty:
    missing.append(dict(model=model,dataset=ds));ax.text(i,.055,'—',ha='center',va='bottom',fontsize=9);continue
   r=q.iloc[0];bottom=0
   for col,g,h in zip(cols,gray,hatches):
    v=float(r[col]);ax.bar(i,v,bottom=bottom,facecolor=g,edgecolor='black',lw=.4,width=.6,hatch=h);bottom+=v
   v=float(r.MSE);lo=float(r.MSE_ci_low);hi=float(r.MSE_ci_high)
   if np.isfinite(lo) and np.isfinite(hi) and lo<=v<=hi:ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor='black',lw=.8,capsize=2)
   upper=max(upper,hi if np.isfinite(hi) else v);records.append(r.to_dict()|{'plot_dataset':ds})
  ax.set_xticks(range(7),labels,rotation=0,ha='center');ax.grid(axis='y',color='#DDDDDD',lw=.45);ax.set_axisbelow(True);panel(ax,letter,title)
 for ax in axes.flat:ax.set_ylim(0,upper*1.12);ax.set_xlim(-.65,6.65)
 for ax in axes[:,0]:ax.set_ylabel('原值均方误差（MSE）',labelpad=5)
 fig.legend([Patch(facecolor=g,edgecolor='black',hatch=h,lw=.4) for g,h in zip(gray,hatches)],['均值偏移平方','相关调整尺度失配平方','线性相关残差'],ncol=3,loc='lower center',bbox_to_anchor=(.54,.008),fontsize=8)
 save(fig,'fig3_mse_seven_methods_zh_v32');pd.DataFrame(records).to_csv(OUT/'fig3_source_rows_zh_v32.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
 return dict(legacy_source=str(S/'exp_C2/results/C3_MSE_decomposition_all_existing_v4.csv'),v2_source=str(S/'exp_C2/results/C3_MSE_decomposition_new_v2_v3.csv'),VAL_source=str(val_source),VAL_analysis_identity='postfreeze_requested_raw_VAL_MSE_20261014',rows=len(records),missing=missing,statistics_recomputed=False)

def pareto():
 source=OUT/'fig4_source_rows_zh_v32.csv';data=pd.read_csv(source);data=data[data.metric.eq('Pearson')].copy()
 scenarios=['VAL','NISQA_TEST_3SET_EQUAL_MACRO','TCD_VOIP'];titles=['验证集（VAL）','三测试集等权宏平均','TCD-VoIP测试集']
 labels={'v1':'首版','v2_pure':'双层GRU96','v2_64':'双层GRU64','NISQA_MOSONLY':'NISQA仅MOS','NISQA5D':'NISQA五维','PLCMOS':'PLCMOS'}
 marks={'v1':'o','v2_pure':'P','v2_64':'D','NISQA_MOSONLY':'X','NISQA5D':'s','PLCMOS':'^'}
 offsets={'v1':(5,-15,'left'),'v2_pure':(5,-13,'left'),'v2_64':(-3,-13,'right'),'NISQA_MOSONLY':(6,-13,'left'),'NISQA5D':(-5,14,'right'),'PLCMOS':(-5,-11,'right')}
 records=[]
 fig,axes=plt.subplots(3,2,figsize=(174/25.4,138/25.4),sharey=True)
 fig.subplots_adjust(left=.11,right=.98,bottom=.09,top=.94,wspace=.28,hspace=.67)
 for i,(ds,title) in enumerate(zip(scenarios,titles)):
  for k,axis in enumerate(['MAC','RTF']):
   ax=axes[i,k];q=data[data.dataset.eq(ds)&data.axis.eq(axis)];plot=q[np.isfinite(q.x)&np.isfinite(q.value)&q.status.str.startswith('PLOTTED')]
   for _,r in plot.iterrows():
    model=r.model;x=float(r.x);y=float(r.value);accepted=axis=='MAC' or str(r.RTF_environment_accepted).lower()=='true';label=labels[model]+('†' if not accepted else '')
    dx,dy,ha=offsets[model]
    if axis=='RTF' and model=='NISQA_MOSONLY':dx,dy,ha=6,12,'left'
    if axis=='RTF' and model=='v2_64':dx,dy,ha=12,14,'right'
    if axis=='RTF' and model=='v2_pure' and i==0:dx,dy,ha=5,-17,'left'
    if i==0 and axis=='MAC' and model=='v2_64':dx,dy,ha=0,20,'left'
    if i==1 and model=='v1':dy=-11
    if i==1 and model=='v2_pure':dy=-8
    if axis=='RTF' and np.isfinite(r.RTF_q1) and np.isfinite(r.RTF_q3):
     low=float(r.RTF_q1);high=float(r.RTF_q3)
     if low<=x<=high:ax.errorbar(x,y,xerr=[[x-low],[high-x]],fmt='none',ecolor='#444444',capsize=1.7,lw=.6,zorder=2)
    ax.plot(x,y,marker=marks[model],ms=6.2 if model=='v2_pure' else 5.4,mfc='black' if accepted else 'white',mec='black',mew=.95,ls='none',zorder=3)
    ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=8.5,fontweight='bold' if model=='v2_pure' else 'normal')
    records.append(r.to_dict())
   # Reproduce existing frontier connections, using the existing point table only.
   seq=(['v1','v2_64','v2_pure','NISQA_MOSONLY'] if i==0 else ['v1','NISQA_MOSONLY'] if i==1 else ['v1','v2_pure','NISQA5D']) if axis=='MAC' else (['v1','v2_pure'] if i==2 else [])
   link=plot.set_index('model').reindex(seq).dropna(subset=['x','value']) if seq else pd.DataFrame()
   if len(link)>1:ax.plot(link.x,link.value,'--',color='#666666',lw=.8,zorder=1)
   ax.set_xscale('log');ax.set_ylim(.56,.98);ax.set_xlim((1.7,7000) if axis=='MAC' else (.0035,.22));ax.set_yticks([.6,.7,.8,.9]);panel(ax,'abcdef'[i*2+k],title)
   ax.set_xticks([10,100,1000] if axis=='MAC' else [.01,.1],['10','100','1000'] if axis=='MAC' else ['0.01','0.1']);ax.minorticks_off()
   ax.set_xlabel('含前端计算量（百万MACeq/音频秒）' if axis=='MAC' else '实时因子（RTF，5轮中位数/IQR）')
   if k==0:ax.set_ylabel('原始Pearson相关系数 ↑')
 save(fig,'fig4_pareto_zh_v32');pd.DataFrame(records).to_csv(OUT/'fig4_source_rows_zh_v32.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
 return dict(source=str(source),points=len(records),excluded_nonplot_rows=data[~(np.isfinite(data.x)&np.isfinite(data.value)&data.status.str.startswith('PLOTTED'))][['axis','dataset','model','status','reason']].to_dict('records'),cost_or_quality_recomputed=False)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['1','2','3','4','all'],default='all');args=ap.parse_args()
 receipt={'font_chinese':fm.FontProperties(fname=CN).get_name(),'font_english':english,'dpi':600,'SVG_editable_text':True,'palette':'grayscale_white_no_background_or_shadow','scientific_recomputations':0,'fits':0,'forwards':0,'timers':0,'root_records_or_manuscript_tables_SI_changed':False}
 if args.only in ['1','all']:framework();receipt['fig1']='fig1_nodes_edges_zh_v32.json'
 if args.only in ['2','all']:receipt['fig2']=quality()
 if args.only in ['3','all']:receipt['fig3']=mse()
 if args.only in ['4','all']:receipt['fig4']=pareto()
 receipt['time']=datetime.datetime.now().isoformat()
 print('Actual Chinese grayscale figure exports complete; history preserved; no scientific recalculation')
