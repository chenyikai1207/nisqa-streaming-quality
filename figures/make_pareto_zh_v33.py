"""Chinese figure 4 from saved published point CSVs; no inference or quality statistics."""
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

def pareto():
 source=OUT/'fig4_all_points_source_zh_v33.csv';data=pd.read_csv(source);data=data[data.metric.eq('Pearson')].copy()
 scenarios=['VAL','NISQA_TEST_3SET_EQUAL_MACRO','TCD_VOIP'];titles=['验证集（VAL）','三测试集等权宏平均','TCD-VoIP测试集']
 labels={'v1':'首版','v2_pure':'双层GRU96','v2_64':'双层GRU64','NISQA_MOSONLY':'NISQA仅MOS','NISQA5D':'NISQA五维','PLCMOS':'PLCMOS'}
 marks={'v1':'o','v2_pure':'P','v2_64':'D','NISQA_MOSONLY':'X','NISQA5D':'s','PLCMOS':'^'}
 offsets={'v1':(5,-15,'left'),'v2_pure':(5,-13,'left'),'v2_64':(-3,-13,'right'),'NISQA_MOSONLY':(6,-13,'left'),'NISQA5D':(-5,14,'right'),'PLCMOS':(-5,-11,'right')}
 records=[]
 fig,axes=plt.subplots(3,2,figsize=(174/25.4,138/25.4),sharey=True)
 fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.94,wspace=.28,hspace=.67)
 for i,(ds,title) in enumerate(zip(scenarios,titles)):
  for k,axis in enumerate(['MAC','RTF']):
   ax=axes[i,k];q=data[data.dataset.eq(ds)&data.axis.eq(axis)];plot=q[np.isfinite(q.x)&np.isfinite(q.value)]
   for _,r in plot.iterrows():
    model=r.model;x=float(r.x);y=float(r.value);accepted=axis=='MAC' or str(r.RTF_environment_accepted).lower()=='true';label=labels.get(model,str(model))
    dx,dy,ha=offsets.get(model,(5,11,'left'))
    if axis=='RTF' and model=='NISQA_MOSONLY':dx,dy,ha=6,12,'left'
    if axis=='RTF' and model=='v2_64':dx,dy,ha=12,14,'right'
    if axis=='RTF' and model=='v2_pure' and i==0:dx,dy,ha=5,-17,'left'
    if i==0 and axis=='MAC' and model=='v2_64':dx,dy,ha=0,20,'left'
    if i==1 and model=='v1':dy=-11
    if i==1 and model=='v2_pure':dy=-8
    if axis=='RTF' and np.isfinite(r.RTF_q1) and np.isfinite(r.RTF_q3):
     low=float(r.RTF_q1);high=float(r.RTF_q3)
     if low<=x<=high:ax.errorbar(x,y,xerr=[[x-low],[high-x]],fmt='none',ecolor='#444444',capsize=1.7,lw=.6,zorder=2)
    ax.plot(x,y,marker=marks.get(model,{'DNSMOS':'v','SQUIM':'*','P563':'h'}.get(model,'o')),ms=7.8 if axis=='MAC' and model=='NISQA5D' else 6.2 if model=='v2_pure' else 5.4,mfc='#BBBBBB' if axis=='MAC' and model=='NISQA_MOSONLY' else 'black',mec='black',mew=.95,ls='none',zorder=2 if axis=='MAC' and model=='NISQA5D' else 3)
    ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=8.5,fontweight='bold' if model=='v2_pure' else 'normal')
    records.append(r.to_dict())
   # Pareto display geometry: increasing cost, keep strictly increasing quality.
   frontier=[];best_quality=-np.inf
   for _,point in plot.sort_values(['x','value'],ascending=[True,False]).iterrows():
    if float(point.value)>best_quality:
     frontier.append((float(point.x),float(point.value)));best_quality=float(point.value)
   if len(frontier)>1:ax.plot([v[0] for v in frontier],[v[1] for v in frontier],'--',color='#666666',lw=.8,zorder=1)
   ax.set_xscale('log');ax.set_ylim(min(.56,float(plot.value.min())-.07),max(.98,float(plot.value.max())+.08));ax.set_xlim((min(1.7,float(plot.x.min())/1.5),max(7000,float(plot.x.max())*1.5)) if axis=='MAC' else (min(.0035,float(plot.x.min())/1.5),max(.22,float(plot.x.max())*1.5)));ax.set_yticks([.6,.7,.8,.9]);panel(ax,'abcdef'[i*2+k],title)
   ax.set_xticks([10,100,1000] if axis=='MAC' else [.01,.1],['10','100','1000'] if axis=='MAC' else ['0.01','0.1']);ax.minorticks_off()
   ax.set_xlabel('含前端计算量（百万MACeq/音频秒）' if axis=='MAC' else '实时因子（RTF，5轮中位数/IQR）')
   if k==0:ax.set_ylabel('原始Pearson相关系数 ↑')
 save(fig,'fig4_pareto_zh_v33');pd.DataFrame(records).to_csv(OUT/'fig4_source_rows_zh_v33.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
 return dict(source=str(source),points=len(records),excluded_nonplot_rows=data[~(np.isfinite(data.x)&np.isfinite(data.value))][['axis','dataset','model','status','reason']].to_dict('records'),cost_or_quality_recomputed=False)


if __name__=='__main__':
 pareto()
 print('Fig4 rendered from saved point CSV; no inference or quality calculation')
