"""Chinese figure 3 from saved published decomposition rows; no inference or new statistics."""
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

def mse():
 source=OUT/'fig3_source_rows_zh_v34.csv'
 saved=pd.read_csv(source,float_precision='round_trip')
 models=['A_distill','v2_pure','NISQA5D','PLCMOS','DNSMOS','SQUIM','P563']
 labels=['首版','本文\n模型','NISQA\n五维','PLC\nMOS','DNS\nMOS','SQUIM','P.563']
 cols=['mean_bias_squared','scale_mismatch_squared','linear_correlation_residual'];gray=['white','#BBBBBB','#777777'];hatches=['///','..',r'\\']
 records=[];missing=[]
 fig,axes=plt.subplots(2,2,figsize=(174/25.4,106/25.4),sharey=True)
 fig.subplots_adjust(left=.105,right=.985,bottom=.16,top=.93,wspace=.23,hspace=.55)
 upper=0
 for ax,ds,title,letter in zip(axes.flat,SETS,SETLABEL,'abcd'):
  for i,model in enumerate(models):
   d=saved
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
 save(fig,'fig3_mse_seven_methods_zh_v34');pd.DataFrame(records).to_csv(OUT/'fig3_source_rows_zh_v34.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
 return dict(source='figures/fig3_source_rows_zh_v34.csv',rows=len(records),missing=missing,statistics_recomputed=False)


if __name__=='__main__':
 mse()
 print('Fig3 rendered from saved decomposition rows; no new statistics or inference')
