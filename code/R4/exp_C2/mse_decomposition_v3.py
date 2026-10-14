"""Raw MOS MSE algebra and cluster CI from existing v1 predictions only."""
from pathlib import Path
import sys,json,datetime
import numpy as np,pandas as pd,torch
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(R/'shared'))
from eval_protocol import _merge
SEED=20261011;BOOT=2000;DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
TERMS=['MSE','mean_bias_squared','scale_mismatch_squared','linear_correlation_residual']
def parts(y,p,w):
 w=w/w.sum(axis=1,keepdims=True);ym=w@y;pm=w@p;yc=y[None,:]-ym[:,None];pc=p[None,:]-pm[:,None];vy=(w*yc**2).sum(axis=1);vp=(w*pc**2).sum(axis=1);cov=(w*yc*pc).sum(axis=1);sy=np.sqrt(vy);sp=np.sqrt(vp)
 # r=0 for constant prediction is the continuous decomposition convention;
 # sigma_p=0 makes the covariance term vanish, ordering error equals truth variance.
 r=np.divide(cov,sy*sp,out=np.zeros(len(w)),where=sy*sp>0)
 return np.stack([(w*(p-y)**2).sum(axis=1),(pm-ym)**2,(sp-r*sy)**2,(1-r*r)*vy],axis=1)
def calculate(y,p,groups,seed=SEED):
 y=np.asarray(y,float);p=np.asarray(p,float);codes,unique=pd.factorize(groups);ng=len(unique);point=parts(y,p,np.ones((1,len(y))))[0]
 rng=np.random.default_rng(seed);draw=rng.integers(0,ng,size=(BOOT,ng));mult=np.zeros((BOOT,ng));np.add.at(mult,(np.repeat(np.arange(BOOT),ng),draw.ravel()),1);weights=mult[:,codes]
 if DEVICE=='cuda':
  w=torch.as_tensor(weights,dtype=torch.float64,device=DEVICE);yt=torch.as_tensor(y,dtype=torch.float64,device=DEVICE);pt=torch.as_tensor(p,dtype=torch.float64,device=DEVICE);w=w/w.sum(1,keepdim=True);ym=w@yt;pm=w@pt;yc=yt[None,:]-ym[:,None];pc=pt[None,:]-pm[:,None];vy=(w*yc.square()).sum(1);vp=(w*pc.square()).sum(1);cov=(w*yc*pc).sum(1);sy=vy.sqrt();sp=vp.sqrt();r=torch.where(sy*sp>0,cov/(sy*sp),0);boot=torch.stack([(w*(pt-yt).square()).sum(1),(pm-ym).square(),(sp-r*sy).square(),(1-r.square())*vy],dim=1).cpu().numpy()
 else:boot=parts(y,p,weights)
 assert np.max(abs(boot[:,0]-boot[:,1:].sum(axis=1)))<1e-10
 assert abs(point[0]-point[1:].sum())<1e-12
 return point,boot,ng

def main():
 models=['A_G0','A_newT','A_tiny','A_distill','NISQA5D','DNSMOS','PLCMOS','SQUIM','P563'];datasets=['NISQA_TEST_FOR','NISQA_TEST_P501','NISQA_TEST_LIVETALK'];rows=[]
 for model in models:
  for ds in datasets:
   path=R/f'exp_B/preds/{model}/{ds}.csv';meta=pd.read_csv(R/f'exp_B/{ds}_labels_postG1.csv');z=_merge(path,meta)
   for level in ['file','condition']:
    a=z[['domain','condition_id','mos_truth','mos_pred']].dropna()
    if level=='condition':a=a.groupby(['domain','condition_id'],as_index=False).mean(numeric_only=True)
    results=[]
    for di,(domain,g) in enumerate(a.groupby('domain',sort=True)):
     point,boot,ng=calculate(g.mos_truth,g.mos_pred,g.condition_id,SEED+di);results.append((point,boot));rec=dict(model=model,dataset=ds,domain=domain,level=level,dim='mos',mapping='none',n=len(g),n_clusters=ng,n_boot=BOOT,bootstrap_device=DEVICE,result_version='v1_existing_predictions_v3_decomposition',source_prediction_path=str(path.relative_to(R)),new_inference=0,protocol='C3_V3_RAW_MSE_POPULATION_VARIANCE')
     for j,name in enumerate(TERMS):rec[name]=float(point[j]);rec[name+'_ci_low'],rec[name+'_ci_high']=np.quantile(boot[:,j],[.025,.975])
     rec['algebra_residual']=float(point[0]-point[1:].sum());rows.append(rec)
    point=np.mean([x[0] for x in results],axis=0);boot=np.mean([x[1] for x in results],axis=0);rec=dict(rows[-1]);rec.update(domain='EQUAL_DOMAIN',n=len(a),n_clusters=a.condition_id.nunique(),aggregation='mean of SIM/LIVE component values and independently seeded domain bootstrap; not pooled moments')
    for j,name in enumerate(TERMS):rec[name]=float(point[j]);rec[name+'_ci_low'],rec[name+'_ci_high']=np.quantile(boot[:,j],[.025,.975])
    rec['algebra_residual']=float(point[0]-point[1:].sum());rows.append(rec)
 out=pd.DataFrame(rows);out.to_csv(P/'results/C3_MSE_decomposition_v1_v3.csv',index=False)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
 selected=['A_distill','PLCMOS','NISQA5D'];names=['v1 GRU','PLCMOS','NISQA'];components=TERMS[1:]
 for ax,ds in zip(axes,datasets):
  g=out[out.dataset.eq(ds)&out.level.eq('file')&out.domain.eq('EQUAL_DOMAIN')].set_index('model').loc[selected];bottom=np.zeros(3)
  for col,label,color,hatch in zip(components,['Mean offset','Scale mismatch','Linear residual'],['#eeeeee','#999999','#333333'],['///','..','']):
   values=g[col].to_numpy();ax.bar(np.arange(3),values,bottom=bottom,label=label,color=color,hatch=hatch,edgecolor='black',linewidth=.6);bottom+=values
  ax.set_xticks(np.arange(3),names);ax.set_title(ds.replace('NISQA_TEST_',''));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_ylabel('Raw MSE (equal SIM/LIVE)');axes[-1].legend(fontsize=8);fig.tight_layout()
 for suffix in ['svg','pdf','png']:fig.savefig(P/'results'/('C3_MSE_decomposition_v1_v3.'+suffix),dpi=180)
 rec=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_EXISTING_V1_DECOMPOSITION_COMPLETE',rows=len(out),methods=len(models),datasets=3,levels=['file','condition'],n_boot=BOOT,device=DEVICE,variance_ddof=0,max_algebra_residual=float(out.algebra_residual.abs().max()),new_inference=0,new_v2_results=0,no_TEST_audio=True,new_v3_freeze_required_for_v2=True)
 (P/'C3_MSE_EXISTING_COMPLETE_v3.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
