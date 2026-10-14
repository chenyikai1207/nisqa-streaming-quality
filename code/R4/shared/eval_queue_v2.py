"""GPU vector metrics with original cluster-weighted rank estimands, additive to v1."""
import numpy as np,pandas as pd,torch
from eval_protocol import _metric,fit_mapping,apply_mapping,DIMS

def bootstrap(y,p,groups,n_boot=2000,seed=20261011):
    y=np.asarray(y,float);p=np.asarray(p,float);_,codes=np.unique(groups,return_inverse=True);ng=codes.max()+1
    rng=np.random.default_rng(seed);draw=rng.integers(0,ng,size=(n_boot,ng));mult=np.zeros((n_boot,ng));np.add.at(mult,(np.repeat(np.arange(n_boot),ng),draw.ravel()),1)
    device='cuda' if torch.cuda.is_available() else 'cpu';w=torch.tensor(mult[:,codes],dtype=torch.float64,device=device);w=w/w.sum(1,keepdim=True);yt=torch.tensor(y,device=device);pt=torch.tensor(p,device=device)
    def corr(a,b):
        ac=a-(w*a).sum(1,keepdim=True);bc=b-(w*b).sum(1,keepdim=True);den=((w*ac.square()).sum(1)*(w*bc.square()).sum(1)).sqrt();return (w*ac*bc).sum(1)/den
    def ranks(a):
        order=np.argsort(a,kind='stable');group=np.r_[0,np.cumsum(np.diff(a[order])!=0)];inverse=np.empty(len(a),int);inverse[order]=group
        idx=torch.tensor(inverse,device=device);mass=torch.zeros((n_boot,group[-1]+1),dtype=torch.float64,device=device);mass.scatter_add_(1,idx[None].expand(n_boot,-1),w);rank=mass.cumsum(1)-mass/2;return rank[:,idx]
    e=pt-yt;result=torch.stack([(w*e.abs()).sum(1),(w*e.square()).sum(1).sqrt(),corr(yt[None],pt[None]),corr(ranks(y),ranks(p))],1).cpu().numpy();return result

def domain_arrays(meta,pred,dim,level,fit=None):
    z=meta[['file','db','domain','condition_id','ref_group',dim]].merge(pred[['file',dim]],on='file',suffixes=('_truth','_pred'),validate='one_to_one')
    if fit is not None:
        fit=fit.copy();raw=z[dim+'_pred'].to_numpy();finite=raw[np.isfinite(raw)]
        if fit['used']=='cubic' and len(finite):
            dc=np.polyder(fit['coef']);pts=[float(finite.min()),float(finite.max())]
            if dc[0]>0:
                vertex=-dc[1]/(2*dc[0])
                if finite.min()<vertex<finite.max():pts.append(vertex)
            if np.min(np.polyval(dc,pts))<0:fit.update(used='linear',coef=fit['linear_coef'],reason='evaluation interval nonmonotone; fixed TRAIN OOF linear fallback')
        z[dim+'_pred']=apply_mapping(raw,fit)
    key='condition_id' if meta.db.str.contains('TEST').any() else 'ref_group'
    if level=='condition':z=z.groupby(['domain','condition_id'],as_index=False).agg({dim+'_truth':'mean',dim+'_pred':'mean',**({key:'first'} if key!='condition_id' else {})})
    return z,key,fit

def evaluate_fast(pred,meta,model,dataset,level,mapping='none',fits=None):
    rows=[];allboots={};
    for dim in DIMS:
        fit=fit_mapping([],[],'none') if mapping=='none' else fits[dim]
        z,key,used=domain_arrays(meta,pred,dim,level,fit);parts=[]
        for di,(domain,g) in enumerate(z.groupby('domain',sort=True)):
            g=g[np.isfinite(g[dim+'_truth'])&np.isfinite(g[dim+'_pred'])]
            if len(g)<2:continue
            y=g[dim+'_truth'].to_numpy();p=g[dim+'_pred'].to_numpy();boot=bootstrap(y,p,g[key],seed=20261011+di);point=_metric(y,p);parts.append((domain,point,boot,len(g),g[key].nunique()));allboots[(dim,domain)]=boot
        if len(parts)>1:parts.append(('EQUAL_DOMAIN',np.mean([a[1] for a in parts],0),np.mean([a[2] for a in parts],0),sum(a[3] for a in parts),sum(a[4] for a in parts)))
        for domain,point,boot,n,clusters in parts:
            for j,metric in enumerate(['MAE','RMSE','Pearson','Spearman']):
                ci=np.nanpercentile(boot[:,j],[2.5,97.5]);rows.append(dict(model=model,dataset=dataset,domain=domain,level=level,dim=dim,metric=metric,mapping=mapping,mapping_used=used['used'],value=point[j],ci_low=ci[0],ci_high=ci[1],n=n,n_clusters=clusters,n_boot=2000,mapping_fit_n=used['n'],mapping_note=used.get('reason',''),metric_device='CUDA float64' if torch.cuda.is_available() else 'CPU float64'))
    return pd.DataFrame(rows)

def secondary(pred,meta,model,dataset,level):
    rows=[]
    for dim in DIMS:
        z,key,_=domain_arrays(meta,pred,dim,level);parts=[]
        for di,(domain,g) in enumerate(z.groupby('domain',sort=True)):
            g=g[np.isfinite(g[dim+'_truth'])&np.isfinite(g[dim+'_pred'])]
            if len(g)<2:continue
            fit=fit_mapping(g[dim+'_pred'],g[dim+'_truth'],'linear');mapped=apply_mapping(g[dim+'_pred'],fit);boot=bootstrap(g[dim+'_truth'],mapped,g[key],seed=20261011+di);parts.append((domain,_metric(g[dim+'_truth'].to_numpy(),mapped),boot,len(g),g[key].nunique(),fit['n']))
        if len(parts)>1:parts.append(('EQUAL_DOMAIN',np.mean([a[1] for a in parts],0),np.mean([a[2] for a in parts],0),sum(a[3] for a in parts),sum(a[4] for a in parts),sum(a[5] for a in parts)))
        for domain,point,boot,n,clusters,fitn in parts:
            for j,metric in enumerate(['MAE','RMSE','Pearson','Spearman']):
                ci=np.nanpercentile(boot[:,j],[2.5,97.5]);rows.append(dict(model=model,dataset=dataset,domain=domain,level=level,dim=dim,metric=metric,mapping='SECONDARY_TARGET_DOMAIN_LINEAR',mapping_used='separate domain linear',value=point[j],ci_low=ci[0],ci_high=ci[1],n=n,n_clusters=clusters,n_boot=2000,mapping_fit_n=fitn,mapping_note='retrospective target/domain fit, fixed fit conditional bootstrap, not selection/deployment'))
    return pd.DataFrame(rows)
