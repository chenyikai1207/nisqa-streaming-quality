"""R4 shared evaluation. mapping_fit is explicit paired prediction/truth data.
fit_mapping(prediction, truth, mapping); evaluate(...,mapping_fit=TRAIN/OOF wide df)
Official NISQA mapping='none' only unless an independent nonTEST mapping set exists.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
DIMS=['mos','noi','col','dis','loud']; R=Path(__file__).resolve().parent.parent

def fit_mapping(prediction,truth,mapping='cubic'):
    x=np.asarray(prediction,float);y=np.asarray(truth,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if mapping=='none':return {'requested':'none','used':'none','coef':[1.,0.],'n':len(x)}
    if len(x)<2 or np.ptp(x)==0:return {'requested':mapping,'used':'unavailable','coef':[],'n':len(x),'reason':'insufficient mapping fit variation'}
    degree=3 if mapping=='cubic' and len(x)>=4 and len(np.unique(x))>=4 else 1
    c=np.polyfit(x,y,degree);reason=''
    if degree==3:
        dc=np.polyder(c); pts=[float(x.min()),float(x.max())]
        # A quadratic derivative has its minimum at the vertex when convex.
        if dc[0]>0:
            z=-dc[1]/(2*dc[0])
            if x.min()<z<x.max():pts.append(z)
        if np.min(np.polyval(dc,pts))<0:degree=1;c=np.polyfit(x,y,1);reason='cubic nonmonotone on TRAIN/OOF prediction interval; linear fallback'
    return {'requested':mapping,'used':'cubic' if degree==3 else 'linear','coef':c.tolist(),'n':len(x),'prediction_range':[float(x.min()),float(x.max())],'reason':reason,'linear_coef':np.polyfit(x,y,1).tolist()}

def apply_mapping(prediction,fit):
    x=np.asarray(prediction,float)
    return x if fit['used']=='none' else np.full(x.shape,np.nan) if fit['used']=='unavailable' else np.polyval(fit['coef'],x)

def _read(x):return x.copy() if isinstance(x,pd.DataFrame) else pd.read_csv(x)

def _metadata(dataset):
    idx=pd.read_csv(R/'shared/data_index.csv')
    if isinstance(dataset,pd.DataFrame):return dataset.copy()
    if isinstance(dataset,(list,tuple)):return idx[idx.db.isin(dataset)].copy()
    return idx[idx.db.str.contains(dataset,regex=False)].copy()

def _merge(pred,dataset):
    p=_read(pred);meta=_metadata(dataset); assert p.file.is_unique, 'prediction file duplicates'
    cols=['file']+[d for d in DIMS if d in p]
    a=meta.merge(p[cols],on='file',how='inner',suffixes=('_truth','_pred'),validate='one_to_one')
    for d in DIMS:
        if d not in p:a[d+'_pred']=np.nan;a[d+'_truth']=a[d] if d in a else np.nan
    return a

def _fit_from(mapping_fit,dim,mapping):
    if mapping=='none':return fit_mapping([],[],'none')
    if mapping_fit is None:raise ValueError('linear/cubic requires explicit nonTEST mapping_fit; no target fitting')
    if isinstance(mapping_fit,dict):return mapping_fit[dim]
    f=_read(mapping_fit)
    if 'db' in f and f.db.astype(str).str.contains('TEST').any():raise ValueError('TEST prohibited in mapping fit')
    xp=dim+'_pred' if dim+'_pred' in f else dim;yt=dim+'_truth' if dim+'_truth' in f else 'truth_'+dim
    return fit_mapping(f[xp],f[yt],mapping)

def _metric(y,p,w=None):
    if w is None:w=np.ones(len(y))
    valid=np.isfinite(y)&np.isfinite(p)&(w>0);y=y[valid];p=p[valid];w=w[valid]
    if not len(y):return np.full(4,np.nan)
    w=w/w.sum();e=p-y;mae=np.sum(w*abs(e));rmse=np.sqrt(np.sum(w*e*e))
    def corr(a,b):
        ac=a-np.sum(w*a);bc=b-np.sum(w*b);den=np.sqrt(np.sum(w*ac*ac)*np.sum(w*bc*bc));return np.sum(w*ac*bc)/den if den>0 else np.nan
    # weighted midranks reproduce the ranks of the duplicated cluster sample exactly.
    def wrank(a):
        order=np.argsort(a,kind='stable'); z=a[order]; ww=w[order]; starts=np.r_[0,np.flatnonzero(np.diff(z)!=0)+1]; counts=np.add.reduceat(ww,starts);r=np.cumsum(counts)-counts/2;out=np.empty(len(a));out[order]=np.repeat(r,np.diff(np.r_[starts,len(a)]));return out
    return np.array([mae,rmse,corr(y,p),corr(wrank(y),wrank(p))])

def _arrays(pred,dataset,level,mapping,mapping_fit,n_boot,seed):
    merged=_merge(pred,dataset); out={}; rng=np.random.default_rng(seed)
    is_test=merged.db.astype(str).str.contains('TEST').any();key='condition_id' if is_test else 'ref_group'
    for dim in DIMS:
        fit=_fit_from(mapping_fit,dim,mapping);z=merged[['domain','condition_id',key] + [dim+'_truth',dim+'_pred']].copy() if key!='condition_id' else merged[['domain','condition_id',dim+'_truth',dim+'_pred']].copy()
        # Check cubic over the actual evaluation prediction interval; coefficients remain TRAIN/OOF fitted.
        fit=fit.copy()
        raw=z[dim+'_pred'].to_numpy(); finite=raw[np.isfinite(raw)]
        if fit['used']=='cubic' and len(finite):
            dc=np.polyder(fit['coef']);pts=[float(finite.min()),float(finite.max())]
            if dc[0]>0:
                v=-dc[1]/(2*dc[0])
                if finite.min()<v<finite.max():pts.append(v)
            if np.min(np.polyval(dc,pts))<0:
                fit['used']='linear';fit['coef']=fit['linear_coef'];fit['reason']='cubic nonmonotone on evaluation prediction interval; TRAIN/OOF linear fallback'
        z[dim+'_pred']=apply_mapping(z[dim+'_pred'],fit)
        if level=='condition':
            # VAL conditions are one file each; TEST conditions aggregate files first.
            z=z.groupby(['domain','condition_id'],as_index=False).agg({dim+'_truth':'mean',dim+'_pred':'mean',**({key:'first'} if key!='condition_id' else {})})
        elif level!='file':raise ValueError(level)
        domains=[]
        for domain, q in z.groupby('domain',sort=True):
            q=q[np.isfinite(q[dim+'_truth'])&np.isfinite(q[dim+'_pred'])]
            if not len(q):continue
            y=q[dim+'_truth'].to_numpy();p=q[dim+'_pred'].to_numpy();codes,groups=pd.factorize(q[key]);ng=len(groups)
            point=_metric(y,p);boot=np.empty((n_boot,4))
            for b in range(n_boot):
                mult=np.bincount(rng.integers(0,ng,ng),minlength=ng);boot[b]=_metric(y,p,mult[codes])
            out[(domain,dim)]=(point,boot,len(q),fit);domains.append(domain)
        if len(domains)>1:
            out[('EQUAL_DOMAIN',dim)]=(np.mean([out[(d,dim)][0] for d in domains],axis=0),np.mean([out[(d,dim)][1] for d in domains],axis=0),sum(out[(d,dim)][2] for d in domains),fit)
    return out

def evaluate(pred_csv,dataset,level='file',mapping='none',n_boot=2000,seed=20261011,mapping_fit=None,model=None):
    a=_arrays(pred_csv,dataset,level,mapping,mapping_fit,n_boot,seed);rows=[]
    for (domain,dim),(point,boot,n,fit) in a.items():
        for j,metric in enumerate(['MAE','RMSE','Pearson','Spearman']):
            ci=np.nanpercentile(boot[:,j],[2.5,97.5]) if np.isfinite(boot[:,j]).any() else [np.nan,np.nan]
            rows.append(dict(model=model or (Path(pred_csv).stem if not isinstance(pred_csv,pd.DataFrame) else 'model'),dataset=str(dataset) if not isinstance(dataset,pd.DataFrame) else 'provided',domain=domain,level=level,dim=dim,metric=metric,mapping=mapping,mapping_used=fit['used'],value=point[j],ci_low=ci[0],ci_high=ci[1],n=n,n_boot=n_boot,mapping_fit_n=fit['n'],mapping_note=fit.get('reason','')))
    return pd.DataFrame(rows)

def paired_diff(pred_a,pred_b,dataset,level='file',metric='RMSE',mapping='none',fit_a=None,fit_b=None,n_boot=2000,seed=20261011):
    pa=_read(pred_a);pb=_read(pred_b);common=set(pa.file)&set(pb.file);meta=_metadata(dataset);meta=meta[meta.file.isin(common)]
    # identical seed/domain/group ordering delivers identical cluster draws for both methods.
    # Synchronize finite support in every dimension so cluster draws remain exactly paired.
    for dim in DIMS:
        if dim in pa and dim in pb:
            aa=pa.set_index('file')[dim];bb=pb.set_index('file')[dim]
            finitefiles=set(aa.index[np.isfinite(aa)])&set(bb.index[np.isfinite(bb)])
            pa.loc[~pa.file.isin(finitefiles),dim]=np.nan;pb.loc[~pb.file.isin(finitefiles),dim]=np.nan
        else:
            pa[dim]=np.nan;pb[dim]=np.nan
    a=_arrays(pa,meta,level,mapping,fit_a,n_boot,seed);b=_arrays(pb,meta,level,mapping,fit_b,n_boot,seed);j=['MAE','RMSE','Pearson','Spearman'].index(metric);rows=[]
    for key in a.keys()&b.keys():
        ap,ab,n,_=a[key];bp,bb,_,_=b[key];delta=ab[:,j]-bb[:,j];ci=np.nanpercentile(delta,[2.5,97.5]);rows.append(dict(domain=key[0],dim=key[1],level=level,metric=metric,mapping=mapping,difference_a_minus_b=ap[j]-bp[j],ci_low=ci[0],ci_high=ci[1],n=n,n_boot=n_boot))
    return pd.DataFrame(rows)
