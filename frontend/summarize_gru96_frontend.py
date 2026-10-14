"""TRAIN OOF metrics and paired ref_group bootstrap; no audio or inference."""
from pathlib import Path
import json,datetime
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DIMS=['mos','noi','col','dis','loud']
NBOOT=2000

def score(t):
    # Last axis: n,y,p,y2,p2,yp,absolute error.
    n,y,p,yy,pp,yp,ae=np.moveaxis(t,-1,0)
    mae=np.divide(ae,n,out=np.full_like(ae,np.nan),where=n>0)
    denom=np.sqrt(np.maximum(n*yy-y*y,0)*np.maximum(n*pp-p*p,0))
    pearson=np.divide(n*yp-y*p,denom,out=np.full_like(ae,np.nan),where=(n>1)&(denom>0))
    return np.stack([mae,pearson],axis=-1)

def main():
    config=json.loads((HERE/'frontend_configs.json').read_text(encoding='utf-8-sig'))
    stage=Path(config['stage'])
    d=pd.read_csv(stage/'shared/data_index.csv')
    d=d[d.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].sort_values('file').set_index('file')
    arms=[a for a in ['A','B','C','D'] if (HERE/f'{a}_FIT_COMPLETE.json').exists()]
    if not arms:raise RuntimeError('No actual complete fresh configuration.')
    points=[];contrasts=[]
    rng=np.random.default_rng(config['bootstrap_seed'])
    for protocol in ['CV','LOSO_SIM','SIM_TO_LIVE']:
        data={}
        for arm in arms:
            for seed in config['seeds']:
                p=pd.read_csv(HERE/'predictions'/arm/f'seed{seed}_{"CV" if protocol=="CV" else "LOSO"}.csv')
                if protocol=='CV':p['split_id']='CV';p['stratum']=p.domain
                else:
                    p=p[p.split_id.str.startswith('LOSO_SIM') if protocol=='LOSO_SIM' else p.split_id.eq('SIM_TO_LIVE')].copy()
                    p['stratum']=p.split_id
                if p.duplicated(['file','split_id']).any():raise RuntimeError('Duplicated OOF file/split.')
                p=p.sort_values(['stratum','file']).reset_index(drop=True)
                data[arm,seed]=p
        ref=data[arms[0],0]
        keys=ref[['file','split_id','stratum','ref_group']]
        for p in data.values():
            if not p[['file','split_id','stratum','ref_group']].equals(keys):raise RuntimeError('Unpaired OOF identities.')
        strata=sorted(ref.stratum.unique())
        expected=2 if protocol=='CV' else 4 if protocol=='LOSO_SIM' else 1
        if len(strata)!=expected:raise RuntimeError('Incomplete protocol strata.')
        actual=[];boots=[]
        for stratum in strata:
            ix=np.flatnonzero(ref.stratum.eq(stratum))
            groups,codes=np.unique(ref.iloc[ix].ref_group.astype(str).to_numpy(),return_inverse=True)
            draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=NBOOT)
            stat=np.zeros((len(arms),3,len(groups),5,7),np.float64)
            y=d.loc[ref.iloc[ix].file,DIMS].to_numpy(np.float64)
            for ai,arm in enumerate(arms):
                for si,seed in enumerate(config['seeds']):
                    pred=data[arm,seed].iloc[ix][DIMS].to_numpy(np.float64)
                    good=np.isfinite(y)&np.isfinite(pred)
                    yy=np.where(good,y,0);pp=np.where(good,pred,0)
                    values=np.stack([good,yy,pp,yy*yy,pp*pp,yy*pp,np.where(good,np.abs(yy-pp),0)],axis=-1)
                    np.add.at(stat[ai,si],codes,values)
            actual.append(score(stat.sum(axis=2)))
            # Batched group sufficient-statistic bootstrap avoids per-file repeated scans.
            bt=np.einsum('bg,asgdv->basdv',draws,stat,optimize=True)
            boots.append(score(bt))
        # Macro over domains/sources, then seeds; no pooled-domain Pearson shortcut.
        seedpoint=np.mean(np.stack(actual),axis=0)
        point=seedpoint.mean(axis=1)
        boot=np.mean(np.stack(boots),axis=0).mean(axis=2)
        for ai,arm in enumerate(arms):
            for j,dim in enumerate(DIMS):
                for mi,metric in enumerate(['MAE','Pearson']):
                    points.append(dict(arm=arm,protocol=protocol,dim=dim,metric=metric,value=float(point[ai,j,mi]),
                        seed0=float(seedpoint[ai,0,j,mi]),seed1=float(seedpoint[ai,1,j,mi]),seed2=float(seedpoint[ai,2,j,mi]),
                        files=len(ref),ref_groups=ref.ref_group.nunique(),strata=len(strata),seeds=3,
                        weighting='within-stratum metric; equal domains/sources; equal seeds',source='fresh postfreeze TRAIN OOF'))
        pairs=[config['primary_comparison']]+config['secondary_comparisons']
        for k,(left,right) in enumerate(pairs):
            if left not in arms or right not in arms:continue
            li=arms.index(left);ri=arms.index(right)
            delta=boot[:,li]-boot[:,ri]
            for j,dim in enumerate(DIMS):
                for mi,metric in enumerate(['MAE','Pearson']):
                    z=delta[:,j,mi];valid=np.isfinite(z)
                    lo,hi=np.quantile(z[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                    contrasts.append(dict(comparison=left+'-'+right,primary=k==0,protocol=protocol,dim=dim,metric=metric,
                        difference=float(point[li,j,mi]-point[ri,j,mi]),ci_low=float(lo),ci_high=float(hi),
                        n_boot=NBOOT,n_boot_finite=int(valid.sum()),unit='ref_group',seeds=3,
                        pairing='same group multiplicity for both arms and all seeds',resampling='within domain/source, fixed macro weights',
                        improvement_direction='negative' if metric=='MAE' else 'positive'))
    frame=pd.DataFrame(points);frame.to_csv(HERE/'TRAIN_frontend_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(HERE/'TRAIN_frontend_paired_bootstrap.csv',index=False)
    b=frame[(frame.arm=='B')&(frame.protocol=='CV')&(frame.dim=='mos')&(frame.metric=='MAE')]
    bv=float(b.value.iloc[0]) if len(b)==1 else None
    check=dict(reference=.4176,tolerance=.002,actual=bv,
        absolute_difference=abs(bv-.4176) if bv is not None else None,
        passed=abs(bv-.4176)<=.002 if bv is not None else None,
        scope='pipeline reproduction check; no hyperparameter/seed retuning or model selection')
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_AVAILABLE_CONFIGURATIONS_SUMMARIZED',
        actual_arms=arms,complete_four=len(arms)==4,actual_fit_count=len(arms)*30,B_reproduction=check,
        metric_rows=len(points),contrast_rows=len(contrasts),ref_group_bootstrap_draws=NBOOT,
        VAL_TEST_reads=0,frozen_changed=False,quality_recomputed_only='new TRAIN-only OOF configurations')
    (HERE/'SUMMARY_COMPLETE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
