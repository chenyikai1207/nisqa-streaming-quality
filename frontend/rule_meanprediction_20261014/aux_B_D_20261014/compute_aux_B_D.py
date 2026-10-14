"""One post-result auxiliary B-D MOS MAE contrast, saved predictions only."""
from pathlib import Path
import datetime,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
RULE=HERE.parent
F=RULE.parent

def main():
    request=json.loads((HERE/'REQUEST_METHOD_BEFORE_AUX_CALCULATION.json').read_text(encoding='utf-8-sig'))
    stage=Path(json.loads((F/'frontend_configs.json').read_text())['stage'])
    labels=pd.read_csv(stage/'shared/data_index.csv')
    labels=labels[labels.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].set_index('file')
    tables=[pd.read_csv(RULE/'mean_predictions'/arm/'SIM_TO_LIVE.csv') for arm in ['B','D']]
    keys=tables[0][['file','ref_group']]
    if keys.duplicated('file').any() or not tables[1][['file','ref_group']].equals(keys):
        raise RuntimeError('Mean prediction identities not paired.')
    y=labels.loc[keys.file,'mos'].to_numpy(np.float64)
    predictions=np.stack([t.mos.to_numpy(np.float64) for t in tables])
    groups,codes=np.unique(keys.ref_group.astype(str).to_numpy(),return_inverse=True)
    rng=np.random.default_rng(20261014)
    draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    stats=np.zeros((2,len(groups),2),np.float64)
    for i,pred in enumerate(predictions):
        good=np.isfinite(y)&np.isfinite(pred)
        np.add.at(stats[i],codes,np.stack([good,np.where(good,np.abs(pred-y),0)],axis=-1))
    total=stats.sum(axis=1);point=total[:,1]/total[:,0]
    boot=np.einsum('bg,agv->bav',draws,stats,optimize=True)
    sample=boot[:,:,1]/boot[:,:,0];difference=sample[:,0]-sample[:,1]
    finite=np.isfinite(difference);lo,hi=np.quantile(difference[finite],[.025,.975])
    row=dict(comparison='B-D',protocol='SIM_TO_LIVE',dim='mos',metric='MAE',aggregation='mean_seed_predictions_first',
        MAE_B=float(point[0]),MAE_D=float(point[1]),difference=float(point[0]-point[1]),ci_low=float(lo),ci_high=float(hi),
        n_boot=2000,n_boot_finite=int(finite.sum()),bootstrap_seed=20261014,unit='ref_group',files=len(keys),ref_groups=len(groups),
        direction='positive means B worse than D',post_result_auxiliary=True,tier_input=False,model_selection=False)
    pd.DataFrame([row]).to_csv(HERE/'B_minus_D_SIMLIVE_MOS_MAE_meanprediction.csv',index=False)
    # Historical only: same reference-group replicates, metrics computed per seed then averaged.
    hist=np.zeros((2,3,len(groups),2),np.float64)
    for ai,arm in enumerate(['B','D']):
        for seed in [0,1,2]:
            t=pd.read_csv(F/'predictions'/arm/f'seed{seed}_LOSO.csv')
            t=t[t.split_id.eq('SIM_TO_LIVE')].set_index('file').loc[keys.file]
            if t.ref_group.tolist()!=keys.ref_group.tolist():raise RuntimeError('Historical seed identities not paired.')
            pred=t.mos.to_numpy(np.float64);good=np.isfinite(y)&np.isfinite(pred)
            np.add.at(hist[ai,seed],codes,np.stack([good,np.where(good,np.abs(pred-y),0)],axis=-1))
    total=hist.sum(axis=2);hist_point=(total[:,:,1]/total[:,:,0]).mean(axis=1)
    hist_boot=np.einsum('bg,asgv->basv',draws,hist,optimize=True)
    hist_sample=(hist_boot[:,:,:,1]/hist_boot[:,:,:,0]).mean(axis=2)
    hist_delta=hist_sample[:,0]-hist_sample[:,1]
    good=np.isfinite(hist_delta);hlo,hhi=np.quantile(hist_delta[good],[.025,.975])
    hrow=dict(row,aggregation='HISTORICAL_mean_seedwise_MAE_metrics',MAE_B=float(hist_point[0]),MAE_D=float(hist_point[1]),
        difference=float(hist_point[0]-hist_point[1]),ci_low=float(hlo),ci_high=float(hhi),n_boot_finite=int(good.sum()),
        public_main_table_use=False,only_purpose='explain historical v30 .0061; not mixed with new meanprediction table')
    pd.DataFrame([hrow]).to_csv(HERE/'HISTORICAL_B_minus_D_SIMLIVE_MOS_MAE_seedmetricmean.csv',index=False)
    complete=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_POSTRESULT_AUX_B_D_COMPLETE',
        request_record=request,main_new_aggregation=row,historical_separate=hrow,new_fit_forward_audio_cost=0,
        tier_unchanged=True,original_statistics_changed=False,frozen_changed=False,extra_model_selection=False)
    (HERE/'B_D_AUX_ACTUAL_COMPLETE.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(main=row,historical=hrow),ensure_ascii=False))

if __name__=='__main__':main()
