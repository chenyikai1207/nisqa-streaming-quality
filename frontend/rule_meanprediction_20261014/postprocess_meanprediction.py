from public_paths import package_path
from public_paths import package_path
from public_paths import package_path
"""Post-result human rule: mean saved seed predictions first, then paired bootstrap.
Preparation only until root sends GO and registers the received method/rule text.
No audio, model loading, forward, training, cost measurement or frozen-source writes.
"""
from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
F=HERE.parent
DIMS=['mos','noi','col','dis','loud']
ARMS=['A','B','C','D']
PAIRS=[('B','A'),('C','A'),('B','C'),('D','A')]
NBOOT=2000

def score(stat):
    n,y,p,y2,p2,yp,ae=np.moveaxis(stat,-1,0)
    mae=np.divide(ae,n,out=np.full_like(ae,np.nan),where=n>0)
    denom=np.sqrt(np.maximum(n*y2-y*y,0)*np.maximum(n*p2-p*p,0))
    r=np.divide(n*yp-y*p,denom,out=np.full_like(ae,np.nan),where=(n>1)&(denom>0))
    return np.stack([mae,r],axis=-1)

def classify(rows):
    # Positive improvement, exactly MAE(A)-MAE(B), only SIM->LIVE MOS/LOUD.
    def passing(dim):return rows[dim]['ci_low']>0
    def crossing(dim):return rows[dim]['ci_low']<=0<=rows[dim]['ci_high']
    if not all(np.isfinite(rows[dim][k]) for dim in ['mos','loud'] for k in ['improvement','ci_low','ci_high']):
        return dict(tier=None,status='RULE_NOT_ESTIMABLE',within_original_rule=False)
    worse=[dim for dim in ['mos','loud'] if rows[dim]['ci_high']<0]
    if worse:return dict(tier=None,status='WORSE_OVERRIDE',worse_dimensions=worse,within_original_rule=True)
    if passing('mos') and passing('loud'):return dict(tier=1,status='MOS_AND_LOUD_PASS',within_original_rule=True)
    if passing('loud') and crossing('mos'):return dict(tier=2,status='LOUD_PASS_MOS_CROSSES_ZERO',within_original_rule=True)
    if passing('mos') and crossing('loud'):
        return dict(tier=2,status='MOS_ONLY_SYMMETRIC_SUPPLEMENT',within_original_rule=False,
                    note='Explicit supplemental symmetric case; outside original three-tier wording.')
    if not passing('loud'):return dict(tier=3,status='LOUD_NOT_PASS',within_original_rule=True)
    return dict(tier=None,status='RULE_CASE_NOT_COVERED',within_original_rule=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration-receipt',type=Path,required=True,
        help='Root-provided actual Git rule/method registration receipt, after explicit GO.')
    args=parser.parse_args()
    registration=json.loads(args.registration_receipt.read_text(encoding='utf-8-sig'))
    if not registration:raise RuntimeError('No actual registration receipt.')
    summary=json.loads((F/'SUMMARY_COMPLETE.json').read_text())
    if not summary['complete_four'] or summary['actual_fit_count']!=120:
        raise RuntimeError('Original complete120fit inputs are required.')
    config=json.loads((F/'frontend_configs.json').read_text(encoding='utf-8-sig'))
    stage=package_path(config['stage'])
    labels=pd.read_csv(stage/'shared/data_index.csv')
    labels=labels[labels.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].set_index('file')
    rng=np.random.default_rng(20261014)
    metric_rows=[];contrast_rows=[];tier_inputs={}
    for protocol in ['CV','LOSO_SIM','SIM_TO_LIVE']:
        predictions={};keys=None
        for arm in ARMS:
            seed_values=[]
            for seed in [0,1,2]:
                filename='CV' if protocol=='CV' else 'LOSO'
                data=pd.read_csv(F/'predictions'/arm/f'seed{seed}_{filename}.csv')
                if protocol=='CV':data['split_id']='CV';data['stratum']=data.domain
                else:
                    select=data.split_id.str.startswith('LOSO_SIM') if protocol=='LOSO_SIM' else data.split_id.eq('SIM_TO_LIVE')
                    data=data[select].copy();data['stratum']=data.split_id
                data=data.sort_values(['stratum','file']).reset_index(drop=True)
                if data.duplicated(['file','split_id']).any():raise RuntimeError('Duplicate OOF identity.')
                these=data[['file','split_id','stratum','ref_group']]
                if keys is None:keys=these
                elif not these.equals(keys):raise RuntimeError('Saved predictions are not file/ref_group paired.')
                seed_values.append(data[DIMS].to_numpy(np.float64))
            predictions[arm]=np.mean(np.stack(seed_values,axis=0),axis=0)
            mean=keys.copy();mean['protocol']=protocol
            for j,dim in enumerate(DIMS):mean[dim]=predictions[arm][:,j]
            dest=HERE/'mean_predictions'/arm;dest.mkdir(parents=True,exist_ok=True)
            mean.to_csv(dest/f'{protocol}.csv',index=False)
        strata=sorted(keys.stratum.unique())
        if len(strata)!={'CV':2,'LOSO_SIM':4,'SIM_TO_LIVE':1}[protocol]:
            raise RuntimeError('Missing original evaluation strata.')
        strata_point=[];strata_boot=[]
        for stratum in strata:
            ids=np.flatnonzero(keys.stratum.eq(stratum))
            groups,codes=np.unique(keys.iloc[ids].ref_group.astype(str),return_inverse=True)
            draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=NBOOT)
            # Sufficient statistics from mean predictions; no mean of seed metrics.
            statistics=np.zeros((4,len(groups),5,7),np.float64)
            y=labels.loc[keys.iloc[ids].file,DIMS].to_numpy(np.float64)
            for ai,arm in enumerate(ARMS):
                pred=predictions[arm][ids]
                good=np.isfinite(y)&np.isfinite(pred)
                yy=np.where(good,y,0);pp=np.where(good,pred,0)
                values=np.stack([good,yy,pp,yy*yy,pp*pp,yy*pp,np.where(good,np.abs(yy-pp),0)],axis=-1)
                np.add.at(statistics[ai],codes,values)
            strata_point.append(score(statistics.sum(axis=1)))
            strata_boot.append(score(np.einsum('bg,agdv->badv',draws,statistics,optimize=True)))
        point=np.mean(np.stack(strata_point),axis=0)
        boot=np.mean(np.stack(strata_boot),axis=0)
        for ai,arm in enumerate(ARMS):
            for j,dim in enumerate(DIMS):
                for mi,metric in enumerate(['MAE','Pearson']):
                    metric_rows.append(dict(arm=arm,protocol=protocol,dim=dim,metric=metric,value=float(point[ai,j,mi]),
                        files=len(keys),ref_groups=keys.ref_group.nunique(),strata=len(strata),seeds_averaged=3,
                        aggregation='filewise mean of seed0/1/2 predictions, then within-stratum metric and equal strata',
                        statistical_rule_received_after_fit_results=True))
        for pi,(left,right) in enumerate(PAIRS):
            li=ARMS.index(left);ri=ARMS.index(right);delta=boot[:,li]-boot[:,ri]
            for j,dim in enumerate(DIMS):
                for mi,metric in enumerate(['MAE','Pearson']):
                    values=delta[:,j,mi];finite=np.isfinite(values)
                    lo,hi=np.quantile(values[finite],[.025,.975]) if finite.any() else [np.nan,np.nan]
                    row=dict(comparison=left+'-'+right,primary=pi==0,protocol=protocol,dim=dim,metric=metric,
                        difference=float(point[li,j,mi]-point[ri,j,mi]),ci_low=float(lo),ci_high=float(hi),
                        n_boot=NBOOT,n_boot_finite=int(finite.sum()),unit='ref_group',bootstrap_seed=20261014,
                        prediction_aggregation='mean seed predictions first',resampling='within domain/source; same group draw in both arms',
                        direction='negative improves' if metric=='MAE' else 'positive improves')
                    contrast_rows.append(row)
                    if left=='B' and right=='A' and protocol=='SIM_TO_LIVE' and dim in ['mos','loud'] and metric=='MAE':
                        tier_inputs[dim]=dict(dim=dim,comparison='A-B',metric='MAE',improvement=-row['difference'],
                            ci_low=-row['ci_high'],ci_high=-row['ci_low'],direction='positive improves',
                            source_comparison='B-A',n_boot=NBOOT,n_boot_finite=int(finite.sum()))
    metric=pd.DataFrame(metric_rows);paired=pd.DataFrame(contrast_rows)
    metric.to_csv(HERE/'TRAIN_frontend_metrics_meanprediction.csv',index=False)
    paired.to_csv(HERE/'TRAIN_frontend_paired_meanprediction.csv',index=False)
    pd.DataFrame(list(tier_inputs.values())).to_csv(HERE/'tier_improvements_A_minus_B.csv',index=False)
    gate=summary['B_reproduction']
    decision=dict(time=datetime.datetime.now().astimezone().isoformat(),evaluation_status='ACTUAL_POST_RESULT_HUMAN_TIER_EVALUATED',
        **classify(tier_inputs),judgment_protocol='SIM_TO_LIVE',judgment_dims=['mos','loud'],
        delta_definition='MAE(A)-MAE(B); positive improvement',inputs=tier_inputs,
        original_B_reproduction_gate=gate,original_gate_aggregation='mean seedwise CV equal-domain MOS MAE, unchanged',
        new_meanprediction_gate_not_refitted=True,original_gate_source=str(F/'SUMMARY_COMPLETE.json'),
        statistical_rule_received_after_fit_results=True,registration_receipt=str(args.registration_receipt),
        registration_record=registration,model_selection=False,frozen_changed=False)
    (HERE/'tier_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    # Read-only old/new numerical contrast table, no rerun of old bootstrap.
    old_metric=pd.read_csv(F/'TRAIN_frontend_metrics.csv')
    changed=metric.merge(old_metric[['arm','protocol','dim','metric','value']],on=['arm','protocol','dim','metric'],
        suffixes=('_meanprediction','_seedwise_metricmean'),validate='one_to_one')
    changed['aggregation_difference']=changed.value_meanprediction-changed.value_seedwise_metricmean
    changed.to_csv(HERE/'aggregation_change_metrics.csv',index=False)
    old_pair=pd.read_csv(F/'TRAIN_frontend_paired_bootstrap.csv')
    changed_pair=paired.merge(old_pair[['comparison','protocol','dim','metric','difference','ci_low','ci_high']],
        on=['comparison','protocol','dim','metric'],suffixes=('_meanprediction','_seedwise_metricmean'),validate='one_to_one')
    changed_pair.to_csv(HERE/'aggregation_change_paired.csv',index=False)
    complete=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_MEANPREDICTION_POSTPROCESS_COMPLETE',
        existing_fit_inputs=120,new_fits=0,metric_rows=len(metric),contrast_rows=len(paired),
        mean_prediction_CSVs=12,bootstrap_draws=NBOOT,bootstrap_seed=20261014,teacher_or_student_forward=0,audio_reads=0,
        old_results_overwritten=False,original_B_reproduction_gate_pass=gate['passed'],tier=decision['tier'],tier_status=decision['status'],
        statistical_rule_received_after_fit_results=True,registration_receipt=str(args.registration_receipt),frozen_changed=False)
    (HERE/'MEANPREDICTION_ACTUAL_COMPLETE.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
    print(json.dumps(complete),flush=True)

if __name__=='__main__':main()
