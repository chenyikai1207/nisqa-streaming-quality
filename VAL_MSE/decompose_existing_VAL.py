from public_paths import data_path, package_path
"""One requested postfreeze raw-MOS decomposition; no inference or source writes."""
from pathlib import Path
import sys, json, datetime
P = Path(__file__).resolve().parent
R = Path(__file__).resolve().parents[1]/'code/R4'
if (P/'VAL_MSE_ACTUAL_COMPLETE.json').exists():
    raise SystemExit('Existing complete receipt retained; no calculation repeated.')
assert (P/'REQUEST_METHOD_BEFORE_CALCULATION.md').exists()
sys.path[:0] = [str(R/'exp_C2'), str(R/'shared')]
import numpy as np
import pandas as pd
import torch
from mse_decomposition_v3 import calculate, TERMS, DEVICE
torch.set_num_threads(2)
pred_path = R/'exp_C2/preds/v3/v2_pure/VAL.csv'
meta_path = R/'shared/data_index.csv'
pred = pd.read_csv(pred_path)
meta = pd.read_csv(meta_path)
meta = meta[meta.db.isin(['NISQA_VAL_SIM','NISQA_VAL_LIVE'])].copy()
assert len(meta)==2700 and meta.file.is_unique and pred.file.is_unique
assert set(meta.file)==set(pred.file)
z=meta[['file','domain','ref_group','condition_id','mos']].merge(pred[['file','mos']],on='file',suffixes=('_truth','_pred'),validate='one_to_one')
assert np.isfinite(z[['mos_truth','mos_pred']]).all().all() and z.ref_group.notna().all()
rows=[];results=[]
for i,(domain,g) in enumerate(z.groupby('domain',sort=True)):
    point,boot,ng=calculate(g.mos_truth,g.mos_pred,g.ref_group,20261011+i)
    results.append((point,boot))
    rec=dict(model='v2_pure',dataset='VAL',domain=domain,level='file',dim='mos',mapping='none',n=len(g),n_clusters=ng,n_boot=2000,bootstrap_unit='ref_group',bootstrap_seed=20261011+i,bootstrap_device=DEVICE,variance_ddof=0,source_prediction_path=str(pred_path),source_metadata_path=str(meta_path),new_inference=0,analysis_identity='postfreeze_requested_raw_VAL_MSE_20261014')
    for j,term in enumerate(TERMS):
        rec[term]=float(point[j]);rec[term+'_ci_low'],rec[term+'_ci_high']=map(float,np.quantile(boot[:,j],[.025,.975]))
    rec['algebra_residual']=float(point[0]-point[1:].sum());rows.append(rec)
point=np.mean([x[0] for x in results],axis=0);boot=np.mean([x[1] for x in results],axis=0)
rec=rows[-1].copy();rec.update(domain='EQUAL_DOMAIN',n=len(z),n_clusters=z.ref_group.nunique(),bootstrap_seed='20261011 LIVE; 20261012 SIM',aggregation='arithmetic mean of domain components and independently seeded domain draws; not pooled moments')
for j,term in enumerate(TERMS):
    rec[term]=float(point[j]);rec[term+'_ci_low'],rec[term+'_ci_high']=map(float,np.quantile(boot[:,j],[.025,.975]))
rec['algebra_residual']=float(point[0]-point[1:].sum());rows.append(rec)
out=pd.DataFrame(rows);out.to_csv(P/'VAL_main_raw_MOS_MSE_decomposition.csv',index=False)
receipt=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ACTUAL_EXISTING_VAL_MSE_DECOMPOSITION_COMPLETE',model='v2_pure',rows=len(out),files=len(z),components=TERMS[1:],equal_domain=rows[-1],new_inference=0,audio_reads=0,new_fit=0,new_timing=0,new_hypothesis_tests=0,old_science_changed=False,source_implementation=str(R/'exp_C2/mse_decomposition_v3.py'),max_algebra_residual=float(out.algebra_residual.abs().max()))
(P/'VAL_MSE_ACTUAL_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
