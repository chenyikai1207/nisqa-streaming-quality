"""Raw file/ALL descriptive MOS metrics; no mapping, bootstrap or paired test."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from scipy.stats import spearmanr

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,default=here/'TCD_MOS_predictions.csv')
    ap.add_argument('--metadata',type=Path,default=here/'TCD_fixed_support_metadata.csv')
    ap.add_argument('--out',type=Path,default=here/'reproduced_metrics.csv');args=ap.parse_args()
    d=pd.read_csv(args.metadata);pred=pd.read_csv(args.predictions)
    assert len(d)==len(pred)==384 and d.file.is_unique and pred.file.is_unique and set(d.file)==set(pred.file)
    assert d.ref_group.nunique()==96 and d.condition_id.nunique()==96
    z=d[['file','mos']].merge(pred[['file','mos']],on='file',suffixes=('_truth','_pred'),validate='one_to_one')
    a=z.mos_truth.to_numpy();b=z.mos_pred.to_numpy();assert np.isfinite(a).all() and np.isfinite(b).all()
    scores={'Pearson':float(np.corrcoef(a,b)[0,1]),'MAE':float(np.mean(abs(b-a))),'RMSE':float(np.sqrt(np.mean((b-a)**2))),'Spearman':float(spearmanr(a,b).statistic)}
    rows=[dict(model='NISQA_MOSONLY',dataset='TCD_VOIP',domain='ALL',level='file',dim='mos',metric=k,mapping='none',value=v,ci_low=None,ci_high=None,n=384,n_boot=0,quality_device='CUDA',model_version='author nisqa_mos_only.tar',parameters=218402) for k,v in scores.items()]
    pd.DataFrame(rows).to_csv(args.out,index=False)

if __name__=='__main__':main()
