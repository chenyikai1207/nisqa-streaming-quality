"""Official MOS-only inference on the deposited fixed384TCD identities.

External author code, pretrained weight and lawfully obtained speech are supplied
through DATA_ROOT/explicit arguments. No cost measurement or other set is run.
"""
from pathlib import Path
import argparse,os,sys,json
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=os.environ.get('DATA_ROOT'))
    ap.add_argument('--code-dir',type=Path);ap.add_argument('--weight',type=Path)
    ap.add_argument('--out',type=Path,default=here/'reproduced_predictions');args=ap.parse_args()
    if args.data_root is None:raise RuntimeError('Provide DATA_ROOT or --data-root for external corpus/code/weights.')
    code=args.code_dir or args.data_root/'third_party/NISQA'
    weight=args.weight or code/'weights/nisqa_mos_only.tar'
    if not weight.is_file() or weight.stat().st_size!=907215:raise RuntimeError('The specified author MOS-only907215B weight is required.')
    d=pd.read_csv(here/'TCD_fixed_support_metadata.csv')
    assert len(d)==384 and d.file.is_unique and d.ref_group.nunique()==96 and d.condition_id.nunique()==96
    d['path']=[str(args.data_root/str(p)) for p in d.archive_member]
    if not all(Path(p).is_file() for p in d.path):raise RuntimeError('Obtain original TCD-VOIP and preserve its archive-relative member paths under DATA_ROOT.')
    args.out.mkdir(parents=True,exist_ok=True)
    d[['file','db','path']].to_csv(args.out/'TCD_official_MOS_input.csv',index=False)
    torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    sys.path.insert(0,str(code));original_load=torch.load
    def load(*a,**kwargs):kwargs.setdefault('weights_only',False);return original_load(*a,**kwargs)
    torch.load=load
    from nisqa.NISQA_model import nisqaModel
    m=nisqaModel(dict(mode='predict_csv',pretrained_model=str(weight),data_dir='',csv_file=str(args.out/'TCD_official_MOS_input.csv'),csv_deg='path',output_dir=str(args.out),tr_num_workers=0,tr_bs_val=16,tr_parallel=False,tr_device='cuda',ms_channel=None))
    m.model.to(m.dev).eval();parameters=sum(p.numel() for p in m.model.parameters());assert parameters==218402
    predictions=[]
    with torch.inference_mode():
        for xb,yb,(ix,nwins) in DataLoader(m.ds_val,batch_size=16,shuffle=False,num_workers=0):
            y=m.model(xb.to(m.dev),nwins.to(m.dev)).cpu().numpy();assert y.ndim==2 and y.shape[1]==1
            predictions.append(y)
    pred=pd.DataFrame({'file':d.file,'mos':np.concatenate(predictions)[:,0]})
    assert len(pred)==384 and pred.file.is_unique and np.isfinite(pred.mos).all()
    pred.to_csv(args.out/'TCD_MOS_predictions.csv',index=False)
    identity=dict(model='NISQA_MOSONLY',parameters=parameters,weight_filename='nisqa_mos_only.tar',weight_bytes=907215,variant='MOSONLY',device='CUDA',torch=torch.__version__,cuda_version=torch.version.cuda,batch_size=16,matmul_tf32=False,cudnn_tf32=True,output_heads=['mos'],frontend='original author dataset/checkpoint whole preprocessing')
    (args.out/'model_identity.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
