from public_paths import data_path, package_path
from public_paths import data_path, package_path
from public_paths import data_path, package_path
"""Fresh TRAIN-only postfreeze frontend fits; never writes the frozen first paper."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, datetime, json, sys, traceback
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIMS = ['mos', 'noi', 'col', 'dis', 'loud']

def now():
    return datetime.datetime.now().astimezone().isoformat()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', choices=['A','B','C','D'], default=['A','B'])
    ap.add_argument('--config', type=Path, default=HERE/'frontend_configs.json')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if not cfg['human_frontend_definition_confirmed']:
        raise RuntimeError('Human A-D frontend definition has not been confirmed.')
    if not (HERE/'prereg_gru96_frontend.md').exists():
        raise RuntimeError('Write actual preregistration before any fit.')
    if any(a in ['C','D'] for a in args.configs):
        gate = HERE/'SUMMARY_COMPLETE.json'
        if not gate.exists() or json.loads(gate.read_text())['B_reproduction']['passed'] is not True:
            raise RuntimeError('C/D require actual fresh A/B summary and B reproduction PASS; preserve failures without retuning.')
    stage = package_path(cfg['stage'])
    sys.path.insert(0, str(stage/'exp_A'))
    import torch
    from models_v3 import Model
    from feature_store_v3 import FeatureStore
    import train_engine_v3 as engine
    # Override inherited writers before the first engine call.
    engine.ROOT = HERE
    def note(message):
        with (HERE/'LOG.md').open('a', encoding='utf-8') as f:
            f.write('\n'+now()+' '+message+'\n')
    engine.note = note
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    def factory():
        model = Model(kind='gru', width=96, layers=2, kernel=3, input_channels=48, side_dim=4)
        assert sum(p.numel() for p in model.parameters()) == 98522
        return model
    d = pd.read_csv(stage/'shared/data_index.csv')
    d = d[d.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].sort_values('file').reset_index(drop=True)
    assert len(d) == 11020
    folds = pd.read_csv(stage/'shared/splits/train_folds.csv').set_index('file').loc[d.file].fold.to_numpy()
    loso = pd.read_csv(stage/'shared/splits/train_loso.csv')
    split_ids = list(loso.split_id.unique())
    assert len(split_ids) == 5 and 'SIM_TO_LIVE' in split_ids
    teacher = pd.read_csv(stage/'exp_A/teacher_TRAIN_predictions.csv').set_index('file').loc[d.file,DIMS].to_numpy()
    target = .7*d[DIMS].to_numpy()+.3*teacher
    raw_side = np.load(stage/'exp_A/cache_v3/packed/power16000_mel48/side.npy', mmap_mode='r')[:,:4]
    for arm in args.configs:
        spec = cfg['configs'][arm]
        source = data_path('cache/packed')/spec['cache']
        packed = json.loads((source/'COMPLETE.json').read_text())
        assert packed['files'] == d.file.tolist() and packed['shape'] == [11020,1199,48]
        # power cache is LINEAR mel power, not logmel: original FeatureStore
        # fits fold raw-RMS gain to -26dBFS, multiplies power before floor/log.
        # It is essential to preserve this original capacity-search producer.
        x = FeatureStore(str(source),'peak' if spec['normalization']=='whole_utterance_peak' else 'global_gain')
        mask = None
        side = torch.as_tensor(np.array(raw_side, copy=True) if spec['side']=='raw4' else np.zeros((len(d),4),np.float32))
        for seed in cfg['seeds']:
            preds = np.empty((len(d),5),np.float32)
            receipts = []
            for fold in range(5):
                ti = np.flatnonzero(folds!=fold);vi = np.flatnonzero(folds==fold)
                folder = HERE/'runs'/arm/f'seed{seed}'/f'cv{fold}'
                write_json(HERE/'STATE.json',dict(time=now(),status='ACTUAL_TRAIN_FIT',arm=arm,seed=seed,split=f'cv{fold}',folder=str(folder)))
                pp,r = engine.fit(x,mask,target,d,ti,vi,factory,seed,folder,side=side,
                    normalizer_key=f'{arm}_cv{fold}',uses_teacher=True)
                preds[vi]=pp;receipts.append(r)
            part=d[['file','domain','ref_group']].copy()
            for j,dim in enumerate(DIMS):part[dim]=preds[:,j]
            dest=HERE/'predictions'/arm;dest.mkdir(parents=True,exist_ok=True)
            part.to_csv(dest/f'seed{seed}_CV.csv',index=False)
            epochs=max(1,int(np.median([r['best_epoch'] for r in receipts])))
            parts=[]
            for sid in split_ids:
                split=loso[loso.split_id.eq(sid)].set_index('file').loc[d.file]
                ti=np.flatnonzero(split.role.eq('train'));vi=np.flatnonzero(split.role.eq('evaluation'))
                folder=HERE/'runs'/arm/f'seed{seed}'/sid
                write_json(HERE/'STATE.json',dict(time=now(),status='ACTUAL_TRAIN_FIT',arm=arm,seed=seed,split=sid,epochs=epochs,folder=str(folder)))
                pp,r=engine.fit(x,mask,target,d,ti,vi,factory,seed,folder,epochs=epochs,early_stop=False,
                    side=side,normalizer_key=f'{arm}_{sid}',uses_teacher=True)
                part=d.iloc[vi][['file','domain','ref_group']].copy();part['split_id']=sid
                for j,dim in enumerate(DIMS):part[dim]=pp[:,j]
                parts.append(part)
            pd.concat(parts,ignore_index=True).to_csv(dest/f'seed{seed}_LOSO.csv',index=False)
            note(f'Complete fresh arm={arm}, seed={seed}, all ten TRAIN split fits.')
        write_json(HERE/f'{arm}_FIT_COMPLETE.json',dict(time=now(),arm=arm,fresh_fits=30,reused_legacy_fits=0,
            frontend=spec,parameters=98522,source_cache=str(source),TRAIN_only=True,VAL_TEST_reads=0))
        del x,mask,side
        torch.cuda.empty_cache()
    write_json(HERE/'REQUESTED_FITS_COMPLETE.json',dict(time=now(),arms=args.configs,fits=30*len(args.configs),
        all_four_complete=all((HERE/f'{a}_FIT_COMPLETE.json').exists() for a in ['A','B','C','D']),
        scope='postfreeze TRAIN robustness, no model selection; summarize separately'))

if __name__=='__main__':
    try:main()
    except Exception:
        write_json(HERE/'STATE.json',dict(time=now(),status='ACTUAL_ERROR_PARTIAL_PRESERVED',error=traceback.format_exc()))
        raise
