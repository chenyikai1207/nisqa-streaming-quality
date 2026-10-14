"""Export unchanged frozen tensors and genuine CPU full-sequence TRAIN200 reference."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,csv,sys,time
import numpy as np,torch,soundfile as sf,psutil
P=Path(__file__).resolve().parent;Q=P/'lean_raw_v5';Q.mkdir(exist_ok=True)
from A7_runtime_entry_v3 import load_predictor,native_path
torch.set_num_threads(1);psutil.Process().cpu_affinity([0]);ready=json.loads((P.parent/'exp_A/A7_MODELS_READY_v3.json').read_text(encoding='utf8'));sample=list(csv.DictReader((P.parent/'exp_C/equivalence_sample200.csv').open(encoding='utf8')))
assert len(sample)==200
for name in ['v2_pure','v2_64']:
    if (Q/(name+'_export_COMPLETE.json')).exists():continue
    row=next(r for r in ready['models'] if r['name']==name);p=load_predictor(ready,row,'cpu');assert p.model.kind=='gru' and not p.model.fusion
    arrays={k:v.detach().cpu().numpy() for k,v in p.model.state_dict().items()};arrays.update(mean=p.mean.numpy(),std=p.std.numpy(),side_mean=p.sm.numpy(),side_std=p.ss.numpy(),window=p.front.window.numpy(),mel=p.front.mel.numpy(),gain=np.array(p.gain),layers=np.array(p.model.layers),width=np.array(p.model.width),sr=np.array(p.front.config.sr))
    np.savez(Q/(name+'_frozen_arrays.npz'),**arrays)
    out=[]
    for i,r in enumerate(sample):
        path=native_path(r['audio_path']);wave,sr=sf.read(path,dtype='float32');pred=p.predict_array(wave,sr,mapped=False).reshape(5)
        out.append(dict(file=r['record_id'],source_path=path,native_sr=sr,native_samples=len(wave),prediction=json.dumps(pred.tolist())))
        if (i+1)%50==0:print(name,'CPU genuinefull',i+1,flush=True)
    with (Q/(name+'_CPU_full_reference_TRAIN200.csv')).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
    (Q/(name+'_export_COMPLETE.json')).write_text(json.dumps(dict(model=name,TRAIN_n=200,device='CPU',reference='genuine native full sequence Predictor.predict_array; original weights unchanged',weight_source=row['folder'],parameters=sum(t.numel() for t in p.model.parameters()),quality_metrics_computed=0,VAL_TEST_audio=0)),encoding='utf8')