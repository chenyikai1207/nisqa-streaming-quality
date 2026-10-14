"""G1-only standard five-dimensional prediction CLI; preserves v1 / selected_v2 / main identity."""
import argparse,json,sys,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
DIMS=['mos','noi','col','dis','loud']
def selection(stage):
    a=Path(stage)/'exp_A';p=a/'FROZEN_SELECTION_v3.json';z=json.loads(p.read_text());assert z['frozen'] is True and z['protocol_revision']=='v3_synthetic_review' and (a/'FROZEN_v2.md').exists();return z
def resolve_model(stage,name):
    stage=Path(stage);z=selection(stage);alias={'selected_v2':z['v2_comparator'],'final_main_model':z['main_model'],'v1':'v1'};actual=alias.get(name,name)
    if actual=='v1':return dict(requested=name,actual='v1',candidate_id='v1_A_distill',kind='legacy_v1',folder=str(stage/'exp_A/frozen/A_distill'),runtime=str(stage/'exp_A/frozen/code/runtime.py'))
    entry=next(x for x in z['candidates'] if x['name']==actual or x['candidate']==actual);return dict(requested=name,actual=entry['name'],candidate_id=entry['candidate'],kind='v2_v3_runtime',folder=entry['folder'],runtime=z['runtime'])
def load_predictor(spec,device):
    code=Path(spec['runtime']).parent;sys.path.insert(0,str(code))
    if spec['kind']=='legacy_v1':
        entry=importlib.util.spec_from_file_location('frozen_legacy_v1_runtime',spec['runtime']);m=importlib.util.module_from_spec(entry);entry.loader.exec_module(m);return m.Predictor(spec['folder'],device)
    from runtime_model_v3 import Predictor
    return Predictor(spec['folder'],device)
def main():
    stage_default=next((p for p in Path(__file__).resolve().parents if (p/'CONTRACT_v3.md').exists()),Path(__file__).resolve().parents[1]);p=argparse.ArgumentParser();p.add_argument('--stage',default=str(stage_default));p.add_argument('--model',default='selected_v2');p.add_argument('--manifest','--input','--csv',dest='manifest',required=True);p.add_argument('--output','--out',dest='output',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--mapped',action='store_true');p.add_argument('--stream-block-ms',type=int,choices=[10,20,100]);a=p.parse_args();spec=resolve_model(a.stage,a.model);d=pd.read_csv(a.manifest);assert {'file','path'}.issubset(d.columns) and not d.file.duplicated().any();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);meta=out.with_suffix(out.suffix+'.metadata.json');identity=dict(model=spec,manifest=str(Path(a.manifest).resolve()),file_keys=d.file.tolist(),mapped=a.mapped,stream_block_ms=a.stream_block_ms,device=a.device)
    if meta.exists():assert json.loads(meta.read_text())['identity']==identity,'existing predictions have a different concrete model/input identity'
    if out.exists():assert meta.exists(),'prediction CSV has no saved concrete model/input identity'
    if not meta.exists():meta.write_text(json.dumps(dict(identity=identity,completed=0,expected=len(d),G1_checked=True,complete=False),indent=2))
    rows=pd.read_csv(out).to_dict('records') if out.exists() else [];done={r['file'] for r in rows};assert done.issubset(set(d.file));model=load_predictor(spec,a.device)
    if spec['kind']=='legacy_v1' and (a.mapped or a.stream_block_ms):raise ValueError('Legacy v1 CLI preserves its original raw whole-peak offline identity')
    for r in d.itertuples():
        if r.file in done:continue
        if spec['kind']=='legacy_v1':w,sr=model.load_audio(r.path);pred=model.infer(model.extract_features(w,sr))[0]
        elif a.stream_block_ms:
            import soundfile as sf
            w,sr=sf.read(r.path,dtype='float32');s=model.new_stream(sr);step=sr*a.stream_block_ms//1000
            for start in range(0,len(w),step):s.push(w[start:start+step])
            s.finish();pred=s.predict(mapped=a.mapped)[0]
        else:pred=model.predict_file(r.path,mapped=a.mapped)[0]
        rows.append(dict(file=r.file,**dict(zip(DIMS,np.asarray(pred,float)))));done.add(r.file)
        if len(rows)%64==0 or len(rows)==len(d):
            table=pd.DataFrame(rows).set_index('file').loc[d[d.file.isin(done)].file].reset_index();tmp=out.with_suffix(out.suffix+'.tmp');table.to_csv(tmp,index=False);tmp.replace(out);meta.write_text(json.dumps(dict(identity=identity,completed=len(rows),expected=len(d),G1_checked=True,complete=len(rows)==len(d)),indent=2));print(spec['actual'],len(rows),'/',len(d),flush=True)
if __name__=='__main__':main()
