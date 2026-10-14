"""Root-serial S3 fullVAL representativeness: two actual2700passes; no600replacement."""
import os,sys,argparse,csv,subprocess,datetime,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--clean-marker',default=str(P/'env/CLEAN_ENV_READY.json'));a=ap.parse_args()
from post_common import read_json,write_json,write_csv,np
selection=read_json(P/'deploy_v2/DEPLOYMENT_SELECTION.json');adapter=selection['selected_adapter_by_model']['v2_96'];cores=read_json(P/'env/cores.json');s=P.parents[1];rows=list(csv.DictReader((s/'shared/val_timing_index.csv').open(encoding='utf8')));assert len(rows)==2700
# Root corpus is already verified and benchmarklist maps the exactsame official relativepaths.
index=list(csv.DictReader((P/'lists/bench_index.csv').open(encoding='utf8')));sample=index[0];relative=sample['file'].split('|',1)[1];root=sample['path'].replace('\\','/')[:-len(relative)];files=P/'bench/representativeness_VAL2700_files.txt';files.write_text(''.join(root+r['file'].split('|',1)[1]+'\n' for r in rows),encoding='utf8')
jobs=[('v2_96',adapter),('NISQA_MOS','adapters.official:NISQAMOS')];out=P/'results/representativeness';out.mkdir(exist_ok=True)
if not a.execute:
    print('READY_FULLVAL_REPRESENTATIVENESS',str(files),jobs);raise SystemExit(0)
gate=read_json(a.clean_marker)
if gate.get('formal_allowed') is not True:raise RuntimeError('Root actual formalenvironment required for fullVAL timing; no runstarted')
formal=list(csv.DictReader((P/'results/cost_final.csv').open(encoding='utf8')));results=[];audio=sum(float(r['duration_s']) for r in rows);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for model,entry in jobs:
    selected_path='deploy' if model=='v2_96' and ':Runner' in entry and entry.endswith(('F32','F64')) else 'native' if model=='v2_96' else 'official';refs=[r for r in formal if r['method']==model and r['path']==selected_path and r['setting']=='S3']
    if len(refs)!=1:raise RuntimeError('Matchingactual600 S3 rowrequired before representativenessrun')
    dest=out/(stamp+'_'+model+'.json');subprocess.run([sys.executable,str(P/'bench/run_one.py'),'--adapter',entry,'--files',str(files),'--core',str(cores['S3']['core_id']),'--warmup_n','20','--expected-n','2700','--out',str(dest),'--clean-marker',a.clean_marker],check=True);run=read_json(dest);assert run['n_files']==2700 and run['warmup_files']==20;rtf=run['wall_ns']/1e9/audio;ref=float(refs[0]['rtf_median']);results.append(dict(method=model,path=selected_path,n_files=2700,audio_s=audio,rtf_fullVAL_single=rtf,rtf_bench600_fiveround_median=ref,relative_difference=abs(rtf/ref-1),within5percent=abs(rtf/ref-1)<.05,fullVAL_source=str(dest),bench600_session=refs[0]['session'],scope='one fullVAL representativenesspass, notfive-round cost replacement'))
write_csv(out/'representativeness.csv',results);write_json(out/'ACTUAL_REPRESENTATIVENESS_COMPLETE.json',dict(status='ACTUAL_TWO_FULLVAL_PASSES_COMPLETE',results=results,all_within5percent=all(r['within5percent'] for r in results),formal_cost_replacement=False))

