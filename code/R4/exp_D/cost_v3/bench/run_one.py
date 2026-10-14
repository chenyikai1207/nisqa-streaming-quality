"""One independent hotcache run; no perfile stdout, no RSS sampling in timingprocess.
Formal mode requires actual root CLEAN_ENV_READY marker; engineering flag is never formalcost.
"""
import os,sys,json,time,argparse,importlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--files',required=True);ap.add_argument('--core',type=int);ap.add_argument('--warmup_n',type=int,default=20);ap.add_argument('--out',required=True);ap.add_argument('--save_pred');ap.add_argument('--engineering',action='store_true');ap.add_argument('--clean-marker');ap.add_argument('--expected-n',type=int);a=ap.parse_args()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if not a.engineering:
    marker=Path(a.clean_marker) if a.clean_marker else ROOT/'env/CLEAN_ENV_READY.json'
    if not marker.exists():raise RuntimeError('Actual root CLEAN_ENV_READY required, no formalmeasurementstarted')
    gate=json.loads(marker.read_text(encoding='utf8'))
    if not (gate.get('ready') is True or gate.get('status') in ['CLEAN_ENV_READY','READY','ACTUAL_CLEAN_ENV_READY']):raise RuntimeError('Root cleanmarker notready')
if a.core is not None:
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
    os.environ['NISQA_COST_CORE']=str(a.core)
else:
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.pop(k,None)
    os.environ.pop('NISQA_COST_CORE',None)
os.environ['NISQA_COST_FILES']=str(Path(a.files).resolve())
import psutil
if a.core is not None:psutil.Process().cpu_affinity([a.core])
files=[x.strip() for x in open(a.files,encoding='utf8') if x.strip()]
if not a.engineering and len(files)>=600 and gate.get('formal_allowed') is not True:raise RuntimeError('Root marker permitsdiagnostic only; formal600/fullVAL notauthorized byactualenvironment gate')
if a.expected_n is not None and len(files)!=a.expected_n:raise RuntimeError('Actual measurement filecount differs from preregistered count')
from adapters.base import StageTimer
mod,cls=a.adapter.split(':');start=time.perf_counter_ns();runner=getattr(importlib.import_module(mod),cls)();cold=time.perf_counter_ns()-start
for path in files:
    with open(path,'rb') as f:f.read()
warm=StageTimer()
for path in files[:a.warmup_n]:runner.process(path,warm)
timer=StageTimer();preds={} if a.save_pred else None;ts=time.time();start=time.perf_counter_ns()
for path in files:
    value=runner.process(path,timer)
    if preds is not None:preds[path]=value
wall=time.perf_counter_ns()-start;end=time.time();proc=psutil.Process()
instrumented=dict(timer.acc);unattributed=max(0,wall-sum(timer.acc.values()));timer.acc['O']+=unattributed
result=dict(instrumented_stages_ns=instrumented,adapter=a.adapter,core=a.core,actual_affinity=proc.cpu_affinity(),n_files=len(files),warmup_files=min(a.warmup_n,len(files)),wall_ns=wall,stages_ns=timer.acc,cold_ns=cold,rss_end_bytes=proc.memory_info().rss,t_start=ts,t_end=end,metadata=runner.metadata,threads_actual=proc.num_threads(),torch_num_threads=__import__('torch').get_num_threads() if 'torch' in sys.modules else None,scope='ENGINEERING_TWO_FILE_INTERFACE_NOT_FORMAL_RTF' if a.engineering else ('CLEAN_COST_V3_DIAGNOSTIC50_ONE_ROUND' if len(files)==50 else 'CLEAN_COST_V3_VAL2700_REPRESENTATIVENESS_ONE_PASS' if len(files)==2700 else 'CLEAN_COST_V3_VAL600_ONE_ROUND'),pagecache='allfilesreadbeforewarmup',timing_observer='no sampling thread, no perfile output',O_unattributed_ns=unattributed)
# Caller can use O total=stagedO+O_unattributed; the raw exclusive decomposition is retained.
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
if a.save_pred:Path(a.save_pred).write_text(json.dumps(preds,ensure_ascii=False,indent=2),encoding='utf8')
