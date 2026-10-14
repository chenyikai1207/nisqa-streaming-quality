"""Separate RSS600 process,10ms sampling; never mixed with timedcost rounds."""
import os,sys,argparse,json,importlib,time,threading
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--files',required=True);ap.add_argument('--core',type=int);ap.add_argument('--clean-marker',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();P=Path(__file__).resolve().parents[1];gate=json.loads(Path(a.clean_marker).read_text(encoding='utf8'));assert gate.get('formal_allowed') is True
if a.core is not None:
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['NISQA_COST_CORE']=str(a.core)
else:
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.pop(k,None)
    os.environ.pop('NISQA_COST_CORE',None)
os.environ['NISQA_COST_FILES']=a.files;sys.path.insert(0,str(P));import psutil
proc=psutil.Process()
if a.core is not None:proc.cpu_affinity([a.core])
from adapters.base import StageTimer
mod,cls=a.adapter.split(':');r=getattr(importlib.import_module(mod),cls)();files=[x.strip() for x in open(a.files,encoding='utf8') if x.strip()];assert len(files)==600
baseline=proc.memory_info().rss;samples=[baseline];stop=threading.Event()
def sample():
    while not stop.wait(.01):samples.append(proc.memory_info().rss)
t=threading.Thread(target=sample,daemon=True);t.start()
try:
    for path in files:r.process(path,StageTimer())
finally:stop.set();t.join()
peak=max(samples+[proc.memory_info().rss]);out=dict(adapter=a.adapter,core=a.core,n_files=600,baseline_rss_bytes=baseline,baseline_rss_mb=baseline/1e6,peak_rss_bytes=peak,peak_rss_mb=peak/1e6,sampling_interval_ms=10,samples=len(samples),lifetime_peak_working_set_bytes=getattr(proc.memory_info(),'peak_wset',None),scope='independentmemorypass; afterlibraries/modelbeforefirstaudio baseline; notRTF',metadata=r.metadata)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf8')
