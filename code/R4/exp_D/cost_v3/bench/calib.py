import os,sys,json,time,argparse
ap=argparse.ArgumentParser();ap.add_argument('--core',type=int);ap.add_argument('--out',required=True);a=ap.parse_args()
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import psutil
if a.core is not None:psutil.Process().cpu_affinity([a.core])
import numpy as np
rng=np.random.default_rng(20261011);x=rng.standard_normal((384,384),dtype=np.float32);y=rng.standard_normal((384,384),dtype=np.float32)
for _ in range(50):x@y
ts=[]
for _ in range(15):
    start=time.perf_counter_ns()
    for _ in range(200):x@y
    ts.append((time.perf_counter_ns()-start)/1e6)
ts.sort();r=dict(core=a.core,median_ms=ts[7],min_ms=ts[0],max_ms=ts[-1],time=time.time(),all_ms=ts,actual_affinity=psutil.Process().cpu_affinity())
from pathlib import Path
Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf8')
