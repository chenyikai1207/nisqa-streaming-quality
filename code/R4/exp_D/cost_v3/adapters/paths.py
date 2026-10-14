from public_paths import data_path, package_path
import os,sys,json
from pathlib import Path
COST=Path(__file__).resolve().parents[1];STAGE=COST.parents[1];D2=STAGE/'exp_D2'
def native_path(s):
    s=str(s).replace(chr(92),'/')
    if s.startswith('DATA_ROOT/'):
        from public_paths import data_path
        return str(data_path(s[len('DATA_ROOT/'):]))
    if s.startswith('code/R4/'):
        from public_paths import package_path
        return str(package_path(s))
    return s

def add_code(path):
    s=str(path)
    if s not in sys.path:sys.path.insert(0,s)
def configure_torch():
    import torch
    if os.environ.get('NISQA_COST_CORE'):torch.set_num_threads(1)
    return torch
def prediction(a):
    import numpy as np
    v=np.asarray(a).reshape(-1);return dict(zip(['mos','noi','col','dis','loud'],map(float,v)))
