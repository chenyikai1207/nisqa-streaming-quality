from public_paths import data_path, package_path
"""Current A7 runtime entry shared by D200 and future D2/D3/D4, read-only model loading."""
import os,sys,json,importlib
from pathlib import Path

def native_path(s):
    s=str(s).replace(chr(92),'/')
    if s.startswith('DATA_ROOT/'):return str(data_path(s[len('DATA_ROOT/'):]))
    if s.startswith('code/R4/'):return str(package_path(s))
    return s

def load_predictor(ready,candidate,device='cuda'):
    code=Path(native_path(ready['runtime_module'])).parent
    if str(code) not in sys.path:sys.path.insert(0,str(code))
    runtime=importlib.import_module('runtime_model_v3')
    predictor=runtime.Predictor(native_path(candidate['folder']),device)
    # Only process-local path adaptation for Windows; no frozen spec/file changes.
    if predictor.spec.get('B2'):
        b=predictor.spec['B2'];b['weights_path']=native_path(b['weights_path'])
        for key in ['stream_entry','synthetic_predict_entry']:
            if b.get(key) and b[key].get('module_dir'):b[key]['module_dir']=native_path(b[key]['module_dir'])
    return predictor