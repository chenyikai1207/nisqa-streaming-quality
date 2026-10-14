from pathlib import Path
import csv,json,datetime
import numpy as np
P=Path(__file__).resolve().parents[1];STAGES=['L','R','G','F','I','O']
def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write_json(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def write_csv(path,rows,fields=None):
    if not rows:raise RuntimeError('No real rows; no placeholder tablewritten')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def index_rows(phase):return [r for r in csv.DictReader((P/'results/run_index.csv').open(encoding='utf8')) if r['phase']==phase]
def actual_run(row,n):
    if int(row['exit_code'])!=0:return None
    path=Path(row['result'])
    if not path.exists():return None
    r=read_json(path)
    if r['n_files']!=n or r['warmup_files']!=20 or r['scope'].startswith('ENGINEERING'):return None
    return r
