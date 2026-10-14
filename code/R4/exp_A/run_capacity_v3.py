"""Budget-filtered 25 registered causal structures; actual same three seeds and shared folds."""
import json,os
from functools import partial
import pandas as pd
from audit_frontend_v3 import ROOT,STAGE,DIMS,note
from models_v3 import Model
from run_frontend_v3 import run_candidate,write_tables
from gpu_workers_v3 import shutdown_pool
def main():
    spec=json.loads((ROOT/'A2_CANDIDATES_SCREENED_v3.json').read_text());assert spec['all_structures']==25
    d=pd.read_csv(STAGE/'shared/data_index.csv');d=d[d.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].sort_values('file').reset_index(drop=True)
    teacher=pd.read_csv(ROOT/'teacher_TRAIN_predictions.csv').set_index('file').loc[d.file,DIMS].to_numpy();target=.7*d[DIMS].to_numpy()+.3*teacher;rows=[]
    note(f'新A2实际PID={os.getpid()}，预算内{spec["eligible_structures"]}结构/{spec["planned_fits"]}fit，固定种子/折/epoch，复用新A1缓存与TRAIN标准化。')
    for arch in spec['candidates']:
        assert arch['within_budget'];cfg=dict(spec['selected_frontend'],candidate=arch['candidate']);factory=partial(Model,**{k:arch[k] for k in ['kind','width','layers','kernel','input_channels','side_dim']});current=[]
        for current in run_candidate(cfg,d,target,factory=factory,phase='A2'):write_tables(rows+current,prefix='A2_capacity')
        rows.extend(current)
    shutdown_pool();assert len(rows)==spec['eligible_structures']*3*5*4
    (ROOT/'A2_COMPLETE_v3.json').write_text(json.dumps({'time':pd.Timestamp.now().isoformat(),'actual_structures':spec['eligible_structures'],'planned_and_completed_fits':spec['planned_fits'],'screened_structures':25,'seeds':[0,1,2],'dimensions':5,'VAL_audio':0,'TEST_audio':0},indent=2));note('新A2实际预算内全部fit完成；结构选模/Pareto进入新B1/A3，旧THCHS100k仍历史排除。')
    from select_capacity_v3 import main as select
    select()
    from handoff_distill_inputs_v3 import main as handoff
    handoff()
if __name__=='__main__':main()
