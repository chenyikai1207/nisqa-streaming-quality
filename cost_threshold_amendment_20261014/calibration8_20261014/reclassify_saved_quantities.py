"""Reproduce threshold classification from saved fixed-session quantities only.

No waveform/model/runtime is used. Original cost data and historical gates are
read only; outputs default to a separate reproduced/ directory.
"""
from pathlib import Path
import argparse,csv,json,copy
HERE=Path(__file__).resolve().parent
REPOSITORY=HERE.parents[1]

def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def truth(x):return x is True or str(x).lower()=='true'
def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'reproduced');args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    original_path=REPOSITORY/'code/R4/exp_D/cost_v3/results/cost_final.csv'
    original_gate_path=REPOSITORY/'code/R4/exp_D/cost_v3/results/COST_ACCEPTANCE.json'
    original=table(original_path)
    original_gate=json.loads(original_gate_path.read_text(encoding='utf-8'))
    quantities=table(HERE/'fixed_session_gate_inputs.csv')
    calibration=table(HERE/'fixed_session_calibration_records.csv')
    lookup={(q['method'],q['path'],q['setting']):q for q in quantities}
    gate_lookup={(q['method'],q['path'],q['setting']):q for q in original_gate['gates']}
    assert len(original)==len(quantities)==16 and len(calibration)==80
    rows=[];costs=[];gates=[]
    for c in original:
        key=(c['method'],c['path'],c['setting']);q=lookup[key];g=copy.deepcopy(gate_lookup[key])
        assert c['session']==q['session_original']==g['session']
        cc=sorted([r for r in calibration if (r['method'],r['path'],r['setting'],r['session'])==key+(c['session'],)],key=lambda r:int(r['round']))
        assert len(cc)==5 and [int(r['round']) for r in cc]==list(range(5))
        drift=[(float(r['method_calib_drift']),float(r['round_calib_drift'])) for r in cc]
        maximum=max(max(a,b) for a,b in drift)
        iqr=float(q['iqr_ratio_from_fixed_runs'])
        other=all(truth(q[k]) for k in ['frequency_pass_unchanged','CPU_numerical_pass_unchanged','temperature_policy_pass_unchanged','formal_marker_pass_unchanged'])
        passed3=all(a<.03 and b<.03 for a,b in drift)
        passed6=all(a<.06 and b<.06 for a,b in drift)
        passed8=all(a<.08 and b<.08 for a,b in drift)
        old=(iqr<.05) and passed3 and other
        previous=(iqr<.10) and passed6 and other
        current=(iqr<.10) and passed8 and other
        assert old==truth(c['accepted'])==truth(q['accepted_original'])
        assert previous==truth(q['accepted_previous6'])
        r=dict(q,session_amended=c['session'],session_unchanged=True,accepted_original=old,accepted_previous6=previous,accepted_amended=current,iqr_threshold_original=.05,iqr_threshold_amended=.10,calibration_threshold_original=.03,calibration_threshold_previous6=.06,calibration_threshold_amended=.08,max_calib_drift_from_original_calibration_records=maximum,post_result_amendment=True)
        rows.append(r);costs.append(dict(c,accepted=current,accepted_original=old,accepted_previous6=previous,accepted_amended=current))
        for record,(a,b) in zip(g['calibration'],drift):
            record.update(calib_pass_original=a<.03 and b<.03,calib_pass_previous6=a<.06 and b<.06,calib_pass=a<.08 and b<.08)
        g.update(accepted_original=old,accepted_previous6=previous,accepted=current,dispersion_pass=iqr<.10,post_result_amendment=True,session_unchanged=True)
        gates.append(g)
    writecsv(out/'cost_final_amended.csv',costs)
    writecsv(out/'setting_reclassification_old_new.csv',rows)
    result=dict(settings=16,accepted_original=sum(r['accepted_original'] for r in rows),accepted_previous6=sum(r['accepted_previous6'] for r in rows),accepted_amended=sum(r['accepted_amended'] for r in rows),gates=gates,post_result_amendment=True,sessions_fixed=True,thresholds=dict(IQR_over_median_strict_lt=.10,method_and_round_calibration_drift_strict_lt=.08),other_gates_unchanged=True)
    (out/'COST_ACCEPTANCE_AMENDED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['settings','accepted_original','accepted_previous6','accepted_amended']}))

if __name__=='__main__':main()
