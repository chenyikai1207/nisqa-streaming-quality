"""Saved-cost contrasts only; calibration sensitivity assumption is not CI or a true error bound."""
from pathlib import Path
import csv, json, datetime, math, itertools
P=Path(__file__).resolve().parents[1]
R=P/'results'
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def truth(v):return v is True or str(v).lower()=='true'
def key(r):return r['method'],r['path'],r['setting']
source_cost=R/'cost_final.csv';source_gate=R/'COST_ACCEPTANCE.json';source_index=R/'run_index.csv';source_mac=R/'mac_table.csv'
cost={key(r):r for r in read_csv(source_cost)}
acceptance=read_json(source_gate);gates={key(r):r for r in acceptance['gates']}
index=[r for r in read_csv(source_index) if r['phase']=='formal']
def saved(k):
    rr=[r for r in index if key(r)==k]
    if not rr:return dict(status='PENDING_NO_SAVED_RUN',key=k)
    session=rr[-1]['session'];rr=[r for r in rr if r['session']==session]
    good=[r for r in rr if r['exit_code']=='0']
    if len(good)!=5 or {int(r['round']) for r in good}!=set(range(5)):
        return dict(status='PENDING_LATEST_SESSION_INCOMPLETE',key=k,session=session,completed_rounds=len(good))
    c=cost.get(k);g=gates.get(k)
    if not c or not g or c['session']!=session or g['session']!=session:
        return dict(status='PENDING_POST_COST_SAME_OBJECT',key=k,session=session,completed_rounds=5)
    d=[x.get('calib_drift') for x in g['calibration']]
    if len(d)!=5 or any(x is None or not math.isfinite(float(x)) for x in d):
        return dict(status='UNAVAILABLE_CALIBRATION_DRIFT',key=k,session=session)
    return dict(status='ACTUAL_SAVED_FIVE_ROUND_OBJECT',key=k,session=session,rtf=float(c['rtf_median']),d=max(float(x) for x in d),accepted=truth(c['accepted']) and g['accepted'] is True,source_results=c['source_results'],same_CPU_prediction=g['first_round_prediction'],historical_prediction_check=g.get('historical_prediction_check'),gate_summary=dict(dispersion_pass=g['dispersion_pass'],calibration_pass=[x['calib_pass'] for x in g['calibration']],frequency_coverage=[x['frequency_coverage'] for x in g['frequency']],frequency_drop=[x['frequency_drop_over15pct_10s'] for x in g['frequency']]))
rows=[];evidence=[]
fields=['comparison','setting','reference','target','status','reference_session','target_session','reference_accepted','target_accepted','reference_rtf_median','target_rtf_median','d_reference','d_target','observed_k','sensitivity_lower','sensitivity_upper','interpretation','source_cost','source_gate']
def contrast(ref,target,name):
    a=saved(ref);b=saved(target);evidence.append(dict(comparison=name,reference=a,target=b))
    row=dict.fromkeys(fields);row.update(comparison=name,setting=target[2],reference='/'.join(ref[:2]),target='/'.join(target[:2]),reference_session=a.get('session'),target_session=b.get('session'),source_cost=str(source_cost),source_gate=str(source_gate))
    if a['status']!='ACTUAL_SAVED_FIVE_ROUND_OBJECT' or b['status']!='ACTUAL_SAVED_FIVE_ROUND_OBJECT':
        row.update(status=a['status'] if a['status']!='ACTUAL_SAVED_FIVE_ROUND_OBJECT' else b['status'],interpretation='No current-final ratio until latest five rounds and post_cost same-session object are complete; previous measured rows retained in original files')
    else:
        k=a['rtf']/b['rtf'];da=a['d'];db=b['d']
        row.update(status='ACTUAL_SAVED_CONTRAST',reference_accepted=a['accepted'],target_accepted=b['accepted'],reference_rtf_median=a['rtf'],target_rtf_median=b['rtf'],d_reference=da,d_target=db,observed_k=k)
        if da>=1 or db>=1:row.update(status='SENSITIVITY_UNAVAILABLE_D_GE_1',interpretation='Calibration sensitivity envelope unavailable; not CI or guaranteed true-error bound')
        else:
            lower=k*(1-da)/(1+db);upper=k*(1+da)/(1-db)
            row.update(sensitivity_lower=lower,sensitivity_upper=upper)
            if a['accepted'] and b['accepted']:label='ENVIRONMENT_ACCEPTED_ACTUAL_RATIO; sensitivity assumption separately reported'
            elif ref[0]=='PLCMOS' and ref[2]=='S3' and lower>5:label='USER_LOWER_GT5_SENSITIVITY_ONLY; environment not accepted; not an accepted measured speedup'
            else:label='OBSERVED_RATIO_AND_SENSITIVITY_ONLY; environment not accepted'
            row['interpretation']=label+'; assumes medians times (1 +/- max calibration drift), not CI or true-error guarantee'
    rows.append(row)
main3=('v2_96','deploy','S3')
contrast(('NISQA_MOS','official','S3'),main3,'S3_NISQA_MOS_over_main_deploy')
contrast(('PLCMOS','official','S3'),main3,'S3_PLCMOS_over_main_deploy')
# All six distinct same-P-core pairs among the four accepted S2 paths; the first three have main-deploy denominator.
main2=('v2_96','deploy','S2');others=[('NISQA_MOS','official','S2'),('v2_96','native','S2'),('v1','native','S2')]
for a in others:contrast(a,main2,'S2_'+a[0]+'_'+a[1]+'_over_main_deploy')
for a,b in itertools.combinations(others,2):contrast(a,b,'S2_'+a[0]+'_'+a[1]+'_over_'+b[0]+'_'+b[1])
macs={r['method']:r for r in read_csv(source_mac) if r['component']=='TOTAL' and r['mmac_per_s']}
macrows=[]
for ref in ['PLCMOS_official','NISQA_MOS_official']:
    for target in ['v2_96_native','v2_96_deploy']:
        a=macs[ref];b=macs[target];macrows.append(dict(reference=ref,target=target,reference_total_mmac_per_audio_second=float(a['mmac_per_s']),target_total_mmac_per_audio_second=float(b['mmac_per_s']),estimated_MAC_ratio=float(a['mmac_per_s'])/float(b['mmac_per_s']),unit='MMAC-equivalent estimate per audio second',source_mac=str(source_mac),scope=a['note'],interpretation='Front-end-included arithmetic estimates; FFT/FIR bounds are estimates; native and wholedeploy counted separately; not an RTF predictor'))
def write_csv(path,rr):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write_csv(R/'cost_contrasts.csv',rows);write_csv(R/'MAC_contrasts.csv',macrows)
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_SAVED_CONTRASTS_WITH_EXPLICIT_PENDING',sensitivity_definition='For each matched latest complete five-round session d=max calib_drift; hypothetical median*(1+-d); lower=k*(1-d_ref)/(1+d_target), upper=k*(1+d_ref)/(1-d_target); d>=1 unavailable',sensitivity_is_confidence_interval=False,sensitivity_guarantees_true_error_bounds=False,posthoc_human_requested=True,reference_acceptance_not_promoted=True,sources=dict(cost=str(source_cost),gate=str(source_gate),index=str(source_index),MAC=str(source_mac)),contrasts=rows,MAC_contrasts=macrows,actual_object_evidence=evidence,new_forward=0,new_measurements=0,quality_recalculations=0,all_cost_accepted=acceptance['all_accepted'],rerunnable_after_root_post_cost_refresh=True)
(R/'COST_CONTRASTS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(status=receipt['status'],contrasts=rows,MAC_contrasts=macrows),ensure_ascii=False))