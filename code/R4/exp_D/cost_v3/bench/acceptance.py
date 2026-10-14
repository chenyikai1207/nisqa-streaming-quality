"""Actual calibration/frequency/registered frozenprediction acceptance; never changes thresholds."""
from post_common import *
import math
MODEL={'v2_96':'v2_pure','v2_64':'v2_64','v1':'v1','NISQA_MOS':'NISQA_MOSONLY','NISQA5D':'NISQA5D','PLCMOS':'PLCMOS','DNSMOS':'DNSMOS','SQUIM':'SQUIM'}
def calibration(row):
    q,r=Path(row['calib_before']),Path(row['calib_after'])
    if not q.exists() or not r.exists():return dict(calib_drift=None,calib_pass=False)
    a,b=read_json(q),read_json(r);drift=abs(b['median_ms']/a['median_ms']-1)
    qb,qa=Path(row['round_calib_before']),Path(row['round_calib_after'])
    if not qb.exists() or not qa.exists():return dict(calib_drift=float(drift),round_calib_drift=None,calib_pass=False)
    before,after=read_json(qb),read_json(qa);round_drift=abs(after['median_ms']/before['median_ms']-1)
    return dict(calib_drift=float(max(drift,round_drift)),method_calib_drift=float(drift),round_calib_drift=float(round_drift),calib_pass=bool(drift<.03 and round_drift<.03))
def frequency(row,run):
    points=[];marker=read_json(P/'env/CLEAN_ENV_READY.json')
    log_path=Path(marker['frequency_log']) if marker.get('frequency_log') else Path(marker['source_monitor']).parent/'frequency_load.jsonl'
    if not log_path.exists():return dict(freq_mean_mhz=None,temp_max_c=None,frequency_source='missing actual canonical logger',frequency_log=str(log_path),frequency_points=0,frequency_coverage=False,frequency_drop_over15pct_10s=False,frequency_baseline='unavailable',temperature_verified=False,gpu_scientific_processes_observed=False,observed_cpu_percent_mean=None,observed_cpu_percent_max=None,background_scope='No fallback to historical UNC frequency segment')
    for line in log_path.open(encoding='utf8'):
        try:r=json.loads(line);ts=datetime.datetime.fromisoformat(r['time']).timestamp()
        except (ValueError,KeyError):continue
        if run['t_start']<=ts<=run['t_end']:points.append((ts,r))
    core=row.get('core_id')
    if core is None:core=run['core']
    coreids=run['actual_affinity'] if core is None else [int(core)]
    values=[];drop=False;coverage=True;baselines={};source=set();max_temp=[]
    for c in coreids:
        seq=[(ts,float(r.get('effective_frequency_mhz',{}).get(str(c),float('nan')))) for ts,r in points];seq=[(ts,v) for ts,v in seq if math.isfinite(v)]
        if not seq:coverage=False;continue
        if seq[0][0]-run['t_start']>2.5 or run['t_end']-seq[-1][0]>2.5:coverage=False
        values.extend(v for ts,v in seq);baseline=float(np.median([v for ts,v in seq[:min(3,len(seq))]]));baselines[str(c)]=baseline;start=None;last=None
        for ts,v in seq:
            if last is not None and ts-last>2.5:start=None;coverage=False
            if v<.85*baseline:
                if start is None:start=ts
                if ts-start>=10:drop=True
            else:start=None
            last=ts
    for ts,r in points:
        source.add(r.get('frequency_source','unknown'))
        value=r.get('cpu_temperature_c')
        if isinstance(value,(int,float)) and math.isfinite(value):max_temp.append(value)
    cpu=[r['cpu_percent'] for ts,r in points if isinstance(r.get('cpu_percent'),(int,float))]
    return dict(freq_mean_mhz=float(np.mean(values)) if values else None,temp_max_c=max(max_temp) if max_temp else None,frequency_source='|'.join(sorted(source)),frequency_log=str(log_path),frequency_points=len(points),frequency_coverage=bool(coverage and points),frequency_drop_over15pct_10s=bool(drop),frequency_baseline='first up-to3 in-run1s samples perselectedcore; noMSRclaim',temperature_verified=bool(max_temp),gpu_scientific_processes_observed=any(r.get('gpu',{}).get('scientific_compute_processes') for ts,r in points),observed_cpu_percent_mean=float(np.mean(cpu)) if cpu else None,observed_cpu_percent_max=max(cpu) if cpu else None,background_scope='actual machine CPU includes the measured worker and basic application load; no idle percentage veto or unloaded claim')
def registered_predictions(row,index_kind='bench',reference_approval=None):
    path=Path(row['prediction']) if row.get('prediction') else None
    if path is None or not path.exists():return dict(prediction_pass=False,prediction_max_abs=None,prediction_failure='actualfirst-round savedpredmissing')
    output=read_json(path);model=MODEL[row['method']];sources=[r for r in csv.DictReader((P/'results/integrity_check.csv').open(encoding='utf8')) if r['dataset']=='VAL' and r['model']==model and r['status']=='PASS']
    if len(sources)!=1:return dict(prediction_pass=False,prediction_max_abs=None,prediction_failure='unique registeredC frozenVAL sourceunavailable')
    source=sources[0];reference_device='registered C original device (CUDA); sameCPU2file isnot substitute'
    if reference_approval:
        approval=read_json(reference_approval)
        if approval.get('approved') is not True or approval.get('reference_device')!='CPU' or approval.get('native_prediction_threshold')!=1e-6:raise RuntimeError('Actual humanapproved rootCPUreference policy required; no silent gatechange')
        entry=approval.get('native_CPU_reference',{}).get(row['method'])
        if not entry:return dict(prediction_pass=False,prediction_max_abs=None,prediction_failure='Actual independentCPUreference for thismethod missing; no fallback to benchmarkoutput')
        policy='default' if row['setting']=='S1' else 'single'
        if policy in entry:entry=entry[policy]
        receipt=read_json(entry['receipt']);identity=receipt['identity']
        if receipt.get('complete') is not True or receipt.get('n_files')!=600 or identity.get('model')!=row['method'] or identity.get('reference_device')!='CPU' or identity.get('staged_adapter_imported') is not False or identity.get('mapped') is not False or identity.get('thread_policy')!=policy:raise RuntimeError('Independent originalCPUproducer identity/completion/threadpolicy mismatch')
        source=dict(path=entry['csv'],required_output_columns='|'.join(identity['output_heads']));reference_device='Independent original frozenCPU wholeproducer, humanapproved sameCPU protocol; legacy C CUDA difference preserved separately'
    ref={r['file']:r for r in csv.DictReader(Path(source['path']).open(encoding='utf8'))};lookup={r['path'].replace('\\','/'):r['file'] for r in csv.DictReader((P/'lists'/(index_kind+'_index.csv')).open(encoding='utf8'))};diffs=[];missing=[]
    expected=source['required_output_columns'].split('|')
    for path,pred in output.items():
        key=lookup.get(path.replace('\\','/'))
        if key is None or key not in ref:missing.append(path);continue
        for dim in expected:
            if dim not in pred:missing.append(key+'|'+dim);continue
            target=float(ref[key][dim]);value=float(pred[dim])
            if not math.isfinite(target) or not math.isfinite(value):missing.append(key+'|nonfinite');continue
            diffs.append(abs(value-target))
    maximum=max(diffs) if diffs else None;threshold=1e-6
    if row['path']=='deploy':threshold=1e-3 if row['adapter'].endswith('F32') else 1e-5
    passed=not missing and bool(diffs) and len(output)==(600 if index_kind=='bench' else 50) and maximum<threshold
    return dict(prediction_pass=bool(passed),prediction_max_abs=maximum,prediction_threshold=threshold,prediction_reference=source['path'],prediction_reference_device=reference_device,prediction_reference_approval=str(reference_approval) if reference_approval else None,prediction_missing=len(missing),prediction_failure='' if passed else 'actualstrictnumericalqualificationfailed; no tolerancechange')
