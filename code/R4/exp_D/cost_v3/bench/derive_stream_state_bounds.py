"""Exact retained-payload phase bounds. No audio read, model forward, timer or benchmark."""
from pathlib import Path
import csv,json,math,datetime,sys
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parents[1];S=P.parents[1];D=S/'exp_D2';sys.path.insert(0,str(P))
from deploy_v2.candidate1_stream import Backend
from lean_raw_numpy_minimal_v5 import FIR
R=P/'results'
metadata=S/'exp_A/results_v2/A0_TRAIN_frontend_metadata.csv'
with metadata.open(encoding='utf-8-sig',newline='') as f:headers=list(csv.DictReader(f))
observed=dict(Counter(int(r['native_sr']) for r in headers))
max_source_frames=max(int(r['native_frames']) for r in headers)
with np.load(D/'lean_raw_v5/exact_causal_FIR_values.npz') as z:cache={k:(len(z[k]),str(z[k].dtype)) for k in z.files}
# Actual factory cache defines available rates, independently of codec bandwidth labels.
supported=sorted({int(k.split('_')[1]) for k in cache if f'h_{k.split("_")[1]}_16000' in cache and f'h_{k.split("_")[1]}_48000' in cache})
rates=sorted(set([8000,16000,32000,48000])|set(observed)|set(supported))
def retained_outputs(q,window,hop):
    return q if q<window else q-((q-window)//hop+1)*hop

def fir_state(n,sr,target,closed):
    fir=FIR(sr,target);up,down=fir.up,fir.down;total=(n*up+down-1)//down
    q=total if closed or up==down else max(0,total-fir.remove)
    if up==down:keep=0;start=n
    elif q==0:keep=n;start=0
    else:
        oldest=max(0,-(-((q+fir.remove)*down-(len(fir.h)-1))//up))
        start=min(n,oldest);keep=n-start
    return dict(output_count=q,history_samples=keep,up=up,down=down,remove=fir.remove,coefficient_length=len(fir.h),coefficient_dtype=str(fir.h.dtype),next_polyphase=((q+fir.remove)*down)%up,start=start)

models=[];components=[];bounds=[];witnesses=[];array_evidence=[]
for method in ['v2_pure','v2_64']:
    b=Backend(method,'float32');probe=b.new_stream(48000)
    initial_num_dtype=str(probe.num.dtype)
    # Exact source scalar-update dtype: np.exp(float score) is an np.float64 scalar.
    # This is a zero-valued tiny array dtype read, not a frame/GRU/attention forward.
    probe.num=probe.num*np.exp(0.)+np.exp(0.)*np.zeros(b.width,np.float32)
    active_num_dtype=str(probe.num.dtype);hbytes=probe.h.nbytes;numbytes=probe.num.nbytes
    fixed=hbytes+numbytes+13*8+2*7*8
    models.append(dict(method=method,layers=b.layers,width=b.width,nfft=b.nfft,main_hop=b.hop,hidden_dtype=str(probe.h.dtype),hidden_bytes=hbytes,initial_numerator_dtype=initial_num_dtype,active_numerator_dtype=active_num_dtype,numerator_bytes=numbytes,logical_counter_slots=27,logical_counter_bytes=216))
    for sr in rates:
        if sr not in supported:
            bounds.append(dict(method=method,input_sr=sr,supported=False,observed_header_rows=observed.get(sr,0),status='UNSUPPORTED_BY_UNCHANGED_FROZEN_FIR_CACHE',state_bytes_upper=None,post_push_upper=None,post_finish_upper=None,startup_upper=None,steady_post_push_upper=None,componentwise_envelope_bytes=None,source_sample_period=None,enumerated_N_max=None,worst_mode='',worst_N=None,worst_input_phase=None,worst_main_residual=None,worst_side_residual=None,worst_main_history=None,worst_side_history=None,state_definition='same existing state_bytes retainedarrays +27 logical64bit slots, excludesweights/scratch/inputblock/PythonobjectRSS'))
            continue
        main=FIR(sr,b.sr);side=FIR(sr,48000)
        threshold_main=((b.nfft+main.remove-1)*main.down)//main.up+1
        threshold_side=((960+side.remove-1)*side.down)//side.up+1
        steady_start=max(threshold_main,threshold_side)
        # Input period that adds integer hops to both output streams, and repeats FIR up/down phases.
        period=math.lcm((b.hop*main.down)//math.gcd(main.up,b.hop*main.down),(480*side.down)//math.gcd(side.up,480*side.down))
        states=[]
        for n in range(steady_start+2*period+1):
            for closed in [False,True]:
                fm=fir_state(n,sr,b.sr,closed);fs=fir_state(n,sr,48000,closed)
                qm,qs=fm['output_count'],fs['output_count'];mb=retained_outputs(qm,b.nfft,b.hop);sb=retained_outputs(qs,960,480)
                has_main_frame=qm>=b.nfft
                if closed and not has_main_frame:mb=b.nfft-b.hop;has_main_frame=True
                if closed and qs>0 and qs<960:sb=480
                nb=numbytes if has_main_frame else b.width*4
                parts=dict(GRU_hidden=hbytes,attention_numerator=nb,attention_maximum_denominator=2*8,mel_residual=mb*4,side_RMS_residual=sb*4,main_FIR_history=fm['history_samples']*4,side_FIR_history=fs['history_samples']*4,main_FIR_logical_registers=7*8,side_FIR_logical_registers=7*8,other_stream_logical_registers=11*8)
                states.append(dict(N=n,mode='post_finish' if closed else'post_push',input_phase=n%period,main_residual=mb,side_residual=sb,main_history=fm['history_samples'],side_history=fs['history_samples'],main_next_polyphase=fm['next_polyphase'],side_next_polyphase=fs['next_polyphase'],state_bytes=sum(parts.values()),parts=parts))
        worst=max(states,key=lambda s:s['state_bytes']);push=max(s['state_bytes'] for s in states if s['mode']=='post_push');finish=max(s['state_bytes'] for s in states if s['mode']=='post_finish');startup=max(s['state_bytes'] for s in states if s['N']<steady_start);steady=max(s['state_bytes'] for s in states if s['N']>=steady_start and s['mode']=='post_push')
        marginal={k:max(s['parts'][k] for s in states) for k in worst['parts']}
        bounds.append(dict(method=method,input_sr=sr,supported=True,observed_header_rows=observed.get(sr,0),status='ACTUAL_SOURCE_EXACT_PHASE_BOUND',state_bytes_upper=worst['state_bytes'],post_push_upper=push,post_finish_upper=finish,startup_upper=startup,steady_post_push_upper=steady,componentwise_envelope_bytes=sum(marginal.values()),source_sample_period=period,enumerated_N_max=steady_start+2*period,worst_mode=worst['mode'],worst_N=worst['N'],worst_input_phase=worst['input_phase'],worst_main_residual=worst['main_residual'],worst_side_residual=worst['side_residual'],worst_main_history=worst['main_history'],worst_side_history=worst['side_history'],state_definition='same existing state_bytes retainedarrays +27 logical64bit slots, excludesweights/scratch/inputblock/PythonobjectRSS'))
        dtypes={'GRU_hidden':str(probe.h.dtype),'attention_numerator':active_num_dtype,'attention_maximum_denominator':'float64 / logical64','mel_residual':'float32','side_RMS_residual':'float32','main_FIR_history':'float32','side_FIR_history':'float32','main_FIR_logical_registers':'7 logical64 slots (Python scalar storage separate)','side_FIR_logical_registers':'7 logical64 slots (Python scalar storage separate)','other_stream_logical_registers':'11 logical64 slots (Python scalar storage separate)'}
        for component,size in marginal.items():
            itemsize=4 if dtypes[component]=='float32' else 8
            components.append(dict(method=method,input_sr=sr,component=component,dtype=dtypes[component],itemsize_bytes=itemsize,length_upper=size//itemsize,component_bytes_upper=size,bytes_at_joint_max=worst['parts'][component],sum_of_marginal_maxima_is_not_joint_max=True))
        witnesses.append(dict(method=method,input_sr=sr,joint_max=worst,post_push_witness=max((s for s in states if s['mode']=='post_push'),key=lambda s:s['state_bytes']),post_finish_witness=max((s for s in states if s['mode']=='post_finish'),key=lambda s:s['state_bytes']),phase_period=period,steady_start=steady_start,all_startup_N_enumerated=True,all_steady_residue_classes_enumerated=True,induction='For N>=steady_start, adding period adds exactinteger main/side hops; residuals/FIRhistory/polyphase/dtypes repeat, so no later larger retainedarray state',main_filter=dict(up=main.up,down=main.down,remove=main.remove,length=len(main.h)),side_filter=dict(up=side.up,down=side.down,remove=side.remove,length=len(side.h))))
        # Bounded actual FIR-array lengths at startup and extrema, without calling Stream.push/finish or any NN.
        ns=sorted(set([0,1,max(0,steady_start-1),steady_start,worst['N']]))
        for n in ns:
            for target in [b.sr,48000]:
                f=FIR(sr,target);f.push(np.zeros(n,np.float32));a=fir_state(n,sr,target,False)
                if len(f.buffer)!=a['history_samples'] or f.index!=a['output_count']:raise RuntimeError('actual FIR arrays disagree with integer-state derivation')
                f.finish();z=fir_state(n,sr,target,True)
                if len(f.buffer)!=z['history_samples'] or f.index!=z['output_count']:raise RuntimeError('actual closedFIR arrays disagree with integer-state derivation')
        # Instantiate payload arrays at analytic witness, then read original state_bytes(), never forward.
        s=b.new_stream(sr);s.num=np.zeros(b.width,np.float64);s.buffer=np.zeros(worst['main_residual'],np.float32);s.sidebuffer=np.zeros(worst['side_residual'],np.float32);s.main.buffer=np.zeros(worst['main_history'],np.float32);s.sidefir.buffer=np.zeros(worst['side_history'],np.float32)
        if s.state_bytes()!=worst['state_bytes']:raise RuntimeError('actual source state_bytes mismatch')
        array_evidence.append(dict(method=method,input_sr=sr,actual_state_bytes_at_witness=s.state_bytes(),actual_hidden_dtype=str(s.h.dtype),actual_attention_numerator_dtype=str(s.num.dtype),FIR_array_length_reads=2*len(ns),NN_forward_calls=0,Stream_push_calls=0,audio_files_opened=0))

def write_csv(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv(R/'stream_state_bounds_by_samplerate.csv',bounds);write_csv(R/'stream_state_bound_components.csv',components)
historical=[]
for m in ['v2_pure','v2_64']:
    q=json.loads((P/'deploy_v2/candidate1/qualification'/(m+'_float32')/'COMPLETE.json').read_text(encoding='utf-8-sig'));historical.append(dict(method=m,observed_state_bytes=q['state_bytes_max'],scope='samefixedTRAIN200 threechunks observedmax, not universalbound',source=str(P/'deploy_v2/candidate1/qualification'/(m+'_float32')/'COMPLETE.json')))
    q=json.loads((R/'streaming'/(m+'_20_timing.json')).read_text(encoding='utf-8-sig'));historical.append(dict(method=m,observed_state_bytes=q['state_bytes_max'],scope='single48kfile cyclic1000blocks20ms observedmax, phase0 afterstartup, not universalbound',source=str(R/'streaming'/(m+'_20_timing.json'))))
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_STATE_BOUNDS_COMPLETE',method='integer-source-count enumeration of allstartup N, alljointresidual/FIRpolyphase states and postfinish states; exact original state_bytes convention; bounded actual zero-array length reads only',state_definition='retainedarrays +27 logical64bit slots (13 Stream +7 perFIR); excludesimmutableweights/coefficients, inputpacket, temporaryFFT/GRU scratch, Pythonobjectoverhead andRSS',actual_inventory=dict(source=str(metadata),header_rows=len(headers),rates=observed,highest_native_input_sr=max(observed),maximum_source_frames=max_source_frames,scope='completeexistingNISQA TRAIN header inventory; codec bandwidth is not WAV samplerate'),frozen_cache=dict(source=str(D/'lean_raw_v5/exact_causal_FIR_values.npz'),keys=cache,supported_native_input_rates=supported,unobserved_supported_rates=[x for x in supported if x not in observed],unsupported_requested_rates=[x for x in rates if x not in supported],changed=False),models=models,bounds=bounds,all_supported_rate_maxima={m:max(r['state_bytes_upper'] for r in bounds if r['method']==m and r['supported']) for m in ['v2_pure','v2_64']},actual_NISQA_inventory_rate_maxima={m:max(r['state_bytes_upper'] for r in bounds if r['method']==m and r['input_sr'] in observed and r['supported']) for m in ['v2_pure','v2_64']},witnesses=witnesses,array_evidence=array_evidence,historical_observations_preserved=historical,phase0_48k_explanation='fixed10/20/100ms packets add480/960/4800 source samples; sameNmod480 zero state gives5160/4648. Lastpartialpacket can enter otherphases. 48k sideFIR identity haszerohistory but sideRMS residual is stillretained.',counter_scope='Existing logical64bit slots exactly match state_bytes; actual Python int object memory can grow with streamlength and is not an RSS bound. All actual metadata samplecounts fit64bits.',new_audio_inference=0,NN_forward_calls=0,model_weights_changed=False,frontend_changed=False,new_timing=0,cost_benchmark=False,outputs=dict(bounds=str(R/'stream_state_bounds_by_samplerate.csv'),components=str(R/'stream_state_bound_components.csv')))
(R/'STATE_BOUNDS_ACTUAL_COMPLETE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(status=receipt['status'],maxima=receipt['all_supported_rate_maxima'],bounds=bounds,inventory=receipt['actual_inventory'],array_evidence=array_evidence),ensure_ascii=False))