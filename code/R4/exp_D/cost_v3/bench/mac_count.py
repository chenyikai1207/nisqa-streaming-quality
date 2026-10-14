from public_paths import data_path, package_path
"""Unified VAL-mean analytical MAC equivalents, not measuredRTF.
FFT=(5/2)Nlog2N FLOP /2; window actualN; densemel M*(N/2+1); log/normalization1 per element.
GRU3H(I+H) eachframe/layer, no bias; attention weightedpool andhead affine. Scalar level costs explicitproxy.
Resample actual coefficient length*outputs/polyphasecount; 16k main bypasszero; actual48kside conversion retained.
Tool-backed NN affine counts: frozen fvcore verified NISQA shape-polynomial; newONNX-tool actualshape;
SQUIMmeta shape hooks add omitted LSTM/MHA; no fvcore missingRNN silently treatedasfullmodel.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,math,datetime,io,contextlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];S=P.parents[1];D=S/'exp_D2';out=P/'results';out.mkdir(exist_ok=True)
val=list(csv.DictReader((S/'shared/val_timing_index.csv').open(encoding='utf8')));assert len(val)==2700;seconds=sum(float(r['duration_s']) for r in val)/2700;native_sr=48000;ns=math.ceil(seconds*native_sr);n16=math.ceil(ns*16000/native_sr);n48=ns;frames=max(1,(n16-320)//160+1);sideframes=max(1,(n48-960)//480+1);rows=[];inventory={};tool_details={}
def size(path):
    with Path(path).open('rb') as f:f.seek(0,2);return f.tell()
def add(method,component,value,source,note):rows.append(dict(method=method,component=component,mmac_per_s=float(value)/seconds/1e6 if value is not None else None,source=source,note=note))
def fir_cost(sr,target,n):
    if sr==target:return 0
    key='h_'+str(sr)+'_'+str(target)
    with np.load(D/'lean_raw_v5/exact_causal_FIR_values.npz') as a:h=a[key]
    up=target//math.gcd(sr,target);return len(h)*math.ceil(n*target/sr)/up
for model,width,layers,side in [('v1',32,1,0),('v2_96',96,2,4),('v2_64',64,2,4)]:
    for path in (['native'] if model=='v1' else ['native','deploy']):
        method=model+'_'+path;nn=sum(3*width*((48 if l==0 else width)+width) for l in range(layers))*frames
        add(method,'GRU_affine',nn,'analytic frozenweightdimensions','3H(I+H), layer/frame; bias/activation notMAC')
        add(method,'attention_affine_and_weighted_pool',2*width*frames,'analytic frozenmodel','attention dot plus weightedpool; softmax transcendental unpriced')
        add(method,'head_affine',5*(width+side),'analytic frozenmodel','oneutterance rawhead; fixedoutputmapping notcalled inregisteredrawcostpath')
        add(method,'main_realFFT_estimate',frames*1.25*320*math.log2(320),'analytic FFT convention','FFT estimate, notexacthardwareMAC')
        add(method,'main_window',frames*320,'analytic actualwindow','1MACeq perwindowmultiply peruserpolicy')
        add(method,'main_dense_mel',frames*48*161,'analytic densemel','dense counts; sparse notused')
        add(method,'main_power_proxy',frames*161*2.5,'analytic scalarproxy','complexabs+square5scalarops/bin ->2.5MACeq; not exacthardwareinstructioncount')
        add(method,'log_and_normalization',frames*48*3,'analytic policyproxy','log1, subtract1, divide1 per element; notNNbias')
        add(method,'main_resampler',fir_cost(native_sr,16000,ns),'analytic actualfrozenFIR coefficientlength','ceil outputs; coefficientlength/output/upfactor, native16k mainbypass0')
        if model=='v1':add(method,'peak_normalization_proxy',2*n16,'analytic scalarpolicy','wholeutterance max/absolute anddivision; originalv1noncausalpeak scope')
        else:
            add(method,'raw_level_and_active_level_proxy',4*ns+2*sideframes*960+sideframes+12,'analytic exactcalledleveldefinitions','raw RMSproduct/sum, clipping/nearzero comparisons; activeframe RMSandterminal scalar proxy')
            add(method,'fixedgain_and_side_normalization',frames*48+8,'analytic scalarpolicy','gain squared multiplication permel; four side subtract/divide')
            add(method,'side_resampler',fir_cost(native_sr,48000,ns),'analytic actualfrozenFIR','native48k bypass0')
            if path=='native':
                add(method,'native_unused_band_realFFT_estimate',sideframes*1.25*960*math.log2(960),'analytic FFT convention','actual frozenRunningSide computes this branch evenband0; nativeonly')
                add(method,'native_unused_band_window',sideframes*960,'analytic actualwindow','actualnative960point window')
                add(method,'native_unused_band_power_and_ratios_proxy',sideframes*(481*2.5+481+400+340+200+481+7)/2,'analytic scalarpolicy','calledpower/band cumulative sums andratios; scalar estimate, removedonlyinqualifieddeployment')
        label='v2_pure' if model=='v2_96' else'v2_64';folder=S/'exp_A/frozen/A_distill' if model=='v1' else S/'exp_A/frozen_v3'/label
        weights=[folder/'model.pt'] if path=='native' else [D/'lean_raw_v5'/(label+'_frozen_arrays.npz'),D/'lean_raw_v5/exact_causal_FIR_values.npz'];params={'v1':8070,'v2_96':98522,'v2_64':47258}[model]
        inventory[method]=dict(parameters=params,weights_bytes=sum(size(w) for w in weights),weight_paths=list(map(str,weights)),parameter_source='actualfrozen recordedscalarcounts; nn bias includedinparametercount butnotMAC',precision='nativefp32 orselecteddeployment; logicalMAC no bitwidthspeedweighting')
# Official NISQA tool profiles reused for unchangedweights at new actualVAL-mean frame/window count.
for method,name,params in [('NISQA_MOS_official','OfficialNISQA_MOS',218402),('NISQA5D_official','OfficialNISQA5D',252458)]:
    source=D/'D0_NISQA_TRAIN200_v3'/(name+'_shape_profiles.json');profile=json.loads(source.read_text());args=profile['author_args'];nf=1+ns//int(native_sr*args['ms_hop_length']);nw=math.ceil((nf-(args['ms_seg_length']-1))/args['ms_seg_hop_length']);nn=float(np.polyval(profile['affine_polynomial'],nw));norms=[p['operators'].get('batch_norm',0)/2+p['operators'].get('layer_norm',0)/5 for p in profile['profiles'][:3]];norm=float(np.polyval(np.polyfit([p['n_windows'] for p in profile['profiles'][:3]],norms,1),nw));add(method,'network_affine',nn,'fvcore verifiedfrozen shape polynomial '+str(source),'CNN/linear/attention N+N2; original400window verificationresidual='+str(profile['validation_residual']))
    add(method,'network_normalization_proxy',norm,'fvcore per-operator normalizedelement conversion','eval batch_norm tool2ops/element and layer_norm5ops/element individually converted to policy1; arithmeticproxy not hardwareMAC')
    fft=args['ms_n_fft'];add(method,'author_realFFT_estimate',nf*1.25*fft*math.log2(fft),'analytic frozenauthorSTFT','center=True frames1+floor(samples/hop)')
    add(method,'author_padded_window',nf*fft,'analytic librosaactualpaddedwindow','actual4096element multiply includingzeropadded960 Hann')
    add(method,'author_dense_mel',nf*args['ms_n_mels']*(fft//2+1),'analytic densemel','frozen48x2049')
    add(method,'author_log_power_proxy',nf*(args['ms_n_mels']+2.5*(fft//2+1)),'analytic scalarpolicy','powerabs-square andauthor log per-element proxy; noframegainchange')
    w=Path(str(data_path('third_party/LANE_C_MODELS/OFFICIAL_NISQA/weights/nisqa.tar'))) if name=='OfficialNISQA5D' else S/'exp_C/weights/nisqa_mos_only.tar';inventory[method]=dict(parameters=params,weights_bytes=size(w),weight_paths=[str(w)],parameter_source='actualfrozen officialtrainednn counts')
# Fresh ONNX tensor-shape counts at representativeVAL-mean length, no numericNN inference.
import onnx_tool,onnx
for method,name in [('PLCMOS_official','PLCMOS'),('DNSMOS_official','DNSMOS')]:
    weights=S/'exp_B/official'/('PLC_WEIGHT.onnx' if name=='PLCMOS' else 'DNSMOS_WEIGHT.onnx');n=n16
    if name=='PLCMOS':
        nf=math.ceil((n+256)/256);inputs={'degraded_audio':np.zeros((1,1,nf,257),np.float32),'rater_embed':np.zeros((1,64),np.float32)};calls=15;add(method,'external_STFT_FFT_estimate',nf*1.25*512*math.log2(512),'analytic authorstana synthframe formula','ceil((n16+256)/256); author run computes degraded_STFT once before15 rater fullnetworkcalls, same officialadapter; noencodercache')
        add(method,'external_STFT_window_power_log_proxy',nf*(512+257*2.5+257),'analytic authorSTFT/scalarpolicy','one degraded_STFT peraudio outside15 raterloop; window1/magnitude/power/log proxy; no mel; NN15fullcalls retained')
    else:
        tiled=n
        while tiled<144160:tiled*=2
        calls=int(math.floor(tiled/16000)-9.01)+1;inputs={'input_1':np.zeros((1,144160),np.float32)}
    with contextlib.redirect_stdout(io.StringIO()):g=onnx_tool.Model(str(weights),{'verbose':False}).graph;g.graph_reorder_nodes();g.shape_infer(inputs);g.profile()
    affine=0;norm=0;detail=[];roles=set()
    for node in g.nodemap.values():
        ins=[g.tensormap[x].get_shape() for x in node.input];outs=[g.tensormap[x].get_shape() for x in node.output];op=node.op_type;cost=0
        if op=='Conv':cost=int(np.prod(outs[0]))*int(np.prod(ins[1][1:]));roles.update(node.input[1:3])
        elif op in ['MatMul','Gemm']:
            reduction=ins[1][-1] if op=='Gemm' and getattr(node,'transB',0)>0 else ins[1][-2] if len(ins[1])>1 else 1;cost=int(np.prod(outs[0]))*reduction;roles.update(node.input[1:3])
        elif op in ['GRU','LSTM']:cost=(int(np.prod(ins[1]))+int(np.prod(ins[2])))*ins[0][0]*ins[0][1];roles.update(node.input[1:4])
        if 'Norm' in op:norm+=int(np.prod(outs[0]))
        affine+=cost;detail.append(dict(node=node.name,op=op,input_shapes=ins,output_shapes=outs,affine_MAC=cost))
    add(method,'network_affine_including_fixed_frontend',affine*calls,'onnx-tool actualnewshape','DNSfixedSTFTconv included; calls='+str(calls)+'; PLC15fullnetworkcalls noencoderreuse')
    add(method,'network_normalization_proxy',norm*calls,'onnx-tool actualnewshape','oneequivalent pernormalizedelement; biases/nonlinear activation unpriced')
    # Actual torchaudio kernel geometry follows frozenofficialresample; densezerotaps counted asupperbound.
    import torch
    from torchaudio.functional.functional import _get_sinc_resample_kernel
    k,width=_get_sinc_resample_kernel(native_sr,16000,math.gcd(native_sr,16000),6,.99,'sinc_interp_hann',None,torch.device('cpu'),torch.float32);positions=((ns+2*width+native_sr//math.gcd(native_sr,16000)-k.shape[-1])//(native_sr//math.gcd(native_sr,16000))+1);add(method,'official_resampler_dense_upperbound',positions*(16000//math.gcd(native_sr,16000))*k.shape[-1],'actualtorchaudio kernelgeometry','denseconv includeszerotaps; upperbound separatefromexactNN affine')
    om=onnx.load(str(weights));params=sum(int(np.prod(t.dims)) for t in om.graph.initializer if t.name in roles);params-=103040 if name=='DNSMOS' else 0;inventory[method]=dict(parameters=params,weights_bytes=size(weights),weight_paths=[str(weights)],parameter_source='unique learnedaffine initializerroles; DNSfixedSTFT103040 excludedfromparams includedMAC');tool_details[method]=detail
# SQUIM exact module-shape affine including omittedLSTM/MHA; META propagation, no numericforward.
import torch,torchaudio
sys.path.insert(0,str(D));torch.hub.set_dir(str(D/'torch_hub'));model=torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval().to('meta');counts={};mha_children={id(c) for v in model.modules() if isinstance(v,torch.nn.MultiheadAttention) for c in v.modules() if c is not v};handles=[]
def hook(v,ins,out):
    z=ins[0];cost=0;key=type(v).__name__
    if isinstance(v,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):cost=out.numel()*v.in_channels*np.prod(v.kernel_size)/v.groups
    elif isinstance(v,torch.nn.Linear):cost=out.numel()*v.in_features
    elif isinstance(v,(torch.nn.LSTM,torch.nn.GRU)):
        steps=int(z.batch_sizes.sum()) if isinstance(z,torch.nn.utils.rnn.PackedSequence) else int(np.prod(z.shape[:-1]));directions=2 if v.bidirectional else 1;gates=4 if isinstance(v,torch.nn.LSTM) else 3
        for layer in range(v.num_layers):cost+=steps*directions*gates*v.hidden_size*((v.input_size if layer==0 else directions*v.hidden_size)+v.hidden_size)
    elif isinstance(v,torch.nn.MultiheadAttention):
        q,k,x=ins[:3];batch=q.shape[0] if v.batch_first else q.shape[1];t=q.shape[1] if v.batch_first else q.shape[0];s=k.shape[1] if v.batch_first else k.shape[0];e=v.embed_dim;cost=batch*(2*t*e*e+2*s*e*e+2*t*s*e)
    elif isinstance(v,(torch.nn.LayerNorm,torch.nn.GroupNorm)):cost=out.numel()
    counts[key]=counts.get(key,0)+int(cost)
for v in model.modules():
    if id(v) not in mha_children and isinstance(v,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear,torch.nn.LSTM,torch.nn.GRU,torch.nn.MultiheadAttention,torch.nn.LayerNorm,torch.nn.GroupNorm)):handles.append(v.register_forward_hook(hook))
with torch.inference_mode():model(torch.empty((1,n16),device='meta'))
for h in handles:h.remove()
for key,value in counts.items():add('SQUIM_official','network_'+key,value,'PyTorchMETA actualshapes calibratedagainstfvcore frozenprofile','LSTM4gate/MHA affine explicitlycounted; norms1/element; allthree objectiveheads computed')
add('SQUIM_official','official_resampler_dense_upperbound',positions*(16000//math.gcd(native_sr,16000))*k.shape[-1],'actualtorchaudio kernelgeometry','same16k conversionasofficial; internallearnedfrontend isalreadyConv1d, noextraSTFT claim')
w=D/'torch_hub/torchaudio/models/squim_objective_dns2020.pth';inventory['SQUIM_official']=dict(parameters=sum(v.numel() for v in model.parameters()),weights_bytes=size(w),weight_paths=[str(w)],parameter_source='actualofficial nn.Parameters');tool_details['SQUIM_official']=counts
add('P563_reference','TOTAL',None,'ITU native C','No validated wholealgorithm operationcounter; unavailableMAC notzero; optionalalgorithm nottimed')
for method in sorted({r['method'] for r in rows if r['component']!='TOTAL'}):
    total=sum(r['mmac_per_s'] for r in rows if r['method']==method);rows.append(dict(method=method,component='TOTAL',mmac_per_s=total,source='sum ofexplicitcomponents',note='VALmean='+str(seconds)+'s; outputlengthceil,centerFalseframefloor/nativeNIcenterTrue; arithmeticMACeqestimates+FFT/FIRupperbounds, nothardwareexact orRTF'))
with (out/'mac_table.csv').open('w',newline='',encoding='utf8') as f:wri=csv.DictWriter(f,fieldnames=['method','component','mmac_per_s','source','note']);wri.writeheader();wri.writerows(rows)
(out/'model_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf8');(out/'MAC_TOOL_DETAILS.json').write_text(json.dumps(tool_details,indent=2),encoding='utf8')
differences=[dict(old_value='v1 2.9020',source='D2_MAC_summary_with_attention_pool.csv',cause='oldTRAIN200 durationweight/window-log-normalization scalar pricing; newVALmean full frontend policy nowexplicit'),dict(old_value='v1 2.9563',source='v1_v15_sameTRAIN200_totals_v5.csv',cause='sameTRAIN200 componentmatched scalar/bias/window proxies differ fromoldregistered2.9020; newtable excludesGRUbias andunifiesmeanframepolicy'),dict(old_value='v2 13.1444',source='ROOT/A_CAPACITY_COST_v2.csv historicalscreen',cause='pre-freeze screenestimate, frame/side/scalarcoverage; replacedbynewactualfrozenrecipecomponenttable'),dict(old_value='v2 13.3611',source='A7_D200_COST_v3/v2_pure/COMPLETE.json',cause='actualTRAIN200 incremental frame/FIR andscalarproxy scope; newVALmean+windowN/log/normalization policy andNNbias exclusion'),dict(old_value='v2 13.3683',source='v1_v15_sameTRAIN200_totals_v5.csv',cause='sameTRAIN200 matched fullnative components slightscalarpricingdifferent fromincremental; nowunifiednewtable')]
with (out/'MAC_historical_differences.csv').open('w',newline='',encoding='utf8') as f:wri=csv.DictWriter(f,fieldnames=list(differences[0]));wri.writeheader();wri.writerows(differences)
(out/'MAC_ACTUAL_COMPLETE.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_UNIFIED_VAL_MEAN_MAC_ESTIMATES_COMPLETE',VAL_n=2700,VAL_mean_audio_s=seconds,native_sr=native_sr,main_frames=frames,side_frames=sideframes,table=str(out/'mac_table.csv'),no_numeric_quality_forward=True,parameter_inventory=str(out/'model_inventory.json'),P563_MAC_unavailable=True),indent=2),encoding='utf8')
print('ACTUAL_UNIFIED_MAC_COMPLETE',seconds)
