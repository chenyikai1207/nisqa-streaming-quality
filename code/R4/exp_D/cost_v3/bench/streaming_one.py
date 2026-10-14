"""Selected minimal NumPy stream: timing and resident modes use separateprocesses, noRSSsampling in timing."""
import os,sys,argparse,json,time,csv,threading,importlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['v2_pure','v2_64'],required=True);ap.add_argument('--mode',choices=['timing','resident'],required=True);ap.add_argument('--chunk-ms',type=int,choices=[10,20,100],required=True);ap.add_argument('--clean-marker',required=True);ap.add_argument('--out',required=True);ap.add_argument('--selection');a=ap.parse_args();P=Path(__file__).resolve().parents[1];gate=json.loads(Path(a.clean_marker).read_text(encoding='utf8'));assert gate.get('formal_allowed') is True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(P));import psutil;cores=json.loads((P/'env/cores.json').read_text(encoding='utf8'));core=cores['S3']['core_id'];proc=psutil.Process();proc.cpu_affinity([core]);import numpy as np,soundfile as sf
selection_path=Path(a.selection) if a.selection else P/'deploy_v2/DEPLOYMENT_SELECTION.json';sel=json.loads(selection_path.read_text(encoding='utf8'));method='v2_96' if a.candidate=='v2_pure' else'v2_64';entry=sel['selected_adapter_by_model'][method];assert entry.endswith(('F32','F64')) and sel['actual_TRAIN200_pass'];precision='float32' if entry.endswith('F32') else'float64';backend_module=sel.get('stream_backend','deploy_v2.numpy_backend');backend_class=sel.get('stream_backend_class','Backend');
if sel.get('implementation_version')=='candidate1':
    qualification=json.loads(Path(sel['stream_qualification'][method]).read_text(encoding='utf8'));assert qualification['passed'] and qualification['n_files']==200 and qualification['chunk_ms']==[10,20,100] and qualification['stream_backend']==backend_module+'.'+backend_class
Backend=getattr(importlib.import_module(backend_module),backend_class);b=Backend(a.candidate,precision);path=next(x.strip() for x in open(P/'lists/equiv_TRAIN200_files.txt',encoding='utf8') if x.strip());reader=sf.SoundFile(path);sr=reader.samplerate;n=round(sr*a.chunk_ms/1000);s=b.new_stream(sr);baseline=proc.memory_info().rss;rss=[];stop=threading.Event();thread=None
if a.mode=='resident':
    def sample():
        while not stop.wait(.01):rss.append(proc.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
def packet():
    parts=[];left=n
    while left:
        v=reader.read(left,dtype='float32',always_2d=True)
        if not len(v):reader.seek(0);continue
        parts.append(v.mean(1));left-=len(v)
    return np.concatenate(parts)
latencies=[];first=None;maxstate=0;components_at_max={};start=time.time()
for k in range(1000):
    x=packet();t=time.perf_counter_ns();s.push(x)
    if s.frames:
        s.predict()
        if first is None:first=(k+1)*a.chunk_ms
    ns=time.perf_counter_ns()-t
    if a.mode=='timing':latencies.append(ns/1e6)
    parts=dict(GRU_hidden=s.h.nbytes,attention_maximum=8,attention_denominator=8,attention_numerator=s.num.nbytes,mel_residual=s.buffer.nbytes,side_frame_residual=s.sidebuffer.nbytes,main_resampler_history=s.main.buffer.nbytes,main_resampler_logical_counters=7*8,side_resampler_history=s.sidefir.buffer.nbytes,side_resampler_logical_counters=7*8,level_and_stream_logical_counters=11*8)
    total=sum(parts.values());assert total==s.state_bytes()
    if total>maxstate:maxstate=total;components_at_max=parts
reader.close();stop.set()
if thread:thread.join()
end_rss=proc.memory_info().rss;r=dict(candidate=a.candidate,precision=precision,implementation_version=sel.get('implementation_version','numpy_original'),stream_backend=backend_module+'.'+backend_class,whole_adapter_selected=entry,selection_source=str(selection_path),whole_process_RSS_inferred_from_stream=False,mode=a.mode,chunk_ms=a.chunk_ms,core_id=core,actual_blocks=1000,source_audio=str(path),source_native_sr=sr,upstream_decode='outside timedkernel; SoundFile packetreader cyclicfills atEOF, no fullaudiohistory retained',state_bytes_max=maxstate,state_components_bytes=components_at_max,state_definition='retainedarraypayload+logical64bitscalarregisters; frozenweights/scratch/Pythonobjects not counted asstate; separate processRSS includes Pythonobjects',resampler_delay_ms=max(s.main.delay,s.sidefir.delay)*1000,stft_future_lookahead_ms=0,window_accumulation_ms=20,algorithm_delay_ms=a.chunk_ms+max(s.main.delay,s.sidefir.delay)*1000,first_prediction_input_ms=first,delay_definition='packet duration +actualresampler latency +zero futurelookahead (centerFalse pastwindow); windowaccumulation/actualstartup listed separately',baseline_rss_mb=baseline/1e6,ending_rss_mb=end_rss/1e6,torch_imported='torch' in sys.modules,ort_imported='onnxruntime' in sys.modules,t_start=start,t_end=time.time(),quality_metrics=False)
if a.mode=='timing':r.update(block_ms_median=float(np.median(latencies)),block_ms_p99=float(np.quantile(latencies,.99)),block_RTF_median=float(np.median(latencies)/a.chunk_ms),block_RTF_p99=float(np.quantile(latencies,.99)/a.chunk_ms),rss_sampling_in_timing=False,latencies_ms=latencies)
else:r.update(resident_rss_mb_median=float(np.median(rss+[end_rss])/1e6),resident_rss_mb_p99=float(np.quantile(rss+[end_rss],.99)/1e6),peak_rss_mb=max(rss+[baseline,end_rss])/1e6,sampling_interval_ms=10,latency_measured=False)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf8')