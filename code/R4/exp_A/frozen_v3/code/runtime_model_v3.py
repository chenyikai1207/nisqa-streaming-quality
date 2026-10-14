"""A7 exported quality model: offline raw prediction, training mapping and finite streaming state."""
import json,pickle,importlib,sys
from pathlib import Path
import numpy as np,torch,soundfile as sf
from frontend_v3 import Frontend,FrontendConfig,RunningSide,resample
from models_post_v3 import PostModel,PostIncrementalEncoder
from stream_time_features_v3 import StreamingTime
from causal_resampler_v3 import CausalResampler
from mapping_v3 import apply_mapping
from qualified_b2_v3 import b2_precision,inference_precision
def module_entry(entry):
    if entry.get('module_dir') and entry['module_dir'] not in sys.path:sys.path.insert(0,entry['module_dir'])
    return getattr(importlib.import_module(entry['module']),entry['callable'])
class Predictor:
    def __init__(self,folder,device='cpu'):
        self.folder=Path(folder);self.spec=json.loads((self.folder/'spec.json').read_text());self.device=device;map_path=self.folder/('NN_input_mapping.json' if self.spec.get('type')=='HGB49' else 'mapping.json');self.mapping=json.loads(map_path.read_text())['maps'];s=self.spec['recipe'];a=s['architecture'];kw={k:a[k] for k in ['kind','width','layers','kernel','input_channels']};self.model=PostModel(**kw,base_side_dim=s['frontend']['side_dim'],fusion=s.get('fusion'),auxiliary=False).to(device).eval();ck=torch.load(self.folder/'model.pt',map_location=device,weights_only=False);self.model.load_state_dict(ck['state_dict']);self.mean=ck['mean'].to(device);self.std=ck['std'].to(device);self.sm=ck['side_mean'].to(device);self.ss=ck['side_std'].to(device);self.gain=float(ck['frontend_gain']);cfg=s['frontend'];self.front=Frontend(FrontendConfig(cfg['sr'],cfg['n_mels'],'global_gain',cfg['bandwidth_side']),device,global_gain=self.gain)
        self.hgb=None
        if self.spec.get('type')=='HGB49':
            with (self.folder/'HGB_models.pkl').open('rb') as f:self.hgb=pickle.load(f)
    def estimates_file(self,path):
        h=self.spec['B2']
        with b2_precision():z=module_entry(h['synthetic_predict_entry'])([str(path)],weights_path=h['weights_path'])
        b=np.asarray(z,np.float32).reshape(1,5)[0];b=b.copy()
        if not np.isfinite(b[1]) or b[4]<.5:b[1]=0.
        return b
    def normalize_side(self,side):
        s=torch.tensor(np.asarray(side),device=self.device,dtype=torch.float32).reshape(1,-1);return (torch.where(torch.isfinite(s),s,self.sm)-self.sm)/self.ss
    def map(self,p):return np.column_stack([apply_mapping(np.asarray(p)[:,j],self.mapping[j]) for j in range(5)])
    @torch.inference_mode()
    def predict_file(self,path,mapped=False):
        w,sr=sf.read(path,dtype='float32');return self.predict_array(w,sr,mapped=mapped,B2_estimates=self.estimates_file(path) if self.spec['recipe'].get('fusion') else None)
    @torch.inference_mode()
    @inference_precision()
    def predict_array(self,w,sr,mapped=False,B2_estimates=None):
        w=np.asarray(w,np.float32);w=w.mean(1) if w.ndim==2 else w;f=self.front.extract_features(w,sr);side=f['side'];time=None
        if self.spec['recipe'].get('fusion') or self.hgb is not None:
            s=StreamingTime(self.gain,self.device);s.update_native(w,sr);s.push16k(resample(w,sr,16000));time=s.finish()
        if self.spec['recipe'].get('fusion'):
            if B2_estimates is None:raise ValueError('Qualified B2 estimates are required for fused offline array prediction')
            side=np.r_[side,time['oldT'],time['newT'],B2_estimates]
        x=torch.tensor(f['logmel'],device=self.device)[None];m=torch.ones(x.shape[:2],device=self.device,dtype=torch.bool);raw=self.model((x-self.mean)/self.std,m,self.normalize_side(side)).cpu().numpy()
        if self.hgb is not None:
            features=np.r_[time['G0'],time['oldT'],time['newT'],self.map(raw)[0]];return np.array([[model.predict(features[None])[0] for model in self.hgb]])
        return self.map(raw) if mapped else raw
    def new_stream(self,native_sr):return Stream(self,native_sr)
class Stream:
    """Outputs only after completed causal FIR/window frames; no retained utterance waveform."""
    def __init__(self,p,native_sr):
        self.p=p;self.native_sr=int(native_sr);needs_time=bool(p.spec['recipe'].get('fusion') or p.hgb is not None);rates={p.front.config.sr,48000}|({16000} if needs_time else set());self.resamplers={rate:CausalResampler(native_sr,rate,p.device) for rate in rates};self.main=self.resamplers[p.front.config.sr];self.raw16=self.resamplers.get(16000);self.raw48=self.resamplers[48000];self.side=RunningSide(p.device);self.time=StreamingTime(p.gain,p.device) if needs_time else None;self.encoder=PostIncrementalEncoder(p.model);self.buffer=np.empty(0,np.float32);self.frames=0;self.closed=False;self.detector=None
        if p.spec['recipe'].get('fusion'):
            h=p.spec['B2']
            with b2_precision():self.detector=module_entry(h['stream_entry'])(weights_path=h['weights_path'],native_sr=native_sr,device=p.device)
    @torch.inference_mode()
    @inference_precision()
    def push_main(self,x):
        self.buffer=np.r_[self.buffer,x];f=self.p.front
        if len(self.buffer)<f.n_fft:return
        n=(len(self.buffer)-f.n_fft)//f.hop+1;frames=np.lib.stride_tricks.sliding_window_view(self.buffer,f.n_fft)[::f.hop][:n];t=torch.tensor(frames.copy(),device=self.p.device);power=torch.fft.rfft(t*f.window,dim=-1).abs().square();mel=((power@f.mel.T)*self.p.gain**2).clamp_min(1e-10).log();self.encoder.push_frames((mel-self.p.mean)/self.p.std);self.frames+=n;self.buffer=self.buffer[n*f.hop:].copy()
    def push(self,w):
        assert not self.closed;w=np.asarray(w,np.float32);w=w.mean(1) if w.ndim==2 else w;self.side.update_samples(w);outputs={rate:r.push(w) for rate,r in self.resamplers.items()};self.push_main(outputs[self.p.front.config.sr]);self.side.push_resampled48(outputs[48000])
        if self.time is not None:self.time.update_native(w,self.native_sr);self.time.push16k(outputs[16000])
        if self.detector is not None:
            with b2_precision():self.detector.push(w)
    def finish(self):
        if not self.closed:
            outputs={rate:r.finish() for rate,r in self.resamplers.items()};self.push_main(outputs[self.p.front.config.sr]);self.side.push_resampled48(outputs[48000])
            if self.time is not None:self.time.push16k(outputs[16000])
            if not self.frames:self.push_main(np.zeros(max(0,self.p.front.n_fft-len(self.buffer)),np.float32))
            if self.time is not None:self.time.finish()
            self.side.finish_resampled48()
            if self.detector is not None:
                with b2_precision():self.detector.finish()
            self.closed=True
        return self.predict()
    @torch.inference_mode()
    @inference_precision()
    def predict(self,mapped=False):
        if not self.frames:raise ValueError('No completed20ms quality frame')
        v=self.time.value() if self.time is not None else None;cfg=self.p.front.config;s=self.side.value(True,cfg.bandwidth_side)
        if self.detector is not None:
            with b2_precision():b=np.asarray(self.detector.predict(),np.float32).reshape(5).copy()
            if not np.isfinite(b[1]) or b[4]<.5:b[1]=0.
            s=np.r_[s,v['oldT'],v['newT'],b]
        raw=self.encoder.predict(self.p.normalize_side(s)).cpu().numpy()
        if self.p.hgb is not None:
            features=np.r_[v['G0'],v['oldT'],v['newT'],self.p.map(raw)[0]];return np.array([[m.predict(features[None])[0] for m in self.p.hgb]])
        return self.p.map(raw) if mapped else raw
    def state_bytes(self):
        detector=0 if self.detector is None else self.detector.state_bytes();time_state=0 if self.time is None else self.time.state_bytes();return self.encoder.state_bytes()+self.side.state_bytes()+time_state+self.buffer.nbytes+sum(r.state_bytes() for r in self.resamplers.values())+detector
    def state_summary(self):return dict(bytes=self.state_bytes(),quality_frames=self.frames,resampler_latency_s=max(r.latency_s for r in self.resamplers.values()),buffer_samples=len(self.buffer),whole_waveform_retained=False,closed=self.closed,raw_quality_predict=True,mapped_quality_predict_available=True)
