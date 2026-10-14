"""Main model v2 frontend, execution v3: fixed TRAIN gain and finite running side states."""
import math
from dataclasses import dataclass,asdict
import numpy as np,torch,soundfile as sf
from scipy.signal import resample_poly
from audit_frontend_v3 import ROOT
from qualified_b2_v3 import inference_precision
LEVEL_COLUMNS=['running_rms_dbfs','active_frame_level_dbfs','clipping_fraction','nearzero_fraction']
BAND_COLUMNS=['mean_energy_above4k_ratio','mean_energy_above7k_ratio','mean_energy_above14k_ratio','mean_rolloff95_hz']
@dataclass(frozen=True)
class FrontendConfig:
    sr:int=16000
    n_mels:int=48
    normalization:str='global_gain'
    bandwidth_side:bool=False
    fmin:float=50.
    window_ms:float=20.
    hop_ms:float=10.
    def as_dict(self):
        return dict(asdict(self),fmax=7600 if self.sr==16000 else 20000,n_fft=self.sr//50,hop_length=self.sr//100,center=False,log_floor=1e-10,window='periodic Hann',gain_fit='current TRAIN sample-weighted raw RMS -> -26dBFS',level_columns=LEVEL_COLUMNS if self.normalization=='global_gain' else [],band_columns=BAND_COLUMNS if self.bandwidth_side else [],strict_causal_main_candidate=self.normalization=='global_gain')
def configs():
    rows=[]
    for sr in [16000,48000]:
        for mel in ([48] if sr==16000 else [48,64]):
            for norm in ['peak','global_gain']:
                for band in [False,True]:
                    cfg=FrontendConfig(sr,mel,norm,band).as_dict();cfg['candidate']=f'A1V3_sr{sr}_mel{mel}_{norm}_band{int(band)}';cfg['side_dim']=(4 if norm=='global_gain' else 0)+(4 if band else 0);rows.append(cfg)
    return rows
def resample(x,sr,target):
    if sr==target:return x
    g=math.gcd(int(sr),target);return resample_poly(x,target//g,int(sr)//g).astype(np.float32)
class Frontend:
    def __init__(self,config=FrontendConfig(),device='cpu',global_gain=1.):
        import librosa
        self.config=config;self.device=device;self.global_gain=float(global_gain)
        self.n_fft=config.sr//50;self.hop=config.sr//100
        self.window=torch.hann_window(self.n_fft,device=device)
        self.mel=torch.as_tensor(librosa.filters.mel(sr=config.sr,n_fft=self.n_fft,n_mels=config.n_mels,fmin=50,fmax=7600 if config.sr==16000 else 20000),device=device)
    def load_audio(self,path):
        w,sr=sf.read(path,dtype='float32');return (w.mean(1) if w.ndim==2 else w),int(sr)
    def prepare(self,w,sr):
        x=resample(np.asarray(w,np.float32),sr,self.config.sr)
        if self.config.normalization=='peak':
            peak=float(np.max(np.abs(x))) if len(x) else 0
            if peak:x=x/peak
        elif self.config.normalization!='global_gain':raise ValueError(self.config.normalization)
        if len(x)<self.n_fft:x=np.pad(x,(0,self.n_fft-len(x)))
        return x
    @inference_precision()
    def extract_features(self,w,sr):
        x=torch.as_tensor(self.prepare(w,sr),device=self.device)
        with torch.inference_mode():
            power=torch.fft.rfft(x.unfold(0,self.n_fft,self.hop)*self.window,dim=-1).abs().square()
            mel_power=power@self.mel.T
            if self.config.normalization=='global_gain':mel_power=mel_power*(self.global_gain**2)
            mel=mel_power.clamp_min(1e-10).log().cpu().numpy()
        state=RunningSide(self.device);state.update_samples(w)
        if self.config.normalization=='global_gain' or self.config.bandwidth_side:
            native=resample(np.asarray(w,np.float32),sr,48000)
            if len(native)<960:native=np.pad(native,(0,960-len(native)))
            frames=np.lib.stride_tricks.sliding_window_view(native,960)[::480]
            state.update_frames(frames)
        return dict(logmel=mel,side=state.value(self.config.normalization=='global_gain',self.config.bandwidth_side))
    def infer(self,features):return features
class RunningSide:
    """Raw sample counters plus completed48k frames; no stored full waveform/sequence."""
    def __init__(self,device='cpu'):
        self.device=device;self.samples=0;self.sumsq=0.;self.clipped=0;self.nearzero=0;self.active_power=0.;self.active_frames=0;self.band_sum=np.zeros(4,np.float64);self.frames=0;self._frame_residual=np.empty(0,np.float32)
    def update_samples(self,x):
        x=np.asarray(x,np.float32);self.samples+=len(x);self.sumsq+=float(np.sum(x.astype(np.float64)**2));self.clipped+=int(np.sum(np.abs(x)>=.999));self.nearzero+=int(np.sum(np.abs(x)<=1e-5))
    def update_frames(self,frames):
        frames=np.asarray(frames,np.float32)
        rms_power=np.mean(frames.astype(np.float64)**2,axis=1);active=rms_power>1e-5
        self.active_power+=float(rms_power[active].sum());self.active_frames+=int(active.sum())
        tensor=torch.as_tensor(frames.copy(),device=self.device);window=torch.hann_window(960,device=self.device)
        power=torch.fft.rfft(tensor*window,dim=-1).abs().square().cpu().numpy().astype(np.float64)
        self.update_band_power(power)
    def push_resampled48(self,samples):
        """Caller supplies contiguous causal48k resampler outputs; retain less than one frame."""
        self._frame_residual=np.concatenate([self._frame_residual,np.asarray(samples,np.float32)])
        if len(self._frame_residual)>=960:
            n=(len(self._frame_residual)-960)//480+1
            frames=np.lib.stride_tricks.sliding_window_view(self._frame_residual,960)[::480][:n]
            self.update_frames(frames);self._frame_residual=self._frame_residual[n*480:].copy()
    def finish_resampled48(self):
        if self.frames==0 and len(self._frame_residual):
            self.update_frames(np.pad(self._frame_residual,(0,960-len(self._frame_residual)))[None,:])
        return self.value()
    def state_bytes(self):return self._frame_residual.nbytes+self.band_sum.nbytes+7*8
    def update_band_power(self,power):
        power=np.asarray(power,np.float64);total=power.sum(1);good=total>0;freq=np.fft.rfftfreq(960,1/48000)
        values=[np.divide(power[:,freq>cutoff].sum(1),total,out=np.zeros_like(total),where=good) for cutoff in [4000,7000,14000]]
        roll_index=(np.cumsum(power,axis=1)<(.95*total)[:,None]).sum(1);roll=freq[np.minimum(roll_index,len(freq)-1)];roll[~good]=0
        self.band_sum+=np.array([v.sum() for v in values]+[roll.sum()]);self.frames+=len(power)
    def value(self,level=True,band=True):
        values=[]
        if level:
            values=[10*np.log10(max(self.sumsq/max(self.samples,1),1e-12)),10*np.log10(max(self.active_power/max(self.active_frames,1),1e-12)),self.clipped/max(self.samples,1),self.nearzero/max(self.samples,1)]
        if band:values.extend(self.band_sum/max(self.frames,1))
        return np.asarray(values,np.float32)
