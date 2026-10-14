"""Raw causal NumPy inference, immutable f32 weights promoted f64 for recurrence.
No Torch/ORT/librosa import. Finite FIR + STFT + raw-level side + GRU + attention.
Unused bandwidth FFT omitted only for actual band0 frozen models. No whole audio retained.
"""
import math
from pathlib import Path
import numpy as np
with np.load(Path(__file__).resolve().parent/'lean_raw_v5/exact_causal_FIR_values.npz') as _z:FIR_VALUES={k:_z[k].copy() for k in _z.files}
def expit(x):
    return 1/(1+np.exp(-x))
class FIR:
    def __init__(self,sr,target):
        g=math.gcd(sr,target);self.up=target//g;self.down=sr//g;self.buffer=np.empty(0,np.float32);self.start=0;self.received=0;self.index=0;self.closed=False
        if self.up==self.down:self.h=np.ones(1,np.float32);self.remove=0;self.delay=0.;return
        half=10*max(self.up,self.down);pre=self.down-half%self.down;self.remove=(half+pre)//self.down;self.h=FIR_VALUES[f'h_{sr}_{target}'];self.delay=(half/self.up)/sr
    def push(self,x,final=False):
        if self.closed:raise ValueError('closed')
        x=np.asarray(x,np.float32);self.received+=len(x)
        if self.up==self.down:self.index=self.received;self.start=self.received;self.closed=final;return x.copy()
        self.buffer=np.r_[self.buffer,x];stop=math.ceil(self.received*self.up/self.down) if final else max(self.index,math.ceil(self.received*self.up/self.down)-self.remove)
        js=np.arange(self.index,stop,dtype=np.int64)
        if not len(js):self.closed=final;return np.empty(0,np.float32)
        t=(js+self.remove)*self.down;right=t//self.up;support=math.ceil(len(self.h)/self.up);ix=right[:,None]-np.arange(support);hi=t[:,None]-ix*self.up;good=(hi>=0)&(hi<len(self.h))&(ix>=self.start)&(ix<self.received)
        products=self.buffer[np.clip(ix-self.start,0,len(self.buffer)-1)]*self.h[np.clip(hi,0,len(self.h)-1)]*good;value=np.zeros(len(js),np.float32)
        for c in range(support-1,-1,-1):value+=products[:,c]
        self.index=stop;oldest=max(0,math.ceil(((stop+self.remove)*self.down-(len(self.h)-1))/self.up));discard=min(max(0,oldest-self.start),len(self.buffer));self.buffer=self.buffer[discard:].copy();self.start+=discard;self.closed=final;return value
    def finish(self):return self.push(np.empty(0,np.float32),True)
    def state_bytes(self):return self.buffer.nbytes+7*8
class Backend:
    def __init__(self,arrays):
        with np.load(arrays) as z:self.a={k:z[k].copy() for k in z.files}
        self.width=int(self.a['width']);self.layers=int(self.a['layers']);self.sr=int(self.a['sr']);self.gain=float(self.a['gain']);self.window=self.a['window'];self.mel=self.a['mel'];self.nfft=len(self.window);self.hop=self.sr//100
        self.weights=[]
        for l in range(self.layers):self.weights.append([self.a['encoder.'+k+'_l'+str(l)].astype(np.float64) for k in ['weight_ih','weight_hh','bias_ih','bias_hh']])
        self.aw=self.a['attention.weight'].astype(np.float64).reshape(-1);self.ab=float(self.a['attention.bias'][0]);self.hw=self.a['head.weight'].astype(np.float64);self.hb=self.a['head.bias'].astype(np.float64)
    def new_stream(self,native_sr):return Stream(self,int(native_sr))
    def model_bytes(self):return sum(a.nbytes for a in self.a.values())+sum(x.nbytes for w in self.weights for x in w)+self.aw.nbytes+self.hw.nbytes+self.hb.nbytes
class Stream:
    def __init__(self,b,sr):
        self.b=b;self.sr=sr;self.main=FIR(sr,b.sr);self.sidefir=FIR(sr,48000);self.buffer=np.empty(0,np.float32);self.sidebuffer=np.empty(0,np.float32);self.h=np.zeros((b.layers,b.width),np.float64);self.maximum=-np.inf;self.den=0.;self.num=np.zeros(b.width,np.float64);self.frames=0;self.sideframes=0;self.samples=0;self.sumsq=0.;self.clipped=0;self.nearzero=0;self.active_power=0.;self.active_frames=0;self.closed=False
    def main_frames(self,x):
        self.buffer=np.r_[self.buffer,x];b=self.b
        if len(self.buffer)<b.nfft:return
        n=(len(self.buffer)-b.nfft)//b.hop+1;f=np.lib.stride_tricks.sliding_window_view(self.buffer,b.nfft)[::b.hop][:n];spec=np.fft.rfft(f*b.window,axis=-1).astype(np.complex64);power=np.abs(spec).astype(np.float32)**2;mel=np.log(np.maximum((power@b.mel.T)*np.float32(b.gain**2),np.float32(1e-10)));features=((mel-b.a['mean'])/b.a['std']).astype(np.float64)
        for x in features:
            for l,(wi,wh,bi,bh) in enumerate(b.weights):
                gi=wi@x+bi;gh=wh@self.h[l]+bh;ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3);r=expit(ir+hr);z=expit(iz+hz);nn=np.tanh(inn+r*hn);self.h[l]=(1-z)*nn+z*self.h[l];x=self.h[l]
            score=float(b.aw@x+b.ab);m=max(self.maximum,score);fac=np.exp(self.maximum-m) if self.frames else 0.;weight=np.exp(score-m);self.den=self.den*fac+weight;self.num=self.num*fac+weight*x;self.maximum=m;self.frames+=1
        self.buffer=self.buffer[n*b.hop:].copy()
    def side_frames(self,x):
        self.sidebuffer=np.r_[self.sidebuffer,x]
        if len(self.sidebuffer)<960:return
        n=(len(self.sidebuffer)-960)//480+1;f=np.lib.stride_tricks.sliding_window_view(self.sidebuffer,960)[::480][:n];power=np.mean(f.astype(np.float64)**2,axis=1);active=power>1e-5;self.active_power+=float(power[active].sum());self.active_frames+=int(active.sum());self.sideframes+=n;self.sidebuffer=self.sidebuffer[n*480:].copy()
    def push(self,w):
        if self.closed:raise ValueError('closed')
        w=np.asarray(w,np.float32);w=w.mean(1) if w.ndim==2 else w;self.samples+=len(w);self.sumsq+=float(np.sum(w.astype(np.float64)**2));self.clipped+=int(np.sum(np.abs(w)>=.999));self.nearzero+=int(np.sum(np.abs(w)<=1e-5));self.main_frames(self.main.push(w));self.side_frames(self.sidefir.push(w))
    def predict(self):
        if not self.frames:raise ValueError('no frame')
        side=np.asarray([10*np.log10(max(self.sumsq/max(self.samples,1),1e-12)),10*np.log10(max(self.active_power/max(self.active_frames,1),1e-12)),self.clipped/max(self.samples,1),self.nearzero/max(self.samples,1)],np.float32);side=((side-self.b.a['side_mean'])/self.b.a['side_std']).astype(np.float32);pool=(self.num/self.den).astype(np.float32);return (self.b.hw@np.r_[pool,side].astype(np.float64)+self.b.hb).astype(np.float32)[None,:]
    def finish(self):
        if not self.closed:
            self.main_frames(self.main.finish());self.side_frames(self.sidefir.finish())
            if not self.frames:self.main_frames(np.zeros(max(0,self.b.nfft-len(self.buffer)),np.float32))
            if not self.sideframes and len(self.sidebuffer):self.side_frames(np.zeros(960-len(self.sidebuffer),np.float32))
            self.closed=True
        return self.predict()
    def state_bytes(self):return self.buffer.nbytes+self.sidebuffer.nbytes+self.h.nbytes+self.num.nbytes+self.main.state_bytes()+self.sidefir.state_bytes()+13*8
    def state_summary(self):return dict(bytes=self.state_bytes(),quality_frames=self.frames,main_buffer_samples=len(self.buffer),side_buffer_samples=len(self.sidebuffer),whole_waveform_retained=False,algorithm_delay_ms=max(self.main.delay,self.sidefir.delay)*1000,closed=self.closed,precision='float32 frontend/FIR; float64 unchanged-weight recurrent arithmetic; float32 returned5D',unused_bandwidth_FFT_omitted=True)