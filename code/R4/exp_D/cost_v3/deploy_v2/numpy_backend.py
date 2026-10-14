from public_paths import data_path
from public_paths import data_path
from public_paths import data_path
"""Whole-sequence NumPy deployment; exact frozen arrays/FIR. No model retraining or mapping changes.
Float32/64 recurrent candidates; frontend/FIR remain fp32, NumPy FFT complex128 castcomplex64 beforepower.
Native unused bandwidth FFT omitted only for band0 (all frozen required candidates). Full prediction is not stream replay.
"""
import numpy as np
from adapters.paths import D2,add_code
add_code(D2)
from lean_raw_numpy_minimal_v5 import Backend as OriginalBackend,FIR,Stream as OriginalStream
class Backend(OriginalBackend):
    def __init__(self,candidate='v2_pure',precision='float64'):
        super().__init__(data_path('deployment_arrays')/(candidate+'_frozen_arrays.npz'));self.precision=np.dtype(precision)
        self.weights=[[x.astype(self.precision) for x in w] for w in self.weights];self.aw=self.aw.astype(self.precision);self.hw=self.hw.astype(self.precision);self.hb=self.hb.astype(self.precision);self.candidate=candidate
    def new_stream(self,sr):return Stream(self,sr)
    def resample(self,w,sr):
        a=FIR(sr,self.sr);b=FIR(sr,48000);return np.r_[a.push(w),a.finish()],np.r_[b.push(w),b.finish()]
    def level(self,w,sidewave):
        if len(sidewave)<960:sidewave=np.pad(sidewave,(0,960-len(sidewave)))
        frames=np.lib.stride_tricks.sliding_window_view(sidewave,960)[::480];p=np.mean(frames.astype(np.float64)**2,axis=1);act=p>1e-5
        s=np.asarray([10*np.log10(max(float(np.sum(w.astype(np.float64)**2))/max(len(w),1),1e-12)),10*np.log10(max(float(p[act].sum())/max(int(act.sum()),1),1e-12)),float(np.sum(np.abs(w)>=.999))/max(len(w),1),float(np.sum(np.abs(w)<=1e-5))/max(len(w),1)],np.float32)
        return ((s-self.a['side_mean'])/self.a['side_std']).astype(np.float32)
    def features(self,x):
        if len(x)<self.nfft:x=np.pad(x,(0,self.nfft-len(x)))
        f=np.lib.stride_tricks.sliding_window_view(x,self.nfft)[::self.hop];spec=np.fft.rfft(f*self.window,axis=-1).astype(np.complex64);power=np.abs(spec).astype(np.float32)**2;mel=np.log(np.maximum((power@self.mel.T)*np.float32(self.gain**2),np.float32(1e-10)))
        return ((mel-self.a['mean'])/self.a['std']).astype(self.precision)
    def infer(self,features,side):
        seq=features
        for wi,wh,bi,bh in self.weights:
            inputs=seq@wi.T+bi;h=np.zeros(self.width,self.precision);out=np.empty((len(seq),self.width),self.precision)
            for i,gi in enumerate(inputs):
                gh=wh@h+bh;ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3);r=1/(1+np.exp(-(ir+hr)));z=1/(1+np.exp(-(iz+hz)));nn=np.tanh(inn+r*hn);h=(1-z)*nn+z*h;out[i]=h
            seq=out
        score=seq@self.aw+self.ab;weights=np.exp(score-score.max());pool=(np.sum(seq*weights[:,None],axis=0)/weights.sum()).astype(np.float32)
        return (self.hw@np.r_[pool,side].astype(self.precision)+self.hb).astype(np.float32)[None,:]
    def predict_array(self,w,sr):
        x,sidewave=self.resample(w,sr);return self.infer(self.features(x),self.level(w,sidewave))
class Stream(OriginalStream):
    def __init__(self,b,sr):
        super().__init__(b,sr);self.h=self.h.astype(b.precision);self.num=self.num.astype(b.precision)
    def main_frames(self,x):
        self.buffer=np.r_[self.buffer,x];b=self.b
        if len(self.buffer)<b.nfft:return
        n=(len(self.buffer)-b.nfft)//b.hop+1;f=np.lib.stride_tricks.sliding_window_view(self.buffer,b.nfft)[::b.hop][:n];spec=np.fft.rfft(f*b.window,axis=-1).astype(np.complex64);power=np.abs(spec).astype(np.float32)**2;mel=np.log(np.maximum((power@b.mel.T)*np.float32(b.gain**2),np.float32(1e-10)));features=((mel-b.a['mean'])/b.a['std']).astype(b.precision)
        for x in features:
            for l,(wi,wh,bi,bh) in enumerate(b.weights):
                gi=wi@x+bi;gh=wh@self.h[l]+bh;ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3);r=1/(1+np.exp(-(ir+hr)));z=1/(1+np.exp(-(iz+hz)));nn=np.tanh(inn+r*hn);self.h[l]=(1-z)*nn+z*self.h[l];x=self.h[l]
            score=float(b.aw@x+b.ab);m=max(self.maximum,score);fac=np.exp(self.maximum-m) if self.frames else 0.;weight=np.exp(score-m);self.den=self.den*fac+weight;self.num=self.num*fac+weight*x;self.maximum=m;self.frames+=1
        self.buffer=self.buffer[n*b.hop:].copy()
