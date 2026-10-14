"""Bounded repaircandidate1: unchanged cachedFIR via offlineupfirdn + unchangedweight Torch wholeGRU.
Original numpy backend/stream/qualifiers are read-only. No fittedweights/frontend/map changes.
"""
import math,numpy as np,torch
from scipy.signal import upfirdn
from .numpy_backend import Backend as OriginalBackend,FIR
class Backend(OriginalBackend):
    def __init__(self,candidate='v2_pure',precision='float32'):
        super().__init__(candidate,precision);dtype=torch.float32 if precision=='float32' else torch.float64
        with torch.random.fork_rng(devices=[]):self.rnn=torch.nn.GRU(48,self.width,self.layers,batch_first=True).to(dtype=dtype).eval()
        self.rnn.load_state_dict({key.removeprefix('encoder.'):torch.as_tensor(value,dtype=dtype) for key,value in self.a.items() if key.startswith('encoder.')})
        self.dtype=dtype
    def offline_fir(self,w,sr,target):
        w=np.asarray(w,np.float32)
        if sr==target:return w.copy()
        f=FIR(sr,target);n=math.ceil(len(w)*f.up/f.down);full=upfirdn(f.h,w,up=f.up,down=f.down);return full[f.remove:f.remove+n].astype(np.float32,copy=False)
    def resample(self,w,sr):return self.offline_fir(w,sr,self.sr),self.offline_fir(w,sr,48000)
    def infer(self,features,side):
        with torch.inference_mode():seq=self.rnn(torch.as_tensor(np.ascontiguousarray(features),dtype=self.dtype)[None])[0][0].numpy()
        score=seq@self.aw+self.ab;weights=np.exp(score-score.max());pool=(np.sum(seq*weights[:,None],axis=0)/weights.sum()).astype(np.float32)
        return (self.hw@np.r_[pool,side].astype(self.precision)+self.hb).astype(np.float32)[None,:]
    # inherited new_stream is ORIGINAL NumPy stream andoriginalFIR; no streaming changes.
