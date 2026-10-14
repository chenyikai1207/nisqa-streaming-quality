"""Bounded delayed polyphase FIR matching scipy resample_poly default Kaiser5 coefficients."""
import math
import numpy as np,torch
from scipy.signal import firwin
class CausalResampler:
    def __init__(self,native_sr,target_sr,device='cpu'):
        g=math.gcd(int(native_sr),int(target_sr));self.up=target_sr//g;self.down=native_sr//g;self.native_sr=native_sr;self.target_sr=target_sr;self.device=device;self.buffer=np.empty(0,np.float32);self.start=0;self.received=0;self.output_index=0;self.closed=False
        if self.up==self.down:self.h=torch.ones(1,device=device);self.remove=0;self.latency_s=0.;return
        half=10*max(self.up,self.down);h=firwin(2*half+1,1/max(self.up,self.down),window=('kaiser',5.)).astype(np.float32)*self.up;pre=self.down-half%self.down;self.remove=(half+pre)//self.down;self.h=torch.tensor(np.r_[np.zeros(pre,np.float32),h],device=device);self.latency_s=(half/self.up)/native_sr
    def push(self,x,final=False):
        assert not self.closed;x=np.asarray(x,np.float32);self.buffer=np.r_[self.buffer,x];self.received+=len(x)
        if self.up==self.down:self.buffer=np.empty(0,np.float32);self.start=self.received;self.output_index=self.received;self.closed=final;return x.copy()
        stop=math.ceil(self.received*self.up/self.down) if final else max(self.output_index,math.ceil((self.received*self.up)/self.down)-self.remove)
        js=torch.arange(self.output_index,stop,device=self.device,dtype=torch.int64)
        if not len(js):self.closed=final;return np.empty(0,np.float32)
        t=(js+self.remove)*self.down;right=torch.div(t,self.up,rounding_mode='floor');support=math.ceil(len(self.h)/self.up);offset=torch.arange(support,device=self.device);indices=right[:,None]-offset[None,:];hi=t[:,None]-indices*self.up;good=(hi>=0)&(hi<len(self.h))&(indices>=self.start)&(indices<self.received)
        samples=torch.tensor(self.buffer,device=self.device);local=(indices-self.start).clamp(0,max(0,len(samples)-1));products=samples[local]*self.h[hi.clamp(0,len(self.h)-1)]*good
        # upfirdn accumulates float32 products in ascending native-input order.
        # A tree reduction changes tiny floor-adjacent mel bands after log.
        values=torch.zeros(len(js),device=self.device)
        for column in range(support-1,-1,-1):values.add_(products[:,column])
        out=values.cpu().numpy();self.output_index=stop
        next_t=(stop+self.remove)*self.down;oldest=max(0,math.ceil((next_t-(len(self.h)-1))/self.up));discard=min(max(0,oldest-self.start),len(self.buffer));self.buffer=self.buffer[discard:].copy();self.start+=discard;self.closed=final;return out
    def finish(self):return self.push(np.empty(0,np.float32),final=True)
    def state_bytes(self):return self.buffer.nbytes+7*8
