"""Finite causal adaptations of oldT12 and no-pYIN newT18, not historical whole-clip T."""
import numpy as np,torch
from qualified_b2_v3 import inference_precision
OLD_COLUMNS=[f'old_{x}_{stat}_stream_v3' for x in ['flux','logjump','flatness'] for stat in ['maxrun','runrate','lag1','transitions']]
NEW_COLUMNS=['t2_80_fraction','t2_80_run_count','t2_80_run_mean','t2_80_run_max_fraction','t2_80_hist1','t2_80_hist2','t2_80_hist34','t2_80_hist58','t2_80_histgt8','t2_80_burstiness','t2_80_p01','t2_80_q10','t2_80_burstR','t2_80_Ppl','t2_flux95','t2_flux_exceed','t2_nearzero_count','t2_nearzero_seconds']
NEW_COLUMNS=[x+'_stream_v3' for x in NEW_COLUMNS]
G0_COLUMNS=['duration','amp_rms','nearzero_fraction','clipping_fraction']+[f'{n}_{s}' for s in ['mean','std'] for n in ['rms_db','flux','centroid','flatness','logjump']]
class Moments:
    def __init__(self):self.n=0;self.s=0.;self.ss=0.
    def add(self,x):self.n+=1;self.s+=float(x);self.ss+=float(x)**2
    def value(self):
        mean=self.s/max(1,self.n);return mean,np.sqrt(max(0,self.ss/max(1,self.n)-mean**2))
class Histogram:
    def __init__(self,bins,lo,hi):self.a=np.zeros(bins,np.int64);self.lo=lo;self.hi=hi;self.n=0
    def add(self,x):i=int(np.clip((x-self.lo)/(self.hi-self.lo)*len(self.a),0,len(self.a)-1));self.a[i]+=1;self.n+=1
    def quantile(self,q):
        if not self.n:return self.lo
        i=np.searchsorted(np.cumsum(self.a),q*self.n,side='left');return self.lo+(i+.5)*(self.hi-self.lo)/len(self.a)
class OldState:
    def __init__(self,hi):
        self.hist=Histogram(128,0,hi);self.n=0;self.run=0;self.maxrun=0;self.runs=0;self.transitions=0;self.flag=False;self.prev=None;self.pairs=0;self.sx=self.sy=self.sxx=self.syy=self.sxy=0.
    def add(self,x):
        flag=self.hist.n>0 and x>self.hist.quantile(.75)
        if flag:self.run+=1;self.maxrun=max(self.maxrun,self.run);self.runs+=int(not self.flag)
        else:self.run=0
        self.transitions+=int(self.n>0 and flag!=self.flag);self.flag=flag
        if self.prev is not None:self.pairs+=1;self.sx+=self.prev;self.sy+=x;self.sxx+=self.prev**2;self.syy+=x*x;self.sxy+=self.prev*x
        self.prev=x;self.n+=1;self.hist.add(x)
    def value(self):
        n=max(1,self.pairs);cov=self.sxy-self.sx*self.sy/n;den=np.sqrt(max(0,self.sxx-self.sx**2/n)*max(0,self.syy-self.sy**2/n));lag=cov/den if den>0 else 0.
        return [self.maxrun/max(1,self.n),self.runs/max(1,self.n),lag,self.transitions/max(1,self.n-1)]
class DropState:
    def __init__(self):
        self.active=0;self.drop=0;self.run=0;self.run_n=0;self.run_total=0;self.maxrun=0;self.hist=np.zeros(5,np.int64);self.previous=None;self.n0=self.n1=self.n01=self.n10=0;self.gap=Moments();self.position=0;self.laststart=None
    def close(self):
        if self.run:self.run_n+=1;self.run_total+=self.run;self.maxrun=max(self.maxrun,self.run);self.hist[0 if self.run==1 else 1 if self.run==2 else 2 if self.run<=4 else 3 if self.run<=8 else 4]+=1;self.run=0
    def add(self,active,drop):
        self.position+=1
        if not active:self.close();self.previous=None;self.laststart=None;return
        self.active+=1;self.drop+=int(drop)
        if self.previous is not None:
            if self.previous:self.n1+=1;self.n10+=int(not drop)
            else:self.n0+=1;self.n01+=int(drop)
        if drop:
            if not self.run:
                if self.laststart is not None:self.gap.add(self.position-self.laststart)
                self.laststart=self.position
            self.run+=1
        else:self.close()
        self.previous=drop
    def value(self):
        hist=self.hist.copy();rn=self.run_n;total=self.run_total;mx=self.maxrun
        if self.run:rn+=1;total+=self.run;mx=max(mx,self.run);hist[0 if self.run==1 else 1 if self.run==2 else 2 if self.run<=4 else 3 if self.run<=8 else 4]+=1
        mean,std=self.gap.value();burst=(std-mean)/(std+mean) if std+mean>0 else 0.;p=(self.n01+.5)/(self.n0+1);q=(self.n10+.5)/(self.n1+1)
        return [self.drop/max(1,self.active),rn,total/max(1,rn),mx/max(1,self.active),*(hist/max(1,rn)).tolist(),burst,p,q,1/(p+q),p/(p+q)]
class StreamingTime:
    """Caller supplies contiguous causal-resampler16k output; three FIFO tails <80ms."""
    def __init__(self,gain=1.,device='cpu'):
        import librosa
        self.gain=float(gain);self.device=device;self.residual={320:np.empty(0,np.float32),640:np.empty(0,np.float32),1280:np.empty(0,np.float32)};self.frames={320:0,640:0,1280:0}
        self.hann40=torch.tensor(np.hanning(640).astype(np.float32),device=device);self.hann20=torch.hann_window(320,device=device);self.mel=torch.tensor(librosa.filters.mel(sr=16000,n_fft=320,n_mels=48,fmin=50,fmax=7600),device=device);self.freq=np.fft.rfftfreq(640,1/16000)/8000
        self.old=[OldState(1),OldState(140),OldState(1)];self.g0=[Moments() for _ in range(5)];self.prev_mag=None;self.prev_db=None;self.prev_mel=None;self.prev_active20=False;self.max80=-np.inf;self.max20=-np.inf;self.hang80=0;self.hang20=0;self.drop=DropState();self.flux=Histogram(1024,0,400);self.flux_exceed=0;self.near_run=0;self.near_count=0;self.near_samples=0;self.native_n=0;self.native_sq=0.;self.native_near=0;self.native_clip=0;self.native_seconds=0.
    def update_native(self,x,sr):
        x=np.asarray(x);self.native_n+=len(x);self.native_sq+=float(np.sum(x.astype(np.float64)**2));self.native_near+=int((np.abs(x)<1e-4).sum());self.native_clip+=int((np.abs(x)>=.98).sum());self.native_seconds+=len(x)/sr
    def vad(self,db,width):
        if width==80:
            self.max80=max(self.max80,db);active=db>=max(-60,self.max80-30)
            if active:self.hang80=2
            elif self.hang80:self.hang80-=1;active=True
        else:
            self.max20=max(self.max20,db);active=db>=max(-60,self.max20-30)
            if active:self.hang20=10
            elif self.hang20:self.hang20-=1;active=True
        return active
    @inference_precision()
    def frames_update(self,frames,size):
        self.frames[size]+=len(frames)
        if size==1280:
            db=20*np.log10(np.maximum(np.sqrt(np.mean(frames.astype(np.float64)**2,1))*self.gain,1e-7))
            for v in db:self.drop.add(self.vad(v,80),v<-30)
            return
        t=torch.as_tensor(frames.copy(),device=self.device)
        if size==640:
            mag=torch.fft.rfft((t*self.hann40).double(),dim=-1).abs().cpu().numpy();power=mag**2;flat=np.exp(np.log(np.maximum(power,1e-10)).mean(1))/np.maximum(power.mean(1),1e-10);mag/=np.maximum(mag.sum(1,keepdims=True),1e-12);cent=mag@self.freq;db=20*np.log10(np.maximum(np.sqrt(np.mean(frames.astype(np.float64)**2,1)),1e-7))
            for m,c,f,v in zip(mag,cent,flat,db):
                flux=0. if self.prev_mag is None else .5*np.abs(m-self.prev_mag).sum();jump=0. if self.prev_db is None else abs(v-self.prev_db);self.prev_mag=m;self.prev_db=v
                for state,value in zip(self.old,[flux,jump,f]):state.add(float(value))
                for state,value in zip(self.g0,[v,flux,c,f,jump]):state.add(float(value))
        else:
            power=torch.fft.rfft(t*self.hann20,dim=-1).abs().square();mel=((power@self.mel.T)*(self.gain**2)).clamp_min(1e-10).log().cpu().numpy();db=20*np.log10(np.maximum(np.sqrt(np.mean(frames.astype(np.float64)**2,1))*self.gain,1e-7))
            for m,v in zip(mel,db):
                active=self.vad(v,20)
                if self.prev_mel is not None and active and self.prev_active20:
                    value=float(np.maximum(m-self.prev_mel,0).mean());self.flux.add(value);self.flux_exceed+=int(value>10)
                self.prev_mel=m;self.prev_active20=active
    def push16k(self,x):
        x=np.asarray(x,np.float32)
        for size in [320,640,1280]:
            a=np.concatenate([self.residual[size],x]);hop=size//2
            if len(a)>=size:
                n=(len(a)-size)//hop+1;frames=np.lib.stride_tricks.sliding_window_view(a,size)[::hop][:n]
                for start in range(0,n,128):self.frames_update(frames[start:start+128],size)
                a=a[n*hop:]
            self.residual[size]=a.copy()
        # Run-length encoding handles zeros crossing caller chunk boundaries.
        flag=np.abs(x*self.gain)<1e-4
        starts=np.r_[0,np.flatnonzero(flag[1:]!=flag[:-1])+1,len(flag)] if len(flag) else [0]
        for a,b in zip(starts[:-1],starts[1:]):
            if flag[a]:self.near_run+=b-a
            elif self.near_run:self.close_near()
    def close_near(self):
        if self.near_run>=80:self.near_count+=1;self.near_samples+=self.near_run
        self.near_run=0
    def finish(self):
        for size in [320,640,1280]:
            if not self.frames[size] and len(self.residual[size]):self.frames_update(np.pad(self.residual[size],(0,size-len(self.residual[size])))[None,:],size)
        return self.value()
    def value(self):
        g0=[self.native_seconds,np.sqrt(self.native_sq/max(1,self.native_n)),self.native_near/max(1,self.native_n),self.native_clip/max(1,self.native_n)]
        moments=[state.value() for state in self.g0];g0.extend(x[0] for x in moments);g0.extend(x[1] for x in moments)
        old=[v for state in self.old for v in state.value()];count=self.near_count+int(self.near_run>=80);samples=self.near_samples+(self.near_run if self.near_run>=80 else 0);new=self.drop.value()+[self.flux.quantile(.95),self.flux_exceed/max(1,self.flux.n),count,samples/16000]
        return dict(G0=np.asarray(g0,np.float64),oldT=np.asarray(old,np.float64),newT=np.asarray(new,np.float64))
    def state_bytes(self):
        arrays=list(self.residual.values())+[s.hist.a for s in self.old]+[self.drop.hist,self.flux.a]
        arrays.extend(a for a in [self.prev_mag,self.prev_mel] if a is not None)
        return sum(a.nbytes for a in arrays)+128*8
