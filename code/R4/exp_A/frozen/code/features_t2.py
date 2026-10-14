"""TRAIN/registered inference temporal features, version A1.1."""
import math,time
import numpy as np
import librosa
from scipy.signal import resample_poly
from scipy.ndimage import binary_dilation

CANDIDATES={'abs50_20':('abs',-50,[20]),'abs40_all':('abs',-40,[20,40,80]),'abs30_80':('abs',-30,[80]),'rel15_all':('rel',-15,[20,40,80]),'rel25_40':('rel',-25,[40]),'q75_all':('q75',0,[20,40,80])}
STAT_NAMES=['fraction','run_count','run_mean','run_max_fraction','hist1','hist2','hist34','hist58','histgt8','burstiness','p01','q10','burstR','Ppl']

def prepare(wav,sr):
    x=np.asarray(wav,dtype=np.float32)
    if x.ndim==2:x=x.mean(1)
    if sr!=16000:
        g=math.gcd(int(sr),16000);x=resample_poly(x,16000//g,int(sr)//g).astype(np.float32)
    peak=float(np.max(np.abs(x))) if len(x) else 0
    if peak>0:x=x/peak
    return x

def runs(b):
    e=np.diff(np.r_[False,b,False].astype(np.int8));return np.where(e==1)[0],np.where(e==-1)[0]

def state_stats(b,segment_lengths=None):
    b=np.asarray(b,bool);n=len(b)
    parts=np.split(b,np.cumsum(segment_lengths)[:-1]) if segment_lengths is not None and len(segment_lengths) else [b]
    starts=[];lens=[];gaps=[];offset=0;n0=n1=n01=n10=0
    for part in parts:
        s,e=runs(part);starts.extend(s+offset);lens.extend(e-s);gaps.extend(np.diff(s));offset+=len(part)
        n0+=int((~part[:-1]).sum());n1+=int(part[:-1].sum())
        n01+=int((~part[:-1]&part[1:]).sum());n10+=int((part[:-1]&~part[1:]).sum())
    s=np.asarray(starts);length=np.asarray(lens)
    # Jeffreys beta(.5,.5), applied to both transition outcomes.
    p=(n01+.5)/(n0+1)
    q=(n10+.5)/(n1+1)
    gaps=np.asarray(gaps)
    mu=float(gaps.mean()) if len(gaps) else 0;sd=float(gaps.std()) if len(gaps) else 0
    vals=[b.mean() if n else 0,len(s),length.mean() if len(s) else 0,length.max()/n if len(s) and n else 0]
    vals += [np.mean(length==1) if len(s) else 0,np.mean(length==2) if len(s) else 0,np.mean((length>=3)&(length<=4)) if len(s) else 0,np.mean((length>=5)&(length<=8)) if len(s) else 0,np.mean(length>8) if len(s) else 0]
    vals += [(sd-mu)/(sd+mu) if sd+mu else 0,p,q,1/(p+q),p/(p+q)]
    return dict(zip(STAT_NAMES,map(float,vals)))

def frame_state(db,hop_ms):
    initial=db>max(-60,float(np.max(db))-30)
    # Symmetric 100ms hangover; bridge original inter-active gaps <=150ms.
    active=binary_dilation(initial,iterations=max(1,int(100/hop_ms)))
    s,e=runs(~active)
    for a,z in zip(s,e):
        if a>0 and z<len(active) and (z-a)*hop_ms<=150:active[a:z]=True
    return active

def primitives(x,include_pitch=True):
    d={}
    for ms in [20,40,80]:
        n=int(ms*16);hop=n//2;xx=np.pad(x,(0,max(0,n-len(x))))
        frames=np.lib.stride_tricks.sliding_window_view(xx,n)[::hop]
        db=10*np.log10(np.maximum(np.mean(frames.astype(np.float64)**2,axis=1),1e-14))
        d[f'db{ms}']=db.astype(np.float32);d[f'active{ms}']=frame_state(db,ms/2)
    mel=librosa.feature.melspectrogram(y=x,sr=16000,n_fft=320,hop_length=160,n_mels=48,fmin=50,fmax=7600,center=False,power=2)
    logmel=np.log(np.maximum(mel,1e-10)).astype(np.float32)
    d['logmel']=logmel.T
    d['zero']=np.array([len(s) for s in []])
    zs,ze=runs(np.abs(x)<1e-4);good=(ze-zs)>=80
    d['zero_count']=np.array(int(good.sum()));d['zero_seconds']=np.array(float((ze[good]-zs[good]).sum()/16000))
    d['zero_bits']=np.packbits(np.abs(x)<1e-4);d['zero_n']=np.array(len(x))
    st=time.perf_counter()
    if include_pitch:
        f0,voiced,_=librosa.pyin(x,sr=16000,fmin=65,fmax=500,frame_length=1024,hop_length=160,center=False,n_thresholds=30)
    else:f0=np.full(len(logmel.T),np.nan);voiced=np.zeros(len(f0),bool)
    d['f0']=f0.astype(np.float32);d['voiced']=voiced;d['pyin_seconds']=np.array(time.perf_counter()-st)
    return d

def features_from_primitives(d,candidate='abs40_all',shuffle_seed=None):
    family,th,scales=CANDIDATES[candidate];out={}
    native20=np.asarray(d['active20'],bool);perm20=np.arange(len(native20))
    if shuffle_seed is not None:
        ids=np.flatnonzero(native20);permuted=ids.copy();np.random.default_rng(shuffle_seed).shuffle(permuted);perm20[ids]=permuted
    for ms in scales:
        db=np.asarray(d[f'db{ms}']);active=np.asarray(d[f'active{ms}'],bool)
        # Preserve each activity-domain energy marginal, not mixing silence with speech.
        if ms==20:seq=db[perm20][active].copy()
        else:
            seq=db[active].copy()
            if shuffle_seed is not None:np.random.default_rng(np.random.SeedSequence([shuffle_seed,ms])).shuffle(seq)
        cutoff=th if family=='abs' else float(np.median(seq)+th) if family=='rel' and len(seq) else -100
        if family=='q75':cutoff=float(np.quantile(seq,.75)) if len(seq) else -100
        b=seq<cutoff
        ast,aend=runs(active)
        for k,v in state_stats(b,aend-ast).items():out[f't2_{ms}_{k}']=v
    lm=np.asarray(d['logmel']);act=np.asarray(d['active20'])[:len(lm)];lm=lm[:len(act)].copy()[perm20[:len(lm)]]
    valid=act[:-1]&act[1:]
    flux=np.linalg.norm(np.diff(lm,axis=0)[valid],axis=1) if valid.any() else np.array([])
    out['t2_flux95']=float(np.quantile(flux,.95)) if len(flux) else float('nan');out['t2_flux_exceed']=float(np.sum(flux>10))
    rawf=np.asarray(d['f0']);f=np.full(len(act),np.nan,dtype=np.float32)
    # pyin1024ms-frame center versus mel320-sample-frame center: nearest 10ms grid.
    offset=round((1024/2-320/2)/160);nf=min(len(rawf),len(f)-offset);f[offset:offset+nf]=rawf[:nf];f=f[perm20[:len(f)]]
    good=act[:-1]&act[1:]&np.isfinite(f[:-1])&np.isfinite(f[1:]);jumps=np.abs(12*np.log2(f[1:][good]/f[:-1][good]))
    out['t2_pitch_jump4']=float(np.mean(jumps>4)) if len(jumps) else float('nan')
    if shuffle_seed is not None:
        bits=np.unpackbits(d['zero_bits'])[:int(d['zero_n'])].astype(bool);nn=len(bits)//160
        chunks=bits[:nn*160].reshape(nn,160).copy();n=min(len(perm20),nn);chunks[:n]=chunks[:n][perm20[:n]];bits=np.r_[chunks.ravel(),bits[nn*160:]]
        zs,ze=runs(bits);good=ze-zs>=80
        out['t2_nearzero_count']=float(good.sum());out['t2_nearzero_seconds']=float((ze[good]-zs[good]).sum()/16000)
    else:
        out['t2_nearzero_count']=float(d['zero_count']);out['t2_nearzero_seconds']=float(d['zero_seconds'])
    return out

def extract_t2(wav:np.ndarray,sr:int)->dict[str,float]:
    return features_from_primitives(primitives(prepare(wav,sr)))

def frame_b(d):
    db=d['db20'];return (db<np.median(db[d['active20']])-15).astype(np.float32)
