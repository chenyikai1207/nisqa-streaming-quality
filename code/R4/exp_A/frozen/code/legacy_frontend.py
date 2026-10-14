"""Minimal native legacy extraction, with separately measured production stages."""
from pathlib import Path
import math,time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASE=['base__duration','base__amp_rms','base__nearzero','base__clipfrac']+[f'base__{n}_{s}' for s in ['mean','std'] for n in ['rms_db','flux','centroid','flatness','logjump']]
ORDER=[f'order__{n}_{s}' for n in ['flux','logjump','flatness'] for s in ['maxrun','runrate','lag1','transitions']]
COLS=BASE+ORDER

def decode(path):
    w,sr=sf.read(str(path),dtype='float32',always_2d=False)
    if w.ndim==2:w=w.mean(axis=1)
    return w,sr

def prepare(w,sr):
    duration=len(w)/sr
    if duration>60:w=w[:sr*60]
    orig=w
    if sr!=16000:
        g=math.gcd(sr,16000);w=resample_poly(w,16000//g,sr//g).astype(np.float32)
    if len(w)<640:w=np.pad(w,(0,640-len(w)))
    return w,orig,duration

def frontend(w,orig,duration,device='cpu',columns=COLS):
    # Original amplitude is intentionally unnormalised. Do not calculate unused tails/q10/q90.
    wanted=set(columns);out={}
    out.update(base__duration=duration,base__amp_rms=float(np.sqrt(np.mean(orig.astype(np.float64)**2))),base__nearzero=float(np.mean(np.abs(orig)<1e-4)),base__clipfrac=float(np.mean(np.abs(orig)>=.98)))
    if device in ('cuda','torchcpu'):
        import torch
        device='cpu' if device=='torchcpu' else device
        a=torch.as_tensor(w,device=device);f=a.unfold(0,640,320)
        rms=20*torch.log10(f.double().square().mean(1).sqrt().clamp_min(1e-7))
        mag=torch.fft.rfft((f*torch.as_tensor(np.hanning(640).astype(np.float32),device=device)).double(),dim=1).abs()
        norm=mag/mag.sum(1,keepdim=True).clamp_min(1e-12);power=mag.square()
        seqs=dict(rms_db=rms,centroid=(norm*torch.arange(321,device=device,dtype=torch.float64)*25).sum(1)/8000,flatness=power.clamp_min(1e-10).log().mean(1).exp()/power.mean(1).clamp_min(1e-10),flux=torch.cat([torch.zeros(1,device=device,dtype=torch.float64),.5*norm.diff(dim=0).abs().sum(1)]),logjump=torch.cat([torch.zeros(1,device=device,dtype=torch.float64),rms.diff().abs()]))
        for name,v in seqs.items():
            out['base__'+name+'_mean']=v.mean().item();out['base__'+name+'_std']=v.std(correction=0).item()
            tags=['q10','q90','tail90','iqr']
            if any('dist__'+name+'_'+s in wanted for s in tags):
                q=torch.quantile(v,torch.tensor([.1,.25,.75,.9],device=device,dtype=torch.float64))
                for s,z in zip(tags,[q[0],q[3],v[v>=q[3]].mean(),q[2]-q[1]]):out['dist__'+name+'_'+s]=z.item()
            if any('order__'+name+'_'+s in wanted for s in ['maxrun','runrate','lag1','transitions']):
                q75=torch.quantile(v,.75);flag=v>q75
                if v.std(correction=0)<1e-10:flag=torch.zeros_like(flag)
                ed=torch.diff(torch.cat([torch.zeros(1,device=device,dtype=torch.int8),flag.to(torch.int8),torch.zeros(1,device=device,dtype=torch.int8)]));starts=torch.where(ed==1)[0];ends=torch.where(ed==-1)[0];a,b=v[:-1],v[1:]
                lag=(((a-a.mean())*(b-b.mean())).mean()/(a.std(correction=0)*b.std(correction=0))).item() if a.std(correction=0)>1e-10 and b.std(correction=0)>1e-10 else np.nan
                vals=[((ends-starts).max().double()/len(v)).item() if len(starts) else 0.,len(starts)/len(v),lag,flag.to(torch.int8).diff().count_nonzero().item()/max(len(v)-1,1)]
                for s,z in zip(['maxrun','runrate','lag1','transitions'],vals):out['order__'+name+'_'+s]=float(z)
    else:
        f=np.lib.stride_tricks.sliding_window_view(w,640)[::320]
        rms=20*np.log10(np.maximum(np.sqrt(np.mean(f.astype(np.float64)**2,axis=1)),1e-7));mag=np.abs(np.fft.rfft((f*np.hanning(640).astype(np.float32)).astype(np.float64),axis=1)).astype(np.float64);norm=mag/np.maximum(mag.sum(1,keepdims=True),1e-12);power=mag**2
        seqs=dict(rms_db=rms,centroid=np.sum(norm*(np.arange(321)*25),axis=1)/8000,flatness=np.exp(np.mean(np.log(np.maximum(power,1e-10)),axis=1))/np.maximum(power.mean(1),1e-10),flux=np.r_[0,.5*np.abs(np.diff(norm,axis=0)).sum(1)],logjump=np.r_[0,np.abs(np.diff(rms))])
        for name,v in seqs.items():
            out['base__'+name+'_mean']=float(v.mean());out['base__'+name+'_std']=float(v.std())
            tags=['q10','q90','tail90','iqr']
            if any('dist__'+name+'_'+s in wanted for s in tags):
                q=np.quantile(v,[.1,.25,.75,.9])
                for s,z in zip(tags,[q[0],q[3],v[v>=q[3]].mean(),q[2]-q[1]]):out['dist__'+name+'_'+s]=float(z)
            if any('order__'+name+'_'+s in wanted for s in ['maxrun','runrate','lag1','transitions']):
                flag=v>np.quantile(v,.75)
                if v.std()<1e-10:flag=np.zeros(len(v),bool)
                edges=np.diff(np.r_[False,flag,False].astype(np.int8));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0];a,b=v[:-1],v[1:]
                vals=[float((ends-starts).max()/len(v)) if len(starts) else 0.,len(starts)/len(v),float(np.corrcoef(a,b)[0,1]) if a.std()>1e-10 and b.std()>1e-10 else np.nan,float(np.count_nonzero(np.diff(flag.astype(np.int8)))/max(len(v)-1,1))]
                for s,z in zip(['maxrun','runrate','lag1','transitions'],vals):out['order__'+name+'_'+s]=z
    return {c:out[c] for c in columns}

def extract(path,device='cpu',columns=COLS):
    w,sr=decode(path);w,orig,d=prepare(w,sr);return frontend(w,orig,d,device,columns)
