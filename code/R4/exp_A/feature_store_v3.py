"""Cached linear mel-power gets current TRAIN constant gain before log floor."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch
GPU_CACHE={}
class PowerTensor:
    def __init__(self,array,gain):self.array=array;self.shape=array.shape;self.frontend_gain=float(gain)
    def __getitem__(self,key):return (self.array[key]*(self.frontend_gain**2)).clamp_min(1e-10).log()
@dataclass
class FeatureStore:
    path:str
    normalization:str='global_gain'
    def resolve(self,train_ids):
        p=Path(self.path);key=(self.path,self.normalization)
        if key not in GPU_CACHE:
            GPU_CACHE.clear();torch.cuda.empty_cache()
            GPU_CACHE[key]=(torch.tensor(np.load(p/'x.npy',mmap_mode='r'),device='cuda'),torch.tensor(np.load(p/'mask.npy',mmap_mode='r'),device='cuda'))
        base,mask=GPU_CACHE[key]
        if self.normalization=='peak':return base,mask
        energy=np.load(p/'raw_energy.npy');sub=energy[np.asarray(train_ids)];rms=float(np.sqrt(sub[:,0].sum()/sub[:,1].sum()));gain=10**(-26/20)/max(rms,1e-12)
        return PowerTensor(base,gain),mask
