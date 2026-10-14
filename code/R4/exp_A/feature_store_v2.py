"""CPU mmap feature reference; each spawned worker owns its CUDA input copy."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np,torch
GPU_CACHE={}
class MmapCudaBatches:
    """Keep immutable feature matrix mapped on CPU, copy exact gathered batch to private CUDA."""
    def __init__(self,path):
        self.array=np.load(path,mmap_mode='r');self.shape=self.array.shape
    def __getitem__(self,key):
        rows,columns=key if isinstance(key,tuple) else (key,slice(None))
        if torch.is_tensor(rows):rows=rows.detach().cpu().numpy()
        values=np.ascontiguousarray(self.array[rows,columns],dtype=np.float32)
        return torch.from_numpy(values).to('cuda')

@dataclass
class FeatureStore:
    path:str
    def resolve(self):
        global GPU_CACHE
        config=Path(__file__).resolve().parent/'GPU_CONCURRENCY_v2.json'
        mode=json.loads(config.read_text()).get('feature_residency','private_cuda_full') if config.exists() else 'private_cuda_full'
        key=(self.path,mode)
        if key not in GPU_CACHE:
            GPU_CACHE.clear();torch.cuda.empty_cache()
            p=Path(self.path)
            # Copy to CUDA from read-only mmap; no CUDA IPC handles cross processes.
            x=MmapCudaBatches(p/'x.npy') if mode=='cpu_mmap_private_cuda_batches' else torch.tensor(np.load(p/'x.npy',mmap_mode='r'),device='cuda')
            mask=torch.tensor(np.load(p/'mask.npy',mmap_mode='r'),device='cuda')
            GPU_CACHE[key]=(x,mask)
        return GPU_CACHE[key]
