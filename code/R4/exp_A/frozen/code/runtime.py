"""Production inference and separable cost stages for frozen A candidates."""
from pathlib import Path
import json,pickle,sys
import numpy as np,pandas as pd,soundfile as sf,torch
from features_t2 import prepare as peak_prepare,primitives,features_from_primitives,frame_b
from models.tiny_temporal import TinyTemporal
from legacy_frontend import prepare as legacy_prepare,frontend

class Predictor:
    def __init__(self,folder,device='cpu'):
        self.folder=Path(folder);self.spec=json.loads((self.folder/'spec.json').read_text());self.device=device;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if self.spec['type']=='hgb':
            with (self.folder/'models.pkl').open('rb') as f:self.model=pickle.load(f)
        else:
            ck=torch.load(self.folder/'model.pt',map_location=device,weights_only=False);self.model=TinyTemporal(ck['kind'],ck['with_b']).to(device);self.model.load_state_dict(ck['state_dict']);self.model.eval();self.mean=ck['mean'].to(device);self.std=ck['std'].to(device);self.with_b=ck['with_b']
    def load_audio(self,path):
        w,sr=sf.read(str(path),dtype='float32');return (w.mean(1) if w.ndim==2 else w),sr
    def extract_features(self,wav,sr):
        if self.spec['type']=='hgb':
            w,orig,duration=legacy_prepare(wav,sr);base=frontend(w,orig,duration,device=self.device,columns=self.spec['base_columns'])
            if self.spec.get('newT'):base.update(features_from_primitives(primitives(peak_prepare(wav,sr)),self.spec['newT']))
            return np.array([[base[c] for c in self.spec['columns']]],dtype=np.float64)
        # Minimal A3 frontend excludes unrelated pyin and multiscale statistics.
        import librosa
        x=peak_prepare(wav,sr);mel=librosa.feature.melspectrogram(y=x,sr=16000,n_fft=320,hop_length=160,n_mels=48,fmin=50,fmax=7600,center=False,power=2);z=np.log(np.maximum(mel,1e-10)).T.astype(np.float32)
        if self.with_b:
            from features_t2 import frame_state
            n=320;frames=np.lib.stride_tricks.sliding_window_view(np.pad(x,(0,max(0,n-len(x)))),n)[::160];db=10*np.log10(np.maximum(np.mean(frames.astype(np.float64)**2,axis=1),1e-14));act=frame_state(db,10);b=(db<np.median(db[act])-15).astype(np.float32);z=np.c_[z,b[:len(z)]]
        return z
    def infer(self,features):
        if self.spec['type']=='hgb':return np.column_stack([m.predict(features) for m in self.model])
        x=torch.tensor(features[None],device=self.device);x[:,:,:48]=(x[:,:,:48]-self.mean)/self.std;mask=torch.ones(x.shape[:2],device=self.device,dtype=torch.bool)
        with torch.inference_mode():return self.model(x,mask).cpu().numpy()
    def predict_csv(self,csv,out):
        d=pd.read_csv(csv);rows=[]
        for r in d.itertuples():
            w,sr=self.load_audio(r.path);p=self.infer(self.extract_features(w,sr))[0];rows.append(dict(file=r.file,**dict(zip(['mos','noi','col','dis','loud'],p))))
        pd.DataFrame(rows).to_csv(out,index=False)

