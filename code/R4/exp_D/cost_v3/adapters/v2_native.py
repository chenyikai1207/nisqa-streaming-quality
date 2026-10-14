"""Actual frozen A7 CPU whole forward; no streaming replay. Native F includes fixedgain andunusedbandFFT."""
import json,importlib,numpy as np
from .paths import STAGE,D2,add_code,configure_torch,prediction
class Runner96:
    name='v2_96_native';candidate='v2_pure'
    def __init__(self):
        add_code(D2);from A7_runtime_entry_v3 import load_predictor
        self.torch=configure_torch();r=json.loads((STAGE/'exp_A/A7_MODELS_READY_v3.json').read_text(encoding='utf8'));row=next(x for x in r['models'] if x['name']==self.candidate);self.p=load_predictor(r,row,'cpu');assert not self.p.spec['recipe'].get('fusion')
        f=importlib.import_module('frontend_v3');self.timer=None;original=f.resample;samples=f.RunningSide.update_samples;frames=f.RunningSide.update_frames;fft=self.torch.fft.rfft
        def wrap(fn,stage):
            def call(*a,**k):
                if self.timer is None:return fn(*a,**k)
                with self.timer(stage):return fn(*a,**k)
            return call
        f.resample=wrap(original,'R');f.RunningSide.update_samples=wrap(samples,'G');f.RunningSide.update_frames=wrap(frames,'G');self.torch.fft.rfft=wrap(fft,'F')
        self.metadata=dict(frozen_folder=str(self.p.folder),candidate=self.candidate,precision='frozen CPU fp32',stages='exclusive nested timing; G raw level/side statistics; F mainSTFT/mel/log/norm and actualunused48kbandFFT; R actualfrozenpolyphase; O tensor/formatcopy; I wholeGRU/pool/head',torch_threads=self.torch.get_num_threads())
    def process(self,path,timer):
        import soundfile as sf
        self.timer=timer
        try:
            with timer('L'):w,sr=sf.read(path,dtype='float32')
            with timer('O'):w=w.mean(1) if w.ndim==2 else w
            with timer('F'):f=self.p.front.extract_features(w,sr)
            with timer('O'):
                x=self.torch.tensor(f['logmel'])[None];mask=self.torch.ones(x.shape[:2],dtype=self.torch.bool);x=(x-self.p.mean)/self.p.std;side=self.p.normalize_side(f['side'])
            with timer('I'),self.torch.inference_mode():y=self.p.model(x,mask,side).cpu().numpy()
            with timer('O'):return prediction(y)
        finally:self.timer=None
    def reference(self,path):return prediction(self.p.predict_file(path,mapped=False))
class Runner64(Runner96):name='v2_64_native';candidate='v2_64'
