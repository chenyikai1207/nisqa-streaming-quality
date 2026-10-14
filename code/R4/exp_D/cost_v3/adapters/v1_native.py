"""Frozen v1 CPU forward; exclusive R=originalpolyphase, G=originalpeak, F=originalmel/log."""
from .paths import STAGE,add_code,configure_torch,prediction
from .base import StageTimer
import contextlib
class Runner:
    name='v1_native'
    def __init__(self):
        add_code(STAGE/'exp_A/frozen/code');import runtime,features_t2
        self.torch=configure_torch();default_threads=self.torch.get_num_threads();self.api=runtime;self.p=runtime.Predictor(STAGE/'exp_A/frozen/A_distill','cpu')
        if not __import__('os').environ.get('NISQA_COST_CORE'):self.torch.set_num_threads(default_threads)
        self.timer=None;orig=runtime.peak_prepare;rs=features_t2.resample_poly
        def resample(*a,**k):
            if self.timer is None:return rs(*a,**k)
            with self.timer('R'):return rs(*a,**k)
        def peak(*a,**k):
            if self.timer is None:return orig(*a,**k)
            with self.timer('G'):return orig(*a,**k)
        features_t2.resample_poly=resample;runtime.peak_prepare=peak
        self.metadata=dict(frozen_folder=str(self.p.folder),precision='CPU fp32 originalruntime',stages='L decode+mono; R originalresample; G peak normalization; F librosa mel/log; I originalmodel (includes normalized inputcopy); O outputformat',threads=self.torch.get_num_threads())
    def process(self,path,timer):
        self.timer=timer
        try:
            with timer('L'):w,sr=self.p.load_audio(path)
            with timer('F'):f=self.p.extract_features(w,sr)
            with timer('I'):y=self.p.infer(f)
            with timer('O'):return prediction(y)
        finally:self.timer=None
    def reference(self,path):
        w,sr=self.p.load_audio(path);return prediction(self.p.infer(self.p.extract_features(w,sr)))
