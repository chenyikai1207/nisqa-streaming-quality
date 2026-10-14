"""Candidate NumPy whole deployment, not selected until registered TRAIN200 numerical gate andclean diagnosis."""
import numpy as np,soundfile as sf
from deploy_v2.numpy_backend import Backend
from .paths import prediction
class Runner96F64:
    name='v2_96_deploy_f64';candidate='v2_pure';precision='float64'
    def __init__(self):
        self.b=Backend(self.candidate,self.precision);self.metadata=dict(candidate=self.candidate,precision=self.precision,weights='unchanged frozen f32 scalarvalues',frontend='original FIR coefficients, full vectorized fp32 STFT/mel/norm, exactleveldefs, band0 unusedFFT omitted',implementation='genuine whole-sequence featurematrix and layer-input affine; boundedstream separately',qualification='candidate; TRAIN200 gate required beforeformal deployment choice')
    def process(self,path,timer):
        with timer('L'):w,sr=sf.read(path,dtype='float32')
        with timer('O'):w=w.mean(1) if w.ndim==2 else w
        with timer('R'):x,sidewave=self.b.resample(w,sr)
        with timer('G'):side=self.b.level(w,sidewave)
        with timer('F'):features=self.b.features(x)
        with timer('I'):y=self.b.infer(features,side)
        with timer('O'):return prediction(y)
    def reference(self,path):
        # Independent frozenCPU reference; imported only inengineering qualifier, never clean deploymenttiming.
        from .v2_native import Runner96,Runner64
        ref=Runner96() if self.candidate=='v2_pure' else Runner64();return ref.reference(path)
class Runner64F64(Runner96F64):name='v2_64_deploy_f64';candidate='v2_64'
class Runner96F32(Runner96F64):name='v2_96_deploy_f32';precision='float32'
class Runner64F32(Runner64F64):name='v2_64_deploy_f32';precision='float32'
