"""Independent unselected boundedrepaircandidate1 adapter; original frozenarrayvalues, same stagecontract."""
import numpy as np,soundfile as sf
from deploy_v2.fir_torch_whole_candidate1 import Backend
from .paths import prediction,configure_torch
class Runner96F32:
    candidate='v2_pure';precision='float32'
    def __init__(self):
        configure_torch();self.b=Backend(self.candidate,self.precision);self.metadata=dict(implementation_version='candidate1',candidate=self.candidate,precision=self.precision,implementation='candidate1 offline scipy.upfirdn exactcachedcoeff/offset, whole TorchGRU unchangedweights, NumPyattention/head; originalNumPystream unchanged',weights_changed=False,frontend_parameters_changed=False,status='unselected pendingactualbounded2 then newqualificationifrootchooses')
    def process(self,path,timer):
        with timer('L'):w,sr=sf.read(path,dtype='float32')
        with timer('O'):w=w.mean(1) if w.ndim==2 else w
        with timer('R'):x,sidewave=self.b.resample(w,sr)
        with timer('G'):side=self.b.level(w,sidewave)
        with timer('F'):features=self.b.features(x)
        with timer('I'):y=self.b.infer(features,side)
        with timer('O'):return prediction(y)
class Runner64F32(Runner96F32):candidate='v2_64'
class Runner96F64(Runner96F32):precision='float64'
class Runner64F64(Runner64F32):precision='float64'
