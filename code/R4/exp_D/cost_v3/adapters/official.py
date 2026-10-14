from public_paths import data_path, package_path
"""Frozen official comparators, original full calls. Unseparable official frontend is explicitly merged."""
import os,sys,io,contextlib,json,math
import numpy as np
from .paths import STAGE,D2,COST,add_code,configure_torch,prediction
class _Base:
    def __init__(self):
        self.torch=configure_torch();self.timer=None
        add_code(STAGE/'exp_A/frozen/code');import legacy_frontend
        self.decode=legacy_frontend.decode
    def reference(self,path):
        from .base import StageTimer
        return self.process(path,StageTimer())
class NISQAMOS(_Base):
    name='nisqa_mos';full=False
    def __init__(self):
        super().__init__();import pandas as pd
        off=PathOFF();add_code(off);from nisqa.NISQA_model import nisqaModel
        import librosa.core.audio as audio
        original=getattr(audio,'__soundfile_load');rs=audio.resample
        def wrap(fn,stage):
            def call(*a,**k):
                if self.timer is None:return fn(*a,**k)
                with self.timer(stage):return fn(*a,**k)
            return call
        setattr(audio,'__soundfile_load',wrap(original,'L'));audio.resample=wrap(rs,'R')
        paths=[x.strip() for x in open(os.environ['NISQA_COST_FILES'],encoding='utf8') if x.strip()];self.lookup={p:i for i,p in enumerate(paths)}
        index=COST/'bench/checks'/('official_index_'+str(os.getpid())+'.csv');index.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(dict(file=[__import__('pathlib').Path(p).name for p in paths],path=paths)).to_csv(index,index=False)
        weights=off/'weights/nisqa.tar' if self.full else STAGE/'exp_C/weights/nisqa_mos_only.tar';old=self.torch.load
        def load(*a,**k):k.setdefault('weights_only',False);return old(*a,**k)
        self.torch.load=load
        try:
            with contextlib.redirect_stdout(io.StringIO()):self.m=nisqaModel(dict(mode='predict_csv',pretrained_model=str(weights),data_dir='',csv_file=str(index),csv_deg='path',output_dir=str(COST/'bench/checks'),tr_num_workers=0,tr_bs_val=1,tr_parallel=False,tr_device='cpu',ms_channel=None))
        finally:self.torch.load=old
        self.m.model.eval();self.metadata=dict(weights=str(weights),stages='officialdataset frontend F merges authorlevel handling/STFT/mel; decodeL/resampleR nestedexclusive; officialwholemodel I',parameters=sum(p.numel() for p in self.m.model.parameters()),official_output_heads=5 if self.full else 1)
    def process(self,path,timer):
        self.timer=timer
        try:
            with timer('F'):xb,_,(_,nw)=self.m.ds_val[self.lookup[path]]
            with timer('O'):x=xb.unsqueeze(0);n=self.torch.as_tensor(nw).reshape(1)
            with timer('I'),self.torch.inference_mode():y=self.m.model(x,n).cpu().numpy()
            with timer('O'):return dict(zip(['mos','noi','dis','col','loud'] if self.full else ['mos'],map(float,np.asarray(y).reshape(-1))))
        finally:self.timer=None
    def reference(self,path):
        xb,_,(_,nw)=self.m.ds_val[self.lookup[path]]
        with self.torch.inference_mode():y=self.m.model(xb.unsqueeze(0),self.torch.as_tensor(nw).reshape(1)).numpy()
        return dict(zip(['mos','noi','dis','col','loud'] if self.full else ['mos'],map(float,np.asarray(y).reshape(-1))))
class NISQAFull(NISQAMOS):name='nisqa_full';full=True
def PathOFF():
    from pathlib import Path
    return Path(str(data_path('third_party/LANE_C_MODELS/OFFICIAL_NISQA'))) if os.name=='nt' else STAGE/'legacy/R2R3_EXECUTION/LANE_C_MODELS/OFFICIAL_NISQA'
class PLCMOS(_Base):
    name='plcmos'
    def __init__(self):
        super().__init__();import onnxruntime as ort,torchaudio
        self.ta=torchaudio;weights=STAGE/'exp_B/official/PLC_WEIGHT.onnx';o=ort.SessionOptions();o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        if os.environ.get('NISQA_COST_CORE'):o.intra_op_num_threads=1;o.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(weights),sess_options=o,providers=['CPUExecutionProvider']);ns={'__file__':str(STAGE/'exp_B/official/plc_mos.py')};exec((STAGE/'exp_B/official/PLC_CODE.txt').read_text(encoding='utf8'),ns);self.feature=ns['PLCMOSEstimator'].__new__(ns['PLCMOSEstimator'])
        self.metadata=dict(weights=str(weights),calls=15,stages='L mono decode; R originaltorchaudio16k; F authorSTFT once; I all15fullofficialcalls noencoderreuse; O peraudio seed23/15rater generation+float32conversion/mean',providers=self.session.get_providers(),rater_embedding_precomputed=False)
    def process(self,path,timer):
        with timer('L'):x,sr=self.decode(path)
        with timer('R'):
            if sr!=16000:x=self.ta.functional.resample(self.torch.from_numpy(x),sr,16000).numpy()
        with timer('F'):data=np.float32(self.feature.stft_transform(x))[None,None]
        with timer('O'):np.random.seed(23);v=[]
        for _ in range(15):
            with timer('O'):e=np.array(np.random.normal(size=(1,64)),dtype=np.float32).reshape(1,-1)
            with timer('I'):value=self.session.run(None,{'degraded_audio':data,'rater_embed':e})[0]
            with timer('O'):v.append(float(value.item()))
        with timer('O'):return {'mos':float(np.mean(v))}
class DNSMOS(PLCMOS):
    name='dnsmos'
    def __init__(self):
        _Base.__init__(self);import onnxruntime as ort,torchaudio
        self.ta=torchaudio;w=STAGE/'exp_B/official/DNSMOS_WEIGHT.onnx';o=ort.SessionOptions();o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        if os.environ.get('NISQA_COST_CORE'):o.intra_op_num_threads=o.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(w),o,providers=['CPUExecutionProvider']);self.metadata=dict(weights=str(w),stages='official9.01s tiling; I includes originalP835 polynomial, fullcalls')
    def process(self,path,timer):
        with timer('L'):x,sr=self.decode(path)
        with timer('R'):
            if sr!=16000:x=self.ta.functional.resample(self.torch.from_numpy(x),sr,16000).numpy()
        with timer('F'):
            while len(x)<144160:x=np.tile(x,2)
            chunks=[x[i*16000:i*16000+144160][None].astype(np.float32) for i in range(int(np.floor(len(x)/16000)-9.01)+1)]
        with timer('I'):
            vals=[]
            for x in chunks:
                sig,bak,ovr=self.session.run(None,{'input_1':x})[0][0];vals.append((np.polyval([-.06766283,1.11546468,.04602535],ovr),np.polyval([-.13166888,1.60915514,-.39604546],bak)))
        with timer('O'):v=np.mean(vals,axis=0);return {'mos':float(v[0]),'noi':float(v[1])}
class SQUIM(_Base):
    name='squim'
    def __init__(self):
        super().__init__();import torchaudio
        self.ta=torchaudio;self.torch.hub.set_dir(str(D2/'torch_hub'));self.model=torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval();self.metadata=dict(stages='officialthreeheadscomputed; PESQindex1 proxy reportedasfrozenmosoutput; internalfrontendandNN mergedI')
    def process(self,path,timer):
        with timer('L'):x,sr=self.decode(path)
        with timer('O'):x=self.torch.from_numpy(x)
        with timer('R'):
            if sr!=16000:x=self.ta.functional.resample(x,sr,16000)
        with timer('I'),self.torch.inference_mode():y=self.model(x[None])[1].numpy()
        with timer('O'):return {'mos':float(np.asarray(y).reshape(-1)[0])}
