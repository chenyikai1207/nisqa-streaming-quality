"""Current model v2 candidate structures, strictly past-frame encoders and incremental attention."""
import torch
from torch import nn
from torch.nn import functional as F
class DSConv(nn.Module):
    def __init__(self,inputs,outputs,kernel=3,dilation=1):
        super().__init__();self.context=(kernel-1)*dilation;self.dw=nn.Conv1d(inputs,inputs,kernel,groups=inputs,dilation=dilation);self.pw=nn.Conv1d(inputs,outputs,1)
    def forward(self,x):return F.gelu(self.pw(self.dw(F.pad(x,(self.context,0)))))
class Model(nn.Module):
    def __init__(self,kind='gru',width=32,layers=1,kernel=3,input_channels=48,side_dim=0):
        super().__init__();self.kind=kind;self.width=width;self.layers=layers;self.side_dim=side_dim
        if kind=='gru':self.encoder=nn.GRU(input_channels,width,layers,batch_first=True)
        elif kind=='dsconv_gru':
            self.convs=nn.Sequential(DSConv(input_channels,width,kernel),DSConv(width,width,kernel));self.encoder=nn.GRU(width,width,1,batch_first=True)
        elif kind=='tcn':self.encoder=nn.Sequential(nn.Conv1d(input_channels,width,1),nn.GELU(),*[DSConv(width,width,3,dilation=d) for d in [1,2,4,8]])
        else:raise ValueError(kind)
        self.attention=nn.Linear(width,1);self.head=nn.Linear(width+side_dim,5)
    def encode(self,x):
        if self.kind=='gru':return self.encoder(x)[0]
        if self.kind=='dsconv_gru':return self.encoder(self.convs(x.transpose(1,2)).transpose(1,2))[0]
        return self.encoder(x.transpose(1,2)).transpose(1,2)
    def forward(self,x,mask,side=None):
        z=self.encode(x);weights=torch.softmax(self.attention(z).squeeze(-1).masked_fill(~mask,-1e9),1);pooled=(weights[:,:,None]*z).sum(1)
        if self.side_dim:pooled=torch.cat([pooled,side],1)
        return self.head(pooled)
def architecture_grid(input_channels=48,side_dim=0):
    configs=[]
    for width in [32,48,64,96,128]:
        for layers in [1,2]:configs.append(dict(candidate=f'A2V3_gru{width}_layers{layers}',kind='gru',width=width,layers=layers,kernel=3))
        for kernel in [3,5]:configs.append(dict(candidate=f'A2V3_dsconvK{kernel}_gru{width}',kind='dsconv_gru',width=width,layers=1,kernel=kernel))
        configs.append(dict(candidate=f'A2V3_tcn{width}',kind='tcn',width=width,layers=1,kernel=3))
    for cfg in configs:
        cfg.update(input_channels=input_channels,side_dim=side_dim,alpha=.7,seeds=[0,1,2]);model=Model(**{k:cfg[k] for k in ['kind','width','layers','kernel','input_channels','side_dim']});cfg['parameters']=sum(p.numel() for p in model.parameters())
    return configs
class IncrementalEncoder:
    """Finite GRU/context/attention state; frontend/resampler state supplied separately."""
    def __init__(self,model):
        self.model=model;self.hidden=None;self.conv_state={};self.maximum=None;self.denominator=None;self.numerator=None
    def convolution(self,key,module,x):
        buffer=self.conv_state.get(key)
        if buffer is None:buffer=x.new_zeros((1,x.shape[1],module.context))
        full=torch.cat([buffer,x],dim=-1);self.conv_state[key]=full[:,:,-module.context:].clone() if module.context else full[:,:,:0];return F.gelu(module.pw(module.dw(full)))
    @torch.inference_mode()
    def push_frames(self,x):
        if x.ndim==2:x=x[None,:]
        if not x.shape[1]:return
        m=self.model
        if m.kind=='gru':z,self.hidden=m.encoder(x,self.hidden)
        elif m.kind=='dsconv_gru':
            h=x.transpose(1,2)
            for i,layer in enumerate(m.convs):h=self.convolution(f'ds{i}',layer,h)
            z,self.hidden=m.encoder(h.transpose(1,2),self.hidden)
        else:
            h=F.gelu(m.encoder[0](x.transpose(1,2)))
            for i,layer in enumerate(m.encoder[2:]):h=self.convolution(f'tcn{i}',layer,h)
            z=h.transpose(1,2)
        scores=m.attention(z).squeeze(-1);block_max=scores.max(1).values;maximum=block_max if self.maximum is None else torch.maximum(self.maximum,block_max);weights=torch.exp(scores-maximum[:,None]);denominator=weights.sum(1);numerator=(weights[:,:,None]*z).sum(1)
        if self.maximum is not None:
            factor=torch.exp(self.maximum-maximum);denominator+=self.denominator*factor;numerator+=self.numerator*factor[:,None]
        self.maximum=maximum;self.denominator=denominator;self.numerator=numerator
    @torch.inference_mode()
    def predict(self,side=None):
        pooled=self.numerator/self.denominator[:,None]
        if self.model.side_dim:pooled=torch.cat([pooled,side],1)
        return self.model.head(pooled)
    def state_bytes(self):
        arrays=list(self.conv_state.values())+[x for x in [self.hidden,self.maximum,self.denominator,self.numerator] if x is not None]
        return sum(x.numel()*x.element_size() for x in arrays)
