"""Quality backbone, masked auxiliary heads, direct/gated causal time descriptors."""
import torch
from torch import nn
from torch.nn import functional as F
from models_v3 import Model,IncrementalEncoder
class PostModel(Model):
    def __init__(self,codec_classes=4,auxiliary=False,ablation='full',fusion=None,base_side_dim=4,time_dim=35,aux_strategy='source_scope',**kwargs):
        super().__init__(side_dim=base_side_dim+(time_dim if fusion else 0),**kwargs);self.base_side_dim=base_side_dim;self.time_dim=time_dim;self.fusion=fusion;self.auxiliary=auxiliary;self.ablation=ablation;self.capture=None;self.aux_strategy=aux_strategy
        if fusion in ['gated','conditional_gated']:self.gate=nn.Linear(time_dim,time_dim);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        if auxiliary:
            self.frame_head=nn.Linear(self.width,1);self.reg_head=nn.Linear(self.width,2);self.class_heads=nn.ModuleDict({'PLC':nn.Linear(self.width,4),'BW':nn.Linear(self.width,4),'codec':nn.Linear(self.width,codec_classes),'SNR':nn.Linear(self.width,4)})
    def quality(self,pool,side):
        if self.fusion in ['gated','conditional_gated']:
            raw=side[:,self.base_side_dim:];time=torch.sigmoid(self.gate(raw))*raw
            if self.fusion=='conditional_gated':
                valid=raw[:,-1:].ge(.5).to(raw.dtype)
                time=torch.cat([time[:,:30],time[:,30:34]*valid,raw[:,-1:]],1)
            side=torch.cat([side[:,:self.base_side_dim],time],1)
        if self.side_dim:pool=torch.cat([pool,side],1)
        return self.head(pool)
    def forward(self,x,mask,side=None):
        z=self.encode(x);w=torch.softmax(self.attention(z).squeeze(-1).masked_fill(~mask,-1e9),1);pool=(w[:,:,None]*z).sum(1)
        aux_pool=(z*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None] if self.aux_strategy=='low_level' else pool
        self.capture=(z,aux_pool,mask) if self.auxiliary else None;return self.quality(pool,side)
    def auxiliary_loss(self,targets):
        if not self.auxiliary:return self.head.weight.sum()*0
        z,pool,mask=self.capture;parts=[];weight=targets.get('aux_weight',torch.ones(len(pool),device=pool.device))
        def weighted(values,ww):
            return (values*ww).sum()/ww.sum().clamp_min(1e-12)
        if self.ablation!='no_frame':
            good=targets['frame_valid']&mask&weight[:,None].gt(0)
            if good.any():parts.append(weighted(F.binary_cross_entropy_with_logits(self.frame_head(z).squeeze(-1)[good],targets['frame'][good],reduction='none'),weight[:,None].expand_as(mask)[good]))
        if self.ablation!='no_sentence':
            r=self.reg_head(pool)
            for name,pred in [('rate',torch.sigmoid(r[:,0])),('logBr',r[:,1])]:
                good=torch.isfinite(targets[name])&weight.gt(0)
                if good.any():parts.append(weighted(F.huber_loss(pred[good],targets[name][good],delta=1.,reduction='none'),weight[good]))
            for name,head in self.class_heads.items():
                good=targets[name].ge(0)&weight.gt(0)
                if good.any():parts.append(weighted(F.cross_entropy(head(pool)[good],targets[name][good],reduction='none'),weight[good]))
        return torch.stack(parts).mean() if parts else pool.sum()*0
def load_warm(model,path):
    state=torch.load(path,map_location='cuda',weights_only=False)['state_dict'];current=model.state_dict()
    for key,value in state.items():
        if key not in current:continue
        if value.shape==current[key].shape:current[key]=value
        elif key=='head.weight' and value.shape[1]<current[key].shape[1]:current[key].zero_();current[key][:,:value.shape[1]]=value
    model.load_state_dict(current)
class PostIncrementalEncoder(IncrementalEncoder):
    @torch.inference_mode()
    def predict(self,side=None):return self.model.quality(self.numerator/self.denominator[:,None],side)
def ccc_loss(pred,target,weights):
    terms=[];ww=[]
    for j in range(pred.shape[1]):
        ok=torch.isfinite(target[:,j]);x=pred[ok,j];y=target[ok,j]
        if len(y)<2 or y.var(correction=0)<=1e-8:continue
        mx,my=x.mean(),y.mean();vx,vy=x.var(correction=0),y.var(correction=0);cov=((x-mx)*(y-my)).mean();terms.append(1-2*cov/(vx+vy+(mx-my).square()+1e-8));ww.append(weights[j])
    return (torch.stack(terms)*torch.stack(ww)).sum()/torch.stack(ww).sum() if terms else pred.sum()*0
