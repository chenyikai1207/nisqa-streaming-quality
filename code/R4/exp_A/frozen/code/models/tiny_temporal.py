import torch
from torch import nn

class TinyTemporal(nn.Module):
    def __init__(self,kind='cnn',with_b=False):
        super().__init__();self.kind=kind;self.with_b=with_b;c=49 if with_b else 48
        if kind=='cnn':
            layers=[nn.Conv1d(c,32,1),nn.GELU()]
            for _ in range(3):layers.extend([nn.Conv1d(32,32,5,padding=2,groups=32),nn.Conv1d(32,32,1),nn.GELU()])
            self.encoder=nn.Sequential(*layers);h=32
        else:self.encoder=nn.GRU(c,32,1,batch_first=True);h=32
        self.attention=nn.Linear(h,1);self.head=nn.Linear(h,5)
    def forward(self,x,mask):
        if self.kind=='cnn':z=self.encoder(x.transpose(1,2)).transpose(1,2)
        else:z,_=self.encoder(x)
        score=self.attention(z).squeeze(-1).masked_fill(~mask,-1e9);a=torch.softmax(score,1)
        return self.head((z*a[:,:,None]).sum(1))

if __name__=='__main__':
    for k in ['cnn','gru']:
        for b in [False,True]:
            m=TinyTemporal(k,b);n=sum(p.numel() for p in m.parameters());assert n<=50000
            print(k,b,n)
