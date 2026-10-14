"""Exclusive monotonic L/R/G/F/I/O nanoseconds; nested child time is excluded from parent."""
import time
STAGES=('L','R','G','F','I','O')
class StageTimer:
    def __init__(self):self.acc={k:0 for k in STAGES};self.stack=[]
    def __call__(self,name):
        if name not in self.acc:raise ValueError(name)
        return _Span(self,name)
class _Span:
    def __init__(self,t,n):self.t=t;self.n=n
    def __enter__(self):self.start=time.perf_counter_ns();self.children=0;self.t.stack.append(self);return self
    def __exit__(self,*exc):
        elapsed=time.perf_counter_ns()-self.start;self.t.stack.pop();self.t.acc[self.n]+=elapsed-self.children
        if self.t.stack:self.t.stack[-1].children+=elapsed
class Runner:
    name='base'
    def process(self,path,timer):raise NotImplementedError
