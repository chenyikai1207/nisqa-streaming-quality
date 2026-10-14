"""Runtime application of fixed TRAIN OOF maps. No target/audio/table reads."""
import numpy as np
def monotonic(coef,lo,hi):
    d=np.polyder(coef);points=[lo,hi]
    if len(d)==3 and d[0]!=0:
        vertex=-d[1]/(2*d[0])
        if lo<vertex<hi:points.append(vertex)
    return np.min(np.polyval(d,points))>=0
def apply_mapping(x,m):
    x=np.asarray(x)
    if m['used']=='none':return x.copy()
    if m['used']=='isotonic':return np.interp(x,m['x'],m['y'])
    coef=m['coef']
    if m['used']=='cubic':
        finite=x[np.isfinite(x)]
        if len(finite) and not monotonic(coef,min(m['prediction_range'][0],float(finite.min())),max(m['prediction_range'][1],float(finite.max()))):coef=m['linear_coef']
    return np.polyval(coef,x)
