"""Count-adaptive calibration policies and declared checkpoint baselines."""
from functools import lru_cache
from fractions import Fraction
import numpy as np
from scipy.stats import binom,beta
import reference
from certificates import bin_index

@lru_cache(maxsize=128)
def schedule(cap,eps,delta):
    m=np.arange(1,cap+1);k=binom.ppf(delta,m,eps).astype(int)
    ranks=np.r_[0,np.minimum(k+(binom.cdf(k,m,eps)<=delta),m)]
    jumps=np.flatnonzero(np.diff(ranks)>0)+1
    return np.array([(int(ranks[v]),int(v)) for v in jumps],dtype=int).reshape(-1,2)

@lru_cache(maxsize=16384)
def beta_nodes(j,m):return beta.ppf((np.arange(64)+.5)/64,j,m-j+1)

class Predictor:
    def __init__(self,scores,mode):
        self.mode=mode;self.empty=False
        if mode=='count':return
        order=np.argsort(scores,kind='stable');s=scores[order];N=len(s)
        w=np.ones(N) if mode=='attachment-flat' else s.copy()
        assert np.isfinite(w).all() and (w>=0).all() and (w<=1).all()
        if w.sum()==0:self.empty=True;return
        self.u=np.linspace(0,1,4001);cdf=np.cumsum(w)/w.sum()
        ix=np.searchsorted(cdf,self.u,'left').clip(0,N-1)
        self.phi=np.searchsorted(s,s[ix],'left')/N
    def values(self,sched):
        if self.mode=='count' or self.empty:return sched[:,0]/(sched[:,1]+1)
        return np.array([np.interp(beta_nodes(int(j),int(m)),self.u,self.phi).mean() for j,m in sched])

def ev_stop(label_path,cap,sched,values):
    hits=np.flatnonzero(label_path[:cap])+1
    for a,(j,m) in enumerate(sched):
        if m>len(hits):break
        n=int(hits[m-1]);phat=(m+1)/(n+2)
        future=n+(sched[a+1:,1]-m)/phat
        feasible=future<=cap
        future_values=(1-future[feasible]/len(label_path))*values[a+1:][feasible]
        now=(1-n/len(label_path))*values[a]
        if not len(future_values) or now>=future_values.max():
            return n,int(m),'value-stop'
    return cap,int(label_path[:cap].sum()),'cap'

def source_target_K(rate,N,grid):
    eligible=[k for k in grid if rate is not None and rate>0 and k/(rate*N)<=.2]
    return max(eligible) if eligible else grid[0]

def target_stop(path,K,cap):
    hits=np.flatnonzero(path[:cap])+1
    if len(hits)>=K:return int(hits[K-1]),int(K),'target'
    return cap,int(len(hits)),'cap'

def certify(scores,labels,perm,n,m,eps,delta,rule='FPC',fallback=False):
    j=0 if fallback else (reference.jstar_fpc(m,len(labels),n,eps,delta) if rule=='FPC' else bin_index(m,eps,delta))
    if j==0:return dict(n=n,m=m,j=0,tau=float('-inf'),skipped=0,hidden=0)
    cal=perm[:n];s=scores[cal];y=labels[cal];failure=s[y==1]
    tau=float(np.partition(failure,j-1)[j-1])
    # Strict comparisons avoid counting ties as omitted; calibration never saves executions.
    skipped=int(np.count_nonzero(scores<tau)-np.count_nonzero(s<tau))
    hidden=int(np.count_nonzero((scores<tau)&(labels==1))-np.count_nonzero((s<tau)&(y==1)))
    return dict(n=n,m=m,j=j,tau=tau,skipped=skipped,hidden=hidden)

def count_checkpoint(path,N,cap,eps,delta):
    for f in [.05,.1,.2,.3]:
        n=min(round(f*N),cap);m=int(path[:n].sum())
        if reference.jstar_fpc(m,N,n,eps,delta)>0 or n==cap:return n,m,'count-checkpoint' if n<cap else 'cap'

def score_checkpoint(scores,labels,perm,cap,eps,delta):
    N=len(labels)
    for f in [.05,.1,.2,.3]:
        n=min(round(f*N),cap);m=int(labels[perm[:n]].sum())
        result=certify(scores,labels,perm,n,m,eps,delta/4)
        if (result['j']>0 and result['skipped']/N>=.1) or n==cap:
            return result,'score-checkpoint' if n<cap else 'cap'
