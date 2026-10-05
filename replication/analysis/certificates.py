from functools import lru_cache
from fractions import Fraction
import numpy as np
from scipy.stats import hypergeom,binom
import reference

# Exact prefix maxima at block endpoints. This changes computation, not the rule.
@lru_cache(maxsize=2048)
def prefix(j,m,eps):
    en,ed=eps.numerator,eps.denominator
    ks=np.arange(m*en//ed,40000*en//ed+1,dtype=np.int64)
    endpoints=((ks+1)*ed-1)//en
    q=j+ks
    vals=np.zeros(len(ks)); ok=(endpoints>=m)&(q<endpoints)
    vals[ok]=hypergeom.cdf(j-1,endpoints[ok],q[ok],m)
    return endpoints,np.maximum.accumulate(vals)

def fast_b(j,m,M_lo,M_hi,eps):
    if M_lo!=m or M_hi>40000:return ORIGINAL_B(j,m,M_lo,M_hi,eps)
    endpoints,vals=prefix(j,m,eps);i=int(np.searchsorted(endpoints,M_hi,side='right'))
    best=float(vals[i-1]) if i else 0.
    if i and endpoints[i-1]==M_hi:return best
    q=j+(M_hi*eps.numerator)//eps.denominator
    return max(best,float(hypergeom.cdf(j-1,M_hi,q,m)) if q<M_hi else 0.)

ORIGINAL_B=reference._b_j
reference._b_j=fast_b

@lru_cache(maxsize=32768)
def bin_index(m,eps=.05,delta=.1):
    lo,hi=0,m
    while lo<hi:
        j=(lo+hi+1)//2
        if binom.cdf(j-1,m,eps)<=delta:lo=j
        else:hi=j-1
    return lo

def validate(scores,labels,perm,n=None,B=None):
    N=len(scores)
    if np.asarray(scores).shape!=(N,) or not np.isfinite(scores).all():raise ValueError('invalid scores')
    if np.asarray(labels).shape!=(N,) or not np.isin(labels,[0,1]).all():raise ValueError('invalid labels')
    if not np.array_equal(np.sort(perm),np.arange(N)):raise ValueError('invalid permutation')
    if n is not None and not 0<n<=N:raise ValueError('invalid calibration size')
    if B is not None and not n<=B<=N:raise ValueError('calibration exceeds budget')

def index(m,N,n,eps,delta,rule):
    if rule=='FPC':return reference.jstar_fpc(m,N,n,eps,delta)
    if rule=='Binomial':return bin_index(m,eps,delta)
    raise ValueError('unknown rule')

def decision(scores,labels,perm,n,eps,delta,rule):
    c=perm[:n];m=int(labels[c].sum());j=index(m,len(scores),n,eps,delta,rule)
    skip,tau=reference._threshold_skip(scores,labels,c,j)
    return dict(m=m,j=j,tau=tau,n_cal=n,skip=skip)

def inverse(scores,labels,perm,K,eps,delta,cap,rule='FPC'):
    if not 1<=K or not 1<=cap<=len(scores):raise ValueError('invalid inverse design')
    hits=np.flatnonzero(labels[perm[:cap]]==1)
    if len(hits)<K:return dict(m=int(labels[perm[:cap]].sum()),j=0,tau=-np.inf,n_cal=cap,skip=np.zeros(len(scores),bool),cap_hit=1)
    n=int(hits[K-1])+1;result=decision(scores,labels,perm,n,eps,delta,rule);result['cap_hit']=0
    return result

def check_backend():
    rng=np.random.default_rng(41)
    for _ in range(250):
        N=int(rng.integers(10,18000));n=int(rng.integers(1,N));m=int(rng.integers(1,min(n,200)+1));j=int(rng.integers(1,m+1));eps=Fraction(str(rng.choice([.01,.02,.05,.1])))
        a=fast_b(j,m,m,m+N-n,eps);b=ORIGINAL_B(j,m,m,m+N-n,eps)
        assert abs(a-b)<1e-13,(N,n,m,j,eps,a,b)
    return {'exact_backend_comparisons':250,'passed':True}
