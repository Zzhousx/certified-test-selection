from functools import lru_cache
import numpy as np
import reference

def bounds(m,x,N,n,Lu,delta=.1,ratio=1.005):
    """Two conservative bounds, sharing the reference's endpoint inversions.

    The refined bound retains the lower feasibility constraint. For an interior
    integer M in (a,b), accepted hidden failures <= Kmax(b)-x and <= Lu; their
    ratio is <= min(Kmax(b)-x,Lu)/(a+1). A block is empty if its smallest required
    hidden count, max(0,a+1-m-(N-n-Lu)), exceeds Kmax(b)-x. Endpoints are treated
    exactly. These bounds enclose the exact confidence-set supremum.
    """
    if Lu==0:return 0.,0.
    if m==0:return 1.,1.
    Kb,Ga,K0,G0=reference._grid_blocks(m,x,N,n,delta,ratio)
    G=np.r_[Ga,m+N-n];K=np.r_[K0,Kb]
    hidden=np.minimum(K,Lu);lower=np.maximum(0,G-m-(N-n-Lu))
    possible=hidden>=lower
    endpoint=float(np.max(hidden[possible]/G[possible])) if possible.any() else 0.
    if len(G)>1:
        amin=G[:-1]+1;b=G[1:];up=np.minimum(Kb,Lu)
        keep=(b>amin)&(up>=np.maximum(0,amin-m-(N-n-Lu)))
        interior=float(np.max(up[keep]/amin[keep])) if keep.any() else 0.
        legacy=float(max(min(K0,Lu)/G0,np.max(np.minimum(Kb,Lu)/Ga)))
    else:interior=0.;legacy=endpoint
    return min(1.,max(endpoint,interior)),min(1.,legacy)

def validate():
    checks=0;max_gap=0.
    for N in [5,8,15,60]:
        for n in [1,max(1,N//3),N-1]:
            for m in range(1,min(n,5)+1):
                for x in range(m+1):
                    for Lu in sorted(set([0,1,(N-n)//2,N-n])):
                        a,b=bounds(m,x,N,n,Lu)
                        exact=reference.ucb_miss_share(m,x,N,n,Lu,.1)
                        assert a+1e-12>=exact,(m,x,N,n,Lu,a,exact)
                        assert b+1e-12>=a,(m,x,N,n,Lu,a,b)
                        checks+=1;max_gap=max(max_gap,a-exact)
    return dict(passed=True,configurations=checks,max_small_grid_gap=max_gap)
