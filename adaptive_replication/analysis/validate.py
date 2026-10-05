import sys,os,json,hashlib,time
sys.modules['bottleneck']=None
from pathlib import Path
from fractions import Fraction as F
from math import comb
from functools import lru_cache
from itertools import combinations
import numpy as np
import reference,certificates
from policies import *
R=Path(__file__).resolve().parents[1]
(R/'audit').mkdir(exist_ok=True)
start=time.perf_counter();checks={}
for eps,delta in [(0.05,.1),(.05,.05),(.1,.1),(.25,.4),(.5,.25)]:
    sc=schedule(300,eps,delta);ranks=dict(sc[:,[1,0]])
    last=0
    for m in range(1,301):
        if m in ranks:last=int(ranks[m])
        assert last==reference.jstar_binomial(m,eps,delta)==certificates.bin_index(m,eps,delta)
checks['binomial_schedule_states']=1500
assert [int(schedule(100,*x)[0,1]) for x in [(.05,.1),(.05,.05),(.1,.1)]]==[45,59,22]
rg=np.random.default_rng(17);N=700;s=rg.random(N);y=(rg.random(N)<.3).astype(int);cap=210;sc=schedule(cap,.05,.1)
for mode in ['count','score','attachment-flat']:
    vals=Predictor(s,mode).values(sc)
    for z in range(30):
        perm=rg.permutation(N);path=y[perm];n,m,reason=ev_stop(path,cap,sc,vals)
        changed=path.copy();changed[n:]=rg.permutation(changed[n:]);assert ev_stop(changed,cap,sc,vals)==(n,m,reason)
        fp=np.flatnonzero(path);pp=np.flatnonzero(1-path);other=perm.copy();other[fp]=rg.permutation(perm[fp]);other[pp]=rg.permutation(perm[pp]);assert np.array_equal(path,y[other])
        assert ev_stop(y[other],cap,sc,vals)==(n,m,reason)
        d=certify(s,y,perm,n,m,.05,.1)
        mask=(s<d['tau']);mask[perm[:n]]=False
        assert d['skipped']==int(mask.sum()) and d['hidden']==int(y[mask].sum())
checks['prefix_identity_strict_mask_checks']=90
assert np.array_equal(Predictor(s,'count').values(sc),Predictor(np.ones(N),'count').values(sc))
assert np.array_equal(Predictor(s,'count').values(sc),sc[:,0]/(sc[:,1]+1))
assert np.all(Predictor(np.ones(N),'attachment-flat').values(sc)==0)
d=certify(np.ones(N),y,np.arange(N),cap,int(y[:cap].sum()),.05,.1);assert d['skipped']==0
checks['ties_and_pure_count']=True
# Independent exact certificate: integer combinatorial tails, no scipy or reference tails.
@lru_cache(None)
def exact_rank(m,N,n,e,d):
    if m==0:return 0
    best=0
    for j in range(1,m+1):
        b=F(0)
        for M in range(m,m+N-n+1):
            q=j+(e.numerator*M)//e.denominator
            tail=F(0) if q>=M else F(sum(comb(q,x)*comb(M-q,m-x) for x in range(j) if x<=q and 0<=m-x<=M-q),comb(M,m))
            b=max(b,tail)
        if b<=d:best=j
    return best
@lru_cache(None)
def bad_fraction(M,m,j,e):
    if not j:return F(0)
    return F(sum(r[j-1]-j>e*M for r in combinations(range(1,M+1),m)),comb(M,m))
def stop(path,cap,rule,e,d):
    m=0
    for n,label in enumerate(path[:cap],1):
        m+=label
        if rule=='first-positive' and exact_rank(m,len(path),n,e,d)>0:return n,m
        if rule=='two-consecutive-failures' and n>=2 and path[n-1]==path[n-2]==1:return n,m
        if rule=='two-consecutive-passes' and n>=2 and path[n-1]==path[n-2]==0:return n,m
        if rule=='adaptive-target' and m>=(1 if path[0] else 3):return n,m
    return cap,m
cases=0;conditional=0;max_bad=F(0)
for N in range(2,10):
 for M in range(1,N+1):
  paths=[]
  for positions in combinations(range(N),M):
    path=[0]*N
    for p in positions:path[p]=1
    paths.append(path)
  for e,d in [(F(1,4),F(1,10)),(F(1,4),F(1,4)),(F(1,2),F(1,4)),(F(1,2),F(2,5))]:
   for cap in sorted(set([max(1,N//2),N])):
    for rule in ['first-positive','two-consecutive-failures','two-consecutive-passes','adaptive-target']:
     total=F(0)
     for path in paths:
        n,m=stop(path,cap,rule,e,d);j=exact_rank(m,N,n,e,d)
        computed=reference.jstar_fpc(m,N,n,e,float(d))
        # At a rational boundary scipy may conservatively reject equality by rounding.
        assert computed<=j and abs(computed-j)<=1,(N,n,m,e,d,j,computed)
        bad=bad_fraction(M,m,j,e);assert bad<=d;total+=bad;conditional+=1
     total/=len(paths);assert total<=d;max_bad=max(max_bad,total);cases+=1
checks.update(exact_population_policy_settings=cases,exact_conditional_paths=conditional,max_exact_bad_probability=str(max_bad))
for z in range(60):
    N=int(rg.integers(80,15000));n=int(rg.integers(1,N));m=int(rg.integers(0,n+1));e,d=[(.05,.1),(.05,.05),(.1,.1)][z%3]
    assert reference.jstar_fpc(m,N,n,e,d)==certificates.ORIGINAL_FPC(m,N,n,e,d),(N,n,m,e,d)
checks['optimized_vs_original_FPC_search']=60
result=dict(passed=True,checks=checks,seconds=time.perf_counter()-start,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'analysis').glob('*.py')},protocol_sha256=hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest())
(R/'audit/implementation_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
