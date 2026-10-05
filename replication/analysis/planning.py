"""Exact availability probabilities; distinguish oracle checks from prior forecasts."""
import json,csv
from functools import lru_cache
import numpy as np
from scipy.stats import hypergeom
from certificates import fast_b
from fractions import Fraction
from inputs import ROOT,tasks,identity

@lru_cache(None)
def minimum_m(N,n,eps=.05,delta=.1):
    # With U=N-n fixed, each compatible M for m+1 corresponds to M-1 for m.
    # Adding one sampled failure multiplies the zero-hit tail by <=1, while
    # floor(eps*M) cannot decrease. Thus b_1 is nonincreasing in m.
    e=Fraction(str(eps));lo,hi=1,n
    if fast_b(1,n,n,N,e)>delta:return None
    while lo<hi:
        mid=(lo+hi)//2
        if fast_b(1,mid,mid,mid+N-n,e)<=delta:hi=mid
        else:lo=mid+1
    return lo

@lru_cache(None)
def availability(N,M,n,eps=.05,delta=.1):
    cutoff=minimum_m(N,n,eps,delta)
    return 0. if cutoff is None else float(hypergeom.sf(cutoff-1,N,M,n))

def fraction(N,M,p):
    # Along a nested uniform sample, m increases and U decreases. Availability
    # is monotone pathwise; binary search over the predeclared 0.5% grid is exact.
    lo,hi=1,199
    if availability(N,M,round(.995*N))<p:return None
    while lo<hi:
        mid=(lo+hi)//2
        if availability(N,M,round(mid*.005*N))>=p:hi=mid
        else:lo=mid+1
    return round(lo*.005,3)

def main():
    checks=0
    for N in range(2,32):
        for n in range(1,N):
            values=[fast_b(1,m,m,m+N-n,Fraction(1,5)) for m in range(1,n+1)]
            assert all(b<=a+1e-13 for a,b in zip(values,values[1:]));checks+=1
    rows=[]
    for subj in ['deepscenario','sensodat','openpilot']:
        for task in tasks(subj,True):
            N=len(task['y']);M=int(task['y'].sum());Me=int(round(task['source_rate']*N))
            row=dict(**identity(task),N=N,M=M,source_M_estimate=Me)
            for p in [.5,.9]:
                for kind,total in [('oracle',M),('source',Me)]:
                    f=fraction(N,total,p);row[f'{kind}_fraction_p{p}']=f
                    row[f'{kind}_actual_availability_p{p}']=availability(N,M,round(f*N)) if f else None
            for f in [.05,.1,.2,.3]:
                row[f'minimum_m_{f}']=minimum_m(N,round(f*N))
                row[f'source_refusal_{f}']=1-availability(N,Me,round(f*N))
                row[f'oracle_refusal_{f}']=1-availability(N,M,round(f*N))
            rows.append(row)
    with (ROOT/'results/planning_sample_sizes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'results/planning_validation.json').write_text(json.dumps(dict(passed=True,monotonic_small_population_cases=checks,rows=len(rows),probability_computation='Exact hypergeometric survival function; no support truncation'),indent=2))
    print('PLANNING COMPLETE',len(rows),checks,flush=True)
if __name__=='__main__':main()
