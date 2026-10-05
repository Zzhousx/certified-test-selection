import csv
from inputs import ROOT
from planning import minimum_m
import reference

rows=[]
for epsilon in [.01,.02,.05,.1]:
    for delta in [.1,.05,.025,.01]:
        rows.append(dict(N=10000,n=2000,epsilon=epsilon,delta=delta,binomial_minimum=reference.m_min_binomial(epsilon,delta),fpc_minimum=minimum_m(10000,2000,epsilon,delta)))
with (ROOT/'results/minimum_failure_counts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('MINIMUM FAILURE COUNTS COMPLETE',flush=True)
