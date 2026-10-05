from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
import itertools
import numpy as np
import pytest
from adaptive_triage import adaptive_triage, checkpoint_decision

def test_complete_partition_and_no_reexecution():
    s=np.linspace(0,1,100); y=np.zeros(100,int); y[40:]=1; calls=[]
    def execute(ix): calls.extend(ix.tolist()); return y[ix]
    r=adaptive_triage(s,execute,[20,50,80],epsilon=.3,delta=.3,
                      stop_skip_fraction=.01,permutation=np.random.default_rng(8).permutation(100))
    assert r['execution_complete'] and r['certified']
    assert len(calls)==len(set(calls))==r['executed']
    assert set(calls).isdisjoint(r['skipped_indices'])
    assert set(calls)|set(r['skipped_indices'])==set(range(100))

def test_final_checkpoint_refusal_is_executed():
    calls=[]
    def execute(ix): calls.extend(ix.tolist()); return np.zeros(len(ix),int)
    r=adaptive_triage(np.arange(10),execute,[2,4],permutation=np.arange(10))
    assert r['checkpoint']==2 and not r['target_met'] and r['rank']==0
    assert len(calls)==10

@pytest.mark.parametrize('ns',[[2,2],[4,2],[0,4],[2,10],[2.5,4],[]])
def test_reject_invalid_checkpoints(ns):
    with pytest.raises(ValueError): adaptive_triage(np.arange(10),lambda ix:np.zeros(len(ix)),ns)

def test_nonbinary_outcomes_rejected_before_cast():
    with pytest.raises(ValueError): adaptive_triage(np.arange(10),lambda ix:np.full(len(ix),.5),[2,4])

def test_retained_execution_failure_propagates():
    calls=0
    def execute(ix):
        nonlocal calls
        calls+=1
        if calls==3: raise RuntimeError('retained execution failed')
        return np.zeros(len(ix))
    with pytest.raises(RuntimeError): adaptive_triage(np.arange(10),execute,[2,4])

@pytest.mark.parametrize('s',[np.arange(7),np.array([0,0,1,1,2,2,3])])
def test_exhaustive_checkpoint_union_bound(s):
    y=np.array([1,0,1,0,1,0,1]); eps=.34; delta=.4; any_bad=selected_bad=0; total=0
    for permutation in itertools.permutations(range(7)):
        p=np.array(permutation); bad=[]
        for n in [3,5]:
            d=checkpoint_decision(s,p[:n],y[p[:n]],eps,delta/2)
            bad.append(y[d['skipped_indices']].sum()>eps*y.sum())
        any_bad+=any(bad)
        r=adaptive_triage(s,lambda ix:y[ix],[3,5],epsilon=eps,delta=delta,
                          stop_skip_fraction=.1,permutation=p)
        selected_bad+=y[r['skipped_indices']].sum()>eps*y.sum(); total+=1
    assert selected_bad<=any_bad<=delta*total
