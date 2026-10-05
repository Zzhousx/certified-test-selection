from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Checks for the new comparison metrics, beyond threshold-unit tests."""
import numpy as np
import run_0910 as r
def test_diversity_does_not_keep_a_virtual_centroid():
    X=np.array([[0.],[1.],[3.],[4.],[10.]])
    assert r.diversity(X).tolist()==[4,0,3,1,2]

def test_diversity_matches_independent_selected_set_distances():
    X=np.random.default_rng(19).normal(size=(40,7));X[1]=X[0]
    A=(X-X.mean(axis=0))/X.std(axis=0);selected=[]
    order=r.diversity(X)
    for k in order:
        scores=np.sum(A*A,axis=1) if not selected else np.array([min(np.sum((x-A[j])**2) for j in selected) for x in A])
        scores[selected]=-np.inf
        assert np.isclose(scores[k],scores.max(),atol=1e-12,rtol=1e-12)
        selected.append(int(k))
    assert len(set(selected))==len(X)

def test_hindsight_prefix_handles_ties_zero_and_integer_recall():
    cum=np.r_[0,np.cumsum([0,1,0,1,1,0])]
    assert r.prefix_cost(cum,0)==0
    assert r.prefix_cost(cum,1)==2
    assert r.prefix_cost(cum,2)==4
    assert r.prefix_cost(cum,3)==5
def test_complete_cost_and_shared_target_definition(monkeypatch):
    monkeypatch.setitem(r.P,'road_repetitions',2)
    y=np.array([1,0,0,1,0,0,0,0,1,0])
    task=dict(scope='road',subject='synthetic',source='source',target='target',scorer='HGB-spec',model_seed=20260905,s=np.arange(10)/10,y=y,X=np.arange(20).reshape(10,2),duration=np.ones(10),history_cut=.5,target_id=0,external=np.arange(10)/10,detour_orders=[np.arange(10),np.arange(9,-1,-1)])
    out=r.evaluate(task)
    for x in out['decisions']:
        assert x['executed']+x['skipped']==10
        assert x['target_met']==int(x['rank']>0 and x['skipped']/10>=.1)
        assert x['found']+x['hidden']==3
    for x in out['matched']:
        assert x['extra_executions']==x['executed']-x['required_same_recall']
    assert not any(x['method'].startswith('FPC') for x in out['curves'])
    for b in out['feasibility']:
        row=next(x for x in out['decisions'] if x['method']=='FPC' and x['rep']==b['rep'])
        assert b['feasible']==int(row['executed']<=round(b['budget']*10))
