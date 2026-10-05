import itertools
import numpy as np
import pytest
import reference as r
from certificates import decision,inverse,validate,index
from budget_bound import bounds
from planning import minimum_m,availability,fraction

def test_fixed_backend_matches_attached_reference():
    from certificates import check_backend
    assert check_backend()['passed']

def test_refined_bound_is_conservative_on_small_grid():
    from budget_bound import validate as check
    assert check()['passed']

def test_refined_bound_random_exact_crosschecks():
    rng=np.random.default_rng(902)
    for _ in range(100):
        N=int(rng.integers(20,5000));n=int(rng.integers(1,N));m=int(rng.integers(1,min(n,200)+1));x=int(rng.integers(0,m+1));Lu=int(rng.integers(0,N-n+1))
        u,legacy=bounds(m,x,N,n,Lu);exact=r.ucb_miss_share(m,x,N,n,Lu,.1)
        assert 0<=exact<=u+1e-12<=legacy+1e-12<=1+1e-12

def test_lower_feasibility_correction_resolves_known_counterexample():
    args=(168,109,1453,470,980)
    refined,legacy=bounds(*args);exact=r.ucb_miss_share(*args,.1)
    assert legacy-exact>.5
    assert refined-exact<.005
    assert bounds(1,1,10000,1,9999)[1]<=1

def test_fpc_minimum_is_not_universal_binomial_minimum():
    assert r.jstar_fpc(25,10000,9990,.05,.1)==1
    assert r.jstar_binomial(25,.05,.1)==0
    assert minimum_m(10000,9990)<45

def test_inverse_cap_and_strict_tie_handling():
    s=np.zeros(12);y=np.ones(12,int);p=np.arange(12)
    d=inverse(s,y,p,10,.05,.1,5)
    assert d['cap_hit']==1 and d['n_cal']==5 and not d['skip'].any()
    d=decision(s,y,p,11,.2,.3,'FPC')
    assert d['j']>0 and not d['skip'].any()

def test_inverse_all_permutations_with_ties():
    N=7;y=np.array([1,0,1,0,1,0,1]);s=np.array([0,0,1,1,2,2,3]);bad=0
    for p in itertools.permutations(range(N)):
        d=inverse(s,y,np.array(p),3,.34,.4,5)
        bad+=int(y[d['skip']].sum()>.34*y.sum())
    assert bad<=.4*5040

def test_invalid_execution_contract_is_rejected():
    s=np.arange(10.);y=np.zeros(10);p=np.arange(10)
    with pytest.raises(ValueError):validate(s,y,p,5,3)
    with pytest.raises(ValueError):validate(s,np.full(10,.5),p)
    with pytest.raises(ValueError):validate(s,y,np.zeros(10,int))
    with pytest.raises(ValueError):index(2,10,5,.05,.1,'unknown')

def test_planning_binary_search_matches_full_predeclared_grid():
    for N,M in [(100,20),(100,80),(200,3)]:
        for p in [.5,.9]:
            grid=[round(i*.005,3) for i in range(1,200) if availability(N,M,round(i*.005*N))>=p]
            assert fraction(N,M,p)==(grid[0] if grid else None)
