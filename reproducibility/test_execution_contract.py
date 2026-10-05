from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Operational failures that would make a completed-remainder claim false."""
import numpy as np
import pytest
from certskip import CertifiedSkip

@pytest.mark.parametrize('bad',[np.array([0.2,1]),np.array([np.nan,1]),np.array([1])])
def test_calibration_rejects_missing_or_nonbinary_results(bad):
    with pytest.raises(ValueError,match='binary outcome'):
        CertifiedSkip(rule='fpc').run_fixed(np.zeros((10,1)),lambda ix:bad,2,scores=np.arange(10.))

def test_second_execution_must_succeed_before_completion():
    calls=[]
    def execute(ix):
        calls.append(ix)
        if len(calls)==2:raise RuntimeError('simulator stopped')
        return np.zeros(len(ix),int)
    with pytest.raises(RuntimeError,match='simulator stopped'):
        CertifiedSkip(rule='fpc').run_fixed(np.zeros((10,1)),execute,2,scores=np.arange(10.))
    assert len(calls)==2

def test_duplicate_order_cannot_omit_an_unrecorded_test():
    with pytest.raises(ValueError,match='permutation'):
        CertifiedSkip(rule='fpc').run_fixed(np.zeros((10,1)),lambda ix:np.zeros(len(ix)),2,
                                           scores=np.arange(10.),order=np.zeros(10,int))
