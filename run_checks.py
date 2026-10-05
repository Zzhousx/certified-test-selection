"""Run independent numerical and execution-contract checks in separate processes."""
from pathlib import Path
import os, subprocess, sys
from prepare_inputs import initialize

ROOT=Path(__file__).resolve().parent

def main():
    initialize()
    env=os.environ.copy();env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    groups={
        'replication/analysis':['test_revision.py'],
        'reproducibility':['test_execution_contract.py','test_adaptive.py','test_0910.py'],
        'simulator_archive':['test_execution_contract.py'],
    }
    for folder,tests in groups.items():
        subprocess.run([sys.executable,'-m','pytest',*tests,'-q','-p','no:cacheprovider'],cwd=ROOT/folder,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'adaptive_replication/analysis/validate.py')],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
