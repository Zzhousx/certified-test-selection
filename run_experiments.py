"""Run the primary study, baselines, extensions, or a small reproducibility check."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from prepare_inputs import initialize

ROOT=Path(__file__).resolve().parent

def run(rel,*args,env=None):
    subprocess.run([sys.executable,str(ROOT/rel),*map(str,args)],cwd=ROOT,env=env,check=True)

def ensure_validated():
    path=ROOT/'adaptive_replication/audit/implementation_validation.json'
    if path.exists():
        record=json.loads(path.read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        core=ROOT/'adaptive_replication/analysis'
        if record.get('passed') and record.get('protocol_sha256')==digest(core.parent/'protocol.json'):
            if all(record.get('code_sha256',{}).get(n)==digest(core/n) for n in ['reference.py','certificates.py','policies.py']):return
    run('adaptive_replication/analysis/validate.py')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scope',choices=['adaptive','baselines','road','duration','refit','references','smoke','all'])
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--repetitions',type=int,default=500,help='Adaptive/baseline repetitions. Paper default: 500.')
    ap.add_argument('--subject',choices=['deepscenario','sensodat','openpilot','road'])
    ap.add_argument('--limit-tasks',type=int)
    a=ap.parse_args();initialize()
    if a.subject=='road' and a.scope!='references':ap.error('road subject applies to references only')
    if a.repetitions<1 or a.workers<1:ap.error('counts must be positive')
    if a.scope=='smoke':
        out=ROOT/'validation/smoke';env=os.environ.copy();env['CTS_ADAPTIVE_OUTPUT']=str(out)
        ensure_validated()
        run('adaptive_replication/analysis/run.py','--workers',1,'--repetitions',2,'--subject','deepscenario','--limit-tasks',1,'--output-dir',out)
        run('adaptive_replication/analysis/summarize.py',env=env)
        run('adaptive_replication/analysis/post_validate.py',env=env)
        return
    scopes=['adaptive','baselines','road','duration'] if a.scope=='all' else [a.scope]
    for scope in scopes:
        args=['--workers',a.workers,'--repetitions',a.repetitions]
        if a.subject:args+=['--subject',a.subject]
        if a.limit_tasks:args+=['--limit-tasks',a.limit_tasks]
        if scope=='adaptive':
            ensure_validated()
            run('adaptive_replication/analysis/run.py',*args)
            run('adaptive_replication/analysis/summarize.py')
            run('adaptive_replication/analysis/post_validate.py')
        elif scope=='baselines':
            run('replication/analysis/run.py',*args)
            run('replication/analysis/planning.py')
            run('replication/analysis/minimum_failures.py')
            run('replication/analysis/summarize.py')
            run('replication/analysis/matched_cost.py')
        elif scope=='road':run('analysis/road_validation.py')
        elif scope=='duration':run('analysis/cost_validation.py')
        elif scope=='refit':run('reproducibility/refit_scores_0910.py')
        elif scope=='references':
            for subject in ([a.subject] if a.subject else ['deepscenario','sensodat','openpilot','road']):
                run('reproducibility/run_0910.py',subject)

if __name__=='__main__':main()
