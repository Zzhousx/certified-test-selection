import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.modules['bottleneck']=None
from pathlib import Path
import json,hashlib,csv,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import reference,certificates
from inputs import tasks,identity
from policies import *
P=json.loads((ROOT/'protocol.json').read_text())
OUT=Path(os.environ.get('CTS_ADAPTIVE_OUTPUT',str(ROOT/'results')));OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(t):return '_'.join(str(t[k]).replace(' ','-') for k in ['subject','source','target','scorer','model_seed'])

def run_task(task,repetitions=None):
    repetitions=P["repetitions"] if repetitions is None else repetitions
    start=time.perf_counter();N=len(task['y']);M=int(task['y'].sum());cap=round(.3*N)
    s,y=task['s'],task['y'];certificates.validate(s,y,np.arange(N));assert np.all((s>=0)&(s<=1))
    ident=identity(task);tid=name(task);rows=[]
    predictors={mode:Predictor(s,mode) for mode in ['count','score','attachment-flat']}
    settings=[]
    for eps,delta in P['settings']:
        sched=schedule(cap,eps,delta);minimum=int(sched[0,1]);grid=[minimum*k for k in [1,2,4,8]]
        settings.append((eps,delta,sched,grid,source_target_K(task['source_rate'],N,grid),{mode:pred.values(sched) for mode,pred in predictors.items()}))
    for rep in range(repetitions):
        seed=2026091100+10000*task['target_id']+rep;perm=np.random.default_rng(seed).permutation(N);path=y[perm]
        for eps,delta,sched,grid,K,values in settings:
            pending=[]
            n,m,reason=target_stop(path,K,cap)
            pending.extend([('TargetK-fallback',n,m,reason,'FPC',reason=='cap',K),('TargetK-cap',n,m,reason,'FPC',False,K),('TargetK-cap-binomial',n,m,reason,'Binomial',False,K)])
            pending.append(('Fixed20',round(.2*N),int(path[:round(.2*N)].sum()),'fixed','FPC',False,0))
            for mode,label in [('count','EV-count'),('score','EV-score'),('attachment-flat','EV-flat-attachment')]:
                n,m,reason=ev_stop(path,cap,sched,values[mode]);pending.append((label,n,m,reason,'FPC',False,0))
                if mode=='count':pending.append(('EV-count-binomial',n,m,reason,'Binomial',False,0))
            n,m,reason=count_checkpoint(path,N,cap,eps,delta);pending.append(('Count-checkpoints',n,m,reason,'FPC',False,0))
            for k in grid:
                n,m,reason=target_stop(path,k,cap);pending.append(('K-grid-cap',n,m,reason,'FPC',False,k))
            calculated=[]
            for policy,n,m,reason,rule,fallback,k in pending:
                result=certify(s,y,perm,n,m,eps,delta,rule,fallback)
                calculated.append((policy,reason,rule,k,result))
            result,reason=score_checkpoint(s,y,perm,cap,eps,delta)
            calculated.append(('Score-checkpoints-split',reason,'FPC',0,result))
            for policy,reason,rule,k,result in calculated:
                skipped,H=result['skipped'],result['hidden']
                assert 0<=H<=M and result['n']<=N-skipped<=N
                rows.append(dict(**ident,N=N,M=M,rep=rep,draw_seed=seed,epsilon=eps,delta=delta,policy=policy,rule=rule,K=k,selected_K=K,stop_reason=reason,
                    cap=cap,cap_position=int(result['n']==cap),cap_target_not_reached=int(reason=='cap' and policy.startswith('TargetK')),**result,
                    saving=skipped/N,calibration=result['n']/N,executed=N-skipped,miss=H/M if M else 0.,violation=int(H>eps*M+1e-12),refusal=int(result['j']==0),attainment=int(result['j']>0 and skipped/N>=.1)))
        if rep%100==99:(OUT/(tid+'_progress.json')).write_text(json.dumps(dict(reps=rep+1,seconds=time.perf_counter()-start)))
    p=OUT/(tid+'_decisions.csv')
    with p.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(task=ident,repetitions=repetitions,rows=len(rows),protocol_sha256=sha(ROOT/'protocol.json'),file=p.name,sha256=sha(p),seconds=time.perf_counter()-start)
    (OUT/(tid+'_manifest.json')).write_text(json.dumps(result,indent=2));return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--repetitions',type=int,default=P['repetitions'])
    ap.add_argument('--subject',choices=['deepscenario','sensodat','openpilot'])
    ap.add_argument('--limit-tasks',type=int)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    if args.workers<1 or args.repetitions<1 or (args.limit_tasks is not None and args.limit_tasks<1):ap.error('counts must be positive')
    global OUT
    if args.output_dir:
        OUT=args.output_dir.resolve();os.environ['CTS_ADAPTIVE_OUTPUT']=str(OUT);OUT.mkdir(parents=True,exist_ok=True)
    validation=ROOT/'audit/implementation_validation.json'
    if not validation.exists():raise SystemExit('Run validate.py before the experiments.')
    checked=json.loads(validation.read_text())
    assert checked['passed'] and checked['protocol_sha256']==sha(ROOT/'protocol.json')
    for filename in ['reference.py','certificates.py','policies.py']:
        assert checked['code_sha256'][filename]==sha(ROOT/'analysis'/filename),'Core code changed. Rerun validate.py.'
    subjects=[args.subject] if args.subject else ['deepscenario','sensodat','openpilot']
    all_tasks=[task for subject in subjects for task in tasks(subject)]
    if args.limit_tasks:all_tasks=all_tasks[:args.limit_tasks]
    remaining=[]
    for task in all_tasks:
        p=OUT/(name(task)+'_manifest.json')
        if p.exists():
            m=json.loads(p.read_text())
            assert m['protocol_sha256']==sha(ROOT/'protocol.json') and m['repetitions']==args.repetitions
            assert sha(OUT/m['file'])==m['sha256']
        else:remaining.append(task)
    print('START',len(all_tasks),'tasks; pending',len(remaining),'workers',args.workers,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_task,t,args.repetitions):name(t) for t in remaining}
        for future in as_completed(futures):
            r=future.result();print('DONE',futures[future],r['rows'],round(r['seconds'],1),'seconds',flush=True)
    manifests=[json.loads((OUT/(name(t)+'_manifest.json')).read_text()) for t in all_tasks]
    result=dict(completed=True,tasks=len(manifests),rows=sum(m['rows'] for m in manifests),repetitions=args.repetitions,
                task_names=[name(t) for t in all_tasks],protocol_sha256=sha(ROOT/'protocol.json'),new_simulator_executions=0)
    (OUT/'run_manifest.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
