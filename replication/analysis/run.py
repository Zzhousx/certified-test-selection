import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.modules['bottleneck']=None
import json,time,argparse,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import hypergeom
from inputs import ROOT,SOURCE,tasks,identity,sha
from certificates import decision,inverse,validate,bin_index
from budget_bound import bounds
import reference as r

P=json.loads((ROOT/'protocol.json').read_text())
OUT=Path(os.environ.get('CTS_BASELINE_OUTPUT',str(ROOT/'results')));OUT.mkdir(parents=True,exist_ok=True)
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def task_id(task):return '_'.join(str(task[k]).replace(' ','-') for k in ['subject','source','target','scorer','model_seed'])
def primary(t):return t['scorer']=='HGB-spec' and t['model_seed']==20260905
def extended(t):return primary(t) or t['scorer']=='ITS4SDC' or (t['subject']=='openpilot' and t['scorer']=='Logistic-spec')
def metrics(t,d,rep,method,design,eps=.05,delta=.1,**extras):
    s,y=t['s'],t['y'];N=len(y);M=int(y.sum());skip=d['skip'];H=int(y[skip].sum());saving=float(skip.mean())
    dur=t['duration'];duration=float(dur[skip].sum()/dur.sum()) if np.isfinite(dur).all() and dur.sum()>0 else ''
    return dict(**identity(t),N=N,M=M,rep=rep,draw_seed=2026091100+10000*t['target_id']+rep,method=method,design=design,epsilon=eps,delta=delta,n=d['n_cal'],m=d['m'],j=d['j'],tau=d['tau'],saving=saving,calibration=d['n_cal']/N,executed=N-int(skip.sum()),hidden=H,miss=H/M if M else 0.,refusal=int(d['j']==0),attainment=int(d['j']>0 and saving>=.1),violation=int(H>eps*M+1e-12),duration_saved=duration,**extras)

def run_task(t,repetitions=500):
    tic=time.perf_counter();name=task_id(t);rows=[];budget=[];family=[];paired=[];planning=[]
    s,y=t['s'],t['y'];N=len(y);M=int(y.sum());ident=identity(t)
    validate(s,y,np.arange(N))
    estimate=t['source_rate'];selected=max([k for k in P['inverse_K'] if estimate and k/(estimate*N)<=.2]+[45])
    configs={(f,.05,.1) for f in P['fractions']}
    if primary(t):configs|={(.2,e,.1) for e in [.01,.02,.05,.1]}|{(.2,.05,d) for d in [.01,.05,.1]}
    for rep in range(repetitions):
        perm=np.random.default_rng(2026091100+10000*t['target_id']+rep).permutation(N)
        anchor=None;fixed20=None;fixed=[]
        for f,eps,delta in sorted(configs):
            n=round(f*N)
            for rule in ['FPC','Binomial']:
                d=decision(s,y,perm,n,eps,delta,rule);row=metrics(t,d,rep,rule,f,eps,delta);rows.append(row)
                if rule=='FPC' and (f,eps,delta)==(.2,.05,.1):anchor=d;fixed20=row
                if (eps,delta)==(.05,.1):fixed.append(row)
        chosen=None
        for ck,f in enumerate(P['fractions'],1):
            d=decision(s,y,perm,round(f*N),.05,.025,'FPC')
            if (d['j']>0 and d['skip'].mean()>=.1) or ck==4:
                chosen=d;break
        a=metrics(t,chosen,rep,'Adaptive','checkpoints',checkpoint=ck);rows.append(a)
        inv_primary=None
        for K in (P['inverse_K'] if extended(t) else [selected]):
            for rule in (['FPC','Binomial'] if extended(t) else ['FPC']):
                d=inverse(s,y,perm,K,.05,.1,round(.3*N),rule)
                row=metrics(t,d,rep,'Target-K-'+rule,K,K=K,primary_K=int(K==selected),cap_hit=d['cap_hit']);rows.append(row)
                if K==selected and rule=='FPC':inv_primary=row
        paired.append(dict(**ident,rep=rep,K=selected,saving_vs_fixed20=inv_primary['saving']-fixed20['saving'],saving_vs_adaptive=inv_primary['saving']-a['saving'],calibration_vs_fixed20=inv_primary['calibration']-fixed20['calibration'],calibration_vs_adaptive=inv_primary['calibration']-a['calibration'],attainment_vs_adaptive=inv_primary['attainment']-a['attainment']))
        if t['family'] is not None and extended(t):
            rates=[]
            for fam in range(4):
                idx=np.flatnonzero(t['family']==fam);Mf=int(y[idx].sum());Hf=int(y[idx][anchor['skip'][idx]].sum());rates.append(Hf/Mf if Mf else 0.)
                family.append(dict(**ident,rep=rep,family=fam,failures=Mf,hidden=Hf,miss=Hf/Mf if Mf else '',violation=int(Mf>0 and Hf>.05*Mf),any_family_violation=int(any(z>.05 for z in rates))))
            anybad=int(any(z>.05 for z in rates))
            for row in family[-4:]:row['any_family_violation']=anybad
            # Optional family-wise certificate, with a predeclared error allocation.
            strat_skip=np.zeros(N,bool);totaln=0;totalm=0;valid=0
            for fam in range(4):
                idx=np.flatnonzero(t['family']==fam);local=perm[np.isin(perm,idx)];lookup=np.empty(N,int);lookup[idx]=np.arange(len(idx));p=lookup[local];d=decision(s[idx],y[idx],p,round(.2*len(idx)),.05,.025,'FPC');strat_skip[idx]=d['skip'];totaln+=d['n_cal'];totalm+=d['m'];valid+=d['j']>0
            rows.append(metrics(t,dict(skip=strat_skip,n_cal=totaln,m=totalm,j=valid,tau=''),rep,'Stratified-FPC',.2))
        if extended(t):
            riskorder=np.argsort(-s,kind='stable')
            for nf in P['budget_n']:
                n=round(nf*N);c=perm[:n];m=int(y[c].sum())
                for bf in P['budget_B']:
                    B=round(bf*N);tau=r.budget_threshold(s,B,n);mask=s<tau;x=int(y[c][mask[c]].sum());mask[c]=False;Lu=int(mask.sum());H=int(y[mask].sum());loss=H/M if M else 0.;start=time.perf_counter();u,legacy=bounds(m,x,N,n,Lu);seconds=time.perf_counter()-start
                    exact=r.ucb_miss_share(m,x,N,n,Lu,.1) if rep%25==0 else None
                    if exact is not None:assert u+1e-12>=exact and legacy+1e-12>=exact,(name,rep,nf,bf,u,legacy,exact)
                    executed=N-Lu;assert executed<=B
                    budget.append(dict(**ident,N=N,M=M,rep=rep,n_fraction=nf,budget_fraction=bf,n=n,B=B,m=m,x=x,executed=executed,miss=loss,eps_hat=u,legacy_eps_hat=legacy,exact_eps_hat=exact if exact is not None else '',violation=int(loss>u+1e-12),legacy_violation=int(loss>legacy+1e-12),exact_violation=int(loss>exact+1e-12) if exact is not None else '',risk_order_miss=float(1-y[riskorder[:B]].sum()/M) if M else 0.,bound_seconds=seconds))
        if rep%100==99:
            (OUT/(name+'_progress.json')).write_text(json.dumps(dict(reps_completed=rep+1,elapsed_seconds=time.perf_counter()-tic)))
    write(name+'_decisions.csv',rows);write(name+'_budget.csv',budget);write(name+'_family.csv',family);write(name+'_paired.csv',paired)
    # Analytical checks use true target M; practical planning uses only source rate.
    for f in P['fractions']:
        n=round(f*N);rv=hypergeom(N,M,n);lo=int(rv.ppf(1e-10));hi=int(rv.ppf(1-1e-10));vals=np.arange(lo,hi+1);weights=rv.pmf(vals)
        js=np.array([r.jstar_fpc(int(m),N,n,.05,.1) for m in vals]);jb=np.array([bin_index(int(m)) for m in vals])
        p=float(weights[js==0].sum()+rv.cdf(lo-1));differ=float(weights[js!=jb].sum())
        observed=np.mean([row['refusal'] for row in rows if row['method']=='FPC' and row['design']==f and row['epsilon']==.05 and row['delta']==.1])
        empirical=np.mean([a['j']!=b['j'] for a,b in zip([q for q in rows if q['method']=='FPC' and q['design']==f and q['epsilon']==.05 and q['delta']==.1],[q for q in rows if q['method']=='Binomial' and q['design']==f and q['epsilon']==.05 and q['delta']==.1])])
        planning.append(dict(**ident,N=N,M=M,fraction=f,predicted_refusal_true_M=p,observed_refusal=observed,refusal_gap=observed-p,three_se=3*np.sqrt(p*(1-p)/repetitions),analytic_rules_differ=differ,observed_rules_differ=empirical,source_failure_rate=estimate,source_M_estimate=estimate*N if estimate else '',primary_K=selected))
    write(name+'_planning.csv',planning)
    manifest=dict(task=ident,repetitions=repetitions,rows=len(rows),budget_rows=len(budget),family_rows=len(family),elapsed_seconds=time.perf_counter()-tic,protocol_sha256=sha(ROOT/'protocol.json'),outputs={p.name:sha(p) for p in OUT.glob(name+'_*.csv')})
    (OUT/(name+'_manifest.json')).write_text(json.dumps(manifest,indent=2));return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--repetitions',type=int,default=500);ap.add_argument('--subject',choices=['deepscenario','sensodat','openpilot']);ap.add_argument('--limit-tasks',type=int);ap.add_argument('--output-dir',type=Path);args=ap.parse_args()
    if args.workers<1 or args.repetitions<1 or (args.limit_tasks is not None and args.limit_tasks<1):ap.error('counts must be positive')
    global OUT
    if args.output_dir:
        OUT=args.output_dir.resolve();os.environ['CTS_BASELINE_OUTPUT']=str(OUT);OUT.mkdir(parents=True,exist_ok=True)
    from certificates import check_backend
    assert check_backend()['passed']
    all_tasks=[t for subject in ([args.subject] if args.subject else ['deepscenario','sensodat','openpilot']) for t in tasks(subject)]
    if args.limit_tasks:all_tasks=all_tasks[:args.limit_tasks]
    # Long jobs first; one independent process per frozen scorer/transfer.
    all_tasks.sort(key=lambda t:not extended(t))
    print('START tasks',len(all_tasks),'workers',args.workers,flush=True)
    for t in all_tasks:
        path=OUT/(task_id(t)+'_manifest.json')
        if path.exists():
            previous=json.loads(path.read_text())
            assert previous['repetitions']==args.repetitions and previous['protocol_sha256']==sha(ROOT/'protocol.json'),'Existing output uses a different protocol; select a new output directory.'
            assert all(sha(OUT/file)==digest for file,digest in previous['outputs'].items()),'Existing output changed; investigate before resuming.'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_task,t,args.repetitions):task_id(t) for t in all_tasks if not (OUT/(task_id(t)+'_manifest.json')).exists()}
        for future in as_completed(futures):
            result=future.result();print('COMPLETE',futures[future],result['rows'],round(result['elapsed_seconds'],1),flush=True)
    manifests=[json.loads((OUT/(task_id(t)+'_manifest.json')).read_text()) for t in all_tasks]
    assert len(manifests)==len(all_tasks)
    (OUT/'run_manifest.json').write_text(json.dumps(dict(tasks=len(manifests),decision_rows=sum(m['rows'] for m in manifests),budget_rows=sum(m['budget_rows'] for m in manifests),new_simulator_executions=0,protocol_sha256=sha(ROOT/'protocol.json'),completed=True),indent=2))
    print('ALL EXPERIMENTS COMPLETE',flush=True)
if __name__=='__main__':main()
