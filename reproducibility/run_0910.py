from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Recompute the frozen follow-up protocol from scores, specifications and outcomes.

No previous draw, summary or publication table is an input. Full labels are
available only to the evaluator; checkpoint_decision receives sampled labels.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from adaptive_triage import checkpoint_decision
from certskip.its4sdc import ITS4SDCScorer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/0910'; OUT.mkdir(parents=True,exist_ok=True)
P=json.loads((ROOT/'protocol_0910.json').read_text())
INPUTS={}
NAMES={'deepscenario':['Random','Greedy','RL'],'sensodat':['AmbieGen','Frenetic','FreneticV']}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=ROOT/p; INPUTS[str(p.relative_to(ROOT))]=sha(p)
    if p.suffix=='.npz':
        with np.load(p) as z:return {k:z[k] for k in z.files}
    if p.suffix=='.json':return json.loads(p.read_text())
    return pd.read_csv(p)

def diversity(X):
    X=np.asarray(X,float); med=np.nanmedian(X,axis=0);med[~np.isfinite(med)]=0
    X=np.where(np.isfinite(X),X,med);sd=X.std(axis=0);sd[sd==0]=1
    X=np.ascontiguousarray((X-X.mean(axis=0))/sd)
    order=np.empty(len(X),int)
    first=int(np.argmax(np.sum(X*X,axis=1)));order[0]=first
    # The centroid chooses only the first item; it is not an extra selected point.
    dist=cdist(X,X[first:first+1],metric='sqeuclidean').ravel();dist[first]=-np.inf
    for k in range(1,len(X)):
        i=int(np.argmax(dist));order[k]=i
        dist=np.minimum(dist,cdist(X,X[i:i+1],metric='sqeuclidean').ravel());dist[order[:k+1]]=-np.inf
    return order

def tasks(subject):
    report=read(Path('results')/(subject+'_run.json'))
    z=read(Path('results')/(subject+'_fresh_scores.npz'))
    if subject in NAMES:
        d=read(Path('data')/(subject+'_fresh.npz'));names=NAMES[subject]
        for r in report['model_records']:
            if r['scorer']=='ITS4SDC':continue
            si,ti=names.index(r['source']),names.index(r['target'])
            key=f"s{si}_t{ti}_{r['scorer']}_{r['model_seed']}"; ix=z[key+'_target_indices']
            yield dict(**r,s=z[key],y=d['y'][ix],X=d['X_spec'][ix],duration=d['duration_s'][ix],target_id=ti+(0 if subject=='deepscenario' else 3),scope='full')
        if subject=='sensodat':
            seq=read(Path('data/its4sdc_official_sequences.npz'))['sequence']
            path=ROOT/'vendor/its4sdc.onnx';INPUTS[str(path.relative_to(ROOT))]=sha(path)
            model=ITS4SDCScorer(path);tic=time.perf_counter();scores=model.score_features(seq);seconds=time.perf_counter()-tic
            np.savez_compressed(OUT/'its4sdc_scores.npz',scores=scores)
            for ti,target in enumerate(names):
                ix=np.flatnonzero(d['generator']==ti)
                yield dict(subject=subject,source='Released model',target=target,scorer='ITS4SDC',model_seed=0,s=scores[ix],y=d['y'][ix],X=d['X_spec'][ix],duration=d['duration_s'][ix],history_cut=None,target_id=ti+3,scope='full',fit_score_seconds=seconds)
    else:
        specs=read(Path('data/openpilot_formal_specs.json'))
        X=np.array([[s[k] for k in report['features']] for s in specs])
        for r in report['model_records']:
            key=f"{r['source']}_{r['target']}_{r['scorer']}_{r['model_seed']}"
            d=read(Path('results/openpilot/formal')/(r['target']+'_outcomes.csv'))
            assert d.scenario_id.tolist()==[s['scenario_id'] for s in specs]
            assert d.solver_errors.sum()==0
            yield dict(**r,s=z[key],y=d.failure.to_numpy(),X=X,duration=d.simulation_seconds.to_numpy(),target_id=6+['v0.9.8','v0.9.9','v0.10.0'].index(r['target']),scope='full')

def prefix_cost(cumulative,found):
    return int(np.searchsorted(cumulative,int(found),side='left'))

def outcome(y,executed):
    ids=np.asarray(executed,int);N=len(y);M=int(y.sum())
    assert len(np.unique(ids))==len(ids)
    found=int(y[ids].sum());hidden=M-found
    return dict(executed=len(ids),execution_fraction=len(ids)/N,skip_rate=1-len(ids)/N,found=found,hidden=hidden,recall=found/M if M else np.nan,miss=hidden/M if M else 0)

def evaluate(task):
    s,y=np.asarray(task['s'],float),np.asarray(task['y'],int);N=len(y);M=int(y.sum())
    assert np.isfinite(s).all() and np.isin(y,[0,1]).all()
    ident={k:task[k] for k in ['scope','subject','source','target','scorer','model_seed']};ident.update(N=N,M=M)
    main=task['scorer']=='HGB-spec' and task['model_seed']==P['main_model_seed']
    official=task['scorer']=='ITS4SDC';road=task['scope']=='road'
    ranking=dict(**ident,auc=roc_auc_score(y,s) if 0<M<N else np.nan,ap=average_precision_score(y,s) if M else np.nan,failure_rate=M/N,fit_score_seconds=task.get('fit_score_seconds',np.nan))
    orders={'Risk-order':np.argsort(-s,kind='stable')}
    tic=time.perf_counter()
    if main:orders['Diversity']=diversity(task['X'])
    ranking['diversity_seconds']=time.perf_counter()-tic
    ranking['detour_seconds']=task.get('detour_seconds',np.nan)
    if official:orders={'ITS4SDC-order':orders.pop('Risk-order')}
    if 'external' in task:orders['ITS4SDC-order']=np.argsort(-task['external'],kind='stable')
    reps=P['road_repetitions'] if road else P['repetitions']
    configs={(.2,.05,.1)}
    if main and not road:
        configs|={(f,.05,.1) for f in P['fractions']}
        configs|={(.2,e,.1) for e in P['epsilon_sensitivity']}
        configs|={(.2,.05,d) for d in P['delta_sensitivity']}
    if official:configs|={(f,.05,.1) for f in P['fractions']}
    decisions=[];curves=[];matched=[];prices=[];feasibility=[];native=[];timing=[]
    for method,ix in [('Full',np.arange(N))]+([('Historical',np.flatnonzero(s>=task['history_cut']))] if task.get('history_cut') is not None else [])+([('ITS4SDC',np.flatnonzero((task.get('external',s))>.5))] if official or 'external' in task else []):
        r=outcome(y,ix);native.append(dict(**ident,method=method,**r,meets_95=int(r['hidden']<=int(np.floor(.05*M+1e-12)))))
    for rep in range(reps):
        seed=(P['road_seed_base'] if road else P['seed_base'])+task['target_id']*10000+rep
        perm=np.random.default_rng(seed).permutation(N)
        active=dict(orders)
        if main:active['Random']=np.random.default_rng(P['random_seed_base']+task['target_id']*10000+rep).permutation(N)
        if road:active['DETOUR']=task['detour_orders'][rep]
        cumulative={name:np.r_[0,np.cumsum(y[order])] for name,order in active.items()}
        for name,cum in cumulative.items():
            stochastic=name in ['Random','DETOUR']
            if rep==0 or stochastic:
                for b in P['budget_fractions']:
                    B=round(b*N);f=int(cum[B]);curves.append(dict(**ident,rep=rep,method=name,budget=b,executed=B,found=f,hidden=M-f,recall=f/M if M else np.nan,miss=(M-f)/M if M else 0))
                k=prefix_cost(cum,int(np.ceil(.95*M)))
                prices.append(dict(**ident,rep=rep,reference=name,recall_target='95-percent',required=k,required_fraction=k/N,found_target=int(np.ceil(.95*M))))
        cache={}
        def fixed(frac,eps,delta):
            n=round(frac*N);key=(n,eps,delta)
            if key not in cache:
                tic=time.perf_counter();d=checkpoint_decision(s,perm[:n],y[perm[:n]],eps,delta)
                cache[key]=d
                timing.append(time.perf_counter()-tic)
            return cache[key],n
        def emit(method,d,n,eps,delta,frac,checkpoint=1,goal=.1,any_bad=None):
            skip=d['skipped_indices'];hidden=int(y[skip].sum());found=M-hidden
            row=dict(**ident,rep=rep,draw_seed=seed,method=method,fraction=frac,epsilon=eps,delta=delta,n=n,m=d['m'],checkpoint=checkpoint,threshold=d['threshold'],rank=d['rank'],calibration=n/N,skipped=len(skip),skip_rate=len(skip)/N,executed=N-len(skip),execution_fraction=1-len(skip)/N,found=found,hidden=hidden,recall=found/M if M else np.nan,miss=hidden/M if M else 0,violation=int(hidden>eps*M+1e-12),no_threshold=int(d['rank']==0),goal=goal,target_met=int(d['rank']>0 and len(skip)/N>=goal),any_checkpoint_violation=any_bad)
            dur=task['duration'];row['duration_saved']=float(dur[skip].sum()/dur.sum()) if np.isfinite(dur).all() and dur.sum()>0 else np.nan
            decisions.append(row);return row
        anchor=None
        for frac,eps,delta in sorted(configs):
            d,n=fixed(frac,eps,delta);r=emit('FPC',d,n,eps,delta,frac)
            if (frac,eps,delta)==(.2,.05,.1):anchor=r
            if main and (frac,eps,delta)==(.2,.05,.1):
                f=np.sort(s[perm[:n]][y[perm[:n]]==1]);tau=float(np.quantile(f,eps,method='linear')) if len(f) else -np.inf
                mask=s<tau;mask[perm[:n]]=False
                emit('Uncorrected',dict(skipped_indices=np.flatnonzero(mask),threshold=tau,rank=int(len(f)>0),m=len(f)),n,eps,delta,frac)
        if main or official:
            for name,cum in cumulative.items():
                T=anchor['executed'];found=int(cum[T]);k=prefix_cost(cum,anchor['found'])
                matched.append(dict(**ident,rep=rep,reference=name,executed=T,execution_fraction=T/N,fpc_recall=anchor['recall'],reference_recall=found/M if M else np.nan,recall_difference=anchor['recall']-found/M if M else np.nan,required_same_recall=k,extra_executions=T-k,extra_fraction=(T-k)/N,fpc_found=anchor['found']))
            for b in P['budget_fractions']:
                feasibility.append(dict(**ident,rep=rep,budget=b,feasible=int(anchor['executed']<=round(b*N))))
        if not road and task['scorer']!='HGB-config':
            settings=[('Adaptive',P['fractions'],.1,True)]
            if main:settings += [('Target-5',P['fractions'],.05,True),('Target-20',P['fractions'],.2,True),('Dense-6',P['dense_checkpoints'],.1,True),('Unadjusted',P['fractions'],.1,False)]
            for name,fs,goal,adjust in settings:
                delta=.1/len(fs) if adjust else .1;chosen=None;bad=False
                for k,fraction in enumerate(fs,1):
                    d,n=fixed(fraction,.05,delta)
                    bad |= int(y[d['skipped_indices']].sum())>.05*M+1e-12
                    if chosen is None and ((d['rank']>0 and len(d['skipped_indices'])/N>=goal) or k==len(fs)):chosen=d,n,k,fraction
                d,n,k,fraction=chosen;emit(name,d,n,.05,.1,fraction,k,goal,int(bad))
    ranking['decision_seconds_mean']=float(np.mean(timing))
    return {'decisions':decisions,'curves':curves,'matched':matched,'prices':prices,'feasibility':feasibility,'native':native,'ranking':[ranking]}

def road_tasks():
    sys.path.insert(0,str(ROOT/'vendor/detour/detour_project'))
    from detour.detour import DETOUR
    from detour.road import Road
    from detour.clustering import HierarchicalClusterer,RoadClusterer
    from detour.features import CurvatureBasedRoadFeatureExtractor
    d=read(Path('data/sensodat_fresh.npz'));g=read(Path('data/se_road_geometry.npz'))
    features={int(i):f for i,f in zip(g['indices'],g['features'])}
    coords={int(i):g['xy'][g['offsets'][j]:g['offsets'][j+1]] for j,i in enumerate(g['indices'])}
    class Cached(HierarchicalClusterer):
        def __init__(self):super().__init__('ward')
        def cluster(self,roads):return super().cluster([features[r.id] for r in roads])
    seq=read(Path('data/its4sdc_official_sequences.npz'))['sequence']
    model=ITS4SDCScorer(ROOT/'vendor/its4sdc.onnx')
    splits=read(Path('data/se_road_splits.json'))['splits'];saved={}
    for sp in splits:
        ix,tx=np.array(sp['source_indices']),np.array(sp['target_indices'])
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=P['main_model_seed']).split(d['X_spec'][ix],groups=d['signature'][ix]))
        clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=P['main_model_seed'])
        tic=time.perf_counter()
        with threadpool_limits(limits=4):
            clf.fit(d['X_spec'][ix[tr]],d['y'][ix[tr]]);s=clf.predict_proba(d['X_spec'][tx])[:,1];v=clf.predict_proba(d['X_spec'][ix[va]])[:,1]
        seconds=time.perf_counter()-tic;hf=v[d['y'][ix[va]]==1];cut=float(np.quantile(hf,.05,method='lower')) if len(hf) else -np.inf
        external=model.score_features(seq[tx]);oracle=[Road(int(i),coords[i][:,0],coords[i][:,1],bool(d['y'][i]),False) for i in ix]
        target=[Road(int(i),coords[i][:,0],coords[i][:,1],None,True) for i in tx];lookup={int(i):j for j,i in enumerate(tx)};orders=[];times=[]
        for rep in range(P['road_repetitions']):
            tic=time.perf_counter();order=DETOUR(oracle,target,Cached(),random_seed=P['detour_seed_base']+rep).prioritize(1.)
            times.append(time.perf_counter()-tic);order=np.array([lookup[r.id] for r in order]);assert np.array_equal(np.sort(order),np.arange(len(tx)));orders.append(order)
        a=DETOUR(oracle[:12],target[:12],Cached(),random_seed=0).prioritize(1.)
        b=DETOUR(oracle[:12],target[:12],RoadClusterer(CurvatureBasedRoadFeatureExtractor(6)),random_seed=0).prioritize(1.)
        assert [r.id for r in a]==[r.id for r in b]
        key=sp['source']+'_'+sp['target'];saved[key+'_DETOUR']=np.stack(orders);saved[key+'_HGB']=s;saved[key+'_ITS4SDC']=external
        yield dict(scope='road',subject='sensodat',source=sp['source'],target=sp['target'],scorer='HGB-spec',model_seed=P['main_model_seed'],s=s,y=d['y'][tx],X=d['X_spec'][tx],duration=d['duration_s'][tx],history_cut=cut,target_id=NAMES['sensodat'].index(sp['target']),external=external,detour_orders=orders,fit_score_seconds=seconds,detour_seconds=float(np.mean(times)))
    np.savez_compressed(OUT/'road_orders.npz',**saved)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('scope',choices=['deepscenario','sensodat','openpilot','road']);a=ap.parse_args();start=time.perf_counter();allrows={}
    for task in (road_tasks() if a.scope=='road' else tasks(a.scope)):
        result=evaluate(task)
        for k,v in result.items():allrows.setdefault(k,[]).extend(v)
        print(a.scope,task['source'],task['target'],task['scorer'],task['model_seed'],'done',round(time.perf_counter()-start,1),flush=True)
    for k,v in allrows.items():pd.DataFrame(v).to_csv(OUT/f'{a.scope}_{k}.csv',index=False)
    report=dict(protocol_sha256=sha(ROOT/'protocol_0910.json'),driver_sha256=sha(__file__),decision_sha256=sha(ROOT/'adaptive_triage.py'),threshold_sha256=sha(ROOT/'certskip/threshold.py'),inputs=INPUTS,rows={k:len(v) for k,v in allrows.items()},elapsed_seconds=time.perf_counter()-start,new_simulator_executions=0,outputs={p.name:sha(p) for p in OUT.glob(a.scope+'_*.csv')})
    (OUT/f'{a.scope}_run.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',a.scope,report['rows'],flush=True)
if __name__=='__main__':main()
