import os,sys,json,csv,time,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';sys.modules['bottleneck']=None
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'adaptive_replication/analysis'))
import numpy as np,pandas as pd
from policies import *
start=time.perf_counter();A=R/'adaptive_replication';O=A/'extension_results';O.mkdir(exist_ok=True)
S=R/'reproducibility';splits=json.loads((S/'data/se_road_splits.json').read_text())['splits'];data=np.load(S/'data/sensodat_fresh.npz');orders=np.load(S/'results/0910/road_orders.npz')
protocol=dict(subject='six paired common-road transfers',N=1024,repetitions=500,epsilon=.05,delta=.1,cap_fraction=.3,DETOUR='Frozen official rankings; ordering r mod 20; convert position to decreasing risk score; no new clustering',policies=['TargetK-fallback','TargetK-cap','Fixed20','EV-count','EV-count-binomial'],source_target='same declared mmin*[1,2,4,8] source prevalence rule',seed='2026191100+10000*targetID+r',purpose='Evaluate frozen native ordering integration under the same omission constraint; outcomes and released model training provenance unchanged',declared_before_new_decision_computations=True,new_training=0,new_simulator_executions=0)
(O/'road_protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
for split in splits:
    src,tgt=split['source'],split['target'];ix=np.array(split['target_indices']);y=data['y'][ix];N=len(y);assert N==1024;cap=round(.3*N);sc=schedule(cap,.05,.1);vals=Predictor(np.zeros(N),'count').values(sc)
    source_rate=float(data['y'][np.array(split['source_indices'])].mean());K=source_target_K(source_rate,N,[45,90,180,360]);tid=['AmbieGen','Frenetic','FreneticV'].index(tgt)
    for rep in range(500):
        seed=2026191100+10000*tid+rep;perm=np.random.default_rng(seed).permutation(N);path=y[perm];tn,tm,tr=target_stop(path,K,cap);en,em,er=ev_stop(path,cap,sc,vals)
        pending=[('TargetK-fallback',tn,tm,'FPC',tr=='cap'),('TargetK-cap',tn,tm,'FPC',False),('Fixed20',round(.2*N),int(path[:round(.2*N)].sum()),'FPC',False),('EV-count',en,em,'FPC',False),('EV-count-binomial',en,em,'Binomial',False)]
        for scorer in ['HGB','ITS4SDC','DETOUR']:
            value=orders[f'{src}_{tgt}_{scorer}']
            if scorer=='DETOUR':
                order=value[rep%20];assert np.array_equal(np.sort(order),np.arange(N));s=np.empty(N);s[order]=(N-np.arange(N))/N
            else:s=value;assert s.shape==(N,) and np.isfinite(s).all()
            for policy,n,m,rule,fallback in pending:
                d=certify(s,y,perm,n,m,.05,.1,rule,fallback)
                rows.append(dict(source=src,target=tgt,scorer=scorer,rep=rep,draw_seed=seed,N=N,M=int(y.sum()),K=K,policy=policy,**d,saving=d['skipped']/N,calibration=n/N,miss=d['hidden']/int(y.sum()),violation=int(d['hidden']>.05*y.sum()+1e-12)))
    print('Road done',src,tgt,flush=True)
df=pd.DataFrame(rows);df.to_csv(O/'road_decisions.csv',index=False)
summary=df.groupby(['scorer','policy'])[['saving','calibration','miss','violation']].mean().reset_index();summary.to_csv(O/'road_summary.csv',index=False)
per=df.groupby(['source','target','scorer','policy'])[['saving','calibration','miss','violation']].mean().reset_index();per.to_csv(O/'road_transfers.csv',index=False)
assert len(df)==45000
(O/'road_manifest.json').write_text(json.dumps(dict(passed=True,rows=len(df),seconds=time.perf_counter()-start,protocol_sha256=hashlib.sha256((O/'road_protocol.json').read_bytes()).hexdigest(),outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('road*.csv')}),indent=2));print(summary.to_string(index=False))
