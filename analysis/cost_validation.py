import os,sys,json,hashlib
os.environ['OMP_NUM_THREADS']='1';sys.modules['bottleneck']=None
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'adaptive_replication/analysis'))
import numpy as np,pandas as pd
from inputs import tasks
A=R/'adaptive_replication';O=A/'extension_results';O.mkdir(exist_ok=True)
protocol={'purpose':'Execution-duration accounting at the frozen primary decision boundaries','policies':['TargetK-fallback','TargetK-cap','EV-count','EV-score','Fixed20'],'risk':[.05,.1],'new_decisions':False,'duration':'Recorded simulator execution duration for public libraries; simulated seconds for component; not a wall-clock guarantee'}
(O/'duration_protocol.json').write_text(json.dumps(protocol,indent=2))
rows=[];excluded=[]
for subject in ['deepscenario','sensodat','openpilot']:
 for t in tasks(subject):
  if t['model_seed'] not in [20260905,0]:continue
  if subject=='openpilot' and t['source']=='v0.9.9' and t['target']=='v0.10.0':continue
  stem=f"{subject}_{t['source'].replace(' ','-')}_{t['target']}_{t['scorer']}_{t['model_seed']}"
  path=A/'results'/f'{stem}_decisions.csv'
  if not path.exists():print('MISSING',stem);raise RuntimeError(stem)
  df=pd.read_csv(path);df=df[(df.epsilon==.05)&(df.delta==.1)&df.policy.isin(protocol['policies'])]
  N=len(t['y']);s=t['s'];y=t['y'];dur=t['duration']
  if not np.isfinite(dur).all():
   excluded.append({'task':stem,'finite_duration_count':int(np.isfinite(dur).sum()),'N':N});continue
  assert (dur>=0).all()
  order=np.argsort(s,kind='stable');sorted_s=s[order];prefix=np.r_[0,np.cumsum(dur[order])];total=float(dur.sum())
  case=subject if subject!='openpilot' else ('openpilot-stable' if t['target']=='v0.9.9' else 'openpilot-changed')
  for rep,g in df.groupby('rep'):
   seed=int(g.iloc[0].draw_seed);perm=np.random.default_rng(seed).permutation(N)
   for d in g.itertuples():
    n=int(d.n);tau=float(d.tau)
    count=int(np.searchsorted(sorted_s,tau,side='left')) if np.isfinite(tau) else 0
    cal=perm[:n];already=cal[s[cal]<tau] if np.isfinite(tau) else np.array([],dtype=int)
    saved=float(prefix[count]-dur[already].sum())
    assert abs((count-len(already))-d.skipped)<1e-8,(stem,rep,d.policy)
    assert -1e-8<=saved<=total+1e-8
    rows.append(dict(case=case,source=t['source'],target=t['target'],scorer=t['scorer'],rep=rep,policy=d.policy,count_saving=d.skipped/N,duration_saving=saved/total))
df=pd.DataFrame(rows);df.to_csv(O/'duration_decisions.csv',index=False)
df.groupby(['case','scorer','policy'])[['count_saving','duration_saving']].mean().reset_index().to_csv(O/'duration_summary.csv',index=False)
(O/'duration_manifest.json').write_text(json.dumps({'rows':len(df),'passed':True,'excluded_missing_duration':excluded,'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('duration*.csv')}},indent=2))
print('Duration rows',len(df))
