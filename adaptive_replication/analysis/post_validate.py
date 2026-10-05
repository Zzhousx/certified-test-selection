import sys,json,hashlib,os
sys.modules['bottleneck']=None
from pathlib import Path
import numpy as np
import pandas as pd
from inputs import tasks
R=Path(__file__).resolve().parents[1];O=Path(os.environ.get('CTS_ADAPTIVE_OUTPUT',str(R/'results')))
run=json.loads((O/'run_manifest.json').read_text());repetitions=run['repetitions']
def name(t):return '_'.join(str(t[k]).replace(' ','-') for k in ['subject','source','target','scorer','model_seed'])
alltasks=[t for sub in ['deepscenario','sensodat','openpilot'] for t in tasks(sub)]
alltasks=[t for t in alltasks if name(t) in run['task_names']]
checks=0;cap_checks=0;count=[]
for task in alltasks:
    df=pd.read_csv(O/(name(task)+'_decisions.csv'));N=len(task['y']);y,s=task['y'],task['s']
    a=df[df.policy=='TargetK-cap'];b=df[df.policy=='K-grid-cap'];paired=a.merge(b,left_on=['rep','epsilon','delta','selected_K'],right_on=['rep','epsilon','delta','K'],suffixes=['_a','_b'])
    assert len(paired)==3*repetitions
    for col in ['n','m','j','skipped','hidden']:assert np.array_equal(paired[col+'_a'],paired[col+'_b'])
    cap_checks+=len(paired)
    for rep in sorted(set(np.linspace(0,repetitions-1,min(5,repetitions),dtype=int))):
        part=df[df.rep==rep];seed=int(part.draw_seed.iloc[0]);perm=np.random.default_rng(seed).permutation(N)
        cm=np.r_[0,np.cumsum(y[perm])]
        for row in part.itertuples():
            n,m,j=int(row.n),int(row.m),int(row.j)
            assert m==cm[n] and 0<=j<=m and n<=row.cap
            assert row.executed==N-row.skipped and abs(row.calibration-n/N)<1e-14 and abs(row.saving-row.skipped/N)<1e-14
            if j:
                fs=s[perm[:n]][y[perm[:n]]==1];tau=float(np.sort(fs)[j-1]);assert abs(tau-row.tau)<1e-14
                skip=s<tau;skip[perm[:n]]=False
                assert int(skip.sum())==row.skipped and int(y[skip].sum())==row.hidden
            else:assert row.skipped==row.hidden==0 and row.tau==float('-inf')
            checks+=1
    count.append(df[df.policy=='EV-count'][['subject','target','epsilon','delta','rep','n','m']])
c=pd.concat(count);g=c.groupby(['subject','target','epsilon','delta','rep'])[['n','m']].nunique();assert (g==1).all().all()
frozen=json.loads((R/'audit/frozen_inputs.json').read_text())
for record in frozen:assert hashlib.sha256((R.parent/'reproducibility'/record['file']).read_bytes()).hexdigest()==record['sha256']
result=dict(passed=True,tasks=len(alltasks),direct_order_statistic_and_strict_mask_rows=checks,source_K_matches_grid_cap_rows=cap_checks,EV_count_identity_across_all_scorers=True,EV_count_label_states=len(g),frozen_input_files_verified=len(frozen))
(O/'post_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
