import sys,os,json,hashlib
sys.modules['bottleneck']=None
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta,t
R=Path(__file__).resolve().parents[1];O=Path(os.environ.get('CTS_ADAPTIVE_OUTPUT',str(R/'results')));S=O/'summary';S.mkdir(exist_ok=True)
run=json.loads((O/'run_manifest.json').read_text());repetitions=run['repetitions']
selected=set(run['task_names'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(v):
    v=np.asarray(v);a=float(v.mean());h=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return a,a-h,a+h
group=['subject','source','target','scorer','model_seed','epsilon','delta','policy','rule','K']
metrics=['saving','calibration','miss','refusal','attainment','cap_position','cap_target_not_reached','n','m','j']
records=[];main=[];allframes=[];anchors=[];rowcount=0
old=R.parent/'replication/results'
for p in sorted(O.glob('*_decisions.csv')):
    if p.name.removesuffix('_decisions.csv') not in selected:continue
    df=pd.read_csv(p);rowcount+=len(df)
    assert len(df)==42*repetitions and not df.duplicated(group+['rep']).any()
    mf=json.loads(p.with_name(p.name.replace('_decisions.csv','_manifest.json')).read_text());assert sha(p)==mf['sha256']
    for key,g in df.groupby(group,dropna=False):
        assert len(g)==repetitions and set(g.rep)==set(range(repetitions))
        rec=dict(zip(group,key));rec['repetitions']=len(g)
        for metric in metrics:
            rec[metric+'_mean']=g[metric].mean();rec[metric+'_median']=g[metric].median();rec[metric+'_p90']=g[metric].quantile(.9)
        k=int(g.violation.sum());r=len(g)
        rec.update(violation_count=k,violation_rate=k/r,violation_ci_low=beta.ppf(.025,k,r-k+1) if k else 0,violation_ci_high=beta.ppf(.975,k+1,r-k) if k<r else 1)
        rec['diagnostic_lower_above_delta']=int(rec['violation_ci_low']>rec['delta'])
        rec['score_below_delta_proxy_only']=False
        records.append(rec)
    # Keep equal-transfer per-repetition data for every scorer and model seed.
    x=df[~((df.subject=='openpilot')&(df.source=='v0.9.8')&(df.target=='v0.10.0'))].copy()
    x['case']=x.subject
    x.loc[(x.subject=='openpilot')&(x.target=='v0.9.9'),'case']='openpilot-stable'
    x.loc[(x.subject=='openpilot')&(x.target=='v0.10.0'),'case']='openpilot-changed'
    # Source-selected K is part of one deployable policy, not separate transfer subgroups.
    x.loc[x.policy!='K-grid-cap','K']=0
    allframes.append(x[['case','source','target','scorer','model_seed','epsilon','delta','policy','K','rep']+metrics+['violation']])
    if df.scorer.iloc[0]=='HGB-spec' and df.model_seed.iloc[0]==20260905:main.append(x)
summary=pd.DataFrame(records);summary.to_csv(S/'per_design.csv',index=False)
panels=pd.concat(allframes,ignore_index=True);primary=pd.concat(main,ignore_index=True)
aggregated=[]
agroup=['case','scorer','model_seed','epsilon','delta','policy','K']
for key,g in panels.groupby(agroup):
    row=dict(zip(agroup,key));row['transfers']=g[['source','target']].drop_duplicates().shape[0]
    rep=g.groupby('rep')[metrics+['violation']].mean()
    for metric in metrics+['violation']:
        a,l,u=ci(rep[metric]);row.update({metric+'_mean':a,metric+'_ci_low':l,metric+'_ci_high':u})
    aggregated.append(row)
agg=pd.DataFrame(aggregated);agg.to_csv(S/'aggregate_all_scorers.csv',index=False)
headline=agg[(agg.scorer=='HGB-spec')&(agg.model_seed==20260905)];headline.to_csv(S/'headline.csv',index=False)
differences=[];hindsight=[]
for (case,e,d),g in primary.groupby(['case','epsilon','delta']):
    ordinary=g[g.policy!='K-grid-cap'].groupby(['rep','policy']).saving.mean().unstack()
    grid=g[g.policy=='K-grid-cap'];chosen=[]
    for (src,tgt),tr in grid.groupby(['source','target']):
        k=int(tr.groupby('K').saving.mean().idxmax());chosen.append(tr[tr.K==k]);hindsight.append(dict(case=case,source=src,target=tgt,epsilon=e,delta=d,best_K=k,saving_mean=tr[tr.K==k].saving.mean()))
    ordinary['BestK-hindsight']=pd.concat(chosen).groupby('rep').saving.mean()
    for policy in ['EV-count','EV-score','EV-flat-attachment','TargetK-cap']:
        for baseline in ['TargetK-fallback','TargetK-cap','Fixed20','Count-checkpoints','BestK-hindsight']:
            if policy==baseline:continue
            diff=ordinary[policy]-ordinary[baseline];a,l,u=ci(diff)
            differences.append(dict(case=case,epsilon=e,delta=d,policy=policy,baseline=baseline,difference_pp=100*a,ci_low_pp=100*l,ci_high_pp=100*u,repetitions=len(diff),interval_scope='paired calibration draws conditional on frozen transfers; exploratory, unadjusted'))
pd.DataFrame(differences).to_csv(S/'paired_headline.csv',index=False);pd.DataFrame(hindsight).to_csv(S/'hindsight_K.csv',index=False)
# FPC/binomial certificate comparison uses the very same stop for paired versions.
certcomp=[]
for policy,other in [('EV-count','EV-count-binomial'),('TargetK-cap','TargetK-cap-binomial')]:
    for key,g in primary.groupby(['case','epsilon','delta']):
        ix=['source','target','rep'];a=g[g.policy==policy].set_index(ix).sort_index();b=g[g.policy==other].set_index(ix).sort_index()
        assert np.array_equal(a.n,b.n) and np.array_equal(a.m,b.m)
        certcomp.append(dict(case=key[0],epsilon=key[1],delta=key[2],policy=policy,draws=len(a),changed_rank=int((a.j!=b.j).sum()),saving_gain_pp=100*(a.saving-b.saving).mean()))
pd.DataFrame(certcomp).to_csv(S/'FPC_vs_binomial.csv',index=False)
main_design=summary[(summary.scorer=='HGB-spec')&(summary.model_seed==20260905)&(summary.epsilon==.05)&(summary.delta==.1)]
main_design.to_csv(S/'headline_by_transfer.csv',index=False)
main_ev=primary[(primary.policy=='EV-count')&(primary.epsilon==.05)&(primary.delta==.1)]
distribution=main_ev.groupby(['case','source','target','n','m','j','stop_reason']).size().reset_index(name='draws');distribution.to_csv(S/'EV_count_stops.csv',index=False)
v=summary[summary.policy.isin(['EV-count','EV-score','TargetK-cap'])]
validation=dict(passed=True,tasks=run["tasks"],rows=rowcount,designs=len(summary),draws_per_design=repetitions,expected_rows=run["tasks"]*42*repetitions,all_rows_accounted_for=rowcount==run["rows"],old_anchor_comparisons=len(anchors),old_anchor_draws=sum(x['draws'] for x in anchors),anchors=anchors,
    per_design_lower_above_delta_count=int(v.diagnostic_lower_above_delta.sum()),note='Unadjusted per-design binomial intervals are diagnostics, not a proof or familywise rejection; transfers and scorer views are correlated.',protocol_sha256=sha(R/'protocol.json'))
assert validation['all_rows_accounted_for'] and len(summary)==run["tasks"]*42
(S/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps({k:v for k,v in validation.items() if k!='anchors'},indent=2))
print(headline[(headline.epsilon==.05)&(headline.delta==.1)&(headline.policy.isin(['TargetK-fallback','TargetK-cap','EV-count','EV-score','Fixed20']))][['case','policy','saving_mean','calibration_mean','violation_mean']].to_string(index=False))
