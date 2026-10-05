import sys,json
sys.modules['bottleneck']=None
import numpy as np
import pandas as pd
from scipy.stats import beta,t
from inputs import ROOT,sha

ID=['subject','source','target','scorer','model_seed']
OUT=ROOT/'results'
def load(suffix):
    # Only per-task raw outputs; derived fpc_binomial_paired.csv must not enter
    # a second summary pass as if it were an additional experimental task.
    paths=[p for p in OUT.glob('*_'+suffix+'.csv') if p.name.startswith(('deepscenario_','sensodat_','openpilot_'))]
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True) if paths else pd.DataFrame()
def included(df):
    return df[~((df.subject=='openpilot')&(df.source=='v0.9.8')&(df.target=='v0.10.0'))].copy()
def case_names(df):
    df=df.copy();df['case']=df.subject
    df.loc[(df.subject=='openpilot')&(df.target=='v0.9.9'),'case']='openpilot_stable'
    df.loc[(df.subject=='openpilot')&(df.target=='v0.10.0'),'case']='openpilot_changed'
    return df
def ci(k,n):
    return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)
def avg_ci(v):
    v=np.asarray(v,float);mean=float(v.mean());half=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.
    return mean,mean-half,mean+half
def summarize_groups(df,keys,metrics):
    rows=[]
    for key,g in df.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,);row=dict(zip(keys,key));row['repetitions']=len(g)
        for metric in metrics:
            if metric not in g:continue
            values=g[metric].dropna()
            if not len(values):continue
            row[metric+'_mean']=values.mean();row[metric+'_median']=values.median();row[metric+'_p90']=values.quantile(.9)
        for metric in ['violation','legacy_violation','exact_violation']:
            if metric not in g or not g[metric].notna().any():continue
            v=g[metric].dropna();lo,hi=ci(int(v.sum()),len(v));row[metric+'_count']=int(v.sum());row[metric+'_rate']=v.mean();row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
        rows.append(row)
    return pd.DataFrame(rows)
def main():
    run=json.loads((OUT/'run_manifest.json').read_text());assert run['completed']
    d=load('decisions');b=load('budget');p=load('paired');f=load('family');plan=load('planning')
    assert len(d)==run['decision_rows'] and len(b)==run['budget_rows']
    duplicate_checks=0
    for frame,keys in [(d,['method','design','epsilon','delta','rep']),(b,['n_fraction','budget_fraction','rep']),(f,['family','rep'])]:
        for (scorer,seed),g in frame[(frame.subject=='openpilot')&(frame.target=='v0.10.0')].groupby(['scorer','model_seed']):
            a=g[g.source=='v0.9.8'].sort_values(keys).reset_index(drop=True);z=g[g.source=='v0.9.9'].sort_values(keys).reset_index(drop=True)
            columns=[c for c in frame if c not in ID and c!='bound_seconds']
            assert a[columns].equals(z[columns]),(scorer,seed,keys)
            duplicate_checks+=1
    ds=summarize_groups(d,ID+['method','design','epsilon','delta'],['saving','calibration','miss','refusal','attainment','duration_saved','cap_hit'])
    ds.to_csv(OUT/'decision_summary.csv',index=False)
    compare=[]
    fixed=d[d.method.isin(['FPC','Binomial'])]
    for key,g in fixed.groupby(ID+['design','epsilon','delta'],dropna=False,sort=False):
        a=g[g.method=='FPC'].sort_values('rep');z=g[g.method=='Binomial'].sort_values('rep')
        assert np.array_equal(a.rep,z.rep)
        row=dict(zip(ID+['design','epsilon','delta'],key));row['rules_differ']=np.mean(a.j.to_numpy()!=z.j.to_numpy())
        row['saving_difference'],row['saving_difference_ci_low'],row['saving_difference_ci_high']=avg_ci(a.saving.to_numpy()-z.saving.to_numpy())
        row['refusal_difference']=a.refusal.mean()-z.refusal.mean();compare.append(row)
    pd.DataFrame(compare).to_csv(OUT/'fpc_binomial_paired.csv',index=False)
    bs=summarize_groups(b,ID+['n_fraction','budget_fraction'],['miss','eps_hat','legacy_eps_hat','exact_eps_hat','risk_order_miss','bound_seconds'])
    bs.to_csv(OUT/'budget_summary.csv',index=False)
    exact=b[b.exact_eps_hat.notna()].copy();exact['refined_gap']=exact.eps_hat-exact.exact_eps_hat;exact['legacy_gap']=exact.legacy_eps_hat-exact.exact_eps_hat
    summarize_groups(exact,ID+['n_fraction','budget_fraction'],['refined_gap','legacy_gap']).to_csv(OUT/'exact_bound_summary.csv',index=False)
    ps=[]
    for key,g in p.groupby(ID,sort=False):
        row=dict(zip(ID,key))
        for c in p.columns:
            if c.endswith(('_fixed20','_adaptive')):
                row[c+'_mean'],row[c+'_ci_low'],row[c+'_ci_high']=avg_ci(g[c])
        row['K']=int(g.K.iloc[0]);ps.append(row)
    pd.DataFrame(ps).to_csv(OUT/'paired_summary.csv',index=False)
    fs=summarize_groups(f,ID+['family'],['failures','miss','any_family_violation'])
    # Family violation rates are conditional on a frozen family, not covered
    # individually by the unstratified global guarantee.
    fs.to_csv(OUT/'family_summary.csv',index=False)
    plan.to_csv(OUT/'planning_summary.csv',index=False)
    primary=case_names(included(d[(d.scorer=='HGB-spec')&(d.model_seed==20260905)]))
    primary=primary[(primary.epsilon==.05)&(primary.delta==.1)&(((primary.method.isin(['FPC','Binomial']))&(primary.design.astype(str)=='0.2'))|(primary.method=='Adaptive')|((primary.method=='Target-K-FPC')&(primary.primary_K==1))|(primary.method=='Stratified-FPC'))]
    headlines=[]
    for (case,method),g in primary.groupby(['case','method'],sort=False):
        per_rep=g.groupby('rep')[['saving','calibration','miss','refusal','attainment','violation']].mean()
        row=dict(case=case,method=method,transfers=len(g[ID].drop_duplicates()),replications=len(per_rep))
        for metric in per_rep:
            row[metric],row[metric+'_ci_low'],row[metric+'_ci_high']=avg_ci(per_rep[metric])
        headlines.append(row)
    h=pd.DataFrame(headlines);h.to_csv(OUT/'headline_summary.csv',index=False)
    warnings=[]
    for _,row in ds.iterrows():
        if row.violation_ci_low>row.delta:warnings.append({k:row[k] for k in ID+['method','design','epsilon','delta','violation_count','violation_ci_low','violation_ci_high']})
    for _,row in bs.iterrows():
        if row.violation_ci_low>.1:warnings.append({k:row[k] for k in ID+['n_fraction','budget_fraction','violation_count','violation_ci_low','violation_ci_high']})
    summary=dict(completed=True,decision_rows=len(d),budget_rows=len(b),exact_bound_checks=len(exact),duplicate_output_checks=duplicate_checks,refined_exact_gap_max=float(exact.refined_gap.max()),legacy_exact_gap_max=float(exact.legacy_gap.max()),diagnostic_warning_count=len(warnings),warnings=warnings,intervals='95% exact Clopper-Pearson for each violation frequency; paired t intervals for Monte Carlo means. Diagnostic flags are unadjusted exploratory checks, not simultaneous hypothesis tests.',headline=h.to_dict('records'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['headline','warnings']},indent=2))
if __name__=='__main__':main()
