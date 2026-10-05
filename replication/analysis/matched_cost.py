"""Postprocess all certified decisions against hindsight risk-ranking costs."""
import sys,json
sys.modules['bottleneck']=None
import numpy as np
import pandas as pd
from inputs import ROOT,tasks,identity
from run import task_id
from summarize import ID,summarize_groups

def main():
    assert json.loads((ROOT/'results/run_manifest.json').read_text())['completed']
    summaries=[];count=0
    for subject in ['deepscenario','sensodat','openpilot']:
        for task in tasks(subject):
            name=task_id(task);rows=pd.read_csv(ROOT/'results'/f'{name}_decisions.csv')
            y=task['y'];N=len(y);M=int(y.sum());order=np.argsort(-task['s'],kind='stable');cum=np.r_[0,np.cumsum(y[order])]
            found=M-rows.hidden.to_numpy();required=np.searchsorted(cum,found,side='left')
            assert np.all(cum[required]>=found)
            rows['same_recall_ranking_cost']=required/N
            rows['certification_extra_cost']=(rows.executed-required)/N
            rows['ranking_miss_at_same_executions']=1-cum[rows.executed.to_numpy()]/M if M else 0.
            cols=ID+['rep','method','design','epsilon','delta','executed','hidden','same_recall_ranking_cost','certification_extra_cost','ranking_miss_at_same_executions']
            rows[cols].to_csv(ROOT/'results'/f'{name}_matched_cost.csv',index=False)
            summaries.append(summarize_groups(rows,ID+['method','design','epsilon','delta'],['same_recall_ranking_cost','certification_extra_cost','ranking_miss_at_same_executions']))
            count+=len(rows)
    pd.concat(summaries,ignore_index=True).to_csv(ROOT/'results/matched_cost_summary.csv',index=False)
    (ROOT/'results/matched_cost_validation.json').write_text(json.dumps(dict(completed=True,decisions=count,comparator='Hindsight shortest risk-score prefix achieving each certified decision\u2019s realized failure recall; cannot be used as an outcome-blind stopping rule.'),indent=2))
    print('MATCHED COST COMPLETE',count,flush=True)
if __name__=='__main__':main()
