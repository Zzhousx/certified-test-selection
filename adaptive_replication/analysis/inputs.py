from pathlib import Path
import json, hashlib, csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'reproducibility'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tasks(subject, primary_only=False):
    report=json.loads((SOURCE/f'results/{subject}_run.json').read_text())
    z=np.load(SOURCE/f'results/{subject}_fresh_scores.npz')
    if subject!='openpilot':
        d=np.load(SOURCE/f'data/{subject}_fresh.npz'); names={'deepscenario':['Random','Greedy','RL'],'sensodat':['AmbieGen','Frenetic','FreneticV']}[subject]
        for rec in report['model_records']:
            if rec['scorer']=='ITS4SDC': continue
            if primary_only and (rec['scorer']!='HGB-spec' or rec['model_seed']!=20260905):continue
            si,ti=names.index(rec['source']),names.index(rec['target'])
            key=f"s{si}_t{ti}_{rec['scorer']}_{rec['model_seed']}";ix=z[key+'_target_indices']
            src=d['y'][d['generator']==si]
            yield dict(**rec,s=z[key],y=d['y'][ix],duration=d['duration_s'][ix],target_id=ti+(0 if subject=='deepscenario' else 3),source_rate=float(src.mean()),family=None)
        if subject=='sensodat' and not primary_only:
            scores=np.load(SOURCE/'results/0910/its4sdc_scores.npz')['scores']
            for ti,target in enumerate(names):
                ix=np.flatnonzero(d['generator']==ti)
                yield dict(subject=subject,source='Released model',target=target,scorer='ITS4SDC',model_seed=0,history_cut=.5,s=scores[ix],y=d['y'][ix],duration=d['duration_s'][ix],target_id=ti+3,source_rate=None,family=None)
    else:
        specs=json.loads((SOURCE/'data/openpilot_formal_specs.json').read_text());family=np.array([s['family'] for s in specs])
        def outcomes(version):
            rows=list(csv.DictReader((SOURCE/f'results/openpilot/formal/{version}_outcomes.csv').open()))
            assert [row['scenario_id'] for row in rows]==[s['scenario_id'] for s in specs]
            assert sum(int(row['solver_errors']) for row in rows)==0
            return np.array([int(row['failure']) for row in rows]),np.array([float(row['simulation_seconds']) for row in rows])
        for rec in report['model_records']:
            if primary_only and (rec['scorer']!='HGB-spec' or rec['model_seed']!=20260905):continue
            key=f"{rec['source']}_{rec['target']}_{rec['scorer']}_{rec['model_seed']}";y,dur=outcomes(rec['target']);src,_=outcomes(rec['source'])
            yield dict(**rec,s=z[key],y=y,duration=dur,target_id=6+['v0.9.8','v0.9.9','v0.10.0'].index(rec['target']),source_rate=float(src.mean()),family=family)
    z.close()

def identity(task):
    return {k:task[k] for k in ['subject','source','target','scorer','model_seed']}
