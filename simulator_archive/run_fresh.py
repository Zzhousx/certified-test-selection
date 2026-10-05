from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Fresh model fitting and paired fixed-design calibration experiments.

Inputs are extract_fresh.py outputs and the released ONNX model. This module
does not import any legacy experiment/export/figure script or result file.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import numpy as np
import scipy, sklearn
from scipy.stats import beta, t
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from certskip.threshold import (finite_population_index, checked_tolerance_index,
                               binomial_tolerance_index)
from certskip.its4sdc import ITS4SDCScorer
PROTOCOL=json.loads((ROOT/"protocol.json").read_text())
NAMES={"deepscenario":["Random","Greedy","RL"],"sensodat":["AmbieGen","Frenetic","FreneticV"]}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mean_ci(a):
    a=np.asarray(a,float); mu=float(a.mean())
    hw=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return mu,mu-hw,mu+hw

def binomial_ci(k,n):
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))

def metrics(y,s):
    return {"auc":float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None,
            "ap":float(average_precision_score(y,s)),"failure_rate":float(np.mean(y))}

def fit_model(X,y,seed,kind="HGB"):
    if len(np.unique(y))!=2:raise ValueError("training set must contain both labels")
    model=(HistGradientBoostingClassifier(random_state=seed,**PROTOCOL["model"]) if kind=="HGB"
           else make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.,random_state=seed)))
    with threadpool_limits(limits=4):model.fit(X,y)
    return model

def configs(full=True):
    eps=PROTOCOL["main_epsilon"];delta=PROTOCOL["main_delta"];frac=PROTOCOL["main_calibration_fraction"]
    if not full:return [(frac,eps,delta)]
    result={(f,eps,delta) for f in PROTOCOL["calibration_fractions"]}
    result|={(frac,e,delta) for e in PROTOCOL["epsilon_sensitivity"]}
    result|={(frac,eps,d) for d in PROTOCOL["delta_sensitivity"]}
    return sorted(result)

def evaluate(writer,subject,source,target,scorer,model_seed,scores,y,durations,history_cut,full=True):
    N=len(y);M=int(y.sum());order_key=202609050+N
    for frac,eps,delta in configs(full):
        n=max(1,min(N-1,round(frac*N)))
        methods=PROTOCOL["methods"] if history_cut is not None else ["FPC","BT-checked","CRC","Native"]
        for rep in range(PROTOCOL["repetitions"]):
            permutation=np.random.default_rng(order_key+rep).permutation(N)
            c=permutation[:n];rest=permutation[n:]; f=np.sort(scores[c][y[c]==1]);m=len(f)
            j_bt=binomial_tolerance_index(m,eps,delta)
            j_checked=checked_tolerance_index(m,N,n,eps,delta)
            j_fpc=finite_population_index(m,N,n,eps,delta)
            j_crc=int(np.floor(eps*(m+1)+1e-12))
            thresholds={"FPC":float(f[j_fpc-1]) if j_fpc else -np.inf,
                        "BT-checked":float(f[j_checked-1]) if j_checked else -np.inf,
                        "CRC":float(f[j_crc-1]) if j_crc else -np.inf,
                        "Historical":history_cut,
                        "Uncorrected":float(np.quantile(f,eps)) if m else -np.inf,
                        "Native":.5}
            for method in methods:
                tau=thresholds[method]
                uncalibrated=method in ["Historical","Native"]
                if uncalibrated:
                    skipped=np.flatnonzero(scores<=tau if method=="Native" else scores<tau)
                    nc=mf=0
                else:
                    skipped=rest[scores[rest]<tau];nc=n;mf=m
                hidden=int(y[skipped].sum());miss=hidden/M if M else 0.
                loss_remaining=hidden/(M-mf) if M>mf else 0.
                duration_valid=np.isfinite(durations).all() and np.all(durations>=0) and durations.sum()>0
                time_saved=float(durations[skipped].sum()/durations.sum()) if duration_valid else None
                writer.writerow(dict(subject=subject,source=source,target=target,scorer=scorer,model_seed=model_seed,
                    fraction=frac,epsilon=eps,delta=delta,rep=rep,draw_seed=order_key+rep,N=N,M=M,n=nc,m=mf,
                    method=method,threshold=tau if np.isfinite(tau) else "-inf",skipped=len(skipped),
                    hidden=hidden,skip_rate=len(skipped)/N,executed=N-len(skipped),miss=miss,
                    remaining_miss=loss_remaining,violation=int(miss>eps+1e-12),
                    no_threshold=int(not np.isfinite(tau)),time_saved=time_saved,
                    bt_capped=int(j_checked!=j_bt) if method=="BT-checked" else 0))
        print(subject,source,"->",target,scorer,model_seed,"n",n,"eps",eps,"delta",delta,"done",flush=True)

FIELDS="subject source target scorer model_seed fraction epsilon delta rep draw_seed N M n m method threshold skipped hidden skip_rate executed miss remaining_miss violation no_threshold time_saved bt_capped".split()

def run_subject(subject,out):
    start=time.perf_counter(); poolpath=ROOT/"data"/(subject+"_fresh.npz")
    with np.load(poolpath) as archive:
        d={key:archive[key] for key in archive.files}
    out.mkdir(parents=True,exist_ok=True)
    model_records=[];split_records=[];scores_store={}; overview=[]
    for g,name in enumerate(NAMES[subject]):
        ix=np.flatnonzero(d["generator"]==g)
        overview.append(dict(subject=subject,generator=name,N=len(ix),M=int(d["y"][ix].sum()),
            duplicate_signatures=int(len(ix)-len(np.unique(d["signature"][ix]))),
            finite_durations=int(np.isfinite(d["duration_s"][ix]).sum())))
    with (out/(subject+"_draws.csv")).open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for seed in PROTOCOL["model_seeds"]:
            for s,source in enumerate(NAMES[subject]):
                all_source=np.flatnonzero(d["generator"]==s)
                # The historical split is independent of outcomes and defined by
                # full specification signatures, so identical inputs stay together.
                train_rel,valid_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(
                    d["X_spec"][all_source],groups=d["signature"][all_source]))
                train0=all_source[train_rel];valid=all_source[valid_rel]
                for t,target in enumerate(NAMES[subject]):
                    if s==t:continue
                    ix=np.flatnonzero(d["generator"]==t)
                    target_signatures=set(d["signature"][ix].tolist())
                    # Source/target specification overlap is removed from both
                    # source training and historical threshold validation.
                    train=np.asarray([i for i in train0 if d["signature"][i] not in target_signatures],int)
                    val=np.asarray([i for i in valid if d["signature"][i] not in target_signatures],int)
                    if not len(val) or not d["y"][val].sum():raise ValueError("source holdout lacks failures")
                    for kind in (["HGB-spec","HGB-config","Logistic-spec"] if seed==PROTOCOL["model_seeds"][0] else ["HGB-spec"]):
                        feats="X_config" if kind=="HGB-config" else "X_spec"
                        model_start=time.perf_counter()
                        model=fit_model(d[feats][train],d["y"][train],seed,"Logistic" if kind.startswith("Logistic") else "HGB")
                        with threadpool_limits(limits=4):
                            hist=model.predict_proba(d[feats][val])[:,1]
                            score=model.predict_proba(d[feats][ix])[:,1]
                        hist_cut=float(np.quantile(hist[d["y"][val]==1],PROTOCOL["main_epsilon"],method="lower"))
                        row=dict(subject=subject,source=source,target=target,scorer=kind,model_seed=seed,
                                 n_train=len(train),n_history=len(val),removed_overlap=len(train0)+len(valid)-len(train)-len(val),
                                 history_cut=hist_cut,fit_score_seconds=time.perf_counter()-model_start,**metrics(d["y"][ix],score))
                        model_records.append(row)
                        key=f"s{s}_t{t}_{kind}_{seed}";scores_store[key]=score;scores_store[key+"_target_indices"]=ix
                        split_records.append(dict(key=key,train=train.tolist(),history=val.tolist(),target=ix.tolist()))
                        if kind!="HGB-config":
                            evaluate(writer,subject,source,target,kind,seed,score,d["y"][ix],d["duration_s"][ix],hist_cut,
                                     full=kind=="HGB-spec" and seed==PROTOCOL["model_seeds"][0])
                        f.flush()
        if subject=="sensodat":
            scorer=ITS4SDCScorer(ROOT/"vendor"/"its4sdc.onnx")
            tic=time.perf_counter();s=scorer.score_features(d["sequence"]);seconds=time.perf_counter()-tic
            scores_store["ITS4SDC"]=s
            for target_id,target in enumerate(NAMES[subject]):
                ix=np.flatnonzero(d["generator"]==target_id)
                model_records.append(dict(subject=subject,source="Released model",target=target,scorer="ITS4SDC",model_seed=0,
                    fit_score_seconds=seconds,model_sha256=scorer.model_sha256,**metrics(d["y"][ix],s[ix])))
                evaluate(writer,subject,"Released model",target,"ITS4SDC",0,s[ix],d["y"][ix],d["duration_s"][ix],None)
                f.flush()
    np.savez_compressed(out/(subject+"_fresh_scores.npz"),**scores_store)
    (out/(subject+"_splits.json")).write_text(json.dumps(split_records),encoding="utf8")
    report=dict(subject=subject,protocol_sha256=sha(ROOT/"protocol.json"),input_sha256=sha(poolpath),
                model_records=model_records,overview=overview,elapsed_seconds=time.perf_counter()-start,
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                result_sha256=sha(out/(subject+"_draws.csv")))
    (out/(subject+"_run.json")).write_text(json.dumps(report,indent=2),encoding="utf8")
    print(subject,"COMPLETE",report["elapsed_seconds"],flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("subject",choices=list(NAMES));a.add_argument("--out",type=Path,default=ROOT/"results")
    args=a.parse_args();run_subject(args.subject,args.out)
