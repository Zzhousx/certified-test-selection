from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Fresh closed-loop executions of three real release planning components."""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,csv,hashlib,io,json,os,time,zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from openpilot_runtime import ROOT,planner

PROTOCOL=json.loads((ROOT/"protocol.json").read_text())
VERSION_PROTOCOL=json.loads((ROOT/"openpilot_protocol.json").read_text())

def make_specs(phase):
    p=VERSION_PROTOCOL;N=p[phase+"_scenarios"];seed=p[phase+"_seed"]
    rng=np.random.default_rng(seed);rows=[]
    for i in range(N):
        family=i%4;v0=float(rng.uniform(8,30));vlead=0. if family==0 else float(v0*rng.uniform(.6,1.1))
        brake=float(rng.uniform(1.,5.)) if family==1 else 0.
        # The initial gap exceeds a shared constant-braking stopping-distance
        # difference; future events and perception delays can still be difficult.
        min_gap=max(0.,(v0*v0-vlead*vlead)/7.)
        rows.append(dict(scenario_id=f"{phase}-{i:05d}",family=family,initial_speed=v0,lead_speed=vlead,
            initial_gap=min_gap+float(rng.uniform(4,80)),brake=brake,event_time=float(rng.uniform(1,4)),
            appearance_time=float(rng.uniform(.5,3)) if family==2 else 0.,
            throttle_probability=float(rng.uniform(.2,1.)),pitch=float(rng.uniform(-.08,.08)),
            speed_bias=float(rng.uniform(-1,2)),action_bias=float(rng.uniform(.05,1)),
            velocity_error=float(rng.uniform(-1,1)),experimental_mode=bool(i%2),
            duration=30.,seed=seed))
    path=ROOT/"data"/("openpilot_"+phase+"_specs.json")
    payload=json.dumps(rows,indent=2)
    if path.exists() and path.read_text()!=payload:raise RuntimeError("refusing to replace frozen scenario specs")
    path.parent.mkdir(exist_ok=True);path.write_text(payload)
    return rows,path

def execute(tag,spec):
    model=planner(tag,spec["initial_speed"])
    from openpilot.selfdrive.modeld.constants import ModelConstants
    T=np.asarray(ModelConstants.T_IDXS);dt=.05
    speed=spec["initial_speed"];a=0.;x=0.;lead_x=spec["initial_gap"];vlead_prev=spec["lead_speed"]
    traces=[];min_gap=spec["initial_gap"];solver_errors=0;first_collision=None
    tic=time.perf_counter()
    for step in range(round(spec["duration"]/dt)):
        t=step*dt
        vl=max(0.,spec["lead_speed"]-spec["brake"]*max(0.,t-spec["event_time"]))
        visible=t>=spec["appearance_time"]
        lead=NS(status=visible,dRel=max(0.,lead_x-x),vRel=vl-speed,vLead=vl,vLeadK=vl,
                aLeadK=(vl-vlead_prev)/dt,aLeadTau=1.5,modelProb=float(visible))
        vlead_prev=vl
        pred_speed=max(0.,speed+spec["speed_bias"])
        velocity=np.full(len(T),pred_speed);velocity[0]=speed
        gas=spec["throttle_probability"] if t>=spec["event_time"] else 1.
        prediction=NS(position=NS(x=pred_speed*T),velocity=NS(x=velocity),acceleration=NS(x=np.zeros(len(T))),
            temporalPose=NS(trans=[speed+spec["velocity_error"]]),
            meta=NS(disengagePredictions=NS(gasPressProbs=[gas]*6)),
            action=NS(desiredAcceleration=a+spec["action_bias"],shouldStop=False))
        messages={"radarState":NS(leadOne=lead,leadTwo=NS(**vars(lead))),
            "carState":NS(vEgo=speed,aEgo=a,standstill=speed<.01,vCruise=spec["initial_speed"]*3.6,steeringAngleDeg=0.),
            "carControl":NS(orientationNED=[0.,spec["pitch"],0.]),
            "controlsState":NS(longControlState=1,forceDecel=False),
            "selfdriveState":NS(experimentalMode=spec["experimental_mode"],personality=1,enabled=True),
            "liveParameters":NS(angleOffsetDeg=0.),"modelV2":prediction}
        model.update(messages)
        solver_errors+=int(model.mpc.solution_status!=0)
        a=float(model.output_a_target)
        speed=max(0.,speed+a*dt);x+=speed*dt;lead_x+=vl*dt
        gap=lead_x-x;min_gap=min(min_gap,gap)
        traces.append([t+dt,x,lead_x,speed,vl,a,gap,int(model.mpc.solution_status)])
        if not np.isfinite(traces[-1]).all():raise RuntimeError("non-finite trajectory")
        if gap<.4:
            first_collision=t+dt;break
    wall=time.perf_counter()-tic
    # Solver failures are an execution-quality problem, never converted to pass.
    if solver_errors:raise RuntimeError(f"solver errors for {tag} {spec['scenario_id']}: {solver_errors}")
    row=dict(scenario_id=spec["scenario_id"],tag=tag,family=spec["family"],failure=int(first_collision is not None),
        min_gap=min_gap,simulation_seconds=len(traces)*dt,wall_seconds=wall,solver_errors=solver_errors,
        max_deceleration=-min(r[5] for r in traces),final_speed=speed)
    return row,np.asarray(traces,dtype=np.float64)

def worker(args):
    return execute(*args)

def main(tag,phase,limit=None,workers=1):
    specs,specpath=make_specs(phase)
    if limit:specs=specs[:limit]
    folder=ROOT/"results"/"openpilot"/phase;folder.mkdir(parents=True,exist_ok=True)
    suffix="_smoke" if limit else "";rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, zipfile.ZipFile(folder/(tag+suffix+"_traces.zip"),"w",compression=zipfile.ZIP_DEFLATED) as z:
        for i,(row,trace) in enumerate(pool.map(worker,[(tag,s) for s in specs],chunksize=10)):
            s=specs[i];rows.append(row)
            b=io.BytesIO();np.save(b,trace,allow_pickle=False);z.writestr(s["scenario_id"]+".npy",b.getvalue())
            if (i+1)%250==0:print(tag,phase,i+1,"failures",sum(r["failure"] for r in rows),"seconds",round(time.perf_counter()-start),flush=True)
    with (folder/(tag+suffix+"_outcomes.csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    spec_hash=hashlib.sha256(specpath.read_bytes()).hexdigest()
    report=dict(tag=tag,phase=phase,N=len(rows),failures=sum(r["failure"] for r in rows),spec_sha256=spec_hash,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((ROOT/"openpilot_runtime.py").read_bytes()).hexdigest(),
        planner_sha256=hashlib.sha256((ROOT/"vendor/openpilot"/tag/"selfdrive/controls/lib/longitudinal_planner.py").read_bytes()).hexdigest(),
        version_protocol_sha256=hashlib.sha256((ROOT/"openpilot_protocol.json").read_bytes()).hexdigest(),
        workers=workers,wall_seconds=time.perf_counter()-start,scope="Real released planning component; shared numerical plant and synthetic perception; no full-stack driving inference")
    (folder/(tag+suffix+"_run.json")).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("tag",choices=PROTOCOL["openpilot"]["candidate_tags"])
    a.add_argument("phase",choices=["pilot","formal"]);a.add_argument("--limit",type=int);a.add_argument("--workers",type=int,default=1)
    args=a.parse_args();main(args.tag,args.phase,args.limit,args.workers)
