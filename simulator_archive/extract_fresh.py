from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""New raw-data ingestion and provenance driver for the revised manuscript.

This driver never opens old pools, result JSON, or cached score arrays. It reuses
the reviewed low-level specification parsers, not the legacy dataset builders or
experiment scripts. Labels and execution duration have separate output arrays.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from certskip import extract as ds
from certskip import extract_sensodat as sd
from certskip.its4sdc import road_points_from_xodr, its4sdc_features


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()


def save(out, subject, rows, provenance, excluded, start):
    arrays = {}
    for k in rows[0]:
        arrays[k] = np.asarray([r[k] for r in rows])
    arrays["subject"] = np.asarray(subject)
    np.savez_compressed(out / (subject + "_fresh.npz"), **arrays)
    with (out / (subject + "_records.csv")).open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=provenance[0]); w.writeheader(); w.writerows(provenance)
    report = {"subject": subject, "records": len(rows), "failures": int(arrays["y"].sum()),
              "feature_count": arrays["X_spec"].shape[1], "config_count": arrays["X_config"].shape[1],
              "excluded": excluded, "seconds": time.perf_counter() - start,
              "npz_sha256": digest(out / (subject + "_fresh.npz")),
              "record_manifest_sha256": digest(out / (subject + "_records.csv"))}
    (out / (subject + "_extraction.json")).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


def deepscenario(rawroot, out):
    start = time.perf_counter(); rows = []; provenance = []; excluded = {}
    for file in sorted(rawroot.glob("*/reward-*/*-scenarios/*-scenario-attributes.csv")):
        folder = file.parent; parts = file.relative_to(rawroot).parts
        gen = ds.STRATEGIES[parts[0]]; reward = ds.REWARDS[parts[1]]
        road_token, environment = parts[2].replace("-scenarios", "").split("-")
        road = int(road_token[4:]) - 1; weather, daypart = environment.split("_")
        env = (0 if weather == "sunny" else 1)*2 + (0 if daypart == "day" else 1)
        with file.open(newline="", encoding="utf8", errors="replace") as f:
            for rec in csv.DictReader(f):
                path = folder / (rec["ScenarioID"] + ".deepscenario")
                base, colour, model = ds.parse_description(rec["Configuration_API_Description"])
                reason = "missing_file" if not path.exists() else "unparsed_configuration" if base is None else None
                if reason: excluded[reason] = excluded.get(reason, 0) + 1; continue
                # The parser's Reader allows only entity/initialization fields.
                # Hour is extracted afresh for each record, not borrowed from a folder.
                numeric = ds.scenario_features(str(path), ds.FRICTION[weather], ds._hour_of(str(path)), colour, model)
                if numeric is None:
                    excluded["missing_ego"] = excluded.get("missing_ego", 0) + 1; continue
                topo = 4 if base == 16 else base//4
                dist = 0 if base == 16 else (base%4)//2
                mane = 0 if base == 16 else base%2
                config = np.concatenate([np.eye(5)[topo], np.eye(2)[dist], np.eye(2)[mane],
                                         np.eye(4)[env], np.eye(4)[road], np.eye(3)[reward]])
                X = np.r_[config, numeric].astype(np.float32)
                sid = path.relative_to(rawroot).as_posix(); source_hash = digest(path)
                signature = hashlib.sha256(X.tobytes()).hexdigest()
                y = int(rec["Attribute[COL]"] == "True")
                rows.append(dict(X_spec=X, X_config=config.astype(np.float32), y=y, generator=gen,
                                 scenario_id=sid, group=file.relative_to(rawroot).as_posix(),
                                 signature=signature, duration_s=np.nan,
                                 severity=ds.COLLISION_TYPES[rec["Attribute[COLT]"]]))
                provenance.append(dict(scenario_id=sid, source_sha256=source_hash, signature=signature, y=y, generator=gen))
        if len(rows) and len(rows)//3000 != (len(rows) - 300)//3000:
            print("DeepScenario extracted", len(rows), "elapsed", round(time.perf_counter()-start), flush=True)
    save(out, "deepscenario", rows, provenance, excluded, start)


def sensodat(rawzip, out):
    start=time.perf_counter(); rows=[]; provenance=[]; excluded={}
    for path, raw in sd.iter_tests(str(rawzip)):
        gen = sd._generator_of(path)
        if gen is None: excluded["unknown_generator"] = excluded.get("unknown_generator", 0)+1; continue
        tree = ET.fromstring(raw); y = sd.outcome_of(tree); X = sd.test_features(tree)
        if y is None or X is None:
            excluded["missing_outcome_or_geometry"] = excluded.get("missing_outcome_or_geometry", 0)+1; continue
        X=X.astype(np.float32); signature=hashlib.sha256(X.tobytes()).hexdigest()
        info=tree.find("./header/sdc_test_info")
        try: duration=float(info.get("test_duration", "nan"))
        except (ValueError, TypeError): duration=np.nan
        sequence=its4sdc_features(road_points_from_xodr(tree))
        rows.append(dict(X_spec=X, X_config=X[:8], y=y, generator=gen, scenario_id=path,
                         group=str(sd._campaign_of(path)), signature=signature, duration_s=duration,
                         severity=y, sequence=sequence))
        provenance.append(dict(scenario_id=path, source_sha256=hashlib.sha256(raw).hexdigest(),
                               signature=signature, y=y, generator=gen))
        if len(rows)%5000==0: print("SensoDat extracted",len(rows),"elapsed",round(time.perf_counter()-start),flush=True)
    save(out, "sensodat", rows, provenance, excluded, start)


if __name__ == "__main__":
    a=argparse.ArgumentParser(); a.add_argument("subject",choices=["deepscenario","sensodat"])
    a.add_argument("--raw",type=Path,required=True); a.add_argument("--out",type=Path,default=ROOT/"data")
    args=a.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    {"deepscenario":deepscenario,"sensodat":sensodat}[args.subject](args.raw,args.out)
