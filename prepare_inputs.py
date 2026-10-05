"""Stage frozen inputs from the replication archive, without publishing data."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, zipfile
from experiment_protocols import PROTOCOLS

ROOT=Path(__file__).resolve().parent
SUBJECTS=['deepscenario','sensodat','openpilot']
FILES=[f'data/{s}_fresh.npz' for s in SUBJECTS[:2]]+[f'results/{s}_{tail}' for s in SUBJECTS for tail in ['fresh_scores.npz','run.json','splits.json']]
FILES+=['data/openpilot_formal_specs.json','results/0910/its4sdc_scores.npz',
        'data/se_road_splits.json','data/se_road_geometry.npz','data/its4sdc_official_sequences.npz',
        'results/0910/road_orders.npz']
FILES+=[f'results/openpilot/formal/{v}_outcomes.csv' for v in ['v0.9.8','v0.9.9','v0.10.0']]
EXTRAS=['reproducibility/vendor/its4sdc.onnx']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def initialize():
    for rel,settings in PROTOCOLS.items():
        p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(settings,indent=2)+'\n'
        if p.exists() and json.loads(p.read_text())!=settings:
            raise ValueError(f'Existing protocol differs: {rel}')
        p.write_text(payload,encoding='utf-8')
    for rel in ['adaptive_replication/audit','adaptive_replication/results',
                'replication/results','reproducibility/data','reproducibility/results/0910',
                'simulator_archive/data','simulator_archive/results']:
        (ROOT/rel).mkdir(parents=True,exist_ok=True)

def allowed_vendor(rel):
    p=PurePosixPath(rel)
    if p.is_absolute() or '..' in p.parts:return False
    return rel.startswith('reproducibility/vendor/detour/') and (p.suffix in ['.py','.toml'] or p.name in ['LICENSE','README.md'])

def destination(rel):
    p=(ROOT/PurePosixPath(rel)).resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError(f'Invalid archive path: {rel}')
    return p

def stage_file(stream,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    scratch=target.with_name(target.name+'.preparing')
    with scratch.open('wb') as f:shutil.copyfileobj(stream,f)
    if target.exists() and digest(target)!=digest(scratch):
        scratch.unlink();raise ValueError(f'Existing input differs: {target.relative_to(ROOT)}')
    scratch.replace(target)

def locate_archive(z,rel):
    candidates=[n for n in z.namelist() if n==rel or n.endswith('/'+rel)]
    if len(candidates)>1:raise ValueError(f'Ambiguous archive entry: {rel}')
    return candidates[0] if candidates else None

def prepare(source=None,archive=None):
    initialize()
    if source is None and archive is None:
        print('Initialized experiment settings. Supply --source or --archive to stage data.');return
    z=zipfile.ZipFile(archive) if archive else None
    records=[]
    try:
        for rel in FILES:
            alternatives=['adaptive_replication/frozen_inputs/'+rel,'reproducibility/'+rel]
            found=False
            for candidate in alternatives:
                member=locate_archive(z,candidate) if z else None
                p=source/candidate if source else None
                if (z and member) or (p and p.is_file()):
                    target=destination('reproducibility/'+rel)
                    with (z.open(member) if z else p.open('rb')) as f:stage_file(f,target)
                    records.append({'file':rel,'sha256':digest(target)});found=True;break
            if not found:raise FileNotFoundError('Replication input missing: '+rel)
        optional=list(EXTRAS)
        if z:
            for n in z.namelist():
                parts=PurePosixPath(n).parts
                for i in range(len(parts)):
                    rel='/'.join(parts[i:])
                    if allowed_vendor(rel):optional.append(rel);break
        else:
            vendor=source/'reproducibility/vendor/detour'
            if vendor.exists():optional += [p.relative_to(source).as_posix() for p in vendor.rglob('*') if p.is_file() and allowed_vendor(p.relative_to(source).as_posix())]
        for rel in sorted(set(optional)):
            member=locate_archive(z,rel) if z else None
            p=source/rel if source else None
            if (z and member) or (p and p.is_file()):
                with (z.open(member) if z else p.open('rb')) as f:stage_file(f,destination(rel))
        (ROOT/'adaptive_replication/audit/frozen_inputs.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(f'Prepared {len(records)} frozen input files. Data and models are ignored by Git.')
    finally:
        if z:z.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    choices=ap.add_mutually_exclusive_group()
    choices.add_argument('--source',type=Path,help='Extracted replication package root')
    choices.add_argument('--archive',type=Path,help='Full replication ZIP')
    a=ap.parse_args()
    prepare(a.source.resolve() if a.source else None,a.archive.resolve() if a.archive else None)
