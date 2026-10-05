from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Fetch public, pinned experiment/build inputs. Never execute fetched files."""
from pathlib import Path
import concurrent.futures, hashlib, io, json, tarfile, zipfile
import requests

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/"vendor"
TAGS={"v0.9.8":"fb7b9c0f9420d228f03362970ebcfb7237095cf3",
      "v0.9.9":"8aadf02b2fd91f4e1285e18c2c7feb32d93b66f5",
      "v0.10.0":"c085b8af19438956c15592828bd082803f43dfaf"}

def fetch(url, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=(20,120));r.raise_for_status()
            path.write_bytes(r.content)
            print(path.name,len(r.content),flush=True)
            return {"path":str(path.relative_to(ROOT)),"url":url,"sha256":hashlib.sha256(r.content).hexdigest(),"bytes":len(r.content)}
        except requests.RequestException:
            if attempt==2:raise

def release(item):
    tag,sha=item;archive=VENDOR/(tag+".tar.gz")
    row=fetch("https://codeload.github.com/commaai/openpilot/tar.gz/"+sha,archive)
    dest=VENDOR/"openpilot"/tag;dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            # Extract regular files only; ignore archive links and device entries.
            if not member.isfile():continue
            rel=Path(*Path(member.name).parts[1:]); target=(dest/rel).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError(member.name)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(t.extractfile(member).read())
    return row

def main():
    VENDOR.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:rows=list(p.map(release,TAGS.items()))
    assets={
      "its4sdc.onnx":"https://raw.githubusercontent.com/christianbirchler-org/sdc-testing-competition/main/tools/selectors/icst2025_selectors/ITS4SDC/its4sdc.onnx",
      "its4sdc_main.py":"https://raw.githubusercontent.com/christianbirchler-org/sdc-testing-competition/main/tools/selectors/icst2025_selectors/ITS4SDC/main.py",
      "opendbc_interfaces.py":"https://raw.githubusercontent.com/commaai/opendbc/e370a036073bf38856f3cd9936c0bd44ff349dea/opendbc/car/interfaces.py",
      "opendbc_car.capnp":"https://raw.githubusercontent.com/commaai/opendbc/e370a036073bf38856f3cd9936c0bd44ff349dea/opendbc/car/car.capnp"}
    for name,url in assets.items():rows.append(fetch(url,VENDOR/name))
    # The repository stores native solver dependencies in Git LFS. Download
    # actual Linux libraries, then verify them against the source LFS pointers.
    for name in ["lib/libacados.so","lib/libblasfeo.so","lib/libhpipm.so","lib/libqpOASES_e.so.3.1","t_renderer"]:
        rel="third_party/acados/x86_64/"+name
        dest=VENDOR/"openpilot"/"v0.9.9"/rel
        pointer=dest.read_text();expected=next(l.split(":",1)[1] for l in pointer.splitlines() if l.startswith("oid sha256:"))
        row=fetch("https://media.githubusercontent.com/media/commaai/openpilot/"+TAGS["v0.9.9"]+"/"+rel,dest)
        if row["sha256"]!=expected:raise ValueError("LFS hash mismatch: "+name)
        rows.append(row)
    (VENDOR/"asset_manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf8")

if __name__=="__main__":main()
