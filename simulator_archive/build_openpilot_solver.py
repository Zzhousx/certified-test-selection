from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Compile the real released MPC once; selected versions have identical MPC code."""
from pathlib import Path
import hashlib,json,os,runpy,shutil,sys
from openpilot_runtime import ROOT,setup

base=setup("v0.9.9")
acados=base/"third_party/acados"
sys.path.insert(0,str(acados))
shutil.copytree(acados/"acados_template",acados/"interfaces/acados_template/acados_template",dirs_exist_ok=True)
(acados/"lib").mkdir(exist_ok=True)
for p in (acados/"x86_64/lib").iterdir():
    if p.stat().st_size>200:shutil.copy2(p,acados/"lib"/p.name)
shutil.copy2(acados/"lib/libqpOASES_e.so.3.1",acados/"lib/libqpOASES_e.so")
os.chmod(acados/"x86_64/t_renderer",0o755)
os.environ["ACADOS_SOURCE_DIR"]=str(acados)
os.environ["ACADOS_PYTHON_INTERFACE_PATH"]=str(acados/"acados_template")
os.environ["TERA_PATH"]=str(acados/"x86_64/t_renderer")
source=base/"selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
for tag in ["v0.9.8","v0.10.0"]:
    other=ROOT/"vendor/openpilot"/tag/"selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
    assert hashlib.sha256(other.read_bytes()).hexdigest()==source_hash
os.chdir(source.parent)
runpy.run_path(str(source),run_name="__main__")
from openpilot.third_party.acados.acados_template import AcadosOcpSolver
generated=source.parent/"c_generated_code"
AcadosOcpSolver.build(str(generated),with_cython=True)
if not list(generated.glob("acados_ocp_solver_pyx*.so")):
    raise RuntimeError("native solver Python extension was not produced")
for tag in ["v0.9.8","v0.10.0"]:
    dest=ROOT/"vendor/openpilot"/tag/"selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code"
    shutil.copytree(generated,dest,dirs_exist_ok=True)
report={"mpc_sha256":source_hash,"versions_share_identical_mpc":True,
        "binaries":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in generated.glob("*.so")}}
(ROOT/"results").mkdir(exist_ok=True)
(ROOT/"results/openpilot_build.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
