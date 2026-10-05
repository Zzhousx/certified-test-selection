from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Component-test dependency boundary around unmodified release planning code.

The real LongitudinalPlanner, its helpers, LongitudinalMpc, generated C solver,
and released native Acados libraries execute here. This adapter supplies typed
messages and constants in place of the vehicle, process scheduler and message
transport. It does not run perception or the full openpilot service graph.
"""
from pathlib import Path
import ast, importlib, logging, sys, types
from enum import IntEnum
from functools import lru_cache

ROOT=Path(__file__).resolve().parent

def constant(path,name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError((path,name))

@lru_cache(maxsize=1)
def setup(tag):
    base=ROOT/"vendor"/"openpilot"/tag
    def module(name,**attrs):
        parts=name.split(".")
        for i in range(1,len(parts)+1):
            n=".".join(parts[:i])
            if n not in sys.modules:
                m=types.ModuleType(n)
                m.__path__=[str(base.joinpath(*parts[1:i]))] if parts[0]=="openpilot" else []
                sys.modules[n]=m
                if i>1:setattr(sys.modules[".".join(parts[:i-1])],parts[i-1],m)
        m=sys.modules[name]
        for k,v in attrs.items():setattr(m,k,v)
        return m
    module("openpilot")
    class Personality(IntEnum): aggressive=0;standard=1;relaxed=2
    class LongState(IntEnum): off=0;pid=1;stopping=2;starting=3
    module("cereal.log",LongitudinalPersonality=Personality,ModelDataV2=types.SimpleNamespace)
    module("cereal.messaging")  # publishing is outside the component under test
    module("opendbc.car.interfaces",ACCEL_MIN=constant(ROOT/"vendor"/"opendbc_interfaces.py","ACCEL_MIN"),
           ACCEL_MAX=constant(ROOT/"vendor"/"opendbc_interfaces.py","ACCEL_MAX"))
    module("opendbc.car.vehicle_model",ACCELERATION_DUE_TO_GRAVITY=9.81)
    module("openpilot.common.realtime",DT_MDL=constant(base/"common/realtime.py","DT_MDL"),
           DT_CTRL=constant(base/"common/realtime.py","DT_CTRL"))
    logger=logging.getLogger("component");logger.setLevel(logging.WARNING)
    module("openpilot.common.swaglog",cloudlog=logger)
    module("openpilot.selfdrive.controls.radard",_LEAD_ACCEL_TAU=constant(base/"selfdrive/controls/radard.py","_LEAD_ACCEL_TAU"))
    module("openpilot.selfdrive.controls.lib.longcontrol",LongCtrlState=LongState)
    module("openpilot.selfdrive.car.cruise",V_CRUISE_MAX=constant(base/"selfdrive/car/cruise.py","V_CRUISE_MAX"),
           V_CRUISE_UNSET=constant(base/"selfdrive/car/cruise.py","V_CRUISE_UNSET"))
    return base

def planner(tag,initial_speed):
    setup(tag)
    klass=importlib.import_module("openpilot.selfdrive.controls.lib.longitudinal_planner").LongitudinalPlanner
    # A shared numerical vehicle, not a claim of a particular production car.
    params=types.SimpleNamespace(openpilotLongitudinalControl=True,steerRatio=15.,wheelbase=2.7,
                                 longitudinalActuatorDelay=.15,vEgoStopping=.05)
    return klass(params,init_v=initial_speed)
