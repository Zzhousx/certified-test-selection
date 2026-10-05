# -*- coding: utf-8 -*-
"""Pre-execution feature extraction from DeepScenario files.

The leakage boundary is enforced here, at the parser, not in a review checklist.
A scenario file holds two kinds of content:

    specification   <Environment>, <Entities>, <Initialization>
                    -- exists before the scenario is run
    execution trace <Story> / <WayPoint>
                    -- exists only after the run

Only the first kind may be read. `LEAKY_ELEMENTS` names the elements that are off
limits and `assert_no_leak` checks a parsed tree against that list. A model built on
trace statistics will score well and will be useless for deciding whether to simulate,
because the trace does not exist at decision time.

One trap worth naming: a finite difference at the first index is not a pre-execution
quantity. `numpy.gradient` uses a forward difference there, which reads the second
sample. Acceleration and jerk "at t=1" are results of running the scenario.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["FEATURE_NAMES", "LEAKY_ELEMENTS", "assert_no_leak", "Reader",
           "parse_description",
           "scenario_features"]

LEAKY_ELEMENTS = ("Story", "ObjectAction", "WayPoint", "DynamicParameters",
                  "GeographicParameters")

MODELS = ("SchoolBus", "SUV", "Jeep", "BoxTruck", "Hatchback", "Sedan")
FRICTION = {"sunny": 0.9, "rain": 0.6}
COLOR_VECTORS = {
    "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "white": (1, 1, 1),
    "black": (0, 0, 0), "yellow": (1, 1, 0), "pink": (1, 0, 1), "skyblue": (0, 1, 1),
}
COLLISION_TYPES = {"None": 0, "npc_vehicle": 1, "pedestrian": 2, "obstacle": 3}
STRATEGIES = {"random-strategy": 0, "greedy-strategy": 1, "rl_based-strategy": 2}
REWARDS = {"reward-dto": 0, "reward-jerk": 1, "reward-ttc": 2}

FEATURE_NAMES = [
    "ego_speed", "ego_vlong", "ego_vlat", "ego_yawrate", "ego_x", "ego_z",
    "n_npc", "n_ped", "n_agents", "n_within20", "n_within50",
    "m_schoolbus", "m_suv", "m_jeep", "m_boxtruck", "m_hatchback", "m_sedan",
    "near_dist", "near_long", "near_lat", "near_vlong", "near_vlat",
    "near_closing", "near_ttc", "near_speed",
    "cfg_dist", "cfg_long", "cfg_lat", "cfg_vlong", "cfg_vlat",
    "cfg_closing", "cfg_ttc", "cfg_speed", "cfg_matched",
    "min_dist", "mean_dist", "min_ttc", "max_closing", "max_npc_speed",
    "mean_npc_speed", "friction", "hour_cos", "hour_sin",
]


class LeakageError(RuntimeError):
    """Raised when extraction touches an element that only exists after execution."""


def assert_no_leak(element_names) -> None:
    """Fail loudly if any post-execution element appears in `element_names`."""
    bad = sorted(set(element_names) & set(LEAKY_ELEMENTS))
    if bad:
        raise LeakageError(
            "extraction touched post-execution elements: %s. These exist only after "
            "the scenario has been run and cannot inform the decision to run it."
            % ", ".join(bad))


class Reader:
    """The only way this module reads a parsed scenario, and a record of what it read.

    Every lookup goes through `findall` or `find`, and each one records the element
    type it asked for. `check()` then tests that record against `LEAKY_ELEMENTS`.

    The point of routing reads through one object is that the record cannot drift from
    the reads. A guard fed by a hand-maintained list of element names only fails when
    the person adding the leak also remembers to declare it, which is the moment they
    are least likely to.
    """

    def __init__(self, root):
        self.root = root
        self.touched = set()

    @staticmethod
    def _tag(path):
        return path.replace(".//", "").rsplit("/", 1)[-1]

    def findall(self, path):
        self.touched.add(self._tag(path))
        return self.root.findall(path)

    def find(self, element, path):
        self.touched.add(self._tag(path))
        return None if element is None else element.find(path)

    def check(self):
        assert_no_leak(self.touched)


def parse_description(text: str):
    """Configuration API description -> (base17, colour word, model word).

    17 base behaviours: 4 topologies x near/far x keep/change lane, plus pedestrian.
    """
    if "pedestrian is crossing" in text:
        return 16, None, None
    m = re.match(r"^A\s+(\S+)\s+(\S+)\s+is\s+", text)
    colour, model = (m.group(1), m.group(2)) if m else (None, None)
    if "driving ahead" in text:
        topo = 0
    elif "overtaking" in text:
        topo = 1
    elif "opposite direction" in text:
        topo = 2
    elif "crossing the road" in text:
        topo = 3
    else:
        return None, None, None
    far = 1 if "(far)" in text else 0
    switching = 1 if "switching lane" in text else 0
    return topo * 4 + far * 2 + switching, colour, model


def _hour_of(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(14):
            m = re.search(r'dateTime="\d{4}-\d{2}-\d{2} (\d{2}):', fh.readline())
            if m:
                return int(m.group(1))
    return 12


def _identify_configured(reader, colour, model):
    """Locate the actor the description names, by colour and vehicle model."""
    want = COLOR_VECTORS.get((colour or "").lower())
    candidates = []
    for obj in reader.findall(".//ScenarioObject"):
        params = reader.find(obj, "ObjectParameter")
        kind = params.get("ObjectType")
        if kind == "Ego":
            continue
        if model is None:                       # pedestrian scenario
            if kind == "Pedestrian":
                candidates.append(obj.get("name"))
            continue
        if obj.get("model") != model:
            continue
        vec = params.get("ObjectColorVector")
        if want is not None and vec:
            nums = tuple(round(float(v)) for v in re.findall(r"-?\d+\.?\d*", vec))
            if nums != want:
                continue
        candidates.append(obj.get("name"))
    return candidates[0] if len(candidates) == 1 else None


def scenario_features(path: str, friction: float, hour: int,
                      colour: str | None, model: str | None,
                      check_leak: bool = True) -> np.ndarray | None:
    """43 numeric pre-execution features for one scenario file."""
    reader = Reader(ET.parse(path).getroot())

    kind, model_of = {}, {}
    for obj in reader.findall(".//ScenarioObject"):
        kind[obj.get("name")] = reader.find(obj, "ObjectParameter").get("ObjectType")
        model_of[obj.get("name")] = obj.get("model")
    target = _identify_configured(reader, colour, model)

    ego, others = None, []
    for init in reader.findall(".//ObjectInitialization"):
        ref = init.get("objectRef")
        pos = reader.find(init, "ObjectInitialPosition")
        rot = reader.find(init, "ObjectInitialRotation")
        vel = reader.find(init, "ObjectInitialVelocity")
        avel = reader.find(init, "ObjectInitialAngularVelocity")
        if pos is None or vel is None:
            continue
        rec = (ref,
               float(pos.get("positionX")), float(pos.get("positionZ")),
               float(rot.get("rotationY")) if rot is not None else 0.0,
               float(vel.get("velocityX")), float(vel.get("velocityZ")),
               float(avel.get("angularVelocityY")) if avel is not None else 0.0)
        if kind.get(ref) == "Ego" or "Ego" in (ref or ""):
            ego = rec
        else:
            others.append(rec)
    if check_leak:
        reader.check()
    if ego is None:
        return None

    _, ex, ez, eyaw, evx, evz, eyaw_rate = ego
    yaw = math.radians(eyaw)
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    lateral = np.array([math.cos(yaw), -math.sin(yaw)])
    ego_v = np.array([evx, evz])
    ego_p = np.array([ex, ez])

    def relative(rec):
        _, x, z, _, vx, vz, _ = rec
        rp = np.array([x, z]) - ego_p
        rv = np.array([vx, vz]) - ego_v
        dist = float(np.linalg.norm(rp))
        unit = rp / max(dist, 1e-6)
        closing = float(-(rv @ unit))
        ttc = dist / closing if closing > 1e-3 else 999.0
        return (dist, float(rp @ forward), float(rp @ lateral),
                float(rv @ forward), float(rv @ lateral),
                closing, min(ttc, 999.0), float(np.linalg.norm([vx, vz])))

    rels = [relative(o) for o in others]
    counts = [sum(1 for o in others if model_of.get(o[0]) == m) for m in MODELS]
    n_npc = sum(1 for o in others if kind.get(o[0]) == "NPC")
    n_ped = sum(1 for o in others if kind.get(o[0]) == "Pedestrian")

    if rels:
        dists = np.array([r[0] for r in rels])
        ttcs = np.array([r[6] for r in rels])
        nearest = rels[int(dists.argmin())]
        aggregate = [float(dists.min()), float(dists.mean()), float(ttcs.min()),
                     float(max(r[5] for r in rels)), float(max(r[7] for r in rels)),
                     float(np.mean([r[7] for r in rels]))]
        within20, within50 = int((dists < 20).sum()), int((dists < 50).sum())
    else:
        nearest = (999.0, 0, 0, 0, 0, 0, 999.0, 0)
        aggregate = [999.0, 999.0, 999.0, 0.0, 0.0, 0.0]
        within20 = within50 = 0

    configured, matched = nearest, 0.0
    if target is not None:
        for o, r in zip(others, rels):
            if o[0] == target or (o[0] or "").endswith(target):
                configured, matched = r, 1.0
                break

    row = ([float(np.linalg.norm(ego_v)), float(ego_v @ forward), float(ego_v @ lateral),
            eyaw_rate, ex, ez,
            float(n_npc), float(n_ped), float(len(others)),
            float(within20), float(within50)]
           + [float(c) for c in counts]
           + list(nearest) + list(configured) + [matched] + aggregate
           + [friction,
              math.cos(2 * math.pi * hour / 24.0),
              math.sin(2 * math.pi * hour / 24.0)])
    if len(row) != len(FEATURE_NAMES):
        raise RuntimeError("feature vector length %d, expected %d"
                           % (len(row), len(FEATURE_NAMES)))
    return np.nan_to_num(np.asarray(row, dtype=np.float64),
                         nan=0.0, posinf=999.0, neginf=-999.0)

