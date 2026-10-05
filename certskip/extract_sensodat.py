# -*- coding: utf-8 -*-
"""Pre-execution feature extraction from SensoDat.

SensoDat is the second subject system: BeamNG.tech lane-keeping tests produced by
three generators, published with the MSR 2024 paper. Everything above this module is
reused unchanged. Only the parse is written per platform, because the leakage boundary
is defined by each platform's scenario format.

Where the boundary falls here
-----------------------------
SensoDat separates the two kinds of content into two files:

    NNNNN-test.xodr              OpenDRIVE. The road the generator emitted, plus a
                                 header naming the run configuration. Exists before
                                 the test runs.
    sim_data/NNNNN-test_sim_data.json
                                 81 sensor channels recorded while the car drove.
                                 Exists only afterwards.

This module never opens the second file, and `iter_tests` refuses any path under a
`sim_data` directory. The boundary is therefore enforced twice: once on the path, and
once on the elements read out of the file that is allowed.

Two attributes inside the allowed file are traps rather than specification.
`test_duration` is how long the run took. `predicted_test_outcome` is a label from
somebody else's model. Both sit in the same element as the label we are predicting,
which is the easiest place in this dataset to leak from. `LEAKY_KEYS` names all three.

Road descriptors
----------------
The features are road geometry statistics, chosen to overlap with what the published
selectors for this benchmark already read. SDC-Scissor uses road attributes and road
statistics; ITS4SDC uses the angles and lengths of road segments. Meeting those
baselines on descriptors they already use keeps the comparison about the decision rule
rather than about who engineered better features.
"""
from __future__ import annotations

import math
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

__all__ = ["FEATURE_NAMES", "LEAKY_KEYS", "GENERATORS", "GENERATOR_NAMES",
           "LeakageError", "assert_no_leak", "Reader", "test_features",
           "iter_tests"]

LEAKY_KEYS = ("execution_data", "sim_data", "test_duration",
              "predicted_test_outcome", "test_outcome")

GENERATORS = {"ambiegen": 0, "frenetic": 1, "frenetic_v": 2, "freneticv": 2}
GENERATOR_NAMES = {0: "ambiegen", 1: "frenetic", 2: "freneticV"}

FEATURE_NAMES = [
    # launch configuration, read from run_config
    "cfg_rf", "cfg_oob", "cfg_max_speed", "cfg_obstacles", "cfg_bump_dist",
    "cfg_delineator_dist", "cfg_tree_dist", "cfg_field_of_view",
    # size and shape of the road
    "n_points", "road_length", "span_x", "span_y", "diagonal", "bbox_ratio",
    # segment lengths
    "seg_len_min", "seg_len_mean", "seg_len_max", "seg_len_std",
    # turn angles between consecutive segments
    "angle_abs_mean", "angle_abs_max", "angle_std", "angle_sum", "angle_net",
    # curvature, the quantity both published selectors key on
    "curv_mean", "curv_max", "curv_std", "curv_p90",
    "frac_straight", "frac_gentle", "frac_sharp",
    # turn structure
    "n_left", "n_right", "n_direction_changes", "turn_balance",
    "longest_straight", "longest_curve",
    # start and end pose
    "start_x", "start_y", "end_x", "end_y", "start_to_end", "tortuosity",
]


class LeakageError(RuntimeError):
    """Raised when extraction touches something that only exists after execution."""


def assert_no_leak(names) -> None:
    """Fail loudly if any post-execution name appears in `names`."""
    bad = sorted(set(names) & set(LEAKY_KEYS))
    if bad:
        raise LeakageError(
            "extraction touched post-execution content: %s. This exists only after "
            "the test has been run and cannot inform the decision to run it."
            % ", ".join(bad))


class Reader:
    """The only way this module reads a parsed test, and a record of what it read.

    Same contract as `certskip.extract.Reader`: every lookup records the name it asked
    for, and `check()` tests the record against `LEAKY_KEYS`. Routing every read
    through one object is what keeps the record from drifting away from the reads.
    """

    def __init__(self, root):
        self.root = root
        self.touched = set()

    def find(self, path):
        self.touched.add(path.rsplit("/", 1)[-1])
        return self.root.find(path)

    def findall(self, path):
        self.touched.add(path.rsplit("/", 1)[-1])
        return self.root.findall(path)

    def attr(self, element, name, default=None):
        self.touched.add(name)
        if element is None:
            return default
        return element.get(name, default)

    def check(self):
        assert_no_leak(self.touched)


def _num(value, default=0.0):
    if value is None or value in ("None", "null", ""):
        return default
    if value in ("True", "true"):
        return 1.0
    if value in ("False", "false"):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def outcome_of(root):
    """The label. Read here and nowhere else, and never as a feature.

    This deliberately bypasses `Reader`, because `test_outcome` is on the leak list.
    Keeping the label read out of the feature path is the point of the separation.
    """
    info = root.find("./header/sdc_test_info")
    if info is None:
        return None
    value = str(info.get("test_outcome", "")).strip().upper()
    if value in ("FAIL", "FAILED", "ERROR"):
        return 1
    if value in ("PASS", "PASSED", "SUCCESS"):
        return 0
    return None


def road_points(reader):
    """Centre line of the road, as the start pose of each plan-view record.

    OpenDRIVE stores a road as a sequence of geometry records, each with a start
    point, a heading and a length. The start points trace the road. The final point is
    reconstructed from the last record's heading and length.
    """
    geoms = reader.findall("./road/planView/geometry")
    pts = []
    for g in geoms:
        x, y = reader.attr(g, "x"), reader.attr(g, "y")
        if x is None or y is None:
            continue
        try:
            pts.append((float(x), float(y)))
        except ValueError:
            continue
    if len(pts) < 3:
        return None
    last = geoms[-1]
    try:
        hdg = float(reader.attr(last, "hdg"))
        length = float(reader.attr(last, "length"))
        pts.append((pts[-1][0] + length * math.cos(hdg),
                    pts[-1][1] + length * math.sin(hdg)))
    except (TypeError, ValueError):
        pass
    return pts


def test_features(root, check_leak: bool = True):
    """Pre-execution feature vector for one parsed test, or None."""
    reader = Reader(root)
    cfg = reader.find("./header/run_config")
    config = [_num(reader.attr(cfg, key), default) for key, default in (
        ("rf", 1.5), ("oob", 0.5), ("max_speed", 120.0), ("obstacles", 0.0),
        ("bump_dist", 0.0), ("delineator_dist", 0.0), ("tree_dist", 0.0),
        ("field_of_view", 120.0))]

    pts = road_points(reader)
    if check_leak:
        reader.check()
    if pts is None or len(pts) < 4:
        return None

    p = np.asarray(pts, dtype=float)
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > 1e-6
    d, seg = d[keep], seg[keep]
    if len(seg) < 3:
        return None

    length = float(seg.sum())
    span_x = float(p[:, 0].max() - p[:, 0].min())
    span_y = float(p[:, 1].max() - p[:, 1].min())
    diagonal = float(math.hypot(span_x, span_y))

    hdg = np.arctan2(d[:, 1], d[:, 0])
    turn = np.diff(hdg)
    turn = (turn + math.pi) % (2 * math.pi) - math.pi        # wrap into [-pi, pi]
    absturn = np.abs(turn)
    denom = 0.5 * (seg[:-1] + seg[1:])
    curv = absturn / np.maximum(denom, 1e-6)

    straight = float((absturn < math.radians(2)).mean())
    gentle = float(((absturn >= math.radians(2)) & (absturn < math.radians(10))).mean())
    sharp = float((absturn >= math.radians(10)).mean())

    sign = np.sign(turn)
    sign[absturn < math.radians(2)] = 0
    n_left = int((sign > 0).sum())
    n_right = int((sign < 0).sum())
    nonzero = sign[sign != 0]
    changes = int((np.diff(nonzero) != 0).sum()) if len(nonzero) > 1 else 0

    def longest_run(mask):
        best = run = 0
        for m in mask:
            run = run + 1 if m else 0
            best = max(best, run)
        return float(best)

    start_to_end = float(math.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]))
    row = config + [
        float(len(p)), length, span_x, span_y, diagonal,
        span_x / max(span_y, 1e-6),
        float(seg.min()), float(seg.mean()), float(seg.max()), float(seg.std()),
        float(absturn.mean()), float(absturn.max()), float(turn.std()),
        float(absturn.sum()), float(turn.sum()),
        float(curv.mean()), float(curv.max()), float(curv.std()),
        float(np.percentile(curv, 90)),
        straight, gentle, sharp,
        float(n_left), float(n_right), float(changes),
        (n_left - n_right) / max(n_left + n_right, 1),
        longest_run(absturn < math.radians(2)),
        longest_run(absturn >= math.radians(10)),
        float(p[0, 0]), float(p[0, 1]), float(p[-1, 0]), float(p[-1, 1]),
        start_to_end, length / max(start_to_end, 1e-6),
    ]
    if len(row) != len(FEATURE_NAMES):
        raise RuntimeError("feature vector length %d, expected %d"
                           % (len(row), len(FEATURE_NAMES)))
    return np.nan_to_num(np.asarray(row, dtype=np.float64),
                         nan=0.0, posinf=1e6, neginf=-1e6)


def _generator_of(path: str):
    """Which generator produced this test, from the directory it sits in.

    Three spellings of the same generator appear across campaigns, and one campaign
    nests the tests one level deeper. Both are naming, not content.
    """
    parts = [p.lower() for p in path.replace("\\", "/").split("/")[:-1]]
    for part in reversed(parts):
        token = part.split("_set_")[-1]
        for key in ("frenetic_v", "freneticv", "frenetic", "ambiegen"):
            if token.endswith(key) or part == key or part.endswith("_" + key):
                return GENERATORS[key]
        if part in GENERATORS:
            return GENERATORS[part]
    return None


def _campaign_of(path: str):
    m = re.search(r"set_?(\d+)", path)
    return int(m.group(1)) if m else -1


def iter_tests(root: str):
    """Yield (path, xml bytes) for every pre-execution test description.

    Paths under a `sim_data` directory are refused rather than skipped quietly. The
    execution trace is not something this module is allowed to be pointed at.
    """
    if root.endswith(".zip"):
        with zipfile.ZipFile(root) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                name = info.filename
                if info.is_dir() or not name.lower().endswith(".xodr"):
                    continue
                if "sim_data" in name.split("/"):
                    raise LeakageError("refusing to read %s" % name)
                with zf.open(info) as fh:
                    yield name, fh.read()
        return
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "sim_data"]
        for f in sorted(files):
            if f.lower().endswith(".xodr"):
                full = os.path.join(dirpath, f)
                with open(full, "rb") as fh:
                    yield full, fh.read()

