# -*- coding: utf-8 -*-
"""Adapter for the official ICST 2025 ITS4SDC ONNX selector.

This module deliberately reproduces the public tool's preprocessing instead of
approximating it with aggregate curvature statistics.  ITS4SDC resamples every road
to 197 points, converts the 196 segments to ``(angle difference, length)`` pairs,
and feeds the resulting sequence to its frozen ONNX model.  The public model emits
``P(PASS)``; CertSkip uses ``1 - P(PASS)`` as a failure-risk score.

The model itself is not vendored here.  It remains available from the official GPL
competition repository.  The expected SHA-256 pins the artifact used in the paper.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

ITS4SDC_MODEL_SHA256 = (
    "bc741d69b88f6a5ce3c283fe73c738805c06b0f25aea3074c758a70044736ef4")
ITS4SDC_SOURCE = (
    "https://github.com/christianbirchler-org/sdc-testing-competition/"
    "tree/main/tools/selectors/icst2025_selectors/ITS4SDC")

__all__ = ["ITS4SDCScorer", "adjust_array_size", "its4sdc_features",
           "road_points_from_xodr", "ITS4SDC_MODEL_SHA256", "ITS4SDC_SOURCE"]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjust_array_size(points, target_size: int = 197) -> np.ndarray:
    """Reproduce the official selector's linear road resampling."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("road points must have shape (n>=2, 2)")
    if len(points) == target_size:
        return points
    if len(points) > target_size:
        indices = np.linspace(0, len(points) - 1, target_size, dtype=int)
        return points[indices]

    current = np.linspace(0, len(points) - 1, len(points))
    target = np.linspace(0, len(points) - 1, target_size)
    x = interp1d(current, points[:, 0], kind="linear")(target)
    y = interp1d(current, points[:, 1], kind="linear")(target)
    # The released selector keeps interp1d's float64 values until feature assembly.
    return np.column_stack((x, y))


def its4sdc_features(points) -> np.ndarray:
    """Return the exact ``(196, 2)`` sequence consumed by ITS4SDC."""
    road = adjust_array_size(points)
    delta = np.diff(road, axis=0)
    raw_angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    segment_angles = np.zeros_like(raw_angles)
    segment_angles[1:] = np.diff(raw_angles)
    segment_lengths = np.linalg.norm(delta, axis=1)
    return np.column_stack((segment_angles, segment_lengths)).astype(np.float32)


def road_points_from_xodr(raw: bytes | str | ET.Element) -> np.ndarray:
    """Read the plan-view geometry starts exposed by the competition evaluator."""
    root = raw if isinstance(raw, ET.Element) else ET.fromstring(raw)
    points = []
    for geometry in root.findall("./road/planView/geometry"):
        try:
            points.append((float(geometry.get("x")), float(geometry.get("y"))))
        except (TypeError, ValueError):
            continue
    if len(points) < 2:
        raise ValueError("OpenDRIVE test has fewer than two usable road points")
    return np.asarray(points, dtype=np.float32)


class ITS4SDCScorer:
    """Frozen official ITS4SDC model exposed as a failure-risk scorer."""

    def __init__(self, model_path: str | Path, verify_hash: bool = True):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        actual = _sha256(model_path)
        if verify_hash and actual != ITS4SDC_MODEL_SHA256:
            raise ValueError(
                "unexpected ITS4SDC model SHA-256: %s (expected %s)" %
                (actual, ITS4SDC_MODEL_SHA256))
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ITS4SDC scoring requires onnxruntime; install requirements.txt") from exc
        self.model_path = str(model_path)
        self.model_sha256 = actual
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        if list(input_meta.shape[1:]) != [196, 2]:
            raise ValueError("unexpected ITS4SDC input shape %r" %
                             (input_meta.shape,))
        self.input_name = input_meta.name

    def score_features(self, sequences, batch_size: int = 512) -> np.ndarray:
        """Return failure probabilities for already extracted road sequences."""
        sequences = np.asarray(sequences, dtype=np.float32)
        if sequences.ndim != 3 or sequences.shape[1:] != (196, 2):
            raise ValueError("sequences must have shape (n, 196, 2)")
        out = []
        for start in range(0, len(sequences), int(batch_size)):
            pass_probability = self.session.run(
                None, {self.input_name: sequences[start:start + batch_size]})[0]
            out.append(1.0 - np.asarray(pass_probability, dtype=float).reshape(-1))
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    def score_roads(self, roads, batch_size: int = 512) -> np.ndarray:
        """Extract and score an iterable of road-point arrays."""
        features = np.stack([its4sdc_features(road) for road in roads])
        return self.score_features(features, batch_size=batch_size)
