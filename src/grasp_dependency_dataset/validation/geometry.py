"""Small geometry helpers for lightweight staged validation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def as_array(values: Iterable[float]) -> np.ndarray:
    """Convert an iterable to a float64 numpy array."""

    return np.asarray(list(values), dtype=float)


def normalize(values: Iterable[float]) -> np.ndarray:
    """Normalize a vector and fall back to a top-down direction if degenerate."""

    array = as_array(values)
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=float)
    return array / norm


def distance_point_to_segment(point: Iterable[float], start: Iterable[float], end: Iterable[float]) -> float:
    """Return the Euclidean distance between a point and a segment."""

    point_arr = as_array(point)
    start_arr = as_array(start)
    end_arr = as_array(end)
    segment = end_arr - start_arr
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point_arr - start_arr))
    t = float(np.clip(np.dot(point_arr - start_arr, segment) / denom, 0.0, 1.0))
    closest = start_arr + t * segment
    return float(np.linalg.norm(point_arr - closest))


def segment_intersects_sphere(
    start: Iterable[float],
    end: Iterable[float],
    center: Iterable[float],
    radius: float,
) -> bool:
    """Check whether a segment intersects or touches a sphere."""

    return distance_point_to_segment(center, start, end) <= radius


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Euclidean distance between two points."""

    return float(np.linalg.norm(as_array(a) - as_array(b)))


def within_sphere(point: Iterable[float], center: Iterable[float], radius: float) -> bool:
    """Check whether a point lies inside a sphere."""

    return distance(point, center) <= radius


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar to a closed interval."""

    return max(low, min(high, value))
