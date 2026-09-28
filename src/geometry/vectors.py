"""
ThermoSEEG 3D geometry primitives.

These functions provide the mathematical foundation for
SEEG electrode planning and LITT trajectory analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Point3D:
    """A point in 3D stereotactic space."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Vector3D:
    """A vector in 3D space."""

    x: float
    y: float
    z: float


def subtract(a: Point3D, b: Point3D) -> Vector3D:
    """Return the vector from point b to point a."""

    return Vector3D(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    )


def add(point: Point3D, vector: Vector3D) -> Point3D:
    """Translate a point by a vector."""

    return Point3D(
        point.x + vector.x,
        point.y + vector.y,
        point.z + vector.z,
    )


def scale(vector: Vector3D, factor: float) -> Vector3D:
    """Scale a vector by a scalar."""

    return Vector3D(
        vector.x * factor,
        vector.y * factor,
        vector.z * factor,
    )


def magnitude(vector: Vector3D) -> float:
    """Return vector magnitude."""

    return math.sqrt(
        vector.x ** 2
        + vector.y ** 2
        + vector.z ** 2
    )


def normalize(vector: Vector3D) -> Vector3D:
    """Return a unit vector."""

    length = magnitude(vector)

    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector.")

    return scale(vector, 1.0 / length)


def dot(a: Vector3D, b: Vector3D) -> float:
    """Return the dot product of two vectors."""

    return (
        a.x * b.x
        + a.y * b.y
        + a.z * b.z
    )


def cross(a: Vector3D, b: Vector3D) -> Vector3D:
    """Return the cross product of two vectors."""

    return Vector3D(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )