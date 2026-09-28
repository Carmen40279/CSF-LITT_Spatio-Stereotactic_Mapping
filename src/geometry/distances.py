"""
ThermoSEEG spatial distance calculations.

These functions provide the geometric foundation for
trajectory-to-structure analysis.
"""

from __future__ import annotations

from src.geometry.vectors import (
    Point3D,
    Vector3D,
    subtract,
    dot,
    scale,
    add,
)


def point_to_point_distance(
    a: Point3D,
    b: Point3D,
) -> float:
    """Calculate Euclidean distance between two points."""

    vector = subtract(a, b)

    return (
        vector.x ** 2
        + vector.y ** 2
        + vector.z ** 2
    ) ** 0.5


def closest_point_on_line(
    point: Point3D,
    line_origin: Point3D,
    line_direction: Vector3D,
) -> Point3D:
    """
    Find the closest point on an infinite line to a point.
    """

    origin_to_point = subtract(
        point,
        line_origin,
    )

    denominator = dot(
        line_direction,
        line_direction,
    )

    if denominator == 0:
        raise ValueError(
            "Line direction cannot be zero."
        )

    parameter = (
        dot(
            origin_to_point,
            line_direction,
        )
        / denominator
    )

    displacement = scale(
        line_direction,
        parameter,
    )

    return add(
        line_origin,
        displacement,
    )


def point_to_line_distance(
    point: Point3D,
    line_origin: Point3D,
    line_direction: Vector3D,
) -> float:
    """Calculate distance from a point to an infinite line."""

    closest = closest_point_on_line(
        point,
        line_origin,
        line_direction,
    )

    return point_to_point_distance(
        point,
        closest,
    )
def closest_point_on_segment(
    point: Point3D,
    segment_start: Point3D,
    segment_end: Point3D,
) -> Point3D:
    """
    Find the closest point on a finite line segment.
    """

    segment = subtract(
        segment_end,
        segment_start,
    )

    point_vector = subtract(
        point,
        segment_start,
    )

    denominator = dot(
        segment,
        segment,
    )

    if denominator == 0:
        return segment_start

    parameter = (
        dot(
            point_vector,
            segment,
        )
        / denominator
    )

    parameter = max(
        0.0,
        min(1.0, parameter),
    )

    displacement = scale(
        segment,
        parameter,
    )

    return add(
        segment_start,
        displacement,
    )


def point_to_segment_distance(
    point: Point3D,
    segment_start: Point3D,
    segment_end: Point3D,
) -> float:
    """Calculate distance from a point to a finite segment."""

    closest = closest_point_on_segment(
        point,
        segment_start,
        segment_end,
    )

    return point_to_point_distance(
        point,
        closest,
    )