"""
ThermoSEEG spatial interaction analysis.

Provides geometric measurements between a trajectory-aligned
LITT ablation volume and a CSF compartment.

This module is computational geometry only. It does not assign
clinical risk. Risk interpretation belongs in the risk layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.anatomy.csf import CSFCompartment
from src.trajectory.ablation import AblationGeometry


# ==============================================================
# RESULT MODEL
# ==============================================================


@dataclass(frozen=True)
class AblationCSFAnalysis:
    """
    Quantitative spatial relationship between a LITT ablation
    volume and a CSF compartment.
    """

    minimum_distance: float
    maximum_distance: float

    intersects: bool

    sampled_points: int
    points_inside_csf: int
    fraction_inside_csf: float

    estimated_ablation_volume: float
    estimated_volume_inside_csf: float
    estimated_fraction_inside_csf: float

    closest_ablation_point: np.ndarray
    closest_csf_point: np.ndarray

    intersection_points: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        """Return the analysis as a serializable dictionary."""

        return {
            "minimum_distance": self.minimum_distance,
            "maximum_distance": self.maximum_distance,
            "intersects": self.intersects,
            "sampled_points": self.sampled_points,
            "points_inside_csf": self.points_inside_csf,
            "fraction_inside_csf": self.fraction_inside_csf,
            "estimated_ablation_volume": self.estimated_ablation_volume,
            "estimated_volume_inside_csf": self.estimated_volume_inside_csf,
            "estimated_fraction_inside_csf": self.estimated_fraction_inside_csf,
            "closest_ablation_point": self.closest_ablation_point.tolist(),
            "closest_csf_point": self.closest_csf_point.tolist(),
            "intersection_points": self.intersection_points.tolist(),
        }


# ==============================================================
# INTERNAL COORDINATE HELPERS
# ==============================================================


def _component_to_float(value: Any, name: str) -> float:
    """Convert one coordinate component to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must contain numeric values."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must contain finite values."
        )

    return result


def _point_to_array(point: Any, name: str) -> np.ndarray:
    """
    Convert a point-like object into a 3-element NumPy array.

    Supports:
        - objects with x/y/z attributes
        - tuples/lists/NumPy arrays of length 3
    """

    if all(hasattr(point, component) for component in ("x", "y", "z")):
        values = [point.x, point.y, point.z]
    else:
        try:
            values = list(point)
        except TypeError as exc:
            raise TypeError(
                f"{name} must provide x/y/z attributes "
                "or contain three numeric values."
            ) from exc

        if len(values) != 3:
            raise ValueError(
                f"{name} must contain exactly three values."
            )

    return np.asarray(
        [
            _component_to_float(values[0], f"{name}[0]"),
            _component_to_float(values[1], f"{name}[1]"),
            _component_to_float(values[2], f"{name}[2]"),
        ],
        dtype=float,
    )


# ==============================================================
# LITT ABLATION SAMPLING
# ==============================================================


def _trajectory_frame(
    ablation: AblationGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct an orthonormal frame around the LITT trajectory.

    Returns:
        axis
        perpendicular_1
        perpendicular_2
    """

    entry = _point_to_array(
        ablation.trajectory.entry,
        "trajectory.entry",
    )

    target = _point_to_array(
        ablation.trajectory.target,
        "trajectory.target",
    )

    direction = target - entry
    length = np.linalg.norm(direction)

    if length <= 1e-8:
        raise ValueError("LITT trajectory has zero length.")

    axis = direction / length

    reference = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )

    perpendicular_1 = np.cross(
        axis,
        reference,
    )

    perpendicular_1 /= np.linalg.norm(
        perpendicular_1
    )

    perpendicular_2 = np.cross(
        axis,
        perpendicular_1,
    )

    perpendicular_2 /= np.linalg.norm(
        perpendicular_2
    )

    return (
        axis,
        perpendicular_1,
        perpendicular_2,
    )


def sample_ablation_volume(
    ablation: AblationGeometry,
    axial_samples: int = 60,
    radial_samples: int = 12,
    angular_samples: int = 24,
) -> np.ndarray:
    """
    Generate deterministic sample points throughout the
    trajectory-aligned cylindrical ablation volume.

    The sampling includes the cylinder interior rather than
    only its surface.
    """

    if axial_samples < 2:
        raise ValueError("axial_samples must be >= 2.")

    if radial_samples < 2:
        raise ValueError("radial_samples must be >= 2.")

    if angular_samples < 4:
        raise ValueError("angular_samples must be >= 4.")

    entry = _point_to_array(
        ablation.trajectory.entry,
        "trajectory.entry",
    )

    axis, perpendicular_1, perpendicular_2 = _trajectory_frame(
        ablation
    )

    trajectory_length = float(
        ablation.trajectory.length
    )

    active_length = float(
        ablation.active_length
    )

    radius = float(
        ablation.radius
    )

    if active_length <= 0:
        raise ValueError(
            "Ablation active length must be positive."
        )

    if radius <= 0:
        raise ValueError(
            "Ablation radius must be positive."
        )

    if active_length > trajectory_length + 1e-8:
        raise ValueError(
            "Ablation active length cannot exceed "
            "the trajectory length."
        )

    start = (
        trajectory_length - active_length
    ) / 2.0

    axial_positions = np.linspace(
        start,
        start + active_length,
        axial_samples,
    )

    radial_positions = np.linspace(
        0.0,
        radius,
        radial_samples,
    )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        angular_samples,
        endpoint=False,
    )

    points = []

    for axial in axial_positions:
        center = (
            entry
            + axis * axial
        )

        points.append(center)

        for radial in radial_positions[1:]:
            for angle in angles:
                offset = (
                    perpendicular_1
                    * (
                        radial
                        * np.cos(angle)
                    )
                    +
                    perpendicular_2
                    * (
                        radial
                        * np.sin(angle)
                    )
                )

                points.append(
                    center + offset
                )

    return np.asarray(
        points,
        dtype=float,
    )


# ==============================================================
# CSF POINT TESTING
# ==============================================================


def _points_inside_csf(
    points: np.ndarray,
    csf: CSFCompartment,
) -> np.ndarray:
    """
    Determine which points lie inside the CSF ellipsoid.

    The current CSF model is represented as an ellipsoid.

    CSF center and radii may be represented either as objects
    with x/y/z attributes or as 3-element tuples/lists.
    """

    center = _point_to_array(
        csf.center,
        "csf.center",
    )

    radii = _point_to_array(
        csf.radii,
        "csf.radii",
    )

    if np.any(radii <= 0):
        raise ValueError(
            "CSF radii must all be positive."
        )

    normalized = (
        points - center
    ) / radii

    ellipsoid_value = np.sum(
        normalized ** 2,
        axis=1,
    )

    return ellipsoid_value <= 1.0


# ==============================================================
# DISTANCE TO CSF SURFACE
# ==============================================================


def _ellipsoid_surface_distance(
    points: np.ndarray,
    csf: CSFCompartment,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate point-to-CSF-surface distance.

    The closest surface point is approximated by projecting
    the normalized radial direction onto the ellipsoid surface.

    This is intended for the current computational prototype.
    """

    center = _point_to_array(
        csf.center,
        "csf.center",
    )

    radii = _point_to_array(
        csf.radii,
        "csf.radii",
    )

    if np.any(radii <= 0):
        raise ValueError(
            "CSF radii must all be positive."
        )

    vectors = points - center

    distances_from_center = np.linalg.norm(
        vectors,
        axis=1,
    )

    safe_vectors = vectors.copy()

    zero_mask = (
        distances_from_center <= 1e-8
    )

    safe_vectors[zero_mask] = np.array(
        [1.0, 0.0, 0.0]
    )

    normalized_direction = (
        safe_vectors
        / np.linalg.norm(
            safe_vectors,
            axis=1,
            keepdims=True,
        )
    )

    denominator = np.sqrt(
        np.sum(
            (
                normalized_direction
                / radii
            ) ** 2,
            axis=1,
        )
    )

    surface_radius = (
        1.0 / denominator
    )

    surface_points = (
        center
        + normalized_direction
        * surface_radius[:, None]
    )

    surface_distances = np.linalg.norm(
        points - surface_points,
        axis=1,
    )

    return (
        surface_distances,
        surface_points,
    )


# ==============================================================
# VOLUME ESTIMATION
# ==============================================================


def estimate_cylinder_volume(
    ablation: AblationGeometry,
) -> float:
    """Calculate the analytical cylindrical ablation volume."""

    radius = float(
        ablation.radius
    )

    length = float(
        ablation.active_length
    )

    return (
        np.pi
        * radius ** 2
        * length
    )


def estimate_inside_volume(
    total_volume: float,
    inside_fraction: float,
) -> float:
    """Estimate the portion of ablation volume inside CSF."""

    return (
        total_volume
        * np.clip(
            inside_fraction,
            0.0,
            1.0,
        )
    )


# ==============================================================
# MAIN ANALYSIS
# ==============================================================


def analyze_ablation_csf(
    ablation: AblationGeometry,
    csf: CSFCompartment,
    *,
    axial_samples: int = 60,
    radial_samples: int = 12,
    angular_samples: int = 24,
) -> AblationCSFAnalysis:
    """
    Analyze the spatial relationship between a LITT ablation
    volume and a CSF compartment.

    Returns quantitative geometry only.

    No clinical risk classification is performed here.
    """

    points = sample_ablation_volume(
        ablation,
        axial_samples=axial_samples,
        radial_samples=radial_samples,
        angular_samples=angular_samples,
    )

    inside_mask = _points_inside_csf(
        points,
        csf,
    )

    inside_count = int(
        np.count_nonzero(
            inside_mask
        )
    )

    total_count = len(points)

    fraction_inside = (
        inside_count / total_count
        if total_count > 0
        else 0.0
    )

    surface_distances, surface_points = (
        _ellipsoid_surface_distance(
            points,
            csf,
        )
    )

    closest_index = int(
        np.argmin(
            surface_distances
        )
    )

    minimum_distance = float(
        surface_distances[
            closest_index
        ]
    )

    maximum_distance = float(
        np.max(
            surface_distances
        )
    )

    closest_ablation_point = (
        points[closest_index]
    )

    closest_csf_point = (
        surface_points[closest_index]
    )

    intersects = bool(
        np.any(inside_mask)
    )

    intersection_points = (
        points[inside_mask]
        if intersects
        else np.empty(
            (0, 3),
            dtype=float,
        )
    )

    total_volume = (
        estimate_cylinder_volume(
            ablation
        )
    )

    inside_volume = (
        estimate_inside_volume(
            total_volume,
            fraction_inside,
        )
    )

    return AblationCSFAnalysis(
        minimum_distance=minimum_distance,
        maximum_distance=maximum_distance,
        intersects=intersects,
        sampled_points=total_count,
        points_inside_csf=inside_count,
        fraction_inside_csf=fraction_inside,
        estimated_ablation_volume=total_volume,
        estimated_volume_inside_csf=inside_volume,
        estimated_fraction_inside_csf=fraction_inside,
        closest_ablation_point=(
            closest_ablation_point.copy()
        ),
        closest_csf_point=(
            closest_csf_point.copy()
        ),
        intersection_points=(
            intersection_points.copy()
        ),
    )