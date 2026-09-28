"""
ThermoSEEG trajectory perturbation utilities.

Generates controlled geometric perturbations of a trajectory for
uncertainty analysis.

This module does NOT calculate clinical risk. It only creates
alternative trajectories by applying small translations and
rotations to an existing entry-to-target trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryPerturbation:
    """
    One perturbed version of a trajectory.

    Parameters
    ----------
    trajectory:
        The resulting perturbed trajectory object.

    label:
        Human-readable description of the perturbation.

    translation:
        Translation applied to the original entry point, in the
        same coordinate system as the trajectory.

    rotation_degrees:
        Rotation magnitude applied to the original trajectory direction.
    """

    trajectory: Any
    label: str
    translation: tuple[float, float, float]
    rotation_degrees: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {
            "label": self.label,
            "translation": list(self.translation),
            "rotation_degrees": self.rotation_degrees,
        }


def _point_to_array(point: Any) -> np.ndarray:
    """
    Convert a point-like object into a NumPy vector.

    Supports:
    - NumPy arrays
    - lists / tuples
    - objects with x/y/z attributes
    """

    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        values = [point.x, point.y, point.z]
    else:
        values = point

    array = np.asarray(values, dtype=float)

    if array.shape != (3,):
        raise ValueError(
            f"Expected a 3D point with shape (3,), got shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("Point coordinates must all be finite.")

    return array


def _point_like(original: Any, coordinates: np.ndarray) -> Any:
    """
    Rebuild a point using the same general representation as the
    original point whenever possible.

    Falls back to a NumPy array if the original type cannot be
    reconstructed.
    """

    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.shape != (3,):
        raise ValueError("coordinates must contain exactly three values.")

    # NumPy array
    if isinstance(original, np.ndarray):
        return coordinates.copy()

    # tuple
    if isinstance(original, tuple):
        return tuple(float(value) for value in coordinates)

    # list
    if isinstance(original, list):
        return [float(value) for value in coordinates]

    # Try constructing the original class with x, y, z.
    original_type = type(original)

    try:
        return original_type(
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2]),
        )
    except Exception:
        pass

    # Try constructing from an iterable.
    try:
        return original_type(coordinates.tolist())
    except Exception:
        pass

    # Safe fallback.
    return coordinates.copy()


def _extract_entry_target(trajectory: Any) -> tuple[Any, Any]:
    """
    Extract entry and target points from a trajectory object.

    The ThermoSEEG Trajectory class is expected to expose
    `.entry` and `.target`.
    """

    if not hasattr(trajectory, "entry"):
        raise AttributeError(
            "Trajectory object must have an 'entry' attribute."
        )

    if not hasattr(trajectory, "target"):
        raise AttributeError(
            "Trajectory object must have a 'target' attribute."
        )

    entry = trajectory.entry
    target = trajectory.target

    return entry, target


def _build_trajectory(
    original_trajectory: Any,
    entry: Any,
    target: Any,
) -> Any:
    """
    Construct a new trajectory using the same class as the original.

    The function first tries the normal ``Trajectory(entry, target)``
    constructor and then a few common keyword variants.
    """

    trajectory_type = type(original_trajectory)

    # Most likely ThermoSEEG constructor.
    try:
        return trajectory_type(entry, target)
    except Exception:
        pass

    # Keyword constructor.
    try:
        return trajectory_type(
            entry=entry,
            target=target,
        )
    except Exception:
        pass

    # If the object has a copy-like constructor, try copying it.
    try:
        new_trajectory = trajectory_type.__new__(trajectory_type)

        if hasattr(original_trajectory, "__dict__"):
            new_trajectory.__dict__.update(
                original_trajectory.__dict__
            )

        new_trajectory.entry = entry
        new_trajectory.target = target

        return new_trajectory
    except Exception as exc:
        raise TypeError(
            "Could not construct a perturbed trajectory using the "
            "original trajectory type. Expected a trajectory class "
            "that can be constructed from entry and target points."
        ) from exc


def _rotation_matrix(
    axis: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    """
    Create a 3D rotation matrix using Rodrigues' rotation formula.
    """

    axis = np.asarray(axis, dtype=float)

    norm = np.linalg.norm(axis)

    if norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")

    axis = axis / norm

    angle = np.deg2rad(float(angle_degrees))

    x, y, z = axis

    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )

    identity = np.eye(3)

    return (
        identity * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )


def perturb_trajectory(
    trajectory: Any,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_axis: tuple[float, float, float] | None = None,
    rotation_degrees: float = 0.0,
    label: str = "perturbed",
) -> TrajectoryPerturbation:
    """
    Create one controlled perturbation of a trajectory.

    Translation moves both entry and target by the same amount.

    Rotation rotates the trajectory direction around the original
    entry point while preserving the original trajectory length.

    Parameters
    ----------
    trajectory:
        Original trajectory.

    translation:
        XYZ translation applied to both endpoints.

    rotation_axis:
        Axis around which the trajectory direction is rotated.
        If None, no rotation is applied.

    rotation_degrees:
        Rotation angle in degrees.

    label:
        Human-readable perturbation label.
    """

    entry_original, target_original = _extract_entry_target(
        trajectory
    )

    entry = _point_to_array(entry_original)
    target = _point_to_array(target_original)

    translation_array = np.asarray(
        translation,
        dtype=float,
    )

    if translation_array.shape != (3,):
        raise ValueError(
            "translation must contain exactly three values."
        )

    if not np.all(np.isfinite(translation_array)):
        raise ValueError(
            "translation values must all be finite."
        )

    direction = target - entry
    length = float(np.linalg.norm(direction))

    if length <= 0:
        raise ValueError(
            "Cannot perturb a zero-length trajectory."
        )

    unit_direction = direction / length

    # Apply rotation around the original entry point.
    if rotation_axis is not None:
        rotation_axis_array = np.asarray(
            rotation_axis,
            dtype=float,
        )

        if rotation_axis_array.shape != (3,):
            raise ValueError(
                "rotation_axis must contain exactly three values."
            )

        if not np.all(np.isfinite(rotation_axis_array)):
            raise ValueError(
                "rotation_axis values must all be finite."
            )

        rotation = _rotation_matrix(
            rotation_axis_array,
            rotation_degrees,
        )

        rotated_direction = rotation @ unit_direction
    else:
        rotated_direction = unit_direction

    # Move the entry point.
    new_entry = entry + translation_array

    # Preserve the original trajectory length.
    new_target = (
        new_entry
        + rotated_direction * length
    )

    new_entry_point = _point_like(
        entry_original,
        new_entry,
    )

    new_target_point = _point_like(
        target_original,
        new_target,
    )

    new_trajectory = _build_trajectory(
        trajectory,
        new_entry_point,
        new_target_point,
    )

    return TrajectoryPerturbation(
        trajectory=new_trajectory,
        label=label,
        translation=(
            float(translation_array[0]),
            float(translation_array[1]),
            float(translation_array[2]),
        ),
        rotation_degrees=float(rotation_degrees),
    )


def generate_local_perturbations(
    trajectory: Any,
    translation_step: float = 1.0,
    rotation_step_degrees: float = 2.0,
) -> list[TrajectoryPerturbation]:
    """
    Generate a compact local uncertainty set.

    Returns 13 trajectories:

    1. Original
    2. +X translation
    3. -X translation
    4. +Y translation
    5. -Y translation
    6. +Z translation
    7. -Z translation
    8. +X rotation
    9. -X rotation
    10. +Y rotation
    11. -Y rotation
    12. +Z rotation
    13. -Z rotation
    """

    if translation_step <= 0:
        raise ValueError(
            "translation_step must be greater than zero."
        )

    if rotation_step_degrees <= 0:
        raise ValueError(
            "rotation_step_degrees must be greater than zero."
        )

    perturbations: list[TrajectoryPerturbation] = []

    # ------------------------------------------------------------------
    # 1. Original trajectory
    # ------------------------------------------------------------------

    perturbations.append(
        perturb_trajectory(
            trajectory=trajectory,
            label="original",
        )
    )

    # ------------------------------------------------------------------
    # 2-7. Translational perturbations
    # ------------------------------------------------------------------

    translations = [
        (
            "+X translation",
            (translation_step, 0.0, 0.0),
        ),
        (
            "-X translation",
            (-translation_step, 0.0, 0.0),
        ),
        (
            "+Y translation",
            (0.0, translation_step, 0.0),
        ),
        (
            "-Y translation",
            (0.0, -translation_step, 0.0),
        ),
        (
            "+Z translation",
            (0.0, 0.0, translation_step),
        ),
        (
            "-Z translation",
            (0.0, 0.0, -translation_step),
        ),
    ]

    for label, translation in translations:
        perturbations.append(
            perturb_trajectory(
                trajectory=trajectory,
                translation=translation,
                label=label,
            )
        )

    # ------------------------------------------------------------------
    # 8-13. Angular perturbations
    # ------------------------------------------------------------------

    rotations = [
        (
            "+X rotation",
            (1.0, 0.0, 0.0),
            rotation_step_degrees,
        ),
        (
            "-X rotation",
            (1.0, 0.0, 0.0),
            -rotation_step_degrees,
        ),
        (
            "+Y rotation",
            (0.0, 1.0, 0.0),
            rotation_step_degrees,
        ),
        (
            "-Y rotation",
            (0.0, 1.0, 0.0),
            -rotation_step_degrees,
        ),
        (
            "+Z rotation",
            (0.0, 0.0, 1.0),
            rotation_step_degrees,
        ),
        (
            "-Z rotation",
            (0.0, 0.0, 1.0),
            -rotation_step_degrees,
        ),
    ]

    for label, axis, angle in rotations:
        perturbations.append(
            perturb_trajectory(
                trajectory=trajectory,
                rotation_axis=axis,
                rotation_degrees=angle,
                label=label,
            )
        )

    return perturbations


def generate_trajectory_perturbations(
    trajectory: Any,
    translation_step: float = 1.0,
    rotation_step_degrees: float = 2.0,
) -> list[TrajectoryPerturbation]:
    """
    Public convenience wrapper.

    This name is kept separate from ``generate_local_perturbations``
    so other ThermoSEEG modules can use the more descriptive function
    name.
    """

    return generate_local_perturbations(
        trajectory=trajectory,
        translation_step=translation_step,
        rotation_step_degrees=rotation_step_degrees,
    )