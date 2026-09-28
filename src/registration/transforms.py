"""
ThermoSEEG spatial transformation utilities.

Provides affine transformations between coordinate spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialTransform:
    """
    Represents an affine transformation between two coordinate spaces.
    """

    source_space: str
    target_space: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(
            self.matrix,
            dtype=float,
        )

        if matrix.shape != (4, 4):
            raise ValueError(
                "Transformation matrix must have shape (4, 4)."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Transformation matrix must contain only finite values."
            )

        object.__setattr__(
            self,
            "matrix",
            matrix,
        )

        if not self.source_space:
            raise ValueError(
                "Source coordinate space cannot be empty."
            )

        if not self.target_space:
            raise ValueError(
                "Target coordinate space cannot be empty."
            )

    def transform_point(
        self,
        point: tuple[float, float, float],
    ) -> np.ndarray:
        """Transform one 3D point."""

        homogeneous = np.array(
            [
                point[0],
                point[1],
                point[2],
                1.0,
            ],
            dtype=float,
        )

        transformed = self.matrix @ homogeneous

        if transformed[3] == 0:
            raise ValueError(
                "Invalid homogeneous coordinate after transformation."
            )

        return transformed[:3] / transformed[3]

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Transform an array of 3D points."""

        points = np.asarray(
            points,
            dtype=float,
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N, 3)."
            )

        homogeneous = np.column_stack(
            [
                points,
                np.ones(len(points)),
            ]
        )

        transformed = (
            homogeneous @ self.matrix.T
        )

        if np.any(transformed[:, 3] == 0):
            raise ValueError(
                "Invalid homogeneous coordinate after transformation."
            )

        return (
            transformed[:, :3]
            / transformed[:, 3:4]
        )

    def inverse(self) -> "SpatialTransform":
        """Return the inverse transformation."""

        try:
            inverse_matrix = np.linalg.inv(
                self.matrix
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Transformation matrix is not invertible."
            ) from exc

        return SpatialTransform(
            source_space=self.target_space,
            target_space=self.source_space,
            matrix=inverse_matrix,
        )

    def compose(
        self,
        other: "SpatialTransform",
    ) -> "SpatialTransform":
        """
        Compose this transformation with another transformation.

        The other transformation must start where this one ends.
        """

        if self.target_space != other.source_space:
            raise ValueError(
                "Cannot compose transformations: "
                f"{self.target_space!r} does not match "
                f"{other.source_space!r}."
            )

        matrix = (
            other.matrix @ self.matrix
        )

        return SpatialTransform(
            source_space=self.source_space,
            target_space=other.target_space,
            matrix=matrix,
        )


def identity_transform(
    space: str,
) -> SpatialTransform:
    """Create an identity transformation."""

    return SpatialTransform(
        source_space=space,
        target_space=space,
        matrix=np.eye(4),
    )


def translation_transform(
    source_space: str,
    target_space: str,
    translation: tuple[float, float, float],
) -> SpatialTransform:
    """Create a pure translation transformation."""

    matrix = np.eye(4)

    matrix[:3, 3] = np.asarray(
        translation,
        dtype=float,
    )

    return SpatialTransform(
        source_space=source_space,
        target_space=target_space,
        matrix=matrix,
    )


def rotation_transform(
    source_space: str,
    target_space: str,
    axis: str,
    angle_degrees: float,
) -> SpatialTransform:
    """Create a rotation around one Cartesian axis."""

    angle = np.deg2rad(
        angle_degrees
    )

    cosine = np.cos(angle)
    sine = np.sin(angle)

    axis = axis.lower()

    if axis == "x":
        rotation = np.array(
            [
                [1, 0, 0],
                [0, cosine, -sine],
                [0, sine, cosine],
            ],
            dtype=float,
        )

    elif axis == "y":
        rotation = np.array(
            [
                [cosine, 0, sine],
                [0, 1, 0],
                [-sine, 0, cosine],
            ],
            dtype=float,
        )

    elif axis == "z":
        rotation = np.array(
            [
                [cosine, -sine, 0],
                [sine, cosine, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            "Axis must be one of: 'x', 'y', or 'z'."
        )

    matrix = np.eye(4)
    matrix[:3, :3] = rotation

    return SpatialTransform(
        source_space=source_space,
        target_space=target_space,
        matrix=matrix,
    )


def validate_transform(
    transform: SpatialTransform,
) -> None:
    """Validate a spatial transformation."""

    if not isinstance(
        transform,
        SpatialTransform,
    ):
        raise TypeError(
            "Expected a SpatialTransform instance."
        )

    if transform.matrix.shape != (4, 4):
        raise ValueError(
            "Transformation matrix must have shape (4, 4)."
        )

    if not np.all(
        np.isfinite(transform.matrix)
    ):
        raise ValueError(
            "Transformation matrix must contain only finite values."
        )