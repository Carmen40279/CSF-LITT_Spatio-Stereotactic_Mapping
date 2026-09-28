

#Represents a finite stereotactic trajectory from an entry point to a target point.

from __future__ import annotations

from dataclasses import dataclass

from src.geometry.vectors import (
    Point3D,
    Vector3D,
    subtract,
    magnitude,
    normalize,
)
@dataclass(frozen=True)
class Trajectory:
    """A finite entry-to-target trajectory."""

    entry: Point3D
    target: Point3D

    @property
    def vector(self) -> Vector3D:
        """Return the entry-to-target vector."""

        return subtract(
            self.target,
            self.entry,
        )

    @property
    def length(self) -> float:
        """Return trajectory length."""

        return magnitude(self.vector)

    @property
    def direction(self) -> Vector3D:
        """Return normalized trajectory direction."""

        return normalize(self.vector)

    def point_at(self, distance: float) -> Point3D:
        """
        Return a point along the trajectory.

        distance = 0 → entry
        distance = length → target
        """

        if distance < 0 or distance > self.length:
            raise ValueError(
                "Distance must lie within the trajectory."
            )

        direction = self.direction

        return Point3D(
            self.entry.x + direction.x * distance,
            self.entry.y + direction.y * distance,
            self.entry.z + direction.z * distance,
        )