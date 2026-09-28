#replace later with live thermal map changing colors
from __future__ import annotations

from dataclasses import dataclass

import pyvista as pv

from src.geometry.vectors import Point3D
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class AblationGeometry:
    """
    Geometric representation of a LITT ablation region.
    """

    trajectory: Trajectory
    active_length: float
    radius: float

    @property
    def start_distance(self) -> float:
        """
        Distance from trajectory entry to the beginning
        of the active ablation region.
        """

        return max(
            0.0,
            self.trajectory.length - self.active_length,
        )

    @property
    def end_distance(self) -> float:
        """
        Distance from trajectory entry to the end
        of the active ablation region.
        """

        return self.trajectory.length

    @property
    def start_point(self) -> Point3D:
        """Return the proximal point of the ablation region."""

        return self.trajectory.point_at(
            self.start_distance
        )

    @property
    def end_point(self) -> Point3D:
        """Return the distal point of the ablation region."""

        return self.trajectory.point_at(
            self.end_distance
        )

    @property
    def center_point(self) -> Point3D:
        """Return the center of the active ablation region."""

        midpoint_distance = (
            self.start_distance
            + self.end_distance
        ) / 2.0

        return self.trajectory.point_at(
            midpoint_distance
        )

    @property
    def approximate_volume(self) -> float:
        """
        Approximate the volume of a capsule-like ablation
        region.

        Volume =
            cylinder
            +
            one sphere equivalent to the two hemispherical ends.
        """

        cylinder_volume = (
            3.141592653589793
            * self.radius ** 2
            * self.active_length
        )

        spherical_ends = (
            4.0
            / 3.0
            * 3.141592653589793
            * self.radius ** 3
        )

        return cylinder_volume + spherical_ends


def create_ablation_meshes(
    ablation: AblationGeometry,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """
    Create the visual meshes representing the ablation region.

    Returns:
    tuple
        Cylinder, proximal cap, distal cap.
    """

    start = ablation.start_point
    end = ablation.end_point
    center = ablation.center_point

    center_position = (
        center.x,
        center.y,
        center.z,
    )

    direction = (
        ablation.trajectory.direction.x,
        ablation.trajectory.direction.y,
        ablation.trajectory.direction.z,
    )

    cylinder = pv.Cylinder(
        center=center_position,
        direction=direction,
        radius=ablation.radius,
        height=ablation.active_length,
        resolution=48,
        capping=True,
    )

    start_cap = pv.Sphere(
        radius=ablation.radius,
        center=(
            start.x,
            start.y,
            start.z,
        ),
        theta_resolution=32,
        phi_resolution=32,
    )

    end_cap = pv.Sphere(
        radius=ablation.radius,
        center=(
            end.x,
            end.y,
            end.z,
        ),
        theta_resolution=32,
        phi_resolution=32,
    )

    return (
        cylinder,
        start_cap,
        end_cap,
    )