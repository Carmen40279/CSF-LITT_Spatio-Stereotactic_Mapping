"""
ThermoSEEG CSF spatial model.

The present implementation uses a synthetic ellipsoidal
compartment to develop and test CSF spatial calculations.

This is NOT a clinical CSF segmentation.

The future implementation will replace this synthetic
representation with CSF derived from medical imaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from src.geometry.vectors import Point3D
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class CSFCompartment:
    """
    Synthetic CSF compartment represented by an ellipsoid.
    """

    center: Point3D
    radii: Point3D
    resolution: int = 48

    def create_mesh(self) -> pv.PolyData:
        """
        Create the 3D synthetic CSF surface.
        """

        sphere = pv.Sphere(
            radius=1.0,
            theta_resolution=self.resolution,
            phi_resolution=self.resolution,
        )

        sphere.scale(
            [
                self.radii.x,
                self.radii.y,
                self.radii.z,
            ],
            inplace=True,
        )

        sphere.translate(
            [
                self.center.x,
                self.center.y,
                self.center.z,
            ],
            inplace=True,
        )

        return sphere

    def contains_point(
        self,
        point: Point3D,
    ) -> bool:
        """
        Determine whether a point lies inside the synthetic
        CSF compartment.
        """

        normalized = (
            (
                (point.x - self.center.x)
                / self.radii.x
            ) ** 2

            +

            (
                (point.y - self.center.y)
                / self.radii.y
            ) ** 2

            +

            (
                (point.z - self.center.z)
                / self.radii.z
            ) ** 2
        )

        return normalized <= 1.0

    def trajectory_analysis(
        self,
        trajectory: Trajectory,
        samples: int = 1000,
    ) -> dict[str, float | bool]:
        """
        Analyse the spatial relationship between a trajectory
        and the CSF compartment.

        The trajectory is sampled between entry and target.

        Returns
        -------
        dict
            Minimum distance to CSF,
            fraction of sampled trajectory points inside CSF,
            and an intersection flag.
        """

        if samples < 2:
            raise ValueError(
                "Trajectory analysis requires at least 2 samples."
            )

        distances = np.linspace(
            0.0,
            trajectory.length,
            samples,
        )

        points = np.array([
            [
                trajectory.point_at(distance).x,
                trajectory.point_at(distance).y,
                trajectory.point_at(distance).z,
            ]
            for distance in distances
        ])

        center = np.array([
            self.center.x,
            self.center.y,
            self.center.z,
        ])

        radii = np.array([
            self.radii.x,
            self.radii.y,
            self.radii.z,
        ])

        normalized = (
            (points - center)
            / radii
        )

        inside = (
            np.sum(
                normalized ** 2,
                axis=1,
            )
            <= 1.0
        )

        inside_fraction = float(
            np.mean(inside)
        )

        # Build a spatial index over the CSF surface.
        mesh = self.create_mesh()

        tree = cKDTree(
            mesh.points
        )

        outside_points = points[~inside]

        if len(outside_points) > 0:

            outside_distances, _ = (
                tree.query(
                    outside_points
                )
            )

            minimum_surface_distance = float(
                np.min(outside_distances)
            )

        else:

            minimum_surface_distance = 0.0

        minimum_distance = (
            0.0
            if np.any(inside)
            else minimum_surface_distance
        )

        return {
            "minimum_distance": minimum_distance,
            "fraction_inside": inside_fraction,
            "intersects": bool(np.any(inside)),
        }