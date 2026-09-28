"""
ThermoSEEG lesion and target representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.vectors import Point3D

from src.anatomy.anatomical_model import AnatomicalMask


@dataclass(frozen=True)
class AnatomicalTarget:
    """
    A point target in an anatomical coordinate space.
    """

    name: str
    point: Point3D
    coordinate_space: str = "patient"
    source: str = "manual"


@dataclass(frozen=True)
class Lesion:
    """
    A lesion represented by an optional segmentation and target.
    """

    name: str
    mask: AnatomicalMask | None = None
    target: AnatomicalTarget | None = None
    description: str = ""


def target_from_voxel(
    mask: AnatomicalMask,
    voxel: tuple[float, float, float],
    name: str = "target",
) -> AnatomicalTarget:
    """
    Convert a voxel coordinate into the mask's anatomical space.
    """

    voxel_array = np.asarray(
        voxel,
        dtype=float,
    )

    if voxel_array.shape != (3,):
        raise ValueError(
            "Voxel coordinate must contain exactly three values."
        )

    homogeneous = np.append(
        voxel_array,
        1.0,
    )

    point = (
        mask.affine @ homogeneous
    )[:3]

    return AnatomicalTarget(
        name=name,
        point=Point3D(
            float(point[0]),
            float(point[1]),
            float(point[2]),
        ),
        coordinate_space=(
            mask.coordinate_space
        ),
        source="voxel-derived",
    )


def lesion_from_mask(
    mask: AnatomicalMask,
    target: AnatomicalTarget | None = None,
    description: str = "",
) -> Lesion:
    """
    Create a lesion representation from an anatomical mask.
    """

    return Lesion(
        name=mask.name,
        mask=mask,
        target=target,
        description=description,
    )