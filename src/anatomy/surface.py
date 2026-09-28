"""
ThermoSEEG anatomical surface generation.

Converts 3D anatomical masks into surface meshes suitable for
geometry processing and later visualization.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes

from src.anatomy.anatomical_model import (
    AnatomicalMask,
    AnatomicalSurface,
)


def mask_to_surface(
    anatomical_mask: AnatomicalMask,
    level: float = 0.5,
) -> AnatomicalSurface:
    """
    Convert a binary anatomical mask into a triangular surface mesh.

    Vertices are returned in physical coordinates using the mask's
    voxel spacing. The affine translation is then applied so that
    the mesh is expressed in the mask's coordinate space.
    """

    mask = np.asarray(
        anatomical_mask.mask,
        dtype=np.uint8,
    )

    if mask.ndim != 3:
        raise ValueError(
            "Anatomical mask must be a 3D array."
        )

    if not np.any(mask):
        raise ValueError(
            f"Anatomical mask {anatomical_mask.name!r} is empty."
        )

    vertices, faces, _, _ = marching_cubes(
        mask,
        level=level,
        spacing=anatomical_mask.spacing,
    )

    affine = np.asarray(
        anatomical_mask.affine,
        dtype=float,
    )

    homogeneous = np.column_stack(
        [
            vertices,
            np.ones(vertices.shape[0]),
        ]
    )

    transformed = (
        affine @ homogeneous.T
    ).T

    vertices_world = transformed[:, :3]

    return AnatomicalSurface(
        name=anatomical_mask.name,
        vertices=vertices_world,
        faces=np.asarray(
            faces,
            dtype=np.int32,
        ),
        coordinate_space=(
            anatomical_mask.coordinate_space
        ),
    )


def create_brain_surface(
    brain_mask: AnatomicalMask,
) -> AnatomicalSurface:
    """
    Generate a surface mesh from a patient brain mask.
    """

    if brain_mask.name != "brain":
        raise ValueError(
            "create_brain_surface() expects a "
            "'brain' anatomical mask."
        )

    return mask_to_surface(
        brain_mask
    )