"""
Map an MRI-derived CSF mask onto the coordinate system used by the
thermal solver. Coordinate flow:
MRI voxel space > patient/world space > thermal-grid space """

from __future__ import annotations

import numpy as np

from src.anatomy.anatomical_model import AnatomicalMask


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Return world-coordinate points as a finite (N, 3) float array."""

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values.")

    return points


def _validate_csf_mask(csf_mask: AnatomicalMask) -> np.ndarray:
    """Check that the CSF mask contains usable image data and an affine."""

    if not isinstance(csf_mask, AnatomicalMask):
        raise TypeError("csf_mask must be an AnatomicalMask.")

    mask = np.asarray(csf_mask.mask)

    if mask.ndim != 3:
        raise ValueError("CSF mask must be a three-dimensional array.")

    if mask.size == 0:
        raise ValueError("CSF mask cannot be empty.")

    if not np.any(mask > 0):
        raise ValueError("CSF mask contains no positive voxels.")

    affine = np.asarray(csf_mask.affine, dtype=float)

    if affine.shape != (4, 4):
        raise ValueError("CSF mask affine must have shape (4, 4).")

    if not np.all(np.isfinite(affine)):
        raise ValueError("CSF mask affine must contain only finite values.")

    return mask


def world_to_voxel(
    points: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Convert patient/world coordinates into continuous MRI voxel coordinates."""

    points = _validate_points(points)
    affine = np.asarray(affine, dtype=float)

    if affine.shape != (4, 4):
        raise ValueError("affine must have shape (4, 4).")

    try:
        inverse_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "CSF affine matrix is not invertible."
        ) from exc

    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=float))
    )

    voxel = homogeneous @ inverse_affine.T

    return voxel[:, :3]


def map_csf_mask_to_points(
    csf_mask: AnatomicalMask,
    points: np.ndarray,
) -> np.ndarray:
    """
    Check which world-coordinate points fall inside the MRI-derived CSF mask.

    Points outside the MRI volume are treated as non-CSF.
    """

    mask = _validate_csf_mask(csf_mask)
    points = _validate_points(points)

    voxel_coordinates = world_to_voxel(
        points,
        csf_mask.affine,
    )

    # The thermal grid is continuous and the MRI mask is voxel-based
    # Use the nearest voxel for the occupancy lookup
    voxel_indices = np.rint(voxel_coordinates).astype(int)

    shape = np.asarray(mask.shape, dtype=int)

    inside = np.all(
        (voxel_indices >= 0)
        & (voxel_indices < shape),
        axis=1,
    )

    occupancy = np.zeros(len(points), dtype=bool)

    valid_indices = voxel_indices[inside]

    if len(valid_indices):
        occupancy[inside] = (
            mask[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ]
            > 0
        )

    return occupancy


def map_csf_mask_to_grid(
    csf_mask: AnatomicalMask,
    points: np.ndarray,
    grid_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Map CSF occupancy onto the thermal solver's grid.

    If ``grid_shape`` is supplied, the flat occupancy array is reshaped
    to match the thermal grid.
    """

    points = _validate_points(points)

    occupancy = map_csf_mask_to_points(
        csf_mask,
        points,
    )

    if grid_shape is None:
        return occupancy

    if len(grid_shape) != 3:
        raise ValueError(
            "grid_shape must contain exactly three dimensions."
        )

    expected_points = int(np.prod(grid_shape))

    if expected_points != len(occupancy):
        raise ValueError(
            "grid_shape does not match the number of thermal-grid points."
        )

    return occupancy.reshape(grid_shape)


def calculate_csf_grid_fraction(
    csf_mask: AnatomicalMask,
    points: np.ndarray,
) -> float:
    """Return the fraction of thermal-grid points occupied by CSF."""

    occupancy = map_csf_mask_to_points(
        csf_mask,
        points,
    )

    if not occupancy.size:
        return 0.0

    return float(np.mean(occupancy))
