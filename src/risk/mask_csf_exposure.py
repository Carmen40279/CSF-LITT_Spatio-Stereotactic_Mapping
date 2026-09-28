"""Patient-specific CSF exposure analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.anatomy.anatomical_model import AnatomicalMask
from src.trajectory.ablation import AblationGeometry


@dataclass(frozen=True)
class MaskCSFExposure:
    """Geometric exposure measurements against a CSF segmentation."""

    minimum_distance: float
    ablation_volume_mm3: float
    estimated_overlap_volume_mm3: float
    exposure_fraction: float
    intersects_csf: bool
    csf_voxel_count: int

    @property
    def exposure_percentage(self) -> float:
        """Return exposure fraction as a percentage."""
        return self.exposure_fraction * 100.0


def _validate_mask(csf_mask: AnatomicalMask) -> None:
    """Validate that the supplied anatomical mask can be used."""

    if not isinstance(csf_mask, AnatomicalMask):
        raise TypeError("csf_mask must be an AnatomicalMask.")

    if csf_mask.is_empty:
        raise ValueError("csf_mask cannot be empty.")

    if csf_mask.mask.ndim != 3:
        raise ValueError("csf_mask must contain a 3D mask.")


def _voxel_centers_in_patient_space(
    csf_mask: AnatomicalMask,
) -> np.ndarray:
    """Return CSF voxel centers transformed into patient coordinates."""

    indices = np.argwhere(csf_mask.mask > 0)

    if len(indices) == 0:
        return np.empty((0, 3), dtype=float)

    homogeneous = np.column_stack(
        (
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
            np.ones(len(indices)),
        )
    )

    coordinates = homogeneous @ csf_mask.affine.T

    return coordinates[:, :3]


def _point_to_segment_distance(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Calculate distances from points to a finite line segment."""

    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))

    if segment_length_squared == 0.0:
        return np.linalg.norm(points - start, axis=1)

    relative = points - start
    projection = (
        relative @ segment
    ) / segment_length_squared

    projection = np.clip(projection, 0.0, 1.0)

    closest = start + projection[:, None] * segment

    return np.linalg.norm(points - closest, axis=1)


def _estimate_ablation_volume(
    ablation: AblationGeometry,
) -> float:
    """Calculate the cylindrical ablation volume in cubic millimetres."""

    radius = float(ablation.radius)
    length = float(ablation.active_length)

    return float(np.pi * radius * radius * length)


def calculate_mask_csf_exposure(
    ablation: AblationGeometry,
    csf_mask: AnatomicalMask,
) -> MaskCSFExposure:
    """
    Calculate geometric LITT exposure relative to a patient-specific CSF mask"""

    _validate_mask(csf_mask)

    csf_points = _voxel_centers_in_patient_space(csf_mask)

    if len(csf_points) == 0:
        raise ValueError("csf_mask contains no positive voxels.")

    start = np.array(
        [
            ablation.trajectory.entry.x,
            ablation.trajectory.entry.y,
            ablation.trajectory.entry.z,
        ],
        dtype=float,
    )

    end = np.array(
        [
            ablation.trajectory.target.x,
            ablation.trajectory.target.y,
            ablation.trajectory.target.z,
        ],
        dtype=float,
    )

    distances = _point_to_segment_distance(
        csf_points,
        start,
        end,
    )

    minimum_distance = float(np.min(distances))

    exposed = distances <= float(ablation.radius)

    voxel_volume = float(np.prod(csf_mask.spacing))

    estimated_overlap_volume = float(
        np.count_nonzero(exposed) * voxel_volume
    )

    ablation_volume = _estimate_ablation_volume(ablation)

    exposure_fraction = min(
        estimated_overlap_volume / ablation_volume,
        1.0,
    )

    return MaskCSFExposure(
        minimum_distance=minimum_distance,
        ablation_volume_mm3=ablation_volume,
        estimated_overlap_volume_mm3=estimated_overlap_volume,
        exposure_fraction=exposure_fraction,
        intersects_csf=bool(np.any(exposed)),
        csf_voxel_count=int(len(csf_points)),
    )