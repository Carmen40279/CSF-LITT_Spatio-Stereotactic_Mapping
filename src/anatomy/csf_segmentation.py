"""
ThermoSEEG prototype CSF segmentation.

Provides a research/prototype CSF candidate mask from structural MRI.
This is NOT a clinically validated CSF segmentation algorithm.

ThermoSEEG should eventually accept externally supplied validated
CSF segmentations.
"""

from __future__ import annotations

import numpy as np

from src.anatomy.anatomical_model import AnatomicalMask
from src.io.patient_case import ImagingStudy


def normalize_volume(
    volume: np.ndarray,
) -> np.ndarray:
    """Normalize finite intensities to the range [0, 1]."""

    data = np.asarray(
        volume,
        dtype=float,
    )

    finite = np.isfinite(data)

    if not np.any(finite):
        raise ValueError(
            "Volume contains no finite values."
        )

    values = data[finite]

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum <= minimum:
        raise ValueError(
            "Volume has no usable intensity range."
        )

    normalized = (
        data - minimum
    ) / (
        maximum - minimum
    )

    normalized[~finite] = 0.0

    return normalized


def create_csf_mask(
    volume: np.ndarray,
    percentile: float = 35.0,
) -> np.ndarray:
    """
    Create a prototype CSF candidate mask.

    Uses low non-background intensities from the volume.
    This is a heuristic research approximation only.
    """

    data = np.asarray(
        volume,
        dtype=float,
    )

    if data.ndim != 3:
        raise ValueError(
            "CSF segmentation requires a 3D volume."
        )

    finite = np.isfinite(data)

    if not np.any(finite):
        raise ValueError(
            "Volume contains no finite values."
        )

    normalized = normalize_volume(
        data
    )

    # Exclude exact background first.
    foreground = (
        finite
        & (normalized > 0.01)
    )

    if not np.any(foreground):
        raise ValueError(
            "No non-background voxels were found."
        )

    foreground_values = normalized[
        foreground
    ]

    threshold = float(
        np.percentile(
            foreground_values,
            percentile,
        )
    )

    mask = (
        foreground
        & (normalized <= threshold)
    )

    return mask.astype(bool)


def segment_csf(
    study: ImagingStudy,
    percentile: float = 35.0,
) -> AnatomicalMask:
    """
    Generate a prototype CSF candidate mask from an imaging study.
    """

    if study.data is None:
        raise ValueError(
            "Imaging study contains no image data."
        )

    if study.data.ndim != 3:
        raise ValueError(
            "CSF segmentation requires a 3D volume."
        )

    spacing = (
        tuple(
            float(value)
            for value in study.spacing[:3]
        )
        if study.spacing is not None
        else (
            1.0,
            1.0,
            1.0,
        )
    )

    affine = (
        np.asarray(
            study.affine,
            dtype=float,
        )
        if study.affine is not None
        else np.eye(4)
    )

    mask = create_csf_mask(
        study.data,
        percentile=percentile,
    )

    return AnatomicalMask(
        name="csf",
        mask=mask,
        affine=affine,
        spacing=spacing,
        coordinate_space=(
            study.coordinate_system
            or "patient"
        ),
        source=(
            study.source
            or study.study_id
        ),
    )