"""
ThermoSEEG brain extraction.

Provides a prototype brain-mask extraction pipeline from
3D structural MRI.

This is intended for research/prototyping and is not a
clinically validated brain-extraction algorithm.
"""

from __future__ import annotations

import numpy as np

from src.anatomy.anatomical_model import AnatomicalMask
from src.io.patient_case import ImagingStudy


def normalize_volume(
    data: np.ndarray,
) -> np.ndarray:
    """
    Normalize an imaging volume to the range [0, 1].
    """

    volume = np.asarray(
        data,
        dtype=float,
    )

    finite = np.isfinite(volume)

    if not np.any(finite):
        raise ValueError(
            "Imaging volume contains no finite values."
        )

    minimum = float(
        np.min(volume[finite])
    )

    maximum = float(
        np.max(volume[finite])
    )

    if maximum <= minimum:
        raise ValueError(
            "Imaging volume has no usable intensity range."
        )

    normalized = (
        volume - minimum
    ) / (
        maximum - minimum
    )

    normalized[~finite] = 0.0

    return normalized


def calculate_threshold(
    volume: np.ndarray,
) -> float:
    """
    Calculate a simple intensity threshold.

    Uses a percentile-based threshold suitable for the
    prototype brain-extraction pipeline.
    """

    values = volume[
        np.isfinite(volume)
    ]

    if values.size == 0:
        raise ValueError(
            "Cannot calculate threshold from empty volume."
        )

    return float(
        np.percentile(
            values,
            35.0,
        )
    )


def create_brain_mask(
    volume: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Create a prototype binary brain mask.
    """

    normalized = normalize_volume(
        volume
    )

    if threshold is None:
        threshold = calculate_threshold(
            normalized
        )

    mask = normalized > threshold

    return mask.astype(bool)


def extract_brain_mask(
    study: ImagingStudy,
    threshold: float | None = None,
) -> AnatomicalMask:
    """
    Extract a prototype brain mask from an ImagingStudy.
    """

    if study.data is None:
        raise ValueError(
            f"Imaging study {study.study_id!r} "
            "does not contain image data."
        )

    if study.data.ndim != 3:
        raise ValueError(
            "Brain extraction requires a 3D imaging volume."
        )

    if study.affine is None:
        affine = np.eye(4)
    else:
        affine = np.asarray(
            study.affine,
            dtype=float,
        )

    if study.spacing is None:
        spacing = (
            1.0,
            1.0,
            1.0,
        )
    else:
        spacing = tuple(
            float(value)
            for value in study.spacing[:3]
        )

    mask = create_brain_mask(
        study.data,
        threshold=threshold,
    )

    return AnatomicalMask(
        name="brain",
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


def extract_brain(
    study: ImagingStudy,
    threshold: float | None = None,
) -> AnatomicalMask:
    """
    Convenience alias for extract_brain_mask().
    """

    return extract_brain_mask(
        study,
        threshold=threshold,
    )