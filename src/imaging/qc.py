"""
ThermoSEEG imaging quality-control checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.patient_case import ImagingStudy


@dataclass(frozen=True)
class ImagingQC:
    """Basic structural quality-control result."""

    valid: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    shape: tuple[int, ...]
    spacing: tuple[float, float, float] | None

    finite_fraction: float
    nonzero_fraction: float


def validate_imaging_study(
    study: ImagingStudy,
) -> ImagingQC:
    """
    Validate basic structural properties of an ImagingStudy.

    This is software/data QC, not clinical interpretation.
    """

    issues: list[str] = []
    warnings: list[str] = []

    if study.data is None:

        issues.append(
            "Imaging study contains no image data."
        )

        return ImagingQC(
            valid=False,
            issues=tuple(issues),
            warnings=tuple(warnings),
            shape=(),
            spacing=study.spacing,
            finite_fraction=0.0,
            nonzero_fraction=0.0,
        )

    data = np.asarray(
        study.data
    )

    shape = tuple(
        int(value)
        for value in data.shape
    )

    if data.ndim < 3:

        issues.append(
            "Imaging volume must have at least three dimensions."
        )

    if data.size == 0:

        issues.append(
            "Imaging volume is empty."
        )

    finite = np.isfinite(
        data
    )

    finite_fraction = float(
        np.mean(finite)
    ) if data.size else 0.0

    if finite_fraction < 1.0:

        issues.append(
            "Imaging volume contains non-finite values."
        )

    nonzero_fraction = float(
        np.count_nonzero(data)
        / data.size
    ) if data.size else 0.0

    if nonzero_fraction == 0.0:

        issues.append(
            "Imaging volume contains no non-zero voxels."
        )

    if study.spacing is None:

        issues.append(
            "Voxel spacing is missing."
        )

    else:

        if len(study.spacing) != 3:

            issues.append(
                "Voxel spacing must contain three values."
            )

        elif any(
            spacing <= 0
            for spacing in study.spacing
        ):

            issues.append(
                "Voxel spacing must be positive."
            )

    if data.ndim > 3:

        warnings.append(
            "Imaging data contains additional dimensions."
        )

    valid = len(issues) == 0

    return ImagingQC(
        valid=valid,
        issues=tuple(issues),
        warnings=tuple(warnings),
        shape=shape,
        spacing=study.spacing,
        finite_fraction=finite_fraction,
        nonzero_fraction=nonzero_fraction,
    )