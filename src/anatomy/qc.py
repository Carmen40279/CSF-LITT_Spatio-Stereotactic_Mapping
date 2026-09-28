"""
ThermoSEEG anatomical quality-control utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.anatomy.anatomical_model import (
    AnatomicalMask,
    AnatomicalModel,
)


@dataclass(frozen=True)
class AnatomicalQC:
    """
    Quality-control result for an anatomical model.
    """

    valid: bool
    issues: tuple[str, ...] = field(
        default_factory=tuple
    )
    warnings: tuple[str, ...] = field(
        default_factory=tuple
    )
    structures_checked: int = 0
    nonempty_structures: int = 0


def validate_anatomical_mask(
    mask: AnatomicalMask,
) -> list[str]:
    """
    Validate an individual anatomical mask.
    """

    issues: list[str] = []

    if mask.mask.ndim != 3:
        issues.append(
            f"{mask.name}: mask is not 3D."
        )

    if not np.issubdtype(
        mask.mask.dtype,
        np.bool_,
    ):
        issues.append(
            f"{mask.name}: mask is not boolean."
        )

    if not np.all(
        np.isfinite(mask.affine)
    ):
        issues.append(
            f"{mask.name}: affine contains non-finite values."
        )

    if mask.is_empty:
        issues.append(
            f"{mask.name}: mask is empty."
        )

    return issues


def validate_anatomical_model(
    model: AnatomicalModel,
) -> AnatomicalQC:
    """
    Validate all masks and surfaces in an anatomical model.
    """

    issues: list[str] = []
    warnings: list[str] = []

    structures = (
        len(model.masks)
        + len(model.surfaces)
    )

    nonempty = 0

    for mask in model.masks.values():

        mask_issues = validate_anatomical_mask(
            mask
        )

        issues.extend(mask_issues)

        if not mask.is_empty:
            nonempty += 1

    for surface in model.surfaces.values():

        if surface.vertex_count == 0:
            issues.append(
                f"{surface.name}: surface has no vertices."
            )

        if surface.face_count == 0:
            issues.append(
                f"{surface.name}: surface has no faces."
            )

        if surface.vertex_count > 0:
            nonempty += 1

    if structures == 0:
        warnings.append(
            "Anatomical model contains no structures."
        )

    return AnatomicalQC(
        valid=len(issues) == 0,
        issues=tuple(issues),
        warnings=tuple(warnings),
        structures_checked=structures,
        nonempty_structures=nonempty,
    )