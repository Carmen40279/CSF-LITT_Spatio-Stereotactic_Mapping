"""
ThermoSEEG registration utilities.

This module currently provides the representation of a registration
result and a wrapper for externally supplied affine transformations.

Actual image-to-image registration will be added later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.registration.transforms import SpatialTransform


@dataclass(frozen=True)
class RegistrationResult:
    """Result of a spatial registration operation."""

    transform: SpatialTransform
    method: str
    metric_name: str | None = None
    metric_value: float | None = None
    iterations: int = 0
    converged: bool = True
    notes: str = ""


def create_affine_registration(
    source_space: str,
    target_space: str,
    matrix: np.ndarray,
    method: str = "provided_affine",
) -> RegistrationResult:
    """
    Wrap an externally supplied affine transformation.

    This does not perform image registration itself.
    """

    transform = SpatialTransform(
        source_space=source_space,
        target_space=target_space,
        matrix=np.asarray(
            matrix,
            dtype=float,
        ),
    )

    return RegistrationResult(
        transform=transform,
        method=method,
        metric_name=None,
        metric_value=None,
        iterations=0,
        converged=True,
        notes=(
            "Affine transformation supplied externally; "
            "image registration was not performed."
        ),
    )