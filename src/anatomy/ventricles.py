"""
ThermoSEEG ventricular representation.

Supports supplied ventricular segmentations and a simple prototype
candidate derived from a CSF mask.

The prototype derivation is not clinically validated.
"""

from __future__ import annotations

import numpy as np

from src.anatomy.anatomical_model import AnatomicalMask


def create_ventricle_candidate(
    csf_mask: AnatomicalMask,
) -> AnatomicalMask:
    """
    Create a conservative prototype ventricular candidate.

    The current implementation selects the central portion of the
    CSF mask. This is a structural placeholder for future
    ventricular segmentation.
    """

    mask = np.asarray(
        csf_mask.mask,
        dtype=bool,
    )

    if not np.any(mask):
        raise ValueError(
            "Cannot derive ventricles from an empty CSF mask."
        )

    coordinates = np.argwhere(mask)

    center = np.mean(
        coordinates,
        axis=0,
    )

    distance = np.linalg.norm(
        coordinates - center,
        axis=1,
    )

    threshold = float(
        np.percentile(
            distance,
            35.0,
        )
    )

    selected = coordinates[
        distance <= threshold
    ]

    ventricle_mask = np.zeros_like(
        mask,
        dtype=bool,
    )

    ventricle_mask[
        tuple(selected.T)
    ] = True

    return AnatomicalMask(
        name="ventricles",
        mask=ventricle_mask,
        affine=csf_mask.affine,
        spacing=csf_mask.spacing,
        coordinate_space=(
            csf_mask.coordinate_space
        ),
        source=(
            "prototype-derived-from-csf"
        ),
    )