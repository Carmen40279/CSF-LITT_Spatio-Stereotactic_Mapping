"""
ThermoSEEG NIfTI imaging loader.

Loads NIfTI medical imaging volumes and converts them into the
ThermoSEEG ImagingStudy abstraction.

This module deliberately does not perform registration,
segmentation, or anatomical interpretation. Its responsibility
is to ingest the imaging data faithfully and expose the spatial
metadata required by downstream processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.io.patient_case import ImagingStudy


# =====================================================================
# NIFTI LOADING
# =====================================================================


def load_nifti(
    path: str | Path,
    *,
    study_id: str | None = None,
    modality: str = "MRI",
    description: str = "",
) -> ImagingStudy:
    """
    Load a NIfTI image into an ImagingStudy.

    Parameters
    ----------
    path:
        Path to a .nii or .nii.gz file.

    study_id:
        Optional identifier. If omitted, the filename is used.

    modality:
        Imaging modality, for example MRI or CT.

    description:
        Human-readable description of the study.

    Returns
    -------
    ImagingStudy
        ThermoSEEG imaging representation.
    """

    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"NIfTI path is not a file: {image_path}"
        )

    suffix = image_path.name.lower()

    if not (
        suffix.endswith(".nii")
        or suffix.endswith(".nii.gz")
    ):
        raise ValueError(
            "Input file must be a NIfTI file (.nii or .nii.gz)."
        )

    # ---------------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------------

    image = nib.load(
        str(image_path)
    )

    # ---------------------------------------------------------------
    # IMAGE DATA
    # ---------------------------------------------------------------

    data = np.asarray(
        image.dataobj
    )

    if data.size == 0:
        raise ValueError(
            "NIfTI image contains no data."
        )

    # ---------------------------------------------------------------
    # SPATIAL METADATA
    # ---------------------------------------------------------------

    affine = np.asarray(
        image.affine,
        dtype=float,
    )

    header = image.header

    zooms = tuple(
        float(value)
        for value in header.get_zooms()[:3]
    )

    if len(zooms) != 3:
        raise ValueError(
            "NIfTI image does not provide three spatial dimensions."
        )

    if any(
        spacing <= 0
        for spacing in zooms
    ):
        raise ValueError(
            "NIfTI voxel spacing must be positive."
        )

    # ---------------------------------------------------------------
    # IDENTIFIER
    # ---------------------------------------------------------------

    resolved_study_id = (
        study_id
        if study_id is not None
        else image_path.name
    )

    # ---------------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------------

    metadata: dict[str, Any] = {
        "source_path": str(
            image_path.resolve()
        ),
        "format": "NIfTI",
        "shape": tuple(
            int(value)
            for value in data.shape
        ),
        "dtype": str(
            data.dtype
        ),
        "ndim": int(
            data.ndim
        ),
        "zooms": zooms,
        "affine": affine.tolist(),
        "qform_code": int(
            header["qform_code"]
        ),
        "sform_code": int(
            header["sform_code"]
        ),
    }

    # ---------------------------------------------------------------
    # CREATE IMAGING STUDY
    # ---------------------------------------------------------------

    return ImagingStudy(
        study_id=resolved_study_id,
        modality=modality,
        description=description,
        source=str(
            image_path.resolve()
        ),
        data=data,
        affine=affine,
        spacing=zooms,
        coordinate_system="nifti",
        metadata=metadata,
    )


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================


def get_nifti_shape(
    path: str | Path,
) -> tuple[int, ...]:
    """Return the spatial/data shape without loading the full array."""

    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {image_path}"
        )

    image = nib.load(
        str(image_path)
    )

    return tuple(
        int(value)
        for value in image.shape
    )


def get_nifti_metadata(
    path: str | Path,
) -> dict[str, Any]:
    """
    Read essential NIfTI metadata without converting the image
    into a NumPy array.
    """

    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {image_path}"
        )

    image = nib.load(
        str(image_path)
    )

    header = image.header

    zooms = tuple(
        float(value)
        for value in header.get_zooms()[:3]
    )

    return {
        "shape": tuple(
            int(value)
            for value in image.shape
        ),
        "zooms": zooms,
        "affine": np.asarray(
            image.affine,
            dtype=float,
        ),
        "qform_code": int(
            header["qform_code"]
        ),
        "sform_code": int(
            header["sform_code"]
        ),
    }