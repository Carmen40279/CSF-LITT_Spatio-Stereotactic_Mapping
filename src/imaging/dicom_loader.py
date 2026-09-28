"""
ThermoSEEG DICOM imaging loader.

Discovers and loads a single DICOM imaging series into the
ThermoSEEG ImagingStudy abstraction.

This module handles ingestion and spatial metadata. It does not
perform registration, segmentation, or clinical interpretation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from src.io.patient_case import ImagingStudy


# =====================================================================
# DICOM DISCOVERY
# =====================================================================


def discover_dicom_files(
    directory: str | Path,
) -> list[Path]:
    """
    Recursively discover readable DICOM files.

    Files that cannot be interpreted as DICOM are ignored.
    """

    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"DICOM directory not found: {root}"
        )

    if not root.is_dir():
        raise ValueError(
            f"DICOM path is not a directory: {root}"
        )

    files: list[Path] = []

    for path in sorted(root.rglob("*")):

        if not path.is_file():
            continue

        try:
            pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
            )

        except (
            InvalidDicomError,
            OSError,
            ValueError,
        ):
            continue

        files.append(path)

    if not files:
        raise ValueError(
            f"No readable DICOM files found in: {root}"
        )

    return files


# =====================================================================
# SERIES DISCOVERY
# =====================================================================


def discover_dicom_series(
    directory: str | Path,
) -> dict[str, list[Path]]:
    """
    Group DICOM files by SeriesInstanceUID.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from series UID to DICOM files.
    """

    files = discover_dicom_files(
        directory
    )

    series: dict[str, list[Path]] = {}

    for path in files:

        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
        )

        series_uid = getattr(
            dataset,
            "SeriesInstanceUID",
            None,
        )

        if series_uid is None:
            continue

        series.setdefault(
            str(series_uid),
            [],
        ).append(path)

    if not series:
        raise ValueError(
            "Readable DICOM files were found, "
            "but no SeriesInstanceUID values were available."
        )

    return series


# =====================================================================
# SLICE SORTING
# =====================================================================


def _slice_position(
    dataset: Any,
) -> float:
    """
    Determine a slice position for ordering.

    ImagePositionPatient is preferred. SliceLocation is used
    as a fallback.
    """

    position = getattr(
        dataset,
        "ImagePositionPatient",
        None,
    )

    if position is not None and len(position) >= 3:

        return float(
            position[2]
        )

    location = getattr(
        dataset,
        "SliceLocation",
        None,
    )

    if location is not None:
        return float(
            location
        )

    instance = getattr(
        dataset,
        "InstanceNumber",
        0,
    )

    return float(
        instance
    )


def _sort_dicom_slices(
    datasets: list[Any],
) -> list[Any]:
    """Sort DICOM slices into spatial order."""

    return sorted(
        datasets,
        key=_slice_position,
    )


# =====================================================================
# SERIES LOADING
# =====================================================================


def load_dicom_series(
    directory: str | Path,
    *,
    series_uid: str | None = None,
    study_id: str | None = None,
    description: str = "",
) -> ImagingStudy:
    """
    Load a DICOM series into an ImagingStudy.

    Parameters
    ----------
    directory:
        Directory containing DICOM files.

    series_uid:
        Specific SeriesInstanceUID to load. If omitted, the largest
        available series is selected.

    study_id:
        Optional ThermoSEEG study identifier.

    description:
        Human-readable description.

    Returns
    -------
    ImagingStudy
        Loaded DICOM volume.
    """

    series = discover_dicom_series(
        directory
    )

    # ---------------------------------------------------------------
    # SELECT SERIES
    # ---------------------------------------------------------------

    if series_uid is not None:

        if series_uid not in series:
            raise KeyError(
                f"DICOM series not found: {series_uid}"
            )

        selected_files = series[
            series_uid
        ]

    else:

        selected_uid = max(
            series,
            key=lambda uid: len(
                series[uid]
            ),
        )

        selected_files = series[
            selected_uid
        ]

        series_uid = selected_uid

    # ---------------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------------

    datasets: list[Any] = []

    for path in selected_files:

        dataset = pydicom.dcmread(
            str(path)
        )

        if not hasattr(
            dataset,
            "PixelData",
        ):
            continue

        datasets.append(
            dataset
        )

    if not datasets:
        raise ValueError(
            "Selected DICOM series contains no pixel data."
        )

    datasets = _sort_dicom_slices(
        datasets
    )

    first = datasets[0]

    # ---------------------------------------------------------------
    # PIXEL DATA
    # ---------------------------------------------------------------

    slices: list[np.ndarray] = []

    for dataset in datasets:

        array = dataset.pixel_array

        array = np.asarray(
            array
        )

        if array.ndim != 2:
            raise ValueError(
                "This loader currently expects "
                "single-frame 2D DICOM slices."
            )

        slices.append(
            array
        )

    reference_shape = slices[0].shape

    if any(
        array.shape != reference_shape
        for array in slices
    ):
        raise ValueError(
            "DICOM slices have inconsistent image dimensions."
        )

    volume = np.stack(
        slices,
        axis=0,
    )

    # ---------------------------------------------------------------
    # RESCALE
    # ---------------------------------------------------------------

    slope = float(
        getattr(
            first,
            "RescaleSlope",
            1.0,
        )
    )

    intercept = float(
        getattr(
            first,
            "RescaleIntercept",
            0.0,
        )
    )

    volume = (
        volume.astype(
            np.float32
        )
        * slope
        + intercept
    )

    # ---------------------------------------------------------------
    # SPACING
    # ---------------------------------------------------------------

    pixel_spacing = getattr(
        first,
        "PixelSpacing",
        None,
    )

    if pixel_spacing is None:
        raise ValueError(
            "DICOM series is missing PixelSpacing."
        )

    row_spacing = float(
        pixel_spacing[0]
    )

    column_spacing = float(
        pixel_spacing[1]
    )

    if len(datasets) > 1:

        positions = [
            _slice_position(
                dataset
            )
            for dataset in datasets
        ]

        differences = np.diff(
            positions
        )

        differences = np.abs(
            differences[
                np.nonzero(
                    differences
                )
            ]
        )

        if len(differences) > 0:
            slice_spacing = float(
                np.median(
                    differences
                )
            )
        else:
            slice_spacing = float(
                getattr(
                    first,
                    "SliceThickness",
                    1.0,
                )
            )

    else:

        slice_spacing = float(
            getattr(
                first,
                "SliceThickness",
                1.0,
            )
        )

    spacing = (
        slice_spacing,
        row_spacing,
        column_spacing,
    )

    if any(
        value <= 0
        for value in spacing
    ):
        raise ValueError(
            "DICOM voxel spacing must be positive."
        )

    # ---------------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------------

    metadata: dict[str, Any] = {
        "format": "DICOM",
        "series_uid": str(
            series_uid
        ),
        "series_description": str(
            getattr(
                first,
                "SeriesDescription",
                "",
            )
        ),
        "study_instance_uid": str(
            getattr(
                first,
                "StudyInstanceUID",
                "",
            )
        ),
        "modality": str(
            getattr(
                first,
                "Modality",
                "",
            )
        ),
        "manufacturer": str(
            getattr(
                first,
                "Manufacturer",
                "",
            )
        ),
        "patient_position": str(
            getattr(
                first,
                "PatientPosition",
                "",
            )
        ),
        "shape": tuple(
            int(value)
            for value in volume.shape
        ),
        "spacing": spacing,
        "slice_count": len(
            datasets
        ),
        "source_directory": str(
            Path(directory).resolve()
        ),
    }

    # ---------------------------------------------------------------
    # STUDY ID
    # ---------------------------------------------------------------

    resolved_study_id = (
        study_id
        if study_id is not None
        else str(series_uid)
    )

    # ---------------------------------------------------------------
    # IMAGING STUDY
    # ---------------------------------------------------------------

    return ImagingStudy(
        study_id=resolved_study_id,
        modality=str(
            getattr(
                first,
                "Modality",
                "UNKNOWN",
            )
        ),
        description=description,
        source=str(
            Path(directory).resolve()
        ),
        data=volume,
        affine=None,
        spacing=spacing,
        coordinate_system="dicom_patient",
        metadata=metadata,
    )


# =====================================================================
# SERIES SUMMARY
# =====================================================================


def summarize_dicom_series(
    directory: str | Path,
) -> list[dict[str, Any]]:
    """
    Return a lightweight summary of every discovered DICOM series.

    This is useful before selecting a series for ingestion.
    """

    series = discover_dicom_series(
        directory
    )

    summaries: list[dict[str, Any]] = []

    for uid, files in series.items():

        first = pydicom.dcmread(
            str(files[0]),
            stop_before_pixels=True,
        )

        summaries.append(
            {
                "series_uid": uid,
                "file_count": len(files),
                "modality": str(
                    getattr(
                        first,
                        "Modality",
                        "UNKNOWN",
                    )
                ),
                "description": str(
                    getattr(
                        first,
                        "SeriesDescription",
                        "",
                    )
                ),
                "study_uid": str(
                    getattr(
                        first,
                        "StudyInstanceUID",
                        "",
                    )
                ),
            }
        )

    return summaries