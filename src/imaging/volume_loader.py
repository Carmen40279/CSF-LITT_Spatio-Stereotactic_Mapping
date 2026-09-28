"""
ThermoSEEG — Medical Imaging Volume Loader

Loads NIfTI (.nii / .nii.gz) medical-imaging volumes and exposes:
    - voxel data
    - image dimensions
    - voxel spacing
    - affine transformation matrix
    - orientation information

This module intentionally performs no segmentation or analysis.
It provides a clean imaging layer for the ThermoSEEG pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class MedicalVolume:
    """Container for a loaded medical-imaging volume."""

    data: np.ndarray
    affine: np.ndarray
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    path: Path

    @property
    def ndim(self) -> int:
        """Number of dimensions in the image."""
        return len(self.shape)


def load_nifti(path: str | Path) -> MedicalVolume:
    """
    Load a NIfTI medical-imaging volume.

    Parameters
    ----------
    path:
        Path to a .nii or .nii.gz file.

    Returns
    -------
    MedicalVolume
        Loaded volume and its spatial metadata.

    Raises
    ------
    FileNotFoundError
        If the supplied file does not exist.

    ValueError
        If the file is not a NIfTI image.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Imaging file not found: {path}")

    if path.suffix not in {".nii", ".gz"}:
        raise ValueError(
            f"Expected a NIfTI file (.nii or .nii.gz), got: {path.name}"
        )

    image = nib.load(str(path))

    # Convert the image data into a NumPy array.
    data = np.asarray(image.dataobj)

    # Spatial transformation from voxel coordinates to world coordinates.
    affine = image.affine

    # Physical voxel dimensions, usually in millimeters.
    spacing = tuple(float(value) for value in image.header.get_zooms())

    return MedicalVolume(
        data=data,
        affine=affine,
        shape=data.shape,
        spacing=spacing,
        path=path,
    )


def describe_volume(volume: MedicalVolume) -> None:
    """Print a concise summary of a loaded medical volume."""

    print("\n=== ThermoSEEG Medical Volume ===")
    print(f"File:       {volume.path}")
    print(f"Dimensions: {volume.shape}")
    print(f"Spacing:    {volume.spacing}")
    print(f"Dimensions: {volume.ndim}D")

    print("\nAffine matrix:")
    print(volume.affine)

    print("\nIntensity range:")
    print(f"Minimum:    {float(np.min(volume.data)):.3f}")
    print(f"Maximum:    {float(np.max(volume.data)):.3f}")
    print("=================================\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "    python -m src.imaging.volume_loader <path_to_nifti>"
        )
        raise SystemExit(1)

    volume = load_nifti(sys.argv[1])
    describe_volume(volume)