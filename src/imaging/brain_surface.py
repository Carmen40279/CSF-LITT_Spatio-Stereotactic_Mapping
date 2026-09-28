"""
ThermoSEEG — Brain Surface Reconstruction

Converts a structural T1-weighted MRI volume into a 3D anatomical
surface mesh suitable for visualization and later spatial planning.

Pipeline:
    NIfTI MRI
        ↓
    intensity normalization
        ↓
    brain-region mask
        ↓
    morphological cleanup
        ↓
    marching-cubes surface extraction
        ↓
    mesh smoothing
        ↓
    PyVista PolyData
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import measure


MRI_PATH = Path(
    "data/sample/sub-01_ses-forrestgump_T1w.nii.gz"
)


def load_mri(path: Path):
    """Load MRI data and spatial metadata."""
    image = nib.load(str(path))

    data = np.asarray(image.dataobj, dtype=np.float32)

    spacing = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]
    )

    affine = image.affine

    return data, spacing, affine


def normalize_volume(data: np.ndarray) -> np.ndarray:
    """
    Normalize MRI intensities using robust percentile scaling.

    This reduces the effect of extreme intensity values and makes
    thresholding more stable across MRI volumes.
    """

    low = np.percentile(data, 1)
    high = np.percentile(data, 99)

    normalized = (data - low) / (high - low + 1e-8)

    return np.clip(normalized, 0.0, 1.0)


def create_brain_mask(
    normalized: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """
    Create an approximate brain-region mask.

    This is intentionally an anatomical visualization mask rather
    than a clinical-grade segmentation.
    """

    mask = normalized > threshold

    # Remove small isolated regions.
    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3, 3)),
    )

    # Fill small internal holes.
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((5, 5, 5)),
    )

    # Keep only the largest connected component.
    labels, number = ndimage.label(mask)

    if number == 0:
        raise RuntimeError(
            "No brain-like region could be extracted from the MRI."
        )

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0

    largest_component = np.argmax(component_sizes)

    mask = labels == largest_component

    return mask


def extract_surface(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> pv.PolyData:
    """
    Convert a binary brain mask into a 3D surface mesh.
    """

    vertices, faces, normals, values = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=spacing,
    )

    # PyVista requires faces in:
    # [number_of_vertices, v1, v2, v3, ...]
    faces_pv = np.column_stack(
        (
            np.full(len(faces), 3),
            faces,
        )
    ).astype(np.int64)

    mesh = pv.PolyData(
        vertices,
        faces_pv,
    )

    return mesh


def smooth_surface(
    mesh: pv.PolyData,
    iterations: int = 30,
) -> pv.PolyData:
    """Smooth the reconstructed anatomical surface."""

    return mesh.smooth(
        n_iter=iterations,
        relaxation_factor=0.01,
        feature_smoothing=False,
        boundary_smoothing=True,
    )


def build_brain_surface(
    path: Path = MRI_PATH,
) -> pv.PolyData:
    """Complete MRI → brain-surface reconstruction pipeline."""

    print("\n=== ThermoSEEG Brain Reconstruction ===")

    data, spacing, affine = load_mri(path)

    print(f"MRI shape:      {data.shape}")
    print(f"Voxel spacing:  {spacing}")

    normalized = normalize_volume(data)

    print("Intensity normalization: complete")

    mask = create_brain_mask(normalized)

    print(
        f"Brain mask voxels: {int(mask.sum()):,}"
    )

    mesh = extract_surface(
        mask,
        spacing,
    )

    print(
        f"Initial mesh: "
        f"{mesh.n_points:,} vertices / "
        f"{mesh.n_cells:,} faces"
    )

    mesh = smooth_surface(mesh)

    print(
        f"Smoothed mesh: "
        f"{mesh.n_points:,} vertices / "
        f"{mesh.n_cells:,} faces"
    )

    print("========================================\n")

    return mesh


def show_brain(
    path: Path = MRI_PATH,
):
    """Display the reconstructed brain in PyVista."""

    brain = build_brain_surface(path)

    plotter = pv.Plotter()

    plotter.add_mesh(
        brain,
        smooth_shading=True,
        opacity=1.0,
        show_edges=False,
    )

    plotter.add_text(
        "ThermoSEEG — Anatomical Reconstruction",
        position="upper_left",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


if __name__ == "__main__":
    show_brain()