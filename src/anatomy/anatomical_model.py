"""
ThermoSEEG patient-specific anatomical model.

Provides structured representations of anatomy derived from
patient imaging or supplied segmentations.

This module is intentionally independent of visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AnatomicalMask:
    """
    A 3D anatomical segmentation mask.

    The mask is stored as a boolean NumPy array. The affine maps
    voxel coordinates into the associated anatomical coordinate space.
    """

    name: str
    mask: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    coordinate_space: str = "patient"
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "Anatomical mask name cannot be empty."
            )

        if self.mask.ndim != 3:
            raise ValueError(
                "Anatomical mask must be a 3D array."
            )

        if self.affine.shape != (4, 4):
            raise ValueError(
                "Anatomical mask affine must have shape (4, 4)."
            )

        if not np.all(np.isfinite(self.affine)):
            raise ValueError(
                "Anatomical mask affine contains non-finite values."
            )

        if len(self.spacing) != 3:
            raise ValueError(
                "Anatomical mask spacing must contain three values."
            )

        if any(value <= 0 for value in self.spacing):
            raise ValueError(
                "Anatomical mask spacing must be positive."
            )

    @property
    def voxel_count(self) -> int:
        """Number of voxels belonging to the structure."""

        return int(np.count_nonzero(self.mask))

    @property
    def volume_mm3(self) -> float:
        """Approximate structure volume in cubic millimetres."""

        voxel_volume = (
            self.spacing[0]
            * self.spacing[1]
            * self.spacing[2]
        )

        return self.voxel_count * voxel_volume

    @property
    def is_empty(self) -> bool:
        """Whether the segmentation contains no voxels."""

        return self.voxel_count == 0


@dataclass(frozen=True)
class AnatomicalSurface:
    """
    Surface representation of an anatomical structure.

    Vertices are stored in the associated coordinate space.
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    coordinate_space: str = "patient"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "Anatomical surface name cannot be empty."
            )

        if self.vertices.ndim != 2:
            raise ValueError(
                "Surface vertices must be a 2D array."
            )

        if self.vertices.shape[1] != 3:
            raise ValueError(
                "Surface vertices must have shape (N, 3)."
            )

        if self.faces.ndim != 2:
            raise ValueError(
                "Surface faces must be a 2D array."
            )

        if not np.all(np.isfinite(self.vertices)):
            raise ValueError(
                "Surface vertices contain non-finite values."
            )

    @property
    def vertex_count(self) -> int:
        """Number of surface vertices."""

        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of surface faces."""

        return int(self.faces.shape[0])


@dataclass
class AnatomicalModel:
    """
    Patient-specific anatomical representation.

    The model can contain masks and surface representations for
    the brain, CSF, ventricles, lesions, targets, vessels, and
    other critical structures.
    """

    coordinate_space: str = "patient"

    masks: dict[str, AnatomicalMask] = field(
        default_factory=dict
    )

    surfaces: dict[str, AnatomicalSurface] = field(
        default_factory=dict
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def add_mask(
        self,
        mask: AnatomicalMask,
    ) -> None:
        """Add or replace an anatomical mask."""

        self.masks[mask.name] = mask

    def add_surface(
        self,
        surface: AnatomicalSurface,
    ) -> None:
        """Add or replace an anatomical surface."""

        self.surfaces[surface.name] = surface

    def get_mask(
        self,
        name: str,
    ) -> AnatomicalMask:
        """Retrieve an anatomical mask by name."""

        if name not in self.masks:
            raise KeyError(
                f"Anatomical mask {name!r} was not found."
            )

        return self.masks[name]

    def get_surface(
        self,
        name: str,
    ) -> AnatomicalSurface:
        """Retrieve an anatomical surface by name."""

        if name not in self.surfaces:
            raise KeyError(
                f"Anatomical surface {name!r} was not found."
            )

        return self.surfaces[name]

    def has_mask(
        self,
        name: str,
    ) -> bool:
        """Check whether a mask exists."""

        return name in self.masks

    def has_surface(
        self,
        name: str,
    ) -> bool:
        """Check whether a surface exists."""

        return name in self.surfaces

    @property
    def available_structures(self) -> list[str]:
        """Return names of all available anatomical structures."""

        return sorted(
            set(self.masks) | set(self.surfaces)
        )

    def summary(self) -> dict[str, object]:
        """Return a compact model summary."""

        return {
            "coordinate_space": self.coordinate_space,
            "masks": sorted(self.masks.keys()),
            "surfaces": sorted(self.surfaces.keys()),
            "structure_count": len(
                self.available_structures
            ),
            "metadata": dict(self.metadata),
        }