"""
ThermoSEEG planning data models.

These models provide a structured representation of a planning
case before it reaches the computational engines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.vectors import Point3D


@dataclass(frozen=True)
class Coordinate3D:
    """A coordinate in the current stereotactic coordinate space."""

    x: float
    y: float
    z: float

    def to_point(self) -> Point3D:
        """Convert to the geometry engine's Point3D."""

        return Point3D(
            self.x,
            self.y,
            self.z,
        )

    def to_array(self) -> np.ndarray:
        """Convert to a NumPy array for visualization."""

        return np.array(
            [
                self.x,
                self.y,
                self.z,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class SEEGPlan:
    """SEEG electrode planning parameters."""

    entry: Coordinate3D
    target: Coordinate3D
    contacts: int


@dataclass(frozen=True)
class AblationParameters:
    """LITT ablation geometry parameters."""

    active_length: float
    radius: float


@dataclass(frozen=True)
class LITTPlan:
    """LITT trajectory planning parameters."""

    entry: Coordinate3D
    target: Coordinate3D
    ablation: AblationParameters


@dataclass(frozen=True)
class CSFParameters:
    """Synthetic CSF compartment parameters."""

    center: Coordinate3D
    radii: Coordinate3D
    resolution: int


@dataclass(frozen=True)
class PlanningCase:
    """
    Complete ThermoSEEG planning case.

    This is the object that future interface systems will create.
    """

    case_id: str
    seeg: SEEGPlan
    litt: LITTPlan
    csf: CSFParameters