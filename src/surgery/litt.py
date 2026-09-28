from __future__ import annotations

from dataclasses import dataclass, field

from src.geometry.vectors import Point3D
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class LITTParameters:
    """Geometric parameters for the active LITT ablation region."""

    active_length: float
    radius: float

    def __post_init__(self) -> None:
        if self.active_length <= 0:
            raise ValueError("LITT active length must be positive.")

        if self.radius <= 0:
            raise ValueError("LITT radius must be positive.")


@dataclass
class LITTPlan:
    """A patient-specific LITT trajectory and its ablation parameters."""

    plan_id: str
    trajectory: Trajectory
    parameters: LITTParameters

    coordinate_space: str = "patient"

    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def entry(self) -> Point3D:
        """Return the LITT entry point."""

        return self.trajectory.entry

    @property
    def target(self) -> Point3D:
        """Return the LITT target point."""

        return self.trajectory.target

    @property
    def trajectory_length(self) -> float:
        """Return the total trajectory length."""

        return self.trajectory.length