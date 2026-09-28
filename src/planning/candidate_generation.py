"""
Deterministic generation of candidate LITT trajectories.

This module creates a configurable set of LITT trajectories that converge
on a specified target. Candidate generation is purely geometric; it does
not determine which trajectory is clinically preferable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from src.geometry.vectors import Point3D
from src.surgery.litt import LITTParameters, LITTPlan
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """
    Configuration for deterministic LITT candidate generation.

    entry_spacing:
        Distance between neighboring candidate entry positions in mm.

    search_radius:
        Maximum distance from the entry-center point in mm.

    active_length:
        Active LITT length in mm.

    radius:
        Ablation radius in mm.

    max_candidates:
        Maximum number of valid candidates returned.
    """

    entry_spacing: float = 3.0
    search_radius: float = 9.0
    active_length: float = 25.0
    radius: float = 4.0
    max_candidates: int = 81

    def __post_init__(self) -> None:
        if self.entry_spacing <= 0:
            raise ValueError("entry_spacing must be positive.")

        if self.search_radius < 0:
            raise ValueError("search_radius must be non-negative.")

        if self.active_length <= 0:
            raise ValueError("active_length must be positive.")

        if self.radius <= 0:
            raise ValueError("radius must be positive.")

        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")


def _distance_between_points(first: Point3D, second: Point3D) -> float:
    """Return Euclidean distance between two 3D points."""

    dx = first.x - second.x
    dy = first.y - second.y
    dz = first.z - second.z

    return (dx * dx + dy * dy + dz * dz) ** 0.5


def generate_entry_points(
    center: Point3D,
    config: CandidateGenerationConfig | None = None,
) -> list[Point3D]:
    """
    Generate a deterministic cubic grid of candidate entry points.

    Only points within search_radius of the center are retained.
    """

    if config is None:
        config = CandidateGenerationConfig()

    spacing = config.entry_spacing
    radius = config.search_radius

    steps = int(radius // spacing)

    points: list[Point3D] = []

    for ix, iy, iz in product(
        range(-steps, steps + 1),
        range(-steps, steps + 1),
        range(-steps, steps + 1),
    ):
        point = Point3D(
            center.x + ix * spacing,
            center.y + iy * spacing,
            center.z + iz * spacing,
        )

        if _distance_between_points(point, center) <= radius + 1e-9:
            points.append(point)

    return points


def generate_litt_candidates(
    target: Point3D,
    entry_center: Point3D,
    config: CandidateGenerationConfig | None = None,
    plan_id_prefix: str = "LITT-CANDIDATE",
) -> list[LITTPlan]:
    """
    Generate valid LITT candidate trajectories converging on a target.

    Each candidate starts at a distinct entry point and terminates at the
    same target.

    Candidates whose trajectory length is shorter than the configured
    active LITT length are rejected because the active ablation segment
    cannot physically extend beyond the trajectory.
    """

    if config is None:
        config = CandidateGenerationConfig()

    entry_points = generate_entry_points(entry_center, config)

    candidates: list[LITTPlan] = []

    for entry in entry_points:
        trajectory = Trajectory(
            entry=entry,
            target=target,
        )

        trajectory_length = trajectory.length

        if trajectory_length <= 1e-9:
            continue

        if trajectory_length + 1e-9 < config.active_length:
            continue

        parameters = LITTParameters(
            active_length=config.active_length,
            radius=config.radius,
        )

        plan_id = f"{plan_id_prefix}-{len(candidates) + 1:03d}"

        plan = LITTPlan(
            plan_id=plan_id,
            trajectory=trajectory,
            parameters=parameters,
        )

        candidates.append(plan)

        if len(candidates) >= config.max_candidates:
            break

    if not candidates:
        raise ValueError(
            "No valid LITT candidates were generated. "
            "Increase search_radius or reduce active_length."
        )

    return candidates


def generate_candidates(
    target: Point3D,
    entry_center: Point3D,
    config: CandidateGenerationConfig | None = None,
    plan_id_prefix: str = "LITT-CANDIDATE",
) -> list[LITTPlan]:
    """
    Convenience alias for generate_litt_candidates().
    """

    return generate_litt_candidates(
        target=target,
        entry_center=entry_center,
        config=config,
        plan_id_prefix=plan_id_prefix,
    )