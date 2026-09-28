"""
Pareto analysis for ThermoSEEG trajectory candidates.

Candidates are compared across multiple computational objectives without
assigning a single weighted score or clinical recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.planning.trajectory_comparison import (
    ComparisonTable,
    TrajectoryComparison,
)


@dataclass(frozen=True)
class ParetoResult:
    """Result of multi-objective Pareto analysis."""

    non_dominated: list[TrajectoryComparison]
    dominated: list[TrajectoryComparison]

    @property
    def non_dominated_count(self) -> int:
        return len(self.non_dominated)

    @property
    def dominated_count(self) -> int:
        return len(self.dominated)

    @property
    def non_dominated_ids(self) -> list[str]:
        return [candidate.plan_id for candidate in self.non_dominated]


def _objective_values(
    candidate: TrajectoryComparison,
) -> tuple[float, float, float, float, float]:
    """
    Return normalized objective values.

    Every objective is represented as a quantity to minimize.
    SEEG clearance is therefore negated because greater clearance is
    preferable from a geometric separation perspective.
    """

    seeg_clearance = candidate.minimum_seeg_distance

    if seeg_clearance is None:
        seeg_objective = 0.0
    else:
        seeg_objective = -seeg_clearance

    return (
        candidate.target_distance,
        candidate.csf_exposure_fraction,
        candidate.maximum_thermal_risk,
        seeg_objective,
        candidate.trajectory_length,
    )


def _dominates(
    first: TrajectoryComparison,
    second: TrajectoryComparison,
) -> bool:
    """
    Return True when first dominates second.

    A candidate dominates another when it is no worse in every objective
    and strictly better in at least one objective.
    """

    first_values = _objective_values(first)
    second_values = _objective_values(second)

    no_worse = all(
        first_value <= second_value
        for first_value, second_value in zip(
            first_values,
            second_values,
        )
    )

    strictly_better = any(
        first_value < second_value
        for first_value, second_value in zip(
            first_values,
            second_values,
        )
    )

    return no_worse and strictly_better


def find_non_dominated(
    candidates: list[TrajectoryComparison],
) -> list[TrajectoryComparison]:
    """Return all candidates on the Pareto frontier."""

    non_dominated: list[TrajectoryComparison] = []

    for candidate in candidates:
        dominated = False

        for other in candidates:
            if other.plan_id == candidate.plan_id:
                continue

            if _dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            non_dominated.append(candidate)

    return non_dominated


def analyze_pareto(
    table: ComparisonTable,
) -> ParetoResult:
    """Perform Pareto analysis on a comparison table."""

    non_dominated = find_non_dominated(table.candidates)

    non_dominated_ids = {
        candidate.plan_id
        for candidate in non_dominated
    }

    dominated = [
        candidate
        for candidate in table.candidates
        if candidate.plan_id not in non_dominated_ids
    ]

    return ParetoResult(
        non_dominated=non_dominated,
        dominated=dominated,
    )