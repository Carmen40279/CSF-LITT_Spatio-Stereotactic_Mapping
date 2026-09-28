"""
Comparison utilities for ThermoSEEG LITT candidate trajectories.
This module converts CandidateEvaluation objects into a structured
comparison table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.planning.evaluation import (
    CandidateEvaluation,
)
from src.planning.planning_engine import (
    PlanningResult,
)


@dataclass(frozen=True)
class TrajectoryComparison:
    """Comparable metrics for one LITT candidate."""

    plan_id: str
    trajectory_length: float

    reaches_target: bool
    target_distance: float

    minimum_csf_distance: float
    csf_exposure_fraction: float
    maximum_thermal_risk: float

    minimum_seeg_distance: float

    uncertainty_exposure_mean: float
    uncertainty_exposure_std: float
    uncertainty_intersection_probability: float


@dataclass(frozen=True)
class ComparisonTable:
    """Collection of candidate trajectory comparisons."""

    candidates: tuple[TrajectoryComparison, ...]

    @property
    def count(self) -> int:
        """Number of candidates in the comparison."""

        return len(self.candidates)

    @property
    def plan_ids(self) -> tuple[str, ...]:
        """Return candidate IDs in comparison order."""

        return tuple(
            candidate.plan_id
            for candidate in self.candidates
        )


def _minimum_seeg_distance(
    evaluation: CandidateEvaluation,
) -> float:
    """
    Return the minimum LITT-to-SEEG trajectory distance.

    If no SEEG electrodes are present, NaN is returned.
    """

    if not evaluation.seeg_relationships:
        return float("nan")

    return min(
        relationship.minimum_distance
        for relationship in evaluation.seeg_relationships
    )


def _uncertainty_metrics(
    evaluation: CandidateEvaluation,
) -> tuple[float, float, float]:
    """
    Extract uncertainty metrics using the actual
    UncertaintyAssessment field names.
    """

    uncertainty = evaluation.uncertainty

    if uncertainty is None:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return (
        float(
            uncertainty.exposure_fraction_mean
        ),
        float(
            uncertainty.exposure_fraction_std
        ),
        float(
            uncertainty.intersection_probability
        ),
    )


def compare_candidate(
    evaluation: CandidateEvaluation,
) -> TrajectoryComparison:
    """
    Convert one CandidateEvaluation into comparison metrics.
    """

    uncertainty_mean, uncertainty_std, intersection_probability = (
        _uncertainty_metrics(evaluation)
    )

    return TrajectoryComparison(
        plan_id=evaluation.plan_id,

        trajectory_length=float(
            evaluation.trajectory_length
        ),

        reaches_target=bool(
            evaluation.target.reaches_target
        ),

        target_distance=float(
            evaluation.target.target_distance
        ),

        minimum_csf_distance=float(
            evaluation.minimum_csf_distance
        ),

        csf_exposure_fraction=float(
            evaluation.csf_exposure_fraction
        ),

        maximum_thermal_risk=float(
            evaluation.maximum_thermal_risk
        ),

        minimum_seeg_distance=float(
            _minimum_seeg_distance(evaluation)
        ),

        uncertainty_exposure_mean=uncertainty_mean,

        uncertainty_exposure_std=uncertainty_std,

        uncertainty_intersection_probability=(
            intersection_probability
        ),
    )


def compare_candidates(
    evaluations: list[CandidateEvaluation],
) -> ComparisonTable:
    """
    Compare a collection of candidate evaluations.
    """

    comparisons = tuple(
        compare_candidate(evaluation)
        for evaluation in evaluations
    )

    return ComparisonTable(
        candidates=comparisons
    )


def compare_planning_result(
    result: PlanningResult,
) -> ComparisonTable:
    """
    Create a comparison table from a complete planning result.
    """

    evaluations = list(
        result.candidate_evaluations
    )

    return compare_candidates(
        evaluations
    )