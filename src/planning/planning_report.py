"""This module packages candidate comparison and Pareto-analysis results into
a serializable research-planning report"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.planning.pareto import ParetoResult, analyze_pareto
from src.planning.planning_engine import PlanningResult
from src.planning.trajectory_comparison import (
    ComparisonTable,
    compare_planning_result,
)


@dataclass(frozen=True)
class PlanningReport:
    """Complete structured output of a ThermoSEEG planning run."""

    case_id: str
    candidate_count: int
    comparison: ComparisonTable
    pareto: ParetoResult
    metadata: dict[str, object]

    @property
    def non_dominated_count(self) -> int:
        return self.pareto.non_dominated_count

    @property
    def non_dominated_ids(self) -> list[str]:
        return self.pareto.non_dominated_ids

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete report into JSON-compatible data."""

        return {
            "case_id": self.case_id,
            "candidate_count": self.candidate_count,
            "comparison": [
                asdict(candidate)
                for candidate in self.comparison.candidates
            ],
            "pareto": {
                "non_dominated": [
                    asdict(candidate)
                    for candidate in self.pareto.non_dominated
                ],
                "dominated": [
                    asdict(candidate)
                    for candidate in self.pareto.dominated
                ],
            },
            "metadata": dict(self.metadata),
        }


def create_planning_report(
    result: PlanningResult,
) -> PlanningReport:
    """
    Build a complete planning report from a PlanningResult.
    """

    comparison = compare_planning_result(result)
    pareto = analyze_pareto(comparison)

    return PlanningReport(
        case_id=result.case_id,
        candidate_count=result.candidate_count,
        comparison=comparison,
        pareto=pareto,
        metadata=dict(result.metadata),
    )


def create_report(
    result: PlanningResult,
) -> PlanningReport:
    """Convenience alias for create_planning_report()."""

    return create_planning_report(result)