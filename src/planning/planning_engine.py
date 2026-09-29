"""
Integrated ThermoSEEG planning engine.

This module provides a high-level interface for generating and evaluating"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.anatomy.anatomical_model import AnatomicalMask
from src.anatomy.csf import CSFCompartment
from src.anatomy.lesion import AnatomicalTarget
from src.geometry.vectors import Point3D
from src.io.patient_case import PatientCase
from src.planning.candidate_generation import (
    CandidateGenerationConfig,
    generate_litt_candidates,
)
from src.planning.evaluation import (
    CandidateEvaluation,
    evaluate_litt_trajectory,
)
from src.surgery.litt import LITTPlan


@dataclass
class PlanningResult:
    """Complete result of one ThermoSEEG planning run."""

    case_id: str
    candidate_evaluations: list[CandidateEvaluation] = field(
        default_factory=list
    )
    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_evaluations)

    @property
    def evaluated_plan_ids(self) -> list[str]:
        return [
            evaluation.plan_id
            for evaluation in self.candidate_evaluations
        ]

    @property
    def seeg_electrode_count(self) -> int:
        return int(
            self.metadata.get(
                "seeg_electrode_count",
                0,
            )
        )

    @property
    def uses_patient_csf(self) -> bool:
        return bool(
            self.metadata.get(
                "uses_patient_csf",
                False,
            )
        )


class PlanningEngine:
    """
    ThermoSEEG integrated planning engine.

    Candidate generation operates on Point3D coordinates, while
    candidate evaluation can retain the AnatomicalTarget object
    and its metadata.
    """

    def __init__(
        self,
        csf: CSFCompartment,
        target: AnatomicalTarget | None = None,
        uncertainty_enabled: bool = True,
        use_patient_csf: bool = True,
    ) -> None:
        self.csf = csf
        self.target = target
        self.uncertainty_enabled = uncertainty_enabled
        self.use_patient_csf = use_patient_csf

    def _get_patient_csf_mask(
        self,
        case: PatientCase,
    ) -> AnatomicalMask | None:
        """
        Retrieve the patient-specific CSF mask when available.
        """

        if not self.use_patient_csf:
            return None

        if case.anatomy is None:
            return None

        if not case.anatomy.has_mask("csf"):
            return None

        return case.anatomy.get_mask("csf")

    def _evaluate_plans(
        self,
        case: PatientCase,
        plans: list[LITTPlan],
        target: AnatomicalTarget | None,
    ) -> list[CandidateEvaluation]:
        """
        Evaluate every LITT plan against the same case anatomy.
        """

        patient_csf_mask = self._get_patient_csf_mask(
            case
        )

        evaluations: list[CandidateEvaluation] = []

        for plan in plans:
            evaluation = evaluate_litt_trajectory(
                litt_plan=plan,
                csf=self.csf,
                target=target,
                seeg_electrodes=case.surgery.seeg_electrodes,
                uncertainty_enabled=self.uncertainty_enabled,
                patient_csf_mask=patient_csf_mask,
            )

            evaluations.append(
                evaluation
            )

        return evaluations

    def evaluate_case(
        self,
        case: PatientCase,
    ) -> PlanningResult:
        """
        Evaluate LITT plans that already exist in the case.
        """

        target = self.target

        evaluations = self._evaluate_plans(
            case=case,
            plans=list(case.surgery.litt_plans),
            target=target,
        )

        patient_csf_mask = self._get_patient_csf_mask(
            case
        )

        return PlanningResult(
            case_id=case.case_id,
            candidate_evaluations=evaluations,
            metadata={
                "seeg_electrode_count": len(
                    case.surgery.seeg_electrodes
                ),
                "litt_plan_count": len(
                    case.surgery.litt_plans
                ),
                "uses_patient_csf": (
                    patient_csf_mask is not None
                ),
                "uncertainty_enabled": (
                    self.uncertainty_enabled
                ),
            },
        )

    def generate_and_evaluate(
        self,
        case: PatientCase,
        target: AnatomicalTarget,
        entry_center: Point3D,
        config: CandidateGenerationConfig,
        plan_id_prefix: str = "LITT-CANDIDATE",
    ) -> PlanningResult:
        """
        Generate LITT trajectories around an entry region and evaluate
        every candidate.

        AnatomicalTarget is deliberately converted to Point3D here
        before entering the geometry/candidate-generation layer.
        """

        if not isinstance(
            target,
            AnatomicalTarget,
        ):
            raise TypeError(
                "target must be an AnatomicalTarget."
            )

        if not isinstance(
            entry_center,
            Point3D,
        ):
            raise TypeError(
                "entry_center must be a Point3D."
            )

        target_point = target.point

        if not isinstance(
            target_point,
            Point3D,
        ):
            raise TypeError(
                "AnatomicalTarget.point must be a Point3D."
            )

       #operare point 3D

        candidates = generate_litt_candidates(
            target=target_point,
            entry_center=entry_center,
            config=config,
            plan_id_prefix=plan_id_prefix,
        )

        #storing candidates in the case
        for plan in candidates:
            existing_ids = {
                existing.plan_id
                for existing in case.surgery.litt_plans
            }

            if plan.plan_id not in existing_ids:
                case.add_litt_plan(plan)

        # candidates vs patient autonom
        evaluations = self._evaluate_plans(
            case=case,
            plans=candidates,
            target=target,
        )

        patient_csf_mask = self._get_patient_csf_mask(
            case
        )

        return PlanningResult(
            case_id=case.case_id,
            candidate_evaluations=evaluations,
            metadata={
                "seeg_electrode_count": len(
                    case.surgery.seeg_electrodes
                ),
                "litt_plan_count": len(
                    candidates
                ),
                "uses_patient_csf": (
                    patient_csf_mask is not None
                ),
                "uncertainty_enabled": (
                    self.uncertainty_enabled
                ),
                "target_name": target.name,
                "target_coordinate_space": (
                    target.coordinate_space
                ),
                "target_source": target.source,
            },
        )


def evaluate_case(
    case: PatientCase,
    csf: CSFCompartment,
    target: AnatomicalTarget | None = None,
    uncertainty_enabled: bool = True,
    use_patient_csf: bool = True,
) -> PlanningResult:
    """Convenience wrapper for evaluating an existing case."""

    engine = PlanningEngine(
        csf=csf,
        target=target,
        uncertainty_enabled=uncertainty_enabled,
        use_patient_csf=use_patient_csf,
    )

    return engine.evaluate_case(
        case
    )


def generate_and_evaluate(
    case: PatientCase,
    csf: CSFCompartment,
    target: AnatomicalTarget,
    entry_center: Point3D,
    config: CandidateGenerationConfig,
    uncertainty_enabled: bool = True,
    use_patient_csf: bool = True,
    plan_id_prefix: str = "LITT-CANDIDATE",
) -> PlanningResult:
    """Convenience wrapper for full candidate generation and evaluation."""

    engine = PlanningEngine(
        csf=csf,
        target=target,
        uncertainty_enabled=uncertainty_enabled,
        use_patient_csf=use_patient_csf,
    )

    return engine.generate_and_evaluate(
        case=case,
        target=target,
        entry_center=entry_center,
        config=config,
        plan_id_prefix=plan_id_prefix,
    )