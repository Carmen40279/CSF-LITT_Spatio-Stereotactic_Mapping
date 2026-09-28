from __future__ import annotations

from src.io.patient_case import PatientCase
from src.surgery.case import SurgicalPlan
from src.surgery.validation import (
    SurgicalQC,
    validate_surgical_qc,
)

def get_surgical_plan(
    case: PatientCase,
) -> SurgicalPlan:
    """
    Return the surgical planning object associated with a case.
    """

    if not isinstance(
        case.surgery,
        SurgicalPlan,
    ):
        raise TypeError(
            "PatientCase does not contain a SurgicalPlan."
        )

    return case.surgery


def validate_patient_surgical_plan(
    case: PatientCase,
) -> SurgicalQC:
    """
    Run surgical QC directly on a PatientCase.
    """

    plan = get_surgical_plan(
        case
    )

    return validate_surgical_qc(
        plan
    )