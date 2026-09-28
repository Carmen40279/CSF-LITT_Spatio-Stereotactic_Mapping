from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.surgery.case import SurgicalPlan
from src.surgery.seeg import SEEGElectrode

# seeg validation

def validate_seeg_electrode(
    electrode: SEEGElectrode,
) -> list[str]:
    """
    Return structural validation errors for an SEEG electrode.
    """

    issues: list[str] = []

    if not electrode.electrode_id:
        issues.append(
            "SEEG electrode has no identifier."
        )

    if electrode.contact_count < 1:
        issues.append(
            f"{electrode.electrode_id}: "
            "contact count must be positive."
        )

    if electrode.contact_spacing <= 0:
        issues.append(
            f"{electrode.electrode_id}: "
            "contact spacing must be positive."
        )

    points = np.array(
        [
            [
                electrode.entry.x,
                electrode.entry.y,
                electrode.entry.z,
            ],
            [
                electrode.target.x,
                electrode.target.y,
                electrode.target.z,
            ],
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(points)
    ):
        issues.append(
            f"{electrode.electrode_id}: "
            "coordinates contain non-finite values."
        )

    if electrode.trajectory_length <= 0:
        issues.append(
            f"{electrode.electrode_id}: "
            "trajectory length must be positive."
        )

    required_length = (
        (electrode.contact_count - 1)
        * electrode.contact_spacing
    )

    if required_length > electrode.trajectory_length:
        issues.append(
            f"{electrode.electrode_id}: "
            "contacts exceed trajectory length."
        )

    # Validate generated contacts when they already exist.
    if electrode.contacts:

        if len(electrode.contacts) != electrode.contact_count:
            issues.append(
                f"{electrode.electrode_id}: "
                "generated contact count does not match "
                "declared contact count."
            )

        for contact in electrode.contacts:

            coordinates = np.array(
                contact.coordinates,
                dtype=float,
            )

            if not np.all(
                np.isfinite(coordinates)
            ):
                issues.append(
                    f"{electrode.electrode_id}: "
                    f"contact {contact.index} contains "
                    "non-finite coordinates."
                )

            if contact.distance_from_entry < 0:
                issues.append(
                    f"{electrode.electrode_id}: "
                    f"contact {contact.index} has a negative "
                    "distance from entry."
                )

            if (
                contact.distance_from_entry
                > electrode.trajectory_length
            ):
                issues.append(
                    f"{electrode.electrode_id}: "
                    f"contact {contact.index} lies beyond "
                    "the trajectory."
                )

    return issues

# surgical plan validation

def validate_surgical_plan(
    plan: SurgicalPlan,
) -> list[str]:
    """
    Validate all surgical plans in a case.
    """

    issues: list[str] = []

    electrode_ids: set[str] = set()

    for electrode in plan.seeg_electrodes:

        if electrode.electrode_id in electrode_ids:
            issues.append(
                "Duplicate SEEG electrode ID: "
                f"{electrode.electrode_id}"
            )

        electrode_ids.add(
            electrode.electrode_id
        )

        issues.extend(
            validate_seeg_electrode(
                electrode
            )
        )

    litt_ids: set[str] = set()

    for litt in plan.litt_plans:

        if not litt.plan_id:
            issues.append(
                "LITT plan has no identifier."
            )

        if litt.plan_id in litt_ids:
            issues.append(
                "Duplicate LITT plan ID: "
                f"{litt.plan_id}"
            )

        litt_ids.add(
            litt.plan_id
        )

        if litt.trajectory_length <= 0:
            issues.append(
                f"{litt.plan_id}: "
                "trajectory length must be positive."
            )

        points = np.array(
            [
                [
                    litt.entry.x,
                    litt.entry.y,
                    litt.entry.z,
                ],
                [
                    litt.target.x,
                    litt.target.y,
                    litt.target.z,
                ],
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(points)
        ):
            issues.append(
                f"{litt.plan_id}: "
                "coordinates contain non-finite values."
            )

    return issues


# SURGICAL QC

@dataclass(frozen=True)
class SurgicalQC:
    """
    Structural quality-control result for surgical planning data.
    """

    valid: bool

    issues: tuple[str, ...]

    electrode_count: int

    litt_count: int

    contact_count: int


def validate_surgical_qc(
    plan: SurgicalPlan,
) -> SurgicalQC:
    """
    Run structural QC on a complete surgical plan.
    """

    issues = validate_surgical_plan(
        plan
    )

    contact_count = sum(
        electrode.contact_count
        for electrode in plan.seeg_electrodes
    )

    return SurgicalQC(
        valid=len(issues) == 0,
        issues=tuple(issues),
        electrode_count=(
            plan.electrode_count
        ),
        litt_count=(
            plan.litt_count
        ),
        contact_count=contact_count,
    )