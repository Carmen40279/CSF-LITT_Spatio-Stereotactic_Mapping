"""
ThermoSEEG application case manager.

Handles creation, loading, and saving of application-level cases.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.application.state import ApplicationState
from src.io.patient_case import create_patient_case


class CaseManager:
    """
    Controls the lifecycle of the currently open ThermoSEEG case.
    """

    def __init__(
        self,
        state: ApplicationState,
    ) -> None:

        self.state = state

    def new_case(
        self,
        case_id: str,
    ):
        """Create and activate a new patient case."""

        if not case_id:
            raise ValueError(
                "Case ID cannot be empty."
            )

        case = create_patient_case(
            case_id
        )

        self.state.set_case(
            case
        )

        return case

    def save_case(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save the structural case representation to JSON.

        Large imaging arrays remain external and are referenced
        through their ImagingStudy source paths.
        """

        if self.state.current_case is None:
            raise RuntimeError(
                "No patient case is currently open."
            )

        output_path = Path(
            path
        )

        case = self.state.current_case

        payload = {
            "case_id": case.case_id,
            "metadata": case.metadata,
            "studies": [
                {
                    "study_id": study.study_id,
                    "modality": study.modality,
                    "description": study.description,
                    "source": study.source,
                    "spacing": study.spacing,
                    "coordinate_system": (
                        study.coordinate_system
                    ),
                    "metadata": study.metadata,
                }
                for study in case.studies
            ],
            "coordinate_spaces": [
                {
                    "name": space.name,
                    "description": space.description,
                    "parent": space.parent,
                }
                for space in case.coordinate_spaces
            ],
            "surgery": {
                "seeg_electrodes": [
                    {
                        "electrode_id": (
                            electrode.electrode_id
                        ),
                        "contact_count": (
                            electrode.contact_count
                        ),
                        "contact_spacing": (
                            electrode.contact_spacing
                        ),
                        "coordinate_space": (
                            electrode.coordinate_space
                        ),
                    }
                    for electrode
                    in case.surgery.seeg_electrodes
                ],
                "litt_plans": [
                    {
                        "plan_id": plan.plan_id,
                        "active_length": (
                            plan.parameters.active_length
                        ),
                        "radius": (
                            plan.parameters.radius
                        ),
                        "coordinate_space": (
                            plan.coordinate_space
                        ),
                    }
                    for plan
                    in case.surgery.litt_plans
                ],
            },
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.state.case_path = str(
            output_path
        )

        self.state.dirty = False

        self.state.set_status(
            f"Case saved: {output_path.name}"
        )

        return output_path