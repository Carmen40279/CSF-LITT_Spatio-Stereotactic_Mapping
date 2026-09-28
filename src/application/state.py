"""
ThermoSEEG application state.

The application state sits between the permanent GUI and the
computational backend. GUI actions modify this state; backend
engines consume the current PatientCase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.io.patient_case import PatientCase


@dataclass
class ApplicationState:
    """
    Runtime state of the ThermoSEEG workstation.
    """

    current_case: PatientCase | None = None

    case_path: str | None = None

    dirty: bool = False

    status_message: str = "Ready."

    view_mode: str = "3D"

    selected_electrode_id: str | None = None

    selected_litt_id: str | None = None

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def set_case(
        self,
        case: PatientCase,
        path: str | None = None,
    ) -> None:
        """Set the active patient case."""

        self.current_case = case
        self.case_path = path
        self.dirty = False
        self.status_message = (
            f"Case loaded: {case.case_id}"
        )

    def mark_dirty(self) -> None:
        """Mark the active case as modified."""

        self.dirty = True

    def set_status(
        self,
        message: str,
    ) -> None:
        """Update the workstation status."""

        self.status_message = message