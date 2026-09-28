"""
ThermoSEEG patient-case abstraction.

This module provides the high-level case container that connects
medical imaging, patient-specific anatomy, surgical planning,
coordinate spaces, and the existing computational PlanningCase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.anatomy.anatomical_model import AnatomicalModel
from src.io.data_models import PlanningCase
from src.surgery.case import SurgicalPlan
from src.surgery.litt import LITTPlan
from src.surgery.seeg import SEEGElectrode


# =====================================================================
# IMAGING STUDY
# =====================================================================


@dataclass
class ImagingStudy:
    """
    Medical imaging study associated with a ThermoSEEG case.

    The actual image volume is stored in ``data`` when available.
    NIfTI and DICOM loaders can populate this model independently
    of the underlying file format.
    """

    study_id: str

    modality: str

    description: str = ""

    source: str | None = None

    data: Any = None

    affine: Any = None

    spacing: tuple[float, float, float] | None = None

    coordinate_system: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# COORDINATE SPACE
# =====================================================================


@dataclass
class CoordinateSpace:
    """
    Coordinate/reference space used by a patient case.
    """

    name: str

    description: str = ""

    parent: str | None = None

    affine_to_parent: Any = None


# =====================================================================
# PATIENT CASE
# =====================================================================


@dataclass
class PatientCase:
    """
    Complete patient-specific ThermoSEEG case.

    A PatientCase is the central object connecting:

    - medical imaging
    - anatomical models
    - SEEG planning
    - LITT planning
    - coordinate spaces
    - the legacy computational PlanningCase

    This allows ThermoSEEG to support multiple independent patient
    cases while keeping the existing geometry and risk engines
    compatible.
    """

    case_id: str

    studies: list[ImagingStudy] = field(
        default_factory=list
    )

    anatomy: AnatomicalModel = field(
        default_factory=AnatomicalModel
    )

    surgery: SurgicalPlan = field(
        default_factory=SurgicalPlan
    )

    coordinate_spaces: list[CoordinateSpace] = field(
        default_factory=list
    )

    planning_case: PlanningCase | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # =================================================================
    # IMAGING
    # =================================================================

    def add_study(
        self,
        study: ImagingStudy,
    ) -> None:
        """
        Add an imaging study to the case.
        """

        if not isinstance(
            study,
            ImagingStudy,
        ):
            raise TypeError(
                "study must be an ImagingStudy."
            )

        if not study.study_id:
            raise ValueError(
                "Imaging study must have a study_id."
            )

        if not study.modality:
            raise ValueError(
                "Imaging study must have a modality."
            )

        if any(
            existing.study_id == study.study_id
            for existing in self.studies
        ):
            raise ValueError(
                "Imaging study ID already exists: "
                f"{study.study_id}"
            )

        self.studies.append(
            study
        )

    def get_study(
        self,
        study_id: str,
    ) -> ImagingStudy:
        """
        Retrieve an imaging study by identifier.
        """

        for study in self.studies:

            if study.study_id == study_id:
                return study

        raise KeyError(
            f"Imaging study not found: {study_id}"
        )

    def get_studies_by_modality(
        self,
        modality: str,
    ) -> list[ImagingStudy]:
        """
        Return all imaging studies matching a modality.
        """

        if not modality:
            return []

        normalized = modality.upper()

        return [
            study
            for study in self.studies
            if study.modality.upper()
            == normalized
        ]

    # =================================================================
    # ANATOMY
    # =================================================================

    def set_anatomy(
        self,
        anatomical_model: AnatomicalModel,
    ) -> None:
        """
        Attach a patient-specific anatomical model.
        """

        if not isinstance(
            anatomical_model,
            AnatomicalModel,
        ):
            raise TypeError(
                "anatomical_model must be an AnatomicalModel."
            )

        self.anatomy = anatomical_model

    # =================================================================
    # COORDINATE SPACES
    # =================================================================

    def add_coordinate_space(
        self,
        space: CoordinateSpace,
    ) -> None:
        """
        Register a coordinate space with the patient case.
        """

        if not isinstance(
            space,
            CoordinateSpace,
        ):
            raise TypeError(
                "space must be a CoordinateSpace."
            )

        if not space.name:
            raise ValueError(
                "Coordinate space must have a name."
            )

        if any(
            existing.name == space.name
            for existing in self.coordinate_spaces
        ):
            raise ValueError(
                "Coordinate-space name already exists: "
                f"{space.name}"
            )

        self.coordinate_spaces.append(
            space
        )

    def get_coordinate_space(
        self,
        name: str,
    ) -> CoordinateSpace:
        """
        Retrieve a coordinate space by name.
        """

        for space in self.coordinate_spaces:

            if space.name == name:
                return space

        raise KeyError(
            f"Coordinate space not found: {name}"
        )

    # =================================================================
    # SEEG
    # =================================================================

    def add_seeg_plan(
        self,
        plan: SEEGElectrode,
    ) -> None:
        """
        Add an SEEG electrode to the surgical plan.
        """

        if not isinstance(
            plan,
            SEEGElectrode,
        ):
            raise TypeError(
                "SEEG plan must be a SEEGElectrode."
            )

        self.surgery.add_seeg_electrode(
            plan
        )

    def get_seeg_electrodes(
        self,
    ) -> list[SEEGElectrode]:
        """
        Return all planned SEEG electrodes.
        """

        return self.surgery.seeg_electrodes

    # =================================================================
    # LITT
    # =================================================================

    def add_litt_plan(
        self,
        plan: LITTPlan,
    ) -> None:
        """
        Add a LITT trajectory to the surgical plan.
        """

        if not isinstance(
            plan,
            LITTPlan,
        ):
            raise TypeError(
                "LITT plan must be a LITTPlan."
            )

        self.surgery.add_litt_plan(
            plan
        )

    def get_litt_plans(
        self,
    ) -> list[LITTPlan]:
        """
        Return all planned LITT trajectories.
        """

        return self.surgery.litt_plans

    # =================================================================
    # SUMMARY
    # =================================================================

    @property
    def study_count(self) -> int:
        """Number of imaging studies in the case."""

        return len(
            self.studies
        )

    @property
    def seeg_count(self) -> int:
        """Number of planned SEEG electrodes."""

        return len(
            self.surgery.seeg_electrodes
        )

    @property
    def litt_count(self) -> int:
        """Number of planned LITT trajectories."""

        return len(
            self.surgery.litt_plans
        )

    # =================================================================
    # VALIDATION
    # =================================================================

    def validate(self) -> None:
        """
        Validate the structural integrity of the case.

        This performs software/data-structure validation only.
        It does not determine clinical suitability.
        """

        if not self.case_id:
            raise ValueError(
                "PatientCase must have a case_id."
            )

        if not isinstance(
            self.anatomy,
            AnatomicalModel,
        ):
            raise TypeError(
                "PatientCase anatomy must be an AnatomicalModel."
            )

        if not isinstance(
            self.surgery,
            SurgicalPlan,
        ):
            raise TypeError(
                "PatientCase surgery must be a SurgicalPlan."
            )

        study_ids = [
            study.study_id
            for study in self.studies
        ]

        if len(study_ids) != len(
            set(study_ids)
        ):
            raise ValueError(
                "Imaging study IDs must be unique."
            )

        coordinate_names = [
            space.name
            for space in self.coordinate_spaces
        ]

        if len(coordinate_names) != len(
            set(coordinate_names)
        ):
            raise ValueError(
                "Coordinate-space names must be unique."
            )

        electrode_ids = [
            electrode.electrode_id
            for electrode in self.surgery.seeg_electrodes
        ]

        if len(electrode_ids) != len(
            set(electrode_ids)
        ):
            raise ValueError(
                "SEEG electrode IDs must be unique."
            )

        litt_ids = [
            plan.plan_id
            for plan in self.surgery.litt_plans
        ]

        if len(litt_ids) != len(
            set(litt_ids)
        ):
            raise ValueError(
                "LITT plan IDs must be unique."
            )


# =====================================================================
# FACTORY
# =====================================================================


def create_patient_case(
    case_id: str,
    planning_case: PlanningCase | None = None,
) -> PatientCase:
    """
    Create a new ThermoSEEG PatientCase.

    An existing PlanningCase can be supplied so the current
    computational planning system can coexist with the newer
    patient-case architecture.
    """

    if not case_id:
        raise ValueError(
            "Patient case must have a case_id."
        )

    case = PatientCase(
        case_id=case_id,
        planning_case=planning_case,
    )

    case.add_coordinate_space(
        CoordinateSpace(
            name="patient",
            description=(
                "Canonical patient-specific coordinate space."
            ),
        )
    )

    return case