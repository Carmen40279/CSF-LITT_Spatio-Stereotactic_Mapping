"""
ThermoSEEG end-to-end imaging and planning pipeline.

Connects:
    NIfTI ingestion
    -> imaging QC
    -> brain extraction
    -> CSF segmentation
    -> anatomical model
    -> brain surface
    -> LITT candidate generation
    -> scientific candidate evaluation
    -> Pareto analysis
    -> planning report

Current segmentation and thermal models are research prototypes and
are not clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.anatomy.anatomical_model import AnatomicalModel
from src.anatomy.brain_extraction import extract_brain_mask
from src.anatomy.csf_segmentation import segment_csf
from src.anatomy.lesion import AnatomicalTarget
from src.anatomy.qc import validate_anatomical_model
from src.anatomy.surface import create_brain_surface
from src.anatomy.csf import CSFCompartment
from src.geometry.vectors import Point3D
from src.imaging.nifti_loader import load_nifti
from src.imaging.qc import validate_imaging_study
from src.io.patient_case import PatientCase
from src.planning.candidate_generation import CandidateGenerationConfig
from src.planning.planning_engine import PlanningEngine
from src.planning.planning_report import PlanningReport, create_planning_report


@dataclass(frozen=True)
class ProcessingResult:
    """Results produced by processing one imaging case."""

    case_id: str
    source_path: str

    imaging_valid: bool
    anatomy_valid: bool

    image_shape: tuple[int, ...]
    voxel_spacing: tuple[float, ...]

    brain_voxel_count: int
    brain_volume_mm3: float

    csf_voxel_count: int
    csf_volume_mm3: float

    brain_surface_vertices: int
    brain_surface_faces: int

    target: Point3D

    candidate_count: int
    pareto_count: int

    planning_report: PlanningReport


def _calculate_mask_center(mask) -> Point3D:
    """Calculate the geometric center of an anatomical mask."""

    voxel_indices = np.argwhere(mask.mask > 0)

    if voxel_indices.size == 0:
        raise ValueError(
            f"Anatomical mask '{mask.name}' is empty."
        )

    homogeneous = np.column_stack(
        (
            voxel_indices,
            np.ones(voxel_indices.shape[0]),
        )
    )

    patient_coordinates = homogeneous @ mask.affine.T
    xyz = patient_coordinates[:, :3]

    center = xyz.mean(axis=0)

    return Point3D(
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def process_imaging_case(
    case: PatientCase,
    imaging_path: str | Path,
    candidate_config: CandidateGenerationConfig | None = None,
) -> ProcessingResult:
    """
    Process a NIfTI imaging study through the ThermoSEEG pipeline.

    The automatically generated target is the geometric center of the
    extracted brain mask. It exists only to provide a reproducible
    computational target for the demonstration pipeline.
    """

    path = Path(imaging_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Imaging file not found: {path}"
        )

    if path.suffix.lower() not in {".nii", ".gz"}:
        raise ValueError(
            "ThermoSEEG currently expects a NIfTI file (.nii or .nii.gz)."
        )

    # ---------------------------------------------------------
    # 1. LOAD IMAGING
    # ---------------------------------------------------------

    study = load_nifti(path)

    imaging_qc = validate_imaging_study(study)

    if not imaging_qc.valid:
        issues = "; ".join(imaging_qc.issues)
        raise ValueError(
            f"Imaging QC failed: {issues}"
        )

    case.imaging = study

    # ---------------------------------------------------------
    # 2. EXTRACT PATIENT-SPECIFIC ANATOMY
    # ---------------------------------------------------------

    brain_mask = extract_brain_mask(study)
    csf_mask = segment_csf(study)

    anatomy = AnatomicalModel(
        coordinate_space="patient",
        metadata={
            "source_imaging": str(path),
            "processing_pipeline": "ThermoSEEG",
            "processing_status": "completed",
        },
    )

    anatomy.add_mask(brain_mask)
    anatomy.add_mask(csf_mask)

    # ---------------------------------------------------------
    # 3. BUILD BRAIN SURFACE
    # ---------------------------------------------------------

    brain_surface = create_brain_surface(brain_mask)

    anatomy.add_surface(brain_surface)

    anatomy_qc = validate_anatomical_model(anatomy)

    if not anatomy_qc.valid:
        issues = "; ".join(anatomy_qc.issues)
        raise ValueError(
            f"Anatomical QC failed: {issues}"
        )

    case.set_anatomy(anatomy)

    # ---------------------------------------------------------
    # 4. CREATE REPRODUCIBLE DEMONSTRATION TARGET
    # ---------------------------------------------------------

    target_point = _calculate_mask_center(brain_mask)

    target = AnatomicalTarget(
        name="DEMO_BRAIN_CENTER_TARGET",
        point=target_point,
        coordinate_space="patient",
        source="automatic_brain_center",
    )

    # ---------------------------------------------------------
    # 5. CANDIDATE SEARCH CONFIGURATION
    # ---------------------------------------------------------

    if candidate_config is None:
        candidate_config = CandidateGenerationConfig(
            entry_spacing=3.0,
            search_radius=9.0,
            active_length=25.0,
            radius=4.0,
            max_candidates=81,
        )

    entry_center = Point3D(
        target_point.x + 25.0,
        target_point.y,
        target_point.z,
    )

    # ---------------------------------------------------------
    # 6. PLANNING ENGINE
    # ---------------------------------------------------------

    synthetic_csf = CSFCompartment(
        center=target_point,
        radii=(18.0, 18.0, 18.0),
        resolution=48,
    )

    engine = PlanningEngine(
        csf=synthetic_csf,
        target=target,
        uncertainty_enabled=True,
        use_patient_csf=True,
    )

    planning_result = engine.generate_and_evaluate(
        case=case,
        target=target,
        entry_center=entry_center,
        config=candidate_config,
        plan_id_prefix="MRI-CANDIDATE",
    )

    # ---------------------------------------------------------
    # 7. BUILD REPORT
    # ---------------------------------------------------------

    report = create_planning_report(planning_result)

    return ProcessingResult(
        case_id=case.case_id,
        source_path=str(path),
        imaging_valid=imaging_qc.valid,
        anatomy_valid=anatomy_qc.valid,
        image_shape=tuple(study.data.shape),
        voxel_spacing=tuple(float(x) for x in study.spacing),
        brain_voxel_count=brain_mask.voxel_count,
        brain_volume_mm3=brain_mask.volume_mm3,
        csf_voxel_count=csf_mask.voxel_count,
        csf_volume_mm3=csf_mask.volume_mm3,
        brain_surface_vertices=brain_surface.vertex_count,
        brain_surface_faces=brain_surface.face_count,
        target=target_point,
        candidate_count=planning_result.candidate_count,
        pareto_count=report.non_dominated_count,
        planning_report=report,
    )