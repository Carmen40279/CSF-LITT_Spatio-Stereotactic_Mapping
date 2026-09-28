"""
Scientific evaluation of a single candidate LITT trajectory.

This module integrates ThermoSEEG geometry, CSF exposure, thermal-field,
uncertainty, and SEEG calculations into one structured result.

Both synthetic CSF geometry and patient-specific segmented CSF masks
are supported.

All outputs are computational prototype metrics and are not clinically
validated predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.anatomy.anatomical_model import AnatomicalMask
from src.anatomy.csf import CSFCompartment
from src.anatomy.lesion import AnatomicalTarget
from src.geometry.trajectory_perturbation import (
    generate_trajectory_perturbations,
)
from src.geometry.vectors import Point3D
from src.risk.csf_exposure import (
    CSFExposureAssessment,
    assess_csf_exposure,
)
from src.risk.mask_csf_exposure import (
    MaskCSFExposure,
    calculate_mask_csf_exposure,
)
from src.risk.thermal_field import (
    ThermalField,
    calculate_thermal_field,
)
from src.risk.uncertainty import (
    UncertaintyAssessment,
    analyze_trajectory_uncertainty,
)
from src.surgery.litt import LITTPlan
from src.surgery.seeg import SEEGElectrode
from src.trajectory.ablation import AblationGeometry
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class TargetEvaluation:
    """Geometric evaluation of the LITT trajectory relative to its target."""

    target_distance: float
    reaches_target: bool


@dataclass(frozen=True)
class SEEGEvaluation:
    """Spatial relationship between a LITT trajectory and a SEEG electrode."""

    electrode_id: str
    minimum_distance: float


@dataclass(frozen=True)
class CandidateEvaluation:
    """
    Combined computational evaluation of one LITT candidate.

    The values describe the current ThermoSEEG prototype model and should
    not be interpreted as clinically validated risk predictions.
    """

    plan_id: str
    trajectory_length: float
    target: TargetEvaluation

    csf: Optional[CSFExposureAssessment]
    patient_csf: Optional[MaskCSFExposure]

    thermal: Optional[ThermalField]
    uncertainty: Optional[UncertaintyAssessment]

    seeg_relationships: tuple[SEEGEvaluation, ...]

    @property
    def minimum_csf_distance(self) -> float:
        """Return the available minimum distance to CSF."""

        if self.patient_csf is not None:
            return self.patient_csf.minimum_distance

        if self.csf is not None:
            return self.csf.minimum_distance

        return float("nan")

    @property
    def csf_exposure_fraction(self) -> float:
        """Return the available CSF exposure fraction."""

        if self.patient_csf is not None:
            return self.patient_csf.exposure_fraction

        if self.csf is not None:
            return self.csf.exposure_fraction

        return float("nan")

    @property
    def maximum_thermal_risk(self) -> float:
        """Return the maximum normalized thermal-risk value."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_risk

    @property
    def maximum_temperature(self) -> float:
        """Return the maximum temperature with the CSF model."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_temperature

    @property
    def maximum_temperature_without_csf(self) -> float:
        """Return the maximum temperature without the CSF heat sink."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_temperature_without_csf

    @property
    def maximum_csf_temperature_reduction(self) -> float:
        """Return the maximum modeled CSF-induced temperature reduction."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_csf_temperature_reduction

    @property
    def maximum_thermal_dose(self) -> float:
        """Return the maximum CEM43 thermal dose with CSF."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_thermal_dose

    @property
    def maximum_thermal_dose_without_csf(self) -> float:
        """Return the maximum CEM43 thermal dose without CSF."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.maximum_thermal_dose_without_csf

    @property
    def csf_thermal_dose_reduction(self) -> float:
        """Return the modeled reduction in maximum CEM43 dose."""

        if self.thermal is None:
            return float("nan")

        return self.thermal.csf_thermal_dose_reduction


# ----------------------------------------------------------------------
# Coordinate normalization
# ----------------------------------------------------------------------


def _as_point3d(
    value: object,
    name: str,
) -> Point3D:
    """
    Convert a coordinate-like object into the ThermoSEEG Point3D type.

    Accepted inputs:
        Point3D
        (x, y, z)
        [x, y, z]
        NumPy arrays containing exactly three values
        objects exposing x/y/z attributes
    """

    if isinstance(value, Point3D):
        return value

    if (
        hasattr(value, "x")
        and hasattr(value, "y")
        and hasattr(value, "z")
    ):
        return Point3D(
            float(value.x),
            float(value.y),
            float(value.z),
        )

    try:
        coordinates = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must contain three spatial coordinates."
        ) from exc

    if len(coordinates) != 3:
        raise TypeError(
            f"{name} must contain exactly three coordinates."
        )

    try:
        return Point3D(
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2]),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} contains non-numeric coordinates."
        ) from exc


def _normalize_trajectory(
    trajectory: Trajectory,
    name: str = "trajectory",
) -> Trajectory:
    """Rebuild a trajectory using guaranteed Point3D endpoints."""

    if not isinstance(trajectory, Trajectory):
        raise TypeError(
            f"{name} must be a Trajectory."
        )

    entry = _as_point3d(
        trajectory.entry,
        f"{name}.entry",
    )

    target = _as_point3d(
        trajectory.target,
        f"{name}.target",
    )

    return Trajectory(
        entry=entry,
        target=target,
    )


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def _distance_between_trajectories(
    first: Trajectory,
    second: Trajectory,
    samples: int = 100,
) -> float:
    """Estimate the minimum distance between two finite trajectories."""

    if samples < 2:
        raise ValueError(
            "samples must be at least 2."
        )

    first = _normalize_trajectory(
        first,
        "first trajectory",
    )

    second = _normalize_trajectory(
        second,
        "second trajectory",
    )

    first_points = [
        first.point_at(
            first.length * index / (samples - 1)
        )
        for index in range(samples)
    ]

    second_points = [
        second.point_at(
            second.length * index / (samples - 1)
        )
        for index in range(samples)
    ]

    minimum_distance = float("inf")

    for first_point in first_points:
        first_point = _as_point3d(
            first_point,
            "first trajectory sample",
        )

        for second_point in second_points:
            second_point = _as_point3d(
                second_point,
                "second trajectory sample",
            )

            dx = first_point.x - second_point.x
            dy = first_point.y - second_point.y
            dz = first_point.z - second_point.z

            distance = (
                dx * dx
                + dy * dy
                + dz * dz
            ) ** 0.5

            if distance < minimum_distance:
                minimum_distance = distance

    return minimum_distance


def _evaluate_target(
    trajectory: Trajectory,
    target: Optional[AnatomicalTarget],
) -> TargetEvaluation:
    """Evaluate the trajectory endpoint relative to an optional target."""

    if target is None:
        return TargetEvaluation(
            target_distance=float("nan"),
            reaches_target=False,
        )

    normalized_trajectory = _normalize_trajectory(
        trajectory,
        "trajectory",
    )

    target_point = _as_point3d(
        target.point,
        "target.point",
    )

    endpoint = _as_point3d(
        normalized_trajectory.target,
        "trajectory.target",
    )

    dx = endpoint.x - target_point.x
    dy = endpoint.y - target_point.y
    dz = endpoint.z - target_point.z

    distance = (
        dx * dx
        + dy * dy
        + dz * dz
    ) ** 0.5

    tolerance = 1e-9

    return TargetEvaluation(
        target_distance=distance,
        reaches_target=distance <= tolerance,
    )


# ----------------------------------------------------------------------
# Candidate evaluation
# ----------------------------------------------------------------------


def evaluate_litt_trajectory(
    litt_plan: LITTPlan,
    csf: CSFCompartment,
    target: Optional[AnatomicalTarget] = None,
    seeg_electrodes: Optional[list[SEEGElectrode]] = None,
    uncertainty_enabled: bool = True,
    patient_csf_mask: Optional[AnatomicalMask] = None,
) -> CandidateEvaluation:
    """
    Evaluate one LITT trajectory.

    Synthetic CSF remains available through ``csf``.

    When ``patient_csf_mask`` is supplied:

        MRI CSF mask
            ↓
        thermal grid
            ↓
        patient-specific CSF heat-sink model

    Patient-specific geometric CSF exposure is also calculated separately.
    """

    if not isinstance(litt_plan, LITTPlan):
        raise TypeError(
            "litt_plan must be a LITTPlan."
        )

    if not isinstance(csf, CSFCompartment):
        raise TypeError(
            "csf must be a CSFCompartment."
        )

    if patient_csf_mask is not None and not isinstance(
        patient_csf_mask,
        AnatomicalMask,
    ):
        raise TypeError(
            "patient_csf_mask must be an AnatomicalMask or None."
        )

    if seeg_electrodes is None:
        seeg_electrodes = []

    # --------------------------------------------------------------
    # Normalize LITT trajectory.
    # --------------------------------------------------------------

    normalized_litt_trajectory = _normalize_trajectory(
        litt_plan.trajectory,
        "LITT trajectory",
    )

    normalized_litt_plan = LITTPlan(
        plan_id=litt_plan.plan_id,
        trajectory=normalized_litt_trajectory,
        parameters=litt_plan.parameters,
        coordinate_space=litt_plan.coordinate_space,
        metadata=dict(litt_plan.metadata),
    )

    # --------------------------------------------------------------
    # LITT ablation geometry.
    # --------------------------------------------------------------

    ablation = AblationGeometry(
        trajectory=normalized_litt_plan.trajectory,
        active_length=normalized_litt_plan.parameters.active_length,
        radius=normalized_litt_plan.parameters.radius,
    )

    # --------------------------------------------------------------
    # Synthetic CSF geometric exposure.
    # --------------------------------------------------------------

    synthetic_csf_assessment = assess_csf_exposure(
        ablation,
        csf,
    )

    # --------------------------------------------------------------
    # Thermal field.
    #
    # IMPORTANT:
    #
    # If patient_csf_mask exists, the thermal engine now uses the
    # MRI-derived CSF mask instead of using the synthetic ellipsoid
    # for the thermal simulation.
    #
    # The synthetic CSF object remains available as a fallback.
    # --------------------------------------------------------------

    thermal_field = calculate_thermal_field(
        ablation,
        csf,
        patient_csf_mask=patient_csf_mask,
    )

    # --------------------------------------------------------------
    # Patient-specific CSF geometric exposure.
    # --------------------------------------------------------------

    patient_csf_assessment: Optional[MaskCSFExposure] = None

    if patient_csf_mask is not None:
        patient_csf_assessment = calculate_mask_csf_exposure(
            ablation,
            patient_csf_mask,
        )

    # --------------------------------------------------------------
    # Trajectory uncertainty.
    #
    # The current uncertainty engine evaluates perturbations against
    # the synthetic CSF compartment.
    #
    # Patient-specific uncertainty analysis can be upgraded separately
    # after the primary thermal pathway is verified.
    # --------------------------------------------------------------

    uncertainty_assessment: Optional[
        UncertaintyAssessment
    ] = None

    if uncertainty_enabled:
        perturbations = generate_trajectory_perturbations(
            trajectory=normalized_litt_plan.trajectory,
            translation_step=1.0,
            rotation_step_degrees=2.0,
        )

        uncertainty_assessment = analyze_trajectory_uncertainty(
            perturbations=perturbations,
            csf=csf,
            active_length=normalized_litt_plan.parameters.active_length,
            radius=normalized_litt_plan.parameters.radius,
        )

    # --------------------------------------------------------------
    # SEEG relationships.
    # --------------------------------------------------------------

    relationships: list[SEEGEvaluation] = []

    for electrode in seeg_electrodes:
        normalized_seeg_trajectory = _normalize_trajectory(
            electrode.trajectory,
            f"SEEG electrode '{electrode.electrode_id}' trajectory",
        )

        relationships.append(
            SEEGEvaluation(
                electrode_id=electrode.electrode_id,
                minimum_distance=_distance_between_trajectories(
                    normalized_litt_plan.trajectory,
                    normalized_seeg_trajectory,
                ),
            )
        )

    # --------------------------------------------------------------
    # Final structured result.
    # --------------------------------------------------------------

    return CandidateEvaluation(
        plan_id=normalized_litt_plan.plan_id,
        trajectory_length=normalized_litt_plan.trajectory.length,
        target=_evaluate_target(
            normalized_litt_plan.trajectory,
            target,
        ),
        csf=synthetic_csf_assessment,
        patient_csf=patient_csf_assessment,
        thermal=thermal_field,
        uncertainty=uncertainty_assessment,
        seeg_relationships=tuple(
            relationships
        ),
    )