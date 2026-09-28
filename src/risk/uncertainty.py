"""Pipeline: perturbed trajectory; ablationGeometry; ablation volume to CSF analysis; CSF exposure assessment; 
    uncertainty statistics """

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.anatomy.csf import CSFCompartment
from src.geometry.spatial_intersection import analyze_ablation_csf
from src.geometry.trajectory_perturbation import (
    TrajectoryPerturbation,
)
from src.risk.csf_exposure import assess_csf_exposure
from src.trajectory.ablation import AblationGeometry

# individual perturbation
@dataclass(frozen=True)
class PerturbationResult:
    """
    Geometric outcome for one perturbed LITT trajectory.
    """

    perturbation: TrajectoryPerturbation

    minimum_distance: float

    exposure_fraction: float
    exposure_percentage: float

    intersects_csf: bool

    ablation_volume: float
    volume_inside_csf: float

    exposure_level: str

    def as_dict(self) -> dict:
        """
        Return a JSON-friendly summary.
        """

        return {
            "label": self.perturbation.label,
            "translation": list(
                self.perturbation.translation
            ),
            "rotation_degrees": (
                self.perturbation.rotation_degrees
            ),
            "minimum_distance": self.minimum_distance,
            "exposure_fraction": self.exposure_fraction,
            "exposure_percentage": self.exposure_percentage,
            "intersects_csf": self.intersects_csf,
            "ablation_volume": self.ablation_volume,
            "volume_inside_csf": self.volume_inside_csf,
            "exposure_level": self.exposure_level,
        }


# overall uncertainty

@dataclass(frozen=True)
class UncertaintyAssessment:
    """
    Aggregate uncertainty statistics across all perturbed
    trajectories.
    """

    results: tuple[PerturbationResult, ...]

    minimum_distance_mean: float
    minimum_distance_std: float
    minimum_distance_min: float
    minimum_distance_max: float

    exposure_fraction_mean: float
    exposure_fraction_std: float
    exposure_fraction_min: float
    exposure_fraction_max: float

    intersection_probability: float

    exposure_level_distribution: dict[str, int]

    most_exposed_perturbation: PerturbationResult
    least_exposed_perturbation: PerturbationResult

    @property
    def number_of_trajectories(self) -> int:
        """
        Number of trajectories evaluated.
        """

        return len(self.results)

    @property
    def intersection_percentage(self) -> float:
        """
        Percentage of perturbations whose ablation volume
        intersects the CSF compartment.
        """

        return self.intersection_probability * 100.0

    @property
    def exposure_percentage_mean(self) -> float:
        """
        Mean CSF exposure percentage.
        """

        return self.exposure_fraction_mean * 100.0

    @property
    def exposure_percentage_std(self) -> float:
        """
        Standard deviation of CSF exposure percentage.
        """

        return self.exposure_fraction_std * 100.0

    @property
    def most_exposed_label(self) -> str:
        """
        Label of the perturbation with the greatest
        estimated CSF exposure.
        """

        return self.most_exposed_perturbation.perturbation.label

    @property
    def least_exposed_label(self) -> str:
        """
        Label of the perturbation with the smallest
        estimated CSF exposure.
        """

        return self.least_exposed_perturbation.perturbation.label

    def as_dict(self) -> dict:
        """
        Return an overall JSON-friendly summary.
        """

        return {
            "number_of_trajectories": (
                self.number_of_trajectories
            ),
            "minimum_distance_mean": (
                self.minimum_distance_mean
            ),
            "minimum_distance_std": (
                self.minimum_distance_std
            ),
            "minimum_distance_min": (
                self.minimum_distance_min
            ),
            "minimum_distance_max": (
                self.minimum_distance_max
            ),
            "exposure_fraction_mean": (
                self.exposure_fraction_mean
            ),
            "exposure_fraction_std": (
                self.exposure_fraction_std
            ),
            "exposure_fraction_min": (
                self.exposure_fraction_min
            ),
            "exposure_fraction_max": (
                self.exposure_fraction_max
            ),
            "intersection_probability": (
                self.intersection_probability
            ),
            "intersection_percentage": (
                self.intersection_percentage
            ),
            "exposure_percentage_mean": (
                self.exposure_percentage_mean
            ),
            "exposure_percentage_std": (
                self.exposure_percentage_std
            ),
            "exposure_level_distribution": (
                self.exposure_level_distribution.copy()
            ),
            "most_exposed_perturbation": (
                self.most_exposed_perturbation.as_dict()
            ),
            "least_exposed_perturbation": (
                self.least_exposed_perturbation.as_dict()
            ),
        }

def analyze_perturbed_trajectory(
    perturbation: TrajectoryPerturbation,
    csf: CSFCompartment,
    active_length: float,
    radius: float,
    axial_samples: int = 40,
    radial_samples: int = 10,
    angular_samples: int = 20,
) -> PerturbationResult:
    """
    Analyse one perturbed LITT trajectory.

    The trajectory is converted into a real 3D ablation geometry.
    The complete ablation volume is then compared against the CSF
    compartment.

    Parameters
    ----------
    perturbation:
        One trajectory generated by trajectory_perturbation.py.

    csf:
        CSF compartment.

    active_length:
        Active LITT ablation length.

    radius:
        Ablation radius.

    axial_samples:
        Number of samples along the ablation axis.

    radial_samples:
        Number of radial samples.

    angular_samples:
        Number of angular samples.
    """

    if active_length <= 0:
        raise ValueError(
            "active_length must be greater than zero."
        )

    if radius <= 0:
        raise ValueError(
            "radius must be greater than zero."
        )

    # 3D ablation geomtery 

    ablation = AblationGeometry(
        trajectory=perturbation.trajectory,
        active_length=active_length,
        radius=radius,
    )

    # complete ablation volume vs CSF
    spatial_analysis = analyze_ablation_csf(
        ablation,
        csf,
        axial_samples=axial_samples,
        radial_samples=radial_samples,
        angular_samples=angular_samples,
    )

    exposure = assess_csf_exposure(
        ablation,
        csf,
    )


    # Clamp the fraction defensively.
    # A sampled geometric estimate should theoretically fall inside
    # [0, 1]. Floating-point/sampling effects should never allow
    # impossible values into the uncertainty summary.

    exposure_fraction = float(
        np.clip(
            exposure.exposure_fraction,
            0.0,
            1.0,
        )
    )

    exposure_percentage = exposure_fraction * 100.0

    return PerturbationResult(
        perturbation=perturbation,

        minimum_distance=float(
            spatial_analysis.minimum_distance
        ),

        exposure_fraction=exposure_fraction,

        exposure_percentage=exposure_percentage,

        intersects_csf=bool(
            spatial_analysis.intersects
        ),

        ablation_volume=float(
            spatial_analysis.estimated_ablation_volume
        ),

        volume_inside_csf=float(
            spatial_analysis.estimated_volume_inside_csf
        ),

        exposure_level=str(
            exposure.exposure_level
        ),
    )

def analyze_trajectory_uncertainty(
    perturbations: list[TrajectoryPerturbation],
    csf: CSFCompartment,
    active_length: float,
    radius: float,
    axial_samples: int = 40,
    radial_samples: int = 10,
    angular_samples: int = 20,
) -> UncertaintyAssessment:
    """
    Analyse a collection of perturbed trajectories.

    Each perturbation is passed through the complete:

        trajectory
        → ablation volume
        → CSF intersection
        → CSF exposure

    pipeline.

    The resulting geometric measurements are then summarized
    statistically.
    """

    if not perturbations:
        raise ValueError(
            "At least one trajectory perturbation is required."
        )

    if active_length <= 0:
        raise ValueError(
            "active_length must be greater than zero."
        )

    if radius <= 0:
        raise ValueError(
            "radius must be greater than zero."
        )


    results: list[PerturbationResult] = []

    for perturbation in perturbations:

        result = analyze_perturbed_trajectory(
            perturbation=perturbation,
            csf=csf,
            active_length=active_length,
            radius=radius,
            axial_samples=axial_samples,
            radial_samples=radial_samples,
            angular_samples=angular_samples,
        )

        results.append(result)

    if not results:
        raise RuntimeError(
            "No perturbation results were produced."
        )

    # Convert measurements into arrays
    distances = np.asarray(
        [
            result.minimum_distance
            for result in results
        ],
        dtype=float,
    )

    exposures = np.asarray(
        [
            result.exposure_fraction
            for result in results
        ],
        dtype=float,
    )

    intersections = np.asarray(
        [
            result.intersects_csf
            for result in results
        ],
        dtype=bool,
    )
    # Exposure-level distribution

    exposure_level_distribution: dict[str, int] = {}

    for result in results:

        level = result.exposure_level

        exposure_level_distribution[level] = (
            exposure_level_distribution.get(level, 0)
            + 1
        )
    # Most / least exposed trajectories

    most_exposed = max(
        results,
        key=lambda result: result.exposure_fraction,
    )

    least_exposed = min(
        results,
        key=lambda result: result.exposure_fraction,
    )


    intersection_probability = float(
        np.mean(intersections)
    )

    return UncertaintyAssessment(
        results=tuple(results),

        minimum_distance_mean=float(
            np.mean(distances)
        ),

        minimum_distance_std=float(
            np.std(distances)
        ),

        minimum_distance_min=float(
            np.min(distances)
        ),

        minimum_distance_max=float(
            np.max(distances)
        ),

        exposure_fraction_mean=float(
            np.mean(exposures)
        ),

        exposure_fraction_std=float(
            np.std(exposures)
        ),

        exposure_fraction_min=float(
            np.min(exposures)
        ),

        exposure_fraction_max=float(
            np.max(exposures)
        ),

        intersection_probability=intersection_probability,

        exposure_level_distribution=(
            exposure_level_distribution
        ),

        most_exposed_perturbation=most_exposed,

        least_exposed_perturbation=least_exposed,
    )