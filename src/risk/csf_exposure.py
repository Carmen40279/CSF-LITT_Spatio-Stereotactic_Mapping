"""ThermoSEEG CSF exposure assessment. It basically converts geometric measurements from the spatial
intersection layer into a transparent CSF exposure assessment"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.anatomy.csf import CSFCompartment
from src.geometry.spatial_intersection import (
    AblationCSFAnalysis,
    analyze_ablation_csf,
)
from src.trajectory.ablation import AblationGeometry

# exposure result

@dataclass(frozen=True)
class CSFExposureAssessment:

    minimum_distance: float

    ablation_volume: float

    volume_inside_csf: float

    fraction_inside_csf: float

    intersects_csf: bool

    exposure_fraction: float

    exposure_percentage: float

    exposure_level: str

    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        """Return the assessment as a serializable dictionary."""

        return {
            "minimum_distance": self.minimum_distance,
            "ablation_volume": self.ablation_volume,
            "volume_inside_csf": self.volume_inside_csf,
            "fraction_inside_csf": self.fraction_inside_csf,
            "intersects_csf": self.intersects_csf,
            "exposure_fraction": self.exposure_fraction,
            "exposure_percentage": self.exposure_percentage,
            "exposure_level": self.exposure_level,
            "interpretation": self.interpretation,
        }


# exposure classification


def classify_exposure(
    *,
    intersects_csf: bool,
    exposure_fraction: float,
    minimum_distance: float,
) -> tuple[str, str]:
    """
    Convert geometric measurements into a descriptive exposure
    category and interpretation."""

    if not intersects_csf:

        if minimum_distance <= 2.0:
            return (
                "NEAR",
                (
                    "The sampled ablation volume does not "
                    "intersect the modeled CSF compartment but "
                    "passes within the near-distance band."
                ),
            )

        return (
            "NONE",
            (
                "The sampled ablation volume does not "
                "intersect the modeled CSF compartment."
            ),
        )

    if exposure_fraction < 0.05:

        return (
            "LOW",
            (
                "A small fraction of the sampled ablation "
                "volume lies inside the modeled CSF compartment."
            ),
        )

    if exposure_fraction < 0.20:

        return (
            "MODERATE",
            (
                "A measurable fraction of the sampled ablation "
                "volume lies inside the modeled CSF compartment."
            ),
        )

    return (
        "HIGH",
        (
            "A substantial fraction of the sampled ablation "
            "volume lies inside the modeled CSF compartment."
        ),
    )

# calculations 
def calculate_csf_exposure(
    analysis: AblationCSFAnalysis,
) -> CSFExposureAssessment:
    
    exposure_fraction = float(
        np.clip(
            analysis.estimated_fraction_inside_csf,
            0.0,
            1.0,
        )
    )

    exposure_percentage = (
        exposure_fraction * 100.0
    )

    exposure_level, interpretation = (
        classify_exposure(
            intersects_csf=analysis.intersects,
            exposure_fraction=exposure_fraction,
            minimum_distance=analysis.minimum_distance,
        )
    )

    return CSFExposureAssessment(
        minimum_distance=float(
            analysis.minimum_distance
        ),
        ablation_volume=float(
            analysis.estimated_ablation_volume
        ),
        volume_inside_csf=float(
            analysis.estimated_volume_inside_csf
        ),
        fraction_inside_csf=float(
            analysis.fraction_inside_csf
        ),
        intersects_csf=bool(
            analysis.intersects
        ),
        exposure_fraction=exposure_fraction,
        exposure_percentage=exposure_percentage,
        exposure_level=exposure_level,
        interpretation=interpretation,
    )


# complete csf exppsure

def assess_csf_exposure(
    ablation: AblationGeometry,
    csf: CSFCompartment,
    *,
    axial_samples: int = 60,
    radial_samples: int = 12,
    angular_samples: int = 24,
) -> CSFExposureAssessment:
    """
    Run the complete geometry pipeline: LITT ablation> 
    spatial intersection> quantitative geometry>CSF exposure assessment"""

    spatial_analysis = analyze_ablation_csf(
        ablation,
        csf,
        axial_samples=axial_samples,
        radial_samples=radial_samples,
        angular_samples=angular_samples,
    )

    return calculate_csf_exposure(
        spatial_analysis
    )