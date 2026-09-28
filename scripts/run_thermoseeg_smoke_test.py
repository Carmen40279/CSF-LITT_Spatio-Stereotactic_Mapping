"""
ThermoSEEG integration smoke test.

Tests the current geometric pipeline:

    LITT trajectory
        ↓
    trajectory perturbations
        ↓
    LITT ablation geometry
        ↓
    ablation volume ↔ CSF
        ↓
    CSF exposure
        ↓
    uncertainty summary

This is a development validation script.
It is not a clinically validated tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# =====================================================================
# REPOSITORY PATH
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =====================================================================
# THERMOSEEG IMPORTS
# =====================================================================

from src.anatomy.csf import CSFCompartment
from src.geometry.trajectory_perturbation import (
    generate_trajectory_perturbations,
)
from src.geometry.vectors import Point3D
from src.risk.uncertainty import (
    analyze_trajectory_uncertainty,
)
from src.trajectory.ablation import AblationGeometry
from src.trajectory.trajectory import Trajectory


# =====================================================================
# HELPERS
# =====================================================================


def make_point(
    x: float,
    y: float,
    z: float,
) -> Point3D:
    """
    Create a ThermoSEEG Point3D.

    The production geometry layer uses Point3D objects with
    .x, .y, and .z attributes.
    """

    return Point3D(
        float(x),
        float(y),
        float(z),
    )


def point_array(point: Point3D) -> np.ndarray:
    """
    Convert a ThermoSEEG Point3D into a NumPy array.

    NumPy is used only for numerical validation inside this
    smoke test; the geometry pipeline itself uses Point3D.
    """

    return np.array(
        [
            point.x,
            point.y,
            point.z,
        ],
        dtype=float,
    )


def trajectory_length(
    trajectory: Trajectory,
) -> float:
    """
    Calculate the Euclidean length of a trajectory.
    """

    entry = point_array(
        trajectory.entry
    )

    target = point_array(
        trajectory.target
    )

    return float(
        np.linalg.norm(
            target - entry
        )
    )


def section(title: str) -> None:
    """
    Print a terminal section heading.
    """

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# MAIN
# =====================================================================


def main() -> int:

    section("THERMOSEEG SMOKE TEST")

    print("Repository:")
    print(REPO_ROOT)

    print()
    print("Testing:")
    print("  Trajectory")
    print("  Perturbation generation")
    print("  Ablation geometry")
    print("  CSF geometry")
    print("  CSF exposure")
    print("  Uncertainty propagation")


    # =================================================================
    # 1. CREATE LITT TRAJECTORY
    # =================================================================

    section("1. LITT TRAJECTORY")

    litt_entry = make_point(
        -20.0,
        -10.0,
        -15.0,
    )

    litt_target = make_point(
        20.0,
        10.0,
        15.0,
    )

    print(
        "Entry :",
        point_array(litt_entry),
    )

    print(
        "Target:",
        point_array(litt_target),
    )

    try:

        litt_trajectory = Trajectory(
            entry=litt_entry,
            target=litt_target,
        )

    except Exception as exc:

        print()
        print("ERROR creating Trajectory.")
        print()
        print("Exception:")
        print(exc)

        return 1

    original_length = trajectory_length(
        litt_trajectory
    )

    print()
    print(
        f"Trajectory length: "
        f"{original_length:.3f}"
    )

    if original_length <= 0:

        print(
            "[FAIL] Trajectory has zero length."
        )

        return 1

    print(
        "[PASS] LITT trajectory created."
    )


    # =================================================================
    # 2. CREATE ABLATION GEOMETRY
    # =================================================================

    section("2. LITT ABLATION GEOMETRY")

    active_length = 25.0
    radius = 4.0

    print(
        f"Active length: {active_length}"
    )

    print(
        f"Radius: {radius}"
    )

    try:

        ablation = AblationGeometry(
            trajectory=litt_trajectory,
            active_length=active_length,
            radius=radius,
        )

    except Exception as exc:

        print()
        print("ERROR creating AblationGeometry.")
        print()
        print("Exception:")
        print(exc)

        return 1

    print(
        "[PASS] Ablation geometry created."
    )


    # =================================================================
    # 3. CREATE CSF COMPARTMENT
    # =================================================================

    section("3. CSF COMPARTMENT")

    csf_center = make_point(
        0.0,
        0.0,
        0.0,
    )

    # CSFCompartment expects three ellipsoid radii as Point3D.
    csf_radii = make_point(
        18.0,
        18.0,
        18.0,
    )

    print(
        "CSF center:",
        point_array(csf_center),
    )

    print(
        "CSF radii :",
        point_array(csf_radii),
    )

    try:

        csf = CSFCompartment(
            center=csf_center,
            radii=csf_radii,
        )

    except Exception as exc:

        print()
        print("ERROR creating CSFCompartment.")
        print()
        print("Exception:")
        print(exc)

        return 1

    print(
        "[PASS] CSF compartment created."
    )


    # =================================================================
    # 4. GENERATE PERTURBATIONS
    # =================================================================

    section("4. TRAJECTORY PERTURBATIONS")

    try:

        perturbations = (
            generate_trajectory_perturbations(
                trajectory=litt_trajectory,
                translation_step=1.0,
                rotation_step_degrees=2.0,
            )
        )

    except Exception as exc:

        print()
        print(
            "ERROR generating trajectory perturbations."
        )

        print()
        print("Exception:")
        print(exc)

        return 1

    print(
        f"Generated: {len(perturbations)} trajectories"
    )

    for index, perturbation in enumerate(
        perturbations,
        start=1,
    ):

        length = trajectory_length(
            perturbation.trajectory
        )

        print(
            f"{index:02d}. "
            f"{perturbation.label:<18} "
            f"length={length:.3f}"
        )

    if len(perturbations) != 13:

        print()
        print(
            "[FAIL] Expected 13 perturbations."
        )

        return 1

    print()
    print(
        "[PASS] 13 perturbations generated."
    )


    # =================================================================
    # 5. CHECK LENGTH PRESERVATION
    # =================================================================

    section("5. PERTURBATION LENGTH CHECK")

    lengths = np.array(
        [
            trajectory_length(
                perturbation.trajectory
            )
            for perturbation in perturbations
        ],
        dtype=float,
    )

    length_errors = np.abs(
        lengths - original_length
    )

    maximum_length_error = float(
        np.max(length_errors)
    )

    print(
        "Original length:",
        f"{original_length:.6f}",
    )

    print(
        "Maximum perturbation length error:",
        f"{maximum_length_error:.6e}",
    )

    if not np.allclose(
        lengths,
        original_length,
        rtol=1e-6,
        atol=1e-6,
    ):

        print(
            "[FAIL] Perturbations did not preserve "
            "trajectory length."
        )

        return 1

    print(
        "[PASS] All perturbations preserve trajectory length."
    )


    # =================================================================
    # 6. RUN FULL UNCERTAINTY PIPELINE
    # =================================================================

    section("6. CSF / ABLATION UNCERTAINTY PIPELINE")

    print("Running:")
    print("  perturbation")
    print("      ↓")
    print("  ablation volume")
    print("      ↓")
    print("  ablation ↔ CSF geometry")
    print("      ↓")
    print("  CSF exposure")
    print("      ↓")
    print("  uncertainty summary")

    try:

        assessment = (
            analyze_trajectory_uncertainty(
                perturbations=perturbations,
                csf=csf,
                active_length=active_length,
                radius=radius,

                # Reduced sampling keeps the smoke test fast.
                axial_samples=20,
                radial_samples=6,
                angular_samples=12,
            )
        )

    except Exception as exc:

        print()
        print(
            "ERROR during uncertainty analysis."
        )

        print()
        print(
            "Exception:"
        )

        print(exc)

        print()
        print(
            "The trajectory and perturbation layers worked, "
            "so the problem is now inside the ablation/CSF/"
            "uncertainty integration."
        )

        return 1

    print()
    print(
        "[PASS] Uncertainty pipeline completed."
    )


    # =================================================================
    # 7. PRINT RESULTS
    # =================================================================

    section("7. RESULTS")

    print(
        "Trajectories analysed:",
        assessment.number_of_trajectories,
    )

    print()

    print(
        "Mean minimum CSF distance:",
        f"{assessment.minimum_distance_mean:.3f}",
    )

    print(
        "Minimum CSF distance:",
        f"{assessment.minimum_distance_min:.3f}",
    )

    print(
        "Maximum CSF distance:",
        f"{assessment.minimum_distance_max:.3f}",
    )

    print()

    print(
        "Mean CSF exposure:",
        f"{assessment.exposure_percentage_mean:.2f}%",
    )

    print(
        "Minimum CSF exposure:",
        f"{assessment.exposure_fraction_min * 100.0:.2f}%",
    )

    print(
        "Maximum CSF exposure:",
        f"{assessment.exposure_fraction_max * 100.0:.2f}%",
    )

    print()

    print(
        "Ablation-CSF intersection:",
        f"{assessment.intersection_percentage:.2f}%",
    )

    print()

    print(
        "Most exposed:",
        assessment.most_exposed_label,
    )

    print(
        "Least exposed:",
        assessment.least_exposed_label,
    )

    print()

    print(
        "Exposure distribution:"
    )

    for level, count in (
        assessment.exposure_level_distribution.items()
    ):

        print(
            f"  {level}: {count}"
        )


    # =================================================================
    # 8. FINAL SANITY CHECKS
    # =================================================================

    section("8. FINAL SANITY CHECKS")

    checks: list[tuple[str, bool]] = []


    # 13 perturbations
    checks.append(
        (
            "13 perturbations",
            assessment.number_of_trajectories == 13,
        )
    )


    # Exposure range
    checks.append(
        (
            "Exposure fractions in [0, 1]",
            (
                0.0
                <= assessment.exposure_fraction_min
                <= assessment.exposure_fraction_max
                <= 1.0
            ),
        )
    )


    # Intersection probability
    checks.append(
        (
            "Intersection probability in [0, 1]",
            (
                0.0
                <= assessment.intersection_probability
                <= 1.0
            ),
        )
    )


    # Ablation volumes
    checks.append(
        (
            "All ablation volumes positive",
            all(
                result.ablation_volume > 0.0
                for result in assessment.results
            ),
        )
    )


    # Volume inside CSF
    checks.append(
        (
            "CSF-intersection volumes non-negative",
            all(
                result.volume_inside_csf >= 0.0
                for result in assessment.results
            ),
        )
    )


    # Distances
    checks.append(
        (
            "All distances finite",
            all(
                np.isfinite(
                    result.minimum_distance
                )
                for result in assessment.results
            ),
        )
    )


    # Exposure values
    checks.append(
        (
            "All exposure values finite",
            all(
                np.isfinite(
                    result.exposure_fraction
                )
                for result in assessment.results
            ),
        )
    )


    # Print checks
    passed = 0

    for description, success in checks:

        if success:

            print(
                f"[PASS] {description}"
            )

            passed += 1

        else:

            print(
                f"[FAIL] {description}"
            )


    # =================================================================
    # FINAL STATUS
    # =================================================================

    print()

    print(
        f"Checks passed: "
        f"{passed}/{len(checks)}"
    )

    if passed != len(checks):

        print()
        print(
            "🦆 SMOKE TEST FAILED"
        )

        return 1

    print()
    print(
        "🦆 SMOKE TEST PASSED"
    )

    print()
    print(
        "The current ThermoSEEG geometric uncertainty "
        "pipeline executed successfully."
    )

    return 0


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    raise SystemExit(main())