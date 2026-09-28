"""
Run the ThermoSEEG scientific planning engine independently
of the GUI.

This script is an engineering/prototype demonstration using
synthetic CSF and SEEG geometry.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anatomy.csf import CSFCompartment
from src.geometry.vectors import Point3D
from src.planning.planning_engine import ScientificPlanningEngine
from src.planning.planning_report import save_planning_report
from src.surgery.seeg import SEEGElectrode
from src.trajectory.trajectory import Trajectory


def main() -> None:
    """
    Execute one complete synthetic ThermoSEEG planning run.
    """

    # -----------------------------------------------------------------------
    # 1. Base LITT trajectory
    # -----------------------------------------------------------------------

    litt_trajectory = Trajectory(
        entry=Point3D(
            -35.0,
            20.0,
            30.0,
        ),
        target=Point3D(
            0.0,
            0.0,
            0.0,
        ),
    )

    # -----------------------------------------------------------------------
    # 2. Synthetic CSF compartment
    # -----------------------------------------------------------------------

    # CSFCompartment.radii must be a Point3D.
    csf = CSFCompartment(
        center=Point3D(
            0.0,
            0.0,
            0.0,
        ),
        radii=Point3D(
            18.0,
            18.0,
            18.0,
        ),
        resolution=48,
    )

    # -----------------------------------------------------------------------
    # 3. Existing SEEG electrode
    # -----------------------------------------------------------------------

    seeg_trajectory = Trajectory(
        entry=Point3D(
            20.0,
            -35.0,
            25.0,
        ),
        target=Point3D(
            0.0,
            0.0,
            0.0,
        ),
    )

    seeg = SEEGElectrode(
        electrode_id="SEEG-01",
        trajectory=seeg_trajectory,
        contact_count=8,
        contact_spacing=3.5,
    )

    seeg.generate_contacts()

    # -----------------------------------------------------------------------
    # 4. Scientific planning engine
    # -----------------------------------------------------------------------

    engine = ScientificPlanningEngine(
        csf=csf,
        seeg_electrodes=[seeg],
        active_length=25.0,
        radius=4.0,
        target_radius=5.0,
        seeg_clearance_limit=3.0,
    )

    # -----------------------------------------------------------------------
    # 5. Generate and evaluate candidate trajectories
    # -----------------------------------------------------------------------

    result = engine.run(
        litt_trajectory,
        entry_step=3.0,
        target_step=2.0,
        max_candidates=81,
    )

    # -----------------------------------------------------------------------
    # 6. Save scientific planning report
    # -----------------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "scientific_planning_H1.json"
    )

    output_path = save_planning_report(
        result,
        output_path,
    )

    # -----------------------------------------------------------------------
    # 7. Console summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("THERMOSEEG — SCIENTIFIC PLANNING ENGINE")
    print("=" * 72)
    print()

    print(
        f"Candidates evaluated: "
        f"{len(result.candidates)}"
    )

    print(
        "Pareto-efficient candidates: "
        f"{len(result.comparison.pareto_candidate_ids)}"
    )

    print()
    print("PARETO CANDIDATES")
    print("-" * 72)

    if not result.comparison.pareto_candidate_ids:
        print(
            "No Pareto-efficient candidates were identified."
        )
    else:
        evaluations_by_id = {
            evaluation.candidate_id: evaluation
            for evaluation in result.evaluations
        }

        for candidate_id in result.comparison.pareto_candidate_ids:
            evaluation = evaluations_by_id[candidate_id]

            print(
                f"{candidate_id} | "
                f"target error="
                f"{evaluation.target_error:.2f} mm | "
                f"coverage="
                f"{evaluation.target_coverage.covered_percentage:.1f}% | "
                f"CSF="
                f"{evaluation.csf_exposure_percentage:.1f}% | "
                f"thermal max="
                f"{evaluation.thermal_maximum_risk:.3f} | "
                f"SEEG clearance="
                f"{evaluation.seeg_constraints.minimum_electrode_clearance:.2f} mm"
            )

    print()
    print(f"Full report: {output_path}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
