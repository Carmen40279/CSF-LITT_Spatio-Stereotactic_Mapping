"""
ThermoSEEG 3D visualization launcher.

Builds a validated PlanningCase and launches the existing
ThermoSEEG 3D visualization.
"""

from __future__ import annotations

import sys
from pathlib import Path


# =====================================================================
# REPOSITORY PATH
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =====================================================================
# THERMOSEEG IMPORTS
# =====================================================================

from src.io.data_models import (
    AblationParameters,
    Coordinate3D,
    CSFParameters,
    LITTPlan,
    PlanningCase,
    SEEGPlan,
)

from src.visualization.brain_scene import (
    create_thermoseeg_scene,
)


# =====================================================================
# PLANNING CASE
# =====================================================================

def build_demo_case() -> PlanningCase:
    """
    Build the current synthetic ThermoSEEG demonstration case.

    This launcher-level case is for visualization testing only.
    It does not represent clinical patient data.
    """

    origin = Coordinate3D(
        0.0,
        0.0,
        0.0,
    )

    return PlanningCase(
        case_id="THERMOSEEG-DEMO",

        seeg=SEEGPlan(
            entry=Coordinate3D(
                20.0,
                -35.0,
                25.0,
            ),
            target=origin,
            contacts=8,
        ),

        litt=LITTPlan(
            entry=Coordinate3D(
                -35.0,
                20.0,
                30.0,
            ),
            target=origin,

            ablation=AblationParameters(
                active_length=25.0,
                radius=4.0,
            ),
        ),

        csf=CSFParameters(
            center=origin,
            radii=Coordinate3D(
                18.0,
                18.0,
                18.0,
            ),
            resolution=48,
        ),
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:

    print()
    print("=" * 72)
    print("THERMOSEEG 3D VISUALIZATION")
    print("=" * 72)

    # ---------------------------------------------------------------
    # BUILD PLANNING CASE
    # ---------------------------------------------------------------

    print()
    print("Building ThermoSEEG planning case...")

    try:

        case = build_demo_case()

    except Exception as exc:

        print()
        print("ERROR building ThermoSEEG planning case.")
        print()
        print("Exception:")
        print(exc)

        return 1

    print("[PASS] Planning case constructed.")
    print()
    print(f"Case ID: {case.case_id}")

    # ---------------------------------------------------------------
    # BUILD 3D SCENE
    # ---------------------------------------------------------------

    print()
    print("Building ThermoSEEG 3D scene...")

    try:

        plotter = create_thermoseeg_scene(
            case,
        )

    except Exception as exc:

        print()
        print("ERROR building ThermoSEEG 3D scene.")
        print()
        print("Exception:")
        print(exc)

        return 1

    print()
    print("[PASS] 3D scene constructed.")
    print()
    print("Launching PyVista...")
    print()

    # ---------------------------------------------------------------
    # SHOW SCENE
    # ---------------------------------------------------------------

    try:

        plotter.show()

    except Exception as exc:

        print()
        print("ERROR launching PyVista.")
        print()
        print("Exception:")
        print(exc)

        return 1

    print()
    print("ThermoSEEG 3D visualization closed.")

    return 0


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    raise SystemExit(main())