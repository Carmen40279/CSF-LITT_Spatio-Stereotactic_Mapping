from __future__ import annotations

from pathlib import Path

from src.interface.case_loader import load_planning_case
from src.visualization import create_thermoseeg_scene


def main() -> None:
    """Load a planning case and launch ThermoSEEG."""

    project_root = Path(__file__).resolve().parent

    case_path = (
        project_root
        / "config"
        / "planning_case.json"
    )

    print()
    print("=" * 60)
    print("THERMOSEEG")
    print("Stereotactic Neuroplanning Prototype")
    print("=" * 60)
    print()

    print(
        f"Loading planning case: {case_path.name}"
    )

    case = load_planning_case(
        case_path
    )

    print(
        f"Case loaded: {case.case_id}"
    )

    print(
        "Launching interactive 3D environment..."
    )

    scene = create_thermoseeg_scene(
        case
    )

    scene.show()


if __name__ == "__main__":
    main()