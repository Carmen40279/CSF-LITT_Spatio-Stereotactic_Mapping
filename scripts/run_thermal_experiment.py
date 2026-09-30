from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.application.case_processor import process_imaging_case
from src.application.state import PatientCase

from src.anatomy.csf import CSFCompartment

from src.planning.candidate_generation import CandidateGenerationConfig
from src.planning.evaluation import evaluate_litt_trajectory

from src.visualization.brain_scene import create_thermal_experiment_scene


def main() -> None:

    print("=" * 72)
    print("THERMOSEEG — PATIENT-SPECIFIC THERMAL EXPERIMENT")
    print("=" * 72)

    # MRI

    mri_path = (
        PROJECT_ROOT
        / "data"
        / "sample"
        / "mni152_brain_T1_2mm.nii.gz"
    )

    print()
    print(f"MRI: {mri_path}")
    print()
    print("Processing MRI and extracting anatomy...")

    # PATIENT CASE

    case = PatientCase(
        case_id="THERMOSEEG-THERMAL-001"
    )

    # LITT CANDIDATE CONFIGURATION (basically inserting it in the brain)

    candidate_config = CandidateGenerationConfig(
        entry_spacing=3.0,
        search_radius=9.0,
        active_length=25.0,
        radius=4.0,
        max_candidates=9,
    )


    result = process_imaging_case(
        case=case,
        imaging_path=mri_path,
        candidate_config=candidate_config,
    )

    # ANATOMY
    print()
    print("-" * 72)
    print("ANATOMY")
    print("-" * 72)

    print(f"Image shape:          {result.image_shape}")
    print(f"Voxel spacing:        {result.voxel_spacing}")
    print(f"Brain voxels:         {result.brain_voxel_count:,}")
    print(f"Brain volume:         {result.brain_volume_mm3:,.1f} mm³")
    print(f"CSF voxels:           {result.csf_voxel_count:,}")
    print(f"CSF volume:           {result.csf_volume_mm3:,.1f} mm³")

    # PLANNING

    report = result.planning_report
    comparison = report.comparison

    print()
    print("-" * 72)
    print("PLANNING")
    print("-" * 72)

    print(f"Generated candidates: {result.candidate_count}")
    print(f"Non-dominated:         {result.pareto_count}")

    # individualised patient anatomy

    brain_mask = case.anatomy.get_mask("brain")
    patient_csf_mask = case.anatomy.get_mask("csf")

    # comparing cand

    print()
    print("-" * 72)
    print("THERMAL / CSF RESULTS")
    print("-" * 72)

    for candidate in comparison.candidates:

        print()
        print(f"Candidate: {candidate.plan_id}")

        print(
            f"  Target distance:              "
            f"{candidate.target_distance:.3f} mm"
        )

        print(
            f"  Patient CSF exposure:         "
            f"{candidate.csf_exposure_fraction * 100:.2f}%"
        )

        print(
            f"  Minimum CSF distance:          "
            f"{candidate.minimum_csf_distance:.3f} mm"
        )

        print(
            f"  Maximum thermal risk:          "
            f"{candidate.maximum_thermal_risk:.4f}"
        )

    
    # COMPATIBILITY CSF OBJECT

    compatibility_csf = CSFCompartment(
        center=result.target,
        radii=(18.0, 18.0, 18.0),
        resolution=48,
    )

    # select litt plan 
    litt_plans = case.surgery.litt_plans

    if not litt_plans:
        raise RuntimeError(
            "No LITT plans were created for this case."
        )

    selected_plan = litt_plans[0]

    # thermal eval

    full_evaluation = evaluate_litt_trajectory(
        litt_plan=selected_plan,
        csf=compatibility_csf,
        target=None,
        seeg_electrodes=case.surgery.seeg_electrodes,
        uncertainty_enabled=True,
        patient_csf_mask=patient_csf_mask,
    )

    thermal = full_evaluation.thermal


    print()
    print("-" * 72)
    print("SELECTED TRAJECTORY — FULL THERMAL MODEL")
    print("-" * 72)

    print(
        f"Plan:                              "
        f"{selected_plan.plan_id}"
    )

    print(
        f"Maximum temperature without CSF: "
        f"{thermal.maximum_temperature_without_csf:.2f} °C"
    )

    print(
        f"Maximum temperature with CSF:    "
        f"{thermal.maximum_temperature:.2f} °C"
    )

    print(
        f"Maximum CSF temperature reduction: "
        f"{thermal.maximum_csf_temperature_reduction:.2f} °C"
    )

    print(
        f"Mean CSF temperature reduction:    "
        f"{thermal.mean_csf_temperature_reduction:.2f} °C"
    )

    print(
        f"Maximum thermal dose without CSF: "
        f"{thermal.maximum_thermal_dose_without_csf:.4f}"
    )

    print(
        f"Maximum thermal dose with CSF:    "
        f"{thermal.maximum_thermal_dose:.4f}"
    )

    print(
        f"Thermal dose reduction:           "
        f"{thermal.csf_thermal_dose_reduction:.4f}"
    )

    
    # INTERPRETATION

    print()
    print("-" * 72)
    print("INTERPRETATION")
    print("-" * 72)

    print(
        "The same LITT trajectory is evaluated with and without "
        "the modeled CSF heat-sink effect."
    )

    print(
        "The difference between the two thermal fields represents "
        "the modeled reduction in temperature caused by CSF proximity."
    )

    print()
    print(
        "NOTE: This is a computational research prototype. "
        "The thermal solver and CSF segmentation are not clinically validated."
    )

    # 3D THERMAL VISUALIZATION

    print()
    print("-" * 72)
    print("LAUNCHING 3D THERMAL VISUALIZATION")
    print("-" * 72)

    plotter = create_thermal_experiment_scene(
        litt_plan=selected_plan,
        thermal_field=thermal,
        brain_mask=brain_mask,
        patient_csf_mask=patient_csf_mask,
        seeg_electrodes=case.surgery.seeg_electrodes,
        case_id=case.case_id,
    )

    plotter.show()


if __name__ == "__main__":
    main()