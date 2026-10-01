"""
ThermoSEEG interactive 3D visualization.

This module contains two visualization paths:

1. create_thermoseeg_scene()
   Preserved regression visualization containing:
   - MRI-derived brain
   - SEEG trajectory
   - SEEG contacts
   - target geometry
   - LITT trajectory
   - LITT ablation geometry
   - synthetic CSF compartment
   - LITT/CSF spatial analysis
   - CSF exposure assessment

2. create_thermal_experiment_scene()
   Scientific thermal visualization containing:
   - patient-specific brain surface
   - patient-specific CSF mask
   - LITT trajectory
   - LITT ablation geometry
   - modeled temperature field
   - CSF-induced temperature reduction
   - thermal dose / CEM43 information.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import filters, measure

from src.anatomy.anatomical_model import AnatomicalMask
from src.anatomy.csf import CSFCompartment
from src.geometry.distances import (
    closest_point_on_segment,
    point_to_segment_distance,
)
from src.geometry.spatial_intersection import (
    analyze_ablation_csf,
)
from src.io.data_models import PlanningCase
from src.risk.csf_exposure import (
    assess_csf_exposure,
)
from src.risk.thermal_field import (
    ThermalField,
)
from src.trajectory.ablation import (
    AblationGeometry,
    create_ablation_meshes,
)
from src.trajectory.entry_target import generate_contacts
from src.trajectory.trajectory import Trajectory


MRI_PATH = Path(
    "data/sample/sub-01_ses-forrestgump_T1w.nii.gz"
)

BRAIN_TARGET_RADIUS = 50.0


def load_mri(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a structural T1 MRI and its voxel spacing."""

    if not path.exists():
        raise FileNotFoundError(
            f"MRI file not found: {path}"
        )

    image = nib.load(str(path))

    data = np.asarray(
        image.dataobj,
        dtype=np.float32,
    )

    spacing = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]
    )

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, got {data.ndim}D."
        )

    return data, spacing


def normalize_mri(
    data: np.ndarray,
) -> np.ndarray:
    """Robustly normalize MRI intensity values."""

    finite = data[np.isfinite(data)]

    if finite.size == 0:
        raise ValueError(
            "MRI contains no finite intensity values."
        )

    low = np.percentile(
        finite,
        1.0,
    )

    high = np.percentile(
        finite,
        99.0,
    )

    normalized = (
        (data - low)
        / (high - low + 1e-8)
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    """Keep the largest connected 3D component."""

    labels, count = ndimage.label(
        mask
    )

    if count == 0:
        raise RuntimeError(
            "No connected anatomical region was found."
        )

    sizes = np.bincount(
        labels.ravel()
    )

    sizes[0] = 0

    largest = np.argmax(
        sizes
    )

    return labels == largest


def create_brain_mask(
    normalized: np.ndarray,
) -> np.ndarray:
    """
    Estimate a brain-tissue mask from the structural MRI.

    This is intended for visualization/prototyping, not clinical
    segmentation.
    """

    head_threshold = filters.threshold_otsu(
        normalized
    )

    head_mask = normalized > (
        head_threshold * 0.35
    )

    head_mask = ndimage.binary_closing(
        head_mask,
        structure=np.ones(
            (5, 5, 5)
        ),
        iterations=2,
    )

    head_mask = ndimage.binary_fill_holes(
        head_mask
    )

    head_mask = largest_component(
        head_mask
    )

    interior = ndimage.binary_erosion(
        head_mask,
        structure=np.ones(
            (3, 3, 3)
        ),
        iterations=8,
    )

    interior_values = normalized[
        interior
    ]

    if interior_values.size == 0:
        raise RuntimeError(
            "MRI interior could not be established."
        )

    tissue_threshold = filters.threshold_otsu(
        interior_values
    )

    brain_mask = (
        normalized > tissue_threshold
    ) & interior

    brain_mask = ndimage.binary_opening(
        brain_mask,
        structure=np.ones(
            (3, 3, 3)
        ),
    )

    brain_mask = ndimage.binary_closing(
        brain_mask,
        structure=np.ones(
            (5, 5, 5)
        ),
        iterations=2,
    )

    brain_mask = ndimage.binary_fill_holes(
        brain_mask
    )

    brain_mask = largest_component(
        brain_mask
    )

    brain_mask = ndimage.binary_dilation(
        brain_mask,
        structure=np.ones(
            (3, 3, 3
        )),
        iterations=2,
    )

    return brain_mask


def extract_brain_mesh(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> pv.PolyData:
    """Extract a 3D anatomical surface using marching cubes."""

    vertices, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=spacing,
        step_size=2,
    )

    faces_pv = np.column_stack(
        (
            np.full(
                len(faces),
                3,
                dtype=np.int64,
            ),
            faces,
        )
    )

    return pv.PolyData(
        vertices,
        faces_pv,
    )


def fit_brain_to_planning_space(
    mesh: pv.PolyData,
    target_radius: float = BRAIN_TARGET_RADIUS,
) -> pv.PolyData:
    """
    Fit the anatomical brain mesh into the coordinate envelope
    previously occupied by the synthetic sphere.
    """

    fitted = mesh.copy()

    bounds = np.array(
        fitted.bounds,
        dtype=float,
    )

    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
    )

    fitted.points -= center

    maximum_extent = max(
        fitted.length,
        np.ptp(
            fitted.points[:, 0]
        ),
        np.ptp(
            fitted.points[:, 1]
        ),
        np.ptp(
            fitted.points[:, 2]
        ),
    )

    if maximum_extent <= 0:
        raise RuntimeError(
            "Invalid reconstructed brain geometry."
        )

    scale = (
        (target_radius * 2.0)
        / maximum_extent
    )

    fitted.points *= scale

    fitted = fitted.clean()

    return fitted


def create_mri_brain() -> pv.PolyData:
    """Complete original MRI → anatomical brain mesh pipeline."""

    print(
        "\n=== ThermoSEEG MRI Anatomical Reconstruction ==="
    )

    data, spacing = load_mri(
        MRI_PATH
    )

    print(
        f"MRI shape:      {data.shape}"
    )

    print(
        f"Voxel spacing:  {spacing}"
    )

    normalized = normalize_mri(
        data
    )

    print(
        "MRI normalization: complete"
    )

    brain_mask = create_brain_mask(
        normalized
    )

    print(
        f"Brain mask voxels: "
        f"{int(brain_mask.sum()):,}"
    )

    brain = extract_brain_mesh(
        brain_mask,
        spacing,
    )

    print(
        f"Raw brain mesh: "
        f"{brain.n_points:,} vertices / "
        f"{brain.n_cells:,} faces"
    )

    brain = brain.smooth(
        n_iter=20,
        relaxation_factor=0.01,
        feature_smoothing=False,
        boundary_smoothing=True,
    )

    brain = fit_brain_to_planning_space(
        brain
    )

    print(
        f"Planning-space brain: "
        f"{brain.n_points:,} vertices / "
        f"{brain.n_cells:,} faces"
    )

    print(
        "MRI anatomical reconstruction complete."
    )

    print(
        "===============================================\n"
    )

    return brain


def create_thermoseeg_scene(
    case: PlanningCase,
) -> pv.Plotter:
    """
    Create the original ThermoSEEG interactive 3D scene.

    This function is intentionally retained as the regression
    visualization.
    """

    plotter = pv.Plotter(
        window_size=(1400, 900),
        title="ThermoSEEG — Stereotactic Neuroplanning",
    )

    brain = create_mri_brain()

    plotter.add_mesh(
        brain,
        opacity=0.28,
        smooth_shading=True,
        color="lightgray",
        name="brain",
    )

    seeg_entry = case.seeg.entry.to_array()
    seeg_target = case.seeg.target.to_array()

    seeg_trajectory = Trajectory(
        entry=case.seeg.entry.to_point(),
        target=case.seeg.target.to_point(),
    )

    seeg_entry_marker = pv.Sphere(
        radius=2.5,
        center=seeg_entry,
        theta_resolution=24,
        phi_resolution=24,
    )

    plotter.add_mesh(
        seeg_entry_marker,
        color="blue",
        smooth_shading=True,
        name="seeg_entry",
    )

    seeg_target_marker = pv.Sphere(
        radius=3.5,
        center=seeg_target,
        theta_resolution=32,
        phi_resolution=32,
    )

    plotter.add_mesh(
        seeg_target_marker,
        color="red",
        smooth_shading=True,
        name="seeg_target",
    )

    seeg_line = pv.Line(
        seeg_entry,
        seeg_target,
    )

    plotter.add_mesh(
        seeg_line,
        color="yellow",
        line_width=7,
        name="seeg_electrode",
    )

    seeg_contacts = generate_contacts(
        seeg_trajectory,
        case.seeg.contacts,
    )

    contact_positions = np.array(
        [
            [
                contact.x,
                contact.y,
                contact.z,
            ]
            for contact in seeg_contacts
        ]
    )

    for index, position in enumerate(
        contact_positions,
        start=1,
    ):
        contact_mesh = pv.Sphere(
            radius=1.6,
            center=position,
            theta_resolution=20,
            phi_resolution=20,
        )

        plotter.add_mesh(
            contact_mesh,
            color="orange",
            smooth_shading=True,
            name=f"seeg_contact_{index}",
        )

    plotter.add_point_labels(
        contact_positions,
        [
            f"SEEG C{index}"
            for index in range(
                1,
                case.seeg.contacts + 1,
            )
        ],
        font_size=11,
        point_size=8,
        name="seeg_contact_labels",
    )

    test_structure = (
        case.seeg.target.to_point()
    )

    test_structure_position = np.array(
        [
            test_structure.x,
            test_structure.y,
            test_structure.z,
        ]
    )

    structure_distance = (
        point_to_segment_distance(
            test_structure,
            seeg_trajectory.entry,
            seeg_trajectory.target,
        )
    )

    closest_point = (
        closest_point_on_segment(
            test_structure,
            seeg_trajectory.entry,
            seeg_trajectory.target,
        )
    )

    closest_position = np.array(
        [
            closest_point.x,
            closest_point.y,
            closest_point.z,
        ]
    )

    distance_line = pv.Line(
        test_structure_position,
        closest_position,
    )

    plotter.add_mesh(
        distance_line,
        color="green",
        line_width=4,
        name="structure_distance",
    )

    litt_entry = case.litt.entry.to_array()
    litt_target = case.litt.target.to_array()

    litt_trajectory = Trajectory(
        entry=case.litt.entry.to_point(),
        target=case.litt.target.to_point(),
    )

    litt_entry_marker = pv.Sphere(
        radius=2.5,
        center=litt_entry,
        theta_resolution=24,
        phi_resolution=24,
    )

    plotter.add_mesh(
        litt_entry_marker,
        color="cyan",
        smooth_shading=True,
        name="litt_entry",
    )

    litt_line = pv.Line(
        litt_entry,
        litt_target,
    )

    plotter.add_mesh(
        litt_line,
        color="magenta",
        line_width=8,
        name="litt_trajectory",
    )

    litt_target_marker = pv.Sphere(
        radius=4.0,
        center=litt_target,
        theta_resolution=32,
        phi_resolution=32,
    )

    plotter.add_mesh(
        litt_target_marker,
        color="magenta",
        smooth_shading=True,
        name="litt_target",
    )

    litt_ablation = AblationGeometry(
        trajectory=litt_trajectory,
        active_length=(
            case.litt.ablation.active_length
        ),
        radius=(
            case.litt.ablation.radius
        ),
    )

    (
        ablation_cylinder,
        ablation_start_cap,
        ablation_end_cap,
    ) = create_ablation_meshes(
        litt_ablation,
    )

    plotter.add_mesh(
        ablation_cylinder,
        opacity=0.25,
        color="magenta",
        smooth_shading=True,
        name="litt_ablation_cylinder",
    )

    plotter.add_mesh(
        ablation_start_cap,
        opacity=0.25,
        color="magenta",
        smooth_shading=True,
        name="litt_ablation_start_cap",
    )

    plotter.add_mesh(
        ablation_end_cap,
        opacity=0.25,
        color="magenta",
        smooth_shading=True,
        name="litt_ablation_end_cap",
    )

    csf = CSFCompartment(
        center=case.csf.center.to_point(),
        radii=case.csf.radii.to_point(),
        resolution=case.csf.resolution,
    )

    csf_mesh = csf.create_mesh()

    plotter.add_mesh(
        csf_mesh,
        opacity=0.18,
        color="deepskyblue",
        smooth_shading=True,
        name="csf_compartment",
    )

    trajectory_csf_analysis = (
        csf.trajectory_analysis(
            litt_trajectory,
            samples=1000,
        )
    )

    trajectory_csf_distance = float(
        trajectory_csf_analysis[
            "minimum_distance"
        ]
    )

    trajectory_csf_fraction = float(
        trajectory_csf_analysis[
            "fraction_inside"
        ]
    )

    trajectory_csf_intersects = bool(
        trajectory_csf_analysis[
            "intersects"
        ]
    )

    ablation_csf_analysis = analyze_ablation_csf(
        litt_ablation,
        csf,
        axial_samples=60,
        radial_samples=12,
        angular_samples=24,
    )

    ablation_csf_distance = float(
        ablation_csf_analysis.minimum_distance
    )

    ablation_csf_fraction = float(
        ablation_csf_analysis.fraction_inside_csf
    )

    ablation_csf_intersects = bool(
        ablation_csf_analysis.intersects
    )

    ablation_volume = float(
        ablation_csf_analysis.estimated_ablation_volume
    )

    ablation_volume_inside_csf = float(
        ablation_csf_analysis.estimated_volume_inside_csf
    )

    csf_exposure = assess_csf_exposure(
        litt_ablation,
        csf,
    )

    csf_exposure_level = (
        csf_exposure.exposure_level
    )

    csf_exposure_fraction = float(
        csf_exposure.exposure_fraction
    )

    csf_exposure_interpretation = (
        csf_exposure.interpretation
    )

    if (
        ablation_csf_analysis.intersection_points.size
        > 0
    ):
        plotter.add_points(
            ablation_csf_analysis.intersection_points,
            color="white",
            point_size=7,
            render_points_as_spheres=True,
            name="ablation_csf_intersection_points",
        )

    closest_ablation_point = (
        ablation_csf_analysis.closest_ablation_point
    )

    closest_csf_point = (
        ablation_csf_analysis.closest_csf_point
    )

    ablation_csf_distance_line = pv.Line(
        closest_ablation_point,
        closest_csf_point,
    )

    plotter.add_mesh(
        ablation_csf_distance_line,
        color="green",
        line_width=4,
        name="ablation_csf_distance",
    )

    plotter.add_point_labels(
        [seeg_target],
        ["SEEG TARGET"],
        font_size=16,
        point_size=10,
        name="seeg_target_label",
    )

    plotter.add_point_labels(
        [litt_target],
        ["LITT TARGET"],
        font_size=16,
        point_size=10,
        name="litt_target_label",
    )

    trajectory_csf_intersection_text = (
        "YES"
        if trajectory_csf_intersects
        else "NO"
    )

    ablation_csf_intersection_text = (
        "YES"
        if ablation_csf_intersects
        else "NO"
    )

    info_text = (
        "THERMOSEEG\n"
        "Stereotactic Neuroplanning Prototype\n"
        "\n"
        f"Case: {case.case_id}\n"
        "\n"
        "ANATOMY\n"
        "MRI-derived brain surface\n"
        "\n"
        "SEEG\n"
        f"Trajectory length: "
        f"{seeg_trajectory.length:.2f} units\n"
        f"Contacts: {case.seeg.contacts}\n"
        f"Distance to target geometry: "
        f"{structure_distance:.2f} units\n"
        "\n"
        "LITT\n"
        f"Trajectory length: "
        f"{litt_trajectory.length:.2f} units\n"
        f"Ablation length: "
        f"{case.litt.ablation.active_length:.2f} units\n"
        f"Ablation radius: "
        f"{case.litt.ablation.radius:.2f} units\n"
        f"Ablation volume: "
        f"{ablation_volume:.2f} units³\n"
        "\n"
        "CSF — TRAJECTORY\n"
        f"Centerline intersects CSF: "
        f"{trajectory_csf_intersection_text}\n"
        f"Minimum centerline-CSF distance: "
        f"{trajectory_csf_distance:.2f} units\n"
        f"Trajectory inside CSF: "
        f"{trajectory_csf_fraction * 100:.1f}%\n"
        "\n"
        "CSF — ABLATION VOLUME\n"
        f"Ablation intersects CSF: "
        f"{ablation_csf_intersection_text}\n"
        f"Minimum ablation-CSF distance: "
        f"{ablation_csf_distance:.2f} units\n"
        f"Ablation volume inside CSF: "
        f"{ablation_volume_inside_csf:.2f} units³\n"
        f"Ablation volume exposed to CSF: "
        f"{ablation_csf_fraction * 100:.1f}%\n"
        "\n"
        "CSF EXPOSURE ASSESSMENT\n"
        f"Exposure level: "
        f"{csf_exposure_level}\n"
        f"Exposure fraction: "
        f"{csf_exposure_fraction * 100:.1f}%\n"
        f"Assessment: "
        f"{csf_exposure_interpretation}"
    )

    plotter.add_text(
        info_text,
        position="upper_left",
        font_size=13,
    )

    plotter.add_legend(
        [
            ["MRI-derived brain", "lightgray"],
            ["SEEG electrode", "yellow"],
            ["SEEG contacts", "orange"],
            ["SEEG target", "red"],
            ["LITT trajectory", "magenta"],
            ["LITT ablation", "magenta"],
            ["Synthetic CSF", "deepskyblue"],
            ["Ablation-CSF intersection", "white"],
            ["Spatial distance", "green"],
        ],
        bcolor="black",
        face="circle",
    )

    plotter.add_axes(
        interactive=True,
    )

    plotter.show_grid(
        color="gray",
    )

    plotter.camera_position = "iso"

    return plotter


def anatomical_mask_to_mesh(
    anatomical_mask: AnatomicalMask,
    *,
    step_size: int = 2,
) -> pv.PolyData:
    """
    Convert a patient-space anatomical mask into a PyVista mesh.

    The mask is assumed to be a 3D binary array with an associated
    4x4 affine transformation matrix mapping voxel coordinates.
    """

    mask = np.asarray(
        anatomical_mask.mask,
        dtype=bool,
    )

    if mask.ndim != 3:
        raise ValueError(
            "Anatomical mask must be 3-dimensional."
        )

    affine = np.asarray(
        anatomical_mask.affine,
        dtype=float,
    )

    if affine.shape != (4, 4):
        raise ValueError(
            "Anatomical mask affine must be 4x4."
        )

    if not np.any(mask):
        raise ValueError(
            f"Anatomical mask '{anatomical_mask.name}' is empty."
        )

    spacing = np.linalg.norm(
        affine[:3, :3],
        axis=0,
    )

    vertices, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        spacing=tuple(
            float(value)
            for value in spacing
        ),
        step_size=step_size,
    )

    homogeneous = np.column_stack(
        (
            vertices,
            np.ones(
                len(vertices),
                dtype=float,
            ),
        )
    )

    world_vertices = (
        homogeneous
        @ affine.T
    )[:, :3]

    faces_pv = np.column_stack(
        (
            np.full(
                len(faces),
                3,
                dtype=np.int64,
            ),
            faces,
        )
    )

    return pv.PolyData(
        world_vertices,
        faces_pv,
    ).clean()


def thermal_field_to_grid(
    thermal_field: ThermalField,
    *,
    value_name: str = "temperature",
) -> pv.ImageData:
    """
    Convert a thermal solver field into a PyVista ImageData grid.

    Solver values are stored in NumPy C-order. PyVista ImageData
    point data uses VTK ordering, so the field is reconstructed in
    solver order and flattened in Fortran order for PyVista.
    """

    shape = tuple(
        int(value)
        for value in thermal_field.grid_shape
    )

    if len(shape) != 3:
        raise ValueError(
            "Thermal field grid_shape must contain three dimensions."
        )

    if any(
        value <= 0
        for value in shape
    ):
        raise ValueError(
            f"Thermal field grid_shape must contain positive "
            f"dimensions, got {shape}."
        )

    expected_size = int(
        shape[0]
        * shape[1]
        * shape[2]
    )

    if not hasattr(
        thermal_field,
        value_name,
    ):
        raise AttributeError(
            f"ThermalField has no field named '{value_name}'."
        )

    values = np.asarray(
        getattr(
            thermal_field,
            value_name,
        ),
        dtype=float,
    )

    if values.size != expected_size:
        raise ValueError(
            f"Thermal field '{value_name}' contains "
            f"{values.size} values, expected "
            f"{expected_size} for grid shape {shape}."
        )

    origin = tuple(
        float(value)
        for value in thermal_field.origin
    )

    spacing = tuple(
        float(value)
        for value in thermal_field.spacing
    )

    if len(origin) != 3:
        raise ValueError(
            "Thermal field origin must contain three values."
        )

    if len(spacing) != 3:
        raise ValueError(
            "Thermal field spacing must contain three values."
        )

    if any(
        not np.isfinite(value)
        for value in origin
    ):
        raise ValueError(
            f"Thermal field origin contains non-finite values: "
            f"{origin}"
        )

    if any(
        not np.isfinite(value)
        or value <= 0.0
        for value in spacing
    ):
        raise ValueError(
            f"Thermal field spacing must contain positive "
            f"finite values, got {spacing}."
        )

    solver_volume = values.reshape(
        shape,
        order="C",
    )

    pyvista_values = solver_volume.ravel(
        order="F"
    )

    if pyvista_values.size != expected_size:
        raise RuntimeError(
            "Thermal field conversion changed the number of points."
        )

    grid = pv.ImageData()

    grid.dimensions = shape
    grid.origin = origin
    grid.spacing = spacing

    grid.point_data[value_name] = pyvista_values

    return grid


def validate_thermal_grid_geometry(
    reference: pv.ImageData,
    candidate: pv.ImageData,
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Verify that two thermal fields use the same physical grid."""

    if tuple(reference.dimensions) != tuple(
        candidate.dimensions
    ):
        raise ValueError(
            f"Thermal grid mismatch: "
            f"{reference_name} dimensions="
            f"{tuple(reference.dimensions)}, "
            f"{candidate_name} dimensions="
            f"{tuple(candidate.dimensions)}."
        )

    if not np.allclose(
        np.asarray(reference.origin, dtype=float),
        np.asarray(candidate.origin, dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Thermal grid mismatch: "
            f"{reference_name} and {candidate_name} "
            f"have different origins."
        )

    if not np.allclose(
        np.asarray(reference.spacing, dtype=float),
        np.asarray(candidate.spacing, dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Thermal grid mismatch: "
            f"{reference_name} and {candidate_name} "
            f"have different spacing."
        )


def create_temperature_isosurface(
    thermal_field: ThermalField,
    *,
    temperature: float,
) -> pv.PolyData:
    """Extract a temperature isosurface from the modeled field."""

    grid = thermal_field_to_grid(
        thermal_field,
        value_name="temperature",
    )

    values = grid.point_data[
        "temperature"
    ]

    if values.size == 0:
        return pv.PolyData()

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return pv.PolyData()

    minimum = float(
        finite_values.min()
    )

    maximum = float(
        finite_values.max()
    )

    if temperature < minimum:
        return pv.PolyData()

    if temperature > maximum:
        return pv.PolyData()

    try:
        return grid.contour(
            isosurfaces=[
                temperature
            ],
            scalars="temperature",
        )
    except Exception:
        return pv.PolyData()


def create_thermal_experiment_scene(
    litt_plan,
    thermal_field: ThermalField,
    *,
    brain_mask: AnatomicalMask | None = None,
    patient_csf_mask: AnatomicalMask | None = None,
    seeg_electrodes=None,
    case_id: str = "THERMOSEEG-THERMAL-001",
) -> pv.Plotter:

    plotter = pv.Plotter(
        window_size=(1500, 950),
        title="ThermoSEEG — Patient-Specific Thermal Field",
    )

    if brain_mask is not None:
        brain_mesh = anatomical_mask_to_mesh(
            brain_mask,
            step_size=2,
        )

        plotter.add_mesh(
            brain_mesh,
            color="lightgray",
            opacity=0.16,
            smooth_shading=True,
            name="patient_brain",
        )

    if patient_csf_mask is not None:
        csf_mesh = anatomical_mask_to_mesh(
            patient_csf_mask,
            step_size=2,
        )

        plotter.add_mesh(
            csf_mesh,
            color="deepskyblue",
            opacity=0.12,
            smooth_shading=True,
            name="patient_csf",
        )

    trajectory = litt_plan.trajectory

    entry = np.array(
        [
            trajectory.entry.x,
            trajectory.entry.y,
            trajectory.entry.z,
        ],
        dtype=float,
    )

    target = np.array(
        [
            trajectory.target.x,
            trajectory.target.y,
            trajectory.target.z,
        ],
        dtype=float,
    )

    plotter.add_mesh(
        pv.Line(
            entry,
            target,
        ),
        color="magenta",
        line_width=8,
        name="litt_trajectory",
    )

    plotter.add_mesh(
        pv.Sphere(
            radius=2.5,
            center=entry,
            theta_resolution=24,
            phi_resolution=24,
        ),
        color="cyan",
        smooth_shading=True,
        name="litt_entry",
    )

    plotter.add_mesh(
        pv.Sphere(
            radius=4.0,
            center=target,
            theta_resolution=32,
            phi_resolution=32,
        ),
        color="magenta",
        smooth_shading=True,
        name="litt_target",
    )

    ablation = AblationGeometry(
        trajectory=trajectory,
        active_length=(
            litt_plan.parameters.active_length
        ),
        radius=(
            litt_plan.parameters.radius
        ),
    )

    (
        ablation_cylinder,
        ablation_start_cap,
        ablation_end_cap,
    ) = create_ablation_meshes(
        ablation,
    )

    plotter.add_mesh(
        ablation_cylinder,
        color="magenta",
        opacity=0.18,
        smooth_shading=True,
        name="thermal_litt_ablation",
    )

    plotter.add_mesh(
        ablation_start_cap,
        color="magenta",
        opacity=0.18,
        smooth_shading=True,
        name="thermal_litt_start_cap",
    )

    plotter.add_mesh(
        ablation_end_cap,
        color="magenta",
        opacity=0.18,
        smooth_shading=True,
        name="thermal_litt_end_cap",
    )

    temperature_grid = thermal_field_to_grid(
        thermal_field,
        value_name="temperature",
    )

    temperature_without_csf_grid = (
        thermal_field_to_grid(
            thermal_field,
            value_name="temperature_without_csf",
        )
    )

    reduction_grid = thermal_field_to_grid(
        thermal_field,
        value_name="csf_temperature_reduction",
    )

    validate_thermal_grid_geometry(
        temperature_grid,
        temperature_without_csf_grid,
        reference_name="temperature",
        candidate_name="temperature_without_csf",
    )

    validate_thermal_grid_geometry(
        temperature_grid,
        reduction_grid,
        reference_name="temperature",
        candidate_name="csf_temperature_reduction",
    )

    temperature_values = np.asarray(
        temperature_grid.point_data[
            "temperature"
        ],
        dtype=float,
    )

    finite_temperature = temperature_values[
        np.isfinite(
            temperature_values
        )
    ]

    if finite_temperature.size > 0:
        temperature_max = float(
            finite_temperature.max()
        )

        if temperature_max > 37.0:
            temperature_clim_max = max(
                60.0,
                temperature_max,
            )

            plotter.add_volume(
                temperature_grid,
                scalars="temperature",
                clim=(
                    37.0,
                    temperature_clim_max,
                ),
                opacity="sigmoid",
                cmap="inferno",
                shade=True,
                name="thermal_temperature_field",
            )

    iso_temperatures = (
        40.0,
        45.0,
        50.0,
        55.0,
    )

    iso_opacities = (
        0.05,
        0.07,
        0.10,
        0.14,
    )

    for temperature, opacity in zip(
        iso_temperatures,
        iso_opacities,
    ):
        surface = create_temperature_isosurface(
            thermal_field,
            temperature=temperature,
        )

        if surface.n_points == 0:
            continue

        plotter.add_mesh(
            surface,
            scalars="temperature",
            clim=(
                37.0,
                max(
                    60.0,
                    float(
                        finite_temperature.max()
                    ),
                ),
            ),
            cmap="inferno",
            opacity=opacity,
            smooth_shading=True,
            name=f"temperature_{temperature:.0f}",
        )

    reduction_values = np.asarray(
        reduction_grid.point_data[
            "csf_temperature_reduction"
        ],
        dtype=float,
    )

    finite_reduction = reduction_values[
        np.isfinite(
            reduction_values
        )
    ]

    if finite_reduction.size > 0:
        maximum_reduction_field = float(
            finite_reduction.max()
        )

        if maximum_reduction_field > 1e-6:
            reduction_clim_max = max(
                1.0,
                maximum_reduction_field,
            )

            plotter.add_volume(
                reduction_grid,
                scalars="csf_temperature_reduction",
                clim=(
                    0.0,
                    reduction_clim_max,
                ),
                opacity="sigmoid",
                cmap="viridis",
                shade=True,
                name="csf_temperature_reduction",
            )

    if seeg_electrodes:
        for electrode_index, electrode in enumerate(
            seeg_electrodes,
            start=1,
        ):
            seeg_entry = np.array(
                [
                    electrode.trajectory.entry.x,
                    electrode.trajectory.entry.y,
                    electrode.trajectory.entry.z,
                ],
                dtype=float,
            )

            seeg_target = np.array(
                [
                    electrode.trajectory.target.x,
                    electrode.trajectory.target.y,
                    electrode.trajectory.target.z,
                ],
                dtype=float,
            )

            plotter.add_mesh(
                pv.Line(
                    seeg_entry,
                    seeg_target,
                ),
                color="yellow",
                line_width=5,
                name=f"seeg_{electrode_index}",
            )

    maximum_temperature = float(
        thermal_field.maximum_temperature
    )

    maximum_temperature_without_csf = float(
        thermal_field.maximum_temperature_without_csf
    )

    maximum_reduction = float(
        thermal_field.maximum_csf_temperature_reduction
    )

    maximum_dose = float(
        thermal_field.maximum_thermal_dose
    )

    maximum_dose_without_csf = float(
        thermal_field.maximum_thermal_dose_without_csf
    )

    dose_reduction = float(
        thermal_field.csf_thermal_dose_reduction
    )

    exposure_fraction = (
        float(
            thermal_field.csf_modifier.mean()
        )
        if thermal_field.csf_modifier.size > 0
        else 0.0
    )

    info_text = (
        "THERMOSEEG\n"
        "Patient-Specific Thermal Experiment\n"
        "\n"
        f"Case: {case_id}\n"
        "\n"
        "LITT\n"
        f"Active length: "
        f"{litt_plan.parameters.active_length:.2f} mm\n"
        f"Radius: "
        f"{litt_plan.parameters.radius:.2f} mm\n"
        "\n"
        "MODELED TEMPERATURE\n"
        f"Maximum + CSF: "
        f"{maximum_temperature:.2f} °C\n"
        f"Maximum - CSF: "
        f"{maximum_temperature_without_csf:.2f} °C\n"
        f"Maximum CSF ΔT: "
        f"{maximum_reduction:.2f} °C\n"
        "\n"
        "THERMAL DOSE — CEM43\n"
        f"Maximum + CSF: "
        f"{maximum_dose:.4g}\n"
        f"Maximum - CSF: "
        f"{maximum_dose_without_csf:.4g}\n"
        f"Relative reduction: "
        f"{dose_reduction:.2f}%\n"
        "\n"
        "CSF MODEL\n"
        f"Mean mapped CSF modifier: "
        f"{exposure_fraction:.3f}\n"
        "\n"
        "Research prototype\n"
        "Not clinically validated"
    )

    plotter.add_text(
        info_text,
        position="upper_left",
        font_size=13,
    )

    plotter.add_legend(
        [
            ["Patient brain", "lightgray"],
            ["Patient CSF", "deepskyblue"],
            ["SEEG", "yellow"],
            ["LITT trajectory", "magenta"],
            ["LITT ablation", "magenta"],
            ["Temperature field", "orange"],
            ["CSF ΔT reduction", "green"],
        ],
        bcolor="black",
        face="circle",
    )

    plotter.add_axes(
        interactive=True,
    )

    plotter.show_grid(
        color="gray",
    )

    plotter.camera_position = "iso"

    return plotter