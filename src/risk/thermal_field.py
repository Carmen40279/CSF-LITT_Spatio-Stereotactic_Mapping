"""
ThermoSEEG computational thermal-field model.
This module implements a research-prototype thermal model for
laser interstitial thermal therapy (LITT).
The model solves a simplified Pennes-style bioheat equation and can
use either:
    1. A synthetic ellipsoidal CSF compartment, or
    2. A patient-specific MRI-derived CSF AnatomicalMask. """

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.anatomy.anatomical_model import AnatomicalMask
from src.anatomy.csf import CSFCompartment
from src.risk.csf_thermal_mapping import map_csf_mask_to_grid
from src.trajectory.ablation import AblationGeometry


#data model

@dataclass(frozen=True)
class ThermalField:
    """Three-dimensional computational thermal field."""

    points: np.ndarray
    intensity: np.ndarray
    csf_modifier: np.ndarray
    risk: np.ndarray

    temperature: np.ndarray
    temperature_without_csf: np.ndarray

    thermal_dose: np.ndarray
    thermal_dose_without_csf: np.ndarray

    csf_temperature_reduction: np.ndarray

    grid_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

    simulation_time_seconds: float
    time_step_seconds: float

    laser_power_watts: float
    baseline_temperature_celsius: float
    csf_sink_strength: float

    @property
    def maximum_risk(self) -> float:
        """Return the maximum normalized thermal-dose metric."""

        return float(np.max(self.risk))

    @property
    def mean_risk(self) -> float:
        """Return the mean normalized thermal-dose metric."""

        return float(np.mean(self.risk))

    @property
    def high_risk_voxel_count(self) -> int:
        """Return the number of voxels above the high-risk threshold."""

        return int(
            np.count_nonzero(
                self.risk >= 0.75
            )
        )

    @property
    def maximum_temperature(self) -> float:
        """Return maximum simulated temperature."""

        return float(
            np.max(self.temperature)
        )

    @property
    def maximum_temperature_without_csf(self) -> float:
        """Return maximum temperature without the CSF heat sink."""

        return float(
            np.max(
                self.temperature_without_csf
            )
        )

    @property
    def maximum_csf_temperature_reduction(self) -> float:
        """Return maximum modeled CSF-induced temperature reduction."""

        return float(
            np.max(
                self.csf_temperature_reduction
            )
        )

    @property
    def mean_csf_temperature_reduction(self) -> float:
        """Return mean modeled CSF-induced temperature reduction."""

        return float(
            np.mean(
                self.csf_temperature_reduction
            )
        )

    @property
    def maximum_thermal_dose(self) -> float:
        """Return maximum CEM43 thermal dose."""

        return float(
            np.max(
                self.thermal_dose
            )
        )

    @property
    def maximum_thermal_dose_without_csf(self) -> float:
        """Return maximum CEM43 dose without the CSF heat sink."""

        return float(
            np.max(
                self.thermal_dose_without_csf
            )
        )

    @property
    def csf_thermal_dose_reduction(self) -> float:
        """
        Return the reduction in maximum CEM43 dose associated with
        the modeled CSF heat sink.
        """

        return max(
            0.0,
            self.maximum_thermal_dose_without_csf
            - self.maximum_thermal_dose,
        )

# geometry solver

def _point_array(point) -> np.ndarray:
    """Convert a Point3D-like object or 3-element sequence."""

    if (
        hasattr(point, "x")
        and hasattr(point, "y")
        and hasattr(point, "z")
    ):
        return np.array(
            [
                float(point.x),
                float(point.y),
                float(point.z),
            ],
            dtype=float,
        )

    try:
        values = np.asarray(
            point,
            dtype=float,
        ).reshape(-1)

    except Exception as exc:
        raise TypeError(
            "Coordinate value must be a Point3D-like object "
            "or a 3-element numeric sequence."
        ) from exc

    if values.size != 3:
        raise ValueError(
            "Coordinate value must contain exactly 3 values."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Coordinate value must contain only finite values."
        )

    return values.astype(
        float,
        copy=False,
    )


def _closest_points_on_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Calculate the closest point on a finite segment."""

    segment = end - start

    segment_length_squared = float(
        np.dot(
            segment,
            segment,
        )
    )

    if segment_length_squared == 0.0:
        return np.repeat(
            start.reshape(1, 3),
            len(points),
            axis=0,
        )

    relative = points - start

    parameter = (
        relative @ segment
    ) / segment_length_squared

    parameter = np.clip(
        parameter,
        0.0,
        1.0,
    )

    return (
        start
        + parameter[:, None] * segment
    )


def _distance_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Return distance from each point to a finite segment."""

    closest = _closest_points_on_segment(
        points,
        start,
        end,
    )

    return np.linalg.norm(
        points - closest,
        axis=1,
    )


def _synthetic_csf_membership(
    points: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Return binary occupancy for the synthetic ellipsoidal CSF."""

    relative = points - center

    normalized = (
        relative / radii
    )

    radial_distance = np.sqrt(
        np.sum(
            normalized ** 2,
            axis=1,
        )
    )

    return (
        radial_distance <= 1.0
    ).astype(float)

# Laser source and bioheat solver

def _create_laser_source(
    points: np.ndarray,
    entry: np.ndarray,
    active_end: np.ndarray,
    radius: float,
    laser_power_watts: float,
) -> np.ndarray:
    """
    Construct a normalized volumetric laser heat source.

    The source uses a Gaussian radial profile around the active
    LITT segment and is normalized to the requested laser power.
    """

    distance = _distance_to_segment(
        points,
        entry,
        active_end,
    )

    sigma = max(
        radius / 2.0,
        1.0e-4,
    )

    source = np.exp(
        -0.5
        * (
            distance / sigma
        ) ** 2
    )

    source = np.clip(
        source,
        0.0,
        1.0,
    )

    return (
        source
        * float(laser_power_watts)
    )

# numerical solver

def _laplacian(
    temperature: np.ndarray,
    spacing_m: tuple[float, float, float],
) -> np.ndarray:
    """Calculate a three-dimensional finite-difference Laplacian."""

    padded = np.pad(
        temperature,
        1,
        mode="edge",
    )

    dx, dy, dz = spacing_m

    center = padded[
        1:-1,
        1:-1,
        1:-1,
    ]

    d2x = (
        padded[
            2:,
            1:-1,
            1:-1,
        ]
        - 2.0 * center
        + padded[
            :-2,
            1:-1,
            1:-1,
        ]
    ) / (dx * dx)

    d2y = (
        padded[
            1:-1,
            2:,
            1:-1,
        ]
        - 2.0 * center
        + padded[
            1:-1,
            :-2,
            1:-1,
        ]
    ) / (dy * dy)

    d2z = (
        padded[
            1:-1,
            1:-1,
            2:,
        ]
        - 2.0 * center
        + padded[
            1:-1,
            1:-1,
            :-2,
        ]
    ) / (dz * dz)

    return (
        d2x
        + d2y
        + d2z
    )


def _cem43_increment(
    temperature_celsius: np.ndarray,
    dt_seconds: float,
) -> np.ndarray:
    """Calculate one CEM43 thermal-dose increment."""

    temperature = np.asarray(
        temperature_celsius,
        dtype=float,
    )

    rate = np.where(
        temperature < 43.0,
        0.5,
        0.25,
    )

    dt_minutes = (
        float(dt_seconds)
        / 60.0
    )

    return (
        rate
        ** (
            43.0 - temperature
        )
    ) * dt_minutes


def _solve_bioheat(
    initial_temperature: np.ndarray,
    laser_source: np.ndarray,
    csf_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    *,
    simulation_time_seconds: float,
    time_step_seconds: float,
    tissue_density: float,
    tissue_specific_heat: float,
    tissue_conductivity: float,
    blood_density: float,
    blood_specific_heat: float,
    blood_perfusion: float,
    blood_temperature_celsius: float,
    csf_density: float,
    csf_specific_heat: float,
    csf_conductivity: float,
    csf_sink_strength: float,
    baseline_temperature_celsius: float,
    include_csf_heat_sink: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the simplified Pennes-style bioheat equation."""

    temperature = initial_temperature.copy()

    dose = np.zeros_like(
        temperature,
        dtype=float,
    )

    thermal_diffusivity = (
        tissue_conductivity
        / (
            tissue_density
            * tissue_specific_heat
        )
    )

    minimum_spacing = min(
        spacing_m
    )

    maximum_stable_dt = (
        minimum_spacing ** 2
        / (
            6.0
            * thermal_diffusivity
        )
    )

    effective_dt = min(
        float(time_step_seconds),
        0.25 * maximum_stable_dt,
    )

    if effective_dt <= 0.0:
        raise ValueError(
            "Calculated thermal time step is invalid."
        )

    step_count = int(
        np.ceil(
            simulation_time_seconds
            / effective_dt
        )
    )

    actual_dt = (
        simulation_time_seconds
        / step_count
    )

    density = np.full_like(
        temperature,
        tissue_density,
        dtype=float,
    )

    specific_heat = np.full_like(
        temperature,
        tissue_specific_heat,
        dtype=float,
    )

    conductivity = np.full_like(
        temperature,
        tissue_conductivity,
        dtype=float,
    )

    if include_csf_heat_sink:
        density = np.where(
            csf_mask > 0.5,
            csf_density,
            density,
        )

        specific_heat = np.where(
            csf_mask > 0.5,
            csf_specific_heat,
            specific_heat,
        )

        conductivity = np.where(
            csf_mask > 0.5,
            csf_conductivity,
            conductivity,
        )

    for _ in range(step_count):
        laplacian = _laplacian(
            temperature,
            spacing_m,
        )

        diffusion = (
            conductivity
            * laplacian
        )

        perfusion = (
            blood_density
            * blood_specific_heat
            * blood_perfusion
            * (
                temperature
                - blood_temperature_celsius
            )
        )

        if include_csf_heat_sink:
            csf_sink = (
                csf_mask
                * csf_sink_strength
                * (
                    temperature
                    - baseline_temperature_celsius
                )
            )
        else:
            csf_sink = np.zeros_like(
                temperature
            )

        temperature_change = (
            diffusion
            + laser_source
            - perfusion
            - csf_sink
        ) / (
            density
            * specific_heat
        )

        temperature = (
            temperature
            + actual_dt
            * temperature_change
        )

        temperature = np.maximum(
            temperature,
            baseline_temperature_celsius,
        )

        dose += _cem43_increment(
            temperature,
            actual_dt,
        )

    return (
        temperature,
        dose,
    )

# thermal field

def calculate_thermal_field(
    litt_ablation: AblationGeometry,
    csf: CSFCompartment,
    *,
    patient_csf_mask: AnatomicalMask | None = None,
    grid_resolution: int = 40,
    margin: float = 12.0,
    thermal_sigma: float | None = None,
    csf_weight: float = 0.65,
    laser_power_watts: float = 10.0,
    simulation_time_seconds: float = 60.0,
    time_step_seconds: float = 0.25,
    baseline_temperature_celsius: float = 37.0,
    tissue_density: float = 1050.0,
    tissue_specific_heat: float = 3600.0,
    tissue_conductivity: float = 0.51,
    blood_density: float = 1060.0,
    blood_specific_heat: float = 3600.0,
    blood_perfusion: float = 0.01,
    blood_temperature_celsius: float = 37.0,
    csf_density: float = 1007.0,
    csf_specific_heat: float = 4180.0,
    csf_conductivity: float = 0.60,
    csf_sink_strength: float = 15000.0,
    dose_normalization: float = 240.0,
) -> ThermalField:
    """
    Calculate a three-dimensional computational LITT thermal field.
    If patient_csf_mask is supplied, the MRI-derived CSF mask is mapped
    directly onto the thermal grid.

    If it is not supplied, the synthetic CSF compartment is used.
    Two simulations are performed:

        WITHOUT CSF heat-sink effect
        WITH CSF heat-sink effect
    Their difference provides the modeled CSF thermal effect.
    """

    if grid_resolution < 12:
        raise ValueError(
            "grid_resolution must be at least 12."
        )

    if margin <= 0.0:
        raise ValueError(
            "margin must be greater than zero."
        )

    if not 0.0 <= csf_weight <= 1.0:
        raise ValueError(
            "csf_weight must lie between 0 and 1."
        )

    if laser_power_watts <= 0.0:
        raise ValueError(
            "laser_power_watts must be greater than zero."
        )

    if simulation_time_seconds <= 0.0:
        raise ValueError(
            "simulation_time_seconds must be greater than zero."
        )

    if time_step_seconds <= 0.0:
        raise ValueError(
            "time_step_seconds must be greater than zero."
        )

    if csf_sink_strength < 0.0:
        raise ValueError(
            "csf_sink_strength cannot be negative."
        )

    trajectory = litt_ablation.trajectory

    entry = _point_array(
        trajectory.entry
    )

    target = _point_array(
        trajectory.target
    )

    center = _point_array(
        csf.center
    )

    radii = np.asarray(
        csf.radii,
        dtype=float,
    ).reshape(-1)

    if radii.size != 3:
        raise ValueError(
            "CSF radii must contain exactly three values."
        )

    if np.any(radii <= 0.0):
        raise ValueError(
            "CSF radii must all be greater than zero."
        )

    trajectory_vector = (
        target - entry
    )

    trajectory_length = float(
        np.linalg.norm(
            trajectory_vector
        )
    )

    if trajectory_length <= 0.0:
        raise ValueError(
            "LITT trajectory must have non-zero length."
        )

    direction = (
        trajectory_vector
        / trajectory_length
    )

    active_length = min(
        float(litt_ablation.active_length),
        trajectory_length,
    )

    active_end = (
        entry
        + direction * active_length
    )

    ablation_radius = float(
        litt_ablation.radius
    )

    if ablation_radius <= 0.0:
        raise ValueError(
            "LITT ablation radius must be greater than zero."
        )

    if thermal_sigma is None:
        thermal_sigma = (
            ablation_radius / 2.0
        )

    minimum = np.minimum(
        entry,
        active_end,
    )

    maximum = np.maximum(
        entry,
        active_end,
    )

    expansion = (
        ablation_radius
        + margin
    )

    minimum = (
        minimum
        - expansion
    )

    maximum = (
        maximum
        + expansion
    )

    if patient_csf_mask is None:
        minimum = np.minimum(
            minimum,
            center - radii - margin,
        )

        maximum = np.maximum(
            maximum,
            center + radii + margin,
        )

    axes = [
        np.linspace(
            minimum[index],
            maximum[index],
            grid_resolution,
        )
        for index in range(3)
    ]

    mesh = np.meshgrid(
        axes[0],
        axes[1],
        axes[2],
        indexing="ij",
    )

    points_mm = np.column_stack(
        [
            mesh[0].ravel(),
            mesh[1].ravel(),
            mesh[2].ravel(),
        ]
    )

    spacing_mm = tuple(
        float(
            axis[1] - axis[0]
        )
        for axis in axes
    )

    spacing_m = tuple(
        value / 1000.0
        for value in spacing_mm
    )

    # csf grid

    if patient_csf_mask is not None:
        csf_grid = map_csf_mask_to_grid(
            patient_csf_mask,
            points_mm,
            grid_shape=(
                grid_resolution,
                grid_resolution,
                grid_resolution,
            ),
        ).astype(float)

    else:
        csf_grid = _synthetic_csf_membership(
            points_mm,
            center,
            radii,
        ).reshape(
            (
                grid_resolution,
                grid_resolution,
                grid_resolution,
            )
        )

    csf_modifier = csf_grid.ravel()

    # laser source grid

    source = _create_laser_source(
        points_mm,
        entry,
        active_end,
        ablation_radius,
        laser_power_watts,
    )

    voxel_volume_m3 = (
        spacing_m[0]
        * spacing_m[1]
        * spacing_m[2]
    )

    source_total = (
        np.sum(source)
        * voxel_volume_m3
    )

    if source_total <= 0.0:
        raise ValueError(
            "Laser source contains no positive energy deposition."
        )

    source = (
        source
        * (
            laser_power_watts
            / source_total
        )
    )

    source_grid = source.reshape(
        (
            grid_resolution,
            grid_resolution,
            grid_resolution,
        )
    )

    initial_temperature = np.full(
        (
            grid_resolution,
            grid_resolution,
            grid_resolution,
        ),
        baseline_temperature_celsius,
        dtype=float,
    )
    # no csf counterfactual simulation

    (
        temperature_without_csf,
        dose_without_csf,
    ) = _solve_bioheat(
        initial_temperature=initial_temperature,
        laser_source=source_grid,
        csf_mask=csf_grid,
        spacing_m=spacing_m,
        simulation_time_seconds=simulation_time_seconds,
        time_step_seconds=time_step_seconds,
        tissue_density=tissue_density,
        tissue_specific_heat=tissue_specific_heat,
        tissue_conductivity=tissue_conductivity,
        blood_density=blood_density,
        blood_specific_heat=blood_specific_heat,
        blood_perfusion=blood_perfusion,
        blood_temperature_celsius=blood_temperature_celsius,
        csf_density=csf_density,
        csf_specific_heat=csf_specific_heat,
        csf_conductivity=csf_conductivity,
        csf_sink_strength=csf_sink_strength,
        baseline_temperature_celsius=baseline_temperature_celsius,
        include_csf_heat_sink=False,
    )
    # with csf simulation
    (
        temperature,
        thermal_dose,
    ) = _solve_bioheat(
        initial_temperature=initial_temperature,
        laser_source=source_grid,
        csf_mask=csf_grid,
        spacing_m=spacing_m,
        simulation_time_seconds=simulation_time_seconds,
        time_step_seconds=time_step_seconds,
        tissue_density=tissue_density,
        tissue_specific_heat=tissue_specific_heat,
        tissue_conductivity=tissue_conductivity,
        blood_density=blood_density,
        blood_specific_heat=blood_specific_heat,
        blood_perfusion=blood_perfusion,
        blood_temperature_celsius=blood_temperature_celsius,
        csf_density=csf_density,
        csf_specific_heat=csf_specific_heat,
        csf_conductivity=csf_conductivity,
        csf_sink_strength=csf_sink_strength,
        baseline_temperature_celsius=baseline_temperature_celsius,
        include_csf_heat_sink=True,
    )

    csf_temperature_reduction = (
        temperature_without_csf
        - temperature
    )

    csf_temperature_reduction = np.maximum(
        csf_temperature_reduction,
        0.0,
    )

    risk = np.clip(
        thermal_dose
        / dose_normalization,
        0.0,
        1.0,
    )

    return ThermalField(
        points=points_mm,
        intensity=source,
        csf_modifier=csf_modifier,
        risk=risk,
        temperature=temperature.ravel(),
        temperature_without_csf=(
            temperature_without_csf.ravel()
        ),
        thermal_dose=thermal_dose.ravel(),
        thermal_dose_without_csf=(
            dose_without_csf.ravel()
        ),
        csf_temperature_reduction=(
            csf_temperature_reduction.ravel()
        ),
        grid_shape=(
            grid_resolution,
            grid_resolution,
            grid_resolution,
        ),
        spacing=spacing_mm,
        origin=(
            float(axes[0][0]),
            float(axes[1][0]),
            float(axes[2][0]),
        ),
        simulation_time_seconds=float(
            simulation_time_seconds
        ),
        time_step_seconds=float(
            time_step_seconds
        ),
        laser_power_watts=float(
            laser_power_watts
        ),
        baseline_temperature_celsius=float(
            baseline_temperature_celsius
        ),
        csf_sink_strength=float(
            csf_sink_strength
        ),
    )

# convenience alias for backwards compatibility with the original thermal-field API
def calculate_thermal_risk_field(
    litt_ablation: AblationGeometry,
    csf: CSFCompartment,
    **kwargs,
) -> ThermalField:
    """Backwards-compatible alias for the thermal-field API."""

    return calculate_thermal_field(
        litt_ablation,
        csf,
        **kwargs,
    )