"""
ThermoSEEG_RC.py
================
Trajectory Planning Prototype with a Simplified Static Gaussian Thermal Approximation
(Release Candidate v5)

Architecture
------------
1. Heuristic brain-region ROI (Otsu + opening to strip thin scalp/skull rings + largest
   component + closing + hole fill) is the ALLOWED region. Target, Entry and the whole trajectory are validated against it.
2. High-signal risk regions = top X% brightest pixels INSIDE the ROI (intensity heuristic,
   NOT an anatomical segmentation of vessels, CSF or eloquent cortex).
3. Dual-Mask System: raw risk mask (thermal attenuation) vs. dilated mask (collision).
4. Single threshold source (_derive_threshold) shared by base run and sensitivity analysis.
5. Sensitivity engine: perturbs the threshold with a FIXED ROI; verdict stability first,
   clearance variability second.
6. Auto-Optimizer: searches the ROI edge, honouring the same rules as the validator
   (inside ROI, outside risk mask, minimum clearance, path stays inside ROI).
7. Thermal model: simplified STATIC Gaussian approximation parameterised by laser power.
   Duration scales the CEM43 dose only. It is NOT a prediction of actual tissue temperature.
   "Thermal attenuation" is applied around intensity-based risk regions (no CSF/vessel model).
8. Live navigation simulation isolated in an st.fragment.
9. Click selection uses a nonce so stale clicks are never re-consumed.
"""

import hashlib
import json
import time
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICK_SELECTION_AVAILABLE = True
except ImportError:
    CLICK_SELECTION_AVAILABLE = False


# ============================================================================
# SECTION 0: CONSTANTS & CONFIGURATION
# ============================================================================

# --- Physics & Biology Constants ---
BASELINE_BODY_TEMP_C = 37.0
CEM43_THRESHOLD_TEMP_C = 43.0
CEM43_R_HIGH = 0.5
CEM43_R_LOW = 0.25
CEM43_ABLATION_THRESHOLD_MIN = 240.0   # Commonly cited necrosis dose
LITT_GAUSSIAN_SIGMA_BASE_PX = 8.0
LITT_POWER_TO_DELTA_T_FACTOR = 4.8
RISK_ATTENUATION_FACTOR = 0.30         # attenuation near intensity-based risk regions
RISK_ATTENUATION_SIGMA_PX = 4.0
MAX_PHYSICAL_TEMP_C = 100.0            # Tissue boiling; model invalid above this
TEMP_DISPLAY_MAX_C = 80.0
CEM43_DISPLAY_MAX_MIN = 480.0          # Heatmap saturation so huge doses stay readable
# (MAX_PHYSICAL_TEMP_C - BASELINE) / FACTOR = 13.1 W -> above this the model is invalid
MAX_LASER_POWER_W = 13.0

# --- Safety & Geometry Constants ---
DANGER_CLEARANCE_PX = 10.0
WARNING_CLEARANCE_PX = 25.0
MAX_SAFE_CLEARANCE_PX = 60.0
DEFAULT_DILATION_RADIUS_PX = 5
ROI_ENTRY_EDGE_TOL_PX = 6.0     # entry deeper than this inside the ROI triggers a warning
ROI_PATH_TOLERANCE_PX = 2       # ROI dilation used for path containment (edge pixels / rasterisation)

# --- Segmentation (heuristic) ---
RISK_PERCENTILE_DEFAULT = 97    # heuristic starting point, NOT a scientific constant
ROI_CLOSE_KERNEL_PX = 9
ROI_STRIP_RADIUS_DEFAULT_PX = 7   # opening radius that removes thin scalp/skull rings
RISK_MASK_MAX_FRACTION = 0.15     # risk mask above this share of the ROI = threshold sits on a tissue plateau

# --- UI Defaults ---
DEFAULT_LASER_POWER_W = 10.0
DEFAULT_ABLATION_TIME_SEC = 30.0
DEFAULT_SEEG_CONTACTS_COUNT = 8
DEFAULT_SEEG_SPACING_PX = 10.0

# --- Sensitivity Analysis Parameters ---
SENSITIVITY_THRESHOLD_SHIFT = 5

TOOL_VERSION = "ThermoSEEG_RC_v5"


# ============================================================================
# SECTION 1: TYPE DEFINITIONS & HELPERS
# ============================================================================

PointTuple = Tuple[int, int]
MaskArray = np.ndarray
FloatArray = np.ndarray


def _streamlit_version() -> Tuple[int, int]:
    parts = st.__version__.split(".")[:2]
    out = []
    for p in parts:
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 2:
        out.append(0)
    return out[0], out[1]


# Newer Streamlit deprecates use_container_width in favour of width="stretch".
STRETCH: Dict[str, Any] = (
    {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}
)


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def get_safe_color(status: str) -> Tuple[int, int, int]:
    colors = {
        "SAFE": (0, 200, 0),
        "WARNING": (0, 215, 255),
        "DANGER": (0, 0, 255),
        "ERROR": (255, 0, 0),
    }
    return colors.get(status, (255, 255, 255))


def generate_config_signature(
    file_hash: str,
    use_otsu: bool,
    manual_thresh: int,
    risk_percentile: float,
    roi_strip_px: int,
    min_area: int,
    dilation_rad: int,
    target_pt: Optional[PointTuple],
    entry_pt: Optional[PointTuple],
    proc_mode: str,
    laser_power: float,
    abl_time: float,
    num_contacts: int,
    spacing_px: float,
) -> str:
    sig_str = (
        f"{file_hash}|{use_otsu}|{manual_thresh}|{risk_percentile}|{roi_strip_px}|{min_area}|{dilation_rad}|"
        f"{target_pt}|{entry_pt}|{proc_mode}|{laser_power}|{abl_time}|"
        f"{num_contacts}|{spacing_px}"
    )
    return hashlib.md5(sig_str.encode()).hexdigest()


# ============================================================================
# SECTION 2: GEOMETRY & TRAJECTORY HELPERS
# ============================================================================

def bresenham_line(p1: PointTuple, p2: PointTuple) -> List[PointTuple]:
    x1, y1 = p1
    x2, y2 = p2
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1
    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def calculate_geometry(entry: PointTuple, target: PointTuple) -> Dict[str, float]:
    ex, ey = entry
    tx, ty = target
    length = float(np.hypot(tx - ex, ty - ey))
    angle_rad = np.arctan2(ty - ey, tx - ex)
    angle_deg = float(np.degrees(angle_rad))
    return {"length": length, "angle": angle_deg}


def _in_bounds_xy(line_points: List[PointTuple], w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(line_points, dtype=int)
    if pts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    xs, ys = pts[:, 0], pts[:, 1]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return xs[ok], ys[ok]


def check_collision(mask: MaskArray, line_points: List[PointTuple]) -> bool:
    h, w = mask.shape
    xs, ys = _in_bounds_xy(line_points, w, h)
    if xs.size == 0:
        return False
    return bool(np.any(mask[ys, xs] > 0))


def min_clearance_along_line(dist_map: FloatArray, line_points: List[PointTuple]) -> float:
    h, w = dist_map.shape
    xs, ys = _in_bounds_xy(line_points, w, h)
    if xs.size == 0:
        return 0.0
    return float(dist_map[ys, xs].min())


def check_roi_containment(roi_allowed: MaskArray, line_points: List[PointTuple]) -> int:
    """Number of trajectory pixels that fall OUTSIDE the allowed ROI (out-of-image counts too)."""
    h, w = roi_allowed.shape
    pts = np.asarray(line_points, dtype=int)
    if pts.size == 0:
        return 0
    xs, ys = pts[:, 0], pts[:, 1]
    inside_img = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    outside = int(np.count_nonzero(~inside_img))
    outside += int(np.count_nonzero(roi_allowed[ys[inside_img], xs[inside_img]] == 0))
    return outside


def build_cumulative_depths(points: List[PointTuple]) -> List[float]:
    depths = [0.0]
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        depths.append(depths[-1] + float(np.hypot(x2 - x1, y2 - y1)))
    return depths


def get_telemetry(
    points: List[PointTuple],
    dist_map: FloatArray,
    idx: int,
    cum_depths: List[float],
) -> Dict[str, Any]:
    idx = int(clamp(int(idx), 0, len(points) - 1))
    x, y = points[idx]
    total_len = cum_depths[-1] if cum_depths[-1] > 0 else 1.0
    curr_depth = cum_depths[idx]
    prog = (curr_depth / total_len) * 100

    h, w = dist_map.shape
    d_to_struct = float(dist_map[y, x]) if (0 <= x < w and 0 <= y < h) else 0.0

    return {
        "position": (int(x), int(y)),
        "current_depth": curr_depth,
        "total_length": total_len,
        "progress_pct": round(prog, 1),
        "distance_to_structure": d_to_struct,
        "proximity_alert": d_to_struct < DANGER_CLEARANCE_PX,
    }


# ============================================================================
# SECTION 3: SEGMENTATION (Brain ROI + High-Signal Risk Regions)
# ============================================================================

def _brain_roi(
    gray_blurred: np.ndarray, strip_radius_px: int = ROI_STRIP_RADIUS_DEFAULT_PX
) -> np.ndarray:
    """
    Heuristic brain-region ROI extraction:
      Otsu -> OPENING (removes thin scalp/skull rings and bridges) -> largest component
      -> closing -> hole fill.
    Opening runs BEFORE closing on purpose: closing first would bridge the dark skull gap
    and fuse scalp fat to the brain. Defines the ALLOWED region (not an obstacle).
    NOT a validated brain segmentation; the largest component is not guaranteed to be
    brain tissue, and the approach assumes an MRI-like contrast (CT bone is not handled).
    """
    _, roi = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if strip_radius_px > 0:
        k = int(strip_radius_px) * 2 + 1
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    kernel = np.ones((ROI_CLOSE_KERNEL_PX, ROI_CLOSE_KERNEL_PX), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        roi = np.where(lab == big, 255, 0).astype(np.uint8)

    # Fill internal holes (dark ventricles/sulci) so the ROI is one solid allowed region.
    # Filling is appropriate HERE (ROI), unlike the obstacle mask.
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(roi)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _derive_threshold(
    gray_blurred: np.ndarray,
    roi: np.ndarray,
    use_otsu: bool,
    manual_threshold: int,
    risk_percentile: float,
) -> float:
    """Single source of truth for the risk threshold (base run AND sensitivity)."""
    if not use_otsu:
        return float(manual_threshold)
    vals = gray_blurred[roi > 0]
    if vals.size == 0:
        return 255.0
    return float(np.percentile(vals, risk_percentile))


def _compute_masks_core(
    gray_blurred: np.ndarray,
    use_otsu: bool,
    manual_threshold: int,
    min_area: int,
    dilation_radius: int,
    risk_percentile: float = RISK_PERCENTILE_DEFAULT,
    roi: Optional[np.ndarray] = None,
    threshold_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (raw_mask, dilated_mask).
    raw_mask: high-signal risk regions inside the brain ROI (thermal physics).
    dilated_mask: buffered risk regions (trajectory safety / collision).
    `threshold_override` lets the sensitivity analysis shift the threshold while
    keeping the exact same ROI.
    """
    if roi is None:
        roi = _brain_roi(gray_blurred)

    if threshold_override is not None:
        t = float(threshold_override)
    else:
        t = _derive_threshold(gray_blurred, roi, use_otsu, manual_threshold, risk_percentile)

    binary = np.where((gray_blurred >= t) & (roi > 0), 255, 0).astype(np.uint8)

    # Area filter via connected components (no contour filling -> no solid-ring problem).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    if n > 1:
        keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        if keep.size:
            cleaned[np.isin(lab, keep)] = 255

    if dilation_radius > 0:
        k_size = dilation_radius * 2 + 1
        kernel = np.ones((k_size, k_size), np.uint8)
        dilated = cv2.dilate(cleaned, kernel, iterations=1)
    else:
        dilated = cleaned.copy()

    return cleaned, dilated


@st.cache_data(show_spinner=False)
def load_and_preprocess_image(
    file_bytes: bytes,
    threshold_value: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    roi_strip_px: int,
) -> Dict[str, Any]:
    try:
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        original = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError("Invalid image format.")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        roi = _brain_roi(blurred, roi_strip_px)
        raw_mask, dilated_mask = _compute_masks_core(
            blurred, use_otsu, threshold_value, min_area, dilation_radius,
            risk_percentile=risk_percentile, roi=roi,
        )

        k_roi = ROI_PATH_TOLERANCE_PX * 2 + 1
        roi_allowed = cv2.dilate(roi, np.ones((k_roi, k_roi), np.uint8))
        roi_depth = distance_transform_edt(roi > 0)   # px inside the ROI (0 outside)

        return {
            "original": original,
            "gray": gray,
            "blurred": blurred,
            "roi": roi,
            "roi_allowed": roi_allowed,
            "roi_depth": roi_depth,
            "raw_structure_mask": raw_mask,   # thermal physics
            "structure_mask": dilated_mask,   # safety / collision
        }
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return {}


def _distance_map(structure_mask: MaskArray) -> FloatArray:
    if structure_mask.max() == 0:
        h, w = structure_mask.shape
        return np.full((h, w), float(np.hypot(h, w)))
    return distance_transform_edt(~(structure_mask > 0))


@st.cache_data(show_spinner=False)
def compute_distance_map_cached(structure_mask: MaskArray) -> FloatArray:
    """Euclidean pixel distance to nearest (dilated) risk region."""
    return _distance_map(structure_mask)


# ============================================================================
# SECTION 4: SENSITIVITY ANALYSIS ENGINE (Collision Aware)
# ============================================================================

def perform_sensitivity_analysis(
    blurred_gray: np.ndarray,
    roi: np.ndarray,
    base_manual_threshold: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    entry: PointTuple,
    target: PointTuple,
    leaves_roi: bool = False,
) -> Dict[str, Any]:
    """
    Tests stability of the trajectory under threshold perturbations (T-shift, T, T+shift)
    with the ROI held fixed.

    Confidence reflects whether the *verdict* is stable:
      - collision appears/disappears across scenarios  -> LOW
      - safety status changes (SAFE/WARNING/DANGER)    -> MODERATE
      - otherwise graded by clearance coefficient of variation.
    A consistently colliding trajectory is stable (and already flagged DANGER),
    so it is not penalised here.
    """
    base_thresh = _derive_threshold(
        blurred_gray, roi, use_otsu, base_manual_threshold, risk_percentile
    )

    thresholds_to_test = [
        base_thresh - SENSITIVITY_THRESHOLD_SHIFT,
        base_thresh,
        base_thresh + SENSITIVITY_THRESHOLD_SHIFT,
    ]

    line_pts = bresenham_line(entry, target)
    clearances: List[float] = []
    statuses: List[str] = []
    collision_flags: List[bool] = []
    risk_fractions: List[float] = []
    roi_px = max(1, int(np.count_nonzero(roi)))

    for t_val in thresholds_to_test:
        t_clamped = float(max(0.0, min(255.0, t_val)))

        temp_raw, temp_dilated = _compute_masks_core(
            blurred_gray, False, 0, min_area, dilation_radius,
            roi=roi, threshold_override=t_clamped,
        )
        risk_fractions.append(float(np.count_nonzero(temp_raw)) / roi_px)
        temp_dist = _distance_map(temp_dilated)

        intersects = check_collision(temp_dilated, line_pts)
        min_d = min_clearance_along_line(temp_dist, line_pts)

        clearances.append(min_d)
        collision_flags.append(intersects)

        # Collision and DANGER are the same status category in the main evaluator.
        if intersects or leaves_roi or min_d < DANGER_CLEARANCE_PX:
            statuses.append("DANGER")
        elif min_d < WARNING_CLEARANCE_PX:
            statuses.append("WARNING")
        else:
            statuses.append("SAFE")

    mean_clearance = float(np.mean(clearances))
    std_clearance = float(np.std(clearances))

    if mean_clearance < 1e-6:
        cv_score = 0.0 if std_clearance < 1e-6 else 1.0
    else:
        cv_score = std_clearance / mean_clearance

    collision_flip = len(set(collision_flags)) > 1
    # The mask itself is unreliable if a +-shift makes it swallow a large part of the ROI
    # (threshold sits on the tissue intensity plateau), regardless of this one trajectory.
    max_risk_fraction = max(risk_fractions) if risk_fractions else 0.0
    mask_unstable = max_risk_fraction > RISK_MASK_MAX_FRACTION
    status_flip = len(set(statuses)) > 1

    if collision_flip or mask_unstable:
        conf_label = "LOW"
    elif status_flip:
        conf_label = "MODERATE"
    elif cv_score < 0.1:
        conf_label = "HIGH"
    elif cv_score < 0.3:
        conf_label = "MODERATE"
    else:
        conf_label = "LOW"

    return {
        "confidence_index": conf_label,
        "variability_score": round(float(cv_score), 3),
        "mean_stable_clearance": round(mean_clearance, 2),
        "status_flipped": status_flip,
        "collision_flipped": collision_flip,
        "mask_unstable": mask_unstable,
        "max_risk_fraction": round(max_risk_fraction, 3),
        "scenario_statuses": statuses,
    }


# ============================================================================
# SECTION 5: VALIDATION & SAFETY ENGINE
# ============================================================================

def validate_target(
    target: PointTuple, structure_mask: MaskArray, roi: MaskArray, w: int, h: int
) -> Tuple[bool, str]:
    if not (0 <= target[0] < w and 0 <= target[1] < h):
        return False, "Target out of bounds."
    if roi[target[1], target[0]] == 0:
        return False, "Target outside the heuristic brain ROI."
    if structure_mask[target[1], target[0]] > 0:
        return False, "Target inside high-signal risk region."
    return True, "Valid Target."


def validate_entry(
    entry: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    w: int,
    h: int,
) -> Tuple[bool, str, bool]:
    """Returns (is_valid, message, is_deep_inside_roi_warning)."""
    if not (0 <= entry[0] < w and 0 <= entry[1] < h):
        return False, "Entry out of bounds.", False

    if roi_allowed[entry[1], entry[0]] == 0:
        return False, "Entry outside the heuristic brain ROI.", False

    if structure_mask[entry[1], entry[0]] > 0:
        return False, "Entry inside high-signal risk region.", False

    d = float(dist_map[entry[1], entry[0]])
    if d < DANGER_CLEARANCE_PX:
        return False, f"Entry too close to risk region ({d:.1f}px < {DANGER_CLEARANCE_PX:.0f}px).", False

    depth = float(roi_depth[entry[1], entry[0]])
    if depth > ROI_ENTRY_EDGE_TOL_PX:
        return True, f"Warning: Entry is {depth:.0f}px inside the brain ROI (not on its edge).", True

    return True, "", False


def evaluate_safety_metrics(
    min_dist_px: float,
    intersects: bool,
    leaves_roi: bool,
    blurred_gray: np.ndarray,
    roi: np.ndarray,
    base_manual_threshold: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    entry: PointTuple,
    target: PointTuple,
) -> Dict[str, Any]:
    """
    Status is DANGER on collision, ROI violation, or clearance < DANGER threshold.
    clearance_score_pct is a DISTANCE-based score (min clearance / MAX_SAFE_CLEARANCE),
    NOT a probability of safety.
    """
    if intersects or leaves_roi or min_dist_px < DANGER_CLEARANCE_PX:
        status = "DANGER"
    elif min_dist_px < WARNING_CLEARANCE_PX:
        status = "WARNING"
    else:
        status = "SAFE"

    clearance_score = clamp((min_dist_px / MAX_SAFE_CLEARANCE_PX) * 100, 0, 100)
    if intersects or min_dist_px < DANGER_CLEARANCE_PX:
        clearance_score = 0.0

    sens_res = perform_sensitivity_analysis(
        blurred_gray, roi, base_manual_threshold, use_otsu, risk_percentile,
        min_area, dilation_radius, entry, target, leaves_roi=leaves_roi,
    )

    return {
        "status": status,
        "clearance_score_pct": round(float(clearance_score), 1),
        "min_distance_px": round(float(min_dist_px), 2),
        "intersects": bool(intersects),
        "leaves_roi": bool(leaves_roi),
        "confidence_index": sens_res["confidence_index"],
        "variability_score": sens_res["variability_score"],
        "status_flipped": sens_res["status_flipped"],
        "collision_flipped": sens_res["collision_flipped"],
        "mask_unstable": sens_res["mask_unstable"],
        "max_risk_fraction": sens_res["max_risk_fraction"],
        "color_bgr": get_safe_color(status),
    }


# ============================================================================
# SECTION 6: LITT THERMAL MODEL & SEEG PLANNING
# ============================================================================

def simulate_litt_thermal_field(
    target: PointTuple,
    raw_structure_mask: MaskArray,
    power_watts: float,
    duration_sec: float,
) -> Dict[str, Any]:
    """
    Simplified STATIC Gaussian thermal approximation parameterised by laser power.
    Duration affects the CEM43 thermal dose only, NOT the spatial temperature field.
    This is not a transient Pennes bio-heat simulation and does not predict actual
    tissue temperature. Attenuation is applied around intensity-based risk regions only
    (there is no CSF or vessel model).
    """
    h, w = raw_structure_mask.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_target = np.hypot(x_coords - target[0], y_coords - target[1])

    sigma = LITT_GAUSSIAN_SIGMA_BASE_PX + (power_watts * 0.4)
    peak_delta_t = power_watts * LITT_POWER_TO_DELTA_T_FACTOR

    raw_delta_t = peak_delta_t * np.exp(-(dist_from_target ** 2) / (2 * sigma ** 2))

    # Attenuation around the raw intensity-based risk regions (not safety buffers).
    risk_influence = gaussian_filter(
        (raw_structure_mask > 0).astype(float), sigma=RISK_ATTENUATION_SIGMA_PX
    )
    max_inf = risk_influence.max()
    risk_influence_norm = risk_influence / max_inf if max_inf > 0 else np.zeros_like(risk_influence)

    reduction_factor = 1.0 - (RISK_ATTENUATION_FACTOR * risk_influence_norm)
    adjusted_delta_t = raw_delta_t * reduction_factor

    temp_map = BASELINE_BODY_TEMP_C + adjusted_delta_t
    max_temp = float(np.max(temp_map))
    model_valid = max_temp <= MAX_PHYSICAL_TEMP_C
    model_warning = ""
    if not model_valid:
        model_warning = (
            f"Peak temperature {max_temp:.0f} °C exceeds {MAX_PHYSICAL_TEMP_C:.0f} °C (tissue boiling). "
            "The Gaussian model is not physically meaningful here; dose maths cap temperature at "
            f"{MAX_PHYSICAL_TEMP_C:.0f} °C. Reduce laser power."
        )

    temp_phys = np.minimum(temp_map, MAX_PHYSICAL_TEMP_C)
    temp_display = np.clip(temp_phys, BASELINE_BODY_TEMP_C, TEMP_DISPLAY_MAX_C)

    cem43_map = np.zeros_like(temp_phys, dtype=float)
    minutes = duration_sec / 60.0

    mask_high = temp_phys >= CEM43_THRESHOLD_TEMP_C
    cem43_map[mask_high] = minutes * (CEM43_R_HIGH ** (CEM43_THRESHOLD_TEMP_C - temp_phys[mask_high]))

    mask_mid = (temp_phys >= BASELINE_BODY_TEMP_C) & (temp_phys < CEM43_THRESHOLD_TEMP_C)
    cem43_map[mask_mid] = minutes * (CEM43_R_LOW ** (CEM43_THRESHOLD_TEMP_C - temp_phys[mask_mid]))

    max_cem43 = float(np.max(cem43_map))
    significant = cem43_map > 0.01
    mean_cem43 = float(np.mean(cem43_map[significant])) if np.any(significant) else 0.0
    risk_attenuation_effect = float(np.max(raw_delta_t - adjusted_delta_t))

    ablation_area_px = int(np.sum(cem43_map >= CEM43_ABLATION_THRESHOLD_MIN))
    structure_pixels = raw_structure_mask > 0
    structure_max_cem43 = float(np.max(cem43_map[structure_pixels])) if np.any(structure_pixels) else 0.0

    return {
        "temp_map": temp_display,
        "cem43_map": cem43_map,
        "max_temp_c": round(max_temp, 2),
        "max_cem43_min": round(max_cem43, 2),
        "mean_cem43_min": round(mean_cem43, 2),
        "risk_attenuation_c": round(risk_attenuation_effect, 2),
        "ablation_area_px": ablation_area_px,
        "structure_max_cem43_min": round(structure_max_cem43, 2),
        "model_valid": model_valid,
        "model_warning": model_warning,
    }


def compute_seeg_contacts(
    entry: PointTuple,
    target: PointTuple,
    traj_length_px: float,
    num_contacts_req: int,
    spacing_px: float,
) -> Dict[str, Any]:
    ex, ey = entry
    tx, ty = target

    if traj_length_px <= 0:
        return {
            "contacts": [],
            "fits_within_trajectory": False,
            "required_span_px": 0.0,
            "num_contacts_placed": 0,
        }

    length = traj_length_px
    ux, uy = (tx - ex) / length, (ty - ey) / length

    required_span = (num_contacts_req - 1) * spacing_px
    fits = required_span <= length

    contacts = []
    for i in range(num_contacts_req):
        depth = i * spacing_px
        if depth > length:
            break
        cx = tx - ux * depth
        cy = ty - uy * depth
        contacts.append({
            "id": i + 1,
            "pos": (int(round(cx)), int(round(cy))),
            "depth_px": round(depth, 1),
        })

    return {
        "contacts": contacts,
        "fits_within_trajectory": fits,
        "required_span_px": round(required_span, 1),
        "num_contacts_placed": len(contacts),
    }


# ============================================================================
# SECTION 7: AUTO ENTRY OPTIMIZER
# ============================================================================

def _evaluate_entry(
    pt: PointTuple,
    target: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
    diagonal: float,
) -> Dict[str, Any]:
    line_pts = bresenham_line(pt, target)
    intersects = check_collision(structure_mask, line_pts)
    leaves_roi = check_roi_containment(roi_allowed, line_pts) > 0
    min_d = min_clearance_along_line(dist_map, line_pts)
    geo = calculate_geometry(pt, target)
    norm_len = geo["length"] / diagonal if diagonal > 0 else 0.0

    safety_reward = clamp(min_d / MAX_SAFE_CLEARANCE_PX, 0, 1) * 100
    score = safety_reward - norm_len * 15.0
    return {
        "entry": pt,
        "score": score,
        "min_dist": min_d,
        "intersects": intersects,
        "leaves_roi": leaves_roi,
        "geo": geo,
        "line_pts": line_pts,
    }


def _entry_is_admissible(
    pt: PointTuple,
    target: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
) -> bool:
    """Same rules the manual validator enforces."""
    h, w = structure_mask.shape
    px, py = pt
    if not (0 <= px < w and 0 <= py < h):
        return False
    if pt == target:
        return False
    if roi_allowed[py, px] == 0:
        return False
    if structure_mask[py, px] > 0:
        return False
    if dist_map[py, px] < DANGER_CLEARANCE_PX:
        return False
    return True


def _roi_edge_candidates(roi: MaskArray, n_candidates: int) -> List[PointTuple]:
    """Evenly spaced points along the outer contour of the largest ROI component."""
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    if len(pts) == 0:
        return []
    idx = np.unique(np.linspace(0, len(pts) - 1, min(n_candidates, len(pts))).astype(int))
    return [(int(pts[i][0]), int(pts[i][1])) for i in idx]


def find_optimal_entry_point(
    structure_mask: MaskArray,
    target: PointTuple,
    dist_map: FloatArray,
    roi: MaskArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    num_candidates_total: int = 72,
) -> Optional[Dict[str, Any]]:
    """
    Searches the heuristic brain-ROI edge for the best entry point: inside the ROI,
    outside the risk mask, and the whole path stays inside the ROI with min clearance
    >= DANGER_CLEARANCE_PX (so the result is never rated DANGER by the evaluator). Returns None if none exists.
    """
    h, w = structure_mask.shape
    diagonal = float(np.hypot(w, h))

    results = []
    seen = set()
    for pt in _roi_edge_candidates(roi, num_candidates_total):
        if pt in seen:
            continue
        seen.add(pt)
        if not _entry_is_admissible(pt, target, structure_mask, dist_map, roi_allowed):
            continue
        r = _evaluate_entry(pt, target, structure_mask, dist_map, roi_allowed, diagonal)
        # Never return an "optimal" path that the evaluator would immediately rate DANGER.
        if r["intersects"] or r["leaves_roi"] or r["min_dist"] < DANGER_CLEARANCE_PX:
            continue
        results.append(r)

    if not results:
        return None

    results.sort(key=lambda r: (-r["score"], r["geo"]["length"], r["entry"]))
    best = results[0]

    # Local refinement (stays near the ROI edge and keeps all admissibility rules)
    bx, by = best["entry"]
    radius = 3
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = bx + dx, by + dy
            cand = (nx, ny)
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if roi_depth[ny, nx] > ROI_ENTRY_EDGE_TOL_PX:
                continue
            if not _entry_is_admissible(cand, target, structure_mask, dist_map, roi_allowed):
                continue
            r = _evaluate_entry(cand, target, structure_mask, dist_map, roi_allowed, diagonal)
            if (not r["intersects"] and not r["leaves_roi"]
                    and r["min_dist"] >= DANGER_CLEARANCE_PX and r["score"] > best["score"]):
                best = r

    return {"best": best, "candidates_evaluated": len(results)}


# ============================================================================
# SECTION 8: VISUALIZATION UTILITIES
# ============================================================================

def draw_trajectory_overlay(
    img: MaskArray,
    entry: PointTuple,
    target: PointTuple,
    color: Tuple[int, int, int],
    seeg_contacts: Optional[List[Dict]] = None,
    probe_pos: Optional[PointTuple] = None,
) -> MaskArray:
    out = img.copy()
    cv2.line(out, entry, target, color, 2, cv2.LINE_AA)
    cv2.circle(out, entry, 6, (255, 0, 0), -1)
    cv2.circle(out, target, 6, (0, 255, 0), -1)

    if seeg_contacts:
        for c in seeg_contacts:
            pos = c["pos"]
            cv2.circle(out, pos, 3, (0, 165, 255), -1)
            cv2.putText(out, str(c["id"]), (pos[0] + 4, pos[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

    if probe_pos:
        cv2.circle(out, probe_pos, 5, (255, 255, 255), -1)
        cv2.circle(out, probe_pos, 7, (0, 0, 0), 1)

    return out


def draw_points_preview(
    img: MaskArray, target: Optional[PointTuple], entry: Optional[PointTuple]
) -> MaskArray:
    out = img.copy()
    if target is not None:
        cv2.circle(out, target, 6, (0, 255, 0), -1)
        cv2.circle(out, target, 9, (0, 0, 0), 1)
    if entry is not None:
        cv2.circle(out, entry, 6, (255, 0, 0), -1)
        cv2.circle(out, entry, 9, (0, 0, 0), 1)
    if target is not None and entry is not None:
        cv2.line(out, entry, target, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_heatmap_with_colorbar(
    base_img: MaskArray,
    data_map: FloatArray,
    vmin: float,
    vmax: float,
    colormap: int,
    title: str,
) -> MaskArray:
    h, w = data_map.shape
    if vmax <= vmin:
        vmax = vmin + 1e-6
    clipped = np.clip(data_map, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    heat_colored = cv2.applyColorMap(normalized, colormap)

    alpha = 0.5
    blended = cv2.addWeighted(base_img, 1 - alpha, heat_colored, alpha, 0)

    cb_height = h
    cb_width = 40
    cb_img = np.zeros((cb_height, cb_width, 3), dtype=np.uint8)
    for i in range(cb_height):
        val = 255 - int((i / cb_height) * 255)
        cb_img[i, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), colormap)[0, 0]

    cv2.putText(cb_img, f"{vmax:.0f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(cb_img, f"{vmin:.0f}", (3, cb_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    final_img = np.hstack([blended, cb_img])

    title_bar = np.zeros((25, final_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return np.vstack([title_bar, final_img])


def show_bgr(img_bgr: MaskArray, caption: Optional[str] = None) -> None:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, **STRETCH)


def build_segmentation_overlay(
    original: MaskArray, roi: MaskArray, raw_mask: MaskArray, dilated_mask: MaskArray
) -> MaskArray:
    """Red = risk regions, Orange = safety buffer, Green outline = brain ROI."""
    vis = original.copy()
    vis[raw_mask > 0] = [0, 0, 255]
    buffer_ring = (dilated_mask > 0) & (raw_mask == 0)
    vis[buffer_ring] = [0, 165, 255]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis


# ============================================================================
# SECTION 9: PLAN BUILDER (pure function, no UI side effects)
# ============================================================================

def build_plan(
    entry: PointTuple,
    target: PointTuple,
    struct_mask: MaskArray,
    raw_mask: MaskArray,
    dist_map: FloatArray,
    roi: MaskArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    blurred_gray: np.ndarray,
    w: int,
    h: int,
    procedure_mode: str,
    use_otsu: bool,
    manual_thresh: int,
    risk_percentile: float,
    min_area: int,
    dilation_rad: int,
    laser_power: float,
    ablation_time: float,
    num_contacts: int,
    contact_spacing: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (results, error_message)."""
    if entry == target:
        return None, "Entry and Target points must be different."

    ok_t, msg_t = validate_target(target, struct_mask, roi, w, h)
    if not ok_t:
        return None, f"Invalid Target: {msg_t}"

    ok_e, msg_e, warn_edge = validate_entry(entry, struct_mask, dist_map, roi_allowed, roi_depth, w, h)
    if not ok_e:
        return None, f"Invalid Entry: {msg_e}"

    line_pts = bresenham_line(entry, target)
    intersects = check_collision(struct_mask, line_pts)
    roi_outside_px = check_roi_containment(roi_allowed, line_pts)
    leaves_roi = roi_outside_px > 0
    min_d = min_clearance_along_line(dist_map, line_pts)
    geo = calculate_geometry(entry, target)

    safety = evaluate_safety_metrics(
        min_d, intersects, leaves_roi, blurred_gray, roi, manual_thresh, use_otsu, risk_percentile,
        min_area, dilation_rad, entry, target,
    )

    results: Dict[str, Any] = {
        "target": target,
        "entry": entry,
        "geometry": geo,
        "safety": safety,
        "line_points": line_pts,
        "entry_warning": msg_e if warn_edge else "",
        "roi_outside_px": roi_outside_px,
        "thermal": None,
        "seeg": None,
    }

    if procedure_mode == "LITT Thermal Ablation":
        results["thermal"] = simulate_litt_thermal_field(target, raw_mask, laser_power, ablation_time)
    else:
        results["seeg"] = compute_seeg_contacts(entry, target, geo["length"], num_contacts, contact_spacing)

    return results, None


# ============================================================================
# SECTION 10: LIVE SIMULATION (isolated fragment)
# ============================================================================

HAS_FRAGMENT = hasattr(st, "fragment")
fragment_decorator = st.fragment if HAS_FRAGMENT else (lambda f: f)


def _rerun_sim() -> None:
    """Rerun only the simulation fragment when supported."""
    if HAS_FRAGMENT:
        try:
            st.rerun(scope="fragment")
            return
        except TypeError:
            pass
    st.rerun()


@fragment_decorator
def live_simulation(res: Dict[str, Any], original_img: MaskArray, dist_map: FloatArray) -> None:
    line_pts = res["line_points"]
    n_pts = len(line_pts)

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1, 1, 1, 3])
    with ctrl_c1:
        if st.button("▶️ Play", key="sim_play_btn"):
            if st.session_state.sim_step >= n_pts - 1:
                st.session_state.sim_step = 0
            st.session_state.sim_playing = True
    with ctrl_c2:
        if st.button("⏸️ Pause", key="sim_pause_btn"):
            st.session_state.sim_playing = False
    with ctrl_c3:
        if st.button("⏹️ Reset", key="sim_reset_btn"):
            st.session_state.sim_playing = False
            st.session_state.sim_step = 0
    with ctrl_c4:
        speed = st.slider("Speed", 1, 10, 5, key="sim_speed")

    curr_idx = int(clamp(st.session_state.sim_step, 0, n_pts - 1))
    probe_pos = line_pts[curr_idx]

    sim_frame = draw_trajectory_overlay(
        original_img,
        res["entry"],
        res["target"],
        res["safety"]["color_bgr"],
        seeg_contacts=res["seeg"]["contacts"] if res["seeg"] else None,
        probe_pos=probe_pos,
    )

    cum_depths = build_cumulative_depths(line_pts)
    telem = get_telemetry(line_pts, dist_map, curr_idx, cum_depths)

    sim_col, tele_col = st.columns([2, 1])
    with sim_col:
        show_bgr(sim_frame, caption=f"Probe Position: {telem['progress_pct']}%")
    with tele_col:
        st.metric("Depth (px)", f"{telem['current_depth']:.1f} / {telem['total_length']:.1f}")
        st.progress(float(clamp(telem["progress_pct"] / 100, 0.0, 1.0)))
        st.metric("Dist. to Risk Region (px)", f"{telem['distance_to_structure']:.1f}")
        if telem["proximity_alert"]:
            st.error("🚨 PROXIMITY ALERT!")
        else:
            st.success("✅ Safe Clearance")

    # Advance AFTER drawing so every frame is actually shown.
    if st.session_state.sim_playing:
        time.sleep(0.15 / speed)
        steps_per_tick = max(1, n_pts // 200)
        if curr_idx < n_pts - 1:
            st.session_state.sim_step = min(curr_idx + steps_per_tick, n_pts - 1)
        else:
            st.session_state.sim_playing = False
        _rerun_sim()


# ============================================================================
# SECTION 11: MAIN STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="ThermoSEEG RC", layout="wide")

st.title("🧠 ThermoSEEG Release Candidate")
st.caption("Trajectory Planning Prototype with Simplified Static Gaussian Thermal Approximation")

disclaimer_text = (
    "**⚠️ MEDICAL DISCLAIMER:** This software is an algorithmic simulation prototype intended for "
    "research and educational demonstration only. It utilizes simplified heuristic models (STATIC Gaussian "
    "temperature approximation parameterized by laser power; duration affects the CEM43 dose estimate only, "
    "not the spatial temperature field) and does not account for patient-specific anatomy, "
    "anisotropic tissue properties, or real-time physiological feedback. "
    "Risk regions are intensity-based heuristics, not anatomical segmentation of vessels or eloquent cortex. "
    "The brain ROI is a heuristic region extraction (not a validated brain segmentation), and thermal "
    "attenuation is applied around intensity-based risk regions only (no CSF or vessel model). "
    "Trajectories are planned from the brain-ROI (cortical) surface; scalp/skull crossing is not modeled. "
    "**DO NOT USE FOR CLINICAL DECISION MAKING OR SURGICAL GUIDANCE.**"
)
st.warning(disclaimer_text)

with st.sidebar:
    st.header("⚙️ Configuration")
    procedure_mode = st.radio(
        "Procedure Type",
        ["SEEG Electrode Planning", "LITT Thermal Ablation"],
        help="Select the surgical modality to configure parameters.",
    )

    st.divider()
    st.subheader("Parameters")

    if procedure_mode == "LITT Thermal Ablation":
        laser_power = st.slider(
            "Laser Power (W)", 2.0, MAX_LASER_POWER_W,
            min(DEFAULT_LASER_POWER_W, MAX_LASER_POWER_W), 0.5,
            help="Capped where the static Gaussian model stays below tissue boiling (~13 W).",
        )
        ablation_time = st.slider("Duration (s)", 5.0, 120.0, DEFAULT_ABLATION_TIME_SEC, 5.0)
        st.caption("Static spatial model: duration scales CEM43 dose, not the temperature field.")
        num_contacts = 0
        contact_spacing = 0.0
    else:
        num_contacts = st.slider("Num Contacts", 2, 18, DEFAULT_SEEG_CONTACTS_COUNT)
        contact_spacing = st.slider("Spacing (px)", 2.0, 30.0, DEFAULT_SEEG_SPACING_PX, 1.0)
        laser_power = 0.0
        ablation_time = 0.0

    st.divider()
    st.subheader("Image Processing (Risk Regions)")
    use_otsu = st.checkbox(
        "Auto Threshold (percentile inside brain ROI)", value=True,
        help="Threshold = chosen percentile of pixel intensity inside the brain ROI.",
    )
    risk_percentile_ui = st.slider(
        "High-Signal Percentile (heuristic)", 80, 99, RISK_PERCENTILE_DEFAULT,
        disabled=not use_otsu,
        help="Top X% brightest pixels inside the brain ROI become risk regions. "
             "If the red area is too large, RAISE this value (98-99).",
    )
    manual_thresh_ui = st.slider("Manual Threshold", 0, 255, 140, disabled=use_otsu)
    # Normalise so the cache/signature don't change when a disabled slider is irrelevant.
    manual_thresh = 0 if use_otsu else int(manual_thresh_ui)
    risk_percentile = float(risk_percentile_ui) if use_otsu else float(RISK_PERCENTILE_DEFAULT)
    roi_strip_px = st.slider(
        "Scalp/Skull Strip Radius (px)", 0, 20, ROI_STRIP_RADIUS_DEFAULT_PX,
        help="Opening radius that removes thin scalp/skull rings from the brain ROI. "
             "Raise it until the green ROI outline hugs the brain; 0 disables stripping.",
    )
    min_area = st.slider("Min Risk Region Area (px²)", 0, 200, 30)
    dilation_rad = st.slider(
        "Safety Buffer (px)", 0, 15, DEFAULT_DILATION_RADIUS_PX,
        help="Expands risk regions for trajectory safety.",
    )

    enable_mm = st.checkbox("Show Measurements in mm", value=False)
    pixel_spacing_mm = (
        st.number_input("Pixel Spacing (mm/px)", 0.01, 10.0, 1.0, 0.01) if enable_mm else 1.0
    )

col_upload, col_preview = st.columns([1, 2])

with col_upload:
    uploaded_file = st.file_uploader("Upload MRI/CT Slice", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.info("⬆️ Please upload a medical scan image to begin planning.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()[:10]

with st.spinner("Processing Image..."):
    prep_data = load_and_preprocess_image(
        file_bytes, manual_thresh, use_otsu, risk_percentile, min_area, dilation_rad, roi_strip_px
    )

if not prep_data or "original" not in prep_data:
    st.error("Failed to process image.")
    st.stop()

original_img = prep_data["original"]
roi_mask = prep_data["roi"]                  # heuristic allowed region (NOT an obstacle)
roi_allowed = prep_data["roi_allowed"]       # ROI + small tolerance, for path containment
roi_depth = prep_data["roi_depth"]           # px inside the ROI
raw_mask = prep_data["raw_structure_mask"]   # thermal physics
struct_mask = prep_data["structure_mask"]    # safety / collision
blurred_gray = prep_data["blurred"]
h, w = original_img.shape[:2]

dist_map = compute_distance_map_cached(struct_mask)

with col_preview:
    tab1, tab2 = st.tabs(["Original Scan", "High-Signal Risk Regions"])
    with tab1:
        show_bgr(original_img, caption=f"Slice Dimensions: {w}x{h}px")
    with tab2:
        show_bgr(
            build_segmentation_overlay(original_img, roi_mask, raw_mask, struct_mask),
            caption="Red = high-signal risk regions | Orange = safety buffer | Green outline = heuristic brain-region ROI",
        )
        risk_pct_area = 100.0 * float(np.count_nonzero(raw_mask)) / max(1, int(np.count_nonzero(roi_mask)))
        st.caption(f"Risk regions cover {risk_pct_area:.1f}% of the brain ROI.")
        st.caption("If the green outline includes scalp/skull, raise 'Scalp/Skull Strip Radius'.")
        if risk_pct_area > RISK_MASK_MAX_FRACTION * 100:
            st.warning("Risk mask is large. Raise the High-Signal Percentile (98-99) or the manual threshold.")

st.divider()

# --- State Initialization ---
default_keys = {
    "target_pt": None, "entry_pt": None, "results": None,
    "sim_playing": False, "sim_step": 0,
    "selection_stage": "target", "last_hash": "",
    "click_nonce": 0, "result_sig": None, "auto_info": None,
    "prev_auto": None,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.last_hash != file_hash:
    st.session_state.target_pt = None
    st.session_state.entry_pt = None
    st.session_state.selection_stage = "target"
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0
    st.session_state.last_hash = file_hash
    st.session_state.results = None
    st.session_state.auto_info = None
    st.session_state.click_nonce += 1


def reset_points() -> None:
    st.session_state.target_pt = None
    st.session_state.entry_pt = None
    st.session_state.selection_stage = "target"
    st.session_state.results = None
    st.session_state.auto_info = None
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0
    st.session_state.click_nonce += 1


# --- Interaction UI ---
st.subheader("2️⃣ Define Trajectory Points")

entry_strategy = st.radio("Entry Strategy", ["Manual Selection", "🎯 Smart Auto-Optimal"], horizontal=True)
auto_mode = entry_strategy.startswith("🎯")

# Switching strategy must not carry over a stale entry point (e.g. a manual one shown as "optimal").
if st.session_state.prev_auto is not None and st.session_state.prev_auto != auto_mode:
    st.session_state.entry_pt = None
    st.session_state.auto_info = None
    st.session_state.results = None
    if not auto_mode:
        st.session_state.selection_stage = "target" if st.session_state.target_pt is None else "entry"
        st.session_state.click_nonce += 1
st.session_state.prev_auto = auto_mode

input_method = st.radio(
    "Input Method", ["Click on Image", "Manual Coordinates"], horizontal=True, disabled=auto_mode
)

if auto_mode:
    st.info("Set the Target (inside the ROI), then run Auto-Search to find the safest entry point on the heuristic brain-ROI edge.")

    c_tx, c_ty = st.columns(2)
    default_tx = st.session_state.target_pt[0] if st.session_state.target_pt else int(w * 0.6)
    default_ty = st.session_state.target_pt[1] if st.session_state.target_pt else int(h * 0.6)
    tx = c_tx.number_input("Target X", 0, w - 1, int(default_tx), key=f"atx_{file_hash}")
    ty = c_ty.number_input("Target Y", 0, h - 1, int(default_ty), key=f"aty_{file_hash}")
    new_target = (int(tx), int(ty))
    if new_target != st.session_state.target_pt:
        st.session_state.target_pt = new_target
        # Previous auto entry was computed for a different target.
        st.session_state.entry_pt = None
        st.session_state.auto_info = None

    if st.button("Run Auto-Search", type="primary"):
        res_auto = find_optimal_entry_point(
            struct_mask, st.session_state.target_pt, dist_map, roi_mask, roi_allowed, roi_depth
        )
        if res_auto:
            best = res_auto["best"]
            st.session_state.entry_pt = best["entry"]
            st.session_state.auto_info = {
                "min_dist": round(float(best["min_dist"]), 1),
                "length": round(float(best["geo"]["length"]), 1),
                "evaluated": res_auto["candidates_evaluated"],
            }
            st.rerun()
        else:
            st.session_state.entry_pt = None
            st.session_state.auto_info = None
            st.error("No safe entry point found on the brain-ROI edge for this target (check ROI outline, risk mask and target position).")

    if st.session_state.auto_info and st.session_state.entry_pt:
        info = st.session_state.auto_info
        st.success(
            f"Optimal entry: `{st.session_state.entry_pt}` | min clearance {info['min_dist']} px | "
            f"length {info['length']} px | {info['evaluated']} valid candidates evaluated"
        )

    show_bgr(
        draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
        caption="Green = Target, Blue = Entry",
    )

else:
    if input_method == "Click on Image":
        if not CLICK_SELECTION_AVAILABLE:
            st.error("`streamlit-image-coordinates` library not installed (pip install streamlit-image-coordinates).")
        else:
            stage = st.session_state.selection_stage

            if stage == "target":
                st.info("🎯 Click to set **Target**.")
            elif stage == "entry":
                st.info("🔴 Click to set **Entry** (on the edge of the green ROI outline).")
            else:
                st.success("Both points set. Press Reset to choose new points.")

            if stage in ("target", "entry"):
                preview = draw_points_preview(
                    original_img, st.session_state.target_pt, st.session_state.entry_pt
                )
                pil_img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                click_key = f"click_coord_{file_hash}_{st.session_state.click_nonce}"
                coords = streamlit_image_coordinates(pil_img, key=click_key, width=w)

                if coords:
                    disp_w = coords.get("width") or w
                    disp_h = coords.get("height") or h
                    cx = int(round(coords["x"] * w / disp_w))
                    cy = int(round(coords["y"] * h / disp_h))
                    if 0 <= cx < w and 0 <= cy < h:
                        if stage == "target":
                            st.session_state.target_pt = (cx, cy)
                            st.session_state.selection_stage = "entry"
                        else:
                            st.session_state.entry_pt = (cx, cy)
                            st.session_state.selection_stage = "done"
                        # Remount the component so this click is never consumed twice.
                        st.session_state.click_nonce += 1
                        st.rerun()
            else:
                show_bgr(
                    draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
                    caption="Green = Target, Blue = Entry",
                )

            st.write(f"**Target:** `{st.session_state.target_pt}` | **Entry:** `{st.session_state.entry_pt}`")

            if st.button("🔄 Reset Points"):
                reset_points()
                st.rerun()
    else:
        col_t, col_e = st.columns(2)
        with col_t:
            tx = st.number_input("Target X", 0, w - 1, int(w * 0.6), key=f"mtx_{file_hash}")
            ty = st.number_input("Target Y", 0, h - 1, int(h * 0.6), key=f"mty_{file_hash}")
            st.session_state.target_pt = (int(tx), int(ty))
        with col_e:
            ex = st.number_input("Entry X", 0, w - 1, int(w * 0.2), key=f"mex_{file_hash}")
            ey = st.number_input("Entry Y", 0, h - 1, int(h * 0.2), key=f"mey_{file_hash}")
            st.session_state.entry_pt = (int(ex), int(ey))

        show_bgr(
            draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
            caption="Green = Target, Blue = Entry",
        )

# Signature is evaluated AFTER inputs update, so stale results are never shown.
current_sig = generate_config_signature(
    file_hash=file_hash,
    use_otsu=use_otsu,
    manual_thresh=manual_thresh,
    risk_percentile=risk_percentile,
    roi_strip_px=roi_strip_px,
    min_area=min_area,
    dilation_rad=dilation_rad,
    target_pt=st.session_state.target_pt,
    entry_pt=st.session_state.entry_pt,
    proc_mode=procedure_mode,
    laser_power=laser_power,
    abl_time=ablation_time,
    num_contacts=num_contacts,
    spacing_px=contact_spacing,
)
if st.session_state.result_sig != current_sig:
    st.session_state.results = None
    st.session_state.result_sig = current_sig
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0

st.divider()
st.subheader("3️⃣ Calculate Plan")

can_compute = st.session_state.target_pt is not None and st.session_state.entry_pt is not None

if can_compute:
    btn_label = (
        "Compute LITT Thermal Model" if procedure_mode == "LITT Thermal Ablation"
        else "Compute SEEG Trajectory"
    )

    if st.button(btn_label, type="primary", **STRETCH):
        plan, err = build_plan(
            entry=st.session_state.entry_pt,
            target=st.session_state.target_pt,
            struct_mask=struct_mask,
            raw_mask=raw_mask,
            dist_map=dist_map,
            roi=roi_mask,
            roi_allowed=roi_allowed,
            roi_depth=roi_depth,
            blurred_gray=blurred_gray,
            w=w,
            h=h,
            procedure_mode=procedure_mode,
            use_otsu=use_otsu,
            manual_thresh=manual_thresh,
            risk_percentile=risk_percentile,
            min_area=min_area,
            dilation_rad=dilation_rad,
            laser_power=laser_power,
            ablation_time=ablation_time,
            num_contacts=num_contacts,
            contact_spacing=contact_spacing,
        )
        if err:
            st.session_state.results = None
            st.error(f"❌ {err}")
        else:
            st.session_state.results = plan
            st.session_state.sim_step = 0
            st.session_state.sim_playing = False
            st.rerun()
else:
    st.warning("Please define both Target and Entry points above.")

# --- Results Dashboard ---
if st.session_state.results:
    res = st.session_state.results
    st.divider()
    st.subheader("4️⃣ Analysis & Visualization")

    if res.get("entry_warning"):
        st.warning(res["entry_warning"])

    if res["safety"]["leaves_roi"]:
        st.error(
            f"🚨 Trajectory leaves the heuristic brain ROI ({res['roi_outside_px']} px outside). "
            "Status is forced to DANGER."
        )

    if res["safety"]["mask_unstable"]:
        st.warning(
            f"⚠️ Unstable risk mask: under a ±{SENSITIVITY_THRESHOLD_SHIFT} threshold shift it grows to "
            f"{res['safety']['max_risk_fraction'] * 100:.0f}% of the brain ROI. The threshold sits on the "
            "tissue intensity plateau. Raise the High-Signal Percentile (98-99) or use a manual threshold."
        )
    if res["safety"]["collision_flipped"]:
        st.warning(
            "⚠️ Segmentation sensitivity: a ±threshold change flips this trajectory between "
            "colliding and non-colliding. Treat the result as unreliable."
        )
    elif res["safety"]["status_flipped"]:
        st.info("ℹ️ Safety status changes under small threshold shifts (borderline clearance).")

    rows = [
        ("Trajectory Length (px)", f"{res['geometry']['length']:.2f}"),
        ("Entry Angle (deg)", f"{res['geometry']['angle']:.2f}"),
        ("Min Clearance (px)", f"{res['safety']['min_distance_px']:.2f}"),
        ("Clearance Score (%) [distance-based, not a probability]", f"{res['safety']['clearance_score_pct']:.1f}"),
        ("Trajectory Inside Brain ROI?",
         "YES ✅" if not res["safety"]["leaves_roi"] else f"NO 🚨 ({res['roi_outside_px']} px outside)"),
        ("Safety Status", res["safety"]["status"]),
        ("Confidence Index (Sensitivity)", res["safety"]["confidence_index"]),
        ("Variability Score", f"{res['safety']['variability_score']:.3f}"),
        ("Collision Detected?", "YES 🚨" if res["safety"]["intersects"] else "NO ✅"),
    ]

    if res["seeg"]:
        rows.append(("SEEG Contacts Placed", f"{res['seeg']['num_contacts_placed']}"))
        rows.append(("Contacts Fit Trajectory?", "YES ✅" if res["seeg"]["fits_within_trajectory"] else "NO ⚠️"))
        rows.append(("Required Contact Span (px)", f"{res['seeg']['required_span_px']:.1f}"))

    if res["thermal"]:
        th = res["thermal"]
        rows.append(("Peak Temperature (°C)", f"{th['max_temp_c']:.2f}"))
        rows.append(("Max CEM43 (min)", f"{th['max_cem43_min']:.2f}"))
        rows.append(("Mean CEM43 (min)", f"{th['mean_cem43_min']:.2f}"))
        rows.append((f"Ablation Zone Area, CEM43≥{CEM43_ABLATION_THRESHOLD_MIN:.0f} (px²)", f"{th['ablation_area_px']}"))
        rows.append(("Max CEM43 Inside Risk Regions (min)", f"{th['structure_max_cem43_min']:.2f}"))
        rows.append(("Risk-Region Thermal Attenuation (°C)", f"{th['risk_attenuation_c']:.2f}"))

    if enable_mm:
        rows.append(("Est. Length (mm)", f"{res['geometry']['length'] * pixel_spacing_mm:.2f}"))
        rows.append(("Est. Clearance (mm)", f"{res['safety']['min_distance_px'] * pixel_spacing_mm:.2f}"))
        if res["seeg"]:
            rows.append(("Est. Contact Span (mm)", f"{res['seeg']['required_span_px'] * pixel_spacing_mm:.2f}"))

    df_metrics = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df_metrics, hide_index=True, **STRETCH)

    if res["thermal"] and not res["thermal"]["model_valid"]:
        st.error(f"🔥 {res['thermal']['model_warning']}")
    if res["thermal"] and res["thermal"]["structure_max_cem43_min"] >= CEM43_ABLATION_THRESHOLD_MIN:
        st.warning(
            "Thermal dose inside high-signal risk regions reaches the necrosis threshold "
            f"(CEM43 ≥ {CEM43_ABLATION_THRESHOLD_MIN:.0f} min)."
        )
    if res["seeg"] and not res["seeg"]["fits_within_trajectory"]:
        st.warning("Requested contact span is longer than the trajectory; some contacts were not placed.")

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("**Planned Trajectory**")
        annot_img = draw_trajectory_overlay(
            original_img,
            res["entry"],
            res["target"],
            res["safety"]["color_bgr"],
            seeg_contacts=res["seeg"]["contacts"] if res["seeg"] else None,
        )
        show_bgr(annot_img)

    with vis_col2:
        if procedure_mode == "LITT Thermal Ablation" and res["thermal"]:
            st.markdown("**Simplified Static Gaussian Thermal Approximation**")
            temp_vis = render_heatmap_with_colorbar(
                original_img,
                res["thermal"]["temp_map"],
                BASELINE_BODY_TEMP_C,
                TEMP_DISPLAY_MAX_C,
                cv2.COLORMAP_JET,
                "Approx. Temp (C), synthetic",
            )
            show_bgr(temp_vis)

            cem_vmax = clamp(res["thermal"]["max_cem43_min"], 1.0, CEM43_DISPLAY_MAX_MIN)
            cem_vis = render_heatmap_with_colorbar(
                original_img,
                res["thermal"]["cem43_map"],
                0.0,
                cem_vmax,
                cv2.COLORMAP_HOT,
                "CEM43 Dose (min)",
            )
            show_bgr(cem_vis)
        else:
            st.markdown("**Distance Map (Clearance)**")
            dist_norm = np.clip(dist_map, 0, MAX_SAFE_CLEARANCE_PX)
            dist_vis = render_heatmap_with_colorbar(
                original_img,
                dist_norm,
                0.0,
                MAX_SAFE_CLEARANCE_PX,
                cv2.COLORMAP_VIRIDIS,
                "Distance to Risk Region (px)",
            )
            show_bgr(dist_vis)

    # --- Live Simulation Section (fragment: reruns do not re-execute the whole app) ---
    st.divider()
    st.subheader("5️⃣ Live Navigation Simulation")
    live_simulation(res, original_img, dist_map)

    # --- Export Section ---
    st.divider()
    st.subheader("6️⃣ Export & Reporting")

    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        export_dict = {
            "metadata": {
                "tool_version": TOOL_VERSION,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "disclaimer": disclaimer_text,
                "pixel_spacing_mm": pixel_spacing_mm if enable_mm else None,
            },
            "procedure": procedure_mode,
            "image_processing": {
                "roi": "heuristic brain-region ROI extraction (Otsu + opening + largest component + closing + hole fill)",
                "roi_strip_radius_px": roi_strip_px,
                "risk_regions": "intensity-based high-signal regions inside ROI (heuristic, not anatomical)",
                "auto_threshold": use_otsu,
                "risk_percentile": risk_percentile if use_otsu else None,
                "manual_threshold": None if use_otsu else manual_thresh,
                "min_area_px2": min_area,
                "buffer_px": dilation_rad,
            },
            "points": {
                "target_px": list(res["target"]),
                "entry_px": list(res["entry"]),
            },
            "trajectory_analysis": {
                "length_px": res["geometry"]["length"],
                "angle_deg": res["geometry"]["angle"],
                "min_clearance_px": res["safety"]["min_distance_px"],
                "clearance_score_pct": res["safety"]["clearance_score_pct"],
                "clearance_score_note": "distance-based score, not a probability of safety",
                "inside_brain_roi": not res["safety"]["leaves_roi"],
                "status": res["safety"]["status"],
                "collision": res["safety"]["intersects"],
                "confidence": res["safety"]["confidence_index"],
                "variability_score": res["safety"]["variability_score"],
                "risk_mask_unstable_under_sensitivity": res["safety"]["mask_unstable"],
                "max_risk_fraction_under_sensitivity": res["safety"]["max_risk_fraction"],
                "status_flipped_under_sensitivity": res["safety"]["status_flipped"],
                "collision_flipped_under_sensitivity": res["safety"]["collision_flipped"],
                "entry_warning": res.get("entry_warning") or None,
            },
            "model_specific_summary": {
                "litt_thermal": {
                    "model": "simplified static Gaussian approximation; duration affects CEM43 only; not a prediction of actual tissue temperature",
                    "max_temp_c": res["thermal"]["max_temp_c"],
                    "model_valid": res["thermal"]["model_valid"],
                    "model_warning": res["thermal"]["model_warning"] or None,
                    "max_cem43_min": res["thermal"]["max_cem43_min"],
                    "mean_cem43_min": res["thermal"]["mean_cem43_min"],
                    "ablation_area_px": res["thermal"]["ablation_area_px"],
                    "risk_region_max_cem43_min": res["thermal"]["structure_max_cem43_min"],
                    "risk_attenuation_c": res["thermal"]["risk_attenuation_c"],
                    "laser_power_w": laser_power,
                    "duration_s": ablation_time,
                } if res["thermal"] else None,
                "seeg_contacts": {
                    "count": res["seeg"]["num_contacts_placed"],
                    "fits": res["seeg"]["fits_within_trajectory"],
                    "required_span_px": res["seeg"]["required_span_px"],
                    "contacts": [
                        {"id": c["id"], "pos_px": list(c["pos"]), "depth_px": c["depth_px"]}
                        for c in res["seeg"]["contacts"]
                    ],
                } if res["seeg"] else None,
            },
        }
        st.download_button(
            label="📥 Download Plan Summary (JSON)",
            data=json.dumps(export_dict, indent=2),
            file_name=f"ThermoSEEG_Plan_{file_hash}.json",
            mime="application/json",
        )

    with exp_c2:
        report_img = annot_img.copy()
        header_h = 40
        header = np.zeros((header_h, report_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"ThermoSEEG Report - {procedure_mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        final_report = np.vstack([header, report_img])

        success, encoded = cv2.imencode(".png", final_report)
        if success:
            st.download_button(
                label="🖼️ Download Annotated Image (PNG)",
                data=encoded.tobytes(),
                file_name=f"ThermoSEEG_Image_{file_hash}.png",
                mime="image/png",
            )
