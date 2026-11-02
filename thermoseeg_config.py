from __future__ import annotations

import json
import math
from typing import Any, Dict


class RunConfigError(ValueError):
    pass


def parse_exported_run_config(data: bytes) -> Dict[str, Any]:
    """Read the reproducibility section from a ThermoSEEG JSON export."""
    def reject_constant(token: str):
        raise RunConfigError(f"Settings JSON contains non-finite constant '{token}'.")
    try:
        payload = json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RunConfigError("Settings file is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise RunConfigError("Settings JSON must be an object.")
    metadata = payload.get("metadata", {})
    reproducibility = payload.get("reproducibility", {})
    if not isinstance(metadata, dict) or not isinstance(reproducibility, dict):
        raise RunConfigError("Settings file has malformed metadata or reproducibility sections.")
    if payload.get("tool_version", metadata.get("tool_version")) is None:
        raise RunConfigError("JSON is not a ThermoSEEG export (tool version is missing).")
    cfg = reproducibility.get("configuration")
    if not isinstance(cfg, dict):
        raise RunConfigError("ThermoSEEG export has no reproducibility configuration section.")
    for key in ("target_px", "entry_px"):
        point = cfg.get(key)
        if not isinstance(point, list) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in point):
            raise RunConfigError(f"Configuration field '{key}' must contain two integer pixel coordinates.")
    return {
        "configuration": cfg,
        "tool_version": metadata.get("tool_version", payload.get("tool_version")),
        "image_sha256": metadata.get("image_sha256"),
        "pixel_spacing_source": metadata.get("pixel_spacing_source", "unknown"),
    }


def configuration_widget_values(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate exported values and return safe widget-key values within current UI ranges."""
    if not isinstance(run_config, dict) or not isinstance(run_config.get("configuration"), dict):
        raise RunConfigError("Settings file has no usable configuration section.")
    cfg = run_config["configuration"]
    procedure = cfg.get("procedure")
    if procedure not in {"SEEG Electrode Planning", "LITT Thermal Ablation"}:
        raise RunConfigError("Unsupported procedure in settings file.")
    def number(name, default, low, high, integer=False):
        raw = cfg.get(name, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RunConfigError(f"Configuration field '{name}' must be numeric.")
        try:
            val = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise RunConfigError(f"Configuration field '{name}' must be a finite number.") from exc
        if not math.isfinite(val):
            raise RunConfigError(f"Configuration field '{name}' must be a finite number.")
        if integer and not val.is_integer():
            raise RunConfigError(f"Configuration field '{name}' must be a whole number.")
        if not low <= val <= high:
            raise RunConfigError(f"Configuration field '{name}' is outside the current UI range ({low} to {high}).")
        return int(val) if integer else val
    use_otsu = cfg.get("use_otsu")
    if not isinstance(use_otsu, bool):
        raise RunConfigError("Configuration field 'use_otsu' must be boolean.")
    spacing_xy = cfg.get("pixel_spacing_mm_xy")
    spacing_source = run_config.get("pixel_spacing_source", "unknown")
    manual_mm_enabled = spacing_source == "manual isotropic assumption"
    manual_mm = 1.0
    if manual_mm_enabled:
        if not isinstance(spacing_xy, list) or len(spacing_xy) != 2:
            raise RunConfigError("Manual spacing in the saved configuration must be isotropic.")
        sx = number_from_spacing(spacing_xy[0])
        sy = number_from_spacing(spacing_xy[1])
        if abs(sx - sy) > 1e-8:
            raise RunConfigError("Manual spacing in the saved configuration must be isotropic.")
        manual_mm = sx
    return {
        "procedure_type": procedure,
        "laser_power": number("laser_power_w", 10.0, 2.0, 13.0) if procedure == "LITT Thermal Ablation" else 10.0,
        "ablation_duration": number("duration_s", 30.0, 5.0, 120.0) if procedure == "LITT Thermal Ablation" else 30.0,
        "seeg_contact_count": number("contact_count", 8, 2, 18, integer=True) if procedure == "SEEG Electrode Planning" else 8,
        "seeg_contact_spacing": number("contact_spacing_px", 10.0, 2.0, 30.0) if procedure == "SEEG Electrode Planning" else 10.0,
        "auto_threshold": use_otsu,
        "risk_percentile": number("risk_percentile", 97, 80, 99, integer=True) if use_otsu else 97,
        "manual_threshold": number("manual_threshold", 140, 0, 255, integer=True) if not use_otsu else 140,
        "roi_strip_radius": number("roi_strip_radius_px", 7, 0, 20, integer=True),
        "minimum_risk_area": number("minimum_risk_area_px2", 30, 0, 200, integer=True),
        "risk_dilation": number("risk_buffer_px", 5, 0, 15, integer=True),
        "show_mm": manual_mm_enabled,
        "manual_pixel_spacing": manual_mm,
        "target_point": _point(cfg, "target_px"),
        "entry_point": _point(cfg, "entry_px"),
    }


def _point(cfg: Dict[str, Any], key: str):
    point = cfg.get(key)
    if not isinstance(point, list) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in point):
        raise RunConfigError(f"Configuration field '{key}' must contain two integer pixel coordinates.")
    return (point[0], point[1])


def number_from_spacing(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunConfigError("Saved manual Pixel Spacing must be numeric.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RunConfigError("Saved manual Pixel Spacing must be a finite number.") from exc
    if not math.isfinite(result):
        raise RunConfigError("Saved manual Pixel Spacing must be a finite number.")
    if not 0.01 <= result <= 10.0:
        raise RunConfigError("Saved manual Pixel Spacing is outside the current input range.")
    return result
