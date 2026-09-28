"""
ThermoSEEG planning case loader.

This module converts the editable JSON planning case into
validated Python data models.

A future graphical interface can create the same PlanningCase
object without changing the computational engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.io.data_models import (
    AblationParameters,
    Coordinate3D,
    CSFParameters,
    LITTPlan,
    PlanningCase,
    SEEGPlan,
)


def _coordinate(
    value: Any,
    field_name: str,
) -> Coordinate3D:
    """Validate and convert a three-value coordinate."""

    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"{field_name} must contain exactly three numbers."
        )

    try:
        x, y, z = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must contain numeric coordinates."
        ) from exc

    return Coordinate3D(
        x=x,
        y=y,
        z=z,
    )


def load_planning_case(
    path: str | Path,
) -> PlanningCase:
    """
    Load a ThermoSEEG planning case from JSON.
    """

    case_path = Path(path)

    if not case_path.exists():
        raise FileNotFoundError(
            f"Planning case not found: {case_path}"
        )

    with case_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    try:
        case_id = str(data["case_id"])

        seeg_data = data["seeg"]

        seeg_entry = _coordinate(
            seeg_data["entry"],
            "SEEG entry",
        )

        seeg_target = _coordinate(
            seeg_data["target"],
            "SEEG target",
        )

        contacts = int(
            seeg_data["contacts"]
        )

        if contacts < 1:
            raise ValueError(
                "SEEG contacts must be at least 1."
            )

        litt_data = data["litt"]

        litt_entry = _coordinate(
            litt_data["entry"],
            "LITT entry",
        )

        litt_target = _coordinate(
            litt_data["target"],
            "LITT target",
        )

        ablation_data = litt_data["ablation"]

        active_length = float(
            ablation_data["active_length"]
        )

        radius = float(
            ablation_data["radius"]
        )

        if active_length <= 0:
            raise ValueError(
                "LITT active length must be greater than zero."
            )

        if radius <= 0:
            raise ValueError(
                "LITT ablation radius must be greater than zero."
            )

        csf_data = data["csf"]

        csf_center = _coordinate(
            csf_data["center"],
            "CSF center",
        )

        csf_radii = _coordinate(
            csf_data["radii"],
            "CSF radii",
        )

        if (
            csf_radii.x <= 0
            or csf_radii.y <= 0
            or csf_radii.z <= 0
        ):
            raise ValueError(
                "CSF radii must all be greater than zero."
            )

        resolution = int(
            csf_data["resolution"]
        )

        if resolution < 8:
            raise ValueError(
                "CSF resolution must be at least 8."
            )

    except KeyError as exc:
        raise ValueError(
            f"Missing planning-case field: {exc}"
        ) from exc

    return PlanningCase(
        case_id=case_id,

        seeg=SEEGPlan(
            entry=seeg_entry,
            target=seeg_target,
            contacts=contacts,
        ),

        litt=LITTPlan(
            entry=litt_entry,
            target=litt_target,

            ablation=AblationParameters(
                active_length=active_length,
                radius=radius,
            ),
        ),

        csf=CSFParameters(
            center=csf_center,
            radii=csf_radii,
            resolution=resolution,
        ),
    )