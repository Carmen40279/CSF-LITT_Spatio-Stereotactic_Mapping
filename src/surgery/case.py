from __future__ import annotations
from dataclasses import dataclass, field
from src.surgery.seeg import SEEGElectrode
from src.surgery.litt import LITTPlan

@dataclass
class SurgicalPlan:
    seeg_electrodes: list[SEEGElectrode] = field(
        default_factory=list
    )
    litt_plans: list[LITTPlan] = field(
        default_factory=list
    )
    targets: list[object] = field(
        default_factory=list
    )
    def add_seeg_electrode(
        self,
        electrode: SEEGElectrode,
    ) -> None:
        """Add an SEEG electrode to the plan."""
        if any(
            existing.electrode_id
            == electrode.electrode_id
            for existing in self.seeg_electrodes
        ):
            raise ValueError(
                f"SEEG electrode ID already exists: "
                f"{electrode.electrode_id}"
            )

        self.seeg_electrodes.append(
            electrode
        )
    def add_litt_plan(
        self,
        plan: LITTPlan,
    ) -> None:
        """Add a LITT trajectory."""
        if any(
            existing.plan_id
            == plan.plan_id
            for existing in self.litt_plans
        ):
            raise ValueError(
                f"LITT plan ID already exists: "
                f"{plan.plan_id}"
            )

        self.litt_plans.append(
            plan
        )
    @property
    def electrode_count(self) -> int:
        """Number of SEEG electrodes."""

        return len(
            self.seeg_electrodes
        )
    @property
    def litt_count(self) -> int:
        """Number of LITT trajectories."""
        return len(
            self.litt_plans
        )
