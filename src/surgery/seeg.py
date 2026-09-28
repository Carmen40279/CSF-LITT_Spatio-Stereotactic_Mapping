
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.vectors import Point3D
from src.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class SEEGContact:
    """A single SEEG contact along an electrode trajectory."""

    index: int
    position: Point3D
    distance_from_entry: float

    @property
    def coordinates(self) -> tuple[float, float, float]:
        """Return contact coordinates."""

        return (
            self.position.x,
            self.position.y,
            self.position.z,
        )


@dataclass
class SEEGElectrode:
    """
    A planned SEEG electrode.

    The trajectory runs from entry to target. Contacts are generated
    along that trajectory using the supplied contact count and spacing.
    """

    electrode_id: str
    trajectory: Trajectory

    contact_count: int

    contact_spacing: float = 3.5

    contacts: list[SEEGContact] = field(
        default_factory=list
    )

    coordinate_space: str = "patient"

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate electrode parameters."""

        if not self.electrode_id:
            raise ValueError(
                "SEEG electrode must have an electrode_id."
            )

        if self.contact_count < 1:
            raise ValueError(
                "SEEG electrode must have at least one contact."
            )

        if self.contact_spacing <= 0:
            raise ValueError(
                "SEEG contact spacing must be positive."
            )

    @property
    def entry(self) -> Point3D:
        """Electrode entry point."""

        return self.trajectory.entry

    @property
    def target(self) -> Point3D:
        """Electrode target point."""

        return self.trajectory.target

    @property
    def trajectory_length(self) -> float:
        """Length of the planned trajectory."""

        return self.trajectory.length

    def generate_contacts(self) -> list[SEEGContact]:
        """
        Generate evenly spaced contacts along the trajectory.

        Contacts are placed starting from the trajectory entry point.
        """

        required_length = (
            (self.contact_count - 1)
            * self.contact_spacing
        )

        if required_length > self.trajectory_length:
            raise ValueError(
                "SEEG contacts do not fit within the electrode "
                "trajectory."
            )

        generated: list[SEEGContact] = []

        for index in range(
            self.contact_count
        ):
            distance = (
                index
                * self.contact_spacing
            )

            point = self.trajectory.point_at(
                distance
            )

            generated.append(
                SEEGContact(
                    index=index + 1,
                    position=point,
                    distance_from_entry=distance,
                )
            )

        self.contacts = generated

        return generated

    def contact_positions(self) -> np.ndarray:
        """
        Return generated contact positions as an Nx3 NumPy array.
        """

        if not self.contacts:
            self.generate_contacts()

        return np.array(
            [
                contact.coordinates
                for contact in self.contacts
            ],
            dtype=float,
        )