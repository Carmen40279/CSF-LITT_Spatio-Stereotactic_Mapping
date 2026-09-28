"""
SEEG electrode contact generation.
"""

from __future__ import annotations

from src.geometry.vectors import Point3D
from src.trajectory.trajectory import Trajectory


def generate_contacts(
    trajectory: Trajectory,
    number_of_contacts: int,
) -> list[Point3D]:
    """
    Generate evenly spaced electrode contacts along a trajectory.

    Contacts are distributed from entry toward target.
    """

    if number_of_contacts < 1:
        raise ValueError(
            "Number of contacts must be at least 1."
        )

    if number_of_contacts == 1:
        return [trajectory.entry]

    spacing = trajectory.length / (number_of_contacts - 1)

    return [
        trajectory.point_at(index * spacing)
        for index in range(number_of_contacts)
    ]