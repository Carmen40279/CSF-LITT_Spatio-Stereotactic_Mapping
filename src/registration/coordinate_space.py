"""
ThermoSEEG coordinate-space registry.

Maintains the spatial reference systems associated with a patient
case and the transformations connecting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.registration.transforms import (
    SpatialTransform,
    identity_transform,
    validate_transform,
)


@dataclass
class CoordinateSpaceRegistry:
    """
    Registry of coordinate spaces and the transformations
    connecting them.
    """

    spaces: set[str] = field(default_factory=set)

    transforms: dict[
        tuple[str, str],
        SpatialTransform,
    ] = field(default_factory=dict)

    def add_space(self, name: str) -> None:
        """Register a coordinate space."""

        if not isinstance(name, str):
            raise TypeError(
                "Coordinate-space name must be a string."
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "Coordinate-space name cannot be empty."
            )

        self.spaces.add(name)

    def add_transform(
        self,
        transform: SpatialTransform,
    ) -> None:
        """Register a transformation between two spaces."""

        validate_transform(transform)

        self.add_space(transform.source_space)
        self.add_space(transform.target_space)

        key = (
            transform.source_space,
            transform.target_space,
        )

        self.transforms[key] = transform

    def add_identity(self, space: str) -> None:
        """Register an identity transformation for a space."""

        self.add_transform(
            identity_transform(space)
        )

    def get_transform(
        self,
        source_space: str,
        target_space: str,
    ) -> SpatialTransform:
        """
        Retrieve a transformation between two spaces.

        Direct transformations are returned unchanged.
        Reverse transformations are automatically inverted.
        """

        if source_space == target_space:
            return identity_transform(source_space)

        key = (
            source_space,
            target_space,
        )

        if key in self.transforms:
            return self.transforms[key]

        reverse_key = (
            target_space,
            source_space,
        )

        if reverse_key in self.transforms:
            return self.transforms[reverse_key].inverse()

        raise KeyError(
            "No direct transformation registered between "
            f"{source_space!r} and {target_space!r}."
        )