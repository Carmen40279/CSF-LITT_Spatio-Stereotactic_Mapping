import numpy as np
import pytest

from src.registration.coordinate_space import CoordinateSpaceRegistry
from src.registration.registration import create_affine_registration
from src.registration.transforms import (
    SpatialTransform,
    identity_transform,
    translation_transform,
)


def test_identity_transform():
    transform = identity_transform("patient")

    result = transform.transform_point(
        [10.0, 20.0, 30.0]
    )

    assert np.allclose(
        result,
        [10.0, 20.0, 30.0],
    )


def test_translation_transform():
    transform = translation_transform(
        "mri",
        "patient",
        (10.0, -5.0, 2.0),
    )

    result = transform.transform_point(
        [1.0, 2.0, 3.0]
    )

    assert np.allclose(
        result,
        [11.0, -3.0, 5.0],
    )


def test_transform_points():
    transform = translation_transform(
        "mri",
        "patient",
        (1.0, 2.0, 3.0),
    )

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = transform.transform_points(points)

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )

    assert np.allclose(result, expected)


def test_inverse_transform():
    transform = translation_transform(
        "mri",
        "patient",
        (10.0, 20.0, 30.0),
    )

    inverse = transform.inverse()

    point = [1.0, 2.0, 3.0]

    transformed = transform.transform_point(point)
    recovered = inverse.transform_point(transformed)

    assert np.allclose(recovered, point)


def test_transform_composition():
    first = translation_transform(
        "mri",
        "ct",
        (10.0, 0.0, 0.0),
    )

    second = translation_transform(
        "ct",
        "patient",
        (0.0, 20.0, 0.0),
    )

    combined = first.compose(second)

    result = combined.transform_point(
        [1.0, 2.0, 3.0]
    )

    assert np.allclose(
        result,
        [11.0, 22.0, 3.0],
    )


def test_registry():
    registry = CoordinateSpaceRegistry()

    registry.add_space("mri")
    registry.add_space("patient")

    transform = translation_transform(
        "mri",
        "patient",
        (5.0, 6.0, 7.0),
    )

    registry.add_transform(transform)

    retrieved = registry.get_transform(
        "mri",
        "patient",
    )

    result = retrieved.transform_point(
        [1.0, 1.0, 1.0]
    )

    assert np.allclose(
        result,
        [6.0, 7.0, 8.0],
    )


def test_registry_reverse_transform():
    registry = CoordinateSpaceRegistry()

    transform = translation_transform(
        "mri",
        "patient",
        (5.0, 6.0, 7.0),
    )

    registry.add_transform(transform)

    reverse = registry.get_transform(
        "patient",
        "mri",
    )

    result = reverse.transform_point(
        [6.0, 7.0, 8.0]
    )

    assert np.allclose(
        result,
        [1.0, 1.0, 1.0],
    )


def test_invalid_transform_shape():
    with pytest.raises(ValueError):
        SpatialTransform(
            source_space="a",
            target_space="b",
            matrix=np.eye(3),
        )


def test_registration_result():
    result = create_affine_registration(
        "mri",
        "patient",
        np.eye(4),
    )

    assert result.converged

    assert result.transform.source_space == "mri"
    assert result.transform.target_space == "patient"