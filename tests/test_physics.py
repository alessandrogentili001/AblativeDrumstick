"""Tests for the physics helpers."""

import pytest

from chicken_from_space.physics import (
    cross_sectional_area,
    drag_force_magnitude,
    drag_power_dissipation,
    local_terminal_velocity,
)


def test_drag_force_is_zero_at_zero_speed() -> None:
    """Quadratic drag should vanish at zero speed."""

    area = float(cross_sectional_area(0.1))
    drag_force = float(
        drag_force_magnitude(
            density_kg_per_m3=1.225,
            speed_m_per_s=0.0,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert drag_force == pytest.approx(0.0)


def test_terminal_velocity_rises_as_density_drops() -> None:
    """Lower density should increase the local terminal speed."""

    area = float(cross_sectional_area(0.1))
    dense_air = float(
        local_terminal_velocity(
            mass_kg=1.5,
            density_kg_per_m3=1.225,
            gravity_m_per_s2=9.81,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )
    thin_air = float(
        local_terminal_velocity(
            mass_kg=1.5,
            density_kg_per_m3=0.1,
            gravity_m_per_s2=9.81,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert thin_air > dense_air > 0.0


def test_drag_power_is_non_negative() -> None:
    """Dissipated drag power should be non-negative."""

    area = float(cross_sectional_area(0.1))
    power = float(
        drag_power_dissipation(
            density_kg_per_m3=1.225,
            velocity_m_per_s=-50.0,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert power >= 0.0
