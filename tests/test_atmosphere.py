"""Tests for the atmosphere module."""

import pytest

from chicken_from_space.atmosphere import density_at_altitude
from chicken_from_space.config import AtmosphereConfig


def test_density_decreases_with_altitude() -> None:
    """Density should fall with altitude in the exponential model."""

    config = AtmosphereConfig()
    sea_level = float(density_at_altitude(0.0, config))
    high_altitude = float(density_at_altitude(10_000.0, config))

    assert sea_level == pytest.approx(config.sea_level_density_kg_per_m3)
    assert sea_level > high_altitude > 0.0


def test_density_is_clamped_below_ground() -> None:
    """Negative altitude should reuse the ground-level density."""

    config = AtmosphereConfig()
    below_ground = float(density_at_altitude(-50.0, config))

    assert below_ground == pytest.approx(config.sea_level_density_kg_per_m3)
