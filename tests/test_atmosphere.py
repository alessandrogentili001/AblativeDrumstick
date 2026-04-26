"""Tests for the atmosphere module."""

import jax
import pytest

from chicken_from_space.atmosphere import (
    atmospheric_density,
    atmospheric_pressure,
    atmospheric_temperature,
)
from chicken_from_space.config import Atmosphere, Config


def test_density_decreases_with_altitude() -> None:
    """Density should fall with altitude in the simplified atmosphere."""

    config = Config()
    sea_level = float(atmospheric_density(0.0, config))
    high_altitude = float(atmospheric_density(10_000.0, config))

    assert sea_level == pytest.approx(1.225, rel=2.0e-2)
    assert sea_level > high_altitude > 0.0


def test_density_is_clamped_below_ground() -> None:
    """Negative altitude should reuse the ground-level density."""

    config = Config()
    below_ground = float(atmospheric_density(-50.0, config))
    at_ground = float(atmospheric_density(0.0, config))

    assert below_ground == pytest.approx(at_ground)


def test_temperature_profile_rises_and_falls_across_layers() -> None:
    """Temperature should follow a piecewise standard-atmosphere profile."""

    config = Config()
    temperature_10_km = float(atmospheric_temperature(10_000.0, config))
    temperature_20_km = float(atmospheric_temperature(20_000.0, config))
    temperature_45_km = float(atmospheric_temperature(45_000.0, config))
    temperature_80_km = float(atmospheric_temperature(80_000.0, config))
    temperature_115_km = float(atmospheric_temperature(115_000.0, config))

    assert temperature_10_km < config.atmosphere.sea_level_temperature_k
    assert temperature_45_km > temperature_20_km
    assert temperature_80_km < temperature_45_km
    assert temperature_115_km > temperature_80_km


def test_pressure_decreases_monotonically() -> None:
    """Pressure should keep decreasing with altitude."""

    config = Config()
    pressure_0_km = float(atmospheric_pressure(0.0, config))
    pressure_20_km = float(atmospheric_pressure(20_000.0, config))
    pressure_50_km = float(atmospheric_pressure(50_000.0, config))
    pressure_100_km = float(atmospheric_pressure(100_000.0, config))

    assert pressure_0_km > pressure_20_km > pressure_50_km > pressure_100_km > 0.0


def test_configured_sea_level_temperature_offsets_profile() -> None:
    """Changing the configured sea-level temperature should shift the profile."""

    config = Config(
        atmosphere=Atmosphere(sea_level_temperature_k=293.15),
    )

    assert float(atmospheric_temperature(0.0, config)) == pytest.approx(293.15)


def test_density_profile_can_be_jitted_with_closed_over_config() -> None:
    """The common closure-based JIT pattern should work cleanly."""

    config = Config()
    density_fn = jax.jit(lambda altitude_m: atmospheric_density(altitude_m, config))

    assert float(density_fn(1_000.0)) > 0.0


def test_density_profile_can_be_jitted_with_config_arg() -> None:
    """Config should behave like a pytree in direct JIT usage."""

    config = Config()
    density_fn = jax.jit(atmospheric_density)

    assert float(density_fn(1_000.0, config)) > 0.0
