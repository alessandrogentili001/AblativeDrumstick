"""Tests for the simulation module."""

import pytest

from chicken_from_space.config import SimulationConfig
from chicken_from_space.simulation import run_simulation


def test_simulation_reaches_ground_and_accumulates_energy() -> None:
    """A short drop should reach the ground with non-negative energy."""

    config = SimulationConfig(
        initial_altitude_m=1_000.0,
        time_step_s=0.05,
        max_time_s=300.0,
    )
    result = run_simulation(config)

    assert result.hit_ground
    assert float(result.altitude_m[-1]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(result.cumulative_drag_energy_j[-1]) >= 0.0
