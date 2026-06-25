"""Tests for the simulation module."""

import pytest

# pyrefly: ignore [missing-import]
from chicken_from_space.config import Config, Simulation
# pyrefly: ignore [missing-import]
from chicken_from_space.simulation import run_simulation, summarize_simulation


def test_simulation_reaches_ground_and_accumulates_energy() -> None:
    """A short drop should reach the ground with non-negative energy."""

    config = Config(
        simulation=Simulation(
            initial_altitude_m=1_000.0,
            initial_time_step_s=0.05,
            max_time_s=300.0,
        ),
    )
    result = run_simulation(config)

    assert result.hit_ground
    assert float(result.altitude_m[-1]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(result.cumulative_drag_energy_j[-1]) >= 0.0


def test_thermal_conduction_and_cooking() -> None:
    """Verify that thermal arrays integrate and compile into the summary correctly."""

    config = Config(
        simulation=Simulation(
            initial_altitude_m=500.0,
            initial_time_step_s=0.1,
            max_time_s=100.0,
        ),
    )
    result = run_simulation(config)
    summary = summarize_simulation(result)

    # State size should match 2 (trajectory) + N nodes
    assert result.temperature_k.shape[1] == config.thermal.grid_nodes
    # Initial temperature check
    assert float(result.temperature_k[0, 0]) == pytest.approx(config.thermal.initial_temperature_k)

    # Core temperature must change due to conduction
    assert summary.max_center_temperature_k >= config.thermal.initial_temperature_k
    assert summary.max_surface_temperature_k >= summary.max_center_temperature_k
    assert isinstance(summary.cooked, bool)
