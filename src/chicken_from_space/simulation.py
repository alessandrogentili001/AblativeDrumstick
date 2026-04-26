"""Simulation driver for the chicken-from-space model."""

from dataclasses import dataclass
from pathlib import Path

import diffrax
import jax.numpy as jnp
import numpy as np

from chicken_from_space.atmosphere import density_at_altitude
from chicken_from_space.config import Config
from chicken_from_space.physics import (
    cross_sectional_area,
    drag_force_magnitude,
    drag_power_dissipation,
    local_terminal_velocity,
)


@dataclass(frozen=True)
class SimulationResult:
    """Full time history and integrated energies."""

    config: Config
    hit_ground: bool
    time_s: jnp.ndarray
    altitude_m: jnp.ndarray
    velocity_m_per_s: jnp.ndarray
    density_kg_per_m3: jnp.ndarray
    drag_force_n: jnp.ndarray
    terminal_velocity_m_per_s: jnp.ndarray
    drag_power_w: jnp.ndarray
    cumulative_drag_energy_j: jnp.ndarray
    cumulative_transferred_energy_j: jnp.ndarray

    def as_numpy_dict(self) -> dict[str, np.ndarray]:
        """Return the stored arrays as NumPy arrays."""

        return {
            "time_s": np.asarray(self.time_s),
            "altitude_m": np.asarray(self.altitude_m),
            "velocity_m_per_s": np.asarray(self.velocity_m_per_s),
            "density_kg_per_m3": np.asarray(self.density_kg_per_m3),
            "drag_force_n": np.asarray(self.drag_force_n),
            "terminal_velocity_m_per_s": np.asarray(self.terminal_velocity_m_per_s),
            "drag_power_w": np.asarray(self.drag_power_w),
            "cumulative_drag_energy_j": np.asarray(self.cumulative_drag_energy_j),
            "cumulative_transferred_energy_j": np.asarray(
                self.cumulative_transferred_energy_j,
            ),
        }


@dataclass(frozen=True)
class SimulationSummary:
    """Small summary of the main run outputs."""

    hit_ground: bool
    total_fall_time_s: float
    final_velocity_m_per_s: float
    max_speed_m_per_s: float
    max_drag_power_w: float
    total_drag_energy_j: float
    transferred_energy_j: float
    energy_transfer_fraction: float


def _ground_event(
    t: float,
    y: jnp.ndarray,
    args: object,
    **kwargs: object,
) -> jnp.ndarray:
    """Return whether the ground has been reached or crossed."""

    del t, args, kwargs
    return y[0] <= 0.0


def _trim_solution(solution: diffrax.Solution) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Drop the padded `inf` values produced by step-based saving."""

    raw_time_s = np.asarray(solution.ts)
    raw_states = np.asarray(solution.ys)
    finite_mask = np.isfinite(raw_time_s)
    trimmed_time_s = jnp.asarray(raw_time_s[finite_mask])
    trimmed_states = jnp.asarray(raw_states[finite_mask])
    return trimmed_time_s, trimmed_states


def _interpolate_ground_impact(
    time_s: jnp.ndarray,
    states: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Replace the final below-ground state with an interpolated impact state."""

    if time_s.shape[0] == 0:
        return time_s, states

    if float(states[-1, 0]) > 0.0:
        return time_s, states

    if time_s.shape[0] == 1:
        clamped_state = states.at[-1, 0].set(0.0)
        return time_s, clamped_state

    previous_time_s = time_s[-2]
    final_time_s = time_s[-1]
    previous_state = states[-2]
    final_state = states[-1]

    altitude_span_m = previous_state[0] - final_state[0]
    if float(jnp.abs(altitude_span_m)) < 1.0e-12:
        impact_state = final_state.at[0].set(0.0)
        return time_s.at[-1].set(final_time_s), states.at[-1].set(impact_state)

    impact_fraction = previous_state[0] / altitude_span_m
    impact_fraction = jnp.clip(impact_fraction, 0.0, 1.0)
    impact_time_s = previous_time_s + impact_fraction * (final_time_s - previous_time_s)
    impact_state = previous_state + impact_fraction * (final_state - previous_state)
    impact_state = impact_state.at[0].set(0.0)

    updated_time_s = time_s.at[-1].set(impact_time_s)
    updated_states = states.at[-1].set(impact_state)
    return updated_time_s, updated_states


def run_simulation(config: Config | None = None) -> SimulationResult:
    """Run a full vertical fall simulation until ground impact."""

    config = config or Config()
    reference_area_m2 = float(cross_sectional_area(config.chicken.equivalent_radius_m))

    def vector_field(t: float, y: jnp.ndarray, args: object) -> jnp.ndarray:
        """Return the vertical dynamics for Diffrax."""

        del t, args
        altitude_m = y[0]
        velocity_m_per_s = y[1]
        density = density_at_altitude(altitude_m, config)
        drag_acceleration_m_per_s2 = (
            0.5
            * density
            * config.chicken.drag_coefficient
            * reference_area_m2
            * velocity_m_per_s
            * jnp.abs(velocity_m_per_s)
            / config.chicken.mass_kg
        )
        acceleration_m_per_s2 = (
            config.constants.gravity_m_per_s2 - drag_acceleration_m_per_s2
        )
        return jnp.array(
            [
                -velocity_m_per_s,
                acceleration_m_per_s2,
            ],
        )

    term = diffrax.ODETerm(vector_field)
    # Diffrax does not ship a classical adaptive RK4 solver; Dopri5 is the
    # standard explicit adaptive Runge-Kutta replacement for this first version.
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(
        rtol=config.simulation.relative_tolerance,
        atol=config.simulation.absolute_tolerance,
    )
    event = diffrax.Event(_ground_event)
    initial_state = jnp.array(
        [
            config.simulation.initial_altitude_m,
            config.simulation.initial_velocity_m_per_s,
        ],
    )
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=config.simulation.max_time_s,
        dt0=config.simulation.initial_time_step_s,
        y0=initial_state,
        args=None,
        saveat=diffrax.SaveAt(t0=True, steps=True),
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=config.simulation.max_solver_steps,
    )

    time_s, states = _trim_solution(solution)
    time_s, states = _interpolate_ground_impact(time_s, states)
    altitude_m = jnp.maximum(states[:, 0], 0.0)
    velocity_m_per_s = states[:, 1]
    density_kg_per_m3 = density_at_altitude(altitude_m, config)
    drag_force_n = drag_force_magnitude(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        drag_coefficient=config.chicken.drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    terminal_velocity_m_per_s = local_terminal_velocity(
        mass_kg=config.chicken.mass_kg,
        density_kg_per_m3=density_kg_per_m3,
        gravity_m_per_s2=config.constants.gravity_m_per_s2,
        drag_coefficient=config.chicken.drag_coefficient,
        reference_area_m2=reference_area_m2,
        density_floor_kg_per_m3=1.0e-12,
    )
    drag_power_w = drag_power_dissipation(
        density_kg_per_m3=density_kg_per_m3,
        velocity_m_per_s=velocity_m_per_s,
        drag_coefficient=config.chicken.drag_coefficient,
        reference_area_m2=reference_area_m2,
    )

    energy_step_j = jnp.maximum(
        0.5 * (drag_power_w[1:] + drag_power_w[:-1]) * jnp.diff(time_s),
        0.0,
    )
    cumulative_drag_energy_j = jnp.concatenate(
        [jnp.asarray([0.0]), jnp.cumsum(energy_step_j)],
    )
    cumulative_transferred_energy_j = (
        cumulative_drag_energy_j * config.simulation.energy_transfer_fraction
    )
    hit_ground = bool(float(altitude_m[-1]) <= config.simulation.absolute_tolerance)

    return SimulationResult(
        config=config,
        hit_ground=hit_ground,
        time_s=time_s,
        altitude_m=altitude_m,
        velocity_m_per_s=velocity_m_per_s,
        density_kg_per_m3=density_kg_per_m3,
        drag_force_n=drag_force_n,
        terminal_velocity_m_per_s=terminal_velocity_m_per_s,
        drag_power_w=drag_power_w,
        cumulative_drag_energy_j=cumulative_drag_energy_j,
        cumulative_transferred_energy_j=cumulative_transferred_energy_j,
    )


def summarize_simulation(result: SimulationResult) -> SimulationSummary:
    """Build a compact summary for a finished simulation."""

    speed_m_per_s = jnp.abs(result.velocity_m_per_s)
    return SimulationSummary(
        hit_ground=result.hit_ground,
        total_fall_time_s=float(result.time_s[-1]),
        final_velocity_m_per_s=float(result.velocity_m_per_s[-1]),
        max_speed_m_per_s=float(jnp.max(speed_m_per_s)),
        max_drag_power_w=float(jnp.max(result.drag_power_w)),
        total_drag_energy_j=float(result.cumulative_drag_energy_j[-1]),
        transferred_energy_j=float(result.cumulative_transferred_energy_j[-1]),
        energy_transfer_fraction=result.config.simulation.energy_transfer_fraction,
    )


def format_summary(summary: SimulationSummary) -> str:
    """Format a readable simulation summary."""

    ground_status = "yes" if summary.hit_ground else "no"
    return "\n".join(
        [
            "Chicken From Space: Basic Simulation",
            "------------------------------------",
            f"Ground reached: {ground_status}",
            f"Total fall time: {summary.total_fall_time_s:,.2f} s",
            f"Final velocity: {summary.final_velocity_m_per_s:,.2f} m/s downward",
            f"Maximum speed: {summary.max_speed_m_per_s:,.2f} m/s",
            f"Peak drag power: {summary.max_drag_power_w:,.2f} W",
            f"Total drag-dissipated energy: {summary.total_drag_energy_j:,.2f} J",
            f"Energy transfer fraction: {summary.energy_transfer_fraction:.2%}",
            f"Estimated energy transferred to chicken: {summary.transferred_energy_j:,.2f} J",
        ],
    )


def save_simulation_data(result: SimulationResult, output_path: str | Path) -> Path:
    """Save simulation arrays to NPZ or CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = result.as_numpy_dict()
    if output_path.suffix.lower() == ".npz":
        np.savez(output_path, **arrays)
        return output_path

    if output_path.suffix.lower() == ".csv":
        header = ",".join(arrays.keys())
        matrix = np.column_stack([arrays[name] for name in arrays])
        np.savetxt(output_path, matrix, delimiter=",", header=header, comments="")
        return output_path

    raise ValueError("Output path must end with .npz or .csv.")
