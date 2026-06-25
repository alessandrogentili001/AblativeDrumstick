"""Simulation driver for the chicken-from-space model."""

from dataclasses import dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import diffrax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
import numpy as np

from chicken_from_space.atmosphere import atmospheric_temperature, density_at_altitude
from chicken_from_space.config import Config
from chicken_from_space.physics import (
    air_conductivity,
    air_viscosity_sutherland,
    convective_heat_transfer_coefficient,
    cross_sectional_area,
    drag_force_magnitude,
    drag_power_dissipation,
    local_terminal_velocity,
    mach_number,
    net_surface_heat_flux,
    recovery_temperature,
    speed_of_sound,
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
    temperature_k: jnp.ndarray  # shape: (num_steps, N)

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
            "center_temperature_k": np.asarray(self.temperature_k[:, 0]),
            "surface_temperature_k": np.asarray(self.temperature_k[:, -1]),
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
    cooked: bool
    cook_time_s: float | None
    cook_altitude_m: float | None
    cook_velocity_m_per_s: float | None
    max_center_temperature_k: float
    max_surface_temperature_k: float


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
    n_nodes = config.thermal.grid_nodes

    def vector_field(t: float, y: jnp.ndarray, args: object) -> jnp.ndarray:
        """Return the vertical dynamics and thermal conduction derivatives."""

        del t, args
        altitude_m = y[0]
        velocity_m_per_s = y[1]
        temps = y[2:]

        # 1. Trajectory dynamics
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

        # 2. Thermal boundary-layer heat transfer
        t_ambient = atmospheric_temperature(altitude_m, config)
        c_s = speed_of_sound(t_ambient)
        mach = mach_number(velocity_m_per_s, c_s)
        t_recovery = recovery_temperature(t_ambient, mach)

        mu = air_viscosity_sutherland(t_ambient)
        k_f = air_conductivity(mu)
        d_chicken = 2.0 * config.chicken.equivalent_radius_m

        h_c = convective_heat_transfer_coefficient(
            density_kg_per_m3=density,
            speed_m_per_s=velocity_m_per_s,
            diameter_m=d_chicken,
            viscosity_pa_s=mu,
            conductivity_w_per_m_k=k_f,
        )

        t_surface = temps[-1]
        q_surf = net_surface_heat_flux(
            h_c=h_c,
            t_recovery=t_recovery,
            t_surface=t_surface,
            t_ambient=t_ambient,
            emissivity=config.thermal.emissivity,
        )

        # 3. 1D Spherical conduction discretization
        rho_c = config.thermal.density_kg_per_m3
        cp_c = config.thermal.specific_heat_j_per_kg_k
        k_c = config.thermal.conductivity_w_per_m_k
        alpha_c = k_c / (rho_c * cp_c)

        r_max = config.chicken.equivalent_radius_m
        dr = r_max / (n_nodes - 1)

        dT_dt = jnp.zeros(n_nodes)

        # Center node (i = 0)
        dT_dt = dT_dt.at[0].set(6.0 * alpha_c * (temps[1] - temps[0]) / (dr**2))

        # Intermediate nodes (i = 1 to n_nodes - 2)
        i_idx = jnp.arange(1, n_nodes - 1)
        r_i = i_idx * dr
        dT_dt = dT_dt.at[i_idx].set(
            alpha_c
            * (
                (temps[i_idx + 1] - 2.0 * temps[i_idx] + temps[i_idx - 1]) / (dr**2)
                + (2.0 / r_i) * (temps[i_idx + 1] - temps[i_idx - 1]) / (2.0 * dr)
            )
        )

        # Surface node (i = n_nodes - 1)
        dT_dt = dT_dt.at[-1].set(
            (2.0 / (rho_c * cp_c * dr))
            * (
                q_surf
                - ((1.0 - dr / (2.0 * r_max)) ** 2) * k_c * (temps[-1] - temps[-2]) / dr
            )
        )

        return jnp.concatenate(
            [
                jnp.array([-velocity_m_per_s, acceleration_m_per_s2]),
                dT_dt,
            ]
        )

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(
        rtol=config.simulation.relative_tolerance,
        atol=config.simulation.absolute_tolerance,
    )
    event = diffrax.Event(_ground_event)

    init_temps = jnp.full((n_nodes,), config.thermal.initial_temperature_k)
    initial_state = jnp.concatenate(
        [
            jnp.array(
                [
                    config.simulation.initial_altitude_m,
                    config.simulation.initial_velocity_m_per_s,
                ]
            ),
            init_temps,
        ]
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
    temperature_k = states[:, 2:]

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

    # Integrated drag energy
    energy_step_j = jnp.maximum(
        0.5 * (drag_power_w[1:] + drag_power_w[:-1]) * jnp.diff(time_s),
        0.0,
    )
    cumulative_drag_energy_j = jnp.concatenate(
        [jnp.asarray([0.0]), jnp.cumsum(energy_step_j)],
    )

    # Integrated transferred thermal energy entering the chicken surface
    t_ambient_history = atmospheric_temperature(altitude_m, config)
    c_s_history = speed_of_sound(t_ambient_history)
    mach_history = mach_number(velocity_m_per_s, c_s_history)
    t_recovery_history = recovery_temperature(t_ambient_history, mach_history)
    mu_history = air_viscosity_sutherland(t_ambient_history)
    k_f_history = air_conductivity(mu_history)
    d_chicken = 2.0 * config.chicken.equivalent_radius_m

    h_c_history = convective_heat_transfer_coefficient(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        diameter_m=d_chicken,
        viscosity_pa_s=mu_history,
        conductivity_w_per_m_k=k_f_history,
    )

    t_surface_history = temperature_k[:, -1]
    q_surf_history = net_surface_heat_flux(
        h_c=h_c_history,
        t_recovery=t_recovery_history,
        t_surface=t_surface_history,
        t_ambient=t_ambient_history,
        emissivity=config.thermal.emissivity,
    )

    a_surf = 4.0 * jnp.pi * (config.chicken.equivalent_radius_m**2)
    heat_power_w = q_surf_history * a_surf
    heat_energy_step_j = 0.5 * (heat_power_w[1:] + heat_power_w[:-1]) * jnp.diff(time_s)
    cumulative_transferred_energy_j = jnp.concatenate(
        [jnp.asarray([0.0]), jnp.cumsum(heat_energy_step_j)],
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
        temperature_k=temperature_k,
    )


def summarize_simulation(result: SimulationResult) -> SimulationSummary:
    """Build a compact summary for a finished simulation."""

    speed_m_per_s = jnp.abs(result.velocity_m_per_s)
    center_temp = result.temperature_k[:, 0]
    surface_temp = result.temperature_k[:, -1]

    cook_threshold = result.config.thermal.target_cook_temperature_k
    cooked_mask = center_temp >= cook_threshold
    cooked = bool(jnp.any(cooked_mask))

    if cooked:
        cook_idx = int(jnp.argmax(cooked_mask))
        cook_time_s = float(result.time_s[cook_idx])
        cook_altitude_m = float(result.altitude_m[cook_idx])
        cook_velocity_m_per_s = float(result.velocity_m_per_s[cook_idx])
    else:
        cook_time_s = None
        cook_altitude_m = None
        cook_velocity_m_per_s = None

    total_drag_energy_j = float(result.cumulative_drag_energy_j[-1])
    transferred_energy_j = float(result.cumulative_transferred_energy_j[-1])

    if total_drag_energy_j > 0.0:
        energy_transfer_fraction = transferred_energy_j / total_drag_energy_j
    else:
        energy_transfer_fraction = 0.0

    return SimulationSummary(
        hit_ground=result.hit_ground,
        total_fall_time_s=float(result.time_s[-1]),
        final_velocity_m_per_s=float(result.velocity_m_per_s[-1]),
        max_speed_m_per_s=float(jnp.max(speed_m_per_s)),
        max_drag_power_w=float(jnp.max(result.drag_power_w)),
        total_drag_energy_j=total_drag_energy_j,
        transferred_energy_j=transferred_energy_j,
        energy_transfer_fraction=energy_transfer_fraction,
        cooked=cooked,
        cook_time_s=cook_time_s,
        cook_altitude_m=cook_altitude_m,
        cook_velocity_m_per_s=cook_velocity_m_per_s,
        max_center_temperature_k=float(jnp.max(center_temp)),
        max_surface_temperature_k=float(jnp.max(surface_temp)),
    )


def format_summary(summary: SimulationSummary) -> str:
    """Format a readable simulation summary."""

    ground_status = "yes" if summary.hit_ground else "no"
    cook_status = "yes" if summary.cooked else "no"

    lines = [
        "Chicken From Space: Basic Simulation",
        "------------------------------------",
        f"Ground reached: {ground_status}",
        f"Total fall time: {summary.total_fall_time_s:,.2f} s",
        f"Final velocity: {summary.final_velocity_m_per_s:,.2f} m/s downward",
        f"Maximum speed: {summary.max_speed_m_per_s:,.2f} m/s",
        f"Peak drag power: {summary.max_drag_power_w:,.2f} W",
        f"Total drag-dissipated energy: {summary.total_drag_energy_j:,.2f} J",
        f"Actual energy absorbed fraction: {summary.energy_transfer_fraction:.2%}",
        f"Actual energy absorbed by chicken: {summary.transferred_energy_j:,.2f} J",
        f"Max core temperature: {summary.max_center_temperature_k - 273.15:.2f} C",
        f"Max surface temperature: {summary.max_surface_temperature_k - 273.15:.2f} C",
        f"Perfectly cooked: {cook_status}",
    ]

    if summary.cooked:
        lines.extend(
            [
                f"  - Cooked time: {summary.cook_time_s:,.2f} s",
                f"  - Cooked altitude: {summary.cook_altitude_m:,.2f} m",
                f"  - Cooked velocity: {summary.cook_velocity_m_per_s:,.2f} m/s",
            ]
        )

    return "\n".join(lines)


def save_simulation_data(result: SimulationResult, output_path: str | Path) -> Path:
    """Save simulation arrays to NPZ or CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = result.as_numpy_dict()
    if output_path.suffix.lower() == ".npz":
        np.savez(output_path, temperature_k=np.asarray(result.temperature_k), **arrays)
        return output_path

    if output_path.suffix.lower() == ".csv":
        header = ",".join(arrays.keys())
        matrix = np.column_stack([arrays[name] for name in arrays])
        np.savetxt(output_path, matrix, delimiter=",", header=header, comments="")
        return output_path

    raise ValueError("Output path must end with .npz or .csv.")
