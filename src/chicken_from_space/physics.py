"""Core physics helpers for the falling-chicken model."""

from typing import NamedTuple

# pyrefly: ignore [missing-import]
import jax.numpy as jnp


class State(NamedTuple):
    """Vertical state with downward velocity taken as positive."""

    altitude_m: float
    velocity_m_per_s: float


class StateDerivative(NamedTuple):
    """Time derivatives for the vertical state."""

    altitude_rate_m_per_s: float
    velocity_rate_m_per_s2: float


def cross_sectional_area(radius_m: float) -> jnp.ndarray:
    """Return the cross-sectional area of an equivalent sphere."""

    r = jnp.maximum(jnp.asarray(radius_m), 0.0)
    return jnp.pi * r**2


def gravity_at_altitude(
    altitude_m: float,
    gravity_m_per_s2: float,
) -> jnp.ndarray:
    """Return the local gravity magnitude for the current model."""

    del altitude_m
    return jnp.asarray(gravity_m_per_s2)


def drag_force_magnitude(
    density_kg_per_m3: float,
    speed_m_per_s: float,
    drag_coefficient: float,
    reference_area_m2: float,
) -> jnp.ndarray:
    """Return the magnitude of the quadratic drag force."""

    rho = jnp.maximum(jnp.asarray(density_kg_per_m3), 0.0)
    v = jnp.abs(jnp.asarray(speed_m_per_s))
    return 0.5 * rho * drag_coefficient * reference_area_m2 * v**2


def signed_drag_force(
    density_kg_per_m3: float,
    velocity_m_per_s: float,
    drag_coefficient: float,
    reference_area_m2: float,
) -> jnp.ndarray:
    """Return drag force in the downward-positive sign convention."""

    f = drag_force_magnitude(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    v = jnp.asarray(velocity_m_per_s)
    return -f * jnp.sign(v)


def local_terminal_velocity(
    mass_kg: float,
    density_kg_per_m3: float,
    gravity_m_per_s2: float,
    drag_coefficient: float,
    reference_area_m2: float,
    density_floor_kg_per_m3: float = 1.0e-12,
) -> jnp.ndarray:
    """Return the local terminal-speed magnitude."""

    rho = jnp.maximum(jnp.asarray(density_kg_per_m3), density_floor_kg_per_m3)
    num = 2.0 * mass_kg * gravity_m_per_s2
    den = rho * drag_coefficient * reference_area_m2
    return jnp.sqrt(num / den)


def drag_power_dissipation(
    density_kg_per_m3: float,
    velocity_m_per_s: float,
    drag_coefficient: float,
    reference_area_m2: float,
) -> jnp.ndarray:
    """Return the rate at which drag dissipates mechanical energy."""

    f = drag_force_magnitude(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    return f * jnp.abs(jnp.asarray(velocity_m_per_s))


def state_derivative(
    state: State,
    mass_kg: float,
    density_kg_per_m3: float,
    drag_coefficient: float,
    reference_area_m2: float,
    gravity_m_per_s2: float,
) -> StateDerivative:
    """Return the instantaneous derivatives of altitude and velocity."""

    f_drag = signed_drag_force(
        density_kg_per_m3=density_kg_per_m3,
        velocity_m_per_s=state.velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    acc = gravity_m_per_s2 + f_drag / mass_kg
    return StateDerivative(
        altitude_rate_m_per_s=-jnp.asarray(state.velocity_m_per_s),
        velocity_rate_m_per_s2=acc,
    )


def speed_of_sound(temperature_k: float) -> jnp.ndarray:
    """Return local speed of sound in air in m/s."""

    gamma = 1.4
    r_spec = 287.05
    t = jnp.maximum(jnp.asarray(temperature_k), 3.0)
    return jnp.sqrt(gamma * r_spec * t)


def mach_number(speed_m_per_s: float, speed_of_sound_m_per_s: float) -> jnp.ndarray:
    """Return the Mach number."""

    v = jnp.abs(jnp.asarray(speed_m_per_s))
    c = jnp.maximum(jnp.asarray(speed_of_sound_m_per_s), 1.0e-3)
    return v / c


def recovery_temperature(
    ambient_temp_k: float,
    mach: float,
    recovery_factor: float = 0.84,
) -> jnp.ndarray:
    """Return the recovery (stagnation) temperature in kelvin."""

    gamma = 1.4
    t_inf = jnp.asarray(ambient_temp_k)
    return t_inf * (1.0 + recovery_factor * 0.5 * (gamma - 1.0) * mach**2)


def air_viscosity_sutherland(temperature_k: float) -> jnp.ndarray:
    """Return air dynamic viscosity in Pa s using Sutherland's law."""

    mu0 = 1.716e-5
    t0 = 273.15
    s = 110.4
    t = jnp.maximum(jnp.asarray(temperature_k), 3.0)
    num = (t / t0) ** 1.5 * (t0 + s)
    den = t + s
    return mu0 * num / den


def air_conductivity(viscosity_pa_s: float) -> jnp.ndarray:
    """Return air thermal conductivity in W/(m K)."""

    cp_air = 1005.0
    pr = 0.71
    mu = jnp.asarray(viscosity_pa_s)
    return mu * cp_air / pr


def convective_heat_transfer_coefficient(
    density_kg_per_m3: float,
    speed_m_per_s: float,
    diameter_m: float,
    viscosity_pa_s: float,
    conductivity_w_per_m_k: float,
) -> jnp.ndarray:
    """Return convective heat transfer coefficient using Ranz-Marshall."""

    rho = jnp.maximum(jnp.asarray(density_kg_per_m3), 0.0)
    v = jnp.abs(jnp.asarray(speed_m_per_s))
    d = jnp.maximum(jnp.asarray(diameter_m), 1.0e-6)
    mu = jnp.maximum(jnp.asarray(viscosity_pa_s), 1.0e-8)
    k_f = jnp.maximum(jnp.asarray(conductivity_w_per_m_k), 1.0e-4)
    pr = 0.71

    re = rho * v * d / mu
    nu = 2.0 + 0.6 * jnp.sqrt(re) * (pr ** (1.0 / 3.0))
    return nu * k_f / d


def net_surface_heat_flux(
    h_c: float,
    t_recovery: float,
    t_surface: float,
    t_ambient: float,
    emissivity: float = 0.95,
) -> jnp.ndarray:
    """Return net surface heat flux in W/m2 (positive = entering chicken)."""

    sigma = 5.670374419e-8
    q_conv = h_c * (t_recovery - t_surface)
    q_rad = emissivity * sigma * (t_surface**4 - t_ambient**4)
    return q_conv - q_rad
