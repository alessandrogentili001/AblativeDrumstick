"""Core physics helpers for the falling-chicken model."""

from typing import NamedTuple

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
