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

    radius = jnp.maximum(jnp.asarray(radius_m), 0.0)
    return jnp.pi * radius**2


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

    density = jnp.maximum(jnp.asarray(density_kg_per_m3), 0.0)
    speed = jnp.abs(jnp.asarray(speed_m_per_s))
    return 0.5 * density * drag_coefficient * reference_area_m2 * speed**2


def signed_drag_force(
    density_kg_per_m3: float,
    velocity_m_per_s: float,
    drag_coefficient: float,
    reference_area_m2: float,
) -> jnp.ndarray:
    """Return drag force in the downward-positive sign convention."""

    magnitude = drag_force_magnitude(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    velocity = jnp.asarray(velocity_m_per_s)
    return -magnitude * jnp.sign(velocity)


def local_terminal_velocity(
    mass_kg: float,
    density_kg_per_m3: float,
    gravity_m_per_s2: float,
    drag_coefficient: float,
    reference_area_m2: float,
    density_floor_kg_per_m3: float = 1.0e-12,
) -> jnp.ndarray:
    """Return the local terminal-speed magnitude."""

    density = jnp.maximum(jnp.asarray(density_kg_per_m3), density_floor_kg_per_m3)
    numerator = 2.0 * mass_kg * gravity_m_per_s2
    denominator = density * drag_coefficient * reference_area_m2
    return jnp.sqrt(numerator / denominator)


def drag_power_dissipation(
    density_kg_per_m3: float,
    velocity_m_per_s: float,
    drag_coefficient: float,
    reference_area_m2: float,
) -> jnp.ndarray:
    """Return the rate at which drag dissipates mechanical energy."""

    drag_force = drag_force_magnitude(
        density_kg_per_m3=density_kg_per_m3,
        speed_m_per_s=velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    return drag_force * jnp.abs(jnp.asarray(velocity_m_per_s))


def state_derivative(
    state: State,
    mass_kg: float,
    density_kg_per_m3: float,
    drag_coefficient: float,
    reference_area_m2: float,
    gravity_m_per_s2: float,
) -> StateDerivative:
    """Return the instantaneous derivatives of altitude and velocity."""

    drag_force = signed_drag_force(
        density_kg_per_m3=density_kg_per_m3,
        velocity_m_per_s=state.velocity_m_per_s,
        drag_coefficient=drag_coefficient,
        reference_area_m2=reference_area_m2,
    )
    acceleration = gravity_m_per_s2 + drag_force / mass_kg
    return StateDerivative(
        altitude_rate_m_per_s=-jnp.asarray(state.velocity_m_per_s),
        velocity_rate_m_per_s2=acceleration,
    )
