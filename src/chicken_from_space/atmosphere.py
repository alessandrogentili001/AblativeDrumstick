"""Atmospheric density models."""

import jax.numpy as jnp

from chicken_from_space.config import AtmosphereConfig


def density_at_altitude(
    altitude_m: float,
    config: AtmosphereConfig,
) -> jnp.ndarray:
    """Return density from a simple exponential atmosphere."""

    clipped_altitude_m = jnp.maximum(jnp.asarray(altitude_m), 0.0)
    density = config.sea_level_density_kg_per_m3 * jnp.exp(
        -clipped_altitude_m / config.scale_height_m,
    )
    return jnp.maximum(density, config.min_density_kg_per_m3)
