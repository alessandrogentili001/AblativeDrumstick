"""Atmospheric state functions for a compact layered model."""

import jax
import jax.numpy as jnp

from chicken_from_space.config import Config

_T_REF = 288.15
_N_INT = 160

_H_NODES = jnp.asarray(
    [
        0.0,
        11_000.0,
        20_000.0,
        32_000.0,
        47_000.0,
        51_000.0,
        71_000.0,
        86_000.0,
        91_000.0,
        110_000.0,
        120_000.0,
        200_000.0,
    ],
    dtype=jnp.float64,
)
_T_NODES = jnp.asarray(
    [
        288.15,
        216.65,
        216.65,
        228.65,
        270.65,
        270.65,
        214.65,
        186.87,
        186.87,
        240.0,
        360.0,
        1_000.0,
    ],
    dtype=jnp.float64,
)


def _cfg(config: Config | None) -> Config:
    """Return the provided config or the default one."""

    return config or Config()


def _clip(h: float | jnp.ndarray) -> jnp.ndarray:
    """Clamp altitude to non-negative values."""

    return jnp.maximum(jnp.asarray(h, dtype=jnp.float64), 0.0)


@jax.jit
def geopotential_altitude(
    altitude_m: float | jnp.ndarray,
    config: Config | None = None,
) -> jnp.ndarray:
    """Convert geometric altitude to geopotential altitude."""

    cfg = _cfg(config)
    h = _clip(altitude_m)
    r = cfg.constants.earth_radius_m
    return r * h / (r + h)


@jax.jit
def atmospheric_temperature(
    altitude_m: float | jnp.ndarray,
    config: Config | None = None,
) -> jnp.ndarray:
    """Return atmospheric temperature in kelvin."""

    cfg = _cfg(config)
    h = _clip(altitude_m)
    dt = cfg.atmosphere.sea_level_temperature_k - _T_REF
    t_nodes = (_T_NODES + dt).at[-1].set(
        cfg.atmosphere.thermosphere_asymptotic_temperature_k,
    )
    return jnp.maximum(jnp.interp(h, _H_NODES, t_nodes), 3.0)


def _pressure_one(h: float | jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Return pressure at one altitude from hydrostatic balance."""

    h = _clip(h)
    xs = jnp.linspace(0.0, h, _N_INT)
    ts = atmospheric_temperature(xs, cfg)
    g = cfg.constants.gravity_m_per_s2
    m = cfg.atmosphere.molar_mass_kg_per_mol
    r = cfg.constants.universal_gas_constant_j_per_mol_k
    integral = jnp.trapezoid(m * g / (r * ts), xs)
    return cfg.atmosphere.sea_level_pressure_pa * jnp.exp(-integral)


@jax.jit
def atmospheric_pressure(
    altitude_m: float | jnp.ndarray,
    config: Config | None = None,
) -> jnp.ndarray:
    """Return atmospheric pressure in pascals."""

    cfg = _cfg(config)
    h = jnp.asarray(altitude_m, dtype=jnp.float64)

    if h.ndim == 0:
        return _pressure_one(h, cfg)

    flat = h.reshape(-1)
    vals = jax.vmap(lambda x: _pressure_one(x, cfg))(flat)
    return vals.reshape(h.shape)


@jax.jit
def atmospheric_density(
    altitude_m: float | jnp.ndarray,
    config: Config | None = None,
) -> jnp.ndarray:
    """Return atmospheric density from the ideal-gas relation."""

    cfg = _cfg(config)
    p = atmospheric_pressure(altitude_m, cfg)
    t = atmospheric_temperature(altitude_m, cfg)
    m = cfg.atmosphere.molar_mass_kg_per_mol
    r = cfg.constants.universal_gas_constant_j_per_mol_k
    return p * m / (r * t)


def density_at_altitude(
    altitude_m: float | jnp.ndarray,
    config: Config | None = None,
) -> jnp.ndarray:
    """Compatibility wrapper returning density at altitude."""

    return atmospheric_density(altitude_m=altitude_m, config=config)
