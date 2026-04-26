"""Project configuration objects."""

import jax

from dataclasses import asdict, dataclass, field

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Constants:
    """Physical constants used by the simplified model."""

    gravity_m_per_s2: float = 9.8067
    earth_radius_m: float = 6_356_766.0
    universal_gas_constant_j_per_mol_k: float = 8.31446261815324


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Atmosphere:
    """Parameters for the atmosphere model."""

    sea_level_pressure_pa: float = 101_325.0
    sea_level_temperature_k: float = 288.0
    molar_mass_kg_per_mol: float = 0.0289644
    thermosphere_asymptotic_temperature_k: float = 1_000.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Chicken:
    """Effective geometric parameters for the chicken."""

    mass_kg: float = 1.25
    equivalent_radius_m: float = 0.12
    drag_coefficient: float = 0.47 # sphere (for a square is ~1.)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Simulation:
    """Numerical and initial-condition parameters."""

    initial_altitude_m: float = 120_000.0
    initial_velocity_m_per_s: float = 0.0
    initial_time_step_s: float = 0.05
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-6
    max_time_s: float = 5_000.0
    max_solver_steps: int = 100_000
    energy_transfer_fraction: float = 0.10


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    constants: Constants = field(default_factory=Constants)
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    chicken: Chicken = field(default_factory=Chicken)
    simulation: Simulation = field(default_factory=Simulation)

    def to_dict(self) -> dict[str, object]:
        """Return a plain nested dictionary representation."""

        return asdict(self)

    def to_values(self) -> dict[str, object]:
        """Compatibility alias for code that expects scalar dictionaries."""

        return self.to_dict()
