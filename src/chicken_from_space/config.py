"""Configuration objects for the chicken-from-space simulation."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtmosphereConfig:
    """Parameters for a simple exponential atmosphere."""

    sea_level_density_kg_per_m3: float = 1.225
    scale_height_m: float = 8_500.0
    min_density_kg_per_m3: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.sea_level_density_kg_per_m3 <= 0.0:
            raise ValueError("Sea-level density must be positive.")
        if self.scale_height_m <= 0.0:
            raise ValueError("Scale height must be positive.")
        if self.min_density_kg_per_m3 <= 0.0:
            raise ValueError("Minimum density must be positive.")


@dataclass(frozen=True)
class ChickenConfig:
    """Geometric and effective energy-coupling properties of the chicken."""

    mass_kg: float = 1.5
    equivalent_radius_m: float = 0.12
    drag_coefficient: float = 0.47
    energy_transfer_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("Mass must be positive.")
        if self.equivalent_radius_m <= 0.0:
            raise ValueError("Equivalent radius must be positive.")
        if self.drag_coefficient <= 0.0:
            raise ValueError("Drag coefficient must be positive.")
        if not 0.0 <= self.energy_transfer_fraction <= 1.0:
            raise ValueError("Energy transfer fraction must lie in [0, 1].")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulation settings."""

    chicken: ChickenConfig = field(default_factory=ChickenConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    initial_altitude_m: float = 120_000.0
    initial_velocity_m_per_s: float = 0.0
    gravity_m_per_s2: float = 9.81
    time_step_s: float = 0.05
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-6
    max_time_s: float = 5_000.0
    max_solver_steps: int = 100_000

    def __post_init__(self) -> None:
        if self.initial_altitude_m < 0.0:
            raise ValueError("Initial altitude must be non-negative.")
        if self.gravity_m_per_s2 <= 0.0:
            raise ValueError("Gravity must be positive.")
        if self.time_step_s <= 0.0:
            raise ValueError("Initial solver step size must be positive.")
        if self.relative_tolerance <= 0.0:
            raise ValueError("Relative tolerance must be positive.")
        if self.absolute_tolerance <= 0.0:
            raise ValueError("Absolute tolerance must be positive.")
        if self.max_time_s <= 0.0:
            raise ValueError("Maximum time must be positive.")
        if self.max_solver_steps <= 0:
            raise ValueError("Maximum solver steps must be positive.")
