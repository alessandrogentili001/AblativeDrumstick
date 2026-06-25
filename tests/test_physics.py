"""Tests for the physics helpers."""

# pyrefly: ignore [missing-import]
import jax
import pytest

# pyrefly: ignore [missing-import]
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


def test_drag_force_is_zero_at_zero_speed() -> None:
    """Quadratic drag should vanish at zero speed."""

    area = float(cross_sectional_area(0.1))
    drag_force = float(
        drag_force_magnitude(
            density_kg_per_m3=1.225,
            speed_m_per_s=0.0,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert drag_force == pytest.approx(0.0)


def test_terminal_velocity_rises_as_density_drops() -> None:
    """Lower density should increase the local terminal speed."""

    area = float(cross_sectional_area(0.1))
    dense_air = float(
        local_terminal_velocity(
            mass_kg=1.5,
            density_kg_per_m3=1.225,
            gravity_m_per_s2=9.81,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )
    thin_air = float(
        local_terminal_velocity(
            mass_kg=1.5,
            density_kg_per_m3=0.1,
            gravity_m_per_s2=9.81,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert thin_air > dense_air > 0.0


def test_drag_power_is_non_negative() -> None:
    """Dissipated drag power should be non-negative."""

    area = float(cross_sectional_area(0.1))
    power = float(
        drag_power_dissipation(
            density_kg_per_m3=1.225,
            velocity_m_per_s=-50.0,
            drag_coefficient=0.47,
            reference_area_m2=area,
        ),
    )

    assert power >= 0.0


def test_drag_force_can_be_jitted() -> None:
    """Physics helpers should work under direct JIT."""

    area = float(cross_sectional_area(0.1))
    drag_fn = jax.jit(drag_force_magnitude)

    assert float(drag_fn(1.225, 10.0, 0.47, area)) > 0.0


def test_composed_physics_can_be_jitted() -> None:
    """JAX should compile a composed function that calls plain helpers."""

    def power_from_radius(radius_m: float) -> float:
        area = cross_sectional_area(radius_m)
        return drag_power_dissipation(1.225, 50.0, 0.47, area)

    assert float(jax.jit(power_from_radius)(0.1)) > 0.0


def test_speed_of_sound_rises_with_temperature() -> None:
    """Speed of sound should rise with temperature."""

    c_cold = float(speed_of_sound(200.0))
    c_hot = float(speed_of_sound(400.0))
    assert c_hot > c_cold
    assert c_cold == pytest.approx(283.4, abs=0.2)


def test_mach_and_recovery_temperature() -> None:
    """Mach and recovery temperature should scale with velocity."""

    c = speed_of_sound(300.0)
    mach = float(mach_number(2.0 * c, c))
    assert mach == pytest.approx(2.0)

    t_rec = float(recovery_temperature(300.0, mach))
    assert t_rec > 300.0
    # recovery = T * (1 + 0.84 * 0.2 * 4) = 300 * (1 + 0.672) = 501.6
    assert t_rec == pytest.approx(501.6, abs=0.1)


def test_air_properties_sutherland() -> None:
    """Viscosity and conductivity should increase at higher temperature."""

    mu_cold = float(air_viscosity_sutherland(273.15))
    mu_hot = float(air_viscosity_sutherland(373.15))
    assert mu_hot > mu_cold

    k_cold = float(air_conductivity(mu_cold))
    k_hot = float(air_conductivity(mu_hot))
    assert k_hot > k_cold


def test_convective_heat_transfer_ranz_marshall() -> None:
    """Heat transfer coefficient should rise with velocity."""

    mu = air_viscosity_sutherland(288.15)
    k_f = air_conductivity(mu)

    h_slow = float(
        convective_heat_transfer_coefficient(
            density_kg_per_m3=1.225,
            speed_m_per_s=10.0,
            diameter_m=0.24,
            viscosity_pa_s=mu,
            conductivity_w_per_m_k=k_f,
        )
    )

    h_fast = float(
        convective_heat_transfer_coefficient(
            density_kg_per_m3=1.225,
            speed_m_per_s=100.0,
            diameter_m=0.24,
            viscosity_pa_s=mu,
            conductivity_w_per_m_k=k_f,
        )
    )

    assert h_fast > h_slow > 0.0


def test_net_surface_heat_flux() -> None:
    """Heat flux should be negative when chicken is hotter than environment (radiative cooling)."""

    q_cool = float(
        net_surface_heat_flux(
            h_c=0.0,  # no convection
            t_recovery=300.0,
            t_surface=400.0,
            t_ambient=300.0,
            emissivity=0.95,
        )
    )

    # Only radiative cooling should occur: - emissivity * sigma * (400^4 - 300^4)
    assert q_cool < 0.0
