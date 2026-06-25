"""Run a thermal simulation to determine cooking times and threshold drop altitudes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# pyrefly: ignore [missing-import]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# pyrefly: ignore [missing-import]
from chicken_from_space.config import Config, Simulation
# pyrefly: ignore [missing-import]
from chicken_from_space.simulation import run_simulation, summarize_simulation


def find_minimum_cooking_altitude(
    config: Config,
    low_alt_m: float = 120_000.0,
    high_alt_m: float = 600_000.0,
    tolerance_m: float = 1_000.0,
) -> float | None:
    """Find the minimum altitude required to cook the chicken core via binary search."""

    print("Searching for the minimum drop altitude required to cook the chicken...")
    
    # Verify if even the highest altitude cooks the chicken
    cfg_high = replace(config, simulation=replace(config.simulation, initial_altitude_m=high_alt_m))
    res_high = run_simulation(cfg_high)
    sum_high = summarize_simulation(res_high)
    if not sum_high.cooked:
        print(f"Chicken does not cook even when dropped from {high_alt_m / 1000:.1f} km.")
        return None

    # Binary search
    low = low_alt_m
    high = high_alt_m
    while high - low > tolerance_m:
        mid = 0.5 * (low + high)
        cfg_mid = replace(config, simulation=replace(config.simulation, initial_altitude_m=mid))
        res_mid = run_simulation(cfg_mid)
        sum_mid = summarize_simulation(res_mid)
        if sum_mid.cooked:
            high = mid
        else:
            low = mid

    return high


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run thermal chicken simulation and find cooking thresholds.",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=None,
        help="Specific initial drop altitude in meters. If not provided, a threshold search is performed.",
    )
    return parser.parse_args()


def main() -> None:
    """Run simulation and print cooking results."""

    args = parse_args()
    config = Config()

    if args.altitude is not None:
        # Run a specific user-requested altitude
        cfg = replace(config, simulation=replace(config.simulation, initial_altitude_m=args.altitude))
        print(f"Running simulation for a drop from {args.altitude / 1000:.1f} km...")
        result = run_simulation(cfg)
        summary = summarize_simulation(result)
        
        print("\nSimulation Results:")
        print("-------------------")
        print(f"Initial Altitude: {args.altitude / 1000:.1f} km")
        print(f"Max Core Temp: {summary.max_center_temperature_k - 273.15:.2f} C")
        print(f"Max Surface Temp: {summary.max_surface_temperature_k - 273.15:.2f} C")
        print(f"Final Core Temp at Ground (Landing): {float(result.temperature_k[-1, 0]) - 273.15:.2f} C")
        print(f"Final Surface Temp at Ground (Landing): {float(result.temperature_k[-1, -1]) - 273.15:.2f} C")
        print(f"Perfectly cooked: {'Yes' if summary.cooked else 'No'}")
        if summary.cooked:
            print(f"Time required to cook: {summary.cook_time_s:,.2f} s")
            print(f"Cooking altitude: {summary.cook_altitude_m:,.2f} m")
            print(f"Cooking velocity: {summary.cook_velocity_m_per_s:,.2f} m/s")
    else:
        # Perform a threshold search
        min_alt = find_minimum_cooking_altitude(config)
        if min_alt is not None:
            print(f"\nFound minimum cooking drop altitude: {min_alt / 1000:.2f} km")
            print("Running simulation at this minimum cooking altitude...")
            
            cfg_min = replace(config, simulation=replace(config.simulation, initial_altitude_m=min_alt))
            result = run_simulation(cfg_min)
            summary = summarize_simulation(result)
            
            print("\nThreshold Simulation Results:")
            print("-----------------------------")
            print(f"Drop Altitude: {min_alt / 1000:.2f} km")
            print(f"Total Fall Time to Ground: {summary.total_fall_time_s:,.2f} s")
            print(f"Max Core Temp reached: {summary.max_center_temperature_k - 273.15:.2f} C")
            print(f"Max Surface Temp reached: {summary.max_surface_temperature_k - 273.15:.2f} C")
            print(f"Final Core Temp at Ground (Landing): {float(result.temperature_k[-1, 0]) - 273.15:.2f} C")
            print(f"Final Surface Temp at Ground (Landing): {float(result.temperature_k[-1, -1]) - 273.15:.2f} C")
            print(f"Time from release until perfectly cooked: {summary.cook_time_s:,.2f} s")
            print(f"Altitude when perfectly cooked: {summary.cook_altitude_m:,.2f} m")
            print(f"Velocity when perfectly cooked: {summary.cook_velocity_m_per_s:,.2f} m/s")
        else:
            # Fallback to run a default simulation at the highest search limit (600 km)
            fallback_alt_m = 600_000.0
            print(f"\nRunning a fallback simulation from {fallback_alt_m / 1000:.1f} km to inspect ground temperatures...")
            cfg_fallback = replace(config, simulation=replace(config.simulation, initial_altitude_m=fallback_alt_m))
            result = run_simulation(cfg_fallback)
            summary = summarize_simulation(result)
            
            print("\nSimulation Results:")
            print("-------------------")
            print(f"Drop Altitude: {fallback_alt_m / 1000:.1f} km")
            print(f"Total Fall Time to Ground: {summary.total_fall_time_s:,.2f} s")
            print(f"Max Core Temp reached: {summary.max_center_temperature_k - 273.15:.2f} C")
            print(f"Max Surface Temp reached: {summary.max_surface_temperature_k - 273.15:.2f} C")
            print(f"Final Core Temp at Ground (Landing): {float(result.temperature_k[-1, 0]) - 273.15:.2f} C")
            print(f"Final Surface Temp at Ground (Landing): {float(result.temperature_k[-1, -1]) - 273.15:.2f} C")
            print(f"Perfectly cooked: No")


if __name__ == "__main__":
    main()
