"""Run a basic chicken-from-space simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chicken_from_space.config import SimulationConfig
from chicken_from_space.simulation import (
    format_summary,
    run_simulation,
    save_simulation_data,
    summarize_simulation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a basic vertical fall simulation for a chicken dropped from space.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path ending in .npz or .csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the default simulation and print a compact summary."""

    args = parse_args()
    result = run_simulation(SimulationConfig())
    summary = summarize_simulation(result)
    print(format_summary(summary))

    if args.output is not None:
        saved_path = save_simulation_data(result, args.output)
        print(f"\nSaved simulation data to: {saved_path}")


if __name__ == "__main__":
    main()
