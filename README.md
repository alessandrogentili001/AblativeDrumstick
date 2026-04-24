# Chicken From Space

`chicken_from_space` is a small Python project that explores a deliberately odd physics question:

Could a chicken be "cooked" by dropping it from space and letting atmospheric drag dissipate energy during re-entry?

This first version does **not** model cooking or internal heat transfer. It only simulates a vertical fall through a simplified Earth atmosphere and estimates how much drag energy is dissipated, plus a configurable fraction of that energy that might couple into the chicken.

## Main assumptions

- One-dimensional vertical motion only.
- Downward velocity is taken as positive.
- The chicken is approximated as an equivalent sphere.
- Drag is quadratic in speed.
- Atmospheric density follows a simple exponential model.
- Gravity is constant.
- Time integration uses a Diffrax adaptive explicit Runge-Kutta solver (`Dopri5`).
- A fixed fraction of drag-dissipated energy is treated as energy transferred to the chicken.

## Model limits

- No lift, tumbling, or attitude dynamics.
- No compressibility or supersonic corrections.
- No altitude-dependent gravity.
- No shock-layer or ablation physics.
- No thermal model and no internal cooking model yet.

This makes the code intentionally modest, readable, and easy to extend.

## Installation

Create or activate a Python 3.11 environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

If you are using the `cock` conda environment created for this repo:

```bash
conda run -n cock python -m pip install -e ".[dev]"
```

## Run a basic simulation

```bash
conda run -n cock python scripts/run_basic_simulation.py
```

Optionally save the trajectory and diagnostic arrays:

```bash
conda run -n cock python scripts/run_basic_simulation.py --output results/basic_run.npz
conda run -n cock python scripts/run_basic_simulation.py --output results/basic_run.csv
```

## What the simulation reports

- Vertical trajectory
- Velocity over time
- Atmospheric density along the path
- Drag force
- Local terminal velocity
- Drag power dissipation
- Total drag-dissipated energy
- Estimated energy transferred to the chicken

## Why JAX

The code uses `jax.numpy` for the numerical pieces while keeping the control flow simple and readable. That keeps the project easy to understand now and leaves a natural path toward future JIT compilation, batched parameter sweeps, `lax.scan`, and hardware acceleration.

For local development on macOS, the starter setup uses a standard CPU JAX install. A future scaling pass can swap in the platform-specific accelerator backend you want to target.

## License

MIT. See [LICENSE](LICENSE).

## Next steps

- More realistic atmosphere models
- Altitude-dependent gravity
- Drag coefficient changes with Reynolds/Mach number
- Compressibility and supersonic effects
- Thermal coupling and chicken temperature model
- Internal cooking model
- Plotting and notebooks
- Parameter sweeps and reporting
- Richer docs, tutorials, and analysis outputs
