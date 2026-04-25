# AblativeDrumstick

`AblativeDrumstick` is a physics playground for an extremely serious scientific question:

Can a chicken be cooked by dropping it from space?

Not metaphorically. Not poetically. Literally.

This repository starts with the first thing we need before anyone talks about roasting, browning, or food safety: a clean numerical model of atmospheric fall and drag dissipation.

## What This Version Does

This first version simulates a chicken falling vertically through Earth’s atmosphere and estimates:

- altitude over time
- velocity over time
- atmospheric density along the path
- drag force
- local terminal velocity
- drag power dissipation
- total drag-dissipated energy
- a configurable estimate of how much of that energy reaches the chicken

It does **not** model internal heating or cooking yet. This is the “how hard did the atmosphere bully the drumstick?” phase.

## Working Model

The current model is intentionally simple and easy to extend:

- 1D vertical motion only
- downward velocity treated as positive
- chicken approximated as an equivalent sphere
- exponential atmosphere
- constant gravity
- quadratic drag
- adaptive Diffrax time integration
- fixed energy-transfer fraction from drag dissipation to chicken

The code is meant to stay readable first, fancy second.

## What It Leaves Out

For now, this model ignores:

- lift, tumbling, and attitude changes
- compressibility and supersonic drag corrections
- altitude-dependent gravity
- shock-layer chemistry
- ablation, charring, or mass loss
- thermal diffusion inside the chicken
- actual cooking

So no, the repo does not yet answer whether the center reaches a safe temperature. It only tells us how violent the fall is.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run

Run the default simulation:

```bash
python scripts/run_basic_simulation.py
```

Save the output data if you want to inspect it later:

```bash
python scripts/run_basic_simulation.py --output results/basic_run.npz
python scripts/run_basic_simulation.py --output results/basic_run.csv
```

## Why JAX

The numerical pieces use `jax.numpy`, and the solver is built on Diffrax. The current code is still small and human-readable, but the structure is pointing toward the future:

- accelerator-friendly parameter sweeps
- larger batches of simulations
- more ambitious thermal models
- cleaner scaling once the project gets weird enough to deserve it

## Where This Is Going

Natural next steps include:

- better atmosphere models
- altitude-dependent gravity
- more realistic drag behavior
- compressible and high-Mach effects
- thermal coupling to the chicken surface
- internal cooking / heat diffusion
- plotting, notebooks, and reports
- parameter studies and sensitivity analysis

## License

MIT. See [LICENSE](LICENSE).
