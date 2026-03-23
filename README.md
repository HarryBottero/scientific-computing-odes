# Scientific Computing ODEs

This repository implements and compares numerical ODE solvers for particle advection in two-dimensional vortex flows. The project focuses on Euler, RK4, and adaptive RKF45 methods, with comparisons against a high-accuracy SciPy benchmark.

## Current status

- [x] Repo initialised
- [ ] Steady dipole flow refactor
- [ ] Euler / RK4 benchmark
- [ ] RKF45 implementation
- [ ] Blinking vortex extension
- [ ] FTLE analysis

## Structure

- `src/`: reusable solver, flow, benchmarking, and plotting code
- `notebooks/`: exploratory and final analysis notebooks
- `tests/`: unit tests for numerical methods
- `figures/`: generated figures
