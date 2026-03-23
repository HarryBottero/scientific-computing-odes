# Scientific Computing ODEs

This project studies Lagrangian particle trajectories in analytically defined 2D vortex flows.
It implements Euler, RK4, and adaptive RKF45 solvers in pure Python, benchmarks them against SciPy's DOP853 solver, and extends the system to a blinking-vortex flow to explore chaotic advection and FTLE diagnostics.

## Repo map

- `src/flows.py` — steady and blinking vortex velocity fields, RHS wrappers, streamfunction
- `src/solvers.py` — Euler, RK4, RKF45 integrators
- `src/benchmarks.py` — reference solution, interpolation, error metrics, benchmarking tables
- `src/ftle.py` — nearby-trajectory divergence and FTLE field estimation
- `src/plotting.py` — figures for streamlines, trajectories, convergence, adaptive step sizes, FTLE heatmap
- `notebooks/01_exploration.ipynb` — scratch analysis and solver checks
- `notebooks/02_final_figures.ipynb` — clean figure-generation notebook for the README / report
- `tests/` — basic numerical and physics **checks**

## Current status

- [x] Repo initialised
- [ ] Steady dipole flow refactor
- [ ] Euler / RK4 benchmark
- [ ] RKF45 implementation
- [ ] Blinking vortex extension
- [ ] FTLE analysis

## Questions this repo is set to answer

- How much more accurate is RK4 than Euler at the same step size?
- How does adaptive RKF45 trade off tolerance against runtime?
- Does the steady-flow streamfunction remain approximately conserved numerically?
- Where in the blinking-vortex flow do nearby trajectories separate fastest?
