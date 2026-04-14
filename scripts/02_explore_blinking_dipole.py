# %%
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%
import numpy as np
import matplotlib.pyplot as plt

from src.flows import blinking_vortex_rhs
from src.ftle import ftle_grid, nearby_separation
from src.solvers import rkf45_integrate
from src.plotting import plot_ftle_field, plot_trajectory, plot_step_sizes

# %%
# Problem setup
# Keep the flow parameters explicit so the notebook reads like a deliberate experiment.
flow_kwargs = {
    "alpha": 1.0,
    "beta0": 0.4,
    "period": 1.0,
}

t_span = (0.0, 4.0)
xlim = (-1.0, 1.0)
ylim = (-1.0, 1.0)

# FTLE settings
# This implementation computes the FTLE field point-by-point, so use a moderate
# grid while iterating. Once you're happy with the framing, you can increase this.
nx = 180
ny = 180
delta0 = (1e-6, 0.0)

ftle_solver_kwargs = {
    "tol": 1e-7,
    "h0": 0.005,
    "hmin": 1e-5,
    "hmax": 0.02,
}

# Representative trajectory / diagnostic settings
representative_y0 = np.array([-0.75, 0.20])

sample_y0s = [
    np.array([-0.85, -0.20]),
    np.array([-0.85,  0.00]),
    np.array([-0.85,  0.20]),
    np.array([-0.20,  0.65]),
    np.array([ 0.20, -0.65]),
]

# %%
# Compute FTLE field
xs, ys, field = ftle_grid(
    blinking_vortex_rhs,
    t_span=t_span,
    xlim=xlim,
    ylim=ylim,
    nx=nx,
    ny=ny,
    delta0=delta0,
    solver="rkf45",
    solver_kwargs=ftle_solver_kwargs,
    rhs_kwargs=flow_kwargs,
    progress=True,
)

print("FTLE field shape:", field.shape)
print("FTLE min / max:", np.nanmin(field), np.nanmax(field))

# %%
# Figure 1: clean FTLE field
fig, ax = plt.subplots(figsize=(7, 6))
plot_ftle_field(xs, ys, field, ax=ax)
ax.set_title("Blinking vortex flow: FTLE field")
fig.tight_layout()
fig.savefig(FIG_DIR / "blinking_vortex_ftle.png", dpi=300)
plt.show()

# %%
# Figure 2: FTLE field with representative trajectories overlaid
fig, ax = plt.subplots(figsize=(7, 6))
plot_ftle_field(xs, ys, field, ax=ax)

for k, y0 in enumerate(sample_y0s, start=1):
    traj = rkf45_integrate(
        blinking_vortex_rhs,
        t_span,
        y0,
        **ftle_solver_kwargs,
        **flow_kwargs,
    )
    plot_trajectory(
        traj,
        ax=ax,
        label=f"seed {k}",
        linewidth=1.2,
        alpha=0.9,
    )

seed_array = np.vstack(sample_y0s)
ax.scatter(seed_array[:, 0], seed_array[:, 1], s=18, marker="o")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_title("Blinking vortex flow: FTLE field with sample trajectories")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "blinking_vortex_ftle_with_trajectories.png", dpi=300)
plt.show()

# %%
# Figure 3: nearby-trajectory separation for one representative initial condition
t_sep, sep = nearby_separation(
    blinking_vortex_rhs,
    t_span=t_span,
    y0=representative_y0,
    delta0=delta0,
    solver="rkf45",
    solver_kwargs=ftle_solver_kwargs,
    rhs_kwargs=flow_kwargs,
)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(t_sep, sep, linewidth=1.8)
ax.set_xlabel("t")
ax.set_ylabel("Separation distance")
ax.set_title("Blinking vortex flow: nearby-trajectory separation")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "blinking_vortex_nearby_separation.png", dpi=300)
plt.show()

# %%
# Representative RKF45 trajectory for diagnostics
traj_rep = rkf45_integrate(
    blinking_vortex_rhs,
    t_span,
    representative_y0,
    **ftle_solver_kwargs,
    **flow_kwargs,
)

print("Representative final state:", traj_rep.y[-1])
print(
    "Representative RKF45 accepted / rejected:",
    traj_rep.accepted_steps,
    traj_rep.rejected_steps,
)

# %%
# Figure 4: accepted step sizes for the representative adaptive run
fig, ax = plt.subplots(figsize=(8, 4))
plot_step_sizes(traj_rep, ax=ax)
ax.set_title("Blinking vortex flow: RKF45 accepted step sizes")
fig.tight_layout()
fig.savefig(FIG_DIR / "blinking_vortex_rkf45_step_sizes.png", dpi=300)
plt.show()