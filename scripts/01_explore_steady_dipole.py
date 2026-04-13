# %% 
# This cell ensures we can import the src package
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
# This cell is just a quick scratch space to test the RK4 integrator on the steady dipole flow before we set up the full benchmark suite in the next cell.
import numpy as np
import matplotlib.pyplot as plt

from src.flows import (
    steady_dipole_rhs,
    steady_dipole_velocity,
    steady_dipole_streamfunction,
)
from src.solvers import euler_integrate, rk4_integrate, rkf45_integrate
from src.benchmarks import run_benchmark_suite
from src.plotting import (
    plot_stream_field,
    plot_trajectory,
    plot_step_sizes,
)

# %%
# Problem setup
y0 = np.array([0.2, 0.3])
t_span = (0.0, 8.0)

fixed_dts = [0.2, 0.1, 0.05, 0.025]
rkf_tols = [1e-4, 1e-6, 1e-8]

# %%
# Run solvers
euler = euler_integrate(steady_dipole_rhs, t_span, y0, dt=0.05)
rk4 = rk4_integrate(steady_dipole_rhs, t_span, y0, dt=0.05)
rkf45 = rkf45_integrate(
    steady_dipole_rhs,
    t_span,
    y0,
    h0=0.05,
    hmin=1e-5,
    hmax=0.1,
    tol=1e-8,
)

print("Euler final state:", euler.y[-1])
print("RK4 final state:", rk4.y[-1])
print("RKF45 final state:", rkf45.y[-1])
print("RKF45 accepted/rejected:", rkf45.accepted_steps, rkf45.rejected_steps)

# %%
# Plot stream field and trajectories
fig, ax = plt.subplots(figsize=(8, 6))
plot_stream_field(
    steady_dipole_velocity,
    xlim=(-1.2, 1.2),
    ylim=(-1.2, 1.2),
    n=200,
    ax=ax,
    density=1.3,
)

plot_trajectory(euler, ax=ax, label="Euler", linewidth=1.5)
plot_trajectory(rk4, ax=ax, label="RK4", linewidth=1.8)
plot_trajectory(rkf45, ax=ax, label="RKF45", linewidth=1.8)

ax.legend()
ax.set_title("Steady dipole flow: trajectory comparison")
fig.tight_layout()
fig.savefig(FIG_DIR / "steady_dipole_trajectories.png", dpi=200)
plt.show()

# %%
# Benchmark suite against SciPy DOP853 reference
df = run_benchmark_suite(
    steady_dipole_rhs,
    t_span,
    y0,
    fixed_dts=fixed_dts,
    rkf_tols=rkf_tols,
    invariant_fn=steady_dipole_streamfunction,
)

display_cols = [
    "method",
    "setting_name",
    "setting_value",
    "final_error",
    "max_path_error",
    "rms_path_error",
    "runtime_sec",
    "nfev",
    "accepted_steps",
    "rejected_steps",
    "invariant_drift",
]
print(df[display_cols].to_string(index=False))

# %%
# Fixed-step convergence: Euler vs RK4
fixed_df = df[df["setting_name"] == "dt"].copy()

fig, ax = plt.subplots(figsize=(7, 5))
for method, group in fixed_df.groupby("method"):
    group = group.sort_values("setting_value")
    ax.loglog(
        group["setting_value"],
        group["max_path_error"],
        marker="o",
        label=method,
    )

ax.set_xlabel("dt")
ax.set_ylabel("max path error")
ax.set_title("Fixed-step convergence")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fixed_step_convergence.png", dpi=200)
plt.show()

# %%
# Cross-method comparison: runtime vs error
fig, ax = plt.subplots(figsize=(7, 5))
for method, group in df.groupby("method"):
    ax.loglog(
        group["runtime_sec"],
        group["max_path_error"],
        marker="o",
        linestyle="none",
        label=method,
    )

ax.set_xlabel("runtime (s)")
ax.set_ylabel("max path error")
ax.set_title("Accuracy-cost trade-off")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "runtime_vs_error.png", dpi=200)
plt.show()

# %%
# Adaptive step sizes for RKF45
fig, ax = plt.subplots(figsize=(8, 4))
plot_step_sizes(rkf45, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "rkf45_step_sizes.png", dpi=200)
plt.show()