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
# This is just a quick scratch space to test the RK4 integrator on the steady dipole flow before we set up the full benchmark suite in the next cell.
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

# Dense geometric sweep for fixed-step methods.
fixed_dts = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]

# Denser tolerance sweep so the RKF45 trade-off and rejection-rate plots
# look properly sampled rather than tokenistic.
rkf_tols = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]

# Use one representative RKF45 run for trajectory + step-size diagnostics.
representative_tol = 1e-6

# %%
# Run representative solver instances
euler = euler_integrate(steady_dipole_rhs, t_span, y0, dt=0.05)
rk4 = rk4_integrate(steady_dipole_rhs, t_span, y0, dt=0.05)
rkf45 = rkf45_integrate(
    steady_dipole_rhs,
    t_span,
    y0,
    h0=0.05,
    hmin=1e-5,
    hmax=0.1,
    tol=representative_tol,
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

plot_trajectory(euler, ax=ax, label="Euler (dt=0.05)", linewidth=1.5)
plot_trajectory(rk4, ax=ax, label="RK4 (dt=0.05)", linewidth=1.8)
plot_trajectory(
    rkf45,
    ax=ax,
    label=f"RKF45 (tol={representative_tol:.0e})",
    linewidth=1.8,
)

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
# Split benchmark table into fixed-step and adaptive subsets for separate plotting
fixed = df[df["setting_name"] == "dt"].copy()
adaptive = df[df["setting_name"] == "tol"].copy()

# %%
# Main figure: work-precision diagram
fig, ax = plt.subplots(figsize=(7, 5))

for method, group in df.groupby("method"):
    group = group.sort_values("nfev")
    ax.loglog(
        group["nfev"],
        group["max_path_error"],
        marker="o",
        linewidth=1.8,
        label=method,
    )

ax.set_xlabel("Function evaluations (nfev)")
ax.set_ylabel("Maximum path error")
ax.set_title("Steady dipole benchmark: work-precision")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "steady_dipole_work_precision.png", dpi=200)
plt.show()

# %%
# Family-specific convergence plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Fixed-step methods: Euler and RK4
for method, group in fixed.groupby("method"):
    group = group.sort_values("setting_value")
    axes[0].loglog(
        group["setting_value"],
        group["max_path_error"],
        marker="o",
        linewidth=1.8,
        label=method,
    )

axes[0].set_xlabel("dt")
axes[0].set_ylabel("Maximum path error")
axes[0].set_title("Fixed-step convergence")
axes[0].grid(True, which="both", alpha=0.3)
axes[0].legend()

# Adaptive method: RKF45
adaptive_sorted = adaptive.sort_values("setting_value")
axes[1].loglog(
    adaptive_sorted["setting_value"],
    adaptive_sorted["max_path_error"],
    marker="o",
    linewidth=1.8,
    label="RKF45",
)

axes[1].set_xlabel("Tolerance")
axes[1].set_ylabel("Maximum path error")
axes[1].set_title("Adaptive tolerance sweep")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

fig.suptitle("Steady dipole benchmark convergence", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "steady_dipole_convergence_panels.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# Invariant preservation vs computational work
if df["invariant_drift"].notna().any():
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, group in df.groupby("method"):
        group = group.sort_values("nfev")
        ax.loglog(
            group["nfev"],
            group["invariant_drift"],
            marker="o",
            linewidth=1.8,
            label=method,
        )

    ax.set_xlabel("Function evaluations (nfev)")
    ax.set_ylabel("Max streamfunction drift")
    ax.set_title("Steady dipole benchmark: invariant preservation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "steady_dipole_invariant_drift.png", dpi=200)
    plt.show()

# %%
# RKF45 rejection ratio vs tolerance
adaptive = adaptive.sort_values("setting_value").copy()
adaptive["total_attempted_steps"] = adaptive["accepted_steps"] + adaptive["rejected_steps"]
adaptive["reject_ratio"] = np.where(
    adaptive["total_attempted_steps"] > 0,
    adaptive["rejected_steps"] / adaptive["total_attempted_steps"],
    np.nan,
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    adaptive["setting_value"],
    adaptive["reject_ratio"],
    marker="o",
    linewidth=1.8,
)

ax.set_xscale("log")
ax.set_xlabel("Tolerance")
ax.set_ylabel("Rejected-step fraction")
ax.set_title("RKF45 rejection rate vs tolerance")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "rkf45_reject_ratio_vs_tol.png", dpi=200)
plt.show()

# %%
# RKF45 adaptive step-size history for one representative run
fig, ax = plt.subplots(figsize=(8, 4))
plot_step_sizes(rkf45, ax=ax)
ax.set_title(f"RKF45 accepted step sizes (tol={representative_tol:.0e})")
fig.tight_layout()
fig.savefig(FIG_DIR / "rkf45_step_sizes_representative.png", dpi=200)
plt.show()