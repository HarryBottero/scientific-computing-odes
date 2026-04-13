# %%
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.flows import blinking_vortex_rhs
from src.ftle import ftle_grid
from src.plotting import plot_ftle_field
import matplotlib.pyplot as plt
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%
xs, ys, field = ftle_grid(
    blinking_vortex_rhs,
    t_span=(0.0, 4.0),
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
    nx=40,
    ny=40,
    delta0=(1e-6, 0.0),
    solver="rkf45",
    solver_kwargs={"tol": 1e-7, "h0": 0.02, "hmin": 1e-5, "hmax": 0.05},
)

fig, ax = plt.subplots(figsize=(7, 6))
plot_ftle_field(xs, ys, field, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "blinking_vortex_ftle.png", dpi=200)
plt.show()
# %%
