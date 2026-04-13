from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.utils import SolverResult



def plot_stream_field(
    velocity_fn,
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
    n=300,
    t: float | None = None,
    ax=None,
    density: float = 1.5,
    cmap: str = "viridis",
    log_colour: bool = False,
    **velocity_kwargs,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    if t is None:
        U, V = velocity_fn(X, Y, **velocity_kwargs)
    else:
        U, V = velocity_fn(X, Y, t=t, **velocity_kwargs)
    speed = np.sqrt(U * U + V * V)

    kwargs = dict(color=speed, cmap=cmap, density=density)
    if log_colour:
        kwargs["norm"] = LogNorm(vmin=max(np.nanmin(speed[speed > 0]), 1e-6), vmax=np.nanmax(speed))

    strm = ax.streamplot(x, y, U, V, **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax, strm



def plot_trajectory(result: SolverResult, ax=None, label: str | None = None, **plot_kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result.y[:, 0], result.y[:, 1], label=label or result.method, **plot_kwargs)
    ax.plot(result.y[0, 0], result.y[0, 1], "o", ms=4)
    ax.plot(result.y[-1, 0], result.y[-1, 1], "s", ms=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax



def plot_step_sizes(result: SolverResult, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if result.h is None or len(result.h) == 0:
        raise ValueError("This result does not store step sizes.")
    ax.plot(result.t[1:], result.h)
    ax.set_xlabel("t")
    ax.set_ylabel("accepted step size")
    ax.set_title(f"Adaptive step sizes: {result.method}")
    return ax



def plot_convergence(df, x_col="setting_value", y_col="max_path_error", hue_col="method", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for method, group in df.groupby(hue_col):
        ax.loglog(group[x_col], group[y_col], marker="o", label=method)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax



def plot_ftle_field(xs, ys, field, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        field,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("FTLE field")
    plt.colorbar(im, ax=ax, label="FTLE")
    return ax
