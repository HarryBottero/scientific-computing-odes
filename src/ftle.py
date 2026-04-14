from __future__ import annotations

from typing import Callable
import time

import numpy as np

from src.benchmarks import resample_result
from src.solvers import rk4_integrate, rkf45_integrate

RHS = Callable[..., np.ndarray]


def _integrate(rhs: RHS, t_span, y0, solver: str, solver_kwargs: dict, rhs_kwargs: dict):
    if solver == "rk4":
        return rk4_integrate(rhs, t_span, y0, **solver_kwargs, **rhs_kwargs)
    if solver == "rkf45":
        return rkf45_integrate(rhs, t_span, y0, **solver_kwargs, **rhs_kwargs)
    raise ValueError("solver must be 'rk4' or 'rkf45'.")


def nearby_separation(
    rhs: RHS,
    t_span: tuple[float, float],
    y0,
    delta0=(1e-6, 0.0),
    n_eval: int = 1000,
    solver: str = "rkf45",
    solver_kwargs: dict | None = None,
    rhs_kwargs: dict | None = None,
):
    solver_kwargs = solver_kwargs or {}
    rhs_kwargs = rhs_kwargs or {}
    y0 = np.asarray(y0, dtype=float)
    delta0 = np.asarray(delta0, dtype=float)

    y1 = y0
    y2 = y0 + delta0

    res1 = _integrate(rhs, t_span, y1, solver, solver_kwargs, rhs_kwargs)
    res2 = _integrate(rhs, t_span, y2, solver, solver_kwargs, rhs_kwargs)

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    y1_eval = resample_result(res1, t_eval)
    y2_eval = resample_result(res2, t_eval)

    sep = np.linalg.norm(y2_eval - y1_eval, axis=1)
    return t_eval, sep


def estimate_ftle(
    rhs: RHS,
    t_span: tuple[float, float],
    y0,
    delta0=(1e-6, 0.0),
    solver: str = "rkf45",
    solver_kwargs: dict | None = None,
    rhs_kwargs: dict | None = None,
) -> float:
    t_eval, sep = nearby_separation(
        rhs,
        t_span,
        y0,
        delta0=delta0,
        n_eval=400,
        solver=solver,
        solver_kwargs=solver_kwargs,
        rhs_kwargs=rhs_kwargs,
    )
    d0 = max(float(np.linalg.norm(delta0)), 1e-16)
    dT = max(float(sep[-1]), 1e-16)
    T = t_span[1] - t_span[0]
    return float(np.log(dT / d0) / T)


def ftle_grid(
    rhs: RHS,
    t_span: tuple[float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int = 40,
    ny: int = 40,
    delta0=(1e-6, 0.0),
    solver: str = "rkf45",
    solver_kwargs: dict | None = None,
    rhs_kwargs: dict | None = None,
    progress: bool = False,
):
    solver_kwargs = solver_kwargs or {}
    rhs_kwargs = rhs_kwargs or {}

    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    field = np.empty((ny, nx))

    start = time.perf_counter()

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            try:
                field[j, i] = estimate_ftle(
                    rhs,
                    t_span,
                    y0=np.array([x, y]),
                    delta0=delta0,
                    solver=solver,
                    solver_kwargs=solver_kwargs,
                    rhs_kwargs=rhs_kwargs,
                )
            except RuntimeError:
                field[j, i] = np.nan

                if progress and ((j + 1) % 5 == 0 or (j + 1) == ny):
            elapsed = time.perf_counter() - start
            rows_done = j + 1
            eta = elapsed * (ny - rows_done) / rows_done

            mins, secs = divmod(elapsed, 60)
            eta_mins, eta_secs = divmod(eta, 60)
            timestamp = time.strftime("%H:%M:%S")

            print(
                f"[{timestamp}] "
                f"FTLE row {rows_done}/{ny} "
                f"({100 * rows_done / ny:5.1f}%)  "
                f"elapsed {int(mins):02d}:{secs:04.1f}  "
                f"ETA {int(eta_mins):02d}:{eta_secs:04.1f}",
                flush=True,
            )

    return xs, ys, field