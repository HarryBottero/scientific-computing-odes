from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.solvers import euler_integrate, rk4_integrate, rkf45_integrate
from src.utils import BenchmarkRecord, SolverResult

RHS = Callable[..., np.ndarray]



def reference_solution(
    rhs: RHS,
    t_span: tuple[float, float],
    y0,
    t_eval: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    **rhs_kwargs,
) -> np.ndarray:
    sol = solve_ivp(
        lambda t, y: rhs(t, y, **rhs_kwargs),
        t_span,
        np.asarray(y0, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solver failed: {sol.message}")
    return sol.y.T



def resample_result(result: SolverResult, t_eval: np.ndarray) -> np.ndarray:
    y_interp = np.empty((len(t_eval), result.y.shape[1]))
    for j in range(result.y.shape[1]):
        y_interp[:, j] = np.interp(t_eval, result.t, result.y[:, j])
    return y_interp



def error_metrics(result: SolverResult, ref_t: np.ndarray, ref_y: np.ndarray) -> dict[str, float]:
    y_interp = resample_result(result, ref_t)
    diffs = y_interp - ref_y
    norms = np.linalg.norm(diffs, axis=1)
    return {
        "final_error": float(norms[-1]),
        "max_path_error": float(np.max(norms)),
        "rms_path_error": float(np.sqrt(np.mean(norms**2))),
    }



def invariant_drift(result: SolverResult, invariant_fn: Callable[..., np.ndarray], **inv_kwargs) -> float:
    vals = invariant_fn(result.y[:, 0], result.y[:, 1], **inv_kwargs)
    return float(np.max(np.abs(vals - vals[0])))



def run_benchmark_suite(
    rhs: RHS,
    t_span: tuple[float, float],
    y0,
    fixed_dts: list[float],
    rkf_tols: list[float],
    reference_points: int = 2000,
    invariant_fn: Callable[..., np.ndarray] | None = None,
    solver_kwargs: dict | None = None,
    rhs_kwargs: dict | None = None,
    invariant_kwargs: dict | None = None,
) -> pd.DataFrame:
    solver_kwargs = solver_kwargs or {}
    rhs_kwargs = rhs_kwargs or {}
    invariant_kwargs = invariant_kwargs or {}

    ref_t = np.linspace(t_span[0], t_span[1], reference_points)
    ref_y = reference_solution(rhs, t_span, y0, ref_t, **rhs_kwargs)

    records: list[BenchmarkRecord] = []

    for dt in fixed_dts:
        for method_name, solver in [("Euler", euler_integrate), ("RK4", rk4_integrate)]:
            result = solver(rhs, t_span, y0, dt=dt, **rhs_kwargs)
            metrics = error_metrics(result, ref_t, ref_y)
            drift = invariant_drift(result, invariant_fn, **invariant_kwargs) if invariant_fn else None
            records.append(BenchmarkRecord(
                method=method_name,
                setting_name="dt",
                setting_value=float(dt),
                runtime_sec=result.runtime_sec,
                nfev=result.nfev,
                accepted_steps=result.accepted_steps,
                rejected_steps=result.rejected_steps,
                invariant_drift=drift,
                **metrics,
            ))

    for tol in rkf_tols:
        result = rkf45_integrate(rhs, t_span, y0, tol=tol, **solver_kwargs, **rhs_kwargs)
        metrics = error_metrics(result, ref_t, ref_y)
        drift = invariant_drift(result, invariant_fn, **invariant_kwargs) if invariant_fn else None
        records.append(BenchmarkRecord(
            method="RKF45",
            setting_name="tol",
            setting_value=float(tol),
            runtime_sec=result.runtime_sec,
            nfev=result.nfev,
            accepted_steps=result.accepted_steps,
            rejected_steps=result.rejected_steps,
            invariant_drift=drift,
            **metrics,
        ))

    df = pd.DataFrame([record.__dict__ for record in records])
    return df.sort_values(["method", "setting_value"], ascending=[True, False]).reset_index(drop=True)
