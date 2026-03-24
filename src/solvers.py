# imports
from __future__ import annotations
import numpy as np
from typing import Callable
from src.utils import SolverResult, Timer, as_1d_state, clip_step

# Type alias for the right-hand side function of an ODE
RHS = Callable[..., np.ndarray]


# Euler's method for integrating ODEs
def euler_integrate(rhs: RHS, t_span: tuple[float, float], y0, dt: float, **rhs_kwargs) -> SolverResult:
    y0 = as_1d_state(y0)                            # Ensure y0 is a 1D array
    t0, tf = map(float, t_span)                     # Unpack and convert t_span to floats
    if dt <= 0:                                     # safety bounds
        raise ValueError("dt must be positive.")

    n_steps = int(np.ceil((tf - t0) / dt))          # Calculate the number of steps needed
    t = np.empty(n_steps + 1)                       # Preallocate time array for efficiency
    y = np.empty((n_steps + 1, y0.size))            # Preallocate solution array for efficiency
    h_used = np.empty(n_steps)                      # Preallocate array to store actual step sizes used

    # Initialize the first time and state
    t[0] = t0
    y[0] = y0
    nfev = 0

    with Timer() as timer:                          # Start timing the integration
        for i in range(n_steps):
            h = min(dt, tf - t[i])
            f = rhs(t[i], y[i], **rhs_kwargs)
            nfev += 1
            y[i + 1] = y[i] + h * f
            t[i + 1] = t[i] + h
            h_used[i] = h

    return SolverResult(                            # Return the results in a structured format
        method="Euler",
        t=t,
        y=y,
        h=h_used,
        nfev=nfev,
        accepted_steps=n_steps,
        rejected_steps=0,
        runtime_sec=timer.elapsed,
    )


# Runge-Kutta 4th order method for a single step, just the step function and copied from wikipedia
def rk4_step(rhs: RHS, t: float, y: np.ndarray, h: float, **rhs_kwargs) -> tuple[np.ndarray, int]:
    k1 = rhs(t, y, **rhs_kwargs)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, **rhs_kwargs)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2, **rhs_kwargs)
    k4 = rhs(t + h, y + h * k3, **rhs_kwargs)
    y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next, 4 # Return the next state and the number of function evaluations (4 for RK4)

# RK4 integration using the rk4_step function, differs from euler_integrate in that it calls rk4_step for each step and accumulates the function evaluation count accordingly
def rk4_integrate(rhs: RHS, t_span: tuple[float, float], y0, dt: float, **rhs_kwargs) -> SolverResult:
    y0 = as_1d_state(y0)
    t0, tf = map(float, t_span)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.empty(n_steps + 1)
    y = np.empty((n_steps + 1, y0.size))
    h_used = np.empty(n_steps)

    t[0] = t0
    y[0] = y0
    nfev = 0

    with Timer() as timer:
        for i in range(n_steps):
            h = min(dt, tf - t[i])
            y_next, cost = rk4_step(rhs, t[i], y[i], h, **rhs_kwargs)
            nfev += cost
            y[i + 1] = y_next
            t[i + 1] = t[i] + h
            h_used[i] = h

    return SolverResult(
        method="RK4",
        t=t,
        y=y,
        h=h_used,
        nfev=nfev,
        accepted_steps=n_steps,
        rejected_steps=0,
        runtime_sec=timer.elapsed,
    )


# Final solver, the adaptive Runge-Kutta-Fehlberg 4(5) method, which is more complex due to the adaptive step size control and error estimation. It implements the RKF45 algorithm, which computes both a 4th order and a 5th order estimate of the solution at each step to control the error and adjust the step size accordingly.
def rkf45_integrate(
    rhs: RHS,
    t_span: tuple[float, float],
    y0,
    h0: float = 1e-2,
    hmin: float = 1e-6,
    hmax: float = 1e-1,
    tol: float = 1e-8,
    safety: float = 0.9,
    max_steps: int = 100000,
    **rhs_kwargs,
) -> SolverResult:
    y = as_1d_state(y0).copy()
    t0, tf = map(float, t_span)
    t = t0
    h = clip_step(h0, hmin, hmax)

    ts = [t]
    ys = [y.copy()]
    hs = []
    nfev = 0
    accepted = 0
    rejected = 0

    with Timer() as timer:
        while t < tf and (accepted + rejected) < max_steps:
            h = min(h, tf - t)
            h = clip_step(h, hmin, hmax)
            if h <= 0:
                break

            k1 = h * rhs(t, y, **rhs_kwargs)
            k2 = h * rhs(t + 0.25 * h, y + 0.25 * k1, **rhs_kwargs)
            k3 = h * rhs(t + 3.0 * h / 8.0, y + 3.0 * k1 / 32.0 + 9.0 * k2 / 32.0, **rhs_kwargs)
            k4 = h * rhs(
                t + 12.0 * h / 13.0,
                y + 1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0,
                **rhs_kwargs,
            )
            k5 = h * rhs(
                t + h,
                y + 439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0,
                **rhs_kwargs,
            )
            k6 = h * rhs(
                t + 0.5 * h,
                y - 8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0,
                **rhs_kwargs,
            )
            nfev += 6

            y4 = y + 25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - 0.2 * k5
            y5 = y + 16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0

            err = np.linalg.norm(y5 - y4, ord=np.inf)
            scale = max(1.0, np.linalg.norm(y, ord=np.inf))
            err_ratio = err / (tol * scale)

            if err_ratio <= 1.0:
                # accept the step using the current h
                t = t + h
                y = y5
                ts.append(t)
                ys.append(y.copy())
                hs.append(h)
                accepted += 1

                if err_ratio == 0.0:
                    factor = 2.0
                else:
                    factor = float(safety * err_ratio ** (-0.2))
                growth = float(np.clip(factor, 0.2, 2.0))
                h = clip_step(h * growth, hmin, hmax)
            else:
                rejected += 1
                factor = float(safety * err_ratio ** (-0.25))
                shrink = float(np.clip(factor, 0.1, 0.5))
                h = clip_step(h * shrink, hmin, hmax)
                if h <= hmin and err_ratio > 1.0:
                    raise RuntimeError("RKF45 could not satisfy tolerance before hitting hmin.")

        if (accepted + rejected) >= max_steps and t < tf:
            raise RuntimeError("RKF45 reached max_steps before finishing integration.")

    return SolverResult(
        method="RKF45",
        t=np.asarray(ts),
        y=np.asarray(ys),
        h=np.asarray(hs) if hs else np.array([], dtype=float),
        nfev=nfev,
        accepted_steps=accepted,
        rejected_steps=rejected,
        runtime_sec=timer.elapsed,
    )