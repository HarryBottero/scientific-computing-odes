import numpy as np


def steady_dipole_velocity(x: float, y: float, alpha: float, beta: float) -> np.ndarray:
    r1_sq = x**2 + (y - beta)**2
    r2_sq = x**2 + (y + beta)**2

    u = -alpha * ((y - beta) / max(r1_sq, 1e-12) - (y + beta) / max(r2_sq, 1e-12))
    v = alpha * (x / max(r1_sq, 1e-12) - x / max(r2_sq, 1e-12))
    return np.array([u, v], dtype=float)


def blinking_vortex_velocity(
    t: float, x: float, y: float, alpha: float, beta0: float, tau: float
) -> np.ndarray:
    beta = beta0 if (t % (2 * tau)) < tau else -beta0
    return steady_dipole_velocity(x, y, alpha, beta)


def rhs_steady_dipole(t: float, state: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x, y = state
    return steady_dipole_velocity(x, y, alpha, beta)


def rhs_blinking_vortex(
    t: float, state: np.ndarray, alpha: float, beta0: float, tau: float
) -> np.ndarray:
    x, y = state
    return blinking_vortex_velocity(t, x, y, alpha, beta0, tau)