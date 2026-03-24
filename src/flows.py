# This file contains the definitions of the velocity fields and their corresponding right-hand sides for the ODEs that govern the particle trajectories in the flows. It also includes the streamfunction for the steady dipole flow and a function to compute the time-varying beta for the blinking vortex flow.

# Import our libraries
from __future__ import annotations
import numpy as np

# Set default parameters for the flows
Default_Alpha = 1.0 # The strength of the vortices in the steady dipole flow
Default_Beta = 0.4 # The distance of the vortices from the origin in the steady dipole flow
Default_Eps = 1e-12 # A lil value to prevent division by 0

# We'll start with a helper function to compute the squared radius from the vortices, ensuring we don't divide by zero.
def _safe_radius_squared(dx: np.ndarray | float, y: np.ndarray | float, eps: float = Default_Eps):
    return np.maximum(dx * dx + y * y, eps)


# Now we can define the velocity field for the steady dipole flow. This is given by the equations:
# u = -(alpha*y)/((x-beta)^2+y^2) - (alpha*y)/((x+beta)^2+y^2)
# v = alpha*(x-beta)/((x-beta)^2+y^2) + alpha*(x+beta)/((x+beta)^2+y^2)
def steady_dipole_velocity(
    x: np.ndarray | float,
    y: np.ndarray | float,
    alpha: float = Default_Alpha,
    beta: float = Default_Beta,
    eps: float = Default_Eps,
):
    r1 = _safe_radius_squared(x - beta, y, eps=eps)
    r2 = _safe_radius_squared(x + beta, y, eps=eps)

    u = -(alpha * y) / r1 - (alpha * y) / r2
    v = alpha * (x - beta) / r1 + alpha * (x + beta) / r2
    return u, v

# Next, we can define the right-hand side of the ODEs for the steady dipole flow. This will be used in our ODE solver to compute the trajectories of particles in the flow.
def steady_dipole_rhs(
    t: float,
    state: np.ndarray,
    alpha: float = Default_Alpha,
    beta: float = Default_Beta,
    eps: float = Default_Eps,
) -> np.ndarray:
    x, y = state
    u, v = steady_dipole_velocity(x, y, alpha=alpha, beta=beta, eps=eps)
    return np.array([u, v], dtype=float)

# We can also define the streamfunction for the steady dipole flow, which is given by:
def steady_dipole_streamfunction(
    x: np.ndarray | float,
    y: np.ndarray | float,
    alpha: float = Default_Alpha,
    beta: float = Default_Beta,
    eps: float = Default_Eps,
):
    r1 = _safe_radius_squared(x - beta, y, eps=eps)
    r2 = _safe_radius_squared(x + beta, y, eps=eps)
    return 0.5 * alpha * np.log(r1) + 0.5 * alpha * np.log(r2)

# Finally, we can define a function to compute the time-varying beta for the blinking vortex flow. This will allow us to simulate the blinking vortex flow by varying the position of the vortices over time.
def blinking_beta(t: float, beta0: float = Default_Beta, period: float = 1.0) -> float:
    return beta0 * np.sign(np.cos(2.0 * np.pi * t / period))


# Next is the velocity field for the blinking vortex flow, which is similar to the steady dipole flow but with a time-varying beta. We can define this as follows:
def blinking_vortex_velocity(
    x: np.ndarray | float,
    y: np.ndarray | float,
    t: float,
    alpha: float = Default_Alpha,
    beta0: float = Default_Beta,
    period: float = 1.0,
    eps: float = Default_Eps,
):
    beta = blinking_beta(t, beta0=beta0, period=period)
    r1 = _safe_radius_squared(x - beta, y, eps=eps)
    r2 = _safe_radius_squared(x + beta, y, eps=eps)

    u = -(alpha * y) / r1 - (alpha * y) / r2
    v = alpha * (x - beta) / r1 + alpha * (x + beta) / r2
    return u, v

# and the corresponding right-hand side for the ODEs:
def blinking_vortex_rhs(
    t: float,
    state: np.ndarray,
    alpha: float = Default_Alpha,
    beta0: float = Default_Beta,
    period: float = 1.0,
    eps: float = Default_Eps,
) -> np.ndarray:
    x, y = state
    u, v = blinking_vortex_velocity(x, y, t, alpha=alpha, beta0=beta0, period=period, eps=eps)
    return np.array([u, v], dtype=float)